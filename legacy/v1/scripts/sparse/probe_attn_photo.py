#!/usr/bin/env python3
"""SPARSE S10 — the attention photograph (LORAMECH's unrun L4).

Measures, at the LAST PROMPT ROW, the attention mass placed on evidence blocks /
non-evidence blocks / everything else (prompt+sink), per head x layer, as k and N
vary. sdpa exposes no weights, so they are computed MANUALLY from FenceHooks
captures: q/k projections + rotary cos/sin -> apply_multimodal_rotary_pos_emb ->
scores = q_row . k / sqrt(d) + injected mask row -> softmax. Per-row sum==1 is
asserted (the sanity the brief requires).

Hook order matters: the PEFT adapter is loaded BEFORE FenceHooks(capture) is
installed, so q_proj/k_proj hooks sit on the LoRA-wrapped modules and capture the
TRAINED projections (the wave-1 lesson inverted: capture hooks must be post-wrap;
the mask pre-hook is wrap-agnostic).

One invocation = one arm x one N over a comma-set of k strata:
  --arm p1b    : P1b adapter, UNGATED fence (hide=[])       [the (N-k) term visible]
  --arm gated  : S8 adapter (default), oracle gate           [k-only prediction]
  --arm frozen : no adapter, ungated fence                   [untrained reference]

Usage:
  python scripts/sparse/probe_attn_photo.py --arm p1b --n 32 --k-list 2,4,8 \
      --limit 30 --output outputs/sparse/s10/p1b_N32
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gnnformer.data import (  # noqa: E402
    iter_sample_dirs_shuffled,
    load_mmred_sample,
    rooms_to_room2chars,
)
from gnnformer.fencing import FenceHooks, build_block_mask, reset_positions  # noqa: E402
from gnnformer.runtime import (  # noqa: E402
    attention_dims,
    get_layers,
    get_rope_index_fn,
    load_runtime,
    move_to_device,
)

_SPARSE = _REPO / "scripts" / "sparse"
if str(_SPARSE) not in sys.path:
    sys.path.insert(0, str(_SPARSE))
from train_sft_gated import build_fenced_messages, parse_layout  # noqa: E402

_REDUX = _REPO / "scripts" / "redux"
if str(_REDUX) not in sys.path:
    sys.path.insert(0, str(_REDUX))
from tasks import target_of  # noqa: E402

FENCED_SDPA = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
POOLS = {2: "data/mmred_images_park/seq_len_2/all_uniform",   # S10b small-N (2026-09-19)
         4: "data/mmred_images_park/seq_len_4/all_uniform",
         8: "data/mmred_images_park/seq_len_8/all_uniform",
         16: "data/mmred_longN_park/seq_len_16/all_uniform",
         32: "data/mmred_longN_park/seq_len_32/all_uniform",
         64: "data/mmred_longN_park/seq_len_64/all_uniform",
         128: "data/mmred_longN_park/seq_len_128/all_uniform"}
ADAPTERS = {"p1b": "checkpoints/sft_fenced_le16_ep10_adapter",
            "gated": "checkpoints/sft_fenced_gated_vn_adapter",
            "frozen": None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=("p1b", "gated", "frozen"), required=True)
    ap.add_argument("--adapter", type=Path, default=None,
                    help="override the arm's default adapter")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--k-list", default="2,4,8")
    ap.add_argument("--limit", type=int, default=30, help="samples per k stratum")
    ap.add_argument("--layers", default="12,16,20,24,27")
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--attn-sharpen", type=float, default=0.0,
                    help="S10b: multiply attention logits by TAU (sets module.scaling on "
                         "decoder attention modules >= --sharpen-from-layer, and scales the "
                         "photographed scores the same way). 0 = off (byte-identical anchor).")
    ap.add_argument("--sharpen-from-layer", type=int, default=0,
                    help="S10b: first decoder layer index the sharpening applies to (S0 used 12).")
    args = ap.parse_args()
    klist = [int(x) for x in args.k_list.split(",")]
    LAYERS = [int(x) for x in args.layers.split(",")]
    gated = args.arm == "gated"

    rt = load_runtime()
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    layers = get_layers(model)
    rope_fn = get_rope_index_fn(model)
    vs_id = int(model.config.vision_start_token_id)
    dims = attention_dims(model)
    # S10b: static sharpening, set PRE-wrap on the raw attention modules (the S0 hook, verbatim)
    attn_mods = [ly.self_attn for ly in layers]
    base_scaling = float(attn_mods[0].scaling)
    if args.attn_sharpen > 0:
        for i_, m_ in enumerate(attn_mods):
            if i_ >= args.sharpen_from_layer:
                m_.scaling = base_scaling * args.attn_sharpen
        print(f"[sharpen] tau={args.attn_sharpen} on layers >= {args.sharpen_from_layer} "
              f"(base_scaling={base_scaling:.6g})", flush=True)
    adapter = args.adapter or ADAPTERS[args.arm]
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
        model.eval()
        print(f"adapter loaded (frozen): {adapter}", flush=True)
    # capture hooks POST-wrap so q/k reflect the LoRA-adapted projections
    hooks = FenceHooks(layers, capture_layers=LAYERS).install()

    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        apply_multimodal_rotary_pos_emb,
        repeat_kv,
    )

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    rows_f = open(out / "mass.csv", "w", newline="")
    w = csv.writer(rows_f)
    w.writerow(["sample", "N", "k", "layer", "head", "evid_mass", "nonevid_mass",
                "other_mass"])
    sep_f = open(out / "block_mass.csv", "w", newline="")
    ws = csv.writer(sep_f)
    ws.writerow(["sample", "N", "k", "layer", "head", "block", "is_evid", "mass"])

    need = {k: args.limit for k in klist}
    n_done = 0
    t0 = time.time()
    for sd in iter_sample_dirs_shuffled(Path(POOLS[args.n]), args.seed):
        if all(v <= 0 for v in need.values()):
            break
        try:
            _sid, frames, q0, states, a0 = load_mmred_sample(sd)
            gold = int(str(a0).strip())
            if need.get(gold, 0) <= 0:
                continue
            tr = target_of(Path(str(sd)), q0)
            char, room = tr
            evid = {t for t, st in enumerate(states)
                    if char in rooms_to_room2chars(st.get("rooms", {})).get(room, [])}
            assert len(evid) == gold
            frames = [f.resize((args.resize, args.resize)) for f in frames]
            inp = move_to_device(dict(processor.apply_chat_template(
                build_fenced_messages(frames, q0), add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors="pt")), rt.device)
            ids = inp["input_ids"][0].tolist()
            parsed = parse_layout(ids, tok, q0, len(frames), vs_id)
            if parsed is None:
                continue
            blocks, fin_start = parsed
            hide = ([p for t, (a, b) in enumerate(blocks) if t not in evid
                     for p in range(int(a), int(b))] if gated else [])
            mask = build_block_mask(len(ids), blocks, hide_cols=hide)
            with torch.inference_mode():
                pos, _ = rope_fn(inp["input_ids"],
                                 image_grid_thw=inp.get("image_grid_thw"),
                                 attention_mask=inp.get("attention_mask"))
            pos = reset_positions(pos, blocks, fin_start)
            inp.pop("attention_mask", None)
            hooks.set_mask(mask, rt.device)
            try:
                with torch.inference_mode(), sdpa_kernel(FENCED_SDPA):
                    model(**inp, position_ids=pos.to(rt.device), use_cache=False)
            finally:
                hooks.clear_mask()
            seq = len(ids)
            row = seq - 1
            mask_row = mask[row].to(torch.float32)
            hd, nH, nKV = dims["head_dim"], dims["n_heads"], dims["n_kv"]
            for L in LAYERS:
                q = hooks.qkv[L]["q_proj"].view(1, seq, nH, hd).transpose(1, 2)
                kk = hooks.qkv[L]["k_proj"].view(1, seq, nKV, hd).transpose(1, 2)
                qr, kr = apply_multimodal_rotary_pos_emb(
                    q.float(), kk.float(), hooks.cos.float(), hooks.sin.float(),
                    dims["mrope_section"])
                kr = repeat_kv(kr, nH // nKV)[0]            # [H, T, D]
                qrow = qr[0][:, row]                        # [H, D]
                sc = torch.einsum("hd,htd->ht", qrow, kr) / (hd ** 0.5)
                if args.attn_sharpen > 0 and L >= args.sharpen_from_layer:
                    sc = sc * args.attn_sharpen           # S10b: photograph under the same tau
                sc = sc + mask_row.to(sc.device)
                wgt = torch.softmax(sc, -1)                 # [H, T]
                ssum = float(wgt.sum(-1).mean())
                assert abs(ssum - 1.0) < 1e-3, f"softmax rows sum to {ssum}"
                for h in range(nH):
                    ev = ne = 0.0
                    for t, (a, b) in enumerate(blocks):
                        m_ = float(wgt[h, a:b].sum())
                        ws.writerow([sd.name, args.n, gold, L, h, t, int(t in evid),
                                     f"{m_:.6f}"])
                        if t in evid:
                            ev += m_
                        else:
                            ne += m_
                    w.writerow([sd.name, args.n, gold, L, h, f"{ev:.6f}",
                                f"{ne:.6f}", f"{1.0 - ev - ne:.6f}"])
        except Exception as e:
            print(f"  [skip] {sd.name}: {e}", flush=True)
            continue
        need[gold] -= 1
        n_done += 1
        rows_f.flush(); sep_f.flush()
        if n_done % 10 == 0:
            print(f"  {n_done} samples {time.time()-t0:.0f}s (need {need})", flush=True)
    hooks.remove()
    rows_f.close(); sep_f.close()

    # per-cell head-mean summary
    agg = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    for r in csv.DictReader(open(out / "mass.csv")):
        a = agg[(int(r["k"]), int(r["layer"]))]
        a[0] += float(r["evid_mass"]); a[1] += float(r["nonevid_mass"])
        a[2] += float(r["other_mass"]); a[3] += 1
    lines = [f"=== ATTN PHOTO arm={args.arm} N={args.n} adapter={adapter} "
             f"(head-mean masses; n_samples={n_done}) ==="]
    for (k, L) in sorted(agg):
        a = agg[(k, L)]
        lines.append(f"  k={k:<3} L{L:<3} evid {a[0]/a[3]:.4f}  nonevid {a[1]/a[3]:.4f}"
                     f"  other {a[2]/a[3]:.4f}")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

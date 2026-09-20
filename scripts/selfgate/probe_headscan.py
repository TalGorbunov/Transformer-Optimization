#!/usr/bin/env python3
"""SELFGATE G0a — per-head evidence/non-evidence separation scan (copy-extended from
scripts/sparse/probe_attn_photo.py; the original stays the S10 anchor).

Deltas vs the original: (1) --nfree-prompt (the S9b adapter's eval contract);
(2) --ungated forces hide=[] for ANY adapter (G0a scans the UNGATED forward);
(3) the report computes per-(layer,head) AUC of block mass vs is_evid — the number
that found L24h20 (S10: AUC ≥0.9927 in P1b) — plus the best-head table per layer.

One invocation = one adapter x one N over a comma-set of k strata:
  python scripts/selfgate/probe_headscan.py --adapter checkpoints/sft_fenced_gated_vn_nfree_adapter \
      --nfree-prompt --n 32 --k-list 2,4,8 --limit 30 --output outputs/selfgate/g0a/s9b_N32
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

_QGATE = _REPO / "scripts" / "qgate"
if str(_QGATE) not in sys.path:
    sys.path.insert(0, str(_QGATE))
from qgate_common import build_task_messages_layout as build_task_messages  # noqa: E402

_REDUX = _REPO / "scripts" / "redux"
if str(_REDUX) not in sys.path:
    sys.path.insert(0, str(_REDUX))
from tasks import target_of  # noqa: E402

FENCED_SDPA = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
POOLS = {8: "data/mmred_images_park/seq_len_8/all_uniform",
         16: "data/mmred_longN_park/seq_len_16/all_uniform",
         32: "data/mmred_longN_park/seq_len_32/all_uniform",
         64: "data/mmred_longN_park/seq_len_64/all_uniform",
         128: "data/mmred_longN_park/seq_len_128/all_uniform"}


def auc(pos, neg):
    """Rank AUC of pos vs neg score lists (ties = 0.5)."""
    if not pos or not neg:
        return float("nan")
    allv = sorted(pos + neg)
    rank = {}
    i = 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j] == allv[i]:
            j += 1
        r = (i + j - 1) / 2 + 1
        rank[allv[i]] = r
        i = j
    rp = sum(rank[v] for v in pos)
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", default=None, help="PEFT adapter dir (None = frozen)")
    ap.add_argument("--task", default="count", choices=("count", "exists", "majority"),
                    help="G0c: question type for the layout (count = byte-identical "
                         "original path); AUC labels stay the same occupancy set")
    ap.add_argument("--layout", default="replica",
                    choices=("replica", "qfirst-once", "qlast-once", "replica-neutral"),
                    help="QGATE A1: query-side layout (replica = byte-identical path)")
    ap.add_argument("--nfree-prompt", action="store_true")
    ap.add_argument("--ungated", action="store_true", default=True,
                    help="G0a default: hide=[] regardless of adapter")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--k-list", default="2,4,8")
    ap.add_argument("--limit", type=int, default=30, help="samples per k stratum")
    ap.add_argument("--layers", default="12,16,20,24,27")
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    klist = [int(x) for x in args.k_list.split(",")]
    LAYERS = [int(x) for x in args.layers.split(",")]

    rt = load_runtime()
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    layers = get_layers(model)
    rope_fn = get_rope_index_fn(model)
    vs_id = int(model.config.vision_start_token_id)
    dims = attention_dims(model)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
        model.eval()
        print(f"adapter loaded (frozen): {args.adapter}", flush=True)
    # capture hooks POST-wrap (S10 lesson): q/k must reflect the trained projections
    hooks = FenceHooks(layers, capture_layers=LAYERS).install()

    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        apply_multimodal_rotary_pos_emb,
        repeat_kv,
    )

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
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
                build_task_messages(frames, args.task, char, room, q0,
                                    nfree=args.nfree_prompt, layout=args.layout),
                add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors="pt")), rt.device)
            ids = inp["input_ids"][0].tolist()
            parsed = parse_layout(ids, tok, q0, len(frames), vs_id)
            if parsed is None:
                continue
            blocks, fin_start = parsed
            mask = build_block_mask(len(ids), blocks, hide_cols=[])
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
                kr = repeat_kv(kr, nH // nKV)[0]
                qrow = qr[0][:, row]
                sc = torch.einsum("hd,htd->ht", qrow, kr) / (hd ** 0.5)
                sc = sc + mask_row.to(sc.device)
                wgt = torch.softmax(sc, -1)
                ssum = float(wgt.sum(-1).mean())
                assert abs(ssum - 1.0) < 1e-3, f"softmax rows sum to {ssum}"
                for h in range(nH):
                    for t, (a, b) in enumerate(blocks):
                        ws.writerow([sd.name, args.n, gold, L, h, t, int(t in evid),
                                     f"{float(wgt[h, a:b].sum()):.6f}"])
        except Exception as e:
            print(f"  [skip] {sd.name}: {e}", flush=True)
            continue
        need[gold] -= 1
        n_done += 1
        sep_f.flush()
        if n_done % 10 == 0:
            print(f"  {n_done} samples {time.time()-t0:.0f}s (need {need})", flush=True)
    hooks.remove()
    sep_f.close()

    # per-(layer,head) AUC of mass vs is_evid — the L24h20 readout
    by_lh = defaultdict(lambda: ([], []))
    for r in csv.DictReader(open(out / "block_mass.csv")):
        pos_neg = by_lh[(int(r["layer"]), int(r["head"]))]
        (pos_neg[0] if r["is_evid"] == "1" else pos_neg[1]).append(float(r["mass"]))
    aucs = {lh: auc(p, n) for lh, (p, n) in by_lh.items()}
    lines = [f"=== HEADSCAN adapter={args.adapter or 'frozen'} N={args.n} "
             f"task={args.task} nfree={args.nfree_prompt} n_samples={n_done} ===",
             "heads with AUC >= 0.98:"]
    for (L, h), a in sorted(aucs.items(), key=lambda kv: -kv[1]):
        if a >= 0.98:
            lines.append(f"  L{L} h{h}: AUC {a:.4f}")
    lines.append("best head per layer:")
    for L in LAYERS:
        cand = {h: a for (l, h), a in aucs.items() if l == L}
        if cand:
            h = max(cand, key=cand.get)
            lines.append(f"  L{L}: h{h} AUC {cand[h]:.4f}")
    (out / "auc.json").write_text(json.dumps(
        {f"L{L}_h{h}": round(a, 6) for (L, h), a in sorted(aucs.items())}, indent=1))
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

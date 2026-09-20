#!/usr/bin/env python3
"""QGATE B-capture — question-blind block states + text-only question vectors.

Per dir (qlast layout, fence+posreset, UNGATED): ONE forward -> per-block span-mean
hidden states at --layers (blocks are question-blind by fence+causality: they precede
the tail and cannot see each other). Per (C,R) pair x task phrasing: ONE text-only
forward of the bare question sentence -> last-token state (the cacheable q_vec).
Pairs = the dir's own target + --alt-pairs ALT (char, room) combos (mixed zero/
nonzero evidence, seeded); evidence bits derived from states per pair.

Output: capture.npz with block_states [n_blocks_total, L, H], block_index (dir, t),
qvec table [(pair,task) -> vec], labels [(dir, pair) -> bits], meta.json.

Usage:
  python scripts/qgate/capture_qb.py --adapter <qlast-gated adapter> \
      --data_root <roots> --limit 150 --output outputs/qgate/bcap
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

_REPO = Path(__file__).resolve().parents[2]
for p in (str(_REPO), str(_REPO / "scripts" / "sparse"), str(_REPO / "scripts" / "qgate"),
          str(_REPO / "scripts" / "redux")):
    if p not in sys.path:
        sys.path.insert(0, p)

from gnnformer.constants import ROOMS  # noqa: E402
from gnnformer.data import (  # noqa: E402
    iter_sample_dirs_shuffled,
    load_mmred_sample,
    read_dirs_file,
    rooms_to_room2chars,
)
from gnnformer.fencing import FenceHooks, build_block_mask, reset_positions  # noqa: E402
from gnnformer.runtime import get_layers, get_rope_index_fn, load_runtime, move_to_device  # noqa: E402
from qgate_common import build_task_messages_layout  # noqa: E402
from train_sft_gated import parse_layout  # noqa: E402
from tasks import evidence_count, replica_text, target_of  # noqa: E402

FENCED_SDPA = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
TASKS = ("count", "exists", "majority")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data_root",
                    default="data/mmred_images_park/seq_len_8/all_uniform,"
                            "data/mmred_longN_park/seq_len_16/all_uniform")
    ap.add_argument("--limit", type=int, default=150, help="dirs per root")
    ap.add_argument("--exclude-dirs-file", action="append", default=[])
    ap.add_argument("--alt-pairs", type=int, default=3)
    ap.add_argument("--layers", default="12,20")
    ap.add_argument("--layout", default="qlast-once",
                    choices=("qlast-once", "replica-neutral"),
                    help="E1: replica-neutral captures the NEUTRAL-SLOT states "
                         "(last token of each block's generic filler) as a 4th "
                         "pooling channel")
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    LAYERS = [int(x) for x in args.layers.split(",")]

    rt = load_runtime()
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    layers = get_layers(model)
    rope_fn = get_rope_index_fn(model)
    vs_id = int(model.config.vision_start_token_id)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
        model.eval()
        print(f"adapter loaded (frozen): {args.adapter}", flush=True)
    hooks = FenceHooks(layers).install()

    excluded = set()
    for f in args.exclude_dirs_file:
        excluded.update(str(Path(p).resolve()) for p in read_dirs_file(Path(f)))

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    rng = np.random.default_rng(args.seed)

    blk_states, blk_dir, blk_t = [], [], []
    labels = []        # rows: (dir_idx, pair_idx, t, bit)
    pair_meta = []     # (dir_idx, pair_idx, char, room, k)
    qvec_keys, qvec_rows = [], []
    dir_names = []
    n_done = 0
    t0 = time.time()

    @torch.inference_mode()
    def qvec(sentence):
        # chat-templated text-only forward (the bare-input path crashed on a
        # zero-element vision reshape — 150720; templated ids take the text route)
        ids = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": sentence}]}],
            add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt")["input_ids"].to(rt.device)
        hs = model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states
        return np.stack([hs[L][0, -1].float().cpu().numpy() for L in LAYERS])

    for root in args.data_root.split(","):
        n_root = 0
        for sd in iter_sample_dirs_shuffled(Path(root.strip()), args.seed):
            if n_root >= args.limit:
                break
            if excluded and str(Path(sd).resolve()) in excluded:
                continue
            try:
                _sid, frames, q0, states, _a0 = load_mmred_sample(sd)
                tr = target_of(Path(str(sd)), q0)
                if tr is None:
                    continue
                frames = [f.resize((args.resize, args.resize)) for f in frames]
                nf = len(frames)
                inp = move_to_device(dict(processor.apply_chat_template(
                    build_task_messages_layout(frames, "count", tr[0], tr[1], q0,
                                               layout=args.layout),
                    add_generation_prompt=True, tokenize=True, return_dict=True,
                    return_tensors="pt")), rt.device)
                ids = inp["input_ids"][0].tolist()
                parsed = parse_layout(ids, tok, q0, nf, vs_id)
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
                        hs = model(**inp, position_ids=pos.to(rt.device),
                                   output_hidden_states=True, use_cache=False
                                   ).hidden_states
                finally:
                    hooks.clear_mask()
                di = len(dir_names)
                # per-dir rows accumulate in tmp lists; committed ATOMICALLY at the
                # end of the try (the 150720 di=0/misalignment bug class)
                t_states, t_bdir, t_bt = [], [], []
                t_labels, t_pmeta, t_qk, t_qr = [], [], [], []
                for t, (a, b) in enumerate(blocks):
                    seg = [hs[L][0, int(a):int(b)].float() for L in LAYERS]
                    # replica-neutral: x[-1] IS the neutral-slot state (block ends
                    # with the filler's last token) — the E1 channel
                    t_states.append(np.stack(
                        [np.concatenate([x.mean(0).cpu().numpy(),
                                         x[-1].cpu().numpy(),
                                         x.max(0).values.cpu().numpy()])
                         for x in seg]))
                    t_bdir.append(di)
                    t_bt.append(t)
                # pairs: own + ALT (chars present in states x park rooms)
                chars = sorted({c for st in states
                               for occ in rooms_to_room2chars(st.get("rooms", {})).values()
                               for c in occ})
                pool = [(c, r) for c in chars for r in ROOMS if (c, r) != tr]
                rng.shuffle(pool)
                pairs = [tr]
                zero, nonzero = [], []
                for c, r in pool:
                    (zero if evidence_count(states, c, r) == 0 else nonzero).append((c, r))
                alt = (nonzero[:max(1, args.alt_pairs - 1)] + zero[:1])[: args.alt_pairs]
                pairs += alt
                for pi, (c, r) in enumerate(pairs):
                    k = evidence_count(states, c, r)
                    t_pmeta.append((di, pi, c, r, k))
                    for t in range(nf):
                        occ = (states[t].get("rooms", {}) or {})
                        bit = int(c in rooms_to_room2chars(occ).get(r, []))
                        t_labels.append((di, pi, t, bit))
                    for task in TASKS:
                        t_qk.append((di, pi, task))
                        t_qr.append(qvec(replica_text(task, c, r, nf)))
                # atomic commit
                blk_states.extend(t_states); blk_dir.extend(t_bdir); blk_t.extend(t_bt)
                labels.extend(t_labels); pair_meta.extend(t_pmeta)
                qvec_keys.extend(t_qk); qvec_rows.extend(t_qr)
                dir_names.append(str(sd))
            except Exception as e:
                print(f"  [skip] {sd.name}: {e}", flush=True)
                continue
            n_root += 1
            n_done += 1
            if n_done % 20 == 0:
                print(f"  {n_done} dirs {time.time()-t0:.0f}s", flush=True)
    hooks.remove()
    np.savez_compressed(
        out / "capture.npz",
        block_states=np.array(blk_states, dtype=np.float32),
        block_dir=np.array(blk_dir), block_t=np.array(blk_t),
        labels=np.array(labels), qvecs=np.array(qvec_rows, dtype=np.float32),
        qvec_keys=np.array([f"{d}|{p}|{t}" for d, p, t in qvec_keys]),
        pair_meta=np.array([f"{d}|{p}|{c}|{r}|{k}" for d, p, c, r, k in pair_meta]),
        layers=np.array(LAYERS))
    (out / "dirs.json").write_text(json.dumps(dir_names, indent=0))
    print(f"wrote {out} — {n_done} dirs, {len(blk_states)} blocks, "
          f"{len(qvec_rows)} qvecs, {len(labels)} labels", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""SPARSE S2(a) pass-1 capture — per-frame replica-slot states through the trained
fenced layout (NO gate), the input to the model gate.

For every sample: the P1 fenced layout ([frame_i + q] x N + count prompt,
build_block_mask(hide_cols=[]) + posreset — byte-identical to the trainer's
ungated forward), ONE forward with output_hidden_states, capture the hidden state
at each replica's room-word token (the verdict locus, = the probes' rep_t) at
--layer. Labels: per-frame evidence (probe_evidence, |evid| == gold asserted,
skip+count on mismatch).

Output: one .npz per run — X [sum_i N_i, hidden] fp16, y [.] {0,1}, plus
sample/frame index arrays; report.txt with counts + class balance.

Usage:
  python scripts/sparse/capture_verdicts.py --dirs-file <dirs.txt> \
      --peft-adapter <P1g> --layer 20 --output outputs/sparse/s2/cap_train
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
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gnnformer.constants import ROOMS  # noqa: E402
from gnnformer.data import (  # noqa: E402
    load_mmred_sample,
    probe_evidence,
    read_dirs_file,
)
from gnnformer.fencing import (  # noqa: E402
    FenceHooks,
    build_block_mask,
    locate_word_token,
    reset_positions,
)
from gnnformer.metrics import format_gold_histogram  # noqa: E402
from gnnformer.runtime import get_layers, get_rope_index_fn, load_runtime, move_to_device  # noqa: E402

_LORAMECH = _REPO / "scripts" / "loramech"
if str(_LORAMECH) not in sys.path:
    sys.path.insert(0, str(_LORAMECH))
from train_sft_fenced import build_fenced_messages, parse_layout  # noqa: E402

_REDUX = _REPO / "scripts" / "redux"
if str(_REDUX) not in sys.path:
    sys.path.insert(0, str(_REDUX))
from tasks import evidence_count, target_of  # noqa: E402

FENCED_SDPA = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs-file", action="append", required=True)
    ap.add_argument("--limit", type=int, default=0, help="cap per dirs-file (0 = all)")
    ap.add_argument("--layer", type=int, default=20,
                    help="hidden_states index (hs[0]=emb, hs[i]=layer i output)")
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--peft-adapter", type=Path, default=None)
    ap.add_argument("--nfree-prompt", action="store_true",
                    help="S9: capture through the N-free prompt layout (must match "
                         "the adapter's training prompt)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.nfree_prompt:
        from train_sft_gated import build_fenced_messages as _bfm_nfree

        def build_messages(frames_, q0_):
            return _bfm_nfree(frames_, q0_, nfree=True)
    else:
        def build_messages(frames_, q0_):
            return build_fenced_messages(frames_, q0_)

    rt = load_runtime(args.model) if args.model else load_runtime()
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    layers = get_layers(model)
    rope_fn = get_rope_index_fn(model)
    vs_id = int(model.config.vision_start_token_id)
    if args.peft_adapter is not None:  # PEFT wrap LAST (137800 lesson)
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.peft_adapter), is_trainable=False)
        model.eval()
        print(f"peft adapter loaded (frozen): {args.peft_adapter}", flush=True)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    hooks = FenceHooks(layers).install()

    X, y, s_idx, f_idx, meta_rows = [], [], [], [], []
    n_done = n_skip = 0
    golds = []
    t0 = time.time()
    for df in args.dirs_file:
        dirs = read_dirs_file(Path(df))
        if args.limit:
            dirs = dirs[: args.limit]
        for sd in dirs:
            try:
                _sid, frames, q0, states, a0 = load_mmred_sample(Path(sd))
                gold = int(str(a0).strip())
                tr = target_of(Path(str(sd)), q0)
                if tr is None:
                    raise ValueError("no target")
                char, room = tr
                assert evidence_count(states, char, room) == gold, "gold mismatch"
                pe = probe_evidence("steps", q0, states, gold, ROOMS)
                if pe is None or len(pe[0]) != gold:
                    raise ValueError("evidence parse/count mismatch")
                evid = pe[0]
                if args.resize > 0:
                    frames = [f.resize((args.resize, args.resize)) for f in frames]
                inp = move_to_device(dict(processor.apply_chat_template(
                    build_messages(frames, q0), add_generation_prompt=True,
                    tokenize=True, return_dict=True, return_tensors="pt")), rt.device)
                ids = inp["input_ids"][0].tolist()
                parsed = parse_layout(ids, tok, q0, len(frames), vs_id)
                if parsed is None:
                    raise ValueError("layout parse failed")
                blocks, fin_start = parsed
                loci = [locate_word_token(ids, tok, room, b) for b in blocks]
                if any(p is None for p in loci):
                    raise ValueError("room-word locus missing in a replica")
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
                                   ).hidden_states[args.layer][0]
                finally:
                    hooks.clear_mask()
                vecs = hs[torch.tensor(loci)].float().cpu().numpy().astype(np.float16)
            except Exception as e:
                print(f"  [skip] {Path(str(sd)).name}: {e}", flush=True)
                n_skip += 1
                continue
            for t in range(len(frames)):
                X.append(vecs[t])
                y.append(int(t in evid))
                s_idx.append(n_done)
                f_idx.append(t)
            meta_rows.append(f"{n_done},{sd},{len(frames)},{gold}")
            golds.append(gold)
            n_done += 1
            if n_done % 25 == 0:
                print(f"  {n_done} samples (skip {n_skip}) {time.time()-t0:.0f}s",
                      flush=True)
    hooks.remove()
    X = np.stack(X) if X else np.zeros((0, 1), dtype=np.float16)
    y = np.array(y, dtype=np.int8)
    np.savez_compressed(out / "verdicts.npz", X=X, y=y,
                        sample_idx=np.array(s_idx), frame_idx=np.array(f_idx),
                        layer=args.layer)
    (out / "samples.csv").write_text("idx,sample_dir,n_frames,gold\n"
                                     + "\n".join(meta_rows) + "\n")
    rep = (f"captured {n_done} samples / {len(y)} frames (skip {n_skip}); "
           f"pos-frac {float(y.mean()) if len(y) else 0:.4f}; layer {args.layer}\n"
           f"[gold-hist] {format_gold_histogram(golds)}\n")
    (out / "report.txt").write_text(rep)
    print(rep, "wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

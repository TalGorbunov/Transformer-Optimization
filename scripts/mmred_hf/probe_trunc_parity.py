#!/usr/bin/env python3
"""Micro-probe: why does the eval truncation path lose sample dependence?

For K samples: build the rec, then
  A) eval path: prefill_capture(trunc=L*) -> lg0 (first-token logits), carrier-state
     norms at L12 entry, top-3 tokens;
  B) training path: build_training_cache(rec, tgt_ids=gold_scan_ids, truncate=True)
     + top_hidden -> logits at the first scan position, top-3 tokens.
Prints per-sample fingerprints; identical fingerprints across samples localize the
information loss. Run on 1 GPU, minutes.

Usage:
  python scripts/mmred_hf/probe_trunc_parity.py --ckpt <carrier_layer_best.pt> \
      --dirs-file data/mmred_hf/dirsfiles/seq_len_16_test_niah.txt --n 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from gnnformer.carriers import attach_lora, load_carrier_layer_ckpt  # noqa: E402
from gnnformer.data import load_mmred_sample, read_dirs_file  # noqa: E402
from gnnformer.engine import CarrierEngine  # noqa: E402
from gnnformer.mmred_hf import build_scan_mmred, qtype_from_dirname  # noqa: E402
from gnnformer.runtime import get_layers, load_runtime  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dirs-file", required=True)
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    rt = load_runtime()
    ck = load_carrier_layer_ckpt(Path(args.ckpt))
    lora = attach_lora(get_layers(rt.model), ck.l_open, rank=ck.rank, alpha=ck.alpha,
                       device=rt.device, state=ck.lora_state)
    eng = CarrierEngine(rt, l_open=ck.l_open, e_c=ck.e_c.float().to(rt.device))
    tok = rt.processor.tokenizer

    dirs = read_dirs_file(Path(args.dirs_file))[: args.n]
    for sd in dirs:
        _sid, frames, q0, states, a0 = load_mmred_sample(sd)
        qtype = qtype_from_dirname(sd.name)
        rec = eng.prepare_sample(frames, q0, gold=0, task=qtype or "steps", resize=392,
                                 with_masks=True, with_trunc_cols=True)
        if rec is None:
            print(f"{sd.name}: prepare_sample FAILED")
            continue
        scan = build_scan_mmred(qtype, q0, states, str(a0).strip())
        tgt_ids = tok(scan, add_special_tokens=False).input_ids

        # A) eval path
        with torch.no_grad():
            caches, lg0, lo_t, hi_t, pos_k, pos_last = eng.prefill_capture(rec, ck.l_open)
        NF = len(rec["cpos"])
        a0d = rec["blocks"][0][0]
        # in truncated coords the carriers sit right after the question block
        kk = rec["keep"]
        car_k = [kk.index(c) for c in rec["cpos"]]
        cnorm = caches[ck.l_open][torch.tensor(car_k)].float().norm(dim=-1)
        top_e = [tok.decode([t]) for t in lg0.topk(3).indices.tolist()]
        print(f"\n== {sd.name} gold={a0}")
        print(f"  eval: lg0 top3={top_e} | L12 carrier-state norms "
              f"min/mean/max = {cnorm.min():.1f}/{cnorm.mean():.1f}/{cnorm.max():.1f}")
        print(f"  eval: lg0 fingerprint = {float(lg0.abs().sum()):.2f}")

        # B) training path
        with torch.no_grad():
            d = eng.build_training_cache(rec, tgt_ids, truncate=True)
            hs = eng.top_hidden(d)
            lg_t = eng.head(hs)
        # first scan-token prediction = logits at the last PROMPT row (position seq-1)
        first_pos = d["seq"] - 1
        lgt0 = lg_t[0, first_pos].float()
        top_t = [tok.decode([t]) for t in lgt0.topk(3).indices.tolist()]
        print(f"  train: first-scan-token top3={top_t} | fingerprint = "
              f"{float(lgt0.abs().sum()):.2f}")
        print(f"  gold scan head: {scan[:60]!r}")

        # C) FORCED-GOLD stepping through the eval decode loop: reveals whether the
        # step math is exact (high per-token top1==gold) or broken (early divergence)
        LO = ck.l_open
        k = pos_k.shape[2]
        hits = 0
        preds = []
        with torch.no_grad():
            from torch.nn.attention import sdpa_kernel
            from gnnformer.engine import SDPA_BACKENDS
            from gnnformer.carriers import ext_mask as _ext
            for step in range(min(len(tgt_ids), 40)):
                if step == 0:
                    lg = lg0
                else:
                    ctx = tgt_ids[:step]
                    h_app = eng.text_model.embed_tokens(
                        torch.tensor([ctx], device=eng.dev)).to(torch.bfloat16)
                    e = len(ctx)
                    inc = torch.arange(1, e + 1, device=eng.dev).view(1, 1, e)
                    pos_step = torch.cat([pos_k, pos_last.to(eng.dev) + inc], dim=2)
                    lo_s = _ext(lo_t, e).to(eng.dev).to(torch.float32).view(1, 1, k + e, k + e)
                    hi_s = _ext(hi_t, e).to(eng.dev).to(torch.float32).view(1, 1, k + e, k + e)
                    cos_, sin_ = eng.text_model.rotary_emb(h_app, pos_step)
                    pe_s = (cos_.to(h_app.dtype), sin_.to(h_app.dtype))
                    hh = h_app
                    with sdpa_kernel(SDPA_BACKENDS):
                        for li, ly in enumerate(eng.layers):
                            hin = torch.cat([caches[li].unsqueeze(0), hh], dim=1)
                            hout = ly(hin, attention_mask=(lo_s if li < LO else hi_s),
                                      position_embeddings=pe_s)[0]
                            hh = hout[:, k:]
                    hn = eng.text_model.norm(hh)
                    lg = eng.model.lm_head(
                        hn[0, -1].to(eng.model.lm_head.weight.dtype)).float()
                p = int(lg.argmax())
                preds.append(p)
                hits += int(p == tgt_ids[step])
            n = min(len(tgt_ids), 40)
        # training-path TF per-position over the same window
        tf_hits = 0
        lg_span = lg_t[0, d["seq"] - 1 : d["seq"] + n - 1]
        for i in range(n):
            tf_hits += int(int(lg_span[i].argmax()) == tgt_ids[i])
        print(f"  FORCED eval-loop: {hits}/{n} tokens match gold | train-TF same window: "
              f"{tf_hits}/{n}")
        print(f"  eval-forced decode: {tok.decode(preds)!r}")
    lora.remove()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

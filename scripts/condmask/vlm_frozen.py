#!/usr/bin/env python3
"""Fully TRAINING-FREE select-and-repack on MMRED VISION (data/mmred_vfiltered): the BABILong
protocol of eval_babilong.py on the frozen 4-bit VLM, one frame (image) = one block. Nothing
is trained anywhere (no LoRA, no readout).

  full        all N frames (the frozen baseline)
  oracle      exactly the frames showing the asked character IN the asked room (state dicts)
  judge       LZ's zero-shot per-frame yes/no judge (vlm_lz.py judge caches, keep*.json):
              label-free but N forwards per sample
  attn        OURS: frozen model to --sel-layer, the tail (question) rows' attention mass
              summed per frame's tokens, top-k frames kept, M-RoPE positions recomputed for
              the short input (one forward selection, one forward answer)
  attn_norp   the SAME frames at their ORIGINAL positions (select without relabeling)
  rand        k random frames (floor)

Every arm answers [prefix | kept frame blocks | the SAME tail text] with one manual forward of
the frozen stack; EM = greedy first token is the gold digit (digit_acc = 0-9-restricted argmax).
Startup checks on sample 0: the manual stack reproduces model(**inputs), and the recomputed
positions of the all-frames sub-record equal the native ones. --scan: per-layer frame
recall@k of the selection signal instead of answering.

  python scripts/condmask/vlm_frozen.py --root data/mmred_vfiltered/seq_len_64/test \\
      --limit 100 --k 16 --sel-layer 16 --arms full+oracle+judge+attn+attn_norp+rand
"""
from __future__ import annotations

import argparse
import ast
import importlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gnnformer.data import build_count_prompt  # noqa: E402
from gnnformer.fencing import frame_blocks  # noqa: E402
from gnnformer.runtime import get_layers, get_rope_index_fn, load_runtime, move_to_device  # noqa: E402
from vlm_lz import dirs_of, load_sample  # noqa: E402

ARMS = ("full", "oracle", "judge", "attn", "attn_norp", "rand")


def facts_of(sd: Path, qc: str, qr: str) -> List[int]:
    """Frames showing character qc in room qr (the state dicts; a character may appear in
    OTHER rooms too, so vlm_lz's per-character judge/oracle keep a superset of these)."""
    lines = (sd / "qa.txt").read_text().splitlines()
    sts = [ast.literal_eval(l) for l in lines[lines.index("question:") + 1: lines.index("answer:")]
           if l.startswith("{")]
    return [j for j, st in enumerate(sts) if qc in st["rooms"].get(qr, [])]


@torch.no_grad()
def prep(rt, frames, question: str, resize: int) -> Dict[str, Any]:
    """Full prompt (N images + the canonical count prompt): embedded tokens with the image
    features scattered in, native M-RoPE positions, per-frame blocks, tail start."""
    model, dev = rt.model, rt.device
    content = [{"type": "image", "image": f.resize((resize, resize))} for f in frames]
    content.append({"type": "text", "text": build_count_prompt(question, len(frames))})
    inp = rt.processor.apply_chat_template([{"role": "user", "content": content}],
                                           add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt")
    inp = move_to_device(inp, dev)
    ids = inp["input_ids"][0].tolist()
    cfg = model.config
    vs = [i for i, t in enumerate(ids) if t == cfg.vision_start_token_id]
    fin = max(i for i, t in enumerate(ids) if t == cfg.vision_end_token_id) + 1
    blocks = frame_blocks(vs, fin)
    assert len(blocks) == len(frames), (len(blocks), len(frames))
    pos, _ = get_rope_index_fn(model)(inp["input_ids"], image_grid_thw=inp["image_grid_thw"],
                                      attention_mask=inp["attention_mask"])
    emb = model.model.language_model.embed_tokens(inp["input_ids"]).clone()
    img = model.model.get_image_features(inp["pixel_values"], inp["image_grid_thw"])
    img = torch.cat(img, 0) if isinstance(img, (list, tuple)) else img
    emb[0, inp["input_ids"][0] == cfg.image_token_id] = img.to(emb.dtype)
    return {"ids": ids, "emb": emb, "pos": pos, "blocks": blocks, "fin": fin,
            "prefix_end": vs[0], "grid": inp["image_grid_thw"], "inp": inp}


def sub(rt, rec, kept: List[int], norp: bool):
    """prefix + kept frame blocks (original order) + tail -> (emb, pos, fin, blocks).
    Positions: recomputed for the short input (repack) or the original ones (norp)."""
    idx = list(range(rec["prefix_end"]))
    blocks = []
    for j in kept:
        a, b = rec["blocks"][j]
        blocks.append((len(idx), len(idx) + b - a))
        idx.extend(range(a, b))
    fin = len(idx)
    idx.extend(range(rec["fin"], len(rec["ids"])))
    t = torch.tensor(idx, device=rt.device)
    emb = rec["emb"][:, t]
    if norp:
        pos = rec["pos"][:, :, t]
    else:
        ids = torch.tensor([[rec["ids"][i] for i in idx]], device=rt.device)
        pos, _ = get_rope_index_fn(rt.model)(ids, image_grid_thw=rec["grid"][kept],
                                             attention_mask=torch.ones_like(ids))
    return emb, pos, fin, blocks


def frame_mass(layer, h, pe, fin: int, blocks) -> torch.Tensor:
    """Mean-over-(heads, tail rows) attention weight at `layer`'s input h, summed per frame
    block -> [F]. M-RoPE applied exactly as the layer does (eval_babilong._row_mass analog)."""
    at = layer.self_attn
    hn = layer.input_layernorm(h)
    S, hd = hn.shape[1], at.head_dim
    q = at.q_proj(hn).view(1, S, -1, hd).transpose(1, 2)
    k = at.k_proj(hn).view(1, S, -1, hd).transpose(1, 2)
    mrope = importlib.import_module(type(at).__module__).apply_multimodal_rotary_pos_emb
    q, k = mrope(q, k, *pe, at.rope_scaling["mrope_section"])
    k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    sc = q[:, :, fin:] @ k.transpose(-1, -2) * at.scaling          # [1,H,T,S]
    T = S - fin
    causal = torch.arange(S, device=h.device)[None] > (fin + torch.arange(T, device=h.device))[:, None]
    w = torch.softmax(sc.masked_fill(causal, float("-inf")).float(), -1).mean((0, 1, 2))
    return torch.stack([w[a:b].sum() for a, b in blocks])


@torch.no_grad()
def run(rt, emb, pos, *, upto: int = -1, fin: int = 0, blocks=None, scan: bool = False):
    """Frozen stack, native causal attention. upto>=0: per-frame selection mass at that
    layer's input; scan: the masses at every layer; else the last-row logits."""
    model = rt.model
    lm, layers = model.model.language_model, get_layers(model)
    c, s = lm.rotary_emb(emb, pos)
    pe = (c.to(emb.dtype), s.to(emb.dtype))
    h, out = emb, []
    for li, ly in enumerate(layers):
        if scan or li == upto:
            out.append(frame_mass(ly, h, pe, fin, blocks))
            if not scan:
                return out[0]
        h = ly(h, attention_mask=None, position_embeddings=pe)[0]
    if scan:
        return out
    h = lm.norm(h)
    return model.lm_head(h[0, -1].to(model.lm_head.weight.dtype)).float()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="", help="default = the repo's Qwen2.5-VL-7B (4-bit)")
    ap.add_argument("--root", default="data/mmred_vfiltered/seq_len_64/test")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--k", type=int, default=16, help="frame budget of attn/attn_norp/rand")
    ap.add_argument("--sel-layer", type=int, default=16)
    ap.add_argument("--arms", default="full+oracle+judge+attn+attn_norp+rand")
    ap.add_argument("--resize", type=int, default=392, help="vlm_lz's frame size")
    ap.add_argument("--scan", action="store_true", help="per-layer frame recall@k; no answering")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="outputs/mmred_vfrozen")
    args = ap.parse_args()
    arms = [a for a in args.arms.split("+") if a]
    assert all(a in ARMS for a in arms), arms

    rt = load_runtime(args.model) if args.model else load_runtime()
    tok = rt.tokenizer
    mtag = (args.model or "7b").split("/")[-1].replace("Qwen", "q").lower()
    keep_name = f"keep_{mtag}.json" if args.model else "keep.json"   # vlm_lz.py's cache names
    digit_ids = torch.tensor([tok(str(d), add_special_tokens=False).input_ids[0] for d in range(10)])
    ds = dirs_of(args.root, args.limit, seed=args.seed)
    rng = random.Random(args.seed)
    root = Path(args.root)
    odir = (Path(args.output) / rt.model_name.split("/")[-1] / f"{root.parent.name}_{root.name}"
            / f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}")
    odir.mkdir(parents=True, exist_ok=True)
    (odir / "config.json").write_text(json.dumps(vars(args), indent=1))
    fout = open(odir / "results.jsonl", "w")
    acc, dacc, rec_at_k = ({a: [] for a in arms} for _ in range(3))
    scan_rec = None
    t0 = time.time()

    for i, sd in enumerate(ds):
        frames, q, gold, chars, qc, qr = load_sample(sd)
        facts = facts_of(sd, qc, qr)
        assert len(facts) == gold, sd
        rec = prep(rt, frames, q, args.resize)
        F = len(frames)
        if i == 0:                                    # startup checks (sample 0)
            with torch.no_grad():
                ref = rt.model(**rec["inp"]).logits[0, -1].float()
            e, p, _, _ = sub(rt, rec, list(range(F)), norp=False)
            assert torch.equal(p, rec["pos"]), "recomputed positions != native"
            mine = run(rt, e, p)
            assert int(mine.argmax()) == int(ref.argmax()), (int(mine.argmax()), int(ref.argmax()))
            print(f"startup: manual stack == model(**inp) (argmax {int(ref.argmax())}), positions ok", flush=True)
        if args.scan:
            per_layer = run(rt, rec["emb"], rec["pos"], fin=rec["fin"], blocks=rec["blocks"], scan=True)
            rc = torch.tensor([len(set(topk(m, args.k)) & set(facts)) / max(len(facts), 1) for m in per_layer])
            scan_rec = rc if scan_rec is None else scan_rec + rc
            continue
        mass = (run(rt, rec["emb"], rec["pos"], upto=args.sel_layer, fin=rec["fin"], blocks=rec["blocks"])
                if any(a.startswith("attn") for a in arms) else None)
        keeps: Dict[str, List[int]] = {}
        for a in arms:
            if a == "full":
                keeps[a] = list(range(F))
            elif a == "oracle":
                keeps[a] = facts or [0]                  # vlm_lz: no evidence -> first frame
            elif a == "judge":
                kj = json.loads((sd / keep_name).read_text())
                keeps[a] = [j for j, k in enumerate(kj) if k] or [0]   # vlm_lz: empty -> first frame
            elif a in ("attn", "attn_norp"):
                keeps[a] = topk(mass, args.k)
            elif a == "rand":
                keeps[a] = sorted(rng.sample(range(F), min(args.k, F)))
        for a in arms:
            kept = keeps[a]
            e, p, _, _ = sub(rt, rec, kept, norp=(a == "attn_norp"))
            lg = run(rt, e, p)
            text = tok.decode([int(lg.argmax())]).strip()
            ok = text == str(gold)
            dok = int(lg[digit_ids.to(lg.device)].argmax()) == gold
            rcl = len(set(kept) & set(facts)) / max(len(facts), 1)
            acc[a].append(int(ok)); dacc[a].append(int(dok)); rec_at_k[a].append(rcl)
            fout.write(json.dumps({"i": i, "sid": sd.name, "arm": a, "ok": ok, "digit_ok": dok, "out": text,
                                   "target": gold, "n_frames": F, "n_facts": len(facts), "kept": len(kept),
                                   "recall": rcl, "tokens": int(e.shape[1])}) + "\n")
        if i % 10 == 0:
            print(f"[{i+1}/{len(ds)}] {time.time()-t0:.0f}s F={F} seq={len(rec['ids'])} "
                  + " ".join(f"{a}={sum(acc[a])/len(acc[a]):.2f}" for a in arms), flush=True)

    if args.scan:
        rc = (scan_rec / len(ds)).tolist()
        print(f"per-layer frame recall@{args.k} (tail rows):")
        for li, v in enumerate(rc):
            print(f"  L{li:02d} {v:.3f}")
        (odir / "scan.json").write_text(json.dumps({"recall_at_k": rc, "k": args.k, "n": len(ds)}))
        print("->", odir)
        return 0
    summ = {a: {"acc": sum(acc[a]) / len(acc[a]), "digit_acc": sum(dacc[a]) / len(dacc[a]),
                "recall": sum(rec_at_k[a]) / len(rec_at_k[a]), "n": len(acc[a])} for a in arms}
    summ["_meta"] = {"model": rt.model_name, "root": args.root, "k": args.k, "sel_layer": args.sel_layer,
                     "resize": args.resize, "elapsed_s": time.time() - t0}
    (odir / "summary.json").write_text(json.dumps(summ, indent=1))
    print(f"\n{rt.model_name} {args.root} n={len(ds)} k={args.k} L{args.sel_layer}")
    for a in arms:
        print(f"  {a:10s} acc {summ[a]['acc']:.3f}  digit_acc {summ[a]['digit_acc']:.3f}  "
              f"frame-recall {summ[a]['recall']:.3f}")
    print("->", odir)
    return 0


def topk(score: torch.Tensor, k: int) -> List[int]:
    return sorted(score.topk(min(k, len(score))).indices.tolist())


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Vision LZ: two-pass select-and-repack on frozen 4-bit Qwen2.5-VL-7B — NO custom
attention machinery. Pass 1 (judge): the VLM zero-shot judges each frame image
("relevant to the question? yes/no" via logit comparison), cached as keep.json per
sample dir. Pass 2: standard VLM forward on the KEPT frames only (physical repack —
positions compact because the input is genuinely shorter).

Modes:
  judge --root DIR [--resize 224]        write keep.json per sample + oracle agreement
  train --selection {none,judge,oracle}  LoRA digit-count training on (selected) frames
  eval  --ckpt X --selection ... --roots a=100+b=100   greedy EM per root

Task: visual filtered-counting (data/mmred_vfiltered — one character per frame, park
renders; gold <= 8 at every N; relevance is question-dependent)."""
from __future__ import annotations

import argparse
import ast as _ast
import json
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gnnformer.carriers import attach_lora
from gnnformer.runtime import get_layers, load_runtime

Q_RE = re.compile(r"How many frames show (\w+) in the (\w+)\?")


def load_sample(sd: Path):
    from PIL import Image
    lines = (sd / "qa.txt").read_text().splitlines()
    qi = lines.index("question:")
    ai = lines.index("answer:")
    states = [l for l in lines[qi + 1: ai] if l.startswith("{")]
    q = [l for l in lines[qi + 1: ai] if l and not l.startswith("{")][0]
    gold = int(lines[ai + 1])
    chars = []
    for s_ in states:
        st = _ast.literal_eval(s_)
        chars.append([c for cs in st["rooms"].values() for c in cs][0])
    frames = [Image.open(sd / f"{i:03d}.png").convert("RGB")
              for i in range(len(states))]
    m = Q_RE.match(q)
    return frames, q, gold, chars, m.group(1), m.group(2)


def dirs_of(root, limit, seed=0):
    ds = [p for p in sorted(Path(root).iterdir()) if (p / "qa.txt").exists()]
    random.Random(seed).shuffle(ds)
    return ds[:limit]


def build_inputs(proc, frames, question, n, dev, resize):
    from gnnformer.data import build_count_prompt
    content = [{"type": "image", "image": f.resize((resize, resize))} for f in frames]
    content.append({"type": "text", "text": build_count_prompt(question, n)})
    inp = proc.apply_chat_template([{"role": "user", "content": content}],
                                   add_generation_prompt=True, tokenize=True,
                                   return_dict=True, return_tensors="pt")
    return {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inp.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("judge", "train", "eval"))
    ap.add_argument("--root", default="")
    ap.add_argument("--roots", default="", help="'+'-separated root[=LIMIT] (eval)")
    ap.add_argument("--train-roots",
                    default="data/mmred_vfiltered/seq_len_8/train=3600"
                            "+data/mmred_vfiltered/seq_len_32/train=900")
    ap.add_argument("--dev-root", default="data/mmred_vfiltered/seq_len_8/dev")
    ap.add_argument("--selection", choices=("none", "judge", "oracle"), default="judge")
    ap.add_argument("--limit", type=int, default=100000)
    ap.add_argument("--dev-limit", type=int, default=150)
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--model", default="")
    ap.add_argument("--judge-resize", type=int, default=392)
    ap.add_argument("--judge-prompt", choices=("relevant", "direct"), default="direct",
                    help="direct = 'Is {C} in this image?' (question-derived, easier "
                         "for a VLM than the meta-question)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--output", default="outputs/vlm_lz")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    rt = load_runtime(args.model) if args.model else load_runtime()
    mtag = (args.model or "7b").split("/")[-1].replace("Qwen", "q").lower()
    keep_name = f"keep_{mtag}.json" if args.model else "keep.json"
    proc, tok, model, dev = rt.processor, rt.tokenizer, rt.model, rt.device
    digit_ids = [tok(str(d), add_special_tokens=False).input_ids[0] for d in range(10)]
    yes_id = tok(" yes", add_special_tokens=False).input_ids[0]
    no_id = tok(" no", add_special_tokens=False).input_ids[0]

    # ------------------------------------------------------------------- judge mode
    if args.mode == "judge":
        assert args.root
        ds = dirs_of(args.root, args.limit)
        agr = n_f = 0
        stats = [0, 0, 0, 0]
        t0 = time.time()
        for di, sd in enumerate(ds):
            if (sd / keep_name).exists():
                continue
            frames, q, gold, chars, qc, qr = load_sample(sd)
            jtxt = (f"Is {qc} in this image? Answer yes or no:"
                    if args.judge_prompt == "direct" else
                    f"Question: {q}\nIs this frame relevant to the question? "
                    f"Answer yes or no:")
            keeps = []
            with torch.no_grad():
                for f in frames:
                    content = [{"type": "image",
                                "image": f.resize((args.judge_resize,) * 2)},
                               {"type": "text", "text": jtxt}]
                    inp = proc.apply_chat_template(
                        [{"role": "user", "content": content}],
                        add_generation_prompt=True, tokenize=True, return_dict=True,
                        return_tensors="pt")
                    inp = {k: (v.to(dev) if hasattr(v, "to") else v)
                           for k, v in inp.items()}
                    lg = model(**inp).logits[0, -1]
                    keeps.append(int(lg[yes_id] > lg[no_id]))
            (sd / keep_name).write_text(json.dumps(keeps))
            for k, c in zip(keeps, chars):
                agr += int(k == (c == qc))
                n_f += 1
                rel = (c == qc)
                stats[0] += int(k and rel); stats[1] += int(rel)      # recall
                stats[2] += int(k and not rel); stats[3] += int(not rel)  # fp rate
            if di % 200 == 0:
                print(f"[judge] {di}/{len(ds)} agreement {agr/max(n_f,1):.3f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        print(f"[judge] DONE {args.root}: agreement {agr/max(n_f,1):.3f} "
              f"recall(rel) {stats[0]/max(stats[1],1):.3f} "
              f"keep(dist) {stats[2]/max(stats[3],1):.3f} (n={n_f})")
        return 0

    def select(frames, chars, qc, sd):
        if args.selection == "none":
            return frames
        if args.selection == "oracle":
            kept = [f for f, c in zip(frames, chars) if c == qc]
        else:
            keeps = json.loads((sd / keep_name).read_text())
            kept = [f for f, k in zip(frames, keeps) if k]
        return kept or frames[:1]

    def answer_logits(frames, q):
        inp = build_inputs(proc, frames, q, len(frames), dev, args.resize)
        return model(**inp).logits[0, -1]

    @torch.no_grad()
    def greedy_em(ds):
        hits = 0
        for sd in ds:
            frames, q, gold, chars, qc, _ = load_sample(sd)
            kept = select(frames, chars, qc, sd)
            inp = build_inputs(proc, kept, q, len(kept), dev, args.resize)
            out = model.generate(**inp, max_new_tokens=3, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
            txt = tok.decode(out[0, inp["input_ids"].shape[1]:],
                             skip_special_tokens=True).strip()
            m = re.match(r"(\d+)", txt)
            hits += int(bool(m) and int(m.group(1)) == gold)
        return hits / max(len(ds), 1)

    # -------------------------------------------------------------------- eval mode
    if args.mode == "eval":
        assert args.ckpt and args.roots
        ck = torch.load(args.ckpt, map_location="cpu")
        lora = attach_lora(get_layers(model), 0, rank=ck["rank"], alpha=ck["alpha"],
                           device=dev, state=ck["lora"])
        for p in lora.parameters():
            p.requires_grad_(False)
        import os
        out = Path(args.output) / (time.strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}")
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        for spec in args.roots.replace("+", ",").split(","):
            root, _, lim = spec.partition("=")
            ds = dirs_of(root, int(lim) if lim else 100, seed=9)
            em = greedy_em(ds)
            rows.append({"root": root, "n": len(ds), "em": em,
                         "selection": args.selection})
            print(f"[{root}] n={len(ds)} EM {em:.3f} sel={args.selection}", flush=True)
            (out / "report.txt").write_text(json.dumps(rows, indent=2) + "\n")
        lora.remove()
        return 0

    # ------------------------------------------------------------------- train mode
    run = f"sel-{args.selection}_seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    out = Path(args.output) / run
    out.mkdir(parents=True, exist_ok=True)
    tr = []
    for spec in args.train_roots.replace("+", ",").split(","):
        root, _, lim = spec.partition("=")
        tr += dirs_of(root, int(lim) if lim else 100000, seed=args.seed)
    random.Random(args.seed).shuffle(tr)
    ev = dirs_of(args.dev_root, args.dev_limit, seed=1234)
    print(f"train {len(tr)} dev {len(ev)} selection={args.selection}")
    lora = attach_lora(get_layers(model), 0, rank=args.rank, alpha=args.alpha,
                       device=dev)
    opt = torch.optim.Adam(lora.parameters(), lr=args.lr)
    best, best_ep, bad = -1.0, -1, 0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        random.Random(ep).shuffle(tr)
        ce_sum = 0
        for i, sd in enumerate(tr):
            frames, q, gold, chars, qc, _ = load_sample(sd)
            kept = select(frames, chars, qc, sd)
            lg = answer_logits(kept, q)
            loss = F.cross_entropy(lg.unsqueeze(0),
                                   torch.tensor([digit_ids[gold]], device=dev))
            (loss / args.accum).backward()
            ce_sum += float(loss.detach())
            if (i + 1) % args.accum == 0:
                opt.step()
                opt.zero_grad()
        opt.step(); opt.zero_grad()
        dev_em = greedy_em(ev)
        print(f"[ep {ep}] ce {ce_sum/len(tr):.4f} | dev EM {dev_em:.3f} | "
              f"{time.time()-t0:.0f}s", flush=True)
        ck = {"selection": args.selection, "config": vars(args), "epoch": ep,
              "dev_em": dev_em, "rank": args.rank, "alpha": args.alpha,
              "lora": lora.state()}
        torch.save(ck, out / "vlmlz_last.pt")
        if dev_em > best:
            best, best_ep, bad = dev_em, ep, 0
            torch.save(ck, out / "vlmlz_best.pt")
        else:
            bad += 1
        (out / "report.txt").write_text(f"{run}\nep {ep} dev {dev_em:.3f} best "
                                        f"{best:.3f}@{best_ep}\n")
        if bad >= args.patience and ep >= 2:
            break
    lora.remove()
    print(f"DONE best dev EM {best:.3f} @ep{best_ep} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

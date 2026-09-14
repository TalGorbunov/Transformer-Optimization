"""Training-free fenced judge, TEXT (PREREG_AGG 2026-09-07: A2, A3, B1). Every unit sits in its own block behind the
fence (per-block position reset); the block ends with an optional cue and " Answer:"; the verdict is the frozen
model's own logit(Yes) - logit(No) at the block's last token; count = #(verdict > 0). No trained parameters.
Cue modes: direct ("Is {who} in the {room} in this frame?"), meta ("Is this frame relevant to the question?"),
none (question only in the prefix). Config "mode:fence" with fence 1 = fence + reset, 0 = causal, native positions.
Reuses the layout/forward of train_bindcount.py (e_c = None, no carriers)."""
import argparse, json, os, random, re, sys, time
from pathlib import Path
import torch
sys.path.insert(0, "scripts/condmask"); sys.path.insert(0, "."); sys.path.insert(0, "outputs/_scratch/dbg")
from modeling import load_text_model  # noqa: E402
from train_bindcount import forward, mmred_units, needle_units  # noqa: E402
from needles_types import load_pool  # noqa: E402

YES, NO = [" Yes", "Yes", " yes", "yes"], [" No", "No", " no", "no"]


def cue_for(kind, question, mode):
    if mode == "none": return ""
    if mode == "meta": return f" Is this {'frame' if kind == 'mmred' else 'sentence'} relevant to the question?"
    if kind == "mmred":
        who, room = re.fullmatch(r"How many frames show (\w+) in the (\w+)\?", question).groups(); return f" Is {who} in the {room} in this frame?"
    p, r = re.fullmatch(r"How many times did (\w+) go to the (\w+)\?", question).groups(); return f" Does {p} go to the {r} in this sentence?"


def build_rec_cue(tok, units, question, gold, facts, kind, mode):
    noun = "frames" if kind == "mmred" else "sentences"
    prefix = f"You will be shown {len(units)} {noun}.\nQuestion: {question}"
    ids = list(tok(prefix, add_special_tokens=False).input_ids); pe = len(ids); blocks = []; cue = cue_for(kind, question, mode)
    for u in units:
        seg = tok("\n" + u + cue + " Answer:", add_special_tokens=False).input_ids
        blocks.append((len(ids), len(ids) + len(seg))); ids.extend(seg)
    fin = len(ids); ids.extend(tok(f"\nQuestion: {question}\nAnswer: ", add_special_tokens=False).input_ids)
    return dict(ids=ids, prefix_end=pe, blocks=blocks, carriers=[], fin=fin, seq=len(ids), gold=gold, facts=facts,
                is_fact=[int(j in set(facts)) for j in range(len(units))], n=len(units))


@torch.no_grad()
def judge(model, tok, rec, dev, fence, yes_ids, no_ids):
    rows = [b - 1 for _, b in rec["blocks"]]
    st = forward(model, rec, None, dev, 10 ** 6 if fence else None, bool(fence), rows)
    lg = model.lm_head(st.to(model.dtype)).float()
    z = torch.logsumexp(lg[:, yes_ids], -1) - torch.logsumexp(lg[:, no_ids], -1)
    top = lg.argmax(-1); yn = torch.tensor([int(t in set(yes_ids + no_ids)) for t in top.tolist()])
    return z.cpu(), yn


def evaluate(model, tok, items, dev, mode, fence, yes_ids, no_ids, tag):
    ex = mae = vc = vn = 0; zs, ys, yns, margins = [], [], [], []
    for kind, units, q, gold, facts, *_ in items:
        rec = build_rec_cue(tok, units, q, gold, facts, kind, mode); z, yn = judge(model, tok, rec, dev, fence, yes_ids, no_ids)
        y = torch.tensor(rec["is_fact"]); pred = int((z > 0).sum()); ex += int(pred == gold); mae += abs(pred - gold)
        vc += int(((z > 0).long() == y).sum()); vn += len(y); zs.append(z); ys.append(y); yns.append(yn)
        if y.sum() > 0 and (1 - y).sum() > 0: margins.append(float(z[y == 1].min() - z[y == 0].max()))
    zz, yy = torch.cat(zs), torch.cat(ys); pos, neg = zz[yy == 1], zz[yy == 0]
    auc = float((pos[:, None] > neg[None]).float().mean() + 0.5 * (pos[:, None] == neg[None]).float().mean()) if len(pos) and len(neg) else None
    return dict(tag=tag, mode=mode, fence=fence, n=len(items), exact=ex / len(items), mae=mae / len(items), verdict_acc=vc / vn, verdict_auc=auc,
                yesno_top1=float(torch.cat(yns).float().mean()), min_margin_med=(float(torch.tensor(margins).median()) if margins else None),
                seq_tokens=rec["seq"])


def run(a):
    dev = "cuda"; tok, model = load_text_model(a.model, device=dev); pool = load_pool()
    yes_ids = sorted({tok(s, add_special_tokens=False).input_ids[0] for s in YES}); no_ids = sorted({tok(s, add_special_tokens=False).input_ids[0] for s in NO})
    odir = Path(a.output) / a.model.split("/")[-1] / f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"; odir.mkdir(parents=True, exist_ok=True)
    (odir / "config.json").write_text(json.dumps(dict(vars(a), yes_ids=yes_ids, no_ids=no_ids), indent=1)); log = open(odir / "log.txt", "a")
    def P(*x):
        s = " ".join(str(v) for v in x); print(s, flush=True); log.write(s + "\n"); log.flush()
    t0 = time.time(); results = []
    ns = [int(x) for x in a.eval_ns.split("+")]; nns = [int(x) for x in a.needle_ns.split("+")] if a.needle_ns else []
    data = {("mmred", N): mmred_units(f"data/mmred_filtered/seq_len_{N}/test", a.limit_eval, random.Random(1)) for N in ns if Path(f"data/mmred_filtered/seq_len_{N}/test").exists()}
    for N in nns: data[("needles", N)] = needle_units(a.limit_eval, N, min(64, N // 2), random.Random(2), pool, kinds=("HC",))
    rec0 = build_rec_cue(tok, data[("mmred", ns[0])][0][1], data[("mmred", ns[0])][0][2], 0, [], "mmred", "direct")
    P("layout check:", repr(tok.decode(rec0["ids"][rec0["blocks"][0][0]:rec0["blocks"][0][1]])), "| read token:", repr(tok.decode([rec0["ids"][rec0["blocks"][0][1] - 1]])))
    for cfg in a.configs.split("+"):
        mode, fence = cfg.split(":"); fence = int(fence)
        for (kind, N), items in data.items():
            if not items: continue
            try:
                r = evaluate(model, tok, items, dev, mode, fence, yes_ids, no_ids, f"{kind} N={N}")
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); r = dict(tag=f"{kind} N={N}", mode=mode, fence=fence, oom=True)
            results.append(r); P(f"[{time.time()-t0:.0f}s] {r}")
            (odir / "results.json").write_text(json.dumps(results, indent=1))
    P("->", odir); return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct"); ap.add_argument("--configs", default="direct:1+meta:1+none:1+direct:0")
    ap.add_argument("--eval-ns", default="8+16+32+64+128+256+512+1024"); ap.add_argument("--needle-ns", default="64+128+256")
    ap.add_argument("--limit-eval", type=int, default=100); ap.add_argument("--output", default="outputs/judge_fenced")
    return run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())

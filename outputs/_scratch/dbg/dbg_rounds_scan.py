"""Do the copy rows carry hop-2 signal at ANY layer? Qwen-3B, qa2/qa3 4k, k=16, 10 samples.
Round-1 kept = layer-27 tail rows. Per layer l: s2 = copy rows' mass (kept zeroed);
report recall of top-k by dist(s1)+dist(s2_l), and precision of s2's top-8 among non-kept."""
import sys
from pathlib import Path
sys.path.insert(0, "scripts/condmask")
import torch
import eval_babilong as eb
from modeling import load_text_model, model_parts

tok, model = load_text_model("Qwen/Qwen2.5-3B-Instruct", device="cuda")
root = Path("data/babilong/1k+data/babilong/100")
K, L1 = 16, 27
nl = len(model_parts(model)[1])
dist = lambda m: (lambda f: f / f.sum().clamp_min(1e-12))(torch.where(torch.isfinite(m), m, 0.).float().cpu())
for task in ("qa2", "qa3"):
    df = eb.load_split(root, task, "4k").iloc[:10]
    vocab = eb.fact_vocab(root, task)
    rec1 = torch.zeros(nl); prec = torch.zeros(nl); base = 0.
    for r in df.itertuples():
        sents = eb.split_sents(r.input)
        rec = eb.build(tok, task, r.question, sents)
        facts = set(j for j, s in enumerate(sents) if s in vocab)
        rc = lambda m: len(set(eb.topk_keep(m, K)) & facts) / len(facts)
        s1 = eb.attn_scores(model, rec, L1, "tail", "cuda").float().cpu()
        kept = eb.topk_keep(s1, K); base += rc(s1)
        r2 = eb.two_round_rec(rec, kept)
        per_layer = eb.attn_scores(model, r2, 0, "copies", "cuda", scan=True)
        for l, s2 in enumerate(per_layer):
            s2 = dist(s2); s2[kept] = 0
            rec1[l] += rc(dist(s1) + dist(s2))
            top8 = eb.topk_keep(s2, 8)
            prec[l] += len(set(top8) & facts) / 8
    n = len(df)
    print(f"{task}: one-round recall {base/n:.3f}; non-kept fact share (chance precision) ~ "
          f"{sum(1 for _ in facts)/len(sents):.2f}")
    for l in range(nl):
        print(f"  L{l:2d} two-round recall {rec1[l]/n:.3f}  s2 top-8 precision {prec[l]/n:.3f}")

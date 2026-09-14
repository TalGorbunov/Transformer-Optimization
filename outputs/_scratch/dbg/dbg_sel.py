"""Round-2 (copy-row) selection vs one round: recall at k=16, Qwen-3B, 4k, qa1-3, 10 samples each."""
import sys, types
from pathlib import Path
sys.path.insert(0, "scripts/condmask")
import torch
import eval_babilong as eb
from modeling import load_text_model

tok, model = load_text_model("Qwen/Qwen2.5-3B-Instruct", device="cuda")
root = Path("data/babilong/1k+data/babilong/100")
K = 16
for task in ("qa1", "qa2", "qa3"):
    df = eb.load_split(root, task, "4k").iloc[:10]
    vocab = eb.fact_vocab(root, task)
    tot = {"one": 0., "two": 0., "two_t": 0.}
    for r in df.itertuples():
        sents = eb.split_sents(r.input)
        rec = eb.build(tok, task, r.question, sents)
        facts = [j for j, s in enumerate(sents) if s in vocab]
        rc = lambda m: len(set(eb.topk_keep(m, K)) & set(facts)) / len(facts)
        a = types.SimpleNamespace(sel_tournament=0, sel_chunk=0, sel_layer=27, sel_rows="tail", sel_rounds=1, k=K)
        one = eb.select_scores(model, rec, a, "cuda"); a.sel_rounds = 2
        two = eb.select_scores(model, rec, a, "cuda")
        a.sel_chunk, a.sel_tournament = 2048, 16
        two_t = eb.select_scores(model, rec, a, "cuda")
        tot["one"] += rc(one); tot["two"] += rc(two); tot["two_t"] += rc(two_t)
        order = two.argsort(descending=True).tolist()
        print(f"{task} F={len(sents)} facts={facts} one={rc(one):.2f} two={rc(two):.2f} two_t={rc(two_t):.2f} "
              f"fact ranks one={[one.argsort(descending=True).tolist().index(j) for j in facts]} two={[order.index(j) for j in facts]}")
    print(task, {k: round(v / len(df), 3) for k, v in tot.items()})

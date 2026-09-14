"""CPU dry run of eval_babilong.py --mmred: (1) load_mmred integrity on every seq_len root
(facts == gold), (2) the full arm loop on a tiny random LLaMA with the real Qwen tokenizer."""
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, "scripts/condmask"); sys.path.insert(0, "tests")
import eval_babilong as N  # noqa: E402
from test_needles import tiny  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

for n in (8, 16, 32, 64, 128, 256, 512, 1024):
    df = N.load_mmred(Path(f"data/mmred_filtered/seq_len_{n}/test"), 0)
    assert (df.input.str.count("\n") + 1 == n).all(), n
    print(f"N{n}: {len(df)} samples, gold hist {sorted(df.target.astype(int).value_counts().to_dict().items())}")

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
df = N.load_mmred(Path("data/mmred_filtered/seq_len_8/test"), 0)
r = df.iloc[0]
sents = r.input.split("\n")
rec = N.build(tok, "x", r.question, sents, N.MMRED_PROMPT)
print("---- sample 0 prompt (blocks marked) ----")
ids = rec["ids"]
out = tok.decode(ids[:rec["prefix_end"]])
for j, (a, b) in enumerate(rec["blocks"]):
    out += f"|B{j}{'*' if j in r.facts else ''}|" + tok.decode(ids[a:b])
out += "|TAIL|" + tok.decode(ids[rec["fin"]:])
print(out); print("facts", r.facts, "gold", r.target)

model = tiny(tok)
N.load_text_model = lambda name, device=None, **kw: (tok, model)
sys.argv = ["x", "--model", "Tiny", "--mmred", "data/mmred_filtered/seq_len_16/test", "--limit", "3",
            "--k", "4", "--sel-layer", "1", "--arms", "full+oracle+attn+attn_norp+lex+rand",
            "--output", "outputs/_scratch/dbg/mmred_dry", "--max-new", "3"]
assert N.main() == 0

# ponytail: CPU dry run of needles.run()+analyze() with the tiny random Llama from tests/test_needles.py
import sys, types, argparse
from pathlib import Path
sys.path.insert(0, "tests"); sys.path.insert(0, "scripts/condmask")
import test_needles as T, needles as N
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct"); model = T.tiny(tok)
N.load_text_model = lambda name, device=None: (tok, model)
out = sys.argv[1]
a = argparse.Namespace(model="Tiny", layer=1, n=4, ks="1+2+4+8", regimes="HC+DUP", max_new=4, seed=0, output=out, analyze=None)
d = N.run(a); N.analyze(Path(d)); print(open(Path(d) / "ANALYSIS.md").read())

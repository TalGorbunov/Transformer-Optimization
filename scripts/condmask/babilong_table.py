"""Aggregate BABILong summary.json files into one table.

    python scripts/condmask/babilong_table.py [outputs/babilong outputs/babilong_yarn4 ...]

Rows: model / task / length; columns: acc per arm (fact-recall in parentheses for the
selection arms). Newest run dir wins when a cell was run twice.
"""
import glob
import json
import sys

LEN_ORDER = ["0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k"]
ARMS = ["full", "oracle", "attn", "attn_norp", "lex", "emb", "rand"]

roots = sys.argv[1:] or ["outputs/babilong", "outputs/babilong_yarn4"]
cells = {}
for root in roots:
    for f in sorted(glob.glob(f"{root}/*/*/*/summary.json")):
        s = json.load(open(f))
        m = s["_meta"]
        tag = root.rstrip("/").split("/")[-1].replace("babilong", "") or "native"
        cells[(m["model"].split("/")[-1], tag, m["task"], m["length"])] = s

def fmt(s, a):
    if a not in s:
        return "   -  "
    r = s[a]
    return f"{r['acc']:.2f}" + (f"({r['recall']:.2f})" if a not in ("full",) else "      ")

hdr = f"{'model':22s} {'cfg':7s} {'task':4s} {'len':5s} {'n':>4s} " + " ".join(f"{a:>10s}" for a in ARMS)
print(hdr)
print("-" * len(hdr))
for key in sorted(cells, key=lambda k: (k[0], k[1], k[2], LEN_ORDER.index(k[3]))):
    s = cells[key]
    n = s["full"]["n"] if "full" in s else next(v["n"] for k, v in s.items() if k != "_meta")
    print(f"{key[0]:22s} {key[1]:7s} {key[2]:4s} {key[3]:5s} {n:4d} " + " ".join(f"{fmt(s, a):>10s}" for a in ARMS))

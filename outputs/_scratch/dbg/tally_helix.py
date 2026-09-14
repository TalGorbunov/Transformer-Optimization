"""S2 in-model tally feasibility (PREREG_AGG 2026-09-07: H1, H2, H3). Frozen Qwen2.5-7B-Instruct.
Number vectors v_k^L = mean answer-row residual (output of layer L) over copy templates where the model is about to emit k.
H1  patch the answer row of plain MMRED prompts (N in 64/128/256) at layer L with v_k (re-applied at every greedy step):
    does the model emit k?  (k = gold and a grid up to 64; frozen no-patch baseline alongside)
H2  is v_k a helix in k?  least squares on [1, k, cos/sin(2 pi k / T), T in 2,5,10,100]; R^2 per layer vs linear-only.
H3  fit the helix on k <= 32 only, reconstruct v_k for k in 33..64, patch: does the model emit k?"""
import argparse, json, math, os, random, sys, time
from pathlib import Path
import torch
sys.path.insert(0, "scripts/condmask"); sys.path.insert(0, "."); sys.path.insert(0, "outputs/_scratch/dbg")
from modeling import load_text_model  # noqa: E402
from train_bindcount import build_rec, mmred_units  # noqa: E402

TEMPLATES = [   # trailing space: Qwen tokenizes " 12" as " ", "1", "2" -> the next token after the space is the first digit
    "There are {k} apples in the basket. How many apples are in the basket? The answer is ",
    "The team scored {k} points. How many points did the team score? Answer: ",
    "Mary counted {k} birds in the garden. How many birds did Mary count? The number is ",
    "The box contains {k} coins. Number of coins in the box: ",
    "John visited the kitchen {k} times today. How many times did John visit the kitchen? Answer: ",
    "Frame count: {k}. The number of frames is ",
]
PERIODS = (2, 5, 10, 100)


def phi(ks, helix=True):
    ks = torch.tensor(ks, dtype=torch.float64); cols = [torch.ones_like(ks), ks / 64.0]
    if helix:
        for T in PERIODS:
            cols.append(torch.cos(2 * math.pi * ks / T))
            if T != 2: cols.append(torch.sin(2 * math.pi * ks / T))      # sin(pi k) == 0 for integer k: drop the zero column
    return torch.stack(cols, -1)


def fit(V, ks, helix):
    X = phi(ks, helix); W = torch.linalg.pinv(X) @ V.double(); R = V.double() - X @ W
    r2 = 1 - (R ** 2).sum() / ((V.double() - V.double().mean(0)) ** 2).sum(); return W, float(r2)


@torch.no_grad()
def decode(model, tok, ids, dev, patch=None, max_new=3):
    """patch = (L, row, vector): replace the residual of `row` at the output of layer L on every step."""
    hd = None
    if patch is not None:
        L, row, v = patch
        def hook(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out; o[:, row, :] = v.to(o.dtype); return out
        hd = model.model.layers[L].register_forward_hook(hook)
    try:
        cur = list(ids); toks = []
        for step in range(max_new + 1):
            t = int(model(torch.tensor([cur], device=dev)).logits[0, -1].argmax()); s = tok.decode([t])
            if step == 0 and s != "" and s.strip() == "": cur.append(t); continue        # a bare space token before the digits
            if not s.strip().isdigit(): break
            toks.append(t); cur.append(t)
    finally:
        if hd is not None: hd.remove()
    txt = tok.decode(toks).strip(); return int(txt) if txt.isdigit() else None


def run(a):
    dev = "cuda"; tok, model = load_text_model(a.model, device=dev); layers = [int(x) for x in a.layers.split("+")]; kmax = 64
    odir = Path(a.output) / a.model.split("/")[-1] / f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"; odir.mkdir(parents=True, exist_ok=True)
    (odir / "config.json").write_text(json.dumps(vars(a), indent=1)); log = open(odir / "log.txt", "a")
    def P(*x):
        s = " ".join(str(v) for v in x); print(s, flush=True); log.write(s + "\n"); log.flush()
    t0 = time.time()
    # ---- number vectors
    V = {L: torch.zeros(kmax + 1, model.config.hidden_size, dtype=torch.float32) for L in layers}; copy_ok = 0; copy_n = 0
    with torch.no_grad():
        for k in range(kmax + 1):
            for tpl in TEMPLATES:
                ids = tok(tpl.format(k=k), add_special_tokens=False).input_ids; out = model(torch.tensor([ids], device=dev), output_hidden_states=True)
                for L in layers: V[L][k] += out.hidden_states[L + 1][0, -1].float().cpu() / len(TEMPLATES)
                copy_ok += int(tok.decode([int(out.logits[0, -1].argmax())]).strip() == str(k)[0]); copy_n += 1
                if k == 12 and tpl is TEMPLATES[0]: P("template check:", repr(tok.decode(ids[-4:])), "-> next", repr(tok.decode([int(out.logits[0, -1].argmax())])))
    P(f"[{time.time()-t0:.0f}s] number vectors done; template copy accuracy (first digit) {copy_ok/copy_n:.3f}")
    torch.save(V, odir / "number_vectors.pt")
    # ---- H2 geometry
    h2 = {}
    for L in layers:
        _, r2h = fit(V[L], list(range(kmax + 1)), True); _, r2l = fit(V[L], list(range(kmax + 1)), False)
        Vc = V[L] - V[L].mean(0); _, r2h_c = fit(Vc, list(range(kmax + 1)), True)
        h2[L] = dict(r2_helix=r2h, r2_linear=r2l, r2_helix_centered=r2h_c); P(f"H2 layer {L}: R2 helix {r2h:.3f} linear {r2l:.3f}")
    # ---- H1 read test on MMRED prompts
    grid = [int(x) for x in a.kgrid.split("+")]; h1 = []; targets = []
    for N in [int(x) for x in a.ns.split("+")]:
        for kind, units, q, gold, facts, *_ in mmred_units(f"data/mmred_filtered/seq_len_{N}/test", a.limit, random.Random(1)):
            rec = build_rec(tok, units, q, gold, facts, None, kind); targets.append((N, rec["ids"], gold))
    for N, ids, gold in targets:
        row = len(ids) - 1; base = decode(model, tok, ids, dev)
        for L in layers:
            for k in sorted(set(grid + [gold])):
                em = decode(model, tok, ids, dev, patch=(L, row, V[L][k].to(dev)))
                h1.append(dict(N=N, gold=gold, L=L, k=k, emitted=em, ok=em == k, is_gold=k == gold, frozen=base, frozen_ok=base == gold))
        (odir / "h1.json").write_text(json.dumps(h1))
    def acc(rows): return sum(r["ok"] for r in rows) / max(1, len(rows))
    summ = {}
    for L in layers:
        rows = [r for r in h1 if r["L"] == L]
        summ[L] = dict(gold=acc([r for r in rows if r["is_gold"]]), k_le8=acc([r for r in rows if r["k"] <= 8]), k_9_32=acc([r for r in rows if 9 <= r["k"] <= 32]), k_33_64=acc([r for r in rows if r["k"] > 32]), all=acc(rows))
        P(f"H1 layer {L}: emitted==k  gold {summ[L]['gold']:.2f}  k<=8 {summ[L]['k_le8']:.2f}  9-32 {summ[L]['k_9_32']:.2f}  33-64 {summ[L]['k_33_64']:.2f}")
    fro = {N: sum(r["frozen_ok"] for r in h1 if r["N"] == N and r["L"] == layers[0] and r["is_gold"]) / max(1, sum(1 for r in h1 if r["N"] == N and r["L"] == layers[0] and r["is_gold"])) for N in sorted({r["N"] for r in h1})}
    P("frozen (no patch) exact by N:", fro)
    best = max(layers, key=lambda L: summ[L]["all"])
    # ---- H3 extrapolation at the best layer
    Wh, _ = fit(V[best][:33], list(range(33)), True); Wl, _ = fit(V[best][:33], list(range(33)), False); h3 = []
    ks3 = [int(x) for x in a.k3.split("+")]; tg3 = [t for t in targets if t[0] == int(a.n3)][: a.limit]
    for N, ids, gold in tg3:
        row = len(ids) - 1
        for k in ks3:
            vh = (phi([k], True) @ Wh)[0].float().to(dev); vl = (phi([k], False) @ Wl)[0].float().to(dev)
            for name, v in (("true", V[best][k].to(dev)), ("helix_fit32", vh), ("linear_fit32", vl)):
                em = decode(model, tok, ids, dev, patch=(best, row, v)); h3.append(dict(k=k, variant=name, emitted=em, ok=em == k))
    h3s = {name: acc([r for r in h3 if r["variant"] == name]) for name in ("true", "helix_fit32", "linear_fit32")}
    P(f"H3 layer {best}, k in {ks3}: emitted==k  true-vector {h3s['true']:.2f}  helix fit on k<=32 {h3s['helix_fit32']:.2f}  linear fit {h3s['linear_fit32']:.2f}")
    json.dump(dict(copy_acc=copy_ok / copy_n, h2=h2, h1=summ, frozen=fro, best_layer=best, h3=h3s, h3_rows=h3), open(odir / "summary.json", "w"), indent=1)
    P("->", odir); return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct"); ap.add_argument("--layers", default="8+12+16+20+24"); ap.add_argument("--ns", default="64+128+256")
    ap.add_argument("--limit", type=int, default=20); ap.add_argument("--kgrid", default="5+12+27+48+64"); ap.add_argument("--k3", default="33+37+42+48+55+64"); ap.add_argument("--n3", default="128")
    ap.add_argument("--output", default="outputs/tally_helix"); return run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())

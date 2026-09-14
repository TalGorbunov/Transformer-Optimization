"""What does length-finetuning actually change inside the model?

For each model (frozen / LoRA-{8,16} / LoRA-{8,32}) x each N: full attention,
native positions (the finetuned regime), measure at the answer row of layer L:
  EM         free digit decode exact match
  mass       attention-mass share on evidence frames (post-softmax, mean heads)
  keff       effective fan-in 1/sum(alpha^2) over frame columns
  gap        PRE-softmax score gap: mean evidence score - mean junk score
             (the anti-dilution margin; constant mass needs gap ~ log N)
  infl       Jacobian evidence-influence share sum_rel/sum_all of
             ||d gold-logit / d h_L(frame)|| (8 samples)
Hypothesis H1 (margin calibration): FT raises `gap` by ~constant -> the EM wall
sits where the fixed margin is exhausted; training longer moves the knee, never
removes it. Repack makes the required margin N-independent (the contrast row).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1]))

from gnnformer.carriers import attach_lora
from modeling import load_text_model, model_parts, pad_batch, sdpa_kernel, SDPA_BACKENDS
from relgate import prep_root_rel, rel_mask_parts

DEV = "cuda"
L_SEL = 16
NS = (8, 16, 32, 64, 128, 256, 1024)
ROOT = "data/mmred_filtered/seq_len_{n}/test"


def att_metrics(model, recs, S, want_infl=False):
    embed, layers, norm, head, rotary = model_parts(model)
    ids, _, ans, S = pad_batch(recs, 0, DEV)
    parts = [rel_mask_parts(r, S, DEV) for r in recs]
    bases = torch.stack([p[0] for p in parts]).unsqueeze(1).to(torch.bfloat16)
    B = len(recs)
    h = embed(ids)
    pos = torch.arange(S, device=DEV).unsqueeze(0).expand(B, -1)
    cos, sin = rotary(h, pos)
    pe = (cos.to(h.dtype), sin.to(h.dtype))
    h_mid = None
    ctx = torch.enable_grad() if want_infl else torch.no_grad()
    with ctx:
        for li in range(len(layers)):
            if li == L_SEL:
                h = h.detach().requires_grad_(want_infl)
                h_mid = h
            with sdpa_kernel(SDPA_BACKENDS):
                out = layers[li](h, attention_mask=bases, position_ids=pos,
                                 position_embeddings=pe)
            h = out[0] if isinstance(out, tuple) else out
        rows = norm(h[torch.arange(B, device=DEV), ans])
        logits = head(rows).float()
    # pre/post-softmax answer-row attention at L_SEL, manual q/k
    lay = layers[L_SEL]
    hn = lay.input_layernorm(h_mid.detach())
    at = lay.self_attn
    hd = pe[0].shape[-1]
    nh = at.q_proj.weight.shape[0] // hd
    q = at.q_proj(hn).view(B, S, nh, hd).transpose(1, 2)
    k = at.k_proj(hn).view(B, S, -1, hd).transpose(1, 2)

    def rope(x):
        x1, x2 = x[..., : hd // 2], x[..., hd // 2:]
        return x * pe[0].unsqueeze(1) + torch.cat([-x2, x1], -1) * pe[1].unsqueeze(1)

    q, k = rope(q), rope(k)
    k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    ans_v = torch.tensor([r["seq"] - 1 for r in recs], device=DEV)
    qa = q[torch.arange(B, device=DEV), :, ans_v]
    sc = (qa.unsqueeze(2) * k).sum(-1).float() / hd ** 0.5   # [B,H,S]
    w = torch.softmax(sc, dim=-1)
    mass, keff, gap = [], [], []
    for b, r in enumerate(recs):
        rel = [i for i, c in enumerate(r["frame_chars"]) if c == r["q_char"]]
        jnk = [i for i, c in enumerate(r["frame_chars"]) if c != r["q_char"]]
        cols = lambda idxs: [t for i in idxs for t in range(*r["blocks"][i])]
        rc, jc = cols(rel), cols(jnk)
        fc = rc + jc
        wm = w[b, :, :].mean(0)
        mass.append(float(wm[rc].sum() / max(float(wm[fc].sum()), 1e-9)))
        a = wm[fc] / max(float(wm[fc].sum()), 1e-9)
        keff.append(float(1.0 / max(float((a ** 2).sum()), 1e-9)))
        sm = sc[b].mean(0)
        gap.append(float(sm[rc].mean() - sm[jc].mean()))
    infl = None
    if want_infl:
        digit = logits.argmax(-1)  # proxy; use gold digit ids passed via recs
        gold = torch.tensor([r["_gold_tok"] for r in recs], device=DEV)
        logits[torch.arange(B, device=DEV), gold].sum().backward()
        g = h_mid.grad
        shares = []
        for b, r in enumerate(recs):
            nr = [i for i, c in enumerate(r["frame_chars"]) if c == r["q_char"]]
            tot = rl = 0.0
            for i, (a0, b0) in enumerate(r["blocks"]):
                v = float(g[b, a0:b0].norm())
                tot += v
                if i in nr:
                    rl += v
            shares.append(rl / max(tot, 1e-9))
        infl = sum(shares) / len(shares)
    return logits, sum(mass) / B, sum(keff) / B, sum(gap) / B, infl


def main():
    tok, model = load_text_model("Qwen/Qwen2.5-1.5B-Instruct", device=DEV)
    digit_ids = [tok(str(d), add_special_tokens=False).input_ids[0]
                 for d in range(10)]
    models = {"frozen": None,
              "ft16": sorted(Path("outputs/cap16/a2mix").glob(
                  "armA2mix_seed0_*/coformer_best.pt"))[-1],
              "ft32": sorted(Path("outputs/coformer").glob(
                  "armA2mix_seed0_*/coformer_best.pt"))[-1],
              "ft1024": sorted(Path("outputs/ft1024").glob(
                  "armA2mix_seed1_*/coformer_best.pt"))[-1]}
    out = []
    for name, ckpt in models.items():
        lora = None
        if ckpt is not None:
            ck = torch.load(ckpt, map_location="cpu")
            lora = attach_lora(model.model.layers, 0, rank=ck["rank"],
                               alpha=ck["alpha"], device=DEV, state=ck["lora"])
            for p in lora.parameters():
                p.requires_grad_(False)
        for n in NS:
            recs = prep_root_rel(tok, Path(ROOT.format(n=n)),
                                 48 if n <= 128 else (32 if n <= 256 else 16),
                                 seed=0)
            for r in recs:
                r["_gold_tok"] = digit_ids[r["gold"]]
            hits = tot_m = tot_k = tot_g = 0.0
            bs = 8 if n <= 64 else (4 if n <= 256 else 2)
            for i in range(0, len(recs), bs):
                ch = recs[i:i + bs]
                lg, m, ke, gp, _ = att_metrics(model, ch, None)
                for b, r in enumerate(ch):
                    d = tok.decode([int(lg[b].argmax())]).strip()
                    hits += int(d.isdigit() and int(d) == r["gold"])
                tot_m += m * len(ch); tot_k += ke * len(ch); tot_g += gp * len(ch)
            _, _, _, _, infl = att_metrics(model, recs[: (8 if n <= 256 else 3)], None, want_infl=True)
            row = {"model": name, "N": n, "em1": hits / len(recs),
                   "mass": tot_m / len(recs), "keff": tot_k / len(recs),
                   "gap": tot_g / len(recs), "infl": infl}
            out.append(row)
            print(json.dumps(row), flush=True)
        if lora is not None:
            lora.remove()
    Path("outputs/_scratch/ft_effects_fixed.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Over-squashing probe at the READ NODE (the answer row) of a frozen decoder.

The answer token is a node aggregating every context token with normalised weights
(sum alpha = 1); over-squashing = its effective degree grows with the context while the
evidence's share of weight and of sensitivity vanishes. Metrics (all over the CONTEXT-
SENTENCE columns only, so the BOS/prefix sink and the question are excluded), head-mean:

  mass       post-softmax weight share on evidence tokens
  keff_tok   effective fan-in 1/sum(alpha^2) over tokens
  keff_sent  effective fan-in over sentences (weights summed per sentence)
  gap        PRE-softmax margin: mean evidence score - mean junk score
  pred_mass  the dispersion-law prediction  kF e^gap / (kF e^gap + NT - kF)
             (kF/NT = evidence/all context tokens) -- a NO-free-parameter prediction of
             `mass` from `gap`; the law fails where junk scores are heavy-tailed
  m_eff      exact per-item margin implied by the mass: logit(mass) - log(kF/(NT-kF))
  lse_e/lse_j/smax_e  raw logsumexp of the evidence / junk scores and the top evidence
             score (head-mean): logit(mass) = lse_e - lse_j; tracked across N they split the
             gap's growth into "evidence score erodes" vs "junk LSE rises"
  infl      Jacobian evidence-influence share: ||d gold-logit / d h_L(token)|| summed on
             evidence tokens / summed on all context tokens (the transformer instance of
             the GNN sensitivity measure)
  keff_jac   effective fan-in of the Jacobian norms over sentences

Same definitions as scripts/condmask/probe_ft_effects.py (MMRED, post-Add.12b: the TRUE
answer row), generalised to arbitrary position ids so the no-repack arm is probed at its
gapped positions.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.checkpoint import checkpoint

from modeling import model_parts

PROBED = ("full", "oracle", "attn", "attn_norp", "oracle_gap", "oracle_pgap")


def rope(x, cos, sin):
    hd = x.shape[-1]
    x1, x2 = x[..., : hd // 2], x[..., hd // 2:]
    return x * cos.unsqueeze(1) + torch.cat([-x2, x1], -1) * sin.unsqueeze(1)


def _blk(blocks: Sequence[Tuple[int, int]]) -> Tuple[int, int, torch.Tensor]:
    """blocks tile [a0, b_last) contiguously (build/sub_rec guarantee) -> (a0, b_last, block id per column)."""
    a0, b1 = blocks[0][0], blocks[-1][1]
    assert all(blocks[j][1] == blocks[j + 1][0] for j in range(len(blocks) - 1)), "blocks must tile"
    return a0, b1, torch.cat([torch.full((b - a,), j, dtype=torch.long) for j, (a, b) in enumerate(blocks)])


def osq_from_scores(sc: torch.Tensor, blocks: Sequence[Tuple[int, int]], facts: Sequence[int]
                    ) -> Dict[str, Optional[float]]:
    """sc: [H, S] pre-softmax scores of ONE query row over all S keys (the row is the
    last token, so causality is vacuous). Head-mean metrics over the context columns."""
    a0, b1, blk = _blk(blocks)
    sc = sc.float()
    fm = torch.zeros(b1 - a0, dtype=torch.bool)
    for j in facts:
        fm[blocks[j][0] - a0: blocks[j][1] - a0] = True
    # context-restricted softmax == full softmax renormalised over the context columns, but
    # immune to the underflow-to-0/0 that a 40k-token gap between tail and context produces
    wb = torch.softmax(sc[:, a0:b1], -1)                               # [H, NT]
    ws = torch.zeros(wb.shape[0], len(blocks)).index_add_(1, blk, wb)  # [H, F]
    kF, NT = int(fm.sum()), int(fm.numel())
    out: Dict[str, Optional[float]] = {
        "ctx_share": float(torch.softmax(sc, -1)[:, a0:b1].sum(-1).mean()),  # weight the row gives the context at all
        "mass": float(wb[:, fm].sum(-1).mean()), "mass_maxh": float(wb[:, fm].sum(-1).max()),
        "keff_tok": float((1 / (wb ** 2).sum(-1)).mean()),
        "keff_sent": float((1 / (ws ** 2).sum(-1)).mean()),
        "n_tok": NT, "n_fact_tok": kF, "gap": None, "pred_mass": None, "m_eff": None}
    if 0 < kF < NT:
        sb = sc[:, a0:b1]
        gap = sb[:, fm].mean(-1) - sb[:, ~fm].mean(-1)                # [H]
        pred = kF * gap.exp() / (kF * gap.exp() + NT - kF)
        m = wb[:, fm].sum(-1).clamp(1e-6, 1 - 1e-6)
        out.update(gap=float(gap.mean()), pred_mass=float(pred.mean()),
                   m_eff=float((torch.log(m / (1 - m)) - math.log(kF / (NT - kF))).mean()),
                   # raw (un-shifted) sides of the gap, so growth with N splits into the
                   # evidence side falling vs the junk side rising (Gollapudi et al. 2026, Tab. 9)
                   lse_e=float(sb[:, fm].logsumexp(-1).mean()), lse_j=float(sb[:, ~fm].logsumexp(-1).mean()),
                   smax_e=float(sb[:, fm].amax(-1).mean()))
    return out


def jac_share(gn: torch.Tensor, blocks: Sequence[Tuple[int, int]], facts: Sequence[int]
              ) -> Dict[str, float]:
    """gn: [S] per-token gradient norms -> evidence share and sentence-level fan-in."""
    a0, b1, blk = _blk(blocks)
    g = gn.float()[a0:b1]
    gs = torch.zeros(len(blocks)).index_add_(0, blk, g)
    tot = float(gs.sum()) or 1e-9
    p = gs / tot
    return {"infl": float(sum(gs[j] for j in facts)) / tot,
            "keff_jac": float(1 / (p ** 2).sum())}


def _call(layer, h, pos, pe):
    o = layer(h, attention_mask=None, position_ids=pos, position_embeddings=pe)
    return o[0] if isinstance(o, tuple) else o


def osq_probe(model, ids: List[int], pos: List[int], blocks, facts, layer: int,
              gold: Optional[int], dev, jac: bool = True, ckpt: bool = True,
              extra: Optional[dict] = None) -> Dict[str, Optional[float]]:
    """Attention metrics of the last row at `layer` (+ Jacobian share of the gold logit
    w.r.t. that layer's input, layers above run under per-layer checkpointing so a 128k
    backward fits). `pos` = the arm's position ids (gapped for no-repack).
    `extra`, if given, receives the raw tensors (needles.py): sc [H,S] scores, h_mid [S,d]
    the probed layer's input, and with jac: gn [S] grad norms, h_fin [d] the final normed
    last-row state, logits [V]."""
    embed, layers, norm, head, rotary = model_parts(model)
    x = torch.tensor([ids], device=dev)
    p = torch.tensor([pos], device=dev)
    with torch.no_grad():
        h = embed(x)
        c, s = rotary(h, p)
        pe = (c.to(h.dtype), s.to(h.dtype))
        for li in range(layer):
            h = _call(layers[li], h, p, pe)
        lay, S = layers[layer], h.shape[1]
        hn = lay.input_layernorm(h)
        at = lay.self_attn
        hd = pe[0].shape[-1]
        nh = at.q_proj.weight.shape[0] // hd
        q = rope(at.q_proj(hn[:, -1:]).view(1, 1, nh, hd).transpose(1, 2), pe[0][:, -1:], pe[1][:, -1:])
        k = rope(at.k_proj(hn).view(1, S, -1, hd).transpose(1, 2), *pe)
        k = k.repeat_interleave(nh // k.shape[1], dim=1)
        sc = (q @ k.transpose(-1, -2))[0, :, 0] / hd ** 0.5           # [H, S]
        del q, k
    out = osq_from_scores(sc.cpu(), blocks, facts)
    if extra is not None:
        extra.update(sc=sc.detach(), h_mid=h[0].detach())
    if not jac or gold is None:
        return out
    h_mid = h.detach().requires_grad_(True)
    with torch.enable_grad():
        hh = h_mid
        for li in range(layer, len(layers)):
            hh = (checkpoint(_call, layers[li], hh, p, pe, use_reentrant=False) if ckpt
                  else _call(layers[li], hh, p, pe))
        h_fin = norm(hh[:, -1])
        logits = head(h_fin).float()[0]
        logits[gold].backward()
    gn = h_mid.grad[0].float().norm(dim=-1)
    out.update(jac_share(gn.cpu(), blocks, facts))
    out["pred_tok"] = int(logits.argmax())
    out["gold_tok"] = int(gold)
    if extra is not None:
        extra.update(gn=gn.detach(), h_fin=h_fin[0].detach(), logits=logits.detach())
    return out

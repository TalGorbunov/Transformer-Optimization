"""Relational per-frame gates: g[b, l, h] decided from (question x frame_b) features.

Semantics: g[b,l,h] = 1 => frame b is CUT OUT of the attention graph at (l,h) — its
tokens read only themselves + the prefix, AND nothing outside (other frames, the tail)
reads into it. g = 0 => fully connected. Which frames deserve cutting depends on the
question (distractor frames are about other characters), so NO static mask is optimal
— this is the setting where input-conditioning is necessary, unlike condmask v1/v2
(cross-input variance 0.000; see CONDMASK_REPORT.md).

Gate field per (l,h): penalty(q,k) = K * OR(g[frame(k)], g[frame(q)]) on region cells,
region = cross-frame cells (both directions) + tail->frame cells; prefix columns and
within-frame cells are never touched. OR(a,b) = a + b - a*b (exact for hard 0/1 gates,
differentiable for ST-relaxed ones).

RelGateNet input per frame: concat[LN(q_feat), LN(f_b), LN(q_feat)*LN(f_b)] — the
elementwise product term is the explicit relational (comparison) channel a global
mean-pool cannot express. Final layer zero-init + bias-init (same anti-S0 rules).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gnnformer.constants import MASK_MIN
from textdata import build_segmented_prompt, load_text_sample  # noqa: E402

Q_RE = re.compile(r"How many frames show (\w+) in the (\w+)\?")
Q_ENTITY = re.compile(r"How many frames show (\w+)\?")
Q_DISTINCT = re.compile(r"How many distinct rooms did (\w+) appear in")
Q_COOC = re.compile(r"In how many frames are (\w+) and (\w+) together\?")


def parse_relevance(question, states):
    """-> (rel [list of bool per frame], q_char) or None. Relevance is
    question-derived: steps/entity/distinct -> C present; cooc -> BOTH present."""
    import ast as _a
    occs = []
    for s_ in states:
        st = _a.literal_eval(s_)
        occs.append({c for cs in st["rooms"].values() for c in cs})
    m = Q_RE.match(question) or Q_ENTITY.match(question) or Q_DISTINCT.match(question)
    if m:
        C = m.group(1)
        return [C in o for o in occs], C
    m = Q_COOC.match(question)
    if m:
        C, D = m.group(1), m.group(2)
        return [(C in o and D in o) for o in occs], C
    return None


def prep_root_rel(tok: Any, root: Path, limit: int, seed: int = 0):
    """prep_root + per-frame character list and parsed (q_char, q_room) for the oracle
    and the g_dist/g_rel telemetry."""
    import ast
    import random as _r
    dirs = [p for p in sorted(Path(root).iterdir())
            if p.is_dir() and (p / "qa.txt").exists()]
    _r.Random(seed).shuffle(dirs)
    out = []
    for sd in dirs:
        if len(out) >= limit:
            break
        loaded = load_text_sample(sd)
        if loaded is None:
            continue
        sid, states, question, gold = loaded
        pr = parse_relevance(question, states)
        if pr is None:
            continue
        rec = build_segmented_prompt(tok, states, question, gold, sid)
        if rec is None:
            continue
        rec["rel"] = pr[0]
        rec["q_char"] = pr[1]
        # back-compat for single-char callers: frame_chars[i] == q_char iff relevant
        rec["frame_chars"] = [pr[1] if r else "OTHER" for r in pr[0]]
        rec["q_room"] = ""
        out.append(rec)
    return out


def oracle_gates(rec: Dict[str, Any], n_layers: int, n_heads: int,
                 device: Any) -> torch.Tensor:
    """[F, L, H]: isolate exactly the irrelevant frames (rec['rel'] precomputed)."""
    rel = rec.get("rel") or [c == rec["q_char"] for c in rec["frame_chars"]]
    d = torch.tensor([0.0 if r else 1.0 for r in rel], device=device)
    return d.view(-1, 1, 1).expand(-1, n_layers, n_heads)


class RelGateNet(nn.Module):
    """(q_feat [B,d], f_feats [B,F,d]) -> logits [B, F, L, H]."""

    def __init__(self, d_model: int, n_layers: int, n_heads: int, hidden: int = 256,
                 init: str = "random", init_bias: float = 1.0, seed: int = 0):
        super().__init__()
        self.n_layers, self.n_heads = n_layers, n_heads
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_f = nn.LayerNorm(d_model)
        self.net = nn.Sequential(nn.Linear(3 * d_model, hidden), nn.GELU(),
                                 nn.Linear(hidden, n_layers * n_heads))
        last = self.net[-1]
        with torch.no_grad():
            last.weight.zero_()
            if init == "blockwise":
                last.bias.fill_(init_bias)
            elif init == "open":
                last.bias.fill_(-init_bias)
            else:
                g = torch.Generator().manual_seed(seed)
                last.bias.copy_((torch.rand(last.bias.shape, generator=g) * 2 - 1)
                                * init_bias)
        self.register_buffer("init_logits",
                             last.bias.detach().clone().view(n_layers, n_heads))

    def forward(self, q_feat: torch.Tensor, f_feats: torch.Tensor) -> torch.Tensor:
        B, F, d = f_feats.shape
        q = self.ln_q(q_feat.float()).unsqueeze(1).expand(-1, F, -1)
        f = self.ln_f(f_feats.float())
        z = torch.cat([q, f, q * f], dim=-1)
        return self.net(z).view(B, F, self.n_layers, self.n_heads)


def frame_feats(emb: torch.Tensor, recs: List[Dict[str, Any]], S: int,
                detach: bool = True
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """-> (q_feat [B,d], f_feats [B,F_max,d], f_valid [B,F_max]); mean-pooled layer-0
    embeddings per prefix / per frame block. Detached (gates train via their own MLP)."""
    B, _, D = emb.shape
    F = max(len(r["blocks"]) for r in recs)
    q = torch.zeros(B, D, device=emb.device)
    f = torch.zeros(B, F, D, device=emb.device)
    v = torch.zeros(B, F, dtype=torch.bool, device=emb.device)
    for b, r in enumerate(recs):
        q[b] = emb[b, : r["prefix_end"]].float().mean(0)
        for i, (a, e) in enumerate(r["blocks"]):
            f[b, i] = emb[b, a:e].float().mean(0)
            v[b, i] = True
    return (q.detach(), f.detach(), v) if detach else (q, f, v)


def rel_mask_parts(rec: Dict[str, Any], S: int, device: Any
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """-> (base [S,S] fp32, region [S,S] bool, col_frame [S], row_frame [S]).
    base: plain causal + padding (reused semantics from masks.layout_mask_parts).
    region: cross-frame cells (i != j, both directions) + tail->frame cells; prefix
    columns and within-frame cells excluded. col/row_frame: frame id per position
    (-1 outside frames) for gathering per-frame gates."""
    from masks import layout_mask_parts
    base, _ = layout_mask_parts(rec, S, device)   # base only; old template unused
    fr = torch.full((S,), -1, dtype=torch.long, device=device)
    for i, (a, b) in enumerate(rec["blocks"]):
        fr[a:b] = i
    pos = torch.arange(S, device=device)
    qf, kf = fr.view(-1, 1), fr.view(1, -1)
    causal = pos.view(1, -1) <= pos.view(-1, 1)
    cross = (qf >= 0) & (kf >= 0) & (qf != kf)
    tail_read = (pos.view(-1, 1) >= rec["fin"]) & (kf >= 0)
    region = (cross | tail_read) & causal
    return base, region, fr, fr


def assemble_rel_masks(base: torch.Tensor, region: torch.Tensor,
                       col_frame: torch.Tensor, g_f: torch.Tensor, K: float,
                       dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """One sample, one layer: g_f [F, H] -> [H, S, S] additive mask.
    penalty = K * OR(g[frame(k)], g[frame(q)]) on region cells."""
    Fn, H = g_f.shape
    gpad = torch.cat([g_f, g_f.new_zeros(1, H)])          # frame -1 -> gate 0
    idx = col_frame.clamp(min=-1) % (Fn + 1)              # -1 -> last row (zeros)
    gc = gpad[idx].T                                      # [H, S] gate of each column
    gr = gc                                               # rows use the same frame map
    orf = gc.unsqueeze(1) + gr.unsqueeze(2) - gc.unsqueeze(1) * gr.unsqueeze(2)
    m = base.unsqueeze(0) + K * orf * region.unsqueeze(0).float()
    return m.to(dtype)


class RelGateTF(nn.Module):
    """Expressive relational gate: a small transformer over [q, f_1..f_F] summaries,
    so a frame's cut/keep decision can depend on the question AND the other frames
    (relevance is relational). Per-frame output head zero-init + bias-init (anti-S0).
    -> logits [B, F, L, H]."""

    def __init__(self, d_model: int, n_layers: int, n_heads: int, d_g: int = 256,
                 depth: int = 2, init: str = "random", init_bias: float = 1.0,
                 seed: int = 0):
        super().__init__()
        self.n_layers, self.n_heads = n_layers, n_heads
        self.proj = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_g))
        enc = nn.TransformerEncoderLayer(d_g, nhead=4, dim_feedforward=2 * d_g,
                                         batch_first=True, norm_first=True,
                                         dropout=0.0)
        self.tf = nn.TransformerEncoder(enc, num_layers=depth)
        self.head = nn.Linear(d_g, n_layers * n_heads)
        with torch.no_grad():
            self.head.weight.zero_()
            if init == "blockwise":
                self.head.bias.fill_(init_bias)
            elif init == "open":
                self.head.bias.fill_(-init_bias)
            else:
                g = torch.Generator().manual_seed(seed)
                self.head.bias.copy_((torch.rand(self.head.bias.shape, generator=g)
                                      * 2 - 1) * init_bias)
        self.register_buffer("init_logits",
                             self.head.bias.detach().clone().view(n_layers, n_heads))

    def forward(self, q_feat: torch.Tensor, f_feats: torch.Tensor,
                f_valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, F, _ = f_feats.shape
        toks = torch.cat([self.proj(q_feat.float()).unsqueeze(1),
                          self.proj(f_feats.float())], dim=1)      # [B, 1+F, d_g]
        pad = None
        if f_valid is not None:
            pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool,
                                         device=f_valid.device), ~f_valid], dim=1)
        h = self.tf(toks, src_key_padding_mask=pad)
        return self.head(h[:, 1:]).view(B, F, self.n_layers, self.n_heads)


def early_hidden(model, ids: torch.Tensor, bases: torch.Tensor, k: int,
                 grad: bool = False) -> torch.Tensor:
    """Hidden states after the FROZEN model's first k layers under plain causal
    attention — contextualized conditioning features (each frame summary has read the
    question prefix and earlier frames). Detached; k=0 -> raw embeddings."""
    from modeling import SDPA_BACKENDS, model_parts, sdpa_kernel
    with torch.set_grad_enabled(grad and torch.is_grad_enabled()):
        return _early_hidden(model, ids, bases, k)


def _early_hidden(model, ids, bases, k):
    from modeling import SDPA_BACKENDS, model_parts, sdpa_kernel
    embed, layers, _, _, rotary = model_parts(model)
    h = embed(ids)
    if k <= 0:
        return h
    B, S = ids.shape
    pos = torch.arange(S, device=ids.device).unsqueeze(0).expand(B, -1)
    cos, sin = rotary(h, pos)
    pe = (cos.to(h.dtype), sin.to(h.dtype))
    m = bases.unsqueeze(1).to(h.dtype)
    for layer in layers[:k]:
        with sdpa_kernel(SDPA_BACKENDS):
            out = layer(h, attention_mask=m, position_ids=pos, position_embeddings=pe)
        h = out[0] if isinstance(out, tuple) else out
    return h


class DynGate(nn.Module):
    """Layerwise dynamic gates — the maximal member of the mask family:
    mask_l = f(h^{l-1}), per (token, head), at EVERY layer. One shared linear head
    (weights tied across layers, GNN-style) + a per-layer bias table, LayerNorm input,
    zero-init weights so training starts at exactly the init regime ('open' = full
    attention). Gradients flow through h^{l-1}: the model learns to encode its own
    routing decisions."""

    def __init__(self, d_model: int, n_layers: int, n_heads: int,
                 init: str = "open", init_bias: float = 1.0, seed: int = 0):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.lin = nn.Linear(d_model, n_heads)
        with torch.no_grad():
            self.lin.weight.zero_()
            self.lin.bias.zero_()
        if init == "blockwise":
            b = torch.full((n_layers, n_heads), init_bias)
        elif init == "open":
            b = torch.full((n_layers, n_heads), -init_bias)
        else:
            g = torch.Generator().manual_seed(seed)
            b = (torch.rand(n_layers, n_heads, generator=g) * 2 - 1) * init_bias
        self.layer_bias = nn.Parameter(b)
        self.register_buffer("init_logits", b.detach().clone())

    def forward(self, h: torch.Tensor, li: int) -> torch.Tensor:
        """h [B,S,d] (previous layer's output) -> cut logits [B, S, H] for layer li."""
        return self.lin(self.ln(h.float())) + self.layer_bias[li]


def assemble_dyn_masks(bases: torch.Tensor, regions: torch.Tensor,
                       in_frame: torch.Tensor, g: torch.Tensor, K: float,
                       dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Token-level OR-field assembly for one layer.
    bases [B,S,S] fp32 · regions [B,S,S] bool · in_frame [B,S] bool ·
    g [B,S,H] in [0,1] -> [B,H,S,S]. Gates apply only to frame tokens; penalty =
    K * OR(g[col], g[row]) on region cells."""
    gm = (g * in_frame.unsqueeze(-1).float()).permute(0, 2, 1)     # [B,H,S]
    orf = gm.unsqueeze(2) + gm.unsqueeze(3) - gm.unsqueeze(2) * gm.unsqueeze(3)
    m = bases.unsqueeze(1) + K * orf * regions.unsqueeze(1).float()
    return m.to(dtype)


def prep_root_crowded(tok, root, limit, seed=0):
    """Crowded frames: per-clause tokenization inside each frame block.
    rec["clauses"][f] = [(a, b, mentions_C, is_evidence)] — token labels for the
    token-level selection aux. Frame-level rel is all-True (nothing droppable)."""
    import ast as _a
    import random as _r
    dirs = [p_ for p_ in sorted(Path(root).iterdir())
            if p_.is_dir() and (p_ / "qa.txt").exists()]
    _r.Random(seed).shuffle(dirs)
    out = []
    for sd in dirs:
        if len(out) >= limit:
            break
        loaded = load_text_sample(sd)
        if loaded is None:
            continue
        sid, states, question, gold = loaded
        m = Q_RE.match(question)
        if m is None:
            continue
        C, R = m.group(1), m.group(2)
        prefix_txt = (f"You will be shown {len(states)} frames describing steps in a "
                      f"house.\nQuestion: {question}")
        ids = list(tok(prefix_txt, add_special_tokens=False).input_ids)
        prefix_end = len(ids)
        blocks, clauses = [], []
        ok = True
        for k, s_ in enumerate(states):
            st = _a.literal_eval(s_)
            head = tok(f"\nFrame {st['step_id']}:", add_special_tokens=False).input_ids
            a0 = len(ids)
            ids += head
            cl = []
            for room, chs in st["rooms"].items():
                names = (chs[0] if len(chs) == 1
                         else ", ".join(chs[:-1]) + f" and {chs[-1]}")
                verb = "is" if len(chs) == 1 else "are"
                seg = tok(f" {names} {verb} in the {room}.",
                          add_special_tokens=False).input_ids
                ca = len(ids)
                ids += seg
                cl.append((ca, len(ids), C in chs, (C in chs) and room == R))
            if len(ids) == a0 + len(head):
                ok = False
                break
            blocks.append((a0, len(ids)))
            clauses.append(cl)
        if not ok:
            continue
        fin = len(ids)
        ids += tok(f"\nQuestion: {question}\nAnswer: ",
                   add_special_tokens=False).input_ids
        out.append({"ids": ids, "prefix_end": prefix_end, "blocks": blocks,
                    "fin": fin, "seq": len(ids), "gold": gold, "sid": sid,
                    "clauses": clauses, "rel": [True] * len(blocks),
                    "q_char": C, "q_room": R,
                    "frame_chars": [C] * len(blocks)})
    return out


def clause_token_labels(rec, S, device, mode="evid"):
    """[S] float: 1 on tokens of evidence clauses (mode='evid': C AND R) or of any
    C-mentioning clause (mode='entity'). 0 elsewhere (prefix/tail/others)."""
    lab = torch.zeros(S, device=device)
    for cl in rec["clauses"]:
        for (a, b, isC, isE) in cl:
            if (isE if mode == "evid" else isC):
                lab[a:b] = 1.0
    return lab

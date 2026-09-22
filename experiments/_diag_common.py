"""Shared pieces of the DIAG instruments (probe_hahn.py, probe_attention.py, evaluate.py knobs).

Everything here is eval-only and model-agnostic apart from the Qwen2.5-VL attention geometry
(28 heads, 4 KV heads, head_dim 128, M-RoPE section [16, 24, 24]) which is read from the config.

  * answer vocabularies + first-token margins per qtype (digits / rooms / people);
  * one-frame flip pairs for count qtypes (k -> k+1, the legacy protocol) and for needle qtypes
    (char_at_frame: move the asked character at the asked step; n_char_at_frame: change the
    number of companions at the asked step) with an answer-preserving control edit in another frame;
  * QKCapture: forward hooks on q_proj / k_proj (post-PEFT-wrap, so LoRA'd projections are seen)
    and on the rotary (cos, sin) the text model hands every decoder layer; row_attention()
    recomputes one query row's softmax over all keys exactly as Qwen2_5_VLAttention does
    (scores = q.k * module.scaling + additive mask) — sdpa never exposes weights;
  * temperature (tau) and log-N logit scaling knobs: both act through module.scaling, so the
    recomputed photograph and the model's own forward are the same number by construction.

Twins: legacy/v1/scripts/sparse/probe_attn_photo.py (photograph), probe_hahn_gated.py (pairs,
--attn-logn-sref), the S10b --attn-sharpen hook. Layer numbers for scaling knobs are DECODER
MODULE indices (0-based); hidden-state layer numbers follow the legacy convention (see probe_hahn).
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from core.constants import CHARS, NOBODY, ROOMS
from core.mmred import (NUMERIC_QTYPES, evidence_frames, parse_question, recompute_answer, states,
                        with_char_moved)
from core.model import text_config

ANSWER_PREFIX = '{ "answer": "'      # teacher-forced assistant prefix; the next token is the answer's first token

ROOM_QTYPES = {"first_app", "final_app", "char_on_char_first_app", "char_on_char_final_app", "char_at_frame",
               "room_empty", "where_spend", "crowded_room"}
PERSON_QTYPES = {"first_at_room", "last_at_room", "room_on_char_first_app", "room_on_char_final_app", "room_at_frame",
                 "char_on_char_at_frame", "who_spend", "spend_alone", "spend_together"}
NEEDLE_FLIP_QTYPES = ("char_at_frame", "n_char_at_frame")
COUNT_FLIP_QTYPES = ("steps_in_room",)
FLIP_QTYPES = COUNT_FLIP_QTYPES + NEEDLE_FLIP_QTYPES


# ------------------------------------------------------------------ answer vocabulary + margins

def answer_vocab(qtype: str, tok: Any) -> Tuple[List[str], List[int]]:
    """(labels, first-token ids). Digits are single tokens; the six rooms and the six people
    have pairwise-distinct first tokens (checked 2026-09-22), so a first-token margin identifies
    the answer. Numeric golds >= 10 have no single-token id and get no margin row."""
    if qtype in NUMERIC_QTYPES:
        labels = [str(d) for d in range(10)]
    elif qtype in ROOM_QTYPES:
        labels = list(ROOMS)
    elif qtype in PERSON_QTYPES:
        labels = list(CHARS) + [NOBODY]
    else:
        raise ValueError(qtype)
    ids = [int(tok(lbl, add_special_tokens=False).input_ids[0]) for lbl in labels]
    assert len(set(ids)) == len(ids), f"first tokens collide for {qtype}: {list(zip(labels, ids))}"
    return labels, ids


def gold_index(qtype: str, gold: str, labels: Sequence[str]) -> Optional[int]:
    g = str(gold).strip()
    if qtype in NUMERIC_QTYPES:
        return int(g) if g.isdigit() and int(g) <= 9 else None
    return labels.index(g) if g in labels else None


def margin_of(vocab_logits: np.ndarray, gold_idx: int) -> Tuple[float, int]:
    """logit(gold) - max logit over the other labels; argmax index."""
    other = np.delete(vocab_logits, gold_idx)
    return float(vocab_logits[gold_idx] - other.max()), int(np.argmax(vocab_logits))


# ------------------------------------------------------------------ pairs

def _char_room(st: Dict[str, Any], char: str) -> Optional[str]:
    for r, occ in st["rooms"].items():
        if char in occ:
            return r
    return None


def _count(sts: Sequence[Dict[str, Any]], char: str, room: str) -> int:
    return sum(1 for s in sts if char in s["rooms"].get(room, []))


def build_count_pair(row: Dict[str, Any], rng: random.Random) -> Optional[Dict[str, Any]]:
    """steps_in_room: base gold k; flip member moves the asked character INTO the asked room in
    one non-evidence frame (k+1); control moves it to a third room in the same frame (k).
    Twin: experiments/probe_hahn.py::build_pair (2026-09-21), == legacy probe_hahn_gated.build_pair."""
    qtype, question = row["qtype"], row["question"]
    st = states(row)
    gold = int(row["answer"])
    char, room = parse_question(qtype, question)
    evid = evidence_frames(qtype, question, st)
    if evid is None or len(evid) != gold or recompute_answer(qtype, question, st) != str(gold):
        raise AssertionError(f"evidence/gold mismatch in {row['qid']}")
    cands = [t for t in range(len(st)) if t not in evid]
    if not cands:
        return None
    flip_t = int(rng.choice(cands))
    cur = _char_room(st[flip_t], char)
    ctrl_room = str(rng.choice([r for r in ROOMS if r not in (cur, room)]))
    flip_states = list(st)
    flip_states[flip_t] = with_char_moved(st[flip_t], char, room)
    ctrl_states = list(st)
    ctrl_states[flip_t] = with_char_moved(st[flip_t], char, ctrl_room)
    assert _count(st, char, room) == gold and _count(flip_states, char, room) == gold + 1 \
        and _count(ctrl_states, char, room) == gold, row["qid"]
    return dict(qid=row["qid"], qtype=qtype, question=question, gold=str(gold), flip_gold=str(gold + 1),
                flip_t=flip_t, ctrl_t=flip_t, flip_kind="evid", evid=set(evid),
                base_states=st, flip_states=flip_states, ctrl_states=ctrl_states)


def build_needle_pair(row: Dict[str, Any], rng: random.Random) -> Optional[Dict[str, Any]]:
    """Needle qtypes (k = 1, the asked step X). Flip member edits frame X so the answer changes;
    control member makes a comparable edit in a different frame so the answer is preserved.
    char_at_frame: move the asked character to another room at X (answer = the new room).
    n_char_at_frame: move one other character into (or, if all are there, out of) the asked
    character's room at X (answer +/- 1). Both members differ from base in exactly one frame."""
    qtype, question = row["qtype"], row["question"]
    st = states(row)
    gold = str(row["answer"])
    if recompute_answer(qtype, question, st) != gold:
        raise AssertionError(f"gold mismatch in {row['qid']}")
    char, step = parse_question(qtype, question)[:2]
    X = int(step) - 1
    N = len(st)
    if not (0 <= X < N):
        return None
    base_room = _char_room(st[X], char)
    if base_room is None:
        return None
    flip_states = list(st)
    if qtype == "char_at_frame":
        new_room = str(rng.choice([r for r in ROOMS if r != base_room]))
        flip_states[X] = with_char_moved(st[X], char, new_room)
    elif qtype == "n_char_at_frame":
        others = [c for c in CHARS if c != char]
        outside = [c for c in others if _char_room(st[X], c) != base_room]
        inside = [c for c in others if _char_room(st[X], c) == base_room]
        if outside:
            c2 = str(rng.choice(outside))
            flip_states[X] = with_char_moved(st[X], c2, base_room)
        elif inside:
            c2 = str(rng.choice(inside))
            flip_states[X] = with_char_moved(st[X], c2, str(rng.choice([r for r in ROOMS if r != base_room])))
        else:
            return None
    else:
        raise ValueError(qtype)
    flip_gold = recompute_answer(qtype, question, flip_states)
    if flip_gold is None or str(flip_gold) == gold:
        return None
    # control: the same kind of edit in a frame t' != X where the asked character is visible
    cands = [t for t in range(N) if t != X and _char_room(st[t], char) is not None]
    if not cands:
        return None
    ctrl_t = int(rng.choice(cands))
    cur = _char_room(st[ctrl_t], char)
    ctrl_states = list(st)
    ctrl_states[ctrl_t] = with_char_moved(st[ctrl_t], char, str(rng.choice([r for r in ROOMS if r != cur])))
    if recompute_answer(qtype, question, ctrl_states) != gold:
        return None
    return dict(qid=row["qid"], qtype=qtype, question=question, gold=gold, flip_gold=str(flip_gold),
                flip_t=X, ctrl_t=ctrl_t, flip_kind="needle", evid={X},
                base_states=st, flip_states=flip_states, ctrl_states=ctrl_states)


def build_pair(row: Dict[str, Any], rng: random.Random, max_gold: Optional[int] = None,
               gold_set: Optional[Set[int]] = None) -> Optional[Dict[str, Any]]:
    """Dispatch by qtype; max_gold / gold_set filter count qtypes only (needle golds are labels)."""
    qtype = row["qtype"]
    if qtype in COUNT_FLIP_QTYPES:
        gold = int(row["answer"])
        if gold_set is not None:
            if gold not in gold_set:
                return None
        elif max_gold is not None and gold > max_gold:
            return None
        return build_count_pair(row, rng)
    if qtype in NEEDLE_FLIP_QTYPES:
        return build_needle_pair(row, rng)
    raise ValueError(f"no flip protocol for qtype {qtype!r}; supported: {FLIP_QTYPES}")


# ------------------------------------------------------------------ attention geometry + hooks

def attention_dims(model: Any) -> Dict[str, Any]:
    tc = text_config(model)
    n_heads = int(tc.num_attention_heads)
    return {"n_heads": n_heads, "n_kv": int(tc.num_key_value_heads), "head_dim": int(tc.hidden_size) // n_heads,
            "mrope_section": list(tc.rope_scaling["mrope_section"])}


def attn_modules(layers: Any) -> List[Any]:
    return [ly.self_attn for ly in layers]


def set_sharpen(layers: Any, tau: float, from_layer: int) -> float:
    """S10b/S0 static sharpening: module.scaling *= tau on decoder modules >= from_layer.
    Returns the base scaling. tau <= 0 or 1 = no-op (byte-identical anchor)."""
    mods = attn_modules(layers)
    base = float(mods[0].scaling)
    if tau and tau > 0 and tau != 1.0:
        for i, m in enumerate(mods):
            if i >= from_layer:
                m.scaling = base * float(tau)
    return base


def set_logn(layers: Any, seq_len: int, sref: int, base: float) -> None:
    """Legacy --attn-logn-sref: scaling = base * ln(seq_len) / ln(sref) on every decoder module,
    set per forward with the prompt's token length. sref <= 0 = off."""
    if sref and sref > 0:
        s = base * math.log(max(int(seq_len), 2)) / math.log(int(sref))
        for m in attn_modules(layers):
            m.scaling = s


class QKCapture:
    """Forward hooks that keep q_proj / k_proj outputs of chosen decoder modules and the rotary
    (cos, sin) passed to decoder module 0. Install AFTER any PEFT wrap so the LoRA'd projections
    are captured. Tensors are kept on device, detached, in the model dtype."""

    def __init__(self, layers: Any, capture_layers: Sequence[int]):
        self._layers = layers
        self._capture = list(capture_layers)
        self._handles: List[Any] = []
        self.q: Dict[int, torch.Tensor] = {}
        self.k: Dict[int, torch.Tensor] = {}
        self.cos: Optional[torch.Tensor] = None
        self.sin: Optional[torch.Tensor] = None

    def install(self) -> "QKCapture":
        def mk(store: Dict[int, torch.Tensor], L: int):
            def hook(_m, _i, out):
                store[L] = out.detach()
            return hook

        for L in self._capture:
            att = self._layers[L].self_attn
            self._handles.append(att.q_proj.register_forward_hook(mk(self.q, L)))
            self._handles.append(att.k_proj.register_forward_hook(mk(self.k, L)))

        def pe_hook(_m, hargs, hkwargs):
            pe = hkwargs.get("position_embeddings")
            if pe is None and len(hargs) >= 8:
                pe = hargs[7]
            if pe is not None:
                self.cos, self.sin = pe[0].detach(), pe[1].detach()
            return hargs, hkwargs

        self._handles.append(self._layers[0].register_forward_pre_hook(pe_hook, with_kwargs=True))
        return self

    def clear(self) -> None:
        self.q.clear(); self.k.clear(); self.cos = self.sin = None

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


def row_attention(qk: QKCapture, layers: Any, L: int, row: int, mask_row: Optional[torch.Tensor],
                  dims: Dict[str, Any]) -> torch.Tensor:
    """[H, T] softmax weights of query position `row` at decoder module L, recomputed as the
    model computes them: rotary-embedded q.k * module.scaling + additive mask row (0 / MASK_MIN).
    mask_row=None means plain causal (row is the last position, so every key is visible)."""
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb, repeat_kv

    nH, nKV, hd = dims["n_heads"], dims["n_kv"], dims["head_dim"]
    q = qk.q[L]
    k = qk.k[L]
    T = int(q.shape[1])
    qh = q.view(1, T, nH, hd).transpose(1, 2).float()
    kh = k.view(1, T, nKV, hd).transpose(1, 2).float()
    assert qk.cos is not None and qk.sin is not None, "rotary embeddings were not captured"
    qr, kr = apply_multimodal_rotary_pos_emb(qh, kh, qk.cos.float(), qk.sin.float(), dims["mrope_section"])
    kr = repeat_kv(kr, nH // nKV)[0]                     # [H, T, D]
    qrow = qr[0][:, row]                                 # [H, D]
    sc = torch.einsum("hd,htd->ht", qrow, kr) * float(layers[L].self_attn.scaling)
    if mask_row is not None:
        sc = sc + mask_row.to(sc.device, sc.dtype)
    else:
        if row + 1 < T:
            sc[:, row + 1:] = float("-inf")
    w = torch.softmax(sc, dim=-1)
    s = float(w.sum(-1).mean())
    assert abs(s - 1.0) < 1e-3, f"softmax rows sum to {s}"
    return w


def block_masses(w: torch.Tensor, blocks: Sequence[Tuple[int, int]]) -> torch.Tensor:
    """[H, n_blocks] mass per frame block."""
    return torch.stack([w[:, a:b].sum(-1) for (a, b) in blocks], dim=1)


def rank_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC with tie correction (ties count 0.5)."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), dtype=np.float64)
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


# ------------------------------------------------------------------ encoding

def encode_prompt(processor: Any, frames: Sequence[Any], question: str, layout: str, prefix_ids: Sequence[int],
                  device: Any) -> Dict[str, Any]:
    """Chat-template encode (core.prompt.build_messages) + the teacher-forced JSON prefix, so the
    last position's next-token logits are the answer's first token."""
    from core.model import move_to_device
    from core.prompt import build_messages

    msgs = build_messages(frames, question, layout=layout)
    enc = dict(processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                             return_tensors="pt"))
    pre = torch.tensor([list(prefix_ids)], dtype=enc["input_ids"].dtype)
    enc["input_ids"] = torch.cat([enc["input_ids"], pre], dim=1)
    enc["attention_mask"] = torch.ones_like(enc["input_ids"])
    return move_to_device(enc, device)


def cond_tag(tau: float, from_layer: int, sref: int) -> str:
    """Condition label used in run-dir names and CSVs: base | tau<t>[L<from>] | logn<sref>."""
    parts = []
    if tau and tau > 0 and tau != 1.0:
        parts.append(f"tau{tau:g}" + (f"L{from_layer}" if from_layer else ""))
    if sref and sref > 0:
        parts.append(f"logn{sref}")
    return "_".join(parts) if parts else "base"

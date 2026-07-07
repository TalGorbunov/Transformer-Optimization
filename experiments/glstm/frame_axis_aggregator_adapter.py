#!/usr/bin/env python3
"""Frame-axis aggregator adapter (one forward pass) on the frozen 7B.

Design (see RESULTS.md [2026-06-19]): the per-frame evidence is linearly present at L~19 (probe AUC
0.98) but the model's single-pass answer is read from the last token after a count-blind, saturating
pool. So: a forward-PRE-hook on decoder layer L_READ reads the per-frame VISION-token reps entering
that layer, an adapter aggregates them in a side channel, and injects the result back into the same
residual at the answer position -- doing extract->aggregate explicitly, in ONE forward pass. Qwen
stays frozen + 4-bit; only the adapter trains.

Two aggregators (flag --aggregator):
  seqmodel  : MAIN. order-aware -- a small TransformerEncoder over [AGG, m_1..m_N] with learned frame
              positional embeddings; read out at the AGG token. Subsumes sum/max, handles order.
  deepsets  : COMPARISON. permutation-invariant concat(sum, mean, max) of per-frame messages + linear.

Readouts (both reported): (a) frozen-LM head via residual injection (the real fix); (b) an auxiliary
9-way count head on the aggregate (diagnostic: did the aggregator capture the count). Primary metric =
bias (mean_pred - mean_gold) by seq_len/count; accuracy secondary.

SPLITS (disjoint at sample-dir level, stratified by seq_len, deterministic+seeded, manifest dumped):
  train    = 70% of seq 1-4         (random character choice on rooms/co-occ = augmentation)
  val      = 15% of seq 1-4         (per-epoch monitor; fixed per-dir character -> stable; best-epoch select)
  test_iid = 15% of seq 1-4         (clean in-distribution, no train overlap)
  test_ood = seq 5-8 (sampled)      (length + higher-count generalization)
Best epoch chosen by mean val LM-accuracy; final test uses that checkpoint.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from evaluations.scripts import eval_mmred_qwen25_vl_accuracy as base
from evaluations.scripts import eval_mmred_rooms_visited_baseline as rv
from evaluations.scripts import eval_mmred_text_frames_acc as tf
from evaluations.scripts import probe_evidence_selection_image as pi
from evaluations.scripts import probe_evidence_selection_linear as pr  # text-frame prompt + char->token spans
from models.model import get_layers, image_token_groups

TASKS = ["steps_in_room", "rooms_visited", "co_occupancy",
         "room_busy", "char_accompanied", "char_alone",  # Cat-1 tasks (sum-of-indicator)
         "first_in_room", "last_in_room", "span_in_room"]  # TEMPORAL/order-dependent (sum cannot answer)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Frame-axis aggregator adapter on frozen 7B.")
    p.add_argument("--aggregator", choices=["seqmodel", "deepsets", "pna", "sum", "summax", "logic", "mamba"], default="seqmodel")
    p.add_argument("--tasks", default="steps_in_room,rooms_visited,co_occupancy",
                   help="comma-separated subset of tasks to train/eval on (e.g. rooms_visited for a single-task run)")
    p.add_argument("--text", action="store_true",
                   help="feed frames as TEXT (per-frame state token spans) instead of rendered images")
    p.add_argument("--phi", choices=["linear", "codebook"], default="linear")
    p.add_argument("--frame-pool", choices=["mean", "attn", "target", "slot", "none"], default="mean",
                   help="pool each frame: mean, query-conditioned attention, queried-entity token (text), "
                        "slot-attention (K query-free entity slots), or none=feed ALL frame tokens to the "
                        "aggregator (use with --aggregator mamba: per-token selective scan, extraction+aggregation in one)")
    p.add_argument("--num-slots", type=int, default=8, help="K for frame-pool=slot")
    p.add_argument("--mamba-readout", choices=["sum", "last"], default="sum",
                   help="sum=Σ outputs (counting-biased, extrapolates); last=final state (TASK-AGNOSTIC: the SSM "
                        "state encodes whatever reduction it learned — count/distinct/order — no additive bias)")
    p.add_argument("--pertoken-maxtok", type=int, default=64,
                   help="frame-pool=none: subsample each frame to this many tokens (0=all; >~64 is slow due to the python-loop scan)")
    p.add_argument("--balanced-loss", action="store_true", help="inverse-frequency count weighting per example")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "mmred_images_park")
    p.add_argument("--split", default="all_uniform")
    p.add_argument("--train-seq-lens", default="1,2,3,4")
    p.add_argument("--ood-seq-lens", default="5,6,7,8")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--ood-per", type=int, default=100, help="test_ood dirs sampled per OOD seq_len")
    p.add_argument("--val-cap", type=int, default=90, help="val dirs evaluated per epoch (each on all tasks)")
    p.add_argument("--max-train-per-seq", type=int, default=None, help="optional cap on train dirs per seq_len")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--patience", type=int, default=0, help="early-stop if val doesn't improve for N epochs (0=off)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--d-mem", type=int, default=256)
    p.add_argument("--read-layer", type=int, default=19, help="0-indexed decoder layer; pre-hook reads/injects here")
    p.add_argument("--aux-weight", type=float, default=0.5)
    p.add_argument("--frame-sup-weight", type=float, default=0.0,
                   help="BCE on per-frame count_scorer logits vs per-frame evidence labels (steps, per-frame pool). "
                        "Calibrates the additive readout -> soft-sum stops compounding (readout_benchmark: 0.52->0.996)")
    p.add_argument("--count-readout", choices=["ce", "additive"], default="ce",
                   help="ce=9-way classifier aux head (caps at top trained label); "
                        "additive=Σ sigmoid(per-frame score), extensive -> extrapolates to unseen counts")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init-from", type=str, default=None, help="resume: load adapter state_dict before training")
    p.add_argument("--split-seed", type=int, default=12345)
    p.add_argument("--holdout-counts", default="",
                   help="comma-separated gold counts held OUT of training -> test_ood (count-extrapolation OOD). Single-task only.")
    p.add_argument("--smoke", action="store_true", help="tiny run to validate end-to-end")
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "frame_axis" / "adapter_live" / "run")
    return p.parse_args()


# ----------------------------- adapter -----------------------------
class SlotAttention(nn.Module):
    """Locatello 2020 slot attention: K query-free entity slots via competitive (softmax-over-slots)
    attention + GRU update. Separates superposed entities so crowding doesn't dilute them."""
    def __init__(self, num_slots: int, dim: int, iters: int = 3, eps: float = 1e-8):
        super().__init__()
        self.num_slots, self.iters, self.eps, self.scale = num_slots, iters, eps, dim ** -0.5
        self.slots_mu = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.slots_logsigma = nn.Parameter(torch.zeros(1, dim))
        self.to_q = nn.Linear(dim, dim); self.to_k = nn.Linear(dim, dim); self.to_v = nn.Linear(dim, dim)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.norm_in = nn.LayerNorm(dim); self.norm_slots = nn.LayerNorm(dim); self.norm_mlp = nn.LayerNorm(dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:        # inputs [T,dim] -> slots [K,dim]
        inputs = self.norm_in(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)
        mu = self.slots_mu.expand(self.num_slots, -1)
        slots = mu + self.slots_logsigma.exp().expand(self.num_slots, -1) * torch.randn_like(mu)
        for _ in range(self.iters):
            prev = slots
            q = self.to_q(self.norm_slots(slots))                  # [K,dim]
            attn = torch.softmax((k @ q.t()) * self.scale, dim=1) + self.eps   # [T,K] compete over slots
            attn = attn / attn.sum(0, keepdim=True)                # weighted mean per slot
            updates = attn.t() @ v                                 # [K,dim]
            slots = self.gru(updates, prev)
            slots = slots + self.mlp(self.norm_mlp(slots))
        return slots


class FrameAggregatorAdapter(nn.Module):
    def __init__(self, hidden: int, d_mem: int, aggregator: str, max_frames: int = 16, n_counts: int = 9,
                 phi_mode: str = "linear", frame_pool: str = "mean", num_slots: int = 8,
                 pertoken_maxtok: int = 64, mamba_readout: str = "sum"):
        super().__init__()
        self.aggregator = aggregator
        self.phi_mode = phi_mode
        self.frame_pool = frame_pool
        self.pertoken_maxtok = pertoken_maxtok
        self.mamba_readout = mamba_readout
        if frame_pool == "attn":
            # query-conditioned attention pool over each frame's vision tokens: query from the
            # question-encoding position -> focus on the queried char/room instead of mean-diluting.
            self.q_proj = nn.Linear(hidden, d_mem)
            self.k_proj = nn.Linear(hidden, d_mem)
        if frame_pool == "slot":
            # K query-FREE entity slots per frame (task-agnostic), then query-select the relevant slot.
            self.slot_in = nn.Linear(hidden, d_mem)
            self.slot = SlotAttention(num_slots=num_slots, dim=d_mem)
            self.slot_q = nn.Linear(hidden, d_mem)
            self.slot_out = nn.Linear(d_mem, hidden)
        if phi_mode == "codebook":
            # project each frame onto a learned codebook; softmax -> near one-hot, so the max-channel
            # does soft set-union (the right structure for distinct-count / dedup).
            self.phi = nn.Linear(hidden, d_mem)
        else:
            self.phi = nn.Sequential(nn.Linear(hidden, d_mem), nn.LayerNorm(d_mem), nn.GELU())
        if aggregator == "seqmodel":
            self.frame_pos = nn.Parameter(torch.zeros(max_frames, d_mem)); nn.init.normal_(self.frame_pos, std=0.02)
            self.agg_tok = nn.Parameter(torch.zeros(1, d_mem)); nn.init.normal_(self.agg_tok, std=0.02)
            layer = nn.TransformerEncoderLayer(d_model=d_mem, nhead=4, dim_feedforward=2 * d_mem,
                                               batch_first=True, activation="gelu")
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            agg_out = d_mem
        elif aggregator == "pna":
            agg_out = 15 * d_mem  # 5 aggregators (sum,mean,max,min,std) x 3 degree-scalers
        elif aggregator == "sum":
            agg_out = d_mem        # sum only (count-preserving, extrapolates; no count-blind mean)
        elif aggregator == "summax":
            agg_out = 2 * d_mem    # sum (counting) + max (union/dedup), drops the count-blind mean
        elif aggregator == "logic":
            agg_out = 3 * d_mem + 3  # logical reductions: sum(count) + soft-OR(union) + soft-AND(all) + 3 cardinality scalars
        elif aggregator == "mamba":
            # selective diagonal SSM scan over frames (per-channel input-dependent step/gates,
            # learnable diagonal decay A=-exp(a_log) init near 0 => decay~1 => near-pure sum =>
            # magnitude-preserving/count-extrapolating, but can gate). Sum-of-outputs readout.
            self.mamba_in = nn.Linear(d_mem, d_mem, bias=False)
            self.mamba_delta = nn.Linear(d_mem, d_mem, bias=True)
            self.mamba_b = nn.Linear(d_mem, d_mem, bias=False)
            self.mamba_c = nn.Linear(d_mem, d_mem, bias=False)
            self.mamba_a_log = nn.Parameter(torch.full((d_mem,), -4.0))
            agg_out = d_mem
        else:  # deepsets
            agg_out = 3 * d_mem
        self.to_hagg = nn.Linear(agg_out, d_mem)
        self.rho = nn.Linear(d_mem, hidden)
        self.gate = nn.Parameter(torch.zeros(1))
        self.aux_head = nn.Linear(d_mem, n_counts)
        # Solution-1 extensive count head: per-frame "does this frame count?" score -> sum.
        # count = Σ_i sigmoid(w·m_i) is structurally unbounded (0..N), so it EXTRAPOLATES to counts
        # never seen in training -- unlike the 9-way CE classifier, which caps at the top trained label.
        self.count_scorer = nn.Linear(d_mem, 1)
        self.cur_spans: Optional[List[List[int]]] = None
        self.cur_pos: int = -1
        self.cur_hagg: Optional[torch.Tensor] = None
        self.cur_chat: Optional[torch.Tensor] = None
        self.cur_frame_logits: Optional[torch.Tensor] = None

    def pool_frame(self, frame_tokens: torch.Tensor, query_rep: torch.Tensor) -> torch.Tensor:
        """frame_tokens [Tf,H] -> [H]. mean, or query-conditioned attention over the frame's tokens."""
        if self.frame_pool == "mean":
            return frame_tokens.mean(0)
        if self.frame_pool == "slot":
            slots = self.slot(self.slot_in(frame_tokens))           # [K,d] query-free entity slots
            a = torch.softmax((slots @ self.slot_q(query_rep)) / (slots.shape[-1] ** 0.5), dim=0)  # [K]
            return self.slot_out((a.unsqueeze(-1) * slots).sum(0))  # [H] query-select relevant slot
        q = self.q_proj(query_rep)                                   # [d]
        k = self.k_proj(frame_tokens)                               # [Tf,d]
        attn = torch.softmax((k @ q) / (k.shape[-1] ** 0.5), dim=0)  # [Tf]
        return (attn.unsqueeze(-1) * frame_tokens).sum(0)           # [H] weighted pool of raw tokens

    def encode(self, reps: torch.Tensor) -> torch.Tensor:  # reps [N,H] -> per-frame messages [N,d]
        z = self.phi(reps)
        return torch.softmax(z, dim=-1) if self.phi_mode == "codebook" else z

    def aggregate(self, m: torch.Tensor) -> torch.Tensor:
        if self.aggregator == "seqmodel":
            n = m.shape[0]
            seq = torch.cat([self.agg_tok, m + self.frame_pos[:n]], dim=0).unsqueeze(0)
            return self.to_hagg(self.encoder(seq)[0, 0])
        if self.aggregator == "pna":
            n = int(m.shape[0])
            std = m.std(0, unbiased=False) if n > 1 else torch.zeros_like(m.sum(0))
            aggs = torch.stack([m.sum(0), m.mean(0), m.max(0).values, m.min(0).values, std], dim=0)  # [5,d]
            lg = torch.log1p(torch.tensor(float(n), device=m.device))
            scalers = torch.stack([torch.ones_like(lg), lg, 1.0 / (lg + 1e-6)])  # [3]
            feats = (aggs.unsqueeze(0) * scalers.view(3, 1, 1)).reshape(-1)  # [15*d]
            return self.to_hagg(feats)
        if self.aggregator == "sum":
            return self.to_hagg(m.sum(0))
        if self.aggregator == "summax":
            return self.to_hagg(torch.cat([m.sum(0), m.max(0).values], dim=-1))
        if self.aggregator == "logic":
            # treat sigmoid(message) as per-frame per-dim probabilities; reduce with the LOGICAL ops.
            p = torch.sigmoid(m).clamp(1e-4, 1 - 1e-4)          # [N,d]
            s_sum = p.sum(0)                                    # count ("how many")
            s_or = 1 - torch.exp(torch.log1p(-p).sum(0))        # soft union ("ever" / distinct)
            s_and = torch.exp(torch.log(p).sum(0))              # soft intersection ("always")
            card = torch.stack([s_sum.sum(), s_or.sum(), s_and.sum()])  # explicit cardinalities (distinct count = sum of soft-OR)
            return self.to_hagg(torch.cat([s_sum, s_or, s_and, card], dim=-1))
        if self.aggregator == "mamba":
            x = self.mamba_in(m)                                        # [N,d]
            delta = torch.nn.functional.softplus(self.mamba_delta(m))  # [N,d]
            b = self.mamba_b(m); c = self.mamba_c(m)                    # [N,d]
            a = -torch.exp(self.mamba_a_log)                           # [d]
            h = torch.zeros(m.shape[-1], device=m.device, dtype=m.dtype)
            ys = []
            for t in range(int(m.shape[0])):
                h = torch.exp(delta[t] * a) * h + delta[t] * b[t] * x[t]
                ys.append(c[t] * h)
            agg = ys[-1] if self.mamba_readout == "last" else torch.stack(ys, 0).sum(0)  # last=agnostic, sum=count-biased
            return self.to_hagg(agg)
        feats = torch.cat([m.sum(0), m.mean(0), m.max(0).values], dim=-1)  # deepsets
        return self.to_hagg(feats)

    def apply_to(self, hs: torch.Tensor) -> torch.Tensor:
        query_rep = hs[0, self.cur_pos, :].detach().float()  # question-encoding position
        if self.frame_pool == "none":
            # Exp-2: feed ALL frame tokens (frames in order) to the aggregator -> per-token selective
            # scan does extraction (gate separates entities) + aggregation in one. Query prepended.
            toks = []
            for span in self.cur_spans:
                ft = hs[0, torch.tensor(span, device=hs.device), :].detach().float()   # [Tf,H]
                if self.pertoken_maxtok and ft.shape[0] > self.pertoken_maxtok:
                    ft = ft[torch.linspace(0, ft.shape[0] - 1, self.pertoken_maxtok, device=ft.device).long()]
                toks.append(ft)
            m = torch.cat([self.encode(query_rep.unsqueeze(0)), self.encode(torch.cat(toks, 0))], dim=0)  # [1+ΣT, d]
        else:
            tgt = getattr(self, "cur_target_tok", None)
            reps = []
            for fi, span in enumerate(self.cur_spans):
                if self.frame_pool == "target" and tgt is not None and tgt[fi] is not None:
                    reps.append(hs[0, int(tgt[fi]), :].detach().float())  # queried-entity token (text)
                    continue
                idx = torch.tensor(span, device=hs.device)
                ft = hs[0, idx, :].detach().float()              # [Tf,H] this frame's vision tokens
                reps.append(self.pool_frame(ft, query_rep))
            m = self.encode(torch.stack(reps, dim=0))
        self.cur_frame_logits = self.count_scorer(m).squeeze(-1)   # per-(frame|token) score (for per-frame supervision)
        self.cur_chat = torch.sigmoid(self.cur_frame_logits).sum()  # extensive count = Σ per-(frame|token) prob
        hagg = self.aggregate(m)
        self.cur_hagg = hagg
        upd = self.rho(hagg).to(hs.dtype) * self.gate.to(hs.dtype)
        new_hs = hs.clone()
        new_hs[0, self.cur_pos, :] = new_hs[0, self.cur_pos, :] + upd
        return new_hs


# ----------------------------- data / splits -----------------------------
def _dirs(data_root: Path, split: str, sl: int) -> List[Path]:
    sr = data_root / f"seq_len_{sl}" / split
    if not sr.is_dir():
        return []
    return [d for d in sorted(sr.iterdir()) if (d / "qa.txt").is_file() and (d / "metadata.json").is_file()]


def declare_splits(data_root, split, train_seq, ood_seq, val_frac, test_frac, ood_per,
                   max_train_per_seq, seed, cap_per_seq=None) -> Dict[str, List[Tuple[str, int]]]:
    rng = random.Random(seed)
    out = {"train": [], "val": [], "test_iid": [], "test_ood": []}
    for sl in train_seq:  # stratified per seq_len
        dirs = _dirs(data_root, split, sl); rng.shuffle(dirs)
        if cap_per_seq:
            dirs = dirs[:cap_per_seq]
        n = len(dirs); n_va = int(val_frac * n); n_te = int(test_frac * n)
        val, test_iid, train = dirs[:n_va], dirs[n_va:n_va + n_te], dirs[n_va + n_te:]
        if max_train_per_seq:
            train = train[:max_train_per_seq]
        out["val"] += [(str(d), sl) for d in val]
        out["test_iid"] += [(str(d), sl) for d in test_iid]
        out["train"] += [(str(d), sl) for d in train]
    for sl in ood_seq:
        dirs = _dirs(data_root, split, sl); rng.shuffle(dirs)
        out["test_ood"] += [(str(d), sl) for d in dirs[:ood_per]]
    # globally shuffle each split so a capped eval (val_cap) is a representative sample across
    # seq_lens, not biased to the short ones (the lists are built seq_len-ordered above).
    for k in out:
        rng.shuffle(out[k])
    return out


def declare_splits_count_holdout(data_root, split, seq_len, task, holdout_counts,
                                 val_frac, test_frac, seed) -> Dict[str, List[Tuple[str, int]]]:
    """Count-extrapolation split: dirs whose canonical gold count is in `holdout_counts` become
    test_ood (counts NEVER seen in training); the rest split into train/val/test_iid. Balanced
    datasets pin the queried entity in metadata, so gold is deterministic per dir -> a clean OOD.
    Single-task (gold is task-specific)."""
    rng = random.Random(seed)
    out = {"train": [], "val": [], "test_iid": [], "test_ood": []}
    indist: List[Tuple[str, int]] = []
    for d in _dirs(data_root, split, seq_len):
        states = rv.states_of(d / "qa.txt")
        if not states:
            continue
        qg = tf.question_and_gold(task, d, states, random.Random(zlib.crc32((d.name + task).encode())))
        if qg is None:
            continue
        gold = int(qg[1])
        (out["test_ood"] if gold in holdout_counts else indist).append((str(d), seq_len))
    rng.shuffle(indist)
    n = len(indist); n_va = int(val_frac * n); n_te = int(test_frac * n)
    out["val"], out["test_iid"], out["train"] = indist[:n_va], indist[n_va:n_va + n_te], indist[n_va + n_te:]
    for k in out:
        rng.shuffle(out[k])
    return out


def make_example(d: Path, task: str, shared_rng: random.Random, eval_mode: bool, text: bool = False):
    states = rv.states_of(d / "qa.txt")
    if not states:
        return None
    meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    r = random.Random(zlib.crc32((d.name + task).encode())) if eval_mode else shared_rng
    qg = tf.question_and_gold(task, d, states, r)
    if qg is None:
        return None
    question, gold, _ = qg
    if text:  # frames fed as text -> no image load; the text path reads per-frame state token spans
        return None, question, int(gold), len(states), states
    try:
        frames = pi.load_frames(d, states, meta)
    except Exception:
        return None
    if len(frames) != len(states):
        return None
    return frames, question, int(gold), len(states), states


def text_frame_inputs(processor, states, question, device):
    """Chat-templated TEXT prompt (per-frame state lines) + per-frame token spans. Mirrors the image
    path so adapter.apply_to reads/pools each frame's TEXT tokens at L_READ instead of vision tokens."""
    user_text, char_spans = pr.build_probe_prompt(question, states)  # "...frames as text:\n" + per-frame blocks
    templated = processor.apply_chat_template([{"role": "user", "content": user_text}],
                                              add_generation_prompt=True, tokenize=False)
    off = templated.find(user_text)
    if off < 0:
        return None  # chat template altered the content -> spans would be wrong
    abs_spans = [(off + a, off + b) for (a, b) in char_spans]
    tok_spans = pr.char_spans_to_token_spans(processor.tokenizer, templated, abs_spans)
    enc = processor.tokenizer(templated, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(device) for k, v in enc.items()}
    # per-frame queried-entity token (for frame_pool='target'): first mention of the queried name in each frame
    m = re.search(r"did (\w+) (?:visit|spend)|were (\w+) and", question)
    tgt = next((g for g in (m.groups() if m else []) if g), None)
    target_tok = []
    for (a, b) in abs_spans:
        pos = templated.find(tgt, a, b) if tgt else -1
        target_tok.append(len(processor.tokenizer(templated[:pos], add_special_tokens=False).input_ids) if pos >= 0 else None)
    return inputs, tok_spans, target_tok


def build_inputs(processor, frames, question, device):
    preamble = f"Question: {question}\nThe following are the {len(frames)} frames showing rooms in a house:"
    messages = [{"role": "user", "content": [{"type": "text", "text": preamble}]
                 + [{"type": "image", "image": im} for im in frames]}]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt")
    return base.move_inputs_to_device(dict(inputs), device)


# ----------------------------- run -----------------------------
def main() -> int:
    args = parse_args()
    tasks_use = [t for t in TASKS if t in {x.strip() for x in str(args.tasks).split(",")}]
    if not tasks_use:
        raise SystemExit(f"--tasks {args.tasks!r} matched none of {TASKS}")
    if args.smoke:
        args.epochs, args.ood_per, args.val_cap, args.max_train_per_seq = 2, 8, 6, 6
    shared_rng = random.Random(int(args.seed))
    device = base.resolve_device(args.device)
    dtype = base.resolve_dtype(args.dtype, device)
    run_dir = args.output / f"{time.strftime('%Y%m%d_%H%M%S')}_{args.aggregator}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log = (run_dir / "run.log").open("w", encoding="utf-8")

    def emit(m: str) -> None:
        print(m, flush=True); log.write(m + "\n"); log.flush()

    train_seq = [int(x) for x in args.train_seq_lens.replace(",", " ").split()]
    ood_seq = [int(x) for x in args.ood_seq_lens.replace(",", " ").split()]
    holdout = [int(x) for x in str(args.holdout_counts).replace(",", " ").split()] if args.holdout_counts else []
    if holdout:
        tasks_list = [t for t in args.tasks.replace(",", " ").split() if t]
        assert len(tasks_list) == 1, "count-holdout requires a single --tasks"
        splits = declare_splits_count_holdout(args.data_root, args.split, train_seq[0], tasks_list[0],
                                              set(holdout), args.val_frac, args.test_frac, args.split_seed)
    else:
        splits = declare_splits(args.data_root, args.split, train_seq, ood_seq, args.val_frac,
                                args.test_frac, args.ood_per, args.max_train_per_seq, args.split_seed)
    (run_dir / "splits.json").write_text(json.dumps(
        {k: [Path(p).name for p, _ in v] for k, v in splits.items()}, indent=0), encoding="utf-8")
    emit(f"aggregator={args.aggregator} read_layer={args.read_layer} epochs={args.epochs} lr={args.lr} smoke={args.smoke}")
    emit(f"splits: train={len(splits['train'])} val={len(splits['val'])} "
         f"test_iid={len(splits['test_iid'])} test_ood={len(splits['test_ood'])}")

    model, processor = base.load_model_and_processor(args.model_name, device, dtype, bool(args.load_in_4bit))
    for p in model.parameters():
        p.requires_grad_(False)
    hidden = int(model.config.text_config.hidden_size) if hasattr(model.config, "text_config") \
        else int(model.config.hidden_size)
    adapter = FrameAggregatorAdapter(hidden, args.d_mem, args.aggregator, phi_mode=args.phi,
                                     frame_pool=args.frame_pool, num_slots=args.num_slots,
                                     pertoken_maxtok=args.pertoken_maxtok,
                                     mamba_readout=args.mamba_readout).to(device).float()
    if args.init_from:
        adapter.load_state_dict(torch.load(args.init_from, map_location=device))
        emit(f"resumed adapter from {args.init_from}")
    emit(f"hidden={hidden} aggregator={args.aggregator} phi={args.phi} balanced={args.balanced_loss} "
         f"adapter_params={sum(p.numel() for p in adapter.parameters()):,}")

    # count-balanced per-example weights (gold-only pass over train; no images -> fast)
    count_w = None
    if args.balanced_loss:
        freq = {}
        for dstr, sl in splits["train"]:
            st = rv.states_of(Path(dstr) / "qa.txt")
            if not st:
                continue
            for t in tasks_use:
                qg = tf.question_and_gold(t, Path(dstr), st, random.Random(0))
                if qg:
                    freq[qg[1]] = freq.get(qg[1], 0) + 1
        tot = sum(freq.values()) or 1
        count_w = {c: (tot / (len(freq) * n)) for c, n in freq.items()}  # inverse-frequency, mean ~1
        emit(f"balanced-loss: count weights {{c: w}} = "
             f"{{{', '.join(f'{c}:{w:.2f}' for c, w in sorted(count_w.items()))}}}")

    layers = get_layers(model)
    target_layer = layers[int(args.read_layer)]

    def pre_hook(module, hargs, hkwargs):
        if len(hargs) >= 1:
            return (adapter.apply_to(hargs[0]),) + tuple(hargs[1:]), hkwargs
        hkwargs = dict(hkwargs); hkwargs["hidden_states"] = adapter.apply_to(hkwargs["hidden_states"])
        return hargs, hkwargs
    handle = target_layer.register_forward_pre_hook(pre_hook, with_kwargs=True)

    def fwd(frames, question, states):
        if args.text:
            r = text_frame_inputs(processor, states, question, device)
            if r is None:
                return None
            inputs, spans, target_tok = r
            if len(spans) != len(states):
                return None
            adapter.cur_spans = spans; adapter.cur_target_tok = target_tok
            adapter.cur_pos = int(inputs["input_ids"].shape[1]) - 1
            out = model(**inputs, use_cache=False)
            return out.logits[0, -1, :], adapter.cur_hagg
        inputs = build_inputs(processor, frames, question, device)
        ids = inputs["input_ids"]
        spans = image_token_groups(ids[0].detach().cpu(), len(frames), processor=processor)
        if len(spans) != len(frames):
            return None
        adapter.cur_spans = spans; adapter.cur_pos = int(ids.shape[1]) - 1
        out = model(**inputs, use_cache=False)
        return out.logits[0, -1, :], adapter.cur_hagg

    def gold_tok(g: int) -> int:
        return int(processor.tokenizer(str(g), add_special_tokens=False).input_ids[0])

    INT_RE = tf.INTEGER_RE

    def eval_items(items, cap=None, records=None, split_name=""):
        """Return agg[(task,readout)]=[n,correct,gold_sum,pred_sum] and per-task grid (LM readout)."""
        adapter.eval()
        agg: Dict[Tuple[str, str], List[float]] = {}
        grids: Dict[str, Dict[Tuple[int, int], Tuple[int, int]]] = {t: {} for t in tasks_use}
        use = items if cap is None else items[:cap]
        for (dstr, sl) in use:
            d = Path(dstr)
            for task in tasks_use:
                ex = make_example(d, task, shared_rng, eval_mode=True, text=args.text)
                if ex is None:
                    continue
                frames, question, gold, n, states = ex
                with torch.inference_mode():
                    res = fwd(frames, question, states)
                    if res is None:
                        continue
                    logits, hagg = res
                    pred_lm_id = int(logits.argmax())
                    pred_aux = int(adapter.aux_head(hagg).argmax())
                    pred_add = (int(round(float(adapter.cur_chat)))
                                if args.count_readout == "additive" and adapter.cur_chat is not None else None)
                m = INT_RE.search(processor.tokenizer.decode([pred_lm_id]))
                pred_lm = int(m.group(0)) if m else None
                if records is not None and pred_lm is not None:
                    records.append((task, split_name, int(sl), int(gold), int(pred_lm)))
                ro_list = [("lm", pred_lm), ("aux", pred_aux)]
                if args.count_readout == "additive":
                    ro_list.append(("add", pred_add))
                for readout, pred in ro_list:
                    ok = int(pred is not None and pred == gold)
                    a = agg.setdefault((task, readout), [0, 0, 0.0, 0.0])
                    a[0] += 1; a[1] += ok; a[2] += gold; a[3] += (pred if pred is not None else 0)
                    if readout == "lm":
                        c, t = grids[task].get((sl, gold), (0, 0)); grids[task][(sl, gold)] = (c + ok, t + 1)
        return agg, grids

    def acc_bias(a):
        n, cor, gs, ps = a
        return cor / max(1, n), (ps - gs) / max(1, n), gs / max(1, n), ps / max(1, n)

    # ---- train with per-epoch validation ----
    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()
    READOUTS = ("lm", "aux", "add") if args.count_readout == "additive" else ("lm", "aux")
    val_rows = ["epoch,task,readout,n,val_acc,val_bias,mean_gold,mean_pred"]
    best_val, best_state, best_epoch = -1.0, None, -1
    for epoch in range(args.epochs):
        adapter.train()
        order = list(splits["train"]); shared_rng.shuffle(order)
        opt.zero_grad(); run_loss = 0.0; seen = 0; step = 0
        for i, (dstr, sl) in enumerate(order):
            task = tasks_use[(i + epoch) % len(tasks_use)]
            ex = make_example(Path(dstr), task, shared_rng, eval_mode=False, text=args.text)
            if ex is None:
                continue
            frames, question, gold, n, states = ex
            try:
                res = fwd(frames, question, states)
            except Exception as exc:
                emit(f"  train skip {Path(dstr).name}: {exc}"); continue
            if res is None:
                continue
            logits, hagg = res
            loss_lm = ce(logits.unsqueeze(0).float(), torch.tensor([gold_tok(gold)], device=device))
            if args.count_readout == "additive":
                # regress the extensive count head to gold (no closed label set -> extrapolates)
                aux_term = torch.nn.functional.smooth_l1_loss(
                    adapter.cur_chat, torch.tensor(float(gold), device=device, dtype=adapter.cur_chat.dtype))
            else:
                aux_term = ce(adapter.aux_head(hagg).unsqueeze(0), torch.tensor([max(0, min(gold, 8))], device=device))
            w = count_w.get(gold, 1.0) if count_w is not None else 1.0
            frame_sup = torch.tensor(0.0, device=device)
            if (args.frame_sup_weight > 0 and task in ("steps_in_room", "co_occupancy")
                    and args.frame_pool != "none"
                    and adapter.cur_frame_logits is not None
                    and adapter.cur_frame_logits.numel() == len(states)):
                md = json.loads((Path(dstr) / "metadata.json").read_text(encoding="utf-8"))
                pf_list = None
                if task == "steps_in_room":
                    C_, R_ = md.get("target_character"), md.get("target_room")
                    if C_ and R_:
                        pf_list = [float(tf.room_of(s, C_) == R_) for s in states]
                else:  # co_occupancy: binary per-frame predicate (queried pair in the same room)
                    qp = md.get("query_pair")
                    if qp and len(qp) == 2:
                        C_, D_ = qp
                        pf_list = [float(tf.room_of(s, C_) == tf.room_of(s, D_)
                                         and tf.room_of(s, C_) != "not present") for s in states]
                if pf_list is not None:
                    pf = torch.tensor(pf_list, device=device, dtype=adapter.cur_frame_logits.dtype)
                    frame_sup = torch.nn.functional.binary_cross_entropy_with_logits(adapter.cur_frame_logits, pf)
            (w * (loss_lm + args.aux_weight * aux_term + args.frame_sup_weight * frame_sup) / args.grad_accum).backward()
            run_loss += float(loss_lm.detach()); seen += 1; step += 1
            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0); opt.step(); opt.zero_grad()
        # validation
        vagg, _ = eval_items(splits["val"], cap=args.val_cap)
        accs = []
        for task in tasks_use:
            for readout in READOUTS:
                if (task, readout) in vagg:
                    acc, bias, mg, mp = acc_bias(vagg[(task, readout)])
                    val_rows.append(f"{epoch},{task},{readout},{vagg[(task,readout)][0]},{acc:.4f},{bias:+.3f},{mg:.3f},{mp:.3f}")
                    if readout == "lm":
                        accs.append(acc)
        mean_val = sum(accs) / max(1, len(accs))
        emit(f"epoch {epoch}: train_lm_loss={run_loss/max(1,seen):.3f} gate={float(adapter.gate):.3f} "
             f"mean_val_lm_acc={mean_val:.3f}")
        if mean_val > best_val:
            best_val, best_epoch = mean_val, epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in adapter.state_dict().items()})
        if args.patience and (epoch - best_epoch) >= args.patience:
            emit(f"early stop: no val improvement in {args.patience} epochs (best_epoch={best_epoch})")
            break
    (run_dir / "val_by_epoch.csv").write_text("\n".join(val_rows) + "\n", encoding="utf-8")
    emit(f"best_epoch={best_epoch} mean_val_lm_acc={best_val:.3f}")

    # ---- final test on best checkpoint ----
    if best_state is not None:
        adapter.load_state_dict(best_state)
    torch.save(best_state or adapter.state_dict(), run_dir / "adapter_best.pt")
    summary = ["split,task,readout,n,accuracy,mean_gold,mean_pred,bias"]
    records: List[Tuple[str, str, int, int, int]] = []  # (task, split, seq_len, gold, pred_lm)
    for split_name in ("test_iid", "test_ood"):
        if not splits[split_name]:
            continue
        agg, grids = eval_items(splits[split_name], records=records, split_name=split_name)
        for task in tasks_use:
            for readout in READOUTS:
                if (task, readout) in agg:
                    acc, bias, mg, mp = acc_bias(agg[(task, readout)])
                    row = f"{split_name},{task},{readout},{agg[(task,readout)][0]},{acc:.4f},{mg:.3f},{mp:.3f},{bias:+.3f}"
                    summary.append(row); emit(row)
            tf.save_heatmap(grids[task], f"{task} ({split_name}, LM)",
                            run_dir / f"heatmap_{task}_{split_name.replace('test_', '')}.png")
    (run_dir / "summary.csv").write_text("\n".join(summary) + "\n", encoding="utf-8")
    import experiments.glstm.frame_axis_aggregator_cached as fc
    pred_rows = ["task,split,seq_len,gold,pred"] + [f"{t},{s},{sl},{g},{p}" for (t, s, sl, g, p) in records]
    (run_dir / "predictions.csv").write_text("\n".join(pred_rows) + "\n", encoding="utf-8")
    # Remove the injection hook BEFORE measuring extraction p (the probe needs the CLEAN L_READ reps).
    handle.remove()
    # Measure each task's per-frame extraction accuracy at the read layer (L19) -> the basis of the
    # extraction-bound ceiling. Falls back to P_EXTRACT defaults for any task it can't measure.
    measured_p = {}
    if not args.text:  # extraction-p is measured on VISION reps; skip for text runs (would be misleading)
        emit(f"measuring per-frame extraction p @L{args.read_layer} on test_iid ...")
        measured_p = fc.measure_extraction_p(model, processor, splits.get("test_iid", []), tasks_use,
                                             args.read_layer, device, cap=150, emit=emit)
    (run_dir / "extraction_p.json").write_text(json.dumps(measured_p, indent=2), encoding="utf-8")
    for sp in ("test_iid", "test_ood"):
        rec_sp = [r for r in records if r[1] == sp]
        if rec_sp:
            ceil = fc.extraction_ceilings(splits[sp], tasks=tasks_use, p_extract=measured_p) \
                if (splits.get(sp) and not args.text) else None
            fc.make_diagnostic_plots(rec_sp, 13, run_dir, suffix=f"_{sp.replace('test_', '')}", ceilings=ceil)
    emit(f"wrote {run_dir}/ (summary.csv, val_by_epoch.csv, splits.json, heatmaps, extraction_p.json, "
         f"acc_per_count plots, adapter_best.pt)")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

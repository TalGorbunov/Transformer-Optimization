"""CarrierEngine: sample geometry prep + all carrier forwards/decodes on the frozen model.

One class, used by every entrypoint:
  prepare_sample()        prompt build + span location + masks + positions + embeddings
  forward_logits()        full-stack lo/hi forward -> next-token logits (train or eval)
  decode_answer()         greedy digit decode (digit-readout ckpts)
  decode_scratchpad()     greedy scratchpad decode + format parser (the exam)
  decode_fast()           cached incremental decode (exact w.r.t. the mask semantics;
                          16-311x decode speedups — RESULTS.md [2026-07-25] TRUNC E1/E6)
  build_training_cache()  cached lo-phase for the production trainer
  top_hidden()            hi-phase over cached states (jitter / grad-ckpt / truncation)

Ported faithfully from legacy/experiments/glstm/carrier_layer_lora.py and
carrier_layer_cached.py — the math, kernel selection (EFFICIENT->MATH), dtype and
ordering are preserved so existing checkpoints reproduce their logged numbers.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from .carriers import ext_mask, frame_cols, keep_cols, make_masks, truncated_masks
from .constants import CARRIER_TOKEN, FRAME_RESIZE, MASK_MIN
from .fencing import find_question_spans, find_subseq, frame_blocks, reset_positions
from .runtime import ModelRuntime, get_layers, get_rope_index_fn, image_token_groups
from .scratchpad import couple_offsets, parse_scratchpad_answer

# EFFICIENT first (O(seq) memory — MATH materializes ~30GB of scores at seq~23k),
# MATH fallback. The same fixed list everywhere -> consistent backend selection.
SDPA_BACKENDS = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


class CarrierEngine:
    def __init__(
        self,
        rt: ModelRuntime,
        *,
        l_open: int,
        e_c: torch.Tensor,
        pos_couple: bool = False,
    ):
        self.rt = rt
        self.model = rt.model
        self.processor = rt.processor
        self.tok = rt.processor.tokenizer
        self.layers = get_layers(rt.model)
        self.n_layers = len(self.layers)
        self.l_open = l_open
        self.pos_couple = pos_couple
        self.text_model = rt.model.model.language_model
        self.dev = rt.model.device
        self.e_c = e_c  # tensor or nn.Parameter on device
        self.carrier_id = self.tok.convert_tokens_to_ids(CARRIER_TOKEN)
        self.vision_start_id = int(rt.model.config.vision_start_token_id)
        self.rope_fn = get_rope_index_fn(rt.model)
        self.digit_ids = [self.tok(str(d), add_special_tokens=False).input_ids[0] for d in range(10)]

    # ------------------------------------------------------------------ sample prep

    def prepare_sample(
        self,
        frames: List[Any],
        question: str,
        *,
        gold: int,
        task: str = "steps",
        resize: int = FRAME_RESIZE,
        qfirst: bool = True,
        posreset: bool = True,
        with_masks: bool = False,
        with_trunc_cols: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Build the carrier prompt [q0][frame_1][c]...[frame_N][c][q0], locate spans,
        compute positions (reset + sequential carrier override + tail re-base) and the
        embedded prompt (image features scattered in; e_c NOT yet injected). Returns the
        sample record dict, or None if any structural check fails (caller counts a skip).
        """
        NF = len(frames)
        if resize > 0:
            frames = [f.resize((resize, resize)) for f in frames]
        content = [{"type": "text", "text": question}] if qfirst else []
        for f in frames:
            content.append({"type": "image", "image": f})
            content.append({"type": "text", "text": CARRIER_TOKEN})
        content.append({"type": "text", "text": question})
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: (v.to(self.dev) if hasattr(v, "to") else v) for k, v in inputs.items()}
        ids = inputs["input_ids"][0].tolist()
        seq = len(ids)
        fg = image_token_groups(
            inputs["input_ids"][0].cpu(), expected_num_frames=NF, processor=self.processor
        )
        cpos = [p for p, t in enumerate(ids) if t == self.carrier_id]
        vstarts = [p for p, t in enumerate(ids) if t == self.vision_start_id]
        n_occ = 2 if qfirst else 1
        spans = find_question_spans(ids, self.tok, question, n_occ)
        if len(fg) != NF or len(cpos) != NF or len(vstarts) != NF or spans is None:
            return None
        fin_start = spans[-1][0]
        blocks = frame_blocks(vstarts, fin_start)

        with torch.no_grad():
            base_pos, _ = self.rope_fn(
                inputs["input_ids"],
                image_grid_thw=inputs.get("image_grid_thw"),
                attention_mask=inputs.get("attention_mask"),
            )
            if not posreset:
                pos = base_pos.clone()
                blk0_max = int(pos.max())  # only used by jitter
            else:
                pos = reset_positions(base_pos, blocks, fin_start).clone()
                blk0_max = int(pos[:, :, blocks[0][0] : blocks[0][1]].max())
                for i, c in enumerate(cpos):  # carriers: sequential ordered positions
                    pos[:, :, c] = blk0_max + 1 + i
                pos[:, :, fin_start:] += NF  # tail re-based after the carrier run
            emb = self.text_model.embed_tokens(inputs["input_ids"])
            img = self.model.model.get_image_features(
                inputs["pixel_values"], inputs["image_grid_thw"]
            )
            img = torch.cat(img, dim=0) if isinstance(img, (list, tuple)) else img
            im_mask = inputs["input_ids"][0] == self.model.config.image_token_id
            emb = emb.clone()
            emb[0, im_mask] = img.to(emb.dtype)

        rec: Dict[str, Any] = {
            "emb": emb[0].to(torch.bfloat16),
            "ids": ids,
            "pos": pos,
            "cpos": cpos,
            "blocks": blocks,
            "fin": fin_start,
            "seq": seq,
            "gold": gold,
            "task": task,
            "grp": f"{task}{NF}",
            "blk0max": blk0_max,
        }
        if with_masks:
            lo, hi = make_masks(seq, blocks, cpos, fin_start)
            rec["lo"] = lo.to(torch.float16)
            rec["hi"] = hi.to(torch.float16)
        if with_trunc_cols:
            rec["keep"] = keep_cols(seq, blocks, cpos)
            rec["fcols"] = frame_cols(seq, blocks, cpos)
        return rec

    # -------------------------------------------------------------- full-stack forward

    def forward_logits(
        self,
        d: Dict[str, Any],
        grad: bool,
        extra: Tuple[int, ...] = (),
        dropkv: bool = False,
        trunc: Optional[int] = None,
    ) -> torch.Tensor:
        """lo/hi forward over [prompt || extra decoded tokens] -> last-row logits."""
        e = len(extra)
        seq = d["seq"] + e
        emb = d["emb"].to(self.dev).unsqueeze(0)
        if e:
            ext = self.text_model.embed_tokens(torch.tensor([list(extra)], device=self.dev))
            emb = torch.cat([emb, ext.to(emb.dtype)], dim=1)
        emb = emb.clone()
        if d["cpos"]:  # replica-scaffold records (learnmask) carry no carriers
            stack = self.e_c.unsqueeze(0).repeat(len(d["cpos"]), 1).to(torch.bfloat16)
            emb[0, torch.tensor(d["cpos"], device=self.dev)] = stack if grad else stack.detach()
        if "lo" in d:
            lo_m, hi_m = d["lo"], d["hi"]
        else:  # training: masks rebuilt lazily (caching seq^2 per sample blows RAM)
            lo_m, hi_m = make_masks(d["seq"], d["blocks"], d["cpos"], d["fin"])
        lo2, hi2 = ext_mask(lo_m, e), ext_mask(hi_m, e)
        if dropkv and e:  # decoded rows lose frame cols
            fc = torch.tensor(d["fcols"], dtype=torch.long)
            lo2[d["seq"] :, fc] = MASK_MIN
            hi2[d["seq"] :, fc] = MASK_MIN
        lo = lo2.to(self.dev).to(torch.float32).view(1, 1, seq, seq)
        hi = hi2.to(self.dev).to(torch.float32).view(1, 1, seq, seq)
        pos = d["pos"].to(self.dev)
        if e:
            if self.pos_couple and d.get("task") != "rooms":
                texts = [self.tok.decode([t]) for t in extra]
                anch = couple_offsets(texts, len(d["cpos"]))
                cp = [int(pos[0, 0, c]) for c in d["cpos"]]
                vals = torch.tensor(
                    [cp[a - 1] + o for a, o in anch], device=self.dev
                ).view(1, 1, e).expand(3, 1, e)
                pos = torch.cat([pos, vals], dim=2)
            else:
                inc = torch.arange(1, e + 1, device=self.dev).view(1, 1, e)
                pos = torch.cat([pos, pos[:, :, -1:] + inc], dim=2)
        cos_, sin_ = self.text_model.rotary_emb(emb, pos)
        pe = (cos_.to(emb.dtype), sin_.to(emb.dtype))
        h = emb
        LO = self.l_open
        with sdpa_kernel(SDPA_BACKENDS):
            if trunc is None:
                for li, ly in enumerate(self.layers):
                    h = ly(h, attention_mask=(lo if li < LO else hi), position_embeddings=pe)[0]
            else:
                for li in range(trunc):
                    h = self.layers[li](
                        h, attention_mask=(lo if li < LO else hi), position_embeddings=pe
                    )[0]
                # physical truncation: index-select rows/masks/positions — ORIGINAL
                # position ids preserved, never renumbered
                kt = torch.tensor(d["keep"] + list(range(d["seq"], seq)), device=self.dev)
                k2 = kt.numel()
                h = h.index_select(1, kt)
                lo_t = lo[0, 0].index_select(0, kt).index_select(1, kt).view(1, 1, k2, k2)
                hi_t = hi[0, 0].index_select(0, kt).index_select(1, kt).view(1, 1, k2, k2)
                cos_t, sin_t = self.text_model.rotary_emb(h, pos.index_select(2, kt))
                pe_t = (cos_t.to(h.dtype), sin_t.to(h.dtype))
                for li in range(trunc, self.n_layers):
                    h = self.layers[li](
                        h, attention_mask=(lo_t if li < LO else hi_t), position_embeddings=pe_t
                    )[0]
        h = self.text_model.norm(h)
        return self.model.lm_head(h[0, -1].to(self.model.lm_head.weight.dtype)).float()

    # ------------------------------------------------------------------- greedy decode

    def decode_answer(self, d, *, decode_tokens: int, dropkv=False, trunc=None):
        """Greedy digit decode. -> (parsed int or None, 0-9-restricted first-token argmax)."""
        toks: List[int] = []
        first_digit = None
        for step in range(decode_tokens):
            lg = self.forward_logits(d, False, extra=tuple(toks), dropkv=dropkv, trunc=trunc)
            if step == 0:
                first_digit = int(np.argmax([float(lg[t]) for t in self.digit_ids]))
            t = int(lg.argmax())
            if not self.tok.decode([t]).strip().isdigit():
                break
            toks.append(t)
        text = self.tok.decode(toks).strip()
        return (int(text) if text.isdigit() else None), first_digit

    def decode_scratchpad(self, d, *, decode_tokens: int, fmt: str, dropkv=False, trunc=None):
        """Greedy scratchpad decode, stop at EOS. -> (parsed int or None, text, token ids)."""
        toks: List[int] = []
        for _step in range(decode_tokens):
            lg = self.forward_logits(d, False, extra=tuple(toks), dropkv=dropkv, trunc=trunc)
            t = int(lg.argmax())
            if t == self.tok.eos_token_id:
                break
            toks.append(t)
        text = self.tok.decode(toks)
        return parse_scratchpad_answer(text, fmt), text, toks

    # ------------------------------------------------------------ cached fast decode

    def prefill_capture(self, d, trunc: Optional[int] = None):
        """One prefill over the prompt (honoring truncation), capturing every layer's
        INPUT hidden states at the keep columns (exact for all decode steps — prompt rows
        never attend appended rows). -> (caches, step0 logits, lo_t, hi_t, pos_keep,
        last prompt position)."""
        seq = d["seq"]
        emb = d["emb"].to(self.dev).unsqueeze(0).clone()
        stack = self.e_c.unsqueeze(0).repeat(len(d["cpos"]), 1).to(torch.bfloat16)
        emb[0, torch.tensor(d["cpos"], device=self.dev)] = stack.detach()
        lo = d["lo"].to(self.dev).to(torch.float32).view(1, 1, seq, seq)
        hi = d["hi"].to(self.dev).to(torch.float32).view(1, 1, seq, seq)
        pos = d["pos"].to(self.dev)
        cos_, sin_ = self.text_model.rotary_emb(emb, pos)
        pe = (cos_.to(emb.dtype), sin_.to(emb.dtype))
        kt = torch.tensor(d["keep"], device=self.dev)
        k = kt.numel()
        pos_k = pos.index_select(2, kt)
        lo_t = lo[0, 0].index_select(0, kt).index_select(1, kt)
        hi_t = hi[0, 0].index_select(0, kt).index_select(1, kt)
        caches: List[torch.Tensor] = []
        h = emb
        done = False
        pe_t = None
        LO = self.l_open
        with torch.no_grad(), sdpa_kernel(SDPA_BACKENDS):
            for li, ly in enumerate(self.layers):
                if trunc is not None and li == trunc:
                    h = h.index_select(1, kt)
                    cos_t, sin_t = self.text_model.rotary_emb(h, pos_k)
                    pe_t = (cos_t.to(h.dtype), sin_t.to(h.dtype))
                    done = True
                caches.append(h[0].clone() if done else h[0].index_select(0, kt).clone())
                if done:
                    h = ly(
                        h,
                        attention_mask=(lo_t if li < LO else hi_t).view(1, 1, k, k),
                        position_embeddings=pe_t,
                    )[0]
                else:
                    h = ly(h, attention_mask=(lo if li < LO else hi), position_embeddings=pe)[0]
            hn = self.text_model.norm(h)
            lg0 = self.model.lm_head(hn[0, -1].to(self.model.lm_head.weight.dtype)).float()
        return (
            caches,
            lg0,
            lo_t.cpu().to(torch.float16),
            hi_t.cpu().to(torch.float16),
            pos_k,
            pos[:, :, -1:],
        )

    def prefill_chunked(self, d, trunc: int):
        """Chunked prefill: fence+posreset make layers 0..trunc-1 EXACTLY per-block
        computable ([prefix+question][frame_i][carrier_i] chunks batched, plain causal) —
        no dense seq^2 mask anywhere. Tail rows come from a frames-free chunk (full
        truncation semantics). Same returns as prefill_capture."""
        seq = d["seq"]
        blocks, cpos, fin, keep = d["blocks"], d["cpos"], d["fin"], d["keep"]
        a0 = blocks[0][0]
        NF = len(cpos)
        assert all(c == b - 1 for c, (a, b) in zip(cpos, blocks)), "carrier must end block"
        bl = blocks[0][1] - blocks[0][0]
        assert all(b - a == bl for a, b in blocks), "posreset requires equal blocks"
        emb = d["emb"].to(self.dev).clone()
        emb[torch.tensor(cpos, device=self.dev)] = self.e_c.detach().to(torch.bfloat16)
        pos = d["pos"].to(self.dev)
        Lb = a0 + bl
        bat = torch.stack([torch.cat([emb[:a0], emb[a:b]]) for a, b in blocks])
        pos_b = torch.cat(
            [torch.cat([pos[:, :, :a0], pos[:, :, a:b]], dim=2) for a, b in blocks], dim=1
        )
        Lt = a0 + (seq - fin)
        tl = torch.cat([emb[:a0], emb[fin:]]).unsqueeze(0)
        pos_t = torch.cat([pos[:, :, :a0], pos[:, :, fin:]], dim=2)

        def _causal(n):
            m = torch.zeros(n, n, dtype=torch.float32, device=self.dev)
            m.masked_fill_(torch.triu(torch.ones(n, n, dtype=torch.bool, device=self.dev), 1), MASK_MIN)
            return m.view(1, 1, n, n)

        mb, mt = _causal(Lb), _causal(Lt)
        cos_b, sin_b = self.text_model.rotary_emb(bat, pos_b)
        pe_b = (cos_b.to(bat.dtype), sin_b.to(bat.dtype))
        cos_t, sin_t = self.text_model.rotary_emb(tl, pos_t)
        pe_tl = (cos_t.to(tl.dtype), sin_t.to(tl.dtype))
        lo_t, hi_t = truncated_masks(keep, cpos)
        k = len(keep)
        kt = torch.tensor(keep, device=self.dev)
        pos_k = pos.index_select(2, kt)

        def _assemble(hb, ht):
            return torch.cat([ht[0, :a0], hb[:, -1, :], ht[0, a0:]], dim=0)

        caches: List[torch.Tensor] = []
        hb, ht = bat, tl
        LO = self.l_open
        with torch.no_grad(), sdpa_kernel(SDPA_BACKENDS):
            for li in range(trunc):
                caches.append(_assemble(hb, ht).clone())
                hb = self.layers[li](hb, attention_mask=mb, position_embeddings=pe_b)[0]
                ht = self.layers[li](ht, attention_mask=mt, position_embeddings=pe_tl)[0]
            h = _assemble(hb, ht).unsqueeze(0)
            lo4 = lo_t.to(self.dev).to(torch.float32).view(1, 1, k, k)
            hi4 = hi_t.to(self.dev).to(torch.float32).view(1, 1, k, k)
            cos_k, sin_k = self.text_model.rotary_emb(h, pos_k)
            pe_k = (cos_k.to(h.dtype), sin_k.to(h.dtype))
            for li in range(trunc, self.n_layers):
                caches.append(h[0].clone())
                h = self.layers[li](
                    h, attention_mask=(lo4 if li < LO else hi4), position_embeddings=pe_k
                )[0]
            hn = self.text_model.norm(h)
            lg0 = self.model.lm_head(hn[0, -1].to(self.model.lm_head.weight.dtype)).float()
        return caches, lg0, lo_t, hi_t, pos_k, pos[:, :, -1:]

    def decode_fast(self, d, *, decode_tokens: int, fmt: str, trunc=None, chunked=False,
                    selector=None):
        """Cached incremental greedy decode — mathematically equal to drop-frame-kv
        (+truncate) with a real speedup. -> (parsed, text, toks, prefill_seconds).
        selector(lg, toks)->token_id optionally replaces plain argmax (grammar
        constraints; see gnnformer.scan_grammar.make_token_selector)."""
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        tp = time.time()
        caches, lg0, lo_t, hi_t, pos_k, pos_last = (
            self.prefill_chunked(d, trunc) if chunked else self.prefill_capture(d, trunc)
        )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        prefill_s = time.time() - tp
        k = pos_k.shape[2]
        toks: List[int] = []
        LO = self.l_open
        with torch.no_grad(), sdpa_kernel(SDPA_BACKENDS):
            for _step in range(decode_tokens):
                if _step == 0:
                    lg = lg0
                else:
                    e = len(toks)
                    h_app = self.text_model.embed_tokens(
                        torch.tensor([toks], device=self.dev)
                    ).to(torch.bfloat16)
                    inc = torch.arange(1, e + 1, device=self.dev).view(1, 1, e)
                    pos_step = torch.cat([pos_k, pos_last + inc], dim=2)
                    lo_s = ext_mask(lo_t, e).to(self.dev).to(torch.float32).view(1, 1, k + e, k + e)
                    hi_s = ext_mask(hi_t, e).to(self.dev).to(torch.float32).view(1, 1, k + e, k + e)
                    cos_, sin_ = self.text_model.rotary_emb(h_app, pos_step)
                    pe_s = (cos_.to(h_app.dtype), sin_.to(h_app.dtype))
                    hh = h_app
                    for li, ly in enumerate(self.layers):
                        hin = torch.cat([caches[li].unsqueeze(0), hh], dim=1)
                        hout = ly(
                            hin, attention_mask=(lo_s if li < LO else hi_s), position_embeddings=pe_s
                        )[0]
                        hh = hout[:, k:]
                    hn = self.text_model.norm(hh)
                    lg = self.model.lm_head(hn[0, -1].to(self.model.lm_head.weight.dtype)).float()
                t = selector(lg, toks) if selector is not None else int(lg.argmax())
                if t == self.tok.eos_token_id:
                    break
                toks.append(t)
        text = self.tok.decode(toks)
        return parse_scratchpad_answer(text, fmt), text, toks, prefill_s

    # ----------------------------------------------------- cached training (lo phase)

    def build_training_cache(
        self,
        rec: Dict[str, Any],
        tgt_ids: List[int],
        *,
        anch: Optional[List[Tuple[int, int]]] = None,
        truncate: bool = False,
    ) -> Dict[str, Any]:
        """Run the fenced lo phase ONCE over [prompt || teacher-forced target rows] with
        the FROZEN e_c baked in, and cache h_{L_OPEN} (bf16, CPU). With truncate=True the
        cache keeps only [question]+[carriers]+[tail]+target rows (~100x smaller) and the
        target rows never see frames (deploy-matched teacher forcing)."""
        seq = rec["seq"]
        blocks, cpos, fin_start = rec["blocks"], rec["cpos"], rec["fin"]
        NF = len(cpos)
        e_len = len(tgt_ids)
        mask_lo, _ = make_masks(seq, blocks, cpos, fin_start)
        mask_lo = ext_mask(mask_lo, e_len)
        keep = None
        if truncate:
            keep = keep_cols(seq, blocks, cpos)
            if e_len:
                mask_lo[seq:, torch.tensor(frame_cols(seq, blocks, cpos), dtype=torch.long)] = MASK_MIN
        pos = rec["pos"].clone()
        with torch.no_grad():
            if e_len:
                inc = torch.arange(1, e_len + 1, device=pos.device).view(1, 1, e_len).expand(3, 1, e_len)
                pos = torch.cat([pos, pos[:, :, -1:] + inc], dim=2)
                if anch is not None:  # E-G coupled target positions (both phases)
                    cvals = [int(pos[0, 0, c]) for c in cpos]
                    pos[:, :, seq:] = torch.tensor(
                        [cvals[a - 1] + o for a, o in anch], device=pos.device
                    ).view(1, 1, e_len).expand(3, 1, e_len)
            emb = rec["emb"].to(self.dev).unsqueeze(0).clone()
            emb[0, torch.tensor(cpos, device=self.dev)] = self.e_c.detach().to(emb.dtype)
            if e_len:
                text = self.text_model.embed_tokens(torch.tensor([tgt_ids], device=self.dev))
                emb = torch.cat([emb, text.to(emb.dtype)], dim=1)
            # fp32 mask (NOT emb.dtype): unquantized norms are fp32 in this runtime, so
            # queries upcast to fp32 inside the layer; SDPA rejects bf16-mask+fp32-query.
            # Matches every other mask cast in this file (decode, EFFICIENT, top_hidden).
            lo4 = mask_lo.to(self.dev).to(torch.float32).view(1, 1, seq + e_len, seq + e_len)
            cos_, sin_ = self.text_model.rotary_emb(emb, pos.to(self.dev))
            pe = (cos_.to(emb.dtype), sin_.to(emb.dtype))
            h = emb
            with sdpa_kernel(SDPA_BACKENDS):
                for ly in self.layers[: self.l_open]:
                    h = ly(h, attention_mask=lo4, position_embeddings=pe)[0]
        if truncate:
            kt = torch.tensor(keep + list(range(seq, seq + e_len)), device=h.device)
            h = h.index_select(1, kt)
            pos = pos.index_select(2, kt.to(pos.device))
            a0t = blocks[0][0]
            cpos = list(range(a0t, a0t + NF))
            fin_start = a0t + NF
            seq = len(keep)
        return {
            "h": h[0].to(torch.bfloat16).cpu(),
            "pos": pos.cpu(),
            "blocks": blocks,
            "cpos": cpos,
            "fin": fin_start,
            "seq": seq,
            "e": e_len,
            "tgt": tgt_ids,
            "blk0max": rec["blk0max"],
            "anch": anch,
            "gold": rec["gold"],
            "task": rec["task"],
            "grp": rec["grp"],
            "trunc": truncate,
        }

    def top_hidden(self, d: Dict[str, Any], *, jitter_gap: int = 0, grad_ckpt: bool = False):
        """hi phase over cached h_{L_OPEN} (prompt + teacher-forced target rows).
        jitter_gap>1 (train only): per-step carrier gap stretch ~U{1..G}."""
        seq = d["seq"]
        e = d.get("e", 0)
        h = d["h"].to(self.dev).unsqueeze(0)
        if d.get("trunc"):
            _, mask_hi = truncated_masks(list(range(seq)), d["cpos"])
        else:
            _, mask_hi = make_masks(seq, d["blocks"], d["cpos"], d["fin"])
        # fp32 mask: matches EFFICIENT exactly and is universally accepted by MATH
        hi4 = ext_mask(mask_hi, e).to(self.dev).to(torch.float32).view(1, 1, seq + e, seq + e)
        pos = d["pos"].to(self.dev)
        if jitter_gap > 1:
            pos = pos.clone()
            NF = len(d["cpos"])
            cum = torch.cumsum(torch.randint(1, jitter_gap + 1, (NF,)), 0)
            for i, c in enumerate(d["cpos"]):
                pos[:, :, c] = d["blk0max"] + int(cum[i])
            pos[:, :, d["fin"] :] += int(cum[-1]) - NF
        if d.get("anch") is not None and e:
            cvals = [int(pos[0, 0, c]) for c in d["cpos"]]
            pos[:, :, d["seq"] :] = torch.tensor(
                [cvals[a - 1] + o for a, o in d["anch"]], device=pos.device
            ).view(1, 1, e).expand(3, 1, e)
        cos_, sin_ = self.text_model.rotary_emb(h, pos)
        pe = (cos_.to(h.dtype), sin_.to(h.dtype))
        use_ckpt = grad_ckpt and torch.is_grad_enabled()

        def _blk(hh, _ly):
            # sdpa_kernel INSIDE: checkpoint recomputation runs outside the caller's
            # context manager — the backend must be re-selected there too
            with sdpa_kernel(SDPA_BACKENDS):
                return _ly(hh, attention_mask=hi4, position_embeddings=pe)[0]

        for ly in self.layers[self.l_open :]:
            if use_ckpt:
                h = torch.utils.checkpoint.checkpoint(_blk, h, ly, use_reentrant=False)
            else:
                h = _blk(h, ly)
        return self.text_model.norm(h)

    def head(self, hs: torch.Tensor) -> torch.Tensor:
        return self.model.lm_head(hs.to(self.model.lm_head.weight.dtype)).float()

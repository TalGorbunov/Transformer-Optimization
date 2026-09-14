"""condmask model plumbing: raw text-Qwen loading + the batched masked forward.

The forward mirrors gnnformer.learnmask.gated_stack_logits (manual per-layer loop,
additive mask straight into the layer, sdpa_kernel(EFFICIENT, MATH)) generalized to a
BATCH with per-sample (B, H, S, S) masks, and written version-robust
(`out[0] if isinstance(out, tuple) else out`) so it runs on transformers 4.x and 5.x.

Gate conditioning during decode: gates are computed ONCE from the PROMPT and stay fixed
for all decode steps (the decision is conditioned on the input, not on the model's own
continuation).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from masks import layout_mask_parts  # noqa: E402  (scripts/condmask on sys.path)

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    SDPA_BACKENDS = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
except Exception:  # very old torch — plain context
    import contextlib
    sdpa_kernel = lambda *_: contextlib.nullcontext()  # noqa: E731
    SDPA_BACKENDS = []


def load_text_model(name: str, device: str = "cuda",
                    dtype: torch.dtype = torch.bfloat16, **cfg_overrides):
    """-> (tok, model). Frozen, eval mode, sdpa. Raw AutoModelForCausalLM — the
    precedent is scripts/gating/probe_text_triple.py:253-256. cfg_overrides are
    forwarded to from_pretrained (e.g. rope_scaling for YaRN beyond the native window)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=dtype, attn_implementation="sdpa",
        **cfg_overrides).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


def model_parts(model) -> Tuple[Any, Any, Any, Any, Any]:
    inner = model.model
    return inner.embed_tokens, inner.layers, inner.norm, model.lm_head, inner.rotary_emb


def model_dims(model) -> Tuple[int, int, int]:
    """(n_layers, n_heads, d_model) from config — never hard-coded."""
    cfg = model.config
    return cfg.num_hidden_layers, cfg.num_attention_heads, cfg.hidden_size


def assert_lora_targets(model) -> None:
    """Blocker check 4: q/k/v/o_proj must exist before attach_lora."""
    attn = model.model.layers[0].self_attn
    for nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
        if not hasattr(attn, nm):
            raise SystemExit(f"LoRA target {nm} missing on {type(attn).__name__}")


def pad_batch(recs: List[Dict[str, Any]], pad_id: int, device: Any
              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """-> (ids [B,S], valid [B,S] bool, ans_pos [B], S). ans_pos = last real token.

    S is padded to a multiple of 8: SDPA's mem-efficient backward requires the LSE
    buffer's head stride to be 32-byte aligned, and mask-grad-ONLY backward (the
    gates-only arms E/F) hard-errors on unaligned S ("LSE is not correctly aligned").
    Pad rows/cols are fully masked, so this is numerically inert."""
    S = (max(r["seq"] for r in recs) + 7) // 8 * 8
    B = len(recs)
    ids = torch.full((B, S), pad_id, dtype=torch.long)
    valid = torch.zeros(B, S, dtype=torch.bool)
    ans = torch.zeros(B, dtype=torch.long)
    for b, r in enumerate(recs):
        ids[b, : r["seq"]] = torch.tensor(r["ids"], dtype=torch.long)
        valid[b, : r["seq"]] = True
        ans[b] = r["seq"] - 1
    return ids.to(device), valid.to(device), ans.to(device), S


def forward_answer_logits(
    model,
    ids: torch.Tensor,
    ans_pos: torch.Tensor,
    layer_masks: Callable[[int], torch.Tensor],
    math_only: bool = False,
    pos_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Full-stack forward under per-layer additive masks -> fp32 logits [B, V] at the
    answer positions. layer_masks(li) returns [B, H or 1, S, S] in the model dtype.

    math_only: force the MATH SDPA backend on gradient-carrying forwards — the
    mem-efficient backward's bias-grad-ONLY path (gates-only arms E/F: frozen q/k/v,
    only the mask requires grad) hard-errors with "LSE is not correctly aligned
    (strideH)" regardless of padding (measured 2026-08-17); with q/k/v grads present
    (LoRA arms) EFFICIENT works fine."""
    embed, layers, norm, head, rotary = model_parts(model)
    B, S = ids.shape
    h = embed(ids)
    if pos_ids is None:
        pos_ids = torch.arange(S, device=ids.device).unsqueeze(0).expand(B, -1)
    cos, sin = rotary(h, pos_ids)
    pe = (cos.to(h.dtype), sin.to(h.dtype))
    backends = ([SDPBackend.MATH] if (math_only and torch.is_grad_enabled())
                else SDPA_BACKENDS)
    for li, layer in enumerate(layers):
        try:
            m = layer_masks(li, h)     # dynamic masks: conditioned on h^{l-1}
        except TypeError:
            m = layer_masks(li)        # static builders keep the old signature
        with sdpa_kernel(backends):
            out = layer(h, attention_mask=m, position_ids=pos_ids,
                        position_embeddings=pe)
        h = out[0] if isinstance(out, tuple) else out
    rows = norm(h[torch.arange(B, device=ids.device), ans_pos])
    return head(rows).float()


def greedy_digit_decode(
    model,
    tok,
    rec: Dict[str, Any],
    mask_builder: Callable[[Dict[str, Any], int], Callable[[int], torch.Tensor]],
    max_new: int = 4,
    device: Any = "cuda",
) -> Tuple[Optional[int], int, str]:
    """Batch-1 free greedy decode, recompute-from-scratch per step (repo convention),
    stop at first non-digit. mask_builder(rec, S) -> layer_masks callable for the
    EXTENDED record (appended tokens are tail rows). -> (parsed int|None,
    digit-restricted first-token argmax, text)."""
    digit_ids = [tok(str(d), add_special_tokens=False).input_ids[0] for d in range(10)]
    cur = dict(rec)
    toks: List[int] = []
    first_digit = -1
    with torch.no_grad():
        for step in range(max_new):
            ids = torch.tensor([cur["ids"]], dtype=torch.long, device=device)
            ans = torch.tensor([cur["seq"] - 1], device=device)
            lg = forward_answer_logits(model, ids, ans,
                                       mask_builder(cur, cur["seq"]))[0]
            if step == 0:
                first_digit = int(torch.stack([lg[t] for t in digit_ids]).argmax())
            t = int(lg.argmax())
            if not tok.decode([t]).strip().isdigit():
                break
            toks.append(t)
            cur = dict(cur, ids=cur["ids"] + [t], seq=cur["seq"] + 1)
    text = tok.decode(toks).strip()
    return (int(text) if text.isdigit() else None), first_digit, text


def single_mask_builder(g_row: torch.Tensor, K: float, dtype: torch.dtype,
                        device: Any) -> Callable:
    """mask_builder for greedy_digit_decode: fixed per-sample gates g_row [L, H] (or
    [L, 1] for fixed regimes), masks rebuilt on the extended layout each step."""
    def build(rec: Dict[str, Any], S: int) -> Callable[[int], torch.Tensor]:
        base, template = layout_mask_parts(rec, S, device)
        def layer_masks(li: int) -> torch.Tensor:
            g = g_row[li].view(1, -1, 1, 1).to(device)
            m = base.view(1, 1, S, S) + g * K * template.view(1, 1, S, S).float()
            return m.to(dtype)
        return layer_masks
    return build

"""The batched fenced path: the fence with no mask at all.

The dense path (core.fence) puts all N frames in ONE sequence and forbids cross-frame attention
with an [S, S] mask, so it is bound by the same total-token limit as the unfenced model
(N x tokens-per-frame in one forward). But a fenced block only ever attends the prefix and
itself, so the same computation factorises (rewrite plan, Phase 3b):

    1. prefix     one forward over the shared prefix (system + question under question-first)
                  -> its key/value cache.
    2. blocks     frames in chunks of B: each block is a row of a batch, the prefix cache is
                  expanded over the batch, every row gets BLOCK 0's positions (= the per-block
                  position reset). Plain causal attention per row, no mask, any SDPA kernel.
                  The slot state (the frame's end token) is read here — this is where the gate
                  reads — and the block's key/values are kept only for the blocks the gate keeps.
    3. read       the tail (the answer prompt; under the paper layout also the question) is run
                  over [prefix cache + kept blocks' caches], positions continuing after block 0's
                  maximum, then greedy decoding WITH the cache.

Cost: blocks are O(N) forwards of (prefix + one frame) tokens; the read sees prefix + k kept
frames whatever N was. Frames can therefore stay at the video's own resolution and N can exceed
what any single context holds.

Equivalence with the dense path (same rows attend the same columns with the same positions) is
pinned by tests/test_fastpath.py on a tiny randomly initialised model in float32 (slot states and
next-token logits to 1e-4, decoded tokens identical) and by a GPU parity run on MMReD.

Everything model-specific comes from the BackboneSpec and from the model's own code: block
embeddings are taken from the model's own image-merge (a pre-hook on the language model captures
`inputs_embeds`), positions from core.backbones.base.base_positions + core.fence.reset_positions
on a two-frame template prompt. Requirements: every frame of a sample has the same size (true
within one video / one MMReD row), causal image attention and global layers (Qwen2.5-VL,
InternVL3.5); Gemma 3's bidirectional image attention and local layers need the masked dense path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch

from .backbones import base as bb
from .backbones.base import BackboneSpec
from .fence import reset_positions
from .prompt import build_messages

KV = List[Tuple[torch.Tensor, torch.Tensor]]          # per layer (key, value), each [batch, heads, seq, dim]


class _Captured(Exception):
    """Raised by the capture hook to stop the forward once the merged embeddings are in hand."""


def unwrap(model: Any) -> Any:
    """The HF model under a PEFT wrapper (LoRA layers stay inside its modules)."""
    base = getattr(model, "get_base_model", None)
    return base() if callable(base) else model


def language_model(model: Any) -> Any:
    """The text decoder that receives `inputs_embeds` (model.model.language_model on the HF VLMs)."""
    m = unwrap(model)
    for path in (("model", "language_model"), ("language_model",)):
        obj = m
        for name in path:
            obj = getattr(obj, name, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "layers"):
            return obj
    raise RuntimeError("couldn't find the language model on this backbone")


def lm_head(model: Any) -> Any:
    head = getattr(unwrap(model), "lm_head", None)
    if head is None:
        raise RuntimeError("couldn't find lm_head on this backbone")
    return head


def merged_embeds(model: Any, enc: Dict[str, Any]) -> torch.Tensor:
    """[1, L, H] input embeddings of a processed prompt with the image features already merged
    in — computed by the model's OWN forward, stopped at the language model's door."""
    lm = language_model(model)
    box: Dict[str, torch.Tensor] = {}

    def hook(_m, _args, kwargs):
        box["e"] = kwargs["inputs_embeds"]
        raise _Captured

    h = lm.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        unwrap(model)(**enc, use_cache=False)
    except _Captured:
        pass
    finally:
        h.remove()
    if "e" not in box:
        raise RuntimeError("the language model was not called with inputs_embeds")
    return box["e"]


@dataclass
class Template:
    """Token layout, positions and text embeddings of one sample, from a TWO-frame prompt
    (inner block = its first block, last block = its second; the tail after the reset)."""
    prefix_embeds: torch.Tensor          # [1, P, H]
    tail_embeds: torch.Tensor            # [1, Tt, H]
    inner_ids: torch.Tensor              # [T] token ids of a block that is followed by another block
    last_ids: torch.Tensor               # [T] token ids of the final block
    pos_prefix: torch.Tensor             # [..., P]
    pos_block: torch.Tensor              # [..., T]   block 0's positions (every block gets them)
    pos_tail: torch.Tensor               # [..., Tt]
    pad_id: int
    slot: int                            # offset of the slot (end token) inside a block

    @property
    def block_len(self) -> int:
        return int(self.inner_ids.shape[0])


@dataclass
class FastResult:
    text: str = ""
    new_ids: List[int] = field(default_factory=list)
    first_logits: Optional[torch.Tensor] = None               # [V] logits of the first generated token
    slot_states: Dict[int, torch.Tensor] = field(default_factory=dict)    # module index -> [N, H]
    pool_states: Dict[int, torch.Tensor] = field(default_factory=dict)    # module index -> [N, H], mean over the block's image tokens
    judge_logits: Optional[torch.Tensor] = None               # [N, len(judge_vocab)]: per-block judge, first answer token
    keep: List[bool] = field(default_factory=list)
    tokens_prefix: int = 0
    tokens_block: int = 0
    tokens_tail: int = 0
    tokens_read: int = 0                 # prefix + kept blocks + tail: what the read actually sees


def _open(frame: Any) -> Any:
    from PIL import Image

    return frame if isinstance(frame, Image.Image) else Image.open(frame).convert("RGB")


def make_template(spec: BackboneSpec, model: Any, base_model: Any, processor: Any, frames: Sequence[Any],
                  question_text: str, layout: str, system_prompt: Optional[str], device: Any) -> Template:
    two = [_open(frames[0]), _open(frames[1 if len(frames) > 1 else 0])]
    if two[0].size != two[1].size:
        raise ValueError(f"frames differ in size ({two[0].size} vs {two[1].size}); the batched path needs one size per sample")
    enc = bb.encode(spec, processor, build_messages(two, question_text, layout=layout, system_prompt=system_prompt), device)
    ids = enc["input_ids"][0]
    sid = bb.special_ids(spec, processor)
    blocks, fin = bb.blocks(spec, ids.cpu(), sid, layout)
    if len(blocks) != 2 or blocks[0][1] - blocks[0][0] != blocks[1][1] - blocks[1][0] or blocks[0][1] != blocks[1][0]:
        raise ValueError(f"unexpected two-frame layout: {blocks}")
    (a0, b0), (a1, b1) = blocks
    with torch.no_grad():
        pos = reset_positions(bb.base_positions(spec, base_model, enc), blocks, fin)
        emb = merged_embeds(model, enc)
    return Template(prefix_embeds=emb[:, :a0], tail_embeds=emb[:, fin:], inner_ids=ids[a0:b0].clone(), last_ids=ids[a1:b1].clone(),
                    pos_prefix=pos[..., :a0], pos_block=pos[..., a0:b0], pos_tail=pos[..., fin:], pad_id=int(sid.image_pad),
                    slot=bb.slots([(0, b0 - a0)], spec)[0])


def block_embeds(spec: BackboneSpec, model: Any, processor: Any, frames: Sequence[Any], tpl: Template,
                 is_last: Sequence[bool], device: Any) -> torch.Tensor:
    """[B, T, H] embeddings of B blocks: the image features come from the model's own merge on a
    frames-only prompt, the marker / separator tokens from the template's ids (inner or last)."""
    imgs = [_open(f) for f in frames]
    msgs = [{"role": "user", "content": [{"type": "image", "image": im} for im in imgs]}]
    enc = dict(processor.apply_chat_template(msgs, add_generation_prompt=False, tokenize=True, return_dict=True, return_tensors="pt"))
    enc = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in enc.items()}
    ids = enc["input_ids"][0]
    runs = (ids == tpl.pad_id).nonzero().flatten()
    n_img = int((tpl.inner_ids == tpl.pad_id).sum())
    if runs.numel() != n_img * len(imgs):
        raise ValueError(f"{runs.numel()} image tokens for {len(imgs)} frames, expected {n_img} each (frame sizes differ?)")
    emb = merged_embeds(model, enc)[0]                                        # [L, H]
    feats = emb[runs].view(len(imgs), n_img, -1)                              # per frame, in order
    embed = unwrap(model).get_input_embeddings()
    rows = []
    for i, last in enumerate(is_last):
        tok = (tpl.last_ids if last else tpl.inner_ids).to(device)
        e = embed(tok).to(emb.dtype).clone()
        e[tok == tpl.pad_id] = feats[i]
        rows.append(e)
    return torch.stack(rows)


def _expand(kv: KV, batch: int) -> Any:
    from transformers import DynamicCache

    return DynamicCache(ddp_cache_data=[(k.expand(batch, -1, -1, -1), v.expand(batch, -1, -1, -1)) for k, v in kv])


def _concat(parts: Sequence[KV]) -> Any:
    from transformers import DynamicCache

    n_layers = len(parts[0])
    return DynamicCache(ddp_cache_data=[(torch.cat([p[l][0] for p in parts], dim=2), torch.cat([p[l][1] for p in parts], dim=2))
                                        for l in range(n_layers)])


def _legacy(cache: Any) -> KV:
    """Per-layer (key, value) of the layers that were written (a DynamicCache pre-allocates one
    slot per configured layer; unused slots hold None)."""
    if hasattr(cache, "layers"):
        return [(ly.keys, ly.values) for ly in cache.layers if getattr(ly, "keys", None) is not None]
    return [(k, v) for k, v in cache.to_legacy_cache()]


def _rep_pos(pos: torch.Tensor, batch: int) -> torch.Tensor:
    """Positions [..., 1, T] -> [..., B, T] (M-RoPE [3, 1, T] or 1-D [1, T])."""
    shape = list(pos.shape)
    shape[-2] = batch
    return pos.expand(*shape)


@torch.inference_mode()
def fenced_forward(spec: BackboneSpec, model: Any, base_model: Any, processor: Any, frames: Sequence[Any],
                   question_text: str, *, layout: str = "question-first", system_prompt: Optional[str] = None,
                   keep: Union[None, Sequence[bool], Callable[[Dict[int, torch.Tensor], int], Sequence[bool]]] = None,
                   capture_layers: Sequence[int] = (), decode: bool = True, max_new: int = 24,
                   stop_fn: Optional[Callable[[str], bool]] = None, chunk_tokens: int = 16000,
                   max_read_tokens: int = 0, pool_layers: Sequence[int] = (),
                   judge_ids: Optional[Sequence[int]] = None, judge_vocab: Sequence[int] = ()) -> FastResult:
    """One fenced sample through the batched path. `keep`: None = every block is read (fenced, no
    gate); a list = the gate's decision per frame (oracle labels, or a fitted gate's output); a
    callable(slot_states_of_chunk, first_index) -> flags = a gate that reads the slot states as the
    chunk is encoded. `capture_layers` are decoder MODULE indices (FenceHooks convention);
    their slot states are returned [N, H]. `pool_layers` (same convention) return the mean of each
    block's IMAGE-token states [N, H]. decode=False stops after the blocks (captures).
    `judge_ids` (token ids of an instruction such as "Does this frame ...? Answer yes or no.")
    runs a per-block judge: [prefix cache + block t] + instruction + the assistant opener, batched
    over the blocks, and returns the first answer token's logits at `judge_vocab` [N, |vocab|] —
    the block sees the question (prefix), the judge sees the block; nothing else is read.
    `max_read_tokens` refuses (ValueError) a read whose prefix + kept blocks + tail would exceed it."""
    if spec.image_attention != "causal" or spec.layer_types is not None:
        raise NotImplementedError(f"{spec.name}: the batched path needs causal image attention and global layers")
    device = next(unwrap(model).parameters()).device
    lm, head = language_model(model), lm_head(model)
    tok = processor.tokenizer
    n = len(frames)
    tpl = make_template(spec, model, base_model, processor, frames, question_text, layout, system_prompt, device)
    T, P = tpl.block_len, int(tpl.prefix_embeds.shape[1])
    res = FastResult(tokens_prefix=P, tokens_block=T, tokens_tail=int(tpl.tail_embeds.shape[1]))
    if decode and max_read_tokens and not callable(keep):
        k = n if keep is None else sum(bool(x) for x in keep)
        if P + k * T + res.tokens_tail > max_read_tokens:
            raise ValueError(f"read too long: prefix {P} + {k} kept blocks x {T} + tail {res.tokens_tail} > {max_read_tokens}")
    embed = unwrap(model).get_input_embeddings()
    judge_E = judge_pos = None
    if judge_ids is not None:
        if not judge_vocab:
            raise ValueError("judge_ids needs judge_vocab (the answer token ids to return)")
        jt = torch.tensor(list(judge_ids), dtype=torch.long, device=device)
        judge_E = torch.cat([embed(jt)[None].to(tpl.tail_embeds.dtype), tpl.tail_embeds], dim=1)       # [1, J + Tt, H]
        judge_pos = tpl.pos_tail[..., :1] + torch.arange(int(judge_E.shape[1]), device=tpl.pos_tail.device)  # continues after block 0
        jv = torch.tensor(list(judge_vocab), dtype=torch.long, device=device)

    out = lm(inputs_embeds=tpl.prefix_embeds, position_ids=tpl.pos_prefix.to(device), use_cache=True)
    prefix_kv = _legacy(out.past_key_values)

    layers = lm.layers
    grabbed: Dict[int, torch.Tensor] = {}
    hook_layers = sorted(set(capture_layers) | set(pool_layers))
    handles = [layers[L].register_forward_hook(
        (lambda L: lambda _m, _i, o: grabbed.__setitem__(L, (o[0] if isinstance(o, (tuple, list)) else o).detach()))(L))
        for L in hook_layers]
    img_mask = (tpl.inner_ids == tpl.pad_id).to(device)
    need_kv = decode
    kept_kv: List[KV] = []
    slot_rows: Dict[int, List[torch.Tensor]] = {L: [] for L in capture_layers}
    pool_rows: Dict[int, List[torch.Tensor]] = {L: [] for L in pool_layers}
    judge_rows: List[torch.Tensor] = []
    flags: List[bool] = [False] * n
    chunk = max(1, int(chunk_tokens) // max(1, T))
    # With the gate's decision known in advance and nothing to capture, a hidden block's forward
    # changes nothing downstream: only the kept frames are encoded (cost O(k), not O(N)).
    known = keep is not None and not callable(keep)
    capturing = bool(capture_layers or pool_layers) or judge_ids is not None
    want = [i for i in range(n) if keep[i]] if (known and not capturing) else list(range(n))
    try:
        for a in range(0, len(want), chunk):
            idx = want[a:a + chunk]
            E = block_embeds(spec, model, processor, [frames[i] for i in idx], tpl, [i == n - 1 for i in idx], device)
            B = E.shape[0]
            grabbed.clear()
            o = lm(inputs_embeds=E, position_ids=_rep_pos(tpl.pos_block.to(device), B), past_key_values=_expand(prefix_kv, B),
                   use_cache=True)
            states = {L: grabbed[L][:, tpl.slot].float() for L in capture_layers}
            for L in capture_layers:
                slot_rows[L].append(states[L].cpu())
            for L in pool_layers:
                pool_rows[L].append(grabbed[L][:, img_mask].float().mean(dim=1).cpu())
            if keep is None:
                kf = [True] * B
            elif callable(keep):
                kf = [bool(x) for x in keep(states, idx[0])]
            else:
                kf = [bool(keep[i]) for i in idx]
            for i, k in zip(idx, kf):
                flags[i] = k
            if need_kv and any(kf):
                kv = _legacy(o.past_key_values)
                for r, k in enumerate(kf):
                    if k:
                        kept_kv.append([(kk[r:r + 1, :, P:P + T].clone(), vv[r:r + 1, :, P:P + T].clone()) for kk, vv in kv])
            if judge_E is not None:      # appends to the chunk's cache; the kept blocks were sliced out above
                oj = lm(inputs_embeds=judge_E.expand(B, -1, -1), position_ids=_rep_pos(judge_pos.to(device), B),
                        past_key_values=o.past_key_values, use_cache=True)
                judge_rows.append(head(oj.last_hidden_state[:, -1])[:, jv].float().cpu())
                del oj
            del o, E
    finally:
        for h in handles:
            h.remove()
    res.keep = flags
    res.slot_states = {L: torch.cat(v) for L, v in slot_rows.items()} if capture_layers else {}
    res.pool_states = {L: torch.cat(v) for L, v in pool_rows.items()} if pool_layers else {}
    res.judge_logits = torch.cat(judge_rows) if judge_rows else None
    res.tokens_read = P + T * sum(flags) + res.tokens_tail
    if not decode:
        return res

    cache = _concat([prefix_kv] + kept_kv)
    o = lm(inputs_embeds=tpl.tail_embeds, position_ids=tpl.pos_tail.to(device), past_key_values=cache, use_cache=True)
    logits = head(o.last_hidden_state[:, -1])
    res.first_logits = logits[0].float().cpu()
    eos ={int(tok.eos_token_id)} | {int(i) for i in [tok.convert_tokens_to_ids(spec.delimiters.turn_end)] if i is not None}
    next_pos = int(tpl.pos_tail.max()) + 1
    ids: List[int] = []
    for step in range(max_new):
        nxt = int(logits[0].argmax())
        if nxt in eos:
            break
        ids.append(nxt)
        if stop_fn is not None and stop_fn(tok.decode(ids, skip_special_tokens=True)):
            break
        pos = torch.full_like(tpl.pos_tail[..., :1], next_pos + step).to(device)
        e = embed(torch.tensor([[nxt]], device=device)).to(tpl.tail_embeds.dtype)
        o = lm(inputs_embeds=e, position_ids=pos, past_key_values=o.past_key_values, use_cache=True)
        logits = head(o.last_hidden_state[:, -1])
    res.new_ids = ids
    res.text = tok.decode(ids, skip_special_tokens=True)
    return res

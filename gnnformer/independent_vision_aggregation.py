"""Software prototype: query visible images using frozen native visual features.

For each complete visible image, attention reads its ordinary Qwen2.5-VL visual
output, then a shared nonlinear map produces a message. Sum/mean merge before a
zero-initialized residual projection. This is a known structured-read pattern,
not an efficacy result or a claim of a new attention family.

Scope: batch one, unpadded contiguous causal inputs, images present in the
initial prefill, ordinary DynamicCache-style decoding. No video, beam batches,
mid-cache image insertion, inputs_embeds-only calls, gradient checkpointing,
or overlapping/interleaved requests on one attached model. The visual encoder
must already be frozen. Runtime image memory is detached, ephemeral, and absent
from state_dict; only trainable branch parameters are serialized.

The final visual MODULE output is used, after Qwen reverses its window packing.
Hooking visual.merger itself would capture patches in the wrong order. Native
QKV, RoPE, attention, KV updates and visual encoding are never called again.
"""
from __future__ import annotations

import math
import weakref

import torch
from torch import nn
from torch.nn import functional as F


def _require(condition, message):
    if not condition:
        raise ValueError(message)


class IndependentVisionAggregation(nn.Module):
    """Per-image residual, with exact zero-read subtraction in branch precision.

    q = Wq RMS(h), k_i = v_i = Wmem RMS(visual_i)
    read_i = softmax(q k_i.T / sqrt(rank)) v_i
    message_i = SiLU(Wr read_i + b + q) - SiLU(b + q)
    delta = U merge_i(message_i), over complete visible images only.

    Wq, Wmem and U have no bias. Wr has one shared bias. RMS has no trainable
    affine term and epsilon1e-6. Memory values are normalized per visual token,
    never jointly across images. Hidden and memory widths are the same native
    decoder width. Sum and mean have exactly identical parameterization.
    """

    def __init__(self, hidden_size: int, *, rank: int = 96, merge: str = "sum",
                 query_chunk_size: int = 64):
        super().__init__()
        _require(hidden_size > 0 and rank > 0 and query_chunk_size > 0, "Dimensions must be positive")
        _require(merge in ("sum", "mean"), "merge must be sum or mean")
        self.hidden_size, self.rank = hidden_size, rank
        self.merge, self.query_chunk_size = merge, query_chunk_size
        self.query = nn.Linear(hidden_size, rank, bias=False)
        self.memory = nn.Linear(hidden_size, rank, bias=False)
        self.read = nn.Linear(rank, rank, bias=True)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.zeros_(self.up.weight)
        self.mode = "all"
        self.record_stats = True
        self.capture_last_query_messages = False
        self.last_call_stats = None
        self.last_query_diagnostics = None
        self._controller = None

    @staticmethod
    def rms(x):
        x = x.float()
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6)

    def forward(self, hidden, image_memory, image_ends, query_positions, language_mask):
        _require(hidden.ndim == 3 and hidden.shape[0] == 1 and hidden.shape[-1] == self.hidden_size,
                 "Independent image reads require B1 native hidden states")
        _require(self.mode in ("all", "off"), "Unknown branch mode")
        _require(self.query.weight.dtype == torch.float32, "Prototype branch parameters must remain fp32")
        length = hidden.shape[1]
        _require(query_positions.shape == (length,) and language_mask.shape == (length,)
                 and language_mask.dtype == torch.bool, "Query metadata shape/type mismatch")
        _require(image_ends.ndim == 1 and len(image_memory) == len(image_ends), "Image metadata mismatch")
        _require(query_positions.device == hidden.device == language_mask.device == image_ends.device,
                 "Metadata and hidden states must share a device")
        _require(self.merge in ("sum", "mean"), "merge must be sum or mean")
        self.last_call_stats = self.last_query_diagnostics = None
        if self.mode == "off" or not image_memory or not bool(language_mask.any()):
            return torch.zeros_like(hidden)
        language_indices = torch.nonzero(language_mask, as_tuple=False).flatten()
        # Project ONLY language queries. Image-pad/delimiter positions neither
        # read memory nor incur native-width query/up projection work.
        language_query = self.query(self.rms(hidden[0].index_select(0, language_indices)))
        q = language_query.new_zeros(length, self.rank).index_copy(0, language_indices, language_query)
        total = q.new_zeros(length, self.rank)
        count = q.new_zeros(length, 1)
        norm_total = q.new_zeros(length, 1) if self.record_stats else None
        capture = self.capture_last_query_messages and bool(language_mask[-1])
        last_messages = []
        for memory, end in zip(image_memory, image_ends):
            _require(memory.ndim == 2 and memory.shape[0] > 0
                     and memory.shape[1] == self.hidden_size and memory.device == hidden.device,
                     "Each image memory must be nonempty native-width visual tokens")
            # Strictly after the closing vision_end delimiter; visual delimiters
            # themselves are excluded from language_mask as well.
            selected = torch.nonzero(language_mask & (query_positions > end), as_tuple=False).flatten()
            if selected.numel() == 0:
                if capture:
                    last_messages.append(q.new_zeros(self.rank))
                continue
            kv = self.memory(self.rms(memory))
            messages = []
            for indices in selected.split(self.query_chunk_size):
                query = q.index_select(0, indices)
                weights = torch.softmax((query @ kv.transpose(0, 1)) / math.sqrt(self.rank), dim=-1)
                read = weights @ kv
                baseline = self.read.bias + query
                message = F.silu(self.read(read) + query) - F.silu(baseline)
                messages.append(message)
            message = torch.cat(messages, dim=0)
            total = total.index_add(0, selected, message)
            count = count.index_add(0, selected, q.new_ones(selected.numel(), 1))
            if self.record_stats:
                norm_total = norm_total.index_add(0, selected, message.detach().norm(dim=-1, keepdim=True))
            if capture:
                last_messages.append(message[-1].detach() if int(selected[-1]) == length - 1
                                     else q.new_zeros(self.rank))
        if self.merge == "mean":
            total = total / count.clamp_min(1)
        language_delta = self.up(total.index_select(0, language_indices)).to(hidden.dtype)
        delta = torch.zeros_like(hidden).index_copy(1, language_indices, language_delta.unsqueeze(0))
        if self.record_stats:
            residual_norm = q.new_zeros(length).index_copy(0, language_indices,
                language_delta.detach().float().norm(dim=-1))
            self.last_call_stats = dict(query_positions=query_positions.detach(),
                language_mask=language_mask.detach(), n_visible=count.detach().squeeze(-1),
                mean_message_norm=(norm_total / count.clamp_min(1)).detach().squeeze(-1),
                residual_norm=residual_norm.detach())
        if capture:
            vectors = torch.stack(last_messages).detach().clone()
            self.last_query_diagnostics = dict(query_position=query_positions[-1].detach().clone(),
                image_ends=image_ends.detach().clone(), visible=(query_positions[-1] > image_ends).detach(),
                frame_messages=vectors, frame_message_norms=vectors.norm(dim=-1),
                merged_message=total[-1].detach().clone(), residual=delta[0, -1].detach().clone())
        return delta

    def export_last_query_diagnostics(self, *, cpu=False):
        """Optional detached rank-space frame messages; never copies to CPU by default.

        Set capture_last_query_messages=True before a forward to record only its
        final language query. These are descriptive contributions, not frame
        labels or a causal decomposition of the contextual language hidden state.
        """
        if self.last_query_diagnostics is None:
            return None
        return {key: value.detach().cpu().clone() if cpu else value.detach().clone()
                for key, value in self.last_query_diagnostics.items()}

    def reset_memory(self):
        if self._controller is not None:
            self._controller.reset()

    @property
    def memory_image_count(self):
        return 0 if self._controller is None else len(self._controller.image_memory or ())

    def remove(self):
        if self._controller is not None:
            self._controller.remove()


def image_layout(input_ids, image_grid_thw, *, image_token_id, vision_start_token_id,
                 vision_end_token_id, video_token_id, spatial_merge_size):
    """Return image patch counts, closing-delimiter positions and language mask.

    Only canonical complete <vision_start><image_pad>...<vision_end> runs are
    accepted. Per-image patch count is checked against the native merge grid.
    Host metadata parsing is deliberate in this bounded, noncompiled prototype.
    """
    _require(input_ids.ndim == 2 and input_ids.shape[0] == 1, "Batch/beam expansion is unsupported; B1 only")
    tokens = input_ids[0].detach().cpu().tolist()
    _require(video_token_id not in tokens, "Videos are unsupported")
    if image_grid_thw is None:
        grids = []
    else:
        _require(image_grid_thw.ndim == 2 and image_grid_thw.shape[1] == 3, "Invalid image_grid_thw")
        grids = image_grid_thw.detach().cpu().tolist()
    sizes = []
    for t, h, w in grids:
        _require(t == 1 and h > 0 and w > 0 and h % spatial_merge_size == 0
                 and w % spatial_merge_size == 0, "Only single-image, divisible native grids are supported")
        sizes.append(int(h * w // spatial_merge_size**2))
    ends, consumed, cursor = [], 0, 0
    while cursor < len(tokens):
        token = tokens[cursor]
        if token == vision_start_token_id:
            _require(consumed < len(sizes), "Vision start has no matching image grid")
            end = cursor + sizes[consumed] + 1
            _require(end < len(tokens) and tokens[end] == vision_end_token_id
                     and all(t == image_token_id for t in tokens[cursor + 1:end]),
                     "Image is incomplete or image-pad count differs from its grid")
            ends.append(end)
            consumed += 1
            cursor = end + 1
        else:
            _require(token not in (image_token_id, vision_end_token_id), "Unmatched image/vision delimiter")
            cursor += 1
    _require(consumed == len(grids), "Image grids do not match complete image spans")
    language = ~((input_ids[0] == image_token_id) | (input_ids[0] == vision_start_token_id)
                 | (input_ids[0] == vision_end_token_id) | (input_ids[0] == video_token_id))
    return sizes, torch.tensor(ends, dtype=torch.long, device=input_ids.device), language


class _ImageMemoryHooks:
    def __init__(self, model, attn, visual, branch, token_config):
        self.model_ref, self.attn_ref = weakref.ref(model), weakref.ref(attn)
        self.visual_ref, self.branch = weakref.ref(visual), branch
        self.token_config = token_config
        self.handles = []
        self.reset()

    def reset(self):
        self.image_memory = None
        self.image_ends = None
        self.image_sizes = []
        self.grid = None
        self.cache_ref = None
        self.total_length = 0
        self.active = False
        self.positions = self.language_mask = None
        self.expected_visual_calls = self.visual_calls = self.attn_calls = 0

    def before_model(self, model, args, kwargs):
        try:
            self.branch.last_call_stats = self.branch.last_query_diagnostics = None
            _require(not self.active, "Overlapping/reentrant model forwards are unsupported")
            _require(len(args) <= 1, "Pass optional model inputs by keyword")
            ids = args[0] if args else kwargs.get("input_ids")
            _require(isinstance(ids, torch.Tensor) and ids.ndim == 2 and ids.shape[0] == 1
                     and ids.shape[1] > 0, "Explicit nonempty B1 input_ids are required; no beams/batches")
            _require(kwargs.get("inputs_embeds") is None, "inputs_embeds-only/injected inputs are unsupported")
            _require(all(kwargs.get(name) is None for name in
                         ("pixel_values_videos", "video_grid_thw", "second_per_grid_ts")), "Videos are unsupported")
            _require(not any(getattr(module, "gradient_checkpointing", False) and module.training
                             for module in model.modules()), "Gradient checkpointing is unsupported")
            _require(self.branch.mode in ("all", "off"), "Unknown branch mode")
            past = kwargs.get("past_key_values")
            _require(past is None or hasattr(past, "get_seq_length"), "Legacy tuple caches are unsupported")
            past_length = 0 if past is None else int(past.get_seq_length())
            positions = kwargs.get("cache_position")
            expected = torch.arange(past_length, past_length + ids.shape[1], device=ids.device)
            if positions is not None:
                _require(positions.ndim == 1 and torch.equal(positions.to(expected), expected),
                         "Only contiguous native cache positions are supported")
            attention_mask = kwargs.get("attention_mask")
            if attention_mask is not None:
                _require(attention_mask.ndim == 2 and attention_mask.shape == (1, past_length + ids.shape[1])
                         and bool((attention_mask == 1).all()), "Only unpadded ordinary causal attention is supported")
            grid, pixels = kwargs.get("image_grid_thw"), kwargs.get("pixel_values")
            if past_length == 0:
                self.reset()
                sizes, ends, language = image_layout(ids, grid, **self.token_config)
                _require((pixels is not None) == bool(sizes), "Pixel inputs and image spans disagree")
                self.image_sizes, self.image_ends = sizes, ends
                self.grid = None if grid is None else grid.detach().clone()
                self.image_memory = None if sizes else ()
                self.expected_visual_calls = int(bool(sizes))
            else:
                _require(self.cache_ref is not None and self.cache_ref() is past
                         and self.total_length == past_length and self.image_memory is not None,
                         "Cache is foreign, stale, or has no matching image-memory prefill")
                _require(pixels is None, "Adding/re-encoding images inside a cache is unsupported")
                if grid is not None:
                    _require(self.grid is not None and torch.equal(grid.to(self.grid), self.grid),
                             "Cached image grid changed")
                _, _, language = image_layout(ids, None, **self.token_config)
                self.expected_visual_calls = 0
            self.positions, self.language_mask = expected, language
            self.active = True
            self.visual_calls = self.attn_calls = 0
            self.total_length = past_length + ids.shape[1]
        except Exception:
            self.reset()
            raise

    def after_visual(self, visual, args, kwargs, output):
        _require(self.active and self.expected_visual_calls == 1 and self.visual_calls == 0,
                 "Unexpected or repeated visual forward; memory capture requires one native prefill")
        _require(isinstance(output, torch.Tensor) and output.ndim == 2
                 and output.shape == (sum(self.image_sizes), self.branch.hidden_size),
                 "Native post-merger visual output does not match image spans")
        grid = kwargs.get("grid_thw", args[1] if len(args) > 1 else None)
        _require(grid is not None and self.grid is not None and torch.equal(grid.to(self.grid), self.grid),
                 "Native visual grid differs from top-forward metadata")
        # Requires frozen visual encoder; do not retain graphs between examples.
        self.image_memory = tuple(output.detach().split(self.image_sizes, dim=0))
        self.visual_calls += 1

    def after_attention(self, attn, args, kwargs, output):
        _require(self.active and self.image_memory is not None, "Attention called without current image metadata")
        self.attn_calls += 1
        _require(self.attn_calls == 1, "Selected native attention must execute exactly once per model forward")
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        _require(isinstance(hidden, torch.Tensor) and hidden.shape[:2] == (1, self.positions.numel()),
                 "Native query shape differs from top-forward input")
        _require(isinstance(output, tuple) and len(output) >= 1 and output[0].shape == hidden.shape,
                 "Unsupported native attention output")
        if self.branch.mode == "off" or not self.image_memory:
            return None  # Leave the exact native output object unchanged.
        delta = self.branch(hidden, self.image_memory, self.image_ends, self.positions, self.language_mask)
        # Receiving ordinary ATTENTION output, before this branch is added.
        # This is not the layer's residual stream. Stats are zero at positions
        # excluded from language queries; no full native-width CPU copy occurs.
        if self.branch.record_stats and self.branch.last_call_stats is not None:
            language_indices = torch.nonzero(self.language_mask, as_tuple=False).flatten()
            norms = output[0][0].detach().index_select(0, language_indices).float().norm(dim=-1)
            self.branch.last_call_stats["native_output_norm"] = norms.new_zeros(hidden.shape[1]).index_copy(
                0, language_indices, norms)
        if self.branch.last_query_diagnostics is not None:
            self.branch.last_query_diagnostics["native_output_norm"] = output[0][0, -1].detach().float().norm()
        return (output[0] + delta, *output[1:])

    def after_model(self, model, args, kwargs, output):
        if output is None:
            self.reset()
            return None
        try:
            _require(self.active and self.attn_calls == 1
                     and self.visual_calls == self.expected_visual_calls, "Native forward/capture count mismatch")
            cache = getattr(output, "past_key_values", None)
            if cache is None:
                self.reset()
            else:
                _require(hasattr(cache, "get_seq_length") and int(cache.get_seq_length()) == self.total_length,
                         "Native cache length differs after forward")
                self.cache_ref = weakref.ref(cache)
                self.active = False
                self.positions = self.language_mask = None
        except Exception:
            self.reset()
            raise
        return None

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        attn = self.attn_ref()
        if attn is not None and getattr(attn, "independent_vision_aggregation", None) is self.branch:
            delattr(attn, "independent_vision_aggregation")
        self.reset()
        self.branch._controller = None


def attach_independent_vision_aggregation(model, *, layer_index: int = 14, rank: int = 96,
                                         merge: str = "sum", query_chunk_size: int = 64):
    """Attach an image-memory branch without replacing any native forward.

    Freeze the visual encoder before attachment. The returned branch owns
    trainable weights, mode all/off, reset_memory(), remove(), and no checkpointed
    image memory. Requires the ordinary Qwen2.5-VL module/cache interface.
    """
    _require(getattr(model.config, "model_type", None) == "qwen2_5_vl", "Only Qwen2.5-VL is supported")
    inner = getattr(model, "model", None)
    _require(inner is not None and hasattr(inner, "visual") and hasattr(inner, "language_model"),
             "Expected the native Qwen visual/language module layout")
    layers, visual = inner.language_model.layers, inner.visual
    _require(0 <= layer_index < len(layers), "Invalid injection layer")
    attn = layers[layer_index].self_attn
    _require(getattr(attn.config, "_attn_implementation", None) == "sdpa"
             and getattr(attn, "sliding_window", None) is None, "Requires native full causal SDPA")
    _require(not any(p.requires_grad for p in visual.parameters()), "Freeze the visual encoder before attaching")
    _require(not any(hasattr(attn, name) for name in
                     ("native_aggregation", "hidden_only_adapter", "value_lifting", "independent_vision_aggregation")),
             "Another residual operator is already attached at this layer")
    token_config = {name: int(getattr(model.config, name)) for name in
                    ("image_token_id", "vision_start_token_id", "vision_end_token_id", "video_token_id")}
    token_config["spatial_merge_size"] = int(visual.spatial_merge_size)
    _require(token_config["spatial_merge_size"] > 0, "Invalid native spatial merge size")
    width = attn.q_proj.in_features
    branch = IndependentVisionAggregation(width, rank=rank, merge=merge, query_chunk_size=query_chunk_size)
    branch.to(device=attn.q_proj.weight.device, dtype=torch.float32)
    controller = _ImageMemoryHooks(model, attn, visual, branch, token_config)
    branch._controller = controller
    attn.add_module("independent_vision_aggregation", branch)
    try:
        controller.handles.append(model.register_forward_pre_hook(controller.before_model, with_kwargs=True))
        controller.handles.append(visual.register_forward_hook(controller.after_visual, with_kwargs=True))
        controller.handles.append(attn.register_forward_hook(controller.after_attention, with_kwargs=True))
        controller.handles.append(model.register_forward_hook(controller.after_model, with_kwargs=True, always_call=True))
    except Exception:
        controller.remove()
        raise
    return branch

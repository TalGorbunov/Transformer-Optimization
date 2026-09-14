"""Independent CPU geometry for a saved learned-selection native query.

This module neither imports the core/runtime nor runs a head or model. The
reporter supplies the frozen CPU state_dict, a current-query capture, and
offline room relevance and canonical Step labels. No labels enter the
functional reconstruction. Every returned value is JSON-safe; no tensors or
cross-call state are retained. Numerical execution belongs in CPU Slurm.
"""
from __future__ import annotations

MODES = ('clip', 'sigmoid', 'softmax')
WEIGHT_KEYS = ('query.weight', 'local.weight', 'local_bias',
               'aggregate_projection.weight', 'aggregate_projection.bias',
               'up.weight', 'selection_weight', 'selection_bias')
CAPTURE_KEYS = ('query', 'payload', 'scores', 'gates', 'messages', 'aggregate',
                'preactivation', 'delta', 'local_states', 'global_states',
                'native_query_hidden', 'native_delta', 'fused_global',
                'fused_query_hidden')


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _tensor(torch, value, shape, dtype, name):
    _need(isinstance(value, torch.Tensor) and tuple(value.shape) == tuple(shape)
          and value.dtype == dtype and value.device.type == 'cpu'
          and not value.requires_grad and bool(torch.isfinite(value).all()),
          name + ' must be a finite detached CPU tensor with the required shape/dtype')


def _gates(torch, scores, mode):
    # Independent implementation: never call the core's selection helper.
    if mode == 'clip':
        return scores.clamp(0, 1)
    if mode == 'sigmoid':
        return torch.sigmoid(4 * (scores - .5))
    return scores.clone() if scores.shape[0] == 0 else torch.softmax(4 * (scores - .5), dim=0)


def _norm(torch, value):
    return float(torch.linalg.vector_norm(value.double()))


def _difference(torch, actual, reference):
    delta = actual.double() - reference.double()
    reference_norm = _norm(torch, reference)
    return dict(exact=torch.equal(actual, reference),
                maximum_absolute=float(delta.abs().max()) if delta.numel() else 0.,
                l2=_norm(torch, delta), rms=float(delta.square().mean().sqrt()) if delta.numel() else None,
                relative_l2=_norm(torch, delta) / reference_norm if reference_norm else None)


def _range(torch, values):
    if not values.numel():
        return dict(minimum=None, maximum=None, mean=None)
    return dict(minimum=float(values.min()), maximum=float(values.max()), mean=float(values.double().mean()))


def _cosine(torch, first, second):
    scale = _norm(torch, first) * _norm(torch, second)
    return float((first.double() * second.double()).sum()) / scale if scale else None


def _functional(torch, cap, weights, mode):
    """CPU FP32 reconstruction; matrix/reduction drift is descriptive only."""
    linear = torch.nn.functional.linear
    def rms(value):
        value = value.float()
        return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + 1e-6)
    query = linear(rms(cap['global_states']), weights['query.weight'])
    payload = torch.tanh(linear(rms(cap['local_states']), weights['local.weight'])
                         + query.unsqueeze(0) + weights['local_bias'])
    scores = linear(payload, weights['selection_weight'].unsqueeze(0)).squeeze(-1) + weights['selection_bias']
    gates = _gates(torch, scores, mode)
    messages = gates.unsqueeze(-1) * payload
    aggregate = messages.sum(dim=0)
    preactivation = linear(aggregate, weights['aggregate_projection.weight'],
                           weights['aggregate_projection.bias']) + query
    delta = linear(torch.nn.functional.silu(preactivation), weights['up.weight'])
    return dict(query=query, payload=payload, scores=scores, gates=gates, messages=messages,
                aggregate=aggregate, preactivation=preactivation, delta=delta)


def analyze_capture(torch, cap, weights, mode, relevant_mask, steps):
    """Return compact statistics for one current native query (Q=1).

    ``cap`` is the runtime's saved tensor capture; ``weights`` is the final
    CPU FP32 state_dict. ``relevant_mask`` is a CPU bool[N] tensor or a list of
    literal bools. ``steps`` is a list/tuple of N distinct positive Python
    integers in the same image order. Neither label input affects functional
    reconstruction. The caller binds scene/prefix/checkpoint/source identity.

    Saved FP32 message multiplication and FP16 cast/add ownership are exact
    checks. Clip from saved scores is exact; sigmoid/softmax CPU recomputation
    is descriptive because their kernels may differ from GPU execution.
    P/Z partitions use saved FP32 messages promoted to FP64, and Wagg acts
    without its bias. Their algebra is checked at rtol=atol=1e-10. All other
    CPU-vs-captured matrix/reduction differences are descriptive, never gates.
    """
    _need(mode in MODES, 'Unknown selection mode')
    _need(isinstance(cap, dict) and set(CAPTURE_KEYS) <= set(cap), 'Incomplete native capture')
    _need(isinstance(weights, dict) and set(weights) == set(WEIGHT_KEYS), 'Unexpected final core state_dict')
    _need('mode' not in cap or cap['mode'] == mode, 'Capture mode differs')
    local = cap['local_states']; global_states = cap['global_states']; query = cap['query']
    _need(isinstance(local, torch.Tensor) and local.ndim == 3 and local.shape[1] == 1,
          'A single current-query local capture [N,1,H] is required')
    _need(isinstance(query, torch.Tensor) and query.ndim == 2 and query.shape[0] == 1,
          'A single current-query projection [1,R] is required')
    n, _, hidden = local.shape; rank = query.shape[1]
    _need(hidden > 0 and rank > 0, 'Hidden and payload dimensions must be positive')
    native_shapes = dict(local_states=(n, 1, hidden), global_states=(1, hidden),
                         native_query_hidden=(n+1, 1, hidden), native_delta=(1, hidden),
                         fused_global=(1, hidden), fused_query_hidden=(n+1, 1, hidden))
    fp32_shapes = dict(query=(1, rank), payload=(n, 1, rank), scores=(n, 1), gates=(n, 1),
                       messages=(n, 1, rank), aggregate=(1, rank),
                       preactivation=(1, rank), delta=(1, hidden))
    for name, shape in native_shapes.items():
        _tensor(torch, cap[name], shape, torch.float16, name)
    for name, shape in fp32_shapes.items():
        _tensor(torch, cap[name], shape, torch.float32, name)
    weight_shapes = {'query.weight': (rank, hidden), 'local.weight': (rank, hidden), 'local_bias': (rank,),
                     'aggregate_projection.weight': (rank, rank), 'aggregate_projection.bias': (rank,),
                     'up.weight': (hidden, rank), 'selection_weight': (rank,), 'selection_bias': (1,)}
    for name, shape in weight_shapes.items():
        _tensor(torch, weights[name], shape, torch.float32, name)
    if isinstance(relevant_mask, (list, tuple)):
        _need(len(relevant_mask) == n and all(type(v) is bool for v in relevant_mask),
              'Offline relevance must contain one literal boolean per actual image')
        relevant_mask = torch.tensor(relevant_mask, dtype=torch.bool)
    _need(isinstance(relevant_mask, torch.Tensor) and relevant_mask.dtype == torch.bool
          and relevant_mask.device.type == 'cpu' and tuple(relevant_mask.shape) == (n,),
          'Offline relevance must be CPU bool[N]')
    _need(isinstance(steps, (list, tuple)) and len(steps) == n
          and all(type(v) is int and v > 0 for v in steps) and len(set(steps)) == n,
          'Canonical Steps must be distinct positive integers aligned to actual images')
    with torch.no_grad():
        gates = cap['gates']; payload = cap['payload']; messages = cap['messages']
        _need(bool(((gates >= 0) & (gates <= 1)).all()) and bool((payload.abs() <= 1).all()),
              'Saved gates or bounded payload escaped their mathematical range')
        _need(torch.equal(messages, gates.unsqueeze(-1) * payload), 'Saved messages are not exact FP32 gate times payload')
        closed = gates == 0
        _need(bool((messages[closed] == 0).all()), 'A closed gate has nonzero message coordinates')
        origin = cap['native_query_hidden']
        _need(torch.equal(origin[:-1], local) and torch.equal(origin[-1], global_states),
              'Actual local/global capture ownership differs')
        _need(torch.equal(cap['native_delta'], cap['delta'].half()), 'Residual was not cast to native FP16 before addition')
        expected = origin.clone(); expected[-1] = global_states + cap['native_delta']
        _need(torch.equal(cap['fused_query_hidden'], expected) and torch.equal(cap['fused_global'], expected[-1]),
              'Native write changed local rows or differs from FP16 global addition')
        from_scores = _gates(torch, cap['scores'], mode)
        _need(bool(torch.isfinite(from_scores).all()), 'Independent score-to-gate recomputation is nonfinite')
        if mode == 'clip':
            _need(torch.equal(from_scores, gates), 'Clip gate differs from the exact saved-score clamp')
        rebuilt = _functional(torch, cap, weights, mode)
        _need(all(bool(torch.isfinite(value).all()) for value in rebuilt.values()), 'CPU functional reconstruction is nonfinite')
        functional = {name: _difference(torch, value, cap[name]) for name, value in rebuilt.items()}

        # Promote already multiplied messages: do not equate FP64 g*u with
        # a stored FP32 product, or mistake their multiplication roundoff for drift.
        message64 = messages[:, 0].double(); payload64 = payload[:, 0].double()
        weight64 = weights['aggregate_projection.weight'].double()
        positive = message64[relevant_mask].sum(0); negative = message64[~relevant_mask].sum(0)
        total = message64.sum(0); projected_positive = weight64 @ positive; projected_negative = weight64 @ negative
        projected_total = weight64 @ total
        _need(torch.allclose(positive + negative, total, rtol=1e-10, atol=1e-10), 'FP64 P/Z partition identity failed')
        _need(torch.allclose(projected_positive + projected_negative, projected_total, rtol=1e-10, atol=1e-10),
              'FP64 projected P/Z partition identity failed')
        early = torch.tensor([v <= 16 for v in steps], dtype=torch.bool)
        masks = dict(all=torch.ones(n, dtype=torch.bool), relevant=relevant_mask, irrelevant=~relevant_mask,
                     step_le16=early, step_gt16=~early,
                     relevant_step_le16=relevant_mask & early, relevant_step_gt16=relevant_mask & ~early,
                     irrelevant_step_le16=~relevant_mask & early, irrelevant_step_gt16=~relevant_mask & ~early)
        groups = {}
        for name, mask in masks.items():
            count = int(mask.sum()); group_gate = gates[:, 0][mask]; group_message = message64[mask]
            group_payload = payload64[mask]; msum = group_message.sum(0); psum = group_payload.sum(0)
            groups[name] = dict(count=count, gates=_range(torch, group_gate), scores=_range(torch, cap['scores'][:, 0][mask]),
                gate_sum=float(group_gate.double().sum()), closed_count=int((group_gate == 0).sum()),
                nonzero_count=int((group_gate != 0).sum()), unit_count=int((group_gate == 1).sum()),
                message_sum_norm=_norm(torch, msum), projected_message_sum_norm=_norm(torch, weight64 @ msum),
                payload_sum_norm=_norm(torch, psum), projected_payload_sum_norm=_norm(torch, weight64 @ psum),
                message_mean_norm=_norm(torch, msum) / count if count else None,
                payload_mean_norm=_norm(torch, psum) / count if count else None,
                mean_message_squared_norm=float(group_message.square().sum(1).mean()) if count else None,
                mean_payload_squared_norm=float(group_payload.square().sum(1).mean()) if count else None)
        relevant_sum = groups['relevant']['gate_sum']; irrelevant_sum = groups['irrelevant']['gate_sum']
        total_sum = groups['all']['gate_sum']
        product64 = gates[:, 0].double().unsqueeze(-1) * payload64
        return dict(schema_version=1, mode=mode, n_frames=n, queries=1, hidden_size=hidden, rank=rank,
            metadata=dict(scope='all supplied actual rows at this one executed current prefix',
                offline_relevance_only=True, step_boundary=16, no_model_or_head_calls=True,
                caller_binds_scene_prefix_checkpoint_and_source=True, projection='final Wagg without bias',
                geometry_dtype='FP64 promotion of saved FP32 messages and final FP32 Wagg',
                functional_dtype='CPU FP32; native ownership FP16',
                cpu_functional_differences_descriptive=True, no_scalar_ceiling_or_rank_conclusion=True),
            checks=dict(finite_shapes=True, messages_exact=True, closed_message_coordinates_exact_zero=True,
                native_delta_cast_exact=True, local_rows_unchanged=True, global_native_add_exact=True,
                clip_from_scores_exact=True if mode == 'clip' else None, partition_identity_passed=True,
                projected_partition_identity_passed=True, partition_rtol=1e-10, partition_atol=1e-10),
            score_gate_recomputation=dict(difference=_difference(torch, from_scores, gates),
                descriptive=mode != 'clip', implementation='independent clamp / sigmoid / dim0 softmax'),
            cpu_functional_differences=functional,
            selection=dict(relevant_count=int(relevant_mask.sum()), irrelevant_count=int((~relevant_mask).sum()),
                gate_sum=total_sum, relevant_gate_sum=relevant_sum, irrelevant_gate_sum=irrelevant_sum,
                irrelevant_fraction_of_gate_sum=irrelevant_sum / total_sum if total_sum else None,
                closed_relevant_count=groups['relevant']['closed_count'], nonzero_irrelevant_count=groups['irrelevant']['nonzero_count'],
                closed_message_coordinate_count=int(closed.sum()) * rank,
                softmax_gate_sum_error=abs(total_sum - (1. if n else 0.)) if mode == 'softmax' else None),
            geometry=dict(P_norm=_norm(torch, positive), Z_norm=_norm(torch, negative), sum_norm=_norm(torch, total),
                projected_P_norm=_norm(torch, projected_positive), projected_Z_norm=_norm(torch, projected_negative),
                projected_sum_norm=_norm(torch, projected_total), P_Z_cosine=_cosine(torch, positive, negative),
                projected_P_Z_cosine=_cosine(torch, projected_positive, projected_negative),
                partition_residual=_difference(torch, positive+negative, total),
                projected_partition_residual=_difference(torch, projected_positive+projected_negative, projected_total),
                saved_aggregate_vs_fp64_message_sum=_difference(torch, cap['aggregate'][0], total),
                fp64_product_vs_saved_fp32_message=_difference(torch, product64, message64)),
            groups=groups)

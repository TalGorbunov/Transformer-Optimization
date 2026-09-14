"""Training-only objectives for paired examples with arbitrary answer lengths.

Metadata does not establish semantic equivalence: the caller must bind identical
questions, full target sequences, and actual strict prefixes within each pair.
No output decoder, vocabulary mask, attention or inference operation is added.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sequence_layout(target_sequences):
    """Pack scenes in adjacent pairs, keeping every valid prediction position.

    Targets include native EOS. This generic layout does not interpret tokens;
    the caller verifies template-boundary tokenization and EOS placement.
    """
    sequences = [list(ids) for ids in target_sequences]
    need(len(sequences) > 0 and len(sequences) % 2 == 0, 'Need adjacent scene pairs')
    need(all(ids and all(type(token) is int and token >= 0 for token in ids) for ids in sequences),
         'Targets must be nonempty nonnegative integer sequences')
    offsets = [0]
    for ids in sequences:
        offsets.append(offsets[-1] + len(ids))
    left, right, pair_weights, scene_weights, pair_ids = [], [], [], [], []
    pairs = len(sequences) // 2
    for pair in range(pairs):
        a, b = 2 * pair, 2 * pair + 1
        need(sequences[a] == sequences[b], 'Paired full target sequences differ')
        length = len(sequences[a])
        left.extend(range(offsets[a], offsets[a + 1]))
        right.extend(range(offsets[b], offsets[b + 1]))
        pair_weights.extend([1. / (pairs * length)] * length)
        pair_ids.extend([pair] * length)
    for ids in sequences:
        scene_weights.extend([1. / (len(sequences) * len(ids))] * len(ids))
    return dict(target_sequences=sequences, targets=[t for ids in sequences for t in ids],
        offsets=offsets, left=left, right=right, pair_ids=pair_ids,
        pair_weights=pair_weights, scene_weights=scene_weights,
        prefixes=[[ids[:position] for position in range(len(ids))] for ids in sequences])


def pack_feature_sequences(local, global_states):
    """Pad only local items with raw zeros; concatenate valid output positions.

    Local entries are [N_s,T_s,H], global entries [T_s,H]. The returned local
    tensor is [max(N_s),sum(T_s),H], with no artificial output-token positions.
    """
    need(len(local) == len(global_states) and len(local) > 0, 'Feature scene lists differ')
    reference = global_states[0]
    need(reference.ndim == 2 and reference.numel() > 0, 'Empty global feature sequence')
    height = reference.shape[-1]
    for h, g in zip(local, global_states):
        need(g.ndim == 2 and g.shape[0] > 0 and g.shape[1] == height
             and h.ndim == 3 and h.shape[0] > 0 and h.shape[1:] == g.shape,
             'Local/global causal positions differ')
        need(h.device == g.device == reference.device and h.dtype == g.dtype == reference.dtype,
             'Feature device or native dtype differs')
        need(not h.requires_grad and not g.requires_grad, 'Backbone features must be frozen')
    width = max(h.shape[0] for h in local)
    padded = [F.pad(h, (0, 0, 0, 0, 0, width - h.shape[0])) for h in local]
    return torch.cat(padded, dim=1), torch.cat(global_states, dim=0)


def sequence_objectives(logits, delta, aggregates, aggregate_weight, output_weight,
                        frozen_global, layout, *, eps=1e-6):
    """Return sequence-balanced native CE and both paired regularizers.

    Learned residual/aggregate/projection arithmetic remains FP32. This is a
    ragged extension of the frozen V8/V9 losses, with equal weight per scene
    and pair regardless of target length. Both penalties are always computed.
    """
    need(layout == sequence_layout(layout['target_sequences']), 'Layout metadata changed')
    need(isinstance(eps, (float, int)) and not isinstance(eps, bool)
         and math.isfinite(eps) and eps > 0, 'Invalid denominator epsilon')
    count = len(layout['targets'])
    need(delta.ndim == 2 and delta.shape[0] == count and delta.shape[1] > 0,
         'Residual positions differ')
    need(frozen_global.shape == delta.shape and not frozen_global.requires_grad,
         'Frozen global state shape/gradient differs')
    need(aggregates.ndim == 2 and aggregates.shape[0] == count and aggregates.shape[1] > 0,
         'Aggregate positions differ')
    rank = aggregates.shape[1]
    need(aggregate_weight.shape == (rank, rank) and output_weight.shape == (delta.shape[1], rank),
         'Projection dimensions differ')
    need(logits.ndim == 2 and logits.shape[0] == count and logits.shape[1] > max(layout['targets']),
         'Native logits/targets differ')
    values = (logits, delta, aggregates, aggregate_weight, output_weight, frozen_global)
    need(all(value.device == delta.device and value.is_floating_point()
             and bool(torch.isfinite(value).all()) for value in values), 'Invalid objective inputs')
    need(all(value.dtype == torch.float32 for value in (delta, aggregates, aggregate_weight, output_weight)),
         'Learned objective inputs must remain FP32')
    left = torch.tensor(layout['left'], device=delta.device, dtype=torch.long)
    right = torch.tensor(layout['right'], device=delta.device, dtype=torch.long)
    need(torch.equal(frozen_global[left], frozen_global[right]), 'Paired frozen query states differ')
    with torch.autocast(device_type=delta.device.type, enabled=False):
        targets = torch.tensor(layout['targets'], device=delta.device, dtype=torch.long)
        scene_weights = torch.tensor(layout['scene_weights'], device=delta.device, dtype=torch.float32)
        pair_weights = torch.tensor(layout['pair_weights'], device=delta.device, dtype=torch.float32)
        ce_positions = F.cross_entropy(logits.float(), targets, reduction='none')
        denominator = frozen_global[left].detach().float().square().sum(-1) + eps
        residual_positions = (delta[right] - delta[left]).square().sum(-1) / denominator
        displacement = F.linear(aggregates[right] - aggregates[left], aggregate_weight)
        bound = (displacement.abs() * torch.linalg.vector_norm(output_weight, dim=0)).sum(-1)
        path_positions = bound.square() / denominator
        result = dict(ce=(ce_positions * scene_weights).sum(),
            residual=(residual_positions * pair_weights).sum(),
            path=(path_positions * pair_weights).sum(),
            ce_positions=ce_positions, residual_positions=residual_positions,
            path_positions=path_positions, bound=bound, denominator=denominator)
    need(all(bool(torch.isfinite(value).all()) for value in result.values()), 'Objective overflow')
    return result

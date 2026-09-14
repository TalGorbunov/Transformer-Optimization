"""Training-only, fixed-coordinate supervision of native image-prefix states.

No labels enter the native forward. This module owns no trainable parameters.
Projection creation and all numerical tests belong in an authorized Slurm job.
"""
from __future__ import annotations

from itertools import combinations
import torch
from torch import nn

HIDDEN_SIZE = 3584
CHANNELS = 40
PROJECTION_SEED = 20260914
VARIANCE_FLOOR = 1e-12
PEOPLE = ('Daniel', 'John', 'Mary', 'Michael', 'Sandra')
ROOMS = ('Bathroom', 'Bedroom', 'Garden', 'Hallway', 'Kitchen', 'Office')
PAIRS = tuple(combinations(PEOPLE, 2))
CHANNEL_NAMES = tuple(f'person_room/{p}/{r}' for p in PEOPLE for r in ROOMS) + tuple(
    f'co_location/{a}/{b}' for a, b in PAIRS)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def world_targets(sequence):
    """Pure Python: original room membership -> local bits and inclusive sums."""
    need(isinstance(sequence, list) and len(sequence) > 0, 'Nonempty original world required')
    local = []
    previous = None
    for index, frame in enumerate(sequence):
        need(isinstance(frame, dict) and type(frame.get('step_id')) is int and frame.get('step_id') == index + 1,
             'Original consecutive one-based step IDs required')
        rooms = frame.get('rooms')
        need(isinstance(rooms, dict) and set(rooms) == set(ROOMS), 'Original six rooms required')
        positions = {}
        for room, persons in rooms.items():
            need(isinstance(persons, list), 'Original room occupants must be lists')
            for person in persons:
                need(person in PEOPLE and person not in positions, 'Each original person occurs exactly once')
                positions[person] = room
        need(set(positions) == set(PEOPLE), 'All five original people required')
        need(previous is None or sum(positions[p] != previous[p] for p in PEOPLE) == 1,
             'Exactly one person must move between original frames')
        previous = positions
        row = [int(positions[p] == r) for p in PEOPLE for r in ROOMS]
        row += [int(positions[a] == positions[b]) for a, b in PAIRS]
        local.append(row)
    running = [0] * CHANNELS
    prefix = []
    for row in local:
        running = [a + b for a, b in zip(running, row)]
        prefix.append(running.copy())
    return dict(local=local, prefix=prefix)


def build_projection():
    """Private CPU Gaussian QR; canonical column signs, saved FP32 row basis."""
    generator = torch.Generator(device='cpu').manual_seed(PROJECTION_SEED)
    gaussian = torch.randn(HIDDEN_SIZE, CHANNELS, dtype=torch.float64, device='cpu', generator=generator)
    q, r = torch.linalg.qr(gaussian, mode='reduced')
    diagonal = r.diagonal()
    need(bool(torch.isfinite(q).all()) and bool((diagonal != 0).all()), 'Finite full-rank QR required')
    signs = torch.where(diagonal > 0, torch.ones_like(diagonal), -torch.ones_like(diagonal))
    projection = (q * signs).T.to(torch.float32).contiguous()
    error = (projection.double() @ projection.double().T - torch.eye(CHANNELS, dtype=torch.float64)).abs().max()
    need(float(error) <= 2e-6, 'Stored FP32 projection failed orthonormality check')
    return projection


def population_normalization(worlds):
    """Equal weight per world, then equal weight per prefix; population moments."""
    total = torch.zeros(CHANNELS, dtype=torch.float64)
    second = torch.zeros_like(total)
    count = 0
    for raw in worlds:
        value = torch.as_tensor(raw, dtype=torch.float64, device='cpu')
        need(value.ndim == 2 and value.shape[1] == CHANNELS and value.shape[0] > 0
             and bool(torch.isfinite(value).all()), 'Finite nonempty N-by40 targets required')
        total += value.mean(0)
        second += value.square().mean(0)
        count += 1
    need(count > 0, 'Training worlds required for normalization')
    mean = total / count
    variance = second / count - mean.square()
    need(float(variance.min()) >= -1e-10, 'Invalid population variance')
    variance = variance.clamp_min(0)
    scale = torch.where(variance <= VARIANCE_FLOOR, torch.ones_like(variance), variance.sqrt())
    return dict(mean=mean.float(), scale=scale.float(), mean_fp64=mean, variance_fp64=variance,
                scale_fp64=scale, worlds=count, weighting='equal_world_then_equal_prefix',
                variance_floor=VARIANCE_FLOOR)


class FixedPrefixSupervision(nn.Module):
    """A frozen probe used only as an auxiliary training objective."""
    def __init__(self, projection, mean, scale):
        super().__init__()
        for name, value, shape in (('projection', projection, (CHANNELS, HIDDEN_SIZE)),
                                   ('mean', mean, (CHANNELS,)), ('scale', scale, (CHANNELS,))):
            need(isinstance(value, torch.Tensor) and tuple(value.shape) == shape
                 and value.dtype == torch.float32 and bool(torch.isfinite(value).all()),
                 'Exact finite FP32 buffer required: ' + name)
            self.register_buffer(name, value.detach().clone())
        need(bool((self.scale > 0).all()), 'Positive affine scales required')

    def _validate(self, hidden):
        need(isinstance(hidden, torch.Tensor) and hidden.ndim in (2, 3)
             and (hidden.ndim == 2 or hidden.shape[0] == 1) and hidden.shape[-1] == HIDDEN_SIZE
             and hidden.numel() > 0 and hidden.dtype in (torch.float16, torch.bfloat16, torch.float32)
             and hidden.device == self.projection.device and bool(torch.isfinite(hidden).all()),
             'Finite native image-end rows [N,H] or [1,N,H] required')
        need(all(v.dtype == torch.float32 and bool(torch.isfinite(v).all()) for v in self.buffers())
             and bool((self.scale > 0).all()), 'Fixed FP32 affine/projection contract changed')

    def forward(self, hidden):
        self._validate(hidden)
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            result = hidden.float() @ self.projection.T
        need(bool(torch.isfinite(result).all()), 'Nonfinite projected hidden state')
        return result

    def loss(self, hidden, raw_targets):
        predicted = self(hidden)
        need(isinstance(raw_targets, torch.Tensor) and raw_targets.numel() == predicted.numel()
             and tuple(raw_targets.shape[-2:]) == tuple(predicted.shape[-2:])
             and raw_targets.ndim in (2, 3) and (raw_targets.ndim == 2 or raw_targets.shape[0] == 1)
             and not raw_targets.requires_grad and bool(torch.isfinite(raw_targets).all()),
             'Matching fixed raw N-by40 target tensor required')
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            target = (raw_targets.to(device=hidden.device, dtype=torch.float32).reshape(predicted.shape) - self.mean) / self.scale
            loss = (predicted - target).square().mean()
        need(bool(torch.isfinite(loss)), 'Finite auxiliary mean squared error required')
        return loss

    def decode(self, hidden):
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return self(hidden) * self.scale + self.mean


def reference_fp64(hidden, raw_targets, projection, mean, scale):
    """Audit interface: FP64 arithmetic on the actual saved FP32 coefficients.

    This deliberately does not reconstruct another QR basis or use unrounded
    population moments in place of the affine buffers used during training.
    """
    h = hidden.detach().to(device='cpu', dtype=torch.float64).reshape(-1, HIDDEN_SIZE)
    raw = raw_targets.detach().to(device='cpu', dtype=torch.float64).reshape(-1, CHANNELS)
    p = projection.detach().to(device='cpu', dtype=torch.float64)
    m = mean.detach().to(device='cpu', dtype=torch.float64)
    s = scale.detach().to(device='cpu', dtype=torch.float64)
    predicted = h @ p.T
    target = (raw - m) / s
    return dict(prediction=predicted, normalized_target=target, decoded=predicted * s + m,
                loss=(predicted - target).square().mean())

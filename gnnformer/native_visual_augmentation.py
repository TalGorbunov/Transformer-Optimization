"""One intact-evidence summary read and an exactly sized pointwise control.

No native decoder, token mask, label, cache or optimizer is constructed here.
The caller selects eligible native rows. Forward returns a live FP32 delta;
native_residual_add implements the single cast-before-add boundary.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn
from gnnformer.native_visual_memory import NativeVisualMemory

HIDDEN_SIZE = 3584
READ_WIDTH = 128
SLOTS = 32
POINTWISE_WIDTH = 321
RMS_EPS = 1e-6
LIVE_PARAMETERS = 2300929
SUMMARY_ALLOCATED_PARAMETERS = 2304513
POOL_SEED = 24


def _need(value, message):
    if not value:
        raise ValueError(message)


def _finite(value, name):
    _need(isinstance(value, Tensor) and bool(torch.isfinite(value).all()), name + ' must be finite')


def _rms(value):
    promoted = value.float()
    moment = promoted.square().mean(dim=-1, keepdim=True)
    _finite(moment, 'RMS second moment')
    return promoted * torch.rsqrt(moment + RMS_EPS)


def _summary_read(hidden, memory, query_weight, key_weight, value_weight, output_weight):
    """Dimension-generic algebra shared by the fixed core and tiny fixtures."""
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        normalized_hidden = _rms(hidden)
        normalized_memory = _rms(memory)
        query = normalized_hidden @ query_weight.T
        keys = normalized_memory @ key_weight.T
        values = memory.float() @ value_weight.T
        scores = (query @ keys.T) / math.sqrt(query_weight.shape[0])
        _finite(scores, 'Summary-read scores')
        attention = scores.softmax(dim=-1)
        mixed_value = attention @ values
        readout = mixed_value @ output_weight.T
        _finite(readout, 'Summary readout')
        return readout, dict(normalized_hidden=normalized_hidden, normalized_memory=normalized_memory,
            query=query, keys=keys, values=values, scores=scores, attention=attention, mixed_value=mixed_value)


def _pointwise_read(hidden, down_weight, up_weight):
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        normalized_hidden = _rms(hidden)
        preactivation = normalized_hidden @ down_weight.T
        activation = torch.nn.functional.silu(preactivation)
        readout = activation @ up_weight.T
        _finite(readout, 'Pointwise readout')
        return readout, dict(normalized_hidden=normalized_hidden, preactivation=preactivation, activation=activation)


def native_residual_add(hidden_rows: Tensor, delta: Tensor) -> Tensor:
    """Exactly one FP32-to-native delta cast followed by native-dtype addition."""
    _need(isinstance(hidden_rows, Tensor) and hidden_rows.dtype in (torch.float16, torch.bfloat16, torch.float32)
          and hidden_rows.numel() > 0, 'Nonempty native floating hidden rows required')
    _need(isinstance(delta, Tensor) and delta.dtype == torch.float32 and delta.shape == hidden_rows.shape
          and delta.device == hidden_rows.device, 'Same-shape/device FP32 delta required')
    _finite(hidden_rows, 'Native hidden rows'); _finite(delta, 'FP32 delta')
    with torch.autocast(device_type=hidden_rows.device.type, enabled=False):
        native_delta = delta.to(dtype=hidden_rows.dtype)
        _finite(native_delta, 'Native delta cast')
        result = hidden_rows + native_delta
        _need(result.dtype == hidden_rows.dtype, 'Native addition dtype changed')
        _finite(result, 'Native residual addition')
        return result


def _weight(rows, columns, generator):
    bound = 1 / math.sqrt(columns)
    return nn.Parameter(torch.empty(rows, columns, dtype=torch.float32, device='cpu').uniform_(
        -bound, bound, generator=generator))


class _Residual(nn.Module):
    def __init__(self, seed):
        super().__init__()
        _need(type(seed) is int and 0 <= seed < 2**63, 'Explicit nonnegative integer initialization seed required')
        self.seed = seed
        self.alpha = nn.Parameter(torch.zeros((), dtype=torch.float32, device='cpu'))

    def _rows(self, hidden_rows):
        _need(isinstance(hidden_rows, Tensor) and hidden_rows.ndim >= 1 and hidden_rows.shape[-1] == HIDDEN_SIZE
              and hidden_rows.numel() > 0 and hidden_rows.dtype in (torch.float16, torch.bfloat16, torch.float32)
              and hidden_rows.device == self.alpha.device, 'Nonempty same-device native [...,3584] rows required')
        _finite(hidden_rows, 'Hidden rows')
        _need(self.alpha.shape == () and self.alpha.dtype == torch.float32, 'Scalar FP32 gate required')
        _finite(self.alpha, 'Gate')
        return hidden_rows.reshape(-1, HIDDEN_SIZE)

    def _weights(self, shapes):
        for name, shape in shapes.items():
            weight = getattr(self, name)
            _need(weight.shape == shape and weight.dtype == torch.float32 and weight.device == self.alpha.device,
                  'Fixed FP32 parameter shape/device changed: ' + name)
            _finite(weight, name)

    def _gated(self, hidden_rows, readout, details, arm):
        with torch.autocast(device_type=hidden_rows.device.type, enabled=False):
            gate = self.alpha.tanh()
            delta = (gate * readout).reshape(hidden_rows.shape)
            _finite(delta, 'Gated delta')
        capture = dict(details, arm=arm, seed=self.seed, gate=gate,
            readout=readout.reshape(hidden_rows.shape), delta=delta)
        return delta, capture

    def forward(self, hidden_rows: Tensor, memory=None) -> Tensor:
        """Live FP32 residual, including tanh gate; arbitrary leading row dimensions."""
        return self.delta_and_capture(hidden_rows, memory)[0]


class SummaryReadResidual(_Residual):
    """Fresh pool seed24; Q,K,V,O use an independent private reader seed.

    Seven potentially live tensors total2,300,929 parameters. The reused pool's
    additional3,584 mass-direction entries remain frozen zero and unused.
    Neither constructor consumes the caller's RNG. No state is detached.
    """
    def __init__(self, seed=25):
        super().__init__(seed)
        self.pool = NativeVisualMemory()
        self.pool.mass_direction.requires_grad_(False)
        generator = torch.Generator(device='cpu').manual_seed(seed)
        self.query_weight = _weight(READ_WIDTH, HIDDEN_SIZE, generator)
        self.key_weight = _weight(READ_WIDTH, HIDDEN_SIZE, generator)
        self.value_weight = _weight(READ_WIDTH, HIDDEN_SIZE, generator)
        self.output_weight = _weight(HIDDEN_SIZE, READ_WIDTH, generator)

    def _pool_valid(self):
        self.pool._parameters_valid()
        _need(not self.pool.mass_direction.requires_grad and self.pool.mass_direction.grad is None
              and not bool(torch.count_nonzero(self.pool.mass_direction)), 'Pool mass direction must stay frozen zero')
        _need(self.pool.key_weight.device == self.alpha.device, 'Pool/read device mismatch')

    def prepare_memory(self, features: Tensor, *, frame_index: Tensor, raster_row: Tensor, raster_col: Tensor,
                       grid_height: Tensor, grid_width: Tensor) -> dict[str, Tensor]:
        """Question-free normalized pool packet; recompute after each parameter update."""
        self._pool_valid()
        return self.pool(features, frame_index, raster_row, raster_col, grid_height, grid_width, retain_mass=False)

    def delta_and_capture(self, hidden_rows: Tensor, memory=None):
        hidden = self._rows(hidden_rows); self._pool_valid()
        self._weights(dict(query_weight=(READ_WIDTH,HIDDEN_SIZE), key_weight=(READ_WIDTH,HIDDEN_SIZE),
                           value_weight=(READ_WIDTH,HIDDEN_SIZE), output_weight=(HIDDEN_SIZE,READ_WIDTH)))
        _need(isinstance(memory, dict) and set(memory) == {'tokens','normalized_value','log_mass','mass_feature'},
              'Exact normalized pool packet required')
        for name, shape in (('tokens',(SLOTS,HIDDEN_SIZE)), ('normalized_value',(SLOTS,HIDDEN_SIZE)),
                            ('log_mass',(SLOTS,1)), ('mass_feature',(SLOTS,1))):
            value = memory[name]
            _need(isinstance(value, Tensor) and value.shape == shape and value.dtype == torch.float32
                  and value.device == hidden.device, 'Fixed FP32 pool packet changed: ' + name)
            _finite(value, name)
        _need(torch.equal(memory['tokens'], memory['normalized_value']) and not bool(torch.count_nonzero(memory['mass_feature'])),
              'Only normalized identity-value memory is permitted')
        readout, details = _summary_read(hidden, memory['tokens'], self.query_weight,
            self.key_weight, self.value_weight, self.output_weight)
        return self._gated(hidden_rows, readout, details, 'summary')


class PointwiseResidual(_Residual):
    """Bias-free321-wide SiLU control: three live tensors, exactly2,300,929 entries."""
    def __init__(self, seed=25):
        super().__init__(seed)
        generator = torch.Generator(device='cpu').manual_seed(seed)
        self.down_weight = _weight(POINTWISE_WIDTH, HIDDEN_SIZE, generator)
        self.up_weight = _weight(HIDDEN_SIZE, POINTWISE_WIDTH, generator)

    def prepare_memory(self, features, *, frame_index, raster_row, raster_col, grid_height, grid_width):
        """Common caller interface; intentionally performs no feature/coordinate access."""
        return None

    def delta_and_capture(self, hidden_rows: Tensor, memory=None):
        _need(memory is None, 'Pointwise control has no memory input')
        hidden = self._rows(hidden_rows)
        self._weights(dict(down_weight=(POINTWISE_WIDTH,HIDDEN_SIZE), up_weight=(HIDDEN_SIZE,POINTWISE_WIDTH)))
        readout, details = _pointwise_read(hidden, self.down_weight, self.up_weight)
        return self._gated(hidden_rows, readout, details, 'pointwise')

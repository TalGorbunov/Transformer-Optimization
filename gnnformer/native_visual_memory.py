"""Fixed learned-slot memory of native postmerger features, before any question.

This is ordinary attention pooling with its log partition retained as an
optional feature. It is not a new attention primitive or a sufficient statistic
for arbitrary relations. No decoder, image processor or task labels enter here.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn
from gnnformer.attention_moments import AttentionMoments, summarize, merge, read_moments

HIDDEN_SIZE = 3584
SLOTS = 32
KEY_WIDTH = 128
POSITION_WIDTH = 24
SEED = 24
RMS_EPS = 1e-6
PARAMETERS = 469504


def position_features(frame_index: Tensor, raster_row: Tensor, raster_col: Tensor,
                      grid_height: Tensor, grid_width: Tensor) -> Tensor:
    """FP32 [items,24], in frame/y/x then frequency/sin/cos order.

    Inputs are int64 [items] on one device. Frame indices are zero based and
    globally fixed within the supported 128-frame collection. Grid coordinates
    refer to the native postmerger raster, not its internal window permutation.
    Chunking never renumbers frames or normalizes them by collection length.
    """
    fields = (frame_index, raster_row, raster_col, grid_height, grid_width)
    if any(not isinstance(x, Tensor) for x in fields):
        raise ValueError('Every coordinate field must be an int64 tensor')
    if any(x.ndim != 1 or x.dtype != torch.int64 or x.shape != frame_index.shape
           or x.device != frame_index.device for x in fields):
        raise ValueError('Matching int64 coordinate vectors on one device required')
    valid = ((frame_index >= 0) & (frame_index < 128) & (grid_height > 0) & (grid_width > 0)
             & (raster_row >= 0) & (raster_row < grid_height)
             & (raster_col >= 0) & (raster_col < grid_width))
    if not bool(valid.all()):
        raise ValueError('Invalid frame index or merged-raster coordinate')
    coordinates = torch.stack((frame_index.float() / 128,
        (raster_row.float() + .5) / grid_height.float(),
        (raster_col.float() + .5) / grid_width.float()), dim=1)
    frequencies = coordinates.new_tensor((1., 2., 4., 8.))
    phase = (2 * math.pi) * coordinates[:, :, None] * frequencies
    return torch.stack((phase.sin(), phase.cos()), dim=-1).reshape(-1, POSITION_WIDTH)


class NativeVisualMemory(nn.Module):
    """One fixed H3584/K32/key128 configuration with three FP32 parameters.

    Initialization uses a private CPU generator at seed24, without consuming the
    caller's RNG: key_weight uniform +/-1/sqrt(3608), then queries N(0,1), then
    mass_direction exactly zero. No biases, learned normalization, value map or
    output projection. Both arms allocate the same tensors; the normalized
    arm's mass_direction is functionally inactive.
    """
    def __init__(self):
        super().__init__()
        generator = torch.Generator(device='cpu').manual_seed(SEED)
        bound = 1 / math.sqrt(HIDDEN_SIZE + POSITION_WIDTH)
        self.key_weight = nn.Parameter(torch.empty(KEY_WIDTH, HIDDEN_SIZE + POSITION_WIDTH,
            dtype=torch.float32, device='cpu').uniform_(-bound, bound, generator=generator))
        self.queries = nn.Parameter(torch.randn(SLOTS, KEY_WIDTH, generator=generator,
            dtype=torch.float32, device='cpu'))
        self.mass_direction = nn.Parameter(torch.zeros(HIDDEN_SIZE, dtype=torch.float32, device='cpu'))

    def _parameters_valid(self):
        parameters = (self.key_weight, self.queries, self.mass_direction)
        if (self.key_weight.shape != (KEY_WIDTH, HIDDEN_SIZE + POSITION_WIDTH)
                or self.queries.shape != (SLOTS, KEY_WIDTH)
                or self.mass_direction.shape != (HIDDEN_SIZE,)
                or any(p.dtype != torch.float32 or p.device != self.key_weight.device
                       or not bool(torch.isfinite(p).all()) for p in parameters)):
            raise ValueError('Exact finite FP32 native-memory parameters required')

    @staticmethod
    def _state_valid(state: AttentionMoments):
        if not isinstance(state, AttentionMoments):
            raise ValueError('An attention_moments state is required')
        tensors = (state.maximum, state.scaled_mass, state.scaled_value)
        if (state.maximum.shape != (SLOTS,) or state.scaled_mass.shape != (SLOTS,)
                or state.scaled_value.shape != (SLOTS, HIDDEN_SIZE)
                or any(x.dtype != torch.float32 or x.device != state.maximum.device for x in tensors)
                or not bool((torch.isfinite(state.maximum) | torch.isneginf(state.maximum)).all())
                or not bool(torch.isfinite(state.scaled_mass).all())
                or not bool((state.scaled_mass >= 0).all())
                or not bool(torch.isfinite(state.scaled_value).all())):
            raise ValueError('Invalid native-width FP32 moment state')

    def summarize(self, features: Tensor, frame_index: Tensor, raster_row: Tensor,
                  raster_col: Tensor, grid_height: Tensor, grid_width: Tensor) -> AttentionMoments:
        """Accumulate one chunk; values are the unnormalized native features.

        Features [items,3584] may be native FP16 or FP32. Only the key input is
        parameter-free RMS normalized. All arithmetic and returned state are
        FP32. Empty chunks are allowed; a completely empty read is rejected.
        Merge only states from the same unchanged parameters and global frame
        coordinates. No parameter-dependent cached state is valid across updates.
        """
        self._parameters_valid()
        if (not isinstance(features, Tensor) or features.ndim != 2
                or features.shape[1] != HIDDEN_SIZE or features.dtype not in (torch.float16, torch.float32)
                or features.device != self.key_weight.device or not bool(torch.isfinite(features).all())):
            raise ValueError('Finite native FP16/FP32 features [items,3584] required')
        positions = position_features(frame_index, raster_row, raster_col, grid_height, grid_width)
        if positions.shape[0] != features.shape[0] or positions.device != features.device:
            raise ValueError('One same-device position per native item required')
        with torch.autocast(device_type=features.device.type, enabled=False):
            values = features.float()
            second_moment = values.square().mean(dim=-1, keepdim=True)
            if not bool(torch.isfinite(second_moment).all()):
                raise ValueError('Native feature RMS overflows FP32')
            normalized = values * torch.rsqrt(second_moment + RMS_EPS)
            keys = torch.cat((normalized, positions), dim=-1) @ self.key_weight.T
            scores = (self.queries @ keys.T) / math.sqrt(KEY_WIDTH)
            return summarize(scores, values)

    def merge(self, left: AttentionMoments, right: AttentionMoments) -> AttentionMoments:
        self._state_valid(left)
        self._state_valid(right)
        return merge(left, right)

    def read(self, state: AttentionMoments, *, retain_mass: bool) -> dict[str, Tensor]:
        """Return FP32 tokens, normalized_value, log_mass, and mass_feature.

        No native cast or decoder call occurs here. Literal replication of the
        same enriched items leaves mu invariant and shifts logZ by log(c).
        Replication with new frame coordinates is a different input operation.
        """
        self._parameters_valid()
        self._state_valid(state)
        if type(retain_mass) is not bool or state.maximum.device != self.mass_direction.device:
            raise ValueError('Explicit boolean retain_mass and matching state device required')
        mean, log_mass = read_moments(state)
        feature = log_mass if retain_mass else torch.zeros_like(log_mass)
        tokens = mean + feature * self.mass_direction[None, :]
        if not bool(torch.isfinite(tokens).all()):
            raise ValueError('Nonfinite memory readout')
        return dict(tokens=tokens, normalized_value=mean, log_mass=log_mass, mass_feature=feature)

    def forward(self, features: Tensor, frame_index: Tensor, raster_row: Tensor,
                raster_col: Tensor, grid_height: Tensor, grid_width: Tensor,
                *, retain_mass: bool) -> dict[str, Tensor]:
        return self.read(self.summarize(features, frame_index, raster_row, raster_col,
                                       grid_height, grid_width), retain_mass=retain_mass)

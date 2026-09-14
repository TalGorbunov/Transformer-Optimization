"""Standalone native readout hook for parallel local/global streams.

Software integration only: this file does not pack tokens, load models, manage
KV caches, or establish model efficacy. The caller supplies N local rows followed
by one global row, with their final query aligned at the last sequence position.
Only that global position receives a residual, immediately before the native
final norm. All preceding native computation is unchanged.

The branch is held by a plain Python controller, never registered on the model.
Freeze the complete native model and keep it in eval mode externally. This
controller refuses a trainable final norm and does not change any parameter,
requires_grad flag, train/eval state, generation history, or incoming tensor.
"""
from __future__ import annotations

import torch

from gnnformer.parallel_local_aggregation import ParallelLocalAggregation


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _local_count(value):
    _require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
             "n_local_rows must be a nonnegative integer")
    return value


class ParallelLocalNative:
    """Context-managed, out-of-place fusion before the native final norm.

    Usage:
        with ParallelLocalNative(model.model.language_model.norm, branch,
                                 n_local_rows=n) as fusion:
            output = model(...)  # ordinary native forward

    The caller must ensure each row's last token is the aligned causal query.
    For prefill this is the final original prompt token; for cached decoding it
    is the current shared generated token. This minimal hook handles one query
    position per forward, not multi-position teacher-forced training.

    Separate controllers must not be nested on the same norm. Re-entering this
    controller while active is rejected. Capture is optional and keeps only
    detached device tensors from the latest call; it never copies to CPU by
    default. Captures describe a computation, not frame-label evidence.
    """

    def __init__(self, norm, branch: ParallelLocalAggregation, *, n_local_rows: int,
                 capture: bool = False):
        _require(isinstance(norm, torch.nn.Module), "norm must be a native torch module")
        _require(isinstance(branch, ParallelLocalAggregation),
                 "branch must be a ParallelLocalAggregation")
        _require(not any(parameter.requires_grad for parameter in norm.parameters()),
                 "The native final norm must already be frozen")
        self.norm = norm
        self.branch = branch
        self.n_local_rows = _local_count(n_local_rows)
        self.capture = bool(capture)
        self.calls = 0
        self.last_query_position = None
        self.last_capture = None
        self._handle = None

    @property
    def active(self):
        return self._handle is not None

    def __enter__(self):
        _require(not self.active, "This fusion controller is already active")
        _require(not any(parameter.requires_grad for parameter in self.norm.parameters()),
                 "The native final norm must remain frozen")
        self.calls = 0
        self.last_query_position = None
        self.last_capture = None
        self._handle = self.norm.register_forward_pre_hook(self._before_norm, with_kwargs=True)
        return self

    def close(self):
        """Remove this hook only; safe after either normal completion or error."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def _before_norm(self, module, args, kwargs):
        _require(module is self.norm and self.active, "Unexpected native norm invocation")
        _require(not any(parameter.requires_grad for parameter in module.parameters()),
                 "The native final norm must remain frozen")
        positional = len(args) > 0
        if positional:
            _require("hidden_states" not in kwargs, "Ambiguous native norm hidden states")
            hidden = args[0]
        else:
            _require("hidden_states" in kwargs, "Native norm did not receive hidden states")
            hidden = kwargs["hidden_states"]
        _require(isinstance(hidden, torch.Tensor) and hidden.is_floating_point()
                 and hidden.ndim == 3 and hidden.shape[0] == self.n_local_rows + 1
                 and hidden.shape[1] > 0 and hidden.shape[2] == self.branch.hidden_size,
                 "Expected native hidden states [N+1,L,H] with L>0 and global row last")
        local = hidden[:self.n_local_rows, -1:, :]
        global_states = hidden[self.n_local_rows, -1:, :]
        delta = self.branch(local, global_states, output_dtype=hidden.dtype)
        _require(delta.shape == global_states.shape and delta.dtype == hidden.dtype,
                 "Branch residual must match the native global query")
        fused_global = global_states + delta
        # Clone and write into new storage. The incoming hidden tensor, local
        # rows, and earlier global positions retain their original values.
        fused = hidden.clone()
        fused[self.n_local_rows, -1:, :] = fused_global
        self.calls += 1
        self.last_query_position = hidden.shape[1] - 1
        self.last_capture = None
        if self.capture:
            self.last_capture = {
                "local_states": local.detach().clone(),
                "global_states": global_states.detach().clone(),
                "delta": delta.detach().clone(),
                "fused_global": fused_global.detach().clone(),
            }
        if positional:
            return (fused,) + args[1:], kwargs
        new_kwargs = dict(kwargs)
        new_kwargs["hidden_states"] = fused
        return args, new_kwargs

    def export_last_capture(self, *, cpu: bool = False):
        """Return detached copies; optional CPU copy is explicit.

        Cloning outside inference mode makes exported tensors normal tensors,
        even when capture happened inside an inference-mode model forward.
        They remain constants; training should not modify or reuse live views.
        """
        if self.last_capture is None:
            return None
        with torch.inference_mode(False), torch.no_grad():
            return {
                name: value.detach().to("cpu").clone() if cpu else value.detach().clone()
                for name, value in self.last_capture.items()
            }


class GlobalBroadcastLogitsProcessor:
    """Choose all rows' next token using the final global row's scores.

    Compatible with the two-argument Hugging Face logits-processor callable
    interface without importing transformers. Use only with externally fixed
    greedy, single-beam generation and a batch of N+1 independent streams.
    prompt_length is the COMMON left-padded prompt width, not an unpadded
    per-row token count. Earlier row-specific prompts may differ; every token
    generated after that boundary must be identical across rows.

    The return value owns fresh storage. Native raw logits/scores and input_ids
    are never modified. Negative infinity from an upstream masking processor
    is allowed, provided the global row retains a finite candidate.
    """

    def __init__(self, *, n_local_rows: int, prompt_length: int):
        self.n_local_rows = _local_count(n_local_rows)
        _require(isinstance(prompt_length, int) and not isinstance(prompt_length, bool)
                 and prompt_length > 0, "prompt_length must be a positive integer")
        self.prompt_length = prompt_length
        self.calls = 0

    def __call__(self, input_ids, scores):
        batch = self.n_local_rows + 1
        _require(isinstance(input_ids, torch.Tensor) and input_ids.ndim == 2
                 and input_ids.shape[0] == batch and input_ids.shape[1] >= self.prompt_length
                 and input_ids.dtype in (torch.int32, torch.int64),
                 "Expected integer input_ids [N+1,L] covering the common prompt")
        _require(isinstance(scores, torch.Tensor) and scores.ndim == 2
                 and scores.shape[0] == batch and scores.shape[1] > 0 and scores.is_floating_point(),
                 "Expected floating scores [N+1,vocabulary]")
        _require(scores.device == input_ids.device, "Scores and input history must share a device")
        _require(not bool(torch.isnan(scores).any()) and not bool(torch.isposinf(scores).any()),
                 "Scores must not contain NaN or positive infinity")
        _require(bool(torch.isfinite(scores[-1]).any()), "Global row has no finite candidate")
        suffix = input_ids[:, self.prompt_length:]
        _require(torch.equal(suffix, suffix[-1:].expand_as(suffix)),
                 "Generated token prefixes must be identical across all local/global rows")
        self.calls += 1
        return scores[-1:].expand_as(scores).clone()


"""Held learned-selection writes before the final block or final norm.

API: ParallelLocalLearnedMemory(read_layer, final_norm, branch, *, n_local_rows,
write_location, query_indices, stream_positions, selection_mode=None,
capture=False, detach_captures=True). Lifecycle, configure_queries,
assert_complete and export_last_capture(cpu=False) follow ParallelLocalMemory.

The caller supplies the actual block-26 module and final norm, freezes/evals
ALL native modules, and aligns N local rows followed by one global row. This
controller cannot infer the block index, validate token/mask/mRoPE ownership,
or prove that full-prefix callers included every historical write. It builds
no cache or model input. Ordinary final-block attention owns KV updates.

Only query_indices in the current [N+1,S,H] FP16 hidden tensor are written.
stream_positions are absolute packed positions, NOT logical mRoPE positions.
A cached call commonly uses query_indices=[0] and its actual stream position;
full-prefix replay must include every Ppacked-1+t historical query. Local rows
are actual items: padded item rows are not supported by this native controller.

Exactly one core forward computes all query residuals. Its FP32 delta remains
live; the native addition is h + delta.to(torch.float16). capture=True retains
actual query/payload/scores/gates/messages/aggregate/preactivation/delta from
that same call, plus V11 read/write/norm fields and native_delta.
common_query_hidden is the shared lower read; final_block_input_queries is the
actual final-block input. native_query_hidden/fused_query_hidden always refer
to the final-norm input before/after the optional late write (early effects are
already present there). Metadata query_indices, stream_positions,
write_location and selection_mode are explicit.
With detach_captures=False, last_capture['delta'] is the actual graph node used
for BOTH placements; export_last_capture always returns detached normal copies.
Do not call a standalone final-norm replay inside an active controller.

No fitting, runtime, efficacy claim or model loading is released by this file.
"""
from __future__ import annotations

import torch

from gnnformer.parallel_local_learned_selection import ParallelLocalLearnedSelection, MODES
from gnnformer.parallel_local_memory import ParallelLocalMemory, _require
from gnnformer.parallel_local_native import ParallelLocalNative


class ParallelLocalLearnedMemory(ParallelLocalMemory):
    """V11 lifecycle with locked learned selection and actual query captures.

    The inherited ownership registry also excludes overlapping V11 memory
    controllers. Existing native final-norm fusion hooks are rejected without
    removing unrelated observation hooks. Branch parameters may be trainable;
    read-layer/final-norm parameters must be frozen and both modules eval.
    The caller is responsible for the intervening final block and native head.
    """

    def __init__(self, penultimate_layer, final_norm, branch, *, n_local_rows,
                 write_location, query_indices, stream_positions,
                 selection_mode=None, capture=False, detach_captures=True):
        _require(type(branch) is ParallelLocalLearnedSelection,
                 'Require the unchanged ParallelLocalLearnedSelection core')
        mode = branch.mode if selection_mode is None else selection_mode
        _require(mode in MODES and mode == branch.mode, 'Selection mode must match the core')
        _require(type(capture) is bool and type(detach_captures) is bool,
                 'Capture settings must be boolean')
        self.selection_mode = mode
        self._fixed = (penultimate_layer, final_norm, branch, n_local_rows,
                       write_location, mode, capture, detach_captures)
        self._configured_queries = None
        self.core_calls = 0
        super().__init__(penultimate_layer, final_norm, branch,
            n_local_rows=n_local_rows, write_location=write_location,
            query_indices=query_indices, stream_positions=stream_positions,
            capture=capture, detach_captures=detach_captures)

    def _mode_guard(self):
        read, norm, branch, n, placement, mode, capture, detach = self._fixed
        _require(self.penultimate_layer is read and self.final_norm is norm
                 and self.branch is branch and type(self.n_local_rows) is int
                 and self.n_local_rows == n and self.write_location == placement
                 and self.selection_mode == mode and self.branch.mode == mode
                 and self.capture is capture and self.detach_captures is detach,
                 'Native boundaries, core, placement, selection mode or capture settings changed')

    def _check_frozen(self):
        self._mode_guard()
        super()._check_frozen()
        for hook in self.final_norm._forward_pre_hooks.values():
            owner = getattr(hook, '__self__', None)
            _require(owner is self or not isinstance(owner, (ParallelLocalNative, ParallelLocalMemory)),
                     'Another native fusion controller already owns the final norm')
        _require(not self.penultimate_layer.training and not self.final_norm.training,
                 'Native read layer and final norm must remain eval')
        _require(all(p.dtype == torch.float16 for p in self.final_norm.parameters()),
                 'Keep the actual native FP16 final norm')
        _require(all(p.dtype == torch.float32 for p in self.branch.parameters()),
                 'All learned-selection parameters must remain FP32')

    def configure_queries(self, query_indices, stream_positions):
        self._mode_guard()
        super().configure_queries(query_indices, stream_positions)
        self._configured_queries = (self.query_indices, self.stream_positions)

    def __enter__(self):
        self._check_frozen()
        _require(not self.active, 'This learned-memory controller is already active')
        for hook in self.final_norm._forward_pre_hooks.values():
            owner = getattr(hook, '__self__', None)
            _require(not isinstance(owner, (ParallelLocalNative, ParallelLocalMemory)),
                     'Another native fusion controller already owns the final norm')
        self.core_calls = 0
        return super().__enter__()

    def close(self):
        """Remove owned handles/registrations even if a public binding was changed."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._pending = None
        for module in self._fixed[:2]:
            owner = self._owners.get(module)
            if owner is not None and owner() is self:
                del self._owners[module]

    def assert_complete(self):
        self._mode_guard()
        super().assert_complete()
        _require(self.core_calls == self.read_calls,
                 'Each completed native forward requires exactly one learned-core call')

    def _hidden(self, hidden):
        self._check_frozen()
        _require(self._configured_queries is not None
                 and self.query_indices is self._configured_queries[0]
                 and self.stream_positions is self._configured_queries[1],
                 'Use configure_queries to change the explicit query layout')
        super()._hidden(hidden)
        _require(hidden.dtype == torch.float16 and bool(torch.isfinite(hidden).all()),
                 'Require finite native FP16 [N+1,S,H] hidden states')

    @staticmethod
    def _fuse_native(hidden, indices, native_delta):
        receiving = hidden[-1].index_select(0, indices)
        _require(native_delta.dtype == hidden.dtype and native_delta.shape == receiving.shape
                 and native_delta.device == hidden.device and bool(torch.isfinite(native_delta).all()),
                 'Native residual cast is nonfinite or has a different shape/device')
        updated = receiving + native_delta
        _require(bool(torch.isfinite(updated).all()), 'Native residual addition overflowed')
        fused = hidden.clone()
        fused[-1, indices, :] = updated
        return fused, receiving, updated

    def _selection_capture(self, values, delta, local, global_states):
        q, n, r, h = global_states.shape[0], self.n_local_rows, self.branch.rank, self.branch.hidden_size
        shapes = dict(query=(q, r), payload=(n, q, r), scores=(n, q), gates=(n, q),
                      messages=(n, q, r), aggregate=(q, r), preactivation=(q, r), delta=(q, h))
        _require(isinstance(values, dict) and set(values) == set(shapes)
                 and values['delta'] is delta, 'Capture must contain the actual single-call delta and selection values')
        for key, shape in shapes.items():
            value = values[key]
            _require(isinstance(value, torch.Tensor) and tuple(value.shape) == shape
                     and value.dtype == torch.float32 and value.device == delta.device
                     and bool(torch.isfinite(value).all()), 'Invalid learned-selection capture: '+key)
        return dict({key: self._remember(value) for key, value in values.items()},
                    local_states=self._remember(local), global_states=self._remember(global_states),
                    query_indices=list(self.query_indices), stream_positions=list(self.stream_positions),
                    write_location=self.write_location, selection_mode=self.selection_mode)

    def _after_read(self, module, args, kwargs, output):
        try:
            _require(self.active and module is self.penultimate_layer, 'Unexpected native read hook')
            _require(self._pending is None, 'Previous read did not reach the native final norm')
            self.last_capture = None
            is_tuple = isinstance(output, tuple)
            _require(not is_tuple or len(output) > 0, 'Empty decoder layer output')
            hidden = output[0] if is_tuple else output
            self._hidden(hidden)
            indices = torch.tensor(self.query_indices, device=hidden.device, dtype=torch.long)
            local = hidden[:-1].index_select(1, indices)
            global_states = hidden[-1].index_select(0, indices)
            self.read_calls += 1
            self.core_calls += 1
            result = self.branch(local, global_states, output_dtype=torch.float32, capture=self.capture)
            delta, values = result if self.capture else (result, None)
            _require(isinstance(delta, torch.Tensor) and delta.shape == global_states.shape
                     and delta.dtype == torch.float32 and delta.device == hidden.device
                     and bool(torch.isfinite(delta).all()), 'Require finite FP32 [Q,H] residuals')
            native_delta = delta.to(hidden.dtype)
            _require(bool(torch.isfinite(native_delta).all()), 'Native FP16 residual cast overflowed')
            cap = self._selection_capture(values, delta, local, global_states) if self.capture else None
            if cap is not None:
                cap['native_delta'] = self._remember(native_delta)
                cap['common_query_hidden'] = self._remember(hidden.index_select(1, indices))
                cap['final_block_input_queries'] = self._remember(hidden.index_select(1, indices))
            pending = dict(indices=indices, delta=delta, native_delta=native_delta,
                           shape=hidden.shape, device=hidden.device, dtype=hidden.dtype, capture=cap)
            replacement = None
            if self.write_location == 'pre_last':
                fused, before, after = self._fuse_native(hidden, indices, native_delta)
                if cap is not None:
                    cap.update(write_input=self._remember(before), write_output=self._remember(after),
                               final_block_input_queries=self._remember(fused.index_select(1, indices)))
                replacement = (fused,) + output[1:] if is_tuple else fused
            self._pending = pending
            return replacement
        except BaseException:
            self.close()
            raise

    def _before_norm(self, module, args, kwargs):
        try:
            _require(self.active and module is self.final_norm, 'Unexpected native final-norm hook')
            _require(self._pending is not None, 'Final norm ran without the common read boundary')
            positional = bool(args)
            _require(not (positional and 'hidden_states' in kwargs), 'Ambiguous native final-norm input')
            _require(positional or 'hidden_states' in kwargs, 'Missing native final-norm input')
            hidden = args[0] if positional else kwargs['hidden_states']
            self._hidden(hidden)
            pending = self._pending
            _require(hidden.shape == pending['shape'] and hidden.device == pending['device']
                     and hidden.dtype == pending['dtype'], 'Native layout/dtype/device changed between boundaries')
            indices, cap = pending['indices'], pending['capture']
            before = hidden[-1].index_select(0, indices)
            fused, after = hidden, before
            if self.write_location == 'post_last':
                fused, receiving, after = self._fuse_native(hidden, indices, pending['native_delta'])
                if cap is not None:
                    cap.update(write_input=self._remember(receiving), write_output=self._remember(after))
            if cap is not None:
                cap.update(norm_input_before_write=self._remember(before),
                           norm_input_after_write=self._remember(after),
                           native_query_hidden=self._remember(hidden.index_select(1, indices)),
                           fused_query_hidden=self._remember(fused.index_select(1, indices)))
            self.last_capture = cap
            self._pending = None
            self.norm_calls += 1
            self.calls += 1
            if self.write_location == 'pre_last':
                return None
            if positional:
                return (fused,) + args[1:], kwargs
            changed = dict(kwargs)
            changed['hidden_states'] = fused
            return args, changed
        except BaseException:
            self.close()
            raise

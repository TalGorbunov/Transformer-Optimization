"""Two native hooks for common reads and alternative residual write locations.

Software prototype only. Read the output of decoder block D-2 for every
explicitly supplied assistant-query position. Write either at that boundary
(`pre_last`) or before the final norm (`post_last`), using exactly those reads.
The caller freezes the complete native model and supplies N local rows followed
by one global row. No tokens, positions, attention masks or KV caches are built
or modified here; ordinary final-block attention owns its native KV updates.

Full-prefix callers MUST list every historical assistant-query position
P-1+t. Cached calls list only new positions in the current hidden tensor (often
[0]), together with their explicit absolute packed stream positions. A cached
sequence width is never treated as a position. The controller cannot verify
that callers omitted no historical query, or that row prefixes are identical;
those are packing/runtime responsibilities.
"""
from __future__ import annotations

import weakref
import torch

from gnnformer.parallel_local_aggregation import ParallelLocalAggregation


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _indices(values, name):
    _require(isinstance(values, (tuple, list)) and len(values) > 0,
             f"{name} must be a nonempty list or tuple")
    result = tuple(values)
    _require(all(isinstance(value, int) and not isinstance(value, bool) and value >= 0
                 for value in result), f"{name} must contain nonnegative integers")
    _require(all(a < b for a, b in zip(result, result[1:])),
             f"{name} must be strictly increasing")
    return result


class ParallelLocalMemory:
    """Context-managed fusion without registering a branch on the native model.

    `query_indices` index the CURRENT forward's hidden sequence. The aligned
    local/global rows use the same indices. `stream_positions` give the absolute
    right-aligned packed positions, not per-row unpadded positions or mRoPE IDs.
    Their offset from query_indices must be constant and nonnegative per call.
    Use configure_queries before each call whose layout changes.

    Two hooks are installed: a penultimate-layer forward hook and final-norm
    pre-hook. Local rows and unwritten positions are copied without mutation.
    Hooks preserve decoder tuple extras and positional/keyword norm arguments.
    `close` removes only this controller's hooks, also after an exception.

    Optional capture retains QUERY STATES ONLY, on the current device. Default
    captures are detached clones; detach_captures=False intentionally retains
    the most recent live graph, including the actual delta node used for fusion.
    Export always returns detached copies outside inference mode. To replay a
    complete last block, capture its full common input separately with a
    penultimate forward hook registered with prepend=True, before this write.
    """

    _owners = weakref.WeakKeyDictionary()

    def __init__(self, penultimate_layer, final_norm, branch, *, n_local_rows,
                 write_location, query_indices, stream_positions,
                 capture=False, detach_captures=True):
        _require(isinstance(penultimate_layer, torch.nn.Module)
                 and isinstance(final_norm, torch.nn.Module)
                 and penultimate_layer is not final_norm,
                 "Read layer and final norm must be distinct native modules")
        _require(isinstance(branch, ParallelLocalAggregation),
                 "branch must be a ParallelLocalAggregation")
        _require(isinstance(n_local_rows, int) and not isinstance(n_local_rows, bool)
                 and n_local_rows >= 0, "n_local_rows must be a nonnegative integer")
        _require(write_location in ('pre_last', 'post_last'),
                 "write_location must be pre_last or post_last")
        self.penultimate_layer, self.final_norm = penultimate_layer, final_norm
        self.branch, self.n_local_rows = branch, n_local_rows
        self.write_location = write_location
        self.capture, self.detach_captures = bool(capture), bool(detach_captures)
        self._handles, self._pending = [], None
        self.calls = self.read_calls = self.norm_calls = 0
        self.last_capture = None
        self.query_indices = self.stream_positions = ()
        self.configure_queries(query_indices, stream_positions)
        self._check_frozen()

    def _check_frozen(self):
        _require(not any(p.requires_grad for module in
                         (self.penultimate_layer, self.final_norm) for p in module.parameters()),
                 "Native read layer and final norm must already be frozen")

    @property
    def active(self):
        return bool(self._handles)

    def configure_queries(self, query_indices, stream_positions):
        """Set explicit layout between complete forwards; does not touch cache."""
        _require(self._pending is None, "Cannot change queries between read and norm hooks")
        queries, positions = _indices(query_indices, 'query_indices'), _indices(stream_positions, 'stream_positions')
        _require(len(queries) == len(positions), "Query and stream position counts differ")
        offsets = {position - query for query, position in zip(queries, positions)}
        _require(len(offsets) == 1 and next(iter(offsets)) >= 0,
                 "Stream positions must share a nonnegative offset from current query indices")
        self.query_indices, self.stream_positions = queries, positions

    def __enter__(self):
        _require(not self.active, "This memory controller is already active")
        self._check_frozen()
        for module in (self.penultimate_layer, self.final_norm):
            owner = self._owners.get(module)
            _require(owner is None or owner() is None,
                     "Another memory controller already owns this native boundary")
        self.calls = self.read_calls = self.norm_calls = 0
        self.last_capture, self._pending = None, None
        try:
            self._handles.append(self.penultimate_layer.register_forward_hook(
                self._after_read, with_kwargs=True))
            self._handles.append(self.final_norm.register_forward_pre_hook(
                self._before_norm, with_kwargs=True))
            for module in (self.penultimate_layer, self.final_norm):
                self._owners[module] = weakref.ref(self)
        except BaseException:
            self.close()
            raise
        return self

    def close(self):
        """Idempotent hook removal, preserving external hooks and native state."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._pending = None
        for module in (self.penultimate_layer, self.final_norm):
            owner = self._owners.get(module)
            if owner is not None and owner() is self:
                del self._owners[module]

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def assert_complete(self):
        """Check hook pairing after a successful top-level native forward."""
        _require(self._pending is None and self.read_calls == self.norm_calls == self.calls,
                 "Native forward did not complete exactly one norm call per read")

    def _hidden(self, hidden):
        _require(isinstance(hidden, torch.Tensor) and hidden.is_floating_point()
                 and hidden.ndim == 3 and hidden.shape[0] == self.n_local_rows + 1
                 and hidden.shape[1] > 0 and hidden.shape[2] == self.branch.hidden_size,
                 "Expected native hidden states [N+1,S,H], global row last")
        _require(self.query_indices[-1] < hidden.shape[1],
                 "Explicit query index lies outside the current forward sequence")

    def _remember(self, tensor):
        return tensor.detach().clone() if self.detach_captures else tensor

    @staticmethod
    def _fuse(hidden, indices, delta):
        receiving = hidden[-1].index_select(0, indices)
        updated = receiving + delta.to(dtype=hidden.dtype)
        fused = hidden.clone()
        fused[-1, indices, :] = updated
        return fused, receiving, updated

    def _after_read(self, module, args, kwargs, output):
        _require(self.active and module is self.penultimate_layer, "Unexpected read hook")
        _require(self._pending is None, "Previous read did not reach the native final norm")
        self._check_frozen()
        is_tuple = isinstance(output, tuple)
        _require(not is_tuple or len(output) > 0, "Empty decoder layer output")
        hidden = output[0] if is_tuple else output
        self._hidden(hidden)
        indices = torch.tensor(self.query_indices, device=hidden.device, dtype=torch.long)
        local_states = hidden[:-1].index_select(1, indices)
        global_states = hidden[-1].index_select(0, indices)
        delta = self.branch(local_states, global_states, output_dtype=torch.float32)
        _require(isinstance(delta, torch.Tensor) and delta.shape == global_states.shape
                 and delta.dtype == torch.float32 and delta.device == hidden.device
                 and bool(torch.isfinite(delta).all()), "Branch must return finite FP32 [Q,H] residuals")
        pending = dict(indices=indices, delta=delta, shape=hidden.shape,
                       device=hidden.device, dtype=hidden.dtype, capture=None)
        if self.capture:
            pending['capture'] = dict(
                local_states=self._remember(local_states), global_states=self._remember(global_states),
                delta=self._remember(delta), query_indices=list(self.query_indices),
                stream_positions=list(self.stream_positions), write_location=self.write_location)
        replacement = None
        if self.write_location == 'pre_last':
            fused, before, after = self._fuse(hidden, indices, delta)
            if self.capture:
                pending['capture'].update(write_input=self._remember(before), write_output=self._remember(after))
            replacement = (fused,) + output[1:] if is_tuple else fused
        self.read_calls += 1
        self._pending = pending
        return replacement

    def _before_norm(self, module, args, kwargs):
        _require(self.active and module is self.final_norm, "Unexpected final-norm hook")
        _require(self._pending is not None, "Final norm ran without the common read boundary")
        self._check_frozen()
        positional = bool(args)
        _require(not positional or 'hidden_states' not in kwargs, "Ambiguous final-norm hidden states")
        _require(positional or 'hidden_states' in kwargs, "Final norm is missing hidden states")
        hidden = args[0] if positional else kwargs['hidden_states']
        self._hidden(hidden)
        pending = self._pending
        _require(hidden.shape == pending['shape'] and hidden.dtype == pending['dtype']
                 and hidden.device == pending['device'], "Native layout/dtype/device changed between boundaries")
        before = hidden[-1].index_select(0, pending['indices'])
        fused, after = hidden, before
        if self.write_location == 'post_last':
            fused, receiving, after = self._fuse(hidden, pending['indices'], pending['delta'])
            if self.capture:
                pending['capture'].update(write_input=self._remember(receiving), write_output=self._remember(after))
        if self.capture:
            pending['capture'].update(norm_input_before_write=self._remember(before),
                                      norm_input_after_write=self._remember(after))
        self.last_capture = pending['capture']
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

    def export_last_capture(self, *, cpu=False):
        """Export detached normal tensor copies, including explicit layout metadata."""
        if self.last_capture is None:
            return None
        with torch.inference_mode(False), torch.no_grad():
            return {key: (value.detach().to('cpu').clone() if cpu else value.detach().clone())
                    if isinstance(value, torch.Tensor) else list(value) if isinstance(value, list) else value
                    for key, value in self.last_capture.items()}

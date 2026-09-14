"""Native final-norm evaluation of the exact frozen, paired factor cores.

Inputs and generation follow the frozen learned-selection runtime. A scoped
hook transforms the already-RMS local input using one fixed global mean/scale.
There are no local labels, probes, origin gates or changes to native KV. Each
current query has one factor call and its own monotonic conditioning capture.
"""
from contextlib import ExitStack
import hashlib
import importlib.util
from pathlib import Path
import time

from scripts import native_learned_selection_runtime as previous
from gnnformer.parallel_local_factor_binding import ParallelLocalFactorBinding, INTERACTIONS
from gnnformer.parallel_local_native import ParallelLocalNative, GlobalBroadcastLogitsProcessor
from scripts.native_vision_reasoning_stream import GlobalLogitRecorder

native=previous.native;need=native.need
prepare_scene=previous.prepare_scene
validate_bundle=previous.validate_bundle
append_observed_prefix=previous.append_observed_prefix
input_identity=previous.input_identity
OWN=tuple(dict.fromkeys((*previous.OWN,'gnnformer/parallel_local_factor_binding.py',
    'scripts/native_factor_binding_runtime.py','tests/test_native_factor_binding_runtime.py')))


def sources():
    root=Path(__file__).resolve().parents[1]
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in OWN}


def tensor_info(value):
    """Original shape/dtype, including scalar scale, with byte-exact fingerprint."""
    import torch
    with torch.inference_mode(False),torch.no_grad():
        copied=value.detach().cpu().contiguous()
        digest=hashlib.sha256(copied.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
    return dict(shape=list(value.shape),dtype=str(value.dtype),sha256=digest)


def conditioning_identity(global_mean,scale):
    return dict(global_mean=tensor_info(global_mean),scale=tensor_info(scale))


class NativeGlobalConditioning:
    """One persistent hook, one separately consumed capture for every query."""
    def __init__(self,core,global_mean,scale):
        import torch
        need(isinstance(global_mean,torch.Tensor) and isinstance(scale,torch.Tensor)
             and global_mean.dtype==scale.dtype==torch.float32 and global_mean.shape==(core.hidden_size,)
             and scale.shape==() and global_mean.device==scale.device==core.local.weight.device
             and not global_mean.requires_grad and not scale.requires_grad
             and global_mean.grad is None and scale.grad is None and bool(torch.isfinite(global_mean).all())
             and bool(torch.isfinite(scale)) and float(scale)>0,'Fixed global conditioning constants differ')
        self.core=core;self.global_mean=global_mean;self.scale=scale
        self._mean=global_mean;self._scale=scale;self._versions=(global_mean._version,scale._version)
        self.handle=None;self.calls=0;self.consumed=0
        self.original_local_rms_input=None;self.conditioned_local_rms_input=None

    @property
    def active(self):return self.handle is not None

    def guard(self):
        need(self.global_mean is self._mean and self.scale is self._scale
             and self._versions==(self.global_mean._version,self.scale._version)
             and not self.global_mean.requires_grad and not self.scale.requires_grad,
             'Frozen conditioning objects or versions changed')

    def __enter__(self):
        self.guard()
        need(not self.active and not getattr(self.core.local,'_fixed_conditioning_active',False)
             and not self.core.local._forward_pre_hooks,'Nested or foreign local conditioning hook')
        self.calls=self.consumed=0;self.original_local_rms_input=self.conditioned_local_rms_input=None
        def hook(module,args):
            import torch
            self.guard()
            need(module is self.core.local and len(args)==1 and args[0].dtype==torch.float32
                 and args[0].ndim==3 and args[0].shape[1:]==(1,self.core.hidden_size)
                 and self.calls==self.consumed,'Exactly one local projection per current query required')
            x=args[0];value=(x-self.global_mean.view(1,1,-1))/self.scale
            need(bool(torch.isfinite(x).all()) and bool(torch.isfinite(value).all()),'Nonfinite native conditioning input')
            self.calls+=1;self.original_local_rms_input=x;self.conditioned_local_rms_input=value
            return (value,)
        self.handle=self.core.local.register_forward_pre_hook(hook)
        self.core.local._fixed_conditioning_active=self
        return self

    def consume(self,expected_call):
        self.guard()
        need(self.active and type(expected_call) is int and self.calls==expected_call==self.consumed+1,
             'Missing, duplicate or stale per-query conditioning capture')
        self.consumed=expected_call
        return dict(original_local_rms_input=self.original_local_rms_input,
                    conditioned_local_rms_input=self.conditioned_local_rms_input)

    def close(self):
        if self.handle is not None:self.handle.remove();self.handle=None
        if getattr(self.core.local,'_fixed_conditioning_active',None) is self:del self.core.local._fixed_conditioning_active
        self.original_local_rms_input=self.conditioned_local_rms_input=None

    def __exit__(self,*_):self.close();return False


class FactorBindingNative(previous.LearnedSelectionNative):
    """Reuse the native global-only cast/write and lifecycle, with fixed factors."""
    def __init__(self,norm,branch,*,n_local_rows,interaction,global_mean,scale,capture=False):
        import torch
        need(type(branch) is ParallelLocalFactorBinding and interaction in INTERACTIONS
             and branch.interaction==interaction and branch.mode=='sigmoid'
             and branch.factor_derangement is False,'Exact original paired factor class required')
        ParallelLocalNative.__init__(self,norm,branch,n_local_rows=n_local_rows,capture=capture)
        self.interaction=interaction;self._initial_interaction=interaction
        self.conditioning=NativeGlobalConditioning(branch,global_mean,scale)
        self._parameters={name:p for name,p in branch.named_parameters()}
        self._parameter_versions={name:p._version for name,p in branch.named_parameters()}
        self.zero_up=bool((branch.up.weight==0).all());self.zero_up_identity_checks=0;self.last_capture_call=None
        self._mode_guard()

    def _mode_guard(self):
        import torch
        need(type(self.branch) is ParallelLocalFactorBinding and self.interaction==self._initial_interaction==self.branch.interaction
             and self.branch.mode=='sigmoid' and self.branch.factor_derangement is False
             and self.branch.merge=='sum' and self.branch.post_activation=='silu' and not self.branch.training
             and self.branch.local.bias is None,'Frozen original factor configuration changed')
        current=dict(self.branch.named_parameters())
        need(set(current)==set(self._parameters) and all(current[k] is self._parameters[k] and p._version==self._parameter_versions[k]
             and p.dtype==torch.float32 and p.device==self.norm.weight.device and not p.requires_grad and p.grad is None for k,p in current.items())
             and bool((self.branch.selection_weight==0).all()) and bool((self.branch.selection_bias==.5).all()),
             'Factor parameters/selector/dtype/device or trainability changed')
        self.conditioning.guard()

    def __enter__(self):
        self._mode_guard();need(self.norm not in self._owners,'Nested native fusion controller')
        self.zero_up_identity_checks=0;self.last_capture_call=None
        try:
            self.conditioning.__enter__()
            return super().__enter__()
        except BaseException:
            self.close();raise

    def close(self):
        super().close();self.conditioning.close()

    def _before_norm(self,module,args,kwargs):
        import torch
        self._mode_guard();need(self.conditioning.calls==self.calls,'Previous conditioning call was not uniquely consumed')
        hidden=args[0] if args else kwargs['hidden_states']
        result=super()._before_norm(module,args,kwargs)
        additions=self.conditioning.consume(self.calls);self.last_capture_call=self.calls
        if self.capture:self.last_capture.update({k:v.detach().clone() for k,v in additions.items()})
        if self.zero_up:
            fused=result[0][0] if args else result[1]['hidden_states']
            need(torch.equal(fused,hidden),'Zero-U native within-forward identity failed')
            if self.capture:need(bool((self.last_capture['delta']==0).all()),'Zero-U FP32 delta differs')
            self.zero_up_identity_checks+=1
        self._mode_guard();return result

    def audit(self):
        self._mode_guard()
        need(self.calls==self.conditioning.calls==self.conditioning.consumed,
             'Native factor/conditioning call counts differ')
        return dict(interaction=self.interaction,pairing='paired',calls=self.calls,
            conditioning_calls=self.conditioning.calls,last_capture_call=self.last_capture_call,
            zero_up=self.zero_up,zero_up_identity_checks=self.zero_up_identity_checks,
            parameter_versions_unchanged=True,statistics_versions_unchanged=True,
            active=self.active,conditioning_active=self.conditioning.active)


def native_contract(model,core,interaction,global_mean,scale):
    norm=native.native_contract(model,None)
    if core is None:
        need(interaction is None and global_mean is None and scale is None,'Bare execution has no factor/conditioning configuration')
    else:
        import torch
        need(type(core) is ParallelLocalFactorBinding and core.interaction==interaction and interaction in INTERACTIONS
             and core.mode=='sigmoid' and core.factor_derangement is False and core.hidden_size==3584 and core.rank==96
             and core.merge=='sum' and core.post_activation=='silu' and not core.training
             and sum(p.numel() for p in core.parameters())==1385857
             and all(p.dtype==torch.float32 and p.device==norm.weight.device and not p.requires_grad and p.grad is None for p in core.parameters())
             and bool((core.selection_weight==0).all()) and bool((core.selection_bias==.5).all()),
             'Require exact frozen/eval original1385857-parameter paired FP32 factor core')
        NativeGlobalConditioning(core,global_mean,scale).guard()
    return norm


_versions=previous._versions


def self_test(torch):
    """Only tiny CPU fixtures; caller must execute inside its Slurm CPU check."""
    path=Path(__file__).resolve().parents[1]/'tests/test_native_factor_binding_runtime.py'
    spec=importlib.util.spec_from_file_location('_factor_native_fixtures',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.self_test(torch)


def forward_native(model, bundle, core=None, *, interaction=None, global_mean=None, scale=None, native_identity_sha256,
                   capture=True, cpu=True, controller_observer=None):
    """One bare/fused uncached last-query forward at any supplied strict prefix."""
    native.require_slurm(gpu=True)
    import torch
    from gnnformer.runtime import get_rope_index_fn, move_to_device
    norm = native_contract(model, core, interaction, global_mean, scale); validate_bundle(bundle)
    binding = input_identity(bundle, native_identity_sha256)
    fixed_identity = None if core is None else conditioning_identity(global_mean, scale)
    layout = native.audit_layout(get_rope_index_fn(model), bundle)
    rope = getattr(model.model, 'rope_deltas', None); versions = _versions(model, core)
    counts = dict(model=0, visual=0, language=0, norm=0, head=0); observed = {}; fusion = None
    def count(key):
        def hook(*_): counts[key] += 1
        return hook
    def language(module, args, kwargs):
        counts['language'] += 1
        observed.update(position_ids=kwargs['position_ids'].detach().cpu().clone(),
                        attention_mask=kwargs['attention_mask'].detach().cpu().clone())
    def before_norm(module, args):
        counts['norm'] += 1
        if capture: observed.update(native.normal_copies(dict(native_query_hidden=args[0][:, -1:, :],
            local_states=args[0][:-1, -1:, :], global_states=args[0][-1, -1:, :]), cpu=cpu))
    start = time.perf_counter()
    with ExitStack() as stack:
        stack.callback(setattr, model.model, 'rope_deltas', rope)
        for handle in (model.register_forward_pre_hook(count('model')), model.model.visual.register_forward_pre_hook(count('visual')),
            model.model.language_model.register_forward_pre_hook(language, with_kwargs=True),
            norm.register_forward_pre_hook(before_norm), model.lm_head.register_forward_pre_hook(count('head'))): stack.callback(handle.remove)
        if core is not None:
            fusion = stack.enter_context(FactorBindingNative(norm, core, n_local_rows=bundle['metadata']['n_frames'],
                                         interaction=interaction, global_mean=global_mean, scale=scale, capture=capture))
        if controller_observer is not None: stack.callback(controller_observer, None); controller_observer(fusion)
        with torch.inference_mode(): output = model(**move_to_device(bundle['inputs'], model.device), use_cache=False, logits_to_keep=1)
        torch.cuda.synchronize()
        need(counts == dict(model=1, visual=1, language=1, norm=1, head=1) and (fusion is None or fusion.calls == 1),
             'One native forward/vision/norm/head and one optional branch call required')
        need(torch.equal(observed['position_ids'], layout['position_ids'])
             and torch.equal(observed['attention_mask'], bundle['inputs']['attention_mask'])
             and torch.equal(model.model.rope_deltas.detach().cpu(), layout['rope_deltas']), 'Actual full-prefix native layout differs')
        need(output.past_key_values is None and output.logits.dtype == torch.float16
             and output.logits.shape[:2] == (bundle['metadata']['row_count'], 1)
             and bool(torch.isfinite(output.logits).all()), 'Native uncached output contract differs')
        values = native.normal_copies(dict(native_logits=output.logits, global_logits=output.logits[-1, -1]), cpu=cpu)
        cap = None if fusion is None else fusion.export_last_capture(cpu=cpu)
    need(versions == _versions(model, core) and (fusion is None or not fusion.active)
         and (core is None or (core.interaction == interaction and core.factor_derangement is False)), 'Native/core mutation or leaked controller')
    return dict(values, **{k: v for k, v in observed.items() if k not in ('position_ids', 'attention_mask')},
        capture=cap, counters=dict(counts, fusion=0 if fusion is None else fusion.calls, conditioning=0 if fusion is None else fusion.conditioning.calls, probe_head=0),
        fusion_audit=None if fusion is None else fusion.audit(), capture_call_indices=[] if cap is None else [fusion.last_capture_call],
        metadata=dict(bundle['metadata'], interaction=interaction, conditioning_identity=fixed_identity, layout=layout['metadata'],
                      native_identity_sha256=native_identity_sha256, scene_input_identity=binding),
        model_seconds=time.perf_counter()-start)


def generate_native(model, processor, core, bundle, *, interaction=None, global_mean=None, scale=None, native_identity_sha256,
                    max_new_tokens=4, capture=True, controller_observer=None):
    """Four-token unmasked native generation, one initial vision pass, no origin probe."""
    native.require_slurm(gpu=True)
    import torch
    from transformers import LogitsProcessorList
    from gnnformer.runtime import get_rope_index_fn, move_to_device
    norm = native_contract(model, core, interaction, global_mean, scale); validate_bundle(bundle)
    meta = bundle['metadata']; width, batch = meta['prompt_width'], meta['row_count']
    need(not meta['prefix_ids'] and width == meta['original_prompt_width'], 'Generation starts from the original prompt')
    need(type(max_new_tokens) is int and max_new_tokens == 4, 'The native answer budget is four tokens')
    binding = input_identity(bundle, native_identity_sha256)
    fixed_identity = None if core is None else conditioning_identity(global_mean, scale)
    config, policy = native.generation_policy(model, processor.tokenizer, max_new_tokens=4)
    config.output_logits = False; config.output_scores = False
    policy = dict(policy, output_logits=False, output_scores=False, global_row_streaming=True)
    recorder = GlobalLogitRecorder(max_steps=4, full_vectors=True)
    broadcast = GlobalBroadcastLogitsProcessor(n_local_rows=meta['n_frames'], prompt_length=width)
    layout = native.audit_layout(get_rope_index_fn(model), bundle)
    rope = getattr(model.model, 'rope_deltas', None); versions = _versions(model, core)
    counts = dict(model=0, visual=0, language=0, norm=0, head=0)
    inputs_seen = []; position_rows = []; captures = []; capture_call_indices = []; fusion = None
    def before(module, args, kwargs):
        step = counts['model']; counts['model'] += 1
        ids, mask = kwargs['input_ids'], kwargs['attention_mask']
        need(ids.shape == (batch, width if step == 0 else 1) and mask.shape == (batch, width+step)
             and torch.equal(mask[:, :width].cpu(), bundle['inputs']['attention_mask'])
             and bool((mask[:, width:] == 1).all()), 'Native query/history mask differs')
        if step == 0:
            need(torch.equal(ids.cpu(), bundle['inputs']['input_ids']) and kwargs.get('pixel_values') is not None,
                 'Original prefill inputs differ')
        else:
            need(kwargs.get('pixel_values') is None and kwargs.get('past_key_values') is not None
                 and kwargs['past_key_values'].get_seq_length() == width+step-1
                 and torch.equal(ids, ids[-1:].expand_as(ids)), 'Cached global history or visual ownership differs')
        inputs_seen.append(ids.detach().cpu().clone())
    def count(key):
        def hook(*_): counts[key] += 1
        return hook
    def language(module, args, kwargs):
        step = counts['language']; counts['language'] += 1
        pos = kwargs['position_ids'].detach().cpu(); mask = kwargs['attention_mask'].detach().cpu()
        text = mask.long().cumsum(-1)-1
        expected = layout['position_ids'] if step == 0 else (layout['rope_deltas'].view(1, batch, 1)+width+step-1).expand(3, -1, -1)
        need(pos.shape == (4, batch, width if step == 0 else 1) and torch.equal(pos[1:], expected), 'Native mRoPE differs')
        need(torch.equal(pos[0][mask.bool()], text[mask.bool()]) if step == 0 else torch.equal(pos[0], text[:, -1:]),
             'Native text positions differ')
        position_rows.append(native.tensor_info(pos))
    def after(module, args, output):
        if fusion is not None:
            need(fusion.calls == counts['model'], 'Factors must be computed once at every current query')
            need(fusion.conditioning.calls == fusion.calls == fusion.conditioning.consumed == fusion.last_capture_call, 'Per-query factor conditioning capture differs')
            if capture:
                captures.append(fusion.export_last_capture(cpu=True)); capture_call_indices.append(fusion.last_capture_call)
    start = time.perf_counter()
    with ExitStack() as stack:
        stack.callback(setattr, model.model, 'rope_deltas', rope)
        if core is not None:
            fusion = stack.enter_context(FactorBindingNative(norm, core, n_local_rows=meta['n_frames'],
                                         interaction=interaction, global_mean=global_mean, scale=scale, capture=capture))
        if controller_observer is not None: stack.callback(controller_observer, None); controller_observer(fusion)
        for handle in (model.register_forward_pre_hook(before, with_kwargs=True), model.register_forward_hook(recorder),
            model.register_forward_hook(after), model.model.visual.register_forward_pre_hook(count('visual')),
            model.model.language_model.register_forward_pre_hook(language, with_kwargs=True),
            norm.register_forward_pre_hook(count('norm'), prepend=True), model.lm_head.register_forward_pre_hook(count('head'))): stack.callback(handle.remove)
        with torch.inference_mode():
            result = model.generate(**move_to_device(bundle['inputs'], model.device), generation_config=config,
                                   logits_processor=LogitsProcessorList([broadcast]), logits_to_keep=1)
        torch.cuda.synchronize()
        suffix = result.sequences[:, width:]; ids = suffix[-1].detach().cpu().tolist(); steps = len(ids)
        need(getattr(result, 'logits', None) is None and getattr(result, 'scores', None) is None, 'HF retained per-row output histories')
        need(1 <= steps <= 4 and torch.equal(suffix, suffix[-1:].expand_as(suffix))
             and counts == dict(model=steps, visual=1, language=steps, norm=steps, head=steps)
             and len(recorder.records) == broadcast.calls == steps
             and all(r['native_dtype'] == 'torch.float16' for r in recorder.records), 'Native generation/call inventory differs')
        need([r['top1_token_id'] for r in recorder.records] == ids
             and all(bool((inputs_seen[t] == ids[t-1]).all()) for t in range(1, steps)), 'Unmasked global argmax/history differs')
        eos = policy['native_eos_token_ids']; completed = ids[-1] in eos
        need(not any(x in eos for x in ids[:-1]) and (completed or steps == 4), 'Unexpected native answer stopping')
    need(versions == _versions(model, core) and (fusion is None or not fusion.active)
         and (core is None or (core.interaction == interaction and core.factor_derangement is False)), 'Native/core mutation or leaked controller')
    return dict(interaction=interaction, generated_ids=ids, raw_logits=torch.stack(recorder.vectors),
        fusion_audit=None if fusion is None else fusion.audit(), capture_call_indices=capture_call_indices,
        text=processor.tokenizer.decode(ids, skip_special_tokens=True), raw_text=processor.tokenizer.decode(ids, skip_special_tokens=False),
        completed=completed, truncated=not completed, finish_reason='eos' if completed else 'length',
        metadata=dict(meta, interaction=interaction, conditioning_identity=fixed_identity, generation=policy, layout=layout['metadata'],
            generation_position_ids=position_rows, native_identity_sha256=native_identity_sha256, scene_input_identity=binding),
        counters=dict(counts, broadcast=broadcast.calls, fusion=0 if fusion is None else fusion.calls, conditioning=0 if fusion is None else fusion.conditioning.calls, probe_head=0),
        logit_records=recorder.records, captures=captures, model_seconds=time.perf_counter()-start)

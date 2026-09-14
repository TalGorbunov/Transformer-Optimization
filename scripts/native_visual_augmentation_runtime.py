"""One intact-input residual before native Qwen block27; no execution release.

Only physical positions >= prompt_width-1 may be written. Frozen teacher
states are checkpoint/history specific. The optimized replay keeps unchanged
prefix K/V and recomputes ALL written suffix K/V with autograd on every call.
All numerical work requires Slurm; this module loads no model or checkpoint.
"""
from __future__ import annotations
from contextlib import contextmanager, ExitStack

from scripts import profile_mmred_official_native_memory as ordinary
from scripts import native_visual_memory_runtime as native
from scripts.stage_native_vision_v6_teacher import need, object_sha

p = ordinary.p
HIDDEN_SIZE = 3584
VOCAB_SIZE = 152064
LAYER = 27
MAX_TOKENS = 50
MEMORY_FIELDS = ('features', 'frame_index', 'raster_row', 'raster_col', 'grid_height', 'grid_width')


def _cpu(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().to('cpu').clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu(item) for item in value)
    return value


def _state(module):
    return {name: dict(object_id=id(value), storage=value.data_ptr(), version=value._version,
        shape=list(value.shape), dtype=str(value.dtype), requires_grad=value.requires_grad)
        for name, value in module.named_parameters()}


def _configuration(model):
    state=native._model_state(model)
    return dict(model_class=type(model).__module__+'.'+type(model).__name__,
        model_config=model.config.to_dict(),generation_config=model.generation_config.to_dict(),
        peft_config=state['peft_config'],lora_runtime=state['lora_runtime'],
        parameter_layout={name:{key:record[key] for key in ('shape','dtype','requires_grad')}
            for name,record in state['parameters'].items()})


def model_metadata(model, checkpoint_identity):
    """Caller has verified checkpoint bytes; record stable binding and live guard.

    No parameter hashing/loading occurs here. Keep this guard for the current
    model instance; a reload creates a new guard with the SAME checkpoint binding.
    External augmentation parameters are deliberately outside the frozen model.
    """
    p.native.require_slurm()
    native._frozen(model)
    language = model.model.language_model
    need(len(language.layers) == 28 and all(block.attention_type == 'full_attention'
        and block.self_attn.layer_idx == index and block.self_attn.config._attn_implementation == 'sdpa'
        for index, block in enumerate(language.layers)), 'Exact native28-layer full-attention SDPA model required')
    need(language.layers[LAYER].hidden_size == HIDDEN_SIZE
        and str(language.norm.weight.dtype) == str(model.lm_head.weight.dtype) == 'torch.float16'
        and model.lm_head.weight.shape == (VOCAB_SIZE, HIDDEN_SIZE), 'Original native hidden/norm/head geometry required')
    need(isinstance(checkpoint_identity, dict) and checkpoint_identity, 'Verified checkpoint/source identity required')
    configuration = _configuration(model)
    return dict(checkpoint_identity=checkpoint_identity, configuration_sha256=object_sha(configuration),
        configuration=configuration, runtime_state=native._model_state(model),
        checkpoint_bytes_verified_by_caller=True, parameter_bytes_rehashed_here=False)


def check_model(model, metadata):
    p.native.require_slurm()
    native._frozen(model)
    need(native._model_state(model) == metadata['runtime_state'], 'Frozen native parameter/storage/PEFT runtime changed')
    need(object_sha(_configuration(model))==metadata['configuration_sha256'], 'Frozen native configuration changed')


def memory_input(torch, model, features, coordinates):
    """Question-free input only; never pass a case, question or target to core."""
    need(set(coordinates) == set(MEMORY_FIELDS[1:]), 'Exact five native raster/frame coordinates required')
    result = dict(features=features.to(model.device), **{key: value.to(model.device) for key, value in coordinates.items()})
    _memory_valid(torch, result)
    return result


def _memory_valid(torch, value):
    need(isinstance(value, dict) and set(value) == set(MEMORY_FIELDS), 'Memory input must contain only visual features and coordinates')
    features = value['features']
    need(features.ndim == 2 and features.shape[1] == HIDDEN_SIZE and features.shape[0] > 0
        and features.dtype == torch.float16 and not features.requires_grad and bool(torch.isfinite(features).all()),
        'Finite frozen native FP16 postmerger features required')
    need(all(value[key].dtype == torch.int64 and value[key].shape == features.shape[:1]
        and value[key].device == features.device and not value[key].requires_grad for key in MEMORY_FIELDS[1:]),
        'Matching frozen int64 visual coordinates required')


def _memory_identity(value):
    return {key: p.tensor_info(value[key]) for key in MEMORY_FIELDS}


def _prepare(torch, core, value, evidence):
    _memory_valid(torch, value)
    evidence['memory_input_identity'] = _memory_identity(value)
    if core is None:
        return None
    need(all(parameter.dtype == torch.float32 and parameter.device == value['features'].device
        and bool(torch.isfinite(parameter).all()) for parameter in core.parameters()), 'Finite same-device FP32 branch required')
    memory = core.prepare_memory(**value)
    evidence['memory_capture'] = _cpu(memory)
    evidence['core_class'] = type(core).__module__+'.'+type(core).__name__
    return memory


def _delta(torch, hidden, core, memory, override, evidence):
    if override is not None:
        delta = override
        capture = dict(intervention_only=True)
    elif core is not None:
        delta, capture = core.delta_and_capture(hidden, memory)
    else:
        return None
    need(delta.dtype == torch.float32 and delta.shape == hidden.shape and delta.device == hidden.device
        and bool(torch.isfinite(delta).all()), 'One finite FP32 residual per eligible native row required')
    evidence.update(delta=_cpu(delta), core_capture=_cpu(capture), delta_override=override is not None)
    return delta


def _add(torch, hidden, delta):
    from gnnformer.native_visual_augmentation import native_residual_add
    return native_residual_add(hidden,delta)


@contextmanager
def _write_hook(torch, model, W, core, memory, evidence, *, delta_override=None, keep_full=False):
    """One hook, one placement; state and hooks restored even on exceptions."""
    name = '_native_visual_augmentation_active'
    need(not hasattr(model, name), 'Overlapping augmentation controllers are forbidden')
    setattr(model, name, True)
    records = []; live = []
    evidence['augmentation_calls'] = records
    def before(module, args, kwargs):
        need(not (args and 'hidden_states' in kwargs), 'Duplicate native block hidden input')
        hidden = args[0] if args else kwargs['hidden_states']
        positions = kwargs['cache_position']
        need(hidden.ndim == 3 and hidden.shape[0] == 1 and hidden.shape[2] == HIDDEN_SIZE
            and hidden.dtype == torch.float16 and positions.shape == hidden.shape[1:2], 'Native single-stream block27 inputs required')
        selected = positions >= W-1
        need(bool(selected.any()), 'Every augmentation call must include an eligible query')
        rows = hidden[:, selected, :]
        record = dict(cache_position=_cpu(positions), write_positions=_cpu(positions[selected]),
            lower_hidden_rows=_cpu(rows), position_embeddings=tuple(_cpu(v[..., selected, :]) for v in kwargs['position_embeddings']),
            native_input_shape=list(hidden.shape), placement='pre_block27', threshold=W-1)
        if keep_full:
            record.update(lower_hidden=_cpu(hidden), full_position_embeddings=_cpu(kwargs['position_embeddings']),
                actual_attention_mask=_cpu(kwargs['attention_mask']))
        records.append(record)
        delta = _delta(torch, rows, core, memory, delta_override, record)
        live.append(delta)
        if delta is None:
            record['native_addition_applied'] = False
            return None
        changed = hidden.clone()
        changed[:, selected, :] = _add(torch, rows, delta)
        need(torch.equal(changed[:, ~selected, :], hidden[:, ~selected, :]), 'A prefix position was modified')
        record.update(native_addition_applied=True, fused_hidden_rows=_cpu(changed[:, selected, :]))
        return ((changed,)+args[1:],kwargs) if args else (args,dict(kwargs,hidden_states=changed))
    try:
        with ExitStack() as stack:
            handle = model.model.language_model.layers[LAYER].register_forward_pre_hook(before, with_kwargs=True)
            stack.callback(handle.remove)
            yield live
    finally:
        delattr(model, name)
        evidence['augmentation_hook_removed'] = True


def _teacher_layout(torch, model, case, features):
    targets = case['target_ids']; L = len(targets); W = case['metadata']['prompt_width']
    need(1 <= L <= MAX_TOKENS and W > 1 and all(type(t) is int and 0 <= t < VOCAB_SIZE for t in targets),
        'One bounded native teacher target sequence required')
    packet = ordinary.ordinary_packet(torch, model, case, features, True)
    T = W+L-1
    need(packet['inputs_embeds'].shape == (1,T,HIDDEN_SIZE) and packet['position_ids'].shape == (4,1,T)
        and torch.equal(packet['position_ids'][0,0], torch.arange(T,device=model.device))
        and case['target_positions'] == list(range(W-1,T)), 'Exact prompt plus strict target prefix and native4-axis positions required')
    need(torch.equal(packet['input_ids'][:,:W], case['inputs']['input_ids'].to(model.device))
        and packet['input_ids'][0,W:].tolist() == targets[:-1], 'Teacher contains a future target or a changed prompt')
    return packet,W,L


def teacher_full(torch, model, case, features, core=None, *, metadata, evidence,
                 memory_input=None, delta_override=None, keep_full=False):
    """Full original decoder reference, native [1,L,V] head; caller backpropagates.

    use_cache=True harvests actual native post-RoPE K/V without extra projections.
    No dropout or native parameter training is enabled; residual autograd is live.
    """
    p.native.require_slurm(gpu=True); check_model(model, metadata)
    before = None if core is None else _state(core)
    packet,W,L = _teacher_layout(torch,model,case,features)
    expected_visual = globals()['memory_input'](torch,model,features,case['coordinates'])
    visual = expected_visual if memory_input is None else memory_input
    need(_memory_identity(visual)==_memory_identity(expected_visual), 'Branch and intact native inputs use different features/coordinates')
    memory = _prepare(torch,core,visual,evidence)
    evidence.update(expected_positions=[_cpu(packet['position_ids'])], target_ids=list(case['target_ids']),
        target_positions=list(case['target_positions']), prompt_width=W, checkpoint_identity=metadata['checkpoint_identity'],
        configuration_sha256=metadata['configuration_sha256'], teacher_use_cache=True)
    with native._observe(model,evidence,L,rope_deltas=packet['rope_deltas']) as counts:
        with _write_hook(torch,model,W,core,memory,evidence,delta_override=delta_override,keep_full=keep_full) as deltas:
            output = model(**{key:packet[key] for key in ('inputs_embeds','attention_mask','position_ids','cache_position')},
                use_cache=True,logits_to_keep=L,return_dict=True)
    ce = torch.nn.functional.cross_entropy(output.logits[0].float(),torch.tensor(case['target_ids'],device=model.device),reduction='none')
    evidence.update(position_ce=_cpu(ce), loss=float(ce.mean().detach()), final_block_calls=len(deltas))
    need(counts == dict(model=1,visual=0,language=1,norm=1,head=1,prefix_decoder=0)
        and len(deltas)==1 and output.logits.shape==(1,L,VOCAB_SIZE) and bool(torch.isfinite(ce).all()), 'Full teacher native calls/finite CE differ')
    check_model(model,metadata)
    need(core is None or _state(core)==before, 'Teacher mutated branch parameters')
    return dict(loss=ce.mean(),position_ce=ce,logits=output.logits,counters=counts,capture=evidence,
        deltas=deltas[0],past_key_values=output.past_key_values)


def capture_teacher(torch, model, case, features, *, metadata, evidence, keep_full=False):
    """Cache ONLY unchanged native states from exact full teacher, no residual.

    Complete lower_hidden/rotary are optional reference artifacts, not main cache.
    Caller saves packet in its designated data root and binds its serialization.
    """
    p.native.require_slurm(gpu=True)
    with torch.no_grad():
        result = teacher_full(torch,model,case,features,metadata=metadata,evidence=evidence,keep_full=keep_full)
    W=case['metadata']['prompt_width']; L=len(case['target_ids']); T=W+L-1
    cache=result['past_key_values']; record=evidence['augmentation_calls'][0]
    from transformers.cache_utils import DynamicCache, DynamicLayer
    need(type(cache) is DynamicCache and len(cache.layers)==28 and all(type(layer) is DynamicLayer
        and layer.get_seq_length()==T for layer in cache.layers), 'Actual complete28-layer native cache required')
    layer=cache.layers[LAYER]
    packet=dict(suffix_hidden=record['lower_hidden_rows'],prefix_key=_cpu(layer.keys[:,:,:W-1,:]),
        prefix_value=_cpu(layer.values[:,:,:W-1,:]),position_ids=_cpu(case['teacher_position_ids']),
        suffix_position_embeddings=record['position_embeddings'],rope_deltas=_cpu(case['rope_deltas']),
        input_ids=_cpu(case['teacher_input_ids']),attention_mask=torch.ones((1,T),dtype=torch.long),
        target_ids=list(case['target_ids']),prompt_width=W,target_positions=list(case['target_positions']),
        checkpoint_identity=metadata['checkpoint_identity'],configuration_sha256=metadata['configuration_sha256'],
        memory_input_identity=evidence['memory_input_identity'],layer_idx=LAYER,post_rotary_keys=True,
        prefix_length=W-1,full_teacher_width=T,all_written_states_recomputed=True,keep_full=keep_full)
    if keep_full:
        packet.update(lower_hidden=record['lower_hidden'],full_position_embeddings=record['full_position_embeddings'],
            full_attention_mask=record['actual_attention_mask'],reference_logits=_cpu(result['logits']))
    packet['tensor_identity']=_packet_tensor_identity(packet)
    packet['packet_identity']=object_sha({key:value for key,value in packet.items()
        if key not in _packet_tensor_fields(packet)})
    evidence['captured_packet_identity']=packet['packet_identity']
    # Do not return the other27 full K/V layers as a persistent cache artifact.
    return packet


def _packet_tensor_fields(packet):
    names=['suffix_hidden','prefix_key','prefix_value','position_ids','suffix_position_embeddings','rope_deltas','input_ids','attention_mask']
    if packet['keep_full']:
        names += ['lower_hidden','full_position_embeddings','full_attention_mask','reference_logits']
    return names


def _packet_tensor_identity(packet):
    def identity(value):
        if value is None:return None
        if isinstance(value,(tuple,list)):return [identity(v) for v in value]
        return p.tensor_info(value)
    return {key:identity(packet[key]) for key in _packet_tensor_fields(packet)}


def validate_packet(torch, packet, metadata):
    """Validate cached tensor bytes and stable checkpoint binding, never a model call."""
    p.native.require_slurm()
    need(packet['checkpoint_identity']==metadata['checkpoint_identity']
        and packet['configuration_sha256']==metadata['configuration_sha256'], 'Teacher cache belongs to another frozen checkpoint')
    need(_packet_tensor_identity(packet)==packet['tensor_identity']
        and object_sha({key:value for key,value in packet.items()
            if key not in _packet_tensor_fields(packet)+['packet_identity']})==packet['packet_identity'], 'Teacher cache tensor/metadata bytes changed')
    W=packet['prompt_width']; L=len(packet['target_ids']); T=W+L-1
    need(packet['layer_idx']==LAYER and packet['post_rotary_keys'] is True
        and packet['prefix_length']==W-1 and packet['full_teacher_width']==T
        and packet['suffix_hidden'].shape==(1,L,HIDDEN_SIZE) and packet['suffix_hidden'].dtype==torch.float16
        and packet['prefix_key'].shape==packet['prefix_value'].shape
        and packet['prefix_key'].shape==(1,4,W-1,128)
        and packet['prefix_key'].dtype==packet['prefix_value'].dtype==torch.float16
        and packet['position_ids'].shape==(4,1,T) and packet['input_ids'].shape==packet['attention_mask'].shape==(1,T)
        and bool((packet['attention_mask']==1).all()) and packet['target_positions']==list(range(W-1,T))
        and torch.equal(packet['position_ids'][0,0],torch.arange(T))
        and packet['input_ids'][0,W:].tolist()==packet['target_ids'][:-1], 'Native teacher prefix/suffix ownership differs')
    for key in ('suffix_hidden','prefix_key','prefix_value'):
        value=packet[key]
        need(not value.requires_grad and not torch.is_inference(value) and bool(torch.isfinite(value).all()), 'Cache must contain finite ordinary frozen tensors')


@contextmanager
def _observe_replay(torch, model, evidence, L):
    counts=dict(model=0,visual=0,language=0,norm=0,head=0,prefix_decoder=0)
    heads=[];shapes=[];latest={}
    evidence.update(counters=counts,profile_head=heads,shapes=shapes,final_block_calls=0)
    language=model.model.language_model
    def block_input(module,args,kwargs):
        evidence['final_block_calls']+=1
        hidden=args[0] if args else kwargs['hidden_states']
        evidence['actual_block_input_shape']=list(hidden.shape)
        evidence['actual_attention_mask']=_cpu(kwargs['attention_mask'])
        evidence['actual_cache_position']=_cpu(kwargs['cache_position'])
        evidence['actual_text_position_ids']=_cpu(kwargs['position_ids'])
    def before_norm(module,args):
        counts['norm']+=1
        latest.update(norm_input_shape=list(args[0].shape),norm_query_input=_cpu(args[0][:,-L:,:]))
    def after_norm(module,args,value):
        latest.update(norm_output_shape=list(value.shape),normalized_query=_cpu(value[:,-L:,:]))
    def after_head(module,args,value):
        counts['head']+=1
        heads.append(dict(norm_query_input=latest['norm_query_input'],normalized_query=latest['normalized_query'],
            head_input=_cpu(args[0]),head_logits=_cpu(value)))
        shapes.append(dict(norm_input_shape=latest['norm_input_shape'],norm_output_shape=latest['norm_output_shape'],
            head_input_shape=list(args[0].shape),head_output_shape=list(value.shape)))
        need(args[0].shape==(1,L,HIDDEN_SIZE) and value.shape==(1,L,VOCAB_SIZE)
            and args[0].dtype==value.dtype==torch.float16
            and torch.equal(heads[-1]['normalized_query'],heads[-1]['head_input']), 'Actual replay native head geometry/input differs')
    try:
        with ExitStack() as stack:
            for handle in (language.layers[LAYER].register_forward_pre_hook(block_input,with_kwargs=True),
                language.norm.register_forward_pre_hook(before_norm),language.norm.register_forward_hook(after_norm),
                model.lm_head.register_forward_hook(after_head)):
                stack.callback(handle.remove)
            yield counts
    finally:
        evidence['hooks_removed']=True
    need(evidence['final_block_calls']==counts['norm']==counts['head']==1, 'Exactly one actual final-block/norm/head replay required')


def replay_teacher(torch, model, packet, memory_input, core=None, *, metadata, evidence,
                   reference_full=False, delta_override=None):
    """One final-block replay; fresh prefix cache, all L suffix writes live.

    reference_full is a software-only reference with complete lower states.
    delta_override is a live FP32 intervention leaf for temporal/future tests.
    position_ce remains live so an individual causal loss can be differentiated.
    """
    p.native.require_slurm(gpu=True); check_model(model,metadata); validate_packet(torch,packet,metadata)
    need(not hasattr(model,'_native_visual_augmentation_active'), 'Replay cannot run under native augmentation hook')
    before=None if core is None else _state(core)
    need(_memory_identity(memory_input)==packet['memory_input_identity'], 'Replay uses another visual feature/coordinate packet')
    memory=_prepare(torch,core,memory_input,evidence)
    W=packet['prompt_width']; L=len(packet['target_ids']); T=W+L-1; device=model.device
    suffix=packet['suffix_hidden'].to(device)
    delta=_delta(torch,suffix,core,memory,delta_override,evidence)
    fused=suffix if delta is None else _add(torch,suffix,delta)
    language=model.model.language_model; block=language.layers[LAYER]
    if reference_full:
        need(packet['keep_full'], 'Full reference states were not captured')
        hidden=packet['lower_hidden'].to(device).clone();hidden[:,W-1:,:]=fused
        positions=packet['position_ids'].to(device);cache_position=torch.arange(T,device=device)
        rotary=tuple(value.to(device) for value in packet['full_position_embeddings'])
        mask=None if packet['full_attention_mask'] is None else packet['full_attention_mask'].to(device)
        cache=None
    else:
        from transformers.cache_utils import DynamicCache
        hidden=fused;positions=packet['position_ids'][:,:,W-1:].to(device)
        cache_position=torch.arange(W-1,T,device=device)
        rotary=tuple(value.to(device) for value in packet['suffix_position_embeddings'])
        # Do not use create_causal_mask's layer0 length for a layer27-only cache.
        # Native SDPA boolean semantics: True means a key is visible.
        mask=(torch.arange(T,device=device)[None,:] <= cache_position[:,None])[None,None,:,:]
        cache=DynamicCache(config=language.config)
        cache.update(packet['prefix_key'].to(device).clone(),packet['prefix_value'].to(device).clone(),LAYER)
        need(cache.get_seq_length(LAYER)==W-1 and block.self_attn.layer_idx==LAYER, 'Final-prefix cache layer/length differs')
    evidence.update(reference_full=reference_full,packet_identity=packet['packet_identity'],
        checkpoint_identity=metadata['checkpoint_identity'],position_ids=_cpu(positions),cache_position=_cpu(cache_position),
        attention_mask=_cpu(mask),position_embeddings=_cpu(rotary),fused_hidden_rows=_cpu(fused),
        prefix_length=W-1,all_written_suffix_kv_live=True,final_block_calls=0)
    with _observe_replay(torch,model,evidence,L) as counts:
        outputs=block(hidden_states=hidden,attention_mask=mask,position_ids=positions[0],past_key_values=cache,
            output_attentions=False,use_cache=cache is not None,cache_position=cache_position,position_embeddings=rotary)
        need(isinstance(outputs,tuple) and len(outputs)==1 and outputs[0].shape==hidden.shape, 'Unexpected native final-block output')
        norm_input=outputs[0];normalized=language.norm(norm_input);selected=normalized[:,-L:,:]
        logits=model.lm_head(selected)
    ce=torch.nn.functional.cross_entropy(logits[0].float(),torch.tensor(packet['target_ids'],device=device),reduction='none')
    evidence.update(position_ce=_cpu(ce),loss=float(ce.mean().detach()),target_ids=list(packet['target_ids']))
    need(logits.shape==(1,L,VOCAB_SIZE) and logits.dtype==torch.float16 and bool(torch.isfinite(logits).all())
        and bool(torch.isfinite(ce).all()) and evidence['final_block_calls']==1, 'One finite native replay/head required')
    if cache is not None:
        need(cache.get_seq_length(LAYER)==T and torch.equal(cache.layers[LAYER].keys[:,:,:W-1,:],packet['prefix_key'].to(device))
            and torch.equal(cache.layers[LAYER].values[:,:,:W-1,:],packet['prefix_value'].to(device)), 'Replay modified a frozen prefix K/V row')
        evidence['suffix_kv_identity']=dict(key=p.tensor_info(cache.layers[LAYER].keys[:,:,W-1:,:]),
            value=p.tensor_info(cache.layers[LAYER].values[:,:,W-1:,:]))
    check_model(model,metadata);need(core is None or _state(core)==before, 'Replay mutated branch parameters')
    return dict(loss=ce.mean(),position_ce=ce,logits=logits,counters=evidence['counters'],capture=evidence,
        deltas=delta,past_key_values=cache)


def history_case(torch, case, generated_ids):
    """Full causal replay of a fixed emitted history; no new generation decision."""
    p.native.require_slurm()
    ids=list(generated_ids); W=case['metadata']['prompt_width']; L=len(ids)
    need(1<=L<=MAX_TOKENS and all(type(token) is int and 0<=token<VOCAB_SIZE for token in ids)
        and not any(token in native.EOS_IDS for token in ids[:-1]), 'Bounded fixed native token history required')
    original=case['inputs']['input_ids'];positions=case['position_ids'];device=original.device
    need(original.shape==(1,W) and positions.shape==(4,1,W), 'Exact original prompt required')
    continuation=torch.arange(W,W+L-1,device=device)
    rotary=continuation.view(1,1,-1)+case['rope_deltas'].to(device).reshape(1,1,1)
    appended=torch.cat((continuation.view(1,1,-1),rotary.expand(3,1,-1)),dim=0)
    return dict(case,target_ids=ids,teacher_input_ids=torch.cat((original,torch.tensor([ids[:-1]],dtype=torch.long,device=device)),dim=1),
        teacher_position_ids=torch.cat((positions,appended),dim=-1),target_positions=list(range(W-1,W+L-1)),
        software_forced_history_only=True,original_gold_not_used=True)


def generate_cold(torch, model, processor, case, features, core=None, *, metadata, evidence,
                  max_tokens=MAX_TOKENS, memory_input=None):
    """Unchanged native greedy path with persistent layer27 writes; one head/token."""
    p.native.require_slurm(gpu=True);check_model(model,metadata)
    need(type(max_tokens) is int and 1<=max_tokens<=MAX_TOKENS, 'Declare one bounded native generation budget <=50')
    need(core is None or all(not module.training for module in core.modules())
        and all(not parameter.requires_grad and parameter.grad is None for parameter in core.parameters()), 'Generation branch must be frozen/eval')
    need(model.generation_config.eos_token_id==list(native.EOS_IDS) and processor.tokenizer.eos_token_id==151645,
        'Native EOS configuration changed')
    before=None if core is None else _state(core)
    expected_visual=globals()['memory_input'](torch,model,features,case['coordinates'])
    visual=expected_visual if memory_input is None else memory_input
    need(_memory_identity(visual)==_memory_identity(expected_visual), 'Branch/native image feature or coordinate bytes differ')
    with torch.no_grad():memory=_prepare(torch,core,visual,evidence)
    packet=ordinary.ordinary_packet(torch,model,case,features,False);W=packet['prompt_width'];delta=packet['rope_deltas'].to(model.device)
    ids=[];raw=[];expected=[];cache=None
    evidence.update(generated_ids=ids,raw_vectors=raw,expected_positions=expected,rope_deltas=_cpu(delta),
        checkpoint_identity=metadata['checkpoint_identity'],maximum_new_tokens=max_tokens)
    with native._observe(model,evidence,1,rope_deltas=delta) as counts:
        with _write_hook(torch,model,W,core,memory,evidence):
            with torch.inference_mode():
                for step in range(max_tokens):
                    start=0 if step==0 else W+step-1;length=W if step==0 else 1
                    positions=packet['position_ids'] if step==0 else torch.cat((torch.tensor([[[start]]],device=model.device),
                        (delta.reshape(1,1,1)+start).expand(3,1,1)),dim=0)
                    expected.append(_cpu(positions))
                    inp=dict(inputs_embeds=packet['inputs_embeds']) if step==0 else dict(input_ids=torch.tensor([[ids[-1]]],device=model.device))
                    output=model(**inp,attention_mask=torch.ones((1,start+length),dtype=torch.long,device=model.device),
                        position_ids=positions,cache_position=torch.arange(start,start+length,device=model.device),
                        past_key_values=cache,use_cache=True,logits_to_keep=1,return_dict=True)
                    need(cache is None or output.past_key_values is cache, 'Native cache object changed')
                    cache=output.past_key_values;need(cache.get_seq_length()==start+length, 'Native cache history length differs')
                    vector=output.logits[0,-1].detach().float().cpu();raw.append(vector);ids.append(int(vector.argmax()))
                    if ids[-1] in native.EOS_IDS:break
    G=len(ids)
    need(counts==dict(model=G,visual=0,language=G,norm=G,head=G,prefix_decoder=0)
        and len(evidence['augmentation_calls'])==G and all(bool(torch.isfinite(value).all()) for value in raw), 'One native model/head/write per output token required')
    for index,(observed,record,head) in enumerate(zip(evidence['native_inputs'],evidence['augmentation_calls'],evidence['profile_head'])):
        start=0 if index==0 else W+index-1;length=W if index==0 else 1
        need(observed['past_length']==start and not observed['has_pixels']
            and observed['cache_position'].tolist()==list(range(start,start+length))
            and observed['attention_mask'].shape==(1,start+length) and bool((observed['attention_mask']==1).all())
            and record['write_positions'].tolist()==[W+index-1]
            and torch.equal(head['head_logits'][0,-1].float(),raw[index]), 'Native greedy/write/head ownership differs')
        if index:need(observed['input_ids'].tolist()==[[ids[index-1]]], 'Generation did not consume its own token')
    check_model(model,metadata);need(core is None or _state(core)==before, 'Generation mutated branch parameters')
    return dict(generated_ids=ids,raw_logits=torch.stack(raw),completed=ids[-1] in native.EOS_IDS,
        truncated=ids[-1] not in native.EOS_IDS,text=processor.tokenizer.decode(ids,skip_special_tokens=True),
        raw_text=processor.tokenizer.decode(ids,skip_special_tokens=False),counters=counts,
        native_inputs=evidence['native_inputs'],native_positions=evidence['native_positions'],profile_head=evidence['profile_head'],
        shapes=evidence['shapes'],rope_restored=True,hooks_removed=True,native_positions_preserved=True,
        past_key_values=cache)

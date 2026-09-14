"""Continuous visual-prefix training and independent native question branches.

The memory encoder lives outside this module. Only a question-free text opening
and actual memory embeddings enter prefix construction. All tensor/model work
requires Slurm. This source releases neither training nor inference jobs.
"""
from __future__ import annotations
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
import hashlib
import json
from scripts import native_joint_prefix_cache as cache_runtime
from scripts import diagnose_native_identity_join_joint_lora_v2 as p
from scripts.stage_native_vision_v6_teacher import need, object_sha

HIDDEN_SIZE = 3584
VOCAB_SIZE = 152064
EOS_IDS = (151645, 151643)
DEFAULT_MAX_TOKENS = 50


def prepare_text(processor, system_prompt, question, target_text=None):
    """CPU-only exact native chat boundaries; no added question prefix.

    The caller binds the recovered official system/question and canonical JSON
    target strings. A single template image placeholder is replaced by memory.
    The returned prefix contains no question or target fields.
    """
    p.native.require_slurm()
    import torch
    need(isinstance(system_prompt, str) and system_prompt and isinstance(question, str) and question,
         'Exact nonempty system and unprefixed question strings required')
    tokenizer = processor.tokenizer
    def messages(text):
        return [dict(role='system', content=[dict(type='text', text=system_prompt)]),
                dict(role='user', content=[dict(type='image'), dict(type='text', text=text)])]
    def render(text):
        rendered = processor.apply_chat_template(messages(text), tokenize=False, add_generation_prompt=True)
        need(isinstance(rendered, str), 'One native rendered chat required')
        ids = tokenizer(rendered, add_special_tokens=False, return_tensors='pt')['input_ids']
        return rendered, ids
    special = [tokenizer.convert_tokens_to_ids(s) for s in ('<|vision_start|>', '<|image_pad|>', '<|vision_end|>')]
    need(len(set(special)) == 3 and all(isinstance(v, int) and v >= 0 for v in special), 'Native vision marker IDs required')
    rendered, ids = render(question)
    _, empty_ids = render('')
    def split(value):
        locations = [(value[0] == token).nonzero(as_tuple=True)[0] for token in special]
        need(value.ndim == 2 and value.shape[0] == 1 and all(len(v) == 1 for v in locations),
             'Exactly one unambiguous vision marker triplet required')
        a, b, c = [int(v[0]) for v in locations]
        need((b, c) == (a + 1, a + 2), 'Native image placeholder boundaries differ')
        return value[:, :b].clone(), value[:, c:c+1].clone(), value[:, c+1:].clone()
    opening, end, suffix = split(ids)
    empty_opening, empty_end, _ = split(empty_ids)
    need(torch.equal(opening, empty_opening) and torch.equal(end, empty_end) and suffix.shape[1] > 0,
         'Text opening must be exactly independent of question')
    prefix = dict(opening_ids=opening, end_ids=end, image_pad_id=special[1],
        system_sha256=hashlib.sha256(system_prompt.encode()).hexdigest(),
        chat_template_sha256=hashlib.sha256(str(processor.chat_template).encode()).hexdigest(),
        opening_identity=p.tensor_info(opening), end_identity=p.tensor_info(end), question_input_absent=True)
    prefix['identity'] = object_sha({k:v for k,v in prefix.items() if k not in ('opening_ids', 'end_ids')})
    target_ids = None
    if target_text is not None:
        need(isinstance(target_text, str) and set(json.loads(target_text)) == {'answer'}, 'Canonical answer JSON string required')
        joined = tokenizer(rendered + target_text, add_special_tokens=False, return_tensors='pt')['input_ids']
        need(torch.equal(joined[:, :ids.shape[1]], ids), 'Answer text changes the native prompt token boundary')
        target_ids = joined[0, ids.shape[1]:].tolist() + [151645]
        need(1 < len(target_ids) <= DEFAULT_MAX_TOKENS and not any(v in EOS_IDS for v in target_ids[:-1]),
             'Complete canonical JSON plus one EOS must fit the declared50-token budget')
        full = processor.apply_chat_template(messages(question) + [dict(role='assistant', content=[dict(type='text', text=target_text)])],
            tokenize=False, add_generation_prompt=False)
        full_ids = tokenizer(full, add_special_tokens=False, return_tensors='pt')['input_ids']
        endings = (full_ids[0,ids.shape[1]:] == 151645).nonzero(as_tuple=True)[0]
        need(len(endings)>0 and torch.equal(full_ids[:,:ids.shape[1]],ids)
             and full_ids[0,ids.shape[1]:ids.shape[1]+int(endings[0])+1].tolist()==target_ids,
             'Teacher target must equal full assistant chat through its first EOS, excluding trailing newline')
    return dict(prefix=prefix, suffix=dict(input_ids=suffix, token_identity=p.tensor_info(suffix),
        prefix_identity=prefix['identity'], question=question), target_ids=target_ids,
        rendered_prompt=rendered, original_prompt_ids=ids, target_text=target_text,
        full_assistant_template_target_exact=None if target_text is None else True)


def _prefix_valid(prefix):
    need(set(prefix) == {'opening_ids','end_ids','image_pad_id','system_sha256','chat_template_sha256',
        'opening_identity','end_identity','question_input_absent','identity'} and prefix['question_input_absent'] is True,
        'Only question-free text fields may enter memory prefill')
    need(p.tensor_info(prefix['opening_ids']) == prefix['opening_identity'] and p.tensor_info(prefix['end_ids']) == prefix['end_identity']
         and prefix['identity'] == object_sha({k:v for k,v in prefix.items() if k not in ('opening_ids','end_ids','identity')}),
         'Question-free prefix bytes changed')


def _positions(torch, start, length, device):
    return torch.arange(start, start+length, device=device).view(1,1,length).expand(4,1,length).clone()


def _prefix_embeddings(model, prefix, memory):
    import torch
    _prefix_valid(prefix)
    embedding = model.get_input_embeddings()
    need(not embedding.weight.requires_grad and embedding.weight.dtype == torch.float16,
         'Original frozen native FP16 token embeddings required')
    need(memory.ndim == 3 and memory.shape[0] == 1 and memory.shape[1] > 0 and memory.shape[2] == HIDDEN_SIZE
         and memory.dtype in (torch.float16, torch.float32) and bool(torch.isfinite(memory).all()),
         'Finite continuous memory [1,K,3584] required')
    memory_native = memory.to(device=embedding.weight.device, dtype=embedding.weight.dtype)
    need(bool(torch.isfinite(memory_native).all()), 'Memory cast overflowed native FP16')
    opening = prefix['opening_ids'].to(embedding.weight.device); end = prefix['end_ids'].to(embedding.weight.device)
    value = torch.cat((embedding(opening), memory_native, embedding(end)), dim=1)
    # These IDs describe layout only. Continuous slots are never re-embedded.
    layout_ids = torch.cat((opening, torch.full((1,memory.shape[1]), prefix['image_pad_id'],
        device=opening.device, dtype=torch.long), end), dim=1)
    metadata = dict(prefix_identity=prefix['identity'], memory_input=p.tensor_info(memory),
        memory_native=p.tensor_info(memory_native), prefix_embeddings=p.tensor_info(value),
        slots=memory.shape[1], prefix_length=value.shape[1], layout_ids_only=True, question_input_absent=True)
    return value, layout_ids, metadata


def compose_embeddings(model, text_packet, memory, *, teacher=False):
    """Preserves the memory/decoder autograd path, including its native cast."""
    p.native.require_slurm(gpu=True)
    import torch
    prefix, layout_ids, metadata = _prefix_embeddings(model, text_packet['prefix'], memory)
    suffix = text_packet['suffix']; need(suffix['prefix_identity'] == metadata['prefix_identity']
        and p.tensor_info(suffix['input_ids']) == suffix['token_identity'], 'Question suffix ownership changed')
    ids = suffix['input_ids'].to(prefix.device); target_ids = text_packet['target_ids']
    prompt_width = prefix.shape[1] + ids.shape[1]
    if teacher:
        need(target_ids is not None and 1 < len(target_ids) <= DEFAULT_MAX_TOKENS and target_ids[-1] == 151645
             and not any(v in EOS_IDS for v in target_ids[:-1]), 'Teacher requires complete canonical JSON/EOS IDs')
        ids = torch.cat((ids, torch.tensor([target_ids[:-1]], device=ids.device, dtype=torch.long)), dim=1)
    value = torch.cat((prefix, model.get_input_embeddings()(ids)), dim=1); width = value.shape[1]
    return dict(inputs_embeds=value, attention_mask=torch.ones((1,width), dtype=torch.long, device=value.device),
        position_ids=_positions(torch,0,width,value.device), cache_position=torch.arange(width,device=value.device),
        layout_ids=torch.cat((layout_ids,ids),dim=1), prompt_width=prompt_width, prefix_length=prefix.shape[1],
        target_ids=target_ids, metadata=dict(metadata, full_embedding_identity=p.tensor_info(value),
            four_axis_sequential=True, rope_delta=0, teacher=teacher))


def _model_state(model):
    return dict(parameters={name:dict(object_id=id(v),storage=v.data_ptr(),version=v._version,shape=list(v.shape),
        dtype=str(v.dtype),requires_grad=v.requires_grad) for name,v in model.named_parameters()},
        peft_config=p.canonical_peft_config(model) if hasattr(model,'peft_config') else None,
        lora_runtime={name:dict(active_adapters=list(v.active_adapters),disable_adapters=v.disable_adapters,
            merged_adapters=list(v.merged_adapters),scaling=dict(v.scaling),rank=dict(v.r),alpha=dict(v.lora_alpha),
            dropout={key:dict(p=drop.p,training=drop.training) for key,drop in v.lora_dropout.items()})
            for name,v in model.named_modules() if hasattr(v,'lora_A')})


def _frozen(model):
    need(all(not v.training for v in model.modules()) and all(not v.requires_grad and v.grad is None for v in model.parameters()),
         'Memory KV/generation requires a frozen evaluation model')


@contextmanager
def _observe(model, evidence, selected_rows, *, headless=False, rope_deltas=None):
    """Observe the actual calls, with caller-owned partial evidence on failure."""
    import torch
    norm=model.model.language_model.norm; head=model.lm_head
    counts=dict(model=0,visual=0,language=0,norm=0,head=0,prefix_decoder=0)
    inputs=[];positions=[];heads=[];shapes=[];latest={}
    evidence.update(counters=counts,native_inputs=inputs,native_positions=positions,profile_head=heads,shapes=shapes)
    rope_before=model.model.rope_deltas
    def bump(key):
        def hook(*_):counts[key]+=1
        return hook
    def before_backbone(module,args,kw):
        if headless:counts['prefix_decoder']+=1
        cache=kw.get('past_key_values'); ids=kw.get('input_ids'); emb=kw.get('inputs_embeds')
        inputs.append(dict(input_ids=None if ids is None else ids.detach().cpu().clone(),
            inputs_embeds_identity=None if emb is None else p.tensor_info(emb),
            attention_mask=kw['attention_mask'].detach().cpu().clone(),cache_position=kw['cache_position'].detach().cpu().clone(),
            past_length=0 if cache is None else cache.get_seq_length(),has_pixels=kw.get('pixel_values') is not None))
    def language(module,args,kw):
        counts['language']+=1;positions.append(kw['position_ids'].detach().cpu().clone())
        need(torch.equal(positions[-1],evidence['expected_positions'][len(positions)-1]), 'Actual four-axis positions differ')
    def before_norm(module,args):
        counts['norm']+=1;latest['norm_input_shape']=list(args[0].shape)
        if not headless:latest['norm_query_input']=args[0][:,-selected_rows:,:].detach().cpu().clone()
    def after_norm(module,args,value):
        latest['norm_output_shape']=list(value.shape)
        if not headless:latest['normalized_query']=value[:,-selected_rows:,:].detach().cpu().clone()
    def after_head(module,args,value):
        counts['head']+=1
        heads.append(dict(norm_query_input=latest['norm_query_input'],normalized_query=latest['normalized_query'],
            head_input=args[0].detach().cpu().clone(),head_logits=value.detach().cpu().clone()))
        shapes.append(dict(norm_input_shape=latest['norm_input_shape'],norm_output_shape=latest['norm_output_shape'],
            head_input_shape=list(args[0].shape),head_output_shape=list(value.shape)))
        need(args[0].shape==(1,selected_rows,HIDDEN_SIZE) and value.shape==(1,selected_rows,VOCAB_SIZE)
             and args[0].dtype==value.dtype==torch.float16 and bool(torch.isfinite(value).all())
             and torch.equal(heads[-1]['normalized_query'],heads[-1]['head_input']),
             'Actual selected native norm/head shape or dtype differs')
    with ExitStack() as stack:
        stack.callback(setattr,model.model,'rope_deltas',rope_before)
        model.model.rope_deltas=(torch.zeros((1,1),dtype=torch.long,device=model.device) if rope_deltas is None
            else rope_deltas.to(device=model.device,dtype=torch.long).clone())
        for handle in (model.register_forward_pre_hook(bump('model')),model.model.visual.register_forward_pre_hook(bump('visual')),
            model.model.register_forward_pre_hook(before_backbone,with_kwargs=True),
            model.model.language_model.register_forward_pre_hook(language,with_kwargs=True),norm.register_forward_pre_hook(before_norm),
            norm.register_forward_hook(after_norm),head.register_forward_hook(after_head)):stack.callback(handle.remove)
        yield counts
    need(model.model.rope_deltas is rope_before,'Native rope object restoration failed')
    evidence.update(rope_restored=True,hooks_removed=True)


def teacher_forward(model, text_packet, memory, *, evidence=None):
    """One cold forward; caller performs (loss / accumulation).backward()."""
    p.native.require_slurm(gpu=True)
    import torch
    import torch.nn.functional as F
    need(not model.model.visual.training, 'Frozen eval vision required; caller chooses decoder train/eval mode')
    evidence={} if evidence is None else evidence
    packet=compose_embeddings(model,text_packet,memory,teacher=True); target=packet['target_ids']; L=len(target)
    versions=_model_state(model)
    evidence.update(metadata=packet['metadata'],layout_ids=packet['layout_ids'].detach().cpu(),
        expected_positions=[packet['position_ids'].detach().cpu().clone()],target_ids=list(target),
        target_positions=list(range(packet['prompt_width']-1,packet['prompt_width']-1+L)))
    with _observe(model,evidence,L) as counts:
        output=model(**{k:packet[k] for k in ('inputs_embeds','attention_mask','position_ids','cache_position')},
            use_cache=False,logits_to_keep=L,return_dict=True)
    logits=output.logits; ce=F.cross_entropy(logits[0].float(),torch.tensor(target,device=logits.device),reduction='none');loss=ce.mean()
    evidence.update(position_ce=ce.detach().cpu().tolist(),loss=float(loss.detach()),prompt_width=packet['prompt_width'],
        decoder_training=model.training,gradient_enabled=torch.is_grad_enabled(),head_and_loss_finite=bool(torch.isfinite(ce).all()))
    need(output.past_key_values is None and counts==dict(model=1,visual=0,language=1,norm=1,head=1,prefix_decoder=0)
         and bool(torch.isfinite(ce).all()) and versions==_model_state(model), 'One finite cache-free teacher forward required')
    return dict(loss=loss,logits=logits,capture=evidence,counters=counts)


@dataclass(frozen=True)
class MemorySnapshot:
    prefix: cache_runtime.PrefixSnapshot
    metadata: dict
    model_state: dict


def _provenance(model, provenance):
    need(isinstance(provenance,dict) and all(k in provenance for k in ('native_identity_sha256','memory','lora'))
         and provenance['memory'] and provenance['lora'] and isinstance(provenance['native_identity_sha256'],str)
         and len(provenance['native_identity_sha256'])==64,
         'Caller must bind native, memory-weight and LoRA source/checkpoint identities')
    # The caller owns checkpoint/source validation. These hashes independently
    # identify the actual adapter bytes used by this running native model.
    return dict(caller_provenance=json.loads(json.dumps(provenance,sort_keys=True,allow_nan=False)),
        actual_lora_tensors={name:p.tensor_info(v) for name,v in p.adapter_parameters(model).items()})


def prefill_memory(model, prefix_packet, memory, *, provenance, evidence=None):
    """Question-free, headless native prefix; every question receives a fork."""
    p.native.require_slurm(gpu=True)
    import torch
    from transformers.cache_utils import DynamicCache, DynamicLayer
    _frozen(model);need(not memory.requires_grad,'Frozen memory embeddings required for inference cache')
    evidence={} if evidence is None else evidence;state=_model_state(model)
    with torch.inference_mode():value,ids,meta=_prefix_embeddings(model,prefix_packet,memory)
    meta.update(_provenance(model,provenance))
    L=value.shape[1];positions=_positions(torch,0,L,value.device);delta=torch.zeros((1,1),dtype=torch.long)
    evidence.update(metadata=meta,expected_positions=[positions.detach().cpu().clone()],prefix_embeddings=value.detach().cpu().clone())
    with _observe(model,evidence,1,headless=True) as counts:
        with torch.inference_mode():
            output=model.model(inputs_embeds=value,attention_mask=torch.ones((1,L),device=value.device,dtype=torch.long),
                position_ids=positions,cache_position=torch.arange(L,device=value.device),use_cache=True,return_dict=True)
        cache=output.past_key_values
        need(type(cache) is DynamicCache and len(cache.layers)==28 and all(type(v) is DynamicLayer for v in cache.layers), 'Exact native28-layer full cache required')
        layers=tuple((v.keys.detach().clone(),v.values.detach().clone()) for v in cache.layers);evidence['layers']=layers
        need(all(k.shape==v.shape==(1,4,L,128) and k.dtype==v.dtype==torch.float16 for k,v in layers)
             and counts==dict(model=0,visual=0,language=1,norm=1,head=0,prefix_decoder=1),'One headless continuous prefill required')
    need(state==_model_state(model),'Prefix prefill changed native model parameters')
    identity=cache_runtime._identity(layers,ids,positions,delta)
    prefix=cache_runtime.PrefixSnapshot(layers,ids.clone(),positions.clone(),delta,identity,provenance['native_identity_sha256'])
    meta=dict(meta,position_identity=p.tensor_info(positions),rope_delta=0)
    return MemorySnapshot(prefix,meta,state),evidence


def export_snapshot(torch, snapshot):
    """Detached audit packet; never a reusable training graph/checkpoint."""
    return dict(prefix=cache_runtime.export_snapshot(torch,snapshot.prefix),metadata=snapshot.metadata,model_state=snapshot.model_state)


def _greedy(model,processor,*,packet=None,snapshot=None,suffix=None,max_tokens=DEFAULT_MAX_TOKENS,evidence=None):
    p.native.require_slurm(gpu=True)
    import torch
    _frozen(model);need(type(max_tokens) is int and 1<=max_tokens<=DEFAULT_MAX_TOKENS,'Explicit1..50 free-greedy token budget required')
    need(model.generation_config.eos_token_id==list(EOS_IDS) and processor.tokenizer.eos_token_id==151645,'Original native EOS IDs required')
    evidence={} if evidence is None else evidence;state=_model_state(model);ids=[];raw=[];expected=[]
    if snapshot is None:
        need(packet is not None and suffix is None,'Cold embedding packet required');cache=None;start=0;W=packet['prompt_width']
        first=dict(inputs_embeds=packet['inputs_embeds']);positions=packet['position_ids'];metadata=packet['metadata']
    else:
        need(packet is None and suffix is not None and snapshot.model_state==state,'Cache belongs to another model state')
        need(suffix['prefix_identity']==snapshot.metadata['prefix_identity'] and p.tensor_info(suffix['input_ids'])==suffix['token_identity'], 'Cached prefix/question boundary differs')
        cache=cache_runtime.fork_cache(model,snapshot.prefix);start=snapshot.prefix.length;W=start+suffix['input_ids'].shape[1]
        first=dict(input_ids=suffix['input_ids'].to(model.device));positions=_positions(torch,start,W-start,model.device);metadata=snapshot.metadata
    evidence.update(metadata=metadata,generated_ids=ids,raw_vectors=raw,expected_positions=expected)
    with _observe(model,evidence,1) as counts:
        with torch.inference_mode():
            for step in range(max_tokens):
                begin=start if step==0 else W+step-1;length=W-start if step==0 else 1
                inputs=first if step==0 else dict(input_ids=torch.tensor([[ids[-1]]],device=model.device))
                pos=positions if step==0 else _positions(torch,begin,1,model.device);expected.append(pos.detach().cpu().clone())
                output=model(**inputs,attention_mask=torch.ones((1,begin+length),device=model.device,dtype=torch.long),
                    position_ids=pos,cache_position=torch.arange(begin,begin+length,device=model.device),past_key_values=cache,
                    use_cache=True,logits_to_keep=1,return_dict=True)
                need(cache is None or output.past_key_values is cache,'Native cache object replaced during generation')
                cache=output.past_key_values;need(cache.get_seq_length()==begin+length,'Actual native cache length differs')
                vector=output.logits[0,-1].detach().float().cpu();raw.append(vector);ids.append(int(vector.argmax()))
                if ids[-1] in EOS_IDS:break
    T=len(ids);need(counts==dict(model=T,visual=0,language=T,norm=T,head=T,prefix_decoder=0)
        and len(raw)==len(evidence['profile_head'])==T and all(bool(torch.isfinite(v).all()) for v in raw), 'One native head per actual emitted token required')
    for i,(observed,head) in enumerate(zip(evidence['native_inputs'],evidence['profile_head'])):
        begin=start if i==0 else W+i-1;length=W-start if i==0 else 1
        need(observed['past_length']==begin and not observed['has_pixels'] and observed['attention_mask'].shape==(1,begin+length)
             and bool((observed['attention_mask']==1).all()) and observed['cache_position'].tolist()==list(range(begin,begin+length))
             and torch.equal(head['head_logits'][0,-1].float(),raw[i]),'Actual generation mask/cache/head history differs')
        if i:need(observed['input_ids'].tolist()==[[ids[i-1]]],'Generation did not consume its own last output')
    if snapshot is not None:
        for layer,(key,value) in zip(cache.layers,snapshot.prefix.layers):
            need(torch.equal(layer.keys[:,:,:start],key) and torch.equal(layer.values[:,:,:start],value),'Question mutated immutable prefix KV')
        cache_runtime.verify_snapshot(snapshot.prefix)
    need(state==_model_state(model),'Generation changed model parameters');_frozen(model)
    completed=ids[-1] in EOS_IDS
    return dict(generated_ids=ids,raw_logits=torch.stack(raw),completed=completed,truncated=not completed,
        finish_reason='eos' if completed else 'length',text=processor.tokenizer.decode(ids,skip_special_tokens=True),
        raw_text=processor.tokenizer.decode(ids,skip_special_tokens=False),counters=counts,metadata=metadata,
        native_inputs=evidence['native_inputs'],native_positions=evidence['native_positions'],profile_head=evidence['profile_head'],shapes=evidence['shapes'],
        parameter_versions_unchanged=True,rope_restored=True,hooks_removed=True,prefix_bytes_unchanged=snapshot is not None,
        generation=dict(max_new_tokens=max_tokens,do_sample=False,num_beams=1,native_eos_token_ids=list(EOS_IDS),
            repetition_penalty=1.,vocabulary_mask=False,other_logits_processors=False,logits_to_keep=1))


def generate_cold(model,processor,text_packet,memory,*,provenance,max_tokens=DEFAULT_MAX_TOKENS,evidence=None):
    p.native.require_slurm(gpu=True)
    import torch
    _frozen(model);need(not memory.requires_grad,'Frozen memory embeddings required for generation')
    with torch.inference_mode():packet=compose_embeddings(model,text_packet,memory)
    packet['metadata'].update(_provenance(model,provenance))
    return _greedy(model,processor,packet=packet,max_tokens=max_tokens,evidence=evidence)


def generate_split(model,processor,snapshot,suffix_packet,*,max_tokens=DEFAULT_MAX_TOKENS,evidence=None):
    return _greedy(model,processor,snapshot=snapshot,suffix=suffix_packet,max_tokens=max_tokens,evidence=evidence)

"""Native parallel Cosmos generation with bounded global-only logit recording.

No fitting, gold-conditioned prefix, custom attention or custom KV cache.
All tensor/image work requires Slurm; this is a software integration utility.
The native model still computes N+1 rows per token in one batched invocation.
"""
from __future__ import annotations
from contextlib import ExitStack
import copy
import hashlib
import time
from scripts import native_vision_v7_runtime as base

need=base.need
SYSTEM=('You are a helpful assistant. Answer the question in the following format: '
        '<think>\nyour reasoning\n</think>\n\n<answer>\nyour answer\n</answer>.')


def prompt(question):
    need(isinstance(question,str) and bool(question.strip()),'A question is required')
    return ('You will be shown frames describing steps in a house.\n'
            'Question: '+question+'\nThe final answer must be a single non-negative integer.')


def prepare_scene(processor,sample,*,verify_processor_parity=True):
    """Identical reasoning system/question for local N1 rows and text-only global.

    The source SID, images and question are the only sample fields consumed.
    No answer, frame count range or forced integer-only assistant instruction.
    """
    base.require_slurm()
    import torch
    from PIL import Image
    frames=[];rows=[];conversations=[];old=processor.tokenizer.padding_side
    try:
        for item in sample['image_files']:
            from pathlib import Path
            path=Path(item['path'])
            need(hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'],'Image bytes changed')
            with Image.open(path) as picture:
                rgb=picture.convert('RGB')
                try:frame=rgb.resize((392,392))
                finally:rgb.close()
            frames.append(frame)
        need(len(frames)==sample['n_frames'] and len(frames)>0,'Frame count differs')
        text=prompt(sample['question'])
        for frame in frames+[None]:
            content=[] if frame is None else [dict(type='image',image=frame)]
            content.append(dict(type='text',text=text))
            conversations.append([dict(role='system',content=[dict(type='text',text=SYSTEM)]),
                                  dict(role='user',content=content)])
        for conversation in conversations:
            rows.append(dict(processor.apply_chat_template(conversation,add_generation_prompt=True,
                tokenize=True,return_dict=True,return_tensors='pt')))
        packed=base.pack_rows(rows,processor.tokenizer.pad_token_id)
        need('pixel_values' not in rows[-1] and all(tuple(r['image_grid_thw'].shape)==(1,3) for r in rows[:-1]),
             'Local/global visual ownership differs')
        if verify_processor_parity:
            processor.tokenizer.padding_side='left'
            actual=dict(processor.apply_chat_template(conversations,add_generation_prompt=True,tokenize=True,
                return_dict=True,return_tensors='pt',padding=True))
            need(set(actual)==set(packed) and all(torch.equal(actual[k],packed[k]) for k in packed),
                 'Packed rows differ from ordinary native processor')
        width=int(packed['input_ids'].shape[1]);n=len(frames)
        meta=dict(arm='parallel',sid=sample['sid'],n_frames=n,question=sample['question'],
            question_sha256=base.object_sha(sample['question']),system_prompt=SYSTEM,global_prompt=text,local_prompt=text,
            row_count=n+1,global_row=n,local_elements=n,row_kinds=['local']*n+['global'],
            original_prompt_width=width,prompt_width=width,row_prompt_tokens=[int(r['input_ids'].shape[1]) for r in rows],
            prefix_ids=[],resize=392,image_paths=[r['path'] for r in sample['image_files']],
            image_sha256=[r['sha256'] for r in sample['image_files']],
            input_identity={k:base.tensor_info(v) for k,v in packed.items()},
            processor_parity_checked=verify_processor_parity)
        return dict(inputs=packed,row_inputs=rows,metadata=meta)
    finally:
        processor.tokenizer.padding_side=old
        for frame in frames:frame.close()


def prefixed_bundle(bundle,ids):
    """Append actual emitted IDs verbatim, including reasoning special tokens.

    Callers replay prefix[:step], hence exclude the terminal EOS predicting stop.
    No tokenizer special-ID exclusion or decode/re-tokenize operation occurs.
    """
    value=dict(inputs=base.append_prefix(bundle['inputs'],ids),
               row_inputs=[base.append_prefix(row,ids) for row in bundle['row_inputs']],
               metadata=copy.deepcopy(bundle['metadata']))
    value['metadata']['prefix_ids']=list(bundle['metadata']['prefix_ids'])+list(ids)
    value['metadata']['prompt_width']+=len(ids)
    value['metadata']['input_identity']={k:base.tensor_info(v) for k,v in value['inputs'].items()}
    return value


class GlobalLogitRecorder:
    """Nonmutating model output hook; no persistent all-row or device logits.

    Full vectors are permitted only for <=8-token software probes. Longer runs
    retain bounded per-token scalars (or send them to a caller-owned sink).
    No longer generation is authorized merely by this utility supporting it.
    """
    def __init__(self,*,max_steps,full_vectors=False,sink=None):
        need(type(max_steps) is int and 1<=max_steps<=4096,'Invalid recorder bound')
        need(not full_vectors or max_steps<=8,'Full vectors are restricted to the short software smoke')
        self.max_steps=max_steps;self.full_vectors=full_vectors;self.sink=sink
        self.records=[];self.vectors=[]

    def __call__(self,module,args,output):
        base.require_slurm()
        import torch
        need(len(self.records)<self.max_steps,'More native forwards than generation budget')
        logits=output.logits
        need(logits.ndim==3 and logits.shape[1]==1 and bool(torch.isfinite(logits[-1,-1]).all()),
             'Expected finite last-query global logits')
        # This exactly follows HF's FP32 promotion before its processors, but
        # copies only the global row. Never return a replacement model output.
        global_logits=logits[-1,-1].detach().float()
        top=int(global_logits.argmax())
        record=dict(step=len(self.records),top1_token_id=top,top1_logit=float(global_logits[top]),
                    log_normalizer=float(torch.logsumexp(global_logits,-1)),native_dtype=str(logits.dtype),
                    vocabulary_size=global_logits.numel())
        self.records.append(record)
        if self.full_vectors:self.vectors.append(global_logits.cpu().clone())
        if self.sink is not None:self.sink(dict(record))
        return None


def contract(model,branch=None):
    base.require_slurm(gpu=True)
    import torch
    need(not model.training and not any(p.requires_grad for p in model.parameters()),'Native model must be frozen/eval')
    norm=model.model.language_model.norm;head=model.lm_head
    need(norm.weight.dtype==head.weight.dtype and norm.weight.dtype in (torch.float16,torch.bfloat16),
         'Keep actual native low-precision norm/head; do not recast')
    need(model.config.text_config._attn_implementation=='sdpa','Native SDPA required')
    if branch is not None:
        need(branch.hidden_size==3584 and branch.rank==96 and branch.merge=='sum' and branch.post_activation=='silu'
             and sum(p.numel() for p in branch.parameters())==1041600
             and all(p.dtype==torch.float32 and p.device==norm.weight.device for p in branch.parameters()),
             'Native branch shape/dtype/operator differs')
    return norm


def generation_policy(model,tokenizer,max_new_tokens):
    base.require_slurm()
    from transformers import GenerationConfig
    need(type(max_new_tokens) is int and 1<=max_new_tokens<=4096,'Unbounded output budget')
    eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else list(eos)
    need(eos==[151645,151643] and tokenizer.eos_token_id==151645,'Cosmos EOS identity differs')
    config=GenerationConfig(max_new_tokens=max_new_tokens,do_sample=False,num_beams=1,num_return_sequences=1,
        repetition_penalty=1.,use_cache=True,bos_token_id=model.generation_config.bos_token_id,
        eos_token_id=eos,pad_token_id=tokenizer.pad_token_id,return_dict_in_generate=True,
        output_logits=False,output_scores=False)
    return config,dict(max_new_tokens=max_new_tokens,do_sample=False,num_beams=1,repetition_penalty=1.,
        use_cache=True,native_eos_token_ids=eos,output_logits=False,output_scores=False,
        global_row_streaming=True,vocabulary_mask=False)


def generate_native(model,processor,branch,bundle,*,max_new_tokens=8,full_vectors=False):
    """One ordinary batched model call per generated token; one visual prefill."""
    base.require_slurm(gpu=True)
    import torch
    from transformers import LogitsProcessorList
    from gnnformer.runtime import move_to_device,get_rope_index_fn
    from gnnformer.parallel_local_native import ParallelLocalNative,GlobalBroadcastLogitsProcessor
    norm=contract(model,branch);meta=bundle['metadata'];batch=meta['row_count'];width=meta['prompt_width']
    layout=base.audit_layout(get_rope_index_fn(model),bundle)
    config,policy=generation_policy(model,processor.tokenizer,max_new_tokens)
    recorder=GlobalLogitRecorder(max_steps=max_new_tokens,full_vectors=full_vectors)
    broadcast=GlobalBroadcastLogitsProcessor(n_local_rows=meta['n_frames'],prompt_length=width)
    counts=dict(model=0,visual=0,language=0);steps=[];positions=[]
    def before(module,args,kwargs):
        i=counts['model'];counts['model']+=1;ids=kwargs['input_ids'];mask=kwargs['attention_mask']
        need(tuple(ids.shape)==(batch,width if i==0 else 1),'Native prefill/decode query shape differs')
        need(mask.shape==(batch,width+i) and torch.equal(mask[:,:width].cpu(),bundle['inputs']['attention_mask'])
             and bool((mask[:,width:]==1).all()),'Native growing padding mask differs')
        if i==0:
            need(torch.equal(ids.cpu(),bundle['inputs']['input_ids']) and kwargs.get('pixel_values') is not None,
                 'Prefill changed exact input/images')
        else:
            need(kwargs.get('pixel_values') is None and kwargs.get('past_key_values') is not None
                 and kwargs['past_key_values'].get_seq_length()==width+i-1
                 and torch.equal(ids,ids[-1:].expand_as(ids)),'Cached common-prefix/visual ownership differs')
        steps.append(ids.detach().cpu().clone())
    def visual(*_):counts['visual']+=1
    def language(module,args,kwargs):
        i=counts['language'];counts['language']+=1;actual=kwargs['position_ids'].detach().cpu()
        mask=kwargs['attention_mask'].detach().cpu();text=mask.cumsum(-1)-1
        need(actual.shape==(4,batch,width if i==0 else 1),'Native position axes differ')
        expected=layout['position_ids'] if i==0 else (layout['rope_deltas'].view(1,batch,1)+width+i-1).expand(3,-1,-1)
        need(torch.equal(actual[1:],expected),'Native mRoPE differs')
        need(torch.equal(actual[0][mask.bool()],text[mask.bool()]) if i==0 else torch.equal(actual[0],text[:,-1:]),
             'Native text positions differ')
        positions.append(base.tensor_info(actual))
    started=time.perf_counter()
    with ExitStack() as stack:
        fusion=stack.enter_context(ParallelLocalNative(norm,branch,n_local_rows=meta['n_frames'])) if branch is not None else None
        for handle in (model.register_forward_pre_hook(before,with_kwargs=True),model.register_forward_hook(recorder),
                       model.model.visual.register_forward_pre_hook(visual),
                       model.model.language_model.register_forward_pre_hook(language,with_kwargs=True)):
            stack.callback(handle.remove)
        with torch.inference_mode():
            result=model.generate(**move_to_device(bundle['inputs'],model.device),generation_config=config,
                logits_processor=LogitsProcessorList([broadcast]),logits_to_keep=1)
        torch.cuda.synchronize();suffix=result.sequences[:,width:];ids=suffix[-1].cpu().tolist();n=len(ids)
        need(getattr(result,'logits',None) is None and getattr(result,'scores',None) is None,
             'HF retained per-row logits/scores unexpectedly')
        need(torch.equal(suffix,suffix[-1:].expand_as(suffix)) and 1<=n<=max_new_tokens
             and counts==dict(model=n,visual=1,language=n) and broadcast.calls==n
             and len(recorder.records)==n and (fusion is None or fusion.calls==n),'Native batched call/history count differs')
        need([r['top1_token_id'] for r in recorder.records]==ids,'Generation differs from streamed raw global argmax')
        need(all(bool((steps[i]==ids[i-1]).all()) for i in range(1,n)),'Executed cached token differs from chosen token')
        completed=ids[-1] in policy['native_eos_token_ids']
        need(not any(x in policy['native_eos_token_ids'] for x in ids[:-1]) and (completed or n==max_new_tokens),
             'Unexpected native stopping')
    return dict(generated_ids=ids,raw_text=processor.tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False),
        completed=completed,truncated=not completed,raw_logits=torch.stack(recorder.vectors) if full_vectors else None,
        logit_records=recorder.records,metadata=dict(meta,generation=policy,layout=layout['metadata'],
            native_dtype=str(norm.weight.dtype),generation_position_ids=positions),
        counters=dict(counts,broadcast=broadcast.calls,fusion=0 if branch is None else n),
        model_seconds=time.perf_counter()-started)


def forward_native(model,bundle,branch):
    """Uncached actual-prefix reference; local rows stay intact."""
    base.require_slurm(gpu=True)
    import torch
    from gnnformer.runtime import move_to_device,get_rope_index_fn
    from gnnformer.parallel_local_native import ParallelLocalNative
    norm=contract(model,branch);layout=base.audit_layout(get_rope_index_fn(model),bundle)
    with ParallelLocalNative(norm,branch,n_local_rows=bundle['metadata']['n_frames']),torch.inference_mode():
        result=model(**move_to_device(bundle['inputs'],model.device),use_cache=False,logits_to_keep=1)
    need(result.past_key_values is None,'Uncached reference used/returned a cache')
    return dict(global_logits=result.logits[-1,-1].detach().float().cpu().clone(),layout=layout['metadata'])

"""Synthetic correct-count latent -> learned readout -> native first token.

This oracle bypasses local evidence extraction entirely. Its scores are NOT
MMReD vision accuracy, complete-answer generation, or a reasoning result.
Frozen protocol in PREREG_AGG.md; all numerical work requires Slurm.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts.stage_native_vision_v6_teacher import MODEL,need,read,save,sha,object_sha,model_metadata,verify_plan
from scripts.stage_native_vision_v6_teacher import SOURCES as TEACHER_SOURCES
DATA=Path('/mnt/data/gabriele/gnn_transformer/readout_capacity')
CKPTS=Path('/mnt/ckpts/gabriele/gnn_transformer/readout_capacity')
OUT=REPO/'outputs/native_aggregation_vlm/readout_capacity'
TEACHER=Path('/mnt/data/gabriele/gnn_transformer/v6_local_teacher/teacher_cache.json')
OWN=tuple(sorted(set(TEACHER_SOURCES)|{'scripts/probe_native_vision_readout_capacity.py',
    'gnnformer/parallel_local_aggregation.py','tests/test_parallel_local_aggregation.py',
    'slurm/parallel_local_aggregation_check.sbatch','slurm/native_vision_readout_capacity_check.sbatch',
    'slurm/native_vision_readout_capacity.sbatch'}))
SEED=20260923
POLICY=dict(seed=SEED,rank=96,steps=500,learning_rate=.001,weight_decay=0.,clip_norm=1.,
            oracle_latent='z=K*e1',fit_rows=72,held_rows_per_cell=72,counts=list(range(9)),
            n_frames=[8,16,32,64],fit_capacity_accuracy=1.,native_replay_tv_max=.02,
            activations=['identity','silu'],selection='final_checkpoint_only')


def sources():return {name:sha(REPO/name) for name in OWN}


def snapshot(out):
    (out/'source').mkdir()
    for name in OWN:(out/'source'/name.replace('/','_')).write_bytes((REPO/name).read_bytes())
    save(out/'source_hashes.json',sources())


def tensor_hash(tensor):
    import torch
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def state_hash(state):
    return object_sha({k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=tensor_hash(v)) for k,v in state.items()})


def text_inputs(processor,question,n):
    from gnnformer.data import build_count_prompt
    messages=[dict(role='user',content=[dict(type='text',text=build_count_prompt(question,n))])]
    result=dict(processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=True,
                                             return_dict=True,return_tensors='pt'))
    need(set(result)=={'input_ids','attention_mask'},'Text-only prompt has unexpected inputs')
    return result


def rows_for(questions):
    prompts=[dict(question=q,question_index=qi,n_frames=n) for qi,q in enumerate(questions) for n in POLICY['n_frames']]
    rows=[]
    for pi,p in enumerate(prompts):
        cell='fit' if p['question_index']<4 and p['n_frames']<=16 else \
             'new_questions' if p['n_frames']<=16 else 'new_lengths' if p['question_index']<4 else 'both'
        rows.extend(dict(prompt_index=pi,k=k,cell=cell) for k in range(9))
    return prompts,rows


def tests():
    import torch
    from collections import Counter
    from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
    prompts,rows=rows_for([f'q{i}' for i in range(8)])
    need(len(prompts)==32 and Counter(x['cell'] for x in rows)==Counter(dict(fit=72,new_questions=72,new_lengths=72,both=72)),
         'Oracle split coverage differs')
    torch.manual_seed(SEED);branch=ParallelLocalAggregation(8,rank=4)
    h=torch.randn(9,8);z=torch.zeros(9,4);z[:,0]=torch.arange(9)
    need(torch.equal(branch.decode(z,h),torch.zeros_like(h)),'Zero-U oracle is not native identity')
    branch.decode(z,h).square().sum().backward()
    need(branch.local.weight.grad is None and branch.local_bias.grad is None,'Bypass touched local encoder')
    torch.nn.init.normal_(branch.up.weight,std=.01);branch.zero_grad(set_to_none=True)
    branch.decode(z,h).square().sum().backward()
    need(all(getattr(branch,n).weight.grad is not None for n in ('query','aggregate_projection','up')),
         'Native readout gradient route absent')
    return dict(passed=True,scope='split coverage, zero-U identity, bypass and readout gradient routes')


def check():
    import torch,transformers
    from transformers import AutoProcessor
    from scripts.probe_native_vision_v2_prefix import fingerprint
    torch.set_num_threads(4);started=time.perf_counter();self_test=tests()
    cache=read(TEACHER);teacher_path=Path(cache['plan_file']);teacher=verify_plan(teacher_path)
    need(str(torch.__version__)==teacher['runtime']['torch_version'] and str(transformers.__version__)==teacher['runtime']['transformers_version'],'CPU runtime differs from teacher')
    questions=sorted({pair['question'] for pair in teacher['pairs'].values()})[:8]
    need(len(questions)==8,'Missing fixed training questions')
    processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,use_fast=False)
    need(fingerprint(processor,str(transformers.__version__))==teacher['processor'],'Processor differs from teacher')
    ids=[processor.tokenizer.encode(str(k),add_special_tokens=False) for k in range(9)]
    need(all(len(x)==1 for x in ids) and len({x[0] for x in ids})==9,'Counts0..8 must have distinct single tokens')
    need(not set(x[0] for x in ids)&set(processor.tokenizer.all_special_ids),'Count token is special')
    prompts,rows=rows_for(questions)
    for p in prompts:
        inputs=text_inputs(processor,p['question'],p['n_frames'])
        need(inputs['input_ids'].shape[0]==1 and bool((inputs['attention_mask']==1).all()),'Unexpected serial padding')
        p.update(input_ids=inputs['input_ids'][0].tolist(),prompt_tokens=inputs['input_ids'].shape[1])
        p['input_ids_sha256']=object_sha(p['input_ids'])
        rendered=processor.tokenizer.decode(p['input_ids'],skip_special_tokens=False,clean_up_tokenization_spaces=False)
        for k in range(9):
            need(processor.tokenizer.encode(rendered+str(k),add_special_tokens=False)==p['input_ids']+ids[k],
                 'Count is not the exact single-token continuation at the native prompt boundary')
    out=OUT/f"check_{os.environ['SLURM_JOB_ID']}";out.mkdir(parents=True,exist_ok=False);snapshot(out)
    plan=dict(schema_version=1,protocol='synthetic_correct_count_native_first_token_capacity',policy=POLICY,
        questions=questions,prompts=prompts,rows=rows,count_token_ids=[x[0] for x in ids],
        teacher_plan_file=str(teacher_path),teacher_plan_sha256=sha(teacher_path),
        model=model_metadata(),runtime=teacher['runtime'],processor=teacher['processor'],source_sha256=sources(),
        cpu_self_test=self_test,cpu_seconds=time.perf_counter()-started)
    save(out/'plan.json',plan);(out/'plan.sha256').write_text(sha(out/'plan.json')+'\n')
    (out/'INDEX.md').write_text('# Frozen synthetic readout capacity plan\n\n[Plan](plan.json) · [Source hashes](source_hashes.json).\n')
    print(json.dumps(dict(output=str(out),plan_sha256=sha(out/'plan.json'),cpu_seconds=plan['cpu_seconds'])),flush=True)


def verify(path):
    need(sha(path)==path.with_suffix('.sha256').read_text().strip(),'Plan sidecar differs')
    plan=read(path)
    need(plan['schema_version']==1 and plan['protocol']=='synthetic_correct_count_native_first_token_capacity' and plan['policy']==POLICY and plan['source_sha256']==sources(),'Frozen policy/source changed')
    need(plan['model']==model_metadata() and sha(plan['teacher_plan_file'])==plan['teacher_plan_sha256'],'Model/teacher binding changed')
    teacher=verify_plan(Path(plan['teacher_plan_file']))
    need(plan['questions']==sorted({x['question'] for x in teacher['pairs'].values()})[:8],'Selected training questions changed')
    prompts,rows=rows_for(plan['questions']);need(rows==plan['rows'],'Split rows changed')
    for actual,expected in zip(plan['prompts'],prompts):
        need(all(actual[k]==v for k,v in expected.items()) and actual['input_ids_sha256']==object_sha(actual['input_ids']) and actual['prompt_tokens']==len(actual['input_ids']),
             'Frozen question/length/token binding changed')
    need(len(prompts)==len(plan['prompts'])==32,'Prompt count differs')
    return plan


def compare_logits(a,b):
    import torch
    a=a.double();b=b.double();pa=a.softmax(-1);pb=b.softmax(-1)
    tv=(pa-pb).abs().sum(-1)/2;match=a.argmax(-1)==b.argmax(-1)
    return dict(maximum_tv=float(tv.max()),top1_mismatches=int((~match).sum()),rows=int(a.shape[0]),
                passed=bool((tv<=.02).all() and match.all()),per_row_tv=tv.cpu().tolist())


def run(args):
    import torch,transformers
    import torch.nn.functional as F
    from gnnformer.runtime import load_runtime,move_to_device
    from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
    from scripts.probe_native_vision_v2_prefix import fingerprint
    plan=verify(args.plan);need(torch.cuda.is_available(),'Slurm GPU required')
    torch.set_num_threads(4);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    need(str(torch.__version__)==plan['runtime']['torch_version'] and str(transformers.__version__)==plan['runtime']['transformers_version'],
         'Native runtime version changed')
    started=time.perf_counter();job=os.environ['SLURM_JOB_ID']
    out=OUT/f'run_{job}';out.mkdir(parents=True,exist_ok=False);snapshot(out)
    data=DATA/f'run_{job}';data.mkdir(parents=True,exist_ok=False)
    ckpts=CKPTS/f'run_{job}';ckpts.mkdir(parents=True,exist_ok=False)
    (out/'plan.json').write_bytes(args.plan.read_bytes())
    (out/'INDEX.md').write_text('# Synthetic oracle first-token capacity\n\n[Summary](summary.json) · [Predictions](predictions.json) · [Report](REPORT.md).\n')
    (data/'INDEX.md').write_text('# Frozen global states for synthetic capacity\n\n[States](global_states.pt) · [Tensor identity](identity.json).\n')
    (ckpts/'INDEX.md').write_text('# Final oracle readout checkpoints\n\n[Affine](identity.pt) · [SiLU](silu.pt). These bypass evidence extraction.\n')
    load_start=time.perf_counter();rt=load_runtime(str(MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda')
    model=rt.model;model.eval();model.requires_grad_(False)
    need(fingerprint(rt.processor,str(transformers.__version__))==plan['processor'],'Actual processor changed')
    norm=model.model.language_model.norm;head=model.lm_head
    need(getattr(head,'bias',None) is None,'Expected native bias-free vocabulary head')
    torch.cuda.synchronize();load_seconds=time.perf_counter()-load_start
    counter=dict(model=0,vision=0,language=0);captured={}
    def count(name):
        def hook(*_):counter[name]+=1
        return hook
    def capture(module,inputs):captured['h']=inputs[0][:,-1,:].detach().clone()
    hooks=[model.register_forward_pre_hook(count('model')),model.model.visual.register_forward_pre_hook(count('vision')),
           model.model.language_model.register_forward_pre_hook(count('language')),norm.register_forward_pre_hook(capture)]
    states=[];reference=[]
    try:
        for p in plan['prompts']:
            inputs=text_inputs(rt.processor,p['question'],p['n_frames'])
            need(inputs['input_ids'][0].tolist()==p['input_ids'],'Native prompt tokens changed')
            captured.clear()
            with torch.no_grad():result=model(**move_to_device(inputs,rt.device),use_cache=False,logits_to_keep=1)
            need(result.past_key_values is None and result.logits.shape[:2]==(1,1),'Unexpected oracle model output')
            states.append(captured['h'].detach().clone());reference.append(result.logits[:,0].detach().clone())
    finally:
        for hook in hooks:hook.remove()
    need(counter==dict(model=32,vision=0,language=32),'Expected32serialtext-only native forwards')
    h=torch.cat(states);refs=torch.cat(reference);need(h.shape==(32,3584),'Wrong native hidden states')
    need(bool(torch.isfinite(h).all() and torch.isfinite(refs).all()),'Nonfinite frozen states/logits')
    need(not torch.is_inference(h) and not h.requires_grad,'Frozen states must support normal readout autograd')
    torch.save(dict(schema_version=1,hidden=h.cpu(),native_logits=refs.cpu(),prompts=plan['prompts']),data/'global_states.pt')
    identity=dict(states_file=str(data/'global_states.pt'),states_sha256=sha(data/'global_states.pt'),
        hidden_sha256=tensor_hash(h),hidden_dtype=str(h.dtype),head_weight_dtype=str(head.weight.dtype),norm_weight_dtype=str(norm.weight.dtype),
        native_norm_weight_sha256=tensor_hash(norm.weight),native_head_weight_sha256=tensor_hash(head.weight),
        model=model_metadata(),plan_file=str(args.plan.resolve()),plan_sha256=sha(args.plan),source_sha256=plan['source_sha256'])
    save(data/'identity.json',identity)
    with torch.no_grad():replay=head(norm(h))
    replay_gate=compare_logits(replay,refs);save(out/'replay_gate.json',replay_gate)
    need(replay_gate['passed'],'Frozen native head replay failed before oracle training')
    prompt_idx=torch.tensor([x['prompt_index'] for x in plan['rows']],device=rt.device)
    global_rows=h[prompt_idx].detach().clone();z=torch.zeros(len(plan['rows']),96,device=rt.device)
    z[:,0]=torch.tensor([x['k'] for x in plan['rows']],device=rt.device)
    targets=torch.tensor([plan['count_token_ids'][x['k']] for x in plan['rows']],device=rt.device)
    indices={cell:torch.tensor([i for i,x in enumerate(plan['rows']) if x['cell']==cell],device=rt.device)
             for cell in ('fit','new_questions','new_lengths','both')}
    fit=indices['fit'];reports=[];all_predictions=[];initial_hash=None
    for activation in POLICY['activations']:
        torch.manual_seed(SEED);branch=ParallelLocalAggregation(3584,rank=96,post_activation=activation).to(device=rt.device)
        branch.local.requires_grad_(False);branch.local_bias.requires_grad_(False)
        init_hash=state_hash(branch.state_dict())
        if initial_hash is None:initial_hash=init_hash
        need(init_hash==initial_hash,'Matched readout initial parameters differ')
        params=[p for p in branch.parameters() if p.requires_grad]
        need(sum(p.numel() for p in branch.parameters())==1041600 and sum(p.numel() for p in params)==697440,'Parameter arithmetic differs')
        optimizer=torch.optim.AdamW(params,lr=.001,weight_decay=0.)
        torch.cuda.reset_peak_memory_stats();begin=time.perf_counter();history=[]
        with torch.no_grad():
            zero=head(norm(global_rows+branch.decode(z,global_rows)))
            need(torch.equal(zero,head(norm(global_rows))),'Zero-U cached model identity differs')
        for step in range(500):
            optimizer.zero_grad(set_to_none=True)
            fused=global_rows[fit]+branch.decode(z[fit],global_rows[fit])
            logits=head(norm(fused));loss=F.cross_entropy(logits.float(),targets[fit])
            need(bool(torch.isfinite(loss)),'Nonfinite loss')
            loss.backward();gn=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            need(branch.local.weight.grad is None and branch.local_bias.grad is None,'Oracle unexpectedly trained local encoder')
            optimizer.step();entry=dict(step=step+1,loss=float(loss),gradient_norm=float(gn),clipped=bool(gn>1))
            history.append(entry)
            if step+1 in (1,100,250,500):print(json.dumps(dict(activation=activation,**entry)),flush=True)
        torch.cuda.synchronize();train_seconds=time.perf_counter()-begin
        need(all(p.grad is None and not p.requires_grad for p in model.parameters()),'Frozen native model received gradients')
        final_state={k:v.detach().cpu() for k,v in branch.state_dict().items()}
        checkpoint=ckpts/f'{activation}.pt';torch.save(dict(state_dict=final_state,activation=activation,policy=POLICY,
                 source_sha256=plan['source_sha256'],plan_sha256=sha(args.plan),states_sha256=identity['states_sha256']),checkpoint)
        cells={}
        with torch.no_grad():
            for cell,ix in indices.items():
                delta=branch.decode(z[ix],global_rows[ix])
                logits=head(norm(global_rows[ix]+delta)).float()
                ratios=delta.float().norm(dim=-1)/global_rows[ix].float().norm(dim=-1).clamp_min(1e-12)
                pred=logits.argmax(-1);correct=pred==targets[ix];nll=F.cross_entropy(logits,targets[ix],reduction='none')
                gold=logits.gather(1,targets[ix,None]).squeeze(1);other=logits.clone()
                other.scatter_(1,targets[ix,None],float('-inf'));margins=gold-other.max(-1).values
                cells[cell]=dict(n=int(ix.numel()),correct=int(correct.sum()),accuracy=float(correct.float().mean()),nll=float(nll.mean()),
                    mean_gold_margin=float(margins.mean()),minimum_gold_margin=float(margins.min()),
                    mean_residual_to_global_norm=float(ratios.mean()),maximum_residual_to_global_norm=float(ratios.max()))
                for local_i,row_i in enumerate(ix.cpu().tolist()):
                    tid=int(pred[local_i]);all_predictions.append(dict(activation=activation,**plan['rows'][row_i],
                        predicted_token_id=tid,predicted_text=rt.tokenizer.decode([tid],skip_special_tokens=False),
                        target_token_id=int(targets[row_i]),correct=bool(correct[local_i]),nll=float(nll[local_i]),
                        gold_margin=float(margins[local_i]),residual_to_global_norm=float(ratios[local_i])))
        save(out/f'{activation}_training.json',history)
        reports.append(dict(activation=activation,cells=cells,fit_capacity_passed=cells['fit']['correct']==72,
            initialized_state_sha256=init_hash,final_state_sha256=state_hash(final_state),checkpoint=str(checkpoint),
            checkpoint_sha256=sha(checkpoint),trainable_parameters=697440,total_branch_parameters=1041600,
            training_seconds=train_seconds,clipped_updates=sum(x['clipped'] for x in history),
            peak_memory_allocated_bytes=torch.cuda.max_memory_allocated()))
        print(json.dumps(reports[-1]),flush=True)
        del optimizer,branch,logits,loss,fused,zero
    save(out/'predictions.json',all_predictions)
    need(verify(args.plan)==plan,'Frozen inputs/source changed during run')
    summary=dict(schema_version=1,completed=True,computational_integrity_passed=True,
        protocol=plan['protocol'],policy=POLICY,replay_gate=replay_gate,conditions=reports,
        native_forwards=counter,model_load_seconds=load_seconds,total_seconds=time.perf_counter()-started,
        states_identity=identity,predictions_file=str(out/'predictions.json'),predictions_sha256=sha(out/'predictions.json'),
        source_sha256=plan['source_sha256'],plan_file=str(args.plan.resolve()),plan_sha256=sha(args.plan),
        gpu=torch.cuda.get_device_name(0),slurm_job_id=job,
        limitations=['Correct K is supplied explicitly as z=K e1; this is not vision aggregation accuracy.',
          'Only first-token0..8 capacity is tested; no EOS, whole answer, multi-digit, cache update or reasoning.',
          'Held oracle cells are descriptive and did not select checkpoints or hyperparameters.',
          'Failure of this fixed optimizer is not an architecture impossibility proof.'])
    save(out/'summary.json',summary)
    lines=['# Synthetic correct-count latent: native first-token capacity','',
       'The correct K is supplied to the readout. These are oracle scores, not MMReD vision accuracy.','',
       '| Readout | Fit /72 | New questions /72 | New lengths /72 | Both /72 | Fit capacity |',
       '|---|---:|---:|---:|---:|---|']
    for r in reports:
        values=[str(r['cells'][cell]['correct']) for cell in ('fit','new_questions','new_lengths','both')]
        lines.append('| '+r['activation']+' | '+' | '.join(values)+' | '+('PASS' if r['fit_capacity_passed'] else 'FAIL')+' |')
    lines+=['',*summary['limitations'],''];(out/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(output=str(out),seconds=summary['total_seconds'])),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--check',action='store_true');g.add_argument('--run',action='store_true');p.add_argument('--plan',type=Path)
    args=p.parse_args();need(bool(os.environ.get('SLURM_JOB_ID')),'All numerical work must use Slurm')
    if args.check:
        need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS',''),'CPU wrapper required');check()
    else:
        need(os.environ.get('SLURM_JOB_PARTITION')=='gpu' and args.plan is not None,'GPU wrapper and frozen plan required');run(args)

if __name__=='__main__':main()

"""One prospective intact-evidence residual software/cost profile, Slurm only."""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
PROTOCOL='native_visual_augmentation_profile'
OUT=REPO/'outputs/native_aggregation_vlm/native_visual_augmentation'
DATA=Path('/mnt/data/gabriele/gnn_transformer/native_visual_augmentation')
CKPT=Path('/mnt/ckpts/gabriele/gnn_transformer/native_visual_augmentation')
EVALUATION_PLAN=REPO/'outputs/native_aggregation_vlm/mmred_official_native_evaluation/check_444051/plan.json'
EVALUATION_PLAN_SHA='b044c69a3a558a5369b093ee23b9cdbbde24923f019eecad19f14f7558b31cb2'
INDICES=(0,5,800,810,1600,1612,2400,2405,2410,3203,3210,3222)
REQUIRED_SOURCES=('scripts/profile_native_visual_augmentation.py','scripts/native_visual_augmentation_runtime.py',
    'gnnformer/native_visual_augmentation.py','tests/test_native_visual_augmentation.py',
    'slurm/native_visual_augmentation_core_check.sbatch','docs/paper/NATIVE_AGGREGATION_AUGMENTATION_HYPOTHESIS.md',
    'docs/paper/NATIVE_VISUAL_AUGMENTATION_SOFTWARE_PROFILE.md','slurm/native_visual_augmentation_profile.sbatch',
    'gnnformer/native_visual_memory.py','gnnformer/attention_moments.py')
POLICY=dict(protocol=PROTOCOL,gpu_seconds=900,cpu_cores=4,memory_gib=16,arms=['summary','pointwise'],seed=25,
    training_witnesses=list(INDICES),diagnostic_alpha=.05,software_updates_per_arm=2,accumulation=12,
    lr=2e-4,betas=[.9,.999],eps=1e-8,weight_decay=.01,clip_norm=1.,maximum_new_tokens=50,
    natural_trajectories=10,tv_limit=.02,mean_ce_absolute_tolerance=.02,gradient_l2_atol=1e-6,
    gradient_l2_rtol=.05,gradient_cosine_min=.999,gradient_cosine_norm_floor=1e-5,
    live_parameters=2300929,no_fit_release=True,no_validation_or_test=True)


def need(value,message):
    if not value:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,sort_keys=True,allow_nan=False,indent=2)


def binding(path):return dict(file=str(Path(path).resolve()),sha256=sha(path))


def cpu_tree(torch,value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(torch,v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return type(value)(cpu_tree(torch,v) for v in value)
    return value


def tensor_save(torch,path,value):
    need(not path.exists(),'Immutable profile tensor already exists')
    torch.save(cpu_tree(torch,value),path);return dict(**binding(path),bytes=path.stat().st_size)


def gradients(torch,core):
    result={}
    for name,p in core.named_parameters():
        if p.requires_grad:
            need(p.grad is not None and bool(torch.isfinite(p.grad).all()),'Missing/nonfinite gradient: '+name)
            result[name]=p.grad.detach().clone()
    return result


def zero(core):
    for p in core.parameters():p.grad=None


def compare(torch,reference,candidate,reference_grads=None,candidate_grads=None):
    a=reference['logits'].detach().float();b=candidate['logits'].detach().float()
    need(a.shape==b.shape and a.ndim==3 and a.shape[0]==1 and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()),'Finite identical selected native head geometry required')
    tv=float((a.softmax(-1)-b.softmax(-1)).abs().sum(-1).mul(.5).max())
    ce=abs(float(reference['loss'].detach())-float(candidate['loss'].detach()))
    per={}
    if reference_grads is not None:
        need(reference_grads.keys()==candidate_grads.keys(),'Gradient parameter inventories differ')
        for name,g in reference_grads.items():
            x=g.double().flatten();y=candidate_grads[name].double().flatten()
            nx=float(x.norm());ny=float(y.norm());error=float((x-y).norm())
            cosine=None if min(nx,ny)==0 else float(torch.dot(x,y)/(nx*ny))
            passed=error<=1e-6+.05*nx and (nx<=1e-5 or (cosine is not None and cosine>=.999))
            per[name]=dict(reference_norm=nx,candidate_norm=ny,l2_error=error,cosine=cosine,passed=passed)
    return dict(passed=tv<=.02 and ce<=.02 and all(v['passed'] for v in per.values()),max_tv=tv,
        mean_ce_absolute_difference=ce,exact_logits=bool(torch.equal(a,b)),argmax_differences=int((a.argmax(-1)!=b.argmax(-1)).sum()),gradient_comparisons=per)


def no_grad_call(torch,function):
    with torch.no_grad():return function()


def public_result(result):
    return {k:v for k,v in result.items() if k!='past_key_values'}


def timed(torch,function):
    torch.cuda.synchronize();start=time.perf_counter();result=function();torch.cuda.synchronize()
    return result,time.perf_counter()-start


def run(args,out,data,ckpt,started,progress):
    import torch
    from gnnformer.runtime import load_runtime
    from gnnformer.native_visual_augmentation import SummaryReadResidual,PointwiseResidual
    from scripts import evaluate_mmred_official_native_memory as old
    from scripts import native_visual_augmentation_runtime as runtime
    training=old.training;profile=old.profile;p=old.p
    torch.set_num_threads(4)
    need(torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA B200','Exactly one B200 required')
    release=read(args.release);release_ref=binding(args.release);core_ref=binding(args.core_check)
    need(release['protocol']==PROTOCOL and release['policy']==POLICY
        and set(REQUIRED_SOURCES)<=set(release['source_sha256']),'Profile release/policy/source inventory mismatch')
    for name,digest in {**old.sources(),**old.inherited_sources()}.items():
        need(release['source_sha256'].get(name)==digest,'Reused frozen native dependency missing from release: '+name)
    for name,digest in release['source_sha256'].items():need(sha(REPO/name)==digest,'Frozen profile source changed: '+name)
    (out/'source').mkdir()
    for name,digest in release['source_sha256'].items():
        target=out/'source'/name.replace('/','_');target.write_bytes((REPO/name).read_bytes());need(sha(target)==digest,'Archived source differs')
    core_check=read(args.core_check)
    need(core_check['passed'] is core_check['completed'] is True and core_check['protocol']=='native_visual_augmentation_core_check'
        and core_check['tests_run']==8 and core_check['live_parameters_per_arm']==2300929
        and not (Path(args.core_check).parent/'failure.json').exists(),'Passed exact core CPU fixtures required')
    for name,digest in {**core_check['source_sha256'],**core_check['inherited_source_sha256']}.items():
        need(sha(REPO/name)==digest and sha(Path(args.core_check).parent/'source'/name.replace('/','_'))==digest
            and release['source_sha256'].get(name)==digest,'Core CPU fixture/source release binding differs')
    for field in ('test_results','report'):need(sha(core_check[field+'_file'])==core_check[field+'_sha256'],'Core test/report bytes changed')
    need(read(core_check['test_results_file'])==dict(tests_run=8,failures=[],errors=[],skipped=[]),'Complete exact core fixture outcomes required')
    need(sha(EVALUATION_PLAN)==EVALUATION_PLAN_SHA,'Frozen ordinary checkpoint release changed')
    evaluation_plan=old.verify_plan(EVALUATION_PLAN);proof=evaluation_plan['main_arms']['ordinary']
    main_plan=training.verify_plan(proof['main_plan']['file']);items=read(main_plan['cases_file'])
    need(len(items)==4000 and all(items[i]['row']['pilot_role']=='train' for i in INDICES),'Only fixed original training witnesses allowed')
    loaded=load_runtime(str(profile.preparation.MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda')
    model=loaded.model.eval().requires_grad_(False);hardware=p.frozen_joint.live_identity(torch,loaded,evaluation_plan)
    need(p.installed({})==evaluation_plan['packages'],'Native package versions changed')
    refs=list(model.named_parameters());base=p.base_metadata(refs)
    p.install_lora(torch,model,out/'actual_peft_config.json');params=p.adapter_parameters(model)
    final=proof['final_checkpoint'];need(sha(final['file'])==final['sha256'],'Competent ordinary checkpoint changed')
    state=torch.load(final['file'],map_location='cpu',weights_only=True)
    need(old.state_proof(torch,state,'ordinary',proof['peft_config'])==proof['state_proof'],'Exact final ordinary tensor proof changed')
    profile.restore(torch,params,state['adapter']);profile.model_mode(model,params,False);del state
    metadata=runtime.model_metadata(model,final);contract=old.frozen_adapter_contract(torch,model)
    save(out/'config.json',dict(protocol=PROTOCOL,policy=POLICY,source_release=binding(args.release),core_check=binding(args.core_check),
        evaluation_plan=binding(EVALUATION_PLAN),main_plan=proof['main_plan'],ordinary_checkpoint=final,peft_config=proof['peft_config'],
        hardware=hardware,native_identity_sha256=evaluation_plan['native_identity_sha256'],data_directory=str(data),checkpoint_directory=str(ckpt),
        witnesses=[dict(index=i,sid=items[i]['row']['sid'],n=items[i]['row']['n'],compact=items[i]['compact'],features=items[i]['features']) for i in INDICES]))
    def guard():
        need(time.perf_counter()-started<870,'Profile reserve boundary reached')
        runtime.check_model(model,metadata);p.check_base(refs,base,model)
        need(old.frozen_adapter_contract(torch,model)==contract,'Frozen adapted ordinary contract changed')
    def memory_input(case,features):return dict(features=features,**{k:v.to(model.device) for k,v in case['coordinates'].items()})
    counts=Counter();layers=[0]*28;cache={};baselines={};results={};natural=[];baseline_natural={}
    progress.update(completed_cache_cases=0,completed_arm_cases=0,counters=counts)
    with ExitStack() as hooks:
        def count(key):
            def callback(*_):counts[key]+=1
            return callback
        for module,key in ((model,'model'),(model.model,'backbone'),(model.model.visual,'vision'),(model.model.language_model,'language'),(model.model.language_model.norm,'norm'),(model.lm_head,'head')):
            hooks.callback(module.register_forward_pre_hook(count(key)).remove)
        for index,layer in enumerate(model.model.language_model.layers):
            def callback(*_,index=index):layers[index]+=1
            hooks.callback(layer.register_forward_pre_hook(callback).remove)
        torch.cuda.synchronize();setup=time.perf_counter()-started
        for index in INDICES:
            case,features=training.load_inputs(torch,model,items[index]);evidence={}
            baseline,base_seconds=timed(torch,lambda:no_grad_call(torch,lambda:runtime.teacher_full(torch,model,case,features,None,metadata=metadata,evidence=evidence)))
            baselines[index]=cpu_tree(torch,dict(logits=baseline['logits'],loss=baseline['loss']))
            baseline_ref=tensor_save(torch,data/f'ordinary_{index}.pt',dict(evidence=evidence,logits=baseline['logits'],loss=baseline['loss']))
            captured={};packet,capture_seconds=timed(torch,lambda:runtime.capture_teacher(torch,model,case,features,metadata=metadata,evidence=captured,keep_full=True))
            need(torch.equal(captured['profile_head'][0]['head_logits'],baselines[index]['logits']),'Captured native teacher changed ordinary head output')
            write_tick=time.perf_counter();packet_ref=tensor_save(torch,data/f'cache_{index}.pt',packet);write_seconds=time.perf_counter()-write_tick
            capture_ref=tensor_save(torch,data/f'capture_evidence_{index}.pt',dict(evidence=captured,logits=captured['profile_head'][0]['head_logits'],loss=captured['loss']))
            cache[index]=dict(packet=packet_ref,baseline=baseline_ref,capture_evidence=capture_ref,ordinary_teacher_seconds=base_seconds,capture_seconds=capture_seconds,serialization_seconds=write_seconds,n=case['metadata']['n'])
            progress['completed_cache_cases']+=1;save(out/f'cache_{index}.json',cache[index]);guard();del baseline,packet,case,features,evidence,captured
        # Independent residual leaves exercise causality in the actual frozen final block.
        index=INDICES[-1];case,features=training.load_inputs(torch,model,items[index])
        packet=torch.load(cache[index]['packet']['file'],map_location='cpu',weights_only=True);mi=memory_input(case,features)
        delta=torch.zeros_like(packet['suffix_hidden'],dtype=torch.float32,device=model.device,requires_grad=True);ev={}
        causal=runtime.replay_teacher(torch,model,packet,mi,None,metadata=metadata,evidence=ev,delta_override=delta)
        need(len(packet['target_ids'])>1,'Causal witness needs multiple teacher positions')
        # A fixed vocabulary-0 software probe avoids an accidentally saturated EOS loss.
        later_probe=torch.nn.functional.cross_entropy(causal['logits'][:,-1,:].float(),torch.tensor([0],device=model.device))
        first_probe=torch.nn.functional.cross_entropy(causal['logits'][:,0,:].float(),torch.tensor([0],device=model.device))
        earlier=torch.autograd.grad(later_probe,delta,retain_graph=True)[0]
        first=torch.autograd.grad(first_probe,delta)[0];counts['temporal_gradient_calls']+=2
        causal_result=dict(passed=bool(torch.isfinite(earlier).all() and torch.isfinite(first).all())
            and float(earlier[:,0,:].float().norm())>0 and int(torch.count_nonzero(first[:,1:,:]))==0,
            earlier_write_to_last_loss_norm=float(earlier[:,0,:].float().norm()),
            future_write_to_first_loss_nonzero=int(torch.count_nonzero(first[:,1:,:])),index=index,software_probe_target_id=0,original_gold_not_used=True)
        causal_result['evidence']=tensor_save(torch,data/'causal_evidence.pt',dict(evidence=ev,logits=causal['logits'],loss=causal['loss']))
        causal_result['gradients']=tensor_save(torch,ckpt/'causal_gradients.pt',dict(later_loss_gradient=earlier,first_loss_gradient=first))
        save(out/'causal.json',causal_result);need(causal_result['passed'],'Actual native temporal/future gradient check failed')
        del case,features,packet,mi,delta,ev,causal,earlier,first
        for index in (INDICES[0],INDICES[-1]):
            tick=time.perf_counter();case,features=training.load_inputs(torch,model,items[index]);load_seconds=time.perf_counter()-tick;evidence={}
            value,seconds=timed(torch,lambda:runtime.generate_cold(torch,model,loaded.processor,case,features,None,metadata=metadata,max_tokens=50,evidence=evidence))
            raw_tick=time.perf_counter();raw=tensor_save(torch,data/f'ordinary_natural_{index}.pt',dict(result=public_result(value),evidence=evidence));publication_seconds=time.perf_counter()-raw_tick
            row=dict(arm='ordinary',phase='anchor',index=index,n=case['metadata']['n'],generated_ids=value['generated_ids'],seconds=seconds,
                input_load_seconds=load_seconds,publication_seconds=publication_seconds,raw=raw,software_only=True)
            natural.append(row);baseline_natural[index]=row;save(out/f'ordinary_natural_{index}.json',row);guard();del case,features,evidence,value
        for arm,cls in (('summary',SummaryReadResidual),('pointwise',PointwiseResidual)):
            core=cls(seed=25).to(model.device);live={k:v for k,v in core.named_parameters() if v.requires_grad}
            need(sum(v.numel() for v in live.values())==2300929,'Exact live parameter match required')
            initial=cpu_tree(torch,core.state_dict());write_tick=time.perf_counter();initial_ref=tensor_save(torch,ckpt/f'{arm}_initial.pt',initial);initial_ref['write_seconds']=time.perf_counter()-write_tick
            arm_results=dict(parity=[],updates=[],training_microbatches=[],natural=[],initial=initial_ref)
            results[arm]=arm_results
            for index in INDICES:
                case,features=training.load_inputs(torch,model,items[index]);mi=memory_input(case,features)
                packet=torch.load(cache[index]['packet']['file'],map_location='cpu',weights_only=True)
                evidence={};actual,seconds=timed(torch,lambda:no_grad_call(torch,lambda:runtime.teacher_full(torch,model,case,features,core,metadata=metadata,evidence=evidence)))
                need(torch.equal(actual['logits'].detach().cpu(),baselines[index]['logits']),'Zero-gate full-native logits must be exactly ordinary')
                zero_ref=tensor_save(torch,data/f'{arm}_{index}_zero.pt',dict(logits=actual['logits'],loss=actual['loss'],evidence=evidence))
                del actual,evidence
                with torch.no_grad():core.alpha.fill_(.05)
                evaluated={};grad={};saved={};times={}
                for mode in ('full','prefix','reference'):
                    zero(core);evidence={}
                    if mode=='full':call=lambda:runtime.teacher_full(torch,model,case,features,core,metadata=metadata,evidence=evidence)
                    else:call=lambda:runtime.replay_teacher(torch,model,packet,mi,core,metadata=metadata,evidence=evidence,reference_full=mode=='reference')
                    value,seconds=timed(torch,call);value['loss'].backward();counts['backward']+=1
                    grad[mode]=gradients(torch,core);evaluated[mode]=dict(logits=value['logits'].detach(),loss=value['loss'].detach())
                    saved[mode]=tensor_save(torch,data/f'{arm}_{index}_{mode}.pt',dict(logits=value['logits'],loss=value['loss'],evidence=evidence))
                    times[mode]=seconds
                    if index in (INDICES[0],INDICES[-1]):tensor_save(torch,ckpt/f'{arm}_{index}_{mode}_grads.pt',grad[mode])
                    del value,evidence
                comparisons={mode:compare(torch,evaluated['full'],evaluated[mode],grad['full'],grad[mode]) for mode in ('prefix','reference')}
                row=dict(index=index,n=case['metadata']['n'],zero_exact=True,zero=zero_ref,comparisons=comparisons,artifacts=saved,forward_seconds=times)
                arm_results['parity'].append(row);save(out/f'{arm}_parity_{index}.json',row)
                need(all(v['passed'] for v in comparisons.values()),'Active-gate full/prefix/reference forward or gradient comparison failed')
                with torch.no_grad():core.alpha.zero_()
                zero(core);progress['completed_arm_cases']+=1;guard();del case,features,packet,mi,evaluated,grad
            core.eval().requires_grad_(False)
            for index in (INDICES[0],INDICES[-1]):
                tick=time.perf_counter();case,features=training.load_inputs(torch,model,items[index]);load_seconds=time.perf_counter()-tick;evidence={}
                value,seconds=timed(torch,lambda:runtime.generate_cold(torch,model,loaded.processor,case,features,core,metadata=metadata,max_tokens=50,evidence=evidence))
                need(value['generated_ids']==baseline_natural[index]['generated_ids'],'Zero-gate complete native trajectory changed')
                raw_tick=time.perf_counter();raw=tensor_save(torch,data/f'{arm}_zero_natural_{index}.pt',dict(result=public_result(value),evidence=evidence));publication_seconds=time.perf_counter()-raw_tick
                row=dict(arm=arm,phase='zero',index=index,n=case['metadata']['n'],generated_ids=value['generated_ids'],seconds=seconds,
                    input_load_seconds=load_seconds,publication_seconds=publication_seconds,raw=raw,zero_ids_exact=True,software_only=True)
                natural.append(row);save(out/f'{arm}_zero_natural_{index}.json',row);guard();del case,features,evidence,value
            for parameter in live.values():parameter.requires_grad_(True)
            core.train()
            core.load_state_dict(initial);zero(core)
            need(all(torch.equal(v.detach().cpu(),initial[k]) for k,v in core.state_dict().items()),'Diagnostic alpha leaked into fresh software state')
            optimizer=torch.optim.AdamW(list(live.values()),lr=2e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01)
            for step in (1,2):
                optimizer.zero_grad(set_to_none=True)
                for index in INDICES:
                    tick=time.perf_counter();case,features=training.load_inputs(torch,model,items[index])
                    packet=torch.load(cache[index]['packet']['file'],map_location='cpu',weights_only=True);mi=memory_input(case,features);evidence={}
                    value=runtime.replay_teacher(torch,model,packet,mi,core,metadata=metadata,evidence=evidence)
                    (value['loss']/12).backward();counts['backward']+=1;torch.cuda.synchronize()
                    raw=tensor_save(torch,data/f'{arm}_training_{step}_{index}.pt',dict(logits=value['logits'],loss=value['loss'],evidence=evidence))
                    arm_results['training_microbatches'].append(dict(step=step,index=index,n=case['metadata']['n'],seconds=time.perf_counter()-tick,loss=float(value['loss'].detach()),raw=raw))
                    del value,evidence,case,features,mi,packet;guard()
                g=gradients(torch,core);norms={k:float(v.float().norm()) for k,v in g.items()}
                need(norms['alpha']>0 and all(v==0 for k,v in norms.items() if k!='alpha') if step==1 else all(v>0 for v in norms.values()),'Expected gate-only then live inner-gradient paths differ')
                tensor_save(torch,ckpt/f'{arm}_update_{step}_grads.pt',g)
                norm=torch.nn.utils.clip_grad_norm_(list(live.values()),1.);need(bool(torch.isfinite(norm)),'Finite preclip norm required')
                _,optimizer_seconds=timed(torch,optimizer.step);counts['optimizer']+=1
                optimizer.zero_grad(set_to_none=True)
                write_tick=time.perf_counter();state_ref=tensor_save(torch,ckpt/f'{arm}_state_{step}.pt',dict(core=core.state_dict(),optimizer=optimizer.state_dict(),updates=step,ordinary_checkpoint=final,software_only=True));state_ref['write_seconds']=time.perf_counter()-write_tick
                arm_results['updates'].append(dict(step=step,gradient_norms=norms,preclip_norm=float(norm),seconds=optimizer_seconds,state=state_ref))
                save(out/f'{arm}_update_{step}.json',arm_results['updates'][-1]);guard()
            core.eval().requires_grad_(False)
            saved_state=torch.load(arm_results['updates'][-1]['state']['file'],map_location='cpu',weights_only=True)
            restored=cls(seed=25).to(model.device);restored.load_state_dict(saved_state['core']);restored.eval().requires_grad_(False)
            need(all(torch.equal(v.detach().cpu(),saved_state['core'][k]) for k,v in core.state_dict().items()),'Final module state bytes changed')
            arm_results['roundtrip']=[]
            for index in (INDICES[0],INDICES[-1]):
                case,features=training.load_inputs(torch,model,items[index]);mi=memory_input(case,features)
                packet=torch.load(cache[index]['packet']['file'],map_location='cpu',weights_only=True);ev1={};ev2={}
                with torch.no_grad():
                    before=runtime.replay_teacher(torch,model,packet,mi,core,metadata=metadata,evidence=ev1)
                    after=runtime.replay_teacher(torch,model,packet,mi,restored,metadata=metadata,evidence=ev2)
                need(torch.equal(before['logits'],after['logits']),'Software endpoint checkpoint round-trip logits changed')
                roundtrip=dict(index=index,exact=True,raw=tensor_save(torch,data/f'{arm}_roundtrip_{index}.pt',
                    dict(before=before['logits'],after=after['logits'],before_evidence=ev1,after_evidence=ev2)))
                arm_results['roundtrip'].append(roundtrip);del before,after,ev1,ev2,case,features,mi,packet
                tick=time.perf_counter();case,features=training.load_inputs(torch,model,items[index]);load_seconds=time.perf_counter()-tick;evidence={}
                value,seconds=timed(torch,lambda:runtime.generate_cold(torch,model,loaded.processor,case,features,core,metadata=metadata,max_tokens=50,evidence=evidence))
                ids=value['generated_ids'];need(1<=len(ids)<=50,'Bounded native trajectory required')
                raw_tick=time.perf_counter();raw=tensor_save(torch,data/f'{arm}_natural_{index}.pt',dict(result=public_result(value),evidence=evidence));publication_seconds=time.perf_counter()-raw_tick
                generated_logits=value['raw_logits'].to(model.device).unsqueeze(0);del value
                history=runtime.history_case(torch,case,ids);he={}
                with torch.no_grad():teacher=runtime.teacher_full(torch,model,history,features,core,metadata=metadata,evidence=he)
                generated_ce=torch.nn.functional.cross_entropy(generated_logits[0],torch.tensor(ids,device=model.device))
                diagnostic=compare(torch,teacher,dict(logits=generated_logits,loss=generated_ce))
                history_ref=tensor_save(torch,data/f'{arm}_history_{index}.pt',dict(logits=teacher['logits'],loss=teacher['loss'],evidence=he))
                row=dict(arm=arm,phase='software_final',index=index,n=case['metadata']['n'],generated_ids=ids,seconds=seconds,
                    input_load_seconds=load_seconds,publication_seconds=publication_seconds,raw=raw,full_history=history_ref,
                    cached_full_history_comparison=diagnostic,history_comparison_is_descriptive=True,software_only=True)
                natural.append(row);arm_results['natural'].append(row);save(out/f'{arm}_natural_{index}.json',row)
                guard();del case,features,evidence,teacher,he,history,generated_logits
            del core,restored,saved_state,optimizer,live,initial
    need(counts['vision']==0 and counts['optimizer']==4 and counts['backward']==120 and counts['temporal_gradient_calls']==2 and len(natural)==10,'Exact profile work inventory differs')
    model_calls=76+sum(len(row['generated_ids']) for row in natural)
    need(counts['model']==counts['backbone']==counts['language']==model_calls
        and counts['norm']==counts['head']==model_calls+105
        and layers==[model_calls]*27+[model_calls+105],'Actual whole-model/selected-block/head counters differ')
    need(profile.tensor_state(params)==proof['state_proof']['adapter'],'Ordinary fitted bytes changed during external-module training')
    guard();result=dict(protocol=PROTOCOL,passed=True,completed=True,policy=POLICY,setup_seconds=setup,cache=cache,arms=results,natural=natural,
        causal_check=causal_result,counters=dict(counts),decoder_layer_calls=layers,ordinary_checkpoint=final,ordinary_checkpoint_unchanged=True,original_replacement_configuration_closed=True,
        max_allocated_bytes=torch.cuda.max_memory_allocated(),max_reserved_bytes=torch.cuda.max_memory_reserved(),no_fit_release=True,requires_independent_cpu_audit=True)
    need(binding(args.release)==release_ref and binding(args.core_check)==core_ref,'Profile release/core-check bytes changed during execution')
    for name,digest in release['source_sha256'].items():need(sha(REPO/name)==digest,'Profile source changed before publication')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--release',required=True);parser.add_argument('--core-check',required=True);args=parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_GPUS'),'GPU Slurm allocation required')
    name='profile_'+os.environ['SLURM_JOB_ID'];out=OUT/name;data=DATA/name;ckpt=CKPT/name
    for p in (out,data,ckpt):p.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();progress={};save(out/'request.json',dict(args=vars(args),protocol=PROTOCOL,policy=POLICY))
    try:
        result=run(args,out,data,ckpt,started,progress);result['elapsed_seconds']=time.perf_counter()-started
        save(out/'analysis.json',result)
        artifacts={str(f):sha(f) for directory in (data,ckpt) for f in directory.iterdir() if f.is_file()}
        save(out/'artifacts.json',artifacts)
        need(time.perf_counter()-started<900,'Profile total allocation-time bound exceeded')
        save(out/'summary.json',dict(protocol=PROTOCOL,passed=True,completed=True,artifacts_file=str(out/'artifacts.json'),artifacts_sha256=sha(out/'artifacts.json'),analysis_file=str(out/'analysis.json'),
            analysis_sha256=sha(out/'analysis.json'),elapsed_seconds=time.perf_counter()-started,no_fit_release=True,requires_independent_cpu_audit=True))
    except BaseException as exc:
        save(out/'failure.json',dict(protocol=PROTOCOL,passed=False,type=type(exc).__name__,message=str(exc),progress=progress,elapsed_seconds=time.perf_counter()-started,no_automatic_retry=True));raise


if __name__=='__main__':main()

"""Fixed original-MMReD cold training pilot; all tensor work is Slurm-only."""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import random
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import profile_mmred_official_native_memory as profile
from scripts import report_mmred_official_native_memory as audit
from scripts import stage_mmred_official_training as compact
from scripts import native_visual_memory_runtime as runtime
from scripts import mmred_official_answer as answers
from scripts.stage_mmred_official_recovery import need,sha,object_sha,save
p=profile.p
OUT=REPO/'outputs/native_aggregation_vlm/mmred_official_native_training'
DATA=Path('/mnt/data/gabriele/gnn_transformer/mmred_official_native_training')
CKPT=Path('/mnt/ckpts/gabriele/gnn_transformer/mmred_official_native_training')
PROTOCOL='mmred_official_native_training'
PROPOSAL='docs/paper/NATIVE_AGGREGATION_MMRED_OFFICIAL_NATIVE_TRAINING_PROPOSAL.md'
PROPOSAL_SHA='5d6413b537c37b637eda45783909869dcacf5c2ee56a870fffbe2746d684273a'
OWN=('scripts/train_mmred_official_native_memory.py',PROPOSAL,'scripts/mmred_official_answer.py',
     'slurm/mmred_official_native_training_check.sbatch','slurm/mmred_official_native_training_run.sbatch')
AUDIT=profile.OUT/'report_443990/summary.json'
AUDIT_SHA='183087eb0bd6ec18a0df4bb19b4dda009b20163863d25db39658460552a3ea2b'
POLICY=dict(protocol=PROTOCOL,arms=list(profile.ARMS),seed=24,worlds=4000,epochs=3,backwards=12000,updates=1500,
    accumulation=8,lr=2e-4,betas=[.9,.999],eps=1e-8,weight_decay=.01,clip_norm=1.,diagnostic_worlds=100,
    maximum_new_tokens=50,roundtrip_teacher_calls=4,retained_training_teachers=2,
    maximum_model_head_calls=17004,vision_calls=0,check_cpu_seconds=600,main_gpu_seconds=7200,
    cpu_cores=4,memory_gib=16,maximum_arrays=1,maximum_attempts_per_arm=1,
    cold_full_prefix=True,no_validation_test_inference=True,no_checkpoint_selection=True,no_resume=True,
    original_cache_failure_preserved=True)


def read(path):return json.loads(Path(path).read_text())


def sources():
    need(sha(REPO/PROPOSAL)==PROPOSAL_SHA,'Training proposal changed')
    return {name:sha(REPO/name) for name in OWN}


def feature_module():
    from scripts import harvest_mmred_official_training_features as module
    return module


def boundary_module():
    from scripts import profile_mmred_official_training_boundary as module
    return module


def inherited_sources():
    mappings=[{**compact.sources(),**compact.inherited_sources()},audit.sources(),profile.sources()]
    mappings.extend([feature_module().sources(),boundary_module().sources()])
    result={}
    for mapping in mappings:
        for name,digest in mapping.items():
            need(name not in result or result[name]==digest,'Training source closure conflict');result[name]=digest
    for name in OWN:result.pop(name,None)
    need(all(sha(REPO/name)==digest for name,digest in result.items()),'Frozen training ancestor changed')
    return result


def bind(path,bindings,expected=None):
    path=Path(path).resolve();digest=sha(path);need(expected is None or digest==expected,'Training bound input changed: '+str(path))
    bindings[str(path)]=digest;return dict(file=str(path),sha256=digest)


def archive(out):
    own=sources();inherited=inherited_sources();(out/'source').mkdir()
    for name,digest in {**own,**inherited}.items():
        file=out/'source'/name.replace('/','_');file.write_bytes((REPO/name).read_bytes());need(sha(file)==digest,'Training source archive changed')
    return own,inherited


def training_order():
    rng=random.Random(24);order=[]
    for epoch in range(3):
        indices=list(range(4000));rng.shuffle(indices)
        order.extend(dict(micro_index=epoch*4000+j,epoch=epoch,index=index,updates_before=(epoch*4000+j)//8) for j,index in enumerate(indices))
    return order


def projections(boundary):
    result={}
    for arm in profile.ARMS:
        values=boundary['projection']['arms'][arm]
        T={int(k):v for k,v in values['microbatch_seconds_by_n'].items()}
        G={int(k):v for k,v in values['generation_fifty_token_seconds_by_n'].items()}
        need(set(T)==set(G)=={1,2,4,8,16} and all(math.isfinite(v) and v>0 for v in [*T.values(),*G.values()]),'Complete finite per-N timing population required')
        setup=values['setup_seconds'];O=values['original_optimizer_seconds'];C=values['original_checkpoint_seconds']
        training=sum(2400*T[n] for n in T);generation=sum(20*G[n] for n in G)
        # Four checkpoint-like writes include pre-final, final and restoration;
        # four reload teacher calls are explicitly priced at the larger N1/N16 time.
        reload_teachers=4*max(T[1],T[16]);estimate=setup+1.25*(training+1500*O+4*C+reload_teachers+generation)+60
        result[arm]=dict(passed=estimate<=7200,seconds=estimate,setup_seconds=setup,training_seconds=training,
            optimizer_seconds=1500*O,checkpoint_seconds=4*C,reload_teacher_seconds=reload_teachers,
            diagnostic_generation_seconds=generation,microbatch_seconds_by_n=T,generation_fifty_token_seconds_by_n=G,
            cap_seconds=7200,shared_feature_cost_separate=True,empirical_not_guarantee=True)
    return result


def check(args,out,started):
    bindings={};bind(AUDIT,bindings,AUDIT_SHA);a=audit.verify_report(AUDIT)
    need(a['cold_training_components_passed'] and not a['original_profile_passed'] and not a['cache_parity_passed'],'Audited cold scope with preserved cache failure required')
    feature_ref=bind(args.features,bindings);features=feature_module().verify_stage(args.features,streaming=True)
    boundary_ref=bind(args.boundary,bindings);boundary=boundary_module().verify_report(args.boundary)
    need(all(boundary['feature_stage'][k]==v for k,v in feature_ref.items()),'Boundary and training must consume the identical common feature stage')
    compact_ref=features['compact_stage'];bind(compact_ref['file'],bindings,compact_ref['sha256'])
    prepared=compact.verify_stage(compact_ref['file'],streaming=True)
    records=read(prepared['cases_file']);rows=read(prepared['rows_file']);diagnostic_sids=read(prepared['diagnostic_sids_file'])
    cached=read(features['feature_index_file'])
    need(len(records)==len(rows)==len(cached)==4000 and [r['index'] for r in records]==list(range(4000))
         and [r['sid'] for r in records]==[r['sid'] for r in rows]==[r['sid'] for r in cached], 'Exact4000 compact/feature/source order required')
    feature_by_index={r['index']:r for r in cached}
    source_profile=read(profile.OUT/'profile_443978/config.json');initial=source_profile['initial'];bind(initial['file'],bindings,initial['sha256'])
    need(a['profile']['config_sha256']==sha(profile.OUT/'profile_443978/config.json'),'Audited initialization owner changed')
    diagnostics=[next(r['index'] for r in records if r['sid']==sid) for sid in diagnostic_sids]
    roundtrip=[next(r['index'] for r in records if r['sid']==source_profile['cases'][i]['sid']) for i in (0,4)]
    projection=projections(boundary)
    for arm in profile.ARMS:
        peak=boundary['projection']['arms'][arm]['estimated_training_peak_allocated_bytes']
        total=source_profile['hardware']['total_memory_bytes']
        projection[arm].update(estimated_peak_allocated_bytes=peak,device_total_bytes=total,memory_eligible=peak<.9*total,optimizer_peak_not_newly_measured=True)
        projection[arm]['passed']=projection[arm]['passed'] and projection[arm]['memory_eligible']
    save(out/'projection.json',projection)
    need(all(v['passed'] for v in projection.values()),'Measured complete main scope exceeds prospective7200s cap')
    test=answers.self_test();save(out/'parser_fixtures.json',test);order=training_order();save(out/'order.json',order)
    need(len(diagnostics)==len(set(diagnostics))==100 and len(roundtrip)==2,'Fixed diagnostic/reload selection differs')
    rows_by_index={r['index']:rows[r['index']] for r in records}
    targets=read(prepared['target_inventory_file']);need(targets['maximum_target_tokens']==8 and targets['supervised_rows_per_epoch']==29555,'Original target/head envelope differs')
    for file in (prepared['cases_file'],prepared['rows_file'],prepared['diagnostic_sids_file'],features['feature_index_file']):bind(file,bindings)
    save(out/'cases.json',[dict(compact=r,features=feature_by_index[r['index']],row=rows_by_index[r['index']]) for r in records])
    plan=dict(protocol=PROTOCOL,policy=POLICY,source_sha256=sources(),inherited_source_sha256=inherited_sources(),
        feature_stage=feature_ref,boundary_report=boundary_ref,compact_stage=compact_ref,profile_audit=dict(file=str(AUDIT),sha256=AUDIT_SHA),
        native_identity=prepared['native_identity'],native_identity_sha256=prepared['native_identity_sha256'],
        initial=initial,peft_config=source_profile['peft_config'],profile_preparation=source_profile['preparation'],
        cases_file=str(out/'cases.json'),order_file=str(out/'order.json'),diagnostic_indices=diagnostics,
        roundtrip_indices=roundtrip,target_rows_per_epoch=29555,projection=projection,input_bindings=bindings,
        artifacts={str(f):sha(f) for f in out.glob('*.json')},no_validation_test_inference=True)
    save(out/'plan.json',plan)
    need(time.perf_counter()-started<600,'CPU main preparation cap exceeded')
    return dict(plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),projection=projection,no_validation_test_inference=True)


def verify_plan(path):
    path=Path(path).resolve();plan=read(path);summary=read(path.parent/'summary.json')
    need(summary['passed'] is summary['completed'] is True and summary['phase']=='check'
         and summary['plan_sha256']==sha(path) and not (path.parent/'failure.json').exists(),'Passed main CPU preparation required')
    need(plan['protocol']==PROTOCOL and plan['policy']==POLICY and plan['source_sha256']==sources()
         and plan['inherited_source_sha256']==inherited_sources(),'Main frozen policy/source changed')
    for mapping in (plan['input_bindings'],plan['artifacts']):
        for file,digest in mapping.items():need(sha(file)==digest,'Prepared main input changed')
    need(read(plan['order_file'])==training_order() and all(v['passed'] for v in plan['projection'].values()),'Fixed main order/resource release differs')
    return plan


def load_inputs(torch,model,item):
    descriptor=item['compact'];feature=item['features']
    need(sha(descriptor['file'])==descriptor['sha256'] and sha(feature['file'])==feature['sha256'],'Consumed main input file changed')
    saved=torch.load(descriptor['file'],map_location='cpu',weights_only=True);case=saved['case']
    cached=torch.load(feature['file'],map_location='cpu',weights_only=True)
    values=cached['features']
    ownership={k:cached[k] for k in ('index','sid','n','input_identity','compact','coordinate_identity','native_identity_sha256')}
    need(ownership=={k:feature[k] for k in ownership}
         and feature['compact']==dict(file=descriptor['file'],sha256=descriptor['sha256'],compact_identity_sha256=descriptor['compact_identity_sha256'])
         and feature['input_identity']==dict(pixel_values=descriptor['omitted_pixel_values'],image_grid_thw=descriptor['grid_identity'])
         and feature['coordinate_identity']=={k:p.tensor_info(v) for k,v in case['coordinates'].items()},'Native feature/compact ownership differs')
    need(compact.object_sha(compact.tensor_tree(torch,p,case))==descriptor['compact_identity_sha256']
         and saved['omitted_pixel_values']==descriptor['omitted_pixel_values']
         and p.tensor_info(values)==feature['tensor'] and values.dtype==torch.float16
         and values.shape==(196*descriptor['n'],3584) and bool(values.isfinite().all()),'Exact compact/native feature bytes differ')
    need(case['metadata']['sid']==descriptor['sid']==feature['sid']==item['row']['sid'],'Main SID ownership differs')
    return case,values.to(model.device)


def train(args,out,started,progress):
    import torch
    from gnnformer.runtime import load_runtime
    from gnnformer.native_visual_memory import NativeVisualMemory
    torch.set_num_threads(4);plan=verify_plan(args.plan);arm=profile.ARMS[args.arm]
    need(torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA B200','One B200 required')
    need(os.environ.get('SLURM_ARRAY_TASK_ID')==str(args.arm) and os.environ.get('SLURM_ARRAY_TASK_COUNT')=='3','One fixed three-arm array required')
    array=os.environ['SLURM_ARRAY_JOB_ID'];need(all(x.name.split('_')[1]==array for x in OUT.glob('run_*')),'Only one training array is released')
    data=DATA/out.name;ckpt=CKPT/out.name;data.mkdir(parents=True,exist_ok=False);ckpt.mkdir(parents=True,exist_ok=False)
    items=read(plan['cases_file']);order=read(plan['order_file']);input_bindings={};bind(args.plan,input_bindings)
    loaded=load_runtime(str(profile.preparation.MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda')
    model=loaded.model.eval().requires_grad_(False);parent=read(plan['profile_preparation']['plan_file'])
    hardware=p.frozen_joint.live_identity(torch,loaded,parent);need(p.installed({})==parent['packages'],'Native packages changed')
    refs=list(model.named_parameters());base=p.base_metadata(refs);save(out/'base_before.json',base)
    contract=p.install_lora(torch,model,out/'actual_peft_config.json');params=p.adapter_parameters(model)
    initial=plan['initial'];need(sha(initial['file'])==initial['sha256'],'Original unfitted initial bytes changed')
    original=torch.load(initial['file'],map_location='cpu',weights_only=True)
    need(original['updates']==0 and original['seed']==24 and original['peft_config']==p.canonical_peft_config(model)==plan['peft_config'],'Only actual unfitted native initialization allowed')
    profile.restore(torch,params,original['adapter']);memory=None if arm=='ordinary' else NativeVisualMemory().to(model.device)
    if memory is not None:profile.restore(torch,dict(memory.named_parameters()),original['memory'])
    torch.random.set_rng_state(original['rng']['cpu']);torch.cuda.set_rng_state_all(original['rng']['cuda'])
    profile.model_mode(model,params,True);live={**params,**({} if memory is None else {'memory.'+k:v for k,v in memory.named_parameters()})}
    optimizer=torch.optim.AdamW(list(live.values()),lr=2e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01);optimizer.zero_grad(set_to_none=True)
    counts=Counter(dict.fromkeys(('model','backbone','visual','language','norm','head','backward','optimizer'),0));layers=[0]*28
    config=dict(protocol=PROTOCOL,policy=POLICY,arm=arm,array_job_id=array,plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),
        source_sha256=sources(),inherited_source_sha256=inherited_sources(),hardware=hardware,initial=initial,
        peft_config=plan['peft_config'],contract=contract,data_directory=str(data),checkpoint_directory=str(ckpt))
    save(out/'config.json',config);states={'0':initial};retained=[];roundtrip={};natural=[];comparisons=[];trace=out/'training.jsonl';updates_file=out/'updates.jsonl'
    progress.update(counters=counts,states=states,retained=retained,natural=natural)
    def check_run():
        need(time.perf_counter()-started<7200,'Fixed main GPU cap exceeded');p.check_base(refs,base,model)
    def state(step,full=False):
        file=ckpt/f'state_{step}.pt';value=dict(arm=arm,updates=step,adapter=profile.cpu_state(torch,params),
            memory=None if memory is None else profile.cpu_state(torch,dict(memory.named_parameters())),peft_config=plan['peft_config'])
        if full:value.update(optimizer=p.cpu_tree(torch,optimizer.state_dict()),rng=dict(cpu=torch.random.get_rng_state(),cuda=torch.cuda.get_rng_state_all()),final_checkpoint=True)
        torch.save(value,file);descriptor=dict(file=str(file),sha256=sha(file),updates=step)
        states[str(step)]=descriptor;return descriptor
    def teacher(item,state_ref,backward,key=None):
        case,features=load_inputs(torch,model,item);evidence=dict(parameter_state=state_ref,feature_packet=item['features'])
        idx=item['compact']['index'];file=data/((key or 'current_failed_teacher')+'.pt')
        try:
            if memory is None:result=profile.ordinary_teacher(torch,model,case,features,evidence=evidence)
            else:
                capture=memory(features,**{k:v.to(model.device) for k,v in case['coordinates'].items()},retain_mass=arm=='mass')
                evidence.update(memory_capture=capture,coordinate_identity={k:p.tensor_info(v) for k,v in case['coordinates'].items()})
                result=runtime.teacher_forward(model,case['text'],capture['tokens'].unsqueeze(0),evidence=evidence)
            if key:profile.retain(torch,file,idx,arm,'teacher',evidence,backward_completed=False)
            if backward:
                (result['loss']/8).backward();counts['backward']+=1;evidence['backward_completed']=True
            descriptor=profile.retain(torch,file,idx,arm,'teacher',evidence,backward_completed=backward) if key else None
            return case,evidence,descriptor
        except BaseException:
            profile.retain(torch,file,idx,arm,'teacher',evidence,partial=True,backward_completed=evidence.get('backward_completed',False));raise
    with ExitStack() as hooks:
        def bump(key):
            def callback(*_):counts[key]+=1
            return callback
        for module,key in ((model,'model'),(model.model,'backbone'),(model.model.visual,'visual'),(model.model.language_model,'language'),
                           (model.model.language_model.norm,'norm'),(model.lm_head,'head')):hooks.callback(module.register_forward_pre_hook(bump(key)).remove)
        for index,layer in enumerate(model.model.language_model.layers):
            def callback(*_,index=index):layers[index]+=1
            hooks.callback(layer.register_forward_pre_hook(callback).remove)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();setup=time.perf_counter()-started
        with trace.open('x') as stream,updates_file.open('x') as update_stream:
            for entry in order:
                m=entry['micro_index'];index=entry['index'];step=entry['updates_before'];tick=time.perf_counter();before_layers=list(layers)
                if m==11992:state(1499)
                state_ref=states['0'] if m==0 else states['1499'] if m==11999 else dict(updates=step)
                key=f'training_{m}' if m in (0,11999) else None
                case,evidence,descriptor=teacher(items[index],state_ref,True,key)
                gradients=profile.grad_record(torch,params,memory)
                need(gradients['all_present'] and gradients['all_finite'] and [a-b for a,b in zip(layers,before_layers)]==[1]*28,
                     'One full forward and present finite trainable gradients required')
                record=dict(entry,arm=arm,sid=items[index]['compact']['sid'],n=items[index]['compact']['n'],target_rows=len(case['target_ids']),
                    mean_ce=evidence['loss'],backward_loss=evidence['loss']/8,gradients_all_present=True,gradients_all_finite=True,
                    layer_forward_calls=[1]*28,seconds=time.perf_counter()-tick,retained=descriptor)
                if descriptor:retained.append(dict(record,parameter_state=state_ref))
                stream.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n')
                del case,evidence
                if (m+1)%8==0:
                    tick=time.perf_counter();norm=torch.nn.utils.clip_grad_norm_(list(live.values()),1.)
                    need(bool(torch.isfinite(norm)),'Nonfinite accumulated gradient norm');optimizer.step();optimizer.zero_grad(set_to_none=True);counts['optimizer']+=1
                    need(all(bool(v.isfinite().all()) for v in live.values()),'Nonfinite updated parameter')
                    update=dict(update=step+1,last_micro_index=m,gradient_norm=float(norm),clip_norm=1.,lr=2e-4,betas=[.9,.999],eps=1e-8,weight_decay=.01,
                        parameter_steps=[float(optimizer.state[v]['step']) for v in live.values()],seconds=time.perf_counter()-tick)
                    need(set(update['parameter_steps'])=={float(step+1)},'Adam update ownership changed')
                    update_stream.write(json.dumps(update,sort_keys=True,allow_nan=False)+'\n');update_stream.flush();stream.flush();check_run()
                    if (step+1)%50==0:print(json.dumps(dict(arm=arm,updates=step+1,last_mean_ce=record['mean_ce'],elapsed=time.perf_counter()-started)),flush=True)
        training_peak=dict(max_allocated_bytes=torch.cuda.max_memory_allocated(),max_reserved_bytes=torch.cuda.max_memory_reserved())
        final=state(1500,True);save(out/'states.json',states);model.eval()
        if memory is not None:memory.eval()
        for phase in ('before','after'):
            if phase=='after':
                restored=torch.load(final['file'],map_location='cpu',weights_only=True)
                profile.restore(torch,params,original['adapter'])
                if memory is not None:profile.restore(torch,dict(memory.named_parameters()),original['memory'])
                profile.restore(torch,params,restored['adapter'])
                if memory is not None:profile.restore(torch,dict(memory.named_parameters()),restored['memory'])
                need(p.adapter_contract(torch,model)==contract and restored['updates']==1500 and restored['peft_config']==p.canonical_peft_config(model),'Exact final adapter configuration required')
            roundtrip[phase]=[]
            for index in plan['roundtrip_indices']:
                with torch.no_grad():_,_,descriptor=teacher(items[index],final,False,f'roundtrip_{phase}_{index}')
                roundtrip[phase].append(dict(index=index,**descriptor))
        for slot,index in enumerate(plan['roundtrip_indices']):
            before=torch.load(roundtrip['before'][slot]['file'],map_location='cpu',weights_only=True);after=torch.load(roundtrip['after'][slot]['file'],map_location='cpu',weights_only=True)
            comparisons.append(dict(index=index,**profile.compare(torch,before,after,exact=True)))
        profile.model_mode(model,params,False)
        if memory is not None:memory.requires_grad_(False)
        provenance=dict(native_identity_sha256=plan['native_identity_sha256'],lora=final,
            memory=None if memory is None else dict(checkpoint=final,tensors=profile.tensor_state(dict(memory.named_parameters()))))
        for index in plan['diagnostic_indices']:
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
            evidence=dict(parameter_state=final,feature_packet=items[index]['features']);file=data/f'natural_{index}.pt'
            try:
                case,features=load_inputs(torch,model,items[index])
                if memory is None:
                    with torch.inference_mode():result=profile.ordinary_generate(torch,model,loaded.processor,case,features,evidence)
                else:
                    with torch.inference_mode():capture=memory(features,**{k:v.to(model.device) for k,v in case['coordinates'].items()},retain_mass=arm=='mass')
                    evidence.update(memory_capture=capture,coordinate_identity={k:p.tensor_info(v) for k,v in case['coordinates'].items()})
                    result=runtime.generate_cold(model,loaded.processor,case['text'],capture['tokens'].unsqueeze(0),provenance=provenance,max_tokens=50,evidence=evidence)
                ids=result['generated_ids'];content=ids[:-1] if ids[-1] in answers.EOS_IDS else ids
                text=loaded.processor.tokenizer.decode(content,skip_special_tokens=False,clean_up_tokenization_spaces=False)
                score=answers.score_answer(text,case['metadata']['atype'],json.loads(case['target_text'])['answer'],ids)
                descriptor=profile.retain(torch,file,index,arm,'cold',evidence,result=result,primary_text=text,score=score)
                torch.cuda.synchronize()
                natural.append(dict(index=index,sid=items[index]['compact']['sid'],n=items[index]['compact']['n'],qtype=items[index]['compact']['qtype'],
                    **descriptor,generated_ids=ids,primary_text=text,score=score,seconds=time.perf_counter()-tick,
                    max_allocated_bytes=torch.cuda.max_memory_allocated(),max_reserved_bytes=torch.cuda.max_memory_reserved()))
                save(out/f'natural_{index}.json',natural[-1]);check_run()
            except BaseException:
                profile.retain(torch,file,index,arm,'cold',evidence,partial=True);raise
    generated=sum(len(r['generated_ids']) for r in natural);expected=dict(model=12004+generated,backbone=12004+generated,visual=0,
        language=12004+generated,norm=12004+generated,head=12004+generated,backward=12000,optimizer=1500)
    need(dict(counts)==expected and layers==[expected['language']]*28 and len(natural)==100,'Complete main actual work inventory differs')
    result=dict(protocol=PROTOCOL,arm=arm,passed=all(r['passed'] for r in comparisons),completed=True,states=states,final_checkpoint=final,
        retained_training=retained,roundtrip=roundtrip,comparisons=comparisons,natural=natural,provenance=provenance,counters=dict(counts),decoder_layer_calls=layers,
        head_rows=3*29555+2*sum(len(items[i]['compact']['target_ids']) for i in plan['roundtrip_indices'])+generated,
        training_trace_file=str(trace),training_trace_sha256=sha(trace),updates_file=str(updates_file),updates_sha256=sha(updates_file),
        setup_seconds=setup,training_peak=training_peak,training_diagnostic_correct=sum(r['score']['correct'] for r in natural),no_validation_test_inference=True,
        objective_achieved=False,requires_independent_cpu_audit=True)
    save(out/'analysis.json',result);save(out/'base_after.json',p.base_metadata(refs));check_run()
    need(read(out/'base_before.json')==read(out/'base_after.json'),'Base parameters changed')
    artifacts={str(file):sha(file) for directory in (out,data,ckpt) for file in directory.iterdir() if file.is_file()};save(out/'artifacts.json',artifacts)
    need(result['passed'],'Final checkpoint teacher parity failed after complete evidence collection')
    return dict(config,analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),artifacts_file=str(out/'artifacts.json'),
        artifacts_sha256=sha(out/'artifacts.json'),final_checkpoint=final,counters=dict(counts),training_diagnostic_correct=result['training_diagnostic_correct'],
        no_validation_test_inference=True,requires_independent_cpu_audit=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check',action='store_true');group.add_argument('--run',action='store_true')
    parser.add_argument('--features',type=Path);parser.add_argument('--boundary',type=Path);parser.add_argument('--plan',type=Path);parser.add_argument('--arm',type=int,choices=range(3));args=parser.parse_args()
    p.native.require_slurm(gpu=args.run);need(int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==4,'Four-core Slurm only')
    if args.check:need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and args.features and args.boundary,'CPU check requires complete feature/boundary reports')
    else:need(args.plan and args.arm is not None,'Main requires frozen plan and arm')
    started=time.perf_counter();tag=f'check_{os.environ["SLURM_JOB_ID"]}' if args.check else f'run_{os.environ["SLURM_ARRAY_JOB_ID"]}_{args.arm}'
    out=OUT/tag;out.mkdir(parents=True,exist_ok=False);own,inherited=archive(out);progress={}
    save(out/'request.json',dict(phase='check' if args.check else 'run',policy=POLICY,source_sha256=own,inherited_source_sha256=inherited))
    try:
        result=check(args,out,started) if args.check else train(args,out,started,progress)
        need(sources()==own and inherited_sources()==inherited,'Main sources changed during execution')
        save(out/'summary.json',dict(result,protocol=PROTOCOL,phase='check' if args.check else 'run',passed=True,completed=True,
            elapsed_seconds=time.perf_counter()-started,source_sha256=own,inherited_source_sha256=inherited))
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),progress=progress,source_sha256=own,
            elapsed_seconds=time.perf_counter()-started,partial_evidence_retained=True,no_automatic_retry=True));raise


if __name__=='__main__':main()

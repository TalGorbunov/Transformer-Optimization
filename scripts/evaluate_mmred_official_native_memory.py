"""All original validation/test worlds, fixed final states, cold native decoding."""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import train_mmred_official_native_memory as training
from scripts import report_mmred_official_native_training as main_audit
from scripts import stage_mmred_official_evaluation_features as features
from scripts.stage_mmred_official_recovery import need,sha,object_sha,save
p=training.p;profile=training.profile;runtime=training.runtime;answers=training.answers;read=training.read
PROTOCOL='mmred_official_native_evaluation'
PROPOSAL='docs/paper/NATIVE_AGGREGATION_MMRED_OFFICIAL_NATIVE_EVALUATOR.md'
PROPOSAL_SHA='7665aaa52b6ecbf5b746d10f0192726efd0ca8bc054e24ff1d4bbd8c815d312f'
STATISTICS_PROTOCOL='docs/paper/NATIVE_AGGREGATION_MMRED_OFFICIAL_EVALUATION_PROTOCOL.md'
STATISTICS_PROTOCOL_SHA='e22eae44ada348fc919830524b86d5cbea755a9a17b851b02e0e049d1e0cd409'
OWN=('scripts/evaluate_mmred_official_native_memory.py',PROPOSAL,
     'slurm/mmred_official_native_evaluation_check.sbatch','slurm/mmred_official_native_evaluation_run.sbatch',STATISTICS_PROTOCOL)
OUT=REPO/'outputs/native_aggregation_vlm/mmred_official_native_evaluation'
DATA=Path('/mnt/data/gabriele/gnn_transformer/mmred_official_native_evaluation')
POLICY=dict(protocol=PROTOCOL,arms=list(profile.ARMS),worlds=1000,validation_worlds=400,test_worlds=600,
    lengths=[8,16,32],checkpoint_update=1500,cpu_seconds=600,gpu_seconds_per_arm=10800,
    maximum_GPU_seconds=32400,maximum_arrays=1,maximum_attempts_per_arm=1,cpu_cores=4,memory_gib=16,
    maximum_new_tokens=50,maximum_model_head_calls_per_arm=50000,vision_calls=0,backward_calls=0,optimizer_steps=0,
    prefix_worlds=3,approximate_DATA_free_space_bytes=300000000000,cold_full_prefix=True,
    no_checkpoint_selection=True,no_accuracy_release_gate=True,all_validation_and_test_unconditional=True,
    original_cache_failure_preserved=True,requires_independent_cpu_audit=True)


def sources():
    need(sha(REPO/PROPOSAL)==PROPOSAL_SHA and sha(REPO/STATISTICS_PROTOCOL)==STATISTICS_PROTOCOL_SHA,'Frozen evaluator/statistical protocol changed')
    return {name:sha(REPO/name) for name in OWN}


def inherited_sources():
    result={}
    for mapping in (training.inherited_sources(),training.sources(),main_audit.sources(),features.inherited_sources(),features.sources()):
        for name,digest in mapping.items():
            need(name not in result or result[name]==digest,'Evaluator dependency conflict');result[name]=digest
    for name in OWN:result.pop(name,None)
    need(all(sha(REPO/name)==digest for name,digest in result.items()),'Frozen evaluator dependency changed')
    return result


def bind(path,bindings,expected=None):
    path=Path(path).resolve();digest=sha(path);need(expected is None or digest==expected,'Evaluation bound input changed')
    bindings[str(path)]=digest;return dict(file=str(path),sha256=digest)


def archive(out):
    own=sources();inherited=inherited_sources();(out/'source').mkdir()
    for name,digest in {**own,**inherited}.items():
        file=out/'source'/name.replace('/','_');file.write_bytes((REPO/name).read_bytes());need(sha(file)==digest,'Evaluator source archive changed')
    return own,inherited


def work_order(rows):
    selected,assigned,prefix=features.selection(rows)
    need(selected==rows,'Only original evaluation rows required')
    return prefix+[r['index'] for r in assigned if r['index'] not in prefix],prefix


def project(boundary,arm,rows,*,completed=(),elapsed=None):
    values=boundary['projection']['arms'][arm];initial={n:values['generation_fifty_token_seconds_by_n'][str(n)] for n in (8,16)}
    initial[32]=2*initial[16];G=dict(initial);done={r['index'] for r in completed}
    for row in completed:G[row['n']]=max(G[row['n']],row['fifty_token_seconds'])
    counts=Counter(r['n'] for i,r in enumerate(rows) if i not in done)
    setup=values['setup_seconds'] if elapsed is None else elapsed;work=sum(counts[n]*G[n] for n in (8,16,32));total=setup+1.25*work+60
    need(all(math.isfinite(v) and v>0 for v in (*G.values(),setup,total)),'Finite positive evaluation timing bound required')
    return dict(passed=total<=10800,projected_seconds=total,cap_seconds=10800,setup_or_elapsed_seconds=setup,
        elapsed_at_prefix_gate=elapsed,remaining_worlds=sum(counts.values()),remaining_counts={str(n):counts[n] for n in (8,16,32)},
        initial_fifty_token_seconds_by_n={str(n):initial[n] for n in (8,16,32)},fifty_token_seconds_by_n={str(n):G[n] for n in (8,16,32)},
        completed_indices=[r['index'] for r in completed],N32_initial_estimate='2 * audited boundary N16 allowance',
        N32_generation_measured=any(r['n']==32 for r in completed),empirical_estimate_not_guarantee=True,
        accuracy_not_used=True,maximum_new_tokens=50)


def state_proof(torch,state,arm,config):
    need(state['arm']==arm and state['updates']==1500 and state['final_checkpoint'] is True
         and state['peft_config']==config and len(state['adapter'])==224
         and sum(v.numel() for v in state['adapter'].values())==10092544
         and all(v.dtype==torch.float32 and bool(v.isfinite().all()) for v in state['adapter'].values()),'Finite exact final adapter required')
    memory=state['memory']
    if arm=='ordinary':need(memory is None,'Ordinary endpoint has no memory core')
    else:
        shapes={'key_weight':(128,3608),'queries':(32,128),'mass_direction':(3584,)}
        need(set(memory)==set(shapes) and all(tuple(memory[k].shape)==shape and memory[k].dtype==torch.float32
             and bool(memory[k].isfinite().all()) for k,shape in shapes.items()),'Finite exact final memory core required')
    return dict(arm=arm,updates=1500,adapter=profile.tensor_state(state['adapter']),
        memory=None if memory is None else profile.tensor_state(memory),peft_config=config,finite=True,checkpoint_used_for_inference_only=True)


def check(args,out,started):
    bindings={};feature_ref=bind(args.features,bindings);stage=features.verify_stage(args.features)
    feature_summary=read(args.features);bind(feature_summary['plan_file'],bindings,feature_summary['plan_sha256'])
    prepared=features.verify_preparation(stage['compact_stage']['file'],streaming=True)
    rows=read(stage['rows_file']);records=read(stage['cases_file']);cached=read(stage['feature_index_file']);order,prefix=work_order(rows)
    need(len(records)==len(cached)==1000 and [r['index'] for r in records]==[r['index'] for r in cached]==list(range(1000))
         and [r['sid'] for r in records]==[r['sid'] for r in cached]==[r['sid'] for r in rows],'Complete exact feature/compact/source order required')
    import torch
    torch.set_num_threads(4);proofs={};boundary_ref=None;common_main_plan=None
    for path in args.main_reports:
        path=Path(path).resolve();ref=bind(path,bindings);summary=read(path);report=main_audit.verify_report(path);arm=report['arm']
        need(arm in profile.ARMS and arm not in proofs and report['passed'] and report['original_run_passed']
             and report['all_scheduled_numerical_evidence_collected'] and not (path.parent/'failure.json').exists(),
             'Three distinct computationally passed main reports required')
        ref.update(analysis_file=summary['analysis_file'],analysis_sha256=summary['analysis_sha256']);bind(ref['analysis_file'],bindings,ref['analysis_sha256'])
        for key in ('input_bindings','artifacts'):bind(summary[key+'_file'],bindings,summary[key+'_sha256'])
        run=report['run'];config=read(run['config_file']);bind(run['config_file'],bindings,run['config_sha256'])
        bind(run['analysis_file'],bindings,run['analysis_sha256']);bind(run['plan_file'],bindings,run['plan_sha256'])
        main_plan=training.verify_plan(run['plan_file'])
        need(config['arm']==arm and config['source_sha256']==training.sources() and config['inherited_source_sha256']==training.inherited_sources()
             and config['plan_file']==run['plan_file'] and config['plan_sha256']==run['plan_sha256']
             and main_plan['native_identity']==stage['native_identity'] and main_plan['native_identity_sha256']==stage['native_identity_sha256'],
             'Main checkpoint/native/source ownership differs')
        current=dict(file=run['plan_file'],sha256=run['plan_sha256'])
        need(common_main_plan is None or current==common_main_plan,'All arms must share one main preparation');common_main_plan=current
        need(boundary_ref is None or boundary_ref==main_plan['boundary_report'],'All arms require one fixed boundary report');boundary_ref=main_plan['boundary_report']
        endpoint=report['final_checkpoint'];need(endpoint['updates']==1500,'Only sole final1500 checkpoint allowed');bind(endpoint['file'],bindings,endpoint['sha256'])
        state=torch.load(endpoint['file'],map_location='cpu',weights_only=True);proof=state_proof(torch,state,arm,config['peft_config'])
        proofs[arm]=dict(main_report=ref,main_run=run,main_plan=current,final_checkpoint=endpoint,peft_config=config['peft_config'],
            state_proof=proof,profile_preparation=main_plan['profile_preparation'],main_array_job_id=config['array_job_id'])
        del state
    need(set(proofs)==set(profile.ARMS) and len({r['main_array_job_id'] for r in proofs.values()})==1,'All three arms of one fixed main array required')
    bind(boundary_ref['file'],bindings,boundary_ref['sha256']);boundary=training.boundary_module().verify_report(boundary_ref['file'])
    boundary_summary=read(boundary_ref['file']);bind(boundary_summary['analysis_file'],bindings,boundary_summary['analysis_sha256'])
    projection={arm:project(boundary,arm,rows) for arm in profile.ARMS};save(out/'projection.json',projection)
    need(all(r['passed'] for r in projection.values()),'Prospective evaluation exceeds10800s per arm; launch held')
    parser=main_audit.parser_proof();save(out/'parser_fixtures.json',parser)
    free=shutil.disk_usage(DATA.parent).free;disk=dict(free_bytes=free,approximate_raw_allowance_bytes=300000000000,
        passed=free>=300000000000,not_a_rigorous_byte_bound=True,no_space_preallocated=True,cpu_audit_storage_separate=True)
    save(out/'disk_preflight.json',disk);need(disk['passed'],'Approximate three-arm raw-output free-space allowance unavailable')
    for key in ('rows_file','cases_file','feature_index_file','target_inventory_file','compact_envelopes_file'):bind(stage[key],bindings)
    save(out/'cases.json',[dict(compact=r,features=c,row=row) for r,c,row in zip(records,cached,rows)]);save(out/'order.json',order)
    plan=dict(protocol=PROTOCOL,policy=POLICY,source_sha256=sources(),inherited_source_sha256=inherited_sources(),
        feature_stage=feature_ref,compact_stage=stage['compact_stage'],main_arms=proofs,boundary_report=boundary_ref,
        native_identity=stage['native_identity'],native_identity_sha256=stage['native_identity_sha256'],precision=stage['precision'],packages=stage['packages'],
        system_prompt=stage['system_prompt'],cases_file=str(out/'cases.json'),rows_file=stage['rows_file'],order_file=str(out/'order.json'),
        prefix_indices=prefix,projection=projection,projection_file=str(out/'projection.json'),
        target_inventory_file=stage['target_inventory_file'],compact_envelopes_file=stage['compact_envelopes_file'],
        statistics_protocol=dict(file=str(REPO/STATISTICS_PROTOCOL),sha256=STATISTICS_PROTOCOL_SHA),input_bindings=bindings,
        artifacts={str(f):sha(f) for f in out.glob('*.json')},all_three_arms_unconditional=True,no_accuracy_release_gate=True)
    save(out/'plan.json',plan);need(time.perf_counter()-started<600,'CPU evaluator preparation cap exceeded')
    return dict(plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),projection=projection,worlds=1000,arms=list(profile.ARMS))


def verify_plan(path):
    path=Path(path).resolve();plan=read(path);summary=read(path.parent/'summary.json')
    need(summary['passed'] is summary['completed'] is True and summary['phase']=='check' and summary['protocol']==PROTOCOL
         and summary['plan_file']==str(path) and summary['plan_sha256']==sha(path) and not (path.parent/'failure.json').exists(),
         'Passed CPU evaluation preparation required')
    need(plan['protocol']==PROTOCOL and plan['policy']==POLICY and plan['source_sha256']==sources()
         and plan['inherited_source_sha256']==inherited_sources() and plan['statistics_protocol']==dict(file=str(REPO/STATISTICS_PROTOCOL),sha256=STATISTICS_PROTOCOL_SHA),
         'Frozen evaluation source/policy/statistics differ')
    for mapping in (plan['input_bindings'],plan['artifacts']):
        for file,digest in mapping.items():need(sha(file)==digest,'Evaluation prepared input changed')
    for name,digest in {**sources(),**inherited_sources()}.items():need(sha(path.parent/'source'/name.replace('/','_'))==digest,'Evaluation archived source changed')
    rows=read(plan['rows_file']);order,prefix=work_order(rows);boundary=read(read(plan['boundary_report']['file'])['analysis_file'])
    need(read(plan['order_file'])==order and plan['prefix_indices']==prefix and set(plan['main_arms'])==set(profile.ARMS)
         and plan['projection']=={arm:project(boundary,arm,rows) for arm in profile.ARMS}
         and all(v['passed'] for v in plan['projection'].values()),'Evaluation source order/projection differs')
    return plan


def frozen_adapter_contract(torch,model):
    from peft.tuners.lora.bnb import Linear4bit
    modules=dict(model.named_modules());wrapped={name:m for name,m in modules.items() if isinstance(m,Linear4bit)}
    need(set(wrapped)==set(p.TARGETS),'Only exact28-language-layer q/k/v/o modules may be adapted')
    for name,m in wrapped.items():
        need(m.r=={'default':16} and m.lora_alpha=={'default':32} and m.scaling=={'default':2.}
             and m.active_adapters==['default'] and not m.disable_adapters and not m.merged
             and set(m.lora_A)==set(m.lora_B)==set(m.lora_dropout)=={'default'}
             and isinstance(m.lora_dropout['default'],torch.nn.Dropout) and m.lora_dropout['default'].p==.05,
             'Exact active default LoRA configuration differs: '+name)
    params=p.adapter_parameters(model)
    expected={target+suffix for target in p.TARGETS for suffix in ('.lora_A.default.weight','.lora_B.default.weight')}
    need(set(params)==expected and len(params)==224 and sum(p.numel() for p in params.values())==10092544
         and all(p.dtype==torch.float32 and p.is_cuda for p in params.values()),'Exact FP32 language adapter parameter budget differs')
    need(all(not value.requires_grad and value.grad is None for value in model.parameters()),'Entire inference model must remain frozen')
    cfg=model.peft_config['default'].to_dict()
    need(cfg['r']==16 and cfg['lora_alpha']==32 and cfg['lora_dropout']==.05 and cfg['bias']=='none'
         and cfg['init_lora_weights'] is True and not cfg['use_rslora'] and not cfg['use_dora']
         and cfg['modules_to_save'] is None and p.require_effective_targets(model,model.peft_config['default'])==set(p.TARGETS),'Saved PEFT configuration differs')
    return dict(targets=list(p.TARGETS),trainable_parameters=10092544,tensors={name:dict(shape=list(p.shape),dtype=str(p.dtype),object_id=id(p)) for name,p in params.items()},
        active_adapter='default',scaling=2.,rank=16,alpha=32,dropout=.05)


def run(args,out,data,started,progress):
    import torch
    from gnnformer.runtime import load_runtime
    from gnnformer.native_visual_memory import NativeVisualMemory
    torch.set_num_threads(4);plan=verify_plan(args.plan);arm=profile.ARMS[args.arm];proof=plan['main_arms'][arm]
    need(torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA B200','One B200 required')
    need(os.environ.get('SLURM_ARRAY_TASK_ID')==str(args.arm) and os.environ.get('SLURM_ARRAY_TASK_COUNT')=='3','One fixed three-arm array required')
    items=read(plan['cases_file']);rows=read(plan['rows_file']);order=read(plan['order_file'])
    boundary=read(read(plan['boundary_report']['file'])['analysis_file']);final=proof['final_checkpoint']
    loaded=load_runtime(str(profile.preparation.MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda')
    model=loaded.model.eval().requires_grad_(False);hardware=p.frozen_joint.live_identity(torch,loaded,plan)
    need(p.installed({})==plan['packages'],'Native packages differ')
    refs=list(model.named_parameters());base=p.base_metadata(refs);save(out/'base_before.json',base)
    contract=p.install_lora(torch,model,out/'actual_peft_config.json');params=p.adapter_parameters(model)
    need(sha(final['file'])==final['sha256'],'Final checkpoint changed')
    state=torch.load(final['file'],map_location='cpu',weights_only=True)
    need(state_proof(torch,state,arm,proof['peft_config'])==proof['state_proof']
         and p.canonical_peft_config(model)==proof['peft_config'],'CPU-bound checkpoint/PEFT proof differs')
    profile.restore(torch,params,state['adapter']);memory=None if arm=='ordinary' else NativeVisualMemory().to(model.device)
    if memory is not None:profile.restore(torch,dict(memory.named_parameters()),state['memory']);memory.eval().requires_grad_(False)
    profile.model_mode(model,params,False);p.check_base(refs,base,model)
    live={**params,**({} if memory is None else {'memory.'+k:v for k,v in memory.named_parameters()})}
    versions={k:(id(v),v._version) for k,v in live.items()}
    need(all(not v.requires_grad and v.grad is None for v in model.parameters())
         and all(not v.requires_grad and v.grad is None for v in live.values()),'Entire fitted model/core must be frozen')
    provenance=dict(native_identity_sha256=plan['native_identity_sha256'],lora=final,
        memory=None if memory is None else dict(checkpoint=final,tensors=profile.tensor_state(dict(memory.named_parameters()))))
    config=dict(protocol=PROTOCOL,policy=POLICY,arm=arm,array_job_id=os.environ['SLURM_ARRAY_JOB_ID'],
        plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),source_sha256=sources(),inherited_source_sha256=inherited_sources(),
        hardware=hardware,native_identity=plan['native_identity'],native_identity_sha256=plan['native_identity_sha256'],
        main_report=proof['main_report'],main_run=proof['main_run'],main_plan=proof['main_plan'],final_checkpoint=final,
        feature_stage=plan['feature_stage'],compact_stage=plan['compact_stage'],profile_preparation=proof['profile_preparation'],
        peft_config=proof['peft_config'],contract=contract,state_proof=proof['state_proof'],provenance=provenance,
        statistics_protocol=plan['statistics_protocol'],data_directory=str(data),no_checkpoint_copy=True)
    save(out/'config.json',config);del state
    counts=Counter(dict.fromkeys(('model','backbone','visual','language','norm','head','backward','optimizer'),0));layers=[0]*28
    natural=[];progress.update(counters=counts,natural=natural);gate=None
    def check_run():
        need(time.perf_counter()-started<10800,'Fixed evaluation GPU cap exceeded');p.check_base(refs,base,model)
        need(versions=={k:(id(v),v._version) for k,v in live.items()} and frozen_adapter_contract(torch,model)==contract
             and all(not v.requires_grad and v.grad is None for v in live.values()),'Fitted endpoint/PEFT state changed')
    with ExitStack() as hooks,(out/'natural.jsonl').open('x') as stream:
        def bump(key):
            def callback(*_):
                counts[key]+=1;need(key!='visual','Evaluation must use shared frozen features, without vision')
            return callback
        for module,key in ((model,'model'),(model.model,'backbone'),(model.model.visual,'visual'),
                           (model.model.language_model,'language'),(model.model.language_model.norm,'norm'),(model.lm_head,'head')):
            hooks.callback(module.register_forward_pre_hook(bump(key)).remove)
        for index,layer in enumerate(model.model.language_model.layers):
            def callback(*_,index=index):layers[index]+=1
            hooks.callback(layer.register_forward_pre_hook(callback).remove)
        torch.cuda.synchronize();setup=time.perf_counter()-started
        for execution_index,index in enumerate(order):
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
            evidence=dict(parameter_state=final,feature_packet=items[index]['features']);file=data/f'natural_{index:04d}.pt'
            result=None;text=None;score=None
            try:
                case,value=training.load_inputs(torch,model,items[index])
                if memory is None:
                    with torch.inference_mode():result=profile.ordinary_generate(torch,model,loaded.processor,case,value,evidence)
                else:
                    with torch.inference_mode():capture=memory(value,**{k:v.to(model.device) for k,v in case['coordinates'].items()},retain_mass=arm=='mass')
                    evidence.update(memory_capture=capture,coordinate_identity={k:p.tensor_info(v) for k,v in case['coordinates'].items()})
                    result=runtime.generate_cold(model,loaded.processor,case['text'],capture['tokens'].unsqueeze(0),provenance=provenance,max_tokens=50,evidence=evidence)
                ids=result['generated_ids'];need(1<=len(ids)<=50,'Fixed native generated-token bound differs')
                content=ids[:-1] if ids[-1] in answers.EOS_IDS else ids
                text=loaded.processor.tokenizer.decode(content,skip_special_tokens=False,clean_up_tokenization_spaces=False)
                score=answers.score_answer(text,case['metadata']['atype'],json.loads(case['target_text'])['answer'],ids)
                descriptor=profile.retain(torch,file,index,arm,'cold',evidence,result=result,primary_text=text,score=score)
                check_run();row=dict(index=index,execution_index=execution_index,sid=rows[index]['sid'],n=rows[index]['n'],
                    qtype=rows[index]['qtype'],pilot_role=rows[index]['pilot_role'],**descriptor,generated_ids=ids,primary_text=text,score=score,
                    max_allocated_bytes=torch.cuda.max_memory_allocated(),max_reserved_bytes=torch.cuda.max_memory_reserved())
                stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');stream.flush();torch.cuda.synchronize()
                row['seconds']=time.perf_counter()-tick;row['fifty_token_seconds']=row['seconds']*50/len(ids)
                natural.append(row);save(out/f'natural_{index:04d}.json',row);progress['completed_worlds']=len(natural)
                del case,value,evidence,result
                if memory is not None:del capture
                if execution_index==2:
                    elapsed=time.perf_counter()-started;gate=project(boundary,arm,rows,completed=natural,elapsed=elapsed)
                    save(out/'prefix_natural.json',natural);save(out/'prefix_projection.json',gate)
                    need([r['index'] for r in natural]==plan['prefix_indices'] and gate['N32_generation_measured']
                         and gate['passed'],'Measured first three worlds hold remaining evaluation beyond10800s')
            except BaseException:
                if 'evidence' in locals():
                    partial=profile.retain(torch,file.with_name(file.stem+'_partial.pt'),index,arm,'cold',evidence,partial=True,result=result,primary_text=text,score=score)
                    progress.setdefault('partial_packets',[]).append(partial)
                raise
    generated=sum(len(r['generated_ids']) for r in natural);expected=dict(model=generated,backbone=generated,visual=0,
        language=generated,norm=generated,head=generated,backward=0,optimizer=0)
    need(len(natural)==1000 and generated<=50000 and gate is not None and gate['passed']
         and dict(counts)==expected and layers==[generated]*28,'Complete evaluation native work inventory differs')
    check_run();need(profile.tensor_state(params)==proof['state_proof']['adapter']
        and (memory is None or profile.tensor_state(dict(memory.named_parameters()))==proof['state_proof']['memory']),
        'Exact fitted endpoint bytes changed')
    natural.sort(key=lambda r:r['index']);need([r['index'] for r in natural]==list(range(1000)),'Source-ordered complete evaluation outputs required')
    save(out/'base_after.json',p.base_metadata(refs));need(read(out/'base_before.json')==read(out/'base_after.json'),'Frozen native base changed')
    analysis=dict(protocol=PROTOCOL,arm=arm,passed=True,completed=True,final_checkpoint=final,natural=natural,provenance=provenance,
        counters=expected,decoder_layer_calls=layers,head_rows=generated,setup_seconds=setup,original_projection=plan['projection'][arm],
        prefix_projection=gate,all1000_completed=True,all_three_arms_required=True,requires_independent_cpu_audit=True,
        computational_completion_only=True,objective_achieved=False,accuracy_not_used_for_execution=True,
        statistics_protocol=plan['statistics_protocol'],no_checkpoint_copy=True)
    save(out/'analysis.json',analysis)
    artifacts={str(f):sha(f) for directory in (out,data) for f in directory.iterdir() if f.is_file()};save(out/'artifacts.json',artifacts)
    check_run()
    return dict(config,analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),
        artifacts_file=str(out/'artifacts.json'),artifacts_sha256=sha(out/'artifacts.json'),counters=expected,head_rows=generated,
        worlds=1000,requires_independent_cpu_audit=True,computational_completion_only=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--check',action='store_true');group.add_argument('--run',action='store_true')
    ap.add_argument('--features',type=Path);ap.add_argument('--main-reports',type=Path,nargs=3)
    ap.add_argument('--plan',type=Path);ap.add_argument('--arm',type=int,choices=range(3));args=ap.parse_args()
    p.native.require_slurm(gpu=args.run);need(int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==4,'Four-core Slurm only')
    phase='check' if args.check else 'run';started=time.perf_counter()
    if args.check:
        need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and args.features and args.main_reports,'CPU check needs features and all three main reports')
        need(not any(OUT.glob('check_*')),'One CPU evaluation release attempt');tag=f'check_{os.environ["SLURM_JOB_ID"]}'
    else:
        need(args.plan and args.arm is not None and os.environ.get('SLURM_ARRAY_JOB_ID'),'Evaluation requires one fixed three-arm array')
        array=os.environ['SLURM_ARRAY_JOB_ID'];need(all(x.name.split('_')[1]==array for x in OUT.glob('run_*')),'Only one evaluation array is released')
        tag=f'run_{array}_{args.arm}'
    out=OUT/tag;out.mkdir(parents=True,exist_ok=False);own,inherited=archive(out)
    save(out/'request.json',dict(phase=phase,features=None if args.features is None else str(args.features.resolve()),
        main_reports=None if args.main_reports is None else [str(v.resolve()) for v in args.main_reports],
        plan=None if args.plan is None else str(args.plan.resolve()),arm=args.arm,policy=POLICY,source_sha256=own,inherited_source_sha256=inherited))
    progress={}
    try:
        if args.check:result=check(args,out,started)
        else:
            data=DATA/tag;data.mkdir(parents=True,exist_ok=False);result=run(args,out,data,started,progress)
        need(time.perf_counter()-started<POLICY['cpu_seconds' if args.check else 'gpu_seconds_per_arm']
             and sources()==own and inherited_sources()==inherited,'Evaluation source or allocation cap changed')
        save(out/'summary.json',dict(result,protocol=PROTOCOL,phase=phase,policy=POLICY,passed=True,completed=True,
            elapsed_seconds=time.perf_counter()-started,source_sha256=own,inherited_source_sha256=inherited))
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),phase=phase,progress=progress,
            elapsed_seconds=time.perf_counter()-started,source_sha256=own,partial_evidence_retained=True,no_automatic_retry=True));raise


if __name__=='__main__':main()

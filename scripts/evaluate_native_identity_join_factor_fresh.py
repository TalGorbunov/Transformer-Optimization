"""Held N16-only natural evaluation of both fixed visual factor endpoints.

One CPU preparation includes the complete timing/envelope release certificate.
No fitting, new statistics, profile, head replay, or longer-length evaluation.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import diagnose_native_identity_join_factor_native_v2 as previous
from scripts import report_native_identity_join_factor_native_v2 as previous_report
from scripts import stage_native_identity_join_factor_fresh as stage
from scripts.stage_native_vision_v6_teacher import MODEL,need,read,save,sha,object_sha
runtime=previous.runtime;native=runtime.native;cached=previous.cached;oracle=previous.oracle
software=previous.software;factor_audit=previous.factor_audit;tensor_info=previous.tensor_info
cpu_tree=previous.cpu_tree;basic_result=previous.basic_result;bind=previous.bind
OUT=REPO/'outputs/native_aggregation_vlm/identity_join_factor_fresh/evaluation'
DATA=Path('/mnt/data/gabriele/gnn_transformer/identity_join_factor_fresh/evaluation')
CKPT=Path('/mnt/ckpts/gabriele/gnn_transformer/identity_join_factor_fresh/evaluation')
PROTOCOL='native_identity_join_factor_fresh_N16';ARMS=('product','additive')
JOBS={arm:f'identity_join_factor_fresh_eval_{arm}' for arm in ARMS}
PROPOSAL='docs/paper/NATIVE_AGGREGATION_FACTOR_FRESH_INFERENCE_PROPOSAL.md'
PROPOSAL_SHA='9124ce838f635df191949c9e924a7d6b11f6ac3923206420957a81f6e04c4eb0'
POLICY=dict(protocol=PROTOCOL,arms=list(ARMS),checkpoint_step=6000,maximum_new_tokens=4,n_frames=16,contexts=270,
    panels={'A':108,'B':108,'C':54},families={'A':36,'B':36,'C':18},whole_threshold={'A':98,'B':98,'C':49},
    family_threshold={'A':33,'B':33,'C':16},paired_only=True,no_fitting=True,no_new_statistics=True,no_checkpoint_selection=True,
    no_profile=True,no_extra_gpu_heads=True,no_cpu_head_replay=True,no_longer_lengths=True,
    model_cap=1080,vision_calls=270,native_head_rows_cap=18360,cpu_check_seconds=600,cpu_report_seconds=600,
    run_seconds=1200,campaign_seconds=2400,max_concurrent_gpus=2,historical_four_token_floor=2.30838355794549,
    projection_multiplier=1.25,projection_reserve=60.,functional_atol=1e-4,functional_rtol=1e-4,
    timing_release_in_same_CPU_check=True,validated_N16_envelope_required=True)
OWN=('scripts/evaluate_native_identity_join_factor_fresh.py','scripts/report_native_identity_join_factor_fresh.py',
    'slurm/native_identity_join_factor_fresh_eval_check.sbatch','slurm/native_identity_join_factor_fresh_eval_run.sbatch',
    'slurm/native_identity_join_factor_fresh_eval_report.sbatch',PROPOSAL)


def sources():
    need(sha(REPO/PROPOSAL)==PROPOSAL_SHA,'Held fresh inference protocol changed')
    return {name:sha(REPO/name) for name in OWN}


def inherited_sources():
    native_sources={**previous.inherited_sources(),**previous.sources()};data_sources=stage.source_hashes()
    need(len(native_sources)==123 and len(data_sources)==11,'Native123/data11 ancestor inventory differs')
    for name in set(native_sources)&set(data_sources):need(native_sources[name]==data_sources[name],'Conflicting inherited source')
    return {**native_sources,**data_sources}


def snapshot(out):
    frozen=sources();(out/'source').mkdir()
    for name,h in frozen.items():
        path=out/'source'/name.replace('/','_');path.write_bytes((REPO/name).read_bytes());need(sha(path)==h,'Source snapshot changed')
    save(out/'source_hashes.json',frozen);save(out/'inherited_sources.json',dict(source_sha256=inherited_sources()));return frozen


def selected_rows(manifest):
    rows=[row for panel in ('A','B','C') for row in manifest['splits'][f'fresh_{panel}_N16']['samples']]
    need(len(rows)==len({r['sid'] for r in rows})==270 and Counter(r['panel'] for r in rows)==POLICY['panels']
        and all(r['n_frames']==16 for r in rows),'Exact frozen N16 cohort required')
    # Reuse the scoring coverage check with no actual outcome attached.
    criterion([dict(row,exact=False,first_token_correct=False) for row in rows]);return rows


def criterion(rows):
    need(len(rows)==len({r['sid'] for r in rows})==270 and all(r['n_frames']==16 for r in rows),'All270 N16 outcomes required')
    panels={}
    for panel,count in POLICY['panels'].items():
        values=[r for r in rows if r['panel']==panel];groups={}
        for r in values:groups.setdefault(r['contrast_id'],[]).append(r)
        need(len(values)==count and len(groups)==POLICY['families'][panel] and all(len(g)==3 and {r['variant'] for r in g}=={0,1,2} for g in groups.values()),
             'Complete three-variant fresh family inventory differs')
        whole=sum(r['exact'] for r in values);complete=sum(all(r['exact'] for r in g) for g in groups.values())
        panels[panel]=dict(passed=whole>=POLICY['whole_threshold'][panel] and complete>=POLICY['family_threshold'][panel],
            whole_correct=whole,first_correct=sum(r['first_token_correct'] for r in values),contexts=count,
            complete_families=complete,families=len(groups),thresholds=dict(whole_correct=POLICY['whole_threshold'][panel],complete_families=POLICY['family_threshold'][panel]))
    return dict(panels=panels,qualifies_A_and_B=panels['A']['passed'] and panels['B']['passed'],C_separate=panels['C']['passed'],
        natural_whole_answer_and_eos=True,N16_only=True)


def timing_certificate(analysis):
    observations=[]
    for phase,group,count in (('profile','profiles',4),('main','runs',108)):
        need(set(analysis[group])==set(ARMS),'Both native arms required in timing pool')
        for arm in ARMS:
            item=analysis[group][arm];previous_report.bound_outputs(item)
            need(item['phase']==phase and item['arm']==arm and len(item['four_token_seconds'])==count,'Complete prior timing cohort differs')
            observations.append(dict(phase=phase,arm=arm,summary_file=str(Path(item['directory'])/'summary.json'),summary_sha256=item['summary_sha256'],
                setup_seconds=item['setup_seconds'],four_token_seconds=item['four_token_seconds']))
    return project_timings(observations)


def project_timings(observations):
    need([(v['phase'],v['arm'],len(v['four_token_seconds'])) for v in observations]==[(phase,arm,count) for phase,count in (('profile',4),('main',108)) for arm in ARMS],
         'All native timing cells must remain in canonical order')
    setups=[v['setup_seconds'] for v in observations];times=[x for v in observations for x in v['four_token_seconds']]
    need(len(setups)==4 and len(times)==224 and all(math.isfinite(x) and x>0 for x in setups+times),'All prior positive timings required')
    setup=max(setups);observed=max(times);bound=max(observed,POLICY['historical_four_token_floor']);seconds=setup+1.25*270*bound+60.
    return dict(passed=seconds<=1200,observations=observations,setup_observations=4,trajectory_observations=224,
        pooled_setup_seconds=setup,maximum_observed_four_token_seconds=observed,historical_floor_seconds=POLICY['historical_four_token_floor'],
        pooled_scene_seconds=bound,multiplier=1.25,contexts=270,reserve_seconds=60.,projected_seconds=seconds,cap_seconds=1200,
        all_native_profiles_mains_both_arms_and_zero_cases=True,physical_runtime_guarantee=False)


def envelope_row(torch,bundle,layout):
    runtime.validate_bundle(bundle);m=bundle['metadata'];inputs=bundle['inputs'];n=m['n_frames']
    need(n==16 and m['row_count']==17 and m['global_row']==16 and m['row_kinds']==['local']*16+['global'] and m['resize']==392
         and m['prefix_ids']==[] and m['processor_parity_checked'] is True and layout['metadata']['every_unpadded_row_exact'] is True,
         'Only original N16 mixed native layouts define this envelope')
    mask=inputs['attention_mask'];positions=layout['position_ids'];deltas=layout['rope_deltas']
    need(bool(((mask==0)|(mask==1)).all()) and bool((mask[:,1:]>=mask[:,:-1]).all())
         and positions.shape==(3,17,m['prompt_width']) and deltas.shape==(17,1),'Mask/native position shape differs')
    return dict(sid=m['sid'],prompt_width=m['prompt_width'],maximum_local_tokens=max(m['row_prompt_tokens'][:16]),global_tokens=m['row_prompt_tokens'][16],
        input_shapes={k:list(v.shape) for k,v in inputs.items()},input_dtypes={k:str(v.dtype) for k,v in inputs.items()},
        image_grid_thw=inputs['image_grid_thw'].tolist(),position_dtype=str(positions.dtype),delta_dtype=str(deltas.dtype),
        maximum_absolute_position=int(positions.abs().max()),maximum_absolute_rope_delta=int(deltas.abs().max()),
        binary_left_padded_mask=True,maximum_new_tokens=4,resize=392,rows=17)


def envelope_limits(rows):
    need(len(rows)==54,'All54 original native N16 training inputs required');first=rows[0]
    exact=('input_dtypes','image_grid_thw','position_dtype','delta_dtype','binary_left_padded_mask','maximum_new_tokens','resize','rows')
    need(all(all(row[k]==first[k] for k in exact) for row in rows),'Validated N16 structural conventions differ')
    shapes={key:[max(row['input_shapes'][key][i] for row in rows) for i in range(len(shape))] for key,shape in first['input_shapes'].items()}
    keys=('prompt_width','maximum_local_tokens','global_tokens','maximum_absolute_position','maximum_absolute_rope_delta')
    return dict(exact={k:first[k] for k in exact},maximum={k:max(row[k] for row in rows) for k in keys},maximum_input_shapes=shapes)


def within_envelope(row,limits):
    need(all(row[k]==v for k,v in limits['exact'].items()) and all(row[k]<=v for k,v in limits['maximum'].items())
        and set(row['input_shapes'])==set(limits['maximum_input_shapes'])
        and all(len(row['input_shapes'][k])==len(shape) and all(a<=b for a,b in zip(row['input_shapes'][k],shape))
                for k,shape in limits['maximum_input_shapes'].items()),'Fresh prepared input exceeds the validated N16 envelope; release held')


def check(args,out,frozen):
    import torch
    from transformers import AutoProcessor
    torch.set_num_threads(4);bindings={};native_path=args.native_report.resolve();native_path=native_path/'summary.json' if native_path.is_dir() else native_path
    analysis=previous_report.verify_native_report(native_path);proof=read(native_path);bind(native_path,bindings)
    native_ref=dict(file=str(native_path),sha256=sha(native_path),analysis_file=proof['analysis_file'],analysis_sha256=proof['analysis_sha256'],
        plan_file=proof['plan_file'],plan_sha256=proof['plan_sha256'])
    for key in ('analysis','plan'):bind(native_ref[key+'_file'],bindings,native_ref[key+'_sha256'])
    parent=previous.verify_plan(native_ref['plan_file'],ancestors=False)
    stage_path=args.stage_report.resolve();stage_path=stage_path/'summary.json' if stage_path.is_dir() else stage_path
    manifest=stage.verify_stage(stage_path);stage_summary=read(stage_path);bind(stage_path,bindings)
    stage_ref=dict(file=str(stage_path),sha256=sha(stage_path),manifest_file=stage_summary['manifest_file'],manifest_sha256=stage_summary['manifest_sha256'])
    bind(stage_ref['manifest_file'],bindings,stage_ref['manifest_sha256'])
    need(manifest['native_report']['file']==str(native_path) and manifest['native_report']['sha256']==native_ref['sha256']
        and manifest['native_report']['native_identity']==parent['native_identity'],'Fresh staging and inference native evidence differs')
    rows=selected_rows(manifest);save(out/'rows.json',rows);save(out/'dependency_bindings.json',dict(native_report=native_ref,stage_report=stage_ref,input_bindings=bindings))
    timing=timing_certificate(analysis);save(out/'timing_certificate.json',timing);need(timing['passed'],'Conservative timing forecast exceeds1200 seconds')
    for value in timing['observations']:bind(value['summary_file'],bindings,value['summary_sha256'])
    processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,use_fast=False,local_files_only=True)
    owner,rope,api=software.backend.native_api(processor);need(api==parent['native_identity']['native_api'],'Installed native API differs')
    bind(parent['prepared_file'],bindings,parent['runtime_bindings'][parent['prepared_file']]);old_prepared=read(parent['prepared_file'])
    bind(parent['rows_file'],bindings,parent['runtime_bindings'][parent['rows_file']]);old_rows=[r for r in read(parent['rows_file']) if r['n_frames']==16]
    bind(parent['layout_file'],bindings,parent['layout_sha256']);old_layouts=torch.load(parent['layout_file'],map_location='cpu',weights_only=True);old_envelopes=[]
    for row in old_rows:
        item=old_prepared[row['sid']];bind(item['file'],bindings,item['sha256']);bundle=torch.load(item['file'],map_location='cpu',weights_only=True)
        need(bundle['metadata']==item['metadata'],'Validated old input metadata changed');old_envelopes.append(envelope_row(torch,bundle,old_layouts[row['sid']]))
    limits=envelope_limits(old_envelopes);del old_layouts
    data=DATA/out.name;data.mkdir(parents=True,exist_ok=False);prepared={};layouts={};envelopes=[]
    for i,row in enumerate(rows):
        need(processor.tokenizer.encode(row['gold'],add_special_tokens=False)+[151645]==row['target_ids'],'Fresh canonical name target differs')
        for image in row['image_files']:bind(image['path'],bindings,image['sha256'])
        bundle=runtime.prepare_scene(processor,stage.runtime_view(row),verify_processor_parity=True)
        layout=native.audit_layout(lambda **kw:rope(owner,**kw),bundle);env=envelope_row(torch,bundle,layout)
        envelopes.append(env)
        try:within_envelope(env,limits)
        except ValueError:
            save(out/'envelope_failure.json',dict(sid=row['sid'],actual=env,limits=limits,held=True));raise
        path=data/f'bundle_{i:03d}.pt';torch.save(cpu_tree(torch,bundle),path)
        prepared[row['sid']]=dict(file=str(path),sha256=sha(path),metadata=bundle['metadata'],layout=layout['metadata']);layouts[row['sid']]=layout
        save(out/f'prepared_{i:03d}.json',dict(sid=row['sid'],file=str(path),sha256=sha(path),envelope=env))
    save(out/'prepared.json',prepared);layout_file=data/'layouts.pt';torch.save(layouts,layout_file)
    envelope=dict(passed=True,original=old_envelopes,fresh=envelopes,limits=limits,old_contexts=54,fresh_contexts=270,
        actual_owner_bound_native_layout=True,no_additional_GPU_profile=True)
    save(out/'envelope.json',envelope)
    ckpt=CKPT/out.name;ckpt.mkdir(parents=True,exist_ok=False);checkpoints={}
    for arm in ARMS:
        descriptor=parent['checkpoints'][arm]['fitted'];need(descriptor==analysis['runs'][arm]['checkpoint'],'Native final endpoint differs')
        bind(descriptor['file'],bindings,descriptor['sha256']);path=ckpt/f'{arm}_fitted.pt';shutil.copyfile(descriptor['file'],path)
        need(sha(path)==descriptor['sha256'],'Fresh checkpoint copy differs');packet=torch.load(path,map_location='cpu',weights_only=True)
        need({k:tensor_info(v) for k,v in packet['branch'].items()}==descriptor['state'],'Fresh checkpoint tensor table differs')
        checkpoints[arm]=dict(fitted=dict(file=str(path),sha256=sha(path),state=descriptor['state']))
    bind(parent['stats_file'],bindings,parent['stats_sha256']);stats=torch.load(parent['stats_file'],map_location='cpu',weights_only=True)
    need({k:tensor_info(stats[k]) for k in ('global_mean','scale')}==parent['conditioning_identity'],'Original global statistics changed')
    tests=dict(driver=self_test(torch))
    from scripts.report_native_identity_join_factor_fresh import self_test as report_self_test
    tests['reporter']=report_self_test(torch);save(out/'selftests.json',tests)
    runtime_bindings={}
    for path in (out/'rows.json',out/'prepared.json',out/'envelope.json',out/'timing_certificate.json',layout_file,Path(parent['stats_file']),Path(parent['native_software_plan']['file'])):
        bind(path,runtime_bindings)
    for item in prepared.values():bind(item['file'],runtime_bindings,item['sha256'])
    for states in checkpoints.values():bind(states['fitted']['file'],runtime_bindings,states['fitted']['sha256'])
    plan=dict(schema_version=1,protocol=PROTOCOL,policy=POLICY,source_sha256=frozen,inherited_source_sha256=inherited_sources(),tests=tests,
        native_report=native_ref,stage_report=stage_ref,timing_certificate=timing,envelope_file=str(out/'envelope.json'),envelope_sha256=sha(out/'envelope.json'),
        rows_file=str(out/'rows.json'),prepared_file=str(out/'prepared.json'),layout_file=str(layout_file),layout_sha256=sha(layout_file),checkpoints=checkpoints,
        stats_file=parent['stats_file'],stats_sha256=parent['stats_sha256'],conditioning_identity=parent['conditioning_identity'],
        native_model_file=parent['native_model_file'],native_model_sha256=parent['native_model_sha256'],
        native_identity=parent['native_identity'],native_identity_sha256=parent['native_identity_sha256'],native_module_identity=parent['native_module_identity'],
        native_software_plan=parent['native_software_plan'],cached_report=parent['cached_report'],cached_plan=parent['cached_plan'],
        input_bindings=bindings,runtime_bindings=runtime_bindings,N16_only=True,no_fitting=True,no_new_statistics=True,timing_release_in_same_CPU_check=True)
    save(out/'plan.json',plan);(out/'plan.sha256').write_text(sha(out/'plan.json')+'\n')
    return dict(passed=True,completed=True,phase='check',protocol=PROTOCOL,source_sha256=frozen,inherited_source_sha256=plan['inherited_source_sha256'],
        plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),tests=tests,contexts=270,timing_certificate=timing,
        envelope_file=plan['envelope_file'],envelope_sha256=plan['envelope_sha256'],no_model_or_head_forward=True)


def verify_plan(path,ancestors=False):
    path=Path(path).resolve();plan=read(path)
    need(plan['protocol']==PROTOCOL and plan['policy']==POLICY and plan['source_sha256']==sources()
         and plan['inherited_source_sha256']==inherited_sources() and sha(path)==path.with_suffix('.sha256').read_text().strip(),
         'Frozen fresh preparation changed')
    for name,h in plan['source_sha256'].items():need(sha(path.parent/'source'/name.replace('/','_'))==h,'CPU source snapshot changed')
    need(read(path.parent/'inherited_sources.json')==dict(source_sha256=plan['inherited_source_sha256']),'CPU ancestor map changed')
    for file,h in plan['runtime_bindings'].items():need(sha(Path(file))==h,'Consumed fresh input changed: '+file)
    if ancestors:
        for file,h in plan['input_bindings'].items():need(sha(Path(file))==h,'Bound fresh ancestry changed: '+file)
    proof=read(path.parent/'summary.json')
    need(proof['passed'] is proof['completed'] is True and proof['phase']=='check' and proof['plan_sha256']==sha(path)
         and proof['source_sha256']==plan['source_sha256'] and proof['timing_certificate']==plan['timing_certificate']
         and plan['timing_certificate']==project_timings(plan['timing_certificate']['observations']) and plan['timing_certificate']['passed'] is True
         and plan['timing_release_in_same_CPU_check'] is True and read(plan['envelope_file'])['passed'] is True,
         'Passed same-job envelope/timing release required')
    return plan


def single_attempt_guard(out,arm):
    need(os.environ.get('SLURM_JOB_NAME')==JOBS[arm],'Explicit registered per-arm fresh job name required')
    command=['sacct','-X','-u',os.environ['USER'],'--starttime=2026-09-01','--parsable2','--noheader',
        '--format=JobIDRaw,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,Start,End']
    raw=subprocess.run(command,capture_output=True,text=True,check=True).stdout;file=out/'launch_sacct.psv';file.write_text(raw)
    save(out/'launch_accounting.json',dict(command=command,file=str(file),sha256=sha(file)))
    allocations=factor_audit.allocation_rows(raw,set(JOBS.values())|{'identity_join_factor_fresh_eval_run'});seen=Counter()
    for row in allocations:
        seen[row['name']]+=1
        need(row['name'] in JOBS.values() and seen[row['name']]==1 and row['partition']=='gpu' and row['gpus']==1,
             'Every failed/zero fresh allocation counts; no second attempt or unregistered job')
        if row['name']==JOBS[arm]:need(row['job_id']==os.environ['SLURM_JOB_ID'],'Previous attempt forbids launch')
    need(sum(row['gpu_seconds'] for row in allocations)<=2400 and len(allocations)<=2,'Fresh campaign cap exceeded')


def self_test(torch):
    import copy
    rows=[dict(sid=f'{panel}/{f}/{v}',panel=panel,n_frames=16,contrast_id=f'{panel}/{f}',variant=v,exact=True,first_token_correct=True)
        for panel,count in POLICY['families'].items() for f in range(count) for v in range(3)]
    need(criterion(rows)['qualifies_A_and_B'] and criterion(rows)['C_separate'],'Complete fresh fixture must pass')
    bad=copy.deepcopy(rows)
    for i in (0,3,6,9):bad[i]['exact']=False
    need(not criterion(bad)['qualifies_A_and_B'] and criterion(bad)['panels']['A']['whole_correct']==104,'Marginal score cannot replace complete triples')
    failed=0
    try:criterion(rows[:-1])
    except ValueError:failed+=1
    observations=[dict(phase=phase,arm=arm,setup_seconds=30.,four_token_seconds=[1.]*count)
        for phase,count in (('profile',4),('main',108)) for arm in ARMS]
    need(project_timings(observations)['passed'] and project_timings(observations)['pooled_scene_seconds']==POLICY['historical_four_token_floor'],
         'Historical timing floor must survive favorable measurements')
    observations[-1]['four_token_seconds'][-1]=10.;need(not project_timings(observations)['passed'],'Worst full-main timing must prevent release')
    sample=dict(sid='fixture',prompt_width=20,maximum_local_tokens=20,global_tokens=5,input_shapes={'input_ids':[17,20]},
        input_dtypes={'input_ids':'torch.int64'},image_grid_thw=[[1,2,2]]*16,position_dtype='torch.int64',delta_dtype='torch.int64',
        maximum_absolute_position=20,maximum_absolute_rope_delta=15,binary_left_padded_mask=True,maximum_new_tokens=4,resize=392,rows=17)
    limits=envelope_limits([sample]*54);within_envelope(sample,limits)
    for update in (dict(prompt_width=21),dict(input_shapes={'input_ids':[17,21]}),dict(maximum_new_tokens=5)):
        try:within_envelope(dict(sample,**update),limits)
        except ValueError:failed+=1
        else:raise AssertionError('Exceeded prepared envelope accepted')
    need(failed==4,'Coverage/envelope corruption must fail')
    return dict(passed=True,groups=4,whole_family_coverage=True,all_native_timing_pool=True,validated_input_envelope=True,no_model_or_head_calls=True)


def run(args,out,frozen,started):
    import torch
    from gnnformer.runtime import load_runtime
    torch.set_num_threads(4);phase='run';single_attempt_guard(out,args.arm)
    args.plan=args.plan.resolve();plan=verify_plan(args.plan)
    need(torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA B200','One B200 required')
    loaded=load_runtime(str(MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda')
    model=loaded.model.eval().requires_grad_(False);native.native_contract(model,None)
    hardware=json.loads(json.dumps(software.live_identity(torch,loaded,read(plan['native_software_plan']['file']))))
    need(hardware['gpu']=='NVIDIA B200','Native hardware identity differs');norm=model.model.language_model.norm;head=model.lm_head
    native_table=dict(norm=tensor_info(norm.weight),head=tensor_info(head.weight))
    need(native_table==dict(norm=plan['native_identity']['norm_weight'],head=plan['native_identity']['head_weight']),'Actual native head/norm differs')
    data=DATA/out.name;ckpt=CKPT/out.name;data.mkdir(parents=True,exist_ok=False);ckpt.mkdir(parents=True,exist_ok=False)
    states={}
    for state in ('fitted',):
        descriptor=plan['checkpoints'][args.arm][state];file=ckpt/f'{state}.pt';shutil.copyfile(descriptor['file'],file)
        need(sha(file)==descriptor['sha256'],'Actual deployed checkpoint copy differs')
        packet=torch.load(file,map_location='cpu',weights_only=True);states[state]=packet['branch']
        need({k:tensor_info(v) for k,v in states[state].items()}==descriptor['state'],'Actual deployed tensor identities differ')
    stats=torch.load(plan['stats_file'],map_location='cpu',weights_only=True)
    mean=stats['global_mean'].to(loaded.device);scale=stats['scale'].to(loaded.device);del stats
    stats_before=dict(global_mean=tensor_info(mean),scale=tensor_info(scale));stats_versions=(mean._version,scale._version)
    need(stats_before==plan['conditioning_identity'],'Actual fixed global statistics differ')
    core=cached.make_core(torch,args.arm).to(loaded.device);core.load_state_dict(states['fitted']);core.eval().requires_grad_(False)
    model_versions={k:v._version for k,v in model.named_parameters()};rows=read(plan['rows_file']);lookup={r['sid']:r for r in rows};prepared=read(plan['prepared_file'])
    cases=[dict(sid=r['sid'],n_frames=r['n_frames'],state='fitted') for r in rows]
    config=dict(protocol=PROTOCOL,policy=POLICY,phase=phase,arm=args.arm,run_id=out.name,slurm_job_id=os.environ['SLURM_JOB_ID'],
        source_sha256=frozen,inherited_source_sha256=plan['inherited_source_sha256'],plan_file=str(args.plan),plan_sha256=sha(args.plan),
        native_report=plan['native_report'],stage_report=plan['stage_report'],timing_certificate=plan['timing_certificate'],
        cached_report=plan['cached_report'],cached_plan=plan['cached_plan'],native_identity=plan['native_identity'],native_identity_sha256=plan['native_identity_sha256'],
        native_weight_identity=native_table,hardware=hardware,conditioning_identity=stats_before,stats_file=plan['stats_file'],stats_sha256=plan['stats_sha256'],
        checkpoints={state:dict(file=str(ckpt/f'{state}.pt'),sha256=sha(ckpt/f'{state}.pt'),state=plan['checkpoints'][args.arm][state]['state']) for state in states},
        cases=cases,data_directory=str(data),checkpoint_directory=str(ckpt),N16_only=True)
    save(out/'config.json',config);torch.cuda.synchronize();setup=time.perf_counter()-started;records=[];total=Counter()
    for number,case in enumerate(cases):
        tick=time.perf_counter();descriptor=plan['checkpoints'][args.arm][case['state']];core.load_state_dict(states[case['state']]);core.eval().requires_grad_(False)
        table={k:tensor_info(v) for k,v in core.state_dict().items()};need(table==descriptor['state'],'Selected per-trajectory state differs')
        versions={k:v._version for k,v in core.named_parameters()};item=prepared[case['sid']]
        need(sha(item['file'])==item['sha256'],'Prepared original image bundle changed');bundle=torch.load(item['file'],map_location='cpu',weights_only=True)
        runtime.validate_bundle(bundle);prep=time.perf_counter()-tick
        work=time.perf_counter();partial=[];controller=[None];native_inputs=[];native_positions=[]
        def observed_input(module,arguments,kwargs):
            cache=kwargs.get('past_key_values')
            native_inputs.append(dict(input_ids=kwargs['input_ids'].detach().cpu().clone(),
                attention_mask=kwargs['attention_mask'].detach().cpu().clone(),has_pixels=kwargs.get('pixel_values') is not None,
                past_length=0 if cache is None else cache.get_seq_length()))
        def observed_positions(module,arguments,kwargs):
            native_positions.append(kwargs['position_ids'].detach().cpu().clone())
        def observe_controller(value):controller[0]=value
        def observed_forward(module,arguments,output):
            item=dict(raw_global_logits=output.logits[-1,-1].detach().cpu().clone())
            if controller[0] is not None:item['capture']=controller[0].export_last_capture(cpu=True)
            partial.append(item)
        try:
            with ExitStack() as stack:
                stack.callback(model.register_forward_hook(observed_forward).remove)
                stack.callback(model.register_forward_pre_hook(observed_input,with_kwargs=True).remove)
                stack.callback(model.model.language_model.register_forward_pre_hook(observed_positions,with_kwargs=True).remove)
                result=runtime.generate_native(model,loaded.processor,core,bundle,interaction=args.arm,global_mean=mean,scale=scale,
                    native_identity_sha256=plan['native_identity_sha256'],max_new_tokens=4,capture=True,controller_observer=observe_controller)
        except BaseException:
            failure_file=data/f'trajectory_{number:03d}_partial.pt'
            torch.save(cpu_tree(torch,dict(case=case,prefixes=partial,profile_native=None,core_state=table,
                conditioning_identity=stats_before,native_inputs=native_inputs,native_positions=native_positions,partial_outputs_retained=True)),failure_file)
            save(out/f'trajectory_{number:03d}_failure.json',dict(case=case,file=str(failure_file),sha256=sha(failure_file),partial_outputs_retained=True))
            raise
        raw_file=data/f'trajectory_{number:03d}.pt';packet=dict(schema_version=1,case=case,result=result,profile_native=None,
            core_state=table,conditioning_identity=stats_before,native_inputs=native_inputs,native_positions=native_positions)
        torch.save(cpu_tree(torch,packet),raw_file);digest=sha(raw_file)
        save(raw_file.with_suffix('.json'),dict(case=case,raw_file=str(raw_file),raw_sha256=digest,validation_pending=True))
        need(result['metadata']['layout']==item['layout'],'Actual native positions differ from CPU-bound original layout')
        audit=basic_result(torch,result,bundle,args.arm,plan['native_identity_sha256'],case['state']);steps=len(result['generated_ids'])
        need({k:v._version for k,v in core.named_parameters()}==versions and {k:tensor_info(v) for k,v in core.state_dict().items()}==table,
             'Native trajectory changed the deployed checkpoint')
        total.update(result['counters'])
        save(out/f'progress_{number+1:03d}.json',dict(processed=number+1,case=case,raw_file=str(raw_file),raw_sha256=digest,accuracy_not_scored=True))
        work_seconds=time.perf_counter()-work
        records.append(dict(index=number,case=case,raw_file=str(raw_file),raw_sha256=digest,prepared_file=item['file'],prepared_sha256=item['sha256'],
            preprocessing_seconds=prep,work_seconds=work_seconds,four_token_seconds=prep+work_seconds*4/steps,generated_tokens=steps,audit=audit))
    need(model_versions=={k:v._version for k,v in model.named_parameters()} and all(not v.requires_grad and v.grad is None for v in model.parameters())
         and native_table==dict(norm=tensor_info(norm.weight),head=tensor_info(head.weight))
         and stats_versions==(mean._version,scale._version) and stats_before==dict(global_mean=tensor_info(mean),scale=tensor_info(scale)),
         'Native model or fixed conditioning changed')
    save(out/'raw_manifest.json',dict(completed=True,rows=records,n=len(records),all_raw_retained_before_scoring=True))
    expected_scenes=270;cap=1080
    need(len(records)==total['visual']==expected_scenes and 1<=total['model']<=cap and total['probe_head']==0,
         'Native phase call inventory exceeds its bound')
    endpoint=dict(passed=True,native_weights_before=native_table,native_weights_after=dict(norm=tensor_info(norm.weight),head=tensor_info(head.weight)),
        native_parameter_versions_unchanged=True,conditioning_before=stats_before,conditioning_after=dict(global_mean=tensor_info(mean),scale=tensor_info(scale)),
        statistics_versions_unchanged=True,all_trajectory_checkpoint_versions_unchanged=True,final_core_state={k:tensor_info(v) for k,v in core.state_dict().items()},
        counters=dict(total),no_optimizer_or_training=True)
    save(out/'endpoint.json',endpoint)
    need(time.perf_counter()-started<=POLICY['run_seconds'],'Native phase exceeded its fixed allocation cap')
    return dict(**config,passed=True,completed=True,raw_manifest_file=str(out/'raw_manifest.json'),raw_manifest_sha256=sha(out/'raw_manifest.json'),
        endpoint_file=str(out/'endpoint.json'),endpoint_sha256=sha(out/'endpoint.json'),counters=dict(total),setup_seconds=setup,
        trajectories=len(records),all_raw_retained_before_scoring=True,no_accuracy_selection=True,extra_gpu_head_calls=0,no_fitting=True)



def main():
    parser=argparse.ArgumentParser(description=__doc__);actions=parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--check',action='store_true');actions.add_argument('--run',action='store_true')
    parser.add_argument('--native-report',type=Path);parser.add_argument('--stage-report',type=Path);parser.add_argument('--plan',type=Path)
    parser.add_argument('--arm',choices=ARMS);args=parser.parse_args();native.require_slurm(gpu=not args.check)
    if args.check:
        need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS') and args.native_report is not None
             and args.stage_report is not None and args.plan is None and args.arm is None,'CPU preparation requires explicit native and rendered-stage proofs')
    else:need(args.plan is not None and args.arm in ARMS and args.native_report is args.stage_report is None,'GPU run consumes only its frozen plan and arm')
    phase='check' if args.check else 'run';suffix='' if args.check else args.arm+'_';out=OUT/f'{phase}_{suffix}{os.environ["SLURM_JOB_ID"]}'
    out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();frozen=snapshot(out)
    save(out/'request.json',dict(phase=phase,arm=args.arm,plan=None if args.plan is None else str(args.plan),
        native_report=None if args.native_report is None else str(args.native_report),stage_report=None if args.stage_report is None else str(args.stage_report),source_sha256=frozen))
    try:
        result=check(args,out,frozen) if args.check else run(args,out,frozen,started)
        elapsed=time.perf_counter()-started;need(elapsed<=POLICY['cpu_check_seconds' if args.check else 'run_seconds'],'Registered phase cap exceeded')
        need(sources()==frozen and inherited_sources()==result['inherited_source_sha256'],'Source changed during fresh evaluation')
        result['elapsed_seconds']=elapsed;save(out/'summary.json',result);print(json.dumps(dict(passed=True,directory=str(out))),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,
            elapsed_seconds=time.perf_counter()-started,partial_outputs_retained=True));raise


if __name__=='__main__':main()

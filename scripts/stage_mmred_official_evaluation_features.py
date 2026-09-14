"""Original MMReD evaluation inputs and shared vision features; no predictions."""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import harvest_mmred_official_training_features as harvest
from scripts.stage_mmred_official_recovery import need,sha,object_sha,save
compact=harvest.compact;preparation=compact.preparation;p=harvest.p;producer=harvest.producer
read=harvest.read;bind=harvest.bind
PROTOCOL='mmred_official_evaluation_shared_features'
PROPOSAL='docs/paper/NATIVE_AGGREGATION_MMRED_OFFICIAL_EVALUATION_FEATURES.md'
PROPOSAL_SHA='766196875ccecd6088248f1191769fce6344198d5001f4677b5837a1e47db581'
OWN=('scripts/stage_mmred_official_evaluation_features.py',PROPOSAL,
     'slurm/mmred_official_evaluation_prepare.sbatch','slurm/mmred_official_evaluation_harvest.sbatch')
OUT=REPO/'outputs/native_aggregation_vlm/mmred_official_evaluation_features'
DATA=Path('/mnt/data/gabriele/gnn_transformer/mmred_official_evaluation_features')
HARVEST_RELEASE=harvest.OUT/'source_release.json'
HARVEST_RELEASE_SHA='e8f77a972d25783e2def7d8fc34391635fe98c49c5640e458dbf50562acf0019'
LENGTHS=(8,16,32)
CELLS=tuple((role,n,q) for role,ns in (('val',(8,16)),('test',LENGTHS)) for n in ns for q in compact.QTYPES)
POLICY=dict(protocol=PROTOCOL,cpu_seconds=600,gpu_seconds=1800,cpu_cores=4,memory_gib=16,
    maximum_prepare_attempts=1,maximum_harvest_attempts=1,maximum_GPUs=1,
    worlds=1000,validation_worlds=400,test_worlds=600,frame_occurrences=16000,cells=20,rows_per_cell=50,
    lengths=list(LENGTHS),native_feature_rows=3136000,native_feature_width=3584,feature_dtype='torch.float16',
    raw_feature_bytes=22478848000,vision_calls=1000,model_calls=0,decoder_calls=0,head_calls=0,
    prefix_vision_worlds=3,prefix_lengths=list(LENGTHS),no_pixels_saved=True,no_adapters_or_training=True,
    no_predictions=True,no_training_or_inference_release=True,no_N64_N128=True)


def sources():
    need(sha(REPO/PROPOSAL)==PROPOSAL_SHA,'Evaluation input proposal changed')
    return {name:sha(REPO/name) for name in OWN}


def inherited_sources():
    need(sha(HARVEST_RELEASE)==HARVEST_RELEASE_SHA,'Frozen shared-feature source release changed')
    release=read(HARVEST_RELEASE);result={**release['inherited_source_sha256'],**release['source_sha256']}
    need(result=={**harvest.inherited_sources(),**harvest.sources()} and len(result)==209
         and all(sha(REPO/name)==digest for name,digest in result.items()),'Frozen evaluation input dependencies changed')
    return result


def check_time(started,phase):
    need(time.perf_counter()-started<POLICY['cpu_seconds' if phase=='prepare' else 'gpu_seconds'],
         'Fixed evaluation input allocation cap exceeded')


def selection(rows):
    selected=[r for r in rows if r['pilot_role'] in ('val','test')];counts=Counter();assignments=[]
    for index,row in enumerate(selected):
        key=(row['pilot_role'],row['n'],row['qtype']);ordinal=counts[key];counts[key]+=1
        assignments.append(dict(index=index,sid=row['sid'],pilot_role=row['pilot_role'],n=row['n'],qtype=row['qtype'],cell_index=ordinal))
    need(len(selected)==len({r['sid'] for r in selected})==1000 and counts==Counter({k:50 for k in CELLS})
         and sum(r['n'] for r in selected)==16000,'Exact original20x50 validation/test cohort required')
    prefix=[next(r['index'] for r in assignments if r['n']==n) for n in LENGTHS]
    need(len(set(prefix))==3,'Three distinct first-source-order length probes required')
    return selected,assignments,prefix


def envelope(records,references):
    uncovered=[];maxima=[]
    for record in records:
        candidates=[r for r in references if r['n']==record['n']]
        if not any(all(record[k]<=r[k] for k in compact.WIDTHS) for r in candidates):
            uncovered.append(dict(index=record['index'],sid=record['sid'],n=record['n'],
                same_length_reference_available=bool(candidates),widths={k:record[k] for k in compact.WIDTHS},
                excess_above_component_max=None if not candidates else
                {k:max(0,record[k]-max(r[k] for r in candidates)) for k in compact.WIDTHS}))
    for n in LENGTHS:
        group=[r for r in records if r['n']==n]
        for key in compact.WIDTHS:
            value=max(r[key] for r in group);owner=next(r for r in group if r[key]==value)
            maxima.append(dict(n=n,metric=key,value=value,index=owner['index'],sid=owner['sid']))
    return dict(reference_cases=references,metrics=list(compact.WIDTHS),uncovered=uncovered,
        uncovered_count=len(uncovered),first_tie_maxima=maxima,requires_one_jointly_covering_reference=True,
        N32_decoder_envelope_validated=False,no_inference_release=True,prepared_shapes_only=True)


def cell_timings(records):
    groups=[]
    for role,n,q in CELLS:
        group=[r for r in records if (r['pilot_role'],r['n'],r['qtype'])==(role,n,q)]
        need(len(group)==50,'Actual role/length/task timing cell differs')
        groups.append(dict(pilot_role=role,n=n,qtype=q,worlds=len(group),
            staging_seconds_max=max(r['staging_seconds'] for r in group),staging_seconds_sum=sum(r['staging_seconds'] for r in group)))
    return dict(cells=groups,compact_bytes=sum(r['bytes'] for r in records),omitted_pixel_bytes=sum(r['pixel_bytes'] for r in records))


def resource_projection(records,original,setup,*,completed=(),elapsed=None):
    F={n:max(r['seconds'] for r in original['feature_records'] if r['n']==n) for n in (8,16)}
    F[32]=2*F[16];initial=dict(F);done={r['index'] for r in completed}
    for record in completed:F[record['n']]=max(F[record['n']],record['feature_phase_seconds'])
    cells=[];work=0.
    for cell in cell_timings(records)['cells']:
        group=[r for r in records if (r['pilot_role'],r['n'],r['qtype'])==(cell['pilot_role'],cell['n'],cell['qtype'])]
        count=sum(r['index'] not in done for r in group);unit=cell['staging_seconds_max']+F[cell['n']]
        work+=count*unit;cells.append(dict(cell,remaining_worlds=count,feature_seconds=F[cell['n']],projected_work_seconds=count*unit))
    base=max(original['setup_seconds'],setup) if elapsed is None else elapsed
    seconds=base+1.25*work+60
    need(all(math.isfinite(v) and v>=0 for v in (base,work,seconds,*F.values())),'Finite nonnegative resource estimate required')
    return dict(passed=seconds<=1800,projected_seconds=seconds,cap_seconds=1800,
        setup_or_elapsed_seconds=base,elapsed_at_probe_gate=elapsed,remaining_worlds=1000-len(done),
        cells=cells,initial_feature_seconds_by_n={str(k):v for k,v in initial.items()},feature_seconds_by_n={str(k):v for k,v in F.items()},
        completed_indices=[r['index'] for r in completed],N32_initial_estimate='2 * original measured N16 maximum',
        N32_vision_measured=any(r['n']==32 for r in completed),empirical_estimate_not_guarantee=True,
        deliberate_conservative_IO_overlap=True,no_inference_release=True)


def make_plan(phase,bindings,out,**extra):
    artifacts={str(f):sha(f) for f in out.iterdir() if f.is_file() and f.name not in ('plan.json','summary.json','failure.json')}
    return dict(protocol=PROTOCOL,phase=phase,policy=POLICY,source_sha256=sources(),inherited_source_sha256=inherited_sources(),
        input_bindings=bindings,artifacts=artifacts,no_predictions=True,no_training_or_inference_release=True,**extra)


def prepare(out,data,started):
    bindings={};parent,_,_,_=compact.dependencies(bindings)
    bind(HARVEST_RELEASE,bindings,HARVEST_RELEASE_SHA)
    config,original,audit_ref=harvest.profile_metadata(bindings,full_audit=True)
    need(config['native_identity']==parent['native_identity'] and config['native_identity_sha256']==parent['native_identity_sha256'],
         'Evaluation preparation/native feature identities differ')
    rows,assigned,prefix=selection(read(parent['rows_file']))
    import torch
    from transformers import AutoProcessor,__version__ as tf_version
    from scripts.stage_native_vision_v6_teacher import model_metadata
    torch.set_num_threads(4)
    processor=AutoProcessor.from_pretrained(str(preparation.MODEL),trust_remote_code=True,use_fast=False,local_files_only=True)
    owner,fn,api=p.backend.native_api(processor);rope=lambda **kw:fn(owner,**kw);identity=parent['native_identity']
    need(api==identity['native_api'] and p.fingerprint(processor,str(tf_version))==identity['processor']
         and p.backend.runtime_identity()==identity['runtime'] and model_metadata()==identity['model'],'Pinned CPU native environment differs')
    references,_=compact.reference_cases(torch,p,parent);setup=time.perf_counter()-started;records=[]
    with (out/'cases.jsonl').open('x') as stream:
        for assignment,row in zip(assigned,rows):
            check_time(started,'prepare');tick=time.perf_counter()
            case=preparation.prepare_joint(processor,row,parent['system_prompt'],rope);prepared=time.perf_counter()-tick
            identity_tick=time.perf_counter();pixels=case['inputs']['pixel_values'];pixel_info=p.tensor_info(pixels)
            identity_seconds=time.perf_counter()-identity_tick
            full=compact.tensor_tree(torch,p,case);expected=dict(full);expected['inputs']=dict(full['inputs']);del expected['inputs']['pixel_values']
            del case['inputs']['pixel_values'];need(compact.tensor_tree(torch,p,case)==expected,'Compaction changed more than pixels')
            file=data/f'case_{assignment["index"]:04d}.pt';torch.save(dict(case=case,omitted_pixel_values=pixel_info),file)
            restored=torch.load(file,map_location='cpu',weights_only=True)
            need(set(restored)=={'case','omitted_pixel_values'} and restored['omitted_pixel_values']==pixel_info
                 and compact.tensor_tree(torch,p,restored['case'])==expected,'Compact packet roundtrip differs')
            target=preparation.canonical_target(row['atype'],row['answer'])
            need(case['target_text']==target and case['target_ids']==parent['targets']['targets'][target]['ids'],
                 'Original validation/test target tokenization differs')
            descriptor=dict(assignment,file=str(file),sha256=sha(file),bytes=file.stat().st_size,
                raw_row_sha256=row['raw_row_sha256'],metadata_sha256=object_sha(case['metadata']),compact_identity_sha256=object_sha(expected),
                omitted_pixel_values=pixel_info,grid_identity=p.tensor_info(case['inputs']['image_grid_thw']),
                pixel_bytes=pixels.numel()*pixels.element_size(),target_text=target,target_ids=case['target_ids'],**compact.widths(case),
                prepare_seconds=prepared,pixel_identity_seconds=identity_seconds)
            descriptor['staging_seconds']=time.perf_counter()-tick
            stream.write(json.dumps(descriptor,sort_keys=True,allow_nan=False)+'\n');stream.flush();records.append(descriptor)
            del case,pixels,restored,full,expected
    projection=resource_projection(records,original,setup)
    for name,value in (('rows',rows),('cases',records),('target_inventory',compact.target_inventory(records)),
                       ('envelopes',envelope(records,references)),('timings',cell_timings(records)),('projection',projection),
                       ('harvest_order',prefix+[r['index'] for r in records if r['index'] not in prefix])):save(out/f'{name}.json',value)
    (out/'REPORT.md').write_text('# Original MMReD evaluation inputs\n\nExactly400 validation and600 test inputs are prepared. No predictions were made. N32 decoder feasibility remains unvalidated. The resource projection separately controls feature-harvest eligibility.\n')
    plan=make_plan('prepare',bindings,out,rows_file=str(out/'rows.json'),cases_file=str(out/'cases.json'),
        target_inventory_file=str(out/'target_inventory.json'),envelopes_file=str(out/'envelopes.json'),timings_file=str(out/'timings.json'),
        projection_file=str(out/'projection.json'),projection=projection,resource_eligible=projection['passed'],
        harvest_order_file=str(out/'harvest_order.json'),prefix_indices=prefix,setup_seconds=setup,
        native_identity=parent['native_identity'],native_identity_sha256=parent['native_identity_sha256'],
        precision=parent['precision'],packages=parent['packages'],system_prompt=parent['system_prompt'],profile_hardware=config['hardware'],
        preparation=dict(file=str(compact.PARENT),sha256=compact.PARENT_SHA,plan_sha256=compact.PARENT_PLAN_SHA),
        render=dict(file=str(compact.RENDER),sha256=compact.RENDER_SHA,plan_sha256=compact.RENDER_PLAN_SHA),
        independent_profile_audit=audit_ref,original_profile_passed=False,cold_training_components_passed=True)
    save(out/'plan.json',plan)
    return dict(plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),worlds=1000,frames=16000,resource_eligible=projection['passed'])


def verify_report(path,phase):
    path=Path(path).resolve();summary=read(path)
    need(summary['passed'] is summary['completed'] is True and summary['protocol']==PROTOCOL and summary['phase']==phase
         and summary['policy']==POLICY and summary['source_sha256']==sources() and summary['inherited_source_sha256']==inherited_sources()
         and not (path.parent/'failure.json').exists(),'Passed source-bound evaluation input report required')
    need(0<summary['elapsed_seconds']<=POLICY['cpu_seconds' if phase=='prepare' else 'gpu_seconds']
         and sha(summary['plan_file'])==summary['plan_sha256'],'Evaluation input report plan/time differs')
    plan=read(summary['plan_file'])
    need(plan['protocol']==PROTOCOL and plan['phase']==phase and plan['policy']==POLICY
         and plan['source_sha256']==sources() and plan['inherited_source_sha256']==inherited_sources()
         and plan['no_predictions'] is plan['no_training_or_inference_release'] is True,'Evaluation input policy/source differs')
    for mapping in (plan['input_bindings'],plan['artifacts']):
        for file,digest in mapping.items():need(sha(file)==digest,'Bound evaluation input metadata changed')
    for name,digest in {**sources(),**inherited_sources()}.items():
        need(sha(path.parent/'source'/name.replace('/','_'))==digest,'Archived evaluation input source changed')
    return plan,dict(file=str(path),sha256=sha(path),plan_file=summary['plan_file'],plan_sha256=summary['plan_sha256'])


def verify_preparation(path,streaming=False):
    plan,_=verify_report(path,'prepare');rows=read(plan['rows_file']);records=read(plan['cases_file'])
    selected,assigned,prefix=selection(rows)
    need(selected==rows and len(records)==1000 and [{k:r[k] for k in assigned[0]} for r in records]==assigned
         and plan['prefix_indices']==prefix and read(plan['harvest_order_file'])==prefix+[r['index'] for r in records if r['index'] not in prefix],
         'Evaluation source cohort/order/prefix differs')
    for record,row in zip(records,rows):
        need(record['raw_row_sha256']==row['raw_row_sha256'] and record['target_text']==preparation.canonical_target(row['atype'],row['answer'])
             and record['omitted_pixel_values']['shape']==[row['n']*784,1176] and record['omitted_pixel_values']['dtype']=='torch.float32'
             and record['grid_identity']['shape']==[row['n'],3] and record['grid_identity']['dtype']=='torch.int64'
             and record['target_rows']==len(record['target_ids']) and record['target_ids'][-1]==151645
             and 0<record['prepare_seconds']<=record['staging_seconds'],'Evaluation compact ownership/shape/timing differs')
        if not streaming:
            need(sha(record['file'])==record['sha256'] and Path(record['file']).stat().st_size==record['bytes'],'Evaluation compact payload changed')
    config,original,audit_ref=harvest.profile_metadata({})
    need(plan['native_identity']==config['native_identity'] and plan['native_identity_sha256']==config['native_identity_sha256']
         and plan['profile_hardware']==config['hardware'] and plan['independent_profile_audit']==audit_ref
         and plan['original_profile_passed'] is False and plan['cold_training_components_passed'] is True,'Evaluation native/audit reference differs')
    expected=resource_projection(records,original,plan['setup_seconds'])
    need(expected==plan['projection']==read(plan['projection_file']) and plan['resource_eligible']==expected['passed']
         and read(plan['timings_file'])==cell_timings(records) and read(plan['target_inventory_file'])==compact.target_inventory(records),
         'CPU target/timing/resource inventory differs')
    saved_envelope=read(plan['envelopes_file'])
    need(saved_envelope==envelope(records,saved_envelope['reference_cases']),'Evaluation envelope inventory differs')
    return plan


def extract_stage(path,out,data,started,progress):
    path=Path(path).resolve();plan=verify_preparation(path,streaming=True)
    need(plan['resource_eligible'],'Initial CPU projection exceeds1800s; feature worker held')
    summary=read(path);compact_ref=dict(file=str(path),sha256=sha(path),plan_file=summary['plan_file'],plan_sha256=summary['plan_sha256'])
    records=read(plan['cases_file']);rows=read(plan['rows_file']);order=read(plan['harvest_order_file'])
    _,original,_=harvest.profile_metadata({})
    import torch
    from gnnformer.runtime import load_runtime
    torch.set_num_threads(4)
    need(torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA B200','One B200 required')
    loaded=load_runtime(str(preparation.MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda')
    model=loaded.model.eval().requires_grad_(False);hardware=p.frozen_joint.live_identity(torch,loaded,plan)
    need(hardware==plan['profile_hardware'] and p.installed({})==plan['packages'],'Original native hardware/packages required')
    refs=list(model.named_parameters());base=p.base_metadata(refs);p.check_base(refs,base,model)
    need(not p.adapter_parameters(model) and not any(v.training for v in model.modules()),'Frozen eval base without adapters required')
    owner,fn,api=p.backend.native_api(loaded.processor);rope=lambda **kw:fn(owner,**kw)
    need(api==plan['native_identity']['native_api'],'Native image position API changed')
    counts=Counter(harvest.counters_expected(0));features=[];progress.update(counters=counts,features=features)
    save(out/'base_before.json',base);save(out/'hardware.json',hardware);setup=time.perf_counter()-started;gate=None
    with ExitStack() as hooks,(out/'features.jsonl').open('x') as stream:
        def count(key):
            def callback(*_):
                counts[key]+=1;need(key=='vision','Feature stage unexpectedly called model/backbone/language/norm/head')
            return callback
        for module,key in ((model,'model'),(model.model,'backbone'),(model.model.language_model,'language'),
                           (model.model.language_model.norm,'norm'),(model.lm_head,'head'),(model.model.visual,'vision')):
            hooks.callback(module.register_forward_pre_hook(count(key)).remove)
        for execution_index,idx in enumerate(order):
            check_time(started,'harvest');torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
            record=records[idx];row=rows[idx]
            need(row['sid']==record['sid'] and row['raw_row_sha256']==record['raw_row_sha256'] and sha(record['file'])==record['sha256'],
                 'Consumed compact/source ownership differs')
            packet=torch.load(record['file'],map_location='cpu',weights_only=True)
            need(set(packet)=={'case','omitted_pixel_values'} and packet['omitted_pixel_values']==record['omitted_pixel_values']
                 and object_sha(compact.tensor_tree(torch,p,packet['case']))==record['compact_identity_sha256'],'Compact packet differs')
            loaded_at=time.perf_counter();case=preparation.prepare_joint(loaded.processor,row,plan['system_prompt'],rope);prepared_at=time.perf_counter()
            pixels=case['inputs']['pixel_values'];pixel_info=p.tensor_info(pixels);grid_info=p.tensor_info(case['inputs']['image_grid_thw'])
            need(pixel_info==packet['omitted_pixel_values'] and grid_info==record['grid_identity'],'Reprocessed pixel/grid identity differs')
            del case['inputs']['pixel_values']
            need(compact.tensor_tree(torch,p,case)==compact.tensor_tree(torch,p,packet['case'])
                 and object_sha(compact.tensor_tree(torch,p,case))==record['compact_identity_sha256'],'Reprocessed full compact packet differs')
            coordinates={k:p.tensor_info(v) for k,v in case['coordinates'].items()};proof_at=time.perf_counter()
            torch.cuda.synchronize();vision_start=time.perf_counter();before=counts['vision']
            with torch.inference_mode():value=producer.extract_features(torch,model,pixels,case['inputs']['image_grid_thw'])
            torch.cuda.synchronize();vision_end=time.perf_counter()
            need(counts['vision']==before+1 and value.dtype==torch.float16 and value.shape==(row['n']*196,3584)
                 and bool(value.isfinite().all()),'One finite native FP16 feature collection required')
            tensor=p.tensor_info(value);input_identity=dict(pixel_values=pixel_info,image_grid_thw=grid_info)
            reference=dict(file=record['file'],sha256=record['sha256'],compact_identity_sha256=record['compact_identity_sha256'])
            ownership=dict(index=idx,sid=row['sid'],n=row['n'],input_identity=input_identity,compact=reference,
                coordinate_identity=coordinates,native_identity_sha256=plan['native_identity_sha256'])
            file=data/f'features_{idx:04d}.pt';torch.save(dict(features=value.cpu(),**ownership),file)
            descriptor=dict(ownership,pilot_role=row['pilot_role'],qtype=row['qtype'],execution_index=execution_index,
                file=str(file),sha256=sha(file),bytes=file.stat().st_size,tensor=tensor,finite=True,raw_feature_bytes=value.numel()*value.element_size())
            p.check_base(refs,base,model);check_time(started,'harvest');torch.cuda.synchronize()
            descriptor.update(compact_load_seconds=loaded_at-tick,full_reprocess_seconds=prepared_at-loaded_at,
                input_proof_seconds=proof_at-prepared_at,vision_seconds=vision_end-vision_start,
                max_allocated_bytes=torch.cuda.max_memory_allocated(),max_reserved_bytes=torch.cuda.max_memory_reserved())
            stream.write(json.dumps(descriptor,sort_keys=True,allow_nan=False)+'\n');stream.flush()
            end=time.perf_counter();descriptor.update(output_and_audit_seconds=end-vision_end,
                feature_phase_seconds=end-proof_at,full_world_seconds=end-tick);features.append(descriptor)
            progress['completed_worlds']=len(features)
            del value,pixels,packet,case
            if execution_index==2:
                elapsed=time.perf_counter()-started
                gate=resource_projection(records,original,setup,completed=features,elapsed=elapsed)
                save(out/'prefix_features.json',features);save(out/'prefix_projection.json',gate)
                need([r['index'] for r in features]==plan['prefix_indices'] and gate['N32_vision_measured']
                     and gate['passed'],'Measured three-case prefix holds remaining feature work above1800s')
    need(gate is not None and gate['passed'] and dict(counts)==harvest.counters_expected(1000)
         and sum(r['raw_feature_bytes'] for r in features)==POLICY['raw_feature_bytes'],'Complete feature-only work/byte inventory differs')
    p.check_base(refs,base,model);save(out/'base_after.json',p.base_metadata(refs));need(read(out/'base_before.json')==read(out/'base_after.json'),'Base changed')
    features.sort(key=lambda r:r['index']);save(out/'feature_index.json',features);save(out/'counters.json',dict(counts))
    save(out/'timings.json',dict(setup_seconds=setup,worlds=1000,vision_calls=1000,raw_feature_bytes=POLICY['raw_feature_bytes'],
        full_world_seconds=sum(r['full_world_seconds'] for r in features),vision_seconds=sum(r['vision_seconds'] for r in features),
        full_reprocess_seconds=sum(r['full_reprocess_seconds'] for r in features),saved_bytes=sum(r['bytes'] for r in features),
        peak_allocated_bytes=max(r['max_allocated_bytes'] for r in features),peak_reserved_bytes=max(r['max_reserved_bytes'] for r in features)))
    (out/'REPORT.md').write_text('# Shared original MMReD evaluation features\n\nExactly400 validation and600 test worlds have bound native FP16 features. There were1,000 vision calls and zero model/decoder/head calls. The three measured prefix cases are included in these outputs. No predictions or N32 decoder feasibility claim is made.\n')
    bindings={str(path):sha(path),summary['plan_file']:summary['plan_sha256']}
    for key in ('rows_file','cases_file','target_inventory_file','envelopes_file','harvest_order_file'):bind(plan[key],bindings)
    result=make_plan('harvest',bindings,out,compact_stage=compact_ref,rows_file=plan['rows_file'],cases_file=plan['cases_file'],
        feature_index_file=str(out/'feature_index.json'),compact_envelopes_file=plan['envelopes_file'],target_inventory_file=plan['target_inventory_file'],
        timings_file=str(out/'timings.json'),prefix_features_file=str(out/'prefix_features.json'),prefix_projection_file=str(out/'prefix_projection.json'),
        counters=dict(counts),projection=gate,original_projection=plan['projection'],native_identity=plan['native_identity'],
        native_identity_sha256=plan['native_identity_sha256'],precision=plan['precision'],packages=plan['packages'],system_prompt=plan['system_prompt'],
        independent_profile_audit=plan['independent_profile_audit'],original_profile_passed=False,cold_training_components_passed=True,
        all_features_finite_native_FP16=True,feature_file_hashes_are_in_bound_index=True,no_merge_tensor_load_or_copy=True)
    save(out/'plan.json',result)
    return dict(plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),worlds=1000,frames=16000,vision_calls=1000)


def verify_stage(path,streaming=False):
    plan,_=verify_report(path,'harvest');reference=plan['compact_stage'];prepared=verify_preparation(reference['file'],streaming=streaming)
    summary=read(reference['file'])
    need(reference==dict(file=str(Path(reference['file']).resolve()),sha256=sha(reference['file']),
        plan_file=summary['plan_file'],plan_sha256=summary['plan_sha256']),'Feature compact-stage descriptor differs')
    records=read(prepared['cases_file']);features=read(plan['feature_index_file']);order=read(prepared['harvest_order_file'])
    need(len(features)==1000 and [r['index'] for r in features]==list(range(1000))
         and plan['counters']==harvest.counters_expected(1000) and plan['all_features_finite_native_FP16'] is True,
         'Complete ordered original evaluation feature cache required')
    for feature,record in zip(features,records):
        need(all(feature[k]==record[k] for k in ('index','sid','n','qtype','pilot_role'))
             and order[feature['execution_index']]==record['index'] and feature['finite'] is True
             and feature['tensor']['shape']==[record['n']*196,3584] and feature['tensor']['dtype']=='torch.float16'
             and feature['raw_feature_bytes']==record['n']*196*3584*2 and feature['bytes']>=feature['raw_feature_bytes']
             and feature['input_identity']==dict(pixel_values=record['omitted_pixel_values'],image_grid_thw=record['grid_identity'])
             and feature['compact']==dict(file=record['file'],sha256=record['sha256'],compact_identity_sha256=record['compact_identity_sha256'])
             and feature['native_identity_sha256']==prepared['native_identity_sha256']
             and 0<feature['vision_seconds']<=feature['feature_phase_seconds']<=feature['full_world_seconds'],
             'Feature shape/input/call/timing ownership differs')
        if not streaming:
            need(sha(feature['file'])==feature['sha256'] and Path(feature['file']).stat().st_size==feature['bytes'],'Feature payload changed')
    prefix=[features[i] for i in prepared['prefix_indices']];_,original,_=harvest.profile_metadata({})
    gate=read(plan['prefix_projection_file'])
    expected=resource_projection(records,original,read(plan['timings_file'])['setup_seconds'],completed=prefix,elapsed=gate['elapsed_at_probe_gate'])
    need(expected==gate==plan['projection'] and gate['passed'] and read(plan['prefix_features_file'])==prefix
         and plan['original_projection']==prepared['projection'] and prepared['resource_eligible']
         and plan['native_identity']==prepared['native_identity'] and plan['native_identity_sha256']==prepared['native_identity_sha256']
         and plan['independent_profile_audit']==prepared['independent_profile_audit']
         and plan['original_profile_passed'] is False and plan['cold_training_components_passed'] is True
         and sum(r['raw_feature_bytes'] for r in features)==POLICY['raw_feature_bytes'],'Feature resource/provenance/inventory differs')
    return plan


def main():
    ap=argparse.ArgumentParser(description=__doc__);group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true');group.add_argument('--harvest',action='store_true')
    ap.add_argument('--compact',type=Path);args=ap.parse_args();phase='prepare' if args.prepare else 'harvest';started=time.perf_counter()
    need(os.environ.get('SLURM_JOB_ID') and int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==4,'Four-core Slurm only')
    if args.prepare:need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS') and args.compact is None,'CPU-only preparation')
    else:
        p.native.require_slurm(gpu=True);need(args.compact is not None and not os.environ.get('SLURM_ARRAY_JOB_ID'),'One non-array feature worker requires compact summary')
    need(not any(OUT.glob(phase+'_*')),'One allocation attempt per phase; no automatic retry')
    tag=f'{phase}_{os.environ["SLURM_JOB_ID"]}';out=OUT/tag;out.mkdir(parents=True,exist_ok=False)
    frozen=sources();inherited=inherited_sources();(out/'source').mkdir()
    for name,digest in {**frozen,**inherited}.items():
        file=out/'source'/name.replace('/','_');file.write_bytes((REPO/name).read_bytes());need(sha(file)==digest,'Source snapshot changed')
    save(out/'request.json',dict(phase=phase,compact=None if args.compact is None else str(args.compact.resolve()),policy=POLICY,
        source_sha256=frozen,inherited_source_sha256=inherited));progress={}
    try:
        data=DATA/tag;data.mkdir(parents=True,exist_ok=False)
        value=prepare(out,data,started) if args.prepare else extract_stage(args.compact,out,data,started,progress)
        check_time(started,phase);need(sources()==frozen and inherited_sources()==inherited,'Source closure changed during stage')
        save(out/'summary.json',dict(passed=True,completed=True,protocol=PROTOCOL,phase=phase,policy=POLICY,
            source_sha256=frozen,inherited_source_sha256=inherited,elapsed_seconds=time.perf_counter()-started,
            no_predictions=True,no_training_or_inference_release=True,**value))
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),phase=phase,progress=progress,
            source_sha256=frozen,elapsed_seconds=time.perf_counter()-started,partial_outputs_retained=True,
            no_predictions=True,no_automatic_retry=True));raise


if __name__=='__main__':main()

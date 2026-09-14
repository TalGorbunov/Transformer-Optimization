"""CPU-only ordinary-joint preparation of unused seed91726342 confirmation.

Freeze all810 metadata/order; materialize only270 N16 inputs. No checkpoint,
backbone/head, prediction, fitting or GPU release is part of this stage.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import math
import os
from pathlib import Path
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import diagnose_native_identity_join_joint_lora_v2 as p
from scripts import report_native_identity_join_joint_lora as audit
from scripts import stage_native_identity_join_factor_orientation_confirmation as stage
from scripts.stage_native_vision_v6_teacher import MODEL,need,read,save,sha,object_sha
OUT=REPO/'outputs/native_aggregation_vlm/identity_join_joint_lora_confirmation/preparation'
DATA=Path('/mnt/data/gabriele/gnn_transformer/identity_join_joint_lora_confirmation/preparation')
PROTOCOL='identity_join_joint_lora_confirmation_preparation'
PROPOSAL='docs/paper/NATIVE_AGGREGATION_JOINT_LORA_CONFIRMATION_PROPOSAL.md'
PROPOSAL_SHA='d0aa05f2e390764c9fa26ff12fe4923906394131f48ae219839d9499d44659c2'
DESIGN='docs/paper/NATIVE_AGGREGATION_JOINT_LORA_CONFIRMATION_DESIGN.md'
DESIGN_SHA='9c285064ae82096ee0b1d9efb6321ab5fa4ba8f18af5c64db9fa72e8ca1bde94'
PROFILE_AUDIT=REPO/'outputs/native_aggregation_vlm/identity_join_joint_lora_audit/profile_443909/summary.json'
PROFILE_AUDIT_SHA='07dcb9b2dda080e17a6c182f4f3328016aea3bd7103993c111a317c697d43b0b'
PROFILE=REPO/'outputs/native_aggregation_vlm/identity_join_joint_lora_v2/profile_443890/summary.json'
PROFILE_SHA='5c75b6919698a8b9832fc6a4bd75106d4937610832999aa1463d497ddf43705a'
STAGE_REPORT=REPO/'outputs/native_aggregation_vlm/identity_join_factor_orientation_confirmation/data_staging/render_443666/summary.json'
MANIFEST_SHA='279193c4f6b9861aa4bca1ee88b011241a87c57705ddf9e38347c59f2df80544'
OWN=('scripts/stage_native_identity_join_joint_lora_confirmation.py',PROPOSAL,
     'slurm/native_identity_join_joint_lora_confirmation_check.sbatch')
POLICY=dict(protocol=PROTOCOL,seed=91726342,lengths=[16,32,64],metadata_contexts=810,prepared_contexts=270,
    prepared_length=16,deferred_contexts=540,panels={'A':108,'B':108,'C':54},families={'A':36,'B':36,'C':18},
    whole_threshold={'A':98,'B':98,'C':49},family_threshold={'A':33,'B':33,'C':16},
    maximum_new_tokens=4,target_eos=151645,resize=392,cpu_seconds=600,cpu_cores=4,memory_gib=16,
    ordinary_joint_images_first=True,no_prompt_wrapper=True,no_broadcast=True,
    no_model_or_head_calls=True,no_checkpoint_load=True,no_new_statistics=True,no_fitting=True,
    confirmation_predictions_accessed=False,no_inference_release=True,longer_feasibility_established=False)


def bind(path,bindings,expected=None):return p.bind(path,bindings,expected)


def sources():
    need(sha(REPO/PROPOSAL)==PROPOSAL_SHA and sha(REPO/DESIGN)==DESIGN_SHA,'Frozen confirmation protocol changed')
    return {name:sha(REPO/name) for name in OWN}


def inherited_sources():
    result={}
    for group in (audit.sources(),audit.inherited_sources(),stage.source_hashes(),{DESIGN:DESIGN_SHA}):
        for name,digest in group.items():
            need(name not in result or result[name]==digest,'Conflicting confirmation ancestor');result[name]=digest
    need(all(sha(REPO/name)==digest for name,digest in result.items()),'Confirmation ancestor changed')
    return result


def snapshot(out):
    own=sources();(out/'source').mkdir()
    for name,digest in own.items():
        file=out/'source'/name.replace('/','_');file.write_bytes((REPO/name).read_bytes());need(sha(file)==digest,'Source copy changed')
    save(out/'source_hashes.json',own);save(out/'inherited_sources.json',dict(source_sha256=inherited_sources()));return own


def cohort(rows,n):
    need(len(rows)==len({r['sid'] for r in rows})==270 and all(r['n_frames']==n and r['seed']==91726342 for r in rows)
         and Counter(r['panel'] for r in rows)==POLICY['panels'],'Exact270-context confirmation cohort required')
    groups=defaultdict(list)
    for row in rows:
        need(row['split']=='fresh_'+row['panel'] and row['question']==row['local_prompt'] and len(row['image_files'])==n
             and row['target_ids'][-1]==151645 and 2<=len(row['target_ids'])<=4,'Frozen sample/input/target metadata differs')
        groups[(row['panel'],row['contrast_id'])].append(row)
    need(Counter(panel for panel,_ in groups)==POLICY['families'] and all(len(g)==3 and {r['variant'] for r in g}=={0,1,2}
         and len({r['family_instance_sha256'] for r in g})==1 and len({r['orientation'] for r in g})==1 for g in groups.values()),
         'Complete three-variant family ownership differs')
    need(Counter(r['matches_reference_orientation'] for r in rows if r['panel']=='A')=={True:54,False:54}
         and all(r['matches_reference_orientation'] is None and r['reference_orientation'] is None for r in rows if r['panel'] in ('B','C')),
         'Reference orientation metadata differs; both orientations were trained')
    return dict(contexts=270,panels=dict(Counter(r['panel'] for r in rows)),families=dict(Counter(panel for panel,_ in groups)))


def selected_rows(manifest):
    result=[];partitions={}
    for n in POLICY['lengths']:
        rows=[r for panel in ('A','B','C') for r in manifest['splits'][f'fresh_{panel}_N{n}']['samples']]
        cohort(rows,n);partitions[str(n)]=[r['sid'] for r in rows];result.extend(rows)
    need(len(result)==len({r['sid'] for r in result})==810,'Exact810 unique frozen scenes required')
    return result,dict(N16=partitions['16'],longer=partitions['32']+partitions['64'],by_length=partitions,
        N16_contexts=270,longer_contexts=540,order='N16,N32,N64; A,B,C; original manifest order within each cell')


def criterion(rows):
    n=rows[0]['n_frames'];cohort(rows,n);panels={}
    for panel in ('A','B','C'):
        values=[r for r in rows if r['panel']==panel];groups=defaultdict(list)
        for row in values:groups[row['contrast_id']].append(row)
        correct=sum(r['exact'] for r in values);complete=sum(all(r['exact'] for r in g) for g in groups.values())
        panels[panel]=dict(passed=correct>=POLICY['whole_threshold'][panel] and complete>=POLICY['family_threshold'][panel],
            whole_correct=correct,contexts=len(values),complete_families=complete,families=len(groups),
            thresholds=dict(whole_correct=POLICY['whole_threshold'][panel],complete_families=POLICY['family_threshold'][panel]))
    return dict(n_frames=n,panels=panels,qualifies_A_and_B=all(panels[k]['passed'] for k in ('A','B')),
        C_separate=panels['C']['passed'],natural_whole_answer_and_eos=True)


def envelope(torch,inputs,positions,deltas):
    need(set(inputs)=={'input_ids','attention_mask','pixel_values','image_grid_thw'} and inputs['input_ids'].shape[0]==1
         and torch.equal(inputs['attention_mask'],torch.ones_like(inputs['input_ids']))
         and inputs['image_grid_thw'].shape==(16,3) and positions.shape==(3,1,inputs['input_ids'].shape[1]) and deltas.shape==(1,1),
         'One unpadded ordinary N16 prompt required')
    return dict(prompt_width=inputs['input_ids'].shape[1],input_shapes={k:list(v.shape) for k,v in inputs.items()},
        input_dtypes={k:str(v.dtype) for k,v in inputs.items()},image_grid_thw=inputs['image_grid_thw'].tolist(),
        position_dtype=str(positions.dtype),delta_dtype=str(deltas.dtype),maximum_absolute_position=int(positions.abs().max()),
        maximum_absolute_rope_delta=int(deltas.abs().max()),rows=1,n_frames=16,resize=392,maximum_new_tokens=4)


def envelope_comparison(value,reference):
    exact=('input_dtypes','image_grid_thw','position_dtype','delta_dtype','rows','n_frames','resize','maximum_new_tokens')
    scalar=('prompt_width','maximum_absolute_position','maximum_absolute_rope_delta')
    violations=[k for k in exact if value[k]!=reference[k]]+[k for k in scalar if value[k]>reference[k]]
    for key,shape in reference['input_shapes'].items():
        actual=value['input_shapes'][key]
        if len(actual)!=len(shape) or any(a>b for a,b in zip(actual,shape)):violations.append('input_shapes.'+key)
    return dict(passed=not violations,violations=violations)


def dependencies(bindings):
    bind(PROFILE_AUDIT,bindings,PROFILE_AUDIT_SHA);bind(PROFILE,bindings,PROFILE_SHA)
    evidence=audit.verify_profile_audit(PROFILE_AUDIT,PROFILE);proof=read(PROFILE_AUDIT)
    for key in ('analysis','input_bindings','artifacts','report'):bind(proof[key+'_file'],bindings,proof[key+'_sha256'])
    for file,digest in read(proof['input_bindings_file']).items():bind(file,bindings,digest)
    for file,digest in read(proof['artifacts_file']).items():bind(file,bindings,digest)
    profile=read(PROFILE);bind(profile['plan_file'],bindings,profile['plan_sha256']);parent=read(profile['plan_file'])
    bind(STAGE_REPORT,bindings);manifest=stage.verify_stage(STAGE_REPORT);stage_proof=read(STAGE_REPORT)
    bind(stage_proof['manifest_file'],bindings,MANIFEST_SHA)
    descriptor=dict(file=str(STAGE_REPORT),sha256=sha(STAGE_REPORT),manifest_file=stage_proof['manifest_file'],manifest_sha256=MANIFEST_SHA)
    need(descriptor==parent['confirmation_report'] and manifest['seed']==91726342 and manifest['manifest_required_before_fitting'] is True
         and manifest['training_stage']==parent['training_stage'] and manifest['new_model_trainability_established'] is False
         and manifest['native_reference']['native_identity']==parent['native_identity']
         and manifest['native_reference']['purpose']=='historical_pipeline_identity_only','Same pre-fit manifest/training/native ancestry required')
    for key in ('samples','runtime_inputs','audit','render_cache','excluded_inventory'):bind(manifest[key+'_file'],bindings,manifest[key+'_sha256'])
    audit_ref=dict(file=str(PROFILE_AUDIT),sha256=PROFILE_AUDIT_SHA,analysis_file=proof['analysis_file'],analysis_sha256=proof['analysis_sha256'])
    return parent,manifest,evidence,audit_ref,descriptor,profile


def self_test(rows):
    base=dict(input_dtypes={'input_ids':'torch.int64'},image_grid_thw=[[1,28,28]]*16,position_dtype='torch.int64',delta_dtype='torch.int64',
        rows=1,n_frames=16,resize=392,maximum_new_tokens=4,prompt_width=32,maximum_absolute_position=31,maximum_absolute_rope_delta=3,input_shapes={'input_ids':[1,32]})
    need(envelope_comparison(base,base)['passed'],'Equal envelope must pass')
    for key in ('prompt_width','maximum_absolute_position','maximum_absolute_rope_delta'):
        need(not envelope_comparison(dict(base,**{key:base[key]+1}),base)['passed'],'Oversized envelope must fail')
    need(not envelope_comparison(dict(base,input_shapes={'input_ids':[1,33]}),base)['passed'],'Expanded shape must fail')
    safe=dict(sid='fixture',n_frames=16,question='unchanged',image_files=[])
    need(p.model_view(dict(safe,gold='forbidden',target_ids=[1],panel='A'))==safe,'Offline labels entered model input')
    groups=defaultdict(list)
    for row in rows:groups[(row['panel'],row['contrast_id'])].append(row['sid'])
    a_groups=[ids for (panel,_),ids in groups.items() if panel=='A']
    c_groups=[ids for (panel,_),ids in groups.items() if panel=='C']
    def evaluate(wrong):return criterion([dict(row,exact=row['sid'] not in wrong) for row in rows])
    a_nine={sid for group in a_groups[:3] for sid in group};boundary=evaluate(a_nine)
    need(boundary['qualifies_A_and_B'] and boundary['panels']['A']['whole_correct']==99
         and boundary['panels']['A']['complete_families']==33,'Effective99 complete-family boundary differs')
    need(not evaluate(a_nine|{a_groups[3][0]})['qualifies_A_and_B'],'98 correct cannot pass the conjunction')
    scattered={group[0] for group in a_groups[:8]}
    need(evaluate(scattered)['panels']['A']['whole_correct']==100 and not evaluate(scattered)['qualifies_A_and_B'],
         'High scattered accuracy cannot replace complete families')
    c_five=set(c_groups[0]+c_groups[1][:2]);c_six=set(c_groups[0]+c_groups[1])
    need(evaluate(c_five)['C_separate'] and not evaluate(c_six)['C_separate']
         and evaluate(c_six)['qualifies_A_and_B'],'Separate C49/16 criterion must not alter A/B eligibility')
    return dict(passed=True,envelope_equal_and_oversize=True,strict_four_key_inputs=True,
        effective99_boundary=True,scattered_error_family_rejection=True,separate_C49_boundary=True,no_native_calls=True)


def check(out,frozen,started):
    import torch
    from transformers import AutoProcessor,__version__ as transformers_version
    torch.set_num_threads(4);bindings={};parent,manifest,profile_audit,audit_ref,stage_ref,profile=dependencies(bindings)
    all_rows,partitions=selected_rows(manifest);rows=all_rows[:270];need(partitions['N16']==[r['sid'] for r in rows],'N16 partition ordering differs')
    processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,use_fast=False,local_files_only=True)
    owner,rope,api=p.backend.native_api(processor);identity=parent['native_identity']
    need(api==identity['native_api'] and p.fingerprint(processor,str(transformers_version))==identity['processor']
         and p.backend.runtime_identity()==identity['runtime'] and p.model_metadata()==identity['model'],'Native processor/API/precision ancestry differs')
    need(manifest['tokenizer']['model_directory']==str(MODEL),'Frozen confirmation tokenizer model differs')
    for name,entry in manifest['tokenizer']['targets'].items():
        need(processor.tokenizer.encode(name,add_special_tokens=False)+[151645]==entry['target_ids'],'Canonical name/EOS tokens changed')
    for row in all_rows:need(row['target_ids']==manifest['tokenizer']['targets'][row['gold']]['target_ids'],'Frozen810 target ownership changed')
    bind(parent['prepared_file'],bindings,parent['runtime_bindings'][parent['prepared_file']]);old_prepared=read(parent['prepared_file'])
    bind(parent['rows_file'],bindings,parent['runtime_bindings'][parent['rows_file']]);old_rows={r['sid']:r for r in read(parent['rows_file'])}
    reference_sid=parent['profile_cases']['training_sids'][1];reference_item=old_prepared[reference_sid]
    bind(reference_item['file'],bindings,reference_item['sha256']);reference_packet=p.load_packet(torch,reference_item,old_rows[reference_sid])
    tf=reference_packet['teacher'];reference=envelope(torch,tf['inputs'],tf['layout']['position_ids'],tf['layout']['rope_deltas'])
    need(old_rows[reference_sid]['n_frames']==16 and reference['prompt_width']==reference_item['teacher_width'],'Measured widest N16 teacher reference differs')
    bind(profile['measurements_file'],bindings,profile['measurements_sha256']);measure=read(profile['measurements_file'])
    need(len(measure['memory'])==len(measure['microbatches'])==16
         and all((m['step'],m['microbatch'])==(r['step'],r['microbatch']) for m,r in zip(measure['memory'],measure['microbatches']))
         and all(r['sid']==reference_sid for r in measure['microbatches'] if r['n_frames']==16),
         'All sixteen memory/timing records must join the measured cases exactly')
    memory=[m for m,r in zip(measure['memory'],measure['microbatches']) if r['n_frames']==16]
    need(len(memory)==8 and all(type(m[k]) is int for m in memory for k in ('max_allocated_bytes','max_reserved_bytes'))
         and all(0<m['max_allocated_bytes']<=m['max_reserved_bytes'] for m in memory),'All measured N16 memory records required')
    data=DATA/out.name;data.mkdir(parents=True,exist_ok=False);prepared={};envelopes=[]
    for index,row in enumerate(rows):
        need(time.perf_counter()-started<=600,'Fixed600-second CPU preparation cap exceeded')
        bind(Path(row['path'])/'qa.txt',bindings,row['qa_sha256'])
        for image in row['image_files']:bind(image['path'],bindings,image['sha256'])
        bundle=p.prepare_joint(processor,p.model_view(row));layout=p.native.audit_layout(lambda **kw:rope(owner,**kw),bundle)
        width=bundle['metadata']['prompt_width'];positions=torch.cat((torch.arange(width).view(1,1,width),layout['position_ids']),dim=0)
        current=envelope(torch,bundle['inputs'],layout['position_ids'],layout['rope_deltas']);comparison=envelope_comparison(current,reference)
        packet=dict(bundle=bundle,layout=layout,position_ids=positions);file=data/f'bundle_{index:03d}.pt';torch.save(packet,file)
        prepared[row['sid']]=dict(file=str(file),sha256=sha(file),metadata=bundle['metadata'],layout_metadata=layout['metadata'],
            position_identity=p.tensor_info(positions),envelope=current,envelope_comparison=comparison)
        envelopes.append(dict(sid=row['sid'],**current,comparison=comparison))
    projection=profile_audit['main_projection'];S=projection['setup_seconds'];G=projection['per_N_generation_seconds']['16']
    need(math.isfinite(S) and S>0 and math.isfinite(G) and G>0,'Passed profile timing must remain positive')
    estimate=dict(projected_seconds=S+1.25*270*G+60,setup_seconds=S,per_scene_four_token_seconds=G,contexts=270,multiplier=1.25,reserve_seconds=60.,
        formula='S_new+1.25*270*G16+60',source_profile_audit=audit_ref,cap_seconds=None,no_inference_release=True,
        empirical_projection_not_guarantee=True,no_longer_length_extrapolation=True)
    coverage=dict(passed=all(r['comparison']['passed'] for r in envelopes),reference_sid=reference_sid,reference_file=reference_item['file'],
        reference_sha256=reference_item['sha256'],reference=reference,reference_kind='profile_widest_N16_teacher_forward_backward',
        measured_memory=memory,maximum_measured_allocated_bytes=max(r['max_allocated_bytes'] for r in memory),
        maximum_measured_reserved_bytes=max(r['max_reserved_bytes'] for r in memory),rows=envelopes,
        finite_sample_evidence_not_memory_guarantee=True,no_inference_release=True)
    tests=self_test(rows)
    for name,value in (('all_rows.json',all_rows),('rows.json',rows),('partitions.json',partitions),('prepared.json',prepared),
                       ('envelope.json',coverage),('timing_estimate.json',estimate),('selftests.json',tests)):save(out/name,value)
    files={str(out/name):sha(out/name) for name in ('all_rows.json','rows.json','partitions.json','prepared.json','envelope.json','timing_estimate.json','selftests.json')}
    runtime=dict(files);runtime.update({v['file']:v['sha256'] for v in prepared.values()})
    plan=dict(protocol=PROTOCOL,policy=POLICY,source_sha256=frozen,inherited_source_sha256=inherited_sources(),profile_audit=audit_ref,
        profile=profile_audit['profile'],stage_report=stage_ref,orientation_plan=parent['orientation_plan'],training_stage=parent['training_stage'],
        native_identity=identity,native_identity_sha256=parent['native_identity_sha256'],precision=parent['precision'],packages=parent['packages'],
        rows_file=str(out/'rows.json'),all_rows_file=str(out/'all_rows.json'),partitions_file=str(out/'partitions.json'),prepared_file=str(out/'prepared.json'),
        envelope_file=str(out/'envelope.json'),timing_estimate_file=str(out/'timing_estimate.json'),files=files,input_bindings=bindings,runtime_bindings=runtime,
        prepared_contexts=270,metadata_contexts=810,deferred_contexts=540,envelope_passed=coverage['passed'],tests=tests,
        no_model_or_head_calls=True,no_checkpoint_loaded=True,confirmation_predictions_accessed=False,no_inference_release=True)
    save(out/'plan.json',plan)
    return dict(passed=True,completed=True,protocol=PROTOCOL,policy=POLICY,source_sha256=frozen,inherited_source_sha256=plan['inherited_source_sha256'],
        plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),prepared_contexts=270,metadata_contexts=810,deferred_contexts=540,
        envelope_passed=coverage['passed'],timing_estimate=estimate,profile_audit=audit_ref,stage_report=stage_ref,
        no_model_or_head_calls=True,no_checkpoint_loaded=True,confirmation_predictions_accessed=False,no_inference_release=True)


def verify_stage(path,streaming=False):
    path=Path(path).resolve();path=path/'summary.json' if path.is_dir() else path;summary=read(path)
    need(summary['passed'] is summary['completed'] is True and summary['protocol']==PROTOCOL and summary['policy']==POLICY
         and summary['source_sha256']==sources() and summary['inherited_source_sha256']==inherited_sources()
         and not (path.parent/'failure.json').exists(),'Passed exact confirmation preparation required')
    need(sha(summary['plan_file'])==summary['plan_sha256'],'Preparation plan changed');plan=read(summary['plan_file'])
    need(plan['protocol']==PROTOCOL and plan['policy']==POLICY and plan['source_sha256']==sources()
         and plan['inherited_source_sha256']==inherited_sources() and plan['no_inference_release'] is True,'Preparation plan/source join differs')
    need(read(path.parent/'source_hashes.json')==sources() and read(path.parent/'inherited_sources.json')==dict(source_sha256=inherited_sources()),'Preparation archive ledger differs')
    for name,digest in sources().items():need(sha(path.parent/'source'/name.replace('/','_'))==digest,'Preparation archived source changed')
    for file,digest in plan['input_bindings'].items():need(sha(file)==digest,'Preparation input changed')
    prepared=read(plan['prepared_file']);bundle_files={r['file'] for r in prepared.values()}
    for file,digest in plan['runtime_bindings'].items():
        if not streaming or file not in bundle_files:need(sha(file)==digest,'Prepared artifact changed')
    need(len(prepared)==270 and all(plan['runtime_bindings'][r['file']]==r['sha256'] for r in prepared.values()),'Complete270 bundle bindings required')
    rows=read(plan['rows_file']);cohort(rows,16)
    need(set(prepared)=={r['sid'] for r in rows} and read(plan['partitions_file'])['N16']==[r['sid'] for r in rows], 'Prepared270 ordering differs')
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu'
         and not os.environ.get('SLURM_JOB_GPUS') and int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==4,'Only four-core Slurm CPU preparation allowed')
    out=OUT/f'check_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();frozen=snapshot(out)
    save(out/'request.json',dict(protocol=PROTOCOL,profile_audit=str(PROFILE_AUDIT),stage_report=str(STAGE_REPORT),source_sha256=frozen))
    try:
        result=check(out,frozen,started);elapsed=time.perf_counter()-started
        need(elapsed<=600 and sources()==frozen,'Preparation cap/source identity differs')
        (out/'REPORT.md').write_text('# Ordinary joint LoRA confirmation preparation\n\nAll810 metadata/order are frozen; only270 N16 native inputs were prepared. No checkpoint, model/head or confirmation prediction was consumed. Envelope and timing are evidence for a separate inference resource review; no inference is released.\n')
        save(out/'summary.json',dict(result,elapsed_seconds=elapsed))
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,elapsed_seconds=time.perf_counter()-started,partial_outputs_retained=True));raise


if __name__=='__main__':main()

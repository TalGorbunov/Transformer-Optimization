"""Resource-only continuation adapter; frozen pilot statistics are called unchanged."""
from __future__ import annotations
import argparse
from collections import Counter
import math
import os
from pathlib import Path
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import report_mmred_official_pilot as base
need=base.need;read=base.read;sha=base.sha;save=base.save;bind=base.bind;answer=base.answer
ARMS=base.ARMS;TASKS=base.TASKS;COLUMNS=base.COLUMNS
original_world=base.original_world;typed_gold=base.typed_gold;question_strata=base.question_strata
check_time=base.check_time;analysis_specs=base.analysis_specs;bootstrap=base.bootstrap
qualification=base.qualification;overlap_inventory=base.overlap_inventory;descriptive=base.descriptive
plot_results=base.plot_results;markdown_report=base.markdown_report
PROTOCOL='mmred_official_pilot_continuation_statistics'
PROPOSAL='docs/paper/NATIVE_AGGREGATION_MMRED_OFFICIAL_PILOT_CONTINUATION_REPORT.md'
PROPOSAL_SHA='79fb63e9001d6294d3b1c31039e4a5ec6b8c5eaeca4d0e1c0e92973f2f009c2d'
OWN=('scripts/report_mmred_official_pilot_continuation.py',PROPOSAL,'slurm/mmred_official_pilot_continuation_report.sbatch')
OUT=REPO/'outputs/native_aggregation_vlm/mmred_official_pilot_continuation'
DATA=Path('/mnt/data/gabriele/gnn_transformer/mmred_official_pilot_continuation')
ORIGINAL_RELEASE=REPO/'outputs/native_aggregation_vlm/mmred_official_pilot/source_release.json'
ORIGINAL_RELEASE_SHA='20c35509a21682fb9b56060c75fa2d8174bb6a20e42814ef81d27e7ce17f083a'
ORIGINAL_SOURCE_SHA='7f32ab89c8012fab358e91d07fbec2990c87a264a2e3ea69711c9387dbdfe6f5'
ORIGINAL_PROTOCOL_SHA='e22eae44ada348fc919830524b86d5cbea755a9a17b851b02e0e049d1e0cd409'
OLD_SECONDS=dict(zip(ARMS,(33,32,33)))
FAILURE_SHAS=dict(zip(ARMS,('79eaa22366cce901ebf6ab9f0aa7f7ea1937c79fdbed3462c759061cb47129ca',
    'cf34ddbfe8a3810822a746731764e9ff96bb2c41cee24d951d94d83f1c1a7aec',
    'd247fb13ebe442b83e835fd1ac11efec6adc757329b80191bb917ab84d1f53b3')))
BASE_PREDICTION_KEYS=('index','sid','n','qtype','pilot_role','generated_ids','primary_text','score','seconds',
    'max_allocated_bytes','max_reserved_bytes','file','sha256')
POLICY=dict(base.POLICY,resource_only_continuation=True,original_array_job_id='444052',
    imported_predictions_per_arm=3,new_predictions_per_arm=997,original_total_gpu_seconds=98,
    cumulative_gpu_seconds_cap_per_arm=14400,new_gpu_seconds_cap_per_arm=14340,
    original_statistics_module_unchanged=True,no_raw_tensor_handoff_rehash=True)


def sources():
    need(sha(REPO/PROPOSAL)==PROPOSAL_SHA,'Continuation report proposal changed')
    need(sha(REPO/'scripts/report_mmred_official_pilot.py')==ORIGINAL_SOURCE_SHA
         and sha(REPO/base.PROPOSAL)==base.PROPOSAL_SHA==ORIGINAL_PROTOCOL_SHA,'Frozen statistical implementation/protocol changed')
    return {name:sha(REPO/name) for name in OWN}


def original_sources(bindings):
    release=bind(ORIGINAL_RELEASE,bindings,ORIGINAL_RELEASE_SHA)
    need(release['source_sha256']==base.sources() and len(release['source_sha256'])==3
         and len(release['inherited_source_sha256'])==232,'Original statistics source release differs')
    result={**release['inherited_source_sha256'],**release['source_sha256']}
    need(len(result)==235 and all(sha(REPO/n)==h for n,h in result.items()),'Original statistics source closure changed')
    return result


def partition(rows):
    first=[next(i for i,r in enumerate(rows) if r['n']==n) for n in (8,16,32)]
    rest=[i for i in range(len(rows)) if i not in first]
    return first,rest


def allocation_fields(lineage,arm):
    old=lineage['original_allocation_seconds'];new=lineage['continuation_allocation_seconds']
    total=lineage['cumulative_allocation_seconds']
    need(all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in (old,new,total))
         and old==OLD_SECONDS[arm] and new<=14340 and total==old+new and total<=14400
         and lineage['cumulative_cap_seconds']==14400,'Original/new allocation arithmetic or fixed caps differ')
    return dict(original_allocation_seconds=old,continuation_allocation_seconds=new,
        cumulative_allocation_seconds=total,cumulative_cap_seconds=14400,new_cap_seconds=14340)


def prediction_origin(entry,origin,directory,index):
    need(entry['origin']==origin and entry['source_run']==directory and entry['source_execution_index']==index,
         'Prediction origin, source run or source execution position changed')


def imported_equal(entry,original):
    need(all(entry[k]==original[k] for k in BASE_PREDICTION_KEYS),'Original prediction content or raw ownership changed')


def verify_lineage(result,values,original,bindings):
    arm=result['arm'];lineage=result['continuation_lineage'];ref=lineage['original_run']
    first,rest=partition(original);directory=str(REPO/'outputs/native_aggregation_vlm/mmred_official_native_evaluation'/
        f'run_444052_{ARMS.index(arm)}');newdir=result['run']['directory']
    need(lineage['original_array_job_id']=='444052' and lineage['original_indices']==first
         and lineage['continuation_indices']==rest and len(first)==3 and len(rest)==997
         and ref['directory']==directory and ref['arm']==arm and ref['array_job_id']=='444052'
         and ref['allocation_id']=='444052_'+str(ARMS.index(arm)) and ref['allocation_seconds']==OLD_SECONDS[arm]
         and str(Path(newdir).resolve())==newdir and newdir!=directory,'Continuation population or allocation ownership differs')
    accounting=allocation_fields(lineage,arm);evidence={}
    for name in ('failure','config','request','prefix_natural','prefix_projection'):
        descriptor=ref[name]
        need(descriptor['file']==str(Path(directory)/(name+'.json')),'Original evidence path changed')
        if name=='failure':need(descriptor['sha256']==FAILURE_SHAS[arm],'Original failure hash changed')
        evidence[name]=bind(descriptor['file'],bindings,descriptor['sha256'])
    failure=evidence['failure'];config=evidence['config'];prefix=evidence['prefix_natural'];gate=evidence['prefix_projection']
    need(failure['type']=='ValueError' and failure['phase']=='run'
         and failure['message']=='Measured first three worlds hold remaining evaluation beyond10800s'
         and failure['progress']['completed_worlds']==3 and failure['progress']['natural']==prefix
         and gate['passed'] is False and gate['projected_seconds']>10800
         and not (Path(directory)/'summary.json').exists() and not (Path(directory)/'analysis.json').exists(),
         'Original exact resource failure must remain failed and complete through only three worlds')
    need(config['arm']==arm and config['array_job_id']=='444052'
         and config['final_checkpoint']==result['final_checkpoint']
         and config['statistics_protocol']==result['statistics_protocol']
         and config['native_identity_sha256']==result['native_identity_sha256']
         and evidence['request']['arm']==ARMS.index(arm),'Original final checkpoint/input/policy ownership differs')
    need(len(prefix)==3 and [r['index'] for r in prefix]==first
         and [r['execution_index'] for r in prefix]==[0,1,2],'Original prefix inventory differs')
    for execution,old in enumerate(prefix):
        need(bind(Path(directory)/f'natural_{old["index"]:04d}.json',bindings)==old,'Original durable prediction JSON changed')
        current=values[old['index']];imported_equal(current,old);prediction_origin(current,'original',directory,execution)
    for execution,index in enumerate(rest):prediction_origin(values[index],'continuation',newdir,execution)
    need(Counter(v['origin'] for v in values)==Counter(original=3,continuation=997),'Original/new prediction counts differ')
    return dict(lineage=lineage,allocations=accounting,original_failure_preserved=True,
        original_base_prediction_fields_identical=True,raw_tensor_payloads_reopened=False,
        original_indices=first,continuation_indices=rest,
        original_trajectory_seconds=sum(values[i]['seconds'] for i in first),
        continuation_trajectory_seconds=sum(values[i]['seconds'] for i in rest),
        original_output_tokens=sum(values[i]['score']['output_tokens'] for i in first),
        continuation_output_tokens=sum(values[i]['score']['output_tokens'] for i in rest))


def self_test(np):
    frozen=base.self_test(np);cases=[dict(n=n) for n in (16,8,16,32,8,32)]
    need(partition(cases)==([1,0,3],[2,4,5]),'First-per-N partition or untouched remainder differs')
    entry={k:k for k in BASE_PREDICTION_KEYS};imported_equal(entry,dict(entry));rejected=0
    changed=dict(entry,sha256='changed')
    try:imported_equal(changed,entry)
    except ValueError:rejected+=1
    origin=dict(origin='original',source_run='/synthetic/old',source_execution_index=1)
    prediction_origin(origin,'original','/synthetic/old',1)
    for field,value in (('origin','continuation'),('source_run','/synthetic/new'),('source_execution_index',2)):
        altered=dict(origin,**{field:value})
        try:prediction_origin(altered,'original','/synthetic/old',1)
        except ValueError:rejected+=1
    for arm in ARMS:
        old=OLD_SECONDS[arm];valid=dict(original_allocation_seconds=old,continuation_allocation_seconds=100,
            cumulative_allocation_seconds=old+100,cumulative_cap_seconds=14400)
        allocation_fields(valid,arm)
        for altered in (dict(valid,original_allocation_seconds=old+1),
            dict(valid,cumulative_allocation_seconds=100),
            dict(valid,continuation_allocation_seconds=14341,cumulative_allocation_seconds=14341+old)):
            try:allocation_fields(altered,arm)
            except ValueError:rejected+=1
    need(rejected==13 and sum(OLD_SECONDS.values())==98,'Lineage mutation/cost fixtures failed')
    return dict(frozen_statistical_fixtures=frozen,continuation_fixture_groups=4,expected_rejections=rejected,
        no_added_statistical_replicates=True,passed=True)


def inputs(paths,bindings):
    from scripts import report_mmred_official_native_continuation as auditor
    need(len(paths)==3,'Three completed independent evaluation audits required')
    audits={};summaries={};predictions={};inherited=original_sources(bindings);descriptors={}
    for path in paths:
        path=Path(path).resolve();summary=bind(path,bindings)
        result=auditor.verify_report(path)
        need(summary['protocol']==auditor.PROTOCOL and result['passed'] is result['completed'] is True
             and result['all_scheduled_numerical_evidence_collected'] is True and result['arm'] in ARMS
             and result['raw_inputs_verified_in_cpu_audit'] is True and result['raw_files_rehashed_in_handoff'] is False
             and result['statistics_protocol']==dict(file=str(REPO/base.PROPOSAL),sha256=ORIGINAL_PROTOCOL_SHA),
             'Complete numerical audit is required; there is no efficacy eligibility test')
        arm=result['arm'];need(arm not in audits,'Repeated evaluation arm')
        bind(summary['analysis_file'],bindings,summary['analysis_sha256'])
        for field in ('input_bindings','artifacts'):
            bind(summary[field+'_file'],bindings,summary[field+'_sha256'])
        values=bind(result['predictions_file'],bindings,result['predictions_sha256'])
        need(len(values)==1000 and [v['index'] for v in values]==list(range(1000)),'Exact ordered1000 lean predictions required')
        for name,digest in {**summary['source_sha256'],**summary['inherited_source_sha256']}.items():
            need(name not in inherited or inherited[name]==digest,'Conflicting inherited source identities')
            inherited[name]=digest
        audits[arm]=result;summaries[arm]=summary;predictions[arm]=values
        descriptors[arm]=dict(file=str(path),sha256=sha(path),analysis_file=summary['analysis_file'],
            analysis_sha256=summary['analysis_sha256'],predictions_file=result['predictions_file'],predictions_sha256=result['predictions_sha256'],
            final_checkpoint=result['final_checkpoint'])
    need(set(audits)==set(ARMS),'All final arms required')
    parent=audits['ordinary']
    for result in audits.values():
        need(result['original_rows_file']==parent['original_rows_file'] and result['original_rows_sha256']==parent['original_rows_sha256']
             and result['evalfeature_stage']==parent['evalfeature_stage']
             and result['native_identity_sha256']==parent['native_identity_sha256'],'All arms must use identical official inputs/features/native backbone')
    original=bind(parent['original_rows_file'],bindings,parent['original_rows_sha256'])
    expected=Counter({(role,n,q):50 for role,ns in (('val',(8,16)),('test',(8,16,32))) for n in ns for q in TASKS})
    need(len(original)==len({r['sid'] for r in original})==1000
         and Counter((r['pilot_role'],r['n'],r['qtype']) for r in original)==expected,'Exact20x50 original evaluation cohort differs')
    lineage={a:verify_lineage(audits[a],predictions[a],original,bindings) for a in ARMS}
    rows=[];files={}
    for index,row in enumerate(original):
        owner=original_world(row,files,bindings);scores={};native={}
        for arm in ARMS:
            entry=predictions[arm][index]
            need(all(entry[k]==row[k] for k in ('sid','n','qtype','pilot_role')),'Cross-arm/source prediction join differs')
            score=answer.score_answer(entry['primary_text'],row['atype'],typed_gold(row),entry['generated_ids'])
            need(score==entry['score'],'Independent strict JSON/native-EOS score differs')
            need(type(entry['seconds']) in (int,float) and math.isfinite(entry['seconds']) and entry['seconds']>0
                 and 0<entry['max_allocated_bytes']<=entry['max_reserved_bytes'],'Observed inference time/memory required')
            scores[arm]=score
            native[arm]={k:entry[k] for k in ('generated_ids','primary_text','seconds','max_allocated_bytes','max_reserved_bytes','file','sha256','origin','source_run','source_execution_index')}
        rows.append(dict(index=index,sid=row['sid'],n=row['n'],qtype=row['qtype'],pilot_role=row['pilot_role'],
            question=row['question'],atype=row['atype'],typed_gold=typed_gold(row),**owner,
            strata=question_strata(row),scores=scores,native=native))
    for name,digest in sources().items():
        if name in inherited:need(inherited.pop(name)==digest,'Shared own protocol has another hash')
    preparation=auditor.producer.profile.preparation
    recovery=bind(preparation.RECOVERY,bindings,preparation.RECOVERY_SHA)
    world_overlap=bind(recovery['overlap_file'],bindings,recovery['artifacts'][recovery['overlap_file']])
    prefix_overlap=bind(recovery['prefix_overlap_file'],bindings,recovery['artifacts'][recovery['prefix_overlap_file']])
    historical_overlap=dict(recovery_plan_file=str(preparation.RECOVERY),recovery_plan_sha256=preparation.RECOVERY_SHA,
        world_overlap_file=recovery['overlap_file'],world_overlap_sha256=sha(recovery['overlap_file']),
        prefix_overlap_file=recovery['prefix_overlap_file'],prefix_overlap_sha256=sha(recovery['prefix_overlap_file']),
        full_recovery_rows=world_overlap['rows'],selected_pilot=world_overlap['selected_pilot'],
        cross_official_split_world_groups=len(world_overlap['cross_official_split_world_groups']),
        original_prefix_groups=prefix_overlap['groups_count'],original_cross_split_prefix_groups=prefix_overlap['cross_official_split_groups'],
        original_inventory_scope_is_all_recovered_tasks_lengths=True,no_new_overlap_reconstruction=True)
    return rows,audits,summaries,descriptors,inherited,historical_overlap,lineage


def costs(rows,audits,summaries,bindings,lineage):
    result=base.costs(rows,audits,summaries,bindings)
    for arm in ARMS:
        origin=lineage[arm];resource=result['per_arm'][arm]
        need(math.isclose(origin['original_trajectory_seconds']+origin['continuation_trajectory_seconds'],
             resource['instrumented_cold_trajectory_seconds'],rel_tol=1e-12,abs_tol=1e-8)
             and origin['original_output_tokens']+origin['continuation_output_tokens']==resource['actual_output_tokens'],
             'Origin work totals must equal the unchanged combined trajectory totals')
        resource['continuation_origins']=origin
    allocated={a:lineage[a]['allocations'] for a in ARMS}
    result['continuation_allocations']=dict(per_arm=allocated,
        original_gpu_seconds=sum(v['original_allocation_seconds'] for v in allocated.values()),
        new_gpu_seconds=sum(v['continuation_allocation_seconds'] for v in allocated.values()),
        cumulative_gpu_seconds=sum(v['cumulative_allocation_seconds'] for v in allocated.values()),
        cumulative_cap_seconds=43200,new_cap_seconds=43020,
        allocation_seconds_not_added_again_to_trajectory_totals=True,original_three_trajectories_counted_once=True)
    need(result['continuation_allocations']['original_gpu_seconds']==98,'Original three-arm allocation charge differs')
    return result


def snapshot(out,inherited):
    own=sources();(out/'source').mkdir()
    for name,digest in {**inherited,**own}.items():
        need(sha(REPO/name)==digest,'Current source differs from audited source closure')
        target=out/'source'/name.replace('/','_');target.write_bytes((REPO/name).read_bytes());need(sha(target)==digest,'Archived source differs')
    return own


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--audits',nargs=3,type=Path,required=True);args=parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu'
         and int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==4 and not os.environ.get('SLURM_JOB_GPUS'),
         'Statistics/plotting requires the fixed CPU Slurm allocation')
    started=time.perf_counter();out=OUT/f'report_{os.environ["SLURM_JOB_ID"]}';data=DATA/out.name
    out.mkdir(parents=True,exist_ok=False);data.mkdir(parents=True,exist_ok=False);bindings={};progress={}
    save(out/'request.json',dict(protocol=PROTOCOL,policy=POLICY,audits=[str(p.resolve()) for p in args.audits],source_sha256=sources()))
    try:
        import numpy as np
        rows,audits,summaries,descriptors,inherited,historical_overlap,lineage=inputs(args.audits,bindings);own=snapshot(out,inherited)
        tests=self_test(np);save(out/'tests.json',tests);save(out/'joined_predictions.json',rows)
        save(out/'continuation_lineage.json',lineage)
        save(out/'world_groups.json',dict(overlap_inventory(rows),recorded_historical_overlap=historical_overlap));save(out/'descriptive_strata.json',descriptive(rows))
        tables={};replicate_files={};clusters={}
        for role,ns,qs in analysis_specs():
            check_time(started);table,values=bootstrap(np,rows,role,ns,qs)
            name=table['key'];clusters[name]=table.pop('strata')
            file=data/(name+'.npy')
            with file.open('xb') as stream:np.save(stream,values,allow_pickle=False)
            replicate_files[name]=dict(file=str(file),sha256=sha(file),bytes=file.stat().st_size,
                shape=list(values.shape),dtype=str(values.dtype),columns=list(COLUMNS))
            tables[name]=table;progress['completed_analyses']=len(tables)
        need(len(tables)==30 and all(t['replicates']==10000 for t in tables.values()),'Exact30x10000 bootstrap population required')
        save(out/'cluster_inventories.json',clusters);save(out/'replicate_files.json',replicate_files);save(out/'tables.json',tables)
        flags=qualification(tables);cost_table=costs(rows,audits,summaries,bindings,lineage)
        save(out/'qualification.json',flags);save(out/'costs.json',cost_table)
        os.environ['MPLCONFIGDIR']=str(data/'matplotlib');os.environ['MPLBACKEND']='Agg'
        plotted=plot_results(tables,out)
        packages=dict(python=sys.version,numpy=np.__version__,matplotlib=plotted['matplotlib_version'])
        result=dict(protocol=PROTOCOL,passed=True,completed=True,policy=POLICY,source_sha256=own,inherited_source_sha256=inherited,
            audits=descriptors,statistics_protocol=dict(file=str(REPO/base.PROPOSAL),sha256=ORIGINAL_PROTOCOL_SHA),
            original_statistics_release=dict(file=str(ORIGINAL_RELEASE),sha256=ORIGINAL_RELEASE_SHA),
            continuation_lineage_file=str(out/'continuation_lineage.json'),packages=packages,rows_per_arm=1000,total_predictions=3000,analysis_count=30,
            data_bootstrap_replicates=300000,tests=tests,tables_file=str(out/'tables.json'),qualification=flags,
            cluster_inventories_file=str(out/'cluster_inventories.json'),replicate_files=replicate_files,
            predictions_file=str(out/'joined_predictions.json'),strata_file=str(out/'descriptive_strata.json'),
            costs_file=str(out/'costs.json'),plots=plotted,all_efficacy_outcomes_retained=True,
            model_or_head_replay=False,raw_tensor_payloads_reopened=False,metadata_only_independent_audit_handoff=True,no_followup_release=True)
        save(out/'analysis.json',result);(out/'REPORT.md').write_text(markdown_report(tables,flags,cost_table)+
            '\nThe original resource-failed array444052 is preserved. Each arm combines its three unchanged original outcomes with997 continuation outcomes. Original allocations totaled98 GPU-seconds; new and cumulative allocations are disclosed separately in costs.json. Statistical calculations and thresholds are the frozen originals.\n')
        save(out/'input_bindings.json',bindings)
        artifacts={str(file):sha(file) for root in (out,data) for file in root.iterdir() if file.is_file()}
        save(out/'artifacts.json',artifacts);check_time(started)
        need(sources()==own and all(sha(REPO/n)==h for n,h in inherited.items()),'Source changed during statistics report')
        save(out/'summary.json',dict(protocol=PROTOCOL,passed=True,completed=True,policy=POLICY,
            source_sha256=own,inherited_source_sha256=inherited,analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),
            input_bindings_file=str(out/'input_bindings.json'),input_bindings_sha256=sha(out/'input_bindings.json'),
            artifacts_file=str(out/'artifacts.json'),artifacts_sha256=sha(out/'artifacts.json'),
            elapsed_seconds=time.perf_counter()-started,promising_mass_specific_pilot=flags['promising_mass_specific_pilot'],
            efficacy_is_not_completion_gate=True,no_followup_release=True))
    except BaseException as exc:
        save(out/'failure_input_bindings.json',bindings)
        save(out/'failure.json',dict(protocol=PROTOCOL,passed=False,type=type(exc).__name__,message=str(exc),
            elapsed_seconds=time.perf_counter()-started,progress=progress,input_bindings_file=str(out/'failure_input_bindings.json'),
            input_bindings_sha256=sha(out/'failure_input_bindings.json'),partial_outputs_retained=True,no_followup_release=True));raise


if __name__=='__main__':main()

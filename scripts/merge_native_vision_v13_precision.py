"""Explicit CPU precision adapter for the frozen V13 null-cache merge.

No module monkeypatches. The original worker remains immutable. This adapter
preserves its native/tensor/metadata gates and changes only exact equality of
three descriptive FP32 reduction scalars to the diagnosed precision tolerance.
"""
from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
import sys
import time
import json
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import cache_native_vision_v13_null_features as worker
from scripts import diagnose_native_vision_v13_null_merge as diagnostic
need,read,sha,save,object_sha,tensor_info=worker.need,worker.read,worker.sha,worker.save,worker.object_sha,worker.tensor_info
OUT,DATA,PROTOCOL,POLICY=worker.OUT,worker.DATA,worker.PROTOCOL,worker.POLICY
require_slurm,verify_plan,verify_snapshot,verify_profile=worker.require_slurm,worker.verify_plan,worker.verify_snapshot,worker.verify_profile
load_controls,hidden_difference,replay_metrics=worker.load_controls,worker.hidden_difference,worker.replay_metrics
validate_locations,validate_observed_inputs,index=worker.validate_locations,worker.validate_observed_inputs,worker.index
OWN=('scripts/merge_native_vision_v13_precision.py','slurm/native_vision_v13_null_merge_precision.sbatch')
PRECISION_POLICY=dict(protocol='v13_descriptive_FP32_merge_precision',fields=['rms','l2','relative_rms'],relative_tolerance=1e-6,absolute_tolerance=1e-8,
    exact_fields=['row','feature_id','exact','maximum_absolute','binding'],native_head_rules_unchanged=True,all_tensor_identities_exact=True,
    no_worker_mutation=True,no_new_inference=True)
DIAGNOSTIC=REPO/'outputs/native_aggregation_vlm/v13/null_merge_diagnostic/run_442416/summary.json'
DIAGNOSTIC_SHA='89debfcadd2cabe9feae17b248a080c940de5afd25932cb1fb19455df6d9ba3b'
COMPARISON_SHA='99c13de54f1acffc5f05e4e2e5d52b7593aa80ab7c913f69f754a3ba1f3b5d56'
FAILED=REPO/'outputs/native_aggregation_vlm/v13/null_cache/merge_442411'
FAILED_LOG=REPO/'logs/v13_null_merge-442411.out'
FAILED_LOG_SHA='ef01cb0faa363b3f79cc5954074250417a21ef25b6a0fb072ab8efcd9141010e'
FAILED_SOURCE_SHA='7223dbf72f7eff2ee07babfd43ec2d9326567448e9d82fb07de1c413e1f670c0'


def sources():return {name:sha(REPO/name) for name in dict.fromkeys((*OWN,*diagnostic.OWN,*worker.OWN))}


def snapshot(out):
    worker.snapshot(out)
    (out/'precision_source').mkdir();frozen=sources()
    for name,digest in frozen.items():
        p=out/'precision_source'/name.replace('/','_');p.write_bytes((REPO/name).read_bytes());need(sha(p)==digest,'Adapter source changed while copying')
    save(out/'precision_source_hashes.json',frozen)


def precision_contract():
    need(sha(DIAGNOSTIC)==DIAGNOSTIC_SHA,'Canonical precision diagnostic changed');d=read(DIAGNOSTIC)
    need(d['passed'] and d['diagnostic_only'] and d['no_model_loaded'] and d['no_input_modified']
        and d['exact_state_and_metadata_checks_passed'] and d['comparisons']==392 and d['old_cache_comparisons']==136
        and d['small_stress_comparisons']==256 and d['nonidentical_descriptive_rows']==17
        and d['all_descriptive_scalars_within_candidate_precision'] and len(d['unchanged_head_replay_checks'])==8
        and all(r['passed'] for r in d['unchanged_head_replay_checks']),'Precision diagnostic did not establish the bounded repair')
    need(sha(d['comparison_file'])==d['comparison_sha256']==COMPARISON_SHA,'Diagnostic comparisons changed')
    need(sha(FAILED_LOG)==FAILED_LOG_SHA and sha(FAILED/'source_hashes.json')==FAILED_SOURCE_SHA,'Original failed merge evidence changed')
    for name,digest in d['source_sha256'].items():need(sha(REPO/name)==digest,'Diagnostic/worker source changed')
    for name,digest in read(FAILED/'source_hashes.json').items():
        need(sha(REPO/name)==digest and sha(FAILED/'source'/name.replace('/','_'))==digest,'Failed merge source snapshot changed')
    return dict(schema_version=1,policy=PRECISION_POLICY,source_sha256=sources(),
        diagnostic_summary=dict(file=str(DIAGNOSTIC),sha256=DIAGNOSTIC_SHA),
        diagnostic_comparisons=dict(file=d['comparison_file'],sha256=COMPARISON_SHA),
        original_failed_merge=dict(directory=str(FAILED),log_file=str(FAILED_LOG),log_sha256=FAILED_LOG_SHA,
            source_hashes_file=str(FAILED/'source_hashes.json'),source_hashes_sha256=FAILED_SOURCE_SHA),
        allowed_profile_plan_sha256=d['profile_plan_sha256'],allowed_profile_summary_sha256=d['profile_summary_sha256'])


def compare_descriptive(expected,recorded):
    need(len(expected)==len(recorded),'Descriptive comparison list length changed')
    for actual,saved in zip(expected,recorded):
        need(set(actual)==set(saved),'Descriptive comparison keys changed')
        for name in actual:
            if name in PRECISION_POLICY['fields']:
                a,b=actual[name],saved[name]
                need(type(a) in (int,float) and type(b) in (int,float) and math.isfinite(a) and math.isfinite(b)
                    and a>=0 and b>=0 and math.isclose(a,b,rel_tol=1e-6,abs_tol=1e-8),'Descriptive FP32 scalar exceeds diagnosed precision: '+name)
            else:need(type(actual[name]) is type(saved[name]) and actual[name]==saved[name],'Exact descriptive field changed: '+name)


def self_test():
    original=dict(row=2,feature_id='fixed',exact=False,maximum_absolute=.25,binding=False,rms=.1,l2=2.,relative_rms=.001)
    close=dict(original,rms=.1+7e-9);compare_descriptive([original],[close])
    for changed in [dict(original,row=3),dict(original,exact=True),dict(original,maximum_absolute=.25+1e-12),
                    dict(original,binding=True),dict(original,rms=.101),dict(original,l2=float('nan'))]:
        try:compare_descriptive([original],[changed])
        except ValueError:pass
        else:raise AssertionError('Adapter accepted changed identity or excessive/nonfinite scalar')
    return dict(passed=True,tests=['bounded_FP32_reduction_variation','exact_row_flag_maximum_and_binding','reject_large_or_nonfinite_change'])


def verify_profile_tensors(torch,result,plan):
    """Independent CPU replay of recorded metrics, with no model or head calls."""
    raw=torch.load(result['raw_file'],map_location='cpu',weights_only=True)
    rows=read(result['observations_file']);need(raw['schema_version']==1 and len(raw['observations'])==8,'Profile raw schema differs')
    controls=load_controls(torch,plan);small={}
    for item,row in zip(raw['observations'],rows):
        ids=row['identity']['feature_ids'];states=item['states']
        need(item['feature_ids']==ids and all(item[k]==row[k] for k in ('phase','stratum','case'))
            and tensor_info(states)==row['states'] and states.shape==(len(ids),3584) and states.dtype==torch.float16
            and bool(torch.isfinite(states).all()),'Profile raw state identity differs')
        need(item['native_logits'].dtype==item['replayed_logits'].dtype==torch.float16, 'Profile raw native logits dtype differs')
        recomputed=replay_metrics(torch,item['native_logits'],item['replayed_logits']);recorded=row['replay']
        need(all(recomputed[key]==recorded[key] for key in ('top1_equal','all_top1_equal','passed'))
            and len(recomputed['tv'])==len(recorded['tv'])
            and all(math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-12) for a,b in zip(recomputed['tv'],recorded['tv']))
            and math.isclose(recomputed['maximum_tv'],recorded['maximum_tv'],rel_tol=1e-9,abs_tol=1e-12),
            'Recorded native head comparisons differ from raw tensors beyond CPU/GPU FP64 reduction precision')
        need(set(item['immutable_old_states'])==set(ids)&set(controls)
            and all(torch.equal(value,controls[fid]) for fid,value in item['immutable_old_states'].items()),
            'Profile old controls differ from the original cache tensors')
        expected=[dict(row=i,feature_id=fid,**hidden_difference(torch,states[i],controls[fid]))
            for i,fid in enumerate(ids) if fid in controls]
        compare_descriptive(expected,row['old_cache_hidden_descriptive'])
        if row['case']=='small':small.update({fid:value.clone() for fid,value in zip(ids,states)})
        else:
            compare_descriptive([hidden_difference(torch,value,small[fid]) for fid,value in zip(ids,states)],
                row['small_vs_stress_hidden_descriptive'])
    return dict(passed=True,batches=8,distinct_missing=8,distinct_reused=8,old_state_comparisons=136,
        small_stress_comparisons=256,head_replays_recomputed=8,no_model_loaded=True)



def merge(args):
    require_slurm();job=os.environ['SLURM_JOB_ID'];start=time.perf_counter();out=OUT/f'merge_{job}'
    out.mkdir(parents=True,exist_ok=False);snapshot(out);index(out,'V13 exact native cache union',[('Summary','summary.json'),('Sources','source_hashes.json')])
    precision=precision_contract();save(out/'precision_dispatcher.json',precision);tests=self_test()
    import torch
    torch.set_num_threads(4);plan=verify_plan(args.plan,pixels=False)
    need(sha(args.plan)==precision['allowed_profile_plan_sha256'],'Precision repair applies only to the diagnosed plan')
    cpu=read(OUT/f'check_{plan["slurm_job_id"]}'/'summary.json')
    need(cpu['passed'] and cpu['plan_sha256']==sha(args.plan),'Completed exact worker CPU freeze required')
    cache=read(plan['parent_cache']['file']);staged=read(plan['inventory_plan']['file']);inv=staged['inventory']
    old_audit=validate_locations(torch,cache['features']);need(old_audit['features']==33658,'Old V10 state coverage differs')
    locations=dict(cache['features']);seen=set();summaries=[];common=None;profile=None;profile_audit=None
    need(len(args.merge)==4 and len({Path(x).resolve() for x in args.merge})==4,'Four distinct shard directories are required')
    for directory in map(lambda p:Path(p).resolve(),args.merge):
        verify_snapshot(directory,plan);summary=read(directory/'summary.json');shard=summary['shard']
        need(type(shard) is int and shard in range(4) and shard not in seen,'Duplicate/invalid shard index');seen.add(shard)
        need(summary['protocol']==PROTOCOL and summary['policy']==POLICY and summary['profile'] is False
            and summary['completed'] is True and summary['passed'] is True and summary['computational_integrity_passed'] is True
            and summary['execution_within_cap'] is True and 0<summary['total_seconds']<=600
            and summary['plan_sha256']==sha(args.plan) and summary['source_sha256']==plan['source_sha256'],
            'A completed bounded exact-plan shard is required')
        need(all(summary[k]==plan[k] for k in ('model','runtime','processor','native_api','read_boundary','parent_cache','inventory_plan'))
            and summary['native_dtypes']==plan['native_dtypes'] and summary['prefix_coverage']==plan['profile_groups'],
            'Shard model/native/source provenance differs')
        config=read(directory/'config.json')
        need(config=={key:summary[key] for key in config} and sha(directory/'plan.json')==sha(args.plan),'Shard config/plan copy changed')
        if common is None:
            common=summary;profile=verify_profile(summary['profile_directory'],plan,args.plan)
            need(sha(Path(summary['profile_directory'])/'summary.json')==summary['profile_summary_sha256'],'Shard profile binding changed')
            need(sha(Path(summary['profile_directory'])/'summary.json')==precision['allowed_profile_summary_sha256'],'Precision repair profile differs')
            profile_audit=verify_profile_tensors(torch,profile,plan)
            for row in read(profile['observations_file']):validate_observed_inputs(torch,plan,row,row['identity']['feature_ids'])
        need(all(summary[k]==common[k] for k in ('profile_directory','profile_summary_sha256','hardware','native_dtypes'))
            and summary['hardware']==profile['hardware'],'Shards must use the same passing profile and native backend')
        need(sha(summary['features_file'])==summary['features_sha256']
            and sha(summary['observations_file'])==summary['observations_sha256'],'Shard raw state/observation files changed')
        observations=read(summary['observations_file']);ids=plan['shards'][shard]
        batches=[ids[offset:offset+64] for offset in range(0,len(ids),64)]
        need(len(observations)==len(batches)==77 and summary['calls']==dict(model=77,vision=77,language=77,norm=77,last_block=77),
            'Shard exact call inventory differs')
        blob=torch.load(summary['features_file'],map_location='cpu',weights_only=True);states=blob['states']
        need(blob['schema_version']==1 and blob['feature_ids']==ids and states.shape==(4916,3584) and states.dtype==torch.float16
            and bool(torch.isfinite(states).all()) and summary['feature_count']==4916 and tensor_info(states)==summary['states']
            and object_sha(ids)==summary['feature_ids_sha256'] and set(summary['feature_state_sha256'])==set(ids),
            'Shard native tensor/feature identities differ')
        offset=0
        for b,(observation,batch_ids) in enumerate(zip(observations,batches)):
            need(observation['phase']=='local_prefix' and observation['stratum']=='mixed_numeric_prefix'
                and observation['case']==f'batch_{b:04d}' and observation['batch_size']==len(batch_ids)
                and observation['native_model_forwards']==observation['native_visual_forwards']==1,
                'Shard batch order/call metadata differs')
            validate_observed_inputs(torch,plan,observation,batch_ids)
            end=offset+len(batch_ids);need(tensor_info(states[offset:end])==observation['states'],'Saved shard state differs from captured batch')
            offset=end
        need(offset==4916,'Observed shard row count differs')
        for row,(fid,value) in enumerate(zip(ids,states)):
            digest=tensor_info(value)['sha256'];need(fid not in locations and digest==summary['feature_state_sha256'][fid],
                'Missing feature overlaps old cache or native bytes differ')
            locations[fid]=dict(file=summary['features_file'],file_sha256=summary['features_sha256'],row=row,state_sha256=digest)
        summaries.append(dict(directory=str(directory),summary_sha256=sha(directory/'summary.json'),shard=shard))
    summaries.sort(key=lambda x:x['shard'])
    need(seen==set(range(4)) and len(locations)==53322 and set(locations)==set(cache['features'])|set(plan['missing_feature_ids'])
        and all(locations[fid]==entry for fid,entry in cache['features'].items()),'Exact V10 union coverage/preservation failed')
    need(len(inv['auxiliary_groups'])==972 and all(len(group['local_feature_ids'])==24
        and all(fid in locations for fid in group['local_feature_ids']) and group['global_feature_id'] in cache['features']
        for group in inv['auxiliary_groups'].values()),'Auxiliary occurrence/global support incomplete')
    need(verify_plan(args.plan,pixels=False)==plan,'Plan/source changed during merge')
    result=dict(schema_version=1,protocol='v13_training_null_union_cache',complete=True,training_only=True,
        plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),source_sha256=plan['source_sha256'],
        model=plan['model'],runtime=plan['runtime'],processor=plan['processor'],native_api=plan['native_api'],
        native_dtypes=common['native_dtypes'],hardware=common['hardware'],read_boundary=plan['read_boundary'],
        parent_cache=plan['parent_cache'],parent_plan=plan['parent_plan'],inventory_plan=plan['inventory_plan'],inventory_sha256=plan['inventory_sha256'],
        target_token_ids=cache['target_token_ids'],scenes=cache['scenes'],features=locations,feature_count=53322,
        original_feature_count=33658,new_feature_count=19664,counts=inv['counts'],null_banks=inv['banks'],
        auxiliary_groups=inv['auxiliary_groups'],required_null_feature_ids=inv['required_local_feature_ids'],
        required_global_feature_ids=inv['required_global_feature_ids'],strict_prefix_vocabulary=inv['strict_prefix_vocabulary'],
        shards=summaries,parent_shards=cache['shards'],profile_directory=common['profile_directory'],
        profile_summary_sha256=common['profile_summary_sha256'],preserved_v10_features_exact=True,
        preserve_occurrence_multiplicity=True,no_new_global_states=True,no_training_release=True,
        merge_directory=str(out),merge_slurm_job_id=job,precision_dispatcher=precision)
    target=DATA/'feature_cache.json';need(not target.exists() and not target.with_suffix('.sha256').exists(),
        'Preserve an existing or partial canonical V13 cache')
    save(target,result);target.with_suffix('.sha256').write_text(sha(target)+'\n')
    save(out/'summary.json',dict(passed=True,completed=True,computational_integrity_passed=True,plan_file=str(Path(args.plan).resolve()),
        plan_sha256=sha(args.plan),source_sha256=plan['source_sha256'],cache_file=str(target),cache_sha256=sha(target),
        feature_count=53322,original_feature_count=33658,new_feature_count=19664,counts=inv['counts'],
        old_tensor_audit=old_audit,profile_tensor_audit=profile_audit,shards=summaries,preserved_v10_features_exact=True,
        successful_artifact_calls=dict(model=316,vision=316,standalone_head=8,last_block=316),
        campaign_allocation_accounting_required=True,no_training_release=True,seconds=time.perf_counter()-start,slurm_job_id=job,
        precision_dispatcher=precision,precision_self_tests=tests))
    print(json.dumps(dict(passed=True,cache_file=str(target),cache_sha256=sha(target),feature_count=53322)),flush=True)



def verify_cache(path,*,verify_tensors=False):
    """Exact original public metadata audit, plus mandatory adapter provenance."""
    result=worker.verify_cache(path,verify_tensors=False);precision=precision_contract()
    need(result['precision_dispatcher']==precision and result['plan_sha256']==precision['allowed_profile_plan_sha256']
        and result['profile_summary_sha256']==precision['allowed_profile_summary_sha256'],'Union dropped or changed precision provenance')
    out=Path(result['merge_directory']);summary=read(out/'summary.json')
    need(summary['precision_dispatcher']==precision and read(out/'precision_dispatcher.json')==precision
        and read(out/'precision_source_hashes.json')==precision['source_sha256'],'Merge adapter evidence differs')
    for name,digest in precision['source_sha256'].items():
        need(sha(out/'precision_source'/name.replace('/','_'))==digest,'Precision adapter copied source differs')
    if verify_tensors:
        require_slurm();import torch
        torch.set_num_threads(4);validate_locations(torch,result['features'])
        plan=verify_plan(result['plan_file'],pixels=False);profile=verify_profile(result['profile_directory'],plan,result['plan_file'])
        verify_profile_tensors(torch,profile,plan)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--plan',type=Path);parser.add_argument('--merge',nargs=4,type=Path);args=parser.parse_args();require_slurm()
    if args.self_test:
        out=OUT/f'precision_check_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False);snapshot(out)
        save(out/'summary.json',dict(passed=True,tests=self_test(),precision_dispatcher=precision_contract()))
        index(out,'V13 precision adapter tests',[('Summary','summary.json'),('Sources','precision_source_hashes.json')])
    else:
        need(args.plan is not None and args.merge is not None,'Pass exact worker plan and all four completed shards')
        args.plan=args.plan.resolve();merge(args)


if __name__=='__main__':main()

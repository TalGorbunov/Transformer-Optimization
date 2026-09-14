"""Read-only CPU diagnosis of frozen V13 descriptive hidden-reduction mismatch.

No merge, tensor mutation, model call, or acceptance-rule change. Native head
replay rules and tensor identities stay exact/unchanged. Only descriptive
FP32 norms are compared across hosts and against a CPU FP64 reference.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
OUT=REPO/'outputs/native_aggregation_vlm/v13/null_merge_diagnostic'
PROFILE=REPO/'outputs/native_aggregation_vlm/v13/null_cache/profile_442394'
OWN=('scripts/diagnose_native_vision_v13_null_merge.py','slurm/native_vision_v13_null_merge_diagnostic.sbatch')


def need(value,message):
    if not value:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()
def save(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def main():
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu'
        and not os.environ.get('SLURM_JOB_GPUS'),'Only CPU Slurm is authorized')
    out=OUT/f'run_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False);(out/'source').mkdir()
    (out/'INDEX.md').write_text('# V13 merge reduction diagnosis\n\n[Summary](summary.json) · [Every comparison](comparisons.json) · [Sources](source_hashes.json).\n')
    from scripts import cache_native_vision_v13_null_features as worker
    sources={name:sha(REPO/name) for name in dict.fromkeys((*OWN,*worker.OWN))}
    for name,digest in sources.items():
        p=out/'source'/name.replace('/','_');p.write_bytes((REPO/name).read_bytes());need(sha(p)==digest,'Source changed during snapshot')
    save(out/'source_hashes.json',sources)
    import torch
    torch.set_num_threads(4);start=time.perf_counter()
    summary=read(PROFILE/'summary.json');plan=worker.verify_plan(summary['plan_file'],pixels=False)
    need(worker.verify_profile(PROFILE,plan,Path(summary['plan_file']))==summary,'Frozen profile metadata changed')
    need(sha(summary['raw_file'])==summary['raw_sha256'] and sha(summary['observations_file'])==summary['observations_sha256'],'Raw evidence changed')
    raw=torch.load(summary['raw_file'],map_location='cpu',weights_only=True);rows=read(summary['observations_file']);controls=worker.load_controls(torch,plan)
    need(raw['schema_version']==1 and len(raw['observations'])==len(rows)==8,'Profile batch coverage differs')
    fields=('rms','l2','relative_rms');records=[];head_results=[];small={}
    def fp64(new,old):
        difference=new.double()-old.double();rms=float(difference.square().mean().sqrt());base=float(old.double().square().mean().sqrt())
        return dict(rms=rms,l2=float(difference.norm()),relative_rms=rms/max(base,1e-12),maximum_absolute=float(difference.abs().max()))
    def compare(kind,batch,i,fid,new,old,saved):
        current=worker.hidden_difference(torch,new,old);reference=fp64(new,old)
        need(saved['binding'] is False and current['binding'] is False and type(saved['exact']) is bool
            and saved['exact']==current['exact'] and saved['maximum_absolute']==current['maximum_absolute'],
            'Exact descriptive flags or maximum difference disagree')
        need(all(isinstance(saved[f],(int,float)) and math.isfinite(saved[f]) and saved[f]>=0 for f in fields), 'Invalid recorded norm')
        diffs={f:dict(recorded=saved[f],current=current[f],reference_fp64=reference[f],
            absolute_difference=abs(saved[f]-current[f]),relative_difference=abs(saved[f]-current[f])/max(abs(saved[f]),abs(current[f]),1e-30),
            recorded_reference_error=abs(saved[f]-reference[f]),current_reference_error=abs(current[f]-reference[f]),
            close_rtol1e6_atol1e8=math.isclose(saved[f],current[f],rel_tol=1e-6,abs_tol=1e-8)) for f in fields}
        records.append(dict(kind=kind,batch=batch,row=i,feature_id=fid,identity_and_exact_fields_passed=True,
            exact_scalar_dict_match=all(saved[f]==current[f] for f in fields),differences=diffs,
            dtype=str(new.dtype),shape=list(new.shape),new_stride=list(new.stride()),old_stride=list(old.stride()),
            native_tensor_hash=worker.tensor_info(new)['sha256'],reference_tensor_hash=worker.tensor_info(old)['sha256']))
    for batch,(item,row) in enumerate(zip(raw['observations'],rows)):
        ids=row['identity']['feature_ids'];states=item['states']
        need(item['feature_ids']==ids and all(item[k]==row[k] for k in ('phase','stratum','case'))
            and worker.tensor_info(states)==row['states'] and states.dtype==torch.float16
            and states.shape==(len(ids),3584) and bool(torch.isfinite(states).all()),'Exact raw state identity changed')
        need(set(item['immutable_old_states'])==set(ids)&set(controls)
            and all(torch.equal(value,controls[fid]) for fid,value in item['immutable_old_states'].items()), 'Immutable old control changed')
        recomputed=worker.replay_metrics(torch,item['native_logits'],item['replayed_logits']);recorded=row['replay']
        need(all(recomputed[k]==recorded[k] for k in ('top1_equal','all_top1_equal','passed'))
            and recomputed['passed'] and all(math.isclose(x,y,rel_tol=1e-9,abs_tol=1e-12) for x,y in zip(recomputed['tv'],recorded['tv']))
            and math.isclose(recomputed['maximum_tv'],recorded['maximum_tv'],rel_tol=1e-9,abs_tol=1e-12),
            'Frozen native-head numerical gate or raw replay changed')
        head_results.append(dict(batch=batch,passed=True,maximum_tv=recomputed['maximum_tv']))
        expected=[(i,fid) for i,fid in enumerate(ids) if fid in controls]
        saved=row['old_cache_hidden_descriptive'];need([(r['row'],r['feature_id']) for r in saved]==expected,'Control row/feature ownership changed')
        for (i,fid),value in zip(expected,saved):compare('old_cache',batch,i,fid,states[i],controls[fid],value)
        if row['case']=='small':small.update({fid:value.clone() for fid,value in zip(ids,states)})
        else:
            need(len(row['small_vs_stress_hidden_descriptive'])==len(ids),'Stress comparison count changed')
            for i,(fid,value) in enumerate(zip(ids,row['small_vs_stress_hidden_descriptive'])):
                compare('small_stress',batch,i,fid,states[i],small[fid],value)
    need(len(records)==392 and sum(r['kind']=='old_cache' for r in records)==136 and len(head_results)==8,'Diagnostic comparison coverage differs')
    mismatches=[r for r in records if not r['exact_scalar_dict_match']]
    by_field={f:dict(nonidentical=sum(r['differences'][f]['absolute_difference']!=0 for r in records),
        maximum_absolute_difference=max(r['differences'][f]['absolute_difference'] for r in records),
        maximum_relative_difference=max(r['differences'][f]['relative_difference'] for r in records),
        all_within_rtol1e6_atol1e8=all(r['differences'][f]['close_rtol1e6_atol1e8'] for r in records)) for f in fields}
    save(out/'comparisons.json',records)
    result=dict(passed=True,diagnostic_only=True,no_model_loaded=True,no_input_modified=True,source_sha256=sources,
        profile_directory=str(PROFILE),profile_summary_sha256=sha(PROFILE/'summary.json'),profile_plan_file=summary['plan_file'],profile_plan_sha256=summary['plan_sha256'],
        profile_raw_file=summary['raw_file'],profile_raw_sha256=summary['raw_sha256'],
        exact_state_and_metadata_checks_passed=True,unchanged_head_replay_checks=head_results,
        comparisons=392,old_cache_comparisons=136,small_stress_comparisons=256,nonidentical_descriptive_rows=len(mismatches),
        by_field=by_field,all_descriptive_scalars_within_candidate_precision=all(v['all_within_rtol1e6_atol1e8'] for v in by_field.values()),
        candidate_precision=dict(relative_tolerance=1e-6,absolute_tolerance=1e-8,fields=list(fields),applied_to_worker=False),
        CPU=platform.processor(),machine=platform.machine(),node=platform.node(),torch_version=str(torch.__version__),threads=torch.get_num_threads(),
        comparison_file=str(out/'comparisons.json'),comparison_sha256=sha(out/'comparisons.json'),seconds=time.perf_counter()-start,
        limitation='Both original and replay hidden statistics run on CPU; host-specific reduction differences are investigated, not assumed. Native head/tensor gates are unchanged.')
    save(out/'summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','unchanged_head_replay_checks')}),flush=True)


if __name__=='__main__':main()

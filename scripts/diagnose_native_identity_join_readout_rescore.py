"""CPU-only diagnosis of preserved readout report443297; no acceptance change.

Compare every saved prediction with raw first-token logits, preserving all metadata
and FP32/FP64 loss differences. No head, model, optimizer, or new fitting calls.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import time
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/native_aggregation_vlm/identity_join_readout'
PLAN=BASE/'check_443294/plan.json'
PLAN_SHA='cb805a97882e7cd97b18993dde742bee82031ae90f1f10fde1a31ca4867e7edc'
RUN=BASE/'run_443295/summary.json'
RUN_SHA='60b8fe719b71d3c3cd8774abc4d360acf407345a8aa98ed7adad01d45de71bfe'
FAILURE=BASE/'report_443297/failure.json'
FAILURE_SHA='6f06a5c8653a385bb8bf36d3792cc757f622d3732b1079eb0525ccd824d37f78'
REQUEST=BASE/'report_443297/request.json'
REQUEST_SHA='96f643b667266ec2b91eb82c3673b7b88130736495d345781cdc8171442eae99'
PROTOCOL='identity_join_readout_rescore_diagnosis'
OWN=('scripts/diagnose_native_identity_join_readout_rescore.py','slurm/native_identity_join_readout_rescore.sbatch')


def need(test,message):
    if not test:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def bind(path,expected,ledger):
    path=Path(path).resolve();value=sha(path)
    need(value==expected,'Changed bound artifact: '+str(path));ledger[str(path)]=value
    return path


def tensor_info(torch,value):
    value=value.detach().cpu().contiguous()
    return dict(shape=list(value.shape),dtype=str(value.dtype),
        sha256=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest())


def bits32(value):
    need(math.isfinite(value) and value>=0,'ULP comparison needs finite nonnegative values')
    if value==0:return 0
    return struct.unpack('!I',struct.pack('!f',value))[0]


def float32(value):return struct.unpack('!f',struct.pack('!f',value))[0]


def difference(a,b):
    bits=bits32(b);spacing=struct.unpack('!f',struct.pack('!I',bits+1))[0]-float32(b)
    return dict(signed=a-b,absolute=abs(a-b),fp32_ulp_distance=abs(bits32(a)-bits),
        second_value_fp32_ulp=spacing,absolute_in_second_fp32_ulps=abs(a-b)/spacing,
        ulp_distance_rule='ordered nonnegative FP32 bit patterns; FP64 reference rounded only for ULP distance')


def original_close(a,b):return math.isfinite(a) and math.isfinite(b) and abs(a-b)<=2e-6+2e-6*max(abs(a),abs(b))


def self_test():
    one=struct.unpack('!f',struct.pack('!I',bits32(1.)+1))[0]
    need(difference(one,1.)['fp32_ulp_distance']==1 and difference(1.,1.)['absolute']==0,
         'FP32 ULP fixture failed')
    need(original_close(1.,1.+1e-6) and not original_close(1.,1.+1e-4),'Original comparator fixture failed')
    return dict(passed=True,checks=2,fp32_ulp_identity_and_adjacency=True,original_comparator_unchanged=True)


def diagnose(out,started):
    ledger={};plan=read(bind(PLAN,PLAN_SHA,ledger));summary=read(bind(RUN,RUN_SHA,ledger))
    failure=read(bind(FAILURE,FAILURE_SHA,ledger));request=read(bind(REQUEST,REQUEST_SHA,ledger))
    need(failure['type']=='ValueError' and failure['message']=='Independent full-vocabulary rescore differs'
         and request['phase']=='report' and request['plan']==str(PLAN) and request['runs']==[str(RUN.parent)]
         and not (FAILURE.parent/'summary.json').exists(),'Exact preserved failed report differs')
    need(summary['passed'] is True and summary['completed'] is True and summary['phase']=='run'
         and summary['plan_file']==str(PLAN) and summary['plan_sha256']==PLAN_SHA
         and summary['source_sha256']==plan['source_sha256']==failure['source_sha256']==request['source_sha256'],
         'Original source/plan/run joins differ')
    check_file=PLAN.parent/'summary.json';check=read(check_file);ledger[str(check_file)]=sha(check_file)
    need(check['passed'] is True and check['completed'] is True and check['plan_sha256']==PLAN_SHA
         and check['source_sha256']==plan['source_sha256'],'Original CPU check differs')
    sources={**plan['source_sha256'],**{name:sha(ROOT/name) for name in OWN}}
    (out/'source').mkdir()
    for name,h in sources.items():
        bind(ROOT/name,h,ledger);dest=out/'source'/name.replace('/','_');dest.write_bytes((ROOT/name).read_bytes())
        need(sha(dest)==h,'Diagnostic source copy changed')
        if name in plan['source_sha256']:
            for directory in (PLAN.parent,RUN.parent,FAILURE.parent):bind(directory/'source'/name.replace('/','_'),h,ledger)
    save(out/'source_hashes.json',sources)
    log=ROOT/'logs/identity_join_readout_report-443297.out'
    # Failure JSON is exactly frozen above; retain the full original traceback log too.
    ledger[str(log)]=sha(log)
    rows=read(bind(plan['rows_file'],plan['runtime_bindings'][plan['rows_file']],ledger))
    predictions=read(bind(summary['predictions_file'],summary['predictions_sha256'],ledger))
    raw_path=bind(summary['raw_file'],summary['raw_sha256'],ledger)
    need(len(rows)==len(predictions)==108 and len({r['sid'] for r in rows})==108,'Full108 input/prediction inventory required')
    save(out/'resolved_inputs.json',dict(protocol=PROTOCOL,files=ledger,plan_file=str(PLAN),plan_sha256=PLAN_SHA,
        run_summary_file=str(RUN),run_summary_sha256=RUN_SHA,failure_file=str(FAILURE),failure_sha256=FAILURE_SHA,
        failure_summary_absent=True,rows=rows,predictions=predictions,original_source_sha256=plan['source_sha256']))
    import torch
    torch.set_num_threads(4)
    raw=torch.load(raw_path,map_location='cpu',weights_only=True);logits=raw['logits'];sids=[r['sid'] for r in rows]
    need(raw['schema_version']==1 and raw['sids']==sids and logits.shape==(108,152064)
         and logits.dtype==torch.float16 and bool(torch.isfinite(logits).all())
         and tensor_info(torch,logits)==summary['raw_tensor'] and raw['batches']==summary['raw_batches']
         and len(raw['batches'])==7,'Raw FP16 full-vocabulary identity/coverage differs')
    targets=torch.tensor([r['first_token_id'] for r in rows],dtype=torch.int64)
    need(bool(((targets>=0)&(targets<152064)).all()),'Target IDs outside full vocabulary')
    ids=logits.argmax(-1);cpu_full=torch.nn.functional.cross_entropy(logits.float(),targets,reduction='none')
    batched=[];cursor=0;batch_members=[]
    for batch_number,batch in enumerate(raw['batches'],1):
        file=bind(batch['file'],batch['sha256'],ledger);value=torch.load(file,map_location='cpu',weights_only=True)
        count=len(batch['sids'])
        need(count==min(16,108-cursor) and value['schema_version']==1
             and value['sids']==batch['sids']==sids[cursor:cursor+count]
             and value['logits'].dtype==torch.float16 and torch.equal(value['logits'],logits[cursor:cursor+count])
             and tensor_info(torch,value['logits'])==batch['logits'],'Original seven-batch raw ownership differs')
        batched.append(torch.nn.functional.cross_entropy(value['logits'].float(),targets[cursor:cursor+count],reduction='none'))
        batch_members += [batch_number]*count;cursor+=count
    need(cursor==108,'Incomplete original batch coverage');cpu_batches=torch.cat(batched)
    promoted=logits.double();centered=promoted-promoted.amax(-1,keepdim=True)
    reference=torch.logsumexp(centered,dim=-1)-centered[torch.arange(108),targets]
    need(all(bool(torch.isfinite(x).all()) and bool((x>=0).all()) for x in (cpu_full,cpu_batches,reference)),
         'Nonfinite/negative recomputed NLL')
    records=[]
    for i,(row,saved) in enumerate(zip(rows,predictions)):
        correct=int(ids[i])==row['first_token_id'];expected=dict(row,argmax_id=int(ids[i]),first_token_correct=correct)
        mismatches=[]
        for key in sorted(set(expected)|(set(saved)-{'nll'})):
            if key not in expected or key not in saved or type(expected.get(key)) is not type(saved.get(key)) or expected.get(key)!=saved.get(key):
                mismatches.append(dict(field=key,expected=expected.get(key),recorded=saved.get(key),
                    expected_type=type(expected.get(key)).__name__,recorded_type=type(saved.get(key)).__name__))
        gpu=saved['nll'];need(type(gpu) in (int,float) and math.isfinite(gpu) and gpu>=0,'Saved GPU NLL invalid')
        cpu=float(cpu_full[i]);batch=float(cpu_batches[i]);ref=float(reference[i])
        comparisons=dict(cpu_full_vs_gpu=difference(cpu,gpu),cpu_original_batch_vs_gpu=difference(batch,gpu),
            cpu_full_vs_original_batch=difference(cpu,batch),gpu_vs_fp64=difference(gpu,ref),
            cpu_full_vs_fp64=difference(cpu,ref),cpu_original_batch_vs_fp64=difference(batch,ref))
        records.append(dict(index=i,sid=row['sid'],original_batch=batch_members[i],target_id=row['first_token_id'],
            raw_argmax_id=int(ids[i]),metadata_differences=mismatches,
            argmax_exact=type(saved.get('argmax_id')) is int and saved['argmax_id']==int(ids[i]),
            correctness_exact=type(saved.get('first_token_correct')) is bool and saved['first_token_correct']==correct,
            gpu_recorded_fp32=gpu,gpu_value_exactly_representable_fp32=float32(gpu)==gpu,
            cpu_fp32_full108=cpu,cpu_fp32_original_batch=batch,fp64_stable_reference=ref,
            original_nll_close=original_close(cpu,gpu),original_assertion_pass=not mismatches and original_close(cpu,gpu),
            comparisons=comparisons))
    save(out/'rows.json',records)
    changed=[r for r in records if r['metadata_differences'] or any(c['absolute']!=0 for c in r['comparisons'].values())]
    save(out/'differences.json',changed)
    save(out/'input_bindings.json',ledger)
    aggregations={}
    for name in records[0]['comparisons']:
        values=[r['comparisons'][name] for r in records]
        aggregations[name]=dict(nonexact_rows=sum(v['absolute']!=0 for v in values),
            maximum_absolute=max(v['absolute'] for v in values),maximum_fp32_ulp_distance=max(v['fp32_ulp_distance'] for v in values),
            maximum_absolute_in_second_fp32_ulps=max(v['absolute_in_second_fp32_ulps'] for v in values))
    metadata=sum(bool(r['metadata_differences']) for r in records);argmax=sum(not r['argmax_exact'] for r in records)
    correctness=sum(not r['correctness_exact'] for r in records);failures=sum(not r['original_assertion_pass'] for r in records)
    result=dict(passed=True,completed=True,protocol=PROTOCOL,source_sha256=sources,rows=108,original_batches=7,
        plan_file=str(PLAN),plan_sha256=PLAN_SHA,run_summary_file=str(RUN),run_summary_sha256=RUN_SHA,
        original_failure_file=str(FAILURE),original_failure_sha256=FAILURE_SHA,original_report_remains_failed=True,
        raw_file=str(raw_path),raw_sha256=summary['raw_sha256'],raw_tensor=tensor_info(torch,logits),
        metadata_mismatch_rows=metadata,argmax_mismatch_rows=argmax,correctness_mismatch_rows=correctness,
        original_assertion_failure_rows=failures,original_nll_failure_rows=sum(not r['original_nll_close'] for r in records),
        all_recorded_gpu_values_exact_fp32=all(r['gpu_value_exactly_representable_fp32'] for r in records),
        observed_mismatches_confined_to_nll=metadata==argmax==correctness==0 and failures>0,
        comparisons=aggregations,original_comparison=dict(atol=2e-6,rtol=2e-6,unchanged=True),
        fp64_reference='logsumexp(raw_logits.double()-row_max) - (target_logit.double()-row_max)',
        rows_file=str(out/'rows.json'),rows_sha256=sha(out/'rows.json'),
        differences_file=str(out/'differences.json'),differences_sha256=sha(out/'differences.json'),
        input_bindings_file=str(out/'input_bindings.json'),input_bindings_sha256=sha(out/'input_bindings.json'),
        self_test=self_test(),no_acceptance_or_tolerance_change=True,no_tensor_parameters_loaded=True,
        counters=dict(vlm=0,vision=0,norm=0,head=0,optimizer=0),elapsed_seconds=time.perf_counter()-started)
    (out/'REPORT.md').write_text('# Preserved readout rescore diagnosis\n\n'
        f'All108 rows compared; metadata mismatches {metadata}, argmax mismatches {argmax}, correctness mismatches {correctness}. '
        f'The unchanged original assertion rejects {failures} rows.\n\n'
        'All CPU full108/original-seven-batch FP32 losses, GPU-recorded FP32 losses and stable FP64 reference differences '
        'are retained. FP32 ULP distances round the FP64 reference only for that descriptive distance. '
        'This diagnostic does not accept the original report, change a tolerance, run a model/head, or refit anything.\n')
    save(out/'summary.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true',required=True);parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu'
         and not os.environ.get('SLURM_JOB_GPUS') and int(os.environ.get('SLURM_GPUS_ON_NODE','0'))==0,'CPU Slurm only')
    out=BASE/f'rescore_diagnosis_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    own={name:sha(ROOT/name) for name in OWN};save(out/'request.json',dict(protocol=PROTOCOL,own_source_sha256=own,
        plan_file=str(PLAN),plan_sha256=PLAN_SHA,run_summary_file=str(RUN),run_summary_sha256=RUN_SHA,
        failure_file=str(FAILURE),failure_sha256=FAILURE_SHA))
    try:
        result=diagnose(out,started)
        need({name:sha(ROOT/name) for name in OWN}==own,'Diagnostic source changed during execution')
        print(json.dumps(dict(passed=result['passed'],directory=str(out))),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),own_source_sha256=own,
            elapsed_seconds=time.perf_counter()-started,partial_outputs_retained=True));raise


if __name__=='__main__':main()

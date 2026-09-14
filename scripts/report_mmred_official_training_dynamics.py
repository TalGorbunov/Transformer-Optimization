"""Posthoc scalar-only training dynamics; no tensor, model or accuracy access.

CE includes the complete structural JSON and EOS target. Reweighting logged
mean CE by target length is descriptive and is not an independent loss replay.
Fixed final-epoch halves do not establish convergence or capacity.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
REPO=Path(__file__).resolve().parents[1]
PROTOCOL='mmred_official_training_scalar_dynamics'
OUT=REPO/'outputs/native_aggregation_vlm/mmred_official_training_dynamics'
MAIN=REPO/'outputs/native_aggregation_vlm/mmred_official_native_training'
RELEASES={'source_release.json':'f9078bf290049cb970d54913375fb792b3d5563d5c3a29064722269340ab19cd',
    'audit_source_release.json':'a3afe0815bdbf2aedf8fb8e0410bbdd6db4ac8c73ddffaf64f54325ab7a09cb8'}
OWN=('scripts/report_mmred_official_training_dynamics.py','slurm/mmred_official_training_dynamics.sbatch')
ARMS=('ordinary','normalized','mass');LENGTHS=(1,2,4,8,16)
TASKS=('char_at_frame','steps_in_room','spend_together','where_spend')
POLICY=dict(cpu_seconds=60,cpu_cores=4,memory_gib=4,arms=list(ARMS),training_records_per_arm=12000,
    optimizer_records_per_arm=1500,epochs=3,scenes_per_epoch=4000,
    final_epoch_halves=[[8000,10000],[10000,12000]],quantiles=[0,.25,.5,.75,.9,.95,.99,1],quantile_method='type7',
    clip_fraction_definition='logged pre-clip total norm > 1',
    no_tensor_files_opened=True,no_model_or_optimizer_execution=True,no_accuracy_recomputed=True,
    posthoc_descriptive_only=True,no_retuning_or_followup_release=True)


def need(ok,message):
    if not ok:raise ValueError(message)


def check_time(started):need(time.perf_counter()-started<60,'Fixed60-second CPU scalar diagnostic cap exceeded')


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):value.update(block)
    return value.hexdigest()


def save(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)


def bind(path,bindings,expected=None):
    path=Path(path).resolve()
    need(path.suffix in ('.json','.jsonl','.py','.md','.sbatch'), 'Only declared text metadata/source/log files may be opened')
    digest=sha(path);need(expected is None or digest==expected,'Scalar diagnostic bound input changed: '+str(path))
    bindings[str(path)]=digest;return path


def read(path,bindings,expected=None):return json.loads(bind(path,bindings,expected).read_text())


def expected_order():
    rng=random.Random(24);result=[]
    for epoch in range(3):
        indices=list(range(4000));rng.shuffle(indices)
        result.extend(dict(micro_index=epoch*4000+j,epoch=epoch,index=index,updates_before=(epoch*4000+j)//8)
            for j,index in enumerate(indices))
    return result


def audit_metadata(path,bindings,releases):
    path=Path(path).resolve();summary=read(path,bindings)
    release=releases['audit_source_release.json']
    need(summary['protocol']=='mmred_official_native_training_independent_audit'
         and summary['passed'] is summary['completed'] is True and summary['arm'] in ARMS
         and summary['source_sha256']==release['source_sha256'] and summary['inherited_source_sha256']==release['inherited_source_sha256']
         and not (path.parent/'failure.json').exists(),'Complete frozen computational main audit required')
    analysis=read(summary['analysis_file'],bindings,summary['analysis_sha256'])
    need(analysis['passed'] and analysis['original_run_passed'] and analysis['completed']
         and analysis['all_scheduled_numerical_evidence_collected'] and analysis['arm']==summary['arm'] and analysis['run']==summary['run']
         and analysis['scalar_trace_audit']['passed'] and analysis['scalar_trace_audit']['examples']==12000
         and analysis['scalar_trace_audit']['updates']==1500,'Complete audited scalar-training lineage required')
    manifest=read(summary['input_bindings_file'],bindings,summary['input_bindings_sha256'])
    read(summary['artifacts_file'],bindings,summary['artifacts_sha256'])
    for name,digest in release['source_sha256'].items():
        bind(REPO/name,bindings,digest);bind(path.parent/'source'/name.replace('/','_'),bindings,digest)
    run=analysis['run']
    def owned(file,digest=None):
        key=str(Path(file).resolve());need(key in manifest and (digest is None or manifest[key]==digest),'Text input absent from passed numerical audit')
        return read(key,bindings,manifest[key])
    config=owned(run['config_file'],run['config_sha256']);original=owned(run['analysis_file'],run['analysis_sha256'])
    plan=owned(run['plan_file'],run['plan_sha256']);main_release=releases['source_release.json']
    need(config['protocol']=='mmred_official_native_training' and config['arm']==summary['arm']
         and config['source_sha256']==main_release['source_sha256'] and config['inherited_source_sha256']==main_release['inherited_source_sha256']
         and config['plan_file']==run['plan_file'] and config['plan_sha256']==run['plan_sha256']
         and original['arm']==summary['arm'] and original['completed'] and original['passed']
         and original['final_checkpoint']==analysis['final_checkpoint'] and original['final_checkpoint']['updates']==1500,
         'Only original complete final main runs may be described')
    for name,digest in main_release['source_sha256'].items():
        bind(REPO/name,bindings,digest);bind(Path(run['directory'])/'source'/name.replace('/','_'),bindings,digest)
    compact=plan['compact_stage'];compact_summary=owned(compact['file'],compact['sha256'])
    need(compact_summary['passed'] and compact_summary['plan_file']==compact['plan_file']
         and compact_summary['plan_sha256']==compact['plan_sha256'],'Transitive compact-plan metadata binding differs')
    data_plan=read(compact['plan_file'],bindings,compact['plan_sha256'])
    rows=owned(data_plan['rows_file']);cases=owned(data_plan['cases_file']);order=owned(plan['order_file'])
    need(len(rows)==len(cases)==4000 and len({r['sid'] for r in rows})==4000
         and Counter((r['n'],r['qtype']) for r in rows)==Counter({(n,q):200 for n in LENGTHS for q in TASKS})
         and order==expected_order(),'Bound original4000 cohort or fixed three-epoch order differs')
    for index,(row,case) in enumerate(zip(rows,cases)):
        need(row['pilot_role']=='train' and case['index']==index and all(case[k]==row[k] for k in ('sid','n','qtype'))
             and case['target_rows']==len(case['target_ids']) and case['target_ids'][-1]==151645,'Bound training target/row metadata differs')
    need(sum(c['target_rows'] for c in cases)==29555,'Original full-JSON/EOS target inventory differs')
    files={}
    for field in ('training_trace','updates'):
        file=original[field+'_file'];digest=original[field+'_sha256']
        need(str(Path(file).resolve()) in manifest and manifest[str(Path(file).resolve())]==digest,'Scalar trace hash absent from passed audit')
        files[field]=dict(file=file,sha256=digest)
    descriptor=dict(file=str(path),sha256=bindings[str(path)],analysis_file=summary['analysis_file'],analysis_sha256=summary['analysis_sha256'],
        main_run=run,scalar_files=files,original_rows_file=data_plan['rows_file'],original_rows_sha256=bindings[str(Path(data_plan['rows_file']).resolve())],
        order_file=plan['order_file'],order_sha256=bindings[str(Path(plan['order_file']).resolve())],checkpoint_identity_recorded_only=original['final_checkpoint'])
    return summary['arm'],descriptor,rows,cases,order


def json_lines(descriptor,bindings,started):
    path=Path(descriptor['file']).resolve();need(path.suffix=='.jsonl','Only scalar JSONL traces may be streamed')
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for index,line in enumerate(stream):
            if index%128==0:check_time(started)
            digest.update(line);yield index,json.loads(line)
    actual=digest.hexdigest();need(actual==descriptor['sha256'],'Immutable scalar trace SHA differs')
    bindings[str(path)]=actual


def summarize(values):
    need(values,'Every descriptive cell requires observations')
    count=len(values);tokens=sum(L for _,L in values)
    return dict(scenes=count,target_tokens=tokens,mean_scene_ce=math.fsum(ce for ce,_ in values)/count,
        token_weighted_ce=math.fsum(ce*L for ce,L in values)/tokens)


def quantile(values,probability):
    ordered=sorted(values);h=(len(ordered)-1)*probability;j=math.floor(h)
    return ordered[j] if j==len(ordered)-1 else ordered[j]+(h-j)*(ordered[j+1]-ordered[j])


def norm_summary(values):
    need(values and all(math.isfinite(v) and v>=0 for v in values),'Finite nonnegative logged total optimizer norms required')
    return dict(updates=len(values),norm_greater_than_clip_threshold=sum(v>1 for v in values),
        clip_fraction=sum(v>1 for v in values)/len(values),clip_threshold=1.,
        quantiles={str(p):quantile(values,p) for p in POLICY['quantiles']},mean_norm=math.fsum(values)/len(values),
        logged_preclip_total_norm=True,subgroup_gradient_norms_available=False)


def analyze(arm,descriptor,rows,cases,order,bindings,started):
    cells=defaultdict(list);halves=defaultdict(list);seen=0
    for index,value in json_lines(descriptor['scalar_files']['training_trace'],bindings,started):
        need(index<12000,'Extra training scalar row');entry=order[index];row=rows[entry['index']];case=cases[entry['index']]
        need(all(value[k]==v for k,v in entry.items()) and value['arm']==arm and value['sid']==row['sid'] and value['n']==row['n']
             and value['target_rows']==case['target_rows'] and type(value['mean_ce']) in (int,float)
             and math.isfinite(value['mean_ce']) and value['mean_ce']>=0 and value['backward_loss']==value['mean_ce']/8
             and value['gradients_all_present'] and value['gradients_all_finite'] and value['layer_forward_calls']==[1]*28,
             'Logged full-JSON mean CE, order or finite-gradient flag differs')
        pair=(value['mean_ce'],value['target_rows']);cells[(entry['epoch'],row['n'],row['qtype'])].append(pair)
        if entry['epoch']==2:halves[((index-8000)//2000,row['n'],row['qtype'])].append(pair)
        seen+=1
    need(seen==12000 and len(cells)==60 and all(len(v)==200 for v in cells.values()),'Complete three-epoch balanced scalar trace required')
    cell_rows=[dict(arm=arm,epoch=e+1,n=n,qtype=q,**summarize(cells[(e,n,q)])) for e in range(3) for n in LENGTHS for q in TASKS]
    half_rows=[dict(arm=arm,epoch=3,half=h+1,n=n,qtype=q,**summarize(halves[(h,n,q)])) for h in (0,1) for n in LENGTHS for q in TASKS]
    need(sum(v['scenes'] for v in half_rows)==4000 and all(sum(v['scenes'] for v in half_rows if v['half']==h)==2000 for h in (1,2)),
         'Fixed final-epoch half inventory differs')
    last_first=[dict(arm=arm,n=n,qtype=q,**{metric:summarize(cells[(2,n,q)])[metric]-summarize(cells[(0,n,q)])[metric]
        for metric in ('mean_scene_ce','token_weighted_ce')}) for n in LENGTHS for q in TASKS]
    balanced=[]
    for half in (1,2):
        for length in (*LENGTHS,'all_lengths'):
            values=[v for v in half_rows if v['half']==half and (length=='all_lengths' or v['n']==length)]
            balanced.append(dict(arm=arm,half=half,n=length,cells=len(values),scenes=sum(v['scenes'] for v in values),
                mean_scene_ce=math.fsum(v['mean_scene_ce'] for v in values)/len(values),
                token_weighted_ce=math.fsum(v['token_weighted_ce'] for v in values)/len(values),
                weighting='equal N-by-task cells' if length=='all_lengths' else 'equal tasks at fixed N'))
    half_differences=[dict(arm=arm,n=n,**{metric:next(v[metric] for v in balanced if v['n']==n and v['half']==2)-
        next(v[metric] for v in balanced if v['n']==n and v['half']==1) for metric in ('mean_scene_ce','token_weighted_ce')}) for n in (*LENGTHS,'all_lengths')]
    norms=defaultdict(list);seen=0;parameters=224 if arm=='ordinary' else 227
    for index,value in json_lines(descriptor['scalar_files']['updates'],bindings,started):
        update=index+1;need(update<=1500 and value['update']==update and value['last_micro_index']==update*8-1
            and value['clip_norm']==1. and value['lr']==2e-4 and value['betas']==[.9,.999] and value['eps']==1e-8 and value['weight_decay']==.01
            and value['parameter_steps']==[float(update)]*parameters and math.isfinite(value['gradient_norm']) and value['gradient_norm']>=0,
            'Fixed1500 update recipe or logged total norm differs')
        norms[index//500].append(value['gradient_norm']);seen+=1
    need(seen==1500 and set(norms)=={0,1,2} and all(len(v)==500 for v in norms.values()),'Complete optimizer scalar trace required')
    return dict(arm=arm,epoch_length_task=cell_rows,epoch3_minus_epoch1=last_first,final_epoch_halves=half_rows,
        task_balanced_final_epoch_halves=balanced,task_balanced_second_minus_first_half=half_differences,
        optimizer_norms=dict(all_updates=norm_summary([v for group in norms.values() for v in group]),
            by_epoch={str(e+1):norm_summary(norms[e]) for e in range(3)}),
        training_scalar_rows=12000,optimizer_scalar_rows=1500,independent_loss_reexecution=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--audits',nargs=3,type=Path,required=True);args=ap.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS')
         and int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==4,'CPU-only four-core Slurm allocation required')
    started=time.perf_counter();out=OUT/f'report_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False);bindings={};progress={}
    own={name:sha(REPO/name) for name in OWN};(out/'source').mkdir()
    for name,digest in own.items():
        file=out/'source'/name.replace('/','_');file.write_bytes((REPO/name).read_bytes());need(sha(file)==digest,'Scalar diagnostic source archive changed')
    save(out/'request.json',dict(protocol=PROTOCOL,policy=POLICY,audits=[str(p.resolve()) for p in args.audits],source_sha256=own))
    try:
        releases={name:read(MAIN/name,bindings,digest) for name,digest in RELEASES.items()};descriptors={};result={};common=None
        for path in args.audits:
            check_time(started);arm,descriptor,rows,cases,order=audit_metadata(path,bindings,releases)
            need(arm not in result,'Repeated scalar diagnostic arm');identity={k:descriptor[k] for k in ('original_rows_file','original_rows_sha256','order_file','order_sha256')}
            need(common is None or identity==common,'All three arms require identical original rows/order');common=identity
            descriptors[arm]=descriptor;result[arm]=analyze(arm,descriptor,rows,cases,order,bindings,started);progress['completed_arms']=list(result)
        need(set(result)==set(ARMS),'All three completed main arms required')
        csv_fields=('arm','epoch','half','n','qtype','scenes','target_tokens','mean_scene_ce','token_weighted_ce')
        with (out/'loss_cells.csv').open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=csv_fields);writer.writeheader()
            for arm in ARMS:
                writer.writerows(result[arm]['epoch_length_task']);writer.writerows(result[arm]['final_epoch_halves'])
        analysis=dict(protocol=PROTOCOL,passed=True,completed=True,policy=POLICY,source_sha256=own,audits=descriptors,arms=result,
            scalar_training_rows=36000,scalar_optimizer_rows=4500,inputs=common,no_tensor_files_opened=True,no_accuracy_recomputed=True,
            limitations=['Posthoc descriptive summaries only.','CE includes structural JSON tokens and native EOS.',
                'Token-weighted CE reweights logged per-scene mean CE; it does not replay token losses.',
                'Fixed final-epoch halves are different shuffled examples, not a convergence or capacity test.',
                'Norms are whole accumulated optimizer gradients; task/length subgroup norms are not stored.',
                'Clip fraction means logged total pre-clip norm greater than one; no epsilon-scale clipping claim.',
                'No retuning, validation/test analysis, or follow-up computation is authorized.'])
        save(out/'analysis.json',analysis);save(out/'input_bindings.json',bindings)
        (out/'REPORT.md').write_text('Posthoc scalar-only training dynamics.\n\nAll36,000 logged full-JSON/EOS scene mean losses and4,500 optimizer norm records were bound to passed main audits and the original fixed order. No tensors were opened, no accuracy was recomputed, and these descriptive differences do not establish convergence or capacity. See analysis.json and loss_cells.csv.\n')
        artifacts={str(f):sha(f) for f in out.iterdir() if f.is_file()};save(out/'artifacts.json',artifacts);check_time(started)
        need(own=={name:sha(REPO/name) for name in OWN},'Diagnostic source changed during execution')
        save(out/'summary.json',dict(protocol=PROTOCOL,passed=True,completed=True,policy=POLICY,source_sha256=own,
            analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),input_bindings_file=str(out/'input_bindings.json'),
            input_bindings_sha256=sha(out/'input_bindings.json'),artifacts_file=str(out/'artifacts.json'),artifacts_sha256=sha(out/'artifacts.json'),
            elapsed_seconds=time.perf_counter()-started,no_tensor_files_opened=True,no_retuning_or_followup_release=True))
    except BaseException as exc:
        save(out/'failure_input_bindings.json',bindings)
        save(out/'failure.json',dict(protocol=PROTOCOL,passed=False,type=type(exc).__name__,message=str(exc),
            elapsed_seconds=time.perf_counter()-started,progress=progress,input_bindings_file=str(out/'failure_input_bindings.json'),
            input_bindings_sha256=sha(out/'failure_input_bindings.json'),no_retuning_or_followup_release=True));raise


if __name__=='__main__':main()

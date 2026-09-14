"""Saved-code SiLU geometry only: no core/head/model call, gradient or fit."""
from __future__ import annotations
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts.stage_native_vision_v6_teacher import need,read,save,sha,object_sha
OUT=REPO/'outputs/native_aggregation_vlm/identity_join_code_readout_geometry'
DATA=Path('/mnt/data/gabriele/gnn_transformer/identity_join_code_readout_geometry')
PROTOCOL='identity_join_saved_code_readout_geometry'
PROPOSAL='docs/paper/NATIVE_AGGREGATION_CODE_READOUT_GEOMETRY_PROPOSAL.md'
OWN=('scripts/analyze_native_identity_join_code_readout_geometry.py','slurm/native_identity_join_code_readout_geometry.sbatch',PROPOSAL)
SPECS={
 'original_order':dict(report='identity_join_orientation_joint_code/report_443718',analysis_sha256='ad46c2472cdc7b6a5791613c478a0fec71c950f7fd0b18de4ea9c6a9cfa41a31',
    failure_sha256='a21a18ae2febcc3da3fa5c5dfb19f2c8cb032d7b35361b018384d2e24304143b',run_id='run_443716'),
 'paired_order':dict(report='identity_join_orientation_paired_code/report_443742',analysis_sha256='32391ac1e17084d915b209e28b0d448a4789f29f043423019a7b6b9dd5b8d502',
    summary_sha256='951130b304f5f21b7750d42ddad9b87289514a04fb26bf72acd2236b8d505ad0',run_id='run_443735')}
STEPS=(1,2,32,128,300,600,2000,6000)
POLICY=dict(cpu_seconds=90,cpu_cores=4,memory_gib=16,runs=2,captures=44,training_captures_per_run=8,final_captures_per_run=14,
    final_first_queries_per_run=216,small_absolute_slope=0.01,small_absolute_curvature=0.01,near_unit_slope_distance=0.01,
    saved_preactivation_derivatives=True,no_counterfactual_scores=True,no_head_or_core_calls=True,no_fit_or_GPU=True,no_scientific_gate_change=True)


def bind(path,bindings,expected=None):
    path=Path(path).resolve();digest=sha(path);need(expected is None or digest==expected,'Changed geometry input: '+str(path))
    need(str(path) not in bindings or bindings[str(path)]==digest,'Conflicting geometry input');bindings[str(path)]=digest;return digest


def sources():return {name:sha(REPO/name) for name in OWN}


def inherited_sources():
    result={}
    for spec in SPECS.values():
        file=REPO/'outputs/native_aggregation_vlm'/spec['report']/'analysis.json';need(sha(file)==spec['analysis_sha256'],'Fixed geometry report changed');a=read(file)
        for key in ('source_sha256','inherited_source_sha256'):
            for name,digest in a[key].items():
                need(name not in result or result[name]==digest,'Conflicting geometry source');result[name]=digest
    for name,digest in result.items():need(sha(REPO/name)==digest,'Inherited source changed: '+name)
    return result


def snapshot(out):
    own=sources();inherited=inherited_sources();(out/'source').mkdir()
    for name,digest in own.items():
        file=out/'source'/name.replace('/','_');file.write_bytes((REPO/name).read_bytes());need(sha(file)==digest,'Geometry source copy differs')
    save(out/'source_hashes.json',own);save(out/'inherited_sources.json',dict(source_sha256=inherited));return own,inherited


def derivatives(torch,a):
    probability=torch.sigmoid(a);common=probability*(1-probability)
    return probability+a*common,common*(2+a*(1-2*probability))


def self_test(torch):
    a=torch.tensor([-8.,-2.,0.,2.,8.],dtype=torch.float64);slope,curvature=derivatives(torch,a)
    need(float(slope[2])==.5 and float(curvature[2])==.5,'SiLU origin derivative differs')
    need(torch.allclose(slope+slope.flip(0),torch.ones_like(slope),atol=1e-14,rtol=1e-14)
         and torch.allclose(curvature,curvature.flip(0),atol=1e-14,rtol=1e-14),'SiLU derivative symmetry differs')
    eps=1e-4;f=lambda x:x*torch.sigmoid(x)
    finite_slope=(f(a+eps)-f(a-eps))/(2*eps);finite_curvature=(f(a+eps)-2*f(a)+f(a-eps))/(eps*eps)
    need(torch.allclose(slope,finite_slope,atol=1e-8,rtol=1e-8) and torch.allclose(curvature,finite_curvature,atol=1e-6,rtol=1e-6),'Independent finite-difference fixture differs')
    return dict(passed=True,origin=True,symmetry=True,finite_difference=True,no_model_or_head=True)


def run_inputs(label,spec,bindings):
    root=REPO/'outputs/native_aggregation_vlm'/spec['report'];file=root/'analysis.json';bind(file,bindings,spec['analysis_sha256']);a=read(file)
    need(a['completed'] is True and a['primary_endpoint_step']==6000 and set(a['runs'])=={'joint_code'},'Complete fixed code report required')
    if label=='original_order':
        bind(root/'failure.json',bindings,spec['failure_sha256']);failure=read(root/'failure.json')
        need(a['passed'] is False and a['all_numerical_audits_collected'] is True and not (root/'summary.json').exists()
             and failure['analysis_sha256']==spec['analysis_sha256'],'Original failed numerical report must remain failed')
    else:
        bind(root/'summary.json',bindings,spec['summary_sha256']);summary=read(root/'summary.json')
        need(a['passed'] is True and summary['passed'] is True and summary['completed'] is True and summary['analysis_sha256']==spec['analysis_sha256'],'Paired computational report differs')
    for name,digest in a['source_sha256'].items():bind(root/'source'/name.replace('/','_'),bindings,digest)
    bind(a['plan_file'],bindings,a['plan_sha256']);plan=read(a['plan_file'])
    run=a['runs']['joint_code'];directory=Path(run['run_directory']);need(directory.name==spec['run_id'] and not run['first_token_fit']['passed'],'Fixed negative code fit differs')
    bind(directory/'summary.json',bindings,run['run_summary_sha256']);summary=read(directory/'summary.json')
    need(summary['passed'] is True and summary['completed'] is True and summary['plan_sha256']==a['plan_sha256'],'Completed GPU source run required')
    bind(summary['captures_file'],bindings,summary['captures_sha256']);records=read(summary['captures_file'])
    for key in ('rows','scenes','order'):bind(plan[key+'_file'],bindings,plan['runtime_bindings'][plan[key+'_file']])
    rows=read(plan['rows_file']);scenes=read(plan['scenes_file']);order=read(plan['order_file'])
    need(len(rows)==len(scenes)==216 and len(order)==48000 and len(records)==len(run['capture_audits'])==22,'Fixed cohort/capture count differs')
    need([(r['phase'],r['step'],r['endpoint_step']) for r in records]==[('training',s,None) for s in STEPS]+[('evaluation',s,6000) for s in range(1,15)],'Capture chronology differs')
    audits={(c['phase'],c['step']):c for c in run['capture_audits']}
    for record in records:
        audit=audits[(record['phase'],record['step'])]
        need(record['file']==audit['capture_file'] and record['sha256']==audit['capture_sha256'] and record['weights_file']==audit['weights_file']
             and record['weights_sha256']==audit['weights_sha256'] and record['sids']==audit['sids'] and audit['fixed_local_codes_exact'] is True
             and audit['fp16_cast_before_add_exact'] is True and all(v['passed'] for v in audit['functional_core'].values()) and all(v['passed'] for v in audit['nll']),
             'Saved geometry source/core/NLL audit differs')
    return plan,rows,scenes,order,records,dict(analysis_file=str(file),analysis_sha256=spec['analysis_sha256'],original_numerical_report_passed=a['passed'],
        run_directory=str(directory),run_summary_sha256=run['run_summary_sha256'],first_token_fit=run['first_token_fit'])


def query_labels(record,packet,rows,scenes,order,label):
    training=record['phase']=='training';step=record['step'];by_sid={r['sid']:r for r in rows}
    expected_sids=[sid for entry in order[(step-1)*8:step*8] for sid in entry['sids']] if training else [r['sid'] for r in rows[(step-1)*16:step*16]]
    need(record['sids']==packet['sids']==expected_sids,'Query SID order differs')
    layout=packet['layout'];targets=[scenes[sid]['target_ids'] for sid in expected_sids]
    full=layout if training else layout['original_full_layout'];offsets=[0]
    for sequence in targets:offsets.append(offsets[-1]+len(sequence))
    need(full['target_sequences']==targets and full['offsets']==offsets and full['targets']==[t for ids in targets for t in ids]
         and full['prefixes']==[[ids[:j] for j in range(len(ids))] for ids in targets],'Strict-prefix layout differs')
    if not training:need(layout['first_query_only'] is True and layout['full_target_ids']==targets and layout['target_ids']==[ids[0] for ids in targets]
         and layout['first_indices']==offsets[:-1],'Final first-query layout differs')
    result=[]
    for occurrence,(sid,ids) in enumerate(zip(expected_sids,targets)):
        row=by_sid[sid];need(ids==row['target_ids'] and len(ids) in (2,3) and ids[-1]==151645,'Full native target differs')
        for position in range(len(ids)) if training else (0,):
            role='first_token' if position==0 else 'EOS' if position==len(ids)-1 else 'name_continuation'
            result.append(dict(run=label,phase=record['phase'],capture_step=step,weights_step=record['weights_step'],endpoint_step=record['endpoint_step'],
                sid=sid,scene_occurrence=occurrence,query_index=len(result),target_position=position,target_id=ids[position],strict_prefix=ids[:position],
                role=role,full_target_length=len(ids),question=row['question'],orientation_version=row['orientation_version'],n_frames=row['n_frames'],base_contrast_id=row['base_contrast_id']))
    return result


def metrics(torch,q,z,b,a,slope,curvature):
    def norm(x):return float(x.norm())
    def cosine(x,y):
        denominator=norm(x)*norm(y);return None if denominator==0 else float(torch.dot(x,y))/denominator
    reconstructed=q+z+b;error=(a-reconstructed).abs()
    return dict(query_norm=norm(q),aggregate_projection_norm=norm(z),bias_norm=norm(b),aggregate_plus_bias_norm=norm(z+b),preactivation_norm=norm(a),
        query_aggregate_twice_dot=float(2*torch.dot(q,z)),query_bias_twice_dot=float(2*torch.dot(q,b)),aggregate_bias_twice_dot=float(2*torch.dot(z,b)),
        query_aggregate_cosine=cosine(q,z),query_bias_cosine=cosine(q,b),aggregate_bias_cosine=cosine(z,b),
        query_to_aggregate_norm=None if norm(z)==0 else norm(q)/norm(z),decomposition_max_absolute=float(error.max()),
        preactivation_min=float(a.min()),preactivation_max=float(a.max()),preactivation_mean=float(a.mean()),
        slope_min=float(slope.min()),slope_max=float(slope.max()),slope_mean=float(slope.mean()),slope_absolute_mean=float(slope.abs().mean()),
        curvature_min=float(curvature.min()),curvature_max=float(curvature.max()),curvature_absolute_mean=float(curvature.abs().mean()),
        small_absolute_slope_fraction=float((slope.abs()<=.01).double().mean()),small_absolute_curvature_fraction=float((curvature.abs()<=.01).double().mean()),
        near_unit_slope_fraction=float(((slope-1).abs()<=.01).double().mean()))


def grouped(rows):
    result=[]
    groups=defaultdict(list)
    for row in rows:
        labels=row['labels'];base=(labels['run'],labels['phase'],labels['capture_step'] if labels['phase']=='training' else 6000,labels['role'])
        groups[base+('all',None)].append(row['metrics'])
        if labels['phase']=='evaluation':
            for key in ('orientation_version','base_contrast_id','question','n_frames'):groups[base+(key,labels[key])].append(row['metrics'])
    for (run,phase,step,role,key,value),values in groups.items():
        names=values[0].keys();summary={}
        for name in names:
            observed=[r[name] for r in values if r[name] is not None]
            summary[name]=dict(defined_queries=len(observed),mean=None if not observed else sum(observed)/len(observed),minimum=None if not observed else min(observed),maximum=None if not observed else max(observed))
        result.append(dict(labels=dict(run=run,phase=phase,update=step,role=role,stratum=key,value=value),queries=len(values),metrics=summary))
    return result


def run(out,data,frozen,inherited,bindings):
    import torch
    torch.set_num_threads(4);save(out/'self_test.json',self_test(torch));all_rows=[];artifacts={};run_proofs={};final_counts={};identities=[];captured=0
    with torch.no_grad():
        for label,spec in SPECS.items():
            plan,rows,scenes,order,records,proof=run_inputs(label,spec,bindings);run_proofs[label]=proof;identities.append(plan['native_identity_sha256']);final_sids=[]
            for record in records:
                bind(record['file'],bindings,record['sha256']);bind(record['weights_file'],bindings,record['weights_sha256'])
                packet=torch.load(record['file'],map_location='cpu',weights_only=True);weights=torch.load(record['weights_file'],map_location='cpu',weights_only=True)
                need(packet['schema_version']==1 and packet['arm']=='joint_code' and all(packet[k]==record[k] for k in ('phase','step','endpoint_step')),'Capture metadata differs')
                training=record['phase']=='training'
                need(weights['step']==record['weights_step']==(record['step']-1 if training else 6000),'Actual pre-update/endpoint weights differ')
                if training:need(weights['optimizer_step']==record['step'] and weights['position']=='before_update' and weights['arm']=='joint_code','Pre-update weight ownership differs')
                labels=query_labels(record,packet,rows,scenes,order,label);cap=packet['capture'];w=weights['branch']
                need(set(w)=={'query.weight','aggregate_projection.weight','aggregate_projection.bias','up.weight'},'Stored readout class differs')
                for name in ('query','aggregate','preactivation'):
                    need(cap[name].dtype==torch.float32 and tuple(cap[name].shape)==(len(labels),96) and bool(torch.isfinite(cap[name]).all()),'Saved geometry tensor invalid: '+name)
                matrix=w['aggregate_projection.weight'];bias=w['aggregate_projection.bias']
                need(matrix.dtype==bias.dtype==torch.float32 and tuple(matrix.shape)==(96,96) and tuple(bias.shape)==(96,)
                     and bool(torch.isfinite(matrix).all()) and bool(torch.isfinite(bias).all()),'Saved aggregate projection differs')
                q=cap['query'].double();aggregate=cap['aggregate'].double();a=cap['preactivation'].double();b=bias.double();z=aggregate@matrix.double().T
                slope,curvature=derivatives(torch,a)
                need(bool(torch.isfinite(slope).all()) and bool(torch.isfinite(curvature).all()),'Nonfinite saved-point derivatives')
                capture_rows=[dict(labels=label_row,metrics=metrics(torch,q[i],z[i],b,a[i],slope[i],curvature[i])) for i,label_row in enumerate(labels)]
                need(all(math.isfinite(v) for row in capture_rows for v in row['metrics'].values() if v is not None),'Nonfinite geometry scalar')
                name=f"{label}_{record['phase']}_{record['step']:04d}";file=data/(name+'.pt')
                torch.save(dict(schema_version=1,capture_file=record['file'],capture_sha256=record['sha256'],weights_file=record['weights_file'],weights_sha256=record['weights_sha256'],
                    labels=labels,query=q,aggregate_projection=z,bias=b,actual_preactivation=a,silu_slope=slope,silu_curvature=curvature),file)
                artifacts[str(file)]=sha(file);json_file=out/(name+'.json');save(json_file,dict(rows=capture_rows));artifacts[str(json_file)]=sha(json_file)
                all_rows.extend(capture_rows);captured+=1
                if not training:final_sids.extend(record['sids'])
                del packet,weights,cap,w,q,aggregate,a,b,z,slope,curvature
            need(final_sids==[r['sid'] for r in rows] and len(set(final_sids))==216,'Complete fixed final cohort differs');final_counts[label]=len(final_sids)
    need(captured==44 and len(set(identities))==1 and sum(final_counts.values())==432,'Fixed run/capture/native scope differs')
    save(out/'rows.json',all_rows);save(out/'strata.json',grouped(all_rows));artifacts[str(out/'rows.json')]=sha(out/'rows.json');artifacts[str(out/'strata.json')]=sha(out/'strata.json')
    artifacts[str(out/'self_test.json')]=sha(out/'self_test.json')
    return dict(passed=True,completed=True,protocol=PROTOCOL,policy=POLICY,source_sha256=frozen,inherited_source_sha256=inherited,input_bindings=bindings,output_bindings=artifacts,
        runs=run_proofs,captures=captured,queries=len(all_rows),final_first_queries=final_counts,native_identity_sha256=identities[0],
        model_calls=0,core_calls=0,norm_calls=0,head_calls=0,backward_calls=0,optimizer_updates=0,
        original_failed_report_remains_failed=True,geometry_is_descriptive=True,no_new_efficiency_or_trainability_claim=True,no_release=True)


def main():
    need(len(sys.argv)==1 and os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_NAME')=='identity_join_code_readout_geometry'
        and os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS') and int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==4,'Fixed CPU Slurm geometry allocation required')
    started=time.perf_counter();name='geometry_'+os.environ['SLURM_JOB_ID'];out=OUT/name;data=DATA/name
    out.mkdir(parents=True,exist_ok=False);data.mkdir(parents=True,exist_ok=False);frozen,inherited=snapshot(out);bindings={}
    try:
        result=run(out,data,frozen,inherited,bindings);need(sources()==frozen and inherited_sources()==inherited,'Geometry source changed')
        result['elapsed_seconds']=time.perf_counter()-started;need(result['elapsed_seconds']<=90,'Fixed90-second geometry cap exceeded')
        save(out/'analysis.json',result)
        (out/'REPORT.md').write_text('# Saved local-code readout geometry\n\n'
            'Completed44 saved captures from the original-order and paired-order216-context code fits, including all432 final first queries. '
            'Reported query, saved aggregate projection, bias, preactivation and SiLU derivatives are descriptive; no altered-input score is computed. '
            'Small absolute curvature and small slope have separate fixed definitions. The original numerical failure remains failed. '
            'No core, norm, head, backbone, gradient or fit ran, and no scientific gate or subsequent release changes.\n')
        save(out/'summary.json',dict(passed=True,completed=True,protocol=PROTOCOL,source_sha256=frozen,inherited_source_sha256=inherited,
            analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),report_file=str(out/'REPORT.md'),report_sha256=sha(out/'REPORT.md'),
            captures=44,final_first_queries=result['final_first_queries'],no_head_or_core_calls=True,no_release=True,elapsed_seconds=result['elapsed_seconds']))
        print(json.dumps(dict(passed=True,directory=str(out))),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,inherited_source_sha256=inherited,input_bindings=bindings,
            elapsed_seconds=time.perf_counter()-started,partial_outputs_retained=True));raise


if __name__=='__main__':main()

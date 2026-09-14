"""CPU-only descriptive query/factor geometry from every frozen paired capture.

Removing q is a saved-preactivation counterfactual only: no downstream core,
native normalization, head, prediction, optimizer, or fitted condition is run.
"""
from pathlib import Path
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
REPO=Path(__file__).resolve().parents[1]
REPORT=REPO/'outputs/native_aggregation_vlm/identity_join_factor_binding/report_443383/summary.json'
REPORT_SHA='de38a1498115bc835955cf264a609ead1a095bacb0c3faa99eba10a1ae962238'
ANALYSIS_SHA='326db6392f648a517e169b206ca2c46b25b28f1d7f2faf3da31a314afcb74f94'
OUT=REPO/'outputs/native_aggregation_vlm/identity_join_factor_binding'
OWN=('scripts/analyze_native_identity_join_factor_query_geometry.py','slurm/native_identity_join_factor_query_geometry.sbatch')
STEPS=(1,2,32,128,300,600)


def need(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def bind(path,digest,ledger):
    path=Path(path).resolve();need(sha(path)==digest,'Changed bound input: '+str(path));ledger[str(path)]=digest
    return path


def cosine(a,b):
    denominator=float(a.norm()*b.norm())
    return None if denominator==0 else float((a*b).sum())/denominator


def geometry(torch,local,query,bias,preactivation,factor,*,interaction='additive',other_actual=None,other_removed=None):
    """FP64 tanh of actual FP32 preacts versus independently recomputed FP32 local+bias."""
    need(local.ndim==2 and local.shape==preactivation.shape==factor.shape and len(local)>0
         and query.shape==bias.shape==(local.shape[1],),'Invalid per-query factor layout')
    need(all(v.dtype==torch.float32 and bool(torch.isfinite(v).all()) for v in (local,query,bias,preactivation,factor)), 'Invalid finite FP32 geometry input')
    l=local.double();q=query.double();b=bias.double();mean=l.mean(0);common=q+b+mean;deviation=l-mean
    terms=dict(query_squared=float(q.square().sum()),bias_squared=float(b.square().sum()),local_mean_squared=float(mean.square().sum()),
        twice_query_bias=float(2*(q*b).sum()),twice_query_local_mean=float(2*(q*mean).sum()),twice_bias_local_mean=float(2*(b*mean).sum()))
    common_sq=float(common.square().sum());residual=abs(sum(terms.values())-common_sq)
    need(residual<=1e-10+1e-10*abs(common_sq),'FP64 common-vector signed identity failed')
    actual=torch.tanh(preactivation.double());without_query=local+bias
    removed=torch.tanh(without_query.double());da=1-actual.square();dr=1-removed.square()
    need(interaction in ('product','additive'),'Unknown interaction')
    if interaction=='product':
        need(other_actual.shape==other_removed.shape==factor.shape and other_actual.dtype==other_removed.dtype==torch.float64,'Other-factor derivative layout differs')
        payload_actual=da*other_actual;payload_removed=dr*other_removed
    else:payload_actual=.5*da;payload_removed=.5*dr
    actual_saturated=int((da<=.01).sum());removed_saturated=int((dr<=.01).sum())
    captured_saturated=int(((1-factor.double().square())<=.01).sum())
    centered=float(deviation.square().sum(-1).mean().sqrt())
    return dict(valid_items=len(local),coordinates=factor.numel(),query_norm=float(q.norm()),bias_norm=float(b.norm()),
        local_projection_mean_norm=float(mean.norm()),common_norm=float(common.norm()),
        local_projection_rms_norm=float(l.square().sum(-1).mean().sqrt()),centered_local_rms_norm=centered,
        common_to_centered_rms_ratio=None if centered==0 else float(common.norm())/centered,
        cosines=dict(query_bias=cosine(q,b),query_local_mean=cosine(q,mean),bias_local_mean=cosine(b,mean),query_common=cosine(q,common)),
        common_norm_squared=common_sq,signed_squared_norm_terms=terms,signed_identity_residual=residual,
        actual_saturated_coordinates=actual_saturated,query_removed_saturated_coordinates=removed_saturated,
        captured_factor_saturated_coordinates=captured_saturated,captured_factor_saturated_fraction=captured_saturated/factor.numel(),
        actual_tanh_derivative_absolute_mean=float(da.abs().mean()),query_removed_tanh_derivative_absolute_mean=float(dr.abs().mean()),
        actual_payload_partial_absolute_mean=float(payload_actual.abs().mean()),query_removed_payload_partial_absolute_mean=float(payload_removed.abs().mean()),
        actual_payload_partial_rms=float(payload_actual.square().mean().sqrt()),query_removed_payload_partial_rms=float(payload_removed.square().mean().sqrt()),
        actual_saturated_fraction=actual_saturated/factor.numel(),query_removed_saturated_fraction=removed_saturated/factor.numel(),
        query_removal_fraction_change=(removed_saturated-actual_saturated)/factor.numel(),
        stable_tanh_vs_saved_factor_max_absolute=float((actual-factor.double()).abs().max()),
        rounded_subtraction_vs_independent_query_removal_max_absolute=float(((preactivation-query.unsqueeze(0)).double()-without_query.double()).abs().max()))


def self_test(torch):
    q=torch.tensor([10.,0.]);b=torch.zeros(2);local=torch.tensor([[-10.,.1],[-10.,-.1]])
    pre=local+q;value=geometry(torch,local,q,b,pre,torch.tanh(pre))
    need(value['common_norm']==0 and value['actual_saturated_coordinates']==0
         and value['query_removed_saturated_coordinates']==2,'Large-query cancellation counterexample failed')
    local=torch.tensor([[.1,.1],[-.1,-.1]]);pre=local+q;value=geometry(torch,local,q,b,pre,torch.tanh(pre))
    need(value['actual_saturated_coordinates']==2 and value['query_removed_saturated_coordinates']==0,'Query saturation counterexample failed')
    q=torch.tensor([1.,2.]);b=torch.tensor([-3.,1.]);local=torch.tensor([[2.,-1.],[4.,3.]])
    value=geometry(torch,local,q,b,local+q+b,torch.tanh(local+q+b))
    need(value['signed_identity_residual']<=1e-12 and value['coordinates']==4 and cosine(torch.zeros(2),q) is None,'Signed decomposition/empty cosine fixture failed')
    zero=torch.zeros((2,2));v=geometry(torch,zero,torch.zeros(2),torch.zeros(2),zero,zero,
        interaction='product',other_actual=zero.double(),other_removed=zero.double())
    need(v['actual_tanh_derivative_absolute_mean']==1 and v['actual_payload_partial_absolute_mean']==0,
         'Unsaturated factors can still suppress product payload derivatives')
    return dict(passed=True,checks=4,large_query_does_not_imply_saturation=True,query_removal_can_increase_or_decrease_saturation=True,
        product_other_factor_can_suppress_local_jacobian=True)


def summarize(rows,labels):
    coordinates=sum(r['coordinates'] for r in rows)
    return dict(labels=labels,metrics=dict(query_factor_rows=len(rows),local_occurrences=sum(r['valid_items'] for r in rows),coordinates=coordinates,
        actual_saturated_fraction=sum(r['actual_saturated_coordinates'] for r in rows)/coordinates,
        query_removed_saturated_fraction=sum(r['query_removed_saturated_coordinates'] for r in rows)/coordinates,
        captured_factor_saturated_fraction=sum(r['captured_factor_saturated_coordinates'] for r in rows)/coordinates,
        actual_payload_partial_absolute_mean=sum(r['actual_payload_partial_absolute_mean']*r['coordinates'] for r in rows)/coordinates,
        query_removed_payload_partial_absolute_mean=sum(r['query_removed_payload_partial_absolute_mean']*r['coordinates'] for r in rows)/coordinates,
        actual_payload_partial_rms=math.sqrt(sum(r['actual_payload_partial_rms']**2*r['coordinates'] for r in rows)/coordinates),
        query_removed_payload_partial_rms=math.sqrt(sum(r['query_removed_payload_partial_rms']**2*r['coordinates'] for r in rows)/coordinates),
        mean_query_norm=sum(r['query_norm'] for r in rows)/len(rows),mean_common_norm=sum(r['common_norm'] for r in rows)/len(rows),
        mean_centered_local_rms_norm=sum(r['centered_local_rms_norm'] for r in rows)/len(rows),
        signed_identity_max_residual=max(r['signed_identity_residual'] for r in rows)))


def analyze(torch,out,summary,analysis,ledger):
    plan_path=bind(analysis['plan_file'],analysis['plan_sha256'],ledger);plan=read(plan_path)
    need(plan['source_sha256']==summary['source_sha256'] and analysis['source_sha256']==summary['source_sha256'], 'Plan/report source union differs')
    bind(plan['features_file'],plan['features_sha256'],ledger)
    bind(plan['stats_file'],plan['stats_sha256'],ledger)
    for key in ('rows_file','scenes_file'):
        bind(plan[key],plan['runtime_bindings'][str(Path(plan[key]).resolve())],ledger)
    metadata=read(plan['rows_file']);by_sid={r['sid']:r for r in metadata};scenes=read(plan['scenes_file'])
    need(len(by_sid)==len(scenes)==108 and len({r['question'] for r in metadata})==6,'Training ownership inventory differs')
    inventories=[];runs={}
    for arm in ('product','additive'):
        published=analysis['runs'][arm];directory=Path(published['run_directory'])
        run=read(bind(directory/'summary.json',published['run_summary_sha256'],ledger));runs[arm]=run
        need(run['passed'] is True and run['completed'] is True and run['arm']==arm and run['source_sha256']==summary['source_sha256'], 'Completed arm/source differs')
        bind(run['checkpoint'],published['checkpoint_sha256'],ledger)
        records=read(bind(run['captures_file'],run['captures_sha256'],ledger))
        chosen=[r for r in records if r['phase'] in ('training','evaluation')]
        need([(r['phase'],r['step']) for r in chosen]==[('training',x) for x in STEPS]+[('evaluation',x) for x in range(1,8)], 'All13 paired captures required')
        audit={(r['phase'],r['step']):r for r in published['capture_audits']}
        for record in chosen:
            proof=audit[record['phase'],record['step']]
            need(record['arm']==arm and record['pairing']=='paired' and proof['passed'] is True
                 and proof['capture_file']==record['file'] and proof['capture_sha256']==record['sha256']
                 and proof['weights_file']==record['weights_file'] and proof['weights_sha256']==record['weights_sha256'], 'Independent capture audit join differs')
            bind(record['file'],record['sha256'],ledger);bind(record['weights_file'],record['weights_sha256'],ledger)
            inventories.append(dict(record))
    need(len(inventories)==26,'Exactly26 paired captures required')
    save(out/'input_inventory.json',dict(plan_file=str(plan_path),plan_sha256=analysis['plan_sha256'],captures=inventories,bindings=ledger))
    rows=[];audits=[]
    for record in inventories:
        packet=torch.load(record['file'],map_location='cpu',weights_only=True);weights=torch.load(record['weights_file'],map_location='cpu',weights_only=True)['branch']
        cap=packet['capture'];valid=packet['valid_mask'];x=cap['conditioned_local_rms_input'];q=cap['query'];bias=weights['local_bias']
        need(packet['arm']==record['arm'] and packet['phase']==record['phase'] and packet['step']==record['step']
             and packet['pairing']=='paired' and packet['sids']==record['sids'],'Raw capture metadata differs')
        need(x.dtype==q.dtype==bias.dtype==weights['local.weight'].dtype==torch.float32 and x.ndim==3 and x.shape[-1]==3584
             and q.shape==(x.shape[1],96) and valid.dtype==torch.bool and valid.shape==x.shape[:2]
             and bias.shape==(192,) and weights['local.weight'].shape==(192,3584), 'Captured factor shapes/dtypes differ')
        expected_targets=[by_sid[sid]['target_ids'] for sid in record['sids']]
        if record['phase']=='training':
            need(packet['layout']['target_sequences']==expected_targets and packet['layout']['prefixes']==[[ids[:j] for j in range(len(ids))] for ids in expected_targets],
                 'Strict training query/prefix ownership differs')
        else:need(packet['layout']['first_query_only'] is True and packet['layout']['full_target_ids']==expected_targets,'Final first-query layout differs')
        local=torch.nn.functional.linear(x,weights['local.weight']);errors={}
        actual_factors={f:torch.tanh(cap['factor_preactivation_'+f].double()) for f in ('a','b')}
        removed_factors={f:torch.tanh((local[...,i*96:(i+1)*96]+bias[i*96:(i+1)*96]).double()) for f,i in (('a',0),('b',1))}
        for f,index in (('a',0),('b',1)):
            lo=local[...,index*96:(index+1)*96];b=bias[index*96:(index+1)*96];reference=lo+q.unsqueeze(0)+b
            actual=cap['factor_preactivation_'+f];error=(reference.double()-actual.double()).abs()
            need(actual.shape==reference.shape and actual.dtype==torch.float32 and bool(torch.isfinite(reference).all())
                 and bool(torch.isfinite(actual).all()) and bool((error<=1e-4+1e-4*reference.double().abs()).all()),'FP32 factor preactivation reconstruction failed')
            errors[f]=dict(max_absolute=float(error.max()),atol=1e-4,rtol=1e-4)
            cursor=0
            for sid in record['sids']:
                owner=by_sid[sid];targets=owner['target_ids'] if record['phase']=='training' else owner['target_ids'][:1]
                for position in range(len(targets)):
                    need(torch.equal(valid[:,cursor],torch.arange(x.shape[0])<owner['n_frames']),'Actual valid-item ownership differs')
                    selected=valid[:,cursor]
                    other='b' if f=='a' else 'a'
                    value=geometry(torch,lo[selected,cursor],q[cursor],b,actual[selected,cursor],cap['factor_'+f][selected,cursor],
                        interaction=record['arm'],other_actual=actual_factors[other][selected,cursor],other_removed=removed_factors[other][selected,cursor])
                    rows.append(dict(arm=record['arm'],phase=record['phase'],step=record['step'],factor=f,query_index=cursor,
                        sid=sid,question=owner['question'],n_frames=owner['n_frames'],contrast_id=owner['contrast_id'],variant=owner['variant'],
                        target_position=position,target_token_id=owner['target_ids'][position],prefix_ids=owner['target_ids'][:position],
                        target_role='first_name_token' if position==0 else ('eos' if position==len(owner['target_ids'])-1 else 'name_continuation'),**value));cursor+=1
            need(cursor==x.shape[1],'Every captured query must be retained')
        audits.append(dict(arm=record['arm'],phase=record['phase'],step=record['step'],capture_file=record['file'],capture_sha256=record['sha256'],
            weights_file=record['weights_file'],weights_sha256=record['weights_sha256'],queries=x.shape[1],factor_reconstruction=errors))
    save(out/'query_factor_rows.json',rows);save(out/'capture_audits.json',audits)
    strata=[]
    for arm in ('product','additive'):
        for factor in ('a','b'):
            final=[r for r in rows if r['arm']==arm and r['factor']==factor and r['phase']=='evaluation']
            need(len(final)==108 and len({r['sid'] for r in final})==108,'All108 final scenes per factor required')
            for field in (None,'question','n_frames','contrast_id'):
                groups=defaultdict(list)
                for r in final:groups['all' if field is None else r[field]].append(r)
                for key,group in groups.items():strata.append(summarize(group,dict(arm=arm,factor=factor,phase='evaluation',group=field or 'all',value=key)))
            for step in STEPS:
                group=[r for r in rows if r['arm']==arm and r['factor']==factor and r['phase']=='training' and r['step']==step]
                strata.append(summarize(group,dict(arm=arm,factor=factor,phase='training',step=step)))
    save(out/'strata.json',strata)
    return dict(passed=True,completed=True,captures=26,paired_final_scenes_per_arm=108,final_query_factor_rows=432,total_query_factor_rows=len(rows),
        strata=len(strata),final_overall=[x for x in strata if x['labels'].get('group')=='all'],
        statistics_dtype='FP64 tanh/reductions of actual FP32 preacts and independently recomputed FP32 Wlocal*x+b; captured-factor proxy separate',
        factor_reconstruction_atol=1e-4,factor_reconstruction_rtol=1e-4,saturation_derivative_threshold=.01,
        counterfactual_is_geometry_only=True,no_downstream_prediction=True,no_fit_or_decision_threshold=True,
        local_jacobian_excludes_outer_half_weight=True,not_a_loss_gradient_or_trainability_proof=True,
        no_across_batch_causal_trend_claim=True,vlm_calls=0,core_calls=0,norm_calls=0,head_calls=0,optimizer_calls=0)


def main():
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),'CPU Slurm only')
    started=time.perf_counter();out=OUT/f'query_geometry_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False)
    ledger={};frozen={}
    save(out/'request.json',dict(report_file=str(REPORT),report_sha256=REPORT_SHA,analysis_sha256=ANALYSIS_SHA,own_sources={n:sha(REPO/n) for n in OWN}))
    try:
        summary=read(bind(REPORT,REPORT_SHA,ledger));analysis=read(bind(summary['analysis_file'],ANALYSIS_SHA,ledger))
        need(summary['passed'] is True and summary['completed'] is True and summary['analysis_sha256']==ANALYSIS_SHA
             and analysis['passed'] is True and analysis['completed'] is True and len(summary['source_sha256'])==99,'Passed factor report required')
        frozen={**summary['source_sha256'],**{n:sha(REPO/n) for n in OWN}};need(len(frozen)==101,'Expected101 source files')
        (out/'source').mkdir()
        for name,digest in frozen.items():
            need(sha(REPO/name)==digest,'Current source differs: '+name)
            if name not in OWN:need(sha(REPORT.parent/'source'/name.replace('/','_'))==digest,'Report source snapshot differs')
            target=out/'source'/name.replace('/','_');target.write_bytes((REPO/name).read_bytes());need(sha(target)==digest,'Source copy changed')
        save(out/'source_hashes.json',frozen)
        import torch
        torch.set_num_threads(4)
        with torch.no_grad():tests=self_test(torch);result=analyze(torch,out,summary,analysis,ledger)
        need(all(sha(REPO/n)==h for n,h in frozen.items()),'Source changed during diagnostic')
        text='Saved-state query geometry only. All26 paired captures and every final scene were retained.\n\n'
        text+='| Arm/factor | Actual saturated | Query removed | Payload partial mean, actual | Query removed |\n|---|---:|---:|---:|---:|\n'
        for row in result['final_overall']:
            m=row['metrics'];lab=row['labels']
            text+=f"| {lab['arm']}/{lab['factor']} | {m['actual_saturated_fraction']:.4f} | {m['query_removed_saturated_fraction']:.4f} | {m['actual_payload_partial_absolute_mean']:.6g} | {m['query_removed_payload_partial_absolute_mean']:.6g} |\n"
        text+='\n'
        text+='Actual and query-removed saturation use consistent FP64 tanh of FP32 preactivations; the latter independently recomputes Wlocal*x+b. Captured-factor derivatives and subtraction roundoff remain descriptive. Payload partials exclude the outer half-weight and are not loss gradients. No downstream prediction was recomputed. Large query norms alone do not identify pathology. Training batches mix name and EOS positions and differ in inputs and weights; their summaries are not a causal learning trajectory.\n'
        (out/'REPORT.md').write_text(text)
        files={str(p):sha(p) for p in out.iterdir() if p.is_file()}
        save(out/'summary.json',dict(result,tests=tests,source_sha256=frozen,input_bindings=ledger,files=files,
            report_file=str(REPORT),report_sha256=REPORT_SHA,analysis_sha256=ANALYSIS_SHA,elapsed_seconds=time.perf_counter()-started))
        print(json.dumps(dict(passed=True,directory=str(out))),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,input_bindings=ledger,
            elapsed_seconds=time.perf_counter()-started,partial_outputs_retained=True));raise


if __name__=='__main__':main()

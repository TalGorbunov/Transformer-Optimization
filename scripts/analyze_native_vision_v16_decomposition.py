"""V16 first-prefix paired decomposition of saved native captures; CPU Slurm only.

Audit every V16 trajectory and every executed capture, then compare only t=0
against its fixed V15 counterpart. Include all272 wrapped pairs and16 original
sentinel controls. No native model/head call, fitting or outcome-based selection.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import evaluate_native_vision_v16_step_wrap as study
from scripts import analyze_native_vision_v15_decomposition as algebra
old=study.old
need,read,sha,save,bind,object_sha=study.need,study.read,study.sha,study.save,study.bind,study.object_sha
OUT=REPO/'outputs/native_aggregation_vlm/v16/first_prefix_decomposition'
DATA=Path('/mnt/data/gabriele/gnn_transformer/v16_first_prefix_decomposition')
OWN=('scripts/analyze_native_vision_v16_decomposition.py','slurm/native_vision_v16_decomposition.sbatch')
TERMS=('P','Z','B','E')
CONTROLS=('global_states','reference_states','reference_messages','query','bank_mean','predicted_mean','used_mean')
POLICY=dict(protocol='v16_all_records_first_prefix_paired_decomposition',records=288,wrapped_pairs=272,sentinel_pairs=16,
    models=list(study.CORE_KEYS),modes=list(study.MODES),lengths=[32,64],counts=list(range(17)),
    audit_every_executed_capture=True,decomposed_position=0,observed_prefix_ids=[],
    pair_key='core_key,original_case_index,mode; preserve source SID and physical frame index',
    coefficients={'centered':'actual N','offset':1},terms=dict(algebra.POLICY['terms']),
    mean_reference='Captured FP32 bank_mean and used_mean promoted to FP64, without replacement',
    projected_item_delta='(current_actual_message-original_actual_message) @ frozen W.T, FP64',
    subsets=['positive_first16','positive_later','negative_first16','negative_later'],
    subset_statistics='count, projected sum, mean-shift norm and mean squared per-occurrence projected shift',
    empty_subset='Exact zero sum; mean vector and mean-based scalar statistics are null',
    delta_identity='Delta q + Delta P + Delta Z + Delta B + Delta E = Delta q + W sum(Delta m) - a W Delta nu',
    algebra_rtol=1e-10,algebra_atol=1e-10,
    control_fields=list(CONTROLS),control_differences='Exact tensor identities and FP64 max/RMS/L2; descriptive, no new threshold',
    native_fp32_differences='Descriptive; no new numerical or accuracy gate',
    no_fit=True,no_model_calls=True,no_head_calls=True,no_outcome_selection=True,
    no_new_accuracy_criterion=True,no_practical_milestone_claim=True)


def sources():return {**study.sources(),**algebra.sources(),**{name:sha(REPO/name) for name in OWN}}


def snapshot(out):
    frozen=sources();(out/'code').mkdir()
    for name,digest in frozen.items():
        p=out/'code'/name.replace('/','_');p.write_bytes((REPO/name).read_bytes())
        need(sha(p)==digest,'Source changed while snapshotting')
    save(out/'source_hashes.json',frozen)
    (out/'INDEX.md').write_text('# V16 paired first-prefix decomposition\n\n[Summary](summary.json) · [Every pair](observations.json) · [Sources](source_hashes.json).\n')
    return frozen


def stats(values):
    present=[x for x in values if x is not None]
    need(all(math.isfinite(x) for x in present),'Nonfinite descriptive statistic')
    return dict(n=len(present),missing=len(values)-len(present),mean=math.fsum(present)/len(present) if present else None,
        minimum=min(present) if present else None,maximum=max(present) if present else None)


def difference(torch,a,b):
    need(a.shape==b.shape and a.dtype==b.dtype and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()),
        'Paired control shape/dtype/finiteness differs')
    d=b.double()-a.double()
    return dict(exact=torch.equal(a,b),max_abs=float(d.abs().max()),rms=float(d.square().mean().sqrt()),l2=float(d.norm()))


def paired_algebra(torch,m0,m1,mu0,mu1,nu0,nu1,q0,q1,w,positive,a):
    m0,m1,mu0,mu1,nu0,nu1,q0,q1,w=[x.double() for x in (m0,m1,mu0,mu1,nu0,nu1,q0,q1,w)]
    n,rank=m0.shape
    need(m1.shape==m0.shape and w.shape==(rank,rank) and positive.shape==(n,) and positive.dtype==torch.bool,
        'Paired message/label shape differs')
    need(all(x.shape==(rank,) for x in (mu0,mu1,nu0,nu1,q0,q1)) and 0<=a<=n,'Paired mean/query/coefficient differs')
    item=(m1-m0)@w.T;dmu=w@(mu1-mu0);dnu=w@(nu1-nu0);dq=q1-q0;k=int(positive.sum())
    dp=item[positive].sum(0)-k*dmu;dz=item[~positive].sum(0)-(n-k)*dmu
    db=(n-a)*dmu;de=a*(dmu-dnu);terms=torch.stack((dp,dz,db,de));delta=terms.sum(0)+dq
    direct=q1+w@m1.sum(0)-a*(w@nu1)-(q0+w@m0.sum(0)-a*(w@nu0))
    need(torch.allclose(delta,direct,rtol=1e-10,atol=1e-10),'Paired FP64 delta-preactivation identity failed')
    first=torch.arange(n)<16;subsets={};recovered=torch.zeros(rank,dtype=torch.float64)
    for label,mask in [('positive_first16',positive&first),('positive_later',positive&~first),
                       ('negative_first16',~positive&first),('negative_later',~positive&~first)]:
        values=item[mask];count=len(values);total=values.sum(0);mean=values.mean(0) if count else None
        subsets[label]=dict(count=count,sum=total,mean=mean,sum_norm=float(total.norm()),
            mean_shift_norm=float(mean.norm()) if count else None,
            mean_square_shift=float(values.square().sum(1).mean()) if count else None)
        recovered+=total
    need(sum(v['count'] for v in subsets.values())==n and torch.allclose(recovered,item.sum(0),rtol=1e-10,atol=1e-10),
        'Positive/negative physical-index partition lost occurrences')
    return dict(projected_item_delta=item,term_deltas=terms,term_delta_gram=terms@terms.T,query_delta=dq,
        projected_bank_mean_delta=dmu,projected_used_mean_delta=dnu,preactivation_delta=delta,
        direct_preactivation_delta=direct,subsets=subsets,identity_max_abs=float((delta-direct).abs().max()))


def self_test():
    import torch
    inherited=algebra.self_test();checks=[];m=torch.arange(60,dtype=torch.float32).reshape(20,3)/20
    w=torch.tensor([[2.,0.,1.],[1.,-1.,0.],[0.,2.,3.]]);mu=torch.tensor([1.,2.,3.]);nu=mu/2;q=mu*2
    positive=torch.tensor([i%3==0 for i in range(20)],dtype=torch.bool)
    for a in (1,20):
        same=paired_algebra(torch,m,m,mu,mu,nu,nu,q,q,w,positive,a)
        need(torch.equal(same['preactivation_delta'],torch.zeros(3,dtype=torch.float64)),'Identical paired capture not zero')
        moved=m.clone();moved[17]+=torch.tensor([.5,-.25,1.])
        changed=paired_algebra(torch,m,moved,mu,mu+.25,nu,nu-.5,q,q+1,w,positive,a)
        expected=torch.ones(3,dtype=torch.float64)+w.double()@torch.tensor([.5,-.25,1.],dtype=torch.float64)+a*w.double()@torch.full((3,),.5,dtype=torch.float64)
        need(torch.allclose(changed['preactivation_delta'],expected,rtol=1e-10,atol=1e-10),'Independent affine change differs')
        need(changed['subsets']['positive_first16']['sum_norm']==0 and changed['subsets']['negative_later']['sum_norm']>0,
            'Physical-index/class partition differs')
        checks.append(f'coefficient{a}_identity_and_changed_query_means')
    empty=paired_algebra(torch,m,m,mu,mu,nu,nu,q,q,w,torch.zeros(20,dtype=torch.bool),20)
    need(all(empty['subsets'][k]['mean'] is None and empty['subsets'][k]['mean_shift_norm'] is None
             and empty['subsets'][k]['mean_square_shift'] is None and empty['subsets'][k]['sum_norm']==0
             for k in ('positive_first16','positive_later')),'Empty positive subset fabricated a mean')
    need(difference(torch,mu,mu.clone())['exact'] and not difference(torch,mu,mu+1)['exact'],'Control identity check differs')
    # Exercise the actual capture adapter and independent V15 decomposition together.
    def capture(messages):
        refs=mu.repeat(24,1)
        return dict(actual_messages=messages[:,None,:],reference_messages=refs[:,None,:],
            bank_mean=mu[None,:],used_mean=nu[None,:],predicted_mean=nu[None,:],query=q[None,:],
            corrected_preactivation=(q+w@messages.sum(0)-20*(w@nu))[None,:],
            delta=torch.zeros(1,3),global_states=torch.zeros(1,3),reference_states=refs[:,None,:])
    moved=m.clone();moved[17]+=torch.tensor([.5,-.25,1.])
    adapter,vectors=pair_result(torch,capture(m),capture(moved),w,torch.zeros(3),positive,20)
    need(adapter['subsets']['negative_later']['count']==3 and adapter['subsets']['positive_later']['count']==1
         and adapter['controls']['query']['exact'] and adapter['metrics']['delta_P_norm']==0
         and adapter['metrics']['delta_Z_norm']>0,'Capture adapter/control/subset arithmetic differs')
    need(torch.allclose(vectors['term_deltas'].sum(0),w.double()@(moved.double()-m.double()).sum(0),
                        rtol=1e-10,atol=1e-10),'Capture adapter lost paired vector identity')
    checks+=['empty_positive_mean_null','control_identity','full_capture_adapter'];return dict(passed=True,inherited=inherited,checks=checks)


def verify_report(path,bindings):
    path=Path(path).resolve();a=read(path);s=read(path.parent/'summary.json')
    need(path.is_relative_to(study.OUT) and all(a.get(k) is True and s.get(k) is True for k in ('passed','completed','audit_passed'))
        and s['analysis_file']==str(path) and s['analysis_sha256']==sha(path),'Passed independent V16 report required')
    need(a['policy']==study.POLICY and a['records']==288 and a['wrapped_records']==272 and a['sentinel_records']==16
        and a['all_sentinels_passed'] is True and a['captures_all_prefixes'] is True and set(a['runs'])==set(study.CORE_KEYS),
        'V16 report scope differs')
    study.check_sources(a['source_sha256'],path.parent);need(s['source_sha256']==a['source_sha256'],'Report source ledger differs')
    bind(path,bindings);bind(path.parent/'summary.json',bindings);bind(a['plan_file'],bindings,a['plan_sha256'])
    p=study.verify_plan(a['plan_file']);need(p['source_sha256']==a['source_sha256'],'V16 report/plan source differs')
    bindings.update(p['artifact_bindings']);bind(Path(a['plan_file']).with_suffix('.sha256'),bindings)
    bind(a['allocation']['raw_file'],bindings,a['allocation']['raw_sha256'])
    accounting=study.parse_allocations(Path(a['allocation']['raw_file']).read_text())
    need(all(a['allocation'][k]==v for k,v in accounting.items()),'Original V16 allocation audit differs')
    original,old_plan,old_rows,lookup=study.verify_original(bindings)
    need(p['original_report']==a['original_report']==dict(file=str(study.ORIGINAL),sha256=sha(study.ORIGINAL))
        and p['selected_models']==old_plan['selected_models'],'Wrong original report or frozen checkpoint set')
    bind(a['rescored_file'],bindings,a['rescored_sha256']);rows=read(a['rescored_file'])
    need(len(rows)==288 and Counter(r['core_key'] for r in rows)==Counter({k:72 for k in study.CORE_KEYS}),
        'Missing current trajectory records')
    return a,p,rows,original,old_plan,old_rows,lookup


def audit_trajectory(torch,tokenizer,row,case,selected,bindings):
    bind(row['raw_file'],bindings,row['raw_sha256']);raw=torch.load(row['raw_file'],map_location='cpu',weights_only=True)
    need(raw['schema_version']==1,'Unknown saved raw archive')
    old.audit_output(torch,tokenizer,row,raw['raw_logits'],case,selected)
    identities=old.audit_captures(torch,raw['captures'],row['generated_ids'],case,selected,row['mode'])
    need(identities==row['capture_identities'] and identities[0]['position']==0 and identities[0]['observed_prefix_ids']==[],
        'Saved prefix capture identity differs')
    count=len(raw['captures']);first=raw['captures'][0];logits=raw['raw_logits']
    return first,logits,count


def pair_result(torch,before,after,w,bias,positive,a):
    def arguments(c):return [c[k][:,0] if k in ('actual_messages','reference_messages') else c[k][0]
        for k in ('actual_messages','reference_messages','bank_mean','used_mean','predicted_mean')]
    old_args,new_args=arguments(before),arguments(after)
    left=algebra.decompose(torch,*old_args,w,positive,a);right=algebra.decompose(torch,*new_args,w,positive,a)
    paired=paired_algebra(torch,old_args[0],new_args[0],old_args[2],new_args[2],old_args[3],new_args[3],
        before['query'][0],after['query'][0],w,positive,a)
    delta_terms=right['vectors']['terms']-left['vectors']['terms']
    need(torch.allclose(delta_terms,paired['term_deltas'],rtol=1e-10,atol=1e-10),'Paired delta P/Z/B/E differs from independent decompositions')
    q0,q1=before['query'][0].double(),after['query'][0].double();bias=bias.double()
    original_expected=left['vectors']['total']+q0+bias;current_expected=right['vectors']['total']+q1+bias
    native0,native1=before['corrected_preactivation'][0].double(),after['corrected_preactivation'][0].double()
    native_delta=native1-native0;residual=after['delta'][0].double()-before['delta'][0].double()
    controls={name:difference(torch,before[name],after[name]) for name in CONTROLS}
    metrics={}
    for index,name in enumerate(TERMS):
        metrics.update({f'original_{name}_norm':left['metrics'][f'{name}_norm'],f'current_{name}_norm':right['metrics'][f'{name}_norm'],
            f'delta_{name}_norm':float(paired['term_deltas'][index].norm())})
    metrics.update(original_cos_PZ=left['term_cosines'][0][1],current_cos_PZ=right['term_cosines'][0][1],
        delta_query_norm=float(paired['query_delta'].norm()),delta_preactivation_norm=float(paired['preactivation_delta'].norm()),
        native_preactivation_delta_norm=float(native_delta.norm()),native_residual_delta_norm=float(residual.norm()),
        original_native_fp32_max_abs=float((native0-original_expected).abs().max()),
        current_native_fp32_max_abs=float((native1-current_expected).abs().max()),
        delta_native_fp32_max_abs=float((native_delta-paired['preactivation_delta']).abs().max()),
        delta_identity_max_abs=paired['identity_max_abs'])
    for label,value in paired['subsets'].items():
        for field in ('sum_norm','mean_shift_norm','mean_square_shift'):metrics[f'{label}_{field}']=value[field]
    for name,value in controls.items():
        for field in ('max_abs','rms','l2'):metrics[f'control_{name}_{field}']=value[field]
    tensors=dict(original_terms=left['vectors']['terms'],current_terms=right['vectors']['terms'],
        original_gram=left['vectors']['gram'],current_gram=right['vectors']['gram'],
        original_query=q0,current_query=q1,projection_bias=bias,
        original_reconstructed_preactivation=original_expected,current_reconstructed_preactivation=current_expected,
        native_preactivation_delta=native_delta,native_residual_delta=residual,**paired)
    subset_records={label:{k:v for k,v in value.items() if k not in ('sum','mean')} for label,value in paired['subsets'].items()}
    return dict(metrics=metrics,controls=controls,subsets=subset_records,
        original_term_cosines=left['term_cosines'],current_term_cosines=right['term_cosines']),tensors


def summarize(rows):
    groups=defaultdict(list)
    scopes=(('all',range(17)),('K0_8',range(9)),('K9_15',range(9,16)),('K16',[16]),('K9_16',range(9,17)))
    for row in rows:
        base=f"{row['role']}/{row['core_key']}/{row['mode']}/N{row['n_frames']}"
        for name,ks in scopes:
            if row['gold'] in ks:groups[f'{base}/{name}'].append(row)
    result={}
    for label,items in sorted(groups.items()):
        result[label]=dict(pairs=len(items),statistics={key:stats([r['metrics'][key] for r in items]) for key in items[0]['metrics']},
            control_exact_counts={key:sum(r['controls'][key]['exact'] for r in items) for key in CONTROLS},
            subset_occurrences={key:sum(r['subsets'][key]['count'] for r in items) for key in POLICY['subsets']})
    return result


def run(args,out,frozen,tests):
    import torch
    from transformers import AutoTokenizer
    started=time.perf_counter();bindings={};analysis,plan,rows,original,old_plan,old_rows,lookup=verify_report(args.analysis,bindings)
    tokenizer=AutoTokenizer.from_pretrained(str(study.MODEL),use_fast=False,local_files_only=True)
    data=DATA/out.name;data.mkdir(parents=True,exist_ok=False);observations=[];archives={};models={};all_current_captures=all_original_captures=0
    for key in study.CORE_KEYS:
        selected=plan['selected_models'][study.CORE_KEYS.index(key)];chosen=selected['selected'];bind(chosen['checkpoint'],bindings,chosen['checkpoint_sha256'])
        saved=torch.load(chosen['checkpoint'],map_location='cpu',weights_only=True)
        need(saved['config']==selected['config'] and saved['step']==4590 and saved['student_step']==8000,'Wrong fixed-final checkpoint')
        old.modules()[4].validate_state(torch,saved,chosen['parameter_sha256'])
        w,bias=saved['branch']['aggregate_projection.weight'],saved['branch']['aggregate_projection.bias']
        need(w.shape==(96,96) and bias.shape==(96,) and w.dtype==bias.dtype==torch.float32,'Fixed projection layout differs')
        models[key]=dict(checkpoint=chosen['checkpoint'],checkpoint_sha256=chosen['checkpoint_sha256'],
            parameter_sha256=chosen['parameter_sha256'],core_step=4590,student_step=8000,
            projection=old.modules()[0].native.tensor_info(w),projection_bias=old.modules()[0].native.tensor_info(bias))
        prior_first={};prior_logits={}
        for row in [r for r in old_rows if r['core_key']==key]:
            case=old_plan['cases'][row['case_index']];c,logits,count=audit_trajectory(torch,tokenizer,row,case,selected,bindings)
            prior_first[row['case_index'],row['mode']]=c;prior_logits[row['case_index'],row['mode']]=logits;all_original_captures+=count
        need(len(prior_first)==68,'Original first-prefix support incomplete')
        directory=Path(analysis['runs'][key]['directory']);s=read(directory/'summary.json');config=read(directory/'config.json')
        for name,field in [('summary.json','summary_sha256'),('config.json','config_sha256')]:bind(directory/name,bindings,analysis['runs'][key][field])
        need(s['passed'] is True and s['completed'] is True and s['computational_integrity_passed'] is True
            and all(s[k]==v for k,v in config.items()) and s['source_sha256']==analysis['source_sha256']
            and s['plan_sha256']==analysis['plan_sha256'] and s['selected_model']==selected and s['core_key']==key,
            'Current run identity/completion differs')
        study.check_sources(s['source_sha256'],directory);bind(s['predictions_file'],bindings,s['predictions_sha256'])
        bind(s['sentinel_gate_file'],bindings,s['sentinel_gate_sha256']);stored=read(s['predictions_file']);model_rows=[r for r in rows if r['core_key']==key]
        need(len(stored)==len(model_rows)==72,'Current model rows incomplete')
        tensors=[];sentinels=[]
        for index,(durable,row) in enumerate(zip(stored,model_rows)):
            case=plan['cases'][index//2];mode=study.MODES[index%2]
            need(all(row[k]==v for k,v in durable.items()) and row['index']==index and row['case_index']==index//2
                and row['role']==case['role'] and row['mode']==mode and row['original_case_index']==case['original_case_index'],
                'Current durable record/order changed')
            record_path=directory/f'record_{index:03d}.json';bind(record_path,bindings);need(read(record_path)==durable,'Saved record differs')
            prior=lookup[key,case['original_case_index'],mode]
            need(all(row[k]==prior[k] for k in ('sid','pair_id','gold','n_frames','content_sha256'))
                and row['original_record']==dict(raw_file=prior['raw_file'],raw_sha256=prior['raw_sha256'],generated_ids=prior['generated_ids'],index=prior['index']),
                'Matched original comparator differs')
            current,logits,count=audit_trajectory(torch,tokenizer,row,case,selected,bindings);all_current_captures+=count
            before=prior_first[case['original_case_index'],mode]
            if row['role']=='original_sentinel':
                comparison=study.sentinel_metrics(torch,logits,prior_logits[case['original_case_index'],mode],row['generated_ids'],prior['generated_ids'])
                need(comparison==row['sentinel_comparison'] and comparison['passed'],'Original sentinel comparison differs')
                sentinels.append(dict(index=index,sid=row['sid'],mode=mode,**comparison))
            labels=algebra.labels_for(case['record'],bindings);prior_labels=algebra.labels_for(case['source_record'],bindings)
            need(labels==prior_labels and sum(labels)==row['gold'],'Original/current physical positive mask differs')
            positive=torch.tensor(labels,dtype=torch.bool);a=row['n_frames'] if selected['condition']=='centered' else 1
            record,values=pair_result(torch,before,current,w,bias,positive,a)
            observation=dict(core_key=key,condition=selected['condition'],seed=selected['seed'],mode=mode,role=row['role'],
                sid=row['sid'],pair_id=row['pair_id'],gold=row['gold'],n_frames=row['n_frames'],position=0,observed_prefix_ids=[],
                current_index=index,original_case_index=case['original_case_index'],coefficient=a,
                original_raw_file=prior['raw_file'],original_raw_sha256=prior['raw_sha256'],
                current_raw_file=row['raw_file'],current_raw_sha256=row['raw_sha256'],
                original_capture_identity=prior['capture_identities'][0],current_capture_identity=row['capture_identities'][0],
                positive_count=sum(labels),negative_count=len(labels)-sum(labels),tensor_index=len(tensors),**record)
            observations.append(observation);tensors.append(dict(index=index,role=row['role'],sid=row['sid'],mode=mode,
                positive=positive,physical_indices=torch.arange(1,len(labels)+1),**values))
        expected=dict(passed=True,records=4,observations=sentinels,exact_generated_ids_required=True,each_prefix_tv_max=.02)
        need(len(sentinels)==4 and read(s['sentinel_gate_file'])==expected,'Current sentinel gate differs')
        path=data/f'{key}.pt';torch.save(dict(schema_version=1,term_order=list(TERMS),model=models[key],rows=tensors),path)
        archives[key]=dict(file=str(path),sha256=sha(path),pairs=len(tensors));del saved,prior_first,prior_logits,tensors
        print(json.dumps(dict(core_key=key,pairs=72)),flush=True)
    need(len(observations)==288 and Counter(r['role'] for r in observations)==Counter(wrapped=272,original_sentinel=16)
        and all_current_captures==analysis['capture_count'] and all_original_captures==original['capture_count'],'All-record/all-prefix coverage differs')
    need(Counter((r['core_key'],r['mode'],r['n_frames'],r['gold']) for r in observations if r['role']=='wrapped')
        ==Counter({(key,mode,n,k):1 for key in study.CORE_KEYS for mode in study.MODES for n in (32,64) for k in range(17)}),
        'Wrapped first-prefix support differs')
    save(out/'observations.json',observations)
    geometry=dict(schema_version=1,passed=True,completed=True,policy=POLICY,source_sha256=frozen,tests=tests,
        independent_report=dict(file=str(Path(args.analysis).resolve()),sha256=sha(args.analysis)),
        original_report=dict(file=str(study.ORIGINAL),sha256=sha(study.ORIGINAL)),plan_file=analysis['plan_file'],plan_sha256=analysis['plan_sha256'],
        artifact_bindings=bindings,models=models,records=288,wrapped_pairs=272,sentinel_pairs=16,
        current_captures_audited=all_current_captures,original_captures_audited=all_original_captures,decomposed_position=0,
        observations_file=str(out/'observations.json'),observations_sha256=sha(out/'observations.json'),
        tensor_archives=archives,summaries=summarize(observations),limitations=[
            'Every registered pair is included regardless of correctness; first-prefix contrasts precede generated-history divergence.',
            'Labels are used only offline to partition physical image occurrences; repeated displayed IDs never merge occurrences.',
            'The intervention changes cyclic Step-footer pixels jointly; glyph range, numeral width and repeated-ordinal effects are not separated.',
            'P/Z/B/E and their deltas precede SiLU/readout and are not additive native-logit or accuracy attributions.',
            'Sentinel and unchanged-first16 controls describe rerun or shared-batch changes; no new numerical threshold filters observations.',
            'Absolute coordinate scales differ across fitted models; summaries remain separated by model, mode, length and role.',
            'No fit, new native model/head call, fresh confirmation, mediation claim or practical milestone follows.'])
    need(sources()==frozen,'Source changed during CPU analysis');save(out/'geometry.json',geometry)
    save(out/'summary.json',dict(passed=True,completed=True,policy=POLICY,source_sha256=frozen,tests=tests,
        geometry_file=str(out/'geometry.json'),geometry_sha256=sha(out/'geometry.json'),records=288,wrapped_pairs=272,sentinel_pairs=16,
        current_captures_audited=all_current_captures,original_captures_audited=all_original_captures,
        all_fp64_identities_passed=True,no_model_loaded=True,model_calls=0,head_calls=0,
        slurm_job_id=os.environ['SLURM_JOB_ID'],seconds=time.perf_counter()-started))
    (out/'REPORT.md').write_text('# V16 matched first-prefix decomposition\n\nAll272 wrapped/original pairs and16 sentinel controls retained; every saved prefix audited. Only t=0 decomposed. No model/head calls or fitting.\n\n[Every pair](observations.json) · [Geometry and limitations](geometry.json).\n')
    (data/'INDEX.md').write_text('# V16 paired decomposition tensors\n\nSigned projected per-occurrence changes, P/Z/B/E/query changes, physical masks and native descriptive differences for every first-prefix pair.\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true',required=True)
    parser.add_argument('--analysis',type=Path,required=True);args=parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),
        'All tensor tests and decomposition require CPU Slurm')
    out=OUT/f'decomposition_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False);frozen=snapshot(out)
    import torch
    torch.set_num_threads(min(4,int(os.environ.get('SLURM_CPUS_PER_TASK','1'))));tests=self_test()
    save(out/'selftest.json',dict(passed=True,tests=tests,policy=POLICY,source_sha256=frozen,no_model_loaded=True))
    with torch.no_grad():run(args,out,frozen,tests)
    need(sources()==frozen,'Source changed after analysis');print(json.dumps(dict(directory=str(out),summary_file=str(out/'summary.json'))),flush=True)


if __name__=='__main__':main()

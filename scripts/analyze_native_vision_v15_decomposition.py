"""Registered offline V15 P/Z/B/E decomposition of every saved native query.

No model/head forward, fitting, answer rescoring or checkpoint selection. All
algebra uses saved FP32 values promoted to FP64, including the actually used
bank mean. Later prefixes follow their own trajectories, not matched histories.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
OUT = REPO/'outputs/native_aggregation_vlm/v15/message_decomposition'
DATA = Path('/mnt/data/gabriele/gnn_transformer/v15_message_decomposition')
OWN = ('scripts/analyze_native_vision_v15_decomposition.py',
       'slurm/native_vision_v15_decomposition.sbatch')
TERM_NAMES = ('P', 'Z', 'B', 'E')
POLICY = dict(protocol='v15_all_prefix_offline_projected_decomposition',
    trajectories=272, cores=4, scenes=34, rank=96, reference_occurrences=24,
    coefficients={'centered':'actual N', 'offset':1},
    terms={'P':'sum_positive W(m-mu)', 'Z':'sum_negative W(m-mu)',
           'B':'(N-a) Wmu', 'E':'a W(mu-nu)'},
    mu='Saved FP32 bank_mean promoted to FP64; preserves deployed reference point',
    nu='Saved FP32 used_mean promoted to FP64',
    arithmetic='FP64 on captured FP32 messages/means/query and checkpoint W/bias',
    covariance='Population covariance: divide by actual class/reference occurrence count',
    algebra_rtol=1e-10, algebra_atol=1e-10,
    native_fp32_comparisons='Descriptive; no threshold, parameter or answer changes',
    positive_empty='K0: zero positive sum, null positive mean; no invented positive observation',
    partition='Exact source QA conjunction, offline only',
    all_prefixes=True, no_fit=True, no_model_or_head_calls=True, no_accuracy_rescoring=True)


def study_module():
    from scripts import evaluate_native_vision_v15_null_comparison as study
    return study


def need(value, message):
    if not value: raise ValueError(message)


def sources():
    study = study_module()
    return {**study.sources(), **{name:study.sha(REPO/name) for name in OWN}}


def snapshot(out):
    study = study_module(); frozen = sources(); (out/'code').mkdir()
    for name, digest in frozen.items():
        path = out/'code'/name.replace('/', '_')
        path.write_bytes((REPO/name).read_bytes())
        need(study.sha(path) == digest, 'Source changed while taking snapshot')
    study.save(out/'source_hashes.json', frozen)
    (out/'INDEX.md').write_text('# V15 offline message decomposition\n\n'
        '[Summary](summary.json) · [Sources](source_hashes.json) · [Report](REPORT.md).\n')
    index = OUT/'INDEX.md'
    if not index.exists(): index.write_text('# V15 message decomposition\n\n')
    with index.open('a') as stream: stream.write(f'- [{out.name}]({out.name}/INDEX.md)\n')
    return frozen


def covariance(values):
    """Empirical population covariance; each row is one retained occurrence."""
    need(values.ndim == 2 and len(values) > 0, 'Nonempty occurrence matrix required')
    mean = values.mean(0); centered = values-mean
    return mean, centered.T @ centered / len(values)


def decompose(torch, messages, references, mu, nu, predicted, weight, positive, coefficient):
    """Small pure algebra function: no native/core forward and no hidden-state use."""
    m, refs, mu, nu, pred, w = [x.double() for x in
        (messages, references, mu, nu, predicted, weight)]
    n, rank = m.shape
    need(w.shape == (rank, rank) and refs.ndim == 2 and refs.shape[1] == rank
         and mu.shape == nu.shape == pred.shape == (rank,) and positive.shape == (n,)
         and positive.dtype == torch.bool and 0 <= coefficient <= n,
         'Decomposition shapes/coefficient differ')
    need(all(bool(torch.isfinite(x).all()) for x in (m, refs, mu, nu, pred, w)), 'Nonfinite inputs')
    need(bool((~positive).any()), 'Registered cases must contain negative frames')
    projected = m @ w.T; centered = (m-mu) @ w.T
    p = centered[positive].sum(0); z = centered[~positive].sum(0)
    b = (n-coefficient)*(w @ mu); e = coefficient*(w @ (mu-nu))
    terms = torch.stack((p, z, b, e)); total = terms.sum(0)
    direct = w @ m.sum(0)-coefficient*(w @ nu)
    need(torch.allclose(total, direct, rtol=POLICY['algebra_rtol'], atol=POLICY['algebra_atol']),
         'Registered FP64 P+Z+B+E identity failed')
    gram = terms @ terms.T
    negative_mean, negative_cov = covariance(m[~positive])
    bank_mean_unrounded, bank_cov = covariance(refs)
    negative_projected = w @ negative_cov @ w.T
    bank_projected = w @ bank_cov @ w.T
    positive_mean = m[positive].mean(0) if bool(positive.any()) else None
    if positive_mean is None: need(torch.equal(p, torch.zeros_like(p)), 'K0 positive sum must be exactly zero')
    vectors = dict(terms=terms, gram=gram, total=total, direct=direct,
        raw_positive_sum=m[positive].sum(0), raw_negative_sum=m[~positive].sum(0),
        positive_mean=positive_mean, negative_mean=negative_mean,
        projected_negative_mean=w @ negative_mean, negative_mean_minus_bank=negative_mean-mu,
        projected_negative_mean_minus_bank=w @ (negative_mean-mu),
        bank_mean=mu, bank_mean_unrounded=bank_mean_unrounded, predicted_mean=pred, used_mean=nu,
        projected_bank_mean=w @ mu, projected_predicted_mean=w @ pred,
        prediction_minus_bank=pred-mu, projected_prediction_minus_bank=w @ (pred-mu),
        projected_used_minus_bank=w @ (nu-mu),
        negative_covariance=negative_cov, projected_negative_covariance=negative_projected,
        bank_covariance=bank_cov, projected_bank_covariance=bank_projected)
    norms = terms.norm(dim=1); denominators = norms[:, None]*norms[None, :]
    cosines = [[float(gram[i,j]/denominators[i,j]) if float(denominators[i,j]) > 0 else None
                for j in range(4)] for i in range(4)]
    metrics = {f'{name}_norm':float(norms[i]) for i,name in enumerate(TERM_NAMES)}
    metrics.update({f'{name}_squared_norm':float(gram[i,i]) for i,name in enumerate(TERM_NAMES)})
    metrics.update(total_norm=float(total.norm()), total_squared_norm=float(total.square().sum()),
        vector_identity_max_abs=float((total-direct).abs().max()),
        gram_identity_abs=float(abs(gram.sum()-total.square().sum())),
        positive_mean_norm=None if positive_mean is None else float(positive_mean.norm()),
        negative_mean_norm=float(negative_mean.norm()),
        negative_mean_minus_bank_norm=float((negative_mean-mu).norm()),
        projected_negative_mean_minus_bank_norm=float((w @ (negative_mean-mu)).norm()),
        negative_covariance_trace=float(negative_cov.trace()),
        projected_negative_covariance_trace=float(negative_projected.trace()),
        bank_covariance_trace=float(bank_cov.trace()), projected_bank_covariance_trace=float(bank_projected.trace()),
        prediction_minus_bank_norm=float((pred-mu).norm()),
        projected_prediction_minus_bank_norm=float((w @ (pred-mu)).norm()),
        projected_used_minus_bank_norm=float((w @ (nu-mu)).norm()),
        bank_mean_norm=float(mu.norm()), predicted_mean_norm=float(pred.norm()), used_mean_norm=float(nu.norm()),
        bank_mean_unrounded_difference_norm=float((bank_mean_unrounded-mu).norm()),
        bank_mean_recomputed_fp32_max_abs=float((bank_mean_unrounded.float().double()-mu).abs().max()),
        positive_uncentered_projected_sum_norm=float(projected[positive].sum(0).norm()),
        negative_uncentered_projected_sum_norm=float(projected[~positive].sum(0).norm()))
    return dict(vectors=vectors, metrics=metrics, term_cosines=cosines,
        positive_count=int(positive.sum()), negative_count=int((~positive).sum()), reference_count=len(refs))


def self_test():
    import torch
    m=torch.tensor([[1.,2.,3.],[4.,-1.,2.],[0.,3.,-2.]],dtype=torch.float32)
    refs=torch.cat((torch.ones(8,3),torch.full((16,3),4.)))
    mu=refs.double().mean(0).float(); pred=torch.tensor([.5,-.5,2.]);w=torch.tensor([[2.,0.,1.],[1.,-1.,0.],[0.,2.,3.]])
    mask=torch.tensor([True,False,False]);checks=[]
    need(torch.equal(mu,torch.full((3,),3.)), 'References lost occurrence weights')
    for a in (1,len(m)):
        for mode,nu in (('learned',pred),('bank',mu)):
            result=decompose(torch,m,refs,mu,nu,pred,w,mask,a)
            need(torch.allclose(result['vectors']['total'], (m.double().sum(0)-a*nu.double())@w.double().T,
                                rtol=1e-10,atol=1e-10), 'Direct algebra differs')
            if mode=='bank': need(torch.equal(result['vectors']['terms'][3],torch.zeros(3,dtype=torch.float64)), 'Bank E must be zero')
            if a==len(m): need(torch.equal(result['vectors']['terms'][2],torch.zeros(3,dtype=torch.float64)), 'Centered B must be zero')
            checks.append(f'coefficient{a}_{mode}')
    empty=decompose(torch,m,refs,mu,pred,pred,w,torch.zeros(3,dtype=torch.bool),3)
    need(empty['vectors']['positive_mean'] is None and empty['metrics']['positive_mean_norm'] is None
         and empty['positive_count']==0,'Invented K0 positive mean')
    zero=decompose(torch,torch.zeros_like(m),torch.zeros_like(refs),torch.zeros(3),torch.zeros(3),torch.zeros(3),w,mask,3)
    need(torch.equal(zero['vectors']['terms'],torch.zeros(4,3,dtype=torch.float64))
         and all(x is None for row in zero['term_cosines'] for x in row),'Zero/cosine case differs')
    cancel=decompose(torch,torch.tensor([[1.,0.,0.],[-1.,0.,0.]]),torch.zeros_like(refs),torch.zeros(3),torch.zeros(3),torch.zeros(3),
                     torch.eye(3),torch.tensor([True,False]),2)
    need(torch.equal(cancel['vectors']['total'],torch.zeros(3,dtype=torch.float64))
         and cancel['term_cosines'][0][1]==-1., 'Cancellation lost signed alignment')
    mean,cov=covariance(refs.double());need(torch.equal(mean,mu.double()) and torch.equal(cov,torch.full((3,3),2.,dtype=torch.float64)),
                                           'Population occurrence covariance differs')
    checks += ['empty_positive', 'all_zero', 'signed_cancellation', 'population_occurrence_covariance']
    return dict(passed=True,checks=len(checks),names=checks,no_model_loaded=True)


def labels_for(sample, bindings):
    study=study_module();background=study.modules()[2]
    background.verify_qa(sample,bindings)
    path=Path(sample['path'])/'qa.txt';lines=path.read_text().splitlines()
    states=[ast.literal_eval(s.strip()) for s in lines[lines.index('question:')+1:lines.index('answer:')]
            if s.strip().startswith('{')]
    labels=[]
    for state in states:
        room,people=next(iter(state['rooms'].items()))
        labels.append(room==sample['target_room'] and people[0]==sample['target_character'])
    need(len(labels)==sample['n_frames'] and sum(labels)==sample['gold'], 'Offline positive/negative recount differs')
    return labels


def verify_inputs(torch, analysis_path, bindings):
    study=study_module();analysis_path=Path(analysis_path).resolve();analysis=study.read(analysis_path)
    summary_path=analysis_path.parent/'summary.json';summary=study.read(summary_path)
    need(analysis_path.is_relative_to(study.OUT) and all(analysis.get(k) is True and summary.get(k) is True
         for k in ('passed','completed','audit_passed')) and summary['analysis_file']==str(analysis_path)
         and summary['analysis_sha256']==study.sha(analysis_path),'Completed independent V15 report required')
    need(analysis['records']==272 and analysis['captures_all_prefixes'] is True and analysis['policy']==study.POLICY
         and set(analysis['runs'])==set(study.CORE_KEYS), 'Incomplete report coverage/policy')
    study.check_sources(analysis['source_sha256'],analysis_path.parent)
    need(summary['source_sha256']==analysis['source_sha256'], 'Report source ledgers differ')
    study.bind(analysis_path,bindings);study.bind(summary_path,bindings)
    study.bind(analysis['plan_file'],bindings,analysis['plan_sha256'])
    plan=study.verify_plan(analysis['plan_file'])
    need(plan['source_sha256']==analysis['source_sha256'], 'Report/plan source changed')
    # verify_plan hashes all frozen source/data/model/checkpoint/QA ancestors once.
    bindings.update(plan['artifact_bindings'])
    study.bind(Path(analysis['plan_file']).with_suffix('.sha256'),bindings)
    study.bind(analysis['rescored_file'],bindings,analysis['rescored_sha256'])
    rows=study.read(analysis['rescored_file']);need(len(rows)==272, 'Missing report rows')
    by_key=defaultdict(list)
    for row in rows: by_key[row['core_key']].append(row)
    need(set(by_key)==set(study.CORE_KEYS), 'Unexpected report model')
    cases=plan['cases'];labels={c['record']['sid']:labels_for(c['record'],bindings) for c in cases}
    selected={r['key']:r for r in plan['selected_models']};weights={};model_identities={}
    for key in study.CORE_KEYS:
        run=analysis['runs'][key];directory=Path(run['directory'])
        for name,field in (('summary.json','summary_sha256'),('config.json','config_sha256')):
            study.bind(directory/name,bindings,run[field])
        run_summary=study.read(directory/'summary.json');config=study.read(directory/'config.json')
        need(run_summary['passed'] is True and run_summary['completed'] is True
             and run_summary['computational_integrity_passed'] is True and run_summary['no_fit'] is True
             and all(run_summary[k]==v for k,v in config.items()) and run_summary['source_sha256']==analysis['source_sha256']
             and run_summary['plan_sha256']==analysis['plan_sha256'] and run_summary['core_key']==key
             and run['selected_model']==run_summary['selected_model']==selected[key], 'Run identity/completion differs')
        study.check_sources(run_summary['source_sha256'],directory)
        study.bind(run_summary['predictions_file'],bindings,run_summary['predictions_sha256'])
        original=study.read(run_summary['predictions_file']);model_rows=by_key[key]
        need(len(original)==len(model_rows)==68, 'Missing model trajectories')
        for index,(raw_record,row) in enumerate(zip(original,model_rows)):
            need(all(row[k]==v for k,v in raw_record.items()) and row['index']==index and row['case_index']==index//2
                 and row['mode']==study.MODES[index%2], 'Report raw record/order changed')
            case=cases[index//2]['record']
            need(all(row[k]==case[k] for k in ('sid','pair_id','gold','n_frames','content_sha256'))
                 and row['condition']==selected[key]['condition'] and row['seed']==selected[key]['seed'], 'Scene/model changed')
            path=directory/f'record_{index:03d}.json';study.bind(path,bindings)
            need(study.read(path)==raw_record, 'Durable record changed')
        chosen=selected[key]['selected'];study.bind(chosen['checkpoint'],bindings,chosen['checkpoint_sha256'])
        saved=torch.load(chosen['checkpoint'],map_location='cpu',weights_only=True)
        need(saved['config']==selected[key]['config'] and saved['step']==4590 and saved['student_step']==8000,
             'Wrong fixed-final checkpoint')
        study.modules()[4].validate_state(torch,saved,chosen['parameter_sha256'])
        w=saved['branch']['aggregate_projection.weight'];bias=saved['branch']['aggregate_projection.bias']
        need(w.shape==(96,96) and bias.shape==(96,) and w.dtype==bias.dtype==torch.float32
             and bool(torch.isfinite(w).all()) and bool(torch.isfinite(bias).all()), 'Invalid fixed projection')
        weights[key]=(w.clone(),bias.clone())
        model_identities[key]=dict(checkpoint=chosen['checkpoint'],checkpoint_sha256=chosen['checkpoint_sha256'],
            parameter_sha256=chosen['parameter_sha256'],core_step=4590,student_step=8000,
            projection=study.modules()[0].native.tensor_info(w),projection_bias=study.modules()[0].native.tensor_info(bias))
        del saved
    return analysis,plan,by_key,labels,weights,model_identities


def summarize(rows):
    groups=defaultdict(list)
    for row in rows:
        base=f"{row['core_key']}/{row['mode']}";n=row['n_frames'];k=row['gold'];t=row['position']
        for label in (base, f'{base}/N{n}', f'{base}/K{k}', f'{base}/position{t}',
                      f'{base}/N{n}/position{t}', f'{base}/N{n}/K{k}/position{t}'):
            groups[label].append(row)
    result={}
    for label,items in sorted(groups.items()):
        stats={}
        for key in items[0]['metrics']:
            values=[r['metrics'][key] for r in items if r['metrics'][key] is not None]
            need(all(math.isfinite(v) for v in values), 'Nonfinite summary statistic')
            stats[key]=dict(n=len(values),missing=len(items)-len(values),
                mean=math.fsum(values)/len(values) if values else None,
                minimum=min(values) if values else None,maximum=max(values) if values else None)
        result[label]=dict(captures=len(items),trajectories=len({r['trajectory_id'] for r in items}),
            positive_occurrences=sum(r['positive_count'] for r in items),
            negative_occurrences=sum(r['negative_count'] for r in items),statistics=stats)
    return result


def run(args,out,frozen):
    import torch
    study=study_module();started=time.perf_counter();bindings={}
    gate_path=Path(args.selftest).resolve();gate=study.read(gate_path)
    need(gate_path.is_relative_to(OUT) and gate['passed'] is True and gate['tests']['passed'] is True
         and gate['policy']==POLICY and gate['no_model_loaded'] is True and gate['source_sha256']==frozen,
         'Prior passed diagnostic source/algebra freeze required')
    for name,digest in frozen.items():need(study.sha(gate_path.parent/'code'/name.replace('/','_'))==digest,'Self-test source snapshot differs')
    study.bind(gate_path,bindings);analysis,plan,by_key,labels,weights,models=verify_inputs(torch,args.analysis,bindings)
    data=DATA/out.name;data.mkdir(parents=True,exist_ok=False);all_rows=[];files={}
    for key in study.CORE_KEYS:
        tensor_rows=[];w,bias=weights[key]
        selected=next(r for r in plan['selected_models'] if r['key']==key)
        for row in by_key[key]:
            study.bind(row['raw_file'],bindings,row['raw_sha256'])
            raw=torch.load(row['raw_file'],map_location='cpu',weights_only=True)
            need(raw['schema_version']==1,'Unknown raw capture archive')
            case=plan['cases'][row['case_index']]
            need(row['capture_identities']==study.audit_captures(torch,raw['captures'],row['generated_ids'],case,selected,row['mode']),
                 'Saved capture identities changed')
            mask=torch.tensor(labels[row['sid']],dtype=torch.bool)
            for t,c in enumerate(raw['captures']):
                result=decompose(torch,c['actual_messages'][:,0],c['reference_messages'][:,0],c['bank_mean'][0],
                    c['used_mean'][0],c['predicted_mean'][0],w,mask,c['coefficient'])
                need(result['positive_count']==row['gold'] and result['negative_count']==row['n_frames']-row['gold']
                     and result['reference_count']==24, 'Semantic occurrence counts changed')
                vectors=result.pop('vectors');query=c['query'][0].double();native_bias=bias.double()
                expected=vectors['total']+query+native_bias
                native_corrected=c['corrected_preactivation'][0].double()
                vectors.update(query=query,projection_bias=native_bias,reconstructed_preactivation=expected,
                    captured_corrected_preactivation=native_corrected,
                    captured_minus_reconstructed_preactivation=native_corrected-expected)
                metrics=result['metrics'];m=c['actual_messages'][:,0].double()
                metrics.update(query_norm=float(query.norm()),projection_bias_norm=float(native_bias.norm()),
                    native_delta_norm=float(c['delta'].double().norm()),native_global_norm=float(c['native_global'].double().norm()),
                    native_fused_global_norm=float(c['fused_global'].double().norm()),
                    captured_preactivation_norm=float(c['preactivation'].double().norm()),
                    captured_corrected_preactivation_norm=float(native_corrected.norm()),
                    native_fp32_preactivation_difference_norm=float((native_corrected-expected).norm()),
                    native_fp32_preactivation_difference_max_abs=float((native_corrected-expected).abs().max()),
                    native_fp32_aggregate_difference_norm=float((c['aggregate'][0].double()-m.sum(0)).norm()),
                    native_fp32_projected_null_difference_norm=float((c['projected_null'][0].double()-w.double()@c['used_mean'][0].double()).norm()))
                trajectory=f"{key}/{row['sid']}/{row['mode']}"
                observation=dict(core_key=key,condition=row['condition'],seed=row['seed'],mode=row['mode'],sid=row['sid'],
                    pair_id=row['pair_id'],gold=row['gold'],n_frames=row['n_frames'],question=case['record']['question'],
                    position=t,observed_prefix_ids=row['generated_ids'][:t],trajectory_id=trajectory,
                    raw_file=row['raw_file'],raw_sha256=row['raw_sha256'],capture_identity=row['capture_identities'][t],
                    coefficient=c['coefficient'],tensor_index=len(tensor_rows),term_order=list(TERM_NAMES),
                    term_vectors=vectors['terms'].tolist(),gram=vectors['gram'].tolist(),**result)
                all_rows.append(observation);tensor_rows.append(dict(trajectory_id=trajectory,position=t,**vectors))
            del raw
        path=data/f'{key}.pt';torch.save(dict(schema_version=1,term_order=list(TERM_NAMES),model=models[key],rows=tensor_rows),path)
        files[key]=dict(file=str(path),sha256=study.sha(path),captures=len(tensor_rows))
        print(json.dumps(dict(model=key,captures=len(tensor_rows),trajectories=68)),flush=True)
        del tensor_rows
    need(len(all_rows)==analysis['capture_count']==sum(len(r['generated_ids']) for rows in by_key.values() for r in rows)
         and len({r['trajectory_id'] for r in all_rows})==272,'Every trajectory and actual prefix is required')
    need(Counter((r['core_key'],r['mode'],r['n_frames'],r['gold']) for r in all_rows if r['position']==0)
         ==Counter({(key,mode,n,k):1 for key in study.CORE_KEYS for mode in study.MODES for n in (32,64) for k in range(17)}),
         'Fixed complete model/mode/N/K first-query inventory differs')
    study.save(out/'observations.json',all_rows)
    geometry=dict(schema_version=1,passed=True,completed=True,policy=POLICY,source_sha256=frozen,
        independent_report=dict(file=str(Path(args.analysis).resolve()),sha256=study.sha(args.analysis)),
        plan_file=analysis['plan_file'],plan_sha256=analysis['plan_sha256'],artifact_bindings=bindings,models=models,
        trajectories=272,captures=len(all_rows),tensor_archives=files,
        observations_file=str(out/'observations.json'),observations_sha256=study.sha(out/'observations.json'),
        summaries=summarize(all_rows),limitations=[
            'QA labels partition saved messages only; no model input, target, generated token or accuracy is changed.',
            'P/Z/B/E live before SiLU and U. They do not additively decompose the nonlinear native residual or logits.',
            'Each trajectory follows its actual generated prefix; later learned/bank states need not share histories.',
            'Z is negative-message residual relative to this finite bank, not an identified population bias, image-position effect or sampling variance.',
            'Covariance is descriptive across actual image occurrences at this query; independence is not assumed.',
            'Gram cancellation can have rounding error; vector identity is binding, native FP32 reduction-order differences are descriptive.',
            'K0 positive sum is exactly zero and its positive mean is undefined, not a synthetic zero observation.',
            'No new attention operator, inference method, reasoning composition or successful milestone follows from this analysis.'])
    need(sources()==frozen,'Sources changed during decomposition')
    study.save(out/'geometry.json',geometry)
    (data/'INDEX.md').write_text('# V15 projected decomposition tensors\n\nAll registered captures; per-model FP64 terms, means and covariance matrices.\n')
    (out/'REPORT.md').write_text('# V15 offline message decomposition\n\n'
        f'All 272 trajectories and {len(all_rows)} actual generated-prefix captures included. '
        'The FP64 P+Z+B+E identities passed. Native FP32 differences remain descriptive. '
        'No model or head was called, and no answer was rescored.\n\n'
        'Terms precede the nonlinear readout; they are not additive logit attributions. '
        'Later comparisons follow each mode’s own generated history.\n\n'
        '[Geometry, source bindings and limitations](geometry.json) · [Every capture](observations.json).\n')
    study.save(out/'summary.json',dict(passed=True,completed=True,policy=POLICY,source_sha256=frozen,
        geometry_file=str(out/'geometry.json'),geometry_sha256=study.sha(out/'geometry.json'),
        trajectories=272,captures=len(all_rows),all_algebra_identities_passed=True,no_model_loaded=True,
        model_calls=0,head_calls=0,slurm_job_id=os.environ['SLURM_JOB_ID'],seconds=time.perf_counter()-started))


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-test',action='store_true');mode.add_argument('--run',action='store_true')
    parser.add_argument('--analysis',type=Path);parser.add_argument('--selftest',type=Path);args=parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu'
         and not os.environ.get('SLURM_JOB_GPUS'),'All decomposition and numerical tests require CPU Slurm')
    import torch
    torch.set_num_threads(min(4,int(os.environ.get('SLURM_CPUS_PER_TASK','1'))))
    out=OUT/f'{"selftest" if args.self_test else "decomposition"}_{os.environ["SLURM_JOB_ID"]}'
    out.mkdir(parents=True,exist_ok=False);frozen=snapshot(out);study=study_module()
    if args.self_test:
        tests=self_test();need(sources()==frozen,'Self-test sources changed')
        study.save(out/'summary.json',dict(passed=True,completed=True,tests=tests,policy=POLICY,
            source_sha256=frozen,no_model_loaded=True,slurm_job_id=os.environ['SLURM_JOB_ID']))
    else:
        need(args.analysis is not None and args.selftest is not None,'Require independent report and prior source/algebra freeze')
        with torch.no_grad(): run(args,out,frozen)
    print(json.dumps(dict(directory=str(out),summary_file=str(out/'summary.json'))),flush=True)


if __name__=='__main__': main()

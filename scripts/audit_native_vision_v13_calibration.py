"""CPU-only fixed-checkpoint V13 calibration geometry; no head or model calls.

All null occurrences and all unique ordinary training scenes are retained.
Empirical bank substitution is a diagnostic, not a deployed method or accuracy
estimate. Population covariance divides by24; IID scaling is a reference only.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import math
import os
from pathlib import Path
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import audit_native_vision_v13_checkpoint as selected
from scripts import merge_native_vision_v13_precision as precision
need,read,sha,save=selected.need,selected.read,selected.sha,selected.save
OUT=REPO/'outputs/native_aggregation_vlm/v13/calibration_geometry'
CACHE=Path('/mnt/data/gabriele/gnn_transformer/v13_null_features/feature_cache.json')
SELECTED=selected.OUT/'check_442467/plan.json'
PROTOCOL='v13_fixed_training_bank_calibration_geometry'
OWN=('scripts/audit_native_vision_v13_calibration.py','slurm/native_vision_v13_calibration_geometry.sbatch')
POLICY=dict(models=4,groups_per_model=972,occurrences_per_group=24,unique_scenes_per_model=1782,
    ordinary_positions_per_model=4266,covariance_divisor=24,iid_reference_N=[16,32,64],iid_reference_K=0,
    mean='FP64_over_all_occurrences_then_FP32',statistics_dtype='mixed: FP32 core/projections/error/MSE; FP64 mean accumulation/norms/covariance/Jacobian statistics',core_dtype='FP32',features_dtype='FP16',
    no_deduplication_of_null_occurrences=True,ordinary_weight='each_unique_scene_and_each_valid_prefix_once',
    no_native_head_or_model=True,no_accuracy=True,no_fit=True)


def sources():return {**selected.sources(),**{p:sha(REPO/p) for p in OWN}}


def snapshot(out):
    (out/'source').mkdir();frozen=sources()
    for name,digest in frozen.items():
        p=out/'source'/name.replace('/','_');p.write_bytes((REPO/name).read_bytes());need(sha(p)==digest,'Source changed while copying')
    save(out/'source_hashes.json',frozen)
    (out/'INDEX.md').write_text('# V13 calibration geometry\n\n[Summary](summary.json) · [Sources](source_hashes.json).\n')
    index=OUT/'INDEX.md'
    if not index.exists():index.write_text('# V13 calibration geometry\n\n')
    with index.open('a') as stream:stream.write(f'- [{out.name}]({out.name}/INDEX.md)\n')
    return frozen


def norm(x):return float(x.double().norm())


def derivative(torch,t):
    s=torch.sigmoid(t.double());return s*(1+t.double()*(1-s))


def covariance_trace(torch,rows,gram=None):
    value=rows.double().square().sum() if gram is None else (rows.double()*(rows.double()@gram)).sum()
    need(float(value)>=-1e-9*max(1.,float(rows.double().square().sum())),'Invalid empirical covariance trace')
    return float(value.clamp_min(0)/len(rows))


def self_test():
    import torch
    from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
    from gnnformer.conditional_null_mean import projected_null_readout
    torch.set_num_threads(4)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261107);t=torch.randn(4,dtype=torch.float64);v=torch.randn(4,dtype=torch.float64)
        u=torch.randn(7,4,dtype=torch.float64);w=torch.randn(4,4,dtype=torch.float64);rows=torch.randn(24,4,dtype=torch.float64)
        rows=rows-rows.mean(0);d=derivative(torch,t);gram=u.T@u
        finite=(torch.nn.functional.silu(t+1e-6*v)-torch.nn.functional.silu(t-1e-6*v))/(2e-6)
        torch.testing.assert_close(u@(d*v),u@finite,atol=1e-8,rtol=1e-7)
        jacobian=u@torch.diag(d)@w
        projected=rows@w.T
        need(math.isclose(covariance_trace(torch,projected*d,gram),float((rows@jacobian.T).square().sum()/24),rel_tol=1e-12),
             'Covariance action disagrees with explicit tiny Jacobian')
        values=torch.tensor([[0.,0.],[0.,0.],[6.,3.]])
        mean=values.double().mean(0).float()
        need(mean.tolist()==[2.,1.] and mean.tolist()!=values.unique(dim=0).mean(0).tolist(),'Occurrence weighting was lost')
        core=ParallelLocalAggregation(8,rank=4);core.up.weight.data.normal_(std=.1)
        g=torch.randn(1,8);z=torch.randn(1,4);c=torch.randn(1,4);mu=torch.randn(1,4)
        for mode,a in (('offset',1),('centered',3)):
            before=projected_null_readout(core,z,g,c,n_elements=3,mode=mode)
            after=projected_null_readout(core,z,g,mu,n_elements=3,mode=mode)
            shift=a*torch.nn.functional.linear(c-mu,core.aggregate_projection.weight)
            torch.testing.assert_close(after['corrected_preactivation']-before['corrected_preactivation'],shift,atol=1e-6,rtol=1e-6)
        a=torch.tensor([.2,-.4],dtype=torch.float64);b=torch.tensor([.6,.9],dtype=torch.float64)
        base=torch.tensor([1.,2.],dtype=torch.float64)
        torch.testing.assert_close((base-3*b)-(base-3*a),3*(a-b),atol=1e-15,rtol=1e-15)
    return dict(passed=True,tests=['finite_difference_SiLU_Jacobian','Gram_covariance_without_full_Jacobian',
        'occurrence_weighted_mean','both_arm_exact_substitution_algebra','FP32_substitution_roundoff'])


def summaries(rows,keys,fields):
    grouped=defaultdict(list)
    for row in rows:grouped[tuple(str(row[k]) for k in keys)].append(row)
    output=[]
    for values,items in sorted(grouped.items()):
        record=dict(zip(keys,values));record['observations']=len(items);record['statistics']={}
        for field in fields:
            numbers=sorted(float(r[field]) for r in items)
            need(all(math.isfinite(v) for v in numbers),'Nonfinite geometry scalar')
            record['statistics'][field]=dict(mean=sum(numbers)/len(numbers),median=(numbers[(len(numbers)-1)//2]+numbers[len(numbers)//2])/2,
                p95=numbers[math.ceil(.95*len(numbers))-1],maximum=numbers[-1])
        output.append(record)
    return output


def model_geometry(torch,run,cache,states,index):
    from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
    from gnnformer.conditional_null_mean import ConditionalNullMean,projected_null_readout
    saved=torch.load(run['selected']['checkpoint'],map_location='cpu',weights_only=True)
    selected.validate_state(torch,saved,run['selected']['parameter_sha256'])
    with torch.random.fork_rng(devices=[]):core=ParallelLocalAggregation();predictor=ConditionalNullMean()
    core.load_state_dict(saved['branch']);predictor.load_state_dict(saved['predictor']);del saved
    core.eval().requires_grad_(False);predictor.eval().requires_grad_(False)
    weight=core.aggregate_projection.weight;up=core.up.weight;gram=up.double().T@up.double()
    groups=[];bank_values={};group_for_global={}
    with torch.no_grad():
        for gid,group in sorted(cache['auxiliary_groups'].items()):
            bank=cache['null_banks'][group['question_sha256']]
            need(group['reference_count']==len(group['local_feature_ids'])==24 and group['bank_sha256']==bank['bank_sha256']
                 and [o['source_n_frames'] for o in bank['occurrences']]==[8]*8+[16]*16,'Null occurrence order differs')
            g=states[index[group['global_feature_id']]].unsqueeze(0)
            h=states[[index[f] for f in group['local_feature_ids']]].unsqueeze(1)
            messages=core.encode(h,g)[:,0];mean64=messages.double().mean(0);mean=mean64.float().unsqueeze(0)
            query=core.query(core.rms(g));prediction=predictor(query);error=prediction-mean
            projected_error=torch.nn.functional.linear(error,weight)
            deviations=(messages.double()-mean64)@weight.double().T
            contrast=messages[:8].double().mean(0)-messages[8:].double().mean(0)
            denominator=float(messages.float().square().mean()+1e-6)
            row=dict(group_id=gid,question=group['question'],question_sha256=group['question_sha256'],prefix_ids=group['prefix_ids'],
                global_feature_id=group['global_feature_id'],reference_occurrences=24,distinct_feature_ids=len(set(group['local_feature_ids'])),
                raw_error_norm=norm(error),raw_error_mse=float(error.square().mean()),normalized_auxiliary_mse=float(error.square().mean())/denominator,
                projected_error_norm=norm(projected_error),prediction_norm=norm(prediction),target_norm=norm(mean),query_norm=norm(query),
                projected_prediction_norm=norm(torch.nn.functional.linear(prediction,weight)),projected_target_norm=norm(torch.nn.functional.linear(mean,weight)),
                message_second_moment=float(messages.float().square().mean()),projected_population_covariance_trace=covariance_trace(torch,deviations),
                source_N8_N16_mean_contrast_norm=norm(contrast),source_N8_N16_projected_mean_contrast_norm=norm(contrast@weight.double().T))
            groups.append(row);bank_values[gid]=(mean,prediction,deviations,row)
            need(group['global_feature_id'] not in group_for_global,'Global prefix aliases two null groups')
            group_for_global[group['global_feature_id']]=gid
        need(len(groups)==len(group_for_global)==972,'Incomplete null group coverage')
        ordinary=[]
        for sid,scene in sorted(cache['scenes'].items()):
            n=scene['n_frames'];tokens=scene['target_ids'];g=states[[index[f] for f in scene['global_feature_ids']]]
            h=states[torch.tensor([[index[f] for f in row] for row in scene['local_feature_ids']])]
            need(h.shape==(n,len(tokens),3584) and scene['target_prefixes']==[tokens[:t] for t in range(len(tokens))],
                 'Ordinary native strict-prefix layout differs')
            messages=core.encode(h,g);z=core.aggregate(messages);query=core.query(core.rms(g));prediction=predictor(query)
            gids=[group_for_global[f] for f in scene['global_feature_ids']]
            need(all(cache['auxiliary_groups'][gid]['question']==scene['question'] and cache['auxiliary_groups'][gid]['prefix_ids']==tokens[:t]
                     for t,gid in enumerate(gids)),'Ordinary prefix/null group mismatch')
            means=torch.cat([bank_values[gid][0] for gid in gids]);a=n if run['condition']=='centered' else 1
            deployed=projected_null_readout(core,z,g,prediction,n_elements=n,mode=run['condition'])
            oracle=projected_null_readout(core,z,g,means,n_elements=n,mode=run['condition'])
            shifts=a*torch.nn.functional.linear(prediction-means,weight)
            for t,gid in enumerate(gids):
                reference=bank_values[gid];pre=deployed['corrected_preactivation'][t]
                difference=oracle['corrected_preactivation'][t]-pre;d=derivative(torch,pre)
                linear_response=(d*shifts[t].double())@up.double().T
                actual_response=oracle['delta_float32'][t]-deployed['delta_float32'][t]
                covariance_after=covariance_trace(torch,reference[2]*d,gram)
                item=dict(sid=sid,question=scene['question'],gold=scene['gold'],n_frames=n,position=t,prefix_ids=tokens[:t],
                    target_token_id=tokens[t],target_role='EOS' if t==len(tokens)-1 else 'count',group_id=gid,coefficient=a,
                    preactivation_norm=norm(pre),empirical_preactivation_norm=norm(oracle['corrected_preactivation'][t]),
                    preactivation_difference_norm=norm(difference),projected_mean_error_shift_norm=norm(shifts[t]),
                    FP32_substitution_algebra_error_norm=norm(difference-shifts[t]),
                    predictor_group_batch_shape_difference_norm=norm(prediction[t]-reference[1][0]),
                    deployed_residual_norm=norm(deployed['delta_float32'][t]),empirical_residual_norm=norm(oracle['delta_float32'][t]),
                    residual_difference_norm=norm(actual_response),linearized_mean_error_response_norm=norm(linear_response),
                    nonlinear_linearized_response_difference_norm=norm(actual_response.double()-linear_response),
                    native_FP16_fused_carry_difference_norm=norm((g[t]+oracle['delta_float32'][t].half())-(g[t]+deployed['delta_float32'][t].half())),
                    projected_null_population_covariance_trace=reference[3]['projected_population_covariance_trace'],
                    readout_null_population_covariance_trace=covariance_after,
                    actual_N_K_IID_negative_variance_reference=(n-scene['gold'])*covariance_after,
                    actual_N_K_IID_projected_negative_variance_reference=(n-scene['gold'])*reference[3]['projected_population_covariance_trace'],
                    linearized_mean_error_response_squared_reference=norm(linear_response)**2,
                    deployed_mean_error_squared_reference=norm(shifts[t])**2)
                ordinary.append(item)
    need(len(cache['scenes'])==1782 and len(ordinary)==4266,'Unique ordinary scene/native-prefix count differs')
    need(selected.native.object_sha(selected.training.state_info(core,predictor))==run['selected']['parameter_sha256'],
         'Diagnostic changed selected parameters')
    fields=['raw_error_norm','normalized_auxiliary_mse','projected_error_norm','projected_population_covariance_trace',
            'source_N8_N16_projected_mean_contrast_norm']
    position_fields=['preactivation_difference_norm','residual_difference_norm','linearized_mean_error_response_norm',
        'readout_null_population_covariance_trace','actual_N_K_IID_negative_variance_reference','FP32_substitution_algebra_error_norm']
    iid=[]
    mean_error_squared=sum(r['projected_error_norm']**2 for r in groups)/972
    mean_cov=sum(r['projected_population_covariance_trace'] for r in groups)/972
    for n in (16,32,64):
        a=n if run['condition']=='centered' else 1
        iid.append(dict(N=n,K=0,groups=972,centered_N_squared_predictor_error_reference=n*n*mean_error_squared,
            deployed_coefficient_squared_predictor_error_reference=a*a*mean_error_squared,
            N_times_projected_null_covariance_reference=n*mean_cov,independent_IID_bank_estimation_reference=n*n/24*mean_cov))
    return dict(run_id=run['run_id'],condition=run['condition'],seed=run['seed'],checkpoint=run['selected'],
        groups=groups,ordinary_positions=ordinary,group_summaries={name:summaries(groups,keys,fields) for name,keys in
            [('all',[]),('question',['question']),('prefix',['prefix_ids'])]},
        ordinary_summaries={name:summaries(ordinary,keys,position_fields) for name,keys in
            [('all',[]),('question',['question']),('prefix',['prefix_ids']),('K',['gold']),('N',['n_frames']),('K_N',['gold','n_frames']),('target_role',['target_role'])]},
        IID_fixed_bank_reference=iid,aggregate_projection_weight_norm=norm(weight),readout_weight_norm=norm(up),
        selected_parameters_unchanged=True,no_full_Jacobian_materialized=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-test',action='store_true');mode.add_argument('--run',action='store_true')
    parser.add_argument('--plan',type=Path,default=SELECTED);parser.add_argument('--source-check',type=Path);args=parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),
         'All execution requires Slurm CPU; no GPU/model/head calls are permitted')
    begin=time.perf_counter();job=os.environ['SLURM_JOB_ID'];out=OUT/f'{"selftest" if args.self_test else "geometry"}_{job}'
    out.mkdir(parents=True,exist_ok=False);frozen=snapshot(out)
    import torch
    torch.set_num_threads(4);tests=self_test()
    if args.self_test:
        save(out/'summary.json',dict(passed=True,protocol=PROTOCOL,policy=POLICY,source_sha256=frozen,tests=tests,no_model_loaded=True,
            slurm_job_id=job,seconds=time.perf_counter()-begin));print(str(out),flush=True);return
    need(args.plan.resolve()==SELECTED and args.source_check is not None,'Require exact registered selected CPU plan and source self-test')
    gate=read(args.source_check/'summary.json')
    need(gate['passed'] and gate['protocol']==PROTOCOL and gate['source_sha256']==frozen and gate['policy']==POLICY,'Source self-test changed')
    for name,digest in frozen.items():need(sha(args.source_check/'source'/name.replace('/','_'))==digest,'Self-test source copy changed')
    plan=selected.verify(args.plan);cpu=read(args.plan.parent/'summary.json')
    need(cpu['passed'] and cpu['selected_models']==4 and cpu['plan_sha256']==sha(args.plan),'Selected CPU audit incomplete')
    cache=precision.verify_cache(CACHE,verify_tensors=False)
    need(all(r['config']['cache_binding']==dict(file=str(CACHE),sha256=sha(CACHE)) for r in plan['runs']), 'Canonical training union differs')
    states,index=selected.training.v7.load_features(torch,cache,'cpu');results=[]
    need(sum(len(s['target_ids']) for s in cache['scenes'].values())==4266,'Unique native position count differs')
    for run in plan['runs']:
        result=model_geometry(torch,run,cache,states,index);path=out/(run['run_id']+'.json');save(path,result)
        results.append(dict(run_id=run['run_id'],condition=run['condition'],seed=run['seed'],file=str(path),sha256=sha(path),
            groups=len(result['groups']),ordinary_scenes=1782,ordinary_positions=len(result['ordinary_positions']),checkpoint=run['selected']))
        print(dict(completed_model=run['run_id'],seconds=time.perf_counter()-begin),flush=True)
    need(sources()==frozen and sha(args.plan)==cpu['plan_sha256'],'Diagnostic source/selected plan changed')
    summary=dict(passed=True,completed=True,protocol=PROTOCOL,policy=POLICY,source_sha256=frozen,tests=tests,models=results,
        selected_plan_file=str(SELECTED),selected_plan_sha256=sha(SELECTED),cache_file=str(CACHE),cache_sha256=sha(CACHE),
        source_check=str(args.source_check.resolve()),source_check_sha256=sha(args.source_check/'summary.json'),
        runtime=dict(torch_version=str(torch.__version__),threads=torch.get_num_threads(),device='cpu'),
        total_groups=3888,total_ordinary_positions=17064,no_model_loaded=True,no_native_head_calls=True,no_accuracy=True,no_fit=True,
        scope='Fixed training-bank geometry; population covariance/24; IID N scaling is not a population estimate or prediction',
        caveats=['Bank fit is not independent calibration','K0 bank includes duplicate occurrences and position-dependent images',
            'N-squared/M assumes an independent IID bank and fixed core, not the actual coadapted V13 estimator',
            'Ordinary examples are unique and unweighted, not the weighted training objective',
            'Local Jacobian response does not establish native vocabulary or complete-answer accuracy',
            'FP32 CPU/shape and substitution arithmetic differences remain descriptive'],
        slurm_job_id=job,seconds=time.perf_counter()-begin)
    save(out/'summary.json',summary);print(str(out),flush=True)


if __name__=='__main__':main()

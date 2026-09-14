"""Paired fresh-world statistics after all seven independent numerical audits."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys
import time

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts.stage_mmred_official_recovery import need,sha,save

PROTOCOL='mmred_prefix_supervision_fresh_statistics'
OUT=REPO/'outputs/native_aggregation_vlm/mmred_prefix_supervision_statistics'
DATA=Path('/mnt/data/gabriele/gnn_transformer/mmred_prefix_supervision_statistics')
STATISTICS='docs/paper/MMRED_PREFIX_SUPERVISION_FRESH_STATISTICS.md'
STATISTICS_SHA='687b1bf5eb192a204aaa5da10b6f0527c2a9e16feecc9bed94610a2c254f4e9e'
METHODS=('anchor','answer25','local25','prefix25','answer26','local26','prefix26')
STRATA=(('spend_together','least'),('spend_together','most'),('where_spend','least'),('where_spend','most'))
OWN=('scripts/summarize_mmred_prefix_supervision.py','slurm/mmred_prefix_supervision_statistics.sbatch',
     'docs/paper/MMRED_PREFIX_SUPERVISION_STATISTICS_EXECUTION.md')
POLICY=dict(cpu_seconds=600,cpu_cores=4,memory_gib=16,worlds=1400,primary_worlds=400,
    bootstrap_draws=10000,bootstrap_seed=20260915,quantile_method='linear',percentiles=[.025,.975],
    methods=list(METHODS),strata=[list(s) for s in STRATA],fixed_checkpoints=True,
    no_model_calls=True,no_tensor_head_replay=True,no_efficacy_completion_gate=True,
    uncertainty_conditional_on_checkpoints=True,missing_predictions_are_unavailable=True)


def read(path):return json.loads(Path(path).read_text())
def ref(path):return dict(file=str(Path(path).resolve()),sha256=sha(path))
def sources():return {name:sha(REPO/name) for name in OWN}


def bind(path,bindings,digest=None):
    value=ref(path);need(digest is None or value['sha256']==digest,'Bound statistics input changed')
    bindings[value['file']]=value['sha256'];return value


def proportion(predictions,indices):
    n=len(indices);need(n>0,'Nonempty fixed population required')
    selected=[predictions[i] for i in indices]
    return dict(worlds=n,correct=sum(r['score']['correct'] for r in selected),
        accuracy=sum(r['score']['correct'] for r in selected)/n,
        invalid=sum(not r['score']['format_valid'] for r in selected),
        truncated=sum(r['score']['truncated'] for r in selected),
        completed=sum(r['score']['completed'] for r in selected),
        generated_tokens=sum(len(r['generated_ids']) for r in selected),
        native_decoder_calls_per_world=sum(len(r['generated_ids']) for r in selected)/n,
        instrumented_generation_seconds=sum(r['seconds'] for r in selected),
        instrumented_seconds_are_not_production_latency=True)


def paired(np,correct,left,right,indices,draws):
    a=np.asarray([[correct[left+str(seed)][i] for i in indices] for seed in (25,26)],dtype=np.float64)
    b=np.asarray([[correct[right+str(seed)][i] for i in indices] for seed in (25,26)],dtype=np.float64)
    diff=a-b;values=diff.mean(axis=0)
    bootstrap=values[draws].mean(axis=1)
    interval=np.quantile(bootstrap,[.025,.975],method='linear')
    seeds=[]
    for j,seed in enumerate((25,26)):
        ci=np.quantile(diff[j][draws].mean(axis=1),[.025,.975],method='linear')
        seeds.append(dict(seed=seed,difference_pp=100*float(diff[j].mean()),
            interval_pp=(100*ci).tolist(),wins=int(((a[j]==1)&(b[j]==0)).sum()),
            losses=int(((a[j]==0)&(b[j]==1)).sum()),both_correct=int(((a[j]==1)&(b[j]==1)).sum()),
            both_wrong=int(((a[j]==0)&(b[j]==0)).sum())))
    return dict(left=left,right=right,worlds=len(indices),difference_pp=100*float(values.mean()),
        interval_pp=(100*interval).tolist(),per_seed=seeds,
        both_seed_point_differences_positive=all(s['difference_pp']>0 for s in seeds),
        interval_lower_positive=bool(interval[0]>0),seed_mean_is_not_independent_replication=True),bootstrap


def fixtures(np):
    # Constant paired effects expose accidental independent-seed sampling or
    # treating the two seed outcomes as twice as many independent worlds.
    correct={'a25':[1]*4,'a26':[1]*4,'b25':[0]*4,'b26':[1]*4}
    draws=np.asarray([[0,1,2,3],[3,3,0,0],[1,1,1,1]],dtype=np.int32)
    forward,_=paired(np,correct,'a','b',list(range(4)),draws)
    reverse,_=paired(np,correct,'b','a',list(range(4)),draws)
    need(forward['difference_pp']==50 and forward['interval_pp']==[50.,50.]
         and reverse['difference_pp']==-50 and reverse['interval_pp']==[-50.,-50.]
         and not forward['both_seed_point_differences_positive'],'Paired world/seed fixture failed')
    return dict(passed=True,fixed_half_effect=True,paired_sign_reversal=True,zero_second_seed_not_both_positive=True)


def run(args,out,data,started):
    import numpy as np
    from scripts import report_mmred_prefix_supervision_evaluation as audit
    from scripts import stage_mmred_fresh_native as fresh
    need(sha(REPO/STATISTICS)==STATISTICS_SHA,'Frozen prospective statistics changed')
    bindings={};bind(REPO/STATISTICS,bindings,STATISTICS_SHA)
    reports={};report_refs={};predictions={}
    need(len(args.reports)==7,'Seven complete audited checkpoints are required')
    for path in args.reports:
        result=audit.verify_report(path);method=result['method']
        need(method in METHODS and method not in reports,'Repeated/unregistered method')
        reports[method]=result;report_refs[method]=bind(path,bindings)
        bind(result['predictions_file'],bindings,result['predictions_sha256'])
        predictions[method]=read(result['predictions_file'])
        bind(result['original_rows_file'],bindings,result['original_rows_sha256'])
    need(set(reports)==set(METHODS),'All seven methods must be available')
    first=reports['anchor'];rows=read(first['original_rows_file'])
    need(len(rows)==1400 and [r['index'] for r in rows]==list(range(1400))
         and len({r['world_sha256'] for r in rows})==1400,'Exact main1400 distinct world population required')
    stage=fresh.verify_stage(first['feature_stage']['file'],streaming=True)
    bind(first['feature_stage']['file'],bindings,first['feature_stage']['sha256'])
    bind(stage['timings_file'],bindings)
    visual_timings=read(stage['timings_file'])
    need(stage['counters']['vision']==1400 and visual_timings['worlds']==1400
         and visual_timings['raw_feature_bytes']==40461926400,'Complete shared visual work required')
    groups=read(stage['populations_file']);bind(stage['populations_file'],bindings)
    need(groups['main']==list(range(1400)) and len(groups['main_N32_primary'])==400
         and len(groups['main_N32_controls'])==200,'Fixed primary/control populations differ')
    need(groups['main_N32_primary']==[r['index'] for r in rows if r['n']==32 and r['qtype'] in ('spend_together','where_spend')]
         and groups['main_N32_controls']==[r['index'] for r in rows if r['n']==32 and r['qtype'] in ('char_at_frame','steps_in_room')],
         'Primary/control membership differs from original fresh rows')
    for method,result in reports.items():
        need(all(result[k]==first[k] for k in ('original_rows_file','original_rows_sha256','feature_stage','native_identity_sha256')),
             'Methods were not evaluated on identical native inputs')
        need(result['statistics_protocol']==dict(file=str(REPO/STATISTICS),sha256=STATISTICS_SHA)
             and result['shareddata_populations']==ref(stage['populations_file']),
             'Audit did not bind this fixed statistics protocol and population')
        expected_arm='ordinary' if method=='anchor' else method[:-2]
        need(result['arm']==expected_arm and (method=='anchor' or result['seed']==int(method[-2:])),
             'Method/continuation seed owner differs')
        values=predictions[method]
        need(len(values)==1400 and [r['index'] for r in values]==list(range(1400)),
             'Missing outcomes are unavailable; no complete-case subset is permitted')
        for row,value in zip(rows,values):
            need(all(value[k]==row[k] for k in ('index','sid','n','qtype'))
                 and all(type(value['score'][k]) is bool for k in ('correct','format_valid','completed','truncated'))
                 and value['score']['completed'] is not value['score']['truncated']
                 and value['score']['output_tokens']==len(value['generated_ids']),
                 'Audited prediction/input/score ownership differs')
    primary=groups['main_N32_primary'];primary_rows=[rows[i] for i in primary]
    need(Counter((r['qtype'],r['direction']) for r in primary_rows)==Counter({s:100 for s in STRATA})
         and all(r['n']==32 for r in primary_rows),'Exact four balanced N32 primary strata required')
    strata=[np.asarray([j for j,r in enumerate(primary_rows) if (r['qtype'],r['direction'])==s],dtype=np.int32) for s in STRATA]
    rng=np.random.default_rng(20260915)
    draws=np.concatenate([positions[rng.integers(0,100,size=(10000,100))] for positions in strata],axis=1)
    np.save(data/'bootstrap_indices.npy',draws,allow_pickle=False)
    correct={k:[int(r['score']['correct']) for r in v] for k,v in predictions.items()}
    contrasts={};bootstrap={}
    for left,right in (('prefix','answer'),('prefix','local'),('local','answer')):
        name=left+'_minus_'+right;contrasts[name],bootstrap[name]=paired(np,correct,left,right,primary,draws)
    primary_gate=contrasts['prefix_minus_answer']['interval_lower_positive']
    contrasts['prefix_minus_answer']['ordered_superiority_claim']=primary_gate
    contrasts['prefix_minus_local']['ordered_superiority_claim']=primary_gate and contrasts['prefix_minus_local']['interval_lower_positive']
    contrasts['local_minus_answer']['ordered_superiority_claim']=False
    for value in contrasts.values():
        value['both_observed_seeds_superiority_claim']=value['ordered_superiority_claim'] and value['both_seed_point_differences_positive']
    np.savez(data/'bootstrap_effects.npz',**bootstrap)
    populations=dict(main=groups['main'],N32_aggregation=primary,N32_controls=groups['main_N32_controls'])
    for n in (8,16,32):
        populations[f'N{n}_all']=groups['main_by_length'][str(n)]
        populations[f'N{n}_aggregation']=[r['index'] for r in rows if r['n']==n and r['qtype'] in ('spend_together','where_spend')]
        for task in ('char_at_frame','steps_in_room','spend_together','where_spend'):
            populations[f'N{n}_{task}']=[r['index'] for r in rows if r['n']==n and r['qtype']==task]
        for task,direction in STRATA:
            populations[f'N{n}_{task}_{direction}']=[r['index'] for r in rows if r['n']==n and (r['qtype'],r['direction'])==(task,direction)]
    table={name:{method:proportion(predictions[method],indices) for method in METHODS} for name,indices in populations.items()}
    anchor_comparisons={}
    for method in METHODS[1:]:
        diff=np.asarray([correct[method][i]-correct['anchor'][i] for i in primary],dtype=np.float64)
        anchor_comparisons[method]=dict(difference_pp=100*float(diff.mean()),descriptive=True,
            wins=int((diff==1).sum()),losses=int((diff==-1).sum()))
    inherited={**audit.inherited_sources(),**audit.sources(),**fresh.source_maps()[0],**fresh.source_maps()[1]}
    need(all(sha(REPO/k)==v for k,v in inherited.items()),'Consumed statistics helper changed')
    result=dict(protocol=PROTOCOL,policy=POLICY,passed=True,completed=True,source_sha256=sources(),
        inherited_source_sha256=inherited,statistics=ref(REPO/STATISTICS),audit_reports=report_refs,
        prediction_files={k:dict(file=v['predictions_file'],sha256=v['predictions_sha256']) for k,v in reports.items()},
        original_rows=ref(first['original_rows_file']),feature_stage=first['feature_stage'],
        populations=populations,results=table,contrasts=contrasts,anchor_comparisons=anchor_comparisons,
        bootstrap_indices=ref(data/'bootstrap_indices.npy'),bootstrap_effects=ref(data/'bootstrap_effects.npz'),
        fixtures=fixtures(np),numpy_version=np.__version__,resources={k:v['resources'] for k,v in reports.items()},
        shared_visual_work=dict(stage=first['feature_stage'],timings=visual_timings,
            actual_preparation_allocated_once=True,
            logical_charge_to_every_method=dict(vision_calls=1400,frames=28800,feature_tokens=5644800,
                measured_visual_seconds=visual_timings['vision_seconds']),not_production_latency=True),
        final_checkpoints={k:v['final_checkpoint'] for k,v in reports.items()},
        no_missing_prediction_imputation=True,uncertainty_conditional_on_fixed_checkpoints=True,
        extra_deployed_auxiliary_operations=0,one_prefill_then_native_cached_answer_tokens=True,
        production_latency_claim=False,structural_capacity_claim=False,novel_method_claim=False,
        reasoning_composition_tested=False,objective_achieved=False,elapsed_seconds=time.perf_counter()-started)
    save(out/'input_bindings.json',bindings)
    return result


def render(result,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    table=result['results'];colors={'answer':'#666666','local':'#dc8b1c','prefix':'#2376b7','anchor':'#222222'}
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for arm in ('answer','local','prefix'):
        series=[[100*table[f'N{n}_aggregation'][arm+str(seed)]['accuracy'] for n in (8,16,32)] for seed in (25,26)]
        for values in series:axes[0].plot((8,16,32),values,color=colors[arm],alpha=.3,linewidth=1)
        axes[0].plot((8,16,32),[(a+b)/2 for a,b in zip(*series)],marker='o',label=arm,color=colors[arm])
        for seed in (25,26):
            value=table['N32_aggregation'][arm+str(seed)]
            axes[1].scatter(value['native_decoder_calls_per_world'],100*value['accuracy'],color=colors[arm],marker='o' if seed==25 else '^')
    axes[0].plot((8,16,32),[100*table[f'N{n}_aggregation']['anchor']['accuracy'] for n in (8,16,32)],'k--',label='original')
    original=table['N32_aggregation']['anchor'];axes[1].scatter(original['native_decoder_calls_per_world'],100*original['accuracy'],color='black',marker='x',label='original')
    axes[0].set(xlabel='Sequence length',ylabel='Aggregation accuracy (%)',xticks=[8,16,32],ylim=(0,100))
    axes[1].set(xlabel='Native decoder calls per answer',ylabel='N32 aggregation accuracy (%)',ylim=(0,100))
    axes[0].legend();axes[1].set_title('Identical native architecture; no auxiliary inference')
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(out/'accuracy_and_work.png',dpi=180);fig.savefig(out/'accuracy_and_work.pdf');plt.close(fig)
    lines=['# Fresh MMReD privileged-supervision results','',
        'All seven fixed checkpoints completed the same1,400 questions and passed independent numerical audit.',
        'Primary results use400 distinct N32 aggregation worlds. Intervals condition on these two continuation seeds and one original backbone.','',
        '| Method | N8 aggregation | N16 aggregation | N32 aggregation | N32 decoder calls/answer |',
        '|---|---:|---:|---:|---:|']
    for method in METHODS:
        numbers=[f"{table[f'N{n}_aggregation'][method]['correct']}/{table[f'N{n}_aggregation'][method]['worlds']}" for n in (8,16,32)]
        lines.append('| '+method+' | '+' | '.join(numbers)+f" | {table['N32_aggregation'][method]['native_decoder_calls_per_world']:.3f} |")
    lines+=['','| Paired contrast | Difference (pp) | 95% interval | Ordered superiority claim |','|---|---:|---:|---|']
    for name,value in result['contrasts'].items():
        lo,hi=value['interval_pp'];lines.append(f"| {name} | {value['difference_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {value['ordered_superiority_claim']} |")
    lines+=['','Invalid/truncated completed model outputs count as incorrect; missing outcomes were not imputed.',
        'The loss adds zero inference operations. Instrumented seconds are not production latency measurements.',
        'Causal state use, generalization beyond these40 targets, novelty and reasoning-model composition remain untested.',
        '', '![Accuracy and native decoder work](accuracy_and_work.png)','']
    (out/'REPORT.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--reports',nargs=7,type=Path,required=True);args=parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu'
         and int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==4 and not os.environ.get('SLURM_JOB_GPUS'),'CPU4 Slurm-only statistics required')
    started=time.perf_counter();tag='report_'+os.environ['SLURM_JOB_ID'];out=OUT/tag;data=DATA/tag
    out.mkdir(parents=True,exist_ok=False);data.mkdir(parents=True,exist_ok=False);(out/'source').mkdir()
    own=sources()
    for name,digest in own.items():
        target=out/'source'/name.replace('/','_');shutil.copy2(REPO/name,target);need(sha(target)==digest,'Statistics source archive differs')
    try:
        result=run(args,out,data,started);save(out/'analysis.json',result);render(result,out)
        save(out/'artifacts.json',{str(p):sha(p) for root in (out,data) for p in root.iterdir() if p.is_file()})
        need(sources()==own and time.perf_counter()-started<600,'Statistics source/time boundary changed')
        save(out/'summary.json',dict(protocol=PROTOCOL,policy=POLICY,passed=True,completed=True,analysis=ref(out/'analysis.json'),
            artifacts=ref(out/'artifacts.json'),source_sha256=own,elapsed_seconds=time.perf_counter()-started,no_efficacy_completion_gate=True))
    except BaseException as error:
        save(out/'failure.json',dict(protocol=PROTOCOL,type=type(error).__name__,message=str(error),passed=False,
            elapsed_seconds=time.perf_counter()-started,partial_artifacts_preserved=True));raise


if __name__=='__main__':main()

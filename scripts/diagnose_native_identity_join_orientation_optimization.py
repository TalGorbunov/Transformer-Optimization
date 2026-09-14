"""Slurm-only stdlib comparison of four frozen factor training JSON logs.

Descriptive pre-update CE/gradient observations only; no acceptance, tensors,
models, fitting, convergence test or continuation release.
"""
import json,hashlib,math,time,os
from pathlib import Path
from collections import defaultdict
REPO=Path(__file__).resolve().parents[1]
BASE=REPO/'outputs/native_aggregation_vlm'
OUT=BASE/'training_budget_review'
RUNS=['identity_join_factor_long/run_product_443540','identity_join_factor_long/run_additive_443541','identity_join_factor_orientation_training/run_product_443677','identity_join_factor_orientation_training/run_additive_443678']

def records(path):
    h=hashlib.sha256();body=[]
    with path.open('rb') as f:
        for raw in f:
            h.update(raw);line=raw.decode()
            if line=='  {\n':assert not body;body=[line]
            elif body:
                body.append(line)
                if line in ('  },\n','  }\n'):
                    yield json.loads(''.join(body).strip().rstrip(','));body=[]
    assert not body
    yield h.hexdigest()

def agg():return dict(scenes=0,scene_ce=0.,first=0.,eos=0.,cont=0.,cont_n=0,first_weighted=0.,eos_weighted=0.,cont_weighted=0.,pairs=0,steps=[],lrs=[],grads=[],clipped=0,payload_max=[],orientations=defaultdict(int),pair_ids=[])
def scene(a,r,i,row):
    start,end=r['scene_offsets'][i:i+2];losses=r['position_losses']['ce'][start:end];L=end-start
    assert L==r['scene_lengths'][i]==len(row['target_ids'])
    assert r['target_ids'][start:end]==row['target_ids'] and row['target_ids'][-1]==151645
    assert r['prefix_ids'][start:end]==[row['target_ids'][:j] for j in range(L)]
    assert abs(sum(losses)/L-r['per_scene_ce'][i])<=1e-6
    a['scenes']+=1;a['scene_ce']+=r['per_scene_ce'][i];a['first']+=losses[0];a['eos']+=losses[-1];a['cont']+=sum(losses[1:-1]);a['cont_n']+=L-2
    a['first_weighted']+=losses[0]/L;a['eos_weighted']+=losses[-1]/L;a['cont_weighted']+=sum(losses[1:-1])/L
    a['orientations'][row.get('orientation_version','original')]+=1

def stat(a):
    n=a['scenes'];g=a['grads'];s=a['steps'];lr=a['lrs'];p=a['payload_max']
    return dict(scenes=n,pairs=a['pairs'],steps=[min(s),max(s)],mean_scene_ce=a['scene_ce']/n,first_ce=a['first']/n,eos_ce=a['eos']/n,
        continuation_ce=a['cont']/a['cont_n'] if a['cont_n'] else None,continuation_count=a['cont_n'],
        objective_contributions={k:a[k]/n for k in ('first_weighted','eos_weighted','cont_weighted')},
        gradient_mean=sum(g)/len(g),gradient_min=min(g),gradient_max=max(g),clipped_fraction=a['clipped']/len(g),
        lr_min=min(lr),lr_max=max(lr),payload_max_range=[min(p),max(p)],orientations=dict(a['orientations']))

def merge(items):
    a=agg()
    for b in items:
        for k in ('scenes','scene_ce','first','eos','cont','cont_n','first_weighted','eos_weighted','cont_weighted','pairs','clipped'):a[k]+=b[k]
        for k in ('steps','lrs','grads','payload_max','pair_ids'):a[k]+=b[k]
        for k,v in b['orientations'].items():a['orientations'][k]+=v
    return a


PINNED={
 RUNS[0]:('8a76979d069f8ce1976b23f4b5087da76e6187786cd350b4c8924d7ed630e736','edbfbf35eb84ae70f9acbc28f7ce58615e0345df9b615fb0326723b9eb49144e','911c9da58a0e5681ee8f83dccecc5d61920652b21a36582fdeca34743fa563c5'),
 RUNS[1]:('ff35926da107f660b274c0641995b6017d86f8ea9d8a8ee023450c4275ea5a3c','da273c328396ce5396fbb8539bfe2fad77431cb2a96dc15df51846d7b6ea6132','b232c8025949da6a6e1dba4fb877808b7dc0307c589e5e6e9edef8e3cf27b6ef'),
 RUNS[2]:('450200843c5fc6c3e72b912dc8db360f05a206afe4ec75db3527087bb8ec328e','92cfd66957fef1e7d9033fd5af965c5a51287fd33dbfbed2f65a2066b367320e','149a9eb7313d779bcf2b68b8f3cc6bc3bc1193daafd86f5295d9115b3c68cc3c'),
 RUNS[3]:('d5e84fb8bc4dc7ff4db855b1b7528a7b3880e98238bc467c6295de5a1838d12b','8a10ca26f954be317d10e647e83fbdc97430a0c5199d22df89fb44bbc8d0fa71','132aa19953362a0d330e9d3f2725fd171c6319cdebb23069ea0393c79e43f9e8')}
OWN=('scripts/diagnose_native_identity_join_orientation_optimization.py','slurm/native_identity_join_orientation_optimization.sbatch')


def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def bind(path,expected,bindings):
    path=Path(path).resolve();actual=digest(path);assert actual==expected, 'Input hash differs: '+str(path);bindings[str(path)]=actual


def prepare(out):
    bindings={};packets={};identity=None
    for name in RUNS:
        directory=BASE/name;summary_sha,config_sha,log_sha=PINNED[name]
        bind(directory/'summary.json',summary_sha,bindings);bind(directory/'config.json',config_sha,bindings)
        summary=read(directory/'summary.json');config=read(directory/'config.json')
        assert summary['passed'] is summary['completed'] is True
        assert summary['training_file']==str(directory/'training.json') and summary['training_sha256']==log_sha
        assert all(summary[k]==v for k,v in config.items())
        bind(config['plan_file'],config['plan_sha256'],bindings);plan=read(config['plan_file'])
        bind(plan['rows_file'],plan['runtime_bindings'][plan['rows_file']],bindings);rows=read(plan['rows_file'])
        assert len(rows)==(216 if 'orientation_training' in name else 108)
        assert len({r['sid'] for r in rows})==len(rows)
        same={k:config[k] for k in ('seed','initialized','native_identity_sha256','stats_sha256','initial_checkpoint_sha256')}
        if identity is None:identity=same
        else:assert same==identity,'Unfitted state/native/statistics/seed comparison differs'
        packets[name]=dict(summary=summary,config=config,plan=plan,rows={r['sid']:r for r in rows})
        # The full log hash is checked during its one streaming pass.
        bindings[str(directory/'training.json')]=log_sha
    save(out/'input_inventory.json',dict(input_bindings=bindings,shared_initial_native_statistics=identity,
        training_logs_hash_verified_during_streaming=True,orientation_independent_report_not_required_or_substituted=True,
        source_log_bytes={name:(BASE/name/'training.json').stat().st_size for name in RUNS}))
    return packets,bindings


def analyze(packets):
    results={};signature=None
    for name in RUNS:
        packet=packets[name];summary=packet['summary'];rows=packet['rows'];cycles=defaultdict(agg);windows=defaultdict(agg);all_steps=agg()
        sig=hashlib.sha256();total_positions=0;count=0;max_ce_reduction_error=0.;keys=None
        for r in records(BASE/name/'training.json'):
            if isinstance(r,str):assert r==PINNED[name][2];break
            count+=1;assert r['step']==count and len(r['sids'])==16 and len(r['pair_ids'])==len(r['cycles'])==8
            if keys is None:keys=sorted(r)
            else:assert sorted(r)==keys,'Within-run training schema changed'
            assert r['scene_offsets'][0]==0 and r['scene_offsets'][-1]==len(r['target_ids'])==len(r['position_losses']['ce'])
            assert all(math.isfinite(r[k]) for k in ('gradient_norm','ce_loss','lr','payload_max_abs')) and r['clipped']==(r['gradient_norm']>1.)
            assert r['weighted_consistency_loss']==r['consistency_coefficient']==0. and r['loss']==r['ce_loss']
            max_ce_reduction_error=max(max_ce_reduction_error,abs(sum(r['per_scene_ce'])/16-r['ce_loss']))
            assert r['gate_min']==r['gate_max']==r['gate_mean']==.5
            total_positions+=len(r['target_ids']);w=windows[(count-1)//500]
            for a in (w,all_steps):
                a['steps'].append(count);a['lrs'].append(r['lr']);a['grads'].append(r['gradient_norm']);a['clipped']+=r['clipped'];a['payload_max'].append(r['payload_max_abs']);a['pairs']+=8
            for i,sid in enumerate(r['sids']):
                row=rows[sid];assert r['pair_ids'][i//2]==row['pair_id'];scene(w,r,i,row);scene(all_steps,r,i,row)
            bases=[]
            for j,(pair,c) in enumerate(zip(r['pair_ids'],r['cycles'])):
                assert c==((count-1)*8+j)//54+1
                group=[rows[x] for x in r['sids'][2*j:2*j+2]];assert [x['n_frames'] for x in group]==[8,16]
                base=group[0].get('base_pair_id',pair);bases.append(base);a=cycles[c];a['pairs']+=1;a['pair_ids'].append(base)
                a['steps'].append(count);a['lrs'].append(r['lr']);a['grads'].append(r['gradient_norm']);a['clipped']+=r['clipped'];a['payload_max'].append(r['payload_max_abs'])
                for i in (2*j,2*j+1):scene(a,r,i,rows[r['sids'][i]])
            sig.update(json.dumps([count,r['lr'],bases,r['cycles'],r['target_ids']],sort_keys=True).encode())
        assert count==6000 and total_positions==213330 and len(cycles)==889
        if signature is None:signature=sig.hexdigest()
        else:assert signature==sig.hexdigest(),'Original base-order/LR/target sequence changed'
        for c,a in cycles.items():
            assert a['pairs']==(54 if c<=888 else 48) and len(set(a['pair_ids']))==a['pairs']
            if 'orientation_version' in next(iter(rows.values())):assert set(a['orientations'])=={'original' if c%2 else 'flipped'}
        paircycles={c//2:merge([cycles[c-1],cycles[c]]) for c in range(2,889,2)}
        results[name]=dict(training_sha256=summary['training_sha256'],scope='saved_GPU_losses_not_independent_native_acceptance',
            provisional_first_token_fit=summary['first_token_fit'],schedule_signature=sig.hexdigest(),updates=6000,pair_presentations=48000,
            target_positions=total_positions,complete_base_cycles=888,complete_two_cycle_groups=444,
            overall=stat(all_steps),windows500={str(k+1):stat(v) for k,v in windows.items()},
            double_cycles={str(c):stat(a) for c,a in paircycles.items()},last10=stat(merge([paircycles[c] for c in range(435,445)])),
            preceding10=stat(merge([paircycles[c] for c in range(425,435)])),partial_cycle889=stat(cycles[889]),
            last20_base_cycles_by_orientation={o:stat(merge([cycles[c] for c in range(869,889)
                if o in cycles[c]['orientations']])) for o in sorted({o for c in range(869,889) for o in cycles[c]['orientations']})},
            maximum_descriptive_CE_reduction_difference=max_ce_reduction_error,log_fields=keys,
            saturation_fraction_available=False,per_module_gradients_available=False,
            payload_maximum_is_not_saturation_fraction=True,
            gradient_weighting='per-update for500-step windows; per-presentation for cycle groups')
    return results


def report_text(results):
    lines=['# Descriptive optimization review','','GPU training JSON only. The current orientation independent audit remains separate. These losses do not establish convergence or authorize continuation.','',
        '| Run | Updates | Scene CE | First-token CE | EOS CE | Mean preclip norm | Clipped |','|---|---|---:|---:|---:|---:|---:|']
    for name,r in results.items():
        for w in ('2','4','6','8','10','12'):
            v=r['windows500'][w];lines.append(f"| {name.rsplit('/',1)[-1]} | {v['steps'][0]}–{v['steps'][1]} | {v['mean_scene_ce']:.7g} | {v['first_ce']:.7g} | {v['eos_ce']:.7g} | {v['gradient_mean']:.7g} | {v['clipped_fraction']:.3f} |")
    lines+=['','Each complete54-base-pair cycle contains108 scene presentations. Adjacent cycles cover both orientations in the216-scene study and repeat the same original108 support in its comparator. All444 complete two-cycle groups are saved; final48-pair partial cycle889 remains separate. The same6000 LRs,48000 base-pair order and213330 target IDs are verified across all four logs.',
        '', 'First and EOS CE each average one token per scene; continuation CE averages only interior name tokens. Full CE retains the recorded mean over each complete target. These are pre-update observations from different minibatches, not endpoint rescoring. Global preclip gradient norms do not identify an individual factor or loss gradient. Payload maximum alone does not measure saturation; no saturation fractions or per-module gradients were logged.']
    return '\n'.join(lines)+'\n'


def main():
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS')
    assert int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==4,'Exactly four Slurm CPU cores required'
    started=time.perf_counter();out=OUT/f'orientation_optimization_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False)
    source={name:digest(REPO/name) for name in OWN};(out/'source').mkdir()
    for name,h in source.items():
        target=out/'source'/name.replace('/','_');target.write_bytes((REPO/name).read_bytes());assert digest(target)==h
    save(out/'source_hashes.json',source)
    try:
        packets,bindings=prepare(out);results=analyze(packets)
        value=dict(passed=True,scope='descriptive_stdlib_JSON_only',input_bindings=bindings,source_sha256=source,results=results,
            no_model_tensor_or_head_calls=True,no_fitting=True,no_convergence_claim=True,no_new_experiment_release=True,
            no_orientation_independent_report_acceptance_claim=True)
        save(out/'analysis.json',value);(out/'REPORT.md').write_text(report_text(results))
        elapsed=time.perf_counter()-started;assert elapsed<=60,'Fixed CPU60-second cap exceeded'
        assert source=={name:digest(REPO/name) for name in OWN}
        save(out/'summary.json',dict(passed=True,completed=True,source_sha256=source,elapsed_seconds=elapsed,input_bindings=bindings,
            analysis_file=str(out/'analysis.json'),analysis_sha256=digest(out/'analysis.json'),report_file=str(out/'REPORT.md'),report_sha256=digest(out/'REPORT.md')))
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),elapsed_seconds=time.perf_counter()-started,source_sha256=source,partial_outputs_retained=True));raise


if __name__=='__main__':main()

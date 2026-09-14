"""All four dev-selected V13 checkpoints: fixed-prefix native integrity audit.

No fitting, generation-based selection, or test efficacy rescoring. Both original
software cases are retained. Captured-state native head replay is binding;
cached/full numerical differences remain descriptive, including all failures.
"""
from __future__ import annotations
import argparse
import gc
import math
import os
from pathlib import Path
import random
import re
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import profile_native_vision_v13_null as software
from scripts import native_vision_v13_runtime as runtime
from scripts import train_native_vision_v13 as training
from scripts.stage_native_vision_v6_teacher import MODEL,need,read,save,sha
native=software.native
OUT=REPO/'outputs/native_aggregation_vlm/v13/checkpoint_audit'
DATA=Path('/mnt/data/gabriele/gnn_transformer/v13_checkpoint_audit')
PROTOCOL='all_selected_v13_native_checkpoint_audit'
CONDITIONS=('centered','offset');SEEDS=(16,17);STEPS=(918,1836,2754,3672,4590)
CALLS=dict(model=46,visual=26,language=46,norm=46,last_block=46)
OWN=('scripts/audit_native_vision_v13_checkpoint.py','slurm/native_vision_v13_checkpoint.sbatch',
     'slurm/native_vision_v13_checkpoint_check.sbatch','slurm/native_vision_v13_checkpoint_selftest.sbatch')


def sources():
    from scripts import report_native_vision_v13 as report
    return {**training.sources(),**report.sources(),**{name:sha(REPO/name) for name in OWN}}


def snapshot(out):
    (out/'source').mkdir();frozen=sources()
    for name,digest in frozen.items():
        p=out/'source'/name.replace('/','_');p.write_bytes((REPO/name).read_bytes());need(sha(p)==digest,'Source changed during snapshot')
    save(out/'source_hashes.json',frozen)
    (out/'INDEX.md').write_text('# V13 selected-checkpoint audit\n\n[Summary](summary.json) · [Plan](plan.json) · [Sources](source_hashes.json).\n')
    index=OUT/'INDEX.md'
    if not index.exists():index.write_text('# V13 selected-checkpoint audits\n\n')
    with index.open('a') as stream:stream.write(f'- [{out.name}]({out.name}/INDEX.md)\n')
    return frozen


def bind(artifacts,path,expected=None):
    path=Path(path).resolve();digest=sha(path)
    need(expected is None or digest==expected,'Bound artifact changed: '+str(path));artifacts[str(path)]=digest
    return str(path)


def select_dev(entries):
    need(len(entries)==5 and [e['step'] for e in entries]==list(STEPS),'Require all five registered dev sweeps')
    need(all(type(e['exact_count']) is int and 0<=e['exact_count']<=64 and math.isfinite(e['nll']) and e['nll']>=0 for e in entries),
         'Invalid development selection statistic')
    return min(entries,key=lambda e:(-e['exact_count'],e['nll'],e['step']))


def strict_answer(tokenizer,ids,eos):
    need(isinstance(ids,list) and 1<=len(ids)<=4 and all(type(i) is int and i>=0 for i in ids),'Invalid generated token history')
    complete=ids[-1] in eos
    need(not any(i in eos for i in ids[:-1]) and (complete or len(ids)==4),'Native stop differs')
    bodyids=ids[:-1] if complete else ids;clean=not any(i in set(tokenizer.all_special_ids) for i in bodyids)
    body=tokenizer.decode(bodyids,skip_special_tokens=False)
    value=int(body.strip()) if clean and re.fullmatch('[0-9]+',body.strip()) else None
    return complete,clean,body,value


def audit_dev(torch,processor,entry,directory,config,dev_samples,artifacts):
    path=directory/f'dev_{entry["step"]}.json'
    need(Path(entry['dev_file']).resolve()==path,'Noncanonical dev file')
    bind(artifacts,path,entry['dev_sha256']);dev=read(path)
    need(dev['n']==len(dev['rows'])==64 and [r['sid'] for r in dev['rows']]==[r['sid'] for r in dev_samples]
         and dev['label']==f'dev_{entry["step"]}' and dev['raw_dtype']=='torch.float32','Development coverage/order/dtype differs')
    rawpath=Path(config['data_directory'])/(dev['label']+'_raw.pt')
    need(Path(dev['raw_file']).resolve()==rawpath,'Noncanonical raw dev archive');bind(artifacts,rawpath,dev['raw_sha256'])
    raw=torch.load(rawpath,map_location='cpu',weights_only=True)
    need(raw['schema_version']==1 and len(raw['raw_logits'])==64,'Incomplete raw development archive')
    exact=0;nlls=[]
    for index,(row,sample,logits) in enumerate(zip(dev['rows'],dev_samples,raw['raw_logits'])):
        ids=row['generated_ids'];complete,clean,body,value=strict_answer(processor.tokenizer,ids,training.POLICY['native_eos'])
        need(logits.dtype==torch.float32 and logits.ndim==2 and logits.shape[0]==len(ids)
             and bool(torch.isfinite(logits).all()) and torch.equal(logits,logits.half().float())
             and logits.argmax(-1).tolist()==ids,'Raw native first/continuation argmax differs')
        target=native.encode_target(processor.tokenizer,sample['gold'])
        nll=float(torch.logsumexp(logits[0],-1)-logits[0,target[0]])
        correct=complete and value==sample['gold']
        need(row['raw_index']==index and row['cell']=='dev_N16' and row['n_frames']==16
             and row['gold']==sample['gold'] and row['content_sha256']==sample['content_sha256']
             and row['completed']==complete and row['truncated']==(not complete)
             and row['answer_body']==body and row['no_nonterminal_special_tokens']==clean
             and row['prediction']==value and row['parseable']==(value is not None)
             and row['parsed_count_correct']==(value==sample['gold']) and row['exact']==correct
             and row['raw_text']==processor.tokenizer.decode(ids,skip_special_tokens=False)
             and row['text']==processor.tokenizer.decode(ids,skip_special_tokens=True)
             and math.isclose(nll,row['first_token_nll'],abs_tol=2e-5,rel_tol=2e-6),
             'Independent strict native development rescoring differs')
        exact+=correct;nlls.append(nll)
    # The original FP32 logsumexp can differ slightly across CPU/GPU kernels;
    # validate every value, but preserve the recorded GPU tie-break statistic.
    recorded=sum(r['first_token_nll'] for r in dev['rows'])/64
    need(exact==entry['exact_count']==dev['exact_count'] and recorded==entry['nll']==dev['first_token_nll']
         and math.isclose(sum(nlls)/64,recorded,abs_tol=2e-5,rel_tol=2e-6),'Development totals differ')
    return dict(step=entry['step'],examples=64,exact_count=exact,recorded_first_token_nll=recorded,
                recomputed_first_token_nll=sum(nlls)/64,raw_argmax_and_strict_EOS_rescored=True)


def verify_training(directory,config,summary,plan,cache,artifacts):
    need(config['policy']==training.POLICY and config['condition'] in CONDITIONS and config['seed'] in SEEDS
         and config['consistency_coefficient']==config['auxiliary_coefficient']==1.,'Registered treatment differs')
    need(config['cache_binding']==plan['cache_binding'] and config['pairing_file']==plan['pairing_file']
         and config['pairing_sha256']==plan['pairing_sha256'] and config['prior_result']==plan['prior_result']
         and config['auxiliary_groups_sha256']==plan['auxiliary_groups_sha256'],'Training source/data bindings differ')
    for key in ('model','runtime','processor','native_dtypes'):need(config[key]==cache[key],'Native identity differs: '+key)
    pairing=read(plan['pairing_file']);order=training.presentation_order(pairing['pairs'],config['seed'])
    auxiliary=training.auxiliary_order(cache['auxiliary_groups'],config['seed'])
    for name,rows,key in (('presentations',order,'order_sha256'),('auxiliary_presentations',auxiliary,'auxiliary_order_sha256')):
        bind(artifacts,directory/(name+'.json'),config[name+'_sha256'])
        need(read(directory/(name+'.json'))==rows and native.object_sha(rows)==config[key]==plan[key][str(config['seed'])],
             'Training stream/order differs: '+name)
    hist=read(directory/'training.json');need(len(hist)==4590,'Incomplete optimizer history')
    for step,row in enumerate(hist,1):
        batch=order[(step-1)*16:step*16];aux=auxiliary[(step-1)*16:step*16];sids=[r['sid'] for r in batch]
        targets=[t for sid in sids for t in cache['scenes'][sid]['target_ids']]
        need(row['step']==step and row['sids']==sids and row['target_ids']==targets
             and row['pair_ids']==[r['pair_id'] for r in batch[::2]] and row['epochs']==[r['epoch'] for r in batch[::2]]
             and row['auxiliary_group_ids']==[r['group_id'] for r in aux] and row['auxiliary_cycles']==[r['cycle'] for r in aux]
             and math.isclose(row['lr'],training.lr(step),abs_tol=1e-12,rel_tol=1e-10),'Actual update/target/auxiliary order differs')
        need(all(math.isfinite(row[k]) and row[k]>=0 for k in ('loss','main_loss','ce_loss','consistency_loss','path_loss','auxiliary_loss'))
             and row['consistency_coefficient']==row['auxiliary_coefficient']==1.
             and row['weighted_consistency_loss']==row['consistency_loss'] and row['weighted_auxiliary_loss']==row['auxiliary_loss']
             and math.isclose(row['main_loss'],row['ce_loss']+row['consistency_loss'],rel_tol=3e-6,abs_tol=3e-6)
             and math.isclose(row['loss'],row['main_loss']+row['auxiliary_loss'],rel_tol=3e-6,abs_tol=3e-6)
             and set(row['gradient_norms'])=={'branch','predictor'}
             and all(math.isfinite(v) and v>=0 and row['clipped_groups'][k]==(v>1.) for k,v in row['gradient_norms'].items()),
             'Recorded selected objective or private gradient clips differ')
    need(sum(len(r['target_ids']) for r in hist)==177120,'Full native target coverage differs')
    for field in ('training','first_gradients','gradient_isolation'):
        need(Path(summary[field+'_file']).resolve()==directory/(field+'.json'),'Noncanonical training evidence')
        bind(artifacts,summary[field+'_file'],summary[field+'_sha256'])
    isolation=read(summary['gradient_isolation_file'])
    need(summary['gradient_isolation_passed'] and [r['step'] for r in isolation]==[1,2,32]
         and all(r['passed'] and r['auxiliary_core_gradients_all_none'] for r in isolation)
         and isolation[0]['zero_initial_main_predictor_gradient'] and isolation[-1]['main_predictor']['norm']>0,
         'Gradient isolation gate missing')
    return dict(steps=4590,scene_presentations=len(order),auxiliary_presentations=len(auxiliary),target_positions=177120,
                objectives_and_orders_checked=True,full_per_position_loss_audit_delegated_to_independent_main_report=True)


def verify_run(torch,processor,directory,plans,caches):
    directory=Path(directory).resolve();need(directory.parent==training.OUT,'Require canonical V13 main run directory')
    config=read(directory/'config.json');summary=read(directory/'summary.json');rid=config['run_id'];artifacts={}
    need(directory.name==rid and rid.startswith(f'run_{config["condition"]}_s{config["seed"]}_') and config['arm']=='parallel'
         and config['profile'] is False and all(summary[k]==v for k,v in config.items())
         and all(summary[k] is True for k in ('passed','completed','computational_integrity_passed'))
         and summary['steps']==4590 and summary['native_test_count']==272,'Incomplete/mismatched V13 main run')
    path=bind(artifacts,config['plan_file'],config['plan_sha256'])
    if path not in plans:plans[path]=training.verify_plan(path)
    plan=plans[path];need(plan['source_sha256']==config['source_sha256'],'Main source differs from CPU plan')
    cpu=read(Path(path).parent/'summary.json');need(cpu['passed'] and cpu['plan_sha256']==config['plan_sha256'],'Missing training CPU gate')
    bind(artifacts,Path(path).parent/'summary.json');bind(artifacts,Path(path).with_suffix('.sha256'))
    for name,digest in config['source_sha256'].items():
        need(sha(REPO/name)==digest,'Main source changed');bind(artifacts,directory/'code'/name.replace('/','_'),digest)
    artifacts.update(plan['artifact_bindings']);bind(artifacts,plan['pairing_file'],plan['pairing_sha256'])
    cachepath=bind(artifacts,config['cache_binding']['file'],config['cache_binding']['sha256'])
    if cachepath not in caches:caches[cachepath]=training.cache_binding(cachepath)
    cache=caches[cachepath];checked=verify_training(directory,config,summary,plan,cache,artifacts)
    release=training.verify_release(config['main_release']['file'],config['plan_file'],plan)
    need(release==config['main_release'],'Training release copy changed');bind(artifacts,release['file'],release['sha256'])
    release_data=read(release['file'])
    bind(artifacts,release_data['data_release']['file'],release_data['data_release']['sha256'])
    for item in release_data['profiles'].values():bind(artifacts,Path(item['directory'])/'summary.json',item['summary_sha256'])
    selected=read(directory/'selection.json');entries=selected['development']
    need(summary['development']==entries and summary['selected']==selected['selected'],'Selection copies disagree')
    optimum=select_dev(entries);need(optimum==selected['selected'],'Selected checkpoint is not registered dev optimum')
    devsamples=read(plan['train_manifest'])['splits']['dev_N16']['samples']
    devchecks=[audit_dev(torch,processor,e,directory,config,devsamples,artifacts) for e in entries]
    for e in entries:
        need(Path(e['checkpoint']).resolve()==training.CKPT/rid/f'step_{e["step"]}.pt','Noncanonical development checkpoint')
        bind(artifacts,e['checkpoint'],e['checkpoint_sha256'])
    saved=torch.load(optimum['checkpoint'],map_location='cpu',weights_only=True)
    need(set(saved)=={'branch','predictor','step','config'} and saved['step']==optimum['step'] and saved['config']==config,
         'Selected checkpoint configuration/step differs')
    validate_state(torch,saved,optimum['parameter_sha256'])
    with torch.random.fork_rng(devices=[]):
        from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
        from gnnformer.conditional_null_mean import ConditionalNullMean
        torch.manual_seed(config['seed']);core=ParallelLocalAggregation();predictor=ConditionalNullMean()
        initial=training.state_info(core,predictor)
    need(initial==config['initialized'] and native.object_sha(initial)==config['initialized_sha256'],'Paired initialization differs')
    for name in ('config.json','summary.json','selection.json'):bind(artifacts,directory/name)
    bind(artifacts,summary['test_file'],summary['test_sha256'])
    test=read(summary['test_file']);need(test['n']==len(test['rows'])==272,'Completed test coverage missing')
    bind(artifacts,test['raw_file'],test['raw_sha256'])
    return dict(run_id=rid,condition=config['condition'],seed=config['seed'],arm='parallel',directory=str(directory),config=config,
                selected=optimum,development_checks=devchecks,training_checks=checked,artifact_sha256=artifacts,
                native_profile=plan['native_profile'])


def validate_state(torch,saved,expected):
    from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
    from gnnformer.conditional_null_mean import ConditionalNullMean
    with torch.random.fork_rng(devices=[]):core=ParallelLocalAggregation();predictor=ConditionalNullMean()
    core.load_state_dict(saved['branch'],strict=True);predictor.load_state_dict(saved['predictor'],strict=True)
    need(sum(p.numel() for p in core.parameters())==1041600 and sum(p.numel() for p in predictor.parameters())==18624,
         'Core/predictor architecture differs')
    need(all(t.dtype==torch.float32 and bool(torch.isfinite(t).all()) for group in ('branch','predictor') for t in saved[group].values()),
         'Selected learned tensors must be finite FP32')
    need(native.object_sha(training.state_info(core,predictor))==expected,'Selected combined parameter digest differs')


def self_test():
    import torch
    torch.set_num_threads(4)
    entries=[dict(step=s,exact_count=10,nll=1.) for s in STEPS]
    need(select_dev(entries)['step']==STEPS[0],'Earliest tie-break failed')
    entries[2]['nll']=.8;need(select_dev(entries)['step']==STEPS[2],'NLL tie-break failed')
    entries[-1].update(exact_count=11,nll=2.);need(select_dev(entries)['step']==STEPS[-1],'Exact count must dominate NLL')
    for invalid in (entries[:-1],entries+entries[:1],[dict(e,nll=float('nan')) for e in entries],[dict(e,exact_count=65) for e in entries]):
        try:select_dev(invalid)
        except ValueError:pass
        else:raise AssertionError('Malformed development inventory accepted')
    class Tokenizer:
        all_special_ids=[9,10]
        def decode(self,ids,skip_special_tokens=False):return ''.join({0:'0',1:'1',2:'6',8:'x',9:'',10:''}[i] for i in ids)
    tok=Tokenizer()
    need(strict_answer(tok,[1,2,9],[9])==(True,True,'16',16),'Multi-token numeral rejected')
    need(strict_answer(tok,[1,10,9],[9])==(True,False,'1',None),'Nonterminal special token wrongly ignored')
    need(strict_answer(tok,[1,8,8,8],[9])==(False,True,'1xxx',None),'Truncated trace incorrectly completed')
    for ids in ([1],[9,1,9],[],[1,2,0,0,9]):
        try:strict_answer(tok,ids,[9])
        except ValueError:pass
        else:raise AssertionError('Wrong EOS/length accepted')
    a=torch.tensor([1.,2.,3.]);same=software.old.metric(torch,a,a+7.);wrong=software.old.metric(torch,a,a.flip(0))
    need(same['numerical_rule_passed'] and same['full_vocabulary_tv']<1e-15 and not wrong['numerical_rule_passed'],
         'Binding replay/descriptive numerical classification differs')
    from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
    from gnnformer.conditional_null_mean import ConditionalNullMean
    rng=torch.random.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        core=ParallelLocalAggregation();predictor=ConditionalNullMean()
        saved=dict(branch=core.state_dict(),predictor=predictor.state_dict());identity=native.object_sha(training.state_info(core,predictor))
    validate_state(torch,saved,identity)
    need(torch.equal(rng,torch.random.get_rng_state()),'State validation changed caller RNG')
    saved['predictor']['fc2.bias']=saved['predictor']['fc2.bias']+1.
    try:validate_state(torch,saved,identity)
    except ValueError:pass
    else:raise AssertionError('Wrong predictor identity accepted')
    need(2*(3+4*5)==CALLS['model'] and 2*(1+4*3)==CALLS['visual'] and 4*2*(17+65)==656 and 4*2*3==24,
         'Registered selected audit arithmetic differs')
    return dict(passed=True,tests_passed=True,tests=['complete_dev_selection','native_multitoken_and_terminal_EOS',
        'reject_nonterminal_special','retain_numerical_failure','combined_predictor_identity','private_validation_RNG','exact_call_inventory'])


def unit():
    out=OUT/f'selftest_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False);frozen=snapshot(out)
    result=self_test();need(sources()==frozen,'Sources changed during self-test')
    save(out/'summary.json',dict(result,protocol=PROTOCOL,source_sha256=frozen,no_model_loaded=True,
        scope='Synthetic selected-audit software; no trained checkpoint or outcome opened',slurm_job_id=os.environ['SLURM_JOB_ID']))
    print(dict(passed=True,directory=str(out)),flush=True)


def verify_source_check(directory):
    directory=Path(directory).resolve();summary=read(directory/'summary.json')
    need(directory.parent==OUT and directory.name.startswith('selftest_') and summary['protocol']==PROTOCOL
         and summary['passed'] and summary['tests_passed'] and summary['no_model_loaded'] and summary['source_sha256']==sources(),
         'Require exact pre-main audit source self-test')
    need(read(directory/'source_hashes.json')==summary['source_sha256'],'Self-test source ledger differs')
    for name,digest in summary['source_sha256'].items():need(sha(directory/'source'/name.replace('/','_'))==digest,'Self-test source copy differs')
    return summary


def check(args):
    import torch,transformers
    from transformers import AutoProcessor
    from scripts.probe_native_vision_v2_prefix import fingerprint
    torch.set_num_threads(4);begin=time.perf_counter();job=os.environ['SLURM_JOB_ID']
    out=OUT/f'check_{job}';out.mkdir(parents=True,exist_ok=False);frozen=snapshot(out)
    source_check=verify_source_check(args.source_check);tests=self_test()
    processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,use_fast=False)
    plans={};caches={};runs=[verify_run(torch,processor,path,plans,caches) for path in args.runs]
    runs.sort(key=lambda r:(r['condition'],r['seed']))
    need(len(runs)==4 and {(r['condition'],r['seed']) for r in runs}=={(c,s) for c in CONDITIONS for s in SEEDS},
         'All four selected models required exactly once')
    configs=[r['config'] for r in runs]
    for key in ('policy','plan_file','plan_sha256','source_sha256','cache_binding','model','runtime','processor','native_dtypes','main_release'):
        need(all(c[key]==configs[0][key] for c in configs),'Unmatched run identity: '+key)
    for seed in SEEDS:
        pair=[r['config'] for r in runs if r['seed']==seed]
        for key in ('initialized','initialized_sha256','order_sha256','auxiliary_order_sha256'):
            need(pair[0][key]==pair[1][key],'Paired initialization/training streams differ')
    profile_binding=runs[0]['native_profile'];profile=software.verify_profile(profile_binding['file'])
    need(sha(profile_binding['file'])==profile_binding['sha256'] and all(r['native_profile']==profile_binding for r in runs),
         'Runs used different native software gate')
    parent=software.verify(profile['plan_file'])
    need(parent['model']==configs[0]['model'] and parent['runtime']==configs[0]['runtime']
         and parent['processor']==configs[0]['processor']==fingerprint(processor,str(transformers.__version__)),
         'Software/main model or processor differs')
    forced=[]
    for word in software.FORCED:
        ids=processor.tokenizer(word,add_special_tokens=False)['input_ids']
        need(len(ids)==1 and ids[0] not in processor.tokenizer.all_special_ids,'Fixed ordinary prefix differs');forced+=ids
    need(forced==parent['forced_ids'],'Frozen forced-prefix IDs differ')
    artifacts={}
    for run in runs:artifacts.update(run['artifact_sha256'])
    bind(artifacts,profile_binding['file'],profile_binding['sha256']);bind(artifacts,profile['plan_file'],profile['plan_sha256'])
    bind(artifacts,parent['prepared_file'],parent['prepared_sha256'])
    bind(artifacts,Path(args.source_check)/'summary.json')
    # Bind all model selections to the prospective source gate actually released
    # for mains, not merely to a later compatible-looking self-test.
    release=read(configs[0]['main_release']['file']);gate=release['selected_audit_freeze']
    need(Path(gate['directory']).resolve()==Path(args.source_check).resolve()
         and gate['summary_sha256']==sha(Path(args.source_check)/'summary.json'),'Released selected-audit source gate differs')
    plan=dict(schema_version=1,protocol=PROTOCOL,source_sha256=frozen,source_check=str(Path(args.source_check).resolve()),
        source_check_sha256=sha(Path(args.source_check)/'summary.json'),tests=tests,runs=runs,artifact_sha256=artifacts,
        policy=training.POLICY,software_profile=profile_binding,parent_plan_file=profile['plan_file'],parent_plan_sha256=profile['plan_sha256'],
        prepared_file=parent['prepared_file'],prepared_sha256=parent['prepared_sha256'],cases=parent['cases'],
        model=parent['model'],runtime=parent['runtime'],processor=parent['processor'],native_api=parent['native_api'],
        forced_text=list(software.FORCED),forced_ids=forced,calls=CALLS,head_replay_calls=46,cache_full_rows=656,
        fixed_prefix_kv_checks=24,selected_fusion_checks=40,ordinary_generations=0,extra_last_block_forwards=0,
        backend='native_bitsandbytes_dispatch_unmodified',original_mixed_failure=parent['original_mixed_failure'],
        original_mixed_numerical_gate_passed=False,no_fitting=True,no_efficacy_scoring=True,
        scope='All dev-selected models; captured native head TV<=.02/top1 binding; cached/full errors descriptive',slurm_job_id=job)
    need(sources()==frozen,'Audit sources changed during CPU binding')
    save(out/'plan.json',plan);(out/'plan.sha256').write_text(sha(out/'plan.json')+'\n');verify(out/'plan.json')
    save(out/'summary.json',dict(passed=True,protocol=PROTOCOL,source_sha256=frozen,plan_file=str(out/'plan.json'),
        plan_sha256=sha(out/'plan.json'),selected_models=4,no_model_loaded=True,seconds=time.perf_counter()-begin,slurm_job_id=job))
    print(dict(passed=True,plan_file=str(out/'plan.json')),flush=True)


def verify(path):
    path=Path(path).resolve();plan=read(path)
    need(path.is_relative_to(OUT) and path.with_suffix('.sha256').read_text().strip()==sha(path)
         and plan['schema_version']==1 and plan['protocol']==PROTOCOL and plan['source_sha256']==sources()
         and plan['policy']==training.POLICY,'Frozen audit source/plan differs')
    for name,digest in plan['source_sha256'].items():need(sha(path.parent/'source'/name.replace('/','_'))==digest,'CPU source copy changed')
    for name,digest in plan['artifact_sha256'].items():need(sha(name)==digest,'Bound selected/data artifact changed: '+name)
    verify_source_check(plan['source_check']);parent=software.verify(plan['parent_plan_file'])
    need(sha(Path(plan['source_check'])/'summary.json')==plan['source_check_sha256']
         and sha(plan['parent_plan_file'])==plan['parent_plan_sha256'] and sha(plan['prepared_file'])==plan['prepared_sha256']
         and plan['prepared_file']==parent['prepared_file'] and plan['cases']==parent['cases']
         and plan['forced_ids']==parent['forced_ids'] and plan['forced_text']==list(software.FORCED)
         and plan['calls']==CALLS and plan['head_replay_calls']==46 and plan['cache_full_rows']==656
         and plan['fixed_prefix_kv_checks']==24 and plan['selected_fusion_checks']==40 and plan['ordinary_generations']==0
         and plan['extra_last_block_forwards']==0,'Complete prescribed case/call coverage differs')
    need(len(plan['runs'])==4 and {(r['condition'],r['seed']) for r in plan['runs']}=={(c,s) for c in CONDITIONS for s in SEEDS},
         'Selected model inventory differs')
    for key in ('model','runtime','processor','native_api'):need(plan[key]==parent[key],'Native ancestor identity changed')
    return plan


def run(args):
    import torch,transformers
    from gnnformer.runtime import load_runtime,get_rope_index_fn,move_to_device
    from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
    from gnnformer.conditional_null_mean import ConditionalNullMean
    from gnnformer.parallel_local_learned_null import ParallelLocalLearnedNull
    from scripts.probe_native_vision_v2_prefix import fingerprint
    torch.set_num_threads(4);begin=time.perf_counter();job=os.environ['SLURM_JOB_ID']
    out=OUT/f'audit_{job}';out.mkdir(parents=True,exist_ok=False);frozen=snapshot(out)
    data=DATA/f'audit_{job}';data.mkdir(parents=True,exist_ok=False)
    (data/'INDEX.md').write_text('# V13 selected-checkpoint raw evidence\n\n[Head replays](head_replays.pt).\n')
    plan=verify(args.plan);cpu=read(Path(args.plan).parent/'summary.json')
    need(cpu['passed'] and cpu['selected_models']==4 and cpu['plan_sha256']==sha(args.plan),'Require exact completed CPU audit binding')
    (out/'plan.json').write_bytes(Path(args.plan).read_bytes())
    blob=torch.load(plan['prepared_file'],map_location='cpu',weights_only=True);need(blob['schema_version']==1,'Prepared schema differs')
    tick=time.perf_counter();loaded=load_runtime(str(MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda')
    model=loaded.model.eval().requires_grad_(False);norm=native.native_contract(model)
    need(fingerprint(loaded.processor,str(transformers.__version__))==plan['processor'] and software.runtime_identity()==plan['runtime'],
         'Loaded native runtime/processor differs')
    _,_,api=software.old.local.native_api(loaded.processor);need(api==plan['native_api'],'Installed native API differs')
    core=ParallelLocalAggregation().eval().requires_grad_(False).to(model.device)
    predictor=ConditionalNullMean().eval().requires_grad_(False).to(model.device);native.native_contract(model,core)
    torch.cuda.synchronize();load_seconds=time.perf_counter()-tick;torch.cuda.reset_peak_memory_stats()
    versions={k:p._version for k,p in model.named_parameters()};checks=[];comparisons=[];controllers=[];restores=[];records=[];counts={}
    try:
        with software.ancestor.NativeAudit(model,data) as audit:
            try:
                for case in plan['cases']:
                    cid=case['case_id'];n=case['n_frames'];width=case['prompt_width'];tokens=plan['forced_ids']
                    bundles=[blob['bundles'][f'{cid}_t{t}'] for t in range(3)]
                    for t,bundle in enumerate(bundles):
                        need(runtime.validate_bundle(bundle)['passed'] and bundle['metadata']['input_identity']==case['variants'][t]['input_identity']
                             and native.audit_layout(get_rope_index_fn(model),bundle)['metadata']==case['variants'][t]['layout'],'Software layout changed')
                    baseline=[];baseline_kv=[];native_full={};cache=None;audit.controller=None
                    for step,bundle in enumerate(bundles):
                        inputs,positions,layout=software.cached_input(model,bundle,cache,step,tokens,model.device)
                        audit.context=dict(label=f'{cid}__native__cached{step}',case_id=cid,n_frames=n,phase='cached',step=step,condition='native')
                        with torch.inference_mode():
                            output=model(**inputs,use_cache=True,logits_to_keep=1) if 'use_cache' not in inputs else model(**inputs,logits_to_keep=1)
                        cache=output.past_key_values;raw=audit.last_raw
                        need(torch.equal(raw['position_ids'],positions.cpu()) and torch.equal(raw['native_norm_input'],raw['fused_norm_input']),
                             'Bare native positions/readout differ')
                        baseline.append(raw);baseline_kv.append(software.snapshot_kv(cache))
                    del cache,output
                    for model_index,selected in enumerate(plan['runs']):
                        rid=selected['run_id'];saved=torch.load(selected['selected']['checkpoint'],map_location='cpu',weights_only=True)
                        core.load_state_dict(saved['branch'],strict=True);predictor.load_state_dict(saved['predictor'],strict=True)
                        need(native.object_sha(training.state_info(core,predictor))==selected['selected']['parameter_sha256'],
                             'Selected combined checkpoint did not restore exactly')
                        original=training.state_info(core,predictor);del saved
                        mode=selected['condition'];cache=None
                        control=ParallelLocalLearnedNull(norm,core,predictor,n_local_rows=n,mode=mode,
                            query_indices=[width-1],stream_positions=[width-1],capture=True)
                        with control:
                            audit.controller=control
                            for step,bundle in enumerate(bundles):
                                query=[width-1] if step==0 else [0];stream=[width+step-1];control.configure_queries(query,stream)
                                inputs,positions,layout=software.cached_input(model,bundle,cache,step,tokens,model.device)
                                audit.context=dict(label=f'{cid}__{rid}__cached{step}',case_id=cid,n_frames=n,phase='cached',step=step,
                                                   condition=mode,mode=mode,seed=selected['seed'],run_id=rid)
                                with torch.inference_mode():
                                    output=model(**inputs,use_cache=True,logits_to_keep=1) if 'use_cache' not in inputs else model(**inputs,logits_to_keep=1)
                                cache=output.past_key_values;raw=audit.last_raw
                                need(torch.equal(raw['position_ids'],positions.cpu()) and torch.equal(raw['native_norm_input'],baseline[step]['native_norm_input']),
                                     'Selected adapter changed fixed-prefix native hidden/positions')
                                checked=software.verify_fusion(raw,control,query,stream)
                                checked.update(label=audit.context['label'],run_id=rid,native_hidden_exact=True,
                                               all_layer_kv_exact=software.ancestor.all_kv_equal(cache,baseline_kv[step]));checks.append(checked)
                                if step>0:
                                    history=list(range(width-1,width+step));control.configure_queries(history,history)
                                    audit.context=dict(label=f'{cid}__{rid}__full{step}',case_id=cid,n_frames=n,phase='full',step=step,
                                                       condition=mode,mode=mode,seed=selected['seed'],run_id=rid)
                                    rope=software.cpu_copy(model.model.rope_deltas)
                                    try:
                                        with torch.inference_mode():full_output=model(**move_to_device(bundle['inputs'],model.device),use_cache=False,logits_to_keep=1)
                                    finally:model.model.rope_deltas=rope.to(model.device)
                                    full=audit.last_raw
                                    need(full_output.past_key_values is None and torch.equal(full['position_ids'],layout['position_ids']),
                                         'Full-prefix native positions/cache differ')
                                    if model_index==0:native_full[step]=full['native_norm_input']
                                    else:need(torch.equal(native_full[step],full['native_norm_input']),'Selected model changed full-prefix native reads')
                                    checked=software.verify_fusion(full,control,history,history)
                                    checked.update(label=audit.context['label'],run_id=rid,native_hidden_exact=True,all_layer_kv_exact=None);checks.append(checked)
                                    for row in range(n+1):
                                        metric=software.old.metric(torch,raw['native_logits'][row,-1],full['native_logits'][row,-1])
                                        metric.update(case_id=cid,n_frames=n,run_id=rid,mode=mode,seed=selected['seed'],step=step,row_index=row,
                                                      role='local' if row<n else 'global',binding=False);comparisons.append(metric)
                                    del full,full_output
                            need(control.calls==5,'Selected controller invocation count differs')
                            controllers.append(dict(case_id=cid,run_id=rid,calls=control.calls,passed=True))
                        audit.controller=None;need(not control.active,'Selected controller hook leaked')
                        need(training.state_info(core,predictor)==original,'Audit changed learned parameters')
                        restores.append(dict(case_id=cid,run_id=rid,parameter_sha256=native.object_sha(original),passed=True))
                        del cache,output,raw;gc.collect()
                    del baseline,baseline_kv,native_full,bundles
                need(audit.counts==CALLS,'Selected fixed-forward inventory differs')
            finally:
                records=audit.records;counts=dict(audit.counts)
                save(out/'forwards.json',records);save(out/'calls.json',counts)
                save(out/'structural_checks.json',dict(checks=checks,controller_counts=controllers,selected_state_restores=restores))
                save(out/'cached_full_comparisons.json',comparisons)
        need(len(records)==46 and len(checks)==40 and len(controllers)==len(restores)==8 and len(comparisons)==656
             and sum(r['all_layer_kv_exact'] is not None for r in checks)==24,'Selected structural coverage differs')
        metrics,raw_metrics=software.replay_heads(torch,model,records)
        need(len(metrics)==46,'Exactly one extra same-captured native head per model call required')
        torch.save(raw_metrics,data/'head_replays.pt');save(out/'head_replay_metrics.json',metrics)
    finally:save(out/'actual_calls.json',dict(calls=counts,standalone_head_replays=len(metrics) if 'metrics' in locals() else 0))
    need(versions=={k:p._version for k,p in model.named_parameters()}
         and not any(p.requires_grad or p.grad is not None for p in model.parameters()),'Frozen backbone changed')
    failures=[r for r in metrics if not r['numerical_rule_passed']];descriptive=[r for r in comparisons if not r['numerical_rule_passed']]
    models=[]
    for selected in plan['runs']:
        rid=selected['run_id'];rmetrics=[m for m in metrics if f'__{rid}__' in m['label']];cm=[m for m in comparisons if m['run_id']==rid]
        need(len(rmetrics)==10 and len(cm)==164,'Per-selected-model replay coverage differs')
        models.append(dict(run_id=rid,arm='parallel',condition=selected['condition'],seed=selected['seed'],
            selected_step=selected['selected']['step'],checkpoint=selected['selected']['checkpoint'],checkpoint_sha256=selected['selected']['checkpoint_sha256'],
            parameter_sha256=selected['selected']['parameter_sha256'],selected_parameter_sha256=selected['selected']['parameter_sha256'],replay_gate_passed=all(m['numerical_rule_passed'] for m in rmetrics),
            cache_full_numeric_passed=all(m['numerical_rule_passed'] for m in cm),head_replays=10,cache_full_rows=164,
            computational_integrity_passed=True,fixed_prefix_kv_checks=6))
    paths={name:out/name for name in ('forwards.json','calls.json','structural_checks.json','cached_full_comparisons.json','head_replay_metrics.json','actual_calls.json')}
    paths['head_replays.pt']=data/'head_replays.pt'
    artifacts={name:dict(path=str(path),sha256=sha(path)) for name,path in paths.items()}
    summary=dict(schema_version=1,protocol=PROTOCOL,completed=True,passed=not failures,computational_integrity_passed=True,
        replay_gate_passed=not failures,native_head_replay_passed=not failures,native_replay_gate_passed=not failures,
        cache_full_numeric_passed=not descriptive,strict_cache_numerical_gate_passed=not descriptive,
        strict_numerical_gates_passed=not failures and not descriptive,cached_full_numeric_is_descriptive=True,
        replay_failures=failures,cache_full_failures=descriptive,models=models,runs=models,plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),
        source_sha256=frozen,model=plan['model'],runtime=plan['runtime'],processor=plan['processor'],native_api=plan['native_api'],
        calls=counts,native_head_replay_calls=46,head_replay_calls=46,cached_full_row_comparisons=656,cache_full_rows=656,
        fixed_prefix_kv_checks=24,selected_fusion_checks=40,ordinary_generations=0,extra_last_block_forwards=0,
        artifacts=artifacts,files=artifacts,backend=plan['backend'],original_mixed_failure=plan['original_mixed_failure'],original_mixed_numerical_gate_passed=False,
        frozen_backbone_unchanged=True,selected_parameters_unchanged=True,no_fitting=True,no_efficacy_scoring=True,
        native_dtypes=dict(norm=str(norm.weight.dtype),lm_head=str(model.lm_head.weight.dtype)),
        hardware=dict(gpu=torch.cuda.get_device_name(),total_memory_bytes=torch.cuda.get_device_properties(0).total_memory),
        peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(),peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(),
        load_seconds=load_seconds,seconds=time.perf_counter()-begin,slurm_job_id=job)
    save(out/'summary.json',summary)
    print(dict(directory=str(out),passed=summary['passed'],calls=counts,binding_failures=len(failures),descriptive_failures=len(descriptive)),flush=True)
    need(summary['passed'],'Captured native head replay failed; all observations preserved')


def main():
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--self-test',action='store_true');modes.add_argument('--check',action='store_true');modes.add_argument('--run',action='store_true')
    parser.add_argument('--source-check',type=Path);parser.add_argument('--runs',nargs=4,type=Path);parser.add_argument('--plan',type=Path)
    args=parser.parse_args();native.require_slurm(gpu=args.run)
    if not args.run:need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),'CPU audit requires Slurm CPU')
    if args.self_test:unit()
    elif args.check:
        need(args.source_check is not None and args.runs is not None,'Pass pre-main source self-test and exactly four completed main directories');check(args)
    else:need(args.plan is not None,'Pass exact completed CPU checkpoint plan');run(args)


if __name__=='__main__':main()

"""Explicit V5 engineering release and descriptive checkpoint-audit validation.

The registered strict numerical criterion is never relaxed here. Schema2 may
accept exploratory native-backend training while that criterion remains false.
Computational-integrity failures always reject an artifact; numerical failures
are retained in every pre/post checkpoint audit and cannot select a model.
"""
from pathlib import Path
import json
import math

BACKEND=dict(quantization='nf4',activation_dtype='bfloat16',attention_implementation='sdpa',
             bitsandbytes_dispatch='native_bitsandbytes_dispatch_unmodified',kernel_route_override=False)
KINDS={'core_cpu','profile_sum_strict','profile_mean_failed','mean_completion_audit',
       'cache_localization_diagnostic','kernel_dispatch_diagnostic'}


def release_header(value,expected_hashes,v5):
    v5.need(value.get('schema_version')==2 and
        value.get('decision')=='explicit_conditional_engineering_acceptance' and
        value.get('accepted_for_exploratory_training') is True and
        value.get('strict_numerical_gates_passed') is False and 'passed' not in value,
        'Schema2 must explicitly retain strict failure; it cannot relabel engineering acceptance as passed')
    v5.need(value.get('source_sha256')==expected_hashes and value.get('native_backend_identity')==BACKEND,
            'Engineering release source/native backend identity differs')
    v5.need(isinstance(value.get('rationale'),str) and bool(value['rationale'].strip()) and
            isinstance(value.get('limitations'),list) and bool(value['limitations']) and
            all(isinstance(x,str) and x.strip() for x in value['limitations']),
            'Explicit engineering rationale and limitations are required')


def self_test_release(v5):
    value=dict(schema_version=2,decision='explicit_conditional_engineering_acceptance',
        accepted_for_exploratory_training=True,strict_numerical_gates_passed=False,
        source_sha256={'core':'a'*64},native_backend_identity=BACKEND,
        rationale='Prospective explicit acceptance despite unchanged failed numerical criterion.',
        limitations=['No exact cache equivalence claim.'])
    release_header(value,value['source_sha256'],v5)
    for key,wrong in (('strict_numerical_gates_passed',True),('passed',True),
                      ('accepted_for_exploratory_training',False),('rationale',''),
                      ('native_backend_identity',dict(BACKEND,kernel_route_override=True))):
        invalid=dict(value);invalid[key]=wrong
        try:release_header(invalid,value['source_sha256'],v5)
        except ValueError:pass
        else:raise AssertionError('Invalid engineering release accepted: '+key)


def artifact(entry,v5):
    path=Path(entry['path'])
    v5.need(path.is_absolute() and path.is_file() and v5.digest(path)==entry['sha256'],
            'Release/audit artifact path or hash changed')
    return path,v5.read(path)


def finite_tree(value,v5):
    if isinstance(value,dict):
        for item in value.values():finite_tree(item,v5)
    elif isinstance(value,list):
        for item in value:finite_tree(item,v5)
    elif isinstance(value,float):v5.need(math.isfinite(value),'Nonfinite audit metric')


def software_sources(v5):
    main=v5.read(v5.DATA_ROOT/'profile_manifest.json')
    count=v5.read(v5.DATA_ROOT/'profile_count_manifest.json')
    sources={}
    for value,family in ((main,'length'),(count,'unseen_count')):
        v5.need(value['schema_version']==1 and Path(value['dataset_root'])==v5.DATA_ROOT,
                'Invalid software manifest')
        for name,cell in value['splits'].items():
            if name.startswith('test_'):sources[name.replace('test_',family+'_',1)]=cell['samples']
    v5.need(set(sources)==set(v5.CELLS) and all(len(rows)==2 for rows in sources.values()),
            'Expected ten fixed software examples')
    cases=[sources[f'length_N{n}'][i] for n,i in ((16,0),(64,0),(16,1),(64,1),(16,0))]
    return sources,cases


def profile_provenance(path,condition,expected,v5):
    config=v5.read(path.parent/'config.json')
    v5.need(config['profile'] is True and config['condition']==condition and config['seed']==4 and
            config['architecture']==v5.architecture(condition) and config['code_sha256']==expected and
            v5.read(path.parent/'source_hashes.json')==expected,'Profile source/model identity differs')
    for name,sha in expected.items():
        v5.need(v5.digest(path.parent/'code'/name.replace('/','_'))==sha,'Profile source snapshot changed')
    for filename,key in (('profile_manifest.json','manifest_sha256'),('profile_count_manifest.json','count_manifest_sha256')):
        staged='staged_manifest.json' if filename=='profile_manifest.json' else 'staged_count_manifest.json'
        v5.need(v5.digest(path.parent/staged)==config[key]==v5.digest(v5.DATA_ROOT/filename),
                'Profile data identity differs')
    return config


def numeric_failures(row):
    result=[]
    for key,limit in (('centered_maximum_absolute_logit_difference','centered_maximum_absolute_tolerance'),
                      ('centered_rms_logit_difference','centered_rms_tolerance'),
                      ('total_variation','total_variation_tolerance')):
        if row[key]>row[limit]:result.append(key)
    if row['raw_top1_equal'] is not True:result.append('raw_top1_equal')
    return result


def audit_state(row,prompt_tokens,v5):
    state=row['state'];finite_tree(state,v5)
    for name in ('enabled_trace','disabled_trace'):
        trace=state[name]
        v5.need(trace['final_position']==prompt_tokens and trace['image_ends_exact'] is True and
                trace['complete_visible_images']==row['n_frames'] and trace['all_raw_memory_exact'] is True,
                'Audit memory/position/visibility integrity failed')
        for records in (trace['memory_per_image'],trace['rotary_cos_sin_last']):
            v5.need(bool(records) and all(item['exact'] is True and item['max_abs']==0 and item['rms']==0 for item in records),
                    'Audit raw-memory or rotary identity failed')
        v5.need(len(trace['memory_per_image'])==row['n_frames'] and len(trace['rotary_cos_sin_last'])==2,
                'Audit raw-memory/rotary coverage differs')
    replay=state['cached_branch_replay']
    v5.need(state['cached_replay_exact'] is True and replay['exact'] is True and replay['max_abs']==0 and replay['rms']==0,
            'Audit fixed-input cached branch replay failed')
    # Full replay is descriptive and deliberately has no new acceptance bound.
    v5.need(isinstance(state['full_branch_replay'],dict),'Missing descriptive full replay')


def audit_descriptive_cache(cache,cases,predictions,v5):
    rows=cache['observations'];by_sid={r['sid']:r for r in predictions}
    v5.need(cache['computational_integrity_passed'] is True and cache['numerical_failures_preserved'] is True and
            cache['criterion']=='paired_native_reference_centered_and_probability' and
            len(rows)==5 and cache['visual_calls']==20 and cache['reset_exact'] is True and
            [r['sid'] for r in rows]==[r['sid'] for r in cases],
            'Audit must complete five fixed A/B/C/D/A cases and20 visual forwards')
    flags=[];failures=[]
    for index,(row,source) in enumerate(zip(rows,cases)):
        v5.need(row['n_frames']==source['n_frames'] and row['observation_index']==index and
                row['reset_repeat_of']==(0 if index==4 else None) and
                row['case_role']==('prospective_unused_record' if index in (2,3) else 'original_case_or_reset'),
                'Audit prospective/reset case identity differs')
        flag=v5.audit_native_reference_observation(row,require_pass=False);flags.append(flag)
        v5.need(row['native_visual_calls_cached_generation']==row['native_visual_calls_uncached_comparison']==
                row['native_visual_calls_disabled_cached']==row['native_visual_calls_disabled_full']==1 and
                row['uncached_memory_cleared'] is True and v5.valid_sha(row['first_logits_sha256']) and
                v5.valid_sha(row['cached_logits_sha256']),'Audit native-call/cache-clear evidence differs')
        audit_state(row,by_sid[row['sid']]['prompt_tokens'],v5)
        if not flag:failures.append(dict(observation_index=index,sid=row['sid'],n_frames=row['n_frames'],
            failed_checks=numeric_failures(row),metrics=row))
    v5.need(all(rows[0][key]==rows[4][key] for key in ('sid','generated_ids','first_logits_sha256','cached_logits_sha256')),
            'Audit A/reset IDs or logits differ')
    v5.need(cache['strict_numerical_gate_passed'] is all(flags),'Cache summary relabeled a numerical failure')
    return dict(strict_numerical_gate_passed=all(flags),passed_observations=sum(flags),observations=5,
                failures=failures,all_observations=rows,computational_integrity_passed=True)


def verify_checkpoint_audit(entry,phase,expected,v5):
    path,payload=artifact(entry,v5);directory=path.parent
    v5.need(payload['schema_version']==1 and payload['phase']==phase and
            payload['backend']=='native_bitsandbytes_dispatch_unmodified' and
            payload['computational_integrity_passed'] is True,'Wrong audit phase/backend or failed integrity')
    plan_path=directory/'plan.json';plan=v5.read(plan_path)
    v5.need(v5.digest(plan_path)==payload['plan_sha256'] and plan['source_sha256']==payload['source_sha256'],
            'Audit plan/source binding differs')
    v5.need(plan['schema_version']==1 and plan['phase']==phase and
            plan['backend']=='native_bitsandbytes_dispatch_unmodified' and
            plan['frozen_profile_source_sha256']==expected,'Audit plan phase/backend/frozen main sources differ')
    import importlib.metadata
    v5.need(plan['bitsandbytes_version']==importlib.metadata.version('bitsandbytes'),
            'Installed native bitsandbytes version changed')
    for name,sha in plan['installed_backend_source_sha256'].items():
        v5.need(Path(name).is_absolute() and v5.digest(name)==sha,'Installed native backend source changed')
    for name,sha in plan['source_sha256'].items():
        v5.need(v5.digest(v5.REPO/name)==sha,'Frozen checkpoint-audit source changed')
    # Detailed run/checkpoint/data bindings below are independent of outcome flags.
    sources,cases=software_sources(v5)
    v5.need(set(plan['manifests'])=={'length','unseen_count'},'Audit manifest coverage differs')
    for family,name in (('length','profile_manifest.json'),('unseen_count','profile_count_manifest.json')):
        value=plan['manifests'][family]
        v5.need(Path(value['path'])==v5.DATA_ROOT/name and v5.digest(value['path'])==value['sha256'],
                'Audit software manifest changed')
    expected_records=[dict(row,cell=cell) for cell in v5.CELLS for row in sources[cell]]
    v5.need(plan['records']==expected_records and plan['cache_case_sids']==[row['sid'] for row in cases],
            'Audit planned software examples or cache cases differ')
    planned_runs={row['run_id']:row for row in plan['runs']}
    v5.need(len(planned_runs)==len(plan['runs'])==len(payload['runs']) and
            set(planned_runs)=={row['run_id'] for row in payload['runs']},'Audit run coverage differs from frozen plan')
    import torch
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(v5.architecture('sum')['model'],local_files_only=True,use_fast=False)
    results=[]
    for run in payload['runs']:
        config_path=Path(run['config_path']);config=v5.read(config_path)
        planned=planned_runs[run['run_id']]
        v5.need(all(run[key]==planned[key] for key in ('run_id','condition','seed','checkpoint','checkpoint_sha256',
                    'selected_epoch','config_path','config_sha256','source_code_sha256')) and
                planned['config']==config and Path(planned['directory'])==config_path.parent and
                Path(planned['training_path'])==config_path.parent/'training.json' and
                v5.digest(planned['training_path'])==planned['training_sha256'],
                'Audit run/model/checkpoint differs from frozen CPU plan')
        history=v5.read(planned['training_path'])
        v5.need(config['profile']==(phase=='pre') and len(history)==(2 if phase=='pre' else 9),
                'Audit phase/training history differs')
        best=max(history,key=lambda e:(sum(m['correct'] for m in e['dev'])/sum(m['n'] for m in e['dev']),
                 -sum(m['gold_first_token_nll']*m['n'] for m in e['dev'])/sum(m['n'] for m in e['dev'])))
        v5.need(best['epoch']==run['selected_epoch'],'Audit selected a checkpoint using a different criterion')
        v5.need(config_path.is_absolute() and v5.digest(config_path)==run['config_sha256'] and
                run['source_code_sha256']==config['code_sha256']==expected and
                config['run_id']==run['run_id'] and config['condition']==run['condition'] and
                config['seed']==run['seed'] and config['architecture']==v5.architecture(run['condition']),
                'Audit original model/config/source identity differs')
        checkpoint=Path(run['checkpoint'])
        v5.need(checkpoint.resolve()==v5.CHECKPOINT_ROOT/run['run_id']/'best.pt' and
                v5.digest(checkpoint)==run['checkpoint_sha256'],'Audit selected checkpoint changed')
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        v5.need(saved['architecture']==config['architecture'] and saved['epoch']==run['selected_epoch'] and
                saved['dev']==best['dev'] and saved['step']==best['step'] and
                all(saved['config'][key]==config[key] for key in ('run_id','condition','seed','code_sha256')),
                'Audit checkpoint metadata differs')
        del saved
        output=Path(run['output_directory'])
        v5.need(output.resolve().parent==directory and output.name==run['run_id'] and
                v5.digest(output/'config.json')==run['config_sha256'] and
                v5.read(output/'cache.json')==run['cache'] and
                v5.read(output/'predictions.json')==run['predictions'] and
                v5.read(output/'results.json')==run['results'] and v5.read(output/'summary.json')==run,
                'Audit copied run files differ')
        v5.need(run['computational_integrity_passed'] is True,'Audit computational integrity failed')
        v5.verify_predictions(run['predictions'],sources,v5.CELLS,'test',tokenizer)
        v5.verify_metrics(run['results'],run['predictions'],v5.CELLS)
        data_directory=v5.DATA_ROOT.parent/'v5_checkpoint_audits'/f"{phase}_{payload['slurm_job_id']}"/run['run_id']/'prefill_diagnostics'
        diagnostic=v5.verify_prefill_diagnostics(output,run['predictions'],run['condition'],torch,
            expected_count=10,directory_override=data_directory)
        v5.need(Path(run['tensor_directory'])==data_directory.parent,'Audit tensor root differs')
        for index,row in enumerate(run['cache']['observations']):
            trace=Path(row['tensors_path'])
            v5.need(trace==data_directory.parent/f'cache_case_{index}.pt' and
                    v5.digest(trace)==row['tensors_sha256'],'Audit numerical trace path/hash differs')
        numeric=audit_descriptive_cache(run['cache'],cases,run['predictions'],v5)
        v5.need(run['strict_numerical_gate_passed'] is numeric['strict_numerical_gate_passed'],
                'Checkpoint audit relabeled the strict numerical result')
        results.append(dict(run_id=run['run_id'],condition=run['condition'],seed=run['seed'],
            checkpoint=str(checkpoint),checkpoint_sha256=run['checkpoint_sha256'],selected_epoch=run['selected_epoch'],
            config_sha256=run['config_sha256'],numerical=numeric,prefill_diagnostics=diagnostic,
            predictions=run['predictions'],results=run['results']))
    v5.need(bool(results) and len({r['run_id'] for r in results})==len(results) and
            payload['strict_numerical_gate_passed'] is all(r['numerical']['strict_numerical_gate_passed'] for r in results),
            'Audit overall strict result or coverage differs')
    return dict(path=str(path),sha256=entry['sha256'],plan_sha256=payload['plan_sha256'],
        source_sha256=payload['source_sha256'],phase=phase,backend=payload['backend'],
        computational_integrity_passed=True,strict_numerical_gate_passed=payload['strict_numerical_gate_passed'],
        runs=results)


def verify_diagnostic(entry,kind,expected,failed_checkpoint,v5):
    path,value=artifact(entry,v5);plan=v5.read(path.parent/'plan.json')
    v5.need(v5.digest(path.parent/'plan.json')==value['plan_sha256'] and
            value['source_sha256']==plan['source_sha256'] and
            v5.read(path.parent/'observations.json')==value['observations'],'Diagnostic plan/observation hashes differ')
    core='gnnformer/independent_vision_aggregation.py'
    v5.need(plan['frozen_profile_source_sha256'][core]==expected[core],
            'Localization used a different aggregation core')
    if kind=='kernel_dispatch_diagnostic':
        v5.need(value['original_dispatch_restored'] is True and
                all(r['original_dispatch_restored'] is True for r in value['observations']) and
                plan['frozen_profile_source_sha256']==expected and plan['checkpoint']==failed_checkpoint['checkpoint'] and
                plan['checkpoint_sha256']==failed_checkpoint['checkpoint_sha256'],
                'Kernel diagnostic backend restoration or failed-checkpoint identity differs')
    # Earlier localization legitimately used earlier software-helper snapshots/checkpoints.
    return dict(path=str(path),sha256=entry['sha256'],plan_sha256=value['plan_sha256'],
        observations_sha256=v5.digest(path.parent/'observations.json'),source_sha256=value['source_sha256'],
        frozen_profile_source_sha256=plan['frozen_profile_source_sha256'],
        checkpoint_provenance=plan.get('runs') if kind=='cache_localization_diagnostic' else
            {key:plan[key] for key in ('checkpoint','checkpoint_sha256','selected_epoch')},
        scope='Binds immutable diagnostic summaries/plans; no new tensor replay or threshold calibration.')


def verify_engineering_release(root,expected,v5):
    path=root/'implementation_gates.json';value=v5.read(path)
    release_header(value,expected,v5)
    entries=value['artifacts'];kinds=[entry['kind'] for entry in entries]
    v5.need(len(kinds)==len(set(kinds)) and set(kinds)==KINDS,'Missing/duplicate engineering-release evidence')
    ledger={entry['kind']:entry for entry in entries}
    _,core=artifact(ledger['core_cpu'],v5)
    v5.need(core.get('passed') is True and core.get('unit_tests_passed') is True and
            core.get('schedule_checks_passed') is True and core['source_sha256']==expected,
            'Required CPU operator/schedule checks did not pass')
    sources,cases=software_sources(v5)
    sum_path,sum_profile=artifact(ledger['profile_sum_strict'],v5)
    profile_provenance(sum_path,'sum',expected,v5)
    v5.need(sum_profile.get('passed') is True,'SUM strict profile did not pass')
    zero=sum_profile['zero_initialization']
    v5.need(zero['passed'] is True and zero['maximum_absolute_logit_difference']==0 and
            zero['criterion']=='exact_equal' and zero['zero_up_projection'] is True and zero['uncached_memory_cleared'] is True,
            'SUM zero-initialization parity failed')
    strict_rows=v5.audit_cache_profile(sum_profile['nonzero_cache_and_reset'])
    v5.need([r['sid'] for r in strict_rows]==[r['sid'] for r in cases],'SUM strict software cases differ')
    import torch
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(v5.architecture('sum')['model'],local_files_only=True,use_fast=False)
    sum_rows=v5.read(sum_path.parent/'predictions.json')
    v5.verify_predictions(sum_rows,sources,v5.CELLS,'test',tokenizer)
    v5.verify_prefill_diagnostics(sum_path.parent,sum_rows,'sum',torch,expected_count=10)
    mean_path,mean_profile=artifact(ledger['profile_mean_failed'],v5)
    mean_config=profile_provenance(mean_path,'mean',expected,v5)
    v5.need(mean_profile.get('passed') is not True and not (mean_path.parent/'summary.json').exists(),
            'Failed MEAN profile was incorrectly marked complete/passed')
    zero=mean_profile['zero_initialization']
    v5.need(zero['passed'] is True and zero['maximum_absolute_logit_difference']==0 and
            zero['criterion']=='exact_equal','Failed MEAN profile lacks its original zero-U equality evidence')
    log=ledger['profile_mean_failed']['failure_log'];log_path=Path(log['path'])
    v5.need(log_path.is_absolute() and v5.digest(log_path)==log['sha256'],'Failed-profile log changed')
    failed_rows=[]
    for line in log_path.read_text().splitlines():
        if line.startswith('{"v5_centered_cache_gate":'):
            failed_rows.append(json.loads(line)['v5_centered_cache_gate'])
    v5.need(len(failed_rows)==4 and [r['sid'] for r in failed_rows]==[r['sid'] for r in cases[:4]],
            'Failed MEAN profile did not stop on its fixed fourth prospective case')
    flags=[v5.audit_native_reference_observation(r,require_pass=False) for r in failed_rows]
    v5.need(flags==[True,True,True,False],'The failed profile numerical history differs')
    pre=verify_checkpoint_audit(ledger['mean_completion_audit'],'pre',expected,v5)
    v5.need(len(pre['runs'])==1 and pre['runs'][0]['run_id']==mean_config['run_id'] and
            pre['runs'][0]['condition']=='mean' and pre['runs'][0]['seed']==4,
            'Pre-release completion must retain the SAME failed MEAN checkpoint')
    model=pre['runs'][0]
    diagnostics={kind:verify_diagnostic(ledger[kind],kind,expected,model,v5)
        for kind in ('cache_localization_diagnostic','kernel_dispatch_diagnostic')}
    return dict(path=str(path.resolve()),sha256=v5.digest(path),schema_version=2,
        accepted_for_exploratory_training=True,strict_numerical_gates_passed=False,
        decision=value['decision'],native_backend_identity=value['native_backend_identity'],
        rationale=value['rationale'],limitations=value['limitations'],artifacts=entries,
        failed_mean_original_observations=failed_rows,pre_release_checkpoint_audit=pre,diagnostics=diagnostics,
        scope='Explicit conditional engineering acceptance; the registered strict numerical criterion remains FAILED.')


def verify_post_training_audits(root,expected,results,runs,v5):
    path=root/'post_training_audits.json';ledger=v5.read(path)
    v5.need(ledger.get('schema_version')==1 and bool(ledger.get('artifacts')),'Missing post-training audit ledger')
    artifacts=[verify_checkpoint_audit(entry,'post',expected,v5) for entry in ledger['artifacts']]
    audited=[run for item in artifacts for run in item['runs']]
    v5.need(len(audited)==4 and {(r['condition'],r['seed']) for r in audited}==
            {(condition,seed) for condition in v5.CONDITIONS for seed in v5.SEEDS},
            'All four dev-selected checkpoints must be audited without dropping numerical failures')
    for item in audited:
        condition,seed=item['condition'],item['seed'];info=results[condition][seed]
        v5.need(item['run_id']==Path(runs[condition][seed]).name and
                item['checkpoint']==info['selected_checkpoint'] and
                item['checkpoint_sha256']==info['selected_checkpoint_sha256'] and
                item['selected_epoch']==info['selected_epoch'],'Post audit differs from the dev-selected main checkpoint')
    return dict(path=str(path.resolve()),sha256=v5.digest(path),artifacts=artifacts,
        computational_integrity_passed=True,strict_numerical_gate_passed=all(a['strict_numerical_gate_passed'] for a in artifacts),
        checkpoint_count=4,observations=20,numerically_passing_observations=sum(r['numerical']['passed_observations'] for r in audited),
        note='All fixed post-training observations are retained; no numerical result selects checkpoints or changes main predictions.')

"""Independent V6 post-checkpoint audit verification, with frozen V5 metric checks."""
from pathlib import Path
from scripts.report_native_vision_v5_release import artifact,software_sources,audit_descriptive_cache

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
    tokenizer=AutoTokenizer.from_pretrained(v5.v5_architecture('sum')['model'],local_files_only=True,use_fast=False)
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
                config['seed']==run['seed'] and config['architecture']==v5.architecture(run['condition'],v5.digest(Path(config['teacher_index']))),
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
        data_directory=v5.DATA_ROOT.parent/'v6_checkpoint_audits'/f"{phase}_{payload['slurm_job_id']}"/run['run_id']/'prefill_diagnostics'
        diagnostic=v5.verify_prefill_diagnostics(output,run['predictions'],'sum',torch,
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

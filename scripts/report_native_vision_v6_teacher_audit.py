"""Independent reconstruction of the V6 training teacher from archived raw shards."""
from pathlib import Path
import math
import os

TARGET_FIELDS=frozenset(('probabilities','image_sha256','question','question_sha256',
    'logit0','logit1','log_normalizer','layout_id','input_ids_sha256'))


def verify_shards(index,plan):
    from scripts.stage_native_vision_v6_teacher import DATA,OUTPUT,REPO,need,read,sha
    from scripts.cache_native_vision_v6_teacher import read_rows,verify_profile,verify_row
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu',
         'Teacher archive audit requires Slurm CPU')
    plan_path=Path(index['plan_file'])
    need(sha(plan_path)==index['plan_sha256'] and read(plan_path)==plan,'Wrong teacher plan')
    need(plan['protocol']=='v6_training_only_local_count_teacher','Wrong teacher protocol')
    for field in ('source_sha256','model','processor','runtime','source_files',
                  'image_processor_settings','resize','quantization','attention','temperature'):
        need(index[field]==plan[field],f'Teacher index/plan differs: {field}')
    need(plan['resize']==392 and plan['quantization']=='nf4_double_bf16'
         and plan['attention']=='sdpa' and plan['temperature']==1,'Teacher backend differs')
    ids=sorted(plan['pairs'])
    need(len(ids)==9980 and plan['shards']==[ids[i::4] for i in range(4)],'Teacher partition differs')
    entries=index['shard_provenance'];need(len(entries)==4,'Expected four teacher shards')
    checked={str(plan_path):index['plan_sha256']}
    for name,expected in plan['source_sha256'].items():
        need(sha(REPO/name)==expected,'Teacher source changed');checked[str(REPO/name)]=expected
    for value in plan['source_files'].values():
        need(sha(value['path'])==value['sha256'],'Teacher input changed');checked[value['path']]=value['sha256']
    seen=set();reconstructed={};profiles={};verified=[];model_calls=visual_calls=0
    for entry in entries:
        directory=Path(entry['directory'])
        need(directory.is_absolute() and directory.resolve().parent==OUTPUT.resolve()
             and not directory.is_symlink(),'Wrong teacher shard directory')
        summary_path=directory/'summary.json';config_path=directory/'config.json'
        for path,key in ((summary_path,'summary_sha256'),(config_path,'config_sha256')):
            need(sha(path)==entry[key],'Archived teacher metadata changed');checked[str(path)]=entry[key]
        summary,config=read(summary_path),read(config_path);shard=summary['shard']
        need(type(shard) is int and shard in range(4) and shard not in seen,'Duplicate teacher shard');seen.add(shard)
        need(summary['schema_version']==config['schema_version']==1
             and summary['completed'] is True and summary['computational_integrity_passed'] is True
             and summary['profile'] is False and config['profile'] is False and config['shard']==shard,
             'Incomplete teacher shard')
        need(summary['slurm_job_id']==config['slurm_job_id']==entry['slurm_job_id']
             and directory.name==f'shard{shard}_{summary["slurm_job_id"]}','Wrong teacher job identity')
        need(summary['source_sha256']==config['source_sha256']==plan['source_sha256']
             and summary['plan_sha256']==config['plan_sha256']==index['plan_sha256']
             and Path(config['plan_file']).resolve()==plan_path.resolve(),'Teacher source/plan changed')
        need(sha(directory/'plan.json')==index['plan_sha256'],'Teacher plan copy changed')
        checked[str(directory/'plan.json')]=index['plan_sha256']
        need(read(directory/'source_hashes.json')==plan['source_sha256'],'Teacher source ledger differs')
        for name,expected in plan['source_sha256'].items():
            snapshot=directory/'source'/name.replace('/','_')
            need(sha(snapshot)==expected,'Teacher source snapshot differs');checked[str(snapshot)]=expected
        for field in ('model','processor','runtime'):need(config[field]==plan[field],'Teacher backend record differs')
        need(config['operation']=='one frozen isolated-image native prefill; no generation or labels in inputs',
             'Teacher operation differs')
        profile_dir=config['profile_provenance']['directory']
        if profile_dir not in profiles:profiles[profile_dir]=verify_profile(profile_dir,plan_path,plan)
        need(profiles[profile_dir]==config['profile_provenance']==summary['source_profile'],'Profile provenance differs')
        checked[str(Path(profile_dir)/'summary.json')]=profiles[profile_dir]['summary_sha256']
        raw=Path(summary['rows_file'])
        need(raw.is_absolute() and not raw.is_symlink() and raw.resolve()==(DATA/directory.name/'rows.jsonl').resolve()
             and entry['rows_file']==str(raw),'Wrong archived teacher rows path')
        need(sha(raw)==summary['rows_sha256']==entry['rows_sha256'],'Teacher raw rows changed')
        checked[str(raw)]=summary['rows_sha256'];rows=read_rows(raw)
        need(list(rows)==plan['shards'][shard]==config['pair_ids'] and len(rows)==summary['pair_count']==2495,
             'Teacher shard subset/order differs')
        for pid,row in rows.items():
            need(pid not in reconstructed,'Teacher shards overlap');verify_row(row,plan['pairs'][pid],plan)
            target={field:row[field] for field in TARGET_FIELDS}
            need(set(index['targets'][pid])==TARGET_FIELDS and index['targets'][pid]==target,
                 'Published teacher target differs from archived raw row')
            reconstructed[pid]=target;model_calls+=row['native_model_forwards'];visual_calls+=row['native_visual_forwards']
        for field in ('pair_seconds','forward_seconds'):
            need(math.isfinite(summary[field]) and summary[field]>=0 and
                 math.isclose(summary[field],sum(row[field] for row in rows.values()),rel_tol=1e-10,abs_tol=1e-8),
                 'Teacher timing arithmetic differs')
        need(math.isfinite(summary['total_seconds']) and summary['total_seconds']>=0
             and entry['total_seconds']==summary['total_seconds'],'Teacher elapsed record differs')
        expected_entry=dict(directory=str(directory.resolve()),summary_sha256=sha(summary_path),
            config_sha256=sha(config_path),rows_file=str(raw),rows_sha256=sha(raw),
            slurm_job_id=summary['slurm_job_id'],total_seconds=summary['total_seconds'])
        need(entry==expected_entry,'Published shard provenance differs');verified.append(dict(shard=shard,pairs=len(rows),**entry))
    need(seen==set(range(4)) and len(reconstructed)==9980 and reconstructed==index['targets'],
         'Incomplete raw teacher reconstruction')
    need(model_calls==visual_calls==9980,'Teacher forward totals differ')
    need(all(sha(path)==expected for path,expected in checked.items()),'Teacher artifact changed during audit')
    return dict(passed=True,shards=sorted(verified,key=lambda r:r['shard']),unique_pairs=9980,
        exact_published_targets_reconstructed=True,exact_shard_partitions_and_order_verified=True,
        source_snapshots_verified=True,recorded_backend_identity_verified=True,native_model_forwards=model_calls,
        native_visual_forwards=visual_calls,profile_provenance=list(profiles.values()),checked_file_count=len(checked),
        plan_sha256=index['plan_sha256'],note='Recorded artifact/source chain verified; no model execution replay.')

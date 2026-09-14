"""Supplemental metadata release for the immutable parallel V7 feature plan.

Verifies the balanced data's semantic/staging/prior ancestry without rewriting
the already-frozen parallel feature programs or their output. This is a metadata
and hash release, not a new image/model execution or scientific efficacy gate.
Run only in CPU Slurm.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import stage_native_vision_v7_features as feature
need,read,save,sha,object_sha=feature.need,feature.read,feature.save,feature.sha,feature.object_sha
OUT=REPO/'outputs/native_aggregation_vlm/v7/features/metadata_release'
OWN=('scripts/release_native_vision_v7_feature_metadata.py','slurm/native_vision_v7_feature_metadata.sbatch')


def verify_hash_map(values,label):
    need(isinstance(values,dict) and values,label+' must be a nonempty hash mapping')
    for name,digest in values.items():
        path=Path(name)
        need(path.is_absolute() and path.is_file() and isinstance(digest,str) and len(digest)==64
             and sha(path)==digest,label+' changed: '+name)


def freshness(rows,excluded):
    identities=[row['content_sha256'] for row in rows]
    need(len(set(identities))==len(identities),'Current contexts must be mutually distinct')
    reused=[]
    for row in rows:
        if row['content_sha256'] not in excluded:continue
        need(row['split']=='train' and row['n_frames']==8 and row['gold']==8,
             'Historical overlap outside the registered saturated training exception')
        reused.append(dict(sid=row['sid'],content_sha256=row['content_sha256']))
    return sorted(reused,key=lambda row:row['sid'])


def self_test():
    saturated=dict(sid='s',split='train',n_frames=8,gold=8,content_sha256='a')
    fresh=dict(sid='f',split='dev',n_frames=16,gold=1,content_sha256='b')
    need(freshness([saturated,fresh],{'a'})==[dict(sid='s',content_sha256='a')],'Saturated exception changed')
    for rows,excluded in (([dict(fresh,content_sha256='a')],{'a'}),([saturated,saturated],set())):
        try:freshness(rows,excluded)
        except ValueError:pass
        else:raise AssertionError('Disallowed metadata overlap accepted')
    return dict(passed=True,tests=['saturation_exception','non_saturated_overlap_rejection','current_duplicate_rejection'])


def release(args):
    start=time.monotonic();tests=self_test();source={name:sha(REPO/name) for name in OWN}
    plan=feature.verify_plan(args.plan,pixels=False);stage=Path(args.stage_directory).resolve()
    need(stage.is_relative_to(feature.OUT),'Parallel CPU stage must be under canonical feature output root')
    summary_file=stage/'summary.json';summary=read(summary_file)
    need(summary['passed'] is True and Path(summary['plan_file']).resolve()==args.plan.resolve()
         and summary['plan_sha256']==sha(args.plan) and summary['source_sha256']==plan['source_sha256']
         and summary['counts']==plan['counts'] and summary['total_features']==len(plan['features']),
         'Parallel CPU summary does not bind the frozen plan')
    need(read(stage/'source_hashes.json')==plan['source_sha256'],'Parallel CPU source ledger differs')
    for name,digest in plan['source_sha256'].items():
        need(sha(stage/'source'/name.replace('/','_'))==digest,'Parallel frozen source snapshot changed')
    manifest_binding=plan['source_files']['training_manifest'];schedule_binding=plan['source_files']['training_schedule']
    manifest=read(manifest_binding['path']);schedule=read(schedule_binding['path'])
    need(sha(manifest_binding['path'])==manifest_binding['sha256']
         and sha(schedule_binding['path'])==schedule_binding['sha256']
         and schedule['manifest_sha256']==manifest_binding['sha256'],'Balanced manifest/schedule binding changed')
    bound={}
    for label in ('audit','stage_plan','prior_inventory'):
        path=Path(manifest[label+'_file']);digest=manifest[label+'_sha256']
        need(path.is_relative_to(feature.DATA.parent/'v7_balanced') and sha(path)==digest,
             'Balanced ancestor changed: '+label)
        bound[label]=dict(path=str(path),sha256=digest)
    audit=read(bound['audit']['path']);stage_plan=read(bound['stage_plan']['path']);prior=read(bound['prior_inventory']['path'])
    flags=('passed','all_current_contexts_disjoint','all_qa_counts_categories_images_verified',
           'global_prompt_has_no_set_size_argument','non_saturated_contexts_fresh_against_18_prior_manifests',
           'question_count_support_uniform_in_slots')
    need(all(audit[name] is True for name in flags),'A balanced semantic audit flag failed')
    need(audit['stage_plan_sha256']==bound['stage_plan']['sha256']
         and stage_plan['purpose']=='v7_balanced_training_and_dev','Semantic/staging provenance differs')
    generator_sources=manifest['source_sha256']
    need(generator_sources==audit['source_sha256']==stage_plan['source_sha256'],'Generator source ledgers differ')
    for name,digest in generator_sources.items():need(sha(REPO/name)==digest,'Balanced generator source changed')
    protected=stage_plan['protected_source_sha256'];priors=stage_plan['prior_manifest_sha256']
    need(protected==prior['protected_source_sha256'] and priors==prior['source_manifest_sha256']
         and len(protected)==20 and len(priors)==18,'Protected/prior manifest maps differ')
    verify_hash_map(protected,'Protected source');verify_hash_map(priors,'Prior manifest')
    need({row['path']:row['sha256'] for row in prior['entries']}==priors,'Prior inventory entries differ from source map')
    excluded=prior['excluded_content_sha256']
    need(excluded==sorted(set(excluded)) and len(excluded)==prior['excluded_content_count']
         and object_sha(excluded)==prior['excluded_content_set_sha256']
         and prior['exclusions_are_complete_context_and_question_hashes'] is True,
         'Prior exclusion inventory is malformed')
    train=manifest['splits']['train_N8']['samples']+manifest['splits']['train_N16']['samples']
    dev=manifest['splits']['dev_N16']['samples'];rows=train+dev
    need(len(train)==1890 and len(dev)==72 and len(rows)==audit['unique_contexts']==1962
         and set(row['sid'] for row in train)==set(stage_plan['train_sids'])==set(plan['scenes'])
         and set(row['sid'] for row in dev)==set(stage_plan['dev_sids']),'Training/dev stage coverage differs')
    reused=freshness(rows,set(excluded))
    need(reused==sorted(audit['historical_saturated_reuse'],key=lambda row:row['sid']) and len(reused)==39,
         'Registered historical saturation reuse differs')
    by_sid={row['sid']:row for row in train};slots=schedule['epoch_slots']
    need(len(slots)==1944 and slots==stage_plan['epoch_slots'],'Balanced epoch slot sequence differs')
    support=Counter((by_sid[sid]['question'],by_sid[sid]['n_frames'],by_sid[sid]['gold']) for sid in slots)
    questions={row['question'] for row in train}
    need(len(questions)==54 and support==Counter({(q,n,k):2 for q in questions for n in (8,16) for k in range(9)}),
         'Training slots do not cover the balanced full factorial')
    for row in train:
        cached=plan['scenes'][row['sid']]
        need(all(cached[key]==row[key] for key in ('question','n_frames','gold','qa_sha256','content_sha256')),
             'Feature scene metadata differs from balanced training manifest')
    job=os.environ['SLURM_JOB_ID'];out=OUT/f'release_{job}';out.mkdir(parents=True,exist_ok=False)
    ledger=dict(schema_version=1,protocol='v7_parallel_balanced_metadata_release',passed=True,slurm_job_id=job,
                parallel_plan=dict(path=str(args.plan.resolve()),sha256=sha(args.plan)),
                parallel_stage_summary=dict(path=str(summary_file),sha256=sha(summary_file)),
                parallel_source_sha256=plan['source_sha256'],balanced_manifest=manifest_binding,
                balanced_schedule=schedule_binding,balanced_ancestors=bound,generator_source_sha256=generator_sources,
                protected_source_sha256=protected,prior_manifest_sha256=priors,
                checks=dict(balanced_audit_flags={key:audit[key] for key in flags},
                    source_snapshots_verified=True,protected_prior_hashes_verified=True,
                    current_metadata_disjoint=True,non_saturated_metadata_fresh=True,
                    saturated_historical_reuse=reused,balanced_1944_slot_support=True,
                    feature_training_scene_metadata_matches=True),
                feature_counts=plan['counts'],training_scenes=1890,development_scenes=72,
                source_sha256=source,tests=tests,seconds=time.monotonic()-start,
                scope='Supplemental metadata/hash verification; no image rerender/recount, model call, or efficacy result')
    save(out/'release.json',ledger);(out/'release.sha256').write_text(sha(out/'release.json')+'\n')
    (out/'source').mkdir()
    for name in OWN:(out/'source'/name.replace('/','_')).write_bytes((REPO/name).read_bytes())
    save(out/'source_hashes.json',source)
    feature.index(out,'V7 parallel supplemental metadata release',[('Release ledger','release.json'),('Source hashes','source_hashes.json')])
    need({name:sha(REPO/name) for name in OWN}==source,'Release verifier source changed during execution')
    print(json.dumps(dict(passed=True,release_file=str(out/'release.json'),sha256=sha(out/'release.json')),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--stage-directory',type=Path,required=True);args=parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu'
         and not os.environ.get('SLURM_JOB_GPUS'),'Metadata release requires CPU Slurm')
    release(args)


if __name__=='__main__':main()


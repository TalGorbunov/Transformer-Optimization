"""Proposed V14 fresh test preparation; exact CPU dry plan before rendering.

272 contexts: eight independent families per K0..16, each N32 anchor extended
to N64 by the unchanged canonical negative-insertion law. No training/dev data,
model calls, or new fitting release. Protect literal V10 training/dev/schedule
and exclude28 prior manifests. Fresh means complete context/question freshness,
not new visual atoms; displayed Step labels are rerendered at new positions.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import stage_native_vision_v10_data as prior
from scripts import stage_native_vision_v11_test as old_fresh
from scripts.stage_native_vision_v9_test import content_rows,canonical_sha
from scripts.stage_native_vision_v2_clean import (DATA_BASE,V1_ROOT,LAW,describe,exclusive_json,
    make_family,prepare_renderer,publish_sample,render_cached,stable_seed)
from scripts.stage_native_vision_pilot import read_candidate,source_states
from scripts.audit_native_vision_data import inspect,summarize

DESTINATION=DATA_BASE/'v14_fresh'
OUT=REPO/'outputs/native_aggregation_vlm/v14/data_staging'
SEED=20261108
PROTOCOL='v14_fresh_test_preparation_only'
SPEC=dict(unique_samples=272,unique_content=272,families=136,image_references=13056,
          paired_extensions=136,test_N32=136,test_N64=136,prior_manifests=28)
OWN=tuple(dict.fromkeys(('scripts/stage_native_vision_v14_test.py','slurm/native_vision_v14_test_stage.sbatch',*old_fresh.OWN)))


def need(condition,message):
    if not condition:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())
def digest(path):return prior.digest(Path(path))
def source_hashes():return {name:digest(REPO/name) for name in OWN}


def snapshot(out):
    (out/'source').mkdir();frozen=source_hashes()
    for name,h in frozen.items():
        value=(REPO/name).read_bytes();need(hashlib.sha256(value).hexdigest()==h,'Source changed during snapshot')
        (out/'source'/name.replace('/','_')).write_bytes(value)
    exclusive_json(out/'source_hashes.json',frozen)
    (out/'INDEX.md').write_text('# Proposed V14 fresh test preparation\n\n[Plan](plan.json) · [Summary](summary.json) · [Sources](source_hashes.json) · [Report](REPORT.md).\n')
    index=OUT/'INDEX.md'
    if not index.exists():index.write_text('# V14 fresh test data preparation\n\n')
    with index.open('a') as stream:stream.write(f'- [{out.name}]({out.name}/INDEX.md)\n')
    return frozen


def prior_inputs():
    # The complete V13 test is now historical. Keep every canonical path entry,
    # including older manifests which happen to share a content digest.
    inventory,manifests,excluded=old_fresh.prior_inputs()
    need(len(inventory['entries'])==len(manifests)==27,'Require exact V13 prior27 inventory')
    prior27=dict(inventory['source_manifest_sha256'])
    root=DATA_BASE/'v11_fresh';path=root/'main_manifest.json'
    need(path.is_file() and not path.is_symlink(),'Missing canonical completed V13 test manifest')
    manifest=read(path)
    need(manifest['dataset_root']==str(root) and manifest['fresh_test_seed']==manifest['data_seed']==20261010
         and manifest['source_manifest_sha256']==prior27 and manifest['source_sha256']==old_fresh.source_hashes(),
         'Frozen V13 test ancestry or generator changed')
    rows=list(content_rows(manifest));hashes={r['content_sha256'] for r in rows};h=digest(path)
    need(len(rows)==len(hashes)==272 and str(path) not in manifests
         and all(isinstance(x,str) and len(x)==64 and all(c in '0123456789abcdef' for c in x) for x in hashes),
         'Invalid V13 test exclusion manifest')
    inventory['entries'].append(dict(path=str(path),sha256=h,record_count=len(rows),unique_content_count=len(hashes)))
    inventory['source_manifest_sha256'][str(path)]=h
    inventory['protected_source_sha256'][str(path)]=h
    manifests[str(path)]=manifest;excluded.update(hashes)
    for field in ('audit','stage_plan','inventory','prior_inventory'):
        target=Path(manifest[field+'_file']);expected=manifest[field+'_sha256']
        need(target.is_file() and not target.is_symlink() and digest(target)==expected,
             'V13 data audit/plan/inventory changed')
        inventory['protected_source_sha256'][str(target)]=expected
    stage_audit=Path(manifest['stage_audit_file']);audit=read(stage_audit)
    need(stage_audit.is_file() and not stage_audit.is_symlink() and audit['passed'] is True
         and audit['completed'] is True and audit['manifest_paths']['main']==str(path)
         and audit['manifest_sha256']['main']==h,'V13 published test did not pass its bound audit')
    inventory['protected_source_sha256'][str(stage_audit)]=digest(stage_audit)
    need(len(inventory['entries'])==len(manifests)==len(inventory['source_manifest_sha256'])==28,
         'Require exactly28 prior canonical manifests')
    ordered=sorted(excluded)
    inventory.update(excluded_content_sha256=ordered,excluded_content_count=len(ordered),
        excluded_content_set_sha256=canonical_sha(ordered),
        scope='Exact V13 prior27 plus the now-used V11fresh main test;28 canonical manifests',
        permitted_historical_reuse='None. All proposed V14 contexts are nonsaturated N32/N64 with K0..16.',
        training_dev_schedule_pairing_unchanged=True,exclusions_are_complete_context_and_question_hashes=True,
        local_atom_repetition_allowed=True)
    return inventory,manifests,excluded


def generate(excluded):
    used=set(excluded);rows=[]
    for k in range(17):
        for replica in range(8):
            label=f'v14_test_K{k}_{replica:03d}';seed=stable_seed(SEED,label)
            family=make_family(random.Random(seed),(32,64),k,label,'test','v14_length',used,seed)
            for row in family:
                row.update(replica=replica,historical_content_reuse=False,saturated_training_exception=False)
            rows.extend(family)
    need(len(rows)==272 and len(used)==len(excluded)+272,'Generated cardinalities or exclusions differ')
    return rows


def check_generated(rows,excluded):
    need(len(rows)==272 and Counter((r['n_frames'],r['gold']) for r in rows)==
         Counter({(n,k):8 for n in (32,64) for k in range(17)}),'Generated N/K balance differs')
    content=set();sids=set();families={};image_references=0
    for row in rows:
        states,q,h,counts=describe(row['frames'],row['target_character'],row['target_room'])
        need(states==row['states'] and q==row['question'] and h==row['content_sha256']
             and counts==row['semantic_counts'] and counts['matches']==row['gold'],'Generated semantic identity differs')
        need(h not in excluded and h not in content and row['sid'] not in sids,'Prior/current complete-context collision')
        need(row['split']=='test' and row['gold']<row['n_frames'] and not row['historical_content_reuse']
             and not row['saturated_training_exception'],'No training/saturation/reuse is allowed')
        need(len(states)==row['n_frames'] and all(s['step_id']==i+1 for i,s in enumerate(states)),'Step/frame indices differ')
        content.add(h);sids.add(row['sid']);image_references+=row['n_frames']
        members=families.setdefault(row['pair_id'],{})
        need(row['n_frames'] not in members,'Duplicate family length');members[row['n_frames']]=row
    for pid,members in families.items():
        need(set(members)=={32,64},'Incomplete family')
        a,b=members[32],members[64];positions=b['parent_positions']
        need(a['anchor_id']==b['anchor_id']==pid and a['question']==b['question'] and a['gold']==b['gold']
             and a['target_character']==b['target_character'] and a['target_room']==b['target_room']
             and a['replica']==b['replica'] and a['anchor_n_frames']==b['anchor_n_frames']==32,'Family target/anchor differs')
        need(a['parent_n_frames'] is None and a['parent_positions']==[] and a['anchor_positions']==list(range(32))
             and b['parent_n_frames']==32 and positions==sorted(set(positions)) and len(positions)==32
             and b['anchor_positions']==positions and [b['frames'][i] for i in positions]==a['frames'],
             'Semantic parent/anchor mapping differs')
        inserted=[f for i,f in enumerate(b['frames']) if i not in set(positions)]
        need(len(inserted)==32 and describe(inserted,b['target_character'],b['target_room'])[3]['matches']==0,
             'Extension introduced positive evidence')
    need(len(content)==272 and len(families)==136 and image_references==13056,'Generated totals differ')
    return dict(passed=True,**SPEC,all_generated_semantics_and_extensions_passed=True,
        all_current_contexts_disjoint=True,all_fresh_content_exclusions_passed=True,
        all_pair_extensions_passed=True,historical_content_reuse_count=0,saturation_exceptions=[])


def verify_protected(plan,frozen):
    need(source_hashes()==frozen==plan['source_sha256'],'Staging source changed')
    need(all(digest(path)==h for path,h in plan['protected_source_sha256'].items()),'Protected prior artifact changed')


def dry(out,frozen):
    need(not DESTINATION.exists() and not DESTINATION.is_symlink(),'Preserve existing/partial proposed V14 data')
    inventory,_,excluded=prior_inputs();rows=generate(excluded);checked=check_generated(rows,excluded)
    need(canonical_sha(rows)==canonical_sha(generate(excluded)),'Nondeterministic proposed test generation')
    first=make_family(random.Random(17),(32,64),3,'software','test','software',set(),17)
    used={r['content_sha256'] for r in first}
    second=make_family(random.Random(17),(32,64),3,'software','test','software',set(used),17)
    need(all(r['generation_attempt']>0 and r['content_sha256'] not in used for r in second),'Collision rejection failed')
    checked.update(deterministic_generation_passed=True,collision_rejection_passed=True)
    exclusive_json(out/'samples.json',rows);exclusive_json(out/'exclusion_inventory.json',inventory)
    exclusive_json(out/'generator_checks.json',checked)
    plan=dict(schema_version=1,protocol=PROTOCOL,dataset_root=str(DESTINATION),data_seed=SEED,fresh_test_seed=SEED,
        expected=SPEC,source_sha256=frozen,samples_file=str(out/'samples.json'),samples_sha256=digest(out/'samples.json'),
        samples_object_sha256=canonical_sha(rows),inventory_file=str(out/'exclusion_inventory.json'),
        inventory_sha256=digest(out/'exclusion_inventory.json'),generator_checks_file=str(out/'generator_checks.json'),
        generator_checks_sha256=digest(out/'generator_checks.json'),
        source_manifest_sha256=inventory['source_manifest_sha256'],protected_source_sha256=inventory['protected_source_sha256'],
        generator_law=LAW,sampling='Canonical V2 family law conditioned on exclusion against28 prior manifests and all new families',
        training_dev_schedule_pairing='Literal V10 files; no new training/dev scenes or schedule changes',
        standard_downstream_resize=392,local_atom_repetition_allowed=True,saturation_exceptions=[],
        no_model_calls=True,preparation_only=True,efficacy_training_released=False)
    exclusive_json(out/'plan.json',plan);(out/'plan.sha256').write_text(digest(out/'plan.json')+'\n')
    verify_protected(plan,frozen)
    summary=dict(passed=True,dry_check_passed=True,no_rendering=True,no_model_calls=True,preparation_only=True,
        plan_file=str(out/'plan.json'),plan_sha256=digest(out/'plan.json'),source_sha256=frozen,checks=checked)
    (out/'REPORT.md').write_text('# V14 prospective test dry plan\n\nPASS: 272 distinct contexts,136 N32→N64 families, eight/K0–16. All28 prior manifests and literal V10 training/dev/schedule/pairing are protected. No images rendered or training released.\n')
    return summary


def load_plan(path,frozen):
    path=Path(path).resolve();need(path.is_relative_to(OUT),'Require a canonical V14 dry plan')
    plan=read(path);need(digest(path)==path.with_suffix('.sha256').read_text().strip(),'Dry-plan sidecar differs')
    summary=read(path.parent/'summary.json')
    need(summary['passed'] and summary['dry_check_passed'] and summary['no_rendering']
         and summary['plan_sha256']==digest(path) and Path(summary['plan_file']).resolve()==path,'Matching passed dry gate required')
    need(plan['protocol']==PROTOCOL and plan['dataset_root']==str(DESTINATION) and plan['expected']==SPEC
         and plan['data_seed']==plan['fresh_test_seed']==SEED and plan['generator_law']==LAW,'Frozen specification differs')
    verify_protected(plan,frozen)
    for name,h in frozen.items():need(digest(path.parent/'source'/name.replace('/','_'))==h,'Dry source copy changed')
    for field in ('samples','inventory','generator_checks'):
        need(digest(plan[field+'_file'])==plan[field+'_sha256'],'Frozen dry artifact changed')
    need(read(plan['generator_checks_file'])['passed'],'Generated semantic gate failed')
    rows=read(plan['samples_file']);need(canonical_sha(rows)==plan['samples_object_sha256'],'Frozen sample semantics changed')
    return plan,rows,read(plan['inventory_file'])


def audit_published(manifest,excluded):
    from PIL import Image
    need(set(manifest['splits'])=={'test_N32','test_N64'},'Published cells differ')
    reconstructed=[];observed=[];images={};cells={}
    for cell,group in manifest['splits'].items():
        records=group['samples'];need(group['count']==len(records)==136,'Published cell cardinality differs')
        need(group['gold_histogram']==dict(sorted(Counter(r['gold'] for r in records).items())),'Published histogram differs')
        checked_rows=[]
        for row in records:
            path=Path(row['path']);n=row['n_frames']
            need(cell==f'test_N{n}' and path.parent==DESTINATION/'mmred_vfiltered'/f'seq_len_{n}'/'test'
                 and path.name==row['sid'] and not path.is_symlink(),'Published record path differs')
            parsed=read_candidate(path,'test',n)
            for key in ('sid','split','gold','n_frames','content_sha256','qa_sha256'):
                need(parsed[key]==row[key],'Published QA mismatch: '+key)
            checked=inspect(dict(row,source_path=str(path)),'fresh',cell)
            states=source_states(path);frames=[(next(iter(s['rooms'].values()))[0],next(iter(s['rooms']))) for s in states]
            need(checked['target_character']==row['target_character'] and checked['target_room']==row['target_room'],
                 'Independent QA target parse differs')
            for key,value in row['semantic_counts'].items():need(checked[key]==value,'Independent count differs')
            need(len(row['image_files'])==len(frames)==n,'Published image/frame count differs')
            for i,meta in enumerate(row['image_files']):
                image=Path(meta['path']);cache=Path(meta['render_cache']);character,room=frames[i]
                need(image==path/f'{i:03d}.png' and not image.is_symlink() and cache.parent==DESTINATION/'render_cache'
                     and cache.name==f'{character}_{room}_{i+1:03d}.png' and image.samefile(cache),
                     'Image semantic-renderer link differs')
                stat=image.stat();identity=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
                if identity not in images:
                    with Image.open(image) as picture:
                        need(picture.mode==meta['mode']=='RGB' and list(picture.size)==meta['dimensions'],'Image mode/dimensions differ')
                        picture.verify()
                    images[identity]=digest(image)
                need(images[identity]==meta['sha256'] and stat.st_size==meta['bytes'],'Image bytes/hash differ')
            reconstructed.append(dict(row,states=states,frames=frames));checked_rows.append(checked);observed.append(checked)
        cells[cell]=summarize(checked_rows)
    semantic=check_generated(reconstructed,excluded)
    semantic.update(records=observed,cells=cells,unique_image_inodes_hashed=len(images),
        all_qa_hashes_and_gold_recounts_passed=True,all_image_hashes_and_dimensions_passed=True,
        all_image_semantic_links_passed=True,excluded_prior_content=len(excluded))
    return semantic


def stage(args,out,frozen):
    need(not DESTINATION.exists() and not DESTINATION.is_symlink(),'Preserve existing/partial proposed V14 data')
    plan,rows,inventory=load_plan(args.plan,frozen);excluded=set(inventory['excluded_content_sha256'])
    need(canonical_sha(generate(excluded))==plan['samples_object_sha256'],'Regenerated exact dry samples differ')
    check_generated(rows,excluded)
    current,manifests,current_excluded=prior_inputs()
    need(current==inventory and current_excluded==excluded,'Prior28 inventory changed after dry gate')
    DESTINATION.mkdir(parents=True,exist_ok=False)
    (DESTINATION/'stage_plan.json').write_bytes(Path(args.plan).read_bytes())
    (DESTINATION/'exclusion_inventory.json').write_bytes(Path(plan['inventory_file']).read_bytes())
    (out/'plan.json').write_bytes(Path(args.plan).read_bytes())
    reference=manifests[str(V1_ROOT/'main_manifest.json')]['splits']['train_N8']['samples']
    renderer,renderer_provenance=prepare_renderer(reference)
    cache_root=DESTINATION/'render_cache';cache_root.mkdir()
    specs=sorted({(c,r,i+1) for row in rows for i,(c,r) in enumerate(row['frames'])})
    print(json.dumps(dict(samples=272,image_references=13056,unique_renders=len(specs))),flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        cache=dict(pool.map(lambda spec:render_cached(spec,cache_root,renderer),specs))
        records=list(pool.map(lambda row:publish_sample(row,DESTINATION,cache),rows))
    for record,sample in zip(records,rows):record.update(question=sample['question'],origin='generated_v14_fresh',fresh_test_seed=SEED)
    cells={}
    for n in (32,64):
        selected=[r for r in records if r['n_frames']==n]
        cells[f'test_N{n}']=dict(samples=selected,count=len(selected),gold_histogram=dict(sorted(Counter(r['gold'] for r in selected).items())))
    manifest=dict(schema_version=1,purpose='main',dataset_root=str(DESTINATION),data_seed=SEED,fresh_test_seed=SEED,
        splits=cells,generator_law=LAW,source_sha256=frozen,generator_code_sha256=frozen,
        source_manifest_sha256=inventory['source_manifest_sha256'],protected_source_sha256=inventory['protected_source_sha256'],
        renderer_provenance=renderer_provenance,paired_test_unit='anchor_id; complete N32/N64 families',
        standard_downstream_resize=392,training_dev_and_schedule_unchanged=True,training_pairing_unchanged=True,
        local_atom_repetition_allowed=True,prior_sources_never_modified=True,saturation_exceptions=[],
        evaluation_status='Prospective complete-context test data; no efficacy training release is implied',
        stage_plan_file=str(DESTINATION/'stage_plan.json'),stage_plan_sha256=digest(DESTINATION/'stage_plan.json'),
        inventory_file=str(DESTINATION/'exclusion_inventory.json'),inventory_sha256=digest(DESTINATION/'exclusion_inventory.json'),
        prior_inventory_file=str(DESTINATION/'exclusion_inventory.json'),prior_inventory_sha256=digest(DESTINATION/'exclusion_inventory.json'),
        stage_audit_file=str(DESTINATION/'stage_audit.json'))
    audit=audit_published(manifest,excluded)
    # Compare every published context to its frozen dry-plan source, not just totals.
    planned={row['sid']:row for row in rows}
    need(len(planned)==len(records)==272 and all(row['content_sha256']==planned[row['sid']]['content_sha256']
         and row['question']==planned[row['sid']]['question'] and row['gold']==planned[row['sid']]['gold'] for row in records),
         'Published records differ from the exact dry plan')
    verify_protected(plan,frozen)
    audit.update(schema_version=1,source_sha256=frozen,generator_code_sha256=frozen,source_manifest_sha256=inventory['source_manifest_sha256'],
        protected_source_sha256=inventory['protected_source_sha256'],renderer_provenance=renderer_provenance,
        dataset_root=str(DESTINATION),data_seed=SEED,fresh_test_seed=SEED,no_model_calls=True,preparation_only=True,
        dry_plan_file=str(Path(args.plan).resolve()),dry_plan_sha256=digest(args.plan),
        inventory_file=str(DESTINATION/'exclusion_inventory.json'),inventory_sha256=digest(DESTINATION/'exclusion_inventory.json'),
        all_published_records_match_dry_plan=True,prior_sources_never_modified=True,
        training_dev_and_schedule_unchanged=True,training_pairing_unchanged=True)
    exclusive_json(DESTINATION/'semantic_audit.json',audit)
    manifest.update(audit_file=str(DESTINATION/'semantic_audit.json'),audit_sha256=digest(DESTINATION/'semantic_audit.json'))
    exclusive_json(DESTINATION/'main_manifest.json',manifest)
    result=dict(audit,completed=True,dry_check_passed=True,slurm_job_id=os.environ['SLURM_JOB_ID'],
        semantic_audit_file=str(DESTINATION/'semantic_audit.json'),semantic_audit_sha256=digest(DESTINATION/'semantic_audit.json'),
        manifest_paths={'main':str(DESTINATION/'main_manifest.json')},manifest_sha256={'main':digest(DESTINATION/'main_manifest.json')},
        unique_renders=len(specs),cached_png_bytes=sum(meta['bytes'] for meta in cache.values()))
    exclusive_json(DESTINATION/'stage_audit.json',result);exclusive_json(out/'stage_audit.json',result)
    (DESTINATION/'INDEX.md').write_text('# V14 fresh test data\n\n[Manifest](main_manifest.json) · [Semantic audit](semantic_audit.json) · [Completed audit](stage_audit.json) · [Dry plan](stage_plan.json) · [Exclusions](exclusion_inventory.json).\n')
    (out/'REPORT.md').write_text('# V14 fresh test staging\n\nPASS:272 contexts,136 N32→N64 families, eight/K0–16. Independent QA, image, renderer-link, exclusion and extension audits passed. All28 prior manifests and literal V10 training/dev/schedule/pairing are unchanged. No model calls or training release.\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-check',action='store_true');mode.add_argument('--stage',action='store_true')
    parser.add_argument('--plan',type=Path);parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu'
         and not os.environ.get('SLURM_JOB_GPUS'),'All generation/rendering/auditing requires CPU Slurm')
    need(1<=args.workers<=int(os.environ.get('SLURM_CPUS_PER_TASK','1'))<=4,'CPU worker allocation exceeds proposal')
    if args.stage:need(args.plan is not None,'Rendering requires the exact passed dry plan')
    out=OUT/f'{"drycheck" if args.dry_check else "stage"}_{os.environ["SLURM_JOB_ID"]}'
    out.mkdir(parents=True,exist_ok=False);frozen=snapshot(out);start=time.perf_counter()
    result=dry(out,frozen) if args.dry_check else stage(args,out,frozen)
    result.update(elapsed_seconds=time.perf_counter()-start,slurm_job_id=os.environ['SLURM_JOB_ID'])
    exclusive_json(out/'summary.json',result)
    print(json.dumps(dict(passed=result['passed'],directory=str(out),plan_file=result.get('plan_file'),
        plan_sha256=result.get('plan_sha256'),elapsed_seconds=result['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()

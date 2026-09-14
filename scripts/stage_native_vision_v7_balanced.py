"""Balanced V7 question/count corpus; all staging/rendering/audits use Slurm CPU.

1890 distinct training contexts,1944 slots per balanced epoch,72 new N16 dev.
Only saturated N8K8 training contexts may repeat historical complete contexts.
This is a changed training distribution; old V1-V6 datasets remain untouched.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import random
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.stage_native_vision_v2_clean import (CHARS,PARK_ROOMS,LAW,DATA_BASE,V1_ROOT,distractor,describe,
    stable_seed,digest,exclusive_json,prepare_renderer,publish_sample,render_cached)
from scripts.stage_native_vision_v6_test import load_inventory as old_inventory,content_rows,canonical_sha
from scripts.stage_native_vision_pilot import read_candidate,source_states
from scripts.audit_native_vision_data import inspect
from gnnformer.parallel_local_prompts import build_set_count_prompt
ROOT=DATA_BASE/'v7_balanced'
OUT=REPO/'outputs/native_aggregation_vlm/v7/balanced_data'
SEED=20260925
DEV_SEED=20260926
OWN=('scripts/stage_native_vision_v7_balanced.py','slurm/native_vision_v7_balanced_stage.sbatch',
     'scripts/stage_native_vision_v6_test.py','scripts/stage_native_vision_v2_clean.py',
     'scripts/stage_native_vision_pilot.py','scripts/audit_native_vision_data.py',
     'datasets/mmred/render_mmred.py','gnnformer/parallel_local_prompts.py')


def need(value,message):
    if not value:raise ValueError(message)


def prior_inputs():
    inv,manifests,excluded=old_inventory()
    for purpose in ('main','count'):
        p=DATA_BASE/'v6_fresh'/f'{purpose}_manifest.json';m=json.loads(p.read_text());rows=list(content_rows(m))
        values={r['content_sha256'] for r in rows};h=digest(p)
        inv['entries'].append(dict(path=str(p),sha256=h,record_count=len(rows),unique_content_count=len(values)))
        inv['source_manifest_sha256'][str(p)]=h;inv['protected_source_sha256'][str(p)]=h;manifests[str(p)]=m;excluded.update(values)
    inv.update(excluded_content_sha256=sorted(excluded),excluded_content_count=len(excluded),
               excluded_content_set_sha256=canonical_sha(sorted(excluded)),scope='18 historical canonical manifests')
    need(len(inv['entries'])==18,'Expected18 historical exclusion manifests')
    return inv,manifests,excluded


def draw(character,room,n,k,rep,split,used,excluded,seed):
    label=f'v7_balanced_{split}_{character}_{room}_N{n}_K{k}_{rep}'
    local_seed=stable_seed(seed,label);rng=random.Random(local_seed);saturated=split=='train' and n==8 and k==8
    for attempt in range(10000):
        frames=[(character,room)]*k+[distractor(rng,character,room) for _ in range(n-k)];rng.shuffle(frames)
        states,question,h,counts=describe(frames,character,room)
        need(counts['matches']==k and len(states)==n,'Generator count changed')
        if h in used:continue
        if h in excluded and not saturated:continue
        used.add(h)
        return dict(sid=label,n_frames=n,split=split,gold=k,seed=local_seed,axis='balanced_training' if split=='train' else 'balanced_dev',
            anchor_id=None,pair_id=None,test_family=None,anchor_n_frames=None,parent_n_frames=None,parent_positions=[],anchor_positions=[],
            target_character=character,target_room=room,question=question,states=states,frames=frames,content_sha256=h,
            semantic_counts=counts,generation_attempt=attempt,historical_content_reuse=h in excluded,
            saturated_training_exception=saturated)
    raise ValueError('Could not draw fixed-target fresh context: '+label)


def generate(excluded):
    used=set();train=[]
    for c in CHARS:
        for r in PARK_ROOMS:
            for n in (8,16):
                for k in range(9):
                    for rep in range(1 if n==8 and k==8 else 2):train.append(draw(c,r,n,k,rep,'train',used,excluded,SEED))
    dev=[];rng=random.Random(DEV_SEED)
    for k in range(9):
        for rep in range(8):
            c,r=rng.choice(CHARS),rng.choice(PARK_ROOMS)
            dev.append(draw(c,r,16,k,rep,'dev',used,excluded,DEV_SEED))
    need(len(train)==1890 and len(dev)==72 and len(used)==1962,'Balanced corpus size differs')
    by_cell=Counter((x['question'],x['n_frames'],x['gold']) for x in train)
    need(len(by_cell)==54*2*9 and all(count==(1 if n==8 and k==8 else 2) for (_,n,k),count in by_cell.items()),
         'Unique training question/count coverage differs')
    slots=[]
    for row in train:slots.extend([row['sid']]*(2 if row['n_frames']==8 and row['gold']==8 else 1))
    by_sid={x['sid']:x for x in train};weighted=Counter((by_sid[s]['question'],by_sid[s]['n_frames'],by_sid[s]['gold']) for s in slots)
    need(len(slots)==1944 and len(weighted)==972 and set(weighted.values())=={2},'Question/N/count balance differs')
    need(Counter(x['gold'] for x in dev)==Counter({k:8 for k in range(9)}),'Dev count balance differs')
    return train,dev,slots


def audit(manifest,slots,excluded):
    from PIL import Image
    hashes=set();sids=set();inodes={};rows=[];historical=[]
    for cell,selection in manifest['splits'].items():
        for row in selection['samples']:
            p=Path(row['path']);parsed=read_candidate(p,row['split'],row['n_frames'])
            for field in ('sid','split','gold','n_frames','content_sha256','qa_sha256'):need(parsed[field]==row[field],'Published QA mismatch: '+field)
            checked=inspect(dict(row,source_path=str(p)),'main',cell)
            need(checked['target_character']==row['target_character'] and checked['target_room']==row['target_room'],'Parsed target differs')
            for key,value in row['semantic_counts'].items():need(checked[key]==value,'Independent category recount differs')
            states=source_states(p);frames=[(next(iter(x['rooms'].values()))[0],next(iter(x['rooms']))) for x in states]
            _,question,h,counts=describe(frames,row['target_character'],row['target_room'])
            need(question==row['question'] and h==row['content_sha256'] and counts==row['semantic_counts'],'Reconstructed context differs')
            need(row['sid'] not in sids and h not in hashes,'Duplicate current corpus context');sids.add(row['sid']);hashes.add(h)
            exception=row['split']=='train' and row['n_frames']==8 and row['gold']==8
            need(row['saturated_training_exception']==exception,'Saturation exception incorrectly assigned')
            need((h in excluded)==row['historical_content_reuse'],'Historical reuse record differs')
            if h in excluded:
                need(exception,'Prohibited historical context reuse');historical.append(dict(sid=row['sid'],content_sha256=h))
            need(len(row['image_files'])==row['n_frames'],'Image count differs')
            for i,image in enumerate(row['image_files']):
                path=Path(image['path']);need(path==p/f'{i:03d}.png' and not path.is_symlink(),'Wrong image ordering/path')
                stat=path.stat();key=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
                if key not in inodes:
                    with Image.open(path) as picture:
                        need(list(picture.size)==image['dimensions'] and picture.mode==image['mode'],'Rendered image metadata differs');picture.verify()
                    inodes[key]=digest(path)
                need(inodes[key]==image['sha256'] and stat.st_size==image['bytes'],'Rendered image hash differs')
            rows.append(row)
    train=[x for x in rows if x['split']=='train'];dev=[x for x in rows if x['split']=='dev'];lookup={x['sid']:x for x in train}
    weighted=Counter((lookup[s]['question'],lookup[s]['n_frames'],lookup[s]['gold']) for s in slots)
    need(len(train)==1890 and len(dev)==72 and len(weighted)==972 and set(weighted.values())=={2},'Published balance differs')
    need(set(slots)==set(lookup) and len(slots)==1944,'Epoch schedule coverage differs')
    need(set(x['n_frames'] for x in dev)=={16} and Counter(x['gold'] for x in dev)==Counter({k:8 for k in range(9)}),'Published dev differs')
    return dict(passed=True,train_contexts=1890,dev_contexts=72,unique_contexts=1962,epoch_slots=1944,
        question_n_count_cells=972,slots_per_cell=2,saturated_training_contexts=54,historical_saturated_reuse=historical,
        training_question_count_pairs=486,question_count_support_uniform_in_slots=True,all_current_contexts_disjoint=True,
        non_saturated_contexts_fresh_against_18_prior_manifests=True,all_qa_counts_categories_images_verified=True,
        image_references=sum(x['n_frames'] for x in rows),unique_image_inodes=len(inodes),
        local_visual_atoms_may_recur=True,global_prompt_has_no_set_size_argument=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dry-check',action='store_true');a=p.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS',''),
         'Use CPU Slurm wrapper for every staging/audit operation')
    need(not ROOT.exists(),'Preserve existing/partial balanced dataset')
    start=time.perf_counter();job=os.environ['SLURM_JOB_ID'];out=OUT/f'{"drycheck" if a.dry_check else "stage"}_{job}'
    out.mkdir(parents=True,exist_ok=False);(out/'source').mkdir()
    sources={name:digest(REPO/name) for name in OWN}
    for name in OWN:(out/'source'/name.replace('/','_')).write_bytes((REPO/name).read_bytes())
    exclusive_json(out/'source_hashes.json',sources)
    inv,prior,excluded=prior_inputs();train,dev,slots=generate(excluded)
    # Independent deterministic generation and pure prompt-interface checks.
    t2,d2,s2=generate(excluded);need(train==t2 and dev==d2 and slots==s2,'Nondeterministic balanced generation')
    q=train[0]['question'];expected='You will be shown frames describing steps in a house.\nRespond with a single non-negative integer (0 is allowed). Output only the integer.\nQuestion: '+q+'\nAnswer: '
    need(build_set_count_prompt(q)==expected,'Fixed global prompt text differs')
    plan=dict(schema_version=1,purpose='v7_balanced_training_and_dev',data_seed=SEED,dev_seed=DEV_SEED,source_sha256=sources,
        prior_manifest_sha256=inv['source_manifest_sha256'],protected_source_sha256=inv['protected_source_sha256'],
        dataset_root=str(ROOT),expected=dict(train=1890,dev=72,epoch_slots=1944,question_count_pairs=486),
        generator_law=dict(LAW,target='Full factorial9characters×6rooms×K0..8×N8/16',
            saturated_exception='N8K8 one uniquecontext, two presentation slots; historical context reuse explicitly recorded'),
        train_sids=[x['sid'] for x in train],dev_sids=[x['sid'] for x in dev],epoch_slots=slots)
    exclusive_json(out/'plan.json',plan);exclusive_json(out/'prior_inventory.json',inv)
    (out/'INDEX.md').write_text('# V7 balanced corpus staging\n\n[Plan](plan.json) · [Sources](source_hashes.json) · [Prior contexts](prior_inventory.json).\n')
    if a.dry_check:
        print(json.dumps(dict(dry_check_passed=True,train=1890,dev=72,epoch_slots=1944,plan_sha256=digest(out/'plan.json'))),flush=True);return
    ROOT.mkdir();exclusive_json(ROOT/'stage_plan.json',plan);exclusive_json(ROOT/'prior_inventory.json',inv)
    reference=prior[str(V1_ROOT/'main_manifest.json')]['splits']['train_N8']['samples'];renderer,provenance=prepare_renderer(reference)
    rows=train+dev;specs=sorted({(c,r,i+1) for row in rows for i,(c,r) in enumerate(row['frames'])})
    cache_root=ROOT/'render_cache';cache_root.mkdir();print(json.dumps(dict(rendering=len(specs),references=sum(r['n_frames'] for r in rows))),flush=True)
    with ThreadPoolExecutor(max_workers=min(4,int(os.environ['SLURM_CPUS_PER_TASK']))) as pool:
        cache=dict(pool.map(lambda spec:render_cached(spec,cache_root,renderer),specs))
        records=list(pool.map(lambda row:publish_sample(row,ROOT,cache),rows))
    for record,row in zip(records,rows):record.update(question=row['question'],origin='generated_v7_balanced')
    splits={}
    for split,n in (('train',8),('train',16),('dev',16)):
        selected=[r for r in records if r['split']==split and r['n_frames']==n]
        splits[f'{split}_N{n}']=dict(samples=selected,count=len(selected),gold_histogram=dict(sorted(Counter(r['gold'] for r in selected).items())))
    manifest=dict(schema_version=1,purpose='main',dataset_root=str(ROOT),data_seed=SEED,dev_seed=DEV_SEED,splits=splits,
        generator_law=plan['generator_law'],source_sha256=sources,renderer_provenance=provenance,
        stage_plan_file=str(ROOT/'stage_plan.json'),stage_plan_sha256=digest(ROOT/'stage_plan.json'),
        prior_inventory_file=str(ROOT/'prior_inventory.json'),prior_inventory_sha256=digest(ROOT/'prior_inventory.json'))
    result=audit(manifest,slots,excluded)
    need(all(digest(REPO/name)==h for name,h in sources.items()),'Staging source changed')
    need(all(digest(Path(path))==h for path,h in inv['protected_source_sha256'].items()),'Historical data changed')
    result.update(source_sha256=sources,stage_plan_sha256=digest(ROOT/'stage_plan.json'),slurm_job_id=job)
    exclusive_json(ROOT/'semantic_audit.json',result);manifest.update(audit_file=str(ROOT/'semantic_audit.json'),audit_sha256=digest(ROOT/'semantic_audit.json'))
    exclusive_json(ROOT/'main_manifest.json',manifest)
    lookup={r['sid']:r for r in records if r['split']=='train'}
    schedule=dict(schema_version=1,purpose='v7_balanced_epoch',manifest_file=str(ROOT/'main_manifest.json'),manifest_sha256=digest(ROOT/'main_manifest.json'),
        epoch_slots=slots,unique_training_sids=sorted(lookup),slot_metadata=[dict(slot=i,sid=s,n_frames=lookup[s]['n_frames'],gold=lookup[s]['gold'],question=lookup[s]['question']) for i,s in enumerate(slots)])
    exclusive_json(ROOT/'schedule.json',schedule)
    (ROOT/'INDEX.md').write_text('# V7 balanced training and development\n\n[Manifest](main_manifest.json) · [Balanced epoch](schedule.json) · [Semantic audit](semantic_audit.json).\n')
    result.update(elapsed_seconds=time.perf_counter()-start,manifest_file=str(ROOT/'main_manifest.json'),manifest_sha256=digest(ROOT/'main_manifest.json'),schedule_file=str(ROOT/'schedule.json'),schedule_sha256=digest(ROOT/'schedule.json'))
    exclusive_json(out/'summary.json',result)
    (out/'REPORT.md').write_text('# Balanced V7 corpus\n\nPASS:1890trainingcontexts,72freshN16developmentcontexts,1944balancedepochslots. Everyquestion/NhasuniformK0..8presentationweights. HistoricalsaturatedN8K8reusecount: '+str(len(result['historical_saturated_reuse']))+'.\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()

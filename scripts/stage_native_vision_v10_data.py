"""V10 balanced count-support data; CPU Slurm dry freeze before rendering.

Training:54 questions x K0..16 x two weighted scene slots,918 pairs. K0..8
pairs independent N8/N16 scenes; K9..15 pairs independent N16 scenes; K16 uses
one saturated N16 scene twice. Only saturated N8K8 and N16K16 may reuse prior
content. Development:64 fresh N16 scenes, four/K0..15. Test:136 fresh N32
anchors, eight/K0..16, each extended to N64 with IID negative insertions.
No model calls, tokenization, image-law changes, or modifications of older data.
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
from scripts import stage_native_vision_v9_test as previous
from scripts.stage_native_vision_v2_clean import (CHARS,PARK_ROOMS,LAW,DATA_BASE,V1_ROOT,distractor,
    describe,stable_seed,digest as _path_digest,exclusive_json,prepare_renderer,publish_sample,render_cached,make_family)
from scripts.stage_native_vision_pilot import read_candidate,source_states
from scripts.audit_native_vision_data import inspect,summarize

ROOT=DATA_BASE/'v10_balanced'
FRESH=DATA_BASE/'v10_fresh'
OUT=REPO/'outputs/native_aggregation_vlm/v10/data_staging'
TRAIN_SEED=20261005
DEV_SEED=20261006
TEST_SEED=20261007
OWN=tuple(dict.fromkeys(('scripts/stage_native_vision_v10_data.py','slurm/native_vision_v10_data_stage.sbatch',
                        *previous.SOURCE_FILES)))
SPEC=dict(train_unique=1782,train_N8=486,train_N16=1296,epoch_slots=1836,pairs=918,
          dev=64,test=272,test_families=136,unique_current_contexts=2118,image_references=38704)


def need(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())

def digest(path):return _path_digest(Path(path))


def canonical_sha(value):return previous.canonical_sha(value)


def source_hashes():return {name:digest(REPO/name) for name in OWN}


def snapshot(out):
    (out/'source').mkdir();frozen=source_hashes()
    for name,h in frozen.items():
        raw=(REPO/name).read_bytes();need(digest(REPO/name)==h,'Source changed during snapshot')
        (out/'source'/name.replace('/','_')).write_bytes(raw)
    exclusive_json(out/'source_hashes.json',frozen)
    (out/'INDEX.md').write_text('# V10 data staging\n\n[Summary](summary.json) · [Plan](plan.json) · [Sources](source_hashes.json) · [Report](REPORT.md).\n')
    return frozen


def prior_inputs():
    inventory,manifests,excluded=previous.load_inventory()
    need(len(inventory['entries'])==23,'Require the exact V9 prior23 ancestry')
    prior23=dict(inventory['source_manifest_sha256'])
    for purpose in ('main','count'):
        path=DATA_BASE/'v9_fresh'/f'{purpose}_manifest.json';manifest=read(path)
        need(manifest['source_manifest_sha256']==prior23,'V9 fresh ancestry differs from exact prior23')
        rows=list(previous.content_rows(manifest));values={r['content_sha256'] for r in rows};h=digest(path)
        need(str(path) not in manifests and rows,'Duplicate or empty V9 exclusion manifest')
        inventory['entries'].append(dict(path=str(path),sha256=h,record_count=len(rows),unique_content_count=len(values)))
        inventory['source_manifest_sha256'][str(path)]=h;inventory['protected_source_sha256'][str(path)]=h
        manifests[str(path)]=manifest;excluded.update(values)
    need(len(inventory['entries'])==len(manifests)==25,'Require exactly25 prior manifests')
    for name in ('main_manifest.json','schedule.json'):
        path=DATA_BASE/'v7_balanced'/name
        need(inventory['protected_source_sha256'].get(str(path))==digest(path),'V7 balanced source not protected')
    inventory.update(excluded_content_sha256=sorted(excluded),excluded_content_count=len(excluded),
        excluded_content_set_sha256=canonical_sha(sorted(excluded)),
        scope='Exact V9 prior23 plus V9 fresh main/count;25 canonical prior manifests',
        permitted_historical_reuse='Training only: saturated N8/K8 or N16/K16. All other current contexts must be fresh.')
    return inventory,manifests,excluded


def draw(character,room,n,k,replica,split,used,excluded,seed):
    label=f'v10_{split}_{character}_{room}_N{n}_K{k}_r{replica}'
    local_seed=stable_seed(seed,label);rng=random.Random(local_seed)
    exception=split=='train' and (n,k) in ((8,8),(16,16))
    for attempt in range(10000):
        frames=[(character,room)]*k+[distractor(rng,character,room) for _ in range(n-k)]
        rng.shuffle(frames);states,question,h,counts=describe(frames,character,room)
        need(len(frames)==n and counts['matches']==k,'Generator changed N or K')
        if h in used or (h in excluded and not exception):continue
        used.add(h)
        return dict(sid=label,n_frames=n,split=split,gold=k,replica=replica,seed=local_seed,
            axis='v10_training' if split=='train' else 'v10_development',anchor_id=None,pair_id=None,test_family=None,
            anchor_n_frames=None,parent_n_frames=None,parent_positions=[],anchor_positions=[],
            target_character=character,target_room=room,question=question,states=states,frames=frames,
            content_sha256=h,semantic_counts=counts,generation_attempt=attempt,
            historical_content_reuse=h in excluded,saturated_training_exception=exception)
    raise ValueError('Cannot draw a fresh context: '+label)


def generate(excluded):
    used=set();train=[];pairs=[]
    for c in CHARS:
        for r in PARK_ROOMS:
            for k in range(17):
                if k<=8:
                    members=[draw(c,r,n,k,0,'train',used,excluded,TRAIN_SEED) for n in (8,16)]
                    kind='cross_length'
                elif k<=15:
                    members=[draw(c,r,16,k,rep,'train',used,excluded,TRAIN_SEED) for rep in (0,1)]
                    kind='same_length'
                else:
                    one=draw(c,r,16,16,0,'train',used,excluded,TRAIN_SEED);members=[one,one]
                    kind='saturated_identity'
                question=members[0]['question'];pid=canonical_sha(['v10_pair',question,k])
                pair=dict(slot=len(pairs),pair_id=pid,question=question,gold=k,target_character=c,target_room=r,
                    sids=[x['sid'] for x in members],n_frames=[x['n_frames'] for x in members],
                    replicas=[x['replica'] for x in members],pair_kind=kind,same_sid=members[0]['sid']==members[1]['sid'])
                pairs.append(pair)
                for row in members[:1] if kind=='saturated_identity' else members:
                    row.update(pair_id=pid,pair_kind=kind);train.append(row)
    dev=[];rng=random.Random(DEV_SEED)
    for k in range(16):
        for rep in range(4):
            dev.append(draw(rng.choice(CHARS),rng.choice(PARK_ROOMS),16,k,rep,'dev',used,excluded,DEV_SEED))
    occupied=set(excluded)|used;test=[]
    for k in range(17):
        for rep in range(8):
            label=f'v10_test_K{k}_{rep:03d}';seed=stable_seed(TEST_SEED,label)
            family=make_family(random.Random(seed),(32,64),k,label,'test','v10_length',occupied,seed)
            for row in family:row.update(replica=rep,historical_content_reuse=False,saturated_training_exception=False)
            test.extend(family)
    slots=[sid for pair in pairs for sid in pair['sids']]
    return dict(train=train,dev=dev,test=test,pairs=pairs,epoch_slots=slots)


def check_generated(payload,excluded):
    train,dev,test,pairs,slots=[payload[k] for k in ('train','dev','test','pairs','epoch_slots')]
    need(len(train)==1782 and len(dev)==64 and len(test)==272 and len(pairs)==918 and len(slots)==1836,'V10 cardinalities differ')
    rows=train+dev+test;seen=set();sids=set();historical=[]
    for row in rows:
        states,q,h,counts=describe(row['frames'],row['target_character'],row['target_room'])
        need(states==row['states'] and q==row['question'] and h==row['content_sha256']
             and counts==row['semantic_counts'] and counts['matches']==row['gold'] and len(states)==row['n_frames'],
             'Generated semantic/state/question identity differs')
        need(h not in seen and row['sid'] not in sids,'Duplicate current context or SID')
        seen.add(h);sids.add(row['sid'])
        exception=row['split']=='train' and (row['n_frames'],row['gold']) in ((8,8),(16,16))
        need(row['saturated_training_exception']==exception and row['historical_content_reuse']==(h in excluded),'Reuse exception metadata differs')
        need(h not in excluded or exception,'Prohibited historical content reuse')
        if h in excluded:historical.append(dict(sid=row['sid'],n_frames=row['n_frames'],gold=row['gold'],content_sha256=h))
    need(len(seen)==2118 and sum(r['n_frames'] for r in rows)==38704,'Current content/image references differ')
    lookup={r['sid']:r for r in train};weighted=Counter(slots)
    need(set(lookup)==set(slots) and Counter(r['n_frames'] for r in train)==Counter({8:486,16:1296}),'Training N inventory differs')
    questions={describe([],c,r)[1] for c in CHARS for r in PARK_ROOMS}
    need(len(questions)==54 and Counter((lookup[s]['question'],lookup[s]['gold']) for s in slots)==
         Counter({(q,k):2 for q in questions for k in range(17)}),'Question/count presentation weights differ')
    need([p['slot'] for p in pairs]==list(range(918)) and len({p['pair_id'] for p in pairs})==918,'Pair IDs/order differ')
    need(Counter(p['pair_kind'] for p in pairs)==Counter(cross_length=486,same_length=378,saturated_identity=54),'Pair-kind support differs')
    need(slots==[sid for p in pairs for sid in p['sids']],'Flat slots do not preserve the registered pairs')
    for p in pairs:
        members=[lookup[s] for s in p['sids']];k=p['gold']
        need(all(x['question']==p['question'] and x['gold']==k and x['pair_id']==p['pair_id']
             and x['pair_kind']==p['pair_kind'] for x in members),'Pair target identity differs')
        need(p['n_frames']==[x['n_frames'] for x in members] and p['replicas']==[x['replica'] for x in members],
             'Pair length/replica metadata differs')
        expected='cross_length' if k<=8 else 'same_length' if k<=15 else 'saturated_identity'
        need(p['pair_kind']==expected and p['same_sid']==(k==16)
             and (p['sids'][0]==p['sids'][1])==(k==16)
             and p['n_frames']==([8,16] if k<=8 else [16,16]),'Wrong pair kind or same-SID reuse')
    need(all(weighted[s]==(2 if r['gold']==16 else 1) for s,r in lookup.items()),'Only N16K16 may repeat in slots')
    need(Counter((r['n_frames'],r['gold']) for r in dev)==Counter({(16,k):4 for k in range(16)}),'Dev support differs')
    need(Counter((r['n_frames'],r['gold']) for r in test)==Counter({(n,k):8 for n in (32,64) for k in range(17)}),'Test support differs')
    families={}
    for row in test:families.setdefault(row['pair_id'],{})[row['n_frames']]=row
    need(len(families)==136,'Test family count differs')
    for pid,members in families.items():
        need(set(members)=={32,64},'Incomplete test extension')
        a,b=members[32],members[64]
        need(a['anchor_id']==b['anchor_id']==pid and a['gold']==b['gold'] and a['question']==b['question']
             and a['anchor_n_frames']==b['anchor_n_frames']==32,'Test anchor metadata differs')
        positions=b['parent_positions']
        need(a['parent_n_frames'] is None and a['parent_positions']==[] and a['anchor_positions']==list(range(32))
             and b['parent_n_frames']==32 and positions==sorted(set(positions)) and len(positions)==32
             and b['anchor_positions']==positions and [b['frames'][i] for i in positions]==a['frames'],
             'Test semantic extension mapping differs')
        inserted=[frame for i,frame in enumerate(b['frames']) if i not in set(positions)]
        need(describe(inserted,b['target_character'],b['target_room'])[3]['matches']==0,'Extension inserted positive evidence')
    return dict(passed=True,**SPEC,pair_kind_counts=dict(Counter(p['pair_kind'] for p in pairs)),
        historical_saturated_reuse=historical,historical_saturated_reuse_count=len(historical),
        saturated_unique_contexts=108,duplicated_saturated_N16_slots=54,
        all_current_contexts_disjoint=True,all_other_contexts_fresh_against_25_priors=True,
        question_count_support_uniform_in_slots=True,paired_extensions=136,
        all_generated_semantics_and_extensions_passed=True)


def slot_metadata(payload):
    lookup={r['sid']:r for r in payload['train']};result=[]
    for p in payload['pairs']:
        for side,sid in enumerate(p['sids']):
            row=lookup[sid];result.append(dict(slot=len(result),pair_slot=p['slot'],pair_id=p['pair_id'],pair_side=side,
                sid=sid,n_frames=row['n_frames'],gold=row['gold'],question=row['question'],replica=row['replica'],pair_kind=p['pair_kind']))
    return result


def dry(args,out,frozen):
    need(not ROOT.exists() and not FRESH.exists(),'Preserve existing or partial V10 datasets')
    inventory,prior,excluded=prior_inputs();payload=generate(excluded);checked=check_generated(payload,excluded)
    repeated=generate(excluded);need(canonical_sha(payload)==canonical_sha(repeated),'Nondeterministic data generation')
    c,r=CHARS[0],next(iter(PARK_ROOMS));occupied=set();first=draw(c,r,8,0,0,'train',occupied,set(),17)
    second=draw(c,r,8,0,0,'train',occupied,set(),17)
    need(first['content_sha256']!=second['content_sha256'] and second['generation_attempt']>0,'Collision rejection test failed')
    historical_index={}
    for path,manifest in prior.items():
        for row in previous.content_rows(manifest):
            historical_index.setdefault(row['content_sha256'],[]).append(dict(manifest=path,sid=row['sid'],n_frames=row['n_frames']))
    for row in checked['historical_saturated_reuse']:row['prior_occurrences']=historical_index[row['content_sha256']]
    exclusive_json(out/'samples.json',payload);exclusive_json(out/'exclusion_inventory.json',inventory)
    checked.update(deterministic_generation_passed=True,collision_rejection_passed=True)
    exclusive_json(out/'generator_checks.json',checked)
    plan=dict(schema_version=1,protocol='v10_balanced_expanded_count_support',dataset_root=str(ROOT),fresh_root=str(FRESH),
        train_seed=TRAIN_SEED,dev_seed=DEV_SEED,test_seed=TEST_SEED,expected=SPEC,source_sha256=frozen,
        samples_file=str(out/'samples.json'),samples_sha256=digest(out/'samples.json'),samples_object_sha256=canonical_sha(payload),
        inventory_file=str(out/'exclusion_inventory.json'),inventory_sha256=digest(out/'exclusion_inventory.json'),
        generator_checks_file=str(out/'generator_checks.json'),generator_checks_sha256=digest(out/'generator_checks.json'),
        prior_manifest_sha256=inventory['source_manifest_sha256'],protected_source_sha256=inventory['protected_source_sha256'],
        generator_law=dict(LAW,finite_pool_exclusion='25 prior manifests plus all current selected contexts, except the two declared saturated training cases'),
        saturation_exceptions=[dict(n_frames=8,gold=8,unique_per_question=1,slots_per_question=1),
                               dict(n_frames=16,gold=16,unique_per_question=1,slots_per_question=2)],
        test_excludes_all_current_train_and_dev=True,dev_excludes_K16=True,no_N16_test=True,no_unseen_K17_plus=True,
        standard_downstream_resize=392,native_tokenization_deferred=True,no_model_calls=True)
    exclusive_json(out/'plan.json',plan);(out/'plan.sha256').write_text(digest(out/'plan.json')+'\n')
    verify_protected(plan,frozen)
    summary=dict(passed=True,dry_check_passed=True,plan_file=str(out/'plan.json'),plan_sha256=digest(out/'plan.json'),
                 source_sha256=frozen,checks=checked,no_rendering=True,no_model_calls=True)
    exclusive_json(out/'summary.json',summary)
    (out/'REPORT.md').write_text('# V10 dry data plan\n\nPASS: 1,782 unique training scenes, 1,836 slots, 918 pairs, 64 development and 272 test scenes. No images rendered. Only N8/K8 and N16/K16 training contexts may reuse historical content.\n')
    return summary


def verify_protected(plan,frozen):
    need(source_hashes()==frozen==plan['source_sha256'],'Staging source changed')
    need(all(digest(path)==h for path,h in plan['protected_source_sha256'].items()),'Protected historical artifact changed')


def load_plan(path,frozen):
    path=path.resolve();need(path.is_relative_to(OUT),'Require canonical CPU dry-plan output')
    plan=read(path);need(digest(path)==path.with_suffix('.sha256').read_text().strip(),'Dry-plan sidecar differs')
    summary=read(path.parent/'summary.json')
    need(summary['passed'] is True and summary['dry_check_passed'] is True and summary['no_rendering'] is True
         and summary['plan_sha256']==digest(path) and Path(summary['plan_file']).resolve()==path,'Completed matching dry gate required')
    need(plan['protocol']=='v10_balanced_expanded_count_support' and plan['expected']==SPEC
         and plan['train_seed']==TRAIN_SEED and plan['dev_seed']==DEV_SEED and plan['test_seed']==TEST_SEED
         and plan['dataset_root']==str(ROOT) and plan['fresh_root']==str(FRESH),'Frozen V10 specification differs')
    verify_protected(plan,frozen)
    for name,h in frozen.items():need(digest(path.parent/'source'/name.replace('/','_'))==h,'Dry source snapshot differs')
    for key in ('samples','inventory','generator_checks'):need(digest(plan[key+'_file'])==plan[key+'_sha256'],'Frozen dry artifact changed')
    need(read(plan['generator_checks_file'])['passed'] is True,'Generator semantic checks failed')
    payload=read(plan['samples_file']);need(canonical_sha(payload)==plan['samples_object_sha256'],'Frozen sample semantics differ')
    return plan,payload,read(plan['inventory_file'])


def audit_published(manifests,payload,excluded,cache_roots):
    from PIL import Image
    rows=[];images={};observed=[];reconstructed={k:[] for k in ('train','dev','test')}
    for purpose,manifest in manifests.items():
        root=ROOT if purpose=='balanced' else FRESH
        for cell,selection in manifest['splits'].items():
            need(selection['count']==len(selection['samples']),'Published cell count differs')
            need(selection['gold_histogram']==dict(sorted(Counter(r['gold'] for r in selection['samples']).items())),'Published gold histogram differs')
            for row in selection['samples']:
                path=Path(row['path']);n=row['n_frames']
                need(path.parent==root/'mmred_vfiltered'/f'seq_len_{n}'/row['split'] and path.name==row['sid']
                     and not path.is_symlink(),'Published path differs')
                parsed=read_candidate(path,row['split'],n)
                for field in ('sid','split','gold','n_frames','content_sha256','qa_sha256'):need(parsed[field]==row[field],'QA mismatch: '+field)
                checked=inspect(dict(row,source_path=str(path)),purpose,cell)
                need(checked['target_character']==row['target_character'] and checked['target_room']==row['target_room'],'Independent target parse differs')
                states=source_states(path);frames=[(next(iter(s['rooms'].values()))[0],next(iter(s['rooms']))) for s in states]
                _,question,h,counts=describe(frames,row['target_character'],row['target_room'])
                need(question==row['question'] and h==row['content_sha256'] and counts==row['semantic_counts'],'Published question/state semantics differ')
                for key,v in counts.items():need(checked[key]==v,'Independent category count differs')
                need(len(row['image_files'])==n,'Image count differs')
                for i,meta in enumerate(row['image_files']):
                    image=Path(meta['path']);cache=Path(meta['render_cache']);c,r=frames[i]
                    need(image==path/f'{i:03d}.png' and not image.is_symlink()
                         and cache.parent==cache_roots[purpose] and cache.name==f'{c}_{r}_{i+1:03d}.png'
                         and image.samefile(cache),'Image/semantic renderer-cache link differs')
                    stat=image.stat();key=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
                    if key not in images:
                        with Image.open(image) as picture:
                            need(picture.mode==meta['mode']=='RGB' and list(picture.size)==meta['dimensions'],'Image dimensions/mode differ');picture.verify()
                        images[key]=digest(image)
                    need(images[key]==meta['sha256'] and stat.st_size==meta['bytes'],'Image checksum differs')
                reconstructed[row['split']].append(dict(row,states=states,frames=frames))
                rows.append(row);observed.append(checked)
    reconstructed.update(pairs=payload['pairs'],epoch_slots=payload['epoch_slots'])
    checked=check_generated(reconstructed,excluded)
    checked.update(all_qa_hashes_and_gold_recounts_passed=True,all_image_hashes_and_dimensions_passed=True,
        all_image_semantic_links_passed=True,unique_image_inodes_hashed=len(images),records=observed,
        per_cell={key:summarize([r for r in observed if r['group']==key]) for key in sorted({r['group'] for r in observed})})
    return checked


def stage(args,out,frozen):
    need(not ROOT.exists() and not FRESH.exists(),'Preserve existing or partial V10 datasets')
    plan,payload,inventory=load_plan(args.plan,frozen);excluded=set(inventory['excluded_content_sha256'])
    regenerated=generate(excluded);need(canonical_sha(regenerated)==plan['samples_object_sha256'],'Regenerated plan differs')
    check_generated(regenerated,excluded)
    current_inventory,prior,current_excluded=prior_inputs()
    need(current_inventory==inventory and current_excluded==excluded,'Historical inventory changed after dry gate')
    for root in (ROOT,FRESH):
        root.mkdir();(root/'stage_plan.json').write_bytes(args.plan.resolve().read_bytes())
        (root/'exclusion_inventory.json').write_bytes(Path(plan['inventory_file']).read_bytes())
    (out/'plan.json').write_bytes(args.plan.resolve().read_bytes())
    reference=prior[str(V1_ROOT/'main_manifest.json')]['splits']['train_N8']['samples']
    renderer,renderer_provenance=prepare_renderer(reference)
    manifests={};cache_roots={};all_cache={};unique_renders=0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for purpose,root,samples in (('balanced',ROOT,payload['train']+payload['dev']),('fresh',FRESH,payload['test'])):
            specs=sorted({(c,r,i+1) for row in samples for i,(c,r) in enumerate(row['frames'])})
            cache_root=root/'render_cache';cache_root.mkdir();cache_roots[purpose]=cache_root
            print(json.dumps(dict(rendering=purpose,unique_renders=len(specs))),flush=True)
            cache=dict(pool.map(lambda spec:render_cached(spec,cache_root,renderer),specs));all_cache[purpose]=cache;unique_renders+=len(specs)
            records=list(pool.map(lambda row:publish_sample(row,root,cache),samples))
            for record,sample in zip(records,samples):record.update(question=sample['question'],origin='generated_v10_'+purpose)
            cells={}
            for split,n in (('train',8),('train',16),('dev',16)) if purpose=='balanced' else (('test',32),('test',64)):
                selected=[r for r in records if r['split']==split and r['n_frames']==n]
                cells[f'{split}_N{n}']=dict(samples=selected,count=len(selected),gold_histogram=dict(sorted(Counter(r['gold'] for r in selected).items())))
            manifests[purpose]=dict(schema_version=1,purpose='main',dataset_root=str(root),data_seed=TRAIN_SEED if purpose=='balanced' else TEST_SEED,
                train_seed=TRAIN_SEED,dev_seed=DEV_SEED,fresh_test_seed=TEST_SEED,splits=cells,generator_law=plan['generator_law'],
                source_sha256=frozen,generator_code_sha256=frozen,source_manifest_sha256=inventory['source_manifest_sha256'],
                renderer_provenance=renderer_provenance,standard_downstream_resize=392,
                stage_plan_file=str(root/'stage_plan.json'),stage_plan_sha256=digest(root/'stage_plan.json'),
                inventory_file=str(root/'exclusion_inventory.json'),inventory_sha256=digest(root/'exclusion_inventory.json'),
                prior_inventory_file=str(root/'exclusion_inventory.json'),prior_inventory_sha256=digest(root/'exclusion_inventory.json'),
                stage_audit_file=str(root/'stage_audit.json'),saturation_exceptions=plan['saturation_exceptions'],
                local_atom_repetition_allowed=True,paired_test_unit='anchor_id' if purpose=='fresh' else None,
                prior_sources_never_modified=True,test_excludes_all_current_train_and_dev=True)
    audit=audit_published(manifests,payload,excluded,cache_roots)
    # Record exactly where permitted historical complete-context reuse occurred.
    audited_reuse=audit['historical_saturated_reuse'];frozen_reuse=read(plan['generator_checks_file'])['historical_saturated_reuse']
    need(sorted(audited_reuse,key=lambda r:r['sid'])==sorted([{k:v for k,v in r.items() if k!='prior_occurrences'} for r in frozen_reuse],key=lambda r:r['sid']),'Published historical reuse differs')
    audit['historical_saturated_reuse']=frozen_reuse
    audit.update(schema_version=1,source_sha256=frozen,renderer_provenance=renderer_provenance,
        dry_plan_file=str(args.plan.resolve()),dry_plan_sha256=digest(args.plan),source_manifest_sha256=inventory['source_manifest_sha256'],
        no_model_calls=True,prior_sources_never_modified=True,local_atom_repetition_allowed=True)
    verify_protected(plan,frozen)
    for purpose,root in (('balanced',ROOT),('fresh',FRESH)):
        exclusive_json(root/'semantic_audit.json',audit)
        manifests[purpose].update(audit_file=str(root/'semantic_audit.json'),audit_sha256=digest(root/'semantic_audit.json'))
        exclusive_json(root/'main_manifest.json',manifests[purpose])
    lookup={r['sid']:r for r in payload['train']}
    schedule=dict(schema_version=1,purpose='v10_balanced_epoch',manifest_file=str(ROOT/'main_manifest.json'),
        manifest_sha256=digest(ROOT/'main_manifest.json'),epoch_slots=payload['epoch_slots'],unique_training_sids=sorted(lookup),
        slot_metadata=slot_metadata(payload),pair_count=918,scene_slots=1836)
    exclusive_json(ROOT/'schedule.json',schedule)
    pairing=dict(schema_version=1,purpose='v10_equal_question_count_pairs',manifest_file=str(ROOT/'main_manifest.json'),
        manifest_sha256=digest(ROOT/'main_manifest.json'),schedule_file=str(ROOT/'schedule.json'),schedule_sha256=digest(ROOT/'schedule.json'),
        pairs=payload['pairs'],pairs_sha256=canonical_sha(payload['pairs']),pair_count=918,weighted_scene_slots=1836,
        duplicate_pair_rule='Only N16/K16 same-SID pairs, exactly54; both regularizers must be zero on identical inputs')
    exclusive_json(ROOT/'pairing.json',pairing)
    summary=dict(audit,completed=True,dry_check_passed=True,slurm_job_id=os.environ['SLURM_JOB_ID'],
        manifest_paths={p:str(root/'main_manifest.json') for p,root in (('balanced',ROOT),('fresh',FRESH))},
        manifest_sha256={p:digest(root/'main_manifest.json') for p,root in (('balanced',ROOT),('fresh',FRESH))},
        schedule_file=str(ROOT/'schedule.json'),schedule_sha256=digest(ROOT/'schedule.json'),
        pairing_file=str(ROOT/'pairing.json'),pairing_sha256=digest(ROOT/'pairing.json'),unique_renders=unique_renders,
        cached_png_bytes=sum(meta['bytes'] for cache in all_cache.values() for meta in cache.values()))
    for root in (ROOT,FRESH):
        exclusive_json(root/'stage_audit.json',summary)
        (root/'INDEX.md').write_text('# V10 data\n\n[Manifest](main_manifest.json) · [Semantic audit](semantic_audit.json) · [Plan](stage_plan.json) · [Prior exclusions](exclusion_inventory.json).\n')
    (ROOT/'INDEX.md').write_text('# V10 balanced training and development\n\n[Manifest](main_manifest.json) · [Schedule](schedule.json) · [Pairs](pairing.json) · [Semantic audit](semantic_audit.json).\n')
    exclusive_json(out/'stage_audit.json',summary)
    (out/'REPORT.md').write_text('# V10 data staging\n\nPASS: 1,782 unique training scenes, 1,836 slots, 918 pairs; 64 fresh N16 development scenes and 272 fresh N32/N64 test scenes. Independent QA, image, semantic, exclusion and extension audits passed. Historical saturated reuse: '+str(audit['historical_saturated_reuse_count'])+' contexts. No model calls.\n')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-check',action='store_true');mode.add_argument('--stage',action='store_true')
    parser.add_argument('--plan',type=Path);parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),
         'All V10 generation, rendering and auditing must run in CPU Slurm')
    need(1<=args.workers<=int(os.environ.get('SLURM_CPUS_PER_TASK','1')),'Workers exceed CPU allocation')
    out=OUT/f'{"drycheck" if args.dry_check else "stage"}_{os.environ["SLURM_JOB_ID"]}'
    out.mkdir(parents=True,exist_ok=False);frozen=snapshot(out);start=time.perf_counter()
    if args.dry_check:result=dry(args,out,frozen)
    else:
        need(args.plan is not None,'Rendering requires a successful frozen --plan')
        result=stage(args,out,frozen);result['elapsed_seconds']=time.perf_counter()-start
        exclusive_json(out/'summary.json',result)
    print(json.dumps(dict(passed=True,directory=str(out),elapsed_seconds=time.perf_counter()-start,
        plan_file=result.get('plan_file'),plan_sha256=result.get('plan_sha256'))),flush=True)


if __name__=='__main__':main()

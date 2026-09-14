"""Held CPU-only216-scene orientation expansion and exact feature-gap inventory.

Every requested-room image is swapped without receiving an answer label. This
publishes training data only: no native forward, feature harvest, fit or release.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import os
from pathlib import Path
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import stage_native_identity_join_factor_fresh as fresh
from scripts import stage_native_identity_join as symbolic
need=symbolic.need;read=symbolic.read;save=symbolic.save;oid=symbolic.oid
def sha(path):return symbolic.digest(Path(path))
OUT=REPO/'outputs/native_aggregation_vlm/identity_join_factor_orientation'
DATA=Path('/mnt/data/gabriele/gnn_transformer/identity_join_factor_orientation')
FACTOR=REPO/'outputs/native_aggregation_vlm/identity_join_factor_binding/check_443373/plan.json'
FACTOR_SHA='aa9a44067f86ce6340d56419b5efe9411ccd84cb52115f0b31d10745b4e68a07'
LONG=REPO/'outputs/native_aggregation_vlm/identity_join_factor_long/check_443538/plan.json'
LONG_SHA='3442ba95f15bcb833291f71d078832e7e95b7e0137e423696ff6ea535fd018e2'
FRESH=REPO/'outputs/native_aggregation_vlm/identity_join_factor_fresh/data_staging/render_443595/summary.json'
FRESH_SHA='aed7387e9fff96b68e0d04d26a3584ca93c335fc9e264d093b123a56cd206754'
MERGE=REPO/'outputs/native_aggregation_vlm/identity_join_learned/features/merge_443031/summary.json'
CACHE_SHA='db060aed422aa05459d3fde9f512daf44bcab447c1426ff8c5be111764190d46'
PROPOSAL='docs/paper/NATIVE_AGGREGATION_FACTOR_ORIENTATION_PROPOSAL.md'
PROPOSAL_SHA='e7b0bc378ca7115a71bc48f033acef693eb1586871bb4ab585bcf0e028fa128e'
PROTOCOL='identity_join_factor_orientation_training_stage'
POLICY=dict(protocol=PROTOCOL,contexts=216,original_contexts=108,flipped_contexts=108,length_pairs=108,
    base_pairs=54,lengths=[8,16],families=18,orientation_contrast_families=36,same_length_triples=72,image_occurrences=2592,
    target_positions=480,steps=6000,pair_presentations=48000,seed=24,
    transform='swap both requested room labels on every selected-room frame; all other frame content unchanged',
    schedule='exact parent48000 base pairs; first original, alternate orientation on each subsequent visit to each base pair',
    initialization_and_statistics_unchanged=True,strict_prefix_features=True,no_local_labels=True,
    no_model_or_head_calls=True,no_tensor_loads=True,no_feature_harvest=True,no_fit_release=True,
    cpu_seconds=300,cpu_cores=4,memory_gib=16)
OWN=('scripts/stage_native_identity_join_factor_orientation.py','scripts/verify_native_identity_join_factor_orientation.py',
     'slurm/native_identity_join_factor_orientation_stage.sbatch',PROPOSAL)
_READ={}


def bind(path,bindings,expected=None):
    path=Path(path);need(path.is_file() and not path.is_symlink(),'Missing regular bound input: '+str(path));path=path.resolve();h=sha(path)
    need(expected is None or expected==h,'Changed orientation input: '+str(path));need(str(path) not in bindings or bindings[str(path)]==h,'Conflicting input hash')
    bindings[str(path)]=h;_READ[str(path)]=h;return h


def source_maps():
    need(sha(FACTOR)==FACTOR_SHA and sha(REPO/PROPOSAL)==PROPOSAL_SHA,'Held factor/proposal changed')
    parent=read(FACTOR);proof=read(MERGE);need(proof['passed'] is proof['completed'] is True and proof['cache_sha256']==CACHE_SHA,'Original complete cache proof required')
    inherited=dict(parent['source_sha256'])
    for group in (proof['source_sha256'],fresh.source_hashes()):
        for name,h in group.items():need(name not in inherited or inherited[name]==h,'Source union conflict');inherited[name]=h
    for name,h in inherited.items():need(sha(REPO/name)==h,'Inherited source changed: '+name)
    return {name:sha(REPO/name) for name in OWN},inherited


def snapshot(out):
    frozen,inherited=source_maps();(out/'source').mkdir()
    for name,h in frozen.items():
        path=out/'source'/name.replace('/','_');path.write_bytes((REPO/name).read_bytes());need(sha(path)==h,'Source copy changed')
    save(out/'source_hashes.json',frozen);save(out/'inherited_sources.json',dict(source_sha256=inherited));return frozen,inherited


def swapped_states(states,requested_rooms):
    """Per-frame room relabeling: no gold, target, trio or bag answer input."""
    need(len(requested_rooms)==2 and len(set(requested_rooms))==2,'Two distinct requested rooms required')
    swap=dict(zip(requested_rooms,reversed(requested_rooms)));result=[]
    for state in states:
        need(set(state)=={'step_id','rooms'} and len(state['rooms'])==1,'Canonical single-occupant frame required')
        room,people=next(iter(state['rooms'].items()));need(len(people)==1,'One person per frame required')
        result.append(dict(step_id=state['step_id'],rooms={swap.get(room,room):list(people)}))
    return result


def frames(states):
    result=[]
    for i,state in enumerate(states,1):
        need(state['step_id']==i and len(state['rooms'])==1,'Canonical physical Step sequence required')
        room,people=next(iter(state['rooms'].items()));need(room in symbolic.PARK_ROOMS and len(people)==1 and people[0] in symbolic.CHARS,'Unknown frame atom')
        result.append([people[0],room])
    return result


def expanded_rows(originals,bindings):
    result=[];qa_cache={}
    for original in originals:
        value=fresh.qa_content(original,bindings,qa_cache);base={k:copy.deepcopy(v) for k,v in original.items() if k not in ('path','qa_sha256','image_files')}
        states=value['states'];need(value['question']==original['question'] and original['local_prompt']==original['question'],'Exact original question required')
        for version in ('original','flipped'):
            row=copy.deepcopy(base);row.update(base_sid=original['sid'],base_pair_id=original['pair_id'],base_contrast_id=original['contrast_id'],orientation_version=version)
            row['states']=copy.deepcopy(states) if version=='original' else swapped_states(states,original['room_pair']);row['frames']=frames(row['states'])
            if version=='flipped':
                row.update(orientation=1-original['orientation'],pair_id=original['pair_id']+'/flipped',contrast_id=original['contrast_id']+'/flipped')
                row['content_sha256']=oid(dict(question=row['question'],states=row['states']));row['sid']='ij_'+row['content_sha256'][:24]
            need(oid(dict(question=row['question'],states=row['states']))==row['content_sha256'],'Semantic content identity differs');result.append(row)
    need(len(result)==len({r['sid'] for r in result})==216 and sum(len(r['target_ids']) for r in result)==480,'Exact unique216/480 expansion required')
    return result


def pair_inventory(rows,old_pairs):
    lookup={(r['base_sid'],r['orientation_version']):r for r in rows};pairs=[];counterparts=[]
    for pair in old_pairs:
        copies={}
        for version in ('original','flipped'):
            a,b=[lookup[(sid,version)] for sid in pair['sids']];need([a['n_frames'],b['n_frames']]==[8,16] and a['pair_id']==b['pair_id'],'Intact length pair required')
            value=dict(pair_id=a['pair_id'],contrast_id=a['contrast_id'],variant=a['variant'],question=a['question'],gold=a['gold'],
                sids=[a['sid'],b['sid']],n_frames=[8,16],target_ids=a['target_ids'],base_pair_id=pair['pair_id'],orientation_version=version)
            pairs.append(value);copies[version]=value
        counterparts.append(dict(base_pair_id=pair['pair_id'],original_pair_id=copies['original']['pair_id'],flipped_pair_id=copies['flipped']['pair_id'],
            original_sids=copies['original']['sids'],flipped_sids=copies['flipped']['sids']))
    need(len(pairs)==108 and len(counterparts)==54,'Complete counterpart pair inventory required');return pairs,counterparts


def presentation_order(old_order,counterparts,rows):
    lookup={r['base_pair_id']:r for r in counterparts};seen=Counter();result=[];samples={r['sid']:r for r in rows}
    for i,old in enumerate(old_order):
        base=old['pair_id'];version='original' if seen[base]%2==0 else 'flipped';seen[base]+=1;pair=lookup[base]
        result.append(dict(old,base_pair_id=base,base_sids=old['sids'],orientation_version=version,base_pair_visit=seen[base],
            pair_id=pair[version+'_pair_id'],sids=pair[version+'_sids']))
        need([samples[s]['target_ids'] for s in result[-1]['sids']]==[samples[s]['target_ids'] for s in old['sids']],'Orientation changes target order')
    counts=[sum(len(samples[s]['target_ids']) for pair in result[i:i+8] for s in pair['sids']) for i in range(0,len(result),8)]
    need(len(result)==48000 and len(counts)==6000,'Exact fixed presentation horizon required');return result,counts


def feature_inventory(rows,cache,cache_plan):
    features={};image_questions={};scenes={};reused={}
    for row in rows:
        target=row['target_ids'];prefixes=[target[:i] for i in range(len(target))];local=[];pids=[]
        for image in row['image_files']:
            pid=oid([image['sha256'],row['question']]);pids.append(pid)
            image_questions.setdefault(pid,dict(pair_id=pid,image_path=image['path'],image_sha256=image['sha256'],question=row['question']))
            ids=[]
            for prefix in prefixes:
                fid=oid(['local',pid,prefix]);features[fid]=dict(feature_id=fid,kind='local',pair_id=pid,question=row['question'],prefix_ids=prefix,phase='local_prefix' if prefix else 'local_empty');ids.append(fid)
            local.append(ids)
        gids=[]
        for prefix in prefixes:
            fid=oid(['global',row['question'],prefix]);features[fid]=dict(feature_id=fid,kind='global',question=row['question'],prefix_ids=prefix,phase='global_prefix' if prefix else 'global_empty');gids.append(fid)
        scenes[row['sid']]=dict(sid=row['sid'],path=row['path'],qa_sha256=row['qa_sha256'],content_sha256=row['content_sha256'],question=row['question'],
            n_frames=row['n_frames'],split='train',gold=row['gold'],target_ids=target,target_prefixes=prefixes,local_feature_ids=local,
            global_feature_ids=gids,image_question_pair_ids=pids,pair_id=row['pair_id'])
    grouped=defaultdict(list)
    for sid,scene in scenes.items():grouped[scene['pair_id']].append(sid)
    for group in grouped.values():
        need(len(group)==2,'Feature scene pairing differs')
        for sid,other in zip(group,reversed(group)):scenes[sid]['paired_scene_id']=other
    for fid,descriptor in features.items():
        descriptor['layout_id']=oid([descriptor['kind'],descriptor['question'],descriptor['prefix_ids']])
        if fid in cache['features']:
            need(cache_plan['features'][fid]==descriptor,'Existing cache input-key ownership differs');reused[fid]=cache['features'][fid]
    missing=sorted(set(features)-set(reused));need(all(features[f]['kind']=='local' for f in missing),'Unchanged questions/targets cannot require new globals')
    need(len(missing)<=1440,'More than six changed images per original target position')
    return dict(schema_version=1,features=features,image_questions=image_questions,scenes=scenes,reused=reused,missing=missing,
        counts=dict(features=len(features),reused=len(reused),missing=len(missing),missing_by_phase=dict(Counter(features[f]['phase'] for f in missing))),
        exact_image_question_prefix_only=True,no_local_labels=True,no_tensor_loads=True,feature_harvest_released=False)


def self_test():
    states=[dict(step_id=1,rooms={'Kitchen':['Sandra']}),dict(step_id=2,rooms={'Office':['Mary']}),dict(step_id=3,rooms={'Park':['Noah']})]
    swapped=swapped_states(states,['Kitchen','Office']);need(swapped==[dict(step_id=1,rooms={'Office':['Sandra']}),dict(step_id=2,rooms={'Kitchen':['Mary']}),states[2]],'Whole relevant-frame swap differs')
    need(swapped_states(swapped,['Kitchen','Office'])==states,'Room swap must be an involution')
    for gold in ('Sandra','Mary','unrelated'):
        sample=dict(states=states,gold=gold);need(swapped_states(sample['states'],['Kitchen','Office'])==swapped,'Gold entered transformation')
    need(oid(['local',oid(['image','question']),[]])!=oid(['local',oid(['image','question']),[50]]),'Strict prefix missing from feature key')
    return dict(passed=True,groups=4,all_relevant_frames_swapped=True,involution=True,gold_independent=True,strict_prefix_key=True)


def parents(bindings):
    for path,h in ((FACTOR,FACTOR_SHA),(LONG,LONG_SHA),(FRESH,FRESH_SHA)):bind(path,bindings,h)
    factor=read(FACTOR);long=read(LONG)
    for path,value in ((FACTOR,factor),(LONG,long)):
        proof=read(path.parent/'summary.json');bind(path.parent/'summary.json',bindings)
        need(proof['passed'] is proof['completed'] is True and proof['plan_sha256']==sha(path),'Passed original CPU plan required')
        for name,h in value['source_sha256'].items():need(sha(path.parent/'source'/name.replace('/','_'))==h,'Parent source snapshot changed')
    need(long['factor_plan']==dict(file=str(FACTOR),sha256=FACTOR_SHA) and long['initial_sha256']==factor['initial_sha256'],'Unfitted factor ancestry differs')
    for key in ('rows_file','pairs_file','scenes_file','stats_file','stats_inventory_file','initial_file'):
        bind(factor[key],bindings,factor['runtime_bindings'][factor[key]])
    bind(long['order_file'],bindings,long['runtime_bindings'][long['order_file']])
    compact=read(factor['rows_file']);old_pairs=read(factor['pairs_file'])
    uniform_ref=factor['uniform_plan'];bind(uniform_ref['file'],bindings,uniform_ref['sha256']);uniform=read(uniform_ref['file'])
    bind(uniform['rows_file'],bindings,uniform['runtime_bindings'][uniform['rows_file']]);full={r['sample']['sid']:r['sample'] for r in read(uniform['rows_file'])['train']}
    need(len(compact)==len(full)==108 and len(old_pairs)==54 and {r['sid'] for r in compact}==set(full),'Original training support differs')
    originals=[full[row['sid']] for row in compact]
    for brief,row in zip(compact,originals):
        need(all(row[k]==v for k,v in brief.items() if k in row) and brief['first_token_id']==row['target_ids'][0],
             'Compact/full original training ownership differs')
    bind(MERGE,bindings);proof=read(MERGE);bind(proof['cache_file'],bindings,CACHE_SHA);cache=read(proof['cache_file'])
    need(proof['passed'] is proof['completed'] is cache['complete'] is cache['training_only'] is True and proof['cache_sha256']==CACHE_SHA
         and proof['feature_count']==len(cache['features'])==64847 and len(cache['scenes'])==6048,'Complete fixed training-only cache required')
    bind(cache['plan_file'],bindings,cache['plan_sha256']);cp=read(cache['plan_file'])
    need(cache['scenes']==cp['scenes'] and cache['training_pairs']==cp['training_pairs'] and set(cache['features'])==set(cp['features'])
         and cache['native_identity']==cp['native_identity']==factor['native_identity'] and cache['native_identity_sha256']==factor['native_identity_sha256'],
         'Original feature/cache/native identity differs')
    need(len(cache['shards'])==4 and {r['shard'] for r in cache['shards']}==set(range(4)),'Four complete cache shards required')
    for shard in cache['shards']:
        path=Path(shard['directory'])/'summary.json';bind(path,bindings,shard['summary_sha256']);value=read(path)
        need(value['passed'] is value['completed'] is True and value['source_sha256']==cache['source_sha256']
             and value['plan_sha256']==cache['plan_sha256'],'Bound complete feature shard differs')
        bind(value['features_file'],bindings,value['features_sha256']);ids=cp['shards'][shard['shard']]
        need(value['feature_count']==len(ids),'Shard row count differs')
        for i,fid in enumerate(ids):need(cache['features'][fid]==dict(file=value['features_file'],file_sha256=value['features_sha256'],row=i,
            state_sha256=value['feature_state_sha256'][fid]),'Per-feature native pointer identity differs')
    old_scenes=read(factor['scenes_file'])
    for row in originals:
        need(row['sid'] in cache['scenes'] and cache['scenes'][row['sid']]==old_scenes[row['sid']],'Original pilot/native-cache scene identity differs')
    manifest=fresh.verify_stage(FRESH);bind(read(FRESH)['manifest_file'],bindings,read(FRESH)['manifest_sha256'])
    return factor,long,originals,old_pairs,cache,cp,manifest,proof


def exclusion_inventory(manifest,rows,bindings,out):
    bind(manifest['excluded_inventory_file'],bindings,manifest['excluded_inventory_sha256']);old=read(manifest['excluded_inventory_file'])
    need(old['passed'] is old['actual_QA_reconstructed'] is old['exposed_planned_and_derived_contexts_included'] is True and old['canonical_manifest_count']==32,
         'Complete frozen old exposure inventory required')
    for path,h in old['input_bindings'].items():bind(path,bindings,h)
    bind(manifest['samples_file'],bindings,manifest['samples_sha256']);fresh_rows=read(manifest['samples_file']);need(len(fresh_rows)==810,'All prior fresh lengths/panels required')
    prior_scenes=set(old['excluded_scene_sha256']);prior_families=set(old['excluded_family_sha256']);fresh_families=defaultdict(list)
    for row in fresh_rows:
        need(oid(fresh.law.scene_content(row))==row['content_sha256'],'Prior fresh actual symbolic content differs')
        prior_scenes.add(row['content_sha256']);fresh_families[(row['contrast_id'],row['n_frames'])].append(row)
    for values in fresh_families.values():prior_families.add(fresh.law.family_identity(values))
    need(len(fresh_families)==270,'All prior fresh concrete triples required')
    groups=defaultdict(list);collisions=[]
    for row in rows:
        if row['orientation_version']=='original':need(row['content_sha256'] in prior_scenes,'Original training reuse lost prior identity')
        elif row['content_sha256'] in prior_scenes:collisions.append(dict(kind='scene',sid=row['sid'],sha256=row['content_sha256']))
        groups[(row['contrast_id'],row['n_frames'])].append(row)
    for values in groups.values():
        need(len(values)==3 and {r['variant'] for r in values}=={0,1,2},'Complete same-length triple required')
        h=fresh.law.family_identity(values)
        if values[0]['orientation_version']=='flipped' and h in prior_families:collisions.append(dict(kind='family',contrast_id=values[0]['contrast_id'],n_frames=values[0]['n_frames'],sha256=h))
    value=dict(passed=not collisions,old_inventory_file=manifest['excluded_inventory_file'],old_inventory_sha256=manifest['excluded_inventory_sha256'],
        fresh_samples_file=manifest['samples_file'],fresh_samples_sha256=manifest['samples_sha256'],fresh_scene_count=810,fresh_triples=270,
        excluded_scene_sha256=sorted(prior_scenes),excluded_family_sha256=sorted(prior_families),collisions=collisions,
        original_scenes_intentionally_reused=108,flipped_scenes_checked=108,flipped_triples_checked=36,atomic_image_reuse_allowed=True,
        no_collision_rejection_resampling=True)
    save(out/'exclusion_inventory.json',value);need(not collisions,'Fixed opposite orientation collides with prior exposure; no altered counterpart permitted');return value


def render_rows(rows,cp,manifest,root,bindings,out):
    bind(cp['dataset_manifest_file'],bindings,cp['dataset_manifest_sha256']);old=read(cp['dataset_manifest_file'])
    bind(old['render_cache_file'],bindings,old['render_cache_sha256']);atoms=read(old['render_cache_file'])
    need(all(key==symbolic.atom_key(value['atom']) for key,value in atoms.items()),'Canonical atom index differs')
    fresh_plan=read(manifest['plan_file']);bind(fresh_plan['reuse_inventory_file'],bindings,fresh_plan['files'][fresh_plan['reuse_inventory_file']])
    references=read(fresh_plan['reuse_inventory_file'])['renderer_references']
    entries=[]
    for atom in symbolic.needed_atoms(rows):
        key=symbolic.atom_key(atom);need(key in atoms,'Expected N<=16 canonical image atom missing')
        value=atoms[key];bind(value['path'],bindings,value['sha256']);reference={k:value[k] for k in ('path','sha256','dimensions','mode')}
        entries.append(dict(atom=list(atom),key=key,reference=reference))
    for row in references:
        directory=Path(row['source_path']);bind(directory/'qa.txt',bindings)
        for i in (0,row['n_frames']//2,row['n_frames']-1):bind(directory/f'{i:03d}.png',bindings)
    save(out/'render_inventory.json',dict(entries=entries,renderer_references=references,all_atoms_previously_canonical=True))
    renderer,provenance=symbolic.prepare_renderer(references);(root/'render_cache').mkdir();(root/'scenes').mkdir()
    targets=old['tokenizer']['targets']
    need({name:value['target_ids'] for name,value in targets.items()}==cp['target_token_ids'],'Original canonical tokenizer target identity differs')
    for path,h in old['tokenizer']['files'].items():bind(path,bindings,h)
    with ThreadPoolExecutor(max_workers=4) as pool:
        rendered=dict(pool.map(lambda item:symbolic.render_atom(item,root/'render_cache',renderer),entries))
        records=list(pool.map(lambda row:symbolic.publish(row,root,rendered,targets),rows))
    for row in records:row['origin']='canonical_identity_join_factor_orientation'
    from PIL import Image
    for value in rendered.values():
        path=Path(value['path']);need(not path.is_symlink() and sha(path)==value['sha256'] and value['canonical_rgb_parity'] is value['reused'] is True,'Canonical reused PNG differs')
        with Image.open(path) as image:need(image.mode=='RGB' and image.size==(512,512) and hashlib.sha256(image.tobytes()).hexdigest()==value['rgb_sha256'],'Canonical RGB identity differs')
    for row,record in zip(rows,records):
        path=Path(record['path']);qa=path/'qa.txt';need(sha(qa)==record['qa_sha256'],'Published QA changed');lines=qa.read_text().splitlines()
        need(lines[0]=='question:' and lines[-3:]==[row['question'],'answer:',row['gold']] and [ast.literal_eval(s) for s in lines[1:-3]]==row['states'],
             'Published training question/answer/semantic ownership differs')
        need(all(record[k]==v for k,v in row.items() if k not in ('frames','states','origin')),'Published orientation metadata differs')
        for i,((person,room),image) in enumerate(zip(row['frames'],record['image_files'])):
            value=rendered[symbolic.atom_key((person,room,i+1))];expected=path/f'{i:03d}.png'
            need(image==dict(path=str(expected),sha256=value['sha256'],dimensions=[512,512],mode='RGB') and not expected.is_symlink()
                 and os.path.samefile(expected,value['path']),'Published Step/image order differs')
    save(root/'render_cache.json',rendered);return records,rendered,provenance


def stage(out,frozen,inherited):
    bindings={};factor,long,originals,old_pairs,cache,cp,manifest,cache_proof=parents(bindings)
    root=DATA/out.name;need(not root.exists() and root.resolve().is_relative_to(DATA.resolve()),'New immutable authorized data root required');root.mkdir(parents=True)
    save(out/'input_binding_before_stage.json',dict(input_bindings=bindings,original_factor_plan=dict(file=str(FACTOR),sha256=FACTOR_SHA),factor_long_plan=dict(file=str(LONG),sha256=LONG_SHA)))
    rows=expanded_rows(originals,bindings);pairs,counterparts=pair_inventory(rows,old_pairs)
    save(out/'original_rows.json',originals);save(root/'semantic_rows.json',rows);save(out/'pairs.json',pairs);save(out/'counterparts.json',counterparts)
    excluded=exclusion_inventory(manifest,rows,bindings,out)
    records,rendered,renderer=render_rows(rows,cp,manifest,root,bindings,out);save(root/'rows.json',records)
    inventory=feature_inventory(records,cache,cp);save(root/'feature_inventory.json',inventory)
    save(root/'runtime_inputs.json',[fresh.runtime_view(row) for row in records])
    order,counts=presentation_order(read(long['order_file']),counterparts,records)
    need(counts==long['head_rows_by_update'] and sum(counts)==213330 and max(counts)==44,'Mapped order changes original training-head work')
    save(out/'order.json',order);save(out/'head_rows.json',dict(by_update=counts,total=sum(counts),maximum=max(counts),
        exposure_counts={pair['pair_id']:sum(v['pair_id']==pair['pair_id'] for v in order) for pair in pairs}))
    tests=self_test()
    from scripts import verify_native_identity_join_factor_orientation as independent
    tests['independent']=independent.self_test();save(out/'selftests.json',tests)
    files={}
    for path in (out/'original_rows.json',root/'semantic_rows.json',out/'pairs.json',out/'counterparts.json',out/'exclusion_inventory.json',
                 out/'render_inventory.json',root/'rows.json',root/'feature_inventory.json',root/'runtime_inputs.json',root/'render_cache.json',
                 out/'order.json',out/'head_rows.json',out/'selftests.json'):files[str(path)]=sha(path)
    for record in records:
        files[str(Path(record['path'])/'qa.txt')]=record['qa_sha256']
        for image in record['image_files']:files[image['path']]=image['sha256']
    plan=dict(schema_version=1,protocol=PROTOCOL,policy=POLICY,source_sha256=frozen,inherited_source_sha256=inherited,
        original_factor_plan=dict(file=str(FACTOR),sha256=FACTOR_SHA),factor_long_plan=dict(file=str(LONG),sha256=LONG_SHA),
        fresh_stage_report=dict(file=str(FRESH),sha256=FRESH_SHA,manifest_file=read(FRESH)['manifest_file'],manifest_sha256=read(FRESH)['manifest_sha256']),
        cache_merge_report=dict(file=str(MERGE),sha256=sha(MERGE)),cache_file=cache_proof['cache_file'],cache_sha256=CACHE_SHA,
        cache_plan_file=cache['plan_file'],cache_plan_sha256=cache['plan_sha256'],native_identity=cache['native_identity'],native_identity_sha256=cache['native_identity_sha256'],
        original_rows_file=str(out/'original_rows.json'),semantic_rows_file=str(root/'semantic_rows.json'),rows_file=str(root/'rows.json'),
        pairs_file=str(out/'pairs.json'),counterparts_file=str(out/'counterparts.json'),feature_inventory_file=str(root/'feature_inventory.json'),
        runtime_inputs_file=str(root/'runtime_inputs.json'),render_cache_file=str(root/'render_cache.json'),exclusion_inventory_file=str(out/'exclusion_inventory.json'),
        order_file=str(out/'order.json'),head_rows_file=str(out/'head_rows.json'),files=files,input_bindings=bindings,data_root=str(root),
        original_initial_file=factor['initial_file'],original_initial_sha256=factor['initial_sha256'],original_stats_file=factor['stats_file'],
        original_stats_sha256=factor['runtime_bindings'][factor['stats_file']],renderer=renderer,counts=inventory['counts'],
        original_targets_and_batch_weights_unchanged=True,no_model_or_head_calls=True,no_tensor_loads=True,no_feature_harvest=True,no_fit_release=True)
    save(out/'plan.json',plan);audit=independent.audit(plan);save(out/'independent_audit.json',audit);need(audit['passed'] is True,'Independent orientation semantics/cache audit failed')
    symbolic.verify_bindings(bindings)
    return dict(passed=True,completed=True,protocol=PROTOCOL,source_sha256=frozen,inherited_source_sha256=inherited,
        plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),independent_audit_file=str(out/'independent_audit.json'),independent_audit_sha256=sha(out/'independent_audit.json'),
        contexts=216,base_families=18,orientation_contrasts=36,length_pairs=108,feature_counts=inventory['counts'],
        image_occurrences=2592,distinct_atoms=len(rendered),all_atoms_reused=True,training_target_positions=480,training_head_rows=sum(counts),
        original_inputs_reused=108,counterpart_collisions=excluded['collisions'],no_model_or_head_calls=True,no_tensor_loads=True,no_feature_harvest=True,no_fit_release=True)


def verify_plan(path):
    path=Path(path).resolve();plan=read(path);summary=read(path.parent/'summary.json');frozen,inherited=source_maps()
    need(summary['passed'] is summary['completed'] is True and summary['plan_file']==str(path) and summary['plan_sha256']==sha(path)
         and plan['protocol']==PROTOCOL and plan['policy']==POLICY and plan['source_sha256']==frozen and plan['inherited_source_sha256']==inherited,'Passed immutable orientation stage required')
    for name,h in frozen.items():need(sha(path.parent/'source'/name.replace('/','_'))==h,'Stage source snapshot changed')
    symbolic.verify_bindings(plan['files']);symbolic.verify_bindings(plan['input_bindings'])
    need(sha(summary['independent_audit_file'])==summary['independent_audit_sha256'] and read(summary['independent_audit_file'])['passed'] is True,'Independent stage audit changed')
    return plan


def verify_stage(summary_path):
    path=Path(summary_path).resolve();path=path/'summary.json' if path.is_dir() else path
    summary=read(path);need(summary['passed'] is summary['completed'] is True and summary['protocol']==PROTOCOL,'Completed orientation stage required')
    plan=verify_plan(summary['plan_file']);need(summary['plan_sha256']==sha(plan_path:=Path(summary['plan_file']))
        and plan_path.parent==path.parent,'Orientation summary/plan ownership differs')
    return dict(plan=plan,summary=summary,summary_file=str(path),summary_sha256=sha(path),
        rows=read(plan['rows_file']),original_rows=read(plan['original_rows_file']),
        source_sha256=plan['source_sha256'],inherited_source_sha256=plan['inherited_source_sha256'],input_bindings=plan['input_bindings'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS')
         and int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==4,'Only the registered four-core CPU stage may execute')
    out=OUT/f'stage_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();frozen,inherited=snapshot(out)
    save(out/'request.json',dict(protocol=PROTOCOL,source_sha256=frozen,policy=POLICY))
    try:
        result=stage(out,frozen,inherited);elapsed=time.perf_counter()-started
        need(elapsed<=300 and source_maps()==(frozen,inherited),'Orientation staging cap/source changed');result['elapsed_seconds']=elapsed
        (out/'REPORT.md').write_text('# Orientation training inventory\n\nAll216 contexts and exact native feature keys are frozen. Missing features are inventoried only; feature harvest and fitting remain held.\n\n'+str(result['feature_counts'])+'\n')
        save(out/'summary.json',result)
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,elapsed_seconds=time.perf_counter()-started,
            accessed_input_bindings=_READ,partial_outputs_retained=True));raise


if __name__=='__main__':main()

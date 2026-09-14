"""Independent stdlib-only audit of the training orientation expansion.

No tensor/model imports, hidden-state computation or fit. Reconstruct semantic
room swaps and causal cache keys independently before any missing-state harvest.
The parent keeps ownership of execution release and the training protocol.
"""
from pathlib import Path
import ast
from collections import Counter,defaultdict
import hashlib
import json


def need(condition,message):
    if not condition:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())


def oid(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def bind(path,digest):need(sha(path)==digest,'Changed orientation input: '+str(path))


def swap_frames(frames,room_pair):
    need(len(room_pair)==len(set(room_pair))==2,'Two distinct requested rooms required')
    left,right=room_pair
    return [[person,right if room==left else left if room==right else room] for person,room in frames]


def states(frames):return [dict(step_id=i+1,rooms={room:[person]}) for i,(person,room) in enumerate(frames)]


def family_id(rows):
    need(len(rows)==3 and len({r['n_frames'] for r in rows})==1 and {r['variant'] for r in rows}=={0,1,2},'Complete same-length answer-changing triple required')
    values=[dict(states=r['states'],question=r['question']) for r in rows]
    need(len({oid(v) for v in values})==3,'Duplicate semantic scene in a contrast')
    return oid(sorted(values,key=oid))


def semantic_row(row):
    frames=row['frames'];pair=row['room_pair'];trio=row['trio'];relevant=[(i,p,r) for i,(p,r) in enumerate(frames) if r in pair]
    need(row['n_frames'] in (8,16) and len(frames)==row['n_frames'] and row['states']==states(frames),'Canonical Step/state/frame sequence differs')
    need(len(pair)==len(set(pair))==2 and len(trio)==len(set(trio))==3 and row['variant'] in (0,1,2) and type(row['orientation']) is int
         and row['orientation'] in (0,1) and row['k']==len(relevant)==6,'Requested-room or marginal inventory differs')
    need(Counter(p for _,p,_ in relevant)==Counter({p:2 for p in trio}) and Counter(r for _,_,r in relevant)==Counter({r:3 for r in pair}),
         'The room swap must preserve both person and room marginals')
    winners=[person for person in trio if {room for _,p,room in relevant if p==person}==set(pair)]
    need(winners==[row['gold']] and row['gold']==trio[row['variant']],'Unique join identity or changing answer differs')
    for index,person in enumerate(p for p in trio if p!=row['gold']):
        need({r for _,p,r in relevant if p==person}=={pair[index ^ row['orientation']]},'Actual nonanswer room assignment disagrees with orientation')
    question=f"Consider only images in the {pair[0]} or the {pair[1]}. Which person appears in both rooms? Reply with the person's name only."
    need(row['split']=='train' and row['question']==row['local_prompt']==question,'Unaltered local/global question required')
    content=oid(dict(states=row['states'],question=row['question']))
    need(content==row['content_sha256'] and row['sid']=='ij_'+content[:24],'Semantic content/SID identity differs')
    return relevant


def feature_keys(rows):
    features={};pairs={};scenes={}
    for row in rows:
        q=row['question'];target=row['target_ids'];need(target[-1]==151645 and len(target) in (2,3) and 151645 not in target[:-1],'Strict native name/EOS target differs')
        prefixes=[target[:j] for j in range(len(target))];local=[];pids=[]
        for image in row['image_files']:
            pid=oid([image['sha256'],q]);pids.append(pid)
            pair=dict(image_path=image['path'],image_sha256=image['sha256'],question=q,pair_id=pid)
            if pid in pairs:need(all(pair[k]==pairs[pid][k] for k in ('image_sha256','question','pair_id')),'Image/question causal identity collision')
            else:pairs[pid]=pair
            ids=[]
            for prefix in prefixes:
                fid=oid(['local',pid,prefix]);descriptor=dict(feature_id=fid,layout_id=oid(['local',q,prefix]),kind='local',question=q,prefix_ids=prefix,phase='local_prefix' if prefix else 'local_empty',pair_id=pid)
                need(fid not in features or features[fid]==descriptor,'Local strict-prefix key conflict');features[fid]=descriptor;ids.append(fid)
            local.append(ids)
        global_ids=[]
        for prefix in prefixes:
            fid=oid(['global',q,prefix]);descriptor=dict(feature_id=fid,layout_id=oid(['global',q,prefix]),kind='global',question=q,prefix_ids=prefix,phase='global_prefix' if prefix else 'global_empty')
            need(fid not in features or features[fid]==descriptor,'Global strict-prefix key conflict');features[fid]=descriptor;global_ids.append(fid)
        scenes[row['sid']]=dict(sid=row['sid'],path=row['path'],qa_sha256=row['qa_sha256'],content_sha256=row['content_sha256'],question=q,n_frames=row['n_frames'],
            split='train',gold=row['gold'],target_ids=target,target_prefixes=prefixes,local_feature_ids=local,global_feature_ids=global_ids,image_question_pair_ids=pids,pair_id=row['pair_id'])
    grouped=defaultdict(list)
    for scene in scenes.values():grouped[scene['pair_id']].append(scene)
    for group in grouped.values():
        need(len(group)==2 and {s['n_frames'] for s in group}=={8,16},'Each native feature scene needs its same-variant N8/N16 partner')
        for left,right in (group,group[::-1]):scenes[left['sid']]['paired_scene_id']=right['sid']
    return features,pairs,scenes


def self_test():
    frames=[['A','R0'],['A','R1'],['B','R0'],['B','R0'],['C','R1'],['C','R1'],['B','outside'],['A','outside']]
    flipped=swap_frames(frames,['R0','R1'])
    need(swap_frames(flipped,['R0','R1'])==frames and flipped[6:]==frames[6:] and [p for p,_ in flipped]==[p for p,_ in frames],
         'All relevant frames, involution and unchanged backgrounds fixture failed')
    need(Counter(r for _,r in flipped)==Counter(r for _,r in frames),'Room marginals changed')
    winners=lambda values:[p for p in ('A','B','C') if {r for person,r in values if person==p and r in ('R0','R1')}=={'R0','R1'}]
    need(winners(frames)==winners(flipped)==['A'] and all(a!=b for a,b in zip(frames[:6],flipped[:6])),'Swap must preserve the answer while changing every relevant room')
    image='a'*64;q='unchanged question';prefix=[50];pid=oid([image,q]);fid=oid(['local',pid,prefix])
    need(fid!=oid(['local',oid(['b'*64,q]),prefix]) and fid!=oid(['local',oid([image,q+'!']),prefix])
         and fid!=oid(['local',pid,[50,23274]]) and fid!=oid(['global',q,prefix]),'Image/question/prefix/kind all belong to causal cache identity')
    need([([50,23274,151645])[:j] for j in range(3)]==[[],[50],[50,23274]],'Future target/EOS entered a cached prefix')
    return dict(passed=True,groups=4,whole_room_swap_and_backgrounds=True,preserved_join_and_marginals=True,exact_causal_cache_keys=True,strict_prefixes=True)


def qa_record(row):
    path=Path(row['path'])/'qa.txt';bind(path,row['qa_sha256']);lines=path.read_text().splitlines()
    need(len(lines)==row['n_frames']+4 and lines[0]=='question:' and lines[-2]=='answer:' and lines[-3]==row['question'] and lines[-1]==row['gold'],
         'Exact native scene QA/question/answer ownership differs')
    value=[ast.literal_eval(line) for line in lines[1:-3]]
    need(oid(dict(states=value,question=row['question']))==row['content_sha256'],'Published QA content identity differs')
    return value


def audit_semantics(originals,published,semantic,counterparts,parent_pairs):
    need(len(originals)==len({r['sid'] for r in originals})==108 and len(published)==len(semantic)==216
         and len({r['sid'] for r in published})==len({r['sid'] for r in semantic})==216,'Complete unique108→216 semantic coverage required')
    by_original={r['sid']:r for r in originals};by_sid={r['sid']:r for r in semantic};seen=[];groups=defaultdict(list)
    for index,original in enumerate(originals):
        old_states=qa_record(original);old_frames=[]
        for i,state in enumerate(old_states,1):
            need(set(state)=={'step_id','rooms'} and state['step_id']==i and len(state['rooms'])==1,'Original canonical Step/frame structure differs')
            room,people=next(iter(state['rooms'].items()));need(len(people)==1,'One native image occupant required');old_frames.append([people[0],room])
        for offset,version in enumerate(('original','flipped')):
            row=semantic[2*index+offset];output=published[2*index+offset];need(row['base_sid']==original['sid'] and row['orientation_version']==version,'Exact original/flipped interleaving required')
            changed=version=='flipped';expected_frames=swap_frames(old_frames,original['room_pair']) if changed else old_frames
            need(row['frames']==expected_frames and row['states']==states(expected_frames) and row['base_pair_id']==original['pair_id']
                 and row['base_contrast_id']==original['contrast_id'] and row['orientation']==(1-original['orientation'] if changed else original['orientation'])
                 and row['pair_id']==original['pair_id']+('/flipped' if changed else '') and row['contrast_id']==original['contrast_id']+('/flipped' if changed else ''),
                 'Room transformation or counterpart IDs differ')
            changed_keys={'sid','content_sha256','pair_id','contrast_id','orientation','path','qa_sha256','image_files'}
            need(all(row[k]==v for k,v in original.items() if k not in changed_keys),'Transformation changed another original semantic field')
            if not changed:need(row['sid']==original['sid'] and row['content_sha256']==original['content_sha256'],'Original identities must remain exact')
            relevant=semantic_row(row);need(len(relevant)==6 and [p for p,_ in row['frames']]==[p for p,_ in old_frames],'People/occurrence positions changed')
            need(all(output[k]==v for k,v in row.items() if k not in ('frames','states','origin')) and qa_record(output)==row['states'],
                 'Published row metadata/QA differs from the fixed semantic record')
            need(len(output['image_files'])==row['n_frames'] and all(set(image)=={'path','sha256','dimensions','mode'} and image['dimensions']==[512,512]
                 and image['mode']=='RGB' for image in output['image_files']),'Canonical image descriptor ownership differs')
            if not changed:need([i['sha256'] for i in output['image_files']]==[i['sha256'] for i in original['image_files']],'Original pixel sequence changed')
            else:
                for i,(_,room) in enumerate(old_frames):
                    need((output['image_files'][i]['sha256']!=original['image_files'][i]['sha256']) if room in original['room_pair'] else
                         (output['image_files'][i]['sha256']==original['image_files'][i]['sha256']),'Requested-room images must change; background bytes must remain exact')
            groups[row['contrast_id']].append(row);seen.append((version,row['n_frames'],row['gold'],row['question']))
    need(len(groups)==36 and Counter(r['orientation_version'] for r in semantic)=={'original':108,'flipped':108}
         and sum(r['n_frames'] for r in semantic)==2592 and sum(len(r['target_ids']) for r in semantic)==480
         and sum(r['n_frames']*len(r['target_ids']) for r in semantic)==5760,'Orientation scene/image/target-position coverage differs')
    family_hashes=[]
    for group in groups.values():
        need(len(group)==6 and {(r['variant'],r['n_frames']) for r in group}=={(v,n) for v in range(3) for n in (8,16)},'All three answers and both lengths required in each orientation')
        for n in (8,16):
            triple=sorted((r for r in group if r['n_frames']==n),key=lambda r:r['variant']);first=triple[0]
            need({r['gold'] for r in triple}==set(first['trio']) and all(r['room_pair']==first['room_pair'] and r['trio']==first['trio']
                 and r['orientation']==first['orientation'] and r['parent_positions']==first['parent_positions'] for r in triple),'Complete matched-marginal answer-changing triple differs')
            relevant=[i for i,(_,room) in enumerate(first['frames']) if room in first['room_pair']]
            for row in triple:
                need([p for p,_ in row['frames']]==[p for p,_ in first['frames']] and [f for i,f in enumerate(row['frames']) if i not in relevant]
                     ==[f for i,f in enumerate(first['frames']) if i not in relevant],'Contrast person sequence/background changed')
            family_hashes.append(dict(contrast_id=first['contrast_id'],orientation_version=first['orientation_version'],n_frames=n,family_instance_sha256=family_id(triple)))
        for v in range(3):
            low=next(r for r in group if (r['variant'],r['n_frames'])==(v,8));high=next(r for r in group if (r['variant'],r['n_frames'])==(v,16));positions=high['parent_positions']
            need(low['parent_n_frames'] is None and low['parent_positions']==[] and high['parent_n_frames']==8 and len(positions)==8
                 and positions==sorted(set(positions)) and all(type(i) is int and 0<=i<16 for i in positions)
                 and [high['frames'][i] for i in positions]==low['frames']
                 and all(room not in high['room_pair'] for i,(_,room) in enumerate(high['frames']) if i not in positions),'N8→N16 unchanged background insertion provenance differs')
    need(len(counterparts)==len(parent_pairs)==54,'Every original pair needs exactly one flipped counterpart')
    expected_pairs=[]
    for parent,mapping in zip(parent_pairs,counterparts):
        a,b=[by_original[s] for s in parent['sids']];original_ids=[a['sid'],b['sid']]
        flipped_ids=[next(r['sid'] for r in semantic if r['base_sid']==s and r['orientation_version']=='flipped') for s in original_ids]
        need(mapping==dict(base_pair_id=parent['pair_id'],original_pair_id=parent['pair_id'],flipped_pair_id=parent['pair_id']+'/flipped',
            original_sids=original_ids,flipped_sids=flipped_ids),'Exact paired counterpart descriptor differs')
        for version,sids in (('original',original_ids),('flipped',flipped_ids)):
            row=by_sid[sids[0]];expected_pairs.append(dict(pair_id=row['pair_id'],contrast_id=row['contrast_id'],variant=row['variant'],question=row['question'],gold=row['gold'],
                sids=sids,n_frames=[8,16],target_ids=row['target_ids'],base_pair_id=parent['pair_id'],orientation_version=version))
    names={r['gold'] for r in semantic};questions={r['question'] for r in semantic}
    need(len(names)==9 and len(questions)==6 and Counter((v,n,g) for v,n,g,q in seen)==Counter({(v,n,g):6 for v in ('original','flipped') for n in (8,16) for g in names})
         and Counter((v,n,q) for v,n,g,q in seen)==Counter({(v,n,q):9 for v in ('original','flipped') for n in (8,16) for q in questions}),
         'Every name/question/orientation/length must retain its fixed balanced count')
    return expected_pairs,family_hashes


def audit(plan):
    for path,digest in plan['files'].items():bind(path,digest)
    for key in ('original_factor_plan','factor_long_plan','fresh_stage_report','cache_merge_report'):
        bind(plan[key]['file'],plan[key]['sha256'])
    parent=read(plan['original_factor_plan']['file']);long=read(plan['factor_long_plan']['file'])
    need(long['factor_plan']==plan['original_factor_plan'] and long['initial_sha256']==parent['initial_sha256'],'Locked original6000 factor ancestry differs')
    for key in ('rows_file','pairs_file','scenes_file'):
        bind(parent[key],parent['runtime_bindings'][parent[key]])
    bind(long['order_file'],long['runtime_bindings'][long['order_file']])
    need(plan['original_initial_file']==parent['initial_file'] and plan['original_initial_sha256']==parent['initial_sha256']
         and plan['original_stats_file']==parent['stats_file'] and plan['original_stats_sha256']==parent['runtime_bindings'][parent['stats_file']],
         'Orientation expansion may not change unfitted weights or original global statistics')
    bind(plan['original_initial_file'],plan['original_initial_sha256']);bind(plan['original_stats_file'],plan['original_stats_sha256'])
    originals=read(plan['original_rows_file']);compact=read(parent['rows_file']);old_pairs=read(parent['pairs_file'])
    uniform_ref=parent['uniform_plan'];bind(uniform_ref['file'],uniform_ref['sha256']);uniform=read(uniform_ref['file'])
    bind(uniform['rows_file'],uniform['runtime_bindings'][uniform['rows_file']]);full={r['sample']['sid']:r['sample'] for r in read(uniform['rows_file'])['train']}
    need(len(full)==108 and originals==[full[row['sid']] for row in compact],'Original full sample records must exactly match the frozen pilot support')
    common=('sid','contrast_id','pair_id','variant','n_frames','question','gold','target_ids')
    need(len(originals)==len(compact)==108 and all(all(full[k]==small[k] for k in common) and small['first_token_id']==small['target_ids'][0]
         for full,small in zip(originals,compact)),'Exact original compact108 ordering/targets required')
    published=read(plan['rows_file']);semantic=read(plan['semantic_rows_file']);counterparts=read(plan['counterparts_file'])
    expected_pairs,family_rows=audit_semantics(originals,published,semantic,counterparts,old_pairs)
    need(read(plan['pairs_file'])==expected_pairs,'Published108 N8/N16 pair table differs')
    runtime=[dict(sid=r['sid'],n_frames=r['n_frames'],question=r['question'],image_files=r['image_files']) for r in published]
    need(read(plan['runtime_inputs_file'])==runtime and all(set(r)=={'sid','n_frames','question','image_files'} for r in runtime),'Offline labels/answer metadata entered runtime inputs')
    render=read(plan['render_cache_file']);atoms={(p,room,i+1) for row in semantic for i,(p,room) in enumerate(row['frames'])}
    need(set(render)=={'/'.join(map(str,a)) for a in atoms},'Exact canonical image-atom inventory differs')
    for key,value in render.items():
        need(key=='/'.join(map(str,value['atom'])) and value['canonical_rgb_parity'] is value['reused'] is True
             and value['dimensions']==[512,512] and value['mode']=='RGB','Canonical renderer/atom provenance differs');bind(value['path'],value['sha256'])
    for row,output in zip(semantic,published):
        directory=Path(output['path'])
        need(directory==Path(plan['data_root'])/'scenes'/row['sid'] and not directory.is_symlink(),'Published scene escaped the new stage')
        for i,((person,room),image) in enumerate(zip(row['frames'],output['image_files'])):
            atom=render[f'{person}/{room}/{i+1}'];path=directory/f'{i:03d}.png'
            need(image==dict(path=str(path),sha256=atom['sha256'],dimensions=[512,512],mode='RGB') and not path.is_symlink()
                 and path.samefile(atom['path']),'Exact physical Step/canonical pixel sequence differs')
    bind(plan['cache_file'],plan['cache_sha256']);bind(plan['cache_plan_file'],plan['cache_plan_sha256']);cache=read(plan['cache_file']);cp=read(plan['cache_plan_file'])
    proof=read(plan['cache_merge_report']['file'])
    need(proof['passed'] is proof['completed'] is cache['complete'] is cache['training_only'] is True and proof['cache_file']==plan['cache_file']
         and proof['cache_sha256']==plan['cache_sha256'] and cache['plan_file']==plan['cache_plan_file'] and cache['plan_sha256']==plan['cache_plan_sha256']
         and len(cache['features'])==64847 and len(cache['scenes'])==6048 and len(cache['training_pairs'])==3024,'Original full training-only cache proof differs')
    need(cache['native_identity']==cp['native_identity']==plan['native_identity']==parent['native_identity'] and cache['native_identity_sha256']==plan['native_identity_sha256']
         ==parent['native_identity_sha256']==oid(plan['native_identity']) and cache['scenes']==cp['scenes'] and cache['training_pairs']==cp['training_pairs']
         and set(cache['features'])==set(cp['features']),'Exact native cache model/scene/key inventory differs')
    parent_scenes=read(parent['scenes_file'])
    for row in originals:need(cache['scenes'][row['sid']]==parent_scenes[row['sid']],'Original scene-feature ownership differs')
    features,image_questions,scenes=feature_keys(published);inventory=read(plan['feature_inventory_file'])
    need(inventory['features']==features and inventory['image_questions']==image_questions and inventory['scenes']==scenes,'Independent image/question/strict-prefix feature reconstruction differs')
    reused={fid:cache['features'][fid] for fid in features if fid in cache['features']};missing=sorted(set(features)-set(cache['features']))
    need(inventory['reused']==reused and inventory['missing']==missing and set(reused).isdisjoint(missing) and set(reused)|set(missing)==set(features)
         and all(features[fid]['kind']=='local' for fid in missing) and len(missing)<=1440,'Exact native feature reuse/missing partition differs')
    files={}
    for fid,pointer in reused.items():
        need(cp['features'][fid]==features[fid] and set(pointer)=={'file','file_sha256','row','state_sha256'} and type(pointer['row']) is int and pointer['row']>=0,
             'Reused pointer no longer owns the exact strict native input')
        if features[fid]['kind']=='local':
            pid=features[fid]['pair_id'];need(cp['pairs'][pid]['image_sha256']==image_questions[pid]['image_sha256'] and cp['pairs'][pid]['question']==features[fid]['question'],
                 'Image-byte/question alias does not own the reused feature')
        need(pointer['file'] not in files or files[pointer['file']]==pointer['file_sha256'],'Conflicting native shard SHA');files[pointer['file']]=pointer['file_sha256']
    for path,digest in files.items():bind(path,digest)
    counts=dict(features=len(features),reused=len(reused),missing=len(missing),missing_by_phase=dict(Counter(features[f]['phase'] for f in missing)))
    need(inventory['counts']==plan['counts']==counts and inventory['exact_image_question_prefix_only'] is inventory['no_local_labels'] is inventory['no_tensor_loads'] is True
         and inventory['feature_harvest_released'] is False,'Feature-gap reporting/no-harvest contract differs')
    expected_order=[];visits=Counter();mapping={r['base_pair_id']:r for r in counterparts};old_order=read(long['order_file'])
    for old in old_order:
        base=old['pair_id'];version='original' if visits[base]%2==0 else 'flipped';visits[base]+=1;pair=mapping[base]
        expected_order.append(dict(old,base_pair_id=base,base_sids=old['sids'],orientation_version=version,base_pair_visit=visits[base],
            pair_id=pair[version+'_pair_id'],sids=pair[version+'_sids']))
    need(len(old_order)==len(expected_order)==48000 and len(visits)==54 and read(plan['order_file'])==expected_order,'Exact old slots with per-pair alternating visits required')
    head_rows=[sum(len(scenes[sid]['target_ids']) for pair in expected_order[i:i+8] for sid in pair['sids']) for i in range(0,48000,8)]
    need(head_rows==long['head_rows_by_update'] and len(head_rows)==6000 and sum(head_rows)==213330 and max(head_rows)==44,'Mapped orientation changed native CE targets or batch weights')
    old_scenes=cache['scenes']
    for old,new in zip(old_order,expected_order):need([scenes[s]['target_ids'] for s in new['sids']]==[old_scenes[s]['target_ids'] for s in old['sids']],
        'Orientation presentation changed target lengths/labels inside a batch')
    exposures=Counter(row['pair_id'] for row in expected_order)
    need(read(plan['head_rows_file'])==dict(by_update=head_rows,total=213330,maximum=44,exposure_counts={p['pair_id']:exposures[p['pair_id']] for p in expected_pairs})
         and all(exposures[c['original_pair_id']]-exposures[c['flipped_pair_id']] in (0,1) for c in counterparts),'Declared visit imbalance/head-row ledger differs')
    excluded=read(plan['exclusion_inventory_file']);fresh_ref=plan['fresh_stage_report'];fresh_proof=read(fresh_ref['file'])
    need(fresh_proof['passed'] is fresh_proof['completed'] is True and fresh_proof['manifest_file']==fresh_ref['manifest_file']
         and fresh_proof['manifest_sha256']==fresh_ref['manifest_sha256'],'Exact passed prior fresh render proof required')
    bind(fresh_ref['manifest_file'],fresh_ref['manifest_sha256']);fresh_manifest=read(fresh_ref['manifest_file'])
    need(excluded['old_inventory_file']==fresh_manifest['excluded_inventory_file'] and excluded['old_inventory_sha256']==fresh_manifest['excluded_inventory_sha256']
         and excluded['fresh_samples_file']==fresh_manifest['samples_file'] and excluded['fresh_samples_sha256']==fresh_manifest['samples_sha256'],'Prior exclusion inputs differ from the frozen full810 manifest')
    bind(excluded['old_inventory_file'],excluded['old_inventory_sha256']);bind(excluded['fresh_samples_file'],excluded['fresh_samples_sha256'])
    old=read(excluded['old_inventory_file']);fresh=read(excluded['fresh_samples_file'])
    need(old['passed'] is old['actual_QA_reconstructed'] is old['exposed_planned_and_derived_contexts_included'] is True and len(fresh)==810,'Frozen prior plus failed810 exposure proof required')
    prior_scenes=set(old['excluded_scene_sha256']);prior_families=set(old['excluded_family_sha256']);fresh_groups=defaultdict(list)
    for row in fresh:
        need(oid(dict(states=row['states'],question=row['question']))==row['content_sha256'],'Prior fresh content changed')
        prior_scenes.add(row['content_sha256']);fresh_groups[(row['contrast_id'],row['n_frames'])].append(row)
    need(len(fresh_groups)==270,'All prior fresh concrete triples required')
    for values in fresh_groups.values():prior_families.add(family_id(values))
    need(excluded['excluded_scene_sha256']==sorted(prior_scenes) and excluded['excluded_family_sha256']==sorted(prior_families)
         and all((row['content_sha256'] in prior_scenes) if row['orientation_version']=='original' else (row['content_sha256'] not in prior_scenes) for row in semantic)
         and all(row['family_instance_sha256'] not in prior_families for row in family_rows if row['orientation_version']=='flipped'),
         'Fixed counterpart collides with a previously registered complete scene/triple')
    need(excluded['passed'] is excluded['atomic_image_reuse_allowed'] is excluded['no_collision_rejection_resampling'] is True and excluded['collisions']==[]
         and excluded['original_scenes_intentionally_reused']==excluded['flipped_scenes_checked']==108 and excluded['flipped_triples_checked']==36,
         'Counterpart freshness and allowed original overlap differ')
    need(plan['original_targets_and_batch_weights_unchanged'] is plan['no_model_or_head_calls'] is plan['no_tensor_loads'] is plan['no_feature_harvest'] is plan['no_fit_release'] is True,
         'CPU training-only inventory cannot release a model call or fit')
    return dict(passed=True,contexts=216,base_families=18,orientation_contrasts=36,same_length_triples=72,length_pairs=108,
        image_occurrences=2592,target_positions=480,local_target_occurrences=5760,feature_counts=counts,exact_cache_reuse=True,strict_prefixes=True,
        fixed_original_training_order=True,training_head_rows=213330,maximum_batch_head_rows=44,base_pair_exposures=[dict(base_pair_id=c['base_pair_id'],
            original=exposures[c['original_pair_id']],flipped=exposures[c['flipped_pair_id']]) for c in counterparts],
        complete_family_identities=family_rows,opposite_orientation_no_prior_scene_or_triple_collision=True,original_scene_reuse_explicit=True,
        no_tensor_or_model_import=True,no_native_forward=True,no_feature_harvest_or_fit_released=True)

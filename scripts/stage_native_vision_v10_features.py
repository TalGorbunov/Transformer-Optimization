"""CPU V10 training-only native features for every strict numeral/EOS prefix.

Keys contain only the actual image/question and preceding token IDs. Target
length, gold and set size never enter feature identities. All1782 unique scenes
are inventoried; weighted adjacent pairs independently validate variable T.
No teacher probabilities, old teacher labels, development or test features.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts.stage_native_vision_v6_teacher import MODEL,need,object_sha,prepared,read,save,sha,model_metadata
from scripts import probe_native_vision_parallel_local as old
from gnnformer.parallel_local_prompts import build_set_count_prompt as global_prompt
DATA=Path('/mnt/data/gabriele/gnn_transformer/v10_parallel_local')
OUT=REPO/'outputs/native_aggregation_vlm/v10/features'
TRAIN=DATA.parent/'v10_balanced'
NATIVE_REFERENCE=REPO/'outputs/native_aggregation_vlm/v7/runtime/profile_441845/summary.json'
PHASES=('local_empty','local_prefix','global_empty','global_prefix')
OWN=('scripts/stage_native_vision_v10_features.py','scripts/cache_native_vision_v10_features.py',
     'slurm/native_vision_v10_feature_stage.sbatch','slurm/native_vision_v10_feature_profile.sbatch',
     'slurm/native_vision_v10_feature_harvest.sbatch','slurm/native_vision_v10_feature_merge.sbatch',
     'scripts/stage_native_vision_v7_features.py','scripts/stage_native_vision_v6_teacher.py',
     'scripts/probe_native_vision_parallel_local.py','scripts/probe_native_vision_v2_prefix.py',
     'gnnformer/data.py','gnnformer/runtime.py','gnnformer/parallel_local_prompts.py',
     'gnnformer/paired_sequence_objectives.py','tests/test_paired_sequence_objectives.py')


def sources():return {name:sha(REPO/name) for name in OWN}


def snapshot(directory):
    (directory/'source').mkdir()
    for name in OWN:(directory/'source'/name.replace('/','_')).write_bytes((REPO/name).read_bytes())
    save(directory/'source_hashes.json',sources())


def index(directory,title,links):
    (directory/'INDEX.md').write_text('# '+title+'\n\n'+''.join(f'- [{label}]({path})\n' for label,path in links))


def tensor_info(value):
    import torch
    value=value.detach().cpu().contiguous()
    return dict(shape=list(value.shape),dtype=str(value.dtype),
                sha256=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest())


def feature_id(kind,identity,prefix):return object_sha([kind,identity,list(prefix)])


def inventory(rows,source_data,target_ids,eos):
    from gnnformer.paired_sequence_objectives import sequence_layout
    features={};scenes={}
    for row in rows:
        source=source_data['scenes'][row['sid']]
        need(all(source[k]==row[k] for k in ('question','n_frames','path','qa_sha256')),'Source scene identity differs')
        need(row['split']=='train' and type(row['gold']) is int and row['gold'] in range(17),'Only registered V10 training scenes allowed')
        target=list(target_ids[str(row['gold'])]);need(target[-1]==eos and eos not in target[:-1],'Invalid full native numeral/EOS target')
        prefixes=sequence_layout([target,target])['prefixes'][0];local_ids=[]
        for pid in source['pair_ids']:
            pair=source_data['pairs'][pid]
            need(pair['question']==row['question'] and pid==object_sha([pair['image_sha256'],pair['question']]),'Local pair identity differs')
            image_features=[]
            for prefix in prefixes:
                fid=feature_id('local',pid,prefix)
                value=dict(feature_id=fid,kind='local',pair_id=pid,question=pair['question'],
                    question_sha256=object_sha(pair['question']),prefix_ids=prefix,base_layout_id=pair['layout_id'],
                    phase='local_prefix' if prefix else 'local_empty')
                need(fid not in features or features[fid]==value,'One feature ID maps to inconsistent actual inputs')
                features[fid]=value;image_features.append(fid)
            local_ids.append(image_features)
        global_ids=[]
        for prefix in prefixes:
            fid=feature_id('global',row['question'],prefix)
            value=dict(feature_id=fid,kind='global',question=row['question'],question_sha256=object_sha(row['question']),
                prefix_ids=prefix,base_layout_id=object_sha(['global',global_prompt(row['question'])]),
                phase='global_prefix' if prefix else 'global_empty')
            need(fid not in features or features[fid]==value,'Global input identity differs')
            features[fid]=value;global_ids.append(fid)
        need(len(local_ids)==row['n_frames'] and all(len(x)==len(target) for x in local_ids),'Scene local feature shape differs')
        scenes[row['sid']]=dict(path=row['path'],qa_sha256=row['qa_sha256'],content_sha256=row['content_sha256'],
            question=row['question'],n_frames=row['n_frames'],target_ids=target,target_prefixes=prefixes,
            image_question_pair_ids=list(source['pair_ids']),local_feature_ids=local_ids,global_feature_ids=global_ids,
            split='train',gold=row['gold'])
    groups={phase:sorted(fid for fid,x in features.items() if x['phase']==phase) for phase in PHASES}
    return features,scenes,groups


def profile_ids(features,groups,labels,count_ids):
    selected={phase:[] for phase in PHASES}
    categories=('positive','char_only','room_only','neither')
    for category in categories:
        candidates=sorted((fid for fid in groups['local_empty'] if labels[features[fid]['pair_id']]==category),
                          key=lambda fid:features[fid]['pair_id'])
        need(len(candidates)>=2,'Missing local empty profile category');selected['local_empty'].extend(candidates[:2])
    wanted={'positive':('9','16'),'char_only':('0','10'),'room_only':('9','10'),'neither':('0','1')}
    for category in categories:
        for text in wanted[category]:
            prefix=[count_ids[d] for d in text]
            candidates=sorted((fid for fid in groups['local_prefix'] if labels[features[fid]['pair_id']]==category
                               and features[fid]['prefix_ids']==prefix),key=lambda fid:features[fid]['pair_id'])
            need(candidates,'Missing exact local prefix/category software case')
            selected['local_prefix'].append(candidates[0])
    selected['global_empty']=sorted(groups['global_empty'],key=lambda fid:features[fid]['question'])[:8]
    for text in ('0','9','1','10','16','2','11','15'):
        prefix=[count_ids[d] for d in text]
        candidates=sorted((fid for fid in groups['global_prefix'] if features[fid]['prefix_ids']==prefix),
                          key=lambda fid:features[fid]['question'])
        need(candidates,'Missing exact global prefix software case');selected['global_prefix'].append(candidates[0])
    need(all(len(v)==len(set(v))==8 for v in selected.values()) and len({f for v in selected.values() for f in v})==32,
         'Profile must contain32 distinct features,8 per phase')
    required={(),(count_ids['0'],),(count_ids['9'],),(count_ids['1'],),(count_ids['1'],count_ids['0']),(count_ids['1'],count_ids['6'])}
    for kind in ('local','global'):
        observed={tuple(features[fid]['prefix_ids']) for phase in PHASES if phase.startswith(kind) for fid in selected[phase]}
        need(required<=observed,'Profile omitted a strict prefix of0/9/10/16')
    return selected


def pack(torch,rows,pad):
    need(rows,'Empty feature batch')
    visual='pixel_values' in rows[0]
    need(all(('pixel_values' in row)==visual for row in rows),'Local/global extraction groups must stay separate')
    length=max(row['input_ids'].shape[1] for row in rows)
    ids=torch.full((len(rows),length),pad,dtype=torch.long);mask=torch.zeros_like(ids)
    for i,row in enumerate(rows):
        need(row['input_ids'].ndim==2 and row['input_ids'].shape[0]==1
             and row['attention_mask'].shape==row['input_ids'].shape
             and bool((row['attention_mask']==1).all()),'Expected complete unpadded input')
        size=row['input_ids'].shape[1];ids[i,-size:]=row['input_ids'][0];mask[i,-size:]=1
    result=dict(input_ids=ids,attention_mask=mask)
    if visual:
        result.update(pixel_values=torch.cat([r['pixel_values'] for r in rows]),
                      image_grid_thw=torch.cat([r['image_grid_thw'] for r in rows]))
    return result


def row_inputs(torch,plan,pixels,fid):
    feature=plan['features'][fid];layout=plan['layouts'][feature['layout_id']]
    ids=torch.tensor([layout['input_ids']],dtype=torch.long)
    result=dict(input_ids=ids,attention_mask=torch.ones_like(ids))
    if feature['kind']=='local':
        pair=plan['pairs'][feature['pair_id']]
        result.update(pixel_values=pixels[pair['image_sha256']],
                      image_grid_thw=torch.tensor(layout['image_grid_thw'],dtype=torch.long))
    return result


def native_api(processor):
    import inspect
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    from bitsandbytes.autograd import _functions
    owner,fn,binding=old.native_api(processor)
    for item in (Qwen2RMSNorm,_functions):
        path=str(Path(inspect.getfile(item)).resolve())
        binding['source_sha256'][path]=sha(path)
    return owner,fn,binding


def runtime_identity():
    import torch,transformers
    return dict(torch_version=str(torch.__version__),transformers_version=str(transformers.__version__),
                bitsandbytes_version=importlib.metadata.version('bitsandbytes'))




def verify_plan(path,*,pixels=True):
    from gnnformer.paired_sequence_objectives import sequence_layout
    path=Path(path).resolve()
    need(path.is_relative_to(DATA) and sha(path)==path.with_suffix('.sha256').read_text().strip(),'V10 feature plan path/sidecar differs')
    plan=read(path)
    need(plan['schema_version']==1 and plan['protocol']=='v10_parallel_local_training_features'
         and plan['source_sha256']==sources() and plan['model']==model_metadata(),'V10 frozen source/model identity changed')
    need(plan['runtime']==runtime_identity(),'V10 native runtime differs')
    for value in plan['source_files'].values():need(sha(value['path'])==value['sha256'],'Frozen source artifact changed')
    need(plan['counts']=={phase:len(plan['groups'][phase]) for phase in PHASES}
         and sum(plan['counts'].values())==len(plan['features']) and len(plan['scenes'])==plan['training_scene_count']==1782
         and plan['training_slots']==1836 and plan['training_pair_count']==918 and plan['training_question_count']==54
         and plan['counts']['global_empty']==54 and plan['counts']['global_prefix']==918,'V10 feature/support counts differ')
    need(plan['global_prompt_template']==global_prompt('{question}'),'Native global prompt differs')
    need(plan['inventory_sha256']==object_sha(dict(features=plan['features'],scenes=plan['scenes'],groups=plan['groups'])),
         'Frozen feature inventory differs')
    for scene in plan['scenes'].values():
        target=plan['target_token_ids'][str(scene['gold'])];prefixes=sequence_layout([target,target])['prefixes'][0]
        need(scene['target_ids']==target and scene['target_prefixes']==prefixes and len(scene['global_feature_ids'])==len(target)
             and len(scene['local_feature_ids'])==scene['n_frames']==len(scene['image_question_pair_ids']),'Variable-T scene target/layout differs')
        for pid,fids in zip(scene['image_question_pair_ids'],scene['local_feature_ids']):
            need(len(fids)==len(target),'Local feature sequence drops a prediction position')
            pair=plan['pairs'][pid]
            need(pair['question']==scene['question'] and pid==object_sha([pair['image_sha256'],pair['question']]),'Image/question key differs')
            for fid,prefix in zip(fids,prefixes):
                need(fid==feature_id('local',pid,prefix) and plan['features'][fid]['prefix_ids']==prefix
                     and plan['features'][fid]['pair_id']==pid,'Local feature has future-target leakage')
        for fid,prefix in zip(scene['global_feature_ids'],prefixes):
            need(fid==feature_id('global',scene['question'],prefix) and plan['features'][fid]['prefix_ids']==prefix,
                 'Global feature depends on non-prefix metadata')
    pairing=read(plan['source_files']['training_pairing']['path'])
    slots=[sid for pair in pairing['pairs'] for sid in pair['sids']]
    layout=sequence_layout([plan['scenes'][sid]['target_ids'] for sid in slots])
    need(layout==plan['paired_sequence_layout'] and object_sha(layout)==plan['paired_sequence_layout_sha256']
         and len(layout['targets'])==plan['training_target_positions']==4428,'Full paired variable-T layout differs')
    for pair in pairing['pairs']:
        a,b=[plan['scenes'][sid] for sid in pair['sids']]
        need(a['question']==b['question']==pair['question'] and a['gold']==b['gold']==pair['gold'] and a['target_ids']==b['target_ids']
             and a['global_feature_ids']==b['global_feature_ids'],'Paired question/target/global prefixes differ')
        if pair['same_sid']:need(a['local_feature_ids']==b['local_feature_ids'],'Same-SID saturation pair differs')
    shards=[sum((plan['groups'][phase][i::4] for phase in PHASES),[]) for i in range(4)]
    need(plan['shards']==shards and sorted(fid for group in shards for fid in group)==sorted(plan['features']),
         'Four deterministic shards do not partition all features')
    need(all(len(plan['profile_groups'][phase])==len(set(plan['profile_groups'][phase]))==8 for phase in PHASES),
         'Profile phase cardinality differs')
    if pixels:need(sha(plan['pixels_file'])==plan['pixels_sha256'],'Pixel blob changed')
    return plan


def self_test(torch):
    from gnnformer.paired_sequence_objectives import sequence_layout
    source=dict(scenes={},pairs={'p':dict(question='q',image_sha256='image',layout_id='layout')})
    pid=object_sha(['image','q']);source['pairs'][pid]=source['pairs'].pop('p');source['pairs'][pid]['pair_id']=pid
    targets={'10':[11,10,99],'16':[11,16,99]};rows=[]
    for k in (10,16):
        row=dict(sid=str(k),path='/'+str(k),qa_sha256=str(k),content_sha256=str(k),question='q',n_frames=1,gold=k,split='train')
        rows.append(row);source['scenes'][row['sid']]=dict(row,pair_ids=[pid])
    f,s,g=inventory(rows,source,targets,99)
    need(s['10']['local_feature_ids'][0][:2]==s['16']['local_feature_ids'][0][:2]
         and s['10']['global_feature_ids'][:2]==s['16']['global_feature_ids'][:2]
         and s['10']['local_feature_ids'][0][2]!=s['16']['local_feature_ids'][0][2],
         'Shared strict prefixes depend on future digits')
    need(all(99 not in x['prefix_ids'] for x in f.values()) and all('gold' not in x and 'n_frames' not in x for x in f.values()),
         'Feature includes EOS/future labels or length metadata')
    layout=sequence_layout([[10,99],[10,99],[11,10,99],[11,10,99]])
    need(layout['offsets']==[0,2,4,7,10] and layout['prefixes'][2]==[[],[11],[11,10]],'Actual variable-T helper differs')
    a=dict(input_ids=torch.tensor([[1,2]]),attention_mask=torch.ones(1,2,dtype=torch.long))
    b=dict(input_ids=torch.tensor([[3]]),attention_mask=torch.ones(1,1,dtype=torch.long))
    packed=pack(torch,[a,b],0)
    need(packed['input_ids'].tolist()==[[1,2],[0,3]] and packed['attention_mask'].tolist()==[[1,1],[0,1]],'Native left-padding differs')
    return dict(passed=True,tests=['shared_prefix_future_digit_independence','no_target_length_or_gold_keys',
        'all_strict_prefixes_no_EOS','actual_variable_T_sequence_layout','native_left_padding'])


def stage(args,out,frozen):
    import ast
    import torch,transformers
    from transformers import AutoProcessor
    from gnnformer.data import build_prompt_inputs,build_count_prompt
    from gnnformer.paired_sequence_objectives import sequence_layout
    from scripts.probe_native_vision_v2_prefix import fingerprint
    torch.set_num_threads(4);start=time.monotonic();tests=self_test(torch)
    unit=subprocess.run([sys.executable,str(REPO/'tests/test_paired_sequence_objectives.py')],capture_output=True,text=True)
    need(unit.returncode==0,'Variable-T helper tests failed: '+unit.stderr)
    tests['sequence_helper_suite']=dict(returncode=unit.returncode,stdout=unit.stdout,stderr=unit.stderr)
    from scripts.cache_native_vision_v10_features import self_test as worker_self_test
    tests['feature_worker_suite']=worker_self_test()
    need(tests['feature_worker_suite']['passed'] is True,'Feature worker metadata tests failed')
    source_files={}
    def bind(label,path,expected=None):
        path=Path(path).resolve();actual=sha(path);need(expected is None or actual==expected,'Source artifact changed: '+str(path))
        source_files[label]=dict(path=str(path),sha256=actual);return read(path)
    manifest_path=args.training_manifest.resolve();schedule_path=args.training_schedule.resolve();pairing_path=args.training_pairing.resolve()
    need((manifest_path,schedule_path,pairing_path)==(TRAIN/'main_manifest.json',TRAIN/'schedule.json',TRAIN/'pairing.json'),
         'Only the canonical V10 training inventory is allowed')
    manifest=bind('training_manifest',manifest_path);schedule=bind('training_schedule',schedule_path);pairing=bind('training_pairing',pairing_path)
    need(schedule['manifest_file']==str(manifest_path) and schedule['manifest_sha256']==sha(manifest_path)
         and pairing['manifest_file']==str(manifest_path) and pairing['manifest_sha256']==sha(manifest_path)
         and pairing['schedule_file']==str(schedule_path) and pairing['schedule_sha256']==sha(schedule_path),'Manifest/schedule/pair bindings differ')
    for label in ('audit','stage_plan','inventory'):bind('training_'+label,manifest[label+'_file'],manifest[label+'_sha256'])
    audited=read(manifest['audit_file']);need(audited['passed'] and audited['all_current_contexts_disjoint']
        and audited['all_qa_hashes_and_gold_recounts_passed'] and audited['all_image_semantic_links_passed'],'V10 semantic audit did not pass')
    stage_audit=bind('training_stage_audit',manifest['stage_audit_file'])
    need(stage_audit['passed'] and stage_audit['completed'] and stage_audit['manifest_sha256']['balanced']==sha(manifest_path)
         and stage_audit['schedule_sha256']==sha(schedule_path) and stage_audit['pairing_sha256']==sha(pairing_path),
         'V10 completed data-stage provenance differs')
    bind('training_dry_plan',audited['dry_plan_file'],audited['dry_plan_sha256'])
    for name,digest in manifest['source_sha256'].items():
        need(sha(REPO/name)==digest,'Frozen V10 data-generation code changed')
        source_files['data_source_'+name]=dict(path=str(REPO/name),sha256=digest)
    rows=manifest['splits']['train_N8']['samples']+manifest['splits']['train_N16']['samples'];by_sid={r['sid']:r for r in rows}
    slots=schedule['epoch_slots'];frequencies=Counter(slots);questions=sorted({r['question'] for r in rows})
    need(len(rows)==len(by_sid)==1782 and Counter(r['n_frames'] for r in rows)=={8:486,16:1296}
         and len(slots)==len(schedule['slot_metadata'])==1836 and len(pairing['pairs'])==918 and set(slots)==set(by_sid)
         and schedule['unique_training_sids']==sorted(by_sid) and slots==[sid for p in pairing['pairs'] for sid in p['sids']],
         'V10 training inventory or weighted pairing differs')
    need(all(frequencies[sid]==(2 if row['gold']==16 else 1) for sid,row in by_sid.items()),'Only saturated N16K16 may repeat')
    need(len(questions)==54 and Counter((by_sid[s]['question'],by_sid[s]['gold']) for s in slots)==
         Counter({(q,k):2 for q in questions for k in range(17)}),'V10 question/count weights differ')
    for i,(sid,slot) in enumerate(zip(slots,schedule['slot_metadata'])):
        need(slot['slot']==i and slot['sid']==sid and all(slot[k]==by_sid[sid][k] for k in ('n_frames','gold','question')),
             'Ordered slot metadata differs')
    native_reference=bind('native_profile',NATIVE_REFERENCE)
    need(native_reference['completed'] and native_reference['computational_integrity_passed'] and native_reference['zero_identity_passed'],
         'Native integration contract did not pass')
    native_plan=bind('native_profile_plan',native_reference['plan_file'],native_reference['plan_sha256'])
    need(native_plan['model']==model_metadata() and native_plan['runtime']==runtime_identity(),'Native model/runtime ancestor differs')
    sidecar=Path(native_reference['plan_file']).with_suffix('.sha256')
    need(sidecar.read_text().strip()==native_reference['plan_sha256'],'Native ancestor plan sidecar differs')
    source_files['native_profile_plan_sidecar']=dict(path=str(sidecar),sha256=sha(sidecar))
    for name,digest in native_reference['source_sha256'].items():
        need(sha(REPO/name)==digest,'Native integration source changed')
        source_files['native_source_'+name]=dict(path=str(REPO/name),sha256=digest)
    source_data=dict(pairs={},scenes={});labels={};image_hash_cache={}
    for row in rows:
        path=Path(row['path']);text=(path/'qa.txt').read_text().splitlines()
        need(row['split']=='train' and sha(path/'qa.txt')==row['qa_sha256'] and int(text[text.index('answer:')+1])==row['gold'],'Training QA changed')
        section=[x.strip() for x in text[text.index('question:')+1:text.index('answer:')] if x.strip()]
        states=[ast.literal_eval(x) for x in section if x.startswith('{')]
        need([x for x in section if not x.startswith('{')]==[row['question']]
             and len(states)==len(row['image_files'])==row['n_frames']
             and object_sha(dict(states=states,question=row['question']))==row['content_sha256'],'Training question/state identity differs')
        ids=[]
        for i,(image,state) in enumerate(zip(row['image_files'],states)):
            path=Path(image['path']);stat=path.stat();key=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
            need(path==Path(row['path'])/f'{i:03d}.png' and stat.st_size==image['bytes'] and state['step_id']==i+1 and len(state['rooms'])==1,
                 'Actual image order/Step differs')
            if key not in image_hash_cache:image_hash_cache[key]=sha(path)
            need(image_hash_cache[key]==image['sha256'],'Training image checksum differs')
            room,characters=next(iter(state['rooms'].items()));need(len(characters)==1,'Expected exactly one character per frame')
            character=characters[0];label='positive' if character==row['target_character'] and room==row['target_room'] else (
                'char_only' if character==row['target_character'] else 'room_only' if room==row['target_room'] else 'neither')
            pid=object_sha([image['sha256'],row['question']]);ids.append(pid)
            if pid in labels:need(labels[pid]==label,'Conflicting image/question semantics')
            else:source_data['pairs'][pid]=dict(pair_id=pid,image_path=str(path),image_sha256=image['sha256'],image_bytes=image['bytes'],question=row['question']);labels[pid]=label
        source_data['scenes'][row['sid']]=dict(row,pair_ids=ids)
        need(sum(labels[pid]=='positive' for pid in ids)==row['gold'],'Independent training label recount differs')
    processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,use_fast=False)
    processor_fp=fingerprint(processor,str(transformers.__version__))
    need(processor_fp==native_plan['processor'],'Native processor differs from unchanged integration contract')
    eos=processor.tokenizer.eos_token_id;need(eos==151645,'Native training EOS changed')
    count_ids={};target_ids={}
    for k in range(17):
        ids=processor.tokenizer(str(k),add_special_tokens=False)['input_ids']
        need(ids and all(t not in processor.tokenizer.all_special_ids for t in ids)
             and processor.tokenizer.decode(ids,skip_special_tokens=False)==str(k),'Native numeral tokenization is unsupported')
        target_ids[str(k)]=ids+[eos]
        if k<10:need(len(ids)==1,'Expected each individual digit as one native token');count_ids[str(k)]=ids[0]
    need(len(set(count_ids.values()))==10,'Digit token IDs collide')
    need(all(target_ids[str(k)][:-1]==[count_ids[d] for d in str(k)] for k in range(17)),'Native decimal digit boundary changed')
    prefix_vocabulary=sorted({tuple(prefix) for ids in target_ids.values() for prefix in sequence_layout([ids,ids])['prefixes'][0]},key=lambda x:(len(x),x))
    need(len(prefix_vocabulary)==18,'Expected18 strict prefixes per question')
    base_layouts={};base_text={};by_question={};local_keys={}
    for pair in source_data['pairs'].values():by_question.setdefault(pair['question'],pair)
    for question in questions:
        pair=by_question[question];inputs=prepared(processor,pair);grid=inputs['image_grid_thw'].tolist()
        key=object_sha(['local',question,grid]);local_keys[question]=key;ids=inputs['input_ids'][0].tolist()
        text=processor.apply_chat_template([dict(role='user',content=[dict(type='image'),dict(type='text',text=build_count_prompt(question,1))])],
            tokenize=False,add_generation_prompt=True)
        image_token=getattr(processor,'image_token','<|image_pad|>');need(text.count(image_token)==1,'Local image placeholder differs')
        expanded=text.replace(image_token,image_token*(math.prod(grid[0])//int(processor.image_processor.merge_size)**2))
        need(processor.tokenizer(expanded,add_special_tokens=False)['input_ids']==ids,'Full local processor/token template differs')
        base_layouts[key]=dict(input_ids=ids,image_grid_thw=grid,prompt_tokens=len(ids),input_ids_sha256=object_sha(ids));base_text[key]=expanded
        prompt=global_prompt(question);inputs=build_prompt_inputs(processor,[],prompt)
        need(set(inputs)=={'input_ids','attention_mask'},'Global stream must be text only')
        key=object_sha(['global',prompt]);ids=inputs['input_ids'][0].tolist()
        text=processor.apply_chat_template([dict(role='user',content=[dict(type='text',text=prompt)])],add_generation_prompt=True,tokenize=False)
        need(processor.tokenizer(text,add_special_tokens=False)['input_ids']==ids,'Full global processor/template differs')
        base_layouts[key]=dict(input_ids=ids,prompt_tokens=len(ids),input_ids_sha256=object_sha(ids));base_text[key]=text
    for pair in source_data['pairs'].values():pair['layout_id']=local_keys[pair['question']]
    features,scenes,groups=inventory(rows,source_data,target_ids,eos);counts={p:len(groups[p]) for p in PHASES}
    need(counts['global_empty']==54 and counts['global_prefix']==918,'Global prefix coverage differs')
    paired_layout=sequence_layout([scenes[sid]['target_ids'] for sid in slots]);need(len(paired_layout['targets'])==4428,'Paired target-position coverage differs')
    for p in pairing['pairs']:
        a,b=[scenes[sid] for sid in p['sids']]
        need(a['question']==b['question']==p['question'] and a['gold']==b['gold']==p['gold']
             and a['target_ids']==b['target_ids'] and a['global_feature_ids']==b['global_feature_ids'],
             'Paired full target or strict global prefix differs')
        if p['same_sid']:need(a==b,'Same-SID saturated pair differs')
    owner,fn,native=native_api(processor);layouts={};eos_text=processor.tokenizer.decode([eos],skip_special_tokens=False)
    for key,base in base_layouts.items():
        for gold,target in target_ids.items():
            need(processor.tokenizer(base_text[key]+gold+eos_text,add_special_tokens=False)['input_ids']==base['input_ids']+target,
                 'Complete numeral/EOS continuation retokenizes the actual prompt boundary')
    for feature in features.values():
        base=base_layouts[feature['base_layout_id']];prefix=feature['prefix_ids'];key=object_sha([feature['base_layout_id'],prefix]);feature['layout_id']=key
        if key in layouts:continue
        ids=base['input_ids']+prefix;continuation=processor.tokenizer.decode(prefix,skip_special_tokens=False)
        need(processor.tokenizer(base_text[feature['base_layout_id']]+continuation,add_special_tokens=False)['input_ids']==ids,
             'Strict continuation prefix retokenizes actual prompt')
        value=dict(input_ids=ids,input_ids_sha256=object_sha(ids),prompt_tokens=len(ids),original_prompt_tokens=len(base['input_ids']),prefix_ids=prefix)
        item=dict(input_ids=torch.tensor([ids]),attention_mask=torch.ones(1,len(ids),dtype=torch.long))
        if feature['kind']=='local':value['image_grid_thw']=base['image_grid_thw'];item['image_grid_thw']=torch.tensor(base['image_grid_thw'])
        positions,deltas=fn(owner,input_ids=item['input_ids'],image_grid_thw=item.get('image_grid_thw'),attention_mask=item['attention_mask'])
        value.update(position_ids=positions.tolist(),position_ids_info=tensor_info(positions),rope_deltas=deltas.tolist());layouts[key]=value
    data=DATA/f'stage_{os.environ["SLURM_JOB_ID"]}';data.mkdir(parents=True,exist_ok=False)
    pixels={};representative={}
    for pid,pair in sorted(source_data['pairs'].items()):representative.setdefault(pair['image_sha256'],pair)
    for i,(image_sha,pair) in enumerate(sorted(representative.items())):
        inputs=prepared(processor,pair);base=base_layouts[pair['layout_id']]
        need(sha(pair['image_path'])==image_sha and inputs['input_ids'][0].tolist()==base['input_ids']
             and inputs['image_grid_thw'].tolist()==base['image_grid_thw'],'Full pixel processor/input parity differs')
        pixels[image_sha]=inputs['pixel_values'].contiguous().clone()
        if i%128==0:print(json.dumps(dict(processed_images=i+1,total=len(representative))),flush=True)
    pixel_info={key:tensor_info(value) for key,value in pixels.items()};pixel_file=data/'pixels.pt';torch.save(dict(schema_version=1,pixels=pixels),pixel_file)
    selected=profile_ids(features,groups,labels,count_ids)
    for fid in selected['local_empty']+selected['local_prefix']:
        pair=source_data['pairs'][features[fid]['pair_id']];inputs=prepared(processor,pair)
        need(tensor_info(inputs['pixel_values'])==pixel_info[pair['image_sha256']]
             and inputs['input_ids'][0].tolist()==base_layouts[pair['layout_id']]['input_ids'],'Profile actual processor differs')
    settings={name:getattr(processor.image_processor,name,None) for name in
              ('min_pixels','max_pixels','patch_size','temporal_patch_size','merge_size','do_resize','do_rescale','do_normalize','image_mean','image_std')}
    plan=dict(schema_version=1,protocol='v10_parallel_local_training_features',data_root=str(DATA),source_sha256=frozen,
        source_files=source_files,model=model_metadata(),runtime=runtime_identity(),processor=processor_fp,image_processor_settings=settings,
        native_reference_role='Immutable native model/runtime/processor contract only; no teacher targets or labels are loaded',
        resize=392,quantization='nf4_double_bf16',native_dtype='torch.float16',attention='sdpa',
        global_prompt_template=global_prompt('{question}'),local_prompt='unchanged canonical build_count_prompt(question,1)',
        features=features,scenes=scenes,groups=groups,counts=counts,layouts=layouts,pairs=source_data['pairs'],
        training_scene_count=1782,training_slots=1836,training_pair_count=918,training_question_count=54,training_target_positions=4428,
        paired_sequence_layout=paired_layout,paired_sequence_layout_sha256=object_sha(paired_layout),
        target_token_ids=target_ids,count_token_ids=count_ids,eos_token_id=eos,pad_token_id=processor.tokenizer.pad_token_id,
        strict_prefix_vocabulary=[list(x) for x in prefix_vocabulary],pixels_file=str(pixel_file),pixels_sha256=sha(pixel_file),pixel_info=pixel_info,native_api=native,
        shards=[sum((groups[phase][i::4] for phase in PHASES),[]) for i in range(4)],
        shard_rule='Within each fixed phase, sorted feature IDs modulo four; phases never share a batch',
        profile_groups=selected,profile_case_count=32,profile_calls=8,batch_size=64,
        profile_rule='Each phase:8 distinct fixed IDs, then cyclically repeat the same IDs to64 rows',
        profile_selection='Local empty first2pairs/category; localprefix positive9/16,char_only0/10,room_only9/10,neither0/1; globalempty first8questions; globalprefix firstquestion for0/9/1/10/16/2/11/15',
        profile_categories={fid:labels[features[fid]['pair_id']] for phase in ('local_empty','local_prefix') for fid in selected[phase]},
        numerical_gate=dict(full_vocabulary_tv_max=.02,top1_equal=True),
        projection_rule='load+pixels+1.5*sum(ceil(phase_shard_rows/64)*phase_stress_harvest_seconds)+30 <=600',
        scope='V10 TRAINING-only strict-prefix features; no teacher probabilities, test/dev harvesting, or labels inside feature keys',
        tests=tests,slurm_job_id=os.environ['SLURM_JOB_ID'])
    plan['inventory_sha256']=object_sha(dict(features=features,scenes=scenes,groups=groups))
    need(sources()==frozen and plan['model']==model_metadata(),'Source/native model changed during stage')
    path=data/'plan.json';save(path,plan);path.with_suffix('.sha256').write_text(sha(path)+'\n');verify_plan(path,pixels=False)
    summary=dict(passed=True,plan_file=str(path),plan_sha256=sha(path),counts=counts,total_features=len(features),
        native_hidden_payload_bytes=len(features)*3584*2,pixel_blob_bytes=pixel_file.stat().st_size,source_sha256=frozen,
        seconds=time.monotonic()-start,training_scene_count=1782,training_target_positions=4428,
        shard_counts=[dict(Counter(features[fid]['phase'] for fid in shard)) for shard in plan['shards']])
    save(out/'summary.json',summary);index(data,'V10 immutable feature inputs',[('Plan','plan.json'),('Processed pixels','pixels.pt')])
    print(json.dumps(summary,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--training-manifest',type=Path,default=TRAIN/'main_manifest.json')
    parser.add_argument('--training-schedule',type=Path,default=TRAIN/'schedule.json')
    parser.add_argument('--training-pairing',type=Path,default=TRAIN/'pairing.json');args=parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),
         'Feature staging/tests require CPU-only Slurm')
    out=OUT/f'{"selftest" if args.self_test else "stage"}_{os.environ["SLURM_JOB_ID"]}'
    out.mkdir(parents=True,exist_ok=False);snapshot(out);frozen=sources()
    index(out,'V10 training-feature CPU inventory',[('Summary','summary.json'),('Frozen sources','source_hashes.json')])
    if args.self_test:
        import torch
        torch.set_num_threads(4);save(out/'summary.json',dict(self_test(torch),source_sha256=frozen))
    else:stage(args,out,frozen)


if __name__=='__main__':main()

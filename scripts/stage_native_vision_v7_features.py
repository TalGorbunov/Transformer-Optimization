"""CPU inventory and immutable inputs for V7 training-only frozen local states.

Only the registered balanced V7 training data is included. Local prompts remain canonical N=1;
the global prompt is N-agnostic. Empty-prefix features are separate from every
gold-specific continuation feature. This cache is not a model/evaluation result.
All tokenizer, image, tensor and inventory work runs in Slurm CPU.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts.stage_native_vision_v6_teacher import (
    MODEL,need,object_sha,prepared,read,save,sha,model_metadata,verify_inputs)
from scripts import probe_native_vision_parallel_local as old
DATA=Path('/mnt/data/gabriele/gnn_transformer/v7_parallel_local')
OUT=REPO/'outputs/native_aggregation_vlm/v7/features'
TRAIN=DATA.parent/'v7_balanced'
PHASES=('local_empty','local_prefix','global_empty','global_prefix')
OWN=('scripts/stage_native_vision_v7_features.py','scripts/cache_native_vision_v7_features.py',
     'slurm/native_vision_v7_features_stage.sbatch','slurm/native_vision_v7_features_profile.sbatch',
     'slurm/native_vision_v7_features_array.sbatch','slurm/native_vision_v7_features_merge.sbatch',
     'scripts/stage_native_vision_v6_teacher.py','scripts/cache_native_vision_v6_teacher.py',
     'scripts/probe_native_vision_parallel_local.py','scripts/probe_native_vision_v2_prefix.py',
     'gnnformer/data.py','gnnformer/runtime.py','gnnformer/parallel_local_prompts.py')


from gnnformer.parallel_local_prompts import build_set_count_prompt as global_prompt


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


def feature_id(kind,identity,prefix):
    return object_sha([kind,identity,list(prefix)])


def inventory(rows,teacher,count_ids,eos):
    features={};scenes={}
    for row in rows:
        source=teacher['scenes'][row['sid']]
        need(source['question']==row['question'] and source['n_frames']==row['n_frames']
             and source['path']==row['path'] and source['qa_sha256']==row['qa_sha256'],
             'V4/teacher scene identity differs')
        need(0<=row['gold']<=8 and row['split']=='train','Only registered training counts are allowed')
        digit=count_ids[str(row['gold'])];local_ids=[]
        for pid in source['pair_ids']:
            pair=teacher['pairs'][pid]
            need(pair['question']==row['question'],'Pair question differs from scene')
            pair_features=[]
            for prefix in ([],[digit]):
                fid=feature_id('local',pid,prefix)
                features[fid]=dict(feature_id=fid,kind='local',pair_id=pid,
                    question=pair['question'],question_sha256=object_sha(pair['question']),
                    prefix_ids=prefix,base_layout_id=pair['layout_id'],
                    phase='local_prefix' if prefix else 'local_empty')
                pair_features.append(fid)
            local_ids.append(pair_features)
        global_ids=[]
        for prefix in ([],[digit]):
            fid=feature_id('global',row['question'],prefix)
            features[fid]=dict(feature_id=fid,kind='global',question=row['question'],
                question_sha256=object_sha(row['question']),prefix_ids=prefix,
                base_layout_id=object_sha(['global',global_prompt(row['question'])]),
                phase='global_prefix' if prefix else 'global_empty')
            global_ids.append(fid)
        need(len(local_ids)==row['n_frames'],'Scene feature order/length differs')
        scenes[row['sid']]=dict(path=row['path'],qa_sha256=row['qa_sha256'],
            content_sha256=row['content_sha256'],question=row['question'],n_frames=row['n_frames'],
            target_ids=[digit,eos],local_feature_ids=local_ids,global_feature_ids=global_ids,
            split='train',gold=row['gold'])
    groups={phase:sorted(fid for fid,x in features.items() if x['phase']==phase) for phase in PHASES}
    return features,scenes,groups


def profile_ids(features,groups,labels,count_ids):
    selected={phase:[] for phase in PHASES};used=set()
    categories=('positive','char_only','room_only','neither')
    for category in categories:
        empty=sorted(fid for fid in groups['local_empty'] if labels[features[fid]['pair_id']]['category']==category)
        # "first sorted/category" means sorted original pair ID, not hash of its feature key.
        empty.sort(key=lambda fid:features[fid]['pair_id'])
        selected['local_empty'].extend(empty[:3])
        for wanted in (0,1,8):
            candidates=[fid for fid in groups['local_prefix'] if fid not in used
                        and labels[features[fid]['pair_id']]['category']==category]
            order=[wanted]+[k for k in range(9) if k!=wanted]
            candidates.sort(key=lambda fid:(order.index(next(k for k in range(9)
                if count_ids[str(k)]==features[fid]['prefix_ids'][0])),features[fid]['pair_id']))
            need(candidates,'Missing deterministic profile continuation')
            selected['local_prefix'].append(candidates[0]);used.add(candidates[0])
    selected['global_empty']=sorted(groups['global_empty'],key=lambda fid:features[fid]['question'])[:4]
    for wanted in (0,1,8,2):
        order=[wanted]+[k for k in range(9) if k!=wanted]
        candidates=[fid for fid in groups['global_prefix'] if fid not in used]
        candidates.sort(key=lambda fid:(order.index(next(k for k in range(9)
            if count_ids[str(k)]==features[fid]['prefix_ids'][0])),features[fid]['question']))
        selected['global_prefix'].append(candidates[0]);used.add(candidates[0])
    need([len(selected[p]) for p in PHASES]==[12,12,4,4]
         and len(set(fid for ids in selected.values() for fid in ids))==32,'Profile coverage differs')
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
    path=Path(path)
    need(path.is_relative_to(DATA) and sha(path)==path.with_suffix('.sha256').read_text().strip(),
         'V7 plan path/sidecar differs')
    plan=read(path)
    need(plan['schema_version']==1 and plan['protocol']=='v7_parallel_local_training_features'
         and plan['source_sha256']==sources() and plan['model']==model_metadata(),'V7 source/model identity changed')
    need(plan['runtime']==runtime_identity(),'V7 runtime differs')
    for value in plan['source_files'].values():need(sha(value['path'])==value['sha256'],'Frozen training/teacher source changed')
    need(plan['counts']=={phase:len(plan['groups'][phase]) for phase in PHASES}
         and sum(plan['counts'].values())==len(plan['features'])
         and len(plan['scenes'])==plan['training_scene_count']==1890
         and plan['training_slots']==1944 and plan['training_question_count']==54
         and plan['counts']['global_empty']==54 and plan['counts']['global_prefix']==486,
         'Balanced feature/scene cardinality differs')
    need(plan['global_prompt_template']==global_prompt('{question}'),'Global N-agnostic prompt differs')
    need(plan['inventory_sha256']==object_sha(dict(features=plan['features'],scenes=plan['scenes'],groups=plan['groups'])),
         'Feature inventory identity differs')
    rebuilt=[sum((plan['groups'][phase][i::4] for phase in PHASES),[]) for i in range(4)]
    need(plan['shards']==rebuilt and sorted(fid for shard in rebuilt for fid in shard)==sorted(plan['features']),
         'Four deterministic shards do not exactly partition the inventory')
    if pixels:need(sha(plan['pixels_file'])==plan['pixels_sha256'],'Frozen pixel tensor blob changed')
    return plan


def self_test(torch):
    need(feature_id('local','pair',[])!=feature_id('local','pair',[7]),'Empty/prefix keys collided')
    need(feature_id('global','question',[])==feature_id('global','question',[]),'Global identity depends on no N')
    teacher=dict(scenes={},pairs={'p':dict(question='q',layout_id='layout')})
    rows=[]
    for i,gold in enumerate((0,1)):
        sid=str(i);row=dict(sid=sid,path='/'+sid,qa_sha256=sid,content_sha256=sid,
                           question='q',n_frames=1,gold=gold,split='train')
        rows.append(row);teacher['scenes'][sid]=dict(row,pair_ids=['p'])
    f,s,g=inventory(rows,teacher,{str(k):k+10 for k in range(9)},99)
    need(len(f)==6 and s['0']['local_feature_ids'][0][0]==s['1']['local_feature_ids'][0][0]
         and s['0']['global_feature_ids'][0]==s['1']['global_feature_ids'][0]
         and s['0']['local_feature_ids'][0][1]!=s['1']['local_feature_ids'][0][1],
         'Gold-dependent empty feature or unobserved prefix expansion')
    a=dict(input_ids=torch.tensor([[1,2]]),attention_mask=torch.ones(1,2,dtype=torch.long))
    b=dict(input_ids=torch.tensor([[3]]),attention_mask=torch.ones(1,1,dtype=torch.long))
    packed=pack(torch,[a,b],0)
    need(packed['input_ids'].tolist()==[[1,2],[0,3]] and packed['attention_mask'].tolist()==[[1,1],[0,1]],
         'Left-padding contract differs')
    return dict(passed=True,tests=['empty_prefix_independence','observed_prefix_only','global_N_independence','left_padding'])


def stage(args):
    import ast
    import torch,transformers
    from transformers import AutoProcessor
    from gnnformer.data import build_prompt_inputs,build_count_prompt
    from scripts.stage_native_vision_v6_teacher import category
    from scripts.probe_native_vision_v2_prefix import fingerprint
    torch.set_num_threads(4);start=time.monotonic();frozen=sources();tests=self_test(torch)
    teacher,cache,_old_labels,_,binding=old.teacher_inputs()
    need(runtime_identity()==teacher['runtime'],'Frozen teacher/runtime differs')
    manifest_path=args.training_manifest.resolve();schedule_path=args.training_schedule.resolve()
    need(manifest_path.is_relative_to(DATA.parent) and schedule_path.is_relative_to(DATA.parent),
         'Training sources must be under the user data root')
    manifest=read(manifest_path);schedule=read(schedule_path)
    need(schedule['purpose']=='v7_balanced_epoch' and sha(manifest_path)==schedule['manifest_sha256']
         and Path(schedule['manifest_file']).resolve()==manifest_path,'Balanced schedule binding differs')
    rows=manifest['splits']['train_N8']['samples']+manifest['splits']['train_N16']['samples']
    by_sid={r['sid']:r for r in rows};slots=schedule['epoch_slots'];frequencies=Counter(slots)
    need(len(rows)==len(by_sid)==1890 and Counter(r['n_frames'] for r in rows)=={8:918,16:972}
         and len(slots)==1944 and set(slots)==set(by_sid)
         and schedule['unique_training_sids']==sorted(by_sid),'Balanced training coverage differs')
    need(all(frequencies[sid]==(2 if row['n_frames']==8 and row['gold']==8 else 1)
             for sid,row in by_sid.items()),'Saturated versus ordinary presentation weights differ')
    support=Counter((by_sid[sid]['question'],by_sid[sid]['n_frames'],by_sid[sid]['gold']) for sid in slots)
    questions=sorted({r['question'] for r in rows})
    need(len(questions)==54 and support==Counter({(q,n,k):2 for q in questions for n in (8,16) for k in range(9)}),
         'Question/N/count training support is not fully balanced')
    need(len(schedule['slot_metadata'])==1944,'Slot metadata coverage differs')
    for i,(sid,slot) in enumerate(zip(slots,schedule['slot_metadata'])):
        need(slot['slot']==i and slot['sid']==sid and all(slot[key]==by_sid[sid][key]
             for key in ('n_frames','gold','question')),'Ordered slot metadata differs')
    source_data=dict(pairs={},scenes={});labels={};image_metadata={};image_hash_cache={}
    for row in rows:
        path=Path(row['path']);text=(path/'qa.txt').read_text().splitlines()
        need(row['split']=='train' and sha(path/'qa.txt')==row['qa_sha256']
             and int(text[text.index('answer:')+1])==row['gold'],'Training QA/target changed')
        section=[x.strip() for x in text[text.index('question:')+1:text.index('answer:')] if x.strip()]
        actual_question=[x for x in section if not x.startswith('{')]
        states=[ast.literal_eval(x) for x in section if x.startswith('{')]
        need(actual_question==[row['question']] and len(states)==len(row['image_files'])==row['n_frames']
             and object_sha(dict(states=states,question=row['question']))==row['content_sha256'],
             'Actual QA question/frame/content inventory differs')
        ids=[]
        for i,(image,state) in enumerate(zip(row['image_files'],states)):
            need(Path(image['path'])==path/f'{i:03d}.png' and Path(image['path']).stat().st_size==image['bytes']
                 and state['step_id']==i+1 and len(state['rooms'])==1,'Image order/Step differs')
            stat=Path(image['path']).stat()
            inode_key=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
            if inode_key not in image_hash_cache:image_hash_cache[inode_key]=sha(image['path'])
            need(image_hash_cache[inode_key]==image['sha256'],'Balanced training image content changed')
            room,characters=next(iter(state['rooms'].items()))
            need(len(characters)==1,'Unexpected multi-character frame')
            label=category(characters[0],room,row['target_character'],row['target_room'])
            pid=object_sha([image['sha256'],row['question']]);ids.append(pid)
            pair=dict(pair_id=pid,image_path=image['path'],image_sha256=image['sha256'],
                      image_bytes=image['bytes'],question=row['question'])
            if pid in labels:need(labels[pid]['category']==label,'Conflicting repeated pair semantics')
            else:
                source_data['pairs'][pid]=pair;labels[pid]=dict(category=label)
            image_metadata.setdefault(image['sha256'],dict(image))
        source_data['scenes'][row['sid']]=dict(row,pair_ids=ids)
        need(sum(labels[pid]['category']=='positive' for pid in ids)==row['gold'],'Independent local truth recount differs')
    processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,use_fast=False)
    need(fingerprint(processor,str(transformers.__version__))==teacher['processor'],'Processor differs from teacher ancestor')
    count_ids={}
    for k in range(9):
        ids=processor.tokenizer(str(k),add_special_tokens=False)['input_ids']
        need(len(ids)==1 and ids[0] not in processor.tokenizer.all_special_ids,'Training count is not one ordinary token')
        count_ids[str(k)]=ids[0]
    need(len(set(count_ids.values()))==9 and processor.tokenizer.eos_token_id is not None,'Invalid targets/EOS')
    base_layouts={};base_text={};by_question={}
    for pair in source_data['pairs'].values():by_question.setdefault(pair['question'],pair)
    local_keys={}
    for question in questions:
        pair=by_question[question];inputs=prepared(processor,pair);grid=inputs['image_grid_thw'].tolist()
        key=object_sha(['local',question,grid]);local_keys[question]=key
        ids=inputs['input_ids'][0].tolist()
        text=processor.apply_chat_template([dict(role='user',content=[dict(type='image'),
                 dict(type='text',text=build_count_prompt(question,1))])],tokenize=False,add_generation_prompt=True)
        image_token=getattr(processor,'image_token','<|image_pad|>')
        need(text.count(image_token)==1,'Expected exactly one local image placeholder')
        expanded=text.replace(image_token,image_token*(math.prod(grid[0])//int(processor.image_processor.merge_size)**2))
        need(processor.tokenizer(expanded,add_special_tokens=False)['input_ids']==ids,'Actual local processor/template mismatch')
        base_layouts[key]=dict(input_ids=ids,image_grid_thw=grid,prompt_tokens=len(ids),input_ids_sha256=object_sha(ids))
        base_text[key]=expanded
        prompt=global_prompt(question);global_inputs=build_prompt_inputs(processor,[],prompt)
        need(set(global_inputs)=={'input_ids','attention_mask'},'Global stream must contain text only')
        key=object_sha(['global',prompt]);ids=global_inputs['input_ids'][0].tolist()
        text=processor.apply_chat_template([dict(role='user',content=[dict(type='text',text=prompt)])],
                                          add_generation_prompt=True,tokenize=False)
        need(processor.tokenizer(text,add_special_tokens=False)['input_ids']==ids,'Global token template mismatch')
        base_layouts[key]=dict(input_ids=ids,prompt_tokens=len(ids),input_ids_sha256=object_sha(ids));base_text[key]=text
    for pair in source_data['pairs'].values():pair['layout_id']=local_keys[pair['question']]
    features,scenes,groups=inventory(rows,source_data,count_ids,processor.tokenizer.eos_token_id)
    counts={phase:len(groups[phase]) for phase in PHASES}
    need(counts['global_empty']==54 and counts['global_prefix']==54*9,'Balanced global prefix coverage differs')
    owner,fn,native=native_api(processor);layouts={}
    for feature in features.values():
        base=base_layouts[feature['base_layout_id']];prefix=feature['prefix_ids']
        key=object_sha([feature['base_layout_id'],prefix]);feature['layout_id']=key
        if key in layouts:continue
        ids=base['input_ids']+prefix
        if prefix:
            k=next(k for k in range(9) if count_ids[str(k)]==prefix[0])
            need(processor.tokenizer(base_text[feature['base_layout_id']]+str(k),add_special_tokens=False)['input_ids']==ids,
                 'Teacher-forced continuation retokenizes original prompt')
        value=dict(input_ids=ids,input_ids_sha256=object_sha(ids),prompt_tokens=len(ids),
                   original_prompt_tokens=len(base['input_ids']),prefix_ids=prefix)
        item=dict(input_ids=torch.tensor([ids]),attention_mask=torch.ones(1,len(ids),dtype=torch.long))
        if feature['kind']=='local':
            value['image_grid_thw']=base['image_grid_thw'];item['image_grid_thw']=torch.tensor(base['image_grid_thw'])
        positions,deltas=fn(owner,input_ids=item['input_ids'],image_grid_thw=item.get('image_grid_thw'),
                            attention_mask=item['attention_mask'])
        value.update(position_ids=positions.tolist(),position_ids_info=tensor_info(positions),rope_deltas=deltas.tolist())
        layouts[key]=value
    job=os.environ['SLURM_JOB_ID'];data=DATA/f'stage_{job}';data.mkdir(parents=True,exist_ok=False)
    out=OUT/f'stage_{job}';out.mkdir(parents=True,exist_ok=False);snapshot(out)
    pixels={};representative={};ancestry={}
    for pid,pair in sorted(source_data['pairs'].items()):representative.setdefault(pair['image_sha256'],pair)
    for i,(image_sha,pair) in enumerate(sorted(representative.items())):
        need(sha(pair['image_path'])==image_sha,'Original balanced training image bytes changed')
        inputs=prepared(processor,pair);expected=base_layouts[pair['layout_id']]
        need(inputs['input_ids'][0].tolist()==expected['input_ids']
             and inputs['image_grid_thw'].tolist()==expected['image_grid_thw'],'Balanced full processor layout differs')
        pixels[image_sha]=inputs['pixel_values'].contiguous().clone()
        previous=teacher['images'].get(image_sha)
        ancestry[image_sha]=previous is not None
        if previous is not None:
            need(list(inputs['pixel_values'].shape)==previous['pixel_shape']
                 and str(inputs['pixel_values'].dtype)==previous['pixel_dtype']
                 and tensor_info(inputs['pixel_values'])['sha256']==previous['pixel_sha256'],
                 'Known image no longer matches frozen teacher processed pixels')
        if i%128==0:print(json.dumps(dict(processed_images=i+1,total=len(representative))),flush=True)
    pixel_info={key:tensor_info(value) for key,value in pixels.items()}
    pixel_file=data/'pixels.pt';torch.save(dict(schema_version=1,pixels=pixels),pixel_file)
    profile=profile_ids(features,groups,labels,count_ids)
    for fid in profile['local_empty']+profile['local_prefix']:
        pair=source_data['pairs'][features[fid]['pair_id']];inputs=prepared(processor,pair)
        need(tensor_info(inputs['pixel_values'])==pixel_info[pair['image_sha256']]
             and inputs['input_ids'][0].tolist()==base_layouts[pair['layout_id']]['input_ids'],
             'Fixed profile actual processor differs')
    source_files=dict(training_manifest=dict(path=str(manifest_path),sha256=sha(manifest_path)),
                      training_schedule=dict(path=str(schedule_path),sha256=sha(schedule_path)),
                      teacher_cache=dict(path=binding['cache_file'],sha256=binding['cache_sha256']),
                      teacher_plan=dict(path=binding['plan_file'],sha256=binding['plan_sha256']),
                      teacher_audit=dict(path=binding['audit_file'],sha256=binding['audit_sha256']))
    plan=dict(schema_version=1,protocol='v7_parallel_local_training_features',data_root=str(DATA),
        source_sha256=frozen,source_files=source_files,teacher_binding=binding,model=model_metadata(),
        teacher_role='Frozen model/processor/pixel ancestor only; no old scene support or probability targets required',
        runtime=runtime_identity(),processor=teacher['processor'],image_processor_settings=teacher['image_processor_settings'],
        resize=392,quantization='nf4_double_bf16',native_dtype='torch.float16',attention='sdpa',
        global_prompt_template=global_prompt('{question}'),local_prompt='unchanged canonical build_count_prompt(question,1)',
        features=features,scenes=scenes,groups=groups,counts=counts,layouts=layouts,pairs=source_data['pairs'],
        training_scene_count=len(scenes),training_slots=len(slots),training_question_count=len(questions),
        pixels_file=str(pixel_file),pixels_sha256=sha(pixel_file),pixel_info=pixel_info,native_api=native,
        processed_pixel_teacher_ancestry=ancestry,count_token_ids=count_ids,eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
        shards=[sum((groups[phase][i::4] for phase in PHASES),[]) for i in range(4)],
        shard_rule='Within each fixed phase, sorted feature IDs modulo four; phases never share a batch',
        profile_groups=profile,profile_case_count=32,profile_calls=8,batch_size=64,
        profile_rule='Each phase: selected small batch, then repeat its frozen IDs cyclically to exactly64 rows',
        profile_selection='Local empty first3 sorted pairs/category; prefix desired0,1,8/category, lowest available fallback, no scores',
        profile_categories={fid:labels[features[fid]['pair_id']]['category']
                            for phase in ('local_empty','local_prefix') for fid in profile[phase]},
        numerical_gate=dict(full_vocabulary_tv_max=.02,top1_equal=True),
        projection_rule='load+pixels+1.5*sum(ceil(phase_shard_rows/64)*phase_stress_harvest_seconds)+30 <=300',
        scope='Balanced TRAINING features only; no test/dev harvesting or teacher probabilities as training targets',
        tests=tests,slurm_job_id=job)
    plan['inventory_sha256']=object_sha(dict(features=features,scenes=scenes,groups=groups))
    need(sources()==frozen and plan['model']==model_metadata(),'Source/model changed during CPU stage')
    path=data/'plan.json';save(path,plan);path.with_suffix('.sha256').write_text(sha(path)+'\n')
    verify_plan(path,pixels=False)
    summary=dict(passed=True,plan_file=str(path),plan_sha256=sha(path),counts=counts,total_features=len(features),
                 native_hidden_payload_bytes=len(features)*3584*2,pixel_blob_bytes=pixel_file.stat().st_size,
                 source_sha256=frozen,seconds=time.monotonic()-start,
                 shard_counts=[dict(Counter(features[fid]['phase'] for fid in shard)) for shard in plan['shards']])
    save(out/'summary.json',summary)
    index(out,'V7 balanced training-feature CPU inventory',[('Summary','summary.json'),('Source hashes','source_hashes.json')])
    index(data,'V7 frozen training feature inputs',[('Plan','plan.json'),('Processed image tensors','pixels.pt')])
    print(json.dumps(summary,indent=2),flush=True)



def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--training-manifest',type=Path,default=TRAIN/'main_manifest.json')
    parser.add_argument('--training-schedule',type=Path,default=TRAIN/'schedule.json')
    args=parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu'
         and not os.environ.get('SLURM_JOB_GPUS'),'Feature staging/tests require CPU Slurm')
    if args.self_test:
        import torch
        print(json.dumps(self_test(torch)));return
    stage(args)


if __name__=='__main__':main()


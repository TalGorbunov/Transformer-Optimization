"""Freeze TRAINING-ONLY V6 isolated-image teacher inputs on Slurm CPU.

No model weights are loaded. Each distinct rendered image is processed once;
native chat templates/token layouts are compiled per exact question and grid,
then checked against the ordinary full processor on all16 fixed profile pairs.
Frame semantics live in a separate audit file, never in the KD target cache.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
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
DATA=Path('/mnt/data/gabriele/gnn_transformer/v6_local_teacher')
TRAIN=DATA.parent/'v4_diversity'
OUTPUT=REPO/'outputs/native_aggregation_vlm/v6/teacher'
MODEL=Path('/mnt/ckpts/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5')
SEED=20260920
CATEGORIES=('positive','char_only','room_only','neither')
SOURCES=('scripts/stage_native_vision_v6_teacher.py','scripts/cache_native_vision_v6_teacher.py',
         'scripts/probe_native_vision_v2_prefix.py','gnnformer/data.py','gnnformer/runtime.py','gnnformer/constants.py',
         'slurm/native_vision_v6_teacher_stage.sbatch','slurm/native_vision_v6_teacher_profile.sbatch',
         'slurm/native_vision_v6_teacher_array.sbatch','slurm/native_vision_v6_teacher_merge.sbatch')


def need(value,message):
    if not value:raise ValueError(message)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def read(path):return json.loads(Path(path).read_text())


def save(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def source_hashes():return {name:sha(REPO/name) for name in SOURCES}


def snapshot(out):
    directory=out/'source';directory.mkdir()
    for name in SOURCES:(directory/name.replace('/','_')).write_bytes((REPO/name).read_bytes())
    save(out/'source_hashes.json',source_hashes())


def model_metadata():
    files={p.name:sha(p) for p in sorted(MODEL.iterdir()) if p.is_file() and
           p.suffix in ('.json','.txt','.md') and p.stat().st_size<30_000_000}
    need(all(k in files for k in ('config.json','generation_config.json','tokenizer_config.json',
                                 'preprocessor_config.json')),'Missing model metadata')
    index=read(MODEL/'model.safetensors.index.json');shards={}
    for name in sorted(set(index['weight_map'].values())):
        path=MODEL/name;stat=path.stat()
        shards[name]=dict(path=str(path.resolve()),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,inode=stat.st_ino)
    return dict(path=str(MODEL),metadata_sha256=files,weight_shard_stat=shards,
                weight_verification='Resolved immutable shard path/inode/size/mtime; no full weight rehash')


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def category(character,room,target_character,target_room):
    if character==target_character:return 'positive' if room==target_room else 'char_only'
    return 'room_only' if room==target_room else 'neither'


def prepared(processor,pair):
    from PIL import Image
    from gnnformer.data import build_count_prompt,build_prompt_inputs
    with Image.open(pair['image_path']) as original:
        rgb=original.convert('RGB')
        try:frame=rgb.resize((392,392))
        finally:rgb.close()
    try:return build_prompt_inputs(processor,[frame],build_count_prompt(pair['question'],1))
    finally:frame.close()


def actual_layout(inputs):
    return dict(input_ids=inputs['input_ids'][0].tolist(),
                input_ids_sha256=object_sha(inputs['input_ids'][0].tolist()),
                prompt_tokens=int(inputs['input_ids'].shape[1]),image_grid_thw=inputs['image_grid_thw'].tolist())


def verify_inputs(inputs,pair,plan):
    need(actual_layout(inputs)==plan['layouts'][pair['layout_id']],'Actual processor tokens/grid differ from CPU plan')
    image=plan['images'][pair['image_sha256']]
    need(list(inputs['pixel_values'].shape)==image['pixel_shape'] and
         str(inputs['pixel_values'].dtype)==image['pixel_dtype'] and
         tensor_sha(inputs['pixel_values'])==image['pixel_sha256'],'Actual processed pixels differ from CPU plan')


def verify_plan(path):
    path=Path(path);need(sha(path)==path.with_suffix('.sha256').read_text().strip(),'CPU plan sidecar differs')
    plan=read(path)
    need(plan['schema_version']==1 and plan['protocol']=='v6_training_only_local_count_teacher',
         'Wrong teacher plan schema/protocol')
    need(plan['source_sha256']==source_hashes() and plan['model']==model_metadata(),'Teacher source/model identity changed')
    for entry in plan['source_files'].values():need(sha(entry['path'])==entry['sha256'],'Training/audit input changed')
    need(sha(plan['audit_file'])==plan['audit_sha256'],'Training semantic audit changed')
    need(plan['resize']==392 and plan['quantization']=='nf4_double_bf16' and plan['attention']=='sdpa' and
         plan['temperature']==1 and plan['runtime']['bitsandbytes_version']==importlib.metadata.version('bitsandbytes'),
         'Teacher runtime/processing policy changed')
    need(len(plan['pairs'])==9980 and len(plan['scenes'])==1540 and len(plan['images'])==864 and
         len(plan['profile_pair_ids'])==16,'Teacher plan coverage differs')
    ids=sorted(plan['pairs']);need(plan['shards']==[ids[i::4] for i in range(4)],'Shard partition changed')
    need(all(len(s)==2495 for s in plan['shards']),'Expected four2495-pair shards')
    need(sum(len(scene['pair_ids']) for scene in plan['scenes'].values())==18800,'Training occurrence coverage changed')
    for scene in plan['scenes'].values():
        need(scene['split']=='train' and len(scene['pair_ids'])==scene['n_frames'] and
             all(pid in plan['pairs'] for pid in scene['pair_ids']),'Nontraining/incomplete scene mapping')
    return plan


def self_test():
    need([category('a','r',*target) for target in (('a','r'),('a','s'),('b','r'),('b','s'))]==
         list(CATEGORIES),'Category semantics differ')
    need(object_sha(['image','question'])==object_sha(['image','question']) and
         object_sha(['image','question'])!=object_sha(['image','other']),'Deduplication key differs')
    from scripts.cache_native_vision_v6_teacher import quality_self_test
    quality_self_test()
    return dict(passed=True,scope='category/deduplication and teacher-probability/weighted-quality boundaries')


def stage():
    import torch
    import transformers
    from transformers import AutoProcessor
    from scripts.probe_native_vision_v2_prefix import fingerprint
    from gnnformer.data import build_count_prompt
    torch.set_num_threads(4)
    started=time.monotonic();sources=source_hashes();tests=self_test()
    manifest_path=TRAIN/'main_manifest.json';schedule_path=TRAIN/'schedule.json'
    manifest=read(manifest_path);schedule=read(schedule_path)
    need(manifest['schema_version']==1 and Path(manifest['dataset_root'])==TRAIN,'Wrong source manifest')
    need(schedule['manifest_sha256']==sha(manifest_path) and schedule['purpose']=='main','Wrong source schedule')
    rows=manifest['splits']['train_N8']['samples']+manifest['splits']['train_N16']['samples']
    need(len(rows)==1540 and Counter(r['n_frames'] for r in rows)==Counter({8:730,16:810}),'Wrong training cells')
    need(len({r['sid'] for r in rows})==1540,'Duplicate training SIDs')
    refresh=schedule['conditions']['refresh'];need(len(refresh)==9 and all(len(b)==180 for b in refresh),
                                                'Wrong refresh schedule dimensions')
    frequencies=Counter(sid for block in refresh for sid in block)
    need(set(frequencies)=={r['sid'] for r in rows},'Refresh schedule training coverage differs')
    need(sum(r['n_frames'] for r in rows)==18800 and
         sum(r['n_frames']*frequencies[r['sid']] for r in rows)==19440,'Frame weights differ')
    pairs={};images={};scenes={};labels={};image_hash_cache={}
    for row in rows:
        directory=Path(row['path'])
        need(row['split']=='train' and directory==TRAIN/'mmred_vfiltered'/f"seq_len_{row['n_frames']}"/'train'/row['sid'],
             'Teacher must use canonical training inputs only')
        qa=directory/'qa.txt';need(sha(qa)==row['qa_sha256'],'Training QA hash changed')
        lines=qa.read_text().splitlines();block=[x.strip() for x in lines[lines.index('question:')+1:lines.index('answer:')] if x.strip()]
        states=[ast.literal_eval(x) for x in block if x.startswith('{')]
        question=[x for x in block if not x.startswith('{')]
        need(question==[row['question']] and len(states)==row['n_frames']==len(row['image_files']),
             'Training question/frame coverage differs')
        need(object_sha(dict(states=states,question=row['question']))==row['content_sha256'],'Training content hash differs')
        scene_pairs=[];positive=0
        for index,(state,image) in enumerate(zip(states,row['image_files'])):
            need(state['step_id']==index+1 and len(state['rooms'])==1,'Wrong rendered Step/room semantics')
            room,characters=next(iter(state['rooms'].items()));need(len(characters)==1,'Expected one character')
            cat=category(characters[0],room,row['target_character'],row['target_room']);positive+=cat=='positive'
            image_path=Path(image['path']);stat=image_path.stat()
            need(image_path==directory/f'{index:03d}.png' and stat.st_size==image['bytes'],'Image order/bytes differ')
            statkey=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
            if statkey not in image_hash_cache:image_hash_cache[statkey]=sha(image_path)
            need(image_hash_cache[statkey]==image['sha256'],'Training image hash changed')
            pid=object_sha([image['sha256'],row['question']])
            if pid not in pairs:
                pairs[pid]=dict(pair_id=pid,image_path=str(image_path),image_sha256=image['sha256'],
                               image_bytes=image['bytes'],question=row['question'])
                labels[pid]=dict(category=cat,gold=int(cat=='positive'),unique_scene_occurrences=0,scheduled_occurrences=0)
            need(labels[pid]['category']==cat,'Same rendered image/question has conflicting audit semantics')
            labels[pid]['unique_scene_occurrences']+=1;labels[pid]['scheduled_occurrences']+=frequencies[row['sid']]
            images.setdefault(image['sha256'],dict(path=str(image_path),bytes=image['bytes']))
            scene_pairs.append(pid)
        need(positive==row['gold']==int(lines[lines.index('answer:')+1]),'Independent training gold recount differs')
        scenes[row['sid']]=dict(path=str(directory),qa_sha256=row['qa_sha256'],question=row['question'],
                               split='train',n_frames=row['n_frames'],pair_ids=scene_pairs,
                               schedule_occurrences=frequencies[row['sid']])
    need(len(pairs)==9980 and len(images)==864 and len({r['question'] for r in rows})==52,'Registered dedup counts differ')
    by_cat={cat:sorted(pid for pid in pairs if labels[pid]['category']==cat) for cat in CATEGORIES}
    profile=[by_cat[cat][i] for i in range(4) for cat in CATEGORIES]
    processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,use_fast=False)
    tokenizer=processor.tokenizer;processor_fp=fingerprint(processor,str(transformers.__version__))
    image_token=getattr(processor,'image_token','<|image_pad|>')
    merge=int(processor.image_processor.merge_size)
    representative={}
    for pid in sorted(pairs):representative.setdefault(pairs[pid]['image_sha256'],pairs[pid])
    for number,(image_sha,image) in enumerate(sorted(images.items())):
        inputs=prepared(processor,representative[image_sha])
        image.update(pixel_shape=list(inputs['pixel_values'].shape),pixel_dtype=str(inputs['pixel_values'].dtype),
                     pixel_sha256=tensor_sha(inputs['pixel_values']),image_grid_thw=inputs['image_grid_thw'].tolist())
        if number%128==0:print(f'CPU image processing {number+1}/864',flush=True)
    layouts={};prompts={};continuations={}
    for pid,pair in sorted(pairs.items()):
        grid=images[pair['image_sha256']]['image_grid_thw'];key=object_sha([pair['question'],grid])
        if key not in layouts:
            prompt=build_count_prompt(pair['question'],1)
            messages=[dict(role='user',content=[dict(type='image'),dict(type='text',text=prompt)])]
            text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            need(text.count(image_token)==1 and len(grid)==1,'Expected one image placeholder/grid')
            tokens=math.prod(grid[0])//(merge*merge)
            expanded=text.replace(image_token,image_token*tokens)
            ids=tokenizer(expanded,add_special_tokens=False)['input_ids']
            choices={}
            for answer in ('0','1'):
                combined=tokenizer(expanded+answer,add_special_tokens=False)['input_ids']
                need(combined[:len(ids)]==ids and len(combined)==len(ids)+1,'0/1 not single-token continuations')
                choices[answer]=combined[-1]
            need(choices['0']!=choices['1'],'0/1 token IDs collide')
            layouts[key]=dict(input_ids=ids,input_ids_sha256=object_sha(ids),prompt_tokens=len(ids),image_grid_thw=grid)
            prompts[key]=dict(question=pair['question'],count_prompt=prompt,chat_template_text=text,
                              expanded_tokenizer_text=expanded)
            continuations[key]=choices
        pair['layout_id']=key
    need(len({object_sha(x) for x in continuations.values()})==1,'Count-token boundary differs across prompts')
    model=model_metadata()
    reference_analysis=REPO/'outputs/native_aggregation_vlm/v5/analysis.json'
    reference_config=Path(read(reference_analysis)['runs']['sum']['4'])/'config.json';reference=read(reference_config)
    settings={name:getattr(processor.image_processor,name,None) for name in reference['image_processor_settings']}
    need(settings==reference['image_processor_settings'] and str(torch.__version__)==reference['torch_version'] and
         str(transformers.__version__)==reference['transformers_version'],'Teacher runtime differs from V5')
    need(all(reference['code_sha256'][name]==sources[name] for name in ('gnnformer/runtime.py','gnnformer/data.py')),
         'Native runtime/prompt source differs from V5')
    job=os.environ['SLURM_JOB_ID'];data_out=DATA/f'stage_{job}';data_out.mkdir(parents=True,exist_ok=False)
    out=OUTPUT/f'stage_{job}';out.mkdir(parents=True,exist_ok=False);snapshot(out)
    audit=dict(schema_version=1,labels=labels,scope='TRAINING ONLY; audit/profile coverage, never teacher targets',
               unique_training_scenes=1540,unique_scene_frame_occurrences=18800,scheduled_frame_occurrences=19440,
               unique_image_question_pairs=9980,distinct_images=864,distinct_questions=52,
               profile_category_counts=dict(Counter(labels[p]['category'] for p in profile)),
               all_qa_content_gold_and_image_hashes_verified=True)
    save(data_out/'audit_labels.json',audit)
    files={name:dict(path=str(path),sha256=sha(path)) for name,path in
           [('training_manifest',manifest_path),('training_schedule',schedule_path),
            ('v5_reference_analysis',reference_analysis),('v5_reference_config',reference_config)]}
    ids=sorted(pairs)
    plan=dict(schema_version=1,protocol='v6_training_only_local_count_teacher',data_root=str(DATA),
              source_sha256=sources,source_files=files,model=model,
              runtime=dict(torch_version=str(torch.__version__),transformers_version=str(transformers.__version__),
                           bitsandbytes_version=importlib.metadata.version('bitsandbytes')),
              processor=processor_fp,image_processor_settings=settings,resize=392,quantization='nf4_double_bf16',
              attention='sdpa',temperature=1,seed=SEED,pairs=pairs,scenes=scenes,images=images,
              layouts=layouts,prompts=prompts,continuations=continuations,
              count_token_ids=next(iter(continuations.values())),profile_pair_ids=profile,
              shards=[ids[i::4] for i in range(4)],shard_rule='lexicographically sorted pair IDs, index modulo4',
              audit_file=str(data_out/'audit_labels.json'),audit_sha256=sha(data_out/'audit_labels.json'),
              quality_thresholds=dict(conditional_balanced_accuracy=.95,each_category_accuracy=.90,mean_numeric_mass=.90),
              quality_weighting='18800 original frame occurrences in1540 distinct training scenes',
              profile_cost_rule='load + first4 pair wall seconds + 1.25*2495*max(last12 pair wall seconds) +30 <=900',
              test_results=tests,slurm_job_id=job)
    for pid in profile:verify_inputs(prepared(processor,pairs[pid]),pairs[pid],plan)
    plan['full_processor_parity_checks']=dict(passed=True,pairs=profile,count=16)
    need(source_hashes()==sources and model_metadata()==model,'Source/model changed during stage')
    path=data_out/'plan.json';save(path,plan);path.with_suffix('.sha256').write_text(sha(path)+'\n')
    verify_plan(path)  # Saved JSON round-trip is part of the CPU gate.
    save(out/'summary.json',dict(passed=True,plan=str(path),plan_sha256=sha(path),data_output=str(data_out),
                                source_sha256=sources,seconds=time.monotonic()-started))
    index='# V6 training-only teacher stage\n\n- [Summary](summary.json)\n- [Source snapshots](source/)\n'
    (out/'INDEX.md').write_text(index)
    (data_out/'INDEX.md').write_text('# V6 teacher inputs\n\n- [Frozen plan](plan.json)\n- [Audit-only local semantics](audit_labels.json)\n')
    print(json.dumps(dict(output=str(out),plan=str(path),sha256=sha(path),seconds=time.monotonic()-started)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args();need(bool(os.environ.get('SLURM_JOB_ID')),'Use Slurm before image/tokenizer/model/CPU work')
    if args.self_test:print(json.dumps(self_test()));return
    stage()


if __name__=='__main__':main()

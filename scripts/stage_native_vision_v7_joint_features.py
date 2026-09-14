"""CPU inventory for the ordinary joint-image V7 control, training only.

The same N-agnostic set prompt is used after all N images. Capture will use the
ordinary joint pre-final-norm state as BOTH the one local item and global input
to the matched aggregation core. This file only stages features and software
profile inputs; full joint harvesting/training are not enabled by this program.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import stage_native_vision_v7_features as parallel
from gnnformer.parallel_local_prompts import build_set_count_prompt
need,read,save,sha,object_sha=parallel.need,parallel.read,parallel.save,parallel.sha,parallel.object_sha
tensor_info,native_api=parallel.tensor_info,parallel.native_api
DATA=Path('/mnt/data/gabriele/gnn_transformer/v7_joint_features')
OUT=REPO/'outputs/native_aggregation_vlm/v7/joint_features'
GROUPS=('N8_empty','N8_prefix','N16_empty','N16_prefix')
COUNTS={'N8_empty':918,'N8_prefix':918,'N16_empty':972,'N16_prefix':972}
OWN=('scripts/stage_native_vision_v7_joint_features.py','scripts/cache_native_vision_v7_joint_features.py',
     'slurm/native_vision_v7_joint_features_stage.sbatch','slurm/native_vision_v7_joint_features_profile.sbatch',
     'scripts/stage_native_vision_v7_features.py','scripts/cache_native_vision_v7_features.py',
     'gnnformer/parallel_local_prompts.py','gnnformer/data.py','gnnformer/runtime.py')


def sources():return {name:sha(REPO/name) for name in OWN}


def snapshot(out):
    (out/'source').mkdir()
    for name in OWN:(out/'source'/name.replace('/','_')).write_bytes((REPO/name).read_bytes())
    save(out/'source_hashes.json',sources())


def ancestors(manifest_file):
    manifest=read(manifest_file);bindings={}
    for field in ('audit','stage_plan','prior_inventory'):
        path=Path(manifest[field+'_file']);digest=manifest[field+'_sha256']
        need(path.is_relative_to(DATA.parent/'v7_balanced') and sha(path)==digest,
             'Balanced semantic/staging/prior artifact changed')
        bindings[field]=dict(path=str(path),sha256=digest)
    audit=read(bindings['audit']['path'])
    for key in ('passed','all_current_contexts_disjoint','all_qa_counts_categories_images_verified',
                'global_prompt_has_no_set_size_argument','non_saturated_contexts_fresh_against_18_prior_manifests',
                'question_count_support_uniform_in_slots'):
        need(audit[key] is True,'Balanced audit flag failed: '+key)
    need(audit['train_contexts']==1890 and audit['epoch_slots']==1944 and audit['dev_contexts']==72
         and audit['saturated_training_contexts']==54 and len(audit['historical_saturated_reuse'])==39,
         'Balanced finite-support exception or cardinalities differ')
    stage=read(bindings['stage_plan']['path']);prior=read(bindings['prior_inventory']['path'])
    need(audit['stage_plan_sha256']==bindings['stage_plan']['sha256']
         and stage['purpose']=='v7_balanced_training_and_dev'
         and prior['exclusions_are_complete_context_and_question_hashes'] is True,
         'Balanced source ancestry is inconsistent')
    for path,digest in prior['source_manifest_sha256'].items():
        need(sha(path)==digest,'A prior exclusion manifest changed')
    return bindings


def pack(torch,rows,pad):
    need(rows,'Empty joint batch');width=max(r['input_ids'].shape[1] for r in rows)
    ids=torch.full((len(rows),width),pad,dtype=torch.long);mask=torch.zeros_like(ids)
    for i,row in enumerate(rows):
        need(set(row)=={'input_ids','attention_mask','pixel_values','image_grid_thw'}
             and row['input_ids'].shape[0]==1 and bool((row['attention_mask']==1).all()),'Malformed complete joint row')
        n=row['input_ids'].shape[1];ids[i,-n:]=row['input_ids'][0];mask[i,-n:]=1
    return dict(input_ids=ids,attention_mask=mask,pixel_values=torch.cat([r['pixel_values'] for r in rows]),
                image_grid_thw=torch.cat([r['image_grid_thw'] for r in rows]))


def row_inputs(torch,plan,pixels,fid):
    f=plan['features'][fid];scene=plan['scenes'][f['sid']];layout=plan['layouts'][f['layout_id']]
    ids=torch.tensor([layout['input_ids']],dtype=torch.long)
    return dict(input_ids=ids,attention_mask=torch.ones_like(ids),
                pixel_values=torch.cat([pixels[key] for key in scene['image_sha256']]),
                image_grid_thw=torch.tensor(layout['image_grid_thw'],dtype=torch.long))


def verify_plan(path,*,pixels=True):
    path=Path(path)
    need(path.is_relative_to(DATA) and sha(path)==path.with_suffix('.sha256').read_text().strip(),'Joint plan path/sidecar differs')
    plan=read(path)
    need(plan['schema_version']==1 and plan['protocol']=='v7_joint_training_feature_profile'
         and plan['source_sha256']==sources(),'Joint source identity differs')
    parent=parallel.verify_plan(plan['parallel_plan_file'],pixels=pixels)
    need(sha(plan['parallel_plan_file'])==plan['parallel_plan_sha256']
         and plan['model']==parent['model'] and plan['runtime']==parent['runtime']
         and plan['processor']==parent['processor'] and plan['pixels_file']==parent['pixels_file']
         and plan['pixels_sha256']==parent['pixels_sha256'],'Parallel ancestor changed')
    need(ancestors(plan['manifest_file'])==plan['balanced_ancestors']
         and sha(plan['manifest_file'])==plan['manifest_sha256'],'Balanced joint source changed')
    need(plan['counts']==COUNTS and len(plan['features'])==3780 and len(plan['scenes'])==1890
         and plan['global_prompt_template']==build_set_count_prompt('{question}'),'Joint inventory counts/prompt differ')
    need(plan['inventory_sha256']==object_sha(dict(features=plan['features'],scenes=plan['scenes'],groups=plan['groups'])),
         'Joint feature inventory changed')
    expected=[sum((plan['groups'][group][i::4] for group in GROUPS),[]) for i in range(4)]
    need(plan['prospective_shards']==expected and sorted(fid for shard in expected for fid in shard)==sorted(plan['features']),
         'Prospective joint partition differs')
    return plan,parent


def self_test(torch):
    a=dict(input_ids=torch.tensor([[1,2]]),attention_mask=torch.ones(1,2,dtype=torch.long),
           pixel_values=torch.tensor([[10.],[11.]]),image_grid_thw=torch.tensor([[1,1,1],[1,1,1]]))
    b=dict(input_ids=torch.tensor([[3]]),attention_mask=torch.ones(1,1,dtype=torch.long),
           pixel_values=torch.tensor([[20.]]),image_grid_thw=torch.tensor([[1,1,1]]))
    result=pack(torch,[a,b],0)
    need(result['input_ids'].tolist()==[[1,2],[0,3]] and result['pixel_values'].flatten().tolist()==[10.,11.,20.]
         and result['image_grid_thw'].shape==(3,3),'Joint row-major image packing differs')
    need(object_sha(['joint','scene',[]])!=object_sha(['joint','scene',[5]]),'Empty/continuation identity collided')
    return dict(passed=True,tests=['row_major_multi_image_packing','left_padding','empty_prefix_separation'])


def full_processor(processor,row):
    from PIL import Image
    from gnnformer.data import build_prompt_inputs
    frames=[]
    try:
        for image in row['image_files']:
            with Image.open(image['path']) as source:
                rgb=source.convert('RGB')
                try:frame=rgb.resize((392,392))
                finally:rgb.close()
            frames.append(frame)
        return build_prompt_inputs(processor,frames,build_set_count_prompt(row['question']))
    finally:
        for frame in frames:frame.close()


def stage(args):
    import torch,transformers
    from transformers import AutoProcessor
    from scripts.probe_native_vision_v2_prefix import fingerprint
    torch.set_num_threads(4);start=time.monotonic();frozen=sources();tests=self_test(torch)
    parent=parallel.verify_plan(args.parallel_plan,pixels=True)
    manifest_file=Path(parent['source_files']['training_manifest']['path']);manifest=read(manifest_file)
    bindings=ancestors(manifest_file)
    rows=manifest['splits']['train_N8']['samples']+manifest['splits']['train_N16']['samples']
    by_sid={row['sid']:row for row in rows}
    need(set(by_sid)==set(parent['scenes']) and len(by_sid)==1890,'Joint/parallel training scenes differ')
    pixels=torch.load(parent['pixels_file'],map_location='cpu',weights_only=True)['pixels']
    for key,value in pixels.items():need(tensor_info(value)==parent['pixel_info'][key],'Parent pixel tensor changed')
    processor=AutoProcessor.from_pretrained(str(parallel.MODEL),trust_remote_code=True,use_fast=False)
    need(fingerprint(processor,str(transformers.__version__))==parent['processor'],'Processor differs')
    owner,fn,native=native_api(processor);need(native==parent['native_api'],'Native implementation differs')
    features={};scenes={};layouts={};grid_by_image={}
    for f in parent['features'].values():
        if f['phase']=='local_empty':
            pair=parent['pairs'][f['pair_id']]
            grid_by_image[pair['image_sha256']]=parent['layouts'][f['layout_id']]['image_grid_thw'][0]
    image_token=getattr(processor,'image_token','<|image_pad|>');merge=int(processor.image_processor.merge_size)
    for row in rows:
        source=parent['scenes'][row['sid']]
        need(all(source[key]==row[key] for key in ('question','gold','n_frames','qa_sha256','content_sha256')),
             'Joint scene target/provenance differs')
        image_shas=[image['sha256'] for image in row['image_files']]
        expected=[parent['pairs'][parent['features'][ids[0]]['pair_id']]['image_sha256'] for ids in source['local_feature_ids']]
        need(image_shas==expected,'Joint image order differs from parallel local evidence')
        grid=[grid_by_image[key] for key in image_shas];scene_ids=[]
        for prefix in ([],source['target_ids'][:-1]):
            need(len(prefix)<=1,'Expected registered single-count-token training prefix')
            fid=object_sha(['joint',row['content_sha256'],prefix])
            lid=object_sha([row['question'],grid,prefix])
            if lid not in layouts:
                messages=[dict(role='user',content=[dict(type='image') for _ in image_shas]+
                               [dict(type='text',text=build_set_count_prompt(row['question']))])]
                text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
                pieces=text.split(image_token);need(len(pieces)==len(grid)+1,'Joint placeholder count differs')
                expanded=pieces[0]+''.join(image_token*(math.prod(g)//(merge*merge))+tail for g,tail in zip(grid,pieces[1:]))
                base_ids=processor.tokenizer(expanded,add_special_tokens=False)['input_ids'];ids=base_ids+prefix
                if prefix:
                    need(processor.tokenizer(expanded+str(row['gold']),add_special_tokens=False)['input_ids']==ids,
                         'Joint gold continuation retokenizes its prompt')
                inputs=dict(input_ids=torch.tensor([ids]),attention_mask=torch.ones(1,len(ids),dtype=torch.long),
                            image_grid_thw=torch.tensor(grid,dtype=torch.long))
                pos,deltas=fn(owner,**inputs)
                layouts[lid]=dict(input_ids=ids,input_ids_sha256=object_sha(ids),prompt_tokens=len(ids),
                                  original_prompt_tokens=len(base_ids),prefix_ids=prefix,image_grid_thw=grid,
                                  position_ids_info=tensor_info(pos),last_position_ids=pos[:,:,-1].tolist(),
                                  rope_deltas=deltas.tolist())
            phase=f'N{row["n_frames"]}_'+('prefix' if prefix else 'empty')
            features[fid]=dict(feature_id=fid,sid=row['sid'],layout_id=lid,prefix_ids=prefix,phase=phase)
            scene_ids.append(fid)
        scenes[row['sid']]=dict(sid=row['sid'],path=row['path'],n_frames=row['n_frames'],gold=row['gold'],
                               question=row['question'],qa_sha256=row['qa_sha256'],content_sha256=row['content_sha256'],
                               target_ids=source['target_ids'],feature_ids=scene_ids,image_sha256=image_shas,split='train')
    groups={group:sorted(fid for fid,f in features.items() if f['phase']==group) for group in GROUPS}
    need({g:len(v) for g,v in groups.items()}==COUNTS and len(features)==3780,'Joint inventory coverage differs')
    profile_sids={n:[sorted(r['sid'] for r in rows if r['n_frames']==n and r['gold']==k)[0] for k in (0,1,4,8)] for n in (8,16)}
    profile_groups={f'N{n}_{kind}':[scenes[sid]['feature_ids'][index] for sid in profile_sids[n]]
                    for n in (8,16) for index,kind in enumerate(('empty','prefix'))}
    partial=dict(features=features,scenes=scenes,layouts=layouts)
    for selected in profile_sids.values():
        for sid in selected:
            actual=full_processor(processor,by_sid[sid]);compiled=row_inputs(torch,partial,pixels,scenes[sid]['feature_ids'][0])
            need(set(actual)==set(compiled) and all(torch.equal(actual[key],compiled[key]) for key in actual),
                 'Actual full joint processor differs from compiled shared pixels/layout')
    job=os.environ['SLURM_JOB_ID'];data=DATA/f'stage_{job}';data.mkdir(parents=True,exist_ok=False)
    out=OUT/f'stage_{job}';out.mkdir(parents=True,exist_ok=False);snapshot(out)
    plan=dict(schema_version=1,protocol='v7_joint_training_feature_profile',source_sha256=frozen,
              parallel_plan_file=str(Path(args.parallel_plan).resolve()),parallel_plan_sha256=sha(args.parallel_plan),
              manifest_file=str(manifest_file),manifest_sha256=sha(manifest_file),balanced_ancestors=bindings,
              model=parent['model'],runtime=parent['runtime'],processor=parent['processor'],native_api=native,
              pixels_file=parent['pixels_file'],pixels_sha256=parent['pixels_sha256'],pixel_info=parent['pixel_info'],
              global_prompt_template=build_set_count_prompt('{question}'),features=features,scenes=scenes,layouts=layouts,
              counts=COUNTS,groups=groups,profile_groups=profile_groups,profile_sids=profile_sids,
              profile_calls=8,profile_distinct_feature_rows=16,profile_total_output_rows=20,
              prospective_shards=[sum((groups[g][i::4] for g in GROUPS),[]) for i in range(4)],
              prospective_batch_size=4,native_dtype='torch.float16',pad_token_id=parent['pad_token_id'],
              numerical_gate=dict(full_vocabulary_tv_max=.02,top1_equal=True),
              full_harvest_authorized=False,tests=tests,slurm_job_id=job)
    plan['inventory_sha256']=object_sha(dict(features=features,scenes=scenes,groups=groups))
    path=data/'plan.json';save(path,plan);path.with_suffix('.sha256').write_text(sha(path)+'\n')
    need(sources()==frozen,'Source changed during joint stage');verify_plan(path,pixels=False)
    summary=dict(passed=True,plan_file=str(path),plan_sha256=sha(path),counts=COUNTS,total_features=3780,
                 native_hidden_payload_bytes=3780*3584*2,profile_calls=8,profile_distinct_feature_rows=16,
                 profile_total_output_rows=20,source_sha256=frozen,seconds=time.monotonic()-start)
    save(out/'summary.json',summary)
    parallel.index(out,'V7 joint feature CPU stage',[('Summary','summary.json'),('Source hashes','source_hashes.json')])
    parallel.index(data,'V7 joint training feature inventory',[('Frozen plan','plan.json')])
    print(json.dumps(summary,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--parallel-plan',type=Path,required=True)
    args=parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu'
         and not os.environ.get('SLURM_JOB_GPUS'),'Joint inventory requires CPU Slurm')
    stage(args)


if __name__=='__main__':main()


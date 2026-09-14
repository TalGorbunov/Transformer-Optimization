"""Prepare original MMReD native inputs and eight training-only software cases.

No model forward or fitting. All processor/tensor/image work is CPU Slurm-only.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts.stage_mmred_official_recovery import need,sha,object_sha,save
OUT=REPO/'outputs/native_aggregation_vlm/mmred_official_native'
DATA=Path('/mnt/data/gabriele/gnn_transformer/mmred_official_native')
PROTOCOL='mmred_official_native_training_profile_preparation'
MODEL=Path('/mnt/ckpts/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5')
RECOVERY=REPO/'outputs/native_aggregation_vlm/mmred_official_recovery/recovery_443939/plan.json'
RECOVERY_SHA='16abc5ef843dab2181a9dea69105b089be81dec26236ec4a744a91d008e107de'
IDENTITY_PLAN=REPO/'outputs/native_aggregation_vlm/identity_join_joint_lora_v2/check_443888/plan.json'
IDENTITY_SHA='f2314c7c689fb489c1efea43c7da625cf53a2822ae1d8467e5208c784010f430'
QTYPES=('char_at_frame','steps_in_room','spend_together','where_spend')
ROOMS=('Kitchen','Bathroom','Garden','Office','Bedroom','Hallway')
PEOPLE=('Sandra','Mary','John','Daniel','Michael','Nobody')
PROPOSAL='docs/paper/NATIVE_AGGREGATION_MMRED_OFFICIAL_NATIVE_PREPARATION.md'
OWN=('scripts/prepare_mmred_official_native.py','scripts/native_visual_memory_runtime.py',PROPOSAL,'slurm/mmred_official_native_prepare.sbatch')
POLICY=dict(protocol=PROTOCOL,cpu_seconds=600,cpu_cores=4,memory_gib=16,rows=5000,training_rows=4000,validation_rows=400,
    primary_test_rows=600,profile_cases=8,resize=392,raw_width=512,raw_height=512,maximum_new_tokens=50,
    target_eos=151645,native_eos=[151645,151643],native_generation_unrestricted=True,original_question_unprefixed=True,
    training_only_profile=True,no_model_forward=True,no_training_release=True)


def read(path):return json.loads(Path(path).read_text())


def sources():return {name:sha(REPO/name) for name in OWN}


def inherited_sources():
    result={}
    for name,digest in (
        ('identity_join_joint_lora_v2','a828f0d71ebbce250923840efd481aca321d2096eba5a1720a2f94b5cb0f5cd4'),
        ('mmred_official_render','1db1855d0ff84b4c4ec7a0ccb4b4288d1f0660447d488e9cde6249b27a7ca587')):
        file=REPO/'outputs/native_aggregation_vlm'/name/'source_release.json'
        need(sha(file)==digest,'Frozen source release changed');release=read(file)
        for path,value in {**release['source_sha256'],**release['inherited_source_sha256']}.items():
            need(path not in result or result[path]==value,'Source closure conflict');result[path]=value
    result['scripts/native_joint_prefix_cache.py']='f0844da3051ff48bd1c23601e1050077b8b5f0da67ead59da90aef8390ef684f'
    need(all(sha(REPO/name)==digest for name,digest in result.items()),'Frozen imported source changed')
    return result


def target_value(atype,answer):
    if atype=='number':
        need(type(answer) in (int,str) and str(answer).isdigit(),'Original nonnegative integer answer required')
        return int(answer)
    vocabulary=ROOMS if atype=='room' else PEOPLE if atype=='person' else ()
    need(type(answer) is str and answer in vocabulary,'Original typed room/person answer required')
    return answer


def canonical_target(atype,answer):return json.dumps({'answer':target_value(atype,answer)},ensure_ascii=False)


def parse_generated(text,atype,generated_ids,gold):
    """Strict typed whole JSON, with duplicate keys/booleans/extra prose rejected."""
    def unique(pairs):
        obj={}
        for k,v in pairs:
            if k in obj:raise ValueError('Duplicate JSON key')
            obj[k]=v
        return obj
    value=None;valid=False
    try:
        obj=json.loads(text,object_pairs_hook=unique,parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonstandard JSON constant')))
        if type(obj) is dict and set(obj)=={'answer'}:
            value=obj['answer']
            valid=(type(value) is int and value>=0) if atype=='number' else (
                type(value) is str and value in (ROOMS if atype=='room' else PEOPLE if atype=='person' else ()))
    except (ValueError,TypeError):pass
    complete=bool(generated_ids) and generated_ids[-1] in POLICY['native_eos']
    semantic=valid and value==target_value(atype,gold)
    return dict(format_valid=valid,parsed_value=value,semantic_correct=semantic,completed=complete,
        correct=semantic and complete,truncated=bool(generated_ids) and not complete and len(generated_ids)==50)


def parser_fixtures():
    good=[('{"answer": 3}','number',3),('{"answer": "Bathroom"}','room','Bathroom'),('{"answer": "Sandra"}','person','Sandra')]
    for text,atype,gold in good:
        need(parse_generated(text,atype,[151645],gold)['correct'],'Strict parser good fixture failed')
        need(not parse_generated(text,atype,[1]*50,gold)['correct'],'Missing EOS accepted')
    for text,atype,gold in [('{"answer": true}','number',1),('{"answer": "3"}','number',3),('{"answer": 3.0}','number',3),
            ('{"answer": 2, "answer": 3}','number',3),('{"answer":3,"extra":1}','number',3),
            ('Answer: {"answer":3}','number',3),('{"answer":"kitchen"}','room','Kitchen'),('{"answer":3} {}','number',3),('{"answer":NaN}','number',3),('{"answer":Infinity}','number',3)]:
        need(not parse_generated(text,atype,[151645],gold)['correct'],'Strict parser invalid fixture accepted')
    return dict(passed=True,good=3,invalid=10,eos_checks=3)


def select_cases(rows):
    train=[r for r in rows if r['pilot_role']=='train'];need(len(train)==4000,'Original4000 training pairs required')
    spec=[(n,'char_at_frame') for n in (1,2,4,8,16)]+[(16,q) for q in QTYPES if q!='char_at_frame']
    selected=[next(r for r in train if r['n']==n and r['qtype']==q) for n,q in spec]
    need(len({r['sid'] for r in selected})==8,'Eight distinct training-only cases required')
    return selected


def prepare_joint(processor,row,system_prompt,rope):
    import torch
    from PIL import Image
    from scripts.native_visual_memory_runtime import prepare_text
    frames=[]
    try:
        for item in row['image_files']:
            need(sha(item['path'])==item['sha256'],'Rendered original image changed')
            with Image.open(item['path']) as raw:
                need(raw.format=='PNG' and raw.size==(512,512),'Original512x512 PNG required')
                frames.append(raw.convert('RGB').resize((392,392)))
        need(len(frames)==row['n'],'Original ordered image count differs')
        conversation=[dict(role='system',content=[dict(type='text',text=system_prompt)]),
            dict(role='user',content=[dict(type='image',image=f) for f in frames]+[dict(type='text',text=row['question'])])]
        inputs=dict(processor.apply_chat_template(conversation,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt'))
        need(set(inputs)=={'input_ids','attention_mask','pixel_values','image_grid_thw'} and inputs['input_ids'].shape[0]==1
             and bool((inputs['attention_mask']==1).all()) and inputs['image_grid_thw'].tolist()==[[1,28,28]]*row['n'],
             'Expected single unpadded392px native images and14x14 merged grids')
        target=canonical_target(row['atype'],row['answer'])
        text=prepare_text(processor,system_prompt,row['question'],target_text=target)
        ids=text['target_ids'];need(0<len(ids)<=50 and ids[-1]==151645,'Complete JSON+EOS target must fit50')
        full=dict(processor.apply_chat_template(conversation+[dict(role='assistant',content=target)],
            add_generation_prompt=False,tokenize=True,return_dict=True,return_tensors='pt'))
        W=inputs['input_ids'].shape[1]
        need(torch.equal(full['input_ids'][:,:W],inputs['input_ids'])
             and full['input_ids'][0,W:W+len(ids)].tolist()==ids,'Full teacher chat must preserve inference prefix and JSON+EOS target')
        positions,delta=rope(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'],image_grid_thw=inputs['image_grid_thw'])
        positions=torch.cat((torch.arange(W).view(1,1,W),positions),dim=0)
        teacher_ids=torch.cat((inputs['input_ids'],torch.tensor([ids[:-1]],dtype=torch.long)),dim=1);S=teacher_ids.shape[1]
        teacher_pos,teacher_delta=rope(input_ids=teacher_ids,attention_mask=torch.ones_like(teacher_ids),image_grid_thw=inputs['image_grid_thw'])
        teacher_pos=torch.cat((torch.arange(S).view(1,1,S),teacher_pos),dim=0)
        need(torch.equal(teacher_pos[:,:,:W],positions) and torch.equal(teacher_delta,delta),'Teacher continuation changed original image positions')
        coords=dict(frame_index=torch.arange(row['n']).repeat_interleave(196),
            raster_row=torch.arange(14).repeat_interleave(14).repeat(row['n']),
            raster_col=torch.arange(14).repeat(row['n']*14),grid_height=torch.full((row['n']*196,),14),grid_width=torch.full((row['n']*196,),14))
        return dict(inputs=inputs,position_ids=positions,rope_deltas=delta,text=text,target_ids=ids,target_text=target,
            teacher_input_ids=teacher_ids,teacher_position_ids=teacher_pos,target_positions=list(range(W-1,S)),coordinates=coords,
            metadata=dict(sid=row['sid'],n=row['n'],qtype=row['qtype'],atype=row['atype'],question=row['question'],prompt_width=W,
                teacher_width=S,image_files=row['image_files'],image_tokens=row['n']*196,raw_row_sha256=row['raw_row_sha256']))
    finally:
        for frame in frames:frame.close()


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--render-plan',required=True,type=Path);args=ap.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),'CPU Slurm-only preparation')
    started=time.perf_counter();out=OUT/f'check_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False)
    data=DATA/out.name;data.mkdir(parents=True,exist_ok=False);frozen=sources();save(out/'request.json',dict(render_plan=str(args.render_plan.resolve()),policy=POLICY,source_sha256=frozen))
    inherited=inherited_sources();(out/'source').mkdir()
    for name,digest in {**frozen,**inherited}.items():
        target=out/'source'/name.replace('/','_');target.write_bytes((REPO/name).read_bytes());need(sha(target)==digest,'Source copy changed')
    try:
        import torch
        from transformers import AutoProcessor,__version__ as tf_version
        from scripts import diagnose_native_identity_join_joint_lora_v2 as p
        from scripts import render_mmred_official_shards as rendering
        from scripts import native_visual_memory_runtime as runtime
        from scripts.stage_native_vision_v6_teacher import model_metadata
        torch.set_num_threads(4);render=rendering.verify_stage(args.render_plan)
        need(sha(RECOVERY)==RECOVERY_SHA and sha(IDENTITY_PLAN)==IDENTITY_SHA,'Pinned recovery/native identities changed')
        recovery=read(RECOVERY);protocol=read(recovery['official_protocol_file']);system=protocol['system_prompt']
        upstream=Path(recovery['data_directory'])/'upstream/scripts/openai_server_inference.py'
        need(sha(upstream)==recovery['input_bindings'][str(upstream)],'Pinned upstream inference source changed')
        tree=ast.parse(upstream.read_text());prompts=[ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign)
            and any(isinstance(target,ast.Name) and target.id=='SYSTEM_PROMPT' for target in node.targets)]
        need(prompts==[system],'Recovered system must equal exact upstream literal')
        need(sha(recovery['official_protocol_file'])==recovery['artifacts'][recovery['official_protocol_file']],'Original system prompt changed')
        native=read(IDENTITY_PLAN);identity=native['native_identity'];processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,use_fast=False,local_files_only=True)
        owner,fn,api=p.backend.native_api(processor);rope=lambda **kw:fn(owner,**kw)
        need(api==identity['native_api'] and p.fingerprint(processor,str(tf_version))==identity['processor']
             and p.backend.runtime_identity()==identity['runtime'] and model_metadata()==identity['model'],'Pinned native processor/runtime identity changed')
        rows=read(render['rows_file']);need(len(rows)==5000 and Counter(r['pilot_role'] for r in rows)=={'train':4000,'val':400,'test':600},'Original5000 pairs required')
        targets={}
        for row in rows:
            text=canonical_target(row['atype'],row['answer'])
            if text not in targets:
                packet=runtime.prepare_text(processor,system,row['question'],target_text=text);ids=packet['target_ids']
                need(0<len(ids)<=50 and ids[-1]==151645,'Complete target outside50-token allowance')
                targets[text]=dict(ids=ids,length=len(ids),atype=row['atype'])
        selected=select_cases(rows);cases=[]
        for row in selected:
            packet=prepare_joint(processor,row,system,rope);file=data/f'case_{len(cases)}.pt';torch.save(packet,file)
            cases.append(dict(index=len(cases),sid=row['sid'],n=row['n'],qtype=row['qtype'],file=str(file),sha256=sha(file),metadata=packet['metadata'],target_ids=packet['target_ids']))
        table=dict(targets=targets,maximum_target_tokens=max(v['length'] for v in targets.values()),
            total_train_rows_per_epoch=sum(targets[canonical_target(r['atype'],r['answer'])]['length'] for r in rows if r['pilot_role']=='train'))
        save(out/'targets.json',table);save(out/'cases.json',cases);tests=parser_fixtures();save(out/'tests.json',tests)
        bindings={str(args.render_plan.resolve()):sha(args.render_plan),str(RECOVERY):RECOVERY_SHA,str(IDENTITY_PLAN):IDENTITY_SHA,
            recovery['official_protocol_file']:sha(recovery['official_protocol_file']),str(upstream):sha(upstream),render['rows_file']:sha(render['rows_file'])}
        plan=dict(protocol=PROTOCOL,policy=POLICY,source_sha256=frozen,inherited_source_sha256=inherited,
            render_plan=dict(file=str(args.render_plan.resolve()),sha256=sha(args.render_plan)),native_identity=identity,
            native_identity_sha256=native['native_identity_sha256'],precision=native['precision'],packages=native['packages'],
            system_prompt=system,rows_file=render['rows_file'],cases=cases,targets=table,input_bindings=bindings,
            artifacts={str(file):sha(file) for file in out.glob('*.json')},tests=tests,no_model_forward=True,no_training_release=True)
        save(out/'plan.json',plan);need(sources()==frozen and inherited_sources()==inherited and time.perf_counter()-started<600,'Preparation source/cap changed')
        save(out/'summary.json',dict(passed=True,completed=True,protocol=PROTOCOL,source_sha256=frozen,plan_file=str(out/'plan.json'),
            plan_sha256=sha(out/'plan.json'),cases=8,maximum_target_tokens=table['maximum_target_tokens'],elapsed_seconds=time.perf_counter()-started,no_training_release=True))
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),elapsed_seconds=time.perf_counter()-started,partial_outputs_retained=True));raise


if __name__=='__main__':main()

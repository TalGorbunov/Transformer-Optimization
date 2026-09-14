"""Structured assistant-content compatibility repair; same original MMReD inputs."""
from __future__ import annotations
import argparse
import ast
import json
import os
from pathlib import Path
import sys
import time
from collections import Counter
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import prepare_mmred_official_native_v2 as original
from scripts.stage_mmred_official_recovery import need,sha,object_sha,save
MODEL=original.MODEL
RECOVERY=original.RECOVERY
RECOVERY_SHA=original.RECOVERY_SHA
IDENTITY_PLAN=original.IDENTITY_PLAN
IDENTITY_SHA=original.IDENTITY_SHA
QTYPES=original.QTYPES
OUT=REPO/'outputs/native_aggregation_vlm/mmred_official_native_v3'
DATA=Path('/mnt/data/gabriele/gnn_transformer/mmred_official_native_v3')
PROTOCOL='mmred_official_native_training_profile_preparation_v3'
PROPOSAL='docs/paper/NATIVE_AGGREGATION_MMRED_OFFICIAL_NATIVE_PREPARATION_V3.md'
OWN=('scripts/prepare_mmred_official_native_v3.py',PROPOSAL,'slurm/mmred_official_native_prepare_v3.sbatch')
POLICY=dict(original.POLICY,protocol=PROTOCOL,prior_failed_cpu_seconds=73)
read=original.read
canonical_target=original.canonical_target
target_value=original.target_value
parse_generated=original.parse_generated
parser_fixtures=original.parser_fixtures
select_cases=original.select_cases
extract_upstream_prompt=original.extract_upstream_prompt
literal_prompt_fixtures=original.literal_prompt_fixtures


def sources():return {name:sha(REPO/name) for name in OWN}


def inherited_sources():
    release=REPO/'outputs/native_aggregation_vlm/mmred_official_native_v2/source_release.json'
    need(sha(release)=='2b8c1aeddc20dacecd05e3afeb5e01548ed38a8d7be9665823fd358cd0b25c86','Original V2 source release changed')
    record=read(release);result={**record['source_sha256'],**record['inherited_source_sha256']}
    need(len(result)==186 and all(sha(REPO/name)==digest for name,digest in result.items()),'Frozen original preparation/runtime changed')
    original.inherited_sources()
    failure=original.OUT/'check_443970/failure.json'
    need(sha(failure)=='9d94454c80251252c24e1616038f5bbfdbfd37722c436c7ccf4a496ac84c9a01' and read(failure)['type']=='TypeError'
         and read(failure)['message']=='string indices must be integers' and not (failure.parent/'summary.json').exists(),
         'Preserved structured-content failure changed')
    return result


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
        full=dict(processor.apply_chat_template(conversation+[dict(role='assistant',content=[dict(type='text',text=target)])],
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
        prompt_proof=literal_prompt_fixtures()
        need(extract_upstream_prompt(upstream.read_text())==system,'Recovered system must equal exact upstream literal')
        save(out/'upstream_literal_proof.json',prompt_proof)
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

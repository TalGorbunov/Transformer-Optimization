"""Bounded V7 joint-feature software profile only; no full cache or training.

Eight native model forwards produce twenty output rows from sixteen distinct
scene/prefix inputs. Only frozen norm/head replay and computational integrity
gate the profile. Aggregate count accuracy is neither scored nor an eligibility
condition. Full harvesting requires a later explicit implementation/release.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts.stage_native_vision_v7_joint_features import (
    DATA,OUT,GROUPS,need,read,save,sha,sources,snapshot,tensor_info,native_api,
    verify_plan,row_inputs,pack,parallel)
from scripts.cache_native_vision_v7_features import replay_metrics


def profile(args):
    import torch,transformers
    from gnnformer.runtime import load_runtime,move_to_device,get_rope_index_fn
    from scripts.probe_native_vision_v2_prefix import fingerprint
    torch.set_num_threads(4);start=time.perf_counter();plan,parent=verify_plan(args.plan,pixels=True)
    job=os.environ['SLURM_JOB_ID'];out=OUT/f'profile_{job}';out.mkdir(parents=True,exist_ok=False);snapshot(out)
    data=DATA/f'profile_{job}';data.mkdir(parents=True,exist_ok=False)
    (out/'plan.json').write_bytes(Path(args.plan).read_bytes())
    pixel_start=time.perf_counter();blob=torch.load(plan['pixels_file'],map_location='cpu',weights_only=True)
    need(blob['schema_version']==1 and set(blob['pixels'])==set(plan['pixel_info']),'Pixel coverage differs')
    pixels=blob['pixels']
    for key,value in pixels.items():need(tensor_info(value)==plan['pixel_info'][key],'Shared raw processed pixels changed')
    pixel_seconds=time.perf_counter()-pixel_start
    load_start=time.perf_counter();runtime=load_runtime(str(parallel.MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda')
    model=runtime.model;model.eval();model.requires_grad_(False)
    need(fingerprint(runtime.processor,str(transformers.__version__))==plan['processor']
         and parallel.runtime_identity()==plan['runtime'],'Joint runtime/processor differs')
    _,_,native=native_api(runtime.processor);need(native==plan['native_api'],'Installed native sources differ')
    norm=model.model.language_model.norm;language=model.model.language_model
    dtypes=dict(norm=str(norm.weight.dtype),lm_head=str(model.lm_head.weight.dtype))
    need(dtypes==dict(norm='torch.float16',lm_head='torch.float16')
         and not any(p.requires_grad for p in model.parameters()),'Expected frozen native FP16 readout')
    torch.cuda.synchronize();load_seconds=time.perf_counter()-load_start
    config=dict(schema_version=1,profile=True,plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),
                source_sha256=plan['source_sha256'],model=plan['model'],runtime=plan['runtime'],processor=plan['processor'],
                native_dtypes=dtypes,hardware=dict(name=torch.cuda.get_device_name(0),
                    capability=list(torch.cuda.get_device_capability(0)),cuda_version=torch.version.cuda,
                    matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32),slurm_job_id=job)
    save(out/'config.json',config)
    parallel.index(out,'V7 joint-feature profile',[('Configuration','config.json'),
                   ('Observations after completion','observations.json'),('Summary after completion','summary.json')])
    counts=dict(model=0,vision=0,language=0,norm=0);capture={};recording=False
    def model_hook(*_):counts['model']+=1
    def visual_hook(*_):counts['vision']+=1
    def language_hook(module,args,kwargs):
        counts['language']+=1;capture['positions']=kwargs['position_ids'].detach().cpu().clone()
        capture['mask']=kwargs['attention_mask'].detach().cpu().clone()
    def norm_hook(module,args):
        if recording:
            counts['norm']+=1;capture['hidden']=args[0][:,-1,:].detach().clone()
    handles=[model.register_forward_pre_hook(model_hook),model.model.visual.register_forward_pre_hook(visual_hook),
             language.register_forward_pre_hook(language_hook,with_kwargs=True),norm.register_forward_pre_hook(norm_hook)]
    rope=get_rope_index_fn(model);observations=[];raw=[];first={}
    try:
        for group in GROUPS:
            for case,ids in (('B1_first',plan['profile_groups'][group][:1]),('B4_all',plan['profile_groups'][group])):
                begin=time.perf_counter();rows=[row_inputs(torch,plan,pixels,fid) for fid in ids]
                inputs=pack(torch,rows,plan['pad_token_id'])
                expected_pos,expected_delta=rope(input_ids=inputs['input_ids'],image_grid_thw=inputs['image_grid_thw'],
                                                  attention_mask=inputs['attention_mask'])
                for i,fid in enumerate(ids):
                    layout=plan['layouts'][plan['features'][fid]['layout_id']];length=len(layout['input_ids'])
                    need(tensor_info(expected_pos[:,i:i+1,-length:])==layout['position_ids_info'],
                         'Joint packed valid positions differ from staged row positions')
                identity=dict(feature_ids=ids,input_ids=tensor_info(inputs['input_ids']),
                              attention_mask=tensor_info(inputs['attention_mask']),
                              image_grid_thw=tensor_info(inputs['image_grid_thw']),
                              position_ids=tensor_info(expected_pos),rope_deltas=expected_delta.tolist(),
                              image_sha256=[plan['scenes'][plan['features'][fid]['sid']]['image_sha256'] for fid in ids])
                item=move_to_device(inputs,runtime.device);torch.cuda.synchronize();prepared_at=time.perf_counter()
                before=dict(counts);capture.clear();recording=True
                with torch.inference_mode():output=model(**item,use_cache=False,logits_to_keep=1)
                recording=False;torch.cuda.synchronize();forward_seconds=time.perf_counter()-prepared_at
                need({key:counts[key]-before[key] for key in counts}==dict(model=1,vision=1,language=1,norm=1),
                     'Joint native forward counts differ')
                need(torch.equal(capture['positions'],expected_pos) and torch.equal(capture['mask'],inputs['attention_mask'])
                     and torch.equal(model.model.rope_deltas.detach().cpu(),expected_delta),'Executed joint layout differs')
                h=capture['hidden'];need(h.shape==(len(ids),3584) and h.dtype==torch.float16 and bool(torch.isfinite(h).all())
                    and output.logits.shape[:2]==(len(ids),1) and output.past_key_values is None,
                    'Malformed native joint state/logits')
                states=h.cpu().clone();harvest_seconds=time.perf_counter()-begin
                with torch.inference_mode():replayed=model.lm_head(norm(h.unsqueeze(1)))[:,0]
                native_logits=output.logits[:,0];metrics=replay_metrics(torch,native_logits,replayed)
                observation=dict(group=group,case=case,batch_size=len(ids),identity=identity,
                    preparation_seconds=prepared_at-begin,forward_seconds=forward_seconds,harvest_seconds=harvest_seconds,
                    states=tensor_info(states),replay=metrics,native_model_forwards=1,native_visual_forwards=1)
                if case=='B1_first':first[group]=dict(hidden=states[0],logits=native_logits[0].detach().cpu())
                else:
                    reference=first[group]
                    observation['B1_B4_first_row_descriptive']=dict(
                        hidden_rms=float((states[0].float()-reference['hidden'].float()).square().mean().sqrt()),
                        logits=replay_metrics(torch,native_logits[:1],reference['logits'].to(runtime.device).unsqueeze(0)),
                        eligibility_gate=False)
                observations.append(observation)
                raw.append(dict(group=group,case=case,feature_ids=ids,states=states,
                                native_logits=native_logits.detach().cpu(),replayed_logits=replayed.detach().cpu()))
                print(json.dumps(dict(group=group,case=case,seconds=harvest_seconds,replay_passed=metrics['passed'])),flush=True)
    finally:
        recording=False
        for handle in handles:handle.remove()
    need(len(observations)==8 and sum(r['batch_size'] for r in observations)==20
         and counts==dict(model=8,vision=8,language=8,norm=8),'Joint profile coverage differs')
    need(verify_plan(args.plan,pixels=False)[0]==plan,'Joint plan/source changed during profile')
    save(out/'observations.json',observations)
    raw_file=data/'raw.pt';torch.save(dict(schema_version=1,observations=raw),raw_file)
    stress={r['group']:r['harvest_seconds'] for r in observations if r['case']=='B4_all'}
    projected=[]
    for shard in plan['prospective_shards']:
        histogram=Counter(plan['features'][fid]['phase'] for fid in shard)
        projected.append(load_seconds+pixel_seconds+1.5*sum(math.ceil(histogram[group]/4)*stress[group] for group in GROUPS)+30)
    passed=all(row['replay']['passed'] for row in observations)
    summary=dict(config,completed=True,computational_integrity_passed=True,numerical_gate_passed=passed,passed=passed,
                 raw_file=str(raw_file),raw_sha256=sha(raw_file),observations_file=str(out/'observations.json'),
                 observations_sha256=sha(out/'observations.json'),model_load_seconds=load_seconds,
                 pixel_load_verify_seconds=pixel_seconds,prospective_shard_projected_seconds=projected,
                 projection_rule='load+pixel_load+1.5*sum(ceil(group_shard_rows/4)*B4_group_harvest_seconds)+30',
                 full_harvest_authorized=False,no_quality_or_accuracy_gate=True,
                 profile_calls=8,distinct_feature_rows=16,total_output_rows=20,
                 total_seconds=time.perf_counter()-start,peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
                 peak_memory_reserved_bytes=torch.cuda.max_memory_reserved())
    save(out/'summary.json',summary)
    parallel.index(data,'V7 joint raw profile tensors',[('Native states and replay logits','raw.pt')])
    print(json.dumps(summary,indent=2),flush=True)
    if not passed:raise SystemExit('Joint native readout replay failed; raw observations retained')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--profile',action='store_true',required=True)
    args=parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='gpu'
         and os.environ.get('SLURM_JOB_GPUS'),'Joint feature profile requires GPU Slurm')
    profile(args)


if __name__=='__main__':main()


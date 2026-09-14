"""Freeze a V11 penultimate cache over the exact immutable V10 input inventory.

CPU Slurm only. No new data, feature identities, tokenization, image processing,
model outcomes, or model weights. The ancestor processed pixels remain read-only.
A completed native V11 software profile is mandatory before this plan exists.
"""
from __future__ import annotations
import argparse
import copy
import os
from pathlib import Path
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import stage_native_vision_v10_features as ancestor
from scripts import profile_native_vision_v11_memory as software
from scripts.stage_native_vision_v10_features import (
    MODEL,PHASES,need,read,save,sha,object_sha,tensor_info,row_inputs,pack,
    model_metadata,runtime_identity,native_api,index)
DATA=Path('/mnt/data/gabriele/gnn_transformer/v11_parallel_local')
OUT=REPO/'outputs/native_aggregation_vlm/v11/features'
PROTOCOL='v11_parallel_local_penultimate_training_features'
READ_BOUNDARY=dict(decoder_blocks=28,after_block_index=26,before_block_index=27,
                   hidden_size=3584,native_dtype='torch.float16',capture='actual final-block input')
OWN=tuple(dict.fromkeys(('scripts/stage_native_vision_v11_features.py','scripts/cache_native_vision_v11_features.py',
    'slurm/native_vision_v11_feature_stage.sbatch','slurm/native_vision_v11_feature_profile.sbatch',
    'slurm/native_vision_v11_feature_harvest.sbatch','slurm/native_vision_v11_feature_merge.sbatch',
    *ancestor.OWN,*software.OWN)))


def sources():return {name:sha(REPO/name) for name in OWN}


def snapshot(directory):
    (directory/'source').mkdir()
    frozen=sources()
    for name,digest in frozen.items():
        path=directory/'source'/name.replace('/','_');path.write_bytes((REPO/name).read_bytes())
        need(sha(path)==digest,'Source changed during V11 snapshot')
    save(directory/'source_hashes.json',frozen)
    return frozen


def check_software_summary(result):
    need(result.get('schema_version')==1 and result.get('protocol')=='v11_native_memory_software_only'
         and all(result.get(k) is True for k in ('completed','passed','computational_integrity_passed',
             'zero_identity_passed','temporal_gradient_passed','binding_numerical_gate_passed',
             'frozen_backbone_unchanged','branch_unfitted_unchanged','no_fit','no_efficacy_scoring')),
         'Require the completed passing native V11 GPU software profile, not a CPU check')
    need(result['calls']==dict(model=39,visual=19,last_block=55)
         and result['last_block_replay_calls']==16 and result['native_head_replay_calls']==39
         and result['native_head_replay_global_positions']==41 and result['same_batch_last_block_replays']==12
         and result['native_forward_artifacts']==39 and result['controller_sequences']==8
         and result['binding_numerical_failures']==[], 'Incomplete V11 software coverage')


def verify_software(path,*,deep=False):
    path=Path(path).resolve();result=read(path);check_software_summary(result)
    need(path.is_relative_to(software.OUT) and path.name=='summary.json','Unexpected V11 GPU summary location')
    plan=software.verify(Path(result['plan_file']))
    need(sha(result['plan_file'])==result['plan_sha256'] and result['source_sha256']==plan['source_sha256']
         and all(result[k]==plan[k] for k in ('model','runtime','processor','native_api')),
         'Software profile differs from its frozen native plan')
    for name,digest in result['source_sha256'].items():
        need(sha(path.parent/'source'/name.replace('/','_'))==digest,'V11 software GPU source snapshot differs')
    if deep:
        for artifact in result['artifacts'].values():
            need(sha(artifact['path'])==artifact['sha256'],'V11 software evidence artifact changed')
    return dict(file=str(path),sha256=sha(path),plan_file=result['plan_file'],plan_sha256=result['plan_sha256'],
                source_sha256=result['source_sha256'],model=result['model'],runtime=result['runtime'],
                processor=result['processor'],native_api=result['native_api'],calls=result['calls'],
                computational_integrity_passed=True,binding_numerical_gate_passed=True)


def global_prompt_inventory(plan):
    result={}
    for fid in plan['groups']['global_empty']:
        feature=plan['features'][fid];layout=plan['layouts'][feature['layout_id']]
        need(feature['kind']=='global' and feature['prefix_ids']==[] and not layout.get('image_grid_thw'),
             'Global prompt must be the complete native text-only empty-prefix input')
        result[fid]=dict(empty_feature_id=fid,question=feature['question'],question_sha256=feature['question_sha256'],
                        layout_id=feature['layout_id'],prompt_tokens=len(layout['input_ids']))
    need(len(result)==54 and len({r['question'] for r in result.values()})==54,'Require all54 full global prompts')
    return result


def verify_plan(path,*,pixels=True):
    path=Path(path).resolve();plan=read(path)
    need(path.is_relative_to(DATA) and sha(path)==path.with_suffix('.sha256').read_text().strip(),'V11 plan path/sidecar changed')
    need(plan['schema_version']==1 and plan['protocol']==PROTOCOL and plan['source_sha256']==sources()
         and plan['read_boundary']==READ_BOUNDARY and plan['data_root']==str(DATA),'V11 source/read-boundary contract differs')
    previous=ancestor.verify_plan(Path(plan['ancestor_plan']['file']),pixels=pixels)
    need(sha(plan['ancestor_plan']['file'])==plan['ancestor_plan']['sha256']
         and plan['ancestor_plan']['source_sha256']==previous['source_sha256'],'V10 ancestor plan changed')
    # Everything not explicitly changed below remains the exact original value.
    changed={'protocol','data_root','source_sha256','source_files','scope','slurm_job_id','tests'}
    need(all(plan[k]==value for k,value in previous.items() if k not in changed),
         'V11 must inherit the exact V10 input/token/pixel/scene/profile inventory')
    expected_sources=dict(previous['source_files'])
    expected_sources.update(v10_feature_plan=dict(path=plan['ancestor_plan']['file'],sha256=plan['ancestor_plan']['sha256']),
        v11_software_summary=dict(path=plan['software_gate']['file'],sha256=plan['software_gate']['sha256']),
        v11_software_plan=dict(path=plan['software_gate']['plan_file'],sha256=plan['software_gate']['plan_sha256']))
    need(plan['source_files']==expected_sources,'Source artifact ledger differs')
    gate=verify_software(plan['software_gate']['file'])
    need(gate==plan['software_gate'] and all(plan[k]==gate[k] for k in ('model','runtime','processor')),
         'Software gate or shared native model/runtime/processor differs')
    # The V10 API additionally binds norm/bitsandbytes files; shared entries agree.
    for filename,digest in gate['native_api']['source_sha256'].items():
        need(plan['native_api']['source_sha256'].get(filename)==digest,'Native implementation differs across the two ancestors')
    need(len(plan['features'])==33658 and plan['global_prompts']==global_prompt_inventory(plan)
         and plan['global_prompt_count']==54,'V11 exact inventory or full-prompt schema differs')
    for name,digest in plan['source_sha256'].items():
        need(sha(path.parent/'source'/name.replace('/','_'))==digest,'V11 CPU source snapshot differs')
    return plan


def self_test():
    from scripts.cache_native_vision_v11_features import self_test as worker_test
    good=dict(schema_version=1,protocol='v11_native_memory_software_only',completed=True,passed=True,
        computational_integrity_passed=True,zero_identity_passed=True,temporal_gradient_passed=True,
        binding_numerical_gate_passed=True,frozen_backbone_unchanged=True,branch_unfitted_unchanged=True,
        no_fit=True,no_efficacy_scoring=True,calls=dict(model=39,visual=19,last_block=55),
        last_block_replay_calls=16,native_head_replay_calls=39,native_head_replay_global_positions=41,
        same_batch_last_block_replays=12,native_forward_artifacts=39,controller_sequences=8,binding_numerical_failures=[])
    check_software_summary(good)
    for field in ('completed','temporal_gradient_passed','binding_numerical_gate_passed'):
        bad=dict(good);bad[field]=False
        try:check_software_summary(bad)
        except ValueError:pass
        else:raise AssertionError('Incomplete software profile accepted: '+field)
    try:check_software_summary(dict(passed=True,no_model_loaded=True))
    except ValueError:pass
    else:raise AssertionError('CPU gate mistaken for GPU completion')
    return dict(passed=True,tests=['reject_cpu_only_summary','reject_incomplete_native_profile'],worker=worker_test())


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--ancestor-plan',type=Path)
    parser.add_argument('--software-summary',type=Path);parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),
         'Feature staging and tests require CPU-only Slurm')
    started=time.perf_counter();job=os.environ['SLURM_JOB_ID'];out=OUT/f'{"selftest" if args.self_test else "stage"}_{job}'
    out.mkdir(parents=True,exist_ok=False);frozen=snapshot(out)
    index(out,'V11 penultimate feature CPU attempt',[('Sources','source_hashes.json'),('Summary after completion','summary.json')])
    tests=self_test()
    if args.self_test:
        save(out/'summary.json',dict(tests,source_sha256=frozen));return
    need(args.ancestor_plan is not None and args.software_summary is not None,'Exact ancestor plan and passed GPU software summary are required')
    import torch
    torch.set_num_threads(4)
    previous=ancestor.verify_plan(args.ancestor_plan.resolve(),pixels=True)
    gate=verify_software(args.software_summary,deep=True)
    plan=copy.deepcopy(previous)
    source_files=dict(previous['source_files'])
    source_files.update(v10_feature_plan=dict(path=str(args.ancestor_plan.resolve()),sha256=sha(args.ancestor_plan)),
        v11_software_summary=dict(path=gate['file'],sha256=gate['sha256']),
        v11_software_plan=dict(path=gate['plan_file'],sha256=gate['plan_sha256']))
    plan.update(protocol=PROTOCOL,data_root=str(DATA),source_sha256=frozen,source_files=source_files,
        ancestor_plan=dict(file=str(args.ancestor_plan.resolve()),sha256=sha(args.ancestor_plan),source_sha256=previous['source_sha256']),
        software_gate=gate,read_boundary=READ_BOUNDARY,global_prompts=global_prompt_inventory(previous),global_prompt_count=54,
        scope='Exact V10 training inputs; fresh frozen states before final block27 and complete54 global prompts; no new data or efficacy claim',
        tests=tests,slurm_job_id=job)
    data=DATA/f'stage_{job}';data.mkdir(parents=True,exist_ok=False)
    # Copy the exact source snapshot alongside the plan for independent consumers.
    (data/'source').mkdir()
    for name in frozen:(data/'source'/name.replace('/','_')).write_bytes((out/'source'/name.replace('/','_')).read_bytes())
    save(data/'source_hashes.json',frozen)
    path=data/'plan.json';save(path,plan);path.with_suffix('.sha256').write_text(sha(path)+'\n')
    verified=verify_plan(path,pixels=False);need(verified==plan and sources()==frozen,'Saved V11 plan failed roundtrip verification')
    summary=dict(passed=True,completed=True,no_model_loaded=True,plan_file=str(path),plan_sha256=sha(path),
        source_sha256=frozen,read_boundary=READ_BOUNDARY,feature_count=len(plan['features']),counts=plan['counts'],
        training_scene_count=1782,global_prompt_count=54,ancestor_plan=plan['ancestor_plan'],software_gate=gate,
        native_hidden_payload_bytes=len(plan['features'])*3584*2,
        global_prompt_payload_bytes=sum(x['prompt_tokens'] for x in plan['global_prompts'].values())*3584*2,
        pixels_reused_without_modification=dict(file=plan['pixels_file'],sha256=plan['pixels_sha256']),
        seconds=time.perf_counter()-started,slurm_job_id=job)
    save(out/'summary.json',summary)
    index(data,'V11 immutable feature plan',[('Plan','plan.json'),('Sources','source_hashes.json')])
    print(__import__('json').dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()

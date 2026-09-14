"""Prospective V13 training-null feature inventory; no GPU worker or fit release.

Reuse unchanged V10 final-norm-input features by image bytes/question/prefix.
The only new states are missing K0 training images under independently assigned
strict numeric prefixes. Occurrence multiplicities remain in each 24-image bag;
unique feature extraction never changes the auxiliary reference weighting.

All execution, including metadata staging, requires CPU Slurm. This source is
preparation for a prospective registration; it submits no jobs and runs no model.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
CACHE=Path('/mnt/data/gabriele/gnn_transformer/v10_parallel_local/feature_cache.json')
DATA=Path('/mnt/data/gabriele/gnn_transformer/v13_null_features')
OUT=REPO/'outputs/native_aggregation_vlm/v13/null_features'
PROTOCOL='v13_training_null_feature_inventory_only'
OWN=('scripts/stage_native_vision_v13_null_features.py','slurm/native_vision_v13_null_inventory.sbatch')
EXPECTED=dict(questions=54,zero_scenes=108,occurrences=1296,unique_image_question_pairs=1274,
    duplicate_occurrences=22,strict_prefixes=18,occurrence_feature_positions=23328,
    required_local_features=22932,reused_local_features=3268,missing_local_features=19664,
    required_global_features=972,reused_global_features=972,new_global_features=0,
    unchanged_v10_features=33658,union_features=53322)
READ_BOUNDARY=dict(layer='actual native final norm input',hidden_size=3584,dtype='torch.float16',
    native_backbone='unchanged NF4 double quantization, BF16 compute, native FP16 norm/head, SDPA',
    ancestor_protocol='v10_parallel_local_training_features')


def need(condition,message):
    if not condition:raise ValueError(message)


def object_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def save(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def require_cpu():
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu'
        and not os.environ.get('SLURM_JOB_GPUS'),'Run inventory checks only in a CPU Slurm allocation')


def sources():return {name:sha(REPO/name) for name in OWN}


def feature_id(kind,identity,prefix):return object_sha([kind,identity,list(prefix)])


def inventory(cache,parent):
    """Pure metadata construction; callers verify source files and tensor hashes.

    No train/dev/test target is inferred for these auxiliary views. A bank's K0
    label justifies irrelevance; assigned prefixes never receive zero-answer CE.
    """
    need(cache['protocol']==parent['protocol']==READ_BOUNDARY['ancestor_protocol']
        and cache['complete'] and cache['training_only'] and cache['scenes']==parent['scenes']
        and set(cache['features'])==set(parent['features']),'Require the complete unchanged V10 training cache')
    need(cache['native_dtypes']==dict(norm='torch.float16',lm_head='torch.float16') and parent['native_dtype']=='torch.float16',
        'Only the native final-norm-input V10 cache is compatible; V11 penultimate states are ineligible')
    targets=parent['target_token_ids'];eos=parent['eos_token_id']
    prefixes=sorted({tuple(ids[:t]) for ids in targets.values() for t in range(len(ids))},key=lambda x:(len(x),x))
    need(list(map(list,prefixes))==parent['strict_prefix_vocabulary'] and len(prefixes)==18
        and set(targets)==set(map(str,range(17))) and eos==151645
        and all(ids[-1]==eos and eos not in ids[:-1] for ids in targets.values()),'Strict native target/prefix support changed')
    zero=[(sid,row) for sid,row in cache['scenes'].items() if row['gold']==0]
    questions=sorted({row['question'] for _,row in zero});banks={};groups={};required={};global_ids=set();pair_ids=set();layouts={}
    for question in questions:
        occurrence=[];source_scenes=[]
        for n in (8,16):
            selected=[(sid,row) for sid,row in zero if row['question']==question and row['n_frames']==n]
            need(len(selected)==1,'Require exactly one N8K0 and one N16K0 training scene for each question')
            sid,scene=selected[0];pids=scene['image_question_pair_ids']
            need(scene['split']=='train' and len(pids)==n and len(scene['local_feature_ids'])==n,'K0 source row inventory differs')
            source_scenes.append(dict(sid=sid,n_frames=n,gold=0,qa_sha256=scene['qa_sha256'],content_sha256=scene['content_sha256'],path=scene['path']))
            for index,pid in enumerate(pids):
                pair=parent['pairs'][pid]
                need(pair['question']==question and pid==object_sha([pair['image_sha256'],question]),'Actual image/question pair key differs')
                occurrence.append(dict(slot=len(occurrence),source_sid=sid,source_n_frames=n,source_frame_index=index,
                    image_question_pair_id=pid,image_sha256=pair['image_sha256'],
                    occurrence_image_path=str(Path(scene['path'])/f'{index:03d}.png')))
                pair_ids.add(pid)
        qid=object_sha(question)
        bank=dict(question=question,question_sha256=qid,source_scenes=source_scenes,occurrences=occurrence,
            ordering='N8K0 images0..7 then N16K0 images0..15; retain every occurrence',reference_count=24)
        banks[qid]=dict(bank,bank_sha256=object_sha(bank))
        need(len(occurrence)==24,'Null occurrence weight changed')
        for prefix in prefixes:
            gid=feature_id('global',question,prefix)
            need(gid in cache['features'] and parent['features'][gid]['kind']=='global'
                and parent['features'][gid]['question']==question and parent['features'][gid]['prefix_ids']==list(prefix),
                'Missing or incorrect frozen global prefix state')
            global_ids.add(gid);local_ids=[]
            for item in occurrence:
                pid=item['image_question_pair_id'];pair=parent['pairs'][pid]
                fid=feature_id('local',pid,prefix);lid=object_sha([pair['layout_id'],list(prefix)])
                descriptor=dict(feature_id=fid,kind='local',pair_id=pid,question=question,question_sha256=qid,
                    prefix_ids=list(prefix),base_layout_id=pair['layout_id'],phase='local_prefix' if prefix else 'local_empty',layout_id=lid)
                need(lid in parent['layouts'],'A required numeric local layout is absent from the V10 input inventory')
                layout=parent['layouts'][lid];empty=parent['layouts'][object_sha([pair['layout_id'],[]])]
                need(layout['input_ids']==empty['input_ids']+list(prefix) and layout['prefix_ids']==list(prefix)
                    and layout['original_prompt_tokens']==len(empty['input_ids'])
                    and layout['image_grid_thw']==empty['image_grid_thw'],'Auxiliary prefix retokenized or changed the visual layout')
                need(fid not in parent['features'] or parent['features'][fid]==descriptor,'Reused feature descriptor changed')
                need(fid not in required or required[fid]==descriptor,'A feature ID aliases different actual inputs')
                required[fid]=descriptor;layouts[lid]=layout;local_ids.append(fid)
            group_id=object_sha(['null_auxiliary',qid,list(prefix)])
            groups[group_id]=dict(question=question,question_sha256=qid,bank_sha256=banks[qid]['bank_sha256'],
                prefix_ids=list(prefix),global_feature_id=gid,local_feature_ids=local_ids,reference_count=24,
                auxiliary_only=True,zero_answer_ce_for_assigned_prefix=False)
    old_ids=set(cache['features']);required_ids=set(required);reused=required_ids&old_ids;missing=required_ids-old_ids
    counts=dict(questions=len(questions),zero_scenes=len(zero),occurrences=sum(len(v['occurrences']) for v in banks.values()),
        unique_image_question_pairs=len(pair_ids),duplicate_occurrences=sum(len(v['occurrences']) for v in banks.values())-len(pair_ids),
        strict_prefixes=len(prefixes),occurrence_feature_positions=sum(len(v['local_feature_ids']) for v in groups.values()),
        required_local_features=len(required_ids),reused_local_features=len(reused),missing_local_features=len(missing),
        required_global_features=len(global_ids),reused_global_features=len(global_ids&old_ids),new_global_features=len(global_ids-old_ids),
        unchanged_v10_features=len(old_ids),union_features=len(old_ids|missing))
    need(counts==EXPECTED,'Frozen V10 null feature inventory differs from the metadata audit')
    need(all(required[fid]['phase']=='local_prefix' for fid in missing),'Empty-prefix feature unexpectedly needs new extraction')
    by_prefix=[]
    for prefix in prefixes:
        ids={fid for fid,row in required.items() if row['prefix_ids']==list(prefix)}
        by_prefix.append(dict(prefix_ids=list(prefix),required_unique=len(ids),reused=len(ids&old_ids),missing=len(ids-old_ids)))
    return dict(counts=counts,strict_prefix_vocabulary=list(map(list,prefixes)),banks=banks,auxiliary_groups=groups,
        required_local_feature_ids=sorted(required),required_global_feature_ids=sorted(global_ids),
        reused_local_features={fid:cache['features'][fid] for fid in sorted(reused)},
        reused_global_features={fid:cache['features'][fid] for fid in sorted(global_ids)},
        missing_features={fid:required[fid] for fid in sorted(missing)},
        required_local_layouts=layouts,required_pairs={pid:parent['pairs'][pid] for pid in sorted(pair_ids)},
        missing_feature_shards=[sorted(missing)[i::4] for i in range(4)],per_prefix=by_prefix,
        missing_fp16_payload_bytes=len(missing)*3584*2,all_required_local_fp16_payload_bytes=len(required)*3584*2,
        proposed_batch_size=64,proposed_shard_model_calls=[(len(sorted(missing)[i::4])+63)//64 for i in range(4)],
        no_new_global_states=True,no_new_images=True,no_new_tokenization=True,no_new_native_layouts=True,
        preserve_occurrence_multiplicity=True,inference_lookup_table=False)


def stage(args):
    require_cpu();begin=time.perf_counter();job=os.environ['SLURM_JOB_ID']
    out=OUT/f'check_{job}';out.mkdir(parents=True,exist_ok=False);(out/'source').mkdir();frozen=sources()
    for name,digest in frozen.items():
        path=out/'source'/name.replace('/','_');path.write_bytes((REPO/name).read_bytes());need(sha(path)==digest,'Source changed during snapshot')
    save(out/'source_hashes.json',frozen)
    # Registration is an explicit artifact, supplied before this first stage.
    registration=read(args.registration)
    need(registration.get('protocol')==PROTOCOL and registration.get('inventory_stage_authorized') is True
        and registration.get('source_sha256')==frozen and registration.get('expected_counts')==EXPECTED,
        'Prospective exact-source inventory registration required; this is not a harvest release')
    from scripts import stage_native_vision_v10_features as ancestor
    cache=read(CACHE);need(sha(CACHE)==registration['parent_cache_sha256'],'Registered V10 cache changed')
    need(sha(cache['plan_file'])==cache['plan_sha256'],'V10 parent plan changed')
    parent=ancestor.verify_plan(cache['plan_file'],pixels=False)
    need(all(cache[k]==parent[k] for k in ('model','runtime','processor','native_api')),'Parent native feature provenance differs')
    result=inventory(cache,parent)
    for bank in result['banks'].values():
        for scene in bank['source_scenes']:need(sha(Path(scene['path'])/'qa.txt')==scene['qa_sha256'],'K0 source QA changed')
        for item in bank['occurrences']:need(sha(item['occurrence_image_path'])==item['image_sha256'],'Reference occurrence image changed')
    source_bindings=dict(cache=dict(file=str(CACHE),sha256=sha(CACHE)),parent_plan=dict(file=cache['plan_file'],sha256=cache['plan_sha256']),
        registration=dict(file=str(Path(args.registration).resolve()),sha256=sha(args.registration)))
    inherited_files={r['file']:r['file_sha256'] for r in list(result['reused_local_features'].values())+list(result['reused_global_features'].values())}
    # This stage binds file hashes recorded by the completed cache, without
    # reloading any tensor or claiming a new native extraction-route test.
    plan=dict(schema_version=1,protocol=PROTOCOL,source_sha256=frozen,source_bindings=source_bindings,
        parent_source_sha256=parent['source_sha256'],read_boundary=READ_BOUNDARY,
        model=parent['model'],runtime=parent['runtime'],processor=parent['processor'],native_api=parent['native_api'],
        parent_pixels_file=parent['pixels_file'],parent_pixels_sha256=parent['pixels_sha256'],
        inherited_state_file_hashes=inherited_files,inventory=result,inventory_sha256=object_sha(result),
        no_tensor_loaded=True,no_model_loaded=True,no_harvest_release=True,no_training_release=True,
        auxiliary_use='Null mean calibration only; ordinary complete-answer CE data and target sequences remain unchanged',
        extraction_route_policy='Reuse each existing state exactly; future missing-state harvest must retain native V10 route and report batch-route discrepancies',
        slurm_job_id=job)
    need(sources()==frozen,'Inventory source changed during stage')
    target=DATA/f'stage_{job}';target.mkdir(parents=True,exist_ok=False);save(target/'plan.json',plan)
    (target/'plan.sha256').write_text(sha(target/'plan.json')+'\n')
    summary=dict(passed=True,protocol=PROTOCOL,source_sha256=frozen,plan_file=str(target/'plan.json'),plan_sha256=sha(target/'plan.json'),
        counts=result['counts'],missing_fp16_payload_bytes=result['missing_fp16_payload_bytes'],
        proposed_shard_rows=list(map(len,result['missing_feature_shards'])),proposed_shard_model_calls=result['proposed_shard_model_calls'],
        no_tensor_loaded=True,no_model_loaded=True,no_harvest_release=True,no_training_release=True,seconds=time.perf_counter()-begin,slurm_job_id=job)
    save(out/'summary.json',summary)
    (out/'INDEX.md').write_text('# V13 prospective null inventory\n\n[Summary](summary.json) · [Sources](source_hashes.json)\n')
    (target/'INDEX.md').write_text('# V13 training-null feature plan\n\n[Plan](plan.json)\n')
    print(json.dumps(summary),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--registration',type=Path,required=True)
    args=parser.parse_args();stage(args)


if __name__=='__main__':main()

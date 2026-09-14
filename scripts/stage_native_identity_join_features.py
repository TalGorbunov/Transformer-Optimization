"""Freeze training-only learned-selection join feature inputs on Slurm CPU.

The local prompt is the unchanged global question. Deduplication uses only the
actual image/question and preceding token IDs. No local labels, gates, origin
probes, teacher outputs, old hidden states or dev/test features are constructed.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import stage_native_identity_join_learned as dataset
from scripts.stage_native_vision_v6_teacher import MODEL,model_metadata
from scripts.stage_native_vision_v10_features import pack,row_inputs,native_api,runtime_identity,tensor_info
need=dataset.need
read=dataset.read
save=dataset.save
def sha(path):return dataset.digest(Path(path))
oid=dataset.oid
DATA=Path('/mnt/data/gabriele/gnn_transformer/identity_join_learned/features')
CKPT=Path('/mnt/ckpts/gabriele/gnn_transformer/identity_join_learned/features')
OUT=REPO/'outputs/native_aggregation_vlm/identity_join_learned/features'
PROTOCOL='identity_join_learned_training_features'
NATIVE_PLAN=REPO/'outputs/native_aggregation_vlm/v18/semantic_gate_software/check_442830/plan.json'
NATIVE_PROOF=REPO/'outputs/native_aggregation_vlm/v18/semantic_gate_software/profile_442831/summary.json'
PHASES=('local_empty','local_prefix','global_empty','global_prefix')
OWN=('scripts/stage_native_identity_join_features.py','scripts/cache_native_identity_join_features.py',
     'slurm/native_identity_join_feature_stage.sbatch','slurm/native_identity_join_feature_profile.sbatch',
     'slurm/native_identity_join_feature_shard.sbatch','slurm/native_identity_join_feature_merge.sbatch',
     'scripts/stage_native_vision_v10_features.py','scripts/stage_native_vision_v6_teacher.py',
     'scripts/probe_native_vision_parallel_local.py','scripts/cache_native_vision_v6_teacher.py',
     'scripts/probe_native_vision_v2_prefix.py','gnnformer/runtime.py','gnnformer/constants.py',
     'gnnformer/data.py','gnnformer/paired_sequence_objectives.py','gnnformer/parallel_local_prompts.py')


def sources():
    from scripts import native_learned_selection_runtime as runtime
    return {**dataset.source_hashes(),**runtime.sources(),**{name:sha(REPO/name) for name in OWN}}


def snapshot(out):
    frozen=sources();(out/'source').mkdir()
    for name,h in frozen.items():
        value=(REPO/name).read_bytes();need(hashlib.sha256(value).hexdigest()==h,'Source changed during snapshot')
        with (out/'source'/name.replace('/','_')).open('xb') as stream:stream.write(value)
    save(out/'source_hashes.json',frozen);return frozen


def backbone_identity(payload):
    keys=('model','runtime','processor','native_api','native_dtypes','norm_weight','head_weight','norm_source_sha256','rms_norm_eps')
    return dict(schema_version=1,**{key:payload[key] for key in keys})


def feature_id(kind,identity,prefix):
    return oid([kind,identity,list(prefix)])


def inventory(rows):
    features={};pairs={};scenes={};training_pairs={};questions=set();images={}
    for row in rows:
        need(row['split']=='train' and row['n_frames'] in (8,16) and row['local_prompt']==row['question'], 'Training-only unchanged-question rows required')
        sid=row['sid'];q=row['question'];target=list(row['target_ids']);questions.add(q)
        need(target[-1]==151645 and 151645 not in target[:-1] and len(target) in (2,3), 'Native name/EOS support differs')
        prefixes=[target[:i] for i in range(len(target))];local=[];pids=[]
        for image in row['image_files']:
            pid=oid([image['sha256'],q]);pids.append(pid)
            value=dict(pair_id=pid,image_path=image['path'],image_sha256=image['sha256'],question=q)
            if pid in pairs:need(all(pairs[pid][key]==value[key] for key in ('image_sha256','question')), 'Conflicting actual local input')
            else:pairs[pid]=value
            images.setdefault(image['sha256'],dict(image,question=q))
            ids=[]
            for prefix in prefixes:
                fid=feature_id('local',pid,prefix);phase='local_prefix' if prefix else 'local_empty'
                descriptor=dict(feature_id=fid,kind='local',pair_id=pid,question=q,prefix_ids=prefix,phase=phase)
                need(fid not in features or features[fid]==descriptor,'Conflicting local causal feature identity')
                features[fid]=descriptor;ids.append(fid)
            local.append(ids)
        global_ids=[]
        for prefix in prefixes:
            fid=feature_id('global',q,prefix);phase='global_prefix' if prefix else 'global_empty'
            descriptor=dict(feature_id=fid,kind='global',question=q,prefix_ids=prefix,phase=phase)
            need(fid not in features or features[fid]==descriptor,'Conflicting global causal feature identity')
            features[fid]=descriptor;global_ids.append(fid)
        scenes[sid]=dict(sid=sid,path=row['path'],qa_sha256=row['qa_sha256'],content_sha256=row['content_sha256'],
            question=q,n_frames=row['n_frames'],split='train',gold=row['gold'],target_ids=target,target_prefixes=prefixes,
            local_feature_ids=local,global_feature_ids=global_ids,image_question_pair_ids=pids,pair_id=row['pair_id'])
        training_pairs.setdefault(row['pair_id'],[]).append(sid)
    groups={phase:sorted(fid for fid,value in features.items() if value['phase']==phase) for phase in PHASES}
    paired=[]
    for pid,sids in training_pairs.items():
        need(len(sids)==2,'Each variant needs exactly two training lengths')
        sids=sorted(sids,key=lambda sid:scenes[sid]['n_frames']);a,b=[scenes[sid] for sid in sids]
        need([a['n_frames'],b['n_frames']]==[8,16] and all(a[k]==b[k] for k in ('question','gold','target_ids','global_feature_ids')), 'Same-variant length pair differs')
        for sid,other in zip(sids,sids[::-1]):scenes[sid]['paired_scene_id']=other
        paired.append(dict(pair_id=pid,sids=sids,question=a['question'],gold=a['gold'],target_ids=a['target_ids']))
    need(len(scenes)==len(rows)==6048 and len(paired)==3024 and len(questions)==12,'Training cardinality differs')
    need(sum(len(x['target_ids']) for x in scenes.values())==13440
         and sum(x['n_frames']*len(x['target_ids']) for x in scenes.values())==161280
         and sum(x['n_frames'] for x in scenes.values())==72576,'Training target/local occurrence count differs')
    need(len(groups['global_empty'])==12 and len(groups['global_prefix'])==132 and len(images)<=864 and len(pairs)<=10368,'Strict-prefix/global/image support differs')
    return features,pairs,scenes,groups,paired,images


def profile_selection(features,groups,scenes):
    prefixes=sorted({tuple(row['prefix_ids']) for row in features.values() if row['prefix_ids']},key=lambda p:(len(p),p))
    need(len(prefixes)==11,'Exactly eleven nonempty strict name prefixes required')
    selected=dict(local_empty=groups['local_empty'][:8],global_empty=list(groups['global_empty']))
    for kind in ('local','global'):
        phase=kind+'_prefix';selected[phase]=[]
        for prefix in prefixes:
            candidates=[fid for fid in groups[phase] if tuple(features[fid]['prefix_ids'])==prefix]
            need(candidates,'Missing causal prefix');selected[phase].append(min(candidates))
    need({phase:len(ids) for phase,ids in selected.items()}==dict(local_empty=8,local_prefix=11,global_empty=12,global_prefix=11), 'Profile small-group counts differ')
    native=[]
    for name in ('Mary','Sandra','Noah'):
        sid=min(sid for sid,row in scenes.items() if row['gold']==name and row['n_frames']==16)
        scene=scenes[sid]
        for position,prefix in enumerate(scene['target_prefixes']):
            native.append(dict(case_id=f'{name}_t{position}',sid=sid,prefix_ids=prefix,
                feature_ids=[ids[position] for ids in scene['local_feature_ids']]+[scene['global_feature_ids'][position]]))
    need(len(native)==8 and sum(len(row['feature_ids']) for row in native)==136,'Fixed native training prefix coverage differs')
    return selected,native


def selftest(torch):
    target=[50,23274,151645]
    need([target[:i] for i in range(len(target))]==[[],[50],[50,23274]],'Strict name prefix includes a future token')
    pid=oid(['image','question'])
    need(feature_id('local',pid,[50])==feature_id('local',pid,target[:1])
         and feature_id('local',pid,[50])!=feature_id('local',pid,[50,23274]),'Causal prefix identity differs')
    a=dict(input_ids=torch.tensor([[7,8]]),attention_mask=torch.ones(1,2,dtype=torch.long))
    b=dict(input_ids=torch.tensor([[9]]),attention_mask=torch.ones(1,1,dtype=torch.long))
    result=pack(torch,[a,b],0)
    need(result['input_ids'].tolist()==[[7,8],[0,9]] and result['attention_mask'].tolist()==[[1,1],[0,1]],'Native left-padding differs')
    need(backbone_identity({k:k for k in ('model','runtime','processor','native_api','native_dtypes','norm_weight','head_weight','norm_source_sha256','rms_norm_eps')})['schema_version']==1,'Backbone identity malformed')
    return dict(passed=True,tests=['strict_name_prefix','causal_identity','native_left_padding','nongate_backbone_identity'])


def bind(path,bindings,expected=None):
    path=Path(path).resolve();actual=sha(path);need(expected is None or actual==expected,'Bound source changed: '+str(path));bindings[str(path)]=actual
    return read(path)


def original_bundle(runtime,processor,pair):
    return runtime.prepare_scene(processor,dict(sid='feature_'+pair['pair_id'],n_frames=1,question=pair['question'],
        image_files=[dict(path=pair['image_path'],sha256=pair['image_sha256'])]),verify_processor_parity=True)


def stage(args,out,frozen):
    import torch,transformers
    from transformers import AutoProcessor
    from scripts import native_learned_selection_runtime as runtime
    from scripts.probe_native_vision_v2_prefix import fingerprint
    from scripts import native_vision_v7_runtime as native
    torch.set_num_threads(4);tests=selftest(torch);bindings={}
    from scripts.cache_native_identity_join_features import selftest as worker_selftest
    tests['worker']=worker_selftest()
    manifest=dataset.verify_stage(args.dataset_summary);summary_path=Path(args.dataset_summary).resolve()
    if summary_path.is_dir():summary_path=summary_path/'summary.json'
    data_summary=bind(summary_path,bindings);bind(data_summary['manifest_file'],bindings,data_summary['manifest_sha256'])
    need(manifest['protocol']==dataset.PROTOCOL and manifest['binary_gate_dependency'] is False,'Require the separately released learned-selection dataset')
    rows=manifest['splits']['train_N8']['samples']+manifest['splits']['train_N16']['samples']
    # Preserve the original registered numeric family order rather than sorting
    # stringified family IDs. No development/test row enters the inventory.
    order={row['sid']:i for i,row in enumerate(read(manifest['samples_file'])) if row['split']=='train'}
    rows.sort(key=lambda row:order[row['sid']]);bind(manifest['samples_file'],bindings,manifest['samples_sha256'])
    features,pairs,scenes,groups,training_pairs,images=inventory(rows)
    inode_hashes={}
    for row in rows:
        need(sha(Path(row['path'])/'qa.txt')==row['qa_sha256'],'Training QA changed')
        for image in row['image_files']:
            path=Path(image['path']);stat=path.stat();key=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
            if key not in inode_hashes:inode_hashes[key]=sha(path)
            need(not path.is_symlink() and inode_hashes[key]==image['sha256'],'Training image changed')
    old=bind(NATIVE_PLAN,bindings,'45921222241820700a8e3cbd6637c819f331c4c9f27348b8896e1a233655b8aa')
    proof=bind(NATIVE_PROOF,bindings,'212567976c10b4c998b8368411a9694ffa66e410db03860f860a653c264510b1')
    need(proof['passed'] and proof['completed'] and proof['native_head_replay_passed'] and proof['plan_sha256']==sha(NATIVE_PLAN),'Frozen native backbone proof differs')
    identity=backbone_identity(old['native_identity_payload'])
    need(identity['model']==model_metadata() and identity['runtime']==runtime_identity(),'Actual frozen model/runtime differs')
    processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,use_fast=False,local_files_only=True)
    owner,rope,api=native_api(processor)
    need(fingerprint(processor,str(transformers.__version__))==identity['processor'] and api==identity['native_api'],'Native processor/API identity differs')
    targets=manifest['tokenizer']['targets']
    for name,record in targets.items():
        need(processor.tokenizer(name,add_special_tokens=False)['input_ids']+[151645]==record['target_ids'],'Actual native name targets differ')
    base={};by_question={}
    for pid,pair in sorted(pairs.items()):by_question.setdefault(pair['question'],pair)
    for question,pair in by_question.items():
        bundle=original_bundle(runtime,processor,pair)
        need(bundle['metadata']['local_prompt']==bundle['metadata']['global_prompt']==question,'New prompt interface differs')
        for kind,row in zip(('local','global'),bundle['row_inputs']):
            key=oid([kind,question]);base[key]=dict(input_ids=row['input_ids'][0].tolist(),original_prompt_tokens=row['input_ids'].shape[1])
            if kind=='local':base[key]['image_grid_thw']=row['image_grid_thw'].tolist()
    layouts={}
    for feature in features.values():
        row=base[oid([feature['kind'],feature['question']])];prefix=feature['prefix_ids'];key=oid([feature['kind'],feature['question'],prefix]);feature['layout_id']=key
        if key in layouts:continue
        ids=row['input_ids']+prefix;inputs=dict(input_ids=torch.tensor([ids]),attention_mask=torch.ones(1,len(ids),dtype=torch.long))
        if feature['kind']=='local':inputs['image_grid_thw']=torch.tensor(row['image_grid_thw'])
        positions,deltas=rope(owner,input_ids=inputs['input_ids'],image_grid_thw=inputs.get('image_grid_thw'),attention_mask=inputs['attention_mask'])
        layouts[key]=dict(row,input_ids=ids,prefix_ids=prefix,position_ids=positions.tolist(),rope_deltas=deltas.tolist(),
            input_ids_sha256=oid(ids),position_ids_info=tensor_info(positions),prompt_tokens=len(ids))
    data=DATA/f'stage_{os.environ["SLURM_JOB_ID"]}';data.mkdir(parents=True,exist_ok=False)
    pixels={};representatives={}
    for pid,pair in sorted(pairs.items()):representatives.setdefault(pair['image_sha256'],pair)
    for i,(image_sha,pair) in enumerate(sorted(representatives.items())):
        bundle=original_bundle(runtime,processor,pair);row=bundle['row_inputs'][0];expected=base[oid(['local',pair['question']])]
        need(row['input_ids'][0].tolist()==expected['input_ids'] and row['image_grid_thw'].tolist()==expected['image_grid_thw'],'Compact pixel/template construction differs')
        pixels[image_sha]=row['pixel_values'].contiguous().clone()
        if i%128==0:print(json.dumps(dict(processed_images=i+1,total=len(representatives))),flush=True)
    selected,native_cases=profile_selection(features,groups,scenes)
    for phase in PHASES:
        for fid in selected[phase]:
            feature=features[fid];pair=by_question[feature['question']] if feature['kind']=='global' else pairs[feature['pair_id']]
            bundle=runtime.append_observed_prefix(original_bundle(runtime,processor,pair),feature['prefix_ids'])
            row=bundle['row_inputs'][0 if feature['kind']=='local' else 1];layout=layouts[feature['layout_id']]
            need(row['input_ids'][0].tolist()==layout['input_ids'],'Actual strict-prefix processor contract differs')
            if feature['kind']=='local':need(tensor_info(row['pixel_values'])==tensor_info(pixels[pair['image_sha256']]),'Actual profile pixels differ')
    native_bundles={};native_layouts={};by_sid={r['sid']:r for r in rows}
    for case in native_cases:
        sid=case['sid']
        if sid not in native_bundles:native_bundles[sid]=runtime.prepare_scene(processor,dataset.runtime_view(by_sid[sid]),verify_processor_parity=True)
        bundle=runtime.append_observed_prefix(native_bundles[sid],case['prefix_ids'])
        layout=native.audit_layout(lambda **kw:rope(owner,**kw),bundle)
        native_layouts[case['case_id']]=dict(metadata=bundle['metadata'],layout=layout['metadata'],
            input_identity={k:tensor_info(v) for k,v in bundle['inputs'].items()})
    pixel_file=data/'pixels.pt';torch.save(dict(schema_version=1,pixels=pixels),pixel_file)
    native_file=data/'native_bundles.pt';torch.save(dict(schema_version=1,bundles=native_bundles),native_file)
    counts={p:len(groups[p]) for p in PHASES};shards=[sum((groups[p][i::4] for p in PHASES),[]) for i in range(4)]
    plan=dict(schema_version=1,protocol=PROTOCOL,source_sha256=frozen,source_files=bindings,dataset_summary=str(summary_path),
        dataset_manifest_file=data_summary['manifest_file'],dataset_manifest_sha256=data_summary['manifest_sha256'],
        native_identity=identity,native_identity_sha256=oid(identity),model=identity['model'],runtime=identity['runtime'],processor=identity['processor'],native_api=api,
        features=features,pairs=pairs,scenes=scenes,groups=groups,counts=counts,layouts=layouts,training_pairs=training_pairs,
        training_scene_count=6048,training_pair_count=3024,training_question_count=12,training_target_positions=13440,
        training_local_occurrences=161280,training_image_occurrences=72576,target_token_ids={name:record['target_ids'] for name,record in targets.items()},
        pad_token_id=processor.tokenizer.pad_token_id,eos_token_id=151645,strict_prefixes=sorted({tuple(v['prefix_ids']) for v in features.values()},key=lambda p:(len(p),p)),
        pixels_file=str(pixel_file),pixels_sha256=sha(pixel_file),pixel_info={k:tensor_info(v) for k,v in pixels.items()},
        native_bundles_file=str(native_file),native_bundles_sha256=sha(native_file),native_cases=native_cases,native_layouts=native_layouts,
        profile_groups=selected,profile_distinct_features=42,profile_binding_rows=434,profile_calls=dict(model=16,vision=12,standalone_heads=16),
        shards=shards,batch_size=64,shard_rule='Each phase sorted feature IDs modulo4; phases use separate batches',
        profile_seconds_cap=300,shard_seconds_cap=1200,local_prompt='unaltered complete global question',global_prompt='unaltered complete global question',
        read_boundary='Actual native pre-final-norm current-query hidden',state_dtype='torch.float16',no_binary_measurement=True,no_local_labels=True,
        no_feature_reuse=True,training_only=True,tests=tests)
    plan['inventory_sha256']=oid(dict(features=features,pairs=pairs,scenes=scenes,groups=groups,training_pairs=training_pairs))
    save(data/'plan.json',plan);save(out/'plan.json',plan)
    return dict(passed=True,completed=True,source_sha256=frozen,plan_file=str(data/'plan.json'),plan_sha256=sha(data/'plan.json'),
        counts=counts,feature_count=len(features),distinct_images=len(pixels),original_image_questions=len(pairs),training_scene_count=6048,
        training_target_positions=13440,training_only=True,no_model_loaded=True,no_binary_measurement=True,tests=tests)


def verify_plan(path,*,pixels=True,ancestors=True):
    path=Path(path).resolve();plan=read(path)
    need(path.is_relative_to(DATA.resolve()) and plan['protocol']==PROTOCOL and plan['source_sha256']==sources(),'New feature plan/source differs')
    need(plan['model']==model_metadata() and plan['runtime']==runtime_identity() and oid(plan['native_identity'])==plan['native_identity_sha256'],'Native backbone identity differs')
    if ancestors:
        for file,h in plan['source_files'].items():need(sha(Path(file))==h,'Bound data/native source changed')
    summary=read(OUT/path.parent.name/'summary.json')
    need(summary['passed'] is True and summary['completed'] is True and summary['plan_file']==str(path)
         and summary['plan_sha256']==sha(path) and summary['source_sha256']==plan['source_sha256'],'Passed exact feature CPU preparation required')
    for name,h in plan['source_sha256'].items():need(sha(OUT/path.parent.name/'source'/name.replace('/','_'))==h,'CPU source snapshot differs')
    need(plan['inventory_sha256']==oid({k:plan[k] for k in ('features','pairs','scenes','groups','training_pairs')}),'Training feature inventory changed')
    scenes=plan['scenes'];features=plan['features']
    need(len(scenes)==6048 and len(plan['training_pairs'])==3024 and sum(len(s['target_ids']) for s in scenes.values())==13440,'Training scene/target cardinality differs')
    for scene in scenes.values():
        target=plan['target_token_ids'][scene['gold']];prefixes=[target[:i] for i in range(len(target))]
        need(scene['split']=='train' and scene['target_ids']==target and scene['target_prefixes']==prefixes
             and len(scene['local_feature_ids'])==len(scene['image_question_pair_ids'])==scene['n_frames'],'Scene causal inventory differs')
        for pid,ids in zip(scene['image_question_pair_ids'],scene['local_feature_ids']):
            need(plan['pairs'][pid]['question']==scene['question'] and pid==oid([plan['pairs'][pid]['image_sha256'],scene['question']])
                 and ids==[feature_id('local',pid,prefix) for prefix in prefixes],'Local inputs have wrong owner/future target')
        need(scene['global_feature_ids']==[feature_id('global',scene['question'],prefix) for prefix in prefixes],'Global prefix identity differs')
    for fid,value in features.items():
        need(fid==value['feature_id'] and 151645 not in value['prefix_ids'] and 'gold' not in value and 'n_frames' not in value,'Feature includes target/length metadata')
        layout=plan['layouts'][value['layout_id']]
        need(layout['prefix_ids']==value['prefix_ids'] and layout['prompt_tokens']==len(layout['input_ids'])
             and layout['input_ids'][layout['original_prompt_tokens']:]==value['prefix_ids'],'Feature causal layout differs')
    need(plan['groups']=={phase:sorted(fid for fid,x in features.items() if x['phase']==phase) for phase in PHASES}
         and plan['counts']=={phase:len(plan['groups'][phase]) for phase in PHASES},'Feature phase coverage differs')
    need(plan['shards']==[sum((plan['groups'][p][i::4] for p in PHASES),[]) for i in range(4)],'Exact shard partition differs')
    selected,native=profile_selection(features,plan['groups'],scenes)
    need(plan['profile_groups']==selected and plan['native_cases']==native,'Frozen native/feature profile selection differs')
    if pixels:
        need(sha(Path(plan['pixels_file']))==plan['pixels_sha256'] and sha(Path(plan['native_bundles_file']))==plan['native_bundles_sha256'],'Consumed processor tensors changed')
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dataset-summary',type=Path,required=True);args=parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS')
         and int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==4,'Feature staging requires Slurm CPU4')
    out=OUT/f'stage_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False);frozen=snapshot(out);start=time.perf_counter()
    save(out/'request.json',dict(dataset_summary=str(args.dataset_summary),source_sha256=frozen))
    try:
        result=stage(args,out,frozen);need(sources()==frozen,'Feature sources changed');result.update(seconds=time.perf_counter()-start,slurm_job_id=os.environ['SLURM_JOB_ID'])
        save(out/'summary.json',result);print(json.dumps(dict(passed=True,directory=str(out))),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,seconds=time.perf_counter()-start));raise


if __name__=='__main__':main()

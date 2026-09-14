"""Exactly64 upstream frame renders on CPU Slurm; no full render/model release."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import zlib

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import stage_mmred_official_recovery as recovery
from scripts.stage_mmred_official_recovery import need,sha,object_sha,save
PROTOCOL='mmred_official_renderer_profile'
PROPOSAL='docs/paper/NATIVE_AGGREGATION_MMRED_OFFICIAL_RENDER_PROFILE.md'
PROPOSAL_SHA='3e944bb4b70848f2a67c35c01d7fa44669b2e47b62936e1e3f79179b8bdd04ee'
OWN=('scripts/profile_mmred_official_renderer.py',PROPOSAL,'slurm/mmred_official_render_profile.sbatch')
PARENT=recovery.OUT/'recovery_443939/summary.json'
PARENT_SHA='71c7cbc6ccc8bd30fed1b1a4465cb36a15ba09e20ac0935b976b258897cef5db'
PLAN_SHA='16abc5ef843dab2181a9dea69105b089be81dec26236ec4a744a91d008e107de'
DATA=recovery.DATA/'render_profile'
OUT=REPO/'outputs/native_aggregation_vlm/mmred_official_render_profile'
ROOMS=['Kitchen','Bathroom','Garden','Office','Bedroom','Hallway']
CHARS=['Sandra','Mary','John','Daniel','Michael']
POLICY=dict(protocol=PROTOCOL,cpu_seconds=300,cpu_cores=4,memory_gib=16,worlds=5000,frame_occurrences=40800,
    lengths=[1,2,4,8,16,32],unique_profile_keys=56,repeat_frames=6,sequence_reference_frames=2,total_frame_calls=64,
    width=512,height=512,deterministic_later_shards=4,render_workers=1,no_full_render_release=True,
    no_longer_render=True,no_model_or_head_calls=True,no_training_or_inference_release=True)


def read(path):return json.loads(Path(path).read_text())


def sources():
    need(sha(REPO/PROPOSAL)==PROPOSAL_SHA,'Renderer profile proposal changed')
    return {name:sha(REPO/name) for name in OWN}


def inherited_sources():return {**recovery.sources(),**recovery.INHERITED}


def bind(path,bindings,digest=None):
    path=Path(path);actual=sha(path);need(digest is None or digest==actual,'Bound renderer/profile input changed')
    bindings[str(path)]=actual;return actual


def check_time(started):need(time.perf_counter()-started<POLICY['cpu_seconds'],'Fixed CPU renderer-profile cap exceeded')


def frame_identity(frame,step):
    need(frame['step_id']==step and set(frame['rooms'])==set(ROOMS),'Original Step/room ownership differs')
    assignment={person:room for room,people in frame['rooms'].items() for person in people}
    need(Counter(person for people in frame['rooms'].values() for person in people)==Counter(CHARS),'Five unique characters required')
    value=dict(step=step,assignments=[[person,assignment[person]] for person in CHARS],
        occupant_order=[[room,list(frame['rooms'][room])] for room in ROOMS])
    return object_sha(value),value,assignment


def inventory(plan,bindings):
    rows=read(plan['rows_file']);need(len(rows)==5400,'Frozen5400 pilot rows required')
    eligible=[r for r in rows if r['n'] in POLICY['lengths']]
    need(len(eligible)==5000 and Counter(r['pilot_role'] for r in eligible)=={'train':4000,'val':400,'test':600},'Original eligible role counts differ')
    files={};frames={};worlds=[];by_length=defaultdict(set);anchor=None
    for row in eligible:
        file=row['renderer_file']
        if file not in files:
            bind(file,bindings,plan['artifacts'][file]);files[file]=read(file)
        sample=files[file][row['renderer_row']]
        need(sample['qid']==row['qid'] and sample['question']==row['question'] and sample['answer']==row['answer']
             and len(sample['sequence'])==row['n'] and object_sha(sample['sequence'])==row['exact_sequence_sha256'],
             'Exact original renderer sample/row ownership differs')
        keys=[]
        for index,frame in enumerate(sample['sequence']):
            key,value,assignment=frame_identity(frame,index+1);keys.append(key);by_length[row['n']].add(key)
            if key not in frames:frames[key]=dict(key=key,identity=value,assignment=assignment,step=index+1,
                representative=dict(sid=row['sid'],renderer_file=file,renderer_row=row['renderer_row'],sequence_index=index),owners=[])
            frames[key]['owners'].append(dict(sid=row['sid'],sequence_index=index,n=row['n']))
        worlds.append(dict(row,frame_keys=keys))
        if anchor is None and row['pilot_role']=='train' and row['n']==2:anchor=dict(sid=row['sid'],sequence=sample['sequence'],keys=keys)
    need(sum(len(r['frame_keys']) for r in worlds)==40800 and anchor is not None,'Original40800 eligible frame occurrences required')
    selected=list(anchor['keys']);need(len(set(selected))==2,'Two distinct N2 reference Step keys required')
    for n in POLICY['lengths']:
        choices=sorted((key for key in by_length[n] if key not in selected),key=lambda key:(-frames[key]['step'],key))
        need(len(choices)>=8,'Each fixed length needs eight unused profile keys');selected.extend(choices[:8])
    selected.extend(key for key in sorted(frames) if key not in selected and len(selected)<56)
    need(len(selected)==len(set(selected))==56,'Exactly56 distinct profile keys required')
    shards=[dict(index=i,keys=[key for j,key in enumerate(sorted(frames)) if j%4==i]) for i in range(4)]
    return worlds,frames,dict(unique_keys=selected,repeat_keys=selected[:6],sequence_reference=anchor,
        selection_rule='N2firstworld anchors; eight unused perN by decreasingStep/hash; fill byhash'),shards


def renderer_environment(plan,bindings):
    upstream=Path(plan['data_directory'])/'upstream';sys.path.insert(0,str(upstream))
    # Must be set before any Matplotlib import, including upstream imports.
    os.environ['MPLCONFIGDIR']=tempfile.mkdtemp(prefix='mmred_mpl_',dir=os.environ.get('TMPDIR','/tmp'))
    os.environ['MPLBACKEND']='Agg'
    import matplotlib
    from matplotlib import font_manager,ft2font
    from PIL import Image,_imaging
    from mmred.vgen import visualization
    need(Path(inspect.getfile(visualization)).resolve()==upstream/'mmred/vgen/visualization.py'
         and visualization.WIDTH==512 and visualization.HEIGHT==512
         and visualization.DEFAULT_ROOMS==ROOMS and visualization.DEFAULT_CHARS==CHARS
         and matplotlib.get_backend().lower()=='agg','Exact upstream renderer/constants/backend required')
    loaded={}
    for name,module in tuple(sys.modules.items()):
        if name=='mmred' or name.startswith('mmred.'):
            file=getattr(module,'__file__',None)
            if file:
                path=Path(file).resolve();need(path.is_relative_to(upstream),'Unexpected installed MMReD module')
                loaded[str(path)]=bind(path,bindings,plan['input_bindings'][str(path)])
    fonts={}
    for weight in ('normal','bold'):
        path=Path(font_manager.findfont(font_manager.FontProperties(weight=weight),fallback_to_default=False)).resolve()
        fonts[weight]=dict(file=str(path),sha256=bind(path,bindings))
    config=Path(matplotlib.matplotlib_fname()).resolve();bind(config,bindings)
    extensions={str(Path(module.__file__).resolve()):bind(Path(module.__file__).resolve(),bindings) for module in (ft2font,_imaging)}
    environment=dict(upstream_commit=plan['upstream_commit'],upstream_sources=loaded,
        packages={name:importlib.metadata.version(name) for name in ('matplotlib','Pillow','numpy','pandas')},
        python=platform.python_version(),backend=matplotlib.get_backend(),freetype=ft2font.__freetype_version__,zlib=zlib.ZLIB_RUNTIME_VERSION,
        rc_file=dict(file=str(config),sha256=sha(config)),rc_params={k:repr(v) for k,v in sorted(matplotlib.rcParams.items())},
        font_files=fonts,extension_files=extensions,width=512,height=512,raw_png_mode_preserved=True)
    return visualization,Image,environment


def inspect_png(path,Image):
    with Image.open(path) as value:
        value.load();need(value.format=='PNG' and value.size==(512,512),'Original512x512 PNG required')
        import hashlib
        pixel=hashlib.sha256(value.tobytes()).hexdigest();mode=value.mode
    return dict(file=str(path),sha256=sha(path),bytes=path.stat().st_size,mode=mode,width=512,height=512,pixel_sha256=pixel)


def profile(out,data,frozen,started):
    bindings={};bind(PARENT,bindings,PARENT_SHA);parent=recovery.verify_stage(PARENT)
    need(sha(read(PARENT)['plan_file'])==PLAN_SHA,'Exact passed recovery plan required')
    bind(read(PARENT)['plan_file'],bindings,PLAN_SHA)
    for file,digest in {**parent['input_bindings'],**parent['artifacts']}.items():bind(file,bindings,digest)
    worlds,frames,selection,shards=inventory(parent,bindings)
    for name,value in (('worlds.json',worlds),('frame_inventory.json',frames),('profile_selection.json',selection),('shards.json',shards)):save(out/name,value)
    visualization,Image,environment=renderer_environment(parent,bindings)
    save(out/'render_environment.json',environment);namespace=object_sha(environment);setup=time.perf_counter()-started
    records=[];direct={}
    for index,key in enumerate(selection['unique_keys']+selection['repeat_keys']):
        check_time(started);kind='unique' if index<56 else 'repeat';tick=time.perf_counter()
        target=data/f'{kind}_{index:02d}_{key}';target.mkdir();frame=frames[key]
        visualization.frame2png(frame['assignment'],frame['step'],target)
        record=dict(kind=kind,key=key,**inspect_png(target/f'frame_{frame["step"]:04d}.png',Image))
        record['seconds']=time.perf_counter()-tick;records.append(record)
        save(out/f'render_{index:02d}.json',record)
        if kind=='unique':direct[key]=record
    check_time(started);tick=time.perf_counter();target=data/'sequence_reference'
    visualization.render_sequence_from_json(selection['sequence_reference']['sequence'],target,as_gif=False)
    reference=[]
    for index,key in enumerate(selection['sequence_reference']['keys']):
        reference.append(dict(kind='sequence_reference',key=key,**inspect_png(target/f'frame_{index+1:04d}.png',Image)))
    reference_seconds=time.perf_counter()-tick
    for item in reference:item['seconds']=reference_seconds;item['seconds_is_two_frame_batch_upper_bound']=True
    records.extend(reference);save(out/'sequence_reference.json',dict(batch_seconds=reference_seconds,frames=reference))
    need(len(records)==64 and len(direct)==56,'Exactly64 frame renders required')
    comparisons=[dict(kind=r['kind'],key=r['key'],png_exact=r['sha256']==direct[r['key']]['sha256'],
        pixels_exact=r['pixel_sha256']==direct[r['key']]['pixel_sha256'],mode_exact=r['mode']==direct[r['key']]['mode']) for r in records if r['kind']!='unique']
    passed=all(r['png_exact'] and r['pixels_exact'] and r['mode_exact'] for r in comparisons)
    T=max(r['seconds'] for r in records);maximum_bytes=max(r['bytes'] for r in records)
    projection=dict(setup_seconds=setup,maximum_measured_frame_seconds=T,maximum_png_bytes=maximum_bytes,
        shards=[dict(index=s['index'],unique_keys=len(s['keys']),projected_seconds=setup+1.5*len(s['keys'])*T+30,
            projected_png_bytes=1.5*len(s['keys'])*maximum_bytes) for s in shards],formula='setup+1.5*keys*max_frame_seconds+30',
        reuses_profile_frames_in_cost=False,cap_seconds=None,no_full_render_release=True,empirical_estimate_not_guarantee=True)
    save(out/'render_records.json',records);save(out/'comparisons.json',comparisons);save(out/'projection.json',projection)
    analysis=dict(passed=passed,worlds=5000,frame_occurrences=40800,unique_cache_keys=len(frames),profile_frame_calls=64,
        direct_frame_calls=62,sequence_wrapper_calls=1,sequence_wrapper_frame_calls=2,cache_namespace=namespace,
        comparisons=comparisons,setup_seconds=setup,frame_seconds=sum(r['seconds'] for r in records[:62])+reference_seconds,
        projection=projection,no_full_render_release=True,no_model_or_head_calls=True)
    save(out/'analysis.json',analysis)
    need(passed,'PNG repeat/sequence equivalence failed; all64 outputs retained')
    publication=time.perf_counter();artifacts={str(path):sha(path) for path in out.glob('*.json')}
    artifacts.update({r['file']:r['sha256'] for r in records})
    plan=dict(protocol=PROTOCOL,policy=POLICY,source_sha256=frozen,inherited_source_sha256=inherited_sources(),
        recovery=dict(file=str(PARENT),sha256=PARENT_SHA,plan_sha256=PLAN_SHA),input_bindings=bindings,artifacts=artifacts,
        worlds_file=str(out/'worlds.json'),frame_inventory_file=str(out/'frame_inventory.json'),selection_file=str(out/'profile_selection.json'),
        environment_file=str(out/'render_environment.json'),records_file=str(out/'render_records.json'),shards_file=str(out/'shards.json'),
        projection_file=str(out/'projection.json'),cache_namespace=namespace,analysis=analysis,no_full_render_release=True)
    save(out/'plan.json',plan)
    (out/'REPORT.md').write_text('# Original MMReD renderer profile\n\nThe exact upstream renderer produced64 frame images:56 distinct input keys, six repeated images and two full-sequence reference frames. Deterministic bytes and sequence equivalence passed. All5,000 eligible worlds/40,800 occurrences are inventoried; full sharded rendering and allN64/N128/model work remain held. See projection.json for measured time/size estimates.\n')
    return dict(passed=True,completed=True,protocol=PROTOCOL,policy=POLICY,source_sha256=frozen,inherited_source_sha256=inherited_sources(),
        plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),
        report_file=str(out/'REPORT.md'),report_sha256=sha(out/'REPORT.md'),publication_before_summary_seconds=time.perf_counter()-publication,
        no_full_render_release=True,no_training_or_inference_release=True)


def verify_profile(path):
    path=Path(path).resolve();path=path/'summary.json' if path.is_dir() else path;summary=read(path)
    need(summary['passed'] is summary['completed'] is True and summary['protocol']==PROTOCOL and summary['policy']==POLICY
         and summary['source_sha256']==sources() and summary['inherited_source_sha256']==inherited_sources()
         and not (path.parent/'failure.json').exists(),'Passed exact renderer-profile summary required')
    need(sha(summary['plan_file'])==summary['plan_sha256'],'Renderer-profile plan changed');plan=read(summary['plan_file'])
    need(plan['source_sha256']==sources() and plan['policy']==POLICY and plan['no_full_render_release'] is True,'Profile plan/source/release flags differ')
    for mapping in (plan['input_bindings'],plan['artifacts']):
        for file,digest in mapping.items():need(sha(file)==digest,'Bound renderer input/output changed')
    for name,digest in {**sources(),**inherited_sources()}.items():need(sha(path.parent/'source'/name.replace('/','_'))==digest,'Renderer source archive changed')
    return plan


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu'
         and not os.environ.get('SLURM_JOB_GPUS') and int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==4,'Only four-core CPU Slurm rendering allowed')
    started=time.perf_counter();tag=f'profile_{os.environ["SLURM_JOB_ID"]}';out=OUT/tag;data=DATA/tag
    out.mkdir(parents=True,exist_ok=False);data.mkdir(parents=True,exist_ok=False);frozen=sources();(out/'source').mkdir()
    for name,digest in {**frozen,**inherited_sources()}.items():
        target=out/'source'/name.replace('/','_');target.write_bytes((REPO/name).read_bytes());need(sha(target)==digest,'Source copy differs')
    save(out/'request.json',dict(protocol=PROTOCOL,policy=POLICY,recovery=str(PARENT),source_sha256=frozen))
    try:
        result=profile(out,data,frozen,started);check_time(started);need(sources()==frozen,'Profile source changed')
        save(out/'summary.json',dict(result,elapsed_seconds=time.perf_counter()-started))
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,
            elapsed_seconds=time.perf_counter()-started,partial_outputs_retained=True,no_full_render_release=True));raise


if __name__=='__main__':main()

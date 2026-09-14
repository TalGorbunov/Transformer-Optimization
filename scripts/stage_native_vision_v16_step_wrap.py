"""Data-only V16 intervention: replace displayed Step i by ((i-1)%16)+1.

Physical frame order, QA bytes, conjunction semantics and all reference images
remain unchanged. Repeated displayed ordinals deliberately violate the original
unique-Step rendering law; these are counterfactual images, not new efficacy
examples. No model, fitting, prediction or GPU work is performed here.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
import sys
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
DEST=Path('/mnt/data/gabriele/gnn_transformer/v16_step_wrap')
OUT=REPO/'outputs/native_aggregation_vlm/v16/data_staging'
SOURCE_PLAN=REPO/'outputs/native_aggregation_vlm/v15/null_comparison_study/check_442691/plan.json'
PROTOCOL='v16_step_wrap16_counterfactual_data_only'
POLICY=dict(protocol=PROTOCOL,cases=34,families=17,counts=list(range(17)),lengths=[32,64],
    actual_image_occurrences=1632,first16_identity_occurrences=544,changed_label_occurrences=1088,
    displayed_step='(zero_based_frame_index %16)+1',physical_step='zero_based_frame_index+1',
    allowed_difference_box=[0,476,512,512],original_dimensions=[512,512],image_mode='RGB',
    preserve_original_qa_bytes=True,preserve_physical_frame_order=True,references_original=True,
    no_model_calls=True,no_gpu_calls=True,no_fit=True,
    limitation='Repeated displayed ordinals are a counterfactual nuisance transformation, not the original unique-Step law or a new method.')
OWN=('scripts/stage_native_vision_v16_step_wrap.py','slurm/native_vision_v16_step_wrap_stage.sbatch',
     'datasets/mmred/render_mmred.py','scripts/stage_native_vision_pilot.py')


def need(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()
def object_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def save(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')


def sources():
    from scripts import evaluate_native_vision_v15_null_comparison as study
    return {**study.sources(),**{name:sha(REPO/name) for name in OWN}}


def snapshot(out):
    (out/'code').mkdir();frozen=sources()
    for name,digest in frozen.items():
        target=out/'code'/name.replace('/','_');target.write_bytes((REPO/name).read_bytes())
        need(sha(target)==digest,'Source changed while snapshotting')
    save(out/'source_hashes.json',frozen)
    (out/'INDEX.md').write_text('# V16 Step-label staging\n\n[Summary](summary.json) · [Frozen plan](stage_plan.json) · [Sources](source_hashes.json).\n')
    return frozen


def displayed_step(index):
    need(type(index) is int and index>=0,'Physical frame index must be a nonnegative integer')
    return index%16+1


def qa_states(record):
    path=Path(record['path'])/'qa.txt';need(sha(path)==record['qa_sha256'],'Original QA bytes changed')
    lines=path.read_text().splitlines();a,b=lines.index('question:'),lines.index('answer:')
    content=[line.strip() for line in lines[a+1:b] if line.strip()]
    states=[ast.literal_eval(line) for line in content if line.startswith('{')]
    need([line for line in content if not line.startswith('{')]==[record['question']]
        and len(states)==record['n_frames'] and int(lines[b+1])==record['gold'],'QA question/count/frame layout differs')
    need(object_sha(dict(states=states,question=record['question']))==record['content_sha256'],'Original semantic content digest differs')
    frames=[]
    for i,state in enumerate(states):
        need(state['step_id']==i+1 and len(state['rooms'])==1,'Physical Step/room count differs')
        room,characters=next(iter(state['rooms'].items()));need(len(characters)==1,'Expected one character occurrence per image')
        frames.append((characters[0],room))
    positives=sum(c==record['target_character'] and r==record['target_room'] for c,r in frames)
    need(positives==record['gold'],'Independent conjunction count changed')
    return states,frames


def image_audit(original,wrapped):
    from PIL import ImageChops
    need(original.mode==wrapped.mode=='RGB' and original.size==wrapped.size==(512,512),'Native render dimensions/mode differ')
    diff=ImageChops.difference(original,wrapped);bbox=diff.getbbox()
    bands=diff.split();maximum=ImageChops.lighter(ImageChops.lighter(bands[0],bands[1]),bands[2])
    changed=512*512-maximum.histogram()[0]
    return dict(original_rgb_sha256=hashlib.sha256(original.tobytes()).hexdigest(),
        wrapped_rgb_sha256=hashlib.sha256(wrapped.tobytes()).hexdigest(),
        pixel_difference_bbox=list(bbox) if bbox else None,changed_pixels=changed,
        outside_footer_equal=diff.crop((0,0,512,476)).getbbox() is None)


def render_exclusive(renderer,rooms,step,path):
    path=Path(path);need(not path.exists() and not path.is_symlink(),'Never overwrite a rendered artifact')
    renderer.render_frame(rooms,step,str(path));need(path.is_file() and not path.is_symlink(),'Renderer did not produce an ordinary PNG')


def self_test(renderer,work):
    from PIL import Image
    folder=work/'renderer_selftest';folder.mkdir()
    rooms={'Kitchen':['Sandra']};before=object_sha(rooms)
    for step in (1,16,17,64):render_exclusive(renderer,rooms,step,folder/f'{step}.png')
    render_exclusive(renderer,rooms,1,folder/'repeat1.png')
    need(sha(folder/'1.png')==sha(folder/'repeat1.png'),'Renderer is not byte deterministic')
    with Image.open(folder/'1.png') as a,Image.open(folder/'17.png') as b:
        report=image_audit(a,b);need(report['outside_footer_equal'] and report['changed_pixels']>0,'Step-only render altered scene pixels')
        changed=a.copy();changed.putpixel((0,0),(0,0,0));need(not image_audit(a,changed)['outside_footer_equal'],'Pixel audit failed to detect off-footer changes')
    need([displayed_step(i) for i in (0,15,16,31,63)]==[1,16,1,16,16] and object_sha(rooms)==before,'Wrap boundaries or immutable scene semantics differ')
    return dict(passed=True,tests=['deterministic_png_render','Step_only_footer_difference','reject_off_footer_pixel_change','wrap_boundaries','unchanged_scene_semantics'])


def stage(out,frozen):
    from PIL import Image
    from scripts import evaluate_native_vision_v15_null_comparison as study
    from scripts.stage_native_vision_pilot import prepare_renderer,PARK_ROOMS
    begin=time.perf_counter();plan=study.verify_plan(SOURCE_PLAN)
    records=[case['record'] for case in plan['cases']]
    need(records==study.choose_records(read(plan['fresh_manifest']['file'])) and len(records)==34,'Require exact frozen V15 selection/order')
    need(not (DEST/'main_manifest.json').exists(),'Canonical V16 data already published; never overwrite')
    DEST.mkdir(parents=True,exist_ok=True);work=DEST/out.name;work.mkdir()
    (work/'INDEX.md').write_text('# V16 counterfactual rendered pixels\n\nOriginal render replays and wrapped renders retain exact occurrence provenance.\n')
    replay=work/'original_replays';cache=work/'render_cache';replay.mkdir();cache.mkdir()
    records_frames={};unique={};source_bindings={str(SOURCE_PLAN):sha(SOURCE_PLAN)}
    for record in records:
        states,frames=qa_states(record);records_frames[record['sid']]=frames
        source_bindings[str(Path(record['path'])/'qa.txt')]=record['qa_sha256']
        need(len(record['image_files'])==record['n_frames'],'Image occurrence count differs')
        for i,(im,(character,room)) in enumerate(zip(record['image_files'],frames)):
            need(room in PARK_ROOMS and character in ('Sandra','Mary','Michael','John','Daniel','Laura','Peter','Emma','Noah'),'Unknown renderer symbol')
            need(im['mode']=='RGB' and im['dimensions']==[512,512] and sha(im['path'])==im['sha256'],'Source image identity changed')
            source_bindings[str(Path(im['path']).resolve())]=im['sha256'];spec=(character,room,i+1)
            previous=unique.get(im['sha256'])
            need(previous is None or previous['spec']==spec,'One source image digest aliases incompatible frame semantics')
            unique[im['sha256']]=dict(spec=spec,image=im)
    for bank in plan['reference_banks'].values():
        for im in bank['occurrences']:
            need(sha(im['path'])==im['sha256'],'Original reference image changed')
            source_bindings[str(Path(im['path']).resolve())]=im['sha256']
    renderer,provenance=prepare_renderer([dict(r,source_path=r['path']) for r in records[:2]])
    original_provenance=read(plan['fresh_manifest']['file'])['renderer_provenance']
    for key in ('renderer_path','renderer_sha256','pillow_version','room_order','fonts'):
        need(provenance[key]==original_provenance[key],'Renderer/font/Pillow provenance changed: '+key)
    stage_plan=dict(schema_version=1,policy=POLICY,source_sha256=frozen,source_v15_plan_file=str(SOURCE_PLAN),source_v15_plan_sha256=sha(SOURCE_PLAN),
        source_bindings=source_bindings,source_records=records,renderer_provenance=provenance,
        unique_original_images=len(unique),wrapped_specs=[list(x) for x in sorted({(c,r,displayed_step(step-1)) for c,r,step in (v['spec'] for v in unique.values())})])
    save(out/'stage_plan.json',stage_plan);(out/'stage_plan.sha256').write_text(sha(out/'stage_plan.json')+'\n')
    tests=self_test(renderer,work)
    def render_wrapped(spec):
        character,room,step=spec;path=cache/f'{character}_{room}_{step:03d}.png'
        render_exclusive(renderer,{room:[character]},step,path)
        with Image.open(path) as image:need(image.mode=='RGB' and image.size==(512,512),'Wrapped render shape changed')
        return spec,dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size,dimensions=[512,512],mode='RGB')
    wrapped_specs=[tuple(x) for x in stage_plan['wrapped_specs']]
    with ThreadPoolExecutor(max_workers=4) as pool:wrapped_cache=dict(pool.map(render_wrapped,wrapped_specs))
    def check_original(item):
        digest,value=item;character,room,step=value['spec'];original=value['image'];path=replay/f'{digest}.png'
        render_exclusive(renderer,{room:[character]},step,path);replacement=wrapped_cache[character,room,displayed_step(step-1)]
        with Image.open(original['path']) as a,Image.open(path) as reproduced,Image.open(replacement['path']) as b:
            need(a.mode==reproduced.mode=='RGB' and a.size==reproduced.size==(512,512) and a.tobytes()==reproduced.tobytes(),
                'Renderer no longer reproduces exact original RGB pixels')
            proof=image_audit(a,b)
        need(proof['outside_footer_equal'] and ((proof['changed_pixels']==0)==(step<=16)),
            'Wrapped label changed forbidden pixels or failed expected first16 identity')
        if step<=16:need(original['sha256']==replacement['sha256'],'First16 PNG bytes must stay exactly identical')
        return digest,dict(original_file=original['path'],original_sha256=digest,original_step=step,displayed_step=displayed_step(step-1),
            character=character,room=room,reproduced_file=str(path),reproduced_sha256=sha(path),original_rgb_exact=True,
            original_png_bytes_equal=sha(path)==digest,wrapped_file=replacement['path'],wrapped_sha256=replacement['sha256'],**proof)
    with ThreadPoolExecutor(max_workers=4) as pool:proofs=dict(pool.map(check_original,sorted(unique.items())))
    cases=[];all_images=[]
    for case_index,record in enumerate(records):
        directory=work/'scenes'/record['sid'];directory.mkdir(parents=True)
        qa=Path(record['path'])/'qa.txt';newqa=directory/'qa.txt'
        with newqa.open('xb') as stream:stream.write(qa.read_bytes())
        need(sha(newqa)==record['qa_sha256'],'Copied physical QA changed')
        images=[]
        for i,(original,(character,room)) in enumerate(zip(record['image_files'],records_frames[record['sid']])):
            ref=wrapped_cache[character,room,displayed_step(i)];path=directory/f'{i:03d}.png';os.link(ref['path'],path);proof=proofs[original['sha256']]
            images.append(dict(ref,path=str(path),render_cache=ref['path'],v16_source_path=original['path'],v16_source_sha256=original['sha256'],
                v16_physical_step=i+1,v16_displayed_step=displayed_step(i),v16_source_rgb_sha256=proof['original_rgb_sha256'],
                v16_rgb_sha256=proof['wrapped_rgb_sha256'],v16_pixel_difference_bbox=proof['pixel_difference_bbox'],
                v16_changed_pixels=proof['changed_pixels'],v16_outside_footer_equal=True,v16_first16_identity=i<16))
        wrapped=dict(record,path=str(directory),source_path=str(directory),image_files=images,
            v16_source_path=record['path'],v16_source_sid=record['sid'],v16_intervention=PROTOCOL)
        need(newqa.read_bytes()==qa.read_bytes() and qa_states(wrapped)[1]==records_frames[record['sid']],'Published semantic/QA identity differs')
        cases.append(dict(source_record=record,wrapped_record=wrapped));all_images.extend(images)
        print(json.dumps(dict(staged=case_index+1,total=34,sid=record['sid'])),flush=True)
    need(len(all_images)==1632 and sum(im['v16_first16_identity'] for im in all_images)==544
        and sum(im['v16_changed_pixels']>0 for im in all_images)==1088,'Actual occurrence/intervention coverage differs')
    audit=dict(schema_version=1,passed=True,protocol=PROTOCOL,source_sha256=frozen,tests=tests,image_audits=proofs,
        cases=34,actual_image_occurrences=1632,unique_original_images=len(proofs),unique_wrapped_images=len(wrapped_cache),
        first16_identity_occurrences=544,changed_label_occurrences=1088,all_original_rgb_exact=True,
        all_original_png_bytes_equal=all(p['original_png_bytes_equal'] for p in proofs.values()),
        all_outside_footer_equal=True,all_QA_bytes_identical=True,all_semantic_counts_identical=True,
        references_untouched=True,reference_bank_sha256=object_sha(plan['reference_banks']),renderer_provenance=provenance)
    audit_path=work/'render_audit.json';save(audit_path,audit)
    manifest=dict(schema_version=1,purpose='counterfactual_step_wrap',protocol=PROTOCOL,policy=POLICY,dataset_root=str(DEST),
        source_v15_plan_file=str(SOURCE_PLAN),source_v15_plan_sha256=sha(SOURCE_PLAN),source_sha256=frozen,
        stage_plan_file=str(out/'stage_plan.json'),stage_plan_sha256=sha(out/'stage_plan.json'),
        render_audit_file=str(audit_path),render_audit_sha256=sha(audit_path),source_bindings=source_bindings,
        cases=cases,reference_banks=plan['reference_banks'],reference_banks_sha256=object_sha(plan['reference_banks']),
        splits={f'test_N{n}':dict(count=17,samples=[c['wrapped_record'] for c in cases if c['wrapped_record']['n_frames']==n]) for n in (32,64)},
        no_model_calls=True,no_gpu_calls=True,no_fit=True,limitation=POLICY['limitation'])
    for path,digest in source_bindings.items():need(sha(path)==digest,'Original input/reference changed during staging')
    need(sources()==frozen,'Staging source changed during execution')
    manifest_path=DEST/'main_manifest.json';save(manifest_path,manifest);(DEST/'main_manifest.sha256').write_text(sha(manifest_path)+'\n')
    save(out/'summary.json',dict(schema_version=1,protocol=PROTOCOL,passed=True,completed=True,source_sha256=frozen,
        manifest_file=str(manifest_path),manifest_sha256=sha(manifest_path),render_audit_file=str(audit_path),render_audit_sha256=sha(audit_path),
        source_v15_plan_file=str(SOURCE_PLAN),source_v15_plan_sha256=sha(SOURCE_PLAN),tests=tests,
        cases=34,actual_image_occurrences=1632,no_model_calls=True,no_gpu_calls=True,no_fit=True,seconds=time.perf_counter()-begin,
        slurm_job_id=os.environ['SLURM_JOB_ID']))
    verify_stage(out/'summary.json')
    index=DEST/'INDEX.md'
    if not index.exists():
        with index.open('x') as stream:stream.write('# V16 displayed-Step counterfactual\n\n[Manifest](main_manifest.json). Original QA/order/labels and all references remain fixed. Displayed ordinals repeat1–16.\n')


def verify_stage(summary_path):
    path=Path(summary_path).resolve();summary=read(path)
    need(path.is_relative_to(OUT) and summary['protocol']==PROTOCOL and summary['passed'] is True and summary['completed'] is True
        and summary['source_sha256']==sources() and summary['no_model_calls'] and summary['no_gpu_calls'],'Complete matching V16 CPU stage required')
    for name,digest in summary['source_sha256'].items():need(sha(path.parent/'code'/name.replace('/','_'))==digest,'Archived stage source differs')
    need(Path(summary['manifest_file']).resolve()==DEST/'main_manifest.json' and sha(summary['manifest_file'])==summary['manifest_sha256']
        and (DEST/'main_manifest.sha256').read_text().strip()==summary['manifest_sha256'],'Staged manifest/sidecar differs')
    manifest=read(summary['manifest_file']);need(manifest['policy']==POLICY and manifest['source_sha256']==summary['source_sha256']
        and manifest['protocol']==PROTOCOL and manifest['dataset_root']==str(DEST),'Manifest source/policy differs')
    need(Path(manifest['source_v15_plan_file']).resolve()==SOURCE_PLAN and sha(SOURCE_PLAN)==manifest['source_v15_plan_sha256']==summary['source_v15_plan_sha256'],
        'Canonical source V15 plan changed')
    source=read(SOURCE_PLAN);need(manifest['reference_banks']==source['reference_banks']
        and object_sha(manifest['reference_banks'])==manifest['reference_banks_sha256'],'Reference bank changed')
    need(sha(manifest['stage_plan_file'])==manifest['stage_plan_sha256']
        and Path(manifest['stage_plan_file']).with_suffix('.sha256').read_text().strip()==manifest['stage_plan_sha256'],'Frozen pre-render plan changed')
    need(sha(manifest['render_audit_file'])==manifest['render_audit_sha256']==summary['render_audit_sha256'],'Pixel audit changed')
    audit=read(manifest['render_audit_file']);need(audit['passed'] is True and audit['all_original_rgb_exact'] is True
        and audit['all_outside_footer_equal'] is True and audit['all_QA_bytes_identical'] is True
        and audit['references_untouched'] is True and audit['reference_bank_sha256']==manifest['reference_banks_sha256'],'Pixel/QA/reference proof failed')
    for original in audit['image_audits'].values():
        need(sha(original['reproduced_file'])==original['reproduced_sha256'] and sha(original['wrapped_file'])==original['wrapped_sha256']
            and original['original_rgb_exact'] is True and original['outside_footer_equal'] is True,'Rendered proof artifact changed')
    cases=manifest['cases'];need(len(cases)==34 and [c['source_record'] for c in cases]==[c['record'] for c in source['cases']],'Source scene selection/order changed')
    count=identity=changed=0
    for case in cases:
        old,new=case['source_record'],case['wrapped_record'];allowed={'path','source_path','image_files'}
        need(all(new[k]==value for k,value in old.items() if k not in allowed) and all(k in old or k.startswith('v16_') for k in new),
            'Nonvisual scene metadata changed')
        need(new['source_path']==new['path'] and Path(new['path']).is_relative_to(DEST) and len(new['image_files'])==old['n_frames'],
            'Wrapped path/image count differs')
        need((Path(new['path'])/'qa.txt').read_bytes()==(Path(old['path'])/'qa.txt').read_bytes(),'Physical QA changed')
        for i,(im,prior) in enumerate(zip(new['image_files'],old['image_files'])):
            proof=audit['image_audits'][prior['sha256']]
            need(im['v16_source_sha256']==prior['sha256'] and im['v16_source_path']==prior['path']
                and im['v16_physical_step']==i+1 and im['v16_displayed_step']==displayed_step(i)
                and im['sha256']==proof['wrapped_sha256'] and sha(im['path'])==im['sha256']
                and im['v16_pixel_difference_bbox']==proof['pixel_difference_bbox'] and im['v16_changed_pixels']==proof['changed_pixels']
                and im['v16_source_rgb_sha256']==proof['original_rgb_sha256'] and im['v16_rgb_sha256']==proof['wrapped_rgb_sha256']
                and im['v16_outside_footer_equal'] is True and im['v16_first16_identity']==(i<16),'Image occurrence/display/proof binding differs')
            need(Path(im['path']).samefile(im['render_cache']) and Path(im['render_cache']).resolve()==Path(proof['wrapped_file']).resolve()
                and Path(im['path'])==Path(new['path'])/f'{i:03d}.png' and Path(im['path']).stat().st_size==im['bytes']
                and im['mode']=='RGB' and im['dimensions']==[512,512],'Wrapped image path/render cache/shape differs')
            count+=1;identity+=i<16;changed+=im['v16_changed_pixels']>0
    need((count,identity,changed)==(1632,544,1088),'Complete wrapped occurrence denominator differs')
    for n in (32,64):
        expected=[c['wrapped_record'] for c in cases if c['wrapped_record']['n_frames']==n]
        need(manifest['splits'][f'test_N{n}']==dict(count=17,samples=expected) and Counter(r['gold'] for r in expected)==Counter(range(17)),
            'Wrapped test partition differs')
    for filename,digest in manifest['source_bindings'].items():need(sha(filename)==digest,'Original image/QA/reference changed')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check-stage',action='store_true',required=True);parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),
        'All rendering, pixel tests and staging require CPU Slurm')
    out=OUT/f'stage_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False);frozen=snapshot(out);stage(out,frozen)
    need(sources()==frozen,'Stage source changed after validation')
    print(json.dumps(dict(passed=True,summary_file=str(out/'summary.json'),manifest_file=str(DEST/'main_manifest.json'))),flush=True)


if __name__=='__main__':main()

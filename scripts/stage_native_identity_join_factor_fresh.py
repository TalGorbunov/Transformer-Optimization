"""Conditional CPU staging of the predeclared810 fresh factor-validation scenes.

No model/head/fit or inference release. An explicit passed native whole-answer
report is required. All exclusions are reconstructed from a finite registered
input inventory; generation never discovers files or changes that inventory.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import native_identity_join_factor_fresh_data as law
from scripts import stage_native_identity_join as symbolic
need=law.need;oid=law.oid;read=symbolic.read;save=symbolic.save
_READ_BINDINGS={}


def digest(path):
    return symbolic.digest(Path(path))
BASE=REPO/'outputs/native_aggregation_vlm';DATA_BASE=Path('/mnt/data/gabriele/gnn_transformer')
OUT=BASE/'identity_join_factor_fresh/data_staging';DATA_ROOT=DATA_BASE/'identity_join_factor_fresh'
PROTOCOL='native_identity_join_factor_fresh_data';SEED=91726341
PROPOSAL='docs/paper/NATIVE_AGGREGATION_FACTOR_FRESH_DATA_PROPOSAL.md'
PROPOSAL_SHA='4b47f0653581ed9e9a5602eba1f88c66e845ed4ff0e9f7e97c7e0f10b702a5a4'
LAW_SHA='77fc5ca203a9c6ac40b42cee0eaade8c167b1681e6615ef17d03f68518e34041'
V18_PLAN=BASE/'v18/data_staging/drycheck_442827/plan.json'
V18_PLAN_SHA='08faed89d9e28569c87027779d473237f040173565396ebf12b658f6f3df31dc'
UNIFORM_PLAN=BASE/'identity_join_uniform/check_443258/plan.json'
UNIFORM_PLAN_SHA='f87d6d49bab87e3baacb436faed88b6561a1dcd3a71b7ccefdebafe2d79c12d9'
STOPPED_PLAN=BASE/'identity_join/data_staging/check_442974/plan.json'
STOPPED_PLAN_SHA='cf1c153d399f73c9a5614f1cc3ee7e4efb5df8256216420d23b70d831d21f668'
SOFTWARE_PLAN=BASE/'identity_join_learned/software/check_443013/plan.json'
SOFTWARE_PLAN_SHA='c1814de78824c9562c4b38a213b87da3ec9a22aebc813148fb507c4a7454386c'
TIMING=BASE/'identity_join_optimization_v3/check_443214/timing_cases.json'
TIMING_SHA='cc563083174b949fc5a9849aca2c436850a7800c1b59580c380d28ef2276b167'
JOIN_MANIFEST=DATA_BASE/'identity_join_learned/stage_442988/main_manifest.json'
EXTRA_MANIFESTS={str(DATA_BASE/'v18_fresh/main_manifest.json'):'b153b1dfea5c11d4fd92b1109ffd235898a32559d3db45aeb4b20b768a164abe',
    str(JOIN_MANIFEST):'db2045c772a475eb36562c70018f0aa7641efcd6d71a5e3201c7da1bf1a0d94f'}
STAGE_PROOFS={str(BASE/'v18/data_staging/stage_442828/summary.json'):'05f1bfe47e2354a40f0a7016ef60ff769ae3b97313af005304d935b52f9931c9',
    str(BASE/'identity_join_learned/data_staging/render_442988/summary.json'):'d893bf436db50fdb98c25c45e841b01c209d8f77cd7d744b69ef68044bf3f2b3'}
POLICY=dict(protocol=PROTOCOL,seed=SEED,contexts=810,families=90,same_length_triples=270,image_occurrences=30240,
    lengths=[16,32,64],panels={'A':36,'B':36,'C':18},canonical_prior_manifests=32,
    additional_exposures=['stopped_binary_symbolic','learned_selection_software','optimization_timing'],
    global_and_local_question_identical=True,complete_scene_and_concrete_triple_exclusions=True,atomic_image_reuse_allowed=True,
    target_eos=151645,max_answer_tokens=4,cpu_check_seconds=300,cpu_render_seconds=600,cpu_cores=4,memory_gib=16,
    native_full_answer_report_required=True,no_model_or_head_calls=True,no_new_statistics_or_training=True,no_inference_release=True)
OWN=('scripts/native_identity_join_factor_fresh_data.py','scripts/stage_native_identity_join_factor_fresh.py',
     'slurm/native_identity_join_factor_fresh_check.sbatch','slurm/native_identity_join_factor_fresh_render.sbatch',PROPOSAL)


def bind(path,bindings,expected=None):
    path=Path(path);need(path.is_file() and not path.is_symlink(),'Missing canonical input: '+str(path))
    path=path.resolve();value=digest(path)
    need(expected is None or value==expected,'Input identity changed: '+str(path))
    need(str(path) not in bindings or bindings[str(path)]==value,'Conflicting input identity')
    bindings[str(path)]=value;_READ_BINDINGS[str(path)]=value;return value


def source_hashes():
    need(digest(REPO/OWN[0])==LAW_SHA and digest(REPO/PROPOSAL)==PROPOSAL_SHA,'Held law/protocol changed')
    return {**symbolic.source_hashes(),**{name:digest(REPO/name) for name in OWN}}


def snapshot(out):
    frozen=source_hashes();(out/'source').mkdir()
    for name,h in frozen.items():
        dst=out/'source'/name.replace('/','_');dst.write_bytes((REPO/name).read_bytes());need(digest(dst)==h,'Source snapshot changed')
    save(out/'source_hashes.json',frozen);return frozen


def content_rows(value):
    if isinstance(value,dict):
        if 'content_sha256' in value and 'n_frames' in value:yield value
        else:
            for item in value.values():yield from content_rows(item)
    elif isinstance(value,list):
        for item in value:yield from content_rows(item)


def qa_content(row,bindings,cache):
    directory=Path(row.get('path',row.get('source_path','')));path=directory/'qa.txt'
    need(str(directory) not in ('','.') and path.is_file(),'Historical published row has no actual QA path')
    if str(path) not in cache:
        bind(path,bindings,row['qa_sha256']);lines=path.read_text().splitlines()
        q=next(i for i,line in enumerate(lines) if line.strip()=='question:');a=next(i for i,line in enumerate(lines) if line.strip()=='answer:')
        need(a>q,'Historical QA sections differ');block=[line.strip() for line in lines[q+1:a] if line.strip()]
        states=[ast.literal_eval(line) for line in block if line.startswith('{') and line.endswith('}')]
        questions=[line for line in block if not (line.startswith('{') and line.endswith('}'))]
        need(len(questions)==1 and all(isinstance(v,dict) for v in states),'Historical QA state/question schema differs')
        cache[str(path)]=dict(question=' '.join(questions[0].split()),states=states)
    value=cache[str(path)]
    need(bindings[str(path)]==row['qa_sha256'] and len(value['states'])==row['n_frames']
         and oid(value)==row['content_sha256'],'Historical actual content/hash differs')
    if 'question' in row:need(' '.join(row['question'].split())==value['question'],'Declared historical question differs')
    return value


def manifest_inventory(bindings):
    for path,h in ((V18_PLAN,V18_PLAN_SHA),(UNIFORM_PLAN,UNIFORM_PLAN_SHA),(STOPPED_PLAN,STOPPED_PLAN_SHA),
                   (SOFTWARE_PLAN,SOFTWARE_PLAN_SHA),(TIMING,TIMING_SHA)):
        bind(path,bindings,h)
    v18=read(V18_PLAN);manifest_map=dict(v18['source_manifest_sha256'])
    need(len(manifest_map)==30,'Frozen V18 prior30 inventory differs')
    bind(v18['inventory_file'],bindings,v18['inventory_sha256'])
    need(read(v18['inventory_file'])['source_manifest_sha256']==manifest_map,'Prior30 inventory/plan mismatch')
    need(not set(EXTRA_MANIFESTS)&set(manifest_map),'Extra canonical manifest duplicated')
    manifest_map.update(EXTRA_MANIFESTS)
    for path,h in {**manifest_map,**STAGE_PROOFS}.items():bind(path,bindings,h)
    for path in STAGE_PROOFS:
        proof=read(path);need(proof['passed'] is True and proof['completed'] is True,'Historical stage did not pass')
    stopped=read(STOPPED_PLAN);bind(stopped['samples_file'],bindings,stopped['files'][stopped['samples_file']])
    uniform=read(UNIFORM_PLAN);bind(uniform['rows_file'],bindings,uniform['runtime_bindings'][uniform['rows_file']])
    need(uniform['timing_cases_file']==str(TIMING) and uniform['runtime_bindings'][str(TIMING)]==TIMING_SHA,'Timing cases lost fixed parent binding')
    support=[item['sample'] for item in read(uniform['rows_file'])['train']]
    need(len(support)==108,'Original pilot support differs')
    return manifest_map,stopped,support


def exclusions(out):
    bindings={};manifests,stopped,support=manifest_inventory(bindings)
    save(out/'exclusion_inputs.json',dict(manifests=manifests,input_bindings=bindings,
        additional_registered_files=[str(STOPPED_PLAN),stopped['samples_file'],str(SOFTWARE_PLAN),str(TIMING)],
        no_recursive_discovery=True))
    excluded=set();families=set();records=[];family_records=[];qa_cache={};join_by_sid={}
    def add(value,row,source,kind):
        h=oid(value);excluded.add(h)
        records.append(dict(source=source,kind=kind,sid=row.get('sid'),n_frames=len(value['states']),
            content_sha256=h,question=value['question'],qa_file=None if kind=='planned_symbolic' else row.get('_qa_file')))
        return dict(value,content_sha256=h,variant=row.get('variant'),contrast_id=row.get('contrast_id'))
    def triples(values,source,strict=False):
        groups=defaultdict(dict)
        for value in values:
            if value.get('contrast_id') is not None and value.get('variant') is not None:
                key=(value['contrast_id'],len(value['states']));variant=value['variant']
                if variant in groups[key]:need(groups[key][variant]['content_sha256']==value['content_sha256'],'Conflicting historical variant')
                groups[key][variant]=value
        for (label,n),variants in groups.items():
            if strict:need(set(variants)=={0,1,2},'Historical join family is not a complete same-length triple')
            if set(variants)=={0,1,2}:
                values=list(variants.values());h=law.family_identity(values);families.add(h)
                family_records.append(dict(source=source,contrast_id=label,n_frames=n,family_instance_sha256=h,
                    scene_sha256=sorted(v['content_sha256'] for v in values)))
    for path,h in manifests.items():
        manifest=read(path);values=[];count=0
        for row in content_rows(manifest):
            value=qa_content(row,bindings,qa_cache);r=dict(row,_qa_file=str(Path(row.get('path',row.get('source_path')))/'qa.txt'))
            values.append(add(value,r,path,'published_manifest'));count+=1
            if path==str(JOIN_MANIFEST):
                need(row['sid'] not in join_by_sid or join_by_sid[row['sid']]==row,'Conflicting canonical join SID');join_by_sid[row['sid']]=row
        need(count>0,'Empty registered canonical manifest');triples(values,path,strict=path==str(JOIN_MANIFEST))
        del manifest,values
    planned=read(stopped['samples_file']);need(len(planned)==6588,'Stopped planned-scene inventory differs')
    values=[]
    for row in planned:
        value=law.scene_content(row);need(oid(value)==row['content_sha256'],'Stopped planned content hash differs')
        need(row['sid'] in join_by_sid and row['content_sha256']==join_by_sid[row['sid']]['content_sha256'],
             'Stopped plan must remain represented exactly in the later canonical dataset')
        values.append(add(value,row,stopped['samples_file'],'planned_symbolic'))
    triples(values,stopped['samples_file'],strict=True)
    # Actual exposed profile inputs append canonical Step17..64 atoms. They do
    # not have a standalone QA, so reconstruct from their bound parent and atoms.
    for path,kind in ((SOFTWARE_PLAN,'software_extension'),(TIMING,'timing_extension')):
        payload=read(path);derivation=payload['case_derivation' if path==SOFTWARE_PLAN else 'derivation']
        atoms=derivation['appended_atoms'];need(len(atoms)==48,'Exposed extension atom count differs')
        appended=[]
        for i,item in enumerate(atoms,17):
            person,room,step=item['atom'];need(step==i and person in law.PEOPLE and room in law.ROOMS and room not in derivation['room_pair'],
                'Exposed extension is not a canonical irrelevant Step sequence')
            bind(item['path'],bindings,item['sha256']);appended.append(dict(step_id=i,rooms={room:[person]}))
        values=[]
        for case in payload['cases']:
            sample=case['sample'];parent_sid=derivation['parent_sid'] if path==SOFTWARE_PLAN else case['parent_sid'];parent=join_by_sid[parent_sid]
            source=qa_content(parent,bindings,qa_cache);n=sample['n_frames'];need(parent['n_frames']==16 and n in (16,32,64),'Derived native input length differs')
            expected_images=parent['image_files']+[{k:item[k] for k in ('path','sha256','dimensions','mode')} for item in atoms[:n-16]]
            need(sample['question']==source['question'] and sample['image_files']==expected_images and len(expected_images)==n,
                 'Actual exposed image order does not match the parent/extension')
            value=dict(question=source['question'],states=source['states']+appended[:n-16])
            r=dict(sample,variant=None if path==SOFTWARE_PLAN else case['variant'],contrast_id=None if path==SOFTWARE_PLAN else case['contrast_id'])
            values.append(add(value,r,str(path),kind))
        need(len(values)==(2 if path==SOFTWARE_PLAN else 9),'Exposed case inventory differs');triples(values,str(path),strict=path==TIMING)
    for sample in support:
        need(sample['sid'] in join_by_sid and sample==join_by_sid[sample['sid']],'Original pilot sample differs from canonical manifest')
    result=dict(passed=True,canonical_manifests=manifests,canonical_manifest_count=len(manifests),input_bindings=bindings,
        excluded_scene_sha256=sorted(excluded),excluded_family_sha256=sorted(families),scene_identities=len(excluded),family_identities=len(families),
        records=records,family_records=family_records,original_training_samples=support,
        actual_QA_reconstructed=True,exposed_planned_and_derived_contexts_included=True,
        atom_reuse_separate=True,no_recursive_discovery=True)
    save(out/'exclusion_inventory.json',result);return result


def native_gate(path):
    from scripts.report_native_identity_join_factor_native_v2 import verify_native_report
    path=Path(path).resolve();path=path/'summary.json' if path.is_dir() else path
    bindings={};bind(path,bindings);analysis=verify_native_report(path);summary=read(path)
    for key in ('analysis','report','plan'):bind(summary[key+'_file'],bindings,summary[key+'_sha256'])
    plan=read(analysis['plan_file']);bind(plan['rows_file'],bindings,plan['runtime_bindings'][plan['rows_file']])
    targets={}
    for row in read(plan['rows_file']):
        need(row['gold'] not in targets or targets[row['gold']]==row['target_ids'],'Native canonical targets conflict')
        targets[row['gold']]=row['target_ids']
    need(set(targets)==set(law.PEOPLE),'Native report lacks a canonical name')
    result=dict(file=str(path),sha256=bindings[str(path)],analysis_file=summary['analysis_file'],analysis_sha256=summary['analysis_sha256'],
        plan_file=summary['plan_file'],plan_sha256=summary['plan_sha256'],input_bindings=bindings,
        source_sha256=summary['source_sha256'],inherited_source_sha256=summary['inherited_source_sha256'],
        native_identity=plan['native_identity'],native_identity_sha256=plan['native_identity_sha256'],name_target_ids=targets,
        native_whole_answer_screen=analysis['native_whole_answer_screen'],both_complete_native_arms_verified=True,
        at_least_one_native_training_screen_passed=True)
    check_native_binding(result);return result


def check_native_binding(binding):
    symbolic.verify_bindings(binding['input_bindings'])
    for name,h in {**binding['source_sha256'],**binding['inherited_source_sha256']}.items():
        need(digest(REPO/name)==h,'Native report source changed: '+name)
    summary=read(binding['file']);analysis=read(binding['analysis_file']);plan=read(binding['plan_file'])
    need(summary['passed'] is summary['completed'] is True and summary['phase']=='report'
         and summary['analysis_file']==binding['analysis_file'] and summary['analysis_sha256']==binding['analysis_sha256']
         and summary['source_sha256']==binding['source_sha256'] and summary['inherited_source_sha256']==binding['inherited_source_sha256']
         and summary['native_whole_answer_screen']==analysis['native_whole_answer_screen']==binding['native_whole_answer_screen']
         and summary['at_least_one_native_training_screen_passed'] is True
         and plan['native_identity']==binding['native_identity'] and oid(plan['native_identity'])==binding['native_identity_sha256'],
         'Bound full native training report changed')


def runtime_view(record):
    return dict(sid=record['sid'],n_frames=record['n_frames'],question=record['question'],
        image_files=[{k:image[k] for k in ('path','sha256','dimensions','mode')} for image in record['image_files']])


def self_test():
    tests=law.self_test()
    safe=dict(sid='fixture',n_frames=1,question=symbolic.global_prompt(('Kitchen','Office')),
        image_files=[dict(path='/fixture.png',sha256='0'*64,dimensions=[512,512],mode='RGB')])
    contaminated=dict(safe,gold='Sandra',states=['offline'],trio=['Sandra'],local_prompt='forbidden',variant=0)
    need(runtime_view(safe)==runtime_view(contaminated) and set(runtime_view(safe))=={'sid','n_frames','question','image_files'},
         'Runtime allowlist leaked labels or the stopped binary prompt')
    return dict(passed=True,symbolic=tests,strict_runtime_allowlist=True)


def tokenizer_contract():
    value=symbolic.tokenizer_contract();value.pop('count_token_ids')
    # The immutable helper also measures the old wrapper. None of that wrapper
    # metadata enters this new dataset: both actual prompts are the question.
    for counts in value['prompt_lengths'].values():counts['local_tokens']=counts['global_tokens']
    value['local_prompt_is_global_question']=True;return value


def check(args,out,frozen):
    gate=native_gate(args.native_report);save(out/'native_report_binding.json',gate)
    tests=self_test();inventory=exclusions(out);support=inventory['original_training_samples']
    scenes=set(inventory['excluded_scene_sha256']);families=set(inventory['excluded_family_sha256'])
    rows=law.generate(support,scenes,families);audit=law.audit(rows,support,scenes,families)
    need(rows==law.generate(support,scenes,families),'Fixed-seed fresh regeneration differs')
    save(out/'samples.json',rows)
    tokens=tokenizer_contract();need({name:item['target_ids'] for name,item in tokens['targets'].items()}==gate['name_target_ids']
        and tokens['model_directory']==gate['native_identity']['model']['path'],'Native tokenizer/name/model binding differs')
    reuse=symbolic.reuse_inventory(rows)
    targets=[dict(sid=row['sid'],target_ids=tokens['targets'][row['gold']]['target_ids']) for row in rows]
    positions={f'fresh_{panel}_N{n}':sum(len(t['target_ids']) for row,t in zip(rows,targets) if row['panel']==panel and row['n_frames']==n)
        for panel in ('A','B','C') for n in law.LENGTHS}
    need(sum(positions.values())==1800,'Balanced name-plus-EOS token inventory differs')
    for name,value in (('targets.json',targets),('tokenizer.json',tokens),('reuse_inventory.json',reuse),('symbolic_audit.json',audit),('selftests.json',tests)):
        save(out/name,value)
    names=('native_report_binding.json','exclusion_inputs.json','exclusion_inventory.json','samples.json','targets.json','tokenizer.json','reuse_inventory.json','symbolic_audit.json','selftests.json')
    files={str(out/name):digest(out/name) for name in names}
    bindings={}
    for group in (inventory['input_bindings'],reuse['input_bindings'],tokens['files'],gate['input_bindings']):
        for path,h in group.items():
            need(path not in bindings or bindings[path]==h,'Conflicting stage input');bindings[path]=h
    symbolic.verify_bindings(bindings);check_native_binding(gate)
    plan=dict(schema_version=1,protocol=PROTOCOL,policy=POLICY,source_sha256=frozen,source_directory=str(out/'source'),files=files,
        samples_file=str(out/'samples.json'),targets_file=str(out/'targets.json'),tokenizer_file=str(out/'tokenizer.json'),
        reuse_inventory_file=str(out/'reuse_inventory.json'),exclusion_inventory_file=str(out/'exclusion_inventory.json'),
        native_report=gate,symbolic_audit=audit,selftests=tests,data_root=str(DATA_ROOT),target_positions_per_cell=positions,
        input_bindings=bindings,all_semantic_scenes_and_targets_frozen=True,no_model_loaded=True,no_images_rendered=True,no_inference_release=True)
    save(out/'plan.json',plan)
    return dict(passed=True,completed=True,check_only=True,protocol=PROTOCOL,plan_file=str(out/'plan.json'),plan_sha256=digest(out/'plan.json'),
        source_sha256=frozen,native_report=gate,counts=audit,distinct_atoms=reuse['distinct_atoms'],reused_atoms=reuse['reused_atoms'],
        new_atoms=reuse['new_atoms'],target_positions_per_cell=positions,excluded_scenes=inventory['scene_identities'],
        excluded_concrete_triples=inventory['family_identities'],all_tokenizer_targets_fit=True,no_model_loaded=True,no_images_rendered=True,no_inference_release=True)


def verify_plan(path):
    path=Path(path).resolve();plan=read(path);summary=read(path.parent/'summary.json')
    need(summary['passed'] is summary['completed'] is True and summary['check_only'] is True and summary['plan_file']==str(path)
        and summary['plan_sha256']==digest(path) and summary['source_sha256']==plan['source_sha256']==source_hashes()
        and plan['protocol']==PROTOCOL and plan['policy']==POLICY and plan['data_root']==str(DATA_ROOT),'Passed exact fresh CPU plan required')
    for name,h in plan['source_sha256'].items():need(digest(Path(plan['source_directory'])/name.replace('/','_'))==h,'CPU source snapshot changed')
    symbolic.verify_bindings(plan['files']);symbolic.verify_bindings(plan['input_bindings']);check_native_binding(plan['native_report'])
    inventory=read(plan['exclusion_inventory_file']);rows=read(plan['samples_file'])
    need(law.audit(rows,inventory['original_training_samples'],inventory['excluded_scene_sha256'],inventory['excluded_family_sha256'])==plan['symbolic_audit'],
         'Frozen symbolic law/exclusions differ')
    return plan


def audit_published(rows,records,cache,root,tokens):
    from PIL import Image
    for entry in cache.values():
        path=Path(entry['path']);need(path.resolve().is_relative_to(root.resolve()) and not path.is_symlink(),'Image cache escaped immutable attempt')
        need(digest(path)==entry['sha256'],'Published PNG bytes differ')
        with Image.open(path) as picture:
            need(picture.mode=='RGB' and picture.size==(512,512) and hashlib.sha256(picture.tobytes()).hexdigest()==entry['rgb_sha256'],'Published RGB pixels differ')
    need(len(rows)==len(records)==810,'Published fresh scene count differs')
    for row,record in zip(rows,records):
        directory=Path(record['path']);qa=directory/'qa.txt'
        need(directory.resolve().is_relative_to(root.resolve()) and not directory.is_symlink() and not qa.is_symlink(),'Scene escaped immutable attempt')
        need(digest(qa)==record['qa_sha256'],'Published QA bytes differ');lines=qa.read_text().splitlines()
        need(lines[0]=='question:' and lines[-2]=='answer:' and lines[-1]==row['gold'] and lines[-3]==row['question']
             and [ast.literal_eval(line) for line in lines[1:-3]]==row['states'],'Published QA semantics/question/name differ')
        need(all(record[k]==v for k,v in row.items() if k not in ('frames','states')) and record['local_prompt']==record['question']
             and record['target_ids']==tokens['targets'][row['gold']]['target_ids'],'Published semantic metadata/targets differ')
        need(len(record['image_files'])==row['n_frames'],'Published image count differs')
        for i,((person,room),image) in enumerate(zip(row['frames'],record['image_files'])):
            path=directory/f'{i:03d}.png';entry=cache[symbolic.atom_key((person,room,i+1))]
            need(not path.is_symlink() and image==dict(path=str(path),sha256=entry['sha256'],dimensions=[512,512],mode='RGB')
                 and os.path.samefile(path,entry['path']),'Image descriptor/order/canonical inode differs')
        need(set(runtime_view(record))=={'sid','n_frames','question','image_files'},'Runtime inputs carry labels')
    return dict(passed=True,contexts=len(records),image_occurrences=sum(r['n_frames'] for r in rows),unique_pngs=len(cache),
        all_QA_states_names_targets_verified=True,all_RGB_hashes_verified=True,all_image_order_inodes_verified=True,runtime_input_allowlist_verified=True)


def render(args,out,frozen):
    need(args.plan is not None,'Render requires a passed frozen CPU plan');plan=verify_plan(args.plan)
    gate=native_gate(args.native_report);need(gate==plan['native_report'],'Render must use the same complete native report')
    root=DATA_ROOT/f'stage_{os.environ["SLURM_JOB_ID"]}'
    need(not root.exists() and root.resolve().is_relative_to(DATA_ROOT.resolve()),'New authorized immutable data attempt required')
    root.mkdir(parents=True,exist_ok=False)
    release=dict(protocol=PROTOCOL,source_sha256=frozen,plan_file=str(args.plan.resolve()),plan_sha256=digest(args.plan),native_report=gate,data_root=str(root))
    save(out/'render_release.json',release);save(root/'attempt.json',release)
    rows=read(plan['samples_file']);inventory=read(plan['reuse_inventory_file']);tokens=read(plan['tokenizer_file'])
    need(tokenizer_contract()==tokens,'Actual tokenizer/runtime changed')
    renderer,provenance=symbolic.prepare_renderer(inventory['renderer_references'])
    provenance['method']='Canonical RGB rendering; every reused atom independently rerendered and compared'
    (root/'render_cache').mkdir();(root/'scenes').mkdir()
    def publish(row):
        value=symbolic.publish(row,root,cache,tokens['targets']);value['origin']='canonical_identity_join_factor_fresh';return value
    with ThreadPoolExecutor(max_workers=4) as pool:
        cache=dict(pool.map(lambda entry:symbolic.render_atom(entry,root/'render_cache',renderer),inventory['entries']))
        save(root/'render_cache.json',cache);records=list(pool.map(publish,rows))
    actual=audit_published(rows,records,cache,root,tokens)
    save(root/'runtime_inputs.json',dict(schema_version=1,protocol=PROTOCOL,records=[runtime_view(row) for row in records],
        note='Only the four allowlisted fields enter native preprocessing; all semantic metadata and labels remain offline'))
    save(root/'samples.json',rows);save(root/'audit.json',dict(symbolic=plan['symbolic_audit'],published=actual,renderer=provenance))
    splits={}
    for panel,count in (('A',108),('B',108),('C',54)):
        for n in law.LENGTHS:
            group=[row for row in records if row['panel']==panel and row['n_frames']==n];need(len(group)==count,'Final panel/length count differs')
            splits[f'fresh_{panel}_N{n}']=dict(count=count,samples=group,answer_histogram=dict(Counter(row['gold'] for row in group)))
    manifest=dict(schema_version=1,protocol=PROTOCOL,purpose='identity_join_factor_fresh',seed=SEED,policy=POLICY,dataset_root=str(root),
        splits=splits,source_sha256=frozen,plan_file=release['plan_file'],plan_sha256=release['plan_sha256'],native_report=gate,
        tokenizer=tokens,renderer=provenance,target_positions_per_cell=plan['target_positions_per_cell'],
        excluded_inventory_file=plan['exclusion_inventory_file'],excluded_inventory_sha256=plan['files'][plan['exclusion_inventory_file']],
        no_model_loaded=True,no_fit_released=True,no_inference_release=True)
    for key in ('runtime_inputs','samples','audit','render_cache'):
        manifest[key+'_file']=str(root/(key+'.json'));manifest[key+'_sha256']=digest(root/(key+'.json'))
    save(root/'main_manifest.json',manifest);symbolic.verify_bindings(plan['input_bindings']);check_native_binding(gate)
    return dict(passed=True,completed=True,check_only=False,protocol=PROTOCOL,source_sha256=frozen,
        manifest_file=str(root/'main_manifest.json'),manifest_sha256=digest(root/'main_manifest.json'),plan_file=release['plan_file'],
        plan_sha256=release['plan_sha256'],native_report=gate,dataset_root=str(root),counts=plan['symbolic_audit'],published_audit=actual,
        distinct_atoms=len(cache),reused_atoms=sum(item['reused'] for item in cache.values()),new_atoms=sum(not item['reused'] for item in cache.values()),
        excluded_scenes=len(read(plan['exclusion_inventory_file'])['excluded_scene_sha256']),
        no_model_loaded=True,no_fit_released=True,no_inference_release=True)


def verify_stage(path):
    path=Path(path).resolve();path=path/'summary.json' if path.is_dir() else path;summary=read(path)
    need(summary['passed'] is summary['completed'] is True and summary['check_only'] is False and summary['protocol']==PROTOCOL
         and summary['source_sha256']==source_hashes(),'Complete current fresh rendered-stage proof required')
    for name,h in summary['source_sha256'].items():need(digest(path.parent/'source'/name.replace('/','_'))==h,'Render source snapshot changed')
    need(digest(summary['manifest_file'])==summary['manifest_sha256'],'Published manifest changed');manifest=read(summary['manifest_file'])
    need(manifest['source_sha256']==summary['source_sha256'] and manifest['protocol']==PROTOCOL and manifest['policy']==POLICY
         and manifest['plan_file']==summary['plan_file'] and manifest['plan_sha256']==summary['plan_sha256']
         and digest(manifest['plan_file'])==manifest['plan_sha256'] and manifest['native_report']==summary['native_report'],'Published provenance differs')
    plan=read(manifest['plan_file']);proof=read(Path(manifest['plan_file']).parent/'summary.json')
    need(plan['source_sha256']==summary['source_sha256'] and plan['native_report']==summary['native_report']
         and proof['passed'] is proof['completed'] is proof['check_only'] is True and proof['plan_sha256']==manifest['plan_sha256'],
         'Passed CPU plan source/report differs')
    check_native_binding(manifest['native_report'])
    for key in ('samples','runtime_inputs','audit','render_cache','excluded_inventory'):
        need(digest(manifest[key+'_file'])==manifest[key+'_sha256'],'Published '+key+' changed')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__);actions=parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--check',action='store_true');actions.add_argument('--render',action='store_true')
    parser.add_argument('--plan',type=Path);parser.add_argument('--native-report',type=Path,required=True);args=parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS')
         and int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==4,'Only a four-core Slurm CPU allocation may stage fresh data')
    need(not args.check or args.plan is None,'CPU check cannot consume an alternate plan')
    mode='check' if args.check else 'render';out=OUT/f'{mode}_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();frozen=snapshot(out)
    save(out/'request.json',dict(mode=mode,plan=str(args.plan) if args.plan else None,native_report=str(args.native_report),source_sha256=frozen))
    try:
        result=check(args,out,frozen) if args.check else render(args,out,frozen)
        need(source_hashes()==frozen,'Sources changed during staging');elapsed=time.perf_counter()-started
        need(elapsed<=POLICY['cpu_'+mode+'_seconds'],'Registered CPU time cap exceeded')
        result.update(seconds=elapsed,slurm_job_id=os.environ['SLURM_JOB_ID'])
        with (out/'REPORT.md').open('x') as stream:
            stream.write('# Fresh factor validation data\n\n'+('All 810 semantic scenes and native name targets are frozen; rendering remains a separate release.' if args.check else
                'All 810 scenes passed canonical rendering and independent publication checks. Inference remains a separate release.')+'\n\n'+
                json.dumps({key:result[key] for key in ('counts','distinct_atoms','reused_atoms','new_atoms')},indent=2)+'\n')
        save(out/'summary.json',result);print(json.dumps(dict(passed=True,directory=str(out))),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,
            seconds=time.perf_counter()-started,accessed_input_bindings=_READ_BINDINGS,partial_outputs_retained=True));raise


if __name__=='__main__':main()

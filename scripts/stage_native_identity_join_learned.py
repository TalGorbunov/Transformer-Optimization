"""A separate prospective data release for learned identity-join selection.

The stopped binary-inclusion renderer is never invoked or modified. Reuse only
its passed CPU symbolic plan, tokenizer targets and canonical pixel inventory.
Every new local stream receives the unaltered global question; there is no
binary instruction, measurement, gate prerequisite or reused hidden feature.
All rendering and substantive checking require Slurm CPU. No fit is released.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import stage_native_identity_join as symbolic
from scripts.stage_native_vision_pilot import prepare_renderer
need=symbolic.need
read=symbolic.read
save=symbolic.save
digest=symbolic.digest
oid=symbolic.oid
CHARS=symbolic.CHARS
PARK_ROOMS=symbolic.PARK_ROOMS
SEED=20261122
PROTOCOL='native_identity_join_learned_selection_data'
DATA_ROOT=Path('/mnt/data/gabriele/gnn_transformer/identity_join_learned')
OUT=REPO/'outputs/native_aggregation_vlm/identity_join_learned/data_staging'
PARENT_PLAN=REPO/'outputs/native_aggregation_vlm/identity_join/data_staging/check_442974/plan.json'
PARENT_PLAN_SHA256='cf1c153d399f73c9a5614f1cc3ee7e4efb5df8256216420d23b70d831d21f668'
POLICY={k:v for k,v in symbolic.POLICY.items() if k not in ('protocol','local_template','complete_gate_judgments_required','source_only_before_gate','scalar_count')}
POLICY.update(protocol=PROTOCOL,local_template='{question}',local_question_identical_to_global=True,
    binary_measurement=False,binary_gate_dependency=False,old_binary_route_stopped=True,
    selection_modes=['clip','sigmoid','softmax'],no_feature_reuse=True,no_fit_released=True,
    reuse='Exact CPU442974 semantic rows and canonical image atoms; original inclusion-wrapper metadata never enters model inputs')
OWN=('scripts/stage_native_identity_join_learned.py','slurm/native_identity_join_learned_check.sbatch',
     'slurm/native_identity_join_learned_render.sbatch')


def source_hashes():
    return {**symbolic.source_hashes(),**{name:digest(REPO/name) for name in OWN}}


def snapshot(out):
    frozen=source_hashes();(out/'source').mkdir()
    for name,expected in frozen.items():
        data=(REPO/name).read_bytes();need(hashlib.sha256(data).hexdigest()==expected,'Source changed during snapshot')
        with (out/'source'/name.replace('/','_')).open('xb') as stream:stream.write(data)
    save(out/'source_hashes.json',frozen)
    return frozen


def global_prompt(room_pair):
    return symbolic.global_prompt(room_pair)


def local_prompt(question):
    need(isinstance(question,str) and bool(question),'Complete global question required')
    return question


def transform_rows(parent_rows):
    return [dict(row,local_prompt=local_prompt(row['question'])) for row in parent_rows]


def verify_transform(rows,parent_rows):
    """Exact transformation of already independently audited semantic records."""
    need(len(rows)==len(parent_rows),'Symbolic transformation changed cardinality')
    for row,parent in zip(rows,parent_rows):
        need(row==dict(parent,local_prompt=parent['question']),'Transformation changed more than the local input text')
        need(row['question']==global_prompt(row['room_pair']) and row['local_prompt']==row['question'],
             'New local/global question contract differs')
        need('inclusion restriction' not in row['local_prompt'] and 'exactly 1 for yes' not in row['local_prompt'],
             'Stopped binary wrapper entered new model text')
    return dict(passed=True,contexts=len(rows),only_local_prompt_metadata_changed=True,
                exact_parent_semantics_states_answers_slots_backgrounds=True,
                local_and_global_question_identical=True,no_binary_wrapper_in_model_text=True)


def runtime_view(record):
    return dict(sid=record['sid'],n_frames=record['n_frames'],question=record['question'],
        image_files=[{key:image[key] for key in ('path','sha256','dimensions','mode')} for image in record['image_files']])


def selftest():
    parent=symbolic.make_family('train',symbolic.TRAIN_PAIRS[0],(0,1,2),(8,16),0,0,set())
    symbolic.audit_family(parent);rows=transform_rows(parent);verify_transform(rows,parent)
    rejected=0
    for field,value in (('gold','Noah'),('question','Which person?'),('local_prompt',parent[0]['local_prompt']),('parent_positions',[0])):
        bad=copy.deepcopy(rows);bad[0][field]=value
        try:verify_transform(bad,parent)
        except ValueError:rejected+=1
        else:raise AssertionError('Invalid symbolic/input transformation accepted: '+field)
    mock=dict(sid='opaque',n_frames=1,question=rows[0]['question'],image_files=[dict(path='/tmp/opaque.png',sha256='a'*64,dimensions=[512,512],mode='RGB')])
    poisoned=dict(mock,gold='Sandra',local_prompt='binary wrapper',trio=['Sandra'],variant=2,states=['offline'])
    view=runtime_view(poisoned)
    need(view==runtime_view(mock) and set(view)=={'sid','n_frames','question','image_files'},'Offline labels or wrapper entered runtime inputs')
    need(all(set(image)=={'path','sha256','dimensions','mode'} for image in view['image_files']),'Image descriptors contain offline labels')
    return dict(passed=True,groups=6,rejected_corruptions=rejected,
        tests=['valid_exact_transform','changed_answer','changed_global_question','old_local_wrapper','changed_insertion','strict_runtime_input_boundary'])


def parent_plan():
    need(digest(PARENT_PLAN)==PARENT_PLAN_SHA256,'Registered CPU442974 symbolic plan changed')
    # This checks the old CPU plan only. It neither calls its renderer nor asks
    # the failed binary gate to authorize this separately registered protocol.
    plan=symbolic.verify_plan(PARENT_PLAN)
    need(plan['symbolic_audit']['passed'] and plan['symbolic_audit']['contexts']==6588
         and plan['target_positions_per_split']==dict(train=13440,dev=240,test_seen=480,test_held=480),
         'Passed symbolic/name-target inventory differs')
    return plan


def tokenizer_metadata(parent):
    tokens=copy.deepcopy(read(parent['tokenizer_file']))
    # Tokenizer/name targets are unchanged. Both literal prompts are now q,
    # whose independently tokenized length is already stored as global_tokens.
    for value in tokens['prompt_lengths'].values():value['local_tokens']=value['global_tokens']
    tokens['local_question_identical_to_global']=True
    return tokens


def source_inputs(parent):
    return {str(PARENT_PLAN):PARENT_PLAN_SHA256,
            str(PARENT_PLAN.parent/'summary.json'):digest(PARENT_PLAN.parent/'summary.json'),
            **parent['files'],**parent['input_bindings']}


def check(out,frozen):
    tests=selftest();parent=parent_plan();parents=read(parent['samples_file'])
    rows=transform_rows(parents);transformation=verify_transform(rows,parents)
    need(rows==transform_rows(symbolic.generate()),'Deterministic semantic regeneration differs')
    audit=dict(parent['symbolic_audit']);audit.pop('scalar_count_identical_in_every_contrast',None)
    tokens=tokenizer_metadata(parent);inventory=read(parent['reuse_inventory_file'])
    save(out/'samples.json',rows);save(out/'symbolic_audit.json',dict(semantic=audit,transformation=transformation))
    save(out/'tokenizer.json',tokens);save(out/'reuse_inventory.json',inventory);save(out/'selftests.json',tests)
    files={str(out/name):digest(out/name) for name in ('samples.json','symbolic_audit.json','tokenizer.json','reuse_inventory.json','selftests.json')}
    plan=dict(schema_version=1,protocol=PROTOCOL,policy=POLICY,source_sha256=frozen,source_directory=str(out/'source'),
        parent_plan_file=str(PARENT_PLAN),parent_plan_sha256=PARENT_PLAN_SHA256,
        files=files,input_bindings=source_inputs(parent),samples_file=str(out/'samples.json'),
        tokenizer_file=str(out/'tokenizer.json'),reuse_inventory_file=str(out/'reuse_inventory.json'),
        symbolic_audit=audit,transformation=transformation,selftests=tests,data_root=str(DATA_ROOT),
        target_positions_per_split=parent['target_positions_per_split'],
        binary_gate_dependency=False,no_model_loaded=True,no_images_rendered=True,no_fit_released=True)
    save(out/'plan.json',plan)
    return dict(passed=True,completed=True,check_only=True,plan_file=str(out/'plan.json'),plan_sha256=digest(out/'plan.json'),
        parent_plan_file=str(PARENT_PLAN),parent_plan_sha256=PARENT_PLAN_SHA256,source_sha256=frozen,counts=audit,
        transformation=transformation,target_positions_per_split=plan['target_positions_per_split'],
        distinct_atoms=inventory['distinct_atoms'],reused_atoms=inventory['reused_atoms'],new_atoms=inventory['new_atoms'],
        all_tokenizer_targets_fit=True,binary_gate_dependency=False,no_model_loaded=True,no_images_rendered=True,no_fit_released=True)


def verify_plan(path):
    path=Path(path).resolve();plan=read(path);summary=read(path.parent/'summary.json')
    need(summary['passed'] is True and summary['completed'] is True and summary['check_only'] is True
         and summary['plan_file']==str(path) and summary['plan_sha256']==digest(path),'Passed exact new CPU plan required')
    need(plan['protocol']==PROTOCOL and oid(plan['policy'])==oid(POLICY) and plan['source_sha256']==source_hashes()
         and summary['source_sha256']==source_hashes() and plan['data_root']==str(DATA_ROOT)
         and plan['binary_gate_dependency'] is False,'New plan source/policy/input identity differs')
    for name,h in plan['source_sha256'].items():
        need(digest(Path(plan['source_directory'])/name.replace('/','_'))==h,'New CPU source snapshot changed')
    symbolic.verify_bindings(plan['files']);symbolic.verify_bindings(plan['input_bindings'])
    parent=parent_plan();rows=read(plan['samples_file'])
    need(plan['parent_plan_file']==str(PARENT_PLAN) and plan['parent_plan_sha256']==PARENT_PLAN_SHA256,'Parent CPU identity differs')
    need(verify_transform(rows,read(parent['samples_file']))==plan['transformation']
         and rows==transform_rows(symbolic.generate()),'New symbolic/input law changed')
    need(tokenizer_metadata(parent)==read(plan['tokenizer_file']),'Native tokenizer/name targets changed')
    return plan


def publish(row,root,cache,targets):
    result=symbolic.publish(row,root,cache,targets)
    result['origin']='canonical_identity_join_learned_selection'
    return result


def audit_published(rows,records,cache,root,parent_rows):
    # The inherited helper checks only saved QA/state/name/image bytes, order
    # and hardlink ownership. The new four-key input boundary is checked here.
    image_audit=symbolic.audit_published(rows,records,cache,root)
    transformation=verify_transform(rows,parent_rows)
    for row,record in zip(rows,records):
        need(record['local_prompt']==record['question']==row['question'],'Published local text differs from q')
        view=runtime_view(record)
        need(set(view)=={'sid','n_frames','question','image_files'} and view['question']==row['question'],
             'Published runtime input contains labels or legacy local wrapper')
    return dict(passed=True,images=image_audit,transformation=transformation,
                exact_four_key_model_input_view=True,no_binary_wrapper_or_probe=True)


def render(args,out,frozen):
    need(args.plan is not None,'Rendering requires the passed new prospective CPU plan')
    plan=verify_plan(args.plan);parent=read(PARENT_PLAN)
    root=DATA_ROOT/f'stage_{os.environ["SLURM_JOB_ID"]}'
    need(root.resolve().is_relative_to(DATA_ROOT.resolve()) and not root.exists(),'New authorized data attempt required')
    root.mkdir(parents=True,exist_ok=False)
    release=dict(protocol=PROTOCOL,plan_file=str(args.plan.resolve()),plan_sha256=digest(args.plan),
        parent_plan_file=str(PARENT_PLAN),parent_plan_sha256=PARENT_PLAN_SHA256,data_root=str(root),
        local_prompt='unaltered complete global question',binary_gate_dependency=False,source_sha256=frozen)
    save(out/'render_release.json',release);save(root/'attempt.json',release)
    rows=read(plan['samples_file']);inventory=read(plan['reuse_inventory_file']);tokens=read(plan['tokenizer_file'])
    renderer,provenance=prepare_renderer(inventory['renderer_references'])
    provenance['method']='Unchanged canonical RGB rendering; every reused atom independently rerendered and compared'
    (root/'render_cache').mkdir();(root/'scenes').mkdir()
    with ThreadPoolExecutor(max_workers=4) as pool:
        cache=dict(pool.map(lambda entry:symbolic.render_atom(entry,root/'render_cache',renderer),inventory['entries']))
        save(root/'render_cache.json',cache)
        records=list(pool.map(lambda row:publish(row,root,cache,tokens['targets']),rows))
    actual=audit_published(rows,records,cache,root,read(parent['samples_file']))
    views=[runtime_view(record) for record in records]
    save(root/'runtime_inputs.json',dict(schema_version=1,protocol=PROTOCOL,records=views,
        input_sha256={view['sid']:oid(view) for view in views},
        note='Each record is exactly the four-key model input; the runtime supplies its unchanged question to every local/global stream'))
    save(root/'samples.json',rows);save(root/'audit.json',dict(symbolic=plan['symbolic_audit'],published=actual,renderer=provenance))
    splits={}
    for cell,count in symbolic.CELL_COUNTS.items():
        group=[r for r in records if f"{r['split']}_N{r['n_frames']}"==cell]
        need(len(group)==count,'Final cell size differs')
        splits[cell]=dict(count=count,samples=group,answer_histogram=dict(Counter(r['gold'] for r in group)))
    manifest=dict(schema_version=1,protocol=PROTOCOL,purpose='identity_join_learned_selection',seed=SEED,policy=POLICY,
        dataset_root=str(root),splits=splits,source_sha256=frozen,plan_file=str(args.plan.resolve()),plan_sha256=digest(args.plan),
        parent_plan_file=str(PARENT_PLAN),parent_plan_sha256=PARENT_PLAN_SHA256,
        tokenizer=tokens,renderer=provenance,runtime_inputs_file=str(root/'runtime_inputs.json'),runtime_inputs_sha256=digest(root/'runtime_inputs.json'),
        samples_file=str(root/'samples.json'),samples_sha256=digest(root/'samples.json'),
        target_positions_per_split=plan['target_positions_per_split'],audit_file=str(root/'audit.json'),audit_sha256=digest(root/'audit.json'),
        render_cache_file=str(root/'render_cache.json'),render_cache_sha256=digest(root/'render_cache.json'),
        binary_gate_dependency=False,no_model_loaded=True,no_feature_reuse=True,no_fit_released=True)
    save(root/'main_manifest.json',manifest);symbolic.verify_bindings(plan['input_bindings'])
    return dict(passed=True,completed=True,check_only=False,protocol=PROTOCOL,source_sha256=frozen,
        manifest_file=str(root/'main_manifest.json'),manifest_sha256=digest(root/'main_manifest.json'),
        plan_file=str(args.plan.resolve()),plan_sha256=digest(args.plan),parent_plan_file=str(PARENT_PLAN),parent_plan_sha256=PARENT_PLAN_SHA256,
        dataset_root=str(root),counts=plan['symbolic_audit'],published_audit=actual,distinct_atoms=len(cache),
        reused_atoms=sum(e['reused'] for e in cache.values()),new_atoms=sum(not e['reused'] for e in cache.values()),
        input_bindings=plan['input_bindings'],binary_gate_dependency=False,no_model_loaded=True,no_feature_reuse=True,no_fit_released=True)


def verify_stage(path):
    path=Path(path).resolve();path=path/'summary.json' if path.is_dir() else path;summary=read(path)
    need(summary['passed'] is True and summary['completed'] is True and summary['check_only'] is False
         and summary['protocol']==PROTOCOL and summary['binary_gate_dependency'] is False,'Complete new learned-selection stage required')
    need(summary['source_sha256']==source_hashes(),'New dataset sources changed')
    for name,h in summary['source_sha256'].items():need(digest(path.parent/'source'/name.replace('/','_'))==h,'Stage source snapshot differs')
    need(digest(Path(summary['manifest_file']))==summary['manifest_sha256'],'Published manifest changed');manifest=read(summary['manifest_file'])
    need(manifest['protocol']==PROTOCOL and manifest['source_sha256']==summary['source_sha256']
         and manifest['binary_gate_dependency'] is False and oid(manifest['policy'])==oid(POLICY),'New manifest policy differs')
    for key in ('plan','parent_plan'):
        need(manifest[key+'_file']==summary[key+'_file'] and manifest[key+'_sha256']==summary[key+'_sha256']
             and digest(Path(manifest[key+'_file']))==manifest[key+'_sha256'],'Bound '+key+' changed')
    need(manifest['parent_plan_file']==str(PARENT_PLAN) and manifest['parent_plan_sha256']==PARENT_PLAN_SHA256,'Registered symbolic parent differs')
    need(read(manifest['plan_file'])['source_sha256']==summary['source_sha256'],'New CPU source ledger differs')
    for key in ('samples','runtime_inputs','audit','render_cache'):
        need(digest(Path(manifest[key+'_file']))==manifest[key+'_sha256'],'Published '+key+' changed')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check',action='store_true');group.add_argument('--render',action='store_true');parser.add_argument('--plan',type=Path)
    args=parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu'
         and not os.environ.get('SLURM_JOB_GPUS') and int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==4,
         'Require a Slurm CPU allocation with4cores')
    need(not args.check or args.plan is None,'Check uses the literal registered symbolic parent')
    mode='check' if args.check else 'render';out=OUT/f'{mode}_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False)
    frozen=snapshot(out);start=time.perf_counter()
    save(out/'request.json',dict(mode=mode,plan=str(args.plan) if args.plan else None,parent_plan=str(PARENT_PLAN),source_sha256=frozen))
    try:
        result=check(out,frozen) if args.check else render(args,out,frozen)
        need(source_hashes()==frozen,'Sources changed during new staging')
        result.update(seconds=time.perf_counter()-start,slurm_job_id=os.environ['SLURM_JOB_ID']);save(out/'summary.json',result)
        with (out/'REPORT.md').open('x') as stream:
            stream.write('# Learned-selection identity-join data\n\n'+('Symbolic/input CPU preparation passed.' if args.check else 'Canonical rendering and independent publication audit passed.')+
                ' Every local stream receives the unchanged global question. The old binary-interface route remains stopped; no fit is released.\n\n'+json.dumps(result['counts'],indent=2)+'\n')
        print(json.dumps(dict(passed=True,directory=str(out))),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,
            seconds=time.perf_counter()-start,partial_outputs_retained=True));raise


if __name__=='__main__':main()

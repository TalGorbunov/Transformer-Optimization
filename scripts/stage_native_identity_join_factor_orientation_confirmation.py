"""CPU-only independent confirmation data, frozen before orientation fitting.

The old native report supplies historical pipeline identity only. The new
216-scene training-stage proof supplies exclusions, never model trainability.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import native_identity_join_factor_orientation_confirmation_data as law
from scripts import stage_native_identity_join_factor_fresh as old
from scripts import stage_native_identity_join_factor_orientation_v2 as training
from scripts import stage_native_identity_join as symbolic
need=law.need;oid=law.oid;read=symbolic.read;save=symbolic.save;digest=old.digest
OUT=old.BASE/'identity_join_factor_orientation_confirmation/data_staging'
DATA_ROOT=old.DATA_BASE/'identity_join_factor_orientation_confirmation'
PROTOCOL='identity_join_factor_orientation_confirmation_data';SEED=91726342
OVERALL='docs/paper/NATIVE_AGGREGATION_FACTOR_ORIENTATION_PROPOSAL.md'
OVERALL_SHA='e7b0bc378ca7115a71bc48f033acef693eb1586871bb4ab585bcf0e028fa128e'
PROPOSAL='docs/paper/NATIVE_AGGREGATION_FACTOR_ORIENTATION_CONFIRMATION_DATA_PROPOSAL.md'
PROPOSAL_SHA='1f1b5fa8e1c75505d0863a5e0fb55aba0a6b986a9c68dd375c6c46c4322161a0'
LAW_SHA='f45451efb278b113188f9adcb2180ae6827c8442a7bc0c695789521b2df6f018'
NATIVE_REPORT=old.BASE/'identity_join_factor_native_v2/report_443592/summary.json'
NATIVE_REPORT_SHA='d2f7a6ebe32020960be9e6811fb622f42c8effc3e35cd30e0578a419103d5a59'
FRESH_REPORT=old.BASE/'identity_join_factor_fresh/data_staging/render_443595/summary.json'
FRESH_REPORT_SHA='aed7387e9fff96b68e0d04d26a3584ca93c335fc9e264d093b123a56cd206754'
POLICY=dict(protocol=PROTOCOL,seed=SEED,contexts=810,families=90,same_length_triples=270,image_occurrences=30240,
    lengths=[16,32,64],panels={'A':36,'B':36,'C':18},reference_training_contexts=108,excluded_training_contexts=216,
    excluded_training_triples=72,excluded_prior_fresh_contexts=810,excluded_prior_fresh_triples=270,
    reference_orientation_is_not_unseen=True,canonical_prior_manifests=32,all_prior_planned_and_derived_exposures=True,
    complete_scene_and_concrete_triple_exclusions=True,atomic_image_reuse_allowed=True,target_eos=151645,max_answer_tokens=4,
    cpu_check_seconds=300,cpu_render_seconds=600,cpu_cores=4,memory_gib=16,training_stage_required=True,
    historical_native_identity_only=True,new_model_trainability_established=False,manifest_required_before_fitting=True,
    no_model_or_head_calls=True,no_tensor_loads=True,no_new_statistics_or_training=True,no_inference_release=True)
OWN=('scripts/native_identity_join_factor_orientation_confirmation_data.py',
    'scripts/stage_native_identity_join_factor_orientation_confirmation.py',
    'slurm/native_identity_join_factor_orientation_confirmation_check.sbatch',
    'slurm/native_identity_join_factor_orientation_confirmation_render.sbatch',PROPOSAL)
_READ={}


def bind(path,bindings,expected=None):
    value=old.bind(path,bindings,expected);_READ.update(bindings);return value


def source_hashes():
    need(digest(REPO/OWN[0])==LAW_SHA and digest(REPO/PROPOSAL)==PROPOSAL_SHA
         and digest(REPO/OVERALL)==OVERALL_SHA and digest(NATIVE_REPORT)==NATIVE_REPORT_SHA,'Held confirmation source/protocol/reference changed')
    native=read(NATIVE_REPORT);own,inherited=training.source_maps();result={}
    for group in (old.source_hashes(),native['inherited_source_sha256'],native['source_sha256'],inherited,own,
                  {name:digest(REPO/name) for name in OWN}):
        for name,h in group.items():need(name not in result or result[name]==h,'Source closure conflict');result[name]=h
    for name,h in result.items():need(digest(REPO/name)==h,'Executed source changed: '+name)
    return result


def snapshot(out):
    frozen=source_hashes();(out/'source').mkdir()
    for name,h in frozen.items():
        path=out/'source'/name.replace('/','_');path.write_bytes((REPO/name).read_bytes());need(digest(path)==h,'Confirmation source copy changed')
    save(out/'source_hashes.json',frozen);return frozen


def training_dependency(path):
    value=training.verify_stage(path);summary=value['summary'];plan=value['plan']
    descriptor=dict(file=value['summary_file'],sha256=value['summary_sha256'],plan_file=summary['plan_file'],plan_sha256=summary['plan_sha256'],
        independent_audit_file=summary['independent_audit_file'],independent_audit_sha256=summary['independent_audit_sha256'],
        native_identity_sha256=plan['native_identity_sha256'],source_sha256=value['source_sha256'],inherited_source_sha256=value['inherited_source_sha256'])
    check_training_binding(descriptor)
    return descriptor,plan,value['rows'],value['original_rows']


def check_training_binding(value):
    for key in ('file','plan_file','independent_audit_file'):
        h=value['sha256' if key=='file' else key.replace('_file','_sha256')];need(digest(value[key])==h,'Bound training-stage proof changed')
    summary=read(value['file']);plan=read(value['plan_file']);own,inherited=training.source_maps()
    need(summary['passed'] is summary['completed'] is True and summary['protocol']==training.PROTOCOL
         and summary['plan_file']==value['plan_file'] and summary['plan_sha256']==value['plan_sha256']
         and Path(value['plan_file']).parent==Path(value['file']).parent
         and plan['source_sha256']==value['source_sha256']==own and plan['inherited_source_sha256']==value['inherited_source_sha256']==inherited
         and plan['native_identity_sha256']==value['native_identity_sha256'] and read(value['independent_audit_file'])['passed'] is True,
         'Passed216-scene training-stage source/identity required')
    for name,h in own.items():need(digest(Path(value['file']).parent/'source'/name.replace('/','_'))==h,'Training-stage source copy changed')
    return plan


def historical_reference(manifest):
    historical=manifest['native_report'];old.check_native_binding(historical)
    need(historical['file']==str(NATIVE_REPORT) and historical['sha256']==NATIVE_REPORT_SHA,'Exact old pipeline reference required')
    return dict(file=historical['file'],sha256=historical['sha256'],native_identity=historical['native_identity'],
        native_identity_sha256=historical['native_identity_sha256'],name_target_ids=historical['name_target_ids'],
        historical_binding=historical,purpose='historical_pipeline_identity_only',new_model_trainability_established=False)


def check_reference(value):
    old.check_native_binding(value['historical_binding']);expected=historical_reference(dict(native_report=value['historical_binding']))
    need(value==expected,'Historical pipeline reference changed or was presented as new-model evidence')


def extend_exclusions(base,cohorts,support,bindings):
    scenes=set(base['excluded_scene_sha256']);families=set(base['excluded_family_sha256'])
    records=list(base['records']);family_records=list(base['family_records']);counts={};qa_cache={}
    need({r['sid']:r for r in support}=={r['sid']:r for r in base['original_training_samples']} and len(support)==108,
         'Reference support must remain the exact original108, not the expanded216')
    for kind,rows,source,expected_count,expected_triples in cohorts:
        need(len(rows)==len({r['sid'] for r in rows})==expected_count,'Complete registered exclusion cohort required')
        groups=defaultdict(list)
        for row in rows:
            content=old.qa_content(row,bindings,qa_cache);h=oid(content);scenes.add(h)
            records.append(dict(source=source,kind=kind,sid=row['sid'],n_frames=row['n_frames'],content_sha256=h,
                question=content['question'],qa_file=str(Path(row['path'])/'qa.txt')))
            groups[(row['contrast_id'],row['n_frames'])].append(dict(content,variant=row['variant'],content_sha256=h))
        need(len(groups)==expected_triples,'Registered complete-triple denominator differs')
        for (label,n),values in groups.items():
            need(len(values)==3 and {r['variant'] for r in values}=={0,1,2},'All three answer-changing exclusion variants required')
            h=law.family_identity(values);families.add(h)
            family_records.append(dict(source=source,contrast_id=label,n_frames=n,family_instance_sha256=h,
                scene_sha256=sorted(r['content_sha256'] for r in values)))
        counts[kind]=dict(scenes=len(rows),triples=len(groups))
    return dict(passed=True,canonical_manifests=base['canonical_manifests'],canonical_manifest_count=32,input_bindings=bindings,
        excluded_scene_sha256=sorted(scenes),excluded_family_sha256=sorted(families),scene_identities=len(scenes),family_identities=len(families),
        records=records,family_records=family_records,reference_training_samples=support,added_cohorts=counts,
        actual_QA_reconstructed=True,exposed_planned_and_derived_contexts_included=True,all_training_orientations_excluded=True,
        old_failed_fresh_all_lengths_excluded=True,atom_reuse_separate=True,no_recursive_discovery=True)


runtime_view=old.runtime_view
tokenizer_contract=old.tokenizer_contract


def self_test():
    tests=law.self_test();safe=dict(sid='fixture',n_frames=1,question=symbolic.global_prompt(('Kitchen','Office')),
        image_files=[dict(path='/fixture.png',sha256='0'*64,dimensions=[512,512],mode='RGB')])
    need(runtime_view(safe)==runtime_view(dict(safe,gold='Sandra',states=['offline'],reference_orientation=0,matches_reference_orientation=True)),
         'Confirmation metadata entered the runtime input')
    need(law.SEED==SEED and law.family_identity is not old.law.family_identity,'New law must not mutate the frozen parent module')
    return dict(passed=True,symbolic=tests,strict_runtime_allowlist=True,independent_seed=True)


def check(args,out,frozen):
    dependency,training_plan,training_rows,support=training_dependency(args.training_stage)
    save(out/'training_stage_binding.json',dependency)
    bindings={};bind(FRESH_REPORT,bindings,FRESH_REPORT_SHA);manifest=old.verify_stage(FRESH_REPORT)
    proof=read(FRESH_REPORT);bind(proof['manifest_file'],bindings,proof['manifest_sha256'])
    reference=historical_reference(manifest);save(out/'native_reference.json',reference)
    need(training_plan['native_identity']==reference['native_identity'],'Training/reference backbone identity differs')
    for key in ('file','plan_file','independent_audit_file'):bind(dependency[key],bindings)
    for key in ('rows_file','original_rows_file','semantic_rows_file'):
        bind(training_plan[key],bindings,training_plan['files'][training_plan[key]])
    prior=out/'prior_exclusions';prior.mkdir();base=old.exclusions(prior)
    for path,h in base['input_bindings'].items():need(path not in bindings or bindings[path]==h,'Conflicting prior exclusion binding');bindings[path]=h
    for path,h in reference['historical_binding']['input_bindings'].items():need(path not in bindings or bindings[path]==h,'Conflicting historical binding');bindings[path]=h
    old_rows=[row for panel in ('A','B','C') for n in law.LENGTHS for row in manifest['splits'][f'fresh_{panel}_N{n}']['samples']]
    cohorts=[('old_failed_fresh',old_rows,proof['manifest_file'],810,270),('orientation_training',training_rows,training_plan['rows_file'],216,72)]
    inventory=extend_exclusions(base,cohorts,support,bindings);save(out/'exclusion_inventory.json',inventory)
    scenes=inventory['excluded_scene_sha256'];families=inventory['excluded_family_sha256'];rows=law.generate(support,scenes,families)
    audit=law.audit(rows,support,scenes,families);need(rows==law.generate(support,scenes,families),'Fixed-seed confirmation regeneration differs')
    tests=self_test();tokens=tokenizer_contract()
    need({name:v['target_ids'] for name,v in tokens['targets'].items()}==reference['name_target_ids']
         and tokens['model_directory']==reference['native_identity']['model']['path'],'Canonical reference tokenizer changed')
    reuse=symbolic.reuse_inventory(rows);targets=[dict(sid=r['sid'],target_ids=tokens['targets'][r['gold']]['target_ids']) for r in rows]
    positions={f'fresh_{panel}_N{n}':sum(len(t['target_ids']) for r,t in zip(rows,targets) if r['panel']==panel and r['n_frames']==n)
        for panel in ('A','B','C') for n in law.LENGTHS};need(sum(positions.values())==1800,'Canonical target-position count differs')
    for name,value in (('samples.json',rows),('targets.json',targets),('tokenizer.json',tokens),('reuse_inventory.json',reuse),('symbolic_audit.json',audit),('selftests.json',tests)):
        save(out/name,value)
    for group in (reuse['input_bindings'],tokens['files']):
        for path,h in group.items():need(path not in bindings or bindings[path]==h,'Conflicting confirmation input');bindings[path]=h
    symbolic.verify_bindings(bindings);check_training_binding(dependency);check_reference(reference)
    files={str(path):digest(path) for path in out.glob('*.json')}
    plan=dict(schema_version=1,protocol=PROTOCOL,policy=POLICY,source_sha256=frozen,source_directory=str(out/'source'),files=files,
        samples_file=str(out/'samples.json'),targets_file=str(out/'targets.json'),tokenizer_file=str(out/'tokenizer.json'),
        reuse_inventory_file=str(out/'reuse_inventory.json'),exclusion_inventory_file=str(out/'exclusion_inventory.json'),
        training_stage=dependency,native_reference=reference,symbolic_audit=audit,selftests=tests,data_root=str(DATA_ROOT),
        target_positions_per_cell=positions,input_bindings=bindings,all_semantic_scenes_and_targets_frozen=True,
        manifest_required_before_fitting=True,new_model_trainability_established=False,no_model_loaded=True,no_images_rendered=True,no_inference_release=True)
    save(out/'plan.json',plan)
    return dict(passed=True,completed=True,check_only=True,protocol=PROTOCOL,plan_file=str(out/'plan.json'),plan_sha256=digest(out/'plan.json'),
        source_sha256=frozen,training_stage=dependency,native_reference=reference,counts=audit,distinct_atoms=reuse['distinct_atoms'],
        reused_atoms=reuse['reused_atoms'],new_atoms=reuse['new_atoms'],excluded_scenes=inventory['scene_identities'],excluded_concrete_triples=inventory['family_identities'],
        target_positions_per_cell=positions,no_model_loaded=True,no_images_rendered=True,new_model_trainability_established=False,no_inference_release=True)


def verify_plan(path):
    path=Path(path).resolve();plan=read(path);summary=read(path.parent/'summary.json')
    need(summary['passed'] is summary['completed'] is summary['check_only'] is True and summary['plan_file']==str(path)
         and summary['plan_sha256']==digest(path) and summary['source_sha256']==plan['source_sha256']==source_hashes()
         and plan['protocol']==PROTOCOL and plan['policy']==POLICY and plan['data_root']==str(DATA_ROOT),'Passed immutable confirmation CPU plan required')
    for name,h in plan['source_sha256'].items():need(digest(Path(plan['source_directory'])/name.replace('/','_'))==h,'Confirmation CPU source copy changed')
    symbolic.verify_bindings(plan['files']);symbolic.verify_bindings(plan['input_bindings']);check_training_binding(plan['training_stage']);check_reference(plan['native_reference'])
    inventory=read(plan['exclusion_inventory_file']);rows=read(plan['samples_file'])
    need(law.audit(rows,inventory['reference_training_samples'],inventory['excluded_scene_sha256'],inventory['excluded_family_sha256'])==plan['symbolic_audit'],
         'Frozen confirmation law/exclusions changed');return plan


def render(args,out,frozen):
    plan=verify_plan(args.plan);dependency,_,_,_=training_dependency(args.training_stage)
    need(dependency==plan['training_stage'],'Render must consume the same passed training stage')
    root=DATA_ROOT/f'stage_{os.environ["SLURM_JOB_ID"]}';need(not root.exists() and root.resolve().is_relative_to(DATA_ROOT.resolve()),'Fresh immutable data attempt required')
    root.mkdir(parents=True,exist_ok=False)
    release=dict(protocol=PROTOCOL,source_sha256=frozen,plan_file=str(args.plan.resolve()),plan_sha256=digest(args.plan),
        training_stage=dependency,native_reference=plan['native_reference'],data_root=str(root),new_model_trainability_established=False)
    save(out/'render_release.json',release);save(root/'attempt.json',release)
    rows=read(plan['samples_file']);inventory=read(plan['reuse_inventory_file']);tokens=read(plan['tokenizer_file'])
    need(tokenizer_contract()==tokens,'Actual tokenizer/reference runtime changed')
    renderer,provenance=symbolic.prepare_renderer(inventory['renderer_references']);provenance['method']='Canonical RGB rendering; reused atoms independently rerendered and compared'
    (root/'render_cache').mkdir();(root/'scenes').mkdir()
    def publish(row):
        value=symbolic.publish(row,root,cache,tokens['targets']);value['origin']='canonical_identity_join_factor_orientation_confirmation';return value
    with ThreadPoolExecutor(max_workers=4) as pool:
        cache=dict(pool.map(lambda entry:symbolic.render_atom(entry,root/'render_cache',renderer),inventory['entries']))
        save(root/'render_cache.json',cache);records=list(pool.map(publish,rows))
    actual=old.audit_published(rows,records,cache,root,tokens)
    save(root/'runtime_inputs.json',dict(schema_version=1,protocol=PROTOCOL,records=[runtime_view(r) for r in records],note='Only four allowlisted fields enter native preprocessing'))
    save(root/'samples.json',rows);save(root/'audit.json',dict(symbolic=plan['symbolic_audit'],published=actual,renderer=provenance));splits={}
    for panel,count in (('A',108),('B',108),('C',54)):
        for n in law.LENGTHS:
            group=[r for r in records if r['panel']==panel and r['n_frames']==n];need(len(group)==count,'Confirmation panel/length count differs')
            splits[f'fresh_{panel}_N{n}']=dict(count=count,samples=group,answer_histogram=dict(Counter(r['gold'] for r in group)))
    manifest=dict(schema_version=1,protocol=PROTOCOL,purpose='identity_join_factor_orientation_confirmation',seed=SEED,policy=POLICY,dataset_root=str(root),
        splits=splits,source_sha256=frozen,plan_file=release['plan_file'],plan_sha256=release['plan_sha256'],training_stage=dependency,native_reference=plan['native_reference'],
        tokenizer=tokens,renderer=provenance,target_positions_per_cell=plan['target_positions_per_cell'],
        excluded_inventory_file=plan['exclusion_inventory_file'],excluded_inventory_sha256=plan['files'][plan['exclusion_inventory_file']],
        manifest_required_before_fitting=True,new_model_trainability_established=False,no_model_loaded=True,no_fit_released=True,no_inference_release=True)
    for key in ('runtime_inputs','samples','audit','render_cache'):
        manifest[key+'_file']=str(root/(key+'.json'));manifest[key+'_sha256']=digest(root/(key+'.json'))
    save(root/'main_manifest.json',manifest);symbolic.verify_bindings(plan['input_bindings']);check_training_binding(dependency);check_reference(plan['native_reference'])
    return dict(passed=True,completed=True,check_only=False,protocol=PROTOCOL,source_sha256=frozen,manifest_file=str(root/'main_manifest.json'),manifest_sha256=digest(root/'main_manifest.json'),
        plan_file=release['plan_file'],plan_sha256=release['plan_sha256'],training_stage=dependency,native_reference=plan['native_reference'],dataset_root=str(root),
        counts=plan['symbolic_audit'],published_audit=actual,distinct_atoms=len(cache),reused_atoms=sum(v['reused'] for v in cache.values()),new_atoms=sum(not v['reused'] for v in cache.values()),
        manifest_required_before_fitting=True,new_model_trainability_established=False,no_model_loaded=True,no_fit_released=True,no_inference_release=True)


def verify_stage(path):
    path=Path(path).resolve();path=path/'summary.json' if path.is_dir() else path;summary=read(path)
    need(summary['passed'] is summary['completed'] is True and summary['check_only'] is False and summary['protocol']==PROTOCOL
         and summary['source_sha256']==source_hashes(),'Passed current confirmation render required')
    for name,h in summary['source_sha256'].items():need(digest(path.parent/'source'/name.replace('/','_'))==h,'Render source copy changed')
    need(digest(summary['manifest_file'])==summary['manifest_sha256'],'Confirmation manifest changed');manifest=read(summary['manifest_file'])
    need(manifest['protocol']==PROTOCOL and manifest['policy']==POLICY and manifest['source_sha256']==summary['source_sha256']
         and manifest['plan_file']==summary['plan_file'] and manifest['plan_sha256']==summary['plan_sha256']
         and digest(manifest['plan_file'])==manifest['plan_sha256'] and manifest['training_stage']==summary['training_stage']
         and manifest['native_reference']==summary['native_reference'] and manifest['new_model_trainability_established'] is False,
         'Confirmation render provenance differs')
    plan=read(manifest['plan_file']);proof=read(Path(manifest['plan_file']).parent/'summary.json')
    need(proof['passed'] is proof['completed'] is proof['check_only'] is True and proof['plan_sha256']==manifest['plan_sha256']
         and plan['source_sha256']==summary['source_sha256'] and plan['training_stage']==manifest['training_stage']
         and plan['native_reference']==manifest['native_reference'],'Passed confirmation preparation differs')
    check_training_binding(manifest['training_stage']);check_reference(manifest['native_reference'])
    for key in ('samples','runtime_inputs','audit','render_cache','excluded_inventory'):
        need(digest(manifest[key+'_file'])==manifest[key+'_sha256'],'Published confirmation '+key+' changed')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__);actions=parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--check',action='store_true');actions.add_argument('--render',action='store_true')
    parser.add_argument('--training-stage',type=Path,required=True);parser.add_argument('--plan',type=Path);args=parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS')
         and int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==4,'Only four-core Slurm CPU staging is authorized')
    need(args.plan is None if args.check else args.plan is not None,'Only render consumes the passed confirmation plan')
    mode='check' if args.check else 'render';out=OUT/f'{mode}_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();frozen=snapshot(out);save(out/'request.json',dict(mode=mode,training_stage=str(args.training_stage),plan=None if args.plan is None else str(args.plan),source_sha256=frozen))
    try:
        value=check(args,out,frozen) if args.check else render(args,out,frozen);elapsed=time.perf_counter()-started
        need(source_hashes()==frozen and elapsed<=POLICY['cpu_'+mode+'_seconds'],'Confirmation source or CPU cap changed')
        value.update(seconds=elapsed,slurm_job_id=os.environ['SLURM_JOB_ID'])
        (out/'REPORT.md').write_text('# Orientation confirmation data\n\nAll810 scenes are frozen. This is data evidence only; no new-model trainability, fit or inference is released.\n')
        save(out/'summary.json',value)
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,seconds=time.perf_counter()-started,
            accessed_input_bindings={**old._READ_BINDINGS,**_READ},partial_outputs_retained=True));raise


if __name__=='__main__':main()

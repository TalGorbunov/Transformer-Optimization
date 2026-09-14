"""Pure-JSON fixed order permutation. No tensor/model imports or fit release.

verify_stage(summary_path) returns a hash-verified plan; paired_order() is the
label-independent public constructor. Historical entry dictionaries stay exact.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import json
import os
from pathlib import Path
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import diagnose_native_identity_join_factor_orientation_training as parent
from scripts.stage_native_vision_v6_teacher import need,read,save,sha,object_sha
OUT=REPO/'outputs/native_aggregation_vlm/identity_join_orientation_paired_order'
PLAN=parent.OUT/'check_443676/plan.json'
PLAN_SHA='d299d42a4444520f148f86252e2931ad599e03afac9edadbe9201836229bf097'
CODE=REPO/'outputs/native_aggregation_vlm/identity_join_orientation_joint_code/report_443718/analysis.json'
CODE_SHA='ad46c2472cdc7b6a5791613c478a0fec71c950f7fd0b18de4ea9c6a9cfa41a31'
CODE_FAILURE_SHA='a21a18ae2febcc3da3fa5c5dfb19f2c8cb032d7b35361b018384d2e24304143b'
VISUAL=parent.OUT/'head_replay_443695/summary.json'
VISUAL_SHA='794361ddba6fef9478f61d9d88f8c12053c2573584caf95c5890ef7549da0533'
OPTIM=REPO/'outputs/native_aggregation_vlm/training_budget_review/orientation_optimization_443696/summary.json'
OPTIM_SHA='704621a97d9b35d04dc3434f292e5293d295a342006ac1a46ddf1f95c3efed7e'
PROTOCOL='identity_join_orientation_paired_order'
PROPOSAL='docs/paper/NATIVE_AGGREGATION_ORIENTATION_PAIRED_ORDER_PROPOSAL.md'
OWN=('scripts/stage_native_identity_join_orientation_paired_order.py','slurm/native_identity_join_orientation_paired_order.sbatch',PROPOSAL)
POLICY=dict(cpu_seconds=90,cpu_cores=4,memory_gib=16,base_pairs=54,contexts=216,presentations=48000,updates=6000,
    complete_cycles=888,two_cycle_groups=444,paired_presentations=47952,paired_updates=5994,tail_presentations=48,tail_updates=6,
    pair_presentations_per_update=8,scenes_per_update=16,total_training_head_rows=213330,maximum_head_rows_ceiling=48,
    no_rotation=True,duplicate_boundary_base_ids_allowed=True,whole_entry_permutation=True,no_tensor_or_head_calls=True,no_fit_release=True)


def bind(path,bindings,expected=None):
    path=Path(path).resolve();digest=sha(path);need(expected is None or digest==expected,'Changed paired-order input: '+str(path))
    need(str(path) not in bindings or bindings[str(path)]==digest,'Conflicting paired-order input');bindings[str(path)]=digest;return digest


def sources():return {name:sha(REPO/name) for name in OWN}


def inherited_sources():
    result={};need(sha(CODE)==CODE_SHA and sha(VISUAL)==VISUAL_SHA and sha(OPTIM)==OPTIM_SHA,'Fixed negative evidence changed')
    for document in (dict(source_sha256=parent.sources(),inherited_source_sha256=parent.inherited_sources()),read(CODE),read(VISUAL),read(OPTIM)):
        for key in ('source_sha256','inherited_source_sha256'):
            for name,digest in document.get(key,{}).items():
                need(name not in result or result[name]==digest,'Conflicting source closure');result[name]=digest
    for name,digest in result.items():need(sha(REPO/name)==digest,'Inherited source changed: '+name)
    return result


def snapshot(out):
    frozen=sources();inherited=inherited_sources();(out/'source').mkdir()
    for name,digest in frozen.items():
        file=out/'source'/name.replace('/','_');file.write_bytes((REPO/name).read_bytes());need(sha(file)==digest,'Source snapshot differs')
    save(out/'source_hashes.json',frozen);save(out/'inherited_sources.json',dict(source_sha256=inherited));return frozen,inherited


def paired_order(old,*,base_count=54,groups=444,tail_count=48):
    """Position/ID/orientation-only permutation; no labels, targets or loss."""
    paired_count=2*base_count*groups
    need(len(old)==paired_count+tail_count and 0<=tail_count<base_count,'Unexpected complete-cycle/tail dimensions')
    expected_ids={r['base_pair_id'] for r in old[:base_count]};need(len(expected_ids)==base_count,'First cycle must have unique base IDs')
    permutation=[]
    for group in range(groups):
        start=2*base_count*group;first=old[start:start+base_count];second=old[start+base_count:start+2*base_count]
        need({r['base_pair_id'] for r in first}=={r['base_pair_id'] for r in second}==expected_ids,'Missing or duplicate cycle member')
        need(all(r['cycle']==2*group+1 and r['base_pair_visit']==2*group+1 and r['orientation_version']=='original' for r in first)
             and all(r['cycle']==2*group+2 and r['base_pair_visit']==2*group+2 and r['orientation_version']=='flipped' for r in second),'Historical cycle/orientation sequence differs')
        lookup={r['base_pair_id']:start+base_count+i for i,r in enumerate(second)}
        for i,row in enumerate(first):permutation.extend((start+i,lookup[row['base_pair_id']]))
    tail=old[paired_count:]
    need(len({r['base_pair_id'] for r in tail})==tail_count and all(r['base_pair_id'] in expected_ids and r['cycle']==2*groups+1
         and r['base_pair_visit']==2*groups+1 and r['orientation_version']=='original' for r in tail),'Original final partial cycle differs')
    permutation.extend(range(paired_count,len(old)));return [old[i] for i in permutation],permutation


def independent_permutation(old):
    """Second construction: per-base successive visits, ordered by source index."""
    visits=defaultdict(list)
    for index,row in enumerate(old):visits[row['base_pair_id']].append(index)
    bundles=[];tail=[]
    for indices in visits.values():
        for start in range(0,len(indices)-1,2):
            a,b=indices[start:start+2]
            need(old[a]['orientation_version']=='original' and old[b]['orientation_version']=='flipped','Successive visit orientation differs')
            bundles.append((a,b))
        if len(indices)%2:tail.append(indices[-1])
    return [index for pair in sorted(bundles) for index in pair]+sorted(tail)


def self_test():
    def fixture(base_count,groups,tail):
        result=[]
        for cycle in range(1,2*groups+2):
            ids=list(range(base_count));ids=ids if cycle%2 else ids[::-1]
            if cycle==2*groups+1:ids=ids[:tail]
            for base in ids:result.append(dict(base_pair_id=str(base),cycle=cycle,base_pair_visit=cycle,
                orientation_version='original' if cycle%2 else 'flipped',untouched_payload=[cycle,base]))
        return result
    old=fixture(3,2,2);new,indices=paired_order(old,base_count=3,groups=2,tail_count=2)
    need(indices==independent_permutation(old) and sorted(indices)==list(range(14)) and new[-2:]==old[-2:],'Permutation/tail fixture failed')
    need(any(len({r['base_pair_id'] for r in new[i:i+8:2]})<len(new[i:i+8:2]) for i in range(0,12,8)),'Boundary duplicates fixture must permit repeats')
    for field,value in (('base_pair_id','duplicate'),('orientation_version','original'),('base_pair_visit',99)):
        bad=[dict(r) for r in old];bad[3][field]=value
        try:paired_order(bad,base_count=3,groups=2,tail_count=2)
        except ValueError:pass
        else:raise ValueError('Corrupt cycle fixture was accepted: '+field)
    need(paired_order(fixture(3,1,0),base_count=3,groups=1,tail_count=0)[1]==[0,5,1,4,2,3],'Exact second-cycle remapping fixture failed')
    return dict(passed=True,checks=6,no_tensor_execution=True)


def negative_evidence(bindings):
    bind(CODE,bindings,CODE_SHA);a=read(CODE);failure_file=CODE.parent/'failure.json';bind(failure_file,bindings,CODE_FAILURE_SHA);failure=read(failure_file)
    need(a['passed'] is False and a['completed'] is True and a['all_numerical_audits_collected'] is True and not (CODE.parent/'summary.json').exists()
         and failure['analysis_sha256']==CODE_SHA and Path(failure['analysis_file'])==CODE,'Original failed complete report differs')
    need(a['orientation_plan']==dict(file=str(PLAN),sha256=PLAN_SHA),'Code failure refers to another training cohort')
    bind(a['plan_file'],bindings,a['plan_sha256']);code_plan=read(a['plan_file'])
    for name,digest in a['source_sha256'].items():bind(CODE.parent/'source'/name.replace('/','_'),bindings,digest)
    run=a['runs']['joint_code'];need(run['completed'] is True and run['all_numerical_audits_collected'] is True and len(run['capture_audits'])==22,'Incomplete failed numerical collection')
    captures=run['capture_audits'];need(all(c['fixed_local_codes_exact'] and c['fp16_cast_before_add_exact'] and all(m['passed'] for m in c['functional_core'].values())
        and all(m['passed'] for m in c['nll']) for c in captures),'Failure is not isolated to the retained native-head gate')
    native=[m for c in captures for m in c['native_head']];need(len(native)==216,'All216 native audit rows required')
    disagreements=sum(not m['argmax_exact'] for m in native);fit=run['first_token_fit']
    need(fit['first_correct']==80 and fit['complete_families']==0 and not fit['passed'] and disagreements==2 and fit['first_correct']==80<206,'Fixed code failure/sensitivity differs')
    run_summary=Path(run['run_directory'])/'summary.json';bind(run_summary,bindings,run['run_summary_sha256']);rs=read(run_summary)
    need(rs['plan_sha256']==a['plan_sha256'] and rs['first_token_fit']==fit,'Code raw-run ownership differs')
    for key in ('raw','predictions'):bind(rs[key+'_file'],bindings,rs[key+'_sha256'])
    endpoint=run['endpoint_audits']['6000'];bind(endpoint['outcomes_file'],bindings,endpoint['outcomes_sha256'])
    outcomes=read(endpoint['outcomes_file']);outcome_map={r['sid']:r for r in outcomes}
    need(len(outcomes)==len(outcome_map)==216 and parent.criteria(outcomes)==fit
         and all(r['first_token_correct']==(r['argmax_id']==r['first_token_id']) for r in outcomes),'Complete code outcomes required')
    native_by_sid={sid:metric for capture in captures for sid,metric in zip(capture['sids'],capture['native_head'])}
    need(set(native_by_sid)==set(outcome_map) and all(outcome_map[sid]['first_token_correct'] for sid,metric in native_by_sid.items() if not metric['argmax_exact']),
         'Both disagreements must be GPU-correct as in the retained evidence')
    cpu_correct=sum(r['first_token_correct'] and native_by_sid[sid]['argmax_exact'] for sid,r in outcome_map.items())
    need(cpu_correct==78,'Exact CPU sensitivity differs')
    result=dict(joint_code=dict(analysis_file=str(CODE),analysis_sha256=CODE_SHA,failure_file=str(failure_file),failure_sha256=CODE_FAILURE_SHA,
        plan_file=a['plan_file'],plan_sha256=a['plan_sha256'],run_summary_file=str(run_summary),run_summary_sha256=run['run_summary_sha256'],
        raw_file=rs['raw_file'],raw_sha256=rs['raw_sha256'],first_correct=80,contexts=216,complete_families=0,
        argmax_disagreements=2,cpu_first_correct=78,optimistic_correct=80,observed_backend_union_correct=80,original_report_remains_failed=True))
    for name,path,digest in (('visual',VISUAL,VISUAL_SHA),('optimization',OPTIM,OPTIM_SHA)):
        bind(path,bindings,digest);s=read(path);need(s['passed'] is True and s['completed'] is True,'Completed negative diagnostic required')
        bind(s['analysis_file'],bindings,s['analysis_sha256']);analysis=read(s['analysis_file']);need(analysis['passed'] is True,'Negative diagnostic analysis failed')
        for source,source_sha in s['source_sha256'].items():bind(path.parent/'source'/source.replace('/','_'),bindings,source_sha)
        result[name]=dict(file=str(path),sha256=digest,analysis_file=s['analysis_file'],analysis_sha256=s['analysis_sha256'])
        if name=='visual':
            need(analysis['plan_sha256']==PLAN_SHA and analysis['original_audit_remains_failed'] is True and analysis['no_acceptance_or_native_release'] is True,'Visual failure preservation differs')
            for arm,expected in (('product',79),('additive',72)):
                sensitivity=analysis['results'][arm]['sensitivity']
                need(sensitivity['gpu']['first_correct']==expected and not sensitivity['every_disagreement_correct']['passed'],'Visual negative sensitivity changed')
            bind(analysis['original_failure_file'],bindings,analysis['original_failure_sha256'])
    return result,dict(file=a['plan_file'],sha256=a['plan_sha256'])


def audit_order(old,new,permutation,rows,pairs):
    need(len(old)==len(new)==len(permutation)==48000 and sorted(permutation)==list(range(48000)),'Exact48000-entry bijection required')
    need(permutation==independent_permutation(old) and all(new[j]==old[i] for j,i in enumerate(permutation)),'Independent permutation or unchanged entry dictionary differs')
    restored=[None]*48000
    for row,index in zip(new,permutation):restored[index]=row
    need(restored==old and new[-48:]==old[-48:] and permutation[-48:]==list(range(47952,48000)),'Inverse reconstruction/tail differs')
    lookup={r['sid']:r for r in rows};pairmap={p['pair_id']:p for p in pairs}
    need(len(lookup)==216 and len(pairmap)==108,'Complete training rows/pairs required')
    for entry in old:
        pair=pairmap[entry['pair_id']];need(entry['sids']==pair['sids'] and [lookup[s]['n_frames'] for s in entry['sids']]==[8,16]
            and all(lookup[s]['base_pair_id']==entry['base_pair_id'] and lookup[s]['orientation_version']==entry['orientation_version'] for s in entry['sids']), 'Whole length-pair ownership differs')
    for i in range(0,47952,2):
        a,b=new[i:i+2];need(a['base_pair_id']==b['base_pair_id'] and (a['orientation_version'],b['orientation_version'])==('original','flipped')
            and [lookup[s]['target_ids'] for s in a['sids']]==[lookup[s]['target_ids'] for s in b['sids']],'Paired counterpart or targets differ')
    original_counts=Counter((r['base_pair_id'],r['orientation_version'],r['pair_id'],tuple(r['sids'])) for r in old)
    counts=Counter((r['base_pair_id'],r['orientation_version'],r['pair_id'],tuple(r['sids'])) for r in new)
    need(counts==original_counts,'Orientation-specific complete presentation multiset differs')
    head_rows=[sum(len(lookup[s]['target_ids']) for entry in new[start:start+8] for s in entry['sids']) for start in range(0,48000,8)]
    old_rows=[sum(len(lookup[s]['target_ids']) for entry in old[start:start+8] for s in entry['sids']) for start in range(0,48000,8)]
    need(len(head_rows)==6000 and sum(head_rows)==sum(old_rows)==213330 and max(head_rows)<=48,'Full CE presentation/target work changed')
    duplicate_batches=[i//8+1 for i in range(0,47952,8) if len({r['base_pair_id'] for r in new[i:i+8:2]})<4]
    orientation_counts=Counter(r['orientation_version'] for r in new)
    need(orientation_counts=={'original':24024,'flipped':23976},'Exact overall orientation exposure changed')
    return head_rows,dict(passed=True,entries=48000,whole_entry_bijection=True,inverse_reconstruction_exact=True,tail_exact=True,
        independent_reconstruction_exact=True,paired_updates=5994,tail_updates=6,scene_presentations=96000,orientation_presentations=dict(orientation_counts),
        total_training_head_rows=213330,max_training_head_rows=max(head_rows),old_max_training_head_rows=max(old_rows),
        updates_with_changed_head_rows=sum(a!=b for a,b in zip(head_rows,old_rows)),duplicate_base_id_batches=duplicate_batches,
        exposure_counts=[dict(base_pair_id=k[0],orientation_version=k[1],pair_id=k[2],sids=list(k[3]),count=v) for k,v in sorted(counts.items())])


def stage(out,frozen,inherited,bindings):
    bind(PLAN,bindings,PLAN_SHA);parent_plan=parent.verify_plan(PLAN,ancestors=False)
    for key in ('rows','scenes','pairs','order'):bind(parent_plan[key+'_file'],bindings,parent_plan['runtime_bindings'][parent_plan[key+'_file']])
    negative,code_plan=negative_evidence(bindings);rows=read(parent_plan['rows_file']);pairs=read(parent_plan['pairs_file']);old=read(parent_plan['order_file'])
    tests=self_test();save(out/'self_test.json',tests);new,permutation=paired_order(old)
    head_rows,audit=audit_order(old,new,permutation,rows,pairs)
    row_lookup={r['sid']:r for r in rows}
    need(parent_plan['head_rows_by_update']==[sum(len(row_lookup[s]['target_ids']) for entry in old[i:i+8] for s in entry['sids']) for i in range(0,48000,8)],'Original head-row ledger differs')
    save(out/'order.json',new);save(out/'permutation.json',permutation);save(out/'head_rows.json',dict(by_update=head_rows,total=sum(head_rows),maximum=max(head_rows)))
    save(out/'independent_audit.json',audit);save(out/'negative_evidence.json',negative)
    runtime={str(out/name):sha(out/name) for name in ('order.json','permutation.json','head_rows.json','independent_audit.json','negative_evidence.json','self_test.json')}
    for key in ('rows','scenes','pairs'):runtime[parent_plan[key+'_file']]=bindings[parent_plan[key+'_file']]
    plan=dict(schema_version=1,protocol=PROTOCOL,policy=POLICY,source_sha256=frozen,inherited_source_sha256=inherited,
        orientation_plan=dict(file=str(PLAN),sha256=PLAN_SHA),joint_code_orientation_plan=code_plan,
        rows_file=parent_plan['rows_file'],scenes_file=parent_plan['scenes_file'],pairs_file=parent_plan['pairs_file'],
        original_order_file=parent_plan['order_file'],original_order_sha256=bindings[parent_plan['order_file']],
        order_file=str(out/'order.json'),order_sha256=sha(out/'order.json'),order_object_sha256=object_sha(new),
        permutation_file=str(out/'permutation.json'),permutation_sha256=sha(out/'permutation.json'),
        head_rows_by_update=head_rows,head_rows_file=str(out/'head_rows.json'),head_rows_sha256=sha(out/'head_rows.json'),
        total_training_head_rows=213330,max_training_head_rows=max(head_rows),independent_audit_file=str(out/'independent_audit.json'),
        independent_audit_sha256=sha(out/'independent_audit.json'),negative_evidence=negative,
        native_identity=parent_plan['native_identity'],native_identity_sha256=parent_plan['native_identity_sha256'],
        input_bindings=bindings,runtime_bindings=runtime,no_tensor_or_head_calls=True,no_fit_release=True)
    save(out/'plan.json',plan);(out/'plan.sha256').write_text(sha(out/'plan.json')+'\n');return plan


def verify_stage(summary_path):
    path=Path(summary_path).resolve();path=path/'summary.json' if path.is_dir() else path;s=read(path)
    need(s['passed'] is True and s['completed'] is True and s['protocol']==PROTOCOL and s['source_sha256']==sources()
        and s['inherited_source_sha256']==inherited_sources() and s['no_fit_release'] is True,'Completed fixed order stage required')
    need(sha(s['plan_file'])==s['plan_sha256']==Path(s['plan_file']).with_suffix('.sha256').read_text().strip(),'Stage plan changed')
    plan=read(s['plan_file']);need(plan['protocol']==PROTOCOL and plan['policy']==POLICY and plan['source_sha256']==s['source_sha256']
        and plan['inherited_source_sha256']==s['inherited_source_sha256'] and plan['orientation_plan']==dict(file=str(PLAN),sha256=PLAN_SHA),'Stage source/ancestor differs')
    for name,digest in s['source_sha256'].items():need(sha(path.parent/'source'/name.replace('/','_'))==digest,'Stage source archive differs')
    need(read(path.parent/'source_hashes.json')==s['source_sha256'] and read(path.parent/'inherited_sources.json')==dict(source_sha256=s['inherited_source_sha256']),'Stage source ledgers differ')
    for group in (plan['input_bindings'],plan['runtime_bindings']):
        for name,digest in group.items():need(sha(name)==digest,'Stage bound file changed: '+name)
    need(sha(s['report_file'])==s['report_sha256'] and read(plan['independent_audit_file'])['passed'] is True
        and plan['total_training_head_rows']==213330 and len(plan['head_rows_by_update'])==6000 and max(plan['head_rows_by_update'])==plan['max_training_head_rows']<=48,
        'Order audit/count ledger differs')
    return plan


def main():
    need(len(sys.argv)==1,'Fixed CPU stage has no tunable arguments');parent.native.require_slurm(gpu=False)
    need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS') and os.environ.get('SLURM_JOB_NAME')=='identity_join_orientation_paired_order'
         and int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==4,'Fixed four-core CPU stage required')
    started=time.perf_counter();out=OUT/('stage_'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True,exist_ok=False);frozen,inherited=snapshot(out);bindings={}
    try:
        plan=stage(out,frozen,inherited,bindings);need(sources()==frozen and inherited_sources()==inherited,'Stage sources changed')
        elapsed=time.perf_counter()-started;need(elapsed<=90,'Fixed90-second stage exceeded')
        (out/'REPORT.md').write_text('# Fixed paired-minibatch order\n\n'
            'The stage passes an exact48,000-entry permutation audit. There are5,994 paired updates and six unchanged original-orientation tail updates. '
            f"Total target rows remain213,330; the new maximum per update is{plan['max_training_head_rows']}. "
            'Entry dictionaries, target sequences and whole N8/N16 pairs are unchanged. Duplicate base IDs at boundary batches are retained. '
            'Examples now meet different learning rates and optimizer states; no identical-trajectory or cancellation claim is made. '
            'The failed numerical reports remain failed. This JSON-only stage performs no tensor/head work and releases no fit.\n')
        save(out/'summary.json',dict(passed=True,completed=True,protocol=PROTOCOL,source_sha256=frozen,inherited_source_sha256=inherited,
            plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),report_file=str(out/'REPORT.md'),report_sha256=sha(out/'REPORT.md'),
            orientation_plan=plan['orientation_plan'],total_training_head_rows=213330,max_training_head_rows=plan['max_training_head_rows'],
            no_tensor_or_head_calls=True,no_fit_release=True,elapsed_seconds=elapsed));print(json.dumps(dict(passed=True,directory=str(out))),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,inherited_source_sha256=inherited,
            input_bindings=bindings,elapsed_seconds=time.perf_counter()-started,partial_outputs_retained=True));raise


if __name__=='__main__':main()

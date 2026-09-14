"""Fixed official-MMReD pilot statistics; CPU Slurm only, no model/tensor replay."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import mmred_official_answer as answer
PROTOCOL='mmred_official_pilot_combined_statistics'
PROPOSAL='docs/paper/NATIVE_AGGREGATION_MMRED_OFFICIAL_EVALUATION_PROTOCOL.md'
PROPOSAL_SHA='e22eae44ada348fc919830524b86d5cbea755a9a17b851b02e0e049d1e0cd409'
OWN=('scripts/report_mmred_official_pilot.py',PROPOSAL,'slurm/mmred_official_pilot_report.sbatch')
OUT=REPO/'outputs/native_aggregation_vlm/mmred_official_pilot'
DATA=Path('/mnt/data/gabriele/gnn_transformer/mmred_official_pilot')
ARMS=('ordinary','normalized','mass')
TASKS=('char_at_frame','steps_in_room','spend_together','where_spend')
PRIMARY=('spend_together','where_spend')
PEOPLE=('Daniel','John','Mary','Michael','Sandra')
ROOMS=('Bathroom','Bedroom','Garden','Hallway','Kitchen','Office')
CONTRASTS=(('mass','ordinary'),('mass','normalized'),('normalized','ordinary'))
COLUMNS=ARMS+tuple(a+'_minus_'+b for a,b in CONTRASTS)
POLICY=dict(cpu_seconds=1800,cpu_cores=4,memory_gib=16,arms=list(ARMS),
    rows_per_arm=1000,total_predictions=3000,analyses=30,replicates_per_analysis=10000,seed=24,
    percentile_method='type7',interval_probabilities=[.025,.975],max_new_tokens=50,
    primary='test_N32_primary:mass_minus_ordinary',no_model_or_head_calls=True,
    no_fitting=True,no_outcome_eligibility=True,no_followup_release=True)


def need(ok,message):
    if not ok:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def object_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def save(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,sort_keys=True,ensure_ascii=False,allow_nan=False);stream.write('\n')


def bind(path,bindings,digest=None):
    path=str(Path(path).resolve());actual=sha(path)
    need(digest is None or actual==digest,'Bound input hash differs: '+path)
    need(path not in bindings or bindings[path]==actual,'Conflicting input identity')
    bindings[path]=actual;return read(path)


def sources():
    need(sha(REPO/PROPOSAL)==PROPOSAL_SHA,'Prospective pilot protocol changed')
    return {name:sha(REPO/name) for name in OWN}


def check_time(started):need(time.perf_counter()-started<1800,'Fixed combined CPU allocation exceeded')


def analysis_specs():
    panels=(('val',(8,)),('val',(16,)),('val',(8,16)),('test',(8,)),('test',(16,)),('test',(32,)))
    return [(role,ns,qs) for role,ns in panels for qs in (*((q,) for q in TASKS),PRIMARY)]


def key(role,ns,qs):return role+'_N'+'_'.join(map(str,ns))+'_'+('primary' if len(qs)==2 else qs[0])


def quantile_type7(values,p):
    ordered=sorted(values);need(ordered and 0<=p<=1,'Quantile input invalid')
    h=(len(ordered)-1)*p;j=math.floor(h)
    return float(ordered[j] if j==len(ordered)-1 else ordered[j]+(h-j)*(ordered[j+1]-ordered[j]))


def point_fraction(table,arm):
    return sum((Fraction(table['correct'][arm][q],table['denominators'][q]) for q in table['tasks']),Fraction())/len(table['tasks'])


def points(rows,qs):
    denominators={q:sum(r['qtype']==q for r in rows) for q in qs}
    need(all(denominators.values()),'Every selected task must be represented')
    correct={a:{q:sum(int(r['scores'][a]['correct']) for r in rows if r['qtype']==q) for q in qs} for a in ARMS}
    table=dict(tasks=list(qs),denominators=denominators,correct=correct)
    values={a:float(point_fraction(table,a)) for a in ARMS}
    values.update({a+'_minus_'+b:float(point_fraction(table,a)-point_fraction(table,b)) for a,b in CONTRASTS})
    outcomes={a:dict(contexts=len(rows),**{k:sum(int(r['scores'][a][k]) for r in rows)
        for k in ('correct','format_valid','semantic_match','completed','truncated')}) for a in ARMS}
    return dict(table,points=values,outcome_counts=outcomes)


def bootstrap(np,rows,role,ns,qs,replicates=10000):
    selected=[r for r in rows if r['pilot_role']==role and r['n'] in ns and r['qtype'] in qs]
    table=points(selected,qs);worlds=defaultdict(list)
    for row in selected:worlds[row['world_sha256']].append(row)
    grouped=defaultdict(list)
    for world,members in worlds.items():
        lengths={r['n'] for r in members};need(len(lengths)==1,'One full world has conflicting lengths')
        pattern=tuple(sorted({r['qtype'] for r in members}))
        grouped[(next(iter(lengths)),pattern)].append((world,members))
    inventory=[];arrays=[]
    for (n,pattern),clusters in sorted(grouped.items()):
        clusters.sort(key=lambda x:x[0]);values=[];records=[]
        for world,members in clusters:
            values.append([[sum(r['qtype']==q for r in members)]+
                [sum(int(r['scores'][a]['correct']) for r in members if r['qtype']==q) for a in ARMS] for q in qs])
            records.append(dict(world_sha256=world,indices=[r['index'] for r in members],sids=[r['sid'] for r in members],
                qtypes=[r['qtype'] for r in members],rows=len(members)))
        inventory.append(dict(n=n,task_pattern=list(pattern),clusters=records,cluster_count=len(clusters)))
        arrays.append(np.asarray(values,dtype=np.int64))
    rng=random.Random(24);draws=np.empty((replicates,len(COLUMNS)),dtype='<f8')
    for b in range(replicates):
        counts=np.zeros((len(qs),4),dtype=np.int64)
        for values in arrays:
            indices=[rng.randrange(len(values)) for _ in range(len(values))]
            counts+=values[indices].sum(axis=0)
        need(bool((counts[:,0]>0).all()),'Bootstrap erased a represented task')
        rates=(counts[:,1:]/counts[:,0,None]).mean(axis=0)
        draws[b,:3]=rates
        draws[b,3:]=[rates[ARMS.index(a)]-rates[ARMS.index(c)] for a,c in CONTRASTS]
    intervals={name:[quantile_type7(draws[:,i],p) for p in (.025,.975)] for i,name in enumerate(COLUMNS)}
    return dict(table,key=key(role,ns,qs),panel=role,lengths=list(ns),rows=len(selected),
        world_clusters=len(worlds),repeated_world_clusters=sum(len(v)>1 for v in worlds.values()),
        intervals=intervals,replicates=replicates,seed=24,columns=list(COLUMNS),
        percentile_method='type7',strata=inventory),draws


def qualification(tables):
    get=lambda role,ns,qs:tables[key(role,ns,qs)]
    frac=lambda role,ns,qs,a:point_fraction(get(role,ns,qs),a)
    diff=lambda role,ns,qs,a,b:frac(role,ns,qs,a)-frac(role,ns,qs,b)
    competence=dict(ordinary_val_retrieval_ge_80_of_100=frac('val',(8,16),('char_at_frame',),'ordinary')>=Fraction(4,5))
    for q in PRIMARY:
        competence['ordinary_val_'+q+'_ge_50_of_100']=frac('val',(8,16),(q,),'ordinary')>=Fraction(1,2)
    conditions=dict(mass_N32_primary_ge_60_percent=frac('test',(32,),PRIMARY,'mass')>=Fraction(3,5),
        mass_N32_primary_gain_ordinary_ge_10pp=diff('test',(32,),PRIMARY,'mass','ordinary')>=Fraction(1,10),
        mass_N32_primary_gain_normalized_ge_5pp=diff('test',(32,),PRIMARY,'mass','normalized')>=Fraction(1,20),
        primary_mass_ordinary_CI_lower_positive=get('test',(32,),PRIMARY)['intervals']['mass_minus_ordinary'][0]>0)
    for q in PRIMARY:conditions['N32_'+q+'_mass_ordinary_nonnegative']=diff('test',(32,),(q,),'mass','ordinary')>=0
    for n in (8,16):conditions['N'+str(n)+'_primary_mass_ordinary_drop_le_5pp']=diff('test',(n,),PRIMARY,'mass','ordinary')>=-Fraction(1,20)
    for q in ('char_at_frame','steps_in_room'):
        conditions['N32_'+q+'_mass_ordinary_drop_le_5pp']=diff('test',(32,),(q,),'mass','ordinary')>=-Fraction(1,20)
    return dict(ordinary_competence_conditions=competence,ordinary_competence_screen_passed=all(competence.values()),
        ordinary_val_count_ge_50_of_100_descriptive=frac('val',(8,16),('steps_in_room',),'ordinary')>=Fraction(1,2),
        mass_pilot_conditions=conditions,all_mass_conditions_passed=all(conditions.values()),
        promising_mass_specific_pilot=all(competence.values()) and all(conditions.values()),
        qualification_is_heuristic=True,five_pp_noninferiority_claim=False,test_historically_used=True,
        requires_new_world_two_seed_confirmation=True,no_automatic_followup_release=True)


def typed_gold(row):
    if row['atype']=='number':
        value=row['answer'];need(type(value) is int or (isinstance(value,str) and re.fullmatch(r'[0-9]+',value)), 'Canonical nonnegative count required')
        return int(value)
    need(row['atype'] in ('room','person') and isinstance(row['answer'],str),'Original typed answer required')
    return row['answer']


def question_strata(row):
    q=row['question'].strip();people='(?:'+'|'.join(PEOPLE)+')'
    result=dict(answer_type=row['atype'],gold_json=json.dumps(typed_gold(row),ensure_ascii=False),
        most_least='not_applicable',numeric_zero_nonzero='not_applicable',retrieval_requested_frame='not_applicable',
        retrieval_relative_position='not_applicable')
    if row['atype']=='number':result['numeric_zero_nonzero']='zero' if typed_gold(row)==0 else 'nonzero'
    if row['qtype'] in PRIMARY:
        prefix='In which room did ' if row['qtype']=='where_spend' else 'With whom did '
        suffix=' time' if row['qtype']=='where_spend' else ' time together in the same room'
        match=re.fullmatch(re.escape(prefix)+people+r' spend the (most|least amount of)'+re.escape(suffix)+r'[?.]?',q)
        result['most_least']='unclassified' if match is None else ('most' if match[1]=='most' else 'least')
    if row['qtype']=='char_at_frame':
        match=re.fullmatch(r'In which room was '+people+r' at step ([0-9]+)[?.]?',q)
        step=None if match is None else int(match[1])
        valid=step is not None and 1<=step<=row['n']
        result['retrieval_requested_frame']=str(step) if valid else 'unclassified'
        result['retrieval_relative_position']=str(Fraction(step,row['n'])) if valid else 'unclassified'
    return result


def descriptive(rows):
    groups=defaultdict(list)
    for row in rows:
        for field,value in row['strata'].items():
            groups[(row['pilot_role'],row['n'],row['qtype'],field,value)].append(row)
    result=[]
    for (role,n,q,field,value),members in sorted(groups.items()):
        result.append(dict(panel=role,n=n,qtype=q,stratum=field,value=value,rows=len(members),
            world_clusters=len({r['world_sha256'] for r in members}),
            counts={a:{k:sum(int(r['scores'][a][k]) for r in members) for k in ('correct','format_valid','semantic_match','completed','truncated')} for a in ARMS}))
    return result


def self_test(np):
    need(quantile_type7([0,1,4,10],.25)==.75 and math.isclose(quantile_type7([0,1,4,10],.975),9.55),'Type7 interpolation fixture failed')
    rows=[]
    for world,good in (('a',True),('b',False)):
        for q in PRIMARY:
            rows.append(dict(index=len(rows),sid=str(len(rows)),world_sha256=world,n=32,pilot_role='test',qtype=q,
                scores={a:dict(correct=(good if a=='mass' else not good),format_valid=True,semantic_match=(good if a=='mass' else not good),completed=True,truncated=False) for a in ARMS}))
    one,a=bootstrap(np,rows,'test',(32,),PRIMARY,64);two,b=bootstrap(np,rows,'test',(32,),PRIMARY,64)
    need(one==two and np.array_equal(a,b) and len(one['strata'])==1
         and [r['rows'] for r in one['strata'][0]['clusters']]==[2,2], 'Seed/duplicate-world joint pattern fixture failed')
    rng=random.Random(24);expected=[]
    for _ in range(64):
        count=sum(rng.randrange(2)==0 for _ in range(2));expected.append(count-1)
    need(a[:,3].tolist()==expected and set(expected)=={-1,0,1},'Both task rows must share each world multiplicity')
    tables={}
    for role,ns,qs in analysis_specs():
        den={q:(100 if len(ns)==2 else 50) for q in qs}
        counts={arm:{q:den[q]//2 for q in qs} for arm in ARMS}
        for arm in ARMS:
            if role=='val':counts[arm]={q:(80 if q=='char_at_frame' and len(ns)==2 else den[q]//2) for q in qs}
            elif ns==(32,):
                counts[arm]={q:(30 if arm=='mass' else (27 if q==PRIMARY[0] else 28) if arm=='normalized' else 25) if q in PRIMARY else 25 for q in qs}
        tables[key(role,ns,qs)]=dict(tasks=list(qs),denominators=den,correct=counts,intervals={'mass_minus_ordinary':[.001,.2]})
    need(qualification(tables)['promising_mass_specific_pilot'],'Exact60/10/5 boundary fixture failed')
    control=tables[key('test',(32,),('char_at_frame',))]
    control['correct']['mass']['char_at_frame']=23
    need(qualification(tables)['promising_mass_specific_pilot'],'Two control errors must pass five-pp heuristic')
    control['correct']['mass']['char_at_frame']=22
    need(not qualification(tables)['mass_pilot_conditions']['N32_char_at_frame_mass_ordinary_drop_le_5pp'],'Three control errors must fail five-pp heuristic')
    control['correct']['mass']['char_at_frame']=25
    safety=tables[key('test',(8,),PRIMARY)]
    safety['correct']['ordinary']={PRIMARY[0]:30,PRIMARY[1]:30}
    safety['correct']['mass']={PRIMARY[0]:27,PRIMARY[1]:28}
    need(qualification(tables)['mass_pilot_conditions']['N8_primary_mass_ordinary_drop_le_5pp'],'Exact five-pp macro safety boundary failed')
    safety['correct']['mass'][PRIMARY[1]]=27
    need(not qualification(tables)['mass_pilot_conditions']['N8_primary_mass_ordinary_drop_le_5pp'],'Six-pp macro drop must fail')
    safety['correct']['mass'][PRIMARY[1]]=28
    tables[key('test',(32,),PRIMARY)]['intervals']['mass_minus_ordinary'][0]=0
    need(not qualification(tables)['promising_mass_specific_pilot'],'Zero CI lower endpoint is not positive')
    parser=answer.self_test()
    known=dict(question='In which room was Mary at step 8?',qtype='char_at_frame',n=16,atype='room',answer='Kitchen')
    need(question_strata(known)['retrieval_relative_position']=='1/2'
         and question_strata(dict(known,question='Tell me where Mary was'))['retrieval_requested_frame']=='unclassified','Anchored frame/unclassified fixture failed')
    need(len(analysis_specs())==30 and len({key(*s) for s in analysis_specs()})==30,'Thirty exact analyses required')
    return dict(passed=True,groups=7,synthetic_bootstrap_calls=2,synthetic_replicates=128,actual_data_bootstrap_calls=30,
        correlated_world_tasks=True,exact_point_thresholds=True,strict_parser=parser,model_calls=0,head_calls=0)


def original_world(row,files,bindings):
    file=row['renderer_file']
    if file not in files:files[file]=bind(file,bindings)
    sample=files[file][row['renderer_row']]
    need(all(sample[k]==row[k] for k in ('qid','question','answer','atype','qtype'))
         and sample['seq_len']==row['n'] and object_sha(sample['sequence'])==row['exact_sequence_sha256'],
         'Bound original renderer row differs')
    sequence=sample['sequence']
    need(len(sequence)==row['n'] and [s['step_id'] for s in sequence]==list(range(1,row['n']+1)),'Original ordered frame/Step inventory differs')
    canonical=[]
    for frame in sequence:
        rooms=frame['rooms']
        need(set(rooms)==set(ROOMS) and Counter(c for v in rooms.values() for c in v)==Counter(PEOPLE),'Original room/person ownership differs')
        lookup={c:ROOMS.index(room) for room,occupants in rooms.items() for c in occupants}
        canonical.append(tuple(lookup[c] for c in PEOPLE))
    need(object_sha(canonical)==row['world_sha256'],'Full question-free world identity differs')
    if 'source_identity' in row:need(row['source_identity']['world_sha256']==row['world_sha256'],'Flat original world identity differs')
    return dict(world_sha256=row['world_sha256'],raw_row_sha256=row['raw_row_sha256'],
        source_file=row['source_file'],source_row=row['source_row'],source_cell_row=row['source_cell_row'],
        renderer_file=file,renderer_file_sha256=bindings[str(Path(file).resolve())],renderer_row=row['renderer_row'],
        exact_sequence_sha256=row['exact_sequence_sha256'],qid=row['qid'],cell=row['cell'])


def inputs(paths,bindings):
    from scripts import report_mmred_official_native_evaluation as auditor
    need(len(paths)==3,'Three completed independent evaluation audits required')
    audits={};summaries={};predictions={};inherited={};descriptors={}
    for path in paths:
        path=Path(path).resolve();summary=bind(path,bindings)
        result=auditor.verify_report(path)
        need(summary['protocol']==auditor.PROTOCOL and result['passed'] is result['completed'] is True
             and result['all_scheduled_numerical_evidence_collected'] is True and result['arm'] in ARMS
             and result['raw_inputs_verified_in_cpu_audit'] is True and result['raw_files_rehashed_in_handoff'] is False
             and result['statistics_protocol']==dict(file=str(REPO/PROPOSAL),sha256=PROPOSAL_SHA),
             'Complete numerical audit is required; there is no efficacy eligibility test')
        arm=result['arm'];need(arm not in audits,'Repeated evaluation arm')
        bind(summary['analysis_file'],bindings,summary['analysis_sha256'])
        for field in ('input_bindings','artifacts'):
            bind(summary[field+'_file'],bindings,summary[field+'_sha256'])
        values=bind(result['predictions_file'],bindings,result['predictions_sha256'])
        need(len(values)==1000 and [v['index'] for v in values]==list(range(1000)),'Exact ordered1000 lean predictions required')
        for name,digest in {**summary['source_sha256'],**summary['inherited_source_sha256']}.items():
            need(name not in inherited or inherited[name]==digest,'Conflicting inherited source identities')
            inherited[name]=digest
        audits[arm]=result;summaries[arm]=summary;predictions[arm]=values
        descriptors[arm]=dict(file=str(path),sha256=sha(path),analysis_file=summary['analysis_file'],
            analysis_sha256=summary['analysis_sha256'],predictions_file=result['predictions_file'],predictions_sha256=result['predictions_sha256'],
            final_checkpoint=result['final_checkpoint'])
    need(set(audits)==set(ARMS),'All final arms required')
    parent=audits['ordinary']
    for result in audits.values():
        need(result['original_rows_file']==parent['original_rows_file'] and result['original_rows_sha256']==parent['original_rows_sha256']
             and result['evalfeature_stage']==parent['evalfeature_stage']
             and result['native_identity_sha256']==parent['native_identity_sha256'],'All arms must use identical official inputs/features/native backbone')
    original=bind(parent['original_rows_file'],bindings,parent['original_rows_sha256'])
    expected=Counter({(role,n,q):50 for role,ns in (('val',(8,16)),('test',(8,16,32))) for n in ns for q in TASKS})
    need(len(original)==len({r['sid'] for r in original})==1000
         and Counter((r['pilot_role'],r['n'],r['qtype']) for r in original)==expected,'Exact20x50 original evaluation cohort differs')
    rows=[];files={}
    for index,row in enumerate(original):
        owner=original_world(row,files,bindings);scores={};native={}
        for arm in ARMS:
            entry=predictions[arm][index]
            need(all(entry[k]==row[k] for k in ('sid','n','qtype','pilot_role')),'Cross-arm/source prediction join differs')
            score=answer.score_answer(entry['primary_text'],row['atype'],typed_gold(row),entry['generated_ids'])
            need(score==entry['score'],'Independent strict JSON/native-EOS score differs')
            need(type(entry['seconds']) in (int,float) and math.isfinite(entry['seconds']) and entry['seconds']>0
                 and 0<entry['max_allocated_bytes']<=entry['max_reserved_bytes'],'Observed inference time/memory required')
            scores[arm]=score
            native[arm]={k:entry[k] for k in ('generated_ids','primary_text','seconds','max_allocated_bytes','max_reserved_bytes','file','sha256')}
        rows.append(dict(index=index,sid=row['sid'],n=row['n'],qtype=row['qtype'],pilot_role=row['pilot_role'],
            question=row['question'],atype=row['atype'],typed_gold=typed_gold(row),**owner,
            strata=question_strata(row),scores=scores,native=native))
    for name,digest in sources().items():
        if name in inherited:need(inherited.pop(name)==digest,'Shared own protocol has another hash')
    preparation=auditor.producer.profile.preparation
    recovery=bind(preparation.RECOVERY,bindings,preparation.RECOVERY_SHA)
    world_overlap=bind(recovery['overlap_file'],bindings,recovery['artifacts'][recovery['overlap_file']])
    prefix_overlap=bind(recovery['prefix_overlap_file'],bindings,recovery['artifacts'][recovery['prefix_overlap_file']])
    historical_overlap=dict(recovery_plan_file=str(preparation.RECOVERY),recovery_plan_sha256=preparation.RECOVERY_SHA,
        world_overlap_file=recovery['overlap_file'],world_overlap_sha256=sha(recovery['overlap_file']),
        prefix_overlap_file=recovery['prefix_overlap_file'],prefix_overlap_sha256=sha(recovery['prefix_overlap_file']),
        full_recovery_rows=world_overlap['rows'],selected_pilot=world_overlap['selected_pilot'],
        cross_official_split_world_groups=len(world_overlap['cross_official_split_world_groups']),
        original_prefix_groups=prefix_overlap['groups_count'],original_cross_split_prefix_groups=prefix_overlap['cross_official_split_groups'],
        original_inventory_scope_is_all_recovered_tasks_lengths=True,no_new_overlap_reconstruction=True)
    return rows,audits,summaries,descriptors,inherited,historical_overlap


def costs(rows,audits,summaries,bindings):
    reference=audits['ordinary']['resources']['shared_evaluation_features']
    need(all(a['resources']['shared_evaluation_features']==reference for a in audits.values()),'Shared measured feature costs differ across arms')
    stage=reference['stage'];feature_summary=bind(stage['file'],bindings,stage['sha256'])
    timing=bind(reference['timings_file'],bindings,reference['timings_sha256'])
    need(timing==reference['timings'] and feature_summary['passed'] is feature_summary['completed'] is True
         and feature_summary['phase']=='harvest' and feature_summary['vision_calls']==1000,'Complete shared evaluation vision accounting required')
    shared=feature_summary['elapsed_seconds'];need(math.isfinite(shared) and shared>0,'Measured feature-stage elapsed missing')
    per_arm={}
    for arm in ARMS:
        resource=audits[arm]['resources'];inference=sum(r['native'][arm]['seconds'] for r in rows)
        tokens=sum(r['scores'][arm]['output_tokens'] for r in rows)
        need(math.isclose(inference,resource['inference_seconds'],rel_tol=1e-12,abs_tol=1e-8)
             and resource['actual_output_tokens']==tokens,'Audited inference cost/output-token totals differ')
        per_arm[arm]=dict(audited_resources=resource,training_exposure=dict(epochs=3,scenes_per_epoch=4000,
                native_scene_forward_backward=12000,optimizer_updates=1500,source='audited fixed main recipe'),
            cpu_evaluation_audit_elapsed_seconds=summaries[arm]['elapsed_seconds'],
            instrumented_cold_trajectory_seconds=inference,actual_output_tokens=tokens,
            deployment_shared_feature_stage_seconds=shared,cold_trajectories_plus_feature_stage_seconds=inference+shared,
            separate_evaluation_setup_seconds=resource.get('setup_seconds'),
            complete_evaluation_elapsed_seconds=resource.get('evaluation_elapsed_seconds'),
            peak_allocated_bytes=max(r['native'][arm]['max_allocated_bytes'] for r in rows),
            peak_reserved_bytes=max(r['native'][arm]['max_reserved_bytes'] for r in rows),
            equal_total_parameters_or_compute_claim=False,
            allocated_memory_parameters=0 if arm=='ordinary' else 469504,
            active_memory_parameters=0 if arm=='ordinary' else (465920 if arm=='normalized' else 469504),
            language_adapter_parameters=10092544)
    return dict(per_arm=per_arm,shared_evaluation_features=reference,
        experimental_shared_feature_stage_seconds_counted_once=shared,
        experimental_cold_trajectories_plus_one_feature_stage_seconds=shared+sum(v['instrumented_cold_trajectory_seconds'] for v in per_arm.values()),
        standalone_feature_cost_not_divided_by_three=True,instrumented_cold_not_production_latency=True,
        captures_hashing_and_publication_included=True,flops_not_measured=True,
        derived_sum_excludes_separately_reported_evaluation_setup=True,
        unavailable_timings_left_null=True,no_independent_question_reuse=True)


def overlap_inventory(rows):
    groups=defaultdict(list)
    for row in rows:groups[row['world_sha256']].append(row)
    repeated=[]
    for world,members in sorted(groups.items()):
        if len(members)>1:
            repeated.append(dict(world_sha256=world,indices=[r['index'] for r in members],
                sids=[r['sid'] for r in members],panels=sorted({r['pilot_role'] for r in members}),
                qtypes=sorted({r['qtype'] for r in members})))
    return dict(evaluation_rows=len(rows),full_world_clusters=len(groups),repeated_world_groups=repeated,
        cross_panel_world_groups=sum(len(v['panels'])>1 for v in repeated),
        historical_test_use=True,prefix_dependence_not_removed_by_full_world_bootstrap=True,
        original_training_overlap_not_recomputed=True)


def plot_results(tables,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.9),sharey=True)
    colors={'ordinary':'#2369a8','normalized':'#36935b','mass':'#cc6c21'}
    for ax,qs,title in zip(axes,(PRIMARY,(PRIMARY[0],),(PRIMARY[1],)),('Aggregation macro','Spend together','Where spend')):
        for ai,arm in enumerate(ARMS):
            x=[i+(ai-1)*.055 for i in range(3)];scores=[];intervals=[]
            for n in (8,16,32):
                table=tables[key('test',(n,),qs)];scores.append(100*table['points'][arm])
                intervals.append([100*v for v in table['intervals'][arm]])
            ax.plot(x,scores,marker='o',markersize=4,color=colors[arm],label=arm)
            for pos,(lo,hi) in zip(x,intervals):
                ax.vlines(pos,lo,hi,color=colors[arm],linewidth=1.2)
                ax.hlines([lo,hi],pos-.025,pos+.025,color=colors[arm],linewidth=1.2)
        ax.set_xticks(range(3),['N8','N16','N32']);ax.set_ylim(0,100);ax.set_title(title);ax.set_xlabel('Test sequence length')
        ax.grid(axis='y',alpha=.22)
    axes[0].set_ylabel('Complete typed JSON + EOS accuracy (%)')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,bbox_to_anchor=(.5,.045))
    fig.suptitle('Original MMReD: fixed final checkpoints (exploratory historical test)')
    fig.text(.5,.008,'Paired full-world bootstrap 95% percentile intervals; greedy cold generation, maximum50 tokens.',
        ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.13,1,.91))
    paths=[]
    for suffix in ('pdf','png'):
        file=out/('test_accuracy.'+suffix);fig.savefig(file,dpi=200,bbox_inches='tight');paths.append(str(file))
    plt.close(fig);return dict(matplotlib_version=matplotlib.__version__,files=paths)


def markdown_report(tables,flags,cost_table):
    lines=['# Original MMReD fixed-checkpoint pilot','',
        'All400 validation and600 test rows were retained for each final arm. Scores require complete typed JSON and native EOS. These are exploratory historical-test results.','',
        'Promising mass-specific pilot heuristic: **'+str(flags['promising_mass_specific_pilot'])+'**. This label does not authorize another experiment or establish five-point noninferiority.','',
        '| Test panel | Ordinary | Normalized | Mass | Mass − ordinary, pp (95% CI) |',
        '|---|---:|---:|---:|---:|']
    for n in (8,16,32):
        table=tables[key('test',(n,),PRIMARY)];p=table['points'];ci=table['intervals']['mass_minus_ordinary']
        lines.append('| N'+str(n)+' primary macro | '+' | '.join(f'{100*p[a]:.1f}%' for a in ARMS)+
            f' | {100*p["mass_minus_ordinary"]:.1f} [{100*ci[0]:.1f}, {100*ci[1]:.1f}] |')
    lines+=['','Individual prospective conditions:','']
    for name,value in {**flags['ordinary_competence_conditions'],**flags['mass_pilot_conditions']}.items():lines.append('- '+name+': '+str(value))
    lines+=['','All30 task/macro analyses, every arm/contrast interval, answer-prior/question strata and unrounded replicate files are bound in analysis.json and artifacts.json.',
        'The ordinary validation count-control50/100 reference is descriptive: '+str(flags['ordinary_val_count_ge_50_of_100_descriptive'])+'.',
        'Native cold timings include measurement/capture/publication overhead. Shared feature construction is counted once in the experiment and once per standalone arm; timings are not production latency or FLOPs.',
        'The pilot uses one fit seed and historically used tests. New-world/two-seed confirmation, relevant-load controls and plain/reasoning replication remain required.','']
    return '\n'.join(lines)


def snapshot(out,inherited):
    own=sources();(out/'source').mkdir()
    for name,digest in {**inherited,**own}.items():
        need(sha(REPO/name)==digest,'Current source differs from audited source closure')
        target=out/'source'/name.replace('/','_');target.write_bytes((REPO/name).read_bytes());need(sha(target)==digest,'Archived source differs')
    return own


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--audits',nargs=3,type=Path,required=True);args=parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu'
         and int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==4 and not os.environ.get('SLURM_JOB_GPUS'),
         'Statistics/plotting requires the fixed CPU Slurm allocation')
    started=time.perf_counter();out=OUT/f'report_{os.environ["SLURM_JOB_ID"]}';data=DATA/out.name
    out.mkdir(parents=True,exist_ok=False);data.mkdir(parents=True,exist_ok=False);bindings={};progress={}
    save(out/'request.json',dict(protocol=PROTOCOL,policy=POLICY,audits=[str(p.resolve()) for p in args.audits],source_sha256=sources()))
    try:
        import numpy as np
        rows,audits,summaries,descriptors,inherited,historical_overlap=inputs(args.audits,bindings);own=snapshot(out,inherited)
        tests=self_test(np);save(out/'tests.json',tests);save(out/'joined_predictions.json',rows)
        save(out/'world_groups.json',dict(overlap_inventory(rows),recorded_historical_overlap=historical_overlap));save(out/'descriptive_strata.json',descriptive(rows))
        tables={};replicate_files={};clusters={}
        for role,ns,qs in analysis_specs():
            check_time(started);table,values=bootstrap(np,rows,role,ns,qs)
            name=table['key'];clusters[name]=table.pop('strata')
            file=data/(name+'.npy')
            with file.open('xb') as stream:np.save(stream,values,allow_pickle=False)
            replicate_files[name]=dict(file=str(file),sha256=sha(file),bytes=file.stat().st_size,
                shape=list(values.shape),dtype=str(values.dtype),columns=list(COLUMNS))
            tables[name]=table;progress['completed_analyses']=len(tables)
        need(len(tables)==30 and all(t['replicates']==10000 for t in tables.values()),'Exact30x10000 bootstrap population required')
        save(out/'cluster_inventories.json',clusters);save(out/'replicate_files.json',replicate_files);save(out/'tables.json',tables)
        flags=qualification(tables);cost_table=costs(rows,audits,summaries,bindings)
        save(out/'qualification.json',flags);save(out/'costs.json',cost_table)
        os.environ['MPLCONFIGDIR']=str(data/'matplotlib');os.environ['MPLBACKEND']='Agg'
        plotted=plot_results(tables,out)
        packages=dict(python=sys.version,numpy=np.__version__,matplotlib=plotted['matplotlib_version'])
        result=dict(protocol=PROTOCOL,passed=True,completed=True,policy=POLICY,source_sha256=own,inherited_source_sha256=inherited,
            audits=descriptors,packages=packages,rows_per_arm=1000,total_predictions=3000,analysis_count=30,
            data_bootstrap_replicates=300000,tests=tests,tables_file=str(out/'tables.json'),qualification=flags,
            cluster_inventories_file=str(out/'cluster_inventories.json'),replicate_files=replicate_files,
            predictions_file=str(out/'joined_predictions.json'),strata_file=str(out/'descriptive_strata.json'),
            costs_file=str(out/'costs.json'),plots=plotted,all_efficacy_outcomes_retained=True,
            model_or_head_replay=False,raw_tensor_payloads_reopened=False,metadata_only_independent_audit_handoff=True,no_followup_release=True)
        save(out/'analysis.json',result);(out/'REPORT.md').write_text(markdown_report(tables,flags,cost_table))
        save(out/'input_bindings.json',bindings)
        artifacts={str(file):sha(file) for root in (out,data) for file in root.iterdir() if file.is_file()}
        save(out/'artifacts.json',artifacts);check_time(started)
        need(sources()==own and all(sha(REPO/n)==h for n,h in inherited.items()),'Source changed during statistics report')
        save(out/'summary.json',dict(protocol=PROTOCOL,passed=True,completed=True,policy=POLICY,
            source_sha256=own,inherited_source_sha256=inherited,analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),
            input_bindings_file=str(out/'input_bindings.json'),input_bindings_sha256=sha(out/'input_bindings.json'),
            artifacts_file=str(out/'artifacts.json'),artifacts_sha256=sha(out/'artifacts.json'),
            elapsed_seconds=time.perf_counter()-started,promising_mass_specific_pilot=flags['promising_mass_specific_pilot'],
            efficacy_is_not_completion_gate=True,no_followup_release=True))
    except BaseException as exc:
        save(out/'failure_input_bindings.json',bindings)
        save(out/'failure.json',dict(protocol=PROTOCOL,passed=False,type=type(exc).__name__,message=str(exc),
            elapsed_seconds=time.perf_counter()-started,progress=progress,input_bindings_file=str(out/'failure_input_bindings.json'),
            input_bindings_sha256=sha(out/'failure_input_bindings.json'),partial_outputs_retained=True,no_followup_release=True));raise


if __name__=='__main__':main()

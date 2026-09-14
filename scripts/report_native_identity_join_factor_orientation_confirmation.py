"""Independent fixed N16 orientation-confirmation audit; no fitting or native-head replay.

Every raw emitted token and factor/conditioning/native write is audited. The
previous native profiles supply head fidelity. A/B qualification only makes
both fixed endpoints eligible for a separately released longer-length stage.
"""
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from collections import Counter,defaultdict
import math
import os
import subprocess
from scripts import report_native_identity_join_factor_orientation_native as native_audit
from scripts import report_native_identity_join_factor_binding as factor
from scripts.stage_native_vision_v6_teacher import need,read,save,sha,object_sha
ARMS=('product','additive');PANELS=('A','B','C');COUNTS={'A':108,'B':108,'C':54}
bind=native_audit.bind;score=native_audit.score;capture=native_audit.capture
audit_history=native_audit.audit_history;source_archive=native_audit.source_archive


def driver():
    from scripts import evaluate_native_identity_join_factor_orientation_confirmation as module
    return module


def criteria(rows):
    need(len(rows)==len({r['sid'] for r in rows})==270 and all(r['n_frames']==16 for r in rows)
         and Counter(r['panel'] for r in rows)==COUNTS,'All270 N16 fresh contexts are required')
    results={}
    for panel in PANELS:
        values=[r for r in rows if r['panel']==panel];groups=defaultdict(list)
        for row in values:groups[row['contrast_id']].append(row)
        families=COUNTS[panel]//3
        need(len(groups)==families and all(len(group)==3 and {r['variant'] for r in group}=={0,1,2} for group in groups.values()),'Every fresh family needs all three answer-changing variants')
        correct=sum(r['exact'] for r in values);complete=sum(all(r['exact'] for r in group) for group in groups.values())
        answer_threshold,family_threshold=(49,16) if panel=='C' else (98,33)
        results[panel]=dict(passed=correct>=answer_threshold and complete>=family_threshold,whole_correct=correct,
            first_correct=sum(r['first_token_correct'] for r in values),contexts=len(values),complete_families=complete,families=families,
            thresholds=dict(whole_correct=answer_threshold,complete_families=family_threshold),
            effective_minimum_whole_correct=max(answer_threshold,3*family_threshold))
    return dict(panels=results,qualifies_AB=results['A']['passed'] and results['B']['passed'],panel_C_passed=results['C']['passed'],
        natural_whole_answer_and_eos=True,all_panels_retained=True,no_checkpoint_selection=True,longer_lengths_require_separate_release=True)


def self_test(torch):
    rows=[dict(sid=f'{p}_{f}_{v}',panel=p,n_frames=16,contrast_id=f'{p}_{f}',variant=v,exact=True,first_token_correct=True)
          for p in PANELS for f in range(COUNTS[p]//3) for v in range(3)]
    good=criteria(rows);need(good['qualifies_AB'] and good['panel_C_passed'],'Complete fresh panel fixture failed')
    # 99 correct can fail when errors touch nine different triples.
    scattered=[dict(r) for r in rows]
    for index in range(0,27,3):scattered[index]['exact']=False
    c=criteria(scattered);need(c['panels']['A']['whole_correct']==99 and c['panels']['A']['complete_families']==27 and not c['qualifies_AB'],'Family criterion must remain independent of marginal accuracy')
    boundary=[dict(r) for r in rows]
    for index in range(9):boundary[index]['exact']=False
    c=criteria(boundary);need(c['panels']['A']['whole_correct']==99 and c['panels']['A']['complete_families']==33 and c['qualifies_AB'],'Literal98/33 conjunction boundary differs')
    boundary[9]['exact']=False;need(not criteria(boundary)['qualifies_AB'],'98 correct cannot satisfy33 complete triples')
    separate=[dict(r) for r in rows]
    for row in separate:
        if row['panel']=='C':row['exact']=False
    need(criteria(separate)['qualifies_AB'] and not criteria(separate)['panel_C_passed'],'Panel C must remain separately reported')
    c_boundary=[dict(r) for r in rows]
    for row in c_boundary[216:221]:row['exact']=False
    need(criteria(c_boundary)['panels']['C']['whole_correct']==49 and criteria(c_boundary)['panel_C_passed'],'C49 answers with16 triples must pass')
    c_boundary[221]['exact']=False;need(criteria(c_boundary)['panels']['C']['complete_families']==16 and not criteria(c_boundary)['panel_C_passed'],'C48 answers must fail even with16 triples')
    need(score([15,151645],[15,151645])['exact'] and not score([15,151643],[15,151645])['exact']
         and not score([15,16,17,18],[15,151645])['exact'],'Canonical full-target EOS fixture failed')
    return dict(passed=True,groups=4,complete_panels=True,family_and_answer_boundaries=True,separate_question_transfer=True,exact_native_EOS=True)


def strata(rows):
    tables={}
    for keys in (('panel',),('panel','gold'),('panel','question'),('panel','orientation'),('panel','reference_orientation'),('panel','matches_reference_orientation'),('panel','variant')):
        groups=defaultdict(list)
        for row in rows:groups[tuple(row[k] for k in keys)].append(row)
        tables['_'.join(keys)]=[dict(labels=dict(zip(keys,key)),metrics=dict(contexts=len(values),whole_correct=sum(r['exact'] for r in values),
            first_correct=sum(r['first_token_correct'] for r in values),completed=sum(r['completed'] for r in values),truncated=sum(r['truncated'] for r in values),
            noncanonical_answers=sum(r['noncanonical_answer'] for r in values),wrong_canonical_names=sum(r['wrong_canonical_name'] for r in values),early_EOS_relative_to_target=sum(r['early_eos_relative_to_target'] for r in values),
            first_query_nll=sum(r['first_query_nll'] for r in values)/len(values))) for key,values in sorted(groups.items(),key=lambda item:repr(item[0]))]
    return tables




def families(rows):
    groups=defaultdict(list)
    for row in rows:groups[(row['panel'],row['contrast_id'])].append(row)
    need(len(groups)==90,'Every confirmation family must be retained')
    result=[]
    for (panel,contrast),values in sorted(groups.items()):
        need(len(values)==3 and {r['variant'] for r in values}=={0,1,2},'Incomplete confirmation family')
        first=values[0];result.append(dict(labels=dict(panel=panel,contrast_id=contrast,
            family_instance_sha256=first['family_instance_sha256'],orientation=first['orientation'],
            reference_orientation=first['reference_orientation'],matches_reference_orientation=first['matches_reference_orientation']),
            metrics=dict(contexts=3,whole_correct=sum(r['exact'] for r in values),first_correct=sum(r['first_token_correct'] for r in values),
                complete_family=all(r['exact'] for r in values)),sids=[r['sid'] for r in values]))
    return result


def inputs(torch,plan):
    from scripts import stage_native_identity_join_factor_orientation_confirmation as stage
    p=driver();native_ref=plan['native_report'];stage_ref=plan['stage_report']
    bind(native_ref['file'],native_ref['sha256']);native_result=native_audit.verify_native_report(native_ref['file'])
    bind(native_ref['analysis_file'],native_ref['analysis_sha256']);bind(native_ref['plan_file'],native_ref['plan_sha256'])
    need(native_result==read(native_ref['analysis_file']) and native_result['plan_file']==native_ref['plan_file'] and native_result['plan_sha256']==native_ref['plan_sha256'],'Original completed native report differs')
    native_plan=read(native_ref['plan_file']);bind(stage_ref['file'],stage_ref['sha256']);manifest=stage.verify_stage(stage_ref['file'])
    bind(stage_ref['manifest_file'],stage_ref['manifest_sha256']);need(manifest==read(stage_ref['manifest_file']),'Published fresh manifest differs')
    need(stage_ref==native_plan['confirmation_report'] and manifest['training_stage']==native_plan['training_stage']==plan['training_stage']
         and manifest['seed']==91726342 and manifest['manifest_required_before_fitting'] is True
         and manifest['new_model_trainability_established'] is False
         and manifest['native_reference']['new_model_trainability_established'] is False
         and manifest['native_reference']['native_identity']==native_plan['native_identity'],
         'Confirmation must be the exact new manifest bound before these models were fitted; historical native evidence is not current efficacy')
    rows=read(plan['rows_file']);expected=[r for panel in PANELS for r in manifest['splits'][f'fresh_{panel}_N16']['samples']]
    need(rows==expected and len(rows)==270 and len({r['sid'] for r in rows})==270 and Counter(r['panel'] for r in rows)==COUNTS,'Fixed complete fresh N16 manifest order differs')
    groups=defaultdict(list)
    for row in rows:
        need(row['n_frames']==16 and row['target_ids']==manifest['tokenizer']['targets'][row['gold']]['target_ids'] and row['target_ids'][-1]==151645,'Actual canonical full-name target differs')
        need(row['orientation'] in (0,1) and row['reference_orientation'] in (None,0,1)
             and row['matches_reference_orientation'] is (None if row['reference_orientation'] is None else row['orientation']==row['reference_orientation']),
             'Reference-orientation label relation differs')
        groups[(row['panel'],row['contrast_id'])].append(row)
    need(len(groups)==90 and all(len(g)==3 and {r['variant'] for r in g}=={0,1,2} and len({r['gold'] for r in g})==3
         and len({r['family_instance_sha256'] for r in g})==1 for g in groups.values()),'Complete answer-changing concrete triples differ')
    need(Counter(r['matches_reference_orientation'] for r in rows if r['panel']=='A')=={True:54,False:54}
         and all(r['reference_orientation'] is r['matches_reference_orientation'] is None for r in rows if r['panel'] in ('B','C')),
         'Reference-orientation strata differ; both A orientations were trained')
    prepared=read(plan['prepared_file']);bind(plan['layout_file'],plan['layout_sha256']);layouts=torch.load(plan['layout_file'],map_location='cpu',weights_only=True)
    need(set(prepared)==set(layouts)=={r['sid'] for r in rows},'Exact270 prepared/layout identities required')
    for row in rows:
        item=prepared[row['sid']];meta=item['metadata'];layout=layouts[row['sid']]
        need(meta['sid']==row['sid'] and meta['question']==row['question'] and meta['n_frames']==16 and meta['prefix_ids']==[]
             and meta['image_paths']==[i['path'] for i in row['image_files']] and meta['image_sha256']==[i['sha256'] for i in row['image_files']]
             and item['layout']==layout['metadata'] and item['layout']['every_unpadded_row_exact'] is True
             and p.tensor_info(layout['position_ids'])==item['layout']['position_ids'] and layout['rope_deltas'].tolist()==item['layout']['rope_deltas'], 'Fresh native preprocessing or position ownership differs')
        need(set(stage.runtime_view(row))=={'sid','n_frames','question','image_files'},'Offline fresh metadata entered the model view')
    for key in ('cached_report','cached_plan','stats_file','stats_sha256','conditioning_identity','native_model_file','native_model_sha256','native_identity','native_identity_sha256','native_module_identity','native_software_plan'):
        need(plan[key]==native_plan[key],'Existing deployed native/statistic input differs: '+key)
    bind(plan['stats_file'],plan['stats_sha256']);stats=torch.load(plan['stats_file'],map_location='cpu',weights_only=True)
    need({k:p.tensor_info(stats[k]) for k in ('global_mean','scale')}==plan['conditioning_identity'],'Actual frozen global mean/scale differ')
    weights={}
    for arm in ARMS:
        need(set(plan['checkpoints'][arm])=={'fitted'},'Only fixed fitted endpoints may enter fresh inference')
        d=plan['checkpoints'][arm]['fitted'];parent=native_plan['checkpoints'][arm]['fitted'];bind(d['file'],d['sha256'])
        need(d['sha256']==parent['sha256'] and d['state']==parent['state']==native_result['runs'][arm]['checkpoint']['state'],'Actual step6000 deployed checkpoint differs')
        w=torch.load(d['file'],map_location='cpu',weights_only=True)['branch'];factor.check_weights(torch,w,w)
        need({k:p.tensor_info(v) for k,v in w.items()}==d['state'],'Fresh checkpoint tensor identities differ');weights[arm]={'fitted':w}
    return rows,prepared,layouts,stats,weights,native_result,manifest


def audit_run(torch,out,directory,plan_path,plan,rows,prepared,layouts,stats,weights,tokenizer):
    p=driver();phase='run';directory=Path(directory).resolve();summary=read(directory/'summary.json');config=read(directory/'config.json');arm=config['arm']
    need(arm in ARMS and config['phase']==summary['phase']==phase and directory.name==f'{phase}_{arm}_{config["slurm_job_id"]}'
         and summary['passed'] is summary['completed'] is True and all(summary[k]==v for k,v in config.items()),'Completed native run/config ownership differs')
    source_archive(directory,plan)
    for key in ('protocol','policy','source_sha256','inherited_source_sha256','cached_report','cached_plan','native_identity','native_identity_sha256','conditioning_identity','stats_file','stats_sha256','native_report','stage_report','timing_certificate','training_stage'):
        need(config[key]==plan[key],'Native configuration differs: '+key)
    need(config['plan_file']==str(plan_path) and config['plan_sha256']==sha(plan_path) and summary['no_fitting'] is summary['no_accuracy_selection'] is summary['all_raw_retained_before_scoring'] is True
         and summary['extra_gpu_head_calls']==0,'Native plan or no-fit/raw-retention contract differs')
    need(config['hardware']['gpu']=='NVIDIA B200' and config['native_weight_identity']==dict(norm=plan['native_identity']['norm_weight'],head=plan['native_identity']['head_weight']), 'Actual hardware/native weights differ')
    need('main_release' not in config and config['N16_only'] is True,'Fresh timing release is bound in the fixed preparation plan')
    cases=[dict(sid=r['sid'],n_frames=16,state='fitted') for r in rows]
    need(config['cases']==cases and summary['trajectories']==len(cases),'Exact natural trajectory cohort differs')
    need(set(config['checkpoints'])=={'fitted'},'Deployed checkpoint inventory differs')
    for state,descriptor in config['checkpoints'].items():
        parent=plan['checkpoints'][arm][state];bind(descriptor['file'],descriptor['sha256'])
        need(Path(descriptor['file'])==Path(config['checkpoint_directory'])/(state+'.pt') and descriptor['sha256']==parent['sha256'] and descriptor['state']==parent['state'], 'Deployed checkpoint copy differs')
    bind(summary['raw_manifest_file'],summary['raw_manifest_sha256']);manifest=read(summary['raw_manifest_file'])
    need(Path(summary['raw_manifest_file'])==directory/'raw_manifest.json' and manifest['completed'] is manifest['all_raw_retained_before_scoring'] is True
         and manifest['n']==len(cases)==len(manifest['rows']),'All raw native trajectories must precede scoring')
    lookup={r['sid']:r for r in rows};canonical_targets={tuple(r['target_ids']) for r in rows};need(len(canonical_targets)==9,'All nine canonical names required');outcomes=[];details=[];counts=Counter();head_calls=head_rows=0;max_tv=0.;auditfiles=[]
    for j,(case,record) in enumerate(zip(cases,manifest['rows'])):
        sid=case['sid'];sample=lookup[sid];item=prepared[sid];n=sample['n_frames'];state=case['state'];w=weights[arm][state]
        need(record['index']==j and record['case']==case and record['prepared_file']==item['file'] and record['prepared_sha256']==item['sha256'], 'Trajectory order/input ownership differs')
        bind(record['raw_file'],record['raw_sha256']);need(Path(record['raw_file'])==Path(config['data_directory'])/f'trajectory_{j:03d}.pt','Unexpected raw trajectory path')
        need(read(directory/f'progress_{j+1:03d}.json')==dict(processed=j+1,case=case,raw_file=record['raw_file'],raw_sha256=record['raw_sha256'],accuracy_not_scored=True),'Per-trajectory publication record differs')
        packet=torch.load(record['raw_file'],map_location='cpu',weights_only=True);result=packet['result'];raw=result['raw_logits'];ids=result['generated_ids'];t=len(ids)
        need(packet['schema_version']==1 and packet['case']==case and packet['core_state']==plan['checkpoints'][arm][state]['state'] and packet['conditioning_identity']==plan['conditioning_identity'],'Raw trajectory checkpoint/statistic binding differs')
        bind(item['file'],item['sha256']);bundle=torch.load(item['file'],map_location='cpu',weights_only=True);p.runtime.validate_bundle(bundle)
        meta=result['metadata'];need(all(meta[k]==v for k,v in bundle['metadata'].items()) and meta['layout']==item['layout'] and meta['interaction']==result['interaction']==arm
             and meta['conditioning_identity']==plan['conditioning_identity'] and meta['native_identity_sha256']==plan['native_identity_sha256']
             and meta['scene_input_identity']==p.runtime.input_identity(bundle,plan['native_identity_sha256']), 'Original native scene/model/statistic identity differs')
        generation=meta['generation'];need(generation['max_new_tokens']==4 and generation['do_sample'] is False and generation['num_beams']==1
             and generation['repetition_penalty']==1. and generation['native_eos_token_ids']==[151645,151643]
             and generation['output_logits'] is generation['output_scores'] is False and generation['global_row_streaming'] is True and generation['vocabulary_mask'] is generation['other_logits_processors'] is False
             and generation['target_eos_token_id']==151645 and generation['pad_token_id']==tokenizer.pad_token_id, 'Natural native generation policy differs')
        need(raw.dtype==torch.float32 and raw.shape==(t,152064) and bool(torch.isfinite(raw).all()) and torch.equal(raw,raw.half().float()) and raw.argmax(-1).tolist()==ids,'Unmasked raw native argmax differs')
        scored=score(ids,sample['target_ids']);need(all(result[k]==scored[k] for k in ('completed','truncated','finish_reason')) and result['text']==tokenizer.decode(ids,skip_special_tokens=True)
             and result['raw_text']==tokenizer.decode(ids,skip_special_tokens=False),'Native stopping or decoded text differs')
        expected=dict(model=t,visual=1,language=t,norm=t,head=t,broadcast=t,fusion=t,conditioning=t,probe_head=0)
        need(result['counters']==expected and record['generated_tokens']==t and result['capture_call_indices']==list(range(1,t+1))
             and len(result['captures'])==len(result['logit_records'])==t,'Every natural token needs exactly one native and factor call')
        fa=result['fusion_audit'];need(fa==dict(interaction=arm,pairing='paired',calls=t,conditioning_calls=t,last_capture_call=t,zero_up=state=='zero',zero_up_identity_checks=t if state=='zero' else 0,
            parameter_versions_unchanged=True,statistics_versions_unchanged=True,active=False,conditioning_active=False),'Actual controller closure/state audit differs')
        for k,lr in enumerate(result['logit_records']):
            need(lr['step']==k and lr['top1_token_id']==ids[k] and lr['top1_logit']==float(raw[k,ids[k]]) and lr['native_dtype']=='torch.float16'
                 and lr['vocabulary_size']==152064 and math.isfinite(lr['log_normalizer']),'Raw native recorder differs')
        history=audit_history(torch,packet,bundle,layouts[sid],result)
        math_rows=[capture(torch,cap,w,stats['global_mean'],stats['scale'],arm,n,state=='zero') for cap in result['captures']]
        replay=[];observations=packet['profile_native']
        need(observations is None,'Fresh inference must not add full-row native-head replay')
        audit=dict(passed=all(x['passed'] for x in math_rows+replay),case=case,history=history,functional_captures=math_rows,native_head_replays=replay,
            cpu_head_replay_scope='none; inherited profile fidelity and every fresh factor/cast/raw-global-output audited')
        auditfile=out/f'{phase}_{arm}_trajectory_{j:03d}_audit.json';save(auditfile,audit);auditfiles.append(dict(file=str(auditfile),sha256=sha(auditfile)))
        need(audit['passed'],'Independent native capture or registered profile head gate failed; audit retained')
        need(record['audit']==dict(passed=True,emitted_tokens=t,counters=expected,zero_up_identity=state=='zero',raw_unmasked_argmax=True),'GPU trajectory audit differs')
        timing=[record['preprocessing_seconds'],record['work_seconds'],record['four_token_seconds']]
        need(all(math.isfinite(x) and x>0 for x in timing) and record['four_token_seconds']==timing[0]+timing[1]*4/t,'Measured complete trajectory timing differs')
        nll=float(torch.logsumexp(raw[0].double()-raw[0].double().max(),0)-(raw[0,sample['target_ids'][0]].double()-raw[0].double().max()))
        outcomes.append(dict(**{key:sample[key] for key in ('sid','contrast_id','pair_id','variant','n_frames','question','gold','target_ids','panel','orientation','reference_orientation','matches_reference_orientation','trio','room_pair','family_instance_sha256')},state=state,
            generated_ids=ids,text=result['text'],raw_text=result['raw_text'],first_query_nll=nll,
            noncanonical_answer=tuple(ids) not in canonical_targets,wrong_canonical_name=tuple(ids) in canonical_targets and not scored['exact'],early_eos_relative_to_target=scored['completed'] and len(ids)<len(sample['target_ids']),**scored))
        details.append(dict(index=j,case=case,raw_file=record['raw_file'],raw_sha256=record['raw_sha256'],generated_tokens=t,four_token_seconds=record['four_token_seconds'],
            matched_gold_prefix_queries=sum(ids[:k]==sample['target_ids'][:k] for k in range(t)),observed_query_prefixes=[ids[:k] for k in range(t)]))
        counts.update(expected);del packet,result,raw,bundle,math_rows,observations
    bind(summary['endpoint_file'],summary['endpoint_sha256']);endpoint=read(summary['endpoint_file']);native_table=config['native_weight_identity']
    need(endpoint['passed'] is endpoint['native_parameter_versions_unchanged'] is endpoint['statistics_versions_unchanged'] is endpoint['all_trajectory_checkpoint_versions_unchanged'] is endpoint['no_optimizer_or_training'] is True
         and endpoint['native_weights_before']==endpoint['native_weights_after']==native_table
         and endpoint['conditioning_before']==endpoint['conditioning_after']==plan['conditioning_identity']
         and endpoint['final_core_state']==plan['checkpoints'][arm]['fitted']['state'] and endpoint['counters']==summary['counters']==dict(counts),'Actual final native/core/statistic identity differs')
    need(counts['visual']==len(cases) and counts['model']<=1080 and math.isfinite(summary['setup_seconds']) and summary['setup_seconds']>0,'Native phase resource inventory differs')
    outcome_file=out/f'{phase}_{arm}_outcomes.json';save(outcome_file,outcomes)
    return dict(passed=True,arm=arm,phase=phase,directory=str(directory),summary_sha256=sha(directory/'summary.json'),slurm_job_id=config['slurm_job_id'],
        raw_manifest_file=summary['raw_manifest_file'],raw_manifest_sha256=summary['raw_manifest_sha256'],endpoint_file=summary['endpoint_file'],endpoint_sha256=summary['endpoint_sha256'],
        outcomes_file=str(outcome_file),outcomes_sha256=sha(outcome_file),trajectory_audits=auditfiles,trajectories=details,counters=dict(counts),
        setup_seconds=summary['setup_seconds'],four_token_seconds=[r['four_token_seconds'] for r in manifest['rows']],
        native_head_rows=sum(r['generated_tokens']*(r['case']['n_frames']+1) for r in details),cpu_head_calls=head_calls,cpu_head_rows=head_rows,maximum_cpu_native_head_tv=None,all_profile_native_head_argmax_exact=None,
        fresh_n16_screen=criteria(outcomes),no_new_head_calls=True,
        checkpoint=plan['checkpoints'][arm]['fitted'],native_identity_sha256=plan['native_identity_sha256'],outcomes=outcomes)



def timing_audit(plan,native_result):
    observations=[]
    for phase,key,count in (('profile','profiles',8),('main','runs',216)):
        need(set(native_result[key])==set(ARMS),'Both completed native arms are required for the timing pool')
        for arm in ARMS:
            value=native_result[key][arm];need(value['phase']==phase and value['arm']==arm and len(value['four_token_seconds'])==count,'Original measured trajectory denominator differs')
            observations.append(dict(phase=phase,arm=arm,summary_file=str(Path(value['directory'])/'summary.json'),summary_sha256=value['summary_sha256'],
                setup_seconds=value['setup_seconds'],four_token_seconds=value['four_token_seconds']))
    setups=[r['setup_seconds'] for r in observations];times=[t for r in observations for t in r['four_token_seconds']]
    need(len(setups)==4 and len(times)==448 and all(math.isfinite(x) and x>0 for x in setups+times),'Complete positive native timings are required')
    setup=max(setups);observed=max(times);floor=2.30838355794549;bound=max(floor,observed);seconds=setup+1.25*270*bound+60.
    expected=dict(passed=seconds<=1200,observations=observations,setup_observations=4,trajectory_observations=448,pooled_setup_seconds=setup,
        maximum_observed_four_token_seconds=observed,historical_floor_seconds=floor,pooled_scene_seconds=bound,multiplier=1.25,
        contexts=270,reserve_seconds=60.,projected_seconds=seconds,cap_seconds=1200,
        all_native_profiles_mains_both_arms_and_zero_cases=True,physical_runtime_guarantee=False)
    need(expected['passed'] and plan['timing_certificate']==expected,'Independent conservative N16 timing release differs or fails')
    for record in observations:bind(record['summary_file'],record['summary_sha256'])
    return expected


def input_envelope(torch,bundle,layout):
    p=driver();p.runtime.validate_bundle(bundle);m=bundle['metadata'];x=bundle['inputs'];mask=x['attention_mask'];pos=layout['position_ids'];delta=layout['rope_deltas']
    need(m['n_frames']==16 and m['row_count']==17 and m['global_row']==16 and m['row_kinds']==['local']*16+['global']
         and m['resize']==392 and m['prefix_ids']==[] and m['processor_parity_checked'] is True and layout['metadata']['every_unpadded_row_exact'] is True
         and pos.shape==(3,17,m['prompt_width']) and delta.shape==(17,1),'Actual N16 native input conventions differ')
    need(bool(((mask==0)|(mask==1)).all()) and bool((mask[:,1:]>=mask[:,:-1]).all()),'Native binary left padding differs')
    return dict(sid=m['sid'],prompt_width=m['prompt_width'],maximum_local_tokens=max(m['row_prompt_tokens'][:16]),global_tokens=m['row_prompt_tokens'][16],
        input_shapes={k:list(v.shape) for k,v in x.items()},input_dtypes={k:str(v.dtype) for k,v in x.items()},image_grid_thw=x['image_grid_thw'].tolist(),
        position_dtype=str(pos.dtype),delta_dtype=str(delta.dtype),maximum_absolute_position=int(pos.abs().max()),maximum_absolute_rope_delta=int(delta.abs().max()),
        binary_left_padded_mask=True,maximum_new_tokens=4,resize=392,rows=17)


def envelope_audit(torch,plan,rows,prepared,layouts):
    bind(plan['envelope_file'],plan['envelope_sha256']);saved=read(plan['envelope_file']);old_plan=read(plan['native_report']['plan_file'])
    old_rows=[r for r in read(old_plan['rows_file']) if r['n_frames']==16];old_prepared=read(old_plan['prepared_file'])
    bind(old_plan['layout_file'],old_plan['layout_sha256']);old_layouts=torch.load(old_plan['layout_file'],map_location='cpu',weights_only=True)
    original=[];fresh=[]
    for cohort,items,positions,output in ((old_rows,old_prepared,old_layouts,original),(rows,prepared,layouts,fresh)):
        for row in cohort:
            item=items[row['sid']];bind(item['file'],item['sha256']);bundle=torch.load(item['file'],map_location='cpu',weights_only=True)
            need(bundle['metadata']==item['metadata'],'Bound preprocessing metadata differs');output.append(input_envelope(torch,bundle,positions[row['sid']]))
    need(len(original)==108 and len(fresh)==270,'Complete measured native/fresh N16 input envelope required')
    exact_keys=('input_dtypes','image_grid_thw','position_dtype','delta_dtype','binary_left_padded_mask','maximum_new_tokens','resize','rows')
    first=original[0];exact={k:first[k] for k in exact_keys};need(all(all(r[k]==v for k,v in exact.items()) for r in original),'Native reference structural conventions differ')
    maximum={k:max(r[k] for r in original) for k in ('prompt_width','maximum_local_tokens','global_tokens','maximum_absolute_position','maximum_absolute_rope_delta')}
    shapes={k:[max(r['input_shapes'][k][i] for r in original) for i in range(len(v))] for k,v in first['input_shapes'].items()}
    limits=dict(exact=exact,maximum=maximum,maximum_input_shapes=shapes)
    for row in fresh:
        need(all(row[k]==v for k,v in exact.items()) and all(row[k]<=v for k,v in maximum.items()) and set(row['input_shapes'])==set(shapes)
             and all(len(row['input_shapes'][k])==len(v) and all(a<=b for a,b in zip(row['input_shapes'][k],v)) for k,v in shapes.items()), 'Fresh input exceeds the complete validated N16 envelope')
    expected=dict(passed=True,original=original,fresh=fresh,limits=limits,old_contexts=108,fresh_contexts=270,actual_owner_bound_native_layout=True,no_additional_GPU_profile=True)
    need(saved==expected,'Independent actual input envelope differs from released CPU certificate');return dict(passed=True,old_contexts=108,fresh_contexts=270,limits=limits)


def resources(out,runs):
    from datetime import datetime
    p=driver();command=['sacct','-X','-u',os.environ['USER'],'--starttime=2026-09-01','--parsable2','--noheader',
        '--format=JobIDRaw,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,Start,End']
    raw=subprocess.run(command,capture_output=True,text=True,check=True).stdout;file=out/'all_user_sacct.psv';file.write_text(raw)
    names=set(p.JOBS.values());allocations=factor.allocation_rows(raw,names|{'identity_join_factor_orientation_confirmation_eval_run'});expected={p.JOBS[a]:runs[a] for a in ARMS}
    need(len(allocations)==2 and {r['name'] for r in allocations}==names,'Exactly one fresh allocation per arm, including failed/zero attempts, is permitted')
    events=[]
    for row in allocations:
        need(row['job_id']==expected[row['name']]['slurm_job_id'] and row['partition']=='gpu' and row['state']=='COMPLETED'
             and row['exit_code']=='0:0' and row['gpus']==1 and row['seconds']<=1200,'Fresh allocation identity or20-minute cap differs')
        start,end=map(datetime.fromisoformat,(row['start'],row['end']));need(end>=start,'Invalid GPU allocation interval')
        if end>start:events.extend(((start,1),(end,-1)))
    active=peak=0
    for _,change in sorted(events):active+=change;need(active>=0,'Invalid allocation overlap');peak=max(peak,active)
    total=sum(r['gpu_seconds'] for r in allocations);need(active==0 and peak<=2 and total<=2400,'Fresh two-GPU/2400-second campaign cap exceeded')
    return dict(passed=True,jobs=allocations,allocated_gpu_seconds=total,campaign_seconds=2400,maximum_concurrent_gpus=peak,
        failed_and_zero_allocations_retained=True,file=str(file),sha256=sha(file),command=command)


def report(args,out,frozen):
    import torch
    from transformers import AutoProcessor
    p=driver();torch.set_num_threads(4);plan_path=args.plan.resolve();plan=p.verify_plan(plan_path,ancestors=True)
    rows,prepared,layouts,stats,weights,native_result,manifest=inputs(torch,plan);timing=timing_audit(plan,native_result)
    envelope=envelope_audit(torch,plan,rows,prepared,layouts);save(out/'input_release_audit.json',dict(passed=True,timing_certificate=timing,envelope=envelope))
    need(native_result['all_profile_native_head_argmax_exact'] is True and native_result['maximum_profile_cpu_native_head_tv']<=.02,'Prior native head fidelity proof failed')
    for value in native_result['profiles'].values():native_audit.bound_outputs(value)
    profile_fidelity=dict(source_native_report=plan['native_report'],cpu_head_calls=native_result['coverage']['profile_cpu_head_calls'],
        cpu_head_rows=native_result['coverage']['profile_cpu_head_rows'],maximum_cpu_native_head_tv=native_result['maximum_profile_cpu_native_head_tv'],
        all_argmax_exact=True,replayed_in_this_report=False,new_native_head_calls=0)
    processor=AutoProcessor.from_pretrained(str(p.MODEL),trust_remote_code=True,use_fast=False,local_files_only=True);runs={}
    for directory in args.runs:
        value=audit_run(torch,out,directory,plan_path,plan,rows,prepared,layouts,stats,weights,processor.tokenizer)
        need(value['arm'] not in runs,'Duplicate fresh model arm');runs[value['arm']]=value
    need(set(runs)==set(ARMS) and all(v['counters']['visual']==270 and v['cpu_head_calls']==v['cpu_head_rows']==0 and v['native_head_rows']<=18360 for v in runs.values()),'Both complete fixed fresh model arms are required')
    budget=resources(out,runs);screens={a:runs[a]['fresh_n16_screen'] for a in ARMS};eligible=any(s['qualifies_AB'] for s in screens.values())
    left={r['sid']:r for r in runs['product']['outcomes']};right={r['sid']:r for r in runs['additive']['outcomes']};transitions={}
    for panel in PANELS:
        transitions[panel]=dict(Counter('both_correct' if left[s]['exact'] and right[s]['exact'] else 'product_only' if left[s]['exact'] else 'additive_only' if right[s]['exact'] else 'both_wrong'
            for s in left if left[s]['panel']==panel))
    coverage=dict(contexts_per_arm=270,joint_contexts=540,panels={'A':108,'B':108,'C':54},families_per_arm={'A':36,'B':36,'C':18},
        vision_calls=540,native_model_norm_head_calls=sum(v['counters']['model'] for v in runs.values()),native_head_rows=sum(v['native_head_rows'] for v in runs.values()),
        functional_captures=sum(v['counters']['fusion'] for v in runs.values()),new_CPU_head_calls=0,extra_GPU_head_calls=0)
    need(coverage['native_model_norm_head_calls']<=2160 and coverage['native_head_rows']<=36720,'Joint fresh call/row budget differs')
    analysis=dict(passed=True,completed=True,phase='report',protocol=p.PROTOCOL,policy=p.POLICY,source_sha256=frozen,inherited_source_sha256=plan['inherited_source_sha256'],
        plan_file=str(plan_path),plan_sha256=sha(plan_path),native_report=plan['native_report'],stage_report=plan['stage_report'],training_stage=plan['training_stage'],timing_certificate=timing,
        input_release_audit_file=str(out/'input_release_audit.json'),input_release_audit_sha256=sha(out/'input_release_audit.json'),
        runs=runs,fresh_N16_screen=screens,longer_lengths_eligible=eligible,longer_lengths_released=False,
        stop_before_N32_N64=not eligible,both_endpoints_retained=True,N16_only=True,no_fitting_or_new_statistics=True,
        all_raw_native_outputs_retained=True,prior_profile_head_fidelity=profile_fidelity,resources=budget,coverage=coverage,
        strata={a:strata(runs[a]['outcomes']) for a in ARMS},families={a:families(runs[a]['outcomes']) for a in ARMS},descriptive_paired_transitions=transitions,
        ordinary_joint_baseline_present=False,reasoning_model_evidence_present=False,confirmation_seed=91726342,
        both_orientations_trained=True,orientation_strata_reference_only=True,prior_failed_fresh_predictions_reused=False)
    save(out/'analysis.json',analysis)
    lines=['# Fixed N16 orientation confirmation','',
        'Both orientation-trained step6000 checkpoints were evaluated on every fixed seed91726342 confirmation N16 context. No fitting, prompt/statistic change, checkpoint selection, N32/N64 evaluation or additional native-head replay occurred.','',
        '| Arm | Panel | Whole answer + EOS | First token | Complete triples | Panel criterion |',
        '| --- | --- | ---: | ---: | ---: | --- |']
    for arm in ARMS:
        for panel in PANELS:
            s=screens[arm]['panels'][panel];lines.append(f'| {arm} | {panel} | {s["whole_correct"]}/{s["contexts"]} | {s["first_correct"]}/{s["contexts"]} | {s["complete_families"]}/{s["families"]} | {"PASS" if s["passed"] else "FAIL"} |')
    lines+=['','A and B each retain the literal98/108 answer and33/36 triple thresholds; together these require at least99 correct answers. C retains its separate49/54 and16/18 criterion.',
        '',f'A/B qualification: product={screens["product"]["qualifies_AB"]}; additive={screens["additive"]["qualifies_AB"]}. '+('Both endpoints are eligible for a separately reviewed longer-length resource release; none is released here.' if eligible else 'Neither endpoint qualifies: stop before N32/N64.'),
        '', 'All names, questions, variants, orientations, noncanonical outputs, premature EOS and truncations are retained in the row and stratum artifacts. Panel A labels agreement with the original reference orientation; both orientations were included in training. A shorter wrong canonical name is an answer error; the relative-length EOS field is descriptive, with no separate format gate. Paired product/additive transitions are descriptive.',
        '',f'Every one of {coverage["functional_captures"]} emitted-prefix factor/conditioning/native writes and raw argmaxes passed the independent audit. Head fidelity is inherited from {profile_fidelity["cpu_head_calls"]} profile calls/{profile_fidelity["cpu_head_rows"]} native rows, maximum TV {profile_fidelity["maximum_cpu_native_head_tv"]:.9g}, with exact argmax. This report performed zero head calls.',
        '',f'Fresh inference used {budget["allocated_gpu_seconds"]} allocated GPU-seconds and {coverage["native_model_norm_head_calls"]} model/norm/head calls; all540 vision prefills are counted.',
        '', 'These fixed panels test fresh realizations, person compositions and question transfer at N16. No ordinary joint-image comparator is evaluated on this confirmation panel. These results do not establish longer-length reliability, a new attention mechanism, superiority to ordinary joint modeling, or reasoning composition.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return dict(passed=True,completed=True,phase='report',protocol=p.PROTOCOL,source_sha256=frozen,inherited_source_sha256=plan['inherited_source_sha256'],
        plan_file=str(plan_path),plan_sha256=sha(plan_path),analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),
        report_file=str(out/'REPORT.md'),report_sha256=sha(out/'REPORT.md'),fresh_N16_screen=screens,longer_lengths_eligible=eligible,
        longer_lengths_released=False,resources=budget,coverage=coverage)


def main():
    import argparse,json,time
    p=driver();parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--runs',type=Path,nargs=2,required=True)
    args=parser.parse_args();p.native.require_slurm();need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),'CPU-only independent fresh report')
    out=p.OUT/f'report_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();frozen=p.snapshot(out)
    save(out/'request.json',dict(plan=str(args.plan),runs=[str(x) for x in args.runs],source_sha256=frozen))
    try:
        value=report(args,out,frozen);need(p.sources()==frozen and p.inherited_sources()==value['inherited_source_sha256'],'Fresh source changed during report')
        value['elapsed_seconds']=time.perf_counter()-started;need(value['elapsed_seconds']<=300,'Fixed CPU report300-second cap exceeded');save(out/'summary.json',value);print(json.dumps(dict(passed=True,directory=str(out))),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,elapsed_seconds=time.perf_counter()-started,partial_outputs_retained=True));raise


if __name__=='__main__':main()

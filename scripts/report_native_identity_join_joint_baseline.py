"""Independent frozen joint-image reference release/report on fixed fresh N16.

The single joint decoder has no trained adapter or broadcast processor. Profile
head fidelity covers selected native query rows; all main raw outcomes remain
in the denominator. This reference is not an adaptation-matched control.
"""
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from collections import Counter,defaultdict
import math,os,subprocess
from scripts import report_native_identity_join_conditioning_v2 as numerical
from scripts.stage_native_vision_v6_teacher import need,read,save,sha,object_sha
PANELS=('A','B','C');COUNTS={'A':108,'B':108,'C':54}


def driver():
    from scripts import evaluate_native_identity_join_joint_baseline as module
    return module


def bind(path,digest):need(sha(Path(path))==digest,'Changed bound joint-baseline artifact: '+str(path))


def score(ids,target):
    need(isinstance(ids,list) and 1<=len(ids)<=4 and all(type(i) is int and 0<=i<152064 for i in ids),'Invalid native joint token IDs')
    complete=ids[-1] in (151645,151643)
    need(not any(i in (151645,151643) for i in ids[:-1]) and (complete or len(ids)==4),'Native joint EOS chronology differs')
    return dict(exact=ids==target,first_token_correct=ids[0]==target[0],completed=complete,truncated=not complete,finish_reason='eos' if complete else 'length')


def criteria(rows):
    need(len(rows)==len({r['sid'] for r in rows})==270 and all(r['n_frames']==16 for r in rows) and Counter(r['panel'] for r in rows)==COUNTS,'All270 fixed fresh joint contexts required')
    panels={}
    for panel in PANELS:
        values=[r for r in rows if r['panel']==panel];families=defaultdict(list)
        for row in values:families[row['contrast_id']].append(row)
        need(len(families)==COUNTS[panel]//3 and all(len(g)==3 and {r['variant'] for r in g}=={0,1,2} for g in families.values()),'Complete three-variant joint baseline families required')
        whole=sum(r['exact'] for r in values);complete=sum(all(r['exact'] for r in g) for g in families.values());a,f=(49,16) if panel=='C' else (98,33)
        panels[panel]=dict(passed=whole>=a and complete>=f,whole_correct=whole,first_correct=sum(r['first_token_correct'] for r in values),contexts=len(values),
            complete_families=complete,families=len(families),thresholds=dict(whole_correct=a,complete_families=f),effective_minimum_whole_correct=max(a,3*f))
    return dict(panels=panels,qualifies_AB=panels['A']['passed'] and panels['B']['passed'],panel_C_passed=panels['C']['passed'],
        frozen_reference_only=True,does_not_change_parallel_length_gate=True)


def project(setup,times):
    need(len(times)==3 and all(math.isfinite(v) and v>0 for v in [setup]+list(times)),'Three positive joint-only profile observations required')
    bound=max(times);seconds=setup+1.25*270*bound+60.
    return dict(passed=seconds<=1200,setup_seconds=setup,maximum_four_token_seconds=bound,multiplier=1.25,contexts=270,reserve_seconds=60.,
        projected_seconds=seconds,cap_seconds=1200,joint_only=True)


def self_test(torch):
    need(score([15,151645],[15,151645])['exact'] and not score([15,151643],[15,151645])['exact'],'Joint full canonical EOS differs')
    try:score([151645,15],[15,151645])
    except ValueError:pass
    else:raise AssertionError('Interior EOS accepted')
    need(project(30.,[1.,2.,3.])['passed'] and not project(30.,[1.,2.,4.])['passed'],'Joint worst-case timing gate differs')
    logits=torch.tensor([[3.,1.,-2.]],dtype=torch.float16);metrics=numerical.replay_metric(torch,logits,logits.clone())
    need(isinstance(metrics,list) and len(metrics)==1 and metrics[0]['passed'] and metrics[0]['argmax_exact'] and metrics[0]['full_vocabulary_tv']==0.,'Selected native head-row fixture differs')
    rows=[dict(sid=f'{p}_{f}_{v}',panel=p,contrast_id=f'{p}_{f}',variant=v,n_frames=16,exact=True,first_token_correct=True) for p in PANELS for f in range(COUNTS[p]//3) for v in range(3)]
    for r in rows[:9]:r['exact']=False
    need(criteria(rows)['qualifies_AB'],'99 answers/33 triples must pass');rows[9]['exact']=False;need(not criteria(rows)['qualifies_AB'],'98 answers cannot satisfy33 triples')
    return dict(passed=True,groups=4,canonical_EOS=True,joint_only_timing=True,selected_head_row_schema=True,whole_family_boundary=True)


def source_archive(directory,plan):
    need(read(directory/'source_hashes.json')==plan['source_sha256'] and read(directory/'inherited_sources.json')==dict(source_sha256=plan['inherited_source_sha256']),'Joint source descriptors differ')
    for name,digest in plan['source_sha256'].items():bind(directory/'source'/name.replace('/','_'),digest)


def history(torch,result,bundle,layout):
    p=driver();base=bundle['inputs'];width=bundle['metadata']['prompt_width'];ids=result['generated_ids']
    need(len(result['native_inputs'])==len(result['native_positions'])==len(ids),'One observed joint history/position record per token required')
    for t,(seen,pos) in enumerate(zip(result['native_inputs'],result['native_positions'])):
        mask=torch.cat((base['attention_mask'],torch.ones((1,t),dtype=base['attention_mask'].dtype)),dim=1)
        tokens=base['input_ids'] if t==0 else torch.full((1,1),ids[t-1],dtype=base['input_ids'].dtype)
        need(torch.equal(seen['input_ids'],tokens) and torch.equal(seen['attention_mask'],mask)
             and seen['has_pixels'] is (t==0) and seen['past_length']==(0 if t==0 else width+t-1),'Actual ordinary joint history, image ownership or cache length differs')
        expected=layout['position_ids'] if t==0 else (layout['rope_deltas'].view(1,1,1)+width+t-1).expand(3,-1,-1)
        text=mask.long().cumsum(-1)-1
        need(pos.dtype==torch.int64 and pos.shape==(4,1,width if t==0 else 1) and torch.equal(pos[1:],expected),'Actual joint mRoPE differs')
        need(torch.equal(pos[0][mask.bool()],text[mask.bool()]) if t==0 else torch.equal(pos[0],text[:,-1:]),'Actual joint text positions differ')
    return dict(passed=True,queries=len(ids),ordinary_single_sequence=True,native_cache_lengths_and_positions=True,all_layer_KV_hashed=False)


def strata(rows):
    tables={}
    for keys in (('panel',),('panel','gold'),('panel','question'),('panel','orientation'),('panel','matches_training_orientation'),('panel','variant')):
        groups=defaultdict(list)
        for row in rows:groups[tuple(row[k] for k in keys)].append(row)
        tables['_'.join(keys)]=[dict(labels=dict(zip(keys,key)),metrics=dict(contexts=len(values),whole_correct=sum(r['exact'] for r in values),
            first_correct=sum(r['first_token_correct'] for r in values),completed=sum(r['completed'] for r in values),truncated=sum(r['truncated'] for r in values),
            noncanonical_answers=sum(r['noncanonical_answer'] for r in values),early_EOS_relative_to_target=sum(r['early_eos_relative_to_target'] for r in values)))
            for key,values in sorted(groups.items(),key=lambda item:repr(item[0]))]
    return tables


def parallel_comparison(path,rows,plan):
    if path is None:return dict(status='pending',reason='A complete passed parallel fresh report containing both fixed endpoints is required',paired_comparison_performed=False)
    path=Path(path).resolve();path=path/'summary.json' if path.is_dir() else path;summary=read(path)
    need(summary['passed'] is summary['completed'] is True and summary['phase']=='report' and summary['protocol']=='native_identity_join_factor_fresh_N16','Complete parallel fresh report required')
    bind(summary['analysis_file'],summary['analysis_sha256']);analysis=read(summary['analysis_file'])
    need(analysis['passed'] is analysis['completed'] is True and analysis['N16_only'] is analysis['both_endpoints_retained'] is analysis['all_raw_native_outputs_retained'] is True
         and analysis['stage_report']==plan['stage_report'] and set(analysis['runs'])=={'product','additive'},'Parallel comparison must use both endpoints and the identical fresh manifest')
    for name,digest in {**summary['source_sha256'],**summary['inherited_source_sha256']}.items():bind(REPO/name,digest)
    source_archive(path.parent,summary);bind(analysis['plan_file'],analysis['plan_sha256']);parallel_plan=read(analysis['plan_file'])
    need(parallel_plan['native_identity']==plan['native_identity'],'Parallel and joint frozen backbone identities differ')
    joint={r['sid']:r for r in rows};comparisons={}
    for arm in ('product','additive'):
        run=analysis['runs'][arm];bind(run['outcomes_file'],run['outcomes_sha256']);outcomes=read(run['outcomes_file'])
        need(outcomes==run['outcomes'] and len(outcomes)==270 and {r['sid'] for r in outcomes}==set(joint),'Parallel comparison row inventory differs')
        bind(Path(run['directory'])/'summary.json',run['summary_sha256']);bind(run['raw_manifest_file'],run['raw_manifest_sha256'])
        for trajectory in run['trajectories']:bind(trajectory['raw_file'],trajectory['raw_sha256'])
        for row in outcomes:
            other=joint[row['sid']];need(all(row[k]==other[k] for k in ('panel','n_frames','contrast_id','variant','gold','question','target_ids'))
                and all(row[k]==v for k,v in score(row['generated_ids'],row['target_ids']).items()),'Parallel exact answer/EOS ownership differs')
        by_sid={r['sid']:r for r in outcomes};panels={}
        for panel in PANELS:
            ids=[s for s,r in joint.items() if r['panel']==panel]
            cells=Counter('both_correct' if joint[s]['exact'] and by_sid[s]['exact'] else 'joint_only' if joint[s]['exact'] else 'parallel_only' if by_sid[s]['exact'] else 'both_wrong' for s in ids)
            panels[panel]=dict(contexts=len(ids),paired_transitions=dict(cells),parallel_minus_joint_correct=cells['parallel_only']-cells['joint_only'])
        comparisons[arm]=panels
    return dict(status='complete',paired_comparison_performed=True,summary_file=str(path),summary_sha256=sha(path),analysis_file=summary['analysis_file'],analysis_sha256=summary['analysis_sha256'],
        comparisons=comparisons,descriptive_only=True,adaptation_matched=False,parallel_length_gate_unchanged=True,parallel_longer_lengths_eligible=analysis['longer_lengths_eligible'])


def inputs(torch,plan):
    p=driver();stage_ref=plan['stage_report'];native_ref=plan['native_report']
    bind(stage_ref['file'],stage_ref['sha256']);bind(stage_ref['manifest_file'],stage_ref['manifest_sha256']);manifest=p.stage.verify_stage(Path(stage_ref['file']))
    need(manifest==read(stage_ref['manifest_file']) and native_ref['file']==str(p.NATIVE_REPORT) and native_ref['sha256']==p.NATIVE_REPORT_SHA
         and manifest['native_report']['file']==native_ref['file'] and manifest['native_report']['sha256']==native_ref['sha256'],'Identical published fresh/native dependency required')
    bind(native_ref['file'],native_ref['sha256']);proof=read(native_ref['file'])
    need(proof['passed'] is proof['completed'] is True and proof['phase']=='report','Passed complete native training proof required')
    for key in ('analysis','plan'):
        need(proof[key+'_file']==native_ref[key+'_file'] and proof[key+'_sha256']==native_ref[key+'_sha256'],'Native report descriptor differs');bind(native_ref[key+'_file'],native_ref[key+'_sha256'])
    native_plan=read(native_ref['plan_file']);native_analysis=read(native_ref['analysis_file'])
    need(native_analysis['passed'] is True and native_analysis['at_least_one_native_training_screen_passed'] is True,'Native training prerequisite not met')
    for key in ('native_identity','native_identity_sha256','native_model_file','native_model_sha256','native_module_identity','native_software_plan'):
        need(plan[key]==native_plan[key],'Joint frozen backbone/processor/head identity differs: '+key)
    need(object_sha(plan['native_identity'])==plan['native_identity_sha256'] and plan['no_model_or_head_forward'] is plan['frozen_comparator'] is plan['no_parallel_factor_input'] is True
         and plan['adaptation_matched'] is False,'Frozen input-only preparation flags differ')
    for path,digest in plan['native_identity']['native_api']['source_sha256'].items():bind(path,digest)
    bind(plan['native_model_file'],plan['native_model_sha256']);bind(plan['native_software_plan']['file'],plan['native_software_plan']['sha256'])
    need(plan['precision']==read(plan['native_software_plan']['file'])['precision'],'Native precision defaults differ')
    rows=read(plan['rows_file']);expected=[r for panel in PANELS for r in manifest['splits'][f'fresh_{panel}_N16']['samples']]
    need(rows==expected and len(rows)==len({r['sid'] for r in rows})==270 and Counter(r['panel'] for r in rows)==COUNTS,'Exact complete published fresh panel order required')
    prepared=read(plan['prepared_file']);bind(plan['layout_file'],plan['layout_sha256']);layouts=torch.load(plan['layout_file'],map_location='cpu',weights_only=True)
    need(set(layouts)==set(prepared)=={r['sid'] for r in rows},'All270 joint layout/preparation records required')
    for row in rows:
        item=prepared[row['sid']];meta=item['metadata'];layout=layouts[row['sid']]
        need(row['n_frames']==16 and row['target_ids']==manifest['tokenizer']['targets'][row['gold']]['target_ids']
             and item['layout']==layout['metadata'] and item['layout']['every_unpadded_row_exact'] is True
             and p.tensor_info(layout['position_ids'])==item['layout']['position_ids'] and layout['rope_deltas'].tolist()==item['layout']['rope_deltas'],'Joint target or CPU position inventory differs')
        need(meta['sid']==row['sid'] and meta['question']==row['question'] and meta['question_sha256']==object_sha(row['question'])
             and meta['global_prompt']==row['question'] and meta['local_prompt'] is None and meta['row_kinds']==['joint']
             and meta['row_count']==1 and meta['global_row']==0 and meta['local_elements']==0 and meta['n_frames']==16
             and meta['row_prompt_tokens']==[meta['prompt_width']] and meta['original_prompt_width']==meta['prompt_width'] and meta['prefix_ids']==[]
             and meta['resize']==392 and meta['processor_parity_checked'] is meta['ordinary_images_first'] is meta['no_prompt_wrapper'] is True
             and meta['image_paths']==[i['path'] for i in row['image_files']] and meta['image_sha256']==[i['sha256'] for i in row['image_files']],
             'Ordinary images-first unchanged question/image ordering differs')
    cases=[]
    for panel in PANELS:
        candidates=[(i,r) for i,r in enumerate(rows) if r['panel']==panel];width=max(prepared[r['sid']]['metadata']['prompt_width'] for _,r in candidates)
        i,row=next((i,r) for i,r in candidates if prepared[r['sid']]['metadata']['prompt_width']==width)
        cases.append(dict(sid=row['sid'],panel=panel,n_frames=16,prompt_width=width,manifest_index=i))
    need(plan['profile_cases']==cases and plan['maximum_joint_prompt_width']==max(x['metadata']['prompt_width'] for x in prepared.values()),'Widest actual joint input/first manifest tie profile selection differs')
    return rows,prepared,layouts


def head_replays(torch,result,raw,width,modules):
    records=result['profile_head'];need(isinstance(records,list) and len(records)==len(result['generated_ids']),'One actual selected native head record per profile token required')
    norm,head=modules;metrics=[]
    for t,record in enumerate(records):
        need(set(record)=={'norm_query_input','normalized_query','head_input','head_logits'},'Profile selected-row tensor inventory differs')
        tail,normal,head_input,actual=(record[k] for k in ('norm_query_input','normalized_query','head_input','head_logits'))
        need(tail.shape==normal.shape==head_input.shape==(1,1,3584) and actual.shape==(1,1,152064)
             and all(v.dtype==torch.float16 and not v.requires_grad and bool(torch.isfinite(v).all()) for v in record.values())
             and torch.equal(normal,head_input) and torch.equal(actual[0,0],raw[t].half()),'Captured last normalized row, actual head input/output or original raw logits differ')
        with torch.inference_mode():replayed=head(norm(tail))
        row_metrics=numerical.replay_metric(torch,actual[:,0],replayed[:,0]);need(len(row_metrics)==1,'Actual single-row CPU replay scope differs')
        metrics.extend(row_metrics)
    return metrics


def audit_run(torch,out,directory,plan_path,plan,rows,prepared,layouts,tokenizer,phase,modules=None):
    p=driver();directory=Path(directory).resolve();summary=read(directory/'summary.json');config=read(directory/'config.json')
    need(phase in ('profile','main') and config['phase']==summary['phase']==phase and directory==p.OUT/f'{phase}_{config["slurm_job_id"]}'
         and summary['passed'] is summary['completed'] is True and all(summary[k]==v for k,v in config.items()),'Completed joint run/config ownership differs')
    source_archive(directory,plan)
    for key in ('protocol','policy','source_sha256','inherited_source_sha256','native_report','stage_report','native_identity','native_identity_sha256'):
        need(config[key]==plan[key],'Joint configuration differs: '+key)
    need(config['plan_file']==str(plan_path) and config['plan_sha256']==sha(plan_path) and config['run_id']==directory.name
         and Path(config['data_directory'])==p.DATA/directory.name and summary['no_fitting'] is summary['no_accuracy_selection'] is summary['all_raw_retained_before_scoring'] is True
         and summary['extra_gpu_head_calls']==0 and config['frozen_comparator'] is config['no_adapter'] is config['equal_calls_do_not_imply_equal_compute'] is True
         and config['adaptation_matched'] is False,'Ordinary frozen reference/no-fit/raw retention contract differs')
    hardware=config['hardware'];quant=hardware['quantization']
    need(hardware['gpu']=='NVIDIA B200' and hardware['precision']==plan['precision'] and quant['load_in_4bit'] is quant['bnb_4bit_use_double_quant'] is True
         and quant['bnb_4bit_quant_type']=='nf4' and str(quant['bnb_4bit_compute_dtype']).removeprefix('torch.')=='bfloat16'
         and config['native_weight_identity']==dict(norm=plan['native_identity']['norm_weight'],head=plan['native_identity']['head_weight']),'Actual joint precision/backbone/native weights differ')
    if phase=='profile':need(config['main_release'] is None and modules is not None,'Profile must precede main release and replay selected native heads')
    else:need(modules is None and config['main_release']==p.verify_release(Path(config['main_release']['file']),plan_path,plan),'Joint main must bind the passed measured release')
    cases=plan['profile_cases'] if phase=='profile' else [dict(sid=r['sid'],panel=r['panel'],n_frames=16) for r in rows]
    need(config['cases']==cases and summary['trajectories']==len(cases),'Exact joint profile/main cohort differs')
    bind(summary['raw_manifest_file'],summary['raw_manifest_sha256']);manifest=read(summary['raw_manifest_file'])
    need(Path(summary['raw_manifest_file'])==directory/'raw_manifest.json' and manifest['completed'] is manifest['all_raw_retained_before_scoring'] is True
         and manifest['n']==len(cases)==len(manifest['rows']),'All native raw trajectories must precede scoring')
    lookup={r['sid']:r for r in rows};outcomes=[];details=[];counts=Counter();auditfiles=[];head_calls=0;max_tv=0.
    for j,(case,record) in enumerate(zip(cases,manifest['rows'])):
        sid=case['sid'];sample=lookup[sid];item=prepared[sid]
        need(record['index']==j and record['case']==case and record['prepared_file']==item['file'] and record['prepared_sha256']==item['sha256'],'Joint trajectory input/order differs')
        bind(record['raw_file'],record['raw_sha256']);need(Path(record['raw_file'])==Path(config['data_directory'])/f'trajectory_{j:03d}.pt','Joint raw trajectory path differs')
        need(read(directory/f'progress_{j+1:03d}.json')==dict(processed=j+1,case=case,raw_file=record['raw_file'],raw_sha256=record['raw_sha256'],accuracy_not_scored=True),'Joint raw publication record differs')
        packet=torch.load(record['raw_file'],map_location='cpu',weights_only=True);need(set(packet)=={'schema_version','case','result'} and packet['schema_version']==1 and packet['case']==case,'Joint raw packet ownership differs')
        result=packet['result'];raw=result['raw_logits'];ids=result['generated_ids'];t=len(ids);scored=score(ids,sample['target_ids'])
        bind(item['file'],item['sha256']);bundle=torch.load(item['file'],map_location='cpu',weights_only=True);p.validate_joint(torch,bundle)
        need(bundle['metadata']==item['metadata'],'Joint source bundle metadata differs');meta=result['metadata'];width=bundle['metadata']['prompt_width']
        need(all(meta[k]==v for k,v in bundle['metadata'].items()) and meta['layout']==item['layout'] and meta['native_identity_sha256']==plan['native_identity_sha256']
             and meta['scene_input_identity']==object_sha(dict(metadata=bundle['metadata'],native_identity_sha256=plan['native_identity_sha256']))
             and meta['generation_position_ids']==[p.tensor_info(v) for v in result['native_positions']],'Joint scene/model/native-position identity differs')
        generation=meta['generation']
        need(generation==dict(max_new_tokens=4,do_sample=False,num_beams=1,repetition_penalty=1.,use_cache=True,native_eos_token_ids=[151645,151643],
            target_eos_token_id=151645,pad_token_id=tokenizer.pad_token_id,output_logits=False,output_scores=False,vocabulary_mask=False,other_logits_processors=False,
            logits_to_keep=1,ordinary_joint=True,broadcast=False),'Natural unmasked ordinary joint generation policy differs')
        need(raw.dtype==torch.float32 and raw.shape==(t,152064) and not raw.requires_grad and bool(torch.isfinite(raw).all())
             and torch.equal(raw,raw.half().float()) and raw.argmax(-1).tolist()==ids,'Unmasked native joint raw argmax differs')
        need(all(result[k]==scored[k] for k in ('completed','truncated','finish_reason')) and result['text']==tokenizer.decode(ids,skip_special_tokens=True)
             and result['raw_text']==tokenizer.decode(ids,skip_special_tokens=False),'Joint natural stopping or decoded text differs')
        expected=dict(model=t,visual=1,language=t,norm=t,head=t,broadcast=0,fusion=0,conditioning=0,probe_head=0)
        need(result['counters']==expected and record['generated_tokens']==t and len(result['shapes'])==len(result['logit_records'])==t
             and result['parameter_versions_unchanged'] is result['hooks_removed'] is result['rope_restored'] is True,'Joint actual calls or frozen native ownership differ')
        for k,lr in enumerate(result['logit_records']):
            need(lr['step']==k and lr['top1_token_id']==ids[k] and lr['top1_logit']==float(raw[k,ids[k]]) and lr['native_dtype']=='torch.float16'
                 and lr['vocabulary_size']==152064 and math.isfinite(lr['log_normalizer']),'Native full-vocabulary joint recorder differs')
        for k,shape in enumerate(result['shapes']):
            need(shape==dict(norm_input_shape=[1,width if k==0 else 1,3584],norm_output_shape=[1,width if k==0 else 1,3584],
                head_input_shape=[1,1,3584],head_output_shape=[1,1,152064],norm_dtype='torch.float16',head_input_dtype='torch.float16',head_output_dtype='torch.float16'),
                'Observed full native normalization or selected head shape differs')
        native_history=history(torch,result,bundle,layouts[sid]);replay=[]
        if phase=='profile':replay=head_replays(torch,result,raw,width,modules);head_calls+=len(replay);max_tv=max(max_tv,max(m['full_vocabulary_tv'] for m in replay))
        else:need(result['profile_head'] is None,'Joint main must not add selected-row head captures/replays')
        audit=dict(passed=all(m['passed'] for m in replay),case=case,history=native_history,actual_shapes=result['shapes'],native_head_replays=replay,
            cpu_head_replay_scope='selected last-query norm/head only; full prefill norm shapes observed' if phase=='profile' else 'none; previously audited profile proof',
            all_raw_native_argmax_exact=True,parameter_versions_unchanged=True,hooks_removed=True,rope_restored=True)
        file=out/f'{phase}_trajectory_{j:03d}_audit.json';save(file,audit);auditfiles.append(dict(file=str(file),sha256=sha(file)))
        need(audit['passed'],'Registered selected-row native head gate failed; offending audit retained')
        timing=[record['preprocessing_seconds'],record['work_seconds'],record['four_token_seconds']]
        need(all(math.isfinite(v) and v>0 for v in timing) and timing[2]==timing[0]+timing[1]*4/t,'Measured full joint trajectory timing differs')
        centered=raw[0].double()-raw[0].double().max();nll=float(torch.logsumexp(centered,0)-centered[sample['target_ids'][0]])
        fields=('sid','contrast_id','pair_id','variant','n_frames','question','gold','target_ids','panel','orientation','matches_training_orientation','trio','room_pair','family_instance_sha256')
        outcomes.append(dict(**{k:sample[k] for k in fields},generated_ids=ids,text=result['text'],raw_text=result['raw_text'],first_query_nll=nll,
            noncanonical_answer=ids not in [v['target_ids'] for v in rows],early_eos_relative_to_target=scored['completed'] and len(ids)<len(sample['target_ids']),**scored))
        details.append(dict(index=j,case=case,raw_file=record['raw_file'],raw_sha256=record['raw_sha256'],generated_tokens=t,four_token_seconds=timing[2],
            observed_query_prefixes=[ids[:k] for k in range(t)]));counts.update(expected);del packet,result,raw,bundle,replay
    bind(summary['endpoint_file'],summary['endpoint_sha256']);need(Path(summary['endpoint_file'])==directory/'endpoint.json','Joint endpoint path differs');endpoint=read(summary['endpoint_file'])
    need(endpoint['passed'] is endpoint['native_parameter_versions_unchanged'] is endpoint['no_optimizer_or_training'] is endpoint['no_adapter'] is True
         and endpoint['native_weights_before']==endpoint['native_weights_after']==config['native_weight_identity']
         and endpoint['counters']==summary['counters']==dict(counts),'Final actual joint native weights/counters differ')
    need(counts['visual']==len(cases) and counts['model']<=(12 if phase=='profile' else 1080) and math.isfinite(summary['setup_seconds']) and summary['setup_seconds']>0
         and math.isfinite(summary['elapsed_seconds']) and 0<summary['elapsed_seconds']<=(120 if phase=='profile' else 1200),'Joint phase resource inventory differs')
    file=out/f'{phase}_outcomes.json';save(file,outcomes)
    return dict(passed=True,phase=phase,directory=str(directory),summary_sha256=sha(directory/'summary.json'),slurm_job_id=config['slurm_job_id'],
        raw_manifest_file=summary['raw_manifest_file'],raw_manifest_sha256=summary['raw_manifest_sha256'],endpoint_file=summary['endpoint_file'],endpoint_sha256=summary['endpoint_sha256'],
        outcomes_file=str(file),outcomes_sha256=sha(file),outcomes=outcomes,trajectory_audits=auditfiles,trajectories=details,counters=dict(counts),
        setup_seconds=summary['setup_seconds'],four_token_seconds=[r['four_token_seconds'] for r in manifest['rows']],native_head_rows=counts['head'],
        cpu_head_calls=head_calls,cpu_head_rows=head_calls,maximum_cpu_native_head_tv=max_tv if phase=='profile' else None,
        all_profile_native_head_argmax_exact=True if phase=='profile' else None,profile_outcomes_descriptive_only=phase=='profile',
        frozen_comparator=True,adaptation_matched=False,native_identity_sha256=plan['native_identity_sha256'])


def bound_outputs(value):
    bind(value['outcomes_file'],value['outcomes_sha256']);need(read(value['outcomes_file'])==value['outcomes'],'Prior independent outcomes changed')
    bind(Path(value['directory'])/'summary.json',value['summary_sha256']);bind(value['raw_manifest_file'],value['raw_manifest_sha256']);bind(value['endpoint_file'],value['endpoint_sha256'])
    for item in value['trajectory_audits']:bind(item['file'],item['sha256']);need(read(item['file'])['passed'] is True,'Prior profile native audit failed')
    for item in value['trajectories']:bind(item['raw_file'],item['raw_sha256'])


def resources(out,profile,run=None):
    import re
    from datetime import datetime
    p=driver();command=['sacct','-X','-u',os.environ['USER'],'--starttime=2026-09-01','--parsable2','--noheader',
        '--format=JobIDRaw,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,Start,End']
    raw=subprocess.run(command,capture_output=True,text=True,check=True).stdout;file=out/'all_user_sacct.psv';file.write_text(raw);allocations=[]
    for line in raw.splitlines():
        fields=line.split('|');need(len(fields)==9,'Unexpected joint accounting fields');job,name,partition,state,exit_code,seconds,tres,start,end=fields
        if name not in p.JOBS.values():continue
        generic=re.findall(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',tres);typed=re.findall(r'(?:^|,)gres/gpu:[^=,]+=(\d+)(?=,|$)',tres)
        fallback=re.findall(r'(?:^|,)gpu(?::[^=,]+)?=(\d+)(?=,|$)',tres);gpus=int(generic[0]) if generic else sum(map(int,typed)) if typed else sum(map(int,fallback))
        allocations.append(dict(job_id=job,name=name,partition=partition,state=state,exit_code=exit_code,seconds=int(seconds),gpus=gpus,gpu_seconds=int(seconds)*gpus,alloc_tres=tres,start=start,end=end))
    expected={p.JOBS['profile']:profile}
    if run is not None:expected[p.JOBS['main']]=run
    need(len(allocations)==len(expected) and {r['name'] for r in allocations}==set(expected),'Exactly one attempt per joint phase required; failed and zero allocations count')
    events=[]
    for row in allocations:
        cap=120 if row['name']==p.JOBS['profile'] else 1200
        need(row['job_id']==expected[row['name']]['slurm_job_id'] and row['partition']=='gpu' and row['state']=='COMPLETED'
             and row['exit_code']=='0:0' and row['gpus']==1 and 0<=row['seconds']<=cap,'Joint allocation identity/fixed cap differs')
        start,end=map(datetime.fromisoformat,(row['start'],row['end']));need(end>=start,'Joint GPU chronology differs')
        if end>start:events.extend(((start,1),(end,-1)))
    active=peak=0
    for _,change in sorted(events):active+=change;need(active>=0,'Invalid GPU allocation interval');peak=max(peak,active)
    used=sum(v['gpu_seconds'] for v in allocations);reserve=1200 if run is None else 0
    need(active==0 and peak<=1 and used+reserve<=1320,'Joint single-GPU/1320-second campaign cap exceeded')
    return dict(passed=True,jobs=allocations,allocated_gpu_seconds=used,reserved_main_gpu_seconds=reserve,total_with_reservation=used+reserve,
        campaign_seconds=1320,maximum_concurrent_gpus=peak,failed_and_zero_allocations_retained=True,file=str(file),sha256=sha(file),command=command)


def release(args,out,frozen):
    import torch
    from transformers import AutoProcessor
    from scripts import diagnose_native_identity_join_readout as oracle
    p=driver();torch.set_num_threads(4);plan_path=args.plan.resolve();plan=p.verify_plan(plan_path,ancestors=True)
    rows,prepared,layouts=inputs(torch,plan);processor=AutoProcessor.from_pretrained(str(p.MODEL),trust_remote_code=True,use_fast=False,local_files_only=True)
    need(oracle.native_module_identity(torch,plan['native_identity'])==plan['native_module_identity'],'Actual selected native norm/head implementation differs')
    packet=torch.load(plan['native_model_file'],map_location='cpu',weights_only=True);norm,head=oracle.native_modules(torch,packet,plan['native_identity'],'cpu')
    before=dict(norm=p.tensor_info(norm.weight),head=p.tensor_info(head.weight));profile=audit_run(torch,out,args.profile,plan_path,plan,rows,prepared,layouts,processor.tokenizer,'profile',(norm,head))
    need(before==dict(norm=p.tensor_info(norm.weight),head=p.tensor_info(head.weight)) and all(not v.requires_grad and v.grad is None for module in (norm,head) for v in module.parameters()),'CPU native replay changed frozen tensors')
    need(3<=profile['cpu_head_calls']==profile['cpu_head_rows']==profile['counters']['head']<=12 and profile['counters']['visual']==3,'Exactly one CPU replay per actual profile head row required')
    projection=project(profile['setup_seconds'],profile['four_token_seconds']);need(projection==p.project(profile['setup_seconds'],profile['four_token_seconds']),'Independent joint-only projection differs')
    budget=resources(out,profile);audit=dict(passed=True,profile=profile,projection=projection,resources=budget,cpu_head_calls=profile['cpu_head_calls'],cpu_head_rows=profile['cpu_head_rows'],
        maximum_cpu_native_head_tv=profile['maximum_cpu_native_head_tv'],all_profile_native_head_argmax_exact=True,profile_natural_outcomes_descriptive_only=True,
        native_cpu_head_replay_scope='selected last-query norm/head at every profile prefix; full prefill norm shapes observed, not replayed')
    save(out/'profile_audit.json',audit)
    need(projection['passed'],'Joint-only timing forecast exceeds1200 seconds; original profile outcomes/audits retained, no main release')
    value=dict(passed=True,protocol=p.PROTOCOL,policy=p.POLICY,source_sha256=frozen,inherited_source_sha256=plan['inherited_source_sha256'],
        plan_file=str(plan_path),plan_sha256=sha(plan_path),profile=profile,projection=projection,reserved_main_gpu_seconds=1200,
        profile_audit_file=str(out/'profile_audit.json'),profile_audit_sha256=sha(out/'profile_audit.json'),resources=budget)
    save(out/'release.json',value)
    return dict(passed=True,completed=True,phase='release',protocol=p.PROTOCOL,source_sha256=frozen,inherited_source_sha256=plan['inherited_source_sha256'],
        plan_file=str(plan_path),plan_sha256=sha(plan_path),release_file=str(out/'release.json'),release_sha256=sha(out/'release.json'),
        profile_audit_file=value['profile_audit_file'],profile_audit_sha256=value['profile_audit_sha256'],projection=projection,resources=budget,
        cpu_head_calls=profile['cpu_head_calls'],cpu_head_rows=profile['cpu_head_rows'],maximum_cpu_native_head_tv=profile['maximum_cpu_native_head_tv'])


def report(args,out,frozen):
    import torch
    from transformers import AutoProcessor
    p=driver();torch.set_num_threads(4);plan_path=args.plan.resolve();plan=p.verify_plan(plan_path,ancestors=True)
    release_ref=p.verify_release(args.main_release,plan_path,plan);released=read(release_ref['file']);source_archive(Path(release_ref['file']).parent,plan)
    bind(released['profile_audit_file'],released['profile_audit_sha256']);profile_audit=read(released['profile_audit_file']);profile=released['profile'];bound_outputs(profile)
    need(released['passed'] is profile_audit['passed'] is profile_audit['all_profile_native_head_argmax_exact'] is True and profile_audit['profile']==profile
         and profile_audit['projection']==released['projection']==project(profile['setup_seconds'],profile['four_token_seconds'])
         and profile['cpu_head_calls']==profile['cpu_head_rows']==profile_audit['cpu_head_calls']==profile_audit['cpu_head_rows']<=12
         and profile['maximum_cpu_native_head_tv']==profile_audit['maximum_cpu_native_head_tv']<=.02,'Passed complete selected-row profile fidelity/projection proof required')
    rows,prepared,layouts=inputs(torch,plan);processor=AutoProcessor.from_pretrained(str(p.MODEL),trust_remote_code=True,use_fast=False,local_files_only=True)
    run=audit_run(torch,out,args.run,plan_path,plan,rows,prepared,layouts,processor.tokenizer,'main')
    need(read(Path(run['directory'])/'config.json')['main_release']==release_ref and run['cpu_head_calls']==run['cpu_head_rows']==0,'Main binds this exact release; no new CPU heads permitted')
    budget=resources(out,profile,run);screen=criteria(run['outcomes']);comparison=parallel_comparison(args.parallel_report,run['outcomes'],plan)
    coverage=dict(main_contexts=270,main_families=90,panel_contexts=COUNTS,panel_families={'A':36,'B':36,'C':18},
        main_native_calls=run['counters']['model'],main_native_head_rows=run['native_head_rows'],main_vision_calls=270,main_cpu_head_calls=0,
        profile_native_calls=profile['counters']['model'],profile_vision_calls=3,profile_cpu_head_calls=profile['cpu_head_calls'],profile_cpu_head_rows=profile['cpu_head_rows'],
        all_gpu_native_calls=run['counters']['model']+profile['counters']['model'],all_gpu_native_head_rows=run['native_head_rows']+profile['native_head_rows'],extra_gpu_head_calls=0)
    need(coverage['main_native_calls']<=1080 and coverage['all_gpu_native_calls']<=1092 and coverage['all_gpu_native_head_rows']==coverage['all_gpu_native_calls'],'Registered ordinary joint call inventory exceeded')
    analysis=dict(passed=True,completed=True,phase='report',protocol=p.PROTOCOL,policy=p.POLICY,source_sha256=frozen,inherited_source_sha256=plan['inherited_source_sha256'],
        plan_file=str(plan_path),plan_sha256=sha(plan_path),native_report=plan['native_report'],stage_report=plan['stage_report'],main_release=release_ref,
        profile=profile,run=run,frozen_joint_screen=screen,strata=strata(run['outcomes']),parallel_comparison=comparison,resources=budget,coverage=coverage,
        all_raw_native_outputs_retained=True,N16_only=True,no_fitting_or_checkpoint_selection=True,frozen_comparator=True,adaptation_matched=False,
        equal_compute_claim=False,parallel_length_gate_changed=False,longer_lengths_released=False,
        native_cpu_head_replay_scope='previous profile selected-query proof only; no main CPU replay',
        maximum_cpu_native_head_tv=profile['maximum_cpu_native_head_tv'],all_profile_native_head_argmax_exact=True)
    save(out/'analysis.json',analysis)
    text=['# Frozen ordinary joint-image reference on fresh N16','',
        'This passed report certifies the complete independent software/output audit. The frozen reference uses all16 images in one ordinary native sequence and the unchanged name question. It has no trained adapter and is not adaptation matched to the product/additive models.','',
        '| Panel | Exact whole name + EOS | Complete triples | Descriptive criterion |','|---|---:|---:|---|']
    for panel in PANELS:
        s=screen['panels'][panel];text.append(f'| {panel} | {s["whole_correct"]}/{s["contexts"]} | {s["complete_families"]}/{s["families"]} | {"PASS" if s["passed"] else "FAIL"} |')
    text.extend(['',f'All270 outcomes and90 three-variant families are retained. GPU allocation: {budget["allocated_gpu_seconds"]} seconds; main native calls/head rows: {run["counters"]["model"]}; vision prefills:270.',
        f'Native head fidelity reuses {profile["cpu_head_calls"]} profile selected-query CPU replays, maximum TV {profile["maximum_cpu_native_head_tv"]:.9g}, all argmaxes exact. Full prefill norm shapes were observed; only selected head queries were replayed. No new CPU/GPU head calls were added for main scoring.','',
        f'Paired parallel comparison: {comparison["status"]}. Complete comparison, when available, is descriptive and retains both fixed endpoints. The frozen reference cannot override the parallel N16-first length gate. N32/N64, equal-compute, adaptation-matched superiority and reasoning efficacy are not established.'])
    (out/'REPORT.md').write_text('\n'.join(text)+'\n')
    return dict(passed=True,completed=True,phase='report',protocol=p.PROTOCOL,source_sha256=frozen,inherited_source_sha256=plan['inherited_source_sha256'],
        plan_file=str(plan_path),plan_sha256=sha(plan_path),analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),
        report_file=str(out/'REPORT.md'),report_sha256=sha(out/'REPORT.md'),frozen_joint_screen=screen,parallel_comparison_status=comparison['status'],
        resources=budget,coverage=coverage,maximum_cpu_native_head_tv=profile['maximum_cpu_native_head_tv'])


def main():
    import argparse,time,json
    p=driver();parser=argparse.ArgumentParser(description=__doc__);action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--release',action='store_true');action.add_argument('--report',action='store_true');parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--profile',type=Path);parser.add_argument('--run',type=Path);parser.add_argument('--main-release',type=Path);parser.add_argument('--parallel-report',type=Path)
    args=parser.parse_args();p.native.require_slurm();need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),'CPU-only independent joint report required')
    need((args.profile is not None and args.run is args.main_release is args.parallel_report is None) if args.release else
         (args.profile is None and args.run is not None and args.main_release is not None),'Exact release/profile or main/report inputs required')
    phase='release' if args.release else 'report';out=p.OUT/f'{phase}_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();frozen=p.snapshot(out)
    save(out/'request.json',dict(phase=phase,plan=str(args.plan),profile=None if args.profile is None else str(args.profile),run=None if args.run is None else str(args.run),
        main_release=None if args.main_release is None else str(args.main_release),parallel_report=None if args.parallel_report is None else str(args.parallel_report),source_sha256=frozen))
    try:
        value=release(args,out,frozen) if args.release else report(args,out,frozen);elapsed=time.perf_counter()-started
        need(elapsed<=300 and p.sources()==frozen and p.inherited_sources()==value['inherited_source_sha256'],'Joint report cap/source identity differs')
        value['elapsed_seconds']=elapsed;save(out/'summary.json',value);print(json.dumps(dict(passed=True,directory=str(out))),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,elapsed_seconds=time.perf_counter()-started,partial_outputs_retained=True));raise


if __name__=='__main__':main()

"""Independent CPU profile release and natural factor trainability report.

Full native head replay is restricted to profile trajectories. Main outcomes
are rescored from every saved native logit, with every factor/cast independently
reconstructed. Neither action fits, selects, masks, or regenerates an answer.
"""
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from collections import Counter,defaultdict
import math
import os
import subprocess
from scripts import report_native_identity_join_factor_binding as factor
from scripts.stage_native_vision_v6_teacher import need,read,save,sha,object_sha
ARMS=('product','additive')


def driver():
    from scripts import diagnose_native_identity_join_factor_native as module
    return module


def bind(path,digest):need(sha(Path(path))==digest,'Bound native artifact changed: '+str(path))


def score(ids,target):
    need(isinstance(ids,list) and 1<=len(ids)<=4 and all(type(i) is int and 0<=i<152064 for i in ids),'Invalid raw generated IDs')
    eos=(151645,151643);completed=ids[-1] in eos
    need(not any(i in eos for i in ids[:-1]) and (completed or len(ids)==4),'Native EOS chronology differs')
    return dict(exact=ids==target,first_token_correct=ids[0]==target[0],completed=completed,truncated=not completed,
        finish_reason='eos' if completed else 'length')


def self_test(torch):
    need(score([15,151645],[15,151645])['exact'] and not score([15,151643],[15,151645])['exact']
         and not score([15,16,17,18],[15,151645])['exact'],'Exact IDs including native EOS fixture failed')
    try:score([151645,15],[15,151645])
    except Exception as exc:need('chronology' in str(exc),'Unexpected EOS fixture failure')
    else:raise AssertionError('An interior EOS must fail')
    p=driver();slow=p.projection([20.,21.],[4.]*8);fast=p.projection([20.,21.],[1.]*8)
    need(not slow['passed'] and fast['passed'] and fast['pooled_scene_seconds']==p.POLICY['historical_four_token_floor'],
         'Pooled maximum/historical floor/cap fixture failed')
    rows=[dict(sid=f'{f}_{i}_{n}',contrast_id=f,variant=i,n_frames=n,exact=True,first_token_correct=True)
          for f in range(18) for i in range(3) for n in (8,16)]
    need(p.criterion(rows)['passed'],'Complete native family fixture failed')
    for i in (0,6,12):rows[i]['exact']=False
    need(not p.criterion(rows)['passed'],'Whole-family requirement must not collapse to marginal accuracy')
    logits=torch.tensor([[3.,1.,-2.],[-2.,3.,1.]],dtype=torch.float16)
    head_rows=factor.replay_metric(torch,logits,logits.clone())
    need(isinstance(head_rows,list) and len(head_rows)==2 and all(r['passed'] and r['argmax_exact'] and r['full_vocabulary_tv']==0. for r in head_rows),
         'Actual immutable head helper must return separate native-row records')
    wrong=factor.replay_metric(torch,logits,logits.flip(-1));need(not all(r['passed'] for r in wrong),'Changed native head rows must fail')
    return dict(passed=True,groups=4,exact_ids_and_EOS=True,pooled_timing_floor_and_cap=True,whole_family_criterion=True,actual_native_head_helper_schema=True)


def inputs(torch,plan):
    p=driver();rows=read(plan['rows_file']);prepared=read(plan['prepared_file'])
    bind(plan['uniform_plan']['file'],plan['uniform_plan']['sha256']);parent=read(plan['uniform_plan']['file'])
    need(rows==[r['sample'] for r in read(parent['rows_file'])['train']] and len(rows)==108
         and set(prepared)=={r['sid'] for r in rows},'Original training image/answer ownership differs')
    old=read(parent['prepared_file']);bind(plan['layout_file'],plan['layout_sha256']);layouts=torch.load(plan['layout_file'],map_location='cpu',weights_only=True)
    need(set(layouts)==set(prepared),'Original position coverage differs')
    for sample in rows:
        sid=sample['sid'];item=prepared[sid]
        need({k:v for k,v in item.items() if k!='layout'}==old[sid] and item['layout']==layouts[sid]['metadata'],'Prepared/layout ancestry changed')
        need(item['layout']['every_unpadded_row_exact'] is True and p.tensor_info(layouts[sid]['position_ids'])==item['layout']['position_ids']
             and layouts[sid]['rope_deltas'].tolist()==item['layout']['rope_deltas'],'Bound position tensors differ')
    first=next(r for r in rows if r['gold']=='Sandra');pair=[r for r in rows if r['pair_id']==first['pair_id']]
    need([r['sid'] for r in pair]==['ij_59947641555d9e5bf02cbb90','ij_517f73318bb2929face59f8d']
         and plan['profile_cases']==[dict(sid=r['sid'],n_frames=r['n_frames'],state=s) for r in pair for s in ('zero','fitted')],
         'Fixed Sandra software cohort differs')
    bind(plan['stats_file'],plan['stats_sha256']);stats=torch.load(plan['stats_file'],map_location='cpu',weights_only=True)
    need({k:p.tensor_info(stats[k]) for k in ('global_mean','scale')}==plan['conditioning_identity'],'Actual global constants differ')
    weights={}
    for arm in ARMS:
        weights[arm]={}
        for state in ('fitted','zero'):
            d=plan['checkpoints'][arm][state];bind(d['file'],d['sha256']);packet=torch.load(d['file'],map_location='cpu',weights_only=True);w=packet['branch']
            factor.check_weights(torch,w,w);need({k:p.tensor_info(v) for k,v in w.items()}==d['state'],'Exact endpoint tensor table differs')
            weights[arm][state]=w
        a,b=weights[arm]['fitted'],weights[arm]['zero'];need(bool((b['up.weight']==0).all()) and all(torch.equal(a[k],b[k]) for k in a if k!='up.weight'),
            'Zero-U diagnostic changed other fitted parameters')
    return rows,prepared,layouts,stats,weights


def capture(torch,cap,w,mean,scale,arm,n,zero):
    valid=torch.ones((n,1),dtype=torch.bool);h=cap['local_states'];g=cap['global_states']
    need(h.shape==(n,1,3584) and g.shape==(1,3584) and h.dtype==g.dtype==torch.float16,'Actual current local/global state shape differs')
    reference=factor.functional_core(torch,h,g,valid,mean.view(1,-1),scale,w,arm,False)
    native_keys={'local_states','global_states','native_query_hidden','native_delta','fused_global','fused_query_hidden'}
    need(set(cap)==set(reference)|native_keys,'Full native factor/conditioning capture inventory differs')
    for key,value in cap.items():need(isinstance(value,torch.Tensor) and not value.requires_grad and bool(torch.isfinite(value).all()),'Invalid captured tensor: '+key)
    indices=reference['factor_permutation_indices'];need(cap['factor_permutation_indices'].dtype==torch.int64 and torch.equal(cap['factor_permutation_indices'],indices),'Factor pairing must remain identity')
    metrics={k:factor.tensor_error(torch,cap[k],v) for k,v in reference.items() if k!='factor_permutation_indices'}
    need(torch.equal(cap['factor_b_used'],cap['factor_b']) and torch.equal(cap['gates'],valid.float()*.5) and bool((cap['scores']==.5).all()),'Fixed paired factors/selector differs')
    payload=cap['factor_a']*cap['factor_b'] if arm=='product' else .5*(cap['factor_a']+cap['factor_b'])
    need(torch.equal(cap['payload'],payload) and torch.equal(cap['messages'],payload*.5) and bool((payload.abs()<=1).all()),'Exact factor/message arithmetic differs')
    need(cap['native_query_hidden'].shape==cap['fused_query_hidden'].shape==(n+1,1,3584)
         and all(cap[k].dtype==torch.float16 for k in native_keys)
         and torch.equal(cap['native_query_hidden'][:-1],h) and torch.equal(cap['native_query_hidden'][-1],g)
         and torch.equal(cap['native_delta'],cap['delta'].half()) and torch.equal(cap['fused_global'],g+cap['native_delta'])
         and torch.equal(cap['fused_query_hidden'][:-1],h) and torch.equal(cap['fused_query_hidden'][-1],cap['fused_global']),
         'Native FP16 cast/add or local-row ownership differs')
    if zero:need(bool((cap['delta']==0).all()) and torch.equal(cap['fused_query_hidden'],cap['native_query_hidden']),'Zero-U identity failed')
    return dict(passed=all(x['passed'] for x in metrics.values()),functional_core=metrics,
        pairing_identity_exact=True,native_cast_and_local_nonmutation_exact=True,zero_up_identity=zero)


def source_archive(directory,plan):
    need(read(directory/'source_hashes.json')==plan['source_sha256']
         and read(directory/'inherited_sources.json')==dict(source_sha256=plan['inherited_source_sha256']), 'Run source descriptors differ')
    for name,digest in plan['source_sha256'].items():bind(directory/'source'/name.replace('/','_'),digest)


def audit_history(torch,packet,bundle,layout,result):
    p=driver();ids=result['generated_ids'];meta=bundle['metadata'];n=meta['row_count'];width=meta['prompt_width'];base=bundle['inputs']
    need(len(packet['native_inputs'])==len(packet['native_positions'])==len(ids),'Actual input/position prefix inventory differs')
    for t,(seen,pos) in enumerate(zip(packet['native_inputs'],packet['native_positions'])):
        mask=seen['attention_mask'];expected_mask=torch.cat((base['attention_mask'],torch.ones((n,t),dtype=base['attention_mask'].dtype)),dim=1)
        expected_ids=base['input_ids'] if t==0 else torch.full((n,1),ids[t-1],dtype=base['input_ids'].dtype)
        need(torch.equal(mask,expected_mask) and torch.equal(seen['input_ids'],expected_ids)
             and seen['has_pixels'] is (t==0) and seen['past_length']==(0 if t==0 else width+t-1), 'Actual raw global history, visual ownership, mask, or native cache length differs')
        rope=layout['position_ids'] if t==0 else (layout['rope_deltas'].view(1,n,1)+width+t-1).expand(3,-1,-1)
        text=mask.long().cumsum(-1)-1
        need(pos.dtype==torch.int64 and pos.shape==(4,n,width if t==0 else 1) and torch.equal(pos[1:],rope), 'Actual native mRoPE differs')
        need(torch.equal(pos[0][mask.bool()],text[mask.bool()]) if t==0 else torch.equal(pos[0],text[:,-1:]), 'Actual native text positions differ')
        need(p.tensor_info(pos)==result['metadata']['generation_position_ids'][t],'Saved native position fingerprint differs')
    return dict(passed=True,prefixes=len(ids),raw_global_token_broadcast=True,original_masks_and_native_positions_exact=True,
        vision_prefill_only=True,cache_lengths_exact=True,all_layer_KV_tensors_hashed=False)


def audit_run(torch,out,directory,plan_path,plan,rows,prepared,layouts,stats,weights,tokenizer,phase,modules=None):
    p=driver();directory=Path(directory).resolve();summary=read(directory/'summary.json');config=read(directory/'config.json');arm=config['arm']
    need(arm in ARMS and config['phase']==summary['phase']==phase and directory.name==f'{phase}_{arm}_{config["slurm_job_id"]}'
         and summary['passed'] is summary['completed'] is True and all(summary[k]==v for k,v in config.items()),'Completed native run/config ownership differs')
    source_archive(directory,plan)
    for key in ('protocol','policy','source_sha256','inherited_source_sha256','cached_report','cached_plan','native_identity','native_identity_sha256','conditioning_identity','stats_file','stats_sha256'):
        need(config[key]==plan[key],'Native configuration differs: '+key)
    need(config['plan_file']==str(plan_path) and config['plan_sha256']==sha(plan_path) and summary['no_fitting'] is summary['no_accuracy_selection'] is summary['all_raw_retained_before_scoring'] is True
         and summary['extra_gpu_head_calls']==0,'Native plan or no-fit/raw-retention contract differs')
    need(config['hardware']['gpu']=='NVIDIA B200' and config['native_weight_identity']==dict(norm=plan['native_identity']['norm_weight'],head=plan['native_identity']['head_weight']), 'Actual hardware/native weights differ')
    if phase=='profile':need(config['main_release'] is None and modules is not None,'Profiles precede release and require CPU heads')
    else:
        need(modules is None and config['main_release']==p.verify_release(config['main_release']['file'],plan_path,plan),'Main must bind the passed measured release')
    cases=plan['profile_cases'] if phase=='profile' else [dict(sid=r['sid'],n_frames=r['n_frames'],state='fitted') for r in rows]
    need(config['cases']==cases and summary['trajectories']==len(cases),'Exact natural trajectory cohort differs')
    need(set(config['checkpoints'])==({'zero','fitted'} if phase=='profile' else {'fitted'}),'Deployed checkpoint inventory differs')
    for state,descriptor in config['checkpoints'].items():
        parent=plan['checkpoints'][arm][state];bind(descriptor['file'],descriptor['sha256'])
        need(Path(descriptor['file'])==Path(config['checkpoint_directory'])/(state+'.pt') and descriptor['sha256']==parent['sha256'] and descriptor['state']==parent['state'], 'Deployed checkpoint copy differs')
    bind(summary['raw_manifest_file'],summary['raw_manifest_sha256']);manifest=read(summary['raw_manifest_file'])
    need(Path(summary['raw_manifest_file'])==directory/'raw_manifest.json' and manifest['completed'] is manifest['all_raw_retained_before_scoring'] is True
         and manifest['n']==len(cases)==len(manifest['rows']),'All raw native trajectories must precede scoring')
    lookup={r['sid']:r for r in rows};outcomes=[];details=[];counts=Counter();head_calls=head_rows=0;max_tv=0.;auditfiles=[]
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
        if phase=='profile':
            need(len(observations)==t,'Profile all-row native head observations differ');norm,head=modules
            for k,(cap,obs) in enumerate(zip(result['captures'],observations)):
                actual=obs['native_logits'];tail=cap['fused_query_hidden'];normal=obs['normalized'];head_input=obs['head_input'];width=bundle['metadata']['prompt_width'] if k==0 else 1
                need(obs['norm_input_shape']==obs['norm_output_shape']==[n+1,width,3584]
                     and obs['head_input_shape']==[n+1,1,3584] and obs['head_output_shape']==[n+1,1,152064]
                     and actual.dtype==normal.dtype==head_input.dtype==torch.float16
                     and actual.shape==(n+1,1,152064) and normal.shape==head_input.shape==tail.shape and torch.equal(normal,head_input)
                     and torch.equal(actual[-1,0],raw[k].half()),'Saved actual profile norm/head batch or global native logits differ')
                with torch.inference_mode():replayed=head(norm(tail))
                metric=factor.replay_metric(torch,actual[:,0],replayed[:,0]);replay.extend(metric);head_calls+=1;head_rows+=n+1
                max_tv=max(max_tv,max(m['full_vocabulary_tv'] for m in metric));del replayed
        else:need(observations is None,'Main must not introduce all-row head replay')
        audit=dict(passed=all(x['passed'] for x in math_rows+replay),case=case,history=history,functional_captures=math_rows,native_head_replays=replay,
            cpu_head_replay_scope='all native profile rows' if phase=='profile' else 'none; every main factor/cast and raw global argmax audited')
        auditfile=out/f'{phase}_{arm}_trajectory_{j:03d}_audit.json';save(auditfile,audit);auditfiles.append(dict(file=str(auditfile),sha256=sha(auditfile)))
        need(audit['passed'],'Independent native capture or registered profile head gate failed; audit retained')
        need(record['audit']==dict(passed=True,emitted_tokens=t,counters=expected,zero_up_identity=state=='zero',raw_unmasked_argmax=True),'GPU trajectory audit differs')
        timing=[record['preprocessing_seconds'],record['work_seconds'],record['four_token_seconds']]
        need(all(math.isfinite(x) and x>0 for x in timing) and record['four_token_seconds']==timing[0]+timing[1]*4/t,'Measured complete trajectory timing differs')
        nll=float(torch.logsumexp(raw[0].double()-raw[0].double().max(),0)-(raw[0,sample['target_ids'][0]].double()-raw[0].double().max()))
        outcomes.append(dict(**{key:sample[key] for key in ('sid','contrast_id','pair_id','variant','n_frames','question','gold','target_ids')},state=state,
            generated_ids=ids,text=result['text'],raw_text=result['raw_text'],first_query_nll=nll,**scored))
        details.append(dict(index=j,case=case,raw_file=record['raw_file'],raw_sha256=record['raw_sha256'],generated_tokens=t,four_token_seconds=record['four_token_seconds'],
            matched_gold_prefix_queries=sum(ids[:k]==sample['target_ids'][:k] for k in range(t)),observed_query_prefixes=[ids[:k] for k in range(t)]))
        counts.update(expected);del packet,result,raw,bundle,math_rows,observations
    bind(summary['endpoint_file'],summary['endpoint_sha256']);endpoint=read(summary['endpoint_file']);native_table=config['native_weight_identity']
    need(endpoint['passed'] is endpoint['native_parameter_versions_unchanged'] is endpoint['statistics_versions_unchanged'] is endpoint['all_trajectory_checkpoint_versions_unchanged'] is endpoint['no_optimizer_or_training'] is True
         and endpoint['native_weights_before']==endpoint['native_weights_after']==native_table
         and endpoint['conditioning_before']==endpoint['conditioning_after']==plan['conditioning_identity']
         and endpoint['final_core_state']==plan['checkpoints'][arm]['fitted']['state'] and endpoint['counters']==summary['counters']==dict(counts),'Actual final native/core/statistic identity differs')
    need(counts['visual']==len(cases) and counts['model']<= (16 if phase=='profile' else 432) and math.isfinite(summary['setup_seconds']) and summary['setup_seconds']>0,'Native phase resource inventory differs')
    outcome_file=out/f'{phase}_{arm}_outcomes.json';save(outcome_file,outcomes)
    return dict(passed=True,arm=arm,phase=phase,directory=str(directory),summary_sha256=sha(directory/'summary.json'),slurm_job_id=config['slurm_job_id'],
        raw_manifest_file=summary['raw_manifest_file'],raw_manifest_sha256=summary['raw_manifest_sha256'],endpoint_file=summary['endpoint_file'],endpoint_sha256=summary['endpoint_sha256'],
        outcomes_file=str(outcome_file),outcomes_sha256=sha(outcome_file),trajectory_audits=auditfiles,trajectories=details,counters=dict(counts),
        setup_seconds=summary['setup_seconds'],four_token_seconds=[r['four_token_seconds'] for r in manifest['rows']],
        native_head_rows=sum(r['generated_tokens']*(r['case']['n_frames']+1) for r in details),cpu_head_calls=head_calls,cpu_head_rows=head_rows,maximum_cpu_native_head_tv=max_tv,all_profile_native_head_argmax_exact=True if phase=='profile' else None,
        native_whole_answer_screen=p.criterion(outcomes) if phase=='main' else None,profile_outcomes_descriptive_only=phase=='profile',
        checkpoint=plan['checkpoints'][arm]['fitted'],native_identity_sha256=plan['native_identity_sha256'],outcomes=outcomes)


def resources(out,profiles,runs=None):
    from datetime import datetime
    p=driver();command=['sacct','-X','-u',os.environ['USER'],'--starttime=2026-09-01','--parsable2','--noheader',
        '--format=JobIDRaw,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,Start,End']
    raw=subprocess.run(command,capture_output=True,text=True,check=True).stdout;file=out/'all_user_sacct.psv';file.write_text(raw)
    names={name for group in p.JOBS.values() for name in group.values()};allocations=factor.allocation_rows(raw,names)
    expected={p.JOBS['profile'][a]:profiles[a] for a in ARMS}
    if runs is not None:expected.update({p.JOBS['main'][a]:runs[a] for a in ARMS})
    need(len(allocations)==len(expected) and {r['name'] for r in allocations}==set(expected),'Exactly one attempt per phase/arm required; no failed or zero allocation omitted')
    events=[]
    for row in allocations:
        phase='profile' if row['name'] in p.JOBS['profile'].values() else 'main';cap=120 if phase=='profile' else 480
        need(row['job_id']==expected[row['name']]['slurm_job_id'] and row['partition']=='gpu' and row['state']=='COMPLETED'
             and row['exit_code']=='0:0' and row['gpus']==1 and row['seconds']<=cap,'Native allocation identity or fixed cap differs')
        start,end=map(datetime.fromisoformat,(row['start'],row['end']));need(end>=start,'Native allocation interval differs')
        if end>start:events.extend(((start,1),(end,-1)))
    active=peak=0
    for _,change in sorted(events):active+=change;need(active>=0,'Invalid GPU chronology');peak=max(peak,active)
    used=sum(r['gpu_seconds'] for r in allocations);reservation=960 if runs is None else 0
    need(active==0 and peak<=2 and used+reservation<=1200,'Native two-GPU/1200-second campaign cap exceeded')
    return dict(passed=True,jobs=allocations,allocated_gpu_seconds=used,reserved_main_gpu_seconds=reservation,total_with_reservation=used+reservation,
        campaign_seconds=1200,maximum_concurrent_gpus=peak,failed_and_zero_allocations_retained=True,file=str(file),sha256=sha(file),command=command)


def pooled_projection(profiles):
    need(set(profiles)==set(ARMS) and all(len(profiles[a]['four_token_seconds'])==4 for a in ARMS),'Exact eight-trajectory profile inventory required')
    setups=[profiles[a]['setup_seconds'] for a in ARMS];values=[t for a in ARMS for t in profiles[a]['four_token_seconds']]
    need(all(math.isfinite(x) and x>0 for x in setups+values),'Invalid profile timings')
    setup=max(setups);observed=max(values);floor=2.30838355794549;bound=max(floor,observed);total=setup+1.25*108*bound+60.
    value=dict(passed=total<=480,pooled_setup_seconds=setup,maximum_observed_four_token_seconds=observed,historical_floor_seconds=floor,
        pooled_scene_seconds=bound,multiplier=1.25,contexts=108,reserve_seconds=60.,projected_seconds=total,cap_seconds=480.)
    need(value==driver().projection(setups,values),'Independent timing projection differs from registered driver policy')
    return value


def bound_outputs(value):
    """Hash-only closure for previously completed report/profile evidence."""
    bind(value['outcomes_file'],value['outcomes_sha256']);need(read(value['outcomes_file'])==value['outcomes'],'Saved independent outcomes differ')
    bind(Path(value['directory'])/'summary.json',value['summary_sha256']);bind(value['raw_manifest_file'],value['raw_manifest_sha256'])
    bind(value['endpoint_file'],value['endpoint_sha256'])
    for item in value['trajectory_audits']:bind(item['file'],item['sha256']);need(read(item['file'])['passed'] is True,'Prior independent capture audit failed')
    for item in value['trajectories']:bind(item['raw_file'],item['raw_sha256'])


def release(args,out,frozen):
    import torch
    from transformers import AutoProcessor
    p=driver();torch.set_num_threads(4);plan_path=args.plan.resolve();plan=p.verify_plan(plan_path,ancestors=True)
    rows,prepared,layouts,stats,weights=inputs(torch,plan)
    processor=AutoProcessor.from_pretrained(str(p.MODEL),trust_remote_code=True,use_fast=False,local_files_only=True)
    bind(plan['native_model_file'],plan['native_model_sha256']);packet=torch.load(plan['native_model_file'],map_location='cpu',weights_only=True)
    need(p.oracle.native_module_identity(torch,plan['native_identity'])==plan['native_module_identity'],'Actual CPU native module source differs')
    norm,head=p.oracle.native_modules(torch,packet,plan['native_identity'],'cpu');native_before=dict(norm=p.tensor_info(norm.weight),head=p.tensor_info(head.weight))
    profiles={}
    for directory in args.profiles:
        value=audit_run(torch,out,directory,plan_path,plan,rows,prepared,layouts,stats,weights,processor.tokenizer,'profile',(norm,head))
        need(value['arm'] not in profiles,'Duplicate native profile arm');profiles[value['arm']]=value
    need(set(profiles)==set(ARMS) and native_before==dict(norm=p.tensor_info(norm.weight),head=p.tensor_info(head.weight))
         and all(not v.requires_grad and v.grad is None for m in (norm,head) for v in m.parameters()),'Both profiles and unchanged CPU native weights required')
    calls=sum(v['cpu_head_calls'] for v in profiles.values());nrows=sum(v['cpu_head_rows'] for v in profiles.values())
    need(8<=calls<=32 and nrows<=416 and all(v['counters']['visual']==4 and v['cpu_head_calls']==v['counters']['head'] for v in profiles.values()),'All actual profile prefixes/rows must be replayed exactly once')
    projection=pooled_projection(profiles);budget=resources(out,profiles)
    audit=dict(passed=True,profiles=profiles,projection=projection,resources=budget,cpu_head_calls=calls,cpu_head_rows=nrows,
        maximum_cpu_native_head_tv=max(v['maximum_cpu_native_head_tv'] for v in profiles.values()),all_profile_native_head_argmax_exact=True,
        profile_natural_outcomes_are_descriptive=True,native_cpu_head_replay_scope='all profile prefixes/all native rows; no main head replay')
    save(out/'profile_audit.json',audit)
    need(projection['passed'],'Registered conservative timing projection exceeds 480 seconds; all profile evidence preserved and no main release')
    value=dict(passed=True,protocol=p.PROTOCOL,policy=p.POLICY,source_sha256=frozen,inherited_source_sha256=plan['inherited_source_sha256'],
        plan_file=str(plan_path),plan_sha256=sha(plan_path),profiles=profiles,projection=projection,main_reservation_seconds=960,
        profile_audit_file=str(out/'profile_audit.json'),profile_audit_sha256=sha(out/'profile_audit.json'),resources=budget)
    save(out/'release.json',value)
    return dict(passed=True,completed=True,phase='release',protocol=p.PROTOCOL,source_sha256=frozen,inherited_source_sha256=plan['inherited_source_sha256'],
        plan_file=str(plan_path),plan_sha256=sha(plan_path),release_file=str(out/'release.json'),release_sha256=sha(out/'release.json'),
        profile_audit_file=str(out/'profile_audit.json'),profile_audit_sha256=sha(out/'profile_audit.json'),projection=projection,resources=budget,
        cpu_head_calls=calls,cpu_head_rows=nrows,maximum_cpu_native_head_tv=audit['maximum_cpu_native_head_tv'])


def strata(rows):
    tables={}
    for keys in (('n_frames',),('gold',),('question',),('variant',)):
        groups=defaultdict(list)
        for row in rows:groups[tuple(row[k] for k in keys)].append(row)
        tables['_'.join(keys)]=[dict(labels=dict(zip(keys,key)),metrics=dict(contexts=len(values),whole_correct=sum(r['exact'] for r in values),
            first_correct=sum(r['first_token_correct'] for r in values),completed=sum(r['completed'] for r in values),truncated=sum(r['truncated'] for r in values),
            first_query_nll=sum(r['first_query_nll'] for r in values)/len(values))) for key,values in sorted(groups.items())]
    return tables


def report(args,out,frozen):
    import torch
    from transformers import AutoProcessor
    p=driver();torch.set_num_threads(4);plan_path=args.plan.resolve();plan=p.verify_plan(plan_path,ancestors=True)
    release_ref=p.verify_release(args.main_release,plan_path,plan);released=read(release_ref['file']);profiles=released['profiles']
    bind(released['profile_audit_file'],released['profile_audit_sha256']);profile_audit=read(released['profile_audit_file'])
    need(profile_audit['passed'] is True and profile_audit['profiles']==profiles and profile_audit['all_profile_native_head_argmax_exact'] is True,'Complete independent profile audit required')
    for a in ARMS:bound_outputs(profiles[a])
    rows,prepared,layouts,stats,weights=inputs(torch,plan);processor=AutoProcessor.from_pretrained(str(p.MODEL),trust_remote_code=True,use_fast=False,local_files_only=True);runs={}
    for directory in args.runs:
        value=audit_run(torch,out,directory,plan_path,plan,rows,prepared,layouts,stats,weights,processor.tokenizer,'main')
        need(value['arm'] not in runs,'Duplicate native main arm');runs[value['arm']]=value
    need(set(runs)==set(ARMS) and all(v['counters']['visual']==108 and v['cpu_head_calls']==v['cpu_head_rows']==0 for v in runs.values()),'Both complete native main arms and profile-only CPU heads required')
    for a in ARMS:
        config=read(Path(runs[a]['directory'])/'config.json');need(config['main_release']==release_ref,'Both mains must bind this exact release')
    budget=resources(out,profiles,runs);screens={a:runs[a]['native_whole_answer_screen'] for a in ARMS}
    left={r['sid']:r for r in runs['product']['outcomes']};right={r['sid']:r for r in runs['additive']['outcomes']}
    transitions=Counter(('both_correct' if left[s]['exact'] and right[s]['exact'] else 'product_only' if left[s]['exact'] else 'additive_only' if right[s]['exact'] else 'both_wrong') for s in left)
    coverage=dict(main_contexts=216,main_families_per_arm=18,main_native_calls=sum(v['counters']['model'] for v in runs.values()),
        main_native_head_rows=sum(v['native_head_rows'] for v in runs.values()),main_vision_calls=216,main_cpu_head_calls=0,
        profile_cpu_head_calls=profile_audit['cpu_head_calls'],profile_cpu_head_rows=profile_audit['cpu_head_rows'],
        profile_vision_calls=8,all_gpu_native_calls=sum(v['counters']['model'] for v in list(runs.values())+list(profiles.values())),
        all_gpu_native_head_rows=sum(v['native_head_rows'] for v in list(runs.values())+list(profiles.values())),extra_gpu_head_calls=0)
    need(coverage['main_native_calls']<=864 and coverage['main_native_head_rows']<=11232 and coverage['profile_cpu_head_calls']<=32 and coverage['profile_cpu_head_rows']<=416,'Joint native inventory exceeds registration')
    analysis=dict(passed=True,completed=True,protocol=p.PROTOCOL,phase='report',policy=p.POLICY,source_sha256=frozen,inherited_source_sha256=plan['inherited_source_sha256'],
        plan_file=str(plan_path),plan_sha256=sha(plan_path),cached_report=plan['cached_report'],main_release=release_ref,
        profiles=profiles,runs=runs,native_whole_answer_screen=screens,at_least_one_native_training_screen_passed=any(v['passed'] for v in screens.values()),
        all_raw_native_outputs_retained=True,native_training_only=True,no_dev_or_fresh_evaluation=True,no_fitting_or_checkpoint_selection=True,
        cpu_native_head_replay_scope='profile only; all full-run factor/cast/raw-global-output evidence audited',
        maximum_profile_cpu_native_head_tv=profile_audit['maximum_cpu_native_head_tv'],all_profile_native_head_argmax_exact=True,
        resources=budget,coverage=coverage,descriptive_product_additive_transitions=dict(transitions),strata={a:strata(runs[a]['outcomes']) for a in ARMS})
    save(out/'analysis.json',analysis)
    lines=['# Fixed factor native training evaluation','',
        'This report evaluates natural full-name plus EOS generation on the same 108 training contexts per arm. It performs no fresh-data or development evaluation.','',
        '| Arm | Whole answer + EOS | First token | Complete six-context families | Registered training screen |',
        '| --- | ---: | ---: | ---: | --- |']
    for a in ARMS:
        s=screens[a];lines.append(f'| {a} | {s["whole_correct"]}/108 | {s["first_correct"]}/108 | {s["complete_families"]}/18 | {"PASS" if s["passed"] else "FAIL"} |')
    lines+=['',f'Both fixed 6000-step checkpoints were evaluated; none was selected by these results. Product/additive paired transitions: {dict(transitions)}.',
        '',f'CPU native norm/head replay covered only the profiles: {coverage["profile_cpu_head_calls"]} calls and {coverage["profile_cpu_head_rows"]} native rows; maximum full-vocabulary TV {analysis["maximum_profile_cpu_native_head_tv"]:.9g}, all argmaxes exact. Main CPU head replay: zero. Every main factor, global conditioning, FP16 cast, raw argmax, native history, mask and position was independently checked.',
        '',f'The native stage used {budget["allocated_gpu_seconds"]} allocated GPU-seconds, {coverage["all_gpu_native_calls"]} native model/norm/head calls, and no extra GPU head calls. All raw answers, premature EOS, truncations and observed continuation prefixes are retained. Full-layer KV tensor equality was not asserted.',
        '', 'A passing training screen establishes native execution/trainability on these seen contexts only. Fresh composition, length transfer, superiority of multiplication, general aggregation, and reasoning composition require separate evidence; this report releases none of those claims.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return dict(passed=True,completed=True,phase='report',protocol=p.PROTOCOL,source_sha256=frozen,inherited_source_sha256=plan['inherited_source_sha256'],
        plan_file=str(plan_path),plan_sha256=sha(plan_path),analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),
        report_file=str(out/'REPORT.md'),report_sha256=sha(out/'REPORT.md'),native_whole_answer_screen=screens,
        at_least_one_native_training_screen_passed=analysis['at_least_one_native_training_screen_passed'],resources=budget,coverage=coverage)


def verify_native_report(summary_path):
    """JSON/hash-only dependency for a separately authorized fresh-data stage."""
    p=driver();summary_path=Path(summary_path).resolve();summary=read(summary_path)
    need(summary['passed'] is summary['completed'] is True and summary['phase']=='report' and summary['protocol']==p.PROTOCOL
         and summary['source_sha256']==p.sources() and summary['inherited_source_sha256']==p.inherited_sources(), 'Complete independent native report required')
    source_archive(summary_path.parent,summary);bind(summary['analysis_file'],summary['analysis_sha256']);bind(summary['report_file'],summary['report_sha256'])
    analysis=read(summary['analysis_file']);need(analysis['passed'] is analysis['completed'] is True and analysis['phase']=='report' and analysis['protocol']==p.PROTOCOL
         and analysis['source_sha256']==summary['source_sha256'] and analysis['inherited_source_sha256']==summary['inherited_source_sha256']
         and analysis['native_whole_answer_screen']==summary['native_whole_answer_screen'] and set(analysis['runs'])==set(ARMS)
         and analysis['all_raw_native_outputs_retained'] is analysis['native_training_only'] is analysis['no_dev_or_fresh_evaluation'] is True,
         'Native whole-answer evidence cannot be replaced by cached/profile results')
    plan=p.verify_plan(analysis['plan_file'],ancestors=False);need(sha(analysis['plan_file'])==analysis['plan_sha256']==summary['plan_sha256'],'Native report plan changed')
    need(p.verify_release(analysis['main_release']['file'],analysis['plan_file'],plan)==analysis['main_release'],'Native profile release changed')
    for a in ARMS:
        value=analysis['runs'][a];bound_outputs(value);rows=value['outcomes'];need(value['phase']=='main' and value['arm']==a and len(rows)==108
            and value['counters']['visual']==108 and all(r['state']=='fitted' for r in rows)
            and p.criterion(rows)==value['native_whole_answer_screen']==analysis['native_whole_answer_screen'][a], 'Complete natural training denominator/decision differs')
    need(analysis['at_least_one_native_training_screen_passed'] is summary['at_least_one_native_training_screen_passed'] is True
         and any(analysis['native_whole_answer_screen'][a]['passed'] for a in ARMS),'No registered native training screen passed')
    return analysis


def main():
    import argparse
    import json
    import time
    p=driver();parser=argparse.ArgumentParser(description=__doc__);action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--release',action='store_true');action.add_argument('--report',action='store_true');parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--profiles',type=Path,nargs=2);parser.add_argument('--runs',type=Path,nargs=2);parser.add_argument('--main-release',type=Path)
    args=parser.parse_args();p.native.require_slurm();need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),'CPU-only independent release/report')
    need(args.profiles is not None if args.release else args.runs is not None and args.main_release is not None,'Exact phase inputs required')
    phase='release' if args.release else 'report';out=p.OUT/f'{phase}_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();frozen=p.snapshot(out)
    save(out/'request.json',dict(phase=phase,plan=str(args.plan),source_sha256=frozen))
    try:
        value=release(args,out,frozen) if args.release else report(args,out,frozen)
        need(p.sources()==frozen and p.inherited_sources()==value['inherited_source_sha256'],'Native report source changed during audit')
        value['elapsed_seconds']=time.perf_counter()-started;save(out/'summary.json',value);print(json.dumps(dict(passed=True,directory=str(out))),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,elapsed_seconds=time.perf_counter()-started,partial_outputs_retained=True));raise


if __name__=='__main__':main()

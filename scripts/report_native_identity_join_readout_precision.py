"""Separate FP64 NLL reporting repair; original failed reporter remains immutable."""
from pathlib import Path
import sys
_REPO=Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:sys.path.insert(0,str(_REPO))
from collections import Counter, defaultdict
import math
import os
import random
import subprocess
from scripts.stage_native_vision_v6_teacher import need, read, save, sha, object_sha


def driver():
    from scripts import diagnose_native_identity_join_readout as module
    return module


def bind(file, digest):
    need(sha(Path(file)) == digest, 'Changed bound file: ' + str(file))


def close(a, b, atol=2e-6, rtol=2e-6):
    return math.isfinite(a) and math.isfinite(b) and abs(a-b) <= atol+rtol*max(abs(a),abs(b))


def expected_order(pairs):
    need(len(pairs)==54 and len({p['pair_id'] for p in pairs})==54, 'Pair inventory differs')
    rng=random.Random(24); result=[]; cycle=0
    while len(result)<4800:
        cycle+=1; slots=list(range(54)); rng.shuffle(slots)
        result.extend(dict(cycle=cycle,pair_slot=i,pair_id=pairs[i]['pair_id'],sids=pairs[i]['sids']) for i in slots)
    return result[:4800]


def independent_criteria(rows):
    need(len(rows)==108 and len({r['sid'] for r in rows})==108, 'First-token denominator differs')
    families=defaultdict(list)
    for row in rows:
        need(type(row['first_token_correct']) is bool, 'Correctness must be Boolean')
        families[row['contrast_id']].append(row)
    need(len(families)==18 and all(len(v)==6 and {(r['variant'],r['n_frames']) for r in v}==
        {(i,n) for i in range(3) for n in (8,16)} for v in families.values()), 'Family coverage differs')
    correct=sum(r['first_token_correct'] for r in rows)
    complete=sum(all(r['first_token_correct'] for r in v) for v in families.values())
    return dict(passed=correct>=103 and complete>=16,correct=correct,contexts=108,complete_families=complete,families=18,
        thresholds=dict(correct=103,contexts=108,complete_families=16,families=18),first_token_only=True,oracle_answer_code_supplied=True)


def replay_metric(torch, reference, replay):
    need(reference.shape==replay.shape and reference.ndim==2 and bool(torch.isfinite(replay).all()), 'Invalid replay shape/logits')
    a=reference.float(); b=replay.float()
    tv=(a.softmax(-1)-b.softmax(-1)).abs().sum(-1)*.5
    match=a.argmax(-1)==b.argmax(-1)
    return [dict(full_vocabulary_tv=float(t),argmax_exact=bool(m),passed=float(t)<=.02 and bool(m)) for t,m in zip(tv,match)]


def self_test(torch):
    rows=[dict(sid=f'{f}_{v}_{n}',contrast_id=f,variant=v,n_frames=n,first_token_correct=True)
          for f in range(18) for v in range(3) for n in (8,16)]
    need(independent_criteria(rows)['passed'], 'Complete-family positive fixture failed')
    for i in (0,6,12):rows[i]['first_token_correct']=False
    value=independent_criteria(rows)
    need(value['correct']==105 and value['complete_families']==15 and not value['passed'], 'Family threshold fixture failed')
    pairs=[dict(pair_id=str(i),sids=[str(i)+'a',str(i)+'b']) for i in range(54)]
    order=expected_order(pairs)
    need(len(order)==4800 and len({p['pair_slot'] for p in order[:54]})==54 and order==expected_order(pairs), 'Persistent order fixture failed')
    logits=torch.tensor([[2.,1.,0.],[0.,1.,2.]])
    need(all(r['passed'] for r in replay_metric(torch,logits,logits))
         and not all(r['passed'] for r in replay_metric(torch,logits,logits.flip(-1))), 'Replay rejection fixture failed')
    return dict(passed=True,groups=4,complete_family_threshold=True,deterministic_pair_order=True,replay_accept_and_reject=True)


def audit_inputs(torch, plan):
    p=driver(); rows=read(plan['rows_file']); pairs=read(plan['pairs_file']); order=read(plan['order_file'])
    parent=read(plan['uniform_plan']['file']); bind(plan['uniform_plan']['file'],plan['uniform_plan']['sha256'])
    originals=read(parent['rows_file'])['train']; scenes=read(parent['scenes_file'])
    gids=sorted({scenes[r['sample']['sid']]['global_feature_ids'][0] for r in originals})
    expected=[]
    for r in originals:
        s=r['sample']; sid=s['sid']; target=s['target_ids']; fid=scenes[sid]['global_feature_ids'][0]
        need(s['split']=='train' and r['cell']=='train_N'+str(s['n_frames']) and scenes[sid]['target_prefixes'][0]==[], 'Nontraining/prefixed input')
        expected.append(dict(sid=sid,contrast_id=s['contrast_id'],pair_id=s['pair_id'],variant=s['variant'],n_frames=s['n_frames'],
            question=s['question'],gold=s['gold'],target_ids=target,class_index=p.PEOPLE.index(s['gold']),global_feature_id=fid,
            global_row=gids.index(fid),full_target_length=len(target),first_token_id=target[0]))
    need(rows==expected and pairs==read(parent['pairs_file']) and order==read(parent['order_file'])==expected_order(pairs), 'Original train/label/order binding differs')
    need(Counter(r['gold'] for r in rows)==Counter({name:12 for name in p.PEOPLE}) and len(gids)==6, 'Class/question coverage differs')
    for r in rows:
        need(r['full_target_length']==(3 if r['gold'] in ('Sandra','Noah') else 2)
             and r['target_ids'][-1]==151645, 'Original target weighting differs')
    by_sid={r['sid']:r for r in rows}
    for pair in pairs:
        left,right=(by_sid[sid] for sid in pair['sids'])
        need([left['n_frames'],right['n_frames']]==[8,16]
             and all(left[k]==right[k] for k in ('gold','target_ids','contrast_id','variant','question','global_row')), 'Length pair differs')
    inputs=torch.load(plan['inputs_file'],map_location='cpu',weights_only=True)
    need(inputs['schema_version']==1 and inputs['global_feature_ids']==gids==plan['global_feature_ids']
         and {k:p.base.v7.tensor_info(inputs[k]) for k in ('global_states','codebook')}==plan['input_tensors'], 'Actual input tensor identities differ')
    expected_code=torch.zeros((9,96),dtype=torch.float32)
    for i in range(9):expected_code[i,i]=math.sqrt(96)
    need(torch.equal(inputs['codebook'],expected_code) and inputs['global_states'].dtype==torch.float16
         and inputs['global_states'].shape==(6,3584), 'Oracle code placement/scale or global inputs differ')
    cached=torch.load(parent['features_file'],map_location='cpu',weights_only=True)
    lookup={fid:i for i,fid in enumerate(cached['feature_ids'])}
    need(torch.equal(inputs['global_states'],torch.stack([cached['states'][lookup[fid]] for fid in gids])), 'Global empty cache bytes differ')
    del cached
    inventory=read(plan['input_inventory_file'])
    need(inventory['rows']==rows and inventory['classes']==list(p.PEOPLE) and inventory['global_only'] and inventory['no_local_features']
         and inventory['no_dev_or_test'] and inventory['global_feature_ids']==gids and inventory['all_canonical_targets']==sum(r['full_target_length'] for r in rows), 'Input inventory differs')
    return rows,order,inputs


def audit_training(directory, summary, rows, order):
    bind(summary['training_file'],summary['training_sha256']); logs=read(summary['training_file']); by_sid={r['sid']:r for r in rows}
    need(len(logs)==600, 'Update coverage differs')
    for step,row in enumerate(logs,1):
        batch=order[(step-1)*8:step*8]; sids=[sid for b in batch for sid in b['sids']]; chosen=[by_sid[sid] for sid in sids]
        lengths=[r['full_target_length'] for r in chosen]; nll=row['first_token_nll']; parts=row['weighted_contributions']
        rate=.001*step/50 if step<=50 else .00001+(.001-.00001)*(1+math.cos(math.pi*(step-50)/550))/2
        need(row['step']==step and row['sids']==sids and row['pair_ids']==[b['pair_id'] for b in batch]
             and row['cycles']==[b['cycle'] for b in batch] and row['first_token_ids']==[r['first_token_id'] for r in chosen]
             and row['full_target_lengths']==lengths and close(row['lr'],rate,atol=1e-12,rtol=1e-12), 'Training schedule/target differs')
        need(len(nll)==len(parts)==16 and all(math.isfinite(v) and v>=0 for v in nll)
             and all(close(a/l,b) for a,l,b in zip(nll,lengths,parts)) and close(sum(parts)/16,row['loss'])
             and close(sum(nll)/16,row['unweighted_ce']), 'First-token CE weighting/reduction differs')
        need(math.isfinite(row['gradient_norm']) and row['gradient_norm']>=0 and row['clipped']==(row['gradient_norm']>1.)
             and row['nonactive_columns_zero'] is True and row['native_frozen'] is True and row['seconds']>=0, 'Gradient/frozen logging contract differs')
    need(sorted(x.name for x in directory.glob('progress_*.json'))==[f'progress_{i:04d}.json' for i in range(100,601,100)], 'Progress file coverage differs')
    for step in range(100,601,100):
        value=read(directory/f'progress_{step:04d}.json')
        need(value==dict(step=step,training=logs[:step],counters=dict(norm=step,head=step,vlm=0,vision=0)), 'Immutable progress prefix differs')
    return dict(passed=True,updates=600,pair_presentations=4800,scene_presentations=9600,
        first_token_loss_positions=9600,terminal_weighted_loss=logs[-1]['loss'],terminal_unweighted_ce=logs[-1]['unweighted_ce'],
        logging_and_source_audit=True,no_independent_optimizer_trajectory_replay=True)


def resources(out, job):
    p=driver(); command=['sacct','-X','-u',os.environ['USER'],'--starttime=2026-09-01','--parsable2','--noheader',
        '--format=JobIDRaw,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,Start,End']
    raw=subprocess.run(command,capture_output=True,text=True,check=True).stdout; file=out/'all_user_sacct.psv';file.write_text(raw); records=[]
    for line in raw.splitlines():
        f=line.split('|');need(len(f)==9,'Scheduler row schema differs')
        if f[1]!=p.RUN_JOB:continue
        generic=[];typed=[]
        for field in f[6].split(','):
            if field.startswith('gres/gpu='):generic.append(int(field.split('=')[1]))
            elif field.startswith('gres/gpu:'):typed.append(int(field.split('=')[1]))
        gpus=max(generic) if generic else sum(typed)
        records.append(dict(job_id=f[0],name=f[1],partition=f[2],state=f[3],exit_code=f[4],seconds=int(f[5]),gpus=gpus,gpu_seconds=int(f[5])*gpus))
    need(len(records)==1 and records[0]['job_id']==job and records[0]['partition']=='gpu'
         and records[0]['state']=='COMPLETED' and records[0]['exit_code']=='0:0' and records[0]['gpus']==1
         and records[0]['gpu_seconds']<=150,'One completed GPU attempt within150 allocated seconds required')
    return dict(passed=True,jobs=records,allocated_gpu_seconds=sum(r['gpu_seconds'] for r in records),maximum_concurrent_gpus=1,
        failed_and_zero_allocations_retained=True,file=str(file),sha256=sha(file),command=command)


def report(args,out,frozen):
    import torch
    torch.set_num_threads(4);p=driver(); plan=p.verify_plan(args.plan,ancestors=True)
    conditional_bindings={};_,conditional=p.conditional_gate(conditional_bindings)
    need(conditional==plan['conditional_report'], 'Uniform trigger changed')
    directory=Path(args.runs[0]).resolve(); config=read(directory/'config.json'); summary=read(directory/'summary.json')
    need(directory.parent==p.OUT and directory.name==config['run_id']=='run_'+config['slurm_job_id'] and config['protocol']==p.PROTOCOL
         and config['policy']==p.POLICY and config['source_sha256']==frozen==plan['source_sha256']
         and Path(config['plan_file']).resolve()==Path(args.plan).resolve() and config['plan_sha256']==sha(args.plan)
         and all(summary.get(k)==v for k,v in config.items()), 'Run/config/source/plan join differs')
    for n,h in frozen.items():bind(directory/'source'/n.replace('/','_'),h)
    for flag in ('passed','completed','computational_integrity_passed','checkpoint_roundtrip_passed','native_weights_unchanged',
                 'no_pretrained_backbone_loaded','no_dev_or_test','no_eos_or_whole_answer_claim'):
        need(summary[flag] is True, 'Run flag failed: '+flag)
    need(summary['phase']=='run' and summary['steps']==600 and summary['training_contexts']==108 and summary['scene_presentations']==9600, 'Run endpoint differs')
    for key in ('native_identity','native_identity_sha256','native_module_identity','native_model_file','native_model_sha256'):
        need(config[key]==plan[key], 'Native source/weight metadata differs')
    need(p.native_module_identity(torch,plan['native_identity'])==plan['native_module_identity'], 'Installed native modules changed')
    for key in ('input_inventory','presentations','training','predictions','raw','calls','endpoint','roundtrip','final_inputs'):
        bind(summary[key+'_file'],summary[key+'_sha256'])
    need(read(config['input_inventory_file'])==read(plan['input_inventory_file']) and read(config['presentations_file'])==read(plan['order_file']), 'Copied prefit artifacts differ')
    rows,order,inputs=audit_inputs(torch,plan); training=audit_training(directory,summary,rows,order)
    tensor_info=p.base.v7.tensor_info
    initial=torch.load(plan['initial_file'],map_location='cpu',weights_only=True)
    need(initial['schema_version']==1 and initial['step']==0 and initial['policy']==p.POLICY and initial['source_sha256']==frozen
         and tensor_info(initial['U'])==plan['initial_U']==config['initialized_U'] and bool((initial['U']==0).all()), 'Zero-only initial tensor differs')
    bind(summary['checkpoint'],summary['checkpoint_sha256']); checkpoint=torch.load(summary['checkpoint'],map_location='cpu',weights_only=True);U=checkpoint['U']
    need(set(checkpoint)=={'schema_version','U','step','config'} and checkpoint['schema_version']==1 and checkpoint['step']==600
         and checkpoint['config']==config and U.dtype==torch.float32 and U.shape==(3584,96) and U.numel()==344064
         and bool(torch.isfinite(U).all()) and bool((U[:,9:]==0).all()) and U[:,:9].numel()==32256, 'Final U-only checkpoint differs')
    need(Path(summary['checkpoint'])==p.CKPT/directory.name/'final.pt' and Path(config['data_directory'])==p.DATA/directory.name, 'Storage roots differ')
    info=tensor_info(U);end=read(summary['endpoint_file']);roundtrip=read(summary['roundtrip_file'])
    native_weights=dict(norm=plan['native_identity']['norm_weight'],head=plan['native_identity']['head_weight'])
    need(info==summary['selected_U']==end['U_before_evaluation']==end['U_after_evaluation']==roundtrip['final_U']==roundtrip['reloaded_U']
         and roundtrip['initial_U']==plan['initial_U'] and roundtrip['passed'] is True and end['passed'] is True
         and end['native_before']==end['native_after']==config['native_weight_identity']==native_weights
         and end['inactive_columns_exactly_zero'] is True and end['native_weights_unchanged'] is True
         and roundtrip['config_sha256']==sha(directory/'config.json') and roundtrip['step']==600, 'Checkpoint reset/reload/frozen endpoint differs')
    for item in (roundtrip,end):need(item['checkpoint']==summary['checkpoint'] and item['checkpoint_sha256']==summary['checkpoint_sha256'], 'Endpoint checkpoint binding differs')
    raw=torch.load(summary['raw_file'],map_location='cpu',weights_only=True);logits=raw['logits']
    need(raw['schema_version']==1 and raw['sids']==[r['sid'] for r in rows] and logits.dtype==torch.float16
         and logits.shape==(108,152064) and bool(torch.isfinite(logits).all()) and tensor_info(logits)==summary['raw_tensor']==end['raw_logits']
         and raw['batches']==summary['raw_batches'] and len(raw['batches'])==7, 'Final unmasked native raw logits differ')
    cursor=0
    for batch in raw['batches']:
        bind(batch['file'],batch['sha256']);value=torch.load(batch['file'],map_location='cpu',weights_only=True);count=len(batch['sids'])
        need(count==min(16,108-cursor) and value['sids']==batch['sids']==raw['sids'][cursor:cursor+count]
             and torch.equal(value['logits'],logits[cursor:cursor+count]) and tensor_info(value['logits'])==batch['logits'], 'Raw batch ownership differs');cursor+=count
    ids=logits.argmax(-1);targets=torch.tensor([r['first_token_id'] for r in rows])
    promoted=logits.double();centered=promoted-promoted.amax(-1,keepdim=True)
    nll=torch.logsumexp(centered,dim=-1)-centered[torch.arange(108),targets]
    # Only descriptive NLL recomputation precision changes; original close() is unchanged.
    cpu_fp32_nll=torch.nn.functional.cross_entropy(logits.float(),targets,reduction='none')
    outcomes=[dict(r,argmax_id=int(ids[i]),first_token_correct=int(ids[i])==r['first_token_id'],nll=float(nll[i])) for i,r in enumerate(rows)]
    saved=read(summary['predictions_file']);need(len(saved)==108, 'Prediction rows differ')
    for a,b in zip(outcomes,saved):need(all(a[k]==b[k] for k in a if k!='nll') and close(a['nll'],b['nll']), 'Independent full-vocabulary rescore differs')
    precision_rows=[dict(sid=a['sid'],fp64_nll=a['nll'],gpu_fp32_nll=b['nll'],cpu_fp32_nll=float(cpu_fp32_nll[i]),
        original_cpu_fp32_comparison_passed=close(float(cpu_fp32_nll[i]),b['nll']),
        fp64_comparison_passed=close(a['nll'],b['nll']),gpu_fp32_vs_fp64_absolute=abs(a['nll']-b['nll']))
        for i,(a,b) in enumerate(zip(outcomes,saved))]
    save(out/'nll_precision.json',dict(diagnosis_file=str(DIAGNOSIS),diagnosis_sha256=DIAGNOSIS_SHA,rows=precision_rows,
        original_report_remains_failed=True,unchanged_tolerance=dict(atol=2e-6,rtol=2e-6),all_fp64_comparisons_passed=True))
    criterion=independent_criteria(outcomes);need(criterion==summary['first_token_fit'], 'Independent first-token fit criterion differs')
    calls=read(summary['calls_file']);expected_calls=[]
    for phase,count in (('training',600),('evaluation',7)):
        for step in range(1,count+1):
            batch=12 if phase=='evaluation' and step==7 else 16
            for module in ('norm','head'):
                expected_calls.append(dict(module=module,phase=phase,step=step,input_shape=[batch,3584],output_shape=[batch,3584 if module=='norm' else 152064],
                    input_dtype='torch.float16',output_dtype='torch.float16'))
    counters=dict(norm=607,head=607,vlm=0,vision=0)
    need(calls==expected_calls and summary['counters']==end['counters']==counters
         and read(directory/'final_counters.json')==dict(counters=counters,recorded_calls=calls,partial_outputs_retained=True), 'Observed norm/head call inventory differs')
    cap=torch.load(summary['final_inputs_file'],map_location='cpu',weights_only=True)
    need(cap['schema_version']==1 and cap['sids']==raw['sids']
         and {k:tensor_info(cap[k]) for k in ('delta','fused_global','normalized')}==summary['final_input_tensors'], 'Final readout capture differs')
    for key,dtype in (('delta',torch.float32),('fused_global',torch.float16),('normalized',torch.float16)):
        need(cap[key].shape==(108,3584) and cap[key].dtype==dtype and bool(torch.isfinite(cap[key]).all()), 'Captured native dtype/shape differs')
    g=inputs['global_states'][torch.tensor([r['global_row'] for r in rows])]
    need(torch.equal(g+cap['delta'].half(),cap['fused_global']), 'Actual FP16 cast-before-add arithmetic differs')
    codes=inputs['codebook'][torch.tensor([r['class_index'] for r in rows])]
    need(config['hardware']['matmul_allow_tf32'] is False, 'Oracle projection must use full FP32 multiplication')
    indices=torch.tensor([r['class_index'] for r in rows])
    amplitude=inputs['codebook'][indices,indices].double().unsqueeze(1)
    column_reference=(U[:,indices].T.double()*amplitude).float()
    column_error=(cap['delta']-column_reference).double()
    column_bound=1e-7+1e-6*column_reference.double().abs()
    need(bool((column_error.abs()<=column_bound).all()), 'Saved U/code column product does not reproduce captured delta')
    recomputed_delta=torch.nn.functional.linear(codes,U);delta_error=(recomputed_delta-cap['delta']).double()
    # General CPU FP32 GEMM/norm differences remain descriptive; the independent one-product oracle link is gated above.
    packet=torch.load(plan['native_model_file'],map_location='cpu',weights_only=True)
    norm,head=p.native_modules(torch,packet,plan['native_identity'],'cpu');del packet
    replays=[];norm_errors=[]
    with torch.no_grad():
        for offset in range(0,108,16):
            normalized=norm(cap['fused_global'][offset:offset+16]);replayed=head(normalized)
            replays+=replay_metric(torch,logits[offset:offset+16],replayed)
            norm_errors.append(float((normalized.float()-cap['normalized'][offset:offset+16].float()).abs().max()))
    replay=dict(passed=all(r['passed'] for r in replays),rows=replays,maximum_full_vocabulary_tv=max(r['full_vocabulary_tv'] for r in replays),
        argmax_matches=sum(r['argmax_exact'] for r in replays),cpu_norm_calls=7,cpu_head_calls=7,
        captured_fp16_cast_before_add_exact=True,oracle_column_product_passed=True,oracle_column_product_exact=bool(torch.equal(column_reference,cap['delta'])),
        oracle_column_product_max_abs=float(column_error.abs().max()),oracle_column_product_tolerance=dict(atol=1e-7,rtol=1e-6),
        delta_recompute_max_abs=float(delta_error.abs().max()),
        delta_recompute_relative_l2=float(delta_error.norm()/cap['delta'].double().norm().clamp_min(1e-30)),
        norm_recompute_max_abs=max(norm_errors),thresholds=dict(full_vocabulary_tv=.02,argmax_exact=True),
        no_full_vlm_equivalence_claim=True)
    save(out/'native_replay.json',replay)
    need(replay['passed'], 'Saved native head CPU consistency failed; raw report preserved')
    budget=resources(out,config['slurm_job_id'])
    from scripts.analyze_native_identity_join_uniform_geometry import analyze
    geometry=analyze(torch,out)
    need(geometry['completed'] is True, 'Uniform geometry incomplete')
    save(out/'uniform_geometry.json',geometry);save(out/'outcomes.json',outcomes)
    strata=[]
    for key in ('gold','question','n_frames','pair_id','contrast_id'):
        groups=defaultdict(list)
        for row in outcomes:groups[row[key]].append(row)
        for label,group in sorted(groups.items(),key=lambda x:str(x[0])):
            strata.append(dict(labels={key:label},metrics=dict(correct=sum(r['first_token_correct'] for r in group),contexts=len(group),
                all_correct=all(r['first_token_correct'] for r in group),mean_nll=sum(r['nll'] for r in group)/len(group))))
    save(out/'strata.json',strata)
    result=dict(passed=True,completed=True,phase='report',protocol=p.PROTOCOL,source_sha256=frozen,
        plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),run_directory=str(directory),run_summary_sha256=sha(directory/'summary.json'),
        conditional_report=conditional,first_token_fit=criterion,training=training,counters=counters,resources=budget,
        native_replay=replay,geometry_file=str(out/'uniform_geometry.json'),geometry_sha256=sha(out/'uniform_geometry.json'),
        outcomes_file=str(out/'outcomes.json'),outcomes_sha256=sha(out/'outcomes.json'),strata_file=str(out/'strata.json'),strata_sha256=sha(out/'strata.json'),
        no_generalization_or_deployable_method=True,no_whole_answer_or_reasoning_claim=True,
        reporting_source_sha256=reporting_sources(),precision_diagnosis_file=str(DIAGNOSIS),precision_diagnosis_sha256=DIAGNOSIS_SHA,
        original_report_remains_failed=True,nll_precision_file=str(out/'nll_precision.json'),nll_precision_sha256=sha(out/'nll_precision.json'))
    save(out/'analysis.json',result)
    (out/'REPORT.md').write_text('# Oracle readout diagnostic\n\n'
        f"Final supplied-answer-code first-token fit: {criterion['correct']}/108; {criterion['complete_families']}/18 complete families. "
        f"Registered first-token fit criterion: {'PASS' if criterion['passed'] else 'FAIL'}.\n\n"
        'This is an offline positive control supplying the correct answer code in both training and evaluation. '
        'It measures neither generalization nor a deployable aggregation method. No EOS/name-completion result is inferred. '
        'Uniform geometry is descriptive and does not identify the cause of failure.\n\n'
        f"Independent input, checkpoint,600update log and607norm/head-call audits passed. All108 CPU head replays passed the fixed consistency gates. "
        f"GPU cost: {budget['allocated_gpu_seconds']} allocated seconds; zero VLM/vision calls.\n")
    return dict(passed=True,completed=True,phase='report',protocol=p.PROTOCOL,source_sha256=frozen,
        analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),first_token_fit=criterion,
        resources=budget,no_generalization_or_deployable_method=True,reporting_source_sha256=reporting_sources(),
        precision_diagnosis_file=str(DIAGNOSIS),precision_diagnosis_sha256=DIAGNOSIS_SHA,original_report_remains_failed=True)


DIAGNOSIS=Path(__file__).resolve().parents[1]/'outputs/native_aggregation_vlm/identity_join_readout/rescore_diagnosis_443302/summary.json'
DIAGNOSIS_SHA='dbe0c1e367ccd24cc7188a4ad8298ee7f1c0be97fcdb0abc514aca7d87b6e5f9'
REPORT_OWN=('scripts/report_native_identity_join_readout_precision.py','slurm/native_identity_join_readout_precision_report.sbatch',
    'scripts/diagnose_native_identity_join_readout_rescore.py','slurm/native_identity_join_readout_rescore.sbatch')


def reporting_sources():
    p=driver()
    return {**p.sources(),**{n:sha(p.REPO/n) for n in REPORT_OWN}}


def verify_diagnosis(out):
    p=driver();bind(DIAGNOSIS,DIAGNOSIS_SHA);proof=read(DIAGNOSIS)
    need(proof['passed'] is True and proof['completed'] is True and proof['rows']==108 and proof['original_batches']==7
        and proof['observed_mismatches_confined_to_nll'] is True and proof['original_report_remains_failed'] is True
        and proof['metadata_mismatch_rows']==proof['argmax_mismatch_rows']==proof['correctness_mismatch_rows']==0
        and proof['original_nll_failure_rows']==108 and proof['no_acceptance_or_tolerance_change'] is True
        and proof['counters']==dict(vlm=0,vision=0,norm=0,head=0,optimizer=0), 'Rescore diagnosis does not support a precision-only repair')
    for n,h in proof['source_sha256'].items():bind(p.REPO/n,h);bind(DIAGNOSIS.parent/'source'/n.replace('/','_'),h)
    for k in ('rows','differences','input_bindings'):bind(proof[k+'_file'],proof[k+'_sha256'])
    for file,h in read(proof['input_bindings_file']).items():bind(file,h)
    bind(proof['original_failure_file'],proof['original_failure_sha256'])
    need(not (Path(proof['original_failure_file']).parent/'summary.json').exists(), 'Original failure replaced')
    save(out/'precision_release.json',dict(passed=True,diagnosis_file=str(DIAGNOSIS),diagnosis_sha256=DIAGNOSIS_SHA,
        change='NLL CPU FP32 reduction replaced by stable FP64 reduction of exact saved FP16 raw logits',
        original_atol=2e-6,original_rtol=2e-6,all_other_checks_unchanged=True,no_refit=True,original_report_remains_failed=True))


def main():
    import argparse,time
    p=driver();parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--runs',type=Path,nargs=1,required=True);args=parser.parse_args()
    p.native.require_slurm(gpu=False)
    need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'), 'CPU Slurm report only')
    out=p.OUT/f'report_precision_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    scientific=p.sources();frozen=reporting_sources();(out/'source').mkdir()
    for n,h in frozen.items():
        file=out/'source'/n.replace('/','_');file.write_bytes((p.REPO/n).read_bytes());bind(file,h)
    save(out/'source_hashes.json',frozen)
    save(out/'request.json',dict(plan=str(args.plan),runs=list(map(str,args.runs)),source_sha256=scientific,reporting_source_sha256=frozen))
    try:
        verify_diagnosis(out);result=report(args,out,scientific)
        need(reporting_sources()==frozen,'Reporting source changed during execution')
        result['elapsed_seconds']=time.perf_counter()-started;save(out/'summary.json',result)
        print(__import__('json').dumps(dict(passed=True,completed=True,directory=str(out))),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=scientific,reporting_source_sha256=frozen,
            elapsed_seconds=time.perf_counter()-started,partial_outputs_retained=True));raise


if __name__=='__main__':main()

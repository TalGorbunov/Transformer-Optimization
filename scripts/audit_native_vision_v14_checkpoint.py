"""All four fixed-final V14 models: native integrity and two-phase audit.

Core4590 and student8000 are fixed in advance; the one final dev sweep is
rescored descriptively and never selects a checkpoint. Native V13 computation
is reused unchanged; all cached/full numerical differences remain descriptive.
"""
from __future__ import annotations
import argparse
import gc
import math
import os
from pathlib import Path
import random
import re
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import profile_native_vision_v13_null as software
from scripts import native_vision_v13_runtime as runtime
from scripts import train_native_vision_v14 as training
from scripts.stage_native_vision_v6_teacher import MODEL,need,read,save,sha
native=software.native
OUT=REPO/'outputs/native_aggregation_vlm/v14/checkpoint_audit'
DATA=Path('/mnt/data/gabriele/gnn_transformer/v14_checkpoint_audit')
PROTOCOL='all_selected_v14_native_checkpoint_audit'
CONDITIONS=('centered','offset');SEEDS=(18,19)
CORE_STEP=4590;STUDENT_STEP=8000;RULE='fixed_final_core4590_student8000'
CALLS=dict(model=46,visual=26,language=46,norm=46,last_block=46)
OWN=('scripts/audit_native_vision_v14_checkpoint.py','slurm/native_vision_v14_checkpoint.sbatch',
     'slurm/native_vision_v14_checkpoint_check.sbatch','slurm/native_vision_v14_checkpoint_selftest.sbatch')


def sources():
    from scripts import report_native_vision_v14 as report
    return {**training.sources(),**report.sources(),**{name:sha(REPO/name) for name in OWN}}


def snapshot(out):
    (out/'source').mkdir();frozen=sources()
    for name,digest in frozen.items():
        p=out/'source'/name.replace('/','_');p.write_bytes((REPO/name).read_bytes());need(sha(p)==digest,'Source changed during snapshot')
    save(out/'source_hashes.json',frozen)
    (out/'INDEX.md').write_text('# V14 selected-checkpoint audit\n\n[Summary](summary.json) · [Plan](plan.json) · [Sources](source_hashes.json).\n')
    index=OUT/'INDEX.md'
    if not index.exists():index.write_text('# V14 selected-checkpoint audits\n\n')
    with index.open('a') as stream:stream.write(f'- [{out.name}]({out.name}/INDEX.md)\n')
    return frozen


def bind(artifacts,path,expected=None):
    path=Path(path).resolve();digest=sha(path)
    need(expected is None or digest==expected,'Bound artifact changed: '+str(path));artifacts[str(path)]=digest
    return str(path)


def strict_answer(tokenizer,ids,eos):
    need(isinstance(ids,list) and 1<=len(ids)<=4 and all(type(i) is int and i>=0 for i in ids),'Invalid generated token history')
    complete=ids[-1] in eos
    need(not any(i in eos for i in ids[:-1]) and (complete or len(ids)==4),'Native stop differs')
    bodyids=ids[:-1] if complete else ids;clean=not any(i in set(tokenizer.all_special_ids) for i in bodyids)
    body=tokenizer.decode(bodyids,skip_special_tokens=False)
    value=int(body.strip()) if clean and re.fullmatch('[0-9]+',body.strip()) else None
    return complete,clean,body,value


def validate_state(torch,saved,expected):
    from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
    from gnnformer.conditional_null_mean import ConditionalNullMean
    with torch.random.fork_rng(devices=[]):core=ParallelLocalAggregation();predictor=ConditionalNullMean()
    core.load_state_dict(saved['branch'],strict=True);predictor.load_state_dict(saved['predictor'],strict=True)
    need(sum(p.numel() for p in core.parameters())==1041600 and sum(p.numel() for p in predictor.parameters())==18624,
         'Core/predictor architecture differs')
    need(all(t.dtype==torch.float32 and bool(torch.isfinite(t).all()) for group in ('branch','predictor') for t in saved[group].values()),
         'Selected learned tensors must be finite FP32')
    need(native.object_sha(training.state_info(core,predictor))==expected,'Selected combined parameter digest differs')


def run(args):
    import torch,transformers
    from gnnformer.runtime import load_runtime,get_rope_index_fn,move_to_device
    from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
    from gnnformer.conditional_null_mean import ConditionalNullMean
    from gnnformer.parallel_local_learned_null import ParallelLocalLearnedNull
    from scripts.probe_native_vision_v2_prefix import fingerprint
    torch.set_num_threads(4);begin=time.perf_counter();job=os.environ['SLURM_JOB_ID']
    out=OUT/f'audit_{job}';out.mkdir(parents=True,exist_ok=False);frozen=snapshot(out)
    data=DATA/f'audit_{job}';data.mkdir(parents=True,exist_ok=False)
    (data/'INDEX.md').write_text('# V14 selected-checkpoint raw evidence\n\n[Head replays](head_replays.pt).\n')
    plan=verify(args.plan);cpu=read(Path(args.plan).parent/'summary.json')
    need(cpu['passed'] and cpu['selected_models']==4 and cpu['plan_sha256']==sha(args.plan),'Require exact completed CPU audit binding')
    (out/'plan.json').write_bytes(Path(args.plan).read_bytes())
    blob=torch.load(plan['prepared_file'],map_location='cpu',weights_only=True);need(blob['schema_version']==1,'Prepared schema differs')
    tick=time.perf_counter();loaded=load_runtime(str(MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda')
    model=loaded.model.eval().requires_grad_(False);norm=native.native_contract(model)
    need(fingerprint(loaded.processor,str(transformers.__version__))==plan['processor'] and software.runtime_identity()==plan['runtime'],
         'Loaded native runtime/processor differs')
    _,_,api=software.old.local.native_api(loaded.processor);need(api==plan['native_api'],'Installed native API differs')
    core=ParallelLocalAggregation().eval().requires_grad_(False).to(model.device)
    predictor=ConditionalNullMean().eval().requires_grad_(False).to(model.device);native.native_contract(model,core)
    torch.cuda.synchronize();load_seconds=time.perf_counter()-tick;torch.cuda.reset_peak_memory_stats()
    versions={k:p._version for k,p in model.named_parameters()};checks=[];comparisons=[];controllers=[];restores=[];records=[];counts={}
    try:
        with software.ancestor.NativeAudit(model,data) as audit:
            try:
                for case in plan['cases']:
                    cid=case['case_id'];n=case['n_frames'];width=case['prompt_width'];tokens=plan['forced_ids']
                    bundles=[blob['bundles'][f'{cid}_t{t}'] for t in range(3)]
                    for t,bundle in enumerate(bundles):
                        need(runtime.validate_bundle(bundle)['passed'] and bundle['metadata']['input_identity']==case['variants'][t]['input_identity']
                             and native.audit_layout(get_rope_index_fn(model),bundle)['metadata']==case['variants'][t]['layout'],'Software layout changed')
                    baseline=[];baseline_kv=[];native_full={};cache=None;audit.controller=None
                    for step,bundle in enumerate(bundles):
                        inputs,positions,layout=software.cached_input(model,bundle,cache,step,tokens,model.device)
                        audit.context=dict(label=f'{cid}__native__cached{step}',case_id=cid,n_frames=n,phase='cached',step=step,condition='native')
                        with torch.inference_mode():
                            output=model(**inputs,use_cache=True,logits_to_keep=1) if 'use_cache' not in inputs else model(**inputs,logits_to_keep=1)
                        cache=output.past_key_values;raw=audit.last_raw
                        need(torch.equal(raw['position_ids'],positions.cpu()) and torch.equal(raw['native_norm_input'],raw['fused_norm_input']),
                             'Bare native positions/readout differ')
                        baseline.append(raw);baseline_kv.append(software.snapshot_kv(cache))
                    del cache,output
                    for model_index,selected in enumerate(plan['runs']):
                        rid=selected['run_id'];saved=torch.load(selected['selected']['checkpoint'],map_location='cpu',weights_only=True)
                        core.load_state_dict(saved['branch'],strict=True);predictor.load_state_dict(saved['predictor'],strict=True)
                        need(native.object_sha(training.state_info(core,predictor))==selected['selected']['parameter_sha256'],
                             'Selected combined checkpoint did not restore exactly')
                        original=training.state_info(core,predictor);del saved
                        mode=selected['condition'];cache=None
                        control=ParallelLocalLearnedNull(norm,core,predictor,n_local_rows=n,mode=mode,
                            query_indices=[width-1],stream_positions=[width-1],capture=True)
                        with control:
                            audit.controller=control
                            for step,bundle in enumerate(bundles):
                                query=[width-1] if step==0 else [0];stream=[width+step-1];control.configure_queries(query,stream)
                                inputs,positions,layout=software.cached_input(model,bundle,cache,step,tokens,model.device)
                                audit.context=dict(label=f'{cid}__{rid}__cached{step}',case_id=cid,n_frames=n,phase='cached',step=step,
                                                   condition=mode,mode=mode,seed=selected['seed'],run_id=rid)
                                with torch.inference_mode():
                                    output=model(**inputs,use_cache=True,logits_to_keep=1) if 'use_cache' not in inputs else model(**inputs,logits_to_keep=1)
                                cache=output.past_key_values;raw=audit.last_raw
                                need(torch.equal(raw['position_ids'],positions.cpu()) and torch.equal(raw['native_norm_input'],baseline[step]['native_norm_input']),
                                     'Selected adapter changed fixed-prefix native hidden/positions')
                                checked=software.verify_fusion(raw,control,query,stream)
                                checked.update(label=audit.context['label'],run_id=rid,native_hidden_exact=True,
                                               all_layer_kv_exact=software.ancestor.all_kv_equal(cache,baseline_kv[step]));checks.append(checked)
                                if step>0:
                                    history=list(range(width-1,width+step));control.configure_queries(history,history)
                                    audit.context=dict(label=f'{cid}__{rid}__full{step}',case_id=cid,n_frames=n,phase='full',step=step,
                                                       condition=mode,mode=mode,seed=selected['seed'],run_id=rid)
                                    rope=software.cpu_copy(model.model.rope_deltas)
                                    try:
                                        with torch.inference_mode():full_output=model(**move_to_device(bundle['inputs'],model.device),use_cache=False,logits_to_keep=1)
                                    finally:model.model.rope_deltas=rope.to(model.device)
                                    full=audit.last_raw
                                    need(full_output.past_key_values is None and torch.equal(full['position_ids'],layout['position_ids']),
                                         'Full-prefix native positions/cache differ')
                                    if model_index==0:native_full[step]=full['native_norm_input']
                                    else:need(torch.equal(native_full[step],full['native_norm_input']),'Selected model changed full-prefix native reads')
                                    checked=software.verify_fusion(full,control,history,history)
                                    checked.update(label=audit.context['label'],run_id=rid,native_hidden_exact=True,all_layer_kv_exact=None);checks.append(checked)
                                    for row in range(n+1):
                                        metric=software.old.metric(torch,raw['native_logits'][row,-1],full['native_logits'][row,-1])
                                        metric.update(case_id=cid,n_frames=n,run_id=rid,mode=mode,seed=selected['seed'],step=step,row_index=row,
                                                      role='local' if row<n else 'global',binding=False);comparisons.append(metric)
                                    del full,full_output
                            need(control.calls==5,'Selected controller invocation count differs')
                            controllers.append(dict(case_id=cid,run_id=rid,calls=control.calls,passed=True))
                        audit.controller=None;need(not control.active,'Selected controller hook leaked')
                        need(training.state_info(core,predictor)==original,'Audit changed learned parameters')
                        restores.append(dict(case_id=cid,run_id=rid,parameter_sha256=native.object_sha(original),passed=True))
                        del cache,output,raw;gc.collect()
                    del baseline,baseline_kv,native_full,bundles
                need(audit.counts==CALLS,'Selected fixed-forward inventory differs')
            finally:
                records=audit.records;counts=dict(audit.counts)
                save(out/'forwards.json',records);save(out/'calls.json',counts)
                save(out/'structural_checks.json',dict(checks=checks,controller_counts=controllers,selected_state_restores=restores))
                save(out/'cached_full_comparisons.json',comparisons)
        need(len(records)==46 and len(checks)==40 and len(controllers)==len(restores)==8 and len(comparisons)==656
             and sum(r['all_layer_kv_exact'] is not None for r in checks)==24,'Selected structural coverage differs')
        kv=[r['all_layer_kv_exact'] for r in checks if r['all_layer_kv_exact'] is not None]
        need(len(kv)==24 and all(type(flags) is list and len(flags)==28 and all(type(flag) is bool and flag is True for flag in flags) for flags in kv),
             'Require exactly24 complete lists of28 literal successful KV checks')
        metrics,raw_metrics=software.replay_heads(torch,model,records)
        need(len(metrics)==46,'Exactly one extra same-captured native head per model call required')
        torch.save(raw_metrics,data/'head_replays.pt');save(out/'head_replay_metrics.json',metrics)
    finally:save(out/'actual_calls.json',dict(calls=counts,standalone_head_replays=len(metrics) if 'metrics' in locals() else 0))
    need(versions=={k:p._version for k,p in model.named_parameters()}
         and not any(p.requires_grad or p.grad is not None for p in model.parameters()),'Frozen backbone changed')
    failures=[r for r in metrics if not r['numerical_rule_passed']];descriptive=[r for r in comparisons if not r['numerical_rule_passed']]
    models=[]
    for selected in plan['runs']:
        rid=selected['run_id'];rmetrics=[m for m in metrics if f'__{rid}__' in m['label']];cm=[m for m in comparisons if m['run_id']==rid]
        need(len(rmetrics)==10 and len(cm)==164,'Per-selected-model replay coverage differs')
        models.append(dict(run_id=rid,arm='parallel',condition=selected['condition'],seed=selected['seed'],
            selected_step=selected['selected']['step'],student_step=selected['selected']['student_step'],selected_student_step=selected['selected']['student_step'],checkpoint=selected['selected']['checkpoint'],checkpoint_sha256=selected['selected']['checkpoint_sha256'],
            parameter_sha256=selected['selected']['parameter_sha256'],selected_parameter_sha256=selected['selected']['parameter_sha256'],replay_gate_passed=all(m['numerical_rule_passed'] for m in rmetrics),
            cache_full_numeric_passed=all(m['numerical_rule_passed'] for m in cm),head_replays=10,cache_full_rows=164,
            computational_integrity_passed=True,fixed_prefix_kv_checks=6,fixed_core_through_student=True,one_dev_only=True))
    paths={name:out/name for name in ('forwards.json','calls.json','structural_checks.json','cached_full_comparisons.json','head_replay_metrics.json','actual_calls.json')}
    paths['head_replays.pt']=data/'head_replays.pt'
    artifacts={name:dict(path=str(path),sha256=sha(path)) for name,path in paths.items()}
    summary=dict(schema_version=1,protocol=PROTOCOL,completed=True,passed=not failures,computational_integrity_passed=True,
        replay_gate_passed=not failures,native_head_replay_passed=not failures,native_replay_gate_passed=not failures,
        cache_full_numeric_passed=not descriptive,strict_cache_numerical_gate_passed=not descriptive,
        strict_numerical_gates_passed=not failures and not descriptive,cached_full_numeric_is_descriptive=True,
        replay_failures=failures,cache_full_failures=descriptive,models=models,runs=models,plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),
        source_sha256=frozen,model=plan['model'],runtime=plan['runtime'],processor=plan['processor'],native_api=plan['native_api'],
        calls=counts,native_head_replay_calls=46,head_replay_calls=46,cached_full_row_comparisons=656,cache_full_rows=656,
        fixed_prefix_kv_checks=24,fixed_prefix_kv_layers=28,selected_fusion_checks=40,ordinary_generations=0,extra_last_block_forwards=0,
        artifacts=artifacts,files=artifacts,backend=plan['backend'],original_mixed_failure=plan['original_mixed_failure'],original_mixed_numerical_gate_passed=False,
        frozen_backbone_unchanged=True,selected_parameters_unchanged=True,fixed_core_through_student=True,one_dev_only=True,no_fitting=True,no_efficacy_scoring=True,
        native_dtypes=dict(norm=str(norm.weight.dtype),lm_head=str(model.lm_head.weight.dtype)),
        hardware=dict(gpu=torch.cuda.get_device_name(),total_memory_bytes=torch.cuda.get_device_properties(0).total_memory),
        peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(),peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(),
        load_seconds=load_seconds,seconds=time.perf_counter()-begin,slurm_job_id=job)
    save(out/'summary.json',summary)
    print(dict(directory=str(out),passed=summary['passed'],calls=counts,binding_failures=len(failures),descriptive_failures=len(descriptive)),flush=True)
    need(summary['passed'],'Captured native head replay failed; all observations preserved')


def audit_dev(torch,processor,entry,directory,config,dev_samples,artifacts):
    path=directory/'dev_final.json'
    need(Path(entry['dev_file']).resolve()==path,'Noncanonical dev file')
    bind(artifacts,path,entry['dev_sha256']);dev=read(path)
    need(dev['n']==len(dev['rows'])==64 and [r['sid'] for r in dev['rows']]==[r['sid'] for r in dev_samples]
         and dev['label']=='dev_final' and dev['raw_dtype']=='torch.float32','Development coverage/order/dtype differs')
    rawpath=Path(config['data_directory'])/(dev['label']+'_raw.pt')
    need(Path(dev['raw_file']).resolve()==rawpath,'Noncanonical raw dev archive');bind(artifacts,rawpath,dev['raw_sha256'])
    raw=torch.load(rawpath,map_location='cpu',weights_only=True)
    need(raw['schema_version']==1 and len(raw['raw_logits'])==64,'Incomplete raw development archive')
    exact=0;nlls=[]
    for index,(row,sample,logits) in enumerate(zip(dev['rows'],dev_samples,raw['raw_logits'])):
        ids=row['generated_ids'];complete,clean,body,value=strict_answer(processor.tokenizer,ids,training.POLICY['native_eos'])
        need(logits.dtype==torch.float32 and logits.ndim==2 and logits.shape[0]==len(ids)
             and bool(torch.isfinite(logits).all()) and torch.equal(logits,logits.half().float())
             and logits.argmax(-1).tolist()==ids,'Raw native first/continuation argmax differs')
        target=native.encode_target(processor.tokenizer,sample['gold'])
        nll=float(torch.logsumexp(logits[0],-1)-logits[0,target[0]])
        correct=complete and value==sample['gold']
        need(row['raw_index']==index and row['cell']=='dev_N16' and row['n_frames']==16
             and row['gold']==sample['gold'] and row['content_sha256']==sample['content_sha256']
             and row['completed']==complete and row['truncated']==(not complete)
             and row['answer_body']==body and row['no_nonterminal_special_tokens']==clean
             and row['prediction']==value and row['parseable']==(value is not None)
             and row['parsed_count_correct']==(value==sample['gold']) and row['exact']==correct
             and row['raw_text']==processor.tokenizer.decode(ids,skip_special_tokens=False)
             and row['text']==processor.tokenizer.decode(ids,skip_special_tokens=True)
             and math.isclose(nll,row['first_token_nll'],abs_tol=2e-5,rel_tol=2e-6),
             'Independent strict native development rescoring differs')
        exact+=correct;nlls.append(nll)
    # The original FP32 logsumexp can differ slightly across CPU/GPU kernels;
    # validate every value and preserve the recorded descriptive GPU statistic.
    recorded=sum(r['first_token_nll'] for r in dev['rows'])/64
    need(exact==entry['exact_count']==dev['exact_count'] and recorded==entry['nll']==dev['first_token_nll']
         and math.isclose(sum(nlls)/64,recorded,abs_tol=2e-5,rel_tol=2e-6),'Development totals differ')
    return dict(step=entry['step'],examples=64,exact_count=exact,recorded_first_token_nll=recorded,
                recomputed_first_token_nll=sum(nlls)/64,raw_argmax_and_strict_EOS_rescored=True)

def verify_source_check(directory):
    directory=Path(directory).resolve();summary=read(directory/'summary.json')
    need(directory.parent==OUT and directory.name.startswith('selftest_') and summary['protocol']==PROTOCOL
         and summary['passed'] and summary['tests_passed'] and summary['no_model_loaded'] and summary['source_sha256']==sources(),
         'Require exact pre-main audit source self-test')
    need(read(directory/'source_hashes.json')==summary['source_sha256'],'Self-test source ledger differs')
    for name,digest in summary['source_sha256'].items():need(sha(directory/'source'/name.replace('/','_'))==digest,'Self-test source copy differs')
    return summary


def final_entry(selection):
    need(selection['rule']==RULE and selection['dev_descriptive_only'] is True
         and len(selection['development'])==1 and selection['development'][0]==selection['selected'],
         'V14 permits one descriptive final dev entry and no checkpoint selection')
    value=selection['selected']
    need(value['step']==CORE_STEP and value['student_step']==STUDENT_STEP
         and type(value['exact_count']) is int and 0<=value['exact_count']<=64
         and math.isfinite(value['nll']) and value['nll']>=0,'Invalid fixed-final checkpoint metadata')
    return value


def audit_phase(torch,directory,config,summary,plan,cache,artifacts):
    """Bind phase endpoints, immutable regression targets, and both update streams."""
    need(config['policy']==training.POLICY and config['condition'] in CONDITIONS and config['seed'] in SEEDS,
         'Registered V14 condition/seed/policy differs')
    pairing=read(plan['pairing_file']);order=training.presentation_order(pairing['pairs'],config['seed'])
    student_order=training.student_order(cache['auxiliary_groups'],config['seed'])
    for name,rows,key in [('presentations',order,'order_sha256'),('student_presentations',student_order,'student_order_sha256')]:
        path=directory/(name+'.json');bind(artifacts,path)
        need(read(path)==rows and native.object_sha(rows)==config[key]==plan[key][str(config['seed'])],
             'Registered persistent training stream differs: '+name)
    hist=read(directory/'training.json');need(len(hist)==CORE_STEP,'Core phase does not contain4590 updates')
    lookup=training.group_lookup(cache)
    for step,row in enumerate(hist,1):
        batch=order[(step-1)*16:step*16];sids=[r['sid'] for r in batch]
        need(row['step']==step and row['sids']==sids and row['target_ids']==[t for sid in sids for t in cache['scenes'][sid]['target_ids']]
             and row['reference_group_ids']==training.position_groups(cache,sids,lookup)
             and row['pair_ids']==[r['pair_id'] for r in batch[::2]] and row['epochs']==[r['epoch'] for r in batch[::2]]
             and math.isclose(row['lr'],training.lr(step),rel_tol=1e-10,abs_tol=1e-12),'Core target/reference/update order differs')
        need(all(math.isfinite(row[k]) and row[k]>=0 for k in ('loss','ce_loss','consistency_loss','path_loss','gradient_norm'))
             and row['consistency_coefficient']==1. and math.isclose(row['loss'],row['ce_loss']+row['consistency_loss'],rel_tol=3e-6,abs_tol=3e-6)
             and row['clipped']==(row['gradient_norm']>1.),'Core-only objective or clipping differs')
    need(sum(len(row['target_ids']) for row in hist)==177120,'Ordinary native target-position count differs')
    bind(artifacts,directory/'training.json')
    freeze=read(directory/'core_freeze.json');bind(artifacts,directory/'core_freeze.json')
    need(freeze['schema_version']==1 and freeze['core_step']==CORE_STEP and freeze['student_step']==STUDENT_STEP
         and freeze['core_unchanged_after_distillation'] is True and freeze['core_unchanged_after_diagnostics'] is True
         and freeze['fixed_query_target_denominator_and_archive_unchanged'] is True
         and summary['core_freeze_passed'] is True and summary['reference_gradient_audit_passed'] is True
         and summary['checkpoint_roundtrip_passed'] is True,
         'Core freeze boundary invariants missing')
    need(Path(freeze['core_checkpoint']).resolve()==training.CKPT/config['run_id']/'core_frozen.pt','Noncanonical frozen core')
    bind(artifacts,freeze['core_checkpoint'],freeze['core_checkpoint_sha256'])
    phase=read(directory/'core_phase.json');bind(artifacts,directory/'core_phase.json')
    need(phase['completed'] is True and phase['step']==CORE_STEP
         and all(phase[k]==freeze[k] for k in ('core_checkpoint','core_checkpoint_sha256','core_parameter_sha256'))
         and phase['training_sha256']==sha(directory/'training.json'),'Recorded core endpoint differs')
    refs=read(directory/'reference_gradient_audit.json');bind(artifacts,directory/'reference_gradient_audit.json',phase['reference_gradient_audit_sha256'])
    need([r['step'] for r in refs]==[1,2,32] and all(all(r[k] is True for k in
         ('passed','mean_and_messages_live','optimizer_buffers_untouched','cached_native_features_frozen')) for r in refs),
         'Reference gradient route evidence is incomplete')
    for r in refs:
        norms=r['core_reference_gradients']
        need(set(norms)=={'reference_message','reference_query_weight','reference_local_weight'}
             and r['zero_up_expected']==(r['step']==1) and all(math.isfinite(v) and (v==0 if r['step']==1 else v>0) for v in norms.values()),
             'Live reference query/message gradients were not exercised')
    for label in ('training','distillation','first_gradients','student_first_gradients','reference_gradient_audit','core_freeze'):
        need(Path(summary[label+'_file']).resolve()==directory/(label+'.json'),'Noncanonical phase artifact: '+label)
        bind(artifacts,summary[label+'_file'],summary[label+'_sha256'])
    conversion=summary['conversion_diagnostics'];bind(artifacts,conversion['file'],conversion['sha256']);bind(artifacts,conversion['raw_file'],conversion['raw_sha256'])
    conversion_value=read(conversion['file'])
    need(conversion['groups']==972 and conversion['ordinary_positions']==4266 and conversion['no_accuracy'] is True
         and conversion_value['groups_count']==972 and conversion_value['unique_scenes']==1782
         and conversion_value['ordinary_positions_count']==4266
         and all(conversion_value[k] is True for k in ('all_groups_and_positions_retained','no_VLM_or_head_calls','no_accuracy','no_fit','no_selection')),
         'Fixed complete training-only conversion diagnostics missing')
    core_blob=torch.load(freeze['core_checkpoint'],map_location='cpu',weights_only=True)
    need(set(core_blob)=={'branch','step','config'} and core_blob['step']==CORE_STEP and core_blob['config']==config,
         'Frozen core payload/step/config differs')
    core_identity={k:native.tensor_info(v) for k,v in core_blob['branch'].items()}
    need(native.object_sha(core_identity)==freeze['core_parameter_sha256'],'Frozen core parameter digest differs')
    from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
    from gnnformer.conditional_null_mean import ConditionalNullMean
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config['seed']);core=ParallelLocalAggregation()
    need(training.v7.state_info(core)==config['initialized']['branch']
         and native.object_sha(config['initialized'])==config['initialized_sha256'],'Registered core initialization differs')
    core.load_state_dict(core_blob['branch'],strict=True)
    need(all(t.dtype==torch.float32 and bool(torch.isfinite(t).all()) for t in core_blob['branch'].values()),'Frozen core dtype/values differ')
    for label in ('target','target_metadata'):bind(artifacts,freeze[label+'_file'],freeze[label+'_sha256'])
    table=torch.load(freeze['target_file'],map_location='cpu',weights_only=True);meta=read(freeze['target_metadata_file'])
    gids=sorted(cache['auxiliary_groups']);need(table['schema_version']==1 and table['group_ids']==gids and len(gids)==972
         and table['core_parameter_sha256']==meta['core_parameter_sha256']==freeze['core_parameter_sha256']
         and table['cache_sha256']==meta['cache_sha256']==config['cache_binding']['sha256']
         and meta['group_ids']==gids and meta['groups']==972 and meta['occurrences']==24 and meta['all_targets_frozen'] is True
         and meta['file']==freeze['target_file'] and meta['sha256']==freeze['target_sha256'],'Fixed target inventory/provenance differs')
    tensors={k:native.tensor_info(v) for k,v in table.items() if isinstance(v,torch.Tensor)}
    need(tensors==meta['tensors'] and summary['target_metadata']==meta,'Fixed target tensor identity or metadata copy differs')
    shapes={'query':(972,96),'mean':(972,96),'denominator':(972,),'messages':(972,24,96),
            'projected_messages':(972,24,96),'projected_population_variance':(972,)}
    for name,shape in shapes.items():
        value=table[name];dtype=torch.float64 if name=='projected_population_variance' else torch.float32
        need(value.shape==shape and value.dtype==dtype and bool(torch.isfinite(value).all()) and not value.requires_grad,
             'Frozen target shape/dtype/finiteness differs: '+name)
    need(bool((table['denominator']>0).all()),'Invalid fixed student normalization')
    torch.testing.assert_close(table['mean'],table['messages'].double().mean(1).float(),atol=1e-8,rtol=1e-6)
    torch.testing.assert_close(table['denominator'],table['messages'].square().mean(dim=(1,2))+1e-6,atol=1e-8,rtol=1e-6)
    p=table['projected_messages'].double();variance=(p-p.mean(1,keepdim=True)).square().sum(dim=(1,2))/24
    torch.testing.assert_close(table['projected_population_variance'],variance,atol=1e-10,rtol=1e-9)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config['seed']+20261110);student=ConditionalNullMean()
    initial=native.object_sha(training.v7.state_info(student))
    need(freeze['student_initialization_seed']==config['student_initialization_seed']==config['seed']+20261110
         and freeze['student_initialized_sha256']==config['student_initialized_sha256']==initial
         and training.v7.state_info(student)==config['initialized']['predictor'],
         'Independent student initialization differs')
    distillation=read(directory/'distillation.json');bind(artifacts,directory/'distillation.json')
    need(len(distillation)==STUDENT_STEP,'Student phase does not contain8000 updates')
    group_index={gid:i for i,gid in enumerate(gids)}
    for step,row in enumerate(distillation,1):
        batch=student_order[(step-1)*64:step*64];ids=[r['group_id'] for r in batch]
        expected_den=table['denominator'][[group_index[g] for g in ids]].tolist()
        need(row['step']==step and row['group_ids']==ids and row['cycles']==[r['cycle'] for r in batch]
             and math.isclose(row['lr'],training.student_lr(step),rel_tol=1e-10,abs_tol=1e-12)
             and row['per_group_denominator']==expected_den and len(row['per_group_mse'])==len(row['per_group_loss'])==64
             and all(row[k] is True for k in ('query_detached','target_detached','denominator_detached')),
             'Student update order/fixed target contract differs')
        mse=torch.tensor(row['per_group_mse'],dtype=torch.float32);den=torch.tensor(expected_den,dtype=torch.float32)
        values=torch.tensor(row['per_group_loss'],dtype=torch.float32)
        need(bool(torch.isfinite(mse).all()) and bool((mse>=0).all()) and math.isfinite(row['gradient_norm']) and row['gradient_norm']>=0
             and row['clipped']==(row['gradient_norm']>1.),'Nonfinite student loss or clipping differs')
        torch.testing.assert_close(values,mse/den,atol=1e-8,rtol=1e-6)
        need(math.isclose(row['loss'],float(values.mean()),rel_tol=3e-6,abs_tol=1e-7),'Student objective is not fixed normalized MSE')
    return freeze,core_blob['branch'],dict(core_updates=4590,student_updates=8000,scene_presentations=73440,
        native_target_positions=177120,student_presentations=512000,frozen_groups=972,
        fixed_targets_and_normalization_verified=True,core_endpoint_and_student_init_verified=True,
        no_cross_device_linear_projection_identity_claim=True,
        live_reference_gradient_records_verified=True,conversion_artifacts_bound=True,
        detailed_conversion_tensor_audit_delegated_to_independent_main_report=True)


def verify_run(torch,processor,directory,plans,caches):
    directory=Path(directory).resolve();need(directory.parent==training.OUT,'Require canonical V14 main directory')
    config=read(directory/'config.json');summary=read(directory/'summary.json');rid=config['run_id'];artifacts={}
    need(directory.name==rid and rid.startswith(f'run_{config["condition"]}_s{config["seed"]}_') and config['arm']=='parallel'
         and config['profile'] is False and all(summary[k]==v for k,v in config.items())
         and all(summary[k] is True for k in ('passed','completed','computational_integrity_passed'))
         and summary['steps']==CORE_STEP and summary['student_steps']==STUDENT_STEP and summary['native_test_count']==272,
         'Incomplete V14 fixed-final main run')
    path=bind(artifacts,config['plan_file'],config['plan_sha256'])
    if path not in plans:plans[path]=training.verify_plan(path)
    plan=plans[path];need(plan['source_sha256']==config['source_sha256'] and config['policy']==plan['policy'],'Main source/policy differs')
    cpu=read(Path(path).parent/'summary.json');need(cpu['passed'] and cpu['plan_sha256']==config['plan_sha256'],'Training CPU gate missing')
    bind(artifacts,Path(path).parent/'summary.json');bind(artifacts,Path(path).with_suffix('.sha256'))
    for name,digest in config['source_sha256'].items():
        need(sha(REPO/name)==digest,'Frozen main source changed');bind(artifacts,directory/'code'/name.replace('/','_'),digest)
    artifacts.update(plan['artifact_bindings']);bind(artifacts,plan['pairing_file'],plan['pairing_sha256'])
    need(config['cache_binding']==plan['cache_binding'] and config['pairing_file']==plan['pairing_file']
         and config['pairing_sha256']==plan['pairing_sha256'] and config['prior_result']==plan['prior_result']
         and config['null_groups_sha256']==plan['null_groups_sha256'],'Run/cache/pairing binding differs')
    cachepath=bind(artifacts,config['cache_binding']['file'],config['cache_binding']['sha256'])
    if cachepath not in caches:caches[cachepath]=training.cache_binding(cachepath)
    cache=caches[cachepath]
    for key in ('model','runtime','processor','native_dtypes'):need(config[key]==cache[key],'Native feature identity differs: '+key)
    release=training.verify_release(config['main_release']['file'],config['plan_file'],plan)
    need(release==config['main_release'],'Main release copy changed');bind(artifacts,release['file'],release['sha256'])
    release_data=read(release['file']);bind(artifacts,release_data['data_release']['file'],release_data['data_release']['sha256'])
    for item in release_data['profiles'].values():bind(artifacts,Path(item['directory'])/'summary.json',item['summary_sha256'])
    selection=read(directory/'selection.json');entry=final_entry(selection)
    need(summary['selected']==entry and summary['development']==[entry],'Fixed-final dev metadata differs')
    devsamples=read(plan['train_manifest'])['splits']['dev_N16']['samples']
    devcheck=audit_dev(torch,processor,entry,directory,config,devsamples,artifacts)
    need(Path(entry['checkpoint']).resolve()==training.CKPT/rid/'final.pt','Noncanonical fixed-final checkpoint')
    bind(artifacts,entry['checkpoint'],entry['checkpoint_sha256'])
    saved=torch.load(entry['checkpoint'],map_location='cpu',weights_only=True)
    need(set(saved)=={'branch','predictor','step','student_step','config'} and saved['config']==config
         and saved['step']==CORE_STEP and saved['student_step']==STUDENT_STEP,'Final payload/phase endpoints differ')
    validate_state(torch,saved,entry['parameter_sha256'])
    freeze,frozen_branch,phases=audit_phase(torch,directory,config,summary,plan,cache,artifacts)
    need(set(frozen_branch)==set(saved['branch']) and all(torch.equal(v,saved['branch'][k]) for k,v in frozen_branch.items()),
         'Student phase changed the final core tensors')
    for name in ('config.json','summary.json','selection.json'):bind(artifacts,directory/name)
    bind(artifacts,summary['test_file'],summary['test_sha256']);test=read(summary['test_file'])
    need(test['n']==len(test['rows'])==272,'Completed test coverage missing');bind(artifacts,test['raw_file'],test['raw_sha256'])
    return dict(run_id=rid,condition=config['condition'],seed=config['seed'],arm='parallel',directory=str(directory),config=config,
                selected=entry,development_checks=[devcheck],phase_checks=phases,core_freeze=freeze,artifact_sha256=artifacts,
                native_profile=plan['native_profile'])


def self_test():
    import torch
    torch.set_num_threads(4)
    e=dict(step=4590,student_step=8000,exact_count=10,nll=1.)
    good=dict(rule=RULE,dev_descriptive_only=True,development=[e],selected=e)
    need(final_entry(good)==e,'Fixed final entry rejected')
    for changed in (dict(good,development=[e,e]),dict(good,dev_descriptive_only=False),dict(good,rule='best_dev'),
                    dict(good,development=[dict(e,student_step=7999)],selected=dict(e,student_step=7999))):
        try:final_entry(changed)
        except ValueError:pass
        else:raise AssertionError('Adaptive/incomplete final selection accepted')
    class Tokenizer:
        all_special_ids=[9,10]
        def decode(self,ids,skip_special_tokens=False):return ''.join({0:'0',1:'1',2:'6',8:'x',9:'',10:''}[i] for i in ids)
    tok=Tokenizer();need(strict_answer(tok,[1,2,9],[9])==(True,True,'16',16),'Native multi-token answer differs')
    need(strict_answer(tok,[1,10,9],[9])==(True,False,'1',None),'Nonterminal special token accepted')
    for ids in ([1],[9,1,9],[],[1,2,0,0,9]):
        try:strict_answer(tok,ids,[9])
        except ValueError:pass
        else:raise AssertionError('Wrong native termination accepted')
    a=torch.tensor([1.,2.,3.]);need(software.old.metric(torch,a,a+7.)['numerical_rule_passed']
        and not software.old.metric(torch,a,a.flip(0))['numerical_rule_passed'],'Numerical classification differs')
    from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
    from gnnformer.conditional_null_mean import ConditionalNullMean
    rng=torch.random.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        core=ParallelLocalAggregation();student=ConditionalNullMean();saved=dict(branch=core.state_dict(),predictor=student.state_dict())
        identity=native.object_sha(training.state_info(core,student))
    validate_state(torch,saved,identity);need(torch.equal(rng,torch.random.get_rng_state()),'Validation changes caller RNG')
    saved['predictor']['fc2.bias']=saved['predictor']['fc2.bias']+1.
    try:validate_state(torch,saved,identity)
    except ValueError:pass
    else:raise AssertionError('Wrong final student identity accepted')
    valid=[[True]*28 for _ in range(24)]
    def check_kv(rows):return len(rows)==24 and all(type(x) is list and len(x)==28 and all(type(v) is bool and v is True for v in x) for x in rows)
    need(check_kv(valid) and not check_kv([True]*24) and not check_kv([[1]*28]*24)
         and not check_kv([[True]*27]*24),'Literal per-layer KV schema differs')
    need(2*(3+4*5)==46 and 2*(1+4*3)==26 and 4*2*(17+65)==656,'Registered native audit counts differ')
    return dict(passed=True,tests_passed=True,tests=['fixed_core4590_student8000_only','one_descriptive_dev_never_selects',
        'strict_native_integer_EOS','combined_core_student_identity','private_validation_RNG','literal24_by28_KV_schema','exact_native_inventory'])


def unit():
    out=OUT/f'selftest_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False);frozen=snapshot(out)
    tests=self_test();need(sources()==frozen,'Sources changed during self-test')
    save(out/'summary.json',dict(tests,protocol=PROTOCOL,source_sha256=frozen,no_model_loaded=True,
        scope='Synthetic final-checkpoint audit software; no trained checkpoint or outcome opened',slurm_job_id=os.environ['SLURM_JOB_ID']))
    print(dict(passed=True,directory=str(out)),flush=True)


def check(args):
    import torch,transformers
    from transformers import AutoProcessor
    from scripts.probe_native_vision_v2_prefix import fingerprint
    torch.set_num_threads(4);begin=time.perf_counter();job=os.environ['SLURM_JOB_ID']
    out=OUT/f'check_{job}';out.mkdir(parents=True,exist_ok=False);frozen=snapshot(out)
    verify_source_check(args.source_check);tests=self_test()
    processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,use_fast=False)
    plans={};caches={};runs=[verify_run(torch,processor,p,plans,caches) for p in args.runs];runs.sort(key=lambda r:(r['condition'],r['seed']))
    need(len(runs)==4 and {(r['condition'],r['seed']) for r in runs}=={(c,s) for c in CONDITIONS for s in SEEDS},
         'Require all four fixed-final models exactly once')
    configs=[r['config'] for r in runs]
    for key in ('policy','plan_file','plan_sha256','source_sha256','cache_binding','model','runtime','processor','native_dtypes','main_release'):
        need(all(c[key]==configs[0][key] for c in configs),'Unmatched main identity: '+key)
    for seed in SEEDS:
        pair=[r for r in runs if r['seed']==seed]
        for key in ('initialized','order_sha256','student_order_sha256'):
            need(pair[0]['config'][key]==pair[1]['config'][key],'Paired initial state/order differs: '+key)
        need(pair[0]['core_freeze']['student_initialized_sha256']==pair[1]['core_freeze']['student_initialized_sha256'],
             'Matched private student initializations differ')
    profile_binding=runs[0]['native_profile'];profile=software.verify_profile(profile_binding['file'])
    need(sha(profile_binding['file'])==profile_binding['sha256'] and all(r['native_profile']==profile_binding for r in runs),
         'Reused native software binding differs')
    parent=software.verify(profile['plan_file'])
    need(parent['model']==configs[0]['model'] and parent['runtime']==configs[0]['runtime']
         and parent['processor']==configs[0]['processor']==fingerprint(processor,str(transformers.__version__)),
         'Software/main model, runtime or processor differs')
    forced=[]
    for word in software.FORCED:
        ids=processor.tokenizer(word,add_special_tokens=False)['input_ids']
        need(len(ids)==1 and ids[0] not in processor.tokenizer.all_special_ids,'Fixed ordinary prefix differs');forced+=ids
    need(forced==parent['forced_ids'],'Reused forced prefix IDs differ')
    artifacts={}
    for run in runs:artifacts.update(run['artifact_sha256'])
    bind(artifacts,profile_binding['file'],profile_binding['sha256']);bind(artifacts,profile['plan_file'],profile['plan_sha256'])
    bind(artifacts,parent['prepared_file'],parent['prepared_sha256']);bind(artifacts,Path(args.source_check)/'summary.json')
    gate=read(configs[0]['main_release']['file'])['selected_audit_freeze']
    need(Path(gate['directory']).resolve()==Path(args.source_check).resolve() and gate['summary_sha256']==sha(Path(args.source_check)/'summary.json'),
         'Mains were released with a different final-checkpoint audit source')
    plan=dict(schema_version=1,protocol=PROTOCOL,source_sha256=frozen,source_check=str(Path(args.source_check).resolve()),
        source_check_sha256=sha(Path(args.source_check)/'summary.json'),tests=tests,runs=runs,artifact_sha256=artifacts,
        policy=training.POLICY,software_profile=profile_binding,parent_plan_file=profile['plan_file'],parent_plan_sha256=profile['plan_sha256'],
        prepared_file=parent['prepared_file'],prepared_sha256=parent['prepared_sha256'],cases=parent['cases'],
        model=parent['model'],runtime=parent['runtime'],processor=parent['processor'],native_api=parent['native_api'],
        forced_text=list(software.FORCED),forced_ids=forced,calls=CALLS,head_replay_calls=46,cache_full_rows=656,
        fixed_prefix_kv_checks=24,fixed_prefix_kv_layers=28,selected_fusion_checks=40,ordinary_generations=0,extra_last_block_forwards=0,
        backend='native_bitsandbytes_dispatch_unmodified',original_mixed_failure=parent['original_mixed_failure'],
        original_mixed_numerical_gate_passed=False,no_fitting=True,no_efficacy_scoring=True,
        fixed_core_through_student=True,one_dev_only=True,
        scope='All fixed-final models; head TV<=.02/top1 binding; every cached/full numerical error descriptive',slurm_job_id=job)
    need(sources()==frozen,'Audit sources changed during CPU binding')
    save(out/'plan.json',plan);(out/'plan.sha256').write_text(sha(out/'plan.json')+'\n');verify(out/'plan.json')
    save(out/'summary.json',dict(passed=True,protocol=PROTOCOL,source_sha256=frozen,plan_file=str(out/'plan.json'),
        plan_sha256=sha(out/'plan.json'),selected_models=4,fixed_core_through_student=True,one_dev_only=True,
        no_model_loaded=True,seconds=time.perf_counter()-begin,slurm_job_id=job))
    print(dict(passed=True,plan_file=str(out/'plan.json')),flush=True)


def verify(path):
    path=Path(path).resolve();plan=read(path)
    need(path.is_relative_to(OUT) and path.with_suffix('.sha256').read_text().strip()==sha(path)
         and plan['schema_version']==1 and plan['protocol']==PROTOCOL and plan['source_sha256']==sources()
         and plan['policy']==training.POLICY,'Frozen final-audit source/plan differs')
    for name,digest in plan['source_sha256'].items():need(sha(path.parent/'source'/name.replace('/','_'))==digest,'CPU source copy differs')
    for name,digest in plan['artifact_sha256'].items():need(sha(name)==digest,'Selected-model artifact changed: '+name)
    verify_source_check(plan['source_check']);parent=software.verify(plan['parent_plan_file'])
    need(sha(Path(plan['source_check'])/'summary.json')==plan['source_check_sha256']
         and sha(plan['parent_plan_file'])==plan['parent_plan_sha256'] and sha(plan['prepared_file'])==plan['prepared_sha256']
         and plan['prepared_file']==parent['prepared_file'] and plan['cases']==parent['cases']
         and plan['forced_ids']==parent['forced_ids'] and plan['forced_text']==list(software.FORCED)
         and plan['calls']==CALLS and plan['head_replay_calls']==46 and plan['cache_full_rows']==656
         and plan['fixed_prefix_kv_checks']==24 and plan['fixed_prefix_kv_layers']==28 and plan['selected_fusion_checks']==40
         and plan['ordinary_generations']==plan['extra_last_block_forwards']==0 and plan['fixed_core_through_student'] and plan['one_dev_only'],
         'Final native phase/case/call coverage differs')
    need(len(plan['runs'])==4 and {(r['condition'],r['seed']) for r in plan['runs']}=={(c,s) for c in CONDITIONS for s in SEEDS}
         and all(r['selected']['step']==CORE_STEP and r['selected']['student_step']==STUDENT_STEP for r in plan['runs']),
         'Fixed-final model inventory differs')
    for key in ('model','runtime','processor','native_api'):need(plan[key]==parent[key],'Native ancestor identity changed')
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-test',action='store_true');mode.add_argument('--check',action='store_true');mode.add_argument('--run',action='store_true')
    parser.add_argument('--source-check',type=Path);parser.add_argument('--runs',nargs=4,type=Path);parser.add_argument('--plan',type=Path)
    args=parser.parse_args();native.require_slurm(gpu=args.run)
    if not args.run:need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),'CPU final-audit requires Slurm CPU')
    if args.self_test:unit()
    elif args.check:
        need(args.source_check is not None and args.runs is not None,'Pass source selftest and exactly four completed main directories');check(args)
    else:need(args.plan is not None,'Pass exact completed CPU final-checkpoint plan');run(args)


if __name__=='__main__':main()

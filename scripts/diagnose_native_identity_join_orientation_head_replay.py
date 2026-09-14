"""Bounded CPU diagnostic; the failed registered head audit remains failed."""
from pathlib import Path
import json
import math
import os
import platform
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import diagnose_native_identity_join_factor_orientation_training as parent
from scripts import report_native_identity_join_factor_orientation_training as audit
from scripts.stage_native_vision_v6_teacher import need,read,save,sha,object_sha
ROOT=parent.OUT
PLAN=ROOT/'check_443676/plan.json'
PLAN_SHA='d299d42a4444520f148f86252e2931ad599e03afac9edadbe9201836229bf097'
FAILURE=ROOT/'report_443682/failure.json'
FAILURE_SHA='4b113df062f18c49973864849378608678264a035e30e25935ab111df525be1c'
FAILED_AUDIT=FAILURE.parent/'product_evaluation_6000_0010_capture_audit.json'
FAILED_AUDIT_SHA='dc9d135ee998ef59845a4001ef5e541637ce8b51c9db2597b77671d8fc7b4b57'
RUNS={'product':('run_product_443677','450200843c5fc6c3e72b912dc8db360f05a206afe4ec75db3527087bb8ec328e'),
      'additive':('run_additive_443678','d5e84fb8bc4dc7ff4db855b1b7528a7b3880e98238bc467c6295de5a1838d12b')}
PROPOSAL='docs/paper/NATIVE_AGGREGATION_ORIENTATION_HEAD_REPLAY_DIAGNOSTIC.md'
PROPOSAL_SHA='dd7b409529d11dc4e7ed9e49db569e8bf3b0f21cf2add151ae4facc7485ffeb7'
OWN=('scripts/diagnose_native_identity_join_orientation_head_replay.py','slurm/native_identity_join_orientation_head_replay.sbatch',PROPOSAL)
PROTOCOL='identity_join_orientation_failed_head_replay_diagnostic'
POLICY=dict(cpu_seconds=300,cpu_cores=4,memory_gib=16,norm_calls=28,norm_rows=432,head_calls=29,head_rows=448,
    primary_batches_per_arm=14,primary_rows_per_arm=216,extra_route=dict(arm='product',batch=10,rows=16),
    no_GPU=True,no_backbone=True,no_core=True,no_fit=True,no_generation=True,no_threshold_change=True,no_release=True)


def bind(path,bindings,expected=None):
    path=Path(path).resolve();digest=sha(path);need(expected is None or digest==expected,'Changed diagnostic input: '+str(path))
    need(str(path) not in bindings or bindings[str(path)]==digest,'Conflicting diagnostic binding');bindings[str(path)]=digest;return digest


def sources():
    need(sha(REPO/PROPOSAL)==PROPOSAL_SHA,'Diagnostic proposal changed')
    return {name:sha(REPO/name) for name in OWN}


def inherited_sources():
    result=parent.inherited_sources()
    for name,digest in parent.sources().items():
        need(name not in result or result[name]==digest,'Conflicting original source');result[name]=digest
    return result


def snapshot(out):
    frozen=sources();inherited=inherited_sources();(out/'source').mkdir()
    for name,digest in frozen.items():
        target=out/'source'/name.replace('/','_');target.write_bytes((REPO/name).read_bytes());need(sha(target)==digest,'Diagnostic source copy differs')
    save(out/'source_hashes.json',frozen);save(out/'inherited_sources.json',dict(source_sha256=inherited));return frozen,inherited


def log_windows(summary,scenes,bindings):
    bind(summary['training_file'],bindings,summary['training_sha256']);logs=read(summary['training_file']);need(len(logs)==6000,'All6000 logs required')
    result=[];start=1
    for end in (600,2000,4000,6000):
        selected=logs[start-1:end];roles={'first_token':[],'name_continuation':[],'EOS':[]};norms=[];ces=[]
        for step,row in enumerate(selected,start):
            need(row['step']==step and len(row['sids'])==16 and row['consistency_coefficient']==row['weighted_consistency_loss']==0.,'Logged update/objective differs')
            values=row['position_losses']['ce'];cursor=0;targets=[]
            for sid in row['sids']:
                ids=scenes[sid]['target_ids'];targets.extend(ids);count=len(ids);part=values[cursor:cursor+count];cursor+=count
                need(len(part)==count and count>=2 and all(math.isfinite(x) and x>=0 for x in part),'Logged CE positions invalid')
                roles['first_token'].append(part[0]);roles['name_continuation'].extend(part[1:-1]);roles['EOS'].append(part[-1])
            need(cursor==len(values) and targets==row['target_ids'] and math.isfinite(row['gradient_norm']) and row['gradient_norm']>=0
                 and row['clipped']==(row['gradient_norm']>1.) and math.isfinite(row['ce_loss']),'Logged target/gradient metadata differs')
            norms.append(row['gradient_norm']);ces.append(row['ce_loss'])
        result.append(dict(labels=dict(first_update=start,last_update=end),metrics=dict(updates=len(selected),
            mean_logged_scene_balanced_ce=sum(ces)/len(ces),gradient_norm_mean=sum(norms)/len(norms),gradient_norm_min=min(norms),gradient_norm_max=max(norms),
            clipped_updates=sum(r['clipped'] for r in selected),clipped_fraction=sum(r['clipped'] for r in selected)/len(selected)),
            token_roles={k:dict(positions=len(v),mean_nll=sum(v)/len(v) if v else None) for k,v in roles.items()}));start=end+1
    return dict(descriptive_logged_values_only=True,no_loss_or_gradient_recomputation=True,windows=result)


def vector_summary(torch,value,target):
    promoted=value.double();prob=promoted.softmax(-1);top=promoted.topk(5);winner=int(promoted.argmax())
    return dict(argmax_id=winner,correct=winner==target,target_logit=float(promoted[target]),target_probability_fp64=float(prob[target]),
        top_two_margin=float(top.values[0]-top.values[1]),tied_maximum_ids=torch.where(promoted==promoted[winner])[0].tolist(),
        top5=[dict(token_id=int(i),logit=float(v),probability_fp64=float(prob[i])) for v,i in zip(top.values,top.indices)])


def row_metrics(torch,reference,replayed,targets,prior_metrics):
    need(reference.shape==replayed.shape and reference.dtype==replayed.dtype==torch.float16
         and reference.ndim==2 and reference.shape[1]==152064 and len(targets)==len(reference)==len(prior_metrics)
         and bool(torch.isfinite(reference).all()) and bool(torch.isfinite(replayed).all()),'Diagnostic native logits differ in shape/dtype/validity')
    result=[]
    for gpu,cpu,target,registered in zip(reference,replayed,targets,prior_metrics):
        a=vector_summary(torch,gpu,target);b=vector_summary(torch,cpu,target);difference=(gpu.double()-cpu.double()).abs()
        maximum=float(difference.max());tv=float(.5*(gpu.double().softmax(-1)-cpu.double().softmax(-1)).abs().sum())
        result.append(dict(gpu=a,cpu=b,argmax_exact=a['argmax_id']==b['argmax_id'],registered_fp32_metric=registered,
            fp64_tv_descriptive=tv,maximum_logit_difference=maximum,changed_logit_coordinates=int((difference!=0).sum()),
            gpu_margin_exceeds_twice_observed_max_error=a['top_two_margin']>2*maximum,
            gpu_winner_logit_in_cpu=float(cpu[a['argmax_id']]),cpu_winner_logit_in_gpu=float(gpu[b['argmax_id']]),
            target_logit_difference=float(cpu[target].double()-gpu[target].double())))
    return result


def sensitivity(rows,details):
    def score(values):return audit.first_token_criterion([dict(row,first_token_correct=bool(value)) for row,value in zip(rows,values)])
    gpu=[r['gpu']['correct'] for r in details];cpu=[r['cpu']['correct'] for r in details];same=[r['argmax_exact'] for r in details]
    return dict(gpu=score(gpu),cpu=score(cpu),
        observed_backend_choice_lower=score([a and b for a,b in zip(gpu,cpu)]),
        observed_backend_choice_upper=score([a or b for a,b in zip(gpu,cpu)]),
        every_disagreement_wrong=score([a if eq else False for a,eq in zip(gpu,same)]),
        every_disagreement_correct=score([a if eq else True for a,eq in zip(gpu,same)]),
        disagreement_rows=sum(not value for value in same),
        deterministic_monotone_sensitivity_only=True,not_a_confidence_interval=True,no_registered_acceptance=True)


def run(out,data,frozen,inherited,bindings):
    import torch
    torch.set_num_threads(4);bind(PLAN,bindings,PLAN_SHA);plan=parent.verify_plan(PLAN,ancestors=True)
    bind(FAILURE,bindings,FAILURE_SHA);failure=read(FAILURE);bind(FAILED_AUDIT,bindings,FAILED_AUDIT_SHA);failed=read(FAILED_AUDIT)
    need(failure['type']=='ValueError' and failure['message']=='Factor/core/native/NLL audit failed; metrics and raw evidence retained'
         and failure['source_sha256']==plan['source_sha256'] and not (FAILURE.parent/'summary.json').exists()
         and failed['phase']=='evaluation' and failed['endpoint_step']==6000 and failed['step']==10 and failed['arm']=='product'
         and [i for i,r in enumerate(failed['native_head']) if not r['passed']]==[8]
         and failed['sids'][8]=='ij_b955ee41a7cec2f79cccdff8','Preserved original failure identity differs')
    for name,digest in plan['source_sha256'].items():bind(FAILURE.parent/'source'/name.replace('/','_'),bindings,digest)
    rows=read(plan['rows_file']);scenes=read(plan['scenes_file']);need(len(rows)==216,'All216 endpoint rows required')
    bind(plan['native_model_file'],bindings,plan['native_model_sha256'])
    module_identity=parent.oracle.native_module_identity(torch,plan['native_identity']);need(module_identity==plan['native_module_identity'],'Native implementation changed')
    packet=torch.load(plan['native_model_file'],map_location='cpu',weights_only=True)
    norm,head=parent.oracle.native_modules(torch,packet,plan['native_identity'],'cpu');del packet
    before=dict(norm=parent.tensor_info(norm.weight),head=parent.tensor_info(head.weight));versions=(norm.weight._version,head.weight._version)
    calls=[];current={}
    def observe(name):
        def hook(module,args,output):calls.append(dict(current,module=name,input_shape=list(args[0].shape),output_shape=list(output.shape),input_dtype=str(args[0].dtype),output_dtype=str(output.dtype)))
        return hook
    handles=[norm.register_forward_hook(observe('norm')),head.register_forward_hook(observe('head'))];results={};extra=None
    try:
        with torch.inference_mode():
            for arm,(name,digest) in RUNS.items():
                directory=ROOT/name;bind(directory/'summary.json',bindings,digest);summary=read(directory/'summary.json');config=read(directory/'config.json')
                need(summary['passed'] is summary['completed'] is True and summary['arm']==arm and summary['steps']==6000
                     and summary['plan_file']==str(PLAN) and summary['plan_sha256']==PLAN_SHA and summary['source_sha256']==plan['source_sha256']
                     and summary['native_identity']==plan['native_identity'] and all(summary[k]==v for k,v in config.items()),'Fixed completed GPU endpoint ownership differs')
                for key in ('raw','captures','calls','predictions'):bind(summary[key+'_file'],bindings,summary[key+'_sha256'])
                bind(summary['checkpoint'],bindings,summary['checkpoint_sha256']);bind(directory/'config.json',bindings)
                endpoint=torch.load(summary['raw_file'],map_location='cpu',weights_only=True)
                need(endpoint['endpoint_step']==6000 and endpoint['sids']==[r['sid'] for r in rows]
                     and endpoint['logits'].shape==(216,152064) and endpoint['logits'].dtype==torch.float16
                     and parent.tensor_info(endpoint['logits'])==summary['raw_tensor'],'Exact original GPU raw endpoint differs')
                records=[r for r in read(summary['captures_file']) if r['phase']=='evaluation'];need(len(records)==14,'All14 endpoint captures required')
                actual_calls=[r for r in read(summary['calls_file']) if r['phase']=='evaluation'];expected_calls=[];details=[];batches=[]
                for batch,record in enumerate(records,1):
                    start=(batch-1)*16;selected=rows[start:start+16];sids=[r['sid'] for r in selected];q=len(selected);targets=[r['first_token_id'] for r in selected]
                    need(record['arm']==arm and record['step']==batch and record['endpoint_step']==record['weights_step']==6000
                         and record['sids']==sids and record['weights_file']==summary['checkpoint'] and record['weights_sha256']==summary['checkpoint_sha256'],
                         'Captured endpoint batch/weights differs')
                    bind(record['file'],bindings,record['sha256']);cap=torch.load(record['file'],map_location='cpu',weights_only=True)
                    need(cap['arm']==arm and cap['phase']=='evaluation' and cap['step']==batch and cap['endpoint_step']==6000
                         and cap['sids']==sids and cap['pairing']==record['pairing']=='paired'
                         and cap['layout']['first_query_only'] is True and cap['layout']['target_ids']==targets
                         and cap['fused_global'].shape==cap['normalized'].shape==(q,3584)
                         and cap['fused_global'].dtype==cap['normalized'].dtype==torch.float16
                         and torch.equal(cap['logits'],endpoint['logits'][start:start+q])
                         and {k:parent.tensor_info(cap[k]) for k in ('fused_global','normalized','logits')}==record['tensors'],
                         'Saved GPU native-head input/output binding differs')
                    for module in ('norm','head'):
                        expected_calls.append(dict(module=module,phase='evaluation',step=batch,endpoint_step=6000,input_shape=[1,q,3584],
                            output_shape=[1,q,3584 if module=='norm' else 152064],input_dtype='torch.float16',output_dtype='torch.float16'))
                    current.update(arm=arm,batch=batch,route='norm_then_head');normalized=norm(cap['fused_global'].unsqueeze(0));replayed=head(normalized)
                    raw=data/f'{arm}_batch_{batch:02d}.pt';torch.save(dict(arm=arm,batch=batch,sids=sids,capture_file=record['file'],capture_sha256=record['sha256'],
                        cpu_normalized=normalized,cpu_logits=replayed),raw)
                    record_out=dict(file=str(raw),sha256=sha(raw),normal_tensor=parent.tensor_info(normalized),logit_tensor=parent.tensor_info(replayed))
                    metrics=audit.replay_metric(torch,cap['logits'],replayed[0]);compared=row_metrics(torch,cap['logits'],replayed[0],targets,metrics)
                    normal_difference=(normalized[0].float()-cap['normalized'].float()).abs()
                    for i,(row,value) in enumerate(zip(selected,compared)):
                        details.append(dict(labels={k:row[k] for k in ('sid','base_contrast_id','orientation_version','n_frames','variant','gold')},
                            row_index=start+i,batch=batch,batch_row=i,**value,normalized_exact=torch.equal(normalized[0,i],cap['normalized'][i]),
                            normalized_max_absolute=float(normal_difference[i].max()),cpu_replay=record_out))
                    batches.append(record_out);save(out/f'{arm}_batch_{batch:02d}.json',dict(batch=batch,rows=details[start:start+q],artifact=record_out))
                    if arm=='product' and batch==10:
                        current.update(route='head_on_saved_GPU_normalized');extra_logits=head(cap['normalized'].unsqueeze(0))
                        extra_file=data/'product_batch10_saved_normalized_head.pt';torch.save(dict(sids=sids,capture_file=record['file'],capture_sha256=record['sha256'],
                            cpu_logits=extra_logits),extra_file)
                        extra=dict(file=str(extra_file),sha256=sha(extra_file),rows=[dict(sid=sid,**value) for sid,value in zip(sids,
                            row_metrics(torch,cap['logits'],extra_logits[0],targets,audit.replay_metric(torch,cap['logits'],extra_logits[0])))],
                            fixed_failed_batch_only=True,no_replacement_of_registered_route=True)
                        save(out/'extra_head_route.json',extra)
                    del cap,normalized,replayed
                need(actual_calls==expected_calls and len(details)==216,'Actual GPU shape/call coverage differs')
                screens=sensitivity(rows,details);need(screens['gpu']==summary['first_token_fit'],'Saved GPU screen recount differs')
                saved=read(summary['predictions_file']);need(len(saved)==216 and all(a['sid']==b['labels']['sid'] and a['argmax_id']==b['gpu']['argmax_id']
                     and a['first_token_correct']==b['gpu']['correct'] for a,b in zip(saved,details)),'Saved GPU prediction recount differs')
                windows=log_windows(summary,scenes,bindings);save(out/f'{arm}_rows.json',details);save(out/f'{arm}_training_windows.json',windows)
                results[arm]=dict(sensitivity=screens,registered_head_rule_passed=all(r['registered_fp32_metric']['passed'] for r in details),
                    maximum_registered_fp32_tv=max(r['registered_fp32_metric']['full_vocabulary_tv'] for r in details),
                    rows_file=str(out/f'{arm}_rows.json'),rows_sha256=sha(out/f'{arm}_rows.json'),disagreements=[r for r in details if not r['argmax_exact']],
                    batches=batches,training_windows_file=str(out/f'{arm}_training_windows.json'),training_windows_sha256=sha(out/f'{arm}_training_windows.json'))
    finally:
        for handle in handles:handle.remove()
        save(out/'calls.json',calls)
    need(before==dict(norm=parent.tensor_info(norm.weight),head=parent.tensor_info(head.weight)) and versions==(norm.weight._version,head.weight._version)
         and all(not p.requires_grad and p.grad is None for module in (norm,head) for p in module.parameters()),'Frozen native modules changed')
    norm_calls=[r for r in calls if r['module']=='norm'];head_calls=[r for r in calls if r['module']=='head']
    need(len(norm_calls)==28 and len(head_calls)==29 and sum(r['input_shape'][1] for r in norm_calls)==432
         and sum(r['input_shape'][1] for r in head_calls)==448 and extra is not None,'Fixed diagnostic native call inventory exceeded or incomplete')
    return dict(passed=True,completed=True,protocol=PROTOCOL,policy=POLICY,source_sha256=frozen,inherited_source_sha256=inherited,
        plan_file=str(PLAN),plan_sha256=PLAN_SHA,original_failure_file=str(FAILURE),original_failure_sha256=FAILURE_SHA,
        original_audit_remains_failed=True,no_acceptance_or_native_release=True,results=results,extra_route=extra,
        norm_calls=28,norm_rows=432,head_calls=29,head_rows=448,calls_file=str(out/'calls.json'),calls_sha256=sha(out/'calls.json'),
        native_identity=plan['native_identity'],native_identity_sha256=plan['native_identity_sha256'],native_module_identity=module_identity,
        runtime=dict(torch_version=str(torch.__version__),python_version=platform.python_version(),platform=platform.platform(),
            node=platform.node(),threads=torch.get_num_threads(),mkldnn_enabled=torch.backends.mkldnn.enabled),
        input_bindings=bindings,no_GPU_or_backbone_or_core=True,no_fitting_or_generation=True)


def main():
    need(len(sys.argv)==1,'This fixed diagnostic has no tunable arguments');parent.native.require_slurm(gpu=False)
    need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),'CPU-only diagnostic required')
    started=time.perf_counter();out=ROOT/f'head_replay_{os.environ["SLURM_JOB_ID"]}';data=parent.DATA/out.name
    out.mkdir(parents=True,exist_ok=False);data.mkdir(parents=True,exist_ok=False);frozen,inherited=snapshot(out);bindings={}
    save(out/'request.json',dict(protocol=PROTOCOL,policy=POLICY,source_sha256=frozen,failure_file=str(FAILURE),failure_sha256=FAILURE_SHA))
    try:
        result=run(out,data,frozen,inherited,bindings);need(sources()==frozen and inherited_sources()==inherited,'Diagnostic source changed')
        result['elapsed_seconds']=time.perf_counter()-started;need(result['elapsed_seconds']<=300,'Fixed CPU diagnostic cap exceeded')
        save(out/'analysis.json',result)
        lines=['# Failed orientation head-audit diagnostic','',
            'The original registered report remains FAILED. This CPU diagnostic does not relax its argmax rule or release native/fresh inference.','',
            '| Arm | GPU saved correct | CPU replay correct | Argmax disagreements | Optimistic disagreement bound |',
            '|---|---:|---:|---:|---:|']
        for arm,value in result['results'].items():
            s=value['sensitivity'];lines.append(f"| {arm} | {s['gpu']['first_correct']}/216 | {s['cpu']['first_correct']}/216 | {s['disagreement_rows']} | {s['every_disagreement_correct']['first_correct']}/216 |")
        lines+=['','All432 rows and full-vocabulary CPU outputs are retained. Sensitivity bounds are deterministic and are not statistical confidence intervals. '
            'The one extra fixed head route uses the saved GPU-normalized product batch10; it does not replace the registered normalization-plus-head replay. '
            'Training-window summaries describe logged quantities only.']
        (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
        save(out/'summary.json',dict(passed=True,completed=True,protocol=PROTOCOL,source_sha256=frozen,inherited_source_sha256=inherited,
            analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),report_file=str(out/'REPORT.md'),report_sha256=sha(out/'REPORT.md'),
            original_audit_remains_failed=True,no_acceptance_or_native_release=True,norm_calls=28,head_calls=29,head_rows=448,elapsed_seconds=result['elapsed_seconds']))
        print(json.dumps(dict(passed=True,directory=str(out))),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,inherited_source_sha256=inherited,
            input_bindings=bindings,elapsed_seconds=time.perf_counter()-started,partial_outputs_retained=True));raise


if __name__=='__main__':main()

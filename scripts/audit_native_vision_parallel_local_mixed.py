"""Independently recompute every mixed-oracle comparison from saved native logits.

CPU only. The original numerical gate and its failures are immutable. Executed
internal causal-mask tensors and K/V values were not retained; their original
source-bound assertions are audited as such, not misrepresented as a replay.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import probe_native_vision_parallel_local_mixed as original
from scripts.stage_native_vision_v6_teacher import need,read,save,sha
BASE=REPO/'outputs/native_aggregation_vlm/parallel_local_mixed'
DEFAULT=BASE/'run_441729'
DATA=Path('/mnt/data/gabriele/gnn_transformer/parallel_local_mixed')
OWN=('scripts/audit_native_vision_parallel_local_mixed.py','slurm/audit_native_vision_parallel_local_mixed.sbatch')
TOL=.02
METRIC_TOL=1e-10


def sources():return {name:sha(REPO/name) for name in OWN}


def snapshot(out):
    (out/'source').mkdir()
    for name in OWN:(out/'source'/name.replace('/','_')).write_bytes((REPO/name).read_bytes())
    save(out/'source_hashes.json',sources())


def tensor_info(torch,value):
    raw=value.detach().cpu().contiguous()
    return dict(shape=list(raw.shape),dtype=str(raw.dtype),sha256=hashlib.sha256(raw.view(torch.uint8).numpy().tobytes()).hexdigest())


def compare(torch,a,b,ha,hb,policy,token_ids):
    """Independent CPU implementation; do not call the GPU reporter formula."""
    a=a.to(dtype=torch.float64);b=b.to(dtype=torch.float64);ha=ha.to(dtype=torch.float64);hb=hb.to(dtype=torch.float64)
    need(a.ndim==b.ndim==1 and a.shape==b.shape and ha.shape==hb.shape and
         all(bool(torch.isfinite(x).all()) for x in (a,b,ha,hb)),'Nonfinite or incompatible saved comparison')
    p=torch.exp(a-torch.logsumexp(a,0));q=torch.exp(b-torch.logsumexp(b,0))
    d=a-b;cd=d-d.mean();dh=ha-hb
    metrics=dict(policy=policy,full_vocabulary_tv=float(torch.sum(torch.abs(p-q))/2),top1_equal=int(a.argmax())==int(b.argmax()),
        maximum_absolute_logit_difference=float(d.abs().max()),centered_maximum_absolute_logit_difference=float(cd.abs().max()),
        centered_rms_logit_difference=float(torch.linalg.vector_norm(cd)/math.sqrt(len(cd))),
        hidden_maximum_absolute_difference=float(dh.abs().max()),hidden_rms_difference=float(torch.linalg.vector_norm(dh)/math.sqrt(len(dh))),
        hidden_relative_l2=float(torch.linalg.vector_norm(dh)/torch.linalg.vector_norm(hb)) if float(hb.norm()) else None,
        hidden_cosine=float(torch.dot(ha,hb)/(ha.norm()*hb.norm())) if float(ha.norm()*hb.norm()) else None,
        hidden_state_equality_claimed=False)
    if policy=='local_prefill':
        i0,i1=token_ids['0'],token_ids['1']
        metrics.update(binary_prediction_exact=bool(a[i1]>a[i0])==bool(b[i1]>b[i0]),
            conditional_p1_difference=abs(float(torch.sigmoid(a[i1]-a[i0]))-float(torch.sigmoid(b[i1]-b[i0]))),
            numeric_mass_difference=abs(float(p[i0]+p[i1])-float(q[i0]+q[i1])))
        metrics['passed']=metrics['binary_prediction_exact'] and metrics['conditional_p1_difference']<=TOL and metrics['numeric_mass_difference']<=TOL
    else:
        need(policy=='full_vocab','Unknown registered policy')
        metrics['passed']=metrics['full_vocabulary_tv']<=TOL and metrics['top1_equal']
    return metrics


def agree(actual,archived,path='metric'):
    if type(actual) is bool or actual is None or isinstance(actual,(str,int)):
        need(actual==archived,'Archived value differs: '+path)
    elif isinstance(actual,float):
        need(math.isfinite(actual) and math.isfinite(archived) and math.isclose(actual,archived,rel_tol=METRIC_TOL,abs_tol=METRIC_TOL),
             'Archived FP64 metric differs: '+path)
    elif isinstance(actual,dict):
        for name,value in actual.items():need(name in archived,'Missing archived '+path+'.'+name);agree(value,archived[name],path+'.'+name)
    else:need(actual==archived,'Archived structure differs: '+path)


def self_test(torch):
    h=torch.ones(3584);a=torch.tensor([.7,.3],dtype=torch.float64).log();b=torch.tensor([.721,.279],dtype=torch.float64).log()
    need(compare(torch,a,a+100,h,h,'full_vocab',{'0':0,'1':1})['passed'],'Common logit shift must pass')
    changed=compare(torch,a,b,h,h,'full_vocab',{'0':0,'1':1})
    need(changed['top1_equal'] and not changed['passed'] and changed['full_vocabulary_tv']>.02,'TV-only failure lost')
    tied=compare(torch,torch.zeros(2),torch.tensor([0.,1.]),h,h,'full_vocab',{'0':0,'1':1})
    need(not tied['top1_equal'] and not tied['passed'],'Top1 mismatch accepted')
    zero=compare(torch,a,a,h,h,'local_prefill',{'0':0,'1':1});need(zero['passed'],'Local identity failed')
    try:agree(dict(passed=False),dict(passed=True))
    except ValueError:pass
    else:raise AssertionError('Gate relabeling was accepted')
    return dict(passed=True,tests=['common_shift','same_top1_tv_failure','top1_failure','local_identity','failed_gate_preserved'])


def audit(torch,args,out):
    started=time.perf_counter();run=Path(args.run_directory).resolve()
    need(run.parent==BASE and run.name.startswith('run_'),'Expected canonical mixed-oracle run')
    summary=read(run/'summary.json');source_summary_sha=sha(run/'summary.json')
    config=read(run/'config.json');plan_path=Path(summary['plan_file'])
    plan=original.verify_plan(plan_path)
    need(summary['completed'] is True and summary['computational_integrity_passed'] is True,'Original run incomplete')
    need(sha(plan_path)==summary['plan_sha256']==config['plan_sha256']==sha(run/'plan.json'),'Plan binding differs')
    for name,digest in plan['source_sha256'].items():
        need(sha(REPO/name)==digest and sha(run/'source'/name.replace('/','_'))==digest
             and sha(plan_path.parent/'source'/name.replace('/','_'))==digest,'Original source/snapshot changed')
    need(summary['source_sha256']==config['source_sha256']==plan['source_sha256'],'Original source ledgers differ')
    for name in ('comparisons','cache_checks'):need(sha(run/(name+'.json'))==summary[name+'_sha256'],'Original '+name+' hash differs')
    need(sha(summary['rows_file'])==summary['rows_sha256'] and
         sha(summary['state_inventory_file'])==summary['state_inventory_sha256'],'Raw/inventory binding differs')
    rows=[json.loads(x) for x in Path(summary['rows_file']).read_text().splitlines()]
    inventory=read(summary['state_inventory_file']);calls=summary['per_call'];archived=read(run/'comparisons.json')
    cache_checks=read(run/'cache_checks.json');blob=torch.load(plan['prepared_inputs_file'],map_location='cpu',weights_only=True)
    need(len(rows)==492 and len(inventory)==len(calls)==92 and len(archived)==246 and len(cache_checks)==6,'All original outputs required')
    raw={(x['scene_id'],x['tag'],x['row_id']):x for x in rows};need(len(raw)==492,'Repeated original response')
    call_map={(x['scene_id'],x['tag']):x for x in calls};need(len(call_map)==92,'Repeated call')
    weight_dtypes=read(run/'native_weight_dtypes.json');need(weight_dtypes==summary['native_weight_dtypes'],'Native weight dtype record differs')
    states={};recounted_calls=Counter();row_count=0;state_bindings={};logical_masks=[]
    scene_map={s['sid']:s for s in plan['scenes']}
    # Native mRoPE is recomputed with config only; no model/processor weights are loaded.
    from transformers import AutoConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
    from types import SimpleNamespace
    owner=SimpleNamespace(config=AutoConfig.from_pretrained(str(original.MODEL),trust_remote_code=True))
    fn=Qwen2_5_VLModel.get_rope_index
    for item in inventory:
        sid,tag=item['scene_id'],item['tag'];scene=scene_map[sid];entry=blob['scenes'][sid];b=len(scene['rows']);l=scene['padded_prompt_length']
        path=Path(item['path']);need(path==DATA/run.name/f'{sid}__{tag}.pt' and sha(path)==item['sha256'],'Saved state path/hash differs')
        call=call_map[(sid,tag)];need(call['state_path']==str(path) and call['state_sha256']==item['sha256'],'Call/state binding differs')
        state=torch.load(path,map_location='cpu',weights_only=True)
        need(state['schema_version']==1 and state['scene_id']==sid and state['tag']==tag and state['row_ids']==item['rows'],
             'Saved state identity differs')
        if tag.startswith('serial_'):
            i=int(tag.split('_')[1]);input_row=entry['rows'][i];expected_ids=[scene['rows'][i]['row_id']];step=0
            expected_positions,_=fn(owner,input_ids=input_row['input_ids'],image_grid_thw=input_row.get('image_grid_thw'),attention_mask=input_row['attention_mask'])
            expected_mask=input_row['attention_mask'];expected_cp=torch.arange(expected_mask.shape[1]);expected_vision=int(i<b-1)
            need(call['mask_audit'] is None,'Serial mask policy differs')
        else:
            step=0 if tag=='mixed_prefill' else int(tag[-1]);need(tag in ('mixed_prefill','cached_step1','full_step1','cached_step2','full_step2'),'Unknown original call')
            expected_ids=[x['row_id'] for x in scene['rows']]
            full=original.extend(torch,entry['mixed'],plan['forced_token_ids'][:step]);expected_mask=full['attention_mask']
            position=entry['full_positions'][step];delta=entry['full_deltas'][step]
            native_position,native_delta=fn(owner,input_ids=full['input_ids'],image_grid_thw=full['image_grid_thw'],attention_mask=full['attention_mask'])
            need(torch.equal(position,native_position) and torch.equal(delta,native_delta),'Frozen mRoPE differs from independently recomputed native positions')
            need(dict(tensors={k:tensor_info(torch,v) for k,v in full.items()},position_ids=tensor_info(torch,position),rope_deltas=delta.tolist())
                 ==scene['full_prefix_identities'][step],'Frozen full-prefix tensors changed')
            if tag.startswith('cached_'):
                expected_positions=torch.cat(((expected_mask.cumsum(-1)-1)[:,-1:].unsqueeze(0),position[:,:,-1:]),dim=0)
                expected_cp=torch.tensor([l+step-1]);expected_vision=0
            else:expected_positions=position;expected_cp=torch.arange(l+step);expected_vision=1
            report=call['mask_audit'];valid=int(expected_mask.index_select(1,expected_cp).sum())
            need(report['valid_queries_exact'] is True and report['valid_query_rows']==valid
                 and report['padding_query_rows']==b*len(expected_cp)-valid and report['mask_shape']==[b,1,len(expected_cp),expected_mask.shape[1]],
                 'Executed mask assertion/count binding differs')
            logical_masks.append(dict(scene_id=sid,tag=tag,valid_queries=valid,actual_internal_mask_retained=False))
        need(state['row_ids']==expected_ids and state['forced_token_ids']==plan['forced_token_ids'][:step]
             and torch.equal(state['native_position_ids'],expected_positions) and torch.equal(state['attention_mask'],expected_mask)
             and torch.equal(state['cache_position'],expected_cp),'Saved logical token/mRoPE/mask/cache-position identity differs')
        expected_calls=dict(model=1,vision=expected_vision,language=1,norm=1)
        need(call['calls']==expected_calls and call['batch_size']==len(expected_ids),'Executed call counts differ');recounted_calls.update(expected_calls)
        logits=state['native_next_token_logits'];hidden=state['pre_final_rms_hidden'];vocab=owner.config.text_config.vocab_size
        need(logits.shape==(len(expected_ids),vocab) and hidden.shape==(len(expected_ids),3584)
             and bool(torch.isfinite(logits).all()) and bool(torch.isfinite(hidden).all()),'Saved native vector shape/finite state differs')
        need(str(logits.dtype)==item['logits_dtype']==call['logits_dtype'] and str(hidden.dtype)==item['hidden_dtype']==call['hidden_dtype']
             and item['native_weight_dtypes']==state['native_weight_dtypes']==weight_dtypes,'Actual dtype binding differs')
        for i,row_id in enumerate(expected_ids):
            record=raw[(sid,tag,row_id)];x=logits[i].double();z=float(torch.logsumexp(x,0));ti=plan['count_token_ids'];a=float(x[ti['0']]);c=float(x[ti['1']])
            p0,p1=math.exp(a-z),math.exp(c-z);pother=max(0.,1-math.fsum((p0,p1)))
            agree(dict(logit0=a,logit1=c,log_normalizer=z,top1_id=int(x.argmax()),step=step,forced_token_ids=plan['forced_token_ids'][:step],state_row=i),record)
            need(record['state_path']==str(path) and len(record['probabilities'])==3 and all(math.isclose(v,w,rel_tol=METRIC_TOL,abs_tol=METRIC_TOL)
                 for v,w in zip((p0,p1,pother),record['probabilities'])),'Raw grouped probability/state binding differs')
        states[(sid,tag)]=state;state_bindings[str(path)]=item['sha256'];row_count+=len(expected_ids)
    need(row_count==492 and dict(recounted_calls)==summary['calls']==dict(model=92,vision=86,language=92,norm=92),'Final coverage/calls differ')
    checks={(x['scene_id'],x['tag']):x for x in cache_checks};need(len(checks)==6,'Repeated cache metadata')
    for scene in plan['scenes']:
        b=len(scene['rows']);l=scene['padded_prompt_length'];cfg=owner.config.text_config
        for step,tag in ((0,'prefill'),(1,'step1'),(2,'step2')):
            record=checks[(scene['sid'],tag)]
            need(record['length']==l+step and len(record['layers'])==cfg.num_hidden_layers,'Cache length/layer binding differs')
            for i,layer in enumerate(record['layers']):
                expected=[b,cfg.num_key_value_heads,l+step,cfg.hidden_size//cfg.num_attention_heads]
                need(layer['layer']==i and layer['shape']==expected and layer['dtype'] in summary['actual_hidden_dtypes'],
                     'Cache geometry/dtype differs')
            if step:
                need(record['old_prefix_exact_by_layer']==[True]*cfg.num_hidden_layers and
                     record['broadcast_token_id']==plan['forced_token_ids'][step-1] and
                     all(record[k] is True for k in ('pixel_values_absent_during_decode','logical_mrope_and_masks_exact',
                                                     'rope_deltas_preserved_around_reference')),'Cache retention assertions differ')
    reconstructed=[]
    for scene in plan['scenes']:
        sid=scene['sid'];b=len(scene['rows']);mixed=states[(sid,'mixed_prefill')]
        for i in range(b):
            serial=states[(sid,f'serial_{i:03d}')]
            result=compare(torch,mixed['native_next_token_logits'][i],serial['native_next_token_logits'][0],
                mixed['pre_final_rms_hidden'][i],serial['pre_final_rms_hidden'][0],
                'local_prefill' if i<b-1 else 'full_vocab',plan['count_token_ids'])
            result.update(scene_id=sid,tag='mixed_prefill',row_id=scene['rows'][i]['row_id'],row_index=i);reconstructed.append(result)
        for step in (1,2):
            cached=states[(sid,f'cached_step{step}')];full=states[(sid,f'full_step{step}')]
            for i in range(b):
                result=compare(torch,cached['native_next_token_logits'][i],full['native_next_token_logits'][i],
                    cached['pre_final_rms_hidden'][i],full['pre_final_rms_hidden'][i],'full_vocab',plan['count_token_ids'])
                result.update(scene_id=sid,tag=f'cached_vs_full_step{step}',row_id=scene['rows'][i]['row_id'],row_index=i);reconstructed.append(result)
    need(len(reconstructed)==246,'Missing numerical comparisons')
    for calculated,prior in zip(reconstructed,archived):agree(calculated,prior,'comparison')
    violations=[x for x in reconstructed if not x['passed']]
    need(summary['numerical_gate_passed']==(not violations) and len(violations)==len(summary['violations']),
         'Original numerical gate/failures changed')
    for a,b in zip(violations,summary['violations']):agree(a,b,'failure')
    max_metric_error=max(abs(a[k]-b[k]) for a,b in zip(reconstructed,archived) for k in a if isinstance(a[k],float))
    groups={name:[x for x in reconstructed if x['policy']==name] for name in ('local_prefill','full_vocab')}
    for name,values in groups.items():
        expected=dict(n=len(values),passed=all(x['passed'] for x in values),violations=sum(not x['passed'] for x in values),
            maximum_full_vocabulary_tv=max(x['full_vocabulary_tv'] for x in values),top1_mismatches=sum(not x['top1_equal'] for x in values),
            maximum_hidden_relative_l2=max(x['hidden_relative_l2'] for x in values if x['hidden_relative_l2'] is not None))
        if name=='local_prefill':expected.update(maximum_conditional_p1_difference=max(x['conditional_p1_difference'] for x in values),
            maximum_numeric_mass_difference=max(x['numeric_mass_difference'] for x in values))
        agree(expected,summary['numerical_groups'][name],'group')
    need(sha(run/'summary.json')==source_summary_sha and original.verify_plan(plan_path)==plan,'Original artifacts changed during audit')
    save(out/'recomputed_comparisons.json',reconstructed)
    result=dict(schema_version=1,audit_passed=True,original_run=str(run),original_summary_sha256=source_summary_sha,
        plan_sha256=sha(plan_path),original_numerical_gate_passed=summary['numerical_gate_passed'],
        original_failure_count=len(violations),recomputed_violations=violations,comparisons=246,states=92,response_rows=492,
        original_calls=dict(recounted_calls),maximum_metric_reproduction_error=max_metric_error,
        metric_comparison_tolerance=METRIC_TOL,unchanged_original_numerical_threshold=TOL,
        source_sha256=sources(),original_source_sha256=plan['source_sha256'],state_sha256=state_bindings,
        rows_sha256=summary['rows_sha256'],comparisons_sha256=summary['comparisons_sha256'],cache_checks_sha256=summary['cache_checks_sha256'],
        logical_masks=logical_masks,actual_hidden_dtypes=summary['actual_hidden_dtypes'],native_weight_dtypes=weight_dtypes,
        total_original_forward_seconds=sum(x['forward_seconds'] for x in calls),original_total_seconds=summary['total_seconds'],
        elapsed_seconds=time.perf_counter()-started,slurm_job_id=os.environ['SLURM_JOB_ID'],
        limitations=['No new model forward, fitting, threshold adjustment or original artifact mutation.',
          'Full-vocabulary metrics were independently recomputed from all saved native logits on CPU.',
          'Actual internal causal-mask tensors and K/V values were not saved; original exact assertions remain source-bound evidence.',
          'Saved logical positions, key masks, cache positions, all cache geometries and assertion bindings were independently verified.',
          'This audit does not make the original failed numerical gate pass or establish aggregation/reasoning efficacy.'])
    save(out/'summary.json',result)
    lines=['# Independent mixed-oracle CPU audit','',f"Audit: PASS. Original numerical gate: {'PASS' if result['original_numerical_gate_passed'] else 'FAIL'}.",'',
        f"All92state files,492logit vectors and246comparisons verified. Original failures retained:{len(violations)}.",
        f"Maximum reproduced metric difference:{max_metric_error:.3g}; audit arithmetic tolerance:{METRIC_TOL:g}.",'',*result['limitations'],'']
    (out/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(output=str(out),audit_passed=True,original_numerical_gate_passed=result['original_numerical_gate_passed'],seconds=result['elapsed_seconds'])),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--source-ledger',type=Path);parser.add_argument('--run-directory',type=Path,default=DEFAULT);args=parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS',''),
         'Every numerical operation requires the CPU Slurm wrapper')
    import torch
    torch.set_num_threads(4);tests=self_test(torch)
    name=('audit_check_' if args.self_test else 'audit_')+os.environ['SLURM_JOB_ID'];out=BASE/name;out.mkdir(parents=True,exist_ok=False);snapshot(out)
    save(out/'tests.json',tests)
    original.local.index(out,'Independent mixed-oracle audit',[('Tests','tests.json'),('Source ledger','source_hashes.json'),('Summary after audit','summary.json')])
    if args.self_test:print(json.dumps(dict(passed=True,source_ledger=str(out/'source_hashes.json'))),flush=True)
    else:
        need(args.source_ledger is not None and read(args.source_ledger)==sources(),'Require previously frozen audit source ledger')
        audit(torch,args,out)
        need(read(args.source_ledger)==sources(),'Audit sources changed during run')


if __name__=='__main__':main()

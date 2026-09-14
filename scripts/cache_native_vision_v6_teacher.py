"""V6 frozen teacher prefills and independent CPU cache/quality publication.

GPU profile:16 fixed training pairs, one original image and original count
question, no generation. GPU shards cover every9980 unique pair exactly once.
CPU merge verifies provenance and recomputes probabilities/quality on all18800
original training frame occurrences. Labels never supply a KD target. A failed
quality gate is retained, and the published cache is explicitly ineligible.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts.stage_native_vision_v6_teacher import (
    CATEGORIES,DATA,MODEL,OUTPUT,SEED,actual_layout,model_metadata,need,object_sha,
    prepared,read,save,sha,snapshot,source_hashes,verify_inputs,verify_plan,
)


def probabilities(logit0,logit1,log_normalizer):
    need(all(math.isfinite(x) for x in (logit0,logit1,log_normalizer)),'Nonfinite raw teacher values')
    need(log_normalizer>=max(logit0,logit1)-1e-10,'Normalizer below a count logit')
    p0,p1=math.exp(logit0-log_normalizer),math.exp(logit1-log_normalizer)
    other=1-math.fsum((p0,p1))
    need(other>=-1e-10,'Count mass exceeds full-vocabulary mass')
    return [p0,p1,max(0.,other)]


def conditional_one(logit0,logit1):
    difference=logit1-logit0
    if difference>=0:return 1/(1+math.exp(-difference))
    x=math.exp(difference);return x/(1+x)


def quality(rows,labels,weight_key):
    total=correct=mass=brier=nll=full_correct=0.
    category_counts={cat:dict(total=0,correct=0) for cat in CATEGORIES}
    bins=[dict(weight=0.,confidence=0.,correct=0.) for _ in range(10)]
    confusion=dict(tp=0,fn=0,fp=0,tn=0)
    for pid,row in rows.items():
        label=labels[pid];weight=1 if weight_key is None else label[weight_key]
        gold=label['gold'];prediction=int(row['logit1']>row['logit0'])
        p=conditional_one(row['logit0'],row['logit1']);prob=row['probabilities'];hit=prediction==gold
        total+=weight;correct+=weight*hit;mass+=weight*(prob[0]+prob[1])
        brier+=weight*(p-gold)**2
        # Stable conditional NLL, including extreme logits whose exp underflows.
        gap=(row['logit0']-row['logit1']) if gold else (row['logit1']-row['logit0'])
        nll+=weight*(max(0.,gap)+math.log1p(math.exp(-abs(gap))))
        full_correct+=weight*(max(range(3),key=lambda i:prob[i])==gold)
        category_counts[label['category']]['total']+=weight
        category_counts[label['category']]['correct']+=weight*hit
        confusion[('tp' if gold else 'fp') if prediction else ('fn' if gold else 'tn')]+=weight
        confidence=max(p,1-p);entry=bins[min(9,int(confidence*10))]
        entry['weight']+=weight;entry['confidence']+=weight*confidence;entry['correct']+=weight*hit
    need(total>0 and confusion['tp']+confusion['fn']>0 and confusion['tn']+confusion['fp']>0,
         'Quality audit requires positive/negative examples')
    tpr=confusion['tp']/(confusion['tp']+confusion['fn'])
    tnr=confusion['tn']/(confusion['tn']+confusion['fp'])
    for item in category_counts.values():
        need(item['total']>0,'Quality category missing');item['accuracy']=item['correct']/item['total']
    ece=sum(abs(item['confidence']-item['correct']) for item in bins)/total
    return dict(weighting='deduplicated_pair' if weight_key is None else weight_key,total_weight=total,
                accuracy=correct/total,conditional_balanced_accuracy=(tpr+tnr)/2,
                conditional_prediction_rule='logit1 > logit0; exact ties predict0',confusion=confusion,
                categories=category_counts,mean_numeric_mass=mass/total,conditional_brier=brier/total,
                conditional_nll=nll/total,three_category_accuracy=full_correct/total,
                conditional_ece_10_bins=ece,calibration_bins=bins)


def quality_gate(result):
    criteria=dict(conditional_balanced_accuracy=result['conditional_balanced_accuracy']>=.95,
                  each_category_accuracy=all(result['categories'][cat]['accuracy']>=.90 for cat in CATEGORIES),
                  mean_numeric_mass=result['mean_numeric_mass']>=.90)
    return dict(passed=all(criteria.values()),criteria=criteria,
                measurements=dict(conditional_balanced_accuracy=result['conditional_balanced_accuracy'],
                                  category_accuracy={cat:result['categories'][cat]['accuracy'] for cat in CATEGORIES},
                                  mean_numeric_mass=result['mean_numeric_mass']),
                thresholds=dict(conditional_balanced_accuracy=.95,each_category_accuracy=.90,mean_numeric_mass=.90))


def quality_self_test():
    need(probabilities(0.,0.,math.log(4.))==[.25,.25,.5],'Full-vocabulary other category lost')
    tiny=probabilities(0.,1.,1000.)
    need(tiny==[0.,0.,1.] and conditional_one(0.,1.)>.7,'Conditional confidence hid other mass')
    rows={};labels={}
    for i,cat in enumerate(CATEGORIES):
        gold=int(cat=='positive');a,b=(0.,8.) if gold else (8.,0.)
        norm=math.log(math.exp(a)+math.exp(b)+1.)
        rows[str(i)]=dict(logit0=a,logit1=b,log_normalizer=norm,probabilities=probabilities(a,b,norm))
        labels[str(i)]=dict(category=cat,gold=gold,unique_scene_occurrences=i+1)
    q=quality(rows,labels,'unique_scene_occurrences')
    need(q['total_weight']==10 and q['conditional_balanced_accuracy']==1 and quality_gate(q)['passed'],
         'Weighted perfect teacher should pass')
    boundary=dict(q,conditional_balanced_accuracy=.95,mean_numeric_mass=.90,
                  categories={cat:dict(accuracy=.90) for cat in CATEGORIES})
    need(quality_gate(boundary)['passed'],'Registered inclusive quality boundaries differ')
    for key in ('conditional_balanced_accuracy','mean_numeric_mass'):
        bad=dict(boundary);bad[key]-=1e-8;need(not quality_gate(bad)['passed'],'Below-boundary quality accepted')
    bad=dict(boundary,categories={cat:dict(accuracy=.90) for cat in CATEGORIES})
    bad['categories']['char_only']['accuracy']=.89
    need(not quality_gate(bad)['passed'],'Hard-negative category failure accepted')
    return dict(passed=True,tests=7)


def read_rows(path):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    need(len(rows)==len({r['pair_id'] for r in rows}),'Duplicate teacher pair output')
    return {row['pair_id']:row for row in rows}


def verify_row(row,pair,plan):
    need(row['pair_id']==pair['pair_id'] and row['image_sha256']==pair['image_sha256'] and
         row['question_sha256']==object_sha(pair['question']) and row['question']==pair['question'] and
         row['layout_id']==pair['layout_id'] and row['input_ids_sha256']==plan['layouts'][pair['layout_id']]['input_ids_sha256'] and
         row['prompt_tokens']==plan['layouts'][pair['layout_id']]['prompt_tokens'] and
         row['count_token_ids']==plan['count_token_ids'],'Teacher row/input/token identity differs')
    expected=probabilities(row['logit0'],row['logit1'],row['log_normalizer'])
    need(len(row['probabilities'])==3 and all(math.isfinite(x) and x>=0 for x in row['probabilities']) and
         all(math.isclose(x,y,rel_tol=1e-10,abs_tol=1e-12) for x,y in zip(expected,row['probabilities'])) and
         math.isclose(sum(row['probabilities']),1.,rel_tol=0,abs_tol=1e-10),'Teacher probability normalization differs')
    need(row['native_model_forwards']==row['native_visual_forwards']==1 and
         all(math.isfinite(row[k]) and row[k]>=0 for k in ('preparation_seconds','forward_seconds','pair_seconds')),
         'Teacher forward/count/cost integrity failed')


def verify_profile(directory,plan_path,plan):
    directory=Path(directory);summary=read(directory/'summary.json');config=read(directory/'config.json')
    need(summary['completed'] is True and summary['computational_integrity_passed'] is True and
         summary['profile'] is True and summary['cost_gate']['passed'] is True,'A successful matching software profile is required')
    need(summary['plan_sha256']==config['plan_sha256']==sha(plan_path) and
         summary['source_sha256']==config['source_sha256']==plan['source_sha256'],'Profile source/plan differs')
    need(sha(summary['rows_file'])==summary['rows_sha256'],'Profile raw rows changed')
    need(config['model']==plan['model'] and config['processor']==plan['processor'] and config['runtime']==plan['runtime'],
         'Profile model/runtime identity differs')
    need(read(directory/'source_hashes.json')==plan['source_sha256'] and
         all(sha(directory/'source'/name.replace('/','_'))==value for name,value in plan['source_sha256'].items()),
         'Profile source snapshot differs')
    need(sha(directory/'plan.json')==sha(plan_path),'Profile frozen plan copy differs')
    rows=read_rows(summary['rows_file']);need(list(rows)==plan['profile_pair_ids'],'Profile pair coverage/order differs')
    for pid,row in rows.items():verify_row(row,plan['pairs'][pid],plan)
    values=list(rows.values())
    projected=summary['model_load_seconds']+sum(r['pair_seconds'] for r in values[:4])+1.25*2495*max(r['pair_seconds'] for r in values[4:])+30
    need(math.isfinite(projected) and projected<=900 and math.isclose(projected,summary['cost_gate']['projected_shard_seconds'],
         rel_tol=1e-10,abs_tol=1e-8),'Profile throughput projection differs/fails')
    return dict(directory=str(directory.resolve()),summary_sha256=sha(directory/'summary.json'))


def gpu_run(args):
    plan=verify_plan(args.plan);quality_self_test()
    profile=args.profile
    need(profile or args.shard in range(4),'GPU run requires profile or shard0..3')
    profile_record=None if profile else verify_profile(args.profile_directory,args.plan,plan)
    ids=plan['profile_pair_ids'] if profile else plan['shards'][args.shard]
    job=os.environ['SLURM_JOB_ID'];name=f'profile_{job}' if profile else f'shard{args.shard}_{job}'
    out=OUTPUT/name;out.mkdir(parents=True,exist_ok=False);snapshot(out)
    data_out=DATA/name;data_out.mkdir(parents=True,exist_ok=False)
    (out/'plan.json').write_bytes(Path(args.plan).read_bytes())
    config=dict(schema_version=1,plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),
                source_sha256=plan['source_sha256'],profile=profile,shard=None if profile else args.shard,
                profile_provenance=profile_record,pair_ids=ids,slurm_job_id=job,
                model=plan['model'],processor=plan['processor'],runtime=plan['runtime'],
                operation='one frozen isolated-image native prefill; no generation or labels in inputs')
    save(out/'config.json',config)
    (out/'INDEX.md').write_text('# V6 teacher GPU artifacts\n\n- [Configuration](config.json)\n- [Summary after completion](summary.json)\n- [Frozen plan](plan.json)\n- [Source snapshots](source/)\n')
    (data_out/'INDEX.md').write_text('# V6 raw teacher outputs\n\n- [Every processed pair](rows.jsonl)\n')
    import torch
    import transformers
    from scripts.probe_native_vision_v2_prefix import fingerprint
    from gnnformer.runtime import load_runtime,move_to_device
    need(torch.cuda.is_available(),'Use a Slurm GPU allocation')
    need(str(torch.__version__)==plan['runtime']['torch_version'] and
         str(transformers.__version__)==plan['runtime']['transformers_version'],'Runtime version differs')
    torch.set_num_threads(4);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    started=time.perf_counter();runtime=load_runtime(str(MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda')
    model=runtime.model;model.requires_grad_(False);model.eval()
    need(not any(p.requires_grad for p in model.parameters()),'Teacher is not frozen')
    need(fingerprint(runtime.processor,str(transformers.__version__))==plan['processor'],'Processor fingerprint differs')
    torch.cuda.synchronize();load_seconds=time.perf_counter()-started;torch.cuda.reset_peak_memory_stats()
    vision=getattr(model,'visual',None)
    if vision is None:vision=getattr(getattr(model,'model',None),'visual',None)
    need(vision is not None,'Native visual module unavailable')
    counters=dict(model=0,vision=0)
    def model_hook(*_):counters['model']+=1
    def vision_hook(*_):counters['vision']+=1
    hooks=[model.register_forward_pre_hook(model_hook),vision.register_forward_pre_hook(vision_hook)]
    rows=[];raw_path=data_out/'rows.jsonl'
    try:
        with raw_path.open('x') as stream:
            for index,pid in enumerate(ids):
                pair=plan['pairs'][pid];begin=time.perf_counter();path=Path(pair['image_path'])
                need(path.stat().st_size==pair['image_bytes'] and sha(path)==pair['image_sha256'],'Teacher source image changed')
                inputs=prepared(runtime.processor,pair);verify_inputs(inputs,pair,plan)
                prep=time.perf_counter()-begin;item=move_to_device(inputs,runtime.device)
                before=dict(counters);torch.cuda.synchronize();forward_start=time.perf_counter()
                with torch.inference_mode():output=model(**item,use_cache=False,logits_to_keep=1)
                torch.cuda.synchronize();forward_seconds=time.perf_counter()-forward_start
                need(tuple(output.logits.shape[:2])==(1,1),'Expected one next-token logit vector')
                logits=output.logits[0,0].double();need(bool(torch.isfinite(logits).all()),'Nonfinite full-vocabulary logits')
                token_ids=plan['count_token_ids'];a=float(logits[token_ids['0']]);b=float(logits[token_ids['1']])
                normalizer=float(torch.logsumexp(logits,dim=0));top=int(logits.argmax())
                row=dict(pair_id=pid,image_sha256=pair['image_sha256'],question=pair['question'],
                         question_sha256=object_sha(pair['question']),layout_id=pair['layout_id'],
                         input_ids_sha256=plan['layouts'][pair['layout_id']]['input_ids_sha256'],
                         prompt_tokens=int(inputs['input_ids'].shape[1]),count_token_ids=token_ids,
                         logit0=a,logit1=b,log_normalizer=normalizer,probabilities=probabilities(a,b,normalizer),
                         top1_id=top,top1_text=runtime.tokenizer.decode([top],skip_special_tokens=False),
                         native_model_forwards=counters['model']-before['model'],
                         native_visual_forwards=counters['vision']-before['vision'],
                         preparation_seconds=prep,forward_seconds=forward_seconds,pair_seconds=time.perf_counter()-begin)
                verify_row(row,pair,plan);rows.append(row)
                stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                if profile or index%100==0:print(json.dumps(dict(completed=index+1,total=len(ids),pair_id=pid)),flush=True)
                del inputs,item,output,logits
    finally:
        for hook in hooks:hook.remove()
    need(verify_plan(args.plan)==plan,'Plan changed while caching')
    summary=dict(schema_version=1,completed=True,computational_integrity_passed=True,profile=profile,
                 shard=None if profile else args.shard,pair_count=len(rows),plan_sha256=sha(args.plan),
                 source_sha256=plan['source_sha256'],rows_file=str(raw_path),rows_sha256=sha(raw_path),
                 model_load_seconds=load_seconds,pair_seconds=sum(r['pair_seconds'] for r in rows),
                 forward_seconds=sum(r['forward_seconds'] for r in rows),total_seconds=time.perf_counter()-started,
                 peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
                 peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(),gpu=torch.cuda.get_device_name(0),
                 slurm_job_id=job,source_profile=profile_record,
                 quality_note='No teacher accuracy gate is applied to the16software inputs; full training gate at CPU merge')
    if profile:
        projected=load_seconds+sum(r['pair_seconds'] for r in rows[:4])+1.25*2495*max(r['pair_seconds'] for r in rows[4:])+30
        summary['cost_gate']=dict(passed=projected<=900,projected_shard_seconds=projected,cap_seconds=900,
                                  warmup_pairs=4,steady_pairs=12,margin=1.25,reserve_seconds=30,
                                  definition=plan['profile_cost_rule'])
    save(out/'summary.json',summary)
    print(json.dumps(dict(output=str(out),data_output=str(data_out),completed=True,seconds=summary['total_seconds'])),flush=True)


def merge(args):
    plan=verify_plan(args.plan);quality_self_test()
    need(len(args.shard_directories)==4,'Merge requires exactly four completed shard directories')
    all_rows={};provenance=[];seen=set()
    for directory in args.shard_directories:
        directory=Path(directory);summary=read(directory/'summary.json');config=read(directory/'config.json')
        shard=summary['shard'];need(type(shard)is int and shard in range(4) and shard not in seen,'Duplicate/invalid shard')
        seen.add(shard)
        need(summary['completed'] is True and summary['computational_integrity_passed'] is True and summary['profile'] is False and
             config['profile'] is False and config['shard']==shard,'Incomplete or non-shard output')
        need(summary['source_sha256']==config['source_sha256']==plan['source_sha256'] and
             summary['plan_sha256']==config['plan_sha256']==sha(args.plan),'Shard source/plan differs')
        need(config['model']==plan['model'] and config['processor']==plan['processor'] and config['runtime']==plan['runtime'],
             'Shard model/runtime differs')
        need(sha(summary['rows_file'])==summary['rows_sha256'],'Shard raw outputs changed')
        profile_provenance=verify_profile(config['profile_provenance']['directory'],args.plan,plan)
        need(profile_provenance==config['profile_provenance']==summary['source_profile'],'Shard profile provenance differs')
        rows=read_rows(summary['rows_file']);need(list(rows)==plan['shards'][shard]==config['pair_ids'] and
             len(rows)==summary['pair_count']==2495,'Shard coverage/order differs')
        for pid,row in rows.items():
            verify_row(row,plan['pairs'][pid],plan);need(pid not in all_rows,'Overlapping shards');all_rows[pid]=row
        for field in ('pair_seconds','forward_seconds'):
            need(math.isclose(summary[field],sum(r[field] for r in rows.values()),rel_tol=1e-10,abs_tol=1e-8),
                 'Shard cost does not equal raw rows')
        provenance.append(dict(directory=str(directory.resolve()),summary_sha256=sha(directory/'summary.json'),
                               config_sha256=sha(directory/'config.json'),rows_file=summary['rows_file'],
                               rows_sha256=summary['rows_sha256'],slurm_job_id=summary['slurm_job_id'],
                               total_seconds=summary['total_seconds']))
    need(set(all_rows)==set(plan['pairs']) and len(all_rows)==9980,'Incomplete teacher target cache')
    labels=read(plan['audit_file'])['labels']
    need(set(labels)==set(all_rows) and sum(x['unique_scene_occurrences'] for x in labels.values())==18800 and
         sum(x['scheduled_occurrences'] for x in labels.values())==19440,'Teacher audit weighting differs')
    audits=dict(deduplicated=quality(all_rows,labels,None),
                original_training_occurrences=quality(all_rows,labels,'unique_scene_occurrences'),
                scheduled_training_occurrences=quality(all_rows,labels,'scheduled_occurrences'))
    gate=quality_gate(audits['original_training_occurrences'])
    job=os.environ['SLURM_JOB_ID'];data_out=DATA/f'merged_{job}';data_out.mkdir(parents=True,exist_ok=False)
    out=OUTPUT/f'merged_{job}';out.mkdir(parents=True,exist_ok=False);snapshot(out)
    # No gold, category, or correctness fields are copied into this training index.
    targets={pid:{key:row[key] for key in ('probabilities','image_sha256','question','question_sha256',
                'logit0','logit1','log_normalizer','layout_id','input_ids_sha256')} for pid,row in sorted(all_rows.items())}
    index=dict(schema_version=1,passed_quality_gate=gate['passed'],plan_file=str(Path(args.plan).resolve()),
               plan_sha256=sha(args.plan),source_sha256=plan['source_sha256'],model=plan['model'],
               processor=plan['processor'],runtime=plan['runtime'],source_files=plan['source_files'],
               image_processor_settings=plan['image_processor_settings'],resize=plan['resize'],
               quantization=plan['quantization'],attention=plan['attention'],temperature=plan['temperature'],
               targets=targets,scenes={sid:record['pair_ids'] for sid,record in plan['scenes'].items()},
               quality_gate=gate,shard_provenance=provenance,
               weighting='Reuse each cached target at every original scheduled frame presentation; no filtering')
    path=data_out/'teacher_cache.json';save(path,index);path.with_suffix('.sha256').write_text(sha(path)+'\n')
    canonical=DATA/'teacher_cache.json'
    selected_path=canonical if gate['passed'] else path
    summary=dict(schema_version=1,completed=True,passed_quality_gate=gate['passed'],quality_gate=gate,
                 quality=audits,teacher_index=str(selected_path),teacher_index_sha256=sha(path),
                 archived_teacher_index=str(path),
                 plan_sha256=sha(args.plan),source_sha256=plan['source_sha256'],shards=provenance,
                 teacher_pairs=9980,training_scenes=1540,original_frame_occurrences=18800,
                 scheduled_frame_occurrences=19440,
                 limitations=['First-token teacher probabilities, not generated whole-answer accuracy.',
                              'Quality labels are training-domain audits only; no evaluation teacher targets.',
                              'Deduplication retains original weights; finite rendered atoms recur.',
                              'A failed quality gate must stop the registered training block; every raw target is retained.'])
    save(out/'summary.json',summary)
    q=audits['original_training_occurrences']
    lines=['# V6 frozen local teacher quality','',f"Registered gate: **{'PASS' if gate['passed'] else 'FAIL'}**.",'',
           f"All9980unique pairs,1540training scenes,18800original frame occurrences retained.",'',
           f"Conditional0/1 balanced accuracy: {q['conditional_balanced_accuracy']:.6f}; mean full-vocabulary numeric mass: {q['mean_numeric_mass']:.6f}.",'',
           '| Category | Original frame occurrences | Conditional0/1 accuracy |','|---|---:|---:|']
    for cat,x in q['categories'].items():lines.append(f"| {cat} | {x['total']} | {x['accuracy']:.6f} |")
    lines+=['','This is training-only teacher feasibility, not native aggregation accuracy. No teacher targets enter dev/test.','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    (out/'INDEX.md').write_text('# V6 teacher publication\n\n- [Report](REPORT.md)\n- [Full quality/provenance](summary.json)\n- [Sources](source/)\n')
    (data_out/'INDEX.md').write_text('# V6 training-only target cache\n\n- [Teacher index](teacher_cache.json)\n- [Index checksum](teacher_cache.sha256)\n')
    need(verify_plan(args.plan)==plan,'Plan changed during CPU merge')
    if gate['passed']:
        with canonical.open('xb') as stream:stream.write(path.read_bytes())
        with canonical.with_suffix('.sha256').open('x') as stream:stream.write(sha(canonical)+'\n')
    print(json.dumps(dict(output=str(out),teacher_index=str(selected_path),passed_quality_gate=gate['passed'])),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--profile',action='store_true');modes.add_argument('--shard',type=int)
    modes.add_argument('--merge',action='store_true')
    parser.add_argument('--profile-directory',type=Path)
    parser.add_argument('--shard-directories',nargs='*',type=Path,default=[])
    args=parser.parse_args();need(bool(os.environ.get('SLURM_JOB_ID')),'Run all teacher work through Slurm')
    if args.merge:merge(args)
    else:
        need(args.profile or args.profile_directory is not None,'Shards require an explicit successful profile')
        gpu_run(args)


if __name__=='__main__':main()

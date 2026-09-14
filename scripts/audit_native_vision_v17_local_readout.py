"""No-fit V17 native local readout: CPU plan, one head-only GPU audit, CPU report."""
from __future__ import annotations
import argparse
import ast
from collections import Counter,defaultdict
from datetime import datetime
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
DATA=Path('/mnt/data/gabriele/gnn_transformer/v17_local_readout')
OUT=REPO/'outputs/native_aggregation_vlm/v17/local_readout'
CACHE=DATA.parent/'v10_parallel_local/feature_cache.json'
PROFILE=REPO/'outputs/native_aggregation_vlm/v10/features/profile_442123/summary.json'
ANALYSIS=REPO/'outputs/native_aggregation_vlm/v15/null_comparison_study/report_442711/analysis.json'
HEAD_REFERENCE=REPO/'outputs/native_aggregation_vlm/readout_capacity/run_441720/summary.json'
PROTOCOL='v17_frozen_native_local_readout_only'
POLICY=dict(protocol=PROTOCOL,calls=432,head_rows=29656,training_batches=152,training_rows=9512,
    profile_calls=8,profile_rows=288,remaining_calls=424,remaining_rows=29368,v15_calls=272,v15_local_rows=13056,
    v15_head_rows=19856,training_scene_occurrences=24624,training_epoch_occurrences=25488,
    v15_physical_occurrences=1632,v15_unique_pairs=1574,hidden_size=3584,vocab_size=152064,
    dtype='torch.float16',probability_dtype='torch.float64',tv_max=.02,top1_exact=True,
    per_job_gpu_seconds=600,campaign_gpu_seconds=900,timing_margin=1.25,timing_reserve=30,
    timing_model='affine_all8_call_max_plus_B64_row_max',timing_all_pilot_calls_retained=True,
    conditional_tie_prediction=0,first_original_prompt_only=True,no_VLM=True,no_vision=True,no_fit=True,
    no_generated_answers=True,no_training_release=True,QA_offline_only=True)
OWN=('scripts/audit_native_vision_v17_local_readout.py','slurm/native_vision_v17_local_readout_check.sbatch',
     'slurm/native_vision_v17_local_readout_run.sbatch','slurm/native_vision_v17_local_readout_report.sbatch',
     'docs/paper/NATIVE_AGGREGATION_VISION_V17_LOCAL_READOUT.md','scripts/probe_native_vision_v7_response_surface.py')


def need(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def oid(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def save(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
def atomic(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)
def bind(path,bindings,digest=None):
    path=Path(path).resolve();actual=sha(path);need(digest is None or actual==digest,'Source artifact changed: '+str(path))
    need(str(path) not in bindings or bindings[str(path)]==actual,'Conflicting source hash');bindings[str(path)]=actual
    return dict(file=str(path),sha256=actual)
def tensor_info(torch,value):
    x=value.detach().cpu().contiguous();return dict(shape=list(x.shape),dtype=str(x.dtype),sha256=hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest())
def sources():
    from scripts import evaluate_native_vision_v15_null_comparison as old
    from scripts import stage_native_vision_v10_features as stage
    return {**old.sources(),**stage.sources(),**{k:sha(REPO/k) for k in OWN}}
def snapshot(out):
    (out/'code').mkdir();frozen=sources()
    for name,digest in frozen.items():
        path=out/'code'/name.replace('/','_');path.write_bytes((REPO/name).read_bytes());need(sha(path)==digest,'Source changed during snapshot')
    save(out/'source_hashes.json',frozen)
    (out/'INDEX.md').write_text('# V17 local-readout audit\n\n[Summary](summary.json) · [Sources](source_hashes.json). No fit, VLM, vision, or generated aggregate answers.\n')
    return frozen

def check_source_copy(value,out):
    need(value==sources(),'Current V17 source differs from frozen plan')
    for name,digest in value.items():need(sha(Path(out)/'code'/name.replace('/','_'))==digest,'Archived source changed')

def parse_allocations(raw):
    rows=[];seen=set();events=[]
    terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED','SPECIAL_EXIT'}
    for line in raw.splitlines():
        if not line.strip():continue
        fields=line.split('|');need(len(fields)==9,'Malformed scheduler allocation row')
        job,name,partition,state,exit_code,elapsed,tres,start,end=fields
        if not name.startswith('v17_') or partition!='gpu':continue
        need(name=='v17_local_head' and job not in seen and elapsed.isdigit(),'Unknown/duplicate V17 GPU job');seen.add(job)
        need(state.split()[0] in terminal and re.fullmatch(r'[0-9]+:[0-9]+',exit_code),'Nonterminal/invalid V17 allocation')
        values=dict(x.split('=',1) for x in tres.split(',') if '=' in x);typed=[int(v) for k,v in values.items() if k.startswith('gres/gpu:')]
        gpus=int(values['gres/gpu']) if 'gres/gpu' in values else sum(typed)
        need(gpus in (0,1) and (not typed or sum(typed)==gpus),'Unexpected allocated GPUs')
        seconds=int(elapsed);need(seconds<=600,'V17 job exceeded600 seconds')
        if gpus and seconds:
            a,b=datetime.fromisoformat(start),datetime.fromisoformat(end);need(b>=a,'Reversed allocation interval');events.extend([(a,1),(b,-1)])
        rows.append(dict(job_id=job,job_name=name,state=state,exit_code=exit_code,gpus=gpus,elapsed_seconds=seconds,gpu_seconds=seconds*gpus,start=start,end=end))
    running=maximum=0
    for _,change in sorted(events,key=lambda x:(x[0],x[1])):running+=change;maximum=max(maximum,running)
    total=sum(r['gpu_seconds'] for r in rows);need(running==0 and maximum<=1 and total<=900,'V17 campaign/concurrency cap exceeded')
    return dict(jobs=rows,allocated_gpu_seconds=total,maximum_concurrent_gpus=maximum,all_failed_and_zero_allocations_retained=True)

def allocation_ledger(out):
    cmd=['sacct','-X','-u',os.environ['USER'],'--starttime=2026-09-01','--parsable2','--noheader',
         '--format=JobIDRaw,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,Start,End']
    result=subprocess.run(cmd,text=True,capture_output=True,check=True);path=Path(out)/'all_user_sacct.psv';path.write_text(result.stdout)
    return dict(parse_allocations(result.stdout),command=cmd,raw_file=str(path),raw_sha256=sha(path))

def scalar_probabilities(values,zero,one):
    need(values and all(math.isfinite(x) for x in values),'Nonfinite scalar logits')
    maximum=max(values);normalizer=maximum+math.log(math.fsum(math.exp(x-maximum) for x in values))
    p0,p1=math.exp(values[zero]-normalizer),math.exp(values[one]-normalizer)
    gap=values[one]-values[zero];conditional=1/(1+math.exp(-gap)) if gap>=0 else math.exp(gap)/(1+math.exp(gap))
    return dict(p0=p0,p1=p1,other=max(0.,1-math.fsum((p0,p1))),conditional_p1=conditional,
                conditional_prediction=int(values[one]>values[zero]),top1_id=max(range(len(values)),key=lambda i:values[i]))

def timing_projection(elapsed,rows):
    need(len(rows)==8 and Counter(x['batch_size'] for x in rows)==Counter({8:4,64:4}),
         'Timing pilot must retain all original four B8 and four B64 calls')
    need(math.isfinite(elapsed) and elapsed>0 and all(math.isfinite(x['seconds']) and x['seconds']>0 for x in rows),
         'Invalid measured timing')
    # Charge the largest entire observed call, including the cold first call,
    # to EVERY future call; add the worst B64 per-row slope independently.
    per_call=max(x['seconds'] for x in rows)
    per_row=max(x['seconds']/x['batch_size'] for x in rows if x['batch_size']==64)
    covered=all(per_call+per_row*x['batch_size']>=x['seconds'] for x in rows)
    need(covered,'Affine resource envelope must cover every pilot observation')
    projected=elapsed+1.25*(424*per_call+29368*per_row)+30
    return dict(passed=projected<=600,elapsed_to_gate_seconds=elapsed,
                timing_model='affine_all8_call_max_plus_B64_row_max',per_call_seconds=per_call,
                worst_B64_seconds_per_row=per_row,all_pilot_calls_retained=True,pilot_envelope_covers_all=covered,
                remaining_calls=424,remaining_rows=29368,margin=1.25,reserve_seconds=30,
                projected_total_seconds=projected,cap_seconds=600)

def self_test():
    p=scalar_probabilities([0.,0.,math.log(2.)],0,1)
    need(abs(p['p0']-.25)<1e-12 and abs(p['other']-.5)<1e-12 and p['conditional_prediction']==0,'Full mass or tie rule failed')
    p=scalar_probabilities([0.,1.,1000.],0,1);need(p['other']==1 and p['conditional_p1']>.7,'Conditional confidence hid other mass')
    need(scalar_probabilities([1.,0.,-1.],0,1)['top1_id']==0,'Raw vocabulary top1 changed')
    pilot=[dict(batch_size=b,seconds=.001*b) for b in [8,64]*4]
    need(timing_projection(1,pilot)['passed'] and not timing_projection(590,pilot)['passed'],'Timing gate not inclusive/fixed')
    cold=[dict(row) for row in pilot];cold[0]['seconds']=.5;bound=timing_projection(1,cold)
    expected=1+1.25*(424*.5+29368*.001)+30
    need(bound['per_call_seconds']==.5 and bound['worst_B64_seconds_per_row']==.001
         and bound['all_pilot_calls_retained'] and bound['pilot_envelope_covers_all']
         and abs(bound['projected_total_seconds']-expected)<1e-10,'Cold call must be charged to every future call')
    try:timing_projection(1,[dict(batch_size=8,seconds=.01) for _ in range(8)])
    except ValueError:pass
    else:raise AssertionError('Missing B64 pilot observations accepted')
    raw='1|v17_local_head|gpu|FAILED|1:0|3|gres/gpu:b200=1|2026-09-11T00:00:00|2026-09-11T00:00:03'
    need(parse_allocations(raw)['allocated_gpu_seconds']==3,'Failed typed GPU allocation omitted')
    for value in (raw+'\n'+raw,raw.replace('FAILED','RUNNING'),raw.replace('v17_local_head','v17_unknown')):
        try:parse_allocations(value)
        except ValueError:pass
        else:raise AssertionError('Invalid scheduler row accepted')
    # Identity weighting averages per-observation errors, not mean probabilities.
    rows=[dict(pair_id='a',correct=1),dict(pair_id='a',correct=0),dict(pair_id='b',correct=1)]
    counts=Counter(r['pair_id'] for r in rows);need(sum(r['correct']/counts[r['pair_id']] for r in rows)/2==.75,'Identity weights changed')
    failed=[dict(call=160,kind='v15',metrics=dict(tv=[.03],top1_equal=[True],maximum_tv=.03,passed=False))]
    need(close_values(failed,json.loads(json.dumps(failed))) and not close_values(failed,[dict(failed[0],kind='profile')]),'Failed binding record comparison lost strings/status')
    return dict(passed=True,head_calls=0,tests=['failed_binding_record_roundtrip','full_mass','other_is_not_unknown','tie_predicts0','raw_argmax','timing_bound','affine_cold_call_retained','exact_pilot_batch_mix','failed_typed_allocations','invalid_allocations','identity_weighting'])

def truth_for(record,bindings):
    path=Path(record['path'])/'qa.txt';bind(path,bindings,record['qa_sha256']);lines=path.read_text().splitlines()
    a,b=lines.index('question:'),lines.index('answer:');content=[x.strip() for x in lines[a+1:b] if x.strip()]
    states=[ast.literal_eval(x) for x in content if x.startswith('{')]
    need([x for x in content if not x.startswith('{')]==[record['question']] and int(lines[b+1])==record['gold']
         and oid(dict(states=states,question=record['question']))==record['content_sha256'],'QA content changed')
    need(len(states)==len(record['image_files'])==record['n_frames'],'QA image count differs');result=[]
    for i,(state,im) in enumerate(zip(states,record['image_files'])):
        need(state['step_id']==i+1 and len(state['rooms'])==1,'Physical frame identity changed')
        room,people=next(iter(state['rooms'].items()));need(len(people)==1,'Expected single character')
        c,r=people[0]==record['target_character'],room==record['target_room'];category='positive' if c and r else 'char_only' if c else 'room_only' if r else 'neither'
        bind(im['path'],bindings,im['sha256']);result.append(dict(gold=int(c and r),category=category,step=i+1,
            pair_id=oid([im['sha256'],record['question']]),image_sha256=im['sha256'],question=record['question']))
    need(sum(x['gold'] for x in result)==record['gold'],'Independent local semantic recount differs');return result

def prepare(args,out,frozen):
    import torch
    from scripts import evaluate_native_vision_v15_null_comparison as study
    from scripts import stage_native_vision_v10_features as feature
    from scripts import cache_native_vision_v10_features as worker
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    torch.set_num_threads(4);start=time.perf_counter();bindings={};tests=self_test();dest=DATA/out.name;dest.mkdir(parents=True,exist_ok=False)
    cache=read(CACHE);bind(CACHE,bindings);bind(CACHE.with_suffix('.sha256'),bindings)
    need(sha(CACHE)==CACHE.with_suffix('.sha256').read_text().strip() and cache['complete'] and cache['training_only'],'Incomplete V10 cache')
    fp=feature.verify_plan(cache['plan_file'],pixels=False);bind(cache['plan_file'],bindings,cache['plan_sha256'])
    for item in fp['source_files'].values():bind(item['path'],bindings,item['sha256'])
    profile=worker.verify_profile(PROFILE.parent,fp,cache['plan_file']);bind(PROFILE,bindings);bind(profile['raw_file'],bindings,profile['raw_sha256'])
    need(cache['model']==profile['model'] and cache['native_dtypes']==dict(norm='torch.float16',lm_head='torch.float16'),'Frozen native cache/head metadata differs')
    summary=read(ANALYSIS.parent/'summary.json');bind(ANALYSIS.parent/'summary.json',bindings)
    bind(ANALYSIS,bindings,summary['analysis_sha256']);analysis=read(ANALYSIS)
    need(all(summary[k] is True and analysis[k] is True for k in ('passed','completed','audit_passed')) and analysis['records']==272,'Completed V15 independent report required')
    study.check_sources(analysis['source_sha256'],ANALYSIS.parent);v15=study.verify_plan(analysis['plan_file']);bind(analysis['plan_file'],bindings)
    need(v15['model']==cache['model'] and v15['runtime']==fp['runtime'] and v15['processor']==fp['processor'],'V10/V15 native identity differs')
    for filename,digest in v15['artifact_bindings'].items():bind(filename,bindings,digest)
    for row in analysis['runs'].values():
        bind(Path(row['directory'])/'summary.json',bindings,row['summary_sha256']);bind(Path(row['directory'])/'config.json',bindings,row['config_sha256'])
    bind(analysis['rescored_file'],bindings,analysis['rescored_sha256']);records=read(analysis['rescored_file'])
    selected={r['key']:r for r in v15['selected_models']};need(len(selected)==4 and len(records)==272,'All selected models required')
    for s in selected.values():bind(s['selected']['checkpoint'],bindings,s['selected']['checkpoint_sha256'])
    head_ref=read(HEAD_REFERENCE);bind(HEAD_REFERENCE,bindings);head_identity=head_ref['states_identity'];need(head_identity['model']==cache['model'],'Head reference model differs')
    model=Path(cache['model']['path']);cfg=read(model/'config.json');cfg=cfg.get('text_config',cfg)
    index=read(model/'model.safetensors.index.json')['weight_map'];head_files={}
    for name,key in (('head','lm_head.weight'),('norm','model.norm.weight')):
        path=model/index[key];stat=path.stat();head_files[name]=dict(file=str(path),key=key,bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,inode=stat.st_ino,
            native_sha256=head_identity['native_'+name+'_weight_sha256'])
    norm_source=Path(inspect.getsourcefile(Qwen2RMSNorm));bind(norm_source,bindings)
    need(cfg['hidden_size']==3584 and cfg['vocab_size']==152064,'Head dimensions changed')
    all_calls=[];labels={};ownership=[]
    def add(kind,states,rows,reference=None,reference_rows=None,origin=None):
        need(states.ndim==3 and states.shape[1:]==(1,3584) and states.dtype==torch.float16 and bool(torch.isfinite(states).all()),'Prepared native state shape/dtype differs')
        number=len(all_calls);path=dest/f'input_{number:03d}.pt';blob=dict(schema_version=1,hidden=states.contiguous().clone())
        if reference is not None:
            need(reference.ndim==2 and reference.shape==(len(reference_rows),152064) and bool(torch.isfinite(reference).all()),'Reference shape differs')
            blob.update(reference=reference.clone(),reference_rows=reference_rows)
        torch.save(blob,path)
        all_calls.append(dict(call=number,kind=kind,batch_size=len(states),file=str(path),sha256=sha(path),hidden=tensor_info(torch,states),
            reference=tensor_info(torch,reference) if reference is not None else None,reference_rows=reference_rows or [],rows=rows,origin=origin))
    raw_blob=torch.load(profile['raw_file'],map_location='cpu',weights_only=True);raw=raw_blob['observations'];need(raw_blob['schema_version']==1 and len(raw)==8,'Original profile needs8 native batches')
    bind(profile['observations_file'],bindings,profile['observations_sha256'])
    for item,observation in zip(raw,read(profile['observations_file'])):
        need(item['phase']==observation['phase'] and item['case']==observation['case'] and item['feature_ids']==observation['identity']['feature_ids']
            and tensor_info(torch,item['states'])==observation['states'],'Profile raw state ownership differs')
        add('profile',item['states'].unsqueeze(1),[dict(role='binding_only',feature_id=fid) for fid in item['feature_ids']],
            item['native_logits'],list(range(len(item['states']))),dict(profile=str(PROFILE),phase=item['phase'],case=item['case']))
    del raw,raw_blob
    manifest=read(fp['source_files']['training_manifest']['path']);schedule=read(fp['source_files']['training_schedule']['path'])
    train_rows={r['sid']:r for block in manifest['splits'].values() for r in block['samples'] if r['split']=='train'}
    need(set(train_rows)==set(fp['scenes']) and len(train_rows)==1782 and len(schedule['epoch_slots'])==1836,'Training scene/schedule coverage differs')
    slot_weights=Counter(schedule['epoch_slots']);counts_unique=Counter();counts_epoch=Counter()
    for sid,row in sorted(train_rows.items()):
        truth=truth_for(row,bindings)
        for i,t in enumerate(truth):
            pid=t['pair_id'];label=dict(gold=t['gold'],category=t['category'],image_sha256=t['image_sha256'],question=t['question'])
            need(pid not in labels or labels[pid]==label,'Conflicting identity truth');labels[pid]=label;counts_unique[pid]+=1;counts_epoch[pid]+=slot_weights[sid]
            fid=fp['scenes'][sid]['local_feature_ids'][i][0]
            need(fp['features'][fid]['prefix_ids']==[] and fp['features'][fid]['pair_id']==pid,'Nonempty/misaligned local teacher state')
            ownership.append(dict(dataset='v10',sid=sid,n_frames=row['n_frames'],K=row['gold'],frame_index=i,feature_id=fid,epoch_weight=slot_weights[sid],**t))
    need(len(labels)==9512 and sum(counts_unique.values())==24624 and sum(counts_epoch.values())==25488,'Training identity/occurrence denominator differs')
    seen=set()
    for shard in cache['shards']:
        directory=Path(shard['directory']);ss=read(directory/'summary.json');bind(directory/'summary.json',bindings,shard['summary_sha256'])
        bind(ss['features_file'],bindings,ss['features_sha256']);bind(ss['observations_file'],bindings,ss['observations_sha256'])
        blob=torch.load(ss['features_file'],map_location='cpu',weights_only=True);observations=read(ss['observations_file']);offset=0
        need(blob['feature_ids']==fp['shards'][shard['shard']] and tensor_info(torch,blob['states'])==ss['states'],'Frozen shard states differ')
        for ob in observations:
            count=ob['batch_size'];ids=ob['identity']['feature_ids'];part=blob['states'][offset:offset+count];need(blob['feature_ids'][offset:offset+count]==ids and tensor_info(torch,part)==ob['states'],'Original native batch ordering differs')
            if ob['phase']=='local_empty':
                rows=[]
                for j,fid in enumerate(ids):
                    desc=fp['features'][fid];need(desc['prefix_ids']==[] and fid not in seen,'Duplicate/nonempty training feature');seen.add(fid)
                    loc=cache['features'][fid];need(loc==dict(file=ss['features_file'],file_sha256=ss['features_sha256'],row=offset+j,state_sha256=tensor_info(torch,part[j])['sha256']),'Original feature locator/hash differs')
                    rows.append(dict(role='actual',dataset='v10',feature_id=fid,pair_id=desc['pair_id'],state_sha256=loc['state_sha256']))
                add('v10',part.unsqueeze(1),rows,origin=dict(shard=shard['shard'],case=ob['case'],source=ss['features_file']))
            offset+=count
        need(offset==len(blob['states']),'Shard observations incomplete');del blob
    need(seen==set(fp['groups']['local_empty']) and len(all_calls)==160,'Complete152 training batches required')
    v15_truth={r['record']['sid']:truth_for(r['record'],bindings) for r in v15['cases']};seen_records=set()
    for row in records:
        key=(row['core_key'],row['case_index'],row['mode']);need(key not in seen_records,'Duplicate V15 trajectory');seen_records.add(key)
        case=v15['cases'][row['case_index']];sample=case['record'];s=selected[row['core_key']]
        need(row['sid']==sample['sid'] and row['condition']==s['condition'] and row['seed']==s['seed'] and row['mode'] in ('learned','bank'),'V15 selected ownership differs')
        bind(row['raw_file'],bindings,row['raw_sha256']);blob=torch.load(row['raw_file'],map_location='cpu',weights_only=True)
        need(row['capture_identities']==study.audit_captures(torch,blob['captures'],row['generated_ids'],case,s,row['mode']),'Original native capture binding differs')
        cap=blob['captures'][0];n=sample['n_frames'];h=torch.cat([cap['actual_states'],cap['reference_states'],cap['fused_global'].unsqueeze(1)],dim=0)
        rows=[]
        for i,t in enumerate(v15_truth[sample['sid']]):
            label={k:t[k] for k in ('gold','category','image_sha256','question')};pid=t['pair_id']
            need(pid not in labels or labels[pid]==label,'Train/evaluation semantic conflict');labels[pid]=label
            meta=dict(role='actual',dataset='v15',core_key=row['core_key'],condition=row['condition'],seed=row['seed'],mode=row['mode'],
                case_index=row['case_index'],sid=row['sid'],family_id=sample['pair_id'],n_frames=n,K=sample['gold'],frame_index=i,
                pair_id=pid,state_sha256=tensor_info(torch,cap['actual_states'][i,0])['sha256'])
            rows.append(meta);ownership.append(dict(meta,**t))
        rows.extend(dict(role='reference',reference_index=i) for i in range(24));rows.append(dict(role='global_binding'))
        add('v15',h,rows,blob['raw_logits'][0:1], [n+24],dict(raw_file=row['raw_file'],raw_sha256=row['raw_sha256'],key=list(key)))
        del blob
    need(seen_records=={(k,i,m) for k in selected for i in range(34) for m in ('learned','bank')},'All V15 model/mode/cases required')
    need(len(all_calls)==432 and sum(c['batch_size'] for c in all_calls)==29656,'Fixed complete head inventory differs')
    label_path=dest/'offline_truth.json';save(label_path,dict(labels=labels,ownership=ownership,
        train_unique_occurrences=dict(counts_unique),train_epoch_occurrences=dict(counts_epoch),QA_offline_only=True))
    ledger=allocation_ledger(out);reserve=ledger['allocated_gpu_seconds']+600<=900
    plan=dict(schema_version=1,protocol=PROTOCOL,policy=POLICY,source_sha256=frozen,artifact_bindings=bindings,model=cache['model'],runtime=fp['runtime'],processor=fp['processor'],
        count_token_ids={k:fp['count_token_ids'][k] for k in ('0','1')},native_dtypes=cache['native_dtypes'],head_files=head_files,
        norm_source_file=str(norm_source),norm_source_sha256=sha(norm_source),rms_norm_eps=cfg['rms_norm_eps'],calls=all_calls,
        offline_truth=bind(label_path,{}),v15_analysis=bind(ANALYSIS,{}),v15_plan=bind(analysis['plan_file'],{}),selected_models=v15['selected_models'],
        prior_allocation=ledger,reserve_passed=reserve,tests=tests,no_model_loaded=True,no_head_calls=True,slurm_job_id=os.environ['SLURM_JOB_ID'])
    save(out/'plan.json',plan);(out/'plan.sha256').write_text(sha(out/'plan.json')+'\n')
    save(out/'summary.json',dict(passed=reserve,completed=True,protocol=PROTOCOL,plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),
        source_sha256=frozen,calls=432,head_rows=29656,tests=tests,reserve_passed=reserve,no_model_loaded=True,no_head_calls=True,seconds=time.perf_counter()-start))
    need(reserve,'V17 prior allocations plus600-second reserve exceed900');verify_plan(out/'plan.json',inputs=True)

def verify_plan(path,*,inputs=False,ancestors=True):
    path=Path(path).resolve();plan=read(path);summary=read(path.parent/'summary.json')
    need(path.is_relative_to(OUT) and plan['protocol']==PROTOCOL and plan['policy']==POLICY and summary['passed'] is True
        and summary['completed'] is True and summary['no_model_loaded'] is True and summary['no_head_calls'] is True
        and summary['plan_sha256']==sha(path)==path.with_suffix('.sha256').read_text().strip(),'Completed matching CPU plan required')
    check_source_copy(plan['source_sha256'],path.parent)
    need(len(plan['calls'])==432 and [r['call'] for r in plan['calls']]==list(range(432))
        and Counter(r['kind'] for r in plan['calls'])==Counter(profile=8,v10=152,v15=272)
        and sum(r['batch_size'] for r in plan['calls'])==29656
        and all(r['kind']=='profile' for r in plan['calls'][:8]),'Complete fixed call inventory changed')
    need(plan['reserve_passed'] is True and plan['prior_allocation']['allocated_gpu_seconds']+600<=900,'CPU campaign reserve failed')
    if ancestors:
        for filename,digest in plan['artifact_bindings'].items():need(sha(filename)==digest,'Upstream frozen artifact changed: '+filename)
        need(sha(plan['offline_truth']['file'])==plan['offline_truth']['sha256'],'Offline truth changed')
    for name,item in plan['head_files'].items():
        st=Path(item['file']).stat();need(st.st_size==item['bytes'] and st.st_mtime_ns==item['mtime_ns'] and st.st_ino==item['inode'],'Immutable head weight shard stat changed')
    need(sha(plan['norm_source_file'])==plan['norm_source_sha256'],'Installed native norm source changed')
    if inputs:
        for row in plan['calls']:need(sha(row['file'])==row['sha256'],'Prepared native hidden bundle changed')
    return plan

def load_native_head(torch,plan):
    from safetensors import safe_open
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    need(str(Path(inspect.getsourcefile(Qwen2RMSNorm)))==plan['norm_source_file'],'Wrong native RMS implementation')
    weights={};identity={}
    for name,item in plan['head_files'].items():
        with safe_open(item['file'],framework='pt',device='cpu') as archive:value=archive.get_tensor(item['key']).to(torch.float16).contiguous()
        identity[name]=tensor_info(torch,value);need(identity[name]['sha256']==item['native_sha256'],'Actual native FP16 '+name+' weight differs')
        weights[name]=value
    need(weights['head'].shape==(152064,3584) and weights['norm'].shape==(3584,),'Native readout dimensions differ')
    norm=Qwen2RMSNorm(3584,eps=plan['rms_norm_eps']).to(dtype=torch.float16)
    norm.load_state_dict({'weight':weights['norm']});norm.eval().requires_grad_(False)
    head=weights['head'].to('cuda').requires_grad_(False);norm=norm.to('cuda')
    return norm,head,identity

def binding_metrics(torch,logits,reference):
    need(logits.shape==reference.shape and logits.ndim==2 and bool(torch.isfinite(logits).all()) and bool(torch.isfinite(reference).all()),'Malformed binding logits')
    a,b=logits.double(),reference.to(logits.device).double();tv=(a.softmax(-1)-b.softmax(-1)).abs().sum(-1)*.5;same=a.argmax(-1)==b.argmax(-1)
    values=tv.cpu().tolist();top=same.cpu().tolist()
    return dict(tv=values,top1_equal=top,maximum_tv=max(values),passed=all(v<=.02 for v in values) and all(top))

def row_probabilities(torch,logits,token_ids):
    need(logits.ndim==2 and logits.shape[1]==152064 and bool(torch.isfinite(logits).all()),'Nonfinite/incorrect full vocabulary')
    values=logits.double();normalizer=torch.logsumexp(values,dim=-1);zero,one=token_ids['0'],token_ids['1']
    l0,l1=values[:,zero],values[:,one];p0=(l0-normalizer).exp();p1=(l1-normalizer).exp();other=1-p0-p1
    need(bool((other>=-1e-10).all()),'Numeric probability mass exceeds one')
    cond=(l1-l0).sigmoid();top=values.argmax(-1);group=torch.stack([p0,p1,other.clamp_min(0)],dim=-1).argmax(-1)
    fields=dict(logit0=l0,logit1=l1,log_normalizer=normalizer,p0=p0,p1=p1,other=other.clamp_min(0),numeric_mass=p0+p1,
        conditional_p1=cond,conditional_prediction=(l1>l0).long(),top1_id=top,three_category_prediction=group)
    cpu={k:v.cpu().tolist() for k,v in fields.items()}
    return [{k:cpu[k][i] for k in cpu} for i in range(len(logits))]

def run(args,out,frozen):
    import torch
    torch.set_num_threads(4);started=time.perf_counter()
    # Upstream proof was completed on CPU. Only current sources, immutable native
    # weights and actually consumed input bundles are revisited on allocated GPU.
    plan=verify_plan(args.plan,ancestors=False);need(torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA B200','One registered B200 required')
    need(str(torch.__version__)==plan['runtime']['torch_version'],'Native torch runtime changed')
    norm,head,initial=load_native_head(torch,plan);torch.cuda.synchronize();load_seconds=time.perf_counter()-started
    dest=DATA/out.name;dest.mkdir(parents=True,exist_ok=False);records=[];norm_calls=0;head_calls=0;failures=[];projection=None
    def counted(*_):
        nonlocal norm_calls
        norm_calls+=1
    handle=norm.register_forward_hook(counted)
    try:
        for item in plan['calls']:
            begin=time.perf_counter();need(sha(item['file'])==item['sha256'],'Consumed input bundle changed')
            blob=torch.load(item['file'],map_location='cpu',weights_only=True);h=blob['hidden']
            need(tensor_info(torch,h)==item['hidden'] and h.shape==(item['batch_size'],1,3584),'Consumed hidden shape/hash changed')
            with torch.inference_mode():
                logits=torch.nn.functional.linear(norm(h.to('cuda')),head);head_calls+=1
            need(logits.dtype==torch.float16 and logits.shape==(item['batch_size'],1,152064),'Native full head shape/dtype changed')
            metrics=None
            if item['reference'] is not None:
                need(tensor_info(torch,blob['reference'])==item['reference'] and blob['reference_rows']==item['reference_rows'],'Original reference logits changed')
                metrics=binding_metrics(torch,logits[blob['reference_rows'],0],blob['reference'])
                if not metrics['passed']:failures.append(dict(call=item['call'],kind=item['kind'],metrics=metrics))
            probabilities=row_probabilities(torch,logits[:,0],plan['count_token_ids'])
            native=logits.cpu();torch.cuda.synchronize();raw=dest/f'head_{item["call"]:03d}.pt'
            torch.save(dict(schema_version=1,logits=native),raw)
            record=dict(call=item['call'],kind=item['kind'],batch_size=item['batch_size'],input_file=item['file'],input_sha256=item['sha256'],
                raw_file=str(raw),raw_sha256=sha(raw),logits=tensor_info(torch,native),binding=metrics,probabilities=probabilities)
            # Include transfer, serialized raw outputs, hashes, and per-row accounting.
            record['seconds']=time.perf_counter()-begin;records.append(record);save(out/f'call_{item["call"]:03d}.json',record)
            atomic(out/'progress.json',dict(completed_calls=len(records),expected_calls=432,norm_calls=norm_calls,head_calls=head_calls,
                binding_failures=failures,seconds=time.perf_counter()-started))
            if item['call']==7:
                projection=timing_projection(time.perf_counter()-started,records);save(out/'timing_gate.json',projection)
                gate=dict(passed=not failures,completed=True,calls=8,rows=288,failures=failures,source_sha256=frozen)
                save(out/'initial_replay_gate.json',gate)
                if failures or not projection['passed']:
                    save(out/'summary.json',dict(protocol=PROTOCOL,passed=False,completed=False,plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),
                        source_sha256=frozen,initial_replay_gate=gate,timing_projection=projection,completed_calls=8,norm_calls=norm_calls,head_calls=head_calls,
                        VLM_calls=0,vision_calls=0,no_fit=True,slurm_job_id=os.environ['SLURM_JOB_ID'],seconds=time.perf_counter()-started))
                    raise SystemExit('Preserved initial replay/timing gate failure; no remaining calls permitted')
            if len(records)%16==0:print(json.dumps(dict(completed_calls=len(records),total=432,seconds=time.perf_counter()-started)),flush=True)
            del logits,native,blob
    finally:handle.remove()
    final=dict(norm=tensor_info(torch,norm.weight),head=tensor_info(torch,head));need(final==initial,'Frozen norm/head changed')
    need(norm_calls==head_calls==432 and sum(r['batch_size'] for r in records)==29656 and sources()==frozen,'Full fixed call/source coverage changed')
    save(out/'records.json',records)
    save(out/'summary.json',dict(schema_version=1,protocol=PROTOCOL,passed=not failures,completed=True,computational_integrity_passed=True,
        numerical_gate_passed=not failures,plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),source_sha256=frozen,
        records_file=str(out/'records.json'),records_sha256=sha(out/'records.json'),initial_weight_identity=initial,final_weight_identity=final,
        norm_calls=norm_calls,head_calls=head_calls,head_rows=29656,VLM_calls=0,vision_calls=0,no_fit=True,no_generated_answers=True,
        timing_projection=projection,load_seconds=load_seconds,binding_failures=failures,hardware=dict(gpu=torch.cuda.get_device_name(0),torch=str(torch.__version__),cuda=torch.version.cuda),
        seconds=time.perf_counter()-started,slurm_job_id=os.environ['SLURM_JOB_ID']))
    need(not failures,'All raw outputs preserved; native numerical binding failed')

def close_values(a,b):
    if isinstance(a,dict):return set(a)==set(b) and all(close_values(v,b[k]) for k,v in a.items())
    if isinstance(a,list):return len(a)==len(b) and all(close_values(x,y) for x,y in zip(a,b))
    if isinstance(a,(bool,int,str)) or a is None:return type(a) is type(b) and a==b
    return isinstance(b,(int,float)) and math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-10)

def aggregate(rows,weights=None):
    if weights is None:weights=[1.]*len(rows)
    need(len(rows)==len(weights) and rows and all(math.isfinite(w) and w>0 for w in weights),'Invalid metric weights')
    total=sum(weights);conf=defaultdict(float);full=defaultdict(float);triple=defaultdict(float);means=defaultdict(float)
    for r,w in zip(rows,weights):
        y=r['gold'];pred=r['conditional_prediction'];conf[f'{y}->{pred}']+=w
        full[f'{y}->{r["full_prediction"]}']+=w;triple[f'{y}->{r["three_category_prediction"]}']+=w
        for key in ('p0','p1','other','numeric_mass','conditional_p1'):means[key]+=w*r[key]
        means['conditional_correct']+=w*(pred==y);means['full_top1_correct']+=w*(r['full_prediction']==y)
        means['three_category_correct']+=w*(r['three_category_prediction']==y);means['full_top1_other']+=w*(r['full_prediction']==2);means['conditional_brier']+=w*(r['conditional_p1']-y)**2
        gap=r['logit0']-r['logit1'] if y else r['logit1']-r['logit0']
        means['conditional_nll']+=w*(max(0.,gap)+math.log1p(math.exp(-abs(gap))))
    class_totals={y:sum(w for r,w in zip(rows,weights) if r['gold']==y) for y in (0,1)}
    rates={str(y):conf[f'{y}->{y}']/class_totals[y] if class_totals[y] else None for y in (0,1)}
    return dict(rows=len(rows),weight=total,conditional_confusion=dict(conf),full_top1_confusion=dict(full),three_category_confusion=dict(triple),
        **{k:v/total for k,v in means.items()},class_accuracy=rates,
        conditional_balanced_accuracy=sum(rates.values())/2 if all(v is not None for v in rates.values()) else None,
        conditional_tpr=rates['1'],conditional_fpr=1-rates['0'] if rates['0'] is not None else None)

def report(args,out,frozen):
    import torch
    torch.set_num_threads(4);start=time.perf_counter();plan=verify_plan(args.plan,inputs=True)
    directory=Path(args.run_directory).resolve();summary=read(directory/'summary.json');check_source_copy(summary['source_sha256'],directory)
    need(directory.is_relative_to(OUT) and summary['protocol']==PROTOCOL and summary['completed'] is True
        and summary['computational_integrity_passed'] is True and summary['plan_sha256']==sha(args.plan)
        and summary['norm_calls']==summary['head_calls']==432 and summary['head_rows']==29656
        and summary['VLM_calls']==summary['vision_calls']==0 and summary['no_fit'] is True,'Complete no-fit head execution required')
    need(summary['initial_weight_identity']==summary['final_weight_identity'] and summary['hardware']['gpu']=='NVIDIA B200','Frozen actual weights/hardware differ')
    for key,identity in summary['initial_weight_identity'].items():need(identity['sha256']==plan['head_files'][key]['native_sha256'],'Loaded native weight identity differs')
    need(sha(summary['records_file'])==summary['records_sha256'],'GPU record ledger changed');records=read(summary['records_file'])
    need(len(records)==432 and summary['timing_projection']==read(directory/'timing_gate.json'),'GPU timing/source ledger differs')
    projection=timing_projection(summary['timing_projection']['elapsed_to_gate_seconds'],records[:8]);need(projection==summary['timing_projection'] and projection['passed'],'Timing gate differs')
    truth=read(plan['offline_truth']['file']);scored=[];bindings=[];failures=[];counts=Counter();v10_by_fid={}
    for item,record in zip(plan['calls'],records):
        need(record==read(directory/f'call_{item["call"]:03d}.json') and record['call']==item['call'] and record['kind']==item['kind']
             and record['batch_size']==item['batch_size'] and record['input_sha256']==item['sha256'],'Per-call ownership record differs')
        need(sha(record['raw_file'])==record['raw_sha256'],'Full vocabulary raw output changed')
        raw=torch.load(record['raw_file'],map_location='cpu',weights_only=True);logits=raw['logits']
        need(raw['schema_version']==1 and logits.dtype==torch.float16 and logits.shape==(item['batch_size'],1,152064)
            and tensor_info(torch,logits)==record['logits'],'Lossless native output identity/shape differs')
        probabilities=row_probabilities(torch,logits[:,0],plan['count_token_ids']);need(close_values(probabilities,record['probabilities']),'Independent full-vocabulary accounting differs')
        if item['reference'] is not None:
            inp=torch.load(item['file'],map_location='cpu',weights_only=True);need(tensor_info(torch,inp['hidden'])==item['hidden'] and tensor_info(torch,inp['reference'])==item['reference'],'CPU reference input changed')
            metric=binding_metrics(torch,logits[item['reference_rows'],0],inp['reference'])
            need(close_values(metric,record['binding']),'Independent native binding metrics differ')
            bindings.append(dict(call=item['call'],kind=item['kind'],metrics=metric))
            if not metric['passed']:failures.append(dict(call=item['call'],kind=item['kind'],metrics=metric))
        for index,(meta,p) in enumerate(zip(item['rows'],probabilities)):
            if meta['role']!='actual':continue
            label=truth['labels'][meta['pair_id']];top=p['top1_id'];zero,one=plan['count_token_ids']['0'],plan['count_token_ids']['1']
            row=dict(meta,**label,**p,call=item['call'],row=index,full_prediction=0 if top==zero else 1 if top==one else 2)
            scored.append(row);counts[meta['dataset']]+=1
            if meta['dataset']=='v10':need(meta['feature_id'] not in v10_by_fid,'Duplicate training feature');v10_by_fid[meta['feature_id']]=row
        if (item['call']+1)%32==0:print(json.dumps(dict(rescored_calls=item['call']+1,total=432,seconds=time.perf_counter()-start)),flush=True)
        del raw,logits
    need(counts==Counter(v10=9512,v15=13056) and len(bindings)==280 and sum(len(r['metrics']['tv']) for r in bindings)==560,'All original local and binding rows required')
    need(close_values(failures,summary['binding_failures']) and summary['numerical_gate_passed']==(not failures),'Preserved numerical status differs')
    training=[r for r in scored if r['dataset']=='v10'];evaluation=[r for r in scored if r['dataset']=='v15']
    train=dict(distinct_identity=aggregate(training),unique_scene_occurrences=aggregate(training,[truth['train_unique_occurrences'][r['pair_id']] for r in training]),
        weighted_epoch_occurrences=aggregate(training,[truth['train_epoch_occurrences'][r['pair_id']] for r in training]))
    need(train['unique_scene_occurrences']['weight']==24624 and train['weighted_epoch_occurrences']['weight']==25488,'Training occurrence denominator changed')
    # Equal-scene averages are distinct from frame-occurrence weighting.
    train_occ=[dict(v10_by_fid[r['feature_id']],**{k:r[k] for k in ('sid','n_frames','K','frame_index','epoch_weight')})
               for r in truth['ownership'] if r['dataset']=='v10']
    need(len(train_occ)==24624,'Original unique-scene occurrence ownership differs')
    train['equal_unique_scene']=aggregate(train_occ,[1/r['n_frames'] for r in train_occ])
    train['equal_weighted_epoch_scene']=aggregate(train_occ,[r['epoch_weight']/r['n_frames'] for r in train_occ])
    need(math.isclose(train['equal_unique_scene']['weight'],1782,abs_tol=1e-7) and math.isclose(train['equal_weighted_epoch_scene']['weight'],1836,abs_tol=1e-7),'Equal-scene denominators changed')
    training_ids={r['pair_id'] for r in training}
    partitions=[]
    for field in ('n_frames','K'):
        for value in sorted({r[field] for r in train_occ}):
            sub=[r for r in train_occ if r[field]==value]
            partitions.append(dict(dataset='v10',field=field,value=value,unique_scene_occurrences=aggregate(sub),weighted_epoch_occurrences=aggregate(sub,[r['epoch_weight'] for r in sub])))
    for dataset,rows in [('v10',training),('v15',evaluation)]:
        for field in ('category','gold')+ (('condition','seed','mode','n_frames','K') if dataset=='v15' else ()):
            for value in sorted({r[field] for r in rows}):
                sub=[r for r in rows if r[field]==value];partitions.append(dict(dataset=dataset,field=field,value=value,metrics=aggregate(sub)))
    # Native execution partitions preserve all four models and both modes separately.
    cells=[];scenes=[];identity_groups=[]
    for key in sorted({(r['core_key'],r['mode'],r['n_frames']) for r in evaluation}):
        rows=[r for r in evaluation if (r['core_key'],r['mode'],r['n_frames'])==key];ic=Counter(r['pair_id'] for r in rows)
        cells.append(dict(core_key=key[0],mode=key[1],n_frames=key[2],occurrence=aggregate(rows),
            distinct_identity=aggregate(rows,[1/ic[r['pair_id']] for r in rows]),unique_identities=len(ic)))
        for k in range(17):
            part=[r for r in rows if r['K']==k];need(len(part)==key[2],'Every K/case/frame must remain')
            scenes.append(dict(core_key=key[0],mode=key[1],n_frames=key[2],K=k,sid=part[0]['sid'],metrics=aggregate(part)))
        for category in ('positive','char_only','room_only','neither'):
            part=[r for r in rows if r['category']==category]
            if part:partitions.append(dict(dataset='v15',core_key=key[0],mode=key[1],n_frames=key[2],field='category',value=category,metrics=aggregate(part)))
        for phase in ('Step_le16','Step_gt16'):
            part=[r for r in rows if (r['frame_index']<16)==(phase=='Step_le16')]
            partitions.append(dict(dataset='v15',core_key=key[0],mode=key[1],n_frames=key[2],field='displayed_Step_support',value=phase,metrics=aggregate(part)))
    by_identity=defaultdict(list)
    for r in evaluation:by_identity[r['pair_id']].append(r)
    need(len(by_identity)==1574,'V15 distinct identity inventory changed')
    for pid,rows in sorted(by_identity.items()):
        identity_groups.append(dict(pair_id=pid,observations=len(rows),shared_with_training=pid in training_ids,state_hashes=sorted({r['state_sha256'] for r in rows}),
            conditional_predictions=sorted({r['conditional_prediction'] for r in rows}),full_predictions=sorted({r['full_prediction'] for r in rows}),
            conditional_p1_min=min(r['conditional_p1'] for r in rows),conditional_p1_max=max(r['conditional_p1'] for r in rows),metrics=aggregate(rows)))
    repeated=dict(occurrence=aggregate(evaluation),identity_equal_weight=aggregate(evaluation,[1/len(by_identity[r['pair_id']]) for r in evaluation]),
        physical_scene_occurrences=1632,execution_occurrences=13056,unique_identities=1574,shared_training_identities=len(set(by_identity)&training_ids),models_and_modes_not_independent_replicates=True)
    need(repeated['shared_training_identities']==226,'Shared training/evaluation identity inventory changed')
    for shared in (True,False):
        sub=[r for r in evaluation if (r['pair_id'] in training_ids)==shared]
        partitions.append(dict(dataset='v15',field='shared_training_identity',value=shared,metrics=aggregate(sub)))
    ledger=allocation_ledger(out);job=str(summary['slurm_job_id']);matches=[r for r in ledger['jobs'] if r['job_id']==job]
    need(len(matches)==1 and matches[0]['gpus']==1,'Actual completed audit allocation missing')
    execution_ok=matches[0]['state']=='COMPLETED' and matches[0]['exit_code']=='0:0'
    need(execution_ok or (bool(failures) and summary['passed'] is False and matches[0]['state']=='FAILED'),'Scheduler status does not match retained numerical failure')
    save(out/'rescored_local_rows.json',scored);save(out/'binding_replays.json',bindings);save(out/'identity_groups.json',identity_groups)
    result=dict(schema_version=1,protocol=PROTOCOL,passed=not failures,completed=True,source_sha256=frozen,plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),
        gpu_summary_file=str(directory/'summary.json'),gpu_summary_sha256=sha(directory/'summary.json'),head_calls=432,head_rows=29656,local_rows=22568,
        training=train,v15=repeated,cells=cells,scenes=scenes,partitions=partitions,binding_failures=failures,allocation=ledger,
        rescored_file=str(out/'rescored_local_rows.json'),rescored_sha256=sha(out/'rescored_local_rows.json'),
        binding_file=str(out/'binding_replays.json'),binding_sha256=sha(out/'binding_replays.json'),
        identities_file=str(out/'identity_groups.json'),identities_sha256=sha(out/'identity_groups.json'),
        no_fit=True,VLM_calls=0,vision_calls=0,report_head_calls=0,no_generated_aggregate_answers=True,no_training_release=True,
        limitation='Isolated native first-token readout on saved states; no whole-answer accuracy, information-absence proof, fresh efficacy, or reasoning claim.')
    save(out/'analysis.json',result)
    lines=['# V17 native local-readout audit','',f'Binding numerical gate: {not failures}. All9512 training identities and13056 V15 local execution rows retained.',
        'Other is vocabulary mass outside digit0/1, not a semantic class. Isolated predictions are never substituted for native aggregate answers.','',
        '| Model | Mode | N | Conditional correct | Native top1 correct | Numeric mass |','|---|---|---:|---:|---:|---:|']
    for row in cells:
        m=row['occurrence'];lines.append(f'| {row["core_key"]} | {row["mode"]} | {row["n_frames"]} | {m["conditional_correct"]:.6f} | {m["full_top1_correct"]:.6f} | {m["numeric_mass"]:.6f} |')
    lines+=['',result['limitation']];(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    save(out/'summary.json',dict(passed=not failures,completed=True,protocol=PROTOCOL,analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),
        source_sha256=frozen,local_rows=22568,head_calls=0,VLM_calls=0,vision_calls=0,no_fit=True,allocated_gpu_seconds=ledger['allocated_gpu_seconds'],seconds=time.perf_counter()-start))
    need(not failures,'All metrics retained, but native binding gate failed')

def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    for name in ('check','run','report'):group.add_argument('--'+name,action='store_true')
    parser.add_argument('--plan',type=Path);parser.add_argument('--run-directory',type=Path);args=parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID'),'All preparation/tensor/head/report work requires Slurm')
    need(os.environ.get('SLURM_JOB_PARTITION')==('gpu' if args.run else 'cpu'),'Wrong Slurm partition')
    if not args.run:need(not os.environ.get('SLURM_JOB_GPUS'),'CPU stage must not allocate a GPU')
    need(args.check or args.plan is not None,'Frozen CPU plan required');need(not args.report or args.run_directory is not None,'Completed GPU directory required')
    mode='check' if args.check else 'run' if args.run else 'report';out=OUT/f'{mode}_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False)
    frozen=snapshot(out)
    (prepare if args.check else run if args.run else report)(args,out,frozen)
    need(sources()==frozen,'Source changed during V17 execution');print(json.dumps(dict(passed=True,output=str(out))),flush=True)


if __name__=='__main__':main()

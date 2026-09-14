"""Independent CPU audit of one completed native prefix-supervision main fit.

All scalar training records and final training diagnostics are audited. Only the
six prospectively retained teachers and actual natural head queries are replayed;
no vision, decoder, backward, optimizer or fresh evaluation is executed here.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import report_mmred_prefix_supervision as software
from scripts import report_mmred_official_native_training as old_audit
from scripts import train_mmred_prefix_supervision as producer
reference=software.reference;p=software.p
need=software.need;read=software.read;save=software.save;sha=software.sha;ref=software.ref
PROTOCOL='mmred_prefix_supervision_main_independent_audit'
OUT=REPO/'outputs/native_aggregation_vlm/mmred_prefix_supervision_main_audit'
DATA=Path('/mnt/data/gabriele/gnn_transformer/mmred_prefix_supervision_main_audit')
OWN=('scripts/report_mmred_prefix_supervision_main.py','slurm/mmred_prefix_supervision_main_report.sbatch',
     'docs/paper/MMRED_PREFIX_SUPERVISION_MAIN_AUDIT.md')
ARMS=('answer','local','prefix');SEEDS=(25,26);ROUNDTRIP=(0,3222)
POLICY=dict(cpu_seconds=3600,cpu_cores=4,memory_gib=16,publication_reserve_seconds=60,
    presentations=12000,updates=1500,epochs=3,accumulation=8,train_worlds=4000,
    teacher_captures=6,natural_training_diagnostics=100,maximum_head_calls=5006,maximum_head_rows=5300,
    cpu_native_tv_max=.02,cpu_argmax_gate=False,core_atol=2e-4,core_rtol=2e-4,ce_absolute_tolerance=2e-6,
    model_calls=0,vision_calls=0,decoder_calls=0,backward_calls=0,optimizer_steps=0,
    no_efficacy_gate=True,no_fresh_predictions=True,no_inference_supervision=True,
    no_automatic_evaluation_release=True,collect_numerical_evidence_before_gate=True)


def sources():return {name:sha(REPO/name) for name in OWN}


def inherited_sources():
    path=producer.OUT/'source_release.json'
    need(sha(path)=='a066ef7fb567765fb91345dc9a6f9a4e30cde1ed9e6281cc5b26d504961dd31f',
         'Frozen privileged-training source release changed')
    return producer.verify_release(path)['source_sha256']


class Audit(software.Audit):
    """Same immutable packet/math helpers with an explicit main-audit budget."""
    def guard(self):need(time.perf_counter()-self.started<3540,'Main audit publication reserve reached')


def expected_order(seed):
    need(seed in SEEDS,'One prospectively selected training seed required')
    rng=random.Random(seed);result=[]
    for epoch in range(3):
        indices=list(range(4000));rng.shuffle(indices)
        result.extend(dict(index=i,epoch=epoch,micro_index=epoch*4000+j) for j,i in enumerate(indices))
    return result


def inputs(a,path):
    from scripts import prepare_mmred_prefix_supervision as preparation
    from scripts import evaluate_mmred_official_native_memory as original_evaluation
    path=Path(path).resolve();directory=path.parent;summary=a.record(path)
    need(summary['protocol']==producer.PROTOCOL and summary['phase']=='run' and summary['passed'] is summary['completed'] is True
         and not (directory/'failure.json').exists(),'Completed successful original main run required')
    config=a.record(directory/'config.json');analysis=a.record(summary['analysis']['file'],summary['analysis']['sha256'])
    a.manifest=a.record(summary['artifacts']['file'],summary['artifacts']['sha256'])
    request=a.record(directory/'request.json');arm=analysis['arm'];seed=analysis['seed'];task=analysis['task_index']
    need(arm in ARMS and seed in SEEDS and task==3*SEEDS.index(seed)+ARMS.index(arm)
         and re.fullmatch(r'run_\d+_'+str(task),directory.name) is not None,'Exact six-fit arm/seed/task membership differs')
    need(config['protocol']==analysis['protocol']==producer.PROTOCOL and config['policy']==analysis['policy']==producer.POLICY
         and analysis['phase']=='run' and analysis['passed'] is analysis['completed'] is True
         and analysis['no_validation_test_inference'] and analysis['requires_independent_audit'] and not analysis['objective_achieved'],
         'Main policy/completion/no-evaluation record differs')
    for key in ('arm','seed','task_index','initial_continuation_checkpoint','final_checkpoint','order','training_trace','updates_trace','counters'):
        need(summary[key]==analysis[key],'Main summary alias differs: '+key)
    for key in ('source_release','targets','main_release'):
        need(config[key]==request[key]==summary[key],'Actual consumed release/input descriptor differs: '+key)
    need(request['phase']=='run' and request['task_index']==task and request['policy']==producer.POLICY,'Main request differs')
    release=a.record(config['source_release']['file'],config['source_release']['sha256'])
    need(producer.verify_release(config['source_release']['file'])==release
         and release['source_sha256']==inherited_sources(),'Producer source release differs')
    for name,digest in release['source_sha256'].items():
        a.bind(REPO/name,digest);a.bind(directory/'source'/name.replace('/','_'),digest)
    for file,digest in a.manifest.items():a.guard();a.bind(file,digest)
    need(a.record(directory/'base_before.json')==a.record(directory/'base_after.json'),'Frozen native base parameters changed')
    need(config['float32_matmul_precision']=='highest' and config['cuda_matmul_allow_tf32'] is False,'Actual auxiliary FP32 arithmetic policy differs')
    main_release=a.record(config['main_release']['file'],config['main_release']['sha256'])
    args=argparse.Namespace(main_release=Path(config['main_release']['file']),release=Path(config['source_release']['file']),targets=Path(config['targets']['file']))
    need(producer.verify_main_release(args)==main_release,'Exact profile-audited main release required')
    profile=software.verify_report(main_release['profile_audit']['file']);a.bind(main_release['profile_audit']['file'],main_release['profile_audit']['sha256'])
    need(profile['targets']==config['targets'] and profile['source_release']==config['source_release']
         and profile['producer_summary']==main_release['profile_summary'],'Main was not released from this exact software proof')
    a.bind(producer.TRAIN_PLAN,producer.TRAIN_PLAN_SHA);a.bind(producer.EVAL_PLAN,producer.EVAL_PLAN_SHA)
    plan=original_evaluation.training.verify_plan(producer.TRAIN_PLAN);evaluation=original_evaluation.verify_plan(producer.EVAL_PLAN)
    need(config['training_plan']==ref(producer.TRAIN_PLAN) and config['evaluation_plan']==ref(producer.EVAL_PLAN)
         and config['native_identity_sha256']==plan['native_identity_sha256']==evaluation['native_identity_sha256'],'Original native/input lineage differs')
    target_plan=preparation.verify_stage(config['targets']['file']);a.bind(config['targets']['file'],config['targets']['sha256'])
    need(target_plan==config['target_plan'] and target_plan['training_plan']==config['training_plan'],'Auxiliary targets refer to a changed cohort')
    target_summary=read(config['targets']['file']);a.bind(target_summary['plan_file'],target_summary['plan_sha256'])
    proof=evaluation['main_arms']['ordinary']
    need(config['initial']==analysis['initial']==profile['initial']==proof['final_checkpoint'] and config['peft_config']==proof['peft_config'],
         'Only the original fitted ordinary1500 checkpoint may initialize a main fit')
    initial=a.load(config['initial'],external=True);reference.adapter_state(a.torch,initial['adapter'])
    need(original_evaluation.state_proof(a.torch,initial,'ordinary',config['peft_config'])==proof['state_proof'],'Actual initial adapter bytes differ')
    contract=config['contract'];need(contract['targets']==list(p.TARGETS) and contract['trainable_parameters']==10092544
         and set(contract['tensors'])==set(initial['adapter']) and contract['rank']==16 and contract['alpha']==32
         and contract['dropout']==.05 and contract['scaling']==2.,'Actual224 language-attention parameter contract differs')
    for name,value in initial['adapter'].items():
        need(contract['tensors'][name]['shape']==list(value.shape) and contract['tensors'][name]['dtype']=='torch.float32','Adapter contract geometry differs')
    items=a.record(plan['cases_file']);need(len(items)==4000,'Exact original4000 training entries required')
    need({v['file'] for v in analysis['states'].values()}.isdisjoint(v['file'] for v in profile['final_software_checkpoints'].values()),
         'Main state ledger reused a fitted software checkpoint')
    return directory,config,analysis,plan,target_plan,items,initial,release,main_release,profile


def checkpoint_audit(a,config,analysis,initial):
    torch=a.torch;arm=analysis['arm'];seed=analysis['seed'];refs=analysis['states'];need(len(refs)==3,'Exactly state0/1499/1500 required')
    result={};initial_identity=reference.state_identity(initial['adapter']);final_identity=None;proof=[]
    for update in (0,1499,1500):
        descriptor=refs[f'{arm}_seed{seed}_update{update}.pt'];state=a.load(descriptor)
        need(descriptor['arm']==state['arm']==arm and descriptor['seed']==state['seed']==seed and descriptor['updates']==state['updates']==update
             and state['peft_config']==config['peft_config'] and state['initial']==config['initial'] and state['source_release']==config['source_release']
             and state['targets']==config['targets'] and state['policy']==producer.POLICY,'Main checkpoint lineage/policy differs')
        reference.adapter_state(torch,state['adapter']);identity=reference.state_identity(state['adapter'])
        need(state['rng']['cpu'].dtype==torch.uint8 and len(state['rng']['cuda'])==1 and state['rng']['cuda'][0].dtype==torch.uint8,
             'Actual CPU/CUDA RNG states required')
        if update==0:
            a.add('initial_adapter_exact',identity==initial_identity)
            a.exact('initial_cpu_dropout_seed',state['rng']['cpu'],
                    torch.Generator(device='cpu').manual_seed(seed).get_state())
        optimizer=state['optimizer'];need(len(optimizer['param_groups'])==1,'One fixed AdamW parameter group required');group=optimizer['param_groups'][0]
        need(group['lr']==2e-4 and tuple(group['betas'])==(.9,.999) and group['eps']==1e-8 and group['weight_decay']==.01
             and len(group['params'])==224 and len(set(group['params']))==224,'Main optimizer recipe or parameter scope differs')
        need(set(optimizer['state'])==(set() if update==0 else set(group['params'])),'Fresh/update optimizer moment population differs')
        if update:
            for identifier,value in zip(group['params'],state['adapter'].values()):
                m=optimizer['state'][identifier]
                need(set(m)=={'step','exp_avg','exp_avg_sq'} and float(m['step'])==update
                     and all(m[k].shape==value.shape and m[k].dtype==torch.float32 and bool(m[k].isfinite().all()) for k in ('exp_avg','exp_avg_sq')),
                     'Complete finite actual AdamW moments/steps required')
        result[update]=descriptor
        proof.append(dict(updates=update,checkpoint=descriptor,optimizer_tensors=len(optimizer['state']),rng={k:software.reference.state_identity({str(i):v for i,v in enumerate(values)})
                     for k,values in (('cpu',[state['rng']['cpu']]),('cuda',state['rng']['cuda']))}))
        if update==1500:final_identity=identity
        del state
    need(analysis['initial_continuation_checkpoint']==result[0] and analysis['final_checkpoint']==result[1500],'Initial/final checkpoint alias differs')
    return result,dict(passed=True,checkpoints=proof,initial_adapter_tensors=initial_identity,final_adapter_tensors=final_identity,
                      trainable_tensors=224,trainable_parameters=10092544,auxiliary_trainable_parameters=0,optimizer_reexecuted=False)


def trace_audit(a,analysis,items):
    seed=analysis['seed'];arm=analysis['arm'];order=a.record(analysis['order']['file'],analysis['order']['sha256'])
    need(order==expected_order(seed),'Actual three successive seed-specific shuffles differ')
    a.bind(analysis['training_trace']['file'],analysis['training_trace']['sha256']);a.bind(analysis['updates_trace']['file'],analysis['updates_trace']['sha256'])
    retained={};head_rows=0;epochs=[dict(examples=0,sum_mean_ce=0.,sum_auxiliary_loss=0.,sum_total_loss=0.) for _ in range(3)]
    counts=Counter();zero_layers=[0]*28;traces=[]
    with Path(analysis['training_trace']['file']).open() as stream:
        count=0
        for count,line in enumerate(stream,1):
            if count%100==1:a.guard()
            need(count<=12000,'Additional unplanned training presentation');r=json.loads(line);entry=order[count-1];item=items[entry['index']]
            need(all(r[k]==v for k,v in entry.items()) and r['arm']==arm and r['seed']==seed and r['n']==item['row']['n']
                 and r['target_rows']==len(item['compact']['target_ids']) and r['gradients_all_present'] is r['gradients_all_finite'] is True,
                 'Actual scalar training exposure/gradient ownership differs')
            need(all(math.isfinite(r[k]) and r[k]>=0 for k in ('mean_ce','auxiliary_loss','total_loss','backward_loss'))
                 and math.isfinite(r['seconds']) and r['seconds']>0 and len(r['gradient_layer_l2'])==28
                 and all(math.isfinite(v) and v>=0 for v in r['gradient_layer_l2']),'Finite full training scalar evidence required')
            weight=0. if arm=='answer' else .1;expected=r['mean_ce']+weight*r['auxiliary_loss']
            if arm=='answer':need(r['auxiliary_loss']==0.,'Answer-only training gained auxiliary loss')
            a.add(f'trace_loss_{count-1}',abs(r['total_loss']-expected)<=2e-6+2e-6*abs(expected)
                  and r['backward_loss']==r['total_loss']/8,actual_total=r['total_loss'],expected_total=expected)
            wanted=count in (1,12000);need((r['retained'] is not None)==wanted,'Unreleased training capture selection differs')
            if wanted:retained[count-1]=r
            for i,v in enumerate(r['gradient_layer_l2']):zero_layers[i]+=int(v==0)
            head_rows+=r['target_rows'];counts[(entry['epoch'],entry['index'])]+=1
            e=epochs[entry['epoch']];e['examples']+=1
            for key in ('mean_ce','auxiliary_loss','total_loss'):e['sum_'+key]+=r[key]
            traces.append(dict(index=r['index'],n=r['n'],seconds=r['seconds'],retained=wanted))
    need(count==12000 and counts==Counter({(epoch,index):1 for epoch in range(3) for index in range(4000)})
         and head_rows==3*sum(len(item['compact']['target_ids']) for item in items),'Exact4000 entries per epoch/head exposure differs')
    with Path(analysis['updates_trace']['file']).open() as stream:
        count=0
        for count,line in enumerate(stream,1):
            need(count<=1500,'Extra optimizer update');r=json.loads(line)
            need(r['arm']==arm and r['update']==count and r['parameter_steps']==[float(count)]*224 and r['gradients'] is None
                 and math.isfinite(r['gradient_norm']) and r['gradient_norm']>=0 and math.isfinite(r['seconds']) and r['seconds']>0
                 and r['gradient_save_seconds']==0.,'Actual main optimizer update/gradient scope differs')
    need(count==1500,'Exactly1500 optimizer updates required')
    return order,retained,traces,dict(passed=True,presentations=12000,updates=1500,training_head_rows=head_rows,epochs=epochs,
        seed=seed,order=analysis['order'],gradient_presence_finiteness_scope='GPU summaries of all224 accumulated gradients for every presentation',
        zero_accumulated_gradient_layer_observations=zero_layers,full_per_parameter_gradients_replayed=False)


def teacher_audits(a,config,analysis,items,order,trace,states,packet,records):
    arm=analysis['arm'];expected=[('training_0',order[0]['index'],states[0],True),('training_11999',order[-1]['index'],states[1499],True)]
    expected += [(f'roundtrip_{phase}_{i}',i,states[1500],False) for phase in ('before','after') for i in ROUNDTRIP]
    rows=analysis['retained_teachers'];need(len(rows)==6 and [r['key'] for r in rows]==[v[0] for v in expected],'Fixed six retained teacher calls required')
    results=[];logits={}
    for row,(key,i,state,backward) in zip(rows,expected):
        a.guard();tick=time.perf_counter();raw=a.load(row);e=raw['evidence'];item=items[i]
        case,features=old_audit.load_case(a.torch,item,a.bindings);del features
        need(raw['arm']==row['arm']==arm and raw['case_index']==row['index']==i and raw['phase']=='teacher'
             and raw['backward_completed']==row['backward']==backward and row['parameter_state']==e['parameter_state']==state
             and e['compact_packet']==item['compact'] and e['feature_packet']==item['features'], 'Retained main teacher owner differs')
        need(e['rope_restored'] and e['hooks_removed'] and not e.get('pixel_route',False)
             and e['actual_language_embeddings']==e['embedding_identity']==e['native_inputs'][0]['inputs_embeds_identity'],
             'Original native teacher embedding/restoration differs')
        # Validate true arm and state above; only select the unchanged ordinary
        # route inside the immutable numerical helper, with no global mutation.
        head_tick=time.perf_counter();metric=reference.teacher_audit(a.torch,dict(raw,arm='ordinary'),case,'ordinary',i,a.modules,a.data,key,True,backward)
        a.head_seconds+=time.perf_counter()-head_tick;a.head_calls+=1;a.head_rows+=metric['target_rows']
        a.add(key+'_native',metric['passed'],result=metric)
        need(e['prefix_supervision']['targets_descriptor']==config['target_plan']['targets'],'Retained auxiliary packet differs')
        software.supervision_audit(a,key,e,case,arm,records[i],packet,backward,8)
        need('auxiliary_gradient_probe' not in e,'Main fit added a software-only backward probe')
        if backward:
            software.gradient_metadata(raw['gradient_summary'],config['contract']['tensors'])
            m=0 if key=='training_0' else 11999;r=trace[m]
            need(all(row[k]==v for k,v in r['retained'].items()) and r['mean_ce']==e['loss'] and r['auxiliary_loss']==e['auxiliary_loss']
                 and r['total_loss']==e['total_loss'] and r['backward_loss']==e['backward_loss']
                 and r['gradient_layer_l2']==raw['gradient_summary']['layer_l2'],'Full trace/retained loss and gradient summary differ')
        else:
            need(raw['gradient_summary'] is None,'Evaluation teacher has training gradients');logits[key]=e['profile_head'][0]['head_logits'].clone()
        results.append(dict(key=key,index=i,arm=arm,passed=metric['passed'],native=metric,raw=dict(file=row['file'],sha256=row['sha256'])))
        a.teacher_seconds+=time.perf_counter()-tick;save(a.out/(key+'_audit.json'),results[-1]);del raw,case
    for i in ROUNDTRIP:a.exact(f'checkpoint_reload_{i}',logits[f'roundtrip_before_{i}'],logits[f'roundtrip_after_{i}'])
    need(a.supervision_calls==6,'All six retained auxiliary captures required')
    return results


def natural_audits(a,config,analysis,items,plan,processor):
    rows=analysis['natural'];indices=plan['diagnostic_indices'];arm=analysis['arm'];state=analysis['final_checkpoint']
    need(len(rows)==len(indices)==len(set(indices))==100 and [r['index'] for r in rows]==indices,'Fixed100 original train-only diagnostics required')
    need(Counter((items[i]['row']['n'],items[i]['row']['qtype']) for i in indices)==Counter({(n,q):5 for n in (1,2,4,8,16)
        for q in ('char_at_frame','steps_in_room','spend_together','where_spend')}),'Diagnostic cell/exposure selection differs')
    results=[]
    for j,row in enumerate(rows):
        a.guard();i=row['index'];item=items[i];raw=a.load(row);e=raw['evidence'];case,features=old_audit.load_case(a.torch,item,a.bindings);del features
        need(row['arm']==raw['arm']==arm and row['parameter_state']==e['parameter_state']==state and row['index']==raw['case_index']==i
             and row['sid']==item['row']['sid'] and row['n']==item['row']['n'] and row['qtype']==item['row']['qtype']
             and e['compact_packet']==item['compact'] and e['feature_packet']==item['features'] and row['auxiliary_hook_installed'] is False
             and e['prefix_supervision_active'] is False and 'prefix_supervision' not in e,'Final diagnostic state/input/hook ownership differs')
        need(1<=len(row['generated_ids'])<=50 and math.isfinite(row['seconds']) and row['seconds']>0,'Original bounded native greedy trajectory required')
        tick=time.perf_counter();metric=old_audit.natural_audit(a.torch,dict(raw,arm='ordinary'),row,case,item,'ordinary',
            dict(final_checkpoint=state),None,processor,a.modules,a.data,f'natural_{j}')
        a.head_seconds+=time.perf_counter()-tick;a.head_calls+=metric['head_calls'];a.head_rows+=metric['head_rows']
        metric.update(actual_arm=arm,memory_reconstruction_scope='not_applicable_ordinary_native',no_inference_supervision=True)
        a.add(f'natural_{j}',metric['passed'],result=metric);results.append(metric);save(a.out/f'natural_{j}_audit.json',metric);del raw,case
    need(sum(v['score']['correct'] for v in results)==analysis['training_diagnostic_correct'],'Descriptive training score aggregation differs')
    return results


def accounting(directory,out,task):
    array=directory.name.split('_')[1];own=array+'_'+str(task)
    command=['sacct','-X','-j',array,'--parsable2','--noheader','--format=JobID%40,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,TimelimitRaw']
    text=subprocess.run(command,capture_output=True,text=True,check=True).stdout;file=out/'sacct.psv';file.write_text(text);rows=[]
    for line in text.splitlines():
        fields=line.split('|')
        if len(fields)!=8 or re.fullmatch(re.escape(array)+r'_[0-5]',fields[0]) is None:continue
        job,name,partition,state,exitcode,elapsed,tres,limit=fields
        gpu=re.findall(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',tres);typed=re.findall(r'(?:^|,)gres/gpu:[^=,]+=(\d+)(?=,|$)',tres)
        count=int(gpu[0]) if gpu else sum(map(int,typed))
        need(name=='mmred_prefix_supervision_train' and partition=='gpu' and int(limit)==120 and int(elapsed)<=7200 and count in (0,1),
             'Registered main allocation scope/cap differs')
        rows.append(dict(job_id=job,name=name,state=state,exit_code=exitcode,allocated_gpu_seconds=int(elapsed)*count,gpus=count,time_limit_minutes=int(limit)))
    current=[r for r in rows if r['job_id']==own];need(len(current)==1 and current[0]['state']=='COMPLETED'
        and current[0]['exit_code']=='0:0' and current[0]['gpus']==1 and len({r['job_id'] for r in rows})==len(rows), 'Own main allocation did not complete within cap')
    return dict(passed=True,array_job_id=array,own=current[0],observed_array_rows=rows,maximum_per_fit_gpu_seconds=7200,
        maximum_six_fit_gpu_seconds=43200,other_fit_completion_not_required=True,raw=ref(file))


def run(a,args):
    directory,config,original,plan,target_plan,items,initial,release,main_release,profile=inputs(a,args.run)
    states,state_proof=checkpoint_audit(a,config,original,initial);del initial
    order,retained,trace_timings,trace_proof=trace_audit(a,original,items)
    packet=a.load(target_plan['projection'],external=True);records=a.load(target_plan['targets'],external=True)['records']
    need(len(records)==4000 and packet['seed']==20260914 and packet['trainable_parameters']==0
         and profile['target_audit']['passed'] and profile['target_audit']['worlds']==4000,'Already independently audited fixed target/projection packet required')
    for i,r in enumerate(records):need(r['index']==i and r['sid']==items[i]['row']['sid'] and r['n']==items[i]['row']['n'],'Target record ownership differs')
    parent=a.record(plan['profile_preparation']['plan_file'],plan['profile_preparation']['plan_sha256'])
    a.modules=reference.native_modules(a.torch,parent,a.bindings)
    from transformers import AutoProcessor,__version__ as version
    processor=AutoProcessor.from_pretrained(str(old_audit.producer.profile.preparation.MODEL),trust_remote_code=True,use_fast=False,local_files_only=True)
    need(p.fingerprint(processor,str(version))==plan['native_identity']['processor'],'Exact original tokenizer required')
    teachers=teacher_audits(a,config,original,items,order,retained,states,packet,records)
    natural=natural_audits(a,config,original,items,plan,processor);G=sum(v['head_rows'] for v in natural);calls=12004+G
    counts=dict(model=calls,backbone=calls,vision=0,language=calls,norm=calls,head=calls,
                backward=12000,optimizer=1500,auxiliary_gradient_probes=0)
    need(original['counters']==counts and original['decoder_layer_calls']==[calls]*28,'Actual full native/backward/update inventory differs')
    expected_teacher_rows=sum(len(items[i]['compact']['target_ids']) for i in (order[0]['index'],order[-1]['index'],*ROUNDTRIP,*ROUNDTRIP))
    need(a.head_calls==6+G and a.head_rows==expected_teacher_rows+G and a.head_calls<=5006 and a.head_rows<=5300,
         'Fixed CPU replay scope/caps differ')
    timings=original['timings'];need(len(timings['teacher'])==12004 and len(timings['optimizer'])==1500
         and len(timings['checkpoint'])==3 and len(timings['generation'])==100,'Complete main timing population required')
    for r,t in zip(trace_timings,timings['teacher'][:12000]):
        need(t['arm']==original['arm'] and t['n']==r['n'] and t['backward'] is True and t['retained']==r['retained'] and t['seconds']==r['seconds']
             and t['auxiliary_probe_seconds']==0. and t['training_work_seconds']==t['seconds']-t['retention_seconds'], 'Full trace/timing/probe scope differs')
    for r,t in zip(original['natural'],timings['generation']):need(t==dict(n=r['n'],tokens=len(r['generated_ids']),seconds=r['seconds']),'Actual generation timing differs')
    need(all(math.isfinite(v) and v>0 for v in [original['setup_seconds'],original['elapsed_seconds'],*timings['optimizer'],*timings['checkpoint']])
         and original['peak_reserved_bytes']>=original['peak_allocated_bytes']>0,'Actual time/memory accounting differs')
    resources=accounting(directory,a.out,original['task_index']);inherited=dict(release['source_sha256'])
    need(all(name in inherited for name in software.OWN),'Frozen profile auditor must belong to the consumed source release')
    return dict(protocol=PROTOCOL,policy=POLICY,phase='run',passed=all(v['passed'] for v in a.checks),completed=True,
        arm=original['arm'],seed=original['seed'],task_index=original['task_index'],final_checkpoint=states[1500],
        initial_checkpoint=config['initial'],initial_continuation_checkpoint=states[0],final_adapter_tensors=state_proof['final_adapter_tensors'],
        trainable_tensors=224,trainable_parameters=10092544,auxiliary_trainable_parameters=0,
        producer_summary=ref(args.run),producer_config=ref(directory/'config.json'),producer_analysis=ref(directory/'analysis.json'),
        source_release=config['source_release'],targets=config['targets'],main_release=config['main_release'],profile_audit=main_release['profile_audit'],
        training_plan=config['training_plan'],evaluation_plan=config['evaluation_plan'],native_identity_sha256=config['native_identity_sha256'],
        peft_config=config['peft_config'],contract=config['contract'],source_sha256=sources(),inherited_source_sha256=inherited,
        state_audit=state_proof,trace_audit=trace_proof,teacher_audits=teachers,natural_audits=natural,
        numerical_checks=a.checks,numerical_failures=[v['key'] for v in a.checks if not v['passed']],resources=resources,
        actual_native_counters=counts,actual_native_head_rows=trace_proof['training_head_rows']+2*sum(len(items[i]['compact']['target_ids']) for i in ROUNDTRIP)+G,
        cpu_head_calls=a.head_calls,cpu_head_rows=a.head_rows,auxiliary_fp64_captures=a.supervision_calls,
        cpu_head_seconds=a.head_seconds,teacher_audit_seconds=a.teacher_seconds,elapsed_seconds=time.perf_counter()-a.started,
        measured_fit_seconds=original['elapsed_seconds'],peak_allocated_bytes=original['peak_allocated_bytes'],peak_reserved_bytes=original['peak_reserved_bytes'],
        raw_inputs_verified_in_cpu_audit=True,raw_inputs_rehashed_in_handoff=False,decoder_backward_reexecuted=False,
        no_efficacy_gate=True,no_fresh_predictions=True,no_inference_supervision=True,no_automatic_evaluation_release=True,
        matched_seed_scope=dict(seed=original['seed'],expected_arms=list(ARMS),all_six_reports_required_for_cohort_claim=True))


def verify_report(summary_path):
    """Bound final checkpoint proof; no raw tensor loads or efficacy condition."""
    path=Path(summary_path).resolve();summary=read(path)
    need(summary['protocol']==PROTOCOL and summary['policy']==POLICY and summary['passed'] is summary['completed'] is True
         and summary['phase']=='run' and not (path.parent/'failure.json').exists(),'Passed complete main audit required')
    for name in ('analysis','input_bindings','artifacts'):need(sha(summary[name]['file'])==summary[name]['sha256'],'Main-audit handoff artifact changed')
    analysis=read(summary['analysis']['file'])
    need(analysis['protocol']==PROTOCOL and analysis['policy']==POLICY and analysis['phase']=='run' and analysis['passed'] is analysis['completed'] is True
         and not analysis['numerical_failures'] and analysis['source_sha256']==sources()
         and analysis['inherited_source_sha256']==inherited_sources() and analysis['no_efficacy_gate']
         and analysis['no_inference_supervision'] and analysis['raw_inputs_verified_in_cpu_audit'] and not analysis['raw_inputs_rehashed_in_handoff'],
         'Main-audit policy/proof differs')
    need(analysis['arm'] in ARMS and analysis['seed'] in SEEDS and analysis['task_index']==3*SEEDS.index(analysis['seed'])+ARMS.index(analysis['arm'])
         and analysis['final_checkpoint']['arm']==analysis['arm'] and analysis['final_checkpoint']['seed']==analysis['seed']
         and analysis['final_checkpoint']['updates']==1500,'Final checkpoint/arm/seed identity differs')
    for key in ('arm','seed','task_index','final_checkpoint'):need(summary[key]==analysis[key],'Main-audit summary alias differs')
    for name,digest in {**analysis['inherited_source_sha256'],**analysis['source_sha256']}.items():need(sha(REPO/name)==digest,'Consumed source changed')
    for name,digest in analysis['source_sha256'].items():need(sha(path.parent/'source'/name.replace('/','_'))==digest,'Main-audit source archive changed')
    for key in ('producer_summary','producer_config','producer_analysis','source_release','targets','main_release','profile_audit'):
        need(sha(analysis[key]['file'])==analysis[key]['sha256'],'Main-audit provenance metadata changed')
    # The final tensor descriptor is immutable proof from this audit. Its consumer
    # must hash/load that single checkpoint at use, not reread every raw capture.
    return analysis


def verify_matched_reports(paths):
    need(len(paths)==6,'All six main audits required for a matched cohort')
    reports=[verify_report(path) for path in paths];need({(r['arm'],r['seed']) for r in reports}=={(a,s) for a in ARMS for s in SEEDS},'Matched arm/seed inventory differs')
    first=reports[0]
    for r in reports:
        for key in ('initial_checkpoint','source_release','targets','main_release','profile_audit','training_plan','evaluation_plan','native_identity_sha256','peft_config'):
            need(r[key]==first[key],'Matched fit shared lineage differs: '+key)
        need(r['state_audit']['initial_adapter_tensors']==first['state_audit']['initial_adapter_tensors'],'Matched fits do not share original initial bytes')
    for seed in SEEDS:
        group=[r for r in reports if r['seed']==seed]
        initial_rng=group[0]['state_audit']['checkpoints'][0]['rng']
        need(all(r['state_audit']['checkpoints'][0]['rng']==initial_rng for r in group),
             'Matched arms do not share initial CPU/CUDA dropout RNG state')
    return reports


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==4
         and os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),'CPU4 Slurm-only main audit required')
    started=time.perf_counter();tag='report_'+os.environ['SLURM_JOB_ID'];out=OUT/tag;data=DATA/tag
    out.mkdir(parents=True,exist_ok=False);data.mkdir(parents=True,exist_ok=False);(out/'source').mkdir()
    own=sources()
    for name,digest in own.items():
        destination=out/'source'/name.replace('/','_');shutil.copy2(REPO/name,destination);need(sha(destination)==digest,'Main-audit source copy differs')
    import torch
    torch.set_num_threads(4);a=Audit(torch,out,data,started)
    save(out/'request.json',dict(protocol=PROTOCOL,policy=POLICY,run=ref(args.run),source_sha256=own))
    try:
        analysis=run(a,args);need(sources()==own,'Main-audit source changed during execution')
        save(out/'analysis.json',analysis);save(out/'input_bindings.json',a.bindings)
        save(out/'artifacts.json',{str(p):sha(p) for root in (out,data) for p in root.iterdir() if p.is_file()})
        need(analysis['passed'],'Independent main numerical audit failed after complete evidence collection')
        need(time.perf_counter()-started<3600,'Main CPU audit budget exceeded')
        save(out/'summary.json',dict(protocol=PROTOCOL,policy=POLICY,phase='run',passed=True,completed=True,
            arm=analysis['arm'],seed=analysis['seed'],task_index=analysis['task_index'],final_checkpoint=analysis['final_checkpoint'],
            analysis=ref(out/'analysis.json'),input_bindings=ref(out/'input_bindings.json'),artifacts=ref(out/'artifacts.json'),
            no_efficacy_gate=True,elapsed_seconds=time.perf_counter()-started))
    except Exception as error:
        save(out/'failure_input_bindings.json',a.bindings);save(out/'partial_numerical_checks.json',a.checks)
        save(out/'failure.json',dict(protocol=PROTOCOL,passed=False,completed=False,type=type(error).__name__,message=str(error),
            elapsed_seconds=time.perf_counter()-started,head_calls=a.head_calls,head_rows=a.head_rows,
            input_bindings=ref(out/'failure_input_bindings.json'),partial_checks=ref(out/'partial_numerical_checks.json'),
            data_directory=str(data),no_automatic_retry=True));raise


if __name__=='__main__':main()

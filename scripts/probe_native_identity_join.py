"""Fixed generic inclusion audit for a prospective native identity join.

One literal prompt; no fitting, generation, filtering, oracle gate or global
answer scoring. Numerical work and canonical rendering require Slurm. Pilot
summaries expose computation/timing only; inclusion outcomes await the complete
90-batch audit. No existing dataset, model or frozen source is modified.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime
import hashlib
import inspect
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import native_vision_v7_runtime as native
from scripts import stage_native_vision_v10_features as backend
from scripts.stage_native_vision_v2_clean import CHARS
from scripts.stage_native_vision_pilot import PARK_ROOMS
from scripts.stage_native_vision_v6_teacher import MODEL,need,read,save,sha
from scripts.probe_native_vision_v2_prefix import fingerprint
ROOMS=tuple(PARK_ROOMS);STEPS=(1,8,16,17,32,64);PAIRS=tuple(itertools.combinations(ROOMS,2))
DATA=Path('/mnt/data/gabriele/gnn_transformer/identity_join_gate')
OUT=REPO/'outputs/native_aggregation_vlm/identity_join_gate'
NATIVE_PLAN=REPO/'outputs/native_aggregation_vlm/v18/semantic_gate_software/check_442830/plan.json'
NATIVE_PROOF=REPO/'outputs/native_aggregation_vlm/v18/semantic_gate_software/profile_442831/summary.json'
MANIFESTS=(Path('/mnt/data/gabriele/gnn_transformer/v18_fresh/main_manifest.json'),Path('/mnt/data/gabriele/gnn_transformer/v10_balanced/main_manifest.json'))
PROTOCOL='native_identity_join_generic_inclusion_audit'
GLOBAL_TEMPLATE="Consider only images in the {room_a} or the {room_b}. Which person appears in both rooms? Reply with the person's name only."
LOCAL_TEMPLATE="Collection question: {question}\nDoes this image satisfy the question's inclusion restriction? Reply with exactly 1 for yes or 0 for no."
POLICY=dict(protocol=PROTOCOL,people=list(CHARS),rooms=list(ROOMS),steps=list(STEPS),room_pairs=[list(p) for p in PAIRS],
    global_template=GLOBAL_TEMPLATE,local_template=LOCAL_TEMPLATE,atoms=324,local_judgments=4860,include=1620,exclude=3240,
    batches=90,local_rows_per_batch=54,global_rows_per_batch=1,head_rows=4950,pilot_indices=[0,89],
    pilot_model_calls=2,full_model_calls=90,head_replays_per_model=1,resize=392,
    measurement='Unmasked native vocabulary argmax: literal1 includes, literal0 excludes, every other token invalid',
    feasibility='All4860 local argmax tokens must be valid and correct; no subset/prompt search',
    replay_tv_max=.02,replay_top1_exact=True,native_dtype='torch.float16',probability_dtype='torch.float64',
    probability_rtol=1e-12,probability_atol=1e-15,profile_seconds=180,full_seconds=600,campaign_gpu_seconds=900,
    maximum_campaign_gpus=1,maximum_user_gpus=4,projection='load+1.5*90*max(two prefills including replay+rawsave)+30',
    no_fit=True,no_generation=True,no_global_answer_score=True,labels_offline_only=True)
OWN=('scripts/probe_native_identity_join.py','slurm/identity_join_gate_check.sbatch','slurm/identity_join_gate_profile.sbatch',
     'slurm/identity_join_gate_run.sbatch','slurm/identity_join_gate_report.sbatch','datasets/mmred/render_mmred.py',
     'scripts/stage_native_vision_v2_clean.py','scripts/stage_native_vision_pilot.py','scripts/native_vision_v7_runtime.py',
     'scripts/stage_native_vision_v10_features.py','scripts/stage_native_vision_v6_teacher.py',
     'scripts/probe_native_vision_v2_prefix.py','scripts/probe_native_vision_parallel_local.py',
     'scripts/cache_native_vision_v6_teacher.py','gnnformer/data.py','gnnformer/parallel_local_prompts.py','gnnformer/constants.py','gnnformer/runtime.py')
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}


def oid(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def sources():return {name:sha(REPO/name) for name in OWN}

def bind(path,bindings,expected=None):
    p=Path(path).resolve();h=sha(p);need(expected is None or h==expected,'Bound input changed: '+str(p));bindings[str(p)]=h;return h

def snapshot(out):
    frozen=sources();(out/'source').mkdir()
    for n,h in frozen.items():p=out/'source'/n.replace('/','_');p.write_bytes((REPO/n).read_bytes());need(sha(p)==h,'Source changed during snapshot')
    save(out/'source_hashes.json',frozen)
    (out/'INDEX.md').write_text('# Native identity-join inclusion audit\n\n[Summary](summary.json) · [Sources](source_hashes.json). Pilot outputs are timing/computational evidence only.\n')
    return frozen


def global_prompt(room_pair):
    need(isinstance(room_pair,(list,tuple)) and len(room_pair)==2 and room_pair[0]!=room_pair[1] and all(r in ROOMS for r in room_pair),'Invalid canonical room pair')
    return GLOBAL_TEMPLATE.format(room_a=room_pair[0],room_b=room_pair[1])


def local_prompt(question):
    need(isinstance(question,str) and question,'Complete global question required')
    return LOCAL_TEMPLATE.format(question=question)


def batch_specs():
    result=[]
    for pair_index,(a,b) in enumerate(PAIRS):
        question=global_prompt((a,b))
        for step in STEPS:
            result.append(dict(index=len(result),batch_id=f'pair{pair_index:02d}_step{step:03d}',pair_index=pair_index,
                rooms=[a,b],step=step,question=question,local_prompt=local_prompt(question),
                atom_ids=[f'{c}_{r}_{step:03d}' for c in CHARS for r in ROOMS]))
    return result


def labels_for(spec):
    return [dict(row=i,person=c,depicted_room=r,step=spec['step'],pair_index=spec['pair_index'],rooms=spec['rooms'],
        atom_id=f'{c}_{r}_{spec["step"]:03d}',label=int(r in spec['rooms'])) for i,(c,r) in enumerate(itertools.product(CHARS,ROOMS))]


def scores(torch,logits):
    need(logits.ndim==2 and logits.dtype==torch.float16 and logits.shape[1]>16 and bool(torch.isfinite(logits).all()),'Finite native full-vocabulary FP16 logits required')
    values=logits.double();normalizer=torch.logsumexp(values,-1);p0=(values[:,15]-normalizer).exp();p1=(values[:,16]-normalizer).exp()
    top=values.argmax(-1);return dict(top1_id=top,p0=p0,p1=p1,digit_mass=p0+p1,log_normalizer=normalizer,logit_gap=values[:,16]-values[:,15],valid=(top==15)|(top==16))


def replay_metrics(torch,a,b):
    need(a.shape==b.shape and a.ndim==2 and a.dtype==b.dtype==torch.float16 and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()),'Native head replay shape/dtype differs')
    tv=.5*(a.double().softmax(-1)-b.double().softmax(-1)).abs().sum(-1);same=a.argmax(-1)==b.argmax(-1)
    return [dict(row=i,tv=float(tv[i]),top1_equal=bool(same[i]),passed=bool(same[i] and tv[i]<=.02)) for i in range(len(a))]


def projection(load_seconds,timings):
    need(len(timings)==2 and all(math.isfinite(x) and x>=0 for x in [load_seconds]+list(timings)),'Two finite fixed pilot timings required')
    value=load_seconds+1.5*90*max(timings)+30
    return dict(seconds=value,passed=value<=600,load_seconds=load_seconds,batch_seconds=list(timings),formula=POLICY['projection'])


def self_test():
    import torch
    specs=batch_specs();labels=[x for s in specs for x in labels_for(s)]
    need(len(specs)==90 and len(labels)==4860 and len({x['atom_id'] for x in labels})==324 and Counter(x['label'] for x in labels)=={0:3240,1:1620},'Full cross-product differs')
    need(all(Counter(x['label'] for x in labels_for(s))=={0:36,1:18} for s in specs),'Batch label balance differs')
    x=torch.zeros(3,24,dtype=torch.float16);x[0,15]=2;x[1,16]=2;x[2,17]=4;s=scores(torch,x)
    need(s['top1_id'].tolist()==[15,16,17] and s['valid'].tolist()==[True,True,False] and s['digit_mass'][0]<.5,'Invalid token or full-vocabulary normalization hidden')
    need(all(m['passed'] and m['tv']==0 for m in replay_metrics(torch,x,x)),'Identical replay differs')
    need(projection(10,[1,2])['seconds']==310 and projection(10,[1,2])['passed'] and not projection(10,[5,1])['passed'],'Fixed timing boundary differs')
    sample='1|identity_join_gate_profile|gpu|FAILED|1:0|7|gres/gpu=1,gres/gpu:b200=1|2026-09-11T10:00:00|2026-09-11T10:00:07'
    zero='2|identity_join_gate_run|gpu|CANCELLED|0:0|0||Unknown|Unknown'
    need(parse_accounting(sample+'\n'+zero)['allocated_gpu_seconds']==7,'Failed/zero allocation accounting differs')
    generic=sample.replace('gres/gpu=1,gres/gpu:b200=1','gres/gpu=1')
    typed=sample.replace('gres/gpu=1,gres/gpu:b200=1','gres/gpu:b200=1')
    need(parse_accounting(generic)['allocated_gpu_seconds']==parse_accounting(typed)['allocated_gpu_seconds']==7,'Generic/typed GPU accounting differs')
    for bad in (sample+'\n'+sample,sample.replace('|7|','|181|'),sample.replace('gpu:b200=1','gpu:b200=2'),sample.replace('|FAILED|','|RUNNING|')):
        try:parse_accounting(bad)
        except ValueError:pass
        else:raise ValueError('Malformed/duplicate/nonterminal resource ledger passed')
    return dict(passed=True,tests=['resource_failed_zero_duplicate_generic_typed_caps','cross_product','label_denominators','invalid_argmax_retained','full_vocabulary_mass','native_replay_identity','timing_boundary'],native_model_calls=0,native_head_calls=0)


def stage_atoms(out,bindings):
    from PIL import Image,__version__ as pillow_version
    sys.path.insert(0,str(REPO/'datasets/mmred'));import render_mmred
    render_mmred.ROOMS=list(ROOMS)
    manifests=[read(p) for p in MANIFESTS]
    for p in MANIFESTS:bind(p,bindings)
    expected=manifests[0]['renderer_provenance'];fonts=[]
    for size in (12,16):
        font_path=render_mmred.load_font(size).path
        p=Path(font_path.decode() if isinstance(font_path,bytes) else font_path).resolve();fonts.append(dict(size=size,path=str(p),sha256=bind(p,bindings)))
    actual=dict(renderer_path=str(REPO/'datasets/mmred/render_mmred.py'),renderer_sha256=sha(REPO/'datasets/mmred/render_mmred.py'),pillow_version=pillow_version,room_order=list(ROOMS),fonts=fonts)
    need(all(actual[k]==expected[k] for k in actual),'Canonical renderer/font/Pillow differs')
    root=DATA/out.name;root.mkdir(parents=True,exist_ok=False);renders=root/'canonical_render_replays';renders.mkdir();atoms={}
    for c,r,step in itertools.product(CHARS,ROOMS,STEPS):
        key=f'{c}_{r}_{step:03d}';replay=renders/(key+'.png');render_mmred.render_frame({r:[c]},step,str(replay))
        candidates=[Path(m['dataset_root'])/'render_cache'/(key+'.png') for m in manifests];existing=next((p for p in candidates if p.is_file()),None)
        chosen=replay if existing is None else existing
        with Image.open(replay) as fresh,Image.open(chosen) as source:
            need(fresh.mode==source.mode=='RGB' and fresh.size==source.size==(512,512) and fresh.tobytes()==source.tobytes(),'Canonical atom RGB parity failed')
            rgb=hashlib.sha256(source.tobytes()).hexdigest()
        h=bind(chosen,bindings);need(existing is None or sha(replay)==h,'Canonical atom PNG bytes differ')
        atoms[key]=dict(atom_id=key,person=c,room=r,step=step,path=str(chosen),sha256=h,rgb_sha256=rgb,dimensions=[512,512],mode='RGB',
            reused=existing is not None,render_replay=str(replay),render_replay_sha256=sha(replay),canonical_rgb_equal=True)
    need(len(atoms)==324,'Canonical atom count differs');return root,atoms,actual


def prepare_batch(processor,spec,atoms,shared):
    import torch
    from PIL import Image
    images=[];old=processor.tokenizer.padding_side
    try:
        for key in spec['atom_ids']:
            with Image.open(atoms[key]['path']) as image:images.append(image.convert('RGB').resize((392,392)))
        conversations=[[dict(role='user',content=[dict(type='image',image=image),dict(type='text',text=spec['local_prompt'])])] for image in images]
        conversations.append([dict(role='user',content=[dict(type='text',text=spec['question'])])])
        processor.tokenizer.padding_side='left'
        ordinary=dict(processor.apply_chat_template(conversations,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt',padding=True))
        first=dict(processor.apply_chat_template(conversations[0],add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt'))
        global_row=dict(processor.apply_chat_template(conversations[-1],add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt'))
        pixels=ordinary['pixel_values'];grids=ordinary['image_grid_thw'];need(grids.shape==(54,3) and bool((grids==grids[0]).all()),'Canonical392 single-image grid differs')
        per_image=int(grids[0].prod());need(len(pixels)==54*per_image,'Visual patch ownership differs')
        rows=[dict(input_ids=first['input_ids'],attention_mask=first['attention_mask'],pixel_values=pixels[i*per_image:(i+1)*per_image],image_grid_thw=grids[i:i+1]) for i in range(54)]+[global_row]
        packed=native.pack_rows(rows,processor.tokenizer.pad_token_id)
        need(set(packed)==set(ordinary) and all(torch.equal(packed[k],ordinary[k]) for k in packed),'Ordinary mixed HF parity failed')
        key=str(spec['step']);visual={k:ordinary[k] for k in ('pixel_values','image_grid_thw')}
        if key in shared:need(all(torch.equal(v,shared[key][k]) for k,v in visual.items()),'Question changed shared image pixels')
        else:shared[key]={k:v.clone() for k,v in visual.items()}
        width=ordinary['input_ids'].shape[1]
        meta=dict(arm='parallel',sid=spec['batch_id'],n_frames=54,row_count=55,global_row=54,local_elements=54,row_kinds=['local']*54+['global'],
            original_prompt_width=width,prompt_width=width,prefix_ids=[],row_prompt_tokens=[first['input_ids'].shape[1]]*54+[global_row['input_ids'].shape[1]],
            question=spec['question'],local_prompt=spec['local_prompt'],global_prompt=spec['question'],resize=392,
            image_sha256=[atoms[k]['sha256'] for k in spec['atom_ids']],input_identity={k:native.tensor_info(v) for k,v in ordinary.items()},processor_parity_checked=True)
        return dict(inputs=ordinary,row_inputs=rows,metadata=meta),dict(inputs={k:v for k,v in ordinary.items() if k not in visual},visual_key=key,metadata=meta)
    finally:
        processor.tokenizer.padding_side=old
        for image in images:image.close()


def materialize(record,shared):return dict(record['inputs'],**shared[record['visual_key']])


def check(out,frozen):
    import torch,transformers
    from transformers import AutoProcessor
    torch.set_num_threads(4);tests=self_test();bindings={}
    bind(NATIVE_PLAN,bindings,'45921222241820700a8e3cbd6637c819f331c4c9f27348b8896e1a233655b8aa')
    bind(NATIVE_PROOF,bindings,'212567976c10b4c998b8368411a9694ffa66e410db03860f860a653c264510b1');parent=read(NATIVE_PLAN);proof=read(NATIVE_PROOF)
    need(proof['passed'] and proof['completed'] and proof['native_head_replay_passed'] and proof['native_identity_sha256']==parent['native_identity_sha256']
         and proof['plan_sha256']==sha(NATIVE_PLAN),'Completed Qwen native proof differs')
    inherited=parent['native_identity_payload'];need(inherited['model']['path']==str(MODEL) and inherited['native_dtypes']==dict(norm='torch.float16',lm_head='torch.float16'),'Actual frozen Qwen identity required')
    identity={k:inherited[k] for k in ('model','runtime','processor','native_api','native_dtypes','norm_weight','head_weight','norm_source_sha256','rms_norm_eps','count_token_ids')}
    identity.update(schema_version=1,measurement=POLICY['measurement'])
    need(backend.model_metadata()==identity['model'] and backend.runtime_identity()==identity['runtime'],'Qwen/runtime ancestor differs')
    processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,use_fast=False,local_files_only=True)
    owner,fn,api=backend.native_api(processor);need(api==identity['native_api'] and fingerprint(processor,str(transformers.__version__))==identity['processor'],'Processor/native API differs')
    for digit,token in [('0',15),('1',16)]:need(processor.tokenizer(digit,add_special_tokens=False)['input_ids']==[token] and processor.tokenizer.decode([token],skip_special_tokens=False)==digit,'Literal native binary token differs')
    eos=processor.tokenizer.eos_token_id;name_tokens={c:processor.tokenizer(c,add_special_tokens=False)['input_ids']+[eos] for c in CHARS}
    need(all(len(ids)<=4 and processor.tokenizer.decode(ids[:-1],skip_special_tokens=False)==c for c,ids in name_tokens.items()),'Canonical name+EOS support exceeds prospective budget')
    root,atoms,renderer=stage_atoms(out,bindings);shared={};records={};batches=[];labels={}
    for spec in batch_specs():
        bundle,record=prepare_batch(processor,spec,atoms,shared);layout=native.audit_layout(lambda **kw:fn(owner,**kw),bundle)
        batches.append(dict(spec,metadata=bundle['metadata'],layout=layout['metadata']));records[spec['batch_id']]=record;labels[spec['batch_id']]=labels_for(spec)
    need(len(shared)==6 and len(records)==90,'Compact visual/token inventory differs')
    visual_file=root/'shared_visual.pt';token_file=root/'tokens.pt';torch.save(shared,visual_file);torch.save(records,token_file)
    save(out/'atoms.json',atoms);save(out/'offline_labels.json',labels)
    plan=dict(schema_version=1,policy=POLICY,source_sha256=frozen,artifact_bindings=bindings,native_identity=identity,native_identity_sha256=oid(identity),
        renderer=renderer,native_hardware_ancestor=proof['hardware'],numeric_backend_policy=dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,float32_matmul_precision=torch.get_float32_matmul_precision()),batches=batches,visual_file=str(visual_file),visual_sha256=sha(visual_file),visual_tensors={s:{k:native.tensor_info(v) for k,v in row.items()} for s,row in shared.items()},
        token_file=str(token_file),token_sha256=sha(token_file),atoms_file=str(out/'atoms.json'),atoms_sha256=sha(out/'atoms.json'),
        labels_file=str(out/'offline_labels.json'),labels_sha256=sha(out/'offline_labels.json'),name_target_ids=name_tokens,tests=tests,
        pilot_outcomes_not_reported=True,no_model_loaded=True,render_calls=324,reused_atoms=sum(a['reused'] for a in atoms.values()))
    save(out/'plan.json',plan);(out/'plan.sha256').write_text(sha(out/'plan.json')+'\n')
    return dict(passed=True,completed=True,source_sha256=frozen,plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),tests=tests,
        atoms=324,batches=90,local_judgments=4860,head_rows=4950,render_calls=324,no_model_loaded=True)


def verify_plan(path):
    path=Path(path).resolve();plan=read(path)
    need(sha(path)==path.with_suffix('.sha256').read_text().strip() and plan['policy']==POLICY and plan['source_sha256']==sources(),'Frozen plan/source/policy differs')
    for name,h in plan['source_sha256'].items():need(sha(path.parent/'source'/name.replace('/','_'))==h,'CPU source snapshot differs')
    for file,h in plan['artifact_bindings'].items():need(sha(file)==h,'Bound input changed')
    for key in ('visual','token','atoms','labels'):need(sha(plan[key+'_file'])==plan[key+'_sha256'],'Prepared packet changed')
    need(oid(plan['native_identity'])==plan['native_identity_sha256'],'Native identity changed')
    summary=read(path.parent/'summary.json');need(summary['passed'] and summary['plan_sha256']==sha(path),'Completed CPU preparation required')
    return plan


def parse_accounting(raw,exclude_job=None):
    jobs=[];events=[]
    for line in raw.splitlines():
        if not line.strip():continue
        fields=line.split('|');need(len(fields)==9,'Unexpected scheduler record');job,name,partition,state,code,seconds,tres,start,end=fields
        if partition!='gpu' or name not in ('identity_join_gate_profile','identity_join_gate_run') or job==exclude_job:continue
        state=state.split()[0].rstrip('+');need(state in TERMINAL and job.isdigit(),'Prior audit allocation is not terminal')
        values=dict(x.split('=',1) for x in tres.split(',') if '=' in x);typed=[int(v) for k,v in values.items() if k.startswith('gres/gpu:')]
        gpus=int(values['gres/gpu']) if 'gres/gpu' in values else sum(typed);seconds=int(seconds);cap=180 if name.endswith('profile') else 600
        need(gpus in (0,1) and seconds>=0 and (not typed or sum(typed)==gpus) and (gpus==0 or seconds<=cap),'Per-allocation GPU cap exceeded')
        if gpus and start!=end:
            a,b=datetime.fromisoformat(start),datetime.fromisoformat(end);need(b>=a,'Invalid GPU interval');events.extend([(a,1),(b,-1)])
        jobs.append(dict(job_id=job,name=name,state=state,exit_code=code,gpus=gpus,elapsed_seconds=seconds,gpu_seconds=gpus*seconds))
    need(len({j['job_id'] for j in jobs})==len(jobs),'Duplicate GPU allocation');total=sum(j['gpu_seconds'] for j in jobs);running=maximum=0
    for _,change in sorted(events):running+=change;need(running>=0,'Invalid allocation order');maximum=max(maximum,running)
    need(running==0 and maximum<=1 and total<=900,'Campaign GPU budget/concurrency exceeded')
    return dict(passed=True,jobs=jobs,allocated_gpu_seconds=total,maximum_concurrent_gpus=maximum,failed_and_zero_allocations_retained=True)


def accounting(out,exclude_job=None):
    command=['sacct','-X','-u',os.environ['USER'],'--starttime=2026-09-01','--parsable2','--noheader',
             '--format=JobIDRaw,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,Start,End']
    result=subprocess.run(command,capture_output=True,text=True,check=True);file=out/'all_user_sacct.psv';file.write_text(result.stdout)
    return dict(parse_accounting(result.stdout,exclude_job),raw_file=str(file),raw_sha256=sha(file),command=command)


def project_concurrency(out):
    result=subprocess.run(['squeue','-h','-u',os.environ['USER'],'-o','%i|%P|%T|%b'],capture_output=True,text=True,check=True)
    file=out/'live_user_queue.psv';file.write_text(result.stdout);total=0
    for line in result.stdout.splitlines():
        if not line.strip():continue
        job,partition,state,gres=[x.strip() for x in line.split('|')]
        if partition=='gpu' and state in ('RUNNING','COMPLETING'):
            pieces=[p for p in gres.split(',') if p.startswith(('gpu:','gres/gpu:'))];need(pieces,'Unknown active GPU allocation')
            total+=sum(int(p.rsplit(':',1)[1].split('(')[0]) for p in pieces)
    need(total<=4,'More than four user GPUs active');return dict(passed=True,active_user_gpus=total,file=str(file),sha256=sha(file))


def verify_profile(directory,plan):
    directory=Path(directory).resolve();summary=read(directory/'summary.json')
    need(summary['profile'] is True and summary['completed'] and summary['passed'] and summary['native_replay_passed']
         and summary['source_sha256']==plan['source_sha256'] and summary['plan_sha256']==sha(summary['plan_file'])
         and read(summary['plan_file'])==plan and summary['batch_indices']==[0,89],'Complete fixed two-prefill profile required')
    need(sha(summary['records_file'])==summary['records_sha256'],'Pilot record ledger changed');records=read(summary['records_file'])
    need(len(records)==2 and [r['batch_index'] for r in records]==[0,89],'Pilot row ownership changed')
    for record in records:need(sha(record['raw_file'])==record['raw_sha256'],'Pilot raw artifact changed')
    actual=projection(summary['model_load_seconds'],[r['seconds'] for r in records]);need(actual==summary['projection'] and actual['passed'],'Measured full-audit resource gate failed')
    for n,h in summary['source_sha256'].items():need(sha(directory/'source'/n.replace('/','_'))==h,'Pilot source snapshot changed')
    return dict(directory=str(directory),summary_sha256=sha(directory/'summary.json'),projection=actual,slurm_job_id=summary['slurm_job_id'])


def run(args,out,frozen):
    import torch,transformers
    from gnnformer.runtime import load_runtime,get_rope_index_fn,move_to_device
    torch.set_num_threads(4);started=time.perf_counter();plan=verify_plan(args.plan);pilot=args.profile
    released=None if pilot else verify_profile(args.profile_directory,plan);reserve=180 if pilot else 600
    ledger=accounting(out,os.environ['SLURM_JOB_ID']);need(ledger['allocated_gpu_seconds']+reserve<=900,'Full new allocation reservation exceeds900GPU-s')
    save(out/'resource_reservation.json',dict(ledger,reserved_seconds=reserve));concurrency=project_concurrency(out)
    if released is not None:
        prior=[j for j in ledger['jobs'] if j['job_id']==str(released['slurm_job_id'])]
        need(len(prior)==1 and prior[0]['state']=='COMPLETED' and prior[0]['exit_code']=='0:0' and prior[0]['gpus']==1,'Successful measured pilot allocation required')
    need(torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA B200','One registered B200 required')
    identity=plan['native_identity'];need(backend.model_metadata()==identity['model'] and backend.runtime_identity()==identity['runtime'],'Qwen/runtime identity changed')
    shared=torch.load(plan['visual_file'],map_location='cpu',weights_only=True);tokens=torch.load(plan['token_file'],map_location='cpu',weights_only=True)
    need({s:{k:native.tensor_info(v) for k,v in row.items()} for s,row in shared.items()}==plan['visual_tensors'],'Shared visual packet identity differs')
    tick=time.perf_counter();loaded=load_runtime(str(MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda')
    model=loaded.model.eval().requires_grad_(False);norm=native.native_contract(model);head=model.lm_head
    need(fingerprint(loaded.processor,str(transformers.__version__))==identity['processor'] and backend.native_api(loaded.processor)[2]==identity['native_api'],'Actual native processor/API differs')
    need(native.tensor_info(norm.weight)==identity['norm_weight'] and native.tensor_info(head.weight)==identity['head_weight']
         and sha(inspect.getfile(type(norm)))==identity['norm_source_sha256'] and float(norm.variance_epsilon)==identity['rms_norm_eps'],'Actual native norm/head tensors/source differ')
    quantization=model.config.quantization_config
    quantization=quantization.to_dict() if hasattr(quantization,'to_dict') else dict(quantization)
    need(quantization['load_in_4bit'] and quantization['bnb_4bit_use_double_quant'] and quantization['bnb_4bit_quant_type']=='nf4'
         and str(quantization['bnb_4bit_compute_dtype']).removeprefix('torch.')=='bfloat16','Actual NF4/BF16 backend differs')
    numeric_backend=dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision())
    hardware=dict(gpu=torch.cuda.get_device_name(0),total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
        capability=list(torch.cuda.get_device_capability(0)),cuda=str(torch.version.cuda),quantization=quantization,
        attention=model.config.text_config._attn_implementation,numeric_backend=numeric_backend)
    need({k:hardware[k] for k in plan['native_hardware_ancestor']}==plan['native_hardware_ancestor']
         and numeric_backend==plan['numeric_backend_policy'],'Hardware ancestor or frozen arithmetic defaults differ')
    torch.cuda.synchronize();load_seconds=time.perf_counter()-tick;versions={n:p._version for n,p in model.named_parameters()}
    indices=[0,89] if pilot else list(range(90));records=[];counts=dict(model=0,vision=0,language=0,norm=0,head=0);extra=0;failure=None
    data=DATA/out.name;data.mkdir(parents=True,exist_ok=False)
    def counter(key):
        def hook(*_):counts[key]+=1
        return hook
    try:
        for index in indices:
            begin=time.perf_counter();batch=plan['batches'][index];record=tokens[batch['batch_id']];inputs=materialize(record,shared)
            need(record['metadata']==batch['metadata'] and {k:native.tensor_info(v) for k,v in inputs.items()}==batch['metadata']['input_identity'],'Native batch input identity differs')
            expected_positions,deltas=get_rope_index_fn(model)(input_ids=inputs['input_ids'],image_grid_thw=inputs['image_grid_thw'],attention_mask=inputs['attention_mask'])
            need(native.tensor_info(expected_positions)==batch['layout']['position_ids'] and deltas.tolist()==batch['layout']['rope_deltas'],'Native batch position layout differs')
            captured={};old_rope=getattr(model.model,'rope_deltas',None)
            def before_norm(module,args):captured['hidden']=args[0][:,-1:,:].detach().clone();counts['norm']+=1
            def after_norm(module,args,output):captured['normalized']=output[:,-1:,:].detach().clone()
            def before_language(module,args,kwargs):
                counts['language']+=1;captured['positions']=kwargs['position_ids'].detach().clone();captured['mask']=kwargs['attention_mask'].detach().clone()
            with ExitStack() as stack:
                stack.callback(setattr,model.model,'rope_deltas',old_rope)
                for handle in (model.register_forward_pre_hook(counter('model')),model.model.visual.register_forward_pre_hook(counter('vision')),
                    model.model.language_model.register_forward_pre_hook(before_language,with_kwargs=True),norm.register_forward_pre_hook(before_norm),
                    norm.register_forward_hook(after_norm),head.register_forward_pre_hook(counter('head'))):stack.callback(handle.remove)
                with torch.inference_mode():output=model(**move_to_device(inputs,model.device),use_cache=False,logits_to_keep=1)
                need(output.past_key_values is None and torch.equal(captured['positions'].cpu(),expected_positions)
                     and torch.equal(captured['mask'].cpu(),inputs['attention_mask']) and torch.equal(model.model.rope_deltas.cpu(),deltas),'Actual native positions/mask/cache differs')
            with torch.inference_mode():
                replay_normalized=norm.forward(captured['hidden']);replay=head.forward(replay_normalized);extra+=1
                metrics=replay_metrics(torch,output.logits[:,0],replay[:,0]);probabilities=scores(torch,output.logits[:,0])
            need(output.logits.shape==(55,1,152064) and captured['hidden'].shape==captured['normalized'].shape==(55,1,3584)
                 and output.logits.dtype==captured['hidden'].dtype==captured['normalized'].dtype==torch.float16,'Native full batch shape/dtype differs')
            raw=dict(schema_version=1,batch_id=batch['batch_id'],metadata=batch['metadata'],hidden=captured['hidden'].cpu(),normalized=captured['normalized'].cpu(),
                logits=output.logits.detach().cpu(),replay_logits=replay.cpu(),replay_normalized=replay_normalized.cpu(),
                position_ids=captured['positions'].cpu(),attention_mask=captured['mask'].cpu(),probabilities={k:v.cpu() for k,v in probabilities.items()})
            rawfile=data/f'batch_{index:03d}.pt';torch.save(raw,rawfile);h=sha(rawfile)
            entry=dict(batch_index=index,batch_id=batch['batch_id'],raw_file=str(rawfile),raw_sha256=h,replay=metrics,
                normalized_exact=torch.equal(captured['normalized'],replay_normalized),tensors={k:native.tensor_info(v) for k,v in raw.items() if isinstance(v,torch.Tensor)},
                probability_tensors={k:native.tensor_info(v) for k,v in raw['probabilities'].items()},seconds=time.perf_counter()-begin)
            records.append(entry);save(out/f'record_{index:03d}.json',entry)
            print(json.dumps(dict(completed_prefills=len(records),required_prefills=len(indices))),flush=True)
    except BaseException as exc:failure=dict(type=type(exc).__name__,message=str(exc));raise
    finally:
        save(out/'records.json',records);complete=len(records)==len(indices) and failure is None
        unchanged=versions=={n:p._version for n,p in model.named_parameters()} and not any(p.requires_grad or p.grad is not None for p in model.parameters())
        numeric=complete and all(m['passed'] for r in records for m in r['replay']);expected=dict.fromkeys(counts,len(indices))
        projected=projection(load_seconds,[r['seconds'] for r in records]) if pilot and len(records)==2 else None
        computational=complete and unchanged and counts==expected and extra==len(indices) and numeric
        passed=computational and (not pilot or projected['passed'])
        save(out/'summary.json',dict(schema_version=1,protocol=PROTOCOL,profile=pilot,completed=complete,passed=passed,computational_integrity_passed=computational,
            native_replay_passed=numeric,native_parameters_unchanged=unchanged,source_sha256=frozen,plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),
            native_identity_sha256=plan['native_identity_sha256'],hardware=hardware,batch_indices=indices,counts=counts,extra_head_calls=extra,extra_norm_calls=extra,
            records_file=str(out/'records.json'),records_sha256=sha(out/'records.json'),head_failures=[m for r in records for m in r['replay'] if not m['passed']],
            projection=projected,profile_release=released,resource_reservation=ledger,project_concurrency=concurrency,
            model_load_seconds=load_seconds,seconds=time.perf_counter()-started,failure=failure,slurm_job_id=os.environ['SLURM_JOB_ID'],
            no_inclusion_outcomes_in_summary=True,no_fit=True,no_generation=True,no_global_answer_score=True))
    need(passed,'Native computational/resource gate failed; every available raw record retained');return None


def summary_metrics(rows):
    need(rows,'Empty measurement group');classes={k:[r for r in rows if r['label']==k] for k in (0,1)}
    accuracy={str(k):sum(r['correct'] for r in subset)/len(subset) if subset else None for k,subset in classes.items()}
    return dict(n=len(rows),valid=sum(r['valid'] for r in rows),invalid=sum(not r['valid'] for r in rows),correct=sum(r['correct'] for r in rows),
        accuracy=sum(r['correct'] for r in rows)/len(rows),class_counts={str(k):len(v) for k,v in classes.items()},class_accuracy=accuracy,
        equal_class_macro_accuracy=sum(accuracy.values())/2 if all(v is not None for v in accuracy.values()) else None,
        minimum_digit_mass=min(r['digit_mass'] for r in rows),mean_digit_mass=math.fsum(r['digit_mass'] for r in rows)/len(rows),
        minimum_target_probability=min(r['target_probability'] for r in rows),mean_target_probability=math.fsum(r['target_probability'] for r in rows)/len(rows),
        errors=[dict(batch_id=r['batch_id'],row=r['row'],top1_id=r['top1_id'],top1_text=r['top1_text'],label=r['label'],valid=r['valid']) for r in rows if not r['correct']])


def report(args,out,frozen):
    import torch
    from transformers import AutoProcessor
    torch.set_num_threads(4);directory=Path(args.run_directory).resolve();summary=read(directory/'summary.json');plan=verify_plan(summary['plan_file'])
    need(summary['profile'] is False and summary['completed'] and summary['passed'] and summary['source_sha256']==frozen==plan['source_sha256']
         and summary['plan_sha256']==sha(summary['plan_file']) and summary['native_identity_sha256']==plan['native_identity_sha256'],'Complete exact-source full audit required')
    for n,h in frozen.items():need(sha(directory/'source'/n.replace('/','_'))==h,'Full GPU source snapshot differs')
    need(sha(summary['records_file'])==summary['records_sha256'],'Full raw ledger changed');records=read(summary['records_file'])
    need([r['batch_index'] for r in records]==list(range(90)) and summary['counts']==dict(model=90,vision=90,language=90,norm=90,head=90)
         and summary['extra_head_calls']==summary['extra_norm_calls']==90 and summary['native_parameters_unchanged'],'Full native invocation inventory differs')
    released=verify_profile(summary['profile_release']['directory'],plan);need(released==summary['profile_release'],'Full audit used another pilot release')
    atoms=read(plan['atoms_file']);truth=read(plan['labels_file']);processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,use_fast=False,local_files_only=True)
    rows=[];max_roundoff=0.;head_rows=0;max_head_tv=0.
    for expected,batch,record in zip(batch_specs(),plan['batches'],records):
        need(all(batch[k]==v for k,v in expected.items()) and record['batch_id']==batch['batch_id'],'Independent batch enumeration differs')
        labels=labels_for(expected);need(truth[batch['batch_id']]==labels,'Offline inclusion labels differ from room membership')
        for label in labels:
            atom=atoms[label['atom_id']];need((atom['person'],atom['room'],atom['step'])==(label['person'],label['depicted_room'],label['step'])
                and sha(atom['path'])==atom['sha256'] and sha(atom['render_replay'])==atom['render_replay_sha256'],'Canonical atom ownership/hash differs')
        need(sha(record['raw_file'])==record['raw_sha256'],'Raw native batch changed');raw=torch.load(record['raw_file'],map_location='cpu',weights_only=True)
        need(raw['schema_version']==1 and raw['batch_id']==batch['batch_id'] and raw['metadata']==batch['metadata'],'Raw model-input identity differs')
        for key,info in record['tensors'].items():need(native.tensor_info(raw[key])==info,'Raw tensor identity differs')
        for key,info in record['probability_tensors'].items():need(native.tensor_info(raw['probabilities'][key])==info,'Raw probability identity differs')
        need(raw['logits'].shape==raw['replay_logits'].shape==(55,1,152064) and raw['hidden'].shape==raw['normalized'].shape==raw['replay_normalized'].shape==(55,1,3584),'Full55-row native tensor shapes differ')
        need(all(raw[k].dtype==torch.float16 and bool(torch.isfinite(raw[k]).all()) for k in ('hidden','normalized','replay_normalized','logits','replay_logits')),'Finite native FP16 states required')
        need(torch.equal(raw['normalized'],raw['replay_normalized'])==record['normalized_exact']
             and native.tensor_info(raw['position_ids'])==batch['layout']['position_ids']
             and native.tensor_info(raw['attention_mask'])==batch['metadata']['input_identity']['attention_mask'],'Native normalization/layout archive differs')
        metrics=replay_metrics(torch,raw['logits'][:,0],raw['replay_logits'][:,0]);need(len(record['replay'])==55,'Replay row count differs')
        for a,b in zip(metrics,record['replay']):need(a['row']==b['row'] and a['top1_equal']==b['top1_equal'] and a['passed']==b['passed']
            and a['passed'] and math.isclose(a['tv'],b['tv'],rel_tol=1e-9,abs_tol=1e-12),'Independent native head decision differs')
        head_rows+=55;max_head_tv=max(max_head_tv,max(m['tv'] for m in metrics));values=scores(torch,raw['logits'][:,0])
        for key,value in values.items():
            saved=raw['probabilities'][key];need(value.shape==saved.shape and value.dtype==saved.dtype,'Probability schema differs')
            if key in ('top1_id','valid'):need(torch.equal(value,saved),'Native argmax/validity differs')
            else:need(torch.allclose(value,saved,rtol=1e-12,atol=1e-15),'Full-vocabulary probability roundoff exceeded');max_roundoff=max(max_roundoff,float((value-saved).abs().max()))
        for label in labels:
            i=label['row'];top=int(values['top1_id'][i]);valid=top in (15,16);target=label['label']
            rows.append(dict(batch_id=batch['batch_id'],**label,top1_id=top,top1_text=processor.tokenizer.decode([top],skip_special_tokens=False),
                valid=valid,prediction=(top-15) if valid else None,correct=valid and top==15+target,
                p0=float(values['p0'][i]),p1=float(values['p1'][i]),digit_mass=float(values['digit_mass'][i]),
                logit_gap=float(values['logit_gap'][i]),target_probability=float(values['p1' if target else 'p0'][i])))
    need(len(rows)==4860 and head_rows==4950 and Counter(r['label'] for r in rows)=={0:3240,1:1620},'Complete denominator differs')
    ledger=accounting(out);own=[j for j in ledger['jobs'] if j['job_id']==str(summary['slurm_job_id'])];pilot=[j for j in ledger['jobs'] if j['job_id']==str(released['slurm_job_id'])]
    need(len(own)==len(pilot)==1 and all(j['state']=='COMPLETED' and j['exit_code']=='0:0' and j['gpus']==1 for j in own+pilot),'Successful full/pilot GPU allocations absent')
    groups={'all':summary_metrics(rows),'per_step':{str(s):summary_metrics([r for r in rows if r['step']==s]) for s in STEPS},
        'per_pair':{str(i):summary_metrics([r for r in rows if r['pair_index']==i]) for i in range(15)},
        'per_class':{str(k):summary_metrics([r for r in rows if r['label']==k]) for k in (0,1)},
        'per_person':{c:summary_metrics([r for r in rows if r['person']==c]) for c in CHARS},
        'per_depicted_room':{r:summary_metrics([x for x in rows if x['depicted_room']==r]) for r in ROOMS}}
    feasible=all(r['valid'] and r['correct'] for r in rows);save(out/'rows.json',rows)
    analysis=dict(schema_version=1,protocol=PROTOCOL,passed=True,completed=True,computational_integrity_passed=True,source_sha256=frozen,
        plan_file=summary['plan_file'],plan_sha256=summary['plan_sha256'],native_identity=plan['native_identity'],native_identity_sha256=plan['native_identity_sha256'],
        run_directory=str(directory),run_summary_sha256=sha(directory/'summary.json'),profile_release=released,local_judgments=4860,head_rows=4950,
        pilot_head_rows=110,full_metrics=groups,feasibility_passed=feasible,rows_file=str(out/'rows.json'),rows_sha256=sha(out/'rows.json'),
        maximum_probability_roundoff=max_roundoff,maximum_native_head_tv=max_head_tv,accounting=ledger,no_fit=True,no_generation=True,no_global_answer_score=True,
        interpretation='Every invalid or wrong local argmax remains in the denominator. A failure stops this fixed interface proposal; no filtered subset or oracle replacement.')
    save(out/'analysis.json',analysis)
    (out/'REPORT.md').write_text('# Generic inclusion measurement\n\nComputational audit passed. Fixed complete-interface feasibility: **'+str(feasible)+'**.\n\n[All4860 rows, errors, strata and provenance](analysis.json). No learned aggregation or global-answer evaluation.\n')
    return dict(passed=True,completed=True,source_sha256=frozen,analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),
        feasibility_passed=feasible,local_judgments=4860,allocated_gpu_seconds=ledger['allocated_gpu_seconds'])



def verify_report(path):
    """Return passed complete-interface analysis after JSON/source/raw-hash checks.

    For prospective renderer release; this performs no model/head/tensor work.
    A completed pilot alone can never authorize rendering a join benchmark.
    """
    path=Path(path).resolve();path=path/'summary.json' if path.is_dir() else path;summary=read(path)
    need(summary['passed'] and summary['completed'] and summary['feasibility_passed']
         and summary['local_judgments']==4860 and summary['source_sha256']==sources(),'Complete passed inclusion report required')
    need(sha(summary['analysis_file'])==summary['analysis_sha256'],'Independent analysis changed');analysis=read(summary['analysis_file'])
    plan=verify_plan(analysis['plan_file']);need(analysis['passed'] and analysis['completed'] and analysis['computational_integrity_passed']
        and analysis['feasibility_passed'] and analysis['local_judgments']==4860 and analysis['head_rows']==4950
        and analysis['native_identity']==plan['native_identity'] and analysis['native_identity_sha256']==plan['native_identity_sha256']
        and analysis['source_sha256']==summary['source_sha256'],'Complete native inclusion binding differs')
    for n,h in summary['source_sha256'].items():need(sha(path.parent/'source'/n.replace('/','_'))==h,'Report source snapshot differs')
    need(sha(analysis['rows_file'])==analysis['rows_sha256'],'Complete measurement rows changed');rows=read(analysis['rows_file'])
    need(len(rows)==4860 and all(r['valid'] is True and r['correct'] is True and r['top1_id']==15+r['label'] for r in rows)
         and Counter(r['label'] for r in rows)=={0:3240,1:1620},'Incomplete/invalid native inclusion measurements')
    runfile=Path(analysis['run_directory'])/'summary.json';need(sha(runfile)==analysis['run_summary_sha256'],'Full GPU summary changed');run=read(runfile)
    need(run['profile'] is False and run['completed'] and run['passed'] and run['batch_indices']==list(range(90))
         and run['source_sha256']==summary['source_sha256'] and run['native_identity_sha256']==analysis['native_identity_sha256'],'Full audit release differs')
    need(sha(run['records_file'])==run['records_sha256'],'Full native record index changed')
    records=read(run['records_file']);need(len(records)==90,'Full native record inventory differs')
    for record in records:need(sha(record['raw_file'])==record['raw_sha256'],'Native raw evidence changed')
    need(verify_profile(analysis['profile_release']['directory'],plan)==analysis['profile_release'],'Pilot release changed')
    return analysis


def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    for mode in ('check','profile','run','report'):group.add_argument('--'+mode,action='store_true')
    parser.add_argument('--plan',type=Path);parser.add_argument('--profile-directory',type=Path);parser.add_argument('--run-directory',type=Path);args=parser.parse_args()
    gpu=args.profile or args.run;native.require_slurm(gpu=gpu)
    if not gpu:need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),'CPU Slurm required')
    mode=next(m for m in ('check','profile','run','report') if getattr(args,m));out=OUT/f'{mode}_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False)
    frozen=snapshot(out);start=time.perf_counter()
    try:
        if args.check:result=check(out,frozen)
        elif gpu:
            need(args.plan is not None and (args.profile or args.profile_directory is not None),'Exact CPU plan and measured profile required');result=run(args,out,frozen)
        else:need(args.run_directory is not None,'Complete full-audit directory required');result=report(args,out,frozen)
        need(sources()==frozen,'Sources changed during execution')
        if result is not None:result.update(seconds=time.perf_counter()-start,slurm_job_id=os.environ['SLURM_JOB_ID']);save(out/'summary.json',result)
        print(json.dumps(dict(passed=True,directory=str(out))),flush=True)
    except BaseException as exc:save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,seconds=time.perf_counter()-start));raise


if __name__=='__main__':main()

"""Independent, source-held ordinary joint LoRA artifact audit; never runs a VLM.

Prospective policy: CPU native norm/head total variation must be <= .02;
CPU/GPU argmax differences are descriptive. Exact GPU base/zero and serialized
adapter roundtrip parity remain mandatory. Earlier strict reports are unchanged.
Profile software validity and the measured main forecast are separate results.
"""
from pathlib import Path
import argparse
from collections import Counter, defaultdict
import json
import math
import os
import random
import re
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import diagnose_native_identity_join_joint_lora_v2 as p
from scripts.stage_native_vision_v6_teacher import need,read,save,sha,object_sha
OWN=('scripts/report_native_identity_join_joint_lora.py','slurm/native_identity_join_joint_lora_report.sbatch')
OUT=REPO/'outputs/native_aggregation_vlm/identity_join_joint_lora_audit'
DATA=Path('/mnt/data/gabriele/gnn_transformer/identity_join_joint_lora_audit')
PROTOCOL='identity_join_joint_lora_independent_audit'
POLICY=dict(profile_cpu_seconds=300,main_cpu_seconds=1800,cpu_cores=4,cpu_memory_gib=16,cpu_native_tv_max=.02,
    cpu_argmax_gate=False,same_gpu_zero_and_reload_exact=True,full_ce_absolute_tolerance=2e-6,
    no_vlm_forward=True,no_backward=True,no_fitting=True,all_numerical_evidence_before_failure=True)


def sources():return {name:sha(REPO/name) for name in OWN}


def inherited_sources():
    result={}
    for mapping in (p.inherited_sources(),p.sources()):
        for name,digest in mapping.items():
            need(name not in result or result[name]==digest,'Conflicting inherited audit source');result[name]=digest
    return result


def bind(path,digest,bindings=None):
    path=Path(path);need(sha(path)==digest,'Changed bound artifact: '+str(path))
    if bindings is not None:bindings[str(path)]=digest


def record(path,bindings):
    path=Path(path);digest=sha(path);bind(path,digest,bindings);return read(path)


def summary_path(path):
    path=Path(path).resolve();return path/'summary.json' if path.is_dir() else path


def archive(directory,own,inherited,bindings):
    need(record(directory/'source_hashes.json',bindings)==own
         and record(directory/'inherited_sources.json',bindings)==dict(source_sha256=inherited),'Archived source maps differ')
    for name,digest in {**inherited,**own}.items():bind(REPO/name,digest,bindings)
    for name,digest in own.items():bind(directory/'source'/name.replace('/','_'),digest,bindings)


def snapshot(out):
    own=sources();(out/'source').mkdir()
    for name,digest in own.items():
        file=out/'source'/name.replace('/','_');file.write_bytes((REPO/name).read_bytes());bind(file,digest)
    save(out/'source_hashes.json',own);save(out/'inherited_sources.json',dict(source_sha256=inherited_sources()));return own


def score(ids,target):
    need(isinstance(ids,list) and 1<=len(ids)<=4 and all(type(v) is int and 0<=v<152064 for v in ids),'Invalid generated IDs')
    complete=ids[-1] in (151645,151643)
    need(not any(v in (151645,151643) for v in ids[:-1]) and (complete or len(ids)==4),'Incorrect native EOS chronology')
    return dict(exact=ids==target,first_token_correct=ids[0]==target[0],completed=complete,truncated=not complete,finish_reason='eos' if complete else 'length')


def criterion(rows):
    need(len(rows)==len({r['sid'] for r in rows})==216,'All216 natural answers required')
    groups=defaultdict(list)
    for row in rows:groups[row['base_contrast_id']].append(row)
    orientations=('original','flipped')
    need(Counter(r['orientation_version'] for r in rows)=={'original':108,'flipped':108} and len(groups)==18
         and all(len(g)==12 and {(r['variant'],r['n_frames'],r['orientation_version']) for r in g}
             =={(v,n,o) for v in range(3) for n in (8,16) for o in orientations} for g in groups.values()),'Complete family ownership differs')
    counts={o:sum(r['exact'] for r in rows if r['orientation_version']==o) for o in orientations}
    whole=sum(counts.values());families=sum(all(r['exact'] for r in g) for g in groups.values())
    return dict(passed=whole>=206 and min(counts.values())>=103 and families>=16,whole_correct=whole,
        first_correct=sum(r['first_token_correct'] for r in rows),contexts=216,orientation_correct=counts,
        complete_families=families,families=18,thresholds=dict(pooled=206,per_orientation=103,complete_families=16),
        natural_whole_answer_and_eos=True,training_only=True)


def project(measured,normal_only=False):
    micro=measured['microbatches'];natural=measured['natural'];optim=measured['optimizer_seconds']
    need(len(micro)==(14 if normal_only else 16) and len(natural)==8 and len(optim)==2,'Fixed profile timing inventory differs')
    T={n:max(r['seconds'] for r in micro if r['n_frames']==n) for n in (8,16)}
    G={n:max(T[n],max(r['preparation_seconds']+r['work_seconds']*4/r['generated_tokens'] for r in natural if r['n_frames']==n)) for n in (8,16)}
    S=measured['setup_seconds'];O=max(optim);C=measured['checkpoint_seconds']
    need(all(math.isfinite(v) and v>0 for v in (S,O,C,*T.values(),*G.values())),'Invalid measured full-work timing')
    seconds=S+1.25*(1296*T[8]+1296*T[16]+324*O+108*G[8]+108*G[16]+2*C)+60.
    return dict(passed=seconds<=3600,setup_seconds=S,per_N_micro_seconds=T,optimizer_seconds=O,per_N_generation_seconds=G,
        checkpoint_roundtrip_seconds=C,projected_seconds=seconds,cap_seconds=3600,multiplier=1.25,reserve_seconds=60,
        empirical_projection_not_latency_guarantee=True,main_implemented=False,main_release=False)


def json_equal(a,b):return object_sha(a)==object_sha(b)


def inputs(plan_path,bindings):
    plan=record(plan_path,bindings);proof=record(plan_path.parent/'summary.json',bindings)
    need(plan['protocol']==p.PROTOCOL and plan['policy']==p.POLICY and proof['passed'] is proof['completed'] is True
         and proof['plan_sha256']==sha(plan_path) and plan['source_sha256']==p.sources()
         and plan['inherited_source_sha256']==p.inherited_sources(),'Passed frozen V2 CPU preparation required')
    archive(plan_path.parent,plan['source_sha256'],plan['inherited_source_sha256'],bindings)
    for mapping in (plan['input_bindings'],plan['runtime_bindings']):
        for file,digest in mapping.items():bind(file,digest,bindings)
    need(plan['original_failure']==p.verify_original_failure({}) and plan['no_native_model_or_head_forward'] is True
         and plan['confirmation_predictions_accessed'] is False and plan['confirmation_reserved_seed']==91726342,'Failure preservation/data boundary differs')
    tests=plan['tests'];fixture=tests['target_condensation']
    need(tests['passed'] is fixture['passed'] is True and fixture['widened_visual_selector_rejected'] is True
         and set(fixture['effective_targets'])==set(fixture['actual_wrapped_modules'])==set(p.TARGETS)
         and fixture['adapter_tensors']==224 and len(fixture['stored_targets'])<112,'Actual112-module PEFT condensation fixture required')
    rows=read(plan['rows_file']);prepared=read(plan['prepared_file']);order=read(plan['order_file'])
    need(len(rows)==len({r['sid'] for r in rows})==216 and set(prepared)=={r['sid'] for r in rows}
         and Counter(r['n_frames'] for r in rows)=={8:108,16:108}
         and Counter(r['orientation_version'] for r in rows)=={'original':108,'flipped':108},'Complete216 prepared training bank required')
    rng=random.Random(24);expected=[]
    for epoch in range(12):
        indices=list(range(216));rng.shuffle(indices)
        expected.extend(dict(epoch=epoch+1,position=j,index=i,sid=rows[i]['sid'],target_length=len(rows[i]['target_ids'])) for j,i in enumerate(indices))
    need(order==expected and object_sha(order)==plan['order_object_sha256'] and sha(plan['order_file'])==plan['order_sha256'],'Twelve advancing-seed epochs differ')
    by_sid={r['sid']:r for r in rows};cases=plan['profile_cases'];base=next(r['base_pair_id'] for r in rows if r['gold']=='Sandra' and r['orientation_version']=='original')
    selected=[next(r['sid'] for r in rows if r['gold']=='Sandra' and r['base_pair_id']==base and r['orientation_version']==o and r['n_frames']==n) for o in ('original','flipped') for n in (8,16)]
    widest=[next(r['sid'] for r in rows if r['n_frames']==n and prepared[r['sid']]['teacher_width']==max(prepared[x['sid']]['teacher_width'] for x in rows if x['n_frames']==n)) for n in (8,16)]
    need(cases==dict(parity_sids=selected,training_sids=widest,microbatch_sids=widest*4,roundtrip_sids=widest),'Prospective profile cases differ')
    h=sum(len(by_sid[s]['target_ids']) for s in widest)
    need(plan['head_rows']==dict(training=sum(r['target_length'] for r in order),by_update=[sum(r['target_length'] for r in order[i:i+8]) for i in range(0,2592,8)],
        profile_training=8*h,profile_roundtrip=2*h,profile_natural_cap=32,profile_total_cap=10*h+32),'Native teacher row arithmetic differs')
    old_ref=plan['original_check'];bind(old_ref['plan_file'],old_ref['plan_sha256'],bindings);bind(old_ref['summary_file'],old_ref['summary_sha256'],bindings)
    old=read(old_ref['plan_file'])
    for key in ('rows_file','prepared_file','order_file'):bind(old[key],old['runtime_bindings'][old[key]],bindings)
    semantic=lambda item:{k:v for k,v in item.items() if k not in ('file','sha256')}
    need(rows==read(old['rows_file']) and order==read(old['order_file']) and cases==old['profile_cases']
         and {k:semantic(v) for k,v in prepared.items()}=={k:semantic(v) for k,v in read(old['prepared_file']).items()},'Original CPU metadata/order parity differs')
    return plan,rows,prepared,order


def native_modules(torch,plan,bindings):
    ref=plan['orientation_plan'];bind(ref['file'],ref['sha256'],bindings);parent=read(ref['file'])
    bind(parent['native_model_file'],parent['native_model_sha256'],bindings)
    packet=torch.load(parent['native_model_file'],map_location='cpu',weights_only=True);identity=plan['native_identity']
    need(packet['schema_version']==1 and packet['native_identity']==identity==parent['native_identity']
         and packet['rms_norm_eps']==identity['rms_norm_eps'],'Original native archive identity differs')
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    import inspect
    bind(inspect.getfile(Qwen2RMSNorm),identity['norm_source_sha256'],bindings)
    for key in ('norm','head'):
        value=packet[key+'_weight'];need(p.tensor_info(value)==identity[key+'_weight'] and value.dtype==torch.float16 and not value.requires_grad,'Original FP16 native weights differ')
    norm=Qwen2RMSNorm(3584,eps=packet['rms_norm_eps']);norm.weight=torch.nn.Parameter(packet['norm_weight'],requires_grad=False)
    head=torch.nn.Linear(3584,152064,bias=False,device='meta',dtype=torch.float16);head.weight=torch.nn.Parameter(packet['head_weight'],requires_grad=False)
    return norm.eval(),head.eval()


def tensor_packet(torch,file,digest,bindings):
    bind(file,digest,bindings);return torch.load(file,map_location='cpu',weights_only=True)


def head_audit(torch,capture,modules,data,key):
    tail,normal,actual_input,actual=(capture[k] for k in ('norm_query_input','normalized_query','head_input','head_logits'))
    L=tail.shape[1];need(tail.shape==normal.shape==actual_input.shape==(1,L,3584) and actual.shape==(1,L,152064)
        and all(t.dtype==torch.float16 and not t.requires_grad for t in (tail,normal,actual_input,actual)),'Captured selected-row shape/dtype differs')
    finite=all(bool(torch.isfinite(t).all()) for t in (tail,normal,actual_input,actual));norm,head=modules
    with torch.inference_mode():replayed_normal=norm(tail);replayed=head(replayed_normal)
    file=data/(key+'.pt');torch.save(dict(normalized_query=replayed_normal,head_logits=replayed),file)
    finite=finite and bool(torch.isfinite(replayed_normal).all()) and bool(torch.isfinite(replayed).all())
    metrics=[]
    for i in range(L):
        tv=float((actual[0,i].double().softmax(-1)-replayed[0,i].double().softmax(-1)).abs().sum()/2) if finite else None
        metrics.append(dict(row=i,full_vocabulary_tv=tv,argmax_exact=bool(actual[0,i].argmax()==replayed[0,i].argmax()),
            gpu_argmax=int(actual[0,i].argmax()),cpu_argmax=int(replayed[0,i].argmax()),passed=finite and tv<=.02))
    normal_identity=torch.equal(normal,actual_input)
    return dict(passed=normal_identity and all(r['passed'] for r in metrics),rows=L,metrics=metrics,
        normalized_equals_head_input=normal_identity,norm_max_abs_error=float((normal.float()-replayed_normal.float()).abs().max()),
        replay_file=str(file),replay_sha256=sha(file),cpu_argmax_gate=False)


def teacher_audit(torch,capture,row,item,packet,modules,data,key,backward,replay=True):
    tf=packet['teacher'];L=len(row['target_ids']);S=item['teacher_width']
    if 'inputs' in tf:p.validate_teacher(torch,packet['bundle'],tf,row)
    need(capture['sid']==row['sid'] and capture['target_ids']==row['target_ids'] and capture['target_positions']==list(range(S-L,S))
         and capture['teacher_input_identity']==item['teacher_input_identity']==tf['input_identity'] and capture['norm_input_shape']==[1,S,3584]
         and capture['decoder_layer_forward_calls']==[1]*28 and torch.equal(capture['position_ids'],tf['position_ids'])
         and capture['backward'] is backward and capture['backward_coefficient']==(1/8 if backward else 0.),'Actual teacher position/target/call ownership differs')
    head_result=head_audit(torch,capture,modules,data,key) if replay else None
    need(capture['head_logits'].shape==(1,L,152064) and capture['head_input'].shape==capture['normalized_query'].shape==capture['norm_query_input'].shape==(1,L,3584)
         and all(capture[k].dtype==torch.float16 and not capture[k].requires_grad for k in ('head_logits','head_input','normalized_query','norm_query_input')),'Full teacher tensor inventory differs')
    direct_identity=torch.equal(capture['normalized_query'],capture['head_input'])
    need(len(capture['position_ce'])==L,'All target-position losses required')
    logits=capture['head_logits'][0].double();targets=torch.tensor(row['target_ids']);ce=logits.logsumexp(-1)-logits[torch.arange(L),targets]
    discrepancy=max(abs(float(a)-float(b)) for a,b in zip(ce,capture['position_ce']))
    loss_error=abs(float(ce.mean())-capture['loss'])
    return dict(passed=(head_result is None or head_result['passed']) and direct_identity and math.isfinite(discrepancy) and math.isfinite(loss_error)
        and discrepancy<=2e-6 and loss_error<=2e-6,sid=row['sid'],head=head_result,normalized_equals_head_input=direct_identity,
        maximum_position_ce_error=discrepancy,mean_ce_error=loss_error,full_name_and_EOS=True,backward_reexecuted=False)


def generation_audit(torch,result,row,item,packet,plan,modules,data,key):
    bundle=packet['bundle'];p.validate_joint(torch,bundle);meta=bundle['metadata'];width=meta['prompt_width'];ids=result['generated_ids'];raw=result['raw_logits'];t=len(ids)
    scored=score(ids,row['target_ids'])
    need(meta==item['metadata'] and all(result['metadata'][k]==v for k,v in meta.items())
         and result['metadata']['native_identity_sha256']==plan['native_identity_sha256']
         and result['metadata']['scene_input_identity']==object_sha(dict(metadata=meta,native_identity_sha256=plan['native_identity_sha256']))
         and raw.shape==(t,152064) and raw.dtype==torch.float32 and torch.equal(raw,raw.half().float())
         and bool(torch.isfinite(raw).all()) and raw.argmax(-1).tolist()==ids,'Native raw output/input ownership differs')
    need(all(result[k]==scored[k] for k in ('completed','truncated','finish_reason')) and result['parameter_versions_unchanged'] is result['hooks_removed'] is result['rope_restored'] is True,'Native generation state/termination flags differ')
    expected=dict(model=t,visual=1,language=t,norm=t,head=t,broadcast=0,fusion=0,conditioning=0,probe_head=0)
    need(result['counters']==expected and all(len(result[k])==t for k in ('native_inputs','native_positions','shapes','logit_records','profile_head')),'Native trajectory call inventory differs')
    layout=result['metadata']['layout'];position0=result['native_positions'][0];teacher_pos=packet['teacher']['position_ids'];delta=torch.tensor(layout['rope_deltas'],dtype=torch.int64).view(1,1,1)
    need(torch.equal(position0,teacher_pos[:,:,:width]),'Generation prefill position differs from bound teacher prefix')
    audits=[]
    for i,(seen,pos,shape,log,cap) in enumerate(zip(result['native_inputs'],result['native_positions'],result['shapes'],result['logit_records'],result['profile_head'])):
        expected_ids=bundle['inputs']['input_ids'] if i==0 else torch.tensor([[ids[i-1]]],dtype=torch.int64)
        need(torch.equal(seen['input_ids'],expected_ids) and torch.equal(seen['attention_mask'],torch.ones((1,width+i),dtype=torch.int64))
             and seen['has_pixels'] is (i==0) and seen['past_length']==(0 if i==0 else width+i-1),'Observed native KV/input chronology differs')
        if i:need(pos.shape==(4,1,1) and int(pos[0,0,0])==width+i-1 and torch.equal(pos[1:],(delta+width+i-1).expand(3,1,1)),'Observed cached text/mRoPE differs')
        need(result['metadata']['generation_position_ids'][i]==p.tensor_info(pos)
             and shape==dict(norm_input_shape=[1,width if i==0 else 1,3584],norm_output_shape=[1,width if i==0 else 1,3584],head_input_shape=[1,1,3584],head_output_shape=[1,1,152064],norm_dtype='torch.float16',head_input_dtype='torch.float16',head_output_dtype='torch.float16')
             and log['step']==i and log['top1_token_id']==ids[i] and log['top1_logit']==float(raw[i,ids[i]])
             and log['native_dtype']=='torch.float16' and log['vocabulary_size']==152064
             and torch.equal(cap['head_logits'][0,0],raw[i].half()),'Actual norm/head/logit observation differs')
        audits.append(head_audit(torch,cap,modules,data,f'{key}_token_{i}'))
    return dict(passed=all(a['passed'] for a in audits),sid=row['sid'],generated_ids=ids,score=scored,counters=expected,heads=audits)


def adapter_audit(torch,directory,bindings):
    initial_ref=record(directory/'initial_adapter.json',bindings);roundtrip=record(directory/'roundtrip.json',bindings)
    initial=tensor_packet(torch,initial_ref['file'],initial_ref['sha256'],bindings)
    trained=tensor_packet(torch,roundtrip['file'],roundtrip['sha256'],bindings)
    bind(initial_ref['config_file'],initial_ref['config_sha256'],bindings);config=read(initial_ref['config_file'])
    need(initial['seed']==24 and initial['profile_initialization_only'] is True and trained['updates']==2 and trained['profile_only'] is True
         and initial['peft_config']==trained['peft_config']==config==record(directory/'actual_peft_config_before_contract.json',bindings),'Actual complete PEFT configuration differs')
    contract=record(directory/'adapter_contract.json',bindings);need(contract==initial['contract']==trained['contract'],'Adapter identity changed across save/load')
    names={target+suffix for target in p.TARGETS for suffix in ('.lora_A.default.weight','.lora_B.default.weight')}
    need(set(initial['adapter'])==set(trained['adapter'])==set(contract['tensors'])==set(initial_ref['tensors'])==names
         and len(names)==224 and contract['targets']==list(p.TARGETS) and contract['trainable_parameters']==10092544
         and contract['active_adapter']=='default' and contract['rank']==16 and contract['alpha']==32 and contract['dropout']==.05 and contract['scaling']==2.,'Exact language-only adapter scope differs')
    total=0;changed=Counter()
    for name in sorted(names):
        a=initial['adapter'][name];b=trained['adapter'][name];is_a='.lora_A.' in name;out=512 if any('.'+q+'_proj.' in name for q in ('k','v')) else 3584
        shape=[16,3584] if is_a else [out,16]
        need(list(a.shape)==list(b.shape)==shape and a.dtype==b.dtype==torch.float32 and not a.requires_grad and not b.requires_grad
             and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()) and p.tensor_info(a)==initial_ref['tensors'][name]
             and contract['tensors'][name]['shape']==shape and contract['tensors'][name]['dtype']=='torch.float32','Adapter raw bytes/shape/finiteness differ')
        if not is_a:need(torch.count_nonzero(a)==0,'Initial B must be exactly zero')
        total+=a.numel();changed['A' if is_a else 'B']+=not torch.equal(a,b)
    need(total==10092544 and all(changed[k]>0 for k in ('A','B')),'Both adapter factors must learn after two updates')
    need(config['r']==16 and config['lora_alpha']==32 and config['lora_dropout']==.05 and config['bias']=='none' and config['init_lora_weights'] is True
         and not config['use_rslora'] and not config['use_dora'] and config['modules_to_save'] is None,'Configured PEFT semantics differ')
    need(set(roundtrip['tensors_exact'])==set(roundtrip['original_reset_exact'])==names and all(roundtrip['tensors_exact'].values())
         and all(roundtrip['original_reset_exact'].values()) and all(roundtrip[k] is True for k in ('passed','contract_restored','full_peft_config_exact','same_parameter_objects','base_frozen','eval_forwards_excluded'))
         and roundtrip['original_file']==initial_ref['file'] and roundtrip['original_sha256']==initial_ref['sha256']
         and roundtrip['config_file']==initial_ref['config_file'] and roundtrip['config_sha256']==initial_ref['config_sha256'],'Exact original reset/trained reload evidence differs')
    return dict(passed=True,parameters=total,tensors=224,changed_factors=dict(changed),contract=contract,initial_reference=initial_ref,
        raw_original_and_trained_bytes_verified=True,reset_and_live_parameter_identity_scope='producer observations; no VLM reconstructed')


def amended_projection(measured,original):
    micro=measured['microbatches']
    need([(r['step'],r['microbatch']) for r in micro]==[(s,j) for s in (1,2) for j in range(1,9)],'Complete fixed timing observations required')
    instrumented=micro[:2];normal=micro[2:]
    need([r['n_frames'] for r in instrumented]==[8,16] and Counter(r['n_frames'] for r in normal)=={8:7,16:7},'Exactly two predeclared instrumented calls required')
    surcharge=sum(r['seconds'] for r in instrumented)
    value=project(dict(measured,setup_seconds=measured['setup_seconds']+surcharge,microbatches=normal),normal_only=True)
    return json.loads(json.dumps(dict(value,accounting_rule='uninstrumented_main_with_once_only_profile_instrumentation',original_projection=original,
        original_setup_seconds=measured['setup_seconds'],instrumented_cold_seconds=surcharge,
        instrumented_microbatches=[{k:r[k] for k in ('step','microbatch','n_frames','seconds')} for r in instrumented],normal_microbatch_count=14)))


def packet_for(torch,item,row,bindings):
    value=tensor_packet(torch,item['file'],item['sha256'],bindings)
    need(value['bundle']['metadata']==item['metadata'] and p.tensor_info(value['teacher']['position_ids'])==item['teacher_position_identity'],'CPU packet metadata differs')
    p.validate_teacher(torch,value['bundle'],value['teacher'],row);return value


def profile_descriptor(path,summary,initial):
    return dict(file=str(path),sha256=sha(path),plan_file=summary['plan_file'],plan_sha256=summary['plan_sha256'],
        initial_adapter_file=initial['file'],initial_adapter_sha256=initial['sha256'],adapter_config_file=initial['config_file'],adapter_config_sha256=initial['config_sha256'])


def resources(out,profile_path,main_path=None):
    command=['sacct','-X','-u',os.environ['USER'],'--starttime=2026-09-01','--parsable2','--noheader',
        '--format=JobIDRaw,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,Start,End']
    raw=subprocess.run(command,capture_output=True,text=True,check=True).stdout;file=out/'sacct.psv';file.write_text(raw)
    expected={p.original.JOB:('443880','FAILED','1:0',34),p.JOB:(profile_path.parent.name.removeprefix('profile_'),'COMPLETED','0:0',240)}
    if main_path is not None:expected['identity_join_joint_lora_main']=(main_path.parent.name.removeprefix('run_'),'COMPLETED','0:0',3600)
    rows=[]
    for line in raw.splitlines():
        fields=line.split('|')
        if len(fields)!=9 or fields[1] not in expected:continue
        job,name,partition,state,code,seconds,tres,start,end=fields
        generic=re.findall(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',tres);typed=re.findall(r'(?:^|,)gres/gpu:[^=,]+=(\d+)(?=,|$)',tres)
        rows.append(dict(job_id=job,name=name,partition=partition,state=state,exit_code=code,seconds=int(seconds),gpus=int(generic[0]) if generic else sum(map(int,typed)),start=start,end=end,alloc_tres=tres))
    need(len(rows)==len(expected) and Counter(r['name'] for r in rows)==Counter(expected.keys()),'All failed and completed allocations must be charged')
    for r in rows:
        job,state,code,cap=expected[r['name']]
        need(r['job_id']==job and r['state']==state and r['exit_code']==code and r['partition']=='gpu' and r['gpus']==1
             and 0<r['seconds']<=cap and (r['seconds']==34 if job=='443880' else True),'Allocation identity/state/cap differs')
    total=sum(r['seconds'] for r in rows);need(total<=274+(3600 if main_path is not None else 0),'Cumulative GPU allocation exceeded')
    return dict(passed=True,jobs=rows,allocated_gpu_seconds=total,original_failed_gpu_seconds=34,
        cap_gpu_seconds=274+(3600 if main_path is not None else 0),file=str(file),sha256=sha(file),command=command)


def numerical_summary(heads):
    metrics=[r for h in heads for r in h['metrics']]
    return dict(cpu_head_calls=len(heads),cpu_head_rows=sum(h['rows'] for h in heads),
        cpu_argmax_disagreements=[dict(query=i,**r) for i,r in enumerate(metrics) if not r['argmax_exact']],
        maximum_cpu_native_head_tv=max((r['full_vocabulary_tv'] for r in metrics if r['full_vocabulary_tv'] is not None),default=None),
        cpu_argmax_gate=False,all_numerical_evidence_collected=True)


def hardware_audit(config,plan):
    hardware=config['hardware'];quant=hardware['quantization']
    need(hardware['gpu']=='NVIDIA B200' and hardware['precision']==plan['precision'] and quant['load_in_4bit'] is quant['bnb_4bit_use_double_quant'] is True
         and quant['bnb_4bit_quant_type']=='nf4' and str(quant['bnb_4bit_compute_dtype']).removeprefix('torch.')=='bfloat16','Native NF4/BF16 compute and precision policy differ')


def profile_audit(torch,path,out,data,bindings):
    summary=record(path,bindings);directory=path.parent;config=record(directory/'config.json',bindings)
    need(summary['passed'] is summary['completed'] is True and summary['phase']=='profile' and summary['protocol']==p.PROTOCOL
         and all(summary[k]==v for k,v in config.items()) and not (directory/'failure.json').exists(),'Completed V2 software profile required')
    plan_path=Path(summary['plan_file']);bind(plan_path,summary['plan_sha256'],bindings)
    plan,rows,prepared,order=inputs(plan_path,bindings);by_sid={r['sid']:r for r in rows}
    archive(directory,plan['source_sha256'],plan['inherited_source_sha256'],bindings);hardware_audit(config,plan)
    need(record(directory/'request.json',bindings)==dict(phase='profile',plan=str(plan_path),source_sha256=p.sources()),'Profile request ownership differs')
    release=record(p.OUT/'source_release.json',bindings)
    need(release['status']=='frozen_before_execution' and release['source_sha256']==p.sources() and release['inherited_source_sha256']==p.inherited_sources(),'Frozen V2 release differs')
    for key in ('policy','source_sha256','inherited_source_sha256','original_failure','original_check','orientation_plan','training_stage','confirmation_report','native_identity','native_identity_sha256','packages'):
        need(config[key]==plan[key],'Profile/CPU binding differs: '+key)
    need(config['cases']==plan['profile_cases'] and config['profile_only'] is True and config['main_release'] is False
         and config['confirmation_predictions_accessed'] is False and summary['profile_checkpoint_must_not_initialize_main'] is summary['future_main_restarts_seed24'] is summary['no_main_or_validation_execution'] is True,'Profile-only scope differs')
    need(Path(config['data_directory'])==p.DATA/directory.name and Path(config['checkpoint_directory'])==p.CKPT/directory.name,'Profile artifact roots differ')
    for key in ('endpoint','measurements','artifacts'):bind(summary[key+'_file'],summary[key+'_sha256'],bindings)
    endpoint=read(summary['endpoint_file']);measured=read(summary['measurements_file']);artifacts=read(summary['artifacts_file'])
    for file,digest in artifacts.items():bind(file,digest,bindings)
    need(endpoint['passed'] is endpoint['zero_lora_exact'] is endpoint['roundtrip_logits_exact'] is endpoint['base_parameters_unchanged'] is True,'Same-backend fidelity endpoint failed')
    bind(endpoint['base_before_file'],endpoint['base_before_sha256'],bindings);base=read(endpoint['base_before_file'])
    need(bool(base) and all('.lora_' not in name for name in base),'Original base identity inventory differs')
    adapters=adapter_audit(torch,directory,bindings);need(adapters['contract']==endpoint['adapter_contract'],'Final live adapter contract differs')
    modules=native_modules(torch,plan,bindings);weights_before={k:p.tensor_info(m.weight) for k,m in zip(('norm','head'),modules)}
    counts=Counter();heads=[];natural=[];pairraw={};raw_files=set();cases=plan['profile_cases'];head_started=time.perf_counter()
    for state in ('base','zero_lora'):
        for index,sid in enumerate(cases['parity_sids']):
            key=f'{state}_{index:02d}';small=record(directory/(key+'.json'),bindings);timing=record(directory/(key+'_timing.json'),bindings)
            need(all(timing[k]==v for k,v in small.items()) and timing==measured['natural'][len(natural)]
                 and timing['sid']==sid and timing['state']==state and timing['n_frames']==by_sid[sid]['n_frames']
                 and timing['four_token_seconds']==timing['preparation_seconds']+timing['work_seconds']*4/timing['generated_tokens'],'Natural timing/order differs')
            need(Path(small['file'])==Path(config['data_directory'])/(key+'.pt') and artifacts[small['file']]==small['sha256'],'Natural raw artifact path differs')
            raw=tensor_packet(torch,small['file'],small['sha256'],bindings);raw_files.add(small['file'])
            need(set(raw)=={'sid','state','result'} and raw['sid']==sid and raw['state']==state,'Natural raw packet identity differs')
            packet=packet_for(torch,prepared[sid],by_sid[sid],bindings)
            audited=generation_audit(torch,raw['result'],by_sid[sid],prepared[sid],packet,plan,modules,data,key)
            need(small['generated_tokens']==len(audited['generated_ids']) and small['counters']==audited['counters'],'Natural count sidecar differs')
            heads.extend(audited['heads']);counts.update(audited['counters']);natural.append(audited)
            pairraw[(state,sid)]={k:raw['result'][k] for k in ('raw_logits','generated_ids','shapes')}
    parity=[]
    for sid in cases['parity_sids']:
        a=pairraw[('base',sid)];b=pairraw[('zero_lora',sid)]
        parity.append(dict(sid=sid,exact_logits=torch.equal(a['raw_logits'],b['raw_logits']),exact_generated_ids=a['generated_ids']==b['generated_ids'],exact_shapes=a['shapes']==b['shapes'],maximum_tv=0.))
    need(parity==record(directory/'zero_lora_parity.json',bindings) and all(all(r[k] for k in ('exact_logits','exact_generated_ids','exact_shapes')) for r in parity),'Exact GPU base/zero parity differs')
    del pairraw
    natural_seconds=time.perf_counter()-head_started;teacher_started=time.perf_counter();teacher=[];roundtrip={};train_keys=[]
    for step in (1,2):
        for j,sid in enumerate(cases['microbatch_sids'],1):
            key=f'training_{step}_{j:02d}';small=record(directory/(key+'.json'),bindings);timing=record(directory/(key+'_timing.json'),bindings)
            need(all(timing[k]==v for k,v in small.items()) and timing==measured['microbatches'][len(teacher)]
                 and small['step']==step and small['microbatch']==j and small['sid']==sid and small['n_frames']==by_sid[sid]['n_frames'],'Teacher timing/order differs')
            raw=tensor_packet(torch,small['file'],small['sha256'],bindings);raw_files.add(small['file']);train_keys.append((step,j))
            need(Path(small['file'])==Path(config['data_directory'])/(key+'.pt') and artifacts[small['file']]==small['sha256']
                 and raw['step']==step and raw['microbatch']==j and raw['n_frames']==by_sid[sid]['n_frames']
                 and raw['counters']==dict(model=1,visual=1,language=1,norm=1,head=1),'Training capture metadata differs')
            if (step,j) in ((1,1),(1,2)):
                kernel=raw['kernel_inventory'];need('aten::scaled_dot_product_attention' in kernel['sdpa_operators'] and kernel['cuda_kernels'] and kernel['existing_training_call_only'] is True,'Actual selected profiler inventory missing')
            else:need('kernel_inventory' not in raw,'Unexpected profiled training call')
            packet=packet_for(torch,prepared[sid],by_sid[sid],bindings);audit=teacher_audit(torch,raw,by_sid[sid],prepared[sid],packet,modules,data,key,True)
            need(small['loss']==raw['loss'] and small['target_rows']==len(raw['target_ids']) and small['all_gradients_finite'] is True
                 and small['accumulated_gradient_norms']['B']>0 and (small['accumulated_gradient_norms']['A']==0 if step==1 else small['accumulated_gradient_norms']['A']>0)
                 and all(math.isfinite(v) for v in small['accumulated_gradient_norms'].values()),'Observed first-zero then live A/B gradients differ')
            teacher.append(audit);heads.append(audit['head']);counts.update(raw['counters']);counts['backward']+=1
        opt=record(directory/f'optimizer_{step}.json',bindings);opt_timing=record(directory/f'optimizer_{step}_timing.json',bindings)
        need(opt['step']==step and opt['learning_rate']==2e-4 and opt['gradient_accumulation']==8 and opt['optimizer_steps']==[float(step)]*224
             and opt['base_frozen'] is True and math.isfinite(opt['unclipped_gradient_norm']) and opt_timing==dict(step=step,seconds=measured['optimizer_seconds'][step-1]),'Exact two AdamW updates differ')
        counts['optimizer']+=1
    for phase in ('before','after'):
        for index,sid in enumerate(cases['roundtrip_sids']):
            key=f'roundtrip_{phase}_{index}';small=record(directory/(key+'.json'),bindings)
            need(small['sid']==sid and Path(small['file'])==Path(config['data_directory'])/(key+'.pt') and artifacts[small['file']]==small['sha256'],'Reload teacher packet ownership differs')
            raw=tensor_packet(torch,small['file'],small['sha256'],bindings);raw_files.add(small['file']);packet=packet_for(torch,prepared[sid],by_sid[sid],bindings)
            audit=teacher_audit(torch,raw,by_sid[sid],prepared[sid],packet,modules,data,key,False);teacher.append(audit);heads.append(audit['head'])
            counts.update(dict(model=1,visual=1,language=1,norm=1,head=1));roundtrip[(phase,sid)]=raw['head_logits']
    reload_exact=all(torch.equal(roundtrip[('before',sid)],roundtrip[('after',sid)]) for sid in cases['roundtrip_sids'])
    need(reload_exact,'Exact GPU serialized adapter roundtrip logits differ')
    need(record(directory/'roundtrip_timing.json',bindings)==dict(seconds=measured['checkpoint_seconds'],eval_forwards_excluded=True),'Checkpoint-only timing differs')
    need(len(measured['memory'])==16 and [(r['step'],r['microbatch']) for r in measured['memory']]==train_keys
         and all(0<r['max_allocated_bytes']<=r['max_reserved_bytes'] for r in measured['memory']),'All16 memory observations required')
    stats=numerical_summary(heads);native_rows=stats['cpu_head_rows'];tokens=sum(len(r['generated_ids']) for r in natural)
    need(dict(counts)==summary['counters']==endpoint['counters']==measured['counters'] and counts['model']==counts['language']==counts['norm']==counts['head']==tokens+20<=52
         and counts['visual']==28 and counts['backward']==16 and counts['optimizer']==2 and native_rows==summary['head_rows']==endpoint['head_rows']==measured['head_rows']
         ==plan['head_rows']['profile_training']+plan['head_rows']['profile_roundtrip']+tokens<=92,'Complete52/28/16/2 and92-row profile inventory differs')
    need(raw_files=={file for file in artifacts if Path(file).parent==Path(config['data_directory'])},'Unexpected/missing profile raw artifacts')
    need(weights_before=={k:p.tensor_info(m.weight) for k,m in zip(('norm','head'),modules)} and all(not v.requires_grad and v.grad is None for m in modules for v in m.parameters()),'CPU replay mutated native weights')
    original=json.loads(json.dumps(project(measured)));need(original==measured['projection']==summary['projection'] and summary['main_implementation_eligible']==original['passed'],'Original profile forecast differs')
    revised=amended_projection(measured,original)
    return dict(profile_audit_passed=all(a['passed'] for a in natural+teacher),profile=profile_descriptor(path,summary,adapters['initial_reference']),
        original_projection=original,main_projection=revised,main_projection_passed=revised['passed'],
        timing_amendment_after_profile_before_main=True,profile_natural_outcomes_descriptive_only=True,
        exact_gpu_zero_parity=parity,exact_gpu_reload_parity=reload_exact,adapter_audit=adapters,natural=natural,teacher=teacher,
        counters=dict(counts),**stats,resources=resources(out,path),no_main_release=True,
        audit_segment_seconds=dict(natural_heads_and_ownership=natural_seconds,teacher_heads_and_ownership=time.perf_counter()-teacher_started))


def verify_profile_audit(audit_summary_path,profile_summary_path):
    """JSON/hash-only release handoff. No tensors, models or numerical replay."""
    path=summary_path(audit_summary_path);profile_path=summary_path(profile_summary_path);summary=read(path)
    need(summary['protocol']==PROTOCOL and summary['phase']=='profile' and summary['passed'] is summary['completed'] is True
         and summary['source_sha256']==sources() and summary['inherited_source_sha256']==inherited_sources()
         and summary['policy']==POLICY and summary['profile_summary_file']==str(profile_path)
         and summary['profile_summary_sha256']==sha(profile_path) and not (path.parent/'failure.json').exists(),'Passed exact profile audit required')
    archive(path.parent,summary['source_sha256'],summary['inherited_source_sha256'],{})
    for key in ('analysis','artifacts','input_bindings','report'):bind(summary[key+'_file'],summary[key+'_sha256'])
    for file,digest in read(summary['input_bindings_file']).items():bind(file,digest)
    for file,digest in read(summary['artifacts_file']).items():bind(file,digest)
    analysis=read(summary['analysis_file'])
    need(analysis['passed'] is analysis['completed'] is analysis['profile_audit_passed'] is True and analysis['policy']==POLICY
         and analysis['profile']['file']==str(profile_path) and analysis['profile']['sha256']==sha(profile_path)
         and analysis['main_projection_passed']==analysis['main_projection']['passed']
         and analysis['all_numerical_evidence_collected'] is True,'Complete independent profile evidence required')
    source=read(profile_path);bind(source['measurements_file'],source['measurements_sha256']);measure=read(source['measurements_file'])
    original=json.loads(json.dumps(project(measure)))
    need(original==analysis['original_projection']==source['projection'] and analysis['main_projection']==amended_projection(measure,original),'Independent original/amended projections changed')
    return analysis


def semantic_contract(value):
    return {k:({name:{field:v for field,v in item.items() if field!='object_id'} for name,item in values.items()} if k=='tensors' else values) for k,values in value.items()}


def main_checkpoints(torch,directory,profile,bindings):
    initial_ref=record(directory/'initial_adapter.json',bindings);final_ref=record(directory/'final_adapter.json',bindings)
    initial=tensor_packet(torch,initial_ref['file'],initial_ref['sha256'],bindings);final=tensor_packet(torch,final_ref['file'],final_ref['sha256'],bindings)
    original=tensor_packet(torch,profile['initial_adapter_file'],profile['initial_adapter_sha256'],bindings)
    bind(profile['adapter_config_file'],profile['adapter_config_sha256'],bindings);config=read(profile['adapter_config_file'])
    proof=record(directory/'initialization.json',bindings);observed=record(directory/'actual_peft_config_before_contract.json',bindings)
    need(initial['step']==0 and initial['epoch']==0 and final['step']==324 and final['epoch']==12
         and initial['full_training_final'] is False and final['full_training_final'] is True
         and all(packet['seed']==24 and packet['no_resume'] is packet['no_checkpoint_selection'] is True and packet['profile_initialization']==profile for packet in (initial,final))
         and initial['peft_config']==final['peft_config']==original['peft_config']==config==observed
         and initial_ref['peft_config']==final_ref['peft_config']==config
         and initial['contract']==final['contract'] and semantic_contract(initial['contract'])==semantic_contract(original['contract']),'Exact initialization/final configuration provenance differs')
    need(all(proof[k] is True for k in ('fresh_seed24_bytes_exact','full_config_exact','no_seed_reset_after_install','main_rng_recorded'))
         and proof['profile_rng_equality_claimed'] is proof['profile_two_update_weights_used_for_initialization'] is False
         and proof['trainable_parameters']==10092544 and proof['trainable_tensors']==224,'Fresh initialization observations differ')
    names=list(initial['adapter']);expected={target+suffix for target in p.TARGETS for suffix in ('.lora_A.default.weight','.lora_B.default.weight')}
    need(set(names)==set(final['adapter'])==set(original['adapter'])==expected and len(names)==224,'Main adapter scope differs')
    changed=Counter()
    for name in names:
        a,b,c=initial['adapter'][name],final['adapter'][name],original['adapter'][name]
        need(a.dtype==b.dtype==c.dtype==torch.float32 and a.shape==b.shape==c.shape and torch.equal(a,c)
             and all(not v.requires_grad and bool(torch.isfinite(v).all()) for v in (a,b,c))
             and p.tensor_info(a)==initial_ref['tensors'][name]==proof['profile_original_tensors'][name]
             and p.tensor_info(b)==final_ref['tensors'][name],'Fresh unfitted bytes/final finite adapter differ')
        changed['A' if '.lora_A.' in name else 'B']+=not torch.equal(a,b)
    need(sum(v.numel() for v in initial['adapter'].values())==10092544,'Main parameter count differs')
    for step,packet,ref in ((0,initial,initial_ref),(324,final,final_ref)):
        optimizer=packet['optimizer'];groups=optimizer['param_groups'];state=optimizer['state']
        need(len(groups)==1 and groups[0]['params']==list(range(224)) and groups[0]['lr']==2e-4 and groups[0]['betas']==(.9,.999)
             and groups[0]['eps']==1e-8 and groups[0]['weight_decay']==.01 and len(state)==(224 if step else 0),'Actual optimizer checkpoint configuration differs')
        for index,name in enumerate(names):
            if not step:continue
            item=state[index];need(set(item)=={'step','exp_avg','exp_avg_sq'} and float(item['step'])==324
                and all(item[k].shape==packet['adapter'][name].shape and item[k].dtype==torch.float32 and bool(torch.isfinite(item[k]).all()) for k in ('exp_avg','exp_avg_sq')),'Actual final Adam state differs')
        need(ref['step']==step and ref['rng_recorded'] is ref['restricted_reload_exact'] is True
             and set(packet['rng'])=={'python','torch_cpu','torch_cuda'} and packet['rng']['torch_cpu'].dtype==torch.uint8
             and len(packet['rng']['torch_cuda'])==1 and packet['rng']['torch_cuda'][0].dtype==torch.uint8,'Recorded fresh main RNG/checkpoint proof differs')
        need(ref['optimizer']==dict(parameter_names=names,step=step,state_tensors=224 if step else 0,learning_rate=2e-4,betas=[.9,.999],epsilon=1e-8,weight_decay=.01),'Saved optimizer descriptor differs')
    return dict(passed=True,parameters=10092544,tensors=224,changed_factors=dict(changed),parameter_names=names,
        original_profile_bytes_exact=True,profile_trained_checkpoint_consumed=False,profile_rng_equality_claimed=False,initial=initial_ref,final=final_ref)


def main_audit(torch,path,out,data,bindings):
    summary=record(path,bindings);directory=path.parent;config=record(directory/'config.json',bindings)
    need(summary['protocol']=='identity_join_joint_lora_main' and summary['phase']=='main' and summary['passed'] is summary['completed'] is summary['computational_passed'] is True
         and all(summary[k]==v for k,v in config.items()) and not (directory/'failure.json').exists(),'Completed fixed main run required')
    own_names={'scripts/train_native_identity_join_joint_lora.py','docs/paper/NATIVE_AGGREGATION_JOINT_LORA_MAIN_PROPOSAL.md','slurm/native_identity_join_joint_lora_main.sbatch'}
    need(set(config['source_sha256'])==own_names and config['inherited_source_sha256']=={**inherited_sources(),**sources()},'Separately held main source closure differs')
    archive(directory,config['source_sha256'],config['inherited_source_sha256'],bindings)
    profile=config['profile'];profile_path=Path(profile['file']);bind(profile_path,profile['sha256'],bindings)
    audit_ref=config['profile_audit'];bind(audit_ref['file'],audit_ref['sha256'],bindings);prior=verify_profile_audit(audit_ref['file'],profile_path)
    need(prior['profile']==profile and prior['main_projection_passed'] is True and config['main_projection']==prior['main_projection']
         and config['profile_projection']==prior['original_projection'],'Positive independent amended cost certificate required')
    plan_path=Path(config['plan_file']);bind(plan_path,config['plan_sha256'],bindings);plan,rows,prepared,order=inputs(plan_path,bindings)
    hardware_audit(config,plan)
    need(config['plan_file']==profile['plan_file'] and config['plan_sha256']==profile['plan_sha256'] and config['confirmation_predictions_accessed'] is config['validation_inference'] is False,'Reserved confirmation must remain unobserved')
    for key in ('orientation_plan','training_stage','confirmation_report','native_identity','native_identity_sha256','packages'):need(config[key]==plan[key],'Main CPU dependency differs: '+key)
    policy=config['policy']
    fixed=dict(contexts=216,epochs=12,scene_presentations=2592,gradient_accumulation=8,updates=324,rank=16,alpha=32,dropout=.05,seed=24,
        trainable_parameters=10092544,adapter_tensors=224,learning_rate=2e-4,betas=[.9,.999],epsilon=1e-8,weight_decay=.01,clip=1.,checkpointing=False,
        maximum_new_tokens=4,main_seconds=3600,main_attempts=1,model_calls_cap=3456,vision_calls=2808,training_head_rows=5760,evaluation_head_rows_cap=864,backwards=2592,
        no_validation_inference=True,no_checkpoint_selection=True,no_resume=True,no_early_stopping=True,no_prepare_model_for_kbit_training=True,
        base_storage_dtypes_preserved=True,train_adapter_dtype='torch.float32',ordinary_joint_images_first=True,no_prompt_wrapper=True,no_broadcast=True,
        profile_trained_checkpoint_forbidden=True,unrecorded_profile_rng_equality_claimed=False)
    need(all(policy[k]==v for k,v in fixed.items()),'Fixed main optimization protocol differs')
    for field in ('endpoint','training','optimizer','evaluations','artifacts'):bind(summary[field+'_file'],summary[field+'_sha256'],bindings)
    endpoint=read(summary['endpoint_file']);artifacts=read(summary['artifacts_file']);evaluations=read(summary['evaluations_file'])
    for file,digest in artifacts.items():bind(file,digest,bindings)
    need(all(endpoint[k] is True for k in ('passed','computational_passed','base_parameters_unchanged','all_adapter_gradients_finite','final_adapter_frozen','final_adapter_unchanged_during_evaluation','full_peft_config_exact')),'Final optimizer/base/evaluation state differs')
    for field in ('base_before','initial_adapter','final_adapter'):bind(endpoint[field+'_file'],endpoint[field+'_sha256'],bindings)
    checkpoints=main_checkpoints(torch,directory,profile,bindings);names=checkpoints['parameter_names'];modules=native_modules(torch,plan,bindings)
    weight_before={k:p.tensor_info(m.weight) for k,m in zip(('norm','head'),modules)}
    # Validate each image/teacher packet once, retaining only small position tensors
    # during the2,592-example audit. Full pixels are never kept as a whole bank.
    compact={}
    for row in rows:
        packet=packet_for(torch,prepared[row['sid']],row,bindings)
        compact[row['sid']]=dict(teacher={k:packet['teacher'][k] for k in ('position_ids','input_identity')})
    del packet
    training=[json.loads(line) for line in Path(summary['training_file']).read_text().splitlines()]
    optimizers=[json.loads(line) for line in Path(summary['optimizer_file']).read_text().splitlines()]
    need(len(training)==2592 and len(optimizers)==324 and len(evaluations)==216,'Complete fixed training/update/evaluation inventory required')
    replay_indices={216*epoch+position for epoch in range(12) for position in (0,215)}
    counts=Counter();teacher=[];heads=[];raw_files=set();training_rows=0;teacher_started=time.perf_counter()
    for index,(entry,small) in enumerate(zip(order,training)):
        row=rows[entry['index']];sid=row['sid'];step=index//8+1;micro=index%8+1
        expected=dict(order_index=index,epoch=entry['epoch'],epoch_position=entry['position'],step=step,microbatch=micro,sid=sid,n_frames=row['n_frames'])
        need(all(small[k]==v for k,v in expected.items()) and small['target_ids']==row['target_ids'] and small['target_rows']==len(row['target_ids'])
             and small['all_gradients_non_none'] is small['all_gradients_finite'] is True and small['gradient_tensors']==224
             and small['backward_coefficient']==1/8 and all(math.isfinite(v) and v>=0 for v in small['accumulated_gradient_norms'].values()),'Training schedule/gradient/target ownership differs')
        need(Path(small['file'])==Path(config['data_directory'])/f'train_{index+1:04d}.pt' and artifacts[small['file']]==small['sha256'],'Training raw artifact identity differs')
        capture=tensor_packet(torch,small['file'],small['sha256'],bindings);raw_files.add(small['file'])
        need(all(capture[k]==v for k,v in expected.items()) and small['loss']==capture['loss'] and small['position_ce']==capture['position_ce'] and small['target_positions']==capture['target_positions']
             and capture['counters']==small['counters']==dict(model=1,visual=1,language=1,norm=1,head=1) and 'kernel_inventory' not in capture,'Actual main training capture must be unprofiled')
        audited=teacher_audit(torch,capture,row,prepared[sid],compact[sid],modules,data,f'train_{index+1:04d}',True,replay=index in replay_indices)
        teacher.append(audited)
        if audited['head'] is not None:heads.append(audited['head'])
        counts.update(capture['counters']);counts['backward']+=1;training_rows+=len(row['target_ids'])
        if micro==8:
            opt=optimizers[step-1]
            need(opt['step']==step and opt['completed_scene_presentations']==index+1 and opt['clip']==1. and math.isfinite(opt['unclipped_gradient_norm'])
                 and opt['base_frozen'] is True and opt['optimizer']==dict(parameter_names=names,step=step,state_tensors=224,learning_rate=2e-4,betas=[.9,.999],epsilon=1e-8,weight_decay=.01),'All324 Adam updates must retain exact names/moments/parameters')
            counts['optimizer']+=1
    teacher_seconds=time.perf_counter()-teacher_started;need(len(heads)==24 and training_rows==5760,'Fixed24 teacher replay/all5,760 CE rows differ')
    natural=[];outcomes=[];head_started=time.perf_counter()
    for index,(row,small) in enumerate(zip(rows,evaluations)):
        sid=row['sid'];need(small['index']==index and small['sid']==sid and all(small[k]==row[k] for k in ('n_frames','orientation_version','base_contrast_id','target_ids','gold')),'Final native cohort/order differs')
        need(Path(small['file'])==Path(config['data_directory'])/f'eval_{index:03d}.pt' and artifacts[small['file']]==small['sha256'],'Native evaluation raw artifact differs')
        raw=tensor_packet(torch,small['file'],small['sha256'],bindings);raw_files.add(small['file']);need(set(raw)=={'sid','result'} and raw['sid']==sid,'Final raw packet identity differs')
        packet=packet_for(torch,prepared[sid],row,bindings);audited=generation_audit(torch,raw['result'],row,prepared[sid],packet,plan,modules,data,f'eval_{index:03d}')
        need(small['generated_ids']==audited['generated_ids'] and small['whole_correct']==audited['score']['exact'] and small['first_correct']==audited['score']['first_token_correct']
             and all(small[k]==raw['result'][k] for k in ('completed','truncated','text','raw_text','counters')),'Natural efficacy must use actual GPU IDs only')
        natural.append(audited);heads.extend(audited['heads']);counts.update(audited['counters']);outcomes.append(dict(row,generated_ids=audited['generated_ids'],**audited['score']))
    competence=criterion(outcomes);need(competence==summary['competence']==endpoint['competence']==record(directory/'competence.json',bindings),'Independently reconstructed whole-answer criterion differs')
    eval_rows=sum(len(r['generated_ids']) for r in outcomes);stats=numerical_summary(heads)
    need(raw_files=={file for file in artifacts if Path(file).parent==Path(config['data_directory'])} and dict(counts)==summary['counters']==endpoint['counters'],'All retained main raw/call inventories required')
    need(counts['model']==counts['language']==counts['norm']==counts['head']==2592+eval_rows<=3456 and counts['visual']==2808 and counts['backward']==2592 and counts['optimizer']==324
         and summary['training_head_rows']==endpoint['training_head_rows']==5760 and summary['evaluation_head_rows']==endpoint['evaluation_head_rows']==eval_rows<=864
         and summary['total_head_rows']==endpoint['total_head_rows']==5760+eval_rows and endpoint['completed_scene_presentations']==2592 and endpoint['completed_updates']==324
         and stats['cpu_head_rows']<=936 and stats['cpu_head_calls']==24+eval_rows,'Fixed main native/replay call budget differs')
    need(summary['downstream_eligibility']==competence['passed'] and summary['downstream_release'] is False and summary['valid_negative_outcomes_retained'] is True,'Competence and software validity must remain separate')
    need(weight_before=={k:p.tensor_info(m.weight) for k,m in zip(('norm','head'),modules)} and all(not v.requires_grad and v.grad is None for m in modules for v in m.parameters()),'CPU replay changed native frozen parameters')
    return dict(main_audit_passed=all(r['passed'] for r in teacher+natural),competence=competence,profile=profile,profile_audit=audit_ref,
        main_projection=config['main_projection'],main_projection_passed=True,checkpoints=checkpoints,teacher=teacher,natural=natural,outcomes=outcomes,
        counters=dict(counts),**stats,teacher_replay_order_indices=sorted(replay_indices),training_head_rows=5760,evaluation_head_rows=eval_rows,
        all2592_teacher_CE_and_ownership_audited=True,valid_negative_is_not_software_failure=True,downstream_release=False,
        resources=resources(out,profile_path,path),audit_segment_seconds=dict(teacher_CE_and_sampled_heads=teacher_seconds,natural_heads_and_ownership=time.perf_counter()-head_started))


def self_test(torch):
    need(score([15,151645],[15,151645])['exact'] and not score([15,151643],[15,151645])['exact'],'Canonical EOS fixture failed')
    try:score([151645,15],[15,151645])
    except ValueError:pass
    else:raise AssertionError('Interior EOS accepted')
    rows=[dict(sid=f'{f}_{v}_{n}_{o}',base_contrast_id=str(f),variant=v,n_frames=n,orientation_version=o,exact=True,first_token_correct=True)
        for f in range(18) for v in range(3) for n in (8,16) for o in ('original','flipped')]
    for o in ('original','flipped'):
        for r in [r for r in rows if r['base_contrast_id']=='0' and r['orientation_version']==o][:5]:r['exact']=False
    need(criterion(rows)['passed'] and criterion(rows)['whole_correct']==206,'103 per orientation/206 pooled must pass with enough families')
    [r for r in rows if r['exact'] and r['orientation_version']=='original'][0]['exact']=False
    need(not criterion(rows)['passed'],'102 per orientation must fail')
    for r in rows:r['exact']=True
    for o in ('original','flipped'):
        for f in range(5):next(r for r in rows if r['base_contrast_id']==str(f) and r['orientation_version']==o)['exact']=False
    need(criterion(rows)['whole_correct']==206 and not criterion(rows)['passed'],'Pooled score cannot substitute for complete families')
    micro=[dict(step=s,microbatch=j,n_frames=8 if j%2 else 16,seconds=10. if (s,j) in ((1,1),(1,2)) else .01) for s in (1,2) for j in range(1,9)]
    natural=[dict(n_frames=n,preparation_seconds=.001,work_seconds=.01,generated_tokens=4,four_token_seconds=.011) for _ in range(4) for n in (8,16)]
    measure=dict(setup_seconds=1.,microbatches=micro,natural=natural,optimizer_seconds=[.01,.01],checkpoint_seconds=.01)
    original=json.loads(json.dumps(project(measure)));amended=amended_projection(measure,original)
    need(not original['passed'] and amended['passed'] and amended['original_projection']==original and amended['instrumented_cold_seconds']==20.
         and amended['normal_microbatch_count']==14 and amended['per_N_micro_seconds']=={'8':.01,'16':.01},'Separated instrumentation accounting fixture failed')
    a=torch.tensor([0.,.0001],dtype=torch.float16);b=a.flip(0)
    tv=float((a.double().softmax(-1)-b.double().softmax(-1)).abs().sum()/2)
    need(a.argmax()!=b.argmax() and tv<=.02 and POLICY['cpu_argmax_gate'] is False,'Small cross-backend argmax difference must remain descriptive')
    need(Counter(['old','new'])==Counter(dict(old=1,new=2).keys()),'Allocation key-count fixture failed')
    replay_indices={216*e+i for e in range(12) for i in (0,215)}
    need(len(replay_indices)==24 and min(replay_indices)==0 and max(replay_indices)==2591,'Fixed main replay sample differs')
    return dict(passed=True,groups=7,canonical_EOS=True,orientation_and_family_boundaries=True,cost_amendment_retains_original=True,
        cross_backend_argmax_descriptive=True,allocation_key_counts=True,main_replay_sample_fixed=True,no_native_head_or_VLM_calls=True)


def publish(args,out,data,frozen,bindings,analysis,started):
    phase='profile' if args.profile is not None else 'main';input_path=summary_path(args.profile if phase=='profile' else args.main)
    passed=analysis['profile_audit_passed' if phase=='profile' else 'main_audit_passed']
    # Always retain all completed numerical results before enforcing TV/CE gates.
    analysis=dict(analysis,passed=passed,completed=True,phase=phase,protocol=PROTOCOL,policy=POLICY,
        source_sha256=frozen,inherited_source_sha256=inherited_sources(),input_summary_file=str(input_path),input_summary_sha256=sha(input_path))
    save(out/'analysis.json',analysis)
    save(out/'input_bindings.json',bindings)
    artifacts={str(file):sha(file) for file in sorted(data.rglob('*')) if file.is_file()};save(out/'artifacts.json',artifacts)
    lines=['# Independent ordinary joint LoRA audit','',
        'CPU norm/head TV must be at most .02. CPU/GPU argmax differences are descriptive; all efficacy uses actual GPU-generated IDs. Exact GPU zero-adapter and serialization parity remain mandatory. Earlier reports retain their original gates.','',
        f'Numerical audit: {"PASS" if passed else "FAIL"}. CPU head calls: {analysis["cpu_head_calls"]}; selected query rows: {analysis["cpu_head_rows"]}; maximum TV: {analysis["maximum_cpu_native_head_tv"]}.',
        f'CPU/GPU argmax differences: {len(analysis["cpu_argmax_disagreements"])}. Every scheduled numerical audit was collected before this decision.','']
    if phase=='profile':
        lines.extend([f'Original profile cost forecast: {analysis["original_projection"]["projected_seconds"]:.6f} seconds; passed={analysis["original_projection"]["passed"]}.',
            f'Separate prospective main accounting: {analysis["main_projection"]["projected_seconds"]:.6f} seconds; passed={analysis["main_projection_passed"]}.',
            'The new accounting was defined after profile timings and before any main fit: the two instrumented calls are priced once in setup, all14 ordinary calls determine per-example cost. No additional model call was run. A separate main source release is still required.'])
    else:
        c=analysis['competence'];lines.extend([f'Natural complete name plus EOS: {c["whole_correct"]}/216; orientation counts: {c["orientation_correct"]}; complete families: {c["complete_families"]}/18; competence passed={c["passed"]}.',
            'All2,592 teacher traces and5,760 target CE positions were audited. Native CPU replay covered the fixed24 first/last-per-epoch examples and every final generation query. A valid negative competence result is not a computational failure. No fresh inference is released.'])
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    elapsed=time.perf_counter()-started;cap=POLICY['profile_cpu_seconds' if phase=='profile' else 'main_cpu_seconds']
    need(elapsed<=cap and sources()==frozen,'Independent audit phase cap/source identity differs')
    need(passed,'Independent native TV or full-target CE audit failed; all scheduled raw replays and metrics retained')
    value=dict(passed=True,completed=True,phase=phase,protocol=PROTOCOL,policy=POLICY,source_sha256=frozen,inherited_source_sha256=inherited_sources(),
        analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),artifacts_file=str(out/'artifacts.json'),artifacts_sha256=sha(out/'artifacts.json'),
        input_bindings_file=str(out/'input_bindings.json'),input_bindings_sha256=sha(out/'input_bindings.json'),
        report_file=str(out/'REPORT.md'),report_sha256=sha(out/'REPORT.md'),cpu_head_calls=analysis['cpu_head_calls'],cpu_head_rows=analysis['cpu_head_rows'],
        maximum_cpu_native_head_tv=analysis['maximum_cpu_native_head_tv'],cpu_argmax_disagreements=len(analysis['cpu_argmax_disagreements']),elapsed_seconds=elapsed,
        phase_cpu_cap_seconds=cap,audit_segment_seconds=analysis['audit_segment_seconds'])
    if phase=='profile':value.update(profile_summary_file=str(input_path),profile_summary_sha256=sha(input_path),profile_audit_passed=True,main_projection_passed=analysis['main_projection_passed'])
    else:value.update(main_summary_file=str(input_path),main_summary_sha256=sha(input_path),main_audit_passed=True,competence=analysis['competence'])
    save(out/'summary.json',value);return value


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--profile',type=Path);mode.add_argument('--main',type=Path);args=parser.parse_args()
    p.native.require_slurm();need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),'CPU-only independent audit required')
    phase='profile' if args.profile is not None else 'main';out=OUT/f'{phase}_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False)
    data=DATA/out.name;data.mkdir(parents=True,exist_ok=False);started=time.perf_counter();frozen=snapshot(out);bindings={}
    path=summary_path(args.profile if phase=='profile' else args.main)
    save(out/'request.json',dict(phase=phase,input_summary_file=str(path),source_sha256=frozen))
    try:
        import torch
        torch.set_num_threads(4);need(not torch.cuda.is_available(),'Independent audit must not have a GPU')
        save(out/'selftests.json',self_test(torch))
        analysis=profile_audit(torch,path,out,data,bindings) if phase=='profile' else main_audit(torch,path,out,data,bindings)
        value=publish(args,out,data,frozen,bindings,analysis,started);print(json.dumps(dict(passed=True,directory=str(out),phase=value['phase'])),flush=True)
    except BaseException as exc:
        save(out/'failure_input_bindings.json',bindings)
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,elapsed_seconds=time.perf_counter()-started,
            input_bindings_file=str(out/'failure_input_bindings.json'),input_bindings_sha256=sha(out/'failure_input_bindings.json'),partial_outputs_retained=True));raise


if __name__=='__main__':main()

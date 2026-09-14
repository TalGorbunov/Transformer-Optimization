"""Held oracle-code readout diagnostic: final training first-token outputs only.

The supplied answer code is an offline positive control, never a runtime method.
One600-update U-only fit; no vision/backbone calls, continuation or EOS scoring.
"""
from __future__ import annotations
import argparse
from collections import Counter
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import diagnose_native_identity_join_uniform as uniform
from scripts.stage_native_vision_v6_teacher import need,read,sha,save,object_sha
base=uniform.base
native=uniform.native
OUT=REPO/'outputs/native_aggregation_vlm/identity_join_readout'
DATA=Path('/mnt/data/gabriele/gnn_transformer/identity_join_readout')
CKPT=Path('/mnt/ckpts/gabriele/gnn_transformer/identity_join_readout')
PARENT=REPO/'outputs/native_aggregation_vlm/identity_join_uniform/check_443258/plan.json'
PARENT_SHA='f87d6d49bab87e3baacb436faed88b6561a1dcd3a71b7ccefdebafe2d79c12d9'
CONDITIONAL=REPO/'outputs/native_aggregation_vlm/identity_join_uniform/report_443271/summary.json'
CONDITIONAL_SHA='1be7e5c8e5c598efd71d996035985e6f8f797dd749e977f4ac7817fc0df424f0'
CONDITIONAL_ANALYSIS_SHA='c2176613b6b1ade6eb4c5a13e7ddfe23cb523b7d7709907a312f834bb5a07056'
PROPOSAL='docs/paper/NATIVE_AGGREGATION_READOUT_DIAGNOSTIC_PROPOSAL.md'
PROPOSAL_SHA='15aafed17f2815fc7544581735ca73a8288df97cb38b4fac74f9489a71ce18ec'
PROTOCOL='identity_join_oracle_code_readout'
RUN_JOB='identity_join_readout_run'
PEOPLE=('Sandra','Mary','Michael','John','Daniel','Laura','Peter','Emma','Noah')
POLICY=dict(protocol=PROTOCOL,seed=24,steps=600,batch_pairs=8,batch_scenes=16,training_contexts=108,
    pairs=54,families=18,hidden_size=3584,rank=96,classes=list(PEOPLE),stored_trainable_coordinates=344064,
    effective_degrees_of_freedom=32256,code='sqrt(96)*one_hot(class,96), FP32, directly into U after bypassing SiLU',
    initialization='U exactly zero',loss='mean_scene(first_token_CE/full_canonical_name_plus_EOS_length)',
    lr=.001,warmup=50,final_lr=1e-5,weight_decay=0.,clip_norm=1.,optimizer='AdamW',betas=[.9,.999],adam_epsilon=1e-8,
    schedule='exact frozen uniform persistent Random(24) pair order;600 batches of8 intact pairs',
    native_dtype='torch.float16',U_dtype='torch.float32',projection_tf32=False,cast_before_add=True,norm_calls=607,head_calls=607,
    training_head_calls=600,evaluation_head_calls=7,vlm_calls=0,vision_calls=0,
    first_token_fit=dict(correct=103,contexts=108,complete_families=16,families=18),
    gpu_seconds_cap=150,maximum_gpus=1,cpu_check_seconds=60,cpu_report_seconds=300,
    final_checkpoint_only=True,no_dev_or_test=True,no_runtime_oracle_claim=True,no_eos_or_whole_answer_claim=True)
OWN=('scripts/diagnose_native_identity_join_readout.py','scripts/report_native_identity_join_readout.py',
     'scripts/analyze_native_identity_join_uniform_geometry.py','slurm/native_identity_join_readout_check.sbatch',
     'slurm/native_identity_join_readout_run.sbatch','slurm/native_identity_join_readout_report.sbatch',PROPOSAL)


def sources():return {**uniform.sources(),**{name:sha(REPO/name) for name in OWN}}


def snapshot(out):
    frozen=sources();(out/'source').mkdir()
    for name,h in frozen.items():
        dest=out/'source'/name.replace('/','_');dest.write_bytes((REPO/name).read_bytes())
        need(sha(dest)==h,'Source changed during snapshot')
    save(out/'source_hashes.json',frozen)
    return frozen


def bind(path,bindings,expected=None):
    path=Path(path).resolve();value=sha(path)
    need(expected is None or value==expected,'Bound artifact differs: '+str(path));bindings[str(path)]=value
    return value


def conditional_gate(bindings):
    """Read/hash only the already independently verified uniform proof."""
    bind(REPO/PROPOSAL,bindings,PROPOSAL_SHA)
    bind(CONDITIONAL,bindings,CONDITIONAL_SHA);proof=read(CONDITIONAL)
    need(proof['passed'] is True and proof['completed'] is True and proof['phase']=='report'
         and proof['protocol']==uniform.PROTOCOL and proof['source_sha256']==uniform.sources(),
         'Complete computationally valid uniform report required')
    bind(proof['analysis_file'],bindings,CONDITIONAL_ANALYSIS_SHA)
    need(proof['analysis_sha256']==CONDITIONAL_ANALYSIS_SHA,'Uniform analysis digest differs')
    analysis=read(proof['analysis_file'])
    need(analysis['passed'] is True and analysis['completed'] is True and analysis['source_sha256']==uniform.sources()
         and analysis['plan_file']==str(PARENT) and analysis['plan_sha256']==PARENT_SHA
         and len(analysis['runs'])==1,'Uniform report/run/plan coverage differs')
    bind(PARENT,bindings,PARENT_SHA);parent=read(PARENT);cpu=PARENT.parent/'summary.json';bind(cpu,bindings)
    check=read(cpu)
    need(check['passed'] is True and check['completed'] is True and check['plan_sha256']==PARENT_SHA
         and check['source_sha256']==uniform.sources() and parent['source_sha256']==uniform.sources()
         and parent['policy']==uniform.POLICY,'Uniform CPU plan/source proof differs')
    for n,h in parent['source_sha256'].items():
        need(sha(PARENT.parent/'source'/n.replace('/','_'))==h
             and sha(CONDITIONAL.parent/'source'/n.replace('/','_'))==h,'Uniform source snapshots differ')
    record=analysis['runs'][0];run_dir=Path(record['directory']);sp=run_dir/'summary.json'
    bind(sp,bindings,record['summary_sha256']);run=read(sp)
    need(record['arm']==run['arm']=='uniform_half_ce' and run['passed'] is True and run['completed'] is True
         and run['computational_integrity_passed'] is True and run['steps']==600 and run['profile'] is False
         and run['source_sha256']==uniform.sources() and run['plan_sha256']==PARENT_SHA
         and run['trainability']==record['trainability']==proof['criteria']['uniform_half_ce']['trainability']
         and record['trainability']['passed'] is False,'Valid uniform training-criterion failure required')
    return parent,dict(summary_file=str(CONDITIONAL),summary_sha256=CONDITIONAL_SHA,
        analysis_file=proof['analysis_file'],analysis_sha256=CONDITIONAL_ANALYSIS_SHA,
        run_directory=str(run_dir),run_summary_file=str(sp),run_summary_sha256=sha(sp),
        trainability=record['trainability'],valid_training_failure=True)


def learning_rate(step):return uniform.learning_rate(step)


def codebook(torch):
    code=torch.zeros((9,96),dtype=torch.float32)
    code[torch.arange(9),torch.arange(9)]=math.sqrt(96)
    return code


def native_add(torch,g,delta):
    need(g.dtype==torch.float16 and delta.dtype==torch.float32 and g.shape==delta.shape,
         'Native FP16 global/FP32 delta contract differs')
    value=g+delta.to(torch.float16)
    need(bool(torch.isfinite(value).all()),'Native cast/add overflow')
    return value


def readout(torch,U,g,codes,norm,head):
    need(U.dtype==codes.dtype==torch.float32 and U.ndim==2 and codes.ndim==2
         and U.shape[1]==codes.shape[1]==96 and g.shape==(codes.shape[0],U.shape[0]),'Oracle readout shapes differ')
    delta=torch.nn.functional.linear(codes,U)
    fused=native_add(torch,g,delta)
    logits=head(norm(fused))
    need(logits.dtype==torch.float16 and bool(torch.isfinite(logits).all()),'Unmasked native FP16 logits invalid')
    return logits,dict(delta=delta,fused_global=fused)


def weighted_loss(torch,logits,first_ids,lengths):
    need(logits.ndim==2 and first_ids.shape==lengths.shape==(logits.shape[0],)
         and first_ids.dtype==lengths.dtype==torch.int64 and bool(((lengths==2)|(lengths==3)).all()),
         'First-token contribution target/length layout differs')
    nll=torch.nn.functional.cross_entropy(logits.float(),first_ids,reduction='none')
    contributions=nll/lengths.float()
    return contributions.mean(),nll,contributions


def native_module_identity(torch,identity):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    import transformers
    norm_file=Path(inspect.getfile(Qwen2RMSNorm)).resolve();head_file=Path(inspect.getfile(torch.nn.Linear)).resolve()
    need(sha(norm_file)==identity['norm_source_sha256']
         and str(torch.__version__)==identity['runtime']['torch_version']
         and str(transformers.__version__)==identity['runtime']['transformers_version'],
         'Installed native norm/torch/transformers identity differs')
    # The frozen Qwen multimodal decoder installs this exact norm and an ordinary bias-free Linear head.
    qwen_file=next(Path(p) for p in identity['native_api']['source_sha256'] if p.endswith('/qwen2_5_vl/modeling_qwen2_5_vl.py'))
    need(sha(qwen_file)==identity['native_api']['source_sha256'][str(qwen_file)],'Native decoder module source changed')
    source=qwen_file.read_text()
    need('self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)' in source
         and 'self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)' in source,
         'Frozen native norm/head owner declarations differ')
    return dict(norm_class=Qwen2RMSNorm.__module__+'.'+Qwen2RMSNorm.__qualname__,norm_source_file=str(norm_file),
        norm_source_sha256=sha(norm_file),head_class='torch.nn.modules.linear.Linear',head_source_file=str(head_file),
        head_source_sha256=sha(head_file),bias=False,rms_norm_eps=identity['rms_norm_eps'],
        torch_version=str(torch.__version__),transformers_version=str(transformers.__version__))


def native_modules(torch,packet,identity,device):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    need(packet['schema_version']==1 and packet['native_identity']==identity
         and packet['rms_norm_eps']==identity['rms_norm_eps'],'Frozen native archive metadata differs')
    for key in ('norm','head'):
        value=packet[key+'_weight']
        need(base.v7.tensor_info(value)==identity[key+'_weight'] and value.dtype==torch.float16
             and not value.requires_grad,'Frozen native tensor bytes differ')
    norm=Qwen2RMSNorm(3584,eps=packet['rms_norm_eps'])
    norm.weight=torch.nn.Parameter(packet['norm_weight'].to(device),requires_grad=False)
    head=torch.nn.Linear(3584,152064,bias=False,device='meta',dtype=torch.float16)
    head.weight=torch.nn.Parameter(packet['head_weight'].to(device),requires_grad=False)
    norm.eval();head.eval()
    return norm,head


def criteria(rows):
    need(len(rows)==108 and len({r['sid'] for r in rows})==108,'Readout denominator differs')
    groups={}
    for r in rows:groups.setdefault(r['contrast_id'],[]).append(r)
    need(len(groups)==18 and all(len(v)==6 and {(r['variant'],r['n_frames']) for r in v}==
         {(i,n) for i in range(3) for n in (8,16)} for v in groups.values()),'Readout complete-family coverage differs')
    correct=sum(r['first_token_correct'] for r in rows)
    complete=sum(all(r['first_token_correct'] for r in group) for group in groups.values())
    return dict(passed=correct>=103 and complete>=16,correct=correct,contexts=108,complete_families=complete,families=18,
        thresholds=POLICY['first_token_fit'],first_token_only=True,oracle_answer_code_supplied=True)


def tensor_self_test(torch):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    codes=codebook(torch)
    need(torch.allclose(codes.square().mean(1),torch.ones(9),rtol=1e-6,atol=1e-6)
         and int(torch.count_nonzero(codes))==9 and bool((codes[:,9:]==0).all()),'Post-SiLU RMS-one code fixture failed')
    g=torch.tensor([[1.]],dtype=torch.float16);delta=torch.tensor([[.0004884]],dtype=torch.float32)
    actual=native_add(torch,g,delta)
    need(torch.equal(actual,g+delta.half()) and not torch.equal(actual,(g.float()+delta).half()),'Cast-before-add fixture failed')
    logits=torch.tensor([[2.,0.,-1.],[2.,0.,-1.]],dtype=torch.float16,requires_grad=True)
    loss,nll,parts=weighted_loss(torch,logits,torch.tensor([0,0]),torch.tensor([2,3]))
    need(torch.allclose(parts,nll*torch.tensor([.5,1/3])) and torch.allclose(loss,(nll[0]/2+nll[1]/3)/2),
         'Original scene/token first-contribution weights differ')
    gradient=torch.autograd.grad(loss,logits)[0].float()
    need(torch.allclose(gradient[0],gradient[1]*1.5,rtol=.003,atol=.0001),'Length-weight gradient fixture failed')
    norm=Qwen2RMSNorm(4,eps=1e-6).half().eval().requires_grad_(False)
    head=torch.nn.Linear(4,3,bias=False,dtype=torch.float16).eval().requires_grad_(False)
    with torch.no_grad():head.weight.copy_(torch.tensor([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.]],dtype=torch.float16))
    U=torch.nn.Parameter(torch.zeros((4,96),dtype=torch.float32));live_codes=codes[:2].clone().requires_grad_()
    global_states=torch.tensor([[1.,2.,3.,4.],[4.,3.,2.,1.]],dtype=torch.float16)
    frozen={k:base.v7.tensor_info(v) for k,v in [('norm',norm.weight),('head',head.weight)]}
    logits,cap=readout(torch,U,global_states,live_codes,norm,head)
    need(torch.equal(cap['fused_global'],global_states) and bool((cap['delta']==0).all()),'Zero-U native identity failed')
    loss,_,_=weighted_loss(torch,logits,torch.tensor([0,1]),torch.tensor([2,3]));loss.backward()
    need(U.grad is not None and bool(torch.isfinite(U.grad).all()) and bool((U.grad[:,:2]!=0).any())
         and bool((U.grad[:,2:]==0).all()) and bool((live_codes.grad==0).all())
         and norm.weight.grad is head.weight.grad is None,'Zero-U/frozen-head/nonactive-column gradient fixture failed')
    optimizer=torch.optim.AdamW([U],lr=.001,weight_decay=0.);optimizer.step()
    need(bool((U[:,2:]==0).all()) and frozen=={k:base.v7.tensor_info(v) for k,v in [('norm',norm.weight),('head',head.weight)]},
         'Optimizer changed inactive columns/native weights')
    return dict(passed=True,groups=6,post_silu_rms_one=True,cast_before_add=True,original_first_token_weights=True,
        zero_U_identity_and_zero_input_gradient=True,active_U_gradient_and_inactive_columns=True,frozen_native_modules=True,
        no_pretrained_model_or_head_loaded=True)


def check(out,frozen):
    import torch
    torch.set_num_threads(4);bindings={};parent,conditional=conditional_gate(bindings)
    tests=tensor_self_test(torch)
    from scripts.report_native_identity_join_readout import self_test as report_self_test
    tests['reporter']=report_self_test(torch)
    from scripts.analyze_native_identity_join_uniform_geometry import self_test as geometry_self_test
    tests['geometry']=geometry_self_test(torch)
    for key in ('rows_file','scenes_file','pairs_file','order_file','features_file','native_model_file'):
        expected=parent['runtime_bindings'][str(Path(parent[key]).resolve())]
        bind(parent[key],bindings,expected)
    original_rows=read(parent['rows_file'])['train'];scenes=read(parent['scenes_file']);pairs=read(parent['pairs_file']);order=read(parent['order_file'])
    need(len(original_rows)==108 and len(scenes)==108 and order==uniform.order_pairs(pairs) and len(order)==4800,
         'Exact uniform training-only inventory/order differs')
    rows=[];targets={};all_gids=sorted({scenes[r['sample']['sid']]['global_feature_ids'][0] for r in original_rows})
    need(len(all_gids)==6,'Six training global empty-prefix queries required')
    for record in original_rows:
        sample=record['sample'];sid=sample['sid'];scene=scenes[sid];target=sample['target_ids'];gold=sample['gold']
        need(sample['split']=='train' and record['cell'] in ('train_N8','train_N16') and sample['n_frames'] in (8,16)
             and scene['target_ids']==target and scene['target_prefixes'][0]==[] and scene['gold']==gold
             and scene['question']==sample['question'] and gold in PEOPLE and target[-1]==151645
             and all(type(v) is int and 0<=v<152064 for v in target)
             and len(target)==(3 if gold in ('Sandra','Noah') else 2),'Canonical frozen first-query/name target mismatch')
        if gold in targets:need(targets[gold]==target,'A name has inconsistent frozen tokenizer targets')
        targets[gold]=target
        rows.append(dict(sid=sid,contrast_id=sample['contrast_id'],pair_id=sample['pair_id'],variant=sample['variant'],
            n_frames=sample['n_frames'],question=sample['question'],gold=gold,target_ids=target,class_index=PEOPLE.index(gold),
            global_feature_id=scene['global_feature_ids'][0],global_row=all_gids.index(scene['global_feature_ids'][0]),
            full_target_length=len(target),first_token_id=target[0]))
    need(set(targets)==set(PEOPLE) and len({v[0] for v in targets.values()})==9
         and Counter(r['gold'] for r in rows)==Counter({p:12 for p in PEOPLE})
         and Counter(r['n_frames'] for r in rows)==Counter({8:54,16:54}),'Balanced code/name/length inventory differs')
    criteria([dict(r,first_token_correct=True) for r in rows])
    # Persist labels, code semantics and all optimization presentations before tensor preparation or any fit.
    save(out/'rows.json',rows);save(out/'pairs.json',pairs);save(out/'order.json',order)
    save(out/'input_inventory.json',dict(schema_version=1,rows=rows,classes=list(PEOPLE),targets_by_name=targets,
        code_rule=POLICY['code'],global_feature_ids=all_gids,training_contexts=108,scene_presentations=9600,
        pair_presentations=4800,first_token_loss_positions=9600,all_canonical_targets=240,
        global_only=True,no_local_features=True,no_dev_or_test=True))
    packet=torch.load(parent['features_file'],map_location='cpu',weights_only=True)
    need(base.v7.tensor_info(packet['states'])==parent['feature_tensor'] and packet['feature_ids']==sorted(packet['feature_ids'])
         and len(set(packet['feature_ids']))==len(packet['feature_ids']),'Frozen feature packet differs')
    index={fid:i for i,fid in enumerate(packet['feature_ids'])}
    global_states=torch.stack([packet['states'][index[fid]].clone() for fid in all_gids]);del packet
    need(global_states.dtype==torch.float16 and global_states.shape==(6,3584) and bool(torch.isfinite(global_states).all()),
         'Frozen global-empty feature contract differs')
    data=DATA/out.name;ckpt=CKPT/out.name;data.mkdir(parents=True,exist_ok=False);ckpt.mkdir(parents=True,exist_ok=False)
    inputs=dict(schema_version=1,global_feature_ids=all_gids,global_states=global_states,codebook=codebook(torch))
    inputs_file=data/'inputs.pt';torch.save(inputs,inputs_file)
    initial_file=ckpt/'initial.pt';initial=torch.zeros((3584,96),dtype=torch.float32)
    torch.save(dict(schema_version=1,U=initial,step=0,policy=POLICY,source_sha256=frozen),initial_file)
    module_identity=native_module_identity(torch,parent['native_identity'])
    native_packet=torch.load(parent['native_model_file'],map_location='cpu',weights_only=True)
    norm,head=native_modules(torch,native_packet,parent['native_identity'],'cpu')
    del native_packet,norm,head
    runtime_bindings={}
    for path in (out/'rows.json',out/'pairs.json',out/'order.json',out/'input_inventory.json',inputs_file,initial_file,Path(parent['native_model_file'])):
        bind(path,runtime_bindings)
    plan=dict(schema_version=1,protocol=PROTOCOL,policy=POLICY,source_sha256=frozen,tests=tests,
        uniform_plan=dict(file=str(PARENT),sha256=PARENT_SHA),conditional_report=conditional,
        input_bindings=bindings,runtime_bindings=runtime_bindings,rows_file=str(out/'rows.json'),pairs_file=str(out/'pairs.json'),
        order_file=str(out/'order.json'),order_object_sha256=object_sha(order),input_inventory_file=str(out/'input_inventory.json'),
        inputs_file=str(inputs_file),inputs_sha256=sha(inputs_file),input_tensors={k:base.v7.tensor_info(inputs[k]) for k in ('global_states','codebook')},
        global_feature_ids=all_gids,initial_file=str(initial_file),initial_sha256=sha(initial_file),initial_U=base.v7.tensor_info(initial),
        native_model_file=parent['native_model_file'],native_model_sha256=parent['native_model_sha256'],
        native_identity=parent['native_identity'],native_identity_sha256=parent['native_identity_sha256'],native_module_identity=module_identity,
        no_pretrained_backbone_loaded=True,no_pretrained_head_forward=True)
    save(out/'plan.json',plan);(out/'plan.sha256').write_text(sha(out/'plan.json')+'\n')
    return dict(passed=True,completed=True,phase='check',protocol=PROTOCOL,source_sha256=frozen,
        plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),tests=tests,conditional_report=conditional,
        training_contexts=108,global_empty_features=6,first_token_positions=108,no_pretrained_backbone_loaded=True,
        no_pretrained_head_forward=True)


def verify_plan(path,ancestors=False):
    path=Path(path).resolve();plan=read(path)
    need(plan['protocol']==PROTOCOL and plan['policy']==POLICY and plan['source_sha256']==sources()
         and sha(path)==path.with_suffix('.sha256').read_text().strip(),'Readout frozen plan/source differs')
    for n,h in plan['source_sha256'].items():need(sha(path.parent/'source'/n.replace('/','_'))==h,'Readout source snapshot changed')
    for file,h in plan['runtime_bindings'].items():need(sha(file)==h,'Readout consumed input changed')
    if ancestors:
        for file,h in plan['input_bindings'].items():need(sha(file)==h,'Readout ancestor binding changed')
    need(object_sha(read(plan['order_file']))==plan['order_object_sha256']
         and object_sha(plan['native_identity'])==plan['native_identity_sha256'],'Readout order/native identity differs')
    proof=read(path.parent/'summary.json')
    need(proof['passed'] is True and proof['completed'] is True and proof['phase']=='check'
         and proof['plan_sha256']==sha(path) and proof['source_sha256']==sources(),'Passed CPU source gate required')
    return plan


def one_run_guard(out):
    command=['sacct','-X','-u',os.environ['USER'],'--starttime=2026-09-01','--parsable2','--noheader',
             '--format=JobIDRaw,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,Start,End']
    result=subprocess.run(command,capture_output=True,text=True,check=True);path=out/'launch_sacct.psv';path.write_text(result.stdout)
    save(out/'launch_accounting.json',dict(command=command,file=str(path),sha256=sha(path)))
    job=os.environ['SLURM_JOB_ID']
    for line in result.stdout.splitlines():
        f=line.split('|');need(len(f)==9,'Unexpected scheduler launch row')
        if f[1]==RUN_JOB and f[2]=='gpu':need(f[0]==job,'This protocol permits a single GPU attempt; no automatic retry')


def run(args,out,frozen,started):
    import torch
    torch.set_num_threads(4);args.plan=args.plan.resolve();plan=verify_plan(args.plan)
    one_run_guard(out)
    need(torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA B200',
         'One B200 required for the fixed head-only job')
    hardware=dict(gpu=torch.cuda.get_device_name(0),total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
        torch_version=str(torch.__version__),matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
    need(not torch.backends.cuda.matmul.allow_tf32,'Oracle FP32 projection requires TF32 already disabled')
    torch.manual_seed(24);torch.cuda.manual_seed_all(24);device=torch.device('cuda')
    data=DATA/out.name;ckpt=CKPT/out.name;data.mkdir(parents=True,exist_ok=False);ckpt.mkdir(parents=True,exist_ok=False)
    rows=read(plan['rows_file']);order=read(plan['order_file']);by_sid={r['sid']:r for r in rows}
    save(out/'input_inventory.json',read(plan['input_inventory_file']));save(out/'presentations.json',order)
    initial=torch.load(plan['initial_file'],map_location='cpu',weights_only=True)
    need(initial['step']==0 and initial['policy']==POLICY and initial['source_sha256']==frozen
         and base.v7.tensor_info(initial['U'])==plan['initial_U'] and bool((initial['U']==0).all()),'Exact zero initialization differs')
    U=torch.nn.Parameter(initial['U'].to(device));initial_info=base.v7.tensor_info(U)
    inputs=torch.load(plan['inputs_file'],map_location='cpu',weights_only=True)
    need(inputs['schema_version']==1 and inputs['global_feature_ids']==plan['global_feature_ids']
         and {k:base.v7.tensor_info(inputs[k]) for k in ('global_states','codebook')}==plan['input_tensors']
         and torch.equal(inputs['codebook'],codebook(torch)),'Frozen global/code inputs differ')
    globals_=inputs['global_states'].to(device);codes=inputs['codebook'].to(device)
    module_identity=native_module_identity(torch,plan['native_identity'])
    need(module_identity==plan['native_module_identity'],'Native module implementation changed after CPU gate')
    packet=torch.load(plan['native_model_file'],map_location='cpu',weights_only=True)
    norm,head=native_modules(torch,packet,plan['native_identity'],device);del packet
    native_initial=dict(norm=base.v7.tensor_info(norm.weight),head=base.v7.tensor_info(head.weight))
    versions=dict(norm=norm.weight._version,head=head.weight._version)
    counters=dict(norm=0,head=0,vlm=0,vision=0);calls=[];normal_outputs=[];current_phase='training';current_step=0
    def observer(name):
        def hook(module,arguments,output):
            counters[name]+=1
            if name=='norm' and current_phase=='evaluation':normal_outputs.append(output.detach().cpu().clone())
            calls.append(dict(module=name,phase=current_phase,step=current_step,input_shape=list(arguments[0].shape),
                output_shape=list(output.shape),input_dtype=str(arguments[0].dtype),output_dtype=str(output.dtype)))
        return hook
    handles=[norm.register_forward_hook(observer('norm')),head.register_forward_hook(observer('head'))]
    config=dict(protocol=PROTOCOL,run_id=out.name,slurm_job_id=os.environ['SLURM_JOB_ID'],seed=24,policy=POLICY,
        source_sha256=frozen,plan_file=str(args.plan),plan_sha256=sha(args.plan),conditional_report=plan['conditional_report'],
        uniform_plan=plan['uniform_plan'],hardware=hardware,native_identity=plan['native_identity'],
        native_identity_sha256=plan['native_identity_sha256'],native_module_identity=module_identity,
        native_model_file=plan['native_model_file'],native_model_sha256=plan['native_model_sha256'],native_weight_identity=native_initial,
        initialized_U=initial_info,initial_checkpoint=plan['initial_file'],initial_checkpoint_sha256=plan['initial_sha256'],
        input_inventory_file=str(out/'input_inventory.json'),input_inventory_sha256=sha(out/'input_inventory.json'),
        presentations_file=str(out/'presentations.json'),presentations_sha256=sha(out/'presentations.json'),
        order_sha256=object_sha(order),data_directory=str(data),checkpoint_directory=str(ckpt))
    save(out/'config.json',config)
    optimizer=torch.optim.AdamW([U],lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0.)
    torch.cuda.synchronize();setup=time.perf_counter()-started;log=[];io_seconds=0.;raw_batches=[]
    try:
        for step in range(1,601):
            current_step=step;tick=time.perf_counter();batch=order[(step-1)*8:step*8]
            sids=[sid for pair in batch for sid in pair['sids']];selected=[by_sid[sid] for sid in sids]
            need(len(selected)==16 and all([by_sid[sid]['n_frames'] for sid in pair['sids']]==[8,16] for pair in batch),
                 'Intact pair batch differs')
            ci=torch.tensor([r['class_index'] for r in selected],device=device);gi=torch.tensor([r['global_row'] for r in selected],device=device)
            targets=torch.tensor([r['first_token_id'] for r in selected],device=device);lengths=torch.tensor([r['full_target_length'] for r in selected],device=device)
            optimizer.zero_grad(set_to_none=True);rate=learning_rate(step)
            optimizer.param_groups[0]['lr']=rate
            logits,cap=readout(torch,U,globals_[gi],codes[ci],norm,head)
            if step==1:need(torch.equal(cap['fused_global'],globals_[gi]) and bool((cap['delta']==0).all()),'Initial native identity failed')
            loss,nll,parts=weighted_loss(torch,logits,targets,lengths);need(bool(torch.isfinite(loss)),'Nonfinite oracle CE')
            loss.backward();need(U.grad is not None and bool(torch.isfinite(U.grad).all()) and bool((U.grad[:,9:]==0).all()),
                 'Finite active-only U gradient invariant failed')
            gradnorm=torch.nn.utils.clip_grad_norm_([U],1.);optimizer.step()
            need(bool(torch.isfinite(U).all()) and bool((U[:,9:]==0).all()) and norm.weight.grad is head.weight.grad is None
                 and not norm.weight.requires_grad and not head.weight.requires_grad
                 and versions==dict(norm=norm.weight._version,head=head.weight._version)
                 and not globals_.requires_grad and not codes.requires_grad,'Inactive/native/input tensor changed')
            torch.cuda.synchronize()
            log.append(dict(step=step,sids=sids,pair_ids=[p['pair_id'] for p in batch],cycles=[p['cycle'] for p in batch],
                first_token_ids=targets.tolist(),full_target_lengths=lengths.tolist(),first_token_nll=nll.detach().tolist(),
                weighted_contributions=parts.detach().tolist(),loss=float(loss),unweighted_ce=float(nll.mean()),lr=rate,
                gradient_norm=float(gradnorm),clipped=float(gradnorm)>1.,nonactive_columns_zero=True,native_frozen=True,
                seconds=time.perf_counter()-tick))
            if step%100==0:
                tick=time.perf_counter();save(out/f'progress_{step:04d}.json',dict(step=step,training=log,counters=counters))
                io_seconds+=time.perf_counter()-tick;print(json.dumps(dict(step=step,loss=float(loss))),flush=True)
        save(out/'training.json',log);optimizer.zero_grad(set_to_none=True);del optimizer,loss,nll,parts,logits,cap
        U.requires_grad_(False);final_info=base.v7.tensor_info(U);checkpoint=ckpt/'final.pt'
        torch.save(dict(schema_version=1,U=U.detach().cpu().clone(),step=600,config=config),checkpoint)
        with torch.no_grad():U.zero_()
        need(base.v7.tensor_info(U)==initial_info,'Exact initial reset failed')
        restored=torch.load(checkpoint,map_location='cpu',weights_only=True)
        need(restored['schema_version']==1 and restored['step']==600 and restored['config']==config
             and base.v7.tensor_info(restored['U'])==final_info,'Restricted final checkpoint metadata/bytes differ')
        with torch.no_grad():U.copy_(restored['U'].to(device))
        need(base.v7.tensor_info(U)==final_info,'Final U reload differs')
        save(out/'checkpoint_roundtrip.json',dict(passed=True,initial_U=initial_info,final_U=final_info,reloaded_U=base.v7.tensor_info(U),
            checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),config_sha256=sha(out/'config.json'),step=600))
        predictions=[];final_deltas=[];final_fused=[];current_phase='evaluation'
        with torch.no_grad():
            for offset in range(0,108,16):
                current_step=offset//16+1;selected=rows[offset:offset+16]
                ci=torch.tensor([r['class_index'] for r in selected],device=device);gi=torch.tensor([r['global_row'] for r in selected],device=device)
                logits,capture=readout(torch,U,globals_[gi],codes[ci],norm,head)
                final_deltas.append(capture['delta'].detach().cpu().clone())
                final_fused.append(capture['fused_global'].detach().cpu().clone())
                ids=logits.argmax(-1);targets=torch.tensor([r['first_token_id'] for r in selected],device=device)
                nll=torch.nn.functional.cross_entropy(logits.float(),targets,reduction='none')
                raw=dict(schema_version=1,sids=[r['sid'] for r in selected],logits=logits.detach().cpu().clone())
                path=data/f'final_batch_{current_step:02d}.pt';torch.save(raw,path)
                raw_batches.append(dict(file=str(path),sha256=sha(path),sids=raw['sids'],logits=base.v7.tensor_info(raw['logits'])))
                predictions += [dict(r,argmax_id=int(ids[i]),first_token_correct=int(ids[i])==r['first_token_id'],nll=float(nll[i])) for i,r in enumerate(selected)]
        saved=[torch.load(item['file'],map_location='cpu',weights_only=True) for item in raw_batches]
        final_raw=dict(schema_version=1,sids=[r['sid'] for r in rows],logits=torch.cat([r['logits'] for r in saved]),batches=raw_batches)
        raw_file=data/'final_logits.pt';torch.save(final_raw,raw_file)
        final_inputs=dict(schema_version=1,sids=[r['sid'] for r in rows],delta=torch.cat(final_deltas),
            fused_global=torch.cat(final_fused),normalized=torch.cat(normal_outputs))
        final_inputs_file=data/'final_inputs.pt';torch.save(final_inputs,final_inputs_file)
        final_input_tensors={k:base.v7.tensor_info(final_inputs[k]) for k in ('delta','fused_global','normalized')}
        save(out/'predictions.json',predictions);save(out/'calls.json',calls)
        native_final=dict(norm=base.v7.tensor_info(norm.weight),head=base.v7.tensor_info(head.weight))
        need(native_final==native_initial and base.v7.tensor_info(U)==final_info
             and versions==dict(norm=norm.weight._version,head=head.weight._version)
             and counters==dict(norm=607,head=607,vlm=0,vision=0),'Final endpoint/native/call inventory changed')
        endpoint=dict(passed=True,U_before_evaluation=final_info,U_after_evaluation=base.v7.tensor_info(U),
            native_before=native_initial,native_after=native_final,native_module_identity=module_identity,
            inactive_columns_exactly_zero=bool((U[:,9:]==0).all()),native_weights_unchanged=True,counters=counters,
            checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),raw_logits=base.v7.tensor_info(final_raw['logits']))
        save(out/'final_endpoint.json',endpoint)
        result=dict(**config,passed=True,completed=True,phase='run',computational_integrity_passed=True,steps=600,
            checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),selected_U=final_info,checkpoint_roundtrip_passed=True,
            training_file=str(out/'training.json'),training_sha256=sha(out/'training.json'),
            predictions_file=str(out/'predictions.json'),predictions_sha256=sha(out/'predictions.json'),
            raw_file=str(raw_file),raw_sha256=sha(raw_file),raw_tensor=base.v7.tensor_info(final_raw['logits']),raw_batches=raw_batches,
            final_inputs_file=str(final_inputs_file),final_inputs_sha256=sha(final_inputs_file),final_input_tensors=final_input_tensors,
            calls_file=str(out/'calls.json'),calls_sha256=sha(out/'calls.json'),counters=counters,
            endpoint_file=str(out/'final_endpoint.json'),endpoint_sha256=sha(out/'final_endpoint.json'),
            roundtrip_file=str(out/'checkpoint_roundtrip.json'),roundtrip_sha256=sha(out/'checkpoint_roundtrip.json'),
            first_token_fit=criteria(predictions),training_contexts=108,scene_presentations=9600,
            setup_before_training_seconds=setup,training_seconds=sum(r['seconds'] for r in log),training_log_io_seconds=io_seconds,
            native_weights_unchanged=True,no_pretrained_backbone_loaded=True,no_dev_or_test=True,no_eos_or_whole_answer_claim=True)
        need(time.perf_counter()-started<=150,'Head-only software elapsed cap exceeded')
        return result
    finally:
        for handle in handles:handle.remove()
        save(out/'final_counters.json',dict(counters=counters,recorded_calls=calls,partial_outputs_retained=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    for action in ('check','run','report'):group.add_argument('--'+action,action='store_true')
    parser.add_argument('--plan',type=Path);parser.add_argument('--runs',type=Path,nargs='+');args=parser.parse_args()
    native.require_slurm(gpu=args.run)
    if not args.run:need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),'CPU-only check/report required')
    if args.run or args.report:need(args.plan is not None,'Frozen CPU plan required')
    if args.report:need(args.runs and len(args.runs)==1,'Exactly one oracle run required')
    phase='check' if args.check else 'run' if args.run else 'report';out=OUT/f'{phase}_{os.environ["SLURM_JOB_ID"]}'
    out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();frozen=snapshot(out)
    save(out/'request.json',dict(phase=phase,plan=None if args.plan is None else str(args.plan),
        runs=None if args.runs is None else list(map(str,args.runs)),source_sha256=frozen))
    try:
        if args.check:result=check(out,frozen)
        elif args.run:result=run(args,out,frozen,started)
        else:
            from scripts import report_native_identity_join_readout as reporter
            result=reporter.report(args,out,frozen)
        need(sources()==frozen,'Source changed during diagnostic execution')
        result['elapsed_seconds']=time.perf_counter()-started;save(out/'summary.json',result)
        print(json.dumps(dict(passed=result['passed'],completed=True,directory=str(out))),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,
            elapsed_seconds=time.perf_counter()-started,partial_outputs_retained=True));raise


if __name__=='__main__':main()

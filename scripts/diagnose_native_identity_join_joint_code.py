"""Held local joint-code oracle: frozen training inputs, ordinary native answer CE.

Per-frame person/requested-room codes replace the learned local encoder. No bag
intersection or answer code is supplied. This is an offline diagnostic only.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import diagnose_native_identity_join_factor_binding as original
from scripts.stage_native_vision_v6_teacher import need,read,sha,save,object_sha
uniform=original.uniform;oracle=original.oracle;base=original.base;native=original.native
tensor_info=original.tensor_info;bind=original.bind;cpu_tree=original.cpu_tree;learning_rate=original.learning_rate
OUT=REPO/'outputs/native_aggregation_vlm/identity_join_joint_code'
DATA=Path('/mnt/data/gabriele/gnn_transformer/identity_join_joint_code')
CKPT=Path('/mnt/ckpts/gabriele/gnn_transformer/identity_join_joint_code')
PROTOCOL='identity_join_local_joint_code_oracle'
ARM='joint_code';ARMS=(ARM,);RUN_JOBS={ARM:'identity_join_joint_code_run'}
PEOPLE=oracle.PEOPLE
ROOMS=('Kitchen','Bathroom','Garden','Office','Bedroom','Park')
READOUT_KEYS=('query.weight','aggregate_projection.weight','aggregate_projection.bias','up.weight')
PARENT=original.OUT/'check_443373/plan.json'
PARENT_SHA='aa9a44067f86ce6340d56419b5efe9411ccd84cb52115f0b31d10745b4e68a07'
CONDITIONAL=REPO/'outputs/native_aggregation_vlm/identity_join_readout_query/report_443418/summary.json'
CONDITIONAL_SHA='fea7ecb1a2f71a943aa126e1f395163a91e2f7ca8bc189c8a2d4ebb1374733a2'
PROPOSAL='docs/paper/NATIVE_AGGREGATION_JOINT_CODE_ORACLE_PROPOSAL.md'
PROPOSAL_SHA='1db9fbd3928b0c076643347d0ac05a65f361bb5765f62192054dd1f33bc7811d'
DIAGNOSTIC_STEPS=(1,2,32,128,300,600)
POLICY=dict(protocol=PROTOCOL,arm=ARM,seed=24,steps=600,batch_pairs=8,batch_scenes=16,
    training_contexts=108,pairs=54,families=18,hidden_size=3584,rank=96,people=list(PEOPLE),
    local_code='raw onehot(person_index*2+requested_room_index) in first18 of96; outside requested rooms allzero',
    local_code_dtype='torch.float32',message_rule='.5*valid*raw_code; ordinary SUM',
    readout_rule='q=Wq*RMS(global); delta=U*SiLU(Wagg*SUM(messages)+bagg+q)',
    retained_state_parameters=697440,effective_trainable_parameters=697440,readout_keys=list(READOUT_KEYS),
    initialization='exact four selected tensors from original unfitted factor443373 checkpoint',
    lr=.001,warmup=50,final_lr=.00001,weight_decay=0.,clip_norm=1.,optimizer='AdamW',
    loss='original mean-scene mean-token full-name-plus-EOS native CE only',consistency_coefficient=0.,
    epsilon=1e-6,projection_tf32=False,native_dtype='torch.float16',core_dtype='torch.float32',
    training_head_rows=21334,final_head_rows=108,total_head_rows=21442,core_calls=607,norm_calls=607,head_calls=607,
    cpu_final_head_calls=7,cpu_final_head_rows=108,training_capture_steps=list(DIAGNOSTIC_STEPS),
    first_token_screen=dict(correct=103,contexts=108,complete_families=16,families=18),
    gpu_seconds_cap=90,campaign_gpu_seconds_cap=90,maximum_gpus=1,final_checkpoint_only=True,
    oracle_diagnostic=True,no_answer_or_intersection_code=True,no_runtime_method_claim=True,
    no_local_encoder=True,no_conditioning_statistics=True,no_dev_or_test=True,no_vlm_or_vision_calls=True)
OWN=('gnnformer/parallel_local_joint_code_oracle.py','scripts/diagnose_native_identity_join_joint_code.py',
     'scripts/report_native_identity_join_joint_code.py','slurm/native_identity_join_joint_code_check.sbatch',
     'slurm/native_identity_join_joint_code_run.sbatch','slurm/native_identity_join_joint_code_report.sbatch',PROPOSAL)


def inherited_sources():
    need(sha(PARENT)==PARENT_SHA,'Frozen parent plan changed');parent=read(PARENT)
    for name,digest in parent['source_sha256'].items():
        need(sha(REPO/name)==digest and sha(PARENT.parent/'source'/name.replace('/','_'))==digest,'Inherited helper source changed')
    return parent['source_sha256']


def sources():return {name:sha(REPO/name) for name in OWN}


def snapshot(out):
    frozen=sources();(out/'source').mkdir()
    for name,digest in frozen.items():
        target=out/'source'/name.replace('/','_');target.write_bytes((REPO/name).read_bytes());need(sha(target)==digest,'Source copy changed')
    save(out/'source_hashes.json',frozen)
    save(out/'inherited_sources.json',dict(plan_file=str(PARENT),plan_sha256=PARENT_SHA,source_sha256=inherited_sources()))
    return frozen


def code_index(person,room,requested_pair):
    """One frame only: no sample, gold, bag, target token, or prefix argument."""
    need(person in PEOPLE and room in ROOMS and len(requested_pair)==2
         and len(set(requested_pair))==2 and all(r in ROOMS for r in requested_pair),'Unknown local semantic input')
    return None if room not in requested_pair else 2*PEOPLE.index(person)+requested_pair.index(room)


def code_indices(sample,states):
    """Pure local mapping; deliberately never inspect sample['gold'] or other frames."""
    pair=sample['room_pair']
    need(len(pair)==2 and len(set(pair))==2 and all(room in ROOMS for room in pair)
         and sample['question']==base.data_stage.global_prompt(pair),'Requested room order/question differs')
    result=[]
    for i,state in enumerate(states):
        need(set(state)=={'step_id','rooms'} and type(state['step_id']) is int and state['step_id']==i+1
             and isinstance(state['rooms'],dict) and len(state['rooms'])==1,'Canonical local state differs')
        room,persons=next(iter(state['rooms'].items()))
        need(room in ROOMS and isinstance(persons,list) and len(persons)==1 and persons[0] in PEOPLE,'Unknown or ambiguous local atom')
        result.append(code_index(persons[0],room,pair))
    return result


def semantic_inventory(samples,rows,scenes,bindings):
    need(len(samples)==len(rows)==108 and [s['sid'] for s in samples]==[r['sid'] for r in rows],'Training sample ownership differs')
    inventory=[];offset=0
    for sample,row in zip(samples,rows):
        sid=row['sid'];scene=scenes[sid];n=row['n_frames'];directory=Path(sample['path'])
        need(sample['split']==scene['split']=='train' and n==sample['n_frames']==scene['n_frames'] and n in (8,16)
             and all(sample[k]==scene[k]==row[k] for k in ('sid','question','gold','target_ids','pair_id'))
             and sample['path']==scene['path'] and len(sample['image_files'])==n,'Original training sample/cache join differs')
        qa=directory/'qa.txt';digest=bind(qa,bindings,sample['qa_sha256']);need(digest==scene['qa_sha256'],'QA ownership differs')
        lines=qa.read_text().splitlines()
        need(len(lines)==n+4 and lines[0]=='question:' and lines[-3]==sample['question'] and lines[-2]=='answer:'
             and lines[-1]==sample['gold'],'QA question/answer structure differs')
        states=[ast.literal_eval(line) for line in lines[1:-3]]
        content=object_sha(dict(states=states,question=sample['question']))
        need(content==sample['content_sha256']==scene['content_sha256'] and sid=='ij_'+content[:24],'Canonical state/content identity differs')
        indices=code_indices(sample,states);frames=[]
        for i,(state,image,code_index) in enumerate(zip(states,sample['image_files'],indices)):
            room,people=next(iter(state['rooms'].items()));path=directory/f'{i:03d}.png'
            need(Path(image['path'])==path and image['dimensions']==[512,512] and image['mode']=='RGB','Image occurrence order differs')
            bind(path,bindings,image['sha256'])
            frames.append(dict(local_index=i,step_id=i+1,person=people[0],room=room,image_file=str(path),image_sha256=image['sha256'],code_index=code_index))
        inventory.append(dict(sid=sid,question=sample['question'],room_pair=sample['room_pair'],n_frames=n,
            qa_file=str(qa),qa_sha256=digest,content_sha256=content,code_start=offset,code_stop=offset+n,frames=frames));offset+=n
    need(offset==1296,'Exactly1296 local training occurrences required')
    return dict(schema_version=1,people=list(PEOPLE),rank=96,active_coordinates=18,scenes=inventory,
        local_mapping_only=True,no_answer_or_intersection_code=True,no_target_dependent_code=True)


def make_core(torch):
    from gnnformer.parallel_local_joint_code_oracle import ParallelLocalJointCodeOracle
    return ParallelLocalJointCodeOracle()


def trainability(core):
    need(tuple(core.state_dict())==READOUT_KEYS and all(p.requires_grad and p.grad is None for p in core.parameters()),'Only four live readout parameters required')
    result=dict(trainable_parameter_names=list(READOUT_KEYS),effective_trainable_parameters=sum(p.numel() for p in core.parameters()),retained_state_parameters=sum(p.numel() for p in core.parameters()))
    need(result['effective_trainable_parameters']==697440,'Readout parameter count differs');return result


def batch_states(torch,scenes,states,index,codes,code_offsets,sids):
    from gnnformer.paired_sequence_objectives import sequence_layout
    targets=[scenes[sid]['target_ids'] for sid in sids];layout=sequence_layout(targets)
    width=layout['offsets'][-1];max_n=max(scenes[sid]['n_frames'] for sid in sids)
    local=torch.zeros((max_n,width,96),device=codes.device,dtype=torch.float32)
    valid=torch.zeros((max_n,width),device=codes.device,dtype=torch.bool);globals_=[]
    for sid,left,right in zip(sids,layout['offsets'],layout['offsets'][1:]):
        scene=scenes[sid];start,stop=code_offsets[sid];n=scene['n_frames']
        need(stop-start==n and len(scene['global_feature_ids'])==right-left,'Local code/global prefix ownership differs')
        local[:n,left:right]=codes[start:stop,None,:].expand(n,right-left,96);valid[:n,left:right]=True
        globals_.append(states[torch.tensor([index[fid] for fid in scene['global_feature_ids']],device=states.device)])
    g=torch.cat(globals_,dim=0)
    need(torch.equal(g[layout['left']],g[layout['right']]),'Paired native globals differ')
    return local,g,layout,valid


def self_test(torch):
    from gnnformer.parallel_local_joint_code_oracle import self_test as core_tests
    sample=dict(room_pair=['Kitchen','Office'],question=base.data_stage.global_prompt(['Kitchen','Office']),gold='Sandra')
    states=[dict(step_id=1,rooms={'Kitchen':['Mary']}),dict(step_id=2,rooms={'Garden':['Sandra']})]
    first=code_indices(sample,states);need(first==[2,None] and code_indices(dict(sample,gold='Noah'),states)==first,'Gold changed local oracle codes')
    changed=[dict(step_id=1,rooms={'Office':['Mary']}),states[1]]
    need(code_indices(sample,changed)==[3,None],'Semantic frame change must change the local code')
    scenes={sid:dict(n_frames=n,target_ids=[3,4,151645],global_feature_ids=['a','b','c']) for sid,n in [('a',1),('b',2)]}
    codes=torch.zeros(3,96);codes[0,2]=codes[1,2]=1;globals_=torch.arange(6,dtype=torch.float16).reshape(3,2)
    local,g,layout,valid=batch_states(torch,scenes,globals_,{'a':0,'b':1,'c':2},codes,{'a':(0,1),'b':(1,3)},['a','b'])
    need(all(torch.equal(local[:scenes[sid]['n_frames'],a:b],codes[start:stop,None].expand(stop-start,b-a,96))
         for sid,(start,stop),a,b in zip(['a','b'],[(0,1),(1,3)],layout['offsets'],layout['offsets'][1:])), 'Codes depend on teacher-forced prefix')
    need(bool((local[~valid]==0).all()),'Code padding differs')
    return dict(passed=True,gold_independent_mapping=True,semantic_frame_sensitivity=True,all_prefix_reuse=True,padding=True,core=core_tests(torch))


def check(out,frozen):
    import torch
    torch.set_num_threads(4);bindings={};bind(PARENT,bindings,PARENT_SHA);parent=read(PARENT)
    proof=read(PARENT.parent/'summary.json')
    need(proof['passed'] is True and proof['completed'] is True and proof['plan_sha256']==PARENT_SHA
         and parent['source_sha256']==inherited_sources(),'Passed frozen parent preparation required')
    bind(CONDITIONAL,bindings,CONDITIONAL_SHA);closed=read(CONDITIONAL)
    bind(closed['analysis_file'],bindings,closed['analysis_sha256'])
    need(closed['passed'] is True and closed['completed'] is True
         and closed['first_token_screen']['product']['passed'] is False,'The previous architecture branch must remain closed')
    bind(REPO/PROPOSAL,bindings,PROPOSAL_SHA)
    for key in ('rows_file','scenes_file','pairs_file','order_file','features_file','initial_file','native_model_file'):
        bind(parent[key],bindings,parent['runtime_bindings'][str(Path(parent[key]).resolve())])
    rows=read(parent['rows_file']);parent_scenes=read(parent['scenes_file']);pairs=read(parent['pairs_file']);order=read(parent['order_file'])
    need(len(rows)==len(parent_scenes)==108 and len(pairs)==54 and order==uniform.order_pairs(pairs)
         and Counter(r['gold'] for r in rows)==Counter({name:12 for name in PEOPLE}),'Exact original training/order inventory differs')
    head_counts=parent['head_rows_by_update'];need(len(head_counts)==600 and sum(head_counts)==21334 and max(head_counts)==44,'Full-CE row inventory differs')
    bind(parent['uniform_plan']['file'],bindings,parent['uniform_plan']['sha256']);up=read(parent['uniform_plan']['file'])
    bind(up['rows_file'],bindings,up['runtime_bindings'][str(Path(up['rows_file']).resolve())])
    samples=[r['sample'] for r in read(up['rows_file'])['train']]
    inventory=semantic_inventory(samples,rows,parent_scenes,bindings)
    data=DATA/out.name;ckpt=CKPT/out.name;data.mkdir(parents=True,exist_ok=False);ckpt.mkdir(parents=True,exist_ok=False)
    save(out/'code_inventory.json',inventory);save(out/'training_samples.json',samples)
    offsets=[0]+[scene['code_stop'] for scene in inventory['scenes']];codes=torch.zeros(1296,96,dtype=torch.float32)
    for scene in inventory['scenes']:
        for frame in scene['frames']:
            if frame['code_index'] is not None:codes[scene['code_start']+frame['local_index'],frame['code_index']]=1.
    codes_file=data/'codes.pt';torch.save(dict(schema_version=1,sids=[r['sid'] for r in rows],scene_offsets=offsets,codes=codes),codes_file)
    packet=torch.load(parent['features_file'],map_location='cpu',weights_only=True)
    need(tensor_info(packet['states'])==parent['feature_tensor'],'Original feature packet changed')
    feature_ids=sorted({fid for scene in parent_scenes.values() for fid in scene['global_feature_ids']});index={fid:i for i,fid in enumerate(packet['feature_ids'])}
    states=packet['states'][torch.tensor([index[fid] for fid in feature_ids])].clone();del packet
    need(states.dtype==torch.float16 and states.shape==(len(feature_ids),3584),'Native global-only feature inventory differs')
    features_file=data/'global_features.pt';torch.save(dict(states=states,feature_ids=feature_ids),features_file)
    scenes={sid:{k:v for k,v in scene.items() if k not in ('local_feature_ids','image_question_pair_ids')} for sid,scene in parent_scenes.items()}
    for name,value in (('rows',rows),('scenes',scenes),('pairs',pairs),('order',order)):save(out/(name+'.json'),value)
    packet=torch.load(parent['initial_file'],map_location='cpu',weights_only=True)
    need(packet['seed']==24 and set(packet)=={'seed','branch'},'Unfitted parent metadata differs')
    selected={key:packet['branch'][key].detach().clone() for key in READOUT_KEYS};del packet
    core=make_core(torch);core.load_state_dict(selected);mask=trainability(core);initial_state=base.v7.state_info(core)
    need(initial_state=={key:parent['initial_state'][key] for key in READOUT_KEYS} and bool((selected['up.weight']==0).all()),'Selected unfitted tensors differ')
    initial_file=ckpt/'initial.pt';torch.save(dict(seed=24,branch=selected,parent_initial_file=parent['initial_file'],parent_initial_sha256=parent['initial_sha256']),initial_file)
    tests=self_test(torch)
    from scripts.report_native_identity_join_joint_code import self_test as reporter_self_test
    tests['reporter']=reporter_self_test(torch)
    module_identity=oracle.native_module_identity(torch,parent['native_identity'])
    native_packet=torch.load(parent['native_model_file'],map_location='cpu',weights_only=True)
    norm,head=oracle.native_modules(torch,native_packet,parent['native_identity'],'cpu');del native_packet,norm,head
    runtime_bindings={}
    for file in (out/'rows.json',out/'scenes.json',out/'pairs.json',out/'order.json',out/'training_samples.json',out/'code_inventory.json',
                 codes_file,features_file,initial_file,Path(parent['native_model_file'])):bind(file,runtime_bindings)
    plan=dict(schema_version=1,protocol=PROTOCOL,policy=POLICY,source_sha256=frozen,inherited_source_sha256=inherited_sources(),tests=tests,
        input_bindings=bindings,runtime_bindings=runtime_bindings,factor_plan=dict(file=str(PARENT),sha256=PARENT_SHA),uniform_plan=parent['uniform_plan'],
        architecture_stop_report=dict(file=str(CONDITIONAL),sha256=CONDITIONAL_SHA),trainability_mask=mask,
        rows_file=str(out/'rows.json'),scenes_file=str(out/'scenes.json'),pairs_file=str(out/'pairs.json'),order_file=str(out/'order.json'),
        training_samples_file=str(out/'training_samples.json'),order_object_sha256=object_sha(order),head_rows_by_update=head_counts,
        code_inventory_file=str(out/'code_inventory.json'),code_inventory_sha256=sha(out/'code_inventory.json'),codes_file=str(codes_file),codes_sha256=sha(codes_file),code_tensor=tensor_info(codes),
        features_file=str(features_file),features_sha256=sha(features_file),feature_tensor=tensor_info(states),global_feature_ids=feature_ids,
        parent_features_file=parent['features_file'],parent_features_sha256=parent['features_sha256'],parent_feature_tensor=parent['feature_tensor'],
        initial_file=str(initial_file),initial_sha256=sha(initial_file),initial_state=initial_state,
        parent_initial_file=parent['initial_file'],parent_initial_sha256=parent['initial_sha256'],
        native_model_file=parent['native_model_file'],native_model_sha256=parent['native_model_sha256'],native_identity=parent['native_identity'],
        native_identity_sha256=parent['native_identity_sha256'],native_module_identity=module_identity,
        reused_unfitted_readout=True,oracle_local_semantics=True,no_answer_or_intersection_code=True,no_pretrained_backbone_loaded=True,no_pretrained_head_forward=True)
    save(out/'plan.json',plan);(out/'plan.sha256').write_text(sha(out/'plan.json')+'\n')
    return dict(passed=True,completed=True,phase='check',protocol=PROTOCOL,source_sha256=frozen,inherited_source_sha256=plan['inherited_source_sha256'],
        plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),tests=tests,training_contexts=108,image_occurrences=1296,
        global_feature_count=len(feature_ids),training_head_rows=21334,final_head_rows=108,oracle_local_semantics=True,no_pretrained_head_forward=True)


def verify_plan(path,ancestors=False):
    path=Path(path).resolve();plan=read(path)
    need(plan['protocol']==PROTOCOL and plan['policy']==POLICY and plan['source_sha256']==sources()
         and plan['inherited_source_sha256']==inherited_sources() and sha(path)==path.with_suffix('.sha256').read_text().strip(),'Joint-code plan/source changed')
    for name,digest in plan['source_sha256'].items():need(sha(path.parent/'source'/name.replace('/','_'))==digest,'New source snapshot changed')
    need(read(path.parent/'inherited_sources.json')==dict(plan_file=str(PARENT),plan_sha256=PARENT_SHA,source_sha256=plan['inherited_source_sha256']),'Inherited source descriptor changed')
    for file,digest in plan['runtime_bindings'].items():need(sha(file)==digest,'Consumed input changed')
    if ancestors:
        for file,digest in plan['input_bindings'].items():need(sha(file)==digest,'Bound training ancestor changed')
    proof=read(path.parent/'summary.json')
    need(proof['passed'] is True and proof['completed'] is True and proof['phase']=='check' and proof['plan_sha256']==sha(path)
         and object_sha(read(plan['order_file']))==plan['order_object_sha256']
         and object_sha(plan['native_identity'])==plan['native_identity_sha256'],'Passed CPU gate/native identity required')
    return plan


def single_run_guard(out):
    need(os.environ.get('SLURM_JOB_NAME')==RUN_JOBS[ARM],'Exact registered single-run job name required')
    command=['sacct','-X','-u',os.environ['USER'],'--starttime=2026-09-01','--parsable2','--noheader',
             '--format=JobIDRaw,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,Start,End']
    raw=subprocess.run(command,capture_output=True,text=True,check=True).stdout;file=out/'launch_sacct.psv';file.write_text(raw)
    save(out/'launch_accounting.json',dict(command=command,file=str(file),sha256=sha(file)))
    for line in raw.splitlines():
        fields=line.split('|');need(len(fields)==9,'Unexpected allocation schema')
        if fields[1]==RUN_JOBS[ARM] and fields[2]=='gpu':need(fields[0]==os.environ['SLURM_JOB_ID'],'One joint-code GPU attempt only')


def run(args,out,frozen,started):
    import torch
    torch.set_num_threads(4);args.plan=args.plan.resolve();plan=verify_plan(args.plan);single_run_guard(out)
    need(torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA B200'
         and not torch.backends.cuda.matmul.allow_tf32,'One B200 with TF32 disabled required')
    hardware=dict(gpu=torch.cuda.get_device_name(0),torch_version=str(torch.__version__),matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    torch.manual_seed(24);torch.cuda.manual_seed_all(24);device=torch.device('cuda')
    data=DATA/out.name;ckpt=CKPT/out.name;data.mkdir(parents=True,exist_ok=False);ckpt.mkdir(parents=True,exist_ok=False)
    rows=read(plan['rows_file']);scenes=read(plan['scenes_file']);order=read(plan['order_file'])
    save(out/'rows.json',rows);save(out/'presentations.json',order);save(out/'code_inventory.json',read(plan['code_inventory_file']))
    packet=torch.load(plan['features_file'],map_location='cpu',weights_only=True)
    need(tensor_info(packet['states'])==plan['feature_tensor'] and packet['feature_ids']==plan['global_feature_ids'],'Frozen global features changed')
    states=packet['states'].to(device);index={fid:i for i,fid in enumerate(packet['feature_ids'])};del packet
    packet=torch.load(plan['codes_file'],map_location='cpu',weights_only=True)
    inventory=read(plan['code_inventory_file']);offsets=[0]+[s['code_stop'] for s in inventory['scenes']]
    need(set(packet)=={'schema_version','sids','scene_offsets','codes'} and packet['schema_version']==1
         and packet['sids']==[r['sid'] for r in rows] and packet['scene_offsets']==offsets
         and tensor_info(packet['codes'])==plan['code_tensor'],'Fixed local-code packet changed')
    codes=packet['codes'].to(device);code_offsets={sid:(a,b) for sid,a,b in zip(packet['sids'],offsets,offsets[1:])};del packet
    input_versions=(codes._version,states._version);input_before=dict(codes=tensor_info(codes),globals=tensor_info(states))
    need(input_before==dict(codes=plan['code_tensor'],globals=plan['feature_tensor']),'Deployed code/global tensor identity differs')
    core=make_core(torch).to(device);initial=torch.load(plan['initial_file'],map_location='cpu',weights_only=True)
    core.load_state_dict(initial['branch']);mask=trainability(core)
    need(base.v7.state_info(core)==plan['initial_state'] and mask==plan['trainability_mask'],'Selected original unfitted readout differs')
    initial_file=ckpt/'initial.pt';shutil.copyfile(plan['initial_file'],initial_file)
    packet=torch.load(plan['native_model_file'],map_location='cpu',weights_only=True)
    norm,head=oracle.native_modules(torch,packet,plan['native_identity'],device);del packet
    module_identity=oracle.native_module_identity(torch,plan['native_identity']);need(module_identity==plan['native_module_identity'],'Native module source changed')
    native_initial=dict(norm=tensor_info(norm.weight),head=tensor_info(head.weight));versions=(norm.weight._version,head.weight._version)
    counters=dict(core=0,norm=0,head=0,vlm=0,vision=0);calls=[];captures=[];current_phase='training';current_step=0;capture_current=False;io={}
    def observer(name):
        def hook(module,arguments,output):
            counters[name]+=1
            calls.append(dict(module=name,phase=current_phase,step=current_step,input_shape=list(arguments[0].shape),
                output_shape=list(output.shape),input_dtype=str(arguments[0].dtype),output_dtype=str(output.dtype)))
            if capture_current:
                if name=='norm':io.update(fused_global=arguments[0][0].detach().cpu().clone(),normalized=output[0].detach().cpu().clone())
                else:io['logits']=output[0].detach().cpu().clone()
        return hook
    def core_observer(module,args,output):counters['core']+=1
    handles=[norm.register_forward_hook(observer('norm')),head.register_forward_hook(observer('head')),core.register_forward_hook(core_observer)]
    config=dict(protocol=PROTOCOL,run_id=out.name,slurm_job_id=os.environ['SLURM_JOB_ID'],arm=ARM,seed=24,policy=POLICY,
        source_sha256=frozen,inherited_source_sha256=plan['inherited_source_sha256'],plan_file=str(args.plan),plan_sha256=sha(args.plan),
        factor_plan=plan['factor_plan'],uniform_plan=plan['uniform_plan'],architecture_stop_report=plan['architecture_stop_report'],
        reused_unfitted_readout=True,oracle_local_semantics=True,no_answer_or_intersection_code=True,
        hardware=hardware,native_identity=plan['native_identity'],native_identity_sha256=plan['native_identity_sha256'],
        native_module_identity=module_identity,native_model_file=plan['native_model_file'],native_model_sha256=plan['native_model_sha256'],
        native_weight_identity=native_initial,initialized=base.v7.state_info(core),initial_checkpoint=str(initial_file),initial_checkpoint_sha256=sha(initial_file),
        parent_initial_file=plan['parent_initial_file'],parent_initial_sha256=plan['parent_initial_sha256'],
        features_file=plan['features_file'],features_sha256=plan['features_sha256'],feature_tensor=plan['feature_tensor'],
        codes_file=plan['codes_file'],codes_sha256=plan['codes_sha256'],code_tensor=plan['code_tensor'],
        rows_file=str(out/'rows.json'),rows_sha256=sha(out/'rows.json'),presentations_file=str(out/'presentations.json'),
        presentations_sha256=sha(out/'presentations.json'),order_sha256=object_sha(order),
        code_inventory_file=str(out/'code_inventory.json'),code_inventory_sha256=sha(out/'code_inventory.json'),
        data_directory=str(data),checkpoint_directory=str(ckpt),**mask)
    save(out/'config.json',config)
    parameters=list(core.parameters());optimizer=torch.optim.AdamW(parameters,lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0.)
    torch.cuda.synchronize();setup=time.perf_counter()-started;logs=[];raw_batches=[]
    def save_capture(local,g,valid,cap,layout,sids,weights_file,weights_step):
        blob=dict(schema_version=1,arm=ARM,phase=current_phase,step=current_step,sids=sids,layout=layout,
            codes=local,global_states=g,valid_mask=valid,capture=cap,**io)
        path=data/f'{current_phase}_capture_{current_step:04d}.pt';torch.save(cpu_tree(torch,blob),path)
        record=dict(arm=ARM,phase=current_phase,step=current_step,sids=sids,file=str(path),sha256=sha(path),
            weights_file=str(weights_file),weights_sha256=sha(weights_file),weights_step=weights_step,
            tensors={k:tensor_info(v) for k,v in io.items()})
        captures.append(record);return record
    try:
        for step in range(1,601):
            current_step=step;capture_current=step in DIAGNOSTIC_STEPS;io={};tick=time.perf_counter()
            batch=order[(step-1)*8:step*8];sids=[sid for pair in batch for sid in pair['sids']]
            h,g,layout,valid=batch_states(torch,scenes,states,index,codes,code_offsets,sids)
            need(g.shape[0]==plan['head_rows_by_update'][step-1]<=44,'Fixed full-target row budget changed')
            optimizer.zero_grad(set_to_none=True);rate=learning_rate(step);optimizer.param_groups[0]['lr']=rate
            weights_file=None
            if capture_current:
                weights_file=ckpt/f'training_core_{step:04d}.pt'
                torch.save(dict(arm=ARM,branch=cpu_tree(torch,core.state_dict()),step=step-1,optimizer_step=step,position='before_update',source_sha256=frozen),weights_file)
            _,ce,residual,parts,cap=base.losses(torch,core,h,g,layout,valid,norm,head)
            need(torch.equal(cap['payload'],h) and torch.equal(cap['gates'],valid.float()*.5) and bool((cap['scores']==.5).all()),'Raw-code half-SUM changed')
            if step==1:need(bool((cap['delta']==0).all()),'Initial U zero identity failed')
            if capture_current:save_capture(h,g,valid,cap,layout,sids,weights_file,step-1)
            ce.backward();need(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters),'Readout gradient missing/nonfinite')
            gradnorm=torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step()
            need(all(bool(torch.isfinite(p).all()) for p in core.parameters()) and norm.weight.grad is head.weight.grad is None
                 and versions==(norm.weight._version,head.weight._version) and input_versions==(codes._version,states._version),
                 'Readout/native/frozen-input invariant failed')
            torch.cuda.synchronize()
            logs.append(dict(step=step,sids=sids,pair_ids=[p['pair_id'] for p in batch],cycles=[p['cycle'] for p in batch],target_ids=layout['targets'],
                lr=rate,loss=float(ce),ce_loss=float(ce),consistency_loss=float(residual),weighted_consistency_loss=0.,consistency_coefficient=0.,
                gradient_norm=float(gradnorm),clipped=float(gradnorm)>1.,native_frozen=True,codes_frozen=True,
                code_input_shape=list(h.shape),seconds=time.perf_counter()-tick,**parts))
            if step%100==0:
                save(out/f'progress_{step:04d}.json',dict(step=step,training=logs,captures=captures,counters=counters))
                print(json.dumps(dict(step=step,ce=float(ce))),flush=True)
        save(out/'training.json',logs);optimizer.zero_grad(set_to_none=True);del optimizer,ce,residual,cap
        core.eval().requires_grad_(False);final=base.v7.state_info(core);checkpoint=ckpt/'final.pt'
        torch.save(dict(branch=cpu_tree(torch,core.state_dict()),step=600,config=config),checkpoint)
        core.load_state_dict(initial['branch']);need(base.v7.state_info(core)==plan['initial_state'],'Initial reset failed')
        packet=torch.load(checkpoint,map_location='cpu',weights_only=True)
        need(packet['step']==600 and packet['config']==config,'Restricted checkpoint metadata mismatch')
        core.load_state_dict(packet['branch']);need(base.v7.state_info(core)==final,'Final checkpoint roundtrip failed')
        save(out/'checkpoint_roundtrip.json',dict(passed=True,initial=plan['initial_state'],final=final,reloaded=base.v7.state_info(core),
            checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),step=600,config_sha256=sha(out/'config.json')))
        current_phase='evaluation';capture_current=True
        with torch.no_grad():
            for offset in range(0,108,16):
                current_step=offset//16+1;io={};chosen=rows[offset:offset+16];sids=[r['sid'] for r in chosen]
                h,g,layout,valid=batch_states(torch,scenes,states,index,codes,code_offsets,sids)
                first=torch.tensor(layout['offsets'][:-1],device=device);h=h[:,first];g=g[first];valid=valid[:,first]
                final_layout=dict(first_query_only=True,target_ids=[r['first_token_id'] for r in chosen],full_target_ids=[r['target_ids'] for r in chosen],
                    first_indices=layout['offsets'][:-1],original_full_layout=layout)
                delta,cap=core(h,g,valid_mask=valid,output_dtype=torch.float32,capture=True)
                logits=head(norm((g+delta.half()).unsqueeze(0)))[0]
                record=save_capture(h,g,valid,cap,final_layout,sids,checkpoint,600)
                gpu_nll=torch.nn.functional.cross_entropy(logits.float(),torch.tensor(final_layout['target_ids'],device=device),reduction='none')
                raw_batches.append(dict(capture_file=record['file'],capture_sha256=record['sha256'],sids=sids,
                    gpu_fp32_nll=gpu_nll.tolist(),logits=tensor_info(io['logits'])))
        packets=[torch.load(r['capture_file'],map_location='cpu',weights_only=True) for r in raw_batches]
        raw_logits=torch.cat([p['logits'] for p in packets]);del packets
        raw_file=data/'final_logits.pt';torch.save(dict(schema_version=1,sids=[r['sid'] for r in rows],logits=raw_logits,batches=raw_batches),raw_file)
        targets=torch.tensor([r['first_token_id'] for r in rows]);ids=raw_logits.argmax(-1)
        promoted=raw_logits.double();shift=promoted-promoted.amax(-1,keepdim=True)
        nll=torch.logsumexp(shift,-1)-shift[torch.arange(108),targets];gpu_losses=[v for batch in raw_batches for v in batch['gpu_fp32_nll']]
        predictions=[dict(r,argmax_id=int(ids[i]),first_token_correct=int(ids[i])==r['first_token_id'],gpu_fp32_nll=gpu_losses[i],nll=float(nll[i])) for i,r in enumerate(rows)]
        prediction_file=out/'predictions.json';save(prediction_file,predictions);save(out/'captures.json',captures);save(out/'calls.json',calls)
        native_final=dict(norm=tensor_info(norm.weight),head=tensor_info(head.weight));input_after=dict(codes=tensor_info(codes),globals=tensor_info(states))
        expected=dict(core=607,norm=607,head=607,vlm=0,vision=0)
        need(counters==expected and native_final==native_initial and base.v7.state_info(core)==final
             and input_before==input_after and input_versions==(codes._version,states._version)
             and all(not p.requires_grad and p.grad is None for p in core.parameters()),'Final endpoint/calls/frozen inputs differ')
        endpoint=dict(passed=True,before_evaluation=final,after_evaluation=base.v7.state_info(core),parameter_sha256=object_sha(final),
            native_before=native_initial,native_after=native_final,native_weights_unchanged=True,counters=counters,
            input_tensors_before=input_before,input_tensors_after=input_after,input_versions_unchanged=True,codes_frozen=True,
            checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),oracle_local_semantics=True,no_answer_or_intersection_code=True)
        save(out/'final_endpoint.json',endpoint)
        criterion=oracle.criteria(predictions);criterion.pop('oracle_answer_code_supplied');criterion['cached_first_query_only']=True
        result=dict(**config,passed=True,completed=True,phase='run',computational_integrity_passed=True,steps=600,
            checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),selected_parameter_sha256=object_sha(final),checkpoint_roundtrip_passed=True,
            training_file=str(out/'training.json'),training_sha256=sha(out/'training.json'),captures_file=str(out/'captures.json'),captures_sha256=sha(out/'captures.json'),
            raw_file=str(raw_file),raw_sha256=sha(raw_file),raw_tensor=tensor_info(raw_logits),raw_batches=raw_batches,
            predictions_file=str(prediction_file),predictions_sha256=sha(prediction_file),calls_file=str(out/'calls.json'),calls_sha256=sha(out/'calls.json'),counters=counters,
            endpoint_file=str(out/'final_endpoint.json'),endpoint_sha256=sha(out/'final_endpoint.json'),
            roundtrip_file=str(out/'checkpoint_roundtrip.json'),roundtrip_sha256=sha(out/'checkpoint_roundtrip.json'),first_token_fit=criterion,
            training_head_rows=21334,final_head_rows=108,total_head_rows=21442,setup_before_training_seconds=setup,training_seconds=sum(r['seconds'] for r in logs),
            native_weights_unchanged=True,codes_frozen=True,input_versions_unchanged=True,cached_first_query_only=True,no_pretrained_backbone_loaded=True,
            no_dev_or_test=True,no_native_or_whole_answer_claim=True)
        need(time.perf_counter()-started<=90,'Single head-only joint-code job exceeded90 seconds');return result
    finally:
        for handle in handles:handle.remove()
        save(out/'final_counters.json',dict(counters=counters,recorded_calls=calls,partial_outputs_retained=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__);action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--check',action='store_true');action.add_argument('--run',action='store_true');parser.add_argument('--plan',type=Path)
    args=parser.parse_args();native.require_slurm(gpu=args.run)
    if args.check:need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),'CPU-only source/input check')
    else:need(args.plan is not None,'Passed frozen CPU plan required')
    phase='check' if args.check else 'run';out=OUT/f'{phase}_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();frozen=snapshot(out);save(out/'request.json',dict(phase=phase,plan=None if args.plan is None else str(args.plan),source_sha256=frozen))
    try:
        result=check(out,frozen) if args.check else run(args,out,frozen,started)
        need(sources()==frozen and inherited_sources()==read(out/'inherited_sources.json')['source_sha256'],'Source changed during joint-code execution')
        result['elapsed_seconds']=time.perf_counter()-started;save(out/'summary.json',result)
        print(json.dumps(dict(passed=result['passed'],directory=str(out))),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,
            elapsed_seconds=time.perf_counter()-started,partial_outputs_retained=True));raise


if __name__=='__main__':main()

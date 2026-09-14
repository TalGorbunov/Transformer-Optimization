"""Independent Slurm CPU software-profile audit of training-only prefix targets.

No vision, decoder, optimizer, fitting, or backward execution. Actual saved native
heads and auxiliary arithmetic are replayed; decoder gradients remain GPU evidence.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import report_mmred_official_native_memory as reference
from scripts import report_mmred_official_native_training as training_audit
from scripts import train_mmred_prefix_supervision as producer

need=reference.need;sha=reference.sha;read=reference.read;save=reference.save
object_sha=reference.object_sha;p=reference.p
PROTOCOL='mmred_prefix_supervision_independent_audit'
OUT=REPO/'outputs/native_aggregation_vlm/mmred_prefix_supervision_audit'
DATA=Path('/mnt/data/gabriele/gnn_transformer/mmred_prefix_supervision_audit')
OWN=('scripts/report_mmred_prefix_supervision.py','slurm/mmred_prefix_supervision_report.sbatch')
PEOPLE=('Daniel','John','Mary','Michael','Sandra')
ROOMS=('Bathroom','Bedroom','Garden','Hallway','Kitchen','Office')
PAIRS=tuple((a,b) for i,a in enumerate(PEOPLE) for b in PEOPLE[i+1:])
CHANNELS=tuple(f'person_room/{p}/{r}' for p in PEOPLE for r in ROOMS)+tuple(f'co_location/{a}/{b}' for a,b in PAIRS)
ARMS=('answer','local','prefix');WITNESSES=(0,5,800,810,1600,1612,2400,2405,2410,3203,3210,3222);ROUNDTRIP=(0,3222)
POLICY=dict(cpu_seconds=900,cpu_cores=4,memory_gib=16,publication_reserve_seconds=30,
    cpu_native_tv_max=.02,cpu_argmax_gate=False,core_atol=2e-4,core_rtol=2e-4,
    ce_absolute_tolerance=2e-6,auxiliary_lambda=.1,projection_seed=20260914,
    boundary='pre_block27',auxiliary_parameters=0,model_calls=0,vision_calls=0,
    decoder_calls=0,backward_calls=0,optimizer_steps=0,no_inference_supervision=True,
    no_efficacy_gate=True,no_fit_release=True,collect_numerical_evidence_before_gate=True,
    teacher_calls=92,auxiliary_captures=90,natural_trajectories=8,maximum_head_calls=492,maximum_head_rows=1136)


def sources():return {name:sha(REPO/name) for name in OWN}
def ref(path):return dict(file=str(Path(path).resolve()),sha256=sha(path))
def finite_number(value):
    value=float(value);return value if math.isfinite(value) else None


def world_targets(states):
    """Independent exact integer oracle, in original exposure order."""
    need(isinstance(states,list) and states,'Nonempty original world required')
    local=[];prefix=[];total=[0]*40;previous=None
    for index,state in enumerate(states):
        need(type(state['step_id']) is int and state['step_id']==index+1 and set(state['rooms'])==set(ROOMS),'Original Step/room inventory differs')
        where={}
        for room,people in state['rooms'].items():
            need(isinstance(people,list),'Original occupant list required')
            for person in people:
                need(person in PEOPLE and person not in where,'Each original person occurs exactly once');where[person]=room
        need(set(where)==set(PEOPLE),'Missing original person')
        need(previous is None or sum(previous[k]!=where[k] for k in PEOPLE)==1,'Original one-person movement law differs')
        row=[int(where[person]==room) for person in PEOPLE for room in ROOMS]+[int(where[x]==where[y]) for x,y in PAIRS]
        total=[a+b for a,b in zip(total,row)];local.append(row);prefix.append(list(total));previous=where
    return dict(local=local,prefix=prefix)


def population_moments(torch,values):
    means=[];seconds=[]
    for value in values:
        need(value.ndim==2 and value.shape[1]==40,'Population N-by40 targets required')
        means.append(value.double().mean(0));seconds.append(value.double().square().mean(0))
    mean=torch.stack(means).mean(0);variance=torch.stack(seconds).mean(0)-mean.square()
    need(bool((variance>=-1e-12).all()),'Population variance invalid');variance=variance.clamp_min(0)
    return dict(mean=mean,variance=variance,scale=torch.where(variance<=1e-12,torch.ones_like(variance),variance.sqrt()))


def fixed_projection(torch):
    generator=torch.Generator(device='cpu').manual_seed(20260914)
    q,r=torch.linalg.qr(torch.randn((3584,40),generator=generator,dtype=torch.float64),mode='reduced')
    need(bool((r.diagonal()!=0).all()),'Full-rank Gaussian draw required')
    return (q*torch.where(r.diagonal()>0,1.,-1.)[None,:]).T.float().contiguous()


def auxiliary_reference(torch,h,projection,mean,scale,targets):
    need(h.shape==(targets.shape[0],3584) and targets.shape[1]==40 and projection.shape==(40,3584)
         and mean.shape==scale.shape==(40,),'Fixed auxiliary tensor geometry differs')
    prediction=h.double()@projection.double().T;normalized=(targets.double()-mean.double())/scale.double()
    residual=prediction-normalized
    return dict(predictions=prediction,normalized_targets=normalized,decoded=prediction*scale.double()+mean.double(),
                loss=residual.square().mean(),prediction_gradient=2*residual/residual.numel())


def safe_tree(torch,value):
    if isinstance(value,torch.Tensor):need(value.device.type=='cpu' and not value.requires_grad,'Detached CPU tensor required')
    elif isinstance(value,dict):
        for key,item in value.items():need(type(key) in (str,int),'Primitive packet keys required');safe_tree(torch,item)
    elif isinstance(value,(tuple,list)):
        for item in value:safe_tree(torch,item)
    else:need(value is None or type(value) in (str,int,float,bool),'Live model/cache/custom object in packet')


class Audit:
    def __init__(self,torch,out,data,started):
        self.torch=torch;self.out=out;self.data=data;self.started=started;self.bindings={};self.manifest={}
        self.checks=[];self.head_calls=0;self.head_rows=0;self.head_seconds=0.;self.teacher_seconds=0.;self.supervision_calls=0
    def guard(self):need(time.perf_counter()-self.started<870,'Audit publication reserve reached')
    def bind(self,path,digest=None):
        path=str(Path(path).resolve())
        if path in self.bindings:
            need(digest is None or self.bindings[path]==digest,'Conflicting bound hash');return self.bindings[path]
        return reference.bind(path,self.bindings,digest)
    def record(self,path,digest=None):self.bind(path,digest);return read(path)
    def load(self,descriptor,external=False):
        self.guard();path=str(Path(descriptor['file']).resolve());digest=descriptor['sha256']
        need(external or self.manifest.get(path)==digest,'Packet missing from producer artifact manifest')
        self.bind(path,digest);value=self.torch.load(path,map_location='cpu',weights_only=True);safe_tree(self.torch,value);return value
    def add(self,key,passed,**details):
        value=dict(key=key,passed=bool(passed),**details);self.checks.append(value);return value
    def close(self,key,got,want,atol=2e-4,rtol=2e-4):
        need(got.shape==want.shape,'Tensor ownership/shape differs: '+key)
        delta=(got.double()-want.double()).abs();finite=bool(got.isfinite().all() and want.isfinite().all())
        return self.add(key,finite and bool((delta<=atol+rtol*want.double().abs()).all()),
            finite=finite,maximum_absolute_error=finite_number(delta.max()),atol=atol,rtol=rtol)
    def exact(self,key,got,want):return self.add(key,self.torch.equal(got,want))


def fixture(torch):
    rooms={r:[] for r in ROOMS};rooms['Office']=list(PEOPLE)
    first=dict(step_id=1,rooms=rooms);second=json.loads(json.dumps(first));second['step_id']=2
    second['rooms']['Office'].remove('Daniel');second['rooms']['Kitchen']=['Daniel']
    target=world_targets([first,second]);need(sum(target['local'][0])==15 and sum(target['local'][1])==11
        and target['prefix'][1]==[x+y for x,y in zip(target['local'][0],target['local'][1])],'Integer oracle fixture failed')
    bad=json.loads(json.dumps(second));bad['rooms']['Office'].remove('John');bad['rooms']['Kitchen'].append('John')
    rejected=False
    try:world_targets([first,bad])
    except ValueError:rejected=True
    need(rejected,'Illegal two-move fixture was accepted')
    projection=torch.zeros((40,3584),dtype=torch.float32);projection[:,:40]=torch.eye(40)
    h=torch.zeros((2,3584),dtype=torch.float16);h[:,:40]=2
    result=auxiliary_reference(torch,h,projection,torch.ones(40),torch.full((40,),2.),torch.ones((2,40)))
    need(float(result['loss'])==4. and bool((result['prediction_gradient']==.05).all())
         and bool((result['decoded']==5).all()),'Affine/MSE/mean-gradient fixture failed')
    moment=population_moments(torch,[torch.zeros((1,40)),torch.ones((4,40))])
    need(bool((moment['mean']==.5).all()) and bool((moment['variance']==.25).all()),'Equal-world weighting fixture failed')
    return dict(passed=True,integer_oracle=True,illegal_transition_rejected=True,affine_mean_gradient=True,equal_world_weighting=True)


def inputs(a,summary_path):
    from scripts import prepare_mmred_prefix_supervision as preparation
    from scripts import evaluate_mmred_official_native_memory as old
    summary_path=Path(summary_path).resolve();directory=summary_path.parent
    summary=a.record(summary_path);need(summary['protocol']==producer.PROTOCOL and summary['passed'] is summary['completed'] is True
        and summary['phase']=='profile' and not (directory/'failure.json').exists(),'Complete successful original profile required')
    config=a.record(directory/'config.json');analysis=a.record(summary['analysis']['file'],summary['analysis']['sha256'])
    need(analysis['passed'] is analysis['completed'] is True and analysis['phase']=='profile'
         and analysis['no_fit_release'] and analysis['no_validation_test_inference']
         and not analysis['objective_achieved'] and summary['counters']==analysis['counters'],
         'Original complete profile scope differs')
    a.manifest=a.record(summary['artifacts']['file'],summary['artifacts']['sha256'])
    need(all(Path(k).is_absolute() for k in a.manifest),'Absolute artifact ownership required')
    release=a.record(config['source_release']['file'],config['source_release']['sha256'])
    need(producer.verify_release(config['source_release']['file'])==release and summary['source_release']==config['source_release']
         and summary['targets']==config['targets'] and config['protocol']==analysis['protocol']==producer.PROTOCOL
         and config['policy']==analysis['policy']==producer.POLICY,'Producer policy/release joins differ')
    for name,digest in release['source_sha256'].items():
        a.bind(REPO/name,digest);a.bind(directory/'source'/name.replace('/','_'),digest)
    for file,digest in a.manifest.items():a.guard();a.bind(file,digest)
    need(a.record(directory/'base_before.json')==a.record(directory/'base_after.json'),'Frozen base identity changed')
    request=a.record(directory/'request.json')
    need(request['phase']=='profile' and request['source_release']==config['source_release'] and request['targets']==config['targets']
         and request['policy']==producer.POLICY,'Profile request differs')
    a.bind(producer.TRAIN_PLAN,producer.TRAIN_PLAN_SHA);a.bind(producer.EVAL_PLAN,producer.EVAL_PLAN_SHA)
    plan=old.training.verify_plan(producer.TRAIN_PLAN);evaluation=old.verify_plan(producer.EVAL_PLAN)
    need(config['training_plan']==ref(producer.TRAIN_PLAN) and config['evaluation_plan']==ref(producer.EVAL_PLAN)
         and config['native_identity_sha256']==plan['native_identity_sha256']==evaluation['native_identity_sha256'],'Original input/native lineage differs')
    target_plan=preparation.verify_stage(config['targets']['file']);a.bind(config['targets']['file'],config['targets']['sha256'])
    need(target_plan==config['target_plan'] and target_plan['training_plan']==config['training_plan'],'Target stage does not bind original training exposure')
    target_summary=read(config['targets']['file']);a.bind(target_summary['plan_file'],target_summary['plan_sha256'])
    for mapping in (target_plan['source_sha256'],target_plan['inherited_source_sha256']):
        for name,digest in mapping.items():a.bind(REPO/name,digest)
    proof=evaluation['main_arms']['ordinary'];need(config['initial']==analysis['initial']==proof['final_checkpoint']
        and config['peft_config']==proof['peft_config'],'Exact fitted ordinary1500 starting checkpoint required')
    initial=a.load(config['initial'],external=True);reference.adapter_state(a.torch,initial['adapter'])
    need(old.state_proof(a.torch,initial,'ordinary',config['peft_config'])==proof['state_proof'],'Actual initial tensor ownership differs')
    contract=config['contract'];need(contract['targets']==list(p.TARGETS) and contract['trainable_parameters']==10092544
         and set(contract['tensors'])==set(initial['adapter']) and contract['rank']==16 and contract['alpha']==32
         and contract['dropout']==.05 and contract['scaling']==2.,'Actual224 language-only adapter contract differs')
    for name,v in initial['adapter'].items():need(contract['tensors'][name]['shape']==list(v.shape)
        and contract['tensors'][name]['dtype']=='torch.float32','Adapter contract tensor geometry differs')
    need(config['float32_matmul_precision']=='highest' and config['cuda_matmul_allow_tf32'] is False,
         'Actual auxiliary FP32 matmul policy required')
    items=a.record(plan['cases_file']);need(len(items)==4000,'Original4000 training entries required')
    return directory,config,analysis,plan,target_plan,items,initial,release


def target_audit(a,target_plan,items):
    torch=a.torch;packet=a.load(target_plan['projection'],external=True);records=a.load(target_plan['targets'],external=True)['records']
    index=a.record(target_plan['target_index']['file'],target_plan['target_index']['sha256'])
    ledger=a.record(target_plan['input_bindings']['file'],target_plan['input_bindings']['sha256'])
    tests=a.record(target_plan['tests']['file'],target_plan['tests']['sha256'])
    need(tests['passed'] is True and tests['tests_run']==7
         and tests['errors']==tests['failures']==tests['skipped']==[]
         and tests['image_positions']==dict(accepted=1,passed=True,rejected=2),
         'Prepared core and image-position fixtures did not all pass')
    need(len(records)==len(index)==4000 and packet['channel_names']==list(CHANNELS) and packet['people']==list(PEOPLE)
         and packet['rooms']==list(ROOMS) and packet['pairs']==[list(x) for x in PAIRS] and packet['seed']==20260914
         and packet['raw_hidden_projection'] and packet['trainable_parameters']==0,'Fixed channel ordering/zero-parameter packet differs')
    a.exact('fixed_QR_projection',packet['projection'],fixed_projection(torch))
    a.close('stored_projection_orthonormal',packet['projection'].double()@packet['projection'].double().T,torch.eye(40,dtype=torch.float64),2e-6,0.)
    rawfiles={};total=0
    for i,(row,entry,item) in enumerate(zip(records,index,items)):
        a.guard();r=item['row'];need(row['index']==entry['index']==i and row['sid']==entry['sid']==r['sid']
            and row['n']==entry['n']==r['n'] and entry['compact']==item['compact'] and entry['features']==item['features'],
            'Every target retains original source/compact/feature order')
        file=str(Path(entry['renderer_file']).resolve());need(file in ledger,'Renderer missing from preparation input bindings')
        if file not in rawfiles:rawfiles[file]=a.record(file,ledger[file])
        raw=rawfiles[file][entry['renderer_row']]
        need(entry['renderer_file']==r['renderer_file'] and entry['renderer_row']==r['renderer_row']
             and object_sha(raw)==entry['raw_source_sha256'] and raw['question']==r['question']
             and object_sha(raw['sequence'])==r['exact_sequence_sha256']
             and raw['qid']==r['qid'] and raw['atype']==r['atype']
             and raw['qtype']==r['qtype'] and str(raw['answer'])==str(r['answer']),'Original question/world ownership differs')
        expected=world_targets(raw['sequence']);need(len(raw['sequence'])==row['n'],'Original N differs')
        for mode in ('local','prefix'):
            need(row[mode].dtype==torch.int64 and row[mode].shape==(row['n'],40)
                 and torch.equal(row[mode],torch.tensor(expected[mode],dtype=torch.int64)),'Independent integer targets differ')
        need(row['image_end_positions'].dtype==torch.int64 and row['image_end_positions'].tolist()==entry['image_end_positions']
             and entry['tensors']=={k:p.tensor_info(row[k]) for k in ('local','prefix','image_end_positions')},'Target/index tensor ownership differs')
        total+=row['n']
    need(total==24800==target_plan['total_prefixes'],'Original frame/exposure count differs')
    for mode in ('local','prefix'):
        normalized=packet['normalization'][mode];computed=population_moments(torch,[v[mode] for v in records])
        need(normalized['worlds']==4000 and normalized['weighting']=='equal_world_then_equal_prefix' and normalized['variance_floor']==1e-12,'Training-only normalization convention differs')
        for field in ('mean','variance','scale'):a.close(mode+'_'+field,normalized[field+'_fp64'],computed[field],1e-11,1e-11)
        a.exact(mode+'_actual_mean_buffer',normalized['mean'],computed['mean'].float())
        a.exact(mode+'_actual_scale_buffer',normalized['scale'],computed['scale'].float())
    return packet,records,dict(passed=True,worlds=4000,prefixes=total,integer_channels=40,equal_world_exposure_multiplicity_preserved=True)


def gradient_metadata(value,names,probe=False):
    need(set(value['present'])==set(value['finite'])==set(names) and value['all_present']==all(value['present'].values())
         and value['all_finite']==all(value['finite'].values()) and value['all_finite']
         and len(value['layer_l2'])==28 and all(math.isfinite(x) and x>=0 for x in value['layer_l2']), 'GPU gradient inventory/finiteness differs')
    if probe:need(all(x>0 for x in value['layer_l2'][:27]) and value['layer_l2'][27]==0.,'Auxiliary gradient reach must end before block27')
    else:need(value['all_present'],'All224 native training gradients required')


def states_audit(a,config,analysis,initial,directory):
    torch=a.torch;refs=analysis['states'];need(len(refs)==9,'Exactly nine profile checkpoints required')
    states={};rng=None;proof=[]
    for arm in ARMS:
        states[arm]={}
        for update in (0,1,2):
            name=f'{arm}_seed25_update{update}.pt';descriptor=refs[name];value=a.load(descriptor)
            need(descriptor['arm']==value['arm']==arm and descriptor['seed']==value['seed']==25 and descriptor['updates']==value['updates']==update
                 and value['policy']==producer.POLICY and value['initial']==config['initial'] and value['source_release']==config['source_release']
                 and value['targets']==config['targets'] and value['peft_config']==config['peft_config'],'Checkpoint owner/policy differs')
            reference.adapter_state(torch,value['adapter']);need(value['rng']['cpu'].dtype==torch.uint8 and len(value['rng']['cuda'])==1
                and value['rng']['cuda'][0].dtype==torch.uint8,'CPU/CUDA RNG record required')
            if update==0:
                for k,v in value['adapter'].items():a.exact(arm+'_initial_'+k,v,initial['adapter'][k])
                if rng is None:rng=value['rng']
                else:
                    a.exact(arm+'_common_cpu_rng',value['rng']['cpu'],rng['cpu']);a.exact(arm+'_common_cuda_rng',value['rng']['cuda'][0],rng['cuda'][0])
            opt=value['optimizer'];need(len(opt['param_groups'])==1,'One AdamW group required');group=opt['param_groups'][0]
            need(group['lr']==2e-4 and tuple(group['betas'])==(.9,.999) and group['eps']==1e-8 and group['weight_decay']==.01
                 and len(group['params'])==224,'Fixed AdamW recipe differs')
            need(set(opt['state'])==(set() if update==0 else set(group['params'])),'Fresh optimizer or complete updated moments required')
            if update:
                for identifier,v in zip(group['params'],value['adapter'].values()):
                    m=opt['state'][identifier];need(set(m)=={'step','exp_avg','exp_avg_sq'} and float(m['step'])==update
                         and all(m[k].shape==v.shape and m[k].dtype==torch.float32 and bool(m[k].isfinite().all()) for k in ('exp_avg','exp_avg_sq')),
                         'Actual optimizer moment tensor/step differs')
            states[arm][update]=descriptor
            del value
        need(analysis['final_checkpoints'][arm]==states[arm][2],'Final software checkpoint alias differs')
        for update in (1,2):
            log=a.record(directory/f'{arm}_update{update}.json');grad=a.load(log['gradients'])
            reference.adapter_state(torch,grad)
            need(log['arm']==arm and log['update']==update and log['parameter_steps']==[float(update)]*224
                 and math.isfinite(log['seconds']) and log['seconds']>=log['gradient_save_seconds']>=0,'Optimizer trace differs')
            norm=sum(v.double().square().sum() for v in grad.values()).sqrt()
            a.add(f'{arm}_gradient_norm_{update}',abs(float(norm)-log['gradient_norm'])<=2e-4+2e-4*float(norm),
                  actual=log['gradient_norm'],expected=float(norm))
            layers=[0.]*28
            for name,v in grad.items():layers[int(re.search(r'\.layers\.(\d+)\.',name).group(1))]+=float(v.double().square().sum())
            broad={letter:math.sqrt(sum(float(v.double().square().sum()) for name,v in grad.items() if '.lora_'+letter+'.' in name)) for letter in ('A','B')}
            a.add(f'{arm}_live_gradient_groups_{update}',all(x>0 for x in layers) and all(x>0 for x in broad.values()),layer_l2=[math.sqrt(x) for x in layers],broad_l2=broad)
            proof.append(dict(arm=arm,update=update,gradients=log['gradients'],layer_l2=[math.sqrt(x) for x in layers],broad_l2=broad))
            del grad
    return states,proof


def supervision_audit(a,key,e,case,arm,row,packet,backward,accumulation):
    torch=a.torch;o=e['prefix_supervision'];mode='local' if arm=='answer' else arm
    positions=(case['teacher_input_ids'][0]==151653).nonzero().flatten();N=row['n']
    need(o['arm']==arm and o['target_arm']==mode and o['layer']==27 and o['no_injected_states'] is True
         and o['accumulation']==accumulation and o['hidden_rows'].shape==(N,3584) and o['hidden_rows'].dtype==torch.float16
         and torch.equal(o['positions'],positions) and torch.equal(positions,row['image_end_positions'])
         and positions.numel()==N and bool((positions<case['metadata']['prompt_width']-1).all()),'Actual native pre-block27 image sites differ')
    P=packet['projection'];mean=packet['normalization'][mode]['mean'];scale=packet['normalization'][mode]['scale']
    for name,v in (('projection',P),('mean',mean),('scale',scale)):
        need(o[name].dtype==torch.float32,'Actual frozen FP32 buffers required');a.exact(key+'_'+name,o[name],v)
    a.exact(key+'_targets',o['raw_targets'],row[mode])
    result=auxiliary_reference(torch,o['hidden_rows'],P,mean,scale,row[mode])
    file=a.data/(key+'_auxiliary_fp64.pt');torch.save(result,file)
    for name in ('predictions','normalized_targets','decoded'):a.close(key+'_'+name,o[name],result[name])
    expected=float(result['loss']);actual=o['diagnostic_auxiliary_loss'];weight=0. if arm=='answer' else .1
    a.add(key+'_MSE',math.isfinite(actual) and abs(actual-expected)<=2e-4+2e-4*abs(expected),actual=actual,expected=expected,reference=ref(file))
    need(o['auxiliary_weight']==e['auxiliary_weight']==weight and o['auxiliary_loss']==e['auxiliary_loss']==(0. if arm=='answer' else actual)
         and o['base_ce']==e['loss'] and e['total_loss']==o['total_loss'] and e['backward_loss']==e['total_loss']/accumulation,
         'Native CE/auxiliary/total backward trace links differ')
    total=e['loss']+weight*o['auxiliary_loss'];a.add(key+'_total',abs(e['total_loss']-total)<=2e-6+2e-6*abs(total),actual=e['total_loss'],expected=total)
    if backward and weight:
        need(o['prediction_gradient'] is not None,'Actual attached prediction gradient required')
        a.close(key+'_prediction_gradient',o['prediction_gradient'],result['prediction_gradient']*(weight/accumulation),2e-7,2e-4)
    else:need(o['prediction_gradient'] is None,'No auxiliary backward allowed for answer-only/evaluation')
    a.supervision_calls+=1


def teachers(a,config,analysis,items,records,packet,states,cases,gradient_proof):
    expected=[]
    for i in ROUNDTRIP:expected.append((f'baseline_{i}',i,'answer',config['initial'],False,False,8))
    for arm in ARMS:
        for i in ROUNDTRIP:expected.append((f'{arm}_initial_{i}',i,arm,config['initial'],False,True,8))
        for update in (1,2):
            for j,i in enumerate(WITNESSES):expected.append((f'{arm}_update{update}_micro{j}',i,arm,states[arm][update-1],True,True,12))
        for phase in ('before','after'):
            for i in ROUNDTRIP:expected.append((f'{arm}_roundtrip_{phase}_{i}',i,arm,states[arm][2],False,True,8))
    entries=analysis['retained_teachers'];need(len(entries)==92 and len({v['key'] for v in entries})==92,'Complete92 unique teacher captures required')
    need([v['key'] for v in entries]==[v[0] for v in expected],'Original teacher call order differs')
    lookup={v['key']:v for v in entries};logits={};results=[];probes=0
    for key,i,arm,state,backward,capture,accumulation in expected:
        a.guard();tick=time.perf_counter();entry=lookup[key];raw=a.load(entry);e=raw['evidence'];case=cases[i]
        need(entry['index']==raw['case_index']==i and entry['arm']==raw['arm']==arm and raw['phase']=='teacher'
             and entry['parameter_state']==e['parameter_state']==state and entry['backward']==raw['backward_completed']==backward
             and e['feature_packet']==items[i]['features'] and e['compact_packet']==items[i]['compact'], 'Teacher input/state/arm owner differs')
        need(e['rope_restored'] and e['hooks_removed'] and not e.get('pixel_route',False),'Original teacher restoration/pixel route differs')
        need(e['actual_language_embeddings']==e['embedding_identity']==e['native_inputs'][0]['inputs_embeds_identity'],
             'Actual native teacher embedding ownership differs')
        # All three experiment arms use the ordinary native route. Verify true
        # owners above; route only the immutable helper's local arm selector.
        routed=dict(raw,arm='ordinary');tick_head=time.perf_counter()
        result=reference.teacher_audit(a.torch,routed,case,'ordinary',i,a.modules,a.data,key,True,backward)
        a.head_seconds+=time.perf_counter()-tick_head;a.head_calls+=1;a.head_rows+=result['target_rows']
        a.add(key+'_native',result['passed'],result=result);results.append(dict(key=key,index=i,arm=arm,head=result['head'],passed=result['passed']))
        need(('prefix_supervision' in e)==capture,'Passive capture scope differs')
        if capture:
            need(e['prefix_supervision']['targets_descriptor']==config['target_plan']['targets'],'Captured target packet differs')
            supervision_audit(a,key,e,case,arm,records[i],packet,backward,accumulation)
        else:need(e['auxiliary_loss']==e['auxiliary_weight']==0. and e['total_loss']==e['loss'],'Baseline teacher gained supervision')
        if backward:gradient_metadata(raw['gradient_summary'],config['contract']['tensors'])
        else:need(raw['gradient_summary'] is None,'Evaluation teacher unexpectedly has gradients')
        if backward and key.endswith('_micro11'):
            update=int(key.split('_update')[1].split('_')[0])
            proof=next(v for v in gradient_proof if v['arm']==arm and v['update']==update)
            a.close(key+'_actual_accumulated_gradient_norms',
                a.torch.tensor(raw['gradient_summary']['layer_l2'],dtype=a.torch.float64),
                a.torch.tensor(proof['layer_l2'],dtype=a.torch.float64))
        is_probe=key in ('local_update1_micro0','prefix_update1_micro0')
        need(('auxiliary_gradient_probe' in e)==is_probe,'Fixed two-probe selection differs')
        if is_probe:gradient_metadata(e['auxiliary_gradient_probe'],config['contract']['tensors'],True);probes+=1
        if not backward:logits[key]=e['profile_head'][0]['head_logits'].clone()
        a.teacher_seconds+=time.perf_counter()-tick
        save(a.out/(key+'_audit.json'),dict(result=results[-1],checks=[v for v in a.checks if v['key'].startswith(key)]))
        del raw
    need(probes==2 and a.supervision_calls==90,'Auxiliary numerical/probe coverage differs')
    parity=[]
    for arm in ARMS:
        for i in ROUNDTRIP:
            equal=a.torch.equal(logits[f'baseline_{i}'],logits[f'{arm}_initial_{i}'])
            parity.append(dict(arm=arm,index=i,logits_bit_equal=equal));a.add(f'{arm}_passive_capture_{i}',equal)
            a.exact(f'{arm}_reload_{i}',logits[f'{arm}_roundtrip_before_{i}'],logits[f'{arm}_roundtrip_after_{i}'])
    need(parity==analysis['parity'],'Saved initial parity metadata differs')
    return results


def naturals(a,config,analysis,items,states,cases,processor):
    expected=[('answer',i,config['initial']) for i in ROUNDTRIP]+[(arm,i,states[arm][2]) for arm in ARMS for i in ROUNDTRIP]
    rows=analysis['natural'];need(len(rows)==8,'Fixed eight native trajectories required');result=[]
    for j,(row,(arm,i,state)) in enumerate(zip(rows,expected)):
        a.guard();raw=a.load(row);e=raw['evidence'];item=items[i]
        need(row['arm']==raw['arm']==arm and row['index']==raw['case_index']==i and row['parameter_state']==e['parameter_state']==state
             and row['sid']==item['row']['sid'] and row['n']==item['row']['n'] and row['qtype']==item['row']['qtype']
             and row['auxiliary_hook_installed'] is False and e['prefix_supervision_active'] is False and 'prefix_supervision' not in e
             and e['compact_packet']==item['compact'] and e['feature_packet']==item['features'],'Inference has a changed input/state or supervision hook')
        need(1<=len(row['generated_ids'])<=50 and math.isfinite(row['seconds']) and row['seconds']>0,'Bounded original native generation required')
        # Metadata-only ordinary routing; all actual experiment owners checked.
        tick=time.perf_counter();metric=training_audit.natural_audit(a.torch,dict(raw,arm='ordinary'),row,cases[i],item,'ordinary',
            dict(final_checkpoint=state),None,processor,a.modules,a.data,f'natural_{j}')
        a.head_seconds+=time.perf_counter()-tick;a.head_calls+=metric['head_calls'];a.head_rows+=metric['head_rows']
        metric['memory_reconstruction_scope']='not_applicable_ordinary_native';metric['actual_arm']=arm
        a.add(f'natural_{j}',metric['passed'],result=metric);result.append(metric);save(a.out/f'natural_{j}_audit.json',metric)
    return result


def projection(analysis):
    t=analysis['timings']
    need(len(t['teacher'])==92 and len(t['optimizer'])==6 and len(t['checkpoint'])==9 and len(t['generation'])==8,'Complete timing population required')
    for row in t['teacher']:
        need(row['seconds']>0 and row['retention_seconds']>=0 and row['auxiliary_probe_seconds']>=0
             and row['training_work_seconds']==row['seconds']-row['retention_seconds']-row['auxiliary_probe_seconds']
             and row['training_work_seconds']>0,'Inclusive teacher timing decomposition differs')
    need(sum(r['backward'] for r in t['teacher'])==72 and sum(r['auxiliary_probe_seconds']>0 for r in t['teacher'])==2,'Training/probe timing scope differs')
    result={}
    for arm in ARMS:
        T={n:max(r['training_work_seconds'] for r in t['teacher'] if r['arm']==arm and r['backward'] and r['n']==n) for n in (1,2,4,8,16)}
        train=sum(2400*T[n] for n in T);update=1500*max(t['optimizer']);checkpoint=4*max(t['checkpoint'])
        generation=5000*max(r['seconds']/r['tokens'] for r in t['generation']);retained=6*max(r['retention_seconds'] for r in t['teacher'])
        estimate=analysis['setup_seconds']+1.25*(train+update+checkpoint+generation+retained+4*max(T.values()))+60
        result[arm]=dict(seconds=estimate,per_n_micro_seconds=T,training_seconds=train,optimizer_seconds=update,checkpoint_seconds=checkpoint,
            retained_teacher_seconds=retained,training_diagnostic_seconds=generation,cap_seconds=7200,passed=estimate<=7200,
            conservative_instrumented_forecast=True,empirical_not_guarantee=True)
    return json.loads(json.dumps(result))


def resources(directory,out):
    job=directory.name.removeprefix('profile_');need(job.isdigit(),'One profile allocation required')
    command=['sacct','-X','-j',job,'--parsable2','--noheader','--format=JobID%40,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,TimelimitRaw']
    text=subprocess.run(command,capture_output=True,text=True,check=True).stdout;file=out/'sacct.psv';file.write_text(text)
    rows=[r.split('|') for r in text.splitlines() if r.split('|')[0]==job];need(len(rows)==1,'Unique original profile accounting required')
    identifier,name,partition,state,exitcode,elapsed,tres,limit=rows[0]
    gpu=re.findall(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',tres);typed=re.findall(r'(?:^|,)gres/gpu:[^=,]+=(\d+)(?=,|$)',tres)
    count=int(gpu[0]) if gpu else sum(map(int,typed))
    need(name=='mmred_prefix_supervision_profile' and partition=='gpu' and state=='COMPLETED' and exitcode=='0:0'
         and count==1 and int(elapsed)<=900 and int(limit)==15,'Bounded completed profile accounting differs')
    return dict(passed=True,job_id=job,job_name=name,allocated_gpu_seconds=int(elapsed),cap_seconds=900,raw=ref(file))


def run(a,args):
    directory,config,original,plan,target_plan,items,initial,release=inputs(a,args.profile)
    save(a.out/'fixtures.json',fixture(a.torch))
    packet,records,target_proof=target_audit(a,target_plan,items)
    states,gradient_proof=states_audit(a,config,original,initial,directory);del initial
    cases={}
    for i in WITNESSES:
        case,features=training_audit.load_case(a.torch,items[i],a.bindings);cases[i]=case;del features
        need(a.torch.equal(records[i]['image_end_positions'],(case['teacher_input_ids'][0]==151653).nonzero().flatten()),'Retained original image-end positions differ')
    parent=a.record(plan['profile_preparation']['plan_file'],plan['profile_preparation']['plan_sha256'])
    a.modules=reference.native_modules(a.torch,parent,a.bindings)
    from transformers import AutoProcessor,__version__ as version
    processor=AutoProcessor.from_pretrained(str(training_audit.producer.profile.preparation.MODEL),trust_remote_code=True,use_fast=False,local_files_only=True)
    need(p.fingerprint(processor,str(version))==plan['native_identity']['processor'],'Original processor/tokenizer differs')
    teacher_results=teachers(a,config,original,items,records,packet,states,cases,gradient_proof)
    natural_results=naturals(a,config,original,items,states,cases,processor)
    G=sum(len(v['generated_ids']) for v in original['natural']);calls=92+G
    expected=dict(model=calls,backbone=calls,vision=0,language=calls,norm=calls,head=calls,backward=72,optimizer=6,auxiliary_gradient_probes=2)
    need(original['counters']==expected and original['decoder_layer_calls']==[calls]*28 and a.head_calls==calls
         and a.head_calls<=492 and a.head_rows<=1136,'Actual native/backward/optimizer/replay inventory differs')
    forecast=projection(original);need(forecast==original['future_main_projection'],'Independent forecast differs')
    accounting=resources(directory,a.out)
    inherited=dict(release['source_sha256'])
    for name in ('scripts/report_mmred_official_native_memory.py','scripts/report_mmred_official_native_training.py'):
        need(name in inherited,'Frozen numerical helper missing from released source closure')
    return dict(protocol=PROTOCOL,policy=POLICY,phase='profile',passed=all(x['passed'] for x in a.checks),completed=True,
        producer_summary=ref(args.profile),source_release=config['source_release'],targets=config['targets'],initial=config['initial'],
        final_software_checkpoints=original['final_checkpoints'],source_sha256=sources(),inherited_source_sha256=inherited,
        target_audit=target_proof,teacher_audits=teacher_results,natural_audits=natural_results,gradient_audits=gradient_proof,
        numerical_checks=a.checks,numerical_failures=[v['key'] for v in a.checks if not v['passed']],
        actual_gpu_counters=expected,cpu_head_calls=a.head_calls,cpu_head_rows=a.head_rows,auxiliary_fp64_captures=a.supervision_calls,
        cpu_head_seconds=a.head_seconds,teacher_audit_seconds=a.teacher_seconds,resources=accounting,
        future_main_projection=forecast,all_main_projections_passed=all(v['passed'] for v in forecast.values()),
        raw_inputs_verified_in_cpu_audit=True,raw_inputs_rehashed_in_handoff=False,decoder_backward_reexecuted=False,
        gpu_gradient_evidence_scope='all224 presence/finiteness each72 backwards; six actual accumulated gradient vectors; two auxiliary reach summaries',
        no_inference_supervision=True,no_fit_release=True,no_efficacy_gate=True,elapsed_seconds=time.perf_counter()-a.started)


def verify_report(summary_path):
    """JSON/hash-only handoff; raw tensors were consumed in the completed audit."""
    path=Path(summary_path).resolve();summary=read(path)
    need(summary['protocol']==PROTOCOL and summary['policy']==POLICY and summary['passed'] is summary['completed'] is True
         and summary['phase']=='profile' and not (path.parent/'failure.json').exists(),'Passed completed profile audit required')
    for field in ('analysis','input_bindings','artifacts'):
        need(sha(summary[field]['file'])==summary[field]['sha256'],'Audit handoff artifact changed')
    analysis=read(summary['analysis']['file']);need(analysis['passed'] is analysis['completed'] is True and analysis['protocol']==PROTOCOL
        and analysis['policy']==POLICY and not analysis['numerical_failures'] and analysis['source_sha256']==sources()
        and analysis['raw_inputs_verified_in_cpu_audit'] and not analysis['raw_inputs_rehashed_in_handoff'],'Audit analysis/policy differs')
    for name,digest in {**analysis['inherited_source_sha256'],**analysis['source_sha256']}.items():need(sha(REPO/name)==digest,'Bound audit/producer source changed')
    for name,digest in analysis['source_sha256'].items():need(sha(path.parent/'source'/name.replace('/','_'))==digest,'Audit source archive changed')
    for field in ('producer_summary','source_release','targets'):
        need(sha(analysis[field]['file'])==analysis[field]['sha256'],'Original handoff metadata changed')
    return analysis


verify_profile_report=verify_report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--profile',type=Path,required=True);args=parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==4,'Slurm CPU4-only audit required')
    need(not os.environ.get('SLURM_JOB_GPUS') and not os.environ.get('SLURM_STEP_GPUS'),'No GPU allocation permitted')
    started=time.perf_counter();tag='report_'+os.environ['SLURM_JOB_ID'];out=OUT/tag;data=DATA/tag
    out.mkdir(parents=True,exist_ok=False);data.mkdir(parents=True,exist_ok=False);(out/'source').mkdir()
    own=sources()
    for name,digest in own.items():
        destination=out/'source'/name.replace('/','_');shutil.copy2(REPO/name,destination);need(sha(destination)==digest,'Audit source copy changed')
    import torch
    torch.set_num_threads(4);a=Audit(torch,out,data,started)
    save(out/'request.json',dict(protocol=PROTOCOL,policy=POLICY,profile=ref(args.profile),source_sha256=own))
    try:
        analysis=run(a,args);need(sources()==own,'Audit source changed during execution')
        save(out/'analysis.json',analysis);save(out/'input_bindings.json',a.bindings)
        save(out/'artifacts.json',{str(p):sha(p) for root in (out,data) for p in root.iterdir() if p.is_file()})
        need(analysis['passed'],'Independent numerical audit failed after complete evidence collection')
        need(time.perf_counter()-started<900,'CPU audit budget exceeded')
        save(out/'summary.json',dict(protocol=PROTOCOL,policy=POLICY,phase='profile',passed=True,completed=True,
            analysis=ref(out/'analysis.json'),input_bindings=ref(out/'input_bindings.json'),artifacts=ref(out/'artifacts.json'),
            future_main_projection=analysis['future_main_projection'],no_fit_release=True,elapsed_seconds=time.perf_counter()-started))
    except Exception as error:
        save(out/'failure_input_bindings.json',a.bindings);save(out/'partial_numerical_checks.json',a.checks)
        save(out/'failure.json',dict(protocol=PROTOCOL,passed=False,completed=False,type=type(error).__name__,message=str(error),
            elapsed_seconds=time.perf_counter()-started,head_calls=a.head_calls,head_rows=a.head_rows,
            input_bindings=ref(out/'failure_input_bindings.json'),partial_checks=ref(out/'partial_numerical_checks.json'),
            data_directory=str(data),no_automatic_retry=True));raise


if __name__=='__main__':main()

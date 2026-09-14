"""Independent privileged local-joint-code cached training audit; no native efficacy."""
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from collections import Counter,defaultdict
import ast
import math
import os
import re
import subprocess
from scripts.stage_native_vision_v6_teacher import need,read,save,sha,object_sha
from scripts import report_native_identity_join_conditioning_v2 as prior
OWN=('scripts/report_native_identity_join_joint_code.py',)
DIAGNOSTIC_STEPS=(1,2,32,128,300,600)
WEIGHT_NAMES=('query.weight','aggregate_projection.weight','aggregate_projection.bias','up.weight')
close=prior.close
expected_order=prior.expected_order
layout_for=prior.layout_for
first_token_criterion=prior.first_token_criterion
fp64_nll=prior.fp64_nll
rms=prior.rms
tensor_error=prior.tensor_error
replay_metric=prior.replay_metric
allocation_rows=prior.allocation_rows


def driver():
    from scripts import diagnose_native_identity_join_joint_code as module
    return module


def bind(file,digest):need(sha(Path(file))==digest,'Changed bound file: '+str(file))


def local_code(person,room,people,requested,rank=96):
    """One image and requested room names only; no bag or answer argument."""
    need(len(people)==len(set(people))==9 and len(requested)==len(set(requested))==2
         and person in people and rank>=18,'Invalid local semantic domain')
    column=None if room not in requested else 2*people.index(person)+requested.index(room)
    value=[0.]*rank
    if column is not None:value[column]=1.
    return value,column


def check_weights(torch,weights,initial):
    shapes={'query.weight':(96,3584),'aggregate_projection.weight':(96,96),'aggregate_projection.bias':(96,),'up.weight':(3584,96)}
    need(set(weights)==set(initial)==set(shapes) and all(v.dtype==torch.float32 and tuple(v.shape)==shapes[k]
         and bool(torch.isfinite(v).all()) for k,v in weights.items()) and sum(v.numel() for v in weights.values())==697440,
         'Exactly four finite FP32 readout tensors required')


def audit_inputs(torch,plan):
    p=driver();rows=read(plan['rows_file']);scenes=read(plan['scenes_file']);pairs=read(plan['pairs_file']);order=read(plan['order_file'])
    parent_ref=plan['factor_plan'];bind(parent_ref['file'],parent_ref['sha256']);parent=read(parent_ref['file'])
    bind(plan['uniform_plan']['file'],plan['uniform_plan']['sha256']);uniform=read(plan['uniform_plan']['file'])
    need(plan['uniform_plan']==parent['uniform_plan'] and rows==read(parent['rows_file'])
         and scenes=={sid:{k:v for k,v in scene.items() if k not in ('local_feature_ids','image_question_pair_ids')} for sid,scene in read(parent['scenes_file']).items()}
         and pairs==read(parent['pairs_file']) and order==read(parent['order_file'])==expected_order(pairs), 'Original training data/order differ')
    bind(uniform['rows_file'],uniform['runtime_bindings'][str(Path(uniform['rows_file']).resolve())])
    originals=read(uniform['rows_file'])['train'];need(read(plan['training_samples_file'])==[r['sample'] for r in originals],'Original image sample inventory differs');samples={r['sample']['sid']:r['sample'] for r in originals}
    need(len(rows)==len(scenes)==len(samples)==108 and Counter(r['gold'] for r in rows)==Counter({name:12 for name in p.PEOPLE}), 'Balanced original training subset differs')
    for r in rows:
        s=scenes[r['sid']];sample=samples[r['sid']];ids=r['target_ids']
        need(sample['split']==s['split']=='train' and sample['n_frames']==s['n_frames']==r['n_frames'] in (8,16)
             and sample['question']==s['question']==r['question'] and sample['gold']==s['gold']==r['gold']
             and sample['target_ids']==s['target_ids']==ids and ids[-1]==151645 and len(ids)==(3 if r['gold'] in ('Sandra','Noah') else 2)
             and s['target_prefixes']==[ids[:i] for i in range(len(ids))] and r['first_token_id']==ids[0], 'Protected scene/full-name prefix fields differ')
    for key in ('native_identity','native_identity_sha256','native_module_identity','native_model_file','native_model_sha256','head_rows_by_update'):
        need(plan[key]==parent[key], 'Original native/schedule identity differs: '+key)
    bind(plan['features_file'],plan['features_sha256']);packet=torch.load(plan['features_file'],map_location='cpu',weights_only=True)
    states=packet['states'];ids=packet['feature_ids'];wanted=sorted({fid for scene in scenes.values() for fid in scene['global_feature_ids']})
    need(ids==wanted==plan['global_feature_ids'] and states.dtype==torch.float16 and states.shape==(len(wanted),3584)
         and p.tensor_info(states)==plan['feature_tensor'] and bool(torch.isfinite(states).all()), 'Global-only frozen feature packet differs')
    need(plan['parent_features_file']==parent['features_file'] and plan['parent_features_sha256']==parent['features_sha256']
         and plan['parent_feature_tensor']==parent['feature_tensor'],'Parent global-subset source binding differs')
    bind(parent['features_file'],parent['features_sha256']);original=torch.load(parent['features_file'],map_location='cpu',weights_only=True)
    original_index={fid:i for i,fid in enumerate(original['feature_ids'])}
    need(torch.equal(states,original['states'][torch.tensor([original_index[fid] for fid in wanted])]), 'Global states differ from original native cache')
    del original;index={fid:i for i,fid in enumerate(ids)}
    for pair in pairs:
        a,b=(scenes[sid] for sid in pair['sids'])
        need([a['n_frames'],b['n_frames']]==[8,16] and a['target_ids']==b['target_ids'] and a['question']==b['question']
             and all(torch.equal(states[index[x]],states[index[y]]) for x,y in zip(a['global_feature_ids'],b['global_feature_ids'])), 'Paired frozen global states differ')
    bind(plan['code_inventory_file'],plan['code_inventory_sha256']);inventory=read(plan['code_inventory_file'])
    bind(plan['codes_file'],plan['codes_sha256']);packet=torch.load(plan['codes_file'],map_location='cpu',weights_only=True)
    expected=[];code_rows=[];offsets=[0]
    for row in rows:
        sid=row['sid'];sample=samples[sid];scene=scenes[sid];directory=Path(sample['path']);qa_file=directory/'qa.txt'
        bind(qa_file,sample['qa_sha256']);need(sample['qa_sha256']==scene['qa_sha256'] and sample['path']==scene['path'],'QA path/hash ownership differs')
        lines=qa_file.read_text().splitlines();need(lines[0]=='question:' and lines[-2]=='answer:' and lines[-3]==row['question']
             and lines[-1]==row['gold'] and len(lines)==row['n_frames']+4,'QA record layout differs')
        semantic=[ast.literal_eval(line) for line in lines[1:-3]]
        content=object_sha(dict(states=semantic,question=row['question']))
        need(content==sample['content_sha256']==scene['content_sha256'] and sid=='ij_'+content[:24], 'Canonical semantic content/SID differs')
        match=re.fullmatch(r"Consider only images in the (.+) or the (.+)\. Which person appears in both rooms\? Reply with the person's name only\.",row['question'])
        need(match is not None,'Unrecognized original question');requested=list(match.groups())
        need(requested==sample['room_pair'] and len(sample['image_files'])==len(semantic)==row['n_frames'],'Requested-room/image order differs')
        frames=[]
        for i,(state,image) in enumerate(zip(semantic,sample['image_files'])):
            occupants=[(person,room) for room,people in state['rooms'].items() for person in people]
            need(len(occupants)==1 and state['step_id']==i+1,'Per-image semantics must contain exactly one ordered occupant')
            person,room=occupants[0];need(room in p.ROOMS and set(state)=={'step_id','rooms'} and len(state['rooms'])==1,'Unknown/noncanonical room state');value,column=local_code(person,room,list(p.PEOPLE),requested)
            image_file=directory/f'{i:03d}.png';need(image['path']==str(image_file) and image['dimensions']==[512,512] and image['mode']=='RGB','Image occurrence position/format changed');bind(image_file,image['sha256'])
            frames.append(dict(local_index=i,step_id=i+1,person=person,room=room,image_file=str(image_file),image_sha256=image['sha256'],code_index=column));code_rows.append(value)
        start=offsets[-1];offsets.append(start+len(frames))
        expected.append(dict(sid=sid,question=row['question'],room_pair=requested,n_frames=row['n_frames'],qa_file=str(qa_file),qa_sha256=sample['qa_sha256'],
            content_sha256=content,code_start=start,code_stop=offsets[-1],frames=frames))
    need(inventory==dict(schema_version=1,people=list(p.PEOPLE),rank=96,active_coordinates=18,scenes=expected,
         local_mapping_only=True,no_answer_or_intersection_code=True,no_target_dependent_code=True),'Code inventory contains wrong semantics/ownership or extra answer fields')
    codes=torch.tensor(code_rows,dtype=torch.float32)
    need(set(packet)=={'schema_version','sids','scene_offsets','codes'} and len(code_rows)==1296 and packet['schema_version']==1 and packet['sids']==[r['sid'] for r in rows]
         and packet['scene_offsets']==offsets and packet['codes'].dtype==torch.float32 and torch.equal(packet['codes'],codes)
         and p.tensor_info(codes)==plan['code_tensor'], 'Actual fixed per-image onehot codes differ')
    by_sid={r['sid']:codes[offsets[i]:offsets[i+1]] for i,r in enumerate(rows)}
    bind(parent['initial_file'],parent['initial_sha256']);old_initial=torch.load(parent['initial_file'],map_location='cpu',weights_only=True)
    bind(plan['initial_file'],plan['initial_sha256']);initial_packet=torch.load(plan['initial_file'],map_location='cpu',weights_only=True)
    need(set(initial_packet)=={'seed','branch','parent_initial_file','parent_initial_sha256'}
         and initial_packet['parent_initial_file']==plan['parent_initial_file']==parent['initial_file']
         and initial_packet['parent_initial_sha256']==plan['parent_initial_sha256']==parent['initial_sha256'],'Unfitted selected-parent checkpoint provenance differs')
    initial=initial_packet['branch'];check_weights(torch,initial,initial)
    need(initial_packet['seed']==old_initial['seed']==24 and set(initial)==set(WEIGHT_NAMES)
         and all(torch.equal(initial[k],old_initial['branch'][k]) for k in WEIGHT_NAMES)
         and {k:p.tensor_info(v) for k,v in initial.items()}==plan['initial_state'] and bool((initial['up.weight']==0).all()), 'Selected unfitted parent readout tensors differ')
    return rows,scenes,order,states,index,by_sid,initial


def gather(torch,scenes,states,index,by_sid,sids,first_only=False):
    sequences=[scenes[sid]['target_ids'] for sid in sids];full=layout_for(sequences)
    counts=[1 if first_only else len(ids) for ids in sequences];n=max(scenes[sid]['n_frames'] for sid in sids);q=sum(counts)
    codes=torch.zeros((n,q,96),dtype=torch.float32);valid=torch.zeros((n,q),dtype=torch.bool);global_ids=[];offset=0
    for sid,count in zip(sids,counts):
        values=by_sid[sid];codes[:len(values),offset:offset+count]=values[:,None,:].expand(-1,count,-1);valid[:len(values),offset:offset+count]=True
        global_ids.extend(scenes[sid]['global_feature_ids'][:count]);offset+=count
    g=states[torch.tensor([index[fid] for fid in global_ids])]
    if first_only:layout=dict(first_query_only=True,target_ids=[ids[0] for ids in sequences],full_target_ids=sequences,first_indices=full['offsets'][:-1],original_full_layout=full)
    else:layout=full
    return codes,g,valid,layout


def functional_core(torch,codes,g,valid,w):
    f=torch.nn.functional;q=f.linear(rms(torch,g),w['query.weight']);gates=valid.float()*.5;messages=codes*gates.unsqueeze(-1);aggregate=messages.sum(0)
    pre=f.linear(aggregate,w['aggregate_projection.weight'],w['aggregate_projection.bias'])+q;delta=f.linear(f.silu(pre),w['up.weight'])
    return dict(query=q,payload=codes,scores=torch.full(valid.shape,.5),gates=gates,messages=messages,aggregate=aggregate,preactivation=pre,delta=delta)


def audit_training(directory,summary,scenes,order,denominators):
    bind(summary['training_file'],summary['training_sha256']);logs=read(summary['training_file']);need(len(logs)==600,'Update coverage differs')
    total=0;maximum=0;expected_calls=[]
    for step,row in enumerate(logs,1):
        batch=order[(step-1)*8:step*8];sids=[sid for pair in batch for sid in pair['sids']]
        sequences=[scenes[sid]['target_ids'] for sid in sids];layout=layout_for(sequences);lengths=list(map(len,sequences));offsets=layout['offsets']
        rate=.001*step/50 if step<=50 else .00001+(.001-.00001)*(1+math.cos(math.pi*(step-50)/550))/2
        need(row['step']==step and row['sids']==sids and row['pair_ids']==[r['pair_id'] for r in batch]
             and row['cycles']==[r['cycle'] for r in batch] and row['target_ids']==layout['targets']
             and row['prefix_ids']==[prefix for group in layout['prefixes'] for prefix in group]
             and close(row['lr'],rate,atol=1e-12,rtol=1e-12),'Training schedule or target ownership differs')
        need(row['scene_lengths']==lengths and row['scene_offsets']==offsets and row['pair_lengths']==lengths[::2], 'Scene/prefix segmentation differs')
        ce=row['position_losses']['ce'];reg=row['position_losses']['consistency']
        need(len(ce)==len(layout['targets']) and len(reg)==len(layout['left']) and all(math.isfinite(x) and x>=0 for x in ce+reg), 'Position losses invalid')
        means=[sum(ce[a:b])/(b-a) for a,b in zip(offsets,offsets[1:])];regularizers=[];cursor=0
        for length in lengths[::2]:regularizers.append(sum(reg[cursor:cursor+length])/length);cursor+=length
        need(len(row['per_scene_ce'])==16 and len(row['per_pair_consistency'])==8
             and all(close(a,b) for a,b in zip(means,row['per_scene_ce'])) and all(close(a,b) for a,b in zip(regularizers,row['per_pair_consistency']))
             and close(sum(means)/16,row['ce_loss']) and close(sum(regularizers)/8,row['consistency_loss'])
             and row['consistency_coefficient']==0. and row['weighted_consistency_loss']==0. and close(row['loss'],row['ce_loss']), 'Full-name mean-scene CE objective differs')
        den=[denominators[fid] for sid in sids[::2] for fid in scenes[sid]['global_feature_ids']]
        need(len(den)==len(reg)==len(row['pair_position_denominator'])==len(row['residual_difference_norms'])
             and all(close(a,b,atol=1e-4,rtol=1e-6) for a,b in zip(den,row['pair_position_denominator']))
             and all(close(a*a/b,c) for a,b,c in zip(row['residual_difference_norms'],row['pair_position_denominator'],reg)), 'Frozen global denominator/residual diagnostic differs')
        valid=[scenes[sid]['n_frames'] for sid,ids in zip(sids,sequences) for _ in ids]
        need(row['valid_item_count_by_position']==valid and row['valid_gate_count']==sum(valid)
             and row['gate_sum_by_position']==[n*.5 for n in valid] and row['gate_min']==row['gate_mean']==row['gate_max']==.5
             and row['gate_zero_count']==row['gate_one_count']==0 and row['padding_messages_exact_zero'] is True
             and row['closed_message_coordinates_exact_zero'] is True and 0<=row['payload_max_abs']<=1, 'Fixed uniform gate or padding differs')
        need(all(row[k] is True for k in ('native_frozen','codes_frozen'))
             and row['code_input_shape']==[16,len(layout['targets']),96]
             and math.isfinite(row['gradient_norm']) and row['gradient_norm']>=0 and row['clipped']==(row['gradient_norm']>1)
             and math.isfinite(row['seconds']) and row['seconds']>0, 'Gradient/timing metadata invalid')
        q=len(layout['targets']);total+=q;maximum=max(maximum,q)
        for module in ('norm','head'):
            expected_calls.append(dict(module=module,phase='training',step=step,input_shape=[1,q,3584],
                output_shape=[1,q,3584 if module=='norm' else 152064],input_dtype='torch.float16',output_dtype='torch.float16'))
    need(total==21334 and maximum==44 and logs[0]['consistency_loss']==0, 'Full training call-row inventory differs')
    for batch in range(1,8):
        q=12 if batch==7 else 16
        for module in ('norm','head'):
            expected_calls.append(dict(module=module,phase='evaluation',step=batch,input_shape=[1,q,3584],
                output_shape=[1,q,3584 if module=='norm' else 152064],input_dtype='torch.float16',output_dtype='torch.float16'))
    return logs,expected_calls,dict(passed=True,updates=600,pair_presentations=4800,scene_presentations=9600,
        training_target_positions=total,maximum_training_positions_per_call=maximum,final_first_query_positions=108,total_head_rows=21442,
        terminal_ce=logs[-1]['ce_loss'],terminal_residual_diagnostic=logs[-1]['consistency_loss'],consistency_coefficient=0.,
        full_sequence_loss_reductions_verified=True,no_independent_optimizer_trajectory_replay=True)


def audit_capture(torch,out,record,plan,scenes,states,index,by_sid,rows,initial,final,logs,norm,head):
    p=driver();phase=record['phase'];step=record['step'];training=phase=='training'
    need(phase in ('training','evaluation') and record['arm']==p.ARM,'Invalid joint-code capture phase/arm')
    bind(record['file'],record['sha256']);bind(record['weights_file'],record['weights_sha256'])
    cap=torch.load(record['file'],map_location='cpu',weights_only=True);packet=torch.load(record['weights_file'],map_location='cpu',weights_only=True);w=packet['branch']
    check_weights(torch,w,initial)
    if training:
        need(packet['arm']==p.ARM and step in DIAGNOSTIC_STEPS and packet['step']==record['weights_step']==step-1
             and packet['optimizer_step']==step and packet['position']=='before_update' and packet['source_sha256']==plan['source_sha256'], 'Pre-update weight identity differs')
        sids=logs[step-1]['sids']
        if step==1:need(all(torch.equal(w[k],initial[k]) for k in initial),'First captured state differs from selected unfitted initialization')
    else:
        need(1<=step<=7 and packet['step']==record['weights_step']==600 and all(torch.equal(w[k],final[k]) for k in final),'Final capture changed fixed step600 endpoint')
        sids=[r['sid'] for r in rows[(step-1)*16:step*16]]
    need(cap['schema_version']==1 and cap['arm']==p.ARM and cap['phase']==phase and cap['step']==step
         and cap['sids']==record['sids']==sids, 'Capture scene/phase ownership differs')
    codes,g,valid,layout=gather(torch,scenes,states,index,by_sid,sids,first_only=not training)
    need(cap['layout']==layout,'Full-target or empty-prefix capture layout differs')
    for key,value in (('codes',codes),('global_states',g),('valid_mask',valid)):
        need(cap[key].dtype==value.dtype and torch.equal(cap[key],value),'Captured fixed code/global ownership differs: '+key)
    observed=cap['capture'];reference=functional_core(torch,codes,g,valid,w)
    need(set(observed)==set(reference),'Readout capture coverage differs')
    metrics={key:tensor_error(torch,observed[key],value) for key,value in reference.items()}
    need(torch.equal(observed['payload'],codes) and torch.equal(observed['gates'],valid.float()*.5)
         and bool((observed['scores']==.5).all()) and torch.equal(observed['messages'],codes*observed['gates'].unsqueeze(-1))
         and torch.equal(observed['aggregate'],.5*codes.sum(0)) and bool((codes[~valid]==0).all()), 'Raw code/half-SUM/padding arithmetic differs')
    need(observed['delta'].dtype==torch.float32 and cap['fused_global'].dtype==cap['normalized'].dtype==cap['logits'].dtype==torch.float16
         and cap['fused_global'].shape==cap['normalized'].shape==g.shape and cap['logits'].shape==(len(g),152064)
         and all(bool(torch.isfinite(cap[k]).all()) for k in ('fused_global','normalized','logits')), 'Native output dtype/shape differs')
    need(torch.equal(g+observed['delta'].half(),cap['fused_global'])
         and {k:p.tensor_info(cap[k]) for k in ('fused_global','normalized','logits')}==record['tensors'],'Native cast-before-add/tensor hash differs')
    if training and step==1:need(bool((observed['delta']==0).all()) and torch.equal(cap['fused_global'],g),'Zero-up identity differs')
    native_metrics=[];normalized_error=None
    if not training:
        normalized=norm(cap['fused_global'].unsqueeze(0))[0];replayed=head(normalized.unsqueeze(0))[0]
        native_metrics=replay_metric(torch,cap['logits'],replayed);normalized_error=float((normalized.float()-cap['normalized'].float()).abs().max())
    targets=layout['targets'] if training else layout['target_ids'];nll=fp64_nll(torch,cap['logits'],targets)
    precision=[dict(position=i,fp64=float(a),gpu_fp32=b,passed=close(float(a),b)) for i,(a,b) in
        enumerate(zip(nll,logs[step-1]['position_losses']['ce']))] if training else []
    audit=dict(passed=all(v['passed'] for v in metrics.values()) and all(v['passed'] for v in native_metrics+precision),
        arm=p.ARM,phase=phase,step=step,sids=sids,capture_file=record['file'],capture_sha256=record['sha256'],
        weights_file=record['weights_file'],weights_sha256=record['weights_sha256'],functional_core=metrics,
        fixed_local_codes_exact=True,fp16_cast_before_add_exact=True,native_head=native_metrics,normalized_max_absolute=normalized_error,
        nll=precision,captured_rows=len(g),head_rows=0 if training else len(g),cpu_norm_calls=0 if training else 1,cpu_head_calls=0 if training else 1)
    save(out/f'{phase}_{step:04d}_capture_audit.json',audit);need(audit['passed'],'Captured readout/native/NLL check failed; raw evidence retained')
    return audit,cap['logits']


def self_test(torch):
    people=[f'p{i}' for i in range(9)];rooms=['r0','r1'];a,col=local_code('p2','r1',people,rooms)
    zero,none=local_code('p2','outside',people,rooms)
    need(col==5 and sum(a)==1 and a[5]==1 and not any(a[18:]) and none is None and not any(zero),'Raw local coding fixture failed')
    swapped,_=local_code('p2','r1',people,list(reversed(rooms)));need(swapped[4]==1,'Requested-room order fixture failed')
    scenes={'a':dict(n_frames=1,target_ids=[3,4,151645],global_feature_ids=['g0','g1','g2']),
            'b':dict(n_frames=2,target_ids=[3,4,151645],global_feature_ids=['g0','g1','g2'])}
    states=torch.tensor([[1.,2.],[2.,1.],[1.,1.]],dtype=torch.float16);index={'g0':0,'g1':1,'g2':2}
    by_sid={'a':torch.tensor([a]),'b':torch.tensor([a,zero])}
    codes,g,valid,layout=gather(torch,scenes,states,index,by_sid,['a','b'])
    need(torch.equal(codes[0],torch.tensor(a).expand(6,96)) and bool((codes[1]==0).all())
         and valid[1].tolist()==[False,False,False,True,True,True], 'Prefix broadcast/ragged padding fixture failed')
    w={'query.weight':torch.zeros(96,2),'aggregate_projection.weight':torch.eye(96),
       'aggregate_projection.bias':torch.zeros(96),'up.weight':torch.zeros(2,96)}
    cap=functional_core(torch,codes,g,valid,w)
    need(torch.equal(cap['aggregate'],.5*codes.sum(0)) and cap['gates'][1,3]==.5 and cap['payload'][1,3].sum()==0
         and bool((cap['delta']==0).all()) and torch.equal(g+cap['delta'].half(),g),'Zero-payload valid item/zero-up fixture failed')
    short=layout_for([[1,151645],[1,2,151645]],paired=False)
    need(sum(short['scene_weights'][:2])==sum(short['scene_weights'][2:])==.5,'Short/long name scene weighting differs')
    rows=[dict(sid=f'{f}_{i}_{n}',contrast_id=f,variant=i,n_frames=n,first_token_correct=True) for f in range(18) for i in range(3) for n in (8,16)]
    for j in (0,6,12):rows[j]['first_token_correct']=False
    need(not first_token_criterion(rows)['passed'],'Family screen fixture failed')
    return dict(passed=True,groups=5,local_raw_semantics=True,all_prefix_ragged_broadcast=True,
        empty_payload_is_not_closed_gate=True,short_long_equal_scene_CE=True,complete_family_screen=True)

def training_loss_descriptions(logs,scenes):
    roles=defaultdict(list);cycles=defaultdict(lambda:dict(pair_presentations=0,scene_ce=[]))
    for row in logs:
        cursor=0
        for i,sid in enumerate(row['sids']):
            length=len(scenes[sid]['target_ids']);values=row['position_losses']['ce'][cursor:cursor+length];cursor+=length
            roles['first_token'].append(values[0]);roles['EOS'].append(values[-1]);roles['name_continuation'].extend(values[1:-1])
            cycles[row['cycles'][i//2]]['scene_ce'].append(sum(values)/length)
            if i%2==0:cycles[row['cycles'][i//2]]['pair_presentations']+=1
    return dict(descriptive_only=True,token_roles=[dict(labels=dict(role=k),metrics=dict(positions=len(v),mean_nll=sum(v)/len(v))) for k,v in sorted(roles.items())],
        cycles=[dict(cycle=k,pair_presentations=v['pair_presentations'],complete_pair_cycle=v['pair_presentations']==54,
            mean_scene_ce=sum(v['scene_ce'])/len(v['scene_ce'])) for k,v in sorted(cycles.items())],no_checkpoint_selection=True)


def audit_run(torch,out,directory,plan_path,plan,frozen,rows,scenes,order,states,index,by_sid,initial,norm,head):
    p=driver();directory=Path(directory).resolve();config=read(directory/'config.json');summary=read(directory/'summary.json')
    need(directory.parent==p.OUT and directory.name==config['run_id']==f'run_{config["slurm_job_id"]}' and config['arm']==p.ARM
         and config['protocol']==p.PROTOCOL and config['policy']==p.POLICY and config['seed']==24
         and config['source_sha256']==frozen==plan['source_sha256'] and all(summary.get(k)==v for k,v in config.items())
         and Path(config['plan_file']).resolve()==Path(plan_path).resolve() and config['plan_sha256']==sha(plan_path),'Run/config/source/plan join differs')
    for name,digest in frozen.items():bind(directory/'source'/name.replace('/','_'),digest)
    inherited=dict(plan_file=plan['factor_plan']['file'],plan_sha256=plan['factor_plan']['sha256'],source_sha256=plan['inherited_source_sha256'])
    need(config['inherited_source_sha256']==plan['inherited_source_sha256'] and read(directory/'inherited_sources.json')==inherited,'Actual inherited source descriptor differs')
    for flag in ('passed','completed','computational_integrity_passed','checkpoint_roundtrip_passed','native_weights_unchanged',
                 'codes_frozen','cached_first_query_only','no_pretrained_backbone_loaded','no_dev_or_test','no_native_or_whole_answer_claim',
                 'oracle_local_semantics','no_answer_or_intersection_code'):
        need(summary[flag] is True,'Run invariant failed: '+flag)
    need(summary['phase']=='run' and summary['steps']==600 and summary['training_head_rows']==21334
         and summary['final_head_rows']==108 and summary['total_head_rows']==21442 and not any(k.startswith('permutation_') for k in summary), 'Fixed endpoint/call inventory differs')
    need(config['hardware']['gpu']=='NVIDIA B200' and config['hardware']['matmul_allow_tf32'] is False
         and config['hardware']['torch_version']==str(torch.__version__), 'Native hardware/precision differs')
    for key in ('native_identity','native_identity_sha256','native_module_identity','native_model_file','native_model_sha256',
                'factor_plan','uniform_plan','architecture_stop_report','codes_file','codes_sha256','code_tensor','code_inventory_sha256',
                'features_file','features_sha256','feature_tensor','parent_initial_file','parent_initial_sha256'):
        need(config[key]==plan[key], 'Fixed input/native/semantic binding differs: '+key)
    bind(config['code_inventory_file'],config['code_inventory_sha256'])
    need(Path(config['code_inventory_file'])==directory/'code_inventory.json' and read(config['code_inventory_file'])==read(plan['code_inventory_file']),'Copied code inventory differs')
    need(config['reused_unfitted_readout'] is plan['reused_unfitted_readout'] is True
         and config['initialized']==plan['initial_state'] and config['initial_checkpoint_sha256']==plan['initial_sha256'], 'Selected unfitted initialization differs')
    mask=dict(trainable_parameter_names=list(WEIGHT_NAMES),effective_trainable_parameters=697440,retained_state_parameters=697440)
    need(plan['trainability_mask']==mask and all(config[k]==v for k,v in mask.items()),'Exactly four trainable readout tensors required')
    for key in ('training','captures','raw','predictions','calls','endpoint','roundtrip'):bind(summary[key+'_file'],summary[key+'_sha256'])
    for key in ('rows','presentations'):bind(config[key+'_file'],config[key+'_sha256'])
    need(read(config['rows_file'])==rows and read(config['presentations_file'])==order and config['order_sha256']==object_sha(order),'Copied rows/presentations differ')
    checkpoint=Path(summary['checkpoint']);bind(checkpoint,summary['checkpoint_sha256']);bind(config['initial_checkpoint'],config['initial_checkpoint_sha256'])
    need(checkpoint==p.CKPT/directory.name/'final.pt' and Path(config['initial_checkpoint'])==checkpoint.parent/'initial.pt'
         and Path(config['checkpoint_directory'])==checkpoint.parent and Path(config['data_directory'])==p.DATA/directory.name,'Authorized artifact roots differ')
    selected=torch.load(config['initial_checkpoint'],map_location='cpu',weights_only=True)
    need(selected['seed']==24 and all(torch.equal(selected['branch'][k],initial[k]) for k in WEIGHT_NAMES),'Run actual selected initial tensors differ')
    packet=torch.load(checkpoint,map_location='cpu',weights_only=True)
    need(set(packet)=={'branch','step','config'} and packet['step']==600 and packet['config']==config,'Fixed final checkpoint metadata differs')
    final=packet['branch'];check_weights(torch,final,initial);table={k:p.tensor_info(v) for k,v in final.items()}
    endpoint=read(summary['endpoint_file']);roundtrip=read(summary['roundtrip_file']);native_table=dict(norm=plan['native_identity']['norm_weight'],head=plan['native_identity']['head_weight'])
    need(endpoint['passed'] is endpoint['native_weights_unchanged'] is endpoint['oracle_local_semantics'] is endpoint['no_answer_or_intersection_code'] is True
         and endpoint['before_evaluation']==endpoint['after_evaluation']==table
         and endpoint['parameter_sha256']==summary['selected_parameter_sha256']==object_sha(table)
         and endpoint['native_before']==endpoint['native_after']==config['native_weight_identity']==native_table,'Actual frozen final endpoint differs')
    need(endpoint['input_tensors_before']==endpoint['input_tensors_after']==dict(codes=plan['code_tensor'],globals=plan['feature_tensor'])
         and endpoint['input_versions_unchanged'] is summary['input_versions_unchanged'] is endpoint['codes_frozen'] is True,'Fixed deployed code/global table changed')
    need(roundtrip['passed'] is True and roundtrip['step']==600 and roundtrip['initial']==plan['initial_state']
         and roundtrip['final']==roundtrip['reloaded']==table and roundtrip['config_sha256']==sha(directory/'config.json'),'Actual reset/restricted reload differs')
    for value in (endpoint,roundtrip):need(value['checkpoint']==str(checkpoint) and value['checkpoint_sha256']==summary['checkpoint_sha256'],'Endpoint checkpoint identity differs')
    denominators={fid:float(states[index[fid]].float().square().sum()+1e-6) for scene in scenes.values() for fid in scene['global_feature_ids']}
    logs,expected_calls,training=audit_training(directory,summary,scenes,order,denominators)
    need([len(r['target_ids']) for r in logs]==plan['head_rows_by_update'] and close(summary['training_seconds'],sum(r['seconds'] for r in logs)), 'Training position/timing totals differ')
    counters=dict(core=607,norm=607,head=607,vlm=0,vision=0);calls=read(summary['calls_file'])
    need(calls==expected_calls and summary['counters']==endpoint['counters']==counters
         and read(directory/'final_counters.json')==dict(counters=counters,recorded_calls=calls,partial_outputs_retained=True),'Actual native work inventory differs')
    records=read(summary['captures_file']);expected=[('training',step) for step in DIAGNOSTIC_STEPS]+[('evaluation',step) for step in range(1,8)]
    need([(r['phase'],r['step']) for r in records]==expected and all(r['arm']==p.ARM for r in records),'Exactly13 captured batch identities required')
    need(sorted(f.name for f in checkpoint.parent.glob('*.pt'))==sorted(['initial.pt','final.pt']+[f'training_core_{step:04d}.pt' for step in DIAGNOSTIC_STEPS]),'Checkpoint inventory differs')
    need(sorted(f.name for f in directory.glob('progress_*.json'))==[f'progress_{step:04d}.json' for step in range(100,601,100)],'Progress snapshot coverage differs')
    for step in range(100,601,100):
        need(read(directory/f'progress_{step:04d}.json')==dict(step=step,training=logs[:step],captures=[r for r in records if r['phase']=='training' and r['step']<=step],
            counters=dict(core=step,norm=step,head=step,vlm=0,vision=0)), 'Progress snapshot prefix differs')
    raw=torch.load(summary['raw_file'],map_location='cpu',weights_only=True);logits=raw['logits']
    need(raw['schema_version']==1 and raw['sids']==[r['sid'] for r in rows] and logits.dtype==torch.float16 and logits.shape==(108,152064)
         and bool(torch.isfinite(logits).all()) and p.tensor_info(logits)==summary['raw_tensor']
         and raw['batches']==summary['raw_batches'] and len(raw['batches'])==7
         and Path(summary['raw_file'])==Path(config['data_directory'])/'final_logits.pt','Final raw output identity differs')
    audits=[];gpu_nll=[];cursor=0
    with torch.no_grad():
        for record in records:
            phase,step=record['phase'],record['step'];weight_file=checkpoint.parent/f'training_core_{step:04d}.pt' if phase=='training' else checkpoint
            need(Path(record['weights_file'])==weight_file and Path(record['file'])==Path(config['data_directory'])/f'{phase}_capture_{step:04d}.pt','Capture artifact roots differ')
            audit,actual=audit_capture(torch,out,record,plan,scenes,states,index,by_sid,rows,initial,final,logs,norm,head);audits.append(audit)
            if phase=='evaluation':
                batch=raw['batches'][step-1];count=len(record['sids'])
                need(batch['capture_file']==record['file'] and batch['capture_sha256']==record['sha256'] and batch['sids']==record['sids']
                     and batch['logits']==record['tensors']['logits'] and len(batch['gpu_fp32_nll'])==count
                     and torch.equal(actual,logits[cursor:cursor+count]), 'Final batch raw/capture ownership differs')
                cursor+=count;gpu_nll+=batch['gpu_fp32_nll']
    need(cursor==108,'All final captured rows required')
    ids=logits.argmax(-1).tolist();nll=fp64_nll(torch,logits,[r['first_token_id'] for r in rows]);saved=read(summary['predictions_file'])
    rescored=[dict(r,argmax_id=ids[i],first_token_correct=ids[i]==r['first_token_id'],nll=float(nll[i]),gpu_fp32_nll=gpu_nll[i]) for i,r in enumerate(rows)]
    precision=[dict(sid=r['sid'],fp64_nll=r['nll'],gpu_fp32_nll=r['gpu_fp32_nll'],absolute_difference=abs(r['nll']-r['gpu_fp32_nll']),
        passed=close(r['nll'],r['gpu_fp32_nll'])) for r in rescored]
    save(out/'outcomes.json',rescored);save(out/'nll_precision.json',precision)
    need(len(saved)==108 and all(set(a)==set(b) and all(a[k]==b[k] for k in a if k!='nll') and close(a['nll'],b['nll']) for a,b in zip(rescored,saved))
         and all(r['passed'] for r in precision), 'Independent argmax/NLL/metadata rescore differs')
    criterion=first_token_criterion(rescored);need(criterion==summary['first_token_fit'],'Unchanged cached first-token screen differs')
    strata=[]
    for key in ('gold','question','n_frames','contrast_id'):
        groups=defaultdict(list)
        for row in rescored:groups[row[key]].append(row)
        for label,group in sorted(groups.items(),key=lambda x:str(x[0])):
            strata.append(dict(labels={key:label},metrics=dict(correct=sum(r['first_token_correct'] for r in group),contexts=len(group),
                mean_nll=sum(r['nll'] for r in group)/len(group),all_correct=all(r['first_token_correct'] for r in group))))
    save(out/'strata.json',strata);description=training_loss_descriptions(logs,scenes);save(out/'training_loss_descriptions.json',description)
    return dict(passed=True,completed=True,arm=p.ARM,slurm_job_id=config['slurm_job_id'],run_directory=str(directory),run_summary_sha256=sha(directory/'summary.json'),
        checkpoint=str(checkpoint),checkpoint_sha256=summary['checkpoint_sha256'],selected_parameter_sha256=object_sha(table),
        first_token_fit=criterion,mean_first_token_nll=sum(r['nll'] for r in rescored)/108,training=training,counters=counters,capture_audits=audits,
        cpu_norm_calls=7,cpu_head_calls=7,cpu_head_rows=sum(r['head_rows'] for r in audits),
        maximum_native_replay_tv=max(m['full_vocabulary_tv'] for r in audits for m in r['native_head']),
        outcomes_file=str(out/'outcomes.json'),outcomes_sha256=sha(out/'outcomes.json'),strata_file=str(out/'strata.json'),strata_sha256=sha(out/'strata.json'),
        training_loss_descriptions_file=str(out/'training_loss_descriptions.json'),training_loss_descriptions_sha256=sha(out/'training_loss_descriptions.json'))


def resources(out,runs):
    command=['sacct','-X','-u',os.environ['USER'],'--starttime=2026-09-01','--parsable2','--noheader',
        '--format=JobIDRaw,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,Start,End']
    raw=subprocess.run(command,capture_output=True,text=True,check=True).stdout;file=out/'all_user_sacct.psv';file.write_text(raw)
    job_arms={name:arm for arm,name in driver().RUN_JOBS.items()};names=set(job_arms);records=allocation_rows(raw,names)
    need(len(records)==1 and {r['name'] for r in records}==names,'Exactly one joint-code attempt required; failed/zero allocations retained')
    for row in records:
        arm=job_arms[row['name']]
        need(row['job_id']==runs[arm]['slurm_job_id'] and row['partition']=='gpu' and row['state']=='COMPLETED'
             and row['exit_code']=='0:0' and row['gpus']==1 and row['seconds']<=90,'Joint-code allocation identity/cap differs')
    from datetime import datetime
    events=[]
    for row in records:
        a,b=map(datetime.fromisoformat,(row['start'],row['end']));need(b>=a,'Invalid allocation interval')
        if b>a:events.extend(((a,1),(b,-1)))
    current=peak=0
    for _,d in sorted(events):current+=d;need(current>=0,'Invalid allocation overlap');peak=max(peak,current)
    total=sum(r['gpu_seconds'] for r in records);need(current==0 and peak<=1 and total<=90,'Joint-code campaign cap exceeded')
    return dict(passed=True,jobs=records,allocated_gpu_seconds=total,maximum_concurrent_gpus=peak,
        failed_and_zero_allocations_retained=True,file=str(file),sha256=sha(file),command=command)

def report(args,out,frozen):
    import torch
    torch.set_num_threads(4);p=driver();plan=p.verify_plan(args.plan,ancestors=True)
    need(frozen==plan['source_sha256']==p.sources() and len(frozen)==7 and len(args.runs)==1
         and plan['inherited_source_sha256']==p.inherited_sources() and len(plan['inherited_source_sha256'])==99, 'New/inherited source or single-run inventory differs')
    expected_inherited=dict(plan_file=plan['factor_plan']['file'],plan_sha256=plan['factor_plan']['sha256'],source_sha256=plan['inherited_source_sha256'])
    need(read(out/'inherited_sources.json')==expected_inherited,'Independent report inherited source descriptor differs')
    need(plan['reused_unfitted_readout'] is plan['oracle_local_semantics'] is plan['no_answer_or_intersection_code'] is True,'Privileged diagnostic scope differs')
    trigger=plan['architecture_stop_report'];bind(trigger['file'],trigger['sha256']);stopped=read(trigger['file'])
    need(stopped['passed'] is stopped['completed'] is True and stopped['phase']=='report'
         and stopped['first_token_screen']['product']['passed'] is False,'Previous architecture stop must remain recorded')
    bind(stopped['analysis_file'],stopped['analysis_sha256'])
    rows,scenes,order,states,index,by_sid,initial=audit_inputs(torch,plan)
    save(out/'code_reconstruction_audit.json',dict(passed=True,contexts=108,local_occurrences=1296,active_coordinates=18,rank=96,
        code_inventory_file=plan['code_inventory_file'],code_inventory_sha256=plan['code_inventory_sha256'],codes_file=plan['codes_file'],codes_sha256=plan['codes_sha256'],
        semantic_qa_and_image_order_verified=True,code_has_no_answer_intersection_or_prefix_input=True,selected_unfitted_readout_exact=True,
        original_global_cache_subset_exact=True,oracle_local_semantics=True,no_runtime_method_claim=True))
    need(p.oracle.native_module_identity(torch,plan['native_identity'])==plan['native_module_identity'],'Actual installed native norm/head source differs')
    packet=torch.load(plan['native_model_file'],map_location='cpu',weights_only=True)
    norm,head=p.oracle.native_modules(torch,packet,plan['native_identity'],'cpu');del packet
    before=dict(norm=p.tensor_info(norm.weight),head=p.tensor_info(head.weight))
    run=audit_run(torch,out,args.runs[0],args.plan,plan,frozen,rows,scenes,order,states,index,by_sid,initial,norm,head)
    need(before==dict(norm=p.tensor_info(norm.weight),head=p.tensor_info(head.weight))
         and norm.weight.grad is head.weight.grad is None and not norm.weight.requires_grad and not head.weight.requires_grad,'Independent native weights changed')
    need(run['cpu_head_calls']==run['cpu_norm_calls']==7 and run['cpu_head_rows']==108 and len(run['capture_audits'])==13,'Exactly seven final CPU head batches and13 captures required')
    runs={p.ARM:run};budget=resources(out,runs)
    result=dict(passed=True,completed=True,phase='report',protocol=p.PROTOCOL,source_sha256=frozen,inherited_source_sha256=plan['inherited_source_sha256'],
        plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),factor_plan=plan['factor_plan'],uniform_plan=plan['uniform_plan'],
        architecture_stop_report=trigger,runs=runs,resources=budget,first_token_screen={p.ARM:run['first_token_fit']},
        cpu_head_calls=7,cpu_norm_calls=7,cpu_head_rows=108,gpu_head_calls=607,gpu_norm_calls=607,gpu_head_rows=21442,vlm_calls=0,vision_calls=0,
        oracle_local_semantics=True,no_answer_or_intersection_code=True,no_runtime_method_claim=True,cached_first_query_only=True,
        no_native_or_whole_answer_claim=True,no_generalization_or_reasoning_claim=True,no_new_fit_or_model_forward=True,all_outcomes_retained=True)
    save(out/'analysis.json',result);c=run['first_token_fit']
    lines=['# Privileged local joint-code learnability control','',
        'Independent computation and provenance audits passed. Perfect per-image person/requested-room labels were supplied as fixed raw joint codes. '
        'The readout received their half-SUM and cached global state; no answer code or precomputed bag intersection was supplied.','',
        '| Cached training first-token screen | Correct /108 | Complete families /18 | Mean first-token NLL |',
        '|---|---:|---:|---:|',f"| {'PASS' if c['passed'] else 'FAIL'} | {c['correct']} | {c['complete_families']} | {run['mean_first_token_nll']:.9f} |",'',
        'All 108 contexts and 18 complete six-context families remain in the denominator. The unchanged screen requires 103/108 first tokens and 16/18 entirely correct families. '
        'Training used the original 600 full-name-plus-EOS CE updates and four exactly selected unfitted readout tensors; only step 600 was evaluated after reset and restricted reload. '
        'First-token, name-continuation and EOS training losses and complete-cycle trends are descriptive artifacts.','',
        ('This fixed recipe fits the privileged local-code/readout screen. It supports learnability on these training examples with perfect local semantics; it does not show that native visual states supply those codes.' if c['passed'] else
         'This fixed recipe does not fit even the privileged local-code/readout screen. The readout and optimization budget remain unresolved; failure does not prove missing information or impossibility.'),'',
        'This is a privileged training diagnostic, not a vision inference method, whole-answer evaluation, generalization result or reasoning-composition result. '
        'Previous failed screens and architecture stopping rules remain unchanged. No subsequent fit or benchmark evaluation is automatically released.','',
        f"Allocated GPU cost: {budget['allocated_gpu_seconds']} seconds. GPU work: 607 core/norm/head calls each and 21,442 head rows; zero VLM/vision calls. "
        f"CPU native replay: seven final batches, 108 rows, maximum full-vocabulary TV {run['maximum_native_replay_tv']:.9f}, fixed TV <= 0.02 and exact argmax. "
        'All 13 captured readout batches and all 1,296 semantic-code occurrences were independently checked.\n']
    (out/'REPORT.md').write_text('\n'.join(lines))
    return dict(passed=True,completed=True,phase='report',protocol=p.PROTOCOL,source_sha256=frozen,inherited_source_sha256=plan['inherited_source_sha256'],
        analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),first_token_screen=result['first_token_screen'],resources=budget,
        oracle_local_semantics=True,no_answer_or_intersection_code=True,no_runtime_method_claim=True,cached_first_query_only=True,no_native_or_whole_answer_claim=True)


def main():
    import argparse,time
    p=driver();parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--runs',type=Path,nargs=1,required=True);args=parser.parse_args();p.native.require_slurm(gpu=False)
    need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),'CPU-only independent report')
    out=p.OUT/f'report_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    frozen=p.snapshot(out);save(out/'request.json',dict(plan=str(args.plan),runs=list(map(str,args.runs)),source_sha256=frozen))
    try:
        result=report(args,out,frozen);need(p.sources()==frozen and p.inherited_sources()==result['inherited_source_sha256'],'Source changed during report')
        result['elapsed_seconds']=time.perf_counter()-started;save(out/'summary.json',result)
        print(__import__('json').dumps(dict(passed=True,completed=True,directory=str(out))),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,
            elapsed_seconds=time.perf_counter()-started,partial_outputs_retained=True));raise


if __name__=='__main__':main()

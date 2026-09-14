"""Independent matched within-item/cross-item moment-basis training audit.

Audits the full216-scene bank, shared paired-minibatch permutation and
only the fixed6000 endpoint. Native whole-answer inference is a separate gate.
"""
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from collections import Counter,defaultdict
import math
import os
import random
import subprocess
from scripts.stage_native_vision_v6_teacher import need,read,save,sha,object_sha
from scripts import report_native_identity_join_factor_binding as old
from scripts import report_native_identity_join_orientation_paired_visual as previous
orientation=previous
OWN=('scripts/report_native_identity_join_moment_basis.py',)
CORE_CLASS='gnnformer.parallel_local_moment_basis.ParallelLocalMomentBasis'
ARMS=('within','cross')
DIAGNOSTIC_STEPS=(1,2,32,128,300,600,2000,6000)
ENDPOINTS=(6000,)
PROGRESS=(600,2000,4000,6000)
PARAMETER_COUNT=1404192
close=old.close
layout_for=old.layout_for
fp64_nll=old.fp64_nll
tensor_error=old.tensor_error
replay_metric=old.replay_metric
allocation_rows=old.allocation_rows
gather=old.gather
bind=old.bind


def driver():
    from scripts import diagnose_native_identity_join_moment_basis as module
    return module


def first_token_criterion(rows):
    need(len(rows)==len({r['sid'] for r in rows})==216,'All216 unique outcomes required')
    orientations=('original','flipped');counts={}
    for o in orientations:
        selected=[r for r in rows if r['orientation_version']==o]
        need(len(selected)==108,'Each orientation requires108 outcomes')
        counts[o]=sum(bool(r['first_token_correct']) for r in selected)
    families=defaultdict(list)
    for r in rows:families[r['base_contrast_id']].append(r)
    expected={(v,n,o) for v in range(3) for n in (8,16) for o in orientations}
    need(len(families)==18 and all(len(group)==12 and {(r['variant'],r['n_frames'],r['orientation_version']) for r in group}==expected
                                 for group in families.values()),'Every family needs its twelve distinct contexts')
    complete=sum(all(r['first_token_correct'] for r in group) for group in families.values());total=sum(counts.values())
    return dict(passed=total>=206 and all(v>=103 for v in counts.values()) and complete>=16,first_correct=total,contexts=216,
        orientation_correct=counts,complete_families=complete,families=18,thresholds=dict(pooled=206,per_orientation=103,complete_families=16),
        cached_first_query_only=True,whole_answer_claim=False)


def learning_rate(step):
    need(type(step) is int and 1<=step<=6000,'Update outside fixed horizon')
    return .001*step/50 if step<=50 else .00001+(.001-.00001)*(1+math.cos(math.pi*(step-50)/5950))/2


def primary_screen(endpoint_rows):
    need(set(endpoint_rows)==set(ENDPOINTS),'Only the fixed6000 endpoint is authorized')
    return first_token_criterion(endpoint_rows[6000])


def check_weights(torch,weights,initial):
    shapes={'local_bias':(192,),'query.weight':(96,3584),'local.weight':(192,3584),
            'aggregate_projection.weight':(96,288),'aggregate_projection.bias':(96,),'up.weight':(3584,96)}
    need(set(weights)==set(initial)==set(shapes) and all(v.dtype==torch.float32 and tuple(v.shape)==shapes[k]
         and bool(torch.isfinite(v).all()) for k,v in weights.items())
         and sum(v.numel() for v in weights.values())==PARAMETER_COUNT,'Six-tensor moment-core table differs')


def functional_core(torch,h,g,valid,mean,scale,w,arm):
    f=torch.nn.functional;rank=w['query.weight'].shape[0]
    original=old.rms(torch,h);conditioned=(original-mean.unsqueeze(0))/scale;q=f.linear(old.rms(torch,g),w['query.weight'])
    local=f.linear(conditioned,w['local.weight']);pa=local[...,:rank]+q.unsqueeze(0)+w['local_bias'][:rank]
    pb=local[...,rank:]+q.unsqueeze(0)+w['local_bias'][rank:];a=torch.tanh(pa);b=torch.tanh(pb)
    need(arm in ARMS,'Unknown moment basis')
    payload=torch.cat((a,b,.5*(a*b)),dim=-1);scores=torch.full(valid.shape,.5,dtype=torch.float32,device=h.device)
    gates=.5*valid.float();messages=gates.unsqueeze(-1)*payload;moments=messages.sum(0);A,B,C=moments.split(rank,dim=-1)
    P=A*B-C;reconstructed=A*B-P;aggregate=torch.cat((A,B,C if arm=='within' else P),dim=-1)
    preactivation=f.linear(aggregate,w['aggregate_projection.weight'],w['aggregate_projection.bias'])+q
    readout=f.silu(preactivation);delta=f.linear(readout,w['up.weight'])
    return dict(original_local_rms_input=original,conditioned_local_rms_input=conditioned,query=q,
        factor_a=a,factor_b=b,factor_preactivation_a=pa,factor_preactivation_b=pb,
        payload=payload,scores=scores,gates=gates,messages=messages,moments=moments,
        moment_a=A,moment_b=B,moment_c=C,moment_pair=P,moment_reconstructed_c=reconstructed,
        moment_inverse_error=reconstructed-C,aggregate=aggregate,preactivation=preactivation,readout=readout,delta=delta)


def ordered_pair_sum(torch,a,b,valid):
    # Direct O(N^2) reference with no diagonal; used only in tiny CPU fixtures.
    result=(a.sum(0)+b.sum(0))*0
    for i in range(len(a)):
        for j in range(len(a)):
            if i!=j:result=result+.25*(valid[i]&valid[j]).unsqueeze(-1)*a[i]*b[j]
    return result


def moment_descriptions(torch,observed,valid):
    # FP64 reconstruction of saved FP32 factors is descriptive, not new fidelity tolerances.
    a=observed['factor_a'].double();b=observed['factor_b'].double();g=.5*valid.double().unsqueeze(-1)
    wa=g*a;wb=g*b;A=wa.sum(0);B=wb.sum(0);C=(wa*wb).sum(0);P=A*B-C
    def stats(x):
        x=x.double();return dict(minimum=float(x.min()),maximum=float(x.max()),mean=float(x.mean()),
            mean_absolute=float(x.abs().mean()),rms=float(x.square().mean().sqrt()))
    denominator=(A*B).abs()+C.abs();defined=denominator>0;cancel=1-P[defined].abs()/denominator[defined]
    return dict(descriptive_only=True,precision='FP64 reconstruction from actual saved FP32 factors; native FP32 discrepancies are descriptive',
        valid_item_counts=valid.sum(0).tolist(),coordinate_count=P.numel(),
        signed_coordinate_statistics={k:stats(observed[k]) for k in ('moment_a','moment_b','moment_c','moment_pair','moment_inverse_error')},
        fp64_reference_statistics={k:stats(v) for k,v in (('A',A),('B',B),('C',C),('P',P))},
        maximum_absolute_fp32_vs_fp64={k:float((observed[k].double()-v).abs().max()) for k,v in
            (('moment_a',A),('moment_b',B),('moment_c',C),('moment_pair',P))},
        maximum_absolute_c_roundtrip_error=float((observed['moment_reconstructed_c']-observed['moment_c']).abs().max()),
        cancellation_fraction=None if not cancel.numel() else stats(cancel),cancellation_defined_coordinates=int(defined.sum()),
        zero_reference_denominators=int((denominator==0).sum()),
        finite_width_function_equivalence_claim=False,no_rescaling_or_extra_head_call=True)


def moment_fixture(torch):
    from gnnformer.parallel_local_moment_basis import ParallelLocalMomentBasis
    checks=[]
    # General signed factors, distinct values and ragged validity; no semantic labels.
    a=torch.tensor([[[.7,-.3],[.2,.9]],[[.4,.8],[-.6,.3]],[[-.5,.1],[.8,-.7]]],dtype=torch.float64,requires_grad=True)
    b=torch.tensor([[[-.2,.6],[.4,-.8]],[[.9,-.4],[.3,.5]],[[.2,.7],[-.1,.6]]],dtype=torch.float64,requires_grad=True)
    valid=torch.tensor([[True,True],[True,False],[True,True]])
    for mask in (valid,torch.zeros_like(valid),torch.tensor([[True,True],[False,False],[False,False]])):
        gates=.5*mask.double().unsqueeze(-1);A=(gates*a).sum(0);B=(gates*b).sum(0);C=(gates.square()*a*b).sum(0)
        P=A*B-C;direct=ordered_pair_sum(torch,a,b,mask)
        need(torch.allclose(P,direct,atol=1e-14,rtol=1e-14),'Ordered cross-image sum algebra differs')
        probe=torch.tensor([[.3,-.7],[.9,.2]],dtype=torch.float64)
        gs=torch.autograd.grad((P*probe).sum(),(a,b),retain_graph=True);gd=torch.autograd.grad((direct*probe).sum(),(a,b),retain_graph=True)
        need(all(torch.allclose(x,y,atol=1e-14,rtol=1e-14) for x,y in zip(gs,gd)), 'Cross-image value equality hid a gradient error')
        need(all(bool((x[~mask]==0).all()) for x in gs),'Padded factor received a cross-moment gradient')
        need(torch.allclose(A*B-P,C,atol=1e-14,rtol=1e-14),'Real-arithmetic inverse fixture differs')
        summaries=[((gates[z]*a[z]).sum(0),(gates[z]*b[z]).sum(0),(gates[z].square()*a[z]*b[z]).sum(0)) for z in (slice(0,1),slice(1,None))]
        (a1,b1,c1),(a2,b2,c2)=summaries
        merged=(a1*b1-c1)+(a2*b2-c2)+a1*b2+a2*b1
        need(torch.allclose(merged,P,atol=1e-14,rtol=1e-14),'Partition merge lost cross-partition pairs')
    empty=a[:0];need(ordered_pair_sum(torch,empty,b[:0],valid[:0]).shape==(2,2)
         and bool((ordered_pair_sum(torch,empty,b[:0],valid[:0])==0).all()),'Empty direct sum differs')
    checks.append(dict(passed=True,signed_ordered_pair_values_and_gradients=True,ragged_padding_and_empty=True,
        singleton_zero_pairs=True,partition_merge_cross_terms=True,real_arithmetic_inverse=True))
    h=torch.arange(24,dtype=torch.float32).sin().reshape(2,3,4).half();g=torch.tensor([[.2,-.3,.4,.1],[.1,.5,-.2,.3],[-.4,.2,.3,.1]],dtype=torch.float16)
    mask=torch.tensor([[True,True,True],[True,False,True]]);cores={};caps={}
    for arm in ARMS:
        with torch.random.fork_rng(devices=[]):torch.manual_seed(24);core=ParallelLocalMomentBasis(4,rank=3,basis=arm)
        cores[arm]=core
        need(core.basis==arm and len(core.state_dict())==6 and all(p.requires_grad for p in core.parameters())
             and sum(p.numel() for p in core.parameters())==4*4*3+3*3*3+3*3 and not list(core.buffers()),'Tiny six-tensor moment parameter scope differs')
        zero,cap=core(h,g,valid_mask=mask,capture=True)
        need(bool((zero==0).all()),'Shared zero-U native residual identity differs')
        with torch.no_grad():core.up.weight.copy_(.07*torch.arange(12,dtype=torch.float32).cos().reshape(4,3))
        up_inputs=[];hook=core.up.register_forward_pre_hook(lambda module,args:up_inputs.append(args[0]))
        try:delta,cap=core(h,g,valid_mask=mask,output_dtype=torch.float32,capture=True)
        finally:hook.remove()
        need(len(up_inputs)==1 and up_inputs[0] is cap['readout'],'Capture must be actual up-projection input')
        caps[arm]=cap
        reference=functional_core(torch,h,g,mask,torch.zeros(3,4),torch.tensor(1.),core.state_dict(),arm)
        reference={k:v for k,v in reference.items() if k not in ('original_local_rms_input','conditioned_local_rms_input')}
        need(set(cap)==set(reference) and all(torch.allclose(cap[k],v,atol=1e-6,rtol=1e-6) for k,v in reference.items()),'Independent six-tensor moment formula differs')
        grads=torch.autograd.grad(delta.square().sum(),tuple(core.parameters()),allow_unused=True)
        need(len(grads)==6 and all(v is not None and bool(torch.isfinite(v).all()) and bool((v!=0).any()) for v in grads),'All six tensors need live gradients')
        checks.append(dict(arm=arm,passed=True,independent_actual_core_formula=True,actual_up_input_identity=True,six_live_gradients=True,query_retained_in_factors_and_readout=True))
    need(all(torch.equal(v,cores['cross'].state_dict()[k]) for k,v in cores['within'].state_dict().items()),'Matched basis initial parameter bytes differ')
    need(all(torch.equal(caps['within'][k],caps['cross'][k]) for k in caps['within'] if k not in ('aggregate','preactivation','readout','delta'))
         and not torch.equal(caps['within']['aggregate'],caps['cross']['aggregate']),'Only the moment-basis aggregate should differ')
    return dict(passed=True,checks=checks,no_pretrained_head_forward=True)


def independent_initialization(torch):
    # Reproduce the prospectively fixed draw sequence independently of the new core factory.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(24)
        query=torch.nn.Linear(3584,96,bias=False,dtype=torch.float32)
        local=torch.nn.Linear(3584,192,bias=False,dtype=torch.float32)
        aggregate=torch.nn.Linear(288,96,bias=True,dtype=torch.float32)
        up=torch.nn.Linear(96,3584,bias=False,dtype=torch.float32)
        return {'local_bias':torch.zeros(192,dtype=torch.float32),'query.weight':query.weight.detach(),
            'local.weight':local.weight.detach(),'aggregate_projection.weight':aggregate.weight.detach(),
            'aggregate_projection.bias':torch.zeros_like(aggregate.bias),'up.weight':torch.zeros_like(up.weight)}


def audit_inputs(torch,plan):
    p=driver();ref=plan['paired_visual_plan'];bind(ref['file'],ref['sha256'])
    need(ref==dict(file=str(p.PARENT),sha256=p.PARENT_SHA),'Exact passed paired-visual data preparation required')
    parent=previous.driver().verify_plan(ref['file'],ancestors=True)
    rows,scenes,order,states,index,stats,_=previous.audit_inputs(torch,parent)
    for key in ('rows_file','scenes_file','pairs_file','stats_inventory_file','order_file','head_rows_file'):
        need(sha(plan[key])==sha(parent[key]),'Original visual input metadata changed: '+key)
    for key in ('features_file','features_sha256','feature_tensor','stats_file','stats_sha256','stats_tensors',
                'native_identity','native_identity_sha256','native_module_identity','native_model_file','native_model_sha256',
                'factor_plan','uniform_plan','oracle_report','conditioning_plan','conditioning_report','cache_report','confirmation_report',
                'orientation_plan','paired_order_stage','negative_evidence','training_stage','order_object_sha256',
                'head_rows_by_update','training_head_rows','total_head_rows'):
        need(plan[key]==parent[key],'Original visual/native/statistic identity differs: '+key)
    bind(plan['initial_file'],plan['initial_sha256']);packet=torch.load(plan['initial_file'],map_location='cpu',weights_only=True)
    need(set(packet)=={'branch','step','seed','initialization','bases','fresh_shared_initialization'}
         and packet['seed']==24 and packet['step']==0 and packet['bases']==list(ARMS) and packet['fresh_shared_initialization'] is True
         and packet['initialization']=='query,local,aggregate,up default Linear draws; zero local/aggregate biases and U',
         'Fresh initial packet schema differs')
    initial=packet['branch'];check_weights(torch,initial,initial);expected=independent_initialization(torch)
    need(all(torch.equal(initial[k],expected[k]) for k in expected)
         and {k:p.tensor_info(v) for k,v in initial.items()}==plan['initial_state'],
         'Independent fresh seed24 query/local/aggregate/up draw sequence differs')
    mask=dict(trainable_parameter_names=list(initial),effective_trainable_parameters=PARAMETER_COUNT,retained_state_parameters=PARAMETER_COUNT)
    need(plan['trainability_mask']==mask and plan['fresh_shared_initialization'] is True
         and plan['reused_unfitted_initialization'] is False and plan['bases']==list(ARMS)
         and plan['query_placement']=='pre_silu' and plan['fixed_weight']==.5 and plan['moment_basis_comparison'] is True
         and plan['paired_minibatch_only'] is False
         and plan['no_automatic_native_or_fresh_release'] is True,'Matched moment-basis/fresh initialization scope differs')
    support=(('paired_visual_report','identity_join_orientation_paired_visual/report_443743','551a03afc57d343653f53f966d79fbcb2957837f16df50a84413ff0fe71775a8',{'product':77,'additive':73}),
             ('post_query_code_report','identity_join_orientation_post_query_code/report_443794','9da20cc3754e734ee62dc55f551c9fe41b58e2947898c9c11d9253d334b0c3f3',{'joint_code':84}),
             ('post_query_visual_report','identity_join_orientation_post_query_visual/report_443795','dd56720c974fcbb44e45b55c9d14472f91d6b122af0522d92e60093ce7ee9489',{'product':84,'additive':74}))
    for key,relative,digest,counts in support:
        evidence=plan[key];need(evidence['file']==str(REPO/'outputs/native_aggregation_vlm'/relative/'summary.json') and evidence['sha256']==digest,'Exact completed negative evidence differs: '+key)
        bind(evidence['file'],evidence['sha256']);bind(evidence['analysis_file'],evidence['analysis_sha256'])
        proof=read(evidence['file']);analysis=read(evidence['analysis_file'])
        need(proof['passed'] is proof['completed'] is analysis['passed'] is analysis['completed'] is analysis['all_numerical_audits_collected'] is True
             and proof['analysis_file']==evidence['analysis_file'] and proof['analysis_sha256']==evidence['analysis_sha256']
             and proof['source_sha256']==analysis['source_sha256'] and proof['inherited_source_sha256']==analysis['inherited_source_sha256']
             and set(analysis['runs'])==set(counts),'Completed bound negative audit differs')
        for field in ('source_sha256','inherited_source_sha256'):
            for name,digest in proof[field].items():
                need(plan['inherited_source_sha256'].get(name)==digest,'Negative evidence source absent from closure: '+name)
        for arm,count in counts.items():
            run=analysis['runs'][arm];endpoint=run['endpoint_audits']['6000'];bind(endpoint['outcomes_file'],endpoint['outcomes_sha256']);outcomes=read(endpoint['outcomes_file'])
            need(run['passed'] is True and len(run['capture_audits'])==22 and run['cpu_head_calls']==14 and run['cpu_head_rows']==216
                 and len(outcomes)==216 and all({k:a[k] for k in b}==b for a,b in zip(outcomes,rows))
                 and all(r['first_token_correct']==(r['argmax_id']==r['first_token_id']) for r in outcomes),'Negative outcome ownership differs')
            criterion=first_token_criterion(outcomes)
            need(criterion==run['first_token_fit']==proof['first_token_screen'][arm] and criterion['first_correct']==count
                 and criterion['complete_families']==0 and criterion['passed'] is False,'Closed post-query negative outcome differs')
    return rows,scenes,order,states,index,stats,initial


def audit_training(directory,summary,scenes,order,denominators):
    bind(summary['training_file'],summary['training_sha256']);logs=read(summary['training_file']);need(len(logs)==6000,'Update coverage differs')
    total=0;maximum=0;expected_calls=[]
    for step,row in enumerate(logs,1):
        batch=order[(step-1)*8:step*8];sids=[sid for pair in batch for sid in pair['sids']]
        sequences=[scenes[sid]['target_ids'] for sid in sids];layout=layout_for(sequences);lengths=list(map(len,sequences));offsets=layout['offsets']
        rate=learning_rate(step)
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
        need(all(row[k] is True for k in ('native_frozen','statistics_frozen','trainable_parameters_finite'))
             and row['conditioned_input_shape']==[16,len(layout['targets']),3584]
             and math.isfinite(row['gradient_norm']) and row['gradient_norm']>=0 and row['clipped']==(row['gradient_norm']>1)
             and math.isfinite(row['seconds']) and row['seconds']>0, 'Gradient/timing metadata invalid')
        q=len(layout['targets']);need(q<=48,'Per-call target row cap exceeded');total+=q;maximum=max(maximum,q)
        for module in ('norm','head'):
            expected_calls.append(dict(module=module,phase='training',step=step,endpoint_step=None,input_shape=[1,q,3584],
                output_shape=[1,q,3584 if module=='norm' else 152064],input_dtype='torch.float16',output_dtype='torch.float16'))
        if step in ENDPOINTS:
            for batch in range(1,15):
                q=8 if batch==14 else 16
                for module in ('norm','head'):
                    expected_calls.append(dict(module=module,phase='evaluation',step=batch,endpoint_step=step,input_shape=[1,q,3584],
                        output_shape=[1,q,3584 if module=='norm' else 152064],input_dtype='torch.float16',output_dtype='torch.float16'))
    independent_total=sum(len(scenes[sid]['target_ids']) for pair in order for sid in pair['sids'])
    need(total==independent_total and maximum==max(len(r['target_ids']) for r in logs) and logs[0]['consistency_loss']==0,
         'Full training call-row inventory differs')
    return logs,expected_calls,dict(passed=True,updates=6000,pair_presentations=48000,scene_presentations=96000,
        training_target_positions=total,maximum_training_positions_per_call=maximum,endpoint_first_query_positions=216,total_head_rows=total+216,
        terminal_ce=logs[-1]['ce_loss'],terminal_residual_diagnostic=logs[-1]['consistency_loss'],consistency_coefficient=0.,
        full_sequence_loss_reductions_verified=True,no_independent_optimizer_trajectory_replay=True)


def audit_capture(torch,out,record,arm,plan,rows,scenes,states,index,stats,initial,endpoints,logs,norm,head):
    p=driver();phase=record['phase'];step=record['step'];training=phase=='training';endpoint_step=record['endpoint_step']
    need(endpoint_step is None if training else endpoint_step in ENDPOINTS,'Invalid capture endpoint association')
    need(phase in ('training','evaluation') and record['arm']==arm and arm in ARMS, 'Invalid capture phase/arm')
    bind(record['file'],record['sha256']);bind(record['weights_file'],record['weights_sha256'])
    cap=torch.load(record['file'],map_location='cpu',weights_only=True);wp=torch.load(record['weights_file'],map_location='cpu',weights_only=True);w=wp['branch']
    check_weights(torch,w,initial)
    if training:
        need(wp['arm']==arm and step in DIAGNOSTIC_STEPS and wp['step']==record['weights_step']==step-1
             and wp['query_placement']=='pre_silu' and wp['basis']==arm and wp['optimizer_step']==step and wp['position']=='before_update' and wp['source_sha256']==plan['source_sha256'], 'Pre-update capture weight binding differs')
        sids=logs[step-1]['sids']
        if step==1:need(all(torch.equal(w[k],initial[k]) for k in initial),'First captured state differs from fresh initialization')
    else:
        need(1<=step<=14 and wp['step']==record['weights_step']==endpoint_step and all(torch.equal(w[k],endpoints[endpoint_step][k]) for k in initial), 'Final capture does not use fixed endpoint')
        sids=[r['sid'] for r in rows[(step-1)*16:step*16]]
    need(cap['schema_version']==1 and cap['arm']==arm and cap['phase']==phase and cap['step']==step
         and cap['sids']==record['sids']==sids and cap['basis']==record['basis']==arm and cap['endpoint_step']==endpoint_step
         and cap['query_placement']==record['query_placement']==plan['query_placement']=='pre_silu', 'Capture scene/phase ownership differs')
    h,g,valid,layout=gather(torch,scenes,states,index,sids,first_only=not training)
    if training:need(cap['layout']==layout,'Training capture full-prefix layout differs')
    else:
        full=layout_for([scenes[sid]['target_ids'] for sid in sids]);expected=dict(first_query_only=True,target_ids=layout['targets'],
            full_target_ids=full['target_sequences'],first_indices=full['offsets'][:-1],original_full_layout=full)
        need(cap['layout']==expected,'Final capture original empty-prefix ownership differs')
    for key,value in (('local_states',h),('global_states',g),('valid_mask',valid)):
        need(cap[key].dtype==value.dtype and torch.equal(cap[key],value),'Actual frozen feature ownership differs: '+key)
    mean=stats['global_mean'].unsqueeze(0).expand(len(g),-1)
    need(cap['question_means'].dtype==torch.float32 and torch.equal(cap['question_means'],mean), 'Global mean not fixed at all prefixes')
    observed=cap['capture'];reference=functional_core(torch,h,g,valid,mean,stats['scale'],w,arm)
    need(set(observed)==set(reference),'Factor capture coverage differs')
    metrics={key:tensor_error(torch,observed[key],reference[key]) for key in reference}
    payload=torch.cat((observed['factor_a'],observed['factor_b'],.5*(observed['factor_a']*observed['factor_b'])),dim=-1)
    A,B,C=observed['moments'].split(96,dim=-1);P=A*B-C
    algebra=dict(payload_exact=torch.equal(observed['payload'],payload),
        fixed_gates_exact=torch.equal(observed['gates'],valid.float()*.5) and bool((observed['scores']==.5).all()),
        messages_exact=torch.equal(observed['messages'],payload*observed['gates'].unsqueeze(-1)),
        padding_zero=bool((observed['messages'][~valid]==0).all()),bounded_factor_payload=bool((payload.abs()<=1).all()),
        moment_views_exact=all(torch.equal(observed[k],v) for k,v in (('moment_a',A),('moment_b',B),('moment_c',C))),
        pair_from_actual_moments_exact=torch.equal(observed['moment_pair'],P),
        inverse_from_actual_moments_exact=torch.equal(observed['moment_reconstructed_c'],A*B-observed['moment_pair']),
        inverse_error_exact=torch.equal(observed['moment_inverse_error'],observed['moment_reconstructed_c']-C),
        basis_aggregate_exact=torch.equal(observed['aggregate'],torch.cat((A,B,C if arm=='within' else observed['moment_pair']),dim=-1)))
    descriptions=moment_descriptions(torch,observed,valid)
    need(observed['delta'].dtype==torch.float32 and cap['fused_global'].dtype==cap['normalized'].dtype==cap['logits'].dtype==torch.float16
         and cap['fused_global'].shape==cap['normalized'].shape==g.shape and cap['logits'].shape==(len(g),152064)
         and all(bool(torch.isfinite(cap[k]).all()) for k in ('fused_global','normalized','logits')), 'Actual native dtype/shape differs')
    need(torch.equal(g+observed['delta'].half(),cap['fused_global'])
         and {k:p.tensor_info(cap[k]) for k in ('fused_global','normalized','logits')}==record['tensors'], 'Native cast-before-add or tensor identities differ')
    if training and step==1:need(bool((observed['delta']==0).all()) and torch.equal(cap['fused_global'],g),'Zero-U identity differs')
    head_metrics=[];normal_error=None;cpu_replay=None
    if not training:
        normalized=norm(cap['fused_global'].unsqueeze(0))[0];replayed=head(normalized.unsqueeze(0))[0]
        replay_file=p.DATA/out.name/f'{arm}_evaluation_{endpoint_step:04d}_batch_{step:04d}_native_replay.pt'
        torch.save(dict(capture_file=record['file'],capture_sha256=record['sha256'],sids=sids,
            normalized=normalized,logits=replayed,native_input_shape=[1,len(g),3584]),replay_file)
        cpu_replay=dict(file=str(replay_file),sha256=sha(replay_file),normalized=p.tensor_info(normalized),logits=p.tensor_info(replayed))
        head_metrics=replay_metric(torch,cap['logits'],replayed);normal_error=float((normalized.float()-cap['normalized'].float()).abs().max())
    nll=fp64_nll(torch,cap['logits'],layout['targets'])
    nll_metrics=[dict(position=i,fp64=float(a),gpu_fp32=b,passed=close(float(a),b)) for i,(a,b) in
        enumerate(zip(nll,logs[step-1]['position_losses']['ce']))] if training else []
    result=dict(arm=arm,phase=phase,step=step,endpoint_step=endpoint_step,sids=sids,capture_file=record['file'],capture_sha256=record['sha256'],
        weights_file=record['weights_file'],weights_sha256=record['weights_sha256'],functional_core=metrics,
        native_head=head_metrics,normalized_max_absolute=normal_error,nll=nll_metrics,cpu_replay=cpu_replay,fp16_cast_before_add_exact=True,
        basis=arm,query_placement='pre_silu',moment_algebra=algebra,moment_descriptions=descriptions,captured_rows=len(g),
        head_rows=0 if training else len(g),cpu_norm_calls=0 if training else 1,cpu_head_calls=0 if training else 1,
        passed=all(algebra.values()) and all(x['passed'] for x in metrics.values()) and all(x['passed'] for x in head_metrics+nll_metrics))
    save(out/(f'{arm}_training_{step:04d}_capture_audit.json' if training else f'{arm}_evaluation_{endpoint_step:04d}_{step:04d}_capture_audit.json'),result)
    return result,cap['logits']



def optimizer_audit(torch,value,step,weights):
    need(set(value)=={'parameter_ids','optimizer_parameter_ids','requires_grad','state_tensor_info','param_group_hyperparameters'},'Optimizer snapshot schema differs')
    names=list(weights);trainable=names;ids=value['parameter_ids']
    need(len(names)==6 and len(trainable)==6 and set(ids)==set(names) and all(type(v) is int and v>0 for v in ids.values())
         and len(set(ids.values()))==6 and value['optimizer_parameter_ids']==[ids[k] for k in trainable]
         and value['requires_grad']=={k:k in trainable for k in names},'Optimizer must retain exactly six live moment parameters')
    need(set(value['state_tensor_info'])==set(trainable),'Adam state must cover all six moment tensors')
    expected_step=driver().tensor_info(torch.tensor(float(step),dtype=torch.float32))
    for name,state in value['state_tensor_info'].items():
        need(set(state)=={'step','exp_avg','exp_avg_sq'} and state['step']==expected_step,'Adam update counter changed or restarted')
        for key in ('exp_avg','exp_avg_sq'):
            info=state[key];need(set(info)=={'shape','dtype','sha256'} and info['shape']==list(weights[name].shape) and info['dtype']=='torch.float32'
                and isinstance(info['sha256'],str) and len(info['sha256'])==64,'Adam moment tensor metadata differs')
    groups=value['param_group_hyperparameters'];need(len(groups)==1,'Single matched AdamW parameter group required');group=groups[0]
    need(close(group['lr'],learning_rate(step),atol=1e-12,rtol=1e-12) and group['betas']==[.9,.999]
         and group['eps']==1e-8 and group['weight_decay']==0 and group['amsgrad'] is False and group['maximize'] is False,
         'Actual AdamW hyperparameters differ at endpoint')
    return ids


def checkpoint_path(root,step):return root/('final.pt' if step==6000 else f'step_{step:04d}.pt')


def training_loss_descriptions(logs,scenes):
    roles=defaultdict(list);cycles=defaultdict(lambda:dict(pair_presentations=0,scene_ce=[],roles=defaultdict(list)))
    for row in logs:
        cursor=0
        for i,sid in enumerate(row['sids']):
            count=len(scenes[sid]['target_ids']);values=row['position_losses']['ce'][cursor:cursor+count];cursor+=count
            group=cycles[row['cycles'][i//2]];group['scene_ce'].append(sum(values)/count)
            if i%2==0:group['pair_presentations']+=1
            for name,items in (('first_token',values[:1]),('EOS',values[-1:]),('name_continuation',values[1:-1])):
                roles[name].extend(items);group['roles'][name].extend(items)
    def metrics(groups):
        return [dict(labels=dict(role=k),metrics=dict(positions=len(v),mean_nll=None if not v else sum(v)/len(v))) for k,v in sorted(groups.items())]
    return dict(descriptive_only=True,token_roles=metrics(roles),cycles=[dict(cycle=k,pair_presentations=v['pair_presentations'],
        complete_pair_cycle=v['pair_presentations']==54,mean_scene_ce=sum(v['scene_ce'])/len(v['scene_ce']),token_roles=metrics(v['roles']))
        for k,v in sorted(cycles.items())],no_checkpoint_selection=True,horizon_and_annealing_unchanged=True,
        all_complete_cycles_and_final_partial_retained=True,no_convergence_certificate=True)


def self_test(torch):
    fixture=[dict(sid=f'{f}/{v}/{n}/{o}',base_contrast_id=str(f),variant=v,n_frames=n,orientation_version=o,first_token_correct=True)
        for f in range(18) for v in range(3) for n in (8,16) for o in ('original','flipped')]
    need(first_token_criterion(fixture)['passed'],'Perfect216 criterion fixture must pass')
    for row in fixture[:12]:row['first_token_correct']=False
    need(not first_token_criterion(fixture)['passed'],'204/216 must fail')
    for row in fixture:row['first_token_correct']=True
    for f in range(3):fixture[12*f]['first_token_correct']=False
    need(not first_token_criterion(fixture)['passed'] and first_token_criterion(fixture)['first_correct']==213,'15 complete families must fail')
    for row in fixture:row['first_token_correct']=True
    for i in range(0,12,2):fixture[i]['first_token_correct']=False
    need(not first_token_criterion(fixture)['passed'] and first_token_criterion(fixture)['orientation_correct']['original']==102,'102/108 orientation must fail')
    return dict(passed=True,criterion_groups=4,moment_algebra=moment_fixture(torch),no_pretrained_head_forward=True)


def audit_run(torch,out,directory,plan_path,plan,frozen,rows,scenes,order,states,index,stats,initial,norm,head):
    p=driver();directory=Path(directory).resolve();config=read(directory/'config.json');summary=read(directory/'summary.json');arm=config['arm']
    need(directory.parent==p.OUT and directory.name==config['run_id']==f'run_{arm}_{config["slurm_job_id"]}' and arm in ARMS
         and config['protocol']==p.PROTOCOL and config['policy']==p.POLICY and config['seed']==24
         and config['source_sha256']==frozen==plan['source_sha256'] and all(summary.get(k)==v for k,v in config.items())
         and Path(config['plan_file']).resolve()==Path(plan_path).resolve() and config['plan_sha256']==sha(plan_path), 'Run/config/source/plan identity differs')
    for name,digest in frozen.items():bind(directory/'source'/name.replace('/','_'),digest)
    inherited=dict(source_sha256=plan['inherited_source_sha256'])
    need(config['inherited_source_sha256']==plan['inherited_source_sha256'] and read(directory/'inherited_sources.json')==inherited, 'Inherited source descriptor differs')
    for flag in ('passed','completed','computational_integrity_passed','checkpoint_roundtrip_passed','native_weights_unchanged','statistics_frozen','statistics_versions_unchanged','input_versions_unchanged',
                 'cached_first_query_only','no_pretrained_backbone_loaded','no_dev_or_test','no_native_or_whole_answer_claim'):
        need(summary[flag] is True,'Run invariant failed: '+flag)
    need(summary['phase']=='run' and summary['steps']==6000 and not any(k.startswith('permutation_') for k in summary),'Horizon/scope differs')
    need(config['hardware']['gpu']=='NVIDIA B200' and config['hardware']['matmul_allow_tf32'] is False
         and config['hardware']['torch_version']==str(torch.__version__),'Native hardware/precision differs')
    for key in ('native_identity','native_identity_sha256','native_module_identity','native_model_file','native_model_sha256',
                'features_file','features_sha256','feature_tensor','stats_file','stats_sha256','stats_tensors',
                'uniform_plan','oracle_report','conditioning_plan','conditioning_report','factor_plan','cache_report','confirmation_report','training_stage',
                'paired_visual_plan','paired_visual_report','post_query_code_report','post_query_visual_report','query_placement','fixed_weight','moment_basis_comparison','orientation_plan','paired_order_stage','negative_evidence','decision_step','evaluation_steps'):
        need(config[key]==plan[key],'Consumed native/input identity differs: '+key)
    need(config['reused_unfitted_initialization'] is plan['reused_unfitted_initialization'] is False
         and config['fresh_shared_initialization'] is plan['fresh_shared_initialization'] is True
         and config['initialized']==plan['initial_state'] and config['initial_checkpoint_sha256']==plan['initial_sha256'],
         'Fresh shared moment initialization differs')
    trainable=list(initial)
    mask=dict(trainable_parameter_names=trainable,effective_trainable_parameters=PARAMETER_COUNT,retained_state_parameters=PARAMETER_COUNT)
    need(plan['trainability_mask']==mask and all(config[k]==v for k,v in mask.items()),'Six live moment parameters required')
    need(summary['basis']==config['basis']==arm and not any(k.startswith('permutation_') for k in summary)
         and not any(k in summary for k in ('interaction','pairing_restored','factor_derangement','frozen_selector_exact')),
         'Only the locked moment basis without selector/permutation state is permitted')
    for key in ('training','captures','calls','evaluations'):bind(summary[key+'_file'],summary[key+'_sha256'])
    for key in ('rows','presentations','stats_inventory'):bind(config[key+'_file'],config[key+'_sha256'])
    need(read(config['rows_file'])==rows and read(config['presentations_file'])==order and config['order_sha256']==object_sha(order)
         and read(config['stats_inventory_file'])==read(plan['stats_inventory_file']),'Copied rows/order/statistic inventory differ')
    root=p.CKPT/directory.name;data=p.DATA/directory.name
    need(Path(config['checkpoint_directory'])==root and Path(config['data_directory'])==data and Path(config['initial_checkpoint'])==root/'initial.pt','Authorized artifact roots differ')
    bind(config['initial_checkpoint'],config['initial_checkpoint_sha256']);selected=torch.load(config['initial_checkpoint'],map_location='cpu',weights_only=True)
    need(selected['seed']==24 and all(torch.equal(selected['branch'][k],initial[k]) for k in initial),'Actual run initialization differs')
    denominators={fid:float(states[index[fid]].float().square().sum()+1e-6) for scene in scenes.values() for fid in scene['global_feature_ids']}
    logs,expected_calls,training=audit_training(directory,summary,scenes,order,denominators)
    need([len(r['target_ids']) for r in logs]==plan['head_rows_by_update'] and close(summary['training_seconds'],sum(r['seconds'] for r in logs))
         and summary['training_head_rows']==training['training_target_positions'] and summary['total_head_rows']==training['total_head_rows'],'Fixed training count/timing differs')
    counters=dict(core=6014,conditioning=6014,norm=6014,head=6014,vlm=0,vision=0);calls=read(summary['calls_file'])
    need(calls==expected_calls and summary['counters']==counters
         and read(directory/'final_counters.json')==dict(counters=counters,recorded_calls=calls,partial_outputs_retained=True),'Chronological actual call ledger differs')
    evaluations=summary['evaluations'];need(read(summary['evaluations_file'])==evaluations and Path(summary['evaluations_file'])==directory/'evaluations.json'
         and summary['final_head_rows']==216 and summary['earlier_endpoints_descriptive_only'] is False and summary['decision_step']==6000
         and set(evaluations)=={str(e) for e in ENDPOINTS},'Only the fixed6000 endpoint required')
    weights={};endpoint_audits={};packets={};optimizer_ids=None;native_table=dict(norm=plan['native_identity']['norm_weight'],head=plan['native_identity']['head_weight'])
    for e in ENDPOINTS:
        desc=evaluations[str(e)];checkpoint=checkpoint_path(root,e)
        need(desc['endpoint_step']==e and desc['checkpoint']==str(checkpoint) and desc['descriptive_only'] is (e!=6000),'Endpoint identity/decision scope differs')
        bind(checkpoint,desc['checkpoint_sha256']);packet=torch.load(checkpoint,map_location='cpu',weights_only=True)
        need(set(packet)=={'branch','step','config'} and packet['step']==e and packet['config']==config,'After-update checkpoint ownership differs')
        w=packet['branch'];check_weights(torch,w,initial);weights[e]=w;table={k:p.tensor_info(v) for k,v in w.items()}
        for key in ('endpoint','roundtrip','raw','predictions'):bind(desc[key+'_file'],desc[key+'_sha256'])
        endpoint=read(desc['endpoint_file']);roundtrip=read(desc['roundtrip_file'])
        need(endpoint['passed'] is True and endpoint['query_placement']=='pre_silu' and endpoint['basis']==arm and endpoint['endpoint_step']==e and endpoint['before_evaluation']==endpoint['after_evaluation']==table
             and endpoint['parameter_sha256']==object_sha(table) and endpoint['native_before']==endpoint['native_after']==config['native_weight_identity']==native_table
             and endpoint['input_tensors_before']==endpoint['input_tensors_after']==dict(states=plan['feature_tensor'])
             and endpoint['input_versions_unchanged'] is True and endpoint['training_mode_restored'] is True
             and endpoint['optimizer_before']==endpoint['optimizer_after'] and endpoint['parameter_versions_before']==endpoint['parameter_versions_after'],
             'Final evaluation mutated readout/native/input/optimizer/trainability state')
        need(endpoint['working_statistics_before']==endpoint['working_statistics_after']=={k:plan['stats_tensors'][k] for k in ('means','global_mean','scale')}
             and endpoint['statistics_versions_unchanged'] is True and endpoint['hooks_removed'] is True
             and endpoint['basis']==arm and not any(k in endpoint for k in ('interaction','pairing_restored','factor_derangement','frozen_selector_exact')),
             'Original statistics or locked moment basis changed during endpoint evaluation')
        optimizer_identity=optimizer_audit(torch,endpoint['optimizer_before'],e,w)
        if optimizer_ids is None:optimizer_ids=optimizer_identity
        need(optimizer_ids==optimizer_identity and set(endpoint['parameter_versions_before'])==set(initial)
             and all(type(v) is int and v>=0 for v in endpoint['parameter_versions_before'].values()),'Parameter objects/version metadata changed across observations')
        count=e+14*sum(x<=e for x in ENDPOINTS)
        need(endpoint['counters']==dict(core=count,conditioning=count,norm=count,head=count,vlm=0,vision=0),'Endpoint cumulative call count differs')
        need(roundtrip['passed'] is True and roundtrip['step']==e and roundtrip['config_sha256']==sha(directory/'config.json')
             and roundtrip['selected']==roundtrip['serialized']==table and roundtrip['reset_performed'] is (e==6000)
             and roundtrip['initial']==(plan['initial_state'] if e==6000 else None),'Endpoint serialization/final reset identity differs')
        for value in (endpoint,roundtrip):need(value['checkpoint']==str(checkpoint) and value['checkpoint_sha256']==desc['checkpoint_sha256'],'Endpoint checkpoint hash differs')
        raw=torch.load(desc['raw_file'],map_location='cpu',weights_only=True);logits=raw['logits']
        need(raw['schema_version']==1 and raw['endpoint_step']==e and Path(desc['raw_file'])==data/f'logits_{e:04d}.pt'
             and Path(desc['predictions_file'])==directory/f'predictions_{e:04d}.json'
             and Path(desc['endpoint_file'])==directory/f'endpoint_{e:04d}.json' and Path(desc['roundtrip_file'])==directory/f'roundtrip_{e:04d}.json'
             and raw['sids']==[r['sid'] for r in rows] and logits.dtype==torch.float16 and logits.shape==(216,152064)
             and bool(torch.isfinite(logits).all()) and p.tensor_info(logits)==desc['raw_tensor']
             and raw['batches']==desc['raw_batches'] and len(raw['batches'])==14,'Endpoint full-vocabulary raw ownership differs')
        packets[e]=raw;endpoint_audits[str(e)]=dict(passed=True,endpoint_step=e,checkpoint=str(checkpoint),checkpoint_sha256=desc['checkpoint_sha256'],
            selected_parameter_sha256=object_sha(table),endpoint_file=desc['endpoint_file'],endpoint_sha256=desc['endpoint_sha256'],
            roundtrip_file=desc['roundtrip_file'],roundtrip_sha256=desc['roundtrip_sha256'],evaluation_preserved_training_state=True,descriptive_only=e!=6000)
    for key in ('checkpoint','checkpoint_sha256','endpoint_file','endpoint_sha256','roundtrip_file','roundtrip_sha256','raw_file','raw_sha256','raw_tensor','raw_batches','predictions_file','predictions_sha256','first_token_fit'):
        need(summary[key]==evaluations['6000'][key],'Ordinary summary alias must bind only step6000: '+key)
    need(summary['selected_parameter_sha256']==endpoint_audits['6000']['selected_parameter_sha256'],'Primary deployed endpoint identity differs')
    records=read(summary['captures_file']);expected=[]
    for step in range(1,6001):
        if step in DIAGNOSTIC_STEPS:expected.append(('training',step,None))
        if step in ENDPOINTS:expected.extend(('evaluation',batch,step) for batch in range(1,15))
    need([(r['phase'],r['step'],r['endpoint_step']) for r in records]==expected and all(r['arm']==r['basis']==arm and r['query_placement']=='pre_silu' for r in records),'Chronological22-capture inventory differs')
    need(sorted(f.name for f in root.glob('*.pt'))==sorted(['initial.pt','final.pt']+[f'training_core_{s:04d}.pt' for s in DIAGNOSTIC_STEPS]),'Fixed checkpoint inventory differs')
    need(sorted(f.name for f in directory.glob('progress_*.json'))==[f'progress_{s:04d}.json' for s in PROGRESS],'Four progress snapshots required')
    for step in PROGRESS:
        count=step+14*sum(e<=step for e in ENDPOINTS)
        prior=[r for r in records if (r['step'] if r['phase']=='training' else r['endpoint_step'])<=step]
        need(read(directory/f'progress_{step:04d}.json')==dict(step=step,training=logs[:step],captures=prior,
            counters=dict(core=count,conditioning=count,norm=count,head=count,vlm=0,vision=0),evaluations={str(e):evaluations[str(e)] for e in ENDPOINTS if e<=step}),'Progress snapshot/endpoint chronology differs')
    audits=[];gpu_nll={e:[] for e in ENDPOINTS};cursors={e:0 for e in ENDPOINTS}
    with torch.no_grad():
        for record in records:
            phase,step,e=record['phase'],record['step'],record['endpoint_step']
            expected_weight=root/f'training_core_{step:04d}.pt' if phase=='training' else checkpoint_path(root,e)
            expected_capture=data/(f'training_capture_{step:04d}.pt' if phase=='training' else f'evaluation_{e:04d}_capture_{step:04d}.pt')
            need(Path(record['weights_file'])==expected_weight and Path(record['file'])==expected_capture,'Capture/checkpoint paths differ')
            audit,actual=audit_capture(torch,out,record,arm,plan,rows,scenes,states,index,stats,initial,weights,logs,norm,head);audits.append(audit)
            if phase=='evaluation':
                batch=packets[e]['batches'][step-1];count=len(record['sids']);cursor=cursors[e]
                need(batch['capture_file']==record['file'] and batch['capture_sha256']==record['sha256'] and batch['sids']==record['sids']
                     and batch['logits']==record['tensors']['logits'] and len(batch['gpu_fp32_nll'])==count
                     and torch.equal(actual,packets[e]['logits'][cursor:cursor+count]), 'Endpoint batch/capture ownership differs')
                cursors[e]+=count;gpu_nll[e]+=batch['gpu_fp32_nll']
    need(set(cursors.values())=={216},'The final endpoint requires216 raw rows')
    endpoint_rows={};strata=[]
    for e in ENDPOINTS:
        desc=evaluations[str(e)];logits=packets[e]['logits'];ids=logits.argmax(-1).tolist();nll=fp64_nll(torch,logits,[r['first_token_id'] for r in rows])
        rescored=[dict(r,argmax_id=ids[i],first_token_correct=ids[i]==r['first_token_id'],nll=float(nll[i]),gpu_fp32_nll=gpu_nll[e][i]) for i,r in enumerate(rows)]
        saved=read(desc['predictions_file'])
        need(len(saved)==216 and all(set(a)==set(b) and all(a[k]==b[k] for k in a if k!='nll') for a,b in zip(rescored,saved)),
             'Endpoint actual argmax/scene ownership differs')
        precision=[dict(sid=r['sid'],fp64_nll=r['nll'],gpu_fp32_nll=r['gpu_fp32_nll'],absolute_difference=abs(r['nll']-r['gpu_fp32_nll']),
            saved_fp64_nll=b['nll'],saved_fp64_passed=close(r['nll'],b['nll']),
            passed=close(r['nll'],r['gpu_fp32_nll']) and close(r['nll'],b['nll'])) for r,b in zip(rescored,saved)]
        save(out/f'{arm}_outcomes_{e:04d}.json',rescored);save(out/f'{arm}_nll_precision_{e:04d}.json',precision)
        criterion=first_token_criterion(rescored);need(criterion==desc['first_token_fit'],'Endpoint screen counts differ')
        endpoint_rows[e]=rescored;endpoint_audits[str(e)].update(nll_precision_passed=all(r['passed'] for r in precision),first_token_fit=criterion,mean_first_token_nll=sum(r['nll'] for r in rescored)/216,
            outcomes_file=str(out/f'{arm}_outcomes_{e:04d}.json'),outcomes_sha256=sha(out/f'{arm}_outcomes_{e:04d}.json'))
        for key in ('gold','question','n_frames','base_contrast_id','orientation_version'):
            groups=defaultdict(list)
            for row in rescored:groups[row[key]].append(row)
            for label,group in sorted(groups.items(),key=lambda x:str(x[0])):
                strata.append(dict(labels=dict(endpoint_step=e,**{key:label}),metrics=dict(correct=sum(r['first_token_correct'] for r in group),contexts=len(group),
                    mean_nll=sum(r['nll'] for r in group)/len(group),all_correct=all(r['first_token_correct'] for r in group))))
    final_criterion=primary_screen(endpoint_rows);need(final_criterion==summary['first_token_fit'],'Only final6000 screen may decide')
    save(out/f'{arm}_strata.json',strata);save(out/f'{arm}_training_loss_descriptions.json',training_loss_descriptions(logs,scenes))
    return dict(passed=all(r['passed'] for r in audits) and all(r['nll_precision_passed'] for r in endpoint_audits.values()),
        completed=True,all_numerical_audits_collected=True,arm=arm,slurm_job_id=config['slurm_job_id'],run_directory=str(directory),run_summary_sha256=sha(directory/'summary.json'),
        first_token_fit=final_criterion,primary_endpoint_step=6000,endpoint_audits=endpoint_audits,training=training,counters=counters,capture_audits=audits,
        cpu_head_calls=14,cpu_norm_calls=14,cpu_head_rows=sum(r['head_rows'] for r in audits),
        maximum_native_replay_tv=max(m['full_vocabulary_tv'] for r in audits for m in r['native_head']),
        strata_file=str(out/f'{arm}_strata.json'),strata_sha256=sha(out/f'{arm}_strata.json'),
        training_loss_descriptions_file=str(out/f'{arm}_training_loss_descriptions.json'),training_loss_descriptions_sha256=sha(out/f'{arm}_training_loss_descriptions.json'))


def resources(out,runs):
    command=['sacct','-X','-u',os.environ['USER'],'--starttime=2026-09-01','--parsable2','--noheader',
        '--format=JobIDRaw,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,Start,End']
    raw=subprocess.run(command,capture_output=True,text=True,check=True).stdout;file=out/'all_user_sacct.psv';file.write_text(raw)
    job_arms={name:arm for arm,name in driver().RUN_JOBS.items()};names=set(job_arms);records=allocation_rows(raw,names|{'identity_join_moment_basis','identity_join_moment_basis_run'})
    need(len(records)==2 and {r['name'] for r in records}==names,'Exactly one attempt per factor arm required; all failed/zero allocations retained')
    for row in records:
        arm=job_arms[row['name']]
        need(row['job_id']==runs[arm]['slurm_job_id'] and row['partition']=='gpu' and row['state']=='COMPLETED'
             and row['exit_code']=='0:0' and row['gpus']==1 and row['seconds']<=120,'Moment-basis allocation identity/cap differs')
    from datetime import datetime
    events=[]
    for row in records:
        a,b=map(datetime.fromisoformat,(row['start'],row['end']));need(b>=a,'Invalid allocation interval')
        if b>a:events.extend(((a,1),(b,-1)))
    current=peak=0
    for _,d in sorted(events):current+=d;need(current>=0,'Invalid allocation overlap');peak=max(peak,current)
    total=sum(r['gpu_seconds'] for r in records);need(current==0 and peak<=2 and total<=240,'Moment-basis campaign cap exceeded')
    return dict(passed=True,jobs=records,allocated_gpu_seconds=total,maximum_concurrent_gpus=peak,
        failed_and_zero_allocations_retained=True,file=str(file),sha256=sha(file),command=command)


def report(args,out,frozen):
    import torch
    torch.set_num_threads(4);p=driver();plan=p.verify_plan(args.plan,ancestors=True)
    need(frozen==plan['source_sha256']==p.sources() and len(frozen)==7 and len(args.runs)==2
         and plan['inherited_source_sha256']==p.inherited_sources(),'New/inherited source or two-run inventory differs')
    need(read(out/'inherited_sources.json')==dict(source_sha256=plan['inherited_source_sha256']),'Report source descriptor differs')
    need(plan['decision_step']==6000 and plan['evaluation_steps']==list(ENDPOINTS)
         and plan['reused_unfitted_initialization'] is False and plan['fresh_shared_initialization'] is True
         and plan['no_new_statistics'] is True and plan['orientation_coverage_only'] is False and plan['paired_minibatch_only'] is False
         and plan['query_placement']=='pre_silu' and plan['fixed_weight']==.5 and plan['bases']==list(ARMS)
         and plan['moment_basis_comparison'] is True,'Registered two-basis scope differs')
    rows,scenes,order,states,index,stats,initial=audit_inputs(torch,plan)
    (p.DATA/out.name).mkdir(parents=True,exist_ok=False)
    exposures=Counter(r['orientation_version'] for r in order)
    save(out/'input_reconstruction_audit.json',dict(passed=True,factor_plan=plan['factor_plan'],cache_report=plan['cache_report'],
        confirmation_report=plan['confirmation_report'],training_stage=plan['training_stage'],contexts=216,local_occurrences=2592,
        original_statistic_contexts=108,original_statistic_occurrences=1296,pair_presentations=48000,
        independent_paired_order_exact=True,orientation_exposure_counts=dict(exposures),training_target_positions=213330,endpoint_rows_per_arm=216,
        actual_native_bank_rows=len(states),all_native_row_hashes_exact=True,global_conditioning_independently_reconstructed=True,
        fresh_shared_seed24_six_tensor_initialization_exact=True,no_fitted_initialization=True,
        actual_trainable_parameters=PARAMETER_COUNT,confirmation_manifest_bound_before_fit=True,matched_288_coordinate_moment_bases=True))
    need(p.oracle.native_module_identity(torch,plan['native_identity'])==plan['native_module_identity'],'Actual native norm/head source differs')
    packet=torch.load(plan['native_model_file'],map_location='cpu',weights_only=True);norm,head=p.oracle.native_modules(torch,packet,plan['native_identity'],'cpu');del packet
    before=dict(norm=p.tensor_info(norm.weight),head=p.tensor_info(head.weight));runs={}
    for directory in args.runs:
        value=audit_run(torch,out,directory,args.plan,plan,frozen,rows,scenes,order,states,index,stats,initial,norm,head)
        need(value['arm'] not in runs,'Duplicate moment arm');runs[value['arm']]=value
    need(set(runs)==set(ARMS) and before==dict(norm=p.tensor_info(norm.weight),head=p.tensor_info(head.weight))
         and norm.weight.grad is head.weight.grad is None and not norm.weight.requires_grad and not head.weight.requires_grad,
         'Two-arm frozen native audit incomplete')
    need(all(r['cpu_head_calls']==r['cpu_norm_calls']==14 and r['cpu_head_rows']==216 and len(r['capture_audits'])==22 for r in runs.values()),
         'Both arms require22 captures/14 CPU head batches/216 rows')
    budget=resources(out,runs)
    outcomes={arm:read(runs[arm]['endpoint_audits']['6000']['outcomes_file']) for arm in ARMS}
    need([r['sid'] for r in outcomes['within']]==[r['sid'] for r in outcomes['cross']],'Matched endpoint order differs')
    transitions=Counter((a['first_token_correct'],b['first_token_correct']) for a,b in zip(outcomes['within'],outcomes['cross']))
    comparison=dict(endpoint_step=6000,contexts=216,all_comparisons_descriptive=True,
        within_to_cross_transitions={f'{int(a)}_to_{int(b)}':transitions[a,b] for a in (False,True) for b in (False,True)},
        cross_minus_within_correct=sum(r['first_token_correct'] for r in outcomes['cross'])-sum(r['first_token_correct'] for r in outcomes['within']),
        mean_first_token_nll={arm:runs[arm]['endpoint_audits']['6000']['mean_first_token_nll'] for arm in ARMS},
        same_input_information_in_real_arithmetic=True,finite_width_function_equivalence_claim=False,
        matched_gradient_or_numerical_conditioning_claim=False)
    result=dict(passed=all(r['passed'] for r in runs.values()),completed=True,all_numerical_audits_collected=True,phase='report',protocol=p.PROTOCOL,
        source_sha256=frozen,inherited_source_sha256=plan['inherited_source_sha256'],plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),
        factor_plan=plan['factor_plan'],cache_report=plan['cache_report'],confirmation_report=plan['confirmation_report'],training_stage=plan['training_stage'],
        orientation_plan=plan['orientation_plan'],paired_visual_plan=plan['paired_visual_plan'],paired_visual_report=plan['paired_visual_report'],
        post_query_code_report=plan['post_query_code_report'],post_query_visual_report=plan['post_query_visual_report'],
        paired_order_stage=plan['paired_order_stage'],negative_evidence=plan['negative_evidence'],stats_file=plan['stats_file'],stats_sha256=plan['stats_sha256'],
        stats_tensors=plan['stats_tensors'],runs=runs,comparison=comparison,resources=budget,
        primary_endpoint_step=6000,first_token_screen={arm:runs[arm]['first_token_fit'] for arm in ARMS},
        cpu_head_calls=28,cpu_norm_calls=28,cpu_head_rows=432,gpu_head_calls=12028,gpu_norm_calls=12028,gpu_head_rows=427092,vlm_calls=0,vision_calls=0,
        paired_minibatch_order_retained=True,horizon_and_annealing_unchanged=True,bases=list(ARMS),query_placement='pre_silu',fixed_weight=.5,
        effective_trainable_parameters_per_arm=PARAMETER_COUNT,moment_basis_comparison=True,no_selector_or_factor_permutation_state=True,
        no_factor_semantics_or_expressiveness_claim=True,cached_first_query_only=True,no_native_or_whole_answer_claim=True,
        no_generalization_or_reasoning_claim=True,no_new_fit_or_backbone_forward=True,no_automatic_native_or_fresh_release=True,all_outcomes_retained=True)
    save(out/'analysis.json',result)
    lines=['# Matched moment-basis cached training','',
        ('Independent computation and provenance audits passed. ' if result['passed'] else 'Unchanged numerical gates failed; complete comparisons and replay tensors are retained. ')+
        'Both arms used the same fresh seed24 initialization, six trainable tensors (1,404,192 coordinates), native feature bank, original global conditioning statistics, '
        'paired presentation order and full-name-plus-EOS CE. A=sum(.5a), B=sum(.5b), C=sum(.25ab), P=A*B-C. '
        'The 288-coordinate readout receives [A,B,C] or [A,B,P]; both use delta=U SiLU(Wagg*aggregate+b+q), with the actual query retained in both factors.','',
        '| Basis | Correct /216 | Original /108 | Flipped /108 | Complete families /18 | Saved GPU screen |',
        '|---|---:|---:|---:|---:|---|']
    for arm in ARMS:
        c=runs[arm]['first_token_fit'];role='PASS' if c['passed'] else 'FAIL'
        lines.append(f"| {arm} | {c['first_correct']} | {c['orientation_correct']['original']} | {c['orientation_correct']['flipped']} | {c['complete_families']} | {role} |")
    lines+=['','Only step6000 determines acceptance:206/216 pooled,103/108 per orientation and16/18 complete12-context families. '
        'All outcomes, strata and training losses are retained. These are cached training first queries. No result automatically releases native or fresh evaluation.','',
        'The bases preserve the same information for fixed factors in real arithmetic. This does not establish identical finite-width function classes, gradients or numerical conditioning. '
        'The cross basis has different length dependence and possible subtraction cancellation; bound per-capture signed statistics and FP64 discrepancies are descriptive and do not alter fidelity gates. '
        'No semantic roles are assigned to learned factors. Neither success nor failure establishes convergence, general reasoning or architectural incapacity.','',
        'The final checkpoint was serialized, reset to the fresh state and reloaded. Audits check optimizer state, native weights, features, statistics, all6000 CE reductions, '
        'actual captured moments/readout and independent ordered-pair value/gradient fixtures. No optimization trajectory was replayed. Prior failures remain unchanged.','',
        f"Allocated GPU cost: {budget['allocated_gpu_seconds']} seconds. Each arm used6014 core/conditioning/norm/head calls and213546 head rows. "
        'VLM/vision calls were zero. CPU audit covered44 captures and28 endpoint head batches/432 rows, with exact argmax and TV<=0.02. '
        f"Maximum TV was {max(r['maximum_native_replay_tv'] for r in runs.values()):.9f}.\n"]
    (out/'REPORT.md').write_text('\n'.join(lines))
    need(result['passed'],'Unchanged numerical core/native/NLL gate failed; all44 captures and28 native replay batches retained')
    return dict(passed=True,completed=True,phase='report',protocol=p.PROTOCOL,source_sha256=frozen,inherited_source_sha256=plan['inherited_source_sha256'],
        plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),
        primary_endpoint_step=6000,first_token_screen=result['first_token_screen'],resources=budget,bases=list(ARMS),query_placement='pre_silu',fixed_weight=.5,
        paired_visual_plan=plan['paired_visual_plan'],paired_visual_report=plan['paired_visual_report'],post_query_code_report=plan['post_query_code_report'],
        post_query_visual_report=plan['post_query_visual_report'],moment_basis_comparison=True,cached_first_query_only=True,
        no_native_or_whole_answer_claim=True,no_automatic_native_or_fresh_release=True)


def main():
    import argparse,time
    p=driver();parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--runs',type=Path,nargs=2,required=True);args=parser.parse_args();p.native.require_slurm(gpu=False)
    need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),'CPU-only independent report')
    out=p.OUT/f'report_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    frozen=p.snapshot(out);save(out/'request.json',dict(plan=str(args.plan),runs=list(map(str,args.runs)),source_sha256=frozen))
    try:
        result=report(args,out,frozen);need(p.sources()==frozen and p.inherited_sources()==result['inherited_source_sha256'],'Source changed during report')
        result['elapsed_seconds']=time.perf_counter()-started;need(result['elapsed_seconds']<=300,'Fixed CPU report cap exceeded');save(out/'summary.json',result)
        print(__import__('json').dumps(dict(passed=True,completed=True,directory=str(out))),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,
            elapsed_seconds=time.perf_counter()-started,partial_outputs_retained=True,
            analysis_file=str(out/'analysis.json') if (out/'analysis.json').exists() else None,
            analysis_sha256=sha(out/'analysis.json') if (out/'analysis.json').exists() else None));raise


if __name__=='__main__':main()

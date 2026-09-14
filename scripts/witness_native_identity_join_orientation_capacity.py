"""Finite-precision existence witness; no fit, backbone, generation or GPU.

verify_witness/verify_report are hash-only dependency APIs and return analysis.
The only tensor execution entry point is the fixed CPU Slurm main().
"""
from __future__ import annotations
import ast
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import diagnose_native_identity_join_factor_orientation_training as parent
from scripts.stage_native_vision_v6_teacher import need,read,save,sha,object_sha
OUT=REPO/'outputs/native_aggregation_vlm/identity_join_orientation_capacity'
DATA=Path('/mnt/data/gabriele/gnn_transformer/identity_join_orientation_capacity')
CKPT=Path('/mnt/ckpts/gabriele/gnn_transformer/identity_join_orientation_capacity')
PLAN=parent.OUT/'check_443676/plan.json'
PLAN_SHA='d299d42a4444520f148f86252e2931ad599e03afac9edadbe9201836229bf097'
REFERENCE=REPO/'outputs/native_aggregation_vlm/identity_join_readout/report_precision_443304/summary.json'
REFERENCE_SHA='cb280b2bf6d769885ee20f395a6adaeed0aaec09ef00a5899258d6049b5490c7'
REFERENCE_ANALYSIS_SHA='936c5a9fee35bda4f67df0c6e20e5ef64055dbaea6ed94cd33edfcae1feabe82'
REFERENCE_U_SHA='5bfb7bef29f200674a87346e9984692e004425fb5d10ed2d4fb182e07597ad74'
CORE='gnnformer/parallel_local_joint_code_oracle.py'
CORE_SHA='3ebcebb62df088bfdbe07042e9b1bd5b87c003dd7108c8c0b506469237794690'
NOTE='docs/paper/NATIVE_AGGREGATION_JOINT_CODE_CAPACITY_NOTE.md'
NOTE_SHA='a65dc9db0a47087c2a45659daa59bd011849c00b76982c03aedf48354cccc625'
PROPOSAL='docs/paper/NATIVE_AGGREGATION_ORIENTATION_CAPACITY_WITNESS_PROPOSAL.md'
OWN=('scripts/witness_native_identity_join_orientation_capacity.py','slurm/native_identity_join_orientation_capacity.sbatch',PROPOSAL)
PROTOCOL='identity_join_orientation_finite_precision_capacity_witness'
PEOPLE=('Sandra','Mary','Michael','John','Daniel','Laura','Peter','Emma','Noah')
KEYS=('query.weight','aggregate_projection.weight','aggregate_projection.bias','up.weight')
POLICY=dict(cpu_seconds=300,cpu_cores=4,memory_gib=16,contexts=216,local_occurrences=2592,rank=96,active_units=36,
    batch_size=16,core_calls=14,norm_calls=28,head_calls=28,head_rows=432,residual_atol=1e-4,residual_rtol=1e-4,
    full_vocabulary_tv=0.02,correct_required=216,complete_families_required=18,
    fixed_constant='binary64 sqrt(96)/tanh(.5); scale FP64, then one FP32 cast',
    no_fit=True,no_GPU=True,no_backbone=True,no_generation=True,no_runtime_semantic_method=True)


def bind(path,bindings,expected=None):
    path=Path(path).resolve();digest=sha(path)
    need(expected is None or digest==expected,'Changed witness binding: '+str(path))
    need(str(path) not in bindings or bindings[str(path)]==digest,'Conflicting witness binding')
    bindings[str(path)]=digest;return digest


def sources():return {name:sha(REPO/name) for name in OWN}


def inherited_sources():
    need(sha(REFERENCE)==REFERENCE_SHA and sha(REPO/CORE)==CORE_SHA and sha(REPO/NOTE)==NOTE_SHA,'Fixed witness ancestry changed')
    proof=read(REFERENCE);result={}
    for group in (parent.sources(),parent.inherited_sources(),proof['reporting_source_sha256'],{CORE:CORE_SHA,NOTE:NOTE_SHA}):
        for name,digest in group.items():
            need(name not in result or result[name]==digest,'Conflicting inherited source');result[name]=digest
    for name,digest in result.items():need(sha(REPO/name)==digest,'Inherited source changed: '+name)
    return result


def snapshot(out):
    frozen=sources();inherited=inherited_sources();(out/'source').mkdir()
    for name,digest in frozen.items():
        dest=out/'source'/name.replace('/','_');dest.write_bytes((REPO/name).read_bytes());need(sha(dest)==digest,'Source copy differs')
    save(out/'source_hashes.json',frozen);save(out/'inherited_sources.json',dict(source_sha256=inherited));return frozen,inherited


def reference(bindings):
    bind(REFERENCE,bindings,REFERENCE_SHA);summary=read(REFERENCE)
    need(summary['passed'] is True and summary['completed'] is True,'Original answer-code report did not pass')
    bind(summary['analysis_file'],bindings,REFERENCE_ANALYSIS_SHA);a=read(summary['analysis_file'])
    need(summary['analysis_sha256']==REFERENCE_ANALYSIS_SHA and a['passed'] is True and a['completed'] is True
         and a['first_token_fit']['correct']==108,'Original answer-code evidence differs')
    for name,digest in summary['reporting_source_sha256'].items():bind(REFERENCE.parent/'source'/name.replace('/','_'),bindings,digest)
    bind(a['plan_file'],bindings,a['plan_sha256']);plan=read(a['plan_file'])
    runfile=Path(a['run_directory'])/'summary.json';bind(runfile,bindings,a['run_summary_sha256']);run=read(runfile)
    need(run['passed'] is True and run['completed'] is True and run['plan_sha256']==a['plan_sha256'],'Original oracle run differs')
    bind(run['checkpoint'],bindings,REFERENCE_U_SHA);need(run['checkpoint_sha256']==REFERENCE_U_SHA,'Original answer-code U differs')
    for key in ('rows','inputs'):
        bind(plan[key+'_file'],bindings,plan.get(key+'_sha256'))
    return plan,run


def semantic_inventory(rows,scenes,bindings):
    """Map one observed person/room to a local code; labels only audit outputs."""
    need(len(rows)==len({r['sid'] for r in rows})==216 and len(scenes)==216,'Complete216 cohort required')
    inventories=[];codes=[];offsets=[0];pairs=defaultdict(dict)
    rooms={'Kitchen','Bathroom','Garden','Office','Bedroom','Park'}
    for row in rows:
        sid=row['sid'];scene=scenes[sid];pair=row['room_pair'];n=row['n_frames']
        need(n in (8,16) and len(pair)==2 and len(set(pair))==2 and set(pair)<=rooms,'Unknown requested-room domain')
        question=f"Consider only images in the {pair[0]} or the {pair[1]}. Which person appears in both rooms? Reply with the person's name only."
        need(row['question']==question and scene['question']==question and scene['target_prefixes'][0]==[]
             and row['target_ids']==scene['target_ids'] and row['first_token_id']==row['target_ids'][0],'Question/first-query ownership differs')
        qa=Path(row['path'])/'qa.txt';bind(qa,bindings,row['qa_sha256']);lines=qa.read_text().splitlines()
        need(len(lines)==n+4 and lines[0]=='question:' and lines[-3]==question and lines[-2]=='answer:' and lines[-1]==row['gold'],'QA envelope differs')
        states=[ast.literal_eval(line) for line in lines[1:-3]]
        need(object_sha(dict(states=states,question=question))==row['content_sha256']==scene['content_sha256'],'Canonical state identity differs')
        need(len(row['image_files'])==n,'Frame count differs');frames=[];counts=[[0,0] for _ in PEOPLE]
        for i,(state,image) in enumerate(zip(states,row['image_files'])):
            need(set(state)=={'step_id','rooms'} and state['step_id']==i+1 and len(state['rooms'])==1,'One ordered state per image required')
            room,occupants=next(iter(state['rooms'].items()))
            need(room in rooms and isinstance(occupants,list) and len(occupants)==1 and occupants[0] in PEOPLE,'Unknown or ambiguous local state')
            person=occupants[0];expected=Path(row['path'])/f'{i:03d}.png'
            need(Path(image['path'])==expected and image['dimensions']==[512,512] and image['mode']=='RGB','Frame path/order/render declaration differs')
            bind(expected,bindings,image['sha256']);coordinate=None
            if room in pair:coordinate=2*PEOPLE.index(person)+pair.index(room);counts[PEOPLE.index(person)][pair.index(room)]+=1
            code=[0]*96
            if coordinate is not None:code[coordinate]=1
            codes.append(code);frames.append(dict(local_index=i,step_id=i+1,person=person,room=room,image_file=str(expected),image_sha256=image['sha256'],code_index=coordinate))
        need(all(tuple(v) in ((0,0),(2,0),(0,2),(1,1)) for v in counts),'Outside certified half-count domain')
        winners=[PEOPLE[i] for i,v in enumerate(counts) if v==[1,1]]
        need(winners==[row['gold']],'Independent domain answer differs from scoring label')
        start=offsets[-1];offsets.append(len(codes));pairs[row['base_sid']][row['orientation_version']]=(row,counts,frames,codes[start:])
        inventories.append(dict(sid=sid,base_sid=row['base_sid'],orientation_version=row['orientation_version'],question=question,room_pair=pair,
            qa_file=str(qa),qa_sha256=row['qa_sha256'],content_sha256=row['content_sha256'],code_start=start,code_stop=len(codes),frames=frames,
            global_feature_id=scene['global_feature_ids'][0]))
    need(len(codes)==2592 and len(offsets)==217 and len(pairs)==108,'Local occurrence/paired orientation coverage differs')
    for group in pairs.values():
        need(set(group)=={'original','flipped'},'Missing orientation counterpart');a,b=group['original'],group['flipped']
        need(a[0]['gold']==b[0]['gold'] and a[0]['question']==b[0]['question'] and len(a[2])==len(b[2]),'Orientation target/question differs')
        need(a[1]==[v[::-1] for v in b[1]],'Requested-room counts did not swap')
        for left,right,lc,rc in zip(a[2],b[2],a[3],b[3]):
            expected_room=a[0]['room_pair'][1-a[0]['room_pair'].index(left['room'])] if left['room'] in a[0]['room_pair'] else left['room']
            need(left['person']==right['person'] and right['room']==expected_room and lc[:18]==[rc[j^1] for j in range(18)],'Per-image full-room swap differs')
    return dict(schema_version=1,people=list(PEOPLE),rank=96,active_coordinates=18,local_mapping_only=True,scenes=inventories),codes,offsets


def construction(torch,U):
    need(U.dtype==torch.float32 and tuple(U.shape)==(3584,96) and bool(torch.isfinite(U).all()),'Original U invalid')
    state={'query.weight':torch.zeros(96,3584),'aggregate_projection.weight':torch.zeros(96,96),
           'aggregate_projection.bias':torch.zeros(96),'up.weight':torch.zeros(3584,96)}
    multiplier=math.sqrt(96)/math.tanh(.5)
    for person in range(9):
        column=(U[:,person].double()*multiplier).float()
        for k,coefficients in enumerate(((1,1),(-1,-1),(1,-1),(-1,1))):
            unit=4*person+k;state['aggregate_projection.weight'][unit,2*person:2*person+2]=torch.tensor(coefficients,dtype=torch.float32)
            state['up.weight'][:,unit]=column if k<2 else -column
    need(tuple(state)==KEYS and sum(v.numel() for v in state.values())==697440,'Existing readout storage differs')
    return state


def self_test():
    silu=lambda x:x/(1+math.exp(-x));f=lambda x:silu(x)+silu(-x)
    for a,b,want in ((0,0,0),(1,0,0),(0,1,0),(.5,.5,1)):
        j=lambda x,y:(f(x+y)-f(x-y))/math.tanh(.5)
        need(abs(j(a,b)-want)<1e-14 and abs(j(a,b)-j(b,a))<1e-14,'Constructive domain identity failed')
    return dict(passed=True,domain_cases=4,room_exchange_invariant=True)


def score(rows,predictions):
    need(len(rows)==len(predictions)==216,'Witness scoring denominator differs')
    enriched=[dict(row,first_token_correct=bool(pred['correct'])) for row,pred in zip(rows,predictions)]
    result=parent.criteria(enriched)
    return dict(correct=result['first_correct'],contexts=216,orientation_correct=result['orientation_correct'],
        complete_families=result['complete_families'],families=18,
        passed=result['first_correct']==216 and result['complete_families']==18)


def verify_witness(summary_path):
    """Read-only hashes/JSON, never a tensor load, solve, fit or head replay."""
    path=Path(summary_path);path=path/'summary.json' if path.is_dir() else path
    summary=read(path);need(summary['passed'] is True and summary['completed'] is True and summary['protocol']==PROTOCOL
        and summary['capacity_witness_passed'] is True and summary['source_sha256']==sources()
        and summary['inherited_source_sha256']==inherited_sources(),'Passed fixed capacity witness required')
    for name,digest in summary['source_sha256'].items():need(sha(path.parent/'source'/name.replace('/','_'))==digest,'Witness source archive differs')
    need(read(path.parent/'source_hashes.json')==summary['source_sha256'] and read(path.parent/'inherited_sources.json')==dict(source_sha256=summary['inherited_source_sha256']), 'Witness source ledgers differ')
    need(sha(summary['analysis_file'])==summary['analysis_sha256'] and sha(summary['report_file'])==summary['report_sha256'],'Witness result changed')
    a=read(summary['analysis_file'])
    need(a['passed'] is True and a['completed'] is True and a['capacity_witness_passed'] is True and a['policy']==POLICY
         and a['source_sha256']==summary['source_sha256'] and a['inherited_source_sha256']==summary['inherited_source_sha256']
         and a['orientation_plan']==dict(file=str(PLAN),sha256=PLAN_SHA) and a['constructed']['passed'] and a['reference']['passed']
         and a['calls']==dict(core=14,norm=28,head=28,head_rows=432),'Witness analysis contract differs')
    for group in (a['input_bindings'],a['output_bindings']):
        for name,digest in group.items():need(sha(name)==digest,'Witness artifact changed: '+name)
    need(object_sha(a['native_identity'])==a['native_identity_sha256'],'Witness native identity differs')
    return a


verify_report=verify_witness


def run(out,data,ckpt,frozen,inherited,bindings):
    import torch
    from gnnformer.parallel_local_joint_code_oracle import ParallelLocalJointCodeOracle
    torch.set_num_threads(4);bind(PLAN,bindings,PLAN_SHA);plan=parent.verify_plan(PLAN,ancestors=True)
    original_plan,original_run=reference(bindings);rows=read(plan['rows_file']);scenes=read(plan['scenes_file'])
    for key in ('rows','scenes','features','native_model'):bind(plan[key+'_file'],bindings,plan.get(key+'_sha256'))
    for name,digest in plan['input_bindings'].items():bindings[name]=digest
    for name,digest in plan['runtime_bindings'].items():bindings[name]=digest
    inventory,code_values,offsets=semantic_inventory(rows,scenes,bindings);save(out/'code_inventory.json',inventory);save(out/'rows.json',rows)
    tests=self_test();save(out/'self_test.json',tests)
    bank=torch.load(plan['features_file'],map_location='cpu',weights_only=True)
    need(parent.tensor_info(bank['states'])==plan['feature_tensor'] and len(bank['index'])==len(bank['feature_ids'])==2940,'Frozen native feature bank differs')
    old_inputs=torch.load(original_plan['inputs_file'],map_location='cpu',weights_only=True)
    need({k:parent.tensor_info(old_inputs[k]) for k in ('global_states','codebook')}==original_plan['input_tensors'],'Original oracle inputs differ')
    old_rows={row['sid']:row for row in read(original_plan['rows_file'])};globals_=[]
    for row in rows:
        old=old_rows[row['base_sid']];fid=scenes[row['sid']]['global_feature_ids'][0]
        need(old['gold']==row['gold'] and old['question']==row['question'] and old['target_ids']==row['target_ids']
             and fid==old['global_feature_id'] and bank['feature_ids'][bank['index'][fid]]==fid,'Reference scene/global ownership differs')
        g=bank['states'][bank['index'][fid]];need(torch.equal(g,old_inputs['global_states'][old['global_row']]),'Original oracle/global bytes differ')
        globals_.append(g)
    globals_=torch.stack(globals_);codes=torch.tensor(code_values,dtype=torch.float32);del bank,code_values
    inputs=dict(schema_version=1,codes=codes,offsets=offsets,sids=[r['sid'] for r in rows],globals=globals_)
    inputs_file=data/'inputs.pt';torch.save(inputs,inputs_file)
    input_tables={k:parent.tensor_info(inputs[k]) for k in ('codes','globals')};input_versions=(codes._version,globals_._version)
    original=torch.load(original_run['checkpoint'],map_location='cpu',weights_only=True);U=original['U']
    need(original['step']==600 and parent.tensor_info(U)==original_run['selected_U'],'Successful answer-code checkpoint differs')
    state=construction(torch,U);state_info={k:parent.tensor_info(v) for k,v in state.items()}
    config=dict(protocol=PROTOCOL,policy=POLICY,source_sha256=frozen,inherited_source_sha256=inherited,orientation_plan=dict(file=str(PLAN),sha256=PLAN_SHA),
        reference_checkpoint=original_run['checkpoint'],reference_checkpoint_sha256=REFERENCE_U_SHA,inputs_file=str(inputs_file),inputs_sha256=sha(inputs_file),input_tensors=input_tables)
    checkpoint=ckpt/'constructed_readout.pt';torch.save(dict(schema_version=1,branch=state,config=config),checkpoint)
    reload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    need(reload['config']==config and {k:parent.tensor_info(v) for k,v in reload['branch'].items()}==state_info,'Restricted constructed checkpoint reload differs')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(24);core=ParallelLocalJointCodeOracle();core.load_state_dict(reload['branch'],strict=True)
    core.eval();core.requires_grad_(False);core_versions={k:v._version for k,v in core.named_parameters()}
    need({k:parent.tensor_info(v) for k,v in core.state_dict().items()}==state_info,'Actual constructed readout differs')
    identity=plan['native_identity'];need(original_plan['native_identity']==identity and object_sha(identity)==plan['native_identity_sha256'],'Reference/native identity differs')
    module_identity=parent.oracle.native_module_identity(torch,identity);need(module_identity==plan['native_module_identity'],'Native implementation differs')
    native_packet=torch.load(plan['native_model_file'],map_location='cpu',weights_only=True)
    norm,head=parent.oracle.native_modules(torch,native_packet,identity,'cpu');del native_packet
    native_before={k:parent.tensor_info(m.weight) for k,m in (('norm',norm),('head',head))};native_versions=(norm.weight._version,head.weight._version)
    calls=[];context={};handles=[]
    def observe(name):
        def hook(module,args,output):
            value=output[0] if name=='core' else output
            calls.append(dict(context,module=name,input_shape=list(args[0].shape),output_shape=list(value.shape),input_dtype=str(args[0].dtype),output_dtype=str(value.dtype)))
        return hook
    for name,module in (('core',core),('norm',norm),('head',head)):handles.append(module.register_forward_hook(observe(name)))
    predictions={'constructed':[],'reference':[]};comparisons=[];batches=[];output_bindings={}
    try:
        with torch.no_grad():
            for start in range(0,216,16):
                stop=min(start+16,216);subset=rows[start:stop];B=stop-start;batch=len(batches)+1;context.clear();context.update(batch=batch,route='constructed')
                local=torch.zeros(16,B,96,dtype=torch.float32);valid=torch.zeros(16,B,dtype=torch.bool)
                for j,index in enumerate(range(start,stop)):
                    n=offsets[index+1]-offsets[index];local[:n,j]=codes[offsets[index]:offsets[index+1]];valid[:n,j]=True
                g=globals_[start:stop];delta,cap=core(local,g,valid_mask=valid,output_dtype=torch.float32,capture=True)
                answer=torch.zeros(B,96,dtype=torch.float32)
                for j,row in enumerate(subset):answer[j,PEOPLE.index(row['gold'])]=math.sqrt(96)
                reference_delta=torch.nn.functional.linear(answer,U)
                expected_pre=torch.nn.functional.linear(cap['aggregate'],reload['branch']['aggregate_projection.weight'])
                need(torch.equal(cap['query'],torch.zeros_like(cap['query'])) and torch.equal(cap['preactivation'],expected_pre)
                     and torch.equal(cap['aggregate'],.5*local.sum(0)) and torch.equal(cap['delta'],delta),'Actual readout arithmetic differs')
                routes={};targets=[r['first_token_id'] for r in subset]
                for name,residual in (('constructed',delta),('reference',reference_delta)):
                    context['route']=name;cast=residual.to(torch.float16);fused=g+cast
                    need(cast.dtype==g.dtype==fused.dtype==torch.float16 and bool(torch.isfinite(fused).all()),'Native cast/add invalid')
                    normalized=norm(fused.unsqueeze(0));logits=head(normalized)
                    need(tuple(logits.shape)==(1,B,152064) and logits.dtype==normalized.dtype==torch.float16 and bool(torch.isfinite(logits).all()),'Native head shape/dtype/finite failure')
                    routes[name]=dict(delta=residual.clone(),cast_delta=cast,fused=fused,normalized=normalized.squeeze(0),logits=logits.squeeze(0))
                packet=dict(schema_version=1,batch=batch,sids=[r['sid'] for r in subset],codes=local,valid_mask=valid,global_states=g.clone(),capture=cap,routes=routes)
                file=data/f'batch_{batch:02d}.pt';torch.save(packet,file);output_bindings[str(file)]=sha(file)
                batch_rows=[]
                for j,row in enumerate(subset):
                    a=routes['constructed']['logits'][j].double();b=routes['reference']['logits'][j].double();tv=float(.5*(a.softmax(-1)-b.softmax(-1)).abs().sum())
                    difference=(delta[j]-reference_delta[j]).abs();close=bool(torch.allclose(delta[j],reference_delta[j],atol=1e-4,rtol=1e-4))
                    relation=dict(sid=row['sid'],residual_max_abs=float(difference.max()),residual_l2=float(difference.double().norm()),residual_close=close,
                        exact_fp16_delta=torch.equal(routes['constructed']['cast_delta'][j],routes['reference']['cast_delta'][j]),
                        full_vocabulary_tv=tv,argmax_exact=int(a.argmax())==int(b.argmax()))
                    relation['passed']=close and tv<=.02 and relation['argmax_exact'];comparisons.append(relation);batch_rows.append(relation)
                    for name,logits in (('constructed',a),('reference',b)):
                        top=logits.topk(2);target=targets[j];nll=torch.logsumexp(logits-logits[target],dim=0)
                        predictions[name].append(dict(sid=row['sid'],gold=row['gold'],target_id=target,argmax_id=int(logits.argmax()),correct=int(logits.argmax())==target,
                            nll_fp64=float(nll),top_two_margin=float(top.values[0]-top.values[1])))
                batches.append(dict(batch=batch,sids=[r['sid'] for r in subset],file=str(file),sha256=output_bindings[str(file)],rows=B))
                save(out/f'batch_{batch:02d}_metrics.json',dict(rows=batch_rows));save(out/f'progress_{batch:02d}.json',dict(completed_rows=stop,calls=calls))
    finally:
        for handle in handles:handle.remove()
        save(out/'calls.json',calls)
    need({k:parent.tensor_info(v) for k,v in core.state_dict().items()}==state_info and {k:v._version for k,v in core.named_parameters()}==core_versions
         and input_versions==(codes._version,globals_._version) and {k:parent.tensor_info(inputs[k]) for k in ('codes','globals')}==input_tables,'Witness parameters/inputs changed')
    need(native_before=={k:parent.tensor_info(m.weight) for k,m in (('norm',norm),('head',head))} and native_versions==(norm.weight._version,head.weight._version)
         and all(not p.requires_grad and p.grad is None for m in (core,norm,head) for p in m.parameters()),'Frozen native/readout weights changed')
    inventory_calls={name:sum(r['module']==name for r in calls) for name in ('core','norm','head')}
    inventory_calls['head_rows']=sum(r['input_shape'][1] for r in calls if r['module']=='head')
    need(inventory_calls==dict(core=14,norm=28,head=28,head_rows=432),'Fixed call inventory differs')
    for call in calls:
        if call['module'] in ('norm','head'):
            B=batches[call['batch']-1]['rows'];need(call['input_shape']==[1,B,3584] and call['input_dtype']==call['output_dtype']=='torch.float16'
                and call['output_shape']==[1,B,152064 if call['module']=='head' else 3584],'Actual native shape/dtype differs')
    for name,values in predictions.items():save(out/(name+'_predictions.json'),values)
    save(out/'comparisons.json',comparisons);save(out/'batches.json',batches);save(out/'checkpoint_roundtrip.json',dict(passed=True,checkpoint=str(checkpoint),sha256=sha(checkpoint),tensor_info=state_info,config=config))
    constructed=score(rows,predictions['constructed']);reference_score=score(rows,predictions['reference'])
    for file in list(out.glob('*.json'))+[inputs_file,checkpoint]:output_bindings[str(file)]=sha(file)
    return dict(passed=True,completed=True,protocol=PROTOCOL,policy=POLICY,source_sha256=frozen,inherited_source_sha256=inherited,
        orientation_plan=dict(file=str(PLAN),sha256=PLAN_SHA),reference_report=dict(file=str(REFERENCE),sha256=REFERENCE_SHA),
        native_identity=identity,native_identity_sha256=plan['native_identity_sha256'],native_module_identity=module_identity,
        constructed=constructed,reference=reference_score,capacity_witness_passed=constructed['passed'] and reference_score['passed'] and all(r['passed'] for r in comparisons),
        numerical_comparison=dict(all_passed=all(r['passed'] for r in comparisons),maximum_tv=max(r['full_vocabulary_tv'] for r in comparisons),
            maximum_residual_error=max(r['residual_max_abs'] for r in comparisons),argmax_disagreements=sum(not r['argmax_exact'] for r in comparisons)),
        calls=inventory_calls,input_bindings=bindings,output_bindings=output_bindings,constructed_checkpoint=str(checkpoint),constructed_checkpoint_sha256=sha(checkpoint),
        runtime=dict(torch_version=str(torch.__version__),python_version=platform.python_version(),platform=platform.platform(),node=platform.node(),threads=torch.get_num_threads()),
        no_training_or_native_release=True,privileged_capacity_only=True,whole_answer_claim=False,backend_universal_identity_claim=False)


def main():
    need(len(sys.argv)==1,'Fixed witness has no tunable arguments');parent.native.require_slurm(gpu=False)
    need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS') and os.environ.get('SLURM_JOB_NAME')=='identity_join_orientation_capacity'
         and int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==4,'Fixed CPU allocation required')
    started=time.perf_counter();name='witness_'+os.environ['SLURM_JOB_ID'];out=OUT/name;data=DATA/name;ckpt=CKPT/name
    for path in (out,data,ckpt):path.mkdir(parents=True,exist_ok=False)
    frozen,inherited=snapshot(out);bindings={};save(out/'request.json',dict(protocol=PROTOCOL,policy=POLICY,source_sha256=frozen,inherited_source_sha256=inherited))
    try:
        result=run(out,data,ckpt,frozen,inherited,bindings);need(sources()==frozen and inherited_sources()==inherited,'Witness source changed')
        result['elapsed_seconds']=time.perf_counter()-started;need(result['elapsed_seconds']<=300,'Fixed CPU witness cap exceeded')
        save(out/'analysis.json',result)
        (out/'REPORT.md').write_text('# Finite-precision symmetric-join capacity witness\n\n'
            f"Constructed first-token accuracy: {result['constructed']['correct']}/216; reference: {result['reference']['correct']}/216. "
            f"Capacity witness passed: {result['capacity_witness_passed']}.\n\n"
            'The existing readout receives separate privileged local codes; no fitting occurs. The reference explicitly receives the gold answer code. '
            'Both use the same frozen native global and selected-query CPU norm/head. All216 rows, raw outputs and finite-precision differences are retained. '
            'This establishes only a finite-domain parameter existence result if its gates pass. It does not establish learnability, visual information sufficiency, '
            'whole-answer completion, generalization or universal CPU/GPU identity. Prior failed fits and native/confirmation gates remain unchanged.\n')
        summary=dict(passed=result['capacity_witness_passed'],completed=True,protocol=PROTOCOL,source_sha256=frozen,inherited_source_sha256=inherited,
            capacity_witness_passed=result['capacity_witness_passed'],orientation_plan=result['orientation_plan'],native_identity_sha256=result['native_identity_sha256'],
            analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),report_file=str(out/'REPORT.md'),report_sha256=sha(out/'REPORT.md'),elapsed_seconds=result['elapsed_seconds'])
        save(out/'summary.json',summary);print(json.dumps(dict(directory=str(out),passed=summary['passed'])),flush=True)
        need(summary['passed'],'Finite-precision capacity witness did not pass; all evidence retained')
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,inherited_source_sha256=inherited,input_bindings=bindings,
            elapsed_seconds=time.perf_counter()-started,partial_outputs_retained=True));raise


if __name__=='__main__':main()

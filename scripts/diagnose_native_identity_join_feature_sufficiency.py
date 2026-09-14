"""One fixed CPU ridge assay of training-only native local feature accessibility.

No model/head, runtime probe, gate, task fit, hyperparameter search or retry.
Input features and every offline label/split are frozen before the sole data fit.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[1]
DATA=Path('/mnt/data/gabriele/gnn_transformer/identity_join_feature_sufficiency')
CKPT=Path('/mnt/ckpts/gabriele/gnn_transformer/identity_join_feature_sufficiency')
OUT=REPO/'outputs/native_aggregation_vlm/identity_join_feature_sufficiency'
CACHE=Path('/mnt/data/gabriele/gnn_transformer/identity_join_learned/features/merge_443031/feature_cache.json')
CACHE_SHA='db060aed422aa05459d3fde9f512daf44bcab447c1426ff8c5be111764190d46'
MERGE=REPO/'outputs/native_aggregation_vlm/identity_join_learned/features/merge_443031/summary.json'
MERGE_SHA='a5b3bdc6fca74d423dc0327b913d0601c98619fb0195b2c19999321022b71d04'
PLAN_SHA='cd8f92fdceb23f69bf52be3c8487a1c722780b2b1444435cd019ae8fcd3514bd'
PROPOSAL='docs/paper/NATIVE_AGGREGATION_FEATURE_SUFFICIENCY_PROPOSAL.md'
PROPOSAL_SHA='bfc7ed371e803d230cde4166d90aab33d95afabf58a5f112e2067ce1bdcdb6ec'
OWN=('scripts/diagnose_native_identity_join_feature_sufficiency.py',
     'slurm/native_identity_join_feature_sufficiency.sbatch',PROPOSAL)
POLICY=dict(protocol='native_identity_join_training_feature_sufficiency',rows=9007,
    fit_rows=4472,evaluation_rows=4535,fit_steps=[1,3,5,7,9,11,13,15],
    evaluation_steps=[2,4,6,8,10,12,14,16],width=3584,person_classes=9,room_classes=6,
    normalization='FP32 h*rsqrt(mean(h**2)+1e-6), then FP64',ridge_lambda=1.,
    loss='mean_rows(sum_15_squared_errors)+sum_W_squared',intercept_penalized=False,
    solver='single FP64 Cholesky',relative_residual_max=1e-8,maximum_seconds=600,
    marginal_correct_min=4490,joint_correct_min=4490,question_joint_min=.98,
    person_room_joint_min=.95,question_majority_advantage_min=.50,
    no_runtime_use=True,no_training_release=True,no_gpu_model_vision_head_calls=True)


def need(condition,message):
    if not condition:raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def oid(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def tensor_info(torch,value):
    value=value.detach().cpu().contiguous()
    return dict(shape=list(value.shape),dtype=str(value.dtype),
        sha256=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest())


def bound(path,bindings,expected=None):
    path=Path(path).resolve();h=sha(path)
    need(expected is None or h==expected,'Bound artifact changed: '+str(path))
    bindings[str(path)]=h;return read(path)


def source_snapshot(out,cache):
    need(sha(REPO/PROPOSAL)==PROPOSAL_SHA,'Fixed proposal changed')
    sources={**cache['source_sha256'],**{p:sha(REPO/p) for p in OWN}}
    (out/'source').mkdir()
    for name,digest in sources.items():
        need(sha(REPO/name)==digest,'Frozen ancestor/current source differs')
        target=out/'source'/name.replace('/','_');target.write_bytes((REPO/name).read_bytes())
        need(sha(target)==digest,'Source snapshot differs')
    save(out/'source_hashes.json',sources);return sources


def verify_inputs(bindings):
    cache=bound(CACHE,bindings,CACHE_SHA);proof=bound(MERGE,bindings,MERGE_SHA)
    need(proof['passed'] is True and proof['completed'] is True and proof['cache_file']==str(CACHE)
         and proof['cache_sha256']==CACHE_SHA and proof['feature_count']==64847
         and proof['source_sha256']==cache['source_sha256'],'Passed exact merge proof required')
    need(cache['complete'] is True and cache['training_only'] is True
         and cache['protocol']=='identity_join_learned_training_feature_cache'
         and cache['state_dtype']=='torch.float16','Native training-only cache differs')
    plan=bound(cache['plan_file'],bindings,PLAN_SHA)
    need(cache['plan_sha256']==PLAN_SHA and cache['scenes']==plan['scenes']
         and cache['source_sha256']==plan['source_sha256'] and len(cache['features'])==64847
         and set(cache['features'])==set(plan['features'])
         and cache['native_identity']==plan['native_identity']
         and oid(cache['native_identity'])==cache['native_identity_sha256']==plan['native_identity_sha256'],
         'Cache plan/source/native identity differs')
    stage=REPO/'outputs/native_aggregation_vlm/identity_join_learned/features/stage_443005'
    stage_proof=bound(stage/'summary.json',bindings)
    need(stage_proof['passed'] is True and stage_proof['completed'] is True
         and stage_proof['plan_sha256']==PLAN_SHA and stage_proof['source_sha256']==cache['source_sha256'],
         'Feature preparation proof differs')
    for name,digest in cache['source_sha256'].items():
        need(sha(REPO/name)==digest and sha(stage/'source'/name.replace('/','_'))==digest
             and sha(MERGE.parent/'source'/name.replace('/','_'))==digest,'Ancestor source snapshot differs')
    manifest=bound(plan['dataset_manifest_file'],bindings,plan['dataset_manifest_sha256'])
    rendered=bound(plan['dataset_summary'],bindings)
    need(rendered['passed'] is True and rendered['completed'] is True
         and rendered['manifest_sha256']==plan['dataset_manifest_sha256'],'Renderer proof differs')
    atoms=bound(manifest['render_cache_file'],bindings,manifest['render_cache_sha256'])
    samples=bound(manifest['samples_file'],bindings,manifest['samples_sha256'])
    bound(manifest['plan_file'],bindings,manifest['plan_sha256'])
    bound(manifest['audit_file'],bindings,manifest['audit_sha256'])
    need(manifest['binary_gate_dependency'] is False and manifest['policy']['local_question_identical_to_global'] is True,
         'Wrong model-input interface')
    profile=bound(Path(cache['profile_directory'])/'summary.json',bindings,cache['profile_summary_sha256'])
    need(profile['passed'] is True and profile['completed'] is True
         and profile['native_identity']==cache['native_identity'],'Native feature profile differs')
    shard_summaries={}
    for entry in cache['shards']:
        s=bound(Path(entry['directory'])/'summary.json',bindings,entry['summary_sha256'])
        need(s['passed'] is True and s['completed'] is True and s['plan_sha256']==PLAN_SHA
             and s['source_sha256']==cache['source_sha256'],'Shard metadata differs')
        shard_summaries[s['features_file']]=s
    need(len(shard_summaries)==4,'Exactly four feature parts required')
    return cache,plan,manifest,atoms,samples,shard_summaries


def metadata_inventory(cache,plan,manifest,atoms,samples):
    people=manifest['policy']['people'];rooms=manifest['policy']['rooms']
    need(len(people)==len(set(people))==9 and len(rooms)==len(set(rooms))==6,'Frozen class order differs')
    training=manifest['splits']['train_N8']['samples']+manifest['splits']['train_N16']['samples']
    symbolic={row['sid']:row for row in samples if row['split']=='train'}
    need(len(training)==len(symbolic)==len(cache['scenes'])==6048
         and {r['sid'] for r in training}==set(cache['scenes'])==set(symbolic),'Exact training scene union differs')
    candidates=defaultdict(list)
    for atom in atoms.values():candidates[atom['sha256']].append(atom)
    failures=defaultdict(list);occurrences=Counter();occurrence_rows=[];image_atoms={}
    local_ids=set();global_ids=set();questions=sorted({s['question'] for s in training})
    for row in training:
        sid=row['sid'];scene=cache['scenes'][sid];semantic=symbolic[sid];n=row['n_frames']
        need(row['split']=='train' and n in (8,16) and scene['n_frames']==n
             and row['question']==scene['question']==semantic['question']
             and row['gold']==scene['gold']==semantic['gold']
             and row['target_ids']==scene['target_ids'] and len(row['image_files'])==n
             and len(scene['local_feature_ids'])==n,'Training metadata ownership differs')
        gid=scene['global_feature_ids'][0];global_ids.add(gid)
        need(plan['features'][gid]['kind']=='global' and plan['features'][gid]['prefix_ids']==[]
             and plan['features'][gid]['question']==row['question'],'Wrong collision-only global feature')
        for i,(image,ids) in enumerate(zip(row['image_files'],scene['local_feature_ids'])):
            fid=ids[0];local_ids.add(fid);occurrences[fid]+=1;desc=plan['features'][fid];pair=plan['pairs'][desc['pair_id']]
            need(desc['kind']=='local' and desc['phase']=='local_empty' and desc['prefix_ids']==[]
                 and desc['question']==pair['question']==row['question'] and pair['image_sha256']==image['sha256']
                 and desc['pair_id']==oid([image['sha256'],row['question']])
                 and fid==oid(['local',desc['pair_id'],[]]),'Local image/question/empty-prefix identity differs')
            entries=candidates.get(image['sha256'],[])
            if not entries:failures['missing_atoms'].append(dict(sid=sid,position=i,image_sha256=image['sha256']))
            valid_atoms=all(isinstance(x.get('atom'),list) and len(x['atom'])==3
                and isinstance(x['atom'][0],str) and isinstance(x['atom'][1],str)
                and type(x['atom'][2]) is int for x in entries)
            unique={tuple(x['atom']) for x in entries} if valid_atoms else set()
            if len(unique)!=1:
                failures['ambiguous_atoms'].append(dict(sid=sid,position=i,atoms=[x.get('atom') for x in entries]));continue
            person,room,step=next(iter(unique))
            if person not in people:failures['unknown_people'].append(dict(sid=sid,position=i,value=person))
            if room not in rooms:failures['unknown_rooms'].append(dict(sid=sid,position=i,value=room))
            state=semantic['states'][i];occupied=[(p,r) for r,persons in state['rooms'].items() for p in persons]
            if len(occupied)!=1:failures['ambiguous_occupants'].append(dict(sid=sid,position=i,occupants=occupied))
            if person not in people or room not in rooms or len(occupied)!=1:continue
            need(type(step) is int and step==i+1==state['step_id'] and 1<=step<=16
                 and semantic['frames'][i]==[person,room] and occupied==[(person,room)]
                 and all(x['canonical_rgb_parity'] is True and x['mode']=='RGB' for x in entries),
                 'Canonical Step/occupant/rendered atom differs')
            previous=image_atoms.setdefault(image['sha256'],(person,room,step))
            need(previous==(person,room,step),'Repeated image has conflicting canonical labels')
            occurrence_rows.append(dict(sid=sid,position=i,feature_id=fid,image_sha256=image['sha256'],step=step))
    need(local_ids==set(plan['groups']['local_empty']) and len(local_ids)==9007
         and len(global_ids)==12 and len(questions)==12 and sum(occurrences.values())==72576,
         'Exact training empty-feature/occurrence union differs')
    inventory=[]
    for fid in sorted(local_ids):
        d=plan['features'][fid];p=plan['pairs'][d['pair_id']];atom=image_atoms.get(p['image_sha256'])
        person,room,step=atom if atom is not None else (None,None,None)
        inventory.append(dict(feature_id=fid,question=d['question'],image_sha256=p['image_sha256'],
            image_path=p['image_path'],person=person,room=room,step=step,
            partition=None if step is None else 'fit' if step%2 else 'evaluate',
            occurrence_weight=occurrences[fid],state=cache['features'][fid]))
    errors={k:failures.get(k,[]) for k in ('missing_atoms','ambiguous_atoms','unknown_people','unknown_rooms','ambiguous_occupants')}
    return inventory,occurrence_rows,sorted(global_ids),people,rooms,questions,errors


def grouped(rows,fields):
    out=defaultdict(list)
    for row in rows:out[tuple(row[k] for k in fields)].append(row)
    return out


def coverage(rows,people,rooms,questions):
    result={};image_sets={}
    for part,count in (('fit',4472),('evaluate',4535)):
        subset=[r for r in rows if r['partition']==part];images={r['image_sha256'] for r in subset};image_sets[part]=images
        need(len(subset)==count and len(images)==432 and {r['question'] for r in subset}==set(questions)
             and sum(r['occurrence_weight'] for r in subset)==36288,'Odd/even fixed coverage differs')
        result[part]=dict(features=count,images=len(images),occurrences=36288,tables={})
        for fields in (('question',),('person',),('room',),('person','room'),('question','person','room'),('step',)):
            table=[]
            for key,values in sorted(grouped(subset,fields).items()):
                table.append(dict(zip(fields,key),features=len(values),images=len({r['image_sha256'] for r in values}),
                    steps=sorted({r['step'] for r in values}),occurrences=sum(r['occurrence_weight'] for r in values)))
            result[part]['tables']['/'.join(fields)]=table
        cells=grouped(subset,('question','person','room'))
        need(set(cells)==set(itertools.product(questions,people,rooms)) and len(cells)==648
             and all(3<=len({r['step'] for r in v})<=8 and len(v)==len({r['image_sha256'] for r in v}) for v in cells.values()),
             'Complete 648 three-way cell coverage differs')
    need(not image_sets['fit']&image_sets['evaluate'] and len(image_sets['fit']|image_sets['evaluate'])==864
         and {r['step'] for r in rows}==set(range(1,17)),'Image/Step holdout overlap differs')
    return result


def load_states(torch,cache,plan,parts,local_ids,global_ids,bindings):
    selected=set(local_ids)|set(global_ids);states={};consumed=[]
    for filename,summary in sorted(parts.items()):
        need(sha(filename)==summary['features_sha256'],'Consumed feature part changed');bindings[filename]=summary['features_sha256']
        blob=torch.load(filename,map_location='cpu',weights_only=True);tensor=blob['states']
        need(tensor.dtype==torch.float16 and tensor.ndim==2 and tensor.shape[1]==3584
             and tensor_info(torch,tensor)==summary['states'] and blob['feature_ids']==plan['shards'][summary['shard']],
             'Native feature part tensor/index differs')
        for i,fid in enumerate(blob['feature_ids']):
            if fid not in selected:continue
            row=tensor[i].clone();location=cache['features'][fid]
            need(location==dict(file=filename,file_sha256=summary['features_sha256'],row=i,state_sha256=summary['feature_state_sha256'][fid])
                 and tensor_info(torch,row)['sha256']==location['state_sha256'] and bool(torch.isfinite(row).all()),
                 'Consumed native FP16 row/hash/ownership differs')
            states[fid]=row
        consumed.append(dict(file=filename,sha256=summary['features_sha256'],tensor=summary['states'],
            selected_rows=sum(fid in selected for fid in blob['feature_ids'])))
        del blob,tensor
    need(set(states)==selected,'Selected local/collision-global tensor coverage differs')
    return states,consumed


def collision_checks(torch,rows,scenes,states,cache):
    local_groups=defaultdict(list)
    for row in rows:local_groups[(row['question'],row['state']['state_sha256'])].append(row)
    local=[]
    for (question,digest),values in sorted(local_groups.items()):
        if len(values)<2:continue
        base=states[values[0]['feature_id']]
        need(all(torch.equal(base,states[r['feature_id']]) for r in values),'Hash collision is not an exact state collision')
        labels=Counter((r['person'],r['room']) for r in values)
        local.append(dict(question=question,state_sha256=digest,feature_ids=[r['feature_id'] for r in values],
            size=len(values),labels=[dict(person=p,room=r,count=n) for (p,r),n in sorted(labels.items())],conflicting_labels=len(labels)>1))
    scene_groups=defaultdict(list)
    for sid,scene in scenes.items():
        ids=[x[0] for x in scene['local_feature_ids']];g=scene['global_feature_ids'][0]
        multiset=tuple(sorted(Counter(cache['features'][fid]['state_sha256'] for fid in ids).items()))
        key=(scene['question'],cache['features'][g]['state_sha256'],multiset)
        scene_groups[key].append((sid,ids,g,scene['gold']))
    whole=[]
    for (question,global_sha,multiset),values in sorted(scene_groups.items()):
        if len(values)<2:continue
        ordered=lambda value:sorted(value[1],key=lambda fid:cache['features'][fid]['state_sha256'])
        base=values[0];base_ids=ordered(base)
        for value in values[1:]:
            ids=ordered(value)
            need(torch.equal(states[base[2]],states[value[2]]) and len(ids)==len(base_ids)
                 and all(torch.equal(states[a],states[b]) for a,b in zip(base_ids,ids)),
                 'Candidate multiset failed exact tensor/multiplicity confirmation')
        answers=Counter(value[3] for value in values)
        whole.append(dict(question=question,global_state_sha256=global_sha,local_state_multiset=list(multiset),
            sids=[v[0] for v in values],size=len(values),answer_counts=dict(answers),conflicting_answers=len(answers)>1))
    return dict(local_candidate_groups=local,local_conflicting_groups=sum(g['conflicting_labels'] for g in local),
        scene_candidate_groups=whole,scene_conflicting_groups=sum(g['conflicting_answers'] for g in whole),
        exact_tensor_confirmation=True,multiplicity_preserved=True,
        global_features_used_only_for_collision_check=True,scope='Stored training FP16 interface only')


def fit_ridge(torch,x,y):
    need(x.dtype==y.dtype==torch.float64 and x.ndim==y.ndim==2 and x.shape[0]==y.shape[0], 'FP64 ridge matrix contract differs')
    m=x.shape[0];xm=x.mean(0);ym=y.mean(0);xc=x-xm;yc=y-ym
    a=xc.T@xc/m+torch.eye(x.shape[1],dtype=torch.float64);rhs=xc.T@yc/m
    factor=torch.linalg.cholesky(a);w=torch.cholesky_solve(rhs,factor);b=ym-xm@w
    denominator=float(torch.linalg.vector_norm(rhs));scale=max(denominator,torch.finfo(torch.float64).tiny)
    matrix_residual=float(torch.linalg.vector_norm(a@w-rhs))/scale
    # Reconstruct the normal-equation residual from centered observations,
    # independently of the Gram matrix supplied to Cholesky.
    data_residual=float(torch.linalg.vector_norm(xc.T@(xc@w-yc)/m+w))/scale
    errors=x@w+b-y;data_loss=float(errors.square().sum()/m);penalty=float(w.square().sum())
    diagnostics=dict(matrix_relative_residual=matrix_residual,data_relative_residual=data_residual,
        residual_denominator=denominator,data_squared_error_per_row=data_loss,penalty=penalty,
        objective=data_loss+penalty,intercept_gradient_norm=float(torch.linalg.vector_norm(errors.mean(0))),
        lambda_value=1.,rows=m,output_columns=y.shape[1],column_averaging=False,solver_calls=1)
    need(all(math.isfinite(v) for v in (matrix_residual,data_residual,data_loss,penalty))
         and bool(torch.isfinite(w).all()) and bool(torch.isfinite(b).all()),'Nonfinite ridge result')
    return dict(weight=w,bias=b,x_mean=xm,y_mean=ym),diagnostics


def majority(rows,people,rooms,questions):
    fit=[r for r in rows if r['partition']=='fit'];out={}
    for key,part in [('global',fit)]+[(q,[r for r in fit if r['question']==q]) for q in questions]:
        labels={};counts={};ties={}
        for field,order in (('person',people),('room',rooms)):
            c=Counter(r[field] for r in part);values=[c[v] for v in order];largest=max(values)
            labels[field]=order[values.index(largest)];counts[field]=values;ties[field]=sum(v==largest for v in values)
        out[key]=dict(predictions=labels,fit_counts=counts,tied_maxima=ties,fit_rows=len(part))
    return out


def score_table(rows,method,fields=()):
    result=[]
    for key,values in sorted(grouped(rows,fields).items()):
        row=dict(zip(fields,key));row.update(rows=len(values),occurrences=sum(r['occurrence_weight'] for r in values))
        for name in ('person','room','joint'):
            correct=sum(r[method][name+'_correct'] for r in values)
            weighted=sum(r['occurrence_weight']*r[method][name+'_correct'] for r in values)
            row[name]=dict(correct=correct,total=len(values),accuracy=correct/len(values),
                occurrence_correct=weighted,occurrence_total=row['occurrences'],occurrence_accuracy=weighted/row['occurrences'])
        result.append(row)
    return result


def evaluate(torch,x,rows,model,people,rooms,questions):
    scores=x@model['weight']+model['bias'];need(bool(torch.isfinite(scores).all()),'Nonfinite prediction scores')
    p=scores[:,:9];r=scores[:,9:];pi=p.argmax(-1);ri=r.argmax(-1)
    pt=(p==p.max(-1,keepdim=True).values).sum(-1);rt=(r==r.max(-1,keepdim=True).values).sum(-1)
    controls=majority(rows,people,rooms,questions);predictions=[]
    for i,row in enumerate(rows):
        result=dict(row,scores=scores[i].tolist(),person_tied_maxima=int(pt[i]),room_tied_maxima=int(rt[i]))
        for name,labels in [('ridge',dict(person=people[int(pi[i])],room=rooms[int(ri[i])])),
                            ('global_majority',controls['global']['predictions']),('question_majority',controls[row['question']]['predictions'])]:
            a=labels['person']==row['person'];b=labels['room']==row['room']
            result[name]=dict(**labels,person_correct=a,room_correct=b,joint_correct=a and b)
        predictions.append(result)
    metrics={}
    for part in ('fit','evaluate'):
        values=[r for r in predictions if r['partition']==part];metrics[part]={}
        for method in ('ridge','global_majority','question_majority'):
            metrics[part][method]=dict(overall=score_table(values,method)[0],tables={
                '/'.join(fields):score_table(values,method,fields) for fields in
                [('question',),('person',),('room',),('person','room'),('question','person','room'),('step',)]})
        metrics[part]['ties']=dict(person_rows=sum(v['person_tied_maxima']>1 for v in values),room_rows=sum(v['room_tied_maxima']>1 for v in values))
    e=metrics['evaluate']['ridge'];base=metrics['evaluate']['question_majority']['overall']['joint']['accuracy']
    decisions=dict(person=e['overall']['person']['correct']>=4490,room=e['overall']['room']['correct']>=4490,
        joint=e['overall']['joint']['correct']>=4490,
        every_question=all(r['joint']['correct']*100>=98*r['rows'] for r in e['tables']['question']),
        every_person_room=all(r['joint']['correct']*100>=95*r['rows'] for r in e['tables']['person/room']),
        question_majority_advantage=e['overall']['joint']['accuracy']-base>=.5)
    decisions['strong_linear_accessibility']=all(decisions.values())
    decisions['diagnostic_only']=True;decisions['native_training_released']=False
    return predictions,controls,metrics,decisions


def selftest(torch):
    x=torch.tensor([[-1.,2.],[0.,2.],[1.,2.]],dtype=torch.float64)
    y=torch.tensor([[0.],[1.],[2.]],dtype=torch.float64);model,d=fit_ridge(torch,x,y)
    need(torch.allclose(model['weight'],torch.tensor([[.4],[0.]],dtype=torch.float64),rtol=0,atol=1e-14)
         and torch.allclose(model['bias'],torch.tensor([1.],dtype=torch.float64),rtol=0,atol=1e-14)
         and d['matrix_relative_residual']<1e-12 and d['data_relative_residual']<1e-12,'Centered/unpenalized intercept ridge fixture failed')
    duplicate,dd=fit_ridge(torch,x,y.repeat(1,2))
    need(torch.allclose(duplicate['weight'][:,0],model['weight'][:,0],atol=1e-14,rtol=0)
         and math.isclose(dd['objective'],2*d['objective'],rel_tol=1e-14),'Output-column sum fixture failed')
    rows=[dict(partition='fit',question='q',person='a',room='r'),dict(partition='fit',question='q',person='b',room='s'),
          dict(partition='evaluate',question='q',person='b',room='s')]
    control=majority(rows,['a','b'],['r','s'],['q'])
    need(control['q']['predictions']==dict(person='a',room='r') and control['q']['tied_maxima']==dict(person=2,room=2),'Fit-only majority/tie fixture failed')
    t=torch.zeros((2,3),dtype=torch.float16);norm=t.float()*torch.rsqrt(t.float().square().mean(-1,keepdim=True)+1e-6)
    need(torch.equal(norm,torch.zeros_like(norm)) and norm.dtype==torch.float32,'FP32 RMS zero fixture failed')
    # Same set with unequal multiplicities must not collide; true duplicate
    # states with conflicting labels and equal full multisets must be reported.
    states={'a':torch.tensor([1.,2.],dtype=torch.float16),'b':torch.tensor([1.,2.],dtype=torch.float16),
            'c':torch.tensor([3.,4.],dtype=torch.float16),'g':torch.tensor([0.,0.],dtype=torch.float16)}
    cache={'features':{k:dict(state_sha256=tensor_info(torch,v)['sha256']) for k,v in states.items()}}
    labels=[dict(question='q',feature_id=k,state=cache['features'][k],person=p,room='r') for k,p in [('a','x'),('b','y'),('c','x')]]
    scenes={sid:dict(question='q',local_feature_ids=[[k] for k in ids],global_feature_ids=['g'],gold=gold)
        for sid,ids,gold in [('s1',['a','a','c'],'x'),('s2',['b','b','c'],'y'),('s3',['a','c','c'],'z')]}
    collisions=collision_checks(torch,labels,scenes,states,cache)
    need(collisions['local_conflicting_groups']==1 and collisions['scene_conflicting_groups']==1
         and collisions['scene_candidate_groups'][0]['size']==2,'Exact collision/multiplicity fixture failed')
    rows=[dict(occurrence_weight=9,m=dict(person_correct=True,room_correct=True,joint_correct=True)),
          dict(occurrence_weight=1,m=dict(person_correct=False,room_correct=False,joint_correct=False))]
    score=score_table(rows,'m')[0]
    need(score['joint']['accuracy']==.5 and score['joint']['occurrence_accuracy']==.9,'Unique/occurrence denominator fixture failed')
    return dict(passed=True,groups=6,scope='Ridge centering/intercept/column sum; fit-only ties; RMS; exact collisions/multiplicity; independent weighting')


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and os.environ.get('SLURM_JOB_ID')
         and not os.environ.get('SLURM_JOB_GPUS') and int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==4,
         'Run only on Slurm CPU4; no local numerical execution')
    job=os.environ['SLURM_JOB_ID'];need(job.isdigit(),'Concrete Slurm job ID required')
    started=time.perf_counter();out=OUT/job;data=DATA/job;ckpt=CKPT/job
    out.mkdir(parents=True,exist_ok=False);sources={};bindings={};timings={}
    try:
        need(sha(CACHE)==CACHE_SHA,'Canonical frozen cache changed');sources=source_snapshot(out,read(CACHE))
        data.mkdir(parents=True,exist_ok=False);ckpt.mkdir(parents=True,exist_ok=False)
        save(out/'request.json',dict(policy=POLICY,source_sha256=sources,cache_file=str(CACHE),cache_sha256=CACHE_SHA,
            merge_file=str(MERGE),merge_sha256=MERGE_SHA,job_id=job,data_directory=str(data),checkpoint_directory=str(ckpt)))
        cache,plan,manifest,atoms,samples,parts=verify_inputs(bindings)
        rows,occurrences,global_ids,people,rooms,questions,errors=metadata_inventory(cache,plan,manifest,atoms,samples)
        save(data/'input_inventory.json',dict(rows=rows,people=people,rooms=rooms,questions=questions,metadata_errors=errors,
            global_collision_only_feature_ids=global_ids,policy=POLICY))
        save(data/'occurrence_inventory.json',occurrences);save(out/'metadata_errors.json',errors)
        need(not any(errors.values()),'Unknown/missing/ambiguous metadata; no row may be dropped')
        panel=coverage(rows,people,rooms,questions);save(out/'coverage.json',panel);save(out/'input_bindings_before_tensors.json',bindings)
        timings['metadata_and_source_seconds']=time.perf_counter()-started
        import torch
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        tests=selftest(torch);save(out/'selftest.json',tests);phase=time.perf_counter()
        states,consumed=load_states(torch,cache,plan,parts,[r['feature_id'] for r in rows],global_ids,bindings)
        collisions=collision_checks(torch,rows,cache['scenes'],states,cache);save(data/'collisions.json',collisions)
        h=torch.stack([states[r['feature_id']] for r in rows]);del states
        x=h.float();x=(x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)).double();del h
        need(x.shape==(9007,3584) and bool(torch.isfinite(x).all()),'RMS design matrix differs')
        fit=torch.tensor([i for i,r in enumerate(rows) if r['partition']=='fit'],dtype=torch.long)
        y=torch.zeros((9007,15),dtype=torch.float64)
        for i,r in enumerate(rows):y[i,people.index(r['person'])]=1;y[i,9+rooms.index(r['room'])]=1
        input_metadata=dict(design=tensor_info(torch,x),targets=tensor_info(torch,y),fit_indices=tensor_info(torch,fit),
            inventory_file=str(data/'input_inventory.json'),inventory_sha256=sha(data/'input_inventory.json'),
            occurrence_inventory_file=str(data/'occurrence_inventory.json'),occurrence_inventory_sha256=sha(data/'occurrence_inventory.json'),
            collisions_file=str(data/'collisions.json'),collisions_sha256=sha(data/'collisions.json'),consumed_parts=consumed,
            input_bindings=bindings,native_identity=cache['native_identity'],native_identity_sha256=cache['native_identity_sha256'])
        save(out/'resolved_inputs.json',input_metadata)
        timings['tensor_load_validation_collision_seconds']=time.perf_counter()-phase;phase=time.perf_counter()
        model,solve=fit_ridge(torch,x[fit],y[fit]);timings['ridge_solve_seconds']=time.perf_counter()-phase
        save(out/'solve.json',solve)
        payload=dict(schema_version=1,**model,people=people,rooms=rooms,policy=POLICY,source_sha256=sources,
            resolved_inputs_file=str(out/'resolved_inputs.json'),resolved_inputs_sha256=sha(out/'resolved_inputs.json'),solve=solve)
        torch.save(payload,ckpt/'ridge.pt');model_metadata={k:tensor_info(torch,v) for k,v in model.items()}
        need(solve['matrix_relative_residual']<=1e-8 and solve['data_relative_residual']<=1e-8,'Independent ridge residual exceeds1e-8; no retry')
        checkpoint_sha=sha(ckpt/'ridge.pt')
        restored=torch.load(ckpt/'ridge.pt',map_location='cpu',weights_only=True)
        need(set(restored)==set(payload) and all(restored[k]==v for k,v in payload.items() if k not in model)
             and all(torch.equal(restored[k],v) and tensor_info(torch,restored[k])==model_metadata[k]
                     for k,v in model.items()) and sha(ckpt/'ridge.pt')==checkpoint_sha,
             'Restricted ridge checkpoint roundtrip tensor/metadata identity failed')
        model={k:restored[k] for k in model}
        roundtrip=dict(passed=True,file=str(ckpt/'ridge.pt'),sha256=checkpoint_sha,tensors=model_metadata,
            all_policy_source_and_resolved_input_metadata_exact=True,predictions_use_reloaded_tensors=True)
        save(out/'checkpoint_roundtrip.json',roundtrip)
        phase=time.perf_counter();predictions,controls,metrics,decisions=evaluate(torch,x,rows,model,people,rooms,questions)
        save(data/'predictions.json',predictions);save(out/'controls.json',controls);save(out/'metrics.json',metrics)
        timings['prediction_and_publication_seconds']=time.perf_counter()-phase
        need(all(sha(REPO/n)==v for n,v in sources.items()),'Source changed during diagnostic')
        elapsed=time.perf_counter()-started;need(elapsed<=600,'Fixed CPU time cap exceeded')
        files={str(p):sha(p) for p in [data/'input_inventory.json',data/'occurrence_inventory.json',data/'collisions.json',
            data/'predictions.json',ckpt/'ridge.pt',out/'coverage.json',out/'controls.json',out/'metrics.json',out/'resolved_inputs.json',out/'solve.json',out/'selftest.json',out/'checkpoint_roundtrip.json']}
        summary=dict(passed=True,completed=True,protocol=POLICY['protocol'],job_id=job,policy=POLICY,source_sha256=sources,
            input_bindings=bindings,files=files,coverage=panel,decisions=decisions,
            ridge_checkpoint_file=str(ckpt/'ridge.pt'),ridge_checkpoint_sha256=sha(ckpt/'ridge.pt'),model_tensors=model_metadata,
            solve=solve,checkpoint_roundtrip=roundtrip,timings=timings,seconds=time.perf_counter()-started,tests=tests,
            local_conflicting_groups=collisions['local_conflicting_groups'],scene_conflicting_groups=collisions['scene_conflicting_groups'],
            fit_metrics=metrics['fit']['ridge']['overall'],evaluation_metrics=metrics['evaluate']['ridge']['overall'],
            question_majority_evaluation=metrics['evaluate']['question_majority']['overall'],
            forced_known_class_predictions=True,unknown_prediction_detection_not_available=True,
            calls=dict(model=0,vision=0,head=0,gpu=0),native_training_released=False,
            interpretation='Linear accessibility on held Step images only; ridge failure without exact conflicts does not establish absent information')
        save(out/'summary.json',summary)
        e=summary['evaluation_metrics'];lines=['# Frozen local-feature sufficiency diagnostic','',
            'Completed one fixed CPU ridge fit. This readout is offline only and releases no native training.','',
            f"Held even-Step rows: person {e['person']['correct']}/4535; room {e['room']['correct']}/4535; joint {e['joint']['correct']}/4535.",
            f"Strong linear-accessibility diagnostic: {decisions['strong_linear_accessibility']}.",
            f"Exact conflicting groups: local {collisions['local_conflicting_groups']}; scene multisets {collisions['scene_conflicting_groups']}.",
            '', 'All 9007 rows, 648 cells per partition, occurrence weights, errors, ties and controls are retained.',
            'The split holds out Step values/images, not people, rooms, artwork or task semantics. Scores are not probabilities.',
            'Ridge failure without exact conflicting collisions is inconclusive about missing information.',
            '', '[Summary](summary.json) · [Metrics](metrics.json) · [Coverage](coverage.json) · [Sources](source_hashes.json)']
        (out/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(passed=True,directory=str(out),decisions=decisions)),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),job_id=job,source_sha256=sources,
            input_bindings=bindings,timings=timings,seconds=time.perf_counter()-started,partial_evidence_retained=True,no_retry=True));raise


if __name__=='__main__':main()

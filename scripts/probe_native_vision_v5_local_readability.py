"""Conditional CPU-only readability of frozen V5 final-prompt frame messages.

Run only after the completed canonical V5 primary fails. This is an external
binary conjunction probe, not a new counting method or additional VLM training.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time

REPO=Path(__file__).resolve().parents[1]
BASE=REPO/'outputs/native_aggregation_vlm/v5'
OUT=BASE/'local_readability'
DATA=Path('/mnt/data/gabriele/gnn_transformer')
CKPTS=Path('/mnt/ckpts/gabriele/gnn_transformer/v5_local_readability')
OWN=('scripts/probe_native_vision_v5_local_readability.py','slurm/native_aggregation_vision_v5_local_readability.sbatch')
SEED=20260918
LAMBDA=.001
CLASSES=('positive','char_only','room_only','neither')
CELLS=('length_N16','length_N32','length_N64','unseen_count_N32','unseen_count_N64')


def ensure(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def sources():return {name:sha(REPO/name) for name in OWN}


def split_families(ids):
    ensure(len(ids)==108 and len(set(ids))==108,'Exactly108 distinct familiar families required')
    shuffled=sorted(ids);random.Random(SEED).shuffle(shuffled)
    return shuffled[:54],shuffled[54:]


def require_disjoint(fit,held):
    ensure(not set(fit)&set(held),'A family occurs in both fitting and held-out sets')


def semantic_labels(row):
    path=Path(row['path'])/'qa.txt';payload=path.read_bytes()
    ensure(hashlib.sha256(payload).hexdigest()==row['qa_sha256'],'QA hash differs')
    lines=payload.decode().splitlines();start=next(i for i,x in enumerate(lines) if x.strip()=='question:')
    end=next(i for i,x in enumerate(lines) if x.strip()=='answer:')
    block=[x.strip() for x in lines[start+1:end] if x.strip()]
    states=[ast.literal_eval(x) for x in block if x.startswith('{') and x.endswith('}')]
    questions=[x for x in block if not(x.startswith('{') and x.endswith('}'))]
    ensure(len(states)==row['n_frames'] and len(questions)==1,'Malformed frame/question record')
    match=re.fullmatch(r'How many frames show ([A-Za-z]+) in the ([A-Za-z]+)\?',questions[0])
    ensure(match is not None,'Unknown question law');character,room=match.groups()
    ensure(row.get('target_character',character)==character and row.get('target_room',room)==room,'Question target differs')
    if 'question' in row:ensure(row['question']==questions[0],'Manifest question differs')
    labels=[]
    for i,state in enumerate(states):
        ensure(state['step_id']==i+1,'Frame order differs')
        occupants=[(person,place) for place,people in state['rooms'].items() for person in people]
        ensure(len(occupants)==1,'Expected one character per frame')
        person,place=occupants[0]
        labels.append('positive' if person==character and place==room else
                      'char_only' if person==character else 'room_only' if place==room else 'neither')
    gold=int(next(x.strip() for x in lines[end+1:] if x.strip()))
    ensure(labels.count('positive')==gold==row['gold'],'Independent conjunction recount differs')
    normalized=dict(states=states,question=' '.join(questions[0].split()))
    content=hashlib.sha256(json.dumps(normalized,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    ensure(content==row['content_sha256'],'Semantic content hash differs')
    return labels


def fit_ridge(np,x,y):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    ensure(x.ndim==2 and y.shape==(len(x),) and np.isfinite(x).all(),'Invalid fit arrays')
    positive=y==1;negative=y==-1
    ensure((positive|negative).all() and positive.any() and negative.any(),'Both binary classes required')
    mean=x.mean(axis=0);scale=np.maximum(x.std(axis=0,ddof=0),1e-6)
    z=(x-mean)/scale;design=np.column_stack((z,np.ones(len(x),dtype=np.float64)))
    weights=np.where(positive,.5/positive.sum(),.5/negative.sum())
    penalty=np.eye(design.shape[1],dtype=np.float64)*LAMBDA;penalty[-1,-1]=0
    matrix=design.T@(weights[:,None]*design)+penalty
    rhs=design.T@(weights*y)
    coefficient=np.linalg.solve(matrix,rhs)
    ensure(np.isfinite(coefficient).all(),'Nonfinite ridge solution')
    residual=design@coefficient-y
    return dict(mean=mean,scale=scale,weight=coefficient[:-1],intercept=coefficient[-1],
        fit_positive=int(positive.sum()),fit_negative=int(negative.sum()),
        weighted_mse=float(weights@(residual*residual)),
        penalized_objective=float(weights@(residual*residual)+LAMBDA*(coefficient[:-1]@coefficient[:-1])),
        normal_equation_residual_max=float(np.abs(matrix@coefficient-rhs).max()))


def auc(np,labels,scores):
    labels=np.asarray(labels,dtype=bool);scores=np.asarray(scores,dtype=np.float64)
    n_positive=int(labels.sum());n_negative=len(labels)-n_positive
    if not n_positive or not n_negative:return None
    order=np.argsort(scores,kind='stable');ordered=scores[order]
    starts=np.r_[0,np.flatnonzero(ordered[1:]!=ordered[:-1])+1];ends=np.r_[starts[1:],len(ordered)]
    average_ranks=np.repeat((starts+1+ends)/2,ends-starts)
    rank_sum=average_ranks[labels[order]].sum()
    return float((rank_sum-n_positive*(n_positive+1)/2)/(n_positive*n_negative))


def binary_metrics(np,labels,scores):
    labels=np.asarray(labels,dtype=bool);scores=np.asarray(scores,dtype=np.float64)
    ensure(labels.shape==scores.shape and np.isfinite(scores).all(),'Invalid evaluation arrays')
    prediction=scores>=0
    tp=int((prediction&labels).sum());fn=int((~prediction&labels).sum())
    fp=int((prediction&~labels).sum());tn=int((~prediction&~labels).sum())
    tpr=tp/(tp+fn) if tp+fn else None;fpr=fp/(fp+tn) if fp+tn else None
    return dict(n=len(labels),positive=tp+fn,negative=fp+tn,tp=tp,fn=fn,fp=fp,tn=tn,
        tpr=tpr,fpr=fpr,balanced_accuracy=(tpr+1-fpr)/2 if tpr is not None and fpr is not None else None,
        auroc=auc(np,labels,scores),threshold=0.,threshold_rule='score >= 0 predicts conjunction',
        exactly_zero_scores=int((scores==0).sum()),
        auc_note='Unweighted frame-level positive-negative rank statistic; exact score ties contribute one half. Undefined if either class is absent.')


def summarize(np,records):
    scores=np.asarray([r['score'] for r in records],dtype=np.float64)
    labels=np.asarray([r['semantic_class']=='positive' for r in records],dtype=bool)
    result=binary_metrics(np,labels,scores)
    result['contexts']=len({r['sid'] for r in records});result['families']=len({r['family_id'] for r in records})
    result['semantic_confusion']={category:dict(n=sum(r['semantic_class']==category for r in records),
        predicted_positive=sum(r['semantic_class']==category and r['score']>=0 for r in records),
        predicted_negative=sum(r['semantic_class']==category and r['score']<0 for r in records)) for category in CLASSES}
    result['per_negative_class']={}
    for category in CLASSES[1:]:
        chosen=[r for r in records if r['semantic_class'] in ('positive',category)]
        result['per_negative_class'][category]=binary_metrics(np,
            [r['semantic_class']=='positive' for r in chosen],[r['score'] for r in chosen])
    return result


def self_tests(np):
    ids=[f'family_{i:03d}' for i in range(108)]
    fit,held=split_families(ids);fit2,held2=split_families(list(reversed(ids)))
    ensure((fit,held)==(fit2,held2) and len(fit)==len(held)==54,'Family split is order-dependent')
    require_disjoint(fit,held)
    try:require_disjoint(fit,[fit[0],*held])
    except ValueError:pass
    else:raise AssertionError('Family leakage not rejected')
    assignments={(family,n):('fit' if family in set(fit) else 'held') for family in ids for n in (16,32,64)}
    ensure(all(len({assignments[(family,n)] for n in (16,32,64)})==1 for family in ids),'One family crosses folds')
    x=np.array([[7.,-2.],[7.,-1.],[7.,0.],[7.,3.]])
    y=np.array([-1.,-1.,-1.,1.]);fitted=fit_ridge(np,x,y)
    ensure(fitted['scale'][0]==1e-6 and abs(fitted['weight'][0])<1e-12,'Constant feature not handled')
    z=(x-fitted['mean'])/fitted['scale'];design=np.column_stack((z,np.ones(4)))
    weights=np.where(y==1,.5,.5/3)
    regularizer=np.sqrt(LAMBDA)*np.column_stack((np.eye(2),np.zeros(2)))
    augmented=np.vstack((np.sqrt(weights)[:,None]*design,regularizer))
    target=np.r_[np.sqrt(weights)*y,np.zeros(2)]
    reference=np.linalg.lstsq(augmented,target,rcond=None)[0]
    ensure(np.allclose(np.r_[fitted['weight'],fitted['intercept']],reference,rtol=1e-10,atol=1e-10),
        'Weighted ridge/unpenalized intercept differs from augmented least squares')
    constant=fit_ridge(np,np.ones((4,2)),y)
    ensure(abs(constant['intercept'])<1e-12 and np.all(constant['weight']==0),'Class balance/intercept wrong')
    ensure(auc(np,[1,1,0,0],[0,1,0,1])==.5 and auc(np,[1,1,0,0],[1,1,0,0])==1
        and auc(np,[1,1,0,0],[0,0,1,1])==0 and auc(np,[1,1],[0,1]) is None,'AUROC/ties/undefined case wrong')
    zero=binary_metrics(np,[1,1,0,0],[0,0,0,0])
    ensure(zero['tp']==zero['fp']==2 and zero['balanced_accuracy']==.5 and zero['auroc']==.5,'Zero threshold/ties wrong')


def load_features(np,torch,row):
    path=Path(row['branch_diagnostics_path'])
    ensure(sha(path)==row['branch_diagnostics_sha256'],'Frozen message tensor hash differs')
    value=torch.load(path,map_location='cpu',weights_only=True)
    ensure(value['frame_messages'].shape==(row['n_frames'],96) and bool(torch.isfinite(value['frame_messages']).all()),'Invalid96-dimensional messages')
    ensure(bool(value['visible'].all()) and int(value['query_position'])==row['prompt_tokens']-1
        and torch.equal(value['visible'],value['query_position']>value['image_ends']), 'Wrong last-prompt query or image visibility')
    return value['frame_messages'].double().numpy()


def run(np,torch,ledger):
    started=time.monotonic();analysis_path=BASE/'analysis.json';analysis=read(analysis_path)
    ensure(analysis['criteria']['primary_both_seeds'] is False and analysis['criteria']['V5_1'] is False,
        'Conditional readability audit is authorized only after the completed V5 primary fails')
    analysis_hash=sha(analysis_path)
    for name,digest in analysis['analysis_source_sha256'].items():
        ensure(sha(REPO/name)==digest and sha(BASE/'analysis_code'/name.replace('/','_'))==digest,'Canonical analysis sources changed')
    manifests={name:read(DATA/'v4_diversity'/name) for name in ('main_manifest.json','count_manifest.json')}
    manifest_hashes={name:sha(DATA/'v4_diversity'/name) for name in manifests}
    source={}
    for cell in CELLS:
        family,n=cell.rsplit('_N',1)
        manifest=manifests['main_manifest.json' if family=='length' else 'count_manifest.json']
        records=manifest['splits'][f'test_N{n}']['samples']
        ensure(len(records)==(108 if family=='length' else 64),'Unexpected canonical cell size')
        for record in records:
            key=(cell,record['sid']);ensure(key not in source,'Repeated source sample')
            source[key]=dict(record,cell=cell)
    families={r['pair_id'] for r in source.values() if r['cell']=='length_N16'}
    fit_ids,held_ids=split_families(list(families));require_disjoint(fit_ids,held_ids)
    fit_set,held_set=set(fit_ids),set(held_ids)
    for family in families:
        group=[r for r in source.values() if r['cell'].startswith('length_') and r['pair_id']==family]
        ensure(len(group)==3 and {r['n_frames'] for r in group}=={16,32,64} and len({r['gold'] for r in group})==1,'Familiar family alignment differs')
    labels={key:semantic_labels(record) for key,record in source.items()}
    job=os.environ['SLURM_JOB_ID'];output=OUT/f'run_{job}';coefficient_dir=CKPTS/f'run_{job}'
    score_dir=DATA/'v5_local_readability'/f'run_{job}'
    for directory in (output,coefficient_dir,score_dir):directory.mkdir(parents=True,exist_ok=False)
    write(output/'family_split.json',dict(seed=SEED,fit_families=fit_ids,held_families=held_ids,
        fit_contexts='54 familiar N16 only',held_contexts='54 disjoint families atN16/N32/N64',
        unused='Fit-family N32/N64 contexts are excluded from scoring',secondary='All128 unseen-count contexts, separately reported'))
    provenance=dict(analysis_path=str(analysis_path),analysis_sha256=analysis_hash,
        analysis_source_sha256=analysis['analysis_source_sha256'],manifest_sha256=manifest_hashes,
        source_sha256=sources(),source_ledger_sha256=sha(ledger),slurm_job_id=job,
        numpy_version=np.__version__,torch_version=str(torch.__version__),runs={})
    all_results={}
    for condition in ('sum','mean'):
        for seed in (4,5):
            key=f'{condition}_seed{seed}';run_dir=Path(analysis['runs'][condition][str(seed)]).resolve()
            ensure(run_dir.parent==BASE/'main'/condition/f'seed{seed}','Noncanonical main run')
            config=read(run_dir/'config.json');summary=read(run_dir/'summary.json');history=read(run_dir/'training.json')
            info=analysis['conditions'][condition][str(seed)]
            ensure(config['run_id']==run_dir.name and config['condition']==condition and config['seed']==seed and not config['profile'],'Wrong run identity')
            ensure(config['code_sha256']==analysis['training_source_sha256'],'Main source ledger differs')
            for name,digest in config['code_sha256'].items():ensure(sha(REPO/name)==digest,'Frozen training source changed')
            ensure(config['manifest_sha256']==info['manifest_sha256']==manifest_hashes['main_manifest.json']
                and config['count_manifest_sha256']==info['count_manifest_sha256']==manifest_hashes['count_manifest.json'],'Dataset hashes differ')
            ensure(history==summary['training'] and len(history)==9,'Incomplete training')
            best=max(history,key=lambda e:(sum(m['correct'] for m in e['dev'])/72,
                -sum(m['gold_first_token_nll']*m['n'] for m in e['dev'])/72))
            checkpoint=Path(summary['selected_checkpoint']).resolve()
            ensure(checkpoint==Path('/mnt/ckpts/gabriele/gnn_transformer/native_aggregation_vlm_v5')/run_dir.name/'best.pt'
                and str(checkpoint)==info['selected_checkpoint'] and sha(checkpoint)==info['selected_checkpoint_sha256'],'Selected checkpoint path/hash differs')
            saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
            ensure(saved['architecture']==config['architecture'] and saved['epoch']==best['epoch']==info['selected_epoch']
                and saved['dev']==best['dev'] and saved['step']==best['step'],'Selected checkpoint differs from dev selection')
            del saved
            rows=read(run_dir/'predictions.json');ensure(len(rows)==452,'Expected all452 original predictions')
            ensure({(r['cell'],r['sid']) for r in rows}==set(source),'Prediction/manifests differ')
            diagnostic_records={r['path']:r['sha256'] for r in info['prefill_diagnostics']['records']}
            ensure(len(diagnostic_records)==452,'Canonical analysis did not bind all message files')
            for row in rows:
                original=source[(row['cell'],row['sid'])]
                ensure(all(row[k]==original[k] for k in ('path','n_frames','gold','pair_id')) and row['mode']=='all' and row['tag']=='test','Frame context identity differs')
                path=Path(row['branch_diagnostics_path'])
                ensure(path.resolve()==DATA/'v5_diagnostics'/run_dir.name/f'{row["sid"]}.pt'
                    and diagnostic_records[str(path)]==row['branch_diagnostics_sha256']==sha(path),'Canonical message path/hash differs')
            fit_rows=sorted([r for r in rows if r['cell']=='length_N16' and r['pair_id'] in fit_set],key=lambda r:r['pair_id'])
            held_rows=sorted([r for r in rows if r['cell'].startswith('length_') and r['pair_id'] in held_set],key=lambda r:(r['n_frames'],r['pair_id']))
            unseen_rows=sorted([r for r in rows if r['cell'].startswith('unseen_count_')],key=lambda r:(r['n_frames'],r['pair_id']))
            ensure(len(fit_rows)==54 and len(held_rows)==162 and len(unseen_rows)==128,'Probe context cardinalities differ')
            require_disjoint([r['pair_id'] for r in fit_rows],[r['pair_id'] for r in held_rows])
            x=np.concatenate([load_features(np,torch,row) for row in fit_rows],axis=0)
            y=np.array([1. if label=='positive' else -1. for row in fit_rows for label in labels[(row['cell'],row['sid'])]],dtype=np.float64)
            fitted=fit_ridge(np,x,y);ensure(x.shape==(864,96),'Fit must contain864 frames/96features')
            coefficient_path=coefficient_dir/f'{key}.npz'
            np.savez(coefficient_path,mean=fitted['mean'],scale=fitted['scale'],weight=fitted['weight'],
                intercept=np.float64(fitted['intercept']),ridge_lambda=np.float64(LAMBDA),fit_family_ids=np.array(fit_ids),
                checkpoint_sha256=np.array(info['selected_checkpoint_sha256']),analysis_sha256=np.array(analysis_hash))
            scored=[]
            for subset,selected in (('held_familiar',held_rows),('unseen_count',unseen_rows)):
                for row in selected:
                    features=load_features(np,torch,row)
                    scores=((features-fitted['mean'])/fitted['scale'])@fitted['weight']+fitted['intercept']
                    for index,(label,score) in enumerate(zip(labels[(row['cell'],row['sid'])],scores)):
                        scored.append(dict(subset=subset,cell=row['cell'],sid=row['sid'],family_id=row['pair_id'],
                            n_frames=row['n_frames'],gold=row['gold'],frame_index=index,semantic_class=label,
                            target=1 if label=='positive' else -1,score=float(score),prediction=1 if score>=0 else -1))
            score_path=score_dir/f'{key}.jsonl'
            with score_path.open('w') as stream:
                for row in scored:stream.write(json.dumps(row,allow_nan=False)+'\n')
            groups={cell:[r for r in scored if r['cell']==cell] for cell in CELLS}
            result=dict(condition=condition,seed=seed,selected_epoch=best['epoch'],checkpoint=str(checkpoint),
                checkpoint_sha256=info['selected_checkpoint_sha256'],coefficient_path=str(coefficient_path),coefficient_sha256=sha(coefficient_path),
                score_path=str(score_path),score_sha256=sha(score_path),
                fit=dict(contexts=54,frames=864,positive=fitted['fit_positive'],negative=fitted['fit_negative'],
                    weighted_mse=fitted['weighted_mse'],penalized_objective=fitted['penalized_objective'],
                    normal_equation_residual_max=fitted['normal_equation_residual_max'],zero_variance_features=int((x.std(axis=0)==0).sum())),
                primary_held_familiar_ood=summarize(np,[r for r in scored if r['subset']=='held_familiar' and r['n_frames'] in (32,64)]),
                cells={cell:summarize(np,group) for cell,group in groups.items()},
                unseen_count_separate=summarize(np,[r for r in scored if r['subset']=='unseen_count']),
                per_N_K={cell:{str(k):summarize(np,[r for r in group if r['gold']==k]) for k in sorted({r['gold'] for r in group})}
                    for cell,group in groups.items()})
            all_results[key]=result;write(output/f'{key}.json',result)
            provenance['runs'][key]=dict(run_directory=str(run_dir),config_sha256=sha(run_dir/'config.json'),
                summary_sha256=sha(run_dir/'summary.json'),predictions_sha256=sha(run_dir/'predictions.json'),
                checkpoint=str(checkpoint),checkpoint_sha256=info['selected_checkpoint_sha256'],
                diagnostic_tensor_sha256={r['branch_diagnostics_path']:r['branch_diagnostics_sha256'] for r in rows})
            print(json.dumps(dict(run=key,held_ood=result['primary_held_familiar_ood'])),flush=True)
            del x,y,fitted,scored
    ensure(sha(analysis_path)==analysis_hash,'Canonical analysis changed during audit')
    for name,digest in manifest_hashes.items():ensure(sha(DATA/'v4_diversity'/name)==digest,'Manifest changed during audit')
    write(output/'provenance.json',provenance)
    summary=dict(schema_version=1,conditional_trigger='completed canonical V5 primary false',
        protocol=dict(seed=SEED,fit_families=54,held_families=54,fit_N=16,features=96,ridge_lambda=LAMBDA,
            objective='sum_i [1/(2*n_class_i)]*(y_i-z_i dot w-b)^2 + .001*||w||^2; intercept unpenalized',
            normalization='Unweighted population mean/std on this checkpoint fit frames only; std floor1e-6',
            solver='FP64 direct normal-equation solve',threshold='score>=0',targets='conjunction+1; other three semantic categories-1'),
        results=all_results,elapsed_seconds=time.monotonic()-started,
        limitations=['External supervised diagnostic, not a deployed count method, VLM update, or aggregation improvement.',
            'Low linear readability does not establish information absence; higher readability does not establish native use or accurate counting.',
            'Frame-level metrics are descriptive and correlated within contexts/families; no confidence or pass/fail claim is attached.',
            'Messages depend on the contextual final language query. The probe cannot isolate independent visual memory from query encoding.',
            'Unseen-count contexts are a separately labeled secondary domain; no fitting or calibration uses them.'])
    write(output/'summary.json',summary)
    lines=['# Frozen V5 message readability','',
        'Conditional diagnostic after the V5 primary failed. One fixed binary class-balanced ridge per development-selected checkpoint; no new VLM training.','',
        '| Checkpoint | Held N32 AUROC / TPR / FPR | Held N64 AUROC / TPR / FPR | Unseen-count AUROC |',
        '|---|---|---|---|']
    fmt=lambda value:'NA' if value is None else f'{value:.4f}'
    for key,result in all_results.items():
        triples=[' / '.join(fmt(result['cells'][f'length_N{n}'][m]) for m in ('auroc','tpr','fpr')) for n in (32,64)]
        lines.append('| '+key+' | '+' | '.join(triples)+' | '+fmt(result['unseen_count_separate']['auroc'])+' |')
    lines+=['','Each held familiar cell has54 contexts; fit uses54 separateN16 contexts. All128 unseen-count contexts remain separate. Confusions, denominators, negative classes and perN/K results are in the JSON reports.','',
        'Low linear scores do not prove information loss; strong scores do not prove that the model uses these messages or counts correctly. Frame-level metrics are descriptive, without confidence or a new pass/fail threshold.','',
        '[Summary](summary.json) · [Family split](family_split.json) · [Provenance](provenance.json)']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (output/'INDEX.md').write_text('# V5 conditional local readability\n\n[Report](REPORT.md) · [Summary](summary.json) · [Provenance](provenance.json).\n')
    print(json.dumps(dict(output=str(output),elapsed_seconds=summary['elapsed_seconds'])),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true');parser.add_argument('--source-ledger',type=Path)
    args=parser.parse_args()
    ensure(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu'
        and not os.environ.get('SLURM_JOB_GPUS',''),'Requires Slurm CPU-only allocation')
    import numpy as np
    import torch
    torch.set_num_threads(max(1,min(4,int(os.environ.get('SLURM_CPUS_PER_TASK','1')))))
    self_tests(np)
    if args.self_test:
        output=OUT/f'check_{os.environ["SLURM_JOB_ID"]}';output.mkdir(parents=True,exist_ok=False)
        write(output/'source_hashes.json',sources());(output/'code').mkdir()
        for name in OWN:(output/'code'/name.replace('/','_')).write_bytes((REPO/name).read_bytes())
        (output/'INDEX.md').write_text('# Readability software checks\n\nFamily leakage, weighted ridge/intercept, zero-variance features and AUROC ties passed. [Frozen sources](source_hashes.json).\n')
        print(json.dumps(dict(passed=True,source_ledger=str(output/'source_hashes.json'))),flush=True)
    else:
        ensure(args.source_ledger is not None and read(args.source_ledger)==sources(),'Pass the prospectively frozen source ledger')
        run(np,torch,args.source_ledger)


if __name__=='__main__':main()

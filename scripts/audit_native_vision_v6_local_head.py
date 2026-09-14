"""Apply every dev-selected V6 auxiliary head to all saved fresh-test messages.

No fitting, threshold search, teacher inference, VLM forward, or external tally.
The ordinary FP32 affine96-to-3 head and softmax operate on the saved final
original-prompt messages. Argmax classes are0,1,other (ties use the first index).
Other is always incorrect against binary conjunction truth, including negatives;
therefore balanced accuracy uses P(pred0|gold0), not one minus false-positive rate.
All452contexts/18240frames per checkpoint remain included regardless of native
count correctness. Frame outputs are descriptive, correlated within families.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts.probe_native_vision_v5_local_readability import semantic_labels,auc,CELLS,CLASSES

BASE=REPO/'outputs/native_aggregation_vlm/v6'
OUT=BASE/'local_head'
DATA=Path('/mnt/data/gabriele/gnn_transformer')
FRESH=DATA/'v6_fresh'
CKPTS=Path('/mnt/ckpts/gabriele/gnn_transformer/native_aggregation_vlm_v6')
OWN=('scripts/audit_native_vision_v6_local_head.py',
     'slurm/native_aggregation_vision_v6_local_head.sbatch',
     'scripts/probe_native_vision_v5_local_readability.py')


def need(value,message):
    if not value:raise ValueError(message)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def sources():return {name:sha(REPO/name) for name in OWN}


def snapshot(out):
    (out/'code').mkdir()
    for name in OWN:(out/'code'/name.replace('/','_')).write_bytes((REPO/name).read_bytes())
    write(out/'source_hashes.json',sources())


def metrics(np,truth,predictions,score1):
    truth=np.asarray(truth,dtype=np.int64);predictions=np.asarray(predictions,dtype=np.int64)
    score1=np.asarray(score1,dtype=np.float64)
    need(truth.shape==predictions.shape==score1.shape and truth.ndim==1 and
         bool(np.isin(truth,[0,1]).all()) and bool(np.isin(predictions,[0,1,2]).all()) and
         bool(np.isfinite(score1).all()) and bool(((score1>=0)&(score1<=1)).all()),'Invalid local-head scores')
    confusion=np.zeros((2,3),dtype=np.int64)
    for gold,prediction in zip(truth,predictions):confusion[gold,prediction]+=1
    negative,positive=(int(confusion[i].sum()) for i in (0,1));n=negative+positive
    correct=int(confusion[0,0]+confusion[1,1]);other=int(confusion[:,2].sum())
    tpr=float(confusion[1,1]/positive) if positive else None
    fpr=float(confusion[0,1]/negative) if negative else None
    negative_recall=float(confusion[0,0]/negative) if negative else None
    return dict(n=n,positive=positive,negative=negative,confusion_2x3=confusion.tolist(),
                confusion_rows=['truth0','truth1'],confusion_columns=['pred0','pred1','other'],
                correct=correct,three_way_accuracy=correct/n if n else None,
                tpr=tpr,fpr=fpr,negative_recall=negative_recall,
                balanced_accuracy=(tpr+negative_recall)/2 if positive and negative else None,
                other=other,other_rate=other/n if n else None,
                positive_other=int(confusion[1,2]),negative_other=int(confusion[0,2]),
                positive_other_rate=float(confusion[1,2]/positive) if positive else None,
                negative_other_rate=float(confusion[0,2]/negative) if negative else None,
                auroc_p1=auc(np,truth==1,score1),
                score='Unconditional FP32 softmax probability ofclass1, including argmax-other rows',
                rule='FP32 logit argmax over0,1,other; ties choose first index; other is always wrong')


def summarize(np,records):
    result=metrics(np,[r['truth'] for r in records],[r['prediction'] for r in records],
                   [r['probabilities'][1] for r in records])
    result.update(contexts=len({r['sid'] for r in records}),families=len({r['family_id'] for r in records}))
    result['semantic_confusion']={category:dict(
        n=sum(r['semantic_class']==category for r in records),
        predictions_0_1_other=[sum(r['semantic_class']==category and r['prediction']==value for r in records)
                              for value in (0,1,2)]) for category in CLASSES}
    result['per_negative_class']={}
    for category in CLASSES[1:]:
        selected=[r for r in records if r['semantic_class'] in ('positive',category)]
        result['per_negative_class'][category]=metrics(np,[r['truth'] for r in selected],
            [r['prediction'] for r in selected],[r['probabilities'][1] for r in selected])
    return result


def apply_head(torch,messages,head):
    need(messages.dtype==torch.float32 and messages.ndim==2 and messages.shape[1]==96 and
         bool(torch.isfinite(messages).all()),'Expected finite saved FP32 image-by96 messages')
    need(set(head)=={'weight','bias'} and head['weight'].shape==(3,96) and head['bias'].shape==(3,) and
         all(value.dtype==torch.float32 and value.device.type=='cpu' and bool(torch.isfinite(value).all())
             for value in head.values()),'Saved auxiliary head is not finite FP32 affine96-to-3')
    with torch.inference_mode():
        logits=torch.nn.functional.linear(messages,head['weight'],head['bias'])
        probabilities=torch.softmax(logits,dim=-1)
        predictions=logits.argmax(dim=-1)
    need(bool(torch.isfinite(logits).all()) and bool(torch.isfinite(probabilities).all()),'Nonfinite auxiliary output')
    return logits,probabilities,predictions


def self_test(np,torch):
    all_other=metrics(np,[0,0,0,1],[2,2,2,2],[.05]*4)
    need(all_other['confusion_2x3']==[[0,0,3],[0,0,1]] and all_other['three_way_accuracy']==0 and
         all_other['balanced_accuracy']==0 and all_other['fpr']==0 and all_other['other_rate']==1 and
         all_other['auroc_p1']==.5,'Other was incorrectly credited as a negative or excluded from AUROC')
    head=dict(weight=torch.zeros(3,96,dtype=torch.float32),bias=torch.zeros(3,dtype=torch.float32))
    _,prob,pred=apply_head(torch,torch.ones(4,96,dtype=torch.float32),head)
    need(pred.tolist()==[0]*4,'Argmax ties do not use first native index')
    tied=metrics(np,[0,0,1,1],pred.tolist(),prob[:,1].tolist())
    need(tied['auroc_p1']==.5 and tied['balanced_accuracy']==.5 and tied['three_way_accuracy']==.5,
         'Tied scores/predictions mishandled')
    records=[]
    for i,(category,prediction,probabilities) in enumerate(zip(CLASSES,[1,0,1,2],
            [[.1,.8,.1],[.8,.1,.1],[.1,.8,.1],[.05,.05,.9]])):
        records.append(dict(sid='s',family_id='f',semantic_class=category,truth=int(category=='positive'),
                            prediction=prediction,probabilities=probabilities))
    mixed=summarize(np,records)
    need(mixed['confusion_2x3']==[[1,1,1],[0,1,0]] and mixed['three_way_accuracy']==.5 and
         math.isclose(mixed['balanced_accuracy'],2/3) and math.isclose(mixed['fpr'],1/3) and
         mixed['semantic_confusion']['neither']['predictions_0_1_other']==[0,0,1],
         'Negative-class confusion/abstention accounting differs')
    need(mixed['per_negative_class']['char_only']['balanced_accuracy']==1 and
         mixed['per_negative_class']['room_only']['balanced_accuracy']==.5 and
         mixed['per_negative_class']['neither']['balanced_accuracy']==.5,'Negative-category denominators differ')
    single=metrics(np,[0,0],[0,2],[.1,.1])
    need(single['tpr'] is None and single['balanced_accuracy'] is None and single['auroc_p1'] is None,
         'Absent-class AUROC/balanced accuracy must be undefined')
    wrong=dict(head,weight=torch.zeros(3,95))
    try:apply_head(torch,torch.ones(4,96),wrong)
    except ValueError:pass
    else:raise AssertionError('Malformed selected head accepted')
    return dict(passed=True,tests=['all_other_wrong','unconditional_auc_ties','argmax_ties',
        'negative_category_confusions','missing_class_denominators','selected_head_shape'])


def load_sources():
    source={};bindings={};hash_cache={}
    for family,filename in (('length','main_manifest.json'),('unseen_count','count_manifest.json')):
        path=FRESH/filename;manifest=read(path);bindings[filename]=sha(path)
        need(manifest['schema_version']==1 and Path(manifest['dataset_root']).resolve()==FRESH,'Wrong fresh manifest')
        need(set(manifest['splits'])==({'test_N16','test_N32','test_N64'} if family=='length' else
                                      {'test_N32','test_N64'}),'Fresh manifest cells differ')
        for split,cell in manifest['splits'].items():
            n=int(split.split('_N')[1]);name=f'{family}_N{n}'
            expected=Counter({k:12 for k in range(9)}) if family=='length' else Counter({k:8 for k in range(9,17)})
            need(Counter(r['gold'] for r in cell['samples'])==expected,'Fresh N/K balance differs')
            for row in cell['samples']:
                directory=Path(row['path']);key=(name,row['sid'])
                need(key not in source and row['split']=='test' and row['n_frames']==n and
                     directory.resolve()==FRESH/'mmred_vfiltered'/f'seq_len_{n}'/'test'/row['sid'],
                     'Wrong fresh context identity/path')
                labels=semantic_labels(row)
                need(len(row['image_files'])==n,'Fresh image coverage differs')
                for index,image in enumerate(row['image_files']):
                    image_path=Path(image['path']);stat=image_path.stat()
                    need(image_path==directory/f'{index:03d}.png' and stat.st_size==image['bytes'],
                         'Fresh frame order/size differs')
                    inode=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
                    if inode not in hash_cache:hash_cache[inode]=sha(image_path)
                    need(hash_cache[inode]==image['sha256'],'Fresh image bytes differ from manifest')
                source[key]=dict(row,cell=name,semantic_labels=labels)
    need(len(source)==452 and len({r['sid'] for r in source.values()})==452 and
         sum(r['n_frames'] for r in source.values())==18240,'Expected all452contexts/18240frames')
    for family,lengths,count in (('length',{16,32,64},108),('unseen_count',{32,64},64)):
        groups={}
        for row in source.values():
            if row['cell'].startswith(family+'_'):groups.setdefault(row['pair_id'],[]).append(row)
        need(len(groups)==count,'Wrong family count')
        for group in groups.values():
            need({r['n_frames'] for r in group}==lengths and len(group)==len(lengths) and
                 len({(r['gold'],r['question']) for r in group})==1,'Incomplete/mismatched paired family')
    return source,bindings


def load_messages(torch,row,run,canonical):
    path=Path(row['branch_diagnostics_path']);n=row['n_frames']
    need(path.resolve()==DATA/'v6_diagnostics'/run.name/f"{row['sid']}.pt" and
         canonical[str(path)]==row['branch_diagnostics_sha256']==sha(path),'Canonical message tensor path/hash differs')
    value=torch.load(path,map_location='cpu',weights_only=True)
    need(value['frame_messages'].shape==(n,96) and value['frame_messages'].dtype==torch.float32 and
         bool(torch.isfinite(value['frame_messages']).all()),'Wrong saved FP32 message matrix')
    need(value['visible'].shape==(n,) and value['visible'].dtype==torch.bool and bool(value['visible'].all()) and
         value['image_ends'].shape==(n,) and value['image_ends'].dtype==torch.int64 and
         value['query_position'].shape==() and value['query_position'].dtype==torch.int64 and
         int(value['query_position'])==row['prompt_tokens']-1 and
         bool((value['image_ends'][1:]>value['image_ends'][:-1]).all()) and
         torch.equal(value['visible'],value['query_position']>value['image_ends']),
         'Messages do not describe every image at the last original prompt query')
    return value['frame_messages']


def run(np,torch,ledger):
    started=time.monotonic();analysis_path=BASE/'analysis.json';analysis=read(analysis_path);analysis_sha=sha(analysis_path)
    need(set(analysis['runs'])=={'control','aligned'} and set(analysis['conditions'])=={'control','aligned'},
         'Require complete canonical V6 comparison')
    for name,digest in analysis['analysis_source_sha256'].items():
        need(sha(REPO/name)==digest and sha(BASE/'analysis_code'/name.replace('/','_'))==digest,
             'Canonical V6 report sources changed')
    source,manifest_hashes=load_sources()
    job=os.environ['SLURM_JOB_ID'];out=OUT/f'run_{job}';out.mkdir(parents=True,exist_ok=False);snapshot(out)
    data_out=DATA/'v6_local_head'/f'run_{job}';data_out.mkdir(parents=True,exist_ok=False)
    provenance=dict(analysis_path=str(analysis_path),analysis_sha256=analysis_sha,
        analysis_source_sha256=analysis['analysis_source_sha256'],training_source_sha256=analysis['training_source_sha256'],
        source_sha256=sources(),source_ledger_sha256=sha(ledger),manifest_sha256=manifest_hashes,
        torch_version=str(torch.__version__),numpy_version=np.__version__,slurm_job_id=job,runs={})
    results={}
    for condition in ('control','aligned'):
        need(set(analysis['runs'][condition])=={'6','7'} and set(analysis['conditions'][condition])=={'6','7'},
             'Both registered V6 seeds must be complete')
        for seed in (6,7):
            key=f'{condition}_seed{seed}';info=analysis['conditions'][condition][str(seed)]
            run_dir=Path(analysis['runs'][condition][str(seed)]).resolve()
            need(run_dir.parent==BASE/'main'/condition/f'seed{seed}','Noncanonical V6 run')
            config=read(run_dir/'config.json');summary=read(run_dir/'summary.json');history=read(run_dir/'training.json')
            need(config['run_id']==summary['run_id']==run_dir.name and config['condition']==summary['condition']==condition and
                 config['seed']==seed and config['profile'] is False and summary['profile'] is False,'Wrong V6 run identity')
            need(config['code_sha256']==read(run_dir/'source_hashes.json')==analysis['training_source_sha256'],
                 'Training source identity differs from canonical analysis')
            for name,digest in config['code_sha256'].items():
                need(sha(REPO/name)==digest and sha(run_dir/'code'/name.replace('/','_'))==digest,'Training source snapshot changed')
            for field,filename,copied in (('test_manifest','main_manifest.json','staged_test_manifest.json'),
                                          ('test_count_manifest','count_manifest.json','staged_test_count_manifest.json')):
                need(Path(config[field]).resolve()==FRESH/filename and
                     config[field+'_sha256']==manifest_hashes[filename]==sha(run_dir/copied),'Fresh evaluation identity differs')
            need(history==summary['training'] and len(history)==9 and
                 all((r['epoch'],r['step'],r['n'])==(i+1,45*(i+1),180) for i,r in enumerate(history)),
                 'Require complete unchanged V6 training history')
            for epoch in history:
                dev=read(run_dir/f"dev_epoch{epoch['epoch']}.json")
                need(dev['metrics']==epoch['dev'] and sum(m['n'] for m in epoch['dev'])==72,'Dev selection evidence differs')
            best=max(history,key=lambda e:(sum(m['correct'] for m in e['dev'])/72,
                -sum(m['gold_first_token_nll']*m['n'] for m in e['dev'])/72))
            checkpoint=Path(summary['selected_checkpoint']).resolve()
            need(checkpoint==CKPTS/run_dir.name/'best.pt' and str(checkpoint)==info['selected_checkpoint'] and
                 sha(checkpoint)==info['selected_checkpoint_sha256'],'Selected checkpoint identity differs')
            saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
            architecture=saved['architecture'];distillation=architecture['local_distillation']
            need(architecture==config['architecture'] and architecture['protocol']=='vision_v6_local_teacher' and
                 architecture['rank']==96 and architecture['operator']['merge']=='sum' and
                 distillation['condition']==condition and distillation['parameters']==291 and
                 distillation['query']=='last_original_prompt_token' and distillation['deployed'] is False and
                 distillation['detach_messages']==(condition=='control'),'Wrong selected local head architecture')
            need(saved['epoch']==best['epoch']==info['selected_epoch'] and saved['step']==best['step'] and
                 saved['dev']==best['dev'],'Selected checkpoint is not the registered dev winner')
            need(all(saved['config'][name]==config[name] for name in ('run_id','condition','seed','architecture','code_sha256',
                 'test_manifest_sha256','test_count_manifest_sha256','teacher_binding')),'Selected checkpoint config differs')
            head=saved['auxiliary_head']
            need(saved['auxiliary_training']==dict(parameters=291,optimizer_step=best['step'],lr=.001,
                weight_decay=0.,max_grad_norm=1.,detached=condition=='control'),'Auxiliary checkpoint training identity differs')
            apply_head(torch,torch.zeros(1,96,dtype=torch.float32),head)
            head_hashes={name:hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest() for name,value in head.items()}
            rows=read(run_dir/'predictions.json')
            need(len(rows)==452 and len({r['sid'] for r in rows})==452 and
                 {(r['cell'],r['sid']) for r in rows}==set(source),'Every fresh native prediction must be included')
            canonical={r['path']:r['sha256'] for r in info['prefill_diagnostics']['records']}
            need(len(canonical)==452,'Canonical report must bind every saved message tensor')
            scored=[]
            for row in rows:
                original=source[(row['cell'],row['sid'])]
                need(all(row[field]==original[field] for field in ('path','sid','n_frames','gold','pair_id','test_family')) and
                     row['mode']=='all' and row['tag']=='test','Prediction/source/evaluation identity differs')
                messages=load_messages(torch,row,run_dir,canonical)
                logits,probabilities,predictions=apply_head(torch,messages,head)
                for index,(label,logit,prob,pred) in enumerate(zip(original['semantic_labels'],
                        logits.tolist(),probabilities.tolist(),predictions.tolist())):
                    scored.append(dict(cell=row['cell'],sid=row['sid'],family_id=row['pair_id'],n_frames=row['n_frames'],
                        gold=row['gold'],frame_index=index,semantic_class=label,truth=int(label=='positive'),
                        logits=logit,probabilities=prob,prediction=pred))
            need(len(scored)==18240,'Expected all18240 frame judgments per checkpoint')
            path=data_out/f'{key}.jsonl'
            with path.open('x') as stream:
                for row in scored:stream.write(json.dumps(row,allow_nan=False)+'\n')
            groups={cell:[r for r in scored if r['cell']==cell] for cell in CELLS}
            result=dict(condition=condition,seed=seed,selected_epoch=best['epoch'],checkpoint=str(checkpoint),
                checkpoint_sha256=info['selected_checkpoint_sha256'],head_tensor_sha256=head_hashes,
                frame_output_path=str(path),frame_output_sha256=sha(path),
                all_contexts=summarize(np,scored),
                familiar_ood=summarize(np,[r for r in scored if r['cell'] in CELLS[1:3]]),
                unseen_count=summarize(np,[r for r in scored if r['cell'] in CELLS[3:]]),
                cells={cell:summarize(np,group) for cell,group in groups.items()},
                per_N_K={cell:{str(k):summarize(np,[r for r in group if r['gold']==k]) for k in sorted({r['gold'] for r in group})}
                         for cell,group in groups.items()})
            results[key]=result;write(out/f'{key}.json',result)
            provenance['runs'][key]=dict(directory=str(run_dir),config_sha256=sha(run_dir/'config.json'),
                summary_sha256=sha(run_dir/'summary.json'),history_sha256=sha(run_dir/'training.json'),
                predictions_sha256=sha(run_dir/'predictions.json'),checkpoint=str(checkpoint),
                checkpoint_sha256=info['selected_checkpoint_sha256'],head_tensor_sha256=head_hashes,
                diagnostic_tensor_sha256=canonical)
            print(json.dumps(dict(run=key,contexts=452,frames=len(scored),
                                  N64=result['cells']['length_N64'])),flush=True)
            del saved,head,scored
    need(sha(analysis_path)==analysis_sha and read(ledger)==sources(),'Analysis/diagnostic sources changed during audit')
    for filename,digest in manifest_hashes.items():need(sha(FRESH/filename)==digest,'Fresh manifest changed during audit')
    write(out/'provenance.json',provenance)
    summary=dict(schema_version=1,protocol='v6_saved_auxiliary_head_on_all_fresh_messages',
        completed=True,refitted=False,new_teacher_forwards=0,new_vlm_forwards=0,contexts_per_checkpoint=452,
        frames_per_checkpoint=18240,arithmetic='FP32 CPU linear +three-way softmax; no feature normalization',
        results=results,elapsed_seconds=time.monotonic()-started,
        limitations=['Diagnostic saved head is absent from native inference; no external count is computed.',
            'Other is an incorrect third prediction for both binary truth classes; no abstention filtering.',
            'These are FP32 CPU applications of the saved head, not a bitwise claim about GPU training logits.',
            'Per-N results are primary descriptive views. Pooled AUROC can reflect length-dependent score offsets and different class prevalences.',
            'Frame observations are correlated within scenes and paired families; no new confidence interval or success criterion.',
            'High head readability does not prove native use or exact counting; low readability does not prove absent information.',
            'All four dev-selected heads and all452fresh contexts are included without outcome-dependent selection.'])
    write(out/'summary.json',summary)
    lines=['# V6 saved local-head diagnostic','',
        'No refitting, new teacher/model forwards, or external counting. Every saved head scores all452fresh contexts.','',
        '| Head | Cell | Frames | Three-way accuracy | Balanced accuracy | TPR | FPR | Other rate | AUROC p1 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(value):return 'undefined' if value is None else f'{value:.4f}'
    for key,result in results.items():
        for cell,x in result['cells'].items():
            values=[fmt(x[k]) for k in ('three_way_accuracy','balanced_accuracy','tpr','fpr','other_rate','auroc_p1')]
            lines.append('| '+key+' | '+cell+' | '+str(x['n'])+' | '+' | '.join(values)+' |')
    lines+=['','Other is always wrong: balanced accuracy averages positive recall and the fraction of negatives predicted0. It does not use1−FPR as negative recall. AUROC uses unconditional p1 and includes other predictions.','',
        'Full2×3confusions, category-specific negatives, perN/K summaries and every frame probability are retained. These correlated frame-level statistics do not establish causal native use or absence of information.','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    (out/'INDEX.md').write_text('# V6 local-head artifacts\n\n[Report](REPORT.md) · [Summary](summary.json) · [Provenance](provenance.json) · [Sources](source_hashes.json).\n')
    (data_out/'INDEX.md').write_text('# Saved auxiliary-head frame judgments\n\n'+
        '\n'.join(f'- [{key}]({key}.jsonl)' for key in results)+'\n')
    print(json.dumps(dict(output=str(out),data_output=str(data_out),seconds=summary['elapsed_seconds'])),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true');parser.add_argument('--source-ledger',type=Path)
    args=parser.parse_args()
    need(bool(os.environ.get('SLURM_JOB_ID')) and os.environ.get('SLURM_JOB_PARTITION')=='cpu' and
         not os.environ.get('SLURM_JOB_GPUS',''),'Use the Slurm CPU-only wrapper before numerical work')
    import numpy as np
    import torch
    torch.set_num_threads(4);tested=self_test(np,torch)
    if args.self_test:
        out=OUT/f"check_{os.environ['SLURM_JOB_ID']}";out.mkdir(parents=True,exist_ok=False);snapshot(out)
        write(out/'tests.json',tested)
        (out/'INDEX.md').write_text('# V6 local-head software checks\n\n[Tests](tests.json) · [Frozen source ledger](source_hashes.json).\n')
        print(json.dumps(dict(passed=True,source_ledger=str(out/'source_hashes.json'))),flush=True)
    else:
        need(args.source_ledger is not None and read(args.source_ledger)==sources(),'Pass the prospectively frozen source ledger')
        run(np,torch,args.source_ledger)


if __name__=='__main__':main()

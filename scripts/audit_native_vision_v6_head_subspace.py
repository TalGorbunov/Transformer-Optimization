"""Label-free selected-head subspaces of every saved V6 test message.

P=(CA)^+(CA), C=I-11'/3, Q=I-P. With the original bias unchanged,
softmax(A Pm+b)=softmax(A m+b) in exact arithmetic. Q is invisible to this
fixed auxiliary head; it is not thereby irrelevant to native count prediction.
This CPU-only diagnostic never fits, selects a checkpoint, or calls a VLM.
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
from scripts.audit_native_vision_v6_local_head import load_sources,load_messages,apply_head
from scripts.probe_native_vision_v5_local_readability import CELLS,CLASSES

BASE=REPO/'outputs/native_aggregation_vlm/v6'
OUT=BASE/'head_subspace'
DATA=Path('/mnt/data/gabriele/gnn_transformer')
CKPTS=Path('/mnt/ckpts/gabriele/gnn_transformer/native_aggregation_vlm_v6')
RANK_RTOL=1e-10
GROUPS=('total','positive','negative','char_only','room_only','neither')
PARTS=('original','head_visible_P','head_invisible_Q')
FIELDS=('message_norm','P_message_norm','Q_message_norm','native_norm','P_native_norm','Q_native_norm',
        'message_PQ_cosine','native_PQ_cosine','native_total_P_cosine','native_total_Q_cosine',
        'P_energy_per_dimension','Q_energy_per_dimension','native_Q_norm_share')
OWN=('scripts/audit_native_vision_v6_head_subspace.py',
     'slurm/native_aggregation_vision_v6_head_subspace.sbatch',
     'scripts/audit_native_vision_v6_local_head.py',
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


def projector(torch,weight):
    need(weight.ndim==2 and weight.shape[0]==3 and bool(torch.isfinite(weight).all()),'Invalid three-way head')
    a=weight.detach().cpu().double()
    c=torch.eye(3,dtype=torch.float64)-torch.ones((3,3),dtype=torch.float64)/3
    centered=c@a
    _,singular,vh=torch.linalg.svd(centered,full_matrices=False)
    cutoff=RANK_RTOL*float(singular[0]) if len(singular) else 0.
    rank=int((singular>cutoff).sum())
    need(rank<=2,'Centered three-way head unexpectedly has numerical rank above2')
    basis=vh[:rank]
    p=basis.T@basis
    q=torch.eye(a.shape[1],dtype=torch.float64)-p
    norms=dict(symmetry_max=float((p-p.T).abs().max()),idempotence_max=float((p@p-p).abs().max()),
               complement_max=float((p+q-torch.eye(a.shape[1],dtype=torch.float64)).abs().max()),
               centered_head_Q_max=float((centered@q).abs().max()),
               centered_head_scale=float(centered.abs().max()))
    need(norms['symmetry_max']<=1e-12 and norms['idempotence_max']<=1e-12
         and norms['complement_max']<=1e-12,'FP64 projection identities failed')
    # This certifies the recorded fixed SVD truncation, not exact real-arithmetic rank.
    allowed=RANK_RTOL*float(singular[0])*2+1e-12*max(1.,norms['centered_head_scale'])
    need(norms['centered_head_Q_max']<=allowed,'Centered-head nullspace residual exceeds SVD precision')
    condition=float(singular[0]/singular[rank-1]) if rank else None
    metadata=dict(rank=rank,head_visible_dimensions=rank,head_invisible_dimensions=a.shape[1]-rank,
        rank_rtol=RANK_RTOL,singular_values=singular.tolist(),rank_cutoff=cutoff,
        retained_condition_number=condition,two_contrast_condition_number=condition if rank==2 else None,
        rank_deficient_contrast=rank<2,projection_identities=norms,
        coordinate_metric='Euclidean metric of the actual96-dimensional messages; not invariant under arbitrary nonorthogonal reparameterization')
    return p,q,metadata


def invariance(torch,messages,p,head):
    """Bias is unchanged; compare FP64 algebra and separate ordinary FP32 replay."""
    m=messages.double(); projected=m@p
    weight,bias=head['weight'].double(),head['bias'].double()
    logits=m@weight.T+bias; changed=projected@weight.T+bias
    centered=logits-logits.mean(-1,keepdim=True)
    changed_centered=changed-changed.mean(-1,keepdim=True)
    probabilities=torch.softmax(logits,-1); changed_probabilities=torch.softmax(changed,-1)
    fp64_tv=.5*(probabilities-changed_probabilities).abs().sum(-1)
    centered_error=(centered-changed_centered).abs()
    # A fixed-rank truncation can discard a very small genuine singular direction.
    # Record both the absolute and relative residual instead of hiding that fact.
    fp64_tolerance=1e-8*(1+float(centered.abs().max()))
    fp64_pass=bool(float(centered_error.max())<=fp64_tolerance and float(fp64_tv.max())<=1e-8)
    old32=torch.nn.functional.linear(messages,head['weight'],head['bias'])
    new32=torch.nn.functional.linear(projected.float(),head['weight'],head['bias'])
    old32p=torch.softmax(old32,-1);new32p=torch.softmax(new32,-1)
    old32c=old32-old32.mean(-1,keepdim=True);new32c=new32-new32.mean(-1,keepdim=True)
    need(bool(torch.isfinite(old32).all()) and bool(torch.isfinite(new32).all()),'Nonfinite native FP32 head replay')
    # FP32 replay is descriptive. A near-tied argmax may flip despite tiny TV.
    report=dict(images=messages.shape[0],same_bias_retained=True,
        fp64_centered_logit_max_error=float(centered_error.max()),
        fp64_centered_logit_max_tolerance=fp64_tolerance,
        fp64_probability_max_tv=float(fp64_tv.max()),fp64_probability_mean_tv=float(fp64_tv.mean()),
        fp64_probability_tv_tolerance=1e-8,fp64_invariance_passed=fp64_pass,
        fp64_argmax_changes=int((logits.argmax(-1)!=changed.argmax(-1)).sum()),
        fp32_centered_logit_max_error=float((old32c-new32c).abs().max()),
        fp32_probability_max_tv=float((.5*(old32p-new32p).abs().sum(-1)).max()),
        fp32_probability_mean_tv=float((.5*(old32p-new32p).abs().sum(-1)).mean()),
        fp32_argmax_changes=int((old32.argmax(-1)!=new32.argmax(-1)).sum()),
        fp32_policy='Numerical replay diagnostics only; no efficacy or argmax-invariance gate')
    return projected,m-projected,report


def geometry(torch,m,p,q,gram,rank):
    need(m.ndim==p.ndim==q.ndim==2 and m.shape==p.shape==q.shape,'Geometry shape mismatch')
    def dot(a,b,g=None):return (a*b).sum(-1) if g is None else ((a@g)*b).sum(-1)
    def norm(a,g=None):return dot(a,a,g).clamp_min(0).sqrt()
    def cosine(a,b,na,nb,g=None):
        denom=na*nb
        values=dot(a,b,g)/denom.clamp_min(1e-300)
        return [None if float(d)==0 else max(-1.,min(1.,float(v))) for v,d in zip(values,denom)]
    mn,pn,qn=(norm(value) for value in (m,p,q))
    un,upn,uqn=(norm(value,gram) for value in (m,p,q))
    penergy=[float(x*x/rank) if rank else None for x in pn]
    qrank=m.shape[1]-rank
    qenergy=[float(x*x/qrank) if qrank else None for x in qn]
    values=[mn.tolist(),pn.tolist(),qn.tolist(),un.tolist(),upn.tolist(),uqn.tolist(),
            cosine(p,q,pn,qn),cosine(p,q,upn,uqn,gram),cosine(m,p,un,upn,gram),
            cosine(m,q,un,uqn,gram),penergy,qenergy,
            [float(qn/(pn+qn)) if float(pn+qn)>0 else None for pn,qn in zip(upn,uqn)]]
    return [dict(zip(FIELDS,row)) for row in zip(*values)]


def numeric_summary(values):
    valid=[float(x) for x in values if x is not None]
    need(all(math.isfinite(x) for x in valid),'Nonfinite summary scalar')
    return dict(n=len(valid),mean=sum(valid)/len(valid) if valid else None,
                minimum=min(valid) if valid else None,maximum=max(valid) if valid else None,
                rms=math.sqrt(sum(x*x for x in valid)/len(valid)) if valid else None)


def summary_rows(rows,fields=FIELDS):
    return dict(records=len(rows),contexts=len({r['sid'] for r in rows}),
        families=len({r['family_id'] for r in rows}),
        metrics={name:numeric_summary([row.get(name) for row in rows]) for name in fields})


def selected_run(torch,analysis,condition,seed,manifest_hashes):
    info=analysis['conditions'][condition][str(seed)]
    run=Path(analysis['runs'][condition][str(seed)]).resolve()
    need(run.parent==BASE/'main'/condition/f'seed{seed}','Wrong canonical V6 run path')
    config,summary,history=(read(run/name) for name in ('config.json','summary.json','training.json'))
    need(config['run_id']==summary['run_id']==run.name and config['condition']==summary['condition']==condition
         and config['seed']==seed and config['profile'] is False and summary['profile'] is False,'Run identity differs')
    need(config['code_sha256']==read(run/'source_hashes.json')==analysis['training_source_sha256'],
         'Canonical training source provenance differs')
    for name,value in config['code_sha256'].items():
        need(sha(REPO/name)==value and sha(run/'code'/name.replace('/','_'))==value,'Training source/snapshot changed')
    for field,filename,copied in (('test_manifest','main_manifest.json','staged_test_manifest.json'),
                                  ('test_count_manifest','count_manifest.json','staged_test_count_manifest.json')):
        need(Path(config[field]).resolve()==DATA/'v6_fresh'/filename and
             config[field+'_sha256']==manifest_hashes[filename]==sha(run/copied),'Fresh manifest binding differs')
    need(history==summary['training'] and len(history)==9 and
         all((r['epoch'],r['step'],r['n'])==(i+1,45*(i+1),180) for i,r in enumerate(history)),
         'Incomplete canonical training history')
    best=max(history,key=lambda e:(sum(m['correct'] for m in e['dev'])/72,
        -sum(m['gold_first_token_nll']*m['n'] for m in e['dev'])/72))
    checkpoint=Path(summary['selected_checkpoint']).resolve()
    need(checkpoint==CKPTS/run.name/'best.pt' and str(checkpoint)==info['selected_checkpoint'] and
         sha(checkpoint)==info['selected_checkpoint_sha256'],'Selected checkpoint identity differs')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    need(saved['architecture']==config['architecture'] and saved['architecture']['protocol']=='vision_v6_local_teacher'
         and saved['architecture']['operator']['merge']=='sum' and saved['architecture']['rank']==96,
         'Expected unchanged V6 native SUM architecture')
    need(saved['epoch']==best['epoch']==info['selected_epoch'] and saved['step']==best['step']
         and saved['dev']==best['dev'],'Checkpoint differs from dev-only selection')
    for name in ('run_id','condition','seed','architecture','code_sha256','teacher_binding'):
        need(saved['config'][name]==config[name],'Selected checkpoint configuration differs')
    head=saved['auxiliary_head'];apply_head(torch,torch.zeros(1,96,dtype=torch.float32),head)
    up=saved['branch']['up.weight']
    need(up.shape==(3584,96) and up.dtype==torch.float32 and bool(torch.isfinite(up).all()),
         'Expected finite selected native U3584x96')
    return run,config,summary,info,head,up,checkpoint


def run(np,torch,ledger):
    started=time.monotonic();analysis_path=BASE/'analysis.json'
    analysis=read(analysis_path);analysis_sha=sha(analysis_path)
    need(set(analysis['runs'])==set(analysis['conditions'])=={'control','aligned'},'All four canonical runs are required')
    for name,value in analysis['analysis_source_sha256'].items():
        need(sha(REPO/name)==value and sha(BASE/'analysis_code'/name.replace('/','_'))==value,
             'Canonical analysis source changed')
    source,manifest_hashes=load_sources()
    job=os.environ['SLURM_JOB_ID'];out=OUT/f'run_{job}';out.mkdir(parents=True,exist_ok=False);snapshot(out)
    data_out=DATA/'v6_head_subspace'/f'run_{job}';data_out.mkdir(parents=True,exist_ok=False)
    provenance=dict(analysis_path=str(analysis_path),analysis_sha256=analysis_sha,
        analysis_source_sha256=analysis['analysis_source_sha256'],training_source_sha256=analysis['training_source_sha256'],
        source_sha256=sources(),source_ledger_sha256=sha(ledger),manifest_sha256=manifest_hashes,
        torch_version=str(torch.__version__),numpy_version=np.__version__,slurm_job_id=job,runs={})
    results={}
    for condition in ('control','aligned'):
        need(set(analysis['runs'][condition])==set(analysis['conditions'][condition])=={'6','7'},'Both seeds required')
        for seed in (6,7):
            key=f'{condition}_seed{seed}'
            directory,config,native_summary,info,head,up,checkpoint=selected_run(torch,analysis,condition,seed,manifest_hashes)
            p,q,projection=projector(torch,head['weight'])
            gram=up.double().T@up.double()
            rows=read(directory/'predictions.json')
            need(len(rows)==452 and len({r['sid'] for r in rows})==452
                 and {(r['cell'],r['sid']) for r in rows}==set(source),'Incomplete or repeated diagnostic contexts')
            canonical={r['path']:r['sha256'] for r in info['prefill_diagnostics']['records']}
            need(len(canonical)==452,'Missing canonical saved-message bindings')
            frames=[];sums=[];vectors=[];identities=[];checks=[]
            for row in rows:
                original=source[(row['cell'],row['sid'])]
                need(all(row[k]==original[k] for k in ('path','sid','n_frames','gold','pair_id','test_family'))
                     and row['mode']=='all' and row['tag']=='test','Native row/canonical source differs')
                messages=load_messages(torch,row,directory,canonical)
                pm,qm,check=invariance(torch,messages,p,head)
                checks.append(dict(cell=row['cell'],sid=row['sid'],**check))
                m=messages.double();rank=projection['rank']
                frame_geometry=geometry(torch,m,pm,qm,gram,rank)
                identity=dict(cell=row['cell'],sid=row['sid'],family_id=row['pair_id'],n_frames=row['n_frames'],gold=row['gold'])
                labels=original['semantic_labels']
                for index,(label,geo) in enumerate(zip(labels,frame_geometry)):
                    frames.append(dict(**identity,frame_index=index,semantic_class=label,**geo))
                group_vectors=[]
                for group in GROUPS:
                    mask=torch.tensor([group=='total' or (label!='positive' if group=='negative' else label==group)
                                       for label in labels],dtype=torch.bool)
                    components=[value[mask].sum(0) for value in (m,pm,qm)]
                    group_vectors.append(torch.stack(components))
                    geo=geometry(torch,*(value.unsqueeze(0) for value in components),gram,rank)[0]
                    coherence={}
                    for prefix,field in (('total','message_norm'),('P','P_message_norm'),('Q','Q_message_norm')):
                        denominator=sum(frame_geometry[i][field] for i in range(len(labels)) if bool(mask[i]))
                        coherence[f'{prefix}_sum_coherence']=geo[field]/denominator if denominator>0 else None
                    sums.append(dict(**identity,group=group,images=int(mask.sum()),**geo,**coherence))
                vectors.append(torch.stack(group_vectors));identities.append(identity)
            need(len(frames)==18240 and len(sums)==452*len(GROUPS),'Full diagnostic denominators differ')
            vector_path=data_out/f'{key}_sums.pt'
            torch.save(dict(schema_version=1,context_order=identities,groups=list(GROUPS),parts=list(PARTS),
                sum_vectors=torch.stack(vectors),P=p,Q=q,U_gram=gram,
                head_weight=head['weight'],head_bias=head['bias'],projection=projection),vector_path)
            frame_groups={cell:[r for r in frames if r['cell']==cell] for cell in CELLS}
            sum_fields=(*FIELDS,'total_sum_coherence','P_sum_coherence','Q_sum_coherence')
            by_cell={cell:dict(
                frames=summary_rows(group),
                frames_by_class={label:summary_rows([r for r in group if r['semantic_class']==label]) for label in CLASSES},
                sums={name:summary_rows([r for r in sums if r['cell']==cell and r['group']==name],sum_fields) for name in GROUPS})
                for cell,group in frame_groups.items()}
            per_k={cell:{str(k):dict(
                frames=summary_rows([r for r in group if r['gold']==k]),
                frames_by_class={label:summary_rows([r for r in group if r['gold']==k and r['semantic_class']==label]) for label in CLASSES},
                sums={name:summary_rows([r for r in sums if r['cell']==cell and r['gold']==k and r['group']==name],sum_fields) for name in GROUPS})
                for k in sorted({r['gold'] for r in group})} for cell,group in frame_groups.items()}
            all64=all(c['fp64_invariance_passed'] for c in checks)
            precision=dict(all_fp64_invariance_passed=all64,
                failed_fp64_contexts=[c['sid'] for c in checks if not c['fp64_invariance_passed']],
                fp64_centered_logit_max_error=max(c['fp64_centered_logit_max_error'] for c in checks),
                fp64_probability_max_tv=max(c['fp64_probability_max_tv'] for c in checks),
                fp64_argmax_changes=sum(c['fp64_argmax_changes'] for c in checks),
                fp32_centered_logit_max_error=max(c['fp32_centered_logit_max_error'] for c in checks),
                fp32_probability_max_tv=max(c['fp32_probability_max_tv'] for c in checks),
                fp32_argmax_changes=sum(c['fp32_argmax_changes'] for c in checks),
                native_decoder_invariance_claimed=False)
            result=dict(condition=condition,seed=seed,contexts=452,frames=18240,checkpoint=str(checkpoint),
                checkpoint_sha256=info['selected_checkpoint_sha256'],projection=projection,precision=precision,
                cells=by_cell,per_N_K=per_k,sum_vectors_path=str(vector_path),sum_vectors_sha256=sha(vector_path))
            results[key]=result;write(out/f'{key}.json',result)
            write(out/f'{key}_context_sums.json',sums);write(out/f'{key}_precision.json',checks)
            provenance['runs'][key]=dict(directory=str(directory),config_sha256=sha(directory/'config.json'),
                summary_sha256=sha(directory/'summary.json'),predictions_sha256=sha(directory/'predictions.json'),
                checkpoint=str(checkpoint),checkpoint_sha256=info['selected_checkpoint_sha256'],
                diagnostic_tensor_sha256=canonical,sum_vectors_path=str(vector_path),sum_vectors_sha256=sha(vector_path))
            print(json.dumps(dict(run=key,rank=projection['rank'],precision=precision)),flush=True)
            del frames,sums,vectors,messages,p,q,gram,head,up
    need(sha(analysis_path)==analysis_sha and read(ledger)==sources(),'Source/analysis changed during diagnostic')
    for filename,value in manifest_hashes.items():need(sha(DATA/'v6_fresh'/filename)==value,'Fresh manifest changed')
    write(out/'provenance.json',provenance)
    limitations=[
        'Head-invisible Q is not nuisance: native count CE also trains these coordinates; the learned head changes during training.',
        'Rank at most2 describes one fixed three-way softmax head, not total representational information or all training-gradient directions.',
        'The same bias is retained in all local probability comparisons; local invariance does not imply native decoder or future-query invariance.',
        'U does not generally preserve orthogonality: P/Q message norms or native norms are not additive explained-variance fractions; native cross-cosines are retained.',
        'P uses the fixed Euclidean message-coordinate metric; dimension-normalized energies are reported because94 versus2 dimensions alone can create norm imbalance.',
        'Question-conditioned hidden states and rendered Step labels can change across paired lengths; observed norm growth is not an intervention holding evidence/query fixed.',
        'Stored native attention-output norms do not establish full residual-stream dominance or a normalization bottleneck.',
        'Gold semantic classes only stratify descriptive reports and sum groups; they never determine P/Q or select examples/checkpoints.',
        'All four dev-selected checkpoints/all452contexts are included; no fitting, efficacy filtering, VLM forward or new external tally is performed.']
    summary=dict(schema_version=1,protocol='v6_fixed_head_visible_invisible_subspaces',completed=True,
        contexts_per_checkpoint=452,frames_per_checkpoint=18240,rank_rtol=RANK_RTOL,
        arithmetic='FP64 SVD/projections and U-transpose-U Gram geometry; separately recorded FP32 local-head replay',
        all_fp64_invariance_passed=all(r['precision']['all_fp64_invariance_passed'] for r in results.values()),
        results=results,elapsed_seconds=time.monotonic()-started,limitations=limitations)
    write(out/'summary.json',summary)
    lines=['# V6 fixed-head message subspaces','',
        'All dev-selected checkpoints and all452contexts each. P preserves the selected local head probabilities mathematically with its bias unchanged; native decoding is not invariant. Q means head-invisible, not irrelevant.','',
        '| Checkpoint | Cell | Rank | Mean negative-sum native P norm | Mean negative-sum native Q norm | Mean native P/Q cosine |',
        '|---|---|---:|---:|---:|---:|']
    def fmt(x):return 'undefined' if x is None else f'{x:.5g}'
    for key,result in results.items():
        for cell,values in result['cells'].items():
            metrics=values['sums']['negative']['metrics']
            entries=[fmt(metrics[k]['mean']) for k in ('P_native_norm','Q_native_norm','native_PQ_cosine')]
            lines.append('| '+' | '.join([key,cell,str(result['projection']['rank']),*entries])+' |')
    lines+=['',f"FP64 invariance checks passed for every context: {summary['all_fp64_invariance_passed']}.",
        'Each run retains rank/condition numbers, every precision discrepancy, perN/K/class summaries, and exact96D total/positive/negative/category sum vectors with P/Q parts. No3584D per-image matrices are written.','',*limitations,'']
    (out/'REPORT.md').write_text('\n'.join(lines))
    (out/'INDEX.md').write_text('# V6 head-subspace diagnostic\n\n[Report](REPORT.md) · [Summary](summary.json) · [Provenance](provenance.json).\n')
    (data_out/'INDEX.md').write_text('# V6 rank-space component sums\n\n'+
        '\n'.join(f'- [{key}]({key}_sums.pt)' for key in results)+'\n')
    print(json.dumps(dict(output=str(out),data_output=str(data_out),seconds=summary['elapsed_seconds'])),flush=True)


def self_test(torch):
    # This toy has an exactly represented coordinate projector, including SVD.
    a=torch.zeros(3,96,dtype=torch.float32);a[0,0]=1;a[1,0]=-1
    head=dict(weight=a,bias=torch.tensor([.25,-.5,.75],dtype=torch.float32))
    p,q,info=projector(torch,a)
    expected=torch.zeros_like(p);expected[0,0]=1
    need(torch.equal(p,expected) and info['rank']==1,'Exact rank1 toy projector differs')
    m=torch.zeros(2,96,dtype=torch.float32);m[:,0]=torch.tensor([1.,-2.]);m[:,1]=torch.tensor([3.,4.])
    pm,qm,report=invariance(torch,m,p,head)
    need(report['fp64_invariance_passed'] and report['fp64_probability_max_tv']==0
         and report['fp32_probability_max_tv']==0,'Same-bias local probability invariance failed exact toy')
    changed_bias=dict(head,bias=torch.zeros(3))
    old=torch.softmax(torch.nn.functional.linear(m,head['weight'],head['bias']),-1)
    wrong=torch.softmax(torch.nn.functional.linear(pm.float(),changed_bias['weight'],changed_bias['bias']),-1)
    need(not torch.equal(old,wrong),'Bias-removal counterexample disappeared')
    up=torch.zeros(4,96,dtype=torch.float64);up[0,0]=2;up[0,1]=1;up[1,1]=3
    gram=up.T@up;geo=geometry(torch,m.double(),pm,qm,gram,1)
    for index,row in enumerate(geo):
        for key,value in (('native_norm',m.double()[index]),('P_native_norm',pm[index]),('Q_native_norm',qm[index])):
            need(math.isclose(row[key],float((up@value).norm()),rel_tol=1e-12,abs_tol=1e-12),
                 'Gram native norm differs from explicit U projection')
    need(all(abs(r['message_PQ_cosine'])<1e-12 for r in geo)
         and any(abs(r['native_PQ_cosine'])>.1 for r in geo),'U-induced nonorthogonality was hidden')
    original=m.double().sum(0);parts=pm.sum(0)+qm.sum(0)
    need(torch.equal(original,parts),'P/Q aggregation does not reconstruct original')
    zero=torch.zeros_like(a);pz,qz,iz=projector(torch,zero)
    need(iz['rank']==0 and torch.equal(pz,torch.zeros_like(pz)) and torch.equal(qz,torch.eye(96,dtype=torch.float64)),
         'Zero/rank-deficient head is mishandled')
    rank2=a.clone();rank2[2,1]=2
    _,_,i2=projector(torch,rank2)
    need(i2['rank']==2,'Centered three-way head rank2 case failed')
    return dict(passed=True,tests=['exact_coordinate_projector','same_bias_softmax_invariance',
        'bias_change_counterexample','Gram_vs_explicit_U_norms','nonorthogonal_native_parts',
        'sum_reconstruction','rank_zero','rank_two'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true');parser.add_argument('--source-ledger',type=Path)
    args=parser.parse_args()
    need(bool(os.environ.get('SLURM_JOB_ID')) and os.environ.get('SLURM_JOB_PARTITION')=='cpu'
         and not os.environ.get('SLURM_JOB_GPUS',''),'All numerical work requires the Slurm CPU wrapper')
    import numpy as np
    import torch
    torch.set_num_threads(4);tested=self_test(torch)
    if args.self_test:
        out=OUT/f"check_{os.environ['SLURM_JOB_ID']}";out.mkdir(parents=True,exist_ok=False);snapshot(out)
        write(out/'tests.json',tested)
        (out/'INDEX.md').write_text('# V6 subspace software checks\n\n[Tests](tests.json) · [Frozen source ledger](source_hashes.json).\n')
        print(json.dumps(dict(passed=True,source_ledger=str(out/'source_hashes.json'))),flush=True)
    else:
        need(args.source_ledger is not None and read(args.source_ledger)==sources(),'Require prospective source ledger')
        run(np,torch,args.source_ledger)


if __name__=='__main__':main()

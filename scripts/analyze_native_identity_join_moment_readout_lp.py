"""Bounded homogeneous nine-class LP on frozen actual pre-U moment readouts.

The prior strict native audit remains failed. This separately audits only the
stored FP32 readout states and certifies candidate LP witnesses independently.
No core, backbone, native norm/head or optimizer is constructed or called.
"""
from __future__ import annotations
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
import warnings
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import diagnose_native_identity_join_moment_basis_v2 as parent
from scripts.stage_native_vision_v6_teacher import need,read,save,sha,object_sha
PROTOCOL='identity_join_moment_readout_lp'
OUT=REPO/'outputs/native_aggregation_vlm'/PROTOCOL
DATA=Path('/mnt/data/gabriele/gnn_transformer')/PROTOCOL
CKPT=Path('/mnt/ckpts/gabriele/gnn_transformer')/PROTOCOL
REPORT=parent.OUT/'report_443841'
FAILURE_SHA='63a8a8bdaaefc69ece5c9bacc417c1820cc356f8bcf59dcfce2ea199f535c094'
ANALYSIS_SHA='5b206f806cc3787b4b9cf3601839d1ccb872c53932d1fd4c02b14aa779e7c256'
PROPOSAL='docs/paper/NATIVE_AGGREGATION_MOMENT_READOUT_LP_PROPOSAL.md'
PROPOSAL_SHA='7a244dd45b8ffd399c5e12b01e4c489d52451e7c969c5e4d2af365b0c2e9591d'
OWN=('scripts/analyze_native_identity_join_moment_readout_lp.py','scripts/moment_readout_lp_certificate.py',
     'slurm/native_identity_join_moment_readout_lp.sbatch',PROPOSAL)
ARMS=('within','cross');PEOPLE=('Sandra','Mary','Michael','John','Daniel','Laura','Peter','Emma','Noah')
OPTIONS=dict(time_limit=45.,maxiter=100000,presolve=True,dual_feasibility_tolerance=1e-9,
    primal_feasibility_tolerance=1e-9,simplex_dual_edge_weight_strategy='steepest-devex',threads=4)
POLICY=dict(protocol=PROTOCOL,method='highs-ds',options=OPTIONS,cpu_seconds_cap=300,
    classes=list(PEOPLE),contexts_per_arm=216,readout_width=96,lp_variables=1729,lp_inequalities=1729,
    intercept=False,feature_centering=False,feature_normalization=False,core_calls=0,norm_calls=0,head_calls=0,
    vlm_calls=0,vision_calls=0,optimizer_calls=0,data_solver_calls=2,fixture_solver_calls=1,
    fixture_seconds_cap=5,fixture_exact_interval_gap_cap='1/100000000',fixture='nine standard basis rows in96D, labels0..8, exact optimum1/9',validation_contexts=0,
    target='strict separation of all216 labels, not the206-context method gate')


def sources():return {name:sha(REPO/name) for name in OWN}


def inherited_sources():
    result={}
    for group in (parent.sources(),parent.inherited_sources()):
        for name,digest in group.items():
            need(name not in result or result[name]==digest,'Inherited source conflict');result[name]=digest
    need(len(result)==229,'Frozen six plus223 ancestor source closure required')
    return result


def bind(path,bindings,expected=None):
    path=Path(path).resolve();digest=sha(path)
    need(expected is None or digest==expected,'Bound input changed: '+str(path))
    need(str(path) not in bindings or bindings[str(path)]==digest,'Conflicting input binding')
    bindings[str(path)]=digest;return digest


def bind_references(value,bindings):
    if isinstance(value,dict):
        for key,item in value.items():
            tag='sha256' if key=='file' else ('checkpoint_sha256' if key=='checkpoint' else key[:-5]+'_sha256' if key.endswith('_file') else None)
            if tag and tag in value and isinstance(item,str):bind(item,bindings,value[tag])
            bind_references(item,bindings)
    elif isinstance(value,list):
        for item in value:bind_references(item,bindings)


def evidence(path,bindings):
    need(Path(path).resolve()==REPORT/'failure.json','Only the fixed preserved443841 failure is authorized')
    bind(path,bindings,FAILURE_SHA);bind(REPORT/'analysis.json',bindings,ANALYSIS_SHA)
    failure=read(path);analysis=read(REPORT/'analysis.json')
    need(failure['type']=='ValueError' and failure['partial_outputs_retained'] is True
         and failure['message']=='Unchanged numerical core/native/NLL gate failed; all44 captures and28 native replay batches retained'
         and failure['analysis_file']==str(REPORT/'analysis.json') and failure['analysis_sha256']==ANALYSIS_SHA
         and not (REPORT/'summary.json').exists(),'Preserved strict failure changed')
    need(analysis['passed'] is False and analysis['completed'] is analysis['all_numerical_audits_collected'] is True
         and analysis['protocol']==parent.PROTOCOL and analysis['phase']=='report' and set(analysis['runs'])==set(ARMS)
         and analysis['cpu_head_calls']==analysis['cpu_norm_calls']==28 and analysis['cpu_head_rows']==432,
         'Complete strict-failed audit required')
    need(analysis['source_sha256']==failure['source_sha256']==parent.sources()
         and analysis['inherited_source_sha256']==parent.inherited_sources(),'Exact failed report source closure differs')
    bind(REPORT/'source_hashes.json',bindings);bind(REPORT/'inherited_sources.json',bindings)
    need(read(REPORT/'source_hashes.json')==analysis['source_sha256']
         and read(REPORT/'inherited_sources.json')==dict(source_sha256=analysis['inherited_source_sha256']),'Report source descriptors differ')
    for name,digest in analysis['source_sha256'].items():bind(REPORT/'source'/name.replace('/','_'),bindings,digest)
    bind(analysis['plan_file'],bindings,analysis['plan_sha256']);plan=parent.verify_plan(analysis['plan_file'],ancestors=True)
    for group in ('runtime_bindings','input_bindings'):
        for file,digest in plan[group].items():bind(file,bindings,digest)
    bind(REPO/PROPOSAL,bindings,PROPOSAL_SHA);bind_references(analysis,bindings)
    rows=read(plan['rows_file']);need(len(rows)==216 and len({r['sid'] for r in rows})==216,'Exact216 rows required')
    negatives={};all_mismatches=[]
    for arm,count in (('within',77),('cross',73)):
        run=analysis['runs'][arm];ep=run['endpoint_audits']['6000'];audits=run['capture_audits']
        need(run['completed'] is run['all_numerical_audits_collected'] is ep['nll_precision_passed'] is True
             and len(audits)==22 and run['cpu_head_calls']==run['cpu_norm_calls']==14 and run['cpu_head_rows']==216,
             'Complete22-capture/14-batch arm audit required')
        outcomes=read(ep['outcomes_file']);need(len(outcomes)==216 and all({k:o[k] for k in r}==r for r,o in zip(rows,outcomes)), 'Outcome row ownership differs')
        need(ep['first_token_fit']==parent.criteria(outcomes)==analysis['first_token_screen'][arm]
             and ep['first_token_fit']['first_correct']==count and ep['first_token_fit']['complete_families']==0
             and not ep['first_token_fit']['passed'],'Fixed negative training outcomes differ')
        need([a['step'] for a in audits if a['phase']=='training']==[1,2,32,128,300,600,2000,6000],
             'All eight pre-update audits must be retained')
        mismatches=[];final=[]
        for audit in audits:
            need(audit['fp16_cast_before_add_exact'] is True and all(v['passed'] is True for v in audit['functional_core'].values())
                 and all(v is True for v in audit['moment_algebra'].values()) and all(v['passed'] is True for v in audit['nll']),
                 'Every functional/moment/cast/NLL audit must pass')
            if audit['phase']=='training':need(audit['native_head']==[] and audit['head_rows']==0,'No training head replay allowed')
            else:
                need(audit['phase']=='evaluation' and audit['endpoint_step']==6000 and len(audit['native_head'])==len(audit['sids'])==audit['head_rows'], 'Final replay ownership differs')
                final.append(audit)
                for sid,metric in zip(audit['sids'],audit['native_head']):
                    need(math.isfinite(metric['full_vocabulary_tv']) and metric['full_vocabulary_tv']<=.02
                         and metric['passed'] is metric['argmax_exact'],'Only recorded argmax disagreement may fail')
                    if not metric['argmax_exact']:mismatches.append(sid)
        need(len(final)==14 and [s for a in final for s in a['sids']]==[r['sid'] for r in rows], 'All216 final captures required')
        expected=[] if arm=='within' else ['ij_915e3e4912bcd3ea440a211a']
        need(mismatches==expected,'Exactly the preserved single cross near-tie is allowed')
        need(all(not o['first_token_correct'] for o in outcomes if o['sid'] in mismatches),
             'The sole CPU disagreement must be GPU-incorrect')
        optimistic=[dict(r,first_token_correct=r['first_token_correct'] or r['sid'] in mismatches) for r in outcomes]
        bound=parent.criteria(optimistic)
        need(bound['first_correct']==(77 if arm=='within' else 74) and not bound['passed'],'Robust negative bound differs')
        negatives[arm]=dict(gpu_correct=count,cpu_correct_lower=count,cpu_correct_upper=bound['first_correct'],
            optimistic_correctness_union=bound,disagreement_sids=mismatches,actual_cpu_argmax_not_loaded=True)
        all_mismatches+=mismatches
    need(len(all_mismatches)==1,'Preserve exactly one failed native argmax check')
    return analysis,plan,rows,negatives


def gather(torch,np,analysis,rows,arm,bindings,data):
    run=analysis['runs'][arm];root=Path(run['run_directory']);bind(root/'summary.json',bindings,run['run_summary_sha256'])
    summary=read(root/'summary.json');bind_references(summary,bindings)
    records=read(summary['captures_file']);final=[r for r in records if r['phase']=='evaluation']
    audits=[a for a in run['capture_audits'] if a['phase']=='evaluation'];features=[];inventory=[]
    need(len(final)==len(audits)==14,'Fourteen saved final capture batches required')
    for i,(record,audit) in enumerate(zip(final,audits),1):
        need(record['step']==audit['step']==i and record['endpoint_step']==6000 and record['basis']==arm
             and record['file']==audit['capture_file'] and record['sha256']==audit['capture_sha256']
             and record['weights_step']==6000 and record['weights_file']==audit['weights_file']==run['endpoint_audits']['6000']['checkpoint']
             and record['weights_sha256']==audit['weights_sha256']==run['endpoint_audits']['6000']['checkpoint_sha256'], 'Actual final state binding differs')
        bind(record['file'],bindings,record['sha256']);packet=torch.load(record['file'],map_location='cpu',weights_only=True)
        selected=rows[(i-1)*16:i*16];sids=[r['sid'] for r in selected];h=packet['capture']['readout']
        need(packet['basis']==arm and packet['phase']=='evaluation' and packet['endpoint_step']==6000 and packet['step']==i
             and packet['sids']==record['sids']==audit['sids']==sids and packet['layout']['first_query_only'] is True
             and packet['layout']['target_ids']==[r['first_token_id'] for r in selected]
             and h.dtype==torch.float32 and h.shape==(len(sids),96) and not h.requires_grad and bool(torch.isfinite(h).all()),
             'Actual raw FP32 pre-U readout ownership differs')
        x=h.numpy().copy();features.append(x)
        for j,row in enumerate(selected):inventory.append(dict(row=row,capture_file=record['file'],capture_sha256=record['sha256'],
            capture_row=j,weights_file=record['weights_file'],weights_sha256=record['weights_sha256'],
            selected_parameter_sha256=run['endpoint_audits']['6000']['selected_parameter_sha256'],
            readout_sha256=hashlib.sha256(x[j].tobytes()).hexdigest(),raw_fp32_l2_norm=float(np.linalg.norm(x[j].astype(np.float64)))))
        del packet,h
    x=np.concatenate(features,axis=0);labels=np.array([PEOPLE.index(r['gold']) for r in rows],dtype=np.int64)
    need(x.shape==(216,96) and x.dtype==np.float32 and Counter(labels.tolist())=={i:24 for i in range(9)},'Frozen feature/class inventory differs')
    np.save(data/'readouts_fp32.npy',x,allow_pickle=False);np.save(data/'readouts_fp64.npy',x.astype(np.float64),allow_pickle=False);np.save(data/'labels.npy',labels,allow_pickle=False)
    save(data/'inventory.json',dict(arm=arm,classes=list(PEOPLE),rows=inventory,preprocessing='FP32 values promoted exactly to FP64; no other transform'))
    return x.astype(np.float64),labels


def problem(np,h,labels,n_classes=9):
    n,r=h.shape;d=n_classes*r;D=np.zeros((n*(n_classes-1),d),dtype=np.float64);constraints=[];cursor=0
    for i,y in enumerate(labels):
        for k in range(n_classes):
            if k==y:continue
            D[cursor,int(y)*r:(int(y)+1)*r]=h[i];D[cursor,k*r:(k+1)*r]=-h[i]
            constraints.append(dict(row=i,target=int(y),other=k));cursor+=1
    A=np.zeros((len(D)+1,2*d+1),dtype=np.float64);A[:-1,:d]=-D;A[:-1,d:2*d]=D;A[:-1,-1]=1.;A[-1,:2*d]=1.
    b=np.zeros(len(A),dtype=np.float64);b[-1]=1.;c=np.zeros(2*d+1,dtype=np.float64);c[-1]=-1.
    return A,b,c,[(0,None)]*(2*d)+[(None,None)],constraints


def plain(value):
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    if hasattr(value,'tolist'):return plain(value.tolist())
    if isinstance(value,float) and not math.isfinite(value):return str(value)
    return value


def solve(np,linprog,certify,h,labels,data,ckpt,out,options,counters,kind):
    data.mkdir();ckpt.mkdir();A,b,c,bounds,indices=problem(np,h,labels)
    need(A.shape==(len(h)*8+1,1729),'Fixed nine-class96D LP shape differs')
    np.savez_compressed(data/'lp_problem.npz',A_ub=A,b_ub=b,c=c)
    save(data/'constraints.json',dict(margin_rows=indices,budget_row=len(indices),
        variable_order='Wplus[9,96] rowmajor, Wminus[9,96] rowmajor, free gamma',bounds=bounds))
    counters[kind]+=1;tick=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');solved=linprog(c,A_ub=A,b_ub=b,bounds=bounds,method='highs-ds',options=dict(options))
    solver=plain(dict(solved));solver.update(method='highs-ds',options=options,
        elapsed_seconds=time.perf_counter()-tick,warnings=[str(w.message) for w in caught])
    save(ckpt/'solver_result.json',solver)
    diagnostics={k:v for k,v in solver.items() if k not in ('x','lower','upper')}
    diagnostics['bound_marginals']={name:solver.get(name,{}).get('marginals') for name in ('lower','upper')}
    save(data/'solver_diagnostics.json',dict(diagnostics,
        coefficients_file=str(ckpt/'solver_result.json'),coefficients_sha256=sha(ckpt/'solver_result.json')))
    cert,witness=certify(h,labels,solver,n_classes=9)
    save(ckpt/'certificate_witness.json',witness);save(out,cert)
    return dict(certificate=cert,file=str(out),sha256=sha(out),
        witness_file=str(ckpt/'certificate_witness.json'),witness_sha256=sha(ckpt/'certificate_witness.json'),
        solver_file=str(ckpt/'solver_result.json'),solver_sha256=sha(ckpt/'solver_result.json'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'), 'CPU Slurm allocation required')
    started=time.perf_counter();out=OUT/f'lp_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False)
    data=DATA/out.name;ckpt=CKPT/out.name;data.mkdir(parents=True,exist_ok=False);ckpt.mkdir(parents=True,exist_ok=False)
    frozen=sources();inherited=inherited_sources();(out/'source').mkdir()
    for name,digest in frozen.items():(out/'source'/name.replace('/','_')).write_bytes((REPO/name).read_bytes())
    save(out/'source_hashes.json',frozen);save(out/'inherited_sources.json',dict(source_sha256=inherited));bindings={};counters=dict(data=0,fixture=0)
    save(out/'request.json',dict(report=str(args.report.resolve()),protocol=PROTOCOL,policy=POLICY,source_sha256=frozen))
    try:
        analysis,plan,rows,negatives=evidence(args.report,bindings)
        import numpy as np
        import scipy
        from scipy.optimize import linprog
        import torch
        from scripts.moment_readout_lp_certificate import certify,self_test
        torch.set_num_threads(4)
        solver_modules=['scipy.optimize._linprog','scipy.optimize._linprog_highs','scipy.optimize._highspy._highs_wrapper','scipy.optimize._highspy._core','scipy.optimize._highspy._highs_options']
        installed={}
        for name in solver_modules:
            module=importlib.import_module(name);path=Path(module.__file__).resolve();installed[name]=dict(file=str(path),sha256=bind(path,bindings))
        highs_core=importlib.import_module('scipy.optimize._highspy._core')
        version_constants={name:getattr(highs_core,name) for name in dir(highs_core)
            if 'VERSION' in name.upper() and isinstance(getattr(highs_core,name),(str,int,float))}
        highs_version=dict(available=bool(version_constants),public_constants=version_constants,
            binary_sha256=installed['scipy.optimize._highspy._core']['sha256'],
            note='Public version constants when exposed; otherwise version unavailable, exact binary hash retained')
        save(out/'solver_implementation.json',dict(numpy_version=np.__version__,scipy_version=scipy.__version__,torch_version=str(torch.__version__),
            highs_version=highs_version,modules=installed,options=OPTIONS,method='highs-ds'))
        tests=self_test();save(out/'certificate_self_test.json',tests);need(tests['passed'] is True,'Independent certificate fixtures failed')
        fixture_h=np.zeros((9,96),dtype=np.float64);fixture_h[:,:9]=np.eye(9,dtype=np.float64);fixture_labels=np.arange(9,dtype=np.int64)
        fixture=solve(np,linprog,certify,fixture_h,fixture_labels,data/'fixture',ckpt/'fixture',out/'fixture_certificate.json',
            dict(OPTIONS,time_limit=5.),counters,'fixture')
        np.save(data/'fixture'/'readouts_fp64.npy',fixture_h,allow_pickle=False)
        np.save(data/'fixture'/'labels.npy',fixture_labels,allow_pickle=False)
        exact=lambda value:Fraction(int(value['numerator']),int(value['denominator']))
        fc=fixture['certificate'];interval=fc['interval']
        need(fc['status']=='strictly_separable' and fc['fp32']['certified_all'] is True
             and interval['upper'] is not None and exact(interval['lower'])<=Fraction(1,9)<=exact(interval['upper'])
             and exact(interval['upper'])-exact(interval['lower'])<=Fraction(1,10**8),
             'Actual synthetic LP/certificate plumbing failed its exact1/9 bound')
        fixture.update(passed=True,known_exact_optimum=dict(numerator='1',denominator='9'))
        banks={}
        for arm in ARMS:
            (data/arm).mkdir();banks[arm]=gather(torch,np,analysis,rows,arm,bindings,data/arm)
        save(out/'input_bindings.json',bindings);save(out/'negative_evidence.json',negatives);certificates={}
        for arm in ARMS:
            h,labels=banks[arm]
            certificates[arm]=solve(np,linprog,certify,h,labels,data/arm/'lp',ckpt/arm,out/f'{arm}_certificate.json',OPTIONS,counters,'data')
        files={str(p):sha(p) for directory in (data,ckpt) for p in sorted(directory.rglob('*')) if p.is_file()}
        save(out/'artifacts.json',files)
        need(counters==dict(data=2,fixture=1) and sources()==frozen and inherited_sources()==inherited
             and all(sha(f)==s for f,s in bindings.items()),'Source/input or fixed-three-solve contract changed')
        elapsed=time.perf_counter()-started;need(elapsed<=300,'Fixed300-second CPU diagnostic cap exceeded')
        save(out/'summary.json',dict(passed=True,completed=True,diagnostic_only=True,protocol=PROTOCOL,policy=POLICY,source_sha256=frozen,inherited_source_sha256=inherited,
            prior_strict_report_passed=False,prior_failure_file=str(REPORT/'failure.json'),prior_failure_sha256=FAILURE_SHA,
            prior_analysis_file=str(REPORT/'analysis.json'),prior_analysis_sha256=ANALYSIS_SHA,
            plan_file=analysis['plan_file'],plan_sha256=analysis['plan_sha256'],negative_evidence=negatives,certificates=certificates,fixture=fixture,
            solver_implementation_file=str(out/'solver_implementation.json'),solver_implementation_sha256=sha(out/'solver_implementation.json'),
            certificate_self_test_file=str(out/'certificate_self_test.json'),certificate_self_test_sha256=sha(out/'certificate_self_test.json'),
            negative_evidence_file=str(out/'negative_evidence.json'),negative_evidence_sha256=sha(out/'negative_evidence.json'),
            input_bindings_file=str(out/'input_bindings.json'),input_bindings_sha256=sha(out/'input_bindings.json'),artifacts_file=str(out/'artifacts.json'),artifacts_sha256=sha(out/'artifacts.json'),
            native_identity_sha256=plan['native_identity_sha256'],data_solver_calls=counters['data'],fixture_solver_calls=counters['fixture'],solver_calls=sum(counters.values()),
            core_calls=0,norm_calls=0,head_calls=0,vlm_calls=0,vision_calls=0,optimizer_calls=0,
            feature_preprocessing='exact FP32 to FP64 promotion only',elapsed_seconds=elapsed,
            limitations='Homogeneous nine-class separation of h only; omits varying global residual, full vocabulary and native-U constraints. No native fit, information-loss or generalization conclusion.'))
        print(json.dumps(dict(completed=True,directory=str(out))),flush=True)
    except BaseException as exc:
        save(out/'failure_input_bindings.json',bindings)
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,elapsed_seconds=time.perf_counter()-started,
            data_solver_calls=counters['data'],fixture_solver_calls=counters['fixture'],solver_calls=sum(counters.values()),
            input_bindings_file=str(out/'failure_input_bindings.json'),input_bindings_sha256=sha(out/'failure_input_bindings.json'),partial_outputs_retained=True));raise


if __name__=='__main__':main()

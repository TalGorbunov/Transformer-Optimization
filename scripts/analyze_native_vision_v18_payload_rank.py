"""HELD descriptive V18 positive-payload statistics from compact saved archives.

CPU Slurm only. Question means use the fresh test sample and are never deployed.
No native model/head, probability calculation, optimization, or new prediction.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import analyze_native_vision_v18 as ancestor
need,read,sha,save,bind=ancestor.need,ancestor.read,ancestor.sha,ancestor.save,ancestor.bind
DEFAULT=REPO/'outputs/native_aggregation_vlm/v18/diagnosis/diagnosis_442927/summary.json'
OUT=REPO/'outputs/native_aggregation_vlm/v18/payload_rank'
OWN=('scripts/analyze_native_vision_v18_payload_rank.py','slurm/native_vision_v18_payload_rank.sbatch')
KEYS=('native_gate_s20','native_gate_s21','all_open_s20','all_open_s21')
POLICY=dict(protocol='v18_saved_positive_payload_question_constant_diagnosis',
    source_diagnosis_job='442927',models=list(KEYS),scenes_per_model=272,positive_occurrences_per_model=2176,
    empty_scenes_per_model=16,lengths=[32,64],counts=list(range(17)),width=96,position=0,prefix_ids=[],
    grouping='exact question and target character/room; preserve every sampled occurrence',
    weights='actual applied gate: native FP32 g or one for all_open, promoted to FP64',
    question_mean='weighted mean over all positive test occurrences for this question; stable anchored FP64 reduction',
    spread='weighted coordinate RMS and maximum row L2 distance; unweighted companions retain closed positives',
    spectra='all96 singular values and squared-energy fractions of sqrt(g)*payload and sqrt(g)*(payload-question_mean)',
    unweighted_spectra='payload and payload-unweighted_question_mean; no occurrence deduplication',
    scene_identity='P-mass*mu = sum g*(payload-mu) + sum(saved_FP32_message-g*payload)',
    algebra_rtol=1e-10,algebra_atol=1e-10,rank_threshold=None,zero_energy_fractions=None,
    question_means_test_derived=True,question_means_never_fed_to_model=True,
    no_optimizer=True,no_predictions=True,no_intervention=True,no_accuracy_evaluation=True,
    no_model_calls=True,no_head_calls=True,no_probability_recomputation=True,
    cpu_cores=4,memory_gb=16,seconds_cap=300)


def own_sources():return {name:sha(REPO/name) for name in OWN}


def snapshot(out):
    frozen=own_sources();(out/'code').mkdir()
    for name,digest in frozen.items():
        target=out/'code'/name.replace('/','_');target.write_bytes((REPO/name).read_bytes())
        need(sha(target)==digest,'Source changed during initial snapshot')
    save(out/'own_source_hashes.json',frozen);save(out/'policy.json',POLICY)
    return frozen


def close(torch,a,b,message):
    need(a.shape==b.shape and torch.allclose(a,b,rtol=1e-10,atol=1e-10),message)
    return float((a-b).abs().max()) if a.numel() else 0.


def ratio(a,b):return float(a/b) if b else None


def validate_values(torch,u,g):
    need(u.ndim==2 and g.shape==(len(u),) and u.dtype==g.dtype==torch.float64
         and bool(torch.isfinite(u).all()) and bool(torch.isfinite(g).all())
         and bool(((g>=0)&(g<=1)).all()),'Invalid FP64 payload/weight rows')


def moments(torch,u,g):
    """Anchoring makes an exactly constant payload have exactly zero spread."""
    validate_values(torch,u,g);n,width=u.shape;mass=float(g.sum());active=g>0
    unweighted=None;mu=None;unweighted_center=torch.zeros_like(u);weighted_center=torch.zeros_like(u)
    if n:
        unweighted=u[0]+(u-u[0]).mean(0);unweighted_center=u-unweighted
    if mass:
        anchor=u[active][0];mu=anchor+(g[:,None]*(u-anchor)).sum(0)/g.sum()
        weighted_center=g.sqrt()[:,None]*(u-mu)
        residual=(g[:,None]*(u-mu)).sum(0)
        residual_error=close(torch,residual,torch.zeros(width,dtype=torch.float64),'Weighted centering residual differs')
    else:residual_error=0.
    weighted_energy=float(weighted_center.square().sum())
    unweighted_energy=float(unweighted_center.square().sum())
    distances=(u-mu).norm(dim=1) if mu is not None else None
    result=dict(positive_occurrences=n,weight_mass=mass,positive_weight_occurrences=int(active.sum()),
        zero_weight_occurrences=int((~active).sum()),weighted_mean_defined=mu is not None,
        weighted_mean=mu.tolist() if mu is not None else None,
        unweighted_mean=unweighted.tolist() if unweighted is not None else None,
        weighted_mean_norm=float(mu.norm()) if mu is not None else None,
        weighted_coordinate_rms_spread=math.sqrt(weighted_energy/(width*mass)) if mass else None,
        max_item_l2_spread_all=float(distances.max()) if distances is not None and n else None,
        max_item_l2_spread_positive_weight=float(distances[active].max()) if distances is not None else None,
        unweighted_coordinate_rms_spread=math.sqrt(unweighted_energy/(width*n)) if n else None,
        unweighted_max_item_l2_spread=float(unweighted_center.norm(dim=1).max()) if n else None,
        weighted_centered_energy=weighted_energy,unweighted_centered_energy=unweighted_energy,
        weighted_uncentered_energy=float((g[:,None]*u.square()).sum()),
        unweighted_uncentered_energy=float(u.square().sum()),centering_identity_max_abs=residual_error)
    return result,mu,weighted_center,unweighted_center


def spectrum(torch,matrix):
    need(matrix.ndim==2 and matrix.dtype==torch.float64 and bool(torch.isfinite(matrix).all()),'Invalid SVD input')
    width=matrix.shape[1]
    values=torch.linalg.svdvals(matrix) if len(matrix) else torch.empty(0,dtype=torch.float64)
    singular=torch.cat((values,torch.zeros(width-len(values),dtype=torch.float64)))
    energy=singular.square();total=float(matrix.square().sum())
    error=close(torch,energy.sum(),matrix.square().sum(),'SVD/Frobenius energy reconciliation failed')
    need(len(singular)==width and bool((singular>=0).all()),'Full singular-value inventory differs')
    return dict(rows=len(matrix),columns=width,singular_values=singular.tolist(),squared_singular_values=energy.tolist(),
        total_energy=total,energy_fractions=(energy/total).tolist() if total else None,
        energy_identity_abs=error,rank_threshold=None,numerical_rank_not_estimated=True)


def scene_deviation(torch,u,g,m,p,mu):
    validate_values(torch,u,g);width=u.shape[1]
    need(m.shape==u.shape and p.shape==(width,) and m.dtype==p.dtype==torch.float64
         and bool(torch.isfinite(m).all()) and bool(torch.isfinite(p).all()),'Saved positive message shape/finiteness differs')
    close(torch,m.sum(0),p,'Compact positive P differs from saved messages')
    mass=g.sum();product=(g[:,None]*u).sum(0)
    if mu is None:
        need(float(mass)==0.,'Undefined question mean with positive mass')
        constant=torch.zeros(width,dtype=torch.float64);heterogeneity=torch.zeros_like(constant)
    else:
        constant=mass*mu;heterogeneity=(g[:,None]*(u-mu)).sum(0)
    multiplication=p-product;deviation=p-constant
    error=close(torch,heterogeneity+multiplication,deviation,'Question-constant decomposition failed')
    return dict(positive_occurrences=len(u),weight_mass=float(mass),no_positive_occurrences=len(u)==0,
        zero_positive_weight_mass=float(mass)==0.,question_mean_defined=mu is not None,
        saved_P_norm=float(p.norm()),question_constant_norm=float(constant.norm()),
        deviation_norm=float(deviation.norm()),relative_deviation_to_P=ratio(float(deviation.norm()),float(p.norm())),
        payload_heterogeneity_norm=float(heterogeneity.norm()),saved_FP32_multiplication_residual_norm=float(multiplication.norm()),
        identity_max_abs=error,saved_P=p.tolist(),question_constant=constant.tolist(),deviation=deviation.tolist(),
        payload_heterogeneity=heterogeneity.tolist(),saved_FP32_multiplication_residual=multiplication.tolist())


def load_inputs(path,out,frozen,bindings):
    path=bind(path,None,bindings)
    need(path==DEFAULT.resolve(),'Only the fixed completed diagnosis442927 is eligible')
    summary=read(path)
    need(summary.get('passed') is True and summary.get('completed') is True and summary['slurm_job_id']=='442927'
         and summary['scenes']==1088 and summary['families']==544
         and summary['no_model_calls'] is True and summary['no_head_calls'] is True,'Completed no-model diagnosis required')
    analysis=read(bind(summary['analysis_file'],summary['analysis_sha256'],bindings))
    need(analysis['passed'] is True and analysis['completed'] is True and analysis['source_sha256']==summary['source_sha256']
         and analysis['archives']==summary['archives'] and set(analysis['models'])==set(KEYS)
         and analysis['policy']['protocol']=='v18_all_scenes_saved_state_diagnosis'
         and analysis['policy']['geometry_position']==0 and analysis['policy']['geometry_prefix_ids']==[],
         'Diagnosis analysis/source/scope differs')
    prior_bindings=read(bind(summary['input_bindings_file'],summary['input_bindings_sha256'],bindings))
    need(read(path.parent/'source_hashes.json')==summary['source_sha256'],'Diagnosis source ledger differs')
    bind(path.parent/'source_hashes.json',None,bindings)
    for name,digest in summary['source_sha256'].items():
        rel=Path(name);need(not rel.is_absolute() and '..' not in rel.parts,'Invalid inherited source path')
        bind(REPO/rel,digest,bindings);old=bind(path.parent/'code'/name.replace('/','_'),digest,bindings)
        need(name not in frozen or frozen[name]==digest,'Conflicting source identity');frozen[name]=digest
        target=out/'code'/name.replace('/','_')
        if not target.exists():target.write_bytes(old.read_bytes())
        need(sha(target)==digest,'Inherited source copy changed')
    original=bind(analysis['report_file'],analysis['report_sha256'],bindings)
    need(prior_bindings[str(original)]['sha256']==analysis['report_sha256'],'Original report not bound by completed diagnosis')
    report,metadata,families=ancestor.load_report(original,out,frozen,bindings)
    need(set(report['runs'])==set(KEYS) and len(metadata)==272 and len(families)==136,'Original complete report inventory differs')
    for key in KEYS:
        need(analysis['models'][key]['endpoint']==report['runs'][key]['endpoint'],'Compact archive endpoint differs from verified final model')
        descriptor=summary['archives'][key]
        need(descriptor['scenes']==272 and descriptor['families']==136,'Compact archive descriptor differs')
        bind(descriptor['file'],descriptor['sha256'],bindings)
    save(out/'source_hashes.json',frozen)
    return summary,analysis,report,metadata


def analyze_model(torch,key,packet,metadata,endpoint):
    need(packet['schema_version']==1 and packet['model']==key and packet['endpoint']==endpoint
         and len(packet['scenes'])==272 and len(packet['families'])==136,'Wrong compact model packet')
    mode=key.rsplit('_s',1)[0];by_sid={};groups=defaultdict(list);question_scenes=defaultdict(list);group_info={}
    for scene in packet['scenes']:
        sid=scene['sid'];need(sid in metadata and sid not in by_sid,'Missing or repeated compact scene')
        spec=metadata[sid];p=spec['planned'];n=p['n_frames'];positive=torch.tensor(spec['labels'],dtype=torch.bool)
        need(scene['pair_id']==p['pair_id'] and torch.equal(scene['positive'],positive),'Compact positive ownership differs')
        u,g,m=scene['payload'],scene['applied_gates'],scene['messages']
        validate_values(torch,u,g)
        need(u.shape==(n,96) and m.shape==u.shape and m.dtype==torch.float64
             and bool(torch.isfinite(m).all()) and bool((u.abs()<=1).all()),'Compact tanh payload/message shape differs')
        need(torch.equal(g,scene['native_gates'] if mode=='native_gate' else torch.ones_like(g)),'Applied mode weights differ')
        need(bool((m[g==0]==0).all()),'A closed gate has nonzero saved message coordinates')
        need(torch.equal(u,u.float().double()) and torch.equal(g,g.float().double())
             and torch.equal(m,m.float().double()),'Compact input is not an exact promotion of saved FP32 values')
        need(scene['P'].dtype==torch.float64 and scene['P'].shape==(96,),'Compact positive aggregate schema differs')
        close(torch,m[positive].sum(0),scene['P'],'Compact positive aggregate ownership differs')
        identity=[p['question'],p['target_character'],p['target_room']];q=ancestor.object_sha(identity)
        group_info[q]=dict(question=p['question'],target_character=p['target_character'],target_room=p['target_room'])
        question_scenes[q].append(dict(sid=sid,n_frames=n,gold=p['gold'],pair_id=p['pair_id']))
        indices=positive.nonzero(as_tuple=False).flatten().tolist()
        for i in indices:
            groups[q].append(dict(sid=sid,frame_index=i,n_frames=n,gold=p['gold'],pair_id=p['pair_id'],
                image_sha256=spec['sample']['image_files'][i]['sha256'],weight=float(g[i])))
        by_sid[sid]=dict(scene=scene,positive=positive,q=q,u=u[positive],g=g[positive],m=m[positive],planned=p)
    need(set(by_sid)==set(metadata) and sum(len(x['u']) for x in by_sid.values())==2176
         and sum(not len(x['u']) for x in by_sid.values())==16,'Complete positive/empty scene inventory differs')
    need(Counter((x['planned']['n_frames'],x['planned']['gold']) for x in by_sid.values())==Counter({(n,k):8 for n in (32,64) for k in range(17)}),
         'N/K sampling weights differ')
    group_rows=[];means={};matrices={name:[] for name in ('weighted_uncentered','weighted_question_centered','unweighted_uncentered','unweighted_question_centered')}
    for q in sorted(question_scenes):
        members=question_scenes[q];records=groups[q]
        u=torch.cat([by_sid[r['sid']]['u'] for r in members],dim=0)
        g=torch.cat([by_sid[r['sid']]['g'] for r in members],dim=0)
        need(len(u)==len(records),'Question occurrence inventory differs')
        result,mu,wcenter,ucenter=moments(torch,u,g);means[q]=mu
        matrices['weighted_uncentered'].append(g.sqrt()[:,None]*u)
        matrices['weighted_question_centered'].append(wcenter)
        matrices['unweighted_uncentered'].append(u);matrices['unweighted_question_centered'].append(ucenter)
        group_rows.append(dict(question_id=q,**group_info[q],scene_count=len(members),
            family_count=len({r['pair_id'] for r in members}),represented_counts=sorted({r['gold'] for r in members}),
            represented_lengths=sorted({r['n_frames'] for r in members}),scenes=members,positive_occurrences=records,moments=result))
    spectra={name:spectrum(torch,torch.cat(parts,dim=0)) for name,parts in matrices.items()}
    scenes=[]
    for sid,x in by_sid.items():
        value=scene_deviation(torch,x['u'],x['g'],x['m'],x['scene']['P'],means[x['q']]);p=x['planned']
        scenes.append(dict(sid=sid,pair_id=p['pair_id'],question_id=x['q'],n_frames=p['n_frames'],gold=p['gold'],**value))
    groups_mass=math.fsum(row['moments']['weight_mass'] for row in group_rows)
    need(math.isclose(groups_mass,math.fsum(row['weight_mass'] for row in scenes),rel_tol=1e-10,abs_tol=1e-10),'Question/scene weighted mass differs')
    summary=dict(model=key,scenes=272,positive_occurrences=2176,no_positive_scenes=16,question_groups=len(group_rows),
        positive_question_groups=sum(row['moments']['positive_occurrences']>0 for row in group_rows),
        positive_weight_question_groups=sum(row['moments']['weighted_mean_defined'] for row in group_rows),weight_mass=groups_mass,
        zero_weight_positive_occurrences=sum(row['moments']['zero_weight_occurrences'] for row in group_rows),
        weighted_centered_energy_fraction=ratio(spectra['weighted_question_centered']['total_energy'],spectra['weighted_uncentered']['total_energy']),
        unweighted_centered_energy_fraction=ratio(spectra['unweighted_question_centered']['total_energy'],spectra['unweighted_uncentered']['total_energy']),
        question_constant_deviation={f'N{n}':dict(n=sum(r['n_frames']==n for r in scenes),
            absolute_norm=ancestor.stats([r['deviation_norm'] for r in scenes if r['n_frames']==n]),
            relative_to_P=ancestor.stats([r['relative_deviation_to_P'] for r in scenes if r['n_frames']==n]),
            heterogeneity_norm=ancestor.stats([r['payload_heterogeneity_norm'] for r in scenes if r['n_frames']==n]),
            multiplication_residual_norm=ancestor.stats([r['saved_FP32_multiplication_residual_norm'] for r in scenes if r['n_frames']==n])) for n in (32,64)},
        maximum_identity_error=max(r['identity_max_abs'] for r in scenes),spectra=spectra)
    return dict(summary=summary,question_groups=group_rows,scenes=scenes)


def self_test(torch):
    dtype=torch.float64
    u=torch.tensor([[.25,-.5]]*3,dtype=dtype);g=torch.tensor([.2,.3,.7],dtype=dtype)
    result,mu,center,_=moments(torch,u,g)
    need(torch.equal(mu,u[0]) and result['weighted_coordinate_rms_spread']==0. and bool((center==0).all()),'Constant payload did not center exactly')
    need(spectrum(torch,center)['energy_fractions'] is None,'Zero-energy spectrum fabricated fractions')
    two=torch.tensor([[1.,0.],[0.,1.]],dtype=dtype)
    sv=spectrum(torch,two);need(sv['singular_values']==[1.,1.] and sv['energy_fractions']==[.5,.5],'Two-direction spectrum differs')
    _,_,center,_=moments(torch,two,torch.ones(2,dtype=dtype));cs=spectrum(torch,center)
    need(math.isclose(cs['squared_singular_values'][0],1.,rel_tol=1e-10,abs_tol=1e-10)
         and cs['squared_singular_values'][1]<1e-20,'Known centered two-direction spectrum differs')
    zero=torch.zeros(2,dtype=dtype);r,undefined,weighted,_=moments(torch,two,zero)
    need(undefined is None and r['weighted_coordinate_rms_spread'] is None and r['unweighted_coordinate_rms_spread']>0.
         and bool((weighted==0).all()),'Zero weights hid unweighted spread or fabricated a mean')
    empty=torch.empty(0,2,dtype=dtype);r,undefined,_,_=moments(torch,empty,zero[:0])
    need(r['positive_occurrences']==0 and undefined is None and spectrum(torch,empty)['singular_values']==[0.,0.],'Empty question group differs')
    values=torch.tensor([[.2,.3],[.7,-.3],[.5,.1]],dtype=torch.float32).double()
    weights=torch.tensor([.3,.7,0.],dtype=torch.float32).double();_,mu,_,_=moments(torch,values,weights)
    messages=(values.float()*weights.float()[:,None]).double();errors=[];residuals=[]
    for ids in ([0,1],[2],[]):
        idx=torch.tensor(ids,dtype=torch.long);p=messages[idx].sum(0)
        d=scene_deviation(torch,values[idx],weights[idx],messages[idx],p,mu);errors.append(d['identity_max_abs']);residuals.append(d['saved_FP32_multiplication_residual_norm'])
    need(max(errors)<=1e-10 and max(residuals)>0. and scene_deviation(torch,empty,zero[:0],empty,torch.zeros(2,dtype=dtype),None)['no_positive_occurrences'],
         'Ragged weighted P/heterogeneity/roundoff identity failed')
    try:validate_values(torch,two,torch.tensor([1.,-1.],dtype=dtype))
    except ValueError:pass
    else:raise ValueError('Invalid weight passed')
    return dict(passed=True,tests=['constant_payload_exact_centering','zero_energy_fraction_undefined','two_direction_singular_values',
        'closed_positive_unweighted_companion','empty_question_group','ragged_weighted_identity_and_FP32_residual','invalid_weight_rejected'],
        no_model_calls=True,no_head_calls=True)


def run(torch,args,out,frozen,bindings,tests):
    summary,diagnosis,report,metadata=load_inputs(args.diagnosis,out,frozen,bindings)
    models={}
    for key in KEYS:
        descriptor=summary['archives'][key]
        packet=torch.load(descriptor['file'],map_location='cpu',weights_only=True)
        result=analyze_model(torch,key,packet,metadata,report['runs'][key]['endpoint'])
        file=out/(key+'.json');save(file,result)
        models[key]=dict(file=str(file),sha256=sha(file),summary=result['summary'],source_archive=descriptor,
                         endpoint=report['runs'][key]['endpoint'])
        del packet,result
        print(json.dumps(dict(model=key,scenes=272,positive_occurrences=2176,model_calls=0)),flush=True)
    for path,record in bindings.items():need(sha(path)==record['sha256'],'Consumed input changed: '+path)
    need(all(sha(REPO/name)==digest for name,digest in frozen.items()),'Source changed during diagnosis')
    result=dict(schema_version=1,passed=True,completed=True,policy=POLICY,source_sha256=frozen,
        diagnosis_summary=str(args.diagnosis.resolve()),diagnosis_summary_sha256=sha(args.diagnosis),
        diagnosis_analysis_file=summary['analysis_file'],diagnosis_analysis_sha256=summary['analysis_sha256'],
        original_report_file=diagnosis['report_file'],original_report_sha256=diagnosis['report_sha256'],
        models=models,scenes=1088,positive_occurrences=8704,no_positive_scenes=64,tests=tests,
        original_decisions=report['decisions'],original_vision_milestone_gate=report['vision_milestone_gate'],
        no_accuracy_evaluation=True,no_new_accuracy_decision=True,no_model_calls=True,no_head_calls=True,
        interpretation=['Question means are descriptive statistics computed from the fresh test sample, never fed to any model.',
            'Every sampled positive occurrence is retained, including repeated images and zero-weight positives; no deduplication.',
            'Within-question variation can reflect image, position, length, and native execution; question support is reported explicitly.',
            'Small deviation is evidence of compatibility with a question-dependent weighted-count approximation at the original query, not a tested replacement.',
            'Large payload spread alone does not show that native outputs use it; no projection/readout intervention or accuracy evaluation occurs.',
            'All singular values are descriptive in the learned coordinates. No rank threshold, capacity theorem, or reasoning claim is inferred.',
            'Both the weighted spectra and unweighted companions are retained; zero-energy fractions and undefined means remain null.'])
    save(out/'analysis.json',result)
    lines=['# V18 positive payloads within each question','',
        'All four fixed models, all272 scenes per model and every2176 positive occurrence per model are retained. Original query only.',
        '', '| Model | Positive occurrences | Questions | Weighted centered energy / uncentered energy | Unweighted centered / uncentered | N32 mean relative P deviation | N64 mean relative P deviation |',
        '|---|---:|---:|---:|---:|---:|---:|']
    number=lambda x:'undefined' if x is None else f'{x:.6g}'
    for key in KEYS:
        s=models[key]['summary'];d=s['question_constant_deviation']
        lines.append(f"| {key} | {s['positive_occurrences']} | {s['question_groups']} | {number(s['weighted_centered_energy_fraction'])} | "
            f"{number(s['unweighted_centered_energy_fraction'])} | {number(d['N32']['relative_to_P']['mean'])} | {number(d['N64']['relative_to_P']['mean'])} |")
    lines+=['','The question means use fresh test data and are not a deployable approximation. No answer was generated, rescored, or changed.',
        '', 'Per-model JSON preserves all96 singular values/energy fractions, each question mean/spread/support, and every scene including K0. '
        'FP64 payload heterogeneity and saved FP32 multiplication residuals are recorded separately.',
        '', 'Near-constant values would support a weighted-count interpretation at this query. Variation alone would not establish its causal use by the native decoder.',
        '', '[Analysis and source identities](analysis.json) · [Consumed input hashes](input_bindings.json).']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return dict(passed=True,completed=True,source_sha256=frozen,policy=POLICY,tests=tests,
        analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),report_file=str(out/'REPORT.md'),
        scenes=1088,positive_occurrences=8704,no_positive_scenes=64,no_model_calls=True,no_head_calls=True,no_new_accuracy_decision=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--diagnosis',type=Path,default=DEFAULT)
    args=parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu'
         and not os.environ.get('SLURM_JOB_GPUS'),'This bounded analysis requires CPU Slurm')
    out=OUT/f'run_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False)
    frozen=snapshot(out);bindings={};start=time.perf_counter();result=None
    save(out/'invocation.json',dict(argv=sys.argv,diagnosis=str(args.diagnosis.resolve())))
    try:
        import torch
        torch.set_num_threads(4);tests=self_test(torch);save(out/'self_tests.json',tests)
        result=run(torch,args,out,frozen,bindings,tests)
    except BaseException as exc:
        result=dict(passed=False,completed=False,policy=POLICY,source_sha256=frozen,error_type=type(exc).__name__,
            error=str(exc),traceback=traceback.format_exc(),all_partial_artifacts_retained=True,no_model_calls=True,no_head_calls=True)
        raise
    finally:
        save(out/'input_bindings.json',bindings)
        result.update(elapsed_seconds=time.perf_counter()-start,slurm_job_id=os.environ['SLURM_JOB_ID'],
            input_bindings_file=str(out/'input_bindings.json'),input_bindings_sha256=sha(out/'input_bindings.json'))
        save(out/'summary.json',result)


if __name__=='__main__':main()

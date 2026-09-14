"""Fixed training-calibrated background-direction diagnostic; no fitting.

All heavy CPU/GPU work is Slurm-only. First token only, not whole-answer
accuracy. Use every selected V10 core, one fixed family/K, both N32/N64.
Only training K0 messages define directions. Test labels never set corrections.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
from scripts import native_vision_v7_runtime as native
from scripts import stage_native_vision_v10_features as feature
from scripts.stage_native_vision_v6_teacher import MODEL, need, read, save, sha

OUT = REPO/'outputs/native_aggregation_vlm/v10/background_diagnostic'
DATA = Path('/mnt/data/gabriele/gnn_transformer/v10_background_diagnostic')
REPORT = REPO/'outputs/native_aggregation_vlm/v10/report_442173/analysis.json'
FINAL = REPO/'outputs/native_aggregation_vlm/v10/finalization/final_442190/final_acceptance.json'
TEST = Path('/mnt/data/gabriele/gnn_transformer/v10_fresh/main_manifest.json')
CACHE = Path('/mnt/data/gabriele/gnn_transformer/v10_parallel_local/feature_cache.json')
OWN = tuple(dict.fromkeys(('scripts/probe_native_vision_v10_background.py',
    'slurm/native_vision_v10_background_check.sbatch', 'slurm/native_vision_v10_background_run.sbatch',
    'scripts/native_vision_v7_runtime.py', 'gnnformer/parallel_local_aggregation.py',
    'gnnformer/parallel_local_native.py', *feature.OWN)))
POLICY = dict(protocol='v10_training_background_first_token', conditions=['base','background','sham'],
    calibration='All occurrence-weighted K0 training image slots per question; empty query only',
    sham_seed=20261101, anchor_n=16, coefficient=1, families=17, scenes=34, cores=4,
    model_calls=34, visual_calls=34, extra_head_calls=442, generation_calls=0,
    bare_identity_tv_max=.02, bare_identity_top1_equal=True, no_fit=True,
    output_scope='First token only; K10..16 share first token1 and are not distinguishable by this endpoint')


def sources(): return {name: sha(REPO/name) for name in OWN}
def objsha(value): return native.object_sha(value)
def snapshot(out):
    (out/'code').mkdir()
    frozen=sources()
    for name,digest in frozen.items():
        payload=(REPO/name).read_bytes();need(hashlib.sha256(payload).hexdigest()==digest,'Source changed during snapshot')
        (out/'code'/name.replace('/','_')).write_bytes(payload)
    save(out/'source_hashes.json',frozen)
    (out/'INDEX.md').write_text('# V10 background diagnostic\n\n[Summary](summary.json) · [Sources](source_hashes.json)\n')
    index=OUT/'INDEX.md'
    if not index.exists(): index.write_text('# V10 background diagnostic\n\n')
    with index.open('a') as stream: stream.write(f'- [{out.name}]({out.name}/INDEX.md)\n')
    return frozen


def binding(path, artifacts, expected=None):
    path=Path(path).resolve();need(path.is_file(),'Missing input: '+str(path))
    digest=sha(path)
    need(expected is None or expected==digest,'Changed input: '+str(path))
    need(str(path) not in artifacts or artifacts[str(path)]==digest,'Inconsistent input binding')
    artifacts[str(path)]=digest
    return digest


def choose_records(manifest):
    need(set(manifest['splits'])=={'test_N32','test_N64'},'Unexpected V10 test cells')
    by=defaultdict(dict)
    for n in (32,64):
        rows=manifest['splits'][f'test_N{n}']['samples']
        need(len(rows)==136 and Counter(row['gold'] for row in rows)==Counter({k:8 for k in range(17)}),
             'Unexpected complete V10 test coverage')
        for row in rows:
            pid=row['pair_id'];need(row['anchor_id']==pid and row['n_frames']==n and row['split']=='test',
                                   'Family/length/split metadata differs')
            need(n not in by[pid],'Duplicate family/length');by[pid][n]=row
    need(len(by)==136 and all(set(x)=={32,64} for x in by.values()),'Incomplete length family')
    result=[]
    for k in range(17):
        choices=sorted(pid for pid,members in by.items() if members[32]['gold']==k)
        need(len(choices)==8,'Missing count families');members=by[choices[0]]
        need(members[32]['question']==members[64]['question'] and members[64]['gold']==k,'Family target changed')
        result.extend(members[n] for n in (32,64))
    need(len(result)==34 and len({row['sid'] for row in result})==34,'Fixed selection differs')
    return result


def verify_qa(row,artifacts):
    path=Path(row['path'])/'qa.txt';binding(path,artifacts,row['qa_sha256'])
    lines=path.read_text().splitlines();a,b=lines.index('question:'),lines.index('answer:')
    contents=[line.strip() for line in lines[a+1:b] if line.strip()]
    states=[ast.literal_eval(line) for line in contents if line.startswith('{')]
    need([line for line in contents if not line.startswith('{')]==[row['question']]
         and len(states)==row['n_frames'] and int(lines[b+1])==row['gold'], 'QA target/layout differs')
    need(objsha(dict(states=states,question=row['question']))==row['content_sha256'],'QA content hash differs')
    positive=0
    for i,state in enumerate(states):
        room,people=next(iter(state['rooms'].items()))
        need(len(state['rooms'])==len(people)==1 and state['step_id']==i+1,'Unexpected frame semantics')
        positive+=room==row['target_room'] and people[0]==row['target_character']
    need(positive==row['gold'],'Independent conjunction recount differs')


def orthogonal_sham(torch,b0,key):
    """Fixed FP64 projection, then FP32 direction; no global RNG consumption."""
    seed=int(objsha([POLICY['sham_seed'],key])[:16],16)%(2**63-1)
    generator=torch.Generator(device='cpu');generator.manual_seed(seed)
    b=b0.detach().cpu().double();norm=b.norm()
    if float(norm)==0:
        return torch.zeros_like(b0),dict(seed=seed,zero_background=True,relative_dot=0.,relative_norm_error=0.)
    u=b/norm;noise=torch.randn(b.shape,generator=generator,dtype=torch.float64)
    perpendicular=noise-(noise@u)*u
    if float(perpendicular.norm())<1e-12:
        axis=torch.zeros_like(b);axis[int(u.abs().argmin())]=1
        perpendicular=axis-(axis@u)*u
    result=(perpendicular/perpendicular.norm()*norm).float().to(b0.device)
    dot=float((result.double().cpu()@b)/(result.double().cpu().norm()*norm))
    error=float(abs(result.double().cpu().norm()-norm)/norm)
    need(abs(dot)<1e-6 and error<1e-6,'Sham orthogonality/norm construction failed')
    return result,dict(seed=seed,zero_background=False,relative_dot=dot,relative_norm_error=error)


def shifted_r(r,direction,n):
    # Test N and fixed training anchor are the only scalar inputs.
    return r-(n-POLICY['anchor_n'])*direction.reshape(1,-1)


def self_test():
    import torch
    from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
    state=torch.random.get_rng_state().clone()
    b=torch.tensor([1.,2.,-3.,.5]);a,meta=orthogonal_sham(torch,b,'fixture')
    a2,_=orthogonal_sham(torch,b,'fixture')
    need(torch.equal(a,a2) and torch.equal(state,torch.random.get_rng_state()),'Sham changed RNG/determinism')
    z,_=orthogonal_sham(torch,torch.zeros(4),'fixture');need(torch.equal(z,torch.zeros(4)),'Zero direction differs')
    r=torch.tensor([[2.,3.,4.,5.]])
    need(torch.equal(shifted_r(r,b,16),r) and torch.equal(shifted_r(r,b,32),r-16*b),'Correction sign/anchor differs')
    # Image occurrence weighting, not an unweighted mean over N8/N16 scenes.
    values=torch.cat((torch.ones(8,4),torch.full((16,4),4.)),0)
    need(torch.equal(values.double().mean(0),torch.full((4,),3.,dtype=torch.float64)),'Occurrence weighting differs')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31);core=ParallelLocalAggregation(8,rank=4).eval()
        with torch.no_grad():core.up.weight.normal_(0,.1)
        h=torch.randn(3,1,8);g=torch.randn(1,8)
        messages=core.encode(h,g);pooled=core.aggregate(messages)
        pre=core.aggregate_projection(pooled)+core.query(core.rms(g))
        manual=core.up(torch.nn.functional.silu(pre))
        need(torch.equal(manual,core(h,g,output_dtype=torch.float32)),'Baseline decomposition differs')
        need(torch.equal(core.up(torch.nn.functional.silu(shifted_r(pre,torch.ones(4),16))),manual),
             'N16 intervention is not exactly identity')
    return dict(passed=True,checks=6,scope='Direction algebra, RNG, occurrence weights and exact branch decomposition')


def load_selected(torch,report,final,artifacts):
    from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
    expected={f'{c}_s{s}' for c in ('ce','consistency') for s in (14,15)}
    need(set(report['runs'])==expected,'All four canonical selected models required')
    finalized={(r['condition'],r['seed']):r for r in final['selected_models']}
    runs=[];branches={};original={}
    for key in sorted(expected):
        audit=report['runs'][key];directory=Path(audit['run_directory'])
        for filename,label in (('config.json','config_sha256'),('summary.json','summary_sha256')):
            binding(directory/filename,artifacts,audit[label])
        config=read(directory/'config.json');summary=read(directory/'summary.json')
        selection=read(directory/'selection.json');binding(directory/'selection.json',artifacts)
        need(config['condition']==audit['condition'] and config['seed']==audit['seed'] and config['arm']=='parallel'
             and summary['passed'] and summary['completed'],'Run condition/completion differs')
        selected=min(selection['development'],key=lambda x:(-x['exact_count'],x['nll'],x['step']))
        need(selected==selection['selected']==summary['selected'],'Development selection differs')
        need(selected['checkpoint']==audit['selected_checkpoint'] and selected['checkpoint_sha256']==audit['selected_checkpoint_sha256']
             and selected['parameter_sha256']==audit['selected_parameter_sha256'],'Report checkpoint identity differs')
        finalrow=finalized[(config['condition'],config['seed'])]
        need(finalrow['checkpoint']==selected['checkpoint'] and finalrow['checkpoint_sha256']==selected['checkpoint_sha256']
             and finalrow['parameter_sha256']==selected['parameter_sha256'],'Final selected identity differs')
        for entry in selection['development']:binding(entry['dev_file'],artifacts,entry['dev_sha256'])
        binding(selected['checkpoint'],artifacts,selected['checkpoint_sha256'])
        saved=torch.load(selected['checkpoint'],map_location='cpu',weights_only=True)
        need(saved['config']==config and saved['step']==selected['step']
             and objsha({k:native.tensor_info(v) for k,v in saved['branch'].items()})==selected['parameter_sha256'],
             'Checkpoint contents differ')
        branch=ParallelLocalAggregation().eval().requires_grad_(False);branch.load_state_dict(saved['branch'],strict=True)
        need(all(p.dtype==torch.float32 and bool(torch.isfinite(p).all()) for p in branch.parameters()),'Invalid selected core')
        branches[key]=branch
        testfile=directory/'test.json';binding(testfile,artifacts,audit['test_sha256']);test=read(testfile)
        binding(test['raw_file'],artifacts,test['raw_sha256'])
        original[key]=dict(metadata=test,raw_file=test['raw_file'])
        for name,digest in config['source_sha256'].items():binding(REPO/name,artifacts,digest)
        runs.append(dict(key=key,condition=config['condition'],seed=config['seed'],run_directory=str(directory),
                         config=config,selected=selected))
    return runs,branches,original


def calibrate(torch,cache,plan,branches,artifacts):
    source=plan['source_files']['training_manifest'];binding(source['path'],artifacts,source['sha256'])
    manifest=read(source['path']);schedule_ref=plan['source_files']['training_schedule']
    binding(schedule_ref['path'],artifacts,schedule_ref['sha256']);schedule=read(schedule_ref['path'])
    rows={r['sid']:r for n in (8,16) for r in manifest['splits'][f'train_N{n}']['samples']}
    need(len(rows)==1782 and set(rows)==set(cache['scenes']),'Training inventory differs')
    slots=[sid for sid in schedule['epoch_slots'] if rows[sid]['gold']==0]
    need(len(slots)==108,'K0 weighted scene slots differ')
    groups=defaultdict(list);wanted=set();ledger=[]
    for sid in slots:
        row=rows[sid];scene=cache['scenes'][sid];verify_qa(row,artifacts)
        need(all(scene[k]==row[k] for k in ('question','gold','n_frames','qa_sha256')),'Calibration scene identity differs')
        ids=[x[0] for x in scene['local_feature_ids']];gid=scene['global_feature_ids'][0]
        need(len(ids)==row['n_frames'] and scene['target_prefixes'][0]==[],'Calibration must use all empty-query image slots')
        for i,fid in enumerate(ids):
            f=plan['features'][fid];pair=plan['pairs'][f['pair_id']]
            need(f['kind']=='local' and f['prefix_ids']==[] and f['question']==row['question']
                 and pair['image_sha256']==row['image_files'][i]['sha256'],'Calibration image/prefix binding differs')
        need(plan['features'][gid]['kind']=='global' and plan['features'][gid]['prefix_ids']==[]
             and plan['features'][gid]['question']==row['question'],'Calibration global query differs')
        groups[row['question']].append((ids,gid));wanted.update(ids);wanted.add(gid)
        ledger.append(dict(sid=sid,question=row['question'],n_frames=row['n_frames'],gold=0,
                           local_feature_ids=ids,global_feature_id=gid))
    need(len(groups)==54 and sum(len(ids) for group in groups.values() for ids,_ in group)==1296,
         'Calibration must cover54 questions and1296 occurrence-weighted image slots')
    states={}
    files={cache['features'][fid]['file']:cache['features'][fid]['file_sha256'] for fid in wanted}
    for path,digest in files.items():
        binding(path,artifacts,digest);blob=torch.load(path,map_location='cpu',weights_only=True)
        need(blob['states'].dtype==torch.float16,'Frozen feature dtype differs')
        for fid in sorted(wanted):
            loc=cache['features'][fid]
            if loc['file']!=path:continue
            value=blob['states'][loc['row']]
            need(blob['feature_ids'][loc['row']]==fid and native.tensor_info(value)['sha256']==loc['state_sha256']
                 and value.shape==(3584,) and bool(torch.isfinite(value).all()),'Calibration state hash/shape differs')
            states[fid]=value.clone()
        del blob
    result={}
    with torch.inference_mode():
        for key,branch in branches.items():
            result[key]={}
            for question,group in sorted(groups.items()):
                gids={gid for _,gid in group};need(len(gids)==1,'Global feature depends on train N')
                gid=next(iter(gids));g=states[gid].unsqueeze(0)
                messages=[branch.encode(torch.stack([states[fid] for fid in ids]).unsqueeze(1),g)[:,0]
                          for ids,_ in group]
                all_messages=torch.cat(messages,0);need(all_messages.shape==(24,96),'K0 slots per question differ')
                mu64=all_messages.double().mean(0);mu=mu64.float()
                b0=torch.nn.functional.linear(mu,branch.aggregate_projection.weight)
                sham,geometry=orthogonal_sham(torch,b0,[key,question])
                result[key][question]=dict(mu0=mu,mu0_fp64=mu64,b0=b0,sham=sham,
                    global_state=g,global_feature_id=gid,image_slots=24,geometry=geometry,
                    calibration_messages=all_messages)
    return result,ledger


def check(args,out,frozen):
    import torch,transformers
    from transformers import AutoProcessor
    from scripts.probe_native_vision_v2_prefix import fingerprint
    torch.set_num_threads(4);tests=self_test();artifacts={}
    report_path=args.report.resolve();binding(report_path,artifacts);report=read(report_path)
    completion_path=report_path.parent/'summary.json';binding(completion_path,artifacts);completion=read(completion_path)
    need(report['passed'] and report['audit_passed'] and completion['passed']
         and completion['analysis_sha256']==sha(report_path),'Verified independent V10 report required')
    binding(args.finalization,artifacts);final=read(args.finalization)
    need(final['verification_passed'] and final['completed'],'Completed V10 verification required; efficacy pass is not required')
    need(final['artifacts']['analysis']['path']==str(report_path)
         and final['artifacts']['analysis']['sha256']==sha(report_path),'Final report binding differs')
    for ref in final['artifacts'].values():binding(ref['path'],artifacts,ref['sha256'])
    for name,digest in report['source_sha256'].items():binding(REPO/name,artifacts,digest)
    runs,branches,original=load_selected(torch,report,final,artifacts)
    binding(CACHE,artifacts);cache=read(CACHE);binding(cache['plan_file'],artifacts,cache['plan_sha256']);feature_plan=read(cache['plan_file'])
    need(cache['complete'] and cache['training_only'] and cache['protocol']=='v10_parallel_local_training_features'
         and cache['scenes']==feature_plan['scenes'],'Frozen training cache identity differs')
    for value in feature_plan['source_files'].values():binding(value['path'],artifacts,value['sha256'])
    for name,digest in cache['source_sha256'].items():binding(REPO/name,artifacts,digest)
    for path,digest in cache['native_api']['source_sha256'].items():binding(path,artifacts,digest)
    for run in runs:
        need(run['config']['cache_binding']==dict(file=str(CACHE),sha256=sha(CACHE)), 'Selected core used a different training cache')
        need(all(run['config'][key]==cache[key] for key in ('model','runtime','processor','native_dtypes')),'Native identities differ')
    need(feature.model_metadata()==cache['model'],'Actual model metadata/weight stat differs')
    processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,use_fast=False,local_files_only=True)
    need(fingerprint(processor,str(transformers.__version__))==cache['processor'],'Actual processor differs')
    need(feature.runtime_identity()==cache['runtime'],'Actual CPU runtime differs')
    _,_,api=feature.native_api(processor);need(api==cache['native_api'],'Actual native API differs')
    targets={str(k):native.encode_target(processor.tokenizer,k) for k in range(17)}
    need(targets==cache['target_token_ids'],'First-token targets differ from trained native numeral support')
    calibration,ledger=calibrate(torch,cache,feature_plan,branches,artifacts)
    data=DATA/out.name;data.mkdir(parents=True,exist_ok=False)
    calibration_path=data/'calibration.pt';torch.save(dict(schema_version=1,models=calibration,slots=ledger),calibration_path)
    binding(calibration_path,artifacts);save(out/'calibration_slots.json',ledger)
    binding(out/'calibration_slots.json',artifacts)
    binding(TEST,artifacts);test_manifest=read(TEST);records=choose_records(test_manifest)
    for key in ('audit_file','stage_audit_file'):
        if key in test_manifest:binding(test_manifest[key],artifacts,test_manifest.get('audit_sha256') if key=='audit_file' else None)
    prepared=[];(data/'prepared').mkdir()
    for row in records:
        verify_qa(row,artifacts)
        for image in row['image_files']:binding(image['path'],artifacts,image['sha256'])
        bundle=native.prepare_scene(processor,row,'parallel',prefix_ids=(),verify_processor_parity=True)
        need(bundle['metadata']['prefix_ids']==[] and bundle['metadata']['question']==row['question'],'Actual prompt contains a prefix')
        path=data/'prepared'/f'{row["sid"]}.pt';torch.save(bundle,path);binding(path,artifacts)
        prepared.append(dict(record=row,bundle_file=str(path),bundle_sha256=sha(path),metadata=bundle['metadata']))
    originals={}
    for run in runs:
        key=run['key'];test=original[key]['metadata'];rows={r['sid']:r for r in test['rows']}
        blob=torch.load(test['raw_file'],map_location='cpu',weights_only=True);originals[key]={}
        for row in records:
            old=rows[row['sid']];vector=blob['raw_logits'][old['raw_index']][0]
            need(vector.ndim==1 and vector.dtype==torch.float32 and bool(torch.isfinite(vector).all()),'Archived first logit vector differs')
            need(old['gold']==row['gold'] and old['content_sha256']==row['content_sha256'],'Archived test SID/content differs')
            originals[key][row['sid']]=dict(logits=vector.clone(),row=old)
        del blob
    original_path=data/'original_logits.pt';torch.save(dict(schema_version=1,models=originals),original_path);binding(original_path,artifacts)
    plan=dict(schema_version=1,policy=POLICY,source_sha256=frozen,artifact_sha256=artifacts,runs=runs,
        report_file=str(report_path),report_sha256=sha(report_path),finalization_file=str(args.finalization.resolve()),
        finalization_sha256=sha(args.finalization),calibration_file=str(calibration_path),calibration_sha256=sha(calibration_path),
        calibration_slots_file=str(out/'calibration_slots.json'),calibration_slots_sha256=sha(out/'calibration_slots.json'),
        original_file=str(original_path),original_sha256=sha(original_path),cases=prepared,target_token_ids=targets,
        model=cache['model'],runtime=cache['runtime'],processor=cache['processor'],native_dtypes=cache['native_dtypes'],
        native_api=cache['native_api'],training_cache=dict(file=str(CACHE),sha256=sha(CACHE)),tests=tests,
        original_v10_practical_passed=final['practical_both_seeds'],original_v10_strict_cache_passed=final['post_strict_cache_numerical_gate_passed'])
    save(out/'plan.json',plan);(out/'plan.sha256').write_text(sha(out/'plan.json')+'\n')
    return dict(passed=True,plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),tests=tests,
                scenes=34,models=4,calibration_questions=54,calibration_image_slots=1296,no_fit=True)


def verify_plan(path):
    path=Path(path);plan=read(path)
    need(path.with_suffix('.sha256').read_text().strip()==sha(path),'Plan sidecar differs')
    need(plan['schema_version']==1 and plan['policy']==POLICY and plan['source_sha256']==sources(),'Plan/source/policy differs')
    need(len(plan['cases'])==34 and {(r['condition'],r['seed']) for r in plan['runs']}=={(c,s) for c in ('ce','consistency') for s in (14,15)},'Frozen diagnostic coverage differs')
    gate=read(path.parent/'summary.json');need(gate['passed'] and gate['plan_sha256']==sha(path),'Matching CPU gate required')
    for name,digest in plan['artifact_sha256'].items():need(sha(name)==digest,'Bound input changed: '+name)
    need(feature.model_metadata()==plan['model'] and feature.runtime_identity()==plan['runtime'],'Actual native model/runtime changed')
    return plan


def metrics(torch,reference,value):
    a,b=reference.detach().double(),value.detach().double();need(a.shape==b.shape and a.ndim==1,'Logit comparison shape differs')
    need(bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()),'Nonfinite logit comparison')
    d=b-a;center=d-d.mean();pa=a.softmax(-1);pb=b.softmax(-1)
    return dict(raw_max=float(d.abs().max()),raw_rms=float(d.square().mean().sqrt()),mean_shift=float(d.mean()),
                centered_max=float(center.abs().max()),centered_rms=float(center.square().mean().sqrt()),
                tv=float((pa-pb).abs().sum()/2),top1_equal=int(a.argmax())==int(b.argmax()),
                reference_top1=int(a.argmax()),observed_top1=int(b.argmax()))


def run(args,out,frozen):
    import torch,transformers
    from gnnformer.runtime import load_runtime
    from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
    from scripts.probe_native_vision_v2_prefix import fingerprint
    torch.set_num_threads(4);plan=verify_plan(args.plan)
    need(torch.cuda.device_count()==1,'Exactly one visible GPU required')
    data=DATA/out.name;data.mkdir(parents=True,exist_ok=False)
    (out/'plan.json').write_bytes(Path(args.plan).read_bytes())
    calibration=torch.load(plan['calibration_file'],map_location='cpu',weights_only=True)['models']
    originals=torch.load(plan['original_file'],map_location='cpu',weights_only=True)['models']
    tick=time.perf_counter();runtime=load_runtime(str(MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda')
    model=runtime.model;model.eval().requires_grad_(False);norm=native.native_contract(model)
    need(fingerprint(runtime.processor,str(transformers.__version__))==plan['processor'],'Loaded processor differs')
    _,_,api=feature.native_api(runtime.processor);need(api==plan['native_api'],'Loaded native API differs')
    branches={}
    for record in plan['runs']:
        blob=torch.load(record['selected']['checkpoint'],map_location='cpu',weights_only=True)
        branch=ParallelLocalAggregation().to(device=runtime.device).eval().requires_grad_(False)
        branch.load_state_dict(blob['branch'],strict=True);native.native_contract(model,branch);branches[record['key']]=branch
    torch.cuda.synchronize();load_seconds=time.perf_counter()-tick
    counts=dict(model=0,visual=0,extra_head=0);rows=[];observations=[];bare_passes=[];decomposition_passes=[]
    def head_read(hidden):
        counts['extra_head']+=1
        return model.lm_head(norm.forward(hidden))[-1,0]
    with torch.inference_mode():
        for case in plan['cases']:
            sample=case['record'];sid=sample['sid'];n=sample['n_frames'];q=sample['question']
            bundle=torch.load(case['bundle_file'],map_location='cpu',weights_only=True)
            need(bundle['metadata']==case['metadata'],'Prepared bundle metadata differs')
            captured=native.forward_native(model,bundle,branch=None,capture=True,cpu=False)
            counts['model']+=captured['counters']['model'];counts['visual']+=captured['counters']['visual']
            local,g=captured['local_states'],captured['global_states'];h=torch.cat((local,g.unsqueeze(0)),0)
            need(h.shape==(n+1,1,3584) and h.dtype==torch.float16,'Captured native full query batch differs')
            bare=head_read(h);identity=metrics(torch,captured['global_logits'],bare)
            bare_pass=identity['tv']<=POLICY['bare_identity_tv_max'] and identity['top1_equal'];bare_passes.append(bare_pass)
            payload=dict(schema_version=1,sid=sid,record=sample,metadata=captured['metadata'],
                raw_query_states=h.detach().cpu(),native_global_logits=captured['global_logits'].detach().cpu(),
                bare_replay_logits=bare.detach().cpu(),bare_identity=identity,models={})
            for selected in plan['runs']:
                key=selected['key'];branch=branches[key];cal=calibration[key][q]
                messages=branch.encode(local,g);z=branch.aggregate(messages)
                r=branch.aggregate_projection(z)+branch.query(branch.rms(g))
                base_delta=branch.up(torch.nn.functional.silu(r))
                direct_delta=branch(local,g,output_dtype=torch.float32)
                decomposition_equal=torch.equal(base_delta,direct_delta)
                decomposition_passes.append(decomposition_equal)
                decomposition_error=float((base_delta-direct_delta).abs().max())
                directions=dict(base=torch.zeros(96,device=r.device),background=cal['b0'].to(r.device),sham=cal['sham'].to(r.device))
                modelsave=dict(messages=messages.detach().cpu(),z=z.detach().cpu(),r=r.detach().cpu(),
                    training_mu0=cal['mu0'],background_direction=cal['b0'],sham_direction=cal['sham'],
                    training_global_state=cal['global_state'],calibration_geometry=cal['geometry'],
                    baseline_decomposition_exact=decomposition_equal,baseline_decomposition_maximum_error=decomposition_error,
                    global_state_maximum_difference=float((g.float()-cal['global_state'].to(g.device).float()).abs().max()),
                    projection_replay_maximum_difference=float((torch.nn.functional.linear(cal['mu0'].to(g.device),branch.aggregate_projection.weight)-cal['b0'].to(g.device)).abs().max()),
                    interventions={})
                base_logits=None
                for intervention in POLICY['conditions']:
                    changed=r if intervention=='base' else shifted_r(r,directions[intervention],n)
                    delta=base_delta if intervention=='base' else branch.up(torch.nn.functional.silu(changed))
                    fused=h.clone();fused[-1,0]=g[0]+delta.to(g.dtype)[0]
                    logits=head_read(fused);target=plan['target_token_ids'][str(sample['gold'])][0]
                    logp=logits.double().log_softmax(-1);top=int(logits.argmax())
                    if intervention=='base':base_logits=logits.clone()
                    comparison=metrics(torch,base_logits,logits)
                    oldcompare=metrics(torch,originals[key][sid]['logits'].to(logits.device),logits) if intervention=='base' else None
                    row=dict(model=key,condition=selected['condition'],seed=selected['seed'],intervention=intervention,
                        sid=sid,pair_id=sample['pair_id'],n_frames=n,gold=sample['gold'],target_first_token=target,
                        first_token_id=top,first_token_text=runtime.tokenizer.decode([top],skip_special_tokens=False),
                        first_token_correct=top==target,first_token_nll=float(-logp[target]),
                        first_token_scope='single_digit_0_9' if sample['gold']<=9 else 'shared_first1_K10_16',
                        comparison_to_base=comparison,comparison_to_original_v10=oldcompare,
                        intervention_norm=float((changed-r).norm()),delta_norm=float(delta.norm()))
                    rows.append(row);modelsave['interventions'][intervention]=dict(r=changed.detach().cpu(),
                        delta=delta.detach().cpu(),global_logits=logits.detach().cpu(),row=row)
                payload['models'][key]=modelsave
            path=data/f'{sid}.pt';torch.save(payload,path)
            observation=dict(sid=sid,pair_id=sample['pair_id'],n_frames=n,gold=sample['gold'],raw_file=str(path),raw_sha256=sha(path),
                bare_identity=identity,bare_identity_passed=bare_pass,
                baseline_decompositions_exact={key:value['baseline_decomposition_exact'] for key,value in payload['models'].items()},
                counters=captured['counters'],model_seconds=captured['model_seconds'])
            observations.append(observation)
            with (out/'observations.jsonl').open('a') as stream:stream.write(json.dumps(observation,allow_nan=False)+'\n')
            print(json.dumps(dict(sid=sid,n=n,bare_identity_passed=bare_pass,completed_scenes=len(observations))),flush=True)
            del payload,bundle,captured,h,local,g,modelsave,logits,base_logits
    need(counts==dict(model=34,visual=34,extra_head=442) and len(rows)==408 and len(observations)==34,'Fixed call/coverage budget differs')
    summary_rows=[]
    for key in sorted(branches):
        for n in (32,64):
            for intervention in POLICY['conditions']:
                for scope in ('all','single_digit_0_9','shared_first1_K10_16'):
                    selected=[r for r in rows if r['model']==key and r['n_frames']==n and r['intervention']==intervention
                              and (scope=='all' or r['first_token_scope']==scope)]
                    summary_rows.append(dict(model=key,n_frames=n,intervention=intervention,scope=scope,n=len(selected),
                        first_token_correct=sum(r['first_token_correct'] for r in selected),
                        first_token_nll=sum(r['first_token_nll'] for r in selected)/len(selected)))
    passed=all(bare_passes) and all(decomposition_passes)
    analysis=dict(schema_version=1,passed=passed,computational_integrity_passed=passed,diagnostic_only=True,no_fit=True,
        policy=POLICY,source_sha256=frozen,plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),
        counters=counts,load_seconds=load_seconds,rows=rows,summaries=summary_rows,observations=observations,
        bare_identity_passes=sum(bare_passes),bare_identity_failures=len(bare_passes)-sum(bare_passes),
        baseline_decomposition_passes=sum(decomposition_passes),baseline_decomposition_observations=len(decomposition_passes),
        limitations=['First-token recognition is not full integer/EOS accuracy; K10..16 share token1.',
            'Correction uses training K0 labels, an explicit N16 anchor and a fixed multiplier; it is a diagnostic, not a proposed inference method.',
            'The reused V10 test subset is exploratory and was selected by family identifier, never model response.',
            'Calibration estimates a first moment from24 image occurrences per question; a null result does not imply absent information.',
            'One fixed orthogonal sham controls intervention magnitude, not every possible nonspecific effect.',
            'CPU training-feature and GPU live-feature numerical differences and original V10 logit differences remain descriptive.',
            'Only first-query native states are captured; no temporal reasoning or state-persistence conclusion follows.',
            'Full N+1 query states and every global raw-logit vector are saved; unused local vocabulary outputs are not archived.'])
    save(out/'analysis.json',analysis)
    lines=['# V10 background-direction diagnostic','',
        'First-token results on17 fixed families at both lengths, all four selected cores. No fitting or generation.','',
        '| Model | N | Intervention | First-token correct | Mean first-token NLL |','|---|---:|---|---:|---:|']
    for item in summary_rows:
        if item['scope']=='all':lines.append(f"| {item['model']} | {item['n_frames']} | {item['intervention']} | {item['first_token_correct']}/{item['n']} | {item['first_token_nll']:.4f} |")
    lines.extend(['','K10–16 all share first token1; these results do not establish exact count decoding.',
        f"Bare native replay checks passed {sum(bare_passes)}/34. Original V10 comparisons are descriptive; original failures remain unchanged.",'',
        '[All K/length/model/intervention rows, raw tensors and limits](analysis.json).'])
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return dict(passed=passed,computational_integrity_passed=passed,diagnostic_only=True,no_fit=True,counters=counts,
        analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),bare_identity_passes=sum(bare_passes),
        bare_identity_failures=34-sum(bare_passes),scenes=34,first_token_observations=408,baseline_decomposition_passes=sum(decomposition_passes))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True);modes.add_argument('--check',action='store_true');modes.add_argument('--run',action='store_true')
    parser.add_argument('--plan',type=Path);parser.add_argument('--report',type=Path,default=REPORT)
    parser.add_argument('--finalization',type=Path,default=FINAL);args=parser.parse_args()
    partition='cpu' if args.check else 'gpu'
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')==partition,'All work requires the correct Slurm partition')
    need(bool(os.environ.get('SLURM_JOB_GPUS'))==(partition=='gpu'),'Unexpected GPU allocation')
    need(int(os.environ.get('SLURM_CPUS_PER_TASK','1'))<=4,'Bounded diagnostic uses at most4 CPUs')
    start=time.perf_counter();out=OUT/f'{"check" if args.check else "run"}_{os.environ["SLURM_JOB_ID"]}'
    out.mkdir(parents=True,exist_ok=False);frozen=snapshot(out)
    if args.run:need(args.plan is not None,'GPU run requires a completed exact CPU plan')
    result=check(args,out,frozen) if args.check else run(args,out,frozen)
    need(sources()==frozen,'Sources changed during diagnostic')
    result.update(source_sha256=frozen,slurm_job_id=os.environ['SLURM_JOB_ID'],elapsed_seconds=time.perf_counter()-start)
    save(out/'summary.json',result)
    print(json.dumps(dict(passed=result['passed'],summary_file=str(out/'summary.json'))),flush=True)
    if not result['passed']:raise SystemExit('Bare native identity check failed; all observations preserved')


if __name__=='__main__':main()

"""V7 matched cached-state training; evaluation always executes the native VLM.

The cache contains frozen causal states, never classifier/tally predictions.
No backbone parameter is updated. Only native count-token plus EOS CE trains
both identical cores. Every native inference step reads original image inputs.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts.stage_native_vision_v7_features import need,read,sha,object_sha,MODEL,tensor_info
from scripts import native_vision_v7_runtime as native
DATA=Path('/mnt/data/gabriele/gnn_transformer')
OUT=REPO/'outputs/native_aggregation_vlm/v7'
CKPT=Path('/mnt/ckpts/gabriele/gnn_transformer/native_aggregation_vlm_v7')
POLICY=dict(arms=['parallel','joint'],seeds=[8,9],epochs=40,slots_per_epoch=1944,batch_size=16,
    steps=4860,dev_steps=[972,1944,2916,3888,4860],lr=.001,warmup=50,final_lr=.00001,
    weight_decay=0.,clip_norm=1.,rank=96,hidden_size=3584,parameters=1041600,
    merge='sum',post_activation='silu',native_dtype='torch.float16',branch_dtype='torch.float32',
    max_new_tokens=4,exact_requires_eos=True,selection='max_dev_exact_then_min_raw_first_token_NLL_then_earliest',
    native_eos=[151645,151643],target_eos=151645,bootstrap_seed=20260928,bootstrap_replicates=10000)
OWN=('scripts/train_native_vision_v7.py','scripts/native_vision_v7_runtime.py',
     'gnnformer/parallel_local_aggregation.py','gnnformer/parallel_local_native.py',
     'gnnformer/parallel_local_prompts.py','gnnformer/runtime.py','gnnformer/data.py',
     'slurm/native_vision_v7_train_check.sbatch','slurm/native_vision_v7_train_profile.sbatch',
     'slurm/native_vision_v7_train.sbatch')

def save(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.temporary')
    with temporary.open('w') as stream:json.dump(value,stream,indent=2,allow_nan=False)
    temporary.replace(path)

def sources():return {p:sha(REPO/p) for p in OWN}
def snapshot(out):
    (out/'code').mkdir()
    for p in OWN:(out/'code'/p.replace('/','_')).write_bytes((REPO/p).read_bytes())
    save(out/'source_hashes.json',sources())
def state_info(branch):return {k:tensor_info(v) for k,v in branch.state_dict().items()}
def lr(step):
    need(1<=step<=4860,'Step out of range')
    return .001*step/50 if step<=50 else 1e-5+(.001-1e-5)*(1+math.cos(math.pi*(step-50)/4810))/2

def presentation_order(slots,seed):
    rng=random.Random(seed);order=[]
    for epoch in range(40):
        permutation=list(range(len(slots)));rng.shuffle(permutation)
        order.extend(dict(epoch=epoch+1,slot=i,sid=slots[i]) for i in permutation)
    need(len(order)==77760,'Training presentation count differs')
    return order

def cache_binding(path):
    value=read(path)
    need(value['complete'] and value['training_only'] and len(value['scenes'])==1890,'Incomplete training-only cache')
    need(sha(value['plan_file'])==value['plan_sha256'],'Cache plan changed')
    for p,h in value['source_sha256'].items():need(sha(REPO/p)==h,'Cache source changed: '+p)
    if 'release_file' in value:
        need(sha(value['release_file'])==value['release_sha256'],'Joint cache release changed')
        released=read(value['release_file']);need(released['passed'] and released['full_harvest_authorized']
            and released['source_sha256']==value['source_sha256'],'Joint cache release authorization differs')
    if 'runtime_extension' in value:
        ext=value['runtime_extension'];need(sha(ext['release_file'])==ext['release_sha256'],'Runtime release changed')
        for p,h in ext['dispatcher_source_sha256'].items():need(sha(REPO/p)==h,'Cache runtime dispatcher source changed')
    return value

def load_features(torch,cache,device):
    files={row['file']:row['file_sha256'] for row in cache['features'].values()};blobs={}
    for path,digest in files.items():
        need(sha(path)==digest,'Cached state file changed')
        blobs[path]=torch.load(path,map_location='cpu',weights_only=True)
    ids=sorted(cache['features']);states=[]
    for fid in ids:
        item=cache['features'][fid];blob=blobs[item['file']];h=blob['states'][item['row']]
        need(blob['feature_ids'][item['row']]==fid and tensor_info(h)['sha256']==item['state_sha256']
             and h.shape==(3584,) and h.dtype==torch.float16,'Cached feature identity differs')
        states.append(h)
    values=torch.stack(states).to(device);need(bool(torch.isfinite(values).all()),'Nonfinite frozen states')
    return values,{fid:i for i,fid in enumerate(ids)}

def batch_states(torch,cache,states,index,sids,arm):
    local=[];global_rows=[];targets=[]
    for sid in sids:
        scene=cache['scenes'][sid];targets.extend(scene['target_ids'])
        if arm=='parallel':
            ids=scene['local_feature_ids'];gids=scene['global_feature_ids']
            h=states[torch.tensor([[index[f] for f in row] for row in ids],device=states.device)]
            if len(ids)<16:h=torch.cat((h,torch.zeros(16-len(ids),2,3584,device=h.device,dtype=h.dtype)))
            g=states[torch.tensor([index[f] for f in gids],device=states.device)]
        else:
            g=states[torch.tensor([index[f] for f in scene['feature_ids']],device=states.device)];h=g.unsqueeze(0)
        local.append(h);global_rows.append(g)
    return torch.cat(local,dim=1),torch.cat(global_rows),torch.tensor(targets,device=states.device)

def prediction_record(torch,processor,sample,result,raw_index,cell):
    ids=native.encode_target(processor.tokenizer,sample['gold']);logits=result['raw_logits'].float()
    parsed=int(result['text'].strip()) if re.fullmatch(r'[0-9]+',result['text'].strip()) else None
    return dict(sid=sample['sid'],cell=cell,n_frames=sample['n_frames'],gold=sample['gold'],
        anchor_id=sample.get('anchor_id'),content_sha256=sample['content_sha256'],
        generated_ids=result['generated_ids'],raw_text=result['raw_text'],text=result['text'],
        prediction=parsed,exact=result['completed'] and parsed==sample['gold'],parsed_count_correct=parsed==sample['gold'],parseable=parsed is not None,
        completed=result['completed'],truncated=result['truncated'],counters=result['counters'],
        first_token_nll=float(torch.logsumexp(logits[0],-1)-logits[0,ids[0]]),
        raw_index=raw_index,metadata=result['metadata'],model_seconds=result['model_seconds'])

def evaluate(torch,model,processor,branch,rows,arm,out,label,dataout):
    records=[];raw=[];begin=time.perf_counter()
    for cell,sample in rows:
        bundle=native.prepare_scene(processor,sample,arm)
        result=native.generate_native(model,processor,branch,bundle)
        records.append(prediction_record(torch,processor,sample,result,len(raw),cell));raw.append(result['raw_logits'])
        if len(records)%36==0:print(json.dumps(dict(evaluation=label,processed=len(records),total=len(rows))),flush=True)
    rawfile=dataout/f'{label}_raw.pt';torch.save(dict(schema_version=1,raw_logits=raw),rawfile)
    value=dict(label=label,rows=records,n=len(records),exact_count=sum(x['exact'] for x in records),
        first_token_nll=sum(x['first_token_nll'] for x in records)/len(records),
        seconds=time.perf_counter()-begin,raw_file=str(rawfile),raw_sha256=sha(rawfile))
    save(out/f'{label}.json',value);return value

def active_cache_replay(torch,model,processor,branch,cache,states,index,manifest,arm,dataout):
    from scripts.cache_native_vision_v7_features import replay_metrics
    observations=[];raw=[]
    for n in (8,16):
        sample=sorted(manifest['splits'][f'train_N{n}']['samples'],key=lambda r:r['sid'])[0]
        local,g,targets=batch_states(torch,cache,states,index,[sample['sid']],arm)
        for j,prefix in enumerate(([],cache['scenes'][sample['sid']]['target_ids'][:1])):
            bundle=native.prepare_scene(processor,sample,arm,prefix_ids=prefix)
            actual=native.forward_native(model,bundle,branch,capture=True,cpu=False)
            with torch.inference_mode():
                cached=model.lm_head(model.model.language_model.norm((g[j:j+1]+branch(local[:,j:j+1],g[j:j+1])).unsqueeze(0)))[0]
                h=actual['global_states'];lh=actual['local_states']
                replay=model.lm_head(model.model.language_model.norm((h+branch(lh,h)).unsqueeze(0)))[0]
            logits=actual['global_logits'].unsqueeze(0)
            metrics=replay_metrics(torch,logits,replay)
            drift=replay_metrics(torch,logits,cached)
            observations.append(dict(sid=sample['sid'],n_frames=n,prefix_ids=prefix,position=j,
                native_head_replay=metrics,cached_vs_native_descriptive=drift,cached_vs_native_gate=False,
                global_hidden_relative_l2=float((h.float()-g[j:j+1].float()).norm()/h.float().norm().clamp_min(1e-12)),
                counters=actual['counters'],native_metadata=actual['metadata']))
            raw.append(dict(native=logits.detach().cpu(),replayed=replay.detach().cpu(),cached=cached.detach().cpu()))
    path=dataout/'active_cache_replay.pt';torch.save(raw,path)
    passed=all(r['native_head_replay']['passed'] for r in observations)
    return dict(passed=passed,observations=observations,raw_file=str(path),raw_sha256=sha(path),
        cached_numerical_failures=sum(not r['cached_vs_native_descriptive']['passed'] for r in observations))

def tests():
    import torch
    from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
    slots=[f's{i}' for i in range(1944)];a=presentation_order(slots,8);b=presentation_order(slots,8)
    need(a==b and a!=presentation_order(slots,9) and Counter(r['sid'] for r in a)==Counter({s:40 for s in slots}),
         'Matched order/epoch coverage failed')
    need(lr(1)==.00002 and lr(50)==.001 and math.isclose(lr(4860),1e-5),'Optimizer endpoints differ')
    branch=ParallelLocalAggregation(8,rank=3);torch.nn.init.normal_(branch.up.weight,std=.01)
    h=torch.randn(2,4,8);g=torch.randn(4,8)
    need(torch.allclose(branch(h,g),branch(torch.cat((h,torch.zeros_like(h))),g),atol=1e-7,rtol=1e-6),
         'Raw-zero local padding changed aggregation')
    return dict(passed=True,tests=['matched_complete_epoch_order','different_seeds','lr_endpoints','zero_padding'])

def check(args):
    import torch
    torch.set_num_threads(4);test=tests();bindings={};cachefiles={}
    for arm in POLICY['arms']:
        path=getattr(args,arm+'_cache');cache=cache_binding(path);cachefiles[arm]=str(path)
        bindings[arm]=dict(file=str(path),sha256=sha(path),plan_sha256=cache['plan_sha256'])
    manifest=read(DATA/'v7_balanced/main_manifest.json');schedule=read(DATA/'v7_balanced/schedule.json')
    train={r['sid']:r for key in ('train_N8','train_N16') for r in manifest['splits'][key]['samples']}
    for arm,path in cachefiles.items():
        cache=read(path);need(set(cache['scenes'])==set(train),'Cache training scene coverage differs')
        for sid,scene in cache['scenes'].items():
            need(all(scene[k]==train[sid][k] for k in ('gold','n_frames','question','content_sha256','qa_sha256'))
                 and len(scene['target_ids'])==2 and scene['target_ids'][1]==151645,'Cache scene target differs')
    need(len(schedule['epoch_slots'])==1944 and set(schedule['epoch_slots'])==set(train),'Balanced schedule changed')
    files=[DATA/'v7_balanced/main_manifest.json',DATA/'v7_balanced/schedule.json',
           DATA/'v7_fresh/main_manifest.json',DATA/'v7_fresh/count_manifest.json']
    for root in ('v7_balanced','v7_fresh'):
        m=read(DATA/root/'main_manifest.json');need(sha(m['audit_file'])==m['audit_sha256'],'Semantic audit binding changed')
        files.append(Path(m['audit_file']))
    software=read(args.native_profile/'summary.json');need(software['computational_integrity_passed'] and software['zero_identity_passed'],'Native integration gate failed')
    from transformers import AutoProcessor
    processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,use_fast=False)
    for cachefile in cachefiles.values():
        for scene in read(cachefile)['scenes'].values():
            need(scene['target_ids']==native.encode_target(processor.tokenizer,scene['gold']),'Cached native answer/EOS target differs')
    need(sha(software['plan_file'])==software['plan_sha256'],'Native CPU plan changed')
    runtimeplan=read(software['plan_file']);need(sha(runtimeplan['source_manifest']['path'])==runtimeplan['source_manifest']['sha256'],'Native source manifest changed')
    need(sha(software['plan_file'])==Path(software['plan_file']).with_suffix('.sha256').read_text().strip(),'Native CPU plan sidecar changed')
    for path,h in software['source_sha256'].items():need(sha(REPO/path)==h,'Native profile source changed')
    profile_manifest=read(runtimeplan['source_manifest']['path'])
    software_rows={r['sid']:r for split in profile_manifest['splits'].values() for r in split['samples']}
    preparation={}
    for case in runtimeplan['cases']:
        meta=case['metadata'];begin=time.perf_counter()
        bundle=native.prepare_scene(processor,software_rows[meta['sid']],meta['arm'])
        preparation[case['case_id']]=time.perf_counter()-begin
        need(bundle['metadata']['input_identity']==meta['input_identity'],'Timed input preparation differs from frozen runtime oracle')
    out=OUT/f'train_check_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False);snapshot(out)
    plan=dict(schema_version=1,policy=POLICY,source_sha256=sources(),cache_bindings=bindings,
        data_bindings={str(p):sha(p) for p in files},native_profile=str(args.native_profile),
        native_profile_sha256=sha(args.native_profile/'summary.json'),preparation_seconds=preparation,tests=test)
    save(out/'plan.json',plan);save(out/'summary.json',dict(passed=True,plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json')))
    (out/'INDEX.md').write_text('# V7 training CPU release\n\n[Plan](plan.json) · [Summary](summary.json)\n')
    print(json.dumps(dict(passed=True,plan_file=str(out/'plan.json'))),flush=True)

def release_mains(args):
    plan=read(args.plan);need(plan['source_sha256']==sources(),'Main release source differs')
    need(args.data_release is not None,'Independent data/report release required before mains')
    data_release=read(args.data_release);need(data_release['passed'],'Independent data release failed')
    report_sources={p:sha(REPO/p) for p in ('scripts/report_native_vision_v7.py','slurm/native_vision_v7_report.sbatch')}
    need(data_release['source_sha256']==report_sources,'Independent report source changed since data release')
    native_profile=read(Path(plan['native_profile'])/'summary.json')
    need(sha(Path(plan['native_profile'])/'summary.json')==plan['native_profile_sha256'],'Native profile changed')
    for value in native_profile['files'].values():need(sha(value['path'])==value['sha256'],'Native profile timing artifact changed')
    forwards=read(Path(plan['native_profile'])/'forwards.json')
    projections={};bindings={}
    for directory in args.release_profiles:
        profile=read(directory/'summary.json');arm=profile['arm']
        need(arm not in projections and profile['profile'] and profile['passed'] and profile['computational_integrity_passed']
            and profile['steps']==32 and profile['plan_sha256']==sha(args.plan)
            and profile['source_sha256']==sources(),'Unmatched training profile')
        timings={}
        for n in (16,64):
            case=f'{arm}_N{n}'
            generation=max(r['total_instrumented_seconds'] for r in native_profile['observations'] if r['case_id']==case)
            decode=max(r['native_forward_with_observer_seconds'] for r in forwards
                if r['context']['case_id']==case and r['context']['phase']=='generation' and r['visual']==0 and r['query_tokens']==1)
            timings[n]=generation+2*decode+plan['preparation_seconds'][case]
        train=4860*profile['step_seconds_max_steady'];dev=360*timings[16]
        test=108*timings[16]+344*timings[64]
        estimate=profile['model_and_features_load_seconds']+1.25*(train+dev+test)+120
        projections[arm]=dict(projected_seconds=estimate,passed=estimate<=1800,training_seconds=train,
            dev_seconds=dev,test_seconds=test,generation_seconds=timings)
        bindings[arm]=dict(directory=str(directory.resolve()),summary_sha256=sha(directory/'summary.json'))
    need(set(projections)==set(POLICY['arms']),'Both arm profiles required')
    out=OUT/f'main_release_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False);snapshot(out)
    result=dict(passed=all(r['passed'] for r in projections.values()),plan_file=str(args.plan.resolve()),
        plan_sha256=sha(args.plan),source_sha256=sources(),profiles=bindings,projections=projections,
        per_main_seconds_cap=1800,main_block_gpu_seconds_cap=9000,
        data_release=dict(file=str(args.data_release.resolve()),sha256=sha(args.data_release)),report_source_sha256=report_sources)
    save(out/'release.json',result);(out/'INDEX.md').write_text('# V7 main runtime release\n\n[Decision](release.json)\n')
    print(json.dumps(result,indent=2),flush=True)
    need(result['passed'],'Prospective main runtime cap failed; do not launch')

def run(args):
    import torch,transformers
    from gnnformer.runtime import load_runtime
    from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
    torch.set_num_threads(4);start=time.perf_counter();plan=read(args.plan)
    need(plan['policy']==POLICY and plan['source_sha256']==sources(),'Frozen main source/policy differs')
    for path,h in plan['data_bindings'].items():need(sha(path)==h,'Frozen dataset/audit differs')
    need(sha(Path(plan['native_profile'])/'summary.json')==plan['native_profile_sha256'],'Native software profile changed')
    arm,seed=args.arm,args.seed;need(arm in POLICY['arms'] and seed in POLICY['seeds'],'Unregistered arm/seed')
    if not args.profile:
        need(args.main_release is not None,'A passing measured main release is required')
        released=read(args.main_release);need(released['passed'] and released['plan_sha256']==sha(args.plan)
            and released['source_sha256']==sources(),'Main resource release differs')
        need(sha(released['data_release']['file'])==released['data_release']['sha256'],'Independent data release changed')
        for p,h in released['report_source_sha256'].items():need(sha(REPO/p)==h,'Frozen independent reporter changed')
        for value in released['profiles'].values():need(sha(Path(value['directory'])/'summary.json')==value['summary_sha256'],'Training profile changed')
    binding=plan['cache_bindings'][arm];need(sha(binding['file'])==binding['sha256'],'Training cache changed')
    cache=cache_binding(binding['file']);job=os.environ['SLURM_JOB_ID']
    runid=f'{"profile" if args.profile else "run"}_{arm}_s{seed}_{job}'
    out=OUT/runid;out.mkdir(parents=True,exist_ok=False);snapshot(out)
    dataout=DATA/'native_aggregation_vlm_v7'/runid;dataout.mkdir(parents=True,exist_ok=False)
    ckpt=CKPT/runid;ckpt.mkdir(parents=True,exist_ok=False)
    loadstart=time.perf_counter();runtime=load_runtime(str(MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda')
    model=runtime.model;model.eval();model.requires_grad_(False);native.native_contract(model)
    from scripts.stage_native_vision_v7_features import runtime_identity,model_metadata,native_api
    from scripts.probe_native_vision_v2_prefix import fingerprint
    need(runtime_identity()==cache['runtime'] and model_metadata()==cache['model']
         and fingerprint(runtime.processor,str(transformers.__version__))==cache['processor'],'Actual loaded model/runtime/processor differs')
    featureplan=read(cache['plan_file']);need(native_api(runtime.processor)[2]==featureplan['native_api'],'Native installed code differs')
    need(native.generation_policy(model,runtime.processor.tokenizer)[1]['native_eos_token_ids']==POLICY['native_eos'],'EOS contract changed')
    states,index=load_features(torch,cache,runtime.device);loadseconds=time.perf_counter()-loadstart
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);branch=ParallelLocalAggregation().to(runtime.device)
    initialized=state_info(branch);need(sum(p.numel() for p in branch.parameters())==1041600,'Parameter budget differs')
    initial={k:v.detach().cpu().clone() for k,v in branch.state_dict().items()}
    optimizer=torch.optim.AdamW(branch.parameters(),lr=.001,weight_decay=0.)
    schedule=read(DATA/'v7_balanced/schedule.json');order=presentation_order(schedule['epoch_slots'],seed)
    save(out/'presentations.json',order)
    config=dict(run_id=runid,arm=arm,seed=seed,profile=args.profile,policy=POLICY,plan_file=str(args.plan),
        plan_sha256=sha(args.plan),source_sha256=sources(),cache_binding=binding,initialized=initialized,
        initialized_sha256=object_sha(initialized),presentations_sha256=sha(out/'presentations.json'),
        model=cache['model'],runtime=cache['runtime'],processor=cache['processor'],native_dtypes=cache['native_dtypes'],
        hardware=dict(name=torch.cuda.get_device_name(0),cuda=torch.version.cuda,torch=str(torch.__version__)),
        model_and_features_load_seconds=loadseconds,checkpoint_directory=str(ckpt),data_directory=str(dataout),
        main_release=None if args.profile else dict(file=str(args.main_release.resolve()),sha256=sha(args.main_release)))
    save(out/'config.json',config);(out/'INDEX.md').write_text('# V7 native aggregation run\n\n[Config](config.json) · [Training](training.json) · [Summary](summary.json)\n')
    manifest=read(DATA/'v7_balanced/main_manifest.json');devrows=[('dev_N16',r) for r in manifest['splits']['dev_N16']['samples']]
    trainlog=[];devs=[];best=None;totalsteps=32 if args.profile else 4860
    norm=model.model.language_model.norm;head=model.lm_head
    for step in range(1,totalsteps+1):
        begin=time.perf_counter();rows=order[(step-1)*16:step*16];sids=[r['sid'] for r in rows]
        local,g,targets=batch_states(torch,cache,states,index,sids,arm)
        optimizer.zero_grad(set_to_none=True);rate=lr(step)
        for group in optimizer.param_groups:group['lr']=rate
        delta=branch(local,g,output_dtype=g.dtype);logits=head(norm((g+delta).unsqueeze(0)))[0]
        loss=torch.nn.functional.cross_entropy(logits.float(),targets);need(bool(torch.isfinite(loss)),'Nonfinite CE')
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(branch.parameters(),1.)
        need(bool(torch.isfinite(grad)) and all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in branch.parameters()),'Missing/nonfinite gradient')
        optimizer.step();need(all(bool(torch.isfinite(p).all()) for p in branch.parameters()),'Nonfinite updated branch')
        need(not any(p.requires_grad or p.grad is not None for p in model.parameters()),'Frozen backbone changed gradient state')
        torch.cuda.synchronize();trainlog.append(dict(step=step,lr=rate,loss=float(loss),gradient_norm=float(grad),
            clipped=float(grad)>1,seconds=time.perf_counter()-begin,target_ids=targets.detach().cpu().tolist(),sids=sids))
        if step%100==0:print(json.dumps(dict(step=step,loss=float(loss))),flush=True)
        if not args.profile and step in POLICY['dev_steps']:
            checkpoint=ckpt/f'step_{step}.pt';torch.save(dict(branch=branch.state_dict(),step=step,config=config),checkpoint)
            dev=evaluate(torch,model,runtime.processor,branch,devrows,arm,out,f'dev_{step}',dataout)
            entry=dict(step=step,exact_count=dev['exact_count'],nll=dev['first_token_nll'],
                checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),parameter_sha256=object_sha(state_info(branch)),
                dev_file=str(out/f'dev_{step}.json'),dev_sha256=sha(out/f'dev_{step}.json'))
            devs.append(entry)
            if best is None or (-entry['exact_count'],entry['nll'],step)<(-best['exact_count'],best['nll'],best['step']):best=entry
            save(out/'training.json',trainlog);save(out/'selection.json',dict(development=devs,selected=best))
    save(out/'training.json',trainlog)
    if args.profile:
        updated=state_info(branch);need(updated!=initialized,'Finite updates did not change parameters')
        path=ckpt/'profile.pt';torch.save(branch.state_dict(),path);branch.load_state_dict(initial)
        need(state_info(branch)==initialized,'Initial restoration failed')
        branch.load_state_dict(torch.load(path,map_location=runtime.device,weights_only=True));need(state_info(branch)==updated,'Updated restoration failed')
        replay=active_cache_replay(torch,model,runtime.processor,branch,cache,states,index,manifest,arm,dataout)
        save(out/'active_cache_replay.json',replay)
        need(replay['passed'],'Active captured-hidden native head replay failed')
        result=dict(config,passed=True,completed=True,active_cache_replay=replay,steps=32,training_seconds=sum(r['seconds'] for r in trainlog),
            step_seconds_max_steady=max(r['seconds'] for r in trainlog[4:]),computational_integrity_passed=True,
            no_dev_or_test_evaluation=True,profile_checkpoint=str(path),profile_checkpoint_sha256=sha(path))
    else:
        need(best is not None and len(devs)==5,'Missing registered selection')
        blob=torch.load(best['checkpoint'],map_location=runtime.device,weights_only=True);branch.load_state_dict(blob['branch'])
        need(object_sha(state_info(branch))==best['parameter_sha256'],'Selected checkpoint restoration changed')
        testrows=[]
        for purpose in ('main','count'):
            m=read(DATA/'v7_fresh'/f'{purpose}_manifest.json')
            for cell,selection in m['splits'].items():testrows.extend((purpose+'_'+cell,r) for r in selection['samples'])
        need(len(testrows)==452,'Incomplete native evaluation')
        final=evaluate(torch,model,runtime.processor,branch,testrows,arm,out,'test',dataout)
        result=dict(config,passed=True,completed=True,steps=4860,training_seconds=sum(r['seconds'] for r in trainlog),
            selected=best,development=devs,test_file=str(out/'test.json'),test_sha256=sha(out/'test.json'),
            native_test_count=452,computational_integrity_passed=True)
    result['elapsed_seconds']=time.perf_counter()-start
    need(sources()==plan['source_sha256'],'Main source changed during run');save(out/'summary.json',result)
    print(json.dumps(dict(run_id=runid,completed=True,elapsed_seconds=result['elapsed_seconds'])),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');p.add_argument('--profile',action='store_true')
    p.add_argument('--self-test',action='store_true');p.add_argument('--parallel-cache',type=Path);p.add_argument('--joint-cache',type=Path);p.add_argument('--native-profile',type=Path)
    p.add_argument('--data-release',type=Path);p.add_argument('--main-release',type=Path);p.add_argument('--release-profiles',nargs=2,type=Path);p.add_argument('--plan',type=Path);p.add_argument('--arm',choices=POLICY['arms']);p.add_argument('--seed',type=int,default=8)
    args=p.parse_args();native.require_slurm(gpu=not args.check)
    if args.self_test:
        need(args.check,'Self-test uses CPU check mode');print(json.dumps(tests()),flush=True)
    elif args.release_profiles:
        need(args.check,'Resource release is CPU only');release_mains(args)
    elif args.check:check(args)
    else:run(args)
if __name__=='__main__':main()

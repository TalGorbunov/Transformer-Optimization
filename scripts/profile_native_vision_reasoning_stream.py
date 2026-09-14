"""Fixed Cosmos streaming software smoke: six natural max8 trajectories.

Native/zero/random-active on two old N16/N64 scenes; replay EVERY active prefix.
At most64 model/22 visual calls, one GPU3min. No fitting or answer scoring.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import native_vision_reasoning_stream as stream
from scripts import profile_native_vision_v7_runtime as old
from scripts import probe_native_vision_parallel_local_mixed as mixed
from scripts.probe_native_vision_reasoning import metadata,fingerprint,MODEL_PATHS
from scripts.stage_native_vision_v7_features import need,read,sha
base=stream.base
MODEL=Path(MODEL_PATHS['cosmos'])
OUT=REPO/'outputs/native_aggregation_vlm/reasoning_stream'
DATA=Path('/mnt/data/gabriele/gnn_transformer/reasoning_stream')
CKPT=Path('/mnt/ckpts/gabriele/gnn_transformer/reasoning_stream')
SEED=20261001
OWN=('scripts/native_vision_reasoning_stream.py','scripts/profile_native_vision_reasoning_stream.py',
     'tests/test_native_vision_reasoning_stream.py',
     'slurm/native_vision_reasoning_stream_selftest.sbatch','slurm/native_vision_reasoning_stream_check.sbatch',
     'slurm/native_vision_reasoning_stream_profile.sbatch','scripts/native_vision_v7_runtime.py',
     'scripts/profile_native_vision_v7_runtime.py','scripts/probe_native_vision_parallel_local_mixed.py',
     'scripts/probe_native_vision_parallel_local.py','scripts/probe_native_vision_mixed_cache_localization.py',
     'scripts/probe_native_vision_reasoning.py','scripts/probe_native_vision_v2_prefix.py',
     'scripts/stage_cosmos_reason1_compat.py','scripts/stage_native_vision_v7_features.py',
     'scripts/stage_native_vision_v6_teacher.py','scripts/probe_native_vision_v5_local_readability.py',
     'gnnformer/runtime.py','gnnformer/parallel_local_native.py','gnnformer/parallel_local_aggregation.py')


def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def sources():return {name:sha(REPO/name) for name in OWN}


def snapshot(out):
    (out/'source').mkdir();frozen=sources()
    for name,digest in frozen.items():
        target=out/'source'/name.replace('/','_');target.write_bytes((REPO/name).read_bytes())
        need(sha(target)==digest,'Source changed while copying')
    save(out/'source_hashes.json',frozen)
    (out/'INDEX.md').write_text('# Cosmos streaming software\n\n[Summary](summary.json) · [Sources](source_hashes.json)\n')
    return frozen


def versions():
    import torch,transformers
    return dict(torch_version=str(torch.__version__),transformers_version=str(transformers.__version__),
                bitsandbytes_version=importlib.metadata.version('bitsandbytes'))


def mirror_identity():
    report_path=MODEL/'COMPATIBILITY_PASSED.json';report=read(report_path)
    need(report['status']=='passed' and report['mirror']==str(MODEL) and report['source_mutated'] is False
         and report['downloaded'] is False,'Cosmos compatibility ancestry differs')
    for group in ('metadata','derived'):
        for name,item in report[group].items():
            need(sha(MODEL/name)==item['sha256'],'Mirror metadata/derived tokenizer changed')
            if 'source' in item:need(sha(item['source'])==item['sha256'],'Original Cosmos metadata changed')
    model=metadata(str(MODEL))
    for shard in report['shards']:
        actual=model['weight_shard_stat'][shard['name']]
        need(actual['resolved_path']==shard['target'] and actual['size']==shard['bytes'],
             'Cosmos weight shard ancestry differs')
    return dict(model=model,compatibility_file=str(report_path),compatibility_sha256=sha(report_path),
                weight_validation='Original compatible shard resolved paths/stat; no new full weight rehash')


def native_api(processor):
    from transformers import AutoConfig
    from transformers.generation.utils import GenerationMixin
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
    config=AutoConfig.from_pretrained(str(MODEL),local_files_only=True,trust_remote_code=True)
    need(config.model_type=='qwen2_5_vl' and config.text_config.hidden_size==3584,'Cosmos native class/width differs')
    classes=(Qwen2_5_VLModel,GenerationMixin,DynamicCache,type(processor),type(processor.image_processor))
    paths=sorted({str(Path(inspect.getfile(c)).resolve()) for c in classes})
    fn=Qwen2_5_VLModel.get_rope_index
    return SimpleNamespace(config=config),fn,dict(config_sha256=sha(MODEL/'config.json'),
        source_sha256={p:sha(p) for p in paths},rope_method=fn.__qualname__)


def tests():
    result=subprocess.run([sys.executable,'tests/test_native_vision_reasoning_stream.py'],cwd=REPO,
                          capture_output=True,text=True)
    need(result.returncode==0,'Streaming tests failed:\n'+result.stdout+result.stderr)
    return dict(passed=True,stdout=result.stdout,stderr=result.stderr)


def check(selftest=False):
    import torch,transformers
    from transformers import AutoProcessor,GenerationConfig
    from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
    torch.set_num_threads(4);started=time.perf_counter()
    label='self_test' if selftest else 'check';job=os.environ['SLURM_JOB_ID'];out=OUT/f'{label}_{job}'
    out.mkdir(parents=True,exist_ok=False);frozen=snapshot(out);unit=tests()
    if selftest:
        save(out/'summary.json',dict(passed=True,source_sha256=frozen,tests=unit,seconds=time.perf_counter()-started));return
    identity=mirror_identity();processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,use_fast=False,local_files_only=True)
    owner,fn,api=native_api(processor)
    def rope(**kwargs):return fn(owner,**kwargs)
    need(stream.SYSTEM in (MODEL/'README.md').read_text().replace('\\n','\n'),'Reasoning system differs from pinned official example')
    scenes,manifest=mixed.selected_sources();bundles={};cases=[]
    for sample in scenes:
        # Deliberately exclude gold/semantic fields from the deployed preparer.
        record={key:sample[key] for key in ('sid','n_frames','question','image_files')}
        bundle=stream.prepare_scene(processor,record);layout=base.audit_layout(rope,bundle)
        key=f'N{sample["n_frames"]}';bundles[key]=bundle
        need(bundle['metadata']['row_prompt_tokens'][-1]<bundle['metadata']['prompt_width'],'Global left padding was not exercised')
        need(bundle['metadata']['prompt_width']+8<=18000,'Software context ceiling exceeded')
        cases.append(dict(case_id=key,sid=sample['sid'],n_frames=sample['n_frames'],metadata=bundle['metadata'],layout=layout['metadata']))
    need([r['n_frames'] for r in cases]==[16,64],'Fixed software scene order differs')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED);branch=ParallelLocalAggregation()
        zero={k:v.detach().clone() for k,v in branch.state_dict().items()}
        with torch.no_grad():branch.up.weight.normal_(0,.001)
        active={k:v.detach().clone() for k,v in branch.state_dict().items()}
    data=DATA/f'check_{job}';data.mkdir(parents=True,exist_ok=False)
    weights=CKPT/f'check_{job}';weights.mkdir(parents=True,exist_ok=False)
    prepared=data/'prepared.pt';initial=weights/'initial.pt'
    torch.save(dict(schema_version=1,bundles=bundles),prepared);torch.save(dict(zero=zero,active=active),initial)
    policy_model=SimpleNamespace(generation_config=GenerationConfig.from_pretrained(str(MODEL)))
    config,policy=stream.generation_policy(policy_model,processor.tokenizer,8)
    plan=dict(schema_version=1,protocol='cosmos_streaming_smoke',seed=SEED,source_sha256=frozen,mirror=identity,
        runtime=versions(),processor=fingerprint(processor,str(transformers.__version__)),native_api=api,
        source_manifest=manifest,cases=cases,generation_policy=policy,generation_config=config.to_dict(),
        prepared_file=str(prepared),prepared_sha256=sha(prepared),initial_file=str(initial),initial_sha256=sha(initial),
        initial_state_identity={label:{k:base.tensor_info(v) for k,v in state.items()} for label,state in (('zero',zero),('active',active))},
        maximum_calls=dict(model=64,visual=22),tests=unit,no_training=True,no_efficacy_scoring=True,
        natural_prefixes='Replay every actual active prefix including special reasoning tokens, excluding terminal EOS',
        numerical_policy='Cache/full global TV<=.02/top1 reported descriptively; same-captured-h native-head replay is a computational check',
        runtime_scope='One batched N+1-row invocation/token, not one-row compute; full-length reasoning not tested')
    path=out/'plan.json';save(path,plan);path.with_suffix('.sha256').write_text(sha(path)+'\n')
    need(sources()==frozen,'Sources changed during CPU preparation')
    save(out/'summary.json',dict(passed=True,source_sha256=frozen,plan_file=str(path),plan_sha256=sha(path),seconds=time.perf_counter()-started))
    for directory in (data,weights):(directory/'INDEX.md').write_text('# Cosmos software inputs\n\nBound by '+str(path)+'\n')
    print(json.dumps(dict(passed=True,plan_file=str(path),plan_sha256=sha(path))),flush=True)


def verify(path):
    path=Path(path).resolve();plan=read(path)
    need(sha(path)==path.with_suffix('.sha256').read_text().strip() and plan['schema_version']==1
         and plan['protocol']=='cosmos_streaming_smoke' and plan['seed']==SEED and plan['source_sha256']==sources(),
         'CPU plan/source freeze differs')
    need(plan['mirror']==mirror_identity() and plan['runtime']==versions(),'Model/runtime ancestry changed')
    for name in ('prepared','initial'):need(sha(plan[name+'_file'])==plan[name+'_sha256'],'Frozen input/branch changed')
    need(sha(plan['source_manifest']['path'])==plan['source_manifest']['sha256'],'Source scene manifest changed')
    for name,digest in plan['source_sha256'].items():need(sha(path.parent/'source'/name.replace('/','_'))==digest,'CPU source copy changed')
    for name,digest in plan['native_api']['source_sha256'].items():need(sha(name)==digest,'Installed native source changed')
    summary=read(path.parent/'summary.json')
    need(summary['passed'] is True and summary['plan_sha256']==sha(path),'Completed CPU input gate required')
    return plan


class GlobalNativeAudit(old.NativeAudit):
    """Reuse actual mask/KV hooks; save only the global vocabulary vector."""
    def after(self,module,args,output):
        import torch
        from scripts.probe_native_vision_mixed_cache_localization import mask_check
        torch.cuda.synchronize();value=self.capture;current=self.counts
        need(all(current[k]-value['before_counts'][k]==1 for k in ('language','norm','attention'))
             and current['visual']-value['before_counts']['visual']==int(value['visual_expected']),'Native submodule calls differ')
        masks=mask_check(torch,value['causal_mask'],value['attention_mask'],value['cache_position'])
        logits=output.logits[-1:,-1,:].detach().cpu().clone();hidden=value['pre_final_rms_hidden']
        need(hidden.dtype==logits.dtype==self.model.model.language_model.norm.weight.dtype
             and bool(torch.isfinite(hidden).all()) and bool(torch.isfinite(logits).all()),'Native dtype/finiteness differs')
        cache=output.past_key_values;retained=None;cache_meta=None
        if value['use_cache']:
            length=value['attention_mask'].shape[1]
            _,cache_meta=mixed.cache_snapshot(cache,hidden.shape[0],length,len(self.layers))
            if self.previous is not None:retained=mixed.prefix_preserved(torch,cache,self.previous,value['old_length'])
        else:need(cache is None and self.previous is None,'Uncached reference returned/used a KV cache')
        self.previous=None
        state=dict(schema_version=1,context=dict(self.context),native_logits=logits,
            **{k:value[k] for k in ('input_ids','attention_mask','position_ids','causal_mask','cache_position','pre_final_rms_hidden')})
        path=self.data/f'forward_{len(self.records):03d}.pt';torch.save(state,path)
        self.records.append(dict(context=dict(self.context),forward_index=len(self.records),path=str(path),sha256=sha(path),
            batch_size=hidden.shape[0],query_tokens=value['input_ids'].shape[1],key_tokens=value['attention_mask'].shape[1],
            visual=int(value['visual_expected']),mask_audit=masks,cache_layers=cache_meta,previous_prefix_exact_by_layer=retained,
            expected_visual_identity=value['expected_visual_identity'],global_logits=base.tensor_info(logits),
            native_hidden=base.tensor_info(hidden),native_forward_with_observer_seconds=time.perf_counter()-value['started']))


def replay_head(torch,model,branch,state):
    h=state['pre_final_rms_hidden'].to(model.device);g=h[-1:];local=h[:-1].unsqueeze(1)
    with torch.inference_mode():
        delta=branch(local,g,output_dtype=g.dtype)
        # .forward bypasses the temporary native norm fusion hook intentionally.
        logits=model.lm_head(model.model.language_model.norm.forward((g+delta).unsqueeze(0)))[0,0].float().cpu()
    result=old.metric(torch,logits,state['native_logits'][0])
    need(result['numerical_rule_passed'],'Same captured-state native head reconstruction failed')
    return result


def run(path):
    import torch,transformers
    from gnnformer.runtime import load_runtime,get_rope_index_fn
    from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
    torch.set_num_threads(4);started=time.perf_counter();plan=verify(path);job=os.environ['SLURM_JOB_ID']
    out=OUT/f'profile_{job}';out.mkdir(parents=True,exist_ok=False);frozen=snapshot(out)
    data=DATA/f'profile_{job}';data.mkdir(parents=True,exist_ok=False)
    (out/'plan.json').write_bytes(Path(path).read_bytes())
    prepared=torch.load(plan['prepared_file'],map_location='cpu',weights_only=True)
    weights=torch.load(plan['initial_file'],map_location='cpu',weights_only=True)
    need({label:{k:base.tensor_info(v) for k,v in state.items()} for label,state in weights.items()}==plan['initial_state_identity'],
         'Frozen random branch identities differ')
    tick=time.perf_counter();loaded=load_runtime(str(MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda')
    model=loaded.model;model.eval();model.requires_grad_(False);norm=stream.contract(model)
    need(fingerprint(loaded.processor,str(transformers.__version__))==plan['processor']
         and native_api(loaded.processor)[2]==plan['native_api'],'Loaded processor/native API differs')
    config,policy=stream.generation_policy(model,loaded.tokenizer,8)
    need(config.to_dict()==plan['generation_config'] and policy==plan['generation_policy'],'Generation config differs')
    torch.cuda.synchronize();load_seconds=time.perf_counter()-tick
    branch=ParallelLocalAggregation().to(model.device);observations=[];comparisons=[];head_replays=[];artifacts=[]
    with GlobalNativeAudit(model,data) as audit:
        for case in plan['cases']:
            key=case['case_id'];bundle=prepared['bundles'][key]
            need(bundle['metadata']==case['metadata'] and base.audit_layout(get_rope_index_fn(model),bundle)['metadata']==case['layout'],
                 'Prepared software input/layout differs')
            baseline=None
            for condition in ('native','zero','active'):
                audit.context=dict(case_id=key,condition=condition,phase='generation');offset=len(audit.records)
                if condition!='native':branch.load_state_dict(weights[condition],strict=True)
                hook_modules=(model,model.model.visual,model.model.language_model,norm)
                before_hooks=[(len(m._forward_pre_hooks),len(m._forward_hooks)) for m in hook_modules]
                result=stream.generate_native(model,loaded.processor,None if condition=='native' else branch,bundle,
                                               max_new_tokens=8,full_vectors=True)
                need(before_hooks==[(len(m._forward_pre_hooks),len(m._forward_hooks)) for m in hook_modules],
                     'Generation observer/fusion hooks leaked after their context')
                records=audit.records[offset:];old.audit_sequence(torch,model,bundle,records,result['generated_ids'])
                if condition=='native':baseline=result
                if condition=='zero':need(result['generated_ids']==baseline['generated_ids']
                    and torch.equal(result['raw_logits'],baseline['raw_logits']),'Zero-U native identity failed')
                for i,record in enumerate(records):
                    saved=torch.load(record['path'],map_location='cpu',weights_only=True)
                    need(torch.equal(saved['native_logits'][0].float(),result['raw_logits'][i]),'Streamed global logits differ from model output')
                artifact=data/f'{key}_{condition}.pt';torch.save(result,artifact)
                artifacts.append(dict(path=str(artifact),sha256=sha(artifact)))
                observations.append(dict(case_id=key,n_frames=case['n_frames'],condition=condition,
                    generated_ids=result['generated_ids'],completed=result['completed'],truncated=result['truncated'],
                    tokens=len(result['generated_ids']),counters=result['counters'],model_seconds=result['model_seconds'],
                    artifact_file=str(artifact),artifact_sha256=sha(artifact),natural_generation=True))
                if condition!='active':continue
                for step in range(len(result['generated_ids'])):
                    prefix=result['generated_ids'][:step];audit.context=dict(case_id=key,condition=condition,phase='full_prefix',step=step)
                    prefixed=stream.prefixed_bundle(bundle,prefix)
                    reference=stream.forward_native(model,prefixed,branch)
                    full=torch.load(audit.records[-1]['path'],map_location='cpu',weights_only=True)
                    cached=torch.load(records[step]['path'],map_location='cpu',weights_only=True)
                    expected,_=get_rope_index_fn(model)(input_ids=prefixed['inputs']['input_ids'],
                        image_grid_thw=prefixed['inputs']['image_grid_thw'],attention_mask=prefixed['inputs']['attention_mask'])
                    need(torch.equal(full['position_ids'],expected) and torch.equal(full['input_ids'],prefixed['inputs']['input_ids']),
                         'Uncached natural prefix inputs/mRoPE differ')
                    comparisons.append(dict(case_id=key,step=step,prefix_ids=prefix,
                        **old.metric(torch,result['raw_logits'][step],reference['global_logits']),descriptive_only=True))
                    for phase,state in (('cached',cached),('full',full)):
                        head_replays.append(dict(case_id=key,step=step,phase=phase,**replay_head(torch,model,branch,state)))
            print(json.dumps(dict(case=key,completed=True,model_calls=audit.counts['model'])),flush=True)
        counts=dict(audit.counts);records=list(audit.records)
    generated=sum(r['tokens'] for r in observations);replays=len(comparisons)
    need(len(observations)==6 and replays==sum(r['tokens'] for r in observations if r['condition']=='active')
         and counts==dict(model=generated+replays,visual=6+replays,language=generated+replays,norm=generated+replays,attention=generated+replays)
         and counts['model']<=64 and counts['visual']<=22,'Software call/prefix coverage differs')
    need(not model.training and not any(p.requires_grad or p.grad is not None for p in model.parameters()),
         'Software observer changed native training/gradient state')
    need(verify(path)==plan and sources()==frozen,'Frozen sources/inputs changed during GPU smoke')
    for name,value in (('forwards.json',records),('observations.json',observations),('comparisons.json',comparisons),
                       ('head_replays.json',head_replays),('artifacts.json',artifacts)):save(out/name,value)
    failures=[r for r in comparisons if not r['numerical_rule_passed']]
    summary=dict(schema_version=1,completed=True,computational_integrity_passed=True,zero_identity_passed=True,
        captured_state_head_replay_passed=True,strict_cache_numerical_passed=not failures,cache_numerical_failures=failures,
        plan_file=str(Path(path).resolve()),plan_sha256=sha(path),source_sha256=frozen,mirror=plan['mirror'],
        runtime=plan['runtime'],processor=plan['processor'],actual_native_dtypes=dict(norm=str(norm.weight.dtype),lm_head=str(model.lm_head.weight.dtype)),
        no_training=True,no_efficacy_scoring=True,reasoning_composition_established=False,
        calls=counts,generated_tokens=generated,active_prefixes=replays,captured_head_replays=len(head_replays),
        max_actual_generated_tokens=max(r['tokens'] for r in observations),max_registered_tokens=8,
        model_load_seconds=load_seconds,total_seconds=time.perf_counter()-started,gpu=torch.cuda.get_device_name(0),
        files={name:dict(path=str(out/name),sha256=sha(out/name)) for name in ('forwards.json','observations.json','comparisons.json','head_replays.json','artifacts.json')},
        limitations=['Short natural traces may stop early; report actual prefix coverage.',
            'Fixed random branch and old scenes; no trained-method, answer-accuracy or reasoning-composition result.',
            'One invocation/token contains N+1 independent decoder rows; this is not one-row compute.',
            'Observed full native KV prefixes are compared exactly on GPU; full KV tensors are not saved.',
            'Audit serialization and CPU copies affect measured timing; long-run throughput is untested.'])
    save(out/'summary.json',summary)
    (out/'REPORT.md').write_text('# Cosmos streaming software smoke\n\nComputational and exact zero-U checks passed. '
        f'Cache/full numerical checks passed: {not failures}. All failures retained. '
        f'{generated} natural tokens, {replays} active prefixes; {counts["model"]} model and {counts["visual"]} visual calls. '
        'No training, efficacy scoring or reasoning-composition claim.\n')
    (data/'INDEX.md').write_text('# Frozen software observations\n\nBound by '+str(out/'summary.json')+'\n')
    print(json.dumps(dict(completed=True,summary_file=str(out/'summary.json'),calls=counts)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-test',action='store_true');mode.add_argument('--check',action='store_true');mode.add_argument('--run',action='store_true')
    parser.add_argument('--plan',type=Path);args=parser.parse_args()
    base.require_slurm(gpu=args.run)
    if args.run:
        need(args.plan is not None,'Frozen CPU plan required');run(args.plan)
    else:
        need(os.environ['SLURM_JOB_PARTITION']=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),'CPU allocation required')
        check(args.self_test)


if __name__=='__main__':main()

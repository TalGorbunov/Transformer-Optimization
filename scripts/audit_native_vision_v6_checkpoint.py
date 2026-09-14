"""Frozen-checkpoint V6 state/numerical audit and ten software evaluations.

This does not waive the failed strict gate. Computational integrity and strict
numerical status are separate. No training, backend changes or main-test scoring.
All imports involving models/images and all numerical work require Slurm.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
DATA=Path('/mnt/data/gabriele/gnn_transformer')
BASE=REPO/'outputs/native_aggregation_vlm/v6'
OUT=BASE/'checkpoint_audit'
PRE=None
OWN=('scripts/audit_native_vision_v6_checkpoint.py',
     'slurm/native_aggregation_vision_v6_audit_check.sbatch',
     'slurm/native_aggregation_vision_v6_audit.sbatch',
     'scripts/diagnose_native_vision_v5_cache.py')
BACKEND='native_bitsandbytes_dispatch_unmodified'


def ensure(ok,message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def sources():
    return {name:sha(REPO/name) for name in OWN}


def installed_backend_sources():
    root=Path(importlib.util.find_spec('bitsandbytes').origin).parent
    return {str(root/name):sha(root/name) for name in
        ('autograd/_functions.py','nn/modules.py','functional.py','__init__.py')}


def make_plan(torch,phase,directories):
    from scripts.diagnose_native_vision_v5_cache import selected_epoch
    from scripts.native_aggregation_vlm_v4 import sample_metadata
    from scripts.profile_native_vision_v5_checks import logit_parity_metrics
    x=torch.tensor([1.,2.,3.,4.,5.])
    metric=logit_parity_metrics(x+8,x)
    ensure(metric['centered_rms_logit_difference']==0 and metric['total_variation']==0,'Shift-invariant metric self-check failed')
    directories=[p.resolve() for p in directories]
    ensure((phase=='post' and len(directories)==4 and len(set(directories))==4),'Wrong fixed run list')
    runs=[]; frozen={}
    for directory in directories:
        config=json.loads((directory/'config.json').read_text())
        history=json.loads((directory/'training.json').read_text())
        ensure(directory.name==config['run_id'] and config['profile']==(phase=='pre'),'Run identity/phase differs')
        ensure(len(history)==(2 if phase=='pre' else 9),'Incomplete training history')
        if phase=='post':
            ensure(directory.parent==BASE/'main'/config['condition']/f'seed{config["seed"]}',
                   'Post audit requires the canonical V6 main condition/seed directory')
        for name,digest in config['code_sha256'].items():
            ensure(sha(REPO/name)==digest,f'Frozen V6 source changed: {name}')
            frozen[name]=digest
        checkpoint=Path(config['checkpoint_root'])/'native_aggregation_vlm_v6'/config['run_id']/'best.pt'
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        epoch=selected_epoch(history)
        ensure(saved['architecture']==config['architecture'] and saved['epoch']==epoch
               and saved['dev']==history[epoch-1]['dev'] and saved['step']==history[epoch-1]['step'],
               'Checkpoint differs from exact dev-selected best')
        ensure(float(saved['branch']['up.weight'].float().norm())>0,'Expected trained nonzero U')
        ensure(config['architecture']['protocol']=='vision_v6_local_teacher' and config['architecture']['operator']['merge']=='sum'
               and config['architecture']['local_distillation']['deployed'] is False,'Unexpected V6 deployment architecture')
        if phase=='post':
            completed=json.loads((directory/'summary.json').read_text())
            ensure(Path(completed['selected_checkpoint']).resolve()==checkpoint.resolve(),'Completed run selected a different checkpoint')
        runs.append(dict(run_id=config['run_id'],condition=config['condition'],seed=config['seed'],
            directory=str(directory),config=config,config_path=str(directory/'config.json'),
            config_sha256=sha(directory/'config.json'),training_path=str(directory/'training.json'),
            training_sha256=sha(directory/'training.json'),checkpoint=str(checkpoint),
            checkpoint_sha256=sha(checkpoint),selected_epoch=epoch,source_code_sha256=config['code_sha256']))
        del saved
    wanted={(arm,seed) for arm in ('control','aligned') for seed in (6,7)}
    ensure({(r['condition'],r['seed']) for r in runs}==wanted,'Wrong conditions/seeds')
    runs.sort(key=lambda r:(r['seed'],r['condition']))
    manifests={}; records=[]
    for family,name,ns in (('length','profile_manifest.json',(16,32,64)),
                           ('unseen_count','profile_count_manifest.json',(32,64))):
        path=DATA/'v4_diversity'/name
        manifest=json.loads(path.read_text()); manifests[family]=dict(path=str(path),sha256=sha(path))
        for n in ns:
            rows=manifest['splits'][f'test_N{n}']['samples']
            ensure(len(rows)==2,'Expected exactly two software samples per cell')
            for row in rows:
                row=dict(row); parsed=sample_metadata(Path(row['path']),n)
                ensure(all(row[k]==parsed[k] for k in ('sid','gold','qa_sha256')),'QA metadata differs')
                row.update(question=parsed['question'],cell=f'{family}_N{n}')
                ensure(sha(Path(row['path'])/'qa.txt')==row['qa_sha256'],'QA hash differs')
                for image in row['image_files']:
                    ensure(sha(image['path'])==image['sha256'],'Image hash differs')
                records.append(row)
    ensure(len(records)==10 and len({r['sid'] for r in records})==10,'Software examples duplicate')
    cells={cell:[r for r in records if r['cell']==cell] for cell in {r['cell'] for r in records}}
    cases=[cells[f'length_N{n}'][index]['sid'] for n,index in ((16,0),(64,0),(16,1),(64,1),(16,0))]
    return dict(schema_version=1,phase=phase,source_sha256=sources(),frozen_profile_source_sha256=frozen,
        installed_backend_source_sha256=installed_backend_sources(),bitsandbytes_version=importlib.metadata.version('bitsandbytes'),
        backend=BACKEND,runs=runs,manifests=manifests,records=records,cache_case_sids=cases,
        scope='Fixed-checkpoint computational/numerical audit only. Numeric failures retained; ten software predictions are not main efficacy results.',
        selection='Exact development-selected best checkpoints; no checkpoint choice from audit or main-test outcomes',
        numerical_policy='Record the frozen centered/probability gate without aborting solely for its failure; no threshold changes',
        limitation='Five two-token paths and small software set do not prove long-reasoning or universal cache parity')


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def cache_audit(torch,model,branch,prepare,records,tensor_root):
    from transformers import LogitsProcessor,LogitsProcessorList
    from scripts.diagnose_native_vision_v5_cache import Trace,compare_traces,tensor_difference,cpu_tree
    from scripts.profile_native_vision_v5_checks import logit_parity_metrics
    class ForceFirst(LogitsProcessor):
        def __init__(self,length,token):self.length,self.token=length,token
        def __call__(self,ids,scores):
            if ids.shape[1]==self.length:
                out=torch.full_like(scores,-torch.inf);out[:,self.token]=0;return out
            return scores
    observations=[]; first=None; calls=[0]
    handle=model.model.visual.register_forward_hook(lambda module,args,output:calls.__setitem__(0,calls[0]+1))
    attn=model.model.language_model.layers[14].self_attn
    try:
        for index,record in enumerate(records):
            inputs=prepare(record); branch.mode='all'
            trace=Trace(torch,model,branch,attn)
            try:
                before=calls[0]
                generated=model.generate(**inputs,do_sample=False,use_cache=True,min_new_tokens=2,max_new_tokens=2,
                    repetition_penalty=1.,return_dict_in_generate=True,output_logits=True)
                ensure(calls[0]-before==1 and branch.memory_image_count==record['n_frames'],'Cached visual/memory count differs')
                ids=generated.sequences[0,inputs['input_ids'].shape[1]:]
                ensure(len(ids)==2 and len(generated.logits)==2,'Expected two generation outputs')
                first_logits=generated.logits[0][0].detach().clone();cached=generated.logits[1][0].detach().clone()
                cached_trace=trace.last()
                if first is None:first=(ids.detach().clone(),first_logits.clone(),cached.clone())
                if index==4:
                    ensure(torch.equal(ids,first[0]) and torch.equal(first_logits,first[1]) and torch.equal(cached,first[2]),'A/B/C/D/A exact reset failed')
                full=dict(inputs);full['input_ids']=torch.cat((inputs['input_ids'],ids[:1][None]),dim=1)
                for key,value in (('attention_mask',1),('token_type_ids',0)):
                    if key in full:full[key]=torch.cat((full[key],torch.full_like(ids[:1][None],value)),dim=1)
                before=calls[0]
                uncached=model(**full,use_cache=False,logits_to_keep=1).logits[0,-1].detach().float().clone()
                ensure(calls[0]-before==1 and branch.memory_image_count==0,'Full visual/memory count differs')
                full_trace=trace.last()
                branch.mode='off';before=calls[0]
                disabled=model.generate(**inputs,do_sample=False,use_cache=True,min_new_tokens=2,max_new_tokens=2,
                    repetition_penalty=1.,logits_processor=LogitsProcessorList([ForceFirst(inputs['input_ids'].shape[1],int(ids[0]))]),
                    return_dict_in_generate=True,output_logits=True)
                ensure(calls[0]-before==1 and branch.memory_image_count==record['n_frames'],'OFF cached visual/memory count differs')
                native_ids=disabled.sequences[0,inputs['input_ids'].shape[1]:]
                ensure(len(native_ids)==2 and len(disabled.logits)==2 and int(native_ids[0])==int(ids[0]),'OFF prefix differs')
                native_cached=disabled.logits[1][0].detach().clone();native_cached_trace=trace.last();before=calls[0]
                native_full=model(**full,use_cache=False,logits_to_keep=1).logits[0,-1].detach().float().clone()
                ensure(calls[0]-before==1 and branch.memory_image_count==0,'OFF full visual/memory count differs')
                native_full_trace=trace.last()
            finally:
                trace.remove()
            branch.mode='all'
            enabled_trace=compare_traces(torch,cached_trace,full_trace)
            disabled_trace=compare_traces(torch,native_cached_trace,native_full_trace)
            for checked in (enabled_trace,disabled_trace):
                ensure(checked['all_raw_memory_exact'] and checked['image_ends_exact']
                    and checked['complete_visible_images']==record['n_frames']
                    and all(x['exact'] for x in checked['rotary_cos_sin_last']), 'Exact memory/rotary/visibility state check failed')
            replays={}
            for name,point in (('cached',cached_trace),('full',full_trace)):
                delta=branch(point['hidden'][None,None],point['raw_memory'],point['image_ends'],
                    point['query_positions'][-1:],point['language_mask'][-1:])[0,-1].detach().clone()
                replays[name]=tensor_difference(torch,point['delta'],delta)
            ensure(replays['cached']['exact'],'Fixed-input cached branch replay differs')
            enabled=logit_parity_metrics(cached,uncached);native=logit_parity_metrics(native_cached,native_full)
            rel_max=1.5*native['centered_maximum_absolute_logit_difference']+.0625
            rel_rms=1.5*native['centered_rms_logit_difference']+.005
            max_limit=min(.25,rel_max);rms_limit=min(.05,rel_rms)
            numeric=(enabled['centered_maximum_absolute_logit_difference']<=max_limit
                and enabled['centered_rms_logit_difference']<=rms_limit
                and enabled['total_variation']<=.01 and enabled['raw_top1_equal'])
            state=dict(enabled_trace=enabled_trace,disabled_trace=disabled_trace,
                cached_branch_replay=replays['cached'],full_branch_replay=replays['full'],cached_replay_exact=True)
            trace_path=tensor_root/f'cache_case_{index}.pt'
            compact=lambda value:{k:v for k,v in value.items() if k!='raw_memory'}
            torch.save(cpu_tree(torch,dict(cached_logits=cached,full_logits=uncached,
                native_cached_logits=native_cached,native_full_logits=native_full,
                cached_trace=compact(cached_trace),full_trace=compact(full_trace),
                native_cached_trace=compact(native_cached_trace),native_full_trace=compact(native_full_trace))),trace_path)
            row=dict(sid=record['sid'],n_frames=record['n_frames'],observation_index=index,
                case_role='prospective_unused_record' if index in (2,3) else 'original_case_or_reset',
                reset_repeat_of=0 if index==4 else None,passed=bool(numeric),
                **enabled,**{'native_reference_'+k:v for k,v in native.items()},
                centered_maximum_absolute_tolerance=max_limit,centered_rms_tolerance=rms_limit,
                relative_centered_maximum_absolute_tolerance=rel_max,relative_centered_rms_tolerance=rel_rms,
                total_variation_tolerance=.01,native_reference_same_first_token=int(native_ids[0]),
                generated_ids=ids.cpu().tolist(),first_logits_sha256=tensor_sha(first_logits),cached_logits_sha256=tensor_sha(cached),
                native_visual_calls_cached_generation=1,native_visual_calls_uncached_comparison=1,
                native_visual_calls_disabled_cached=1,native_visual_calls_disabled_full=1,uncached_memory_cleared=True,
                state=state,tensors_path=str(trace_path),tensors_sha256=sha(trace_path))
            observations.append(row)
            print(json.dumps(dict(cache_audit={k:row[k] for k in ('sid','observation_index','passed','total_variation','raw_top1_equal')})),flush=True)
            del inputs,full,generated,disabled,cached_trace,full_trace,native_cached_trace,native_full_trace,point
            branch.reset_memory();gc.collect()
    finally:
        handle.remove();branch.mode='all';branch.reset_memory()
    ensure(calls[0]==20 and len(observations)==5,'Expected all five cache cases and20 visual calls')
    return dict(observations=observations,visual_calls=calls[0],reset_exact=True,
        computational_integrity_passed=True,strict_numerical_gate_passed=all(x['passed'] for x in observations),
        criterion='paired_native_reference_centered_and_probability',
        numerical_failures_preserved=True,case_role_note='Roles refer to their status before the frozen gate attempt; no new untouched-set claim')


def evaluate(torch,model,branch,runtime,prepare,records,diagnostic_root):
    from scripts.native_aggregation import parse_answer
    from scripts.diagnose_native_vision_v5_cache import cpu_tree
    tokenizer=runtime.tokenizer;predictions=[];metrics=[]
    cells=list(dict.fromkeys(r['cell'] for r in records))
    for cell in cells:
        torch.cuda.reset_peak_memory_stats();rows=[]
        for record in [r for r in records if r['cell']==cell]:
            start=time.monotonic();inputs=prepare(record);preprocessing=time.monotonic()-start
            prompt_length=inputs['input_ids'].shape[1];captured=[]
            def capture(module,args,output):
                if not captured:
                    value=branch.export_last_query_diagnostics(cpu=False)
                    ensure(value is not None,'Missing first-prefill diagnostics');captured.append(value)
            hook=model.register_forward_hook(capture)
            branch.mode='all';torch.cuda.synchronize();start=time.monotonic()
            try:
                generated=model.generate(**inputs,do_sample=False,temperature=None,top_p=None,top_k=None,
                    use_cache=True,max_new_tokens=4,repetition_penalty=1.,
                    eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=True,output_logits=True)
                torch.cuda.synchronize();elapsed=time.monotonic()-start
            finally:hook.remove()
            ids=generated.sequences[0,prompt_length:];text=tokenizer.decode(ids,skip_special_tokens=True)
            prediction=parse_answer(text)
            target=tokenizer(str(record['gold']),add_special_tokens=False).input_ids[0]
            nll=-float(generated.logits[0][0].float().log_softmax(-1)[target])
            ensure(len(captured)==1,'Expected one prefill snapshot')
            diagnostic=cpu_tree(torch,captured[0])
            ensure(diagnostic['frame_messages'].shape==(record['n_frames'],96)
                and bool(diagnostic['visible'].all()) and int(diagnostic['query_position'])==prompt_length-1,
                'Diagnostic is not the last prompt token over all images')
            path=diagnostic_root/f'{record["sid"]}.pt';torch.save(diagnostic,path)
            native_norm=float(diagnostic['native_output_norm']);residual_norm=float(diagnostic['residual'].float().norm())
            row=dict(tag='test',mode='all',cell=cell,pair_id=record.get('pair_id'),test_family=record.get('test_family'),
                path=record['path'],sid=record['sid'],n_frames=record['n_frames'],gold=record['gold'],prediction=prediction,
                exact=prediction==record['gold'],output_text=text,generated_token_ids=ids.cpu().tolist(),
                prompt_tokens=prompt_length,generated_tokens=len(ids),gold_first_token_nll=nll,
                preprocessing_seconds=preprocessing,model_seconds=elapsed,
                branch_diagnostics_path=str(path),branch_diagnostics_sha256=sha(path),
                branch_residual_norm=residual_norm,native_attention_output_norm=native_norm,
                branch_to_native_attention_norm_ratio=residual_norm/native_norm if native_norm else None)
            rows.append(row);predictions.append(row)
            del generated,inputs,captured,diagnostic
            branch.reset_memory()
        parsed=[r for r in rows if r['prediction'] is not None];count=len(rows)
        metrics.append(dict(tag='test',mode='all',cell=cell,n_frames=rows[0]['n_frames'],n=count,
            correct=sum(r['exact'] for r in rows),exact=sum(r['exact'] for r in rows)/count,
            parsed=len(parsed),parse_rate=len(parsed)/count,
            mae_parsed=sum(abs(r['prediction']-r['gold']) for r in parsed)/len(parsed) if parsed else None,
            bias_parsed=sum(r['prediction']-r['gold'] for r in parsed)/len(parsed) if parsed else None,
            gold_first_token_nll=sum(r['gold_first_token_nll'] for r in rows)/count,
            mean_prompt_tokens=sum(r['prompt_tokens'] for r in rows)/count,
            mean_generated_tokens=sum(r['generated_tokens'] for r in rows)/count,
            mean_preprocessing_seconds=sum(r['preprocessing_seconds'] for r in rows)/count,
            mean_model_seconds=sum(r['model_seconds'] for r in rows)/count,
            mean_total_seconds=sum(r['preprocessing_seconds']+r['model_seconds'] for r in rows)/count,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved()))
    ensure(len(predictions)==10 and len(metrics)==5,'Expected ten software predictions in five cells')
    return metrics,predictions


def execute(torch,plan,plan_path):
    import bitsandbytes as bnb
    from transformers import __version__ as transformers_version
    from gnnformer.runtime import load_runtime,move_to_device,get_layers
    from gnnformer.data import load_mmred_sample,build_prompt_inputs,build_count_prompt
    from gnnformer.carriers import attach_lora
    from gnnformer.independent_vision_aggregation import attach_independent_vision_aggregation
    ensure(torch.cuda.is_available(),'CUDA required')
    ensure(bnb.matmul_4bit.__module__=='bitsandbytes.autograd._functions' and bnb.matmul_4bit.__name__=='matmul_4bit',
        'Unexpected patched bitsandbytes dispatcher')
    original_dispatch=bnb.matmul_4bit;started=time.monotonic();job=os.environ['SLURM_JOB_ID'];phase=plan['phase']
    output=OUT/f'{phase}_run_{job}';tensor_root=DATA/'v6_checkpoint_audits'/f'{phase}_{job}'
    output.mkdir(parents=True,exist_ok=False);tensor_root.mkdir(parents=True,exist_ok=False)
    (output/'plan.json').write_bytes(plan_path.read_bytes())
    runtime=load_runtime('Qwen/Qwen2.5-VL-7B-Instruct',use_4bit=True,attn_implementation='sdpa',device_map='cuda')
    model,processor=runtime.model,runtime.processor
    for parameter in model.parameters():parameter.requires_grad_(False)
    model.eval();layers=get_layers(model)
    if runtime.tokenizer.pad_token_id is None:runtime.tokenizer.pad_token_id=runtime.tokenizer.eos_token_id
    settings={k:getattr(processor.image_processor,k,None) for k in
        ('size','min_pixels','max_pixels','patch_size','temporal_patch_size','merge_size')}
    by_sid={r['sid']:r for r in plan['records']};runs=[]
    def prepare(record):
        sid,frames,question,_states,answer=load_mmred_sample(Path(record['path']));resized=[]
        ensure(sid==record['sid'] and question==record['question'] and int(answer)==record['gold'],'Sample metadata changed')
        try:
            resized=[frame.resize((392,392)) for frame in frames]
            inputs=build_prompt_inputs(processor,resized,build_count_prompt(question,len(resized)))
        finally:
            for frame in frames+resized:frame.close()
        ensure(inputs['input_ids'].shape[1]+3<=16000 and inputs['image_grid_thw'].shape[0]==record['n_frames'],'Input layout differs')
        return move_to_device(inputs,runtime.device)
    with torch.no_grad():
        for source in plan['runs']:
            config=source['config'];run_output=output/source['run_id'];run_data=tensor_root/source['run_id']
            run_output.mkdir();run_data.mkdir();(run_data/'prefill_diagnostics').mkdir()
            (run_output/'config.json').write_bytes(Path(source['config_path']).read_bytes())
            ensure(str(torch.__version__)==config['torch_version'] and str(transformers_version)==config['transformers_version'], 'Runtime versions differ')
            ensure(settings==config['image_processor_settings'],'Processor settings differ')
            branch=attach_independent_vision_aggregation(model,layer_index=14,rank=96,merge='sum')
            branch.capture_last_query_messages=True
            lora=attach_lora(layers,len(layers)-4,rank=8,alpha=16.,device=runtime.device)
            saved=torch.load(source['checkpoint'],map_location='cpu',weights_only=True)
            ensure(saved['architecture']==config['architecture'] and saved['epoch']==source['selected_epoch'],'Checkpoint metadata differs')
            branch.load_state_dict(saved['branch'],strict=True)
            ensure(set(saved['lora'])=={f'{i}.{n}' for i,n in lora.params},'LoRA keys differ')
            for (index,name),(a,b) in lora.params.items():
                a.copy_(saved['lora'][f'{index}.{name}'][0].to(a));b.copy_(saved['lora'][f'{index}.{name}'][1].to(b))
            for parameter in [*branch.parameters(),*lora.parameters()]:parameter.requires_grad_(False)
            del saved
            model.eval()
            try:
                cache=cache_audit(torch,model,branch,prepare,[by_sid[s] for s in plan['cache_case_sids']],run_data)
                save_json(run_output/'cache.json',cache)
                results,predictions=evaluate(torch,model,branch,runtime,prepare,plan['records'],run_data/'prefill_diagnostics')
                save_json(run_output/'predictions.json',predictions);save_json(run_output/'results.json',results)
                row={k:source[k] for k in ('run_id','condition','seed','checkpoint','checkpoint_sha256','selected_epoch',
                                           'config_path','config_sha256','source_code_sha256')}
                row.update(computational_integrity_passed=True,strict_numerical_gate_passed=cache['strict_numerical_gate_passed'],
                    cache=cache,predictions=predictions,results=results,output_directory=str(run_output),
                    tensor_directory=str(run_data),parameters=config['parameters'])
                save_json(run_output/'summary.json',row);runs.append(row)
                (run_output/'INDEX.md').write_text('# Fixed-checkpoint audit\n\n[Summary](summary.json) · [Cache states and numerical status](cache.json) · [Ten software predictions](predictions.json) · [Metrics](results.json) · [Original configuration](config.json).\n')
                print(json.dumps(dict(run_id=source['run_id'],computational_integrity_passed=True,
                    strict_numerical_gate_passed=row['strict_numerical_gate_passed'],software_predictions=len(predictions))),flush=True)
            finally:
                branch.remove();lora.remove();del branch,lora;gc.collect();torch.cuda.empty_cache()
    ensure(bnb.matmul_4bit is original_dispatch,'Native dispatch changed during audit')
    torch.cuda.synchronize()
    summary=dict(schema_version=1,phase=phase,plan_sha256=sha(plan_path),source_sha256=sources(),slurm_job_id=job,
        gpu=torch.cuda.get_device_name(0),elapsed_seconds=time.monotonic()-started,backend=BACKEND,
        computational_integrity_passed=all(r['computational_integrity_passed'] for r in runs),
        strict_numerical_gate_passed=all(r['strict_numerical_gate_passed'] for r in runs),runs=runs,
        scope=plan['scope'],limitation=plan['limitation'],numerical_policy=plan['numerical_policy'])
    save_json(output/'summary.json',summary)
    (output/'INDEX.md').write_text('# V6 frozen-checkpoint audit\n\n[Summary](summary.json) · [Plan](plan.json).\n\n'
        'Computational integrity and strict numerical status are separate. All numerical failures and ten software predictions per checkpoint are retained.\n')
    print(json.dumps(dict(output=str(output),computational_integrity_passed=summary['computational_integrity_passed'],
        strict_numerical_gate_passed=summary['strict_numerical_gate_passed'],elapsed_seconds=summary['elapsed_seconds'])),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check',action='store_true');group.add_argument('--plan',type=Path)
    parser.add_argument('--phase',choices=('post',),default='post')
    parser.add_argument('--run-dir',type=Path,action='append')
    args=parser.parse_args()
    ensure(os.environ.get('SLURM_JOB_ID'),'All audit compute requires Slurm')
    ensure(os.environ.get('SLURM_JOB_PARTITION')==('cpu' if args.check else 'gpu'),'Wrong Slurm partition')
    ensure(bool(os.environ.get('SLURM_JOB_GPUS',''))!=args.check,'Wrong GPU allocation')
    import torch
    torch.set_num_threads(max(1,min(4,int(os.environ.get('SLURM_CPUS_PER_TASK','1')))))
    if args.check:
        directories=args.run_dir or ([PRE] if args.phase=='pre' else [])
        plan=make_plan(torch,args.phase,directories)
        output=OUT/f'{args.phase}_check_{os.environ["SLURM_JOB_ID"]}';output.mkdir(parents=True,exist_ok=False)
        save_json(output/'plan.json',plan);(output/'plan.sha256').write_text(sha(output/'plan.json')+'\n')
        (output/'code').mkdir()
        for name in OWN:(output/'code'/name.replace('/','_')).write_bytes((REPO/name).read_bytes())
        (output/'INDEX.md').write_text('# Frozen checkpoint-audit plan\n\n[Plan](plan.json) · [SHA](plan.sha256).\n')
        print(json.dumps(dict(passed=True,plan=str(output/'plan.json'),plan_sha256=sha(output/'plan.json'))),flush=True)
    else:
        ensure(args.run_dir is None,'GPU execution uses only the frozen plan')
        ensure(sha(args.plan)==args.plan.with_suffix('.sha256').read_text().strip(),'Plan checksum differs')
        plan=json.loads(args.plan.read_text());ensure(plan['source_sha256']==sources(),'Audit source changed')
        ensure(plan['installed_backend_source_sha256']==installed_backend_sources(),'Installed backend sources changed')
        ensure(plan['bitsandbytes_version']==importlib.metadata.version('bitsandbytes'),'Bitsandbytes version changed')
        for name,digest in plan['frozen_profile_source_sha256'].items():ensure(sha(REPO/name)==digest,'Frozen V6 source changed')
        for value in plan['manifests'].values():ensure(sha(value['path'])==value['sha256'],'Manifest changed')
        for row in plan['records']:
            ensure(sha(Path(row['path'])/'qa.txt')==row['qa_sha256'],'QA changed')
            for image in row['image_files']:ensure(sha(image['path'])==image['sha256'],'Image changed')
        for run in plan['runs']:
            for path_key,hash_key in (('config_path','config_sha256'),('training_path','training_sha256'),('checkpoint','checkpoint_sha256')):
                ensure(sha(run[path_key])==run[hash_key],f'{path_key} changed')
        execute(torch,plan,args.plan)


if __name__=='__main__':main()

"""Exploratory Cosmos4096-token extension of the frozen reasoning assay.

No training, new data, prompt search, or direct-answer rerun. The exact54 main
scenes, native processor/prompt/greedy policy and strict grammar/EOS rule remain
fixed. CPU freezes this extension and original direct32/reason512 provenance.
Each new trajectory supplies censored128/512 and4096 endpoints. Prefix drift
against original512 is reported, never used to filter or select observations.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
from statistics import mean
import sys
import time

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0,str(REPO))
from scripts.probe_native_vision_reasoning import (
    OUTPUT_ROOT, MODEL_PATHS, fingerprint, metrics, prepare, require, save_json,
    score_tokens, self_test as parser_self_test, sha_file, user_text, verify_images, verify_plan,
)
from scripts.probe_native_vision_reasoning_execute import bootstrap, summarize as original_summarize

BASE_PLAN=OUTPUT_ROOT/'check_441210/plan.json'
BASE_SHA='6de7e0c1610c4ab928fec6b513fc5ef70afb2d90477c7a1b0a88c37812c3373f'
ORIGINAL_REPORT=OUTPUT_ROOT/'report_441252/analysis.json'
ORIGINAL_MAIN=OUTPUT_ROOT/'main/cosmos_441250'
ORIGINAL_PROFILE=OUTPUT_ROOT/'profile/cosmos_441213'
ORIGINAL_MAIN_HASHES=dict(summary='22e4edc9e9bed158167f10af7443501ac9123f62a9d2d99e50b0b545cf6f3270',
    predictions='57212b1058bf7c1f4922965f327d5285565fd10aacc27abdcebd454be4689830')
ROOT=REPO/'outputs/native_aggregation_vlm/reasoning_long'
CAP=4096
TOTAL_CONTEXT_CAP=18000
BOOTSTRAP_SEED=20260917
SOURCES=('scripts/probe_native_vision_cosmos_long.py',
    'slurm/native_vision_cosmos_long_check.sbatch','slurm/native_vision_cosmos_long_profile.sbatch',
    'slurm/native_vision_cosmos_long_main.sbatch','slurm/native_vision_cosmos_long_report.sbatch')
NOTES=[
    'Adaptive budget extension on already inspected scenes, prompted by original Cosmos512 truncation28/36 OOD. This is exploratory and not fresh confirmation.',
    'Only the reasoning output cap changes to4096. Direct32 predictions are reused from the verified original frozen Cosmos run; no direct rerun, prompt search, model selection or new data.',
    'All128/512/4096 new endpoints come from one4096-cap trajectory. Earlier endpoints require native EOS by that budget and have no separately measured latency.',
    'Original512 outcomes remain reported. Raw-prefix disagreements are disclosed without dropping, rerunning or choosing examples; unequal prefixes limit interpretation as continuation of the old run.',
    'The official Cosmos guide recommends4096 or more output tokens. Native EOS can still arrive later than this cap; the assay retains NF4/bf16, greedy decoding and repetition_penalty1 deviations from the official BF16/sampling example.',
    'Whole-trace think/answer grammar and native EOS are mandatory. Every truncated/malformed output stays wrong; MAE uses its explicit parsed denominator.',
    'Within-Cosmos reasoning-policy minus existing direct exact is descriptive. System policy and output budget differ; this is not an isolated causal effect of token count or evidence of an improved aggregation method.',
    'Bootstrap resamples18 complete anchors, retaining N32/N64 pairs,10000 draws seed20260917. The5pp effect screen is descriptive; no seed-population or cross-model causal inference.',
    'A natural profile may stop before4096. Report the actually exercised cache length; declaring a4096 cap alone does not demonstrate4096 decode steps.',
]


def read(path):
    return json.loads(Path(path).read_text())


def sources():
    return {name:sha_file(REPO/name) for name in SOURCES}


def snapshot(output):
    for name in SOURCES:
        target=output/'code'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((REPO/name).read_bytes())


def baseline():
    require(sha_file(BASE_PLAN)==BASE_SHA==BASE_PLAN.with_suffix('.sha256').read_text().strip(),'Original CPU plan differs')
    plan=read(BASE_PLAN)
    verify_plan(plan)
    require(plan['models']['cosmos']['path']==MODEL_PATHS['cosmos'],'Cosmos snapshot differs')
    return plan


def position_limit(config):
    text_config=getattr(config,'text_config',None)
    value=getattr(text_config,'max_position_embeddings',getattr(config,'max_position_embeddings',None))
    require(type(value) is int and value>0,'Resolved model config has no valid position limit')
    return value


def validate_native_ids(ids,eos,cap):
    require(0<len(ids)<=cap and all(type(v) is int for v in ids),'Raw generated IDs/count invalid')
    stops=[i for i,token in enumerate(ids) if token in eos]
    require(not stops or stops==[len(ids)-1],'Native EOS must be unique and terminal')
    require(bool(stops) or len(ids)==cap,'Unexpected early stop without native EOS')
    return 'native_eos' if stops else 'max_new_tokens'


def prefix_comparison(new_ids,old_ids):
    width=min(len(new_ids),len(old_ids),512)
    first=next((i for i in range(width) if new_ids[i]!=old_ids[i]),None)
    if first is None and len(new_ids)<min(len(old_ids),512):
        first=len(new_ids)
    matched=first is None and new_ids[:min(len(old_ids),512)]==old_ids[:512]
    return dict(matches_original_prefix=matched,first_mismatch_zero_based=first,
        original_tokens=len(old_ids),compared_tokens=width,
        scope='Entire old trace including EOS if completed before512; otherwise first512 generated IDs')


def verify_original(directory,base,tokenizer,profile=False):
    config,summary=read(directory/'config.json'),read(directory/'summary.json')
    require(config['model_key']=='cosmos' and config['profile'] is profile and summary['completed'] is True,
            'Expected completed original Cosmos run')
    require(config['plan_sha256']==summary['plan_sha256']==BASE_SHA==sha_file(directory/'plan.json'),
            'Original run CPU plan identity differs')
    require(config['sources']==summary['source_sha256']==base['sources'] and config['model']==base['models']['cosmos'],
            'Original source/model identity differs')
    require(sha_file(directory/'predictions.jsonl')==summary['predictions_sha256'],'Original prediction hash differs')
    for name,digest in base['sources'].items():
        require(sha_file(directory/'code'/name)==digest,'Original source snapshot differs')
    records=base['profile_records'] if profile else base['main_records']
    if profile:records=[r for r in records if r['n_frames'] in (16,64)]
    index={r['sid']:r for r in records}
    rows=[json.loads(line) for line in (directory/'predictions.jsonl').read_text().splitlines()]
    require(len(rows)==2*len(records) and len({(r['sid'],r['condition']) for r in rows})==len(rows),
            'Original scene/policy coverage differs')
    eos=base['models']['cosmos']['eos_token_ids']
    for row in rows:
        record=index[row['sid']]
        require(row['condition'] in ('direct','reason') and row['model_key']=='cosmos' and
                all(row[k]==record[k] for k in ('pair_id','n_frames','gold','qa_sha256')),'Original row identity differs')
        condition=row['condition'];cap=32 if condition=='direct' else 512
        ids=row['generated_ids'];finish=validate_native_ids(ids,eos,cap)
        require(row['generated_tokens']==len(ids) and row['generation_budget']==cap and row['finish_reason']==finish and
                row['native_eos_ids']==eos and row['raw_text']==tokenizer.decode(ids,skip_special_tokens=False,
                clean_up_tokenization_spaces=False),'Original generation metadata differs')
        layout=base['models']['cosmos']['layouts'][row['sid']][condition]
        require(row['input_ids_sha256']==layout['input_ids_sha256'] and row['prompt_tokens']==layout['prompt_tokens'] and
                row['user_text']==user_text(record),'Original input identity differs')
        expected={str(cap):score_tokens(ids,tokenizer,eos,cap,condition,record['gold'])}
        if condition=='reason':expected['128']=score_tokens(ids,tokenizer,eos,128,condition,record['gold'])
        require(row['endpoints']==expected,'Original strict scoring differs')
    computed=original_summarize(rows,profile=profile)
    require(all(summary[key]==value for key,value in computed.items()),'Original summary differs from raw rows')
    hashes={str(directory/name):sha_file(directory/name) for name in ('config.json','summary.json','predictions.jsonl','plan.json')}
    return rows,hashes


def check(args):
    import torch
    import transformers
    from transformers import AutoConfig,AutoProcessor
    torch.set_num_threads(4)
    base=baseline()
    require(torch.__version__==base['runtime']['torch_version'],'CPU torch version changed from original assay')
    processor=AutoProcessor.from_pretrained(MODEL_PATHS['cosmos'],trust_remote_code=True,use_fast=False,local_files_only=True)
    require(fingerprint(processor,transformers.__version__)==base['models']['cosmos']['processor'],'Frozen processor differs')
    report=read(ORIGINAL_REPORT)
    original=report['provenance']['cosmos']
    require(Path(original['path'])==ORIGINAL_MAIN and original['summary_sha256']==ORIGINAL_MAIN_HASHES['summary'] and
            original['predictions_sha256']==ORIGINAL_MAIN_HASHES['predictions'] and original['plan_sha256']==BASE_SHA,
            'Verified original report provenance differs')
    require(sha_file(ORIGINAL_MAIN/'summary.json')==original['summary_sha256'] and
            sha_file(ORIGINAL_MAIN/'predictions.jsonl')==original['predictions_sha256'],'Verified baseline bytes differ')
    original_main,hashes_main=verify_original(ORIGINAL_MAIN,base,processor.tokenizer)
    original_profile,hashes_profile=verify_original(ORIGINAL_PROFILE,base,processor.tokenizer,profile=True)
    require(read(ORIGINAL_MAIN/'summary.json')==report['results']['cosmos'],'Original report/run summaries disagree')
    profile=[r for r in base['profile_records'] if r['gold']==3 and r['n_frames']==64]
    require(len(profile)==1,'Expected fixed K3/N64 software scene')
    records=base['main_records']+profile
    verify_images(records)
    model_config=AutoConfig.from_pretrained(MODEL_PATHS['cosmos'],trust_remote_code=True,local_files_only=True)
    max_positions=position_limit(model_config)
    require(TOTAL_CONTEXT_CAP<=max_positions,'Requested context cap exceeds resolved model config')
    layouts={}
    for record in records:
        inputs,actual=prepare(processor,record)
        require(actual==base['models']['cosmos']['layouts'][record['sid']],'Prepared input IDs/prompt changed')
        require(actual['reason']['prompt_tokens']+CAP<=TOTAL_CONTEXT_CAP,'Reasoning prompt plus4096 exceeds18000')
        layouts[record['sid']]=actual['reason']
        del inputs
    tests=parser_self_test()
    require(prefix_comparison([1,2,3],[1,2])['matches_original_prefix'],'Matching prefix rejected')
    require(not prefix_comparison([1,9],[1,2])['matches_original_prefix'],'Mismatched prefix accepted')
    output=Path(args.output or ROOT/f'check_{os.environ["SLURM_JOB_ID"]}')
    output.mkdir(parents=True,exist_ok=False)
    plan=dict(schema_version=1,protocol='exploratory_cosmos4096',baseline_plan=str(BASE_PLAN),baseline_plan_sha256=BASE_SHA,
        main_records=base['main_records'],profile_records=profile,original_main_rows=original_main,
        original_profile_rows=original_profile,original_artifact_sha256={**hashes_main,**hashes_profile,str(ORIGINAL_REPORT):sha_file(ORIGINAL_REPORT)},
        model=base['models']['cosmos'],runtime=base['runtime'],layouts=layouts,max_new_tokens=CAP,total_context_cap=TOTAL_CONTEXT_CAP,
        model_max_position_embeddings=max_positions,bootstrap_seed=BOOTSTRAP_SEED,source_sha256=sources(),
        decoding=dict(do_sample=False,use_cache=True,repetition_penalty=1.0,temperature=None,top_p=None,top_k=None),
        tests=tests,limitations=NOTES,slurm_job_id=os.environ['SLURM_JOB_ID'])
    save_json(output/'plan.json',plan)
    (output/'plan.sha256').write_text(sha_file(output/'plan.json')+'\n')
    verify_extension_plan(output/'plan.json')
    snapshot(output)
    (output/'REPORT.md').write_text('CPU plan passed: exact54 original main inputs plus fixed K3/N64 software input, verified direct32/original512 artifacts, native EOS/parser,4096-cap/18000-total context validated against model config.\n')
    (output/'INDEX.md').write_text('# Cosmos long-budget CPU gate\n\n- [Validation](REPORT.md)\n- [Frozen plan](plan.json)\n- [Plan checksum](plan.sha256)\n- [Source snapshots](code/)\n')
    print(json.dumps(dict(plan=str(output/'plan.json'),sha256=sha_file(output/'plan.json'))),flush=True)


def verify_extension_plan(path):
    path=Path(path)
    require(sha_file(path)==path.with_suffix('.sha256').read_text().strip(),'Extension plan sidecar differs')
    plan=read(path)
    require(plan['schema_version']==1 and plan['protocol']=='exploratory_cosmos4096' and plan['source_sha256']==sources(),
            'Extension protocol/source freeze differs')
    base=baseline()
    require(plan['baseline_plan_sha256']==BASE_SHA and plan['model']==base['models']['cosmos'] and
            plan['main_records']==base['main_records'] and plan['max_new_tokens']==CAP and
            plan['total_context_cap']==TOTAL_CONTEXT_CAP and plan['bootstrap_seed']==BOOTSTRAP_SEED and
            plan['runtime']==base['runtime'],'Extension protocol/model/data changed')
    for name,digest in plan['original_artifact_sha256'].items():
        require(sha_file(name)==digest,'Original audited baseline artifact changed')
    require(plan['profile_records']==[r for r in base['profile_records'] if r['gold']==3 and r['n_frames']==64],
            'Fixed software profile scene differs')
    return plan


def verify_software_profile(directory,plan,expected_plan_sha):
    directory=Path(directory).resolve()
    require(directory.parent==ROOT/'profile','Software profile is outside the canonical extension profile root')
    config,summary=read(directory/'config.json'),read(directory/'summary.json')
    require(config['profile'] is True and summary['profile'] is True and summary['completed'] is True and
            config['protocol']==summary['protocol']==plan['protocol'],'A completed extension software profile is required')
    require(sha_file(directory/'plan.json')==expected_plan_sha==config['plan_sha256']==summary['plan_sha256'] and
            (directory/'plan.sha256').read_text().strip()==expected_plan_sha,'Software profile uses a different frozen plan')
    require(config['source_sha256']==summary['source_sha256']==plan['source_sha256']==sources() and
            config['model']==plan['model'] and config['max_new_tokens']==CAP and config['total_context_cap']==TOTAL_CONTEXT_CAP,
            'Software profile source/model/decoding identity differs')
    for name,digest in sources().items():
        require(sha_file(directory/'code'/name)==digest,'Software profile source snapshot changed')
    require(sha_file(directory/'predictions.jsonl')==summary['predictions_sha256'],'Software profile prediction artifact changed')
    rows=[json.loads(line) for line in (directory/'predictions.jsonl').read_text().splitlines()]
    require(len(rows)==summary['scene_count']==1,'Expected exactly one registered software trajectory')
    row=rows[0];record=plan['profile_records'][0];layout=plan['layouts'][record['sid']]
    require(all(row[key]==record[key] for key in ('sid','pair_id','n_frames','gold','qa_sha256')) and
            (row['n_frames'],row['gold'])==(64,3),'Software profile scene differs from fixed K3/N64')
    require(row['condition']=='reason' and row['generation_budget']==CAP and
            row['prompt_tokens']==layout['prompt_tokens'] and row['input_ids_sha256']==layout['input_ids_sha256'] and
            row['user_text']==user_text(record) and row['prompt_tokens']+CAP<=TOTAL_CONTEXT_CAP,
            'Software profile prompt or context budget differs')
    ids=row['generated_ids'];eos=plan['model']['eos_token_ids']
    require(row['generated_tokens']==len(ids) and row['native_eos_ids']==eos and
            row['finish_reason']==validate_native_ids(ids,eos,CAP),'Software profile native stopping/accounting differs')
    require(row['native_vision_forward_calls']==1 and len(row['forward_gpu_seconds'])==len(ids) and
            row['native_cache_lengths']==list(range(row['prompt_tokens'],row['prompt_tokens']+len(ids))) and
            row['maximum_native_cache_tokens']==row['native_cache_lengths'][-1],
            'Software profile did not pass one-vision-forward/native-cache growth checks')
    old=next(r for r in plan['original_profile_rows'] if r['sid']==record['sid'] and r['condition']=='reason')
    comparison=prefix_comparison(ids,old['generated_ids'])
    require(row['prefix_comparison']==comparison and summary['prefix_mismatches']==int(not comparison['matches_original_prefix']),
            'Software profile prefix discrepancy bookkeeping differs')
    require(row['full4096_steps_exercised']==summary['full4096_steps_exercised']==(len(ids)==CAP) and
            summary['total_generated_tokens']==len(ids) and summary['maximum_native_cache_tokens']==row['maximum_native_cache_tokens'],
            'Software profile exercised-context summary differs')
    return dict(path=str(directory),summary_sha256=sha_file(directory/'summary.json'),
        predictions_sha256=summary['predictions_sha256'],plan_sha256=expected_plan_sha,
        prefix_comparison=comparison,full4096_steps_exercised=row['full4096_steps_exercised'],
        maximum_native_cache_tokens=row['maximum_native_cache_tokens'],
        gate_scope='Completed fixed-scene software execution with matching plan/model/source/cache identity; no accuracy or prefix-agreement selection criterion')


def run(args):
    plan=verify_extension_plan(args.plan)
    require(args.profile or args.profile_run,'Main requires an explicitly supplied matching successful software profile')
    profile_provenance=None if args.profile else verify_software_profile(args.profile_run,plan,sha_file(args.plan))
    records=plan['profile_records'] if args.profile else plan['main_records']
    original=plan['original_profile_rows'] if args.profile else plan['original_main_rows']
    old={r['sid']:r for r in original if r['condition']=='reason'}
    verify_images(records)
    output=Path(args.output or ROOT/('profile' if args.profile else 'main')/f'cosmos_{os.environ["SLURM_JOB_ID"]}')
    output.mkdir(parents=True,exist_ok=False)
    (output/'plan.json').write_bytes(Path(args.plan).read_bytes())
    (output/'plan.sha256').write_text(sha_file(output/'plan.json')+'\n')
    snapshot(output)
    config=dict(protocol=plan['protocol'],profile=args.profile,plan_sha256=sha_file(args.plan),source_sha256=sources(),
        model=plan['model'],max_new_tokens=CAP,total_context_cap=TOTAL_CONTEXT_CAP,
        software_profile=profile_provenance,slurm_job_id=os.environ['SLURM_JOB_ID'])
    save_json(output/'config.json',config)
    import torch
    import transformers
    from gnnformer.runtime import load_runtime,move_to_device
    require(torch.cuda.is_available(),'Run requires a Slurm GPU')
    require(transformers.__version__==plan['runtime']['transformers_version'] and
            torch.__version__==plan['runtime']['torch_version'],'Original torch/transformers version differs')
    torch.set_num_threads(4)
    # Preserve the original generation seed; bootstrap has its separately registered seed.
    torch.manual_seed(20260915);torch.cuda.manual_seed_all(20260915)
    started=time.perf_counter()
    rt=load_runtime(plan['model']['path'],use_4bit=True,attn_implementation='sdpa',device_map='cuda')
    rt.model.requires_grad_(False);rt.model.eval()
    require(fingerprint(rt.processor,transformers.__version__)==plan['model']['processor'],'Loaded processor differs')
    require(rt.model.generation_config.eos_token_id==plan['model']['eos_token_ids'],'Native EOS settings differ')
    maximum=position_limit(rt.model.config)
    require(maximum==plan['model_max_position_embeddings'] and maximum>=TOTAL_CONTEXT_CAP,'Loaded model context limit differs')
    torch.cuda.synchronize();load_seconds=time.perf_counter()-started;torch.cuda.reset_peak_memory_stats()
    rows=[]
    with (output/'predictions.jsonl').open('x') as stream:
        for record in records:
            prep_started=time.perf_counter();prepared,layouts=prepare(rt.processor,record)
            require(layouts['reason']==plan['layouts'][record['sid']],'Actual extension prompt differs')
            inputs=move_to_device(prepared['reason'],rt.device)
            require(inputs['input_ids'].shape[1]+CAP<=TOTAL_CONTEXT_CAP,'Runtime total-context guard exceeded')
            preparation_seconds=time.perf_counter()-prep_started
            events=[];vision_calls=[0];cache_lengths=[]
            def before(module,positional):
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record();events.append((begin,end))
            def after(module,positional,result):
                events[-1][1].record()
                cache=getattr(result,'past_key_values',None)
                require(cache is not None and hasattr(cache,'get_seq_length'),'Native forward did not return a dynamic cache')
                cache_lengths.append(int(cache.get_seq_length()))
            handles=[rt.model.register_forward_pre_hook(before),rt.model.register_forward_hook(after),
                rt.model.model.visual.register_forward_hook(lambda module,args,result:vision_calls.__setitem__(0,vision_calls[0]+1))]
            torch.cuda.synchronize();generation_started=time.perf_counter()
            try:
                with torch.inference_mode():
                    generated=rt.model.generate(**inputs,do_sample=False,use_cache=True,max_new_tokens=CAP,
                        repetition_penalty=1.0,temperature=None,top_p=None,top_k=None,
                        eos_token_id=plan['model']['eos_token_ids'],pad_token_id=plan['model']['pad_token_id'],
                        return_dict_in_generate=True,output_scores=False)
                torch.cuda.synchronize()
            finally:
                for handle in handles:handle.remove()
            seconds=time.perf_counter()-generation_started;prompt=inputs['input_ids'].shape[1]
            require(generated.sequences.shape[0]==1 and torch.equal(generated.sequences[0,:prompt],inputs['input_ids'][0]),
                    'Generation altered original input prefix')
            ids=generated.sequences[0,prompt:].tolist();finish=validate_native_ids(ids,plan['model']['eos_token_ids'],CAP)
            require(vision_calls[0]==1 and len(events)==len(ids) and
                    cache_lengths==list(range(prompt,prompt+len(ids))),'Native one-vision-forward/cache-growth accounting differs')
            durations=[begin.elapsed_time(end)/1000 for begin,end in events]
            endpoints={str(b):score_tokens(ids,rt.tokenizer,plan['model']['eos_token_ids'],b,'reason',record['gold']) for b in (128,512,CAP)}
            row=dict(sid=record['sid'],pair_id=record['pair_id'],n_frames=record['n_frames'],gold=record['gold'],
                qa_sha256=record['qa_sha256'],condition='reason',generated_ids=ids,generated_tokens=len(ids),
                raw_text=rt.tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False),
                finish_reason=finish,native_eos_ids=plan['model']['eos_token_ids'],generation_budget=CAP,
                prompt_tokens=prompt,input_ids_sha256=layouts['reason']['input_ids_sha256'],user_text=user_text(record),
                endpoints=endpoints,prefix_comparison=prefix_comparison(ids,old[record['sid']]['generated_ids']),
                preprocessing_seconds=preparation_seconds,generation_seconds=seconds,
                forward_gpu_seconds=durations,prefill_gpu_seconds=durations[0],decode_gpu_seconds=sum(durations[1:]),
                native_vision_forward_calls=vision_calls[0],native_cache_lengths=cache_lengths,
                maximum_native_cache_tokens=cache_lengths[-1],full4096_steps_exercised=len(ids)==CAP,
                sequence_tokens_exercised=prompt+len(ids),maximum_forward_input_position=prompt+len(ids)-2,
                peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(),peak_memory_reserved_bytes=torch.cuda.max_memory_reserved())
            rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
            print(json.dumps({key:row[key] for key in ('sid','n_frames','generated_tokens','finish_reason','generation_seconds','prefix_comparison')}),flush=True)
            del prepared,inputs,generated,events
    verify_extension_plan(args.plan)
    summary=dict(protocol=plan['protocol'],completed=True,profile=args.profile,scene_count=len(rows),
        plan_sha256=config['plan_sha256'],source_sha256=sources(),predictions_sha256=sha_file(output/'predictions.jsonl'),
        model_load_seconds=load_seconds,total_seconds=time.perf_counter()-started,
        total_generation_seconds=sum(r['generation_seconds'] for r in rows),total_generated_tokens=sum(r['generated_tokens'] for r in rows),
        maximum_sequence_tokens_exercised=max(r['sequence_tokens_exercised'] for r in rows),
        maximum_native_cache_tokens=max(r['maximum_native_cache_tokens'] for r in rows),
        full4096_steps_exercised=any(r['full4096_steps_exercised'] for r in rows),
        prefix_mismatches=sum(not r['prefix_comparison']['matches_original_prefix'] for r in rows),
        peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(),peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(),limitations=NOTES)
    if not args.profile:summary['results']=summarize(rows,plan['original_main_rows'])
    save_json(output/'summary.json',summary)
    (output/'INDEX.md').write_text('# Cosmos long-budget run\n\n- [Summary](summary.json)\n- [All raw trajectories](predictions.jsonl)\n- [Frozen plan](plan.json)\n')
    print(json.dumps(dict(output=str(output),completed=True,summary={k:summary[k] for k in ('total_seconds','total_generated_tokens','prefix_mismatches','full4096_steps_exercised')})),flush=True)


def summarize(rows,original):
    index={r['sid']:r for r in rows}
    require(len(rows)==len(index)==54,'All54 new trajectories required')
    old={(r['sid'],r['condition']):r for r in original}
    groups={}
    for row in rows:
        groups.setdefault(row['pair_id'],{})[row['n_frames']]=row['sid']
    require(len(groups)==18 and all(set(g)=={16,32,64} for g in groups.values()),'Complete18 three-length families required')
    endpoints={'direct32':{sid:old[sid,'direct']['endpoints']['32'] for sid in index},
        'original_reason512':{sid:old[sid,'reason']['endpoints']['512'] for sid in index}}
    endpoints.update({f'extended_reason{budget}':{sid:row['endpoints'][str(budget)] for sid,row in index.items()} for budget in (128,512,CAP)})
    cells={}
    for endpoint,values in endpoints.items():
        cells[endpoint]={name:metrics([values[sid] for sid,row in index.items() if row['n_frames'] in lengths])
            for name,lengths in [('all',(16,32,64)),('familiar_ood',(32,64)),('N16',(16,)),('N32',(32,)),('N64',(64,))]}
        cells[endpoint]['per_k']={f'N{n}_K{k}':metrics([values[sid] for sid,row in index.items() if row['n_frames']==n and row['gold']==k]) for n in (16,32,64) for k in range(9)}
    contrasts={}
    for endpoint in ('original_reason512','extended_reason128','extended_reason512','extended_reason4096'):
        contrasts[endpoint]={}
        for name,lengths in [('familiar_ood',(32,64)),('N16',(16,))]:
            effects=[mean(int(endpoints[endpoint][group[n]]['correct'])-int(endpoints['direct32'][group[n]]['correct']) for n in lengths) for pair,group in sorted(groups.items())]
            contrasts[endpoint][name]=bootstrap(effects,seed=BOOTSTRAP_SEED)
    extension={}
    for endpoint,values in endpoints.items():
        extension[endpoint]={}
        for n in (32,64):
            paired=[(values[group[16]],values[group[n]]) for group in groups.values()]
            extension[endpoint][f'N16_N{n}']=dict(n=18,both_correct=sum(a['correct'] and b['correct'] for a,b in paired),
                short_correct_long_wrong=sum(a['correct'] and not b['correct'] for a,b in paired),
                both_parsed=sum(a['parsed'] and b['parsed'] for a,b in paired),
                invariant_parsed=sum(a['parsed'] and b['parsed'] and a['prediction']==b['prediction'] for a,b in paired))
    return dict(cells=cells,contrasts_vs_existing_direct=contrasts,extension=extension,
        primary_screen_pass=contrasts['extended_reason4096']['familiar_ood']['estimate']>=.05,
        primary='Extended4096 minus existing direct32 OOD exact >=5pp, descriptive; N16 loss reported separately',
        prefix_mismatch_by_n={str(n):dict(n=18,mismatches=sum(not r['prefix_comparison']['matches_original_prefix'] for r in rows if r['n_frames']==n)) for n in (16,32,64)})


def report(args):
    import transformers
    from transformers import AutoProcessor
    run_dir=Path(args.run_dir)
    config,summary=read(run_dir/'config.json'),read(run_dir/'summary.json')
    plan=verify_extension_plan(run_dir/'plan.json')
    require(not config['profile'] and summary['completed'] is True and summary['profile'] is False,'Expected completed main extension')
    require(config['plan_sha256']==summary['plan_sha256']==sha_file(run_dir/'plan.json') and
            config['source_sha256']==summary['source_sha256']==sources(),'Run plan/source identity differs')
    require(config['model']==plan['model'] and config['max_new_tokens']==CAP and config['total_context_cap']==TOTAL_CONTEXT_CAP,
            'Run model/cap settings differ')
    require(config['software_profile']==verify_software_profile(config['software_profile']['path'],plan,config['plan_sha256']),
            'Main software-profile provenance differs')
    for name,digest in sources().items():require(sha_file(run_dir/'code'/name)==digest,'Run source snapshot differs')
    require(sha_file(run_dir/'predictions.jsonl')==summary['predictions_sha256'],'Extension raw predictions changed')
    processor=AutoProcessor.from_pretrained(plan['model']['path'],trust_remote_code=True,use_fast=False,local_files_only=True)
    require(fingerprint(processor,transformers.__version__)==plan['model']['processor'],'Report processor differs')
    rows=[json.loads(line) for line in (run_dir/'predictions.jsonl').read_text().splitlines()]
    records={r['sid']:r for r in plan['main_records']};old={r['sid']:r for r in plan['original_main_rows'] if r['condition']=='reason'}
    require(len(rows)==len({r['sid'] for r in rows})==54 and {r['sid'] for r in rows}==set(records),'Incomplete/mismatched new sample identities')
    for row in rows:
        record=records[row['sid']];ids=row['generated_ids'];eos=plan['model']['eos_token_ids']
        require(all(row[k]==record[k] for k in ('pair_id','n_frames','gold','qa_sha256')),'New row identity differs')
        require(row['condition']=='reason' and row['generation_budget']==CAP and row['generated_tokens']==len(ids) and
                row['finish_reason']==validate_native_ids(ids,eos,CAP) and row['native_eos_ids']==eos,'New generation metadata differs')
        require(row['raw_text']==processor.tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False),
                'New raw text differs from IDs')
        require(row['endpoints']=={str(b):score_tokens(ids,processor.tokenizer,eos,b,'reason',record['gold']) for b in (128,512,CAP)},
                'New endpoints differ from strict complete-trace rescoring')
        require(row['prefix_comparison']==prefix_comparison(ids,old[row['sid']]['generated_ids']),'Prefix comparison differs')
        layout=plan['layouts'][row['sid']]
        require(row['input_ids_sha256']==layout['input_ids_sha256'] and row['prompt_tokens']==layout['prompt_tokens'] and
                row['user_text']==user_text(record) and row['prompt_tokens']+CAP<=TOTAL_CONTEXT_CAP,'New prompt/context metadata differs')
        require(row['native_cache_lengths']==list(range(row['prompt_tokens'],row['prompt_tokens']+len(ids))) and
                row['maximum_native_cache_tokens']==row['native_cache_lengths'][-1] and
                row['maximum_forward_input_position']==row['prompt_tokens']+len(ids)-2, 'Actual native cache lengths differ')
        require(row['native_vision_forward_calls']==1 and len(row['forward_gpu_seconds'])==len(ids) and
                row['sequence_tokens_exercised']==row['prompt_tokens']+len(ids) and row['full4096_steps_exercised']==(len(ids)==CAP),
                'Native cache/context accounting differs')
        require(all(isinstance(row[k],(int,float)) and math.isfinite(row[k]) and row[k]>=0 for k in
                    ('preprocessing_seconds','generation_seconds','prefill_gpu_seconds','decode_gpu_seconds')) and
                all(math.isfinite(value) and value>=0 for value in row['forward_gpu_seconds']), 'Nonfinite/negative timing data')
        require(row['prefill_gpu_seconds']==row['forward_gpu_seconds'][0] and
                row['decode_gpu_seconds']==sum(row['forward_gpu_seconds'][1:]), 'Prefill/decode timing sums differ')
    computed=summarize(rows,plan['original_main_rows'])
    require(summary['results']==computed and summary['total_generated_tokens']==sum(r['generated_tokens'] for r in rows) and
            summary['total_generation_seconds']==sum(r['generation_seconds'] for r in rows),'Result or cost summary differs')
    require(summary['scene_count']==54 and summary['prefix_mismatches']==sum(not r['prefix_comparison']['matches_original_prefix'] for r in rows) and
            summary['maximum_sequence_tokens_exercised']==max(r['sequence_tokens_exercised'] for r in rows) and
            summary['maximum_native_cache_tokens']==max(r['maximum_native_cache_tokens'] for r in rows) and
            summary['full4096_steps_exercised']==any(r['full4096_steps_exercised'] for r in rows), 'Prefix/context aggregate bookkeeping differs')
    output=Path(args.output or ROOT/f'report_{os.environ["SLURM_JOB_ID"]}')
    output.mkdir(parents=True,exist_ok=False);snapshot(output)
    analysis=dict(protocol=plan['protocol'],results=computed,cost={k:summary[k] for k in (
        'model_load_seconds','total_seconds','total_generation_seconds','total_generated_tokens',
        'maximum_sequence_tokens_exercised','maximum_native_cache_tokens','peak_memory_allocated_bytes','peak_memory_reserved_bytes')},
        run=str(run_dir.resolve()),run_summary_sha256=sha_file(run_dir/'summary.json'),predictions_sha256=summary['predictions_sha256'],
        plan_sha256=config['plan_sha256'],source_sha256=sources(),software_profile=config['software_profile'],
        original_artifact_sha256=plan['original_artifact_sha256'],limitations=NOTES)
    save_json(output/'analysis.json',analysis)
    lines=['# Cosmos: exploratory4096-token reasoning extension','',
        'The existing direct32 and original512 outcomes remain fixed. New128/512/4096 endpoints share one4096-cap trace per scene.','',
        '| Policy | N16 exact | N32/N64 exact | OOD completed | OOD parsed | OOD truncated | OOD MAE (parsed n) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for endpoint,cells in computed['cells'].items():
        short,long=cells['N16'],cells['familiar_ood'];mae='NA' if long['mae'] is None else f"{long['mae']:.3f} ({long['mae_denominator']})"
        lines.append(f"| {endpoint} | {short['correct']}/18 | {long['correct']}/36 | {long['completed']}/36 | {long['parsed']}/36 | {long['truncated']}/36 | {mae} |")
    effect=computed['contrasts_vs_existing_direct']['extended_reason4096'];delta=effect['familiar_ood']
    lines+=['',f"Primary descriptive difference: {100*delta['estimate']:+.2f}pp, paired-anchor interval[{100*delta['lower']:+.2f},{100*delta['upper']:+.2f}].5pp screen: {computed['primary_screen_pass']}. N16 difference: {100*effect['N16']['estimate']:+.2f}pp.",
        f"Prefix mismatches against original512: {summary['prefix_mismatches']}/54. No scenes were removed or rerun.",
        f"Measured generation {summary['total_generation_seconds']:.1f}s, total {summary['total_seconds']:.1f}s including load; {summary['total_generated_tokens']} generated tokens. No separate128/512 latency is inferred.",
        '', '## Interpretation limits','']+['- '+note for note in NOTES]
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    for endpoint,cells in computed['cells'].items():
        ax.plot([16,32,64],[100*cells[f'N{n}']['exact'] for n in (16,32,64)],marker='o',label=endpoint)
    ax.set(xlabel='Frames',ylabel='Strict completed exact (%)',xticks=[16,32,64],ylim=(-2,102),title='Exploratory Cosmos budget extension')
    ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=8);fig.tight_layout()
    fig.savefig(output/'comparison.png',dpi=170);fig.savefig(output/'comparison.pdf');plt.close(fig)
    (output/'INDEX.md').write_text('# Cosmos long-budget analysis\n\n- [Report](REPORT.md)\n- [Audited results](analysis.json)\n- [Figure](comparison.png)\n- [Figure PDF](comparison.pdf)\n')
    print(json.dumps(dict(output=str(output),primary=computed['primary_screen_pass'])),flush=True)


def main():
    require(bool(os.environ.get('SLURM_JOB_ID')),'All CPU/GPU processing requires Slurm')
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-plan',action='store_true');mode.add_argument('--run',action='store_true');mode.add_argument('--report',action='store_true')
    parser.add_argument('--plan');parser.add_argument('--profile-run');parser.add_argument('--profile',action='store_true');parser.add_argument('--run-dir');parser.add_argument('--output')
    args=parser.parse_args()
    if args.check_plan:check(args)
    elif args.run:run(args)
    else:report(args)


if __name__=='__main__':main()

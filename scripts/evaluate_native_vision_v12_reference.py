"""Exploratory V12 whole-answer reference study on34 fixed old V10 scenes.

No fitting or model selection. All four dev-selected cores generate all three
modes from identical augmented actual/reference/native global prompts. Reference
states are recomputed at every generated prefix; static first-query calibration
vectors never substitute for live reference rows. Software and resource gates
must pass before the four bounded Slurm allocations may run.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
OUT=REPO/'outputs/native_aggregation_vlm/v12/reference_study'
DATA=Path('/mnt/data/gabriele/gnn_transformer/v12_reference_study')
MODEL=Path('/mnt/ckpts/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5')
BACKGROUND=REPO/'outputs/native_aggregation_vlm/v10/background_diagnostic/run_442260/summary.json'
MODES=('base','background','sham')
CORE_KEYS=('ce_s14','ce_s15','consistency_s14','consistency_s15')
POLICY=dict(protocol='v12_reference_whole_answer_exploratory',modes=list(MODES),core_keys=list(CORE_KEYS),
    scenes=34,families=17,counts=list(range(17)),lengths=[32,64],reference_occurrences=24,anchor_n=16,
    coefficient=1,sham_seed=20261101,reference_mean='FP64 occurrence mean followed by FP32',
    reference_rule='Frozen N8K0 then N16K0 training scenes for the exact question; no deduplication or target-prefix selection',
    native_rows='N actual image streams +24 training-reference image streams +one text-only global stream',
    max_new_tokens=4,native_eos=[151645,151643],target_eos=151645,
    generations_per_model=102,maximum_model_forwards_per_model=408,visual_forwards_per_model=102,
    total_generations=408,maximum_model_forwards=1632,total_visual_forwards=408,
    per_model_gpu_seconds_cap=1200,study_gpu_seconds_cap=5100,software_gpu_seconds_cap_separate=480,
    projection='software_load +1.25*102*augmented_N64_four_token_time +30',
    scene_order='exact old background plan order; modes base/background/sham within each scene',
    primary_measure='Complete stripped ASCII integer equals gold AND native EOS within four tokens; no nonterminal special tokens',
    no_fit=True,no_selection=True,reused_test_exploratory=True,no_practical_milestone_claim=True)
OWN=('scripts/evaluate_native_vision_v12_reference.py',
     'slurm/native_vision_v12_reference_study_check.sbatch','slurm/native_vision_v12_reference_study.sbatch')


def need(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def object_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def save(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2)

def atomic_json(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2);stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)


def modules():
    # Called only after the CLI Slurm guard; native modules import torch.
    return (importlib.import_module('scripts.native_vision_v12_runtime'),
            importlib.import_module('scripts.profile_native_vision_v12_reference'),
            importlib.import_module('scripts.probe_native_vision_v10_background'))


def sources():
    runtime,software,background=modules()
    names=tuple(dict.fromkeys((*OWN,*software.OWN,*background.OWN,'scripts/native_vision_v12_runtime.py',
        'gnnformer/parallel_local_reference.py','tests/test_parallel_local_reference.py')))
    return {name:sha(REPO/name) for name in names}


def snapshot(directory):
    (directory/'code').mkdir();frozen=sources()
    for name,digest in frozen.items():
        path=directory/'code'/name.replace('/','_');path.write_bytes((REPO/name).read_bytes());need(sha(path)==digest,'Source changed during snapshot')
    save(directory/'source_hashes.json',frozen)
    (directory/'INDEX.md').write_text('# V12 whole-answer study\n\n[Summary](summary.json) · [Sources](source_hashes.json) · [Progress](progress.json).\n')
    return frozen


def bind(path,bindings,expected=None):
    path=Path(path).resolve();digest=sha(path)
    need(expected is None or digest==expected,'Bound artifact changed: '+str(path))
    need(str(path) not in bindings or bindings[str(path)]==digest,'Conflicting artifact digest')
    bindings[str(path)]=digest
    return dict(file=str(path),sha256=digest)


def check_sources(value,directory):
    need(value==sources(),'Frozen V12 study source changed')
    for name,digest in value.items():need(sha(Path(directory)/'code'/name.replace('/','_'))==digest,'Copied study source differs')


def parse_integer(text):
    text=text.strip()
    return int(text) if re.fullmatch(r'[0-9]+',text) else None


def score_ids(tokenizer,ids,gold):
    need(isinstance(ids,list) and 1<=len(ids)<=4 and all(type(x) is int and x>=0 for x in ids),'Malformed native generation IDs')
    complete=ids[-1] in POLICY['native_eos']
    need(not any(t in POLICY['native_eos'] for t in ids[:-1]) and (complete or len(ids)==4),'Interior EOS or unsupported early stop')
    raw=tokenizer.decode(ids,skip_special_tokens=False);text=tokenizer.decode(ids,skip_special_tokens=True)
    body_ids=ids[:-1] if complete else ids
    body=tokenizer.decode(body_ids,skip_special_tokens=False)
    no_nonterminal_special_tokens=not any(t in set(tokenizer.all_special_ids) for t in body_ids)
    prediction=parse_integer(body) if no_nonterminal_special_tokens else None
    first=tokenizer(str(gold),add_special_tokens=False)['input_ids'][0]
    return dict(generated_ids=ids,raw_text=raw,text=text,answer_body=body,prediction=prediction,parseable=prediction is not None,
        no_nonterminal_special_tokens=no_nonterminal_special_tokens,
        parsed_count_correct=prediction==gold,completed=complete,truncated=not complete,
        finish_reason='eos' if complete else 'length',exact=prediction==gold and complete,
        first_token_correct=ids[0]==first,correct_first_but_wrong_whole=ids[0]==first and not(prediction==gold and complete))


def self_test():
    class Tokenizer:
        all_special_ids=[0,*POLICY['native_eos']]
        def __call__(self,text,**kwargs):return {'input_ids':[int(c)+10 for c in text]}
        def decode(self,ids,skip_special_tokens=False):
            return ''.join(str(i-10) if 10<=i<=19 else ('' if skip_special_tokens and i in self.all_special_ids else '<end>' if i in POLICY['native_eos'] else '<bos>' if i==0 else 'x') for i in ids)
    tok=Tokenizer();eos=151645
    need(score_ids(tok,[11,16,eos],16)['exact'],'Multi-digit integer/EOS positive case failed')
    proxy=score_ids(tok,[11,eos],16);need(proxy['first_token_correct'] and not proxy['exact'],'First token incorrectly counted as full answer')
    need(not score_ids(tok,[11,16,0,eos],16)['parseable'],'Malformed output accepted')
    special=score_ids(tok,[0,11,eos],1)
    need(special['text']=='1' and not special['no_nonterminal_special_tokens'] and not special['parseable'] and not special['exact'],
         'Decoded-away nonterminal special token accepted')
    need(score_ids(tok,[11,16,0,0],16)['truncated'] and not score_ids(tok,[11,16,0,0],16)['exact'],'Truncated trace accepted')
    for bad in ([11,16],[11,eos,16],[],[11,16,eos,0]):
        try:score_ids(tok,bad,16)
        except ValueError:pass
        else:raise AssertionError('Invalid stop sequence accepted')
    need(parse_integer(' 10\n')==10 and parse_integer('1 0') is None and parse_integer('一') is None,'Strict integer grammar differs')
    return dict(passed=True,tests=['full_integer_EOS','reject_shared_first_digit_proxy','malformed_and_truncated_in_denominator','reject_interior_EOS','reject_nonterminal_special_tokens','strict_ASCII'])


def verify_background(bindings,*,deep=False):
    _,_,background=modules();summary=read(BACKGROUND);bind(BACKGROUND,bindings)
    need(summary['passed'] is True and summary['computational_integrity_passed'] is True and summary['diagnostic_only'] is True
         and summary['no_fit'] is True and summary['first_token_observations']==408 and summary['scenes']==34
         and summary['bare_identity_passes']==34 and summary['bare_identity_failures']==0
         and summary['baseline_decomposition_passes']==136 and summary['counters']==dict(model=34,visual=34,extra_head=442),
         'Canonical background diagnostic incomplete')
    bind(summary['analysis_file'],bindings,summary['analysis_sha256']);analysis=read(summary['analysis_file'])
    bind(analysis['plan_file'],bindings,analysis['plan_sha256']);plan=read(analysis['plan_file'])
    need(plan['source_sha256']==background.sources() and plan['policy']==background.POLICY
         and plan['runs'] and len(plan['cases'])==34,'Frozen background source/selection differs')
    need(Path(analysis['plan_file']).with_suffix('.sha256').read_text().strip()==analysis['plan_sha256'],'Background plan sidecar differs')
    for name,digest in summary['source_sha256'].items():
        bind(REPO/name,bindings,digest);need(sha(BACKGROUND.parent/'code'/name.replace('/','_'))==digest,'Background source copy differs')
    for field in ('report','finalization','calibration_slots'):
        bind(plan[field+'_file'],bindings,plan[field+'_sha256'])
    if deep:
        checked=background.verify_plan(Path(analysis['plan_file']));need(checked==plan,'Completed background source gate differs')
    rows=analysis['rows'];need(len(rows)==len({(r['model'],r['sid'],r['intervention']) for r in rows})==408,'Background row coverage differs')
    for row in rows:need(row['first_token_correct']==(row['first_token_id']==row['target_first_token']),'Background first-token labels differ')
    for cell in analysis['summaries']:
        selected=[r for r in rows if (r['model'],r['n_frames'],r['intervention'])==(cell['model'],cell['n_frames'],cell['intervention'])
                  and (cell['scope']=='all' or r['first_token_scope']==cell['scope'])]
        need(len(selected)==cell['n'] and sum(r['first_token_correct'] for r in selected)==cell['first_token_correct']
             and math.isclose(sum(r['first_token_nll'] for r in selected)/len(selected),cell['first_token_nll'],rel_tol=1e-12),
             'Background summary differs from all raw JSON observations')
    return plan,analysis


def audit_banks(runtime,background_plan,bindings):
    slots=read(background_plan['calibration_slots_file']);need(len(slots)==108,'Original training calibration has108 scene slots')
    manifest=read(runtime.REFERENCE_MANIFEST);bind(runtime.REFERENCE_MANIFEST,bindings)
    train={r['sid']:r for n in (8,16) for r in manifest['splits'][f'train_N{n}']['samples']}
    questions=sorted({r['question'] for r in slots});need(len(questions)==54,'Reference question support changed')
    banks={};_,_,background=modules()
    for q in questions:
        expected=[r for r in slots if r['question']==q]
        need([r['n_frames'] for r in expected]==[8,16] and all(r['gold']==0 for r in expected),'Reference scene order/support changed')
        bank=runtime.reference_bank(q);need([r['sid'] for r in bank['scenes']]==[r['sid'] for r in expected],'Live bank chose different calibration scenes')
        occurrences=[]
        for row in expected:
            sample=train[row['sid']];background.verify_qa(sample,bindings)
            need(sample['gold']==0 and sample['question']==q and len(row['local_feature_ids'])==sample['n_frames'],'Calibration null label or occurrence count differs')
            for i,image in enumerate(sample['image_files']):
                bind(image['path'],bindings,image['sha256']);occurrences.append((row['sid'],i,image['sha256']))
        need([(r['source_sid'],r['source_frame_index'],r['sha256']) for r in bank['occurrences']]==occurrences and len(occurrences)==24,
             'Bank deduplicated/reordered images or includes different evidence')
        banks[q]=bank
    need(sum(len(b['occurrences']) for b in banks.values())==1296,'Training reference weighting differs')
    return banks


def check(args,out,frozen):
    import torch,transformers
    from transformers import AutoProcessor
    from scripts.probe_native_vision_v2_prefix import fingerprint
    runtime,software,background=modules();torch.set_num_threads(4);started=time.perf_counter();bindings={}
    tests=self_test();old,old_analysis=verify_background(bindings,deep=True)
    gate,software_plan=verify_software(args.software_summary,bindings)
    processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,use_fast=False)
    need(fingerprint(processor,str(transformers.__version__))==old['processor']==software_plan['processor']
         and old['model']==software_plan['model'] and old['runtime']==software_plan['runtime'],
         'Current processor/model/runtime differs from both frozen ancestors')
    report=read(old['report_file']);final=read(old['finalization_file'])
    selected,branches,_=background.load_selected(torch,report,final,bindings)
    need(selected==old['runs'] and [r['key'] for r in selected]==list(CORE_KEYS),'Selected model set/order changed')
    del branches
    banks=audit_banks(runtime,old,bindings)
    records=[c['record'] for c in old['cases']]
    need(records==background.choose_records(read(background.TEST)),'Study does not use the original fixed34 scenes')
    data=DATA/out.name;data.mkdir(parents=True,exist_ok=False)
    prepared=[];owner,fn,_=background.feature.native_api(processor)
    for i,sample in enumerate(records):
        background.verify_qa(sample,bindings)
        for im in sample['image_files']:bind(im['path'],bindings,im['sha256'])
        bundle=runtime.prepare_scene(processor,sample,banks[sample['question']],verify_processor_parity=True)
        layout=runtime.native.audit_layout(lambda **kwargs:fn(owner,**kwargs),bundle)
        need(bundle['metadata']['prefix_ids']==[],'No study answer prefix may be supplied')
        path=data/f'input_{i:03d}.pt';torch.save(bundle,path)
        prepared.append(dict(record=sample,file=str(path),sha256=sha(path),metadata=bundle['metadata'],layout=layout['metadata']))
        print(json.dumps(dict(prepared=i+1,total=34,sid=sample['sid'])),flush=True)
    target_ids={str(k):runtime.native.encode_target(processor.tokenizer,k) for k in range(17)}
    projection=study_projection(gate);allocation=allocation_ledger()
    need(allocation['allocated_gpu_seconds']+4800<=5100,'Previous study failures leave insufficient four-allocation reserve')
    plan=dict(schema_version=1,policy=POLICY,source_sha256=frozen,artifact_bindings=bindings,software_gate=gate,
        software_plan_file=gate['plan_file'],software_plan_sha256=gate['plan_sha256'],
        background_summary_file=str(BACKGROUND),background_summary_sha256=sha(BACKGROUND),
        background_plan_file=old_analysis['plan_file'],background_plan_sha256=old_analysis['plan_sha256'],
        selected_models=selected,cases=prepared,reference_banks=banks,reference_occurrences_total=1296,
        model=old['model'],runtime=old['runtime'],processor=old['processor'],native_api=software_plan['native_api'],
        native_dtypes=old['native_dtypes'],target_token_ids=target_ids,projection=projection,prior_allocation=allocation,tests=tests,
        no_fit=True,no_model_selection=True,no_static_prefix_calibration=True,slurm_job_id=os.environ['SLURM_JOB_ID'])
    save(out/'plan.json',plan);(out/'plan.sha256').write_text(sha(out/'plan.json')+'\n')
    summary=dict(passed=projection['passed'],completed=True,no_model_loaded=True,plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),
        source_sha256=frozen,projection=projection,cases=34,cores=4,tests=tests,seconds=time.perf_counter()-started)
    save(out/'summary.json',summary)
    need(sources()==frozen,'Study source changed during CPU preparation')
    need(summary['passed'],'Augmented native profile projects beyond the registered1200-second per-model cap; preserve failure')


def verify_plan(path,*,prepared=False):
    path=Path(path).resolve();plan=read(path);gate=read(path.parent/'summary.json')
    need(path.is_relative_to(OUT) and path.with_suffix('.sha256').read_text().strip()==sha(path)
         and plan['schema_version']==1 and plan['policy']==POLICY and gate['passed'] is True and gate['completed'] is True
         and gate['no_model_loaded'] is True and gate['plan_sha256']==sha(path),'Completed exact CPU study plan required')
    check_sources(plan['source_sha256'],path.parent)
    for filename,digest in plan['artifact_bindings'].items():need(sha(filename)==digest,'Study input changed: '+filename)
    software_gate,_=verify_software(plan['software_gate']['file'],{})
    need(software_gate==plan['software_gate'] and study_projection(software_gate)==plan['projection'] and plan['projection']['passed'],
         'Software/resource gate changed')
    need([r['key'] for r in plan['selected_models']]==list(CORE_KEYS) and len(plan['cases'])==34
         and Counter((c['record']['n_frames'],c['record']['gold']) for c in plan['cases'])==Counter({(n,k):1 for n in (32,64) for k in range(17)}),
         'Fixed model/scene inventory differs')
    if prepared:
        for row in plan['cases']:need(sha(row['file'])==row['sha256'],'Prepared augmented native inputs changed')
    return plan


def run(args,out,frozen):
    import torch,transformers
    from gnnformer.runtime import load_runtime,get_rope_index_fn
    from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
    from scripts.probe_native_vision_v2_prefix import fingerprint
    runtime,_,background=modules();torch.set_num_threads(4);start=time.perf_counter();plan=verify_plan(args.plan)
    need(torch.cuda.device_count()==1 and args.model_index in range(4),'One GPU and registered selected-model index required')
    selected=plan['selected_models'][args.model_index];key=selected['key'];checkpoint=selected['selected']['checkpoint']
    need(sha(checkpoint)==selected['selected']['checkpoint_sha256'],'Selected checkpoint changed')
    load_start=time.perf_counter();native=load_runtime(str(MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda')
    model=native.model.eval().requires_grad_(False)
    need(background.feature.model_metadata()==plan['model'] and background.feature.runtime_identity()==plan['runtime']
         and fingerprint(native.processor,str(transformers.__version__))==plan['processor'],'Live model/runtime/processor changed')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    need(saved['config']==selected['config'] and saved['step']==selected['selected']['step'],'Selected checkpoint metadata differs')
    branch=ParallelLocalAggregation().eval().requires_grad_(False);branch.load_state_dict(saved['branch'],strict=True);branch.to(model.device)
    identity={name:runtime.native.tensor_info(t) for name,t in branch.state_dict().items()}
    need(object_sha(identity)==selected['selected']['parameter_sha256'],'Restored selected core parameter bytes differ')
    runtime.native.native_contract(model,branch);torch.cuda.synchronize();load_seconds=time.perf_counter()-load_start
    data=DATA/out.name;data.mkdir(parents=True,exist_ok=False)
    config=dict(schema_version=1,plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),source_sha256=frozen,
        policy=POLICY,model_index=args.model_index,core_key=key,selected_model=selected,checkpoint_sha256=sha(checkpoint),
        model=plan['model'],runtime=plan['runtime'],processor=plan['processor'],
        hardware=dict(name=torch.cuda.get_device_name(0),cuda=torch.version.cuda),model_load_seconds=load_seconds,
        slurm_job_id=os.environ['SLURM_JOB_ID'])
    save(out/'config.json',config);records=[];totals=dict(model=0,visual=0,language=0,fusion=0,broadcast=0)
    versions={name:p._version for name,p in model.named_parameters()}
    branch_versions={name:p._version for name,p in branch.named_parameters()}
    for case_index,case in enumerate(plan['cases']):
        need(sha(case['file'])==case['sha256'],'Prepared scene file changed')
        bundle=torch.load(case['file'],map_location='cpu',weights_only=True)
        need(bundle['metadata']==case['metadata'] and runtime.validate_bundle(bundle)['passed'],'Augmented scene metadata changed')
        need(runtime.native.audit_layout(get_rope_index_fn(model),bundle)['metadata']==case['layout'],'Native input/mRoPE identity differs')
        sample=case['record']
        for mode in MODES:
            index=len(records);result=runtime.generate_native(model,native.processor,branch,bundle,mode=mode,core_key=key,max_new_tokens=4,capture=False)
            raw=result['raw_logits'].detach().cpu()
            need(raw.dtype==torch.float32 and torch.equal(raw,raw.half().float()) and raw.shape[0]==len(result['generated_ids']) and bool(torch.isfinite(raw).all()),'Raw generation logits must be exact FP32 promotion of native FP16 values')
            raw_path=data/f'raw_{index:03d}.pt';torch.save(dict(schema_version=1,raw_logits=raw.float()),raw_path)
            scored=score_ids(native.processor.tokenizer,result['generated_ids'],sample['gold'])
            need(all(result[k]==scored[k] for k in ('raw_text','text','completed','truncated','finish_reason')),'Runtime text/stop score differs')
            record=dict(index=index,core_key=key,case_index=case_index,mode=mode,sid=sample['sid'],pair_id=sample['pair_id'],
                gold=sample['gold'],n_frames=sample['n_frames'],content_sha256=sample['content_sha256'],**scored,
                raw_file=str(raw_path),raw_sha256=sha(raw_path),raw_dtype='torch.float32',metadata=result['metadata'],
                counters=result['counters'],model_seconds=result['model_seconds'])
            save(out/f'record_{index:03d}.json',record);records.append(record)
            for name in totals:totals[name]+=record['counters'][name]
            atomic_json(out/'progress.json',dict(completed=False,records=len(records),expected102=102,core_key=key,counters=totals,
                last_record_file=str(out/f'record_{index:03d}.json'),elapsed_seconds=time.perf_counter()-start))
            if len(records)%6==0:print(json.dumps(dict(core_key=key,completed=len(records),expected=102)),flush=True)
        del bundle
    need(len(records)==102 and totals['visual']==102 and 102<=totals['model']<=408,'Study generation coverage/cap differs')
    need(versions=={name:p._version for name,p in model.named_parameters()} and branch_versions=={name:p._version for name,p in branch.named_parameters()}
         and not any(p.requires_grad or p.grad is not None for p in model.parameters()) and not any(p.requires_grad or p.grad is not None for p in branch.parameters()),
         'Frozen model or selected branch changed')
    need(sha(checkpoint)==config['checkpoint_sha256'] and sources()==frozen,'Checkpoint/source changed during study')
    save(out/'predictions.json',records)
    summary=dict(config,passed=True,completed=True,computational_integrity_passed=True,no_fit=True,records=102,counters=totals,
        predictions_file=str(out/'predictions.json'),predictions_sha256=sha(out/'predictions.json'),
        peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(),peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(),
        generation_seconds=sum(r['model_seconds'] for r in records),elapsed_seconds=time.perf_counter()-start)
    save(out/'summary.json',summary);atomic_json(out/'progress.json',dict(completed=True,records=102,summary_file=str(out/'summary.json'),counters=totals))


def verify_software(path,bindings):
    _,software,_=modules();path=Path(path).resolve();summary=software.verify_profile(path)
    need(summary==read(path) and summary['protocol']=='v12_native_reference_software_only'
         and all(summary[k] is True for k in ('completed','passed','computational_integrity_passed','zero_identity_passed',
             'fixed_prefix_native_kv_unchanged','fixed_prefix_hidden_unchanged','native_head_replay_passed')),
         'Require completed matching V12 GPU software gates, not CPU-only validation')
    plan=software.verify(Path(summary['plan_file']))
    need(summary['source_sha256']==plan['source_sha256'] and all(summary[k]==plan[k] for k in ('model','runtime','processor','native_api')),
         'V12 software summary/native plan differs')
    bind(path,bindings);bind(summary['plan_file'],bindings,summary['plan_sha256'])
    timing=summary['timing'];need([r['actual_n_frames'] for r in timing['rows']]==[16,64],'Augmented native timing cases differ')
    times={}
    for row in timing['rows']:
        steps=row['generated_tokens'];need(1<=steps<=4 and row['preprocessing_seconds']>0 and row['generation_seconds']>0,
            'Natural timing duration/token count differs')
        bound=row['preprocessing_seconds']+row['generation_seconds']*4/steps
        need(math.isclose(bound,row['four_token_seconds_bound'],rel_tol=1e-12),'Augmented timing four-token projection differs')
        times[str(row['actual_n_frames'])]=bound
    need(math.isclose(times['16'],timing['T16'],rel_tol=1e-12) and math.isclose(times['64'],timing['T64'],rel_tol=1e-12),
         'Augmented native timing summary differs')
    return dict(file=str(path),sha256=sha(path),plan_file=summary['plan_file'],plan_sha256=summary['plan_sha256'],
        source_sha256=summary['source_sha256'],load_seconds=summary['load_seconds'],T16=times['16'],T64=times['64'],
        augmented_rows=True,completed=True,passed=True),plan


def study_projection(gate):
    load=gate['load_seconds'];bound=gate['T64']
    need(all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in (load,bound)),'Invalid augmented resource timing')
    projected=load+1.25*102*bound+30
    return dict(passed=projected<=1200,load_seconds=load,T64=bound,generations=102,all_lengths_charged_N64=True,
        projected_seconds_per_model=projected,per_model_cap=1200,four_max_allocations=4800,study_cap=5100)


def allocation_ledger():
    """Allocation rows only; completed failures consume the same GPU budget."""
    import subprocess
    result=subprocess.run(['sacct','-X','-u',os.environ['USER'],'--name=v12_ref_study','--starttime=2026-09-01',
        '--parsable2','--noheader','--format=JobIDRaw,State,ElapsedRaw,AllocTRES'],text=True,capture_output=True,check=True)
    rows=[];seen=set()
    for line in result.stdout.splitlines():
        fields=line.split('|');need(len(fields)>=4,'Malformed Slurm allocation row')
        job,state,elapsed,tres=fields[:4]
        values=dict(x.split('=',1) for x in tres.split(',') if '=' in x);gpus=int(values.get('gres/gpu','0'))
        if not gpus:continue
        need(job not in seen and gpus==1 and elapsed.isdigit(),'Duplicate or unexpected GPU allocation');seen.add(job)
        need(state.split()[0] not in ('RUNNING','PENDING','CONFIGURING','COMPLETING','SUSPENDED'),
             'Study allocation still active; its full cost cannot yet be audited')
        rows.append(dict(job_id=job,state=state,elapsed_seconds=int(elapsed),gpus=gpus,gpu_seconds=int(elapsed)*gpus))
    return dict(job_name='v12_ref_study',jobs=rows,allocated_gpu_seconds=sum(r['gpu_seconds'] for r in rows),
                allocation_rows_only=True,failed_allocations_included=True,software_jobs_excluded=True)


def audit_output(torch,tokenizer,record,raw,case,selected):
    runtime,_,_=modules();sample=case['record'];mode=record['mode'];ids=record['generated_ids']
    need(record['core_key']==selected['key'] and mode in MODES and all(record[k]==sample[k] for k in ('sid','pair_id','gold','n_frames','content_sha256')),
         'Output sample/model/mode identity differs')
    need(record['raw_dtype']=='torch.float32' and raw.dtype==torch.float32 and raw.ndim==2 and raw.shape[0]==len(ids)
         and bool(torch.isfinite(raw).all()) and torch.equal(raw,raw.half().float()),'Raw vectors are not losslessly promoted native FP16 logits')
    need(raw.argmax(-1).tolist()==ids,'Generated IDs differ from raw native argmax')
    score=score_ids(tokenizer,ids,sample['gold']);need(all(record[k]==v for k,v in score.items()),'Independent full-answer/EOS rescoring differs')
    meta=record['metadata'];base=case['metadata']
    need(all(meta[k]==v for k,v in base.items()) and meta['reference_mode']==mode and meta['core_key']==selected['key']
         and meta['read_boundary']=='before_native_final_norm' and meta['native_dtype']=='torch.float16' and meta['branch_dtype']=='torch.float32',
         'Generation changed frozen augmented inputs or reference mode')
    n=sample['n_frames'];width=base['prompt_width'];rows=n+25
    need(meta['row_count']==rows and meta['global_row']==rows-1 and meta['actual_n_frames']==n
         and meta['reference_rows']==24 and meta['processed_image_rows']==n+24
         and meta['row_kinds']==['actual']*n+['reference']*24+['global'] and meta['prefix_ids']==[],
         'Actual/reference/global rows or pre-answer prefix differ')
    need(meta['layout']==case['layout'] and meta['input_identity']['input_ids']['shape']==[rows,width]
         and meta['input_identity']['image_grid_thw']['shape']==[n+24,3]
         and meta['layout']['every_unpadded_row_exact'] is True,'Native input/position layout differs')
    policy=meta['generation']
    need(policy['max_new_tokens']==4 and policy['do_sample'] is False and policy['num_beams']==1
         and policy['repetition_penalty']==1 and policy['native_eos_token_ids']==POLICY['native_eos']
         and policy['target_eos_token_id']==151645 and policy['vocabulary_mask'] is False
         and policy['other_logits_processors'] is False,'Unrestricted native generation policy differs')
    steps=len(ids);expected=dict(model=steps,visual=1,language=steps,fusion=steps,broadcast=steps)
    need(record['counters']==expected and len(meta['generation_position_ids'])==steps
         and all(p['shape']==[4,rows,width if t==0 else 1] for t,p in enumerate(meta['generation_position_ids'])),
         'Native cached generation calls/positions differ')
    need(math.isfinite(record['model_seconds']) and record['model_seconds']>0,'Invalid recorded inference timing')
    target=tokenizer(str(sample['gold']),add_special_tokens=False)['input_ids'][0]
    return dict(record,first_token_nll=float(torch.logsumexp(raw[0],-1)-raw[0,target]))


def cell_metrics(rows):
    n=len(rows);need(n>0,'Empty declared reporting cell')
    parsed=[r for r in rows if r['prediction'] is not None]
    return dict(n=n,correct=sum(r['exact'] for r in rows),accuracy=sum(r['exact'] for r in rows)/n,
        parseable=len(parsed),completed=sum(r['completed'] for r in rows),truncated=sum(r['truncated'] for r in rows),
        malformed=sum(not r['parseable'] for r in rows),first_token_correct=sum(r['first_token_correct'] for r in rows),
        first_correct_whole_wrong=sum(r['correct_first_but_wrong_whole'] for r in rows),
        parsed_mae=None if not parsed else sum(abs(r['prediction']-r['gold']) for r in parsed)/len(parsed),
        parsed_bias=None if not parsed else sum(r['prediction']-r['gold'] for r in parsed)/len(parsed),
        parsed_error_denominator=len(parsed),first_token_nll=sum(r['first_token_nll'] for r in rows)/n,
        model_seconds=sum(r['model_seconds'] for r in rows),generated_tokens=sum(len(r['generated_ids']) for r in rows))


def report(args,out,frozen):
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(4);plan=verify_plan(args.plan);tests=self_test()
    need(len(args.report)==len({str(Path(p).resolve()) for p in args.report})==4,'Require all four distinct completed model runs')
    tokenizer=AutoTokenizer.from_pretrained(str(MODEL),use_fast=False,local_files_only=True)
    model_config=read(MODEL/'config.json');vocab=model_config.get('text_config',model_config)['vocab_size']
    allrows=[];runs={};job_ids=set()
    for value in args.report:
        directory=Path(value).resolve();need(directory.is_relative_to(OUT),'Run outside canonical V12 study outputs')
        summary=read(directory/'summary.json');config=read(directory/'config.json');key=summary['core_key']
        need(key in CORE_KEYS and key not in runs and summary['passed'] is True and summary['completed'] is True
             and summary['computational_integrity_passed'] is True and summary['no_fit'] is True
             and all(summary[k]==v for k,v in config.items()) and summary['policy']==POLICY
             and summary['plan_sha256']==sha(args.plan) and summary['source_sha256']==frozen,'Incomplete or mismatched study run')
        check_sources(summary['source_sha256'],directory)
        selected=plan['selected_models'][CORE_KEYS.index(key)]
        need(summary['selected_model']==selected and summary['checkpoint_sha256']==selected['selected']['checkpoint_sha256']
             and sha(selected['selected']['checkpoint'])==summary['checkpoint_sha256'],'Frozen selected model changed')
        need(sha(summary['predictions_file'])==summary['predictions_sha256'],'Prediction records changed')
        records=read(summary['predictions_file']);need(len(records)==summary['records']==102,'Missing whole-answer traces')
        rescored=[];counts=Counter()
        for index,row in enumerate(records):
            case_index=index//3;case=plan['cases'][case_index]
            need(row['index']==index and row['case_index']==case_index and row['mode']==MODES[index%3]
                 and read(directory/f'record_{index:03d}.json')==row,'Fixed generation order or durable record differs')
            need(sha(row['raw_file'])==row['raw_sha256'],'Raw native output changed')
            raw=torch.load(row['raw_file'],map_location='cpu',weights_only=True)
            need(raw['schema_version']==1 and raw['raw_logits'].shape[1]==vocab,'Native vocabulary archive differs')
            checked=audit_output(torch,tokenizer,row,raw['raw_logits'],case,selected);rescored.append(checked)
            counts.update(row['counters']);del raw
        need(dict(counts)==summary['counters'] and counts['visual']==102 and 102<=counts['model']<=408
             and math.isclose(sum(r['model_seconds'] for r in rescored),summary['generation_seconds'],rel_tol=1e-12),
             'Aggregate raw generation count/time differs')
        job_ids.add(str(summary['slurm_job_id']));allrows.extend(rescored)
        runs[key]=dict(directory=str(directory),summary_sha256=sha(directory/'summary.json'),config_sha256=sha(directory/'config.json'),
            selected_model=selected,counters=dict(counts),generation_seconds=summary['generation_seconds'],
            model_load_seconds=summary['model_load_seconds'],elapsed_seconds=summary['elapsed_seconds'],
            peak_memory_allocated_bytes=summary['peak_memory_allocated_bytes'],peak_memory_reserved_bytes=summary['peak_memory_reserved_bytes'])
    need(set(runs)==set(CORE_KEYS) and len(allrows)==408,'Model/generation study coverage differs')
    cells={};by_k={};comparisons={}
    for key in CORE_KEYS:
        for n in (32,64):
            for mode in MODES:
                group=[r for r in allrows if r['core_key']==key and r['n_frames']==n and r['mode']==mode]
                need(len(group)==17 and {r['gold'] for r in group}==set(range(17)),'Whole-answer count support differs')
                label=f'{key}/N{n}/{mode}'
                cells[label]={scope:cell_metrics([r for r in group if r['gold'] in ks]) for scope,ks in
                              (('all',range(17)),('K0_9',range(10)),('K10_16',range(10,17)))}
                for r in group:by_k[label+f'/K{r["gold"]}']=cell_metrics([r])
            base=cells[f'{key}/N{n}/base']['all']['correct'];back=cells[f'{key}/N{n}/background']['all']['correct'];sham=cells[f'{key}/N{n}/sham']['all']['correct']
            comparisons[f'{key}/N{n}']=dict(n=17,background_minus_base_correct=back-base,background_minus_sham_correct=back-sham,
                background_minus_base_pp=100*(back-base)/17,background_minus_sham_pp=100*(back-sham)/17)
    allocation=allocation_ledger()
    need(job_ids<={r['job_id'] for r in allocation['jobs']} and allocation['allocated_gpu_seconds']<=5100,
         'Study completed IDs or total allocated GPU budget differs')
    save(out/'rescored_predictions.json',allrows)
    analysis=dict(schema_version=1,passed=True,completed=True,audit_passed=True,source_sha256=frozen,policy=POLICY,
        plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),software_gate=plan['software_gate'],runs=runs,cells=cells,by_k=by_k,
        comparisons=comparisons,allocation=allocation,self_tests=tests,records=408,
        rescored_file=str(out/'rescored_predictions.json'),rescored_sha256=sha(out/'rescored_predictions.json'),
        no_fit=True,no_model_selection=True,exploratory_reused_test=True,no_practical_milestone_claim=True,
        limitations=['Only17 previously inspected families; this is an exploratory diagnostic, not fresh confirmation.',
          'Every mode uses N+25 native streams; added reference compute is shared, not free.',
          'The bank uses training K0 labels and a fixedN16 anchor; general neutral evidence selection is unresolved.',
          'Raw complete numeral+EOS is primary here; first-token correctness does not distinguish10..16.',
          'All four selected models and malformed/truncated outputs remain included.',
          'This prospectively rejects every nonterminal tokenizer special token, including tokens hidden by skip_special_tokens decoding.',
          'This does not establish a new attention operator, reasoning composition, or a general exact counting algorithm.'])
    save(out/'analysis.json',analysis)
    lines=['# V12 exploratory whole-answer reference study','',
        'Independent audit passed. Complete numeral plus native EOS within four tokens; all408 generations retained.','',
        '| Model | N | Mode | Whole answer | First token | Malformed | Truncated |','|---|---:|---|---:|---:|---:|---:|']
    for key in CORE_KEYS:
        for n in (32,64):
            for mode in MODES:
                v=cells[f'{key}/N{n}/{mode}']['all'];lines.append(f'| {key} | {n} | {mode} | {v["correct"]}/17 | {v["first_token_correct"]}/17 | {v["malformed"]}/17 | {v["truncated"]}/17 |')
    lines+=['','These are17 reused families, all four frozen V10 selected models, and matched augmented compute. Training-derived K0 references and the N16 anchor are explicit assumptions. No practical research milestone is certified by this diagnostic.','',
        '[All K cells, raw rescoring, sources and cost](analysis.json).']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    save(out/'summary.json',dict(passed=True,completed=True,audit_passed=True,source_sha256=frozen,analysis_file=str(out/'analysis.json'),
        analysis_sha256=sha(out/'analysis.json'),records=408,allocated_gpu_seconds=allocation['allocated_gpu_seconds'],no_practical_milestone_claim=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true');mode.add_argument('--run',action='store_true');mode.add_argument('--report',type=Path,nargs=4)
    mode.add_argument('--self-test',action='store_true');parser.add_argument('--software-summary',type=Path)
    parser.add_argument('--plan',type=Path);parser.add_argument('--model-index',type=int);args=parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID'),'All study work requires Slurm')
    if args.run:need(os.environ.get('SLURM_JOB_PARTITION')=='gpu' and os.environ.get('SLURM_JOB_GPUS'),'Study inference requires one Slurm GPU')
    else:need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),'Study staging/report/tests require CPU Slurm')
    label='check' if args.check else 'run' if args.run else 'report' if args.report else 'selftest'
    out=OUT/f'{label}_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False);frozen=snapshot(out)
    if args.check:
        need(args.software_summary is not None,'Completed exact V12 GPU software summary required');check(args,out,frozen)
    elif args.run:
        need(args.plan is not None and args.model_index is not None,'Frozen CPU study plan and model index required');args.plan=args.plan.resolve();run(args,out,frozen)
    elif args.report:
        need(args.plan is not None,'Exact common study plan required');args.plan=args.plan.resolve();report(args,out,frozen)
    else:save(out/'summary.json',dict(passed=True,tests=self_test(),source_sha256=frozen))
    need(sources()==frozen,'Study source changed during execution')
    print(json.dumps(dict(directory=str(out),summary_file=str(out/'summary.json'))),flush=True)


if __name__=='__main__':main()

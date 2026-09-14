"""V15 exploratory native learned-versus-empirical null comparison.

All four fixed-final V14 cores/predictors generate both mean sources on the
same34 deterministically selected V14 scenes. Every mode computes both means
in the same N+25 native stream layout. No fitting, selection, anchor correction,
new efficacy criterion or causal claim about unknown population means.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from datetime import datetime
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
OUT=REPO/'outputs/native_aggregation_vlm/v15/null_comparison_study'
DATA=Path('/mnt/data/gabriele/gnn_transformer/v15_null_comparison_study')
MODEL=Path('/mnt/ckpts/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5')
FRESH=Path('/mnt/data/gabriele/gnn_transformer/v14_fresh/main_manifest.json')
FINAL=REPO/'outputs/native_aggregation_vlm/v14/finalization/final_442626/final_acceptance.json'
REPORT=REPO/'outputs/native_aggregation_vlm/v14/report_442582/analysis.json'
MODES=('learned','bank')
CORE_KEYS=('centered_s18','centered_s19','offset_s18','offset_s19')
JOB_CAPS={'v15_null_profile':300,'v15_null_study':900}
POLICY=dict(protocol='v15_fixed_v14_native_null_comparison_exploratory',modes=list(MODES),core_keys=list(CORE_KEYS),
    scenes=34,families=17,counts=list(range(17)),lengths=[32,64],reference_occurrences=24,
    correction_coefficients={'centered':'actual_scene_N','offset':1},anchor=None,
    reference_mean='FP64 occurrence mean followed by FP32',
    reference_rule='Exact V14 training bank: N8K0 then N16K0 for the question, retaining all24 image occurrences',
    native_rows='N actual image streams +24 training-reference image streams +one text-only global stream',
    both_means_computed_in_both_modes=True,capture_all_generated_queries=True,max_new_tokens=4,native_eos=[151645,151643],target_eos=151645,
    generations_per_model=68,maximum_model_forwards_per_model=272,visual_forwards_per_model=68,
    total_generations=272,maximum_model_forwards=1088,total_visual_forwards=272,
    per_model_gpu_seconds_cap=900,campaign_gpu_seconds_cap=4200,software_gpu_seconds_cap=600,
    software_per_job_gpu_seconds_cap=300,maximum_concurrent_gpus=4,
    projection='software_load +1.25*68*max_both_modes_augmented_N64_four_token_time +120',
    scene_order='ascending K, lexicographic lowest pair_id for K, N32 then N64; learned then bank within each scene',
    primary_measure='Complete stripped ASCII integer equals gold AND native EOS within four tokens; no nonterminal special tokens',
    no_fit=True,no_selection=True,reused_test_exploratory=True,no_practical_milestone_claim=True)
OWN=('scripts/evaluate_native_vision_v15_null_comparison.py',
     'slurm/native_vision_v15_null_comparison_study_check.sbatch','slurm/native_vision_v15_null_comparison_study.sbatch')


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
    with Path(path).open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)



def atomic_json(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)



def modules():
    # Native/Torch imports occur only after the CLI Slurm guard.
    return (importlib.import_module('scripts.native_vision_null_comparison_runtime'),
            importlib.import_module('scripts.profile_native_vision_v15_null_comparison'),
            importlib.import_module('scripts.probe_native_vision_v10_background'),
            importlib.import_module('scripts.train_native_vision_v14'),
            importlib.import_module('scripts.audit_native_vision_v14_checkpoint'),
            importlib.import_module('scripts.finalize_native_vision_v14'))


def sources():
    runtime,software,background,training,auditor,finalizer=modules()
    ancestors={**training.sources(),**auditor.sources(),**finalizer.sources(),**background.sources()}
    names=tuple(dict.fromkeys((*OWN,*software.OWN,*runtime.OWN)))
    return {**ancestors,**{name:sha(REPO/name) for name in names}}


def snapshot(directory):
    (directory/'code').mkdir();frozen=sources()
    for name,digest in frozen.items():
        path=directory/'code'/name.replace('/','_');path.write_bytes((REPO/name).read_bytes());need(sha(path)==digest,'Source changed during snapshot')
    save(directory/'source_hashes.json',frozen)
    (directory/'INDEX.md').write_text('# V15 learned versus empirical null study\n\n[Summary](summary.json) · [Sources](source_hashes.json) · [Progress](progress.json).\n')
    return frozen



def bind(path,bindings,expected=None):
    path=Path(path).resolve();digest=sha(path)
    need(expected is None or digest==expected,'Bound artifact changed: '+str(path))
    need(str(path) not in bindings or bindings[str(path)]==digest,'Conflicting artifact digest')
    bindings[str(path)]=digest
    return dict(file=str(path),sha256=digest)



def check_sources(value,directory):
    need(value==sources(),'Frozen V15 study source changed')
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
    selection_tests=selection_self_test();allocation_tests=allocation_self_test();capture_tests=capture_self_test();prefix_tests=prefix_self_test()
    return dict(passed=True,selection_tests=selection_tests,allocation_tests=allocation_tests,capture_tests=capture_tests,prefix_tests=prefix_tests,tests=['full_integer_EOS','reject_shared_first_digit_proxy','malformed_and_truncated_in_denominator','reject_interior_EOS','reject_nonterminal_special_tokens','strict_ASCII'])



def choose_records(manifest):
    need(set(manifest['splits'])=={'test_N32','test_N64'},'Unexpected V14 test cells')
    by=defaultdict(dict)
    for n in (32,64):
        rows=manifest['splits'][f'test_N{n}']['samples']
        need(len(rows)==136 and Counter(row['gold'] for row in rows)==Counter({k:8 for k in range(17)}),
             'Require complete136-family support before fixed subset selection')
        for row in rows:
            pid=row['pair_id'];need(row['anchor_id']==pid and row['n_frames']==n and row['split']=='test'
                and n not in by[pid],'Family/length/split identity differs')
            by[pid][n]=row
    need(len(by)==136 and all(set(x)=={32,64} for x in by.values()),'Incomplete length family')
    result=[]
    for k in range(17):
        choices=sorted(pid for pid,members in by.items() if members[32]['gold']==k)
        need(len(choices)==8,'Missing count families');members=by[choices[0]]
        need(members[32]['question']==members[64]['question'] and members[64]['gold']==k,'Family target changed')
        result.extend(members[n] for n in (32,64))
    need(len(result)==len({row['sid'] for row in result})==34,'Duplicate fixed selected scene')
    return result


def selection_self_test():
    import copy
    rows={n:[] for n in (32,64)}
    for n in rows:
        for k in range(17):
            for a in reversed(range(8)):
                pid=f'K{k:02d}_family{a}'
                rows[n].append(dict(sid=f'{pid}_N{n}',pair_id=pid,anchor_id=pid,n_frames=n,split='test',gold=k,question=f'q{k}'))
    fixture=dict(splits={f'test_N{n}':dict(samples=rows[n]) for n in rows})
    selected=choose_records(fixture)
    need([r['pair_id'] for r in selected]==[f'K{k:02d}_family0' for k in range(17) for _ in range(2)],'Selection depends on row order')
    changed=copy.deepcopy(fixture);changed['splits']['test_N64']['samples'][7]['question']='mismatched'
    try:choose_records(changed)
    except ValueError:pass
    else:raise AssertionError('Mismatched chosen family accepted')
    return dict(passed=True,tests=['all136_families_required','lexicographic_selection_without_outcomes','paired_question_identity'])


def verify_prior(torch,bindings):
    _,_,_,training,auditor,finalizer=modules()
    final=read(FINAL);bind(FINAL,bindings);final_summary=read(FINAL.parent/'summary.json');bind(FINAL.parent/'summary.json',bindings)
    need(final['completed'] is True and final['verification_passed'] is True and final_summary['passed'] is True
        and final_summary['final_acceptance_sha256']==sha(FINAL),'Completed V14 verification is required, independently of efficacy')
    for name,digest in final['source_sha256'].items():bind(REPO/name,bindings,digest)
    refs=final['artifacts']
    for item in refs.values():bind(item['path'],bindings,item['sha256'])
    need(Path(refs['analysis']['path']).resolve()==REPORT,'Wrong canonical completed V14 report')
    report=read(REPORT);summary_path=Path(refs['independent_report_summary']['path']);summary=read(summary_path)
    need(report['passed'] is True and report['audit_passed'] is True and summary['passed'] is True
        and Path(summary['analysis_file']).resolve()==REPORT and summary['analysis_sha256']==sha(REPORT),
        'V14 report completion/analysis binding differs')
    for name,digest in report['source_sha256'].items():bind(REPO/name,bindings,digest)
    post_path=Path(refs['selected_checkpoint_audit']['path']);post=read(post_path)
    need(post['completed'] is True and post['passed'] is True and post['computational_integrity_passed'] is True
        and post['native_replay_gate_passed'] is True and post['protocol']=='all_selected_v14_native_checkpoint_audit'
        and post['calls']==dict(model=46,visual=26,language=46,norm=46,last_block=46)
        and post['cache_full_rows']==656 and post['fixed_prefix_kv_checks']==24,'Completed all-four native audit required')
    bind(post['plan_file'],bindings,post['plan_sha256']);post_plan=read(post['plan_file'])
    need(Path(post['plan_file']).with_suffix('.sha256').read_text().strip()==post['plan_sha256']
        and post_plan['source_sha256']==post['source_sha256'],'Selected-audit plan/source differs')
    for name,digest in post['source_sha256'].items():bind(REPO/name,bindings,digest)
    need(finalizer.match_selected(report,post)==final['selected_models'],'Finalizer selected model identities differ')
    need(set(report['runs'])==set(CORE_KEYS),'All four fixed-final models required')
    selected=[]
    for key in CORE_KEYS:
        row=report['runs'][key];directory=Path(row['run_directory'])
        for filename,label in (('config.json','config_sha256'),('summary.json','summary_sha256')):bind(directory/filename,bindings,row[label])
        config=read(directory/'config.json');run=read(directory/'summary.json');selection=read(directory/'selection.json');bind(directory/'selection.json',bindings)
        chosen=selection['selected']
        need(config['condition']==row['condition'] and config['seed']==row['seed'] and config['arm']=='parallel'
            and key==f"{config['condition']}_s{config['seed']}" and config['policy']==training.POLICY
            and run['passed'] is True and run['completed'] is True and run['selected']==chosen
            and selection['rule']==training.POLICY['selection'] and selection['development']==[chosen]
            and selection['dev_descriptive_only'] is True and run['development']==[chosen]
            and chosen['step']==4590 and chosen['student_step']==8000,'Fixed-final phase/checkpoint contract differs')
        need(chosen['checkpoint']==row['selected_checkpoint'] and chosen['checkpoint_sha256']==row['selected_checkpoint_sha256']
            and chosen['parameter_sha256']==row['selected_parameter_sha256'],'Report selected checkpoint differs')
        bind(chosen['checkpoint'],bindings,chosen['checkpoint_sha256']);bind(chosen['dev_file'],bindings,chosen['dev_sha256'])
        bind(config['plan_file'],bindings,config['plan_sha256']);bind(config['cache_binding']['file'],bindings,config['cache_binding']['sha256'])
        for name,digest in config['source_sha256'].items():bind(REPO/name,bindings,digest)
        saved=torch.load(chosen['checkpoint'],map_location='cpu',weights_only=True)
        need(saved['config']==config and saved['step']==4590 and saved['student_step']==8000,'Actual checkpoint metadata differs')
        auditor.validate_state(torch,saved,chosen['parameter_sha256']);del saved
        test_path=directory/'test.json';bind(test_path,bindings,row['test_sha256']);original=read(test_path)
        need(original['n']==272 and len(original['rows'])==272 and len({r['sid'] for r in original['rows']})==272,
            'Original N+1 prediction inventory differs')
        bind(original['raw_file'],bindings,original['raw_sha256'])
        selected.append(dict(key=key,condition=config['condition'],seed=config['seed'],run_directory=str(directory),config=config,selected=chosen,
            original_test=dict(file=str(test_path),sha256=sha(test_path),raw_file=original['raw_file'],raw_sha256=original['raw_sha256'])))
    need(all(r['config']['cache_binding']==selected[0]['config']['cache_binding'] for r in selected),'Reference training-cache identities differ')
    fresh=report['data']['fresh_data'];need(Path(fresh['manifest_file']).resolve()==FRESH and fresh['manifest_sha256']==sha(FRESH)
        and fresh['passed'] is True and fresh['all_qa_image_semantic_extensions_verified'] is True,'V14 fresh manifest was not independently audited')
    for name,digest in report['data']['data_bindings'].items():bind(name,bindings,digest)
    prior=dict(finalization=bind(FINAL,bindings),report=bind(REPORT,bindings),selected_audit=bind(post_path,bindings),
        historical_efficacy_not_a_gate=True,fixed_core_step=4590,fixed_student_step=8000)
    return selected,prior


def audit_banks(runtime,selected,bindings):
    _,_,background,_,_,_=modules();cache_ref=selected[0]['config']['cache_binding'];cache=read(cache_ref['file'])
    need(cache['complete'] is True and cache['training_only'] is True and len(cache['null_banks'])==54,'Complete immutable training banks required')
    manifest=read(runtime.REFERENCE_MANIFEST);bind(runtime.REFERENCE_MANIFEST,bindings)
    train={r['sid']:r for n in (8,16) for r in manifest['splits'][f'train_N{n}']['samples']};banks={}
    for qid,old_bank in sorted(cache['null_banks'].items()):
        q=old_bank['question'];need(qid==object_sha(q),'Question identity differs')
        bank=runtime.reference_bank(q);expected=old_bank['occurrences']
        actual=[(r['source_sid'],r['source_n_frames'],r['source_frame_index'],r['sha256']) for r in bank['occurrences']]
        need(actual==[(r['source_sid'],r['source_n_frames'],r['source_frame_index'],r['image_sha256']) for r in expected]
            and len(actual)==24 and [r['n_frames'] for r in bank['scenes']]==[8,16],
            'Live bank changed the exact24 frozen training occurrences')
        for row in bank['scenes']:
            source=train[row['sid']];need(source['question']==q and source['gold']==0,'Non-null or wrong-question bank scene')
            background.verify_qa(source,bindings)
        for im in bank['occurrences']:bind(im['path'],bindings,im['sha256'])
        banks[q]=bank
    need(len(banks)==54 and sum(len(b['occurrences']) for b in banks.values())==1296,'Full question/occurrence bank coverage differs')
    return banks


def check(args,out,frozen):
    import torch,transformers
    from transformers import AutoProcessor
    from scripts.probe_native_vision_v2_prefix import fingerprint
    runtime,_,background,_,_,_=modules();torch.set_num_threads(4);started=time.perf_counter();bindings={};tests=self_test()
    selected,prior=verify_prior(torch,bindings);gate,software_plan=verify_software(args.software_summary,bindings)
    processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,use_fast=False)
    need(fingerprint(processor,str(transformers.__version__))==software_plan['processor'],'Current processor differs from software gate')
    for row in selected:
        need(all(row['config'][key]==software_plan[key] for key in ('model','runtime','processor'))
            and row['config']['native_dtypes']==dict(norm='torch.float16',lm_head='torch.float16'),'Selected native runtime/dtypes differ')
    banks=audit_banks(runtime,selected,bindings);records=choose_records(read(FRESH));bind(FRESH,bindings)
    data=DATA/out.name;data.mkdir(parents=True,exist_ok=False)
    (data/'INDEX.md').write_text('# V15 immutable augmented native inputs\n\n34 scenes, exact actual/reference/global rows.\n')
    from scripts.probe_native_vision_parallel_local import native_api
    owner,fn,api=native_api(processor);need(api==software_plan['native_api'],'Installed native source differs')
    prepared=[]
    for i,sample in enumerate(records):
        background.verify_qa(sample,bindings)
        for im in sample['image_files']:bind(im['path'],bindings,im['sha256'])
        bundle=runtime.prepare_scene(processor,sample,banks[sample['question']],verify_processor_parity=True)
        layout=runtime.native.audit_layout(lambda **kwargs:fn(owner,**kwargs),bundle)
        need(bundle['metadata']['prefix_ids']==[],'No answer prefix may be supplied')
        path=data/f'input_{i:03d}.pt';torch.save(bundle,path)
        prepared.append(dict(record=sample,file=str(path),sha256=sha(path),metadata=bundle['metadata'],layout=layout['metadata']))
        print(json.dumps(dict(prepared=i+1,total=34,sid=sample['sid'])),flush=True)
    targets={str(k):runtime.native.encode_target(processor.tokenizer,k) for k in range(17)}
    projection=study_projection(gate);allocation=allocation_ledger(out)
    previous={r['job_id']:r for r in allocation['jobs']}
    need(gate['software_job_id'] in previous and previous[gate['software_job_id']]['job_name']=='v15_null_profile'
        and previous[gate['software_job_id']]['state']=='COMPLETED' and previous[gate['software_job_id']]['gpus']==1 and previous[gate['software_job_id']]['exit_code']=='0:0','Successful software allocation is missing')
    reserve_passed=allocation['allocated_gpu_seconds']+3600<=4200
    plan=dict(schema_version=1,policy=POLICY,source_sha256=frozen,artifact_bindings=bindings,software_gate=gate,
        software_plan_file=gate['plan_file'],software_plan_sha256=gate['plan_sha256'],prior=prior,
        fresh_manifest=bind(FRESH,bindings),selected_models=selected,cases=prepared,reference_banks=banks,reference_occurrences_total=1296,
        model=software_plan['model'],runtime=software_plan['runtime'],processor=software_plan['processor'],native_api=software_plan['native_api'],
        native_dtypes=dict(norm='torch.float16',head='torch.float16'),target_token_ids=targets,
        projection=projection,prior_allocation=allocation,reserve_passed=reserve_passed,tests=tests,
        no_fit=True,no_model_selection=True,no_static_prefix_calibration=True,slurm_job_id=os.environ['SLURM_JOB_ID'])
    save(out/'plan.json',plan);(out/'plan.sha256').write_text(sha(out/'plan.json')+'\n')
    summary=dict(passed=projection['passed'] and reserve_passed,completed=True,no_model_loaded=True,plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),
        source_sha256=frozen,projection=projection,prior_allocation=allocation,reserve_passed=reserve_passed,cases=34,cores=4,tests=tests,seconds=time.perf_counter()-started)
    save(out/'summary.json',summary);need(sources()==frozen,'Source changed during preparation')
    need(summary['passed'],'Preserved timing/campaign gate failure: four900-second allocations must fit4200 total with all prior V15 GPU spend')


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
    need(plan['reserve_passed'] is True and [c['record'] for c in plan['cases']]==choose_records(read(FRESH)),
         'Frozen scene selection or allocation reserve differs')
    if prepared:
        for row in plan['cases']:need(sha(row['file'])==row['sha256'],'Prepared augmented native inputs changed')
    return plan



def run(args,out,frozen):
    import torch,transformers
    from gnnformer.runtime import load_runtime,get_rope_index_fn
    from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
    from gnnformer.conditional_null_mean import ConditionalNullMean
    from scripts.probe_native_vision_v2_prefix import fingerprint
    runtime,_,background,training,auditor,_=modules();torch.set_num_threads(4);start=time.perf_counter();plan=verify_plan(args.plan)
    need(torch.cuda.device_count()==1 and args.model_index in range(4),'One GPU and registered selected-model index required')
    selected=plan['selected_models'][args.model_index];key=selected['key'];checkpoint=selected['selected']['checkpoint']
    need(sha(checkpoint)==selected['selected']['checkpoint_sha256'],'Selected checkpoint changed')
    load_start=time.perf_counter();native=load_runtime(str(MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda')
    model=native.model.eval().requires_grad_(False)
    need(torch.cuda.get_device_name(0)==plan['software_gate']['hardware']['gpu'] and 'B200' in torch.cuda.get_device_name(0),
         'Study GPU differs from measured B200 software profile')
    need(background.feature.model_metadata()==plan['model'] and background.feature.runtime_identity()==plan['runtime']
         and fingerprint(native.processor,str(transformers.__version__))==plan['processor'],'Live model/runtime/processor changed')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    need(saved['config']==selected['config'] and saved['step']==4590 and saved['student_step']==8000,'Selected checkpoint metadata differs')
    branch=ParallelLocalAggregation().eval().requires_grad_(False);branch.load_state_dict(saved['branch'],strict=True);branch.to(model.device)
    predictor=ConditionalNullMean().eval().requires_grad_(False);predictor.load_state_dict(saved['predictor'],strict=True);predictor.to(model.device)
    auditor.validate_state(torch,saved,selected['selected']['parameter_sha256']);del saved
    identity=training.state_info(branch,predictor)
    need(object_sha(identity)==selected['selected']['parameter_sha256'],'Restored selected core/predictor bytes differ')
    runtime.native.native_contract(model,branch);torch.cuda.synchronize();load_seconds=time.perf_counter()-load_start
    data=DATA/out.name;data.mkdir(parents=True,exist_ok=False)
    (data/'INDEX.md').write_text('# V15 raw global native logits\n\nAll68 complete natural trajectories, no masking or discarded failures.\n')
    config=dict(schema_version=1,plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),source_sha256=frozen,
        policy=POLICY,model_index=args.model_index,core_key=key,selected_model=selected,checkpoint_sha256=sha(checkpoint),
        model=plan['model'],runtime=plan['runtime'],processor=plan['processor'],
        hardware=dict(name=torch.cuda.get_device_name(0),cuda=torch.version.cuda),model_load_seconds=load_seconds,
        slurm_job_id=os.environ['SLURM_JOB_ID'])
    save(out/'config.json',config);records=[];totals=dict(model=0,visual=0,language=0,fusion=0,broadcast=0)
    versions={name:p._version for name,p in model.named_parameters()}
    branch_versions={(kind,name):p._version for kind,module in [('core',branch),('predictor',predictor)] for name,p in module.named_parameters()}
    for case_index,case in enumerate(plan['cases']):
        need(sha(case['file'])==case['sha256'],'Prepared scene file changed')
        bundle=torch.load(case['file'],map_location='cpu',weights_only=True)
        need(bundle['metadata']==case['metadata'] and runtime.validate_bundle(bundle)['passed'],'Augmented scene metadata changed')
        need(runtime.native.audit_layout(get_rope_index_fn(model),bundle)['metadata']==case['layout'],'Native input/mRoPE identity differs')
        sample=case['record']
        for mode in MODES:
            index=len(records);result=runtime.generate_native(model,native.processor,branch,predictor,bundle,condition=selected['condition'],mean_source=mode,core_key=key,max_new_tokens=4,capture=True)
            raw=result['raw_logits'].detach().cpu()
            need(raw.dtype==torch.float32 and torch.equal(raw,raw.half().float()) and raw.shape[0]==len(result['generated_ids']) and bool(torch.isfinite(raw).all()),'Raw generation logits must be exact FP32 promotion of native FP16 values')
            raw_path=data/f'raw_{index:03d}.pt';torch.save(dict(schema_version=1,raw_logits=raw.float(),captures=result['captures']),raw_path)
            scored=score_ids(native.processor.tokenizer,result['generated_ids'],sample['gold'])
            need(all(result[k]==scored[k] for k in ('raw_text','text','completed','truncated','finish_reason')),'Runtime text/stop score differs')
            record=dict(index=index,core_key=key,condition=selected['condition'],seed=selected['seed'],case_index=case_index,mode=mode,sid=sample['sid'],pair_id=sample['pair_id'],
                gold=sample['gold'],n_frames=sample['n_frames'],content_sha256=sample['content_sha256'],**scored,
                raw_file=str(raw_path),raw_sha256=sha(raw_path),raw_dtype='torch.float32',metadata=result['metadata'],
                counters=result['counters'],model_seconds=result['model_seconds'],capture_identities=audit_captures(torch,result['captures'],result['generated_ids'],case,selected,mode))
            save(out/f'record_{index:03d}.json',record);records.append(record)
            for name in totals:totals[name]+=record['counters'][name]
            atomic_json(out/'progress.json',dict(completed=False,records=len(records),expected=68,core_key=key,counters=totals,
                last_record_file=str(out/f'record_{index:03d}.json'),elapsed_seconds=time.perf_counter()-start))
            if len(records)%6==0:print(json.dumps(dict(core_key=key,completed=len(records),expected=68)),flush=True)
        del bundle
    need(len(records)==68 and totals['visual']==68 and 68<=totals['model']<=272,'Study generation coverage/cap differs')
    need(versions=={name:p._version for name,p in model.named_parameters()} and branch_versions=={(kind,name):p._version for kind,module in [('core',branch),('predictor',predictor)] for name,p in module.named_parameters()}
         and not any(p.requires_grad or p.grad is not None for p in model.parameters()) and not any(p.requires_grad or p.grad is not None for module in (branch,predictor) for p in module.parameters())
         and training.state_info(branch,predictor)==identity,
         'Frozen model or selected branch changed')
    need(sha(checkpoint)==config['checkpoint_sha256'] and sources()==frozen,'Checkpoint/source changed during study')
    save(out/'predictions.json',records)
    summary=dict(config,passed=True,completed=True,computational_integrity_passed=True,no_fit=True,records=68,counters=totals,
        predictions_file=str(out/'predictions.json'),predictions_sha256=sha(out/'predictions.json'),
        peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(),peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(),
        generation_seconds=sum(r['model_seconds'] for r in records),elapsed_seconds=time.perf_counter()-start)
    save(out/'summary.json',summary);atomic_json(out/'progress.json',dict(completed=True,records=68,summary_file=str(out/'summary.json'),counters=totals))



def verify_software(path,bindings):
    _,software,_,_,_,_=modules();path=Path(path).resolve();summary=software.verify_profile(path)
    need(summary==read(path) and summary['protocol']=='v15_null_comparison_software_only'
        and all(summary[k] is True for k in ('completed','passed','computational_integrity_passed','zero_identity_passed',
            'fixed_prefix_native_kv_unchanged','fixed_prefix_hidden_unchanged','native_head_replay_passed',
            'frozen_backbone_unchanged','core_unfitted_unchanged','predictor_unfitted_unchanged','no_fit','no_efficacy_scoring')),
        'Require completed V15 GPU software gate, not CPU-only validation')
    plan=software.verify(Path(summary['plan_file']))
    need(summary['source_sha256']==plan['source_sha256'] and all(summary[k]==plan[k] for k in ('model','runtime','processor','native_api')),
        'V15 software/native plan differs')
    bind(path,bindings);bind(summary['plan_file'],bindings,summary['plan_sha256'])
    need(summary['hardware']['gpu']=='NVIDIA B200','Require registered B200 measured timing')
    timing=summary['timing'];need(set(timing['by_mode'])==set(MODES),'Both mean-source timing modes required')
    bound16=max(timing['by_mode'][mode]['16'] for mode in MODES);bound64=max(timing['by_mode'][mode]['64'] for mode in MODES)
    # verify_profile independently checks every observed token count and prep+gen*4/tokens formula.
    return dict(file=str(path),sha256=sha(path),plan_file=summary['plan_file'],plan_sha256=summary['plan_sha256'],
        source_sha256=summary['source_sha256'],load_seconds=summary['load_seconds'],T16=bound16,T64=bound64,
        by_mode=timing['by_mode'],hardware=summary['hardware'],software_job_id=str(summary['slurm_job_id']),
        augmented_rows=True,completed=True,passed=True),plan


def study_projection(gate):
    load=gate['load_seconds'];bound=gate['T64']
    need(all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and v>0 for v in (load,bound)),
        'Invalid measured augmented timing')
    projected=load+1.25*68*bound+120
    return dict(passed=projected<=900,load_seconds=load,T64=bound,generations=68,all_lengths_charged_N64=True,
        projected_seconds_per_model=projected,per_model_cap=900,four_max_allocations=3600,campaign_cap=4200)


def parse_allocations(raw):
    rows=[];seen=set();events=[]
    terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED','SPECIAL_EXIT'}
    for line in raw.splitlines():
        if not line.strip():continue
        fields=line.split('|');need(len(fields)==9,'Malformed Slurm allocation row')
        job,name,partition,state,exit_code,elapsed,tres,start,end=fields
        if not name.startswith('v15_') or partition!='gpu':continue
        need(name in JOB_CAPS and job not in seen and elapsed.isdigit(),'Unknown/duplicate V15 GPU job');seen.add(job)
        need(state.split()[0] in terminal and re.fullmatch(r'[0-9]+:[0-9]+',exit_code),'All V15 GPU jobs must have recognized terminal states/exit codes')
        values=dict(x.split('=',1) for x in tres.split(',') if '=' in x)
        typed=[int(v) for k,v in values.items() if k.startswith('gres/gpu:')]
        gpus=int(values['gres/gpu']) if 'gres/gpu' in values else sum(typed)
        need(gpus in (0,1) and (not typed or sum(typed)==gpus),'Unexpected GPU allocation multiplicity')
        seconds=int(elapsed);need(seconds<=JOB_CAPS[name],'V15 job exceeded registered per-job cap')
        if gpus and seconds:
            a=datetime.fromisoformat(start);b=datetime.fromisoformat(end);need(b>=a,'Scheduler interval reversed')
            events.extend([(a,1),(b,-1)])
        rows.append(dict(job_id=job,job_name=name,state=state,exit_code=exit_code,elapsed_seconds=seconds,gpus=gpus,gpu_seconds=seconds*gpus,start=start,end=end))
    current=maximum=0
    for _,change in sorted(events,key=lambda x:(x[0],x[1])):current+=change;maximum=max(maximum,current)
    need(current==0 and maximum<=4,'V15 maximum simultaneous GPU allocation exceeded')
    software=sum(r['gpu_seconds'] for r in rows if r['job_name']=='v15_null_profile')
    total=sum(r['gpu_seconds'] for r in rows);need(software<=600 and total<=4200,'V15 software/campaign GPU cap exceeded')
    return dict(jobs=rows,allocated_gpu_seconds=total,software_gpu_seconds=software,maximum_concurrent_gpus=maximum,
        allocation_rows_only=True,failed_allocations_included=True,zero_allocation_attempts_retained=True,all_user_v15_prefix_jobs_discovered=True)


def allocation_ledger(out):
    import subprocess
    command=['sacct','-X','-u',os.environ['USER'],'--starttime=2026-09-01','--parsable2','--noheader',
        '--format=JobIDRaw,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,Start,End']
    result=subprocess.run(command,text=True,capture_output=True,check=True)
    path=Path(out)/'all_user_sacct.psv';path.write_text(result.stdout)
    return dict(parse_allocations(result.stdout),command=command,raw_file=str(path),raw_sha256=sha(path))


def allocation_self_test():
    raw='1|v15_null_profile|gpu|FAILED|1:0|3|gres/gpu=1|2026-09-11T00:00:00|2026-09-11T00:00:03\n2|v15_null_profile|gpu|COMPLETED|0:0|20|gres/gpu=1|2026-09-11T00:00:03|2026-09-11T00:00:23'
    value=parse_allocations(raw);need(value['allocated_gpu_seconds']==23 and value['software_gpu_seconds']==23,'Failed allocation omitted')
    for bad in (raw+'\n'+raw,raw.replace('FAILED','RUNNING'),raw.replace('v15_null_profile','v15_unknown')):
        try:parse_allocations(bad)
        except ValueError:pass
        else:raise AssertionError('Invalid allocation ledger accepted')
    typed=raw.replace('gres/gpu=1','gres/gpu:b200=1');need(parse_allocations(typed)['allocated_gpu_seconds']==23,'Typed GPU allocation omitted')
    zero=raw+'\n3|v15_null_study|gpu|FAILED|1:0|0||Unknown|Unknown';need(len(parse_allocations(zero)['jobs'])==3,'Zero-allocation failed attempt omitted')
    return dict(passed=True,tests=['include_failed_GPU_seconds','reject_duplicates_active_unknown_roles','typed_GPU_TRES','retain_zero_allocation_attempt'])


def audit_captures(torch,captures,ids,case,selected,mode):
    runtime,_,_,_,_,_=modules();n=case['record']['n_frames'];width=case['metadata']['prompt_width']
    need(isinstance(captures,list) and len(captures)==len(ids),'Every generated query must retain its capture')
    shapes=dict(actual_states=(n,1,3584),reference_states=(24,1,3584),global_states=(1,3584),native_global=(1,3584),fused_global=(1,3584),
        actual_messages=(n,1,96),reference_messages=(24,1,96),aggregate=(1,96),query=(1,96),predicted_mean=(1,96),bank_mean=(1,96),used_mean=(1,96),
        preactivation=(1,96),projected_null=(1,96),corrected_preactivation=(1,96),delta=(1,3584))
    native_fields={'actual_states','reference_states','global_states','native_global','fused_global'};identities=[]
    for t,capture in enumerate(captures):
        need(capture['query_indices']==[width-1 if t==0 else 0] and capture['stream_positions']==[width+t-1]
            and capture['n_actual_rows']==n and capture['n_reference_rows']==24 and capture['condition']==selected['condition']
            and capture['mean_source']==mode and capture['coefficient']==(n if selected['condition']=='centered' else 1),
            'Captured physical/stream query or mean source differs')
        for key,shape in shapes.items():
            value=capture[key];dtype=torch.float16 if key in native_fields else torch.float32
            need(isinstance(value,torch.Tensor) and tuple(value.shape)==shape and value.dtype==dtype and value.device.type=='cpu'
                and not value.requires_grad and bool(torch.isfinite(value).all()),'Invalid captured query tensor: '+key)
        need(torch.equal(capture['global_states'],capture['native_global'])
            and torch.equal(capture['used_mean'],capture['predicted_mean' if mode=='learned' else 'bank_mean'])
            and torch.equal(capture['fused_global'],capture['native_global']+capture['delta'].half()),
            'Captured selected mean/native cast-add identity differs')
        identities.append(dict(position=t,observed_prefix_ids=ids[:t],
            tensors={key:runtime.native.tensor_info(capture[key]) for key in shapes},
            metadata={key:value for key,value in capture.items() if not isinstance(value,torch.Tensor)}))
    return identities


def audit_output(torch,tokenizer,record,raw,case,selected):
    runtime,_,_,_,_,_=modules();sample=case['record'];mode=record['mode'];ids=record['generated_ids']
    need(record['core_key']==selected['key'] and record['condition']==selected['condition'] and record['seed']==selected['seed'] and mode in MODES and all(record[k]==sample[k] for k in ('sid','pair_id','gold','n_frames','content_sha256')),
         'Output sample/model/mode identity differs')
    need(record['raw_dtype']=='torch.float32' and raw.dtype==torch.float32 and raw.ndim==2 and raw.shape[0]==len(ids)
         and bool(torch.isfinite(raw).all()) and torch.equal(raw,raw.half().float()),'Raw vectors are not losslessly promoted native FP16 logits')
    need(raw.argmax(-1).tolist()==ids,'Generated IDs differ from raw native argmax')
    score=score_ids(tokenizer,ids,sample['gold']);need(all(record[k]==v for k,v in score.items()),'Independent full-answer/EOS rescoring differs')
    meta=record['metadata'];base=case['metadata']
    need(all(meta[k]==v for k,v in base.items()) and meta['mean_source']==mode and meta['null_condition']==selected['condition']
         and meta['coefficient']==(sample['n_frames'] if selected['condition']=='centered' else 1) and meta['core_key']==selected['key']
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



def prefix_relation(a,b):
    common=0
    for left,right in zip(a,b):
        if left!=right:break
        common+=1
    return dict(generated_ids_equal=a==b,common_generated_tokens=common,
        shared_executed_prefixes=min(len(a),len(b),common+1),
        first_divergent_output_position=None if a==b else common)


def prefix_self_test():
    need(prefix_relation([1,2],[1,3])==dict(generated_ids_equal=False,common_generated_tokens=1,shared_executed_prefixes=2,first_divergent_output_position=1),
        'A changed second output still used the same first two input prefixes')
    need(prefix_relation([1,2],[3,4])['shared_executed_prefixes']==1 and prefix_relation([1,2],[1,2])['first_divergent_output_position'] is None,
        'Empty/shared-prefix relation differs')
    return dict(passed=True,tests=['distinguish_output_divergence_from_executed_prefix_identity'])


def capture_self_test():
    import copy
    import torch
    n=2;width=9;case=dict(record=dict(n_frames=n),metadata=dict(prompt_width=width));selected=dict(condition='centered')
    native={'actual_states':(n,1,3584),'reference_states':(24,1,3584),'global_states':(1,3584),'native_global':(1,3584),'fused_global':(1,3584)}
    rank={'actual_messages':(n,1,96),'reference_messages':(24,1,96),'delta':(1,3584)}
    for key in ('aggregate','query','predicted_mean','bank_mean','used_mean','preactivation','projected_null','corrected_preactivation'):rank[key]=(1,96)
    first={**{k:torch.zeros(shape,dtype=torch.float16) for k,shape in native.items()},**{k:torch.zeros(shape) for k,shape in rank.items()},
        'query_indices':[width-1],'stream_positions':[width-1],'n_actual_rows':n,'n_reference_rows':24,'condition':'centered','mean_source':'learned','coefficient':n}
    second=copy.deepcopy(first);second.update(query_indices=[0],stream_positions=[width]);captures=[first,second]
    need(len(audit_captures(torch,captures,[11,151645],case,selected,'learned'))==2,'Complete synthetic prefix captures rejected')
    invalid=[captures[:1]]
    wrong=copy.deepcopy(captures);wrong[1]['stream_positions']=[width-1];invalid.append(wrong)
    wrong=copy.deepcopy(captures);wrong[0]['used_mean'].fill_(1);invalid.append(wrong)
    for values in invalid:
        try:audit_captures(torch,values,[11,151645],case,selected,'learned')
        except ValueError:pass
        else:raise AssertionError('Missing/misaligned/incorrect-mean capture accepted')
    return dict(passed=True,tests=['every_generated_prefix_captured','reject_missing_capture','reject_wrong_stream_position','reject_wrong_used_mean'])


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
        directory=Path(value).resolve();need(directory.is_relative_to(OUT),'Run outside canonical V15 study outputs')
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
        records=read(summary['predictions_file']);need(len(records)==summary['records']==68,'Missing whole-answer traces')
        rescored=[];counts=Counter()
        for index,row in enumerate(records):
            case_index=index//2;case=plan['cases'][case_index]
            need(row['index']==index and row['case_index']==case_index and row['mode']==MODES[index%2]
                 and read(directory/f'record_{index:03d}.json')==row,'Fixed generation order or durable record differs')
            need(sha(row['raw_file'])==row['raw_sha256'],'Raw native output changed')
            raw=torch.load(row['raw_file'],map_location='cpu',weights_only=True)
            need(raw['schema_version']==1 and raw['raw_logits'].shape[1]==vocab,'Native vocabulary archive differs')
            checked=audit_output(torch,tokenizer,row,raw['raw_logits'],case,selected)
            need(row['capture_identities']==audit_captures(torch,raw['captures'],row['generated_ids'],case,selected,row['mode']),
                'Captured prefix states/messages/means changed')
            rescored.append(checked)
            counts.update(row['counters']);del raw
        need(dict(counts)==summary['counters'] and counts['visual']==68 and 68<=counts['model']<=272
             and math.isclose(sum(r['model_seconds'] for r in rescored),summary['generation_seconds'],rel_tol=1e-12),
             'Aggregate raw generation count/time differs')
        job_ids.add(str(summary['slurm_job_id']));allrows.extend(rescored)
        runs[key]=dict(directory=str(directory),summary_sha256=sha(directory/'summary.json'),config_sha256=sha(directory/'config.json'),
            selected_model=selected,counters=dict(counts),generation_seconds=summary['generation_seconds'],
            model_load_seconds=summary['model_load_seconds'],elapsed_seconds=summary['elapsed_seconds'],
            peak_memory_allocated_bytes=summary['peak_memory_allocated_bytes'],peak_memory_reserved_bytes=summary['peak_memory_reserved_bytes'])
    need(set(runs)==set(CORE_KEYS) and len(allrows)==272,'Model/generation study coverage differs')
    cells={};by_k={};comparisons={}
    for key in CORE_KEYS:
        for n in (32,64):
            for mode in MODES:
                group=[r for r in allrows if r['core_key']==key and r['n_frames']==n and r['mode']==mode]
                need(len(group)==17 and {r['gold'] for r in group}==set(range(17)),'Whole-answer count support differs')
                label=f'{key}/N{n}/{mode}'
                cells[label]={scope:cell_metrics([r for r in group if r['gold'] in ks]) for scope,ks in
                              (('all',range(17)),('K0_8',range(9)),('K9_15',range(9,16)),('K16',[16]),('K9_16',range(9,17)),('K0_9',range(10)),('K10_16',range(10,17)))}
                for r in group:by_k[label+f'/K{r["gold"]}']=cell_metrics([r])
            learned=cells[f'{key}/N{n}/learned']['all']['correct'];bank=cells[f'{key}/N{n}/bank']['all']['correct']
            matched={mode:{r['sid']:r for r in allrows if r['core_key']==key and r['n_frames']==n and r['mode']==mode} for mode in MODES}
            need(set(matched['learned'])==set(matched['bank']),'Paired mean-source scene IDs differ')
            transitions=Counter((matched['learned'][sid]['exact'],matched['bank'][sid]['exact']) for sid in matched['learned'])
            comparisons[f'{key}/N{n}']=dict(n=17,bank_minus_learned_correct=bank-learned,bank_minus_learned_pp=100*(bank-learned)/17,
                both_correct=transitions[True,True],learned_only=transitions[True,False],bank_only=transitions[False,True],neither_correct=transitions[False,False],
                paired_rows=[dict(sid=sid,gold=matched['learned'][sid]['gold'],learned_ids=matched['learned'][sid]['generated_ids'],bank_ids=matched['bank'][sid]['generated_ids'],
                    learned_exact=matched['learned'][sid]['exact'],bank_exact=matched['bank'][sid]['exact'],
                    **prefix_relation(matched['learned'][sid]['generated_ids'],matched['bank'][sid]['generated_ids'])) for sid in sorted(matched['learned'])])
    allocation=allocation_ledger(out)
    need(job_ids<={r['job_id'] for r in allocation['jobs']} and allocation['allocated_gpu_seconds']<=4200,
         'Study completed IDs or total allocated GPU budget differs')
    scheduler={r['job_id']:r for r in allocation['jobs']}
    need(len(job_ids)==4 and all(scheduler[j]['job_name']=='v15_null_study' and scheduler[j]['state']=='COMPLETED' and scheduler[j]['gpus']==1 and scheduler[j]['exit_code']=='0:0' for j in job_ids)
        and plan['software_gate']['software_job_id'] in scheduler
        and scheduler[plan['software_gate']['software_job_id']]['job_name']=='v15_null_profile'
        and scheduler[plan['software_gate']['software_job_id']]['state']=='COMPLETED' and scheduler[plan['software_gate']['software_job_id']]['gpus']==1 and scheduler[plan['software_gate']['software_job_id']]['exit_code']=='0:0','Selected runs/software missing successful allocations')
    original_comparison=[]
    for selected in plan['selected_models']:
        source=selected['original_test'];need(sha(source['file'])==source['sha256'] and sha(source['raw_file'])==source['raw_sha256'],
            'Prior verified N+1 output archive changed')
        prior_rows={r['sid']:r for r in read(source['file'])['rows']}
        for case in plan['cases']:
            sample=case['record'];before=prior_rows[sample['sid']]
            need(all(before[k]==sample[k] for k in ('sid','gold','n_frames','content_sha256')),'Original comparison scene changed')
            prior_score=score_ids(tokenizer,before['generated_ids'],sample['gold'])
            need(prior_score['exact']==before['exact'] and prior_score['prediction']==before['prediction'],'Original complete-answer score differs')
            now=next(r for r in allrows if r['core_key']==selected['key'] and r['sid']==sample['sid'] and r['mode']=='learned')
            original_comparison.append(dict(core_key=selected['key'],sid=sample['sid'],n_frames=sample['n_frames'],gold=sample['gold'],
                original_generated_ids=before['generated_ids'],augmented_generated_ids=now['generated_ids'],
                original_exact=prior_score['exact'],augmented_exact=now['exact'],ids_equal=before['generated_ids']==now['generated_ids'],
                original_file=source['file'],original_sha256=source['sha256'],descriptive_only=True))
    need(len(original_comparison)==136,'Prior N+1 comparison must retain34 scenes for every model')
    save(out/'rescored_predictions.json',allrows)
    analysis=dict(schema_version=1,passed=True,completed=True,audit_passed=True,source_sha256=frozen,policy=POLICY,
        plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),software_gate=plan['software_gate'],prior=plan['prior'],runs=runs,cells=cells,by_k=by_k,
        comparisons=comparisons,original_N_plus_1_comparison=original_comparison,allocation=allocation,self_tests=tests,records=272,
        captures_all_prefixes=True,capture_count=sum(len(r['generated_ids']) for r in allrows),
        rescored_file=str(out/'rescored_predictions.json'),rescored_sha256=sha(out/'rescored_predictions.json'),
        no_fit=True,no_model_selection=True,exploratory_reused_test=True,no_practical_milestone_claim=True,
        limitations=['Only17 deterministically selected V14 families already used for evaluation; exploratory diagnostic, not fresh confirmation.',
          'Every mode uses N+25 native streams; added reference compute is shared, not free; generated token counts may differ.',
          'Original N+1 versus augmented learned is descriptive because native batch routes and kernels differ.',
          "Later prefix captures follow each mode's own generated history and are not same-state causal comparisons after divergence.",
          'The bank uses training K0 labels and exact24 occurrences; it is not a population-null oracle. No N16 anchor is used.',
          'Raw complete numeral+EOS is primary here; first-token correctness does not distinguish10..16.',
          'All four selected models and malformed/truncated outputs remain included.',
          'This prospectively rejects every nonterminal tokenizer special token, including tokens hidden by skip_special_tokens decoding.',
          'This does not establish a new attention operator, reasoning composition, or a general exact counting algorithm.'])
    save(out/'analysis.json',analysis)
    lines=['# V15 exploratory learned versus empirical null comparison','',
        'Independent audit passed. Complete numeral plus native EOS within four tokens; all272 generations retained.','',
        '| Model | N | Mode | Whole answer | First token | Malformed | Truncated |','|---|---:|---|---:|---:|---:|---:|']
    for key in CORE_KEYS:
        for n in (32,64):
            for mode in MODES:
                v=cells[f'{key}/N{n}/{mode}']['all'];lines.append(f'| {key} | {n} | {mode} | {v["correct"]}/17 | {v["first_token_correct"]}/17 | {v["malformed"]}/17 | {v["truncated"]}/17 |')
    lines+=['','These are17 reused V14 families, all four fixed-final cores/predictors, and matched N+25 augmented compute. Only the selected mean source differs within each model; both means are computed in both modes. Bank rescue supports a mean-source effect in these fixed augmented native executions. Cached-versus-live reference states and the finite bank prevent attributing it specifically to predictor capacity; failure of both does not isolate aggregation capacity. No practical milestone or reasoning composition is certified.','',
        '[All K cells, raw rescoring, sources and cost](analysis.json).']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    save(out/'summary.json',dict(passed=True,completed=True,audit_passed=True,source_sha256=frozen,analysis_file=str(out/'analysis.json'),
        analysis_sha256=sha(out/'analysis.json'),records=272,allocated_gpu_seconds=allocation['allocated_gpu_seconds'],no_practical_milestone_claim=True))



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
        need(args.software_summary is not None,'Completed exact V15 GPU software summary required');check(args,out,frozen)
    elif args.run:
        need(args.plan is not None and args.model_index is not None,'Frozen CPU study plan and model index required');args.plan=args.plan.resolve();run(args,out,frozen)
    elif args.report:
        need(args.plan is not None,'Exact common study plan required');args.plan=args.plan.resolve();report(args,out,frozen)
    else:save(out/'summary.json',dict(passed=True,tests=self_test(),source_sha256=frozen))
    need(sources()==frozen,'Study source changed during execution')
    print(json.dumps(dict(directory=str(out),summary_file=str(out/'summary.json'))),flush=True)



if __name__=='__main__':main()

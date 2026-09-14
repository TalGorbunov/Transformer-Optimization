"""Independent CPU audit of one complete unassisted fresh1400 evaluation.

Only saved native norm/head arithmetic is replayed. No decoder, vision,
supervision core, backward, optimizer, or checkpoint selection is executed.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from scripts import report_mmred_official_native_memory as reference
from scripts import report_mmred_official_native_training as training_audit

need = reference.need
sha = reference.sha
read = reference.read
save = reference.save
p = reference.p
PROTOCOL = 'mmred_prefix_supervision_fresh_evaluation_audit'
OUT = REPO / 'outputs/native_aggregation_vlm/mmred_prefix_supervision_evaluation_audit'
DATA = Path('/mnt/data/gabriele/gnn_transformer/mmred_prefix_supervision_evaluation_audit')
OWN = ('scripts/report_mmred_prefix_supervision_evaluation.py',
       'slurm/mmred_prefix_supervision_evaluation_report.sbatch')
METHODS = ('anchor', 'answer25', 'local25', 'prefix25', 'answer26', 'local26', 'prefix26')
POLICY = dict(protocol=PROTOCOL, cpu_seconds=7200, cpu_cores=4, memory_gib=16,
    worlds=1400, native_maximum_new_tokens=50, maximum_cpu_head_calls=70000,
    maximum_cpu_head_rows=70000, native_tv_limit=.02, cpu_argmax_gate=False,
    model_calls=0, decoder_calls=0, vision_calls=0, backward_calls=0,
    optimizer_steps=0, auxiliary_core_calls=0, no_efficacy_gate=True,
    no_checkpoint_selection=True, no_further_release=True,
    all_numerical_evidence_collected_before_gate=True)


def sources():
    return {name: sha(REPO / name) for name in OWN}


def ref(path):
    path = Path(path).resolve()
    return dict(file=str(path), sha256=sha(path))


def require_ref(value):
    need(sha(value['file']) == value['sha256'], 'Bound file changed: ' + value['file'])
    return Path(value['file'])


def expected_cohort(rows):
    need(len(rows) == len({r['sid'] for r in rows}) == len({r['world_sha256'] for r in rows}) == 1400,
         'All1400 distinct fresh main worlds required')
    need([r['index'] for r in rows] == list(range(1400))
         and all(r['cohort'] == 'main' and r['pilot_role'] == 'fresh_main' for r in rows),
         'Exact main-only source ordering required')
    cells = Counter((r['n'], r['qtype']) for r in rows)
    expected = {(n, q): (200 if n == 32 and q in ('spend_together', 'where_spend') else 100)
                for n in (8, 16, 32)
                for q in ('char_at_frame', 'steps_in_room', 'spend_together', 'where_spend')}
    need(cells == Counter(expected), 'Fresh task/length population differs')
    return dict(worlds=1400, by_length={str(n): sum(r['n'] == n for r in rows) for n in (8, 16, 32)},
                primary_N32_worlds=400, controls_N32_worlds=200,
                distinct_worlds=True, original_order=True, diagnostic_questions=0)


class Audit:
    def __init__(self, torch, out, data, started):
        self.torch, self.out, self.data, self.started = torch, out, data, started
        self.bindings, self.manifest = {}, {}
        self.metrics, self.predictions = [], []
        self.head_calls = self.head_rows = 0
        self.head_seconds = 0.

    def guard(self):
        need(time.perf_counter() - self.started < POLICY['cpu_seconds'] - 60,
             'Fresh evaluation audit publication reserve reached')

    def bind(self, path, expected=None):
        path = str(Path(path).resolve())
        if path in self.bindings:
            need(expected is None or self.bindings[path] == expected, 'Conflicting input hash')
            return self.bindings[path]
        return reference.bind(path, self.bindings, expected)

    def record(self, path, expected=None):
        self.bind(path, expected)
        return read(path)

    def load(self, value):
        self.guard()
        path = str(Path(value['file']).resolve())
        need(self.manifest.get(path) == value['sha256'], 'Raw capture missing from producer artifact ledger')
        self.bind(path, value['sha256'])
        return self.torch.load(path, map_location='cpu', weights_only=True)


def verify_report(summary_path):
    """Metadata/hash handoff: raw1400 captures were read by the completed audit."""
    path = Path(summary_path).resolve()
    summary = read(path)
    need(summary['protocol'] == PROTOCOL and summary['policy'] == POLICY
         and summary['passed'] is summary['completed'] is True
         and not (path.parent / 'failure.json').exists(), 'Complete passed fresh evaluation audit required')
    for key in ('analysis', 'input_bindings', 'artifacts', 'predictions'):
        require_ref(summary[key])
    analysis = read(summary['analysis']['file'])
    need(analysis['protocol'] == PROTOCOL and analysis['policy'] == POLICY
         and analysis['passed'] is analysis['completed'] is True
         and analysis['all_scheduled_numerical_evidence_collected'] is True
         and analysis['raw_inputs_verified_in_cpu_audit'] is True
         and analysis['raw_files_rehashed_in_handoff'] is False
         and analysis['source_sha256'] == sources()
         and analysis['inherited_source_sha256'] == inherited_sources()
         and analysis['numerical_failures'] == [] and analysis['no_efficacy_gate'] is True,
         'Fresh audit scope/source differs')
    for key in ('method','arm','seed','final_checkpoint'):
        need(summary[key]==analysis[key],'Fresh audit summary owner differs')
    need(summary['predictions'] == dict(file=analysis['predictions_file'], sha256=analysis['predictions_sha256']),
         'Prediction handoff differs')
    for key in ('producer_summary', 'final_checkpoint', 'statistics_protocol', 'feature_stage', 'shareddata_populations'):
        require_ref(analysis[key])
    require_ref(dict(file=analysis['original_rows_file'], sha256=analysis['original_rows_sha256']))
    for name, digest in {**analysis['inherited_source_sha256'], **analysis['source_sha256']}.items():
        need(sha(REPO / name) == digest, 'Bound audit/producer source changed')
    for name, digest in analysis['source_sha256'].items():
        need(sha(path.parent / 'source' / name.replace('/', '_')) == digest, 'Auditor source archive changed')
    return analysis


def inherited_sources():
    from scripts import evaluate_mmred_prefix_supervision as producer
    return {**producer.inherited_sources(), **producer.sources()}


def method_id(arm, seed):
    return 'anchor' if arm == 'ordinary' and seed is None else arm + str(seed)


def input_audit(a, summary_path):
    from scripts import evaluate_mmred_prefix_supervision as producer
    from scripts import report_mmred_official_native_evaluation as old_evaluation_audit
    path=Path(summary_path).resolve();directory=path.parent;summary=a.record(path)
    need(summary['protocol']==producer.PROTOCOL and summary['policy']==producer.POLICY
         and summary['phase']=='run' and summary['passed'] is summary['completed'] is True
         and not (directory/'failure.json').exists(),'Complete computational fresh evaluation required')
    config=a.record(directory/'config.json')
    need(all(summary[k]==v for k,v in config.items()),'Producer summary/config aliases differ')
    original=a.record(summary['analysis_file'],summary['analysis_sha256'])
    need(summary['analysis_file']==str(directory/'analysis.json') and summary['artifacts_file']==str(directory/'artifacts.json'),'Producer artifact paths differ')
    a.manifest=a.record(summary['artifacts_file'],summary['artifacts_sha256'])
    need(all(Path(k).is_absolute() for k in a.manifest),'Absolute artifact paths required')
    for file,digest in a.manifest.items():a.guard();a.bind(file,digest)
    need(config['source_sha256']==producer.sources() and config['inherited_source_sha256']==producer.inherited_sources(),'Frozen evaluator source differs')
    inherited={**config['inherited_source_sha256'],**config['source_sha256']}
    for name,digest in inherited.items():a.bind(REPO/name,digest);a.bind(directory/'source'/name.replace('/','_'),digest)
    need(set(OWN).isdisjoint(inherited),'Auditor/producer source ownership overlap')
    a.bind(config['plan_file'],config['plan_sha256']);plan=producer.verify_plan(config['plan_file'])
    for mapping in (plan['input_bindings'],plan['artifacts']):
        for file,digest in mapping.items():a.bind(file,digest)
    arm,seed,task=config['arm'],config['seed'],config['task_index'];key=producer.task_key(arm,seed)
    need(0<=task<7 and (arm,seed)==producer.TASKS[task] and key==config['task_key'] and method_id(arm,seed) in METHODS
         and directory==producer.OUT/f'run_{config["array_job_id"]}_{task}'
         and Path(config['data_directory'])==producer.DATA/directory.name,'Registered method/task/data ownership differs')
    request=a.record(directory/'request.json')
    need(request['phase']=='run' and request['task']==task and request['plan']==config['plan_file']
         and request['policy']==producer.POLICY and request['source_sha256']==config['source_sha256']
         and request['inherited_source_sha256']==config['inherited_source_sha256'],'Actual evaluation request differs')
    owner=plan['checkpoints'][key]
    need(config['checkpoint_owner']==owner and config['final_checkpoint']==original['final_checkpoint']==owner['final_checkpoint']
         and config['main_report']==owner['main_report'] and config['peft_config']==owner['peft_config'],'Only this audited final checkpoint may generate')
    for name in ('feature_stage','compact_stage','fresh_cohort','native_identity','native_identity_sha256','statistics_protocol','statistics_release'):
        need(config[name]==plan[name],'Evaluation input/protocol alias differs: '+name)
    need(config['hardware']==plan['reference_hardware'] and all(config[k] is False for k in
         ('supervision_module_constructed','supervision_hook_installed','auxiliary_target_tensors_opened','checkpoint_optimizer_used'))
         and all(v==[[],[],[]] for v in config['baseline_hook_signature'].values()),'Native precision/no-supervision contract differs')
    need(a.record(directory/'base_before.json')==a.record(directory/'base_after.json')
         and a.record(directory/'actual_peft_config.json')==owner['peft_config'],'Frozen base or installed PEFT configuration changed')
    stage=producer.features.verify_stage(plan['feature_stage']['file'],streaming=True)
    a.bind(plan['feature_stage']['file'],plan['feature_stage']['sha256']);fs=read(plan['feature_stage']['file']);a.bind(fs['plan_file'],fs['plan_sha256'])
    need(plan['feature_plan']==dict(file=fs['plan_file'],sha256=fs['plan_sha256']) and all(stage[k]==plan[k] for k in
         ('rows_file','cases_file','feature_index_file','populations_file','compact_stage','fresh_cohort','native_identity','native_identity_sha256','precision','packages','system_prompt')),
         'Fresh feature/compact/native bank differs')
    rows=a.record(plan['rows_file']);items=a.record(plan['cases_file']);cohort=expected_cohort(rows)
    need(len(items)==1400 and all(item['row']==rows[i] and item['compact']['index']==item['features']['index']==i for i,item in enumerate(items)),
         'Exact source-ordered feature/compact records required')
    populations=a.record(plan['populations_file']);prefix=[next(i for i,r in enumerate(rows) if r['n']==n) for n in (8,16,32)]
    order=prefix+[i for i in range(1400) if i not in prefix]
    need(a.record(plan['order_file'])==order and plan['prefix_indices']==prefix and populations['main']==list(range(1400))
         and populations['main_N32_primary']==[i for i,r in enumerate(rows) if r['n']==32 and r['qtype'] in ('spend_together','where_spend')]
         and populations['main_N32_controls']==[i for i,r in enumerate(rows) if r['n']==32 and r['qtype'] in ('char_at_frame','steps_in_room')]
         and populations['main_by_length']=={str(n):[i for i,r in enumerate(rows) if r['n']==n] for n in (8,16,32)},'Fixed populations/order differ')
    for value in (plan['statistics_protocol'],plan['statistics_release'],plan['fresh_cohort']):a.bind(value['file'],value['sha256'])
    continued=producer.main_auditor().verify_matched_reports([v['main_report']['file'] for k,v in plan['checkpoints'].items() if k!='ordinary'])
    by_arm={(v['arm'],v['seed']):v for v in continued}
    need(set(by_arm)=={(x,s) for x in ('answer','local','prefix') for s in (25,26)},'All six matched continuation audits required')
    for task_owner in plan['checkpoints'].values():
        a.bind(task_owner['main_report']['file'],task_owner['main_report']['sha256'])
        if task_owner['arm']!='ordinary':
            audit=by_arm[(task_owner['arm'],task_owner['seed'])]
            need(audit['final_checkpoint']==task_owner['final_checkpoint'] and audit['final_adapter_tensors']==task_owner['final_adapter_tensors']
                 and audit['native_identity_sha256']==plan['native_identity_sha256'] and audit['peft_config']==task_owner['peft_config'],'Continuation main audit endpoint differs')
    main=old_evaluation_audit.main_report_metadata(owner['main_report'],a.bindings) if arm=='ordinary' else by_arm[(arm,seed)]
    return producer,directory,summary,config,original,plan,stage,rows,items,order,cohort,main,inherited


def checkpoint_audit(a,config,plan):
    torch=a.torch;owner=config['checkpoint_owner'];descriptor=owner['final_checkpoint'];a.bind(descriptor['file'],descriptor['sha256'])
    state=torch.load(descriptor['file'],map_location='cpu',weights_only=True);reference.adapter_state(torch,state['adapter'])
    arm,seed=config['arm'],config['seed']
    need(state['arm']==arm and state['updates']==1500 and state['peft_config']==owner['peft_config'],'Exact final1500 native adapter required')
    if arm=='ordinary':need(seed is None and state['final_checkpoint'] is True and state['memory'] is None,'Original ordinary anchor required')
    else:
        need(state['seed']==seed and state['initial']==owner['initial_checkpoint'] and state['source_release']==owner['source_release']
             and state['targets']==owner['targets'] and state['policy']==owner['training_policy'] and 'memory' not in state,
             'Final continuation checkpoint lineage differs')
    identity=reference.state_identity(state['adapter']);need(identity==owner['final_adapter_tensors'],'Final endpoint tensor identities differ')
    proof=dict(arm=arm,seed=seed,updates=1500,adapter=identity,peft_config=owner['peft_config'],tensors=224,parameters=10092544,
               finite=True,adapter_only_restored=True,no_inference_supervision=True)
    need(config['state_proof']==owner['state_proof']==proof,'Actual checkpoint proof differs')
    contract=config['contract'];expected={name:dict(shape=list(v.shape),dtype=str(v.dtype)) for name,v in state['adapter'].items()}
    need({name:{k:v for k,v in item.items() if k!='object_id'} for name,item in contract['tensors'].items()}==expected
         and {k:v for k,v in contract.items() if k!='tensors'}==dict(targets=list(p.TARGETS),trainable_parameters=10092544,
             active_adapter='default',scaling=2.,rank=16,alpha=32,dropout=.05),'Actual frozen native adapter scope differs')
    provenance=dict(native_identity_sha256=plan['native_identity_sha256'],lora=descriptor,memory=None,arm=arm,seed=seed,
                    task_key=config['task_key'],no_inference_supervision=True)
    need(config['provenance']==provenance,'Actual unassisted checkpoint provenance differs')
    return proof


def forecast(reference_timing,rows,setup,completed=(),elapsed=None):
    units={int(n):v for n,v in reference_timing['complete_world_seconds_by_n'].items()};done=set()
    for row in completed:
        need(row['index'] not in done,'Repeated resource-prefix case');done.add(row['index']);units[row['n']]=max(units[row['n']],row['seconds'])
    remaining=Counter(r['n'] for i,r in enumerate(rows) if i not in done);base=setup if elapsed is None else elapsed
    total=base+1.25*sum(remaining[n]*units[n] for n in (8,16,32))+60
    need(all(math.isfinite(v) and v>0 for v in (setup,base,total,*units.values())),'Finite positive forecast required')
    return dict(passed=total<=3600,projected_seconds=total,cap_seconds=3600,setup_seconds=setup,elapsed_at_prefix_gate=elapsed,
        base_seconds=base,remaining_worlds=sum(remaining.values()),remaining_counts={str(n):remaining[n] for n in (8,16,32)},
        complete_world_seconds_by_n={str(n):units[n] for n in (8,16,32)},completed_indices=[r['index'] for r in completed],margin=1.25,reserve_seconds=60,
        timing_population='all1000 audited original ordinary complete-world measurements plus retained fresh prefix',
        maximum_new_tokens=50,no_token_length_rescaling=True,not_a_fifty_token_worst_case_bound=True,empirical_estimate_not_guarantee=True,accuracy_not_used=True)


def allocation(config,out):
    identifier=str(config['array_job_id'])+'_'+str(config['task_index'])
    command=['sacct','-X','-j',identifier,'--parsable2','--noheader','--format=JobID%40,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,TimelimitRaw']
    text=subprocess.run(command,capture_output=True,text=True,check=True).stdout;file=out/'sacct.psv';file.write_text(text)
    rows=[line.split('|') for line in text.splitlines() if line.split('|')[0]==identifier];need(len(rows)==1,'One completed evaluator allocation required')
    job,name,partition,state,exitcode,seconds,tres,limit=rows[0]
    generic=re.findall(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',tres);typed=re.findall(r'(?:^|,)gres/gpu:[^=,]+=(\d+)(?=,|$)',tres)
    gpus=int(generic[0]) if generic else sum(map(int,typed))
    need(name=='mmred_prefix_supervision_evaluation_run' and partition=='gpu' and state=='COMPLETED' and exitcode=='0:0'
         and gpus==1 and 0<int(seconds)<=3600 and int(limit)==60,'One-hour B200 evaluation accounting differs')
    return dict(job_id=job,job_name=name,state=state,exit_code=exitcode,allocated_gpu_seconds=int(seconds),gpu_count=gpus,cap_seconds=3600,raw=ref(file))


def run(a,args):
    producer,directory,summary,config,original,plan,stage,rows,items,order,cohort,main,inherited=input_audit(a,args.run)
    proof=checkpoint_audit(a,config,plan)
    for name,digest in sources().items():
        target=a.out/'source'/name.replace('/','_');target.write_bytes((REPO/name).read_bytes());need(sha(target)==digest,'Auditor source copy differs')
    save(a.out/'parser_fixtures.json',training_audit.parser_proof())
    parent=plan['checkpoints']['ordinary']['profile_preparation'];native_plan=a.record(parent['plan_file'],parent['plan_sha256'])
    modules=reference.native_modules(a.torch,native_plan,a.bindings)
    from transformers import AutoProcessor,__version__ as version
    processor=AutoProcessor.from_pretrained(str(producer.profile.preparation.MODEL),trust_remote_code=True,use_fast=False,local_files_only=True)
    need(p.fingerprint(processor,str(version))==plan['native_identity']['processor'] and processor.tokenizer.eos_token_id==151645,'Native fresh tokenizer differs')
    need(original['protocol']==producer.PROTOCOL and original['policy']==producer.POLICY and original['passed'] is original['completed'] is True
         and original['all1400_completed'] and original['all_seven_checkpoints_required'] and original['no_inference_supervision']
         and original['no_accuracy_release_gate'] and original['computational_completion_only'] and original['requires_independent_cpu_audit']
         and original['no_further_release'] and original['objective_achieved'] is False
         and original['statistics_protocol']==plan['statistics_protocol'] and original['provenance']==config['provenance'],
         'Complete unassisted computational evaluation required')
    for key in ('arm','seed','task_key','task_index'):need(original[key]==config[key],'Actual method alias differs')
    natural=original['natural'];need([r['index'] for r in natural]==list(range(1400)),'All1400 source-ordered outcomes required')
    totals=Counter();tables=defaultdict(Counter);method=method_id(config['arm'],config['seed'])
    for row in natural:
        a.guard();index=row['index'];item=items[index];source=item['row']
        need(all(row[k]==source[k] for k in ('sid','n','qtype','pilot_role')) and all(row[k]==config[k] for k in ('arm','seed','task_key'))
             and order[row['execution_index']]==index and row['auxiliary_hook_installed'] is False
             and row['actual_output_tokens']==len(row['generated_ids']) and 1<=row['actual_output_tokens']<=50,'Fresh source/method/token ownership differs')
        need(row['file']==str(Path(config['data_directory'])/f'natural_{index:04d}.pt') and row['raw_bytes']==Path(row['file']).stat().st_size
             and row['max_reserved_bytes']>=row['max_allocated_bytes']>0
             and all(math.isfinite(row[k]) and row[k]>=0 for k in ('seconds','load_transfer_seconds','generation_with_observation_seconds','post_generation_evidence_seconds'))
             and row['seconds']>0 and row['generation_with_observation_seconds']>0
             and row['post_generation_evidence_seconds']==row['seconds']-row['load_transfer_seconds']-row['generation_with_observation_seconds'],
             'Complete finite native cost/capture metadata required')
        need(a.record(directory/f'natural_{index:04d}.json')==row and a.record(directory/f'natural_{index:04d}_observation.json')==
             {k:v for k,v in row.items() if k not in ('seconds','post_generation_evidence_seconds')},'Durable per-world observation/timing differs')
        raw=a.load(row);e=raw['evidence']
        need(raw['arm']==config['arm'] and raw['seed']==config['seed'] and raw['task_key']==config['task_key'] and raw['task_index']==config['task_index']
             and raw['case_index']==index and raw['phase']=='cold' and e['parameter_state']==config['final_checkpoint']
             and e['feature_packet']==item['features'] and e['compact_packet']==item['compact'] and e['provenance']==config['provenance']
             and e['prefix_supervision_active'] is e['supervision_hook_installed'] is False and 'prefix_supervision' not in e,
             'Actual native input/state acquired an auxiliary intervention')
        case,features=training_audit.load_case(a.torch,item,a.bindings)
        need(all(case['metadata'][k]==source[k] for k in ('sid','n','qtype','atype','question','raw_row_sha256'))
             and row['prompt_tokens']==case['inputs']['input_ids'].shape[-1]==case['metadata']['prompt_width']
             and row['visual_tokens']==196*source['n']
             and case['target_text']==producer.features.preparation.canonical_target(source['atype'],source['answer']),
             'Exact fresh prompt/answer/image ownership differs')
        tick=time.perf_counter()
        # Only the immutable helper route is ordinary; all actual method owners are joined above.
        metric=training_audit.natural_audit(a.torch,dict(raw,arm='ordinary'),row,case,item,'ordinary',
            dict(final_checkpoint=config['final_checkpoint']),None,processor,modules,a.data,f'natural_{index}')
        a.head_seconds+=time.perf_counter()-tick;a.head_calls+=metric['head_calls'];a.head_rows+=metric['head_rows']
        metric.pop('training_diagnostic_only');metric.pop('memory_reconstruction_scope');metric.update(method=method,arm=config['arm'],seed=config['seed'],fresh_evaluation=True)
        path=a.out/f'natural_{index:04d}.json';save(path,metric)
        a.metrics.append(dict(index=index,sid=row['sid'],passed=metric['passed'],head_calls=metric['head_calls'],head_rows=metric['head_rows'],**ref(path)))
        prediction={k:row[k] for k in ('index','sid','n','qtype','pilot_role','generated_ids','primary_text','score','seconds',
            'load_transfer_seconds','generation_with_observation_seconds','post_generation_evidence_seconds','actual_output_tokens',
            'prompt_tokens','visual_tokens','max_allocated_bytes','max_reserved_bytes','file','sha256')}
        prediction.update(method=method,arm=config['arm'],seed=config['seed'],task_key=config['task_key'],world_sha256=source['world_sha256'],cohort_index=source['cohort_index'])
        a.predictions.append(prediction)
        count=dict(contexts=1,correct=int(metric['score']['correct']),format_valid=int(metric['score']['format_valid']),
            completed=int(metric['score']['completed']),truncated=int(metric['score']['truncated']),output_tokens=metric['score']['output_tokens'])
        totals.update(count);tables[f'{source["n"]}:{source["qtype"]}'].update(count)
        del raw,e,case,features,metric
    token_count=sum(r['actual_output_tokens'] for r in natural)
    expected=dict(model=token_count,backbone=token_count,visual=0,language=token_count,norm=token_count,head=token_count,backward=0,optimizer=0)
    need(a.head_calls==a.head_rows==token_count<=70000 and original['counters']==summary['counters']==expected
         and original['decoder_layer_calls']==[token_count]*28 and original['head_rows']==summary['head_rows']==token_count
         and original['actual_visual_calls']==0 and original['logical_visual_calls']==1400,'Native all-row work inventory differs')
    reference_timing=a.record(plan['reference_timing_file']);prior=reference_timing['rows']
    need(len(prior)==1000 and [r['index'] for r in prior]==list(range(1000)) and Counter(r['n'] for r in prior)=={8:400,16:400,32:200}
         and reference_timing['complete_world_seconds_by_n']=={str(n):max(r['seconds'] for r in prior if r['n']==n) for n in (8,16,32)},
         'All original complete-world timing maxima required')
    prefix=[natural[i] for i in plan['prefix_indices']];gate=a.record(directory/'prefix_projection.json');setup=original['setup_seconds']
    need(original['original_projection']==plan['projection']==forecast(reference_timing,rows,plan['projection']['setup_seconds'])
         and original['setup_projection']==a.record(directory/'setup_projection.json')==forecast(reference_timing,rows,setup)
         and original['prefix_projection']==gate==forecast(reference_timing,rows,setup,prefix,gate['elapsed_at_prefix_gate'])
         and a.record(directory/'prefix_natural.json')==prefix
         and all(v['passed'] for v in (plan['projection'],original['setup_projection'],gate)),'Independent resource projection differs')
    shared=plan['shared_visual_work']
    need(original['shared_visual_work']==shared and shared['stage']==plan['feature_stage'] and shared['actual_preparation_once']
         and shared['logical_charge_per_checkpoint']==dict(vision=1400,frames=28800,feature_tokens=5644800)
         and shared['timings']==a.record(shared['timings_file'],shared['timings_sha256']),'Shared visual cost attribution differs')
    for key in ('seconds','load_transfer_seconds','generation_with_observation_seconds','post_generation_evidence_seconds'):
        actual_key='inference_seconds' if key=='seconds' else key
        need(original[actual_key]==sum(r[key] for r in natural),'Complete native timing total differs')
    need(original['raw_bytes']==sum(r['raw_bytes'] for r in natural),'Raw capture storage total differs')
    resources=dict(allocation=allocation(config,a.out),inference_seconds=original['inference_seconds'],setup_seconds=setup,
        evaluation_elapsed_seconds=summary['elapsed_seconds'],actual_output_tokens=token_count,actual_gpu_counters=expected,
        decoder_layer_calls=original['decoder_layer_calls'],main_training=main['resources'],shared_evaluation_features=shared,
        logical_visual_calls=1400,logical_visual_frames=28800,logical_visual_tokens=5644800,
        instrumented_latency_not_production=True,projection=gate,cpu_cap_is_conservative_not_runtime_prediction=True)
    predictions=a.data/'predictions.json';save(predictions,a.predictions)
    return dict(protocol=PROTOCOL,policy=POLICY,passed=all(v['passed'] for v in a.metrics),completed=True,
        method=method,arm=config['arm'],seed=config['seed'],task_key=config['task_key'],task_index=config['task_index'],
        producer_summary=ref(args.run),final_checkpoint=config['final_checkpoint'],state_proof=proof,main_report=config['main_report'],
        feature_stage=plan['feature_stage'],fresh_cohort=plan['fresh_cohort'],shareddata_populations=ref(plan['populations_file']),
        statistics_protocol=plan['statistics_protocol'],original_rows_file=plan['rows_file'],original_rows_sha256=sha(plan['rows_file']),
        predictions_file=str(predictions),predictions_sha256=sha(predictions),native_identity_sha256=plan['native_identity_sha256'],
        cohort=cohort,totals=dict(totals),tables={k:dict(v) for k,v in tables.items()},natural_audits=a.metrics,
        numerical_failures=[r['index'] for r in a.metrics if not r['passed']],cpu_head_calls=a.head_calls,cpu_head_rows=a.head_rows,
        cpu_head_seconds=a.head_seconds,resources=resources,source_sha256=sources(),inherited_source_sha256=inherited,
        all_scheduled_numerical_evidence_collected=True,raw_inputs_verified_in_cpu_audit=True,raw_files_rehashed_in_handoff=False,
        no_inference_supervision=True,no_efficacy_gate=True,no_further_release=True,objective_achieved=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu' and int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==4
         and not os.environ.get('SLURM_JOB_GPUS') and not os.environ.get('SLURM_STEP_GPUS'),'CPU4 Slurm-only audit required')
    started=time.perf_counter();tag='report_'+os.environ['SLURM_JOB_ID'];out=OUT/tag;data=DATA/tag
    out.mkdir(parents=True,exist_ok=False);data.mkdir(parents=True,exist_ok=False);(out/'source').mkdir()
    own=sources();save(out/'request.json',dict(protocol=PROTOCOL,policy=POLICY,run=ref(args.run),source_sha256=own))
    import torch
    torch.set_num_threads(4);a=Audit(torch,out,data,started)
    try:
        analysis=run(a,args);elapsed=time.perf_counter()-started;analysis['resources']['cpu_audit_elapsed_seconds']=elapsed;analysis['elapsed_seconds']=elapsed
        need(sources()==own,'Auditor source changed during execution')
        save(out/'analysis.json',analysis);save(out/'input_bindings.json',a.bindings)
        save(out/'artifacts.json',{str(f):sha(f) for root in (out,data) for f in root.iterdir() if f.is_file()})
        need(analysis['passed'],'Fresh numerical audit failed after complete all1400 evidence collection')
        need(time.perf_counter()-started<POLICY['cpu_seconds'],'Fresh CPU audit cap exceeded')
        save(out/'summary.json',dict(protocol=PROTOCOL,policy=POLICY,passed=True,completed=True,
            method=analysis['method'],arm=analysis['arm'],seed=analysis['seed'],final_checkpoint=analysis['final_checkpoint'],
            analysis=ref(out/'analysis.json'),input_bindings=ref(out/'input_bindings.json'),artifacts=ref(out/'artifacts.json'),
            predictions=dict(file=analysis['predictions_file'],sha256=analysis['predictions_sha256']),
            elapsed_seconds=time.perf_counter()-started,no_efficacy_gate=True,no_further_release=True))
    except BaseException as error:
        save(out/'failure_input_bindings.json',a.bindings);save(out/'partial_numerical_audits.json',a.metrics)
        save(out/'failure.json',dict(protocol=PROTOCOL,passed=False,completed=False,type=type(error).__name__,message=str(error),
            completed_worlds=len(a.metrics),cpu_head_calls=a.head_calls,cpu_head_rows=a.head_rows,elapsed_seconds=time.perf_counter()-started,
            input_bindings=ref(out/'failure_input_bindings.json'),partial_audits=ref(out/'partial_numerical_audits.json'),
            data_directory=str(data),no_automatic_retry=True));raise


if __name__=='__main__':main()

"""Independent N16 joint-LoRA confirmation audit; CPU norm/head only.

Every native generated prefix is audited. GPU-generated IDs define efficacy;
CPU TV<=.02 is mandatory and CPU/GPU argmax differences remain descriptive.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import math
import os
from pathlib import Path
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import evaluate_native_identity_join_joint_lora_confirmation as producer
from scripts import stage_native_identity_join_joint_lora_confirmation as preparation
from scripts import report_native_identity_join_joint_lora as native_audit
from scripts.stage_native_vision_v6_teacher import need,read,save,sha
p=producer.p
OUT=REPO/'outputs/native_aggregation_vlm/identity_join_joint_lora_confirmation/evaluation'
DATA=Path('/mnt/data/gabriele/gnn_transformer/identity_join_joint_lora_confirmation/audit')
PROTOCOL='identity_join_joint_lora_confirmation_independent_audit'


def criterion(rows):
    preparation.cohort(rows,16);panels={};families=[]
    for panel,total,groups_count,whole_threshold,family_threshold in (('A',108,36,98,33),('B',108,36,98,33),('C',54,18,49,16)):
        values=[r for r in rows if r['panel']==panel];groups=defaultdict(list)
        for row in values:groups[row['contrast_id']].append(row)
        need(len(values)==total and len(groups)==groups_count and all(len(g)==3 and {r['variant'] for r in g}=={0,1,2} for g in groups.values()),'Complete fixed panel families required')
        whole=sum(r['exact'] for r in values);complete=sum(all(r['exact'] for r in g) for g in groups.values())
        panels[panel]=dict(passed=whole>=whole_threshold and complete>=family_threshold,whole_correct=whole,contexts=total,
            complete_families=complete,families=groups_count,thresholds=dict(whole_correct=whole_threshold,complete_families=family_threshold))
        for contrast_id,group in groups.items():
            families.append(dict(panel=panel,contrast_id=contrast_id,complete=all(r['exact'] for r in group),
                contexts=3,whole_correct=sum(r['exact'] for r in group),sids=[r['sid'] for r in group],
                question=group[0]['question'],trio=group[0]['trio'],orientation=group[0]['orientation']))
    return dict(n_frames=16,panels=panels,qualifies_A_and_B=panels['A']['passed'] and panels['B']['passed'],
        C_separate=panels['C']['passed'],natural_whole_answer_and_eos=True),families


def strata(rows):
    output={}
    for fields in (('panel','question'),('panel','gold'),('panel','trio'),('panel','orientation'),('panel','matches_reference_orientation')):
        groups=defaultdict(list)
        for row in rows:
            key=tuple(tuple(row[k]) if isinstance(row[k],list) else row[k] for k in fields);groups[key].append(row)
        output['/'.join(fields)]=[dict(key=dict(zip(fields,key)),contexts=len(group),whole_correct=sum(r['exact'] for r in group),
            first_correct=sum(r['first_token_correct'] for r in group)) for key,group in groups.items()]
    return output


def verify_report(path):
    """Hash/JSON-only handoff for any separately reviewed longer-length stage."""
    path=native_audit.summary_path(path);summary=read(path)
    need(summary['protocol']==PROTOCOL and summary['passed'] is summary['completed'] is summary['numerical_audit_passed'] is True
         and summary['source_sha256']==producer.sources() and summary['inherited_source_sha256']==producer.inherited_sources()
         and summary['policy']==producer.POLICY and not (path.parent/'failure.json').exists(),'Passed exact N16 confirmation audit required')
    native_audit.archive(path.parent,summary['source_sha256'],summary['inherited_source_sha256'],{})
    for key in ('analysis','artifacts','input_bindings','report'):native_audit.bind(summary[key+'_file'],summary[key+'_sha256'])
    for file,digest in read(summary['input_bindings_file']).items():native_audit.bind(file,digest)
    for file,digest in read(summary['artifacts_file']).items():native_audit.bind(file,digest)
    analysis=read(summary['analysis_file'])
    need(analysis['passed'] is analysis['completed'] is analysis['numerical_audit_passed'] is True and analysis['policy']==producer.POLICY
         and analysis['criterion']==summary['criterion'] and analysis['all_numerical_evidence_collected'] is True
         and analysis['longer_lengths_eligible']==analysis['criterion']['qualifies_A_and_B']
         and analysis['longer_lengths_released'] is False,'Complete N16 audit and separate length eligibility required')
    return analysis


def report(torch,path,out,data,bindings):
    p.bind(path,bindings);summary=read(path);directory=path.parent;config=native_audit.record(directory/'config.json',bindings)
    need(summary['protocol']==producer.PROTOCOL and summary['phase']=='N16' and summary['passed'] is summary['completed'] is True
         and summary['policy']==producer.POLICY and all(summary[k]==v for k,v in config.items())
         and summary['source_sha256']==producer.sources() and summary['inherited_source_sha256']==producer.inherited_sources()
         and not (directory/'failure.json').exists(),'Completed fixed native N16 run required')
    native_audit.archive(directory,summary['source_sha256'],summary['inherited_source_sha256'],bindings)
    stage_ref=config['prepared_stage'];main_ref=config['main_audit']
    plan,rows,prepared,parent,expected_stage,projection=producer.inputs(stage_ref['file'],main_ref['file'],bindings)
    need(stage_ref==expected_stage and main_ref==parent['descriptor'] and config['final_adapter']==parent['final']
         and config['projection']==projection and config['no_optimizer_created'] is config['no_training'] is config['no_checkpoint_selection'] is True,
         'Only the fixed audited final checkpoint may enter confirmation')
    for key in ('profile','profile_audit','stage_report','training_stage','orientation_plan','native_identity','native_identity_sha256','packages'):
        need(config[key]==plan[key],'Native preparation/configuration join differs: '+key)
    native_audit.hardware_audit(config,plan)
    need(Path(config['data_directory'])==producer.DATA/directory.name,'Native raw artifact root differs')
    need(native_audit.record(directory/'request.json',bindings)==dict(prepared_stage=stage_ref['file'],main_audit=main_ref['file'],source_sha256=producer.sources()),'Fixed inference request differs')
    for key in ('endpoint','evaluations','artifacts','input_bindings'):p.bind(summary[key+'_file'],bindings,summary[key+'_sha256'])
    for file,digest in read(summary['input_bindings_file']).items():p.bind(file,bindings,digest)
    artifacts=read(summary['artifacts_file'])
    for file,digest in artifacts.items():p.bind(file,bindings,digest)
    endpoint=read(summary['endpoint_file']);evaluations=read(summary['evaluations_file'])
    need(len(evaluations)==270 and all(endpoint[k] is True for k in ('passed','base_parameters_unchanged','adapter_parameters_unchanged','all_parameters_frozen','full_peft_config_exact')),'Final frozen inference state differs')
    p.bind(endpoint['base_before_file'],bindings,endpoint['base_before_sha256'])
    configuration=native_audit.record(directory/'actual_peft_config_before_contract.json',bindings)
    profile_ref=plan['profile_audit'];p.bind(profile_ref['analysis_file'],bindings,profile_ref['analysis_sha256'])
    profile_contract=read(profile_ref['analysis_file'])['adapter_audit']['contract']
    need(configuration==parent['final']['peft_config'] and native_audit.semantic_contract(config['adapter_contract'])==native_audit.semantic_contract(profile_contract),
         'Actual full PEFT configuration and112-module/224-tensor scope differ')
    modules=native_audit.native_modules(torch,plan,bindings);weight_before={k:p.tensor_info(m.weight) for k,m in zip(('norm','head'),modules)}
    outcomes=[];native=[];heads=[];counts=Counter();raw_files=set();timings=[]
    for index,(row,small) in enumerate(zip(rows,evaluations)):
        sid=row['sid'];need(small['index']==index and all(small[k]==row[k] for k in ('sid','panel','n_frames','contrast_id','variant','target_ids','gold')),'Exact270 cohort/order/target join differs')
        need(Path(small['file'])==Path(config['data_directory'])/f'eval_{index:03d}.pt' and artifacts[small['file']]==small['sha256'],'Native capture ownership differs')
        raw=native_audit.tensor_packet(torch,small['file'],small['sha256'],bindings);raw_files.add(small['file'])
        need(set(raw)=={'sid','result'} and raw['sid']==sid,'Actual native packet SID differs')
        packet=producer.load_packet(torch,prepared[sid]);p.bind(prepared[sid]['file'],bindings,prepared[sid]['sha256'])
        adapted=dict(bundle=packet['bundle'],teacher=dict(position_ids=packet['position_ids']))
        inspected=native_audit.generation_audit(torch,raw['result'],row,prepared[sid],adapted,plan,modules,data,f'eval_{index:03d}')
        need(small['generated_ids']==inspected['generated_ids'] and all(small[k]==v for k,v in inspected['score'].items())
             and all(small[k]==raw['result'][k] for k in ('text','raw_text','counters')),'Actual GPU IDs define complete-answer efficacy')
        native.append(inspected);heads.extend(inspected['heads']);counts.update(inspected['counters'])
        outcomes.append(dict(row,generated_ids=inspected['generated_ids'],**inspected['score']))
        timing=native_audit.record(directory/f'timing_{index:03d}.json',bindings)
        need(timing['index']==index and math.isfinite(timing['seconds']) and timing['seconds']>0,'Complete actual per-scene timing required');timings.append(timing)
    decision,families=criterion(outcomes);need(decision==summary['criterion']==endpoint['criterion']==native_audit.record(directory/'criterion.json',bindings),'Independent panel/family criterion differs')
    tokens=sum(len(r['generated_ids']) for r in outcomes);stats=native_audit.numerical_summary(heads)
    need(raw_files==set(artifacts) and dict(counts)==summary['counters']==endpoint['counters']
         and counts['model']==counts['language']==counts['norm']==counts['head']==tokens<=1080 and counts['visual']==270
         and all(counts.get(k,0)==0 for k in ('broadcast','fusion','conditioning','probe_head'))
         and stats['cpu_head_calls']==stats['cpu_head_rows']==tokens==summary['native_head_rows']==endpoint['native_head_rows'],
         'All270 trajectories/every selected native head row required')
    need(summary['longer_lengths_eligible']==decision['qualifies_A_and_B'] and summary['longer_lengths_released'] is False
         and summary['valid_negative_outcomes_retained'] is True,'Negative efficacy and software validity must remain separate')
    need(weight_before=={k:p.tensor_info(m.weight) for k,m in zip(('norm','head'),modules)}
         and all(not v.requires_grad and v.grad is None for m in modules for v in m.parameters()),'CPU native replay changed frozen weights')
    resources=producer.accounting(out,completed=True,job_id=directory.name.removeprefix('run_'))
    return dict(numerical_audit_passed=all(v['passed'] for v in native),criterion=decision,outcomes=outcomes,families=families,strata=strata(outcomes),
        native=native,counters=dict(counts),**stats,prepared_stage=stage_ref,main_audit=main_ref,final_adapter=parent['final'],
        run_summary_file=str(path),run_summary_sha256=sha(path),projection=projection,timings=timings,
        resources=resources,prior_campaign_gpu_seconds=parent['analysis']['resources']['allocated_gpu_seconds'],
        cumulative_campaign_gpu_seconds=parent['analysis']['resources']['allocated_gpu_seconds']+resources['allocated_gpu_seconds'],
        longer_lengths_eligible=decision['qualifies_A_and_B'],longer_lengths_released=False,
        reference_orientation_is_not_unseen=True,no_VLM_or_backward=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS')
         and int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==4,'Only four-core Slurm CPU audit allowed')
    out=OUT/f'report_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False);data=DATA/out.name;data.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();frozen=producer.snapshot(out);bindings={}
    try:
        import torch
        torch.set_num_threads(4);path=native_audit.summary_path(args.run);analysis=report(torch,path,out,data,bindings)
        passed=analysis['numerical_audit_passed'];analysis=dict(analysis,passed=passed,completed=True,protocol=PROTOCOL,policy=producer.POLICY,
            source_sha256=frozen,inherited_source_sha256=producer.inherited_sources())
        save(out/'analysis.json',analysis);save(out/'input_bindings.json',bindings)
        artifacts={str(file):sha(file) for file in sorted(data.rglob('*')) if file.is_file()};save(out/'artifacts.json',artifacts)
        lines=['# Ordinary joint LoRA N16 confirmation','',
            'All outcomes use actual native GPU name IDs plus EOS. Every selected query was replayed through the CPU native norm/head; TV must be at most .02 and argmax differences are descriptive.','']
        for panel,value in analysis['criterion']['panels'].items():lines.append(f'{panel}: {value["whole_correct"]}/{value["contexts"]} complete answers; {value["complete_families"]}/{value["families"]} complete families; passed={value["passed"]}.')
        lines.extend(['',f'CPU fidelity passed={passed}; maximum TV={analysis["maximum_cpu_native_head_tv"]}; rows={analysis["cpu_head_rows"]}.',
            f'N16 A/B eligibility for separately reviewed longer evaluation={analysis["longer_lengths_eligible"]}. No N32/N64 inference is released.',
            'Both room orientations were trained. C combines new-group and held-question shifts. Prior frozen-joint scores used a different fresh seed and are not a matched treatment effect.'])
        (out/'REPORT.md').write_text('\n'.join(lines)+'\n');elapsed=time.perf_counter()-started
        need(elapsed<=producer.POLICY['report_seconds'] and producer.sources()==frozen,'CPU report cap/source identity differs')
        need(passed,'CPU native TV audit failed; all scheduled numerical evidence was retained')
        save(out/'summary.json',dict(passed=True,completed=True,numerical_audit_passed=True,protocol=PROTOCOL,policy=producer.POLICY,
            source_sha256=frozen,inherited_source_sha256=producer.inherited_sources(),criterion=analysis['criterion'],elapsed_seconds=elapsed,
            analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),artifacts_file=str(out/'artifacts.json'),artifacts_sha256=sha(out/'artifacts.json'),
            input_bindings_file=str(out/'input_bindings.json'),input_bindings_sha256=sha(out/'input_bindings.json'),report_file=str(out/'REPORT.md'),report_sha256=sha(out/'REPORT.md')))
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,elapsed_seconds=time.perf_counter()-started,all_completed_numerical_evidence_retained=True));raise


if __name__=='__main__':main()

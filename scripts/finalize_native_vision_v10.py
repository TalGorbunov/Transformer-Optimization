"""Plot existing V10 metrics and bind final acceptance to both completed audits.

No new predictions, fits, thresholds, metric estimates, bootstrap draws or model
selection. Two fixed-seed columns show N32/N64 and all supported count partitions.
Labels are exactly the report's correct/total counts; uncertainty stays in the
independent report. Descriptive cache failures are preserved, not reclassified.
All plotting/finalization and its small self-test require CPU Slurm.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
import re
import getpass
import subprocess
from datetime import datetime
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
OUT = REPO/'outputs/native_aggregation_vlm/v10/finalization'
OWN = ('scripts/finalize_native_vision_v10.py', 'slurm/native_vision_v10_finalize.sbatch')
CELLS = (('main_test_N32', 136), ('main_test_N64', 136))
PARTITIONS = (('K0_8', 72), ('K9_15', 56), ('K16', 8))
CONDITIONS = ('ce', 'consistency')
SEEDS = (14, 15)
GPU_SECONDS_CAP = 16200
TERMINAL = {'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}



def need(value, message):
    if not value: raise ValueError(message)


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''): value.update(block)
    return value.hexdigest()


def save(path, value):
    with Path(path).open('x') as stream: json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def bind(path, digest): need(sha(path) == digest, 'Artifact changed: '+str(path))


def sources(): return {name: sha(REPO/name) for name in OWN}


def snapshot(out):
    (out/'source').mkdir()
    for name in OWN: (out/'source'/name.replace('/', '_')).write_bytes((REPO/name).read_bytes())
    save(out/'source_hashes.json', sources())


def source_chain(directory, ledger, folder):
    need(read(directory/'source_hashes.json') == ledger, 'Source ledger differs')
    for name, digest in ledger.items():
        bind(Path(name) if Path(name).is_absolute() else REPO/name, digest)
        bind(directory/folder/name.replace('/', '_'), digest)


def metric_check(metric, denominator):
    need(metric['n'] == denominator and type(metric['correct']) is int and 0 <= metric['correct'] <= denominator
         and math.isfinite(metric['accuracy']) and math.isclose(metric['accuracy'], metric['correct']/denominator, abs_tol=1e-12, rel_tol=0),
         'Reported count/denominator/accuracy is inconsistent')


def plotted_rows(analysis):
    rows = []
    need(set(analysis['runs']) == {f'{c}_s{s}' for c in CONDITIONS for s in SEEDS}, 'Require all four registered runs')
    for seed in SEEDS:
        for condition in CONDITIONS:
            run = analysis['runs'][f'{condition}_s{seed}']; metrics = run['metrics']
            need(run['condition'] == condition and run['seed'] == seed and run['verified_test_examples'] == 272
                 and run['verified_target_positions'] == 177120 and run['verified_scene_presentations'] == 73440
                 and run['verified_pair_presentations'] == 36720 and run['verified_optimizer_log_rows'] == 4590,
                 'Reported run or complete training/test coverage differs')
            metric_check(metrics['all'], 272)
            for n in (32, 64):
                cell = f'main_test_N{n}'; metric = metrics['cells'][cell]; metric_check(metric, 136)
                per_k = [metrics['by_n_k'][f'{cell}/K{k}'] for k in range(17)]
                for value in per_k: metric_check(value, 8)
                need(sum(v['correct'] for v in per_k) == metric['correct'], 'Per-K cells do not sum to the length total')
                partitions = metrics['partitions'][f'N{n}']
                for key, denominator, ks in (('K0_8', 72, range(9)), ('K9_15', 56, range(9,16)),
                                            ('K16', 8, (16,)), ('K9_16', 64, range(9,17)), ('nonzero', 128, range(1,17))):
                    value = partitions[key]; metric_check(value, denominator)
                    need(value['correct'] == sum(per_k[k]['correct'] for k in ks), 'Mandatory count partition differs from its per-K cells')
                need(sum(partitions[k]['correct'] for k, _ in PARTITIONS) == metric['correct'], 'Disjoint count partitions do not sum to total')
                for label, value in [('all', metric)] + [(k, partitions[k]) for k, _ in PARTITIONS]:
                    rows.append(dict(condition=condition, seed=seed, cell=cell, n_frames=n, partition=label,
                        n=value['n'], correct=value['correct'], accuracy=value['accuracy']))
            need(sum(metrics['cells'][cell]['correct'] for cell, _ in CELLS) == metrics['all']['correct'], 'Length totals do not sum to272')
    return rows


def registered_decisions(analysis):
    decisions = {}
    for seed in SEEDS:
        a = analysis['runs'][f'consistency_s{seed}']['metrics']; b = analysis['runs'][f'ce_s{seed}']['metrics']
        n32 = a['cells']['main_test_N32']['correct']; n64 = a['cells']['main_test_N64']['correct']
        d32 = n32 - b['cells']['main_test_N32']['correct']; d64 = n64 - b['cells']['main_test_N64']['correct']
        larger = a['partitions']['N64']['K9_16']['correct']; primary = d64 >= 7 and d32 >= -6
        decisions[str(seed)] = dict(n64_additional_correct=d64, n32_additional_correct=d32, primary=primary,
            consistency_n32_correct=n32, consistency_n64_correct=n64, consistency_larger_k_n64_correct=larger,
            practical=primary and n32 >= 123 and n64 >= 109 and larger >= 52)
        reported = analysis['decisions'][str(seed)]
        need(all(reported.get(k) == value for k, value in decisions[str(seed)].items()), 'Registered integer decisions differ from reported counts')
    need(set(analysis['decisions']) == set(decisions)
         and analysis['primary_both_seeds'] == all(d['primary'] for d in decisions.values())
         and analysis['practical_both_seeds'] == all(d['practical'] for d in decisions.values()), 'Both-seed report decisions differ')
    return decisions


def match_selected(analysis, post):
    need(len(post['runs']) == 4, 'Post audit must retain all four selected models')
    rows = {(row['condition'], row['seed']): row for row in post['runs']}
    need(set(rows) == {(condition, seed) for condition in ('ce', 'consistency') for seed in (14, 15)},
         'Duplicate or missing post-audit model')
    identities = []
    for condition in ('ce', 'consistency'):
        for seed in (14, 15):
            reported, audited = analysis['runs'][f'{condition}_s{seed}'], rows[condition, seed]
            need(Path(reported['run_directory']).name == audited['run_id']
                 and Path(reported['selected_checkpoint']).resolve() == Path(audited['checkpoint']).resolve()
                 and reported['selected_step'] == audited['selected_step']
                 and reported['selected_checkpoint_sha256'] == audited['checkpoint_sha256']
                 and reported['selected_parameter_sha256'] == audited['selected_parameter_sha256'],
                 'Report and post audit evaluated different selected parameters')
            identities.append(dict(condition=condition, seed=seed, run_id=audited['run_id'], selected_step=audited['selected_step'],
                checkpoint=audited['checkpoint'], checkpoint_sha256=audited['checkpoint_sha256'],
                parameter_sha256=audited['selected_parameter_sha256']))
    return identities


def acceptance(analysis, report_summary, post, accounting):
    flags = [analysis['passed'], analysis['audit_passed'], report_summary['passed'], post['completed'],
             post['computational_integrity_passed'], post['native_replay_gate_passed'], post['passed'], accounting['passed'],
             analysis['primary_both_seeds'], analysis['practical_both_seeds'], analysis['vision_milestone_gate']]
    need(all(isinstance(flag, bool) for flag in flags), 'Final acceptance fields must be explicit booleans')
    need(analysis['vision_milestone_gate'] == (analysis['primary_both_seeds'] and analysis['practical_both_seeds']),
         'Reported vision decision is internally inconsistent')
    verified = all(flags[:8])
    return dict(report_audit_passed=analysis['passed'] and analysis['audit_passed'] and report_summary['passed'],
        post_completed=post['completed'], post_computational_integrity_passed=post['computational_integrity_passed'],
        post_native_replay_gate_passed=post['native_replay_gate_passed'], verification_passed=verified,
        gpu_accounting_passed=accounting['passed'], gpu_budget_passed=accounting['budget_passed'],
        v10_concurrency_passed=accounting['concurrency_passed'],
        primary_both_seeds=analysis['primary_both_seeds'], practical_both_seeds=analysis['practical_both_seeds'],
        reported_vision_milestone_gate=analysis['vision_milestone_gate'],
        accepted_vision_milestone=verified and analysis['vision_milestone_gate'],
        post_strict_cache_numerical_gate_passed=post['strict_cache_numerical_gate_passed'],
        original_mixed_numerical_gate_passed=post['original_mixed_numerical_gate_passed'],
        reasoning_composition_established=False)


def plot(rows, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    colors = dict(ce='#66717e', consistency='#1465ac')
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42}):
        figure, axes = plt.subplots(2, 2, figsize=(11, 7.4), sharey=True)
        for column, seed in enumerate(SEEDS):
            for row_index, n in enumerate((32, 64)):
                axis = axes[row_index, column]
                for arm_index, condition in enumerate(CONDITIONS):
                    points = [next(row for row in rows if row['condition'] == condition and row['seed'] == seed
                                   and row['n_frames'] == n and row['partition'] == partition)
                              for partition in ('all', 'K0_8', 'K9_15', 'K16')]
                    x = [index + (arm_index-.5)*.34 for index in range(4)]
                    bars = axis.bar(x, [100*p['accuracy'] for p in points], width=.31, color=colors[condition], zorder=3)
                    for bar, point in zip(bars, points):
                        axis.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.8,
                            f"{point['correct']}/{point['n']}", ha='center', va='bottom', fontsize=8, color=colors[condition])
                axis.set_title(f'{n} frames · seed {seed}', loc='left', fontsize=11)
                axis.set_xticks(range(4), ['All K', 'K0–8', 'K9–15', 'K16'])
                axis.set_xlabel('Supported answer values')
                axis.set_ylim(0, 110); axis.set_yticks((0, 20, 40, 60, 80, 100))
                axis.grid(axis='y', color='#dfe3e7', linewidth=.65, zorder=0)
                if column == 0: axis.set_ylabel('Whole answer + EOS (%)')
        figure.suptitle('MMReD Vision: supported-answer length extrapolation', fontsize=15, x=.06, ha='left')
        figure.legend(handles=[Patch(facecolor=colors[a], label=label) for a, label in
                      (('ce', 'Native CE'), ('consistency', 'Native CE + residual consistency'))],
                      loc='upper center', bbox_to_anchor=(.53, .945), ncol=2, frameon=False)
        figure.text(.06, .025, 'Maximum training length: 16 frames. All K0–16 values appear in adaptation; this does not test unseen answer values.\n'
                    'K0–8: cross-length pairs; K9–15: same-length pairs; K16: identity pairs. Labels copy existing correct/total.',
                    fontsize=8, color='#424a52')
        figure.subplots_adjust(left=.08, right=.98, top=.85, bottom=.14, hspace=.42, wspace=.17)
        paths = []
        for extension in ('png', 'pdf'):
            path = directory/f'native_vision_v10.{extension}'
            figure.savefig(path, dpi=300, facecolor='white', bbox_inches='tight')
            paths.append(dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size))
        plt.close(figure)
    return dict(files=paths, matplotlib_version=matplotlib.__version__, row_count=len(rows),
        no_new_estimation=True, labels='Existing correct/n; All K is total and other three categories form its disjoint partition',
        intervals='Use the unchanged independent report')


def parse_accounting(raw):
    rows=[];identities=set()
    for line in raw.splitlines():
        fields=line.split('|')
        if fields and fields[-1]=='':fields.pop()
        need(len(fields)==8, 'Scheduler accounting field count differs')
        job_raw,job_id,name,state,elapsed,tres,start,end=(field.strip() for field in fields)
        if not re.search(r'(?:^|[_-])v10(?:[_-]|$)',name,re.I):continue
        need(job_raw not in identities, 'Duplicate scheduler allocation row');identities.add(job_raw)
        allocations={}
        for component in tres.split(','):
            if '=' in component:
                key,value=component.split('=',1);allocations[key]=value
        gpu_value=allocations.get('gres/gpu')
        typed=[int(value) for key,value in allocations.items() if key.startswith('gres/gpu:')]
        gpus=int(gpu_value) if gpu_value is not None else sum(typed)
        need(gpus>=0 and elapsed.isdigit(), 'Invalid allocated GPU count or elapsed seconds')
        seconds=int(elapsed);state=state.split()[0].rstrip('+')
        if gpus:
            need(state in TERMINAL,
                 'V10 GPU allocation is not terminal; accounting is incomplete')
        row=dict(job_id_raw=job_raw,job_id=job_id,job_name=name,state=state,allocated_gpus=gpus,
            elapsed_seconds=seconds,allocated_gpu_seconds=seconds*gpus,allocation_tres=tres,start=start,end=end)
        rows.append(row)
    need(rows, 'Scheduler returned no V10 jobs')
    return rows


def accounting_metrics(rows):
    total=sum(row['allocated_gpu_seconds'] for row in rows);events=[]
    for row in rows:
        if row['allocated_gpus'] and row['elapsed_seconds']:
            need(row['start'] not in ('Unknown','None','') and row['end'] not in ('Unknown','None',''), 'GPU interval timestamps missing')
            a,b=datetime.fromisoformat(row['start']),datetime.fromisoformat(row['end'])
            need(a<=b, 'GPU allocation interval reversed')
            events.extend(((a,row['allocated_gpus']),(b,-row['allocated_gpus'])))
    # End events precede starts at the same scheduler timestamp.
    active=peak=0
    for stamp,delta in sorted(events,key=lambda item:(item[0],item[1])):
        active+=delta;peak=max(peak,active)
    need(active==0, 'Allocation intervals do not close')
    return dict(allocated_gpu_seconds=total,allocated_gpu_hours=total/3600,
        gpu_seconds_cap=GPU_SECONDS_CAP,budget_passed=total<=GPU_SECONDS_CAP,
        maximum_concurrent_v10_gpus=peak,concurrency_passed=peak<=4,
        passed=total<=GPU_SECONDS_CAP and peak<=4,
        retained_noncompleted_attempts=[row for row in rows if row['state'] in TERMINAL and row['state']!='COMPLETED'],
        allocation_count=sum(row['allocated_gpus']>0 for row in rows),
        scope='All scheduler allocations with the dedicated v10 job-name token, including failed retries. Concurrency covers V10; other project campaigns remain the root scheduler responsibility.')


def scheduler_accounting(analysis,post,out):
    required={};bindings={}
    def add(directory,phase,expected_digest=None):
        directory=Path(directory);path=directory/'summary.json'
        if expected_digest is not None:bind(path,expected_digest)
        summary=read(path);bindings[str(path)]=sha(path)
        job=str(summary.get('slurm_job_id',directory.name.rsplit('_',1)[-1]))
        need(job.isdigit() and job not in required, 'Missing or duplicated expected GPU allocation identity')
        required[job]=dict(phase=phase,summary_file=str(path),summary_sha256=sha(path))
    cache_ref=analysis['cache'];bind(cache_ref['file'],cache_ref['sha256']);cache=read(cache_ref['file'])
    bindings[cache_ref['file']]=cache_ref['sha256']
    add(cache['profile_directory'],'cache_profile',cache['profile_summary_sha256'])
    for shard in cache['shards']:add(shard['directory'],'cache_shard_'+str(shard['shard']),shard['summary_sha256'])
    release_ref=analysis['release'];bind(release_ref['file'],release_ref['sha256']);release=read(release_ref['file'])
    bindings[release_ref['file']]=release_ref['sha256']
    for condition,item in release['profiles'].items():add(item['directory'],'training_profile_'+condition,item['summary_sha256'])
    for name,run in analysis['runs'].items():add(run['run_directory'],'main_'+name,run['summary_sha256'])
    job=str(post.get('slurm_job_id',Path(post['files']['replay.json']['path']).parent.name.rsplit('_',1)[-1]))
    need(job.isdigit() and job not in required, 'Post-audit GPU allocation identity missing or duplicated')
    required[job]=dict(phase='selected_checkpoint_native_audit')
    need(len(required)==12, 'Require one cache profile, four shards, two training profiles, four mains and one selected audit')
    command=['sacct','-X','-n','-P','-S','2026-09-11T00:00:00','-u',getpass.getuser(),
        '--format=JobIDRaw,JobID,JobName%120,State,ElapsedRaw,AllocTRES%240,Start,End']
    environment=dict(os.environ,TZ='UTC',SLURM_TIME_FORMAT='standard')
    result=subprocess.run(command,check=True,capture_output=True,text=True,env=environment,timeout=60)
    rows=parse_accounting(result.stdout)
    lookup={alias:row for row in rows for alias in (row['job_id_raw'],row['job_id'])}
    for job,item in required.items():
        need(job in lookup and lookup[job]['allocated_gpus']==1 and lookup[job]['state']=='COMPLETED',
             'A required completed one-GPU allocation is absent from scheduler accounting: '+job)
        item['accounted_job_id_raw']=lookup[job]['job_id_raw']
    metrics=accounting_metrics(rows)
    # Preserve verbatim selected scheduler records, including zero-allocation
    # failed/cancelled attempts. No batch/extern rows are requested (-X).
    selected_raw='\n'.join(line for line in result.stdout.splitlines()
        if len(line.split('|'))>=3 and re.search(r'(?:^|[_-])v10(?:[_-]|$)',line.split('|')[2],re.I))+'\n'
    (out/'slurm_accounting.psv').write_text(selected_raw)
    value=dict(schema_version=1,command=command,raw_file=str(out/'slurm_accounting.psv'),raw_sha256=sha(out/'slurm_accounting.psv'),
        required_allocations=required,artifact_bindings=bindings,rows=rows,**metrics)
    save(out/'gpu_accounting.json',value)
    return value


def self_test():
    analysis=dict(passed=True,audit_passed=True,primary_both_seeds=True,practical_both_seeds=True,vision_milestone_gate=True)
    report=dict(passed=True)
    post=dict(completed=True,computational_integrity_passed=True,native_replay_gate_passed=True,passed=True,
        strict_cache_numerical_gate_passed=False,original_mixed_numerical_gate_passed=False)
    accounting=dict(passed=True,budget_passed=True,concurrency_passed=True)
    need(acceptance(analysis,report,post,accounting)['accepted_vision_milestone'],'Descriptive cache failure incorrectly blocks engineering verification')
    for key in ('completed','computational_integrity_passed','native_replay_gate_passed','passed'):
        need(not acceptance(analysis,report,dict(post,**{key:False}),accounting)['accepted_vision_milestone'],'Mandatory audit failure ignored')
    need(not acceptance(analysis,report,post,dict(accounting,passed=False))['accepted_vision_milestone'],'GPU budget verification ignored')
    failed=dict(analysis,primary_both_seeds=False,practical_both_seeds=False,vision_milestone_gate=False)
    need(acceptance(failed,report,post,accounting)['verification_passed'] and not acceptance(failed,report,post,accounting)['accepted_vision_milestone'],'Verification rescued failed efficacy')
    analysis['runs']={};post['runs']=[]
    zero=lambda n:dict(n=n,correct=0,accuracy=0.)
    for condition in CONDITIONS:
        for seed in SEEDS:
            name=f'{condition}_s{seed}'
            metrics=dict(all=zero(272),cells={cell:zero(n) for cell,n in CELLS},
                by_n_k={f'main_test_N{n}/K{k}':zero(8) for n in (32,64) for k in range(17)},
                partitions={f'N{n}':{k:zero(size) for k,size in (*PARTITIONS,('K9_16',64),('nonzero',128))} for n in (32,64)})
            analysis['runs'][name]=dict(condition=condition,seed=seed,run_directory='/fixture/'+name,
                selected_step=918,selected_checkpoint='/fixture/'+name,selected_checkpoint_sha256=name,selected_parameter_sha256=name,
                verified_test_examples=272,verified_target_positions=177120,verified_scene_presentations=73440,
                verified_pair_presentations=36720,verified_optimizer_log_rows=4590,metrics=metrics)
            post['runs'].append(dict(condition=condition,seed=seed,run_id=name,selected_step=918,checkpoint='/fixture/'+name,
                checkpoint_sha256=name,selected_parameter_sha256=name))
    need(len(match_selected(analysis,post))==4 and len(plotted_rows(analysis))==32,'Four models or fixed32 plotted cells missing')
    changed=copy.deepcopy(post);changed['runs'][0]['selected_parameter_sha256']='different'
    try:match_selected(analysis,changed)
    except ValueError:pass
    else:raise AssertionError('Mismatched selected parameters accepted')
    for edit in ('denominator','partition'):
        changed=copy.deepcopy(analysis);m=changed['runs']['ce_s14']['metrics']
        if edit=='denominator':m['cells']['main_test_N32']['n']=135
        else:m['partitions']['N64']['K16'].update(correct=1,accuracy=1/8)
        try:plotted_rows(changed)
        except ValueError:pass
        else:raise AssertionError('Changed count partition/denominator accepted')
    zero_decision=dict(n64_additional_correct=0,n32_additional_correct=0,primary=False,
        consistency_n32_correct=0,consistency_n64_correct=0,consistency_larger_k_n64_correct=0,practical=False)
    analysis.update(decisions={str(seed):dict(zero_decision) for seed in SEEDS},
        primary_both_seeds=False,practical_both_seeds=False,vision_milestone_gate=False)
    need(registered_decisions(analysis)==analysis['decisions'],'Registered decisions are not reproduced from raw counts')
    changed=copy.deepcopy(analysis);changed['decisions']['14']['primary']=True
    try:registered_decisions(changed)
    except ValueError:pass
    else:raise AssertionError('Changed per-seed efficacy decision accepted')
    raw='1|1|v10_profile|COMPLETED|10|gres/gpu=1,gres/gpu:b200=1|2026-09-11T01:00:00|2026-09-11T01:00:10\n2|2|native_v10_train|FAILED|20|gres/gpu=1|2026-09-11T01:00:00|2026-09-11T01:00:20\n3|3|other|COMPLETED|999|gres/gpu=1|2026-09-11T01:00:00|2026-09-11T01:16:39\n'
    measured=accounting_metrics(parse_accounting(raw))
    need(measured['allocated_gpu_seconds']==30 and measured['maximum_concurrent_v10_gpus']==2
         and len(measured['retained_noncompleted_attempts'])==1,'Failed allocation accounting or typed-TRES handling differs')
    at_limit=[dict(job_id_raw='4',state='COMPLETED',allocated_gpus=1,elapsed_seconds=16200,allocated_gpu_seconds=16200,
        start='2026-09-11T01:00:00',end='2026-09-11T05:30:00')]
    need(accounting_metrics(at_limit)['passed'],'Inclusive16200-second cap failed')
    at_limit[0].update(elapsed_seconds=16201,allocated_gpu_seconds=16201)
    need(not accounting_metrics(at_limit)['passed'],'Exceeded GPU budget accepted')
    overlapping=[dict(at_limit[0],job_id_raw=str(i),allocated_gpu_seconds=1,elapsed_seconds=1,
        start='2026-09-11T01:00:00',end='2026-09-11T01:00:01') for i in range(5)]
    need(not accounting_metrics(overlapping)['concurrency_passed'],'More than four simultaneous V10 GPUs accepted')
    return dict(passed=True,tests=['mandatory_report_post_and_resource_gates','descriptive_cache_failure_retained',
        'failed_efficacy_not_rescued','four_selected_models_bound','reject_checkpoint_mismatch','fixed32_plot_cells_and_count_partitions',
        'scheduler_failed_attempts_counted','typed_GPU_TRES_not_double_counted','inclusive16200_second_budget',
        'registered_per_seed_decisions_rederived','four_GPU_concurrency_limit'])


def finalize(args, out):
    source_check = args.source_check.resolve(); checked = read(source_check/'summary.json')
    need(checked['passed'] and checked['source_sha256'] == sources(), 'Matching prospective finalizer source check required')
    source_chain(source_check, sources(), 'source')
    analysis_path = args.analysis.resolve(); analysis = read(analysis_path); report_dir = analysis_path.parent
    report_summary = read(report_dir/'summary.json')
    need(report_summary['analysis_sha256'] == sha(analysis_path)
         and Path(report_summary['analysis_file']).resolve() == analysis_path, 'Independent report/analysis binding differs')
    source_chain(report_dir, analysis['source_sha256'], 'code')
    need(report_summary['source_sha256'] == analysis['source_sha256'], 'Report source summaries differ')
    for key in ('primary_both_seeds', 'practical_both_seeds', 'vision_milestone_gate'):
        need(report_summary[key] == analysis[key], 'Reported efficacy booleans differ')
    need(analysis['interpretation']['does_not_establish_reasoning_composition'] is True, 'Unexpected reasoning-composition claim')
    post_path = args.post_audit.resolve(); post = read(post_path); post_dir = post_path.parent
    source_chain(post_dir, post['source_sha256'], 'source')
    bind(post['plan_file'], post['plan_sha256'])
    plan_path = Path(post['plan_file'])
    need(plan_path.with_suffix('.sha256').read_text().strip() == post['plan_sha256'], 'Post-audit plan sidecar differs')
    post_plan = read(plan_path)
    need(post_plan['source_sha256'] == post['source_sha256'], 'Post-audit plan source differs')
    for item in post['files'].values(): bind(item['path'], item['sha256'])
    stored_replays = read(post['files']['replay.json']['path'])
    stored_cache = read(post['files']['comparisons.json']['path'])
    need(len(stored_replays) == 40 and len(stored_cache) == 656
         and [row for row in stored_replays if not row['numerical_rule_passed']] == post['replay_failures']
         and [row for row in stored_cache if not row['numerical_rule_passed']] == post['cache_numerical_failures'],
         'Post audit omitted or changed a stored numerical failure')
    identities = match_selected(analysis, post)
    for identity in identities: bind(identity['checkpoint'], identity['checkpoint_sha256'])
    rows = plotted_rows(analysis); registered_decisions(analysis)
    accounting = scheduler_accounting(analysis, post, out)
    gates = acceptance(analysis, report_summary, post, accounting)
    need(gates['post_strict_cache_numerical_gate_passed'] == (not post['cache_numerical_failures'])
         and post['native_replay_gate_passed'] == (not post['replay_failures']), 'Post numerical summaries disagree')
    save(out/'plot_input.json', dict(analysis_file=str(analysis_path), analysis_sha256=sha(analysis_path), rows=rows))
    plots = plot(rows, out)
    bound = dict(analysis=dict(path=str(analysis_path), sha256=sha(analysis_path)),
        independent_report_summary=dict(path=str(report_dir/'summary.json'), sha256=sha(report_dir/'summary.json')),
        selected_checkpoint_audit=dict(path=str(post_path), sha256=sha(post_path)),
        gpu_accounting=dict(path=str(out/'gpu_accounting.json'), sha256=sha(out/'gpu_accounting.json')),
        checkpoint_audit_plan=dict(path=str(plan_path), sha256=sha(plan_path)),
        prospective_source_check=dict(path=str(source_check/'summary.json'), sha256=sha(source_check/'summary.json')))
    summary = dict(schema_version=1, completed=True, **gates, source_sha256=sources(), artifacts=bound,
        selected_models=identities, reported_decisions=analysis['decisions'], plot=plots,
        plot_input_sha256=sha(out/'plot_input.json'), cache_numerical_failures=post['cache_numerical_failures'],
        native_replay_failures=post['replay_failures'],
        allocated_gpu_seconds=accounting['allocated_gpu_seconds'], allocated_gpu_hours=accounting['allocated_gpu_hours'],
        gpu_seconds_cap=GPU_SECONDS_CAP, retained_noncompleted_attempts=accounting['retained_noncompleted_attempts'],
        scope='Artifact finalization only; no new predictions, fits, metrics, intervals or selection',
        independent_report_interpretation=analysis['interpretation'])
    save(out/'final_acceptance.json', summary)
    text = ['# V10 final artifact', '',
        f"Report and checkpoint verification: **{gates['verification_passed']}**.",
        f"Reported both-seed primary/practical: **{gates['primary_both_seeds']} / {gates['practical_both_seeds']}**.",
        f"Accepted vision milestone: **{gates['accepted_vision_milestone']}**.", '',
        '![Existing per-cell native accuracy](native_vision_v10.png)', '',
        '[PDF figure](native_vision_v10.pdf) · [Acceptance and provenance](final_acceptance.json)', '',
        f"[Independent report]({report_dir/'REPORT.md'}) · [Selected-checkpoint audit]({post_dir/'REPORT.md'})", '',
        'Figures copy the reported metrics; labels retain every denominator. Descriptive cache failures remain recorded.',
        f"V10 allocated compute: {accounting['allocated_gpu_seconds']} GPU-seconds / {GPU_SECONDS_CAP}; all failed attempts remain in the accounting.",
        'All tested answer values occur in adaptation. This artifact does not establish unseen-answer transfer or reasoning composition.', '']
    (out/'REPORT.md').write_text('\n'.join(text))
    return dict(passed=gates['verification_passed'], finalized=True,
        accepted_vision_milestone=gates['accepted_vision_milestone'], final_acceptance_file=str(out/'final_acceptance.json'),
        final_acceptance_sha256=sha(out/'final_acceptance.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true'); parser.add_argument('--source-check', type=Path)
    parser.add_argument('--analysis', type=Path); parser.add_argument('--post-audit', type=Path)
    args = parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION') == 'cpu'
         and not os.environ.get('SLURM_JOB_GPUS'), 'CPU Slurm allocation required')
    begin = time.perf_counter(); mode = 'selftest' if args.self_test else 'final'
    out = OUT/f'{mode}_{os.environ["SLURM_JOB_ID"]}'; out.mkdir(parents=True, exist_ok=False); snapshot(out)
    frozen = sources()
    if args.self_test: result = dict(passed=True, tests=self_test())
    else:
        need(all((args.source_check, args.analysis, args.post_audit)), 'Need frozen source check, analysis.json and post-audit summary.json')
        result = finalize(args, out)
    need(sources() == frozen, 'Finalizer source changed during execution')
    result.update(source_sha256=frozen, seconds=time.perf_counter()-begin, slurm_job_id=os.environ['SLURM_JOB_ID'])
    save(out/'summary.json', result)
    (out/'INDEX.md').write_text('# V10 finalization\n\n[Summary](summary.json) · [Source hashes](source_hashes.json)\n')
    print(json.dumps(dict(directory=str(out), **result)), flush=True)
    if not result['passed']: raise SystemExit('Final verification failed; plots and all failed gates remain recorded')


if __name__ == '__main__': main()

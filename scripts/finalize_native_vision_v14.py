"""Plot existing V14 metrics and bind final acceptance to both completed audits.

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
from collections import Counter
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
OUT = REPO/'outputs/native_aggregation_vlm/v14/finalization'
OWN = ('scripts/finalize_native_vision_v14.py', 'slurm/native_vision_v14_finalize.sbatch')
CELLS = (('main_test_N32', 136), ('main_test_N64', 136))
PARTITIONS = (('K0_8', 72), ('K9_15', 56), ('K16', 8))
CONDITIONS = ('offset', 'centered')
SEEDS = (18, 19)
GPU_SECONDS_CAP = 16200
ANCESTORS = ('scripts/finalize_native_vision_v13.py', 'slurm/native_vision_v13_finalize.sbatch')
ACCOUNTING_FIELDS = ('JobIDRaw','JobID','JobName','Partition','State','ElapsedRaw','AllocTRES','ExitCode','Submit','Start','End')
JOB_CAPS = {'v14_train_profile':300, 'v14_main':2700, 'v14_selected':300}
POST_CALLS = dict(model=46, visual=26, language=46, norm=46, last_block=46)
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


def sources(): return {name: sha(REPO/name) for name in (*OWN, *ANCESTORS)}


def snapshot(out):
    (out/'source').mkdir()
    for name in sources(): (out/'source'/name.replace('/', '_')).write_bytes((REPO/name).read_bytes())
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
                 and run['verified_pair_presentations'] == 36720 and run['verified_optimizer_log_rows'] == 4590
                 and run['verified_student_presentations'] == 512000
                 and run['verified_student_optimizer_log_rows'] == 8000
                 and run['fixed_core_through_student'] is True and run['one_dev_only'] is True
                 and run['selected_step'] == 4590 and run['selected_student_step'] == 8000,
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
        a = analysis['runs'][f'centered_s{seed}']['metrics']; b = analysis['runs'][f'offset_s{seed}']['metrics']
        n32 = a['cells']['main_test_N32']['correct']; n64 = a['cells']['main_test_N64']['correct']
        d32 = n32 - b['cells']['main_test_N32']['correct']; d64 = n64 - b['cells']['main_test_N64']['correct']
        larger = a['partitions']['N64']['K9_16']['correct']; primary = d64 >= 7 and d32 >= -6
        decisions[str(seed)] = dict(n64_additional_correct=d64, n32_additional_correct=d32, primary=primary,
            centered_n32_correct=n32, centered_n64_correct=n64, centered_larger_k_n64_correct=larger,
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
    need(set(rows) == {(condition, seed) for condition in CONDITIONS for seed in SEEDS},
         'Duplicate or missing post-audit model')
    identities = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            reported, audited = analysis['runs'][f'{condition}_s{seed}'], rows[condition, seed]
            need(Path(reported['run_directory']).name == audited['run_id']
                 and Path(reported['selected_checkpoint']).resolve() == Path(audited['checkpoint']).resolve()
                 and reported['selected_step'] == audited['selected_step'] == 4590
                 and reported['selected_student_step'] == audited['selected_student_step'] == 8000
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
    need(type(post['strict_cache_numerical_gate_passed']) is bool and post['original_mixed_numerical_gate_passed'] is False,
         'Preserved numerical status differs')
    return dict(report_audit_passed=analysis['passed'] and analysis['audit_passed'] and report_summary['passed'],
        post_completed=post['completed'], post_computational_integrity_passed=post['computational_integrity_passed'],
        post_native_replay_gate_passed=post['native_replay_gate_passed'], verification_passed=verified,
        gpu_accounting_passed=accounting['passed'], gpu_budget_passed=accounting['budget_passed'],
        v14_concurrency_passed=accounting['concurrency_passed'],
        gpu_per_job_caps_passed=accounting['per_job_caps_passed'], gpu_subcaps_passed=accounting['subcaps_passed'],
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
    colors = dict(offset='#66717e', centered='#1465ac')
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
                      (('offset', 'Offset: r − b'), ('centered', 'Centered: r − N b'))],
                      loc='upper center', bbox_to_anchor=(.53, .945), ncol=2, frameon=False)
        figure.text(.06, .025, 'Maximum training length: 16 frames. All K0–16 values appear in adaptation; this does not test unseen answer values.\n'
                    'Both arms: exact-bank core training, then frozen-core mean distillation. Labels copy existing correct/total.',
                    fontsize=8, color='#424a52')
        figure.subplots_adjust(left=.08, right=.98, top=.85, bottom=.14, hspace=.42, wspace=.17)
        paths = []
        for extension in ('png', 'pdf'):
            path = directory/f'native_vision_v14.{extension}'
            figure.savefig(path, dpi=300, facecolor='white', bbox_inches='tight')
            paths.append(dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size))
        plt.close(figure)
    return dict(files=paths, matplotlib_version=matplotlib.__version__, row_count=len(rows),
        no_new_estimation=True, labels='Existing correct/n; All K is total and other three categories form its disjoint partition',
        intervals='Use the unchanged independent report')


def parse_accounting(raw):
    rows=[]; seen=set()
    for line in raw.splitlines():
        if not line.strip(): continue
        values=line.split('|')
        need(len(values)==len(ACCOUNTING_FIELDS), 'Scheduler accounting field count differs')
        value=dict(zip(ACCOUNTING_FIELDS,(v.strip() for v in values)))
        if value['Partition']!='gpu' or not value['JobName'].startswith('v14_'): continue
        job=value['JobIDRaw']; state=value['State'].split()[0].rstrip('+')
        need(re.fullmatch(r'[0-9]+',job) and re.fullmatch(r'[0-9]+(?:_[0-9]+)?',value['JobID'])
             and job not in seen, 'Nonallocation or duplicate scheduler row')
        need(state in TERMINAL and value['ElapsedRaw'].isdigit(), 'Unfinished or malformed V14 allocation')
        allocation=dict(part.split('=',1) for part in value['AllocTRES'].split(',') if '=' in part)
        generic=allocation.get('gres/gpu'); typed=[v for k,v in allocation.items() if k.startswith('gres/gpu:')]
        need((generic is None or generic.isdigit()) and all(v.isdigit() for v in typed), 'Invalid GPU TRES')
        gpus=int(generic) if generic is not None else sum(map(int,typed))
        need(gpus in (0,1), 'Registered jobs allocate at most one GPU')
        need(value['JobName'] in JOB_CAPS, 'Unregistered V14 GPU job name: '+value['JobName'])
        elapsed=int(value['ElapsedRaw']); seen.add(job)
        rows.append(dict(value, state=state, allocated_gpus=gpus, elapsed_seconds=elapsed,
            allocated_gpu_seconds=gpus*elapsed, seconds_cap=JOB_CAPS[value['JobName']],
            per_job_cap_passed=gpus==0 or elapsed<=JOB_CAPS[value['JobName']]))
    need(rows, 'Scheduler returned no V14 GPU allocations')
    return rows


def accounting_metrics(rows):
    total=sum(row['allocated_gpu_seconds'] for row in rows); events=[]
    for row in rows:
        if row['allocated_gpus'] and row['elapsed_seconds']:
            need(row['Start'] not in ('Unknown','None','') and row['End'] not in ('Unknown','None',''), 'GPU interval timestamps missing')
            a,b=datetime.fromisoformat(row['Start']),datetime.fromisoformat(row['End'])
            need(a<b, 'Nonpositive GPU allocation interval')
            events.extend(((a,row['allocated_gpus']),(b,-row['allocated_gpus'])))
    active=peak=0
    for stamp,delta in sorted(events,key=lambda item:(item[0],item[1])):
        active+=delta; peak=max(peak,active)
    need(active==0, 'Allocation intervals do not close')
    per_job=all(r['per_job_cap_passed'] for r in rows)
    subcaps=True  # No V14 feature harvest or separate runtime software jobs.
    return dict(allocated_gpu_seconds=total,allocated_gpu_hours=total/3600,
        gpu_seconds_cap=GPU_SECONDS_CAP,budget_passed=total<=GPU_SECONDS_CAP,
        maximum_concurrent_v14_gpus=peak,concurrency_passed=peak<=4,
        per_job_caps_passed=per_job,reused_v13_inputs_not_recharged=True,subcaps_passed=subcaps,
        passed=total<=GPU_SECONDS_CAP and peak<=4 and per_job and subcaps,
        retained_noncompleted_attempts=[r for r in rows if r['state']!='COMPLETED' or r['ExitCode']!='0:0'],
        allocation_count=sum(r['allocated_gpus']>0 for r in rows),
        scope='All user v14_ GPU allocations since 2026-09-01, including failed and zero-allocation attempts. Concurrency is campaign-wide; other project campaigns remain the root scheduler responsibility.')


def scheduler_accounting(analysis,post,out):
    required={}; bindings={}
    def add(directory,phase,job_name,expected_digest=None):
        directory=Path(directory); path=directory/'summary.json'
        if expected_digest is not None: bind(path,expected_digest)
        summary=read(path); bindings[str(path)]=sha(path); job=str(summary['slurm_job_id'])
        need(job.isdigit() and job not in required and summary.get('passed') is True,
             'Missing, failed or duplicated successful allocation artifact')
        required[job]=dict(phase=phase,job_name=job_name,seconds_cap=JOB_CAPS[job_name],summary_file=str(path),summary_sha256=sha(path))
    cache_ref=analysis['cache']; bind(cache_ref['file'],cache_ref['sha256']); cache=read(cache_ref['file'])
    need(cache['protocol']=='v13_training_null_union_cache' and cache['complete'] and len(cache['features'])==53322,
         'Wrong learned-null union cache')
    bindings[cache_ref['file']]=cache_ref['sha256']
    # Bound V13 cache and software inputs are reused; no new extraction charge.
    release_ref=analysis['release']; bind(release_ref['file'],release_ref['sha256']); release=read(release_ref['file'])
    need(release['protocol']=='v14_exact_bank_distillation_main_release' and release['passed'] and set(release['profiles'])==set(CONDITIONS),
         'Wrong measured main release')
    bindings[release_ref['file']]=release_ref['sha256']
    software=release_ref['native_software']; bind(software['file'],software['sha256'])
    bindings[software['file']]=software['sha256']
    for condition,item in release['profiles'].items(): add(item['directory'],'training_profile_'+condition,'v14_train_profile',item['summary_sha256'])
    for name,run in analysis['runs'].items(): add(run['run_directory'],'main_'+name,'v14_main',run['summary_sha256'])
    job=str(post['slurm_job_id']); need(job.isdigit() and job not in required, 'Post GPU allocation missing or duplicated')
    required[job]=dict(phase='selected_checkpoint_native_audit',job_name='v14_selected',seconds_cap=300)
    need(len(required)==7, 'Require two training profiles, four mains and one final-checkpoint audit')
    command=['sacct','-X','--noheader','--parsable2','--starttime','2026-09-01','--user',getpass.getuser(),
        '--format',','.join(name+('%120' if name=='JobName' else '%240' if name=='AllocTRES' else '') for name in ACCOUNTING_FIELDS)]
    environment=dict(os.environ,TZ='UTC',SLURM_TIME_FORMAT='standard')
    save(out/'accounting_request.json',dict(command=command,scope='Independent all-user discovery; retain every v14_ GPU allocation'))
    result=subprocess.run(command,capture_output=True,text=True,env=environment,timeout=60)
    (out/'slurm_all_user.psv').write_text(result.stdout); (out/'slurm_accounting.stderr.txt').write_text(result.stderr)
    need(result.returncode==0, 'Scheduler accounting request failed; stdout/stderr retained')
    rows=parse_accounting(result.stdout)
    lookup={alias:r for r in rows for alias in (r['JobIDRaw'],r['JobID'])}
    for job,item in required.items():
        need(job in lookup and lookup[job]['allocated_gpus']==1 and lookup[job]['state']=='COMPLETED'
             and lookup[job]['ExitCode']=='0:0' and lookup[job]['elapsed_seconds']>0
             and lookup[job]['JobName']==item['job_name'], 'Required successful one-GPU allocation missing: '+job)
        item['accounted_job_id_raw']=lookup[job]['JobIDRaw']
    need(len({i['accounted_job_id_raw'] for i in required.values()})==7, 'Successful artifact roles reused one allocation')
    # Earlier failed allocations frozen in the release must still be present.
    prior=release['campaign_budget']['allocations']
    for old in prior:
        need(old['JobIDRaw'] in lookup and all(lookup[old['JobIDRaw']][k]==old[k] for k in ACCOUNTING_FIELDS),
             'A pre-main allocation disappeared or changed after release')
    selected_raw='\n'.join('|'.join(r[k] for k in ACCOUNTING_FIELDS) for r in rows)+'\n'
    (out/'slurm_accounting.psv').write_text(selected_raw)
    value=dict(schema_version=1,command=command,raw_file=str(out/'slurm_accounting.psv'),raw_sha256=sha(out/'slurm_accounting.psv'),
        all_user_raw_file=str(out/'slurm_all_user.psv'),all_user_raw_sha256=sha(out/'slurm_all_user.psv'),
        required_allocations=required,artifact_bindings=bindings,rows=rows,**accounting_metrics(rows))
    save(out/'gpu_accounting.json',value)
    return value


def audit_post(post,plan):
    need(post['protocol']==plan['protocol']=='all_selected_v14_native_checkpoint_audit'
         and post['calls']==plan['calls']==POST_CALLS and post['native_head_replay_calls']==46
         and post['cache_full_rows']==656 and post['fixed_prefix_kv_checks']==24 and post['selected_fusion_checks']==40
         and post['ordinary_generations']==post['extra_last_block_forwards']==0
         and post['cached_full_numeric_is_descriptive'] is True and post['no_fitting'] is True
         and post['frozen_backbone_unchanged'] is True and post['selected_parameters_unchanged'] is True,
         'Selected audit protocol or native call inventory differs')
    files=post['files']; need(files==post['artifacts'], 'Post file ledgers differ')
    replays=read(files['head_replay_metrics.json']['path']); cache=read(files['cached_full_comparisons.json']['path'])
    forwards=read(files['forwards.json']['path']); structural=read(files['structural_checks.json']['path'])
    for row in forwards: bind(row['path'],row['sha256'])
    return audit_post_records(post, replays, cache, forwards, structural,
                              read(files['calls.json']['path']), read(files['actual_calls.json']['path']))


def audit_post_records(post, replays, cache, forwards, structural, calls, actual_calls):
    need(len(replays)==len(forwards)==46 and len(cache)==656 and calls==POST_CALLS
         and actual_calls==dict(calls=POST_CALLS,standalone_head_replays=46), 'Stored post counters differ')
    for row in replays+cache:
        need(type(row['numerical_rule_passed']) is bool and type(row['top1_equal']) is bool
             and math.isfinite(row['full_vocabulary_tv']) and 0<=row['full_vocabulary_tv']<=1
             and row['numerical_rule_passed']==(row['top1_equal'] and row['full_vocabulary_tv']<=.02), 'Post numeric decision differs')
    need([r for r in replays if not r['numerical_rule_passed']]==post['replay_failures']
         and [r for r in cache if not r['numerical_rule_passed']]==post['cache_full_failures'], 'Stored post failure lists differ')
    runs=post['runs']; ids={r['run_id'] for r in runs}
    need(len(ids)==len(runs)==4 and post['models']==runs, 'Post model list differs')
    need(Counter((r['run_id'],r['n_frames'],r['step'],r['row_index']) for r in cache)==Counter(
         (rid,n,t,i) for rid in ids for n in (16,64) for t in (1,2) for i in range(n+1)), 'Missing/duplicate cached-full rows')
    need(all(r['binding'] is False for r in cache), 'Cached-full diagnostics relabeled as binding')
    checks=structural['checks']; controls=structural['controller_counts']; restores=structural['selected_state_restores']
    need(len(checks)==40 and len(controls)==len(restores)==8
         and Counter(r['run_id'] for r in checks)==Counter({rid:10 for rid in ids})
         and sum(r['all_layer_kv_exact'] is None for r in checks)==16
         and sum(isinstance(r['all_layer_kv_exact'],list) for r in checks)==24
         and all(r['passed'] is True and r['native_hidden_exact'] is True
             and (r['all_layer_kv_exact'] is None or
                  (isinstance(r['all_layer_kv_exact'],list) and len(r['all_layer_kv_exact'])==28
                   and all(value is True for value in r['all_layer_kv_exact']))) for r in checks)
         and all(r['passed'] is True and r['calls']==5 for r in controls)
         and Counter(r['run_id'] for r in controls)==Counter({rid:2 for rid in ids}), 'Post fusion/KV/closure evidence incomplete')
    for run in runs:
        rid=run['run_id']; native=[r for r in replays if f'__{rid}__' in r['label']]; cached=[r for r in cache if r['run_id']==rid]
        need(len(native)==run['head_replays']==10 and len(cached)==run['cache_full_rows']==164
             and run['fixed_prefix_kv_checks']==6 and run['computational_integrity_passed'] is True
             and run['replay_gate_passed']==all(r['numerical_rule_passed'] for r in native)
             and run['cache_full_numeric_passed']==all(r['numerical_rule_passed'] for r in cached)
             and sum(r['run_id']==rid and r['passed'] is True and r['parameter_sha256']==run['selected_parameter_sha256'] for r in restores)==2,
             'Per-model audit or combined checkpoint restore differs')
    need(sum('__native__' in r['label'] for r in replays)==6, 'Six bare-native reference head replays missing')
    return dict(native_head_replays=46,selected_head_replays=40,bare_native_head_replays=6,cache_full_comparisons=656,
        exact_selected_kv_checks=24,selected_fusion_checks=40,all_forward_artifacts_hash_verified=True)


def self_test():
    analysis=dict(passed=True,audit_passed=True,primary_both_seeds=True,practical_both_seeds=True,vision_milestone_gate=True)
    report=dict(passed=True)
    post=dict(completed=True,computational_integrity_passed=True,native_replay_gate_passed=True,passed=True,
        strict_cache_numerical_gate_passed=False,original_mixed_numerical_gate_passed=False)
    accounting=dict(passed=True,budget_passed=True,concurrency_passed=True,per_job_caps_passed=True,subcaps_passed=True)
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
                selected_step=4590,selected_student_step=8000,selected_checkpoint='/fixture/'+name,selected_checkpoint_sha256=name,selected_parameter_sha256=name,
                verified_test_examples=272,verified_target_positions=177120,verified_scene_presentations=73440,
                verified_pair_presentations=36720,verified_student_presentations=512000,verified_optimizer_log_rows=4590,
                verified_student_optimizer_log_rows=8000,fixed_core_through_student=True,one_dev_only=True,metrics=metrics)
            post['runs'].append(dict(condition=condition,seed=seed,run_id=name,selected_step=4590,selected_student_step=8000,checkpoint='/fixture/'+name,
                checkpoint_sha256=name,selected_parameter_sha256=name))
    need(len(match_selected(analysis,post))==4 and len(plotted_rows(analysis))==32,'Four models or fixed32 plotted cells missing')
    changed=copy.deepcopy(post);changed['runs'][0]['selected_parameter_sha256']='different'
    try:match_selected(analysis,changed)
    except ValueError:pass
    else:raise AssertionError('Mismatched selected parameters accepted')
    for edit in ('denominator','partition'):
        changed=copy.deepcopy(analysis);m=changed['runs']['offset_s18']['metrics']
        if edit=='denominator':m['cells']['main_test_N32']['n']=135
        else:m['partitions']['N64']['K16'].update(correct=1,accuracy=1/8)
        try:plotted_rows(changed)
        except ValueError:pass
        else:raise AssertionError('Changed count partition/denominator accepted')
    zero_decision=dict(n64_additional_correct=0,n32_additional_correct=0,primary=False,
        centered_n32_correct=0,centered_n64_correct=0,centered_larger_k_n64_correct=0,practical=False)
    analysis.update(decisions={str(seed):dict(zero_decision) for seed in SEEDS},
        primary_both_seeds=False,practical_both_seeds=False,vision_milestone_gate=False)
    need(registered_decisions(analysis)==analysis['decisions'],'Registered decisions are not reproduced from raw counts')
    changed=copy.deepcopy(analysis);changed['decisions']['18']['primary']=True
    try:registered_decisions(changed)
    except ValueError:pass
    else:raise AssertionError('Changed per-seed efficacy decision accepted')
    def decision_fixture(n32=123,n64=109,larger=52,d32=-6,d64=7):
        value=copy.deepcopy(analysis)
        for seed in SEEDS:
            a=value['runs'][f'centered_s{seed}']['metrics']; b=value['runs'][f'offset_s{seed}']['metrics']
            a['cells']['main_test_N32']['correct']=n32; a['cells']['main_test_N64']['correct']=n64
            a['partitions']['N64']['K9_16']['correct']=larger
            b['cells']['main_test_N32']['correct']=n32-d32; b['cells']['main_test_N64']['correct']=n64-d64
            primary=d64>=7 and d32>=-6; practical=primary and n32>=123 and n64>=109 and larger>=52
            value['decisions'][str(seed)]=dict(n64_additional_correct=d64,n32_additional_correct=d32,primary=primary,
                centered_n32_correct=n32,centered_n64_correct=n64,centered_larger_k_n64_correct=larger,practical=practical)
        value.update(primary_both_seeds=primary,practical_both_seeds=practical,vision_milestone_gate=practical)
        return value
    edge=decision_fixture(); need(all(x['practical'] for x in registered_decisions(edge).values()), 'Inclusive integer decision boundary failed')
    for changes in ({'n32':122},{'n64':108},{'larger':51},{'d32':-7},{'d64':6}):
        need(not any(x['practical'] for x in registered_decisions(decision_fixture(**changes)).values()), 'Below-threshold decision accepted')
    # Real post schema:46 heads include6 bare references, all4x164 cache rows.
    replay=[]; cache=[]; checks=[]; controls=[]; restores=[]; models=[]
    metric=dict(numerical_rule_passed=True,top1_equal=True,full_vocabulary_tv=0.)
    for condition in CONDITIONS:
        for seed in SEEDS:
            rid=f'{condition}_s{seed}'
            models.append(dict(run_id=rid,head_replays=10,cache_full_rows=164,fixed_prefix_kv_checks=6,
                computational_integrity_passed=True,replay_gate_passed=True,cache_full_numeric_passed=True,selected_parameter_sha256=rid))
            for n in (16,64):
                controls.append(dict(run_id=rid,passed=True,calls=5)); restores.append(dict(run_id=rid,passed=True,parameter_sha256=rid))
                for i in range(5):
                    replay.append(dict(metric,label=f'N{n}__{rid}__step{i}'))
                    checks.append(dict(run_id=rid,passed=True,native_hidden_exact=True,all_layer_kv_exact=[True]*28 if i<3 else None))
                cache.extend(dict(metric,run_id=rid,n_frames=n,step=t,row_index=i,binding=False)
                             for t in (1,2) for i in range(n+1))
    replay.extend(dict(metric,label=f'N{n}__native__cached{t}') for n in (16,64) for t in range(3))
    fixture=dict(runs=models,models=models,replay_failures=[],cache_full_failures=[])
    structural=dict(checks=checks,controller_counts=controls,selected_state_restores=restores)
    actual=dict(calls=POST_CALLS,standalone_head_replays=46)
    need(audit_post_records(fixture,replay,cache,[{}]*46,structural,POST_CALLS,actual)['cache_full_comparisons']==656,
         'Complete post schema rejected')
    changed=copy.deepcopy(cache); changed[0].update(numerical_rule_passed=False,full_vocabulary_tv=.021)
    reported=copy.deepcopy(fixture); reported['cache_full_failures']=[changed[0]]
    for row in reported['runs']:
        if row['run_id']==changed[0]['run_id']: row['cache_full_numeric_passed']=False
    audit_post_records(reported,replay,changed,[{}]*46,structural,POST_CALLS,actual)
    for bad_layers in ([True]*27, [True]*27+[False], [True]*27+[1], True):
        bad_structure=copy.deepcopy(structural)
        bad_structure['checks'][0]['all_layer_kv_exact']=bad_layers
        try: audit_post_records(fixture,replay,cache,[{}]*46,bad_structure,POST_CALLS,actual)
        except ValueError: pass
        else: raise AssertionError('Missing, unequal, nonboolean or scalar native KV layer flags accepted')
    for bad_cache in (cache[:-1],cache[:-1]+[cache[0]],changed):
        try: audit_post_records(fixture,replay,bad_cache,[{}]*46,structural,POST_CALLS,actual)
        except ValueError: pass
        else: raise AssertionError('Missing/duplicate/suppressed post numerical evidence accepted')
    def record(job,name='v14_main',elapsed=10,start='2026-09-11T01:00:00',end='2026-09-11T01:00:10',state='COMPLETED',tres='gres/gpu=1',exitcode='0:0'):
        return f'{job}|{job}|{name}|gpu|{state}|{elapsed}|{tres}|{exitcode}|2026-09-11T00:00:00|{start}|{end}\n'
    raw=record(1,tres='gres/gpu=1,gres/gpu:b200=1')+record(2,elapsed=20,state='FAILED',exitcode='1:0')
    raw+=record(3,name='other',elapsed=999)
    measured=accounting_metrics(parse_accounting(raw))
    need(measured['allocated_gpu_seconds']==30 and measured['maximum_concurrent_v14_gpus']==2
         and len(measured['retained_noncompleted_attempts'])==1, 'Failed allocation or typed TRES handling differs')
    row=parse_accounting(record(1,elapsed=2700,end='2026-09-11T01:45:00'))[0]
    # Six nonoverlapping legal-duration allocations exercise the aggregate boundary.
    limits=[dict(row,JobIDRaw=str(i),Start=f'2026-09-11T{i:02d}:00:00',End=f'2026-09-11T{i:02d}:45:00') for i in range(6)]
    need(accounting_metrics(limits)['passed'], 'Inclusive16200-second campaign cap failed')
    too_many=limits+[dict(row,elapsed_seconds=1,allocated_gpu_seconds=1,Start='2026-09-11T07:00:00',End='2026-09-11T07:00:01')]
    need(not accounting_metrics(too_many)['budget_passed'], 'Exceeded campaign cap accepted')
    over=parse_accounting(record(1,elapsed=2701,end='2026-09-11T01:45:01'))
    need(not accounting_metrics(over)['per_job_caps_passed'], 'Per-main cap ignored')
    overlapping=[dict(row,JobIDRaw=str(i)) for i in range(5)]
    need(not accounting_metrics(overlapping)['concurrency_passed'], 'More than four simultaneous GPUs accepted')
    for key,bad in (('verified_student_presentations',511999),('verified_student_optimizer_log_rows',7999),
                    ('selected_student_step',7999),('selected_step',4589),('fixed_core_through_student',False),('one_dev_only',False)):
        changed=copy.deepcopy(analysis);changed['runs']['offset_s18'][key]=bad
        try:plotted_rows(changed)
        except ValueError:pass
        else:raise AssertionError('Incomplete or adaptive two-phase fit accepted')
    for malformed in (record(1,state='RUNNING'),record(1,name='v14_unknown'),record(1,name='v14_native_profile'),record(1,tres='gres/gpu=2'),record(1)+record(1)):
        try: parse_accounting(malformed)
        except ValueError: pass
        else: raise AssertionError('Invalid/unfinished/duplicate campaign allocation accepted')
    return dict(passed=True,tests=['mandatory_report_post_and_resource_gates','descriptive_cache_failure_retained',
        'failed_efficacy_not_rescued','four_combined_selected_models_bound','reject_checkpoint_mismatch','fixed32_plot_cells_and_count_partitions',
        'scheduler_failed_attempts_counted','typed_GPU_TRES_not_double_counted','inclusive16200_second_budget',
        'registered_per_seed_decisions_rederived','four_GPU_concurrency_limit','actual_per_job_caps','fixed_complete_two_phase_training_required',
        'reject_unknown_unfinished_duplicate_allocations','integer_threshold_boundaries','complete46_head656_cache_schema',
        'reject_missing_duplicate_or_suppressed_post_rows','exact28_layer_KV_lists_and16_full_reference_None',
        'reject_missing_false_nonboolean_or_scalar_KV_flags'])


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
    need(post_plan['policy']==analysis['policy'], 'Post audit and independent report use different training policy')
    for path,digest in post_plan['artifact_sha256'].items(): bind(path,digest)
    replay_audit = audit_post(post, post_plan)
    identities = match_selected(analysis, post)
    for identity in identities: bind(identity['checkpoint'], identity['checkpoint_sha256'])
    for run in analysis['runs'].values():
        directory=Path(run['run_directory']); bind(directory/'config.json',run['config_sha256'])
        bind(directory/'summary.json',run['summary_sha256'])
        config=read(directory/'config.json')
        need(config['policy']==analysis['policy'] if 'policy' in config else post_plan['policy']==analysis['policy'],
             'Main/report policy differs')
        need(config['condition']==run['condition'] and config['seed']==run['seed']
             and all(config[key]==post_plan[key] for key in ('model','runtime','processor')), 'Selected model/runtime identity differs')
    rows = plotted_rows(analysis); registered_decisions(analysis)
    accounting = scheduler_accounting(analysis, post, out)
    gates = acceptance(analysis, report_summary, post, accounting)
    need(gates['post_strict_cache_numerical_gate_passed'] == (not post['cache_full_failures'])
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
        selected_models=identities, reported_decisions=analysis['decisions'], plot=plots, post_evidence_audit=replay_audit,
        plot_input_sha256=sha(out/'plot_input.json'), cache_numerical_failures=post['cache_full_failures'],
        native_replay_failures=post['replay_failures'],
        allocated_gpu_seconds=accounting['allocated_gpu_seconds'], allocated_gpu_hours=accounting['allocated_gpu_hours'],
        gpu_seconds_cap=GPU_SECONDS_CAP, retained_noncompleted_attempts=accounting['retained_noncompleted_attempts'],
        scope='Artifact finalization only; no new predictions, fits, metrics, intervals or selection',
        independent_report_interpretation=analysis['interpretation'])
    save(out/'final_acceptance.json', summary)
    text = ['# V14 final artifact', '',
        f"Report and checkpoint verification: **{gates['verification_passed']}**.",
        f"Reported both-seed primary/practical: **{gates['primary_both_seeds']} / {gates['practical_both_seeds']}**.",
        f"Accepted vision milestone: **{gates['accepted_vision_milestone']}**.", '',
        '![Existing per-cell native accuracy](native_vision_v14.png)', '',
        '[PDF figure](native_vision_v14.pdf) · [Acceptance and provenance](final_acceptance.json)', '',
        f"[Independent report]({report_dir/'REPORT.md'}) · [Selected-checkpoint audit]({post_dir/'summary.json'})", '',
        'Figures copy the reported metrics; labels retain every denominator. Descriptive cache failures remain recorded.',
        f"V14 allocated compute: {accounting['allocated_gpu_seconds']} GPU-seconds / {GPU_SECONDS_CAP}; all failed attempts remain in the accounting.",
        'Both arms use the same 1,060,224 parameters and training-only null calibration; inference uses N+1 streams without reference rows. All tested answer values occur in adaptation. This artifact does not establish unseen-answer transfer or reasoning composition.', '']
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
    (out/'INDEX.md').write_text('# V14 finalization\n\n[Summary](summary.json) · [Source hashes](source_hashes.json)\n')
    print(json.dumps(dict(directory=str(out), **result)), flush=True)
    if not result['passed']: raise SystemExit('Final verification failed; plots and all failed gates remain recorded')


if __name__ == '__main__': main()

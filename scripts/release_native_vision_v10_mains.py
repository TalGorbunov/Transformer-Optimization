"""CPU-only V10 measured release, with complete declared prior GPU accounting.

The job list is a prospective operator-supplied campaign inventory, not inferred
from successful outputs. Failed allocations remain in the budget. sacct is read
with -X so batch/extern steps cannot double count allocation time.
"""
from pathlib import Path
import argparse
import json
import math
import os
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from scripts import train_native_vision_v10 as train
from scripts.stage_native_vision_v7_features import need, read, sha

OWN = ('scripts/release_native_vision_v10_mains.py', 'slurm/native_vision_v10_main_release.sbatch')
TIMING = REPO/'outputs/native_aggregation_vlm/v7/main_runtime_extension_441884/release.json'
FIELDS = ('JobIDRaw', 'JobID', 'JobName', 'Partition', 'State', 'ElapsedRaw',
          'AllocTRES', 'ExitCode', 'Submit', 'Start', 'End')
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def positive(value, name):
    need(isinstance(value, (int, float)) and not isinstance(value, bool)
         and math.isfinite(value) and value > 0, 'Invalid positive timing: '+name)
    return float(value)


def parse_accounting(raw, requested):
    need(requested and len(requested) == len(set(requested))
         and all(re.fullmatch(r'[0-9]+(?:_[0-9]+)?', x) for x in requested),
         'Require distinct explicit allocation/array-task IDs')
    rows = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        values = line.split('|')
        need(len(values) == len(FIELDS), 'Unexpected sacct columns')
        row = dict(zip(FIELDS, values))
        need(re.fullmatch(r'[0-9]+(?:_[0-9]+)?', row['JobIDRaw'])
             and re.fullmatch(r'[0-9]+(?:_[0-9]+)?', row['JobID']),
             'Accounting must contain allocation rows only')
        state = row['State'].split()[0].rstrip('+')
        need(state in TERMINAL, 'Unfinished or unknown allocation state: '+row['State'])
        need(row['Partition'] == 'gpu' and row['JobName'].startswith('v10_'),
             'Non-V10 GPU allocation in the declared campaign list')
        need(row['ElapsedRaw'].isdigit(), 'Malformed allocation elapsed seconds')
        tres = dict(item.split('=', 1) for item in row['AllocTRES'].split(',') if '=' in item)
        count = tres.get('gres/gpu')
        if count is None:
            typed = [int(v) for k, v in tres.items() if k.startswith('gres/gpu:')]
            count = sum(typed)
        need(str(count) == '1', 'Expected one allocated GPU per registered V10 job')
        row.update(state=state, gpu_count=1, allocated_gpu_seconds=int(row['ElapsedRaw']))
        need(row['End'] not in ('', 'Unknown', 'None') and row['Start'] not in ('', 'Unknown', 'None'),
             'Missing allocation start/end times')
        rows.append(row)
    need(rows and len({r['JobIDRaw'] for r in rows}) == len(rows),
         'Empty or duplicate allocation accounting')
    aliases = {r[k] for r in rows for k in ('JobIDRaw', 'JobID')}
    need(set(requested) <= aliases, 'A requested job is missing; expand array parents to explicit tasks')
    need(all(any(x in (r['JobIDRaw'], r['JobID']) for x in requested) for r in rows),
         'sacct returned unrequested allocations')
    need(len(rows) == len(requested), 'Duplicate aliases or unexpanded array job request')
    return rows


def measured_projection(load_seconds, steady_seconds, times):
    load = positive(load_seconds, 'load')
    steady = positive(steady_seconds, 'shared maximum steady step')
    t16 = positive(times['16'], 'N16'); t64 = positive(times['64'], 'N64')
    need(t64 >= t16, 'Malformed fixed native timing bound')
    training = 4590*steady; development = 320*t16; test = 272*t64
    projected = load + 1.25*(training+development+test) + 120
    return dict(projected_seconds=projected, passed=projected <= 2700,
                training_seconds=training, development_seconds=development, test_seconds=test,
                generation_seconds=times, N32_uses_N64_bound=True,
                shared_step_seconds_max_steady=steady)


def self_test():
    raw = ('42|42|v10_features_profile|gpu|COMPLETED|30|cpu=4,gres/gpu=1,gres/gpu:b200=1|0:0|s|s|e\n'
           '43|41_1|v10_train_profile|gpu|FAILED|7|gres/gpu=1|1:0|s|s|e\n')
    rows = parse_accounting(raw, ['42', '41_1'])
    need(sum(r['allocated_gpu_seconds'] for r in rows) == 37, 'Failures or typed GPU double counted')
    for text, ids in [(raw, ['42', '43', '41_1']), (raw.replace('FAILED', 'RUNNING'), ['42', '43']),
                      (raw.replace('gres/gpu=1|1:', 'gres/gpu=2|1:'), ['42', '43']),
                      (raw+'43.batch|43.batch|x|gpu|COMPLETED|7|gres/gpu=1|0:0|s|s|e\n', ['42', '43'])]:
        try:
            parse_accounting(text, ids)
        except ValueError:
            pass
        else:
            raise AssertionError('Malformed or double-counted accounting accepted')
    p = measured_projection(5, .1, {'16': 1., '64': 2.})
    need(p['training_seconds'] == 459 and p['development_seconds'] == 320
         and p['test_seconds'] == 544 and p['projected_seconds'] == 1778.75,
         'Registered projection arithmetic differs')
    need(5100 + 4*2700 + 300 == 16200 and 5101 + 4*2700 + 300 > 16200,
         'Campaign boundary arithmetic differs')
    return dict(passed=True, tests=7)


def ledger(mapping):
    need(mapping, 'Empty source/artifact ledger')
    for name, digest in mapping.items():
        path = Path(name); path = path if path.is_absolute() else REPO/path
        need(sha(path) == digest, 'Bound source/artifact changed: '+name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--profiles', type=Path, nargs=2)
    parser.add_argument('--data-release', type=Path)
    parser.add_argument('--prior-gpu-jobs', nargs='+')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION') == 'cpu'
         and not os.environ.get('SLURM_JOB_GPUS'), 'CPU Slurm required')
    out = train.OUT/f'main_release_{os.environ["SLURM_JOB_ID"]}'
    out.mkdir(parents=True, exist_ok=False); (out/'source').mkdir()
    own = {name: sha(REPO/name) for name in OWN}
    for name in OWN:
        (out/'source'/name.replace('/', '_')).write_bytes((REPO/name).read_bytes())
    train.save(out/'source_hashes.json', own)
    (out/'INDEX.md').write_text('# V10 measured main release\n\n[Decision](release.json) · [Tests](self_test.json) · [Sources](source_hashes.json) · [GPU accounting](gpu_accounting.json)\n')
    train.save(out/'self_test.json', self_test())
    if args.self_test:
        print(json.dumps(dict(passed=True, output=str(out))), flush=True); return
    need(args.plan and args.profiles and args.data_release and args.prior_gpu_jobs,
         'Require plan, both profiles, independent data release and complete prior GPU job inventory')
    args.plan = args.plan.resolve(); args.data_release = args.data_release.resolve()
    plan = train.verify_plan(args.plan)
    data = read(args.data_release)
    need(data['passed'] is True, 'Completed independent data/report release required')
    ledger(data['source_sha256']); ledger(data['data_bindings'])
    need(sha(data['audit_file']) == data['audit_sha256'], 'Independent data audit changed')
    old = read(TIMING); oldplan = read(old['plan_file'])
    need(old['passed'] is True and sha(old['plan_file']) == old['plan_sha256'], 'Timing ancestor changed')
    need(Path(oldplan['native_profile']).resolve() == Path(plan['native_profile']['directory']).resolve()
         and oldplan['native_profile_sha256'] == plan['native_profile']['sha256']
         and sha(plan['native_profile']['file']) == plan['native_profile']['sha256'], 'Different native timing profile')
    for name in ('scripts/native_vision_v7_runtime.py', 'gnnformer/parallel_local_native.py',
                 'gnnformer/parallel_local_aggregation.py', 'gnnformer/parallel_local_prompts.py'):
        need(old['source_sha256'][name] == plan['source_sha256'][name] == sha(REPO/name),
             'Deployed native computation changed')
    extension = old['resource_extension']; ledger(extension['source_sha256'])
    need(sha(extension['original_release_file']) == extension['original_release_sha256'], 'Prior timing failure changed')
    original = read(extension['original_release_file'])
    need(extension['scope'] == 'resource_only_1800_to_2700_seconds'
         and extension['original_passed'] is False and original['passed'] is False
         and extension['no_model_data_optimizer_selection_or_decision_change'] is True,
         'Original failed timing release must remain explicit')
    times = old['projections']['parallel']['generation_seconds']
    need(old['projections']['parallel']['original_1800_second_passed'] is False
         and times == original['projections']['parallel']['generation_seconds'], 'Native timing lineage differs')
    old_profile_binding = old['profiles']['parallel']
    old_profile_path = Path(old_profile_binding['directory'])/'summary.json'
    need(sha(old_profile_path) == old_profile_binding['summary_sha256'], 'Timing hardware reference changed')
    old_profile = read(old_profile_path)
    profiles = {}; summaries = {}; successful = []
    for directory in args.profiles:
        directory = directory.resolve(); path = directory/'summary.json'; s = read(path); condition = s['condition']
        need(condition in train.POLICY['conditions'] and condition not in profiles, 'Duplicate or incorrect profile')
        need(s['profile'] is True and s['completed'] is True and s['arm'] == 'parallel' and s['seed'] == 14 and s['passed'] is True
             and s['steps'] == 32 and s['no_dev_or_test_evaluation'] is True
             and s['computational_integrity_passed'] is True and s['active_cache_replay']['passed'] is True
             and s['active_cache_replay']['counts_covered'] == [0, 9, 10, 16]
             and s['active_cache_replay']['prefixes_checked'] == 10
             and s['plan_sha256'] == sha(args.plan) and s['source_sha256'] == train.sources(),
             'Failed or unmatched full-prefix training profile')
        zero = s['zero_initialization_auxiliary_gradients']
        need(all(zero.get(key) is True for key in ('passed', 'up_exactly_zero',
                  'all_residual_gradients_zero', 'all_path_gradients_zero'))
             and zero['residual_loss'] == zero['path_loss'] == 0., 'Zero-U auxiliary profile gate failed')
        need(s['all_parameter_gradients_exercised']
             and all(x is True for x in s['all_parameter_gradients_exercised'].values())
             and s['frozen_backbone_gradient_state_preserved'] is True
             and s['checkpoint_roundtrip_passed'] is True, 'Training gradient/restore gate failed')
        need(sha(s['profile_checkpoint']) == s['profile_checkpoint_sha256']
             and sha(s['training_file']) == s['training_sha256'], 'Profile checkpoint or training ledger changed')
        ledger(s['source_sha256'])
        need(s['hardware'] == old_profile['hardware'], 'Timing hardware/backend differs')
        positive(s['step_seconds_max_steady'], 'profile steady step')
        profiles[condition] = dict(directory=str(directory), summary_sha256=sha(path))
        summaries[condition] = s
        successful.append(dict(role='training_profile_'+condition, job_id=str(s['slurm_job_id']),
                               file=str(path), sha256=sha(path), seconds_cap=180))
    need(set(profiles) == set(train.POLICY['conditions']), 'Both conditions required')
    maximum = max(s['step_seconds_max_steady'] for s in summaries.values())
    projections = {condition: measured_projection(s['model_and_features_load_seconds'], maximum, times)
                   for condition, s in summaries.items()}
    cache_binding = plan['cache_binding']; need(sha(cache_binding['file']) == cache_binding['sha256'], 'Feature cache changed')
    cache = read(cache_binding['file'])
    feature_profile_path = Path(cache['profile_directory'])/'summary.json'
    need(sha(feature_profile_path) == cache['profile_summary_sha256'], 'Feature profile changed')
    feature_profile = read(feature_profile_path)
    need(feature_profile['profile'] is True and feature_profile['passed'] is True
         and feature_profile['completed'] is True and feature_profile['computational_integrity_passed'] is True
         and feature_profile['runtime_projection_passed'] is True
         and feature_profile['plan_sha256'] == cache['plan_sha256']
         and len(feature_profile['shard_projected_seconds']) == 4
         and all(0 < x <= 600 and math.isfinite(x) for x in feature_profile['shard_projected_seconds']),
         'Feature cache profile failed or differs')
    successful.append(dict(role='feature_profile', job_id=str(feature_profile['slurm_job_id']),
                           file=str(feature_profile_path), sha256=sha(feature_profile_path), seconds_cap=180))
    seen_shards = set()
    for binding in cache['shards']:
        path = Path(binding['directory'])/'summary.json'; need(sha(path) == binding['summary_sha256'], 'Feature shard changed')
        s = read(path); shard = s['shard']
        need(s['completed'] is True and s['passed'] is True and s['computational_integrity_passed'] is True
             and s['profile'] is False and shard == binding['shard'] and shard not in seen_shards
             and s['plan_sha256'] == cache['plan_sha256'], 'Incomplete or duplicate feature shard')
        seen_shards.add(shard)
        successful.append(dict(role='feature_shard_'+str(shard), job_id=str(s['slurm_job_id']),
                               file=str(path), sha256=sha(path), seconds_cap=600))
    need(seen_shards == set(range(4)), 'Exactly four feature shards required')
    command = ['sacct', '-X', '--noheader', '--parsable2', '--jobs', ','.join(args.prior_gpu_jobs),
               '--format', ','.join(FIELDS)]
    train.save(out/'accounting_request.json', dict(command=command, requested_jobs=args.prior_gpu_jobs,
               scope='All prior V10 GPU allocations including failed attempts, explicitly supplied by campaign operator'))
    proc = subprocess.run(command, text=True, capture_output=True, check=False, timeout=60)
    (out/'sacct.txt').write_text(proc.stdout); (out/'sacct.stderr.txt').write_text(proc.stderr)
    need(proc.returncode == 0, 'sacct failed; preserve raw output')
    rows = parse_accounting(proc.stdout, args.prior_gpu_jobs)
    successful_ids = set()
    for item in successful:
        matches = [r for r in rows if item['job_id'] in (r['JobIDRaw'], r['JobID'])]
        need(len(matches) == 1, 'Successful artifact job missing from allocation inventory: '+item['role'])
        row = matches[0]
        need(row['JobIDRaw'] not in successful_ids, 'One allocation cannot represent multiple successful jobs')
        successful_ids.add(row['JobIDRaw'])
        need(row['state'] == 'COMPLETED' and row['ExitCode'] == '0:0'
             and row['allocated_gpu_seconds'] <= item['seconds_cap'], 'Successful job failed its allocation cap')
        item['accounting_job_id'] = row['JobIDRaw']
    prior = sum(r['allocated_gpu_seconds'] for r in rows)
    campaign = dict(passed=prior+4*2700+300 <= 16200, prior_allocated_gpu_seconds=prior,
                    reserved_main_gpu_seconds=4*2700, reserved_post_gpu_seconds=300,
                    reserved_total_gpu_seconds=prior+4*2700+300, campaign_gpu_seconds_cap=16200,
                    max_concurrent_project_gpus=4, remaining_after_reserved_seconds=16200-prior-4*2700-300,
                    complete_inventory_is_operator_declared=True, requested_jobs=args.prior_gpu_jobs,
                    allocations=rows, successful_artifacts=successful,
                    sacct_file=str(out/'sacct.txt'), sacct_sha256=sha(out/'sacct.txt'))
    train.save(out/'gpu_accounting.json', campaign)
    result = dict(passed=campaign['passed'] and all(p['passed'] for p in projections.values()),
        plan_file=str(args.plan), plan_sha256=sha(args.plan), source_sha256=train.sources(),
        release_source_sha256=own, profiles=profiles, projections=projections,
        shared_step_seconds_max_steady=maximum,
        data_release=dict(file=str(args.data_release), sha256=sha(args.data_release)),
        report_source_sha256=data['source_sha256'], per_main_seconds_cap=2700,
        main_block_gpu_seconds_cap=16200, campaign_budget=campaign,
        native_timing_ancestor=dict(file=str(TIMING), sha256=sha(TIMING), plan_file=old['plan_file'],
            plan_sha256=old['plan_sha256'], native_profile_file=plan['native_profile']['file'],
            native_profile_sha256=plan['native_profile']['sha256'],
            scope='Unchanged native parallel inference; prior measured four-token-plus-CPU-preparation bound, no prior efficacy used'))
    train.save(out/'release.json', result)
    if result['passed']:
        train.verify_release(out/'release.json', args.plan, plan)
    ledger(own); ledger(data['source_sha256'])
    print(json.dumps(dict(passed=result['passed'], release_file=str(out/'release.json'),
                         projections=projections, prior_allocated_gpu_seconds=prior)), flush=True)
    if not result['passed']:
        raise SystemExit('V10 runtime or campaign cap failed; preserve this release')


if __name__ == '__main__':
    main()

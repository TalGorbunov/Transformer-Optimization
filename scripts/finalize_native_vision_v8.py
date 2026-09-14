"""Plot existing V8 metrics and bind final acceptance to both completed audits.

No new predictions, fits, thresholds, metric estimates, bootstrap draws or model
selection. Two fixed-seed columns separate familiar counts from unseen counts.
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
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
OUT = REPO/'outputs/native_aggregation_vlm/v8/finalization'
OWN = ('scripts/finalize_native_vision_v8.py', 'slurm/native_vision_v8_finalize.sbatch')
CELLS = (('main_test_N16', 108), ('main_test_N32', 108), ('main_test_N64', 108),
         ('count_test_N32', 64), ('count_test_N64', 64))


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


def plotted_rows(analysis):
    rows = []
    need(set(analysis['runs']) == {f'{condition}_s{seed}' for condition in ('ce', 'consistency') for seed in (10, 11)},
         'Require exactly the four registered reported runs')
    for seed in (10, 11):
        for condition in ('ce', 'consistency'):
            run = analysis['runs'][f'{condition}_s{seed}']
            need(run['condition'] == condition and run['seed'] == seed, 'Reported condition/seed differs')
            for cell, denominator in CELLS:
                metric = run['metrics']['cells'][cell]
                need(metric['n'] == denominator and isinstance(metric['correct'], int)
                     and 0 <= metric['correct'] <= denominator and math.isfinite(metric['accuracy'])
                     and math.isclose(metric['accuracy'], metric['correct']/denominator, abs_tol=1e-12, rel_tol=0),
                     'Reported cell count/accuracy is inconsistent')
                rows.append(dict(condition=condition, seed=seed, cell=cell, n=metric['n'], correct=metric['correct'],
                                 accuracy=metric['accuracy']))
    return rows


def match_selected(analysis, post):
    need(len(post['runs']) == 4, 'Post audit must retain all four selected models')
    rows = {(row['condition'], row['seed']): row for row in post['runs']}
    need(set(rows) == {(condition, seed) for condition in ('ce', 'consistency') for seed in (10, 11)},
         'Duplicate or missing post-audit model')
    identities = []
    for condition in ('ce', 'consistency'):
        for seed in (10, 11):
            reported, audited = analysis['runs'][f'{condition}_s{seed}'], rows[condition, seed]
            need(Path(reported['run_directory']).name == audited['run_id']
                 and reported['selected_step'] == audited['selected_step']
                 and reported['selected_checkpoint_sha256'] == audited['checkpoint_sha256']
                 and reported['selected_parameter_sha256'] == audited['selected_parameter_sha256'],
                 'Report and post audit evaluated different selected parameters')
            identities.append(dict(condition=condition, seed=seed, run_id=audited['run_id'], selected_step=audited['selected_step'],
                checkpoint=audited['checkpoint'], checkpoint_sha256=audited['checkpoint_sha256'],
                parameter_sha256=audited['selected_parameter_sha256']))
    return identities


def acceptance(analysis, report_summary, post):
    flags = [analysis['passed'], analysis['audit_passed'], report_summary['passed'], post['completed'],
             post['computational_integrity_passed'], post['native_replay_gate_passed'], post['passed'],
             analysis['primary_both_seeds'], analysis['practical_both_seeds'], analysis['vision_milestone_gate']]
    need(all(isinstance(flag, bool) for flag in flags), 'Final acceptance fields must be explicit booleans')
    need(analysis['vision_milestone_gate'] == (analysis['primary_both_seeds'] and analysis['practical_both_seeds']),
         'Reported vision decision is internally inconsistent')
    verified = all(flags[:7])
    return dict(report_audit_passed=analysis['passed'] and analysis['audit_passed'] and report_summary['passed'],
        post_completed=post['completed'], post_computational_integrity_passed=post['computational_integrity_passed'],
        post_native_replay_gate_passed=post['native_replay_gate_passed'], verification_passed=verified,
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
        figure, axes = plt.subplots(2, 2, figsize=(10, 7.2), sharey=True)
        for column, seed in enumerate((10, 11)):
            for row_index, (family, lengths, title) in enumerate((
                ('main', (16, 32, 64), 'Familiar counts K0–8'),
                ('count', (32, 64), 'Unseen counts K9–16'))):
                axis = axes[row_index, column]
                for arm_index, condition in enumerate(('ce', 'consistency')):
                    points = [next(row for row in rows if row['condition'] == condition and row['seed'] == seed
                                   and row['cell'] == f'{family}_test_N{n}') for n in lengths]
                    x = [index + (arm_index-.5)*.34 for index in range(len(lengths))]
                    bars = axis.bar(x, [100*point['accuracy'] for point in points], width=.31, color=colors[condition], zorder=3)
                    for bar, point in zip(bars, points):
                        axis.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.8,
                                  f"{point['correct']}/{point['n']}", ha='center', va='bottom', fontsize=8, color=colors[condition])
                axis.set_title(f'{title} · seed {seed}', loc='left', fontsize=11)
                axis.set_xticks(range(len(lengths)), [str(n) for n in lengths])
                axis.set_xlabel('Number of frames')
                axis.set_ylim(0, 110); axis.set_yticks((0, 20, 40, 60, 80, 100))
                axis.grid(axis='y', color='#dfe3e7', linewidth=.65, zorder=0)
                if column == 0: axis.set_ylabel('Exact count + EOS (%)')
        figure.suptitle('MMReD Vision: V8 native evaluation', fontsize=15, x=.06, ha='left')
        figure.legend(handles=[Patch(facecolor=colors[a], label=label) for a, label in
                      (('ce', 'Paired CE'), ('consistency', 'Paired CE + consistency'))],
                      loc='upper center', bbox_to_anchor=(.53, .945), ncol=2, frameon=False)
        figure.text(.06, .025, 'Labels: correct / total. Exact requires EOS within four tokens. No new intervals or metrics estimated.\n'
                    'These two fitted seeds do not establish reasoning composition; uncertainty is retained in the independent report.',
                    fontsize=8, color='#424a52')
        figure.subplots_adjust(left=.08, right=.98, top=.85, bottom=.14, hspace=.42, wspace=.17)
        paths = []
        for extension in ('png', 'pdf'):
            path = directory/f'native_vision_v8.{extension}'
            figure.savefig(path, dpi=300, facecolor='white', bbox_inches='tight')
            paths.append(dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size))
        plt.close(figure)
    return dict(files=paths, matplotlib_version=matplotlib.__version__, row_count=len(rows),
                no_new_estimation=True, labels='Existing report correct/n', intervals='Use the unchanged independent report')


def self_test():
    analysis = dict(passed=True, audit_passed=True, primary_both_seeds=True, practical_both_seeds=True, vision_milestone_gate=True)
    report = dict(passed=True)
    post = dict(completed=True, computational_integrity_passed=True, native_replay_gate_passed=True, passed=True,
                strict_cache_numerical_gate_passed=False, original_mixed_numerical_gate_passed=False)
    need(acceptance(analysis, report, post)['accepted_vision_milestone'], 'Descriptive cache failure incorrectly blocks engineering release')
    for key in ('completed', 'computational_integrity_passed', 'native_replay_gate_passed', 'passed'):
        need(not acceptance(analysis, report, dict(post, **{key: False}))['accepted_vision_milestone'], 'A mandatory audit failure was ignored')
    failed = dict(analysis, primary_both_seeds=False, practical_both_seeds=False, vision_milestone_gate=False)
    need(acceptance(failed, report, post)['verification_passed']
         and not acceptance(failed, report, post)['accepted_vision_milestone'], 'Verification rescued failed efficacy')
    analysis['runs'] = {}; post['runs'] = []
    for condition in ('ce', 'consistency'):
        for seed in (10, 11):
            name = f'{condition}_s{seed}'
            analysis['runs'][name] = dict(condition=condition, seed=seed, run_directory='/fixture/'+name,
                selected_step=972, selected_checkpoint_sha256=name, selected_parameter_sha256=name,
                metrics=dict(cells={cell: dict(n=n, correct=0, accuracy=0.) for cell, n in CELLS}))
            post['runs'].append(dict(condition=condition, seed=seed, run_id=name, selected_step=972, checkpoint='/fixture/'+name,
                                    checkpoint_sha256=name, selected_parameter_sha256=name))
    need(len(match_selected(analysis, post)) == 4 and len(plotted_rows(analysis)) == 20, 'Plot/checkpoint coverage differs')
    changed = copy.deepcopy(post); changed['runs'][0]['selected_parameter_sha256'] = 'different'
    try: match_selected(analysis, changed)
    except ValueError: pass
    else: raise AssertionError('A mismatched selected checkpoint was accepted')
    changed = copy.deepcopy(analysis); changed['runs']['ce_s10']['metrics']['cells']['main_test_N16']['n'] = 107
    try: plotted_rows(changed)
    except ValueError: pass
    else: raise AssertionError('Plot silently changed a denominator')
    return dict(passed=True, tests=['mandatory_report_post_gates', 'descriptive_cache_failure_retained',
        'failed_efficacy_not_rescued', 'four_selected_models_bound', 'reject_checkpoint_mismatch', 'fixed_twenty_cells_denominators'])


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
    rows = plotted_rows(analysis); gates = acceptance(analysis, report_summary, post)
    need(gates['post_strict_cache_numerical_gate_passed'] == (not post['cache_numerical_failures'])
         and post['native_replay_gate_passed'] == (not post['replay_failures']), 'Post numerical summaries disagree')
    save(out/'plot_input.json', dict(analysis_file=str(analysis_path), analysis_sha256=sha(analysis_path), rows=rows))
    plots = plot(rows, out)
    bound = dict(analysis=dict(path=str(analysis_path), sha256=sha(analysis_path)),
        independent_report_summary=dict(path=str(report_dir/'summary.json'), sha256=sha(report_dir/'summary.json')),
        selected_checkpoint_audit=dict(path=str(post_path), sha256=sha(post_path)),
        checkpoint_audit_plan=dict(path=str(plan_path), sha256=sha(plan_path)),
        prospective_source_check=dict(path=str(source_check/'summary.json'), sha256=sha(source_check/'summary.json')))
    summary = dict(schema_version=1, completed=True, **gates, source_sha256=sources(), artifacts=bound,
        selected_models=identities, reported_decisions=analysis['decisions'], plot=plots,
        plot_input_sha256=sha(out/'plot_input.json'), cache_numerical_failures=post['cache_numerical_failures'],
        native_replay_failures=post['replay_failures'],
        scope='Artifact finalization only; no new predictions, fits, metrics, intervals or selection',
        independent_report_interpretation=analysis['interpretation'])
    save(out/'final_acceptance.json', summary)
    text = ['# V8 final artifact', '',
        f"Report and checkpoint verification: **{gates['verification_passed']}**.",
        f"Reported both-seed primary/practical: **{gates['primary_both_seeds']} / {gates['practical_both_seeds']}**.",
        f"Accepted vision milestone: **{gates['accepted_vision_milestone']}**.", '',
        '![Existing per-cell native accuracy](native_vision_v8.png)', '',
        '[PDF figure](native_vision_v8.pdf) · [Acceptance and provenance](final_acceptance.json)', '',
        f"[Independent report]({report_dir/'REPORT.md'}) · [Selected-checkpoint audit]({post_dir/'REPORT.md'})", '',
        'Figures copy the reported metrics; labels retain every denominator. Descriptive cache failures remain recorded.',
        'This artifact does not establish reasoning composition.', '']
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
    (out/'INDEX.md').write_text('# V8 finalization\n\n[Summary](summary.json) · [Source hashes](source_hashes.json)\n')
    print(json.dumps(dict(directory=str(out), **result)), flush=True)
    if not result['passed']: raise SystemExit('Final verification failed; plots and all failed gates remain recorded')


if __name__ == '__main__': main()

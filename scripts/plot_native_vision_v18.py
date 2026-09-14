"""HELD CPU-only figures from a completed independent V18 report; no rescoring."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
OWN = ('scripts/plot_native_vision_v18.py', 'slurm/native_vision_v18_plot.sbatch')
MODES = ('all_open', 'native_gate')
SEEDS = (20, 21)
DECISIONS = ('primary_both_seeds', 'practical_both_seeds', 'vision_milestone_gate', 'reasoning_objective_achieved')


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def sources():
    return {name: sha(REPO / name) for name in OWN}


def bind(path, digest, bindings):
    path = Path(path).resolve()
    need(sha(path) == digest, 'File identity changed: ' + str(path))
    bindings[str(path)] = digest


def snapshot(out, ledger):
    folder = out / 'code'
    folder.mkdir(exist_ok=True)
    for name, digest in ledger.items():
        path = Path(name)
        need(not path.is_absolute() and '..' not in path.parts, 'Source path must be repository relative')
        raw = (REPO / path).read_bytes()
        need(hashlib.sha256(raw).hexdigest() == digest, 'Source changed before snapshot: ' + name)
        target = folder / name.replace('/', '_')
        need(not target.exists() or target.read_bytes() == raw, 'Source snapshot name collision')
        target.write_bytes(raw)


def metric_check(metric, denominator):
    need(type(metric['n']) is int and metric['n'] == denominator, 'Reported denominator differs')
    need(type(metric['correct']) is int and 0 <= metric['correct'] <= denominator, 'Invalid exact count')
    need(type(metric['first_token_correct']) is int and 0 <= metric['first_token_correct'] <= denominator,
         'Invalid first-token count')
    need(isinstance(metric['accuracy'], (int, float)) and math.isfinite(metric['accuracy'])
         and math.isclose(metric['accuracy'], metric['correct'] / denominator, rel_tol=0, abs_tol=1e-12),
         'Reported accuracy is inconsistent with its count')


def plotted_rows(analysis):
    need(set(analysis['runs']) == {f'{mode}_s{seed}' for mode in MODES for seed in SEEDS}, 'Require all four modes/seeds')
    rows = []
    for seed in SEEDS:
        for mode in MODES:
            run = analysis['runs'][f'{mode}_s{seed}']
            need(run['condition'] == mode and run['seed'] == seed and run['selected_step'] == 4590
                 and run['native_test_examples'] == 272 and run['one_dev_only'] is True
                 and run['dev_descriptive_only'] is True, 'Final run identity or coverage differs')
            metrics = run['metrics']; metric_check(metrics['all'], 272)
            need(set(metrics['cells']) == {'main_test_N32', 'main_test_N64'} and set(metrics['by_n_k']) ==
                 {f'main_test_N{n}/K{k}' for n in (32, 64) for k in range(17)}, 'Length/count cell inventory differs')
            for n in (32, 64):
                cell = metrics['cells'][f'main_test_N{n}']; metric_check(cell, 136)
                per_k = [metrics['by_n_k'][f'main_test_N{n}/K{k}'] for k in range(17)]
                for metric in per_k:
                    metric_check(metric, 8)
                for key in ('n', 'correct', 'first_token_correct'):
                    need(sum(metric[key] for metric in per_k) == cell[key], 'Per-count metrics do not sum to length total')
                for k, metric in [(None, cell), *enumerate(per_k)]:
                    rows.append(dict(mode=mode, seed=seed, n_frames=n, count=k, n=metric['n'],
                        correct=metric['correct'], accuracy=metric['accuracy'], first_token_correct=metric['first_token_correct']))
            for key in ('n', 'correct', 'first_token_correct'):
                need(sum(metric[key] for metric in metrics['cells'].values()) == metrics['all'][key], 'Length totals differ')
    return rows


def verify_report(summary_path, out):
    summary_path = Path(summary_path).resolve(); summary = read(summary_path); bindings = {}
    bind(summary_path, sha(summary_path), bindings)
    need(summary.get('passed') is True and summary.get('completed') is True, 'Independent report must have completed and passed')
    analysis_path = Path(summary['analysis_file']).resolve()
    need(analysis_path == summary_path.parent / 'analysis.json', 'Unexpected independent analysis path')
    bind(analysis_path, summary['analysis_sha256'], bindings); analysis = read(analysis_path)
    need(analysis.get('passed') is True and analysis.get('audit_passed') is True and analysis['schema_version'] == 1,
         'Independent analysis did not pass')
    ledger = analysis['source_sha256']
    need(ledger == summary['source_sha256'] == read(summary_path.parent / 'source_hashes.json')
         and 'scripts/report_native_vision_v18.py' in ledger and 'slurm/native_vision_v18_report.sbatch' in ledger,
         'Independent source ledger differs')
    bind(summary_path.parent / 'source_hashes.json', sha(summary_path.parent / 'source_hashes.json'), bindings)
    for name, digest in ledger.items():
        relative = Path(name)
        need(not relative.is_absolute() and '..' not in relative.parts, 'Invalid report source path')
        bind(REPO / relative, digest, bindings)
        bind(summary_path.parent / 'code' / name.replace('/', '_'), digest, bindings)
    snapshot(out, ledger)
    policy = analysis['policy']
    need(policy['protocol'] == 'v18_native_semantic_gate_final_only' and policy['conditions'] == ['native_gate', 'all_open']
         and policy['seeds'] == list(SEEDS) and policy['steps'] == 4590 and policy['test_examples'] == 272
         and policy['max_new_tokens'] == 4 and policy['test_count_support'] == list(range(17))
         and policy['maximum_training_N'] == 16, 'Wrong experiment policy')
    for key in DECISIONS:
        need(type(analysis[key]) is bool and summary[key] == analysis[key], 'Report decision copy differs: ' + key)
    need(analysis['reasoning_objective_achieved'] is False, 'No reasoning claim belongs in this plot')
    rows = plotted_rows(analysis)
    for run in analysis['runs'].values():
        directory = Path(run['directory']).resolve()
        bind(directory / 'summary.json', run['summary_sha256'], bindings)
        bind(run['test_file'], run['test_sha256'], bindings)
        endpoint = run['endpoint']
        bind(endpoint['endpoint_file'], endpoint['endpoint_sha256'], bindings)
        bind(endpoint['checkpoint'], endpoint['checkpoint_sha256'], bindings)
    report_path = Path(summary['report_file']).resolve()
    need(report_path == summary_path.parent / 'REPORT.md', 'Unexpected report text path')
    bind(report_path, sha(report_path), bindings)
    for name, path in (('report_summary.json', summary_path), ('report_analysis.json', analysis_path), ('report_text.md', report_path)):
        (out / name).write_bytes(path.read_bytes())
    return analysis, rows, bindings, ledger


def plot(rows, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    colors = {'all_open': '#77818b', 'native_gate': '#1465ac'}
    labels = {'all_open': 'All open', 'native_gate': 'Native gate'}
    files = []
    def point(mode, seed, n, k=None):
        return next(row for row in rows if (row['mode'], row['seed'], row['n_frames'], row['count']) == (mode, seed, n, k))
    def finish(figure, stem):
        for extension in ('png', 'pdf'):
            path = out / f'{stem}.{extension}'
            figure.savefig(path, dpi=220, facecolor='white', bbox_inches='tight')
            files.append(dict(file=str(path), sha256=sha(path), bytes=path.stat().st_size))
        plt.close(figure)
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42}):
        figure, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
        for axis, seed in zip(axes, SEEDS):
            for j, mode in enumerate(MODES):
                for i, n in enumerate((32, 64)):
                    row = point(mode, seed, n); x = i + (j - .5) * .34
                    whole = 100 * row['accuracy']; first = 100 * row['first_token_correct'] / row['n']
                    axis.bar(x, whole, width=.30, color=colors[mode], zorder=3)
                    axis.plot([x, x], [whole, first], ':', color=colors[mode], zorder=4)
                    axis.scatter([x], [first], marker='D', color=colors[mode], s=32, edgecolor='white', linewidth=.5, zorder=5)
                    axis.annotate(f"{row['correct']}/{row['n']}", (x, whole), xytext=(0, 5), textcoords='offset points',
                                  ha='center', va='bottom', fontsize=9, color=colors[mode])
            axis.set_title(f'Seed {seed}', loc='left'); axis.set_xticks((0, 1), ('32 frames', '64 frames'))
            axis.set_ylim(0, 112); axis.set_yticks((0, 20, 40, 60, 80, 100)); axis.grid(axis='y', alpha=.2, zorder=0)
        axes[0].set_ylabel('Correct (%)')
        figure.suptitle('V18: native whole-answer accuracy', x=.08, ha='left', fontsize=14)
        handles = [Patch(facecolor=colors[mode], label=labels[mode] + ': whole answer + EOS') for mode in MODES]
        handles.append(Line2D([], [], marker='D', color='#4b5055', linestyle=':', label='First numeral token only'))
        figure.legend(handles=handles, loc='upper center', bbox_to_anchor=(.53, .90), ncol=3, frameon=False, fontsize=8)
        figure.text(.08, .035, 'Labels: reported whole-answer correct/136. Diamonds are descriptive first-token counts/136.\n'
                    'All K0–16 values occur in training; maximum training length is 16. Four-token native EOS scoring; all failures retained.', fontsize=8)
        figure.subplots_adjust(top=.74, bottom=.20, left=.08, right=.98, wspace=.16)
        finish(figure, 'native_vision_v18_accuracy')
        figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
        for i, n in enumerate((32, 64)):
            for j, seed in enumerate(SEEDS):
                axis = axes[i, j]
                for mode in MODES:
                    axis.plot(range(17), [100 * point(mode, seed, n, k)['accuracy'] for k in range(17)],
                              color=colors[mode], marker='o' if mode == 'native_gate' else 's', markersize=3.5,
                              linewidth=1.4, label=labels[mode])
                axis.set_title(f'{n} frames · seed {seed}', loc='left'); axis.set_ylim(-4, 104)
                axis.set_xticks(range(17)); axis.set_yticks((0, 25, 50, 75, 100)); axis.grid(alpha=.2)
                if j == 0: axis.set_ylabel('Whole answer + EOS (%)')
                if i == 1: axis.set_xlabel('Gold count K (all values supported in training)')
        figure.suptitle('V18: whole-answer accuracy by count', x=.07, ha='left', fontsize=14)
        handles, names = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, names, loc='upper center', bbox_to_anchor=(.54, .94), ncol=2, frameon=False)
        figure.text(.07, .025, 'Each point copies one reported correct/8 cell. Lines connect discrete counts; no smoothing or new estimates.\n'
                    'Seeds remain separate. The independent report retains all decisions and uncertainty intervals.', fontsize=8)
        figure.subplots_adjust(top=.84, bottom=.15, left=.07, right=.98, hspace=.30, wspace=.16)
        finish(figure, 'native_vision_v18_by_count')
    return dict(files=files, matplotlib_version=matplotlib.__version__)


def self_test():
    metric = lambda n: dict(n=n, correct=0, accuracy=0., first_token_correct=n)
    analysis = dict(runs={f'{mode}_s{seed}': dict(condition=mode, seed=seed, selected_step=4590,
        native_test_examples=272, one_dev_only=True, dev_descriptive_only=True, metrics=dict(all=metric(272),
        cells={f'main_test_N{n}': metric(136) for n in (32, 64)},
        by_n_k={f'main_test_N{n}/K{k}': metric(8) for n in (32, 64) for k in range(17)})) for mode in MODES for seed in SEEDS})
    before = deepcopy(analysis); need(len(plotted_rows(analysis)) == 144 and analysis == before, 'Pure metric extraction changed its input')
    # The registered integer parser permits leading zeros/whitespace; whole-answer
    # correctness therefore need not imply the canonical first numeral token.
    metric_check(dict(n=8, correct=1, accuracy=.125, first_token_correct=0), 8)
    cases = []
    bad = deepcopy(analysis); del bad['runs']['all_open_s20']; cases.append(bad)
    bad = deepcopy(analysis); bad['runs']['all_open_s20']['metrics']['by_n_k']['main_test_N32/K0']['n'] = 7; cases.append(bad)
    bad = deepcopy(analysis); bad['runs']['all_open_s20']['metrics']['cells']['main_test_N64']['first_token_correct'] = 135; cases.append(bad)
    bad = deepcopy(analysis); bad['runs']['all_open_s20']['metrics']['all']['accuracy'] = float('nan'); cases.append(bad)
    for bad in cases:
        try:
            plotted_rows(bad)
        except ValueError:
            pass
        else:
            raise ValueError('Malformed reported metric fixture passed')
    return dict(passed=True, tests=['pure_existing_metric_extraction', 'four_run_coverage', 'per_count_denominators',
                                  'first_token_partition_consistency', 'independent_first_token_and_whole_counts', 'finite_reported_accuracy'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--report-summary', type=Path); mode.add_argument('--self-test', action='store_true')
    parser.add_argument('--output', type=Path); args = parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION') == 'cpu'
         and not os.environ.get('SLURM_JOB_GPUS'), 'Plotting and self-tests require CPU Slurm')
    started = time.perf_counter()
    out = (args.output or REPO / 'outputs/native_aggregation_vlm/v18/plots' / f'plot_{os.environ["SLURM_JOB_ID"]}').resolve()
    out.mkdir(parents=True, exist_ok=False); own = sources(); snapshot(out, own); save(out / 'source_hashes.json', own)
    tests = self_test()
    summary = dict(passed=True, completed=True, self_test=args.self_test, source_sha256=own, tests=tests,
                   no_rescoring=True, no_new_statistics=True, no_model_loaded=True, no_decision_changes=True)
    if not args.self_test:
        analysis, rows, bindings, report_sources = verify_report(args.report_summary, out)
        save(out / 'plot_data.json', rows); drawn = plot(rows, out)
        for path, digest in bindings.items():
            need(sha(path) == digest, 'Input changed during plotting: ' + path)
        summary.update(drawn, input_bindings=bindings, report_source_sha256=report_sources,
            report_summary=str(args.report_summary.resolve()), report_decisions={key: analysis[key] for key in DECISIONS},
            decisions=analysis['decisions'], plot_data_file=str(out / 'plot_data.json'), plot_data_sha256=sha(out / 'plot_data.json'))
        (out / 'INDEX.md').write_text('# V18 report figures\n\n'
            '![Whole answers and first tokens](native_vision_v18_accuracy.png)\n\n'
            '![Per-count whole answers](native_vision_v18_by_count.png)\n\n'
            '[Accuracy PDF](native_vision_v18_accuracy.pdf) · [Per-count PDF](native_vision_v18_by_count.pdf) · '
            '[Verified input report](report_text.md) · [Copied decisions and provenance](summary.json).\n')
    need(sources() == own, 'Plot source changed during execution')
    summary.update(seconds=time.perf_counter() - started, slurm_job_id=os.environ['SLURM_JOB_ID'])
    save(out / 'summary.json', summary)
    print(json.dumps(dict(passed=True, summary_file=str(out / 'summary.json'))), flush=True)


if __name__ == '__main__':
    main()

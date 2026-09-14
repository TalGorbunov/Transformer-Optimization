"""Evaluate six selected native VLMs on fixed-marginal K2/K6 binding pairs.

The existing V2/V3 harnesses run as sequential eval-only subprocesses. No model
or dataset changes are made. All metadata hashing and GPU work require Slurm.
Input-pair sensitivity is descriptive; it is not proof of a causal internal
mechanism or a new architecture. Oracle marginal references are kept separate.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import importlib
import os
from pathlib import Path
import re
from statistics import mean
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
DATA_BASE = Path('/mnt/data/gabriele/gnn_transformer')
CHECKPOINT_BASE = Path('/mnt/ckpts/gabriele/gnn_transformer')
MODELS = [('v2', 'global', 0), ('v2', 'hidden', 0),
          ('v3', 'lift_pre', 0), ('v3', 'lift_post', 0),
          ('v3', 'lift_pre', 1), ('v3', 'lift_post', 1)]
POLICY = dict(do_sample=False, repetition_penalty=1.0, max_new_tokens=4, output_logits=True)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def exclusive_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def verify_manifest(path):
    """Independent QA, category, full marginal and pair validation; no decoding."""
    raw = path.read_bytes()
    manifest = json.loads(raw)
    root = Path(manifest['dataset_root']).resolve()
    if manifest.get('schema_version') != 1 or not root.is_relative_to(DATA_BASE):
        raise ValueError('Invalid binding manifest schema/root')
    if set(manifest['splits']) != {'test_N32'}:
        raise ValueError('Binding probe must contain exactly test_N32')
    audit_path = Path(manifest['audit_file']).resolve()
    if audit_path != root / 'audit.json' or digest(audit_path) != manifest['audit_sha256']:
        raise ValueError('Independent staging audit hash/path differs')
    audit = json.loads(audit_path.read_text())
    for key in ('all_gold_labels_independently_recounted', 'all_new_content_independently_verified_disjoint',
                'all_pairs_exactly_eight_pixel_changes', 'all_pairs_character_identity_preserved_at_every_position',
                'all_pairs_full_character_and_room_marginals_preserved'):
        if audit.get(key) is not True:
            raise ValueError(f'Missing independent staging assertion: {key}')
    records = manifest['splits']['test_N32']['samples']
    if len(records) != 32:
        raise ValueError('Binding probe requires exactly 32 records')
    pairs, metadata, seen = defaultdict(dict), [], set()
    image_cache = {}
    for record in records:
        directory = Path(record['path']).resolve()
        condition = record['condition']
        if condition not in ('low', 'high') or record['gold'] != (2 if condition == 'low' else 6):
            raise ValueError('Expected low/high K2/K6 conditions')
        if record['n_frames'] != 32 or record.get('split', 'test') != 'test':
            raise ValueError('Binding record must be N32/test')
        if directory.parent != root / 'mmred_vfiltered/seq_len_32/test' or directory.name != record['sid']:
            raise ValueError('Binding sample outside declared test parent')
        if str(directory) in seen:
            raise ValueError('Duplicate binding sample')
        seen.add(str(directory))
        payload = (directory / 'qa.txt').read_bytes()
        if hashlib.sha256(payload).hexdigest() != record['qa_sha256']:
            raise ValueError('Binding QA hash mismatch')
        lines = payload.decode().splitlines()
        begin, end = lines.index('question:'), lines.index('answer:')
        content = [line.strip() for line in lines[begin + 1:end] if line.strip()]
        states = [ast.literal_eval(line) for line in content if line.startswith('{') and line.endswith('}')]
        questions = [line for line in content if not (line.startswith('{') and line.endswith('}'))]
        if len(states) != 32 or len(questions) != 1 or int(lines[end + 1].strip()) != record['gold']:
            raise ValueError('Malformed binding QA or changed gold')
        target = re.fullmatch(r'How many frames show ([A-Za-z]+) in the ([A-Za-z]+)\?', questions[0])
        if target is None:
            raise ValueError('Unexpected question template')
        character, room = target.groups()
        if record['target_character'] != character or record['target_room'] != room:
            raise ValueError('Manifest target differs from question')
        frames, categories = [], Counter()
        for index, state in enumerate(states):
            occupied = [(person, location) for location, people in state['rooms'].items() for person in people]
            if len(occupied) != 1 or state['step_id'] != index + 1:
                raise ValueError('Expected one character per canonically indexed frame')
            person, location = occupied[0]
            frames.append((person, location))
            category = ('matches' if person == character and location == room else
                        'same_character_wrong_room' if person == character else
                        'queried_room_other_character' if location == room else 'neither')
            categories[category] += 1
        c = sum(person == character for person, _ in frames)
        r = sum(location == room for _, location in frames)
        if c != 12 or r != 12 or categories['matches'] != record['gold']:
            raise ValueError('Fixed marginals or conjunction count violated')
        if record['marginal_C'] != c or record['marginal_R'] != r:
            raise ValueError('Declared marginal metadata differs')
        for key, actual in (('character_histogram', Counter(person for person, _ in frames)),
                            ('room_histogram', Counter(location for _, location in frames))):
            if Counter({name: value for name, value in record[key].items() if value}) != actual:
                raise ValueError('Declared complete marginal histogram differs')
        for key in ('matches', 'same_character_wrong_room', 'queried_room_other_character', 'neither'):
            if record['semantic_counts'].get(key) != categories[key]:
                raise ValueError('Declared semantic counts differ from QA')
        if len(record['image_files']) != 32:
            raise ValueError('Expected exactly32 image files')
        for index, image in enumerate(record['image_files']):
            image_path = Path(image['path']).resolve()
            if image_path != directory / f'{index:03d}.png':
                raise ValueError('Unexpected staged image path')
            stat = image_path.stat()
            identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            if identity not in image_cache:
                image_cache[identity] = digest(image_path)
            if image['bytes'] != stat.st_size or image['sha256'] != image_cache[identity]:
                raise ValueError('Binding image hash/size mismatch')
        pair_id = record['pair_id']
        if not isinstance(pair_id, str) or not pair_id or condition in pairs[pair_id]:
            raise ValueError('Invalid or duplicated pair condition')
        meta = dict(path=str(directory), sid=record['sid'], pair_id=pair_id, condition=condition,
                    gold=record['gold'], n_frames=32, question=questions[0],
                    marginal_C=c, marginal_R=r, qa_sha256=record['qa_sha256'],
                    changed_positions=record['changed_positions'])
        metadata.append(meta)
        pairs[pair_id][condition] = (record, frames, meta)
    if len(pairs) != 16:
        raise ValueError('Expected exactly16 complete pairs')
    for pair_id, pair in pairs.items():
        if set(pair) != {'low', 'high'}:
            raise ValueError('Incomplete binding pair')
        low, lo_frames, lo_meta = pair['low']; high, hi_frames, hi_meta = pair['high']
        if lo_meta['question'] != hi_meta['question']:
            raise ValueError('Paired questions differ')
        if [person for person, _ in lo_frames] != [person for person, _ in hi_frames]:
            raise ValueError('A character changed identity or position')
        if Counter(location for _, location in lo_frames) != Counter(location for _, location in hi_frames):
            raise ValueError('Full room marginals changed')
        changed = [i for i, (a, b) in enumerate(zip(lo_frames, hi_frames)) if a != b]
        if len(changed) != 8 or changed != sorted(lo_meta['changed_positions']) or changed != sorted(hi_meta['changed_positions']):
            raise ValueError('Expected exactly eight changed image positions')
        image_changed = [i for i, (a, b) in enumerate(zip(low['image_files'], high['image_files'])) if a['sha256'] != b['sha256']]
        if image_changed != changed:
            raise ValueError('Semantic and image changes disagree')
    return metadata, dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(),
                          dataset_root=str(root), samples=32, pairs=16,
                          audit_path=str(audit_path), audit_sha256=digest(audit_path),
                          unique_image_files_hashed=len(image_cache), all_qa_gold_marginal_image_checks_passed=True)


def selected_model(version, arm, seed, root):
    analysis_path = root / 'analysis.json'
    analysis = json.loads(analysis_path.read_text())
    info = analysis['arms'][arm] if version == 'v2' else analysis['arms'][arm][str(seed)]
    run = Path(analysis['runs'][arm] if version == 'v2' else analysis['runs'][arm][str(seed)]).resolve()
    expected_parent = root.resolve() / 'main' / arm
    if version == 'v3':
        expected_parent = expected_parent / f'seed{seed}'
    if run.parent != expected_parent:
        raise ValueError('Canonical analysis points outside the selected arm/seed directory')
    config = json.loads((run / 'config.json').read_text())
    summary = json.loads((run / 'summary.json').read_text())
    hashes = json.loads((root / 'main/source_hashes.json').read_text())
    if config['code_sha256'] != hashes or config['arm'] != arm or config['seed'] != seed:
        raise ValueError('Canonical run source/identity differs')
    for name, sha in hashes.items():
        if digest(REPO / name) != sha or digest(run / 'code' / name.replace('/', '_')) != sha:
            raise ValueError(f'Frozen main source differs: {name}')
    if config['generation_policy'] != POLICY or len(summary['training']) != 9:
        raise ValueError('Canonical training/generation policy differs')
    history = summary['training']
    if [entry['epoch'] for entry in history] != list(range(1, 10)) or any(entry['step'] != 45 * entry['epoch'] or entry['n'] != 180 for entry in history):
        raise ValueError('Canonical training did not complete the registered schedule')
    best = max(history, key=lambda entry: (sum(row['correct'] for row in entry['dev']) / 72,
                        -sum(row['gold_first_token_nll'] * row['n'] for row in entry['dev']) / 72))
    if info['selected_epoch'] != best['epoch'] or info['selected_checkpoint'] != summary['selected_checkpoint']:
        raise ValueError('Canonical checkpoint was not selected on development data')
    checkpoint = Path(info['selected_checkpoint']).resolve()
    expected_checkpoint_parent = CHECKPOINT_BASE / f'native_aggregation_vlm_{version}' / run.name
    if checkpoint.parent != expected_checkpoint_parent or checkpoint.name != 'best.pt':
        raise ValueError('Checkpoint outside canonical selected run')
    if digest(checkpoint) != info['selected_checkpoint_sha256']:
        raise ValueError('Selected checkpoint hash differs from completed canonical analysis')
    arch = config['architecture']
    for key, expected in dict(model='Qwen/Qwen2.5-VL-7B-Instruct', arm=arm, layer_index=14,
                               rank=64, lora_layers=4, lora_rank=8, lora_alpha=16.0, resize=392,
                               quantization='nf4').items():
        if arch.get(key) != expected:
            raise ValueError(f'Unexpected canonical architecture {key}')
    if version == 'v2' and (arch['protocol'] != 'vision_v2_clean' or arch['hidden_rank'] != 96):
        raise ValueError('Unexpected V2 architecture')
    if version == 'v3' and (arch['protocol'] != 'vision_v3_value_lifting' or arch['lift_rank'] != 8):
        raise ValueError('Unexpected V3 architecture')
    return dict(key=f'{version}_{arm}_seed{seed}', version=version, arm=arm, seed=seed,
                root=str(root.resolve()), run=str(run), run_id=config['run_id'],
                analysis_path=str(analysis_path.resolve()), analysis_sha256=digest(analysis_path),
                config_sha256=digest(run / 'config.json'), summary_sha256=digest(run / 'summary.json'),
                checkpoint=str(checkpoint), checkpoint_sha256=info['selected_checkpoint_sha256'],
                selected_epoch=info['selected_epoch'], architecture=arch,
                parameters=summary['parameters'], source_sha256=hashes,
                base_runtime={key: config[key] for key in
                              ('torch_version', 'transformers_version', 'quantization', 'prompt',
                               'image_encoding', 'decoding', 'image_processor_settings')},
                main_source_manifest_sha256=digest(root / 'main/source_hashes.json'))


def command(model, manifest, dataset_root, output):
    arguments = [sys.executable, str(REPO / f'scripts/native_aggregation_vlm_{model["version"]}.py'),
        '--model', model['architecture']['model'],
        '--dataset-root', dataset_root, '--checkpoint-root', str(CHECKPOINT_BASE),
        '--manifest', str(manifest.resolve()), '--checkpoint', model['checkpoint'],
        '--eval-only', '--epochs', '0', '--eval-ns', '32', '--limit-eval', '32',
        '--arm', model['arm'], '--seed', str(model['seed']), '--layer-index', '14',
        '--rank', '64', '--lora-layers', '4', '--lora-rank', '8', '--lora-alpha', '16',
        '--resize', '392', '--max-new-tokens', '4', '--max-seq-tokens', '16000',
        '--gold-max', '16', '--eval-modes', 'all', '--data-seed', '20260910',
        '--output', str(output)]
    if model['version'] == 'v2':
        arguments += ['--hidden-rank', '96', '--block-size', '64', '--query-chunk-size', '32']
    else:
        arguments += ['--lift-rank', '8']
    return arguments


def validate_interface(model, arguments):
    # Both harness modules expose stdlib-only parsers; no model/torch import.
    module = importlib.import_module(f'scripts.native_aggregation_vlm_{model["version"]}')
    parsed = module.parser().parse_args(arguments[2:])
    module.validate_args(parsed)
    if (not parsed.eval_only or parsed.epochs != 0 or parsed.eval_ns != [32]
            or parsed.limit_eval != 32 or parsed.count_manifest is not None
            or parsed.seed != model['seed'] or str(parsed.checkpoint) != model['checkpoint']):
        raise ValueError('Eval-only subprocess interface differs from the planned probe')
    for field in ('model', 'arm', 'layer_index', 'rank', 'lora_layers', 'lora_rank',
                  'lora_alpha', 'resize', 'block_size', 'hidden_rank', 'middle_rank',
                  'middle_alpha', 'center_messages', 'lift_rank'):
        if field in model['architecture'] and getattr(parsed, field) != model['architecture'][field]:
            raise ValueError(f'CLI architecture differs from selected checkpoint: {field}')
    if 'torch' in sys.modules:
        raise RuntimeError('CPU interface validation unexpectedly imported torch')


def verify_canonical_unchanged(model):
    paths = {model['analysis_path']: model['analysis_sha256'],
             model['checkpoint']: model['checkpoint_sha256'],
             str(Path(model['run']) / 'config.json'): model['config_sha256'],
             str(Path(model['run']) / 'summary.json'): model['summary_sha256'],
             str(Path(model['root']) / 'main/source_hashes.json'): model['main_source_manifest_sha256']}
    if any(digest(path) != sha for path, sha in paths.items()):
        raise ValueError('Canonical analysis/config/summary/source ledger/checkpoint changed after planning')


def metrics(rows):
    parsed = [row for row in rows if row['prediction'] is not None]
    return dict(n=len(rows), correct=sum(row['exact'] for row in rows), exact=mean(row['exact'] for row in rows),
                parsed=len(parsed), parse_rate=len(parsed) / len(rows),
                mae_parsed=mean(abs(row['prediction'] - row['gold']) for row in parsed) if parsed else None)


def paired_metrics(rows):
    pairs = defaultdict(dict)
    for row in rows:
        if row['condition'] in pairs[row['pair_id']]:
            raise ValueError('Duplicate pair condition')
        pairs[row['pair_id']][row['condition']] = row
    if len(pairs) != 16 or any(set(pair) != {'low', 'high'} for pair in pairs.values()):
        raise ValueError('Expected exactly16 complete output pairs')
    deltas = [pair['high']['prediction'] - pair['low']['prediction'] for pair in pairs.values()
              if pair['low']['prediction'] is not None and pair['high']['prediction'] is not None]
    both = sum(pair['low']['exact'] and pair['high']['exact'] for pair in pairs.values())
    return dict(pairs=16, both_parsed_pairs=len(deltas), both_correct_pairs=both, both_correct_fraction=both / 16,
                mean_delta_prediction=mean(deltas) if deltas else None,
                mean_delta_error=mean(delta - 4 for delta in deltas) if deltas else None,
                mean_absolute_delta_error=mean(abs(delta - 4) for delta in deltas) if deltas else None,
                positive_delta_pairs=sum(delta > 0 for delta in deltas),
                positive_delta_fraction_parsed=mean(delta > 0 for delta in deltas) if deltas else None,
                invariant_prediction_pairs=sum(delta == 0 for delta in deltas),
                invariant_prediction_fraction_parsed=mean(delta == 0 for delta in deltas) if deltas else None,
                contrast_denominator='Only pairs with both outputs parsed; both-correct uses all16 pairs')


def verify_child(output, model, records, manifest_info):
    matches = list(output.glob('*/summary.json'))
    if len(matches) != 1:
        raise ValueError('Expected one completed eval-only child run')
    run = matches[0].parent
    config = json.loads((run / 'config.json').read_text())
    summary = json.loads(matches[0].read_text())
    rows = json.loads((run / 'predictions.json').read_text())
    if (config['epochs'] != 0 or not config['eval_only'] or summary['training']
            or summary['selected_checkpoint'] != model['checkpoint']
            or config['architecture'] != model['architecture']
            or config['seed'] != model['seed'] or summary['parameters'] != model['parameters']
            or config['code_sha256'] != model['source_sha256']
            or config['generation_policy'] != POLICY or config.get('count_manifest') is not None
            or config['manifest_sha256'] != manifest_info['sha256']):
        raise ValueError('Child did not preserve frozen eval-only configuration')
    if any(config[key] != value for key, value in model['base_runtime'].items()):
        raise ValueError('Child changed the native runtime, prompt, quantization, or image processor configuration')
    for name, sha in model['source_sha256'].items():
        if digest(run / 'code' / name.replace('/', '_')) != sha or digest(REPO / name) != sha:
            raise ValueError('Child source snapshot differs from canonical main')
    index = {record['sid']: record for record in records}
    if len(rows) != 32 or {row['sid'] for row in rows} != set(index):
        raise ValueError('Missing or extra binding predictions')
    for row in rows:
        record = index[row['sid']]
        if (row['path'], row['gold'], row['pair_id'], row['n_frames']) != (record['path'], record['gold'], record['pair_id'], 32):
            raise ValueError('Binding prediction identity differs')
        if row['tag'] != 'test' or row['mode'] != 'all' or row['cell'] != 'length_N32' or row['exact'] != (row['prediction'] == row['gold']):
            raise ValueError('Invalid child prediction bookkeeping')
        row['condition'] = record['condition']
    if digest(model['checkpoint']) != model['checkpoint_sha256']:
        raise ValueError('Canonical checkpoint changed during evaluation')
    return rows, dict(child_run=str(run), all=metrics(rows),
                      low=metrics([row for row in rows if row['condition'] == 'low']),
                      high=metrics([row for row in rows if row['condition'] == 'high']), paired=paired_metrics(rows),
                      checkpoint_sha256_after=digest(model['checkpoint']))


def self_test():
    rows = [dict(pair_id=str(i), condition=condition, gold=gold, prediction=2, exact=gold == 2)
            for i in range(16) for condition, gold in [('low', 2), ('high', 6)]]
    output = paired_metrics(rows)
    if metrics(rows)['correct'] != 16 or output['both_correct_pairs'] != 0 or output['mean_delta_prediction'] != 0 or output['mean_absolute_delta_error'] != 4:
        raise AssertionError('Fixed-marginal oracle convention failed')
    rows[0]['prediction'], rows[0]['exact'] = None, False
    output = paired_metrics(rows)
    if output['both_parsed_pairs'] != 15 or output['pairs'] != 16 or metrics(rows)['n'] != 32:
        raise AssertionError('Missing parse altered all-example denominator')
    for row in rows:
        row['prediction'], row['exact'] = row['gold'], True
    perfect = paired_metrics(rows)
    if (perfect['both_correct_pairs'] != 16 or perfect['mean_delta_prediction'] != 4
            or perfect['mean_absolute_delta_error'] != 0 or perfect['positive_delta_fraction_parsed'] != 1):
        raise AssertionError('Perfect pair contrast failed')
    for row in rows:
        row['prediction'], row['exact'] = 8 - row['gold'], False
    reversed_result = paired_metrics(rows)
    if reversed_result['mean_delta_prediction'] != -4 or reversed_result['positive_delta_fraction_parsed'] != 0:
        raise AssertionError('Pair direction convention failed')
    print('PASS: fixed oracle, missing-parse denominators, perfect and reversed pair contrasts')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=DATA_BASE / 'v3_binding_probe/manifest.json')
    p.add_argument('--v2-root', type=Path, default=REPO / 'outputs/native_aggregation_vlm/v2')
    p.add_argument('--v3-root', type=Path, default=REPO / 'outputs/native_aggregation_vlm/v3')
    p.add_argument('--output', type=Path, default=REPO / 'outputs/native_aggregation_vlm/v3/binding_probe')
    p.add_argument('--max-seconds', type=float, default=570)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--self-test', action='store_true')
    mode.add_argument('--check-plan', action='store_true',
                      help='CPU-only: audit data/checkpoints/source identity and exact unchanged harness CLI; execute no model')
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Run this diagnostic inside a Slurm allocation; GPU evaluation requires one allocated GPU')
    if a.self_test:
        self_test(); return
    if not a.manifest.resolve().is_relative_to(DATA_BASE) or a.max_seconds <= 0:
        raise SystemExit('Invalid data path or time budget')
    started = time.monotonic()
    self_test()
    records, manifest_info = verify_manifest(a.manifest)
    models = [selected_model(version, arm, seed, a.v2_root if version == 'v2' else a.v3_root)
              for version, arm, seed in MODELS]
    output = a.output / f'{"check" if a.check_plan else "probe"}_{os.environ["SLURM_JOB_ID"]}'
    output.mkdir(parents=True, exist_ok=False)
    sources = output / 'code'; sources.mkdir()
    all_sources = {name: digest(REPO / name) for name in
                   ('scripts/probe_native_vision_binding.py', 'slurm/native_vision_binding.sbatch')}
    for model in models:
        for name, sha in model['source_sha256'].items():
            if name in all_sources and all_sources[name] != sha:
                raise ValueError('Incompatible frozen source versions across selected models')
            all_sources[name] = sha
        model['command'] = command(model, a.manifest, manifest_info['dataset_root'], output / model['key'])
        validate_interface(model, model['command'])
    for name, sha in all_sources.items():
        payload = (REPO / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != sha:
            raise ValueError('Source changed while planning')
        (sources / name.replace('/', '_')).write_bytes(payload)
    limits = [
        'Six development-selected checkpoints evaluated without training or parameter updates; canonical analyses and checkpoints are hashed.',
        'Sixteen new matched N32 pairs: K2/K6, C=R=12, full character/room marginals fixed, eight room-switched images.',
        'All32 examples remain in exact-accuracy denominators; contrast means/fractions condition on both outputs parsing and report that count.',
        'Input-pair contrasts do not identify an internal causal mechanism or establish a new architecture/general aggregation algorithm.',
        'This fixed-contingency, coupled-pair diagnostic is OOD relative to the V2 training generator; negative outcomes can include distribution shift.',
        'Room swaps also change other associations; fixed marginals exclude marginal-only shortcuts but do not exclude every alternative shortcut.',
        'The diagnostic is small and descriptive, with no uncertainty interval or seed-population claim.',
        'The oracle marginal moment is always2, giving16/32 example exact but0/16 both-pair correct; it is not a competitive VLM baseline.',
    ]
    plan = dict(schema_version=1, slurm_job_id=os.environ['SLURM_JOB_ID'], manifest=manifest_info,
                models=models, records=records, source_sha256=all_sources, limits=limits,
                check_plan_only=a.check_plan, max_seconds=a.max_seconds, command_execution='Structured argument lists, shell=False, sequential six subprocesses')
    exclusive_json(output / 'plan.json', plan)
    (output / 'manifest.json').write_bytes(a.manifest.read_bytes())
    if a.check_plan:
        print(f'PASS: all data, six selected checkpoint/source identities, and eval-only interfaces; plan {output / "plan.json"}', flush=True)
        return
    results, predictions = {}, {}
    for model in models:
        remaining = a.max_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError('Total diagnostic time budget exhausted')
        for name, sha in all_sources.items():
            if digest(REPO / name) != sha:
                raise ValueError('Frozen source changed before child execution')
        verify_canonical_unchanged(model)
        print(f"Evaluating {model['key']} with {remaining:.1f}s remaining", flush=True)
        with (output / f"{model['key']}.log").open('x') as log:
            subprocess.run(model['command'], cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=remaining, shell=False)
        rows, result = verify_child(output / model['key'], model, records, manifest_info)
        results[model['key']], predictions[model['key']] = result, rows
        print(json.dumps(dict(model=model['key'], **result)), flush=True)
    after_records, after_manifest = verify_manifest(a.manifest)
    if after_manifest != manifest_info or after_records != records:
        raise ValueError('Binding data changed during evaluation')
    for model in models:
        verify_canonical_unchanged(model)
    summary = dict(schema_version=1, plan_sha256=digest(output / 'plan.json'), models=results,
                   elapsed_seconds=time.monotonic() - started, limits=limits,
                   all_data_sources_checkpoints_unchanged=True,
                   oracle_marginal_reference=dict(estimate=2, example_correct=16, example_n=32,
                                                  both_correct_pairs=0, pairs=16, mae=2.0))
    exclusive_json(output / 'summary.json', summary)
    exclusive_json(output / 'predictions.json', predictions)
    lines = ['# Fixed-marginal native vision binding diagnostic', '',
             '| Selected model | Low K2 exact /16 | High K6 exact /16 | Parsed /32 | MAE (parsed) | Both pair answers correct /16 |',
             '|---|---:|---:|---:|---:|---:|']
    fmt = lambda value: 'NA' if value is None else f'{value:.3f}'
    for key, result in results.items():
        lines.append(f"| {key} | {result['low']['correct']}/16 | {result['high']['correct']}/16 | {result['all']['parsed']}/32 | {fmt(result['all']['mae_parsed'])} | {result['paired']['both_correct_pairs']}/16 |")
    lines += ['', '| Selected model | Pairs both parsed /16 | Mean prediction Δ | Mean Δ error (target4) | Mean absolute Δ error | Positive Δ fraction | Invariant fraction |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for key, result in results.items():
        pair = result['paired']
        values = [fmt(pair[name]) for name in ('mean_delta_prediction', 'mean_delta_error', 'mean_absolute_delta_error', 'positive_delta_fraction_parsed', 'invariant_prediction_fraction_parsed')]
        lines.append('| ' + ' | '.join([key, f"{pair['both_parsed_pairs']}/16", *values]) + ' |')
    lines += ['', 'Each Δ is high-K prediction minus low-K prediction. Missing parses remain incorrect; Δ summaries condition on complete pairs.', '',
              'The known-generator oracle marginal moment 0.75(C+R)−0.5N equals2 for every scene here. It gets16/32 examples correct while never answering both members of a pair correctly. This reference is separate from the model table.', '',
              '## Scope and provenance', ''] + ['- ' + limit for limit in limits] + ['', '[Immutable plan](plan.json) · [Summary](summary.json) · [All predictions](predictions.json)', '']
    (output / 'REPORT.md').write_text('\n'.join(lines))
    index = a.output / 'INDEX.md'
    if not index.exists():
        index.write_text('# Fixed-marginal binding probe\n\n')
    with index.open('a') as stream:
        stream.write(f"- Slurm {os.environ['SLURM_JOB_ID']}: [report]({output.name}/REPORT.md), [plan]({output.name}/plan.json), [summary]({output.name}/summary.json). Six selected models, no training.\n")
    print(f'Completed binding diagnostic: {output}', flush=True)


if __name__ == '__main__':
    main()

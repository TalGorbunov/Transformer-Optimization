"""Independently audit and report the registered V4 scene-refresh control.

All imports that perform numerical/model work are behind the CPU Slurm guard.
The V2 test scenes are reused exploratory evaluation, not fresh confirmation.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
from statistics import mean
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from scripts.report_native_vision_v2 import CELLS, compare, extension, summarize, verify_anchors
from scripts.report_native_vision_v3 import compare_pooled, describe_rows

CONDITIONS = ('repeat', 'refresh')
SEEDS = (2, 3)
DATA_ROOT = Path('/mnt/data/gabriele/gnn_transformer/v4_diversity')
CHECKPOINT_ROOT = Path('/mnt/ckpts/gabriele/gnn_transformer/native_aggregation_vlm_v4')
CONTRASTS = {'familiar_ood': CELLS[1:3], 'N16': CELLS[:1], 'unseen_count': CELLS[3:]}
REPORT_SOURCES = ('scripts/report_native_vision_v4.py', 'scripts/report_native_vision_v3.py',
                  'scripts/report_native_vision_v2.py')
RUNTIME_FIELDS = ('torch_version', 'transformers_version', 'image_processor_settings',
                  'quantization', 'prompt', 'image_encoding', 'supervision', 'decoding', 'generation_policy')
CELL_HISTOGRAMS = {
    'train_N8': {**{k: 90 for k in range(8)}, 8: 10},
    'train_N16': {k: 90 for k in range(9)},
    **{f'dev_N{n}': {k: 4 for k in range(9)} for n in (8, 16)},
    **{f'length_N{n}': {k: 12 for k in range(9)} for n in (16, 32, 64)},
    **{f'unseen_count_N{n}': {k: 8 for k in range(9, 17)} for n in (32, 64)},
}
ARCHITECTURE = dict(protocol='vision_v4_diversity', hidden_rank=96, middle_rank=32,
    middle_alpha=64.0, model='Qwen/Qwen2.5-VL-7B-Instruct', arm='hidden', layer_index=14,
    block_size=64, rank=64, lora_layers=4, lora_rank=8, lora_alpha=16.0, resize=392,
    quantization='nf4', center_messages=False, variant='hidden', routing_rank=None)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def close(a, b):
    return finite(a) and finite(b) and math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-8)


def audit_schedule(schedule, sources):
    """An independent implementation of the registered finite-support schedule."""
    need(schedule.get('schema_version') == 1 and schedule.get('purpose') == 'main'
         and schedule.get('data_seed') == 20260914, 'Wrong schedule schema/purpose/seed')
    need(Path(schedule['dataset_root']).resolve() == DATA_ROOT, 'Wrong schedule root')
    slots = schedule['slot_metadata']
    need(len(slots) == 180 and [s['slot'] for s in slots] == list(range(180)),
         'Incomplete or reordered slot metadata')
    indexed = {r['sid']: r for n in (8, 16) for r in sources[f'train_N{n}']}
    need(len(indexed) == 1540 and len({r['content_sha256'] for r in indexed.values()}) == 1540,
         'Expected 1540 distinct whole training scenes')
    conditions = schedule['conditions']
    need(set(conditions) == set(CONDITIONS), 'Unexpected conditions')
    original = conditions['repeat'][0]
    need(len(original) == 180 and len(set(original)) == 180, 'Original block malformed')
    need([s['original_sid'] for s in slots] == original, 'Original slot identities differ')
    need([s['original_slot'] for s in slots] == list(range(180)), 'Original slot order differs')
    need(Counter((s['n_frames'], s['gold']) for s in slots)
         == Counter({(n, k): 10 for n in (8, 16) for k in range(9)}), 'Original slot support differs')
    need([s['n_frames'] for s in slots] == [8] * 90 + [16] * 90, 'Original N8/N16 slot blocks reordered')
    saturated = [s['slot'] for s in slots if s['n_frames'] == 8 and s['gold'] == 8]
    need(len(saturated) == 10 and all(s.get('saturated') == (s['slot'] in saturated) for s in slots),
         'Finite-support saturation exception differs')
    fields = ('n_frames', 'gold', 'question', 'target_character', 'target_room')
    for condition in CONDITIONS:
        blocks = conditions[condition]
        need(len(blocks) == 9, 'Nine blocks required')
        seen_fresh = set(original)
        for block_index, block in enumerate(blocks):
            need(len(block) == 180 and len(set(block)) == 180, 'Wrong block width or duplicate')
            for slot, sid in enumerate(block):
                need(sid in indexed, 'Scheduled scene absent from manifest')
                need(all(indexed[sid].get(k) == slots[slot].get(k) for k in fields),
                     'A fixed slot changed N/K/question/targets')
                if condition == 'repeat' or block_index == 0 or slot in saturated:
                    need(sid == original[slot], 'Shared original or saturated slot changed')
                else:
                    need(sid not in seen_fresh, 'Refresh repeated nonsaturated content')
                    seen_fresh.add(sid)
        expected_unique = 180 if condition == 'repeat' else 1540
        need(len({sid for block in blocks for sid in block}) == expected_unique, 'Distinct scene count differs')
    need({sid for block in conditions['refresh'] for sid in block} == set(indexed),
         'Manifest contains unused training scenes')
    return dict(blocks=9, slots=180, saturated_slots=saturated, presentations=1620,
                image_frame_presentations=19440, distinct_by_condition={'repeat': 180, 'refresh': 1540},
                unique_training_cells={'8': 730, '16': 810}, schedule_conditions_verified=True)


def verify_data():
    """Verify frozen staging metadata; image bytes were independently audited in the CPU stage."""
    main, count = read(DATA_ROOT / 'main_manifest.json'), read(DATA_ROOT / 'count_manifest.json')
    sources = {}
    for manifest, family in ((main, 'length'), (count, 'unseen_count')):
        need(manifest.get('schema_version') == 1 and Path(manifest['dataset_root']).resolve() == DATA_ROOT,
             'Invalid staged schema/root')
        need(Path(manifest['audit_file']).resolve() == DATA_ROOT / 'audit.json', 'Unexpected staging audit')
        for key, cell in manifest['splits'].items():
            mapped = key.replace('test_', family + '_', 1) if key.startswith('test_') else key
            need(mapped not in sources, 'Duplicate staged cell')
            sources[mapped] = cell['samples']
    need(set(sources) == set(CELL_HISTOGRAMS), 'Unexpected staged cells')
    paths, sids = set(), set()
    for cell, histogram in CELL_HISTOGRAMS.items():
        rows = sources[cell]
        need(Counter(r['gold'] for r in rows) == Counter(histogram), f'Wrong support/count in {cell}')
        n = int(cell.rsplit('_N', 1)[1])
        split = 'train' if cell.startswith('train_') else 'dev' if cell.startswith('dev_') else 'test'
        for row in rows:
            path = Path(row['path'])
            need(path.resolve().parent == DATA_ROOT / 'mmred_vfiltered' / f'seq_len_{n}' / split
                 and path.name == row['sid'] and row['n_frames'] == n and row['split'] == split,
                 'Staged path/SID/split mismatch')
            need(row['path'] not in paths and row['sid'] not in sids, 'Duplicate staged identity')
            paths.add(row['path']); sids.add(row['sid'])
            need(valid_sha(row['qa_sha256']) and valid_sha(row['content_sha256'])
                 and isinstance(row['question'], str), 'Missing staged content provenance')
            need(len(row['image_files']) == n, 'Staged image count mismatch')
            for i, image in enumerate(row['image_files']):
                need(Path(image['path']) == path / f'{i:03d}.png' and image['bytes'] > 0
                     and valid_sha(image['sha256']), 'Invalid ordered image provenance')
    schedule = read(DATA_ROOT / 'schedule.json')
    audit = audit_schedule(schedule, sources)
    stage = read(DATA_ROOT / 'audit.json')
    for flag in ('all_copied_pair_mappings_preserved', 'all_fresh_content_exclusions_passed',
                 'all_image_hashes_and_dimensions_passed', 'all_original_copy_bytes_preserved',
                 'all_qa_hashes_and_gold_recounts_passed', 'prior_source_files_never_modified'):
        need(stage.get(flag) is True, f'Staging audit did not pass {flag}')
    need((stage['unique_samples'], stage['unique_content'], stage['fresh_samples']) == (2078, 2078, 1360),
         'Full staged counts differ')
    for key, filename in (('main', 'main_manifest.json'), ('count', 'count_manifest.json')):
        sha = digest(DATA_ROOT / filename)
        field = 'manifest' if key == 'main' else 'count_manifest'
        need(schedule[field] == str(DATA_ROOT / filename) and schedule[field + '_sha256'] == sha
             and stage['manifest_sha256'][key] == sha, 'Schedule/audit does not bind manifests')
    need(stage['schedule_sha256']['main'] == {'path': str(DATA_ROOT / 'schedule.json'),
         'sha256': digest(DATA_ROOT / 'schedule.json')}, 'Staging audit does not bind schedule')
    need(schedule['source_manifest_sha256'] == stage['source_manifest_sha256']
         == main['source_manifest_sha256'] == count['source_manifest_sha256'],
         'Source-manifest ledgers disagree')
    need(bool(schedule['source_manifest_sha256']), 'No source manifest hashes')
    for path, sha in schedule['source_manifest_sha256'].items():
        need(Path(path).is_absolute() and Path(path).resolve().is_relative_to(DATA_ROOT.parent)
             and digest(path) == sha, 'Frozen prior manifest changed')
    # Original scenes really are the original V2 order/content, not a substitute baseline.
    prior = read(DATA_ROOT.parent / 'v2_clean/main_manifest.json')
    original_rows = [r for n in (8, 16) for r in prior['splits'][f'train_N{n}']['samples']]
    need(schedule['conditions']['repeat'][0] == [r['sid'] for r in original_rows], 'Original V2 order changed')
    indexed = {r['sid']: r for n in (8, 16) for r in sources[f'train_N{n}']}
    for old in original_rows:
        new = indexed[old['sid']]
        need(new['qa_sha256'] == old['qa_sha256'] and new['content_sha256'] == old['content_sha256']
             and [im['sha256'] for im in new['image_files']] == [im['sha256'] for im in old['image_files']],
             'Original V2 training bytes differ')
    token_info = schedule['prompt_token_audit']
    need(Path(token_info['path']).resolve() == DATA_ROOT / 'prompt_token_audit.json'
         and digest(token_info['path']) == token_info['sha256'], 'CPU token-audit source changed')
    tokens = read(token_info['path'])
    need(tokens['model'] == ARCHITECTURE['model'] and tokens['resize'] == 392
         and tokens['distinct_training_scenes'] == 1540 and set(tokens['records']) == set(indexed)
         and tokens['all_matched_slot_input_ids_and_grids_equal'] is True
         and token_info['all_matched_slot_input_ids_and_grids_equal'] is True,
         'CPU processor ledger coverage/settings differ')
    for slot, original_sid in enumerate(schedule['conditions']['repeat'][0]):
        signature = tokens['records'][original_sid]
        need(valid_sha(signature['input_ids_sha256']) and signature['prompt_tokens'] > 0,
             'Invalid actual processor token signature')
        for block in schedule['conditions']['refresh']:
            need(tokens['records'][block[slot]] == signature, 'CPU token layouts differ for a matched slot')
    return dict(sources=sources, schedule=schedule, schedule_audit=audit, token_audit=tokens,
                stage_audit_sha256=digest(DATA_ROOT / 'audit.json'))


def verify_predictions(rows, sources, cells, tag, tokenizer):
    expected = {(cell, r['sid']): r for cell in cells for r in sources[cell]}
    need(len(rows) == len(expected) and len({(r['cell'], r['sid']) for r in rows}) == len(rows),
         'Missing/duplicated prediction rows')
    need({(r['cell'], r['sid']) for r in rows} == set(expected), 'Prediction cell/identity coverage differs')
    for row in rows:
        source = expected[(row['cell'], row['sid'])]
        need(all(row.get(k) == source.get(k) for k in ('path', 'sid', 'n_frames', 'gold', 'pair_id', 'test_family')),
             'Prediction differs from its exact staged source')
        need(row['mode'] == 'all' and row['tag'] == tag, 'Unexpected intervention or evaluation tag')
        need(row['output_text'] == tokenizer.decode(row['generated_token_ids'], skip_special_tokens=True),
             'Saved generation text differs from native token IDs')
        match = re.fullmatch(r'\s*([0-9]+)\s*', row['output_text'])
        prediction = int(match.group(1)) if match else None
        need(row['prediction'] == prediction and type(row['exact']) is bool
             and row['exact'] == (prediction == row['gold']), 'Strict parsing/exact bookkeeping differs')
        need(1 <= row['generated_tokens'] <= 4 and row['generated_tokens'] == len(row['generated_token_ids']),
             'Invalid generated-token count')
        need(all(finite(row[k]) and row[k] >= 0 for k in
                 ('gold_first_token_nll', 'preprocessing_seconds', 'model_seconds'))
             and row['prompt_tokens'] > 0, 'Invalid likelihood/timing/layout metadata')


def verify_metrics(metrics, rows, cells):
    need(len(metrics) == len(cells) and {m['cell'] for m in metrics} == set(cells), 'Metric cells differ')
    for metric in metrics:
        selected = [r for r in rows if r['cell'] == metric['cell']]
        computed = summarize(selected)
        need(metric['n'] == computed['n'] and metric['correct'] == computed['correct']
             and metric['parsed'] == sum(r['prediction'] is not None for r in selected), 'Metric denominator differs')
        for published, actual in (('exact', 'exact'), ('parse_rate', 'parse_rate'),
                                  ('gold_first_token_nll', 'first_token_nll'),
                                  ('mean_model_seconds', 'model_seconds'), ('mean_total_seconds', 'total_seconds'),
                                  ('mae_parsed', 'mae_parsed'), ('bias_parsed', 'bias_parsed')):
            need((metric[published] is None and computed[actual] is None)
                 or close(metric[published], computed[actual]), f'Published metric differs: {published}')


def verify_presentations(run, config, history, data, initialization, tokenizer):
    schedule, tokens = data['schedule'], data['token_audit']['records']
    sources = {r['sid']: r for n in (8, 16) for r in data['sources'][f'train_N{n}']}
    order = read(run / 'training_order.json')
    layouts = read(run / 'training_slot_layouts.json')
    rows = [json.loads(line) for line in (run / 'presentations.jsonl').read_text().splitlines()]
    need(len(order) == 9 and len(rows) == 1620 and set(layouts) == {str(i) for i in range(180)},
         'Incomplete training order/presentation/layout ledger')
    need(len(initialization['tensors']) == 35, 'Expected 35 trainable tensor states')
    for block in range(9):
        slots = list(range(180)); random.Random(config['seed'] + block).shuffle(slots)
        sids = [schedule['conditions'][config['condition']][block][slot] for slot in slots]
        need(order[block] == dict(block_index=block, epoch=block + 1,
             ordered_slot_indices=slots, ordered_sids=sids), 'Actual training order differs from registered shuffle')
        epoch = history[block]
        need(epoch['ordered_slots_sha256'] == object_sha(slots)
             and epoch['ordered_sids_sha256'] == object_sha(sids)
             and epoch['optimizer_state_steps'] == [45 * (block + 1)]
             and epoch['optimizer_state_entries'] == 35, 'Adam/order history mismatch')
        losses = []
        for order_index, (slot, sid) in enumerate(zip(slots, sids)):
            row, source = rows[block * 180 + order_index], sources[sid]
            expected = dict(block_index=block, epoch=block + 1, optimizer_step=45 * block + order_index // 4 + 1,
                order_index=order_index, slot=slot, sid=sid, path=source['path'], n_frames=source['n_frames'],
                gold=source['gold'], qa_sha256=source['qa_sha256'], content_sha256=source['content_sha256'],
                question_sha256=hashlib.sha256(source['question'].encode()).hexdigest())
            need(all(row.get(k) == v for k, v in expected.items()), 'Training presentation identity/step mismatch')
            target = tokenizer(str(source['gold']), add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
            signature = layouts[str(slot)]
            need(row['target_ids'] == target == signature['target_ids'], 'Training answer/EOS targets differ')
            for key in ('prompt_tokens', 'input_ids_sha256'):
                need(row[key] == tokens[sid][key] == signature[key], 'Actual training token hash/length differs')
            need(signature['image_grid_thw'] == tokens[sid]['image_grid_thw']
                 and signature['image_tokens'] == sum(math.prod(grid) // 4 for grid in signature['image_grid_thw'])
                 and len(signature['image_grid_thw']) == source['n_frames'], 'Image layout differs')
            need(finite(row['answer_token_ce']) and row['answer_token_ce'] >= 0, 'Nonfinite training CE')
            losses.append(row['answer_token_ce'])
        need(close(epoch['mean_answer_token_ce'], mean(losses)), 'Block loss differs from all presentations')
    expected_audit = dict(presentations=1620, image_frame_presentations=19440,
        unique_training_sids_seen=180 if config['condition'] == 'repeat' else 1540,
        presentations_sha256=digest(run / 'presentations.jsonl'),
        training_order_sha256=digest(run / 'training_order.json'),
        training_slot_layouts_sha256=digest(run / 'training_slot_layouts.json'),
        all_observed_slot_token_layouts_equal=True)
    need(sum(r['n_frames'] for r in rows) == 19440
         and len({r['sid'] for r in rows}) == expected_audit['unique_training_sids_seen']
         and read(run / 'presentation_audit.json') == expected_audit, 'Presentation budget/audit differs')
    return dict(rows=rows, order=order, layouts=layouts, audit=expected_audit, initialization=initialization)


def verify_run(run, condition, seed, expected_hashes, data, torch, tokenizer):
    config, summary = read(run / 'config.json'), read(run / 'summary.json')
    need(config['code_sha256'] == expected_hashes, 'Frozen training source ledger differs')
    for name, sha in expected_hashes.items():
        need(digest(run / 'code' / name.replace('/', '_')) == sha, f'Training snapshot changed: {name}')
    required = dict(arm='hidden', condition=condition, seed=seed, profile=False, eval_only=False,
        epochs=9, accumulation=4, layer_index=14, hidden_rank=96, lora_layers=4, lora_rank=8,
        lora_alpha=16.0, resize=392, lr=.001, lr_lora=.0001, max_grad_norm=1.0, max_steps=0,
        max_seq_tokens=16000, max_new_tokens=4, train_ns=[8, 16], dev_ns=[8, 16], eval_ns=[16, 32, 64],
        gold_max=16, eval_modes='all', limit_dev=36, limit_eval=108, limit_count=64,
        checkpoint=None, gradient_checkpointing=False, model=ARCHITECTURE['model'])
    need(all(config.get(k) == v for k, v in required.items()), 'Registered model/training configuration differs')
    need(not config.get('test_epochs') and config['architecture'] == ARCHITECTURE, 'Architecture/test selection differs')
    need(config['generation_policy'] == dict(do_sample=False, repetition_penalty=1.0,
         max_new_tokens=4, output_logits=True), 'Generation policy differs')
    need(run.name == config['run_id'] == summary['run_id'] and summary['condition'] == condition
         and summary['arm'] == 'hidden' and summary['profile'] is False, 'Run identity differs')
    for key, count in (('branch', 688224), ('lora', 720896), ('total_trainable', 1409120)):
        need(config['parameters'][key] == summary['parameters'][key] == count, 'Parameter budget differs')
    for copied, configured, filename, hashkey in (
        ('staged_manifest.json', 'manifest', 'main_manifest.json', 'manifest_sha256'),
        ('staged_count_manifest.json', 'count_manifest', 'count_manifest.json', 'count_manifest_sha256'),
        ('schedule.json', 'schedule', 'schedule.json', 'schedule_sha256')):
        need(Path(config[configured]).resolve() == DATA_ROOT / filename
             and digest(run / copied) == config[hashkey] == digest(DATA_ROOT / filename), 'Copied/frozen data differs')
    need(config['schedule_audit'] == data['schedule_audit']
         and summary['schedule_sha256'] == config['schedule_sha256']
         and config['prompt_token_audit'] == data['schedule']['prompt_token_audit']
         and digest(run / 'staged_prompt_token_audit.json') == config['prompt_token_audit']['sha256'],
         'Schedule or CPU token-audit provenance differs')
    manifest = read(run / 'data_manifest.json')
    need(set(manifest) == set(CELL_HISTOGRAMS), 'Selected manifest cells differ')
    fields = ('path', 'sid', 'n_frames', 'gold', 'question', 'qa_sha256', 'split', 'content_sha256',
              'target_character', 'target_room', 'pair_id', 'anchor_id', 'test_family',
              'parent_n_frames', 'parent_positions', 'anchor_positions')
    for cell, sources in data['sources'].items():
        selected = manifest[cell]
        need(selected['n'] == len(sources) == len(selected['samples'])
             and selected['gold_histogram'] == {str(k): v for k, v in CELL_HISTOGRAMS[cell].items()},
             'Selected sample count/gold histogram differs')
        for actual, source in zip(selected['samples'], sources):
            need(all(actual.get(k) == source.get(k) for k in fields), 'Selected source order/content differs')
            need(len(actual['image_files']) == len(source['image_files'])
                 and all(all(a.get(k) == b.get(k) for k in ('path', 'bytes', 'sha256'))
                         for a, b in zip(actual['image_files'], source['image_files'])), 'Selected image provenance differs')
    initialization = read(run / 'initialization.json')
    need(initialization['combined_sha256'] == config['initial_parameter_sha256']
         == summary['initial_parameter_sha256'], 'Initialization hash differs')
    need(all(valid_sha(initialization[k]) for k in ('combined_sha256', 'branch_sha256', 'lora_sha256')),
         'Invalid initialization digest')
    names = [r['name'] for r in initialization['tensors']]
    need(len(names) == len(set(names)) == 35 and sum(r['numel'] for r in initialization['tensors']) == 1409120,
         'Initialization tensor coverage differs')
    need(all(r['dtype'] == 'torch.float32' and math.prod(r['shape']) == r['numel'] and valid_sha(r['sha256'])
             for r in initialization['tensors']), 'Invalid initialization tensor metadata')
    history = summary['training']
    need(history == read(run / 'training.json') and len(history) == 9, 'Incomplete or inconsistent block history')
    for block, epoch in enumerate(history):
        need((epoch['block_index'], epoch['epoch'], epoch['step'], epoch['n']) == (block, block + 1, 45 * (block + 1), 180),
             'Nine complete 45-update blocks required')
        dev = read(run / f'dev_epoch{block + 1}.json')
        need(dev['metrics'] == epoch['dev'], 'Saved development evaluations differ from training history')
        verify_predictions(dev['predictions'], data['sources'], ('dev_N8', 'dev_N16'), f'dev_epoch{block + 1}', tokenizer)
        verify_metrics(epoch['dev'], dev['predictions'], ('dev_N8', 'dev_N16'))
    ledgers = verify_presentations(run, config, history, data, initialization, tokenizer)
    need(summary['presentation_audit'] == ledgers['audit'], 'Summary presentation audit differs')
    best = max(history, key=lambda e: (sum(m['correct'] for m in e['dev']) / 72,
        -sum(m['gold_first_token_nll'] * m['n'] for m in e['dev']) / 72))
    checkpoint = Path(summary['selected_checkpoint']).resolve()
    need(checkpoint == CHECKPOINT_ROOT / run.name / 'best.pt', 'Selected checkpoint path differs')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    need(saved['architecture'] == ARCHITECTURE and saved['epoch'] == best['epoch']
         and saved['step'] == best['step'] and saved['dev'] == best['dev'], 'Checkpoint selection/architecture differs')
    for key in ('run_id', 'condition', 'seed', 'architecture', 'code_sha256', 'manifest_sha256',
                'count_manifest_sha256', 'schedule_sha256', 'initial_parameter_sha256', 'parameters'):
        need(saved['config'][key] == config[key], f'Selected checkpoint configuration differs: {key}')
    tensor_map = {'branch.' + k: v for k, v in saved['branch'].items()}
    tensor_map.update({f'lora.{key}.{suffix}': value for key, pair in saved['lora'].items()
                       for suffix, value in zip(('A', 'B'), pair)})
    need(set(tensor_map) == set(names) and all(len(pair) == 2 for pair in saved['lora'].values()),
         'Selected checkpoint parameter identities differ')
    for item in initialization['tensors']:
        tensor = tensor_map[item['name']]
        need(list(tensor.shape) == item['shape'] and str(tensor.dtype) == item['dtype']
             and bool(torch.isfinite(tensor).all()), 'Saved trained tensor shape/dtype/finite check failed')
    del tensor_map, saved
    rows = read(run / 'predictions.json')
    verify_predictions(rows, data['sources'], CELLS, 'test', tokenizer)
    verify_anchors(rows, 'length', CELLS[:3], 108); verify_anchors(rows, 'unseen_count', CELLS[3:], 64)
    verify_metrics(summary['results'], rows, CELLS)
    info = describe_rows(rows)
    info.update(runtime_provenance={key: config[key] for key in RUNTIME_FIELDS},
        parameters=summary['parameters'], selected_epoch=best['epoch'],
        selected_checkpoint=str(checkpoint), selected_checkpoint_sha256=digest(checkpoint),
        training=history, presentation_audit=ledgers['audit'], initial_parameter_sha256=initialization['combined_sha256'],
        manifest_sha256=config['manifest_sha256'], count_manifest_sha256=config['count_manifest_sha256'],
        schedule_sha256=config['schedule_sha256'], stage_audit_sha256=data['stage_audit_sha256'],
        extensions={f'{family}_{lo}_to_{hi}': extension(rows, family, lo, hi)
                    for family, lo, hi in [('length', 16, 32), ('length', 32, 64), ('length', 16, 64), ('unseen_count', 32, 64)]},
        memory=dict(training_peak_allocated_bytes=max(e['peak_cuda_allocated_bytes'] for e in history),
                    training_peak_reserved_bytes=max(e['peak_cuda_reserved_bytes'] for e in history),
                    evaluation_peak_allocated_bytes=max(m['peak_cuda_allocated_bytes'] for m in summary['results'])))
    info['deterioration'] = {f'N16_to_N{n}': info['cells']['length_N16']['exact'] - info['cells'][f'length_N{n}']['exact']
                             for n in (32, 64)}
    return rows, info, ledgers


def audit_shared_runtime(provenances):
    """Compare the recorded execution environment across all four matched runs."""
    need(len(provenances) == 4, 'Runtime comparison requires all four registered runs')
    baseline = provenances[0]
    need(set(baseline) == set(RUNTIME_FIELDS), 'Incomplete runtime provenance')
    need(isinstance(baseline['image_processor_settings'], dict)
         and isinstance(baseline['generation_policy'], dict), 'Malformed processor/decoding provenance')
    need(all(isinstance(baseline[key], str) and baseline[key] for key in RUNTIME_FIELDS
             if key not in ('image_processor_settings', 'generation_policy')), 'Missing runtime descriptors')
    need(all(provenance == baseline for provenance in provenances[1:]),
         'Library versions, processor settings or forward/decoding provenance differ across runs')
    return dict(all_four_runs_identical=True, fields=list(RUNTIME_FIELDS), values=baseline,
                scope='Checks recorded versions/settings and source hashes; does not hash installed library binaries or frozen base-weight files.')


def paired_training_audit(repeat, refresh):
    need(repeat['initialization'] == refresh['initialization'], 'Within-seed initialized parameter bytes differ')
    need(repeat['layouts'] == refresh['layouts'], 'Within-seed actual per-slot training token layouts differ')
    keys = ('block_index', 'epoch', 'optimizer_step', 'order_index', 'slot', 'n_frames', 'gold',
            'question_sha256', 'prompt_tokens', 'input_ids_sha256', 'target_ids')
    need(len(repeat['rows']) == len(refresh['rows']) == 1620, 'Paired presentation denominators differ')
    for a, b in zip(repeat['rows'], refresh['rows']):
        need(all(a[k] == b[k] for k in keys), 'Ordered training tokens/targets/steps differ across conditions')
    first_a, first_b = repeat['rows'][:180], refresh['rows'][:180]
    need(all(all(a[k] == b[k] for k in ('sid', 'path', 'qa_sha256', 'content_sha256')) for a, b in zip(first_a, first_b)),
         'First block is not identical across conditions')
    differences = [b['answer_token_ce'] - a['answer_token_ce'] for a, b in zip(first_a, first_b)]
    return dict(presentations_matched=1620, initial_parameters_identical=True,
        ordered_slot_token_layouts_targets_and_steps_identical=True, first_block_scenes_identical=True,
        first_block_loss_difference=dict(n=180, mean_refresh_minus_repeat=mean(differences),
            mean_absolute=mean(abs(v) for v in differences), maximum_absolute=max(abs(v) for v in differences),
            exactly_equal=sum(v == 0 for v in differences)),
        optimizer_note='All 35 Adam states report cumulative steps 45..405 in both runs; momentum tensors were not saved for an independent numerical comparison.',
        loss_note='First-block loss differences are descriptive; concurrent floating-point executions need not be bitwise identical.')


def decision_criteria(contrasts):
    per_seed = {seed: contrasts[seed]['familiar_ood']['exact_gain'] >= .05 - 1e-12
                and contrasts[seed]['N16']['exact_gain'] >= -.05 - 1e-12 for seed in SEEDS}
    return dict(V4_1=all(per_seed.values()), primary_each_seed=per_seed, primary_both_seeds=all(per_seed.values()))


def self_test(np):
    candidate, control = {}, {}
    for seed in SEEDS:
        candidate[seed], control[seed] = [], []
        for i in range(4):
            for cell in CONTRASTS['familiar_ood']:
                base = dict(cell=cell, sid=f'{i}_{cell}', path=f'/sample/{i}/{cell}', gold=i, pair_id=str(i))
                candidate[seed].append(dict(base, exact=i % 2 == 0))
                control[seed].append(dict(base, exact=False))
    rng = lambda: np.random.default_rng(20260914)
    single = compare(candidate[2], control[2], CONTRASTS['familiar_ood'], np, rng())
    pooled = compare_pooled(candidate, control, CONTRASTS['familiar_ood'], np, rng())
    need((pooled['anchors'], pooled['n'], pooled['unique_examples']) == (4, 16, 8), 'Pooled denominators wrong')
    np.testing.assert_allclose(single['bootstrap95'], pooled['bootstrap95'])
    good = {seed: {'familiar_ood': {'exact_gain': .05}, 'N16': {'exact_gain': -.05}} for seed in SEEDS}
    need(decision_criteria(good)['V4_1'], 'Boundary criterion failed')
    good[3]['familiar_ood']['exact_gain'] = .049
    need(not decision_criteria(good)['V4_1'], 'A failing seed was rescued')
    good[3]['familiar_ood']['exact_gain'] = .8; good[2]['N16']['exact_gain'] = -.051
    need(not decision_criteria(good)['V4_1'], 'N16 degradation was ignored')
    bad = {seed: [dict(r) for r in rows] for seed, rows in candidate.items()}
    bad[3][0]['path'] = '/mismatch'
    try:
        compare_pooled(bad, control, CONTRASTS['familiar_ood'], np, rng())
    except ValueError:
        pass
    else:
        raise AssertionError('Pooled identity mismatch accepted')
    rows = [dict(cell='length_N16', pair_id='a', prediction=None, exact=False),
            dict(cell='length_N32', pair_id='a', prediction=None, exact=False)]
    ex = extension(rows, 'length', 16, 32)
    need(ex['n'] == 1 and ex['both_parsed'] == 0 and ex['both_correct'] == 0
         and ex['same_parsed_prediction_all_denominator'] == 0, 'Invalid parses counted as invariance')
    provenance = {key: 'recorded descriptor' for key in RUNTIME_FIELDS}
    provenance.update(image_processor_settings={'merge_size': 2}, generation_policy={'do_sample': False})
    need(audit_shared_runtime([dict(provenance) for _ in range(4)])['all_four_runs_identical'],
         'Identical runtime provenance rejected')
    bad_runtime = [dict(provenance) for _ in range(4)]
    bad_runtime[3]['torch_version'] = 'different version'
    try:
        audit_shared_runtime(bad_runtime)
    except ValueError:
        pass
    else:
        raise AssertionError('Cross-run runtime mismatch accepted')
    print('PASS: V4 criterion boundaries, anchor/seed pooling, identity mismatch, invalid-parse denominators and runtime matching')


def render(root, analysis, np, plt):
    results, contrasts, pooled = analysis['conditions'], analysis['refresh_minus_repeat'], analysis['pooled_refresh_minus_repeat']
    lines = ['# Native vision: matched training diversity (V4)', '',
        'Exploratory reused V2 test; nine equal-budget training blocks, two fixed seeds, development-selected checkpoints.', '',
        '| Condition | Seed | N16 | N32 | N64 | Familiar OOD | Unseen K9–16 | Selected block |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for condition in CONDITIONS:
        for seed in SEEDS:
            info = results[condition][seed]
            cells = [info['cells'][cell] for cell in CELLS[:3]] + [info['familiar_ood'], info['unseen_count']]
            values = [f"{v['correct']}/{v['n']} ({100*v['exact']:.1f}%)" for v in cells]
            lines.append('| ' + ' | '.join([condition, str(seed), *values, str(info['selected_epoch'])]) + ' |')
    lines += ['', f"Both-seed primary screen: **{analysis['criteria']['primary_both_seeds']}**.", '',
        '| Refresh minus repeat | Familiar OOD gain | Paired-anchor 95% interval | N16 gain | Pass |',
        '|---|---:|---:|---:|---:|']
    for label, group, passed in [(f'seed{seed}', contrasts[seed], analysis['criteria']['primary_each_seed'][seed]) for seed in SEEDS] + [('pooled fixed seeds', pooled, 'descriptive')]:
        value = group['familiar_ood']; lo, hi = value['bootstrap95']
        lines.append(f"| {label} | {100*value['exact_gain']:+.2f}pp | [{100*lo:+.2f}, {100*hi:+.2f}]pp | {100*group['N16']['exact_gain']:+.2f}pp | {passed} |")
    lines += ['', 'Each seed must gain at least5pp on familiar N32/N64 and lose no more than5pp on N16. Pooling cannot rescue a failed seed.',
        'Familiar OOD has216 observations from108 anchors per model; unseen counts have128 observations from64 anchors. The pooled interval resamples108 shared anchors while retaining both lengths and both fixed seeds.', '',
        '## Precision and paired extensions', '',
        '| Condition/seed | OOD parse | OOD MAE (parsed) | Count parse | Count MAE (parsed) | N16→N64 accuracy loss | N16/N64 both correct | Same parsed prediction /108 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    fmt = lambda x: 'NA' if x is None else f'{x:.3f}'
    for condition in CONDITIONS:
        for seed in SEEDS:
            info = results[condition][seed]; ood = info['familiar_ood']; count = info['unseen_count']; ext = info['extensions']['length_16_to_64']
            lines.append(f"| {condition}/{seed} | {100*ood['parse_rate']:.1f}% | {fmt(ood['mae_parsed'])} | {100*count['parse_rate']:.1f}% | {fmt(count['mae_parsed'])} | {100*info['deterioration']['N16_to_N64']:+.2f}pp | {ext['both_correct']}/108 | {ext['same_parsed_prediction']}/108 |")
    lines += ['', '## Training audit', '',
        'Every run passed checks of1540 available training scenes,1620 actual presentations,19440 training frames,405 updates,35 persistent Adam step counters, all nine development evaluations, checkpoint selection and452 test predictions. Repeat used180 distinct scenes; refresh used1540, including ten saturated N8/K8 scenes repeated in both conditions.',
        'Initial trainable tensor hashes, slot shuffles, ordered input-token hashes, image layouts and final-answer targets match across conditions within each seed. All four runs have identical recorded torch/transformers versions, image-processor settings, quantization, supervision, prompt, image-encoding and decoding configuration. Source manifests, CPU processor audit, copied run manifests, code snapshots and selected checkpoint hashes are bound in analysis.json.', '',
        '| Seed | First-block mean refresh−repeat CE | Mean absolute CE difference | Maximum absolute CE difference | Exactly equal /180 |',
        '|---|---:|---:|---:|---:|']
    for seed in SEEDS:
        loss = analysis['paired_training_audit'][seed]['first_block_loss_difference']
        lines.append(f"| {seed} | {loss['mean_refresh_minus_repeat']:.8g} | {loss['mean_absolute']:.8g} | {loss['maximum_absolute']:.8g} | {loss['exactly_equal']}/180 |")
    lines += ['', 'First-block losses are a descriptive reproducibility diagnostic, not an efficacy selection rule. Adam momentum tensors were not saved; the audit checks cumulative state counters plus the frozen optimizer implementation.', '',
        '## Measured cost', '', '| Condition/seed | OOD model sec/example | Total sec/example | Peak training allocated GiB |',
        '|---|---:|---:|---:|']
    for condition in CONDITIONS:
        for seed in SEEDS:
            info = results[condition][seed]; metric = info['familiar_ood']
            lines.append(f"| {condition}/{seed} | {metric['model_seconds']:.3f} | {metric['total_seconds']:.3f} | {info['memory']['training_peak_allocated_bytes']/2**30:.2f} |")
    lines += ['', 'All cell/per-K outcomes, likelihoods, K9 versus K10–16 support, extension transitions, tokenization and provenance are in [analysis.json](analysis.json).', '', '## Interpretation limits', '']
    lines += ['- ' + note for note in analysis['notes']] + ['']
    (root / 'REPORT.md').write_text('\n'.join(lines))
    colors = {'repeat': '#D55E00', 'refresh': '#0072B2'}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for condition in CONDITIONS:
        for seed in SEEDS:
            info = results[condition][seed]; style = '-' if seed == 2 else '--'
            axes[0, 0].plot([16, 32, 64], [info['cells'][f'length_N{n}']['exact'] for n in (16, 32, 64)],
                marker='o', linestyle=style, color=colors[condition], label=f'{condition} seed{seed}')
            axes[0, 1].plot([32, 64], [info['cells'][f'unseen_count_N{n}']['exact'] for n in (32, 64)],
                marker='o', linestyle=style, color=colors[condition])
            axes[1, 0].plot(range(1, 10), [sum(m['correct'] for m in epoch['dev']) / 72 for epoch in info['training']],
                linestyle=style, color=colors[condition])
    groups = [contrasts[s]['familiar_ood'] for s in SEEDS] + [pooled['familiar_ood']]
    means = np.array([g['exact_gain'] for g in groups]) * 100
    bounds = np.array([g['bootstrap95'] for g in groups]) * 100
    # Plot interval endpoints directly: a percentile interval need not contain the point estimate.
    for y, estimate, (lo, hi) in zip(range(3), means, bounds):
        axes[1, 1].plot([lo, hi], [y, y], color='#333333')
        axes[1, 1].plot([lo, hi], [y, y], '|', color='#333333', markersize=8)
        axes[1, 1].plot(estimate, y, 'o', color='#333333')
    axes[1, 1].axvline(0, color='grey', linewidth=1)
    axes[1, 1].axvline(5, color='#009E73', linestyle=':')
    axes[1, 1].set(yticks=[0, 1, 2], yticklabels=['seed2', 'seed3', 'pooled fixed seeds'],
        xlabel='Refresh minus repeat (percentage points)', title='Familiar OOD · anchor intervals')
    axes[0, 0].set(title='Familiar counts 0–8', xlabel='Frames', ylabel='Exact answer', xticks=[16, 32, 64])
    axes[0, 1].set(title='Unseen counts 9–16', xlabel='Frames', ylabel='Exact answer', xticks=[32, 64])
    axes[1, 0].set(title='In-range development', xlabel='Block (45 updates)', ylabel='Exact answer')
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.set_ylim(-.02, 1.02)
    for axis in axes.flat:
        axis.grid(alpha=.2)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center', ncol=4, frameon=False)
    fig.suptitle('Matched training diversity · exploratory reused V2 test', y=.945)
    fig.tight_layout(rect=(0, 0, 1, .91))
    fig.savefig(root / 'comparison.png', dpi=170); fig.savefig(root / 'comparison.pdf')
    plt.close(fig)
    print('\n'.join(lines[:25]), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('outputs/native_aggregation_vlm/v4'))
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--check-data', action='store_true', help='Audit finalized data/schedules without any trained results')
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Run V4 report/checks on CPU Slurm')
    import numpy as np
    if a.self_test:
        self_test(np)
        return
    data = verify_data()
    if a.check_data:
        # Mutation checks exercise the independent schedule audit against real finalized metadata.
        for mutation in ('repeat', 'saturation', 'slot_question'):
            bad = json.loads(json.dumps(data['schedule']))
            if mutation == 'repeat':
                bad['conditions']['repeat'][1][0] = bad['conditions']['refresh'][1][0]
            elif mutation == 'saturation':
                slot = data['schedule_audit']['saturated_slots'][0]
                bad['conditions']['refresh'][1][slot] = bad['conditions']['refresh'][1][0]
            else:
                bad['slot_metadata'][0]['question'] += ' altered'
            try:
                audit_schedule(bad, data['sources'])
            except ValueError:
                pass
            else:
                raise AssertionError(f'Independent schedule audit accepted {mutation} mutation')
        print('PASS: finalized V4 staged provenance,730/810 training counts,1540 processor signatures,9x180 schedules and adversarial mutations')
        return
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from transformers import AutoTokenizer
    frozen_analysis = read(a.root / 'analysis_source_hashes.json')
    need(all(frozen_analysis.get(name) == digest(REPO / name) for name in REPORT_SOURCES), 'Analysis source changed after freeze')
    expected_hashes = read(a.root / 'main/source_hashes.json')
    tokenizer = AutoTokenizer.from_pretrained(ARCHITECTURE['model'], local_files_only=True, use_fast=False)
    results, predictions, ledgers, runs = {}, {}, {}, {}
    for condition in CONDITIONS:
        results[condition], predictions[condition], ledgers[condition], runs[condition] = {}, {}, {}, {}
        for seed in SEEDS:
            candidates = list((a.root / 'main' / condition / f'seed{seed}').glob('*/summary.json'))
            need(len(candidates) == 1, f'Expected exactly one completed {condition}/seed{seed}: {candidates}')
            run = candidates[0].parent
            rows, info, ledger = verify_run(run, condition, seed, expected_hashes, data, torch, tokenizer)
            predictions[condition][seed], results[condition][seed] = rows, info
            ledgers[condition][seed], runs[condition][seed] = ledger, str(run)
    runtime_audit = audit_shared_runtime([results[condition][seed]['runtime_provenance']
                                          for condition in CONDITIONS for seed in SEEDS])
    paired = {seed: paired_training_audit(ledgers['repeat'][seed], ledgers['refresh'][seed]) for seed in SEEDS}
    rng = lambda: np.random.default_rng(20260914)
    contrasts = {seed: {name: compare(predictions['refresh'][seed], predictions['repeat'][seed], cells, np, rng())
                       for name, cells in CONTRASTS.items()} for seed in SEEDS}
    pooled = {name: compare_pooled(predictions['refresh'], predictions['repeat'], cells, np, rng())
              for name, cells in CONTRASTS.items()}
    notes = [
        'The V2 test scenes have been inspected in earlier blocks. This is exploratory reused-test evaluation; a positive effect needs fresh confirmation without further tuning.',
        'This isolates scene refresh under the same slot N/K/question, model, initial parameters, targets and405-update budget. It does not isolate diversity from the accompanying reduction in repetition.',
        'Refresh has1540 distinct scenes, not1620: ten N8/K8 fixed-question slots are deterministic and repeat in both conditions.',
        'Primary refresh-minus-repeat familiar OOD gain >=5pp and N16 loss <=5pp must hold in each seed. The pooled contrast cannot rescue a failure.',
        'Bootstrap uses10000 whole-anchor draws with seed20260914, retaining paired lengths and both fixed seeds together. It quantifies conditional test-anchor variation, not population training-seed uncertainty.',
        'All invalid answers remain incorrect. MAE and signed bias condition on parsing; two invalid answers do not count as prediction invariance.',
        'K9 is single-token while K10–16 are multi-digit. Raw first-answer-token NLL is not whole-answer likelihood.',
        'Saved cumulative Adam counters and frozen code establish the observed state-persistence checks; optimizer momentum tensors were not retained for an independent value-by-value audit.',
        'Timing includes concurrent cluster conditions. The inference architecture is identical across conditions.',
        'A refresh effect is an ordinary supervised-data result, not a novel aggregation operator or proof of general aggregation/reasoning composition. Failure does not authorize another architecture or learning-rate grid.',
    ]
    analysis = dict(criteria=decision_criteria(contrasts), conditions=results, runs=runs,
        refresh_minus_repeat=contrasts, pooled_refresh_minus_repeat=pooled,
        pooled_conditions={condition: describe_rows([r for seed in SEEDS for r in predictions[condition][seed]])
                           for condition in CONDITIONS},
        paired_training_audit=paired, runtime_audit=runtime_audit, schedule_audit=data['schedule_audit'],
        bootstrap=dict(draws=10000, seed=20260914, unit='shared anchor family; lengths and fixed seeds retained'),
        answer_tokenization={str(k): tokenizer(str(k), add_special_tokens=False).input_ids for k in range(17)},
        training_source_sha256=expected_hashes, analysis_source_sha256={name: digest(REPO / name) for name in REPORT_SOURCES},
        analysis_job=os.environ['SLURM_JOB_ID'], notes=notes)
    write(a.root / 'analysis.json', analysis)
    code = a.root / 'analysis_code'; code.mkdir(exist_ok=True)
    for name in REPORT_SOURCES:
        (code / name.replace('/', '_')).write_bytes((REPO / name).read_bytes())
    render(a.root, analysis, np, plt)
    (a.root / 'INDEX.md').write_text('# V4 analysis artifacts\n\n- [Report](REPORT.md)\n- [Verified metrics and provenance](analysis.json)\n- [Figure PNG](comparison.png)\n- [Figure PDF](comparison.pdf)\n')


if __name__ == '__main__':
    main()

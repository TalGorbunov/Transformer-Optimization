"""Frozen-model, fixed-budget MMReD Vision reasoning baseline; no training.

All processing runs in Slurm. --check-plan freezes prompts, model metadata,
EOS identities, data and processor layouts before any outcomes. --run preserves
every generated ID. --report compares within-model policies on paired anchors.
The 128-token endpoint is a censored prefix of the same 512-cap generation.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
from statistics import mean
import sys
import time

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from scripts.probe_native_vision_v2_prefix import fingerprint, sha_object, sha_file, verify_images

DATA_ROOT = Path('/mnt/data/gabriele/gnn_transformer/reasoning_baseline')
OUTPUT_ROOT = REPO / 'outputs/native_aggregation_vlm/reasoning_baseline'
MODEL_PATHS = {
    'qwen': '/mnt/ckpts/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5',
    'cosmos': '/mnt/ckpts/gabriele/gnn_transformer/cosmos_reason1_compat_440953',
}
SYSTEMS = {
    'direct': 'You are a helpful assistant. Answer the question without reasoning, in the following format: <answer>\nyour answer\n</answer>. The answer must contain only a single integer.',
    'reason': 'You are a helpful assistant. Answer the question in the following format: <think>\nyour reasoning\n</think>\n\n<answer>\nyour answer\n</answer>.',
}
CAPS = {'direct': 32, 'reason': 512}
SEED = 20260915
SOURCES = (
    'scripts/probe_native_vision_reasoning.py',
    'scripts/probe_native_vision_reasoning_execute.py',
    'scripts/probe_native_vision_v2_prefix.py',
    'gnnformer/runtime.py', 'gnnformer/constants.py',
    'slurm/native_vision_reasoning_check.sbatch',
    'slurm/native_vision_reasoning_profile.sbatch',
    'slurm/native_vision_reasoning_main.sbatch',
    'slurm/native_vision_reasoning_report.sbatch',
)
LIMITS = [
    'Frozen-model baseline assay only; no improved aggregation method, external tally, oracle prefix, training, or composition experiment.',
    'Primary contrasts compare policies within each fixed model. Cosmos and Qwen differ in pretraining/post-training; their difference is not a causal reasoning effect.',
    'System instruction and output budget change together. This tests a prompted-reasoning policy, not the isolated causal effect of adding a token.',
    'The official Cosmos guide recommends at least 4096 output tokens and reports BF16 testing. This bounded assay uses 512, NF4/bf16, greedy decoding and repetition penalty1; negative or truncated outcomes do not establish general reasoning failure.',
    'The 128-token endpoint is censored from the same 512-cap trace, with no independently measured 128 latency. Later normal EOS is incorrect at128, even if a prefix already contains an answer.',
    'Whole-trace grammar and normal EOS are required. Every malformed/truncated output remains in exact denominators. MAE is conditional on parsed, completed outputs.',
    'Intervals resample18 entire paired anchors and are descriptive for these fixed models/scenes, not seed-population inference. No prompt/model/budget selection is performed.',
]


def save_json(path, value):
    path = Path(path)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_hashes():
    return {name: sha_file(REPO / name) for name in SOURCES}


def user_text(record):
    return (f'You will be shown {record["n_frames"]} frames describing steps in a house.\n'
            f'Question: {record["question"]}\n'
            f'The final answer must be a single integer from 0 to {record["n_frames"]}.')


def messages(record, condition, frames=None):
    images = ([dict(type='image') for _ in range(record['n_frames'])] if frames is None else
              [dict(type='image', image=frame) for frame in frames])
    return [dict(role='system', content=[dict(type='text', text=SYSTEMS[condition])]),
            dict(role='user', content=images + [dict(type='text', text=user_text(record))])]


def parse_text(text, condition):
    """Only whole grammar matches; trace numerals and extra answer tags never count."""
    if text.count('<answer>') != 1 or text.count('</answer>') != 1:
        return None
    answer = r'<answer>\s*([0-9]+)\s*</answer>'
    if condition == 'direct':
        if '<think>' in text or '</think>' in text:
            return None
        match = re.fullmatch(r'\s*' + answer + r'\s*', text)
    elif condition == 'reason':
        if text.count('<think>') != 1 or text.count('</think>') != 1:
            return None
        match = re.fullmatch(r'\s*<think>(.*?)</think>\s*' + answer + r'\s*', text, re.S)
        if match is None or not match.group(1).strip():
            return None
    else:
        raise ValueError('Unknown condition')
    return None if match is None else int(match.group(match.lastindex))


def score_tokens(ids, tokenizer, eos_ids, budget, condition, gold):
    require(budget > 0, 'Nonpositive generation budget')
    prefix = ids[:budget]
    eos_positions = [i for i, token in enumerate(prefix) if token in eos_ids]
    completed = bool(eos_positions)
    eos_index = eos_positions[0] if completed else None
    text_ids = prefix[:eos_index] if completed else prefix
    text = tokenizer.decode(text_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    prediction = parse_text(text, condition) if completed else None
    return dict(budget=budget, completed=completed, truncated=not completed,
                eos_token_id=prefix[eos_index] if completed else None,
                completion_tokens=eos_index + 1 if completed else None,
                observed_tokens=len(prefix), text=text, prediction=prediction,
                parsed=prediction is not None, correct=prediction == gold,
                absolute_error=abs(prediction - gold) if prediction is not None else None,
                failure=None if prediction is not None else ('grammar' if completed else 'no_eos_by_budget'))


def self_test():
    valid = '<think>Frames 9, 10 and 11 do not match; there are 2.</think>\n<answer>2</answer>'
    require(parse_text(valid, 'reason') == 2, 'Trace numerals affected final answer')
    require(parse_text(' \n<answer>002</answer>\t', 'direct') == 2, 'Unsigned integer/whitespace failed')
    bad = [valid + ' extra', valid + '<answer>2</answer>', '<answer>2</answer>',
           '<think> </think><answer>2</answer>', '<think>x<answer>4</answer></think><answer>2</answer>',
           '<think>x</think><answer>-2</answer>', '<think>x</think><answer>2.0</answer>']
    require(all(parse_text(text, 'reason') is None for text in bad), 'Malformed trace accepted')
    require(parse_text(valid, 'direct') is None, 'Direct accepted reasoning block')
    class FakeTokenizer:
        def decode(self, ids, **kwargs):
            return ''.join(chr(token) for token in ids)
    tokens = list(map(ord, '<answer>2</answer>'))
    eos = 999
    tokenizer = FakeTokenizer()
    require(score_tokens(tokens, tokenizer, [eos], len(tokens), 'direct', 2)['truncated'], 'Missing EOS accepted')
    exact = score_tokens(tokens + [eos], tokenizer, [eos], len(tokens) + 1, 'direct', 2)
    require(exact['correct'] and exact['completed'], 'EOS exactly at budget rejected')
    short = score_tokens(tokens + [eos], tokenizer, [eos], len(tokens), 'direct', 2)
    require(not short['correct'] and short['truncated'], 'Later EOS accepted at short endpoint')
    rows = [dict(endpoint=exact), dict(endpoint=short)]
    require(metrics([r['endpoint'] for r in rows])['exact'] == 0.5, 'Invalid rows excluded from denominator')
    return dict(passed=True, tests=8)


def metadata(model_path):
    """Hash small metadata; record shard identity/stat without reading huge weights."""
    root = Path(model_path)
    require(root.is_dir(), f'Model snapshot missing: {root}')
    files = {p.name: sha_file(p) for p in sorted(root.iterdir())
             if p.is_file() and p.suffix in ('.json', '.txt', '.md') and p.stat().st_size < 30_000_000}
    for name in ('config.json', 'generation_config.json', 'tokenizer_config.json', 'preprocessor_config.json'):
        require(name in files, f'Missing required metadata {name}')
    index = json.loads((root / 'model.safetensors.index.json').read_text())
    shards = {}
    for name in sorted(set(index['weight_map'].values())):
        path = root / name
        stat = path.stat()
        shards[name] = dict(resolved_path=str(path.resolve()), size=stat.st_size,
                            mtime_ns=stat.st_mtime_ns, inode=stat.st_ino)
    require(bool(shards), 'No model shards')
    generation = json.loads((root / 'generation_config.json').read_text())
    eos = generation['eos_token_id']
    eos = [eos] if isinstance(eos, int) else eos
    require(eos == [151645, 151643], 'Native EOS list differs from registration')
    return dict(path=str(root), metadata_sha256=files, weight_shard_stat=shards,
                native_generation_config=generation, eos_token_ids=eos,
                pad_token_id=generation['pad_token_id'],
                weight_validation='Immutable resolved shard path/stat and model load; no full weight-content rehash')


def manifest_records(path, purpose):
    manifest = json.loads(Path(path).read_text())
    require(manifest.get('schema_version') == 1 and Path(manifest['dataset_root']).resolve() == DATA_ROOT,
            'Unexpected manifest schema/root')
    require(manifest['purpose'] == purpose and manifest['data_seed'] == SEED, 'Manifest purpose/seed differs')
    require(set(manifest['splits']) == {'test_N16', 'test_N32', 'test_N64'}, 'Unexpected manifest cells')
    require(sha_file(manifest['audit_file']) == manifest['audit_sha256'], 'Stage audit checksum differs')
    records = []
    by_anchor = defaultdict(dict)
    for n in (16, 32, 64):
        cell = manifest['splits'][f'test_N{n}']['samples']
        require(len(cell) == (18 if purpose == 'main' else 2), 'Unexpected cell size')
        expected = Counter({k: 2 for k in range(9)}) if purpose == 'main' else Counter({3: 1, 6: 1})
        require(Counter(r['gold'] for r in cell) == expected, 'Unexpected gold balance')
        for record in cell:
            directory = Path(record['path'])
            require(directory.parent.resolve() == DATA_ROOT / 'mmred_vfiltered' / f'seq_len_{n}' / 'test',
                    'Scene path outside registered test cell')
            require(record['n_frames'] == n and directory.name == record['sid'], 'Scene identity differs')
            raw = (directory / 'qa.txt').read_bytes()
            require(hashlib.sha256(raw).hexdigest() == record['qa_sha256'], 'QA checksum differs')
            lines = raw.decode().splitlines()
            question_lines = [line.strip() for line in lines[lines.index('question:') + 1:lines.index('answer:')]
                              if line.strip()]
            states = [ast.literal_eval(line) for line in question_lines if line.startswith('{')]
            questions = [line for line in question_lines if not line.startswith('{')]
            answer = next(line.strip() for line in lines[lines.index('answer:') + 1:] if line.strip())
            require(questions == [record['question']] and len(states) == n and int(answer) == record['gold'],
                    'QA question/count/answer differs')
            frames = []
            for state in states:
                occupants = [(c, room) for room, chars in state['rooms'].items() for c in chars]
                require(len(occupants) == 1, 'Expected exactly one character per frame')
                frames.append(occupants[0])
            target = (record['target_character'], record['target_room'])
            require(sum(frame == target for frame in frames) == record['gold'], 'Independent gold recount failed')
            anchor = record['pair_id']
            require(n not in by_anchor[anchor], 'Duplicate anchor/length')
            by_anchor[anchor][n] = record
            records.append(record)
    for anchor, cells in by_anchor.items():
        require(set(cells) == {16, 32, 64}, 'Incomplete paired anchor')
        require(len({(r['gold'], r['question']) for r in cells.values()}) == 1, 'Paired target/question changed')
    verify_images(records)
    return records, dict(path=str(Path(path).resolve()), sha256=sha_file(path),
                         audit_file=manifest['audit_file'], audit_sha256=manifest['audit_sha256'],
                         source_manifest_sha256=manifest.get('source_manifest_sha256', {}))


def prepare(processor, record):
    from PIL import Image
    frames = []
    try:
        for image in record['image_files']:
            with Image.open(image['path']) as source:
                rgb = source.convert('RGB')
                try:
                    frames.append(rgb.resize((392, 392)))
                finally:
                    rgb.close()
        inputs = {condition: dict(processor.apply_chat_template(
            messages(record, condition, frames), add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors='pt')) for condition in CAPS}
    finally:
        for frame in frames:
            frame.close()
    require(inputs['direct']['pixel_values'].equal(inputs['reason']['pixel_values']), 'Pixels differ by policy')
    layouts = {}
    for condition, item in inputs.items():
        ids = item['input_ids'][0].tolist()
        grid = item['image_grid_thw'].tolist()
        require(len(grid) == record['n_frames'] and all(g == [1, 28, 28] for g in grid), 'Unexpected image grid')
        require(len(ids) + CAPS[condition] <= 16000, 'Context cap exceeded')
        template = processor.apply_chat_template(messages(record, condition), add_generation_prompt=True, tokenize=False)
        layouts[condition] = dict(input_ids=ids, input_ids_sha256=sha_object(ids), prompt_tokens=len(ids),
                                  image_grid_thw=grid, rendered_chat=template,
                                  user_text=user_text(record), template_boundary_ids=ids[-24:])
    return inputs, layouts


def check_plan(args):
    import torch
    import transformers
    from transformers import AutoProcessor
    torch.set_num_threads(4)
    output = Path(args.output or OUTPUT_ROOT / f'check_{os.environ["SLURM_JOB_ID"]}')
    output.mkdir(parents=True, exist_ok=False)
    tests = self_test()
    main, main_provenance = manifest_records(args.manifest, 'main')
    profile, profile_provenance = manifest_records(args.profile_manifest, 'profile')
    all_records = main + profile
    require(len({r['sid'] for r in all_records}) == 60, 'Main/profile SID overlap')
    require(len({r['content_sha256'] for r in all_records}) == 60, 'Main/profile content overlap')
    models = {}
    for key, path in MODEL_PATHS.items():
        model = metadata(path)
        processor = AutoProcessor.from_pretrained(path, trust_remote_code=True, use_fast=False, local_files_only=True)
        tokenizer = processor.tokenizer
        model['processor'] = fingerprint(processor, transformers.__version__)
        model['eos_tokens'] = {str(token): tokenizer.convert_ids_to_tokens(token) for token in model['eos_token_ids']}
        require(model['eos_tokens'] == {'151645': '<|im_end|>', '151643': '<|endoftext|>'}, 'EOS token identities differ')
        require(tokenizer.eos_token_id in model['eos_token_ids'], 'Tokenizer EOS outside native stop list')
        model['layouts'] = {}
        for record in all_records:
            inputs, layouts = prepare(processor, record)
            model['layouts'][record['sid']] = layouts
            del inputs
        models[key] = model
        del processor
        gc.collect()
    plan = dict(schema_version=1, protocol='frozen_native_vision_reasoning_baseline', seed=SEED,
                systems=SYSTEMS, caps=CAPS, main_records=main, profile_records=profile,
                manifests={'main': main_provenance, 'profile': profile_provenance}, models=models,
                sources=source_hashes(), tests=tests, decoding=dict(do_sample=False, repetition_penalty=1.0,
                use_cache=True, temperature=None, top_p=None, top_k=None), limitations=LIMITS,
                runtime=dict(torch_version=torch.__version__, transformers_version=transformers.__version__),
                slurm_job_id=os.environ['SLURM_JOB_ID'])
    save_json(output / 'plan.json', plan)
    verify_plan(json.loads((output / 'plan.json').read_text()))
    (output / 'plan.sha256').write_text(sha_file(output / 'plan.json') + '\n')
    snapshot_sources(output)
    (output / 'REPORT.md').write_text('CPU plan passed:60 scenes,2 frozen model processors,2 fixed policies, native EOS identities, all QA/image hashes and exact prompt layouts. No model outcomes.\n')
    print(json.dumps({'plan': str(output / 'plan.json'), 'sha256': sha_file(output / 'plan.json')}), flush=True)


def snapshot_sources(output):
    directory = output / 'code'
    directory.mkdir()
    for name in SOURCES:
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO / name).read_bytes())


def verify_plan(plan):
    require(plan['schema_version'] == 1 and plan['protocol'] == 'frozen_native_vision_reasoning_baseline', 'Wrong plan')
    require(plan['sources'] == source_hashes(), 'Frozen source hashes changed')
    require(plan['systems'] == SYSTEMS and plan['caps'] == CAPS and plan['seed'] == SEED, 'Frozen protocol changed')
    for item in plan['manifests'].values():
        require(sha_file(item['path']) == item['sha256'], 'Manifest changed after CPU freeze')
        require(sha_file(item['audit_file']) == item['audit_sha256'], 'Stage audit changed after CPU freeze')
        for path, digest in item['source_manifest_sha256'].items():
            require(sha_file(path) == digest, 'Historical source manifest changed')
    for key, frozen in plan['models'].items():
        current = metadata(frozen['path'])
        require(all(current[name] == frozen[name] for name in current), f'{key} model metadata/shards changed')


def metrics(endpoints):
    n = len(endpoints)
    require(n > 0, 'Empty outcome cell')
    parsed = [e for e in endpoints if e['parsed']]
    return dict(n=n, correct=sum(e['correct'] for e in endpoints),
                exact=sum(e['correct'] for e in endpoints) / n,
                parsed=len(parsed), completed=sum(e['completed'] for e in endpoints),
                truncated=sum(e['truncated'] for e in endpoints),
                mae=mean(e['absolute_error'] for e in parsed) if parsed else None,
                mae_denominator=len(parsed))


if __name__ == '__main__':
    from scripts.probe_native_vision_reasoning_execute import cli
    cli()

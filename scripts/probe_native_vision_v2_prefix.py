"""CPU-checked prompt-prefix diagnostic for frozen native MMReD Vision.

First run --check-only in CPU Slurm; pass its exact plan.json to the GPU run.
No generation occurs while selecting or matching neutral prefixes. This reuses
36 fixed V2 main-test examples and is descriptive, not a new efficacy benchmark.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import re
from statistics import mean
import sys
import time


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
DATA_ROOT = Path('/mnt/data/gabriele/gnn_transformer/v2_clean')
OUTPUT_ROOT = REPO / 'outputs/native_aggregation_vlm/v2'
MODEL = 'Qwen/Qwen2.5-VL-7B-Instruct'
CONDITIONS = ('baseline', 'correct_prefix', 'neutral_prefix')
SOURCE_FILES = ('scripts/probe_native_vision_v2_prefix.py', 'gnnformer/data.py',
                'gnnformer/runtime.py')
NEUTRAL_PROSE = (
    'The following pictures show a sequence of illustrated scenes. Each picture '
    'represents a moment in that sequence. The scenes use a consistent drawing '
    'style and visual arrangement.'
)
DECODING = dict(do_sample=False, use_cache=True, max_new_tokens=4,
                repetition_penalty=1.0, output_logits=True)
LIMITS = [
    'Diagnostic subset of the existing V2 main test: 18 anchors, 36 examples; no model or prompt selection from outcomes.',
    'Correct and neutral prefixes match actual processor token layout, including image positions, but differ in semantics and question repetition.',
    'Baseline is shorter. Prefix-minus-baseline includes context length, positions, repetition and semantic changes.',
    'Intervals resample paired anchors, preserving both lengths; they are descriptive conditional intervals with no multiplicity adjustment.',
    'Frozen direct-answer model only; this is neither a new aggregation method nor evidence of reasoning composition.',
]


def sha_file(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def sha_object(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def select_records(manifest):
    if manifest.get('schema_version') != 1 or Path(manifest['dataset_root']).resolve() != DATA_ROOT:
        raise ValueError('Expected the registered V2 main manifest')
    indexes = {}
    for n in (16, 64):
        samples = manifest['splits'][f'test_N{n}']['samples']
        indexes[n] = {r['pair_id']: r for r in samples}
        if len(samples) != 108 or len(indexes[n]) != 108:
            raise ValueError('Incomplete or duplicated main-test anchors')
    if indexes[16].keys() != indexes[64].keys():
        raise ValueError('Main-test lengths do not share anchors')
    records = []
    for gold in range(9):
        anchors = sorted(k for k, r in indexes[16].items() if r['gold'] == gold)
        if len(anchors) != 12:
            raise ValueError('Incorrect registered count balance')
        for anchor in anchors[:2]:
            for n in (16, 64):
                sample = indexes[n][anchor]
                if sample['gold'] != gold or sample['n_frames'] != n or sample['test_family'] != 'length':
                    raise ValueError('Paired anchor metadata differs')
                directory = Path(sample['path']).resolve()
                if directory.parent != DATA_ROOT / 'mmred_vfiltered' / f'seq_len_{n}' / 'test':
                    raise ValueError('Sample path escapes its registered split')
                raw = (directory / 'qa.txt').read_bytes()
                if hashlib.sha256(raw).hexdigest() != sample['qa_sha256']:
                    raise ValueError('QA checksum differs')
                lines = raw.decode().splitlines()
                q_index, a_index = lines.index('question:'), lines.index('answer:')
                content = [line.strip() for line in lines[q_index + 1:a_index] if line.strip()]
                questions = [line for line in content if not (line.startswith('{') and line.endswith('}'))]
                answer = next(line.strip() for line in lines[a_index + 1:] if line.strip())
                if len(questions) != 1 or len(content) != n + 1 or int(answer) != gold:
                    raise ValueError('QA boundaries, frame count or gold differ')
                records.append(dict(path=str(directory), sid=sample['sid'], pair_id=anchor,
                                    n_frames=n, gold=gold, question=questions[0],
                                    qa_sha256=sample['qa_sha256'], image_files=sample['image_files']))
    if len(records) != 36 or Counter(r['n_frames'] for r in records) != {16: 18, 64: 18}:
        raise ValueError('Diagnostic selection differs from registration')
    return records


def verify_images(records):
    verified = {}
    for record in records:
        if len(record['image_files']) != record['n_frames']:
            raise ValueError('Image list length differs')
        for index, image in enumerate(record['image_files']):
            path = Path(image['path'])
            if path.resolve() != Path(record['path']) / f'{index:03d}.png':
                raise ValueError('Unexpected image path')
            stat = path.stat()
            key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            if key not in verified:
                verified[key] = sha_file(path)
            if stat.st_size != image['bytes'] or verified[key] != image['sha256']:
                raise ValueError('Image checksum or byte size differs')


def messages(record, final_prompt, prefix, frames=None):
    content = [] if prefix is None else [dict(type='text', text=prefix)]
    content += ([dict(type='image') for _ in range(record['n_frames'])] if frames is None
                else [dict(type='image', image=frame) for frame in frames])
    content += [dict(type='text', text=final_prompt)]
    return [dict(role='user', content=content)]


def template_ids(processor, record, final_prompt, prefix):
    text = processor.apply_chat_template(messages(record, final_prompt, prefix),
                                         add_generation_prompt=True, tokenize=False)
    return processor.tokenizer(text, add_special_tokens=False).input_ids


def token_layout(ids, image_id, marker_ids):
    positions = [i for i, token in enumerate(ids) if token == image_id]
    if not positions:
        raise ValueError('No image tokens in the processor layout')
    spans = []
    for index in positions:
        if spans and spans[-1][1] == index:
            spans[-1][1] += 1
        else:
            spans.append([index, index + 1])
    return dict(prompt_tokens=len(ids), input_ids_sha256=sha_object(ids),
                image_tokens=len(positions), image_token_spans=spans,
                vision_marker_positions=[[i, token] for i, token in enumerate(ids) if token in marker_ids])


def assert_matched(correct, neutral):
    for key in ('prompt_tokens', 'image_tokens', 'image_token_spans', 'vision_marker_positions'):
        if correct[key] != neutral[key]:
            raise ValueError(f'Correct/neutral processor layouts differ: {key}')


def neutral_prefix(processor, record, final_prompt, correct, image_id, marker_ids):
    target = token_layout(template_ids(processor, record, final_prompt, correct), image_id, marker_ids)
    tokenizer = processor.tokenizer
    tokens = tokenizer(NEUTRAL_PROSE, add_special_tokens=False).input_ids
    # Deterministic token truncation and punctuation adjustment, using layouts
    # only. Reject truncation inside a word; no model responses are available.
    candidates = []
    for width in range(1, len(tokens) + 1):
        stem = tokenizer.decode(tokens[:width], skip_special_tokens=False,
                                clean_up_tokenization_spaces=False).strip()
        if not NEUTRAL_PROSE.startswith(stem):
            continue
        if len(stem) < len(NEUTRAL_PROSE) and NEUTRAL_PROSE[len(stem)].isalnum():
            continue
        for ending in ('.\n', '\n', '.', ''):
            candidate = stem.rstrip('.,;:!?') + ending
            candidates.append(candidate)
    for candidate in dict.fromkeys(candidates):
        ids = template_ids(processor, record, final_prompt, candidate)
        layout = token_layout(ids, image_id, marker_ids)
        if all(layout[k] == target[k] for k in
               ('prompt_tokens', 'image_tokens', 'image_token_spans', 'vision_marker_positions')):
            return candidate
    raise ValueError('No neutral prose prefix matches the actual template token layout')


def fingerprint(processor, transformers_version):
    tokenizer = processor.tokenizer
    tokenizer_state = dict(vocab=tokenizer.get_vocab(), special_tokens=tokenizer.special_tokens_map,
                           padding_side=tokenizer.padding_side, truncation_side=tokenizer.truncation_side)
    if hasattr(tokenizer, 'backend_tokenizer'):
        tokenizer_state['backend'] = tokenizer.backend_tokenizer.to_str()
    elif hasattr(tokenizer, 'bpe_ranks'):
        tokenizer_state['bpe_merges'] = [list(pair) + [rank] for pair, rank in
                                       sorted(tokenizer.bpe_ranks.items(), key=lambda item: item[1])]
    else:
        raise ValueError('Unsupported tokenizer fingerprint representation')
    return dict(transformers_version=transformers_version,
                processor_class=type(processor).__name__, tokenizer_class=type(tokenizer).__name__,
                tokenizer_sha256=sha_object(tokenizer_state),
                chat_template_sha256=sha_object(processor.chat_template),
                image_processor_sha256=hashlib.sha256(processor.image_processor.to_json_string().encode()).hexdigest())


def prepare_all(processor, record, image_id, marker_ids):
    from PIL import Image
    from gnnformer.data import build_count_prompt, build_prompt_inputs
    frames = []
    try:
        for image in record['image_files']:
            with Image.open(image['path']) as source:
                converted = source.convert('RGB')
                try:
                    frames.append(converted.resize((392, 392)))
                finally:
                    converted.close()
        prompt = build_count_prompt(record['question'], record['n_frames'])
        inputs = {}
        for condition in CONDITIONS:
            prefix = None if condition == 'baseline' else record[condition]
            inputs[condition] = (build_prompt_inputs(processor, frames, prompt) if prefix is None else
                                 dict(processor.apply_chat_template(
                                     messages(record, prompt, prefix, frames), add_generation_prompt=True,
                                     tokenize=True, return_dict=True, return_tensors='pt')))
    finally:
        for frame in frames:
            frame.close()
    layouts = {}
    for condition, item in inputs.items():
        ids = item['input_ids'][0].tolist()
        if len(ids) + 4 > 16000:
            raise ValueError('Diagnostic prompt exceeds the registered context cap')
        grid = item['image_grid_thw'].tolist()
        if len(grid) != record['n_frames']:
            raise ValueError('Processor image count differs')
        layouts[condition] = dict(token_layout(ids, image_id, marker_ids), image_grid_thw=grid)
    assert_matched(layouts['correct_prefix'], layouts['neutral_prefix'])
    if any(not inputs[c]['pixel_values'].equal(inputs['baseline']['pixel_values']) for c in CONDITIONS):
        raise ValueError('Processed pixel values differ between prompt conditions')
    if any(layouts[c]['image_grid_thw'] != layouts['baseline']['image_grid_thw'] for c in CONDITIONS):
        raise ValueError('Image grids differ between prompt conditions')
    return inputs, layouts


def summarize(rows):
    parsed = [r for r in rows if r['prediction'] is not None]
    return dict(n=len(rows), correct=sum(r['exact'] for r in rows), exact=mean(r['exact'] for r in rows),
                parse_rate=len(parsed) / len(rows),
                mae_parsed=mean(abs(r['prediction'] - r['gold']) for r in parsed) if parsed else None,
                first_token_nll=mean(r['gold_first_token_nll'] for r in rows))


def analysis(rows):
    import numpy as np
    keyed = {(r['condition'], r['pair_id'], r['n_frames']): r for r in rows}
    anchors = sorted({r['pair_id'] for r in rows})
    if len(rows) != 108 or len(keyed) != 108 or len(anchors) != 18:
        raise ValueError('Incomplete diagnostic outputs')
    for anchor in anchors:
        if len({keyed[c, anchor, n]['gold'] for c in CONDITIONS for n in (16, 64)}) != 1:
            raise ValueError('Diagnostic paired labels differ')
    comparisons = {}
    for control in ('neutral_prefix', 'baseline'):
        comparisons[control] = {}
        for label, lengths in (('N16', (16,)), ('N64', (64,)), ('pooled', (16, 64))):
            differences = np.array([mean(int(keyed['correct_prefix', anchor, n]['exact'])
                                        - int(keyed[control, anchor, n]['exact']) for n in lengths)
                                    for anchor in anchors])
            rng = np.random.default_rng(20260910)
            draws = differences[rng.integers(0, len(anchors), size=(10000, len(anchors)))].mean(axis=1)
            comparisons[control][label] = dict(n=18 * len(lengths), anchors=18,
                                             exact_gain=float(differences.mean()),
                                             descriptive_bootstrap95=np.quantile(draws, [.025, .975]).tolist())
    return dict(cells={f'{condition}_N{n}': summarize([r for r in rows if r['condition'] == condition and r['n_frames'] == n])
                       for condition in CONDITIONS for n in (16, 64)},
                correct_prefix_minus_control=comparisons, limits=LIMITS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--plan', type=Path, help='Exact plan.json from a completed CPU check')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('All tokenizer, image, CPU analysis and GPU work must run through Slurm')
    if args.check_only == (args.plan is not None):
        raise SystemExit('Use --check-only on CPU or --plan on GPU, exclusively')
    output = args.output or OUTPUT_ROOT / ('prefix_check' if args.check_only else 'prefix')
    if not output.resolve().is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError('Diagnostic artifacts must remain under the V2 output directory')
    started = time.monotonic()
    manifest_path = DATA_ROOT / 'main_manifest.json'
    records = select_records(json.loads(manifest_path.read_text()))
    verify_images(records)
    selection_sha = sha_object(records)
    source_hashes = {name: sha_file(REPO / name) for name in SOURCE_FILES}
    plan = None
    if args.plan:
        if not args.plan.resolve().is_relative_to(OUTPUT_ROOT.resolve()):
            raise ValueError('CPU plan must be an existing V2 diagnostic artifact')
        plan = json.loads(args.plan.read_text())
        if (plan.get('schema_version') != 1 or not plan.get('all_layouts_verified')
                or plan['manifest_sha256'] != sha_file(manifest_path)
                or plan['selection_sha256'] != selection_sha or plan['source_sha256'] != source_hashes
                or plan['model'] != MODEL or plan['resize'] != 392 or plan['conditions'] != list(CONDITIONS)):
            raise ValueError('CPU plan does not match current data, source or diagnostic settings')
        for record, checked in zip(records, plan['records']):
            if any(record[key] != checked[key] for key in record):
                raise ValueError('CPU plan sample identity differs')
        if len(plan['records']) != 36:
            raise ValueError('CPU plan has an incorrect sample count')
        records = plan['records']

    # No heavy imports occur before the Slurm and provenance guards.
    import torch
    import torch.nn.functional as F
    from transformers import AutoProcessor, __version__ as transformers_version
    from gnnformer.data import build_count_prompt
    torch.set_num_threads(max(1, min(4, int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))))
    torch.manual_seed(0)
    model = runtime = None
    if args.check_only:
        processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True, use_fast=False,
                                                  local_files_only=True)
    else:
        from gnnformer.runtime import load_runtime, move_to_device
        if not torch.cuda.is_available():
            raise SystemExit('GPU execution requires a Slurm CUDA allocation')
        runtime = load_runtime(MODEL, use_4bit=True, attn_implementation='sdpa', device_map='cuda')
        model, processor = runtime.model, runtime.processor
        model.requires_grad_(False)
        model.eval()
    tokenizer = processor.tokenizer
    if tokenizer.eos_token_id is None:
        raise ValueError('Tokenizer has no EOS token')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    processor_fingerprint = fingerprint(processor, transformers_version)
    if plan is not None and plan['processor_fingerprint'] != processor_fingerprint:
        raise ValueError('CPU and GPU processors/tokenizers differ')
    image_id = tokenizer.convert_tokens_to_ids('<|image_pad|>')
    marker_ids = {tokenizer.convert_tokens_to_ids(token) for token in ('<|vision_start|>', '<|vision_end|>')}
    if model is not None and model.config.image_token_id != image_id:
        raise ValueError('Processor and model image-token IDs differ')
    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{os.environ['SLURM_JOB_ID']}_{os.getpid()}"
    outdir = output / run_id
    outdir.mkdir(parents=True, exist_ok=False)
    code_dir = outdir / 'code'
    code_dir.mkdir()
    for name in SOURCE_FILES:
        (code_dir / name.replace('/', '_')).write_bytes((REPO / name).read_bytes())
    common = dict(schema_version=1, model=MODEL, resize=392, conditions=list(CONDITIONS),
                  manifest=str(manifest_path), manifest_sha256=sha_file(manifest_path),
                  selection_sha256=selection_sha, source_sha256=source_hashes,
                  processor_fingerprint=processor_fingerprint,
                  selection='First two lexicographically sorted pair_id anchors per K0..8; both N16 and N64',
                  slurm_job_id=os.environ['SLURM_JOB_ID'], run_id=run_id,
                  neutral_prose=NEUTRAL_PROSE, decoding=DECODING, limits=LIMITS)
    write_json(outdir / 'config.json', common)
    rows = []
    try:
        for index, record in enumerate(records):
            if args.check_only:
                record['correct_prefix'] = f"Question: {record['question']}\n"
                prompt = build_count_prompt(record['question'], record['n_frames'])
                record['neutral_prefix'] = neutral_prefix(processor, record, prompt,
                                                         record['correct_prefix'], image_id, marker_ids)
            inputs, layouts = prepare_all(processor, record, image_id, marker_ids)
            if args.check_only:
                record['layouts'] = layouts
                print(json.dumps(dict(checked=index + 1, sid=record['sid'], n=record['n_frames'],
                                      prefix=record['neutral_prefix'],
                                      prompt_tokens={c: layouts[c]['prompt_tokens'] for c in CONDITIONS})), flush=True)
                del inputs
                continue
            if layouts != record['layouts']:
                raise ValueError('Actual GPU-run processor layouts differ from the checked CPU plan')
            gold_ids = tokenizer(str(record['gold']), add_special_tokens=False).input_ids
            if len(gold_ids) != 1:
                raise ValueError('This diagnostic requires single-token familiar-count answers')
            # Fixed rotation balances condition order across the 36 examples.
            order = CONDITIONS[index % 3:] + CONDITIONS[:index % 3]
            for condition in order:
                device_inputs = move_to_device(inputs.pop(condition), runtime.device)
                torch.cuda.synchronize()
                generation_started = time.monotonic()
                with torch.inference_mode():
                    generated = model.generate(
                        **device_inputs, **DECODING, return_dict_in_generate=True,
                        output_scores=False, pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id, temperature=None, top_p=None, top_k=None)
                torch.cuda.synchronize()
                elapsed = time.monotonic() - generation_started
                answer_ids = generated.sequences[0, device_inputs['input_ids'].shape[1]:]
                text = tokenizer.decode(answer_ids, skip_special_tokens=True)
                match = re.fullmatch(r'\s*([0-9]+)\s*', text)
                prediction = int(match.group(1)) if match else None
                first_logits = generated.logits[0][0].float()
                row = dict(condition=condition, pair_id=record['pair_id'], sid=record['sid'],
                           path=record['path'], n_frames=record['n_frames'], gold=record['gold'],
                           prediction=prediction, exact=prediction == record['gold'], output_text=text,
                           generated_token_ids=answer_ids.cpu().tolist(),
                           gold_first_token_nll=-float(F.log_softmax(first_logits, -1)[gold_ids[0]]),
                           raw_first_token_argmax=int(first_logits.argmax()),
                           model_seconds=elapsed, layout=layouts[condition],
                           prefix=None if condition == 'baseline' else record[condition])
                rows.append(row)
                write_json(outdir / 'predictions.json', rows)
                print(json.dumps({key: row[key] for key in ('condition', 'sid', 'n_frames', 'gold', 'prediction', 'exact')}), flush=True)
                del generated, device_inputs, first_logits, answer_ids
            del inputs
        if args.check_only:
            result = dict(common, records=records, all_layouts_verified=True,
                          cpu_check_seconds=time.monotonic() - started)
            write_json(outdir / 'plan.json', result)
            (outdir / 'INDEX.md').write_text('# Prefix diagnostic CPU check\n\nAll 36 examples passed exact correct/neutral token-layout checks.\n\n[plan.json](plan.json) fixes prefixes and layouts before generation. No model was loaded.\n')
            print(f'CPU_PLAN={outdir / "plan.json"}', flush=True)
        else:
            result = dict(common, **analysis(rows), cpu_plan=str(args.plan.resolve()),
                          cpu_plan_sha256=sha_file(args.plan), seconds=time.monotonic() - started,
                          gpu=torch.cuda.get_device_name(0),
                          peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
            write_json(outdir / 'summary.json', result)
            lines = ['# Frozen MMReD Vision: question-prefix diagnostic', '',
                     '| Condition | N16 exact /18 | N64 exact /18 |', '|---|---:|---:|']
            for condition in CONDITIONS:
                cells = [result['cells'][f'{condition}_N{n}'] for n in (16, 64)]
                lines.append('| ' + condition + ' | ' + ' | '.join(f"{c['correct']}/18 ({100*c['exact']:.1f}%)" for c in cells) + ' |')
            lines += ['', 'Correct-prefix minus control; paired-anchor descriptive 95% intervals:', '']
            for control, contrasts in result['correct_prefix_minus_control'].items():
                for cell, contrast in contrasts.items():
                    lo, hi = contrast['descriptive_bootstrap95']
                    lines.append(f"- {control}, {cell}: {100*contrast['exact_gain']:+.1f}pp [{100*lo:+.1f}, {100*hi:+.1f}]pp.")
            lines += ['', *['- ' + limit for limit in LIMITS], '',
                      '[All 108 condition/example outputs](predictions.json) · [Verified summary](summary.json)', '']
            (outdir / 'REPORT.md').write_text('\n'.join(lines))
            (outdir / 'INDEX.md').write_text('# Prefix diagnostic\n\n[Report](REPORT.md) · [All outputs](predictions.json) · [Summary](summary.json)\n')
            print(f'PREFIX_REPORT={outdir / "REPORT.md"}', flush=True)
    finally:
        del model, runtime, processor
        gc.collect()
        if not args.check_only and torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

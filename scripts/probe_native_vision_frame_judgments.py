"""Diagnostic frame judgments in isolated images and ordinary full-sequence VLM prompts.

CPU Slurm --check-plan selects and audits the fixed 16 binding pairs, verifies
Yes/No as single-token continuations at every actual chat-template boundary,
and freezes a plan before model outcomes. GPU execution requires that plan.
No training, external evidence tally, attention fence, or cached visual feature
is used. The isolated image retains its original visible Step label.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from scripts.probe_native_vision_binding import (
    DATA_BASE, digest, exclusive_json, selected_model, verify_canonical_unchanged,
    verify_manifest,
)
from scripts.probe_native_vision_v2_prefix import fingerprint, sha_object

OUTPUT_ROOT = REPO / 'outputs/native_aggregation_vlm/v3/frame_judgments'
MODEL = 'Qwen/Qwen2.5-VL-7B-Instruct'
PRESENTATIONS = ('isolated', 'full')
STRATA = ('C_switch', 'R_swapped', 'positive_unchanged')
MODEL_KEYS = ('frozen', 'v2_hidden_seed0')
DECODING = dict(do_sample=False, use_cache=True, max_new_tokens=4,
                repetition_penalty=1.0, output_logits=True)
SOURCE_FILES = ('scripts/probe_native_vision_frame_judgments.py',
                'scripts/probe_native_vision_binding.py',
                'scripts/probe_native_vision_v2_prefix.py',
                'slurm/native_vision_frame_judgments_check.sbatch',
                'slurm/native_vision_frame_judgments.sbatch',
                'gnnformer/constants.py', 'gnnformer/fencing.py')
LIMITS = [
    'Diagnostic-only, fixed 16 OOD binding pairs; models fixed in advance: frozen base and V2 hidden seed0, independent of V3 rankings.',
    'Six selected original frame positions per scene, both low/high scenes and isolated/full presentations: 384 judgments per model, 768 total.',
    'Isolated images retain original pixels and Step labels; the identical indexed question is used in both presentations. Context, image position, and indexing demands change.',
    'Primary readout is the raw first-token logit(Yes)-logit(No); full-vocabulary binary mass and gold-token probability expose off-format readouts.',
    'Generation is secondary: stripped, case-folded Yes/No only. Additional words or punctuation are unparsed and incorrect; all examples remain in accuracy denominators.',
    'Intervals resample all judgments from each of 16 pairs together. They are descriptive conditional intervals, with no multiplicity or seed-population inference.',
    'R-swapped and unchanged-positive labels are invariant, but their margins need not be invariant under changed images or surrounding context; those changes have no imposed direction.',
    'The preceding fixed-marginal count probe showed positive count changes for every pair in every tested model. This diagnostic examines precision and context effects; it does not presume absent conjunction binding.',
    'Success on selected frame questions does not prove conjunction information survives ordinary counting prompts or identify where aggregation fails.',
    'This is not a trained frame classifier, external tally method, new aggregation architecture, or test of reasoning composition.',
]


def frames_from_qa(record):
    lines = (Path(record['path']) / 'qa.txt').read_text().splitlines()
    block = lines[lines.index('question:') + 1:lines.index('answer:')]
    states = [ast.literal_eval(line.strip()) for line in block if line.strip().startswith('{')]
    frames = []
    for state in states:
        occupants = [(c, r) for r, chars in state['rooms'].items() for c in chars]
        if len(occupants) != 1:
            raise ValueError('Expected one character per frame')
        frames.append(occupants[0])
    return frames


def select_tasks(samples, frame_lookup):
    """Choose the first two frozen swap entries, not outcome-dependent positions."""
    pairs = defaultdict(dict)
    for record in samples:
        if record['condition'] in pairs[record['pair_id']]:
            raise ValueError('Duplicate pair condition')
        pairs[record['pair_id']][record['condition']] = record
    if len(pairs) != 16 or any(set(pair) != {'low', 'high'} for pair in pairs.values()):
        raise ValueError('Expected exactly16 complete binding pairs')
    tasks = []
    for pair_id in sorted(pairs):
        pair = pairs[pair_id]
        low, high = pair['low'], pair['high']
        swaps = low['room_swap_pairs']
        if len(swaps) != 4 or swaps != high['room_swap_pairs']:
            raise ValueError('Expected four shared frozen room-swap entries')
        character, room = low['target_character'], low['target_room']
        originals = frame_lookup[low['sid']]
        positives = [i for i, (c, r) in enumerate(originals) if (c, r) == (character, room)]
        if len(positives) != 2:
            raise ValueError('Expected two unchanged original positive frames')
        positions = {'C_switch': [entry[0] for entry in swaps[:2]],
                     'R_swapped': [entry[1] for entry in swaps[:2]],
                     'positive_unchanged': positives}
        if len({p for selected in positions.values() for p in selected}) != 6:
            raise ValueError('Frame selection is not six distinct positions')
        for condition in ('low', 'high'):
            record = pair[condition]
            frames = frame_lookup[record['sid']]
            for stratum in STRATA:
                for position in positions[stratum]:
                    expected = (stratum == 'positive_unchanged' or
                                (stratum == 'C_switch' and condition == 'high'))
                    actual = frames[position] == (character, room)
                    if actual != expected:
                        raise ValueError('Selected frame stratum/gold mismatch')
                    question = f'In frame {position + 1}, is {character} in the {room}? Answer Yes or No.'
                    for presentation in PRESENTATIONS:
                        tasks.append(dict(
                            task_id=f'{pair_id}/{condition}/{stratum}/{position}/{presentation}',
                            pair_id=pair_id, condition=condition, stratum=stratum,
                            presentation=presentation, sid=record['sid'], path=record['path'],
                            position_zero_based=position, visible_step=position + 1,
                            target_character=character, target_room=room,
                            question=question, gold='Yes' if actual else 'No',
                            image_count=1 if presentation == 'isolated' else 32,
                        ))
    balance = Counter((t['presentation'], t['gold']) for t in tasks)
    if len(tasks) != 384 or len({t['task_id'] for t in tasks}) != 384 or any(
            balance[p, label] != 96 for p in PRESENTATIONS for label in ('Yes', 'No')):
        raise ValueError('Incorrect judgment count or Yes/No balance')
    return tasks


def template_messages(task):
    return [dict(role='user', content=[dict(type='image') for _ in range(task['image_count'])]
                 + [dict(type='text', text=task['question'])])]


def one_token_suffix(prefix, completed):
    if completed[:len(prefix)] != prefix or len(completed) != len(prefix) + 1:
        raise ValueError('Yes/No is not a single-token continuation at this exact template boundary')
    return completed[-1]


def token_plan(processor, task, image_id, tokens_per_image):
    text = processor.apply_chat_template(template_messages(task), add_generation_prompt=True, tokenize=False)
    tokenizer = processor.tokenizer
    ids = tokenizer(text, add_special_tokens=False).input_ids
    labels = {label: one_token_suffix(ids, tokenizer(text + label, add_special_tokens=False).input_ids)
              for label in ('Yes', 'No')}
    if labels['Yes'] == labels['No'] or any(tokenizer.decode([token]) != label for label, token in labels.items()):
        raise ValueError('Fixed Yes/No continuation identities differ')
    if ids.count(image_id) != task['image_count']:
        raise ValueError('Unexpected unexpanded processor image marker count')
    expanded = [piece for token in ids for piece in ([token] * tokens_per_image if token == image_id else [token])]
    if len(expanded) + 4 > 16000:
        raise ValueError('Diagnostic prompt would exceed the fixed context cap')
    return dict(template_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                template_ids_sha256=sha_object(ids), input_ids_sha256=sha_object(expanded),
                prompt_tokens=len(expanded), yes_no_ids=labels,
                template_boundary_ids=ids[-16:],
                continued_boundary_ids={label: ids[-16:] + [token] for label, token in labels.items()})


def load_resized_images(sample):
    from PIL import Image
    frames = []
    try:
        for image in sample['image_files']:
            with Image.open(image['path']) as source:
                rgb = source.convert('RGB')
                try:
                    frames.append(rgb.resize((392, 392)))
                finally:
                    rgb.close()
    except BaseException:
        for frame in frames:
            frame.close()
        raise
    return frames


def prepare(processor, task, frames):
    from gnnformer.data import build_prompt_inputs
    selected = [frames[task['position_zero_based']]] if task['presentation'] == 'isolated' else frames
    return build_prompt_inputs(processor, selected, task['question'])


def check_inputs(inputs, task, checked, image_id, grid, tokens_per_image):
    ids = inputs['input_ids'][0].tolist()
    if sha_object(ids) != checked['input_ids_sha256'] or len(ids) != checked['prompt_tokens']:
        raise ValueError('Actual processed input IDs differ from the CPU-frozen template')
    if inputs['image_grid_thw'].tolist() != [grid] * task['image_count']:
        raise ValueError('Actual image grid differs from CPU calibration')
    if ids.count(image_id) != tokens_per_image * task['image_count']:
        raise ValueError('Actual image-token expansion differs')


def source_hashes(hidden):
    paths = set(SOURCE_FILES) | set(hidden['source_sha256'])
    return {name: digest(REPO / name) for name in sorted(paths)}


def snapshot_sources(output, hashes):
    destination = output / 'code'
    destination.mkdir()
    for name, sha in hashes.items():
        payload = (REPO / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != sha:
            raise ValueError('Source changed while freezing the plan')
        (destination / name.replace('/', '_')).write_bytes(payload)


def processor_settings(processor):
    return {key: getattr(processor.image_processor, key, None)
            for key in ('size', 'min_pixels', 'max_pixels', 'patch_size', 'temporal_patch_size', 'merge_size')}


def parse_generated(text):
    normalized = text.strip().casefold()
    return {'yes': 'Yes', 'no': 'No'}.get(normalized)


def self_test():
    assert parse_generated(' YES\n') == 'Yes' and parse_generated('no') == 'No'
    assert parse_generated('Yes.') is None and parse_generated('Yes because') is None
    assert one_token_suffix([1, 2], [1, 2, 3]) == 3
    for bad in ([1, 2, 3, 4], [1, 4, 3], [1, 2]):
        try:
            one_token_suffix([1, 2], bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Unsupported token continuation accepted')
    samples, lookup = [], {}
    for index in range(16):
        pair = f'pair{index}'
        low = [('C', 'R')] * 2 + [('C', 'X')] * 10 + [('D', 'R')] * 10 + [('D', 'X')] * 10
        high = list(low)
        swaps = [[i, i + 10] for i in range(2, 6)]
        for left, right in swaps:
            high[left], high[right] = ('C', 'R'), ('D', 'X')
        for condition, frames in [('low', low), ('high', high)]:
            sid = pair + condition
            samples.append(dict(pair_id=pair, condition=condition, sid=sid, path='/unused/' + sid,
                                target_character='C', target_room='R', room_swap_pairs=swaps))
            lookup[sid] = frames
    tasks = select_tasks(samples, lookup)
    for pair in {task['pair_id'] for task in tasks}:
        for presentation in PRESENTATIONS:
            assert Counter(task['gold'] for task in tasks if task['pair_id'] == pair and task['presentation'] == presentation) == {'Yes': 6, 'No': 6}
    assert {task['visible_step'] for task in tasks if task['stratum'] == 'positive_unchanged'} == {1, 2}
    synthetic = []
    for task in tasks:
        margin = 1.0 if task['gold'] == 'Yes' else -1.0
        synthetic.append(dict(task, margin_yes_minus_no=margin, signed_correct_margin=1.0,
                              margin_prediction=task['gold'], margin_correct=True, margin_tie=False,
                              binary_probability_mass=0.8, gold_token_probability=0.6,
                              prediction=task['gold'], parsed=True, exact=True))
    synthetic[0].update(prediction=None, parsed=False, exact=False)
    report = analyze(synthetic, tasks)
    isolated = report['cells']['isolated/all']
    assert isolated['n'] == 192 and isolated['parsed'] == 191 and isolated['generated_correct'] == 191
    assert isolated['generated_exact_given_parsed'] == 1.0
    assert report['full_minus_isolated']['all']['signed_correct_margin']['mean'] == 0
    for presentation in PRESENTATIONS:
        assert report['high_minus_low'][f'{presentation}/C_switch']['metrics']['raw_margin_delta']['mean'] == 2
        for stratum in ('R_swapped', 'positive_unchanged'):
            assert report['high_minus_low'][f'{presentation}/{stratum}']['metrics']['raw_margin_delta']['mean'] == 0
    assert report['high_minus_low']['isolated/C_switch']['complete_generated_pairs'] == 31
    print('PASS: task/label balance, original indices, strict parsing, continuation rejection, paired directions and missing-parse report denominators', flush=True)


def check_plan(args):
    self_test()
    records, metadata = verify_manifest(args.manifest)
    manifest = json.loads(args.manifest.read_text())
    samples = manifest['splits']['test_N32']['samples']
    tasks = select_tasks(samples, {sample['sid']: frames_from_qa(sample) for sample in samples})
    hidden = selected_model('v2', 'hidden', 0, args.v2_root)
    hashes = source_hashes(hidden)
    import torch
    from transformers import AutoProcessor, __version__ as transformers_version
    torch.set_num_threads(min(4, int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))))
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True, use_fast=False, local_files_only=True)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.eos_token_id is None:
        raise ValueError('Missing EOS token')
    if processor_settings(processor) != hidden['base_runtime']['image_processor_settings']:
        raise ValueError('CPU image processor differs from the canonical fixed V2 hidden run')
    image_id = tokenizer.convert_tokens_to_ids('<|image_pad|>')
    indexed = {sample['sid']: sample for sample in samples}
    calibration_task = tasks[0]
    frames = load_resized_images(indexed[calibration_task['sid']])
    try:
        initial = prepare(processor, calibration_task, frames)
        grids = initial['image_grid_thw'].tolist()
        if len(grids) != 1:
            raise ValueError('Expected isolated-image CPU calibration')
        grid = grids[0]
        tokens_per_image = initial['input_ids'][0].tolist().count(image_id)
        if grid != [1, 28, 28] or tokens_per_image != 196:
            raise ValueError('Unexpected 392px canonical image layout')
        for task in tasks:
            task['token_check'] = token_plan(processor, task, image_id, tokens_per_image)
        for task in tasks[:2]:
            inputs = prepare(processor, task, frames)
            check_inputs(inputs, task, task['token_check'], image_id, grid, tokens_per_image)
            del inputs
        del initial
    finally:
        for frame in frames:
            frame.close()
    labels = tasks[0]['token_check']['yes_no_ids']
    if any(task['token_check']['yes_no_ids'] != labels for task in tasks):
        raise ValueError('Yes/No token identities vary across checked contexts')
    if (str(torch.__version__) != hidden['base_runtime']['torch_version'] or
            str(transformers_version) != hidden['base_runtime']['transformers_version']):
        raise ValueError('Software differs from the fixed V2 hidden model')
    plan = dict(schema_version=1, purpose='frame_judgments', model=MODEL, resize=392,
                created_slurm_job_id=os.environ['SLURM_JOB_ID'], manifest=metadata, records=records,
                samples=samples, tasks=tasks, tasks_sha256=sha_object(tasks), models=list(MODEL_KEYS),
                hidden=hidden, source_sha256=hashes, yes_no_ids=labels,
                processor_fingerprint=fingerprint(processor, transformers_version),
                torch_version=str(torch.__version__), image_id=image_id, image_grid=grid,
                tokens_per_image=tokens_per_image, decoding=DECODING, limits=LIMITS,
                selection='First two entries of frozen room_swap_pairs, both endpoints, plus both original positive positions in ascending order',
                all_template_boundaries_single_token_verified=True,
                isolated_and_full_actual_processor_calibrated=True)
    output = args.output / f'check_{os.environ["SLURM_JOB_ID"]}'
    output.mkdir(parents=True, exist_ok=False)
    snapshot_sources(output, hashes)
    exclusive_json(output / 'plan.json', plan)
    (output / 'plan.sha256').write_text(digest(output / 'plan.json') + '\n')
    (output / 'manifest.json').write_bytes(args.manifest.read_bytes())
    print(f'CPU plan complete: {output / "plan.json"}; Yes/No ids={labels}; 384 tasks/model; no model outcomes', flush=True)


def validate_plan(path):
    if not path.resolve().is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError('Plan must be a completed frame-judgment CPU artifact')
    if digest(path) != path.with_name('plan.sha256').read_text().strip():
        raise ValueError('CPU plan checksum differs')
    plan = json.loads(path.read_text())
    if (plan['schema_version'] != 1 or plan['purpose'] != 'frame_judgments'
            or plan['models'] != list(MODEL_KEYS) or plan['model'] != MODEL
            or plan['resize'] != 392 or plan['decoding'] != DECODING
            or not plan['all_template_boundaries_single_token_verified']
            or not plan['isolated_and_full_actual_processor_calibrated']
            or len(plan['tasks']) != 384 or plan['tasks_sha256'] != sha_object(plan['tasks'])):
        raise ValueError('CPU plan protocol differs')
    if any(digest(REPO / name) != sha for name, sha in plan['source_sha256'].items()):
        raise ValueError('Source changed after CPU plan freeze')
    records, metadata = verify_manifest(Path(plan['manifest']['path']))
    if records != plan['records'] or metadata != plan['manifest']:
        raise ValueError('Binding data differs from CPU plan')
    manifest = json.loads(Path(plan['manifest']['path']).read_text())
    if manifest['splits']['test_N32']['samples'] != plan['samples']:
        raise ValueError('Full sample metadata changed')
    tasks = select_tasks(plan['samples'], {sample['sid']: frames_from_qa(sample) for sample in plan['samples']})
    if tasks != [{key: value for key, value in task.items() if key != 'token_check'} for task in plan['tasks']]:
        raise ValueError('Task selection differs from the pre-outcome CPU plan')
    verify_canonical_unchanged(plan['hidden'])
    return plan


def restore_hidden(runtime, metadata):
    import torch
    from gnnformer.aggregation_controls import attach_hidden_only_adapter
    from gnnformer.carriers import attach_lora
    from gnnformer.runtime import get_layers
    arch = metadata['architecture']
    saved = torch.load(metadata['checkpoint'], map_location='cpu', weights_only=True)
    if (saved['architecture'] != arch or saved['config']['run_id'] != metadata['run_id']
            or saved['epoch'] != metadata['selected_epoch']
            or saved['config']['code_sha256'] != metadata['source_sha256']):
        raise ValueError('Fixed hidden checkpoint provenance differs')
    branch = attach_hidden_only_adapter(runtime.model, arch['layer_index'], rank=arch['hidden_rank'])
    branch.load_state_dict(saved['branch'], strict=True)
    branch.mode = 'all'
    branch.requires_grad_(False)
    layers = get_layers(runtime.model)
    lora = attach_lora(layers, len(layers) - arch['lora_layers'], rank=arch['lora_rank'],
                       alpha=arch['lora_alpha'], device=runtime.device)
    if set(saved['lora']) != {f'{index}.{name}' for index, name in lora.params}:
        raise ValueError('Checkpoint LoRA keys differ')
    with torch.no_grad():
        for (index, name), parameters in lora.params.items():
            stored_pair = saved['lora'][f'{index}.{name}']
            if len(stored_pair) != 2:
                raise ValueError('Checkpoint LoRA pair malformed')
            for parameter, stored in zip(parameters, stored_pair):
                if parameter.shape != stored.shape or not torch.isfinite(stored).all():
                    raise ValueError('Checkpoint LoRA shape/value differs')
                parameter.copy_(stored.to(parameter))
                parameter.requires_grad_(False)
    if (any(not torch.isfinite(parameter).all() for parameter in branch.parameters())
            or sum(p.numel() for p in branch.parameters()) != metadata['parameters']['branch']
            or lora.num_parameters() != metadata['parameters']['lora']):
        raise ValueError('Restored adapter values/parameter counts differ')
    return branch, lora


def worker(args):
    plan = validate_plan(args.plan)
    import torch
    import torch.nn.functional as F
    from transformers import __version__ as transformers_version
    from gnnformer.runtime import load_runtime, move_to_device
    if not torch.cuda.is_available():
        raise SystemExit('One Slurm CUDA allocation is required')
    torch.set_num_threads(min(4, int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))))
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    runtime = load_runtime(MODEL, use_4bit=True, attn_implementation='sdpa', device_map='cuda')
    runtime.model.requires_grad_(False)
    branch = lora = None
    if args.worker == 'v2_hidden_seed0':
        branch, lora = restore_hidden(runtime, plan['hidden'])
    runtime.model.eval()
    processor, tokenizer = runtime.processor, runtime.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if (fingerprint(processor, transformers_version) != plan['processor_fingerprint']
            or str(torch.__version__) != plan['torch_version']
            or runtime.model.config.image_token_id != plan['image_id']
            or processor_settings(processor) != plan['hidden']['base_runtime']['image_processor_settings']):
        raise ValueError('GPU model/processor/software differs from CPU plan')
    if any(parameter.requires_grad for parameter in runtime.model.parameters()):
        raise ValueError('Diagnostic must have no trainable model parameters')
    output = args.worker_output
    output.mkdir(parents=True, exist_ok=False)
    samples = {sample['sid']: sample for sample in plan['samples']}
    rows, frames, current_sid = [], [], None
    started = time.monotonic()
    try:
        for index, task in enumerate(plan['tasks']):
            if task['sid'] != current_sid:
                for frame in frames:
                    frame.close()
                frames = load_resized_images(samples[task['sid']])
                current_sid = task['sid']
            preprocessing_start = time.monotonic()
            inputs = prepare(processor, task, frames)
            check_inputs(inputs, task, task['token_check'], plan['image_id'], plan['image_grid'], plan['tokens_per_image'])
            if token_plan(processor, task, plan['image_id'], plan['tokens_per_image']) != task['token_check']:
                raise ValueError('GPU template/token continuation changed')
            inputs = move_to_device(inputs, runtime.device)
            torch.cuda.synchronize()
            preprocessing = time.monotonic() - preprocessing_start
            generation_start = time.monotonic()
            with torch.inference_mode():
                generated = runtime.model.generate(
                    **inputs, **DECODING, return_dict_in_generate=True, output_scores=False,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                    temperature=None, top_p=None, top_k=None)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - generation_start
            first = generated.logits[0][0].float()
            logprob = F.log_softmax(first, dim=-1)
            yes, no = plan['yes_no_ids']['Yes'], plan['yes_no_ids']['No']
            margin = float(first[yes] - first[no])
            signed = margin if task['gold'] == 'Yes' else -margin
            gold_id = yes if task['gold'] == 'Yes' else no
            top_id = int(first.argmax())
            answer_ids = generated.sequences[0, inputs['input_ids'].shape[1]:].cpu().tolist()
            text = tokenizer.decode(answer_ids, skip_special_tokens=True)
            prediction = parse_generated(text)
            row = {key: value for key, value in task.items() if key != 'token_check'}
            row.update(model=args.worker, margin_yes_minus_no=margin, signed_correct_margin=signed,
                       margin_prediction='Yes' if margin > 0 else 'No' if margin < 0 else None,
                       margin_correct=signed > 0, margin_tie=margin == 0,
                       yes_logit=float(first[yes]), no_logit=float(first[no]),
                       yes_probability=float(logprob[yes].exp()), no_probability=float(logprob[no].exp()),
                       binary_probability_mass=float(logprob[[yes, no]].exp().sum()),
                       gold_token_probability=float(logprob[gold_id].exp()),
                       top1_id=top_id, top1_text=tokenizer.decode([top_id], skip_special_tokens=False),
                       output_text=text, generated_token_ids=answer_ids, prediction=prediction,
                       parsed=prediction is not None, exact=prediction == task['gold'],
                       prompt_tokens=inputs['input_ids'].shape[1], preprocessing_seconds=preprocessing,
                       model_seconds=elapsed)
            if not all(math.isfinite(row[key]) for key in ('margin_yes_minus_no', 'signed_correct_margin',
                                                           'binary_probability_mass', 'gold_token_probability')):
                raise ValueError('Nonfinite native first-token statistics')
            rows.append(row)
            with (output / 'predictions.partial.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            del generated, inputs, first, logprob
            if (index + 1) % 24 == 0:
                print(f'{args.worker}: {index + 1}/384 judgments; {time.monotonic() - started:.1f}s', flush=True)
    finally:
        for frame in frames:
            frame.close()
    verify_canonical_unchanged(plan['hidden'])
    exclusive_json(output / 'predictions.json', rows)
    exclusive_json(output / 'provenance.json', dict(model=args.worker, plan_sha256=digest(args.plan),
                   processor_fingerprint=plan['processor_fingerprint'], source_sha256=plan['source_sha256'],
                   checkpoint=plan['hidden']['checkpoint'] if args.worker == 'v2_hidden_seed0' else None,
                   checkpoint_sha256=plan['hidden']['checkpoint_sha256'] if args.worker == 'v2_hidden_seed0' else None,
                   elapsed_seconds=time.monotonic() - started,
                   peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(), judgments=len(rows), training=False))


def cluster_summary(values):
    """values: (pair_id, scalar); equally weighted fixed pair clusters."""
    import numpy as np
    grouped = defaultdict(list)
    for pair, value in values:
        grouped[pair].append(float(value))
    if len(grouped) != 16:
        raise ValueError('Every cluster summary must retain all16 pairs')
    pair_means = np.array([mean(grouped[pair]) for pair in sorted(grouped)])
    rng = np.random.default_rng(20260913)
    samples = pair_means[rng.integers(0, 16, size=(10000, 16))].mean(axis=1)
    return dict(mean=float(pair_means.mean()), descriptive_pair_bootstrap95=np.quantile(samples, [.025, .975]).tolist(),
                pairs=16, observations=len(values))


def summarize_rows(rows):
    parsed = [row for row in rows if row['parsed']]
    scalar = ('margin_correct', 'signed_correct_margin', 'binary_probability_mass', 'gold_token_probability', 'exact')
    return dict(n=len(rows), yes_gold=sum(row['gold'] == 'Yes' for row in rows),
                no_gold=sum(row['gold'] == 'No' for row in rows),
                margin_correct=sum(row['margin_correct'] for row in rows), margin_ties=sum(row['margin_tie'] for row in rows),
                generated_correct=sum(row['exact'] for row in rows), parsed=len(parsed), parse_rate=len(parsed) / len(rows),
                generated_exact_given_parsed=mean(row['exact'] for row in parsed) if parsed else None,
                metrics={key: cluster_summary([(row['pair_id'], row[key]) for row in rows]) for key in scalar})


def analyze(rows, tasks):
    if len(rows) != 384 or [row['task_id'] for row in rows] != [task['task_id'] for task in tasks]:
        raise ValueError('Missing, reordered, or extra frame judgments')
    for row, task in zip(rows, tasks):
        if any(row[key] != value for key, value in task.items() if key != 'token_check'):
            raise ValueError('Output task metadata differs from CPU plan')
        if row['parsed'] != (row['prediction'] is not None) or row['exact'] != (row['prediction'] == row['gold']):
            raise ValueError('Incorrect secondary parsing bookkeeping')
    cells = {}
    for presentation in PRESENTATIONS:
        for stratum in ('all', *STRATA):
            selected = [r for r in rows if r['presentation'] == presentation and (stratum == 'all' or r['stratum'] == stratum)]
            cells[f'{presentation}/{stratum}'] = summarize_rows(selected)
            if stratum != 'all':
                for condition in ('low', 'high'):
                    cells[f'{presentation}/{stratum}/{condition}'] = summarize_rows([r for r in selected if r['condition'] == condition])
    indexed = {(r['pair_id'], r['condition'], r['position_zero_based'], r['presentation']): r for r in rows}
    full_minus_isolated = {}
    for stratum in ('all', *STRATA):
        isolated = [r for r in rows if r['presentation'] == 'isolated' and (stratum == 'all' or r['stratum'] == stratum)]
        full_minus_isolated[stratum] = {key: cluster_summary([
            (r['pair_id'], float(indexed[r['pair_id'], r['condition'], r['position_zero_based'], 'full'][key]) - float(r[key]))
            for r in isolated]) for key in ('signed_correct_margin', 'margin_correct', 'binary_probability_mass', 'gold_token_probability', 'exact')}
    high_minus_low = {}
    for presentation in PRESENTATIONS:
        for stratum in STRATA:
            low_rows = [r for r in rows if r['presentation'] == presentation and r['stratum'] == stratum and r['condition'] == 'low']
            pairs = [(low, indexed[low['pair_id'], 'high', low['position_zero_based'], presentation]) for low in low_rows]
            parsed = [(lo, hi) for lo, hi in pairs if lo['parsed'] and hi['parsed']]
            values = dict(
                raw_margin_delta=[(lo['pair_id'], hi['margin_yes_minus_no'] - lo['margin_yes_minus_no']) for lo, hi in pairs],
                absolute_margin_delta=[(lo['pair_id'], abs(hi['margin_yes_minus_no'] - lo['margin_yes_minus_no'])) for lo, hi in pairs],
                positive_margin_delta=[(lo['pair_id'], hi['margin_yes_minus_no'] > lo['margin_yes_minus_no']) for lo, hi in pairs],
                both_margin_correct=[(lo['pair_id'], lo['margin_correct'] and hi['margin_correct']) for lo, hi in pairs],
                both_generated_correct=[(lo['pair_id'], lo['exact'] and hi['exact']) for lo, hi in pairs],
                margin_decision_stable=[(lo['pair_id'], lo['margin_prediction'] is not None and lo['margin_prediction'] == hi['margin_prediction']) for lo, hi in pairs])
            high_minus_low[f'{presentation}/{stratum}'] = dict(
                metrics={key: cluster_summary(value) for key, value in values.items()},
                frame_pairs=len(pairs), complete_generated_pairs=len(parsed),
                generated_stable_given_both_parsed=mean(lo['prediction'] == hi['prediction'] for lo, hi in parsed) if parsed else None,
                expected_label_change='No→Yes' if stratum == 'C_switch' else 'No→No' if stratum == 'R_swapped' else 'Yes→Yes')
    return dict(cells=cells, full_minus_isolated=full_minus_isolated, high_minus_low=high_minus_low)


def execute(args):
    started = time.monotonic()
    plan = validate_plan(args.plan)
    output = args.output / f'judgments_{os.environ["SLURM_JOB_ID"]}'
    output.mkdir(parents=True, exist_ok=False)
    snapshot_sources(output, plan['source_sha256'])
    (output / 'plan.json').write_bytes(args.plan.read_bytes())
    results, all_rows = {}, {}
    for model in MODEL_KEYS:
        remaining = args.max_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError('Total frame diagnostic time budget exhausted')
        command = [sys.executable, str(Path(__file__).resolve()), '--plan', str(args.plan.resolve()),
                   '--worker', model, '--worker-output', str(output / model)]
        with (output / f'{model}.log').open('x') as log:
            subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=remaining, shell=False)
        rows = json.loads((output / model / 'predictions.json').read_text())
        provenance = json.loads((output / model / 'provenance.json').read_text())
        if (provenance['model'] != model or provenance['plan_sha256'] != digest(args.plan)
                or provenance['source_sha256'] != plan['source_sha256']
                or provenance['processor_fingerprint'] != plan['processor_fingerprint']
                or provenance['judgments'] != 384 or provenance['training'] is not False
                or any(row['model'] != model for row in rows)):
            raise ValueError('Frame worker provenance/count/model identity differs')
        expected_checkpoint = plan['hidden']['checkpoint_sha256'] if model == 'v2_hidden_seed0' else None
        if provenance['checkpoint_sha256'] != expected_checkpoint:
            raise ValueError('Frame worker restored an unexpected checkpoint')
        results[model], all_rows[model] = analyze(rows, plan['tasks']), rows
        results[model]['provenance'] = provenance
    validate_plan(args.plan)
    summary = dict(schema_version=1, plan_sha256=digest(args.plan), results=results, limits=LIMITS,
                   elapsed_seconds=time.monotonic() - started, judgments=768,
                   all_plan_data_sources_checkpoint_checks_passed=True)
    exclusive_json(output / 'summary.json', summary)
    exclusive_json(output / 'predictions.json', all_rows)
    lines = ['# Native frame-judgment diagnostic', '',
             '| Model | Presentation | First-token binary correct /192 | Generated correct /192 | Parsed /192 | Mean signed margin | Mean P(Yes)+P(No) |',
             '|---|---|---:|---:|---:|---:|---:|']
    for model, result in results.items():
        for presentation in PRESENTATIONS:
            cell = result['cells'][f'{presentation}/all']
            lines.append(f"| {model} | {presentation} | {cell['margin_correct']}/192 | {cell['generated_correct']}/192 | {cell['parsed']}/192 | {cell['metrics']['signed_correct_margin']['mean']:.4f} | {cell['metrics']['binary_probability_mass']['mean']:.4f} |")
    lines += ['', '| Model | Full−isolated signed margin [95% pair bootstrap] | C-switch isolated margin Δ | C-switch full margin Δ |',
              '|---|---:|---:|---:|']
    for model, result in results.items():
        contrast = result['full_minus_isolated']['all']['signed_correct_margin']
        lo, hi = contrast['descriptive_pair_bootstrap95']
        switch = [result['high_minus_low'][f'{p}/C_switch']['metrics']['raw_margin_delta']['mean'] for p in PRESENTATIONS]
        lines.append(f"| {model} | {contrast['mean']:.4f} [{lo:.4f}, {hi:.4f}] | {switch[0]:.4f} | {switch[1]:.4f} |")
    lines += ['', 'All margins are raw first-answer-token logits. Binary accuracy uses the sign of Yes−No, with ties incorrect; it is not native full-vocabulary top-1 accuracy. C-switch Δ is high-scene minus low-scene margin, where the frame label changes No→Yes.', '',
              'Detailed strata, low/high cells, invariant-label stability, probabilities, conditional parsed denominators, and paired intervals are in [summary.json](summary.json). Every judgment and native top-1 token are in [predictions.json](predictions.json).', '',
              '## Scope', ''] + ['- ' + limitation for limitation in LIMITS] + ['', '[Frozen CPU plan](plan.json)', '']
    (output / 'REPORT.md').write_text('\n'.join(lines))
    index = args.output / 'INDEX.md'
    if not index.exists():
        index.write_text('# Native frame-judgment diagnostics\n\n')
    with index.open('a') as stream:
        stream.write(f"- Slurm {os.environ['SLURM_JOB_ID']}: [report]({output.name}/REPORT.md), [summary]({output.name}/summary.json), 768 fixed diagnostic judgments.\n")
    print(f'Frame diagnostic complete: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DATA_BASE / 'v3_binding_probe/manifest.json')
    parser.add_argument('--v2-root', type=Path, default=REPO / 'outputs/native_aggregation_vlm/v2')
    parser.add_argument('--output', type=Path, default=OUTPUT_ROOT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-plan', action='store_true')
    mode.add_argument('--self-test', action='store_true')
    mode.add_argument('--plan', type=Path)
    parser.add_argument('--worker', choices=MODEL_KEYS)
    parser.add_argument('--worker-output', type=Path)
    parser.add_argument('--max-seconds', type=float, default=870)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('All tokenizer/image/CPU analysis and GPU work must run through Slurm')
    if not args.output.resolve().is_relative_to(OUTPUT_ROOT.resolve()) or args.max_seconds <= 0:
        raise ValueError('Invalid output path or budget')
    if bool(args.worker) != bool(args.worker_output) or (args.worker and args.plan is None):
        raise ValueError('Workers require both a checked plan and an output directory')
    if args.worker_output and not args.worker_output.resolve().is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError('Worker outputs must remain in frame diagnostic directory')
    if args.self_test:
        self_test()
    elif args.check_plan:
        check_plan(args)
    elif args.worker:
        worker(args)
    else:
        execute(args)


if __name__ == '__main__':
    main()

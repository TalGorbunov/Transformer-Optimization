"""Stage the registered V4 repeat-versus-refresh data control on Slurm CPU.

Both schedules have 9x180 presentations and exactly the same N/K/question per
slot. Refresh draws a new conditional V2 scene after block0, except ten saturated
N8/K8 slots whose fixed question determines the scene. Thus refresh contains
1540 distinct sequences, repeat180; 1360 new sequences are content-disjoint from
all prior selected data. Existing sources are never modified. CPU verification
includes every QA/image, independent gold recounts, schedule constraints, renderer
RGB parity, and actual processor input-token equality for every training slot.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from scripts.stage_native_vision_v2_clean import (
    CHARS, DATA_BASE, LAW, PARK_ROOMS, V1_ROOT, describe, digest, distractor,
    exclusive_json, prepare_renderer, publish_sample, render_cached, stable_seed,
)
from scripts.stage_native_vision_pilot import read_candidate, source_states
from scripts.audit_native_vision_data import inspect, summarize

V2_ROOT = DATA_BASE / 'v2_clean'
DESTINATION = DATA_BASE / 'v4_diversity'
OUTPUT_ROOT = REPO / 'outputs/native_aggregation_vlm/v4/data_staging'
MODEL = 'Qwen/Qwen2.5-VL-7B-Instruct'
SOURCE_FILES = ('scripts/stage_native_vision_v4_diversity.py',
                'slurm/native_aggregation_vision_v4_stage.sbatch',
                'scripts/stage_native_vision_v2_clean.py',
                'scripts/stage_native_vision_pilot.py',
                'scripts/audit_native_vision_data.py',
                'scripts/probe_native_vision_v2_prefix.py',
                'gnnformer/data.py', 'datasets/mmred/render_mmred.py')
V4_LAW = dict(base_conditional_law=LAW,
              target_assignment='Fixed original V2 N/K/question/character/room per slot across conditions and blocks',
              original_order='V2 main train_N8 listed order followed by train_N16 listed order',
              refresh='Independent slot/block-derived random draw, rejecting all prior/new whole-scene content collisions',
              saturated_exception='Ten N8/K8 slots necessarily reuse block0 in both conditions',
              blocks=9, slots=180, presentations=1620, training_frames=19440,
              repeat_distinct_scenes=180, refresh_distinct_scenes=1540, new_scenes=1360,
              evaluation_status='Copied V2 dev/test/count data; reused exploratory evaluation, not fresh confirmation')


def rows_of(manifest):
    rows = [row for cell in manifest.get('splits', {}).values() for row in cell['samples']]
    if not rows and 'pairs' in manifest:
        rows = [pair[condition] for pair in manifest['pairs'] for condition in ('low', 'high')]
    if not rows:
        raise ValueError('Exclusion manifest has no samples')
    return rows


def load_sources():
    paths = [V1_ROOT / f'{purpose}_manifest.json' for purpose in ('main', 'profile')]
    paths += [V2_ROOT / f'{purpose}_manifest.json' for purpose in
              ('main', 'count', 'profile', 'profile_count', 'causal_probe')]
    paths += [DATA_BASE / 'v3_binding_probe/manifest.json']
    manifests, hashes, excluded = {}, {}, set()
    for path in paths:
        if path.is_symlink():
            raise ValueError('Exclusion manifests cannot be symlinks')
        raw = path.read_bytes()
        manifest = json.loads(raw)
        manifests[str(path)] = manifest
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        for row in rows_of(manifest):
            value = row['content_sha256']
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError('Missing source content hash')
            excluded.add(value)
    return manifests, hashes, excluded


def original_slots(main):
    originals = main['splits']['train_N8']['samples'] + main['splits']['train_N16']['samples']
    if len(originals) != 180 or len({r['sid'] for r in originals}) != 180:
        raise ValueError('Original training must contain exactly180 unique SIDs')
    if Counter((r['n_frames'], r['gold']) for r in originals) != Counter({(n, k): 10 for n in (8, 16) for k in range(9)}):
        raise ValueError('Original training N/K balance differs')
    slots = []
    for index, row in enumerate(originals):
        character, room = row['target_character'], row['target_room']
        slots.append(dict(slot=index, original_slot=index, original_sid=row['sid'],
                          n_frames=row['n_frames'], gold=row['gold'],
                          question=f'How many frames show {character} in the {room}?',
                          target_character=character, target_room=room,
                          saturated=row['n_frames'] == row['gold']))
    if sum(slot['saturated'] for slot in slots) != 10:
        raise ValueError('Expected exactly ten saturated N8/K8 slots')
    return originals, slots


def fresh_scene(slot, block, used, seed):
    if block not in range(1, 9) or slot['saturated']:
        raise ValueError('Only nonsaturated slots in blocks1..8 can be refreshed')
    derived = stable_seed(seed, f'v4_diversity:block={block}:slot={slot["slot"]}')
    rng = random.Random(derived)
    n, k = slot['n_frames'], slot['gold']
    character, room = slot['target_character'], slot['target_room']
    for attempt in range(10000):
        frames = [(character, room)] * k + [distractor(rng, character, room) for _ in range(n - k)]
        rng.shuffle(frames)
        states, question, sha, counts = describe(frames, character, room)
        if question != slot['question'] or counts['matches'] != k:
            raise AssertionError('Fresh draw changed the slot question/gold')
        if sha in used:
            continue
        used.add(sha)
        return dict(sid=f'v4d_b{block:02d}_slot{slot["slot"]:03d}_N{n}_K{k}',
                    split='train', n_frames=n, gold=k, seed=derived, data_seed=seed,
                    slot=slot['slot'], block=block, original_sid=slot['original_sid'],
                    target_character=character, target_room=room, question=question,
                    frames=frames, states=states, content_sha256=sha, semantic_counts=counts,
                    generation_attempt=attempt, pair_id=None, anchor_id=None, test_family=None,
                    axis='training_scene_refresh')
    raise RuntimeError(f'Conditional finite pool exhausted for block{block}/slot{slot["slot"]}')


def make_schedules(slots, excluded, seed):
    original = [slot['original_sid'] for slot in slots]
    repeat, refresh, fresh, used = [list(original) for _ in range(9)], [list(original)], [], set(excluded)
    for block in range(1, 9):
        current = []
        for slot in slots:
            if slot['saturated']:
                current.append(slot['original_sid'])
            else:
                row = fresh_scene(slot, block, used, seed)
                fresh.append(row)
                current.append(row['sid'])
        refresh.append(current)
    if len(fresh) != 1360 or len(used) != len(excluded) + 1360:
        raise AssertionError('Incorrect fresh scene count or exclusions')
    return dict(repeat=repeat, refresh=refresh), fresh


def copy_sample(source, destination, split=None):
    split = split or source['split']
    original = Path(source['path']).resolve()
    directory = destination / 'mmred_vfiltered' / f'seq_len_{source["n_frames"]}' / split / source['sid']
    directory.mkdir(parents=True, exist_ok=False)
    payload = (original / 'qa.txt').read_bytes()
    if hashlib.sha256(payload).hexdigest() != source['qa_sha256']:
        raise ValueError('Source QA changed before copying')
    with (directory / 'qa.txt').open('xb') as stream:
        stream.write(payload)
    images = []
    for index, image in enumerate(source['image_files']):
        origin = Path(image['path'])
        if origin.is_symlink() or origin.resolve() != original / f'{index:03d}.png':
            raise ValueError('Unexpected source image path or symlink')
        target = directory / origin.name
        os.link(origin, target)
        images.append(dict(image, path=str(target.resolve()), copied_from=str(origin.resolve())))
    record = dict(source, split=split, path=str(directory.resolve()), source_path=str(original),
                  origin='copied_v2_immutable', original_origin=source['origin'], image_files=images,
                  source_split=source['split'], source_sid=source['sid'])
    record['question'] = f'How many frames show {source["target_character"]} in the {source["target_room"]}?'
    if split != source['split']:
        record['profile_split_reuse'] = 'V2 software test_N8 copied to V4 software dev_N8; no main evaluation overlap'
        record.update(pair_id=None, anchor_id=None, test_family=None)
    return record


def group(rows):
    return dict(samples=rows, count=len(rows),
                gold_histogram=dict(sorted(Counter(row['gold'] for row in rows).items())))


def audit_records(manifests, dataset_root, source_lookup, excluded, fresh_sids):
    """Reparse published QA and independently count categories; hash every inode."""
    from PIL import Image
    unique, semantics, image_hashes, checked_cells = {}, {}, {}, {}
    content_to_sid = {}
    image_references, repeated_profile_training = 0, []
    for purpose, manifest in manifests.items():
        for cell, selection in manifest['splits'].items():
            observed = []
            for record in selection['samples']:
                sid, directory = record['sid'], Path(record['path']).resolve()
                expected_parent = dataset_root / 'mmred_vfiltered' / f'seq_len_{record["n_frames"]}' / record['split']
                if directory.parent != expected_parent or directory.name != sid:
                    raise ValueError('Staged record path differs from split/N/SID')
                if sid in unique:
                    if purpose != 'profile' or record['split'] != 'train' or unique[sid] != record:
                        raise ValueError('Only exact profile-training reuse of main records is permitted')
                    repeated_profile_training.append(sid)
                    continue
                parsed = read_candidate(directory, record['split'], record['n_frames'])
                qa_lines = (directory / 'qa.txt').read_text().splitlines()
                qa_block = qa_lines[qa_lines.index('question:') + 1:qa_lines.index('answer:')]
                actual_question = [line.strip() for line in qa_block if line.strip() and not line.strip().startswith('{')]
                if actual_question != [record['question']]:
                    raise ValueError('Actual QA question differs from reconstructed manifest question')
                for field in ('sid', 'gold', 'n_frames', 'split', 'qa_sha256', 'content_sha256'):
                    if parsed[field] != record[field]:
                        raise ValueError(f'Published QA metadata differs: {sid}/{field}')
                checked = inspect(dict(record, source_path=str(directory)), purpose, cell)
                if (checked['target_character'], checked['target_room']) != (record['target_character'], record['target_room']):
                    raise ValueError('Question targets differ')
                for key in ('matches', 'same_character_wrong_room', 'queried_room_other_character', 'neither'):
                    if checked[key] != record['semantic_counts'][key]:
                        raise ValueError('Independent QA category recount differs')
                states = source_states(directory)
                frames = [(next(iter(state['rooms'].values()))[0], next(iter(state['rooms']))) for state in states]
                semantics[sid] = frames
                sha = parsed['content_sha256']
                if sha in content_to_sid:
                    raise ValueError(f'Duplicate content across distinct staged SIDs: {sid}/{content_to_sid[sha]}')
                content_to_sid[sha] = sid
                if sid in fresh_sids:
                    if sha in excluded or record['source_path'] != record['path']:
                        raise ValueError('Fresh sequence collides with prior content or has nonlocal provenance')
                else:
                    source = source_lookup[sid]
                    if (directory / 'qa.txt').read_bytes() != (Path(source['path']) / 'qa.txt').read_bytes():
                        raise ValueError('Copied QA differs byte-for-byte')
                    if record['source_path'] != source['path'] or record['content_sha256'] != source['content_sha256']:
                        raise ValueError('Copied source identity differs')
                if len(record['image_files']) != record['n_frames']:
                    raise ValueError('Wrong image count')
                for index, image in enumerate(record['image_files']):
                    path = Path(image['path'])
                    if path.resolve() != directory / f'{index:03d}.png' or path.is_symlink():
                        raise ValueError('Unexpected staged image path')
                    stat = path.stat()
                    identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
                    if identity not in image_hashes:
                        image_hashes[identity] = digest(path)
                        with Image.open(path) as picture:
                            if list(picture.size) != image['dimensions'] or picture.mode != image['mode']:
                                raise ValueError('Image dimensions/mode differ')
                            picture.verify()
                    if stat.st_size != image['bytes'] or image_hashes[identity] != image['sha256']:
                        raise ValueError('Image SHA256/bytes differ')
                    if sid not in fresh_sids:
                        old = source_lookup[sid]['image_files'][index]
                        if image['sha256'] != old['sha256'] or not path.samefile(old['path']):
                            raise ValueError('Copied image is not byte-identical to original source')
                    else:
                        character, room = frames[index]
                        if Path(image['render_cache']).name != f'{character}_{room}_{index + 1:03d}.png' or not path.samefile(image['render_cache']):
                            raise ValueError('Generated image does not match its semantic render-cache key')
                    image_references += 1
                unique[sid] = record
                observed.append(checked)
            if observed:
                checked_cells[f'{purpose}/{cell}'] = summarize(observed)
    if len(unique) != 2078 or len(fresh_sids) != 1360 or len(repeated_profile_training) != 8:
        raise ValueError('Unique/copy/profile-overlap counts differ from registered staging')
    if len(set(repeated_profile_training)) != 8:
        raise ValueError('Profile training reuse is not eight distinct main scenes')
    return unique, semantics, dict(unique_samples=len(unique), fresh_samples=len(fresh_sids),
        copied_samples=len(unique) - len(fresh_sids), image_references=image_references,
        unique_image_inodes_hashed=len(image_hashes), unique_content=len(content_to_sid),
        all_qa_hashes_and_gold_recounts_passed=True, all_image_hashes_and_dimensions_passed=True,
        all_fresh_content_exclusions_passed=True, all_original_copy_bytes_preserved=True,
        intentional_profile_train_sids=repeated_profile_training, cells=checked_cells)


def audit_schedule(conditions, slots, records, profile=False):
    blocks, width = (2, 4) if profile else (9, 180)
    if set(conditions) != {'repeat', 'refresh'} or len(slots) != width:
        raise ValueError('Invalid schedule schema')
    if [slot['slot'] for slot in slots] != list(range(width)):
        raise ValueError('Slot indices are not contiguous')
    summaries = {}
    for condition, schedule in conditions.items():
        if len(schedule) != blocks:
            raise ValueError('Incorrect number of blocks')
        used = []
        for block, sids in enumerate(schedule):
            if len(sids) != width or len(set(sids)) != width:
                raise ValueError('Block size/uniqueness differs')
            for slot, sid in zip(slots, sids):
                record = records[sid]
                for key in ('n_frames', 'gold', 'question', 'target_character', 'target_room'):
                    if record[key] != slot[key]:
                        raise ValueError('Training block changed fixed slot semantics')
                should_repeat = condition == 'repeat' or block == 0 or slot['saturated']
                if (sid == slot['original_sid']) != should_repeat:
                    raise ValueError('Unexpected repeat/refresh assignment')
                if not should_repeat and (record.get('original_sid') != slot['original_sid']
                        or record.get('slot') != slot['original_slot'] or record.get('block') != block):
                    raise ValueError('Refreshed scene provenance differs from its assigned block/slot')
                if record['split'] != 'train':
                    raise ValueError('Training schedule uses a nontraining sample')
            used.extend(sids)
        summaries[condition] = dict(blocks=blocks, presentations=len(used), distinct_sids=len(set(used)),
                                   frames=sum(records[sid]['n_frames'] for sid in used))
    if conditions['repeat'][0] != conditions['refresh'][0]:
        raise ValueError('Conditions do not share block0')
    expected = {'repeat': 4, 'refresh': 8} if profile else {'repeat': 180, 'refresh': 1540}
    for condition in expected:
        if summaries[condition]['distinct_sids'] != expected[condition]:
            raise ValueError('Unexpected number of distinct scenes')
        if not profile and (summaries[condition]['frames'] != 19440 or summaries[condition]['presentations'] != 1620):
            raise ValueError('Main presentation/frame budget differs')
    return summaries


def audit_extensions(manifests, records, semantics):
    families = {}
    for purpose in ('main', 'count', 'profile', 'profile_count'):
        for cell, group_value in manifests[purpose]['splits'].items():
            if not cell.startswith('test_'):
                continue
            for row in group_value['samples']:
                if row.get('pair_id'):
                    families.setdefault((purpose, row['pair_id']), {})[row['n_frames']] = row['sid']
    checks = 0
    for (_purpose, _pair), lengths in families.items():
        anchor = semantics[lengths[min(lengths)]]
        for n, sid in lengths.items():
            row, frames = records[sid], semantics[sid]
            if [frames[index] for index in row['anchor_positions']] != anchor:
                raise ValueError('Copied anchor positions differ')
            if row['parent_n_frames'] is not None:
                parent = semantics[lengths[row['parent_n_frames']]]
                positions = row['parent_positions']
                if [frames[index] for index in positions] != parent:
                    raise ValueError('Copied extension mapping differs')
                target = row['target_character'], row['target_room']
                if any(frame == target for index, frame in enumerate(frames) if index not in set(positions)):
                    raise ValueError('Copied extension adds matching events')
                checks += 1
    return dict(paired_families=len(families), extension_checks=checks, all_copied_pair_mappings_preserved=True)


def audit_processed_tokens(training, slots, output_path):
    """Actual ordinary processor on every distinct training scene; no model load."""
    import torch
    from PIL import Image
    from transformers import AutoProcessor, __version__ as transformers_version
    from gnnformer.data import build_count_prompt, build_prompt_inputs
    from scripts.probe_native_vision_v2_prefix import fingerprint, sha_object
    torch.set_num_threads(min(4, int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))))
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True, use_fast=False, local_files_only=True)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    cache, tokens = {}, {}
    started = time.monotonic()
    try:
        for count, row in enumerate(training):
            frames = []
            for image in row['image_files']:
                key = image['sha256']
                if key not in cache:
                    with Image.open(image['path']) as source:
                        rgb = source.convert('RGB')
                        try:
                            cache[key] = rgb.resize((392, 392))
                        finally:
                            rgb.close()
                frames.append(cache[key])
            inputs = build_prompt_inputs(processor, frames, build_count_prompt(row['question'], row['n_frames']))
            ids = inputs['input_ids'][0].tolist()
            grid = inputs['image_grid_thw'].tolist()
            if len(grid) != row['n_frames'] or any(item != [1, 28, 28] for item in grid) or len(ids) + 4 > 16000:
                raise ValueError('Actual processed training scene layout differs')
            tokens[row['sid']] = dict(input_ids_sha256=sha_object(ids), prompt_tokens=len(ids), image_grid_thw=grid)
            del inputs
            if (count + 1) % 180 == 0:
                print(f'Actual processor audit {count + 1}/{len(training)} scenes, {time.monotonic() - started:.1f}s', flush=True)
    finally:
        for frame in cache.values():
            frame.close()
    slot_index = {slot['original_sid']: slot for slot in slots}
    for row in training:
        original = row.get('original_sid', row['sid'])
        if original not in slot_index or tokens[row['sid']] != tokens[original]:
            raise ValueError('Actual prompt token IDs differ across matched training slots')
    result = dict(schema_version=1, model=MODEL, resize=392, source='Actual canonical processor on every distinct training scene',
                  processor_fingerprint=fingerprint(processor, transformers_version), torch_version=str(torch.__version__),
                  records=tokens, original_slot_sids=[slot['original_sid'] for slot in slots],
                  distinct_training_scenes=len(tokens), all_matched_slot_input_ids_and_grids_equal=True,
                  elapsed_seconds=time.monotonic() - started)
    exclusive_json(output_path, result)
    return dict(path=str(output_path.resolve()), sha256=digest(output_path), distinct_training_scenes=len(tokens),
                all_matched_slot_input_ids_and_grids_equal=True, elapsed_seconds=result['elapsed_seconds'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=DESTINATION)
    parser.add_argument('--output', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--seed', type=int, default=20260914)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('All V4 staging, rendering, tokenization, and auditing require Slurm CPU')
    if args.dataset_root.resolve() != DESTINATION or args.seed != 20260914:
        raise ValueError('V4 destination/seed must match the registered protocol')
    if not 1 <= args.workers <= int(os.environ.get('SLURM_CPUS_PER_TASK', '1')):
        raise ValueError('Workers exceed the Slurm allocation')
    if args.dataset_root.exists() or args.dataset_root.is_symlink():
        raise ValueError('V4 destination already exists; preserve any existing or partial data')
    started = time.monotonic()
    sources, source_hashes, excluded = load_sources()
    source_main = sources[str(V2_ROOT / 'main_manifest.json')]
    source_count = sources[str(V2_ROOT / 'count_manifest.json')]
    source_profile = sources[str(V2_ROOT / 'profile_manifest.json')]
    source_profile_count = sources[str(V2_ROOT / 'profile_count_manifest.json')]
    originals, slots = original_slots(source_main)
    conditions, fresh = make_schedules(slots, excluded, args.seed)
    args.dataset_root.mkdir(parents=True, exist_ok=False)
    renderer, renderer_provenance = prepare_renderer(sources[str(V1_ROOT / 'main_manifest.json')]['splits']['train_N8']['samples'])
    renderer_provenance = dict(renderer_provenance, method='Canonical renderer, verified original V1 RGB parity; fresh fixed-slot V2-law draws')
    cache_root = args.dataset_root / 'render_cache'; cache_root.mkdir()
    specs = sorted({(c, r, index + 1) for row in fresh for index, (c, r) in enumerate(row['frames'])})
    print(json.dumps(dict(new_scenes=len(fresh), fresh_training_images=sum(r['n_frames'] for r in fresh),
                          unique_fresh_renders=len(specs), excluded_prior_content=len(excluded))), flush=True)
    staged, source_lookup = {}, {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        cache = dict(pool.map(lambda spec: render_cached(spec, cache_root, renderer), specs))
        generated = list(pool.map(lambda row: publish_sample(row, args.dataset_root, cache), fresh))
        for row, original in zip(generated, fresh):
            row.update(question=original['question'], origin='generated_v4_scene_refresh')
            staged[row['sid']] = row
        for original_manifest in (source_main, source_count):
            copied_sources = rows_of(original_manifest)
            copied = list(pool.map(lambda row: copy_sample(row, args.dataset_root), copied_sources))
            for source, row in zip(copied_sources, copied):
                staged[row['sid']], source_lookup[row['sid']] = row, source
        profile_sources = []
        for source_key, destination_split in (('test_N8', 'dev'), ('dev_N16', 'dev'),
                                             ('test_N16', 'test'), ('test_N32', 'test'), ('test_N64', 'test')):
            profile_sources += [(row, destination_split) for row in source_profile['splits'][source_key]['samples']]
        profile_sources += [(row, 'test') for row in rows_of(source_profile_count)]
        for source, split in profile_sources:
            row = copy_sample(source, args.dataset_root, split)
            staged[row['sid']], source_lookup[row['sid']] = row, source
    main_cells = {}
    for key, cell in source_main['splits'].items():
        rows = [staged[row['sid']] for row in cell['samples']]
        if key.startswith('train_'):
            n = int(key.split('_N')[1])
            rows += [staged[row['sid']] for row in fresh if row['n_frames'] == n]
        main_cells[key] = group(rows)
    count_cells = {key: group([staged[row['sid']] for row in cell['samples']]) for key, cell in source_count['splits'].items()}
    if len(main_cells['train_N8']['samples']) != 730 or len(main_cells['train_N16']['samples']) != 810:
        raise ValueError('Main manifest unique training counts differ')
    selected_slots = [slot['slot'] for n in (8, 16) for slot in [s for s in slots if s['n_frames'] == n and not s['saturated']][:2]]
    profile_slots = [dict(slots[original], slot=index, original_slot=original) for index, original in enumerate(selected_slots)]
    profile_conditions = {condition: [[block[index] for index in selected_slots] for block in blocks[:2]]
                          for condition, blocks in conditions.items()}
    profile_training = list(dict.fromkeys(sid for block in profile_conditions['refresh'] for sid in block))
    profile_cells = {f'train_N{n}': group([staged[sid] for sid in profile_training if staged[sid]['n_frames'] == n]) for n in (8, 16)}
    profile_cells['dev_N8'] = group([staged[row['sid']] for row in source_profile['splits']['test_N8']['samples']])
    profile_cells['dev_N16'] = group([staged[row['sid']] for row in source_profile['splits']['dev_N16']['samples']])
    for n in (16, 32, 64):
        profile_cells[f'test_N{n}'] = group([staged[row['sid']] for row in source_profile['splits'][f'test_N{n}']['samples']])
    profile_count_cells = {key: group([staged[row['sid']] for row in cell['samples']]) for key, cell in source_profile_count['splits'].items()}
    code_hashes = {name: digest(REPO / name) for name in SOURCE_FILES}
    manifests = {}
    for purpose, cells in (('main', main_cells), ('count', count_cells), ('profile', profile_cells), ('profile_count', profile_count_cells)):
        manifests[purpose] = dict(schema_version=1, purpose=purpose, dataset_root=str(args.dataset_root),
                                 data_seed=args.seed, splits=cells, generator_law=V4_LAW,
                                 generator_code_sha256=code_hashes, source_manifest_sha256=source_hashes,
                                 renderer_provenance=renderer_provenance, audit_file=str(args.dataset_root / 'audit.json'),
                                 profile_training_reuses_main_subset=purpose == 'profile',
                                 main_evaluation_is_reused=True)
    records, semantics, audit = audit_records(manifests, args.dataset_root, source_lookup, excluded, {row['sid'] for row in fresh})
    schedule_audit = audit_schedule(conditions, slots, records)
    profile_schedule_audit = audit_schedule(profile_conditions, profile_slots, records, profile=True)
    extension_audit = audit_extensions(manifests, records, semantics)
    training = [row for n in (8, 16) for row in main_cells[f'train_N{n}']['samples']]
    token_audit = audit_processed_tokens(training, slots, args.dataset_root / 'prompt_token_audit.json')
    if any(digest(Path(path)) != sha for path, sha in source_hashes.items()):
        raise ValueError('Existing source manifest changed during staging')
    if any(digest(REPO / name) != sha for name, sha in code_hashes.items()):
        raise ValueError('Generator/audit source changed during staging')
    for purpose, manifest in manifests.items():
        exclusive_json(args.dataset_root / f'{purpose}_manifest.json', manifest)
    schedules = {}
    for purpose, assignments, slot_rows, manifest_name, count_name in (
            ('main', conditions, slots, 'main_manifest.json', 'count_manifest.json'),
            ('profile', profile_conditions, profile_slots, 'profile_manifest.json', 'profile_count_manifest.json')):
        schedule = dict(schema_version=1, purpose=purpose, dataset_root=str(args.dataset_root),
                        conditions=assignments, slot_metadata=slot_rows, data_seed=args.seed,
                        manifest=str(args.dataset_root / manifest_name), manifest_sha256=digest(args.dataset_root / manifest_name),
                        count_manifest=str(args.dataset_root / count_name), count_manifest_sha256=digest(args.dataset_root / count_name),
                        source_manifest_sha256=source_hashes, generator_code_sha256=code_hashes,
                        prompt_token_audit=token_audit, shuffle_rule='Random(training_seed + block_index).shuffle(slot_indices); same across conditions')
        name = 'schedule.json' if purpose == 'main' else 'profile_schedule.json'
        exclusive_json(args.dataset_root / name, schedule)
        schedules[purpose] = dict(path=str(args.dataset_root / name), sha256=digest(args.dataset_root / name))
    final = dict(schema_version=1, slurm_job_id=os.environ['SLURM_JOB_ID'], **audit,
                 schedules=schedule_audit, profile_schedules=profile_schedule_audit, **extension_audit,
                 prompt_token_audit=token_audit, renderer_provenance=renderer_provenance,
                 generator_code_sha256=code_hashes, source_manifest_sha256=source_hashes,
                 manifest_sha256={purpose: digest(args.dataset_root / f'{purpose}_manifest.json') for purpose in manifests},
                 schedule_sha256=schedules, generator_law=V4_LAW,
                 profile_dev_N8_exception='Original V2 software test_N8 copied into new V4 dev directories; no main train/dev/test overlap',
                 prior_source_files_never_modified=True, elapsed_seconds=time.monotonic() - started)
    exclusive_json(args.dataset_root / 'audit.json', final)
    output = args.output / f'stage_{os.environ["SLURM_JOB_ID"]}'
    output.mkdir(parents=True, exist_ok=False)
    (output / 'code').mkdir()
    for name, sha in code_hashes.items():
        payload = (REPO / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != sha:
            raise ValueError('Source changed before output snapshot')
        (output / 'code' / name.replace('/', '_')).write_bytes(payload)
    exclusive_json(output / 'audit.json', final)
    cost = dict(elapsed_seconds=time.monotonic() - started, unique_samples=len(records),
                fresh_render_bytes=sum(image['bytes'] for image in cache.values()),
                qa_bytes=sum((Path(row['path']) / 'qa.txt').stat().st_size for row in records.values()),
                logical_png_bytes=sum(image['bytes'] for row in records.values() for image in row['image_files']),
                existing_images='Hardlinks only; existing files never overwritten',
                schedules=schedules, dataset_root=str(args.dataset_root), audit_sha256=digest(args.dataset_root / 'audit.json'))
    exclusive_json(output / 'cost.json', cost)
    (output / 'REPORT.md').write_text('# V4 diversity staging\n\n'
        'Nine matched blocks: repeat180 distinct scenes; refresh1540, including1360 fresh draws. '
        'Both conditions have1620 presentations and19440 frames. The ten saturated N8/K8 slots repeat.\n\n'
        f'All2078 unique samples, {audit["image_references"]} image references, QA/gold/content/copy checks and paired extensions passed. '
        'Actual processor token IDs match across every fixed training slot.\n\n'
        'Software training reuses eight declared main scenes. Software dev_N8 reuses the two original V2 software test_N8 scenes in new paths; original data are untouched.\n\n'
        '[Audit](audit.json) · [Cost and schedule paths](cost.json)\n')
    index = args.output / 'INDEX.md'
    if not index.exists():
        index.write_text('# V4 data staging\n\n')
    with index.open('a') as stream:
        stream.write(f'- Slurm {os.environ["SLURM_JOB_ID"]}: [report]({output.name}/REPORT.md), [audit]({output.name}/audit.json), [cost]({output.name}/cost.json).\n')
    print(json.dumps(cost, indent=2), flush=True)


if __name__ == '__main__':
    main()

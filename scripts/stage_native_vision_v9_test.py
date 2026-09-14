"""Stage fresh V9 evaluation only; run generation/rendering/audits on Slurm CPU.

V7 balanced training, development and schedules remain byte-for-byte unchanged.
Main:108 N16 anchors -> N32 -> N64. Count:64 N32 anchors -> N64.
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
sys.path.insert(0, str(REPO))
from scripts.stage_native_vision_v2_clean import (
    DATA_BASE, V1_ROOT, LAW, describe, digest, exclusive_json, make_family,
    prepare_renderer, publish_sample, render_cached, stable_seed,
)
from scripts.stage_native_vision_pilot import read_candidate, source_states
from scripts.audit_native_vision_data import inspect, summarize

DESTINATION = DATA_BASE / 'v9_fresh'
OUTPUT_ROOT = REPO / 'outputs/native_aggregation_vlm/v9/data_staging'
SEED = 20261002
SOURCE_FILES = ('scripts/stage_native_vision_v9_test.py',
    'slurm/native_aggregation_vision_v9_test_stage.sbatch',
    'scripts/stage_native_vision_v8_test.py',
    'slurm/native_aggregation_vision_v8_test_stage.sbatch',
    'scripts/stage_native_vision_v7_test.py',
    'slurm/native_aggregation_vision_v7_test_stage.sbatch',
    'scripts/stage_native_vision_v2_clean.py', 'scripts/stage_native_vision_pilot.py',
    'scripts/audit_native_vision_data.py', 'datasets/mmred/render_mmred.py')
SPEC = {'main': ((16, 32, 64), tuple(range(9)), 12, 'length'),
        'count': ((32, 64), tuple(range(9, 17)), 8, 'unseen_count')}


def ensure(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def content_rows(value):
    if isinstance(value, dict):
        if 'content_sha256' in value and 'n_frames' in value:
            yield value
        else:
            for item in value.values():
                yield from content_rows(item)
    elif isinstance(value, list):
        for item in value:
            yield from content_rows(item)


def load_inventory():
    paths = [V1_ROOT / f'{p}_manifest.json' for p in ('main', 'profile')]
    paths += [DATA_BASE / 'v2_clean' / f'{p}_manifest.json' for p in
              ('main', 'count', 'profile', 'profile_count', 'causal_probe')]
    paths += [DATA_BASE / 'v3_binding_probe/manifest.json']
    paths += [DATA_BASE / 'v4_diversity' / f'{p}_manifest.json' for p in
              ('main', 'count', 'profile', 'profile_count')]
    paths += [DATA_BASE / 'reasoning_baseline' / f'{p}_manifest.json' for p in ('main', 'profile')]
    paths += [DATA_BASE / 'v2_recoverability/harvest_440791' / f'{p}_manifest.json' for p in ('main', 'count')]
    paths += [DATA_BASE / 'v6_fresh' / f'{p}_manifest.json' for p in ('main', 'count')]
    paths += [DATA_BASE / 'v7_balanced' / 'main_manifest.json']
    v7_prior_paths = list(paths)
    paths += [DATA_BASE / 'v7_fresh' / f'{p}_manifest.json' for p in ('main', 'count')]
    ensure(len(v7_prior_paths) == 19 and len(paths) == len(set(paths)) == 21,
           'V8 ancestry must be the exact V7 prior19 plus both V7 fresh manifests')
    v8_prior_paths = list(paths)
    paths += [DATA_BASE / 'v8_fresh' / f'{p}_manifest.json' for p in ('main', 'count')]
    ensure(len(paths) == len(set(paths)) == 23,
           'V9 must exclude the exact V8 prior21 plus both V8 fresh manifests')
    entries, excluded, source_hashes, manifests = [], set(), {}, {}
    for path in paths:
        ensure(path.is_file() and not path.is_symlink(), f'Missing/noncanonical exclusion manifest: {path}')
        payload = path.read_bytes()
        data = json.loads(payload)
        rows = list(content_rows(data))
        ensure(rows, f'No content records in exclusion manifest: {path}')
        values = {row['content_sha256'] for row in rows}
        ensure(all(isinstance(x, str) and len(x) == 64 and all(c in '0123456789abcdef' for c in x)
                   for x in values), f'Invalid content hash in {path}')
        sha = hashlib.sha256(payload).hexdigest()
        entries.append(dict(path=str(path), sha256=sha, record_count=len(rows),
                            unique_content_count=len(values)))
        source_hashes[str(path)], manifests[str(path)] = sha, data
        excluded.update(values)
    v7_prior_hashes = {str(path): source_hashes[str(path)] for path in v7_prior_paths}
    for purpose in ('main', 'count'):
        ancestor = manifests[str(DATA_BASE / 'v7_fresh' / f'{purpose}_manifest.json')]
        ensure(ancestor['source_manifest_sha256'] == v7_prior_hashes,
               'V7 archived exclusion hashes differ from the V8 prior19')
    v8_prior_hashes = {str(path): source_hashes[str(path)] for path in v8_prior_paths}
    for purpose in ('main', 'count'):
        ancestor = manifests[str(DATA_BASE / 'v8_fresh' / f'{purpose}_manifest.json')]
        ensure(ancestor['source_manifest_sha256'] == v8_prior_hashes,
               'V8 archived exclusion hashes differ from the V9 prior21')
    protected = dict(source_hashes)
    for name in ('schedule.json', 'profile_schedule.json'):
        path = DATA_BASE / 'v4_diversity' / name
        ensure(path.is_file() and not path.is_symlink(), f'Missing V4 schedule: {path}')
        protected[str(path)] = digest(path)
    balanced_schedule = DATA_BASE / 'v7_balanced' / 'schedule.json'
    ensure(balanced_schedule.is_file() and not balanced_schedule.is_symlink(), 'Missing balanced schedule')
    protected[str(balanced_schedule)] = digest(balanced_schedule)
    ordered = sorted(excluded)
    inventory = dict(schema_version=1, entries=entries, excluded_content_sha256=ordered,
        excluded_content_set_sha256=canonical_sha(ordered), excluded_content_count=len(ordered),
        source_manifest_sha256=source_hashes, protected_source_sha256=protected,
        scope='All23 canonical manifests: the exact V8 prior21 plus V8-fresh main/count; V7 balanced training and development remain protected',
        reuse_aliases={'V3_main_and_V5_main': 'V2/V4 main and count contexts',
                       'frame_judgments': 'V3 binding pairs and isolated atoms from those pairs',
                       'long_Cosmos': 'reasoning_baseline main/profile contexts'},
        exclusions_are_complete_context_and_question_hashes=True,
        local_atom_repetition_allowed=True,
        caveat='The finite character-room atom vocabulary is shared. Freshness is whole-context/question freshness, not novel visual atoms; displayed Step indices change on extension.')
    return inventory, manifests, excluded


def generate_groups(excluded):
    used, groups = set(excluded), {}
    for purpose, (lengths, golds, repeats, axis) in SPEC.items():
        rows = []
        for gold in golds:
            for replicate in range(repeats):
                label = f'v9_{purpose}_K{gold}_{replicate:03d}'
                seed = stable_seed(SEED, label)
                rows.extend(make_family(random.Random(seed), lengths, gold, label, 'test', axis, used, seed))
        groups[purpose] = rows
    ensure(sum(map(len, groups.values())) == 452 and len(used) == len(excluded) + 452,
           'Wrong generated cardinality or content collision')
    return groups


def generator_checks(groups, excluded):
    families, content, image_references, extensions = {}, set(), 0, 0
    for purpose, rows in groups.items():
        lengths, golds, repeats, _ = SPEC[purpose]
        for n in lengths:
            ensure(Counter(r['gold'] for r in rows if r['n_frames'] == n) == Counter({k: repeats for k in golds}),
                   'Generated N/K balance differs')
        for row in rows:
            _, question, sha, counts = describe(row['frames'], row['target_character'], row['target_room'])
            ensure(question == row['question'] and sha == row['content_sha256'] and counts == row['semantic_counts'],
                   'Generator semantic hash/count mismatch')
            ensure(counts['matches'] == row['gold'] and sha not in excluded and sha not in content,
                   'Generated label/content exclusion failure')
            content.add(sha)
            image_references += row['n_frames']
            families.setdefault((purpose, row['pair_id']), {})[row['n_frames']] = row
    for (purpose, pair), members in families.items():
        lengths = SPEC[purpose][0]
        ensure(set(members) == set(lengths), 'Incomplete generated family')
        anchor = members[lengths[0]]
        for i, n in enumerate(lengths):
            row = members[n]
            ensure(row['anchor_id'] == pair and row['anchor_n_frames'] == lengths[0], 'Anchor metadata mismatch')
            ensure(row['gold'] == anchor['gold'] and row['question'] == anchor['question'], 'Family target changed')
            ensure([row['frames'][p] for p in row['anchor_positions']] == anchor['frames'], 'Anchor mapping failure')
            if i == 0:
                ensure(row['parent_n_frames'] is None and row['parent_positions'] == [], 'Invalid anchor parent')
                continue
            parent = members[lengths[i - 1]]
            positions = row['parent_positions']
            ensure(row['parent_n_frames'] == parent['n_frames'] and positions == sorted(set(positions))
                   and [row['frames'][p] for p in positions] == parent['frames'], 'Parent mapping failure')
            inserted = [f for j, f in enumerate(row['frames']) if j not in set(positions)]
            ensure(describe(inserted, row['target_character'], row['target_room'])[3]['matches'] == 0,
                   'Extension added target evidence')
            extensions += 1
    ensure(len(content) == 452 and len(families) == 172 and image_references == 18240 and extensions == 280,
           'Generated global cardinalities differ')
    a = make_family(random.Random(17), (16, 32, 64), 3, 'software', 'test', 'software', set(), 17)
    b = make_family(random.Random(17), (16, 32, 64), 3, 'software', 'test', 'software', set(), 17)
    ensure(a == b, 'Canonical generator is not deterministic')
    occupied = {r['content_sha256'] for r in a}
    c = make_family(random.Random(17), (16, 32, 64), 3, 'software', 'test', 'software', set(occupied), 17)
    ensure(all(r['generation_attempt'] > 0 and r['content_sha256'] not in occupied for r in c),
           'Canonical collision rejection failed')
    return dict(unique_samples=452, unique_content=452, families=172, image_references=18240,
                paired_extensions=280, deterministic_generation_passed=True, collision_rejection_passed=True,
                all_generated_semantics_and_extensions_passed=True)


def audit_published(manifests, excluded):
    from PIL import Image
    unique, hashes, semantics, cells = {}, {}, {}, {}
    observed, families, image_references = [], {}, 0
    for purpose, manifest in manifests.items():
        lengths, golds, repeats, _ = SPEC[purpose]
        ensure(set(manifest['splits']) == {f'test_N{n}' for n in lengths}, 'Unexpected test cells')
        for cell, group in manifest['splits'].items():
            rows, checked_rows = group['samples'], []
            ensure(group['count'] == len(rows), 'Published cell count differs')
            ensure(Counter(r['gold'] for r in rows) == Counter({k: repeats for k in golds}), 'Published gold balance differs')
            for row in rows:
                ensure(row['n_frames'] == int(cell.removeprefix('test_N')), 'Row is in the wrong length cell')
                directory = Path(row['path'])
                ensure(directory.parent == DESTINATION / 'mmred_vfiltered' / f'seq_len_{row["n_frames"]}' / 'test'
                       and directory.name == row['sid'] and not directory.is_symlink(), 'Unexpected sample path')
                ensure(row['split'] == 'test' and row['sid'] not in unique, 'Duplicate SID/non-test sample')
                parsed = read_candidate(directory, 'test', row['n_frames'])
                for key in ('sid', 'split', 'n_frames', 'gold', 'qa_sha256', 'content_sha256'):
                    ensure(parsed[key] == row[key], f'Published QA mismatch: {key}')
                checked = inspect(row, purpose, cell)
                ensure((checked['target_character'], checked['target_room']) ==
                       (row['target_character'], row['target_room']), 'Question target mismatch')
                frames = [(next(iter(s['rooms'].values()))[0], next(iter(s['rooms']))) for s in source_states(directory)]
                _, question, content_sha, counts = describe(frames, row['target_character'], row['target_room'])
                ensure(question == row['question'] and content_sha == row['content_sha256']
                       and counts == row['semantic_counts'], 'Published semantic/question mismatch')
                ensure(content_sha not in excluded and content_sha not in semantics, 'Prior/new context collision')
                for metric, value in counts.items():
                    ensure(checked[metric] == value, 'Independent semantic recount differs')
                ensure(len(row['image_files']) == len(frames) == row['n_frames'], 'Frame cardinality mismatch')
                for position, meta in enumerate(row['image_files']):
                    path = Path(meta['path'])
                    ensure(path == directory / f'{position:03d}.png' and not path.is_symlink(), 'Image path mismatch')
                    stat = path.stat()
                    identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
                    if identity not in hashes:
                        hashes[identity] = digest(path)
                        with Image.open(path) as picture:
                            ensure(picture.mode == meta['mode'] == 'RGB' and list(picture.size) == meta['dimensions'],
                                   'Image RGB/dimensions mismatch')
                            picture.verify()
                    ensure(stat.st_size == meta['bytes'] and hashes[identity] == meta['sha256'], 'Image checksum mismatch')
                    character, room = frames[position]
                    cache_path = Path(meta['render_cache'])
                    ensure(cache_path.parent == DESTINATION / 'render_cache'
                           and cache_path.name == f'{character}_{room}_{position + 1:03d}.png'
                           and path.samefile(cache_path), 'Semantic renderer-cache link mismatch')
                    image_references += 1
                unique[row['sid']], semantics[content_sha] = row, frames
                families.setdefault((purpose, row['pair_id']), {})[row['n_frames']] = row
                checked_rows.append(checked)
                observed.append(checked)
            cells[f'{purpose}/{cell}'] = summarize(checked_rows)
    reconstructed = {purpose: [] for purpose in SPEC}
    for (purpose, _), members in families.items():
        for row in members.values():
            reconstructed[purpose].append(dict(row, frames=semantics[row['content_sha256']]))
    generated = generator_checks(reconstructed, excluded)
    ensure(len(unique) == 452 and image_references == 18240, 'Published cardinalities differ')
    return dict(**generated, cells=cells, records=observed, unique_image_inodes_hashed=len(hashes),
        excluded_prior_content=len(excluded), all_qa_hashes_and_gold_recounts_passed=True,
        all_image_hashes_and_dimensions_passed=True, all_image_semantic_links_passed=True,
        all_fresh_content_exclusions_passed=True, all_pair_extensions_passed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=DESTINATION)
    parser.add_argument('--output', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--dry-check', action='store_true', help='Generate/audit semantics and freeze plan; do not render or publish data')
    args = parser.parse_args()
    ensure(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION') == 'cpu',
           'All generation/rendering/auditing requires Slurm cpu allocation')
    ensure(not os.environ.get('SLURM_JOB_GPUS', ''), 'CPU staging must not allocate a GPU')
    ensure(args.dataset_root.resolve() == DESTINATION and args.seed == SEED, 'Destination/seed differs from protocol')
    ensure(1 <= args.workers <= int(os.environ.get('SLURM_CPUS_PER_TASK', '1')), 'Workers exceed CPU allocation')
    ensure(not DESTINATION.exists() and not DESTINATION.is_symlink(), 'Preserve existing or partial destination')
    started = time.monotonic()
    inventory, prior, excluded = load_inventory()
    code_hashes = {name: digest(REPO / name) for name in SOURCE_FILES}
    job = os.environ['SLURM_JOB_ID']
    output = args.output / f'{"drycheck" if args.dry_check else "stage"}_{job}'
    output.mkdir(parents=True, exist_ok=False)
    (output / 'code').mkdir()
    for name, sha in code_hashes.items():
        raw = (REPO / name).read_bytes()
        ensure(hashlib.sha256(raw).hexdigest() == sha, 'Source changed before snapshot')
        (output / 'code' / name.replace('/', '_')).write_bytes(raw)
    exclusive_json(output / 'exclusion_inventory.json', inventory)
    inventory_sha = digest(output / 'exclusion_inventory.json')
    plan = dict(schema_version=1, purpose='v9_fresh_evaluation_only', slurm_job_id=job,
        dataset_root=str(DESTINATION), data_seed=SEED, fresh_test_seed=SEED,
        generator_law=LAW, generator_code_sha256=code_hashes,
        source_manifest_sha256=inventory['source_manifest_sha256'],
        protected_source_sha256=inventory['protected_source_sha256'],
        inventory_file=str(output / 'exclusion_inventory.json'), inventory_sha256=inventory_sha,
        expected=dict(unique_samples=452, unique_content=452, families=172,
                      image_references=18240, paired_extensions=280,
                      main={'test_N16': 108, 'test_N32': 108, 'test_N64': 108},
                      count={'test_N32': 64, 'test_N64': 64}),
        training_dev_and_schedule='Literal V7 balanced train/dev/schedule files remain unchanged; runner uses separate fresh-test overrides',
        sampling='Unchanged V2 conditional generator, conditioned on rejecting prior/new whole-family content collisions',
        extensions='Semantic subsequences preserved with negative-only insertions; Step labels are rerendered',
        local_atom_repetition_allowed=True,
        source_inventory_gap='No additional canonical V3/V5/frame-judgment/long-reasoning context manifests were found; their reused sources are listed as aliases in the inventory')
    exclusive_json(output / 'stage_plan.json', plan)
    plan_sha = digest(output / 'stage_plan.json')
    groups = generate_groups(excluded)
    dry_checks = generator_checks(groups, excluded)
    exclusive_json(output / 'generator_checks.json', dry_checks)
    (output / 'INDEX.md').write_text('# V9 fresh-data staging artifacts\n\n[Plan](stage_plan.json) · [Generator checks](generator_checks.json) · [Exclusion inventory](exclusion_inventory.json). Completed staging also writes REPORT.md, stage_audit.json and cost.json.\n')
    print(json.dumps(dict(stage='generator_checks', **dry_checks)), flush=True)
    if args.dry_check:
        ensure(all(digest(Path(p)) == sha for p, sha in inventory['protected_source_sha256'].items()),
               'Protected source changed during dry check')
        print(json.dumps(dict(dry_check_passed=True, plan=str(output / 'stage_plan.json'),
                              plan_sha256=plan_sha, inventory_sha256=inventory_sha)), flush=True)
        return
    DESTINATION.mkdir(parents=True, exist_ok=False)
    exclusive_json(DESTINATION / 'exclusion_inventory.json', inventory)
    ensure(digest(DESTINATION / 'exclusion_inventory.json') == inventory_sha, 'Inventory publication changed bytes')
    exclusive_json(DESTINATION / 'stage_plan.json', plan)
    ensure(digest(DESTINATION / 'stage_plan.json') == plan_sha, 'Plan publication changed bytes')
    reference = prior[str(V1_ROOT / 'main_manifest.json')]['splits']['train_N8']['samples']
    renderer, renderer_provenance = prepare_renderer(reference)
    all_samples = [row for rows in groups.values() for row in rows]
    specs = sorted({(c, r, i + 1) for row in all_samples for i, (c, r) in enumerate(row['frames'])})
    cache_root = DESTINATION / 'render_cache'
    cache_root.mkdir()
    print(json.dumps(dict(stage='rendering', unique_renders=len(specs), image_references=18240)), flush=True)
    manifests = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        cache = dict(pool.map(lambda spec: render_cached(spec, cache_root, renderer), specs))
        for purpose, rows in groups.items():
            cells = {}
            for n in SPEC[purpose][0]:
                selected = [row for row in rows if row['n_frames'] == n]
                random.Random(stable_seed(SEED, f'shuffle:{purpose}:test_N{n}')).shuffle(selected)
                records = list(pool.map(lambda row: publish_sample(row, DESTINATION, cache), selected))
                for record, sample in zip(records, selected):
                    record.update(question=sample['question'], origin='generated_v9_fresh', fresh_test_seed=SEED)
                cells[f'test_N{n}'] = dict(samples=records, count=len(records),
                    gold_histogram=dict(sorted(Counter(r['gold'] for r in records).items())))
            manifests[purpose] = dict(schema_version=1, purpose=purpose, dataset_root=str(DESTINATION),
                data_seed=SEED, fresh_test_seed=SEED, splits=cells, generator_law=LAW,
                generator_code_sha256=code_hashes,
                source_manifest_sha256=inventory['source_manifest_sha256'],
                renderer_provenance=renderer_provenance,
                paired_test_unit='anchor_id; resample complete length families',
                main_count_content_disjoint=True, evaluation_status='Fresh complete context/question content, fixed before V9 model outcomes; V8 results informed the research direction',
                training_dev_and_schedule_unchanged=True,
                inventory_file=str(DESTINATION / 'exclusion_inventory.json'), inventory_sha256=inventory_sha,
                stage_plan_file=str(DESTINATION / 'stage_plan.json'), stage_plan_sha256=plan_sha,
                stage_audit_file=str(DESTINATION / 'stage_audit.json'))
    semantic = audit_published(manifests, excluded)
    ensure(all(digest(Path(p)) == sha for p, sha in inventory['protected_source_sha256'].items()),
           'Protected V7 balanced/prior source changed during staging')
    ensure(all(digest(REPO / name) == sha for name, sha in code_hashes.items()), 'Staging code changed')
    semantic.update(schema_version=1, slurm_job_id=job, dataset_root=str(DESTINATION),
        data_seed=SEED, fresh_test_seed=SEED, generator_law=LAW,
        generator_code_sha256=code_hashes, source_sha256=code_hashes,
        source_manifest_sha256=inventory['source_manifest_sha256'],
        protected_source_sha256=inventory['protected_source_sha256'],
        inventory_sha256=inventory_sha, stage_plan_sha256=plan_sha,
        renderer_provenance=renderer_provenance, prior_sources_never_modified=True,
        training_dev_and_schedule_unchanged=True, local_atom_repetition_allowed=True)
    semantic_path = DESTINATION / 'semantic_audit.json'
    exclusive_json(semantic_path, semantic)
    semantic_sha = digest(semantic_path)
    for purpose, manifest in manifests.items():
        manifest.update(audit_file=str(semantic_path), audit_sha256=semantic_sha)
        exclusive_json(DESTINATION / f'{purpose}_manifest.json', manifest)
    manifest_paths = {p: str(DESTINATION / f'{p}_manifest.json') for p in manifests}
    manifest_hashes = {p: digest(Path(path)) for p, path in manifest_paths.items()}
    audit = dict(semantic)
    audit.update(semantic_audit_file=str(semantic_path), semantic_audit_sha256=semantic_sha,
                 manifest_paths=manifest_paths, manifest_sha256=manifest_hashes,
                 elapsed_seconds=time.monotonic() - started)
    exclusive_json(DESTINATION / 'stage_audit.json', audit)
    audit_sha = digest(DESTINATION / 'stage_audit.json')
    (DESTINATION / 'INDEX.md').write_text('# V9 fresh evaluation data\n\n[Main](main_manifest.json) · [Unseen counts](count_manifest.json) · [Audit](stage_audit.json) · [Exclusions](exclusion_inventory.json). V7 balanced training/dev/schedule remain unchanged.\n')
    exclusive_json(output / 'stage_audit.json', audit)
    cost = dict(slurm_job_id=job, elapsed_seconds=time.monotonic() - started,
        dataset_root=str(DESTINATION), unique_samples=452, image_references=18240,
        unique_renders=len(specs), cached_png_bytes=sum(x['bytes'] for x in cache.values()),
        stage_plan_sha256=plan_sha, inventory_sha256=inventory_sha,
        stage_audit_file=str(DESTINATION / 'stage_audit.json'), stage_audit_sha256=audit_sha,
        manifest_paths=manifest_paths, manifest_sha256=manifest_hashes, source_sha256=code_hashes)
    exclusive_json(output / 'cost.json', cost)
    (output / 'REPORT.md').write_text('# V9 fresh evaluation staging\n\n'
        '452 unique complete contexts: 108 familiar-count anchors at N16/32/64 and '
        '64 unseen-count anchors at N32/64. All QA, image, semantic, exclusion and '
        '280 paired-extension audits passed. V7 balanced training/development/schedule unchanged. '
        'Visual atoms are shared; Step-label pixels change with inserted positions.\n\n'
        '[Audit](stage_audit.json) · [Plan](stage_plan.json) · [Inventory](exclusion_inventory.json) · [Paths/cost](cost.json)\n')
    index = args.output / 'INDEX.md'
    if not index.exists():
        index.write_text('# V9 fresh evaluation staging\n\n')
    with index.open('a') as stream:
        stream.write(f'- Slurm {job}: [report]({output.name}/REPORT.md), [audit]({output.name}/stage_audit.json), [cost]({output.name}/cost.json).\n')
    print(json.dumps(cost, indent=2), flush=True)


if __name__ == '__main__':
    main()

"""Fresh frozen-model reasoning assay data; Slurm CPU staging only.

20 families x N16/32/64 = 60 examples: main two/K0..8, profile K3 and K6.
Reuse canonical V2 conditional generation and renderer. No model or outcomes.
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

DESTINATION = DATA_BASE / 'reasoning_baseline'
OUTPUT_ROOT = REPO / 'outputs/native_aggregation_vlm/reasoning_baseline/data_staging'
SOURCE_FILES = ('scripts/stage_native_vision_reasoning_baseline.py',
                'slurm/stage_native_vision_reasoning_baseline.sbatch',
                'scripts/stage_native_vision_v2_clean.py', 'scripts/stage_native_vision_pilot.py',
                'scripts/audit_native_vision_data.py', 'datasets/mmred/render_mmred.py')
GENERATOR_LAW = dict(base_conditional_law=LAW, data_seed=20260915,
    main='18 anchors: two each at K0..8; N16 anchors extended to N32 and N64',
    profile='Two independent anchors at K3,K6; N16 extended to N32 and N64',
    exclusion='Reject any whole family colliding with prior selected V1/V2/V3binding/V4 content or another new family',
    sampling='V2 base law conditioned on whole-family content uniqueness',
    purpose='Frozen Qwen/Cosmos direct versus reasoning assay; no method efficacy claim',
    evaluation_status='New scene/question content, fixed before model outcomes; not a training split')


def ensure(condition, message):
    if not condition:
        raise ValueError(message)


def source_manifests():
    paths = [V1_ROOT / f'{purpose}_manifest.json' for purpose in ('main', 'profile')]
    paths += [DATA_BASE / 'v2_clean' / f'{purpose}_manifest.json' for purpose in
              ('main', 'count', 'profile', 'profile_count', 'causal_probe')]
    paths += [DATA_BASE / 'v3_binding_probe/manifest.json']
    paths += [DATA_BASE / 'v4_diversity' / f'{purpose}_manifest.json' for purpose in
              ('main', 'count', 'profile', 'profile_count')]
    sources, hashes, excluded = {}, {}, set()
    for path in paths:
        ensure(path.is_file() and not path.is_symlink(), f'Missing/noncanonical exclusion manifest: {path}')
        raw = path.read_bytes()
        data = json.loads(raw)
        rows = [row for cell in data.get('splits', {}).values() for row in cell['samples']]
        if not rows and 'pairs' in data:
            rows = [pair[condition] for pair in data['pairs'] for condition in ('low', 'high')]
        ensure(rows, f'No exclusion rows: {path}')
        for row in rows:
            value = row['content_sha256']
            ensure(isinstance(value, str) and len(value) == 64, 'Invalid source content hash')
            excluded.add(value)
        sources[str(path)], hashes[str(path)] = data, hashlib.sha256(raw).hexdigest()
    return sources, hashes, excluded


def audit_published(manifests, excluded):
    from PIL import Image
    observed, semantics, unique, hashes, families, cells = [], {}, {}, {}, {}, {}
    image_references = 0
    for purpose, manifest in manifests.items():
        ensure(set(manifest['splits']) == {'test_N16', 'test_N32', 'test_N64'}, 'Unexpected split cells')
        for cell, group in manifest['splits'].items():
            records = group['samples']
            wanted = Counter({k: 2 for k in range(9)}) if purpose == 'main' else Counter({3: 1, 6: 1})
            ensure(Counter(row['gold'] for row in records) == wanted, 'Incorrect count/label balance')
            checked_rows = []
            for row in records:
                directory = Path(row['path'])
                ensure(directory.parent == DESTINATION / 'mmred_vfiltered' / f'seq_len_{row["n_frames"]}' / 'test'
                       and directory.name == row['sid'] and not directory.is_symlink(), 'Unexpected sample path')
                ensure(row['split'] == 'test' and row['sid'] not in unique, 'Duplicate SID/non-test sample')
                parsed = read_candidate(directory, 'test', row['n_frames'])
                for key in ('sid', 'split', 'n_frames', 'gold', 'qa_sha256', 'content_sha256'):
                    ensure(parsed[key] == row[key], f'Published QA mismatch: {key}')
                lines = (directory / 'qa.txt').read_text().splitlines()
                section = lines[lines.index('question:') + 1:lines.index('answer:')]
                question = [line.strip() for line in section if line.strip() and not line.strip().startswith('{')]
                ensure(question == [row['question']], 'Actual QA question differs from manifest')
                sha = row['content_sha256']
                ensure(sha not in excluded and sha not in {r['content_sha256'] for r in unique.values()},
                       'Prior/new whole-content collision')
                checked = inspect(row, purpose, cell)
                ensure((checked['target_character'], checked['target_room']) ==
                       (row['target_character'], row['target_room']), 'Question target mismatch')
                for key in ('matches', 'same_character_wrong_room', 'queried_room_other_character', 'neither'):
                    ensure(checked[key] == row['semantic_counts'][key], 'Independent semantic recount mismatch')
                states = source_states(directory)
                frames = [(next(iter(state['rooms'].values()))[0], next(iter(state['rooms']))) for state in states]
                ensure(len(row['image_files']) == len(frames) == row['n_frames'], 'Incorrect frame count')
                for position, image in enumerate(row['image_files']):
                    path = Path(image['path'])
                    ensure(path == directory / f'{position:03d}.png' and not path.is_symlink(), 'Image path mismatch')
                    stat = path.stat()
                    identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
                    if identity not in hashes:
                        hashes[identity] = digest(path)
                        with Image.open(path) as picture:
                            ensure(list(picture.size) == image['dimensions'] and picture.mode == image['mode'],
                                   'Image dimensions/mode mismatch')
                            picture.verify()
                    ensure(stat.st_size == image['bytes'] and hashes[identity] == image['sha256'], 'Image checksum mismatch')
                    character, room = frames[position]
                    cache_path = Path(image['render_cache'])
                    ensure(cache_path.parent == DESTINATION / 'render_cache'
                           and cache_path.name == f'{character}_{room}_{position + 1:03d}.png'
                           and path.samefile(cache_path), 'Semantic renderer-cache link mismatch')
                    image_references += 1
                unique[row['sid']], semantics[row['sid']] = row, frames
                families.setdefault((purpose, row['pair_id']), {})[row['n_frames']] = row['sid']
                checked_rows.append(checked)
                observed.append(checked)
            cells[f'{purpose}/{cell}'] = summarize(checked_rows)
    ensure(len(unique) == 60 and len(families) == 20 and image_references == 2240, 'Assay cardinalities differ')
    ensure(Counter(purpose for purpose, _ in families) == {'main': 18, 'profile': 2}, 'Wrong family counts')
    extensions = []
    for (purpose, pair), members in families.items():
        ensure(set(members) == {16, 32, 64}, 'Incomplete length family')
        anchor = semantics[members[16]]
        original = unique[members[16]]
        for n in (16, 32, 64):
            row, frames = unique[members[n]], semantics[members[n]]
            ensure(row['anchor_id'] == pair and row['pair_id'] == pair and row['anchor_n_frames'] == 16,
                   'Anchor metadata mismatch')
            ensure(row['gold'] == original['gold'] and row['question'] == original['question'], 'Family question/count changed')
            ensure([frames[index] for index in row['anchor_positions']] == anchor, 'Anchor positions mismatch')
            expected_parent = {16: None, 32: 16, 64: 32}[n]
            ensure(row['parent_n_frames'] == expected_parent, 'Incorrect parent length')
            if n == 16:
                ensure(row['parent_positions'] == [], 'Anchor unexpectedly has parent positions')
                continue
            parent = semantics[members[expected_parent]]
            positions = row['parent_positions']
            ensure(len(positions) == expected_parent and positions == sorted(set(positions))
                   and [frames[index] for index in positions] == parent, 'Parent subsequence mismatch')
            inserted = [frame for index, frame in enumerate(frames) if index not in set(positions)]
            _, _, _, counts = describe(inserted, row['target_character'], row['target_room'])
            ensure(counts['matches'] == 0, 'Extension introduced new evidence')
            extensions.append(dict(purpose=purpose, pair_id=pair, parent=members[expected_parent],
                                   child=row['sid'], added_semantic_counts=counts))
    ensure(len(extensions) == 40, 'Incorrect extension count')
    return dict(samples=60, main_samples=54, profile_samples=6, families=20,
                image_references=image_references, unique_image_inodes_hashed=len(hashes),
                excluded_prior_content=len(excluded), cells=cells, records=observed,
                paired_extensions=extensions, all_qa_hashes_and_gold_recounts_passed=True,
                all_image_hashes_dimensions_and_semantic_links_passed=True,
                all_content_exclusions_and_main_profile_disjointness_passed=True,
                all_parent_and_anchor_mappings_passed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=DESTINATION)
    parser.add_argument('--output', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--seed', type=int, default=20260915)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    ensure(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION') == 'cpu',
           'All rendering/auditing requires Slurm cpu allocation')
    ensure(not os.environ.get('SLURM_JOB_GPUS', ''), 'CPU-only staging requires no GPU')
    ensure(args.dataset_root.resolve() == DESTINATION and args.seed == 20260915, 'Destination/seed differs from protocol')
    ensure(1 <= args.workers <= int(os.environ.get('SLURM_CPUS_PER_TASK', '1')), 'Too many workers')
    ensure(not args.dataset_root.exists() and not args.dataset_root.is_symlink(), 'Preserve existing/partial destination')
    started = time.monotonic()
    sources, source_hashes, excluded = source_manifests()
    code_hashes = {name: digest(REPO / name) for name in SOURCE_FILES}
    used, groups = set(excluded), {}
    for purpose in ('main', 'profile'):
        rows = []
        for gold in (range(9) if purpose == 'main' else (3, 6)):
            for replicate in range(2 if purpose == 'main' else 1):
                label = f'rbase_{purpose}_K{gold}_{replicate:02d}'
                seed = stable_seed(args.seed, label)
                rows.extend(make_family(random.Random(seed), (16, 32, 64), gold, label, 'test',
                            'reasoning_baseline_length' if purpose == 'main' else 'reasoning_baseline_software', used, seed))
        groups[purpose] = rows
    all_samples = groups['main'] + groups['profile']
    ensure(len(all_samples) == 60 and len(used) == len(excluded) + 60, 'Generated cardinalities differ')
    args.dataset_root.mkdir(parents=True, exist_ok=False)
    reference = sources[str(V1_ROOT / 'main_manifest.json')]['splits']['train_N8']['samples']
    renderer, renderer_provenance = prepare_renderer(reference)
    cache_root = args.dataset_root / 'render_cache'
    cache_root.mkdir()
    specs = sorted({(c, r, index + 1) for row in all_samples for index, (c, r) in enumerate(row['frames'])})
    print(json.dumps(dict(samples=60, image_references=2240, unique_renders=len(specs))), flush=True)
    manifests = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        cache = dict(pool.map(lambda spec: render_cached(spec, cache_root, renderer), specs))
        for purpose, samples in groups.items():
            records = list(pool.map(lambda row: publish_sample(row, args.dataset_root, cache), samples))
            for record, sample in zip(records, samples):
                record.update(question=sample['question'], origin='generated_reasoning_baseline')
            cells = {}
            for n in (16, 32, 64):
                selected = [row for row in records if row['n_frames'] == n]
                cells[f'test_N{n}'] = dict(samples=selected, count=len(selected),
                    gold_histogram=dict(sorted(Counter(row['gold'] for row in selected).items())))
            manifests[purpose] = dict(schema_version=1, purpose=purpose, dataset_root=str(args.dataset_root),
                data_seed=args.seed, splits=cells, generator_law=GENERATOR_LAW,
                generator_code_sha256=code_hashes, source_manifest_sha256=source_hashes,
                renderer_provenance=renderer_provenance, main_profile_content_disjoint=True,
                paired_test_unit='pair_id; resample complete three-length families')
    audit = audit_published(manifests, excluded)
    ensure(all(digest(Path(path)) == sha for path, sha in source_hashes.items()), 'Source manifests changed')
    ensure(all(digest(REPO / name) == sha for name, sha in code_hashes.items()), 'Staging sources changed')
    audit.update(schema_version=1, slurm_job_id=os.environ['SLURM_JOB_ID'], data_seed=args.seed,
                 dataset_root=str(args.dataset_root), generator_law=GENERATOR_LAW,
                 generator_code_sha256=code_hashes, source_manifest_sha256=source_hashes,
                 renderer_provenance=renderer_provenance, prior_sources_never_modified=True,
                 elapsed_seconds=time.monotonic() - started)
    audit_path = args.dataset_root / 'audit.json'
    exclusive_json(audit_path, audit)
    audit_sha = digest(audit_path)
    for purpose, manifest in manifests.items():
        manifest.update(audit_file=str(audit_path), audit_sha256=audit_sha)
        exclusive_json(args.dataset_root / f'{purpose}_manifest.json', manifest)
    cost = dict(slurm_job_id=os.environ['SLURM_JOB_ID'], elapsed_seconds=time.monotonic() - started,
                cached_png_bytes=sum(item['bytes'] for item in cache.values()), samples=60, image_references=2240,
                unique_renders=len(specs), dataset_root=str(args.dataset_root), audit_sha256=audit_sha,
                manifests={purpose: dict(path=str(args.dataset_root / f'{purpose}_manifest.json'),
                    sha256=digest(args.dataset_root / f'{purpose}_manifest.json')) for purpose in manifests})
    output = args.output / f'stage_{os.environ["SLURM_JOB_ID"]}'
    output.mkdir(parents=True, exist_ok=False)
    (output / 'code').mkdir()
    for name, sha in code_hashes.items():
        payload = (REPO / name).read_bytes()
        ensure(hashlib.sha256(payload).hexdigest() == sha, 'Source changed before snapshot')
        (output / 'code' / name.replace('/', '_')).write_bytes(payload)
    exclusive_json(output / 'audit.json', audit)
    exclusive_json(output / 'cost.json', cost)
    (output / 'REPORT.md').write_text('# Fresh reasoning-baseline data\n\n'
        'Main: 18 anchors, two per K0..8. Software: two separate anchors, K3 and K6. '
        'Each has N16/N32/N64, for 60 examples and 2,240 frame references. '
        'All QA/image/semantic/exclusion/extension checks passed; no training or development data.\n\n'
        '[Audit](audit.json) · [Canonical manifests and cost](cost.json)\n')
    index = args.output / 'INDEX.md'
    if not index.exists():
        index.write_text('# Reasoning-baseline data staging\n\n')
    with index.open('a') as stream:
        stream.write(f'- Slurm {os.environ["SLURM_JOB_ID"]}: [report]({output.name}/REPORT.md), [audit]({output.name}/audit.json), [cost]({output.name}/cost.json).\n')
    print(json.dumps(cost, indent=2), flush=True)


if __name__ == '__main__':
    main()

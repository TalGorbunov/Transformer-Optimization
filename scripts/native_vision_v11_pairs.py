"""V11 seed-aware order over the unchanged V10 pairs and new penultimate cache."""
from __future__ import annotations
from collections import Counter
import json
import os
from pathlib import Path
import random
import sys
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
from scripts.stage_native_vision_v7_features import need, read, sha, object_sha
DATA = Path('/mnt/data/gabriele/gnn_transformer')
MANIFEST = DATA/'v10_balanced/main_manifest.json'
SCHEDULE = DATA/'v10_balanced/schedule.json'
PAIRING = DATA/'v10_balanced/pairing.json'
CACHE = DATA/'v11_parallel_local/feature_cache.json'


def presentation_order(pairs, seed):
    need(seed in (16, 17) and len(pairs) == 918, 'Unregistered seed or pair count')
    rng = random.Random(seed); rows = []
    for epoch in range(1, 41):
        permutation = list(range(918)); rng.shuffle(permutation)
        for slot in permutation:
            p = pairs[slot]
            for side, sid in enumerate(p['sids']):
                rows.append(dict(epoch=epoch, slot=slot, pair_id=p['pair_id'], question=p['question'],
                    gold=p['gold'], sid=sid, pair_side=side, pair_kind=p['pair_kind'],
                    n_frames=p['n_frames'][side], replica=p['replicas'][side]))
    need(len(rows) == 73440 and all(rows[i]['pair_side'] == 0 and rows[i+1]['pair_side'] == 1
         and rows[i]['pair_id'] == rows[i+1]['pair_id'] for i in range(0, len(rows), 2)), 'Pair order differs')
    return rows


def load_pairs(manifest_file=MANIFEST, cache_file=CACHE, schedule_file=SCHEDULE):
    need(os.environ.get('SLURM_JOB_ID'), 'Artifact validation requires Slurm')
    manifest_file, cache_file, schedule_file = map(lambda p: Path(p).resolve(), (manifest_file, cache_file, schedule_file))
    need((manifest_file, cache_file, schedule_file) == (MANIFEST, CACHE, SCHEDULE), 'Require the unchanged V10 corpus and exact V11 penultimate cache')
    manifest, cache, schedule, pairing = map(read, (manifest_file, cache_file, schedule_file, PAIRING))
    need(schedule['manifest_file'] == str(manifest_file) and schedule['manifest_sha256'] == sha(manifest_file)
         and pairing['manifest_sha256'] == sha(manifest_file) and pairing['schedule_sha256'] == sha(schedule_file),
         'Pairing or schedule binding changed')
    feature_plan = read(cache['plan_file'])
    need(sha(cache['plan_file']) == cache['plan_sha256'] and cache['complete'] and cache['training_only']
         and cache['scenes'] == feature_plan['scenes'], 'Incomplete or mismatched training cache')
    train = {r['sid']: r for n in (8, 16) for r in manifest['splits'][f'train_N{n}']['samples']}
    need(len(train) == 1782 and set(train) == set(cache['scenes']), 'Unique training scene inventory differs')
    for sid, row in train.items():
        scene = cache['scenes'][sid]
        need(all(scene[k] == row[k] for k in ('question', 'gold', 'n_frames', 'qa_sha256', 'content_sha256')),
             'Cached scene metadata differs')
        targets = scene['target_ids']; size = len(targets)
        need(size == (2 if row['gold'] < 10 else 3) and targets[-1] == 151645
             and 151645 not in targets[:-1] and 151643 not in targets[:-1], 'Full numeral/EOS sequence differs')
        local_ids, gids = scene['local_feature_ids'], scene['global_feature_ids']
        need(len(gids) == size and len(local_ids) == row['n_frames']
             and all(len(ids) == size for ids in local_ids), 'Individual scene prefix segmentation differs')
        for item_index, ids in enumerate([gids] + local_ids):
            descriptors = [feature_plan['features'][fid] for fid in ids]
            base_ids = feature_plan['layouts'][descriptors[0]['layout_id']]['input_ids']
            for t, (fid, descriptor) in enumerate(zip(ids, descriptors)):
                need(fid in cache['features'] and descriptor['question'] == row['question']
                     and descriptor['prefix_ids'] == targets[:t], 'Cached strict prefix/question differs')
                need(feature_plan['layouts'][descriptor['layout_id']]['input_ids'] == base_ids + targets[:t],
                     'Earlier feature includes an extra target token or wrong template')
            need(all(x['kind'] == ('global' if item_index == 0 else 'local') for x in descriptors),
                 'Global/local feature ownership differs')
            if item_index:
                need(all(feature_plan['pairs'][x['pair_id']]['image_sha256'] == row['image_files'][item_index-1]['sha256']
                     for x in descriptors), 'Cached local image identity/order differs')
    pairs = []
    for i, pair in enumerate(pairing['pairs']):
        need(pair['slot'] == i and pair['pair_id'] == object_sha(['v10_pair', pair['question'], pair['gold']]), 'Canonical pair identity differs')
        a, b = [cache['scenes'][sid] for sid in pair['sids']]; k = pair['gold']
        expected_kind = 'cross_length' if k <= 8 else 'same_length' if k <= 15 else 'saturated_identity'
        need(pair['pair_kind'] == expected_kind and pair['n_frames'] == ([8,16] if k <= 8 else [16,16])
             and pair['same_sid'] == (k == 16) and (pair['sids'][0] == pair['sids'][1]) == (k == 16), 'Pair support differs')
        need(a['question'] == b['question'] == pair['question'] and a['gold'] == b['gold'] == k
             and a['target_ids'] == b['target_ids'] and a['global_feature_ids'] == b['global_feature_ids'],
             'Paired targets or frozen causal query identities differ')
        pairs.append(dict(pair, target_ids=a['target_ids'], global_feature_ids=a['global_feature_ids']))
    slots = [sid for p in pairs for sid in p['sids']]
    need(len(pairs) == 918 and len({p['pair_id'] for p in pairs}) == 918 and slots == schedule['epoch_slots'], 'Pair order or weight differs')
    weighted = Counter(slots)
    need(all(weighted[sid] == (2 if row['gold'] == 16 else 1) for sid,row in train.items()), 'Only saturated K16 may have duplicate slots')
    need(Counter((train[s]['question'], train[s]['gold']) for s in slots) ==
         Counter({(q,k):2 for q in {r['question'] for r in train.values()} for k in range(17)}), 'Question/count weights differ')
    need(sum(len(cache['scenes'][sid]['target_ids']) for sid in slots) == 4428, 'Valid causal targets per epoch differ')
    return dict(schema_version=1, protocol='v11_same_v10_question_count_pairing', pairs=pairs,
        pairs_sha256=object_sha(pairs), original_pairing_file=str(PAIRING), original_pairing_sha256=sha(PAIRING),
        pair_count=918, weighted_scenes_per_epoch=1836, unique_training_scenes=1782,
        target_positions_per_epoch=4428, epochs=40, seeds=[16,17])


def self_test():
    pairs = [dict(pair_id=str(i), question=f'q{i}', gold=i%17, sids=[f'a{i}',f'a{i}' if i%17==16 else f'b{i}'],
        pair_kind='saturated_identity' if i%17==16 else 'cross_length', n_frames=[8,16], replicas=[0,0]) for i in range(918)]
    a = presentation_order(pairs, 16)
    need(a == presentation_order(pairs,16) and a != presentation_order(pairs,17), 'Seed/order determinism differs')
    expected = Counter(s for p in pairs for s in p['sids'])
    need(Counter(r['sid'] for r in a) == Counter({s:40*n for s,n in expected.items()}), 'Epoch weighting differs')
    need(Counter(r['epoch'] for r in a[1824:1840]) == {1:12, 2:4}, 'Six/two-pair carried boundary differs')
    rng = random.Random(16)
    for epoch in (1,2):
        ids=list(range(918)); rng.shuffle(ids)
        need([r['slot'] for r in a[(epoch-1)*1836:epoch*1836:2]] == ids, 'RNG must persist between epochs')
    return dict(passed=True, tests=['persistent_RNG', 'different_seeds', 'complete_weighted_epochs', 'intact_pairs_and_carried_boundary'])


if __name__ == '__main__':
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION') == 'cpu', 'CPU Slurm required')
    print(json.dumps(self_test()), flush=True)

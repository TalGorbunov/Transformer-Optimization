"""Deterministic V8 paired schedules from the unchanged balanced V7 scenes.

No data/model mutation or feature harvesting. Pair N8 then N16 with identical
question/count; repeat only the deterministic N8K8 scene to preserve V7 weights.
One persistent random.Random(seed) shuffles a fresh canonical pair list each
of40 epochs. Eight intact pairs per batch; batches cross epoch boundaries.
The pure metadata helpers are separate from Slurm-guarded artifact loading.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import random

REPO=Path(__file__).resolve().parents[1]
DATA=Path('/mnt/data/gabriele/gnn_transformer')
MANIFEST=DATA/'v7_balanced/main_manifest.json'
SCHEDULE=DATA/'v7_balanced/schedule.json'
CACHE=DATA/'v7_parallel_local/feature_cache.json'
PAIR_COUNT=972
EPOCHS=40
SEEDS=(10,11)
SOURCE='scripts/native_vision_v8_pairs.py'


def need(value,message):
    if not value:raise ValueError(message)


def objsha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def build_pairs(manifest,cache,schedule,feature_plan):
    """Return972 canonical pair records after exact weighting/target checks.

    slot is the canonical PAIR index, not a previous V7 individual-scene slot.
    Both prediction positions use the same global feature across pair members.
    This checks pairing metadata; the parent cache audit verifies tensor files.
    """
    need(set(manifest['splits'])=={'train_N8','train_N16','dev_N16'},'Unexpected balanced split cells')
    cells={n:manifest['splits'][f'train_N{n}']['samples'] for n in (8,16)}
    need(len(cells[8])==918 and len(cells[16])==972,'Balanced training cardinalities differ')
    train={r['sid']:r for n in (8,16) for r in cells[n]}
    need(len(train)==1890 and set(train)==set(cache['scenes'])==set(feature_plan['scenes'])
         and cache['scenes']==feature_plan['scenes'],'Training/cache scene inventory differs')
    need(cache['complete'] is True and cache['training_only'] is True,'Require a complete training-only cache')
    tokens=feature_plan['count_token_ids'];eos=feature_plan['eos_token_id']
    need(set(tokens)=={str(k) for k in range(9)} and len(set(tokens.values()))==9
         and eos==151645 and all(isinstance(t,int) and t>=0 and t!=eos for t in tokens.values()),'Count/EOS token contract differs')
    groups=defaultdict(list)
    for n in (8,16):
        for row in cells[n]:
            sid=row['sid'];scene=cache['scenes'][sid];q=row['question'];k=row['gold']
            need(row['split']=='train' and row['n_frames']==n and isinstance(q,str) and bool(q)
                 and isinstance(k,int) and not isinstance(k,bool) and k in range(9),'Malformed training scene')
            need(all(scene[key]==row[key] for key in ('question','gold','n_frames','qa_sha256','content_sha256')),
                 'Cache scene metadata differs from its exact training sample')
            need(scene['target_ids']==[tokens[str(k)],eos],'Cached answer/EOS token differs from gold')
            gids=scene['global_feature_ids'];need(len(gids)==2 and gids[0]!=gids[1],'Need distinct empty and gold-prefix global features')
            for index,fid in enumerate(gids):
                prefix=[] if index==0 else scene['target_ids'][:1]
                need(fid in cache['features'] and fid in feature_plan['features'],'Missing global cached feature')
                feature=feature_plan['features'][fid]
                need(fid==objsha(['global',q,prefix]) and feature['kind']=='global'
                     and feature['question']==q and feature['prefix_ids']==prefix,'Global feature identity/question/prefix differs')
            left,right=[feature_plan['layouts'][feature_plan['features'][fid]['layout_id']]['input_ids'] for fid in gids]
            need(right==left+scene['target_ids'][:1],'EOS feature changes pre-answer global input')
            groups[q,k,n].append(sid)
    questions=sorted({q for q,_,_ in groups})
    need(len(questions)==54 and set(groups)=={(q,k,n) for q in questions for k in range(9) for n in (8,16)},
         'Question/count/length support differs')
    pairs=[]
    for q in questions:
        for k in range(9):
            small=sorted(groups[q,k,8]);large=sorted(groups[q,k,16])
            need(len(small)==(1 if k==8 else 2) and len(large)==2,'Expected two replicas except saturated N8K8')
            if k==8:small=small*2
            for replica,(a,b) in enumerate(zip(small,large)):
                first,second=cache['scenes'][a],cache['scenes'][b]
                need(first['target_ids']==second['target_ids'] and first['global_feature_ids']==second['global_feature_ids'],
                     'Paired native targets or global feature IDs differ')
                pairs.append(dict(slot=len(pairs),pair_id=objsha(['v8_pair',q,k,replica]),question=q,gold=k,
                    replica=replica,sids=[a,b],target_ids=list(first['target_ids']),
                    global_feature_ids=list(first['global_feature_ids'])))
    need(len(pairs)==PAIR_COUNT and len({p['pair_id'] for p in pairs})==PAIR_COUNT,'Canonical pair count/identity differs')
    slots=schedule['epoch_slots'];weighted=Counter(s for p in pairs for s in p['sids'])
    need(len(slots)==1944 and weighted==Counter(slots) and set(slots)==set(train)
         and schedule['unique_training_sids']==sorted(train),'Paired scenes change V7 weighted multiplicities')
    need(schedule['slot_metadata']==[dict(slot=i,sid=s,n_frames=train[s]['n_frames'],gold=train[s]['gold'],question=train[s]['question'])
         for i,s in enumerate(slots)],'Original V7 slot metadata differs')
    need(all(weighted[s]==(2 if r['n_frames']==8 and r['gold']==8 else 1) for s,r in train.items()),
         'Only saturated N8K8 scenes may repeat within an epoch')
    return pairs


def presentation_order(pairs,seed):
    """Flat77,760-scene list; consecutive16 records are8 complete pairs."""
    need(seed in SEEDS and len(pairs)==PAIR_COUNT,'Unregistered seed or pair count')
    need([p['slot'] for p in pairs]==list(range(PAIR_COUNT)) and len({p['pair_id'] for p in pairs})==PAIR_COUNT,
         'Pairs must be in unique canonical slot order')
    rng=random.Random(seed);rows=[]
    for epoch in range(1,EPOCHS+1):
        slots=list(range(PAIR_COUNT));rng.shuffle(slots)
        for slot in slots:
            pair=pairs[slot]
            common={key:pair[key] for key in ('slot','pair_id','question','gold','replica')}
            for side,sid in zip(('N8','N16'),pair['sids']):
                rows.append(dict(common,epoch=epoch,sid=sid,pair_side=side))
    validate_order(rows,pairs)
    return rows


def validate_order(rows,pairs):
    need(len(rows)==77760 and len(rows)%16==0,'Incorrect presentation or batch count')
    seen=defaultdict(list)
    for first,second in zip(rows[::2],rows[1::2]):
        slot=first['slot'];need(isinstance(slot,int) and slot in range(PAIR_COUNT),'Invalid canonical pair slot')
        pair=pairs[slot];epoch=first['epoch']
        need(epoch in range(1,EPOCHS+1) and second['epoch']==epoch
             and first['pair_side']=='N8' and second['pair_side']=='N16'
             and [first['sid'],second['sid']]==pair['sids'],'An adjacent pair was split, reversed or relabeled')
        need(all(first[key]==second[key]==pair[key] for key in ('slot','pair_id','question','gold','replica')),
             'Paired presentation metadata differs')
        seen[epoch].append(slot)
    need(set(seen)==set(range(1,EPOCHS+1)) and all(sorted(v)==list(range(PAIR_COUNT)) for v in seen.values()),
         'Each epoch must contain every canonical pair once')
    need([r['epoch'] for r in rows]==[e for e in range(1,EPOCHS+1) for _ in range(1944)],'Epoch boundaries/order differ')
    return dict(passed=True,pairs_per_epoch=972,scenes_per_epoch=1944,epochs=40,presentations=77760,
                pair_presentations=38880,batches=4860,scenes_per_batch=16,pairs_per_batch=8)


def iter_batches(rows):
    need(len(rows)==77760,'Use the complete fixed presentation stream')
    for offset in range(0,len(rows),16):yield rows[offset:offset+16]


def load_pairs(manifest_file=MANIFEST,cache_file=CACHE,schedule_file=SCHEDULE):
    """Load and bind immutable source artifacts; return metadata, never tensors."""
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION') in ('cpu','gpu'),
         'Artifact validation requires Slurm')
    manifest_file,cache_file,schedule_file=[Path(p).resolve() for p in (manifest_file,cache_file,schedule_file)]
    need((manifest_file,cache_file,schedule_file)==(MANIFEST,CACHE,SCHEDULE),'V8 pairing requires literal unchanged V7 artifacts')
    bindings={};source_hashes={SOURCE:sha(REPO/SOURCE)}
    def bind(label,path,digest=None):
        path=Path(path).resolve();actual=sha(path)
        need(digest is None or actual==digest,'Source artifact changed: '+str(path))
        bindings[label]=dict(file=str(path),sha256=actual)
        return read(path)
    manifest=bind('manifest',manifest_file);schedule=bind('schedule',schedule_file);cache=bind('cache',cache_file)
    need(schedule['manifest_file']==str(manifest_file) and schedule['manifest_sha256']==sha(manifest_file),'Original schedule/manifest binding differs')
    plan=bind('feature_plan',cache['plan_file'],cache['plan_sha256'])
    need(Path(plan['source_files']['training_manifest']['path']).resolve()==manifest_file
         and plan['source_files']['training_manifest']['sha256']==sha(manifest_file)
         and Path(plan['source_files']['training_schedule']['path']).resolve()==schedule_file
         and plan['source_files']['training_schedule']['sha256']==sha(schedule_file),'Feature plan used different train/schedule files')
    need(plan['protocol']=='v7_parallel_local_training_features' and cache['source_sha256']==plan['source_sha256']
         and all(cache[key]==plan[key] for key in ('model','runtime','processor')),'Cache and parent source/native identities differ')
    for label in ('audit','stage_plan','prior_inventory'):
        bound=bind('balanced_'+label,manifest[label+'_file'],manifest[label+'_sha256'])
        if label=='audit':need(bound['all_current_contexts_disjoint'] and bound['question_count_support_uniform_in_slots'],
                               'Balanced semantic/weight audit failed')
    for label,value in plan['source_files'].items():bind('feature_source_'+label,value['path'],value['sha256'])
    for mapping in (manifest['source_sha256'],cache['source_sha256']):
        for path,digest in mapping.items():
            actual=Path(path) if Path(path).is_absolute() else REPO/path
            need(sha(actual)==digest,'Frozen source code changed: '+str(path));source_hashes[path]=digest
    extension=cache['runtime_extension'];released=bind('runtime_release',extension['release_file'],extension['release_sha256'])
    need(extension==dict(released,release_file=extension['release_file'],release_sha256=extension['release_sha256'])
         and released['passed'] is True and released['original_profile_passed'] is False and released['cap_seconds']==360
         and released['plan_sha256']==cache['plan_sha256'],'Explicit V7 runtime extension changed')
    for path,digest in released['dispatcher_source_sha256'].items():
        need(sha(REPO/path)==digest,'Cache dispatcher source changed');source_hashes[path]=digest
    profile=bind('feature_profile',Path(cache['profile_directory'])/'summary.json',cache['profile_summary_sha256'])
    need(profile['passed'] is False and profile['computational_integrity_passed'] and profile['numerical_gate_passed'],
         'Original runtime-only profile failure was lost or computational checks failed')
    pairs=build_pairs(manifest,cache,schedule,plan)
    return dict(schema_version=1,protocol='v8_equal_question_count_pairing',pairs=pairs,pairs_sha256=objsha(pairs),
        bindings=bindings,source_sha256=source_hashes,count_token_ids=plan['count_token_ids'],eos_token_id=plan['eos_token_id'],
        pair_count=972,weighted_scenes_per_epoch=1944,unique_training_scenes=1890,
        seeds=list(SEEDS),epochs=40,batch_size=16,pairs_per_batch=8,optimizer_updates=4860,
        shuffle='One persistent random.Random(seed); fresh canonical pair slots shuffled each epoch',
        fixed_within_pair_order=['N8','N16'],epoch_boundary_batches_carried=True,
        both_count_and_EOS_targets_and_global_features_identical=True,
        scope='Pairing/provenance metadata only; native tensor files remain subject to the unchanged cache and training audits')


def self_test():
    """Full-sized metadata fixtures test weights, alignment and adversarial cases."""
    manifest=dict(splits={key:dict(samples=[]) for key in ('train_N8','train_N16','dev_N16')})
    scenes={};features={};layouts={};slots=[];tokens={str(k):15+k for k in range(9)}
    for qi in range(54):
        q=f'fixture_question_{qi:02d}'
        for k in range(9):
            for n in (8,16):
                for replica in range(1 if n==8 and k==8 else 2):
                    sid=f'q{qi:02d}_K{k}_N{n}_r{replica}'
                    row=dict(sid=sid,question=q,gold=k,n_frames=n,split='train',qa_sha256=objsha(['qa',sid]),content_sha256=objsha(['content',sid]))
                    manifest['splits'][f'train_N{n}']['samples'].append(row);gids=[]
                    for prefix in ([],[tokens[str(k)]]):
                        fid=objsha(['global',q,prefix]);gids.append(fid)
                        features[fid]=dict(kind='global',question=q,prefix_ids=prefix,layout_id=fid)
                        layouts[fid]=dict(input_ids=[100+qi]+prefix)
                    scenes[sid]=dict(row,target_ids=[tokens[str(k)],151645],global_feature_ids=gids)
                    slots.extend([sid]*(2 if n==8 and k==8 else 1))
    train={r['sid']:r for group in manifest['splits'].values() for r in group['samples']}
    schedule=dict(epoch_slots=slots,unique_training_sids=sorted(scenes),slot_metadata=[dict(slot=i,sid=s,n_frames=train[s]['n_frames'],
        gold=train[s]['gold'],question=train[s]['question']) for i,s in enumerate(slots)])
    cache=dict(complete=True,training_only=True,scenes=scenes,features=features)
    plan=dict(scenes=scenes,count_token_ids=tokens,eos_token_id=151645,features=features,layouts=layouts)
    pairs=build_pairs(manifest,cache,schedule,plan)
    a=presentation_order(pairs,10);b=presentation_order(pairs,10)
    need(a==b and a!=presentation_order(pairs,11),'Paired order must be deterministic and seed-dependent')
    expected=Counter({sid:40*count for sid,count in Counter(slots).items()})
    need(Counter(r['sid'] for r in a)==expected,'Forty epochs changed the original weighted multiset')
    rng=random.Random(10)
    for epoch in (1,2):
        perm=list(range(PAIR_COUNT));rng.shuffle(perm)
        need([r['slot'] for r in a[(epoch-1)*1944:epoch*1944:2]]==perm,'RNG must advance persistently across epochs')
    batches=list(iter_batches(a));need(len(batches)==4860 and Counter(r['epoch'] for r in batches[121])=={1:8,2:8},
                                     'Epoch boundary must carry four complete pairs on each side')
    def reject(call):
        try:call()
        except ValueError:return
        raise AssertionError('Malformed pairing metadata was accepted')
    bad=copy.deepcopy(cache);sid=pairs[0]['sids'][1];bad['scenes'][sid]['target_ids'][0]+=1
    reject(lambda:build_pairs(manifest,bad,schedule,dict(plan,scenes=bad['scenes'])))
    bad=copy.deepcopy(cache);bad['scenes'][sid]['global_feature_ids'].reverse()
    reject(lambda:build_pairs(manifest,bad,schedule,dict(plan,scenes=bad['scenes'])))
    bad=copy.deepcopy(schedule);bad['epoch_slots'][0]=bad['epoch_slots'][1]
    reject(lambda:build_pairs(manifest,cache,bad,plan))
    bad=copy.deepcopy(a);bad[1]['pair_side']='N8';reject(lambda:validate_order(bad,pairs))
    return dict(passed=True,tests=['full972_pair_inventory','saturated54_slots_repeat_only','native_count_EOS_alignment',
        'paired_global_feature_identity','exact_V7_weighted_multiset','persistent_RNG_not_seed_plus_epoch',
        'fixed_seed_and_distinct_seeds','eight_intact_pairs_per_batch','carry_epoch_boundaries',
        'reject_wrong_target','reject_reversed_prefix_features','reject_weight_change','reject_split_or_reversed_pair'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-test',action='store_true',required=True)
    parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),
         'Schedule self-tests require CPU Slurm')
    print(json.dumps(self_test(),indent=2),flush=True)


if __name__=='__main__':main()

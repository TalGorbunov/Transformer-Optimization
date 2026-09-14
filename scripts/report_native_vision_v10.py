"""Independent V10 CPU audit: supported-answer native vision extrapolation.

No model fitting or new predictions. Independently bind data, causal training
features, paired sequence weights, all optimizer records and raw greedy outputs.
Efficacy is separate from the mandatory all-selected-checkpoint native audit.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import random
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import report_native_vision_v9 as prior
from scripts import stage_native_vision_v10_data as stage
from scripts import stage_native_vision_v10_features as feature_stage
need,read,sha,objsha,save,close,bind,ledger,tensor_info=(prior.need,prior.read,prior.sha,prior.objsha,prior.save,prior.close,prior.bind,prior.ledger,prior.tensor_info)
verify_inventory,verify_source=prior.verify_inventory,prior.verify_source
parse_text,select,stored_native_logits=prior.parse_text,prior.select,prior.stored_native_logits
DATA,MODEL=prior.DATA,prior.MODEL
OUT=REPO/'outputs/native_aggregation_vlm/v10'
CKPT=Path('/mnt/ckpts/gabriele/gnn_transformer/native_aggregation_vlm_v10')
CACHE=DATA/'v10_parallel_local/feature_cache.json'
DEV_STEPS=[918,1836,2754,3672,4590]
CONDITIONS=('ce','consistency')
SEEDS=(14,15)
OWN=tuple(dict.fromkeys(('scripts/report_native_vision_v10.py','slurm/native_vision_v10_report.sbatch',
    'gnnformer/paired_sequence_objectives.py','scripts/native_vision_v10_pairs.py',*prior.OWN,*stage.OWN,*feature_stage.OWN)))
POLICY = dict(
    protocol='v10_supported_answer_native_residual_consistency', native_arm='parallel',
    conditions=['ce', 'consistency'], seeds=[14, 15], epochs=40,
    pair_slots_per_epoch=918, scene_slots_per_epoch=1836, unique_training_scenes=1782,
    scene_presentations=73440, pair_presentations=36720, target_positions=177120,
    batch_size=16, pairs_per_batch=8, steps=4590,
    dev_steps=[918, 1836, 2754, 3672, 4590], dev_examples=64, test_examples=272,
    lr=.001, warmup=50, final_lr=.00001, weight_decay=0., clip_norm=1.,
    rank=96, hidden_size=3584, parameters=1041600, merge='sum', post_activation='silu',
    native_dtype='torch.float16', branch_dtype='torch.float32',
    consistency_coefficients={'ce': 0., 'consistency': 1.}, consistency_epsilon=1e-6,
    ce_reduction='mean_over_scenes_of_mean_over_valid_native_target_positions',
    consistency_reduction='mean_over_pairs_of_mean_over_corresponding_strict_prefix_positions',
    denominator='detached_squared_L2_of_identical_frozen_global_state_plus_fixed_epsilon',
    pair_order='persistent_random.Random(seed)_fresh_canonical_pair_permutation_each_epoch',
    within_pair_order='registered_pair_sides_0_then_1',
    saturation_rule='retain54_N16_K16_same_SID_pairs_with_exact_zero_regularizer',
    path_loss_role='diagnostic_only_no_loss_weight_no_inference_role',
    full_answer_targets='canonical_native_numeral_tokens_plus_exactly_one_terminal_EOS',
    max_new_tokens=4, exact_requires_eos=True,
    selection='max_dev_exact_then_min_raw_first_token_NLL_then_earliest',
    native_eos=[151645, 151643], target_eos=151645, profile_steps=32,
    training_count_support=list(range(17)), development_count_support=list(range(16)),
    test_count_support=list(range(17)), maximum_training_N=16, test_N=[32, 64],
    bootstrap_seed=20261008, bootstrap_replicates=10000,
)


def sources():return {name:sha(REPO/name) for name in OWN}


def snapshot(out):
    frozen=sources();(out/'code').mkdir()
    for name,digest in frozen.items():
        target=out/'code'/name.replace('/','_');target.write_bytes((REPO/name).read_bytes());bind(target,digest)
    save(out/'source_hashes.json',frozen)
    return frozen


def learning_rate(step):
    return .001*step/50 if step<=50 else .00001+.00099*(1+math.cos(math.pi*(step-50)/4540))/2


def independent_pairs(data,cache=None):
    """Reconstruct semantic pairs without trusting the published pair selection."""
    grouped=defaultdict(list)
    for sid,row in data['train'].items():grouped[row['target_character'],row['target_room'],row['gold']].append(row)
    result=[]
    for c in stage.CHARS:
        for room in stage.PARK_ROOMS:
            for k in range(17):
                rows=grouped[c,room,k]
                if k<=8:members=sorted(rows,key=lambda r:r['n_frames'])
                elif k<=15:members=sorted(rows,key=lambda r:r['replica'])
                else:
                    need(len(rows)==1,'Saturated K16 has more than one distinct training scene');members=rows*2
                need(len(members)==2 and [r['n_frames'] for r in members]==([8,16] if k<=8 else [16,16]),'Pair frame support differs')
                kind='cross_length' if k<=8 else 'same_length' if k<=15 else 'saturated_identity'
                q=members[0]['question'];pid=objsha(['v10_pair',q,k]);ids=[r['sid'] for r in members]
                need(all(r['question']==q and r['gold']==k and r['pair_id']==pid and r['pair_kind']==kind for r in members),'Pair semantic identity differs')
                need((ids[0]==ids[1])==(k==16),'Same-SID exception differs')
                item=dict(slot=len(result),pair_id=pid,question=q,gold=k,target_character=c,target_room=room,
                    sids=ids,n_frames=[r['n_frames'] for r in members],replicas=[r['replica'] for r in members],
                    pair_kind=kind,same_sid=k==16)
                if cache is not None:
                    a,b=[cache['scenes'][sid] for sid in ids]
                    need(a['target_ids']==b['target_ids'] and a['target_prefixes']==b['target_prefixes']
                         and a['global_feature_ids']==b['global_feature_ids'],'Pair native causal target/global identities differ')
                result.append(item)
    need(len(result)==918 and Counter(s for p in result for s in p['sids'])==Counter(data['schedule']['epoch_slots']),
         'Pair schedule changes registered scene weights')
    return result


def audit_data():
    roots={'balanced':DATA/'v10_balanced','fresh':DATA/'v10_fresh'}
    manifests={name:read(root/'main_manifest.json') for name,root in roots.items()}
    b,m=manifests['balanced'],manifests['fresh'];schedule=read(roots['balanced']/'schedule.json');published_pairs=read(roots['balanced']/'pairing.json')
    inv,excluded=verify_inventory(b['inventory_file'],25)
    priorpaths={p:DATA/'v9_fresh'/f'{p}_manifest.json' for p in ('main','count')}
    previous={p:read(path) for p,path in priorpaths.items()}
    need(previous['main']['source_manifest_sha256']==previous['count']['source_manifest_sha256'],'V9 prior inventory differs')
    exact=dict(previous['main']['source_manifest_sha256'],**{str(path):sha(path) for path in priorpaths.values()})
    need(len(exact)==25 and inv['source_manifest_sha256']==exact,'V10 does not preserve exact V9 prior23 plus both V9 tests')
    paths=[roots['balanced']/'schedule.json',roots['balanced']/'pairing.json']
    for name,manifest in manifests.items():
        need(manifest['dataset_root']==str(roots[name]) and manifest['train_seed']==20261005
             and manifest['dev_seed']==20261006 and manifest['fresh_test_seed']==20261007,'V10 roots or seeds differ')
        need(manifest['source_manifest_sha256']==exact and manifest['standard_downstream_resize']==392
             and manifest['test_excludes_all_current_train_and_dev'] is True,'V10 exclusions/resize metadata differs')
        for field in ('audit','stage_plan','inventory'):bind(manifest[field+'_file'],manifest[field+'_sha256'])
        ledger(manifest['source_sha256']);ledger(manifest['generator_code_sha256'])
        need(read(manifest['inventory_file'])==inv,'Train/test exclusion inventories differ')
        paths.extend([roots[name]/'main_manifest.json',*[Path(manifest[k+'_file']) for k in ('audit','stage_plan','inventory','stage_audit')]])
    plan=read(b['stage_plan_file']);need(plan==read(m['stage_plan_file']),'Train/test frozen plans differ')
    need(plan['source_sha256']==b['source_sha256'] and plan['expected']==stage.SPEC
         and plan['train_seed']==20261005 and plan['dev_seed']==20261006 and plan['test_seed']==20261007,'Data plan protocol differs')
    ledger(plan['protected_source_sha256'])
    for key in ('samples','inventory','generator_checks'):
        bind(plan[key+'_file'],plan[key+'_sha256']);paths.append(Path(plan[key+'_file']))
    payload=read(plan['samples_file']);need(objsha(payload)==plan['samples_object_sha256'],'Frozen semantic payload changed')
    need(set(b['splits'])=={'train_N8','train_N16','dev_N16'} and set(m['splits'])=={'test_N32','test_N64'},'Unexpected V10 data splits')
    need({k:len(v['samples']) for k,v in b['splits'].items()}==dict(train_N8=486,train_N16=1296,dev_N16=64)
         and all(len(v['samples'])==136 for v in m['splits'].values()),'V10 split cardinalities differ')
    # The immutable renderer audit expects Python integer histogram keys before
    # JSON serialization. Validate the serialized map first, then normalize only
    # that representational detail in a private copy for its semantic audit.
    audit_manifests=json.loads(json.dumps(manifests))
    # Published JSON sorts mapping keys; the original stager records use its
    # explicit split insertion order. Restore that order only in this audit
    # copy, keeping exact equality for every reconstructed semantic record.
    for purpose,cells in (('balanced',('train_N8','train_N16','dev_N16')),('fresh',('test_N32','test_N64'))):
        audit_manifests[purpose]['splits']={cell:audit_manifests[purpose]['splits'][cell] for cell in cells}
    for manifest in audit_manifests.values():
        for cell in manifest['splits'].values():
            expected={str(k):v for k,v in sorted(Counter(r['gold'] for r in cell['samples']).items())}
            need(cell['gold_histogram']==expected,'Serialized gold histogram differs')
            cell['gold_histogram']={int(k):v for k,v in cell['gold_histogram'].items()}
    checked=json.loads(json.dumps(stage.audit_published(audit_manifests,payload,excluded,{k:r/'render_cache' for k,r in roots.items()})))
    archived=read(b['audit_file']);need(archived==read(m['audit_file']),'Train/test semantic audits differ')
    stripped=[{k:v for k,v in r.items() if k!='prior_occurrences'} for r in archived['historical_saturated_reuse']]
    need(checked['historical_saturated_reuse']==stripped,'Permitted historical reuse differs')
    for key,value in checked.items():
        if key!='historical_saturated_reuse':need(archived[key]==value,'Independent image/semantic audit differs: '+key)
    historical_index=defaultdict(list)
    for p,h in exact.items():
        for old in stage.previous.content_rows(read(p)):
            historical_index[old['content_sha256']].append(dict(manifest=p,sid=old['sid'],n_frames=old['n_frames']))
    for row in archived['historical_saturated_reuse']:
        occurrences=historical_index[row['content_sha256']]
        need(sorted(row['prior_occurrences'],key=objsha)==sorted(occurrences,key=objsha) and occurrences,'Historical saturated provenance differs')
    train={r['sid']:r for cell in ('train_N8','train_N16') for r in b['splits'][cell]['samples']}
    data=dict(train=train,schedule=schedule)
    pairs=independent_pairs(data)
    need(pairs==payload['pairs']==published_pairs['pairs'] and objsha(pairs)==published_pairs['pairs_sha256'],'Independent pair selection differs')
    need(schedule['epoch_slots']==[sid for p in pairs for sid in p['sids']]==payload['epoch_slots']
         and schedule['unique_training_sids']==sorted(train) and schedule['pair_count']==918 and schedule['scene_slots']==1836,'Weighted schedule differs')
    expected_meta=[]
    for p in pairs:
        for side,sid in enumerate(p['sids']):
            row=train[sid];expected_meta.append(dict(slot=len(expected_meta),pair_slot=p['slot'],pair_id=p['pair_id'],pair_side=side,
                sid=sid,n_frames=row['n_frames'],gold=row['gold'],question=row['question'],replica=row['replica'],pair_kind=p['pair_kind']))
    need(schedule['slot_metadata']==expected_meta,'Scene-slot semantic metadata differs')
    need(schedule['manifest_file']==published_pairs['manifest_file']==str(roots['balanced']/'main_manifest.json')
         and schedule['manifest_sha256']==published_pairs['manifest_sha256']==sha(roots['balanced']/'main_manifest.json')
         and published_pairs['schedule_file']==str(roots['balanced']/'schedule.json')
         and published_pairs['schedule_sha256']==sha(roots['balanced']/'schedule.json')
         and published_pairs['pair_count']==918 and published_pairs['weighted_scene_slots']==1836,'Schedule/pair manifest bindings differ')
    finished=read(b['stage_audit_file']);need(finished==read(m['stage_audit_file']) and finished['passed'] is True
        and finished['completed'] is True and finished['manifest_sha256']=={k:sha(r/'main_manifest.json') for k,r in roots.items()}
        and finished['schedule_sha256']==sha(roots['balanced']/'schedule.json') and finished['pairing_sha256']==sha(roots['balanced']/'pairing.json'),'Completed image stage bindings differ')
    testrows=[('main_'+cell,row) for cell,group in m['splits'].items() for row in group['samples']]
    summary=dict(passed=True,prior_manifests=25,data_bindings={str(p):sha(p) for p in paths},
        balanced={k:v for k,v in checked.items() if k not in ('records','per_cell')},fresh_contexts=272,test_families=136,
        local_visual_atoms_may_recur=True,historical_saturation_exception='Only training N8K8 and N16K16; K16 same-SID pair retains two CE slots and zero regularizer',
        all_test_answer_values_supported_in_training=True,no_unseen_answer_value_claim=True,
        training_length_count_correlation='Only K0..8 cross-length pairs; K9..15 same-length pairs; K16 identity pairs')
    return dict(summary=summary,balanced=b,fresh=m,schedule=schedule,pairing=published_pairs,train=train,
        dev=[('dev_N16',r) for r in b['splits']['dev_N16']['samples']],test=testrows)


def expected_order(pairs,seed):
    rng=random.Random(seed);rows=[]
    for epoch in range(1,41):
        slots=list(range(918));rng.shuffle(slots)
        for slot in slots:
            pair=pairs[slot]
            for side,sid in enumerate(pair['sids']):
                rows.append(dict(epoch=epoch,slot=slot,pair_id=pair['pair_id'],question=pair['question'],gold=pair['gold'],
                    sid=sid,pair_side=side,pair_kind=pair['pair_kind'],n_frames=pair['n_frames'][side],replica=pair['replicas'][side]))
    return rows


def metrics(rows):
    result=prior.metrics(rows)
    if all('first_token_correct' in row for row in rows):result['first_token_correct']=sum(row['first_token_correct'] for row in rows)
    return result


def summarize(rows):
    need(len(rows)==272 and Counter((r['n_frames'],r['gold']) for r in rows)==Counter({(n,k):8 for n in (32,64) for k in range(17)}),'Registered test denominators differ')
    cells={f'main_test_N{n}':metrics([r for r in rows if r['n_frames']==n]) for n in (32,64)}
    by_k={f'main_test_N{n}/K{k}':metrics([r for r in rows if r['n_frames']==n and r['gold']==k]) for n in (32,64) for k in range(17)}
    partitions={}
    for n in (32,64):
        partitions[f'N{n}']={label:metrics([r for r in rows if r['n_frames']==n and r['gold'] in ks])
            for label,ks in (('K0_8',range(9)),('K9_15',range(9,16)),('K16',(16,)),('K9_16',range(9,17)),('nonzero',range(1,17)))}
    return dict(all=metrics(rows),cells=cells,by_n_k=by_k,partitions=partitions)


def integer_decision(treated,control):
    n32=treated['cells']['main_test_N32']['correct'];n64=treated['cells']['main_test_N64']['correct']
    delta=n64-control['cells']['main_test_N64']['correct'];delta32=n32-control['cells']['main_test_N32']['correct']
    larger=treated['partitions']['N64']['K9_16']['correct'];primary=delta>=7 and delta32>=-6
    practical=primary and n32>=123 and n64>=109 and larger>=52
    return dict(n64_additional_correct=delta,n64_difference_pp=100*delta/136,n32_additional_correct=delta32,n32_difference_pp=100*delta32/136,
        n64_gain_at_least_5pp=delta>=7,n32_loss_at_most_5pp=delta32>=-6,primary=primary,
        consistency_n32_correct=n32,consistency_n32_at_least_90pct=n32>=123,
        consistency_n64_correct=n64,consistency_n64_at_least_80pct=n64>=109,
        consistency_larger_k_n64_correct=larger,consistency_larger_k_n64_at_least_80pct=larger>=52,practical=practical)


def bootstrap(all_rows,replicates=10000,seed=20261008):
    import numpy as np
    rng=np.random.default_rng(seed);reference=all_rows['consistency',14];groups=defaultdict(dict)
    for r in reference:groups[r['gold']].setdefault(r['anchor_id'],set()).add(r['n_frames'])
    need(set(groups)==set(range(17)) and all(len(groups[k])==8 and None not in groups[k]
        and all(ns=={32,64} for ns in groups[k].values()) for k in range(17)),'Bootstrap family coverage differs')
    ordered=[(k,anchor) for k in range(17) for anchor in sorted(groups[k])];arrays={}
    for key,rows in all_rows.items():
        lookup={(r['gold'],r['anchor_id'],r['n_frames']):float(r['exact']) for r in rows}
        need(len(lookup)==len(rows)==272,'Duplicate or missing family rows')
        arrays[key]=np.asarray([[lookup[k,a,n] for n in (32,64)] for k,a in ordered])
    # One draw matrix shared by every reported subgroup, length, condition and
    # fitted seed. Subgroups select columns from these same stratified draws.
    indices=np.concatenate([rng.integers(0,8,size=(replicates,8))+k*8 for k in range(17)],axis=1)
    result={}
    for label,ks in (('all',range(17)),('K0_8',range(9)),('K9_15',range(9,16)),('K16',(16,)),('K9_16',range(9,17)),('nonzero',range(1,17))):
        selected=np.concatenate([np.arange(k*8,(k+1)*8) for k in ks]);draws=indices[:,selected];endpoints={}
        for n,column in ((32,0),(64,1)):
            samples={key:value[:,column][draws].mean(1) for key,value in arrays.items()};cell={}
            for fit_seed in SEEDS:
                a,b=samples['consistency',fit_seed],samples['ce',fit_seed]
                cell[str(fit_seed)]=dict(difference_ci95_pp=(100*np.quantile(a-b,[.025,.975])).tolist(),
                    consistency_ci95_pp=(100*np.quantile(a,[.025,.975])).tolist(),ce_ci95_pp=(100*np.quantile(b,[.025,.975])).tolist())
            pooled=sum(samples['consistency',s]-samples['ce',s] for s in SEEDS)/2
            cell['pooled_fixed_two_seeds']=dict(difference_ci95_pp=(100*np.quantile(pooled,[.025,.975])).tolist());endpoints[f'N{n}']=cell
        result[label]=endpoints
    return dict(replicates=replicates,seed=seed,unit='Complete N32-to-N64 family stratified by K; shared draws across all lengths, conditions, subgroups and fixed seeds',
        inference_scope='Conditional example uncertainty for these fitted seeds, not a training-seed population',results=result)


def verify_cache(path,data,tokenizer):
    import torch
    cache=read(path);bind(cache['plan_file'],cache['plan_sha256']);plan=read(cache['plan_file'])
    need(cache['protocol']=='v10_parallel_local_training_features' and cache['complete'] is True and cache['training_only'] is True,'Incomplete or wrong feature cache')
    need(set(cache['scenes'])==set(data['train']) and cache['scenes']==plan['scenes'],'Cache training scene coverage differs')
    ledger(cache['source_sha256']);ledger(plan['source_sha256'])
    for record in plan.get('source_files',{}).values():bind(record['path'],record['sha256'])
    need(cache['model']==plan['model'] and cache['runtime']==plan['runtime'] and cache['processor']==plan['processor']
         and cache['native_api']==plan['native_api'] and cache['source_files']==plan['source_files'],'Cache model/runtime/native API ancestry differs')
    need(feature_stage.verify_plan(cache['plan_file'],pixels=False)==plan,'Frozen training feature plan differs')
    need(cache['native_dtypes']==dict(norm='torch.float16',lm_head='torch.float16'),'Cached native dtype differs')
    need(set(cache['features'])==set(plan['features']) and cache['feature_count']==len(plan['features']),'Feature coverage differs')
    features=plan['features'];used=set();global_by_prefix=defaultdict(set);target_sequences={}
    for k in range(17):
        tokens=tokenizer(str(k),add_special_tokens=False)['input_ids']
        need(len(tokens)==(1 if k<10 else 2) and tokenizer.decode(tokens)==str(k)
             and all(t not in tokenizer.all_special_ids for t in tokens),'Canonical numeral target tokenization differs')
        target_sequences[str(k)]=tokens+[tokenizer.eos_token_id]
    need(tokenizer.eos_token_id==151645 and plan['target_token_ids']==target_sequences,'Cache target inventory differs')
    for sid,scene in cache['scenes'].items():
        sample=data['train'][sid]
        need(all(scene[k]==sample[k] for k in ('gold','question','n_frames','content_sha256','qa_sha256')) and scene['split']=='train','Cache scene metadata differs')
        targets=target_sequences[str(sample['gold'])];prefixes=[targets[:t] for t in range(len(targets))]
        need(scene['target_ids']==targets and scene['target_prefixes']==prefixes,'Cache sees wrong native answer/prefix sequence')
        groups=scene['local_feature_ids']+[scene['global_feature_ids']]
        need(len(groups)==sample['n_frames']+1,'Feature group count differs')
        for i,ids in enumerate(groups):
            need(len(ids)==len(targets) and len(set(ids))==len(ids),'Causal target positions missing or collide')
            base_layout=None
            for j,fid in enumerate(ids):
                f=features[fid];used.add(fid);kind='local' if i<sample['n_frames'] else 'global'
                need(f['prefix_ids']==prefixes[j] and f['kind']==kind and f['question']==sample['question'],'Feature strict causal prefix/question differs')
                if kind=='local':
                    image=sample['image_files'][i]['sha256'];pid=objsha([image,sample['question']])
                    need(f['pair_id']==pid and fid==objsha(['local',pid,prefixes[j]]),'Local cache key depends on wrong metadata')
                    pair=plan['pairs'][pid];need(pair['image_sha256']==image and pair['question']==sample['question'],'Local image/question key differs')
                else:
                    need(fid==objsha(['global',sample['question'],prefixes[j]]),'Global cache key depends on gold/length/N')
                    global_by_prefix[tuple(prefixes[j])].add(fid)
                layout=plan['layouts'][f['layout_id']]
                if j==0:base_layout=layout['input_ids']
                need(layout['input_ids']==base_layout+prefixes[j],'Continuation prefix changes pre-answer input tokens')
    need(used==set(features) and sum(f['kind']=='global' for f in features.values())==972,'Unused features or global prefix inventory differs')
    # Shared [1] is one cache identity per question for both K1 and K10..16.
    one=tuple(target_sequences['1'][:-1]);need(len(global_by_prefix[()])==54 and len(global_by_prefix[one])==54,'Shared empty/[1] global feature identities split by answer')
    locations=cache['features'];files={}
    for fid,item in locations.items():
        if item['file'] in files:need(files[item['file']]==item['file_sha256'],'Conflicting tensor file hashes')
        files[item['file']]=item['file_sha256']
    seen=set()
    for filename,digest in files.items():
        bind(filename,digest);blob=torch.load(filename,map_location='cpu',weights_only=True);ids=blob['feature_ids'];states=blob['states']
        need(blob['schema_version']==1 and states.shape==(len(ids),3584) and states.dtype==torch.float16
             and bool(torch.isfinite(states).all()) and len(ids)==len(set(ids)),'Malformed frozen feature tensor')
        for row,fid in enumerate(ids):
            need(fid not in seen and fid in locations,'Duplicate/unknown cached feature');seen.add(fid);item=locations[fid]
            need(item['file']==filename and item['row']==row and tensor_info(states[row])['sha256']==item['state_sha256'],'Native feature bytes changed')
        del blob,states
    need(seen==set(features) and len(cache['shards'])==4,'Incomplete feature shards')
    shard_ids=set()
    for item in cache['shards']:
        directory=Path(item['directory']);bind(directory/'summary.json',item['summary_sha256']);summary=read(directory/'summary.json')
        need(summary['completed'] is True and summary['passed'] is True and summary['computational_integrity_passed'] is True
             and summary['profile'] is False and summary['shard']==item['shard'] and item['shard'] not in shard_ids,'Feature shard failed or duplicated')
        shard_ids.add(item['shard'])
        need(summary['features_file'] in files and summary['features_sha256']==files[summary['features_file']]
             and summary['plan_sha256']==cache['plan_sha256'],'Shard tensor/plan binding differs')
        bind(summary['observations_file'],summary['observations_sha256']);ledger(summary['source_sha256'])
    need(shard_ids==set(range(4)),'Feature shard indices differ')
    profile=Path(cache['profile_directory']);bind(profile/'summary.json',cache['profile_summary_sha256']);prof=read(profile/'summary.json')
    need(prof['passed'] is True and prof['computational_integrity_passed'] is True and prof['runtime_projection_passed'] is True
         and prof['plan_sha256']==cache['plan_sha256'] and len(prof['shard_projected_seconds'])==4
         and all(0<x<=600 for x in prof['shard_projected_seconds']),'Prospective cache software/runtime profile failed')
    from scripts.cache_native_vision_v10_features import verify_profile
    need(verify_profile(profile,plan,cache['plan_file'])==prof,'Cache profile native replay/coverage arithmetic differs')
    return cache,dict(file=str(path),sha256=sha(path),plan_file=cache['plan_file'],plan_sha256=cache['plan_sha256'],features=len(features),
        training_scenes=len(cache['scenes']),all_strict_causal_prefixes_verified=True,shared_prefix_keys_verified=True,
        complete_native_target_sequences=target_sequences,cache_profile_passed=prof['passed'],cache_numerical_gate_passed=prof.get('numerical_gate_passed'),
        numerical_differences_descriptive=True)


def independent_layout(sequences):
    need(len(sequences)==16 and all(isinstance(ids,list) and len(ids) in (2,3) for ids in sequences),'Expected sixteen native target sequences')
    offsets=[0]
    for ids in sequences:offsets.append(offsets[-1]+len(ids))
    pair_lengths=[];pair_offsets=[0]
    for i in range(0,16,2):
        need(sequences[i]==sequences[i+1],'Mismatched paired answer sequence')
        pair_lengths.append(len(sequences[i]));pair_offsets.append(pair_offsets[-1]+len(sequences[i]))
    return dict(target_ids=[t for ids in sequences for t in ids],scene_lengths=[len(ids) for ids in sequences],scene_offsets=offsets,
        prefix_ids=[ids[:t] for ids in sequences for t in range(len(ids))],pair_lengths=pair_lengths,pair_offsets=pair_offsets)


def check_loss_row(row,condition,sequences,identity):
    import numpy as np
    layout=independent_layout(sequences);coefficient=0. if condition=='ce' else 1.
    need(row['consistency_coefficient']==coefficient,'Registered consistency coefficient differs')
    for key in ('target_ids','scene_lengths','scene_offsets','prefix_ids','pair_lengths'):
        need(row[key]==layout[key],'Ragged causal layout differs: '+key)
    need(row['saturated_identity_pairs']==identity and len(identity)==8 and all(type(x) is bool for x in identity),'Same-SID pair metadata differs')
    fields=('loss','ce_loss','consistency_loss','path_loss','weighted_consistency_loss','gradient_norm','seconds')
    need(all(math.isfinite(row[k]) and row[k]>=0 for k in fields),'Nonfinite or negative training statistic')
    need(close(row['weighted_consistency_loss'],coefficient*row['consistency_loss'])
         and close(row['loss'],row['ce_loss']+coefficient*row['consistency_loss'])
         and row['clipped']==(row['gradient_norm']>1),'Weighted objective or clipping arithmetic differs')
    q=layout['scene_offsets'][-1];p=layout['pair_offsets'][-1];positions={}
    for key,size in (('ce',q),('consistency',p),('path',p)):
        x=np.asarray(row['position_losses'][key],dtype=np.float64)
        need(x.shape==(size,) and np.isfinite(x).all() and (x>=0).all(),'Malformed valid-position losses: '+key);positions[key]=x
    per_scene=[float(positions['ce'][a:b].mean()) for a,b in zip(layout['scene_offsets'][:-1],layout['scene_offsets'][1:])]
    need(len(row['per_scene_ce'])==16 and all(close(a,b) for a,b in zip(per_scene,row['per_scene_ce']))
         and close(row['ce_loss'],sum(per_scene)/16),'CE is not the equal mean of scene token means')
    for name in ('consistency','path'):
        values=[float(positions[name][a:b].mean()) for a,b in zip(layout['pair_offsets'][:-1],layout['pair_offsets'][1:])]
        need(len(row['per_pair_'+name])==8 and all(close(a,b) for a,b in zip(values,row['per_pair_'+name]))
             and close(row[name+'_loss'],sum(values)/8),'Regularizer is not the equal mean of pair prefix means')
        for i,is_identity in enumerate(identity):
            if is_identity:
                a,b=layout['pair_offsets'][i:i+2]
                need(bool((positions[name][a:b]==0).all()) and row['per_pair_'+name][i]==0.,'Saturated identical pair must have exactly zero regularizer')
    denominator=np.asarray(row['pair_position_denominator'],dtype=np.float64)
    residual=np.asarray(row['residual_difference_norms'],dtype=np.float64);bound=np.asarray(row['B_norms'],dtype=np.float64)
    need(all(x.shape==(p,) and np.isfinite(x).all() for x in (denominator,residual,bound))
         and (denominator>=1e-6).all() and (residual>=0).all() and (bound>=0).all(),'Malformed per-prefix objective norms')
    need(all(close(float(a),float(b)) for a,b in zip(residual**2/denominator,positions['consistency']))
         and all(close(float(a),float(b)) for a,b in zip(bound**2/denominator,positions['path'])),'Norm/denominator regularizer identity differs')


def loss_summary(logs):
    fields=('loss','ce_loss','consistency_loss','path_loss','weighted_consistency_loss','gradient_norm')
    blocks=[]
    for start in range(0,4590,918):
        rows=logs[start:start+918]
        blocks.append(dict(first_step=start+1,last_step=start+918,
            means={key:sum(row[key] for row in rows)/len(rows) for key in fields},
            clipping_fraction=sum(row['clipped'] for row in rows)/len(rows),
            valid_target_positions=sum(len(row['target_ids']) for row in rows),
            saturated_identity_pairs=sum(sum(row['saturated_identity_pairs']) for row in rows)))
    return dict(blocks=blocks,final_update={k:logs[-1][k] for k in fields},
        path_penalty_diagnostic_only=True,
        validation_scope='Recorded per-position/per-scene/per-pair arithmetic and every causal target; intermediate model trajectories are not reconstructed')


def audit_eval(path,expected,arm,tokenizer):
    value=prior.audit_eval(path,expected,arm,tokenizer)
    for row in value['rows']:
        first=tokenizer(str(row['gold']),add_special_tokens=False)['input_ids'][0]
        actual=row['generated_ids'][0]==first
        if 'first_token_correct' in row:need(row['first_token_correct']==actual,'Recorded first-token correctness differs')
        row['first_token_correct']=actual
    return value


def verify_history():
    root=REPO/'outputs/native_aggregation_vlm/v9'
    final_path=root/'finalization/final_442084/summary.json'
    diagnostic_path=root/'response_surface/run_442099/summary.json'
    completed=read(final_path)
    need(completed['passed'] is True and completed['finalized'] is True,'Canonical V9 final verification incomplete')
    bind(completed['final_acceptance_file'],completed['final_acceptance_sha256']);final=read(completed['final_acceptance_file'])
    need(final['completed'] is True and final['verification_passed'] is True and final['accepted_vision_milestone'] is False,
         'Canonical V9 failure status differs')
    for item in final['artifacts'].values():bind(item['path'],item['sha256'])
    report_path=Path(final['artifacts']['analysis']['path']);report=read(report_path)
    need(report['passed'] is True and report['audit_passed'] is True
         and report['primary_both_seeds'] is False and report['practical_both_seeds'] is False,
         'Previous path-objective failure must be retained')
    diagnostic=read(diagnostic_path)
    need(diagnostic['passed'] is True and diagnostic['diagnostic_only'] is True and diagnostic['no_fit'] is True
         and diagnostic['vlm_forward_calls']==0 and diagnostic['synthetic_points']==432,'Canonical V9 diagnostic incomplete')
    bind(diagnostic['analysis_file'],diagnostic['analysis_sha256']);bind(diagnostic['raw_file'],diagnostic['raw_sha256'])
    analysis=read(diagnostic['analysis_file'])
    need(Path(analysis['report_file']).resolve()==report_path.resolve() and analysis['report_sha256']==sha(report_path),
         'Historical diagnostic chose another V9 report')
    bind(analysis['report_completion_file'],analysis['report_completion_sha256']);completion=read(analysis['report_completion_file'])
    need(completion['passed'] is True and completion['analysis_file']==str(report_path)
         and completion['analysis_sha256']==sha(report_path),'Canonical V9 report completion differs')
    for value,directory,folder in ((completed,final_path.parent,'source'),(report,report_path.parent,'code'),(diagnostic,diagnostic_path.parent,'source')):
        ledger(value['source_sha256'])
        for name,digest in value['source_sha256'].items():bind(directory/folder/name.replace('/','_'),digest)
    post=read(final['artifacts']['selected_checkpoint_audit']['path'])
    need(final['cache_numerical_failures']==post['cache_numerical_failures']
         and final['post_strict_cache_numerical_gate_passed']==post['strict_cache_numerical_gate_passed'],
         'Prior descriptive numerical failures changed')
    return dict(finalization=dict(file=str(final_path),sha256=sha(final_path)),
        diagnostic=dict(file=str(diagnostic_path),sha256=sha(diagnostic_path)),previous_path_objective_failed=True,
        new_training_cache=True,no_selection_on_prior_diagnostic_predictions=True)


def verify_run(directory,data,tokenizer,cache,pairing_value):
    import torch
    from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
    directory=Path(directory).resolve();config=read(directory/'config.json');summary=read(directory/'summary.json')
    condition,seed=config['condition'],config['seed'];runid=config['run_id']
    need(directory.parent==OUT and directory.name==runid and runid.startswith(f'run_{condition}_s{seed}_')
         and condition in CONDITIONS and seed in SEEDS and config['arm']=='parallel' and config['profile'] is False,'Unregistered V10 main run')
    need(config['policy']==POLICY and all(summary.get(k)==v for k,v in config.items()),'V10 policy/config/summary differs')
    coefficient=0. if condition=='ce' else 1.
    need(config['consistency_coefficient']==coefficient and summary['passed'] is True and summary['completed'] is True
         and summary['computational_integrity_passed'] is True and summary['steps']==4590 and summary['native_test_count']==272,
         'Incomplete main or changed objective coefficient')
    verify_source(directory,config['source_sha256']);bind(config['plan_file'],config['plan_sha256'])
    plan_path=Path(config['plan_file']);plan=read(plan_path)
    need(plan_path.with_suffix('.sha256').read_text().strip()==config['plan_sha256']
         and plan['source_sha256']==config['source_sha256'] and plan['policy']==POLICY,'Training CPU plan/source differs')
    cpu=read(plan_path.parent/'summary.json')
    need(cpu['passed'] is True and cpu['plan_sha256']==config['plan_sha256'] and cpu['source_sha256']==config['source_sha256'],'Matching completed training CPU gate required')
    ledger(plan['artifact_bindings'])
    need(config['prior_result']==plan['prior_result']==verify_history(),'Prior-result provenance differs')
    for path,digest in data['summary']['data_bindings'].items():
        if path in plan['artifact_bindings']:need(plan['artifact_bindings'][path]==digest,'Training/data audit binding differs')
    need(Path(plan['train_manifest']).resolve()==DATA/'v10_balanced/main_manifest.json'
         and Path(plan['schedule_file']).resolve()==DATA/'v10_balanced/schedule.json'
         and plan['fresh_manifests']=={'main':str(DATA/'v10_fresh/main_manifest.json')},'Run uses different data paths')
    binding=config['cache_binding'];need(binding==plan['cache_binding'] and Path(binding['file']).resolve()==CACHE,'Wrong V10 training cache')
    bind(binding['file'],binding['sha256'])
    need(all(config[k]==cache[k] for k in ('model','runtime','processor','native_dtypes')),'Native model/runtime/cache identity differs')
    need(config['pairing_file']==plan['pairing_file'] and config['pairing_sha256']==plan['pairing_sha256'],'Pairing binding differs')
    bind(plan['pairing_file'],plan['pairing_sha256']);saved_pairing=read(plan['pairing_file'])
    need(saved_pairing==pairing_value and objsha(saved_pairing)==plan['pairing_object_sha256'],'Saved pairing differs from metadata reconstruction')
    base=independent_pairs(data,cache)
    pairs=[dict(p,target_ids=cache['scenes'][p['sids'][0]]['target_ids'],global_feature_ids=cache['scenes'][p['sids'][0]]['global_feature_ids']) for p in base]
    need(pairs==saved_pairing['pairs'] and objsha(pairs)==saved_pairing['pairs_sha256'],'Independent augmented pair construction differs')
    order=expected_order(pairs,seed);bind(directory/'presentations.json',config['presentations_sha256'])
    need(read(directory/'presentations.json')==order and objsha(order)==config['order_sha256']==plan['order_sha256'][str(seed)]
         and len(order)==73440,'Paired presentation order differs')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);initial=ParallelLocalAggregation()
        expected_initial={k:tensor_info(v) for k,v in initial.state_dict().items()}
    need(sum(p.numel() for p in initial.parameters())==1041600 and config['initialized']==expected_initial
         and config['initialized_sha256']==objsha(expected_initial),'Seed initialization or parameter count differs')
    need(Path(summary['training_file']).resolve()==directory/'training.json','Unexpected training log path')
    bind(summary['training_file'],summary['training_sha256']);logs=read(summary['training_file'])
    need(len(logs)==4590,'Missing optimizer update records');total_targets=0;identity_presentations=0
    for step,row in enumerate(logs,1):
        positions=order[(step-1)*16:step*16];sids=[r['sid'] for r in positions]
        sequences=[cache['scenes'][sid]['target_ids'] for sid in sids];identity=[sids[i]==sids[i+1] for i in range(0,16,2)]
        need(row['step']==step and row['sids']==sids and row['pair_ids']==[r['pair_id'] for r in positions[::2]]
             and row['epochs']==[r['epoch'] for r in positions[::2]],'Actual pair/update order differs')
        need(all(positions[i]['pair_id']==positions[i+1]['pair_id'] and positions[i]['pair_side']==0
                 and positions[i+1]['pair_side']==1 for i in range(0,16,2)),'An update split or reversed a pair')
        need(close(row['lr'],learning_rate(step),atol=1e-12),'Optimizer schedule differs')
        check_loss_row(row,condition,sequences,identity);total_targets+=sum(map(len,sequences));identity_presentations+=sum(identity)
    need(total_targets==177120 and identity_presentations==2160,'Ragged target or weighted identity-pair totals differ')
    need(close(summary['training_seconds'],sum(r['seconds'] for r in logs),atol=1e-4),'Training time sum differs')
    need(logs[0]['consistency_loss']==logs[0]['path_loss']==logs[0]['weighted_consistency_loss']==0.,'Zero-initialized U must give zero first-step regularizers')
    need(Path(summary['first_gradients_file']).resolve()==directory/'first_gradients.json','Unexpected gradient ledger path')
    bind(summary['first_gradients_file'],summary['first_gradients_sha256']);gradients=read(summary['first_gradients_file'])
    need([r['step'] for r in gradients]==[1,2],'First two gradient records required')
    for row in gradients:
        need(set(row['pre_clip'])==set(expected_initial),'Gradient parameter coverage differs')
        for name,info in row['pre_clip'].items():
            need(info['shape']==expected_initial[name]['shape'] and info['dtype']=='torch.float32'
                 and isinstance(info['sha256'],str) and len(info['sha256'])==64,'Gradient tensor metadata differs')
    selection=read(directory/'selection.json');entries=selection['development']
    need([e['step'] for e in entries]==DEV_STEPS and summary['development']==entries,'All five registered dev checkpoints required')
    for entry in entries:
        step=entry['step'];devpath=directory/f'dev_{step}.json';need(Path(entry['dev_file']).resolve()==devpath,'Unexpected dev artifact')
        bind(devpath,entry['dev_sha256']);dev=audit_eval(devpath,data['dev'],'parallel',tokenizer)
        need(dev['label']==f'dev_{step}' and dev['exact_count']==entry['exact_count'] and close(dev['first_token_nll'],entry['nll']),'Rescored dev selection differs')
        checkpoint=Path(entry['checkpoint']).resolve();need(checkpoint==CKPT/runid/f'step_{step}.pt','Checkpoint outside V10 run directory')
        bind(checkpoint,entry['checkpoint_sha256']);saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        need(saved['step']==step and saved['config']==config and set(saved['branch'])==set(expected_initial),'Checkpoint step/config/parameters differ')
        need(all(t.dtype==torch.float32 and list(t.shape)==expected_initial[k]['shape'] and bool(torch.isfinite(t).all())
             for k,t in saved['branch'].items()),'Checkpoint dtype/shape/finite contract differs')
        need(objsha({k:tensor_info(t) for k,t in saved['branch'].items()})==entry['parameter_sha256'],'Checkpoint parameter hash differs')
    best=select(entries);need(selection['selected']==best and summary['selected']==best,'Checkpoint is not registered dev optimum')
    testpath=directory/'test.json';need(Path(summary['test_file']).resolve()==testpath,'Unexpected test artifact path')
    bind(testpath,summary['test_sha256']);test=audit_eval(testpath,data['test'],'parallel',tokenizer);need(test['label']=='test','Unexpected test label')
    return dict(condition=condition,arm='parallel',seed=seed,run_directory=str(directory),config_sha256=sha(directory/'config.json'),
        summary_sha256=sha(directory/'summary.json'),test_sha256=sha(testpath),selected_step=best['step'],
        selected_checkpoint=str(CKPT/runid/f"step_{best['step']}.pt"),selected_checkpoint_sha256=best['checkpoint_sha256'],selected_parameter_sha256=best['parameter_sha256'],
        initialized_sha256=config['initialized_sha256'],presentations_sha256=config['presentations_sha256'],pairing_object_sha256=plan['pairing_object_sha256'],
        training_seconds=summary['training_seconds'],evaluation_seconds=test['seconds'],clipping_fraction=sum(r['clipped'] for r in logs)/4590,
        loss_diagnostics=loss_summary(logs),first_preclip_gradient_hashes=gradients[0]['pre_clip'],
        native_test_model_forwards=sum(r['counters']['model'] for r in test['rows']),native_test_visual_forwards=272,
        verified_optimizer_log_rows=4590,verified_scene_presentations=73440,verified_pair_presentations=36720,
        verified_target_positions=177120,verified_saturated_identity_pair_presentations=2160,
        verified_dev_evaluations=5,verified_test_examples=272),test['rows'],config


def verify_release(config,data,frozen):
    binding=config['main_release'];bind(binding['file'],binding['sha256']);release=read(binding['file'])
    need(release['passed'] is True and release['plan_file']==config['plan_file'] and release['plan_sha256']==config['plan_sha256']
         and release['source_sha256']==config['source_sha256'],'Run uses a different or failed main release')
    need(release['per_main_seconds_cap']==2700 and release['main_block_gpu_seconds_cap']==16200,'V10 resource envelope differs')
    need(release['report_source_sha256']==frozen,'Main release did not freeze this independent report ancestry')
    ledger(release['report_source_sha256'])
    own=release['release_source_sha256']
    need(set(own)=={'scripts/release_native_vision_v10_mains.py','slurm/native_vision_v10_main_release.sbatch'},
         'Unexpected main-release source ledger')
    ledger(own)
    release_directory=Path(binding['file']).parent
    need(read(release_directory/'source_hashes.json')==own,'Main-release copied ledger differs')
    for name,digest in own.items():bind(release_directory/'source'/name.replace('/','_'),digest)
    ancestor=release['native_timing_ancestor'];bind(ancestor['file'],ancestor['sha256'])
    old=read(ancestor['file']);bind(ancestor['plan_file'],ancestor['plan_sha256'])
    oldplan=read(ancestor['plan_file']);newplan=read(config['plan_file'])
    need(old['passed'] is True and old['plan_file']==ancestor['plan_file'] and old['plan_sha256']==ancestor['plan_sha256'],
         'Native timing ancestor plan or acceptance differs')
    bind(ancestor['native_profile_file'],ancestor['native_profile_sha256'])
    need(ancestor['native_profile_file']==newplan['native_profile']['file']
         and ancestor['native_profile_sha256']==newplan['native_profile']['sha256']
         and Path(oldplan['native_profile']).resolve()==Path(newplan['native_profile']['directory']).resolve()
         and oldplan['native_profile_sha256']==ancestor['native_profile_sha256'],'Native timing profile differs')
    for name in ('scripts/native_vision_v7_runtime.py','gnnformer/parallel_local_native.py',
                 'gnnformer/parallel_local_aggregation.py','gnnformer/parallel_local_prompts.py'):
        need(old['source_sha256'][name]==config['source_sha256'][name]==sha(REPO/name),
             'Native computation changed since timing profile')
    extension=old['resource_extension'];ledger(extension['source_sha256'])
    bind(extension['original_release_file'],extension['original_release_sha256'])
    original=read(extension['original_release_file'])
    need(extension['scope']=='resource_only_1800_to_2700_seconds' and extension['original_passed'] is False
         and original['passed'] is False and extension['no_model_data_optimizer_selection_or_decision_change'] is True,
         'Prior failed timing release was not explicitly preserved')
    times=old['projections']['parallel']['generation_seconds']
    need(old['projections']['parallel']['original_1800_second_passed'] is False
         and times==original['projections']['parallel']['generation_seconds'],'Native timing failure lineage differs')
    oldprofile_binding=old['profiles']['parallel'];oldprofile_path=Path(oldprofile_binding['directory'])/'summary.json'
    bind(oldprofile_path,oldprofile_binding['summary_sha256']);oldprofile=read(oldprofile_path)
    checked_binding=release['data_release'];bind(checked_binding['file'],checked_binding['sha256']);checked=read(checked_binding['file'])
    need(checked['passed'] is True and checked['source_sha256']==frozen
         and checked['data_bindings']==data['summary']['data_bindings'],'Independent data/report release differs')
    bind(checked['audit_file'],checked['audit_sha256'])
    need(set(release['profiles'])==set(CONDITIONS) and set(release['projections'])==set(CONDITIONS),
         'Both fixed software profiles/projections are required')
    shared_max_step=max(read(Path(item['directory'])/'summary.json')['step_seconds_max_steady'] for item in release['profiles'].values())
    for condition,item in release['profiles'].items():
        path=Path(item['directory'])/'summary.json';bind(path,item['summary_sha256']);profile=read(path)
        need(profile['profile'] is True and profile['condition']==condition and profile['seed']==14 and profile['arm']=='parallel'
             and profile['passed'] is True and profile['computational_integrity_passed'] is True
             and profile['steps']==32 and profile['no_dev_or_test_evaluation'] is True
             and profile['plan_sha256']==config['plan_sha256'] and profile['source_sha256']==config['source_sha256']
             and profile['active_cache_replay']['passed'] is True
             and profile['active_cache_replay']['counts_covered']==[0,9,10,16]
             and profile['active_cache_replay']['prefixes_checked']==10,'Matched training software profile failed or changed')
        zero=profile['zero_initialization_auxiliary_gradients']
        need(zero['passed'] is True and zero['residual_loss']==zero['path_loss']==0.
             and zero['up_exactly_zero'] is True and zero['all_residual_gradients_zero'] is True
             and zero['all_path_gradients_zero'] is True,'Profile zero-U auxiliary losses/gradients differ')
        projection=release['projections'][condition]
        training=4590*shared_max_step;development=320*times['16'];test=272*times['64']
        projected=profile['model_and_features_load_seconds']+1.25*(training+development+test)+120
        need(profile['hardware']==oldprofile['hardware'] and projection['generation_seconds']==times
             and projection['N32_uses_N64_bound'] is True and close(projection['training_seconds'],training)
             and close(projection['development_seconds'],development) and close(projection['test_seconds'],test)
             and close(projection['projected_seconds'],projected),'Independent measured runtime arithmetic differs')
        need(projection['passed'] is True and math.isfinite(projection['projected_seconds'])
             and 0<projection['projected_seconds']<=2700,'Prospective runtime criterion failed')
    return dict(file=binding['file'],sha256=binding['sha256'],data_release=checked_binding,projections=release['projections'],
                release_source_sha256=own,native_timing_ancestor=ancestor,retained_original_native_timing_failure=True)



def self_test():
    import torch
    import numpy as np
    good=torch.tensor([0.,1.,-7.5],dtype=torch.float16).float()
    need(stored_native_logits(good) and not stored_native_logits(good.half()) and not stored_native_logits(good+1e-6)
         and not stored_native_logits(torch.tensor([float('nan')])),'Native lossless FP16 promotion check failed')
    need(parse_text(' 12\n')==12 and parse_text('00')==0 and all(parse_text(t) is None for t in ('-1','1.','1 2','八','')),'Strict ASCII integer parsing failed')
    def score(n32,n64,larger):return dict(cells={f'main_test_N{n}':dict(correct=k) for n,k in ((32,n32),(64,n64))},partitions={'N64':{'K9_16':dict(correct=larger)}})
    reference=score(129,102,0);boundary=score(123,109,52)
    need(integer_decision(boundary,reference)['practical'],'Inclusive practical thresholds failed')
    for value in (score(122,109,52),score(123,108,52),score(123,109,51)):
        need(integer_decision(value,score(120,100,0))['primary']
             and not integer_decision(value,score(120,100,0))['practical'],'Absolute practical boundary ignored despite primary success')
    need(not integer_decision(boundary,score(129,103,0))['primary'] and not integer_decision(boundary,score(130,102,0))['primary'],'Integer primary boundary ignored')
    sequences=[ids[:] for j in range(8) for ids in ([[1,151645]]*2 if j%2==0 else [[1,2,151645]]*2)]
    layout=independent_layout(sequences);identity=[i==7 for i in range(8)];p=layout['pair_offsets'][-1]
    ce=[float(i%3+1) for i in range(layout['scene_offsets'][-1])]
    residual=[.1]*(p-3)+[0.]*3;path=[.2]*(p-3)+[0.]*3
    per_scene=[sum(ce[a:b])/(b-a) for a,b in zip(layout['scene_offsets'][:-1],layout['scene_offsets'][1:])]
    per_res=[sum(residual[a:b])/(b-a) for a,b in zip(layout['pair_offsets'][:-1],layout['pair_offsets'][1:])]
    per_path=[sum(path[a:b])/(b-a) for a,b in zip(layout['pair_offsets'][:-1],layout['pair_offsets'][1:])]
    def example(condition):
        coefficient=0. if condition=='ce' else 1.;cons=sum(per_res)/8;ce_mean=sum(per_scene)/16
        return dict({k:v for k,v in layout.items() if k!='pair_offsets'},consistency_coefficient=coefficient,
            loss=ce_mean+coefficient*cons,ce_loss=ce_mean,consistency_loss=cons,path_loss=sum(per_path)/8,
            weighted_consistency_loss=coefficient*cons,gradient_norm=1.5,clipped=True,seconds=.01,
            saturated_identity_pairs=identity,per_scene_ce=per_scene,per_pair_consistency=per_res,per_pair_path=per_path,
            position_losses=dict(ce=ce,consistency=residual,path=path),pair_position_denominator=[10.]*p,
            residual_difference_norms=[math.sqrt(10*x) for x in residual],B_norms=[math.sqrt(10*x) for x in path])
    for c in CONDITIONS:check_loss_row(example(c),c,sequences,identity)
    mutations=[dict(loss=2.),dict(consistency_coefficient=0.),dict(clipped=False),dict(ce_loss=sum(ce)/len(ce)),
        dict(prefix_ids=[[999]]*len(ce)),dict(scene_lengths=[2]*16),dict(saturated_identity_pairs=[False]*8)]
    for change in mutations:
        value=json.loads(json.dumps(example('consistency')));value.update(change)
        try:check_loss_row(value,'consistency',sequences,identity)
        except ValueError:pass
        else:raise AssertionError('Malformed ragged objective was accepted: '+str(change))
    wrong=json.loads(json.dumps(example('consistency')));wrong['position_losses']['consistency'][-1]=.1
    try:check_loss_row(wrong,'consistency',sequences,identity)
    except ValueError:pass
    else:raise AssertionError('Nonzero saturated identity penalty was accepted')
    pairs=[dict(pair_id=str(i),question=f'q{i}',gold=i%17,sids=[f'a{i}',f'b{i}'],pair_kind='cross_length',n_frames=[8,16],replicas=[0,0]) for i in range(918)]
    order=expected_order(pairs,14)
    need(order==expected_order(pairs,14) and order!=expected_order(pairs,15) and len(order)==73440
         and Counter(r['epoch'] for r in order[1824:1840])=={1:12,2:4}
         and all(order[i]['pair_side']==0 and order[i+1]['pair_side']==1 and order[i]['pair_id']==order[i+1]['pair_id'] for i in range(0,len(order),2)),
         'Persistent complete-pair order or carried boundary failed')
    fixture={}
    for condition in CONDITIONS:
        for seed in SEEDS:
            fixture[condition,seed]=[dict(cell=f'main_test_N{n}',gold=k,anchor_id=f'{k}_{i}',n_frames=n,exact=condition=='consistency')
                for k in range(17) for i in range(8) for n in (32,64)]
    intervals=bootstrap(fixture,replicates=100)
    need(intervals==bootstrap(fixture,replicates=100),'Shared family bootstrap is not deterministic')
    need(all(np.allclose(endpoint['pooled_fixed_two_seeds']['difference_ci95_pp'],[100,100])
         for group in intervals['results'].values() for endpoint in group.values()),'Paired family bootstrap lost a fixed difference')
    entries=[dict(step=918,exact_count=60,nll=.2),dict(step=1836,exact_count=61,nll=9.),dict(step=2754,exact_count=61,nll=.1),dict(step=3672,exact_count=61,nll=.1)]
    need(select(entries)==entries[2],'Whole-answer exact then raw first-token NLL then earliest selection differs')
    need(close(learning_rate(50),.001,atol=1e-12) and close(learning_rate(4590),.00001,atol=1e-12),'Optimizer endpoint differs')
    return ['lossless_native_FP16_promotion','strict_ASCII_integer_parser','inclusive_per_seed_primary_and_practical_boundaries',
        'ragged_per_scene_CE_and_per_pair_regularizer_reductions','strict_prefix_offsets','coefficient0_and1',
        'reject_malformed_ragged_logs','exact_zero_saturated_identity_regularizer','persistent_paired_order_and_carried_epoch_boundary',
        'shared_stratified_family_bootstrap_all_subgroups','registered_optimizer_endpoints','complete_answer_dev_selection_with_first_token_tiebreak']


def get_pairing(data,cache):
    from scripts import native_vision_v10_pairs as pairing
    value=pairing.load_pairs();base=independent_pairs(data,cache)
    expected=[dict(p,target_ids=cache['scenes'][p['sids'][0]]['target_ids'],global_feature_ids=cache['scenes'][p['sids'][0]]['global_feature_ids']) for p in base]
    need(value['pairs']==expected and value['pairs_sha256']==objsha(expected) and value['pair_count']==918
         and value['weighted_scenes_per_epoch']==1836 and value['unique_training_scenes']==1782
         and value['target_positions_per_epoch']==4428 and value['epochs']==40 and value['seeds']==[14,15],
         'Independent augmented pairing/target count differs')
    return value


def report_runs(directories,out,frozen):
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(4);tests=self_test();data=audit_data()
    tokenizer=AutoTokenizer.from_pretrained(str(MODEL),use_fast=False,local_files_only=True)
    cache,cache_audit=verify_cache(CACHE,data,tokenizer);pairing_value=get_pairing(data,cache)
    save(out/'data_audit.json',dict(data['summary'],cache_audit=cache_audit));save(out/'pairing.json',pairing_value)
    audits={};rows={};configs={};releases=[]
    need(len(directories)==4 and len({Path(p).resolve() for p in directories})==4,'All four distinct V10 main runs required')
    for directory in directories:
        audited,predictions,config=verify_run(directory,data,tokenizer,cache,pairing_value)
        key=audited['condition'],audited['seed'];need(key not in audits,'Duplicate condition/seed')
        audits[key],rows[key],configs[key]=audited,predictions,config;releases.append(verify_release(config,data,frozen))
        print(json.dumps(dict(audited_run=str(directory),native_examples=len(predictions))),flush=True)
    need(set(audits)=={(c,s) for c in CONDITIONS for s in SEEDS},'Missing registered condition/seed')
    need(all(item==releases[0] for item in releases),'Mains used different report/data/resource releases')
    gradients={}
    for seed in SEEDS:
        a,b=configs['ce',seed],configs['consistency',seed]
        need(a['initialized_sha256']==b['initialized_sha256'] and a['presentations_sha256']==b['presentations_sha256']
             and a['order_sha256']==b['order_sha256'],'Matched condition initialization/order differs')
        need(all(a[k]==b[k] for k in ('policy','plan_file','plan_sha256','source_sha256','cache_binding','pairing_sha256',
             'model','runtime','processor','native_dtypes','prior_result')),'Matched treatment conditions differ')
        gradients[str(seed)]=dict(first_preclip_gradient_hashes_equal=audits['ce',seed]['first_preclip_gradient_hashes']==audits['consistency',seed]['first_preclip_gradient_hashes'],
            scope='Descriptive recorded hashes; first-step zero-U penalties independently verified')
    runs={f'{c}_s{s}':dict(audits[c,s],metrics=summarize(rows[c,s])) for c in CONDITIONS for s in SEEDS}
    decisions={str(s):integer_decision(runs[f'consistency_s{s}']['metrics'],runs[f'ce_s{s}']['metrics']) for s in SEEDS}
    primary=all(d['primary'] for d in decisions.values());practical=all(d['practical'] for d in decisions.values());intervals=bootstrap(rows)
    analysis=dict(schema_version=1,passed=True,audit_passed=True,source_sha256=frozen,policy=POLICY,data=data['summary'],cache=cache_audit,
        pairing=dict(file=str(out/'pairing.json'),sha256=sha(out/'pairing.json'),object_sha256=objsha(pairing_value),pair_count=918,
            unique_training_scenes=1782,weighted_scenes_per_epoch=1836,target_positions_per_epoch=4428),
        release=releases[0],self_tests=tests,runs=runs,decisions=decisions,first_gradient_comparison=gradients,
        primary_both_seeds=primary,practical_both_seeds=practical,vision_milestone_gate=primary and practical,
        post_checkpoint_verification='Separate mandatory selected-checkpoint native audit; not performed by this efficacy report',
        pooled_fixed_two_seed_n64_difference_pp=sum(d['n64_difference_pp'] for d in decisions.values())/2,bootstrap=intervals,
        interpretation=dict(treatment='CE versus CE plus native residual consistency, on matched data and unchanged native parallel aggregation',
            unchanged_inference_operator=True,additional_inference_work=False,no_attention_operator_novelty_claim=True,
            does_not_establish_reasoning_composition=True,all_final_answer_labels_used_for_pairing=True,
            all_test_answer_values_supported_in_training=True,no_unseen_answer_value_claim=True,
            training_length_count_correlation='K0..8 cross-length consistency; K9..15 same-length consistency; K16 identity with zero regularizer',
            primary_exact='Complete stripped ASCII integer equals gold AND native EOS within four tokens; malformed/truncated stay in denominators',
            native_first_token_nll='Raw first-token NLL used only for dev tie-break; shared first digit does not distinguish10..16',
            raw_logit_storage='FP32 archives verified exact lossless promotions of FP16 native logits',
            matched_control='Identical paired scene order, targets, initialization,73440scene/36720pair/177120target presentations and4590updates',
            bootstrap='K-stratified intact family intervals condition on these two fitted seeds; pooling cannot rescue per-seed failure',
            path_penalty='Logged diagnostic only; not optimized',local_visual_atoms_may_recur=True,
            loss_audit='All logged causal-position/scene/pair arithmetic; unavailable intermediate model states are not reconstructed'))
    save(out/'analysis.json',analysis)
    lines=['# V10 independent native evaluation','',f'Audit passed. Both-seed primary: **{primary}**. Both-seed practical: **{practical}**.','',
        'All test answer values0–16 occur in adaptation. Training length is at most16; fresh test families extend from32 to64frames. The two conditions use identical native inference and differ only by residual-consistency loss.','',
        '| Condition | Seed | N32 | N64 | N64 K0–8 | N64 K9–15 | N64 K16 | N64 K9–16 | Selected step |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    fmt=lambda v:f"{v['correct']}/{v['n']} ({100*v['accuracy']:.1f}%)"
    for c in CONDITIONS:
        for s in SEEDS:
            run=runs[f'{c}_s{s}'];m=run['metrics'];values=[m['cells'][f'main_test_N{n}'] for n in (32,64)]+[m['partitions']['N64'][k] for k in ('K0_8','K9_15','K16','K9_16')]
            lines.append(f'| {c} | {s} | '+' | '.join(fmt(v) for v in values)+f" | {run['selected_step']} |")
    lines+=['','| Seed | N64 gain | Family bootstrap95%CI | N32 change | Primary | Practical |','|---:|---:|---:|---:|---|---|']
    for s in SEEDS:
        d=decisions[str(s)];ci=intervals['results']['all']['N64'][str(s)]['difference_ci95_pp']
        lines.append(f"| {s} | {d['n64_difference_pp']:.2f}pp | [{ci[0]:.2f},{ci[1]:.2f}]pp | {d['n32_difference_pp']:.2f}pp | {d['primary']} | {d['practical']} |")
    lines+=['','Exact requires the complete correct integer plus native EOS within four tokens. All272 examples per run remain in the denominator.',
        '', 'Primary requires at least7/136 additional N64 answers and no more than6/136 fewer N32 answers in each seed. Practical additionally requires consistency N32≥123/136, N64≥109/136, and N64 K9–16≥52/64.',
        '', 'Only K0–8 has cross-length pair supervision. K9–15 has same-length scene pairs;54saturated K16pairs duplicate one SID and contribute exactly zero regularizer while retaining both CE slots. Historical saturated reuse is explicitly recorded.',
        '', 'Independent checks bind25prior exclusions,1,782unique training contexts,918pairs,all177,120valid causal training targets,the shared[1]prefix,complete log reductions,all five dev checkpoints,and raw full-vocabulary native argmax/EOS/NLL.',
        '', 'Final acceptance also requires the separate all-selected-checkpoint audit. This comparison does not establish unseen-answer transfer, reasoning composition, or a new attention operator.',
        '', '[Every K/length cell, parse/EOS/first-token statistics, loss diagnostics and provenance](analysis.json).']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return dict(passed=True,source_sha256=frozen,analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),
        report_file=str(out/'REPORT.md'),primary_both_seeds=primary,practical_both_seeds=practical,vision_milestone_gate=primary and practical)


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-test',action='store_true');mode.add_argument('--check-data',action='store_true');mode.add_argument('--check-manifests',action='store_true')
    mode.add_argument('--runs','--runs4',nargs=4,type=Path);parser.add_argument('--output',type=Path);args=parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),
         'All reporter/data/self-test executions require CPU Slurm')
    started=time.perf_counter();label='self_test' if args.self_test else 'manifest_check' if args.check_manifests else 'data_check' if args.check_data else 'report'
    out=args.output or OUT/f'{label}_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False);frozen=snapshot(out)
    if args.self_test:summary=dict(passed=True,source_sha256=frozen,tests=self_test())
    elif args.check_data or args.check_manifests:
        import torch
        torch.set_num_threads(4);data=audit_data();audited=data['summary'];cache_audit=None
        if args.check_data:
            from transformers import AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained(str(MODEL),use_fast=False,local_files_only=True)
            cache,cache_audit=verify_cache(CACHE,data,tokenizer);value=get_pairing(data,cache);save(out/'pairing.json',value)
            audited=dict(audited,cache_audit=cache_audit,pairing_object_sha256=objsha(value))
        save(out/'data_audit.json',audited)
        summary=dict(passed=True,source_sha256=frozen,data_bindings=data['summary']['data_bindings'],cache_audit=cache_audit,
            audit_file=str(out/'data_audit.json'),audit_sha256=sha(out/'data_audit.json'),manifest_only=args.check_manifests)
        if args.check_data:summary.update(pairing_file=str(out/'pairing.json'),pairing_sha256=sha(out/'pairing.json'),pairing_object_sha256=objsha(value))
    else:summary=report_runs(args.runs,out,frozen)
    need(sources()==frozen,'Reporter source changed during execution');summary.update(seconds=time.perf_counter()-started,slurm_job_id=os.environ['SLURM_JOB_ID'])
    save(out/'summary.json',summary);(out/'INDEX.md').write_text('# V10 independent CPU audit\n\n[Summary](summary.json) · [Sources](source_hashes.json)\n')
    print(json.dumps(dict(passed=True,summary_file=str(out/'summary.json'),seconds=summary['seconds'])),flush=True)


if __name__=='__main__':main()

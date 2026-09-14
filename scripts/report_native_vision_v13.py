"""Independent V13 CPU audit of learned null centering and native answers.

The only treatment is cardinality scaling of the same learned query offset.
Every ordinary/native target and ordered auxiliary occurrence is source-bound.
Recorded losses are audited; unsaved intermediate model states are not inferred.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import report_native_vision_v9 as prior
from scripts import report_native_vision_v10 as v10
from scripts import report_native_vision_v11 as fresh_audit
from scripts import stage_native_vision_v10_data as stage
from scripts import stage_native_vision_v11_test as fresh_stage
from scripts import cache_native_vision_v13_null_features as worker
from scripts import merge_native_vision_v13_precision as precision
need,read,sha,objsha,save,close,bind,ledger,tensor_info=(prior.need,prior.read,prior.sha,prior.objsha,prior.save,prior.close,prior.bind,prior.ledger,prior.tensor_info)
verify_source,select,stored_native_logits=prior.verify_source,prior.select,prior.stored_native_logits
DATA,MODEL=prior.DATA,prior.MODEL
OUT=REPO/'outputs/native_aggregation_vlm/v13'
CKPT=Path('/mnt/ckpts/gabriele/gnn_transformer/native_aggregation_vlm_v13')
CACHE=DATA/'v13_null_features/feature_cache.json'
DEV_STEPS=[918,1836,2754,3672,4590]
CONDITIONS=('centered','offset')
SEEDS=(16,17)
PARTITIONS=(('K0_8',tuple(range(9))),('K9_15',tuple(range(9,16))),('K16',(16,)),('K9_16',tuple(range(9,17))),
    ('single_digit_K0_9',tuple(range(10))),('two_digit_K10_16',tuple(range(10,17))),
    ('shared_first_digit_K1_10_16',(1,*range(10,17))),('nonzero',tuple(range(1,17))))
OWN=tuple(dict.fromkeys(('scripts/report_native_vision_v13.py','slurm/native_vision_v13_report.sbatch',
    'scripts/report_native_vision_v11.py','scripts/train_native_vision_v13.py',
    'gnnformer/conditional_null_mean.py','gnnformer/parallel_local_learned_null.py',
    'scripts/native_vision_v13_runtime.py','scripts/profile_native_vision_v13_null.py',
    'tests/test_conditional_null_mean.py',
    'slurm/native_vision_v13_train_check.sbatch','slurm/native_vision_v13_train_profile.sbatch','slurm/native_vision_v13_train.sbatch',
    *v10.OWN,*fresh_stage.OWN,*worker.OWN)))
POLICY=dict(
    protocol='v13_learned_null_centering',native_arm='parallel',conditions=['centered','offset'],seeds=[16,17],epochs=40,
    pair_slots_per_epoch=918,scene_slots_per_epoch=1836,unique_training_scenes=1782,
    scene_presentations=73440,pair_presentations=36720,target_positions=177120,
    batch_size=16,pairs_per_batch=8,steps=4590,dev_steps=[918,1836,2754,3672,4590],dev_examples=64,test_examples=272,
    lr=.001,warmup=50,final_lr=.00001,weight_decay=0.,clip_norm=1.,
    optimizer='one_AdamW_two_parameter_groups_core_and_predictor',clipping='independent_norm1_core_and_predictor',
    rank=96,hidden_size=3584,core_parameters=1041600,predictor_parameters=18624,parameters=1060224,
    merge='sum',post_activation='silu',native_dtype='torch.float16',branch_dtype='torch.float32',predictor_dtype='torch.float32',
    correction_coefficients={'centered':'actual_scene_N','offset':1},
    predictor_input='continuous_live_q_equals_Wq_RMS_g_no_N_ID_or_prefix_lookup',
    predictor_initialization='core_then_predictor_from_same_seed_second_weight_and_bias_exact_zero',
    consistency_coefficients={'centered':1.,'offset':1.},consistency_epsilon=1e-6,
    ce_reduction='mean_over_scenes_of_mean_over_valid_native_target_positions',
    consistency_reduction='mean_over_pairs_of_mean_over_corresponding_strict_prefix_positions',
    denominator='detached_squared_L2_of_identical_frozen_global_state_plus_fixed_epsilon',
    pair_order='persistent_random.Random(seed)_fresh_canonical_pair_permutation_each_epoch',
    within_pair_order='registered_pair_sides_0_then_1',
    saturation_rule='retain54_N16_K16_same_SID_pairs_with_exact_zero_regularizer',
    path_loss_role='diagnostic_only_on_corrected_aggregate_no_loss_weight',
    auxiliary_groups=972,auxiliary_batch_size=16,auxiliary_presentations=73440,auxiliary_coefficient=1.,
    auxiliary_seed_offset=20261105,auxiliary_order='persistent_random.Random(seed+20261105)_sorted_group_permutation_each_cycle_carried_batches',
    auxiliary_target='no_grad_core_messages_FP64_mean_over24_occurrences_cast_FP32',
    auxiliary_denominator='detached_FP32_message_squared_mean_over24_times96_plus1e-6',
    auxiliary_reduction='mean16_of_mean96_squared_prediction_error_divided_by_group_denominator',
    auxiliary_query_and_target_detached=True,main_predictor_and_query_fully_differentiable=True,
    gradient_isolation_steps=[1,2,32],
    full_answer_targets='canonical_native_numeral_tokens_plus_exactly_one_terminal_EOS',
    max_new_tokens=4,exact_requires_eos=True,reject_nonterminal_special_tokens=True,
    selection='max_dev_exact_then_min_raw_first_token_NLL_then_earliest',native_eos=[151645,151643],target_eos=151645,
    profile_steps=32,training_count_support=list(range(17)),development_count_support=list(range(16)),
    test_count_support=list(range(17)),maximum_training_N=16,test_N=[32,64],
    bootstrap_seed=20261106,bootstrap_replicates=10000,
    per_main_seconds_cap=2700,campaign_gpu_seconds_cap=16200,profile_seconds_cap=300,
)

PROFILE_CALLS=dict(training_updates=32,native_prefill_forwards=10,visual_forwards=10,
    same_state_native_shaped_head_replays=10,cached_state_head_projections=10,standalone_head_calls=20,natural_generations=0)

def sources():return {**{name:sha(REPO/name) for name in OWN},**precision.sources()}

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

def audit_data():
    # Only the frozen data function is reused: no V11 feature or memory methods.
    return fresh_audit.audit_data()


def summarize(rows):
    need(len(rows)==272 and Counter((r['n_frames'],r['gold']) for r in rows)==Counter({(n,k):8 for n in (32,64) for k in range(17)}),'Registered test denominators differ')
    cells={f'main_test_N{n}':metrics([r for r in rows if r['n_frames']==n]) for n in (32,64)}
    by_k={f'main_test_N{n}/K{k}':metrics([r for r in rows if r['n_frames']==n and r['gold']==k]) for n in (32,64) for k in range(17)}
    partitions={}
    for n in (32,64):
        partitions[f'N{n}']={label:metrics([r for r in rows if r['n_frames']==n and r['gold'] in ks])
            for label,ks in PARTITIONS}
    return dict(all=metrics(rows),cells=cells,by_n_k=by_k,partitions=partitions)

def integer_decision(treated,control):
    n32=treated['cells']['main_test_N32']['correct'];n64=treated['cells']['main_test_N64']['correct']
    delta=n64-control['cells']['main_test_N64']['correct'];delta32=n32-control['cells']['main_test_N32']['correct']
    larger=treated['partitions']['N64']['K9_16']['correct'];primary=delta>=7 and delta32>=-6
    practical=primary and n32>=123 and n64>=109 and larger>=52
    return dict(n64_additional_correct=delta,n64_difference_pp=100*delta/136,n32_additional_correct=delta32,n32_difference_pp=100*delta32/136,
        n64_gain_at_least_5pp=delta>=7,n32_loss_at_most_5pp=delta32>=-6,primary=primary,
        centered_n32_correct=n32,centered_n32_at_least_90pct=n32>=123,
        centered_n64_correct=n64,centered_n64_at_least_80pct=n64>=109,
        centered_larger_k_n64_correct=larger,centered_larger_k_n64_at_least_80pct=larger>=52,practical=practical)

def bootstrap(all_rows,replicates=10000,seed=20261106):
    import numpy as np
    rng=np.random.default_rng(seed);reference=all_rows['centered',16];groups=defaultdict(dict)
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
    for label,ks in (('all',tuple(range(17))),*PARTITIONS):
        selected=np.concatenate([np.arange(k*8,(k+1)*8) for k in ks]);draws=indices[:,selected];endpoints={}
        for n,column in ((32,0),(64,1)):
            samples={key:value[:,column][draws].mean(1) for key,value in arrays.items()};cell={}
            for fit_seed in SEEDS:
                a,b=samples['centered',fit_seed],samples['offset',fit_seed]
                cell[str(fit_seed)]=dict(difference_ci95_pp=(100*np.quantile(a-b,[.025,.975])).tolist(),
                    centered_ci95_pp=(100*np.quantile(a,[.025,.975])).tolist(),offset_ci95_pp=(100*np.quantile(b,[.025,.975])).tolist())
            pooled=sum(samples['centered',s]-samples['offset',s] for s in SEEDS)/2
            cell['pooled_fixed_two_seeds']=dict(difference_ci95_pp=(100*np.quantile(pooled,[.025,.975])).tolist());endpoints[f'N{n}']=cell
        result[label]=endpoints
    return dict(replicates=replicates,seed=seed,unit='Complete N32-to-N64 family stratified by K; shared draws across all lengths, conditions, subgroups and fixed seeds',
        inference_scope='Conditional example uncertainty for these fitted seeds, not a training-seed population',results=result)

def parse_answer(ids,tokenizer):
    completed=ids[-1] in (151645,151643);bodyids=ids[:-1] if completed else ids
    clean=not any(t in set(tokenizer.all_special_ids) for t in bodyids)
    body=tokenizer.decode(bodyids,skip_special_tokens=False)
    parsed=int(body.strip()) if clean and re.fullmatch('[0-9]+',body.strip()) else None
    return dict(completed=completed,clean=clean,body=body,parsed=parsed)


def audit_eval(path,expected,arm,tokenizer,condition):
    import torch
    value=read(path);bind(value['raw_file'],value['raw_sha256'])
    need(value['raw_dtype']=='torch.float32','Native generation archive dtype metadata differs')
    blob=torch.load(value['raw_file'],map_location='cpu',weights_only=True);raw=blob['raw_logits']
    rows=value['rows'];need(blob['schema_version']==1 and len(rows)==len(raw)==len(expected)==value['n'],'Evaluation coverage differs')
    vocab=json.loads((MODEL/'config.json').read_text()).get('text_config',{}).get('vocab_size')
    if vocab is None: vocab=json.loads((MODEL/'config.json').read_text())['vocab_size']
    from gnnformer.parallel_local_prompts import build_set_count_prompt
    from gnnformer.data import build_count_prompt
    for index,((cell,sample),row,x) in enumerate(zip(expected,rows,raw)):
        need(row['raw_index']==index and row['cell']==cell,'Evaluation ordering/index differs')
        need(all(row[k]==sample[k] for k in ('sid','gold','n_frames','content_sha256'))
             and row['anchor_id']==sample.get('anchor_id'),'Evaluation sample identity differs')
        ids=row['generated_ids'];steps=len(ids)
        need(1<=steps<=4 and all(type(i) is int for i in ids) and x.shape==(steps,vocab) and stored_native_logits(x),'Malformed native raw logits')
        need(x.argmax(-1).tolist()==ids,'Output is not native raw greedy argmax')
        need(row['raw_text']==tokenizer.decode(ids,skip_special_tokens=False)
             and row['text']==tokenizer.decode(ids,skip_special_tokens=True),'Decoded text differs')
        completed=ids[-1] in (151645,151643)
        need(not any(t in (151645,151643) for t in ids[:-1]) and (completed or steps==4),'Invalid native stop sequence')
        parsed_answer=parse_answer(ids,tokenizer);clean=parsed_answer['clean'];body=parsed_answer['body'];parsed=parsed_answer['parsed']
        count_correct=parsed==sample['gold'];exact=count_correct and completed
        need(row['answer_body']==body and row['no_nonterminal_special_tokens']==clean
            and row['pair_id']==sample.get('pair_id'),'Native full-body/special-token/family parser differs')
        need(row['prediction']==parsed and row['parseable']==(parsed is not None)
             and row['parsed_count_correct']==count_correct and row['exact']==exact
             and row['completed']==completed and row['truncated']==(not completed),'Parser/EOS/exact accounting differs')
        token=tokenizer(str(sample['gold']),add_special_tokens=False)['input_ids'][0]
        nll=float(torch.logsumexp(x[0].float(),-1)-x[0,token].float())
        need(close(nll,row['first_token_nll']),'Raw first-token NLL differs')
        counters=dict(model=steps,visual=1,language=steps,fusion=steps,broadcast=steps if arm=='parallel' else 0)
        need(row['counters']==counters,'Native call counters differ')
        if 'first_token_correct' in row:need(row['first_token_correct']==(ids[0]==token),'Stored first-token correctness differs')
        row['first_token_correct']=ids[0]==token
        meta=row['metadata'];n=sample['n_frames'];r=n+1 if arm=='parallel' else 1
        need(condition in CONDITIONS and arm=='parallel' and meta['null_mode']==condition
            and meta['read_boundary']=='before_native_final_norm' and meta['null_predictor_parameters']==18624,
            'Native learned-null placement/mode/parameter count differs')
        need(meta['arm']==arm and meta['sid']==sample['sid'] and meta['n_frames']==n
             and meta['question']==sample['question'] and meta['question_sha256']==objsha(sample['question'])
             and meta['global_prompt']==build_set_count_prompt(sample['question'])
             and meta['local_prompt']==(build_count_prompt(sample['question'],1) if arm=='parallel' else None),'Native prompt identity differs')
        need(meta['image_sha256']==[im['sha256'] for im in sample['image_files']]
             and meta['image_paths']==[im['path'] for im in sample['image_files']]
             and meta['prefix_ids']==[] and meta['resize']==392,'Evaluation evidence/prefix differs')
        need(meta['row_count']==r and meta['global_row']==r-1 and meta['local_elements']==(n if arm=='parallel' else 1)
             and meta['row_kinds']==(['local']*n+['global'] if arm=='parallel' else ['joint']), 'Native row layout differs')
        width=meta['prompt_width']
        need(width==meta['original_prompt_width']==max(meta['row_prompt_tokens']) and len(meta['row_prompt_tokens'])==r,'Prompt widths differ')
        need(meta['input_identity']['input_ids']['shape']==[r,width]
             and meta['input_identity']['attention_mask']['shape']==[r,width]
             and meta['input_identity']['image_grid_thw']['shape']==[n,3]
             and meta['layout']['every_unpadded_row_exact'] is True,'Native layout audit differs')
        need(meta['native_dtype']=='torch.float16' and meta['branch_dtype']=='torch.float32','Native dtype differs')
        policy=meta['generation']
        need(policy['max_new_tokens']==4 and policy['do_sample'] is False and policy['num_beams']==1
             and policy['repetition_penalty']==1 and policy['native_eos_token_ids']==[151645,151643]
             and policy['target_eos_token_id']==151645 and policy['vocabulary_mask'] is False
             and policy['other_logits_processors'] is False,'Greedy generation policy differs')
        positions=meta['generation_position_ids']
        need(len(positions)==steps and all(p['shape']==[4,r,width if j==0 else 1] for j,p in enumerate(positions)), 'Cached native position shapes differ')
    need(value['exact_count']==sum(r['exact'] for r in rows)
         and close(value['first_token_nll'],sum(r['first_token_nll'] for r in rows)/len(rows)),'Evaluation summary differs')
    return value


def expected_auxiliary_order(groups,seed):
    need(seed in SEEDS and len(groups)==972,'Unregistered auxiliary stream')
    canonical=sorted(groups);rng=random.Random(seed+20261105);result=[];cycle=0
    while len(result)<73440:
        cycle+=1;permutation=list(range(972));rng.shuffle(permutation)
        for slot in permutation:
            result.append(dict(cycle=cycle,slot=slot,group_id=canonical[slot]))
            if len(result)==73440:break
    need(Counter(r['cycle'] for r in result)==Counter({**{c:972 for c in range(1,76)},76:540}),
        'Auxiliary occurrence-cycle weights differ')
    return result


def verify_cache(path,data,tokenizer):
    union=precision.verify_cache(path,verify_tensors=True)
    parent,parent_audit=v10.verify_cache(Path(union['parent_cache']['file']),data,tokenizer)
    plan=read(union['parent_plan']['file']);newplan=worker.verify_plan(union['plan_file']);features=dict(plan['features'])
    for fid,descriptor in newplan['features'].items():
        need(fid not in features or features[fid]==descriptor,'New cache changed an old descriptor');features[fid]=descriptor
    need(set(features)==set(union['features']) and union['scenes']==parent['scenes'] and len(features)==53322,
        'Union descriptor or unchanged ordinary scene coverage differs')
    prefixes=sorted({tuple(tokens[:i]) for tokens in parent['target_token_ids'].values() for i in range(len(tokens))},key=lambda p:(len(p),p))
    need(len(prefixes)==18 and all(151645 not in p for p in prefixes),'Null views must have every strict prefix and no target EOS')
    questions=sorted({row['question'] for row in data['train'].values()});banks={};groups={};required=set();pairs=set()
    for question in questions:
        occurrences=[];source_scenes=[]
        for n in (8,16):
            candidates=[r for r in data['train'].values() if r['question']==question and r['gold']==0 and r['n_frames']==n]
            need(len(candidates)==1,'Each question needs exactly one N8K0 and one N16K0 training scene')
            sample=candidates[0];sid=sample['sid'];scene=parent['scenes'][sid]
            source_scenes.append(dict(sid=sid,n_frames=n,gold=0,qa_sha256=sample['qa_sha256'],content_sha256=sample['content_sha256'],path=scene['path']))
            for i,image in enumerate(sample['image_files']):
                pid=objsha([image['sha256'],question]);pairs.add(pid)
                expected_path=str(Path(scene['path'])/f'{i:03d}.png')
                need(image['path']==expected_path and plan['pairs'][pid]['image_sha256']==image['sha256'],
                    'Null occurrence image order/path differs from independently audited training data')
                occurrences.append(dict(slot=len(occurrences),source_sid=sid,source_n_frames=n,source_frame_index=i,
                    image_question_pair_id=pid,image_sha256=image['sha256'],occurrence_image_path=expected_path))
        qid=objsha(question);bank=dict(question=question,question_sha256=qid,source_scenes=source_scenes,occurrences=occurrences,
            ordering='N8K0 images0..7 then N16K0 images0..15; retain every occurrence',reference_count=24)
        banks[qid]=dict(bank,bank_sha256=objsha(bank));need(len(occurrences)==24,'Auxiliary occurrence multiplicity changed')
        for prefix in prefixes:
            prefix=list(prefix);gid=objsha(['global',question,prefix]);local_ids=[]
            need(gid in parent['features'] and features[gid]['kind']=='global' and features[gid]['prefix_ids']==prefix,
                'Auxiliary query is not the reused current native global prefix')
            for occurrence in occurrences:
                pid=occurrence['image_question_pair_id'];pair=plan['pairs'][pid];fid=objsha(['local',pid,prefix]);local_ids.append(fid);required.add(fid)
                descriptor=features[fid];lid=objsha([pair['layout_id'],prefix]);empty=plan['layouts'][objsha([pair['layout_id'],[]])]
                need(descriptor==dict(feature_id=fid,kind='local',pair_id=pid,question=question,question_sha256=qid,prefix_ids=prefix,
                    base_layout_id=pair['layout_id'],phase='local_prefix' if prefix else 'local_empty',layout_id=lid)
                    and plan['layouts'][lid]['input_ids']==empty['input_ids']+prefix,
                    'Auxiliary feature key or exact input depends on something beyond image/question/strict prefix')
            group_id=objsha(['null_auxiliary',qid,prefix])
            groups[group_id]=dict(question=question,question_sha256=qid,bank_sha256=banks[qid]['bank_sha256'],prefix_ids=prefix,
                global_feature_id=gid,local_feature_ids=local_ids,reference_count=24,auxiliary_only=True,zero_answer_ce_for_assigned_prefix=False)
    need(len(questions)==54 and len(banks)==54 and len(groups)==972 and len(required)==22932 and len(pairs)==1274
        and len(required&set(parent['features']))==3268 and len(required-set(parent['features']))==19664
        and sum(len(g['local_feature_ids']) for g in groups.values())==23328
        and union['null_banks']==banks and union['auxiliary_groups']==groups,'Independent training-only null population/occurrence reconstruction differs')
    return union,dict(file=str(Path(path).resolve()),sha256=sha(path),plan_file=union['plan_file'],plan_sha256=union['plan_sha256'],
        parent_cache_audit=parent_audit,features=53322,training_scenes=1782,unchanged_v10_features=33658,new_local_features=19664,
        auxiliary_groups=972,reference_occurrences_per_group=24,occurrence_feature_positions=23328,
        distinct_image_question_pairs=1274,duplicate_reference_occurrences=22,auxiliary_groups_sha256=objsha(groups),
        all_strict_causal_prefixes_verified=True,all_native_tensor_bytes_verified=True,no_false_zero_answer_ce=True,
        no_reference_rows_at_inference=True,no_new_global_states=True,native_read_boundary='actual native final norm input',
        precision_dispatcher=union['precision_dispatcher'],retained_failed_cpu_merge_job_id='442411',
        precision_scope='CPU verification of descriptive hidden metrics only; original GPU inputs, native states, head gates and raw metrics remain unchanged')


def get_pairing(data,cache):
    parent=read(cache['parent_cache']['file']);value=v10.get_pairing(data,parent)
    return dict(value,protocol='v13_unchanged_v10_pairing_new_seeds',seeds=[16,17],parent_cache=cache['parent_cache'])


def check_auxiliary_row(row):
    import numpy as np
    need(row['auxiliary_coefficient']==1. and close(row['weighted_auxiliary_loss'],row['auxiliary_loss'])
        and close(row['main_loss'],row['ce_loss']+row['consistency_loss'])
        and close(row['loss'],row['main_loss']+row['auxiliary_loss']), 'Three-term objective or coefficients differ')
    d=row['auxiliary'];values={}
    for key in ('per_group_mse','per_group_denominator','per_group_loss','target_norms','prediction_norms',
        'per_group_projected_mse','per_group_projected_error_norms','projected_prediction_norms','projected_target_norms','query_norms'):
        x=np.asarray(d[key],dtype=np.float64)
        need(x.shape==(16,) and np.isfinite(x).all() and (x>=0).all(),'Malformed per-group auxiliary statistic: '+key);values[key]=x
    need((values['per_group_denominator']>=float(np.float32(1e-6))).all()
        and all(close(float(a),float(b)) for a,b in zip(values['per_group_mse']/values['per_group_denominator'],values['per_group_loss']))
        and close(row['auxiliary_loss'],float(values['per_group_loss'].mean())),'Occurrence-group normalized MSE arithmetic differs')
    need(all(d[key] is True for key in ('query_detached','target_detached','denominator_detached')),
        'Auxiliary must detach current core query, mean target and normalization')
    need(all(close(float(a),float(b)) for a,b in zip(values['per_group_projected_error_norms']**2/96,values['per_group_projected_mse']))
        and all(math.isfinite(d[k]) and d[k]>=0 for k in ('aggregate_projection_weight_norm','readout_weight_norm'))
        and d['projected_diagnostics_detached'] is True and d['projected_diagnostics_include_bias'] is False,
        'Projected calibration diagnostic arithmetic or detached no-bias scope differs')
    need(set(row['gradient_norms'])==set(row['clipped_groups'])=={'branch','predictor'},'Two independent gradient clipping groups required')
    for group,norm in row['gradient_norms'].items():
        need(math.isfinite(norm) and norm>=0 and type(row['clipped_groups'][group]) is bool
            and row['clipped_groups'][group]==(norm>1.),'Independent norm1 clipping differs')


def check_loss_row(row,condition,sequences,identity,counts):
    need(condition in CONDITIONS and row['consistency_coefficient']==1.,'Both matched conditions require residual coefficient1')
    # Reuse unchanged ragged CE/pair arithmetic via a private compatibility view.
    # The actual total and both clipping groups are verified separately below.
    copied=dict(row,loss=row['main_loss'],gradient_norm=row['gradient_norms']['branch'],clipped=row['clipped_groups']['branch'])
    v10.check_loss_row(copied,'consistency',sequences,identity);check_auxiliary_row(row)
    expected=[n for n,sequence in zip(counts,sequences) for _ in sequence]
    need(row['position_n_frames']==expected,'Centering coefficient used padding width or wrong actual scene N')
    need(len(row['null_prediction_norms'])==len(expected)
        and all(math.isfinite(x) and x>=0 for x in row['null_prediction_norms']),'Malformed ordinary predictor norms')


def verify_isolation(directory,summary,initialized):
    need(summary['gradient_isolation_passed'] is True
        and Path(summary['gradient_isolation_file']).resolve()==directory/'gradient_isolation.json','Missing isolated-gradient evidence')
    bind(summary['gradient_isolation_file'],summary['gradient_isolation_sha256']);rows=read(summary['gradient_isolation_file'])
    need([row['step'] for row in rows]==[1,2,32],'Fixed initial/early/late profile gradient coverage differs')
    for row in rows:
        need(row['passed'] is True and row['auxiliary_core_gradients_all_none'] is True and row['optimizer_buffers_untouched'] is True,
            'Auxiliary gradient leaked into core or inspection changed optimizer state')
        for key,group in (('main_core','branch'),('main_predictor','predictor'),('auxiliary_predictor','predictor')):
            value=row[key];count=len(initialized[group])
            need(math.isfinite(value['norm']) and value['norm']>=0 and value['connected_tensors']==count
                and type(value['nonzero_tensors']) is int and 0<=value['nonzero_tensors']<=count,
                'Isolated gradient connectivity/finite norm differs')
        if row['step']==1:
            need(row['main_predictor']['norm']==0 and row['main_predictor']['nonzero_tensors']==0
                and row['zero_initial_main_predictor_gradient'] is True,'Initial zero-U must block all main predictor gradients')
        else:need(row['zero_initial_main_predictor_gradient'] is None,'Zero-initialization label used after an update')
    need(rows[-1]['main_predictor']['norm']>0,'Main loss predictor path was not exercised after learning')
    return rows


def verify_history():
    path=OUT.parent/'v12/reference_study/report_442359/summary.json';summary=read(path)
    need(summary['completed'] is True and summary['audit_passed'] is True and summary['no_practical_milestone_claim'] is True
        and summary['records']==408,'Canonical postdiagnostic V12 provenance differs')
    bind(summary['analysis_file'],summary['analysis_sha256']);analysis=read(summary['analysis_file'])
    bind(analysis['plan_file'],analysis['plan_sha256']);need(analysis['no_fit'] is True and analysis['exploratory_reused_test'] is True,
        'Prior result must retain exploratory reused-test/no-fit status')
    ledger(summary['source_sha256'])
    return dict(summary=dict(file=str(path),sha256=sha(path)),analysis_file=summary['analysis_file'],analysis_sha256=summary['analysis_sha256'],
        no_prior_milestone_claim=True)


def initialized_state(seed):
    import torch
    from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
    from gnnformer.conditional_null_mean import ConditionalNullMean
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);core=ParallelLocalAggregation();predictor=ConditionalNullMean()
    need(sum(p.numel() for p in core.parameters())==1041600 and sum(p.numel() for p in predictor.parameters())==18624
        and bool((core.up.weight==0).all()) and bool((predictor.fc2.weight==0).all()) and bool((predictor.fc2.bias==0).all()),
        'Seeded core/predictor count or zero initialization differs')
    return {group:{key:tensor_info(value) for key,value in module.state_dict().items()} for group,module in (('branch',core),('predictor',predictor))}


def checkpoint_state(path,step,config,initialized):
    import torch
    saved=torch.load(path,map_location='cpu',weights_only=True)
    need(set(saved)=={'branch','predictor','step','config'} and saved['step']==step and saved['config']==config,
        'Combined checkpoint step/config/schema differs')
    result={}
    for group in ('branch','predictor'):
        need(set(saved[group])==set(initialized[group]),'Combined checkpoint learned tensor coverage differs')
        for name,value in saved[group].items():
            need(value.dtype==torch.float32 and list(value.shape)==initialized[group][name]['shape'] and bool(torch.isfinite(value).all()),
                'Checkpoint dtype/shape/finite contract differs')
        result[group]={name:tensor_info(value) for name,value in saved[group].items()}
    return result

def loss_summary(logs):
    fields=('loss','main_loss','ce_loss','consistency_loss','path_loss','auxiliary_loss')
    blocks=[]
    for start in range(0,len(logs),918):
        rows=logs[start:start+918]
        blocks.append(dict(first_step=start+1,last_step=start+len(rows),means={k:sum(r[k] for r in rows)/len(rows) for k in fields},
            clipping_fractions={g:sum(r['clipped_groups'][g] for r in rows)/len(rows) for g in ('branch','predictor')},
            auxiliary_means={k:sum(sum(r['auxiliary'][k]) for r in rows)/(16*len(rows))
                for k in ('per_group_mse','per_group_denominator','target_norms','prediction_norms','per_group_projected_mse',
                    'per_group_projected_error_norms','projected_prediction_norms','projected_target_norms','query_norms')},
            weight_norm_means={k:sum(r['auxiliary'][k] for r in rows)/len(rows) for k in ('aggregate_projection_weight_norm','readout_weight_norm')},
            valid_target_positions=sum(len(r['target_ids']) for r in rows),auxiliary_group_presentations=len(rows)*16,
            saturated_identity_pairs=sum(sum(r['saturated_identity_pairs']) for r in rows)))
    return dict(blocks=blocks,final_update={k:logs[-1][k] for k in fields},path_penalty_diagnostic_only=True,
        validation_scope='All recorded position/scene/pair/group arithmetic and source-bound detach/routing contract; unsaved changing messages, targets and intermediate parameters are not reconstructed')


def verify_training_records(directory,data,cache,pairing_value,*,profile):
    directory=Path(directory).resolve();config=read(directory/'config.json');summary=read(directory/'summary.json')
    condition,seed=config['condition'],config['seed'];runid=config['run_id'];steps=32 if profile else 4590
    need(directory.parent==OUT and directory.name==runid and runid.startswith(f'{"profile" if profile else "run"}_{condition}_s{seed}_')
        and condition in CONDITIONS and seed in SEEDS and (not profile or seed==16) and config['arm']=='parallel'
        and config['profile'] is profile,'Unregistered V13 training run')
    need(config['policy']==POLICY and all(summary.get(k)==v for k,v in config.items())
        and config['consistency_coefficient']==config['auxiliary_coefficient']==1. and summary['passed'] is True
        and summary['completed'] is True and summary['computational_integrity_passed'] is True and summary['steps']==steps,
        'V13 policy/config/completion/coefficients differ')
    verify_source(directory,config['source_sha256']);bind(config['plan_file'],config['plan_sha256']);plan_path=Path(config['plan_file']);plan=read(plan_path)
    need(plan_path.with_suffix('.sha256').read_text().strip()==config['plan_sha256'] and plan['source_sha256']==config['source_sha256']
        and plan['policy']==POLICY and plan['profile_calls']==PROFILE_CALLS,'Training CPU source/plan differs')
    cpu=read(plan_path.parent/'summary.json')
    need(cpu['passed'] is True and cpu['plan_sha256']==config['plan_sha256'] and cpu['source_sha256']==config['source_sha256'],
        'Matching completed CPU training check required')
    ledger(plan['artifact_bindings'])
    need(config['prior_result']==plan['prior_result']==verify_history(),'Prior diagnostic provenance changed')
    need(Path(plan['train_manifest']).resolve()==DATA/'v10_balanced/main_manifest.json'
        and Path(plan['schedule_file']).resolve()==DATA/'v10_balanced/schedule.json'
        and Path(plan['original_pairing_file']).resolve()==DATA/'v10_balanced/pairing.json'
        and plan['fresh_manifests']=={} and plan['fresh_test_bound_by_future_independent_release'] is True,
        'Training data paths or independently released fresh-test contract differs')
    for path,digest in data['summary']['data_bindings'].items():
        if path in plan['artifact_bindings']:need(plan['artifact_bindings'][path]==digest,'Training/data audit binding conflicts')
    binding=config['cache_binding'];need(binding==plan['cache_binding'] and Path(binding['file']).resolve()==CACHE,'Wrong training cache')
    bind(binding['file'],binding['sha256']);need(all(config[k]==cache[k] for k in ('model','runtime','processor','native_dtypes')),
        'Native model/runtime/cache differs')
    need(config['pairing_file']==plan['pairing_file'] and config['pairing_sha256']==plan['pairing_sha256'],'Pairing binding differs')
    bind(plan['pairing_file'],plan['pairing_sha256']);need(read(plan['pairing_file'])==pairing_value
        and objsha(pairing_value)==plan['pairing_object_sha256'],'Independent pair construction differs')
    order=expected_order(pairing_value['pairs'],seed);aux=expected_auxiliary_order(cache['auxiliary_groups'],seed)
    bind(directory/'presentations.json',config['presentations_sha256']);bind(directory/'auxiliary_presentations.json',config['auxiliary_presentations_sha256'])
    need(read(directory/'presentations.json')==order and objsha(order)==config['order_sha256']==plan['order_sha256'][str(seed)]
        and read(directory/'auxiliary_presentations.json')==aux and objsha(aux)==config['auxiliary_order_sha256']==plan['auxiliary_order_sha256'][str(seed)]
        and config['auxiliary_groups_sha256']==plan['auxiliary_groups_sha256']==objsha(cache['auxiliary_groups']),
        'Ordinary or independently shuffled auxiliary presentation order differs')
    initial=initialized_state(seed);need(config['initialized']==initial and config['initialized_sha256']==objsha(initial),
        'Combined seeded initialization differs')
    need(Path(config['checkpoint_directory']).resolve()==CKPT/runid
        and Path(config['data_directory']).resolve()==DATA/'native_aggregation_vlm_v13'/runid,'Training model/data storage roots differ')
    need(Path(summary['training_file']).resolve()==directory/'training.json','Unexpected training ledger path')
    bind(summary['training_file'],summary['training_sha256']);logs=read(summary['training_file']);need(len(logs)==steps,'Missing optimizer updates')
    total_targets=0;identities=0
    for step,row in enumerate(logs,1):
        positions=order[(step-1)*16:step*16];aux_positions=aux[(step-1)*16:step*16];sids=[p['sid'] for p in positions]
        sequences=[cache['scenes'][sid]['target_ids'] for sid in sids];identity=[sids[i]==sids[i+1] for i in range(0,16,2)]
        need(row['step']==step and row['sids']==sids and row['pair_ids']==[p['pair_id'] for p in positions[::2]]
            and row['epochs']==[p['epoch'] for p in positions[::2]] and row['auxiliary_group_ids']==[p['group_id'] for p in aux_positions]
            and row['auxiliary_cycles']==[p['cycle'] for p in aux_positions] and close(row['lr'],learning_rate(step),atol=1e-12),
            'Actual ordinary/auxiliary update order or optimizer schedule differs')
        need(all(positions[i]['pair_id']==positions[i+1]['pair_id'] and positions[i]['pair_side']==0
            and positions[i+1]['pair_side']==1 for i in range(0,16,2)),'A carried batch split/reversed a semantic pair')
        check_loss_row(row,condition,sequences,identity,[p['n_frames'] for p in positions]);total_targets+=sum(map(len,sequences));identities+=sum(identity)
    if not profile:need(total_targets==177120 and identities==2160,'Weighted target or saturated identity totals differ')
    need(close(summary['training_seconds'],sum(r['seconds'] for r in logs),atol=1e-4)
        and logs[0]['consistency_loss']==logs[0]['path_loss']==logs[0]['weighted_consistency_loss']==0.,'Training timing or zero-U first losses differ')
    need(Path(summary['first_gradients_file']).resolve()==directory/'first_gradients.json','Unexpected first-gradient file')
    bind(summary['first_gradients_file'],summary['first_gradients_sha256']);gradients=read(summary['first_gradients_file'])
    need([r['step'] for r in gradients]==[1,2],'Both first gradient records required')
    for row in gradients:
        need(set(row['pre_clip'])=={'branch','predictor'},'Gradient clipping groups differ')
        for group in ('branch','predictor'):
            need(set(row['pre_clip'][group])==set(initial[group]),'Gradient tensor coverage differs')
            for name,info in row['pre_clip'][group].items():
                need(info['shape']==initial[group][name]['shape'] and info['dtype']=='torch.float32'
                    and isinstance(info['sha256'],str) and len(info['sha256'])==64,'Gradient tensor shape/dtype/hash differs')
    isolation=verify_isolation(directory,summary,initial)
    audited=dict(condition=condition,arm='parallel',seed=seed,run_directory=str(directory),config_sha256=sha(directory/'config.json'),
        summary_sha256=sha(directory/'summary.json'),initialized_sha256=config['initialized_sha256'],presentations_sha256=config['presentations_sha256'],
        auxiliary_presentations_sha256=config['auxiliary_presentations_sha256'],auxiliary_order_sha256=config['auxiliary_order_sha256'],
        pairing_object_sha256=plan['pairing_object_sha256'],training_seconds=summary['training_seconds'],loss_diagnostics=loss_summary(logs),
        first_preclip_gradient_hashes=gradients[0]['pre_clip'],first_update={k:logs[0][k] for k in ('loss','ce_loss','consistency_loss','path_loss','auxiliary_loss','auxiliary')},
        gradient_isolation=isolation,verified_optimizer_log_rows=steps,verified_scene_presentations=steps*16,
        verified_pair_presentations=steps*8,verified_auxiliary_presentations=steps*16,verified_target_positions=total_targets,
        verified_saturated_identity_pair_presentations=identities,
        clipping_fractions={g:sum(r['clipped_groups'][g] for r in logs)/steps for g in ('branch','predictor')})
    return audited,config,summary,logs


def verify_run(directory,data,tokenizer,cache,pairing_value):
    directory=Path(directory).resolve();audited,config,summary,logs=verify_training_records(directory,data,cache,pairing_value,profile=False)
    need(summary['native_test_count']==272,'Fresh main test coverage differs');runid=config['run_id'];initial=config['initialized'];condition=config['condition']
    selection=read(directory/'selection.json');entries=selection['development']
    need([e['step'] for e in entries]==DEV_STEPS and summary['development']==entries,'All five registered dev checkpoints required')
    for entry in entries:
        step=entry['step'];devpath=directory/f'dev_{step}.json';need(Path(entry['dev_file']).resolve()==devpath,'Unexpected dev artifact path')
        bind(devpath,entry['dev_sha256']);dev=audit_eval(devpath,data['dev'],'parallel',tokenizer,condition)
        need(dev['label']==f'dev_{step}' and dev['exact_count']==entry['exact_count'] and close(dev['first_token_nll'],entry['nll']),
            'Rescored development selection differs')
        path=Path(entry['checkpoint']).resolve();need(path==CKPT/runid/f'step_{step}.pt','Checkpoint outside registered run directory')
        bind(path,entry['checkpoint_sha256']);need(objsha(checkpoint_state(path,step,config,initial))==entry['parameter_sha256'],
            'Combined checkpoint parameter hash differs')
    best=select(entries);need(selection['selected']==best and summary['selected']==best,'Selected checkpoint differs from registered optimum')
    testpath=directory/'test.json';need(Path(summary['test_file']).resolve()==testpath,'Unexpected test artifact path')
    bind(testpath,summary['test_sha256']);test=audit_eval(testpath,data['test'],'parallel',tokenizer,condition);need(test['label']=='test','Unexpected main test label')
    audited.update(test_sha256=sha(testpath),selected_step=best['step'],selected_checkpoint=best['checkpoint'],
        selected_checkpoint_sha256=best['checkpoint_sha256'],selected_parameter_sha256=best['parameter_sha256'],
        evaluation_seconds=test['seconds'],native_test_model_forwards=sum(r['counters']['model'] for r in test['rows']),
        native_test_visual_forwards=272,verified_dev_evaluations=5,verified_test_examples=272)
    return audited,test['rows'],config


def numeric_metric(stored,recomputed):
    need(set(stored)==set(recomputed) and all(stored[k]==recomputed[k] for k in ('passed','all_top1_equal','top1_equal'))
        and len(stored['tv'])==len(recomputed['tv'])
        and all(math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-12) for a,b in zip(stored['tv'],recomputed['tv']))
        and math.isclose(stored['maximum_tv'],recomputed['maximum_tv'],rel_tol=1e-9,abs_tol=1e-12),
        'Raw profile metrics differ beyond CPU/GPU FP64 reduction precision')


def verify_training_profile(item,condition,config,data,cache):
    import torch
    from scripts.cache_native_vision_v7_features import replay_metrics
    directory=Path(item['directory']).resolve();bind(directory/'summary.json',item['summary_sha256'])
    audited,pc,profile,logs=verify_training_records(directory,data,cache,get_pairing(data,cache),profile=True)
    need(profile['condition']==condition and profile['profile_calls']==PROFILE_CALLS and profile['no_dev_or_test_evaluation'] is True
        and profile['frozen_backbone_gradient_state_preserved'] is True and profile['checkpoint_roundtrip_passed'] is True
        and all(profile[k]==config[k] for k in ('plan_sha256','source_sha256','model','runtime','processor','native_dtypes','hardware','cache_binding','pairing_sha256')),
        'Matched training profile/backend differs')
    zero=profile['zero_initialization_auxiliary_gradients']
    need(zero['passed'] is True and zero['residual_loss']==zero['path_loss']==0. and zero['up_exactly_zero'] is True
        and zero['predictor_second_layer_exactly_zero'] is True and zero['all_residual_gradients_zero'] is True
        and zero['all_path_gradients_zero'] is True,'Initial U/C2 regularizer gradient gate differs')
    exercised=profile['all_parameter_gradients_exercised'];need(set(exercised)=={'branch','predictor'},'Missing learned gradient group')
    for group in exercised:need(set(exercised[group])==set(profile['initialized'][group]) and all(v is True for v in exercised[group].values()),
        'Profile failed to exercise all learned tensors')
    need(close(profile['step_seconds_max_steady'],max(r['seconds'] for r in logs[4:])) and 0<profile['elapsed_seconds']<=300,
        'Profile steady update time or registered cap differs')
    checkpoint=Path(profile['profile_checkpoint']).resolve();need(checkpoint==CKPT/profile['run_id']/'profile.pt','Profile checkpoint path differs')
    bind(checkpoint,profile['profile_checkpoint_sha256']);checkpoint_state(checkpoint,32,pc,profile['initialized'])
    replay=profile['active_cache_replay'];need(read(directory/'active_cache_replay.json')==replay and replay['passed'] is True
        and replay['counts_covered']==[0,9,10,16] and replay['prefixes_checked']==10
        and replay['native_model_calls']==replay['vision_calls']==10 and replay['standalone_head_calls']==20,
        'Profile native strict-prefix replay coverage differs')
    bind(replay['raw_file'],replay['raw_sha256']);raw=torch.load(replay['raw_file'],map_location='cpu',weights_only=True)
    expected=[]
    for k in (0,9,10,16):
        sample=sorted((r for r in data['train'].values() if r['n_frames']==16 and r['gold']==k),key=lambda r:r['sid'])[0]
        for t in range(len(cache['scenes'][sample['sid']]['target_ids'])):expected.append((sample,t))
    need(len(raw)==len(replay['observations'])==len(expected)==10,'Profile saved replay tensor count differs')
    for row,tensors,(sample,t) in zip(replay['observations'],raw,expected):
        prefix=cache['scenes'][sample['sid']]['target_ids'][:t]
        need(row['sid']==sample['sid'] and row['gold']==sample['gold'] and row['position']==t and row['prefix_ids']==prefix
            and row['n_frames']==16 and row['counters']==dict(model=1,visual=1,norm=1) and row['cached_vs_native_gate'] is False,
            'Fixed training profile input/call/descriptive scope differs')
        meta=row['native_metadata'];need(meta['sid']==sample['sid'] and meta['n_frames']==16 and meta['prefix_ids']==prefix
            and meta['question']==sample['question'] and meta['row_count']==17 and meta['global_row']==16
            and meta['null_mode']==condition and meta['null_predictor_parameters']==18624
            and meta['native_dtype']=='torch.float16' and meta['branch_dtype']=='torch.float32'
            and meta['layout']['every_unpadded_row_exact'] is True,'Profile native local/global ownership or dtype differs')
        need(set(tensors)=={'native','replayed','cached'} and all(value.ndim==2 and value.shape[0]==1
            and value.dtype==torch.float16 and bool(torch.isfinite(value).all()) for value in tensors.values()),
            'Profile raw native logits are malformed')
        numeric_metric(row['native_head_replay'],replay_metrics(torch,tensors['native'],tensors['replayed']))
        numeric_metric(row['cached_vs_native_descriptive'],replay_metrics(torch,tensors['native'],tensors['cached']))
        need(row['native_head_replay']['passed'] is True,'Same-captured native-shaped head replay failed')
    need(replay['cached_numerical_failures']==sum(not row['cached_vs_native_descriptive']['passed'] for row in replay['observations']),
        'Known cached/native batch-route differences were removed')
    return profile

ACCOUNTING_FIELDS=('JobIDRaw','JobID','JobName','Partition','State','ElapsedRaw','AllocTRES','ExitCode','Submit','Start','End')
TERMINAL_STATES={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}


def parse_accounting(raw,*,all_user=False):
    rows=[];seen=set()
    for line in raw.splitlines():
        if not line.strip():continue
        values=line.split('|');need(len(values)==len(ACCOUNTING_FIELDS),'Unexpected raw Slurm accounting columns')
        row=dict(zip(ACCOUNTING_FIELDS,values))
        if all_user and (row['Partition']!='gpu' or not row['JobName'].startswith('v13_')):continue
        need(re.fullmatch('[0-9]+',row['JobIDRaw']) and re.fullmatch('[0-9]+(?:_[0-9]+)?',row['JobID'])
            and row['JobIDRaw'] not in seen and row['Partition']=='gpu' and row['JobName'].startswith('v13_'),
            'Accounting includes nonallocation, duplicate or noncampaign rows')
        state=row['State'].split()[0].rstrip('+');need(state in TERMINAL_STATES,'Every prior campaign allocation must be terminal')
        need(row['ElapsedRaw'].isdigit(),'Malformed allocation duration');elapsed=int(row['ElapsedRaw'])
        tres=dict(item.split('=',1) for item in row['AllocTRES'].split(',') if '=' in item)
        generic=tres.get('gres/gpu');typed=[v for k,v in tres.items() if k.startswith('gres/gpu:')]
        need((generic is None or generic.isdigit()) and all(v.isdigit() for v in typed),'Malformed GPU TRES')
        count=int(generic) if generic is not None else sum(map(int,typed))
        need(count in (0,1),'Unexpected number of GPUs in a registered allocation')
        if count:need(row['Start'] not in ('','Unknown','None') and row['End'] not in ('','Unknown','None'),'Missing allocated start/end')
        row.update(state=state,gpu_count=count,allocated_gpu_seconds=count*elapsed);seen.add(row['JobIDRaw']);rows.append(row)
    need(rows,'No prior V13 GPU accounting records');return sorted(rows,key=lambda r:int(r['JobIDRaw']))


def verify_budget(budget):
    bind(budget['sacct_file'],budget['sacct_sha256']);bind(budget['all_user_sacct_file'],budget['all_user_sacct_sha256'])
    rows=parse_accounting(Path(budget['sacct_file']).read_text())
    all_rows=parse_accounting(Path(budget['all_user_sacct_file']).read_text(),all_user=True)
    need(rows==all_rows and sorted(budget['allocations'],key=lambda r:int(r['JobIDRaw']))==rows
        and len(budget['prior_jobs'])==len(rows) and len(set(budget['prior_jobs']))==len(rows)
        and set(budget['prior_jobs'])=={r['JobIDRaw'] for r in rows},'A prior V13 GPU allocation was omitted, duplicated or changed')
    total=sum(r['allocated_gpu_seconds'] for r in rows)
    need(total==budget['prior_allocated_gpu_seconds'] and budget['passed'] is True and budget['campaign_gpu_seconds_cap']==16200
        and budget['reserved_main_gpu_seconds']==10800 and budget['reserved_selected_audit_gpu_seconds']==300
        and budget['projected_campaign_gpu_seconds']==total+11100<=16200,'Independent allocation/reservation arithmetic differs')
    return dict(passed=True,prior_allocations=len(rows),failed_allocations=sum(r['state']!='COMPLETED' for r in rows),
        prior_allocated_gpu_seconds=total,all_user_campaign_coverage_verified=True)


def verify_software_timing(software):
    import torch
    plan=read(software['plan_file']);bounds={c:{} for c in CONDITIONS}
    for row in software['timing']['rows']:
        bind(row['raw_file'],row['raw_sha256']);raw=torch.load(row['raw_file'],map_location='cpu',weights_only=True)
        ids=raw['generated_ids'];x=raw['raw_logits'];steps=len(ids);complete=ids[-1] in POLICY['native_eos'];mode=row['mode'];n=row['n_frames']
        case=plan['cases'][0 if n==16 else 1]
        need(steps==row['generated_tokens'] and 1<=steps<=4 and stored_native_logits(x) and x.ndim==2 and x.shape[0]==steps
            and x.argmax(-1).tolist()==ids and not any(i in POLICY['native_eos'] for i in ids[:-1]) and (complete or steps==4)
            and raw['completed']==complete and raw['truncated']==(not complete) and raw['captures']==[],
            'Native timing raw prefix/EOS/FP16-promotion trace differs')
        need(raw['metadata']['sid']==case['sid']==row['sid'] and raw['metadata']['null_mode']==mode
            and raw['metadata']['n_frames']==n and raw['metadata']['row_count']==n+1
            and row['counters']==raw['counters']==dict(model=steps,visual=1,language=steps,fusion=steps,broadcast=steps)
            and row['no_efficacy_scoring'] is True and row['no_full_hidden_or_kv_copies'] is True,
            'Timing scene/mode/call inventory differs')
        need(all(math.isfinite(row[k]) and row[k]>0 for k in ('preparation_seconds','generation_seconds','runtime_internal_seconds'))
            and close(raw['model_seconds'],row['runtime_internal_seconds'])
            and row['generation_seconds']>=row['runtime_internal_seconds'],'Native timed envelope differs')
        bound=row['preparation_seconds']+row['generation_seconds']*4/steps
        need(close(bound,row['four_token_seconds_bound']) and close(bound,software['timing']['by_mode'][mode][str(n)]),
            'Four-token timing arithmetic differs');bounds[mode][str(n)]=bound
    need(all(set(v)=={'16','64'} for v in bounds.values()),'Mode/length timing coverage incomplete');return bounds


def verify_release(config,data,frozen):
    binding=config['main_release'];bind(binding['file'],binding['sha256']);release=read(binding['file'])
    need(release['protocol']=='v13_learned_null_main_release' and release['passed'] is True
        and release['plan_file']==config['plan_file'] and release['plan_sha256']==config['plan_sha256']
        and release['source_sha256']==config['source_sha256'] and release['report_source_sha256']==frozen,
        'Main release source/report/training-plan identity differs')
    need(release['per_main_seconds_cap']==2700 and release['main_block_gpu_seconds_cap']==16200,'Registered V13 resource cap differs')
    own=release['release_source_sha256'];need(set(own)=={'scripts/release_native_vision_v13_mains.py','slurm/native_vision_v13_main_release.sbatch'},
        'Unexpected measured-release source ledger')
    ledger(own);directory=Path(binding['file']).parent;need(read(directory/'source_hashes.json')==own,'Release source ledger copy differs')
    selected=release['selected_audit_freeze'];selected_directory=Path(selected['directory']).resolve()
    bind(selected_directory/'summary.json',selected['summary_sha256']);selected_summary=read(selected_directory/'summary.json')
    need(selected_summary['passed'] is True and selected_summary['tests_passed'] is True
        and selected_summary['protocol']=='all_selected_v13_native_checkpoint_audit' and selected_summary['no_model_loaded'] is True
        and all(selected_summary['source_sha256'].get(name)==digest for name,digest in frozen.items()),
        'Pre-main selected-checkpoint auditor source/selftest freeze differs')
    ledger(selected_summary['source_sha256']);need(read(selected_directory/'source_hashes.json')==selected_summary['source_sha256'],
        'Selected-checkpoint auditor source ledger copy differs')
    for name,digest in selected_summary['source_sha256'].items():bind(selected_directory/'source'/name.replace('/','_'),digest)
    for name,digest in own.items():bind(directory/'source'/name.replace('/','_'),digest)
    checked_binding=release['data_release'];bind(checked_binding['file'],checked_binding['sha256']);checked=read(checked_binding['file'])
    need(checked['passed'] is True and checked['source_sha256']==frozen and checked['data_bindings']==data['summary']['data_bindings']
        and checked['manifest_only'] is False and checked['cache_audit']['features']==53322
        and checked['cache_audit']['all_native_tensor_bytes_verified'] is True,'Independent fresh-data/cache/report release differs')
    bind(checked['audit_file'],checked['audit_sha256']);bind(checked['pairing_file'],checked['pairing_sha256'])
    fresh=release['fresh_manifest'];expected_path=DATA/'v11_fresh/main_manifest.json'
    need(Path(fresh['file']).resolve()==expected_path and fresh['sha256']==sha(expected_path)
        and checked['data_bindings'][str(expected_path)]==fresh['sha256'] and binding['fresh_manifest']==fresh,
        'First-used fresh test differs from the independent prospective release')
    plan=read(config['plan_file']);software_binding=plan['native_profile'];bind(software_binding['file'],software_binding['sha256'])
    from scripts.profile_native_vision_v13_null import verify_profile
    software=verify_profile(software_binding['file'])
    need(software['plan_file']==software_binding['plan_file'] and software['plan_sha256']==software_binding['plan_sha256']
        and software['source_sha256']==software_binding['source_sha256']
        and all(software[k]==config[k] for k in ('model','runtime','processor')),'Native timing software differs from the train plan')
    timings=verify_software_timing(software);need(set(timings)==set(CONDITIONS) and all(set(t)=={'16','64'} for t in timings.values()),
        'Fresh mode-specific native timing coverage differs')
    shared={n:max(t[n] for t in timings.values()) for n in ('16','64')};cache=read(CACHE)
    need(set(release['profiles'])==set(release['projections'])==set(CONDITIONS),'Both fixed matched profiles/projections required')
    profiles={condition:verify_training_profile(item,condition,config,data,cache) for condition,item in release['profiles'].items()}
    max_step=max(p['step_seconds_max_steady'] for p in profiles.values())
    for condition,profile in profiles.items():
        need(profile['hardware']['name']==software['hardware']['gpu'],'Training and native timing GPU hardware differs')
        projection=release['projections'][condition];training=4590*max_step;development=320*shared['16'];test=272*shared['64']
        projected=profile['model_and_features_load_seconds']+1.25*(training+development+test)+120
        need(projection['generation_seconds']==shared and projection['N32_uses_N64_bound'] is True
            and close(projection['training_seconds'],training) and close(projection['development_seconds'],development)
            and close(projection['test_seconds'],test) and close(projection['projected_seconds'],projected)
            and projection['passed'] is True and 0<projected<=2700,'Independent measured V13 time projection differs or exceeds cap')
    budget=release['campaign_budget'];allocation_audit=verify_budget(budget)
    successful={**{'training_'+c:(str(p['slurm_job_id']),300) for c,p in profiles.items()},
        'native_software':(str(software['slurm_job_id']),300)}
    feature_profile=read(Path(cache['profile_directory'])/'summary.json')
    successful['feature_profile']=(str(feature_profile['slurm_job_id']),180)
    for item in cache['shards']:
        feature=read(Path(item['directory'])/'summary.json');successful['feature_shard_'+str(feature['shard'])]=(str(feature['slurm_job_id']),600)
    roles=budget['successful_artifacts'];need(len(roles)==8 and {item['role'] for item in roles}==set(successful),
        'Eight successful software/feature/training allocation roles required')
    for item in roles:
        job,cap=successful[item['role']];matches=[r for r in budget['allocations'] if job in (r['JobIDRaw'],r['JobID'])]
        need(len(matches)==1 and item['job_id']==job and item['seconds_cap']==cap and item['accounting_job_id']==matches[0]['JobIDRaw']
            and matches[0]['state']=='COMPLETED' and matches[0]['ExitCode']=='0:0' and 0<matches[0]['allocated_gpu_seconds']<=cap,
            'Successful artifact is not bound to its actual completed GPU allocation/cap')
    need(len({item['accounting_job_id'] for item in roles})==8,'Successful roles reused one allocation')
    feature_cost=sum(r['allocated_gpu_seconds'] for r in budget['allocations'] if r['JobName'] in ('v13_null_profile','v13_null_harvest'))
    software_cost=sum(r['allocated_gpu_seconds'] for r in budget['allocations'] if r['JobName']=='v13_native_profile')
    need(budget['feature_allocated_gpu_seconds']==feature_cost<=2700 and budget['native_software_allocated_gpu_seconds']==software_cost<=600
        and not any(r['JobName'] in ('v13_main','v13_selected') for r in budget['allocations']),
        'Feature/software subcap or pre-outcome release scope differs')
    need(budget['passed'] is True and budget['campaign_gpu_seconds_cap']==16200 and budget['reserved_main_gpu_seconds']==10800
        and budget['reserved_selected_audit_gpu_seconds']==300 and isinstance(budget['prior_jobs'],list) and budget['prior_jobs']
        and math.isfinite(budget['prior_allocated_gpu_seconds']) and budget['prior_allocated_gpu_seconds']>=0,
        'Campaign allocation ledger or future reservations differ')
    projected=budget['prior_allocated_gpu_seconds']+10800+300
    need(close(projected,budget['projected_campaign_gpu_seconds']) and projected<=16200,'Insufficient remaining campaign budget')
    return dict(file=binding['file'],sha256=binding['sha256'],data_release=checked_binding,fresh_manifest=fresh,
        projections=release['projections'],release_source_sha256=own,selected_audit_freeze=selected,campaign_budget=budget,allocation_audit=allocation_audit,native_software=software_binding,
        retained_software_cached_full_failures=software['cached_full_failures'],
        feature_route_numerical_failures={c:p['active_cache_replay']['cached_numerical_failures'] for c,p in profiles.items()},
        matched_main_and_auxiliary_gradient_paths_verified=True)


def self_test():
    import ast
    import copy
    import numpy as np
    tests=[]
    # Compare literal independent POLICY, without importing or executing trainer.
    source=(REPO/'scripts/train_native_vision_v13.py').read_text();tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='POLICY' for t in n.targets))
    expected=eval(compile(ast.Expression(node.value),'<trainer literal policy>','eval'),{'__builtins__':{},'dict':dict,'list':list,'range':range})
    need(expected==POLICY,'Independent trainer/reporter policies differ');tests.append('literal_policy_parity')
    # Integer decisions must not round 6/136 upward to the registered 7/136 gate.
    metric=lambda n32,n64,large:dict(cells={'main_test_N32':{'correct':n32},'main_test_N64':{'correct':n64}},partitions={'N64':{'K9_16':{'correct':large}}})
    need(integer_decision(metric(123,109,52),metric(129,102,0))['practical'],'Exact threshold should pass')
    for treated,control in ((metric(123,109,52),metric(129,103,0)),(metric(123,109,52),metric(130,102,0)),
        (metric(122,109,52),metric(122,102,0)),(metric(123,108,52),metric(123,101,0)),(metric(123,109,51),metric(123,102,0))):
        need(not integer_decision(treated,control)['practical'],'A failed seed/threshold was rescued by rounding')
    tests.append('integer_primary_and_practical_boundaries')
    class Tokens:
        all_special_ids=[151645,151643,99]
        def decode(self,ids,skip_special_tokens=False):
            table={0:'0',1:'1',2:'2',8:'١',9:' ',99:'',151645:'<eos>',151643:'<eot>'}
            return ''.join(table[t] for t in ids if not skip_special_tokens or t not in self.all_special_ids)
    tokenizer=Tokens()
    need(parse_answer([1,0,151645],tokenizer)['parsed']==10 and parse_answer([1,0,151645],tokenizer)['completed'],
        'Complete two-digit native answer rejected')
    need(parse_answer([1,99,151645],tokenizer)['parsed'] is None and parse_answer([8,151645],tokenizer)['parsed'] is None
        and not parse_answer([1,0,9,9],tokenizer)['completed'],'Special-token sanitization, non-ASCII numeral or truncation accepted')
    tests.append('whole_ascii_integer_native_eos_and_nonterminal_specials')
    groups={str(i):{} for i in range(972)};order=expected_auxiliary_order(groups,16)
    need(order==expected_auxiliary_order(groups,16) and order!=expected_auxiliary_order(groups,17)
        and [r['cycle'] for r in order[960:976]]==[1]*12+[2]*4,'Independent auxiliary shuffle/carry differs')
    tests.append('persistent_auxiliary_shuffle_and_carried_cycles')
    def accounting_row(job,state,elapsed,tres,start='2026-09-11T00:00:00',end='2026-09-11T00:00:30'):
        return '|'.join([job,job,'v13_test','gpu',state,str(elapsed),tres,'0:0','2026-09-11T00:00:00',start,end])
    text='\n'.join([accounting_row('41','COMPLETED',30,'gres/gpu=1,gres/gpu:b200=1'),
        accounting_row('42','FAILED',7,'gres/gpu:b200=1'),accounting_row('43','CANCELLED by1',0,'','Unknown','Unknown')])
    accounted=parse_accounting(text)
    need(sum(r['allocated_gpu_seconds'] for r in accounted)==37 and [r['gpu_count'] for r in accounted]==[1,1,0],
        'Failed allocation, typed GPU alias or zero allocation accounting differs')
    for bad in (text+'\n'+accounting_row('41','COMPLETED',1,'gres/gpu=1'),accounting_row('44','RUNNING',3,'gres/gpu=1')):
        try:parse_accounting(bad)
        except ValueError:pass
        else:raise AssertionError('Duplicate or active allocation accepted')
    tests.append('independent_failed_allocation_and_gpu_tres_accounting')
    # Reject wrong auxiliary group weighting and silently combined clipping.
    row=dict(auxiliary_coefficient=1.,weighted_auxiliary_loss=.5,auxiliary_loss=.5,main_loss=3.,ce_loss=2.,consistency_loss=1.,loss=3.5,
        auxiliary=dict(per_group_mse=[1.]*16,per_group_denominator=[2.]*16,per_group_loss=[.5]*16,target_norms=[1.]*16,prediction_norms=[2.]*16,
            query_detached=True,target_detached=True,denominator_detached=True,
            per_group_projected_mse=[1.]*16,per_group_projected_error_norms=[math.sqrt(96)]*16,projected_prediction_norms=[1.]*16,
            projected_target_norms=[2.]*16,query_norms=[1.]*16,aggregate_projection_weight_norm=1.,readout_weight_norm=0.,
            projected_diagnostics_detached=True,projected_diagnostics_include_bias=False),gradient_norms={'branch':2.,'predictor':.5},clipped_groups={'branch':True,'predictor':False})
    check_auxiliary_row(row)
    for mutate in (lambda r:r['auxiliary']['per_group_denominator'].__setitem__(0,1.),
        lambda r:r['clipped_groups'].__setitem__('predictor',True),lambda r:r['auxiliary'].__setitem__('target_detached',False)):
        broken=copy.deepcopy(row);mutate(broken)
        try:check_auxiliary_row(broken)
        except ValueError:pass
        else:raise AssertionError('Invalid auxiliary arithmetic/clipping/graph contract accepted')
    tests.append('auxiliary_group_arithmetic_and_separate_clipping')
    # Full-support deterministic fixtures make paired bootstrap intervals exact.
    all_rows={}
    for c in CONDITIONS:
        for seed in SEEDS:
            all_rows[c,seed]=[dict(gold=k,anchor_id=f'{k}_{i}',n_frames=n,exact=(c=='centered'))
                for k in range(17) for i in range(8) for n in (32,64)]
    intervals=bootstrap(all_rows,replicates=50)
    need(set(intervals['results'])=={'all',*(name for name,_ in PARTITIONS)}
        and all(value['difference_ci95_pp']==[100.,100.] for part in intervals['results'].values() for cell in part.values() for value in cell.values()),
        'Bootstrap failed to preserve intact families or common paired draws')
    tests.append('paired_family_bootstrap_and_prefix_partitions')
    # FP32 storage must be a lossless native FP16 promotion, not approximate.
    import torch
    need(stored_native_logits(torch.tensor([1.,2.],dtype=torch.float16).float()) and not stored_native_logits(torch.tensor([.1],dtype=torch.float32)),
        'Native FP16 storage contract was weakened');tests.append('exact_native_fp16_promotion')
    return dict(passed=True,tests=tests)


def report_runs(directories,out,frozen):
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(4);tests=self_test();data=audit_data();tokenizer=AutoTokenizer.from_pretrained(str(MODEL),use_fast=False,local_files_only=True)
    cache,cache_audit=verify_cache(CACHE,data,tokenizer);pairing=get_pairing(data,cache)
    save(out/'data_audit.json',dict(data['summary'],cache_audit=cache_audit));save(out/'pairing.json',pairing)
    audits={};rows={};configs={};releases=[]
    need(len(directories)==4 and len({Path(p).resolve() for p in directories})==4,'All four distinct registered V13 mains required')
    for directory in directories:
        audited,predictions,config=verify_run(directory,data,tokenizer,cache,pairing);key=audited['condition'],audited['seed']
        need(key not in audits,'Duplicate condition/seed');audits[key],rows[key],configs[key]=audited,predictions,config
        releases.append(verify_release(config,data,frozen));print(json.dumps(dict(audited_run=str(directory),native_examples=len(predictions))),flush=True)
    need(set(audits)=={(c,s) for c in CONDITIONS for s in SEEDS} and all(r==releases[0] for r in releases),'Missing run or inconsistent prospective release')
    parity={}
    for seed in SEEDS:
        a,b=configs['offset',seed],configs['centered',seed]
        need(all(a[k]==b[k] for k in ('initialized_sha256','presentations_sha256','order_sha256','auxiliary_presentations_sha256','auxiliary_order_sha256',
            'policy','plan_file','plan_sha256','source_sha256','cache_binding','pairing_sha256','auxiliary_groups_sha256','model','runtime','processor','native_dtypes','prior_result')),
            'Matched conditions differ in initialization, data, capacity or optimizer policy')
        aa,bb=audits['offset',seed],audits['centered',seed]
        need(aa['first_preclip_gradient_hashes']==bb['first_preclip_gradient_hashes'] and aa['first_update']==bb['first_update'],
            'Matched zero-U/zero-C2 first losses or gradient bytes differ')
        parity[str(seed)]=dict(passed=True,combined_initialization_equal=True,ordinary_and_auxiliary_orders_equal=True,
            first_update_losses_and_gradients_equal=True,later_gradients_may_differ=True)
    runs={f'{c}_s{s}':dict(audits[c,s],metrics=summarize(rows[c,s])) for c in CONDITIONS for s in SEEDS}
    decisions={str(s):integer_decision(runs[f'centered_s{s}']['metrics'],runs[f'offset_s{s}']['metrics']) for s in SEEDS}
    primary=all(d['primary'] for d in decisions.values());practical=all(d['practical'] for d in decisions.values());intervals=bootstrap(rows)
    analysis=dict(schema_version=1,passed=True,audit_passed=True,source_sha256=frozen,policy=POLICY,data=data['summary'],cache=cache_audit,
        pairing=dict(file=str(out/'pairing.json'),sha256=sha(out/'pairing.json'),object_sha256=objsha(pairing),pair_count=918,
            unique_training_scenes=1782,weighted_scenes_per_epoch=1836,target_positions_per_epoch=4428),release=releases[0],self_tests=tests,
        runs=runs,decisions=decisions,matched_initialization_and_first_update=parity,primary_both_seeds=primary,practical_both_seeds=practical,
        vision_milestone_gate=primary and practical,post_checkpoint_verification='Separate mandatory all-selected-checkpoint native audit remains required',
        pooled_fixed_two_seed_n64_difference_pp=sum(d['n64_difference_pp'] for d in decisions.values())/2,bootstrap=intervals,
        interpretation=dict(treatment='Cardinality scaling of the same learned query-conditioned null offset: r-N*b versus r-b',
            same_capacity_and_auxiliary_supervision=True,no_runtime_reference_images=True,no_attention_operator_novelty_claim=True,
            no_unseen_answer_value_claim=True,does_not_establish_reasoning_composition=True,all_test_answer_values_supported_in_training=True,
            cache_precision_verification='Failed CPU merge442411 retained; the separately frozen precision dispatcher verifies descriptive hidden reductions without changing native states or binding head gates',
            no_claim_final_nonlinear_residual_is_additive=True,auxiliary_only_updates_predictor=True,main_losses_update_predictor_and_core=True,
            no_claim_calibration_is_exact_or_immune_to_coadaptation=True,
            training_length_count_correlation='K0..8 cross-length consistency; K9..15 same-length; K16 identical pair with zero regularizer and retained CE slots',
            primary_exact='Complete stripped ASCII integer equals gold, no nonterminal special tokens, native EOS within four tokens; all failures remain in denominator',
            native_first_token_nll='Raw first-token NLL for dev tie-break only; it cannot distinguish10..16',
            raw_logit_storage='FP32 archives verified exact lossless promotions of FP16 native logits',
            matched_control='Identical1060224 parameters, ordinary and auxiliary data/order, initial bytes,73440scene and73440auxiliary presentations,4590updates',
            bootstrap='K-stratified intact-family intervals condition on these fitted seeds; pooling cannot rescue either seed',
            path_penalty='Diagnostic only, never optimized',local_visual_atoms_may_recur=True,
            loss_audit='All logged causal-position/scene/pair/group arithmetic and source-bound routing; unsaved intermediate core/messages/targets are not reconstructed'))
    save(out/'analysis.json',analysis)
    lines=['# V13 independent native evaluation','',f'Audit passed. Both-seed primary: **{primary}**. Both-seed practical: **{practical}**.','',
        'Both methods use the same trainable core and null predictor, full-answer CE, residual consistency and training-only null calibration. Centered subtracts N times the predicted projected null mean; offset subtracts it once. Neither uses reference images at inference.','',
        '| Condition | Seed | N32 | N64 | N64 K0–8 | N64 K9–15 | N64 K16 | N64 K9–16 | Selected step |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    fmt=lambda value:f"{value['correct']}/{value['n']} ({100*value['accuracy']:.1f}%)"
    for condition in CONDITIONS:
        for seed in SEEDS:
            run=runs[f'{condition}_s{seed}'];m=run['metrics'];cells=[m['cells'][f'main_test_N{n}'] for n in (32,64)]
            cells.extend(m['partitions']['N64'][k] for k in ('K0_8','K9_15','K16','K9_16'))
            lines.append(f'| {condition} | {seed} | '+' | '.join(fmt(v) for v in cells)+f" | {run['selected_step']} |")
    lines+=['','| Seed | N64 gain | Family bootstrap95%CI | N32 change | Primary | Practical |','|---:|---:|---:|---:|---|---|']
    for seed in SEEDS:
        d=decisions[str(seed)];ci=intervals['results']['all']['N64'][str(seed)]['difference_ci95_pp']
        lines.append(f"| {seed} | {d['n64_difference_pp']:.2f}pp | [{ci[0]:.2f},{ci[1]:.2f}]pp | {d['n32_difference_pp']:.2f}pp | {d['primary']} | {d['practical']} |")
    lines+=['','Primary requires centered to gain at least 7/136 N64 answers and lose at most 6/136 N32 answers in each seed. Practical additionally requires centered N32 ≥123/136, N64 ≥109/136, and N64 K9–16 ≥52/64.',
        '', 'All 272 fresh contexts per run remain in the denominator, including malformed or truncated outputs. Every answer value 0–16 occurs in training. The test measures length extrapolation.',
        '', 'Independent checks bind 27 prior exclusions, 1,782 ordinary training scenes, 918 pairs, 177,120 valid target positions, 972 null groups with 24 ordered occurrences each, 73,440 auxiliary presentations, five development checkpoints, and raw native argmax/EOS/NLL.',
        '', 'Initial losses and gradients match at zero U/C2; subsequent gradients may differ. The report audits recorded losses and source-bound graph routing, without reconstructing unsaved intermediate messages or targets. Calibration may coadapt with the core.',
        '', 'Final acceptance also requires the separate audit of all selected checkpoints and complete GPU-allocation accounting. No reasoning-composition or new attention-operator claim follows from this comparison.',
        '', '[Every count/length and numeral-prefix partition, parse/EOS statistics, losses and provenance](analysis.json).']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return dict(passed=True,source_sha256=frozen,analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),
        report_file=str(out/'REPORT.md'),primary_both_seeds=primary,practical_both_seeds=practical,vision_milestone_gate=primary and practical)


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-test','--check-source',action='store_true');mode.add_argument('--check-data',action='store_true')
    mode.add_argument('--check-manifests',action='store_true');mode.add_argument('--runs','--runs4',nargs=4,type=Path)
    parser.add_argument('--output',type=Path);args=parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),
        'All reporter/data/self-test work requires CPU Slurm')
    started=time.perf_counter();label='self_test' if args.self_test else 'manifest_check' if args.check_manifests else 'data_check' if args.check_data else 'report'
    out=args.output or OUT/f'{label}_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False);frozen=snapshot(out)
    if args.self_test:summary=dict(passed=True,source_sha256=frozen,tests=self_test())
    elif args.check_data or args.check_manifests:
        import torch
        torch.set_num_threads(4);data=audit_data();audited=data['summary'];cache_audit=None
        if args.check_data:
            from transformers import AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained(str(MODEL),use_fast=False,local_files_only=True)
            cache,cache_audit=verify_cache(CACHE,data,tokenizer);pairing=get_pairing(data,cache);save(out/'pairing.json',pairing)
            audited=dict(audited,cache_audit=cache_audit,pairing_object_sha256=objsha(pairing))
        save(out/'data_audit.json',audited)
        summary=dict(passed=True,source_sha256=frozen,data_bindings=data['summary']['data_bindings'],cache_audit=cache_audit,
            audit_file=str(out/'data_audit.json'),audit_sha256=sha(out/'data_audit.json'),manifest_only=args.check_manifests)
        if args.check_data:summary.update(pairing_file=str(out/'pairing.json'),pairing_sha256=sha(out/'pairing.json'),pairing_object_sha256=objsha(pairing))
    else:summary=report_runs(args.runs,out,frozen)
    need(sources()==frozen,'Reporter source changed during execution');summary.update(seconds=time.perf_counter()-started,slurm_job_id=os.environ['SLURM_JOB_ID'])
    save(out/'summary.json',summary);(out/'INDEX.md').write_text('# V13 independent CPU audit\n\n[Summary](summary.json) · [Sources](source_hashes.json)\n')
    print(json.dumps(dict(passed=True,summary_file=str(out/'summary.json'),seconds=summary['seconds'])),flush=True)


if __name__=='__main__':main()

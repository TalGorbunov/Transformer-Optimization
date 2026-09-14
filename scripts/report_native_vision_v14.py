"""Independent V14 audit of exact-bank training and frozen-core null distillation.

Only final core4590/student8000 endpoints are eligible. Native full-answer
scoring and the intact-family experiment remain separate from calibration.
Every execution, including data/source checks, requires CPU Slurm.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
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
from scripts import report_native_vision_v13 as v13
from scripts import stage_native_vision_v10_data as stage
from scripts import stage_native_vision_v14_test as fresh_stage
from scripts import merge_native_vision_v13_precision as precision
need,read,sha,objsha,save,close,bind,ledger,tensor_info=(prior.need,prior.read,prior.sha,prior.objsha,prior.save,prior.close,prior.bind,prior.ledger,prior.tensor_info)
verify_source,stored_native_logits=prior.verify_source,prior.stored_native_logits
DATA,MODEL=prior.DATA,prior.MODEL
OUT=REPO/'outputs/native_aggregation_vlm/v14'
CKPT=Path('/mnt/ckpts/gabriele/gnn_transformer/native_aggregation_vlm_v14')
CACHE=DATA/'v13_null_features/feature_cache.json'
CONDITIONS=('centered','offset')
SEEDS=(18,19)
PARTITIONS=v13.PARTITIONS
OWN=('scripts/report_native_vision_v14.py','slurm/native_vision_v14_report.sbatch',
     'scripts/train_native_vision_v14.py','slurm/native_vision_v14_train_check.sbatch',
     'slurm/native_vision_v14_train_profile.sbatch','slurm/native_vision_v14_train.sbatch',
     'tests/test_native_vision_v14_training.py')
POLICY=dict(
    protocol='v14_exact_bank_then_frozen_core_distillation',native_arm='parallel',conditions=['centered','offset'],seeds=[18,19],
    epochs=40,pair_slots_per_epoch=918,scene_slots_per_epoch=1836,unique_training_scenes=1782,
    scene_presentations=73440,pair_presentations=36720,target_positions=177120,batch_size=16,pairs_per_batch=8,steps=4590,
    lr=.001,warmup=50,final_lr=.00001,weight_decay=0.,clip_norm=1.,core_optimizer='AdamW_core_only',
    rank=96,hidden_size=3584,core_parameters=1041600,predictor_parameters=18624,parameters=1060224,
    merge='sum',post_activation='silu',native_dtype='torch.float16',branch_dtype='torch.float32',predictor_dtype='torch.float32',
    correction_coefficients={'centered':'actual_scene_N','offset':1},
    core_null_mean='live_FP64_mean_over24_occurrences_cast_FP32',core_reference_query_and_messages_fully_differentiable=True,
    core_predictor_present=False,core_auxiliary_loss_present=False,consistency_coefficients={'centered':1.,'offset':1.},
    consistency_epsilon=1e-6,ce_reduction='mean_over_scenes_of_mean_over_valid_native_target_positions',
    consistency_reduction='mean_over_pairs_of_mean_over_corresponding_strict_prefix_positions',
    denominator='detached_squared_L2_of_identical_frozen_global_state_plus_fixed_epsilon',
    pair_order='persistent_random.Random(seed)_fresh_canonical_pair_permutation_each_epoch',within_pair_order='registered_pair_sides_0_then_1',
    saturation_rule='retain54_N16_K16_same_SID_pairs_with_exact_zero_regularizer',path_loss_role='diagnostic_only_on_corrected_aggregate',
    reference_gradient_audit_steps=[1,2,32],core_then_student_no_overlap=True,
    student_groups=972,student_steps=8000,student_batch_size=64,student_presentations=512000,
    student_optimizer='AdamW_predictor_only',student_lr=.001,student_warmup=100,student_final_lr=.00001,
    student_weight_decay=0.,student_clip_norm=1.,student_initialization_seed_offset=20261110,student_order_seed_offset=20261111,
    student_initialization='independent_private_Torch_seed_then_zero_C2_weight_and_bias',
    student_order='persistent_random.Random(seed+20261111)_sorted_group_permutation_each_cycle_carried_batches',
    target_precompute_groups=972,target_precompute_batch_size=64,target_mean='FP64_occurrence_mean_cast_FP32',
    target_denominator='frozen_FP32_message_squared_mean_over24_times96_plus1e-6',
    student_loss='mean64_of_mean96_squared_error_divided_by_fixed_group_denominator',
    student_core_query_target_and_denominator_frozen=True,
    conversion_groups=972,conversion_unique_scenes=1782,conversion_ordinary_positions=4266,conversion_batch_scenes=32,
    conversion_no_VLM_head_accuracy_or_selection=True,
    full_answer_targets='canonical_native_numeral_tokens_plus_exactly_one_terminal_EOS',max_new_tokens=4,exact_requires_eos=True,
    reject_nonterminal_special_tokens=True,native_eos=[151645,151643],target_eos=151645,
    selection='fixed_final_core4590_student8000',dev_label='dev_final',dev_examples=64,dev_descriptive_only=True,test_examples=272,
    profile_core_steps=32,profile_student_steps=32,training_count_support=list(range(17)),development_count_support=list(range(16)),
    test_count_support=list(range(17)),maximum_training_N=16,test_N=[32,64],bootstrap_seed=20261109,bootstrap_replicates=10000,
    per_main_seconds_cap=2700,campaign_gpu_seconds_cap=16200,profile_seconds_cap=300)
PROFILE_CALLS=dict(core_training_updates=32,student_training_updates=32,target_precompute_groups=972,conversion_groups=972,
    conversion_ordinary_positions=4266,native_prefill_forwards=10,visual_forwards=10,same_state_native_shaped_head_replays=10,
    cached_state_head_projections=10,standalone_head_calls=20,natural_generations=0)


def sources():
    return {**v13.sources(),**fresh_stage.source_hashes(),**{name:sha(REPO/name) for name in OWN}}


def snapshot(out):
    frozen=sources();(out/'code').mkdir()
    for name,digest in frozen.items():
        target=out/'code'/name.replace('/','_');target.write_bytes((REPO/name).read_bytes());bind(target,digest)
    save(out/'source_hashes.json',frozen)
    return frozen


def learning_rate(step):
    return .001*step/50 if step<=50 else .00001+.00099*(1+math.cos(math.pi*(step-50)/4540))/2


def student_learning_rate(step):
    return .001*step/100 if step<=100 else .00001+.00099*(1+math.cos(math.pi*(step-100)/7900))/2


verify_cache=v13.verify_cache
numeric_metric=v13.numeric_metric
verify_software_timing=v13.verify_software_timing


def get_pairing(data,cache):
    parent=read(cache['parent_cache']['file']);value=v10.get_pairing(data,parent)
    return dict(value,protocol='v14_unchanged_v10_pairing_new_seeds',seeds=list(SEEDS),parent_cache=cache['parent_cache'])



def audit_data():
    # Revalidate the unchanged corpus as V10; never relabel its test predictions
    # as V14. Only its frozen training/dev rows are used in the new experiment.
    previous=v10.audit_data();manifest_path=DATA/'v14_fresh/main_manifest.json';fresh=read(manifest_path)
    need(fresh['schema_version']==1 and fresh['purpose']=='main' and fresh['dataset_root']==str(DATA/'v14_fresh')
         and fresh['data_seed']==fresh['fresh_test_seed']==20261108 and fresh['standard_downstream_resize']==392
         and fresh['training_dev_and_schedule_unchanged'] is True and fresh['training_pairing_unchanged'] is True
         and fresh['saturation_exceptions']==[],'V14 fresh data protocol differs')
    current,_,excluded=fresh_stage.prior_inputs()
    need(len(current['source_manifest_sha256'])==28 and fresh['source_manifest_sha256']==current['source_manifest_sha256']
         and fresh['protected_source_sha256']==current['protected_source_sha256'],'Exact28 prior exclusions changed')
    paths=[manifest_path]
    for field in ('audit','stage_plan','inventory','prior_inventory'):
        bind(fresh[field+'_file'],fresh[field+'_sha256']);paths.append(Path(fresh[field+'_file']))
    inventory=read(fresh['inventory_file'])
    need(inventory==current==read(fresh['prior_inventory_file']),'Published exclusion inventory differs')
    ledger(fresh['source_sha256']);ledger(fresh['generator_code_sha256']);ledger(fresh['protected_source_sha256'])
    archived=read(fresh['audit_file'])
    plan,planned,loaded_inventory=fresh_stage.load_plan(Path(archived['dry_plan_file']),fresh_stage.source_hashes())
    need(plan==read(fresh['stage_plan_file']) and loaded_inventory==inventory
         and sha(archived['dry_plan_file'])==archived['dry_plan_sha256'],'Published data changed its passed prospective dry plan')
    paths.extend([Path(archived['dry_plan_file']),Path(archived['dry_plan_file']).with_suffix('.sha256'),
                  Path(archived['dry_plan_file']).parent/'summary.json'])
    for field in ('samples','inventory','generator_checks'):
        bind(plan[field+'_file'],plan[field+'_sha256']);paths.append(Path(plan[field+'_file']))
    # Only normalize JSON integer-map keys in a private audit copy. Source files
    # and every sample's metadata remain byte-bound and unchanged.
    copied=json.loads(json.dumps(fresh))
    copied['splits']={cell:copied['splits'][cell] for cell in ('test_N32','test_N64')}
    for split in copied['splits'].values():
        observed=dict(sorted(Counter(r['gold'] for r in split['samples']).items()))
        need(split['gold_histogram']=={str(k):v for k,v in observed.items()},'Serialized K histogram differs')
        split['gold_histogram']=observed
    checked=json.loads(json.dumps(fresh_stage.audit_published(copied,excluded)))
    need(all(archived[key]==value for key,value in checked.items()),'Independent QA/image/semantic/extension audit differs')
    need(archived['all_published_records_match_dry_plan'] is True and archived['historical_content_reuse_count']==0
         and archived['saturation_exceptions']==[],'Fresh data completeness or no-reuse audit differs')
    published=[r for split in fresh['splits'].values() for r in split['samples']];by_sid={r['sid']:r for r in planned}
    need(len(by_sid)==len(published)==272 and set(by_sid)=={r['sid'] for r in published},'Dry/published context coverage differs')
    fields=('sid','question','gold','n_frames','content_sha256','target_character','target_room','pair_id','anchor_id',
            'parent_positions','anchor_positions','replica')
    for row in published:need(all(row[key]==by_sid[row['sid']][key] for key in fields),'Published test context differs from frozen dry selection')
    stage_path=Path(fresh['stage_audit_file']);finished=read(stage_path);paths.append(stage_path)
    need(finished['passed'] is True and finished['completed'] is True
         and finished['manifest_paths']=={'main':str(manifest_path)} and finished['manifest_sha256']=={'main':sha(manifest_path)}
         and finished['semantic_audit_file']==fresh['audit_file'] and finished['semantic_audit_sha256']==fresh['audit_sha256']
         and all(finished[key]==value for key,value in archived.items()),'Completed stage artifact does not bind the published manifest/audit')
    rows=[('main_'+cell,row) for cell in ('test_N32','test_N64') for row in fresh['splits'][cell]['samples']]
    need(Counter((r['n_frames'],r['gold']) for _,r in rows)==Counter({(n,k):8 for n in (32,64) for k in range(17)}),
         'Fresh native length/count support differs')
    bindings=dict(previous['summary']['data_bindings'])
    for path in paths:
        absolute=str(path.resolve());digest=sha(path)
        need(absolute not in bindings or bindings[absolute]==digest,'Conflicting independent data binding');bindings[absolute]=digest
    fresh_summary=dict(passed=True,fresh_contexts=272,test_families=136,prior_manifests=28,data_seed=20261108,
        all_qa_image_semantic_extensions_verified=True,historical_reuse_count=0,local_visual_atoms_may_recur=True,
        manifest_file=str(manifest_path),manifest_sha256=sha(manifest_path),audit_file=fresh['audit_file'],audit_sha256=fresh['audit_sha256'])
    summary=dict(previous['summary'],data_bindings=bindings,prior_manifests=28,fresh_data=fresh_summary,
        training_and_development_unchanged=True,test_dataset_root=str(DATA/'v14_fresh'),
        fresh_test_seed=20261108,old_v10_test_is_provenance_only=True)
    return dict(previous,summary=summary,fresh=fresh,test=rows)


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


def bootstrap(all_rows,replicates=10000,seed=20261109):
    import numpy as np
    rng=np.random.default_rng(seed);reference=all_rows['centered',18];groups=defaultdict(dict)
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


def expected_student_order(groups,seed):
    need(seed in SEEDS and len(groups)==972,'Unregistered student stream')
    canonical=sorted(groups);rng=random.Random(seed+20261111);result=[];cycle=0
    while len(result)<512000:
        cycle+=1;permutation=list(range(972));rng.shuffle(permutation)
        for slot in permutation:
            result.append(dict(cycle=cycle,slot=slot,group_id=canonical[slot]))
            if len(result)==512000:break
    need(Counter(r['cycle'] for r in result)==Counter({**{c:972 for c in range(1,527)},527:728}),
         'Student occurrence-cycle weights differ')
    return result



ACCOUNTING_FIELDS=v13.ACCOUNTING_FIELDS
TERMINAL_STATES=v13.TERMINAL_STATES

def parse_accounting(raw,*,all_user=False):
    rows=[];seen=set()
    for line in raw.splitlines():
        if not line.strip():continue
        values=line.split('|');need(len(values)==len(ACCOUNTING_FIELDS),'Unexpected raw Slurm accounting columns')
        row=dict(zip(ACCOUNTING_FIELDS,values))
        if all_user and (row['Partition']!='gpu' or not row['JobName'].startswith('v14_')):continue
        need(re.fullmatch('[0-9]+',row['JobIDRaw']) and re.fullmatch('[0-9]+(?:_[0-9]+)?',row['JobID'])
            and row['JobIDRaw'] not in seen and row['Partition']=='gpu' and row['JobName'].startswith('v14_'),
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
    need(rows,'No prior V14 GPU accounting records');return sorted(rows,key=lambda r:int(r['JobIDRaw']))

def verify_budget(budget):
    bind(budget['sacct_file'],budget['sacct_sha256']);bind(budget['all_user_sacct_file'],budget['all_user_sacct_sha256'])
    rows=parse_accounting(Path(budget['sacct_file']).read_text())
    all_rows=parse_accounting(Path(budget['all_user_sacct_file']).read_text(),all_user=True)
    need(rows==all_rows and sorted(budget['allocations'],key=lambda r:int(r['JobIDRaw']))==rows
        and len(budget['prior_jobs'])==len(rows) and len(set(budget['prior_jobs']))==len(rows)
        and set(budget['prior_jobs'])=={r['JobIDRaw'] for r in rows},'A prior V14 GPU allocation was omitted, duplicated or changed')
    total=sum(r['allocated_gpu_seconds'] for r in rows)
    need(total==budget['prior_allocated_gpu_seconds'] and budget['passed'] is True and budget['campaign_gpu_seconds_cap']==16200
        and budget['reserved_main_gpu_seconds']==10800 and budget['reserved_selected_audit_gpu_seconds']==300
        and budget['projected_campaign_gpu_seconds']==total+11100<=16200,'Independent allocation/reservation arithmetic differs')
    return dict(passed=True,prior_allocations=len(rows),failed_allocations=sum(r['state']!='COMPLETED' for r in rows),
        prior_allocated_gpu_seconds=total,all_user_campaign_coverage_verified=True)

def verify_release(config,data,frozen):
    binding=config['main_release'];bind(binding['file'],binding['sha256']);release=read(binding['file'])
    need(release['protocol']=='v14_exact_bank_distillation_main_release' and release['passed'] is True
        and release['plan_file']==config['plan_file'] and release['plan_sha256']==config['plan_sha256']
        and release['source_sha256']==config['source_sha256'] and release['report_source_sha256']==frozen,
        'Main release source/report/training-plan identity differs')
    need(release['per_main_seconds_cap']==2700 and release['main_block_gpu_seconds_cap']==16200,'Registered V14 resource cap differs')
    own=release['release_source_sha256'];need(set(own)=={'scripts/release_native_vision_v14_mains.py','slurm/native_vision_v14_main_release.sbatch'},
        'Unexpected measured-release source ledger')
    ledger(own);directory=Path(binding['file']).parent;need(read(directory/'source_hashes.json')==own,'Release source ledger copy differs')
    selected=release['selected_audit_freeze'];selected_directory=Path(selected['directory']).resolve()
    bind(selected_directory/'summary.json',selected['summary_sha256']);selected_summary=read(selected_directory/'summary.json')
    need(selected_summary['passed'] is True and selected_summary['tests_passed'] is True
        and selected_summary['protocol']=='all_selected_v14_native_checkpoint_audit' and selected_summary['no_model_loaded'] is True
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
    fresh=release['fresh_manifest'];expected_path=DATA/'v14_fresh/main_manifest.json'
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
    maximum_core=max(p['core_step_seconds_max_steady'] for p in profiles.values())
    maximum_student=max(p['student_step_seconds_max_steady'] for p in profiles.values())
    maximum_precompute=max(p['target_precompute_seconds'] for p in profiles.values())
    maximum_conversion=max(p['conversion_diagnostics_seconds'] for p in profiles.values())
    for key,value in (('shared_core_step_seconds_max_steady',maximum_core),
        ('shared_student_step_seconds_max_steady',maximum_student),('shared_target_precompute_seconds',maximum_precompute),
        ('shared_conversion_diagnostics_seconds',maximum_conversion)):
        need(math.isfinite(value) and value>0 and close(release[key],value),'Shared phase timing bound differs')
    for condition,profile in profiles.items():
        need(profile['hardware']['name']==software['hardware']['gpu'],'Training and reused native timing hardware differs')
        projection=release['projections'][condition]
        core_time=4590*maximum_core;student_time=8000*maximum_student;development=64*shared['16'];test=272*shared['64']
        projected=profile['model_and_features_load_seconds']+1.25*(core_time+student_time+maximum_precompute+maximum_conversion+development+test)+120
        expected=dict(core_training_seconds=core_time,student_training_seconds=student_time,
            target_precompute_seconds=maximum_precompute,conversion_diagnostics_seconds=maximum_conversion,
            development_seconds=development,test_seconds=test,projected_seconds=projected,
            shared_core_step_seconds_max_steady=maximum_core,shared_student_step_seconds_max_steady=maximum_student)
        need(projection['generation_seconds']==shared and projection['N32_uses_N64_bound'] is True
            and all(close(projection[k],v) for k,v in expected.items())
            and projection['passed'] is True and 0<projected<=2700,'Independent two-phase time projection differs or exceeds cap')
    budget=release['campaign_budget'];allocation_audit=verify_budget(budget)
    successful={'training_'+c:(str(p['slurm_job_id']),300) for c,p in profiles.items()}
    roles=budget['successful_artifacts']
    need(len(roles)==2 and {item['role'] for item in roles}==set(successful),'Both successful new training profiles required')
    for item in roles:
        job,cap=successful[item['role']];matches=[r for r in budget['allocations'] if job in (r['JobIDRaw'],r['JobID'])]
        need(len(matches)==1 and item['job_id']==job and item['seconds_cap']==cap and item['accounting_job_id']==matches[0]['JobIDRaw']
            and matches[0]['JobName']=='v14_train_profile' and matches[0]['state']=='COMPLETED' and matches[0]['ExitCode']=='0:0'
            and 0<matches[0]['allocated_gpu_seconds']<=cap,'Successful profile is not bound to its completed allocation/cap')
    need(len({item['accounting_job_id'] for item in roles})==2 and budget['max_concurrent_project_gpus']==4
        and budget['reused_v13_inputs_not_recharged'] is True and budget['reused_native_software_job_id']==str(software['slurm_job_id'])
        and all(r['JobName']=='v14_train_profile' and r['allocated_gpu_seconds']<=300 for r in budget['allocations']),
        'Pre-main campaign scope/caps or explicit V13 reuse accounting differs')
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
        exact_bank_core_gradients_and_frozen_core_student_paths_verified=True,
        no_cross_arm_first_gradient_equality_claim=True,reused_v13_inputs_not_recharged=True)


def initialized_state(seed):
    import torch
    from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
    from gnnformer.conditional_null_mean import ConditionalNullMean
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);core=ParallelLocalAggregation()
        torch.manual_seed(seed+20261110);predictor=ConditionalNullMean()
    need(sum(p.numel() for p in core.parameters())==1041600 and sum(p.numel() for p in predictor.parameters())==18624
         and bool((core.up.weight==0).all()) and bool((predictor.fc2.weight==0).all())
         and bool((predictor.fc2.bias==0).all()),'Independent core/student initialization differs')
    return {group:{key:tensor_info(value) for key,value in module.state_dict().items()}
            for group,module in (('branch',core),('predictor',predictor))}


def checkpoint_state(path,step,student_step,config,initialized,*,core_only=False):
    import torch
    saved=torch.load(path,map_location='cpu',weights_only=True)
    expected={'branch','step','config'} if core_only else {'branch','predictor','step','student_step','config'}
    need(set(saved)==expected and saved['step']==step and saved['config']==config
         and (core_only or saved['student_step']==student_step),'Fixed checkpoint phase/config/schema differs')
    result={}
    for group in ('branch',) if core_only else ('branch','predictor'):
        need(set(saved[group])==set(initialized[group]),'Checkpoint learned tensor coverage differs')
        for name,value in saved[group].items():
            need(value.dtype==torch.float32 and list(value.shape)==initialized[group][name]['shape']
                 and bool(torch.isfinite(value).all()),'Checkpoint tensor shape/dtype/finite contract differs')
        result[group]={name:tensor_info(value) for name,value in saved[group].items()}
    return result


def reduction_comparison(torch,stored,recomputed,*,fp64=False):
    # Exact tensor identity plus a declared bound only for recomputed reductions.
    # Independently executed CPU GEMMs are never native identity tests.
    rtol,atol=(1e-9,1e-10) if fp64 else (1e-6,1e-8)
    need(stored.shape==recomputed.shape and bool(torch.isfinite(stored).all()) and bool(torch.isfinite(recomputed).all())
         and torch.allclose(stored.double(),recomputed.double(),rtol=rtol,atol=atol),
         'Stored arithmetic exceeds the registered reduction precision bound')
    return dict(passed=True,exact=torch.equal(stored,recomputed),rtol=rtol,atol=atol,
                maximum_absolute_difference=float((stored.double()-recomputed.double()).abs().max()))


def verify_frozen_targets(config,freeze,cache,core_state):
    import torch
    path=Path(freeze['target_file']).resolve();metadata_path=Path(freeze['target_metadata_file']).resolve()
    data_directory=Path(config['data_directory']).resolve()
    need(path==data_directory/'frozen_targets.pt' and metadata_path==data_directory/'target_metadata.json',
         'Frozen targets outside the registered run data directory')
    bind(path,freeze['target_sha256']);bind(metadata_path,freeze['target_metadata_sha256'])
    blob=torch.load(path,map_location='cpu',weights_only=True);metadata=read(metadata_path)
    ids=sorted(cache['auxiliary_groups']);core_sha=objsha(core_state['branch'])
    shapes={'query':(972,96),'mean':(972,96),'messages':(972,24,96),'projected_messages':(972,24,96),
            'denominator':(972,),'projected_population_variance':(972,)}
    need(set(blob)=={'schema_version','group_ids','core_parameter_sha256','cache_sha256',*shapes}
         and blob['schema_version']==metadata['schema_version']==1 and blob['group_ids']==metadata['group_ids']==ids
         and blob['core_parameter_sha256']==metadata['core_parameter_sha256']==freeze['core_parameter_sha256']==core_sha
         and blob['cache_sha256']==metadata['cache_sha256']==config['cache_binding']['sha256']
         and metadata['file']==str(path) and metadata['sha256']==sha(path)
         and metadata['groups']==972 and metadata['occurrences']==24 and metadata['all_targets_frozen'] is True
         and metadata['mean_rule']==POLICY['target_mean'] and metadata['denominator_rule']==POLICY['target_denominator']
         and metadata['covariance_rule']=='FP32 projected messages; FP64 centered population trace with divisor24',
         'Frozen target group/core/cache/provenance differs')
    need(set(metadata['tensors'])==set(shapes),'Missing exact frozen-target tensor hashes')
    for name,shape in shapes.items():
        value=blob[name];dtype=torch.float64 if name=='projected_population_variance' else torch.float32
        need(tuple(value.shape)==shape and value.dtype==dtype and not value.requires_grad
             and bool(torch.isfinite(value).all()) and tensor_info(value)==metadata['tensors'][name],
             'Frozen target tensor identity/dtype/shape differs: '+name)
    messages=blob['messages'];projected=blob['projected_messages'].double()
    mean=messages.double().mean(1).float();denominator=messages.square().mean((1,2))+1e-6
    variance=(projected-projected.mean(1,keepdim=True)).square().sum((1,2))/24
    need(bool((blob['denominator']>=torch.tensor(1e-6,dtype=torch.float32)).all())
         and bool((blob['projected_population_variance']>=0).all()),'Invalid target normalization or covariance')
    arithmetic=dict(mean=reduction_comparison(torch,blob['mean'],mean),
                    denominator=reduction_comparison(torch,blob['denominator'],denominator),
                    projected_population_variance=reduction_comparison(torch,blob['projected_population_variance'],variance,fp64=True))
    return blob,dict(file=str(path),sha256=sha(path),metadata_file=str(metadata_path),metadata_sha256=sha(metadata_path),
        core_parameter_sha256=core_sha,groups=972,occurrences=24,all_tensor_hashes_exact=True,
        saved_occurrence_reductions_independently_verified=arithmetic,no_cross_device_linear_projection_identity_claim=True)


def verify_conversion(config,summary,cache,targets):
    import torch
    binding=summary['conversion_diagnostics'];path=Path(binding['file']).resolve();data_directory=Path(config['data_directory']).resolve()
    need(path==data_directory/'conversion_diagnostics.json' and Path(binding['raw_file']).resolve()==data_directory/'conversion_raw.pt',
         'Unexpected conversion artifact path')
    bind(path,binding['sha256']);bind(binding['raw_file'],binding['raw_sha256']);value=read(path)
    raw=torch.load(binding['raw_file'],map_location='cpu',weights_only=True)
    need(value['schema_version']==1 and value['condition']==config['condition']
         and value['raw_file']==binding['raw_file'] and value['raw_sha256']==binding['raw_sha256']
         and value['groups_count']==binding['groups']==972 and value['unique_scenes']==1782
         and value['ordinary_positions_count']==binding['ordinary_positions']==4266
         and all(value[k] is True for k in ('all_groups_and_positions_retained','no_VLM_or_head_calls','no_accuracy','no_fit','no_selection',
             'empirical_targets_are_training_bank_not_population','native_batch_route_differences_descriptive'))
         and value['ordinary_weight']=='unique_scene_and_valid_prefix_once' and binding['no_accuracy'] is True,
         'Conversion coverage or strictly descriptive scope differs')
    need(set(raw)=={'group_ids','predictions','projected_error','ordinary'} and raw['group_ids']==targets['group_ids'],
         'Conversion frozen-group order differs')
    for key in ('predictions','projected_error'):
        need(raw[key].shape==(972,96) and raw[key].dtype==torch.float32 and bool(torch.isfinite(raw[key]).all()),
             'Malformed conversion group tensor')
    group_rows=value['groups'];need(len(group_rows)==972,'Conversion group count differs')
    for i,(gid,row) in enumerate(zip(targets['group_ids'],group_rows)):
        group=cache['auxiliary_groups'][gid];prediction=raw['predictions'][i];target=targets['mean'][i];denom=float(targets['denominator'][i])
        mse=float((prediction-target).square().mean());error_squared=float(raw['projected_error'][i].double().square().sum())
        variance=float(targets['projected_population_variance'][i])
        need(row['group_id']==gid and row['question']==group['question'] and row['prefix_ids']==group['prefix_ids']
             and row['denominator']==denom and row['projected_population_variance']==variance
             and row['zero_variance']==(variance==0),'Conversion group identity/denominator/variance differs')
        expected={'mse':mse,'normalized_mse':mse/denom,'projected_error_squared':error_squared,
            'projected_error_norm':math.sqrt(error_squared),'target_norm':float(target.norm()),
            'prediction_norm':float(prediction.norm()),'query_norm':float(targets['query'][i].norm())}
        for key,x in expected.items():
            rtol,atol=(1e-9,1e-10) if key in ('projected_error_squared','projected_error_norm') else (1e-6,1e-8)
            need(math.isfinite(row[key]) and math.isclose(row[key],x,rel_tol=rtol,abs_tol=atol),'Conversion group arithmetic differs: '+key)
        ratio=row['projected_error_squared_over_variance']
        need(ratio is None if variance==0 else math.isclose(ratio,error_squared/variance,rel_tol=1e-9,abs_tol=1e-10),
             'Conversion dimensionless ratio differs')
    lookup={g['global_feature_id']:gid for gid,g in cache['auxiliary_groups'].items()}
    sids=sorted(cache['scenes']);rows=value['ordinary_positions'];cursor=0
    need(len(raw['ordinary'])==math.ceil(1782/32) and len(rows)==4266,'All unique conversion training batches required')
    maxima=dict(preactivation=0.,residual=0.,algebra=0.)
    for start,entry in zip(range(0,len(sids),32),raw['ordinary']):
        current=sids[start:start+32];offsets=[0];gids=[]
        for sid in current:
            scene=cache['scenes'][sid];offsets.append(offsets[-1]+len(scene['target_ids']))
            gids.extend(lookup[f] for f in scene['global_feature_ids'])
        count=offsets[-1]
        need(entry['sids']==current and entry['offsets']==offsets and entry['group_ids']==gids,
             'Conversion scene/prefix batch order differs')
        for name,width in (('bank_preactivation',96),('distilled_preactivation',96),('projected_error',96),
                           ('bank_residual',3584),('distilled_residual',3584)):
            need(entry[name].shape==(count,width) and entry[name].dtype==torch.float32 and bool(torch.isfinite(entry[name]).all()),
                 'Malformed conversion ordinary tensor: '+name)
        coefficients=[cache['scenes'][sid]['n_frames'] if config['condition']=='centered' else 1
                      for sid in current for _ in cache['scenes'][sid]['target_ids']]
        need(entry['coefficients'].shape==(count,) and entry['coefficients'].dtype==torch.int64
             and entry['coefficients'].tolist()==coefficients,'Conversion uses wrong actual cardinality coefficient')
        pre=(entry['distilled_preactivation']-entry['bank_preactivation']).double().norm(dim=-1)
        residual=(entry['distilled_residual']-entry['bank_residual']).double().norm(dim=-1)
        algebra=(entry['distilled_preactivation']-entry['bank_preactivation']+
                 entry['coefficients'][:,None]*entry['projected_error']).double().norm(dim=-1)
        for j,sid in enumerate(current):
            scene=cache['scenes'][sid]
            for position,index in enumerate(range(offsets[j],offsets[j+1])):
                row=rows[cursor];cursor+=1
                need(row['sid']==sid and row['n_frames']==scene['n_frames'] and row['gold']==scene['gold']
                     and row['position']==position and row['prefix_ids']==scene['target_ids'][:position] and row['group_id']==gids[index],
                     'Ordinary conversion row differs from exact source context/prefix')
                for key,tensor,label in (('preactivation_difference_norm',pre,'preactivation'),('residual_difference_norm',residual,'residual'),
                                         ('FP32_substitution_algebra_error_norm',algebra,'algebra')):
                    actual=float(tensor[index]);need(math.isclose(row[key],actual,rel_tol=1e-9,abs_tol=1e-10),
                                                   'Saved conversion FP64 norm differs: '+key)
                    maxima[label]=max(maxima[label],abs(row[key]-actual))
                for key,tensor in (('empirical_residual_norm',entry['bank_residual']),('distilled_residual_norm',entry['distilled_residual'])):
                    need(math.isclose(row[key],float(tensor[index].norm()),rel_tol=1e-6,abs_tol=1e-8),'Saved conversion FP32 norm differs')
    total_error=sum(r['projected_error_squared'] for r in group_rows);total_variance=sum(r['projected_population_variance'] for r in group_rows)
    ratio=total_error/total_variance if total_variance>0 else None;mean_loss=sum(r['normalized_mse'] for r in group_rows)/972
    need(cursor==4266 and binding['ratio_of_mean_projected_error_squared_to_mean_variance']==value['ratio_of_mean_projected_error_squared_to_mean_variance']==ratio
         and close(binding['mean_normalized_mse'],mean_loss) and close(value['mean_normalized_mse'],mean_loss),
         'Conversion summary changed its complete fixed-group estimand')
    return dict(**binding,passed=True,ordinary_scenes=1782,all_saved_arithmetic_verified=True,
        maximum_FP64_norm_differences=maxima,no_accuracy_or_population_claim=True)


def verify_history():
    final_path=OUT.parent/'v13/finalization/final_442481/final_acceptance.json'
    geometry_path=OUT.parent/'v13/calibration_geometry/geometry_442483/summary.json'
    final=read(final_path);geometry=read(geometry_path)
    need(final['completed'] is True and final['verification_passed'] is True and final['accepted_vision_milestone'] is False
         and geometry['passed'] is True and geometry['completed'] is True and geometry['no_fit'] is True and geometry['no_accuracy'] is True
         and geometry['total_groups']==3888 and geometry['total_ordinary_positions']==17064,'Prior failure/geometry provenance differs')
    for item in final['artifacts'].values():bind(item['path'],item['sha256'])
    for item in geometry['models']:bind(item['file'],item['sha256'])
    ledger(geometry['source_sha256'])
    return dict(finalization=dict(file=str(final_path),sha256=sha(final_path)),
                geometry=dict(file=str(geometry_path),sha256=sha(geometry_path)),no_prior_milestone_claim=True)


def check_student_row(row,denominators,target_norms):
    import numpy as np
    values={}
    for key in ('per_group_mse','per_group_denominator','per_group_loss','target_norms','prediction_norms'):
        value=np.asarray(row[key],dtype=np.float64)
        need(value.shape==(64,) and np.isfinite(value).all() and (value>=0).all(),'Malformed student group statistic: '+key)
        values[key]=value
    need(values['per_group_denominator'].tolist()==denominators
         and all(math.isclose(a,b,rel_tol=1e-6,abs_tol=1e-8) for a,b in zip(values['target_norms'],target_norms))
         and all(math.isclose(float(a),float(b),rel_tol=1e-6,abs_tol=1e-8) for a,b in zip(values['per_group_mse']/values['per_group_denominator'],values['per_group_loss']))
         and math.isclose(row['loss'],float(values['per_group_loss'].mean()),rel_tol=1e-6,abs_tol=1e-8),'Fixed student target/group loss reduction differs')
    need(all(row[k] is True for k in ('query_detached','target_detached','denominator_detached')),
         'Student query/target/normalization is not fixed')
    need(math.isfinite(row['gradient_norm']) and row['gradient_norm']>=0 and type(row['clipped']) is bool
         and row['clipped']==(row['gradient_norm']>1.) and math.isfinite(row['seconds']) and row['seconds']>0,
         'Student norm1 clipping or timing differs')


def gradient_records(directory,summary,initial):
    result={}
    for group,key,filename in (('branch','first_gradients','first_gradients.json'),
                               ('predictor','student_first_gradients','student_first_gradients.json')):
        path=Path(summary[key+'_file']).resolve();need(path==directory/filename,'Unexpected first-gradient artifact')
        bind(path,summary[key+'_sha256']);rows=read(path);need([r['step'] for r in rows]==[1,2],'Both initial phase gradient records required')
        for row in rows:
            need(set(row['pre_clip'])==set(initial[group]),'Phase gradient tensor coverage differs')
            for name,info in row['pre_clip'].items():
                need(info['shape']==initial[group][name]['shape'] and info['dtype']=='torch.float32'
                     and isinstance(info['sha256'],str) and re.fullmatch('[0-9a-f]{64}',info['sha256']),
                     'Phase gradient tensor shape/dtype/hash differs')
        result[group]=rows
    path=Path(summary['reference_gradient_audit_file']).resolve()
    need(path==directory/'reference_gradient_audit.json' and summary['reference_gradient_audit_passed'] is True,
         'Reference-gradient audit binding differs')
    bind(path,summary['reference_gradient_audit_sha256']);reference=read(path)
    need([r['step'] for r in reference]==[1,2,32],'Registered reference-route gradient checks missing')
    for row in reference:
        need(all(row[k] is True for k in ('passed','mean_and_messages_live','optimizer_buffers_untouched','cached_native_features_frozen'))
             and row['zero_up_expected']==(row['step']==1),'Reference route was detached or audit scope changed')
        norms=row['core_reference_gradients']
        need(set(norms)=={'reference_message','reference_query_weight','reference_local_weight'}
             and all(math.isfinite(v) and (v==0 if row['step']==1 else v>0) for v in norms.values()),
             'Required exact-bank reference gradient path was not exercised')
    exercised=summary['all_parameter_gradients_exercised']
    need(set(exercised)=={'branch','predictor'} and all(set(exercised[g])==set(initial[g])
         and all(value is True for value in exercised[g].values()) for g in exercised),'Learned phase parameters were not all exercised')
    zero=summary['zero_initialization_core_gradients']
    need(zero['passed'] is True and zero['up_exactly_zero'] is True and zero['residual_loss']==zero['path_loss']==0.
         and zero['all_residual_gradients_zero'] is True and zero['all_path_gradients_zero'] is True,
         'Initial zero-U consistency/path gradient gate differs')
    return dict(core=result['branch'],student=result['predictor'],reference=reference,
                first_gradients_need_not_match_across_conditions=True)


def loss_summary(core,student):
    blocks=[]
    for start in range(0,len(core),918):
        rows=core[start:start+918]
        blocks.append(dict(first_step=start+1,last_step=start+len(rows),
            means={k:sum(r[k] for r in rows)/len(rows) for k in ('loss','ce_loss','consistency_loss','path_loss')},
            clipping_fraction=sum(r['clipped'] for r in rows)/len(rows),valid_target_positions=sum(len(r['target_ids']) for r in rows)))
    students=[]
    for start in range(0,len(student),1000):
        rows=student[start:start+1000]
        students.append(dict(first_step=start+1,last_step=start+len(rows),mean_loss=sum(r['loss'] for r in rows)/len(rows),
            mean_raw_mse=sum(sum(r['per_group_mse']) for r in rows)/(64*len(rows)),
            clipping_fraction=sum(r['clipped'] for r in rows)/len(rows)))
    return dict(core_blocks=blocks,student_blocks=students,path_penalty_diagnostic_only=True,
        scope='Every logged loss reduction/order is audited; frozen targets are reconstructed from saved messages. Unsaved intermediate weights/predictions are not reconstructed.')


def verify_training_records(directory,data,cache,pairing_value,*,profile):
    import torch
    directory=Path(directory).resolve();config=read(directory/'config.json');summary=read(directory/'summary.json')
    condition,seed=config['condition'],config['seed'];runid=config['run_id'];steps=32 if profile else 4590;student_steps=32 if profile else 8000
    need(directory.parent==OUT and directory.name==runid and runid.startswith(f'{"profile" if profile else "run"}_{condition}_s{seed}_')
         and condition in CONDITIONS and seed in SEEDS and (not profile or seed==18) and config['arm']=='parallel'
         and config['profile'] is profile,'Unregistered V14 phase/run')
    need(config['policy']==POLICY and all(summary.get(k)==v for k,v in config.items())
         and config['consistency_coefficient']==1. and summary['passed'] is True and summary['completed'] is True
         and summary['computational_integrity_passed'] is True and summary['steps']==steps and summary['student_steps']==student_steps
         and summary['checkpoint_roundtrip_passed'] is True and summary['frozen_backbone_gradient_state_preserved'] is True,
         'V14 policy/config/phase completion differs')
    verify_source(directory,config['source_sha256']);bind(config['plan_file'],config['plan_sha256']);plan_path=Path(config['plan_file']);plan=read(plan_path)
    need(plan_path.with_suffix('.sha256').read_text().strip()==config['plan_sha256'] and plan['source_sha256']==config['source_sha256']
         and plan['policy']==POLICY and plan['profile_calls']==PROFILE_CALLS,'Training CPU source/plan differs')
    cpu=read(plan_path.parent/'summary.json')
    need(cpu['passed'] is True and cpu['plan_sha256']==config['plan_sha256'] and cpu['source_sha256']==config['source_sha256'],
         'Matching completed CPU phase-training check required')
    ledger(plan['artifact_bindings']);need(config['prior_result']==plan['prior_result']==verify_history(),'Prior failure/geometry binding changed')
    need(Path(plan['train_manifest']).resolve()==DATA/'v10_balanced/main_manifest.json'
         and Path(plan['schedule_file']).resolve()==DATA/'v10_balanced/schedule.json'
         and Path(plan['original_pairing_file']).resolve()==DATA/'v10_balanced/pairing.json'
         and plan['fresh_manifests']=={} and plan['fresh_test_bound_by_future_independent_release'] is True,
         'Training sources or future-independent fresh-test contract differs')
    for path,digest in data['summary']['data_bindings'].items():
        if path in plan['artifact_bindings']:need(plan['artifact_bindings'][path]==digest,'Training/data audit binding conflicts')
    binding=config['cache_binding'];need(binding==plan['cache_binding'] and Path(binding['file']).resolve()==CACHE,'Wrong immutable union cache')
    bind(binding['file'],binding['sha256']);need(all(config[k]==cache[k] for k in ('model','runtime','processor','native_dtypes')),
        'Native model/runtime/cache differs')
    need(config['pairing_file']==plan['pairing_file'] and config['pairing_sha256']==plan['pairing_sha256'],'Pairing binding differs')
    bind(plan['pairing_file'],plan['pairing_sha256']);need(read(plan['pairing_file'])==pairing_value and objsha(pairing_value)==plan['pairing_object_sha256'],
        'Independent ordinary pair construction differs')
    order=expected_order(pairing_value['pairs'],seed);student_order=expected_student_order(cache['auxiliary_groups'],seed)
    bind(directory/'presentations.json',config['presentations_sha256']);bind(directory/'student_presentations.json',config['student_presentations_sha256'])
    need(read(directory/'presentations.json')==order and objsha(order)==config['order_sha256']==plan['order_sha256'][str(seed)]
         and read(directory/'student_presentations.json')==student_order and objsha(student_order)==config['student_order_sha256']==plan['student_order_sha256'][str(seed)]
         and config['null_groups_sha256']==plan['null_groups_sha256']==objsha(cache['auxiliary_groups']),
         'Ordinary or persistent student presentation order differs')
    initial=initialized_state(seed)
    need(config['initialized']==initial and config['initialized_sha256']==objsha(initial)
         and config['student_initialization_seed']==seed+20261110 and config['student_initialized_sha256']==objsha(initial['predictor']),
         'Private seeded core/student initialization differs')
    need(Path(config['checkpoint_directory']).resolve()==CKPT/runid
         and Path(config['data_directory']).resolve()==DATA/'native_aggregation_vlm_v14'/runid,'Model/data storage roots differ')
    need(Path(summary['training_file']).resolve()==directory/'training.json','Unexpected core ledger path')
    bind(summary['training_file'],summary['training_sha256']);logs=read(summary['training_file']);need(len(logs)==steps,'Missing core optimizer updates')
    total_targets=identities=0;lookup={g['global_feature_id']:gid for gid,g in cache['auxiliary_groups'].items()}
    for step,row in enumerate(logs,1):
        positions=order[(step-1)*16:step*16];sids=[p['sid'] for p in positions]
        sequences=[cache['scenes'][sid]['target_ids'] for sid in sids];identity=[sids[i]==sids[i+1] for i in range(0,16,2)]
        reference_ids=[lookup[f] for sid in sids for f in cache['scenes'][sid]['global_feature_ids']]
        counts=[p['n_frames'] for p,sequence in zip(positions,sequences) for _ in sequence]
        need(row['step']==step and row['sids']==sids and row['pair_ids']==[p['pair_id'] for p in positions[::2]]
             and row['epochs']==[p['epoch'] for p in positions[::2]] and row['reference_group_ids']==reference_ids
             and row['reference_mean_live'] is True and row['position_n_frames']==counts
             and close(row['lr'],learning_rate(step),atol=1e-12) and close(row['main_loss'],row['loss']),
             'Core/reference presentation order/cardinality/live-mean or LR differs')
        need(all(positions[i]['pair_id']==positions[i+1]['pair_id'] and positions[i]['pair_side']==0
             and positions[i+1]['pair_side']==1 for i in range(0,16,2)),'A carried batch split or reversed a semantic pair')
        v10.check_loss_row(row,'consistency',sequences,identity)
        need(len(row['empirical_mean_norms'])==len(reference_ids) and all(math.isfinite(x) and x>=0 for x in row['empirical_mean_norms']),
             'Malformed live empirical-mean norm ledger')
        total_targets+=sum(map(len,sequences));identities+=sum(identity)
    if not profile:need(total_targets==177120 and identities==2160,'Weighted target or saturated identity totals differ')
    need(close(summary['training_seconds'],sum(r['seconds'] for r in logs),atol=1e-4)
         and close(summary['core_step_seconds_max_steady'],max(r['seconds'] for r in logs[4:]))
         and logs[0]['consistency_loss']==logs[0]['path_loss']==logs[0]['weighted_consistency_loss']==0.,
         'Core timing or initial zero-U losses differ')
    gradients=gradient_records(directory,summary,initial)
    need(summary['core_freeze_passed'] is True and Path(summary['core_freeze_file']).resolve()==directory/'core_freeze.json',
         'Missing phase-boundary core freeze')
    bind(summary['core_freeze_file'],summary['core_freeze_sha256']);freeze=read(summary['core_freeze_file'])
    need(freeze['schema_version']==1 and freeze['core_step']==steps and freeze['student_step']==student_steps
         and freeze['core_unchanged_after_distillation'] is True and freeze['core_unchanged_after_diagnostics'] is True
         and freeze['fixed_query_target_denominator_and_archive_unchanged'] is True
         and freeze['student_initialization_seed']==config['student_initialization_seed']
         and freeze['student_initialized_sha256']==config['student_initialized_sha256'], 'Frozen core/student phase contract differs')
    core_path=Path(freeze['core_checkpoint']).resolve();need(core_path==CKPT/runid/'core_frozen.pt','Unexpected fixed core checkpoint')
    bind(core_path,freeze['core_checkpoint_sha256']);core_state=checkpoint_state(core_path,steps,0,config,initial,core_only=True)
    need(objsha(core_state['branch'])==freeze['core_parameter_sha256'],'Core freeze learned parameter identity differs')
    phase=read(directory/'core_phase.json')
    need(phase==dict(completed=True,step=steps,core_checkpoint=str(core_path),core_checkpoint_sha256=sha(core_path),
         core_parameter_sha256=freeze['core_parameter_sha256'],training_sha256=summary['training_sha256'],
         reference_gradient_audit_sha256=summary['reference_gradient_audit_sha256']),'Core phase completion boundary differs')
    targets,target_audit=verify_frozen_targets(config,freeze,cache,core_state)
    need(summary['target_metadata']==read(freeze['target_metadata_file']),'Summary changed the frozen target metadata')
    need(Path(summary['distillation_file']).resolve()==directory/'distillation.json','Unexpected student ledger path')
    bind(summary['distillation_file'],summary['distillation_sha256']);student=read(summary['distillation_file'])
    need(len(student)==student_steps,'Missing student optimizer updates');indices={gid:i for i,gid in enumerate(targets['group_ids'])}
    denoms=targets['denominator'].tolist();norms=targets['mean'].norm(dim=-1).tolist()
    for step,row in enumerate(student,1):
        positions=student_order[(step-1)*64:step*64];ids=[p['group_id'] for p in positions]
        need(row['step']==step and row['group_ids']==ids and row['cycles']==[p['cycle'] for p in positions]
             and close(row['lr'],student_learning_rate(step),atol=1e-12),'Student update order, carried cycles or LR differs')
        check_student_row(row,[denoms[indices[gid]] for gid in ids],[norms[indices[gid]] for gid in ids])
    need(close(summary['student_training_seconds'],sum(r['seconds'] for r in student),atol=1e-4)
         and close(summary['student_step_seconds_max_steady'],max(r['seconds'] for r in student[4:]))
         and all(math.isfinite(summary[k]) and summary[k]>0 for k in ('target_precompute_seconds','conversion_diagnostics_seconds')),
         'Student/precompute/conversion measured phase times differ')
    phase_times=('setup_before_core_seconds','training_seconds','student_training_seconds','target_precompute_seconds',
                 'conversion_diagnostics_seconds','phase_log_io_seconds','final_validation_seconds')
    need(summary['setup_includes_model_and_feature_load'] is True
         and all(math.isfinite(summary[k]) and summary[k]>=0 for k in (*phase_times,'elapsed_seconds'))
         and summary['setup_before_core_seconds']>=summary['model_and_features_load_seconds']>0
         and sum(summary[k] for k in phase_times)<=summary['elapsed_seconds']+1e-4,
         'Recorded phase envelope or setup/load accounting differs')
    final_path=CKPT/runid/('profile.pt' if profile else 'final.pt');final=checkpoint_state(final_path,steps,student_steps,config,initial)
    need(final['branch']==core_state['branch'],'Student/native conversion changed frozen core bytes')
    conversion=verify_conversion(config,summary,cache,targets)
    audited=dict(condition=condition,arm='parallel',seed=seed,run_directory=str(directory),config_sha256=sha(directory/'config.json'),
        summary_sha256=sha(directory/'summary.json'),initialized_sha256=config['initialized_sha256'],presentations_sha256=config['presentations_sha256'],
        student_presentations_sha256=config['student_presentations_sha256'],student_order_sha256=config['student_order_sha256'],
        pairing_object_sha256=plan['pairing_object_sha256'],training_seconds=summary['training_seconds'],student_training_seconds=summary['student_training_seconds'],
        loss_diagnostics=loss_summary(logs,student),gradient_audit=gradients,core_freeze=freeze,core_phase_sha256=sha(directory/'core_phase.json'),
        frozen_targets=target_audit,conversion_diagnostics=conversion,fixed_core_through_student=True,
        verified_optimizer_log_rows=steps,verified_student_optimizer_log_rows=student_steps,verified_student_presentations=student_steps*64,
        verified_scene_presentations=steps*16,verified_pair_presentations=steps*8,verified_target_positions=total_targets,
        verified_saturated_identity_pair_presentations=identities,final_parameter_sha256=objsha(final),
        selected_step=steps,selected_student_step=student_steps,selected_checkpoint=str(final_path),
        selected_checkpoint_sha256=sha(final_path),selected_parameter_sha256=objsha(final))
    return audited,config,summary,logs


def verify_training_profile(item,condition,config,data,cache):
    import torch
    from scripts.cache_native_vision_v7_features import replay_metrics
    directory=Path(item['directory']).resolve();bind(directory/'summary.json',item['summary_sha256'])
    audited,pc,profile,logs=verify_training_records(directory,data,cache,get_pairing(data,cache),profile=True)
    need(profile['condition']==condition and profile['profile_calls']==PROFILE_CALLS and profile['no_dev_or_test_evaluation'] is True
         and all(profile[k]==config[k] for k in ('plan_sha256','source_sha256','model','runtime','processor','native_dtypes','hardware','cache_binding','pairing_sha256'))
         and 0<profile['elapsed_seconds']<=300,'Matched32-core/32-student profile/backend or resource cap differs')
    checkpoint=Path(profile['profile_checkpoint']).resolve()
    need(checkpoint==CKPT/profile['run_id']/'profile.pt' and profile['profile_checkpoint_sha256']==audited['selected_checkpoint_sha256'],
         'Profile combined fixed checkpoint differs')
    bind(checkpoint,profile['profile_checkpoint_sha256'])
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


def verify_run(directory,data,tokenizer,cache,pairing_value):
    directory=Path(directory).resolve();audited,config,summary,logs=verify_training_records(directory,data,cache,pairing_value,profile=False)
    need(summary['native_test_count']==272 and summary['native_dev_count']==64 and summary['dev_descriptive_only'] is True,
         'Final-only native evaluation coverage differs')
    selection=read(directory/'selection.json');entry=summary['selected']
    need(selection==dict(rule=POLICY['selection'],development=[entry],selected=entry,dev_descriptive_only=True)
         and summary['development']==[entry] and entry['step']==4590 and entry['student_step']==8000
         and set(p.name for p in directory.glob('dev_*.json'))=={'dev_final.json'},
         'Development selected a checkpoint or included an unregistered sweep')
    devpath=directory/'dev_final.json';need(Path(entry['dev_file']).resolve()==devpath,'Unexpected final descriptive dev artifact')
    bind(devpath,entry['dev_sha256']);dev=audit_eval(devpath,data['dev'],'parallel',tokenizer,config['condition'])
    need(dev['label']=='dev_final' and dev['exact_count']==entry['exact_count'] and close(dev['first_token_nll'],entry['nll']),
         'Descriptive development scores differ')
    need(entry['checkpoint']==audited['selected_checkpoint'] and entry['checkpoint_sha256']==audited['selected_checkpoint_sha256']
         and entry['parameter_sha256']==audited['selected_parameter_sha256'],'Final-only combined checkpoint identity differs')
    testpath=directory/'test.json';need(Path(summary['test_file']).resolve()==testpath,'Unexpected test artifact path')
    bind(testpath,summary['test_sha256']);test=audit_eval(testpath,data['test'],'parallel',tokenizer,config['condition'])
    need(test['label']=='test','Unexpected fresh-test evaluation label')
    audited.update(test_sha256=sha(testpath),evaluation_seconds=test['seconds'],
        native_test_model_forwards=sum(r['counters']['model'] for r in test['rows']),native_test_visual_forwards=272,
        verified_dev_evaluations=1,verified_test_examples=272,one_dev_only=True,dev_descriptive_only=True,
        development=entry,selection_rule=POLICY['selection'])
    return audited,test['rows'],config


def self_test():
    import ast
    import copy
    import torch
    tests=[];source=(REPO/'scripts/train_native_vision_v14.py').read_text();tree=ast.parse(source)
    for name,value in (('POLICY',POLICY),('PROFILE_CALLS',PROFILE_CALLS)):
        node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
        expected=eval(compile(ast.Expression(node.value),'<trainer literal>','eval'),{'__builtins__':{},'dict':dict,'list':list,'range':range})
        need(expected==value,'Independent literal phase policy/call inventory differs')
    tests.append('independent_literal_two_phase_policy_and_calls')
    metric=lambda n32,n64,large:dict(cells={'main_test_N32':{'correct':n32},'main_test_N64':{'correct':n64}},partitions={'N64':{'K9_16':{'correct':large}}})
    need(integer_decision(metric(123,109,52),metric(129,102,0))['practical'],'Exact integer threshold should pass')
    for a,b in ((metric(123,109,52),metric(129,103,0)),(metric(123,109,52),metric(130,102,0)),
                (metric(122,109,52),metric(122,102,0)),(metric(123,108,52),metric(123,101,0)),(metric(123,109,51),metric(123,102,0))):
        need(not integer_decision(a,b)['practical'],'A failed integer endpoint was rescued')
    tests.append('both_seed_integer_endpoint_boundaries')
    class Tokens:
        all_special_ids=[151645,151643,99]
        def decode(self,ids,skip_special_tokens=False):
            table={0:'0',1:'1',2:'2',8:'١',9:' ',99:'',151645:'<eos>',151643:'<eot>'}
            return ''.join(table[t] for t in ids if not skip_special_tokens or t not in self.all_special_ids)
    tokenizer=Tokens()
    need(parse_answer([1,0,151645],tokenizer)['parsed']==10 and parse_answer([1,0,151645],tokenizer)['completed']
         and parse_answer([1,99,151645],tokenizer)['parsed'] is None and parse_answer([8,151645],tokenizer)['parsed'] is None
         and not parse_answer([1,0,9,9],tokenizer)['completed'],'Whole ASCII integer/native EOS/special-token contract differs')
    tests.append('native_full_integer_and_unsanitized_special_tokens')
    groups={str(i):{} for i in range(972)};order=expected_student_order(groups,18)
    need(len(order)==512000 and [r['cycle'] for r in order[960:1024]]==[1]*12+[2]*52
         and len({r['group_id'] for r in order[:972]})==972 and order[-1]['cycle']==527,
         'Student coverage/carried cycle boundary differs')
    first=list(range(972));random.Random(18+20261111).shuffle(first)
    need([r['slot'] for r in order[:972]]==first,'Student first permutation differs from its private registered RNG')
    del order;tests.append('512000_student_presentations_and_carried_cycles')
    row=dict(loss=.5,gradient_norm=2.,clipped=True,seconds=.1,per_group_mse=[1.]*64,per_group_denominator=[2.]*64,
             per_group_loss=[.5]*64,target_norms=[1.]*64,prediction_norms=[2.]*64,query_detached=True,target_detached=True,denominator_detached=True)
    check_student_row(row,[2.]*64,[1.]*64)
    for mutate in (lambda x:x['per_group_denominator'].__setitem__(0,1.),lambda x:x.update(clipped=False),lambda x:x.update(target_detached=False)):
        wrong=copy.deepcopy(row);mutate(wrong)
        try:check_student_row(wrong,[2.]*64,[1.]*64)
        except ValueError:pass
        else:raise AssertionError('Wrong student frozen target/loss/clipping accepted')
    tests.append('student_fixed_normalization_mean64_and_norm1')
    x=torch.tensor([[1.,2.],[1.,2.],[4.,5.]])
    mean=x.double().mean(0).float();need(mean.tolist()==[2.,3.],'Occurrence-weighted mean differs')
    reduction_comparison(torch,mean,mean)
    try:reduction_comparison(torch,mean,mean+.01)
    except ValueError:pass
    else:raise AssertionError('Large target reduction disagreement accepted')
    tests.append('saved_occurrence_mean_and_precision_bounds')
    all_rows={(c,s):[dict(gold=k,anchor_id=f'{k}_{i}',n_frames=n,exact=c=='centered')
        for k in range(17) for i in range(8) for n in (32,64)] for c in CONDITIONS for s in SEEDS}
    intervals=bootstrap(all_rows,replicates=50)
    need(intervals['seed']==20261109 and all(v['difference_ci95_pp']==[100.,100.] for part in intervals['results'].values()
         for cell in part.values() for v in cell.values()),'Paired family/condition/seed bootstrap draws differ')
    tests.append('shared_K_stratified_family_bootstrap')
    need(stored_native_logits(torch.tensor([1.,2.],dtype=torch.float16).float())
         and not stored_native_logits(torch.tensor([.1],dtype=torch.float32)),'Native FP16 promotion contract weakened')
    tests.append('raw_native_FP16_lossless_promotion')
    raw='41|41|v14_train_profile|gpu|COMPLETED|30|gres/gpu=1,gres/gpu:b200=1|0:0|s|s|e\n'
    raw+='42|42|v14_train_profile|gpu|FAILED|7|gres/gpu:b200=1|1:0|s|s|e\n'
    need(sum(r['allocated_gpu_seconds'] for r in parse_accounting(raw))==37,'Failed or typed GPU allocation omitted/doubled')
    for invalid in (raw+raw,raw.replace('FAILED','RUNNING')):
        try:parse_accounting(invalid)
        except ValueError:pass
        else:raise AssertionError('Duplicate/unfinished allocation accepted')
    tests.append('complete_failed_GPU_allocation_accounting')
    return dict(passed=True,tests=tests)


def report_runs(directories,out,frozen):
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(4);tests=self_test();data=audit_data();tokenizer=AutoTokenizer.from_pretrained(str(MODEL),use_fast=False,local_files_only=True)
    cache,cache_audit=verify_cache(CACHE,data,tokenizer);pairing=get_pairing(data,cache)
    save(out/'data_audit.json',dict(data['summary'],cache_audit=cache_audit));save(out/'pairing.json',pairing)
    audits={};rows={};configs={};releases=[]
    need(len(directories)==4 and len({Path(p).resolve() for p in directories})==4,'All four distinct registered V14 mains required')
    for directory in directories:
        audited,predictions,config=verify_run(directory,data,tokenizer,cache,pairing);key=audited['condition'],audited['seed']
        need(key not in audits,'Duplicate condition/seed');audits[key],rows[key],configs[key]=audited,predictions,config
        releases.append(verify_release(config,data,frozen));print(json.dumps(dict(audited_run=str(directory),native_examples=len(predictions))),flush=True)
    need(set(audits)=={(c,s) for c in CONDITIONS for s in SEEDS} and all(r==releases[0] for r in releases),'Missing run or inconsistent prospective release')
    parity={}
    for seed in SEEDS:
        a,b=configs['offset',seed],configs['centered',seed]
        need(all(a[k]==b[k] for k in ('initialized_sha256','student_initialized_sha256','student_initialization_seed',
            'presentations_sha256','order_sha256','student_presentations_sha256','student_order_sha256',
            'policy','plan_file','plan_sha256','source_sha256','cache_binding','pairing_sha256','null_groups_sha256',
            'model','runtime','processor','native_dtypes','prior_result')),'Matched conditions differ in initialization, data/order or optimizer policy')
        parity[str(seed)]=dict(passed=True,combined_initialization_equal=True,ordinary_and_student_orders_equal=True,
            same_fixed_phase_budgets=True,first_gradients_need_not_match=True)
    runs={f'{c}_s{s}':dict(audits[c,s],metrics=summarize(rows[c,s])) for c in CONDITIONS for s in SEEDS}
    decisions={str(s):integer_decision(runs[f'centered_s{s}']['metrics'],runs[f'offset_s{s}']['metrics']) for s in SEEDS}
    primary=all(d['primary'] for d in decisions.values());practical=all(d['practical'] for d in decisions.values());intervals=bootstrap(rows)
    analysis=dict(schema_version=1,passed=True,audit_passed=True,source_sha256=frozen,policy=POLICY,data=data['summary'],cache=cache_audit,
        pairing=dict(file=str(out/'pairing.json'),sha256=sha(out/'pairing.json'),object_sha256=objsha(pairing),pair_count=918,
            unique_training_scenes=1782,weighted_scenes_per_epoch=1836,target_positions_per_epoch=4428),release=releases[0],self_tests=tests,
        runs=runs,decisions=decisions,matched_initialization_and_orders=parity,primary_both_seeds=primary,practical_both_seeds=practical,
        vision_milestone_gate=primary and practical,post_checkpoint_verification='Separate mandatory all-final-checkpoint native audit remains required',
        pooled_fixed_two_seed_n64_difference_pp=sum(d['n64_difference_pp'] for d in decisions.values())/2,bootstrap=intervals,
        interpretation=dict(treatment='Cardinality scaling of the same empirical null mean during core fit, then frozen-core predictor distillation: r-N*b versus r-b',
            same_capacity_and_phase_budgets=True,no_runtime_reference_images=True,no_attention_operator_novelty_claim=True,
            no_unseen_answer_value_claim=True,does_not_establish_reasoning_composition=True,all_test_answer_values_supported_in_training=True,
            cache_precision_verification='V13 failed CPU merge442411 and explicit precision dispatcher retained unchanged; no native tensor or head gate modified',
            core_reference_messages_and_query_fully_differentiable=True,student_only_updates_predictor=True,fixed_core_through_student=True,
            empirical_training_bank_not_population_null=True,no_claim_final_nonlinear_residual_is_additive=True,
            training_length_count_correlation='K0..8 cross-length consistency; K9..15 same-length; K16 identical pair with zero regularizer and retained CE slots',
            primary_exact='Complete stripped ASCII integer equals gold, no nonterminal special tokens, native EOS within four tokens; all failures remain in denominator',
            native_first_token_nll='Descriptive only; no checkpoint selection and cannot distinguish10..16',
            raw_logit_storage='FP32 archives verified exact lossless promotions of FP16 native logits',
            matched_control='Same1060224 parameters,73440scene presentations,4590core updates,512000student group presentations and8000student updates',
            fixed_final_checkpoint='Only core4590/student8000; one64-example descriptive development pass',
            bootstrap='K-stratified intact-family intervals condition on these fitted seeds; pooling cannot rescue either seed',
            conversion='All972groups/4266ordinary prefixes retained; training-bank calibration and residual differences do not establish native accuracy causation',
            path_penalty='Diagnostic only, never optimized',local_visual_atoms_may_recur=True,
            loss_audit='All logged causal-position/scene/pair/group arithmetic, exact frozen artifacts and source-bound graph routing; unsaved intermediate weights/predictions are not reconstructed'))
    save(out/'analysis.json',analysis)
    lines=['# V14 independent native evaluation','',f'Audit passed. Both-seed primary: **{primary}**. Both-seed practical: **{practical}**.','',
        'Both arms train the core against an empirical24-occurrence null mean, then freeze it and distill the mean into the same predictor. Centered subtracts N times the projected mean; offset subtracts it once. Native inference uses N+1 streams without reference images.','',
        '| Condition | Seed | N32 | N64 | N64 K0–8 | N64 K9–15 | N64 K16 | N64 K9–16 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    fmt=lambda value:f"{value['correct']}/{value['n']} ({100*value['accuracy']:.1f}%)"
    for condition in CONDITIONS:
        for seed in SEEDS:
            m=runs[f'{condition}_s{seed}']['metrics'];cells=[m['cells'][f'main_test_N{n}'] for n in (32,64)]
            cells.extend(m['partitions']['N64'][k] for k in ('K0_8','K9_15','K16','K9_16'))
            lines.append(f'| {condition} | {seed} | '+' | '.join(fmt(v) for v in cells)+' |')
    lines+=['','| Seed | N64 gain | Family bootstrap95%CI | N32 change | Primary | Practical |','|---:|---:|---:|---:|---|---|']
    for seed in SEEDS:
        d=decisions[str(seed)];ci=intervals['results']['all']['N64'][str(seed)]['difference_ci95_pp']
        lines.append(f"| {seed} | {d['n64_difference_pp']:.2f}pp | [{ci[0]:.2f},{ci[1]:.2f}]pp | {d['n32_difference_pp']:.2f}pp | {d['primary']} | {d['practical']} |")
    lines+=['','Each seed must gain at least7/136 N64 answers and lose at most6/136 N32 answers. Practical additionally requires centered N32≥123/136,N64≥109/136,and N64K9–16≥52/64.',
        '', 'Only the fixed final core4590/student8000 checkpoint is evaluated. Development64 is descriptive. All272 fresh contexts remain in every denominator; all answer values0–16 occur in training.',
        '', 'Independent checks bind28prior exclusions,918pairs/177120target positions,972frozen null groups with24ordered occurrences,512000student presentations,all conversion prefixes,and raw native argmax/EOS/NLL. Initial core/student bytes and data orders match; first core gradients may differ between arms.',
        '', 'Final acceptance requires the separate audit of all four final checkpoints and complete GPU accounting. Empirical bank fit is not independent population calibration. No reasoning-composition or new attention-operator claim follows.',
        '', '[Every count/length partition, calibration diagnostic and source binding](analysis.json).']
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
    save(out/'summary.json',summary);(out/'INDEX.md').write_text('# V14 independent CPU audit\n\n[Summary](summary.json) · [Sources](source_hashes.json)\n')
    print(json.dumps(dict(passed=True,summary_file=str(out/'summary.json'),seconds=summary['seconds'])),flush=True)


if __name__=='__main__':main()

"""Independent raw-outcome and resource audit for learned-selection identity joins.

The fixed final checkpoint is the only evaluated endpoint. All arms, seeds,
contrasts and outputs are retained. Example-bootstrap intervals condition on
these two fitted seeds; success is not a claim of general aggregation bandwidth.
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
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts.stage_native_vision_v6_teacher import need,read,save,sha
from scripts import native_vision_v7_runtime as native
OUT=REPO/'outputs/native_aggregation_vlm/identity_join_learned'
MODES=('clip','sigmoid','softmax');SEEDS=(22,23)
NAMES=('Sandra','Mary','Michael','John','Daniel','Laura','Peter','Emma','Noah')
CELLS=('test_seen_N32','test_seen_N64','test_held_N32','test_held_N64')
POLICY = dict(protocol='identity_join_learned_selection_final_only', native_arm='parallel',
    conditions=['clip', 'sigmoid', 'softmax'], seeds=[22, 23], epochs=12,
    pair_slots_per_epoch=3024, scene_slots_per_epoch=6048, unique_training_scenes=6048,
    scene_presentations=72576, pair_presentations=36288, target_positions=161280,
    batch_size=16, pairs_per_batch=8, steps=4536, lr=.001, warmup=50,
    final_lr=.00001, weight_decay=0., clip_norm=1., optimizer='AdamW_core_only',
    rank=96, hidden_size=3584, parameters=1041697, merge='sum', post_activation='silu',
    payload_activation='tanh', native_dtype='torch.float16', branch_dtype='torch.float32',
    selection_weight_initial=0., selection_bias_initial=.5, zero_up_initialization=True,
    clip_rule='clamp(score,0,1)', sigmoid_rule='sigmoid(4*(score-.5))',
    softmax_rule='softmax_over_valid_items(4*(score-.5))',
    selection_from_current_prefix=True, selection_gradient='live',
    validity_mask='actual_item_existence_only', local_prompt='unaltered_global_question',
    local_supervision=False, origin_probes=0, reference_rows=0, predictor_parameters=0,
    consistency_coefficient=1., consistency_epsilon=1e-6,
    ce_reduction='mean_over_scenes_of_mean_over_valid_native_target_positions',
    consistency_reduction='mean_over_pairs_of_mean_over_corresponding_strict_prefix_positions',
    denominator='detached_squared_L2_of_identical_frozen_global_state_plus_fixed_epsilon',
    pair_order='persistent_random.Random(seed)_fresh_canonical_generation_pair_permutation_each_epoch',
    within_pair_order='N8_then_N16_same_contrast_and_variant', path_loss_coefficient=0.,
    selection='fixed_final_step4536', dev_label='dev_final', dev_examples=108,
    dev_descriptive_only=True, test_examples=432, target_eos=151645,
    native_eos=[151645, 151643], max_new_tokens=4, reject_nonterminal_special_tokens=True,
    full_answer='canonical_name_plus_native_EOS_surrounding_whitespace_only',
    profile_steps=32, profile_seed=22, profile_timing_seed=20261124,
    profile_generations=9, profile_max_model_calls=36, profile_vision_calls=9,
    profile_max_standalone_head_calls=36, profile_seconds_cap=300,
    per_main_seconds_cap=3300, campaign_gpu_seconds_cap=24000,
    bootstrap_seed=20261124, bootstrap_replicates=10000)


def close(a,b,*,atol=1e-5,rtol=1e-5):
    return math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(float(a),float(b),rel_tol=rtol,abs_tol=atol)


def bind(path,digest):
    need(sha(Path(path))==digest,'Bound artifact changed: '+str(path))


def score(tokenizer,ids,gold):
    need(type(ids) is list and 1<=len(ids)<=4 and all(type(t) is int and 0<=t<152064 for t in ids),'Invalid native sequence')
    completed=ids[-1] in (151645,151643)
    need(not any(t in (151645,151643) for t in ids[:-1]) and (completed or len(ids)==4),'Invalid native termination')
    body_ids=ids[:-1] if completed else ids
    body=tokenizer.decode(body_ids,skip_special_tokens=False)
    clean=not any(t in tokenizer.all_special_ids for t in body_ids)
    prediction=body.strip() if clean and body.strip() in NAMES else None
    return dict(answer_body=body,no_nonterminal_special_tokens=clean,prediction=prediction,
        parseable=prediction is not None,parsed_name_correct=prediction==gold,
        exact=completed and prediction==gold,completed=completed,truncated=not completed)


def criteria(counts):
    decisions=[]
    for seed in SEEDS:
        for regime in ('seen','held'):
            clip=counts[('clip',seed)]
            for control in ('sigmoid','softmax'):
                other=counts[(control,seed)];c32='test_'+regime+'_N32';c64='test_'+regime+'_N64'
                gain=clip[c64]-other[c64];loss=clip[c32]-other[c32]
                decisions.append(dict(seed=seed,regime=regime,control=control,clip_N32=clip[c32],clip_N64=clip[c64],
                    control_N32=other[c32],control_N64=other[c64],N64_additional_correct=gain,N32_additional_correct=loss,
                    N64_gain_at_least6=gain>=6,N32_loss_at_most3=loss>=-3,primary=gain>=6 and loss>=-3))
    primary=all(d['primary'] for d in decisions)
    practical_cells=[dict(seed=seed,cell=cell,correct=counts[('clip',seed)][cell],
        required=98 if 'seen' in cell else 87,
        passed=counts[('clip',seed)][cell]>=(98 if 'seen' in cell else 87)) for seed in SEEDS for cell in CELLS]
    return dict(primary=primary,absolute_accuracy_target=all(x['passed'] for x in practical_cells),practical=primary and all(x['passed'] for x in practical_cells),
                decisions=decisions,practical_cells=practical_cells,reasoning_composition_demonstrated=False)


def summarize(rows):
    result={}
    for cell in CELLS:
        selected=[r for r in rows if r['cell']==cell]
        need(len(selected)==108,'Exact108-row test cell required')
        triples=defaultdict(list)
        for row in selected:triples[row['contrast_id']].append(row)
        need(len(triples)==36 and all(len(x)==3 and {r['variant'] for r in x}=={0,1,2} for x in triples.values()),'Incomplete answer-changing triples')
        result[cell]=dict(examples=108,correct=sum(r['exact'] for r in selected),
            accuracy=sum(r['exact'] for r in selected)/108,
            completed=sum(r['completed'] for r in selected),truncated=sum(r['truncated'] for r in selected),
            parseable=sum(r['parseable'] for r in selected),parsed_name_correct=sum(r['parsed_name_correct'] for r in selected),
            complete_triples=36,all_three_correct=sum(all(r['exact'] for r in x) for x in triples.values()),
            by_name={name:dict(examples=sum(r['gold']==name for r in selected),
                correct=sum(r['gold']==name and r['exact'] for r in selected)) for name in NAMES})
        need(all(x['examples']==12 for x in result[cell]['by_name'].values()),'Native answer-name test balance differs')
    pairs=defaultdict(dict)
    for row in rows:pairs[(row['regime'],row['pair_id'])][row['n_frames']]=row
    need(len(pairs)==216 and all(set(v)=={32,64} for v in pairs.values()),'All test length pairs required')
    transitions={regime:{key:0 for key in ('both_correct','lost_at64','gained_at64','both_wrong')} for regime in ('seen','held')}
    for (regime,_),v in pairs.items():
        a,b=v[32]['exact'],v[64]['exact'];key='both_correct' if a and b else 'lost_at64' if a else 'gained_at64' if b else 'both_wrong'
        transitions[regime][key]+=1
    return dict(cells=result,paired_transitions=transitions)


def bootstrap(all_rows,replicates=10000,seed=20261124):
    """One shared family draw for every arm/seed/length; room-pair strata."""
    import numpy as np
    need(replicates==10000 and seed==20261124 and set(all_rows)=={(m,s) for m in MODES for s in SEEDS},'Fixed bootstrap protocol differs')
    reference=all_rows[('clip',22)];strata=defaultdict(set)
    for r in reference:strata[(r['regime'],tuple(r['room_pair']))].add(r['contrast_id'])
    need(len(strata)==6 and all(len(v)==12 for v in strata.values()),'Twelve complete families per room-pair required')
    tables={}
    for key,rows in all_rows.items():
        table={(r['regime'],r['contrast_id'],r['variant'],r['n_frames']):int(r['exact']) for r in rows}
        need(len(table)==432,'Bootstrap missing/duplicate context')
        tables[key]=table
    rng=np.random.default_rng(seed);draws={k:rng.integers(0,12,size=(replicates,12)) for k in sorted(strata)}
    distributions={}
    for key,table in tables.items():
        for regime in ('seen','held'):
            for n in (32,64):
                result=np.zeros(replicates,dtype=np.float64)
                for stratum,families in sorted(strata.items()):
                    if stratum[0]!=regime:continue
                    totals=np.asarray([sum(table[(regime,c,v,n)] for v in (0,1,2)) for c in sorted(families)])
                    result+=totals[draws[stratum]].sum(1)
                distributions[(key,regime,n)]=100*result/108
    intervals=[]
    for regime in ('seen','held'):
        for n in (32,64):
            for mode in MODES:
                a=sum(distributions[((mode,s),regime,n)] for s in SEEDS)/2
                intervals.append(dict(regime=regime,n_frames=n,mode=mode,kind='accuracy_percent_two_seed_mean',
                    lower=float(np.quantile(a,.025)),upper=float(np.quantile(a,.975))))
            for control in ('sigmoid','softmax'):
                a=sum(distributions[(('clip',s),regime,n)]-distributions[((control,s),regime,n)] for s in SEEDS)/2
                intervals.append(dict(regime=regime,n_frames=n,control=control,kind='clip_gain_percentage_points_two_seed_mean',
                    lower=float(np.quantile(a,.025)),upper=float(np.quantile(a,.975))))
    return dict(replicates=replicates,seed=seed,unit='complete_three_variant_two_length_family',strata='room_pair_within_regime',
        shared_draws_all_arms_and_seeds=True,conditional_on_fitted_two_seeds=True,training_seed_population_inference=False,intervals=intervals)


def expected_pairs(manifest,cache):
    groups={r['pair_id']:[r] for r in manifest['splits']['train_N8']['samples']}
    for r in manifest['splits']['train_N16']['samples']:groups[r['pair_id']].append(r)
    pairs=[]
    for pid,values in groups.items():
        need(len(values)==2,'A training variant lacks one length');a,b=values
        need(a['n_frames']==8 and b['n_frames']==16 and all(a[k]==b[k] for k in
             ('contrast_id','variant','question','gold','target_ids')),'Length pair changes the answer or variant')
        pairs.append(dict(pair_id=pid,contrast_id=a['contrast_id'],variant=a['variant'],question=a['question'],
            gold=a['gold'],sids=[a['sid'],b['sid']],n_frames=[8,16],target_ids=a['target_ids']))
    need(len(pairs)==3024 and [{k:p[k] for k in ('pair_id','sids','question','gold','target_ids')} for p in pairs]==cache['training_pairs'],
         'Canonical training pair inventory/order differs')
    return dict(schema_version=1,pairs=pairs,canonical_order='original_generation_first_occurrence',
                same_variant_only=True,pairs_per_epoch=3024,scenes_per_epoch=6048)


def expected_order(pairs,seed):
    rng=random.Random(seed);rows=[]
    for epoch in range(1,13):
        slots=list(range(3024));rng.shuffle(slots)
        for slot in slots:
            p=pairs[slot]
            for side,sid in enumerate(p['sids']):
                rows.append(dict(epoch=epoch,slot=slot,pair_id=p['pair_id'],contrast_id=p['contrast_id'],
                    variant=p['variant'],sid=sid,pair_side=side,n_frames=p['n_frames'][side]))
    need(len(rows)==72576,'Twelve complete paired epochs required');return rows


def learning_rate(step):
    need(1<=step<=4536,'Unregistered update index')
    return .001*step/50 if step<=50 else 1e-5+(.001-1e-5)*(1+math.cos(math.pi*(step-50)/4486))/2


def close(a,b,*,atol=1e-5,rtol=1e-5):
    return math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(float(a),float(b),rel_tol=rtol,abs_tol=atol)


def bind(path,digest):need(sha(Path(path))==digest,'Bound artifact changed: '+str(path))


def audit_data(summary_path):
    from scripts import stage_native_identity_join_learned as stage
    manifest=stage.verify_stage(summary_path);published=read(summary_path);bindings={}
    for p,h in [(str(Path(summary_path).resolve()),sha(summary_path)),(published['manifest_file'],published['manifest_sha256'])]:
        bind(p,h);bindings[p]=h
    for key in ('plan','parent_plan','samples','runtime_inputs','audit','render_cache'):
        p,h=manifest[key+'_file'],manifest[key+'_sha256'];bind(p,h);bindings[p]=h
    rows=read(manifest['samples_file']);parent=read(manifest['parent_plan_file'])
    parents=read(parent['samples_file']);cache=read(manifest['render_cache_file'])
    records={r['sid']:r for cell in manifest['splits'].values() for r in cell['samples']}
    need(len(rows)==len(records)==6588 and len(cache)==3016,'Complete independently registered corpus required')
    actual=stage.audit_published(rows,[records[r['sid']] for r in rows],cache,Path(published['dataset_root']),parents)
    need(actual==published['published_audit'] and actual['passed'],'Independent pixels/QA/semantic publication audit differs')
    views=read(manifest['runtime_inputs_file'])
    need(views['records']==[stage.runtime_view(records[r['sid']]) for r in rows],'Published model input views differ')
    counts={k:v['count'] for k,v in manifest['splits'].items()}
    need(counts==dict(train_N8=3024,train_N16=3024,dev_N16=108,test_seen_N32=108,test_seen_N64=108,
                     test_held_N32=108,test_held_N64=108),'Registered split matrix differs')
    states={r['sid']:r['states'] for r in rows}
    return dict(passed=True,manifest=manifest,states=states,data_bindings=bindings,
        counts=counts,all_images_QA_contrasts_input_views_verified=True,prior_binary_failure_unchanged=True)


def audit_eval(torch,path,expected,condition,seed,tokenizer,identity,weights,states):
    from scripts.analyze_native_identity_join_learned import analyze_capture
    value=read(path);rows=value['rows'];need(len(rows)==len(expected)==value['n'],'Evaluation coverage differs')
    bind(value['raw_manifest_file'],value['raw_manifest_sha256']);raw_manifest=read(value['raw_manifest_file'])
    need(raw_manifest['completed'] and raw_manifest['all_raw_retained_before_scoring'] and raw_manifest['n']==len(rows) and len(raw_manifest['rows'])==len(rows),
         'All raw outputs must precede scoring')
    results=[];geometry=[];inv=dict(model=0,visual=0,language=0,norm=0,head=0,broadcast=0,selection=0,probe_head=0)
    for item,row,(cell,sample) in zip(raw_manifest['rows'],rows,expected):
        need(row['cell']==cell and row['sid']==sample['sid'] and all(row[k]==sample[k] for k in
            ('n_frames','gold','target_ids','contrast_id','variant','pair_id','room_pair','trio','content_sha256')),
            'Prediction/sample metadata ownership differs')
        need(all(row[k]==v for k,v in item.items()),'Raw record changed during scoring')
        bind(row['raw_file'],row['raw_sha256']);raw=torch.load(row['raw_file'],map_location='cpu',weights_only=True)
        ids=raw['generated_ids'];t=len(ids);n=sample['n_frames'];logits=raw['raw_logits'];meta=raw['metadata']
        need(row['text']==raw['text']==tokenizer.decode(ids,skip_special_tokens=True) and row['raw_text']==raw['raw_text']==tokenizer.decode(ids,skip_special_tokens=False),'Unstripped native decoded text differs')
        need(ids==row['generated_ids'] and logits.shape==(t,152064) and logits.dtype==torch.float32
             and bool(torch.isfinite(logits).all()) and torch.equal(logits,logits.half().float())
             and logits.argmax(-1).tolist()==ids,'Native FP16 logits/unmasked argmax differs')
        scored=score(tokenizer,ids,sample['gold']);need(all(row[k]==v for k,v in scored.items()),'Independent whole-answer score differs')
        need(raw['completed']==scored['completed'] and raw['truncated']==scored['truncated'],'Native completion differs')
        target=tokenizer.encode(sample['gold'],add_special_tokens=False)+[151645]
        nll=float(torch.logsumexp(logits[0].double(),-1)-logits[0,target[0]].double())
        need(target==sample['target_ids'] and close(nll,row['first_token_nll'],atol=1e-10,rtol=1e-10)
             and row['first_token_correct']==(ids[0]==target[0]),'Native target/NLL differs')
        need(meta==row['metadata'] and meta['sid']==sample['sid'] and meta['question']==sample['question']
             and meta['local_prompt']==meta['global_prompt']==sample['question'] and meta['prefix_ids']==[]
             and meta['n_frames']==n and meta['row_count']==n+1 and meta['global_row']==n
             and meta['image_paths']==[r['path'] for r in sample['image_files']]
             and meta['image_sha256']==[r['sha256'] for r in sample['image_files']]
             and meta['native_identity_sha256']==identity and meta['selection_mode']==condition
             and raw['selection_mode']==condition,'Current input/model/mode ownership differs')
        expected_binding=dict(inputs=meta['input_identity'],metadata={k:meta[k] for k in ('arm','n_frames','row_kinds','row_count','global_row','local_elements','original_prompt_width','row_prompt_tokens','local_prompt','global_prompt','resize')})
        need(meta['question_sha256']==native.object_sha(sample['question']) and meta['scene_input_identity']==dict(sid=sample['sid'],question_sha256=native.object_sha(sample['question']),image_sha256=[im['sha256'] for im in sample['image_files']],prompt_width=meta['original_prompt_width'],input_identity_sha256=native.object_sha(expected_binding),native_identity_sha256=identity),'Independent original input/native digest differs')
        need(meta['resize']==392 and meta['row_kinds']==['local']*n+['global'] and meta['layout']['every_unpadded_row_exact'],
             'Native local/global row layout differs')
        policy=meta['generation']
        need(policy['max_new_tokens']==4 and policy['do_sample'] is False and policy['num_beams']==1
             and policy['native_eos_token_ids']==[151645,151643],'Native unrestricted greedy policy differs')
        counters=dict(model=t,visual=1,language=t,norm=t,head=t,broadcast=t,selection=t,probe_head=0)
        need(raw['counters']==row['counters']==counters and len(raw['captures'])==t,'One native call per emitted token required')
        need(len(raw['logit_records'])==len(meta['generation_position_ids'])==t and [r['top1_token_id'] for r in raw['logit_records']]==ids and all(r['native_dtype']=='torch.float16' for r in raw['logit_records']), 'Native per-token record ownership differs')
        for k,v in counters.items():inv[k]+=v
        if raw['captures']:
            relevant=[any(room in sample['room_pair'] and people for room,people in s['rooms'].items()) for s in states[sample['sid']]]
            need(len(relevant)==n and sum(relevant)==6,'Offline canonical relevance differs')
            for step,cap in enumerate(raw['captures']):
                diagnostic=analyze_capture(torch,cap,weights,condition,relevant,list(range(1,n+1)))
                if cell in CELLS:geometry.append(dict(cell=cell,sid=sample['sid'],seed=seed,token_index=step,
                    prefix_ids=ids[:step],emitted_id=ids[step],exact=scored['exact'],**diagnostic))
        results.append(dict(row,regime=sample['split'].removeprefix('test_'),**scored))
    need(value['exact_count']==sum(r['exact'] for r in results)
         and close(value['first_token_nll'],sum(r['first_token_nll'] for r in results)/len(results),atol=1e-10,rtol=1e-10),
         'Evaluation summary differs from every retained raw output')
    return dict(rows=results,geometry=geometry,inventory=inv,exact_count=value['exact_count'])


def source_hashes():
    from scripts import train_native_identity_join_learned_v2 as training
    own=('scripts/report_native_identity_join_learned_v2.py','scripts/analyze_native_identity_join_learned.py',
         'tests/test_native_identity_join_learned_analysis.py','tests/test_native_identity_join_learned_report_v2.py',
         'slurm/native_identity_join_learned_v2_report.sbatch')
    return {**training.sources(),**{n:sha(REPO/n) for n in own}}


def snapshot(out):
    frozen=source_hashes();(out/'source').mkdir()
    for n,h in frozen.items():
        p=out/'source'/n.replace('/','_');p.write_bytes((REPO/n).read_bytes());bind(p,h)
    save(out/'source_hashes.json',frozen);return frozen


def audit_config(directory,*,profile):
    from scripts import train_native_identity_join_learned_v2 as training
    directory=Path(directory).resolve();config=read(directory/'config.json');summary=read(directory/'summary.json')
    need(config['profile'] is profile and config['condition'] in MODES and config['seed'] in SEEDS
         and (not profile or config['seed']==22) and config['policy']==POLICY==training.POLICY
         and directory.parent==OUT/'training' and directory.name==config['run_id'],'Fit identity or fixed policy differs')
    need(all(summary.get(k)==v for k,v in config.items()) and summary['source_sha256']==training.sources(),
         'Run/source/config ownership differs')
    for n,h in config['source_sha256'].items():bind(directory/'code'/n.replace('/','_'),h)
    for k in ('passed','completed','computational_integrity_passed','checkpoint_roundtrip_passed',
              'final_endpoint_unchanged','frozen_backbone_gradient_state_preserved'):
        need(summary[k] is True,'Unfinished native fit gate: '+k)
    bind(config['plan_file'],config['plan_sha256']);plan=read(config['plan_file'])
    need(plan['policy']==POLICY and plan['source_sha256']==training.sources(),'CPU plan policy/source differs')
    cpu=read(Path(config['plan_file']).parent/'summary.json')
    need(cpu['passed'] and cpu['completed'] and cpu['plan_sha256']==config['plan_sha256'],'Complete CPU training proof required')
    for n,h in plan['source_sha256'].items():bind(Path(config['plan_file']).parent/'code'/n.replace('/','_'),h)
    for p,h in plan['runtime_bindings'].items():bind(p,h)
    for key in ('cache_binding','manifest','data_release','native_profile','native_software_plan','native_identity','native_identity_sha256'):
        need(config[key]==plan[key],'Run changed bound '+key)
    need(native.object_sha(config['native_identity'])==config['native_identity_sha256'],'Native identity digest differs')
    return config,summary,plan


def endpoint(torch,directory,config,summary,profile):
    from scripts.stage_native_vision_v10_features import tensor_info
    root=Path('/mnt/ckpts/gabriele/gnn_transformer/identity_join_learned/training')
    file=root/config['run_id']/('profile.pt' if profile else 'final.pt');step=32 if profile else 4536
    bind(summary['final_endpoint_file'],summary['final_endpoint_sha256']);proof=read(summary['final_endpoint_file'])
    need(Path(config['checkpoint_directory'])==file.parent and Path(proof['checkpoint'])==file and proof['step']==step,
         'Final endpoint path/step differs')
    bind(file,proof['checkpoint_sha256']);packet=torch.load(file,map_location='cpu',weights_only=True)
    need(set(packet)=={'branch','step','config'} and packet['step']==step and packet['config']==config,'Final actual tensor packet differs')
    weights=packet['branch'];table={k:tensor_info(v) for k,v in weights.items()}
    need(sum(v.numel() for v in weights.values())==1041697 and all(v.dtype==torch.float32 and bool(torch.isfinite(v).all()) for v in weights.values())
         and table==proof['before_evaluation']==proof['after_evaluation'] and native.object_sha(table)==proof['parameter_sha256'],
         'Actual saved/deployed endpoint table differs')
    need(all(proof[k] is True for k in ('checkpoint_matches_deployed_endpoint','core_versions_unchanged','native_versions_unchanged')),
         'Frozen native final endpoint changed')
    need(proof['native_weight_identity_before']==proof['native_weight_identity_after']==config['native_weight_identity'],
         'Native head/norm changed at evaluation')
    bind(config['native_model_file'],config['native_model_sha256']);native_weights=torch.load(config['native_model_file'],map_location='cpu',weights_only=True)
    need(Path(config['native_model_file']).is_relative_to(root) and tensor_info(native_weights['norm_weight'])==config['native_identity']['norm_weight']
         and tensor_info(native_weights['head_weight'])==config['native_identity']['head_weight'],'Actual native weight copy differs')
    return weights,dict(checkpoint=str(file),checkpoint_sha256=proof['checkpoint_sha256'],parameter_sha256=proof['parameter_sha256'])


def audit_training(torch,directory,config,summary,plan,cache,manifest,profile):
    from scripts.stage_native_vision_v10_features import tensor_info
    from gnnformer.parallel_local_learned_selection import ParallelLocalLearnedSelection
    pairs=expected_pairs(manifest,cache);need(read(plan['pairing_file'])==pairs,'Independent pair reconstruction differs')
    order=expected_order(pairs['pairs'],config['seed']);bind(directory/'presentations.json',config['presentations_sha256'])
    need(read(directory/'presentations.json')==order and native.object_sha(order)==config['order_sha256'],
         'Actual paired training order differs')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config['seed']);core=ParallelLocalLearnedSelection(mode=config['condition'])
        expected={k:tensor_info(v) for k,v in core.state_dict().items()}
    need(config['initialized']==expected and config['initialized_sha256']==native.object_sha(expected),'Initial matched parameters differ')
    bind(config['initial_checkpoint'],config['initial_checkpoint_sha256']);initial=torch.load(config['initial_checkpoint'],map_location='cpu',weights_only=True)
    need(initial['seed']==config['seed'] and initial['step']==0 and {k:tensor_info(v) for k,v in initial['branch'].items()}==expected,
         'Saved actual initial parameters differ')
    # Recompute every consistency denominator from the actual frozen global
    # feature, rather than trusting the scalar recorded by the trainer.
    gids={fid for scene in cache['scenes'].values() for fid in scene['global_feature_ids']};denominators={}
    byfile=defaultdict(list)
    for fid in gids:byfile[cache['features'][fid]['file']].append(fid)
    for filename,ids in byfile.items():
        bind(filename,cache['features'][ids[0]]['file_sha256']);blob=torch.load(filename,map_location='cpu',weights_only=True)
        for fid in ids:
            item=cache['features'][fid];g=blob['states'][item['row']]
            need(blob['feature_ids'][item['row']]==fid and tensor_info(g)['sha256']==item['state_sha256'], 'Actual frozen global feature differs')
            denominators[fid]=float(g.float().square().sum()+1e-6)
        del blob
    bind(summary['training_file'],summary['training_sha256']);logs=read(summary['training_file']);steps=32 if profile else 4536
    need(len(logs)==steps==summary['steps'],'Fixed optimizer update coverage differs');total_targets=0
    for step,row in enumerate(logs,1):
        batch=order[(step-1)*16:step*16];sids=[r['sid'] for r in batch];targets=[cache['scenes'][sid]['target_ids'] for sid in sids]
        lengths=list(map(len,targets));offsets=[0]
        for length in lengths:offsets.append(offsets[-1]+length)
        flat=[t for ids in targets for t in ids];prefixes=[ids[:i] for ids in targets for i in range(len(ids))]
        need(row['step']==step and row['sids']==sids and row['pair_ids']==[r['pair_id'] for r in batch[::2]]
             and row['epochs']==[r['epoch'] for r in batch[::2]] and close(row['lr'],learning_rate(step),atol=1e-12,rtol=1e-12)
             and row['target_ids']==flat and row['prefix_ids']==prefixes,'Training order/causal targets/schedule differs')
        need(row['scene_lengths']==lengths and row['scene_offsets']==offsets and row['pair_lengths']==lengths[::2],
             'Native loss segmentation differs')
        pos=row['position_losses'];ce=pos['ce'];reg=pos['consistency'];plengths=lengths[::2]
        need(len(ce)==len(flat) and len(reg)==sum(plengths) and len(row['pair_position_denominator'])==len(reg)
             and all(math.isfinite(x) and x>=0 for x in ce+reg) and all(math.isfinite(x) and x>=1e-6 for x in row['pair_position_denominator']),
             'Malformed native per-position objective')
        ce_scene=[sum(ce[a:b])/(b-a) for a,b in zip(offsets,offsets[1:])];perpair=[];i=0
        for length in plengths:perpair.append(sum(reg[i:i+length])/length);i+=length
        need(len(row['per_scene_ce'])==16 and len(row['per_pair_consistency'])==8
             and all(close(a,b) for a,b in zip(ce_scene,row['per_scene_ce']))
             and all(close(a,b) for a,b in zip(perpair,row['per_pair_consistency']))
             and close(sum(ce_scene)/16,row['ce_loss']) and close(sum(perpair)/8,row['consistency_loss'])
             and row['consistency_coefficient']==1. and row['weighted_consistency_loss']==row['consistency_loss']
             and close(row['loss'],row['ce_loss']+row['consistency_loss']),'Scene/pair objective reductions differ')
        expected_denominators=[denominators[fid] for sid in sids[::2] for fid in cache['scenes'][sid]['global_feature_ids']]
        need(len(expected_denominators)==len(reg) and all(close(a,b,atol=1e-4,rtol=1e-6) for a,b in zip(expected_denominators,row['pair_position_denominator'])),'Consistency denominator differs from actual frozen global states')
        need(len(row['residual_difference_norms'])==len(reg) and all(close(a*a/b,c,atol=1e-5) for a,b,c in
             zip(row['residual_difference_norms'],row['pair_position_denominator'],reg)),'Residual norm/denominator arithmetic differs')
        nitems=[cache['scenes'][sid]['n_frames'] for sid in sids];expected_valid=[n for n,length in zip(nitems,lengths) for _ in range(length)]
        need(row['valid_item_count_by_position']==expected_valid and row['valid_gate_count']==sum(expected_valid)
             and row['padding_messages_exact_zero'] and row['closed_message_coordinates_exact_zero']
             and 0<=row['gate_min']<=row['gate_mean']<=row['gate_max']<=1 and 0<=row['payload_max_abs']<=1
             and math.isfinite(row['gradient_norm']) and row['gradient_norm']>=0
             and row['clipped']==(row['gradient_norm']>1),'Training validity/gates/gradients differ')
        total_targets+=len(flat)
    need(profile or total_targets==161280,'Full native target count differs')
    need(logs[0]['consistency_loss']==0 and close(summary['training_seconds'],sum(r['seconds'] for r in logs),atol=1e-4)
         and summary['first_four_step_seconds']==sum(r['seconds'] for r in logs[:4])
         and summary['step_seconds_max_steady']==max(r['seconds'] for r in logs[4:]),'Initial residual or timing sums differ')
    bind(summary['first_gradients_file'],summary['first_gradients_sha256']);grads=read(summary['first_gradients_file'])
    need([x['step'] for x in grads]==[1,2],'First two gradient records required')
    for item in grads:
        need(set(item['pre_clip'])==set(expected) and all(item['pre_clip'][k]['shape']==expected[k]['shape']
             and item['pre_clip'][k]['dtype']=='torch.float32' for k in expected),'Core gradient table differs')
    bind(summary['gradient_isolation_file'],summary['gradient_isolation_sha256']);isolated=read(summary['gradient_isolation_file'])
    need([x['step'] for x in isolated]==[1,2,32] and all(all(x[k] is True for k in
         ('passed','score_gradient_present','payload_gradient_present','frozen_features','no_local_label_input')) for x in isolated),
         'Live selector/payload or frozen-state gradient proof differs')
    exemptions=['selection_bias'] if config['condition']=='softmax' else []
    exercised=summary['all_parameter_gradients_exercised']
    need(summary['gradient_nonzero_exemptions']==exemptions and set(exercised)==set(expected)
         and all(v is True or k in exemptions for k,v in exercised.items()) and exercised['selection_weight']
         and summary['zero_initialization_core_gradients']==dict(passed=True,up_exactly_zero=True,residual_loss=0.,all_residual_gradients_zero=True),
         'Nonzero learned gradient or zero-initialization proof differs')
    return dict(updates=steps,scene_presentations=16*steps,target_positions=total_targets,paired_order_verified=True,
        native_loss_reductions_verified=True,first_preclip_gradients=grads[0]['pre_clip']),logs


def audit_profile(torch,directory,plan_path,tokenizer):
    from scripts import cache_native_identity_join_features as worker
    config,summary,plan=audit_config(directory,profile=True)
    need(config['plan_file']==str(Path(plan_path).resolve()),'Profile belongs to another plan')
    cache=worker.verify_cache(plan['cache_binding']['file'],ancestors=False);manifest=read(plan['manifest']['file'])
    training,logs=audit_training(torch,Path(directory),config,summary,plan,cache,manifest,True)
    weights,final=endpoint(torch,Path(directory),config,summary,True);timing=summary['native_timing']
    need(timing['passed'] and timing['no_dev_or_test_evaluation'] and summary['no_dev_or_test_evaluation']
         and not (Path(directory)/'test.json').exists() and not (Path(directory)/'dev_final.json').exists(),
         'Timing profile cannot contain development/test outcomes')
    cases=read(plan['timing_cases_file'])['cases'];need(len(cases)==len(timing['rows'])==9,'Nine timing cases required')
    head_rows=0
    for row,case in zip(timing['rows'],cases):
        need(all(row[k]==case[k] for k in ('parent_sid','contrast_id','variant','n_frames')) and row['no_accuracy_scoring'],
             'Timing case selection differs')
        for kind in ('raw','natural_raw'):bind(row[kind+'_file'],row[kind+'_sha256'])
        archive=torch.load(row['raw_file'],map_location='cpu',weights_only=True);result=archive['result']
        original=torch.load(row['natural_raw_file'],map_location='cpu',weights_only=True)['result']
        need(original['generated_ids']==result['generated_ids'] and torch.equal(original['raw_logits'],result['raw_logits']),
             'Natural output changed before native replay')
        ids=result['generated_ids'];t=len(ids);raw=result['raw_logits'];replay=archive['replay_global_logits']
        need(t==row['generated_tokens']==row['standalone_head_calls'] and 1<=t<=4 and raw.shape==replay.shape==(t,152064)
             and raw.dtype==torch.float32 and replay.dtype==torch.float16 and torch.equal(raw,raw.half().float())
             and raw.argmax(-1).tolist()==ids and len(result['captures'])==t,'Timing native output/capture differs')
        need(result['metadata']['input_identity']==case['input_identity'] and result['metadata']['selection_mode']==config['condition']
             and result['metadata']['native_identity_sha256']==plan['native_identity_sha256'],'Timing native ownership differs')
        need(row['counters']==result['counters']==dict(model=t,visual=1,language=t,norm=t,head=t,broadcast=t,selection=t,probe_head=0),
             'Timing work differs')
        need(not any(i in (151645,151643) for i in ids[:-1]) and (ids[-1] in (151645,151643) or t==4), 'Native profile termination differs')
        for i,m in enumerate(row['native_head_replays']):
            a,b=raw[i].double(),replay[i].double();tv=float(.5*(a.softmax(-1)-b.softmax(-1)).abs().sum());top=int(a.argmax())==int(b.argmax())
            need(m['passed'] and m['top1_equal']==[top] and top and tv<=.02 and len(m['tv'])==1
                 and close(tv,m['tv'][0],atol=1e-12,rtol=1e-9),'Independent native head replay differs');head_rows+=1
        need(len(row['native_head_replays'])==t,'Timing head replay coverage differs')
        actual=row['preprocessing_seconds']+(row['generation_seconds']+row['replay_seconds']+row['archive_seconds'])*4/t
        need(close(actual,row['four_token_seconds_bound'],atol=1e-12,rtol=1e-12),'Four-token full-work bound differs')
    need(timing['natural_generations']==timing['vision_calls']==9 and timing['origin_probe_calls']==0
         and timing['native_model_calls']==timing['standalone_head_calls']==head_rows<=36,'Profile invocation inventory differs')
    for n in (16,32,64):need(timing['T'+str(n)]==max(r['four_token_seconds_bound'] for r in timing['rows'] if r['n_frames']==n),'Native length timing bound differs')
    projected=summary['setup_before_training_seconds']+1.25*(summary['first_four_step_seconds']+
        4532*summary['step_seconds_max_steady']+108*timing['T16']+216*timing['T32']+216*timing['T64'])+120
    return dict(directory=str(Path(directory).resolve()),summary_sha256=sha(Path(directory)/'summary.json'),
        condition=config['condition'],job_id=config['slurm_job_id'],training=training,endpoint=final,head_rows=head_rows,
        timing_inputs=dict(setup=summary['setup_before_training_seconds'],first4=summary['first_four_step_seconds'],steady=summary['step_seconds_max_steady'],T16=timing['T16'],T32=timing['T32'],T64=timing['T64']),
        projected_seconds=projected,passed=projected<=3300,formula='setup+1.25*(first4+4532*steady+108*T16+216*T32+216*T64)+120'),config


JOB_CAPS={'v19_selection_software':300,'identity_join_learned_feature_profile':300,
    'identity_join_learned_feature_shard':1200,'identity_join_learned_train_profile':300,'identity_join_learned_main':3300}
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}


def accounting_rows(raw):
    from datetime import datetime
    rows=[];events=[]
    for line in raw.splitlines():
        if not line.strip():continue
        f=line.split('|');need(len(f)==9,'Scheduler accounting schema differs')
        job,name,partition,state,exit_code,elapsed,tres,start,end=f
        if name not in JOB_CAPS or partition!='gpu':continue
        state=state.split()[0].rstrip('+');need(state in TERMINAL,'A campaign allocation is not terminal')
        values=dict(v.split('=',1) for v in tres.split(',') if '=' in v)
        typed=[int(v) for k,v in values.items() if k.startswith('gres/gpu:')]
        ngpu=int(values['gres/gpu']) if 'gres/gpu' in values else sum(typed);seconds=int(elapsed)
        need(ngpu in (0,1) and (not typed or sum(typed)==ngpu) and 0<=seconds<=JOB_CAPS[name],
             'Allocated GPU count or per-job resource cap differs')
        rows.append(dict(job_id=job,name=name,state=state,exit_code=exit_code,elapsed_seconds=seconds,
            gpus=ngpu,gpu_seconds=ngpu*seconds,start=start,end=end))
        if ngpu and start!=end:
            a,b=datetime.fromisoformat(start),datetime.fromisoformat(end);need(a<b,'Invalid allocation times');events.extend([(a,1),(b,-1)])
    need(len({r['job_id'] for r in rows})==len(rows),'Duplicate allocated job counted')
    total=sum(r['gpu_seconds'] for r in rows);concurrent=maximum=0
    for _,change in sorted(events):concurrent+=change;need(concurrent>=0,'Invalid concurrency intervals');maximum=max(maximum,concurrent)
    need(concurrent==0 and maximum<=4 and total<=24000,'Whole campaign resource cap exceeded')
    need(sum(r['gpu_seconds'] for r in rows if r['name']=='v19_selection_software')<=900,'Software subcampaign cap exceeded')
    return dict(passed=True,rows=rows,allocated_gpu_seconds=total,maximum_concurrent_gpus=maximum,
        failed_and_zero_allocations_retained=True,generic_and_typed_gpu_not_double_counted=True)


def accounting(out):
    cmd=['sacct','-X','-u',os.environ['USER'],'--starttime=2026-09-01','--parsable2','--noheader',
         '--format=JobIDRaw,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,Start,End']
    result=subprocess.run(cmd,capture_output=True,text=True,check=True);p=out/'all_user_sacct.psv';p.write_text(result.stdout)
    return dict(accounting_rows(result.stdout),file=str(p),sha256=sha(p),command=cmd)


def verified_data_report(path,frozen):
    summary=read(path);need(summary['passed'] and summary['completed'] and summary['phase']=='check'
        and summary['source_sha256']==frozen,'Completed independent source/data check required')
    for p,h in summary['data_bindings'].items():bind(p,h)
    for n,h in frozen.items():bind(Path(path).parent/'source'/n.replace('/','_'),h)
    return summary


def check(args,out,frozen):
    tests=[]
    for name in ('tests/test_native_identity_join_learned_report_v2.py','tests/test_native_identity_join_learned_analysis.py'):
        result=subprocess.run([sys.executable,str(REPO/name)],capture_output=True,text=True)
        log=out/(Path(name).stem+'.log');log.write_text(result.stdout+result.stderr)
        need(result.returncode==0,'Independent analysis/report tests failed: '+name)
        tests.append(dict(file=name,passed=True,log_file=str(log),log_sha256=sha(log)))
    data=audit_data(args.data_summary)
    return dict(passed=True,completed=True,phase='check',source_sha256=frozen,data_bindings=data['data_bindings'],
        counts=data['counts'],all_images_QA_contrasts_input_views_verified=True,tests=tests,no_model_loaded=True)


def release(args,out,frozen):
    import torch
    from transformers import AutoTokenizer
    from scripts.stage_native_vision_v6_teacher import MODEL
    from scripts import train_native_identity_join_learned_v2 as training
    torch.set_num_threads(4);checked=verified_data_report(args.data_report,frozen);plan=training.verify_plan(args.plan)
    need(checked['data_bindings'].get(plan['manifest']['file'])==plan['manifest']['sha256'],'Independent full dataset binding differs')
    tokenizer=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,trust_remote_code=True)
    need(len(args.profile_directories)==3,'Three matched training profiles required')
    profiles={};projections={}
    for directory in args.profile_directories:
        proof,config=audit_profile(torch,directory,args.plan,tokenizer);mode=config['condition']
        need(mode not in profiles,'Repeated profile arm');profiles[mode]=dict(directory=proof['directory'],summary_sha256=proof['summary_sha256'])
        projections[mode]=proof
    need(set(profiles)==set(MODES),'Missing profile arm')
    pooled={k:max(p['timing_inputs'][k] for p in projections.values()) for k in ('setup','first4','steady','T16','T32','T64')}
    pooled_seconds=pooled['setup']+1.25*(pooled['first4']+4532*pooled['steady']+108*pooled['T16']+216*pooled['T32']+216*pooled['T64'])+120
    for p in projections.values():p.update(unpooled_projected_seconds=p['projected_seconds'],projected_seconds=pooled_seconds,pooled_timing_inputs=pooled,passed=pooled_seconds<=3300)
    budget=accounting(out)
    need(not any(r['name']=='identity_join_learned_main' for r in budget['rows']),'A final main was attempted before release')
    consumed=[(p['job_id'],'identity_join_learned_train_profile') for p in projections.values()]
    cache=read(plan['cache_binding']['file'])
    feature_profile=read(Path(cache['profile_directory'])/'summary.json');consumed.append((feature_profile['slurm_job_id'],'identity_join_learned_feature_profile'))
    for shard in cache['shards']:consumed.append((read(Path(shard['directory'])/'summary.json')['slurm_job_id'],'identity_join_learned_feature_shard'))
    software_report=read(plan['native_profile']['file']);software_analysis=read(software_report['analysis_file'])
    consumed.append((read(software_analysis['gpu_summary_file'])['slurm_job_id'],'v19_selection_software'))
    for job,name in consumed:
        matches=[r for r in budget['rows'] if r['job_id']==job]
        need(len(matches)==1 and matches[0]['name']==name and matches[0]['state']=='COMPLETED' and matches[0]['exit_code']=='0:0' and matches[0]['gpus']==1,'Consumed GPU prerequisite lacks a successful exact allocation')
    budget.update(reserved_main_gpu_seconds=19800,projected_total_with_main_reservations=budget['allocated_gpu_seconds']+19800)
    passed=all(x['passed'] for x in projections.values()) and budget['projected_total_with_main_reservations']<=24000
    budget['passed']=passed
    result=dict(protocol='identity_join_learned_selection_main_release',passed=passed,completed=True,
        plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),source_sha256=training.sources(),report_source_sha256=frozen,
        data_release=dict(file=str(Path(args.data_report).resolve()),sha256=sha(args.data_report)),manifest=plan['manifest'],
        profiles=profiles,projections=projections,campaign_budget=budget,per_main_seconds_cap=3300,campaign_gpu_seconds_cap=24000,
        main_matrix=[dict(mode=m,seed=s) for s in SEEDS for m in MODES],no_outcome_dependent_release=True)
    save(out/'release.json',result)
    need(passed,'Fixed main timing/resource release failed; profile evidence retained')
    return dict(passed=True,completed=True,phase='release',source_sha256=frozen,release_file=str(out/'release.json'),
        release_sha256=sha(out/'release.json'),projections=projections,campaign_budget=budget)


def report(args,out,frozen):
    import torch
    from transformers import AutoTokenizer
    from scripts.stage_native_vision_v6_teacher import MODEL
    from scripts import cache_native_identity_join_features as worker
    torch.set_num_threads(4);checked=verified_data_report(args.data_report,frozen)
    need(len(args.run_directories)==6,'All six fixed final runs are required')
    tokenizer=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,trust_remote_code=True)
    manifest_path=next(p for p in checked['data_bindings'] if p.endswith('/main_manifest.json'))
    manifest=read(manifest_path);states={r['sid']:r['states'] for r in read(manifest['samples_file'])}
    all_rows={};runs=[];geometries=[];release_identity=None;initial_by_seed={};first_gradient_by_seed={}
    for directory in args.run_directories:
        directory=Path(directory).resolve();config,summary,plan=audit_config(directory,profile=False);key=(config['condition'],config['seed'])
        need(key not in all_rows and config['manifest']==dict(file=manifest_path,sha256=sha(manifest_path)),'Repeated arm/seed or different corpus')
        bind(config['main_release']['file'],config['main_release']['sha256']);release_value=read(config['main_release']['file'])
        need(release_value['passed'] and release_value['report_source_sha256']==frozen and release_value['plan_sha256']==config['plan_sha256'],
             'Run did not consume the exact independent release')
        if release_identity is None:release_identity=config['main_release']
        need(config['main_release']==release_identity,'Six runs did not share one fixed release')
        cache=worker.verify_cache(plan['cache_binding']['file'],ancestors=False)
        training,logs=audit_training(torch,directory,config,summary,plan,cache,manifest,False)
        weights,final=endpoint(torch,directory,config,summary,False)
        if config['seed'] not in initial_by_seed:initial_by_seed[config['seed']]=config['initialized']
        need(config['initialized']==initial_by_seed[config['seed']],'Matched arms have different initialized parameters')
        # Clip and sigmoid have equal initial values/derivatives. Softmax is a different normalized map.
        if config['condition'] in ('clip','sigmoid'):
            first_gradient_by_seed.setdefault(config['seed'],[]).append(training['first_preclip_gradients'])
        selected=summary['selected'];selection=read(directory/'selection.json')
        need(selection==dict(rule='fixed_final_step4536',selected=selected,development=[selected],dev_descriptive_only=True)
             and summary['development']==[selected] and selected['step']==4536
             and selected['checkpoint']==final['checkpoint'] and selected['checkpoint_sha256']==final['checkpoint_sha256']
             and selected['parameter_sha256']==final['parameter_sha256'],'Final endpoint was selected using development accuracy')
        bind(selected['dev_file'],selected['dev_sha256']);bind(summary['test_file'],summary['test_sha256'])
        dev_expected=[('dev_N16',r) for r in manifest['splits']['dev_N16']['samples']]
        test_expected=[(cell,r) for cell,part in manifest['splits'].items() if cell.startswith('test_') for r in part['samples']]
        dev=audit_eval(torch,selected['dev_file'],dev_expected,*key,tokenizer,plan['native_identity_sha256'],weights,states)
        test=audit_eval(torch,summary['test_file'],test_expected,*key,tokenizer,plan['native_identity_sha256'],weights,states)
        need(dev['exact_count']==selected['exact_count'] and summary['native_dev_count']==108 and summary['native_test_count']==432,
             'Final development/test inventory differs')
        all_rows[key]=test['rows'];geometries.extend(test['geometry']);stats=summarize(test['rows'])
        runs.append(dict(mode=key[0],seed=key[1],directory=str(directory),summary_sha256=sha(directory/'summary.json'),
            job_id=config['slurm_job_id'],endpoint=final,training=training,dev_correct=dev['exact_count'],
            statistics=stats,dev_inventory=dev['inventory'],test_inventory=test['inventory']))
    need(set(all_rows)=={(m,s) for m in MODES for s in SEEDS},'Incomplete six-fit matrix')
    need(all(len(v)==2 and v[0]==v[1] for v in first_gradient_by_seed.values()),'Matched clip/sigmoid first native gradients differ')
    counts={(r['mode'],r['seed']):{c:v['correct'] for c,v in r['statistics']['cells'].items()} for r in runs}
    decision=criteria(counts);intervals=bootstrap(all_rows);budget=accounting(out)
    for run in runs:
        matches=[x for x in budget['rows'] if x['job_id']==run['job_id']]
        need(len(matches)==1 and matches[0]['state']=='COMPLETED' and matches[0]['exit_code']=='0:0' and matches[0]['gpus']==1,
             'Final run did not complete successfully within its allocation')
    save(out/'geometry.json',dict(scope='all executed test prefixes in all six models',rows=geometries,descriptive=True,
        no_local_labels_in_training_or_runtime=True,no_causal_zero_suppression_claim=True))
    save(out/'outcomes.json',dict(rows=[dict(mode=m,seed=s,**r) for (m,s),rows in all_rows.items() for r in rows]))
    analysis=dict(protocol='identity_join_learned_selection_result',passed=True,completed=True,source_sha256=frozen,
        runs=runs,criteria=decision,bootstrap=intervals,resources=budget,
        geometry_file=str(out/'geometry.json'),geometry_sha256=sha(out/'geometry.json'),
        outcomes_file=str(out/'outcomes.json'),outcomes_sha256=sha(out/'outcomes.json'),
        interpretation='Known learned gated set pooling; contentsensitive MMReD-derived task. Fixed six included images, nine familiar people; not a general relational or reasoning-composition proof.')
    save(out/'analysis.json',analysis)
    lines=['# Learned selection on the MMReD-derived identity join','',
        '| Method | Seed | Seen N32 | Seen N64 | Held N32 | Held N64 |','|---|---:|---:|---:|---:|---:|']
    for run in sorted(runs,key=lambda x:(x['seed'],MODES.index(x['mode']))):
        values=[str(run['statistics']['cells'][c]['correct'])+'/108' for c in CELLS]
        lines.append('| '+run['mode']+' | '+str(run['seed'])+' | '+' | '.join(values)+' |')
    lines.extend(['','Primary criterion: **'+str(decision['primary'])+'**. Practical criterion: **'+str(decision['practical'])+'**.',
        '',analysis['interpretation'],'','All malformed and truncated outputs are included. Checkpoint selection was fixed-final only.',
        '','Allocated GPU-seconds including failures: '+str(budget['allocated_gpu_seconds'])+'.',
        '','[Full analysis](analysis.json) · [Every outcome](outcomes.json) · [Every test-prefix diagnostic](geometry.json)'])
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return dict(passed=True,completed=True,phase='report',source_sha256=frozen,analysis_file=str(out/'analysis.json'),
        analysis_sha256=sha(out/'analysis.json'),criteria=decision,resources=budget)


def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check',action='store_true');group.add_argument('--release',action='store_true');group.add_argument('--report',action='store_true')
    parser.add_argument('--data-summary',type=Path);parser.add_argument('--data-report',type=Path);parser.add_argument('--plan',type=Path)
    parser.add_argument('--profile-directories',type=Path,nargs='*',default=[]);parser.add_argument('--run-directories',type=Path,nargs='*',default=[])
    args=parser.parse_args();need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu'
        and not os.environ.get('SLURM_JOB_GPUS'),'Independent audits require CPU Slurm')
    kind='check' if args.check else 'release' if args.release else 'report';out=OUT/'reporting'/f'{kind}_{os.environ["SLURM_JOB_ID"]}'
    out.mkdir(parents=True,exist_ok=False);frozen=snapshot(out);started=time.perf_counter()
    try:
        if args.check:need(args.data_summary is not None,'Full data summary required');result=check(args,out,frozen)
        elif args.release:need(args.plan is not None and args.data_report is not None,'Plan/data report required');result=release(args,out,frozen)
        else:need(args.data_report is not None,'Full data report required');result=report(args,out,frozen)
        need(source_hashes()==frozen,'Independent source changed during execution')
        result.update(seconds=time.perf_counter()-started,slurm_job_id=os.environ['SLURM_JOB_ID']);save(out/'summary.json',result)
        print(json.dumps(dict(passed=True,directory=str(out))),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,
            seconds=time.perf_counter()-started,partial_outputs_retained=True));raise


if __name__=='__main__':main()

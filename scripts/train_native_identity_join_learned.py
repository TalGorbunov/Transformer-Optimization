"""Held three-arm learned-selection training; every execution requires Slurm.

Native answer CE and same-variant length consistency are the only objectives.
Local labels never enter the trainable operator. The fixed final core is reset,
reloaded and deployed through the actual frozen native FP16 norm/head. No job
or efficacy release is implied by this source.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
from scripts import train_native_vision_v7 as v7
from scripts import native_vision_v7_runtime as native
from scripts import native_learned_selection_runtime as deployment
from scripts import stage_native_identity_join_learned as data_stage
from scripts.stage_native_vision_v7_features import MODEL, need, read, sha, object_sha

DATA = Path('/mnt/data/gabriele/gnn_transformer')
OUT = REPO / 'outputs/native_aggregation_vlm/identity_join_learned/training'
CKPT = Path('/mnt/ckpts/gabriele/gnn_transformer/identity_join_learned')
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
    per_main_seconds_cap=2700, campaign_gpu_seconds_cap=24000,
    bootstrap_seed=20261124, bootstrap_replicates=10000)
OWN = ('scripts/train_native_identity_join_learned.py',
    'tests/test_native_identity_join_learned_training.py',
    'slurm/native_identity_join_learned_train_check.sbatch',
    'slurm/native_identity_join_learned_train_profile.sbatch',
    'slurm/native_identity_join_learned_train.sbatch',
    'scripts/train_native_vision_v7.py', 'scripts/stage_native_vision_v7_features.py',
    'scripts/stage_native_vision_v6_teacher.py', 'scripts/probe_native_vision_v2_prefix.py',
    'scripts/probe_native_vision_parallel_local.py', 'gnnformer/paired_sequence_objectives.py',
    'tests/test_paired_sequence_objectives.py')
save = v7.save


def lr(step):
    need(type(step) is int and 1 <= step <= POLICY['steps'], 'Step out of range')
    if step <= 50: return .001 * step / 50
    return 1e-5 + (.001-1e-5) * (1+math.cos(math.pi*(step-50)/(4536-50))) / 2


def cpu_tree(torch, value):
    if isinstance(value, torch.Tensor):
        with torch.inference_mode(False), torch.no_grad(): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k:cpu_tree(torch,v) for k,v in value.items()}
    if isinstance(value, list): return [cpu_tree(torch,v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_tree(torch,v) for v in value)
    return value


def bind_file(bindings, path, digest=None):
    path = Path(path).resolve(); actual = sha(path)
    need(digest is None or actual == digest, 'Changed artifact: '+str(path))
    need(str(path) not in bindings or bindings[str(path)] == actual, 'Conflicting artifact binding')
    bindings[str(path)] = actual
    return dict(file=str(path), sha256=actual)


def name_target(tokenizer, name):
    need(name in data_stage.CHARS, 'Unknown canonical target name')
    tokens = list(tokenizer.encode(name, add_special_tokens=False))
    need(tokens and len(tokens)+1 <= 4 and not any(t in tokenizer.all_special_ids for t in tokens)
         and tokenizer.decode(tokens, skip_special_tokens=False) == name
         and tokenizer.eos_token_id == POLICY['target_eos'], 'Native name target or EOS differs')
    return tokens+[tokenizer.eos_token_id]


def pairing_binding(manifest, cache):
    grouped = {}; records = {}
    for cell in ('train_N8', 'train_N16'):
        for row in manifest['splits'][cell]['samples']:
            need(row['sid'] not in records, 'Repeated unique training scene')
            records[row['sid']] = row; grouped.setdefault(row['pair_id'], []).append(row)
    need(len(records) == 6048 and len(grouped) == 3024 and set(records) == set(cache['scenes']),
         'Complete training pair/cache inventory differs')
    pairs = []
    for pair_id, rows in grouped.items():
        rows = sorted(rows, key=lambda r:r['n_frames']); a,b = rows
        need(len(rows) == 2 and [r['n_frames'] for r in rows] == [8,16]
             and a['contrast_id'] == b['contrast_id'] and a['variant'] == b['variant']
             and a['question'] == b['question'] and a['gold'] == b['gold'],
             'Pair is not one intact same-answer variant at N8/N16')
        for row in rows:
            scene = cache['scenes'][row['sid']]
            need(scene['n_frames'] == row['n_frames'] and scene['target_ids'] == row['target_ids']
                 and scene['question'] == row['question'], 'Cached scene/native target ownership differs')
        need(cache['scenes'][a['sid']]['global_feature_ids'] == cache['scenes'][b['sid']]['global_feature_ids'],
             'Paired strict-prefix globals differ')
        pairs.append(dict(pair_id=pair_id, contrast_id=a['contrast_id'], variant=a['variant'],
            question=a['question'], gold=a['gold'], sids=[a['sid'],b['sid']], n_frames=[8,16],
            target_ids=list(a['target_ids'])))
    need(sum(2*len(p['target_ids']) for p in pairs) == 13440, 'Epoch native target inventory differs')
    need([{k:p[k] for k in ('pair_id','sids','question','gold','target_ids')} for p in pairs]
         ==cache['training_pairs'], 'Canonical cache pair order differs')
    return dict(schema_version=1, pairs=pairs, canonical_order='original_generation_first_occurrence',
        same_variant_only=True, pairs_per_epoch=3024, scenes_per_epoch=6048)


def presentation_order(pairs, seed):
    need(seed in POLICY['seeds'] and len(pairs) == 3024, 'Unregistered seed or pair inventory')
    rng = random.Random(seed); rows = []
    for epoch in range(1,13):
        permutation = list(range(len(pairs))); rng.shuffle(permutation)
        for slot in permutation:
            pair = pairs[slot]
            for side,sid in enumerate(pair['sids']):
                rows.append(dict(epoch=epoch, slot=slot, pair_id=pair['pair_id'],
                    contrast_id=pair['contrast_id'], variant=pair['variant'], sid=sid,
                    pair_side=side, n_frames=pair['n_frames'][side]))
    need(len(rows) == 72576 and all(rows[i]['pair_id'] == rows[i+1]['pair_id']
         and rows[i]['pair_side'] == 0 and rows[i+1]['pair_side'] == 1 for i in range(0,len(rows),2)),
         'Presentation stream broke intact length pairs')
    return rows


def batch_states(torch, cache, states, index, sids):
    from gnnformer.paired_sequence_objectives import sequence_layout, pack_feature_sequences
    local=[]; globals_=[]; targets=[]; sizes=[]
    for sid in sids:
        row=cache['scenes'][sid]; ids=row['local_feature_ids']; gids=row['global_feature_ids']; tokens=row['target_ids']
        need(len(ids)==row['n_frames'] and len(gids)==len(tokens) and all(len(r)==len(tokens) for r in ids),
             'Native causal feature segmentation differs')
        local.append(states[torch.tensor([[index[f] for f in r] for r in ids],device=states.device)])
        globals_.append(states[torch.tensor([index[f] for f in gids],device=states.device)])
        targets.append(tokens); sizes.append(len(ids))
    layout=sequence_layout(targets); h,g=pack_feature_sequences(local,globals_)
    valid=torch.zeros(h.shape[:2],device=h.device,dtype=torch.bool)
    for n,a,b in zip(sizes,layout['offsets'],layout['offsets'][1:]): valid[:n,a:b]=True
    need(g.shape[0]==layout['offsets'][-1] and torch.equal(g[layout['left']],g[layout['right']]),
         'Paired actual frozen globals differ')
    return h,g,layout,valid


def losses(torch, core, local, g, layout, valid, norm, head):
    from gnnformer.paired_sequence_objectives import sequence_layout
    import torch.nn.functional as F
    need(layout==sequence_layout(layout['target_sequences']) and not local.requires_grad and not g.requires_grad,
         'Changed layout or trainable frozen features')
    delta,cap=core(local,g,valid_mask=valid,output_dtype=torch.float32,capture=True)
    native_hidden=g+delta.to(g.dtype)
    need(bool(torch.isfinite(native_hidden).all()), 'Native training addition overflowed')
    logits=head(norm(native_hidden.unsqueeze(0)))[0]
    left=torch.tensor(layout['left'],device=g.device); right=torch.tensor(layout['right'],device=g.device)
    need(torch.equal(g[left],g[right]), 'Paired frozen strict-prefix states differ')
    with torch.autocast(device_type=g.device.type,enabled=False):
        ce_positions=F.cross_entropy(logits.float(),torch.tensor(layout['targets'],device=g.device),reduction='none')
        denominator=g[left].detach().float().square().sum(-1)+1e-6
        residual_positions=(delta[right]-delta[left]).square().sum(-1)/denominator
        ce=(ce_positions*torch.tensor(layout['scene_weights'],device=g.device)).sum()
        residual=(residual_positions*torch.tensor(layout['pair_weights'],device=g.device)).sum()
        total=ce+residual
    with torch.no_grad():
        offsets=layout['offsets']; lengths=[b-a for a,b in zip(offsets,offsets[1:])]
        pair_losses=[];cursor=0
        for length in lengths[::2]:
            pair_losses.append(float(residual_positions[cursor:cursor+length].mean()));cursor+=length
        selected=cap['gates'][valid]
        parts=dict(scene_lengths=lengths,scene_offsets=offsets,pair_lengths=lengths[::2],
            prefix_ids=[prefix for row in layout['prefixes'] for prefix in row],
            per_scene_ce=[float(ce_positions[a:b].mean()) for a,b in zip(offsets,offsets[1:])],
            per_pair_consistency=pair_losses,
            position_losses=dict(ce=ce_positions.tolist(),consistency=residual_positions.tolist()),
            pair_position_denominator=denominator.tolist(),
            residual_difference_norms=torch.linalg.vector_norm(delta[right]-delta[left],dim=-1).tolist(),
            valid_item_count_by_position=valid.sum(0).tolist(), gate_sum_by_position=cap['gates'].sum(0).tolist(),
            gate_min=float(selected.min()),gate_max=float(selected.max()),gate_mean=float(selected.mean()),
            gate_zero_count=int((selected==0).sum()),gate_one_count=int((selected==1).sum()),
            valid_gate_count=selected.numel(),payload_max_abs=float(cap['payload'].abs().max()),
            padding_messages_exact_zero=bool((cap['messages'][~valid]==0).all()),
            closed_message_coordinates_exact_zero=bool((cap['messages'][cap['gates']==0]==0).all()))
    need(bool(torch.isfinite(total)) and parts['padding_messages_exact_zero']
         and parts['closed_message_coordinates_exact_zero'], 'Nonfinite objective or invalid message path')
    return total,ce,residual,parts,cap


def zero_gradients(torch,core,residual):
    need(bool((core.up.weight==0).all()) and float(residual)==0., 'Initial U or residual is nonzero')
    gradients=torch.autograd.grad(residual,tuple(core.parameters()),retain_graph=True,allow_unused=True)
    need(all(v is None or bool(torch.isfinite(v).all() and (v==0).all()) for v in gradients),
         'Initial residual consistency gradient must be zero')
    return dict(passed=True,up_exactly_zero=True,residual_loss=0.,all_residual_gradients_zero=True)


def score_output(tokenizer,ids,gold):
    ids=list(ids);need(1<=len(ids)<=4 and all(type(t) is int and t>=0 for t in ids),'Invalid native token sequence')
    complete=ids[-1] in POLICY['native_eos']
    need(not any(t in POLICY['native_eos'] for t in ids[:-1]) and (complete or len(ids)==4),'Unexpected native stopping')
    body_ids=ids[:-1] if complete else ids; clean=not any(t in tokenizer.all_special_ids for t in body_ids)
    body=tokenizer.decode(body_ids,skip_special_tokens=False); prediction=body.strip() if clean and body.strip() in data_stage.CHARS else None
    return dict(answer_body=body,no_nonterminal_special_tokens=clean,prediction=prediction,
        parseable=prediction is not None,parsed_name_correct=prediction==gold,
        exact=complete and prediction==gold,completed=complete,truncated=not complete)


def sources():
    from scripts import stage_native_identity_join_features as feature_stage
    from scripts import profile_native_learned_selection as software
    names=set(OWN)|set(deployment.sources())|set(data_stage.source_hashes())|set(feature_stage.sources())|set(software.sources())
    return {name:sha(REPO/name) for name in sorted(names)}


def snapshot(out):
    (out/'code').mkdir(); frozen=sources()
    for name,digest in frozen.items():
        target=out/'code'/name.replace('/','_');target.write_bytes((REPO/name).read_bytes())
        need(sha(target)==digest, 'Source changed during snapshot')
    save(out/'source_hashes.json',frozen);return frozen


def test_helpers():
    result=[]
    for name in ('tests/test_native_identity_join_learned_training.py',
                 'tests/test_paired_sequence_objectives.py'):
        run=subprocess.run([sys.executable,str(REPO/name)],capture_output=True,text=True)
        result.append(dict(file=name,sha256=sha(REPO/name),returncode=run.returncode,stdout=run.stdout,stderr=run.stderr))
        need(run.returncode==0,'CPU helper failed: '+name+'\n'+run.stderr)
    return dict(passed=True,suites=result)


def timing_cases(manifest):
    """Training contrast plus fixed shared irrelevant extensions; no test rows."""
    rows=manifest['splits']['train_N16']['samples']; first=min(r['contrast_id'] for r in rows)
    selected=sorted((r for r in rows if r['contrast_id']==first),key=lambda r:r['variant'])
    need(len(selected)==3 and [r['variant'] for r in selected]==[0,1,2]
         and len({tuple(r['room_pair']) for r in selected})==1, 'Incomplete training timing contrast')
    atoms=read(manifest['render_cache_file']); rng=random.Random(POLICY['profile_timing_seed']); chosen=[]
    for step in range(17,65):
        candidates=sorted((a for a in atoms.values() if a['atom'][2]==step
            and a['atom'][1] not in selected[0]['room_pair']),key=lambda a:a['atom'])
        need(candidates,'No canonical outside-room extension atom')
        chosen.append(candidates[rng.randrange(len(candidates))])
    cases=[]
    for row in selected:
        view=data_stage.runtime_view(row)
        for n in (16,32,64):
            sample=dict(view,sid=view['sid']+f'__training_timing_N{n}',n_frames=n,
                image_files=view['image_files']+[{k:a[k] for k in ('path','sha256','dimensions','mode')} for a in chosen[:n-16]])
            cases.append(dict(sample=sample,parent_sid=row['sid'],contrast_id=first,variant=row['variant'],n_frames=n))
    return cases,dict(seed=POLICY['profile_timing_seed'],training_contrast_id=first,
        parent_sids=[r['sid'] for r in selected],room_pair=selected[0]['room_pair'],
        appended_atoms=chosen,shared_extensions_across_variants=True,no_dev_or_test_scene_used=True)


def check(args,out,frozen):
    import torch
    from transformers import AutoProcessor
    from scripts import cache_native_identity_join_features as worker
    from scripts import profile_native_learned_selection as software
    torch.set_num_threads(4);started=time.perf_counter();bindings={};tests=test_helpers()
    manifest=data_stage.verify_stage(args.data_summary)
    data_ref=bind_file(bindings,args.data_summary); stage_summary=read(args.data_summary)
    manifest_ref=bind_file(bindings,stage_summary['manifest_file'],stage_summary['manifest_sha256'])
    for key in ('samples','runtime_inputs','audit','render_cache','plan','parent_plan'):
        bind_file(bindings,manifest[key+'_file'],manifest[key+'_sha256'])
    need({k:v['count'] for k,v in manifest['splits'].items()}==dict(train_N8=3024,train_N16=3024,
        dev_N16=108,test_seen_N32=108,test_seen_N64=108,test_held_N32=108,test_held_N64=108),
        'Complete train/dev/test matrix differs')
    cache=worker.verify_cache(args.cache,verify_tensors=True);cache_ref=bind_file(bindings,args.cache)
    feature_plan=read(cache['plan_file']);bind_file(bindings,cache['plan_file'],cache['plan_sha256'])
    need(feature_plan['dataset_manifest_file']==manifest_ref['file']
         and feature_plan['dataset_manifest_sha256']==manifest_ref['sha256'], 'Feature cache uses a different corpus')
    for path,digest in {r['file']:r['file_sha256'] for r in cache['features'].values()}.items():bind_file(bindings,path,digest)
    bind_file(bindings,cache['native_model_file'],cache['native_model_sha256'])
    proof=software.verify_profile(args.native_profile);software_ref=bind_file(bindings,args.native_profile)
    software_plan=bind_file(bindings,proof['plan_file'],proof['plan_sha256'])
    need(proof['native_identity']==cache['native_identity']
         and proof['native_identity_sha256']==cache['native_identity_sha256'], 'Native software/cache identity differs')
    processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,use_fast=False,local_files_only=True)
    for split in manifest['splits'].values():
        for row in split['samples']:
            need(name_target(processor.tokenizer,row['gold'])==row['target_ids'], 'Native full name target changed')
    pairs=pairing_binding(manifest,cache);states,index=v7.load_features(torch,cache,'cpu')
    for pair in pairs['pairs']:
        h,g,layout,valid=batch_states(torch,cache,states,index,pair['sids'])
        need(not h.requires_grad and not g.requires_grad and valid.sum().item()==24*len(pair['target_ids']),
             'Training feature/padding ownership differs')
    save(out/'pairing.json',pairs);pair_ref=bind_file(bindings,out/'pairing.json')
    cases,derivation=timing_cases(manifest);dataout=DATA/'identity_join_learned/training'/out.name
    dataout.mkdir(parents=True,exist_ok=False);prepared=[]
    for i,case in enumerate(cases):
        bundle=deployment.prepare_scene(processor,case['sample'],verify_processor_parity=True)
        path=dataout/f'timing_{i:02d}.pt';torch.save(cpu_tree(torch,bundle),path)
        item=dict(case,bundle_file=str(path),bundle_sha256=sha(path),input_identity=bundle['metadata']['input_identity'])
        prepared.append(item);bind_file(bindings,path)
    save(out/'timing_cases.json',dict(cases=prepared,derivation=derivation));bind_file(bindings,out/'timing_cases.json')
    orders={str(seed):object_sha(presentation_order(pairs['pairs'],seed)) for seed in POLICY['seeds']}
    runtime_paths={cache_ref['file'],manifest_ref['file'],data_ref['file'],software_ref['file'],pair_ref['file'],
        str(out/'timing_cases.json'),cache['native_model_file'],cache['plan_file'],software_plan['file']}
    runtime_paths.update(r['file'] for r in cache['features'].values())
    runtime_paths.update(r['bundle_file'] for r in prepared)
    plan=dict(schema_version=1,policy=POLICY,source_sha256=frozen,artifact_bindings=bindings,
        runtime_bindings={p:bindings[p] for p in sorted(runtime_paths)},data_release=data_ref,manifest=manifest_ref,
        cache_binding=cache_ref,native_profile=software_ref,native_software_plan=software_plan,native_identity=cache['native_identity'],
        native_identity_sha256=cache['native_identity_sha256'],pairing_file=pair_ref['file'],pairing_sha256=pair_ref['sha256'],
        pairing_object_sha256=object_sha(pairs),order_sha256=orders,timing_cases_file=str(out/'timing_cases.json'),
        timing_cases_sha256=sha(out/'timing_cases.json'),tests=tests,actual_frozen_pair_equality_passed=True,
        all_native_name_targets_verified=True,no_model_loaded=True,elapsed_seconds=time.perf_counter()-started)
    save(out/'plan.json',plan);(out/'plan.sha256').write_text(sha(out/'plan.json')+'\n')
    verify_plan(out/'plan.json',ancestors=True)
    return dict(passed=True,completed=True,plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),
        source_sha256=frozen,no_model_loaded=True,tests=tests,elapsed_seconds=time.perf_counter()-started)


def verify_plan(path,*,ancestors=True):
    path=Path(path).resolve();need(sha(path)==path.with_suffix('.sha256').read_text().strip(),'Frozen CPU plan sidecar differs')
    plan=read(path);need(plan['policy']==POLICY and plan['source_sha256']==sources(),'CPU source/policy changed')
    for name,digest in plan['source_sha256'].items():
        need(sha(path.parent/'code'/name.replace('/','_'))==digest,'CPU source snapshot changed')
    for artifact,digest in plan['artifact_bindings' if ancestors else 'runtime_bindings'].items():
        need(sha(artifact)==digest,'Frozen artifact changed: '+artifact)
    need(object_sha(plan['native_identity'])==plan['native_identity_sha256']
         and object_sha(read(plan['pairing_file']))==plan['pairing_object_sha256'],'Native/pairing object identity differs')
    if ancestors:
        data_stage.verify_stage(plan['data_release']['file'])
        need(pairing_binding(read(plan['manifest']['file']),read(plan['cache_binding']['file']))==read(plan['pairing_file']),
             'Independent pairing reconstruction differs')
    return plan


def verify_release(path,plan_path,plan):
    need(path is not None,'Mains require a measured independent release')
    path=Path(path).resolve();release=read(path)
    need(release['protocol']=='identity_join_learned_selection_main_release' and release['passed'] is True
         and release['plan_sha256']==sha(plan_path) and release['source_sha256']==sources()
         and release['per_main_seconds_cap']==2700 and release['campaign_gpu_seconds_cap']==24000,
         'Main release identity/resource policy differs')
    data=release['data_release'];need(sha(data['file'])==data['sha256'],'Independent source/data release changed')
    checked=read(data['file']);need(checked['passed'] is True and checked['source_sha256']==release['report_source_sha256'],
        'Independent reporter source/data gate failed')
    for name,digest in release['report_source_sha256'].items():need(sha(REPO/name)==digest,'Independent reporter source changed')
    need(release['manifest']==plan['manifest'] and checked['data_bindings'].get(plan['manifest']['file'])==plan['manifest']['sha256'],
         'Independent release does not bind all development/test contexts')
    need(release['campaign_budget']['passed'] is True and set(release['profiles'])==set(POLICY['conditions'])
         and set(release['projections'])==set(POLICY['conditions']), 'Campaign/profile completeness differs')
    for condition,item in release['profiles'].items():
        sp=Path(item['directory'])/'summary.json';need(sha(sp)==item['summary_sha256'],'Training profile changed');p=read(sp)
        need(p['passed'] is True and p['completed'] is True and p['profile'] is True and p['condition']==condition
             and p['seed']==22 and p['steps']==32 and p['no_dev_or_test_evaluation'] is True
             and p['source_sha256']==sources() and p['plan_sha256']==sha(plan_path)
             and p['checkpoint_roundtrip_passed'] and p['native_timing']['passed'], 'Invalid matched training profile')
        projection=release['projections'][condition]
        need(projection['passed'] is True and 0<projection['projected_seconds']<=2700,'Main projection exceeds fixed cap')
    return dict(file=str(path),sha256=sha(path),manifest=plan['manifest'])


def audit_runtime_evidence(torch,result,bundle,condition,identity):
    meta=bundle['metadata'];n=meta['n_frames'];ids=result['generated_ids'];steps=len(ids)
    need(result['selection_mode']==condition and result['metadata']['native_identity_sha256']==identity
         and result['metadata']['scene_input_identity']==deployment.input_identity(bundle,identity)
         and len(result['captures'])==steps, 'Native trajectory source/capture ownership differs')
    shapes=dict(local_states=(n,1,3584),global_states=(1,3584),payload=(n,1,96),scores=(n,1),
        gates=(n,1),messages=(n,1,96),query=(1,96),aggregate=(1,96),preactivation=(1,96),
        delta=(1,3584),native_delta=(1,3584),fused_global=(1,3584),
        native_query_hidden=(n+1,1,3584),fused_query_hidden=(n+1,1,3584))
    native_keys={'local_states','global_states','native_delta','fused_global','native_query_hidden','fused_query_hidden'}
    for cap in result['captures']:
        need(set(cap)==set(shapes), 'Native query evidence fields differ')
        for key,shape in shapes.items():
            value=cap[key];dtype=torch.float16 if key in native_keys else torch.float32
            need(value.shape==shape and value.dtype==dtype and not value.requires_grad
                 and bool(torch.isfinite(value).all()), 'Malformed native capture: '+key)
        gates=cap['gates']
        need(bool(((gates>=0)&(gates<=1)).all()) and bool((cap['payload'].abs()<=1).all())
             and torch.equal(cap['messages'],gates.unsqueeze(-1)*cap['payload'])
             and bool((cap['messages'][gates==0]==0).all()), 'Selection/payload message identity differs')
        need(torch.equal(cap['native_delta'],cap['delta'].half())
             and torch.equal(cap['fused_global'],cap['global_states']+cap['native_delta'])
             and torch.equal(cap['fused_query_hidden'][:-1],cap['native_query_hidden'][:-1])
             and torch.equal(cap['fused_query_hidden'][-1],cap['fused_global']), 'Native cast/write identity differs')
    counts=result['counters']
    need(all(counts[k]==steps for k in ('model','language','norm','head','selection','broadcast'))
         and counts['visual']==1 and counts['probe_head']==0, 'Natural single-forward inventory differs')
    raw=result['raw_logits']
    need(raw.dtype==torch.float32 and raw.shape==(steps,152064) and bool(torch.isfinite(raw).all())
         and torch.equal(raw,raw.half().float()) and raw.argmax(-1).tolist()==ids,
         'Raw logits must be exact native FP16 promotion and unmasked argmax')
    return dict(passed=True,capture_count=steps,closed_message_coordinates_zero=True,
        native_cast_and_local_nonmutation=True,current_prefix_selection=True,origin_probe_calls=0)


def evaluate(torch,model,processor,core,rows,out,label,dataout,condition,identity):
    """Save every unfiltered native trajectory before calculating any accuracy."""
    started=time.perf_counter();destination=dataout/label;destination.mkdir();pending=[];io_seconds=0.
    for cell,sample in rows:
        tick=time.perf_counter();bundle=deployment.prepare_scene(processor,data_stage.runtime_view(sample))
        prep=time.perf_counter()-tick
        result=deployment.generate_native(model,processor,core,bundle,selection_mode=condition,
            native_identity_sha256=identity,capture=True,max_new_tokens=4)
        tick=time.perf_counter();path=destination/f'trajectory_{len(pending):03d}.pt'
        torch.save(cpu_tree(torch,dict(schema_version=1,**result)),path);digest=sha(path);io_seconds+=time.perf_counter()-tick
        save(path.with_suffix('.json'),dict(sid=sample['sid'],cell=cell,raw_file=str(path),raw_sha256=digest,validation_pending=True))
        evidence=audit_runtime_evidence(torch,result,bundle,condition,identity)
        pending.append(dict(cell=cell,sid=sample['sid'],raw_file=str(path),raw_sha256=digest,
            preprocessing_seconds=prep,model_seconds=result['model_seconds'],evidence=evidence))
        if len(pending)%16==0:
            save(out/(label+'_progress.json'),dict(processed=len(pending),total=len(rows),rows=pending,accuracy_not_scored=True))
            print(json.dumps(dict(evaluation=label,processed=len(pending),total=len(rows))),flush=True)
    need(len(pending)==len(rows), 'Every native trajectory must be saved before scoring')
    save(out/(label+'_raw_manifest.json'),dict(completed=True,n=len(pending),rows=pending,all_raw_retained_before_scoring=True))
    records=[]
    for item,(cell,sample) in zip(pending,rows):
        need(sha(item['raw_file'])==item['raw_sha256'], 'Saved native trajectory changed')
        raw=torch.load(item['raw_file'],map_location='cpu',weights_only=True);logits=raw['raw_logits']
        target=name_target(processor.tokenizer,sample['gold'])
        row=dict(item,n_frames=sample['n_frames'],gold=sample['gold'],target_ids=target,
            contrast_id=sample['contrast_id'],variant=sample['variant'],pair_id=sample['pair_id'],
            room_pair=sample['room_pair'],trio=sample['trio'],content_sha256=sample['content_sha256'],
            generated_ids=raw['generated_ids'],text=raw['text'],raw_text=raw['raw_text'],
            metadata=raw['metadata'],counters=raw['counters'],raw_dtype='torch.float32',
            first_token_correct=raw['generated_ids'][0]==target[0],
            first_token_nll=float(torch.logsumexp(logits[0].double(),-1)-logits[0,target[0]].double()),
            **score_output(processor.tokenizer,raw['generated_ids'],sample['gold']))
        need(row['completed']==raw['completed'] and row['truncated']==raw['truncated'],'Scorer/native stopping differs')
        records.append(row)
    result=dict(label=label,n=len(records),rows=records,exact_count=sum(r['exact'] for r in records),
        first_token_nll=sum(r['first_token_nll'] for r in records)/len(records),seconds=time.perf_counter()-started,
        raw_manifest_file=str(out/(label+'_raw_manifest.json')),raw_manifest_sha256=sha(out/(label+'_raw_manifest.json')),
        archive_io_seconds=io_seconds,all_raw_retained_before_scoring=True,all_prefix_captures_retained=True)
    save(out/(label+'.json'),result);return result


def replay_metrics(torch,a,b):
    a=a.double();b=b.double()
    need(a.shape==b.shape and a.ndim==2 and bool(torch.isfinite(a).all() and torch.isfinite(b).all()),'Malformed replay logits')
    tv=.5*(a.softmax(-1)-b.softmax(-1)).abs().sum(-1);same=a.argmax(-1)==b.argmax(-1)
    return dict(tv=tv.tolist(),top1_equal=same.tolist(),maximum_tv=float(tv.max()),
        all_top1_equal=bool(same.all()),passed=bool((tv<=.02).all() and same.all()))


def profile_native(torch,model,processor,core,plan,cache,states,index,dataout,condition):
    """Nine training-derived natural trajectories; no answer scoring."""
    cases=read(plan['timing_cases_file'])['cases'];observations=[];norm=model.model.language_model.norm;head=model.lm_head
    for i,case in enumerate(cases):
        tick=time.perf_counter();bundle=deployment.prepare_scene(processor,case['sample'],verify_processor_parity=True)
        preparation=time.perf_counter()-tick
        need(bundle['metadata']['input_identity']==case['input_identity'], 'Training timing input differs from CPU plan')
        generation_start=time.perf_counter()
        result=deployment.generate_native(model,processor,core,bundle,selection_mode=condition,
            native_identity_sha256=plan['native_identity_sha256'],capture=True,max_new_tokens=4)
        generation=time.perf_counter()-generation_start
        archive_start=time.perf_counter();natural_path=dataout/f'native_timing_{i:02d}_natural.pt'
        torch.save(cpu_tree(torch,dict(schema_version=1,result=result)),natural_path);natural_sha=sha(natural_path)
        save(natural_path.with_suffix('.json'),dict(case=i,raw_file=str(natural_path),raw_sha256=natural_sha,validation_pending=True))
        initial_archive_seconds=time.perf_counter()-archive_start;replay_start=time.perf_counter()
        audit_runtime_evidence(torch,result,bundle,condition,plan['native_identity_sha256'])
        replayed=[];metrics=[]
        with torch.inference_mode():
            for t,cap in enumerate(result['captures']):
                # Preserve the actual [N+1,1,H] projection shape. Only the global
                # vector was retained by native generation; compare that row.
                full=cap['fused_query_hidden'].to(model.device)
                logits=head.forward(norm.forward(full));replayed.append(logits[-1,0].cpu())
                metrics.append(replay_metrics(torch,result['raw_logits'][t:t+1],logits[-1,0][None].cpu()))
        torch.cuda.synchronize();replay_seconds=time.perf_counter()-replay_start
        tick=time.perf_counter();path=dataout/f'native_timing_{i:02d}.pt'
        torch.save(cpu_tree(torch,dict(schema_version=1,result=result,replay_global_logits=torch.stack(replayed))),path)
        digest=sha(path);archive_seconds=initial_archive_seconds+time.perf_counter()-tick
        tokens=len(result['generated_ids'])
        bound=preparation+(generation+replay_seconds+archive_seconds)*4/tokens
        observations.append(dict(case=i,parent_sid=case['parent_sid'],contrast_id=case['contrast_id'],variant=case['variant'],
            n_frames=case['n_frames'],generated_tokens=tokens,preprocessing_seconds=preparation,
            generation_seconds=generation,replay_seconds=replay_seconds,archive_seconds=archive_seconds,
            four_token_seconds_bound=bound,raw_file=str(path),raw_sha256=digest,
            natural_raw_file=str(natural_path),natural_raw_sha256=natural_sha,counters=result['counters'],
            native_head_replays=metrics,standalone_head_calls=tokens,no_accuracy_scoring=True))
    need(len(observations)==9 and Counter(r['n_frames'] for r in observations)==Counter({16:3,32:3,64:3}),
         'Native training timing inventory differs')
    return dict(passed=all(m['passed'] for r in observations for m in r['native_head_replays']),rows=observations,
        T16=max(r['four_token_seconds_bound'] for r in observations if r['n_frames']==16),
        T32=max(r['four_token_seconds_bound'] for r in observations if r['n_frames']==32),
        T64=max(r['four_token_seconds_bound'] for r in observations if r['n_frames']==64),
        natural_generations=9,native_model_calls=sum(r['generated_tokens'] for r in observations),vision_calls=9,
        standalone_head_calls=sum(r['standalone_head_calls'] for r in observations),origin_probe_calls=0,
        all_timing_cases_training_derived=True,no_dev_or_test_evaluation=True,
        bound_rule='prepare+(generation+same_shape_head_replay+raw_save_and_hash)*4/observed_tokens')


def run(args,out,frozen,started):
    import shutil
    import torch
    from gnnformer.runtime import load_runtime
    from gnnformer.parallel_local_learned_selection import ParallelLocalLearnedSelection
    from scripts import cache_native_identity_join_features as worker
    from scripts import profile_native_learned_selection as software
    torch.set_num_threads(4);args.plan=args.plan.resolve();plan=verify_plan(args.plan,ancestors=False)
    need(args.condition in POLICY['conditions'] and args.seed in POLICY['seeds']
         and (not args.profile or args.seed==22), 'Unregistered condition/seed/profile')
    release=None if args.profile else verify_release(args.main_release,args.plan,plan)
    runid=out.name;dataout=DATA/'identity_join_learned/training'/runid;dataout.mkdir(parents=True,exist_ok=False)
    ckpt=CKPT/'training'/runid;ckpt.mkdir(parents=True,exist_ok=False)
    load_start=time.perf_counter();cache=worker.verify_cache(plan['cache_binding']['file'],ancestors=False,verify_tensors=False)
    need(cache['native_identity']==plan['native_identity'],'Consumed cache native identity differs')
    loaded=load_runtime(str(MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda')
    model=loaded.model.eval().requires_grad_(False);native.native_contract(model)
    software_plan=read(plan['native_software_plan']['file'])
    hardware=software.live_identity(torch,loaded,software_plan)
    need(hardware['gpu']=='NVIDIA B200' and torch.cuda.device_count()==1,'Require one measured B200 GPU')
    norm=model.model.language_model.norm;head=model.lm_head
    native_weights=dict(norm=v7.tensor_info(norm.weight),head=v7.tensor_info(head.weight))
    need(native_weights['norm']==plan['native_identity']['norm_weight']
         and native_weights['head']==plan['native_identity']['head_weight'], 'Actual native training tensors differ')
    native_file=ckpt/'native_norm_head.pt';shutil.copyfile(cache['native_model_file'],native_file)
    need(sha(native_file)==cache['native_model_sha256'],'Per-run native model tensor copy differs')
    states,index=v7.load_features(torch,cache,loaded.device);torch.cuda.synchronize()
    load_seconds=time.perf_counter()-load_start
    torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    core=ParallelLocalLearnedSelection(mode=args.condition).to(loaded.device)
    initial=cpu_tree(torch,core.state_dict());initialized=v7.state_info(core)
    initial_file=ckpt/'initial.pt';torch.save(dict(branch=initial,seed=args.seed,step=0),initial_file)
    pairing=read(plan['pairing_file']);order=presentation_order(pairing['pairs'],args.seed)
    need(object_sha(order)==plan['order_sha256'][str(args.seed)],'Frozen presentation order differs')
    save(out/'presentations.json',order)
    config=dict(run_id=runid,arm='parallel',condition=args.condition,selection_mode=args.condition,seed=args.seed,
        profile=args.profile,policy=POLICY,consistency_coefficient=1.,slurm_job_id=os.environ['SLURM_JOB_ID'],
        plan_file=str(args.plan),plan_sha256=sha(args.plan),source_sha256=frozen,
        cache_binding=plan['cache_binding'],manifest=plan['manifest'],data_release=plan['data_release'],
        native_identity=plan['native_identity'],native_identity_sha256=plan['native_identity_sha256'],
        native_weight_identity=native_weights,native_model_file=str(native_file),native_model_sha256=sha(native_file),
        native_profile=plan['native_profile'],native_software_plan=plan['native_software_plan'],
        initialized=initialized,initialized_sha256=object_sha(initialized),initial_checkpoint=str(initial_file),
        initial_checkpoint_sha256=sha(initial_file),pairing_file=plan['pairing_file'],pairing_sha256=plan['pairing_sha256'],
        presentations_sha256=sha(out/'presentations.json'),order_sha256=object_sha(order),
        model=plan['native_identity']['model'],runtime=plan['native_identity']['runtime'],
        processor=plan['native_identity']['processor'],native_dtypes=plan['native_identity']['native_dtypes'],
        model_and_features_load_seconds=load_seconds,hardware=hardware,checkpoint_directory=str(ckpt),
        data_directory=str(dataout),main_release=release)
    save(out/'config.json',config)
    versions={name:p._version for name,p in model.named_parameters()}
    optimizer=torch.optim.AdamW(core.parameters(),lr=.001,weight_decay=0.)
    setup=time.perf_counter()-started;log=[];first_gradients=[];isolation=[];zero=None;io_seconds=0.
    exercised={name:False for name,_ in core.named_parameters()};steps=32 if args.profile else 4536
    for step in range(1,steps+1):
        tick=time.perf_counter();batch=order[(step-1)*16:step*16];sids=[r['sid'] for r in batch]
        need(len(batch)==16 and all(batch[i]['pair_id']==batch[i+1]['pair_id']
             and batch[i]['variant']==batch[i+1]['variant'] and batch[i]['pair_side']==0
             and batch[i+1]['pair_side']==1 for i in range(0,16,2)), 'Batch broke an intact same-variant length pair')
        h,g,layout,valid=batch_states(torch,cache,states,index,sids)
        optimizer.zero_grad(set_to_none=True);rate=lr(step)
        for group in optimizer.param_groups:group['lr']=rate
        total,ce,residual,parts,cap=losses(torch,core,h,g,layout,valid,norm,head)
        if step==1:zero=zero_gradients(torch,core,residual)
        if step in (1,2,32):
            sg,pg=torch.autograd.grad(total,(cap['scores'],cap['payload']),retain_graph=True,allow_unused=True)
            need(sg is not None and pg is not None and bool(torch.isfinite(sg).all() and torch.isfinite(pg).all())
                 and not states.requires_grad and not h.requires_grad and not g.requires_grad,
                 'Selection/payload graph detached or frozen features acquired gradients')
            isolation.append(dict(step=step,passed=True,score_gradient_present=True,payload_gradient_present=True,
                score_gradient_norm=float(torch.linalg.vector_norm(sg)),payload_gradient_norm=float(torch.linalg.vector_norm(pg)),
                frozen_features=True,no_local_label_input=True))
        total.backward()
        need(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in core.parameters()),
             'Missing/nonfinite core parameter gradient')
        for name,p in core.named_parameters():exercised[name]|=bool((p.grad!=0).any())
        if step<=2:first_gradients.append(dict(step=step,pre_clip={name:v7.tensor_info(p.grad) for name,p in core.named_parameters()}))
        gradient_norm=torch.nn.utils.clip_grad_norm_(core.parameters(),1.)
        need(bool(torch.isfinite(gradient_norm)), 'Nonfinite clipping norm')
        optimizer.step();need(all(bool(torch.isfinite(p).all()) for p in core.parameters()),'Nonfinite updated core')
        need(not any(p.requires_grad or p.grad is not None for p in model.parameters()),'Frozen native gradient state changed')
        torch.cuda.synchronize()
        log.append(dict(step=step,lr=rate,loss=float(total),ce_loss=float(ce),consistency_loss=float(residual),
            weighted_consistency_loss=float(residual),consistency_coefficient=1.,gradient_norm=float(gradient_norm),
            clipped=float(gradient_norm)>1.,seconds=time.perf_counter()-tick,target_ids=layout['targets'],sids=sids,
            pair_ids=[r['pair_id'] for r in batch[::2]],epochs=[r['epoch'] for r in batch[::2]],**parts))
        if step%100==0:
            tick=time.perf_counter();save(out/'training.json',log);io_seconds+=time.perf_counter()-tick
            print(json.dumps(dict(step=step,total=float(total),ce=float(ce),consistency=float(residual))),flush=True)
    exemptions=['selection_bias'] if args.condition=='softmax' else []
    need(all(value or name in exemptions for name,value in exercised.items()),'A required parameter gradient was never exercised')
    need(exercised['selection_weight'] and (args.profile or sum(len(r['target_ids']) for r in log)==161280),
         'Selection weight never learned or target presentation inventory differs')
    tick=time.perf_counter();save(out/'training.json',log);save(out/'first_gradients.json',first_gradients)
    save(out/'gradient_isolation.json',isolation);io_seconds+=time.perf_counter()-tick
    optimizer.zero_grad(set_to_none=True);del optimizer,total,ce,residual,cap
    core.eval().requires_grad_(False);final_state=v7.state_info(core)
    checkpoint=ckpt/('profile.pt' if args.profile else 'final.pt')
    torch.save(dict(branch=core.state_dict(),step=steps,config=config),checkpoint);checkpoint_sha=sha(checkpoint)
    core.load_state_dict(initial);need(v7.state_info(core)==initialized,'Initial core reset differs')
    packet=torch.load(checkpoint,map_location=loaded.device,weights_only=True);core.load_state_dict(packet['branch'])
    need(packet['step']==steps and packet['config']==config and v7.state_info(core)==final_state,'Final checkpoint reload differs')
    before_eval=v7.state_info(core);core_versions={name:p._version for name,p in core.named_parameters()}
    result=dict(config,passed=True,completed=True,computational_integrity_passed=True,steps=steps,
        zero_initialization_core_gradients=zero,all_parameter_gradients_exercised=exercised,
        gradient_nonzero_exemptions=exemptions,checkpoint_roundtrip_passed=True,
        training_file=str(out/'training.json'),training_sha256=sha(out/'training.json'),
        first_gradients_file=str(out/'first_gradients.json'),first_gradients_sha256=sha(out/'first_gradients.json'),
        gradient_isolation_file=str(out/'gradient_isolation.json'),gradient_isolation_sha256=sha(out/'gradient_isolation.json'),
        training_seconds=sum(r['seconds'] for r in log),first_four_step_seconds=sum(r['seconds'] for r in log[:4]),
        step_seconds_max_steady=max(r['seconds'] for r in log[4:]),setup_before_training_seconds=setup,
        setup_includes_model_and_feature_load=True,training_log_io_seconds=io_seconds)
    manifest=read(plan['manifest']['file'])
    if args.profile:
        timing=profile_native(torch,model,loaded.processor,core,plan,cache,states,index,dataout,args.condition)
        save(out/'native_timing.json',timing)
        result.update(native_timing=timing,no_dev_or_test_evaluation=True,
            profile_checkpoint=str(checkpoint),profile_checkpoint_sha256=checkpoint_sha)
        if not timing['passed']:
            result.update(passed=False,computational_integrity_passed=False,elapsed_seconds=time.perf_counter()-started)
            save(out/'summary.json',result);need(False,'Native captured-state head replay failed; all raw observations retained')
    else:
        devrows=[('dev_N16',r) for r in manifest['splits']['dev_N16']['samples']]
        testrows=[(cell,r) for cell,split in manifest['splits'].items() if cell.startswith('test_') for r in split['samples']]
        need(len(devrows)==108 and len(testrows)==432,'Final evaluation denominators differ')
        dev=evaluate(torch,model,loaded.processor,core,devrows,out,'dev_final',dataout,args.condition,plan['native_identity_sha256'])
        selected=dict(step=4536,exact_count=dev['exact_count'],nll=dev['first_token_nll'],checkpoint=str(checkpoint),
            checkpoint_sha256=checkpoint_sha,parameter_sha256=object_sha(final_state),
            dev_file=str(out/'dev_final.json'),dev_sha256=sha(out/'dev_final.json'))
        save(out/'selection.json',dict(rule=POLICY['selection'],selected=selected,development=[selected],dev_descriptive_only=True))
        evaluate(torch,model,loaded.processor,core,testrows,out,'test',dataout,args.condition,plan['native_identity_sha256'])
        result.update(selected=selected,development=[selected],dev_descriptive_only=True,native_dev_count=108,
            native_test_count=432,test_file=str(out/'test.json'),test_sha256=sha(out/'test.json'))
    after_eval=v7.state_info(core);after_native=dict(norm=v7.tensor_info(norm.weight),head=v7.tensor_info(head.weight))
    need(before_eval==after_eval==final_state and after_native==native_weights
         and core_versions=={name:p._version for name,p in core.named_parameters()}
         and versions=={name:p._version for name,p in model.named_parameters()}
         and not any(p.requires_grad or p.grad is not None for p in model.parameters())
         and sha(checkpoint)==checkpoint_sha and sha(native_file)==config['native_model_sha256'],
         'Native evaluation changed core/model weights or checkpoint artifacts')
    endpoint=dict(schema_version=1,step=steps,checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_sha,
        before_evaluation=before_eval,after_evaluation=after_eval,parameter_sha256=object_sha(final_state),
        native_weight_identity_before=native_weights,native_weight_identity_after=after_native,
        checkpoint_matches_deployed_endpoint=True,core_versions_unchanged=True,native_versions_unchanged=True)
    save(out/'final_endpoint.json',endpoint)
    validation=time.perf_counter();verify_plan(args.plan,ancestors=False)
    if release is not None:need(sha(release['file'])==release['sha256'],'Main release changed during run')
    result.update(final_endpoint_file=str(out/'final_endpoint.json'),final_endpoint_sha256=sha(out/'final_endpoint.json'),
        final_endpoint_unchanged=True,frozen_backbone_gradient_state_preserved=True,
        final_validation_seconds=time.perf_counter()-validation,elapsed_seconds=time.perf_counter()-started)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true');mode.add_argument('--profile',action='store_true');mode.add_argument('--run',action='store_true')
    parser.add_argument('--data-summary',type=Path);parser.add_argument('--cache',type=Path)
    parser.add_argument('--native-profile',type=Path);parser.add_argument('--plan',type=Path)
    parser.add_argument('--main-release',type=Path);parser.add_argument('--condition',choices=POLICY['conditions'])
    parser.add_argument('--seed',type=int,choices=POLICY['seeds']);args=parser.parse_args()
    native.require_slurm(gpu=not args.check)
    if args.check:
        need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS')
             and all(x is not None for x in (args.data_summary,args.cache,args.native_profile)),
             'CPU check requires passed data/cache/native software artifacts')
    else:need(args.plan is not None and args.condition is not None and args.seed is not None,'GPU mode requires plan/condition/seed')
    started=time.perf_counter();job=os.environ['SLURM_JOB_ID']
    runid=f'train_check_{job}' if args.check else f'{"profile" if args.profile else "run"}_{args.condition}_s{args.seed}_{job}'
    out=OUT/runid;out.mkdir(parents=True,exist_ok=False);frozen=snapshot(out)
    save(out/'request.json',dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},source_sha256=frozen))
    try:
        result=check(args,out,frozen) if args.check else run(args,out,frozen,started)
        need(sources()==frozen,'Sources changed during execution')
        save(out/'summary.json',result);print(json.dumps(dict(completed=True,passed=result['passed'],directory=str(out))),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,
            elapsed_seconds=time.perf_counter()-started,partial_outputs_retained=True));raise


if __name__=='__main__':main()

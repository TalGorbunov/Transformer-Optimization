"""Held V18 native semantic-gate training, with one fixed final checkpoint.

Both modes use the same bounded vector core, canonical cached features, paired
sequence CE plus residual consistency, and native unmasked generation. The
all-open control gathers the same original gates and executes the same native
probe; only applied gates differ. No predictor, reference bank or augmentation.
All tensor/data/model execution requires Slurm. This source does not release jobs.
"""
from __future__ import annotations
import argparse
from collections import Counter
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
from scripts import train_native_vision_v10 as ancestor
from scripts import train_native_vision_v7 as v7
from scripts import native_vision_v7_runtime as native
from scripts.stage_native_vision_v7_features import MODEL,need,read,sha,object_sha
DATA=ancestor.DATA
OUT=REPO/'outputs/native_aggregation_vlm/v18'
CKPT=Path('/mnt/ckpts/gabriele/gnn_transformer/native_aggregation_vlm_v18')
TRAIN=ancestor.TRAIN;SCHEDULE=ancestor.SCHEDULE;PAIRING=ancestor.PAIRING
CACHE=DATA/'v10_parallel_local/feature_cache.json'
GATE_CACHE=DATA/'v18_semantic_gates/feature_cache.json'
FRESH=DATA/'v18_fresh'
PRIOR=REPO/'outputs/native_aggregation_vlm/v17/local_readout/report_442798/summary.json'
POLICY=dict(
    protocol='v18_native_semantic_gate_final_only',native_arm='parallel',conditions=['native_gate','all_open'],seeds=[20,21],
    epochs=40,pair_slots_per_epoch=918,scene_slots_per_epoch=1836,unique_training_scenes=1782,
    scene_presentations=73440,pair_presentations=36720,target_positions=177120,batch_size=16,pairs_per_batch=8,steps=4590,
    lr=.001,warmup=50,final_lr=.00001,weight_decay=0.,clip_norm=1.,optimizer='AdamW_core_only',
    rank=96,hidden_size=3584,parameters=1041600,merge='sum',post_activation='silu',payload_activation='tanh',
    native_dtype='torch.float16',branch_dtype='torch.float32',zero_up_initialization=True,
    gate_rule='max(0,full_vocabulary_p1-full_vocabulary_p0); FP64 probabilities then detached FP32',
    gate_origin='original_empty_prefix_local_feature_identity',gate_gradient='always_detached',
    gate_prefix_rule='same_original_gate_broadcast_over_all_valid_target_prefixes',padding_gate=0.,
    all_open_rule='same_gate_gather_and_native_probe_then_ones_on_actual_rows_only',
    consistency_coefficients={'native_gate':1.,'all_open':1.},consistency_epsilon=1e-6,
    ce_reduction='mean_over_scenes_of_mean_over_valid_native_target_positions',
    consistency_reduction='mean_over_pairs_of_mean_over_corresponding_strict_prefix_positions',
    denominator='detached_squared_L2_of_identical_frozen_global_state_plus_fixed_epsilon',
    pair_order='persistent_random.Random(seed)_fresh_canonical_pair_permutation_each_epoch',
    within_pair_order='registered_pair_sides_0_then_1',
    saturation_rule='retain54_N16_K16_same_SID_pairs_with_exact_zero_regularizer',
    path_loss_role='diagnostic_only_no_loss_weight_no_inference_role',predictor_parameters=0,reference_rows=0,
    full_answer_targets='canonical_native_numeral_tokens_plus_exactly_one_terminal_EOS',
    max_new_tokens=4,exact_requires_eos=True,reject_nonterminal_special_tokens=True,
    native_eos=[151645,151643],target_eos=151645,
    selection='fixed_final_step4590',dev_label='dev_final',dev_examples=64,dev_descriptive_only=True,test_examples=272,
    profile_steps=32,training_count_support=list(range(17)),development_count_support=list(range(16)),
    test_count_support=list(range(17)),maximum_training_N=16,test_N=[32,64],fresh_test_seed=20261118,
    per_main_seconds_cap=2700,campaign_gpu_seconds_cap=14400,profile_seconds_cap=300)
PROFILE_CALLS=dict(training_updates=32,native_prefill_forwards=10,visual_forwards=10,
    original_probe_norm_calls=4,original_probe_head_calls=4,original_probability_calls=4,
    same_state_native_shaped_head_replays=10,cached_state_head_projections=10,
    standalone_head_calls=20,natural_generations=0)
TEST_FILES=('tests/test_native_vision_v18_training.py','tests/test_parallel_local_semantic_aggregation.py',
    'tests/test_paired_sequence_objectives.py','scripts/native_vision_v10_pairs.py')
OWN=tuple(sorted(set(ancestor.OWN)|{'scripts/train_native_vision_v18.py',*TEST_FILES,
    'gnnformer/parallel_local_semantic_aggregation.py','gnnformer/parallel_local_semantic_gate.py',
    'scripts/native_vision_semantic_gate_runtime.py','slurm/native_vision_v18_train_check.sbatch',
    'slurm/native_vision_v18_train_profile.sbatch','slurm/native_vision_v18_train.sbatch'}))
save=ancestor.save;bind_file=ancestor.bind_file;bind_manifest=ancestor.bind_manifest
lr=ancestor.lr;batch_states=ancestor.batch_states


def sources():
    from scripts import stage_native_vision_v18_gates as gates
    from scripts import profile_native_vision_v18_semantic_gate as software
    from scripts import stage_native_vision_v18_test as fresh
    names=set(OWN)|set(gates.sources())|set(software.sources())|set(fresh.source_hashes())
    return {name:sha(REPO/name) for name in sorted(names)}


def snapshot(out):
    (out/'code').mkdir();frozen=sources()
    for name,digest in frozen.items():
        target=out/'code'/name.replace('/','_');target.write_bytes((REPO/name).read_bytes())
        need(sha(target)==digest,'Source changed during snapshot')
    save(out/'source_hashes.json',frozen)
    (out/'INDEX.md').write_text('# V18 semantic-gate training\n\n[Plan](plan.json) · [Configuration](config.json) · [Training](training.json) · [Summary](summary.json)\n')
    return frozen


def presentation_order(pairs,seed):
    need(seed in POLICY['seeds'] and len(pairs)==918,'Unregistered V18 seed or pair inventory')
    rng=random.Random(seed);rows=[]
    for epoch in range(1,41):
        permutation=list(range(918));rng.shuffle(permutation)
        for slot in permutation:
            p=pairs[slot]
            for side,sid in enumerate(p['sids']):
                rows.append(dict(epoch=epoch,slot=slot,pair_id=p['pair_id'],question=p['question'],gold=p['gold'],sid=sid,
                    pair_side=side,pair_kind=p['pair_kind'],n_frames=p['n_frames'][side],replica=p['replicas'][side]))
    need(len(rows)==73440 and all(rows[i]['pair_id']==rows[i+1]['pair_id'] and rows[i]['pair_side']==0
         and rows[i+1]['pair_side']==1 for i in range(0,len(rows),2)),'Broken paired presentation order')
    return rows


def pairing_binding(cache_path):
    from scripts.native_vision_v10_pairs import load_pairs
    value=load_pairs(TRAIN,cache_path,SCHEDULE)
    return dict(value,protocol='v18_unchanged_v10_pairing_new_seeds',seeds=POLICY['seeds'])


def test_helpers():
    suites=[]
    for name in TEST_FILES:
        result=subprocess.run([sys.executable,str(REPO/name)],capture_output=True,text=True)
        suites.append(dict(file=name,sha256=sha(REPO/name),returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        need(result.returncode==0,'CPU helper failed: '+name+'\n'+result.stderr)
    return dict(passed=True,suites=suites)


def pack_gates(torch,cache,gate_cache,gate_values,gate_index,sids,layout,condition):
    """Gather only original empty-prefix gates, then pad item rows explicitly.

    Both conditions follow this same gather. Prefix feature IDs validate origin
    ownership but never choose a different probability or a target-conditioned
    gate. No QA label or count enters the gate values.
    """
    need(condition in POLICY['conditions'],'Unknown gate condition')
    need(len(sids)==len(layout['target_sequences']),'Gate scene layout differs')
    need(gate_values.dtype==torch.float32 and gate_values.ndim==1 and not gate_values.requires_grad
         and bool(torch.isfinite(gate_values).all()) and bool(((gate_values>=0)&(gate_values<=1)).all()),
         'Require detached finite FP32 origin gate vector')
    counts=[cache['scenes'][sid]['n_frames'] for sid in sids];width=max(counts)
    pieces=[];origin_rows=[]
    for sid,n,tokens in zip(sids,counts,layout['target_sequences']):
        scene=cache['scenes'][sid];origins=gate_cache['scenes'][sid]['local_empty_feature_ids']
        need(len(origins)==n and scene['target_ids']==tokens and len(scene['local_feature_ids'])==n,
             'Gate/feature scene identity or target segmentation differs')
        for origin,features in zip(origins,scene['local_feature_ids']):
            need(features[0]==origin and len(features)==len(tokens) and all(
                gate_cache['prefix_to_empty_feature_id'][fid]==origin for fid in features),
                'Strict-prefix feature maps to a different original gate')
        native_gates=gate_values[torch.tensor([gate_index[fid] for fid in origins],device=gate_values.device)]
        applied=native_gates if condition=='native_gate' else torch.ones_like(native_gates)
        part=torch.zeros(width,len(tokens),device=gate_values.device,dtype=torch.float32)
        part[:n]=applied[:,None].expand(n,len(tokens));pieces.append(part)
        origin_rows.append(list(origins))
    packed=torch.cat(pieces,dim=1).detach()
    need(packed.shape==(width,layout['offsets'][-1]),'Gate packing lost a valid causal position')
    return packed,origin_rows


def semantic_batch(torch,cache,states,index,sids,gate_cache,gate_values,gate_index,condition):
    local,g,layout=batch_states(torch,cache,states,index,sids)
    gates,origins=pack_gates(torch,cache,gate_cache,gate_values,gate_index,sids,layout,condition)
    need(gates.shape==local.shape[:2] and gates.device==local.device,'Packed gates and features differ')
    return local,g,layout,gates,origins


def losses(torch,branch,local,g,layout,gates,norm,head,sids):
    from gnnformer.paired_sequence_objectives import sequence_objectives
    delta,capture=branch(local,g,gates=gates,output_dtype=torch.float32,capture=True)
    logits=head(norm((g+delta.to(g.dtype)).unsqueeze(0)))[0]
    result=sequence_objectives(logits,delta,capture['aggregate'],branch.aggregate_projection.weight,
        branch.up.weight,g,layout,eps=POLICY['consistency_epsilon'])
    total=result['ce']+result['residual']
    with torch.no_grad():
        offsets=layout['offsets'];lengths=[b-a for a,b in zip(offsets,offsets[1:])];pair_lengths=lengths[::2]
        per_scene=[float(result['ce_positions'][a:b].mean()) for a,b in zip(offsets,offsets[1:])]
        per_pair={}
        for label,values in (('consistency',result['residual_positions']),('path',result['path_positions'])):
            cursor=0;per_pair[label]=[]
            for length in pair_lengths:
                per_pair[label].append(float(values[cursor:cursor+length].mean()));cursor+=length
        identity=[sids[i]==sids[i+1] for i in range(0,len(sids),2)];cursor=0
        for same,length in zip(identity,pair_lengths):
            if same:need(bool((result['residual_positions'][cursor:cursor+length]==0).all()),'Saturated same-SID residual is nonzero')
            cursor+=length
        left=torch.tensor(layout['left'],device=g.device);right=torch.tensor(layout['right'],device=g.device)
        components=dict(scene_lengths=lengths,scene_offsets=offsets,pair_lengths=pair_lengths,
            prefix_ids=[prefix for row in layout['prefixes'] for prefix in row],per_scene_ce=per_scene,
            per_pair_consistency=per_pair['consistency'],per_pair_path=per_pair['path'],
            position_losses=dict(ce=result['ce_positions'].tolist(),consistency=result['residual_positions'].tolist(),path=result['path_positions'].tolist()),
            pair_position_denominator=result['denominator'].tolist(),
            residual_difference_norms=torch.linalg.vector_norm(delta[right]-delta[left],dim=-1).tolist(),
            B_norms=result['bound'].tolist(),saturated_identity_pairs=identity,
            gate_tensor=v7.tensor_info(gates),applied_gate_sum_by_position=gates.sum(0).tolist(),
            closed_gate_count_by_position=(gates==0).sum(0).tolist(),
            payload_max_abs=float(capture['payload'].abs().max()),
            closed_message_coordinates_exact_zero=bool((capture['messages'][gates==0]==0).all()))
    need(bool(torch.isfinite(total)) and components['closed_message_coordinates_exact_zero'],'Nonfinite loss or closed gate payload bypass')
    return total,result['ce'],result['residual'],result['path'],components


def zero_gradients(torch,branch,consistency):
    need(bool((branch.up.weight==0).all()) and float(consistency)==0.,'Initial U/residual penalty is nonzero')
    values=torch.autograd.grad(consistency,tuple(branch.parameters()),retain_graph=True,allow_unused=True)
    passed=all(v is None or bool(torch.isfinite(v).all() and (v==0).all()) for v in values)
    need(passed and all(p.grad is None for p in branch.parameters()),'Zero-U consistency gradient changed')
    return dict(passed=True,up_exactly_zero=True,residual_loss=0.,all_residual_gradients_zero=True)


def score_output(tokenizer,generated_ids,gold):
    """All generated failures remain in the denominator; EOS is never added."""
    ids=list(generated_ids)
    need(1<=len(ids)<=4 and all(type(i) is int and i>=0 for i in ids),'Invalid generated ID sequence')
    complete=ids[-1] in POLICY['native_eos']
    need(not any(i in POLICY['native_eos'] for i in ids[:-1]) and (complete or len(ids)==4),'Unexpected native stop')
    bodyids=ids[:-1] if complete else ids
    clean=not any(i in set(tokenizer.all_special_ids) for i in bodyids)
    body=tokenizer.decode(bodyids,skip_special_tokens=False)
    parsed=int(body.strip()) if clean and re.fullmatch('[0-9]+',body.strip()) else None
    return dict(answer_body=body,no_nonterminal_special_tokens=clean,prediction=parsed,
        parseable=parsed is not None,parsed_count_correct=parsed==gold,exact=complete and parsed==gold,
        completed=complete,truncated=not complete)


def strict_prediction(torch,processor,sample,result,index,cell):
    row=v7.prediction_record(torch,processor,sample,result,index,cell)
    row.update(score_output(processor.tokenizer,result['generated_ids'],sample['gold']),pair_id=sample.get('pair_id'))
    need(row['completed']==result['completed'] and row['truncated']==result['truncated'],'Native stop/scorer differs')
    return row


def validate_fresh(bindings,manifest):
    fresh=bind_manifest(bindings,FRESH/'main_manifest.json')
    need(Path(fresh['dataset_root']).resolve()==FRESH and fresh['purpose']=='main'
        and fresh['data_seed']==fresh['fresh_test_seed']==20261118
        and set(fresh['splits'])=={'test_N32','test_N64'} and len(fresh['source_manifest_sha256'])==30,
        'Wrong V18 fresh corpus, seed or exclusion inventory')
    proof=read(fresh['stage_audit_file']);bind_file(bindings,fresh['stage_audit_file'])
    flags=('passed','completed','all_published_records_match_dry_plan','all_fresh_content_exclusions_passed',
        'all_qa_hashes_and_gold_recounts_passed','all_image_hashes_and_dimensions_passed','all_image_semantic_links_passed',
        'training_dev_and_schedule_unchanged','training_pairing_unchanged')
    need(all(proof.get(k) is True for k in flags),'Fresh data preparation or semantic proof failed')
    need(proof['manifest_sha256']['main']==sha(FRESH/'main_manifest.json')
        and proof['semantic_audit_sha256']==fresh['audit_sha256'],'Fresh completed artifact binding differs')
    used={r['content_sha256'] for split in manifest['splits'].values() for r in split['samples']};families={}
    for cell,split in fresh['splits'].items():
        rows=split['samples'];n=int(cell.split('_N')[1])
        need(len(rows)==136 and Counter(r['gold'] for r in rows)==Counter({k:8 for k in range(17)}),'Fresh K support differs')
        for row in rows:
            need(row['n_frames']==n and row['content_sha256'] not in used,'Fresh context overlap or N mismatch')
            used.add(row['content_sha256']);families.setdefault(row['pair_id'],[]).append(row)
    need(len(families)==136 and all(len(rows)==2 and {r['n_frames'] for r in rows}=={32,64}
        and len({(r['question'],r['gold']) for r in rows})==1 for rows in families.values()),'Incomplete paired fresh families')
    return fresh


def bind_software(path,bindings):
    from scripts import profile_native_vision_v18_semantic_gate as software
    path=Path(path).resolve();summary=software.verify_profile(path);ref=bind_file(bindings,path)
    need(summary['hardware']['gpu']=='NVIDIA B200','Require the measured B200 software runtime')
    bind_file(bindings,summary['plan_file'],summary['plan_sha256'])
    for name,digest in summary['source_sha256'].items():bind_file(bindings,REPO/name,digest)
    return dict(ref,plan_file=summary['plan_file'],plan_sha256=summary['plan_sha256'],
        native_identity_sha256=summary['native_identity_sha256'],timing=summary['timing']),summary


def bind_prior(bindings):
    ref=bind_file(bindings,PRIOR);summary=read(PRIOR)
    need(summary['passed'] and summary['completed'] and summary['no_fit'] and summary['local_rows']==22568,
         'Require completed canonical V17 local-readout audit')
    bind_file(bindings,summary['analysis_file'],summary['analysis_sha256']);analysis=read(summary['analysis_file'])
    need(analysis['no_training_release'] and analysis['no_generated_aggregate_answers'] and analysis['VLM_calls']==0,
         'Prior local audit scope differs')
    return dict(summary=ref,analysis_file=summary['analysis_file'],analysis_sha256=summary['analysis_sha256'])


def load_gate_values(torch,gate_cache,device):
    need(sha(gate_cache['tensor_file'])==gate_cache['tensor_sha256'],'Original gate tensor archive changed')
    packet=torch.load(gate_cache['tensor_file'],map_location='cpu',weights_only=True)
    ids=packet['feature_ids'];need(packet['schema_version']==1 and ids==sorted(gate_cache['gates']) and len(ids)==9512,
        'Gate feature identity order or cardinality differs')
    for key,dtype in (('gates',torch.float32),('p0',torch.float64),('p1',torch.float64)):
        value=packet[key]
        need(value.dtype==dtype and value.shape==(9512,) and not value.requires_grad and bool(torch.isfinite(value).all())
            and v7.tensor_info(value)==gate_cache['tensor_info'][key],'Gate tensor metadata/value changed: '+key)
    need(torch.equal(packet['gates'],(packet['p1']-packet['p0']).clamp_min(0).float()),'Frozen gate rule differs')
    for row,fid in enumerate(ids):
        item=gate_cache['gates'][fid]
        need(item['row']==row and float(packet['gates'][row])==item['gate'] and float(packet['p0'][row])==item['p0']
            and float(packet['p1'][row])==item['p1'],'Gate scalar/feature ownership differs')
    return packet['gates'].to(device).detach(),{fid:i for i,fid in enumerate(ids)}


def check(args):
    out=OUT/f'train_check_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False);frozen=snapshot(out)
    import torch
    from transformers import AutoProcessor
    from scripts import stage_native_vision_v18_gates as gate_stage
    from scripts import native_vision_semantic_gate_runtime as deployment
    torch.set_num_threads(4);begin=time.perf_counter();bindings={};tests=test_helpers()
    manifest=bind_manifest(bindings,TRAIN)
    for path in (SCHEDULE,PAIRING):bind_file(bindings,path)
    dev=manifest['splits']['dev_N16']['samples']
    need(len(dev)==64 and Counter(r['gold'] for r in dev)==Counter({k:4 for k in range(16)}),'Development support differs')
    fresh=validate_fresh(bindings,manifest)
    cache_ref=bind_file(bindings,CACHE);cache=ancestor.cache_binding(CACHE)
    bind_file(bindings,cache['plan_file'],cache['plan_sha256'])
    for path,digest in {r['file']:r['file_sha256'] for r in cache['features'].values()}.items():bind_file(bindings,path,digest)
    gate_ref=bind_file(bindings,args.gate_cache);gates=gate_stage.verify_cache(args.gate_cache,ancestors=True,tensors=True)
    need(gates['parent_cache']==cache_ref and gates['complete'] and gates['training_only'], 'Gate cache is not bound to the canonical V10 features')
    for file,hashkey in (('tensor_file','tensor_sha256'),('proof_file','proof_sha256')):bind_file(bindings,gates[file],gates[hashkey])
    native_profile,software=bind_software(args.native_profile,bindings)
    need(gates['native_identity_sha256']==native_profile['native_identity_sha256']==object_sha(gates['native_identity']),
        'Training gates and deployed original probe have different native identities')
    processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,use_fast=False)
    need(deployment.digit_token_ids(processor.tokenizer)==(15,16),'Native gate tokenizer changed')
    for scene in cache['scenes'].values():
        need(scene['target_ids']==native.encode_target(processor.tokenizer,scene['gold']),'Cached complete native target differs')
    for row in dev+[r for split in fresh['splits'].values() for r in split['samples']]:
        need(native.encode_target(processor.tokenizer,row['gold'])[-1]==POLICY['target_eos'],'Fresh/dev native target EOS differs')
    pairing=pairing_binding(CACHE);states,index=v7.load_features(torch,cache,'cpu')
    gate_values,gate_index=load_gate_values(torch,gates,'cpu')
    for pair in pairing['pairs']:
        h,g,layout,applied,_=semantic_batch(torch,cache,states,index,pair['sids'],gates,gate_values,gate_index,'native_gate')
        need(torch.equal(g[layout['left']],g[layout['right']]) and not g.requires_grad,'Paired actual frozen query states differ')
        for side,sid in enumerate(pair['sids']):
            n=cache['scenes'][sid]['n_frames'];a,b=layout['offsets'][side:side+2]
            need(bool((applied[n:,a:b]==0).all()),'Padded raw-zero rows need explicit closed gates')
    save(out/'pairing.json',pairing);bind_file(bindings,out/'pairing.json')
    prior=bind_prior(bindings);orders={str(seed):object_sha(presentation_order(pairing['pairs'],seed)) for seed in POLICY['seeds']}
    need(len(set(orders.values()))==2,'Seed orders unexpectedly coincide')
    # Full ancestry was audited above. GPU validation retains exact hashes of
    # consumed cache/state/gate/data artifacts, without rereading old raw heads.
    runtime_paths={str(p.resolve()) for p in (TRAIN,SCHEDULE,PAIRING,CACHE,Path(args.gate_cache),FRESH/'main_manifest.json',
        out/'pairing.json',Path(gates['tensor_file']),Path(gates['proof_file']),Path(args.native_profile),Path(software['plan_file']))}
    runtime_paths.update(r['file'] for r in cache['features'].values())
    runtime_bindings={path:bindings[path] for path in sorted(runtime_paths)}
    plan=dict(schema_version=1,policy=POLICY,source_sha256=frozen,artifact_bindings=bindings,runtime_bindings=runtime_bindings,
        cache_binding=cache_ref,gate_cache_binding=gate_ref,native_identity=gates['native_identity'],
        native_identity_sha256=gates['native_identity_sha256'],native_profile=native_profile,
        pairing_file=str(out/'pairing.json'),pairing_sha256=sha(out/'pairing.json'),pairing_object_sha256=object_sha(pairing),
        order_sha256=orders,train_manifest=str(TRAIN),schedule_file=str(SCHEDULE),original_pairing_file=str(PAIRING),
        fresh_manifests={'main':str(FRESH/'main_manifest.json')},prior_result=prior,tests=tests,
        actual_frozen_pair_equality_passed=True,original_gate_prefix_and_padding_alignment_passed=True,
        no_model_loaded=True,elapsed_seconds=time.perf_counter()-begin)
    save(out/'plan.json',plan);(out/'plan.sha256').write_text(sha(out/'plan.json')+'\n');verify_plan(out/'plan.json',ancestors=True)
    save(out/'summary.json',dict(passed=True,completed=True,plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),
        source_sha256=frozen,tests=tests,no_model_loaded=True,elapsed_seconds=time.perf_counter()-begin))
    print(json.dumps(dict(passed=True,plan_file=str(out/'plan.json'))),flush=True)


def verify_plan(path,*,ancestors=True):
    path=Path(path).resolve();need(sha(path)==path.with_suffix('.sha256').read_text().strip(),'Frozen CPU plan sidecar differs')
    plan=read(path)
    need(plan['schema_version']==1 and plan['policy']==POLICY and plan['source_sha256']==sources(),'V18 plan source/policy changed')
    for artifact,digest in plan['artifact_bindings' if ancestors else 'runtime_bindings'].items():
        need(sha(artifact)==digest,'Frozen artifact changed: '+artifact)
    need(sha(plan['pairing_file'])==plan['pairing_sha256'] and object_sha(read(plan['pairing_file']))==plan['pairing_object_sha256'],
        'Frozen paired schedule changed')
    need(object_sha(plan['native_identity'])==plan['native_identity_sha256'],'Native identity digest differs')
    if ancestors:need(pairing_binding(plan['cache_binding']['file'])==read(plan['pairing_file']),'Paired metadata reconstruction differs')
    return plan


def verify_release(path,plan_path,plan):
    need(path is not None,'V18 mains require a measured independent release')
    path=Path(path).resolve();release=read(path)
    need(release['protocol']=='v18_native_semantic_gate_main_release' and release['passed']
        and release['plan_sha256']==sha(plan_path) and release['source_sha256']==sources()
        and release['per_main_seconds_cap']==2700 and release['campaign_gpu_seconds_cap']==14400,
        'V18 main release identity/resource policy differs')
    data=release['data_release'];need(sha(data['file'])==data['sha256'],'Independent data/source release changed')
    report_release=read(data['file'])
    need(report_release['passed'] and report_release['source_sha256']==release['report_source_sha256'],'Independent source/data gate failed')
    for name,digest in release['report_source_sha256'].items():need(sha(REPO/name)==digest,'Independent reporter source changed')
    fresh=plan['fresh_manifests']['main']
    need(release['fresh_manifest']==dict(file=fresh,sha256=plan['artifact_bindings'][fresh])
        and report_release['data_bindings'].get(fresh)==plan['artifact_bindings'][fresh], 'Release does not bind the staged fresh test')
    need(release['campaign_budget']['passed'] and set(release['profiles'])==set(POLICY['conditions']), 'Campaign or matched profile release failed')
    for condition,item in release['profiles'].items():
        sp=Path(item['directory'])/'summary.json';need(sha(sp)==item['summary_sha256'],'Training profile changed');summary=read(sp)
        need(summary['passed'] and summary['completed'] and summary['profile'] and summary['condition']==condition
            and summary['seed']==20 and summary['steps']==32 and summary['no_dev_or_test_evaluation']
            and summary['plan_sha256']==sha(plan_path) and summary['source_sha256']==sources()
            and summary['profile_calls']==PROFILE_CALLS and summary['active_cache_replay']['passed']
            and summary['checkpoint_roundtrip_passed'] and summary['zero_initialization_core_gradients']['passed']
            and summary['frozen_backbone_gradient_state_preserved'],'Invalid matched training profile')
    need(set(release['projections'])==set(POLICY['conditions']) and all(
        p['passed'] and 0<p['projected_seconds']<=2700 for p in release['projections'].values()),'Measured main timing exceeds cap')
    return dict(file=str(path),sha256=sha(path),fresh_manifest=release['fresh_manifest'])


def audit_runtime_evidence(torch,result,bundle,condition,native_identity_sha256):
    from gnnformer.parallel_local_semantic_gate import artifact_digest
    meta=bundle['metadata'];n=meta['n_frames'];width=meta['original_prompt_width'];ids=result['generated_ids']
    origin=result['origin_artifact'];captures=result['captures'];identity=result['metadata']['origin_identity']
    need(origin['artifact_sha256']==artifact_digest(origin) and origin['origin_identity']==identity
        and identity['sid']==meta['sid'] and identity['image_sha256']==meta['image_sha256']
        and identity['native_identity_sha256']==native_identity_sha256 and origin['mode']==condition
        and origin['n_local_rows']==n and len(captures)==len(ids),'Native origin/trajectory evidence differs')
    p0,p1,ng=origin['p0'],origin['p1'],origin['native_gates']
    need(p0.dtype==p1.dtype==torch.float64 and ng.dtype==torch.float32 and ng.shape==(n,)
        and torch.equal(ng,(p1-p0).clamp_min(0).float()),'Native original full-mass gate arithmetic differs')
    applied=ng if condition=='native_gate' else torch.ones_like(ng)
    for t,cap in enumerate(captures):
        need(cap['origin_identity']==identity and cap['query_indices']==[width-1 if t==0 else 0]
            and cap['stream_positions']==[width+t-1] and cap['mode']==condition
            and cap['n_local_rows']==n and torch.equal(cap['native_gates'],ng)
            and torch.equal(cap['applied_gates'],applied[:,None]),'Query capture changed original gates or ownership')
        shapes={'local_states':(n,1,3584),'global_states':(1,3584),'payload':(n,1,96),
            'messages':(n,1,96),'query':(1,96),'aggregate':(1,96),'preactivation':(1,96),
            'delta':(1,3584),'fused_global':(1,3584)}
        for key,shape in shapes.items():
            value=cap[key];expected=torch.float16 if key in ('local_states','global_states','fused_global') else torch.float32
            need(value.shape==shape and value.dtype==expected and not value.requires_grad
                and bool(torch.isfinite(value).all()),'Invalid query capture: '+key)
        need(bool((cap['payload'].abs()<=1).all()) and torch.equal(cap['messages'],cap['payload']*applied[:,None,None])
            and bool((cap['messages'][applied==0]==0).all()),'Captured local message bypasses gate/bound')
        need(torch.equal(cap['fused_global'],cap['global_states']+cap['delta'].half()),'Captured native cast/add changed')
    counters=result['counters'];steps=len(ids)
    need(all(counters[k]==steps for k in ('model','language','norm','head','fusion','broadcast'))
        and counters['visual']==1 and all(counters[k]==1 for k in ('probe_norm','probe_head','probability')),
        'Natural call/probe inventory differs')
    return dict(passed=True,origin_artifact_sha256=origin['artifact_sha256'],origin_identity=identity,
        capture_count=len(captures),original_gate_reuse=True,closed_message_coordinates_zero=True)


def evaluate(torch,model,processor,branch,rows,out,label,dataout,condition,native_identity_sha256):
    from scripts import native_vision_semantic_gate_runtime as deployment
    records=[];raw=[];begin=time.perf_counter();io_seconds=0.;destination=dataout/label;destination.mkdir()
    for cell,sample in rows:
        prep=time.perf_counter();bundle=deployment.prepare_scene(processor,sample);preparation=time.perf_counter()-prep
        result=deployment.generate_native(model,processor,branch,bundle,mode=condition,
            native_identity_sha256=native_identity_sha256,capture=True,cpu=True)
        values=result['raw_logits'];ids=result['generated_ids']
        need(values.dtype==torch.float32 and values.shape==(len(ids),152064) and bool(torch.isfinite(values).all())
            and torch.equal(values,values.half().float()) and values.argmax(-1).tolist()==ids,
            'Raw native logits are not exact FP16 promotion/raw-argmax output')
        evidence=audit_runtime_evidence(torch,result,bundle,condition,native_identity_sha256)
        row=strict_prediction(torch,processor,sample,result,len(raw),cell)
        tick=time.perf_counter();path=destination/f'trajectory_{len(raw):03d}.pt'
        torch.save(dict(schema_version=1,**result),path);digest=sha(path);io_seconds+=time.perf_counter()-tick
        row.update(raw_file=str(path),raw_sha256=digest,raw_dtype='torch.float32',preprocessing_seconds=preparation,
            origin_artifact_sha256=evidence['origin_artifact_sha256'],capture_count=evidence['capture_count'],
            gate_evidence=evidence,fusion_audit=result['fusion_audit'])
        records.append(row);raw.append(values)
        if len(records)%16==0:
            save(out/(label+'_progress.json'),dict(label=label,processed=len(records),total=len(rows),rows=records))
            print(json.dumps(dict(evaluation=label,processed=len(records),total=len(rows))),flush=True)
    tick=time.perf_counter();rawfile=dataout/(label+'_raw.pt');torch.save(dict(schema_version=1,raw_logits=raw),rawfile)
    digest=sha(rawfile);io_seconds+=time.perf_counter()-tick
    result=dict(label=label,rows=records,n=len(records),exact_count=sum(r['exact'] for r in records),
        first_token_nll=sum(r['first_token_nll'] for r in records)/len(records),seconds=time.perf_counter()-begin,
        raw_file=str(rawfile),raw_sha256=digest,raw_dtype='torch.float32',archive_io_seconds=io_seconds,
        all_origin_artifacts_and_prefix_captures_retained=True)
    save(out/(label+'.json'),result);return result


def cpu_tree(torch,value):
    """Archive nested native evidence without retaining inference/GPU tensors."""
    if isinstance(value,torch.Tensor):
        with torch.inference_mode(False),torch.no_grad():return value.detach().cpu().clone()
    if isinstance(value,dict):return {key:cpu_tree(torch,item) for key,item in value.items()}
    if isinstance(value,list):return [cpu_tree(torch,item) for item in value]
    if isinstance(value,tuple):return tuple(cpu_tree(torch,item) for item in value)
    return value


def active_cache_replay(torch,model,processor,branch,cache,states,index,gate_cache,gate_values,gate_index,
                        manifest,dataout,condition,native_identity_sha256):
    from scripts import native_vision_semantic_gate_runtime as deployment
    from scripts.cache_native_vision_v7_features import replay_metrics
    from gnnformer.parallel_local_semantic_gate import artifact_digest
    observations=[];raw=[];norm=model.model.language_model.norm;head=model.lm_head
    for k in (0,9,10,16):
        sample=sorted((r for r in manifest['splits']['train_N16']['samples'] if r['gold']==k),key=lambda r:r['sid'])[0]
        h,g,layout,cached_gates,origins=semantic_batch(torch,cache,states,index,[sample['sid']]*2,
            gate_cache,gate_values,gate_index,condition)
        targets=cache['scenes'][sample['sid']]['target_ids'];artifact=None;origin_sha=None
        for t in range(len(targets)):
            bundle=deployment.prepare_scene(processor,sample,prefix_ids=targets[:t])
            actual=deployment.forward_native(model,processor,branch,bundle,mode=condition,
                native_identity_sha256=native_identity_sha256,origin_artifact=artifact,capture=True,cpu=False)
            artifact=actual['origin_artifact'];digest=artifact_digest(artifact)
            if t==0:origin_sha=digest
            need(digest==origin_sha and actual['counters']['probe_head']==int(t==0)
                and actual['counters']['probe_norm']==int(t==0) and actual['counters']['probability']==int(t==0),
                'Training profile recomputed or changed a gate at a target prefix')
            with torch.inference_mode():
                cap=actual['fusion'];ag,al=actual['global_states'],actual['local_states']
                live_delta=branch(al,ag,gates=cap['applied_gates'],output_dtype=torch.float32)
                full=torch.cat([al,ag.unsqueeze(0)],dim=0)
                need(full.shape==(17,1,3584),'Preserve complete native head batch shape')
                full[-1]=full[-1]+live_delta.to(full.dtype)
                replay=head.forward(norm.forward(full))
                cached_delta=branch(h[:,t:t+1],g[t:t+1],gates=cached_gates[:,t:t+1],output_dtype=torch.float32)
                cached=head.forward(norm.forward((g[t:t+1]+cached_delta.to(g.dtype)).unsqueeze(0)))[0]
            metrics=replay_metrics(torch,actual['native_logits'][:,0],replay[:,0])
            drift=replay_metrics(torch,actual['global_logits'][None],cached)
            observations.append(dict(sid=sample['sid'],gold=k,n_frames=16,prefix_ids=targets[:t],position=t,
                native_head_replay=metrics,cached_vs_native_descriptive=drift,cached_vs_native_gate=False,
                counters=actual['counters'],native_metadata=actual['metadata'],origin_artifact_sha256=digest,
                cached_empty_feature_ids=origins[0],original_gate_reused=True,
                cached_vs_live_gate_max_abs=float((gate_values[torch.tensor([gate_index[f] for f in origins[0]],device=g.device)]
                    -artifact['native_gates']).abs().max())))
            raw.append(cpu_tree(torch,dict(actual=actual,replayed=replay,cached=cached,
                cached_gates=cached_gates[:,t:t+1])))
    need(len(observations)==10 and sum(r['counters']['model'] for r in observations)==10
        and sum(r['counters']['visual'] for r in observations)==10
        and sum(r['counters']['probe_head'] for r in observations)==4,'Profile native/probe inventory differs')
    path=dataout/'active_cache_replay.pt';torch.save(dict(schema_version=1,observations=raw),path)
    return dict(passed=all(r['native_head_replay']['passed'] for r in observations),observations=observations,
        counts_covered=[0,9,10,16],prefixes_checked=10,raw_file=str(path),raw_sha256=sha(path),
        cached_numerical_failures=sum(not r['cached_vs_native_descriptive']['passed'] for r in observations),
        native_model_calls=10,vision_calls=10,original_probe_norm_calls=4,original_probe_head_calls=4,
        original_probability_calls=4,standalone_head_calls=20,same_state_rows=170,
        cached_differences_descriptive=True,gate_values_never_replaced=True)


def live_native_identity(torch,model,processor,cache,identity):
    import transformers
    from scripts.stage_native_vision_v10_features import runtime_identity,model_metadata,native_api
    from scripts.probe_native_vision_v2_prefix import fingerprint
    need(runtime_identity()==cache['runtime']==identity['runtime'] and model_metadata()==cache['model']==identity['model']
        and fingerprint(processor,str(transformers.__version__))==cache['processor']==identity['processor'],
        'Live model/runtime/processor differs from frozen features and gates')
    need(native_api(processor)[2]==cache['native_api']==identity['native_api'],'Installed native layout/processor code differs')
    norm=model.model.language_model.norm;head=model.lm_head
    actual=dict(norm=v7.tensor_info(norm.weight),head=v7.tensor_info(head.weight))
    need(actual['norm']==identity['norm_weight'] and actual['head']==identity['head_weight'],
        'Actual loaded native FP16 norm/head differ from gate identity')
    need(float(norm.variance_epsilon)==identity['rms_norm_eps'],'Actual native norm epsilon differs')
    need(native.generation_policy(model,processor.tokenizer)[1]['native_eos_token_ids']==POLICY['native_eos'],
        'Native EOS generation contract differs')
    need(torch.cuda.get_device_name(0)=='NVIDIA B200','Measured runtime requires B200')
    return actual


def run(args):
    job=os.environ['SLURM_JOB_ID'];runid=f'{"profile" if args.profile else "run"}_{args.condition}_s{args.seed}_{job}'
    out=OUT/runid;out.mkdir(parents=True,exist_ok=False);frozen=snapshot(out)
    import torch
    from gnnformer.runtime import load_runtime
    from gnnformer.parallel_local_semantic_aggregation import ParallelLocalSemanticAggregation
    from scripts import stage_native_vision_v18_gates as gate_stage
    from scripts import native_vision_semantic_gate_runtime as deployment
    torch.set_num_threads(4);start=time.perf_counter();args.plan=args.plan.resolve();plan=verify_plan(args.plan,ancestors=False)
    need(args.condition in POLICY['conditions'] and args.seed in POLICY['seeds'] and (not args.profile or args.seed==20),
        'Unregistered condition/seed/profile')
    release=None if args.profile else verify_release(args.main_release,args.plan,plan)
    dataout=DATA/'native_aggregation_vlm_v18'/runid;dataout.mkdir(parents=True,exist_ok=False)
    ckpt=CKPT/runid;ckpt.mkdir(parents=True,exist_ok=False)
    loadstart=time.perf_counter();cache=read(plan['cache_binding']['file'])
    gates=gate_stage.verify_cache(plan['gate_cache_binding']['file'],ancestors=False,tensors=False)
    need(gates['parent_cache']==plan['cache_binding'] and gates['native_identity']==plan['native_identity'],
        'Consumed cache/gate native identity changed')
    runtime=load_runtime(str(MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda')
    model=runtime.model.eval().requires_grad_(False);native.native_contract(model)
    native_weights=live_native_identity(torch,model,runtime.processor,cache,plan['native_identity'])
    states,index=v7.load_features(torch,cache,runtime.device);gate_values,gate_index=load_gate_values(torch,gates,runtime.device)
    torch.cuda.synchronize();loadseconds=time.perf_counter()-loadstart
    torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    branch=ParallelLocalSemanticAggregation().to(runtime.device);deployment.native_contract(model,branch)
    initialized=v7.state_info(branch);initial={k:v.detach().cpu().clone() for k,v in branch.state_dict().items()}
    need(sum(p.numel() for p in branch.parameters())==1041600,'Trainable core capacity differs')
    optimizer=torch.optim.AdamW(branch.parameters(),lr=.001,weight_decay=0.)
    pairing=read(plan['pairing_file']);order=presentation_order(pairing['pairs'],args.seed)
    need(object_sha(order)==plan['order_sha256'][str(args.seed)],'Frozen presentation order differs')
    save(out/'presentations.json',order)
    config=dict(run_id=runid,arm='parallel',condition=args.condition,mode=args.condition,seed=args.seed,profile=args.profile,
        policy=POLICY,consistency_coefficient=1.,slurm_job_id=job,plan_file=str(args.plan),plan_sha256=sha(args.plan),
        source_sha256=frozen,cache_binding=plan['cache_binding'],gate_cache_binding=plan['gate_cache_binding'],
        native_identity=plan['native_identity'],native_identity_sha256=plan['native_identity_sha256'],native_weight_identity=native_weights,
        native_profile=plan['native_profile'],pairing_file=plan['pairing_file'],pairing_sha256=plan['pairing_sha256'],
        prior_result=plan['prior_result'],initialized=initialized,initialized_sha256=object_sha(initialized),
        presentations_sha256=sha(out/'presentations.json'),order_sha256=object_sha(order),model=cache['model'],runtime=cache['runtime'],
        processor=cache['processor'],native_dtypes=cache['native_dtypes'],model_and_features_load_seconds=loadseconds,
        hardware=dict(name=torch.cuda.get_device_name(0),cuda=torch.version.cuda,torch=str(torch.__version__)),
        checkpoint_directory=str(ckpt),data_directory=str(dataout),main_release=release)
    save(out/'config.json',config);model_versions={name:p._version for name,p in model.named_parameters()}
    setup=time.perf_counter()-start;trainlog=[];first_gradients=[];isolation=[];zero=None;io_seconds=0.
    exercised={name:False for name,_ in branch.named_parameters()};steps=32 if args.profile else 4590
    for step in range(1,steps+1):
        begin=time.perf_counter();rows=order[(step-1)*16:step*16];sids=[r['sid'] for r in rows]
        need(len(rows)==16 and all(rows[i]['pair_id']==rows[i+1]['pair_id'] and rows[i]['pair_side']==0
            and rows[i+1]['pair_side']==1 for i in range(0,16,2)),'Batch split an intact training pair')
        local,g,layout,applied,origins=semantic_batch(torch,cache,states,index,sids,gates,gate_values,gate_index,args.condition)
        audit_gate=step in (1,2,32)
        if audit_gate:applied.requires_grad_(True)
        optimizer.zero_grad(set_to_none=True);rate=lr(step)
        for group in optimizer.param_groups:group['lr']=rate
        total,ce,consistency,path_loss,components=losses(torch,branch,local,g,layout,applied,
            model.model.language_model.norm,model.lm_head,sids)
        if step==1:zero=zero_gradients(torch,branch,consistency)
        if audit_gate:
            gate_gradient=torch.autograd.grad(total,applied,allow_unused=True,retain_graph=True)[0]
            need(gate_gradient is None and not states.requires_grad and not gate_values.requires_grad,'Gate or cached backbone acquired a gradient path')
            isolation.append(dict(step=step,passed=True,gate_gradient_is_none=True,frozen_features=True))
        total.backward()
        need(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in branch.parameters()),'Missing/nonfinite core gradient')
        if args.profile or step<=2:
            for name,p in branch.named_parameters():exercised[name]|=bool((p.grad!=0).any())
        if step<=2:first_gradients.append(dict(step=step,pre_clip={name:v7.tensor_info(p.grad) for name,p in branch.named_parameters()}))
        gradient_norm=torch.nn.utils.clip_grad_norm_(branch.parameters(),1.);need(bool(torch.isfinite(gradient_norm)),'Nonfinite clip norm')
        optimizer.step();need(all(bool(torch.isfinite(p).all()) for p in branch.parameters()),'Nonfinite updated core')
        need(not any(p.requires_grad or p.grad is not None for p in model.parameters()),'Frozen backbone gradient state changed')
        torch.cuda.synchronize()
        trainlog.append(dict(step=step,lr=rate,loss=float(total),ce_loss=float(ce),consistency_loss=float(consistency),
            weighted_consistency_loss=float(consistency),consistency_coefficient=1.,path_loss=float(path_loss),
            path_loss_coefficient=0.,gradient_norm=float(gradient_norm),clipped=float(gradient_norm)>1.,seconds=time.perf_counter()-begin,
            target_ids=layout['targets'],sids=sids,pair_ids=[rows[i]['pair_id'] for i in range(0,16,2)],
            epochs=[rows[i]['epoch'] for i in range(0,16,2)],local_empty_feature_ids=origins,**components))
        if step%100==0:
            tick=time.perf_counter();save(out/'training.json',trainlog);io_seconds+=time.perf_counter()-tick
            print(json.dumps(dict(step=step,total=float(total),ce=float(ce),consistency=float(consistency))),flush=True)
    need(all(exercised.values()),'Not every core parameter received an exercised gradient')
    need(len(trainlog)==steps and (args.profile or sum(len(r['target_ids']) for r in trainlog)==177120),
        'Complete training step/valid-target inventory differs')
    tick=time.perf_counter();save(out/'training.json',trainlog);save(out/'first_gradients.json',first_gradients)
    save(out/'gradient_isolation.json',isolation);io_seconds+=time.perf_counter()-tick
    optimizer.zero_grad(set_to_none=True);del optimizer,total,ce,consistency,path_loss
    branch.eval().requires_grad_(False);final_state=v7.state_info(branch)
    checkpoint=ckpt/('profile.pt' if args.profile else 'final.pt')
    torch.save(dict(branch=branch.state_dict(),step=steps,config=config),checkpoint);checkpoint_sha=sha(checkpoint)
    branch.load_state_dict(initial);need(v7.state_info(branch)==initialized,'Initial checkpoint restore failed')
    saved=torch.load(checkpoint,map_location=runtime.device,weights_only=True);branch.load_state_dict(saved['branch'])
    need(saved['step']==steps and saved['config']==config and v7.state_info(branch)==final_state,'Final checkpoint restore failed')
    before_eval=v7.state_info(branch);core_versions={name:p._version for name,p in branch.named_parameters()}
    result=dict(config,passed=True,completed=True,computational_integrity_passed=True,steps=steps,
        zero_initialization_core_gradients=zero,all_parameter_gradients_exercised=exercised,
        checkpoint_roundtrip_passed=True,gradient_isolation_file=str(out/'gradient_isolation.json'),
        gradient_isolation_sha256=sha(out/'gradient_isolation.json'),
        training_seconds=sum(r['seconds'] for r in trainlog),step_seconds_max_steady=max(r['seconds'] for r in trainlog[4:]),
        training_file=str(out/'training.json'),training_sha256=sha(out/'training.json'),
        first_gradients_file=str(out/'first_gradients.json'),first_gradients_sha256=sha(out/'first_gradients.json'))
    manifest=read(plan['train_manifest'])
    if args.profile:
        replay=active_cache_replay(torch,model,runtime.processor,branch,cache,states,index,gates,gate_values,gate_index,
            manifest,dataout,args.condition,plan['native_identity_sha256'])
        save(out/'active_cache_replay.json',replay)
        result.update(profile_calls=PROFILE_CALLS,active_cache_replay=replay,no_dev_or_test_evaluation=True,
            profile_checkpoint=str(checkpoint),profile_checkpoint_sha256=checkpoint_sha)
        if not replay['passed']:
            result.update(passed=False,computational_integrity_passed=False,elapsed_seconds=time.perf_counter()-start)
            save(out/'summary.json',result);need(False,'Native same-captured-state head replay failed; raw evidence retained')
    else:
        devrows=[('dev_N16',r) for r in manifest['splits']['dev_N16']['samples']]
        dev=evaluate(torch,model,runtime.processor,branch,devrows,out,'dev_final',dataout,args.condition,plan['native_identity_sha256'])
        selected=dict(step=4590,exact_count=dev['exact_count'],nll=dev['first_token_nll'],checkpoint=str(checkpoint),
            checkpoint_sha256=checkpoint_sha,parameter_sha256=object_sha(final_state),dev_file=str(out/'dev_final.json'),dev_sha256=sha(out/'dev_final.json'))
        save(out/'selection.json',dict(rule=POLICY['selection'],development=[selected],selected=selected,dev_descriptive_only=True))
        fresh=read(plan['fresh_manifests']['main']);testrows=[('main_'+cell,r) for cell,split in fresh['splits'].items() for r in split['samples']]
        need(len(devrows)==64 and len(testrows)==272,'Final native evaluation denominators differ')
        evaluate(torch,model,runtime.processor,branch,testrows,out,'test',dataout,args.condition,plan['native_identity_sha256'])
        result.update(selected=selected,development=[selected],dev_descriptive_only=True,native_dev_count=64,native_test_count=272,
            test_file=str(out/'test.json'),test_sha256=sha(out/'test.json'))
    after_eval=v7.state_info(branch)
    need(before_eval==after_eval==final_state and core_versions=={name:p._version for name,p in branch.named_parameters()}
        and model_versions=={name:p._version for name,p in model.named_parameters()}
        and not any(p.requires_grad or p.grad is not None for p in model.parameters())
        and sha(checkpoint)==checkpoint_sha,'Evaluation changed final core/native endpoint or checkpoint file')
    endpoint=dict(schema_version=1,step=steps,checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_sha,
        before_evaluation=before_eval,after_evaluation=after_eval,parameter_sha256=object_sha(final_state),
        checkpoint_matches_deployed_endpoint=True,core_versions_unchanged=True,native_versions_unchanged=True)
    save(out/'final_endpoint.json',endpoint)
    validation=time.perf_counter();verify_plan(args.plan,ancestors=False)
    if release is not None:need(sha(release['file'])==release['sha256'],'Main release changed during run')
    result.update(final_endpoint_file=str(out/'final_endpoint.json'),final_endpoint_sha256=sha(out/'final_endpoint.json'),
        frozen_backbone_gradient_state_preserved=True,final_endpoint_unchanged=True,
        setup_before_training_seconds=setup,setup_includes_model_and_feature_load=True,training_log_io_seconds=io_seconds,
        final_validation_seconds=time.perf_counter()-validation,elapsed_seconds=time.perf_counter()-start)
    save(out/'summary.json',result)
    print(json.dumps(dict(run_id=runid,completed=True,elapsed_seconds=result['elapsed_seconds'])),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true');mode.add_argument('--profile',action='store_true');mode.add_argument('--run',action='store_true')
    parser.add_argument('--gate-cache',type=Path,default=GATE_CACHE);parser.add_argument('--native-profile',type=Path)
    parser.add_argument('--plan',type=Path);parser.add_argument('--main-release',type=Path)
    parser.add_argument('--condition',choices=POLICY['conditions']);parser.add_argument('--seed',type=int,choices=POLICY['seeds'])
    args=parser.parse_args();native.require_slurm(gpu=not args.check)
    if args.check:
        need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),'CPU preparation requires CPU Slurm')
        need(args.native_profile is not None,'CPU training check requires a completed native software profile');check(args)
    else:
        need(args.plan is not None and args.condition is not None and args.seed is not None,'GPU modes require plan/condition/seed');run(args)


if __name__=='__main__':main()

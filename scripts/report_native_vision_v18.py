"""Independent HELD V18 native semantic-gate audit; CPU Slurm only.

The report binds every final trained core and native answer/capture trajectory.
Only the vision milestone can pass; reasoning composition is never inferred.
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
from scripts import report_native_vision_v14 as v14
from scripts import stage_native_vision_v18_test as fresh_stage
from scripts import stage_native_vision_v18_gates as gate_stage
from scripts import profile_native_vision_v18_semantic_gate as software
need,read,sha,objsha,save,close,bind,ledger,tensor_info=(prior.need,prior.read,prior.sha,prior.objsha,prior.save,prior.close,prior.bind,prior.ledger,prior.tensor_info)
DATA,MODEL=prior.DATA,prior.MODEL
OUT=REPO/'outputs/native_aggregation_vlm/v18'
CKPT=Path('/mnt/ckpts/gabriele/gnn_transformer/native_aggregation_vlm_v18')
CACHE=DATA/'v10_parallel_local/feature_cache.json'
GATE_CACHE=DATA/'v18_semantic_gates/feature_cache.json'
CONDITIONS=('native_gate','all_open');SEEDS=(20,21);PARTITIONS=v14.PARTITIONS
OWN=('scripts/report_native_vision_v18.py','slurm/native_vision_v18_report.sbatch',
     'scripts/prepare_native_vision_v18_release.py','slurm/native_vision_v18_release.sbatch')
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


def sources():
    from scripts import train_native_vision_v18 as training
    return {**v10.sources(),**fresh_stage.source_hashes(),**gate_stage.sources(),**software.sources(),
            **training.sources(),**{name:sha(REPO/name) for name in OWN}}


def snapshot(out):
    frozen=sources();(out/'code').mkdir()
    for name,digest in frozen.items():
        target=out/'code'/name.replace('/','_');target.write_bytes((REPO/name).read_bytes());bind(target,digest)
    save(out/'source_hashes.json',frozen);return frozen


learning_rate=v10.learning_rate
independent_pairs=v10.independent_pairs
expected_order=v10.expected_order
metrics=v14.metrics
summarize=v14.summarize
independent_layout=v10.independent_layout


def get_pairing(data,cache):
    parent=v10.get_pairing(data,cache)
    return dict(parent,protocol='v18_unchanged_v10_pairing_new_seeds',seeds=list(SEEDS))


def integer_decision(treated,control):
    n32=treated['cells']['main_test_N32']['correct'];n64=treated['cells']['main_test_N64']['correct']
    d64=n64-control['cells']['main_test_N64']['correct'];d32=n32-control['cells']['main_test_N32']['correct']
    larger=treated['partitions']['N64']['K9_16']['correct'];primary=d64>=7 and d32>=-6
    practical=primary and n32>=123 and n64>=109 and larger>=52
    return dict(n64_additional_correct=d64,n64_difference_pp=100*d64/136,n32_additional_correct=d32,n32_difference_pp=100*d32/136,
        n64_gain_at_least_5pp=d64>=7,n32_loss_at_most_5pp=d32>=-6,primary=primary,practical=practical,
        native_gate_n32_correct=n32,native_gate_n64_correct=n64,native_gate_n64_K9_16_correct=larger,
        native_gate_n32_at_least90pct=n32>=123,native_gate_n64_at_least80pct=n64>=109,native_gate_n64_K9_16_at_least80pct=larger>=52)


def audit_data():
    # Revalidate the unchanged corpus as V10; never relabel its test predictions
    # as V18. Only its frozen training/dev rows are used in the new experiment.
    previous=v10.audit_data();manifest_path=DATA/'v18_fresh/main_manifest.json';fresh=read(manifest_path)
    need(fresh['schema_version']==1 and fresh['purpose']=='main' and fresh['dataset_root']==str(DATA/'v18_fresh')
         and fresh['data_seed']==fresh['fresh_test_seed']==20261118 and fresh['standard_downstream_resize']==392
         and fresh['training_dev_and_schedule_unchanged'] is True and fresh['training_pairing_unchanged'] is True
         and fresh['saturation_exceptions']==[],'V18 fresh data protocol differs')
    current,_,excluded=fresh_stage.prior_inputs()
    need(len(current['source_manifest_sha256'])==30 and fresh['source_manifest_sha256']==current['source_manifest_sha256']
         and fresh['protected_source_sha256']==current['protected_source_sha256'],'Exact30 prior exclusions changed')
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
    fresh_summary=dict(passed=True,fresh_contexts=272,test_families=136,prior_manifests=30,data_seed=20261118,
        all_qa_image_semantic_extensions_verified=True,historical_reuse_count=0,local_visual_atoms_may_recur=True,
        manifest_file=str(manifest_path),manifest_sha256=sha(manifest_path),audit_file=fresh['audit_file'],audit_sha256=fresh['audit_sha256'])
    summary=dict(previous['summary'],data_bindings=bindings,prior_manifests=30,fresh_data=fresh_summary,
        training_and_development_unchanged=True,test_dataset_root=str(DATA/'v18_fresh'),
        fresh_test_seed=20261118,old_v10_test_is_provenance_only=True)
    return dict(previous,summary=summary,fresh=fresh,test=rows)

def bootstrap(all_rows,replicates=10000,seed=20261118):
    import numpy as np
    rng=np.random.default_rng(seed);reference=all_rows['native_gate',20];groups=defaultdict(dict)
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
                a,b=samples['native_gate',fit_seed],samples['all_open',fit_seed]
                cell[str(fit_seed)]=dict(difference_ci95_pp=(100*np.quantile(a-b,[.025,.975])).tolist(),
                    native_gate_ci95_pp=(100*np.quantile(a,[.025,.975])).tolist(),all_open_ci95_pp=(100*np.quantile(b,[.025,.975])).tolist())
            pooled=sum(samples['native_gate',s]-samples['all_open',s] for s in SEEDS)/2
            cell['pooled_fixed_two_seeds']=dict(difference_ci95_pp=(100*np.quantile(pooled,[.025,.975])).tolist());endpoints[f'N{n}']=cell
        result[label]=endpoints
    return dict(replicates=replicates,seed=seed,unit='Complete N32-to-N64 family stratified by K; shared draws across all lengths, conditions, subgroups and fixed seeds',
        inference_scope='Conditional example uncertainty for these fitted seeds, not a training-seed population',results=result)


def verify_cache(path,data,tokenizer):
    cache,audit=v10.verify_cache(path,data,tokenizer)
    need(Path(path).resolve()==CACHE,'Only the canonical V10 native feature cache is eligible')
    return cache,audit


def verify_gates(cache):
    """Independently reconstruct every original image/question/prefix mapping."""
    import torch
    value=gate_stage.verify_cache(GATE_CACHE,ancestors=True,tensors=True);plan=read(cache['plan_file'])
    need(value['parent_cache']==dict(file=str(CACHE),sha256=sha(CACHE)),'Gate cache parent differs')
    original={fid for fid,f in plan['features'].items() if f['kind']=='local' and f['prefix_ids']==[]}
    mapping={fid:objsha(['local',f['pair_id'],[]]) for fid,f in plan['features'].items() if f['kind']=='local'}
    need(len(original)==9512 and len(mapping)==32686 and set(value['gates'])==original
         and value['prefix_to_empty_feature_id']==mapping,'Gate inventory is not every original native local feature')
    for sid,scene in cache['scenes'].items():
        expected=[ids[0] for ids in scene['local_feature_ids']]
        need(value['scenes'][sid]['local_empty_feature_ids']==expected,'Scene origin occurrence order differs')
        for fid in expected:
            f=plan['features'][fid];pair=plan['pairs'][f['pair_id']];entry=value['gates'][fid]
            need(entry['pair_id']==f['pair_id'] and entry['image_sha256']==pair['image_sha256']
                 and entry['question_sha256']==objsha(f['question']) and entry['hidden_sha256']==cache['features'][fid]['state_sha256'],
                 'Gate image/question/native hidden ownership differs')
    packet=torch.load(value['tensor_file'],map_location='cpu',weights_only=True)
    need(packet['feature_ids']==sorted(original) and packet['gates'].dtype==torch.float32 and packet['p0'].dtype==packet['p1'].dtype==torch.float64,
         'Original gate tensor layout/dtype differs')
    need(bool(((packet['p0']>=0)&(packet['p1']>=0)&(packet['p0']+packet['p1']<=1+1e-12)).all())
         and torch.equal(packet['gates'],(packet['p1']-packet['p0']).clamp_min(0).float()),'Full-vocabulary gate scalar arithmetic differs')
    audit=dict(file=str(GATE_CACHE),sha256=sha(GATE_CACHE),parent_cache=value['parent_cache'],native_identity_sha256=value['native_identity_sha256'],
        original_features=9512,local_prefix_features=32686,training_scenes=1782,all_original_occurrences_and_prefix_maps_verified=True,
        tensor_file=value['tensor_file'],tensor_sha256=value['tensor_sha256'],proof_file=value['proof_file'],proof_sha256=value['proof_sha256'],
        no_gate_label_calculation=True,no_classifier_fitting=True)
    return value,audit


def verify_software_timing(path):
    value=software.verify_profile(path);rows=value['timing']['rows']
    times={str(n):max(r['four_token_seconds_bound'] for r in rows if r['n_frames']==n) for n in (16,64)}
    need(value['hardware']['gpu']=='NVIDIA B200' and all(math.isfinite(v) and v>0 for v in times.values()),'Wrong software hardware/timing')
    return dict(summary=value,file=str(Path(path).resolve()),sha256=sha(path),T16=times['16'],T64=times['64'],
        native_identity_sha256=value['native_identity_sha256'],shared_worst_mode_four_token_seconds=times)


def score_sequence(tokenizer,ids,gold):
    need(isinstance(ids,list) and 1<=len(ids)<=4 and all(type(i) is int and i>=0 for i in ids),'Malformed generated token list')
    complete=ids[-1] in (151645,151643)
    need(not any(i in (151645,151643) for i in ids[:-1]) and (complete or len(ids)==4),'Native EOS/length termination differs')
    body_ids=ids[:-1] if complete else ids;clean=not any(i in set(tokenizer.all_special_ids) for i in body_ids)
    body=tokenizer.decode(body_ids,skip_special_tokens=False);stripped=body.strip()
    prediction=int(stripped) if clean and re.fullmatch(r'[0-9]+',stripped) else None
    return dict(answer_body=body,no_nonterminal_special_tokens=clean,prediction=prediction,parseable=prediction is not None,
        parsed_count_correct=prediction==gold,exact=complete and prediction==gold,completed=complete,truncated=not complete)


def original_identity(meta,native_identity_sha256):
    keys=('arm','n_frames','row_kinds','row_count','global_row','local_elements','original_prompt_width',
          'row_prompt_tokens','local_prompt','global_prompt','resize')
    binding=dict(inputs=meta['input_identity'],metadata={k:meta[k] for k in keys})
    return dict(sid=meta['sid'],question_sha256=meta['question_sha256'],image_sha256=meta['image_sha256'],
                prompt_width=meta['original_prompt_width'],input_identity_sha256=objsha(binding),native_identity_sha256=native_identity_sha256)


def check_origin(artifact,identity,condition):
    """Rescore full native vocabulary in FP64; record CPU/GPU roundoff explicitly."""
    import torch
    from gnnformer.parallel_local_semantic_gate import GATE_RULE
    n=len(identity['image_sha256']);h=3584;origin=identity['prompt_width']-1
    need(artifact['schema_version']==1 and artifact['mode']==condition and artifact['gate_rule']==GATE_RULE
         and artifact['origin_identity']==identity and artifact['n_local_rows']==n and artifact['zero_token_id']==15
         and artifact['one_token_id']==16 and artifact['origin_query_index']==artifact['origin_stream_position']==origin
         and artifact['probe_shape']==[n+1,1,h],'Origin artifact metadata/ownership differs')
    digest=objsha({k:tensor_info(v) if isinstance(v,torch.Tensor) else v for k,v in artifact.items() if k!='artifact_sha256'})
    need(digest==artifact['artifact_sha256'],'Original native artifact tensor/metadata digest differs')
    for key,shape in [('origin_hidden',(n+1,1,h)),('origin_normalized',(n+1,1,h)),('origin_logits',(n+1,1,152064))]:
        x=artifact[key];need(isinstance(x,torch.Tensor) and x.dtype==torch.float16 and x.shape==shape
                            and not x.requires_grad and bool(torch.isfinite(x).all()),'Malformed original native tensor: '+key)
    p0,p1,gates=artifact['p0'],artifact['p1'],artifact['native_gates']
    need(p0.dtype==p1.dtype==torch.float64 and gates.dtype==torch.float32 and all(x.shape==(n,) and not x.requires_grad
         and bool(torch.isfinite(x).all()) for x in (p0,p1,gates)),'Malformed original native probability/gate vectors')
    need(bool(((p0>=0)&(p1>=0)&(p0+p1<=1+1e-12)).all()) and torch.equal(gates,(p1-p0).clamp_min(0).float()),
         'Stored original gate formula differs')
    x=artifact['origin_logits'][:n,0].double();z=torch.logsumexp(x,-1)
    a=(x[:,15]-z).exp();b=(x[:,16]-z).exp()
    need(torch.allclose(a,p0,rtol=1e-12,atol=1e-15) and torch.allclose(b,p1,rtol=1e-12,atol=1e-15),
         'Saved probabilities are not native full-vocabulary probabilities within FP64 roundoff')
    return dict(passed=True,rows=n,artifact_sha256=digest,probability_max_abs_difference=max(float((a-p0).abs().max()),float((b-p1).abs().max())),
                probability_recompute_dtype='torch.float64',probability_rtol=1e-12,probability_atol=1e-15)


def check_capture(cap,artifact,condition,query,stream,*,probe_count,origin_source):
    import torch
    identity=artifact['origin_identity'];n=artifact['n_local_rows'];gates=artifact['native_gates']
    applied=gates if condition=='native_gate' else torch.ones_like(gates)
    need(cap['mode']==condition and cap['n_local_rows']==n and cap['query_indices']==query and cap['stream_positions']==stream
        and cap['origin_identity']==identity and cap['origin_identity_sha256']==objsha(identity) and cap['origin_source']==origin_source
        and cap['probe_norm_calls']==cap['probe_head_calls']==cap['probability_calls']==probe_count,'Captured causal query/origin ledger differs')
    q=len(query);shapes={'local_states':(n,q,3584),'global_states':(q,3584),'native_global':(q,3584),
        'payload':(n,q,96),'messages':(n,q,96),'query':(q,96),'aggregate':(q,96),'preactivation':(q,96),'delta':(q,3584),'fused_global':(q,3584)}
    for key,shape in shapes.items():
        x=cap[key];dtype=torch.float16 if key in ('local_states','global_states','native_global','fused_global') else torch.float32
        need(x.shape==shape and x.dtype==dtype and not x.requires_grad and bool(torch.isfinite(x).all()),'Malformed query capture: '+key)
    expanded=applied[:,None].expand(n,q)
    need(torch.equal(cap['native_gates'],gates) and torch.equal(cap['applied_gates'],expanded) and torch.equal(cap['gates'],expanded),
         'Captured payload used a later or differently normalized gate')
    need(bool((cap['payload'].abs()<=1).all()) and torch.equal(cap['messages'],cap['payload']*expanded[:,:,None])
         and bool((cap['messages'][expanded==0]==0).all()),'Closed gate coordinate or bounded payload bypass')
    need(torch.equal(cap['global_states'],cap['native_global']) and torch.equal(cap['fused_global'],cap['global_states']+cap['delta'].half()),
         'Native global-only residual cast/add differs')
    aggregate_difference=float((cap['aggregate'].double()-cap['messages'].double().sum(0)).abs().max())
    return dict(passed=True,closed_message_coordinates_zero=True,original_gate_reused=True,sum_reduction_max_abs_difference=aggregate_difference)


def audit_eval(path,expected,condition,tokenizer,native_identity_sha256):
    import torch
    from gnnformer.parallel_local_prompts import build_set_count_prompt
    from gnnformer.data import build_count_prompt
    value=read(path);bind(value['raw_file'],value['raw_sha256'])
    archive=torch.load(value['raw_file'],map_location='cpu',weights_only=True);vectors=archive['raw_logits'];rows=value['rows']
    need(value['raw_dtype']=='torch.float32' and archive['schema_version']==1 and len(rows)==len(vectors)==len(expected)==value['n']
         and value['all_origin_artifacts_and_prefix_captures_retained'] is True,'Complete native trajectory coverage required')
    audits=[]
    for index,((cell,sample),row,x) in enumerate(zip(expected,rows,vectors)):
        need(row['raw_index']==index and row['cell']==cell and all(row[k]==sample[k] for k in ('sid','gold','n_frames','content_sha256'))
             and row['anchor_id']==sample.get('anchor_id') and row['pair_id']==sample.get('pair_id'),'Evaluation sample/order differs')
        ids=row['generated_ids'];steps=len(ids);n=sample['n_frames'];r=n+1
        need(x.shape==(steps,152064) and prior.stored_native_logits(x) and x.argmax(-1).tolist()==ids,'Generation is not exact native FP16 raw argmax')
        scored=score_sequence(tokenizer,ids,sample['gold'])
        need(all(row[k]==v for k,v in scored.items()) and row['raw_text']==tokenizer.decode(ids,skip_special_tokens=False)
             and row['text']==tokenizer.decode(ids,skip_special_tokens=True),'Native whole-answer parser/scoring differs')
        first=tokenizer(str(sample['gold']),add_special_tokens=False)['input_ids'][0]
        nll=float(torch.logsumexp(x[0].float(),-1)-x[0,first].float());need(close(nll,row['first_token_nll']),'First-token native NLL differs')
        row['first_token_correct']=ids[0]==first;meta=row['metadata'];width=meta['prompt_width']
        need(meta['arm']=='parallel' and meta['sid']==sample['sid'] and meta['n_frames']==n and meta['question']==sample['question']
            and meta['question_sha256']==objsha(sample['question']) and meta['global_prompt']==build_set_count_prompt(sample['question'])
            and meta['local_prompt']==build_count_prompt(sample['question'],1),'Original native question/prompt differs')
        need(meta['image_sha256']==[im['sha256'] for im in sample['image_files']] and meta['image_paths']==[im['path'] for im in sample['image_files']]
             and meta['prefix_ids']==[] and meta['resize']==392 and meta['row_count']==r and meta['global_row']==n
             and meta['local_elements']==n and meta['row_kinds']==['local']*n+['global'],'Actual local/global visual layout differs')
        need(width==meta['original_prompt_width']==max(meta['row_prompt_tokens']) and len(meta['row_prompt_tokens'])==r
            and meta['input_identity']['input_ids']['shape']==[r,width] and meta['input_identity']['attention_mask']['shape']==[r,width]
            and meta['input_identity']['image_grid_thw']['shape']==[n,3] and meta['layout']['every_unpadded_row_exact'] is True,
            'Packed native original input layout differs')
        policy=meta['generation']
        need(policy['max_new_tokens']==4 and policy['do_sample'] is False and policy['num_beams']==1 and policy['repetition_penalty']==1
             and policy['native_eos_token_ids']==[151645,151643] and policy['target_eos_token_id']==151645
             and policy['vocabulary_mask'] is False and policy['other_logits_processors'] is False,'Native greedy/EOS policy differs')
        need(meta['native_dtype']=='torch.float16' and meta['branch_dtype']=='torch.float32' and meta['gate_mode']==condition
             and meta['native_identity_sha256']==native_identity_sha256,'Native source/dtype/mode differs')
        positions=meta['generation_position_ids']
        need(len(positions)==steps and all(p['shape']==[4,r,width if t==0 else 1] for t,p in enumerate(positions)),'Native cached positional history differs')
        identity=original_identity(meta,native_identity_sha256);need(meta['origin_identity']==identity,'Original input/native identity digest differs')
        bind(row['raw_file'],row['raw_sha256']);result=torch.load(row['raw_file'],map_location='cpu',weights_only=True)
        need(result['schema_version']==1 and result['metadata']==meta and result['generated_ids']==ids and torch.equal(result['raw_logits'],x)
             and all(result[k]==row[k] for k in ('counters','raw_text','text','completed','truncated','model_seconds')),'Per-scene raw trajectory/summary differs')
        artifact=result['origin_artifact'];origin=check_origin(artifact,identity,condition)
        need(row['origin_artifact_sha256']==origin['artifact_sha256'] and row['capture_count']==len(result['captures'])==steps,'Raw original gate/capture binding differs')
        capture_audits=[]
        for t,cap in enumerate(result['captures']):
            capture_audits.append(check_capture(cap,artifact,condition,[width-1 if t==0 else 0],[width+t-1],probe_count=1,origin_source='native_prefill'))
            if t==0:need(torch.equal(cap['local_states'],artifact['origin_hidden'][:-1])
                         and torch.equal(cap['global_states'],artifact['origin_hidden'][-1]),'Origin probe and first payload read different native queries')
        counters=dict(model=steps,visual=1,language=steps,norm=steps,head=steps,fusion=steps,broadcast=steps,probe_norm=1,probe_head=1,probability=1)
        fusion=dict(passed=True,calls=steps,origin_source='native_prefill',probe_norm_calls=1,probe_head_calls=1,probability_calls=1)
        need(row['counters']==counters and result['fusion_audit']==row['fusion_audit']==fusion,'Actual native and extra probe counts differ')
        need(row['gate_evidence']==dict(passed=True,origin_artifact_sha256=origin['artifact_sha256'],origin_identity=identity,
             capture_count=steps,original_gate_reuse=True,closed_message_coordinates_zero=True),'Stored gate audit differs')
        audits.append(dict(sid=row['sid'],origin=origin,captures=steps,sum_reduction_max_abs_difference=max(a['sum_reduction_max_abs_difference'] for a in capture_audits)))
        del result,artifact
    need(value['exact_count']==sum(r['exact'] for r in rows) and close(value['first_token_nll'],sum(r['first_token_nll'] for r in rows)/len(rows)),
         'Evaluation summary denominator/scoring differs')
    return dict(value,independent_gate_audit=dict(trajectories=len(rows),origin_rows=sum(a['origin']['rows'] for a in audits),
        captures=sum(a['captures'] for a in audits),maximum_probability_difference=max(a['origin']['probability_max_abs_difference'] for a in audits),
        probability_rtol=1e-12,probability_atol=1e-15,maximum_descriptive_sum_reduction_difference=max(a['sum_reduction_max_abs_difference'] for a in audits),
        all_original_gates_reused=True,all_closed_payload_coordinates_zero=True,all_raw_whole_answers_retained=True))


def expected_gate_tensor(cache,gates,sids,condition):
    import torch
    widths=[cache['scenes'][sid]['n_frames'] for sid in sids];width=max(widths);parts=[];origins=[]
    for sid,n in zip(sids,widths):
        scene=cache['scenes'][sid];ids=[row[0] for row in scene['local_feature_ids']];origins.append(ids)
        need(ids==gates['scenes'][sid]['local_empty_feature_ids'],'Training scene gate origin order differs')
        values=torch.tensor([gates['gates'][fid]['gate'] for fid in ids],dtype=torch.float32)
        if condition=='all_open':values=torch.ones_like(values)
        part=torch.zeros(width,len(scene['target_ids']),dtype=torch.float32);part[:n]=values[:,None]
        parts.append(part)
    return torch.cat(parts,1),origins


def training_records(directory,config,summary,cache,gates,pairing,*,profile):
    import torch
    from gnnformer.parallel_local_semantic_aggregation import ParallelLocalSemanticAggregation
    steps=32 if profile else 4590;seed=config['seed'];condition=config['condition'];pairs=pairing['pairs']
    order=expected_order(pairs,seed);bind(directory/'presentations.json',config['presentations_sha256'])
    need(read(directory/'presentations.json')==order and len(order)==73440 and objsha(order)==config['order_sha256'],
         'Canonical paired presentation sequence differs')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);initial=ParallelLocalSemanticAggregation()
        expected_initial={k:tensor_info(v) for k,v in initial.state_dict().items()}
    need(sum(p.numel() for p in initial.parameters())==1041600 and config['initialized']==expected_initial
         and config['initialized_sha256']==objsha(expected_initial),'Exact semantic core initialization differs')
    bind(summary['training_file'],summary['training_sha256']);need(Path(summary['training_file'])==directory/'training.json','Training log path differs')
    logs=read(summary['training_file']);need(len(logs)==steps,'Optimizer update log coverage differs');total_targets=0;identity_count=0
    for step,row in enumerate(logs,1):
        positions=order[(step-1)*16:step*16];sids=[r['sid'] for r in positions];targets=[cache['scenes'][sid]['target_ids'] for sid in sids]
        identical=[sids[i]==sids[i+1] for i in range(0,16,2)]
        need(row['step']==step and row['sids']==sids and row['pair_ids']==[r['pair_id'] for r in positions[::2]]
             and row['epochs']==[r['epoch'] for r in positions[::2]] and close(row['lr'],learning_rate(step),atol=1e-12),'Actual optimizer pair/order/schedule differs')
        v10.check_loss_row(row,'consistency',targets,identical)
        need(row['path_loss_coefficient']==0. and row['consistency_coefficient']==1.,'Path diagnostic was optimized or residual coefficient changed')
        packed,origins=expected_gate_tensor(cache,gates,sids,condition)
        need(row['local_empty_feature_ids']==origins and row['gate_tensor']==tensor_info(packed)
             and row['closed_gate_count_by_position']==(packed==0).sum(0).tolist()
             and len(row['applied_gate_sum_by_position'])==packed.shape[1]
             and all(close(a,b) for a,b in zip(row['applied_gate_sum_by_position'],packed.sum(0).tolist()))
             and row['closed_message_coordinates_exact_zero'] is True and math.isfinite(row['payload_max_abs'])
             and 0<=row['payload_max_abs']<=1,'Training gates do not preserve original identity, prefix repeat, padding or closed coordinates')
        total_targets+=sum(map(len,targets));identity_count+=sum(identical)
    if not profile:need(total_targets==177120 and identity_count==2160,'Full training target/identity-pair totals differ')
    need(logs[0]['consistency_loss']==logs[0]['path_loss']==logs[0]['weighted_consistency_loss']==0.
         and close(summary['training_seconds'],sum(r['seconds'] for r in logs),atol=1e-4)
         and summary['step_seconds_max_steady']==max(r['seconds'] for r in logs[4:]),'Zero-U regularizer or timing reduction differs')
    bind(summary['first_gradients_file'],summary['first_gradients_sha256']);gradients=read(summary['first_gradients_file'])
    need([r['step'] for r in gradients]==[1,2],'First two gradient records required')
    for row in gradients:
        need(set(row['pre_clip'])==set(expected_initial),'Gradient parameter coverage differs')
        for name,info in row['pre_clip'].items():
            need(info['shape']==expected_initial[name]['shape'] and info['dtype']=='torch.float32' and len(info['sha256'])==64,
                 'Gradient shape/dtype/source identity differs')
    bind(summary['gradient_isolation_file'],summary['gradient_isolation_sha256']);isolated=read(summary['gradient_isolation_file'])
    need(isolated==[dict(step=s,passed=True,gate_gradient_is_none=True,frozen_features=True) for s in (1,2,32)],'Detached gate gradient audit incomplete')
    need(summary['zero_initialization_core_gradients']==dict(passed=True,up_exactly_zero=True,residual_loss=0.,all_residual_gradients_zero=True)
         and set(summary['all_parameter_gradients_exercised'])==set(expected_initial)
         and all(v is True for v in summary['all_parameter_gradients_exercised'].values()),'Zero-U or parameter gradient coverage differs')
    return dict(verified_optimizer_log_rows=steps,verified_scene_presentations=steps*16,verified_target_positions=total_targets,
        weighted_identity_pairs=identity_count,original_gate_order_and_padding_verified=True,gradient_isolation_verified=True,
        first_preclip_gradient_hashes=gradients[0]['pre_clip']),logs


def verify_config(directory,cache,gates,*,profile):
    from scripts import train_native_vision_v18 as training
    directory=Path(directory).resolve();config=read(directory/'config.json');summary=read(directory/'summary.json')
    condition,seed=config['condition'],config['seed'];prefix='profile' if profile else 'run';steps=32 if profile else 4590
    need(directory.parent==OUT and directory.name==config['run_id'] and directory.name.startswith(f'{prefix}_{condition}_s{seed}_')
         and condition in CONDITIONS and seed in SEEDS and (not profile or seed==20) and config['profile'] is profile
         and config['mode']==condition and config['arm']=='parallel','Unknown V18 fit/profile/seed')
    need(config['policy']==POLICY==training.POLICY and config['source_sha256']==training.sources()
         and all(summary.get(k)==v for k,v in config.items()) and summary['steps']==steps
         and all(summary.get(k) is True for k in ('passed','completed','computational_integrity_passed','checkpoint_roundtrip_passed',
             'frozen_backbone_gradient_state_preserved','final_endpoint_unchanged')),'Unfinished run or changed policy/source')
    prior.verify_source(directory,config['source_sha256']);bind(config['plan_file'],config['plan_sha256']);plan=read(config['plan_file'])
    need(Path(config['plan_file']).with_suffix('.sha256').read_text().strip()==config['plan_sha256']
         and plan['source_sha256']==config['source_sha256'] and plan['policy']==POLICY,'Training plan/source changed')
    cpu=read(Path(config['plan_file']).parent/'summary.json')
    need(cpu['passed'] is True and cpu['completed'] is True and cpu['plan_sha256']==config['plan_sha256'],'Completed exact CPU train gate required')
    ledger(plan['artifact_bindings'])
    for key in ('cache_binding','gate_cache_binding','native_identity','native_identity_sha256','native_profile','pairing_file','pairing_sha256','prior_result'):
        need(config[key]==plan[key],'Config/plan binding differs: '+key)
    need(config['cache_binding']==dict(file=str(CACHE),sha256=sha(CACHE))
         and config['gate_cache_binding']==dict(file=str(GATE_CACHE),sha256=sha(GATE_CACHE))
         and config['native_identity']==gates['native_identity'] and config['native_identity_sha256']==objsha(gates['native_identity'])
         and config['native_weight_identity']==dict(norm=gates['native_identity']['norm_weight'],head=gates['native_identity']['head_weight'])
         and all(config[k]==cache[k] for k in ('model','runtime','processor','native_dtypes')),'Actual deployed native identity/cache differs')
    need(plan['fresh_manifests']=={'main':str(DATA/'v18_fresh/main_manifest.json')} and config['hardware']['name']=='NVIDIA B200',
         'Fresh corpus or measured native hardware changed')
    bind(plan['pairing_file'],plan['pairing_sha256']);pairing=read(plan['pairing_file'])
    need(objsha(pairing)==plan['pairing_object_sha256'] and config['order_sha256']==plan['order_sha256'][str(seed)],'Paired order identity differs')
    return directory,config,summary,plan,pairing


def endpoint(directory,config,summary,*,profile):
    import torch
    step=32 if profile else 4590;file=CKPT/config['run_id']/('profile.pt' if profile else 'final.pt')
    need(Path(config['checkpoint_directory'])==file.parent,'Checkpoint directory differs')
    bind(summary['final_endpoint_file'],summary['final_endpoint_sha256']);proof=read(summary['final_endpoint_file'])
    need(Path(summary['final_endpoint_file'])==directory/'final_endpoint.json' and proof['schema_version']==1
         and proof['step']==step and Path(proof['checkpoint'])==file,'Final endpoint path/step differs')
    bind(file,proof['checkpoint_sha256']);checkpoint=torch.load(file,map_location='cpu',weights_only=True)
    need(set(checkpoint)=={'branch','step','config'} and checkpoint['step']==step and checkpoint['config']==config,'Final checkpoint payload differs')
    table={k:tensor_info(v) for k,v in checkpoint['branch'].items()}
    need(sum(v.numel() for v in checkpoint['branch'].values())==1041600 and set(table)==set(config['initialized'])
         and all(table[k]['shape']==config['initialized'][k]['shape'] for k in table)
         and all(v.dtype==torch.float32 and bool(torch.isfinite(v).all()) for v in checkpoint['branch'].values())
         and proof['before_evaluation']==proof['after_evaluation']==table and proof['parameter_sha256']==objsha(table)
         and all(proof.get(k) is True for k in ('checkpoint_matches_deployed_endpoint','core_versions_unchanged','native_versions_unchanged')),
         'Actual trained/evaluated endpoint does not match final CPU checkpoint')
    return dict(checkpoint=str(file),checkpoint_sha256=proof['checkpoint_sha256'],parameter_sha256=objsha(table),
                parameter_table=table,actual_before_after_evaluation_bound=True,endpoint_file=summary['final_endpoint_file'],endpoint_sha256=summary['final_endpoint_sha256'])


def verify_training_profile(directory,plan_file,plan,condition,tokenizer,cache,gate_cache):
    import torch
    from scripts.cache_native_vision_v7_features import replay_metrics
    from scripts import train_native_vision_v18 as training
    directory,config,summary,actual_plan,pairing=verify_config(directory,cache,gate_cache,profile=True)
    need(actual_plan==plan and config['plan_file']==str(Path(plan_file).resolve()) and condition==config['condition'],
         'Measured profile belongs to another CPU plan/condition')
    manifest=read(plan['train_manifest']);small=dict(train={r['sid']:r for c in ('train_N8','train_N16') for r in manifest['splits'][c]['samples']},
                                                  schedule=read(plan['schedule_file']))
    need(pairing==get_pairing(small,cache),'Profile pair metadata differs from independent canonical reconstruction')
    audited,logs=training_records(directory,config,summary,cache,gate_cache,pairing,profile=True)
    final=endpoint(directory,config,summary,profile=True)
    need(summary['profile_calls']==training.PROFILE_CALLS and summary['no_dev_or_test_evaluation'] is True
         and summary['profile_checkpoint']==final['checkpoint'] and summary['profile_checkpoint_sha256']==final['checkpoint_sha256']
         and not (directory/'dev_final.json').exists() and not (directory/'test.json').exists(),'Profile included outcomes or wrong checkpoint')
    replay=summary['active_cache_replay'];need(read(directory/'active_cache_replay.json')==replay,'Profile replay summary copy differs')
    bind(replay['raw_file'],replay['raw_sha256']);raw=torch.load(replay['raw_file'],map_location='cpu',weights_only=True)
    need(raw['schema_version']==1 and len(raw['observations'])==len(replay['observations'])==10
         and replay['passed'] is True and replay['counts_covered']==[0,9,10,16] and replay['same_state_rows']==170
         and replay['prefixes_checked']==10 and replay['native_model_calls']==replay['vision_calls']==10
         and replay['original_probe_norm_calls']==replay['original_probe_head_calls']==replay['original_probability_calls']==4
         and replay['standalone_head_calls']==20 and replay['cached_differences_descriptive'] is True
         and replay['gate_values_never_replaced'] is True,'Training profile native probe/head coverage differs')
    expected=[]
    for k in (0,9,10,16):
        sample=min((r for r in manifest['splits']['train_N16']['samples'] if r['gold']==k),key=lambda r:r['sid'])
        expected.extend((sample,t) for t in range(len(cache['scenes'][sample['sid']]['target_ids'])))
    origins={};failures=[];probability_differences=[]
    for (sample,t),record,packet in zip(expected,replay['observations'],raw['observations']):
        sid=sample['sid'];targets=cache['scenes'][sid]['target_ids'];prefix=targets[:t];actual=packet['actual'];meta=actual['metadata']
        need(record['sid']==sid and record['gold']==sample['gold'] and record['position']==t and record['prefix_ids']==prefix
             and record['n_frames']==16 and meta==record['native_metadata'] and meta['prefix_ids']==prefix
             and meta['sid']==sid and meta['gate_mode']==condition and meta['native_identity_sha256']==config['native_identity_sha256'],
             'Training profile strict-prefix/native metadata differs')
        artifact=actual['origin_artifact'];identity=meta['origin_identity']
        if t==0:
            need(identity==original_identity(meta,config['native_identity_sha256']),'Profile original input identity differs')
            origins[sid]=artifact['artifact_sha256']
            check=check_origin(artifact,identity,condition);probability_differences.append(check['probability_max_abs_difference'])
        need(origins[sid]==artifact['artifact_sha256']==record['origin_artifact_sha256'] and record['original_gate_reused'] is True,
             'Profile recomputed a gate at a later target prefix')
        width=meta['prompt_width'];check_capture(actual['fusion'],artifact,condition,[width-1],[width-1],
            probe_count=int(t==0),origin_source='native_prefill' if t==0 else 'bound_artifact')
        need(torch.equal(actual['local_states'],actual['fusion']['local_states']) and torch.equal(actual['global_states'],actual['fusion']['global_states']),
             'Full-prefix profile replay uses different captured native states')
        counts=dict(model=1,visual=1,language=1,norm=1,head=1,fusion=1,probe_norm=int(t==0),probe_head=int(t==0),probability=int(t==0))
        need(actual['counters']==record['counters']==counts,'Profile direct probe versus normal-head counters differ')
        packed,ids=expected_gate_tensor(cache,gate_cache,[sid]*2,condition)
        need(record['cached_empty_feature_ids']==ids[0] and torch.equal(packet['cached_gates'],packed[:,t:t+1]),'Profile cached origin/prefix gates differ')
        delta=max(abs(gate_cache['gates'][fid]['gate']-float(artifact['native_gates'][i])) for i,fid in enumerate(ids[0]))
        need(close(delta,record['cached_vs_live_gate_max_abs']),'Cached/live gate drift statistic differs')
        native_logits=actual['native_logits'];replayed=packet['replayed'];cached=packet['cached']
        need(native_logits.shape==replayed.shape==(17,1,152064) and cached.shape==(1,152064)
             and all(v.dtype==torch.float16 and bool(torch.isfinite(v).all()) for v in (native_logits,replayed,cached))
             and torch.equal(actual['global_logits'],native_logits[-1,-1]),'Profile native fullbatch/raw shape differs')
        same=replay_metrics(torch,native_logits[:,0],replayed[:,0]);drift=replay_metrics(torch,actual['global_logits'][None],cached)
        v14.numeric_metric(record['native_head_replay'],same);v14.numeric_metric(record['cached_vs_native_descriptive'],drift)
        need(same['passed'] and record['cached_vs_native_gate'] is False,'Captured native head binding failed or descriptive drift promoted to gate')
        if not drift['passed']:failures.append(dict(sid=sid,position=t,metric=drift))
    need(replay['cached_numerical_failures']==len(failures),'Profile descriptive numerical failures omitted')
    return dict(passed=True,directory=str(directory),summary_sha256=sha(directory/'summary.json'),condition=condition,
        native_identity_sha256=config['native_identity_sha256'],training=audited,endpoint=final,native_replays=10,native_rows=170,
        original_probes=4,standalone_heads=20,maximum_probability_difference=max(probability_differences),
        cached_failures=failures,cached_failures_descriptive=True,first_four_seconds=sum(r['seconds'] for r in logs[:4]),
        maximum_steady_step_seconds=max(r['seconds'] for r in logs[4:]),setup_seconds=summary['setup_before_training_seconds'])


def verify_run(directory,data,tokenizer,cache,gates,pairing_value):
    directory,config,summary,plan,pairing=verify_config(directory,cache,gates,profile=False)
    need(pairing==pairing_value,'Run pair selection differs from independent metadata reconstruction')
    audited,logs=training_records(directory,config,summary,cache,gates,pairing,profile=False);final=endpoint(directory,config,summary,profile=False)
    selection=read(directory/'selection.json');entry=selection['selected']
    need(selection['rule']==POLICY['selection'] and selection['dev_descriptive_only'] is True and selection['development']==[entry]
         and summary['development']==[entry] and summary['selected']==entry and entry['step']==4590
         and entry['checkpoint']==final['checkpoint'] and entry['checkpoint_sha256']==final['checkpoint_sha256']
         and entry['parameter_sha256']==final['parameter_sha256'],'Only the final4590 endpoint is eligible')
    need(Path(entry['dev_file'])==directory/'dev_final.json' and summary['native_dev_count']==64 and summary['native_test_count']==272
         and summary['dev_descriptive_only'] is True,'One descriptive dev64 and complete fresh272 required')
    bind(entry['dev_file'],entry['dev_sha256']);dev=audit_eval(entry['dev_file'],data['dev'],config['condition'],tokenizer,config['native_identity_sha256'])
    need(dev['label']=='dev_final' and dev['exact_count']==entry['exact_count'] and close(dev['first_token_nll'],entry['nll']),
         'Descriptive final dev rescore differs')
    bind(summary['test_file'],summary['test_sha256']);need(Path(summary['test_file'])==directory/'test.json','Wrong final test artifact')
    test=audit_eval(summary['test_file'],data['test'],config['condition'],tokenizer,config['native_identity_sha256'])
    need(test['label']=='test','Wrong final test label')
    result=dict(audited,run_id=config['run_id'],condition=config['condition'],seed=config['seed'],directory=str(directory),
        summary_sha256=sha(directory/'summary.json'),selected_step=4590,one_dev_only=True,dev_descriptive_only=True,
        endpoint=final,selected=entry,test_file=summary['test_file'],test_sha256=summary['test_sha256'],
        dev_gate_audit=dev['independent_gate_audit'],test_gate_audit=test['independent_gate_audit'],
        loss=v10.loss_summary(logs),native_test_examples=272,reasoning_objective_achieved=False)
    return result,test['rows'],config


ACCOUNT_FIELDS=('JobIDRaw','JobID','JobName','Partition','State','ExitCode','ElapsedRaw','AllocTRES','Start','End')
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}


def accounting_rows(raw):
    from datetime import datetime
    rows=[];seen=set()
    for line in raw.splitlines():
        if not line.strip():continue
        fields=line.split('|');need(len(fields)==10,'Unexpected full Slurm ledger schema')
        row=dict(zip(ACCOUNT_FIELDS,fields))
        if row['Partition']!='gpu' or re.match(r'^(?:native_)?v18_',row['JobName']) is None:continue
        need(re.fullmatch('[0-9]+',row['JobIDRaw']) and re.fullmatch('[0-9]+(?:_[0-9]+)?',row['JobID'])
             and row['JobIDRaw'] not in seen,'Nonallocation or duplicate campaign job')
        state=row['State'].split()[0].rstrip('+');need(state in TERMINAL and row['ElapsedRaw'].isdigit(),'Unfinished or invalid campaign allocation')
        values=dict(part.split('=',1) for part in row['AllocTRES'].split(',') if '=' in part)
        generic=values.get('gres/gpu');typed=[v for k,v in values.items() if k.startswith('gres/gpu:')]
        need((generic is None or generic.isdigit()) and all(v.isdigit() for v in typed),'Invalid generic/typed GPU resource count')
        count=int(generic) if generic is not None else sum(map(int,typed));need(count in (0,1),'Unexpected GPU count per allocation')
        if count:need(datetime.fromisoformat(row['End'])>=datetime.fromisoformat(row['Start']),'Invalid allocated interval')
        rows.append(dict(row,state=state,gpu_count=count,allocated_gpu_seconds=count*int(row['ElapsedRaw'])));seen.add(row['JobIDRaw'])
    return sorted(rows,key=lambda r:int(r['JobIDRaw']))


def concurrency(rows):
    events=[]
    for row in rows:
        if row['gpu_count'] and row['Start']!=row['End']:events.extend([(row['Start'],row['gpu_count']),(row['End'],-row['gpu_count'])])
    value=maximum=0
    for _,change in sorted(events):value+=change;need(value>=0,'Malformed accounting interval balance');maximum=max(maximum,value)
    need(value==0,'Unclosed GPU allocation interval');return maximum


def projected_seconds(setup,first_four,steady,t16,t64):
    need(all(math.isfinite(x) and x>=0 for x in (setup,first_four,steady,t16,t64)) and min(steady,t16,t64)>0,'Invalid measured projection input')
    return setup+1.25*(first_four+4586*steady+64*t16+272*t64)+120


def verify_release(config,data,frozen):
    item=config['main_release'];need(isinstance(item,dict),'Main has no independent release')
    bind(item['file'],item['sha256']);release=read(item['file'])
    need(release['protocol']=='v18_native_semantic_gate_main_release' and release['passed'] is True
         and release['plan_file']==config['plan_file'] and release['plan_sha256']==config['plan_sha256']
         and release['source_sha256']==config['source_sha256'] and release['report_source_sha256']==frozen
         and release['native_identity_sha256']==config['native_identity_sha256'] and release['per_main_seconds_cap']==2700
         and release['campaign_gpu_seconds_cap']==14400 and release['main_jobs']==4 and release['maximum_concurrent_project_gpus']==4,
         'Independent main release source/policy differs')
    ledger(release['release_source_sha256']);approved=release['data_release'];bind(approved['file'],approved['sha256']);audit=read(approved['file'])
    need(audit['passed'] is True and audit['manifest_only'] is False and audit['source_sha256']==frozen
         and audit['native_identity_sha256']==config['native_identity_sha256'],'Incomplete independent data/cache/gate release')
    bind(audit['audit_file'],audit['audit_sha256']);ledger(audit['data_bindings'])
    for path,digest in data['summary']['data_bindings'].items():need(audit['data_bindings'].get(path)==digest,'Fresh/train/data release binding differs')
    fresh=str(DATA/'v18_fresh/main_manifest.json')
    need(item['fresh_manifest']==release['fresh_manifest']==dict(file=fresh,sha256=sha(fresh)),'Main did not use released fresh test')
    budget=release['campaign_budget'];bind(budget['all_user_sacct_file'],budget['all_user_sacct_sha256'])
    rows=accounting_rows(Path(budget['all_user_sacct_file']).read_text());total=sum(r['allocated_gpu_seconds'] for r in rows)
    need(rows==budget['allocations'] and budget['prior_jobs']==[r['JobIDRaw'] for r in rows]
         and budget['prior_allocated_gpu_seconds']==total and budget['reserved_main_gpu_seconds']==10800
         and budget['projected_campaign_gpu_seconds']==total+10800<=14400 and budget['campaign_gpu_seconds_cap']==14400
         and budget['maximum_concurrent_gpus']==concurrency(rows)<=4 and budget['passed'] is True
         and budget['all_failed_and_zero_allocations_retained'] is True,'Pre-main allocation ledger/reserve arithmetic differs')
    software_summary=read(config['native_profile']['file']);bind(config['native_profile']['file'],config['native_profile']['sha256'])
    times=software_summary['timing']['rows'];t16=max(r['four_token_seconds_bound'] for r in times if r['n_frames']==16)
    t64=max(r['four_token_seconds_bound'] for r in times if r['n_frames']==64)
    success=[str(software_summary['slurm_job_id'])]
    need(set(release['profiles'])==set(release['projections'])==set(CONDITIONS),'Both measured mode profiles required')
    for condition,entry in release['profiles'].items():
        directory=Path(entry['directory']);bind(directory/'summary.json',entry['summary_sha256']);profile=read(directory/'summary.json')
        bind(profile['training_file'],profile['training_sha256']);logs=read(profile['training_file'])
        need(profile['passed'] is True and profile['profile'] is True and profile['condition']==condition and profile['seed']==20
             and len(logs)==32 and profile['plan_sha256']==config['plan_sha256'],'Release profile identity differs')
        expected=projected_seconds(profile['setup_before_training_seconds'],sum(r['seconds'] for r in logs[:4]),max(r['seconds'] for r in logs[4:]),t16,t64)
        observed=release['projections'][condition]
        need(observed['passed'] is True and observed['projected_seconds']==expected<=2700
             and observed['T16']==t16 and observed['T64']==t64 and observed['dev_examples']==64 and observed['test_examples']==272,
             'Measured mode projection arithmetic differs')
        success.append(str(profile['slurm_job_id']))
    indexed={r['JobIDRaw']:r for r in rows}
    for job in success:
        need(job in indexed and indexed[job]['state']=='COMPLETED' and indexed[job]['ExitCode']=='0:0'
             and indexed[job]['gpu_count']==1 and int(indexed[job]['ElapsedRaw'])<=300,'Successful profile GPU allocation missing or invalid')
    return dict(file=item['file'],sha256=item['sha256'],prior_allocated_gpu_seconds=total,projections=release['projections'],
                native_identity_sha256=config['native_identity_sha256'],data_release=approved)


def final_accounting(out,configs):
    import subprocess
    command=['sacct','-X','-u',os.environ['USER'],'--starttime=2026-09-01','--parsable2','--noheader',
             '--format=JobIDRaw,JobID,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,Start,End']
    result=subprocess.run(command,capture_output=True,text=True,check=True);path=out/'all_user_sacct.psv';path.write_text(result.stdout)
    rows=accounting_rows(result.stdout);lookup={r['JobIDRaw']:r for r in rows};main_ids={str(c['slurm_job_id']) for c in configs.values()}
    need(len(main_ids)==4,'Four distinct actual fit allocations required')
    for job in main_ids:
        need(job in lookup and lookup[job]['state']=='COMPLETED' and lookup[job]['ExitCode']=='0:0'
             and lookup[job]['gpu_count']==1 and int(lookup[job]['ElapsedRaw'])<=2700,'Main allocation did not complete within registered cap')
    for row in rows:
        if row['JobIDRaw'] not in main_ids:need(int(row['ElapsedRaw'])<=300 or row['gpu_count']==0,'Non-main V18 GPU allocation exceeded profile cap')
    software_seconds=sum(r['allocated_gpu_seconds'] for r in rows if r['JobName'].startswith('v18_semantic_profile'))
    total=sum(r['allocated_gpu_seconds'] for r in rows);maximum=concurrency(rows)
    need(total<=14400 and maximum<=4 and software_seconds<=900,'Complete V18 resource cap exceeded')
    return dict(passed=True,allocated_gpu_seconds=total,campaign_gpu_seconds_cap=14400,maximum_concurrent_gpus=maximum,
        software_allocated_gpu_seconds=software_seconds,software_seconds_cap=900,allocations=rows,main_job_ids=sorted(main_ids),
        all_failed_and_zero_allocations_retained=True,all_user_sacct_file=str(path),all_user_sacct_sha256=sha(path),command=command)


def self_test():
    import torch
    tests=[]
    class Tokenizer:
        all_special_ids=[9,151645,151643]
        def decode(self,ids,skip_special_tokens=False):
            mapping={0:'0',1:'1',2:'2',3:' ',4:'١',5:'-',6:'.',7:'x',9:'',151645:'<eos>',151643:'<end>'}
            return ''.join(mapping[i] for i in ids if not skip_special_tokens or i not in self.all_special_ids)
    tokenizer=Tokenizer()
    need(score_sequence(tokenizer,[1,0,151645],10)['exact'],'Two-digit native EOS answer failed')
    need(not score_sequence(tokenizer,[1,151645],10)['exact'] and not score_sequence(tokenizer,[4,151645],1)['parseable']
        and not score_sequence(tokenizer,[9,1,151645],1)['parseable'] and not score_sequence(tokenizer,[1,0,0,0],1000)['exact'],
        'First digit, Unicode, hidden special or missing EOS was accepted')
    for bad in ([1,0],[1,151645,0,151645]):
        try:score_sequence(tokenizer,bad,10)
        except ValueError:pass
        else:raise ValueError('Invalid native termination passed')
    tests.append('whole_ASCII_integer_native_EOS_and_nonterminal_specials')
    def endpoint_counts(n32,n64,large):
        return dict(cells={'main_test_N32':dict(correct=n32),'main_test_N64':dict(correct=n64)},partitions={'N64':{'K9_16':dict(correct=large)}})
    control=endpoint_counts(129,102,0);treated=endpoint_counts(123,109,52)
    need(integer_decision(treated,control)['practical'] and not integer_decision(endpoint_counts(123,108,52),control)['primary']
         and not integer_decision(endpoint_counts(122,109,52),control)['primary']
         and not integer_decision(endpoint_counts(123,109,51),control)['practical'],'Exact integer gates differ')
    tests.append('per_seed_integer_threshold_boundaries')
    rows={}
    for mode in CONDITIONS:
        for seed in SEEDS:rows[mode,seed]=[dict(gold=k,anchor_id=f'k{k}r{i}',n_frames=n,exact=mode=='native_gate')
                                           for n in (32,64) for k in range(17) for i in range(8)]
    result=bootstrap(rows,replicates=32,seed=20261118)
    need(all(result['results'][part][n][str(seed)]['difference_ci95_pp']==[100.,100.]
             for part in ('all','K0_8','K9_15','K16','K9_16','nonzero') for n in ('N32','N64') for seed in SEEDS),
         'Family bootstrap mode/key/partition contract differs')
    damaged={k:list(v) for k,v in rows.items()};damaged['all_open',20].pop()
    try:bootstrap(damaged,replicates=2)
    except ValueError:pass
    else:raise ValueError('Missing family endpoint passed')
    tests.append('intact_family_bootstrap_partitions_and_missing_endpoint')
    def allocation(job,state,seconds,tres):
        return '|'.join((str(job),str(job),'v18_train_profile','gpu',state,'0:0' if state=='COMPLETED' else '1:0',str(seconds),tres,
                         '2026-09-11T10:00:00','2026-09-11T10:01:00'))
    raw='\n'.join((allocation(1,'COMPLETED',60,'gres/gpu=1,gres/gpu:b200=1'),allocation(2,'FAILED',7,'gres/gpu:b200=1'),allocation(3,'CANCELLED',0,'')))
    alloc=accounting_rows(raw);need(len(alloc)==3 and sum(r['allocated_gpu_seconds'] for r in alloc)==67 and concurrency(alloc)==2,
                                  'Typed/generic GPU counts or failed/zero retention differs')
    for invalid in (raw+'\n'+raw.splitlines()[0],allocation(9,'RUNNING',1,'gres/gpu=1')):
        try:accounting_rows(invalid)
        except ValueError:pass
        else:raise ValueError('Duplicate/unfinished resource accounting passed')
    need(projected_seconds(10,4,.1,1,2)==10+1.25*(4+4586*.1+64+544)+120,'Fixed timing projection differs')
    tests.append('complete_allocation_accounting_and_projection')
    from gnnformer.parallel_local_semantic_gate import GATE_RULE
    identity=dict(sid='fixture',question_sha256='a'*64,image_sha256=['b'*64,'c'*64],prompt_width=3,input_identity_sha256='d'*64,native_identity_sha256='e'*64)
    logits=torch.zeros(3,1,152064,dtype=torch.float16);logits[0,0,15]=1;logits[0,0,16]=2;logits[1,0,15]=2;logits[1,0,16]=1
    x=logits[:2,0].double();z=torch.logsumexp(x,-1);p0=(x[:,15]-z).exp();p1=(x[:,16]-z).exp();ng=(p1-p0).clamp_min(0).float()
    artifact=dict(schema_version=1,gate_rule=GATE_RULE,mode='native_gate',origin_identity=identity,zero_token_id=15,one_token_id=16,n_local_rows=2,
        origin_query_index=2,origin_stream_position=2,probe_shape=[3,1,3584],origin_hidden=torch.zeros(3,1,3584,dtype=torch.float16),
        origin_normalized=torch.zeros(3,1,3584,dtype=torch.float16),origin_logits=logits,p0=p0,p1=p1,native_gates=ng)
    def seal(a):a['artifact_sha256']=objsha({k:tensor_info(v) if isinstance(v,torch.Tensor) else v for k,v in a.items() if k!='artifact_sha256'});return a
    seal(artifact);check_origin(artifact,identity,'native_gate')
    wrong=dict(artifact,p0=torch.tensor([.25,.75],dtype=torch.float64),p1=torch.tensor([.75,.25],dtype=torch.float64))
    wrong['native_gates']=(wrong['p1']-wrong['p0']).clamp_min(0).float();seal(wrong)
    try:check_origin(wrong,identity,'native_gate')
    except ValueError:pass
    else:raise ValueError('Binary-renormalized probabilities passed full-vocabulary check')
    tests.append('full_native_vocabulary_probability_not_binary_normalization')
    cap=dict(mode='native_gate',n_local_rows=2,query_indices=[0],stream_positions=[3],origin_identity=identity,origin_identity_sha256=objsha(identity),
        origin_source='native_prefill',probe_norm_calls=1,probe_head_calls=1,probability_calls=1,
        local_states=torch.zeros(2,1,3584,dtype=torch.float16),global_states=torch.zeros(1,3584,dtype=torch.float16),
        native_global=torch.zeros(1,3584,dtype=torch.float16),payload=torch.full((2,1,96),.25),messages=torch.full((2,1,96),.25)*ng[:,None,None],
        query=torch.zeros(1,96),aggregate=torch.zeros(1,96),preactivation=torch.zeros(1,96),delta=torch.zeros(1,3584),
        fused_global=torch.zeros(1,3584,dtype=torch.float16),native_gates=ng,applied_gates=ng[:,None],gates=ng[:,None])
    check_capture(cap,artifact,'native_gate',[0],[3],probe_count=1,origin_source='native_prefill')
    for key,value in [('stream_positions',[4]),('native_gates',torch.ones(2)),('messages',torch.ones(2,1,96))]:
        bad=dict(cap,**{key:value})
        try:check_capture(bad,artifact,'native_gate',[0],[3],probe_count=1,origin_source='native_prefill')
        except ValueError:pass
        else:raise ValueError('Changed causal origin or closed-coordinate bypass passed')
    tests.append('fixed_origin_causal_position_and_all_coordinate_gate_checks')
    return dict(passed=True,tests=tests,native_model_calls=0,native_head_calls=0)


def report_runs(directories,out,frozen):
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(4);data=audit_data();tokenizer=AutoTokenizer.from_pretrained(str(MODEL),use_fast=False,local_files_only=True)
    cache,cache_audit=verify_cache(CACHE,data,tokenizer);gates,gate_audit=verify_gates(cache);pairing=get_pairing(data,cache)
    save(out/'data_audit.json',dict(data['summary'],cache_audit=cache_audit,gate_cache_audit=gate_audit));save(out/'pairing.json',pairing)
    need(len(directories)==len({Path(p).resolve() for p in directories})==4,'All four distinct final models required')
    audits={};rows={};configs={};releases=[]
    for directory in directories:
        audited,predictions,config=verify_run(directory,data,tokenizer,cache,gates,pairing);key=(config['condition'],config['seed'])
        need(key not in audits,'Duplicate condition/seed');audits[key]=audited;rows[key]=predictions;configs[key]=config
        releases.append(verify_release(config,data,frozen));print(json.dumps(dict(audited_run=str(directory),examples=len(predictions))),flush=True)
    need(set(audits)=={(c,s) for c in CONDITIONS for s in SEEDS} and all(r==releases[0] for r in releases),'Missing mode/seed or different main releases')
    for seed in SEEDS:
        a,b=configs['native_gate',seed],configs['all_open',seed]
        for key in ('initialized','initialized_sha256','presentations_sha256','order_sha256','policy','plan_sha256','source_sha256',
                    'cache_binding','gate_cache_binding','native_identity','native_weight_identity','prior_result'):
            need(a[key]==b[key],'Matched mode source/init/data/order differs: '+key)
    accounting=final_accounting(out,configs)
    runs={f'{c}_s{s}':dict(audits[c,s],metrics=summarize(rows[c,s])) for c in CONDITIONS for s in SEEDS}
    decisions={str(s):integer_decision(runs[f'native_gate_s{s}']['metrics'],runs[f'all_open_s{s}']['metrics']) for s in SEEDS}
    primary=all(v['primary'] for v in decisions.values());practical=all(v['practical'] for v in decisions.values());intervals=bootstrap(rows)
    analysis=dict(schema_version=1,passed=True,audit_passed=True,source_sha256=frozen,policy=POLICY,data=data['summary'],cache=cache_audit,
        gate_cache=gate_audit,release=releases[0],accounting=accounting,runs=runs,decisions=decisions,bootstrap=intervals,
        primary_both_seeds=primary,practical_both_seeds=practical,vision_milestone_gate=primary and practical,reasoning_objective_achieved=False,
        pooled_fixed_two_seed_n64_difference_pp=sum(v['n64_difference_pp'] for v in decisions.values())/2,
        interpretation=dict(treatment='Fixed original full-vocabulary native semantic gate versus all-open, with identical tanh payload/SUM core and native probe work',
            original_gate_never_reclassified=True,fixed_final_step=4590,one_dev_sweep_descriptive_only=True,all_answer_values_seen_in_training=True,
            no_unseen_answer_value_claim=True,no_attention_operator_novelty_claim=True,no_reasoning_composition_evidence=True,
            no_external_count_tally=True,all_generated_failures_retained=True,local_visual_atoms_may_recur=True,
            parameter_proof='Actual parameter tables before and after native evaluation equal the archived final CPU checkpoint',
            first_token_metric='Descriptive only; shared first digits do not prove whole multi-digit answers',
            gate_probability_scope='FP64 full-vocabulary probability recomputation with registered roundoff tolerance; exact source/tensor identities',
            sum_reduction_scope='FP32 saved aggregate versus FP64 saved-message sum is descriptive; no CPU/GPU kernel-equivalence claim',
            loss_scope='All recorded causal-position/pair/scene arithmetic and exact applied gate tensors; intermediate parameter trajectories not reconstructed'))
    save(out/'analysis.json',analysis)
    lines=['# V18 independent native evaluation','',f'Both-seed primary: **{primary}**. Both-seed practical: **{practical}**. Vision milestone: **{primary and practical}**.','',
        '| Mode | Seed | N32 | N64 | N64 K0–8 | N64 K9–15 | N64 K16 | N64 K9–16 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for c in CONDITIONS:
        for s in SEEDS:
            m=runs[f'{c}_s{s}']['metrics'];cells=[m['cells'][f'main_test_N{n}'] for n in (32,64)]+[m['partitions']['N64'][k] for k in ('K0_8','K9_15','K16','K9_16')]
            lines.append(f'| {c} | {s} | '+' | '.join(f"{x['correct']}/{x['n']}" for x in cells)+' |')
    lines+=['','All272 fresh examples remain in each denominator. Exact requires the complete correct ASCII integer and native EOS within four tokens.',
        '', 'Both modes use the same original native full-vocabulary probe, parameter count, paired data/order and CE plus residual-consistency objective. Every final core is fixed at4590 updates; its one dev sweep is descriptive.',
        '', f"All V18 GPU allocations, including failures and zero allocations, total {accounting['allocated_gpu_seconds']} GPU-seconds. Peak concurrency is {accounting['maximum_concurrent_gpus']} GPUs.",
        '', 'These results do not establish composition with reasoning tokens. The gate remains fixed to the original question, and answer values0–16 are supported during training.',
        '', '[Full per-K results, native parsing, capture checks, family intervals and provenance](analysis.json).']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return dict(passed=True,completed=True,source_sha256=frozen,analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),
        report_file=str(out/'REPORT.md'),primary_both_seeds=primary,practical_both_seeds=practical,vision_milestone_gate=primary and practical,
        reasoning_objective_achieved=False,allocated_gpu_seconds=accounting['allocated_gpu_seconds'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-test',action='store_true');mode.add_argument('--check-data',action='store_true');mode.add_argument('--check-manifests',action='store_true')
    mode.add_argument('--runs','--runs4',nargs=4,type=Path);parser.add_argument('--output',type=Path);args=parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),
         'All reporter/source/data/numerical checks require CPU Slurm')
    started=time.perf_counter();label='selftest' if args.self_test else 'data_check' if args.check_data else 'manifest_check' if args.check_manifests else 'report'
    out=args.output or OUT/f'{label}_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False);frozen=snapshot(out)
    import torch
    torch.set_num_threads(4);tests=self_test()
    if args.self_test:summary=dict(passed=True,completed=True,source_sha256=frozen,tests=tests,no_model_loaded=True)
    elif args.check_data or args.check_manifests:
        data=audit_data();audited=data['summary'];cache_audit=gate_audit=None;identity=None;bindings=dict(audited['data_bindings'])
        if args.check_data:
            from transformers import AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained(str(MODEL),use_fast=False,local_files_only=True)
            cache,cache_audit=verify_cache(CACHE,data,tokenizer);gates,gate_audit=verify_gates(cache);pairing=get_pairing(data,cache)
            save(out/'pairing.json',pairing);identity=gates['native_identity_sha256']
            for path in (CACHE,GATE_CACHE,Path(gates['tensor_file']),Path(gates['proof_file'])):bindings[str(path)]=sha(path)
            audited=dict(audited,cache_audit=cache_audit,gate_cache_audit=gate_audit,native_identity_sha256=identity,pairing_object_sha256=objsha(pairing))
        save(out/'data_audit.json',audited)
        summary=dict(passed=True,completed=True,source_sha256=frozen,tests=tests,data_bindings=bindings,cache_audit=cache_audit,gate_cache_audit=gate_audit,
            native_identity_sha256=identity,audit_file=str(out/'data_audit.json'),audit_sha256=sha(out/'data_audit.json'),manifest_only=args.check_manifests)
        if args.check_data:summary.update(pairing_file=str(out/'pairing.json'),pairing_sha256=sha(out/'pairing.json'),pairing_object_sha256=objsha(pairing))
    else:summary=report_runs(args.runs,out,frozen);summary['tests']=tests
    need(sources()==frozen,'Reporter source changed during execution');summary.update(seconds=time.perf_counter()-started,slurm_job_id=os.environ['SLURM_JOB_ID'])
    save(out/'summary.json',summary);(out/'INDEX.md').write_text('# V18 independent audit\n\n[Summary](summary.json) · [Sources](source_hashes.json).\n')
    print(json.dumps(dict(passed=True,summary_file=str(out/'summary.json'),seconds=summary['seconds'])),flush=True)


if __name__=='__main__':main()

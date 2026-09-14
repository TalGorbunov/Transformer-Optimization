"""Independently verify and report V5 extensive versus averaged visual evidence.

Frozen V4 data/prediction validators are reused without changing those files.
SUM=N_visible*MEAN at fixed parameters/query: this is an exploratory scaling
and optimization comparison, not a new attention family or bandwidth theorem.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
from statistics import mean
import sys

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0,str(REPO))
from scripts.report_native_vision_v2 import CELLS, compare, extension, verify_anchors
from scripts.report_native_vision_v3 import compare_pooled, describe_rows
from scripts.report_native_vision_v4 import (
    CELL_HISTOGRAMS, DATA_ROOT, RUNTIME_FIELDS, audit_shared_runtime, close,
    digest, finite, need, object_sha, read, valid_sha, verify_data,
    verify_metrics, verify_predictions, write,
)

CONDITIONS=('sum','mean')
SEEDS=(4,5)
BOOTSTRAP_SEED=20260916
CHECKPOINT_ROOT=Path('/mnt/ckpts/gabriele/gnn_transformer/native_aggregation_vlm_v5')
CONTRASTS={'familiar_ood':CELLS[1:3],'N16':CELLS[:1],'unseen_count':CELLS[3:]}
REPORT_SOURCES=('scripts/report_native_vision_v5.py','scripts/report_native_vision_v5_release.py','scripts/report_native_vision_v4.py',
                'scripts/report_native_vision_v3.py','scripts/report_native_vision_v2.py',
                'slurm/native_aggregation_vision_v5_report_check.sbatch',
                'slurm/native_aggregation_vision_v5_report.sbatch')
BRANCH_SHAPES={'query.weight':[96,3584],'memory.weight':[96,3584],
               'read.weight':[96,96],'read.bias':[96],'up.weight':[3584,96]}
PARAMETERS={'branch':1041504,'lora':720896,'total_trainable':1762400}


def architecture(condition):
    need(condition in CONDITIONS,'Unregistered pooling condition')
    operator=dict(variant='independent_visual_memory',merge=condition,
        memory='ordinary_final_visual_output_after_window_restoration',
        normalized_inputs='fixed_tokenwise_RMS_eps1e-6',centered_messages=True,
        keys_values='shared_learned_projection',language_only=True,
        visibility='only_complete_images_before_each_query')
    return dict(protocol='vision_v5_extensive',model='Qwen/Qwen2.5-VL-7B-Instruct',
        arm='independent',layer_index=14,rank=96,lora_layers=4,lora_rank=8,lora_alpha=16.0,
        resize=392,quantization='nf4',operator=operator)


def verify_presentations(run, config, history, data, initialization, tokenizer):
    schedule, tokens = data['schedule'], data['token_audit']['records']
    sources = {r['sid']: r for n in (8, 16) for r in data['sources'][f'train_N{n}']}
    order = read(run / 'training_order.json')
    layouts = read(run / 'training_slot_layouts.json')
    rows = [json.loads(line) for line in (run / 'presentations.jsonl').read_text().splitlines()]
    need(len(order) == 9 and len(rows) == 1620 and set(layouts) == {str(i) for i in range(180)},
         'Incomplete training order/presentation/layout ledger')
    need(len(initialization['tensors']) == 37, 'Expected 37 trainable tensor states')
    for block in range(9):
        slots = list(range(180)); random.Random(config['seed'] + block).shuffle(slots)
        sids = [schedule['conditions']['refresh'][block][slot] for slot in slots]
        need(order[block] == dict(block_index=block, epoch=block + 1,
             ordered_slot_indices=slots, ordered_sids=sids), 'Actual training order differs from registered shuffle')
        epoch = history[block]
        need(epoch['ordered_slots_sha256'] == object_sha(slots)
             and epoch['ordered_sids_sha256'] == object_sha(sids)
             and epoch['optimizer_state_steps'] == [45 * (block + 1)]
             and epoch['optimizer_state_entries'] == 37, 'Adam/order history mismatch')
        losses = []
        for order_index, (slot, sid) in enumerate(zip(slots, sids)):
            row, source = rows[block * 180 + order_index], sources[sid]
            expected = dict(block_index=block, epoch=block + 1, optimizer_step=45 * block + order_index // 4 + 1,
                order_index=order_index, slot=slot, sid=sid, path=source['path'], n_frames=source['n_frames'],
                gold=source['gold'], qa_sha256=source['qa_sha256'], content_sha256=source['content_sha256'],
                question_sha256=hashlib.sha256(source['question'].encode()).hexdigest())
            need(all(row.get(k) == v for k, v in expected.items()), 'Training presentation identity/step mismatch')
            target = tokenizer(str(source['gold']), add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
            signature = layouts[str(slot)]
            need(row['target_ids'] == target == signature['target_ids'], 'Training answer/EOS targets differ')
            for key in ('prompt_tokens', 'input_ids_sha256'):
                need(row[key] == tokens[sid][key] == signature[key], 'Actual training token hash/length differs')
            need(signature['image_grid_thw'] == tokens[sid]['image_grid_thw']
                 and signature['image_tokens'] == sum(math.prod(grid) // 4 for grid in signature['image_grid_thw'])
                 and len(signature['image_grid_thw']) == source['n_frames'], 'Image layout differs')
            need(finite(row['answer_token_ce']) and row['answer_token_ce'] >= 0, 'Nonfinite training CE')
            losses.append(row['answer_token_ce'])
        need(close(epoch['mean_answer_token_ce'], mean(losses)), 'Block loss differs from all presentations')
    expected_audit = dict(presentations=1620, image_frame_presentations=19440,
        unique_training_sids_seen=1540,
        presentations_sha256=digest(run / 'presentations.jsonl'),
        training_order_sha256=digest(run / 'training_order.json'),
        training_slot_layouts_sha256=digest(run / 'training_slot_layouts.json'),
        all_observed_slot_token_layouts_equal=True)
    need(sum(r['n_frames'] for r in rows) == 19440
         and len({r['sid'] for r in rows}) == expected_audit['unique_training_sids_seen']
         and read(run / 'presentation_audit.json') == expected_audit, 'Presentation budget/audit differs')
    return dict(rows=rows, order=order, layouts=layouts, audit=expected_audit, initialization=initialization)


def verify_run(run, condition, seed, expected_hashes, data, torch, tokenizer):
    config, summary = read(run / 'config.json'), read(run / 'summary.json')
    need(config['code_sha256'] == expected_hashes and read(run / 'source_hashes.json') == expected_hashes, 'Frozen training source ledger differs')
    for name, sha in expected_hashes.items():
        need(digest(run / 'code' / name.replace('/', '_')) == sha, f'Training snapshot changed: {name}')
    required = dict(arm='independent', condition=condition, seed=seed, profile=False, eval_only=False,
        epochs=9, accumulation=4, layer_index=14, rank=96, lora_layers=4, lora_rank=8,
        lora_alpha=16.0, resize=392, lr=.001, lr_lora=.0001, max_grad_norm=1.0, max_steps=0,
        max_seq_tokens=16000, max_new_tokens=4, train_ns=[8, 16], dev_ns=[8, 16], eval_ns=[16, 32, 64],
        gold_max=16, eval_modes='all', limit_dev=36, limit_eval=108, limit_count=64,
        checkpoint=None, gradient_checkpointing=False, center_messages=False, data_seed=20260914, training_schedule_condition='refresh', model=architecture(condition)['model'])
    need(all(config.get(k) == v for k, v in required.items()), 'Registered model/training configuration differs')
    need(not config.get('test_epochs') and config['architecture'] == architecture(condition), 'Architecture/test selection differs')
    need(config['generation_policy'] == dict(do_sample=False, repetition_penalty=1.0,
         max_new_tokens=4, output_logits=True), 'Generation policy differs')
    need(run.name == config['run_id'] == summary['run_id'] and summary['condition'] == condition
         and summary['arm'] == 'independent' and summary['profile'] is False, 'Run identity differs')
    for key, count in (('branch', 1041504), ('lora', 720896), ('total_trainable', 1762400)):
        need(config['parameters'][key] == summary['parameters'][key] == count, 'Parameter budget differs')
    for copied, configured, filename, hashkey in (
        ('staged_manifest.json', 'manifest', 'main_manifest.json', 'manifest_sha256'),
        ('staged_count_manifest.json', 'count_manifest', 'count_manifest.json', 'count_manifest_sha256'),
        ('schedule.json', 'schedule', 'schedule.json', 'schedule_sha256')):
        need(Path(config[configured]).resolve() == DATA_ROOT / filename
             and digest(run / copied) == config[hashkey] == digest(DATA_ROOT / filename), 'Copied/frozen data differs')
    need(config['schedule_audit'] == data['schedule_audit']
         and summary['schedule_sha256'] == config['schedule_sha256']
         and config['prompt_token_audit'] == data['schedule']['prompt_token_audit']
         and digest(run / 'staged_prompt_token_audit.json') == config['prompt_token_audit']['sha256'],
         'Schedule or CPU token-audit provenance differs')
    manifest = read(run / 'data_manifest.json')
    need(set(manifest) == set(CELL_HISTOGRAMS), 'Selected manifest cells differ')
    fields = ('path', 'sid', 'n_frames', 'gold', 'question', 'qa_sha256', 'split', 'content_sha256',
              'target_character', 'target_room', 'pair_id', 'anchor_id', 'test_family',
              'parent_n_frames', 'parent_positions', 'anchor_positions')
    for cell, sources in data['sources'].items():
        selected = manifest[cell]
        need(selected['n'] == len(sources) == len(selected['samples'])
             and selected['gold_histogram'] == {str(k): v for k, v in CELL_HISTOGRAMS[cell].items()},
             'Selected sample count/gold histogram differs')
        for actual, source in zip(selected['samples'], sources):
            need(all(actual.get(k) == source.get(k) for k in fields), 'Selected source order/content differs')
            need(len(actual['image_files']) == len(source['image_files'])
                 and all(all(a.get(k) == b.get(k) for k in ('path', 'bytes', 'sha256'))
                         for a, b in zip(actual['image_files'], source['image_files'])), 'Selected image provenance differs')
    initialization = read(run / 'initialization.json')
    need(initialization['combined_sha256'] == config['initial_parameter_sha256']
         == summary['initial_parameter_sha256'], 'Initialization hash differs')
    need(all(valid_sha(initialization[k]) for k in ('combined_sha256', 'branch_sha256', 'lora_sha256')),
         'Invalid initialization digest')
    names = [r['name'] for r in initialization['tensors']]
    need(len(names) == len(set(names)) == 37 and sum(r['numel'] for r in initialization['tensors']) == 1762400,
         'Initialization tensor coverage differs')
    need(all(r['dtype'] == 'torch.float32' and math.prod(r['shape']) == r['numel'] and valid_sha(r['sha256'])
             for r in initialization['tensors']), 'Invalid initialization tensor metadata')
    branch_shapes={r['name'].removeprefix('branch.'):r['shape'] for r in initialization['tensors'] if r['name'].startswith('branch.')}
    need(branch_shapes == BRANCH_SHAPES, 'Exact independent-memory initialization shapes differ')
    need(len([name for name in names if name.startswith('lora.')]) == 32, 'Expected32 upper-LoRA tensors')
    history = summary['training']
    need(history == read(run / 'training.json') and len(history) == 9, 'Incomplete or inconsistent block history')
    for block, epoch in enumerate(history):
        need((epoch['block_index'], epoch['epoch'], epoch['step'], epoch['n']) == (block, block + 1, 45 * (block + 1), 180),
             'Nine complete 45-update blocks required')
        dev = read(run / f'dev_epoch{block + 1}.json')
        need(dev['metrics'] == epoch['dev'], 'Saved development evaluations differ from training history')
        verify_predictions(dev['predictions'], data['sources'], ('dev_N8', 'dev_N16'), f'dev_epoch{block + 1}', tokenizer)
        verify_metrics(epoch['dev'], dev['predictions'], ('dev_N8', 'dev_N16'))
    ledgers = verify_presentations(run, config, history, data, initialization, tokenizer)
    need(summary['presentation_audit'] == ledgers['audit'], 'Summary presentation audit differs')
    gradients = verify_gradients(run)
    ledgers['gradients'] = gradients
    best = max(history, key=lambda e: (sum(m['correct'] for m in e['dev']) / 72,
        -sum(m['gold_first_token_nll'] * m['n'] for m in e['dev']) / 72))
    checkpoint = Path(summary['selected_checkpoint']).resolve()
    need(checkpoint == CHECKPOINT_ROOT / run.name / 'best.pt', 'Selected checkpoint path differs')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    need(saved['architecture'] == architecture(condition) and saved['epoch'] == best['epoch']
         and saved['step'] == best['step'] and saved['dev'] == best['dev'], 'Checkpoint selection/architecture differs')
    for key in ('run_id', 'condition', 'seed', 'architecture', 'code_sha256', 'manifest_sha256',
                'count_manifest_sha256', 'schedule_sha256', 'initial_parameter_sha256', 'parameters'):
        need(saved['config'][key] == config[key], f'Selected checkpoint configuration differs: {key}')
    need(set(saved['branch']) == set(BRANCH_SHAPES), 'Independent-memory checkpoint keys differ')
    need(sum(v.numel() for v in saved['branch'].values()) == PARAMETERS['branch'] and
         sum(v.numel() for pair in saved['lora'].values() for v in pair) == PARAMETERS['lora'], 'Checkpoint parameter totals differ')
    tensor_map = {'branch.' + k: v for k, v in saved['branch'].items()}
    tensor_map.update({f'lora.{key}.{suffix}': value for key, pair in saved['lora'].items()
                       for suffix, value in zip(('A', 'B'), pair)})
    need(set(tensor_map) == set(names) and all(len(pair) == 2 for pair in saved['lora'].values()),
         'Selected checkpoint parameter identities differ')
    for item in initialization['tensors']:
        tensor = tensor_map[item['name']]
        need(list(tensor.shape) == item['shape'] and str(tensor.dtype) == item['dtype']
             and bool(torch.isfinite(tensor).all()), 'Saved trained tensor shape/dtype/finite check failed')
    del tensor_map, saved
    rows = read(run / 'predictions.json')
    verify_predictions(rows, data['sources'], CELLS, 'test', tokenizer)
    verify_anchors(rows, 'length', CELLS[:3], 108); verify_anchors(rows, 'unseen_count', CELLS[3:], 64)
    verify_metrics(summary['results'], rows, CELLS)
    prefill_diagnostics=verify_prefill_diagnostics(run,rows,condition,torch)
    info = describe_rows(rows)
    info['prefill_diagnostics']=prefill_diagnostics
    info.update(runtime_provenance={key: config[key] for key in RUNTIME_FIELDS},
        parameters=summary['parameters'], gradients=gradients['summary'], selected_epoch=best['epoch'],
        selected_checkpoint=str(checkpoint), selected_checkpoint_sha256=digest(checkpoint),
        training=history, presentation_audit=ledgers['audit'], initial_parameter_sha256=initialization['combined_sha256'],
        manifest_sha256=config['manifest_sha256'], count_manifest_sha256=config['count_manifest_sha256'],
        schedule_sha256=config['schedule_sha256'], stage_audit_sha256=data['stage_audit_sha256'],
        extensions={f'{family}_{lo}_to_{hi}': extension(rows, family, lo, hi)
                    for family, lo, hi in [('length', 16, 32), ('length', 32, 64), ('length', 16, 64), ('unseen_count', 32, 64)]},
        memory=dict(training_peak_allocated_bytes=max(e['peak_cuda_allocated_bytes'] for e in history),
                    training_peak_reserved_bytes=max(e['peak_cuda_reserved_bytes'] for e in history),
                    evaluation_peak_allocated_bytes=max(m['peak_cuda_allocated_bytes'] for m in summary['results'])))
    info['deterioration'] = {f'N16_to_N{n}': info['cells']['length_N16']['exact'] - info['cells'][f'length_N{n}']['exact']
                             for n in (32, 64)}
    info['diagnostic_artifact_sha256'] = {path.name: digest(path) for path in sorted(run.glob('*diagnostic*.json'))}
    return rows, info, ledgers




def verify_prefill_diagnostics(run,rows,condition,torch,expected_count=452,directory_override=None):
    """Audit descriptive first-query tensors; no oracle labels enter the branch."""
    directory=DATA_ROOT.parent/'v5_diagnostics'/run.name if directory_override is None else Path(directory_override)
    expected_shapes=lambda n:dict(query_position=(),image_ends=(n,),visible=(n,),
        frame_messages=(n,96),frame_message_norms=(n,),merged_message=(96,),residual=(3584,),native_output_norm=())
    records=[]
    for row in rows:
        path=Path(row['branch_diagnostics_path'])
        need(path.resolve()==directory/f"{row['sid']}.pt" and valid_sha(row['branch_diagnostics_sha256']) and
             digest(path)==row['branch_diagnostics_sha256'],'First-prefill diagnostic path/hash differs')
        value=torch.load(path,map_location='cpu',weights_only=True)
        shapes=expected_shapes(row['n_frames'])
        need(set(value)==set(shapes) and all(isinstance(value[k],torch.Tensor) and tuple(value[k].shape)==shape
             for k,shape in shapes.items()),'First-prefill tensor keys/shapes differ')
        need(all(bool(torch.isfinite(t).all()) for t in value.values()),'Nonfinite first-prefill diagnostic')
        need(value['visible'].dtype==torch.bool and bool(value['visible'].all()) and
             int(value['query_position'])==row['prompt_tokens']-1,'Diagnostic is not the last original prompt query')
        ends=value['image_ends']
        need(ends.dtype==torch.int64 and value['query_position'].dtype==torch.int64 and
             bool((ends>=0).all()) and bool((ends[1:]>ends[:-1]).all()) and
             torch.equal(value['visible'],value['query_position']>ends),'Diagnostic visibility/boundaries differ')
        frames=value['frame_messages'].float()
        need(torch.allclose(value['frame_message_norms'].float(),frames.norm(dim=-1),rtol=1e-5,atol=1e-6),
             'Recorded per-frame norms differ from saved messages')
        merged=frames.sum(0) if condition=='sum' else frames.mean(0)
        need(torch.allclose(value['merged_message'].float(),merged,rtol=1e-4,atol=1e-5),
             'First-prefill merged message does not match registered SUM/MEAN')
        residual=float(value['residual'].float().norm());native=float(value['native_output_norm'])
        need(native>=0 and close(row['branch_residual_norm'],residual) and
             close(row['native_attention_output_norm'],native),'Saved residual/native scalar norms differ')
        ratio=residual/native if native else None
        need((ratio is None and row['branch_to_native_attention_norm_ratio'] is None) or
             close(ratio,row['branch_to_native_attention_norm_ratio']),'Saved branch/native norm ratio differs')
        records.append(dict(cell=row['cell'],sid=row['sid'],path=str(path),sha256=row['branch_diagnostics_sha256'],
            branch_residual_norm=residual,native_attention_output_norm=native,branch_to_native_norm_ratio=ratio,
            mean_frame_message_norm=float(value['frame_message_norms'].float().mean()),
            merged_message_norm=float(value['merged_message'].float().norm())))
        del value
    need(len(records)==expected_count and len({r['path'] for r in records})==expected_count,'Diagnostic coverage differs from expected test scenes')
    cells={}
    for cell in CELLS:
        group=[r for r in records if r['cell']==cell]
        ratios=[r['branch_to_native_norm_ratio'] for r in group if r['branch_to_native_norm_ratio'] is not None]
        cells[cell]=dict(n=len(group),mean_branch_residual_norm=mean(r['branch_residual_norm'] for r in group),
            mean_native_attention_output_norm=mean(r['native_attention_output_norm'] for r in group),
            mean_branch_to_native_norm_ratio=mean(ratios) if ratios else None,ratio_denominator=len(ratios),
            mean_frame_message_norm=mean(r['mean_frame_message_norm'] for r in group),
            mean_merged_message_norm=mean(r['merged_message_norm'] for r in group))
    return dict(records=records,cells=cells,all_shapes_hashes_visibility_and_merge_checks_passed=True,
        scope='Detached first-final-prompt-query diagnostics. Message/residual magnitude is descriptive; it is not evidence attribution, a frame-label classifier, or a causal explanation of model predictions.')


def audit_gradient_rows(rows, hashes):
    need(len(rows)==405 and [r['step'] for r in rows]==list(range(1,406)), 'Incomplete405-update gradient audit')
    expected={f'branch.{i}' for i in range(5)} | {f'lora.{i}' for i in range(32)}
    need(set(hashes)==expected and all(valid_sha(v) for v in hashes.values()), 'First-step gradient hash coverage differs')
    for row in rows:
        fields=('branch_preclip_norm','lora_preclip_norm','combined_preclip_norm')
        need(all(finite(row[k]) and row[k]>=0 for k in fields), 'Nonfinite/negative gradient norm')
        need(type(row['clipped']) is bool and row['clipped']==(row['combined_preclip_norm']>1), 'Clipping flag differs from registered threshold')
        combined=math.hypot(row['branch_preclip_norm'],row['lora_preclip_norm'])
        need(math.isclose(combined,row['combined_preclip_norm'],rel_tol=2e-5,abs_tol=2e-6), 'Group gradient norms do not reconstruct combined norm')
    return dict(updates=405,clipped_updates=sum(r['clipped'] for r in rows),
        clipped_fraction=mean(r['clipped'] for r in rows),
        preclip_norm={group:dict(mean=mean(r[group+'_preclip_norm'] for r in rows),
            minimum=min(r[group+'_preclip_norm'] for r in rows),maximum=max(r[group+'_preclip_norm'] for r in rows))
            for group in ('branch','lora','combined')},
        first_step_gradient_hashes=hashes,
        note='First-step hashes identify emitted preclip float32 gradient bytes; full gradient tensors and Adam moments are not retained. Norm/clipping comparisons are descriptive optimization diagnostics.')


def verify_gradients(run):
    rows=read(run/'gradient_audit.json')
    hashes=read(run/'first_step_gradient_hashes.json')
    summary=audit_gradient_rows(rows,hashes)
    summary.update(gradient_audit_sha256=digest(run/'gradient_audit.json'),
                   first_step_gradient_hashes_sha256=digest(run/'first_step_gradient_hashes.json'))
    return dict(rows=rows,hashes=hashes,summary=summary)


def paired_training_audit(total,average):
    need(total['initialization']==average['initialization'], 'Within-seed initialized parameter bytes differ')
    need(total['layouts']==average['layouts'] and total['order']==average['order'], 'Paired token layouts or scene order differ')
    keys=('block_index','epoch','optimizer_step','order_index','slot','n_frames','gold',
          'sid','path','qa_sha256','content_sha256','question_sha256','prompt_tokens','input_ids_sha256','target_ids')
    need(len(total['rows'])==len(average['rows'])==1620, 'Paired presentation denominators differ')
    need(all(all(a[k]==b[k] for k in keys) for a,b in zip(total['rows'],average['rows'])),
         'The arms did not receive the exact same ordered1620 scenes/targets/tokens')
    first_a,first_b=total['rows'][:4],average['rows'][:4]
    first_losses=[a['answer_token_ce']-b['answer_token_ce'] for a,b in zip(first_a,first_b)]
    hashes_a,hashes_b=total['gradients']['hashes'],average['gradients']['hashes']
    return dict(presentations_matched=1620,distinct_scenes=1540,frame_presentations=19440,
        initial_parameters_identical=True,all_ordered_scenes_tokens_targets_steps_identical=True,
        first_optimizer_batch_loss_difference=dict(n=4,mean_sum_minus_mean=mean(first_losses),
            max_absolute=max(abs(v) for v in first_losses),exactly_equal=sum(v==0 for v in first_losses)),
        first_step_gradient_hash_equality={group:dict(equal=sum(hashes_a[k]==hashes_b[k] for k in hashes_a if k.startswith(group+'.')),
            tensors=sum(k.startswith(group+'.') for k in hashes_a)) for group in ('branch','lora')},
        note='Equal nominal initialization/order does not imply bitwise-identical training. SUM/MEAN alter gradient magnitude, common clipping, and later optimization; gradient hashes are diagnostics, not selection criteria.')


def decision_criteria(contrasts):
    per_seed={seed:bool(contrasts[seed]['familiar_ood']['exact_gain']>=.05-1e-12 and
                         contrasts[seed]['N16']['exact_gain']>=-.05-1e-12) for seed in SEEDS}
    return dict(V5_1=all(per_seed.values()),primary_each_seed=per_seed,primary_both_seeds=all(per_seed.values()))


def audit_native_reference_observation(row,require_pass=True):
    """Verify the frozen centered/probability gate, retaining raw diagnostics."""
    nonnegative=('maximum_absolute_logit_difference','rms_logit_difference',
                 'centered_maximum_absolute_logit_difference','centered_rms_logit_difference')
    for prefix in ('','native_reference_'):
        need(all(finite(row[prefix+k]) and row[prefix+k]>=0 for k in nonnegative),
             'Nonfinite/negative native-reference discrepancy')
        need(all(finite(row[prefix+k]) for k in
                 ('mean_logit_difference','kl_cached_to_full','kl_full_to_cached')),
             'Nonfinite raw-shift/probability diagnostic')
        need(finite(row[prefix+'total_variation']) and 0<=row[prefix+'total_variation']<=1,
             'Invalid total variation')
        left,right=row[prefix+'cached_top1_token'],row[prefix+'uncached_top1_token']
        need(type(left) is int and type(right) is int and min(left,right)>=0 and
             row[prefix+'raw_top1_equal'] is (left==right), 'Inconsistent raw top-1 diagnostic')
    relative_maximum=1.5*row['native_reference_centered_maximum_absolute_logit_difference']+.0625
    relative_rms=1.5*row['native_reference_centered_rms_logit_difference']+.005
    maximum,rms=min(.25,relative_maximum),min(.05,relative_rms)
    limits=dict(relative_centered_maximum_absolute_tolerance=relative_maximum,
                relative_centered_rms_tolerance=relative_rms,
                centered_maximum_absolute_tolerance=maximum,centered_rms_tolerance=rms,
                total_variation_tolerance=.01)
    need(all(finite(row[k]) and close(row[k],v) for k,v in limits.items()),
         'Stored cache tolerances differ from the registered centered native-reference/cap rule')
    passed=bool(row['centered_maximum_absolute_logit_difference']<=maximum and
                row['centered_rms_logit_difference']<=rms and row['total_variation']<=.01 and
                row['raw_top1_equal'] is True)
    need(row.get('passed') is passed,'Stored strict numerical result differs from unchanged criterion')
    need(not require_pass or passed,'Centered/probability cached-full gate failed')
    need(len(row['generated_ids'])==2 and row['native_reference_same_first_token']==row['generated_ids'][0],
         'Native reference was not conditioned on the enabled first-token prefix')
    return passed


def audit_cache_profile(cache):
    need(cache.get('passed') is True and finite(cache['up_projection_norm']) and cache['up_projection_norm']>0 and
         cache['criterion']=='paired_native_reference_centered_and_probability' and
         cache['native_reference_multiplier']==1.5 and cache['maximum_absolute_floor']==.0625 and
         cache['rms_floor']==.005 and cache['centered_maximum_absolute_cap']==.25 and
         cache['centered_rms_cap']==.05 and cache['total_variation_cap']==.01 and
         cache['raw_top1_equality_required'] is True and cache['raw_logit_differences_diagnostic_only'] is True and
         cache['probability_metrics_dtype']=='float64' and
         cache['disabled_reference_forces_only_enabled_first_token'] is True and
         cache['visual_calls']==20 and cache['software_only_forced_two_token_trajectories'] is True and
         cache['fresh_example_reset']=='N16_A,N64_B,N16_C,N64_D,N16_A: exact repeated greedy IDs and logits',
         'Nonzero/cache/reset centered-probability protocol differs')
    observations=cache['observations']
    need([row['n_frames'] for row in observations]==[16,64,16,64,16] and
         len({row['sid'] for row in observations[:4]})==4, 'Four distinct N16/N64 cases plus reset required')
    for index,row in enumerate(observations):
        audit_native_reference_observation(row)
        need(row['observation_index']==index and row['reset_repeat_of']==(0 if index==4 else None) and
             row['case_role']==('prospective_unused_record' if index in (2,3) else 'original_case_or_reset'),
             'Prospective/reset case role differs')
        need(row['native_visual_calls_cached_generation']==row['native_visual_calls_uncached_comparison']==1 and
             row['native_visual_calls_disabled_cached']==row['native_visual_calls_disabled_full']==1 and
             row['uncached_memory_cleared'] is True and valid_sha(row['first_logits_sha256']),
             'Observed one-vision-forward/cache-clear check failed')
    need(all(observations[0][k]==observations[4][k] for k in ('sid','generated_ids','first_logits_sha256')),
         'Observed fresh-example reset differs')
    return observations


def verify_strict_gates(root,expected_hashes):
    path=root/'implementation_gates.json'
    gates=read(path)
    need(gates.get('schema_version')==1 and gates.get('passed') is True and
         gates.get('source_sha256')==expected_hashes, 'Implementation gates do not bind the frozen launch sources')
    entries=gates['artifacts']
    kinds=[entry['kind'] for entry in entries]
    need(len(kinds)==len(set(kinds)) and {'core_cpu','profile_sum','profile_mean'}<=set(kinds), 'Missing/duplicate required implementation gates')
    for entry in entries:
        artifact=Path(entry['path'])
        need(artifact.is_absolute() and digest(artifact)==entry['sha256'], 'Implementation gate artifact missing or changed')
        if entry['kind']=='core_cpu':
            payload=read(artifact)
            need(payload.get('passed') is True and payload.get('unit_tests_passed') is True and
                 payload.get('schedule_checks_passed') is True and payload.get('source_sha256')==expected_hashes,
                 'CPU implementation/schedule gate did not pass with current launch sources')
        elif entry['kind'] in ('profile_sum','profile_mean'):
            condition=entry['kind'].removeprefix('profile_')
            payload=read(artifact)
            need(payload.get('passed') is True, 'Native profile gate did not pass')
            zero=payload['zero_initialization'];cache=payload['nonzero_cache_and_reset']
            need(zero.get('passed') is True and zero['maximum_absolute_logit_difference']==0 and
                 zero['criterion']=='exact_equal' and zero['zero_up_projection'] is True and
                 zero['uncached_memory_cleared'] is True, 'Zero-initialization profile parity failed')
            observations=audit_cache_profile(cache)
            profile_config=read(artifact.parent/'config.json')
            need(profile_config['profile'] is True and profile_config['condition']==condition and
                 profile_config['code_sha256']==expected_hashes and profile_config['architecture']==architecture(condition),
                 'Profile arm, architecture or source ledger differs')
            staged_path=artifact.parent/'staged_manifest.json'
            need(digest(staged_path)==profile_config['manifest_sha256']==digest(DATA_ROOT/'profile_manifest.json'),
                 'Profile manifest differs from frozen software data')
            profile_splits=read(staged_path)['splits']
            expected_cases=[profile_splits[f'test_N{n}']['samples'][i]['sid']
                            for n,i in ((16,0),(64,0),(16,1),(64,1),(16,0))]
            need([row['sid'] for row in observations]==expected_cases,
                 'Cache gate did not use the registered first/second frozen software records')
            import torch
            profile_rows=read(artifact.parent/'predictions.json')
            need(Counter(row['cell'] for row in profile_rows)==Counter({cell:2 for cell in CELLS}),
                 'Profile did not finish all10 length/multitoken software examples')
            verify_prefill_diagnostics(artifact.parent,profile_rows,condition,torch,expected_count=10)
    return dict(path=str(path.resolve()),sha256=digest(path),artifacts=entries,
                scope='Binds the pre-main implementation/core/profile acceptance ledger and exact artifacts; gate details remain in those source artifacts.')


def verify_gates(root,expected_hashes):
    schema=read(root/'implementation_gates.json').get('schema_version')
    if schema==1:
        result=verify_strict_gates(root,expected_hashes)
        return dict(result,schema_version=1,accepted_for_exploratory_training=True,
                    strict_numerical_gates_passed=True)
    need(schema==2,'Unknown implementation release schema')
    from scripts.report_native_vision_v5_release import verify_engineering_release
    return verify_engineering_release(root,expected_hashes,sys.modules[__name__])


def self_test(np):
    candidate,control={},{}
    for seed in SEEDS:
        candidate[seed],control[seed]=[],[]
        for i in range(4):
            for cell in CONTRASTS['familiar_ood']:
                base=dict(cell=cell,sid=f'{i}_{cell}',path=f'/sample/{i}/{cell}',gold=i,pair_id=str(i))
                candidate[seed].append(dict(base,exact=i%2==0))
                control[seed].append(dict(base,exact=False))
    rng=lambda:np.random.default_rng(BOOTSTRAP_SEED)
    single=compare(candidate[4],control[4],CONTRASTS['familiar_ood'],np,rng())
    pooled=compare_pooled(candidate,control,CONTRASTS['familiar_ood'],np,rng())
    need((pooled['anchors'],pooled['n'],pooled['unique_examples'])==(4,16,8), 'Fixed-seed pooling doubled independent sample count')
    np.testing.assert_allclose(single['bootstrap95'],pooled['bootstrap95'])
    good={seed:{'familiar_ood':{'exact_gain':.05},'N16':{'exact_gain':-.05}} for seed in SEEDS}
    need(decision_criteria(good)['V5_1'],'Inclusive primary boundaries rejected')
    good[5]['familiar_ood']['exact_gain']=.049
    need(not decision_criteria(good)['V5_1'],'Pooling rescued a failing seed')
    good[5]['familiar_ood']['exact_gain']=.8;good[4]['N16']['exact_gain']=-.051
    need(not decision_criteria(good)['V5_1'],'N16 guard ignored')
    bad={seed:[dict(r) for r in rows] for seed,rows in candidate.items()};bad[5][0]['path']='/changed'
    try:
        compare_pooled(bad,control,CONTRASTS['familiar_ood'],np,rng())
    except ValueError:
        pass
    else:
        raise AssertionError('Pooled identity mismatch accepted')
    bad={seed:[dict(r) for r in rows] for seed,rows in candidate.items()};bad[5].append(dict(bad[5][0]))
    try:
        compare_pooled(bad,control,CONTRASTS['familiar_ood'],np,rng())
    except ValueError:
        pass
    else:
        raise AssertionError('Duplicate pooled observation accepted')
    invalid=[dict(cell='length_N16',pair_id='a',prediction=None,exact=False),
             dict(cell='length_N32',pair_id='a',prediction=None,exact=False)]
    ex=extension(invalid,'length',16,32)
    need(ex['both_correct']==ex['both_parsed']==ex['same_parsed_prediction']==0, 'Two malformed answers counted as invariance')
    for condition in CONDITIONS:
        need(architecture(condition)['operator']['merge']==condition,'Architecture pooling identity differs')
    try:
        architecture('max')
    except ValueError:
        pass
    else:
        raise AssertionError('Unregistered pooling accepted')
    hashes={f'{group}.{i}':'a'*64 for group,count in (('branch',5),('lora',32)) for i in range(count)}
    gradients=[dict(step=i,branch_preclip_norm=3.,lora_preclip_norm=4.,combined_preclip_norm=5.,clipped=True) for i in range(1,406)]
    need(audit_gradient_rows(gradients,hashes)['clipped_updates']==405,'Valid gradient ledger rejected')
    gradients[-1]['combined_preclip_norm']=1.
    try:
        audit_gradient_rows(gradients,hashes)
    except ValueError:
        pass
    else:
        raise AssertionError('Inconsistent gradient/clipping ledger accepted')
    reference=dict(passed=True,generated_ids=[7,8],native_reference_same_first_token=7,
        relative_centered_maximum_absolute_tolerance=1.5*.2+.0625,
        relative_centered_rms_tolerance=1.5*.04+.005,
        centered_maximum_absolute_tolerance=.25,centered_rms_tolerance=.05,total_variation_tolerance=.01)
    # A large common logit shift is diagnostic only; centered/probability limits apply.
    for prefix in ('','native_reference_'):
        reference.update({prefix+k:v for k,v in dict(maximum_absolute_logit_difference=10.,
            rms_logit_difference=9.,mean_logit_difference=-9.,
            centered_maximum_absolute_logit_difference=.25 if not prefix else .2,
            centered_rms_logit_difference=.05 if not prefix else .04,total_variation=.01,
            kl_cached_to_full=.001,kl_full_to_cached=.001,cached_top1_token=8,uncached_top1_token=8,
            raw_top1_equal=True).items()})
    audit_native_reference_observation(reference)
    for field,value in (('centered_maximum_absolute_logit_difference',.251),
                        ('centered_rms_logit_difference',.051),('total_variation',.0101),
                        ('raw_top1_equal',False),('uncached_top1_token',9),('passed',False),
                        ('native_reference_same_first_token',9),('centered_rms_tolerance',.1),
                        ('relative_centered_rms_tolerance',.1),('native_reference_rms_logit_difference',float('nan')),
                        ('mean_logit_difference',float('inf')),('native_reference_kl_cached_to_full',float('nan'))):
        invalid=dict(reference);invalid[field]=value
        try:
            audit_native_reference_observation(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Invalid centered/probability observation accepted: {field}')
    # Passing the absolute caps alone cannot bypass a tighter native-relative limit.
    tight=dict(reference,native_reference_centered_maximum_absolute_logit_difference=0.,
        native_reference_centered_rms_logit_difference=0.,relative_centered_maximum_absolute_tolerance=.0625,
        relative_centered_rms_tolerance=.005,centered_maximum_absolute_tolerance=.0625,centered_rms_tolerance=.005,
        centered_maximum_absolute_logit_difference=.0625,centered_rms_logit_difference=.005)
    audit_native_reference_observation(tight)
    for field in ('centered_maximum_absolute_logit_difference','centered_rms_logit_difference'):
        invalid=dict(tight);invalid[field]+=.0001
        try:
            audit_native_reference_observation(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('Absolute cap bypassed tighter native-relative limit')
    cache=dict(passed=True,up_projection_norm=1.,criterion='paired_native_reference_centered_and_probability',
        native_reference_multiplier=1.5,maximum_absolute_floor=.0625,rms_floor=.005,
        centered_maximum_absolute_cap=.25,centered_rms_cap=.05,total_variation_cap=.01,
        raw_top1_equality_required=True,raw_logit_differences_diagnostic_only=True,probability_metrics_dtype='float64',
        disabled_reference_forces_only_enabled_first_token=True,visual_calls=20,
        software_only_forced_two_token_trajectories=True,
        fresh_example_reset='N16_A,N64_B,N16_C,N64_D,N16_A: exact repeated greedy IDs and logits',
        observations=[dict(reference,n_frames=n,sid=f'case{idx if idx<4 else 0}',observation_index=idx,
            reset_repeat_of=0 if idx==4 else None,
            case_role='prospective_unused_record' if idx in (2,3) else 'original_case_or_reset',
            native_visual_calls_cached_generation=1,native_visual_calls_uncached_comparison=1,
            native_visual_calls_disabled_cached=1,native_visual_calls_disabled_full=1,
            uncached_memory_cleared=True,first_logits_sha256='a'*64)
            for idx,n in enumerate((16,64,16,64,16))])
    audit_cache_profile(cache)
    for field,value in (('criterion','paired_native_reference'),('visual_calls',12),
                        ('centered_rms_cap',.1),('probability_metrics_dtype','float32')):
        invalid=dict(cache);invalid[field]=value
        try:
            audit_cache_profile(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Old/changed profile protocol accepted: {field}')
    invalid=dict(cache,observations=[dict(row) for row in cache['observations']])
    invalid['observations'][2]['sid']=invalid['observations'][0]['sid']
    try:
        audit_cache_profile(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError('Original case reused as prospective software record')
    failed=dict(reference,total_variation=.02,passed=False)
    need(audit_native_reference_observation(failed,require_pass=False) is False,
         'Descriptive audit relabeled a numerical failure')
    try:
        audit_native_reference_observation(failed)
    except ValueError:
        pass
    else:
        raise AssertionError('Strict schema accepted a failed observation')
    from scripts.report_native_vision_v5_release import self_test_release
    self_test_release(sys.modules[__name__])
    print('PASS: V5 boundaries, shared-anchor pooling, identity/duplicate rejection, invalid parses, pooling, gradients, strict/descriptive numerical gates and explicit release flags')


def render(root,analysis,np,plt):
    results,contrasts,pooled=analysis['conditions'],analysis['sum_minus_mean'],analysis['pooled_sum_minus_mean']
    lines=['# Native vision: extensive versus averaged independent evidence (V5)','',
        'Exploratory reused V2 tests; identical V4 refresh presentations, two fixed seeds and development-selected checkpoints.','',
        '| Pooling | Seed | N16 | N32 | N64 | Familiar OOD | Unseen K9–16 | Selected block |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for condition in CONDITIONS:
        for seed in SEEDS:
            info=results[condition][seed]
            cells=[info['cells'][cell] for cell in CELLS[:3]]+[info['familiar_ood'],info['unseen_count']]
            values=[f"{v['correct']}/{v['n']} ({100*v['exact']:.1f}%)" for v in cells]
            lines.append('| '+' | '.join([condition,str(seed),*values,str(info['selected_epoch'])])+' |')
    lines+=['',f"Both-seed primary screen: **{analysis['criteria']['primary_both_seeds']}**.",'',
        '| SUM minus MEAN | Familiar OOD difference | Paired-anchor95% interval | N16 difference | Pass |',
        '|---|---:|---:|---:|---:|']
    for label,group,passed in [(f'seed{seed}',contrasts[seed],analysis['criteria']['primary_each_seed'][seed]) for seed in SEEDS]+[('pooled fixed seeds',pooled,'descriptive')]:
        metric=group['familiar_ood'];lo,hi=metric['bootstrap95']
        lines.append(f"| {label} | {100*metric['exact_gain']:+.2f}pp | [{100*lo:+.2f},{100*hi:+.2f}]pp | {100*group['N16']['exact_gain']:+.2f}pp | {passed} |")
    lines+=['','Every seed must improve familiar N32/N64 exact by at least5pp and lose at most5pp on N16. Pooling cannot rescue a failed seed.',
        'Each run has216 familiar OOD observations from108 anchors and128 unseen-count observations from64 anchors. Pooled intervals retain both fixed seeds and lengths in each whole-anchor draw.','',
        '## Precision and extensions','',
        '| Pooling/seed | OOD parse | OOD MAE | OOD bias | Count MAE | Count bias | N16→N64 loss | Both correct /108 | Same parsed prediction /108 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    fmt=lambda value:'NA' if value is None else f'{value:.3f}'
    for condition in CONDITIONS:
        for seed in SEEDS:
            info=results[condition][seed];ood=info['familiar_ood'];count=info['unseen_count'];ext=info['extensions']['length_16_to_64']
            lines.append(f"| {condition}/{seed} | {100*ood['parse_rate']:.1f}% | {fmt(ood['mae_parsed'])} | {fmt(ood['bias_parsed'])} | {fmt(count['mae_parsed'])} | {fmt(count['bias_parsed'])} | {100*info['deterioration']['N16_to_N64']:+.2f}pp | {ext['both_correct']}/108 | {ext['same_parsed_prediction']}/108 |")
    lines+=['','MAE/bias condition on parsed outputs; exact and invariance denominators retain invalid outputs. Per-cell and per-K results, count parse denominators, K9 versus K10–16, and all extension transitions are preserved in analysis.json.','',
        '## Training and gradient audit','',
        'Each run used the same1620 ordered presentations,1540 distinct scenes,19440 image frames,405 updates,37 persistent Adam states and nine72-example development evaluations. All452 selected-checkpoint predictions were independently verified. The branch has1,041,504 parameters plus720,896 upper-LoRA parameters,1,762,400 total.','',
        '| Pooling/seed | Clipped updates /405 | Mean branch preclip norm | Mean LoRA preclip norm | First-step branch hash matches /5 | First-step LoRA matches /32 |',
        '|---|---:|---:|---:|---:|---:|']
    for condition in CONDITIONS:
        for seed in SEEDS:
            gradient=results[condition][seed]['gradients'];pairs=analysis['paired_training_audit'][seed]['first_step_gradient_hash_equality']
            lines.append(f"| {condition}/{seed} | {gradient['clipped_updates']}/405 | {gradient['preclip_norm']['branch']['mean']:.5g} | {gradient['preclip_norm']['lora']['mean']:.5g} | {pairs['branch']['equal']}/5 | {pairs['lora']['equal']}/32 |")
    lines+=['','Initial parameter bytes, all scene/order/token/target ledgers and recorded runtime settings match within each seed. First-gradient hashes and preclip norms are reproducibility/optimization diagnostics; gradient tensors and Adam moments were not saved for independent numerical replay. All452 first-prefill diagnostic tensor files are hash-bound and independently checked for shapes, visibility, SUM/MEAN reconstruction and residual/native scalar norms; they do not select checkpoints.','',
        '## Measured cost','',
        '| Pooling/seed | OOD model sec/example | Total sec/example | Training seconds | Peak training allocated GiB |',
        '|---|---:|---:|---:|---:|']
    for condition in CONDITIONS:
        for seed in SEEDS:
            info=results[condition][seed];metric=info['familiar_ood']
            lines.append(f"| {condition}/{seed} | {metric['model_seconds']:.3f} | {metric['total_seconds']:.3f} | {sum(e['seconds'] for e in info['training']):.1f} | {info['memory']['training_peak_allocated_bytes']/2**30:.2f} |")
    gates=analysis['implementation_gates'];post=analysis['post_training_checkpoint_audits']
    lines+=['','## Computational and numerical audit','',
        f"Exploratory-training engineering acceptance: **{gates['accepted_for_exploratory_training']}**. Registered pre-training strict numerical gate passed: **{gates['strict_numerical_gates_passed']}**. These are separate decisions. The native NF4/bf16/SDPA/bitsandbytes backend was retained.",'',
        '| Pooling/seed | State integrity | Strict numerical observations passed | Overall strict result | Failed checks |',
        '|---|---|---:|---|---|']
    for artifact in post['artifacts']:
        for item in artifact['runs']:
            numerical=item['numerical']
            failures='; '.join(f"case{x['observation_index']} N{x['n_frames']}: "+', '.join(x['failed_checks']) for x in numerical['failures']) or 'none'
            lines.append(f"| {item['condition']}/{item['seed']} | True | {numerical['passed_observations']}/5 | {numerical['strict_numerical_gate_passed']} | {failures} |")
    lines+=['','All four development-selected checkpoints completed the same five cache comparisons and ten software evaluations. Every numerical failure is retained; audit results do not select checkpoints or alter main predictions. Complete raw/centered/probability metrics, state checks, software predictions and source/checkpoint hashes are in analysis.json. Five two-token probes do not establish general or long-reasoning cache equivalence.','']
    lines+=['','## Interpretation limits','']+['- '+note for note in analysis['notes']]+['']
    (root/'REPORT.md').write_text('\n'.join(lines))
    colors={'sum':'#0072B2','mean':'#D55E00'}
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for condition in CONDITIONS:
        for seed in SEEDS:
            info=results[condition][seed];style='-' if seed==4 else '--'
            axes[0,0].plot([16,32,64],[info['cells'][f'length_N{n}']['exact'] for n in (16,32,64)],marker='o',linestyle=style,color=colors[condition],label=f'{condition} seed{seed}')
            axes[0,1].plot([32,64],[info['cells'][f'unseen_count_N{n}']['exact'] for n in (32,64)],marker='o',linestyle=style,color=colors[condition])
            axes[1,0].plot(range(1,10),[sum(m['correct'] for m in e['dev'])/72 for e in info['training']],linestyle=style,color=colors[condition])
    groups=[contrasts[seed]['familiar_ood'] for seed in SEEDS]+[pooled['familiar_ood']]
    for y,group in enumerate(groups):
        lo,hi=[100*v for v in group['bootstrap95']]
        axes[1,1].plot([lo,hi],[y,y],color='#333333');axes[1,1].plot([lo,hi],[y,y],'|',color='#333333')
        axes[1,1].plot(100*group['exact_gain'],y,'o',color='#333333')
    axes[1,1].axvline(0,color='grey',linewidth=1);axes[1,1].axvline(5,color='#009E73',linestyle=':')
    axes[1,1].set(yticks=[0,1,2],yticklabels=['seed4','seed5','pooled fixed seeds'],xlabel='SUM minus MEAN (percentage points)',title='Familiar OOD · anchor intervals')
    axes[0,0].set(title='Familiar counts0–8',xlabel='Frames',ylabel='Exact answer',xticks=[16,32,64])
    axes[0,1].set(title='Unseen counts9–16',xlabel='Frames',ylabel='Exact answer',xticks=[32,64])
    axes[1,0].set(title='In-range development',xlabel='Block (45 updates)',ylabel='Exact answer')
    for axis in (axes[0,0],axes[0,1],axes[1,0]):axis.set_ylim(-.02,1.02)
    for axis in axes.flat:axis.grid(alpha=.2)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=4,frameon=False)
    fig.suptitle('Independent visual evidence: extensive vs averaged · exploratory reused test',y=.945)
    fig.tight_layout(rect=(0,0,1,.91));fig.savefig(root/'comparison.png',dpi=170);fig.savefig(root/'comparison.pdf');plt.close(fig)
    print('\n'.join(lines[:23]),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('outputs/native_aggregation_vlm/v5'))
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--check-data',action='store_true')
    parser.add_argument('--check-gates',action='store_true')
    args=parser.parse_args()
    need(bool(os.environ.get('SLURM_JOB_ID')),'Run V5 report/checks on CPU Slurm')
    import numpy as np
    if args.self_test:self_test(np);return
    data=verify_data()
    if args.check_data:
        need(data['schedule_audit']['distinct_by_condition']['refresh']==1540,'V5 requires the immutable V4 refresh schedule')
        print('PASS: unchanged V4 data, refresh schedule, processor token ledger and independent stage audit');return
    expected_hashes=read(args.root/'main/source_hashes.json')
    need(all(digest(REPO/name)==sha for name,sha in expected_hashes.items()),'Frozen launch source files changed')
    gates=verify_gates(args.root,expected_hashes)
    if args.check_gates:
        print('PASS: implementation release provenance; strict_numerical_gates_passed='+str(gates['strict_numerical_gates_passed']));return
    frozen_report=read(args.root/'analysis_source_hashes.json')
    need(frozen_report=={name:digest(REPO/name) for name in REPORT_SOURCES},'Analysis source changed after freeze')
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(architecture('sum')['model'],local_files_only=True,use_fast=False)
    results,predictions,ledgers,runs={},{},{},{}
    for condition in CONDITIONS:
        results[condition],predictions[condition],ledgers[condition],runs[condition]={},{},{},{}
        for seed in SEEDS:
            candidates=list((args.root/'main'/condition/f'seed{seed}').glob('*/summary.json'))
            need(len(candidates)==1,f'Expected exactly one completed {condition}/seed{seed}')
            run=candidates[0].parent
            rows,info,ledger=verify_run(run,condition,seed,expected_hashes,data,torch,tokenizer)
            predictions[condition][seed],results[condition][seed],ledgers[condition][seed],runs[condition][seed]=rows,info,ledger,str(run)
    from scripts.report_native_vision_v5_release import verify_post_training_audits
    post_audits=verify_post_training_audits(args.root,expected_hashes,results,runs,sys.modules[__name__])
    paired={seed:paired_training_audit(ledgers['sum'][seed],ledgers['mean'][seed]) for seed in SEEDS}
    runtime=audit_shared_runtime([results[c][s]['runtime_provenance'] for c in CONDITIONS for s in SEEDS])
    rng=lambda:np.random.default_rng(BOOTSTRAP_SEED)
    contrasts={seed:{name:compare(predictions['sum'][seed],predictions['mean'][seed],cells,np,rng()) for name,cells in CONTRASTS.items()} for seed in SEEDS}
    pooled={name:compare_pooled(predictions['sum'],predictions['mean'],cells,np,rng()) for name,cells in CONTRASTS.items()}
    notes=[
        'V2/V3/V4 test scenes have already been inspected. These are reused exploratory evaluations, not fresh confirmation or a general benchmark win.',
        'At identical parameters/query, delta_SUM=N_visible*delta_MEAN. This tests extensive scaling and its optimization consequences; it does not establish increased information or bandwidth at fixed N.',
        'SUM/MEAN change branch-gradient scale, global clipping and relative branch/LoRA optimization. Identical nominal learning rates do not remove these confounds.',
        'Both arms access the same independently encoded visual memory. Their contrast cannot isolate memory source, contextual corruption, or the causal effect of independent encoding.',
        'The adapter uses established conditional Deep Sets and visual-memory ingredients, with particularly close PVM precedent. No new attention-family or broad persistent-memory novelty is claimed.',
        'Zero-read centering does not guarantee zero mean distractor messages. Distractor contributions can accumulate with length; query states remain contextual.',
        'Primary SUM-minus-MEAN familiar OOD >=5pp and N16 loss <=5pp must hold in EACH seed. Pooled effects cannot rescue a failed seed.',
        'Bootstrap resamples10000 whole anchor families with seed20260916, retaining both lengths and fixed seeds. Two seeds do not establish seed-population variance.',
        'Both arms use all1620 identical ordered presentations of1540 scenes, including ten deterministic N8/K8 scenes repeated across blocks. This is not a data-diversity contrast.',
        'All invalid outputs stay incorrect. MAE/bias are conditional on parsing; two invalid answers are not invariant predictions. K9 and K10–16 answer support are reported separately.',
        'Training initial hashes, first-step gradient hashes and cumulative Adam counters diagnose reproducibility; gradient tensors/momentum states were not retained for independent numerical replay.',
        'Initial absolute-RMS and subsequent raw native-reference software gates failed; those failures remain recorded and are not accepted by this report. The final registered gate centers cached/full logit differences, applying maximum <=min(0.25,1.5*OFF-centered maximum+0.0625), RMS <=min(0.05,1.5*OFF-centered RMS+0.005), total variation <=0.01 and identical enabled raw top-1. Raw common shifts/KL remain diagnostics. Five checks use two original and two previously unused software scenes plus an exact A/B/C/D/A reset (20 visual forwards); OFF references share the enabled first token. This bounded two-token software check is not exact numerical parity, long-reasoning validation or efficacy evidence.',
        'Exploratory training was explicitly accepted under schema2 while the strict numerical gate remained FAILED. The native NF4/bf16/SDPA/bitsandbytes backend was unchanged. Completion/state-integrity evidence and kernel localization support this engineering decision; they do not make the failed1%TV criterion pass. All four dev-selected checkpoints receive the same post-training numerical audit; failures remain reported and do not select/drop a model or main prediction.',
        'First-prefill frame-message and residual norms are detached descriptive diagnostics, not semantic evidence labels or causal attribution. The prototype is batch1, images only, unpadded greedy caching, without beams or mid-cache image insertion.',
        'Recorded model_seconds includes enabled GPU diagnostic capture; subsequent CPU tensor transfer, serialization and hashing are excluded. SUM/MEAN share instrumentation; cross-version timing against bare inference is not strictly matched. Native attention norm is measured before branch addition, not over the full residual stream.',
        'No intermediate reasoning tokens, frame-label supervision or external tally are used. This direct-answer experiment does not establish composition with reasoning.',
    ]
    analysis=dict(criteria=decision_criteria(contrasts),conditions=results,runs=runs,
        sum_minus_mean=contrasts,pooled_sum_minus_mean=pooled,
        pooled_conditions={c:describe_rows([r for seed in SEEDS for r in predictions[c][seed]]) for c in CONDITIONS},
        paired_training_audit=paired,runtime_audit=runtime,schedule_audit=data['schedule_audit'],implementation_gates=gates,post_training_checkpoint_audits=post_audits,
        gradient_audits={c:{s:ledgers[c][s]['gradients'] for s in SEEDS} for c in CONDITIONS},
        bootstrap=dict(draws=10000,seed=BOOTSTRAP_SEED,unit='shared whole anchor; all lengths and fixed seeds retained'),
        answer_tokenization={str(k):tokenizer(str(k),add_special_tokens=False).input_ids for k in range(17)},
        training_source_sha256=expected_hashes,analysis_source_sha256=frozen_report,analysis_job=os.environ['SLURM_JOB_ID'],notes=notes)
    write(args.root/'analysis.json',analysis)
    code=args.root/'analysis_code';code.mkdir(exist_ok=True)
    for name in REPORT_SOURCES:(code/name.replace('/','_')).write_bytes((REPO/name).read_bytes())
    render(args.root,analysis,np,plt)
    (args.root/'INDEX.md').write_text('# V5 analysis artifacts\n\n- [Report](REPORT.md)\n- [Verified analysis](analysis.json)\n- [Figure PNG](comparison.png)\n- [Figure PDF](comparison.pdf)\n')


if __name__=='__main__':
    main()

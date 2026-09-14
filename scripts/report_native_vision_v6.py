"""Independent V6 analysis: aligned training-only local KD versus detached control.

New V6 protocol on fresh sequence evaluations. Native prediction, training-order,
checkpoint and numerical arithmetic validators reuse frozen earlier implementations.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean
import sys
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts.report_native_vision_v2 import CELLS,compare,extension,verify_anchors,summarize
from scripts.report_native_vision_v3 import compare_pooled,describe_rows
from scripts.report_native_vision_v4 import (CELL_HISTOGRAMS,DATA_ROOT,RUNTIME_FIELDS,
    audit_shared_runtime,close,digest,finite,need,object_sha,read,valid_sha,
    verify_data as verify_v4_data,verify_metrics,verify_predictions,write)
from scripts.report_native_vision_v5 import (BRANCH_SHAPES,PARAMETERS,
    architecture as v5_architecture,verify_presentations,verify_gradients,
    verify_prefill_diagnostics,paired_training_audit,audit_native_reference_observation)
from scripts.report_native_vision_v6_auxiliary import verify_auxiliary
ROOT=REPO/'outputs/native_aggregation_vlm/v6'
DATA_BASE=Path('/mnt/data/gabriele/gnn_transformer')
FRESH=DATA_BASE/'v6_fresh'
CHECKPOINT_ROOT=Path('/mnt/ckpts/gabriele/gnn_transformer/native_aggregation_vlm_v6')
CONDITIONS=('control','aligned')
SEEDS=(6,7)
BOOTSTRAP_SEED=20260921
CONTRASTS={'familiar_ood':CELLS[1:3],'N16':CELLS[:1],'unseen_count':CELLS[3:]}
REPORT_SOURCES=('scripts/report_native_vision_v6.py','scripts/report_native_vision_v5.py',
    'scripts/report_native_vision_v4.py','scripts/report_native_vision_v3.py',
    'scripts/report_native_vision_v2.py','slurm/native_aggregation_vision_v6_report.sbatch',
    'scripts/report_native_vision_v6_auxiliary.py','scripts/report_native_vision_v6_post.py',
    'scripts/report_native_vision_v5_release.py','scripts/report_native_vision_v6_teacher_audit.py',
    'scripts/probe_native_vision_v5_local_readability.py',
    'scripts/stage_native_vision_v6_teacher.py','scripts/cache_native_vision_v6_teacher.py',
    'scripts/audit_native_vision_v6_checkpoint.py',
    'slurm/native_aggregation_vision_v6_audit.sbatch','slurm/native_aggregation_vision_v6_audit_check.sbatch')


def architecture(condition,teacher_sha):
    need(condition in CONDITIONS and valid_sha(teacher_sha),'Invalid architecture identity')
    value=v5_architecture('sum')
    value['protocol']='vision_v6_local_teacher'
    value['local_distillation']=dict(condition=condition,rank=96,classes=3,parameters=291,
        temperature=1.,weight=1.,detach_messages=condition=='control',
        teacher_index_sha256=teacher_sha,query='last_original_prompt_token',deployed=False)
    return value


def criteria(contrasts,conditions):
    primary={seed:contrasts[seed]['familiar_ood']['exact_gain']>=.05-1e-12
        and contrasts[seed]['N16']['exact_gain']>=-.05-1e-12 for seed in SEEDS}
    useful={seed:primary[seed] and conditions['aligned'][seed]['familiar_ood']['exact']>=.70-1e-12
        and conditions['aligned'][seed]['nonzero_familiar_ood']['exact']>=.60-1e-12 for seed in SEEDS}
    return dict(primary_each_seed=primary,primary_both_seeds=all(primary.values()),
        practical_each_seed=useful,practical_both_seeds=all(useful.values()),
        primary_definition='Aligned-control familiar OOD >=5pp and N16 loss <=5pp, EACH seed',
        practical_definition='Primary plus aligned familiar OOD >=70% and nonzero-K OOD >=60%, EACH seed')


def self_test():
    contrasts={s:{'familiar_ood':{'exact_gain':.05},'N16':{'exact_gain':-.05}} for s in SEEDS}
    conditions={'aligned':{s:{'familiar_ood':{'exact':.7},'nonzero_familiar_ood':{'exact':.6}} for s in SEEDS}}
    need(criteria(contrasts,conditions)['practical_both_seeds'],'Inclusive boundary failed')
    contrasts[7]['familiar_ood']['exact_gain']=.049
    need(not criteria(contrasts,conditions)['primary_both_seeds'],'One seed incorrectly rescued another')
    contrasts[7]['familiar_ood']['exact_gain']=.05
    conditions['aligned'][6]['nonzero_familiar_ood']['exact']=.59
    need(criteria(contrasts,conditions)['primary_both_seeds'] and not criteria(contrasts,conditions)['practical_both_seeds'],
         'Relative efficacy conflated with practical milestone')
    need(architecture('control','0'*64)['local_distillation']['parameters']==96*3+3,'Auxiliary parameter count differs')


def verify_run(run, condition, seed, expected_hashes, data, torch, tokenizer):
    config, summary = read(run / 'config.json'), read(run / 'summary.json')
    teacher_sha=digest(Path(config['teacher_index']))
    expected_architecture=architecture(condition,teacher_sha)
    need(config['code_sha256'] == expected_hashes and read(run / 'source_hashes.json') == expected_hashes, 'Frozen training source ledger differs')
    for name, sha in expected_hashes.items():
        need(digest(run / 'code' / name.replace('/', '_')) == sha, f'Training snapshot changed: {name}')
    required = dict(arm='independent', condition=condition, seed=seed, profile=False, eval_only=False,
        epochs=9, accumulation=4, layer_index=14, rank=96, lora_layers=4, lora_rank=8,
        lora_alpha=16.0, resize=392, lr=.001, lr_lora=.0001, max_grad_norm=1.0, max_steps=0,
        max_seq_tokens=16000, max_new_tokens=4, train_ns=[8, 16], dev_ns=[8, 16], eval_ns=[16, 32, 64],
        gold_max=16, eval_modes='all', limit_dev=36, limit_eval=108, limit_count=64,
        checkpoint=None, gradient_checkpointing=False, center_messages=False, data_seed=20260914, training_schedule_condition='refresh', model=expected_architecture['model'])
    need(all(config.get(k) == v for k, v in required.items()), 'Registered model/training configuration differs')
    need(not config.get('test_epochs') and config['architecture'] == expected_architecture, 'Architecture/test selection differs')
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
    need(saved['architecture'] == expected_architecture and saved['epoch'] == best['epoch']
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
    prefill_diagnostics=verify_prefill_diagnostics(run,rows,'sum',torch,directory_override=DATA_BASE/'v6_diagnostics'/run.name)
    auxiliary=verify_auxiliary(run,config,summary,history,ledgers,torch)
    for field,filename in (('test_manifest','staged_test_manifest.json'),('test_count_manifest','staged_test_count_manifest.json')):
        need(digest(Path(config[field]))==digest(run/filename)==data[field+'_sha256'],'Fresh evaluation binding differs')
    info = describe_rows(rows)
    info['auxiliary_audit']=auxiliary
    info['nonzero_familiar_ood']=summarize([r for r in rows if r['cell'] in CELLS[1:3] and r['gold']>0])
    info['prefill_diagnostics']=prefill_diagnostics
    info.update(runtime_provenance={key: config[key] for key in RUNTIME_FIELDS},
        parameters=summary['parameters'], gradients=gradients['summary'], selected_epoch=best['epoch'],
        selected_checkpoint=str(checkpoint), selected_checkpoint_sha256=digest(checkpoint),
        training=history, presentation_audit=ledgers['audit'], initial_parameter_sha256=initialization['combined_sha256'],
        manifest_sha256=config['manifest_sha256'], count_manifest_sha256=config['count_manifest_sha256'],
        schedule_sha256=config['schedule_sha256'], training_stage_audit_sha256=data['stage_audit_sha256'],
        fresh_stage_audit_sha256=data['fresh_stage_audit_sha256'],
        extensions={f'{family}_{lo}_to_{hi}': extension(rows, family, lo, hi)
                    for family, lo, hi in [('length', 16, 32), ('length', 32, 64), ('length', 16, 64), ('unseen_count', 32, 64)]},
        memory=dict(training_peak_allocated_bytes=max(e['peak_cuda_allocated_bytes'] for e in history),
                    training_peak_reserved_bytes=max(e['peak_cuda_reserved_bytes'] for e in history),
                    evaluation_peak_allocated_bytes=max(m['peak_cuda_allocated_bytes'] for m in summary['results'])))
    info['deterioration'] = {f'N16_to_N{n}': info['cells']['length_N16']['exact'] - info['cells'][f'length_N{n}']['exact']
                             for n in (32, 64)}
    info['diagnostic_artifact_sha256'] = {path.name: digest(path) for path in sorted(run.glob('*diagnostic*.json'))}
    return rows, info, ledgers


def verify_teacher(path):
    from scripts.stage_native_vision_v6_teacher import verify_plan
    from scripts.cache_native_vision_v6_teacher import probabilities,quality,quality_gate
    path=Path(path).resolve();index=read(path)
    need(path.is_relative_to(DATA_BASE/'v6_local_teacher') and index['schema_version']==1,
         'Teacher cache outside registered location/schema')
    need(digest(path)==path.with_suffix('.sha256').read_text().strip(),'Teacher sidecar changed')
    plan=verify_plan(Path(index['plan_file']))
    need(digest(index['plan_file'])==index['plan_sha256'] and index['passed_quality_gate'] is True,
         'Teacher not eligible or plan differs')
    for key in ('source_sha256','model','processor','runtime','source_files'):
        need(index[key]==plan[key],f'Teacher {key} identity differs')
    need(index['scenes']=={sid:row['pair_ids'] for sid,row in plan['scenes'].items()},'Teacher scene correspondence differs')
    need(set(index['targets'])==set(plan['pairs']) and len(index['targets'])==9980,'Incomplete teacher cache')
    fields={'probabilities','image_sha256','question','question_sha256','logit0','logit1',
            'log_normalizer','layout_id','input_ids_sha256'}
    for pid,row in index['targets'].items():
        pair=plan['pairs'][pid]
        need(set(row)==fields,'Unexpected target fields: audit labels must not enter KD targets')
        need(all(row[k]==pair[k] for k in ('image_sha256','question','layout_id')),'Teacher pair identity differs')
        expected=probabilities(row['logit0'],row['logit1'],row['log_normalizer'])
        need(all(close(a,b) for a,b in zip(row['probabilities'],expected)),'Teacher probabilities differ from raw logits')
        need(row['input_ids_sha256']==plan['layouts'][row['layout_id']]['input_ids_sha256'],
             'Teacher prompt token identity differs')
    from scripts.report_native_vision_v6_teacher_audit import verify_shards
    raw_audit=verify_shards(index,plan)
    labels=read(plan['audit_file'])['labels']
    measured=quality(index['targets'],labels,'unique_scene_occurrences')
    gate=quality_gate(measured)
    saved_gate=index['quality_gate']
    need(gate['passed'] is saved_gate['passed'] is True and gate['criteria']==saved_gate['criteria']
         and gate['thresholds']==saved_gate['thresholds']
         and gate['measurements']['conditional_balanced_accuracy']==saved_gate['measurements']['conditional_balanced_accuracy']
         and gate['measurements']['category_accuracy']==saved_gate['measurements']['category_accuracy']
         and math.isclose(gate['measurements']['mean_numeric_mass'],saved_gate['measurements']['mean_numeric_mass'],
                          rel_tol=1e-12,abs_tol=1e-12),
         'Teacher quality gate was not independently reproduced')
    return dict(path=str(path),sha256=digest(path),plan_sha256=index['plan_sha256'],
        quality=measured,quality_gate=gate,unique_pairs=9980,training_frames=18800,
        source_sha256=index['source_sha256'],shard_provenance=index['shard_provenance'],raw_shard_audit=raw_audit)


def verify_fresh_data():
    from scripts.probe_native_vision_v5_local_readability import semantic_labels
    data=verify_v4_data()
    inventory=read(FRESH/'exclusion_inventory.json');audit=read(FRESH/'stage_audit.json')
    need(audit['inventory_sha256']==digest(FRESH/'exclusion_inventory.json'),'Fresh exclusion inventory changed')
    for flag in ('all_qa_hashes_and_gold_recounts_passed','all_image_hashes_and_dimensions_passed',
                 'all_fresh_content_exclusions_passed','all_pair_extensions_passed'):
        need(audit[flag] is True,f'Fresh stage did not pass {flag}')
    need(audit['unique_samples']==audit['unique_content']==452,'Wrong fresh stage size')
    for entry in inventory['entries']:
        need(digest(entry['path'])==entry['sha256'],'Prior exclusion manifest changed')
    for path,sha in inventory['protected_source_sha256'].items():
        need(digest(path)==sha,'Protected source or V4 schedule changed')
    excluded=set(inventory['excluded_content_sha256'])
    need(object_sha(sorted(excluded))==inventory['excluded_content_set_sha256'],'Exclusion set hash differs')
    sources=dict(data['sources']);fresh=[]
    for family,filename,label in (('length','main_manifest.json','main'),('unseen_count','count_manifest.json','count')):
        manifest=read(FRESH/filename)
        need(manifest['schema_version']==1 and Path(manifest['dataset_root']).resolve()==FRESH,'Wrong fresh manifest/root')
        need(digest(FRESH/filename)==audit['manifest_sha256'][label],'Fresh manifest hash differs')
        need(Path(manifest['stage_audit_file']).resolve()==FRESH/'stage_audit.json','Wrong final stage audit path')
        need(digest(manifest['audit_file'])==manifest['audit_sha256'],'Fresh semantic audit changed')
        for key,cell in manifest['splits'].items():
            need(key.startswith('test_N'),'Fresh evaluation manifest contains non-test data')
            name=key.replace('test_',family+'_',1);need(name in CELLS,'Unexpected fresh cell')
            rows=cell['samples'];need(Counter(r['gold'] for r in rows)==Counter(CELL_HISTOGRAMS[name]),'Fresh label law changed')
            for row in rows:
                path=Path(row['path']);n=int(key.split('_N')[1])
                need(path.resolve().parent==FRESH/'mmred_vfiltered'/f'seq_len_{n}'/'test'
                     and row['split']=='test' and row['sid']==path.name and row['n_frames']==n,'Fresh path/split/order differs')
                need(row['content_sha256'] not in excluded,'Fresh complete context overlaps prior inventory')
                semantic_labels(row)
                need(len(row['image_files'])==n,'Fresh image count differs')
                for i,img in enumerate(row['image_files']):
                    need(Path(img['path'])==path/f'{i:03d}.png' and digest(img['path'])==img['sha256'],
                         'Fresh image order/hash differs')
            sources[name]=rows;fresh.extend(rows)
    need(len(fresh)==len({r['sid'] for r in fresh})==len({r['content_sha256'] for r in fresh})==452,
         'Fresh context identities are duplicated')
    need(set(sources)==set(CELL_HISTOGRAMS),'Unexpected combined data cells')
    data['sources']=sources
    data.update(test_manifest_sha256=digest(FRESH/'main_manifest.json'),
        test_count_manifest_sha256=digest(FRESH/'count_manifest.json'),
        fresh_stage_audit_sha256=digest(FRESH/'stage_audit.json'),
        fresh_stage_audit=audit,exclusion_inventory=inventory)
    return data


def render(root,analysis,np,plt):
    lines=['# Native aggregation with training-only local distillation (V6)','',
        'Fresh sequence evaluation; identical native SUM inference, training scenes and decoding. The auxiliary head is absent at inference.','',
        '| Training | Seed | N16 | N32 | N64 | Familiar OOD | Nonzero OOD | Unseen K9–16 | Selected block |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def exact(value):return f"{value['correct']}/{value['n']} ({100*value['exact']:.1f}%)"
    for condition in CONDITIONS:
        for seed in SEEDS:
            r=analysis['conditions'][condition][seed]
            values=[exact(r['cells'][c]) for c in CELLS[:3]]
            values += [exact(r[k]) for k in ('familiar_ood','nonzero_familiar_ood','unseen_count')]
            lines.append('| '+' | '.join([condition,str(seed),*values,str(r['selected_epoch'])])+' |')
    lines += ['',f"Both-seed relative efficacy criterion: **{analysis['criteria']['primary_both_seeds']}**.",
              f"Both-seed practical milestone: **{analysis['criteria']['practical_both_seeds']}**.",'',
              '| Aligned minus control | Familiar OOD difference | Paired-anchor 95% interval | N16 difference |',
              '|---|---:|---:|---:|']
    for seed in SEEDS:
        r=analysis['aligned_minus_control'][seed];a=r['familiar_ood']
        lines.append(f"| seed{seed} | {a['exact_gain']*100:+.2f}pp | [{a['bootstrap95'][0]*100:+.2f},{a['bootstrap95'][1]*100:+.2f}]pp | {r['N16']['exact_gain']*100:+.2f}pp |")
    lines += ['',analysis['criteria']['primary_definition']+'.',analysis['criteria']['practical_definition']+'.','',
        'Every invalid output remains incorrect. MAE and bias condition on parsed answers; full denominators, per-K results and length transitions are in analysis.json.',
        'This tests the auxiliary training objective. Without a permuted-correspondence control, it does not isolate alignment-specific learning from other effects of that objective. It does not establish reasoning composition or a new attention family.',
        'Teacher judgments and local supervision are additional training resources. Local rendered atoms recur in this finite compositional benchmark; fresh complete sequences do not establish unseen perception or natural-video generalization.',
        'The unchanged native numerical thresholds are descriptive post-training checks. Earlier V5 engineering failures are preserved; no full/cache or long-reasoning equivalence claim follows.','']
    write(root/'analysis.json',analysis)
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for condition,color in (('control','#6b7280'),('aligned','#087e8b')):
        for seed,style in ((6,'-'),(7,'--')):
            r=analysis['conditions'][condition][seed]
            axes[0].plot([16,32,64],[r['cells'][f'length_N{n}']['exact'] for n in (16,32,64)],
                marker='o',linestyle=style,color=color,label=f'{condition} seed{seed}')
            axes[1].plot([32,64],[r['cells'][f'unseen_count_N{n}']['exact'] for n in (32,64)],
                marker='o',linestyle=style,color=color)
    for axis,title,ticks in zip(axes,('Familiar counts 0–8','Unseen counts 9–16'),([16,32,64],[32,64])):
        axis.set(ylim=(-.02,1.02),xticks=ticks,xlabel='Frames',ylabel='Native exact answer',title=title)
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('V6: local teacher training, unchanged native SUM inference')
    fig.tight_layout();fig.savefig(root/'comparison.png',dpi=180);fig.savefig(root/'comparison.pdf');plt.close(fig)
    (root/'INDEX.md').write_text('# V6 verified native aggregation results\n\n[Report](REPORT.md) · [Analysis](analysis.json) · [Figure](comparison.png) · [Execution/cost](execution.json).\n')


def artifact(entry):
    path=Path(entry['path']).resolve()
    need(digest(path)==entry['sha256'],'Release artifact hash differs')
    return path,read(path)


def runtime_projection(summary):
    history=summary['training']
    training=sum(e['seconds'] for e in history)/sum(e['n'] for e in history)*1620
    dev=9*sum(max(e['dev'][i]['mean_total_seconds'] for e in history)*36 for i in range(2))
    evaluation=sum(m['mean_total_seconds']*(108 if m['cell'].startswith('length') else 64) for m in summary['results'])
    return dict(projected_main_seconds=1.25*(training+dev+evaluation)+120,
        training_seconds=training,dev_seconds=dev,evaluation_seconds=evaluation,margin=1.25,fixed_seconds=120,
        formula='1620 mean-profile training presentations + nine full72-item dev sweeps using max per-cell profile time + full452-item eval; all times multiplied by1.25 plus120s')


def verify_release(root,data):
    gate_path=root/'implementation_gates.json';gate=read(gate_path)
    need(gate['schema_version']==1 and gate['native_backend']=='unchanged_V5_native_nf4_bfloat16_sdpa',
         'Wrong V6 implementation release')
    ledger_path,ledger=artifact(gate['source_ledger'])
    core_path,core=artifact(gate['cpu_core'])
    need(core['passed'] is True and core['source_sha256']==ledger,'CPU release sources differ')
    need(core['unit_tests_passed'] and core['local_evidence_tests_passed'] and core['schedule_checks_passed'],
         'CPU checks did not pass')
    for name,sha in ledger.items():need(digest(REPO/name)==sha,'Frozen main source changed')
    teacher_path,_=artifact(gate['teacher_index']);teacher=verify_teacher(teacher_path)
    stage_path,stage=artifact(gate['fresh_stage_audit'])
    need(stage_path==FRESH/'stage_audit.json' and stage==data['fresh_stage_audit'],'Fresh staging release differs')
    profiles={}
    for condition in CONDITIONS:
        path,summary=artifact(gate['profiles'][condition]);run=path.parent;config=read(run/'config.json')
        need(config['condition']==condition and config['seed']==6 and config['profile'] is True,
             'Wrong software profile')
        need(config['code_sha256']==ledger==read(run/'source_hashes.json'),'Profile code differs from main release')
        need(config['architecture']==architecture(condition,teacher['sha256']),'Profile architecture/teacher differs')
        checks=read(run/'profile_checks.json')
        need(checks['passed'] is True,'Software profile checks failed')
        names=('zero_initialization','live_messages','observer_removal','gradient_routing',
               'rng_preservation','finite_updates_and_selected_restore')
        need(all(checks[k]['passed'] is True for k in names),'Incomplete computational profile checks')
        need(checks['observer_removal']['exact_native_logits'] is True and checks['rng_preservation']['main_rng_unchanged'] is True,
             'Observer removal or RNG preservation failed')
        route=checks['gradient_routing']['connected_native_indices']
        need(route==([] if condition=='control' else [0,1,2,3]),'Profile gradient route differs')
        need(len(summary['training'])==2 and summary['training'][-1]['step']==2,'Profile training incomplete')
        need(len(read(run/'predictions.json'))==10 and summary['auxiliary_head_removed_from_inference'] is True,
             'Software predictions missing or auxiliary retained at inference')
        bound=gate['runtime_projection'][condition]
        need(bound==runtime_projection(summary),'Stored projection differs from profile timing arithmetic')
        need(bound['projected_main_seconds']<=3000 and bound['margin']==1.25 and bound['fixed_seconds']==120,
             'Profile did not support the registered main time cap')
        profiles[condition]=dict(path=str(path),sha256=digest(path),checks=checks,runtime_projection=bound)
    need(gate['budget_gpu_hours']==5,'Wrong V6 resource cap')
    return ledger,dict(path=str(gate_path),sha256=digest(gate_path),teacher=teacher,profiles=profiles,
        cpu_source_ledger=str(ledger_path),cpu_core=str(core_path),fresh_stage_audit_sha256=digest(stage_path),
        inference='Existing V5 SUM operator; auxiliary teacher/head are training-only',
        numerical_policy=gate['numerical_policy'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--check-data',action='store_true')
    parser.add_argument('--check-gates',action='store_true')
    args=parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu',
         'All V6 reporting/tests require Slurm CPU')
    self_test()
    if args.self_test:
        from scripts.report_native_vision_v6_auxiliary import self_test as auxiliary_self_test
        import torch
        auxiliary_self_test(torch)
        print('PASS: V6 relative/practical criteria and auxiliary audit self-tests');return
    data=verify_fresh_data()
    if args.check_data:
        print('PASS: unchanged V4 training/dev and all452 fresh QA/image/exclusion records');return
    expected,release=verify_release(args.root,data)
    if args.check_gates:
        print('PASS: V6 CPU, teacher quality, fresh data and both computational profiles');return
    frozen=read(args.root/'analysis_source_hashes.json')
    need(frozen=={name:digest(REPO/name) for name in REPORT_SOURCES},'Analysis source changed after freeze')
    import torch
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from transformers import AutoTokenizer
    from scripts.report_native_vision_v6_post import verify_post_training_audits
    tokenizer=AutoTokenizer.from_pretrained(v5_architecture('sum')['model'],local_files_only=True,use_fast=False)
    results,predictions,ledgers,runs={},{},{},{}
    for condition in CONDITIONS:
        results[condition],predictions[condition],ledgers[condition],runs[condition]={},{},{},{}
        for seed in SEEDS:
            candidates=list((args.root/'main'/condition/f'seed{seed}').glob('*/summary.json'))
            need(len(candidates)==1,'Each registered arm/seed must have exactly one complete main')
            run=candidates[0].parent
            rows,info,ledger=verify_run(run,condition,seed,expected,data,torch,tokenizer)
            predictions[condition][seed],results[condition][seed],ledgers[condition][seed],runs[condition][seed]=rows,info,ledger,str(run)
    post=verify_post_training_audits(args.root,expected,results,runs,sys.modules[__name__])
    paired={}
    for seed in SEEDS:
        pair=paired_training_audit(ledgers['aligned'][seed],ledgers['control'][seed])
        difference=pair['first_optimizer_batch_loss_difference']
        difference['mean_aligned_minus_control']=difference.pop('mean_sum_minus_mean')
        need(read(Path(runs['aligned'][seed])/'auxiliary_initialization.json')==
             read(Path(runs['control'][seed])/'auxiliary_initialization.json'),'Paired auxiliary initialization differs')
        pair['note']='Native and auxiliary initialization/order match. Aligned KD changes reader gradients and resulting common clipping/optimization; bitwise training equality is not claimed.'
        paired[seed]=pair
    runtime=audit_shared_runtime([results[c][s]['runtime_provenance'] for c in CONDITIONS for s in SEEDS])
    contrasts={s:{name:compare(predictions['aligned'][s],predictions['control'][s],cells,np,
                    np.random.default_rng(BOOTSTRAP_SEED)) for name,cells in CONTRASTS.items()} for s in SEEDS}
    pooled={name:compare_pooled(predictions['aligned'],predictions['control'],cells,np,
                    np.random.default_rng(BOOTSTRAP_SEED)) for name,cells in CONTRASTS.items()}
    analysis=dict(criteria=criteria(contrasts,results),conditions=results,runs=runs,
        aligned_minus_control=contrasts,pooled_aligned_minus_control=pooled,
        paired_training_audit=paired,runtime_audit=runtime,implementation_gates=release,
        post_training_checkpoint_audits=post,
        fresh_data=dict(stage_audit_sha256=data['fresh_stage_audit_sha256'],
                        exclusion_inventory_sha256=digest(FRESH/'exclusion_inventory.json'),
                        test_manifest_sha256=data['test_manifest_sha256'],
                        test_count_manifest_sha256=data['test_count_manifest_sha256']),
        training_source_sha256=expected,analysis_source_sha256=frozen,
        bootstrap=dict(draws=10000,seed=BOOTSTRAP_SEED,unit='whole anchor; paired lengths and fixed seeds retained'),
        analysis_job=os.environ['SLURM_JOB_ID'])
    render(args.root,analysis,np,plt)
    print(json.dumps(dict(criteria=analysis['criteria'],report=str(args.root/'REPORT.md'))))


if __name__=='__main__':main()

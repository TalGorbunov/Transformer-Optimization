"""Independent V6 auxiliary-training ledger audit; no model forward or selection.

Torch is supplied by the CPU Slurm report caller. This verifies saved arithmetic,
correspondence, private initialization/optimizer metadata and checkpoint tensors.
Live per-image student logits and Adam moments were not saved: their trajectories
cannot be independently reconstructed from these summaries.
"""
from __future__ import annotations
from collections import Counter
import copy
import hashlib
import math
import os
from pathlib import Path
from statistics import mean

from scripts.report_native_vision_v4 import digest, finite, need, object_sha, read, valid_sha

REPO = Path(__file__).resolve().parents[1]
DATA_BASE = Path('/mnt/data/gabriele/gnn_transformer')
TEACHER = DATA_BASE / 'v6_local_teacher/teacher_cache.json'
CKPTS = Path('/mnt/ckpts/gabriele/gnn_transformer/native_aggregation_vlm_v6')


def close(left, right):
    return finite(left) and finite(right) and math.isclose(left, right, rel_tol=3e-6, abs_tol=1e-6)


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def local_record(row, source, pair_ids):
    """Check one observed training presentation against its canonical teacher order."""
    count, kl, total = (row[key] for key in ('answer_token_ce', 'auxiliary_kl', 'total_loss'))
    need(finite(count) and count >= 0 and finite(kl) and kl >= -1e-5 and finite(total),
         'Invalid count CE/auxiliary KL/total loss')
    need(close(total, count + kl), 'Total loss is not native CE plus coefficient-one auxiliary KL')
    diagnostic = row['local_evidence']
    n, length = source['n_frames'], row['prompt_tokens']
    expected = dict(prompt_length=length, total_length=length + len(row['target_ids']) - 1,
                    expected_images=n, branch_calls=1, query_position=length-1, all_images_visible=True)
    need(type(length) is int and length > 0 and len(row['target_ids']) >= 2,
         'Malformed prompt/answer-EOS training layout')
    need(all(diagnostic.get(k) == value for k, value in expected.items()),
         'Auxiliary query used an answer token, nonoriginal index, or wrong image visibility')
    need(diagnostic['all_images_visible'] is True and diagnostic['branch_calls'] == 1,
         'Expected exactly one native branch observation with all images already complete')
    need(len(pair_ids) == n and diagnostic['teacher_pair_ids_sha256'] == object_sha(pair_ids),
         'Observed teacher image order/hash differs')
    norm = diagnostic['message_mean_norm']
    agreement = diagnostic['student_teacher_argmax_agreement']
    probabilities = diagnostic['student_mean_probabilities']
    need(finite(norm) and norm >= 0 and finite(agreement) and 0 <= agreement <= 1,
         'Nonfinite local-message norm or student-teacher agreement')
    need(close(agreement*n, round(agreement*n)), 'Agreement denominator is not the image count')
    need(len(probabilities) == 3 and all(finite(p) and 0 <= p <= 1 for p in probabilities)
         and abs(sum(probabilities)-1) <= 2e-6, 'Invalid mean local-student probabilities')
    return dict(count_ce=count, auxiliary_kl=kl, total_loss=total, agreement=agreement,
                agreeing_images=round(agreement*n), images=n, message_mean_norm=norm,
                student_mean_probabilities=probabilities)


def gradient_rows(rows):
    need(len(rows) == 405 and [r['step'] for r in rows] == list(range(1,406)),
         'Auxiliary gradient ledger must contain all405 private optimizer updates')
    for row in rows:
        norm = row['combined_preclip_norm']
        need(finite(norm) and norm >= 0 and row['parameters'] == 291 and row['tensors'] == 2,
             'Auxiliary gradient dimensions or norm differs')
        need(type(row['clipped']) is bool and row['clipped'] == (norm > 1),
             'Auxiliary clip flag differs from its private clip1 threshold')
    return dict(updates=405, tensors=2, parameters=291,
                clipped_updates=sum(r['clipped'] for r in rows),
                clipped_fraction=mean(r['clipped'] for r in rows),
                mean_preclip_norm=mean(r['combined_preclip_norm'] for r in rows),
                min_preclip_norm=min(r['combined_preclip_norm'] for r in rows),
                max_preclip_norm=max(r['combined_preclip_norm'] for r in rows))


def gradient_route(condition, value):
    expected = [] if condition == 'control' else [0,1,2,3]
    need(condition in ('control','aligned') and value['condition'] == condition,
         'Gradient-routing arm differs')
    need(value['native_parameter_order'] == 'branch5_then_upper_lora32'
         and value['connected_native_indices'] == expected, 'Auxiliary reached the wrong native tensors')
    norms = value['native_gradient_norms']
    need(len(norms) == 37 and [i for i,x in enumerate(norms) if x is not None] == expected,
         'Gradient-routing norm coverage differs')
    need(all(norms[i] is not None and finite(norms[i]) and norms[i] >= 0 for i in expected),
         'Nonfinite auxiliary native gradient')
    need(value['detached_control_native_gradient_is_structurally_zero'] is (condition == 'control'),
         'Detached-control gradient claim contradicts the recorded graph')
    return dict(connected_native_indices=expected, native_gradient_norms=norms,
                native_auxiliary_graph_absent=condition=='control',
                scope='First presentation autograd connectivity; CPU test checks exact control count-gradient identity.')


def initialization(record, seed, torch):
    need(seed in (6,7), 'Unregistered initialization seed')
    expected = dict(seed=seed, stream_seed=seed+1000003, weight_std=.01, bias_zero=True,
                    parameters=291, main_rng_unchanged=True)
    need(all(record.get(k) == value for k,value in expected.items()), 'Auxiliary initialization settings differ')
    before, after = record['main_rng_before'], record['main_rng_after']
    need(before == after and set(before) == {'cpu','cuda'} and valid_sha(before['cpu'])
         and isinstance(before['cuda'],list) and len(before['cuda']) == 1
         and all(valid_sha(value) for value in before['cuda']), 'Main RNG changed or RNG coverage differs')
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed+1000003)
    expected_tensors = dict(weight=torch.randn((3,96), generator=generator, dtype=torch.float32)*.01,
                            bias=torch.zeros(3,dtype=torch.float32))
    hashes = {name:tensor_sha(tensor) for name,tensor in expected_tensors.items()}
    need(record['state_sha256'] == hashes, 'Auxiliary initialization differs from its private Gaussian RNG')
    return dict(parameters=291, tensors=2, independent_stream_seed=seed+1000003,
                state_sha256=hashes, logged_main_rng_unchanged=True,
                expected_cpu_generator_reproduced=True)


def teacher_bindings(run, config, summary):
    path = Path(config['teacher_index']).resolve()
    need(path == TEACHER, 'Teacher index is not the canonical passing cache')
    sha = digest(path)
    need(digest(run/'teacher_cache.json') == sha, 'Run copied a different teacher cache')
    cache = read(run/'teacher_cache.json')
    plan_path = Path(cache['plan_file']).resolve()
    need(plan_path.is_relative_to(DATA_BASE/'v6_local_teacher') and digest(plan_path) == cache['plan_sha256']
         == digest(run/'teacher_plan.json'), 'Teacher plan/source-copy hash differs')
    plan = read(run/'teacher_plan.json')
    need(cache['schema_version'] == 1 and cache['passed_quality_gate'] is True
         and cache['quality_gate']['passed'] is True, 'Teacher failed its prospective eligibility gate')
    need(len(cache['targets']) == 9980 and len(cache['scenes']) == 1540,
         'Teacher targets or training-scene coverage is incomplete')
    binding = dict(index_path=str(path), index_sha256=sha, plan_path=str(plan_path),
                   plan_sha256=cache['plan_sha256'], targets=9980, scenes=1540,
                   original_frame_occurrences=18800, passed_quality_gate=True)
    need(config['teacher_binding'] == summary['teacher_binding'] == read(run/'teacher_binding.json') == binding,
         'Run/config/summary teacher provenance differs')
    for key in ('model','runtime','processor','image_processor_settings','resize','quantization',
                'attention','temperature','source_files','source_sha256'):
        need(cache[key] == plan[key], f'Teacher cache/plan {key} differs')
    need(config['teacher_runtime'] == cache['runtime'] and config['teacher_processor'] == cache['processor']
         and config['frozen_model_identity'] == cache['model'], 'Observed native model/runtime/processor differs from teacher')
    need(config['torch_version'] == cache['runtime']['torch_version']
         and config['transformers_version'] == cache['runtime']['transformers_version']
         and config['image_processor_settings'] == cache['image_processor_settings'], 'Native and teacher runtime settings differ')
    need(cache['resize'] == config['resize'] == 392 and cache['quantization'] == 'nf4_double_bf16'
         and cache['attention'] == 'sdpa' and cache['temperature'] == 1, 'Teacher distribution/preprocessing policy differs')
    for key in ('training_manifest','training_schedule'):
        value=cache['source_files'][key]
        expected=DATA_BASE/'v4_diversity'/('main_manifest.json' if key=='training_manifest' else 'schedule.json')
        need(Path(value['path']).resolve() == expected and digest(expected) == value['sha256'],
             'Teacher uses a different original training manifest/schedule')
    for name, expected_sha in cache['source_sha256'].items():
        source=(REPO/name).resolve()
        need(source.is_relative_to(REPO) and digest(source) == expected_sha,
             f'Teacher generation source changed: {name}')
    for name in ('gnnformer/runtime.py','gnnformer/data.py'):
        need(config['code_sha256'][name] == cache['source_sha256'][name], 'Teacher/native source identity differs')
    selected = read(run/'staged_manifest.json')
    rows = [row for n in (8,16) for row in selected['splits'][f'train_N{n}']['samples']]
    training = {row['sid']:row for row in rows}
    need(len(training) == len(rows) == 1540 and set(training) == set(cache['scenes']) == set(plan['scenes']),
         'Teacher contains absent, duplicate or nontraining scene identities')
    occurrence_count, used = 0, set()
    for sid, source in training.items():
        scene, pair_ids = plan['scenes'][sid], cache['scenes'][sid]
        need(pair_ids == scene['pair_ids'] and len(pair_ids) == source['n_frames'] == len(source['image_files']),
             'Teacher image ordering differs from original scene')
        need(all(scene[k] == source[k] for k in ('path','qa_sha256','question','split','n_frames'))
             and scene['split'] == 'train', 'Teacher scene metadata includes a different/nontraining context')
        for pair_id, image in zip(pair_ids,source['image_files']):
            target=cache['targets'][pair_id]
            need(pair_id == object_sha([image['sha256'],source['question']])
                 and target['image_sha256'] == image['sha256']
                 and target['question'] == source['question']
                 and target['question_sha256'] == object_sha(source['question']),
                 'Teacher probability bound to wrong image/question pair')
            p=target['probabilities']
            need(len(p)==3 and all(finite(x) and 0<=x<=1 for x in p) and abs(sum(p)-1)<=2e-6,
                 'Teacher probabilities are not a valid complete three-way distribution')
            occurrence_count += 1; used.add(pair_id)
    need(occurrence_count == 18800 and used == set(cache['targets']), 'Teacher coverage/occurrence denominator differs')
    return cache, training, binding


def checkpoint_head(path, config, epoch, torch):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    need(saved['epoch'] == epoch['epoch'] and saved['step'] == epoch['step']
         and saved['dev'] == epoch['dev'] and saved['architecture'] == config['architecture'],
         'Auxiliary checkpoint does not accompany the same selected native epoch')
    for key in ('run_id','condition','seed','architecture','teacher_binding','local_distillation',
                'auxiliary_initialization','parameters','code_sha256'):
        need(saved['config'][key] == config[key], f'Auxiliary checkpoint config differs: {key}')
    expected=dict(parameters=291,optimizer_step=epoch['step'],lr=.001,weight_decay=0.0,
                  max_grad_norm=1.0,detached=config['condition']=='control')
    need(saved['auxiliary_training'] == expected, 'Saved auxiliary optimizer/settings metadata differs')
    values=saved['auxiliary_head']
    need(set(values)=={'weight','bias'}, 'Saved auxiliary head has unexpected tensors')
    for name,shape in (('weight',(3,96)),('bias',(3,))):
        value=values[name]
        need(tuple(value.shape)==shape and value.dtype==torch.float32 and bool(torch.isfinite(value).all()),
             'Auxiliary checkpoint tensor shape/dtype/finite check failed')
    need(sum(value.numel() for value in values.values())==291, 'Saved auxiliary head parameter count differs')
    need(not any('auxiliary' in name for name in saved['branch'])
         and not any('auxiliary' in name for name in saved['lora']), 'Auxiliary head entered deployed native parameter groups')
    result=dict(path=str(path),sha256=digest(path),epoch=epoch['epoch'],step=epoch['step'],
                state_sha256={name:tensor_sha(value) for name,value in values.items()})
    del saved
    return result


def verify_auxiliary(run,config,summary,history,ledgers,torch):
    need(os.environ.get('SLURM_JOB_ID'), 'Run the auxiliary audit inside a CPU Slurm allocation')
    run=Path(run)
    need(config['condition'] in ('control','aligned') and config['seed'] in (6,7)
         and config['profile'] is False, 'Unregistered V6 main condition/seed')
    for key,value in (('auxiliary_trainable',291),('total_trainable_including_auxiliary',1762691)):
        need(config['parameters'][key] == summary['parameters'][key] == value, 'Auxiliary parameter budget differs')
    need(config['auxiliary_optimizer'] == dict(lr=.001,weight_decay=0.0,max_grad_norm=1.0,separate=True),
         'Auxiliary optimizer is not the registered private group')
    cache,training,binding=teacher_bindings(run,config,summary)
    expected_distillation=dict(condition=config['condition'],rank=96,classes=3,parameters=291,
        temperature=1.,weight=1.,detach_messages=config['condition']=='control',
        teacher_index_sha256=binding['index_sha256'],query='last_original_prompt_token',deployed=False)
    need(config['local_distillation'] == config['architecture']['local_distillation'] == expected_distillation,
         'Auxiliary coefficient, query, temperature or gradient-routing architecture differs')
    need(config['architecture']['operator']['merge'] == 'sum'
         and summary['auxiliary_head_removed_from_inference'] is True, 'Native SUM inference/head-removal declaration differs')
    init_record=read(run/'auxiliary_initialization.json')
    need(config['auxiliary_initialization'] == init_record, 'Copied head initialization differs')
    init=initialization(init_record,config['seed'],torch)
    route_record=read(run/'auxiliary_gradient_route.json')
    need(summary['auxiliary_gradient_route'] == route_record, 'Summary auxiliary route differs')
    route=gradient_route(config['condition'],route_record)
    gradients=gradient_rows(read(run/'auxiliary_gradient_audit.json'))
    rows=ledgers['rows']
    need(len(rows)==1620 and len(history)==9, 'Auxiliary audit requires every training presentation/block')
    blocks=[]
    for block,epoch in enumerate(history):
        group=rows[block*180:(block+1)*180]
        need(len(group)==180 and all(r['block_index']==block for r in group), 'Auxiliary block order differs')
        observed=[local_record(row,training[row['sid']],cache['scenes'][row['sid']]) for row in group]
        count=mean(x['count_ce'] for x in observed);kl=mean(x['auxiliary_kl'] for x in observed)
        total=mean(x['total_loss'] for x in observed)
        need(close(epoch['mean_answer_token_ce'],count) and close(epoch['mean_auxiliary_kl'],kl)
             and close(epoch['mean_total_loss'],total) and close(total,count+kl),
             'Block auxiliary/count/total means do not reproduce all180 presentations')
        step=45*(block+1)
        need(epoch['auxiliary_optimizer_state_steps']==epoch['optimizer_state_steps']==[step]
             and epoch['auxiliary_optimizer_state_entries']==2 and epoch['optimizer_state_entries']==37,
             'Native37/private2 optimizer states did not persist for the same405 updates')
        images=sum(x['images'] for x in observed)
        blocks.append(dict(block_index=block,epoch=block+1,step=step,presentations=180,images=images,
            mean_answer_token_ce=count,mean_auxiliary_kl=kl,mean_total_loss=total,
            mean_context_student_teacher_argmax_agreement=mean(x['agreement'] for x in observed),
            image_weighted_student_teacher_argmax_agreement=sum(x['agreeing_images'] for x in observed)/images,
            mean_message_norm=mean(x['message_mean_norm'] for x in observed),
            mean_context_student_probabilities=[mean(x['student_mean_probabilities'][i] for x in observed) for i in range(3)]))
    need(sum(x['images'] for x in blocks)==19440, 'Auxiliary image presentation denominator differs')
    best=max(history,key=lambda e:(sum(x['correct'] for x in e['dev'])/72,
             -sum(x['gold_first_token_nll']*x['n'] for x in e['dev'])/72))
    selected=Path(summary['selected_checkpoint']).resolve()
    need(selected==CKPTS/run.name/'best.pt', 'Unexpected selected auxiliary checkpoint path')
    selected_head=checkpoint_head(selected,config,best,torch)
    last_head=checkpoint_head(selected.parent/'last.pt',config,history[-1],torch)
    artifacts=('auxiliary_initialization.json','auxiliary_gradient_route.json','auxiliary_gradient_audit.json',
               'teacher_binding.json','teacher_cache.json','teacher_plan.json','presentations.jsonl','training.json')
    return dict(passed=True,presentations=1620,image_presentations=19440,training_scenes=1540,
        initialization=init,teacher_binding=binding,gradient_route=route,private_gradient_summary=gradients,
        blocks=blocks,selected_head=selected_head,last_head=last_head,
        artifact_sha256={name:digest(run/name) for name in artifacts},
        limitations=[
          'Validates recorded loss arithmetic and metadata; per-image live student logits were not retained, so individual KL values and agreements cannot be independently recomputed.',
          'The291-parameter decoder and teacher are training resources, not part of the native SUM inference parameter groups.',
          'Optimizer step/entry records are audited; actual Adam moment tensors and complete per-step gradients were not retained.',
          'Student-teacher agreement measures fit to the frozen teacher, not independent semantic accuracy or model efficacy.',
          'Auxiliary and native clipping are separate; aligned reader gradients still alter native combined clipping and can indirectly change LoRA updates.'])


def self_test(torch):
    """Small adversarial audit checks; execute only in the CPU report allocation."""
    need(os.environ.get('SLURM_JOB_ID'), 'Run auxiliary self-tests inside Slurm')
    source=dict(n_frames=8)
    pairs=[str(i) for i in range(8)]
    row=dict(answer_token_ce=.5,auxiliary_kl=.25,total_loss=.75,prompt_tokens=10,target_ids=[1,2],
             local_evidence=dict(prompt_length=10,total_length=11,expected_images=8,branch_calls=1,
                query_position=9,all_images_visible=True,teacher_pair_ids_sha256=object_sha(pairs),
                message_mean_norm=1.,student_teacher_argmax_agreement=.5,student_mean_probabilities=[.2,.7,.1]))
    local_record(row,source,pairs)
    def rejected(call):
        try:call()
        except ValueError:return
        raise AssertionError('Adversarial auxiliary audit change was accepted')
    bad=copy.deepcopy(row);bad['local_evidence']['query_position']=10
    rejected(lambda:local_record(bad,source,pairs))
    rejected(lambda:local_record(row,source,list(reversed(pairs))))
    bad=copy.deepcopy(row);bad['total_loss']=.5
    rejected(lambda:local_record(bad,source,pairs))
    bad=copy.deepcopy(row);bad['local_evidence']['student_teacher_argmax_agreement']=.53
    rejected(lambda:local_record(bad,source,pairs))
    gradient=[dict(step=i,combined_preclip_norm=2.,clipped=True,parameters=291,tensors=2) for i in range(1,406)]
    gradient_rows(gradient)
    bad=copy.deepcopy(gradient);bad[0]['clipped']=False
    rejected(lambda:gradient_rows(bad))
    route=dict(condition='control',native_parameter_order='branch5_then_upper_lora32',
               connected_native_indices=[],native_gradient_norms=[None]*37,
               detached_control_native_gradient_is_structurally_zero=True)
    gradient_route('control',route)
    bad=copy.deepcopy(route);bad['connected_native_indices']=[0];bad['native_gradient_norms'][0]=.1
    rejected(lambda:gradient_route('control',bad))
    generator=torch.Generator(device='cpu');generator.manual_seed(1000009)
    tensors=dict(weight=torch.randn((3,96),generator=generator)*.01,bias=torch.zeros(3))
    state=dict(cpu='a'*64,cuda=['b'*64])
    record=dict(seed=6,stream_seed=1000009,weight_std=.01,bias_zero=True,parameters=291,
        main_rng_unchanged=True,main_rng_before=state,main_rng_after=copy.deepcopy(state),
        state_sha256={k:tensor_sha(v) for k,v in tensors.items()})
    before=torch.get_rng_state().clone();initialization(record,6,torch)
    need(torch.equal(before,torch.get_rng_state()), 'Audit initialization check consumed the global RNG')
    bad=copy.deepcopy(record);bad['state_sha256']['weight']='0'*64
    rejected(lambda:initialization(bad,6,torch))
    bad=copy.deepcopy(record);bad['main_rng_after']['cpu']='c'*64
    rejected(lambda:initialization(bad,6,torch))
    return dict(passed=True,adversarial_checks=8,private_rng_reproduced_without_global_consumption=True)

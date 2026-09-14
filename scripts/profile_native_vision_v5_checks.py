"""Actual HF V5 software gates; called only inside the Slurm GPU profile.

Parity probes are software-only and do not replace registered dev/test outputs.
The nonzero probe forces two tokens only to compare full and cached logits at
an identical prefix. Gate centered logits and probabilities: a constant logit
shift cancels in softmax. Raw discrepancies remain recorded. Native bf16
cached/full kernels differ; this bounded integration gate is not a parity proof.
"""
from __future__ import annotations
import hashlib
import json
import torch
from transformers import LogitsProcessor, LogitsProcessorList


@torch.no_grad()
def zero_initialization(model, branch, prepare, record):
    model.eval()
    assert torch.count_nonzero(branch.up.weight) == 0, 'Profile must start with zero U'
    inputs=prepare(record)
    branch.mode='off'
    native=model(**inputs,use_cache=False,logits_to_keep=1).logits.detach().float()
    branch.mode='all'
    augmented=model(**inputs,use_cache=False,logits_to_keep=1).logits.detach().float()
    maximum=float((native-augmented).abs().max())
    assert torch.equal(native,augmented), f'Zero-init native parity failed: max difference {maximum}'
    assert branch.memory_image_count==0, 'Uncached full forward did not clear ephemeral image memory'
    return dict(passed=True,sid=record['sid'],n_frames=record['n_frames'],
                maximum_absolute_logit_difference=maximum,criterion='exact_equal',
                zero_up_projection=True,uncached_memory_cleared=True)


@torch.no_grad()
def logit_parity_metrics(cached, uncached):
    """Raw diagnostics and shift-invariant checks; no inference policy changes."""
    assert cached.shape == uncached.shape and cached.ndim == 1
    assert bool(torch.isfinite(cached).all() and torch.isfinite(uncached).all()), 'Nonfinite parity logits'
    difference=cached.float()-uncached.float()
    centered=difference-difference.mean()
    # Double precision avoids misleading tiny negative KL from float32 sums.
    log_cached=cached.double().log_softmax(-1)
    log_uncached=uncached.double().log_softmax(-1)
    probability_cached=log_cached.exp(); probability_uncached=log_uncached.exp()
    cached_top1=int(cached.argmax()); uncached_top1=int(uncached.argmax())
    return dict(maximum_absolute_logit_difference=float(difference.abs().max()),
        rms_logit_difference=float(difference.square().mean().sqrt()),
        mean_logit_difference=float(difference.mean()),
        centered_maximum_absolute_logit_difference=float(centered.abs().max()),
        centered_rms_logit_difference=float(centered.square().mean().sqrt()),
        total_variation=float((probability_cached-probability_uncached).abs().sum()/2),
        kl_cached_to_full=float((probability_cached*(log_cached-log_uncached)).sum()),
        kl_full_to_cached=float((probability_uncached*(log_uncached-log_cached)).sum()),
        cached_top1_token=cached_top1,uncached_top1_token=uncached_top1,
        raw_top1_equal=cached_top1==uncached_top1)


@torch.no_grad()
def nonzero_cached_and_reset(model,branch,prepare,records):
    model.eval()
    branch.mode='all'
    assert float(branch.up.weight.detach().norm())>0, 'Post-training profile U must be nonzero'
    assert len(records)==4 and [r['n_frames'] for r in records]==[16,64,16,64]
    assert len({r['sid'] for r in records})==4, 'Expected four distinct frozen profile cases'
    class ForceFirst(LogitsProcessor):
        def __init__(self, prompt_length, token):
            self.prompt_length,self.token=prompt_length,token
        def __call__(self,input_ids,scores):
            if input_ids.shape[1]==self.prompt_length:
                scores=torch.full_like(scores,-torch.inf)
                scores[:,self.token]=0
            return scores
    calls=[0]
    handle=model.model.visual.register_forward_hook(lambda module,args,output: calls.__setitem__(0,calls[0]+1))
    observations=[]
    first=None
    try:
        for index,record in enumerate([*records,records[0]]):
            inputs=prepare(record)
            before=calls[0]
            generated=model.generate(**inputs,do_sample=False,use_cache=True,
                min_new_tokens=2,max_new_tokens=2,repetition_penalty=1.0,
                return_dict_in_generate=True,output_logits=True)
            assert calls[0]-before==1, 'Native cached generation encoded images more than once'
            ids=generated.sequences[0,inputs['input_ids'].shape[1]:]
            assert len(ids)==2 and len(generated.logits)==2
            assert branch.memory_image_count==record['n_frames'], 'Wrong retained image count after cached generation'
            cached=generated.logits[1][0].detach().float().clone()
            first_logits=generated.logits[0][0].detach().float().clone()
            if first is None:
                first=(ids.detach().clone(),first_logits.clone(),cached.clone())
            if index==4:
                assert torch.equal(ids,first[0]), 'A/B/C/D/A fresh-example reset changed greedy IDs'
                assert torch.equal(first_logits,first[1]) and torch.equal(cached,first[2]), 'A/B/C/D/A fresh-example reset changed logits'
            full=dict(inputs)
            full['input_ids']=torch.cat((inputs['input_ids'],ids[:1][None]),dim=1)
            if 'attention_mask' in full:
                full['attention_mask']=torch.cat((full['attention_mask'],torch.ones_like(ids[:1][None])),dim=1)
            if 'token_type_ids' in full:
                full['token_type_ids']=torch.cat((full['token_type_ids'],torch.zeros_like(ids[:1][None])),dim=1)
            before_full=calls[0]
            uncached=model(**full,use_cache=False,logits_to_keep=1).logits[0,-1].detach().float()
            assert calls[0]-before_full==1 and branch.memory_image_count==0
            enabled=logit_parity_metrics(cached,uncached)
            # Force ONLY the enabled branch's first token. Both OFF paths then
            # condition on exactly that prefix; raw logits are never overwritten.
            branch.mode='off'
            before_off=calls[0]
            native_generated=model.generate(**inputs,do_sample=False,use_cache=True,
                min_new_tokens=2,max_new_tokens=2,repetition_penalty=1.0,
                logits_processor=LogitsProcessorList([ForceFirst(inputs['input_ids'].shape[1],int(ids[0]))]),
                return_dict_in_generate=True,output_logits=True)
            assert calls[0]-before_off==1
            native_ids=native_generated.sequences[0,inputs['input_ids'].shape[1]:]
            assert len(native_ids)==2 and len(native_generated.logits)==2 and int(native_ids[0])==int(ids[0])
            assert branch.memory_image_count==record['n_frames']
            native_cached=native_generated.logits[1][0].detach().float()
            before_off_full=calls[0]
            native_full=model(**full,use_cache=False,logits_to_keep=1).logits[0,-1].detach().float()
            assert calls[0]-before_off_full==1 and branch.memory_image_count==0
            disabled=logit_parity_metrics(native_cached,native_full)
            relative_max_limit=1.5*disabled['centered_maximum_absolute_logit_difference']+0.0625
            relative_rms_limit=1.5*disabled['centered_rms_logit_difference']+0.005
            max_limit=min(0.25,relative_max_limit)
            rms_limit=min(0.05,relative_rms_limit)
            branch.mode='all'
            passed=(enabled['centered_maximum_absolute_logit_difference']<=max_limit
                and enabled['centered_rms_logit_difference']<=rms_limit
                and enabled['total_variation']<=0.01 and enabled['raw_top1_equal'])
            observation=dict(sid=record['sid'],n_frames=record['n_frames'],observation_index=index,
                case_role='prospective_unused_record' if index in (2,3) else 'original_case_or_reset',
                reset_repeat_of=0 if index==4 else None,passed=passed,
                **enabled,**{'native_reference_'+key:value for key,value in disabled.items()},
                centered_maximum_absolute_tolerance=max_limit,centered_rms_tolerance=rms_limit,
                relative_centered_maximum_absolute_tolerance=relative_max_limit,
                relative_centered_rms_tolerance=relative_rms_limit,total_variation_tolerance=0.01,
                native_reference_same_first_token=int(native_ids[0]),generated_ids=ids.detach().cpu().tolist(),
                native_visual_calls_cached_generation=1,native_visual_calls_uncached_comparison=1,
                native_visual_calls_disabled_cached=1,native_visual_calls_disabled_full=1,
                uncached_memory_cleared=True,
                first_logits_sha256=hashlib.sha256(first_logits.cpu().numpy().tobytes()).hexdigest())
            # Preserve all numerical evidence in the Slurm log even if a new
            # case fails. A failure stops this profile; thresholds are not tuned.
            print(json.dumps(dict(v5_centered_cache_gate=observation),allow_nan=False),flush=True)
            assert passed, f'Centered/probability cache integration gate failed: {observation}'
            observations.append(observation)
            del generated,inputs,full,cached,uncached,native_generated,native_full,native_cached
    finally:
        handle.remove()
        branch.mode='all'
        branch.reset_memory()
    assert calls[0]==20, 'Five observations must execute exactly twenty visual forwards'
    return dict(passed=True,up_projection_norm=float(branch.up.weight.detach().norm()),
                criterion='paired_native_reference_centered_and_probability',
                native_reference_multiplier=1.5,maximum_absolute_floor=0.0625,rms_floor=0.005,
                centered_maximum_absolute_cap=0.25,centered_rms_cap=0.05,total_variation_cap=0.01,
                raw_top1_equality_required=True,raw_logit_differences_diagnostic_only=True,
                probability_metrics_dtype='float64',
                disabled_reference_forces_only_enabled_first_token=True,
                fresh_example_reset='N16_A,N64_B,N16_C,N64_D,N16_A: exact repeated greedy IDs and logits',
                observations=observations,visual_calls=calls[0],
                scope='Bounded software integration check; not exact numerical parity or long-reasoning validation',
                software_only_forced_two_token_trajectories=True)

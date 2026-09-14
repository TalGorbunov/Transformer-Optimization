"""Small native-factor controller fixtures; execution only in Slurm CPU checks."""
import hashlib
import os
from pathlib import Path
import struct
import sys


def self_test(torch):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION')!='cpu' or os.environ.get('SLURM_JOB_GPUS'):
        raise RuntimeError('Run native factor fixtures in a CPU-only Slurm allocation')
    from scripts.native_factor_binding_runtime import FactorBindingNative,NativeGlobalConditioning,tensor_info,GlobalBroadcastLogitsProcessor
    from gnnformer.parallel_local_factor_binding import ParallelLocalFactorBinding
    checks=[]
    class Norm(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.linspace(.8,1.2,8,dtype=torch.float16),requires_grad=False)
        def forward(self,hidden_states):return hidden_states*self.weight
    def fixture(arm='product',active=False):
        core=ParallelLocalFactorBinding(8,rank=4,interaction=arm)
        if active:
            with torch.no_grad():core.up.weight.normal_(0,.2)
        core.eval().requires_grad_(False)
        return core,Norm().eval(),torch.linspace(-.3,.2,8),torch.tensor(.7)
    def controller(core,norm,mean,scale,capture=True):
        return FactorBindingNative(norm,core,n_local_rows=3,interaction=core.interaction,global_mean=mean,scale=scale,capture=capture)
    def reject(fn):
        try:fn()
        except (ValueError,RuntimeError,AssertionError):return
        raise AssertionError('Expected contract rejection')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261125)
        for arm in ('product','additive'):
            core,norm,mean,scale=fixture(arm);before={k:v.clone() for k,v in core.state_dict().items()}
            fusion=controller(core,norm,mean,scale)
            with fusion,torch.inference_mode():
                for step in range(4):
                    hidden=torch.randn(4,7 if step==0 else 1,8).half();baseline=hidden*norm.weight
                    actual=norm(hidden);cap=fusion.export_last_capture(cpu=True)
                    assert torch.equal(actual,baseline) and fusion.calls==fusion.conditioning.calls==fusion.conditioning.consumed==step+1
                    assert fusion.last_capture_call==step+1 and fusion.zero_up_identity_checks==step+1
                    assert len(cap)==22 and all(not torch.is_inference(v) and not v.requires_grad and v.grad_fn is None for v in cap.values())
                    x=core.rms(hidden[:-1,-1:]);z=(x-mean.view(1,1,-1))/scale
                    assert torch.equal(cap['original_local_rms_input'],x) and torch.equal(cap['conditioned_local_rms_input'],z)
                    assert torch.equal(cap['native_query_hidden'],hidden[:,-1:]) and torch.equal(cap['fused_query_hidden'],hidden[:,-1:])
                    assert bool((cap['delta']==0).all())
                    cap['conditioned_local_rms_input'].fill_(999)
                    assert not bool((fusion.export_last_capture()['conditioned_local_rms_input']==999).all())
            audit=fusion.audit();assert audit['active'] is audit['conditioning_active'] is False and audit['calls']==4
            assert not norm._forward_pre_hooks and not core.local._forward_pre_hooks and not hasattr(core.local,'_fixed_conditioning_active')
            assert all(torch.equal(v,before[k]) for k,v in core.state_dict().items())
        checks.append('both_modes_four_calls_zero_U_current_capture_normal_export_and_cleanup')

        for arm in ('product','additive'):
            core,norm,mean,scale=fixture(arm,True);hidden=torch.randn(4,5,8).half();untouched=hidden.clone()
            fusion=controller(core,norm,mean,scale)
            with fusion:
                actual=norm(hidden_states=hidden);cap=fusion.export_last_capture()
                x=core.rms(hidden[:-1,-1:]);z=(x-mean.view(1,1,-1))/scale
                q=torch.nn.functional.linear(core.rms(hidden[-1,-1:]),core.query.weight)
                projected=torch.nn.functional.linear(z,core.local.weight);pa,pb=projected.split(4,-1);ba,bb=core.local_bias.split(4)
                a=torch.tanh(pa+q.unsqueeze(0)+ba);b=torch.tanh(pb+q.unsqueeze(0)+bb)
                payload=a*b if arm=='product' else .5*(a+b)
                aggregate=(.5*payload).sum(0);pre=torch.nn.functional.linear(aggregate,core.aggregate_projection.weight,core.aggregate_projection.bias)+q
                delta=torch.nn.functional.linear(torch.nn.functional.silu(pre),core.up.weight)
                assert torch.equal(cap['factor_a'],a) and torch.equal(cap['factor_b'],b) and torch.equal(cap['query'],q)
                assert torch.equal(cap['delta'],delta) and torch.equal(cap['native_delta'],delta.half())
                expected=hidden.clone();expected[-1,-1:]=hidden[-1,-1:]+delta.half()
                assert torch.equal(actual,expected*norm.weight) and torch.equal(hidden,untouched)
                assert torch.equal(actual[:-1],(hidden*norm.weight)[:-1]) and torch.equal(actual[-1,:-1],(hidden*norm.weight)[-1,:-1])
                assert bool((actual[-1,-1]!=(hidden*norm.weight)[-1,-1]).any())
                changed=hidden.clone();changed[:,:-1]+=10
                norm(changed);later=fusion.export_last_capture()
                assert torch.equal(later['delta'],cap['delta']) and fusion.calls==2
                assert torch.equal(later['factor_permutation_indices'][:,0],torch.arange(3))
            assert all(p.grad is None and not p.requires_grad for p in core.parameters())
        checks.append('independent_factor_math_true_query_native_cast_global_only_and_past_position_exclusion')

        core,norm,mean,scale=fixture(active=True);fusion=controller(core,norm,mean,scale,False)
        with fusion:
            for _ in range(3):norm(torch.randn(4,1,8).half())
            assert fusion.calls==fusion.conditioning.consumed==3 and fusion.export_last_capture() is None
        checks.append('capture_disabled_still_consumes_each_current_conditioning_call')

        core,norm,mean,scale=fixture();outer=controller(core,norm,mean,scale);inner=controller(core,norm,mean,scale)
        with outer:
            reject(inner.__enter__)
            norm(torch.randn(4,1,8).half());assert outer.calls==1 and outer.active
        try:
            with controller(core,norm,mean,scale):raise RuntimeError('fixture')
        except RuntimeError as e:assert str(e)=='fixture'
        assert not norm._forward_pre_hooks and not core.local._forward_pre_hooks
        def fail(module,args,output):raise RuntimeError('forward fixture')
        handle=core.register_forward_hook(fail)
        try:
            try:
                with controller(core,norm,mean,scale):norm(torch.randn(4,1,8).half())
            except RuntimeError as e:assert str(e)=='forward fixture'
        finally:handle.remove()
        assert not norm._forward_pre_hooks and not core.local._forward_pre_hooks and not hasattr(core.local,'_fixed_conditioning_active')
        checks.append('nesting_and_body_or_forward_exception_cleanup')

        core,norm,mean,scale=fixture();fusion=controller(core,norm,mean,scale)
        with fusion:
            norm(torch.randn(4,1,8).half());reject(lambda:fusion.conditioning.consume(1))
            with torch.no_grad():mean.add_(1)
            reject(lambda:norm(torch.randn(4,1,8).half()))
        core,norm,mean,scale=fixture();fusion=controller(core,norm,mean,scale)
        with fusion:
            core.up.weight.requires_grad_(True);reject(lambda:norm(torch.randn(4,1,8).half()));core.up.weight.requires_grad_(False)
            core.set_factor_derangement(True);reject(lambda:norm(torch.randn(4,1,8).half()));core.set_factor_derangement(False)
            core.mode='clip';reject(lambda:norm(torch.randn(4,1,8).half()));core.mode='sigmoid'
            reject(lambda:norm(torch.randn(4,1,8)))
        core,norm,mean,scale=fixture();core.train();reject(lambda:controller(core,norm,mean,scale))
        class WrongSubclass(ParallelLocalFactorBinding):pass
        wrong=WrongSubclass(8,rank=4).eval().requires_grad_(False)
        reject(lambda:controller(wrong,norm,mean,scale))
        reject(lambda:NativeGlobalConditioning(core,mean,torch.tensor(0.)))
        reject(lambda:NativeGlobalConditioning(core,mean.view(1,-1),scale))
        checks.append('stale_capture_stat_mutation_dtype_grad_mode_derangement_and_exact_class_rejection')

        # Left-padded prompt rows differ; only generated suffixes must agree.
        ids=torch.tensor([[0,0,4,5],[0,6,7,8],[1,2,3,4],[0,0,9,8]])
        broadcast=GlobalBroadcastLogitsProcessor(n_local_rows=3,prompt_length=4)
        for token in (3,1,6):
            scores=torch.randn(4,7);scores[-1].fill_(-3);scores[-1,token]=5;original=scores.clone()
            selected=broadcast(ids,scores);next_ids=selected.argmax(-1)
            assert bool((next_ids==token).all()) and torch.equal(scores,original)
            ids=torch.cat((ids,next_ids[:,None]),1)
        assert broadcast.calls==3 and torch.equal(ids[:,4:],ids[-1:,4:].expand(4,-1))
        ids[0,-1]=2;reject(lambda:broadcast(ids,torch.randn(4,7)))
        checks.append('left_padded_original_rows_and_unmasked_common_native_token_broadcast')

        x=torch.tensor(.7,dtype=torch.float32);info=tensor_info(x)
        assert info==dict(shape=[],dtype='torch.float32',sha256=hashlib.sha256(struct.pack('<f',.7)).hexdigest())
        checks.append('scalar_conditioning_identity_retains_zero_dimensional_shape')
    return dict(passed=True,checks=checks,count=len(checks),device='cpu',no_pretrained_model_or_head=True)


if __name__=='__main__':
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION')!='cpu' or os.environ.get('SLURM_JOB_GPUS'):
        raise SystemExit('Run native factor fixtures in a CPU-only Slurm allocation')
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    import json,torch
    print(json.dumps(self_test(torch)))

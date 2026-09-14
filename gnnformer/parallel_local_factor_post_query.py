"""Existing local factors; query bypasses only the final readout SiLU."""
from __future__ import annotations
import os
import torch
from torch.nn import functional as F
from .parallel_local_aggregation import _require
from .parallel_local_factor_binding import ParallelLocalFactorBinding


class ParallelLocalFactorPostQuery(ParallelLocalFactorBinding):
    """delta=U[SiLU(Wagg SUM(messages)+b)+q]; factors still receive q.

    No constructor override: parameter names, shapes and seeded initialization
    are exactly inherited. Capture.preactivation is the actual SiLU input;
    the additional capture.readout is the actual up-projection input.
    """

    @property
    def query_placement(self):return 'post_silu'

    def forward(self,local_states,global_states,*,valid_mask=None,output_dtype=None,capture=False):
        self._global(global_states);self._local(local_states,global_states)
        _require(type(capture) is bool,'capture must be boolean')
        dtype=self._output_dtype(global_states,output_dtype)
        with torch.autocast(device_type=global_states.device.type,enabled=False):
            query=self.query(self.rms(global_states))
            values=dict(query=query,**self._encode(local_states,query,valid_mask))
            aggregate=self.aggregate(values['messages'])
            preactivation=self.aggregate_projection(aggregate)
            readout=F.silu(preactivation)+query
            delta=self.up(readout)
            self._floating_finite(delta,'delta');result=delta.to(dtype)
        if capture:return result,dict(values,aggregate=aggregate,preactivation=preactivation,readout=readout,delta=delta)
        return result


def self_test(torch_module):
    _require(torch_module is torch and bool(os.environ.get('SLURM_JOB_ID')),'Run post-query fixtures only in the CPU Slurm check')
    checks=[]
    with torch.random.fork_rng(devices=[]):
        for interaction in ('product','additive'):
            torch.manual_seed(24);old=ParallelLocalFactorBinding(4,rank=2,interaction=interaction)
            torch.manual_seed(24);new=ParallelLocalFactorPostQuery(4,rank=2,interaction=interaction)
            assert tuple(old.state_dict())==tuple(new.state_dict()) and all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items())
            assert sum(p.numel() for p in new.parameters())==45 and sum(p.numel() for p in new.parameters() if p.requires_grad)==42
            assert not list(new.buffers()) and new.query_placement=='post_silu'
            h=torch.tensor([[[1.,.2,-.3,.4],[.1,-.2,.6,.7]],[[.4,-.3,.2,1.],[.2,.5,-.1,.3]],[[.9,.8,-.2,.1],[-.5,.3,.2,.8]]],dtype=torch.float16)
            g=torch.tensor([[.2,.4,.6,.8],[-.3,.4,.2,.7]],dtype=torch.float16);mask=torch.tensor([[True,True],[True,False],[False,False]])
            d,cap=new(h,g,valid_mask=mask,output_dtype=torch.float16,capture=True);old_d,old_cap=old(h,g,valid_mask=mask,capture=True)
            assert d.dtype==torch.float16 and cap['delta'].dtype==torch.float32 and torch.equal(d,torch.zeros_like(d)) and torch.equal(g+d,g)
            for key in old_cap:
                if key not in ('preactivation','delta'):assert torch.equal(cap[key],old_cap[key]),key
            assert torch.equal(cap['messages'][~mask],torch.zeros_like(cap['messages'][~mask]))
            cap['delta'].sum().backward()
            for name,p in new.named_parameters():
                if name.startswith('selection_'):assert not p.requires_grad and p.grad is None
                elif name=='up.weight':assert p.grad is not None and bool((p.grad!=0).any())
                else:assert p.grad is not None and torch.equal(p.grad,torch.zeros_like(p.grad))
            new.zero_grad(set_to_none=True)
            with torch.no_grad():new.up.weight.copy_(torch.tensor([[.1,.2],[.3,-.2],[-.1,.4],[.5,.3]]));old.load_state_dict(new.state_dict())
            observed=[];handle=new.up.register_forward_pre_hook(lambda module,args:observed.append(args[0]))
            try:d,cap=new(h,g,valid_mask=mask,output_dtype=torch.float16,capture=True)
            finally:handle.remove()
            old_d,old_cap=old(h,g,valid_mask=mask,output_dtype=torch.float16,capture=True)
            a=F.linear(cap['aggregate'],new.aggregate_projection.weight,new.aggregate_projection.bias)
            expected=F.linear(F.silu(a)+cap['query'],new.up.weight)
            assert torch.equal(cap['preactivation'],a) and torch.equal(cap['readout'],observed[0]) and torch.equal(cap['delta'],expected)
            assert torch.equal(d,expected.half()) and not torch.allclose(cap['delta'],old_cap['delta'],atol=1e-6,rtol=1e-6)
            for key in ('query','factor_a','factor_b','factor_preactivation_a','factor_preactivation_b','payload','messages','aggregate'):
                assert torch.equal(cap[key],old_cap[key]),key
            cap['delta'].square().sum().backward()
            assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in new.parameters() if p.requires_grad)
            assert bool((new.query.weight.grad!=0).any()) and bool((new.local.weight.grad!=0).any())
            new.zero_grad(set_to_none=True)
            with torch.no_grad():new.query.weight.zero_();old.load_state_dict(new.state_dict())
            new_zero,new_cap=new(h,g,valid_mask=mask,capture=True);old_zero,old_cap=old(h,g,valid_mask=mask,capture=True)
            assert torch.equal(new_zero,old_zero) and torch.equal(new_cap['preactivation'],old_cap['preactivation'])
            changed=h.clone();changed[~mask]=100
            changed_d,changed_cap=new(changed,g,valid_mask=mask,capture=True)
            assert torch.equal(changed_cap['aggregate'],new_cap['aggregate']) and torch.equal(changed_d,new_zero)
            try:new.query_placement='pre_silu'
            except AttributeError:pass
            else:raise AssertionError('Read-only query placement changed')
        checks.extend(('same_seed_state_and_parameter_mask','zero_up_function_and_gradient','unchanged_nonzero_query_local_factors',
            'actual_up_input_and_post_silu_formula','live_local_and_query_gradients','zero_query_old_formula_parity','padding_and_query_placement_lock'))
    return dict(passed=True,checks=checks)

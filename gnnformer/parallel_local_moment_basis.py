"""Matched within/cross bases of the same query-conditioned set moments.

Both arms compute A=sum(.5*a), B=sum(.5*b), C=sum(.25*a*b), and
P=A*B-C. They differ only in the third readout block, C versus P.
The factors have no assigned semantics. Real-valued information equivalence
does not imply equal numerical conditioning or finite-width function classes.
"""
from __future__ import annotations
import os
import torch
from torch import nn
from torch.nn import functional as F
from .parallel_local_aggregation import ParallelLocalAggregation, _require

BASES = ('within', 'cross')


def moment_values(factor_a, factor_b, valid_mask):
    """Fixed operation order; FP32 in production, also FP64 in CPU fixtures."""
    scores = torch.full(factor_a.shape[:2], .5, dtype=factor_a.dtype, device=factor_a.device)
    gates = scores * valid_mask.to(factor_a.dtype)
    payload = torch.cat((factor_a, factor_b, .5 * (factor_a * factor_b)), dim=-1)
    messages = gates.unsqueeze(-1) * payload
    moments = messages.sum(0)
    a, b, c = moments.chunk(3, dim=-1)
    pair = a * b - c
    reconstructed = a * b - pair
    return dict(payload=payload, scores=scores, gates=gates, messages=messages,
                moments=moments, moment_a=a, moment_b=b, moment_c=c,
                moment_pair=pair, moment_reconstructed_c=reconstructed,
                moment_inverse_error=reconstructed-c)


class ParallelLocalMomentBasis(nn.Module):
    """Six FP32 tensors, 1,404,192 coordinates at H=3584,R=96; no selector.

Initialization draws query, local, aggregate projection and up in that order,
then zeros both biases and up. There are no discarded parent parameter draws.
The caller may condition the already-RMS input through a local prehook.
"""
    rms=staticmethod(ParallelLocalAggregation.rms)
    _floating_finite=staticmethod(ParallelLocalAggregation._floating_finite)
    _output_dtype=staticmethod(ParallelLocalAggregation._output_dtype)
    _global=ParallelLocalAggregation._global
    _local=ParallelLocalAggregation._local

    def __init__(self, hidden_size=3584, *, rank=96, basis='within'):
        nn.Module.__init__(self)
        _require(type(hidden_size) is int and hidden_size > 0, 'Positive integer hidden_size required')
        _require(type(rank) is int and rank > 0, 'Positive integer rank required')
        _require(basis in BASES, 'Unknown moment basis')
        self.hidden_size=hidden_size;self.rank=rank;self._basis=basis
        self.merge='sum';self.post_activation='silu'
        self.query=nn.Linear(hidden_size,rank,bias=False,dtype=torch.float32)
        self.local=nn.Linear(hidden_size,2*rank,bias=False,dtype=torch.float32)
        self.local_bias=nn.Parameter(torch.zeros(2*rank,dtype=torch.float32))
        self.aggregate_projection=nn.Linear(3*rank,rank,bias=True,dtype=torch.float32)
        self.up=nn.Linear(rank,hidden_size,bias=False,dtype=torch.float32)
        nn.init.zeros_(self.aggregate_projection.bias);nn.init.zeros_(self.up.weight)

    @property
    def basis(self):return self._basis

    def _parameters_on(self,device):
        ParallelLocalAggregation._parameters_on(self,device)
        _require(self.basis in BASES and self.merge=='sum' and self.post_activation=='silu'
                 and self.local.bias is None and self.local.weight.shape==(2*self.rank,self.hidden_size)
                 and self.local_bias.shape==(2*self.rank,)
                 and self.aggregate_projection.weight.shape==(self.rank,3*self.rank),
                 'Fixed moment architecture changed')

    def forward(self,local_states,global_states,*,valid_mask=None,output_dtype=None,capture=False):
        self._global(global_states);self._local(local_states,global_states)
        _require(type(capture) is bool,'capture must be boolean')
        shape=local_states.shape[:2]
        if valid_mask is None:valid_mask=torch.ones(shape,dtype=torch.bool,device=local_states.device)
        _require(isinstance(valid_mask,torch.Tensor) and valid_mask.dtype==torch.bool
                 and valid_mask.shape==shape and valid_mask.device==local_states.device,
                 'valid_mask must be boolean [N,Q] on the input device')
        dtype=self._output_dtype(global_states,output_dtype)
        with torch.autocast(device_type=global_states.device.type,enabled=False):
            query=self.query(self.rms(global_states))
            projected=self.local(self.rms(local_states));first,second=projected.chunk(2,dim=-1)
            bias_a,bias_b=self.local_bias.chunk(2)
            pre_a=first+query.unsqueeze(0)+bias_a;pre_b=second+query.unsqueeze(0)+bias_b
            factor_a=torch.tanh(pre_a);factor_b=torch.tanh(pre_b)
            values=moment_values(factor_a,factor_b,valid_mask)
            third=values['moment_c'] if self.basis=='within' else values['moment_pair']
            aggregate=torch.cat((values['moment_a'],values['moment_b'],third),dim=-1)
            preactivation=self.aggregate_projection(aggregate)+query
            readout=F.silu(preactivation);delta=self.up(readout)
            self._floating_finite(delta,'delta');result=delta.to(dtype)
        if capture:return result,dict(query=query,factor_a=factor_a,factor_b=factor_b,
            factor_preactivation_a=pre_a,factor_preactivation_b=pre_b,**values,
            aggregate=aggregate,preactivation=preactivation,readout=readout,delta=delta)
        return result


def self_test(torch_module):
    """Nonsemantic value/gradient fixtures, only in the bounded Slurm CPU check."""
    _require(torch_module is torch and bool(os.environ.get('SLURM_JOB_ID')),'Slurm CPU fixtures required')
    checks=[];roundoff=[]
    def explicit(a,b,mask):
        out=(a.sum()+b.sum())*0+torch.zeros(a.shape[1:],dtype=a.dtype)
        for i in range(a.shape[0]):
            for j in range(a.shape[0]):
                if i!=j:out=out+.25*a[i]*b[j]*(mask[i]&mask[j]).unsqueeze(-1)
        return out
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(24);one=ParallelLocalMomentBasis()
        torch.manual_seed(24);two=ParallelLocalMomentBasis(basis='cross')
        assert len(list(one.parameters()))==6 and not list(one.buffers())
        assert sum(p.numel() for p in one.parameters())==1404192
        assert all(p.requires_grad and p.dtype==torch.float32 for p in one.parameters())
        assert all(torch.equal(v,two.state_dict()[k]) for k,v in one.state_dict().items())
        del one,two
        checks.append('fresh_shared_initialization_six_live_tensors_no_selector')
        for dtype in (torch.float32,torch.float64):
            atol=2e-6 if dtype==torch.float32 else 1e-12
            for n in (0,1,4):
                a=(torch.arange(n*4,dtype=dtype).reshape(n,2,2)/9-.4).requires_grad_()
                b=(torch.arange(n*4,dtype=dtype).reshape(n,2,2).flip(-1)/11-.2).requires_grad_()
                mask=torch.ones((n,2),dtype=torch.bool)
                if n>1:mask[-1,1]=False
                value=moment_values(a,b,mask);reference=explicit(a,b,mask)
                assert torch.allclose(value['moment_pair'],reference,atol=atol,rtol=atol)
                weights=torch.tensor([[.3,-.4],[.7,.2]],dtype=dtype)
                actual_grad=torch.autograd.grad((value['moment_pair']*weights).sum(),(a,b),retain_graph=True)
                expected_grad=torch.autograd.grad((reference*weights).sum(),(a,b))
                assert all(torch.allclose(x,y,atol=atol,rtol=atol) for x,y in zip(actual_grad,expected_grad))
                if n<=1:assert torch.equal(value['moment_pair'],torch.zeros_like(reference))
                reverse=torch.arange(n-1,-1,-1)
                perm=moment_values(a[reverse],b[reverse],mask[reverse])
                assert torch.allclose(value['moments'],perm['moments'],atol=atol,rtol=atol)
                doubled=moment_values(torch.cat((a,a)),torch.cat((b,b)),torch.cat((mask,mask)))
                assert torch.allclose(doubled['moment_pair'],4*value['moment_pair']+2*value['moment_c'],atol=atol,rtol=atol)
                left=moment_values(a[:n//2],b[:n//2],mask[:n//2]);right=moment_values(a[n//2:],b[n//2:],mask[n//2:])
                merged=left['moment_pair']+right['moment_pair']+left['moment_a']*right['moment_b']+right['moment_a']*left['moment_b']
                assert torch.allclose(merged,value['moment_pair'],atol=atol,rtol=atol)
                changed_a=a.detach().clone();changed_a[~mask]=100
                assert torch.equal(moment_values(changed_a,b,mask)['moments'],value['moments'])
            # Deliberately ill-conditioned generic factors, outside the bounded tanh range.
            a=torch.tensor([[[1e8]],[[1.]]],dtype=dtype);b=torch.tensor([[[1e8]],[[2.]]],dtype=dtype);mask=torch.ones((2,1),dtype=torch.bool)
            value=moment_values(a,b,mask);ref=explicit(a.double(),b.double(),mask)
            roundoff.append(dict(dtype=str(dtype),pair_absolute_error=float((value['moment_pair'].double()-ref).abs().max()),
                inverse_absolute_error=float(value['moment_inverse_error'].abs().max()),descriptive_only=True))
        checks.extend(('ordered_pair_values_and_gradients_fp32_fp64','empty_singleton_padding_permutation',
                       'partition_merge_and_duplicate_laws','cancellation_descriptive_not_gate'))
        for basis in BASES:
            torch.manual_seed(24);core=ParallelLocalMomentBasis(4,rank=3,basis=basis)
            h=torch.tensor([[[.2,.4,-.1,.8]],[[.5,-.7,.3,.2]],[[.9,.1,-.4,.6]]],dtype=torch.float16)
            g=torch.tensor([[.3,-.2,.7,.4]],dtype=torch.float16);mask=torch.tensor([[True],[True],[False]])
            delta,cap=core(h,g,valid_mask=mask,capture=True)
            assert delta.dtype==torch.float16 and torch.equal(delta,torch.zeros_like(delta)) and torch.equal(g+delta,g)
            cap['delta'].sum().backward()
            for name,p in core.named_parameters():
                assert p.grad is not None and bool(torch.isfinite(p.grad).all())
                assert bool((p.grad!=0).any()) if name=='up.weight' else torch.equal(p.grad,torch.zeros_like(p.grad))
            core.zero_grad(set_to_none=True)
            with torch.no_grad():core.up.weight.copy_(torch.arange(12,dtype=torch.float32).reshape(4,3)/13-.2)
            observed=[];handle=core.up.register_forward_pre_hook(lambda module,args:observed.append(args[0]))
            try:delta,cap=core(h,g,valid_mask=mask,capture=True)
            finally:handle.remove()
            assert torch.equal(observed[0],cap['readout'])
            cap['delta'].square().sum().backward()
            assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) and bool((p.grad!=0).any()) for p in core.parameters())
            expected=F.linear(F.silu(F.linear(cap['aggregate'],core.aggregate_projection.weight,core.aggregate_projection.bias)+cap['query']),core.up.weight)
            assert torch.equal(cap['delta'],expected) and torch.equal(delta,expected.half())
            try:core.basis='cross'
            except AttributeError:pass
            else:raise AssertionError('Basis property must be read-only')
        checks.extend(('zero_up_identity_and_first_gradient','all_six_live_gradients_actual_native_cast','basis_property_and_actual_readout'))
    return dict(passed=True,checks=checks,cancellation=roundoff)

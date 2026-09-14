"""Joint-code residual with the direct query added after aggregate SiLU.

Only query placement changes. The inherited constructor preserves all four
parameter names, shapes, initialization bytes and random-number consumption.
"""
from __future__ import annotations
import os
import torch
from torch.nn import functional as F
from .parallel_local_aggregation import ParallelLocalAggregation, _require
from .parallel_local_joint_code_oracle import ParallelLocalJointCodeOracle, CODE_DIM, READOUT_KEYS, local_joint_code


class ParallelLocalJointCodePostQuery(ParallelLocalJointCodeOracle):
    """delta=U[SiLU(Wagg*.5*SUM_valid(codes)+b)+Wq*RMS(global)]."""

    @property
    def query_placement(self):
        return 'post_silu'

    def forward(self, codes, global_states, *, valid_mask=None, output_dtype=None, capture=False):
        ParallelLocalAggregation._floating_finite(global_states, 'global_states')
        _require(global_states.ndim == 2 and global_states.shape[0] > 0
                 and global_states.shape[1] == self.hidden_size, 'global_states must be [Q,H] with Q>0')
        _require(all(p.dtype == torch.float32 and p.device == global_states.device
                     and bool(torch.isfinite(p).all()) for p in self.parameters()),
                 'Readout parameters must be finite FP32 on the input device')
        _require(isinstance(codes, torch.Tensor) and codes.dtype == torch.float32
                 and codes.ndim == 3 and codes.shape[1:] == (global_states.shape[0], self.rank)
                 and codes.device == global_states.device and bool(torch.isfinite(codes).all()),
                 'codes must be finite FP32 [N,Q,R] on the global-state device')
        _require(bool(((codes == 0) | (codes == 1)).all()) and bool((codes[..., CODE_DIM:] == 0).all())
                 and bool((codes[..., :CODE_DIM].sum(-1) <= 1).all()),
                 'Codes must be raw one-hot-or-zero in the first18 coordinates, with zero tail')
        if valid_mask is None:
            valid_mask = torch.ones(codes.shape[:2], dtype=torch.bool, device=codes.device)
        _require(isinstance(valid_mask, torch.Tensor) and valid_mask.dtype == torch.bool
                 and valid_mask.shape == codes.shape[:2] and valid_mask.device == codes.device,
                 'valid_mask must be boolean [N,Q] on the code device')
        _require(type(capture) is bool, 'capture must be boolean')
        dtype = ParallelLocalAggregation._output_dtype(global_states, output_dtype)
        with torch.autocast(device_type=global_states.device.type, enabled=False):
            query = self.query(self.rms(global_states))
            scores = torch.full(codes.shape[:2], .5, dtype=torch.float32, device=codes.device)
            gates = scores * valid_mask.float()
            messages = gates.unsqueeze(-1) * codes
            aggregate = messages.sum(0)
            preactivation = self.aggregate_projection(aggregate)
            readout = F.silu(preactivation) + query
            delta = self.up(readout)
            ParallelLocalAggregation._floating_finite(delta, 'delta')
            result = delta.to(dtype)
        if capture:
            return result, dict(query=query, payload=codes, scores=scores, gates=gates,
                                messages=messages, aggregate=aggregate,
                                preactivation=preactivation, readout=readout, delta=delta)
        return result


def self_test(torch_module):
    """Tiny CPU Slurm fixtures; no native model, head or saved-data access."""
    _require(torch_module is torch and bool(os.environ.get('SLURM_JOB_ID')),'CPU Slurm fixtures required')
    checks=[]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(24);old=ParallelLocalJointCodeOracle();old_state={k:v.clone() for k,v in old.state_dict().items()};rng=torch.random.get_rng_state()
        torch.manual_seed(24);default=ParallelLocalJointCodePostQuery()
        assert torch.equal(rng,torch.random.get_rng_state()) and tuple(default.state_dict())==READOUT_KEYS
        assert all(torch.equal(v,old_state[k]) for k,v in default.state_dict().items())
        assert sum(p.numel() for p in default.parameters())==697440 and not list(default.buffers())
        assert default.query_placement=='post_silu' and all(p.dtype==torch.float32 and p.requires_grad for p in default.parameters())
        checks.append('exact_parent_initialization_rng_four_tensors_and_budget');del old,old_state,default
        core=ParallelLocalJointCodePostQuery(4,rank=18)
        codes=torch.tensor([[local_joint_code(0,0,rank=18)],[local_joint_code(0,1,rank=18)],[local_joint_code(2,None,rank=18)]],dtype=torch.float32)
        g=torch.tensor([[1.,.2,.3,.4]],dtype=torch.float16)
        result,cap=core(codes,g,output_dtype=torch.float16,capture=True)
        assert torch.equal(result,torch.zeros_like(result)) and torch.equal(g+result,g)
        assert torch.equal(cap['aggregate'],.5*codes.sum(0)) and torch.equal(cap['readout'],F.silu(cap['preactivation'])+cap['query'])
        cap['delta'].sum().backward();assert bool((core.up.weight.grad!=0).any())
        assert all(p.grad is not None and bool((p.grad==0).all()) for name,p in core.named_parameters() if name!='up.weight')
        checks.append('zero_U_identity_and_gradient_boundary');core.zero_grad(set_to_none=True)
        with torch.no_grad():
            core.query.weight.fill_(.2);core.aggregate_projection.weight.copy_(torch.eye(18));core.aggregate_projection.bias.zero_();core.up.weight.fill_(.1)
        a,ca=core(codes,g,capture=True);b,cb=core(codes,-g,capture=True)
        assert torch.equal(ca['preactivation'],cb['preactivation']) and torch.equal(ca['aggregate'],cb['aggregate'])
        assert not torch.equal(ca['query'],cb['query']) and not torch.equal(ca['readout'],cb['readout']) and not torch.equal(a,b)
        assert torch.autograd.grad(ca['preactivation'].sum(),core.query.weight,retain_graph=True,allow_unused=True)[0] is None
        ca['delta'].square().sum().backward()
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) and bool((p.grad!=0).any()) for p in core.parameters())
        checks.append('query_cannot_shift_inner_preactivation_but_changes_live_readout');core.zero_grad(set_to_none=True)
        with torch.no_grad():core.query.weight.zero_()
        old=ParallelLocalJointCodeOracle(4,rank=18);old.load_state_dict(core.state_dict())
        a,ca=core(codes,g,output_dtype=torch.float16,capture=True);b,cb=old(codes,g,output_dtype=torch.float16,capture=True)
        assert torch.equal(a,b) and all(torch.equal(ca[k],v) for k,v in cb.items())
        assert torch.equal(ca['readout'],F.silu(cb['preactivation']))
        checks.append('frozen_original_q_zero_equivalence')
        valid=torch.tensor([[True],[False],[True]])
        _,padded=core(codes,g,valid_mask=valid,capture=True)
        assert bool((padded['messages'][~valid]==0).all()) and torch.equal(padded['aggregate'],.5*codes[:1].sum(0))
        for values,mask in ((codes[:0],None),(codes,torch.zeros((3,1),dtype=torch.bool))):
            _,empty=core(values,g,valid_mask=mask,capture=True)
            assert bool((empty['aggregate']==0).all()) and bool(torch.isfinite(empty['delta']).all())
        checks.append('padding_empty_and_all_padded')
        bad=codes.clone();bad[0,0,2]=1
        for values in (bad,codes*2):
            try:core(values,g)
            except ValueError:pass
            else:raise AssertionError('Malformed or scaled local code accepted')
        checks.append('unchanged_raw_code_validation')
    return dict(passed=True,checks=checks,count=len(checks),device='cpu',query_placement='post_silu')

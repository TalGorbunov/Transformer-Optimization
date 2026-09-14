"""A cancellation-free bound on native residual sensitivity along paired chords.

For delta(r)=U SiLU(r), d=r_right-r_left and columns u_j of U, define
B(d)=sum_j ||u_j||_2 |d_j|. Since SiLU is globally 1.1-Lipschitz,
||delta(r+t*d)-delta(r)||_2 <= 1.1*|t|*B(d), for any r and real t.
The loss is mean B(d)^2/(||frozen_global||_2^2+eps), before native casting.
This bounds only the measured directions, not arbitrary scene extrapolation
or native answer accuracy. It is not a new attention operator or theorem.
"""
from __future__ import annotations
import math
import torch


def _need(condition,message):
    if not condition:raise ValueError(message)


def native_path_aggregate(branch,local_states,global_states):
    """Read the unchanged core's SUM statistic; no backbone computation.

    Native CE may use the original branch.forward independently. This extra
    graph is training-only, and never changes the deployed core.
    """
    _need(branch.merge=='sum' and branch.post_activation=='silu','Expected the unchanged SUM/SiLU core')
    with torch.autocast(device_type=global_states.device.type,enabled=False):
        return branch.aggregate(branch.encode(local_states,global_states))


def paired_native_path_bound(aggregate_states,aggregate_weight,output_weight,frozen_global,*,eps=1e-6):
    """Return a scalar FP32 loss for adjacent same-answer pairs.

    aggregate_states:[2P,T,R], aggregate_weight:[R,R], output_weight:[H,R],
    frozen_global:[2P,T,H]. Compute d=Wagg(z_right-z_left) directly: the
    identical query and aggregate bias cancel algebraically. This avoids
    subtracting large shared offsets in rounded endpoint preactivations.
    All learned inputs remain FP32; global states may use their native float
    dtype and must be frozen and identical within each pair. Output column
    norms use torch.linalg.vector_norm, whose zero subgradient is tested.
    No epsilon is added to those norms: zero-U initialization has exact zero
    loss and gradient, while denominator epsilon is explicit and fixed.
    """
    for name,value in (('aggregate_states',aggregate_states),('aggregate_weight',aggregate_weight),('output_weight',output_weight),('frozen_global',frozen_global)):
        _need(isinstance(value,torch.Tensor) and value.is_floating_point(),name+' must be a floating tensor')
        _need(bool(torch.isfinite(value).all()),name+' must be finite')
    _need(aggregate_states.ndim==3 and aggregate_states.shape[0]>0 and aggregate_states.shape[0]%2==0
          and aggregate_states.shape[1]>0 and aggregate_states.shape[2]>0,'Expected nonempty adjacent paired latent states')
    _need(aggregate_weight.shape==(aggregate_states.shape[2],aggregate_states.shape[2]),'Aggregate projection must have shape [R,R]')
    _need(output_weight.ndim==2 and output_weight.shape[0]>0 and output_weight.shape[1]==aggregate_states.shape[2],
          'Output projection must have shape [H,R]')
    _need(frozen_global.shape==(*aggregate_states.shape[:2],output_weight.shape[0]),'Frozen global shape differs')
    _need(aggregate_states.dtype==aggregate_weight.dtype==output_weight.dtype==torch.float32,'Learned bound arithmetic must be FP32')
    _need(aggregate_states.device==aggregate_weight.device==output_weight.device==frozen_global.device,'All tensors must share a device')
    _need(not frozen_global.requires_grad,'Denominator must be frozen')
    _need(torch.equal(frozen_global[0::2],frozen_global[1::2]),'Paired frozen global states must be identical')
    _need(isinstance(eps,(int,float)) and not isinstance(eps,bool) and math.isfinite(eps) and eps>0,
          'Denominator epsilon must be finite and positive')
    with torch.autocast(device_type=aggregate_states.device.type,enabled=False):
        difference=torch.nn.functional.linear(aggregate_states[1::2]-aggregate_states[0::2],aggregate_weight)
        column_norms=torch.linalg.vector_norm(output_weight,ord=2,dim=0)
        envelope=(difference.abs()*column_norms).sum(dim=-1)
        denominator=frozen_global[0::2].detach().float().square().sum(dim=-1)+eps
        result=(envelope.square()/denominator).mean()
    _need(bool(torch.isfinite(result)),'Native path bound overflowed')
    return result

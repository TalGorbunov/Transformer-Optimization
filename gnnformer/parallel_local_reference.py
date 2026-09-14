"""Software prototype for current-prefix reference centering of a native readout.

Rows are N actual image streams, R reference image streams, then one global
stream. The caller supplies complete ordinary N1 local prompts and broadcasts
only the global generated token to ALL rows. Reference selection is external;
this module never reads labels, tokens, images, or a lookup table of prefixes.

The existing ParallelLocalAggregation core is unchanged. Both sets are encoded
separately at the same global query. In projected readout coordinates:
  r = Wagg sum_i phi(actual_i, g) + b_rho + Wq RMS(g)
  b0 = Wagg mean_j phi(reference_j, g)
  delta = U SiLU(r - (N-anchor_n) b0)
A matched base skips the subtraction. A fixed orthogonal, norm-matched sham
uses the same computation and native rows. The N16 anchor and training bank
are explicit diagnostic assumptions, not a general counting algorithm.
"""
from __future__ import annotations
import hashlib
import json
import weakref

import torch
from gnnformer.parallel_local_aggregation import ParallelLocalAggregation


def need(condition,message):
    if not condition:raise ValueError(message)


def positions(values,name):
    need(isinstance(values,(list,tuple)) and len(values)>0,name+' must be a nonempty list or tuple')
    result=tuple(values)
    need(all(type(x) is int and x>=0 for x in result) and all(a<b for a,b in zip(result,result[1:])),
         name+' must contain strictly increasing nonnegative integers')
    return result


def orthogonal_directions(background,noise):
    """Per-query perpendicular directions, exact zero handling, no RNG calls.

    FP64 geometric construction followed by an explicit FP32 result. A single
    fixed noise vector is projected separately for each current-prefix b0.
    Near-parallel noise uses the least-aligned coordinate deterministically.
    """
    need(isinstance(background,torch.Tensor) and background.ndim==2 and background.shape[1]>=2
         and background.dtype==torch.float32 and bool(torch.isfinite(background).all()),
         'background must be finite FP32 [Q,rank>=2]')
    need(isinstance(noise,torch.Tensor) and noise.shape==(background.shape[1],)
         and bool(torch.isfinite(noise).all()),'noise must be one finite rank-vector')
    b=background.double();v=noise.to(device=b.device,dtype=torch.float64).unsqueeze(0).expand_as(b)
    length=b.norm(dim=-1,keepdim=True);unit=b/length.clamp_min(torch.finfo(torch.float64).tiny)
    perpendicular=v-(v*unit).sum(-1,keepdim=True)*unit
    fallback=torch.zeros_like(unit).scatter_(1,unit.abs().argmin(-1,keepdim=True),1.)
    fallback=fallback-(fallback*unit).sum(-1,keepdim=True)*unit
    perpendicular=torch.where(perpendicular.norm(dim=-1,keepdim=True)<1e-12,fallback,perpendicular)
    result=perpendicular/perpendicular.norm(dim=-1,keepdim=True).clamp_min(torch.finfo(torch.float64).tiny)*length
    return torch.where(length==0,torch.zeros_like(result),result).float()


class ParallelLocalReference:
    """One final-norm hook; native actual/reference K/V streams stay untouched.

    A plain controller holds the branch without registering it as a model child.
    Only explicit query_indices in the LAST global row receive a residual.
    Current sequence indices and absolute packed stream_positions are supplied
    by the runtime, never inferred from cache length. Cached calls generally use
    [0] with the current absolute position. Full-prefix calls may list all desired
    queries; final-norm writes do not propagate into past K/V, unlike pre-last writes.

    All three modes compute actual and reference messages with identical shapes.
    FP32 branch arithmetic is followed by native hidden + delta.to(hidden.dtype).
    Capture defaults to detached query-only tensors; live capture is opt-in.
    The complete native model must already be frozen/eval externally.
    """
    _owners=weakref.WeakKeyDictionary()

    def __init__(self,norm,branch,*,n_actual_rows,n_reference_rows=24,mode='background',
                 anchor_n=16,sham_key,query_indices,stream_positions,capture=False,detach_captures=True):
        need(isinstance(norm,torch.nn.Module) and isinstance(branch,ParallelLocalAggregation),
             'Require a native norm and existing ParallelLocalAggregation')
        need(branch.merge=='sum' and branch.post_activation=='silu' and branch.rank>=2,
             'Reference diagnostic uses the unchanged SUM/SiLU core')
        need(type(n_actual_rows) is int and n_actual_rows>=0 and type(n_reference_rows) is int and n_reference_rows>0,
             'Actual count must be nonnegative and reference count positive')
        need(type(anchor_n) is int and anchor_n>=0 and mode in ('base','background','sham'),
             'Invalid anchor or intervention mode')
        need(not any(p.requires_grad for p in norm.parameters()),'Native norm must already be frozen')
        self.norm,self.branch=norm,branch
        self.n_actual_rows,self.n_reference_rows=n_actual_rows,n_reference_rows
        self.anchor_n,self.mode=anchor_n,mode
        self.capture,self.detach_captures=bool(capture),bool(detach_captures)
        payload=json.dumps([20261101,sham_key],sort_keys=True,separators=(',',':'),allow_nan=False)
        # Runtime keys are [core_key, question], unchanged across N, mode and prefix.
        self.sham_seed=int(hashlib.sha256(payload.encode()).hexdigest()[:16],16)%(2**63-1)
        generator=torch.Generator(device='cpu');generator.manual_seed(self.sham_seed)
        self._noise=torch.randn(branch.rank,generator=generator,dtype=torch.float64)
        self._handle=None;self.calls=self.actual_reads=self.reference_reads=0;self.last_capture=None
        self.configure_queries(query_indices,stream_positions)

    @property
    def active(self):return self._handle is not None

    def configure_queries(self,query_indices,stream_positions):
        indices=positions(query_indices,'query_indices');stream=positions(stream_positions,'stream_positions')
        need(len(indices)==len(stream),'Query/stream position count differs')
        offsets={b-a for a,b in zip(indices,stream)}
        need(len(offsets)==1 and next(iter(offsets))>=0,'Require one nonnegative packed stream offset')
        self.query_indices,self.stream_positions=indices,stream

    def __enter__(self):
        need(not self.active,'Reference controller is already active')
        owner=self._owners.get(self.norm)
        need(owner is None or owner() is None,'Another reference controller owns the final norm')
        need(not any(p.requires_grad for p in self.norm.parameters()),'Native norm must remain frozen')
        self.calls=self.actual_reads=self.reference_reads=0;self.last_capture=None
        self._handle=self.norm.register_forward_pre_hook(self._before_norm,with_kwargs=True)
        self._owners[self.norm]=weakref.ref(self)
        return self

    def close(self):
        if self._handle is not None:self._handle.remove();self._handle=None
        owner=self._owners.get(self.norm)
        if owner is not None and owner() is self:del self._owners[self.norm]

    def __exit__(self,exc_type,exc_value,traceback):
        self.close();return False

    def remember(self,value):return value.detach().clone() if self.detach_captures else value

    def _before_norm(self,module,args,kwargs):
        need(self.active and module is self.norm,'Unexpected norm invocation')
        need(not any(p.requires_grad for p in module.parameters()),'Native norm must remain frozen')
        positional=bool(args)
        need(not positional or 'hidden_states' not in kwargs,'Ambiguous hidden states')
        need(positional or 'hidden_states' in kwargs,'Missing hidden states')
        h=args[0] if positional else kwargs['hidden_states'];n=self.n_actual_rows;r=self.n_reference_rows
        need(isinstance(h,torch.Tensor) and h.is_floating_point() and h.ndim==3
             and h.shape[0]==n+r+1 and h.shape[2]==self.branch.hidden_size
             and self.query_indices[-1]<h.shape[1],'Expected [N+R+1,current_sequence,H], global row last')
        index=torch.tensor(self.query_indices,dtype=torch.long,device=h.device)
        actual=h[:n].index_select(1,index);reference=h[n:n+r].index_select(1,index);g=h[-1].index_select(0,index)
        core=self.branch
        with torch.autocast(device_type=h.device.type,enabled=False):
            actual_messages=core.encode(actual,g);reference_messages=core.encode(reference,g)
            aggregate=core.aggregate(actual_messages)
            # Preserve the completed V10 diagnostic's occurrence-mean arithmetic.
            mean=reference_messages.double().mean(0).float()
            preactivation=core.aggregate_projection(aggregate)+core.query(core.rms(g))
            b0=torch.nn.functional.linear(mean,core.aggregate_projection.weight)
            sham=orthogonal_directions(b0,self._noise)
            direction=torch.zeros_like(b0) if self.mode=='base' else b0 if self.mode=='background' else sham
            changed=preactivation if self.mode=='base' else preactivation-(n-self.anchor_n)*direction
            delta=core.up(torch.nn.functional.silu(changed))
        need(delta.dtype==torch.float32 and delta.shape==g.shape and bool(torch.isfinite(delta).all()),
             'Nonfinite or malformed reference residual')
        fused_global=g+delta.to(h.dtype);fused=h.clone();fused[-1,index,:]=fused_global
        self.calls+=1;self.actual_reads+=1;self.reference_reads+=1;self.last_capture=None
        if self.capture:
            values=dict(actual_states=actual,reference_states=reference,global_states=g,
                actual_messages=actual_messages,reference_messages=reference_messages,actual_aggregate=aggregate,
                reference_mean=mean,preactivation=preactivation,background_direction=b0,sham_direction=sham,
                used_direction=direction,changed_preactivation=changed,delta=delta,native_global=g,fused_global=fused_global)
            self.last_capture={key:self.remember(value) for key,value in values.items()}
            self.last_capture.update(query_indices=list(self.query_indices),stream_positions=list(self.stream_positions),
                n_actual_rows=n,n_reference_rows=r,anchor_n=self.anchor_n,mode=self.mode,sham_seed=self.sham_seed)
        if positional:return (fused,)+args[1:],kwargs
        updated=dict(kwargs);updated['hidden_states']=fused
        return args,updated

    def export_last_capture(self,*,cpu=False):
        if self.last_capture is None:return None
        with torch.inference_mode(False),torch.no_grad():
            return {key:(value.detach().cpu().clone() if cpu else value.detach().clone())
                    if isinstance(value,torch.Tensor) else list(value) if isinstance(value,list) else value
                    for key,value in self.last_capture.items()}

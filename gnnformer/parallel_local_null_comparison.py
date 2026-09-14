"""Diagnostic-only learned versus live empirical null means in one native batch.

Rows are N actual streams, exactly24 reference occurrences, then the global
stream. Both choices compute the same messages and predictor; only the mean
passed to the immutable projected readout changes. No training, bank selection,
token interpretation, numerical acceptance or efficacy policy lives here.
"""
from __future__ import annotations
import weakref
import torch
from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
from gnnformer.conditional_null_mean import ConditionalNullMean, projected_null_readout
from gnnformer.parallel_local_learned_null import positions


def need(value,message):
    if not value:raise ValueError(message)


class ParallelLocalNullComparison:
    """Out-of-place final-norm write, with explicit physical/stream queries.

    The plain controller never registers learned modules on the backbone or
    touches its cache. Full-prefix callers specify every desired query; cached
    callers specify only current physical queries and their absolute positions.
    Writes cannot affect earlier decoder blocks or fixed-prefix native K/V.
    Default captures are off; enabled captures contain query-only copies.
    """
    _owners=weakref.WeakKeyDictionary()

    def __init__(self,norm,core,predictor,*,n_actual_rows,condition,mean_source,
                 query_indices,stream_positions,n_reference_rows=24,
                 capture=False,detach_captures=True):
        need(isinstance(norm,torch.nn.Module) and isinstance(core,ParallelLocalAggregation)
             and isinstance(predictor,ConditionalNullMean),'Require native norm, original core and separate predictor')
        need(core.merge=='sum' and core.post_activation=='silu' and predictor.rank==core.rank,
             'Require matching SUM/SiLU core and signed null predictor')
        need(type(n_actual_rows) is int and n_actual_rows>=0 and type(n_reference_rows) is int and n_reference_rows==24,
             'Require N>=0 actual rows and exactly24 ordered reference occurrences')
        need(condition in ('offset','centered') and mean_source in ('learned','bank'),'Unknown coefficient or mean source')
        need(not any(p.requires_grad for p in norm.parameters()),'Native norm must be frozen')
        self.norm,self.core,self.predictor=norm,core,predictor
        self.n_actual_rows,self.n_reference_rows=n_actual_rows,n_reference_rows
        self.condition,self.mean_source=condition,mean_source
        self.capture,self.detach_captures=bool(capture),bool(detach_captures)
        self._handle=None;self.last_capture=None
        self.calls=self.actual_reads=self.reference_reads=self.predictor_reads=0
        self.configure_queries(query_indices,stream_positions)

    @property
    def active(self):return self._handle is not None

    def configure_queries(self,query_indices,stream_positions):
        query=positions(query_indices,'query_indices');stream=positions(stream_positions,'stream_positions')
        need(len(query)==len(stream),'Query/stream position counts differ')
        offsets={s-q for q,s in zip(query,stream)}
        need(len(offsets)==1 and next(iter(offsets))>=0,'Require one nonnegative stream offset')
        self.query_indices,self.stream_positions=query,stream

    def __enter__(self):
        need(not self.active,'Comparison controller already active')
        # Refuse another known fusion controller while allowing read-only audit hooks.
        from gnnformer.parallel_local_reference import ParallelLocalReference
        from gnnformer.parallel_local_learned_null import ParallelLocalLearnedNull
        for registry in (self._owners,ParallelLocalReference._owners,ParallelLocalLearnedNull._owners):
            owner=registry.get(self.norm)
            need(owner is None or owner() is None,'Another null controller owns this norm')
        need(not any(p.requires_grad for p in self.norm.parameters()),'Native norm must remain frozen')
        self.calls=self.actual_reads=self.reference_reads=self.predictor_reads=0;self.last_capture=None
        self._handle=self.norm.register_forward_pre_hook(self._before_norm,with_kwargs=True)
        self._owners[self.norm]=weakref.ref(self)
        return self

    def close(self):
        if self._handle is not None:self._handle.remove();self._handle=None
        owner=self._owners.get(self.norm)
        if owner is not None and owner() is self:del self._owners[self.norm]

    def __exit__(self,*_):self.close();return False

    def _before_norm(self,module,args,kwargs):
        need(self.active and module is self.norm and not any(p.requires_grad for p in module.parameters()),
             'Unexpected norm call or unfrozen native norm')
        positional=bool(args)
        need((positional and 'hidden_states' not in kwargs) or (not positional and 'hidden_states' in kwargs),
             'Missing or ambiguous native hidden states')
        h=args[0] if positional else kwargs['hidden_states'];n=self.n_actual_rows
        need(isinstance(h,torch.Tensor) and h.is_floating_point() and h.ndim==3
             and h.shape[0]==n+25 and h.shape[2]==self.core.hidden_size
             and self.query_indices[-1]<h.shape[1],'Expected [N+25,current_width,H], global row last')
        ix=torch.tensor(self.query_indices,dtype=torch.long,device=h.device)
        actual=h[:n].index_select(1,ix);reference=h[n:n+24].index_select(1,ix);g=h[-1].index_select(0,ix)
        with torch.autocast(device_type=h.device.type,enabled=False):
            actual_messages=self.core.encode(actual,g);reference_messages=self.core.encode(reference,g)
            aggregate=self.core.aggregate(actual_messages);query=self.core.query(self.core.rms(g))
            predicted=self.predictor(query);bank=reference_messages.double().mean(0).float()
            used=predicted if self.mean_source=='learned' else bank
            result=projected_null_readout(self.core,aggregate,g,used,n_elements=n,mode=self.condition,output_dtype=torch.float32)
        delta=result['delta_float32'];fused_global=g+delta.to(h.dtype)
        need(bool(torch.isfinite(fused_global).all()),'Native cast/add produced nonfinite global hidden states')
        fused=h.clone();fused[-1,ix,:]=fused_global
        self.calls+=1;self.actual_reads+=1;self.reference_reads+=1;self.predictor_reads+=1;self.last_capture=None
        if self.capture:
            values=dict(actual_states=actual,reference_states=reference,global_states=g,
                actual_messages=actual_messages,reference_messages=reference_messages,aggregate=aggregate,
                query=query,predicted_mean=predicted,bank_mean=bank,used_mean=used,
                preactivation=result['preactivation'],projected_null=result['projected_null'],
                corrected_preactivation=result['corrected_preactivation'],delta=delta,native_global=g,fused_global=fused_global)
            self.last_capture={k:v.detach().clone() if self.detach_captures else v for k,v in values.items()}
            self.last_capture.update(query_indices=list(self.query_indices),stream_positions=list(self.stream_positions),
                n_actual_rows=n,n_reference_rows=24,condition=self.condition,mean_source=self.mean_source,
                coefficient=result['coefficient'])
        if positional:return (fused,)+args[1:],kwargs
        changed=dict(kwargs);changed['hidden_states']=fused
        return args,changed

    def export_last_capture(self,*,cpu=False):
        if self.last_capture is None:return None
        with torch.inference_mode(False),torch.no_grad():
            return {k:(v.detach().cpu().clone() if cpu else v.detach().clone()) if isinstance(v,torch.Tensor)
                    else list(v) if isinstance(v,list) else v for k,v in self.last_capture.items()}

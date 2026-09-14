"""Plain native norm controller for a separate learned conditional null mean.

No fitting or auxiliary gradient policy is implemented here. Every arithmetic
path through the original core and c(q) stays live for ordinary main losses.
"""
from __future__ import annotations
import weakref
import torch
from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
from gnnformer.conditional_null_mean import ConditionalNullMean, projected_null_readout


def need(condition, message):
    if not condition:raise ValueError(message)


def positions(values, name):
    need(isinstance(values, (list, tuple)) and len(values)>0, name+' must be a nonempty list')
    result=tuple(values)
    need(all(type(x) is int and x>=0 for x in result) and all(a<b for a,b in zip(result,result[1:])),
         name+' must contain strictly increasing nonnegative integers')
    return result


class ParallelLocalLearnedNull:
    """Only explicit query positions in the last/global row receive a write.

    Rows are N actual local streams and one global stream. The original core
    and predictor remain independent modules, never registered on the backbone.
    Fixed-prefix native K/V is unchanged because writes occur before final norm.
    Full-prefix callers list all historical queries; cached callers list only
    current physical positions and their explicit packed stream positions.
    """
    _owners=weakref.WeakKeyDictionary()

    def __init__(self, norm, core, predictor, *, n_local_rows, mode,
                 query_indices, stream_positions, capture=False, detach_captures=True):
        need(isinstance(norm,torch.nn.Module) and isinstance(core,ParallelLocalAggregation)
             and isinstance(predictor,ConditionalNullMean), 'Require native norm, original core and separate predictor')
        need(core.merge=='sum' and core.post_activation=='silu' and predictor.rank==core.rank,
             'Require matching-rank SUM/SiLU core and null predictor')
        need(type(n_local_rows) is int and n_local_rows>=0 and mode in ('offset','centered'),
             'Invalid local count or deployed null mode')
        need(not any(p.requires_grad for p in norm.parameters()), 'Native norm must be frozen')
        self.norm,self.core,self.predictor=norm,core,predictor
        self.n_local_rows,self.mode=n_local_rows,mode
        self.capture,self.detach_captures=bool(capture),bool(detach_captures)
        self._handle=None;self.calls=0;self.last_capture=None
        self.configure_queries(query_indices,stream_positions)

    @property
    def active(self):return self._handle is not None

    def configure_queries(self,query_indices,stream_positions):
        query=positions(query_indices,'query_indices');stream=positions(stream_positions,'stream_positions')
        need(len(query)==len(stream),'Query/stream counts differ')
        offsets={s-q for q,s in zip(query,stream)}
        need(len(offsets)==1 and next(iter(offsets))>=0,'Require one nonnegative packed-stream offset')
        self.query_indices,self.stream_positions=query,stream

    def __enter__(self):
        need(not self.active,'Controller already active')
        owner=self._owners.get(self.norm)
        need(owner is None or owner() is None,'Another learned-null controller owns this norm')
        self.calls=0;self.last_capture=None
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
             'Missing or ambiguous hidden states')
        h=args[0] if positional else kwargs['hidden_states']
        need(isinstance(h,torch.Tensor) and h.is_floating_point() and h.ndim==3
             and h.shape[0]==self.n_local_rows+1 and h.shape[2]==self.core.hidden_size
             and self.query_indices[-1]<h.shape[1], 'Expected [N+1,current_width,H], global row last')
        index=torch.tensor(self.query_indices,device=h.device,dtype=torch.long)
        local=h[:-1].index_select(1,index);g=h[-1].index_select(0,index)
        with torch.autocast(device_type=h.device.type,enabled=False):
            messages=self.core.encode(local,g)
            aggregate=self.core.aggregate(messages)
            query=self.core.query(self.core.rms(g))
            mean=self.predictor(query)
            result=projected_null_readout(self.core,aggregate,g,mean,n_elements=self.n_local_rows,
                mode=self.mode,output_dtype=torch.float32)
        delta=result['delta_float32']
        fused_global=g+delta.to(h.dtype);fused=h.clone();fused[-1,index,:]=fused_global
        self.calls+=1;self.last_capture=None
        if self.capture:
            values=dict(local_states=local,global_states=g,messages=messages,aggregate=aggregate,null_mean=mean,
                query=query,preactivation=result['preactivation'],projected_null=result['projected_null'],
                corrected_preactivation=result['corrected_preactivation'],delta=delta,
                native_global=g,fused_global=fused_global)
            self.last_capture={k:v.detach().clone() if self.detach_captures else v for k,v in values.items()}
            self.last_capture.update(query_indices=list(self.query_indices),stream_positions=list(self.stream_positions),
                n_local_rows=self.n_local_rows,mode=self.mode,coefficient=result['coefficient'])
        if positional:return (fused,)+args[1:],kwargs
        changed=dict(kwargs);changed['hidden_states']=fused
        return args,changed

    def export_last_capture(self,*,cpu=False):
        if self.last_capture is None:return None
        with torch.inference_mode(False),torch.no_grad():
            return {k:(v.detach().cpu().clone() if cpu else v.detach().clone()) if isinstance(v,torch.Tensor)
                    else list(v) if isinstance(v,list) else v for k,v in self.last_capture.items()}

"""Source-only native semantic-gate controller; no efficacy/software release.

One final-norm pre-hook probes the ORIGINAL empty-prefix query exactly once.
Cached decoding and explicit full-prefix replay reuse a scene-bound artifact.
The direct norm/head calls bypass hooks and have separate explicit counters.
The controller never registers the trainable branch as a native model child.
"""
from __future__ import annotations
import hashlib
import json
import weakref
import torch
from gnnformer.parallel_local_semantic_aggregation import ParallelLocalSemanticAggregation

GATE_RULE='max(0,full_vocabulary_p1-full_vocabulary_p0); FP64 probabilities then detached FP32'
ORIGIN_KEYS={'sid','question_sha256','image_sha256','prompt_width','input_identity_sha256','native_identity_sha256'}


def need(value,message):
    if not value:raise ValueError(message)


def positions(values,name):
    need(isinstance(values,(list,tuple)) and len(values)>0,name+' must be a nonempty list')
    value=tuple(values)
    need(all(type(x) is int and x>=0 for x in value) and all(a<b for a,b in zip(value,value[1:])),
        name+' must be strictly increasing nonnegative integers')
    return value


def json_copy(value):return json.loads(json.dumps(value,sort_keys=True,allow_nan=False))


def object_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def tensor_info(value):
    x=value.detach().cpu().contiguous()
    return dict(shape=list(x.shape),dtype=str(x.dtype),sha256=hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest())


def artifact_digest(value):
    return object_sha({k:tensor_info(v) if isinstance(v,torch.Tensor) else v
                       for k,v in value.items() if k!='artifact_sha256'})


def normal_copy(value,*,cpu=False):
    with torch.inference_mode(False),torch.no_grad():
        return value.detach().cpu().clone() if cpu else value.detach().clone()


def full_vocabulary_gates(logits,zero_token_id,one_token_id,n_local_rows):
    """No binary renormalization; native logits are retained at their dtype."""
    need(isinstance(logits,torch.Tensor) and logits.dtype==torch.float16 and logits.ndim==3
        and logits.shape[:2]==(n_local_rows+1,1) and bool(torch.isfinite(logits).all()),'Expected finite native [N+1,1,V] FP16 logits')
    need(type(zero_token_id) is int and type(one_token_id) is int and zero_token_id!=one_token_id
        and min(zero_token_id,one_token_id)>=0 and max(zero_token_id,one_token_id)<logits.shape[-1],
        'Require distinct valid original ASCII0/1 token IDs')
    with torch.no_grad(),torch.autocast(device_type=logits.device.type,enabled=False):
        values=logits[:n_local_rows,0].double();normalizer=torch.logsumexp(values,dim=-1)
        p0=(values[:,zero_token_id]-normalizer).exp();p1=(values[:,one_token_id]-normalizer).exp()
        gates=(p1-p0).clamp_min(0).float().detach()
    return p0.detach(),p1.detach(),gates


class ParallelLocalSemanticGate:
    """Explicit scene-bound origin gates and query-only global residual writes.

    origin_identity contains exactly ORIGIN_KEYS. The caller binds its native
    model/processor/layout and original empty-prefix inputs through the two
    supplied identity digests. This controller checks equality of those bindings;
    it does not hash a multi-GB backbone or inspect questions/QA itself.

    Rows are N valid actual image streams plus one global stream; padded image
    rows are unsupported. Token padding is handled by the caller's native layout.
    New gates initialize only at original width and singleton final prompt query.
    A fresh controller with origin_artifact is replay-only and makes no probe.
    configure_queries accepts the same explicit physical/packed positions as the
    learned-null controller. Full-prefix callers may list historical queries;
    only their listed global positions change. Reentry needs explicit reset.
    """
    _owners=weakref.WeakKeyDictionary()

    def __init__(self,norm,head,core,*,n_local_rows,mode,zero_token_id,one_token_id,
                 origin_identity,query_indices,stream_positions,origin_artifact=None,
                 capture=False,detach_captures=True):
        need(isinstance(norm,torch.nn.Module) and isinstance(head,torch.nn.Module)
            and isinstance(core,ParallelLocalSemanticAggregation),'Require native norm/head and semantic aggregation core')
        need(type(n_local_rows) is int and n_local_rows>=0 and mode in ('native_gate','all_open'),
            'Invalid actual-row count or gate mode')
        need(type(zero_token_id) is int and type(one_token_id) is int and min(zero_token_id,one_token_id)>=0
            and zero_token_id!=one_token_id,'Distinct nonnegative ASCII0/1 token IDs required')
        self.norm,self.head,self.core=norm,head,core
        self.n_local_rows,self.mode=n_local_rows,mode
        self.zero_token_id,self.one_token_id=zero_token_id,one_token_id
        self.capture,self.detach_captures=bool(capture),bool(detach_captures)
        self._handle=None;self._entered=False;self._origin=None;self._artifact=None;self._artifact_source=None
        self._native_signature=self._signature();self._native_guard()
        self.reset(origin_identity,query_indices=query_indices,stream_positions=stream_positions,origin_artifact=origin_artifact)

    @property
    def active(self):return self._handle is not None

    def _signature(self):
        return tuple((kind,name,id(p),p._version,tuple(p.shape),str(p.dtype),str(p.device))
            for kind,module in [('norm',self.norm),('head',self.head)] for name,p in module.named_parameters())

    def _native_guard(self):
        params=[p for module in (self.norm,self.head) for p in module.parameters()]
        need(params and not self.norm.training and not self.head.training
            and all(not p.requires_grad and p.grad is None and p.dtype==torch.float16 for p in params),
            'Native norm/head must remain frozen, eval, FP16 and without gradients')
        need(self._signature()==self._native_signature,'Native norm/head parameters, device or version changed')

    def _identity(self,value):
        need(isinstance(value,dict) and set(value)==ORIGIN_KEYS,'Exact original scene/input/native identity fields required')
        result=json_copy(value)
        need(isinstance(result['sid'],str) and result['sid'] and type(result['prompt_width']) is int and result['prompt_width']>0
            and isinstance(result['image_sha256'],list) and len(result['image_sha256'])==self.n_local_rows,
            'Invalid original scene, prompt width or image occurrence count')
        hashes=[result[k] for k in ('question_sha256','input_identity_sha256','native_identity_sha256')]+result['image_sha256']
        need(all(isinstance(x,str) and len(x)==64 and all(c in '0123456789abcdef' for c in x) for x in hashes),
            'Origin identities must be lowercase SHA256 strings')
        return result

    @property
    def origin_identity(self):return json_copy(self._origin)

    def configure_queries(self,query_indices,stream_positions,*,origin_identity=None):
        if origin_identity is not None:need(self._identity(origin_identity)==self._origin,'Unknown scene or origin identity changed')
        q,s=positions(query_indices,'query_indices'),positions(stream_positions,'stream_positions')
        need(len(q)==len(s) and len({b-a for a,b in zip(q,s)})==1 and s[0]>=q[0],
            'Require one nonnegative physical-to-packed offset')
        need(s[0]>=self._origin['prompt_width']-1,'Queries cannot precede the original prediction query')
        self.query_indices,self.stream_positions=q,s

    def reset(self,origin_identity,*,query_indices,stream_positions,origin_artifact=None):
        need(not self.active,'Close the controller before explicit scene reset');self._native_guard()
        self._origin=self._identity(origin_identity);self._origin_sha256=object_sha(self._origin)
        self._artifact=None;self._artifact_source=None;self._entered=False
        self.calls=self.probe_norm_calls=self.probe_head_calls=self.probability_calls=0
        self.last_capture=None;self.configure_queries(query_indices,stream_positions)
        if origin_artifact is not None:self.bind_origin_artifact(origin_artifact)
        return self

    def bind_origin_artifact(self,artifact):
        need(not self.active and self._artifact is None,'Bind an origin artifact once, before attaching')
        need(isinstance(artifact,dict) and artifact.get('schema_version')==1 and artifact.get('gate_rule')==GATE_RULE
            and artifact.get('origin_identity')==self._origin and artifact.get('mode')==self.mode
            and artifact.get('zero_token_id')==self.zero_token_id and artifact.get('one_token_id')==self.one_token_id
            and artifact.get('n_local_rows')==self.n_local_rows,'Bound gate artifact belongs to another scene/native identity/mode')
        need(artifact.get('artifact_sha256')==artifact_digest(artifact),'Origin artifact content identity differs')
        n,h=self.n_local_rows,self.core.hidden_size
        for key in ('origin_hidden','origin_normalized'):
            x=artifact[key];need(isinstance(x,torch.Tensor) and x.shape==(n+1,1,h) and x.dtype==torch.float16
                and not x.requires_grad and bool(torch.isfinite(x).all()),'Invalid native origin query tensor')
        logits=artifact['origin_logits'];need(isinstance(logits,torch.Tensor) and logits.ndim==3 and logits.shape[:2]==(n+1,1)
            and logits.dtype==torch.float16 and not logits.requires_grad and bool(torch.isfinite(logits).all())
            and max(self.zero_token_id,self.one_token_id)<logits.shape[-1],'Invalid native origin logit archive')
        p0,p1,gates=artifact['p0'],artifact['p1'],artifact['native_gates']
        need(all(isinstance(x,torch.Tensor) and x.shape==(n,) and not x.requires_grad and bool(torch.isfinite(x).all()) for x in (p0,p1,gates))
            and p0.dtype==p1.dtype==torch.float64 and gates.dtype==torch.float32
            and bool(((p0>=0)&(p0<=1)&(p1>=0)&(p1<=1)&(p0+p1<=1+1e-12)).all())
            and torch.equal(gates,(p1-p0).clamp_min(0).float()),'Invalid detached full-vocabulary probability/gate values')
        origin=self._origin['prompt_width']-1
        need(artifact['origin_query_index']==artifact['origin_stream_position']==origin
            and artifact['probe_shape']==[n+1,1,h],'Origin probe position/shape differs')
        # The external artifact digest binds raw logits and saved FP64 reductions.
        # Do not silently replace saved gates with a new CPU/GPU reduction route.
        self._artifact={k:normal_copy(v) if isinstance(v,torch.Tensor) else json_copy(v) for k,v in artifact.items()}
        self._artifact_source='bound_artifact'
        return self

    def __enter__(self):
        need(not self.active and not self._entered,'Controller reentry requires explicit reset or a new bound-replay controller')
        self._native_guard();owner=self._owners.get(self.norm)
        need(owner is None or owner() is None,'Another semantic-gate controller owns this norm')
        self._handle=self.norm.register_forward_pre_hook(self._before_norm,with_kwargs=True)
        self._owners[self.norm]=weakref.ref(self);self._entered=True;return self

    def attach(self):return self.__enter__()

    def close(self):
        if self._handle is not None:self._handle.remove();self._handle=None
        owner=self._owners.get(self.norm)
        if owner is not None and owner() is self:del self._owners[self.norm]

    def __exit__(self,*_):self.close();return False

    def _probe_origin(self,hidden):
        origin=self._origin['prompt_width']-1
        need(hidden.shape[1]==self._origin['prompt_width'] and self.query_indices==self.stream_positions==(origin,),
            'Later/full-prefix execution requires a bound original gate artifact; never reclassify the current query')
        with torch.no_grad(),torch.autocast(device_type=hidden.device.type,enabled=False):
            queries=hidden[:,origin:origin+1].detach().contiguous()
            self.probe_norm_calls+=1;normalized=self.norm.forward(queries)
            need(normalized.shape==queries.shape and normalized.dtype==torch.float16,'Native probe norm changed shape/dtype')
            self.probe_head_calls+=1;logits=self.head.forward(normalized)
            self.probability_calls+=1;p0,p1,gates=full_vocabulary_gates(logits,self.zero_token_id,self.one_token_id,self.n_local_rows)
        self._artifact=dict(schema_version=1,gate_rule=GATE_RULE,mode=self.mode,origin_identity=self.origin_identity,
            zero_token_id=self.zero_token_id,one_token_id=self.one_token_id,n_local_rows=self.n_local_rows,
            origin_query_index=origin,origin_stream_position=origin,probe_shape=list(queries.shape),
            origin_hidden=queries.detach().clone(),origin_normalized=normalized.detach().clone(),
            origin_logits=logits.detach().clone(),p0=p0.detach().clone(),p1=p1.detach().clone(),native_gates=gates.detach().clone())
        self._artifact_source='native_prefill'

    def _before_norm(self,module,args,kwargs):
        need(self.active and module is self.norm,'Unexpected native norm invocation');self._native_guard()
        positional=bool(args)
        need((positional and 'hidden_states' not in kwargs) or (not positional and 'hidden_states' in kwargs),
            'Missing or ambiguous native hidden states')
        hidden=args[0] if positional else kwargs['hidden_states']
        need(isinstance(hidden,torch.Tensor) and hidden.dtype==torch.float16 and hidden.ndim==3
            and hidden.shape[0]==self.n_local_rows+1 and hidden.shape[2]==self.core.hidden_size
            and self.query_indices[-1]<hidden.shape[1] and bool(torch.isfinite(hidden).all()),
            'Expected finite FP16 [N+1,current_width,H], global row last')
        need(all(p.device==hidden.device for module in (self.norm,self.head,self.core) for p in module.parameters()),
            'Native norm/head/core and query devices differ')
        if self._artifact is None:self._probe_origin(hidden)
        native_gates=self._artifact['native_gates'].to(device=hidden.device)
        applied=native_gates if self.mode=='native_gate' else torch.ones_like(native_gates)
        index=torch.tensor(self.query_indices,device=hidden.device,dtype=torch.long)
        local=hidden[:-1].index_select(1,index);global_states=hidden[-1].index_select(0,index)
        gates=applied[:,None].expand(self.n_local_rows,len(self.query_indices)).detach()
        value=self.core(local,global_states,gates=gates,output_dtype=torch.float32,capture=self.capture)
        if self.capture:delta,core_capture=value
        else:delta=value;core_capture={}
        need(delta.shape==global_states.shape and delta.dtype==torch.float32 and bool(torch.isfinite(delta).all()),
            'Core must return finite FP32 global query residuals')
        fused_global=global_states+delta.to(hidden.dtype);fused=hidden.clone();fused[-1,index]=fused_global
        self.calls+=1;self.last_capture=None
        if self.capture:
            values=dict(core_capture,local_states=local,global_states=global_states,native_global=global_states,
                native_gates=native_gates,applied_gates=gates,delta=delta,fused_global=fused_global)
            self.last_capture={k:v.detach().clone() if self.detach_captures else v for k,v in values.items()}
            self.last_capture.update(mode=self.mode,n_local_rows=self.n_local_rows,query_indices=list(self.query_indices),
                stream_positions=list(self.stream_positions),origin_identity=self.origin_identity,
                origin_identity_sha256=self._origin_sha256,origin_source=self._artifact_source,
                probe_norm_calls=self.probe_norm_calls,probe_head_calls=self.probe_head_calls,probability_calls=self.probability_calls)
        if positional:return (fused,)+args[1:],kwargs
        changed=dict(kwargs);changed['hidden_states']=fused;return args,changed

    def assert_complete(self):
        self._native_guard();need(self.calls>0 and self._artifact is not None,'No completed semantic-gate forward')
        expected=1 if self._artifact_source=='native_prefill' else 0
        need(self.probe_norm_calls==self.probe_head_calls==self.probability_calls==expected,
            'Origin probe must occur once on prefill or zero times for bound replay')
        return dict(passed=True,calls=self.calls,origin_source=self._artifact_source,probe_norm_calls=self.probe_norm_calls,
            probe_head_calls=self.probe_head_calls,probability_calls=self.probability_calls)

    def export_origin_artifact(self,*,cpu=True):
        if self._artifact is None:return None
        values={k:normal_copy(v,cpu=cpu) if isinstance(v,torch.Tensor) else json_copy(v)
                for k,v in self._artifact.items() if k!='artifact_sha256'}
        values['artifact_sha256']=artifact_digest(values);return values

    def export_last_capture(self,*,cpu=False):
        if self.last_capture is None:return None
        return {k:normal_copy(v,cpu=cpu) if isinstance(v,torch.Tensor) else json_copy(v) for k,v in self.last_capture.items()}

"""Native binary-origin measurement with an explicit invalid-result boundary.

Reuse the immutable semantic controller's lifecycle, query indexing, native
parameter guards and out-of-place final-norm write. Replace its soft measurement
and artifact validation; no oracle substitution or new learned parameters.
"""
import torch
from gnnformer.parallel_local_binary_aggregation import ParallelLocalBinaryAggregation
from gnnformer.parallel_local_semantic_gate import (
    ParallelLocalSemanticGate,need,normal_copy,json_copy,artifact_digest)

BINARY_GATE_RULE='unmasked native vocabulary argmax: literal0 excludes, literal1 includes, other invalid'


class InvalidBinaryMeasurement(RuntimeError):
    """Retains the complete original measurement; callers must count failure."""
    def __init__(self,artifact):
        self.artifact=artifact
        super().__init__('Invalid native binary argmax at local rows '+str(artifact['invalid_rows']))


def binary_measurement(logits,zero_token_id,one_token_id,n_local_rows):
    need(type(n_local_rows) is int and n_local_rows>=0,'Nonnegative integer local-row count required')
    need(isinstance(logits,torch.Tensor) and logits.ndim==3 and logits.shape[:2]==(n_local_rows+1,1)
         and logits.dtype==torch.float16 and bool(torch.isfinite(logits).all()),'Finite native FP16 origin logits required')
    need(type(zero_token_id) is int and type(one_token_id) is int and zero_token_id!=one_token_id
         and min(zero_token_id,one_token_id)>=0 and max(zero_token_id,one_token_id)<logits.shape[-1],'Distinct literal binary token IDs required')
    with torch.no_grad(),torch.autocast(device_type=logits.device.type,enabled=False):
        values=logits[:n_local_rows,0].double();top=values.argmax(-1)
        valid=(top==zero_token_id)|(top==one_token_id);labels=torch.full_like(top,-1,dtype=torch.int8)
        labels[top==zero_token_id]=0;labels[top==one_token_id]=1
        normalizer=torch.logsumexp(values,-1);p0=(values[:,zero_token_id]-normalizer).exp();p1=(values[:,one_token_id]-normalizer).exp()
        passed=bool(valid.all());gates=labels.float().detach() if passed else None
    return dict(top1_ids=top.detach(),valid=valid.detach(),binary_measurements=labels.detach(),
        p0=p0.detach(),p1=p1.detach(),native_gates=gates,measurement_passed=passed,
        invalid_rows=(~valid).nonzero(as_tuple=False).flatten().cpu().tolist())


class ParallelLocalBinaryGate(ParallelLocalSemanticGate):
    """Arm-independent original gate, fixed vector/scalar core mode, one probe.

    An invalid original argmax is saved before InvalidBinaryMeasurement is
    raised. It cannot become an applied zero gate. Binding that failed artifact
    also fails explicitly on forward, without another native measurement.
    Artifacts carry no core weights or branch mode and may be shared by arms.
    """
    def __init__(self,norm,head,core,*,n_local_rows,zero_token_id,one_token_id,
                 origin_identity,query_indices,stream_positions,origin_artifact=None,
                 capture=False,detach_captures=True):
        need(type(core) is ParallelLocalBinaryAggregation and core.mode in ('vector','scalar'),'Explicit binary vector/scalar core required')
        self.branch_mode=core.mode;self.argmax_calls=0;self.artifact_validation_argmax_calls=0
        super().__init__(norm,head,core,n_local_rows=n_local_rows,mode='native_gate',
            zero_token_id=zero_token_id,one_token_id=one_token_id,origin_identity=origin_identity,
            query_indices=query_indices,stream_positions=stream_positions,origin_artifact=origin_artifact,
            capture=capture,detach_captures=detach_captures)

    def reset(self,*args,**kwargs):
        need(not self.active,'Close the controller before explicit scene reset')
        self.argmax_calls=0;self.artifact_validation_argmax_calls=0;return super().reset(*args,**kwargs)

    def _mode_guard(self):
        need(self.mode=='native_gate','Binary controller must remain native_gate')
        need(self.core.mode==self.branch_mode,'Binary core.mode changed after controller construction')

    def bind_origin_artifact(self,artifact):
        need(not self.active and self._artifact is None,'Bind one original artifact before attaching')
        need(isinstance(artifact,dict) and artifact.get('schema_version')==1 and artifact.get('gate_rule')==BINARY_GATE_RULE
             and artifact.get('origin_identity')==self._origin and artifact.get('measurement_mode')=='binary_origin'
             and artifact.get('zero_token_id')==self.zero_token_id and artifact.get('one_token_id')==self.one_token_id
             and artifact.get('n_local_rows')==self.n_local_rows and 'branch_mode' not in artifact,'Wrong native binary origin identity')
        need(artifact.get('artifact_sha256')==artifact_digest(artifact),'Original artifact content changed')
        n,h=self.n_local_rows,self.core.hidden_size
        for key in ('origin_hidden','origin_normalized'):
            value=artifact[key];need(isinstance(value,torch.Tensor) and value.shape==(n+1,1,h) and value.dtype==torch.float16
                and not value.requires_grad and bool(torch.isfinite(value).all()),'Invalid native origin state')
        logits=artifact['origin_logits'];need(isinstance(logits,torch.Tensor) and logits.shape[:2]==(n+1,1) and logits.ndim==3
            and logits.dtype==torch.float16 and not logits.requires_grad and bool(torch.isfinite(logits).all()),'Invalid native origin logits')
        need(max(self.zero_token_id,self.one_token_id)<logits.shape[-1],'Origin vocabulary differs')
        self.artifact_validation_argmax_calls+=1
        top=logits[:n,0].argmax(-1);valid=(top==self.zero_token_id)|(top==self.one_token_id)
        labels=torch.full_like(top,-1,dtype=torch.int8);labels[top==self.zero_token_id]=0;labels[top==self.one_token_id]=1
        for key,value in (('top1_ids',top),('valid',valid),('binary_measurements',labels)):
            need(artifact[key].dtype==value.dtype and artifact[key].shape==value.shape and not artifact[key].requires_grad
                 and torch.equal(artifact[key],value),'Stored binary measurements differ from full-vocabulary argmax')
        passed=bool(valid.all());need(type(artifact['measurement_passed']) is bool and artifact['measurement_passed']==passed
            and artifact['invalid_rows']==(~valid).nonzero(as_tuple=False).flatten().cpu().tolist(),'Measurement failure status differs')
        if passed:need(isinstance(artifact['native_gates'],torch.Tensor) and artifact['native_gates'].dtype==torch.float32
            and not artifact['native_gates'].requires_grad and torch.equal(artifact['native_gates'],labels.float()),'Binary gates differ from actual argmax')
        else:need(artifact['native_gates'] is None,'Invalid measurements must not become applied gates')
        p0,p1=artifact['p0'],artifact['p1'];need(all(isinstance(x,torch.Tensor) and x.shape==(n,) and x.dtype==torch.float64
            and not x.requires_grad and bool(torch.isfinite(x).all()) for x in (p0,p1))
            and bool(((p0>=0)&(p1>=0)&(p0+p1<=1+1e-12)).all()),'Diagnostic full-vocabulary probabilities malformed')
        origin=self._origin['prompt_width']-1
        need(artifact['origin_query_index']==artifact['origin_stream_position']==origin and artifact['probe_shape']==[n+1,1,h],'Original query position differs')
        self._artifact={k:normal_copy(v) if isinstance(v,torch.Tensor) else json_copy(v) for k,v in artifact.items()}
        self._artifact_source='bound_artifact';return self

    def _probe_origin(self,hidden):
        origin=self._origin['prompt_width']-1
        need(hidden.shape[1]==self._origin['prompt_width'] and self.query_indices==self.stream_positions==(origin,),
             'Later/full-prefix execution requires a bound original gate artifact')
        with torch.no_grad(),torch.autocast(device_type=hidden.device.type,enabled=False):
            queries=hidden[:,origin:origin+1].detach().contiguous()
            self.probe_norm_calls+=1;normalized=self.norm.forward(queries)
            need(normalized.shape==queries.shape and normalized.dtype==torch.float16,'Native origin norm changed shape/dtype')
            self.probe_head_calls+=1;logits=self.head.forward(normalized)
            self.probability_calls+=1;self.argmax_calls+=1
            measured=binary_measurement(logits,self.zero_token_id,self.one_token_id,self.n_local_rows)
        self._artifact=dict(schema_version=1,gate_rule=BINARY_GATE_RULE,measurement_mode='binary_origin',origin_identity=self.origin_identity,
            zero_token_id=self.zero_token_id,one_token_id=self.one_token_id,n_local_rows=self.n_local_rows,
            origin_query_index=origin,origin_stream_position=origin,probe_shape=list(queries.shape),
            origin_hidden=queries.detach().clone(),origin_normalized=normalized.detach().clone(),origin_logits=logits.detach().clone(),**measured)
        self._artifact_source='native_prefill'
        if not measured['measurement_passed']:raise InvalidBinaryMeasurement(self.export_origin_artifact(cpu=True))

    def _before_norm(self,module,args,kwargs):
        self._mode_guard()
        if self._artifact is not None and not self._artifact['measurement_passed']:
            raise InvalidBinaryMeasurement(self.export_origin_artifact(cpu=True))
        result=super()._before_norm(module,args,kwargs)
        if self.last_capture is not None:
            self.last_capture.update(branch_mode=self.branch_mode,measurement_rule=BINARY_GATE_RULE,argmax_calls=self.argmax_calls,artifact_validation_argmax_calls=self.artifact_validation_argmax_calls)
        return result

    def assert_complete(self):
        self._mode_guard();status=super().assert_complete();need(self._artifact['measurement_passed'],'Cannot complete an invalid measurement')
        expected=1 if self._artifact_source=='native_prefill' else 0;need(self.argmax_calls==expected,'Original argmax must execute once or zero on bound replay')
        return dict(status,branch_mode=self.branch_mode,argmax_calls=self.argmax_calls,
            artifact_validation_argmax_calls=self.artifact_validation_argmax_calls,measurement_passed=True)

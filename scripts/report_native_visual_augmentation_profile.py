"""Independent CPU evidence audit of the intact-input augmentation software profile.

No decoder, vision, backward, optimizer or learned-core forward is executed.
Native FP16 norm/head and independent FP64 formulas replay saved observations.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import profile_native_visual_augmentation as producer
from scripts import report_mmred_official_native_memory as reference
from scripts import report_mmred_official_native_training as training_audit
from scripts import native_visual_augmentation_runtime as runtime
p=reference.p
need=reference.need;sha=reference.sha;read=reference.read;save=reference.save;object_sha=reference.object_sha
PROTOCOL='native_visual_augmentation_profile_independent_audit'
PROPOSAL='docs/paper/NATIVE_VISUAL_AUGMENTATION_PROFILE_AUDIT.md'
PROPOSAL_SHA='PENDING'
OWN=('scripts/report_native_visual_augmentation_profile.py',PROPOSAL,'slurm/native_visual_augmentation_profile_report.sbatch')
OUT=producer.OUT
DATA=Path('/mnt/data/gabriele/gnn_transformer/native_visual_augmentation_audit')
INDICES=(0,5,800,810,1600,1612,2400,2405,2410,3203,3210,3222)
ARMS=('summary','pointwise')
POLICY=dict(cpu_seconds=900,cpu_cores=4,memory_gib=16,publication_reserve_seconds=30,
    cpu_native_tv_max=.02,cpu_argmax_gate=False,core_atol=2e-4,core_rtol=2e-4,
    ce_absolute_tolerance=2e-6,maximum_native_head_calls=681,maximum_native_head_rows=3000,
    gradient_vector_witnesses_per_arm=2,full_decoder_cpu_replay=False,
    model_calls=0,vision_calls=0,decoder_calls=0,backward_calls=0,optimizer_steps=0,
    native_parameter_training=False,no_efficacy_gate=True,no_fit_release=True,
    collect_scheduled_numerical_results_before_gate=True)


def sources():
    need(sha(REPO/PROPOSAL)==PROPOSAL_SHA,'Audit proposal changed')
    return {name:sha(REPO/name) for name in OWN}


def number(value):
    value=float(value);return value if math.isfinite(value) else None


def safe_tree(torch,value):
    if isinstance(value,torch.Tensor):
        need(value.device.type=='cpu' and not value.requires_grad,'Saved evidence must be detached CPU tensors')
    elif isinstance(value,dict):
        for k,v in value.items():need(isinstance(k,(str,int)),'Unexpected packet key');safe_tree(torch,v)
    elif isinstance(value,(tuple,list)):
        for v in value:safe_tree(torch,v)
    else:need(value is None or type(value) in (str,int,float,bool),'Live object in saved evidence')


class Audit:
    def __init__(self,torch,out,data,started):
        self.torch=torch;self.out=out;self.data=data;self.started=started;self.bindings={};self.manifest={}
        self.results=[];self.head_calls=0;self.head_rows=0;self.core_calls=0;self.head_seconds=0.;self.core_seconds=0.
        self.pool_cache={};self.counts=Counter();self.raw_seen=set();self.modules=None
    def guard(self):need(time.perf_counter()-self.started<870,'CPU audit reached its publication reserve')
    def bind(self,file,expected=None):return reference.bind(file,self.bindings,expected)
    def record(self,file,expected=None):self.bind(file,expected);return read(file)
    def load(self,descriptor):
        self.guard();file=str(Path(descriptor['file']).resolve());digest=descriptor['sha256']
        need(file in self.manifest and self.manifest[file]==digest,'Raw packet missing from producer artifact manifest')
        self.bind(file,digest);value=self.torch.load(file,map_location='cpu',weights_only=True);safe_tree(self.torch,value)
        self.raw_seen.add(file);return value
    def add(self,key,passed,**details):
        row=dict(key=key,passed=bool(passed),**details);self.results.append(row);return row
    def exact(self,key,a,b):return self.add(key,self.torch.equal(a,b))
    def close(self,key,got,want,atol=None,rtol=None):
        torch=self.torch;need(got.shape==want.shape,'Tensor ownership/shape differs: '+key)
        atol=POLICY['core_atol'] if atol is None else atol;rtol=POLICY['core_rtol'] if rtol is None else rtol
        finite=bool(got.isfinite().all() and want.isfinite().all());difference=(got.double()-want.double()).abs()
        return self.add(key,finite and bool((difference<=atol+rtol*want.double().abs()).all()),finite=finite,
            max_absolute_error=number(difference.max()),atol=atol,rtol=rtol)
    def head(self,key,cap):
        self.guard();tick=time.perf_counter();result=reference.head_audit(self.torch,cap,self.modules,self.data,key)
        self.head_seconds+=time.perf_counter()-tick;self.head_calls+=1;self.head_rows+=result['rows']
        need(self.head_calls<=POLICY['maximum_native_head_calls'] and self.head_rows<=POLICY['maximum_native_head_rows'],'CPU head scope exceeded')
        self.add(key, result.pop('passed'),**result);return cap['head_logits']


def shapes(arm):
    common={'alpha':()}
    if arm=='pointwise':return dict(common,down_weight=(321,3584),up_weight=(3584,321))
    return {**common,'pool.key_weight':(128,3608),'pool.queries':(32,128),'pool.mass_direction':(3584,),
        'query_weight':(128,3584),'key_weight':(128,3584),'value_weight':(128,3584),'output_weight':(3584,128)}


def state_check(torch,state,arm):
    want=shapes(arm);need(set(state)==set(want),'Exact augmentation parameter inventory differs')
    for name,shape in want.items():
        v=state[name];need(tuple(v.shape)==shape and v.dtype==torch.float32 and not v.requires_grad and bool(v.isfinite().all()),'Finite FP32 state differs: '+name)
    live=[k for k in state if k!='pool.mass_direction']
    need(sum(state[k].numel() for k in live)==2300929,'Matched live parameter count differs')
    if arm=='summary':need(int(torch.count_nonzero(state['pool.mass_direction']))==0,'Frozen unused mass direction changed')
    return live


def fresh_state(torch,arm):
    state={'alpha':torch.zeros((),dtype=torch.float32)}
    if arm=='summary':
        generator=torch.Generator(device='cpu').manual_seed(24);bound=1/math.sqrt(3608)
        state['pool.key_weight']=torch.empty((128,3608),dtype=torch.float32).uniform_(-bound,bound,generator=generator)
        state['pool.queries']=torch.randn((32,128),dtype=torch.float32,generator=generator)
        state['pool.mass_direction']=torch.zeros(3584,dtype=torch.float32)
    generator=torch.Generator(device='cpu').manual_seed(25)
    names=('query_weight','key_weight','value_weight','output_weight') if arm=='summary' else ('down_weight','up_weight')
    for name in names:
        size=shapes(arm)[name];bound=1/math.sqrt(size[1]);state[name]=torch.empty(size,dtype=torch.float32).uniform_(-bound,bound,generator=generator)
    return state


def functional_read(torch,hidden,memory,state,arm):
    """Independent FP64 formula, with no call to the augmentation classes."""
    h=hidden.double().reshape(-1,hidden.shape[-1]);hn=h/(h.square().mean(-1,keepdim=True)+1e-6).sqrt()
    if arm=='summary':
        m=memory.double();mn=m/(m.square().mean(-1,keepdim=True)+1e-6).sqrt()
        q=hn@state['query_weight'].double().T;k=mn@state['key_weight'].double().T;v=m@state['value_weight'].double().T
        scores=q@k.T/math.sqrt(q.shape[-1]);attention=scores.softmax(-1);mixed=attention@v
        readout=mixed@state['output_weight'].double().T
        details=dict(normalized_hidden=hn,normalized_memory=mn,query=q,keys=k,values=v,scores=scores,attention=attention,mixed_value=mixed)
    else:
        pre=hn@state['down_weight'].double().T;activation=pre*pre.sigmoid();readout=activation@state['up_weight'].double().T
        details=dict(normalized_hidden=hn,preactivation=pre,activation=activation)
    gate=state['alpha'].double().tanh();readout=readout.reshape(hidden.shape)
    return dict(details,gate=gate,readout=readout,delta=gate*readout)


def core_audit(a,key,ev,record,hidden,state,arm,case,features):
    if arm is None:
        need('core_capture' not in record and not record.get('native_addition_applied',False),'Ordinary call acquired an augmentation')
        return
    torch=a.torch;tick=time.perf_counter();state_check(torch,state,arm)
    need(ev['core_class']=='gnnformer.native_visual_augmentation.'+('SummaryReadResidual' if arm=='summary' else 'PointwiseResidual'),'Actual core class differs')
    observed=record['core_capture'];need(observed['arm']==arm and observed['seed']==25 and not record['delta_override'],'Actual residual owner differs')
    memory=None
    if arm=='summary':
        pool={k.removeprefix('pool.'):v for k,v in state.items() if k.startswith('pool.')}
        cache_key=(case['metadata']['sid'],object_sha(reference.state_identity(pool)))
        if cache_key not in a.pool_cache:
            a.pool_cache[cache_key]=reference.functional_memory(torch,features,case['coordinates'],pool,False)
        want=a.pool_cache[cache_key];got=ev['memory_capture'];need(set(got)==set(want),'Pool evidence inventory differs')
        for name,v in want.items():a.close(key+'_pool_'+name,got[name],v)
        a.exact(key+'_normalized_identity',got['tokens'],got['normalized_value'])
        need(int(torch.count_nonzero(got['mass_feature']))==0,'Normalized pool acquired mass feature')
        memory=want['tokens']
    else:need(ev['memory_capture'] is None,'Pointwise branch acquired visual memory')
    want=functional_read(torch,hidden,memory,state,arm)
    need(set(observed)==set(want)|{'arm','seed'},'Complete actual reader capture required')
    for name,v in want.items():
        need(observed[name].dtype==torch.float32,'Core arithmetic escaped FP32')
        a.close(key+'_'+name,observed[name],v)
    a.exact(key+'_delta_owner',record['delta'],observed['delta'])
    fused=hidden+record['delta'].to(hidden.dtype)
    a.exact(key+'_native_cast_add',record['fused_hidden_rows'],fused)
    a.core_calls+=1;a.core_seconds+=time.perf_counter()-tick
    return dict(alpha=float(state['alpha']),gate=float(state['alpha'].double().tanh()),delta_norm=float(record['delta'].double().norm()))


def ce_audit(a,key,logits,ev,targets,loss=None):
    torch=a.torch;need(logits.shape==(1,len(targets),152064) and ev['target_ids']==targets,'Complete selected target/logit ownership differs')
    got=ev['position_ce'];need(got.shape==(len(targets),),'All teacher position losses required')
    expected=logits[0].double().logsumexp(-1)-logits[0].double().gather(1,torch.tensor(targets)[:,None])[:,0]
    a.close(key+'_position_ce',got,expected,atol=2e-6,rtol=0.)
    a.add(key+'_mean_ce',abs(float(ev['loss'])-float(expected.mean()))<=2e-6,observed=float(ev['loss']),expected=float(expected.mean()))
    if loss is not None:a.add(key+'_loss_owner',float(loss)==float(ev['loss']))


def input_identity(a,ev,case,features):
    want={'features':p.tensor_info(features),**{k:p.tensor_info(v) for k,v in case['coordinates'].items()}}
    need(ev['memory_input_identity']==want,'Core/intact-native input ownership differs')
    need(ev['checkpoint_identity']==a.config['ordinary_checkpoint'],'Raw native checkpoint differs')


def teacher(a,key,raw,case,features,packet,state=None,arm=None,mode='full',targets=None):
    torch=a.torch;ev=raw['evidence'];targets=list(case['target_ids'] if targets is None else targets)
    W=case['metadata']['prompt_width'];L=len(targets);T=W+L-1
    input_identity(a,ev,case,features);need(ev['hooks_removed'] and ev['final_block_calls']==1,'Actual hook/final-block lifecycle differs')
    need(len(ev['profile_head'])==len(ev['shapes'])==1,'Exactly one selected native head per teacher')
    if mode=='full':
        need(ev['counters']==dict(model=1,visual=0,language=1,norm=1,head=1,prefix_decoder=0)
            and ev['rope_restored'] and ev['augmentation_hook_removed'] and ev['teacher_use_cache'],'Full native teacher call inventory differs')
        need(ev['prompt_width']==W and ev['target_positions']==list(range(W-1,T)),'Teacher target positions differ')
        positions=case['teacher_position_ids'] if targets==case['target_ids'] else runtime.history_case(torch,case,targets)['teacher_position_ids']
        need(len(ev['native_inputs'])==len(ev['native_positions'])==len(ev['augmentation_calls'])==1,'One whole-stream teacher record required')
        observed=ev['native_inputs'][0]
        need(observed['input_ids'] is None and observed['inputs_embeds_identity']['shape']==[1,T,3584]
            and observed['inputs_embeds_identity']['dtype']=='torch.float16' and not observed['has_pixels'] and observed['past_length']==0
            and torch.equal(observed['cache_position'],torch.arange(T)) and torch.equal(observed['attention_mask'],torch.ones(1,T,dtype=torch.long)),'Exact native teacher prefix/history differs')
        a.exact(key+'_positions',ev['native_positions'][0],positions);a.exact(key+'_expected_positions',ev['expected_positions'][0],positions)
        record=ev['augmentation_calls'][0];need(record['placement']=='pre_block27' and record['threshold']==W-1
            and record['native_input_shape']==[1,T,3584] and record['write_positions'].tolist()==list(range(W-1,T)),'Write placement differs')
        hidden=record['lower_hidden_rows'];a.counts['model']+=1
    else:
        need(ev['counters']==dict(model=0,visual=0,language=0,norm=1,head=1,prefix_decoder=0)
            and ev['reference_full']==(mode=='reference') and ev['packet_identity']==packet['packet_identity']
            and ev['all_written_suffix_kv_live'] and ev['prefix_length']==W-1,'Selected-block replay/cache ownership differs')
        begin=0 if mode=='reference' else W-1;position=packet['position_ids'][:,:,begin:];cache_position=torch.arange(begin,T)
        a.exact(key+'_positions',ev['position_ids'],position);a.exact(key+'_cache_position',ev['cache_position'],cache_position)
        a.exact(key+'_observed_cache_position',ev['actual_cache_position'],cache_position)
        a.exact(key+'_text_positions',ev['actual_text_position_ids'],position[0])
        need(ev['actual_block_input_shape']==[1,T if mode=='reference' else L,3584],'Actual replay block width differs')
        if mode=='prefix':
            want=(torch.arange(T)[None,:]<=cache_position[:,None])[None,None,:,:]
            a.exact(key+'_causal_mask',ev['actual_attention_mask'],want);a.exact(key+'_declared_mask',ev['attention_mask'],want)
            for k in ('key','value'):need(ev['suffix_kv_identity'][k]['shape']==[1,4,L,128] and ev['suffix_kv_identity'][k]['dtype']=='torch.float16','Live suffix KV geometry differs')
        else:
            actual=ev['actual_attention_mask'];want=packet['full_attention_mask']
            need((actual is None)==(want is None),'Reference mask optionality differs')
            if actual is not None:a.exact(key+'_reference_mask',actual,want)
        record=ev;hidden=packet['suffix_hidden'];a.counts['replay']+=1
    need(hidden.shape==(1,L,3584) and hidden.dtype==torch.float16,'Actual written hidden rows differ')
    if not record.get('delta_override',False):core_audit(a,key,ev,record,hidden,state,arm,case,features)
    cap=ev['profile_head'][0];a.exact(key+'_raw_head_owner',raw['logits'],cap['head_logits'])
    a.head(key,cap);ce_audit(a,key,raw['logits'],ev,targets,raw.get('loss'))
    norm_width=T if mode in ('full','reference') else L;shape=ev['shapes'][0]
    need(shape==dict(norm_input_shape=[1,norm_width,3584],norm_output_shape=[1,norm_width,3584],
        head_input_shape=[1,L,3584],head_output_shape=[1,L,152064]),'Actual full norm/selected head shape differs')
    return raw['logits'],float(ev['loss'])

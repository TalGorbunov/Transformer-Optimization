"""CPU Slurm software tests for live-prefix null-reference fusion, not efficacy."""
from __future__ import annotations
import os
from pathlib import Path
import sys
import unittest
if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION')!='cpu' or os.environ.get('SLURM_JOB_GPUS'):
    raise SystemExit('Run reference-controller numerical tests in CPU Slurm')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
from gnnformer.parallel_local_reference import ParallelLocalReference,orthogonal_directions


class Norm(nn.Module):
    def __init__(self):
        super().__init__();self.weight=nn.Parameter(torch.linspace(.8,1.2,8),requires_grad=False)
    def forward(self,hidden_states):return hidden_states*self.weight.to(hidden_states.dtype)


class CausalNative(nn.Module):
    """Frozen causal per-row prefix mean and a cache, before the final norm."""
    def __init__(self):
        super().__init__();self.norm=Norm();self.head=nn.Linear(8,11,bias=False);self.requires_grad_(False)
    def forward(self,h,cache=None,keyword=False):
        past=0 if cache is None else cache.shape[1]
        full=h if cache is None else torch.cat((cache,h),1)
        hidden=full.cumsum(1)/torch.arange(1,full.shape[1]+1,dtype=full.dtype).reshape(1,-1,1)
        hidden=hidden[:,past:]
        result=self.norm(hidden_states=hidden) if keyword else self.norm(hidden)
        return self.head(result),full,hidden


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261101);self.model=CausalNative().eval();self.branch=ParallelLocalAggregation(8,rank=4)
        self.h=torch.randn(7,4,8) # three actual, three reference, one global
    def activate(self):
        with torch.no_grad():self.branch.up.weight.normal_(0,.2)
    def controller(self,mode,**kwargs):
        args=dict(n_actual_rows=3,n_reference_rows=3,mode=mode,anchor_n=2,
                  sham_key=['fixed_model','fixed_question'],query_indices=[2,3],stream_positions=[2,3])
        args.update(kwargs);return ParallelLocalReference(self.model.norm,self.branch,**args)

    def test_zero_identity_all_modes_native_cache_and_no_model_registration(self):
        original=self.model(self.h);saved={k:v.clone() for k,v in self.model.state_dict().items()}
        for mode in ('base','background','sham'):
            with self.controller(mode) as controller:
                self.model.eval();self.assertTrue(self.branch.training)
                self.assertFalse(any(m is self.branch for m in self.model.modules()))
                output,cache,_=self.model(self.h,keyword=True)
                self.assertTrue(torch.equal(output,original[0]));self.assertTrue(torch.equal(cache,original[1]))
                self.assertEqual((controller.calls,controller.actual_reads,controller.reference_reads),(1,1,1))
            self.assertTrue(torch.equal(self.model(self.h)[0],original[0]))
        for k,v in self.model.state_dict().items():self.assertTrue(torch.equal(v,saved[k]))

    def test_baseline_equals_existing_core_on_actual_rows_and_leaves_others(self):
        self.activate();original=self.h.clone();native=self.model(self.h);hidden=native[2]
        expected=hidden.clone();expected[-1,2:]+=self.branch(hidden[:3,2:],hidden[-1,2:],output_dtype=hidden.dtype)
        with self.controller('base',capture=True) as controller:
            actual,cache,_=self.model(self.h)
            wanted=self.model.head(self.model.norm.forward(expected))
            self.assertTrue(torch.equal(actual,wanted));self.assertTrue(torch.equal(actual[:-1],native[0][:-1]))
            self.assertTrue(torch.equal(actual[-1,:2],native[0][-1,:2]));self.assertTrue(torch.equal(cache,native[1]))
            self.assertTrue(torch.equal(controller.last_capture['delta'],self.branch(hidden[:3,2:],hidden[-1,2:],output_dtype=torch.float32)))
        self.assertTrue(torch.equal(self.h,original))

    def test_background_exact_formula_reference_mean_and_sign(self):
        self.activate()
        with self.controller('background',capture=True) as controller:
            self.model(self.h);c=controller.last_capture
            self.assertTrue(torch.equal(c['reference_mean'],c['reference_messages'].double().mean(0).float()))
            b=torch.nn.functional.linear(c['reference_mean'],self.branch.aggregate_projection.weight)
            self.assertTrue(torch.equal(b,c['background_direction']))
            self.assertTrue(torch.equal(c['changed_preactivation'],c['preactivation']-b))
            self.assertTrue(torch.equal(c['delta'],self.branch.up(torch.nn.functional.silu(c['preactivation']-b))))

    def test_anchor_identity_every_mode(self):
        self.activate();outputs=[]
        for mode in ('base','background','sham'):
            with self.controller(mode,anchor_n=3):outputs.append(self.model(self.h)[0])
        self.assertTrue(all(torch.equal(outputs[0],x) for x in outputs[1:]))

    def test_orthogonal_zero_parallel_and_private_rng(self):
        state=torch.random.get_rng_state().clone()
        first=self.controller('sham');second=self.controller('sham')
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
        self.assertTrue(torch.equal(first._noise,second._noise))
        b=torch.tensor([[1.,2.,3.,4.],[0.,0.,0.,0.],[2.,4.,6.,8.]])
        result=orthogonal_directions(b,torch.tensor([1.,2.,3.,4.]))
        self.assertTrue(torch.equal(result[1],torch.zeros(4)))
        torch.testing.assert_close((b.double()*result.double()).sum(-1),torch.zeros(3,dtype=torch.float64),atol=2e-6,rtol=0)
        torch.testing.assert_close(b.double().norm(dim=-1),result.double().norm(dim=-1),atol=2e-6,rtol=1e-6)

    def test_reference_permutation_and_duplicate_mean_invariance(self):
        self.activate()
        with self.controller('background'):
            baseline=self.model(self.h)[0][-1]
            permuted=self.h.clone();permuted[3:6]=self.h[torch.tensor([5,3,4])]
            actual=self.model(permuted)[0][-1]
        torch.testing.assert_close(actual,baseline,rtol=2e-6,atol=2e-6)
        doubled=torch.cat((self.h[:3],self.h[3:6],self.h[3:6],self.h[-1:]),0)
        with self.controller('background',n_reference_rows=6):actual=self.model(doubled)[0][-1]
        torch.testing.assert_close(actual,baseline,rtol=2e-6,atol=2e-6)

    def test_live_prefix_changes_reference_mean_and_cached_equals_full(self):
        self.activate()
        for mode in ('base','background','sham'):
            with self.controller(mode,capture=True) as controller:
                full,full_cache,_=self.model(self.h)
                full_mean=controller.last_capture['reference_mean'].clone()
                controller.configure_queries([2],[2]);prefill,cache,_=self.model(self.h[:,:3])
                first_mean=controller.last_capture['reference_mean'].clone()
                controller.configure_queries([0],[3]);decoded,cache,_=self.model(self.h[:,3:],cache)
                self.assertFalse(torch.equal(first_mean,controller.last_capture['reference_mean']))
                torch.testing.assert_close(controller.last_capture['reference_mean'],full_mean[-1:],rtol=2e-6,atol=2e-6)
                torch.testing.assert_close(torch.cat((prefill,decoded),1),full,rtol=2e-6,atol=2e-6)
                self.assertTrue(torch.equal(cache,full_cache))

    def test_future_reference_tokens_cannot_change_previous_query(self):
        self.activate();changed=self.h.clone();changed[3:6,-1]+=15
        for mode in ('base','background','sham'):
            with self.controller(mode):
                a=self.model(self.h)[0];b=self.model(changed)[0]
            self.assertTrue(torch.equal(a[:,:3],b[:,:3]))
            if mode=='background':self.assertFalse(torch.equal(a[-1,-1],b[-1,-1]))
            if mode=='base':self.assertTrue(torch.equal(a[-1],b[-1]))

    def test_native_dtype_addition_and_detached_normal_exports(self):
        self.activate();self.model.to(dtype=torch.bfloat16);h=self.h.to(torch.bfloat16)
        with self.controller('background',capture=True) as controller,torch.inference_mode():
            self.model(h);c=controller.export_last_capture(cpu=True)
            self.assertEqual(c['delta'].dtype,torch.float32)
            self.assertTrue(torch.equal(c['fused_global'],c['native_global']+c['delta'].to(torch.bfloat16)))
            self.assertTrue(all(not x.is_inference() and not x.requires_grad for x in c.values() if isinstance(x,torch.Tensor)))

    def test_empty_actual_set_keeps_references_native_and_aggregate_zero(self):
        empty=self.h[3:].clone() # three reference streams and one global, no actual rows
        native=self.model(empty)
        for mode in ('base','background','sham'):
            with self.controller(mode,n_actual_rows=0,capture=True) as controller:
                output,cache,_=self.model(empty)
                self.assertTrue(torch.equal(output,native[0]))
                self.assertTrue(torch.equal(cache,native[1]))
                self.assertEqual(tuple(controller.last_capture['actual_messages'].shape),(0,2,4))
                self.assertTrue(torch.equal(controller.last_capture['actual_aggregate'],torch.zeros(2,4)))
        self.activate()
        for mode in ('base','background','sham'):
            with self.controller(mode,n_actual_rows=0,capture=True) as controller:
                output,cache,_=self.model(empty);c=controller.last_capture
                self.assertTrue(bool(torch.isfinite(output).all()))
                self.assertTrue(torch.equal(output[:-1],native[0][:-1]))
                self.assertTrue(torch.equal(output[-1,:2],native[0][-1,:2]))
                self.assertTrue(torch.equal(cache,native[1]))
                self.assertTrue(torch.equal(c['actual_aggregate'],torch.zeros(2,4)))
                expected=c['preactivation'] if mode=='base' else c['preactivation']+2*c['used_direction']
                self.assertTrue(torch.equal(c['changed_preactivation'],expected))

    def test_cleanup_layout_rejection_and_no_native_gradients(self):
        self.activate();controller=self.controller('background')
        with self.assertRaisesRegex(RuntimeError,'fixture'):
            with controller:
                self.model(self.h);raise RuntimeError('fixture')
        controller.close();self.assertFalse(controller.active);self.assertFalse(self.model.norm._forward_pre_hooks)
        with self.assertRaises(ValueError):self.controller('base',stream_positions=[2,5])
        with self.assertRaises(ValueError):
            with self.controller('background',n_reference_rows=2):self.model(self.h)
        with self.controller('background'):
            self.model(self.h)[0][-1,-1].square().sum().backward()
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in self.branch.parameters()))


if __name__=='__main__':unittest.main()

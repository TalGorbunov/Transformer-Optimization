"""Unexecuted CPU software tests; run only in an authorized Slurm CPU job."""
from pathlib import Path
import os
import sys
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
if __name__=='__main__' and (not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION')!='cpu'):
    raise SystemExit('CPU Slurm required; no numerical tests on the login node')
import torch
from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
from gnnformer.conditional_null_mean import ConditionalNullMean,projected_null_readout
from gnnformer.parallel_local_null_comparison import ParallelLocalNullComparison
from gnnformer.parallel_local_native import GlobalBroadcastLogitsProcessor


class NativeNorm(torch.nn.Module):
    def forward(self,hidden_states):return hidden_states*1.25


class TinyNativeFlow(torch.nn.Module):
    """Cache stores only pre-norm states, as a final-norm write cannot change it."""
    def __init__(self):super().__init__();self.norm=NativeNorm()
    def forward(self,hidden,cache):
        cache.append(hidden.detach().clone())
        return self.norm(hidden)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(831)
        self.core=ParallelLocalAggregation(hidden_size=8,rank=4)
        self.predictor=ConditionalNullMean(4);self.norm=NativeNorm()
        with torch.no_grad():
            self.core.up.weight.normal_(0,.1);self.predictor.fc2.weight.normal_(0,.1);self.predictor.fc2.bias.normal_(0,.1)
        self.h=torch.randn(27,4,8).half()

    def control(self,condition='centered',source='bank',indices=(3,),stream=(3,),norm=None,n=2,capture=True):
        return ParallelLocalNullComparison(norm or self.norm,self.core,self.predictor,n_actual_rows=n,
            condition=condition,mean_source=source,query_indices=indices,stream_positions=stream,capture=capture)

    def test_zero_up_is_native_for_both_means_and_coefficients(self):
        with torch.no_grad():self.core.up.weight.zero_()
        original=self.h.clone();native=self.norm(self.h)
        for condition in ('offset','centered'):
            for source in ('learned','bank'):
                with self.control(condition,source) as c:out=self.norm(self.h)
                self.assertTrue(torch.equal(out,native));self.assertTrue(torch.equal(self.h,original))
                self.assertEqual(c.calls,c.predictor_reads);self.assertEqual(c.calls,c.reference_reads)
                self.assertFalse(c.active)

    def test_same_native_inputs_only_global_queries_change_and_formula_matches(self):
        reads=[]
        for condition in ('offset','centered'):
            for source in ('learned','bank'):
                with self.control(condition,source,indices=(1,3),stream=(1,3)) as c:out=self.norm(hidden_states=self.h)
                v=c.export_last_capture();reads.append(v)
                expected=projected_null_readout(self.core,v['aggregate'],v['global_states'],v['used_mean'],
                    n_elements=2,mode=condition,output_dtype=torch.float32)
                self.assertTrue(torch.equal(v['delta'],expected['delta_float32']))
                self.assertTrue(torch.equal(v['bank_mean'],v['reference_messages'].double().mean(0).float()))
                self.assertTrue(torch.equal(v['fused_global'],v['native_global']+v['delta'].half()))
                self.assertTrue(torch.equal(out[:-1],self.norm(self.h)[:-1]))
                self.assertTrue(torch.equal(out[-1,[0,2]],self.norm(self.h)[-1,[0,2]]))
                self.assertEqual(v['coefficient'],2 if condition=='centered' else 1)
        for key in ('actual_states','reference_states','global_states','actual_messages','reference_messages','predicted_mean','bank_mean'):
            self.assertTrue(all(torch.equal(reads[0][key],r[key]) for r in reads[1:]))

    def test_reference_occurrences_are_not_deduplicated(self):
        self.h[3:26]=self.h[2:3];self.h[25]+=1
        with self.control() as c:self.norm(self.h)
        m=c.last_capture['reference_messages'];mu=c.last_capture['bank_mean']
        self.assertTrue(torch.equal(mu,m.double().mean(0).float()))
        self.assertFalse(torch.equal(mu,torch.stack((m[0],m[-1])).double().mean(0).float()))

    def test_full_and_cached_explicit_positions_match_and_mock_cache_is_unchanged(self):
        h=self.h.float();native=TinyNativeFlow();untouched=[];native(h,untouched)
        for source in ('learned','bank'):
            full_cache=[]
            with self.control(source=source,indices=(1,2,3),stream=(1,2,3),norm=native.norm):full=native(h,full_cache)
            self.assertTrue(torch.equal(full_cache[0],untouched[0]))
            cache=[]
            with self.control(source=source,indices=(1,),stream=(1,),norm=native.norm) as c:
                prefill=native(h[:,:2],cache)
                for t in (2,3):
                    c.configure_queries([0],[t]);out=native(h[:,t:t+1],cache)
                    torch.testing.assert_close(out[:,-1],full[:,t],atol=2e-5,rtol=2e-5)
                    self.assertEqual(c.last_capture['query_indices'],[0]);self.assertEqual(c.last_capture['stream_positions'],[t])
            torch.testing.assert_close(prefill[:,-1],full[:,1],atol=2e-5,rtol=2e-5)
            self.assertTrue(torch.equal(torch.cat(cache,1),h));self.assertFalse(c.active)

    def test_error_cleanup_reentry_and_no_model_registration(self):
        before=dict(self.norm.named_modules());c=self.control()
        with self.assertRaisesRegex(RuntimeError,'synthetic failure'):
            with c:
                self.norm(self.h)
                with self.assertRaises(ValueError):c.__enter__()
                raise RuntimeError('synthetic failure')
        self.assertFalse(c.active);self.assertEqual(len(self.norm._forward_pre_hooks),0)
        self.assertEqual(before,dict(self.norm.named_modules()))
        with c:self.norm(self.h)
        self.assertEqual(c.calls,1)
        exported=c.export_last_capture(cpu=True);exported['delta'].zero_()
        self.assertFalse(torch.equal(exported['delta'],c.last_capture['delta']))

    def test_empty_actual_centered_coefficient_and_invalid_layouts(self):
        h=self.h[:25]
        with self.control(n=0) as c:self.norm(h)
        self.assertEqual(c.last_capture['coefficient'],0)
        self.assertTrue(torch.equal(c.last_capture['aggregate'],torch.zeros_like(c.last_capture['aggregate'])))
        for q,s in (((1,1),(1,1)),((1,3),(1,4)),((2,),(1,))):
            with self.assertRaises(ValueError):self.control(indices=q,stream=s)
        with self.control() as c:
            with self.assertRaises(ValueError):self.norm(self.h[:-1])
        with self.assertRaises(ValueError):self.control(source='anchored')

    def test_shared_generated_special_prefix_is_not_filtered_or_mutated(self):
        from scripts import native_vision_null_comparison_runtime as runtime
        inputs=dict(input_ids=torch.tensor([[4,5],[6,7]]),attention_mask=torch.ones(2,2,dtype=torch.long))
        bundle=dict(inputs=inputs,row_inputs=[{k:v[i:i+1] for k,v in inputs.items()} for i in range(2)],
                    metadata=dict(prompt_width=2,original_prompt_width=2,prefix_ids=[]))
        with patch.object(runtime.reference,'validate_bundle',return_value=dict(passed=True)):
            appended=runtime.append_observed_prefix(bundle,[151652,1,151645])
        self.assertEqual(appended['inputs']['input_ids'][:,-3:].tolist(),[[151652,1,151645]]*2)
        self.assertEqual(bundle['inputs']['input_ids'].shape,(2,2));self.assertEqual(bundle['metadata']['prefix_ids'],[])
        self.assertEqual(appended['metadata']['prompt_width'],5)
        scores=torch.randn(2,11);original=scores.clone();ids=appended['inputs']['input_ids']
        p=GlobalBroadcastLogitsProcessor(n_local_rows=1,prompt_length=2);result=p(ids,scores)
        self.assertTrue(torch.equal(scores,original));self.assertTrue(torch.equal(result,original[-1:].expand_as(result)))
        bad=ids.clone();bad[0,-1]=0
        with self.assertRaises(ValueError):p(bad,scores)


if __name__=='__main__':unittest.main()

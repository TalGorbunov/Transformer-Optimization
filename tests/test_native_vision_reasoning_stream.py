"""CPU-only checks for global streaming, natural prefixes and storage bounds."""
import os
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
if not(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS')):
    raise SystemExit('All tensor tests require CPU Slurm')
import torch
from scripts.native_vision_reasoning_stream import GlobalLogitRecorder,prefixed_bundle,generation_policy,prompt
from gnnformer.parallel_local_native import GlobalBroadcastLogitsProcessor
from transformers import GenerationConfig

class StreamingTests(unittest.TestCase):
    def test_only_global_and_no_mutation(self):
        logits=torch.tensor([[[99.,0.,1.]],[[1.,2.,3.]]],dtype=torch.float16)
        original=logits.clone();output=SimpleNamespace(logits=logits)
        recorder=GlobalLogitRecorder(max_steps=8,full_vectors=True)
        self.assertIsNone(recorder(None,(),output));self.assertTrue(torch.equal(logits,original))
        self.assertEqual(recorder.records[0]['top1_token_id'],2)
        self.assertEqual(tuple(recorder.vectors[0].shape),(3,));self.assertEqual(recorder.vectors[0].dtype,torch.float32)
        logits[-1,-1,0]=100
        self.assertEqual(recorder.vectors[0][0],1)
    def test_compact_and_bound(self):
        recorder=GlobalLogitRecorder(max_steps=1)
        recorder(None,(),SimpleNamespace(logits=torch.ones(3,1,7)))
        self.assertEqual(recorder.vectors,[])
        with self.assertRaises(ValueError):recorder(None,(),SimpleNamespace(logits=torch.ones(3,1,7)))
        with self.assertRaises(ValueError):GlobalLogitRecorder(max_steps=9,full_vectors=True)
    def test_no_graph_or_alias(self):
        x=torch.randn(2,1,4,requires_grad=True);recorder=GlobalLogitRecorder(max_steps=1,full_vectors=True)
        recorder(None,(),SimpleNamespace(logits=x));self.assertFalse(recorder.vectors[0].requires_grad)
        self.assertIsNone(recorder.vectors[0].grad_fn)
    def test_sink_gets_copy(self):
        recorder=GlobalLogitRecorder(max_steps=1,sink=lambda row:row.update(top1_token_id=-1))
        recorder(None,(),SimpleNamespace(logits=torch.tensor([[[0.,1.]]])))
        self.assertEqual(recorder.records[0]['top1_token_id'],1)
    def test_special_prefix_is_verbatim(self):
        row=dict(input_ids=torch.tensor([[1,2]]),attention_mask=torch.ones(1,2,dtype=torch.long))
        bundle=dict(inputs=row,row_inputs=[row],metadata=dict(prefix_ids=[],prompt_width=2))
        copy=prefixed_bundle(bundle,[151667,151668])
        self.assertEqual(copy['inputs']['input_ids'].tolist(),[[1,2,151667,151668]])
        self.assertEqual(bundle['inputs']['input_ids'].tolist(),[[1,2]])
        self.assertEqual(copy['metadata']['prefix_ids'],[151667,151668])
    def test_broadcast_preserves_raw_and_rejects_divergence(self):
        hook=GlobalBroadcastLogitsProcessor(n_local_rows=1,prompt_length=2)
        ids=torch.tensor([[1,2,8],[3,4,8]]);scores=torch.tensor([[8.,0.],[0.,8.]])
        original=scores.clone();result=hook(ids,scores)
        self.assertTrue(torch.equal(scores,original));self.assertEqual(result.argmax(-1).tolist(),[1,1])
        ids[0,-1]=7
        with self.assertRaises(ValueError):hook(ids,scores)
    def test_policy_no_hf_retention(self):
        model=SimpleNamespace(generation_config=GenerationConfig(eos_token_id=[151645,151643],bos_token_id=151643))
        tokenizer=SimpleNamespace(eos_token_id=151645,pad_token_id=151643)
        config,policy=generation_policy(model,tokenizer,8)
        self.assertFalse(config.output_logits);self.assertFalse(config.output_scores);self.assertTrue(config.use_cache)
        self.assertEqual(config.max_new_tokens,8);self.assertEqual(config.repetition_penalty,1.)
        self.assertNotIn('Output only the integer',prompt('Where is Alice?'))

if __name__=='__main__':unittest.main()

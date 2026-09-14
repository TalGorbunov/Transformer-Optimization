"""CPU-Slurm tests using the installed actual tiny Qwen final decoder block."""
from pathlib import Path
import os
import sys
import unittest

if not (os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION') == 'cpu'
        and not os.environ.get('SLURM_JOB_GPUS')):
    raise RuntimeError('Run this tensor/model test only through CPU Slurm')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLDecoderLayer, Qwen2_5_VLRotaryEmbedding, Qwen2RMSNorm)
from transformers.masking_utils import create_causal_mask
from scripts.native_vision_v11_last_block import replay_last_block, reconstruct_global_sequence


class LastBlockReplayTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        torch.set_num_threads(2)

    def fixture(self, *, batch=1, length=7, dtype=torch.float32):
        config = Qwen2_5_VLTextConfig(vocab_size=23, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=128, use_sliding_window=False,
            layer_types=['full_attention', 'full_attention'], attention_dropout=0.,
            rope_scaling={'rope_type':'default', 'mrope_section':[1, 1, 2]})
        config._attn_implementation = 'sdpa'
        modules = (Qwen2_5_VLDecoderLayer(config, layer_idx=1),
                   Qwen2RMSNorm(32, eps=config.rms_norm_eps), nn.Linear(32, 23, bias=False),
                   Qwen2_5_VLRotaryEmbedding(config=config))
        for module in modules:
            module.eval().to(dtype=dtype)
            for parameter in module.parameters(): parameter.requires_grad_(False)
        hidden = torch.randn(batch, length, 32).to(dtype)
        ids = torch.arange(length).expand(batch, -1).clone()
        mask = torch.ones(batch, length, dtype=torch.long)
        position = torch.arange(length).view(1, 1, -1).expand(3, batch, -1).clone()
        return modules, dict(hidden_states=hidden, input_ids=ids, attention_mask=mask,
                             position_ids=position)

    def replay(self, modules, values, deltas, placement, indices=None):
        return replay_last_block(*modules, **values, deltas=deltas, placement=placement,
                                 query_indices=indices if indices is not None else [[3, 4, 5]])

    def test_zero_write_matches_actual_unmodified_decoder_and_inputs(self):
        modules, values = self.fixture(); originals={k:v.clone() for k,v in values.items()}
        states=[{k:v.clone() for k,v in m.state_dict().items()} for m in modules]
        block, norm, head, rotary = modules; x=values['hidden_states']
        cp=torch.arange(x.shape[1]); mask=create_causal_mask(config=block.self_attn.config,
            input_embeds=x, attention_mask=values['attention_mask'],cache_position=cp,
            past_key_values=None,position_ids=None)
        raw=block(hidden_states=x,attention_mask=mask,position_ids=None,past_key_values=None,
            use_cache=False,output_attentions=False,cache_position=cp,
            position_embeddings=rotary(x,values['position_ids']))[0]
        expected=head(norm(raw[0,[3,4,5]]))
        for placement in ('pre_last','post_last'):
            delta=torch.zeros(3,32,requires_grad=True)
            result=self.replay(modules,values,delta,placement)
            self.assertTrue(torch.equal(result['logits'],expected))
            self.assertTrue(torch.equal(result['block_output'],raw))
            self.assertFalse(result['use_cache'])
            result['logits'].square().mean().backward()
            self.assertIsNotNone(delta.grad);self.assertTrue(torch.isfinite(delta.grad).all())
            for module, state in zip(modules,states):
                self.assertTrue(all(torch.equal(state[k],v) for k,v in module.state_dict().items()))
                self.assertTrue(all(p.grad is None and not p.requires_grad for p in module.parameters()))
        for key in values:self.assertTrue(torch.equal(values[key],originals[key]))

    def test_second_numeral_ce_reaches_earlier_delta_only_before_last_block(self):
        modules, values = self.fixture()
        gradients={}
        for placement in ('pre_last','post_last'):
            delta=(torch.randn(3,32)*.1).requires_grad_()
            result=self.replay(modules,values,delta,placement)
            loss=F.cross_entropy(result['logits'][1:2],torch.tensor([7]))
            gradient=torch.autograd.grad(loss,delta)[0];gradients[placement]=gradient
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient[1].norm()),1e-7)
            self.assertTrue(torch.equal(gradient[2],torch.zeros_like(gradient[2])))
        self.assertGreater(float(gradients['pre_last'][0].norm()),1e-7)
        self.assertTrue(torch.equal(gradients['post_last'][0],torch.zeros(32)))

    def test_future_write_cannot_change_earlier_prediction(self):
        modules, values = self.fixture(); base=torch.randn(3,32)*.1
        changed=base.clone();changed[2]+=torch.linspace(-2,2,32)
        for placement in ('pre_last','post_last'):
            a=self.replay(modules,values,base,placement)['logits']
            b=self.replay(modules,values,changed,placement)['logits']
            self.assertTrue(torch.equal(a[:2],b[:2]))
            self.assertFalse(torch.equal(a[2],b[2]))

    def test_prior_write_changes_later_output_only_before_last_block(self):
        modules, values = self.fixture(); base=torch.randn(3,32)*.1
        changed=base.clone();changed[0]+=torch.linspace(-1,1,32)
        for placement in ('pre_last','post_last'):
            a=self.replay(modules,values,base,placement)['logits'][1]
            b=self.replay(modules,values,changed,placement)['logits'][1]
            self.assertEqual(torch.equal(a,b),placement=='post_last')

    def test_detached_fixed_history_preserves_values_but_cuts_temporal_gradient(self):
        modules, values = self.fixture()
        for placement in ('pre_last','post_last'):
            delta=(torch.randn(3,32)*.1).requires_grad_()
            full=self.replay(modules,values,delta,placement)
            fixed=torch.cat((delta[:1].detach(),delta[1:]),dim=0)
            replay=self.replay(modules,values,fixed,placement)
            self.assertTrue(torch.equal(full['logits'],replay['logits']))
            a=torch.autograd.grad(F.cross_entropy(full['logits'][1:2],torch.tensor([7])),delta)[0]
            b=torch.autograd.grad(F.cross_entropy(replay['logits'][1:2],torch.tensor([7])),delta)[0]
            self.assertTrue(torch.equal(b[0],torch.zeros_like(b[0])))
            self.assertEqual(bool((a[0]!=0).any()),placement=='pre_last')

    def test_complete_replay_agrees_with_each_causal_prefix(self):
        modules, values = self.fixture(); delta=torch.randn(3,32)*.1
        for placement in ('pre_last','post_last'):
            full=self.replay(modules,values,delta,placement)
            for t, end in enumerate([4,5,6]):
                prefix={k:(v[:,:end] if k!='position_ids' else v[:,:,:end]) for k,v in values.items()}
                replay=self.replay(modules,prefix,delta[:t+1],placement,[[3+i for i in range(t+1)]])
                torch.testing.assert_close(full['logits'][t],replay['logits'][-1],atol=2e-6,rtol=2e-5)

    def test_ragged_queries_empty_local_row_and_padding_are_respected(self):
        modules, values=self.fixture(batch=3)
        values['attention_mask'][1,:2]=0;values['attention_mask'][2,-2:]=0
        pos=values['attention_mask'].cumsum(-1)-1;pos.masked_fill_(values['attention_mask']==0,1)
        values['position_ids']=pos.unsqueeze(0).expand(4,-1,-1).clone()
        indices=[[],[3,4],[2,3,4]];delta=torch.randn(5,32)*.1
        changed={k:v.clone() for k,v in values.items()}
        changed['hidden_states'][~values['attention_mask'].bool()]+=100
        for placement in ('pre_last','post_last'):
            a=self.replay(modules,values,delta,placement,indices)
            b=self.replay(modules,changed,delta,placement,indices)
            self.assertEqual(a['offsets'],[0,0,2,5])
            self.assertEqual(a['flat_batch_indices'].tolist(),[1,1,2,2,2])
            torch.testing.assert_close(a['logits'],b['logits'],atol=0,rtol=0)
            self.assertTrue(torch.equal(a['final_norm_input'][0],a['block_output'][0]))

    def test_native_cast_before_add_is_preserved(self):
        modules, values=self.fixture(dtype=torch.float16)
        values['hidden_states'].fill_(1)
        delta=torch.full((3,32),.0004884,dtype=torch.float32,requires_grad=True)
        result=self.replay(modules,values,delta,'pre_last')
        actual=result['block_input'][0,[3,4,5]]
        wrong=(values['hidden_states'][0,[3,4,5]].float()+delta).half()
        self.assertTrue(torch.equal(actual,torch.ones_like(actual)))
        self.assertFalse(torch.equal(actual,wrong))
        F.cross_entropy(result['logits'][1:2].float(),torch.tensor([7])).backward()
        self.assertTrue(torch.isfinite(delta.grad).all())

    def test_reconstruction_uses_all_and_only_strict_target_prefixes(self):
        prompt=torch.randn(4,32);prompt_ids=torch.tensor([1,2,3,4])
        for target in ([6,22],[1,0,22]):
            states=torch.randn(len(target),32);states[0]=prompt[-1]
            prefixes=[list(target[:t]) for t in range(len(target))]
            result=reconstruct_global_sequence(prompt,prompt_ids,states,list(target),prefixes,eos_token_id=22)
            self.assertEqual(result['input_ids'].tolist(),prompt_ids.tolist()+list(target[:-1]))
            self.assertEqual(result['query_indices'],list(range(3,3+len(target))))
            self.assertTrue(torch.equal(result['hidden_states'][result['query_indices']],states))
            self.assertEqual(result['hidden_states'].shape[0],4+len(target)-1)
        with self.assertRaises(ValueError):
            reconstruct_global_sequence(prompt,prompt_ids,states,[1,0,22],[[],[1],[1,0,22]],eos_token_id=22)
        bad=states.clone();bad[0,0]+=1
        with self.assertRaises(ValueError):
            reconstruct_global_sequence(prompt,prompt_ids,bad,[1,0,22],[[],[1],[1,0]],eos_token_id=22)

    def test_rejects_invalid_queries_nonfinite_and_trainable_backbone(self):
        modules, values=self.fixture();delta=torch.zeros(3,32)
        for indices in ([[3,3,5]],[[5,4,3]],[[True,4,5]],[[]]):
            with self.assertRaises(ValueError):self.replay(modules,values,delta,'pre_last',indices)
        with self.assertRaises(ValueError):self.replay(modules,values,delta,'unknown')
        bad=delta.clone();bad[0,0]=float('nan')
        with self.assertRaises(ValueError):self.replay(modules,values,bad,'pre_last')
        with self.assertRaises(ValueError):self.replay(modules,values,delta.half(),'pre_last')
        values['attention_mask'][0,3]=0
        with self.assertRaises(ValueError):self.replay(modules,values,delta,'pre_last')
        values['attention_mask'][0,3]=1
        next(modules[0].parameters()).requires_grad_(True)
        with self.assertRaises(ValueError):self.replay(modules,values,delta,'pre_last')


if __name__=='__main__':
    unittest.main()

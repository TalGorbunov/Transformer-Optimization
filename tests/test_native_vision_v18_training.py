"""CPU Slurm tests for V18 ragged gate ownership, gradients and final scoring."""
from __future__ import annotations
from collections import Counter
import copy
import os
from pathlib import Path
import random
import sys
import unittest
if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION')!='cpu' or os.environ.get('SLURM_JOB_GPUS'):
    raise SystemExit('Run in a CPU Slurm allocation, without GPUs')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from gnnformer.parallel_local_semantic_aggregation import ParallelLocalSemanticAggregation
from gnnformer.paired_sequence_objectives import sequence_layout
from scripts import train_native_vision_v18 as train


class Tokenizer:
    all_special_ids=[100,151645,151643]
    tokens={0:'0',1:'1',2:'2',3:'3',6:'6',9:'9',20:' ',21:'\uff11',100:'<bos>',151645:'<eos>',151643:'<end>'}
    def decode(self,ids,skip_special_tokens=False):
        return ''.join(self.tokens[i] for i in ids if not skip_special_tokens or i not in self.all_special_ids)


class TrainingTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(91)

    @staticmethod
    def fixture():
        cache={'scenes':{}};gates={'scenes':{},'prefix_to_empty_feature_id':{}};values=[];index={};states=[];features={}
        def feature(name,value):
            if name not in features:features[name]=len(states);states.append(value)
            return name
        for sid,n,tokens,family in [('a',1,[1,7],'x'),('b',3,[1,7],'x'),('c',2,[1,0,7],'y'),('d',2,[1,0,7],'y')]:
            local=[];origins=[]
            for i in range(n):
                origin=f'{sid}_{i}_0';origins.append(origin);index[origin]=len(values);values.append(0. if i==0 else .3+i*.1)
                row=[]
                for t in range(len(tokens)):
                    fid=feature(f'{sid}_{i}_{t}',torch.tensor([i+1.,t+1.,.5,1.]));row.append(fid);gates['prefix_to_empty_feature_id'][fid]=origin
                local.append(row)
            gids=[feature(f'g_{family}_{t}',torch.tensor([1.,.5,t+1.,2.])) for t in range(len(tokens))]
            cache['scenes'][sid]=dict(n_frames=n,target_ids=tokens,local_feature_ids=local,global_feature_ids=gids)
            gates['scenes'][sid]={'local_empty_feature_ids':origins}
        return cache,gates,torch.tensor(values,dtype=torch.float32),index,torch.stack(states),features

    def test_ragged_origin_broadcast_and_padding_in_both_modes(self):
        cache,gates,values,gidx,states,index=self.fixture();sids=['a','b','c','d']
        for mode in train.POLICY['conditions']:
            h,g,layout,packed,origins=train.semantic_batch(torch,cache,states,index,sids,gates,values,gidx,mode)
            self.assertEqual(layout['offsets'],[0,2,4,7,10]);self.assertEqual(tuple(packed.shape),(3,10))
            self.assertEqual(tuple(h.shape),(3,10,4));self.assertFalse(packed.requires_grad)
            for j,sid in enumerate(sids):
                n=cache['scenes'][sid]['n_frames'];a,b=layout['offsets'][j:j+2]
                expected=values[[gidx[f] for f in origins[j]]]
                if mode=='all_open':expected=torch.ones_like(expected)
                self.assertTrue(torch.equal(packed[:n,a:b],expected[:,None].expand(n,b-a)))
                self.assertTrue(torch.equal(packed[n:,a:b],torch.zeros(3-n,b-a)))
            self.assertTrue(torch.equal(g[layout['left']],g[layout['right']]))

    def test_wrong_prefix_origin_cannot_borrow_another_images_gate(self):
        cache,gates,values,gidx,states,index=self.fixture();bad=copy.deepcopy(gates)
        bad['prefix_to_empty_feature_id']['b_1_1']='a_0_0'
        for mode in train.POLICY['conditions']:
            with self.assertRaises(ValueError):train.semantic_batch(torch,cache,states,index,['a','b'],bad,values,gidx,mode)
        with self.assertRaises(ValueError):
            train.semantic_batch(torch,cache,states,index,['a','b'],gates,values.requires_grad_(),gidx,'native_gate')

    def test_explicit_gate_padding_has_no_readout_effect(self):
        cache,gates,values,gidx,states,index=self.fixture()
        h,g,layout,packed,_=train.semantic_batch(torch,cache,states,index,['a','b'],gates,values,gidx,'all_open')
        core=ParallelLocalSemanticAggregation(4,rank=3)
        with torch.no_grad():core.up.weight.normal_(0,.2);core.local_bias.fill_(.7)
        full=core(h,g,gates=packed)
        alone=core(h[:1,:2],g[:2],gates=packed[:1,:2])
        torch.testing.assert_close(full[:2],alone)
        wrong=packed.clone();wrong[1:,:2]=1
        self.assertGreater(float((core(h,g,gates=wrong)[:2]-alone).abs().max()),1e-5)

    def test_native_sequence_ce_and_residual_are_the_only_objective(self):
        cache,gates,values,gidx,states,index=self.fixture()
        h,g,layout,packed,_=train.semantic_batch(torch,cache,states,index,['a','b','c','d'],gates,values,gidx,'all_open')
        core=ParallelLocalSemanticAggregation(4,rank=3)
        with torch.no_grad():core.up.weight.normal_(0,.2)
        head=nn.Linear(4,8,bias=False).requires_grad_(False);norm=nn.Identity()
        packed.requires_grad_();total,ce,residual,path,parts=train.losses(torch,core,h,g,layout,packed,norm,head,['a','b','c','d'])
        self.assertAlmostEqual(float(ce),sum(parts['per_scene_ce'])/4,places=6)
        self.assertAlmostEqual(float(residual),sum(parts['per_pair_consistency'])/2,places=6)
        self.assertGreater(float(path),0.)
        self.assertTrue(torch.equal(total,ce+residual))
        self.assertFalse(torch.equal(total,ce+residual+path))
        self.assertIsNone(torch.autograd.grad(total,packed,allow_unused=True,retain_graph=True)[0])
        total.backward();self.assertIsNone(packed.grad);self.assertIsNone(head.weight.grad)
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in core.parameters()))
        self.assertGreater(float(core.local.weight.grad.abs().sum()),0)
        self.assertGreater(float(core.up.weight.grad.abs().sum()),0)

    def test_zero_initialization_and_identity_pair_regularizer(self):
        cache,gates,values,gidx,states,index=self.fixture()
        h,g,layout,packed,_=train.semantic_batch(torch,cache,states,index,['c','c'],gates,values,gidx,'native_gate')
        core=ParallelLocalSemanticAggregation(4,rank=3);head=nn.Linear(4,8,bias=False).requires_grad_(False)
        total,ce,residual,_,_=train.losses(torch,core,h,g,layout,packed,nn.Identity(),head,['c','c'])
        self.assertTrue(train.zero_gradients(torch,core,residual)['passed']);total.backward()
        self.assertGreater(float(core.up.weight.grad.abs().sum()),0)
        for name,p in core.named_parameters():
            if name!='up.weight':self.assertTrue(torch.equal(p.grad,torch.zeros_like(p.grad)))
        with torch.no_grad():core.up.weight.normal_(0,.2)
        _,_,residual,_,parts=train.losses(torch,core,h,g,layout,packed,nn.Identity(),head,['c','c'])
        self.assertEqual(float(residual),0.);self.assertEqual(parts['saturated_identity_pairs'],[True])

    def test_persistent_pair_rng_and_epoch_tail_are_preserved(self):
        pairs=[dict(pair_id=str(i),question='q',gold=i%17,sids=[f'a{i}',f'b{i}'],pair_kind='test',n_frames=[8,16],replicas=[0,0]) for i in range(918)]
        order=train.presentation_order(pairs,20)
        self.assertEqual(len(order),73440);self.assertEqual(order,train.presentation_order(pairs,20))
        self.assertNotEqual(order,train.presentation_order(pairs,21))
        self.assertEqual(Counter(r['epoch'] for r in order[1824:1840]),{1:12,2:4})
        rng=random.Random(20)
        for epoch in range(1,41):
            indices=list(range(918));rng.shuffle(indices)
            self.assertEqual([r['slot'] for r in order[(epoch-1)*1836:epoch*1836:2]],indices)
        self.assertTrue(all(order[i]['pair_id']==order[i+1]['pair_id'] for i in range(0,73440,2)))
        self.assertAlmostEqual(train.lr(50),.001);self.assertAlmostEqual(train.lr(4590),.00001)

    def test_eos_complete_ascii_and_special_token_failures(self):
        tokenizer=Tokenizer()
        self.assertTrue(train.score_output(tokenizer,[1,6,151645],16)['exact'])
        self.assertTrue(train.score_output(tokenizer,[1,6,151643],16)['exact'])
        self.assertFalse(train.score_output(tokenizer,[1,151645],16)['exact'])
        malformed=train.score_output(tokenizer,[100,1,151645],1)
        self.assertFalse(malformed['exact']);self.assertFalse(malformed['no_nonterminal_special_tokens'])
        self.assertFalse(train.score_output(tokenizer,[21,151645],1)['parseable'])
        truncated=train.score_output(tokenizer,[0,0,0,1],1)
        self.assertTrue(truncated['parsed_count_correct']);self.assertTrue(truncated['truncated']);self.assertFalse(truncated['exact'])
        with self.assertRaises(ValueError):train.score_output(tokenizer,[1],1)
        with self.assertRaises(ValueError):train.score_output(tokenizer,[1,151645,1,151645],1)

    def test_nested_archive_copy_detaches_without_changing_metadata(self):
        with torch.inference_mode():tensor=torch.tensor([1.,2.])
        value={'actual':{'hidden':tensor,'metadata':['q',{'n':2}]},'sequence':(tensor,)}
        copied=train.cpu_tree(torch,value)
        self.assertTrue(torch.equal(copied['actual']['hidden'],tensor));self.assertFalse(copied['actual']['hidden'].is_inference())
        self.assertIsNot(copied['actual']['hidden'],tensor);self.assertEqual(copied['actual']['metadata'],value['actual']['metadata'])
        self.assertIsInstance(copied['sequence'],tuple)


if __name__=='__main__':
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','1')))
    unittest.main()

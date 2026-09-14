"""CPU-Slurm checks of ragged full-sequence reconstruction and live objectives."""
from pathlib import Path
import copy
import os
import sys
import unittest
if not (os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu'
        and not os.environ.get('SLURM_JOB_GPUS')):
    raise RuntimeError('Run tensor/model checks only through CPU Slurm')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from scripts.native_vision_v11_batches import batch_states,forward_objectives
from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
from tests import test_native_vision_v11_last_block as native_fixture


class CompleteBatchTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(37);torch.set_num_threads(2)

    def fixture(self):
        cache=dict(scenes={});plan=dict(pad_token_id=0,features={},layouts={});prompts={};values=[];index={}
        def feature(fid,value,ids):
            index[fid]=len(values);values.append(value)
            plan['features'][fid]=dict(layout_id=fid)
            plan['layouts'][fid]=dict(input_ids=ids,position_ids=[[list(range(len(ids)))]]*3)
        for tag,ids,targets in [('a',[3,4,5],[7,151645]),('b',[3,4,5,6,8],[8,9,151645])]:
            hidden=torch.randn(len(ids),32);gids=[]
            for t in range(len(targets)):
                gid=f'g{tag}{t}';gids.append(gid)
                feature(gid,hidden[-1].clone() if t==0 else torch.randn(32),ids+targets[:t])
            prompts[gids[0]]=dict(hidden_states=hidden,input_ids=torch.tensor(ids),attention_mask=torch.ones(len(ids),dtype=torch.long),
                position_ids=torch.arange(len(ids)).expand(3,-1).clone())
            for side,n in ((0,2),(1,3)) if tag=='a' else ((0,2),):
                sid=f'{tag}{side}';local=[]
                for row in range(n):
                    rowids=[]
                    for t in range(len(targets)):
                        fid=f'l{sid}{row}{t}';rowids.append(fid);feature(fid,torch.randn(32),ids+targets[:t])
                    local.append(rowids)
                cache['scenes'][sid]=dict(local_feature_ids=local,global_feature_ids=gids,n_frames=n,
                    target_ids=targets,target_prefixes=[targets[:t] for t in range(len(targets))])
        return cache,plan,torch.stack(values),index,prompts,['a0','a1','b0','b0']

    def pack(self,fixture):return batch_states(torch,*fixture)

    def test_ragged_full_sequences_keep_every_query_and_omit_target_eos(self):
        fixture=self.fixture();cache,plan,states,index,prompts,sids=fixture
        local,g,layout,replay=self.pack(fixture)
        self.assertEqual(layout['offsets'],[0,2,4,7,10]);self.assertEqual(tuple(local.shape),(3,10,32))
        self.assertEqual(replay['attention_mask'].sum(-1).tolist(),[4,4,7,7])
        self.assertEqual(replay['query_indices'],[[5,6],[5,6],[4,5,6],[4,5,6]])
        self.assertFalse(bool((replay['input_ids']==151645).any()))
        for i,sid in enumerate(sids):
            scene=cache['scenes'][sid];prompt=prompts[scene['global_feature_ids'][0]];mask=replay['attention_mask'][i].bool()
            p=prompt['input_ids'].numel();hidden=replay['hidden_states'][i,mask]
            self.assertTrue(torch.equal(hidden[:p],prompt['hidden_states']))
            a,b=layout['offsets'][i:i+2]
            self.assertTrue(torch.equal(hidden[p:],g[a+1:b]))
            self.assertTrue(torch.equal(replay['hidden_states'][i,replay['query_indices'][i]],g[a:b]))
            self.assertEqual(replay['input_ids'][i,mask].tolist(),prompt['input_ids'].tolist()+scene['target_ids'][:-1])
            self.assertEqual(replay['position_ids'][0,i,mask].tolist(),list(range(int(mask.sum()))))
        self.assertTrue(torch.equal(local[2,:2],torch.zeros_like(local[2,:2])))

    def test_changed_prompt_last_state_is_rejected(self):
        value=self.fixture();value[4]['ga0']['hidden_states'][-1]+=1
        with self.assertRaisesRegex(ValueError,'empty-prefix identity'):self.pack(value)

    def test_future_target_in_native_layout_is_rejected(self):
        value=self.fixture();value[1]['layouts']['gb2']['input_ids'].append(151645)
        with self.assertRaisesRegex(ValueError,'teacher prefix'):self.pack(value)

    def test_wrong_strict_prefix_is_rejected(self):
        value=self.fixture();value[0]['scenes']['b0']['target_prefixes'][1]=[999]
        with self.assertRaisesRegex(ValueError,'exact strict prefixes'):self.pack(value)

    def test_inference_prompt_is_rejected_before_training(self):
        value=self.fixture()
        with torch.inference_mode():value[4]['ga0']['hidden_states']=value[4]['ga0']['hidden_states'].clone()
        with self.assertRaisesRegex(ValueError,'ordinary'):self.pack(value)

    def test_full_packed_objective_has_temporal_credit_without_cross_scene_leakage(self):
        local,g,layout,replay=self.pack(self.fixture())
        fixture=native_fixture.LastBlockReplayTests();modules,_=fixture.fixture(batch=4,length=7)
        block,norm,_,rotary=modules;head=nn.Linear(32,151646,bias=False).eval().requires_grad_(False)
        branch=ParallelLocalAggregation(32,rank=8)
        with torch.no_grad():branch.up.weight.normal_(std=.01)
        for placement in ('pre_last','post_last'):
            result=forward_objectives(torch,branch,local,g,layout,replay,block,norm,head,rotary,placement)
            # First token of the K10 pair is packed at4; its second numeral at5.
            ce=torch.nn.functional.cross_entropy(result['replay']['logits'][5:6].float(),torch.tensor([9]))
            gradient=torch.autograd.grad(ce,result['delta'],retain_graph=True)[0]
            self.assertEqual(bool(gradient[4].ne(0).any()),placement=='pre_last')
            self.assertTrue(bool(gradient[5].ne(0).any()));self.assertFalse(bool(gradient[6].ne(0).any()))
            self.assertFalse(bool(gradient[:4].ne(0).any()));self.assertFalse(bool(gradient[7:].ne(0).any()))
            self.assertTrue(torch.equal(result['residual_positions'][2:],torch.zeros(3)))
            scene_means=[result['ce_positions'][a:b].mean() for a,b in zip(layout['offsets'],layout['offsets'][1:])]
            self.assertTrue(torch.allclose(result['ce'],torch.stack(scene_means).mean()))
            self.assertTrue(torch.equal(result['total'],result['ce']+result['residual']))
            branch.zero_grad(set_to_none=True);result['total'].backward()
            self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in branch.parameters()))
            self.assertTrue(all(p.grad is None and not p.requires_grad for m in (block,norm,head,rotary) for p in m.parameters()))


if __name__=='__main__':unittest.main()

"""Meaningful gate-origin/precision/padding tests; CPU Slurm execution only."""
from pathlib import Path
import os
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
if __name__=='__main__' and (not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION')!='cpu'):
    raise SystemExit('CPU Slurm required')
import torch
from scripts.stage_native_vision_v18_gates import gates_from_probabilities,broadcast_gates,build_origin_map,object_sha,GATE_RULE
from gnnformer.parallel_local_semantic_gate import GATE_RULE as CONTROLLER_RULE


class GateCacheTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_full_mass_not_conditional_probability(self):
        p0=torch.tensor([.1,.4,0.],dtype=torch.float64);p1=torch.tensor([.2,.4,0.],dtype=torch.float64)
        g=gates_from_probabilities(torch,p0,p1)
        self.assertTrue(torch.equal(g,torch.tensor([.1,0.,0.],dtype=torch.float32)))
        self.assertNotAlmostEqual(float(g[0]),float(p1[0]/(p0[0]+p1[0])))
        self.assertEqual(GATE_RULE,CONTROLLER_RULE)

    def test_detached_FP32_closed_negative_and_tie(self):
        p0=torch.tensor([.8,.1,.3],dtype=torch.float64,requires_grad=True)
        p1=torch.tensor([.1,.8,.3],dtype=torch.float64,requires_grad=True)
        g=gates_from_probabilities(torch,p0,p1)
        self.assertEqual(g.dtype,torch.float32);self.assertFalse(g.requires_grad)
        self.assertEqual(float(g[0]),0);self.assertEqual(float(g[2]),0)
        self.assertTrue(torch.equal(g[1:],torch.tensor([.7,0.],dtype=torch.float32)))

    def test_reject_invalid_mass_and_dtype(self):
        for a,b in [([-.1],[.2]),([.8],[.8]),([float('nan')],[0.])]:
            with self.assertRaises(ValueError):gates_from_probabilities(torch,torch.tensor(a,dtype=torch.float64),torch.tensor(b,dtype=torch.float64))
        with self.assertRaises(ValueError):gates_from_probabilities(torch,torch.zeros(1),torch.zeros(1))

    def fixture(self):
        features={};rows=[]
        for pair in ('a','b'):
            ids=[]
            for prefix in ([],[15],[16],[16,15],[16,21],[24]):
                fid=object_sha(['local',pair,prefix]);features[fid]=dict(kind='local',pair_id=pair,prefix_ids=prefix,question_sha256='q');ids.append(fid)
            rows.append(ids)
        return features,{'scene':dict(local_feature_ids=rows,gold=16,target_ids=[16,21,151645])}

    def test_one_origin_for_all_strict_prefixes_independent_of_gold(self):
        f,s=self.fixture();mapping,scenes=build_origin_map(f,s)
        self.assertEqual(len(mapping),12)
        for row,owner in zip(s['scene']['local_feature_ids'],scenes['scene']['local_empty_feature_ids']):
            self.assertTrue(all(mapping[fid]==owner for fid in row))
        s['scene']['gold']=0;s['scene']['target_ids']=[15,151645]
        self.assertEqual((mapping,scenes),build_origin_map(f,s))

    def test_reject_cross_image_owner_or_missing_empty(self):
        f,s=self.fixture();s['scene']['local_feature_ids'][0][2]=s['scene']['local_feature_ids'][1][2]
        with self.assertRaises(ValueError):build_origin_map(f,s)
        f,s=self.fixture();del f[s['scene']['local_feature_ids'][0][0]]
        with self.assertRaises(ValueError):build_origin_map(f,s)

    def test_padding_stays_zero_in_all_open_control(self):
        g=torch.tensor([0.,.6],requires_grad=True)
        a=broadcast_gates(torch,g,3,padded_items=4,mode='native_gate')
        b=broadcast_gates(torch,g,3,padded_items=4,mode='all_open')
        self.assertTrue(torch.equal(a[:2],g.detach()[:,None].expand(2,3)))
        self.assertTrue(torch.equal(b[:2],torch.ones(2,3)))
        self.assertTrue(torch.equal(a[2:],torch.zeros(2,3)));self.assertTrue(torch.equal(b[2:],torch.zeros(2,3)))
        self.assertFalse(a.requires_grad);self.assertFalse(b.requires_grad)
        self.assertTrue(torch.equal(broadcast_gates(torch,torch.empty(0),2,padded_items=3,mode='all_open'),torch.zeros(3,2)))

    def test_causal_broadcast_uses_identical_origin_every_position(self):
        g=torch.tensor([.125,0.,.875],dtype=torch.float32)
        a=broadcast_gates(torch,g,7)
        self.assertTrue(all(torch.equal(a[:,i],g) for i in range(7)))
        with self.assertRaises(ValueError):broadcast_gates(torch,g,2,padded_items=2)


if __name__=='__main__':unittest.main()

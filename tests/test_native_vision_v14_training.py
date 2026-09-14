"""CPU-only graph, phase-boundary and schedule tests for V14 (no native VLM)."""
from pathlib import Path
import os
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
if __name__=='__main__' and (not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION')!='cpu'):
    raise SystemExit('CPU Slurm required')
import torch
from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
from gnnformer.paired_sequence_objectives import sequence_layout
from scripts import train_native_vision_v14 as train


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(813)
        self.core=ParallelLocalAggregation(hidden_size=8,rank=4)
        with torch.no_grad():self.core.up.weight.normal_(0,.15)
        self.g=torch.randn(2,8).half();self.local=torch.randn(4,2,8).half();self.local[2:,0]=0
        self.refs=torch.randn(24,2,8).half()

    def read(self,mode='centered'):
        return train.empirical_batch(torch,self.core,self.local,self.g,self.refs,[2,4],mode)

    def test_reference_credit_is_part_of_total_query_and_local_gradients(self):
        result=self.read();loss=result['delta'].square().sum()
        dm=torch.autograd.grad(loss,result['reference_messages'],retain_graph=True)[0]
        weights=(self.core.query.weight,self.core.local.weight)
        reference=torch.autograd.grad(result['reference_messages'],weights,grad_outputs=dm,retain_graph=True)
        full=torch.autograd.grad(loss,weights,retain_graph=True)
        from gnnformer.conditional_null_mean import projected_null_readout
        stopped=[]
        for i,n in enumerate((2,4)):
            g=self.g[i:i+1];z=self.core.aggregate(self.core.encode(self.local[:n,i:i+1],g))
            stopped.append(projected_null_readout(self.core,z,g,result['mean'][i:i+1].detach(),n_elements=n,mode='centered')['delta_float32'])
        stop=torch.autograd.grad(torch.cat(stopped).square().sum(),weights)
        for a,b,c in zip(full,stop,reference):
            self.assertGreater(float(c.norm()),1e-5)
            torch.testing.assert_close(a-b,c,rtol=3e-4,atol=3e-5)
        self.assertTrue(result['mean'].requires_grad)
        self.assertTrue(all(p.grad is None for p in self.core.parameters()))

    def test_reference_occurrences_not_deduplicated(self):
        self.refs[1:]=self.refs[0:1]
        a=self.read()['mean'];self.refs[-1]=self.refs[-1]+2
        result=self.read();messages=result['reference_messages']
        torch.testing.assert_close(result['mean'],messages.double().mean(0).float(),rtol=0,atol=0)
        self.assertFalse(torch.equal(a,result['mean']))
        self.assertFalse(torch.equal(result['mean'],torch.stack([messages[0],messages[-1]]).mean(0)))

    def test_padding_is_not_reference_or_real_evidence(self):
        result=self.read();self.assertEqual(result['delta'].shape,self.g.shape)
        self.local[3,0,0]=1
        with self.assertRaises(ValueError):self.read()

    def test_zero_up_reference_gradient_and_different_first_up_gradients(self):
        with torch.no_grad():self.core.up.weight.zero_()
        gradients=[]
        for mode in ('centered','offset'):
            result=self.read(mode);loss=result['delta'].sum()
            audit=train.reference_gradient_audit(torch,self.core,loss,result,1)
            self.assertEqual(audit['core_reference_gradients']['reference_message'],0.)
            gradients.append(torch.autograd.grad(loss,self.core.up.weight)[0])
        self.assertFalse(torch.equal(*gradients))

    def test_native_cast_and_ragged_pair_loss(self):
        g=torch.cat([self.g,self.g]);local=torch.randn(4,4,8).half();local[2:,:2]=0
        refs=torch.cat([self.refs,self.refs],dim=1)
        layout=sequence_layout([[1,2],[1,2]])
        head=torch.nn.Linear(8,5,bias=False,dtype=torch.float16).requires_grad_(False)
        result=train.losses(torch,self.core,local,g,refs,layout,torch.nn.Identity(),head,'centered',['a','b'],[2,2,4,4])
        main,ce,consistency,_,details,readout=result
        expected=head((g+readout['delta'].half()).unsqueeze(0))[0]
        target=torch.tensor(layout['targets'])
        direct=torch.nn.functional.cross_entropy(expected.float(),target,reduction='none').mean()
        torch.testing.assert_close(ce,direct)
        torch.testing.assert_close(main,ce+consistency)
        main.backward();self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.core.parameters()))
        self.assertEqual(details['scene_lengths'],[2,2])

    def test_identical_pair_has_zero_regularizer(self):
        g=torch.cat([self.g,self.g]);h=torch.randn(4,2,8).half();local=torch.cat([h,h],dim=1);refs=torch.cat([self.refs,self.refs],dim=1)
        head=torch.nn.Linear(8,5,dtype=torch.float16).requires_grad_(False)
        result=train.losses(torch,self.core,local,g,refs,sequence_layout([[1,2],[1,2]]),torch.nn.Identity(),head,'centered',['same','same'],[4]*4)
        self.assertEqual(float(result[2]),0.)

    def test_student_only_updates_predictor_and_preserves_fixed_inputs(self):
        core=self.core.requires_grad_(False)
        with torch.no_grad():
            q=core.query(core.rms(self.g));m=core.encode(self.refs,self.g);target=m.double().mean(0).float();den=m.square().mean((0,2))+1e-6
        snapshot={k:v.clone() for k,v in core.state_dict().items()};fixed=[v.clone() for v in (q,target,den)]
        student=train.make_student(torch,18,rank=4);opt=torch.optim.AdamW(student.parameters(),lr=.001,weight_decay=0.)
        for i in range(3):
            opt.zero_grad(set_to_none=True);loss,_=train.student_loss(torch,student,q,target,den);loss.backward();opt.step()
        self.assertTrue(all(torch.equal(snapshot[k],v) for k,v in core.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in core.parameters()))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(fixed,(q,target,den))))
        self.assertGreater(float(student.fc1.weight.grad.norm()),0.)
        with self.assertRaises(ValueError):train.student_loss(torch,student,q.requires_grad_(True),target,den)

    def test_student_init_is_independent_and_rng_preserving(self):
        before=torch.random.get_rng_state().clone();a=train.make_student(torch,18,rank=4)
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        torch.randn(57);b=train.make_student(torch,18,rank=4);c=train.make_student(torch,19,rank=4)
        self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))
        self.assertFalse(torch.equal(a.fc1.weight,c.fc1.weight))
        self.assertEqual(float(a.fc2.weight.abs().sum()+a.fc2.bias.abs().sum()),0.)

    def conversion_fixture(self,specs):
        scenes={};vectors=[];index={}
        def feature(fid):
            index[fid]=len(vectors);vectors.append([len(vectors)+1.,-len(vectors)-1.]);return fid
        for i,(n,tokens) in enumerate(specs):
            sid=f'scene{i:04d}'
            local=[[feature(f'{sid}_image{j}_prefix{k}') for k in range(len(tokens))] for j in range(n)]
            global_ids=[feature(f'{sid}_global_prefix{k}') for k in range(len(tokens))]
            scenes[sid]=dict(n_frames=n,target_ids=tokens,local_feature_ids=local,global_feature_ids=global_ids)
        return dict(scenes=scenes),torch.tensor(vectors,dtype=torch.float32),index

    def test_conversion_accepts_mixed_targets_lengths_and_odd_single_scene_batches(self):
        specs=[(2,[7,99]),(4,[1,0,99]),(1,[9,99])]
        cache,states,index=self.conversion_fixture(specs);sids=list(cache['scenes'])
        with self.assertRaises(ValueError):sequence_layout([tokens for _,tokens in specs[:2]])
        local,g,layout=train.conversion_states(torch,cache,states,index,sids)
        self.assertEqual(layout,dict(target_sequences=[tokens for _,tokens in specs],offsets=[0,2,5,7]))
        self.assertEqual(local.shape,(4,7,2));self.assertEqual(g.dtype,states.dtype)
        for i,sid in enumerate(sids):
            a,b=layout['offsets'][i:i+2];scene=cache['scenes'][sid];n=scene['n_frames']
            self.assertTrue(torch.equal(g[a:b],states[[index[f] for f in scene['global_feature_ids']]]))
            self.assertTrue(torch.equal(local[:n,a:b],states[torch.tensor([[index[f] for f in row] for row in scene['local_feature_ids']])]))
            self.assertEqual(float(local[n:,a:b].abs().sum()),0.)
        _,single,single_layout=train.conversion_states(torch,cache,states,index,sids[-1:])
        self.assertEqual(single_layout['offsets'],[0,2]);self.assertTrue(torch.equal(single,g[-2:]))

    def test_conversion_inventory_preserves_all_unique_scene_prefixes_and_partial_batch(self):
        # Same 1782-scene/4266-position cardinalities, with heterogeneous adjacent targets.
        specs=[(1+i%3,[i%10,99] if i<1080 else [1,i%7,99]) for i in range(1782)]
        cache,states,index=self.conversion_fixture(specs);sids=sorted(cache['scenes']);seen=[];sizes=[]
        for start in range(0,len(sids),32):
            selected=sids[start:start+32];_,g,layout=train.conversion_states(torch,cache,states,index,selected);sizes.append(len(selected))
            for i,sid in enumerate(selected):
                scene=cache['scenes'][sid];a,b=layout['offsets'][i:i+2]
                self.assertEqual(layout['target_sequences'][i],scene['target_ids'])
                self.assertTrue(torch.equal(g[a:b],states[[index[f] for f in scene['global_feature_ids']]]))
                seen.extend((sid,position) for position in range(b-a))
        expected=[(sid,position) for sid in sids for position in range(len(cache['scenes'][sid]['target_ids']))]
        self.assertEqual(seen,expected);self.assertEqual(len(seen),4266);self.assertEqual(len(set(seen)),4266)
        self.assertEqual(len({sid for sid,_ in seen}),1782);self.assertEqual(sizes,[32]*55+[22])

    def test_conversion_rejects_missing_prefix_features_without_pairing_constraints(self):
        cache,states,index=self.conversion_fixture([(2,[1,0,99])]);sid=next(iter(cache['scenes']))
        cache['scenes'][sid]['local_feature_ids'][0].pop()
        with self.assertRaises(ValueError):train.conversion_states(torch,cache,states,index,[sid])

    def test_phase_schedules_and_lr_boundaries(self):
        import random
        groups={str(i):{} for i in range(972)};rows=train.student_order(groups,18)
        self.assertEqual(len(rows),512000);self.assertEqual(rows[-1]['cycle'],527)
        self.assertEqual([r['cycle'] for r in rows[960:1024]],[1]*12+[2]*52)
        rng=random.Random(18+20261111)
        for cycle in (1,2):
            slots=list(range(972));rng.shuffle(slots)
            self.assertEqual([r['slot'] for r in rows[(cycle-1)*972:cycle*972]],slots)
        self.assertAlmostEqual(train.student_lr(100),.001);self.assertAlmostEqual(train.student_lr(8000),.00001)
        pairs=[dict(pair_id=str(i),question='q',gold=i%17,sids=[f'a{i}',f'b{i}'],pair_kind='test',n_frames=[8,16],replicas=[0,0]) for i in range(918)]
        ordinary=train.presentation_order(pairs,18);self.assertEqual(len(ordinary),73440)
        self.assertEqual([r['epoch'] for r in ordinary[1824:1840]],[1]*12+[2]*4)
        self.assertTrue(all(ordinary[i]['pair_id']==ordinary[i+1]['pair_id'] for i in range(0,len(ordinary),2)))


if __name__=='__main__':unittest.main()

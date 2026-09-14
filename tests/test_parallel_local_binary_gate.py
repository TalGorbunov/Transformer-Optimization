"""New binary-controller integration tests; execute only in CPU Slurm."""
from pathlib import Path
import copy
import os
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION')!='cpu':
    raise SystemExit('CPU Slurm required; no numerical execution on login')
import torch
from gnnformer.parallel_local_binary_aggregation import ParallelLocalBinaryAggregation
from gnnformer.parallel_local_binary_gate import ParallelLocalBinaryGate,InvalidBinaryMeasurement,binary_measurement
from scripts import native_identity_join_runtime as runtime


class Norm(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.ones(4,dtype=torch.float16),requires_grad=False);self.calls=0;self.eval()
    def forward(self,hidden_states):self.calls+=1;return hidden_states*self.weight


class Head(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.eye(4,dtype=torch.float16),requires_grad=False);self.calls=0;self.shapes=[];self.eval()
    def forward(self,hidden_states):
        self.calls+=1;self.shapes.append(tuple(hidden_states.shape));return torch.nn.functional.linear(hidden_states,self.weight)


class BinaryGateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(731)
        self.norm,self.head=Norm(),Head();self.core=ParallelLocalBinaryAggregation(hidden_size=4,rank=3,mode='vector')
        with torch.no_grad():self.core.up.weight.normal_(0,.2)
        self.h=torch.randn(3,5,4).half();self.h[0,2]=torch.tensor([5.,0.,1.,0.]);self.h[1,2]=torch.tensor([0.,5.,1.,0.])
        self.identity=dict(sid='A',question_sha256='a'*64,image_sha256=['b'*64,'c'*64],prompt_width=3,
            input_identity_sha256='d'*64,native_identity_sha256='e'*64)
    def controller(self,*,core=None,q=(2,),s=(2,),artifact=None,identity=None,detach=True):
        return ParallelLocalBinaryGate(self.norm,self.head,core or self.core,n_local_rows=2,zero_token_id=0,one_token_id=1,
            origin_identity=identity or self.identity,query_indices=q,stream_positions=s,origin_artifact=artifact,capture=True,detach_captures=detach)
    def origin(self,core=None):
        c=self.controller(core=core)
        with c:self.norm(self.h[:,:3])
        return c,c.export_origin_artifact(cpu=True)
    def test_native_argmax_only_full_mass_and_invalid_not_closed(self):
        x=torch.tensor([[[2.,1.,0.,0.]],[[0.,1.,12.,0.]],[[0.,0.,0.,0.]]],dtype=torch.float16)
        for bad in (-1,True,1.5):
            with self.assertRaisesRegex(ValueError,'local-row count'):binary_measurement(x,0,1,bad)
        m=binary_measurement(x,0,1,2)
        self.assertEqual(m['binary_measurements'].tolist(),[0,-1]);self.assertIsNone(m['native_gates']);self.assertEqual(m['invalid_rows'],[1])
        self.assertFalse(m['measurement_passed']);self.assertLess(float(m['p0'][1]+m['p1'][1]),.001)
        p=x[:2,0].double().softmax(-1);torch.testing.assert_close(m['p1'],p[:,1],rtol=1e-12,atol=0)
        x[1,0]=torch.tensor([0.,2.,1.,0.]);m=binary_measurement(x,0,1,2)
        self.assertTrue(m['measurement_passed']);self.assertEqual(m['native_gates'].tolist(),[0.,1.]);self.assertEqual(m['native_gates'].dtype,torch.float32)
    def test_invalid_origin_retained_cleanup_and_bound_failure(self):
        bad=self.h[:,:3].clone();bad[1,2]=torch.tensor([0.,1.,12.,0.]);c=self.controller()
        with self.assertRaises(InvalidBinaryMeasurement) as caught:
            with c:self.norm(bad)
        artifact=caught.exception.artifact
        self.assertFalse(c.active);self.assertEqual(c.calls,0);self.assertEqual(c.probe_head_calls,1);self.assertEqual(c.argmax_calls,1)
        self.assertTrue(torch.equal(artifact['origin_hidden'],bad[:,2:3]));self.assertIsNone(artifact['native_gates']);self.assertEqual(artifact['binary_measurements'].tolist(),[0,-1])
        self.assertFalse(torch.is_inference(artifact['origin_logits']));calls=self.head.calls
        with self.assertRaises(InvalidBinaryMeasurement):
            with self.controller(artifact=artifact,q=(4,),s=(4,)) as replay:self.norm(self.h)
        self.assertEqual(self.head.calls,calls);self.assertEqual(replay.argmax_calls,0);self.assertEqual(replay.artifact_validation_argmax_calls,1)
        result=runtime._failure(caught.exception,{},'vector',{})
        self.assertEqual(result['generated_ids'],[]);self.assertFalse(result['completed']);self.assertFalse(result['truncated']);self.assertTrue(result['measurement_failure'])
    def test_zero_identity_native_cast_and_probe_hook_bypass(self):
        with torch.no_grad():self.core.up.weight.zero_()
        hits=[];handle=self.norm.register_forward_pre_hook(lambda *_:hits.append(1));h=self.h[:,:3].clone()
        try:
            with self.controller() as c:actual=self.norm(h)
        finally:handle.remove()
        self.assertTrue(torch.equal(actual,h));self.assertEqual(hits,[1]);self.assertEqual(self.norm.calls,2);self.assertEqual(self.head.shapes,[(3,1,4)])
        self.assertEqual(c.assert_complete()['argmax_calls'],1)
    def test_out_of_place_global_only_cast_and_gradient(self):
        h=self.h[:,:3].clone().requires_grad_(True);old=h.detach().clone()
        with self.controller(detach=False) as c:out=self.norm(h)
        self.assertTrue(torch.equal(h.detach(),old));self.assertTrue(torch.equal(out[:-1],old[:-1]));self.assertTrue(torch.equal(out[-1,:2],old[-1,:2]))
        self.assertTrue(torch.equal(out[-1,2:3],old[-1,2:3]+c.last_capture['delta'].half()))
        out[-1,2].float().square().sum().backward()
        self.assertIsNotNone(self.core.up.weight.grad);self.assertIsNone(self.norm.weight.grad);self.assertIsNone(self.head.weight.grad)
        self.assertTrue(torch.equal(h.grad[0,2],torch.zeros(4,dtype=h.grad.dtype)));self.assertFalse(c.last_capture['native_gates'].requires_grad)
    def test_cached_gate_reused_despite_opposite_current_argmax(self):
        with self.controller() as c:
            self.norm(self.h[:,:3]);g=c.last_capture['native_gates'].clone()
            for pos in (3,4):
                current=self.h[:,pos:pos+1].clone();current[0,0]=torch.tensor([0.,20.,0.,0.]);current[1,0]=torch.tensor([20.,0.,0.,0.])
                c.configure_queries([0],[pos]);self.norm(current)
                self.assertTrue(torch.equal(c.last_capture['native_gates'],g))
        self.assertEqual(c.calls,3);self.assertEqual(c.probe_norm_calls,1);self.assertEqual(c.argmax_calls,1)
    def test_artifact_arm_independence_and_fullprefix_causality(self):
        _,artifact=self.origin();self.assertNotIn('branch_mode',artifact)
        scalar=ParallelLocalBinaryAggregation(hidden_size=4,rank=3,mode='scalar');scalar.load_state_dict(self.core.state_dict())
        head_calls=self.head.calls
        with self.controller(core=scalar,artifact=artifact,q=(2,3,4),s=(2,3,4)) as c:before=self.norm(self.h)
        h=self.h.clone();h[:,4]+=10
        with self.controller(core=scalar,artifact=artifact,q=(2,3,4),s=(2,3,4)) as d:after=self.norm(h)
        self.assertEqual(self.head.calls,head_calls);self.assertTrue(torch.equal(before[-1,2:4],after[-1,2:4]));self.assertEqual(c.last_capture['branch_mode'],'scalar')
        self.assertTrue(torch.equal(c.last_capture['aggregate'],c.last_capture['count'][:,None]*c.last_capture['scalar_payload']))
        self.assertEqual(c.assert_complete()['argmax_calls'],0)
    def test_scalar_information_boundary_and_vector_uses_local_state(self):
        scalar=ParallelLocalBinaryAggregation(hidden_size=4,rank=3,mode='scalar');scalar.load_state_dict(self.core.state_dict())
        a=self.h[:,:3].clone();b=a.clone();b[0,2]=torch.tensor([0.,7.,3.,2.]);b[1,2]=torch.tensor([7.,0.,2.,3.])
        ident=dict(self.identity,sid='B',image_sha256=['f'*64,'0'*64],input_identity_sha256='1'*64)
        with self.controller(core=scalar) as c:sa=self.norm(a)
        with self.controller(core=scalar,identity=ident) as d:sb=self.norm(b)
        self.assertTrue(torch.equal(c.last_capture['count'],d.last_capture['count']));self.assertTrue(torch.equal(c.last_capture['aggregate'],d.last_capture['aggregate']))
        self.assertTrue(torch.equal(sa[-1,2],sb[-1,2]))
        with self.controller() as va:self.norm(a)
        with self.controller(identity=ident) as vb:self.norm(b)
        self.assertFalse(torch.equal(va.last_capture['aggregate'],vb.last_capture['aggregate']))
    def test_mode_guard_nesting_reset_and_identity_tamper(self):
        c,artifact=self.origin();bad=copy.deepcopy(artifact);bad['binary_measurements'][0]=1
        with self.assertRaises(ValueError):self.controller(artifact=bad)
        with self.assertRaises(ValueError):self.controller(artifact=artifact,identity=dict(self.identity,sid='other'))
        c=self.controller();self.core.mode='scalar'
        with self.assertRaisesRegex(ValueError,'core.mode changed'):
            with c:self.norm(self.h[:,:3])
        self.core.mode='vector';c=self.controller();c.mode='all_open'
        with self.assertRaisesRegex(ValueError,'native_gate'):
            with c:self.norm(self.h[:,:3])
        c=self.controller()
        with self.assertRaisesRegex(RuntimeError,'synthetic'):
            with c:
                self.norm(self.h[:,:3])
                with self.assertRaises(ValueError):c.reset(self.identity,query_indices=[2],stream_positions=[2])
                self.assertEqual(c.argmax_calls,1)
                with self.assertRaises(ValueError):self.controller().__enter__()
                raise RuntimeError('synthetic')
        self.assertFalse(c.active);self.assertEqual(len(self.norm._forward_pre_hooks),0)
        with self.assertRaises(ValueError):c.__enter__()
        c.reset(self.identity,query_indices=[2],stream_positions=[2])
        with c:self.norm(self.h[:,:3])
        self.assertEqual(c.argmax_calls,1)
    def test_nonempty_prefix_without_origin_rejected_and_runtime_input_boundary(self):
        with self.assertRaisesRegex(ValueError,'bound original'):
            with self.controller(q=(4,),s=(4,)):self.norm(self.h)
        sample=dict(sid='A',n_frames=1,question='Unchanged complete question',image_files=[],answer='Mary')
        with self.assertRaisesRegex(ValueError,'labels/roles/answers'):runtime.prepare_scene(None,sample)
        self.assertIn('Collection question: Unchanged complete question',runtime.local_prompt(sample['question']))


if __name__=='__main__':unittest.main()

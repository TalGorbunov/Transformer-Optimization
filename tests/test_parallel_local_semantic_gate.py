"""Source-only tests; numerical execution requires an authorized CPU Slurm job."""
from pathlib import Path
import copy
import os
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
if __name__=='__main__' and (not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION')!='cpu'):
    raise SystemExit('CPU Slurm required; no numerical tests on login')
import torch
from gnnformer.parallel_local_semantic_aggregation import ParallelLocalSemanticAggregation
from gnnformer.parallel_local_semantic_gate import ParallelLocalSemanticGate,full_vocabulary_gates


class NativeNorm(torch.nn.Module):
    def __init__(self,h=4):
        super().__init__();self.weight=torch.nn.Parameter(torch.ones(h,dtype=torch.float16),requires_grad=False);self.calls=0;self.eval()
    def forward(self,hidden_states):self.calls+=1;return hidden_states*self.weight


class NativeHead(torch.nn.Module):
    def __init__(self,h=4):
        super().__init__();self.weight=torch.nn.Parameter(torch.eye(h,dtype=torch.float16),requires_grad=False);self.calls=0;self.shapes=[];self.eval()
    def forward(self,hidden_states):
        self.calls+=1;self.shapes.append(tuple(hidden_states.shape));return torch.nn.functional.linear(hidden_states,self.weight)


class GateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(219)
        self.norm,self.head=NativeNorm(),NativeHead();self.core=ParallelLocalSemanticAggregation(hidden_size=4,rank=3)
        with torch.no_grad():self.core.up.weight.normal_(0,.2)
        self.h=torch.randn(3,5,4).half();self.h[0,2]=torch.tensor([2.,0.,0.,0.]);self.h[1,2]=torch.tensor([0.,2.,0.,0.])
        self.identity=dict(sid='scene_a',question_sha256='a'*64,image_sha256=['b'*64,'c'*64],prompt_width=3,
            input_identity_sha256='d'*64,native_identity_sha256='e'*64)

    def controller(self,mode='native_gate',q=(2,),s=(2,),artifact=None,identity=None,capture=True,detach=True):
        return ParallelLocalSemanticGate(self.norm,self.head,self.core,n_local_rows=2,mode=mode,zero_token_id=0,one_token_id=1,
            origin_identity=identity or self.identity,query_indices=q,stream_positions=s,origin_artifact=artifact,
            capture=capture,detach_captures=detach)

    def origin(self,mode='native_gate'):
        c=self.controller(mode)
        with c:self.norm(self.h[:,:3])
        c.assert_complete();return c,c.export_origin_artifact()

    def test_full_vocabulary_mass_and_tie_close_all_coordinates(self):
        logits=torch.tensor([[[0.,1.,12.,0.]],[[1.,1.,0.,0.]],[[0.,0.,0.,0.]]],dtype=torch.float16)
        p0,p1,g=full_vocabulary_gates(logits,0,1,2)
        probability=logits[:2,0].double().softmax(-1)
        torch.testing.assert_close(p0,probability[:,0],rtol=1e-12,atol=0)
        torch.testing.assert_close(p1,probability[:,1],rtol=1e-12,atol=0)
        torch.testing.assert_close(g,(probability[:,1]-probability[:,0]).clamp_min(0).float(),rtol=1e-6,atol=0)
        self.assertLess(float(g[0]),.0001);self.assertGreater(float(g[0]),0);self.assertEqual(float(g[1]),0)
        c,artifact=self.origin();v=c.last_capture
        self.assertEqual(float(artifact['native_gates'][0]),0)
        self.assertTrue(torch.equal(v['messages'][0],torch.zeros_like(v['messages'][0])))

    def test_same_probe_work_all_open_and_gate(self):
        a,pa=self.origin('native_gate');b,pb=self.origin('all_open')
        self.assertEqual(self.head.shapes,[(3,1,4),(3,1,4)])
        for c in (a,b):self.assertEqual((c.calls,c.probe_norm_calls,c.probe_head_calls,c.probability_calls),(1,1,1,1))
        self.assertTrue(torch.equal(pa['origin_logits'],pb['origin_logits']))
        self.assertTrue(torch.equal(pa['native_gates'],pb['native_gates']))
        self.assertTrue(torch.equal(b.last_capture['applied_gates'],torch.ones(2,1)))
        self.assertFalse(torch.equal(a.last_capture['applied_gates'],b.last_capture['applied_gates']))

    def test_zero_up_identity_and_probe_bypasses_registered_hooks(self):
        with torch.no_grad():self.core.up.weight.zero_()
        seen=[];handle=self.norm.register_forward_pre_hook(lambda *_:seen.append('norm'))
        original=self.h[:,:3].clone()
        try:
            for mode in ('native_gate','all_open'):
                with self.controller(mode) as c:out=self.norm(original)
                self.assertTrue(torch.equal(out,original));self.assertTrue(torch.equal(original,self.h[:,:3]));c.assert_complete()
        finally:handle.remove()
        self.assertEqual(seen,['norm','norm']);self.assertEqual(self.norm.calls,4);self.assertEqual(self.head.calls,2)

    def test_cached_reuse_never_reclassifies_later_query(self):
        c=self.controller()
        with c:
            self.norm(self.h[:,:3]);gate=c.last_capture['native_gates'].clone()
            for t in (3,4):
                c.configure_queries([0],[t],origin_identity=self.identity)
                changed=self.h[:,t:t+1].clone();changed[0,0]=torch.tensor([0.,20.,0.,0.])
                self.norm(changed);self.assertTrue(torch.equal(c.last_capture['native_gates'],gate))
        self.assertEqual(c.calls,3);self.assertEqual(c.probe_head_calls,1);c.assert_complete()

    def test_fullprefix_requires_bound_origin_and_preserves_earlier_queries(self):
        with self.controller(q=(4,),s=(4,)) as c:
            with self.assertRaisesRegex(ValueError,'bound original'):self.norm(self.h)
        self.assertEqual(c.probe_head_calls,0)
        _,artifact=self.origin();head_calls=self.head.calls
        with self.controller(q=(2,3,4),s=(2,3,4),artifact=artifact) as a:full=self.norm(self.h)
        modified=self.h.clone();modified[:,4]+=8
        with self.controller(q=(2,3,4),s=(2,3,4),artifact=artifact) as b:future=self.norm(modified)
        self.assertTrue(torch.equal(full[-1,2:4],future[-1,2:4]));self.assertFalse(torch.equal(full[-1,4],future[-1,4]))
        self.assertEqual(self.head.calls,head_calls);a.assert_complete();b.assert_complete()

    def test_global_only_out_of_place_write_and_live_core_gradient(self):
        hidden=self.h[:,:3].clone().requires_grad_(True);original=hidden.detach().clone()
        with self.controller(detach=False) as c:out=self.norm(hidden)
        self.assertTrue(torch.equal(out[:-1],original[:-1]));self.assertTrue(torch.equal(out[-1,:2],original[-1,:2]))
        self.assertTrue(torch.equal(hidden.detach(),original));self.assertFalse(torch.equal(out[-1,2],original[-1,2]))
        self.assertFalse(c.last_capture['native_gates'].requires_grad);self.assertTrue(c.last_capture['delta'].requires_grad)
        out[-1,2].float().square().sum().backward()
        self.assertIsNotNone(self.core.up.weight.grad);self.assertIsNone(self.norm.weight.grad);self.assertIsNone(self.head.weight.grad)
        self.assertTrue(torch.equal(hidden.grad[0,2],torch.zeros_like(hidden.grad[0,2])))

    def test_artifact_identity_normal_export_and_tampering(self):
        with torch.inference_mode():
            with self.controller() as c:self.norm(self.h[:,:3])
        artifact=c.export_origin_artifact();self.assertFalse(torch.is_inference(artifact['origin_logits']))
        for changed in ('sid','question_sha256','native_identity_sha256'):
            identity=copy.deepcopy(self.identity);identity[changed]='other' if changed=='sid' else 'f'*64
            with self.assertRaises(ValueError):self.controller(artifact=artifact,identity=identity)
        bad=copy.deepcopy(artifact);bad['native_gates'][1]=0
        with self.assertRaisesRegex(ValueError,'content identity'):self.controller(artifact=bad)
        captured=c.export_last_capture(cpu=True);captured['delta'].zero_()
        self.assertFalse(torch.equal(captured['delta'],c.last_capture['delta']))

    def test_unknown_scene_reset_and_exception_cleanup(self):
        c=self.controller();before=dict(self.norm.named_modules())
        with self.assertRaisesRegex(RuntimeError,'synthetic'):
            with c:
                self.norm(self.h[:,:3])
                with self.assertRaises(ValueError):c.reset(self.identity,query_indices=[2],stream_positions=[2])
                with self.assertRaises(ValueError):c.configure_queries([0],[3],origin_identity=dict(self.identity,sid='other'))
                with self.assertRaises(ValueError):self.controller().__enter__()
                raise RuntimeError('synthetic')
        self.assertFalse(c.active);self.assertEqual(len(self.norm._forward_pre_hooks),0);self.assertEqual(before,dict(self.norm.named_modules()))
        with self.assertRaises(ValueError):c.__enter__()
        new=dict(self.identity,sid='new_scene');c.reset(new,query_indices=[2],stream_positions=[2])
        with c:self.norm(self.h[:,:3])
        self.assertEqual(c.calls,1);self.assertEqual(c.origin_identity['sid'],'new_scene');c.assert_complete()

    def test_invalid_queries_and_native_parameter_guards(self):
        for q,s in [((2,2),(2,2)),((2,3),(2,4)),((2,),(1,))]:
            with self.assertRaises(ValueError):self.controller(q=q,s=s)
        self.head.weight.requires_grad_(True)
        with self.assertRaises(ValueError):self.controller()
        self.head.weight.requires_grad_(False);c=self.controller()
        with torch.no_grad():self.head.weight.add_(.1)
        with self.assertRaisesRegex(ValueError,'version changed'):c.__enter__()


if __name__=='__main__':unittest.main()

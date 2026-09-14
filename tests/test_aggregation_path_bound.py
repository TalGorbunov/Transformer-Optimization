"""CPU Slurm checks for the sensitivity bound and its training gradients."""
import os
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
if not(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS')):
    raise SystemExit('All tensor tests require CPU Slurm')
import torch
from torch.nn import functional as F
from gnnformer.aggregation_path_bound import native_path_aggregate,paired_native_path_bound as projected_bound
from gnnformer.parallel_local_aggregation import ParallelLocalAggregation

def paired_native_path_bound(states,up,global_states,*,eps=1e-6):
    # Identity projection isolates the mathematical directional bound.
    return projected_bound(states,torch.eye(states.shape[-1],device=states.device),up,global_states,eps=eps)

class PathBoundTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(20261003)
    def test_all_base_and_extrapolation_directions(self):
        u=torch.randn(7,5);d=torch.randn(5);g=torch.ones(2,1,7)
        pair=torch.stack((torch.zeros_like(d),d)).unsqueeze(1)
        bound=paired_native_path_bound(pair,u,g)*(7+1e-6)
        for base in (torch.zeros(5),torch.randn(5)*10,torch.randn(5)*100):
            for t in (-6.,-1.,0.,.3,1.,6.):
                change=u@(F.silu(base+t*d)-F.silu(base))
                self.assertLessEqual(float(change.square().sum()),float(1.21*t*t*bound)+1e-4)
    def test_endpoint_cancellation_is_detected(self):
        pair=torch.tensor([[[-1.,1.]],[[1.,-1.]]]);u=torch.tensor([[1.,1.]])
        endpoints=F.linear(F.silu(pair),u)
        self.assertTrue(torch.equal(endpoints[0],endpoints[1]))
        self.assertGreater(float(paired_native_path_bound(pair,u,torch.ones(2,1,1))),15.99)
        far=F.linear(F.silu(torch.tensor([13.,-13.])),u)
        self.assertGreater(float((far-endpoints[0,0]).abs()),12.)
    def test_inverse_channel_scale_invariance_of_bound(self):
        pair=torch.randn(4,2,5);u=torch.randn(7,5);g=torch.randn(2,2,7).repeat_interleave(2,0)
        scale=torch.tensor([.1,2.,-3.,.5,7.])
        a=paired_native_path_bound(pair,u,g);b=paired_native_path_bound(pair/scale,u*scale,g)
        self.assertTrue(torch.allclose(a,b,rtol=2e-6,atol=1e-6))
    def test_zero_u_has_finite_zero_gradients(self):
        r=torch.randn(4,2,5,requires_grad=True);u=torch.zeros(7,5,requires_grad=True);g=torch.ones(4,2,7)
        value=paired_native_path_bound(r,u,g);value.backward()
        self.assertEqual(float(value),0.)
        for grad in (r.grad,u.grad):self.assertTrue(torch.isfinite(grad).all());self.assertTrue((grad==0).all())
    def test_individual_zero_column_gradient_is_finite(self):
        r=torch.randn(4,2,5,requires_grad=True);u=torch.randn(7,5);u[:,2]=0;u.requires_grad_(True)
        value=paired_native_path_bound(r,u,torch.ones(4,2,7));value.backward()
        self.assertGreater(float(value),0.)
        self.assertTrue(torch.isfinite(r.grad).all());self.assertTrue(torch.isfinite(u.grad).all())
    def test_actual_core_preactivation_reconstructs_residual(self):
        branch=ParallelLocalAggregation(8,rank=3)
        with torch.no_grad():branch.up.weight.normal_(0,.1)
        local=torch.randn(9,4,8,dtype=torch.float16);g=torch.randn(4,8,dtype=torch.float16)
        z=native_path_aggregate(branch,local,g)
        r=branch.aggregate_projection(z)+branch.query(branch.rms(g))
        self.assertTrue(torch.equal(branch.up(F.silu(r)),branch(local,g,output_dtype=torch.float32)))
    def test_pair_and_position_reduction(self):
        r=torch.tensor([[[0.],[0.]],[[1.],[2.]],[[0.],[0.]],[[3.],[4.]]])
        u=torch.ones(1,1);g=torch.ones(4,2,1)
        self.assertAlmostEqual(float(paired_native_path_bound(r,u,g)),7.5/(1+1e-6),places=5)
    def test_equal_latents_are_zero(self):
        r=torch.randn(2,2,5).repeat_interleave(2,0);u=torch.randn(7,5);g=torch.ones(4,2,7)
        self.assertEqual(float(paired_native_path_bound(r,u,g)),0.)
    def test_rejects_invalid_input(self):
        r=torch.randn(4,2,5);u=torch.randn(7,5);g=torch.ones(4,2,7)
        for rr,uu,gg,eps in [(r[:3],u,g[:3],1e-6),(r.half(),u,g,1e-6),
                           (r,u,g.clone().requires_grad_(True),1e-6),(r,u,g,0.),
                           (r,u,g,float('nan')),(r*float('inf'),u,g,1e-6)]:
            with self.assertRaises(ValueError):paired_native_path_bound(rr,uu,gg,eps=eps)
        bad=g.clone();bad[1,0,0]=2
        with self.assertRaises(ValueError):paired_native_path_bound(r,u,bad)
    def test_fp32_autocast_and_nonzero_gradient(self):
        r=torch.randn(4,2,5,requires_grad=True);u=torch.randn(7,5,requires_grad=True);g=torch.ones(4,2,7,dtype=torch.bfloat16)
        with torch.autocast('cpu',dtype=torch.bfloat16):value=paired_native_path_bound(r,u,g)
        value.backward();self.assertEqual(value.dtype,torch.float32)
        self.assertGreater(float(r.grad.norm()),0.);self.assertGreater(float(u.grad.norm()),0.)
    def test_direct_projection_avoids_shared_offset_rounding(self):
        z=torch.tensor([[[0.,0.]],[[1.,2.]]]);a=torch.eye(2);u=torch.eye(2);g=torch.ones(2,1,2)
        rounded=torch.nn.functional.linear(z,a,torch.full((2,),1e10))
        self.assertTrue(torch.equal(rounded[0],rounded[1]))
        self.assertGreater(float(projected_bound(z,a,u,g)),4.49)
    def test_projection_receives_gradient(self):
        z=torch.randn(4,2,5,requires_grad=True);a=torch.randn(5,5,requires_grad=True)
        u=torch.randn(7,5,requires_grad=True);g=torch.ones(4,2,7)
        value=projected_bound(z,a,u,g);value.backward()
        for grad in (z.grad,a.grad,u.grad):
            self.assertTrue(torch.isfinite(grad).all());self.assertGreater(float(grad.norm()),0.)
if __name__=='__main__':unittest.main()

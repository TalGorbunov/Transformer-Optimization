"""CPU-Slurm-only algebra/gradient fixtures; no native model, data or fitting."""
from __future__ import annotations
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import unittest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
import torch
from gnnformer.native_visual_augmentation import (
    SummaryReadResidual, PointwiseResidual, native_residual_add,
    _summary_read, _pointwise_read, LIVE_PARAMETERS, SUMMARY_ALLOCATED_PARAMETERS,
)

PROPOSAL = 'docs/paper/NATIVE_AGGREGATION_AUGMENTATION_HYPOTHESIS.md'
PROPOSAL_SHA = 'ef108beae765b3f1c20037ddcb0a279ef33b2f96009c0fdbd2cb9180dd506224'
OWN = ('gnnformer/native_visual_augmentation.py', 'tests/test_native_visual_augmentation.py',
       'slurm/native_visual_augmentation_core_check.sbatch', PROPOSAL)
INHERITED = {
    'gnnformer/native_visual_memory.py': '44e02c2272f4e4d84e929a3dd14007c7540e5726d85d78555aaaab6825dea135',
    'gnnformer/attention_moments.py': '2513044c9ea490098217fc55ac68aac5c798ce3eed7ca746e47a447a3e0fd804',
}


def cpu_slurm():
    if (not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION') != 'cpu'
            or os.environ.get('SLURM_JOB_GPUS') or int(os.environ.get('SLURM_CPUS_PER_TASK', '0')) != 4):
        raise RuntimeError('Four-core CPU Slurm allocation required for numerical fixtures')


def sample():
    g = torch.Generator().manual_seed(107)
    features = .3 * torch.randn(9, 3584, generator=g)
    hidden = .4 * torch.randn(1, 3, 3584, generator=g)
    frames = torch.tensor([0, 0, 1, 3, 7, 15, 31, 63, 127])
    coordinates = dict(frame_index=frames, raster_row=torch.arange(9) % 3,
        raster_col=torch.arange(9) % 5, grid_height=torch.full_like(frames, 3), grid_width=torch.full_like(frames, 5))
    return features, coordinates, hidden


def live(core):
    return {n:p for n,p in core.named_parameters() if p.requires_grad}


def fp64_summary(hidden, memory, wq, wk, wv, wo):
    # Scalar-channel dot products, independent of production GEMMs/RMS helper.
    def norm(row):
        return row / torch.sqrt(sum(v*v for v in row) / len(row) + 1e-6)
    def mv(weight, vector):
        return torch.stack([sum(a*b for a,b in zip(row, vector)) for row in weight])
    keys = [mv(wk, norm(row)) for row in memory]
    values = [mv(wv, row) for row in memory]
    result = []
    for row in hidden:
        query = mv(wq, norm(row))
        scores = torch.stack([sum(a*b for a,b in zip(query, key)) / math.sqrt(len(query)) for key in keys])
        exponential = (scores - scores.max()).exp()
        weights = exponential / exponential.sum()
        mixed = torch.stack([sum(weights[j]*values[j][r] for j in range(len(values))) for r in range(len(query))])
        result.append(mv(wo, mixed))
    return torch.stack(result)


class NativeVisualAugmentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cpu_slurm(); torch.set_num_threads(4)

    def test_exact_live_counts_private_rng_and_initialization(self):
        before = torch.random.get_rng_state().clone()
        summary, pointwise = SummaryReadResidual(seed=25), PointwiseResidual(seed=25)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in live(summary).values()), LIVE_PARAMETERS)
        self.assertEqual(sum(p.numel() for p in live(pointwise).values()), LIVE_PARAMETERS)
        self.assertEqual(LIVE_PARAMETERS, 2300929)
        self.assertEqual(sum(p.numel() for p in summary.parameters()), SUMMARY_ALLOCATED_PARAMETERS)
        self.assertEqual(sum(p.numel() for p in pointwise.parameters()), LIVE_PARAMETERS)
        self.assertEqual(len(live(summary)), 7); self.assertEqual(len(live(pointwise)), 3)
        self.assertFalse(summary.pool.mass_direction.requires_grad)
        self.assertEqual(int(torch.count_nonzero(summary.pool.mass_direction)), 0)
        for core, names in ((summary, ('query_weight','key_weight','value_weight','output_weight')),
                            (pointwise, ('down_weight','up_weight'))):
            generator = torch.Generator().manual_seed(25)
            for name in names:
                weight = getattr(core, name); fan_in = weight.shape[1]
                expected = torch.empty(weight.shape).uniform_(-1/math.sqrt(fan_in), 1/math.sqrt(fan_in), generator=generator)
                self.assertTrue(torch.equal(weight, expected)); self.assertEqual(weight.dtype, torch.float32)
            self.assertEqual(float(core.alpha), 0)
        g = torch.Generator().manual_seed(24)
        expected_key = torch.empty(128,3608).uniform_(-1/math.sqrt(3608), 1/math.sqrt(3608), generator=g)
        expected_queries = torch.randn(32,128,generator=g)
        self.assertTrue(torch.equal(summary.pool.key_weight, expected_key))
        self.assertTrue(torch.equal(summary.pool.queries, expected_queries))
        other = SummaryReadResidual(seed=26)
        self.assertTrue(all(torch.equal(v, other.pool.state_dict()[n]) for n,v in summary.pool.state_dict().items()))
        self.assertFalse(torch.equal(summary.query_weight, other.query_weight))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_zero_gate_native_identity_and_live_gate_gradient(self):
        features, coords, hidden = sample(); hidden = hidden.half()
        for core in (SummaryReadResidual(), PointwiseResidual()):
            memory = core.prepare_memory(features, **coords)
            delta, capture = core.delta_and_capture(hidden, memory)
            fused = native_residual_add(hidden, delta)
            self.assertTrue(torch.equal(fused, hidden)); self.assertEqual(delta.dtype, torch.float32)
            self.assertEqual(int(torch.count_nonzero(delta)), 0)
            self.assertGreater(float(capture['readout'].norm()), 0)
            # A fixed algebraic probe through the actual native cast/add makes
            # the initial gate derivative positive without any task labels.
            loss = (fused.float() * capture['readout'].detach()).sum()
            gradients = torch.autograd.grad(loss, tuple(live(core).values()))
            for (name,_), grad in zip(live(core).items(), gradients):
                self.assertTrue(bool(torch.isfinite(grad).all()))
                if name == 'alpha': self.assertGreater(float(grad), 0)
                else: self.assertEqual(int(torch.count_nonzero(grad)), 0)

    def test_nonzero_gate_every_live_parameter_and_feature_gradient(self):
        features, coords, hidden = sample()
        for core in (SummaryReadResidual(), PointwiseResidual()):
            with torch.no_grad(): core.alpha.fill_(.2)
            x = features.clone().requires_grad_(); h = hidden.clone().requires_grad_()
            memory = core.prepare_memory(x, **coords)
            delta = core(h, memory)
            probe = torch.sin(torch.arange(h.numel(), dtype=torch.float32) * .17).reshape(h.shape)
            variables = (h, *live(core).values(), *((x,) if memory is not None else ()))
            gradients = torch.autograd.grad((delta*probe).sum()/3584, variables)
            for grad in gradients:
                self.assertTrue(bool(torch.isfinite(grad).all())); self.assertGreater(float(grad.norm()), 0)
            if memory is not None: self.assertIsNone(core.pool.mass_direction.grad)

    def test_independent_tiny_fp64_read_formulas_and_gradients(self):
        g = torch.Generator().manual_seed(113)
        shapes = ((2,5),(4,5),(3,5),(3,5),(3,5),(5,3))
        actual = tuple((.3*torch.randn(s,generator=g)).requires_grad_() for s in shapes)
        expected = tuple(x.detach().double().requires_grad_() for x in actual)
        value, _ = _summary_read(*actual); reference = fp64_summary(*expected)
        torch.testing.assert_close(value.double(), reference, rtol=2e-5, atol=2e-6)
        probe = torch.linspace(-.8,.9,10).reshape(2,5)
        ga = torch.autograd.grad((value*probe).sum(), actual)
        gb = torch.autograd.grad((reference*probe.double()).sum(), expected)
        for a,b in zip(ga,gb): torch.testing.assert_close(a.double(),b,rtol=2e-4,atol=2e-6)
        h = actual[0].detach().clone().requires_grad_()
        down = (.3*torch.randn(7,5,generator=g)).requires_grad_()
        up = (.3*torch.randn(5,7,generator=g)).requires_grad_()
        output, _ = _pointwise_read(h,down,up)
        hd,dd,ud = (x.detach().double().requires_grad_() for x in (h,down,up))
        normalized = hd / torch.sqrt(hd.square().mean(-1,keepdim=True)+1e-6)
        pre = normalized @ dd.T; wanted = (pre*torch.sigmoid(pre)) @ ud.T
        torch.testing.assert_close(output.double(),wanted,rtol=2e-5,atol=2e-6)
        a = torch.autograd.grad((output*probe).sum(),(h,down,up))
        b = torch.autograd.grad((wanted*probe.double()).sum(),(hd,dd,ud))
        for x,y in zip(a,b): torch.testing.assert_close(x.double(),y,rtol=2e-4,atol=2e-6)

    def test_enriched_item_permutation_partition_and_gradients(self):
        features, coords, hidden = sample(); core = SummaryReadResidual()
        with torch.no_grad(): core.alpha.fill_(.3)
        x = features.clone().requires_grad_()
        memory = core.prepare_memory(x, **coords); output = core(hidden, memory)
        order = torch.tensor([8,3,0,7,1,6,2,5,4])
        xp = features[order].clone().requires_grad_()
        permuted = core.prepare_memory(xp, **{k:v[order] for k,v in coords.items()})
        result = core(hidden, permuted)
        torch.testing.assert_close(output,result,rtol=2e-5,atol=2e-6)
        probe = torch.linspace(-.4,.6,output.numel()).reshape(output.shape)
        ga = torch.autograd.grad((output*probe).sum(),(x,*live(core).values()))
        gb = torch.autograd.grad((result*probe).sum(),(xp,*live(core).values()))
        torch.testing.assert_close(ga[0][order],gb[0],rtol=3e-4,atol=3e-5)
        for a,b in zip(ga[1:],gb[1:]): torch.testing.assert_close(a,b,rtol=3e-4,atol=3e-5)
        chunks=[]
        for lo,hi in ((0,2),(2,6),(6,9)):
            chunks.append(core.pool.summarize(features[lo:hi],**{k:v[lo:hi] for k,v in coords.items()}))
        merged=core.pool.read(core.pool.merge(chunks[2],core.pool.merge(chunks[0],chunks[1])),retain_mass=False)
        torch.testing.assert_close(core(hidden,merged),output,rtol=2e-5,atol=2e-6)

    def test_arbitrary_rows_outer_autocast_and_pointwise_no_feature_access(self):
        features, coords, hidden = sample()
        for core in (SummaryReadResidual(), PointwiseResidual()):
            with torch.no_grad(): core.alpha.fill_(.1)
            memory = core.prepare_memory(features.half(), **coords)
            native = hidden.half(); reference = core(native,memory)
            for shape in ((3,3584),(1,1,3,3584)):
                torch.testing.assert_close(core(native.reshape(shape),memory).reshape(reference.shape),reference,rtol=0,atol=0)
            torch.testing.assert_close(core(native[0,0],memory),reference[0,0],rtol=2e-5,atol=2e-6)
            with torch.autocast('cpu',dtype=torch.bfloat16):
                actual = core(native,memory); fused = native_residual_add(native,actual)
            self.assertEqual(actual.dtype,torch.float32); self.assertEqual(fused.dtype,torch.float16)
            torch.testing.assert_close(actual,reference,rtol=0,atol=0)
        # None values would fail any feature/coordinate tensor access.
        self.assertIsNone(PointwiseResidual().prepare_memory(None,**{k:None for k in coords}))

    def test_cast_before_add_gradient_and_finite_boundaries(self):
        native = torch.ones(1,dtype=torch.float16)
        delta = torch.tensor([2**-11 + 2**-23],requires_grad=True)
        result = native_residual_add(native,delta)
        self.assertTrue(torch.equal(result,native+delta.half()))
        self.assertFalse(torch.equal(result,(native.float()+delta).half()))
        result.float().sum().backward(); self.assertTrue(torch.equal(delta.grad,torch.ones_like(delta)))
        for value in (float('nan'),float('inf'),1e20):
            with self.assertRaises(ValueError): native_residual_add(native,torch.tensor([value]))
        with self.assertRaises(ValueError): native_residual_add(torch.tensor([65504.],dtype=torch.float16),torch.tensor([65504.]))
        with self.assertRaises(ValueError): native_residual_add(native,torch.zeros(1,dtype=torch.float16))

    def test_invalid_empty_frozen_mass_and_no_label_interface(self):
        features, coords, hidden = sample(); summary = SummaryReadResidual(); pointwise = PointwiseResidual()
        memory = summary.prepare_memory(features,**coords)
        for core, mem in ((summary,memory),(pointwise,None)):
            for h in (hidden[...,:-1],hidden[:,:0],hidden.double(),torch.full_like(hidden,float('nan')),
                      torch.full_like(hidden,torch.finfo(torch.float32).max)):
                with self.assertRaises(ValueError): core(h,mem)
            with self.assertRaises(TypeError): core(hidden,mem,labels=torch.tensor([1]))
            with self.assertRaises(TypeError): core.prepare_memory(features,**coords,question='not an input')
            self.assertEqual(list(inspect.signature(core.prepare_memory).parameters),
                ['features','frame_index','raster_row','raster_col','grid_height','grid_width'])
        with self.assertRaises(ValueError): pointwise(hidden,memory)
        with self.assertRaises(ValueError): summary(hidden,None)
        with self.assertRaises(ValueError): summary.prepare_memory(features[:0],**{k:v[:0] for k,v in coords.items()})
        with self.assertRaises(ValueError): summary.prepare_memory(features,**dict(coords,frame_index=coords['frame_index']+128))
        with self.assertRaises(ValueError): summary(hidden,dict(memory,mass_feature=torch.ones_like(memory['mass_feature'])))
        with torch.no_grad(): summary.pool.mass_direction[0]=1
        with self.assertRaises(ValueError): summary(hidden,memory)
        with torch.no_grad(): summary.pool.mass_direction.zero_()
        summary.pool.mass_direction.requires_grad_(True)
        with self.assertRaises(ValueError): summary(hidden,memory)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')


def main():
    cpu_slurm(); started=time.perf_counter()
    out=REPO/'outputs/native_aggregation_vlm/native_visual_augmentation_core'/('check_'+os.environ['SLURM_JOB_ID'])
    out.mkdir(parents=True,exist_ok=False)
    own={n:digest(REPO/n) for n in OWN}
    if own[PROPOSAL]!=PROPOSAL_SHA or any(digest(REPO/n)!=h for n,h in INHERITED.items()):
        raise RuntimeError('New proposal or immutable inherited core changed')
    (out/'source').mkdir()
    for name,h in {**own,**INHERITED}.items():
        target=out/'source'/name.replace('/','_');target.write_bytes((REPO/name).read_bytes())
        if digest(target)!=h:raise RuntimeError('Source copy changed')
    policy=dict(cpu_seconds=90,cpu_cores=4,memory_gib=16,model_calls=0,vision_calls=0,head_calls=0,
        optimizer_steps=0,fixture_groups=8,no_native_integration=True,no_fit_or_inference_release=True)
    save(out/'request.json',dict(source_sha256=own,inherited_source_sha256=INHERITED,policy=policy))
    try:
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(NativeVisualAugmentationTests))
        save(out/'test_results.json',dict(tests_run=result.testsRun,failures=[dict(test=str(t),trace=s) for t,s in result.failures],
            errors=[dict(test=str(t),trace=s) for t,s in result.errors],skipped=result.skipped))
        if not result.wasSuccessful() or result.testsRun!=8 or time.perf_counter()-started>90:
            raise RuntimeError('Augmentation fixture or fixed90-second cap failed')
        if any(digest(REPO/n)!=h for n,h in {**own,**INHERITED}.items()):raise RuntimeError('Sources changed during check')
        (out/'REPORT.md').write_text('Eight augmentation CPU fixture groups passed. Algebra, parameter counts, interfaces and gradients only; no native model, head, images, optimizer or fit. Native zero/full/cached parity remains untested.\n')
        save(out/'summary.json',dict(protocol='native_visual_augmentation_core_check',passed=True,completed=True,
            source_sha256=own,inherited_source_sha256=INHERITED,policy=policy,tests_run=8,
            test_results_file=str(out/'test_results.json'),test_results_sha256=digest(out/'test_results.json'),
            report_file=str(out/'REPORT.md'),report_sha256=digest(out/'REPORT.md'),torch_version=torch.__version__,
            live_parameters_per_arm=LIVE_PARAMETERS,summary_allocated_parameters=SUMMARY_ALLOCATED_PARAMETERS,
            elapsed_seconds=time.perf_counter()-started,no_native_fidelity_claim=True))
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=own,
            inherited_source_sha256=INHERITED,policy=policy,elapsed_seconds=time.perf_counter()-started,partial_evidence_retained=True))
        raise


if __name__=='__main__':main()

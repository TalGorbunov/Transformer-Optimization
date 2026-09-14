"""Seven CPU tests for unchanged native training and JSON-safe hardware metadata."""
import copy
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest
if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION')!='cpu' or os.environ.get('SLURM_JOB_GPUS'):
    raise SystemExit('Run numerical tests only in CPU Slurm')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from scripts import train_native_identity_join_learned_v2 as train
from gnnformer.parallel_local_learned_selection import ParallelLocalLearnedSelection


class Tokenizer:
    eos_token_id=151645
    all_special_ids=[100,151645,151643]
    table={1:'Mary',2:'Sand',3:'ra',4:'John',5:' ',6:' extra',7:'mary',8:'Ｍary',100:'<bos>',151645:'<eos>',151643:'<end>'}
    def decode(self,ids,skip_special_tokens=False):
        return ''.join(self.table[i] for i in ids if not skip_special_tokens or i not in self.all_special_ids)
    def encode(self,name,add_special_tokens=False):return {'Mary':[1],'Sandra':[2,3],'John':[4]}[name]


class TrainingTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(813);torch.set_num_threads(2)

    @staticmethod
    def fixture():
        cache={'scenes':{}};values=[];index={}
        def feature(fid,x):
            if fid not in index:index[fid]=len(values);values.append(x)
            return fid
        for sid,n,target,family in [('a',1,[1,7],'x'),('b',3,[1,7],'x'),('c',2,[2,3,7],'y'),('d',3,[2,3,7],'y')]:
            ids=[]
            for i in range(n):
                ids.append([feature(f'{sid}_{i}_{t}',torch.tensor([i+1.,t+1.,.2,1.])) for t in range(len(target))])
            gids=[feature(f'g_{family}_{t}',torch.tensor([1.,.3,t+1.,2.])) for t in range(len(target))]
            cache['scenes'][sid]=dict(n_frames=n,target_ids=target,local_feature_ids=ids,global_feature_ids=gids,gold='unused')
        return cache,torch.stack(values).half(),index

    def test_ragged_padding_and_offline_metadata_do_not_select(self):
        cache,states,index=self.fixture();sids=['a','b','c','d']
        h,g,layout,valid=train.batch_states(torch,cache,states,index,sids)
        self.assertEqual(layout['offsets'],[0,2,4,7,10]);self.assertEqual(tuple(h.shape),(3,10,4))
        self.assertEqual(valid.sum(0).tolist(),[1,1,3,3,2,2,2,3,3,3])
        self.assertTrue(torch.equal(g[layout['left']],g[layout['right']]))
        poisoned=copy.deepcopy(cache)
        for row in poisoned['scenes'].values():row.update(gold='Noah',room_pair=['invented'],local_truth=[True],trio=['invented'])
        ph,pg,pl,pv=train.batch_states(torch,poisoned,states,index,sids)
        self.assertTrue(torch.equal(h,ph) and torch.equal(g,pg) and torch.equal(valid,pv));self.assertEqual(layout,pl)
        for mode in train.POLICY['conditions']:
            core=ParallelLocalLearnedSelection(4,rank=3,mode=mode)
            with torch.no_grad():core.up.weight.normal_(0,.2);core.local_bias.fill_(.8)
            full=core(h,g,valid_mask=valid)
            alone=core(h[:1,:2],g[:2],valid_mask=valid[:1,:2])
            torch.testing.assert_close(full[:2],alone,rtol=1e-3,atol=1e-3)

    def test_sequence_weighting_and_live_selection_frozen_native_gradient(self):
        cache,states,index=self.fixture();h,g,layout,valid=train.batch_states(torch,cache,states,index,['a','b','c','d'])
        head=nn.Linear(4,8,bias=False,dtype=torch.float16).requires_grad_(False)
        for mode in train.POLICY['conditions']:
            core=ParallelLocalLearnedSelection(4,rank=3,mode=mode)
            with torch.no_grad():core.up.weight.normal_(0,.2)
            total,ce,residual,parts,cap=train.losses(torch,core,h,g,layout,valid,nn.Identity(),head)
            delta=cap['delta'];logits=head((g+delta.half()).unsqueeze(0))[0].float()
            scene_losses=[];pair_losses=[]
            for a,b,tokens in zip(layout['offsets'],layout['offsets'][1:],layout['target_sequences']):
                scene_losses.append(nn.functional.cross_entropy(logits[a:b],torch.tensor(tokens)))
            for i in (0,2):
                a,b=layout['offsets'][i:i+2];c,d=layout['offsets'][i+1:i+3]
                pair_losses.append(((delta[c:d]-delta[a:b]).square().sum(-1)/(g[a:b].float().square().sum(-1)+1e-6)).mean())
            torch.testing.assert_close(ce,torch.stack(scene_losses).mean())
            torch.testing.assert_close(residual,torch.stack(pair_losses).mean())
            self.assertTrue(torch.equal(total,ce+residual));self.assertNotIn('path',parts['position_losses'])
            score_grad,=torch.autograd.grad(total,cap['scores'],retain_graph=True)
            self.assertGreater(float(score_grad.abs().sum()),0.)
            total.backward();self.assertGreater(float(core.selection_weight.grad.abs().sum()),0.)
            self.assertIsNone(head.weight.grad);self.assertFalse(h.requires_grad or g.requires_grad or states.requires_grad)

    def test_zero_up_residual_and_first_gradient_routing(self):
        cache,states,index=self.fixture();h,g,layout,valid=train.batch_states(torch,cache,states,index,['a','b'])
        head=nn.Linear(4,8,bias=False,dtype=torch.float16).requires_grad_(False)
        for mode in train.POLICY['conditions']:
            core=ParallelLocalLearnedSelection(4,rank=3,mode=mode)
            total,_,residual,_,_=train.losses(torch,core,h,g,layout,valid,nn.Identity(),head)
            self.assertTrue(train.zero_gradients(torch,core,residual)['passed']);total.backward()
            self.assertGreater(float(core.up.weight.grad.abs().sum()),0.)
            for name,p in core.named_parameters():
                if name!='up.weight':self.assertTrue(torch.equal(p.grad,torch.zeros_like(p.grad)))

    def test_canonical_pair_rng_complete_epochs_and_fixed_lr_endpoint(self):
        pairs=[dict(pair_id=str(i),contrast_id=str(i//3),variant=i%3,sids=[f'a{i}',f'b{i}'],n_frames=[8,16]) for i in range(3024)]
        order=train.presentation_order(pairs,22);other=train.presentation_order(pairs,23)
        self.assertEqual(len(order),72576);self.assertNotEqual(order,other)
        rng=random.Random(22)
        for epoch in range(1,13):
            permutation=list(range(3024));rng.shuffle(permutation)
            block=order[(epoch-1)*6048:epoch*6048]
            self.assertEqual([r['slot'] for r in block[::2]],permutation)
            self.assertEqual({r['epoch'] for r in block},{epoch})
            self.assertTrue(all(block[i]['pair_id']==block[i+1]['pair_id'] for i in range(0,6048,2)))
        self.assertAlmostEqual(train.lr(50),.001);self.assertAlmostEqual(train.lr(4536),.00001)
        with self.assertRaises(ValueError):train.lr(4537)

    def test_exact_whole_name_native_eos_and_all_failure_denominators(self):
        t=Tokenizer();self.assertEqual(train.name_target(t,'Sandra'),[2,3,151645])
        for ids in ([1,151645],[1,151643],[5,1,5,151645]):self.assertTrue(train.score_output(t,ids,'Mary')['exact'])
        for ids in ([4,151645],[2,151645],[1,6,151645],[100,1,151645],[7,151645],[8,151645],[5,5,5,1]):
            self.assertFalse(train.score_output(t,ids,'Mary')['exact'])
        self.assertTrue(train.score_output(t,[5,5,5,1],'Mary')['truncated'])
        with self.assertRaises(ValueError):train.score_output(t,[1],'Mary')
        with self.assertRaises(ValueError):train.score_output(t,[1,151645,1,151645],'Mary')

    def test_shared_training_only_timing_extensions_and_tensor_archive(self):
        rows=[]
        for variant in range(3):
            rows.append(dict(sid=f'x{variant}',contrast_id='first',variant=variant,room_pair=['Kitchen','Bathroom'],
                n_frames=16,question='original q',image_files=[dict(path=f'/tmp/{variant}_{i}',sha256='a'*64,dimensions=[512,512],mode='RGB') for i in range(16)]))
        atoms={str(i):dict(atom=['Mary','Park',i],path=f'/tmp/a{i}',sha256='b'*64,dimensions=[512,512],mode='RGB') for i in range(17,65)}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'atoms.json';train.save(path,atoms)
            cases,proof=train.timing_cases(dict(splits={'train_N16':{'samples':rows}},render_cache_file=str(path)))
        self.assertEqual(len(cases),9);self.assertTrue(proof['no_dev_or_test_scene_used'])
        for i in (0,3,6):
            self.assertEqual([c['n_frames'] for c in cases[i:i+3]],[16,32,64])
            self.assertEqual(cases[i+1]['sample']['image_files'][16:],cases[1]['sample']['image_files'][16:])
            self.assertEqual(cases[i+2]['sample']['image_files'][16:],cases[2]['sample']['image_files'][16:])
        with torch.inference_mode():x=torch.tensor([1.,2.])
        copied=train.cpu_tree(torch,{'nested':[x,{'plain':'value'}]})
        self.assertFalse(copied['nested'][0].is_inference());self.assertTrue(torch.equal(x,copied['nested'][0]))


    def test_json_hardware_metadata_has_weights_only_roundtrip_without_allowlist(self):
        import io
        import json
        from transformers.utils.quantization_config import QuantizationMethod
        enum=QuantizationMethod('bitsandbytes')
        self.assertIs(type(enum),QuantizationMethod)
        prior=set(torch.serialization.get_safe_globals())
        self.assertNotIn(QuantizationMethod,prior)
        hardware=dict(gpu='NVIDIA B200',capability=[10,0],quantization=dict(
            quant_method=enum,load_in_4bit=True,bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype='bfloat16'))
        original=copy.deepcopy(hardware)
        branch={'up.weight':torch.tensor([[.25,-.5],[1.,0.]],dtype=torch.float32)}
        unsafe=io.BytesIO();torch.save(dict(branch=branch,step=32,config={'hardware':hardware}),unsafe);unsafe.seek(0)
        self.assertIn('transformers.utils.quantization_config.QuantizationMethod',
            torch.serialization.get_unsafe_globals_in_checkpoint(unsafe))
        normalized=json.loads(json.dumps(hardware))
        self.assertEqual(normalized,hardware)
        self.assertIs(type(normalized['quantization']['quant_method']),str)
        self.assertIs(type(hardware['quantization']['quant_method']),QuantizationMethod)
        self.assertEqual(hardware,original)
        buffer=io.BytesIO();torch.save(dict(branch=branch,step=32,config={'hardware':normalized}),buffer);buffer.seek(0)
        self.assertEqual(torch.serialization.get_unsafe_globals_in_checkpoint(buffer),[]);buffer.seek(0)
        loaded=torch.load(buffer,map_location='cpu',weights_only=True)
        self.assertEqual(loaded['step'],32);self.assertEqual(loaded['config']['hardware'],normalized)
        self.assertIs(type(loaded['config']['hardware']['quantization']['quant_method']),str)
        self.assertTrue(torch.equal(loaded['branch']['up.weight'],branch['up.weight']))
        self.assertEqual(set(torch.serialization.get_safe_globals()),prior)
        self.assertEqual(train.POLICY['per_main_seconds_cap'],3300)
        self.assertEqual(train.POLICY['campaign_gpu_seconds_cap'],24000)


if __name__=='__main__':unittest.main()

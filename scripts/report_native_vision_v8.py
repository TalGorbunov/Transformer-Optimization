"""Independent V8 CPU data, paired-training and native-output audit.

No model fitting or new predictions. Both conditions use the SAME native
parallel operator; compare CE-only against CE+fixed residual consistency.
Data/native-output checks derive from the explicitly source-bound immutable V7
report. Generation archives must be exact FP32 copies of native FP16 logits.
Efficacy decisions here remain separate from subsequent checkpoint verification.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import random
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import report_native_vision_v7 as v7
from scripts import native_vision_v8_pairs as pairing
need,read,sha,objsha,save,close,bind,ledger,tensor_info=(v7.need,v7.read,v7.sha,v7.objsha,v7.save,v7.close,v7.bind,v7.ledger,v7.tensor_info)
verify_inventory,verify_cache,verify_source=v7.verify_inventory,v7.verify_cache,v7.verify_source
metrics,summarize,parse_text,learning_rate,select=v7.metrics,v7.summarize,v7.parse_text,v7.learning_rate,v7.select
DATA=v7.DATA
MODEL=v7.MODEL
OUT=REPO/'outputs/native_aggregation_vlm/v8'
CKPT=Path('/mnt/ckpts/gabriele/gnn_transformer/native_aggregation_vlm_v8')
DEV_STEPS=[972,1944,2916,3888,4860]
CONDITIONS=('ce','consistency')
SEEDS=(10,11)
OWN=('scripts/report_native_vision_v8.py','slurm/native_vision_v8_report.sbatch',
     'scripts/report_native_vision_v7.py','scripts/report_native_vision_v7_storage_repair.py',
     'scripts/native_vision_v8_pairs.py','scripts/stage_native_vision_v8_test.py',
     'scripts/stage_native_vision_v7_balanced.py','scripts/stage_native_vision_v6_test.py',
     'gnnformer/parallel_local_aggregation.py','gnnformer/parallel_local_prompts.py','gnnformer/data.py')
POLICY=dict(protocol='v8_paired_residual_consistency',native_arm='parallel',conditions=['ce','consistency'],seeds=[10,11],epochs=40,
    pair_slots_per_epoch=972,scene_slots_per_epoch=1944,scene_presentations=77760,pair_presentations=38880,
    batch_size=16,pairs_per_batch=8,steps=4860,dev_steps=DEV_STEPS,lr=.001,warmup=50,final_lr=.00001,
    weight_decay=0.,clip_norm=1.,rank=96,hidden_size=3584,parameters=1041600,merge='sum',post_activation='silu',
    native_dtype='torch.float16',branch_dtype='torch.float32',consistency_coefficients={'ce':0.,'consistency':1.},
    consistency_epsilon=1e-6,consistency_reduction='mean_over_pairs_and_separate_count_EOS_positions_of_squared_L2_ratio',
    denominator='detached_squared_L2_of_identical_frozen_global_state_plus_fixed_epsilon',
    pair_order='persistent_random.Random(seed)_fresh_canonical_pair_permutation_each_epoch',target_positions=['count','EOS'],
    ce_reduction='equal_mean_over_16_scenes_and_two_positions',max_new_tokens=4,exact_requires_eos=True,
    selection='max_dev_exact_then_min_raw_first_token_NLL_then_earliest',native_eos=[151645,151643],target_eos=151645,profile_steps=32)


def sources():return {name:sha(REPO/name) for name in OWN}


def snapshot(out):
    frozen=sources();(out/'code').mkdir()
    for name,digest in frozen.items():
        target=out/'code'/name.replace('/','_');target.write_bytes((REPO/name).read_bytes());bind(target,digest)
    save(out/'source_hashes.json',frozen)
    return frozen


def stored_native_logits(x):
    import torch
    return x.dtype==torch.float32 and bool(torch.isfinite(x).all()) and torch.equal(x,x.half().float())


def audit_data():
    from scripts import stage_native_vision_v7_balanced as balanced
    from scripts import stage_native_vision_v8_test as fresh
    bpath=DATA/'v7_balanced/main_manifest.json';spath=bpath.parent/'schedule.json'
    b=read(bpath);schedule=read(spath)
    need(b['dataset_root']==str(bpath.parent) and b['data_seed']==20260925 and b['dev_seed']==20260926,'Balanced roots/seeds differ')
    need(set(b['splits'])=={'train_N8','train_N16','dev_N16'},'Unexpected balanced splits')
    need({k:len(v['samples']) for k,v in b['splits'].items()}=={'train_N8':918,'train_N16':972,'dev_N16':72},'Balanced cardinalities differ')
    for field in ('audit','stage_plan','prior_inventory'): bind(b[field+'_file'],b[field+'_sha256'])
    ledger(b['source_sha256']);inv,excluded=verify_inventory(b['prior_inventory_file'],18)
    plan=read(b['stage_plan_file']);ledger(plan['protected_source_sha256'])
    need(plan['source_sha256']==b['source_sha256'] and plan['data_seed']==20260925 and plan['dev_seed']==20260926,'Balanced ancestry differs')
    need(schedule['manifest_file']==str(bpath) and schedule['manifest_sha256']==sha(bpath)
         and schedule['epoch_slots']==plan['epoch_slots'],'Balanced schedule binding differs')
    train={r['sid']:r for k in ('train_N8','train_N16') for r in b['splits'][k]['samples']}
    need(schedule['unique_training_sids']==sorted(train),'Unique scheduled IDs differ')
    need(schedule['slot_metadata']==[dict(slot=i,sid=s,n_frames=train[s]['n_frames'],gold=train[s]['gold'],question=train[s]['question'])
         for i,s in enumerate(schedule['epoch_slots'])],'Slot metadata changed')
    ba=balanced.audit(b,schedule['epoch_slots'],excluded);archived=read(b['audit_file'])
    need(all(archived[k]==v for k,v in ba.items()),'Independent balanced audit disagrees with archived audit')
    mains={p:read(DATA/'v8_fresh'/f'{p}_manifest.json') for p in ('main','count')}
    m=mains['main'];fi,fexcluded=verify_inventory(m['inventory_file'],21)
    need({str(bpath),str(DATA/'v7_fresh/main_manifest.json'),str(DATA/'v7_fresh/count_manifest.json')} <= set(fi['source_manifest_sha256']),
         'V8 exclusions omit V7 balanced training/dev or prior V7 tests')
    for purpose,manifest in mains.items():
        need(manifest['dataset_root']==str(DATA/'v8_fresh') and manifest['fresh_test_seed']==20260929,'Fresh root/seed differs')
        for field in ('audit','stage_plan','inventory'): bind(manifest[field+'_file'],manifest[field+'_sha256'])
        ledger(manifest['generator_code_sha256']);ledger(manifest['source_manifest_sha256'])
        need(manifest['inventory_file']==m['inventory_file'] and manifest['inventory_sha256']==m['inventory_sha256'],'Fresh inventories differ')
    fa=json.loads(json.dumps(fresh.audit_published(mains,fexcluded)));previous=read(m['audit_file'])
    need(all(previous[k]==v for k,v in fa.items()),'Independent fresh audit disagrees with archived audit')
    stage=read(m['stage_audit_file'])
    need(stage['manifest_sha256']=={p:sha(DATA/'v8_fresh'/f'{p}_manifest.json') for p in mains}
         and stage['inventory_sha256']==sha(m['inventory_file']) and stage['semantic_audit_sha256']==sha(m['audit_file']),
         'Fresh stage audit bindings differ')
    testrows=[(purpose+'_'+cell,row) for purpose,manifest in mains.items() for cell,group in manifest['splits'].items() for row in group['samples']]
    need(len(testrows)==452 and len({r['content_sha256'] for _,r in testrows})==452,'Fresh unique coverage differs')
    need(not ({r['content_sha256'] for _,r in testrows}&{r['content_sha256'] for g in b['splits'].values() for r in g['samples']}),'Train/dev/test overlap')
    paths=[bpath,spath,Path(b['audit_file']),Path(b['stage_plan_file']),Path(b['prior_inventory_file']),
           DATA/'v8_fresh/main_manifest.json',DATA/'v8_fresh/count_manifest.json',Path(m['audit_file']),
           Path(m['stage_audit_file']),Path(m['stage_plan_file']),Path(m['inventory_file'])]
    summary=dict(passed=True,balanced=ba,fresh={k:v for k,v in fa.items() if k!='records'},
        prior_manifests=21,data_bindings={str(p):sha(p) for p in paths},
        local_visual_atoms_may_recur=True,historical_saturation_exception='Only 54 N8K8 training contexts; historical reuse audited explicitly')
    return dict(summary=summary,balanced=b,schedule=schedule,train=train,
                dev=[('dev_N16',r) for r in b['splits']['dev_N16']['samples']],test=testrows)


def audit_eval(path,expected,arm,tokenizer):
    import torch
    value=read(path);bind(value['raw_file'],value['raw_sha256'])
    need(value['raw_dtype']=='torch.float32','Native generation archive dtype metadata differs')
    blob=torch.load(value['raw_file'],map_location='cpu',weights_only=True);raw=blob['raw_logits']
    rows=value['rows'];need(blob['schema_version']==1 and len(rows)==len(raw)==len(expected)==value['n'],'Evaluation coverage differs')
    vocab=json.loads((MODEL/'config.json').read_text()).get('text_config',{}).get('vocab_size')
    if vocab is None: vocab=json.loads((MODEL/'config.json').read_text())['vocab_size']
    from gnnformer.parallel_local_prompts import build_set_count_prompt
    from gnnformer.data import build_count_prompt
    for index,((cell,sample),row,x) in enumerate(zip(expected,rows,raw)):
        need(row['raw_index']==index and row['cell']==cell,'Evaluation ordering/index differs')
        need(all(row[k]==sample[k] for k in ('sid','gold','n_frames','content_sha256'))
             and row['anchor_id']==sample.get('anchor_id'),'Evaluation sample identity differs')
        ids=row['generated_ids'];steps=len(ids)
        need(1<=steps<=4 and x.shape==(steps,vocab) and stored_native_logits(x),'Malformed native raw logits')
        need(x.argmax(-1).tolist()==ids,'Output is not native raw greedy argmax')
        need(row['raw_text']==tokenizer.decode(ids,skip_special_tokens=False)
             and row['text']==tokenizer.decode(ids,skip_special_tokens=True),'Decoded text differs')
        completed=ids[-1] in (151645,151643)
        need(not any(t in (151645,151643) for t in ids[:-1]) and (completed or steps==4),'Invalid native stop sequence')
        parsed=parse_text(row['text']);count_correct=parsed==sample['gold'];exact=count_correct and completed
        need(row['prediction']==parsed and row['parseable']==(parsed is not None)
             and row['parsed_count_correct']==count_correct and row['exact']==exact
             and row['completed']==completed and row['truncated']==(not completed),'Parser/EOS/exact accounting differs')
        token=tokenizer(str(sample['gold']),add_special_tokens=False)['input_ids'][0]
        nll=float(torch.logsumexp(x[0].float(),-1)-x[0,token].float())
        need(close(nll,row['first_token_nll']),'Raw first-token NLL differs')
        counters=dict(model=steps,visual=1,language=steps,fusion=steps,broadcast=steps if arm=='parallel' else 0)
        need(row['counters']==counters,'Native call counters differ')
        meta=row['metadata'];n=sample['n_frames'];r=n+1 if arm=='parallel' else 1
        need(meta['arm']==arm and meta['sid']==sample['sid'] and meta['n_frames']==n
             and meta['question']==sample['question'] and meta['question_sha256']==objsha(sample['question'])
             and meta['global_prompt']==build_set_count_prompt(sample['question'])
             and meta['local_prompt']==(build_count_prompt(sample['question'],1) if arm=='parallel' else None),'Native prompt identity differs')
        need(meta['image_sha256']==[im['sha256'] for im in sample['image_files']]
             and meta['image_paths']==[im['path'] for im in sample['image_files']]
             and meta['prefix_ids']==[] and meta['resize']==392,'Evaluation evidence/prefix differs')
        need(meta['row_count']==r and meta['global_row']==r-1 and meta['local_elements']==(n if arm=='parallel' else 1)
             and meta['row_kinds']==(['local']*n+['global'] if arm=='parallel' else ['joint']), 'Native row layout differs')
        width=meta['prompt_width']
        need(width==meta['original_prompt_width']==max(meta['row_prompt_tokens']) and len(meta['row_prompt_tokens'])==r,'Prompt widths differ')
        need(meta['input_identity']['input_ids']['shape']==[r,width]
             and meta['input_identity']['attention_mask']['shape']==[r,width]
             and meta['input_identity']['image_grid_thw']['shape']==[n,3]
             and meta['layout']['every_unpadded_row_exact'] is True,'Native layout audit differs')
        need(meta['native_dtype']=='torch.float16' and meta['branch_dtype']=='torch.float32','Native dtype differs')
        policy=meta['generation']
        need(policy['max_new_tokens']==4 and policy['do_sample'] is False and policy['num_beams']==1
             and policy['repetition_penalty']==1 and policy['native_eos_token_ids']==[151645,151643]
             and policy['target_eos_token_id']==151645 and policy['vocabulary_mask'] is False
             and policy['other_logits_processors'] is False,'Greedy generation policy differs')
        positions=meta['generation_position_ids']
        need(len(positions)==steps and all(p['shape']==[4,r,width if j==0 else 1] for j,p in enumerate(positions)), 'Cached native position shapes differ')
    need(value['exact_count']==sum(r['exact'] for r in rows)
         and close(value['first_token_nll'],sum(r['first_token_nll'] for r in rows)/len(rows)),'Evaluation summary differs')
    return value


def bootstrap(all_rows,replicates=10000,seed=20260930):
    import numpy as np
    rng=np.random.default_rng(seed);result={}
    for purpose,ks,repeats,ns in (('main',range(9),12,(16,32,64)),('count',range(9,17),8,(32,64))):
        reference=[r for r in all_rows[('consistency',10)] if r['cell'].startswith(purpose+'_')]
        groups=defaultdict(dict)
        for r in reference: groups[r['gold']].setdefault(r['anchor_id'],set()).add(r['n_frames'])
        need(set(groups)==set(ks) and all(len(groups[k])==repeats and None not in groups[k]
             and all(v==set(ns) for v in groups[k].values()) for k in ks),'Bootstrap family coverage differs')
        ordered=[(k,anchor) for k in ks for anchor in sorted(groups[k])]
        arrays={}
        for key,rows in all_rows.items():
            lookup={(r['gold'],r['anchor_id'],r['n_frames']):float(r['exact']) for r in rows if r['cell'].startswith(purpose+'_')}
            need(len(lookup)==len(ordered)*len(ns),'Duplicate/missing paired rows')
            arrays[key]=np.asarray([[lookup[k,a,n] for n in ns] for k,a in ordered])
        indices=np.concatenate([rng.integers(0,repeats,size=(replicates,repeats))+i*repeats for i,_ in enumerate(ks)],axis=1)
        endpoints={}
        for label,columns in ([('ood',[1,2]),('N16',[0]),('N32',[1]),('N64',[2])] if purpose=='main'
                              else [('unseen',[0,1]),('N32',[0]),('N64',[1])]):
            samples={key:value[:,columns].mean(1)[indices].mean(1) for key,value in arrays.items()}
            cell={}
            for fit_seed in (10,11):
                p,j=samples[('consistency',fit_seed)],samples[('ce',fit_seed)]
                diff=p-j
                cell[str(fit_seed)]=dict(difference_ci95_pp=(100*np.quantile(diff,[.025,.975])).tolist(),
                    consistency_ci95_pp=(100*np.quantile(p,[.025,.975])).tolist(),ce_ci95_pp=(100*np.quantile(j,[.025,.975])).tolist())
            pooled=sum(samples[('consistency',s)]-samples[('ce',s)] for s in (10,11))/2
            cell['pooled_fixed_two_seeds']=dict(difference_ci95_pp=(100*np.quantile(pooled,[.025,.975])).tolist())
            endpoints[label]=cell
        result[purpose]=endpoints
    return dict(replicates=replicates,seed=seed,unit='Complete anchor family, stratified by K; same draws across arms and fixed seeds',
        inference_scope='Conditional example uncertainty for these two fitted seeds; not uncertainty over training seeds',results=result)


def independent_pairs(data,cache):
    grouped=defaultdict(list)
    for sid,row in data['train'].items():grouped[row['question'],row['gold'],row['n_frames']].append(sid)
    questions=sorted({q for q,_,_ in grouped});result=[]
    need(len(questions)==54,'Expected54 training questions')
    for q in questions:
        for k in range(9):
            left=sorted(grouped[q,k,8]);right=sorted(grouped[q,k,16])
            need(len(left)==(1 if k==8 else 2) and len(right)==2,'Pair replica support differs')
            if k==8:left=left*2
            for replica in range(2):
                a,b=left[replica],right[replica];sa,sb=cache['scenes'][a],cache['scenes'][b]
                need(sa['target_ids']==sb['target_ids'] and sa['global_feature_ids']==sb['global_feature_ids'],
                     'Paired causal targets/global features differ')
                result.append(dict(slot=len(result),pair_id=objsha(['v8_pair',q,k,replica]),question=q,gold=k,
                    replica=replica,sids=[a,b],target_ids=sa['target_ids'],global_feature_ids=sa['global_feature_ids']))
    need(len(result)==972 and Counter(s for p in result for s in p['sids'])==Counter(data['schedule']['epoch_slots']),
         'Pair schedule changes original V7 scene weights')
    return result


def expected_order(pairs,seed):
    rng=random.Random(seed);rows=[]
    for epoch in range(1,41):
        slots=list(range(972));rng.shuffle(slots)
        for slot in slots:
            pair=pairs[slot]
            for side,sid in zip(('N8','N16'),pair['sids']):
                rows.append(dict(epoch=epoch,slot=slot,pair_id=pair['pair_id'],question=pair['question'],
                    gold=pair['gold'],replica=pair['replica'],sid=sid,pair_side=side))
    return rows


def check_loss_row(row,coefficient):
    fields=('loss','ce_loss','consistency_loss','weighted_consistency_loss','ce_count_loss','ce_eos_loss',
            'consistency_count_loss','consistency_eos_loss','gradient_norm','seconds')
    need(all(math.isfinite(row[k]) and row[k]>=0 for k in fields),'Nonfinite/negative objective record')
    need(row['consistency_coefficient']==coefficient
         and close(row['weighted_consistency_loss'],coefficient*row['consistency_loss'])
         and close(row['loss'],row['ce_loss']+row['weighted_consistency_loss'])
         and close(row['ce_loss'],(row['ce_count_loss']+row['ce_eos_loss'])/2)
         and close(row['consistency_loss'],(row['consistency_count_loss']+row['consistency_eos_loss'])/2),
         'CE/consistency/total or separate causal-position arithmetic differs')
    need(row['clipped']==(row['gradient_norm']>1),'Clipping indicator differs')


def loss_summary(logs):
    fields=('loss','ce_loss','consistency_loss','weighted_consistency_loss','ce_count_loss','ce_eos_loss',
            'consistency_count_loss','consistency_eos_loss','gradient_norm')
    blocks=[]
    for start in range(0,4860,972):
        rows=logs[start:start+972]
        blocks.append(dict(first_step=start+1,last_step=start+972,
            means={key:sum(row[key] for row in rows)/len(rows) for key in fields},
            clipping_fraction=sum(row['clipped'] for row in rows)/len(rows)))
    return dict(blocks=blocks,final_update={key:logs[-1][key] for key in fields},
                validation_scope='Recorded arithmetic, targets and complete update coverage; unavailable intermediate weights are not reconstructed')


def verify_run(directory,data,tokenizer,cache,pairing_value):
    import torch
    from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
    directory=Path(directory).resolve();config=read(directory/'config.json');summary=read(directory/'summary.json')
    condition,seed=config['condition'],config['seed'];runid=config['run_id']
    need(directory.parent==OUT and directory.name==runid and runid.startswith(f'run_{condition}_s{seed}_')
         and condition in CONDITIONS and seed in SEEDS and config['arm']=='parallel' and config['profile'] is False,
         'Unregistered V8 native run')
    need(config['policy']==POLICY and all(summary.get(k)==v for k,v in config.items()),'V8 policy/config/summary differs')
    coefficient=0. if condition=='ce' else 1.
    need(config['consistency_coefficient']==coefficient and summary['passed'] is True and summary['completed'] is True
         and summary['computational_integrity_passed'] is True and summary['steps']==4860 and summary['native_test_count']==452,
         'Incomplete run or changed objective coefficient')
    verify_source(directory,config['source_sha256']);bind(config['plan_file'],config['plan_sha256'])
    plan_path=Path(config['plan_file']);plan=read(plan_path)
    need(plan_path.with_suffix('.sha256').read_text().strip()==config['plan_sha256']
         and plan['source_sha256']==config['source_sha256'] and plan['policy']==POLICY,'Training CPU plan/source differs')
    cpu=read(plan_path.parent/'summary.json')
    need(cpu['passed'] is True and cpu['plan_sha256']==config['plan_sha256'] and cpu['source_sha256']==config['source_sha256'],
         'Matching completed training CPU gate required')
    ledger(plan['artifact_bindings'])
    for path,digest in data['summary']['data_bindings'].items():
        if path in plan['artifact_bindings']:need(plan['artifact_bindings'][path]==digest,'Training/fresh data audit differs')
    need(Path(plan['train_manifest']).resolve()==DATA/'v7_balanced/main_manifest.json'
         and Path(plan['schedule_file']).resolve()==DATA/'v7_balanced/schedule.json'
         and plan['fresh_manifests']=={p:str(DATA/'v8_fresh'/f'{p}_manifest.json') for p in ('main','count')},
         'Run uses different training/dev/schedule/fresh evaluation paths')
    binding=config['cache_binding'];need(binding==plan['cache_binding'],'Cache config/plan binding differs')
    need(Path(binding['file']).resolve()==pairing.CACHE,'Only the fixed parallel V7 training cache is permitted')
    bind(binding['file'],binding['sha256'])
    need(all(config[k]==cache[k] for k in ('model','runtime','processor','native_dtypes')),'Native model/runtime/cache identity differs')
    need(config['pairing_file']==plan['pairing_file'] and config['pairing_sha256']==plan['pairing_sha256'],
         'Run pairing binding differs')
    bind(plan['pairing_file'],plan['pairing_sha256']);saved_pairing=read(plan['pairing_file'])
    need(saved_pairing==pairing_value and objsha(saved_pairing)==plan['pairing_object_sha256'],
         'Saved pairing differs from immutable metadata reconstruction')
    pairs=independent_pairs(data,cache)
    need(pairs==saved_pairing['pairs'] and objsha(pairs)==saved_pairing['pairs_sha256'],'Independent pair construction differs')
    order=expected_order(pairs,seed)
    bind(directory/'presentations.json',config['presentations_sha256'])
    need(read(directory/'presentations.json')==order and objsha(order)==config['order_sha256']==plan['order_sha256'][str(seed)]
         and len(order)==77760,'Paired presentation order differs')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);initial=ParallelLocalAggregation()
        expected_initial={k:tensor_info(v) for k,v in initial.state_dict().items()}
    need(sum(p.numel() for p in initial.parameters())==1041600 and config['initialized']==expected_initial
         and config['initialized_sha256']==objsha(expected_initial),'Fixed seed initialization or parameter count differs')
    need(Path(summary['training_file']).resolve()==directory/'training.json','Unexpected training log path')
    bind(summary['training_file'],summary['training_sha256']);logs=read(summary['training_file'])
    need(len(logs)==4860,'Missing optimizer update records')
    for step,row in enumerate(logs,1):
        positions=order[(step-1)*16:step*16];sids=[r['sid'] for r in positions]
        targets=[t for sid in sids for t in cache['scenes'][sid]['target_ids']]
        need(row['step']==step and row['sids']==sids and row['target_ids']==targets and len(targets)==32
             and row['pair_ids']==[r['pair_id'] for r in positions[::2]]
             and row['epochs']==[r['epoch'] for r in positions[::2]],'Actual target/pair/update order differs')
        need(all(positions[i]['pair_id']==positions[i+1]['pair_id'] and positions[i]['pair_side']=='N8'
                 and positions[i+1]['pair_side']=='N16' for i in range(0,16,2)),'An update split/reversed a pair')
        need(close(row['lr'],learning_rate(step),atol=1e-12),'Optimizer learning rate differs')
        check_loss_row(row,coefficient)
    need(close(summary['training_seconds'],sum(r['seconds'] for r in logs),atol=1e-4),'Training time sum differs')
    need(logs[0]['consistency_loss']==0. and logs[0]['weighted_consistency_loss']==0.,'Zero-initialized U should yield zero first-step consistency')
    need(Path(summary['first_gradients_file']).resolve()==directory/'first_gradients.json','Unexpected gradient ledger path')
    bind(summary['first_gradients_file'],summary['first_gradients_sha256']);gradients=read(summary['first_gradients_file'])
    need([row['step'] for row in gradients]==[1,2],'First two gradient records required')
    for row in gradients:
        need(set(row['pre_clip'])==set(expected_initial),'Gradient parameter coverage differs')
        for name,info in row['pre_clip'].items():
            need(info['shape']==expected_initial[name]['shape'] and info['dtype']=='torch.float32'
                 and isinstance(info['sha256'],str) and len(info['sha256'])==64,'Gradient tensor metadata differs')
    selection=read(directory/'selection.json');entries=selection['development']
    need([entry['step'] for entry in entries]==DEV_STEPS and summary['development']==entries,'All five registered dev checkpoints required')
    for entry in entries:
        step=entry['step'];devpath=directory/f'dev_{step}.json'
        need(Path(entry['dev_file']).resolve()==devpath,'Unexpected dev artifact path')
        bind(devpath,entry['dev_sha256']);dev=audit_eval(devpath,data['dev'],'parallel',tokenizer)
        need(dev['label']==f'dev_{step}' and dev['exact_count']==entry['exact_count']
             and close(dev['first_token_nll'],entry['nll']),'Raw independently rescored dev selection differs')
        checkpoint=Path(entry['checkpoint']).resolve()
        need(checkpoint==CKPT/runid/f'step_{step}.pt','Checkpoint outside exact V8 run directory')
        bind(checkpoint,entry['checkpoint_sha256']);saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        need(saved['step']==step and saved['config']==config and set(saved['branch'])==set(expected_initial),
             'Checkpoint step/config/parameter coverage differs')
        need(all(t.dtype==torch.float32 and list(t.shape)==expected_initial[k]['shape'] and bool(torch.isfinite(t).all())
                 for k,t in saved['branch'].items()),'Checkpoint native branch dtype/shape/finite contract differs')
        need(objsha({k:tensor_info(t) for k,t in saved['branch'].items()})==entry['parameter_sha256'],
             'Saved checkpoint parameter hash differs')
    best=select(entries)
    need(selection['selected']==best and summary['selected']==best,'Checkpoint is not the fixed dev optimum')
    testpath=directory/'test.json';need(Path(summary['test_file']).resolve()==testpath,'Unexpected test artifact path')
    bind(testpath,summary['test_sha256']);test=audit_eval(testpath,data['test'],'parallel',tokenizer)
    need(test['label']=='test','Unexpected final evaluation label')
    return dict(condition=condition,arm='parallel',seed=seed,run_directory=str(directory),
        config_sha256=sha(directory/'config.json'),summary_sha256=sha(directory/'summary.json'),test_sha256=sha(testpath),
        selected_step=best['step'],selected_checkpoint=str(CKPT/runid/f"step_{best['step']}.pt"),
        selected_checkpoint_sha256=best['checkpoint_sha256'],selected_parameter_sha256=best['parameter_sha256'],
        initialized_sha256=config['initialized_sha256'],presentations_sha256=config['presentations_sha256'],
        pairing_object_sha256=plan['pairing_object_sha256'],training_seconds=summary['training_seconds'],
        evaluation_seconds=test['seconds'],clipping_fraction=sum(r['clipped'] for r in logs)/4860,
        loss_diagnostics=loss_summary(logs),first_preclip_gradient_hashes=gradients[0]['pre_clip'],
        native_test_model_forwards=sum(r['counters']['model'] for r in test['rows']),native_test_visual_forwards=452,
        verified_optimizer_log_rows=4860,verified_scene_presentations=77760,verified_pair_presentations=38880,
        verified_target_positions=155520,verified_dev_evaluations=5,verified_test_examples=452),test['rows'],config


def verify_release(config,data,frozen):
    binding=config['main_release'];bind(binding['file'],binding['sha256']);release=read(binding['file'])
    need(release['passed'] is True and release['plan_file']==config['plan_file'] and release['plan_sha256']==config['plan_sha256']
         and release['source_sha256']==config['source_sha256'],'Run uses a different or failed main release')
    need(release['per_main_seconds_cap']==2700 and release['main_block_gpu_seconds_cap']==11700,'V8 resource envelope differs')
    need(release['report_source_sha256']==frozen,'Main release did not freeze this independent report ancestry')
    ledger(release['report_source_sha256'])
    own=release['release_source_sha256']
    need(set(own)=={'scripts/release_native_vision_v8_mains.py','slurm/native_vision_v8_main_release.sbatch'},
         'Unexpected main-release source ledger')
    ledger(own)
    release_directory=Path(binding['file']).parent
    need(read(release_directory/'source_hashes.json')==own,'Main-release copied ledger differs')
    for name,digest in own.items():bind(release_directory/'source'/name.replace('/','_'),digest)
    ancestor=release['native_timing_ancestor'];bind(ancestor['file'],ancestor['sha256'])
    old=read(ancestor['file']);bind(ancestor['plan_file'],ancestor['plan_sha256'])
    oldplan=read(ancestor['plan_file']);newplan=read(config['plan_file'])
    need(old['passed'] is True and old['plan_file']==ancestor['plan_file'] and old['plan_sha256']==ancestor['plan_sha256'],
         'Native timing ancestor plan or acceptance differs')
    bind(ancestor['native_profile_file'],ancestor['native_profile_sha256'])
    need(ancestor['native_profile_file']==newplan['native_profile']['file']
         and ancestor['native_profile_sha256']==newplan['native_profile']['sha256']
         and Path(oldplan['native_profile']).resolve()==Path(newplan['native_profile']['directory']).resolve()
         and oldplan['native_profile_sha256']==ancestor['native_profile_sha256'],'Native timing profile differs')
    for name in ('scripts/native_vision_v7_runtime.py','gnnformer/parallel_local_native.py',
                 'gnnformer/parallel_local_aggregation.py','gnnformer/parallel_local_prompts.py'):
        need(old['source_sha256'][name]==config['source_sha256'][name]==sha(REPO/name),
             'Native computation changed since timing profile')
    extension=old['resource_extension'];ledger(extension['source_sha256'])
    bind(extension['original_release_file'],extension['original_release_sha256'])
    original=read(extension['original_release_file'])
    need(extension['scope']=='resource_only_1800_to_2700_seconds' and extension['original_passed'] is False
         and original['passed'] is False and extension['no_model_data_optimizer_selection_or_decision_change'] is True,
         'Prior failed timing release was not explicitly preserved')
    times=old['projections']['parallel']['generation_seconds']
    need(old['projections']['parallel']['original_1800_second_passed'] is False
         and times==original['projections']['parallel']['generation_seconds'],'Native timing failure lineage differs')
    oldprofile_binding=old['profiles']['parallel'];oldprofile_path=Path(oldprofile_binding['directory'])/'summary.json'
    bind(oldprofile_path,oldprofile_binding['summary_sha256']);oldprofile=read(oldprofile_path)
    checked_binding=release['data_release'];bind(checked_binding['file'],checked_binding['sha256']);checked=read(checked_binding['file'])
    need(checked['passed'] is True and checked['source_sha256']==frozen
         and checked['data_bindings']==data['summary']['data_bindings'],'Independent data/report release differs')
    bind(checked['audit_file'],checked['audit_sha256'])
    need(set(release['profiles'])==set(CONDITIONS) and set(release['projections'])==set(CONDITIONS),
         'Both fixed software profiles/projections are required')
    for condition,item in release['profiles'].items():
        path=Path(item['directory'])/'summary.json';bind(path,item['summary_sha256']);profile=read(path)
        need(profile['profile'] is True and profile['condition']==condition and profile['seed']==10 and profile['arm']=='parallel'
             and profile['passed'] is True and profile['computational_integrity_passed'] is True
             and profile['steps']==32 and profile['no_dev_or_test_evaluation'] is True
             and profile['plan_sha256']==config['plan_sha256'] and profile['source_sha256']==config['source_sha256']
             and profile['active_cache_replay']['passed'] is True,'Matched training software profile failed or changed')
        projection=release['projections'][condition]
        training=4860*profile['step_seconds_max_steady'];development=360*times['16'];test=108*times['16']+344*times['64']
        projected=profile['model_and_features_load_seconds']+1.25*(training+development+test)+120
        need(profile['hardware']==oldprofile['hardware'] and projection['generation_seconds']==times
             and projection['N32_uses_N64_bound'] is True and close(projection['training_seconds'],training)
             and close(projection['development_seconds'],development) and close(projection['test_seconds'],test)
             and close(projection['projected_seconds'],projected),'Independent measured runtime arithmetic differs')
        need(projection['passed'] is True and math.isfinite(projection['projected_seconds'])
             and 0<projection['projected_seconds']<=2700,'Prospective runtime criterion failed')
    return dict(file=binding['file'],sha256=binding['sha256'],data_release=checked_binding,projections=release['projections'],
                release_source_sha256=own,native_timing_ancestor=ancestor,retained_original_native_timing_failure=True)


def integer_decision(treated,control):
    delta=treated['ood']['correct']-control['ood']['correct']
    id_delta=treated['cells']['main_test_N16']['correct']-control['cells']['main_test_N16']['correct']
    n64=treated['cells']['main_test_N64']['correct']
    primary=delta>=11 and id_delta>=-5
    practical=primary and treated['ood']['correct']>=152 and treated['ood_nonzero']['correct']>=116 and n64>=76
    return dict(ood_additional_correct=delta,ood_difference_pp=100*delta/216,
        n16_additional_correct=id_delta,n16_difference_pp=100*id_delta/108,
        ood_gain_at_least_5pp=delta>=11,n16_loss_at_most_5pp=id_delta>=-5,primary=primary,
        consistency_ood_at_least_70pct=treated['ood']['correct']>=152,
        consistency_nonzero_ood_at_least_60pct=treated['ood_nonzero']['correct']>=116,
        consistency_n64_correct=n64,consistency_n64_at_least_70pct=n64>=76,practical=practical)


def self_test():
    import torch
    import numpy as np
    good=torch.tensor([0.,1.,-7.5],dtype=torch.float16).float()
    need(stored_native_logits(good) and not stored_native_logits(good.half())
         and not stored_native_logits(good+1e-6) and not stored_native_logits(torch.tensor([float('nan')])),
         'Lossless native FP16-to-FP32 archive check failed')
    need(parse_text(' 12\n')==12 and parse_text('00')==0 and all(parse_text(t) is None for t in ('-1','1.','1 2','八','')),
         'Unchanged strict integer parser failed')
    def score(ood,idcorrect,nonzero,n64):
        return dict(ood=dict(correct=ood),ood_nonzero=dict(correct=nonzero),
                    cells={'main_test_N16':dict(correct=idcorrect),'main_test_N64':dict(correct=n64)})
    reference=score(141,65,0,0)
    need(integer_decision(score(152,60,116,76),reference)['practical'],'Inclusive V8 thresholds failed')
    for value in (score(151,60,116,76),score(152,59,116,76),score(152,60,115,76),score(152,60,116,75)):
        need(not integer_decision(value,reference)['practical'],'A V8 practical boundary was ignored')
    need(not integer_decision(score(152,60,116,76),score(142,65,0,0))['primary'],
         'A ten-example gain must not pass the11-example primary threshold')
    row=dict(loss=2.2,ce_loss=2.,consistency_loss=.2,weighted_consistency_loss=.2,
        ce_count_loss=1.,ce_eos_loss=3.,consistency_count_loss=.1,consistency_eos_loss=.3,
        consistency_coefficient=1.,gradient_norm=1.5,clipped=True,seconds=.01)
    check_loss_row(row,1.)
    check_loss_row(dict(row,loss=2.,weighted_consistency_loss=0.,consistency_coefficient=0.),0.)
    for change in (dict(loss=2.),dict(ce_count_loss=2.),dict(consistency_eos_loss=.8),dict(clipped=False)):
        try:check_loss_row(dict(row,**change),1.)
        except ValueError:pass
        else:raise AssertionError('Malformed component/weight/clip arithmetic was accepted')
    pairs=[dict(slot=i,pair_id=str(i),question=f'q{i}',gold=i%9,replica=i%2,sids=[f'a{i}',f'b{i}']) for i in range(972)]
    a=expected_order(pairs,10)
    need(a==expected_order(pairs,10) and a!=expected_order(pairs,11) and len(a)==77760
         and Counter(r['epoch'] for r in a[1936:1952])=={1:8,2:8}
         and all(a[i]['pair_side']=='N8' and a[i+1]['pair_side']=='N16' and a[i]['pair_id']==a[i+1]['pair_id']
                 for i in range(0,len(a),2)),'Paired order or carried epoch boundary failed')
    fixture={}
    for condition in CONDITIONS:
        for seed in SEEDS:
            rows=[]
            for purpose,ks,repeats,ns in (('main',range(9),12,(16,32,64)),('count',range(9,17),8,(32,64))):
                for k in ks:
                    for i in range(repeats):
                        for n in ns:rows.append(dict(cell=f'{purpose}_test_N{n}',gold=k,anchor_id=f'{purpose}_{k}_{i}',
                                                       n_frames=n,exact=condition=='consistency'))
            fixture[condition,seed]=rows
    intervals=bootstrap(fixture,replicates=100,seed=20260930)
    need(intervals==bootstrap(fixture,replicates=100,seed=20260930),'Bootstrap determinism failed')
    need(all(np.allclose(endpoint['pooled_fixed_two_seeds']['difference_ci95_pp'],[100,100])
             for family in intervals['results'].values() for endpoint in family.values()),'Shared paired bootstrap lost fixed difference')
    return ['exact_FP32_archive_FP16_roundtrip','strict_native_integer_parser','new_N64_practical_boundary',
            'both_objective_coefficients','separate_count_EOS_component_arithmetic','reject_malformed_loss_logs',
            'persistent_paired_seed_orders','intact_pairs_across_epoch_boundary','shared_K_stratified_family_bootstrap']


def report_runs(directories,out,frozen):
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(4);tests=self_test();data=audit_data()
    tokenizer=AutoTokenizer.from_pretrained(str(MODEL),use_fast=False,local_files_only=True)
    need(tokenizer.eos_token_id==151645,'Native target EOS differs')
    cache,cache_audit=verify_cache(pairing.CACHE,'parallel',data,tokenizer)
    pairing_value=pairing.load_pairs();need(pairing_value['pairs']==independent_pairs(data,cache),'Independent canonical pairing differs')
    save(out/'data_audit.json',dict(data['summary'],cache_audit=cache_audit))
    save(out/'pairing.json',pairing_value)
    audits={};rows={};configs={};releases=[]
    need(len(directories)==4 and len({Path(p).resolve() for p in directories})==4,'All four distinct V8 main runs required')
    for directory in directories:
        audited,predictions,config=verify_run(directory,data,tokenizer,cache,pairing_value)
        key=(audited['condition'],audited['seed']);need(key not in audits,'Duplicate condition/seed')
        audits[key],rows[key],configs[key]=audited,predictions,config
        releases.append(verify_release(config,data,frozen))
        print(json.dumps(dict(audited_run=str(directory),native_examples=len(predictions))),flush=True)
    need(set(audits)=={(c,s) for c in CONDITIONS for s in SEEDS},'Missing registered condition/seed')
    need(all(item==releases[0] for item in releases),'Mains used different data/resource/report releases')
    gradient_comparison={}
    for seed in SEEDS:
        a,b=configs['ce',seed],configs['consistency',seed]
        need(a['initialized_sha256']==b['initialized_sha256'] and a['presentations_sha256']==b['presentations_sha256']
             and a['order_sha256']==b['order_sha256'],'Matched condition initialization/order differs')
        need(all(a[k]==b[k] for k in ('policy','plan_file','plan_sha256','source_sha256','cache_binding','pairing_sha256',
                 'model','runtime','processor','native_dtypes')),'Matched treatment conditions differ')
        gradient_comparison[str(seed)]=dict(first_preclip_gradient_hashes_equal=
            audits['ce',seed]['first_preclip_gradient_hashes']==audits['consistency',seed]['first_preclip_gradient_hashes'],
            scope='Descriptive recorded hashes; zero-U first-step consistency was independently checked as zero in the logs')
    runs={f'{condition}_s{seed}':dict(audits[condition,seed],metrics=summarize(rows[condition,seed]))
          for condition in CONDITIONS for seed in SEEDS}
    decisions={str(seed):integer_decision(runs[f'consistency_s{seed}']['metrics'],runs[f'ce_s{seed}']['metrics']) for seed in SEEDS}
    primary=all(value['primary'] for value in decisions.values());practical=all(value['practical'] for value in decisions.values())
    intervals=bootstrap(rows);pooled=sum(value['ood_additional_correct'] for value in decisions.values())/432
    analysis=dict(schema_version=1,passed=True,audit_passed=True,source_sha256=frozen,policy=POLICY,
        data=data['summary'],cache=cache_audit,pairing=dict(file=str(out/'pairing.json'),sha256=sha(out/'pairing.json'),
            object_sha256=objsha(pairing_value),pair_count=972,unique_training_scenes=1890,weighted_scenes_per_epoch=1944),
        release=releases[0],self_tests=tests,runs=runs,decisions=decisions,first_gradient_comparison=gradient_comparison,
        primary_both_seeds=primary,practical_both_seeds=practical,vision_milestone_gate=primary and practical,
        post_checkpoint_verification='Separate mandatory selected-checkpoint artifact; not performed by this efficacy report',
        pooled_fixed_two_seed_ood_difference_pp=100*pooled,bootstrap=intervals,
        interpretation=dict(treatment='Fixed native-residual consistency objective versus CE-only on identical paired data and native parallel architecture',
            unchanged_inference_operator=True,additional_inference_work=False,does_not_establish_reasoning_composition=True,
            no_attention_operator_novelty_claim=True,all_final_answer_labels_used_for_pairing=True,
            primary_exact='Strict ASCII integer equals gold AND native EOS within four tokens; malformed/truncated examples stay in denominators',
            native_first_token_nll='Raw full-vocabulary first-token NLL; not full multi-digit-answer likelihood',
            raw_logit_storage='FP32 generation archives are verified exact lossless promotions of FP16 native logits',
            separate_unseen_counts='All128 K9-16 examples remain a separate whole-answer secondary endpoint',
            practical_change='V8 prospectively adds N64>=76/108; V7 decisions are unchanged',
            matched_control='Both objectives share paired ordering, scene weights, targets, initialization and 4860 optimizer updates',
            bootstrap='K-stratified complete-family intervals condition on these two fixed fitted seeds; pooling cannot rescue per-seed failure',
            local_atom_repetition=True,step_label_and_real_distribution_effects_not_identified=True,
            loss_audit='Component arithmetic and complete logged update/target coverage, not reconstruction of unavailable intermediate models'))
    save(out/'analysis.json',analysis)
    lines=['# V8 independent native evaluation','',
        f'Audit passed. Both-seed primary: **{primary}**. Both-seed practical: **{practical}**.',
        '', 'The two conditions use the same native parallel architecture and paired scenes. Only the training objective coefficient differs.',
        '', '| Condition | Seed | N16 | N32 | N64 | Familiar OOD | Nonzero OOD | Unseen counts | Selected step |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    fmt=lambda value:f"{value['correct']}/{value['n']} ({100*value['accuracy']:.1f}%)"
    for condition in CONDITIONS:
        for seed in SEEDS:
            run=runs[f'{condition}_s{seed}'];m=run['metrics']
            values=[m['cells'][f'main_test_N{n}'] for n in (16,32,64)]+[m['ood'],m['ood_nonzero'],m['unseen']]
            lines.append(f"| {condition} | {seed} | "+' | '.join(fmt(value) for value in values)+f" | {run['selected_step']} |")
    lines+=['','| Seed | OOD gain | Family bootstrap95%CI | N16 change | Primary | Practical |',
            '|---:|---:|---:|---:|---|---|']
    for seed in SEEDS:
        d=decisions[str(seed)];ci=intervals['results']['main']['ood'][str(seed)]['difference_ci95_pp']
        lines.append(f"| {seed} | {d['ood_difference_pp']:.2f}pp | [{ci[0]:.2f},{ci[1]:.2f}]pp | {d['n16_difference_pp']:.2f}pp | {d['primary']} | {d['practical']} |")
    lines+=['', 'Exact requires the correct integer and native EOS within four tokens. All452 examples per run remain in the denominator.',
        '', 'Primary requires at least11/216 additional familiar-OOD answers and no more than5/108 fewer N16 answers in each seed. Practical additionally requires at least152/216 OOD,116/192 nonzero OOD, and76/108 N64 correct.',
        '', 'Independent checks cover21 prior-manifest exclusions, unchanged V7 training/dev data and cache,972same-question/count pairs,77760scene presentations,4860logged target batches, objective component arithmetic, all five development checkpoints, and raw native argmax/EOS/NLL.',
        '', 'Final acceptance also requires the separate selected-checkpoint audit. A strong CE-only result does not establish a consistency-objective benefit. This experiment makes no reasoning-composition or new-attention claim.',
        '', '[Complete per-N/K metrics, losses, provenance and fixed-seed family intervals](analysis.json).']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return dict(passed=True,source_sha256=frozen,analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),
        report_file=str(out/'REPORT.md'),primary_both_seeds=primary,practical_both_seeds=practical,vision_milestone_gate=primary and practical)


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-test',action='store_true');mode.add_argument('--check-data',action='store_true')
    mode.add_argument('--runs','--runs4',nargs=4,type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),
         'Every report/data/self-test execution requires CPU Slurm')
    started=time.perf_counter();label='self_test' if args.self_test else ('data_check' if args.check_data else 'report')
    out=args.output or OUT/f'{label}_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False);frozen=snapshot(out)
    if args.self_test:summary=dict(passed=True,source_sha256=frozen,tests=self_test())
    elif args.check_data:
        import torch
        from transformers import AutoTokenizer
        torch.set_num_threads(4);data=audit_data()
        tokenizer=AutoTokenizer.from_pretrained(str(MODEL),use_fast=False,local_files_only=True)
        cache,cache_audit=verify_cache(pairing.CACHE,'parallel',data,tokenizer)
        value=pairing.load_pairs();need(value['pairs']==independent_pairs(data,cache),'Independent pair construction differs')
        save(out/'pairing.json',value);save(out/'data_audit.json',dict(data['summary'],cache_audit=cache_audit,pairing_object_sha256=objsha(value)))
        summary=dict(passed=True,source_sha256=frozen,data_bindings=data['summary']['data_bindings'],cache_audit=cache_audit,
            pairing_file=str(out/'pairing.json'),pairing_sha256=sha(out/'pairing.json'),pairing_object_sha256=objsha(value),
            audit_file=str(out/'data_audit.json'),audit_sha256=sha(out/'data_audit.json'))
    else:summary=report_runs(args.runs,out,frozen)
    need(sources()==frozen,'Report source changed during execution')
    summary.update(seconds=time.perf_counter()-started,slurm_job_id=os.environ['SLURM_JOB_ID']);save(out/'summary.json',summary)
    (out/'INDEX.md').write_text('# V8 independent CPU audit\n\n[Summary](summary.json) · [Sources](source_hashes.json)\n')
    print(json.dumps(dict(passed=True,summary_file=str(out/'summary.json'),seconds=summary['seconds'])),flush=True)


if __name__=='__main__':main()

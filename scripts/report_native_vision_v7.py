"""Independent V7 CPU data and native-output audit. No model or fitting work."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
DATA=Path('/mnt/data/gabriele/gnn_transformer')
OUT=REPO/'outputs/native_aggregation_vlm/v7'
MODEL=Path('/mnt/ckpts/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5')
DEV_STEPS=[972,1944,2916,3888,4860]
POLICY=dict(arms=['parallel','joint'],seeds=[8,9],epochs=40,slots_per_epoch=1944,batch_size=16,
    steps=4860,dev_steps=DEV_STEPS,lr=.001,warmup=50,final_lr=.00001,weight_decay=0.,clip_norm=1.,
    rank=96,hidden_size=3584,parameters=1041600,merge='sum',post_activation='silu',
    native_dtype='torch.float16',branch_dtype='torch.float32',max_new_tokens=4,
    selection='max_dev_exact_then_min_raw_first_token_NLL_then_earliest',native_eos=[151645,151643],
    target_eos=151645,bootstrap_seed=20260928,bootstrap_replicates=10000,exact_requires_eos=True)
OWN=('scripts/report_native_vision_v7.py','slurm/native_vision_v7_report.sbatch')
def need(condition,message):
    if not condition: raise ValueError(message)
def read(path): return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()
def objsha(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def save(path,value):
    with Path(path).open('x') as f: json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def close(a,b,atol=2e-5): return math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(a,b,rel_tol=1e-6,abs_tol=atol)
def bind(path,digest): need(sha(path)==digest,'Artifact changed: '+str(path))
def ledger(values):
    for path,digest in values.items(): bind(Path(path) if Path(path).is_absolute() else REPO/path,digest)
def tensor_info(value):
    import torch
    x=value.detach().cpu().contiguous()
    return dict(shape=list(x.shape),dtype=str(x.dtype),sha256=hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest())
def snapshot(out):
    hashes={p:sha(REPO/p) for p in OWN};(out/'code').mkdir()
    for p,h in hashes.items():
        (out/'code'/p.replace('/','_')).write_bytes((REPO/p).read_bytes());bind(out/'code'/p.replace('/','_'),h)
    save(out/'source_hashes.json',hashes);return hashes

def verify_inventory(path,expected):
    from scripts.stage_native_vision_v6_test import content_rows
    inv=read(path);need(len(inv['entries'])==expected,'Historical inventory cardinality differs')
    paths=set();excluded=set()
    for entry in inv['entries']:
        need(entry['path'] not in paths,'Duplicate historical manifest');paths.add(entry['path'])
        bind(entry['path'],entry['sha256']);rows=list(content_rows(read(entry['path'])))
        values={r['content_sha256'] for r in rows};excluded.update(values)
        need(len(rows)==entry['record_count'] and len(values)==entry['unique_content_count'],'Prior record counts differ')
    need(sorted(excluded)==inv['excluded_content_sha256'] and len(excluded)==inv['excluded_content_count']
         and objsha(sorted(excluded))==inv['excluded_content_set_sha256'],'Historical content set differs')
    need(inv['source_manifest_sha256']=={e['path']:e['sha256'] for e in inv['entries']},'Inventory source ledger differs')
    ledger(inv['protected_source_sha256'])
    return inv,excluded

def audit_data():
    from scripts import stage_native_vision_v7_balanced as balanced
    from scripts import stage_native_vision_v7_test as fresh
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
    mains={p:read(DATA/'v7_fresh'/f'{p}_manifest.json') for p in ('main','count')}
    m=mains['main'];fi,fexcluded=verify_inventory(m['inventory_file'],19)
    need(str(bpath) in fi['source_manifest_sha256'],'Fresh exclusions omit current balanced train/dev')
    for purpose,manifest in mains.items():
        need(manifest['dataset_root']==str(DATA/'v7_fresh') and manifest['fresh_test_seed']==20260924,'Fresh root/seed differs')
        for field in ('audit','stage_plan','inventory'): bind(manifest[field+'_file'],manifest[field+'_sha256'])
        ledger(manifest['generator_code_sha256']);ledger(manifest['source_manifest_sha256'])
        need(manifest['inventory_file']==m['inventory_file'] and manifest['inventory_sha256']==m['inventory_sha256'],'Fresh inventories differ')
    fa=json.loads(json.dumps(fresh.audit_published(mains,fexcluded)));previous=read(m['audit_file'])
    need(all(previous[k]==v for k,v in fa.items()),'Independent fresh audit disagrees with archived audit')
    stage=read(m['stage_audit_file'])
    need(stage['manifest_sha256']=={p:sha(DATA/'v7_fresh'/f'{p}_manifest.json') for p in mains}
         and stage['inventory_sha256']==sha(m['inventory_file']) and stage['semantic_audit_sha256']==sha(m['audit_file']),
         'Fresh stage audit bindings differ')
    testrows=[(purpose+'_'+cell,row) for purpose,manifest in mains.items() for cell,group in manifest['splits'].items() for row in group['samples']]
    need(len(testrows)==452 and len({r['content_sha256'] for _,r in testrows})==452,'Fresh unique coverage differs')
    need(not ({r['content_sha256'] for _,r in testrows}&{r['content_sha256'] for g in b['splits'].values() for r in g['samples']}),'Train/dev/test overlap')
    paths=[bpath,spath,Path(b['audit_file']),Path(b['stage_plan_file']),Path(b['prior_inventory_file']),
           DATA/'v7_fresh/main_manifest.json',DATA/'v7_fresh/count_manifest.json',Path(m['audit_file']),
           Path(m['stage_audit_file']),Path(m['stage_plan_file']),Path(m['inventory_file'])]
    summary=dict(passed=True,balanced=ba,fresh={k:v for k,v in fa.items() if k!='records'},
        prior_manifests=19,data_bindings={str(p):sha(p) for p in paths},
        local_visual_atoms_may_recur=True,historical_saturation_exception='Only 54 N8K8 training contexts; historical reuse audited explicitly')
    return dict(summary=summary,balanced=b,schedule=schedule,train=train,
                dev=[('dev_N16',r) for r in b['splits']['dev_N16']['samples']],test=testrows)

def parse_text(text): return int(text.strip()) if re.fullmatch(r'[0-9]+',text.strip()) else None

def expected_order(slots,seed):
    rng=random.Random(seed);result=[]
    for epoch in range(1,41):
        indices=list(range(1944));rng.shuffle(indices)
        result.extend(dict(epoch=epoch,slot=i,sid=slots[i]) for i in indices)
    return result

def learning_rate(step):
    return .001*step/50 if step<=50 else .00001+.00099*(1+math.cos(math.pi*(step-50)/4810))/2

def select(entries): return min(entries,key=lambda e:(-e['exact_count'],e['nll'],e['step']))

def verify_source(directory,values):
    ledger(values);need(read(directory/'source_hashes.json')==values,'Run source ledger differs')
    for path,digest in values.items(): bind(directory/'code'/path.replace('/','_'),digest)

def verify_cache(path,arm,data,tokenizer):
    import torch
    cache=read(path);bind(cache['plan_file'],cache['plan_sha256']);plan=read(cache['plan_file'])
    need(cache['complete'] is True and cache['training_only'] is True,'Incomplete or nontraining cache')
    need(set(cache['scenes'])==set(data['train']) and cache['scenes']==plan['scenes'],'Cache scene coverage differs')
    ledger(cache['source_sha256']);ledger(plan['source_sha256'])
    for record in plan.get('source_files',{}).values(): bind(record['path'],record['sha256'])
    if 'parallel_plan_file' in plan: bind(plan['parallel_plan_file'],plan['parallel_plan_sha256'])
    if 'release_file' in cache: bind(cache['release_file'],cache['release_sha256'])
    need(cache['model']==plan['model'] and cache['runtime']==plan['runtime'] and cache['processor']==plan['processor'],'Cache model/runtime ancestry differs')
    need(cache['native_dtypes']=={'norm':'torch.float16','lm_head':'torch.float16'},'Cached native dtype differs')
    need(set(cache['features'])==set(plan['features']) and cache['feature_count']==len(plan['features']),'Feature coverage differs')
    features=plan['features'];used=set()
    for sid,scene in cache['scenes'].items():
        sample=data['train'][sid]
        need(all(scene[k]==sample[k] for k in ('gold','question','n_frames','content_sha256','qa_sha256')),'Cache scene mismatch')
        targets=tokenizer(str(sample['gold']),add_special_tokens=False)['input_ids']+[tokenizer.eos_token_id]
        need(len(targets)==2 and scene['target_ids']==targets and targets[1]==151645,'Native count/EOS targets differ')
        groups=scene['local_feature_ids']+[scene['global_feature_ids']] if arm=='parallel' else [scene['feature_ids']]
        need(len(groups)==(sample['n_frames']+1 if arm=='parallel' else 1),'Feature group count differs')
        for i,ids in enumerate(groups):
            need(len(ids)==2 and ids[0]!=ids[1],'Empty and gold-prefix identities collide')
            for j,fid in enumerate(ids):
                f=features[fid];used.add(fid)
                need(f['prefix_ids']==([] if j==0 else targets[:1]),'Prediction state sees wrong answer prefix')
                if arm=='parallel':
                    need(f['question']==sample['question'],'Feature question differs')
                    kind='local' if i<sample['n_frames'] else 'global'
                    identity=f['pair_id'] if kind=='local' else sample['question']
                    need(fid==objsha([kind,identity,f['prefix_ids']]) and f['kind']==kind,'Parallel feature key differs')
                    if kind=='local':
                        pair=plan['pairs'][identity]
                        need(pair['image_sha256']==sample['image_files'][i]['sha256'] and pair['question']==sample['question'],'Local evidence order differs')
                else:
                    need(f['sid']==sid and fid==objsha(['joint',sample['content_sha256'],f['prefix_ids']]),'Joint feature identity differs')
            left,right=(plan['layouts'][features[fid]['layout_id']]['input_ids'] for fid in ids)
            need(right==left+targets[:1],'Gold-prefix feature changes pre-answer tokens')
    need(used==set(features),'Unused or missing cached features')
    locations=cache['features'];files={}
    for fid,item in locations.items():
        if item['file'] in files: need(files[item['file']]==item['file_sha256'],'Conflicting shard digests')
        files[item['file']]=item['file_sha256']
    seen=set()
    for filename,digest in files.items():
        bind(filename,digest);blob=torch.load(filename,map_location='cpu',weights_only=True)
        ids=blob['feature_ids'];states=blob['states']
        need(blob['schema_version']==1 and states.shape==(len(ids),3584) and states.dtype==torch.float16
             and bool(torch.isfinite(states).all()) and len(ids)==len(set(ids)),'Malformed feature shard')
        for row,fid in enumerate(ids):
            need(fid not in seen and fid in locations,'Duplicate/unknown feature');seen.add(fid);item=locations[fid]
            need(item['file']==filename and item['row']==row and tensor_info(states[row])['sha256']==item['state_sha256'],'Cached native state changed')
        del blob,states
    need(seen==set(features) and len(cache['shards'])==4,'Incomplete cache')
    shard_ids=set()
    for entry in cache['shards']:
        directory=Path(entry['directory']);bind(directory/'summary.json',entry['summary_sha256']);s=read(directory/'summary.json')
        need(s['completed'] and s['passed'] and s['computational_integrity_passed'] and s['profile'] is False,'Cache shard failed')
        need(s['shard']==entry['shard'] and s['shard'] not in shard_ids,'Duplicate shard');shard_ids.add(s['shard'])
        need(s['features_file'] in files and s['features_sha256']==files[s['features_file']],'Shard/file binding differs')
        bind(s['observations_file'],s['observations_sha256']);ledger(s['source_sha256'])
        need(s['plan_sha256']==cache['plan_sha256'],'Shard plan differs')
    need(shard_ids==set(range(4)),'Shard indices differ')
    profile=Path(cache['profile_directory']);bind(profile/'summary.json',cache['profile_summary_sha256'])
    prof=read(profile/'summary.json')
    if arm=='parallel':
        ext=cache['runtime_extension'];bind(ext['release_file'],ext['release_sha256']);ledger(ext['dispatcher_source_sha256'])
        need(ext['passed'] and ext['cap_seconds']==360 and ext['original_profile_passed'] is False
             and prof['passed'] is False and prof['runtime_projection_passed'] is False
             and prof['computational_integrity_passed'] and prof['numerical_gate_passed']
             and 300<max(prof['shard_projected_seconds'])<=360,'Explicit runtime-only extension differs')
    else:need(prof['passed'] is True,'Bound joint feature profile failed')
    return cache,dict(file=str(path),sha256=sha(path),plan_sha256=cache['plan_sha256'],features=len(features),
                      training_scenes=len(cache['scenes']),all_causal_prefixes_verified=True,original_profile_passed=prof['passed'],runtime_extension=cache.get('runtime_extension'))

def audit_eval(path,expected,arm,tokenizer):
    import torch
    value=read(path);bind(value['raw_file'],value['raw_sha256'])
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
        need(1<=steps<=4 and x.shape==(steps,vocab) and x.dtype==torch.float16 and bool(torch.isfinite(x).all()),'Malformed native raw logits')
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

def verify_run(directory,data,tokenizer,cache_memo):
    import torch
    from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
    directory=Path(directory).resolve();config=read(directory/'config.json');summary=read(directory/'summary.json')
    need(config['profile'] is False and config['arm'] in ('parallel','joint') and config['seed'] in (8,9),'Unregistered run')
    need(all(config['policy'].get(k)==v for k,v in POLICY.items()),'Registered policy differs')
    need(all(summary.get(k)==v for k,v in config.items()),'Summary/config mismatch')
    need(summary['passed'] and summary['completed'] and summary['computational_integrity_passed']
         and summary['steps']==4860 and summary['native_test_count']==452,'Incomplete native main run')
    verify_source(directory,config['source_sha256']);bind(config['plan_file'],config['plan_sha256']);plan=read(config['plan_file'])
    need(plan['source_sha256']==config['source_sha256'] and plan['policy']==config['policy'],'Plan/config differs')
    ledger(plan['data_bindings'])
    for path,digest in plan['data_bindings'].items():
        if path in data['summary']['data_bindings']: need(data['summary']['data_bindings'][path]==digest,'Run used different audited data')
    binding=config['cache_binding'];arm=config['arm'];seed=config['seed']
    need(plan['cache_bindings'][arm]==binding,'Wrong cache arm binding');bind(binding['file'],binding['sha256'])
    if arm not in cache_memo: cache_memo[arm]=verify_cache(binding['file'],arm,data,tokenizer)
    cache,cache_audit=cache_memo[arm]
    need(cache_audit['sha256']==binding['sha256'] and cache['plan_sha256']==binding['plan_sha256'],'Cache changed between runs')
    need(all(config[k]==cache[k] for k in ('model','runtime','processor','native_dtypes')),'Run/cache model identity differs')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);initial=ParallelLocalAggregation()
        need(sum(p.numel() for p in initial.parameters())==1041600,'Independent parameter count differs')
        expected_initial={k:tensor_info(v) for k,v in initial.state_dict().items()}
    need(config['initialized']==expected_initial and config['initialized_sha256']==objsha(expected_initial),'Initialization differs from fixed seed')
    order=expected_order(data['schedule']['epoch_slots'],seed)
    bind(directory/'presentations.json',config['presentations_sha256'])
    need(read(directory/'presentations.json')==order and len(order)==77760,'Training presentation order differs')
    logs=read(directory/'training.json');need(len(logs)==4860,'Incomplete optimizer log')
    for step,row in enumerate(logs,1):
        sids=[r['sid'] for r in order[(step-1)*16:step*16]]
        targets=[token for sid in sids for token in cache['scenes'][sid]['target_ids']]
        need(row['step']==step and row['sids']==sids and row['target_ids']==targets and len(targets)==32,'Optimizer batch/target ledger differs')
        need(close(row['lr'],learning_rate(step),atol=1e-12) and row['clipped']==(row['gradient_norm']>1), 'Optimizer schedule/clipping differs')
        need(all(math.isfinite(row[k]) and row[k]>=0 for k in ('loss','gradient_norm','seconds')),'Nonfinite training record')
    need(close(summary['training_seconds'],sum(r['seconds'] for r in logs),atol=1e-4),'Training timing sum differs')
    selection=read(directory/'selection.json');entries=selection['development']
    need([e['step'] for e in entries]==DEV_STEPS and summary['development']==entries,'Development checkpoint coverage differs')
    for entry in entries:
        step=entry['step'];need(Path(entry['dev_file']).resolve()==directory/f'dev_{step}.json','Wrong dev artifact')
        bind(entry['dev_file'],entry['dev_sha256']);dev=audit_eval(entry['dev_file'],data['dev'],arm,tokenizer)
        need(dev['label']==f'dev_{step}' and dev['exact_count']==entry['exact_count']
             and close(dev['first_token_nll'],entry['nll']),'Selection dev score differs')
        bind(entry['checkpoint'],entry['checkpoint_sha256'])
        checkpoint=torch.load(entry['checkpoint'],map_location='cpu',weights_only=True)
        need(checkpoint['step']==step and checkpoint['config']==config,'Checkpoint metadata differs')
        state=checkpoint['branch'];need(set(state)==set(expected_initial),'Checkpoint parameter set differs')
        need(all(list(v.shape)==expected_initial[k]['shape'] and v.dtype==torch.float32
             and bool(torch.isfinite(v).all()) for k,v in state.items()),'Checkpoint parameter contract differs')
        need(objsha({k:tensor_info(v) for k,v in state.items()})==entry['parameter_sha256'],'Checkpoint tensor hash differs')
        del checkpoint,state,dev
    best=select(entries)
    need(selection['selected']==best and summary['selected']==best,'Selected checkpoint violates fixed dev rule')
    need(Path(summary['test_file']).resolve()==directory/'test.json','Wrong final test file')
    bind(summary['test_file'],summary['test_sha256']);test=audit_eval(summary['test_file'],data['test'],arm,tokenizer)
    need(test['label']=='test','Wrong final evaluation label')
    result=dict(arm=arm,seed=seed,run_directory=str(directory),config_sha256=sha(directory/'config.json'),
        summary_sha256=sha(directory/'summary.json'),test_sha256=summary['test_sha256'],
        selected_step=best['step'],selected_checkpoint_sha256=best['checkpoint_sha256'],
        selected_parameter_sha256=best['parameter_sha256'],cache=cache_audit,
        initialized_sha256=config['initialized_sha256'],presentations_sha256=config['presentations_sha256'],
        training_seconds=summary['training_seconds'],evaluation_seconds=test['seconds'],
        clipping_fraction=sum(r['clipped'] for r in logs)/4860,last_training_loss=logs[-1]['loss'],
        native_model_forwards=sum(r['counters']['model'] for r in test['rows']),native_visual_forwards=452,
        verified_optimizer_updates=4860,verified_presentations=77760,verified_target_tokens=155520,
        verified_dev_evaluations=5,verified_test_examples=452)
    return result,test['rows'],config

def metrics(rows):
    need(rows,'Empty metric denominator')
    n=len(rows);correct=sum(r['exact'] for r in rows)
    return dict(n=n,correct=correct,accuracy=correct/n,
        parsed_count_correct=sum(r['parsed_count_correct'] for r in rows),
        parseable=sum(r['parseable'] for r in rows),completed=sum(r['completed'] for r in rows),
        truncated=sum(r['truncated'] for r in rows),first_token_nll=sum(r['first_token_nll'] for r in rows)/n,
        native_model_forwards=sum(r['counters']['model'] for r in rows))

def summarize(rows):
    cells={cell:metrics([r for r in rows if r['cell']==cell]) for cell in sorted({r['cell'] for r in rows})}
    by_k={f'{cell}/K{k}':metrics([r for r in rows if r['cell']==cell and r['gold']==k])
          for cell in sorted({r['cell'] for r in rows}) for k in sorted({r['gold'] for r in rows if r['cell']==cell})}
    familiar=[r for r in rows if r['cell'].startswith('main_') and r['n_frames']>16]
    nonzero=[r for r in familiar if r['gold']>0]
    unseen=[r for r in rows if r['cell'].startswith('count_')]
    need(len(familiar)==216 and len(nonzero)==192 and len(unseen)==128,'Registered denominators differ')
    return dict(all=metrics(rows),cells=cells,by_n_k=by_k,ood=metrics(familiar),ood_nonzero=metrics(nonzero),unseen=metrics(unseen))

def integer_decision(p,j):
    delta=p['ood']['correct']-j['ood']['correct']
    id_delta=p['cells']['main_test_N16']['correct']-j['cells']['main_test_N16']['correct']
    primary=delta>=11 and id_delta>=-5
    practical=primary and p['ood']['correct']>=152 and p['ood_nonzero']['correct']>=116
    return dict(ood_additional_correct=delta,ood_difference_pp=100*delta/216,
        n16_additional_correct=id_delta,n16_difference_pp=100*id_delta/108,
        ood_gain_at_least_5pp=delta>=11,n16_loss_at_most_5pp=id_delta>=-5,
        primary=primary,parallel_ood_at_least_70pct=p['ood']['correct']>=152,
        parallel_nonzero_ood_at_least_60pct=p['ood_nonzero']['correct']>=116,practical=practical)

def bootstrap(all_rows,replicates=10000,seed=20260928):
    import numpy as np
    rng=np.random.default_rng(seed);result={}
    for purpose,ks,repeats,ns in (('main',range(9),12,(16,32,64)),('count',range(9,17),8,(32,64))):
        reference=[r for r in all_rows[('parallel',8)] if r['cell'].startswith(purpose+'_')]
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
            for fit_seed in (8,9):
                p,j=samples[('parallel',fit_seed)],samples[('joint',fit_seed)]
                diff=p-j
                cell[str(fit_seed)]=dict(difference_ci95_pp=(100*np.quantile(diff,[.025,.975])).tolist(),
                    parallel_ci95_pp=(100*np.quantile(p,[.025,.975])).tolist(),joint_ci95_pp=(100*np.quantile(j,[.025,.975])).tolist())
            pooled=sum(samples[('parallel',s)]-samples[('joint',s)] for s in (8,9))/2
            cell['pooled_fixed_two_seeds']=dict(difference_ci95_pp=(100*np.quantile(pooled,[.025,.975])).tolist())
            endpoints[label]=cell
        result[purpose]=endpoints
    return dict(replicates=replicates,seed=seed,unit='Complete anchor family, stratified by K; same draws across arms and fixed seeds',
        inference_scope='Conditional example uncertainty for these two fitted seeds; not uncertainty over training seeds',results=result)

def verify_main_release(config,data,frozen):
    binding=config['main_release'];bind(binding['file'],binding['sha256']);release=read(binding['file'])
    need(release['passed'] is True,'Main resource release did not pass')
    need(release['report_source_sha256']==frozen,'Main release did not freeze this independent report')
    ledger(release['report_source_sha256'])
    need(release['plan_file']==config['plan_file'] and release['plan_sha256']==config['plan_sha256'],'Main release plan differs')
    source=release.get('source_sha256')
    if source is not None: ledger(source)
    b=release['data_release'];bind(b['file'],b['sha256']);checked=read(b['file'])
    need(checked['passed'] is True and checked['source_sha256']==frozen,'Main data release/report source differs')
    need(checked['data_bindings']==data['summary']['data_bindings'],'Main release used a different independent data audit')
    bind(checked['audit_file'],checked['audit_sha256'])
    extension=release.get('resource_extension')
    if extension is not None:
        bind(extension['original_release_file'],extension['original_release_sha256'])
        original=read(extension['original_release_file'])
        need(extension['original_passed'] is False and original['passed'] is False,'Resource extension lost the original failed decision')
        ledger(extension['source_sha256'])
    return dict(file=binding['file'],sha256=binding['sha256'],data_release=b,
                resource_extension=extension,projections=release.get('projections'))

def self_test():
    import numpy as np
    need(parse_text(' 12\n')==12 and parse_text('00')==0 and all(parse_text(x) is None for x in ('Yes.','1.','-1','1 2','')),'Strict parser self-test failed')
    need((parse_text('3')==3 and False) is False,'EOS requirement self-test failed')
    entries=[dict(step=20,exact_count=9,nll=1.),dict(step=10,exact_count=9,nll=1.),dict(step=30,exact_count=8,nll=.1)]
    need(select(entries)['step']==10,'Development selection tie-break failed')
    need(close(learning_rate(1),.00002,1e-12) and close(learning_rate(50),.001,1e-12)
         and close(learning_rate(4860),.00001,1e-12),'Learning-rate endpoints failed')
    def score(ood,n16,nonzero): return dict(ood=dict(correct=ood),ood_nonzero=dict(correct=nonzero),cells={'main_test_N16':dict(correct=n16)})
    need(integer_decision(score(152,60,116),score(141,65,0))['primary']
         and integer_decision(score(152,60,116),score(141,65,0))['practical'],'Inclusive integer threshold failed')
    need(not integer_decision(score(152,60,116),score(142,65,0))['primary']
         and not integer_decision(score(152,59,116),score(141,65,0))['primary']
         and not integer_decision(score(151,60,116),score(140,65,0))['practical']
         and not integer_decision(score(152,60,115),score(141,65,0))['practical'],'Threshold boundary rejection failed')
    slots=[str(i) for i in range(1944)];a=expected_order(slots,8)
    need(a==expected_order(slots,8) and a!=expected_order(slots,9) and len(a)==77760
         and set(Counter(r['sid'] for r in a).values())=={40},'Presentation reconstruction failed')
    fixtures={}
    for arm in ('parallel','joint'):
        for seed in (8,9):
            rows=[]
            for purpose,ks,repeats,ns in (('main',range(9),12,(16,32,64)),('count',range(9,17),8,(32,64))):
                for k in ks:
                    for i in range(repeats):
                        for n in ns: rows.append(dict(cell=f'{purpose}_test_N{n}',gold=k,anchor_id=f'{purpose}_{k}_{i}',n_frames=n,exact=arm=='parallel'))
            fixtures[arm,seed]=rows
    result=bootstrap(fixtures,replicates=100,seed=20260928)
    need(result==bootstrap(fixtures,replicates=100,seed=20260928),'Bootstrap is nondeterministic')
    for purpose in result['results'].values():
        for cell in purpose.values():
            need(np.allclose(cell['pooled_fixed_two_seeds']['difference_ci95_pp'],[100,100]),'Paired bootstrap lost a constant arm difference')
    return ['strict_parser','EOS_required','selection_ties','optimizer_schedule','integer_threshold_edges',
            'matched_complete_orders','K_stratified_family_bootstrap_shared_across_seeds']

def report_runs(directories,out,frozen):
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(min(4,int(os.environ.get('SLURM_CPUS_PER_TASK','1'))))
    tests=self_test();data=audit_data();save(out/'data_audit.json',data['summary'])
    tokenizer=AutoTokenizer.from_pretrained(str(MODEL),use_fast=False,local_files_only=True)
    need(tokenizer.eos_token_id==151645,'Native target EOS differs')
    audits={};rows={};configs={};cache_memo={};releases=[]
    need(len(set(Path(p).resolve() for p in directories))==4,'Four distinct main run directories required')
    for directory in directories:
        audit,records,config=verify_run(directory,data,tokenizer,cache_memo)
        key=(audit['arm'],audit['seed']);need(key not in audits,'Duplicate arm/seed')
        audits[key]=audit;rows[key]=records;configs[key]=config
        releases.append(verify_main_release(config,data,frozen))
        print(json.dumps(dict(audited_run=str(directory),native_test_examples=len(records))),flush=True)
    need(set(audits)=={(a,s) for a in ('parallel','joint') for s in (8,9)},'Missing fixed arm/seed')
    need(all(r==releases[0] for r in releases),'Mains used different resource/data/report releases')
    for seed in (8,9):
        p,j=configs['parallel',seed],configs['joint',seed]
        need(p['initialized_sha256']==j['initialized_sha256'] and p['presentations_sha256']==j['presentations_sha256'],'Matched arms differ in initialization/order')
        need(all(p[k]==j[k] for k in ('policy','plan_file','plan_sha256','source_sha256','model','runtime','processor','native_dtypes')),'Matched arm invariants differ')
    summaries={f'{a}_s{s}':dict(audits[a,s],metrics=summarize(rows[a,s])) for a,s in audits}
    decisions={str(s):integer_decision(summaries[f'parallel_s{s}']['metrics'],summaries[f'joint_s{s}']['metrics']) for s in (8,9)}
    primary=all(d['primary'] for d in decisions.values());practical=all(d['practical'] for d in decisions.values())
    intervals=bootstrap(rows)
    pooled_delta=sum(d['ood_additional_correct'] for d in decisions.values())/432
    analysis=dict(schema_version=1,passed=True,audit_passed=True,source_sha256=frozen,data=data['summary'],
        release=releases[0],self_tests=tests,runs=summaries,decisions=decisions,
        primary_both_seeds=primary,practical_both_seeds=practical,
        vision_milestone_gate=primary and practical,pooled_fixed_two_seed_ood_difference_pp=100*pooled_delta,
        bootstrap=intervals,
        interpretation=dict(treatment='Whole parallel isolated local computation plus learned native SUM fusion versus same-parameter joint hidden adaptation',
            cannot_attribute_to_sum_alone=True,does_not_establish_reasoning_composition=True,
            native_first_token_nll='Raw full-vocabulary first-token NLL; unseen multi-digit values are not scored as whole-answer likelihood',
            primary_exact='Strict nonnegative integer parse equals gold AND native EOS completed within four tokens',
            parsed_count_correct='Secondary parse-only metric includes unfinished but numerically correct outputs',
            data_change='Balanced q x N x K training differs from V1-V6; only the new matched joint control is a primary comparator',
            numerical_scope='Known native quantized batching/cache differences remain descriptive; original failed runtime/numerical artifacts are not erased',
            local_atom_repetition=True,bootstrap='Fixed-seed paired family intervals; pooled estimates cannot rescue a failed per-seed criterion'))
    save(out/'analysis.json',analysis)
    lines=['# V7 independent native evaluation','',
        f'Audit passed. Both-seed primary: **{primary}**. Both-seed practical threshold: **{practical}**.','',
        'Exact match requires the correct integer and EOS completion within four generated tokens. Every one of the 452 test examples per run remains in the denominator.','',
        '| Arm | Seed | N16 familiar | Familiar OOD | Nonzero OOD | Unseen counts | Selected step |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ('joint','parallel'):
        for seed in (8,9):
            a=summaries[f'{arm}_s{seed}'];m=a['metrics']
            fmt=lambda x:f"{x['correct']}/{x['n']} ({100*x['accuracy']:.1f}%)"
            lines.append(f"| {arm} | {seed} | {fmt(m['cells']['main_test_N16'])} | {fmt(m['ood'])} | {fmt(m['ood_nonzero'])} | {fmt(m['unseen'])} | {a['selected_step']} |")
    lines+=['','| Seed | OOD additional correct | OOD difference | Family bootstrap 95% CI | N16 difference | Primary | Practical |',
            '|---:|---:|---:|---:|---:|---|---|']
    for seed in (8,9):
        d=decisions[str(seed)];ci=intervals['results']['main']['ood'][str(seed)]['difference_ci95_pp']
        lines.append(f"| {seed} | {d['ood_additional_correct']}/216 | {d['ood_difference_pp']:.2f} pp | [{ci[0]:.2f}, {ci[1]:.2f}] pp | {d['n16_difference_pp']:.2f} pp | {d['primary']} | {d['practical']} |")
    lines+=['',f'Pooled fixed-two-seed OOD difference: {100*pooled_delta:.2f} percentage points (descriptive).',
        '', 'Primary integer gates, required separately in both seeds: at least 11 more correct OOD answers out of 216; no more than 5 fewer N16 answers out of 108. Practical requires primary plus at least 152/216 OOD and 116/192 nonzero OOD correct.',
        '', 'This contrast tests the complete parallel local computation and fusion treatment. It does not isolate SUM, equalize inference compute, or establish reasoning composition. Training data were newly balanced; local image atoms can recur. Family intervals condition on these two fitted seeds.',
        '', 'Raw full-vocabulary logits, all five development selections, all checkpoint digests, all 4,860 target batches, native call counts, and the 19-manifest data exclusions were independently checked. First-token NLL is distinct from whole-answer likelihood for multi-digit counts.',
        '', '[Complete per-N/K metrics, provenance, numerical/runtime exceptions, and bootstrap](analysis.json).']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return dict(passed=True,source_sha256=frozen,analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),
        report_file=str(out/'REPORT.md'),primary_both_seeds=primary,practical_both_seeds=practical,vision_milestone_gate=primary and practical)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--check-data',action='store_true');group.add_argument('--self-test',action='store_true')
    group.add_argument('--runs','--runs4',nargs=4,type=Path)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),'CPU Slurm allocation required')
    begin=time.perf_counter();mode='data_check' if args.check_data else ('self_test' if args.self_test else 'report')
    out=args.output or OUT/f'{mode}_{os.environ["SLURM_JOB_ID"]}'
    out.mkdir(parents=True,exist_ok=False);frozen=snapshot(out)
    if args.check_data:
        import torch
        from transformers import AutoTokenizer
        torch.set_num_threads(4)
        data=audit_data();result=data['summary'];tokenizer=AutoTokenizer.from_pretrained(str(MODEL),use_fast=False,local_files_only=True)
        caches={arm:verify_cache(DATA/dirname/'feature_cache.json',arm,data,tokenizer)[1]
            for arm,dirname in (('parallel','v7_parallel_local'),('joint','v7_joint_features'))}
        save(out/'data_audit.json',dict(result,cache_audits=caches))
        summary=dict(passed=True,source_sha256=frozen,data_bindings=result['data_bindings'],cache_audits=caches,audit_file=str(out/'data_audit.json'),audit_sha256=sha(out/'data_audit.json'))
    elif args.self_test: summary=dict(passed=True,source_sha256=frozen,tests=self_test())
    else: summary=report_runs(args.runs,out,frozen)
    ledger(frozen);summary.update(seconds=time.perf_counter()-begin,slurm_job_id=os.environ['SLURM_JOB_ID'])
    save(out/'summary.json',summary)
    (out/'INDEX.md').write_text('# V7 independent CPU audit\n\n[Summary](summary.json) · [Sources](source_hashes.json)\n')
    print(json.dumps(dict(passed=True,summary_file=str(out/'summary.json'),seconds=summary['seconds'])),flush=True)

if __name__=='__main__':main()

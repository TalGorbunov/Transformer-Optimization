"""Fresh main1400 native inputs/features; frozen renderer and V3 processor math."""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import prepare_mmred_fresh_evaluation as fresh
from scripts import stage_mmred_official_evaluation_features as old
from scripts import render_mmred_official_shards as rendering
from scripts.stage_mmred_official_recovery import need,sha,object_sha,save
renderer=rendering.p
compact=old.compact;preparation=old.preparation;p=old.p;producer=old.producer
PROTOCOL='mmred_fresh_native_inputs'
OUT=REPO/'outputs/native_aggregation_vlm/mmred_fresh_native'
DATA=Path('/mnt/data/gabriele/gnn_transformer/mmred_fresh_native')
PROPOSAL='docs/paper/MMRED_FRESH_NATIVE_INPUTS_PROTOCOL.md'
OWN=('scripts/stage_mmred_fresh_native.py',PROPOSAL,
 'slurm/mmred_fresh_native_prepare.sbatch','slurm/mmred_fresh_native_shard.sbatch',
 'slurm/mmred_fresh_native_merge.sbatch','slurm/mmred_fresh_native_harvest.sbatch')
POLICY=dict(protocol=PROTOCOL,prepare_cpu_seconds=600,shard_cpu_seconds=3000,merge_cpu_seconds=600,
 harvest_gpu_seconds=3600,cpu_cores=4,memory_gib=16,shards=4,worlds_per_shard=350,
 worlds=1400,frames=28800,lengths=[8,16,32],worlds_by_length={'8':400,'16':400,'32':600},
 feature_rows=5644800,feature_width=3584,feature_dtype='torch.float16',raw_feature_bytes=40461926400,
 vision_calls=1400,model_calls=0,decoder_calls=0,head_calls=0,maximum_GPUs=1,
 maximum_attempts_per_phase=1,maximum_array_attempts=1,main_only=True,diagnostic_questions_processed=0,
 original_render_pixels=512,native_processor_pixels=392,merged_tokens_per_frame=196,
 no_adapter_or_weight_changes=True,no_predictions=True,no_training_or_inference_release=True,
 frozen_renderer_namespace=rendering.NAMESPACE,resource_forecast_is_empirical=True)
CAPS={'prepare':600,'shard':3000,'merge':600,'harvest':3600}


def read(path):return json.loads(Path(path).read_text())


def ref(path):
 path=Path(path).resolve();return dict(file=str(path),sha256=sha(path))


def bind(path,bindings,expected=None):
 value=ref(path);need(expected is None or value['sha256']==expected,'Bound input changed: '+value['file'])
 bindings[value['file']]=value['sha256'];return value


def source_maps():
 fo,fi,_=fresh.source_maps();inherited={}
 for mapping in (fo,fi,old.sources(),old.inherited_sources(),rendering.sources(),rendering.inherited_sources()):
  for name,digest in mapping.items():
   need(name not in inherited or inherited[name]==digest,'Native input source ancestry conflict');inherited[name]=digest
 own={name:sha(REPO/name) for name in OWN}
 need(not set(own).intersection(inherited) and all(sha(REPO/name)==digest for name,digest in inherited.items()),'Native input source changed')
 return own,inherited


def check_time(started,phase):
 need(time.perf_counter()-started<CAPS[phase]-20,'Fixed native-input work deadline; preserve partial artifacts')


def published(out,phase,bindings,**extra):
 own,inherited=source_maps()
 artifacts={str(file):sha(file) for file in out.iterdir() if file.is_file()
  and file.name not in ('plan.json','summary.json','failure.json')}
 plan=dict(protocol=PROTOCOL,phase=phase,policy=POLICY,passed=True,completed=True,
  source_sha256=own,inherited_source_sha256=inherited,input_bindings=bindings,artifacts=artifacts,
  no_predictions=True,no_training_or_inference_release=True,**extra)
 save(out/'plan.json',plan)
 return dict(plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'))


def verify_report(path,phase):
 path=Path(path).resolve();summary=read(path);own,inherited=source_maps()
 need(summary['protocol']==PROTOCOL and summary['phase']==phase and summary['policy']==POLICY
  and summary['passed'] is summary['completed'] is True and not (path.parent/'failure.json').exists()
  and 0<summary['elapsed_seconds']<=CAPS[phase] and summary['source_sha256']==own
  and summary['inherited_source_sha256']==inherited,'Complete source-bound native input stage required')
 need(sha(summary['plan_file'])==summary['plan_sha256'],'Native input plan changed');plan=read(summary['plan_file'])
 need(plan['protocol']==PROTOCOL and plan['phase']==phase and plan['policy']==POLICY
  and plan['source_sha256']==own and plan['inherited_source_sha256']==inherited
  and plan['passed'] is plan['completed'] is plan['no_predictions'] is plan['no_training_or_inference_release'] is True,
  'Native input plan/source/scope differs')
 for mapping in (plan['input_bindings'],plan['artifacts']):
  for file,digest in mapping.items():need(sha(file)==digest,'Bound native input metadata changed')
 for key,value in plan.items():
  if key.endswith('_file'):need(value in plan['artifacts'] or value in plan['input_bindings'],
   'Published external file lacks a direct input binding: '+key)
 for name,digest in {**own,**inherited}.items():
  need(sha(path.parent/'source'/name.replace('/','_'))==digest,'Native input source archive differs')
 return plan,dict(file=str(path),sha256=sha(path),plan_file=summary['plan_file'],plan_sha256=summary['plan_sha256'])


def reference_metadata(path,bindings):
 # Only feature/compact metadata; no fitted checkpoint, predictions or head arrays.
 plan,descriptor=old.verify_report(path,'harvest')
 bind(descriptor['file'],bindings,descriptor['sha256']);bind(descriptor['plan_file'],bindings,descriptor['plan_sha256'])
 for key in ('feature_index_file','timings_file'):bind(plan[key],bindings,plan['artifacts'][plan[key]])
 features=read(plan['feature_index_file']);timings=read(plan['timings_file'])
 need(len(features)==1000 and [r['index'] for r in features]==list(range(1000))
  and plan['counters']==old.harvest.counters_expected(1000),'Complete original feature measurements required')
 for row in features:
  need(row['n'] in (8,16,32) and row['finite'] is True and row['tensor']['dtype']=='torch.float16'
   and row['tensor']['shape']==[row['n']*196,3584]
   and 0<row['vision_seconds']<=row['feature_phase_seconds']<=row['full_world_seconds'],'Original feature timing/shape differs')
 c=plan['compact_stage'];bind(c['file'],bindings,c['sha256']);summary=read(c['file'])
 need(summary['plan_file']==c['plan_file'] and summary['plan_sha256']==c['plan_sha256'],'Original compact descriptor differs')
 bind(c['plan_file'],bindings,c['plan_sha256']);prepared=read(c['plan_file'])
 need(prepared['native_identity']==plan['native_identity'] and prepared['native_identity_sha256']==plan['native_identity_sha256'],
  'Original feature/processor identities differ')
 return dict(stage=descriptor,setup_seconds=timings['setup_seconds'],
  feature_phase_seconds_by_n={str(n):max(r['feature_phase_seconds'] for r in features if r['n']==n) for n in (8,16,32)},
  native_identity=plan['native_identity'],native_identity_sha256=plan['native_identity_sha256'],
  precision=plan['precision'],packages=plan['packages'],system_prompt=plan['system_prompt'],profile_hardware=prepared['profile_hardware'])


def prepare(cohort_path,reference_path,out,started):
 bindings={};cohort=fresh.verify_stage(cohort_path);cohort_ref=bind(cohort_path,bindings)
 summary=read(cohort_path);cohort_ref.update(plan_file=summary['plan_file'],plan_sha256=summary['plan_sha256'])
 bind(summary['plan_file'],bindings,summary['plan_sha256'])
 for key in ('rows','questions','populations','original_protocol'):
  bind(cohort[key+'_file'],bindings,cohort[key+'_sha256'])
 original=reference_metadata(reference_path,bindings)
 need(read(cohort['original_protocol_file'])['system_prompt']==original['system_prompt'],'Original system prompt changed')
 questions=[json.loads(line) for line in Path(cohort['questions_file']).read_text().splitlines()]
 populations=read(cohort['populations_file']);indices=populations['main']
 selected=[questions[i] for i in indices]
 need(len(selected)==1400 and all(r['cohort']=='main' for r in selected)
  and Counter(r['n'] for r in selected)=={8:400,16:400,32:600}
  and len({r['world_sha256'] for r in selected})==1400,'Exact fresh main-only population required')
 renderer_rows=read(cohort['rows_file']);rows=[];frames={}
 for index,(cohort_index,row) in enumerate(zip(indices,selected)):
  check_time(started,'prepare');sample=renderer_rows[cohort_index]
  need(sample=={key:row[key] for key in ('qid','seq_len','qtype','atype','question','answer','sequence')},
   'Fresh question/renderer row ownership differs')
  keys=[]
  for step,frame in enumerate(sample['sequence'],1):
   key,identity,assignment=renderer.frame_identity(frame,step);keys.append(key)
   value=dict(key=key,identity=identity,assignment=assignment,step=step)
   need(key not in frames or frames[key]==value,'Fresh render frame identity collision');frames[key]=value
  rows.append(dict(row,index=index,cohort_index=cohort_index,pilot_role='fresh_main',
   renderer_file=cohort['rows_file'],renderer_row=cohort_index,source_file=cohort['rows_file'],source_row=cohort_index,
   raw_row_sha256=object_sha(sample),exact_sequence_sha256=object_sha(sample['sequence']),frame_keys=keys))
 shards=[dict(index=i,indices=list(range(i,1400,4)),
  keys=sorted({key for row in rows if row['index']%4==i for key in row['frame_keys']})) for i in range(4)]
 need(sum(len(r['frame_keys']) for r in rows)==28800 and all(len(s['indices'])==350 and len(s['keys'])<=7200 for s in shards),
  'Fresh native frame/shard inventory differs')
 prefix=[next(r['index'] for r in rows if r['n']==n) for n in (8,16,32)]
 local={cohort_index:index for index,cohort_index in enumerate(indices)}
 groups=dict(main=list(range(1400)),main_N32_primary=[local[i] for i in populations['main_N32_primary']],
  main_N32_controls=[local[i] for i in populations['main_N32_controls']],
  main_by_length={str(n):[r['index'] for r in rows if r['n']==n] for n in (8,16,32)})
 recovery_summary=read(fresh.RECOVERY);recovery_ref=bind(recovery_summary['plan_file'],bindings,recovery_summary['plan_sha256'])
 for name,value in (('rows',rows),('frames',frames),('shards',shards),('populations',groups),('reference',original)):
  save(out/(name+'.json'),value)
 return published(out,'prepare',bindings,fresh_cohort=cohort_ref,reference_features=original,
  recovery_plan=recovery_ref,rows_file=str(out/'rows.json'),frame_inventory_file=str(out/'frames.json'),
  shards_file=str(out/'shards.json'),populations_file=str(out/'populations.json'),prefix_indices=prefix,
  frame_render_calls=sum(len(s['keys']) for s in shards),distinct_frame_keys=len(frames),
  duplicate_keys_across_workers_accounted=True,**{k:original[k] for k in
   ('native_identity','native_identity_sha256','precision','packages','system_prompt','profile_hardware')})


def shard(path,index,out,data,started,progress):
 plan,descriptor=verify_report(path,'prepare');bindings={};bind(descriptor['file'],bindings,descriptor['sha256'])
 bind(descriptor['plan_file'],bindings,descriptor['plan_sha256']);shards=read(plan['shards_file']);spec=shards[index]
 rows=read(plan['rows_file']);frames=read(plan['frame_inventory_file'])
 recovery=plan['recovery_plan'];bind(recovery['file'],bindings,recovery['sha256'])
 visualization,Image,environment=renderer.renderer_environment(read(recovery['file']),bindings)
 need(object_sha(environment)==rendering.NAMESPACE,'Frozen renderer/font/package environment changed')
 save(out/'render_environment.json',environment);images={}
 with (out/'images.jsonl').open('x') as journal:
  for ordinal,key in enumerate(spec['keys']):
   check_time(started,'shard');tick=time.perf_counter();frame=frames[key];target=data/key;target.mkdir()
   visualization.frame2png(frame['assignment'],frame['step'],target)
   entry=renderer.inspect_png(target/f'frame_{frame["step"]:04d}.png',Image)
   entry['path']=entry.pop('file');entry.update(cache_key=key,step=frame['step'],cache_namespace=rendering.NAMESPACE)
   images[key]=entry;journal.write(json.dumps(dict(ordinal=ordinal,key=key,image=entry,seconds=time.perf_counter()-tick),sort_keys=True)+'\n')
   if ordinal%64==0:journal.flush()
   progress['rendered_images']=len(images)
 save(out/'image_index.json',images)
 import torch
 from transformers import AutoProcessor,__version__ as tf_version
 from scripts.stage_native_vision_v6_teacher import model_metadata
 torch.set_num_threads(4)
 processor=AutoProcessor.from_pretrained(str(preparation.MODEL),trust_remote_code=True,use_fast=False,local_files_only=True)
 owner,fn,api=p.backend.native_api(processor);rope=lambda **kw:fn(owner,**kw);identity=plan['native_identity']
 need(api==identity['native_api'] and p.fingerprint(processor,str(tf_version))==identity['processor']
  and p.backend.runtime_identity()==identity['runtime'] and model_metadata()==identity['model'],'Frozen CPU native environment differs')
 records=[];rendered=[]
 with (out/'cases.jsonl').open('x') as journal:
  for idx in spec['indices']:
   check_time(started,'shard');tick=time.perf_counter();row=rows[idx]
   row=dict(row,n_frames=row['n'],image_files=[dict(path=images[k]['path'],sha256=images[k]['sha256']) for k in row['frame_keys']])
   case=preparation.prepare_joint(processor,row,plan['system_prompt'],rope);pixels=case['inputs']['pixel_values']
   pixel_info=p.tensor_info(pixels);full=compact.tensor_tree(torch,p,case);expected=dict(full);expected['inputs']=dict(full['inputs'])
   del expected['inputs']['pixel_values'];del case['inputs']['pixel_values']
   need(compact.tensor_tree(torch,p,case)==expected,'Compaction changed more than pixels')
   file=data/f'case_{idx:04d}.pt';torch.save(dict(case=case,omitted_pixel_values=pixel_info),file)
   restored=torch.load(file,map_location='cpu',weights_only=True)
   need(set(restored)=={'case','omitted_pixel_values'} and restored['omitted_pixel_values']==pixel_info
    and compact.tensor_tree(torch,p,restored['case'])==expected,'Native compact roundtrip differs')
   record=dict(index=idx,sid=row['sid'],n=row['n'],qtype=row['qtype'],pilot_role='fresh_main',file=str(file),
    sha256=sha(file),bytes=file.stat().st_size,raw_row_sha256=row['raw_row_sha256'],
    metadata_sha256=object_sha(case['metadata']),compact_identity_sha256=object_sha(expected),
    omitted_pixel_values=pixel_info,grid_identity=p.tensor_info(case['inputs']['image_grid_thw']),
    pixel_bytes=pixels.numel()*pixels.element_size(),target_text=case['target_text'],target_ids=case['target_ids'],**compact.widths(case))
   journal.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n');journal.flush()
   record['staging_seconds']=time.perf_counter()-tick;records.append(record);rendered.append(row)
   progress['prepared_cases']=len(records);del case,pixels,restored,full,expected
 need([r['index'] for r in records]==spec['indices'] and len(images)==len(spec['keys']),'Complete fresh CPU shard required')
 for name,value in (('rows',rendered),('cases',records)):save(out/(name+'.json'),value)
 return published(out,'shard',bindings,preparation=descriptor,shard_index=index,array_job_id=os.environ['SLURM_ARRAY_JOB_ID'],
  rows_file=str(out/'rows.json'),cases_file=str(out/'cases.json'),image_index_file=str(out/'image_index.json'),
  rendered_images=len(images),prepared_cases=len(records))


def projection(records,reference,*,completed=(),elapsed=None,setup=None):
 done={r['index'] for r in completed};F=dict(reference['feature_phase_seconds_by_n'])
 for row in completed:F[str(row['n'])]=max(F[str(row['n'])],row['feature_phase_seconds'])
 costs={str(n):max(r['staging_seconds'] for r in records if r['n']==n)+F[str(n)] for n in (8,16,32)}
 base=reference['setup_seconds'] if setup is None else max(setup,reference['setup_seconds'])
 if elapsed is not None:base=elapsed
 work=sum(costs[str(row['n'])] for row in records if row['index'] not in done);seconds=base+1.25*work+60
 need(all(math.isfinite(v) and v>=0 for v in (base,work,seconds,*F.values(),*costs.values())),'Invalid feature resource estimate')
 return dict(passed=seconds<=3600,projected_seconds=seconds,cap_seconds=3600,
  setup_or_elapsed_seconds=base,elapsed_at_prefix_gate=elapsed,remaining_worlds=1400-len(done),
  completed_indices=[r['index'] for r in completed],feature_phase_seconds_by_n=F,full_world_seconds_by_n=costs,
  formula='setup_or_elapsed + 1.25 * sum_remaining(CPU_staging_max_N + feature_phase_max_N) + 60',
  empirical_not_guarantee=True,no_accuracy_or_model_output_access=True)


def merge(path,paths,out,started):
 parent,parent_ref=verify_report(path,'prepare');bindings={};bind(parent_ref['file'],bindings,parent_ref['sha256'])
 bind(parent_ref['plan_file'],bindings,parent_ref['plan_sha256'])
 bind(parent['populations_file'],bindings,parent['artifacts'][parent['populations_file']])
 need(len(paths)==4,'Exactly four CPU shard reports required');reports={};rows={};records={};images={}
 spec=read(parent['shards_file']);original_rows=read(parent['rows_file']);frames=read(parent['frame_inventory_file']);arrays=set()
 for path in paths:
  check_time(started,'merge');plan,descriptor=verify_report(path,'shard');i=plan['shard_index']
  need(i not in reports and plan['preparation']==parent_ref,'Duplicate/mismatched CPU shard');reports[i]=descriptor;arrays.add(plan['array_job_id'])
  bind(descriptor['file'],bindings,descriptor['sha256']);bind(descriptor['plan_file'],bindings,descriptor['plan_sha256'])
  shard_images=read(plan['image_index_file']);need(sorted(shard_images)==spec[i]['keys'],'CPU shard render-key inventory differs')
  for key,entry in shard_images.items():
   need(sha(entry['path'])==entry['sha256'] and entry['step']==frames[key]['step']
    and entry['cache_namespace']==rendering.NAMESPACE,'Rendered PNG identity differs')
   if key in images:need(all(images[key][k]==entry[k] for k in ('sha256','pixel_sha256','mode','width','height','step')),
    'Duplicate cross-worker frame render differs')
   images[key]=entry
  shard_rows=read(plan['rows_file']);shard_records=read(plan['cases_file'])
  need([r['index'] for r in shard_rows]==[r['index'] for r in shard_records]==spec[i]['indices'],'CPU row/compact order differs')
  for row,record in zip(shard_rows,shard_records):
   idx=row['index'];need(idx not in rows and {k:row[k] for k in original_rows[idx]}==original_rows[idx]
    and row['image_files']==[dict(path=shard_images[k]['path'],sha256=shard_images[k]['sha256']) for k in row['frame_keys']]
    and all(record[k]==row[k] for k in ('index','sid','n','qtype','pilot_role','raw_row_sha256'))
    and sha(record['file'])==record['sha256'],'Merged fresh compact/image ownership differs')
   need(record['target_text']==preparation.canonical_target(row['atype'],row['answer'])
    and record['target_rows']==len(record['target_ids']) and 0<len(record['target_ids'])<=50
    and record['target_ids'][-1]==151645 and record['grid_identity']['shape']==[row['n'],3]
    and record['omitted_pixel_values']['shape']==[row['n']*784,1176],'Merged native target/grid ownership differs')
   rows[idx]=row;records[idx]=record
 need(set(reports)==set(range(4)) and len(arrays)==1 and set(rows)==set(records)==set(range(1400))
  and set(images)==set(frames),'Complete four-shard main1400 merge required')
 rows=[rows[i] for i in range(1400)];records=[records[i] for i in range(1400)]
 gate=projection(records,parent['reference_features'])
 order=parent['prefix_indices']+[i for i in range(1400) if i not in parent['prefix_indices']]
 for name,value in (('rows',rows),('cases',records),('image_index',images),('target_inventory',compact.target_inventory(records)),
  ('projection',gate),('harvest_order',order)):save(out/(name+'.json'),value)
 return published(out,'merge',bindings,preparation=parent_ref,shard_reports=[reports[i] for i in range(4)],array_job_id=next(iter(arrays)),
  rows_file=str(out/'rows.json'),cases_file=str(out/'cases.json'),image_index_file=str(out/'image_index.json'),
  target_inventory_file=str(out/'target_inventory.json'),harvest_order_file=str(out/'harvest_order.json'),
  populations_file=parent['populations_file'],prefix_indices=parent['prefix_indices'],projection=gate,resource_eligible=gate['passed'],
  fresh_cohort=parent['fresh_cohort'],reference_features=parent['reference_features'],**{k:parent[k] for k in
   ('native_identity','native_identity_sha256','precision','packages','system_prompt','profile_hardware')})


def harvest(path,out,data,started,progress):
 plan,descriptor=verify_report(path,'merge')
 need(plan['resource_eligible'] and plan['projection']['passed'],'CPU resource projection does not release vision harvest')
 rows=read(plan['rows_file']);records=read(plan['cases_file']);order=read(plan['harvest_order_file'])
 need(len(rows)==len(records)==1400 and sorted(order)==list(range(1400)),'Fresh harvest population/order differs')
 import torch
 from gnnformer.runtime import load_runtime
 torch.set_num_threads(4);need(torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA B200','One B200 required')
 loaded=load_runtime(str(preparation.MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda')
 model=loaded.model.eval().requires_grad_(False);hardware=p.frozen_joint.live_identity(torch,loaded,plan)
 need(hardware==plan['profile_hardware'] and p.installed({})==plan['packages'],'Fixed native hardware/packages differ')
 refs=list(model.named_parameters());base=p.base_metadata(refs);p.check_base(refs,base,model)
 need(not p.adapter_parameters(model) and not any(v.training for v in model.modules()),'Frozen base without adapters required')
 owner,fn,api=p.backend.native_api(loaded.processor);rope=lambda **kw:fn(owner,**kw)
 need(api==plan['native_identity']['native_api'],'Original native position API differs')
 counts=Counter(old.harvest.counters_expected(0));features=[];progress.update(counters=counts,features=features)
 save(out/'base_before.json',base);save(out/'hardware.json',hardware);setup=time.perf_counter()-started;gate=None
 with ExitStack() as hooks,(out/'features.jsonl').open('x') as stream:
  def count(key):
   def callback(*_):
    counts[key]+=1;need(key=='vision','Fresh feature harvest unexpectedly called model/decoder/norm/head')
   return callback
  for module,key in ((model,'model'),(model.model,'backbone'),(model.model.language_model,'language'),
   (model.model.language_model.norm,'norm'),(model.lm_head,'head'),(model.model.visual,'vision')):
   hooks.callback(module.register_forward_pre_hook(count(key)).remove)
  for execution_index,idx in enumerate(order):
   check_time(started,'harvest');torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
   record=records[idx];row=rows[idx];need(sha(record['file'])==record['sha256'],'Consumed fresh compact changed')
   packet=torch.load(record['file'],map_location='cpu',weights_only=True)
   need(set(packet)=={'case','omitted_pixel_values'} and packet['omitted_pixel_values']==record['omitted_pixel_values']
    and object_sha(compact.tensor_tree(torch,p,packet['case']))==record['compact_identity_sha256'],'Fresh compact ownership differs')
   case=preparation.prepare_joint(loaded.processor,row,plan['system_prompt'],rope);pixels=case['inputs']['pixel_values']
   pixel_info=p.tensor_info(pixels);grid_info=p.tensor_info(case['inputs']['image_grid_thw'])
   need(pixel_info==packet['omitted_pixel_values'] and grid_info==record['grid_identity'],'Reprocessed fresh pixels/grid differ')
   del case['inputs']['pixel_values']
   need(compact.tensor_tree(torch,p,case)==compact.tensor_tree(torch,p,packet['case']),'Reprocessed full fresh compact differs')
   coordinates={k:p.tensor_info(v) for k,v in case['coordinates'].items()};proof_at=time.perf_counter()
   torch.cuda.synchronize();vision_start=time.perf_counter();before=counts['vision']
   with torch.inference_mode():value=producer.extract_features(torch,model,pixels,case['inputs']['image_grid_thw'])
   torch.cuda.synchronize();vision_end=time.perf_counter()
   need(counts['vision']==before+1 and value.dtype==torch.float16 and value.shape==(row['n']*196,3584)
    and bool(value.isfinite().all()),'One finite original native feature tensor required')
   tensor=p.tensor_info(value);input_identity=dict(pixel_values=pixel_info,image_grid_thw=grid_info)
   compact_ref=dict(file=record['file'],sha256=record['sha256'],compact_identity_sha256=record['compact_identity_sha256'])
   ownership=dict(index=idx,sid=row['sid'],n=row['n'],input_identity=input_identity,compact=compact_ref,
    coordinate_identity=coordinates,native_identity_sha256=plan['native_identity_sha256'])
   file=data/f'features_{idx:04d}.pt';torch.save(dict(features=value.cpu(),**ownership),file)
   item=dict(ownership,pilot_role='fresh_main',qtype=row['qtype'],execution_index=execution_index,file=str(file),
    sha256=sha(file),bytes=file.stat().st_size,tensor=tensor,finite=True,raw_feature_bytes=value.numel()*value.element_size(),
    vision_seconds=vision_end-vision_start,max_allocated_bytes=torch.cuda.max_memory_allocated(),
    max_reserved_bytes=torch.cuda.max_memory_reserved())
   p.check_base(refs,base,model);check_time(started,'harvest');torch.cuda.synchronize()
   stream.write(json.dumps(item,sort_keys=True,allow_nan=False)+'\n');stream.flush();end=time.perf_counter()
   item.update(feature_phase_seconds=end-proof_at,full_world_seconds=end-tick);features.append(item)
   progress['completed_worlds']=len(features);del value,pixels,packet,case
   if execution_index==2:
    gate=projection(records,plan['reference_features'],completed=features,elapsed=time.perf_counter()-started,setup=setup)
    save(out/'prefix_features.json',features);save(out/'prefix_projection.json',gate)
    need([r['index'] for r in features]==plan['prefix_indices'] and gate['passed'],'Three retained feature measurements exceed remaining3600s budget')
 need(gate is not None and gate['passed'] and dict(counts)==old.harvest.counters_expected(1400)
  and sum(r['raw_feature_bytes'] for r in features)==40461926400,'Complete fresh feature/call/byte inventory differs')
 p.check_base(refs,base,model);save(out/'base_after.json',p.base_metadata(refs))
 need(read(out/'base_before.json')==read(out/'base_after.json'),'Frozen native weights changed')
 features.sort(key=lambda r:r['index'])
 cases=[dict(row=row,compact=record,features=feature) for row,record,feature in zip(rows,records,features)]
 for name,value in (('feature_index',features),('cases',cases),('counters',dict(counts))):save(out/(name+'.json'),value)
 timings=dict(setup_seconds=setup,worlds=1400,vision_calls=1400,raw_feature_bytes=40461926400,
  full_world_seconds=sum(r['full_world_seconds'] for r in features),vision_seconds=sum(r['vision_seconds'] for r in features),
  saved_bytes=sum(r['bytes'] for r in features),peak_allocated_bytes=max(r['max_allocated_bytes'] for r in features),
  peak_reserved_bytes=max(r['max_reserved_bytes'] for r in features))
 save(out/'timings.json',timings);bindings={};bind(descriptor['file'],bindings,descriptor['sha256'])
 bind(descriptor['plan_file'],bindings,descriptor['plan_sha256'])
 parent_bindings={**plan['input_bindings'],**plan['artifacts']}
 for key in ('rows_file','cases_file','populations_file'):
  bind(plan[key],bindings,parent_bindings[plan[key]])
 return published(out,'harvest',bindings,compact_stage=descriptor,fresh_cohort=plan['fresh_cohort'],
  rows_file=plan['rows_file'],compact_cases_file=plan['cases_file'],cases_file=str(out/'cases.json'),
  feature_index_file=str(out/'feature_index.json'),populations_file=plan['populations_file'],timings_file=str(out/'timings.json'),
  prefix_features_file=str(out/'prefix_features.json'),prefix_projection_file=str(out/'prefix_projection.json'),
  counters=dict(counts),projection=gate,original_projection=plan['projection'],
  all_features_finite_native_FP16=True,feature_file_hashes_are_in_bound_index=True,**{k:plan[k] for k in
   ('native_identity','native_identity_sha256','precision','packages','system_prompt')})


def verify_stage(path,streaming=False):
 plan,_=verify_report(path,'harvest');parent,parent_ref=verify_report(plan['compact_stage']['file'],'merge')
 need(parent_ref==plan['compact_stage'],'Fresh compact-stage descriptor differs')
 need(plan['rows_file']==parent['rows_file'] and plan['compact_cases_file']==parent['cases_file']
  and plan['populations_file']==parent['populations_file']
  and all(plan[k]==parent[k] for k in ('native_identity_sha256','precision','packages','system_prompt')),
  'Fresh inherited output/native configuration differs')
 rows=read(parent['rows_file']);records=read(parent['cases_file']);features=read(plan['feature_index_file'])
 need(len(rows)==len(records)==len(features)==1400 and [r['index'] for r in features]==list(range(1400))
  and plan['counters']==old.harvest.counters_expected(1400) and plan['all_features_finite_native_FP16'] is True,
  'Complete source-ordered fresh native bank required')
 for row,record,item in zip(rows,records,features):
  need(all(item[k]==record[k]==row[k] for k in ('index','sid','n','qtype','pilot_role'))
   and item['finite'] is True and item['tensor']['shape']==[row['n']*196,3584] and item['tensor']['dtype']=='torch.float16'
   and item['input_identity']==dict(pixel_values=record['omitted_pixel_values'],image_grid_thw=record['grid_identity'])
   and item['compact']==dict(file=record['file'],sha256=record['sha256'],compact_identity_sha256=record['compact_identity_sha256'])
   and item['native_identity_sha256']==plan['native_identity_sha256']
   and 0<item['vision_seconds']<=item['feature_phase_seconds']<=item['full_world_seconds'],'Fresh feature/input ownership differs')
  if not streaming:need(sha(item['file'])==item['sha256'] and sha(record['file'])==record['sha256'],'Fresh native payload changed')
 need(read(plan['cases_file'])==[dict(row=r,compact=c,features=v) for r,c,v in zip(rows,records,features)],
  'Native load_inputs case records differ')
 prefix=[features[i] for i in parent['prefix_indices']];gate=read(plan['prefix_projection_file'])
 expected=projection(records,parent['reference_features'],completed=prefix,elapsed=gate['elapsed_at_prefix_gate'])
 need(gate==expected==plan['projection'] and gate['passed'] and read(plan['prefix_features_file'])==prefix
  and plan['original_projection']==parent['projection'] and parent['resource_eligible']
  and plan['native_identity']==parent['native_identity'] and plan['fresh_cohort']==parent['fresh_cohort'],
  'Fresh native resource/parent proof differs')
 return plan


def main():
 ap=argparse.ArgumentParser(description=__doc__);group=ap.add_mutually_exclusive_group(required=True)
 for phase in CAPS:group.add_argument('--'+phase,action='store_true')
 ap.add_argument('--cohort',type=Path);ap.add_argument('--reference-features',type=Path)
 ap.add_argument('--prepared',type=Path);ap.add_argument('--shards',nargs=4,type=Path);ap.add_argument('--compact',type=Path)
 args=ap.parse_args();phase=next(key for key in CAPS if getattr(args,key));started=time.perf_counter()
 need(os.environ.get('SLURM_JOB_ID') and int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==4,'Four-core Slurm required')
 if phase=='harvest':p.native.require_slurm(gpu=True);need(not os.environ.get('SLURM_ARRAY_JOB_ID'),'Exactly one non-array GPU worker')
 else:need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),'CPU-only native preparation')
 if phase=='shard':
  index=int(os.environ.get('SLURM_ARRAY_TASK_ID','-1'));array=os.environ.get('SLURM_ARRAY_JOB_ID')
  need(index in range(4) and array,'Four fixed CPU shard indices required');tag=f'shard_{array}_{index}'
  need(not any(p.name.split('_')[1]!=array for p in OUT.glob('shard_*')),'Earlier CPU shard-array attempt requires a new reviewed release')
 else:
  index=None;tag=f'{phase}_{os.environ["SLURM_JOB_ID"]}';need(not any(OUT.glob(phase+'_*')),'One attempt per phase; no automatic retry')
 out=OUT/tag;data=DATA/tag;out.mkdir(parents=True,exist_ok=False);data.mkdir(parents=True,exist_ok=False)
 own,inherited=source_maps();(out/'source').mkdir();progress={}
 for name,digest in {**own,**inherited}.items():
  file=out/'source'/name.replace('/','_');file.write_bytes((REPO/name).read_bytes());need(sha(file)==digest,'Native source archive differs')
 save(out/'request.json',dict(phase=phase,policy=POLICY,arguments={k:str(v) for k,v in vars(args).items()},
  source_sha256=own,inherited_source_sha256=inherited))
 try:
  if phase=='prepare':
   need(args.cohort and args.reference_features,'Fresh cohort and measured reference feature summary required')
   result=prepare(args.cohort,args.reference_features,out,started)
  elif phase=='shard':
   need(args.prepared,'Prepared fresh rendering/input plan required');result=shard(args.prepared,index,out,data,started,progress)
  elif phase=='merge':
   need(args.prepared and args.shards,'Prepared plan and four shard summaries required');result=merge(args.prepared,args.shards,out,started)
  else:
   need(args.compact,'Complete fresh compact merge required');result=harvest(args.compact,out,data,started,progress)
  need(time.perf_counter()-started<CAPS[phase] and source_maps()==(own,inherited),'Native source/time boundary changed')
  save(out/'summary.json',dict(protocol=PROTOCOL,phase=phase,policy=POLICY,passed=True,completed=True,
   source_sha256=own,inherited_source_sha256=inherited,elapsed_seconds=time.perf_counter()-started,**result))
 except BaseException as exc:
  save(out/'failure.json',dict(protocol=PROTOCOL,phase=phase,passed=False,type=type(exc).__name__,message=str(exc),
   elapsed_seconds=time.perf_counter()-started,progress=progress,source_sha256=own,inherited_source_sha256=inherited,
   partial_outputs_retained=True,no_predictions=True,no_automatic_retry=True));raise


if __name__=='__main__':main()

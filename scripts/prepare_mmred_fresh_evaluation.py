"""Fixed fresh original-MMReD data only; CPU Slurm, no images/models/results."""
from __future__ import annotations
import ast
from collections import Counter, defaultdict, deque
from contextlib import ExitStack
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import sys
import time

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import audit_mmred_suffix_feasibility as f
from scripts import stage_mmred_official_recovery as recovery
need=f.need;sha=f.sha;object_sha=f.object_sha
PROTOCOL='mmred_fresh_evaluation_cohort'
PROPOSAL='docs/paper/MMRED_FRESH_EVALUATION_COHORT_PROTOCOL.md'
PROPOSAL_SHA='85b8ac2708e1ef7ca35fd1c6f6debae5201dcb00a7e005215019141b1e34c6ae'
DESIGN='docs/paper/MMRED_FRESH_EVALUATION_COHORT_DESIGN.md'
DESIGN_SHA='b69d57f6c76fe52da5c39ce875b901fc2ef4f3e40ef491c44831474408fc59d1'
OWN=('scripts/prepare_mmred_fresh_evaluation.py',PROPOSAL,'slurm/mmred_fresh_evaluation_prepare.sbatch')
PINNED={'scripts/audit_mmred_suffix_feasibility.py':'748187f9f4770080a9030bb04a76b5ee738232ae5bffcf69eb0f7ad7cb147898',
 'scripts/stage_mmred_official_recovery.py':'161163301ff56949598b6fcf921245ecc1c15cebcbec9bf070b5ed2d36fb6770',DESIGN:DESIGN_SHA}
RECOVERY=REPO/'outputs/native_aggregation_vlm/mmred_official_recovery/recovery_443939/summary.json'
RECOVERY_SHA='71c7cbc6ccc8bd30fed1b1a4465cb36a15ba09e20ac0935b976b258897cef5db'
FEASIBILITY=REPO/'outputs/native_aggregation_vlm/mmred_suffix_feasibility/audit_444121/summary.json'
FEASIBILITY_SHA='fe369079c1dbcbbf6c42804141cf02574c889e5adaeab47548a968ba51e1094a'
USER_DATA=Path('/mnt/data/gabriele/gnn_transformer')
DATA=USER_DATA/'mmred_fresh_evaluation'
OUT=REPO/'outputs/native_aggregation_vlm/mmred_fresh_evaluation'
CHARS=tuple(sorted(f.CHARS));ROOMS=tuple(sorted(f.ROOMS))
TASKS=('spend_together','where_spend','steps_in_room','char_at_frame')
POLICY=dict(protocol=PROTOCOL,cpu_seconds=600,work_deadline_seconds=540,cpu_cores=4,memory_gib=16,
 main_lengths=[8,16,32],main_worlds=1400,diagnostic_families=160,worlds=2040,questions=2360,world_frame_occurrences=47232,
 main_attempts_per_stratum=100000,main_attempt_cap=1800000,diagnostic_attempts_per_length=250000,
 changed_quotas={'16':16,'32':64},main_world_seed_base=71000000,main_question_seed_base=72000000,
 diagnostic_world_seeds={'16':81000016,'32':81000032},diagnostic_choice_seeds={'16':82000016,'32':82000032},
 first_changed_candidates_not_replaced=True,control_matching='earliest_unused_exact_bin_fifo',
 exact_world_exclusion=True,prefix_exclusion=False,permutation_orbit_exclusion=False,
 primary_population='main_N32_two_aggregation_tasks',primary_worlds=400,primary_contrast_deferred_to_separate_statistics_protocol=True,
 ordinary_anchor_comparison_essential=True,statistics_protocol_separate=True,images=0,models=0,tensors=0,gpu_calls=0,
 no_training_or_inference_release=True,no_current_model_output_access=True,no_automatic_retry=True)


def read(path):return json.loads(Path(path).read_text())


def save(path,value):
 with Path(path).open('x') as stream:json.dump(value,stream,sort_keys=True,allow_nan=False,separators=(',',':'));stream.write('\n')


def emit(stream,value):stream.write(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':'))+'\n')


def bind(path,bindings,expected=None):
 path=Path(path).resolve();digest=sha(path);need(expected is None or digest==expected,'Input changed: '+str(path))
 bindings[str(path)]=digest;return dict(file=str(path),sha256=digest)


def source_maps():
 need(sha(REPO/PROPOSAL)==PROPOSAL_SHA and all(sha(REPO/name)==digest for name,digest in PINNED.items()),'Fresh data proposal/pinned helper changed')
 own={name:sha(REPO/name) for name in OWN};fo,fi,bindings=f.source_maps()
 inherited={**fo,**fi,**recovery.sources(),**recovery.INHERITED,**PINNED}
 need(all(sha(REPO/name)==digest for name,digest in inherited.items()),'Source ancestry conflict')
 return own,inherited,bindings


def deadline(started):need(time.perf_counter()-started<POLICY['work_deadline_seconds'],'Data work deadline reached; preserve shortfall and publication margin')


def named_positions(frames):
 return [[f.ROOMS[row[f.CHARS.index(person)]] for person in CHARS] for row in frames]


def canonical_hash(frames):
 return object_sha(dict(characters=CHARS,rooms=ROOMS,step_ids=list(range(1,len(frames)+1)),positions=named_positions(frames)))


def from_states(states):
 need(isinstance(states,list) and states and [row.get('step_id') for row in states]==list(range(1,len(states)+1)),'Ordered Step1..N state sequence required')
 frames=[]
 for state in states:
  rooms=state['rooms'];need(set(rooms)==set(ROOMS),'Not the original six-room inventory')
  need(all(isinstance(x,list) for x in rooms.values()) and Counter(p for group in rooms.values() for p in group)==Counter(CHARS),'Not one copy of all original five people')
  lookup={person:f.ROOMS.index(room) for room,people in rooms.items() for person in people}
  frames.append(tuple(lookup[person] for person in f.CHARS))
 result=tuple(frames);f.validate(result);return result


def legacy_recovery_hash(frames):
 return recovery.object_sha([[ROOMS.index(room) for room in row] for row in named_positions(frames)])


class Exclusions:
 def __init__(self,stream):
  self.stream=stream;self.worlds=set();self.prefixes=Counter();self.full=Counter();self.events=Counter()
 def add(self,frames,origin):
  f.validate(frames);digest=canonical_hash(frames);self.events[origin['kind']]+=1
  first=digest not in self.worlds
  if first:
   self.worlds.add(digest);self.full[(len(frames),digest)]+=1
   for k in (1,2,4,8,16,32):
    if k<=len(frames):self.prefixes[(k,canonical_hash(frames[:k]))]+=1
  emit(self.stream,dict(world_sha256=digest,n=len(frames),first_occurrence=first,origin=origin))
  return digest
 def overlap(self,frames):
  return [dict(length=k,exposed_prefix_worlds=self.prefixes[(k,canonical_hash(frames[:k]))],
    exposed_complete_worlds=self.full[(k,canonical_hash(frames[:k]))]) for k in (1,2,4,8,16,32) if k<=len(frames)]


def parents(bindings):
 rs=read(RECOVERY);fs=read(FEASIBILITY)
 bind(RECOVERY,bindings,RECOVERY_SHA);bind(FEASIBILITY,bindings,FEASIBILITY_SHA)
 need(rs['passed'] is rs['completed'] is fs['passed'] is fs['completed'] is True
  and rs['protocol']==recovery.PROTOCOL and fs['protocol']==f.PROTOCOL,'Passed fixed recovery and feasibility required')
 bind(rs['plan_file'],bindings,rs['plan_sha256']);rp=read(rs['plan_file'])
 bind(fs['analysis_file'],bindings,fs['analysis_sha256']);fp=read(fs['analysis_file'])
 need(rp['policy']==recovery.POLICY and fp['policy']==f.POLICY and fp['world_trials']==40000,'Exact parent data policies required')
 for summary,path in ((rs,RECOVERY),(fs,FEASIBILITY)):
  need(not (path.parent/'failure.json').exists(),'A parent failure is present')
  for name,digest in {**summary['source_sha256'],**summary['inherited_source_sha256']}.items():
   need(sha(REPO/name)==digest,'Parent source changed');bind(path.parent/'source'/name.replace('/','_'),bindings,digest)
 identities=RECOVERY.parent/'row_identities.json';bind(identities,bindings,rp['artifacts'][str(identities)])
 bind(rp['official_protocol_file'],bindings,rp['artifacts'][rp['official_protocol_file']])
 bind(fp['ledger_file'],bindings,fp['ledger_sha256']);bind(fp['tests']['file'],bindings,fp['tests']['sha256'])
 return rp,fp,read(identities),read(rp['official_protocol_file'])


def recovery_exposure(rp,identities,exclusions,bindings,started):
 import pyarrow as pa
 import pyarrow.ipc as ipc
 expected={(row['source_file'],row['source_row']):row for row in identities}
 need(len(identities)==len(expected)==39600,'All39600 original source identities required')
 files=sorted(path for path in rp['input_bindings'] if path.endswith('.arrow'))
 need(len(files)==18,'All18 original Arrow files required');seen=set()
 for path in files:
  bind(path,bindings,rp['input_bindings'][path]);index=0
  with pa.memory_map(path,'r') as source:
   for batch in ipc.open_stream(source):
    for row in batch.to_pylist():
     if index%100==0:deadline(started)
     key=(path,index);entry=expected[key];parsed,old_world,_=recovery.parse_row(row,entry['n'])
     need(recovery.object_sha(row)==entry['raw_row_sha256'] and recovery.object_sha(old_world)==entry['world_sha256']
      and recovery.object_sha(parsed['sequence'])==entry['exact_sequence_sha256'],'Original legacy row/world identity differs')
     frames=from_states(parsed['sequence']);need(legacy_recovery_hash(frames)==entry['world_sha256'],'Named-world canonicalization changed original semantics')
     exclusions.add(frames,dict(kind='original_recovery',file=path,row=index,legacy_world_sha256=entry['world_sha256']))
     seen.add(key);index+=1
 need(seen==set(expected),'Original exposure inventory incomplete')
 return dict(rows=len(seen),files=len(files),all_legacy_hashes_verified=True)


def proposal_draw(world_rng,choice_rng,n):
 frames=f.world(world_rng,n);c=choice_rng.randrange(1,n);q=choice_rng.randrange(5)
 people=tuple(sorted(choice_rng.sample([p for p in range(5) if p!=q],2)))
 rooms=tuple(sorted(choice_rng.sample(range(6),2)))
 return frames,c,q,people,rooms


def feasibility_exposure(fp,exclusions,recompute,started):
 seen=0;legal=0
 with Path(fp['ledger_file']).open() as stream:
  for n in f.POLICY['lengths']:
   wr=random.Random(f.POLICY['world_seeds'][str(n)]);qr=random.Random(f.POLICY['choice_seeds'][str(n)])
   for trial in range(f.POLICY['attempts_per_length']):
    if trial%100==0:deadline(started)
    row=json.loads(next(stream));frames,c,q,people,rooms=proposal_draw(wr,qr,n)
    need((row['n'],row['trial'],row['cut'],row['queried_actor'],row['person_pair'],row['room_pair'])==
     (n,trial,c,f.CHARS[q],[f.CHARS[i] for i in people],[f.ROOMS[i] for i in rooms]) and f.object_sha(frames)==row['world_sha256'],
     'Feasibility source/choice/legacy hash replay differs')
    exclusions.add(frames,dict(kind='feasibility_source',n=n,trial=trial,legacy_world_sha256=row['world_sha256']))
    for kind,pair in (('person',people),('room',rooms)):
     allowed=(frames[c-1][pair[0]]==frames[c-1][pair[1]]) if kind=='person' else not any(x in pair for x in frames[c-1])
     need(allowed==row[kind]['legal'],'Feasibility legal-transform inventory differs')
     if allowed:
      after=f.transform(frames,c,pair,kind);need(f.object_sha(after)==row[kind]['after_world_sha256'],'Feasibility transformed legacy hash differs')
      exclusions.add(after,dict(kind='feasibility_legal_transform',n=n,trial=trial,transform=kind,legacy_world_sha256=f.object_sha(after)));legal+=1
    seen+=1
  need(not stream.read().strip(),'Extra feasibility ledger rows')
 tests=f.self_test(recompute);stored=read(fp['tests']['file']);need(object_sha(tests)==object_sha(stored),'Frozen symbolic fixture replay differs')
 fixtures=[]
 for item in tests['witnesses'].values():fixtures.extend((item['before'],item['after']))
 fixtures.extend((tests['room_fixture']['before'],tests['room_fixture']['after']))
 witness=tuple(tuple(row) for row in tests['witnesses']['7']['before']);u,v=2,0
 fixtures.append(tuple(tuple(row[v] if i==u else row[u] if i==v else x for i,x in enumerate(row)) for row in witness))
 fixtures.append(tuple(tuple(1 if r==0 else 0 if r==1 else r for r in row) for row in witness))
 fixtures.append(f.world(random.Random(17),16))
 for index,frames in enumerate(fixtures):exclusions.add(frames,dict(kind='feasibility_fixture',index=index,legacy_world_sha256=f.object_sha(frames)))
 return dict(source_worlds=seen,legal_transforms=legal,fixture_worlds=len(fixtures),all_legacy_hashes_verified=True,fixture_replay_passed=True)


def legacy_qa_exposure(exclusions,bindings,started,stream):
 # Enumerate raw question inventories only, never arbitrary result JSON/tensors/images.
 required=[REPO/'data'/name for name in ('mmred_filtered','mmred_filtered_train','mmred_vfiltered')]
 need(all(path.is_dir() for path in required),'A HANDOFF raw-data inventory is missing')
 roots=sorted(path for path in (REPO/'data').iterdir() if path.is_dir() and path.name.startswith('mmred'))
 roots+=sorted(path for path in USER_DATA.iterdir() if path.is_dir() and not path.name.startswith('mmred_official')
  and path.name not in ('mmred_suffix_feasibility','mmred_fresh_evaluation'))
 bind(REPO/'HANDOFF.md',bindings);seen=set();counts=Counter();inventories=[]
 for root in roots:
  local=Counter();files=0
  for directory,dirs,names in os.walk(root,followlinks=False):
   deadline(started);dirs.sort()
   if 'qa.txt' not in names:continue
   path=(Path(directory)/'qa.txt').resolve();files+=1
   if path in seen:local['already_inspected_path']+=1;continue
   seen.add(path);descriptor=bind(path,bindings);local['files']+=1
   with path.open() as source:
    first=next((line.strip() for line in source if line.strip().startswith('{') and 'step_id' in line and 'rooms' in line),None)
   need(first is not None,'Unrecognized potentially compatible QA format: '+str(path))
   first_state=ast.literal_eval(first)
   try:from_states([dict(first_state,step_id=1)])
   except (ValueError,KeyError,TypeError):
    local['incompatible_first_frame']+=1;emit(stream,dict(**descriptor,compatible=False,reason='first_frame_not_complete_original_inventory'));continue
   states=[ast.literal_eval(line.strip()) for line in path.read_text().splitlines()
    if line.strip().startswith('{') and 'step_id' in line and 'rooms' in line]
   try:frames=from_states(states)
   except (ValueError,KeyError,TypeError):
    local['incompatible_sequence_or_movement']+=1;emit(stream,dict(**descriptor,compatible=False,reason='not_complete_original_sequence_or_movement'));continue
   digest=exclusions.add(frames,dict(kind='legacy_qa',**descriptor));local['compatible']+=1
   emit(stream,dict(**descriptor,compatible=True,n=len(frames),world_sha256=digest))
  counts.update(local);inventories.append(dict(root=str(root),discovered_qa_paths=files,counts=dict(local)))
 return dict(roots=inventories,counts=dict(counts),known_handoff_qa_inventories_inspected=True,
  scope='All qa.txt under enumerated roots, all recovered original rows, frozen feasibility sources/transforms/fixtures; official derivatives share recovered worlds',
  globally_exhaustive_exposure_claim=False,unrecognized_formats=0)


def margin(vector,direction='most'):
 values=sorted(vector.values(),reverse=direction=='most');return abs(values[0]-values[1])


def integer_answer(frames,qtype,q,*,direction=None,room=None,frame=None):
 if qtype in TASKS[:2]:
  counts=f.count_vector(frames,q,'person' if qtype==TASKS[0] else 'room')
  return f.winners(counts,direction)['unique'],counts
 if qtype=='steps_in_room':
  value=sum(state[q]==room for state in frames);return value,{f.ROOMS[room]:value}
 need(qtype=='char_at_frame' and 1<=frame<=len(frames),'Unknown original question task/frame')
 return f.ROOMS[frames[frame-1][q]],None


def question_text(qtype,q,*,direction=None,room=None,frame=None):
 if qtype in TASKS[:2]:return f.question(q,'person' if qtype==TASKS[0] else 'room',direction)
 if qtype=='steps_in_room':return f'How many steps did {f.CHARS[q]} spend in the {f.ROOMS[room]}?'
 return f'In which room was {f.CHARS[q]} at step {frame}?'


class Cohort:
 def __init__(self,data,exclusions,recompute,world_stream,question_stream,family_stream):
  self.data=data;self.exclusions=exclusions;self.recompute=recompute;self.ws=world_stream;self.qs=question_stream;self.fs=family_stream
  self.used=set();self.world_rows=[];self.questions=[];self.families=[];self.oracle_checks=0
 def world(self,frames,origin):
  digest=canonical_hash(frames);need(digest not in self.exclusions.worlds and digest not in self.used,'Retained world overlaps history or another world')
  row=dict(world_id='fresh_'+digest[:24],world_sha256=digest,n=len(frames),sequence=f.states(frames),
   canonical_characters=list(CHARS),canonical_rooms=list(ROOMS),canonical_positions=named_positions(frames),origin=origin,
   prefix_overlap=self.exclusions.overlap(frames))
  need(canonical_hash(from_states(json.loads(json.dumps(row['sequence']))))==digest,'World serialization changes named semantics')
  self.used.add(digest);self.world_rows.append(row);emit(self.ws,row);return row
 def question(self,world,qtype,q,*,direction=None,room=None,frame=None,cohort,extra=None):
  frames=from_states(world['sequence']);answer,counts=integer_answer(frames,qtype,q,direction=direction,room=room,frame=frame)
  need(answer is not None,'Only unique full-span aggregation questions may be retained')
  text=question_text(qtype,q,direction=direction,room=room,frame=frame)
  actual=self.recompute(qtype,text,world['sequence']);self.oracle_checks+=1
  need(type(actual) is type(answer) and actual==answer,'Independent original question/typed answer oracle differs')
  atype='person' if qtype==TASKS[0] else 'number' if qtype==TASKS[2] else 'room'
  row=dict(index=len(self.questions),qid=f'fresh_q_{len(self.questions):04d}',sid=f'fresh_q_{len(self.questions):04d}',
   world_id=world['world_id'],world_sha256=world['world_sha256'],n=world['n'],seq_len=world['n'],qtype=qtype,atype=atype,
   question=text,answer=answer,target_text=json.dumps(dict(answer=answer),separators=(',',':')),sequence=world['sequence'],
   cohort=cohort,queried_actor=f.CHARS[q],direction=direction,queried_room=None if room is None else f.ROOMS[room],
   queried_frame=frame,counts=counts,full_margin=None if qtype not in TASKS[:2] else margin(counts,direction),
   relevant_events=(counts[answer] if qtype in TASKS[:2] else answer if qtype==TASKS[2] else 1),**(extra or {}))
  self.questions.append(row);emit(self.qs,row);return row


def main_cohort(cohort,started,progress):
 results=[]
 for n in POLICY['main_lengths']:
  for k,qtype in enumerate(TASKS):
   for d in ((0,1) if k<2 else (2,)):
    quota=(100 if n==32 else 50) if k<2 else 100
    direction=('most' if d==0 else 'least') if k<2 else None
    ws=POLICY['main_world_seed_base']+10000*n+100*k+d;qs=POLICY['main_question_seed_base']+10000*n+100*k+d
    wr=random.Random(ws);qr=random.Random(qs);rejected=Counter();accepted=[]
    for attempt in range(POLICY['main_attempts_per_stratum']):
     if attempt%100==0:deadline(started)
     frames=f.world(wr,n);q=qr.randrange(5);room=qr.randrange(6) if k==2 else None;frame=qr.randrange(1,n+1) if k==3 else None
     progress['current_main']=dict(n=n,task_index=k,direction_index=d,completed_draws=attempt+1,world_seed=ws,question_seed=qs)
     digest=canonical_hash(frames)
     if digest in cohort.exclusions.worlds:rejected['historical_world']+=1;continue
     if digest in cohort.used:rejected['retained_world']+=1;continue
     answer,_=integer_answer(frames,qtype,q,direction=direction,room=room,frame=frame)
     if answer is None:rejected['full_span_tie']+=1;continue
     world=cohort.world(frames,dict(cohort='main',n=n,task_index=k,direction_index=d,attempt=attempt,world_seed=ws,question_seed=qs))
     row=cohort.question(world,qtype,q,direction=direction,room=room,frame=frame,cohort='main')
     accepted.append(dict(attempt=attempt,qid=row['qid'],world_sha256=digest))
     if len(accepted)==quota:break
    result=dict(n=n,qtype=qtype,direction=direction,quota=quota,attempts=attempt+1,accepted=accepted,rejected=dict(rejected),
     complete=len(accepted)==quota,world_seed=ws,question_seed=qs)
    results.append(result);progress['main_strata']=results
 return results


def match_key(n,kind,q,c,result):
 m=result['invariants']['changed_frames'];v=min(margin(counts) for counts in result['counts']['full'])
 return (n,kind,q,min(3,4*c//n),min(3,4*m//n),0 if v==1 else 1 if v<=3 else 2 if v<=7 else 3)


def diagnostic(cohort,recompute,started,progress,sampling):
 all_stats={};reserved=set();oracle=Counter()
 for n in (16,32):
  wr=random.Random(POLICY['diagnostic_world_seeds'][str(n)]);qr=random.Random(POLICY['diagnostic_choice_seeds'][str(n)])
  quota=POLICY['changed_quotas'][str(n)];changed={kind:[] for kind in ('person','room')};bank=defaultdict(deque);pending=defaultdict(deque)
  count=Counter();matched=Counter();last_attempt=-1
  def unavailable(candidate):return any(h in cohort.used or h in reserved or h in cohort.exclusions.worlds for h in candidate['hashes'])
  def finish(change,control):
   key=change['key'];kind=change['kind'];q=change['q'];family_id=f'fresh_family_{len(cohort.families):03d}'
   need(change['hashes'][0]!=control['hashes'][0] and len(set(change['hashes']+control['hashes']))==4,'Four distinct source/control worlds required')
   for h in change['hashes']:reserved.remove(h)
   members=[];questions=[]
   for status,candidate in (('changed',change),('unchanged',control)):
    pair_worlds=[]
    for member,frames in enumerate((candidate['before'],candidate['after'])):
     world=cohort.world(frames,dict(cohort='diagnostic',family_id=family_id,pair_status=status,member=member,n=n,kind=kind,source_attempt=candidate['attempt']))
     pair_worlds.append(world);members.append(world['world_sha256'])
     row=cohort.question(world,TASKS[0] if kind=='person' else TASKS[1],q,direction='most',cohort='diagnostic_aggregation',
      extra=dict(family_id=family_id,pair_status=status,member=member));questions.append(row['qid'])
    if status=='changed':
     atomic_frame=next(i+1 for i,(a,b) in enumerate(zip(candidate['before'],candidate['after'])) if a!=b) if kind=='person' else candidate['cut']
     atom=[cohort.question(world,'char_at_frame',q,frame=atomic_frame,cohort='diagnostic_atomic',
      extra=dict(family_id=family_id,pair_status=status,member=i)) for i,world in enumerate(pair_worlds)]
     need(atom[0]['answer']==atom[1]['answer'],'Diagnostic atomic answers are not invariant');questions.extend(row['qid'] for row in atom)
   def desc(candidate):return {k:v for k,v in candidate.items() if k not in ('before','after')}
   family=dict(family_id=family_id,n=n,kind=kind,qtype=TASKS[0] if kind=='person' else TASKS[1],queried_actor=f.CHARS[q],
    match_key=list(key),world_sha256=members,question_ids=questions,changed=desc(change),unchanged=desc(control),all_four_worlds_disjoint=True)
   family=json.loads(json.dumps(family));cohort.families.append(family);emit(cohort.fs,family);matched[kind]+=1
  def try_match(key):
   while pending[key] and bank[key]:
    while bank[key] and unavailable(bank[key][0]):bank[key].popleft();count['control_invalidated_by_reservation']+=1
    if not bank[key]:break
    finish(pending[key].popleft(),bank[key].popleft())
  for attempt in range(POLICY['diagnostic_attempts_per_length']):
   last_attempt=attempt
   if attempt%100==0:deadline(started)
   frames,c,q,people,rooms=proposal_draw(wr,qr,n);source_hash=canonical_hash(frames)
   progress['current_diagnostic']=dict(n=n,completed_draws=attempt+1,world_sha256=source_hash,cut=c,actor=q,person_pair=list(people),room_pair=list(rooms))
   for kind,pair in (('person',people),('room',rooms)):
    count[kind+'_proposals']+=1
    result,after=f.candidate(frames,c,q,pair,kind,recompute,oracle)
    if not result['legal']:count[kind+'_illegal_anchor']+=1;continue
    if not result['nonglobal']:count[kind+'_trivial_or_global']+=1;continue
    value=result['directions']['most'];strict=value['strong_third'] and value['unique_partial_all']
    is_control=value['unchanged']
    if not strict and not is_control:count[kind+'_predicate_rejected']+=1;continue
    candidate=dict(before=frames,after=after,hashes=[source_hash,canonical_hash(after)],key=match_key(n,kind,q,c,result),
     n=n,kind=kind,q=q,cut=c,pair=list(pair),attempt=attempt,result=result)
    emit(sampling,dict(n=n,attempt=attempt,kind=kind,category='changed' if strict else 'unchanged',
     world_sha256=candidate['hashes'],match_key=list(candidate['key'])))
    if unavailable(candidate):count[kind+'_world_overlap_or_reserved']+=1;continue
    if strict:
     if len(changed[kind])>=quota:count[kind+'_changed_quota_already_reserved']+=1;continue
     changed[kind].append(candidate);reserved.update(candidate['hashes']);pending[candidate['key']].append(candidate);try_match(candidate['key'])
    else:
     bank[candidate['key']].append(candidate);try_match(candidate['key'])
   progress['diagnostic']=dict(n=n,attempts=attempt+1,changed_reserved={k:len(v) for k,v in changed.items()},matched=dict(matched),counts=dict(count))
   if all(matched[kind]==quota for kind in ('person','room')):break
  all_stats[str(n)]=dict(attempts=last_attempt+1,quota_per_task=quota,changed_reserved={k:len(v) for k,v in changed.items()},
   matched={k:matched[k] for k in changed},unmatched=[{k:v for k,v in item.items() if k not in ('before','after')}
    for queue in pending.values() for item in queue],counts=dict(count),complete=all(matched[kind]==quota for kind in changed),
   banked_unused_controls=sum(len(queue) for queue in bank.values()))
 return all_stats,dict(oracle)


def fixtures(recompute,fp):
 old=read(fp['tests']['file']);frames=tuple(tuple(row) for row in old['witnesses']['7']['before'])
 states=f.states(frames);permuted=[dict(step_id=row['step_id'],rooms={room:list(reversed(people))
  for room,people in reversed(list(row['rooms'].items()))}) for row in states]
 need(from_states(permuted)==frames and canonical_hash(from_states(permuted))==canonical_hash(frames),'Canonical identity depends on dictionary/occupant order')
 need(legacy_recovery_hash(frames)!=f.object_sha(frames),'Fixture must detect the two distinct legacy array conventions')
 for qtype,args in (('char_at_frame',dict(frame=1)),('steps_in_room',dict(room=5)),('where_spend',dict(direction='most'))):
  answer,_=integer_answer(frames,qtype,1,**args);text=question_text(qtype,1,**args)
  need(recompute(qtype,text,states)==answer,'Original control/aggregation typed oracle fixture failed')
 need(f.winners(dict(A=0,B=0),'least')['unique'] is None and margin(dict(A=5,B=3))==2,
  'Full ties/margin fixture failed')
 need([0 if v==1 else 1 if v<=3 else 2 if v<=7 else 3 for v in (1,2,3,4,7,8)]==[0,1,1,2,2,3],
  'Fixed matching-margin boundaries differ')
 return dict(passed=True,canonical_person_room_order=True,occupant_dictionary_order_invariance=True,
  legacy_hash_conventions_distinguished=True,original_question_oracles=True,ties_and_match_bins=True,
  additional_generated_fixture_worlds=0)


def verify_serialized(cohort,data):
 worlds={};questions=[];frames_total=0
 with (data/'worlds.jsonl').open() as stream:
  for row in map(json.loads,stream):
   frames=from_states(row['sequence']);digest=canonical_hash(frames)
   need(digest==row['world_sha256'] and named_positions(frames)==row['canonical_positions']
    and digest not in worlds and digest not in cohort.exclusions.worlds,'Serialized world identity/exclusion differs')
   worlds[digest]=row;frames_total+=len(frames)
 with (data/'questions.jsonl').open() as stream:
  for row in map(json.loads,stream):
   need(row['index']==len(questions) and row['sequence']==worlds[row['world_sha256']]['sequence']
    and row['world_id']==worlds[row['world_sha256']]['world_id'],'Serialized question/world ownership differs')
   answer=cohort.recompute(row['qtype'],row['question'],row['sequence'])
   need(type(answer) is type(row['answer']) and answer==row['answer'] and json.loads(row['target_text'])==dict(answer=answer),
    'Serialized typed JSON/original oracle differs')
   questions.append(row)
 families=[json.loads(line) for line in (data/'families.jsonl').read_text().splitlines()]
 need(worlds=={row['world_sha256']:row for row in cohort.world_rows} and questions==cohort.questions and families==cohort.families,
  'On-disk cohort differs from the retained selection')
 known={row['qid']:row for row in questions};family_worlds=set()
 for family in families:
  need(len(family['world_sha256'])==len(set(family['world_sha256']))==4
   and not family_worlds.intersection(family['world_sha256']) and len(family['question_ids'])==6,'Complete disjoint six-question families required')
  family_worlds.update(family['world_sha256'])
  for status in ('changed','unchanged'):
   item=family[status];before,after=[from_states(worlds[digest]['sequence']) for digest in item['hashes']]
   actual,actual_after=f.candidate(before,item['cut'],item['q'],tuple(item['pair']),item['kind'],cohort.recompute,Counter())
   need(actual_after==after and json.loads(json.dumps(actual))==item['result'],
    'Serialized counterfactual counts, predicates, invariants or original oracles differ')
   need(list(match_key(family['n'],item['kind'],item['q'],item['cut'],actual))==item['key']==family['match_key'],
    'Serialized matching bins differ')
   value=actual['directions']['most']
   need(value['unique_full_both'] and item['result']['nonglobal'] and (value['strong_third'] and value['unique_partial_all']
    if status=='changed' else value['unchanged']),'Serialized fixed diagnostic predicate differs')
  atoms=[known[qid] for qid in family['question_ids'] if known[qid]['cohort']=='diagnostic_atomic']
  need(len(atoms)==2 and atoms[0]['answer']==atoms[1]['answer'],'Serialized atomic invariance changed')
 return dict(passed=True,worlds=len(worlds),questions=len(questions),families=len(families),world_frame_occurrences=frames_total,
  full_worlds_disjoint_from_inventory=True,independent_original_oracle_checks=len(questions),serialization_reparsed=True)


def population_indices(questions):
 result={key:[] for key in ('all','main','diagnostic','diagnostic_aggregation','diagnostic_atomic','main_N32_primary','main_N32_controls')}
 result['main_by_length']={str(n):[] for n in (8,16,32)}
 for row in questions:
  index=row['index'];result['all'].append(index)
  if row['cohort']=='main':
   result['main'].append(index);result['main_by_length'][str(row['n'])].append(index)
   if row['n']==32:result['main_N32_primary' if row['qtype'] in TASKS[:2] else 'main_N32_controls'].append(index)
  else:
   result['diagnostic'].append(index)
   result['diagnostic_atomic' if row['cohort']=='diagnostic_atomic' else 'diagnostic_aggregation'].append(index)
 return result


def histograms(questions):
 groups=defaultdict(lambda:defaultdict(Counter))
 for row in questions:
  key=f"{row['cohort']}/N{row['n']}/{row['qtype']}/{row['direction']}"
  for name in ('answer','queried_actor','queried_room','queried_frame','direction','full_margin','relevant_events'):
   groups[key][name][str(row[name])]+=1
 return {key:{name:dict(counter) for name,counter in fields.items()} for key,fields in groups.items()}


def run(out,data,started,bindings,progress):
 from gnnformer.mmred_hf import recompute_answer
 need(not any(name=='torch' or name.startswith('torch.') or name=='transformers' or name.startswith('transformers.') for name in sys.modules),
  'Data-only preparation must not import tensor/model libraries')
 rp,fp,identities,official=parents(bindings)
 need({path.name for path in (USER_DATA/'mmred_suffix_feasibility').iterdir() if path.is_dir()}=={'audit_444121'},
  'Additional feasibility attempts require an explicit exposure inventory')
 tests=fixtures(recompute_answer,fp);save(out/'fixtures.json',tests)
 with ExitStack() as stack:
  streams={name:stack.enter_context((data/(name+'.jsonl')).open('x')) for name in
   ('exclusion_ledger','legacy_inventory','worlds','questions','families','diagnostic_candidate_ledger')}
  exclusion=Exclusions(streams['exclusion_ledger'])
  exposure=dict(recovery=recovery_exposure(rp,identities,exclusion,bindings,started),
   feasibility=feasibility_exposure(fp,exclusion,recompute_answer,started))
  exposure['legacy']=legacy_qa_exposure(exclusion,bindings,started,streams['legacy_inventory'])
  exposure.update(unique_excluded_worlds=len(exclusion.worlds),event_counts=dict(exclusion.events),
   canonical_identity=dict(characters=list(CHARS),rooms=list(ROOMS),ordered_steps=True,named_room_values=True),
   known_inventory_complete=True,global_exhaustiveness_claim=False)
  save(out/'exposure.json',exposure);progress['exposure_complete']=True
  cohort=Cohort(data,exclusion,recompute_answer,streams['worlds'],streams['questions'],streams['families'])
  main=main_cohort(cohort,started,progress)
  diag,oracle=diagnostic(cohort,recompute_answer,started,progress,streams['diagnostic_candidate_ledger'])
  for stream in streams.values():stream.flush()
  verification=verify_serialized(cohort,data)
 complete=all(row['complete'] for row in main) and all(row['complete'] for row in diag.values())
 populations=population_indices(cohort.questions)
 save(out/'populations.json',populations)
 if complete:
  need((verification['worlds'],verification['questions'],verification['families'],verification['world_frame_occurrences'])==(2040,2360,160,47232),
   'Complete fixed cohort totals differ')
  main_rows=[row for row in cohort.questions if row['cohort']=='main']
  need(len(main_rows)==1400 and len([row for row in main_rows if row['n']==32 and row['qtype'] in TASKS[:2]])==400,
   'Main/primary populations differ')
  need({key:len(populations[key]) for key in ('all','main','diagnostic','diagnostic_aggregation','diagnostic_atomic','main_N32_primary','main_N32_controls')}
   ==dict(all=2360,main=1400,diagnostic=960,diagnostic_aggregation=640,diagnostic_atomic=320,main_N32_primary=400,main_N32_controls=200)
   and {key:len(value) for key,value in populations['main_by_length'].items()}=={'8':400,'16':400,'32':600},
   'Complete explicit evaluation populations differ')
 deadline(started)
 save(out/'histograms.json',histograms(cohort.questions));save(out/'selection.json',dict(main=main,diagnostic=diag))
 renderer_rows=[{key:row[key] for key in ('qid','seq_len','qtype','atype','question','answer','sequence')} for row in cohort.questions]
 save(data/'renderer_rows.json',renderer_rows)
 need(read(data/'renderer_rows.json')==renderer_rows,'Original-renderer row serialization differs')
 save(out/'original_protocol.json',dict(system_prompt=official['system_prompt'],images_before_original_question=True,
  original_templates=True,canonical_typed_json=True,rendering_not_executed=True))
 return dict(protocol=PROTOCOL,passed=True,completed=True,cohort_complete=complete,fresh_confirmation_eligible=complete,
  exposure_inventory_complete=True,global_exhaustiveness_claim=False,policy=POLICY,verification=verification,
  exposure_file=str(out/'exposure.json'),exposure_sha256=sha(out/'exposure.json'),
  populations_file=str(out/'populations.json'),populations_sha256=sha(out/'populations.json'),
  rows_file=str(data/'renderer_rows.json'),rows_sha256=sha(data/'renderer_rows.json'),
  worlds_file=str(data/'worlds.jsonl'),worlds_sha256=sha(data/'worlds.jsonl'),questions_file=str(data/'questions.jsonl'),
  questions_sha256=sha(data/'questions.jsonl'),families_file=str(data/'families.jsonl'),families_sha256=sha(data/'families.jsonl'),
  exclusion_ledger_file=str(data/'exclusion_ledger.jsonl'),exclusion_ledger_sha256=sha(data/'exclusion_ledger.jsonl'),
  selection_file=str(out/'selection.json'),selection_sha256=sha(out/'selection.json'),histograms_file=str(out/'histograms.json'),
  histograms_sha256=sha(out/'histograms.json'),original_protocol_file=str(out/'original_protocol.json'),
  original_protocol_sha256=sha(out/'original_protocol.json'),oracle_checks=dict(retained=cohort.oracle_checks,diagnostic_candidates=oracle),
  no_training_or_inference_release=True,no_model_output_access=True,no_images_models_tensors=True,
  shortage_is_not_relaxed=True,original_recovery=dict(file=str(RECOVERY),sha256=RECOVERY_SHA),
  original_feasibility=dict(file=str(FEASIBILITY),sha256=FEASIBILITY_SHA))


def verify_stage(summary_path):
 path=Path(summary_path).resolve();summary=read(path);own,inherited,_=source_maps()
 need(summary['passed'] is summary['completed'] is summary['cohort_complete'] is summary['fresh_confirmation_eligible'] is True
  and summary['protocol']==PROTOCOL and summary['source_sha256']==own and summary['inherited_source_sha256']==inherited
  and not (path.parent/'failure.json').exists(),'Complete, verified fresh cohort required')
 need(sha(summary['plan_file'])==summary['plan_sha256'],'Cohort plan changed');plan=read(summary['plan_file'])
 need(plan['policy']==POLICY and plan['source_sha256']==own and plan['inherited_source_sha256']==inherited,'Cohort source/policy differs')
 need(plan['protocol']==PROTOCOL and all(plan[key] is summary[key] is True for key in
  ('passed','completed','cohort_complete','fresh_confirmation_eligible'))
  and 0<=plan['elapsed_seconds']<=summary['elapsed_seconds']<=600,'Cohort completion or timing differs')
 need(sha(summary['artifacts_file'])==summary['artifacts_sha256'],'Cohort artifact manifest changed')
 artifacts=read(summary['artifacts_file'])
 need(artifacts.get(summary['plan_file'])==summary['plan_sha256'],'Plan absent from artifact manifest')
 for file,digest in artifacts.items():need(sha(file)==digest,'Manifest-bound cohort artifact changed')
 need(sha(plan['input_bindings_file'])==plan['input_bindings_sha256']
  and artifacts.get(plan['input_bindings_file'])==plan['input_bindings_sha256'],'Input binding manifest differs')
 bindings=read(plan['input_bindings_file'])
 need(bindings.get(str(RECOVERY.resolve()))==RECOVERY_SHA and bindings.get(str(FEASIBILITY.resolve()))==FEASIBILITY_SHA,
  'Original recovery/feasibility input owners differ')
 for field in ('rows','worlds','questions','families','exclusion_ledger','selection','histograms','original_protocol','exposure','populations'):
  need(sha(plan[field+'_file'])==plan[field+'_sha256'],'Cohort published artifact changed')
  need(artifacts.get(plan[field+'_file'])==plan[field+'_sha256'],'Published artifact absent from complete manifest')
 questions=[json.loads(line) for line in Path(plan['questions_file']).read_text().splitlines()]
 need(read(plan['populations_file'])==population_indices(questions),'Question population indices differ')
 need(read(plan['rows_file'])==[{key:row[key] for key in ('qid','seq_len','qtype','atype','question','answer','sequence')}
  for row in questions],'Renderer/question projection differs')
 need(plan['verification']['passed'] is True and tuple(plan['verification'][key] for key in
  ('worlds','questions','families','world_frame_occurrences'))==(2040,2360,160,47232),'Complete serialized cohort verification differs')
 for name,digest in {**own,**inherited}.items():need(sha(path.parent/'source'/name.replace('/','_'))==digest,'Archived cohort source changed')
 return plan


def main():
 need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu'
  and int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==4 and not os.environ.get('SLURM_JOB_GPUS'),'Four-core CPU-only Slurm required')
 started=time.perf_counter();out=OUT/f'prepare_{os.environ["SLURM_JOB_ID"]}';data=DATA/out.name
 need(not DATA.exists() or not any(DATA.iterdir()),'Earlier fresh-cohort attempts require explicit exposure accounting; no automatic retry')
 out.mkdir(parents=True,exist_ok=False);data.mkdir(parents=True,exist_ok=False)
 own,inherited,bindings=source_maps();(out/'source').mkdir();progress={}
 for name,digest in {**own,**inherited}.items():
  file=out/'source'/name.replace('/','_');file.write_bytes((REPO/name).read_bytes());need(sha(file)==digest,'Cohort source archive differs')
 save(out/'request.json',dict(protocol=PROTOCOL,policy=POLICY,source_sha256=own,inherited_source_sha256=inherited))
 try:
  result=run(out,data,started,bindings,progress)
  save(out/'input_bindings.json',bindings)
  result.update(source_sha256=own,inherited_source_sha256=inherited,input_bindings_file=str(out/'input_bindings.json'),
   input_bindings_sha256=sha(out/'input_bindings.json'),elapsed_seconds=time.perf_counter()-started,
   environment=dict(python=sys.version,pyarrow=importlib.metadata.version('pyarrow'),pillow=importlib.metadata.version('Pillow')))
  save(out/'plan.json',result)
  (out/'REPORT.md').write_text('Data-only cohort preparation completed.\n\n'+
   ('All fixed quotas were filled and verified.' if result['cohort_complete'] else 'Fixed quota shortfall retained; this is not an eligible confirmation cohort.')+
   ' The plan records exact inventory scope, canonical exclusions, oracle checks and histograms. No model, image, fit or inference was run or released.\n')
  artifacts={str(file):sha(file) for root in (out,data) for file in root.iterdir() if file.is_file()};save(out/'artifacts.json',artifacts)
  need(source_maps()[:2]==(own,inherited) and time.perf_counter()-started<600,'Source or CPU allocation cap changed')
  save(out/'summary.json',dict(protocol=PROTOCOL,passed=True,completed=True,cohort_complete=result['cohort_complete'],
   fresh_confirmation_eligible=result['fresh_confirmation_eligible'],source_sha256=own,inherited_source_sha256=inherited,
   plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),artifacts_file=str(out/'artifacts.json'),
   artifacts_sha256=sha(out/'artifacts.json'),elapsed_seconds=time.perf_counter()-started,no_training_or_inference_release=True))
 except BaseException as exc:
  save(out/'failure_input_bindings.json',bindings)
  save(out/'failure.json',dict(protocol=PROTOCOL,passed=False,type=type(exc).__name__,message=str(exc),progress=progress,
   elapsed_seconds=time.perf_counter()-started,source_sha256=own,inherited_source_sha256=inherited,
   input_bindings_file=str(out/'failure_input_bindings.json'),input_bindings_sha256=sha(out/'failure_input_bindings.json'),
   partial_artifacts={str(file):sha(file) for root in (out,data) for file in root.iterdir() if file.is_file()},
   partial_outputs_retained=True,cohort_complete=False,fresh_confirmation_eligible=False,no_automatic_retry=True));raise


if __name__=='__main__':main()

"""Bounded fresh-world suffix feasibility; CPU only, no model or benchmark cohort."""
from __future__ import annotations
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
PROTOCOL='mmred_suffix_feasibility'
PROPOSAL='docs/paper/MMRED_SUFFIX_FEASIBILITY_PROTOCOL.md'
PROPOSAL_SHA='2d634136b0e51430d602f5ddda1a0fc746905caa61c807762894b30a0fb8863b'
OWN=('scripts/audit_mmred_suffix_feasibility.py',PROPOSAL,'slurm/mmred_suffix_feasibility.sbatch')
INHERITED={'gnnformer/mmred_hf.py':'3fb97185198d1d9e90eaeef2414e01b07d8f84271598601f0c372bb89e37d492',
 'gnnformer/__init__.py':'36c7a94bb10d9c59b25838220e35e9c696b0b9fd4f3cdc7e9352eeda52f10a75',
 'docs/paper/MMRED_FRESH_SUFFIX_COUNTERFACTUAL_DESIGN.md':'182a8ea67df5670ff6472d0723d22f03c0232313b2d1f8e90775ce3f605db42e'}
UPSTREAM=Path('/mnt/data/gabriele/gnn_transformer/mmred_official/recovery_443939/upstream')
UPSTREAM_SHAS={'mmred/qgen/utils.py':'eab9ee7a900328b6df8166f81539842438fa0a0ab78858b79f241994908a8f27',
 'mmred/qgen/questions.py':'e33129c9d21c4522d4633005dc2ce7a83a6979698d3bf0d1b142d37483536875',
 'mmred/const.py':'60c9923501568cfab7e9d18a27fa0c04d6d2cd0109aece2867c024c59635106b',
 'mmred/data_model.py':'b3392d496c95e7f71c7bea98d3289acb2a01beccf0419a47a3ce896565e1ff3d'}
CHARS=('Sandra','Mary','John','Daniel','Michael')
ROOMS=('Kitchen','Bathroom','Garden','Office','Bedroom','Hallway')
DESTINATIONS={i:tuple(ROOMS.index(r) for r in sorted(set(ROOMS)-{ROOMS[i]})) for i in range(6)}
POLICY=dict(protocol=PROTOCOL,lengths=[16,32],attempts_per_length=20000,
    world_seeds={'16':2026091216,'32':2026091232},choice_seeds={'16':2026092216,'32':2026092232},
    cpu_seconds=300,sampling_deadline_seconds=270,cpu_cores=4,memory_gib=16,
    directions=['most','least'],candidate_families=['person','room'],one_candidate_per_family_per_world=True,
    maximum_saved_examples=48,model_calls=0,images=0,tensors=0,gpu_calls=0,benchmark_cohort=False,
    recovered_world_overlap_check_deferred=True,no_current_model_output_access=True,no_followup_release=True)
OUT=REPO/'outputs/native_aggregation_vlm/mmred_suffix_feasibility'
DATA=Path('/mnt/data/gabriele/gnn_transformer/mmred_suffix_feasibility')
CATEGORIES=('legal_nonglobal','unchanged_unique','changed_unique','equal_hist_changed','strong_third','equal_hist_strong_third')
COUNT_KEYS=('legal','nontrivial','nonglobal','equal_hist_nonglobal','two_way_sign_flip')
DIRECTION_KEYS=('unique_full_both','changed_unique','unchanged_unique','changed_nonglobal','unchanged_nonglobal',
    'equal_hist_changed','strong_pair','strong_pair_unique_partial','strong_third','strong_third_unique_partial',
    'equal_hist_strong_third','prefix_pair_correct','suffix_pair_correct')


def need(value,message):
    if not value:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def object_sha(value):return hashlib.sha256(json.dumps(value,separators=(',',':'),sort_keys=True).encode()).hexdigest()


def save(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,sort_keys=True,allow_nan=False,indent=2)


def source_maps():
    need(sha(REPO/PROPOSAL)==PROPOSAL_SHA,'Prospective feasibility protocol changed')
    own={name:sha(REPO/name) for name in OWN}
    need(all(sha(REPO/name)==digest for name,digest in INHERITED.items()),'Small pinned repository dependency changed')
    bindings={str(UPSTREAM/name):digest for name,digest in UPSTREAM_SHAS.items()}
    need(all(sha(path)==digest for path,digest in bindings.items()),'Pinned upstream law/oracle source changed')
    return own,dict(INHERITED),bindings


def world(rng,n):
    frames=[tuple(rng.randrange(6) for _ in CHARS)]
    for _ in range(1,n):
        person=rng.randrange(5);frame=list(frames[-1]);frame[person]=rng.choice(DESTINATIONS[frame[person]]);frames.append(tuple(frame))
    return tuple(frames)


def validate(frames):
    need(len(frames)>0 and all(len(row)==5 and all(type(x) is int and 0<=x<6 for x in row) for row in frames),'Exact five-person/six-room state required')
    movers=[]
    for before,after in zip(frames,frames[1:]):
        moved=[p for p in range(5) if before[p]!=after[p]];need(len(moved)==1,'Every transition must move exactly one person');movers.append(moved[0])
    return movers


def transform(frames,c,pair,kind):
    need(1<=c<len(frames) and len(pair)==2 and pair[0]!=pair[1],'Nonempty fixed prefix/suffix and distinct pair required')
    a,b=pair
    if kind=='person':need(frames[c-1][a]==frames[c-1][b],'Person swap anchor must be co-located')
    else:need(kind=='room' and a not in frames[c-1] and b not in frames[c-1],'Room swap anchor must have both rooms empty')
    changed=[]
    for row in frames[c:]:
        if kind=='person':r=list(row);r[a],r[b]=r[b],r[a];changed.append(tuple(r))
        else:changed.append(tuple(b if x==a else a if x==b else x for x in row))
    return frames[:c]+tuple(changed)


def histograms(frames):return tuple(tuple(sum(row[p]==r for row in frames) for r in range(6)) for p in range(5))


def count_vector(frames,q,kind):
    if kind=='person':return {CHARS[p]:sum(row[p]==row[q] for row in frames) for p in range(5) if p!=q}
    return {ROOMS[r]:sum(row[q]==r for row in frames) for r in range(6)}


def winners(counts,direction):
    target=(max if direction=='most' else min)(counts.values());found=sorted(k for k,v in counts.items() if v==target)
    return dict(winners=found,unique=found[0] if len(found)==1 else None,tied=len(found)!=1)


def global_equivalent(before,after,q,kind):
    if kind=='person':
        if tuple(r[q] for r in before)!=tuple(r[q] for r in after):return False
        return Counter(tuple(r[p] for r in before) for p in range(5) if p!=q)==Counter(tuple(r[p] for r in after) for p in range(5) if p!=q)
    mapping={}
    for left,right in zip(before,after):
        for a,b in zip(left,right):
            if a in mapping and mapping[a]!=b:return False
            mapping[a]=b
    return len(set(mapping.values()))==len(mapping)


def states(frames):
    return [dict(step_id=i+1,rooms={room:[CHARS[p] for p,x in enumerate(row) if x==r] for r,room in enumerate(ROOMS)}) for i,row in enumerate(frames)]


def question(q,kind,direction):
    word='most' if direction=='most' else 'least amount of'
    return (f'With whom did {CHARS[q]} spend the {word} time together in the same room?' if kind=='person'
        else f'In which room did {CHARS[q]} spend the {word} time?')


def invariant_check(before,after,c,q,pair,kind):
    movers=validate(before);new_movers=validate(after)
    need(before[:c]==after[:c] and len(before)==len(after) and transform(after,c,pair,kind)==before,'Prefix/Step range/involution changed')
    if kind=='person':
        need(q not in pair and all(left[q]==right[q] and Counter(left)==Counter(right) for left,right in zip(before,after)),
             'Person swap changed queried trajectory or per-frame room occupancy')
        need(all(left[p]==right[p] for left,right in zip(before,after) for p in range(5) if p not in pair), 'Person swap changed another trajectory')
        need(all(sum(x==left[q] for x in left)==sum(x==right[q] for x in right) for left,right in zip(before,after)), 'Per-frame queried companion load changed')
        equal=histograms(before)==histograms(after);suffix=histograms(before[c:])
        need(equal==(suffix[pair[0]]==suffix[pair[1]]),'Equal whole person-room marginals disagree with suffix histogram condition')
        need(count_vector(before,q,'room')==count_vector(after,q,'room'),'Invariant actor-room count vector changed')
    else:
        need(movers==new_movers and all((left[p]==left[r])==(right[p]==right[r]) for left,right in zip(before,after)
             for p in range(5) for r in range(5)), 'Room swap changed moving person or co-location relations')
        need(count_vector(before,q,'person')==count_vector(after,q,'person'),'Invariant partner-count vector changed')
        equal=histograms(before)==histograms(after)
    return dict(passed=True,equal_person_room_histograms=equal,changed_frames=sum(a!=b for a,b in zip(before,after)))


def compare_counts(full,prefix,suffix,labels):
    need(all(full[i][k]==prefix[i][k]+suffix[i][k] for i in (0,1) for k in full[i]) and prefix[0]==prefix[1], 'Exact prefix/suffix decomposition failed')
    a,b=labels;pd=prefix[0][a]-prefix[0][b];sd=suffix[0][a]-suffix[0][b]
    old=full[0][a]-full[0][b];new=full[1][a]-full[1][b]
    need(old==pd+sd and new==pd-sd and suffix[1][a]==suffix[0][b] and suffix[1][b]==suffix[0][a], 'Swap count algebra failed')
    need(all(full[0][k]==full[1][k] for k in full[0] if k not in labels),'Unchanged competitor total changed')
    flip=old*new<0
    if flip:need(abs(sd)>abs(pd) and old*sd>0 and new*(-sd)>0,'Two-way flip must be suffix-determined')
    outcomes={}
    for direction in POLICY['directions']:
        parts={name:[winners(vector,direction) for vector in vectors] for name,vectors in (('full',full),('prefix',prefix),('suffix',suffix))}
        gold=[r['unique'] for r in parts['full']];unique=all(v is not None for v in gold)
        pc=[unique and parts['prefix'][i]['unique']==gold[i] for i in (0,1)]
        sc=[unique and parts['suffix'][i]['unique']==gold[i] for i in (0,1)]
        changed=unique and gold[0]!=gold[1];strong=changed and not all(pc) and not all(sc)
        third=strong and any(g not in labels for g in gold)
        outcomes[direction]=dict(**parts,unique_full_both=unique,changed=changed,unchanged=unique and not changed,
            prefix_correct=pc,suffix_correct=sc,prefix_pair_correct=all(pc),suffix_pair_correct=all(sc),
            unique_partial_all=all(r['unique'] is not None for key in ('prefix','suffix') for r in parts[key]),
            strong_pair=strong,strong_third=third)
    return dict(prefix_difference=pd,suffix_difference=sd,old_difference=old,new_difference=new,two_way_sign_flip=flip),outcomes


def candidate(frames,c,q,pair,kind,recompute,oracle_counts):
    legal=(frames[c-1][pair[0]]==frames[c-1][pair[1]]) if kind=='person' else not any(r in pair for r in frames[c-1])
    if not legal:return dict(legal=False,reason='anchor_not_fixed'),None
    after=transform(frames,c,pair,kind);invariants=invariant_check(frames,after,c,q,pair,kind)
    full=[count_vector(v,q,kind) for v in (frames,after)]
    prefix=[count_vector(v[:c],q,kind) for v in (frames,after)];suffix=[count_vector(v[c:],q,kind) for v in (frames,after)]
    labels=tuple((CHARS if kind=='person' else ROOMS)[i] for i in pair)
    algebra,outcomes=compare_counts(full,prefix,suffix,labels)
    before_states=states(frames);after_states=states(after)
    need([s['step_id'] for s in before_states]==[s['step_id'] for s in after_states]==list(range(1,len(frames)+1)), 'Serialized Step labels changed')
    for direction,value in outcomes.items():
        qtype='spend_together' if kind=='person' else 'where_spend';text=question(q,kind,direction)
        for i,st in enumerate((before_states,after_states)):
            actual=recompute(qtype,text,st);oracle_counts['full_answer_checks']+=1
            need(actual==value['full'][i]['unique'],'Independent full-span original oracle differs, including tie behavior')
    control_kind='room' if kind=='person' else 'person';control_type='where_spend' if kind=='person' else 'spend_together'
    for direction in POLICY['directions']:
        expected=winners(count_vector(frames,q,control_kind),direction)['unique']
        for st in (before_states,after_states):
            oracle_counts['invariant_control_checks']+=1
            need(recompute(control_type,question(q,control_kind,direction),st)==expected,'Invariant cross-task oracle differs')
    nontrivial=frames!=after;global_swap=global_equivalent(frames,after,q,kind)
    return dict(legal=True,nontrivial=nontrivial,nonglobal=nontrivial and not global_swap,global_permutation_equivalent=global_swap,
        after_world_sha256=object_sha(after),invariants=invariants,counts=dict(full=full,prefix=prefix,suffix=suffix),
        algebra=algebra,directions=outcomes),after


def self_test(recompute):
    q,u,v=1,2,0;fixtures={};oracle=Counter();rejected=0
    witness=((0,1,1,2,3),(0,1,0,2,3),(0,1,1,2,3),(0,1,0,2,3),(0,0,0,2,3),(1,0,0,2,3),(0,0,0,2,3))
    for n in (7,16):
        rows=list(witness)
        while len(rows)<n:
            row=list(rows[-1]);row[3]=5 if row[3]==2 else 2;rows.append(tuple(row))
        frames=tuple(rows);result,after=candidate(frames,2,q,(u,v),'person',recompute,oracle)
        extra=n-7;old=result['counts']['full'][0];new=result['counts']['full'][1]
        need(result['nonglobal'] and result['invariants']['equal_person_room_histograms']
             and [old[CHARS[u]],old[CHARS[v]],new[CHARS[u]],new[CHARS[v]]]==[5+extra,2+extra,3+extra,4+extra]
             and result['directions']['most']['full'][0]['unique']==CHARS[u]
             and result['directions']['most']['full'][1]['unique']==CHARS[v], 'Symbolic witness or N16 extension failed')
        fixtures[str(n)]=dict(before=frames,after=after,result=result)
    for function in (lambda:validate((witness[0],witness[0])),lambda:transform(witness,1,(u,v),'person'),
                     lambda:transform(witness,2,(0,1),'room')):
        try:function()
        except ValueError:rejected+=1
    need(rejected==3,'Illegal transition/cut anchor fixture failed')
    permuted=[]
    for row in witness:
        row=list(row);row[u],row[v]=row[v],row[u];permuted.append(tuple(row))
    need(global_equivalent(witness,tuple(permuted),q,'person'),'Global fixed non-query permutation was missed')
    room_permuted=tuple(tuple(1 if r==0 else 0 if r==1 else r for r in row) for row in witness)
    need(global_equivalent(witness,room_permuted,q,'room'),'Global fixed room permutation was missed')
    room_world=((2,0,1,2,3),(2,2,1,2,3),(2,2,2,2,3),(2,0,2,2,3),(2,1,2,2,3),(2,0,2,2,3))
    room_result,room_after=candidate(room_world,3,q,(0,1),'room',recompute,oracle)
    need(room_result['nonglobal'] and room_result['invariants']['passed'],'Legal non-global room fixture failed')
    need(winners({'A':1,'B':1},'most')['unique'] is None and winners({'A':0,'B':0},'least')['tied'],'Unique/tied oracle fixture failed')
    prefix=[dict(A=4,B=0,C=5),dict(A=4,B=0,C=5)];suffix=[dict(A=4,B=2,C=2),dict(A=2,B=4,C=2)]
    full=[{k:prefix[i][k]+suffix[i][k] for k in prefix[i]} for i in (0,1)]
    algebra,third=compare_counts(full,prefix,suffix,('A','B'))
    need(third['most']['strong_third'] and third['most']['unique_partial_all']
         and third['most']['prefix_correct']==[False,True] and third['most']['suffix_correct']==[True,False],
         'Abstract third-competitor and paired-baseline fixture failed')
    need(world(random.Random(17),16)==world(random.Random(17),16),'Fresh generator replay fixture failed')
    return dict(passed=True,witnesses=fixtures,room_fixture=dict(before=room_world,after=room_after,result=room_result),
        abstract_third_competitor=dict(counts=full,prefix=prefix,suffix=suffix,result=third,temporal_feasibility_not_claimed=True),
        rejected_invalid_fixtures=rejected,oracle_checks=dict(oracle),sampled_trial_count=0)


def empty_stats():
    return dict(attempts=0,counts=Counter({k:0 for k in COUNT_KEYS}),
        directions={d:Counter({k:0 for k in DIRECTION_KEYS}) for d in POLICY['directions']})


def collect(stats,result):
    stats['attempts']+=1
    if not result['legal']:return
    count=stats['counts'];count['legal']+=1
    count['nontrivial']+=int(result['nontrivial']);count['nonglobal']+=int(result['nonglobal'])
    count['equal_hist_nonglobal']+=int(result['nonglobal'] and result['invariants']['equal_person_room_histograms'])
    count['two_way_sign_flip']+=int(result['algebra']['two_way_sign_flip'])
    for direction,value in result['directions'].items():
        dc=stats['directions'][direction];dc['unique_full_both']+=int(value['unique_full_both'])
        dc['changed_unique']+=int(value['changed']);dc['unchanged_unique']+=int(value['unchanged'])
        if not result['nonglobal']:continue
        dc['changed_nonglobal']+=int(value['changed']);dc['unchanged_nonglobal']+=int(value['unchanged'])
        if not value['changed']:continue
        equal=result['invariants']['equal_person_room_histograms'];dc['equal_hist_changed']+=int(equal)
        dc['strong_pair']+=int(value['strong_pair']);dc['strong_pair_unique_partial']+=int(value['strong_pair'] and value['unique_partial_all'])
        dc['strong_third']+=int(value['strong_third']);dc['strong_third_unique_partial']+=int(value['strong_third'] and value['unique_partial_all'])
        dc['equal_hist_strong_third']+=int(equal and value['strong_third'])
        dc['prefix_pair_correct']+=int(value['prefix_pair_correct']);dc['suffix_pair_correct']+=int(value['suffix_pair_correct'])


def ratio(n,d):return dict(numerator=n,denominator=d,rate=n/d if d else None)


def report_stats(stats):
    a=stats['attempts'];c=dict(stats['counts']);directions={}
    for direction,values in stats['directions'].items():
        d=dict(values);conditional={
            'unique_full_given_legal':ratio(d['unique_full_both'],c['legal']),
            'changed_given_unique':ratio(d['changed_unique'],d['unique_full_both']),
            'changed_given_nonglobal':ratio(d['changed_nonglobal'],c['nonglobal']),
            'unchanged_given_nonglobal':ratio(d['unchanged_nonglobal'],c['nonglobal'])}
        for key in ('equal_hist_changed','strong_pair','strong_pair_unique_partial','strong_third','strong_third_unique_partial',
                    'prefix_pair_correct','suffix_pair_correct'):
            conditional[key+'_given_changed_nonglobal']=ratio(d[key],d['changed_nonglobal'])
        conditional['strong_third_given_equal_hist_changed']=ratio(d['equal_hist_strong_third'],d['equal_hist_changed'])
        directions[direction]=dict(counts=d,per_attempt={k:ratio(v,a) for k,v in d.items()},conditional=conditional)
    return dict(attempts=a,counts=c,per_attempt={k:ratio(v,a) for k,v in c.items()},directions=directions,
        conditional=dict(nontrivial_given_legal=ratio(c['nontrivial'],c['legal']),nonglobal_given_nontrivial=ratio(c['nonglobal'],c['nontrivial']),
            equal_hist_given_nonglobal=ratio(c['equal_hist_nonglobal'],c['nonglobal'])))


def evidence_categories(result,direction):
    if not result['legal'] or not result['nonglobal']:return []
    value=result['directions'][direction];equal=result['invariants']['equal_person_room_histograms'];names=['legal_nonglobal']
    if value['unchanged']:names.append('unchanged_unique')
    if value['changed']:names.append('changed_unique')
    if value['changed'] and equal:names.append('equal_hist_changed')
    if value['strong_third']:names.append('strong_third')
    if value['strong_third'] and equal:names.append('equal_hist_strong_third')
    return names


def run(out,data,started,progress):
    from gnnformer.mmred_hf import recompute_answer
    need(not any(name=='torch' or name.startswith('torch.') or name=='transformers' or name.startswith('transformers.')
                 for name in sys.modules),'Feasibility must not load tensor/model libraries')
    tests=self_test(recompute_answer);fixture_file=data/'fixtures.json';save(fixture_file,tests)
    test_descriptor=dict(file=str(fixture_file),sha256=sha(fixture_file),passed=tests['passed'],oracle_checks=tests['oracle_checks'])
    population={};seen_worlds={};seen_counterfactuals={};examples={};oracle=Counter();ledger_path=data/'trials.jsonl'
    ledger_rows=0
    with ledger_path.open('x') as ledger:
        for n in POLICY['lengths']:
            source_rng=random.Random(POLICY['world_seeds'][str(n)]);choice_rng=random.Random(POLICY['choice_seeds'][str(n)])
            stats={kind:empty_stats() for kind in POLICY['candidate_families']};duplicates=Counter();seen_worlds[n]=set();seen_counterfactuals[n]=set()
            for trial in range(POLICY['attempts_per_length']):
                need(time.perf_counter()-started<POLICY['sampling_deadline_seconds'],'Sampling deadline reached; reserve publication time')
                frames=world(source_rng,n);validate(frames);c=choice_rng.randrange(1,n);q=choice_rng.randrange(5)
                people=tuple(sorted(choice_rng.sample([p for p in range(5) if p!=q],2)));rooms=tuple(sorted(choice_rng.sample(range(6),2)))
                identity=object_sha(frames);duplicates['source_repeat_within_run']+=int(identity in seen_worlds[n]);seen_worlds[n].add(identity)
                record=dict(n=n,trial=trial,world_sha256=identity,cut=c,queried_actor=CHARS[q],person_pair=[CHARS[i] for i in people],room_pair=[ROOMS[i] for i in rooms])
                progress['current_trial']=dict(record,world=frames)
                for kind,pair in (('person',people),('room',rooms)):
                    progress['current_trial']['active_family']=kind
                    result,after=candidate(frames,c,q,pair,kind,recompute_answer,oracle);record[kind]=result;collect(stats[kind],result)
                    if result['legal']:
                        digest=result['after_world_sha256'];duplicates['transformed_repeat_within_run']+=int(digest in seen_counterfactuals[n])
                        duplicates['transformed_equals_seen_source']+=int(digest in seen_worlds[n]);seen_counterfactuals[n].add(digest)
                        for direction in POLICY['directions']:
                            for category in evidence_categories(result,direction):
                                key=f'{n}_{kind}_{direction}_{category}'
                                if key not in examples:
                                    need(len(examples)<POLICY['maximum_saved_examples'],'Fixed feasibility witness cap exceeded')
                                    file=data/(key+'.json');save(file,dict(n=n,trial=trial,cut=c,queried_actor=CHARS[q],pair=list(pair),family=kind,
                                        direction=direction,category=category,before=frames,after=after,result=result,feasibility_example_only=True))
                                    examples[key]=dict(file=str(file),sha256=sha(file))
                ledger.write(json.dumps(record,sort_keys=True,allow_nan=False,separators=(',',':'))+'\n');ledger_rows+=1
                progress['completed_worlds']=ledger_rows;progress['oracle_checks']=dict(oracle)
                if (trial+1)%1000==0:
                    ledger.flush();save(out/f'progress_{n}_{trial+1:05d}.json',dict(n=n,completed_trials=trial+1,
                        statistics={kind:report_stats(value) for kind,value in stats.items()},duplicates=dict(duplicates),oracle_checks=dict(oracle)))
            population[str(n)]=dict(statistics={kind:report_stats(value) for kind,value in stats.items()},duplicates=dict(duplicates),
                distinct_sampled_sources=len(seen_worlds[n]),distinct_transformed_worlds=len(seen_counterfactuals[n]),
                all_draws_counted_without_deduplication=True)
    need(ledger_rows==40000 and all(population[str(n)]['statistics'][kind]['attempts']==20000
         for n in POLICY['lengths'] for kind in POLICY['candidate_families']),'Exact fixed attempt population incomplete')
    progress.pop('current_trial',None)
    return dict(protocol=PROTOCOL,passed=True,completed=True,policy=POLICY,tests=test_descriptor,statistics=population,
        oracle_checks=dict(oracle),world_trials=ledger_rows,candidate_proposals=2*ledger_rows,ledger_file=str(ledger_path),ledger_sha256=sha(ledger_path),
        examples=examples,example_count=len(examples),examples_are_not_a_benchmark_cohort=True,
        acceptance_outcome_is_not_completion_gate=True,original_exposure_overlap_audit_deferred=True,
        no_model_output_access=True,no_followup_release=True)


def main():
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu'
         and int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==4 and not os.environ.get('SLURM_JOB_GPUS'), 'CPU-only four-core Slurm allocation required')
    started=time.perf_counter();out=OUT/f'audit_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False)
    data=DATA/out.name;data.mkdir(parents=True,exist_ok=False)
    own,inherited,bindings=source_maps();(out/'source').mkdir();progress={}
    for name,digest in {**own,**inherited}.items():
        target=out/'source'/name.replace('/','_');target.write_bytes((REPO/name).read_bytes());need(sha(target)==digest,'Archived feasibility source changed')
    for file,digest in bindings.items():
        target=out/'source'/('upstream_'+str(Path(file).relative_to(UPSTREAM)).replace('/','_'))
        target.write_bytes(Path(file).read_bytes());need(sha(target)==digest,'Archived upstream source changed')
    save(out/'request.json',dict(protocol=PROTOCOL,policy=POLICY,source_sha256=own,inherited_source_sha256=inherited,input_bindings=bindings))
    try:
        result=run(out,data,started,progress);result.update(source_sha256=own,inherited_source_sha256=inherited,input_bindings=bindings,
            environment=dict(python=sys.version,pillow=importlib.metadata.version('Pillow')),elapsed_seconds=time.perf_counter()-started)
        save(out/'analysis.json',result)
        (out/'REPORT.md').write_text('CPU suffix-counterfactual feasibility completed the fixed40,000 worlds.\n\n'
            'Acceptance rates and exact denominators are in analysis.json. Strong-subset zero counts are valid outcomes. '
            'No images, models, predictions or final benchmark cohort were used or released.\n')
        artifacts={str(f):sha(f) for root in (out,data) for f in root.iterdir() if f.is_file()};save(out/'artifacts.json',artifacts)
        need(source_maps()==(own,inherited,bindings) and time.perf_counter()-started<300,'Feasibility source or CPU cap changed')
        save(out/'summary.json',dict(protocol=PROTOCOL,passed=True,completed=True,policy=POLICY,source_sha256=own,inherited_source_sha256=inherited,
            input_bindings=bindings,analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),
            artifacts_file=str(out/'artifacts.json'),artifacts_sha256=sha(out/'artifacts.json'),elapsed_seconds=time.perf_counter()-started,
            world_trials=40000,acceptance_outcome_is_not_completion_gate=True,no_followup_release=True))
    except BaseException as exc:
        if 'current_trial' in progress:
            failure_context=data/'failure_context.json';save(failure_context,progress.pop('current_trial'))
            progress['current_trial_packet']=dict(file=str(failure_context),sha256=sha(failure_context))
        save(out/'failure.json',dict(protocol=PROTOCOL,passed=False,type=type(exc).__name__,message=str(exc),progress=progress,
            elapsed_seconds=time.perf_counter()-started,source_sha256=own,inherited_source_sha256=inherited,input_bindings=bindings,
            partial_outputs_retained=True,no_automatic_retry=True,no_followup_release=True));raise


if __name__=='__main__':main()

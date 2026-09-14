"""Pinned original-MMReD recovery/answer/overlap audit; CPU Slurm only.

No downloaded Python source is executed. No images, models, tokens or fits.
Official splits remain unchanged even when their semantic worlds overlap.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
PROTOCOL = 'mmred_official_pinned_recovery'
HF_REPO = 'ef1e43ce/mmred'
HF_REVISION = 'd7963ebac13621dd1105e7596d8ed4a4c442119a'
UPSTREAM = 'https://github.com/Fr0do/mmred.git'
UPSTREAM_COMMIT = '56c6ee7041d539c7273d42d5a6c4c5e922015e40'
DATA = Path('/mnt/data/gabriele/gnn_transformer/mmred_official')
OUT = REPO / 'outputs/native_aggregation_vlm/mmred_official_recovery'
PROPOSAL = 'docs/paper/NATIVE_AGGREGATION_MMRED_OFFICIAL_RECOVERY_PROPOSAL.md'
PROPOSAL_SHA = 'cdc4ea66f3aaa255a3106ca5924ed638e38404bf7b621b437d3e56aa67858843'
OWN = ('scripts/stage_mmred_official_recovery.py', PROPOSAL, 'slurm/mmred_official_recovery.sbatch')
INHERITED = {
    'gnnformer/mmred_hf.py': '3fb97185198d1d9e90eaeef2414e01b07d8f84271598601f0c372bb89e37d492',
    'scripts/mmred_hf/eval_frozen.py': '2157088de321af6d797976a7bcfe987715bbc1026e73b88f980d18fa4054b986',
}
LENGTHS = (1, 2, 4, 8, 16, 32, 64, 128)
TASKS = ('spend_together', 'where_spend', 'steps_in_room', 'char_at_frame')
CHARS = ('Daniel', 'John', 'Mary', 'Michael', 'Sandra')
ROOMS = ('Bathroom', 'Bedroom', 'Garden', 'Hallway', 'Kitchen', 'Office')
CELLS = tuple((n, split) for n in LENGTHS for split in (('train', 'val', 'test') if n <= 16 else ('test',)))
POLICY = dict(protocol=PROTOCOL, cpu_seconds=600, cpu_cores=4, memory_gib=16,
    hf_bytes_cap=256*1024**2, hf_files=66, hf_declared_bytes=82014317, upstream_bytes_cap=128*1024**2, upstream_fetch_seconds=120,
    upstream_tree='8faf4592c479b37fb3a675ee35a97c3f30b06ac0', upstream_files=96, upstream_worktree_bytes=629906,
    expected_cells=18, expected_rows=39600, tasks=list(TASKS), train_lengths=[1,2,4,8,16],
    val_lengths=[8,16], test_lengths=[8,16,32,64,128], train_rows=4000, val_rows=400, test_rows=1000,
    original_questions_and_splits=True, primary_test_lengths=[8,16,32], held_longer_lengths=[64,128],
    overlap_is_descriptive=True, no_training_or_inference_release=True, no_rendering=True,
    no_model_or_head_calls=True, no_added_questions=True, historical_test_use_disclosed=True)


def need(condition, message):
    if not condition: raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def sources():
    need(sha(REPO/PROPOSAL) == PROPOSAL_SHA, 'Recovery proposal changed')
    need(all(sha(REPO/name) == digest for name, digest in INHERITED.items()), 'Bound local helper changed')
    return {name: sha(REPO/name) for name in OWN}


def check_time(started):
    need(time.perf_counter()-started < POLICY['cpu_seconds'], 'Recovery CPU wall cap exceeded')


def parse_row(row, n):
    required = {'question', 'seq_len', 'answer', 'qid', 'qtype', 'atype'}
    need(required <= set(row) and row['seq_len'] == n and isinstance(row['question'], str), 'Official row schema/length differs')
    sequence, nonstates = [], []
    for line in row['question'].splitlines():
        text = line.strip()
        if not text: continue
        if text.startswith('{') and text.endswith('}'):
            value = ast.literal_eval(text)
            need(set(value) == {'step_id','rooms'}, 'Unknown state-frame fields')
            sequence.append(value)
        else: nonstates.append(text)
    need(len(sequence) == n and nonstates, 'State/question separation differs')
    need([s['step_id'] for s in sequence] == list(range(1,n+1)), 'Noncontiguous official steps')
    canonical = []
    for frame in sequence:
        rooms = frame['rooms']
        need(isinstance(rooms, dict) and set(rooms) == set(ROOMS), 'Official six-room schema differs')
        need(all(isinstance(v, list) and all(isinstance(c, str) for c in v) for v in rooms.values()), 'Invalid occupant list')
        occupants = [c for values in rooms.values() for c in values]
        need(Counter(occupants) == Counter(CHARS), 'Each of the five people must occur exactly once per frame')
        lookup = {c: ROOMS.index(room) for room, values in rooms.items() for c in values}
        canonical.append(tuple(lookup[c] for c in CHARS))
    need(all(sum(a != b for a,b in zip(left,right)) == 1 for left,right in zip(canonical,canonical[1:])),
         'Each official transition must move exactly one person')
    result = {key: row[key] for key in ('qid','seq_len','qtype','atype','answer')}
    result.update(question=nonstates[-1], sequence=sequence)
    return result, canonical, nonstates[:-1]


def overlap_audit(records):
    full = defaultdict(list)
    for row in records: full[row['world_sha256']].append(row)
    repeated = {key: [dict(sid=r['sid'], cell=r['cell'], split=r['split'], n=r['n'], qtype=r['qtype']) for r in rows]
                for key, rows in full.items() if len(rows) > 1}
    cross = {key: rows for key, rows in repeated.items() if len({r['split'] for r in rows}) > 1}
    train_worlds = {r['world_sha256'] for r in records if r['pilot_role'] == 'train'}
    selected = {}
    for role in ('val','test'):
        rows = [r for r in records if r['pilot_role'] == role]
        matches = [r['sid'] for r in rows if r['world_sha256'] in train_worlds]
        selected[role] = dict(rows=len(rows), exact_training_world_matches=len(matches), matching_sids=matches)
    return dict(rows=len(records), unique_worlds=len(full), repeated_world_groups=repeated,
        cross_official_split_world_groups=cross, selected_pilot=selected,
        selected_pilot_world_disjoint=all(not v['exact_training_world_matches'] for v in selected.values()),
        official_splits_unchanged=True, overlap_is_not_an_integrity_failure=True)


def self_test():
    rooms = {room: [] for room in ROOMS}; rooms['Kitchen'] = list(CHARS)
    def raw(sequence):
        return dict(qid='tiny',seq_len=len(sequence),qtype='char_at_frame',atype='room',answer='Kitchen',
                    question='\n'.join(repr(dict(step_id=i+1,rooms=x)) for i,x in enumerate(sequence))+'\nIn which room was Mary at step 1?')
    a, canonical, _ = parse_row(raw([rooms]), 1)
    reversed_rooms = {r:list(reversed(v)) for r,v in rooms.items()}
    b, other, _ = parse_row(raw([reversed_rooms]), 1)
    need(object_sha(canonical) == object_sha(other) and object_sha(a['sequence']) != object_sha(b['sequence']), 'Semantic versus rendered-order identity fixture failed')
    bad = {r:list(v) for r,v in rooms.items()}; bad['Office'] = ['Mary']
    for sequence in ([bad], [rooms,rooms]):
        try: parse_row(raw(sequence),len(sequence))
        except ValueError: pass
        else: raise ValueError('Invalid ownership/transition fixture did not reject')
    records = [dict(sid=str(i),cell=split,n=1,split=split,qtype='char_at_frame',world_sha256=world,pilot_role=split)
               for i,(split,world) in enumerate((('train','a'),('train','a'),('val','a'),('test','b')))]
    audit = overlap_audit(records)
    need(audit['unique_worlds']==2 and len(audit['cross_official_split_world_groups'])==1
         and audit['selected_pilot']['val']['exact_training_world_matches']==1
         and audit['selected_pilot']['test']['exact_training_world_matches']==0, 'Overlap accounting fixture failed')
    return dict(passed=True,semantic_identity=True,occupant_order_invariance=True,invalid_ownership_and_transition_rejected=True,overlap_accounting=True)


def fetch_inputs(data, out, started):
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().repo_info(HF_REPO, repo_type='dataset', revision=HF_REVISION, files_metadata=True)
    need(info.sha == HF_REVISION, 'HF revision resolution differs')
    inventory = []
    for item in info.siblings:
        path = Path(item.rfilename)
        need(not path.is_absolute() and '..' not in path.parts and isinstance(item.size,int) and item.size >= 0, 'Unsafe/unknown-size HF entry')
        lfs = getattr(item,'lfs',None)
        digest = lfs.get('sha256') if isinstance(lfs,dict) else getattr(lfs,'sha256',None)
        inventory.append(dict(path=item.rfilename,bytes=item.size,lfs_sha256=digest,git_blob_id=getattr(item,'blob_id',None)))
    save(out/'remote_inventory.json',dict(repo=HF_REPO,requested_revision=HF_REVISION,resolved_revision=info.sha,
        files=inventory,total_declared_bytes=sum(x['bytes'] for x in inventory)))
    need(len(inventory)==POLICY['hf_files'] and sum(x['bytes'] for x in inventory)==POLICY['hf_declared_bytes']
         and POLICY['hf_declared_bytes']<=POLICY['hf_bytes_cap'], 'HF exact inventory/size cap differs')
    check_time(started); raw=data/'hf_snapshot'
    snapshot_download(HF_REPO,repo_type='dataset',revision=HF_REVISION,local_dir=str(raw),cache_dir=str(data/'hf_cache'),max_workers=4)
    files={}
    for item in inventory:
        path=raw/item['path'];need(path.is_file() and not path.is_symlink() and path.stat().st_size==item['bytes'],'HF file missing/size differs')
        digest=sha(path);need(not item['lfs_sha256'] or digest==item['lfs_sha256'],'HF LFS content digest differs');files[str(path)]=digest
        if not item['lfs_sha256'] and item['git_blob_id']:
            content=path.read_bytes();git_digest=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
            need(git_digest==item['git_blob_id'],'HF Git blob identity differs')
    save(out/'hf_files.json',files);check_time(started)
    upstream=data/'upstream';upstream.mkdir()
    commands=(['git','init',str(upstream)],['git','-C',str(upstream),'remote','add','origin',UPSTREAM],
              ['git','-C',str(upstream),'-c','core.hooksPath=/dev/null','fetch','--depth=1','--no-tags','origin',UPSTREAM_COMMIT],
              ['git','-C',str(upstream),'-c','core.hooksPath=/dev/null','checkout','--detach',UPSTREAM_COMMIT])
    git_logs=[]
    for command in commands:
        remaining=max(1,min(POLICY['upstream_fetch_seconds'],int(POLICY['cpu_seconds']-(time.perf_counter()-started))))
        result=subprocess.run(command,capture_output=True,text=True,timeout=remaining,check=False)
        git_logs.append(dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        save(out/f'git_command_{len(git_logs)}.json',git_logs[-1]);need(result.returncode==0,'Pinned upstream source recovery failed')
    commit=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True,timeout=10).strip()
    status=subprocess.check_output(['git','-C',str(upstream),'status','--porcelain'],text=True,timeout=10)
    tree=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD^{tree}'],text=True,timeout=10).strip()
    need(commit==UPSTREAM_COMMIT and tree==POLICY['upstream_tree'] and not status,'Upstream commit/tree/worktree differs')
    upstream_files={};total=0
    for path in upstream.rglob('*'):
        need(not path.is_symlink(),'Unexpected upstream symlink')
        if path.is_file():
            total+=path.stat().st_size
            if '.git' not in path.relative_to(upstream).parts:upstream_files[str(path)]=sha(path)
    need(total<=POLICY['upstream_bytes_cap'] and (upstream/'scripts/render_images.py').is_file(),'Upstream size/renderer inventory differs')
    worktree_bytes=sum(Path(path).stat().st_size for path in upstream_files)
    need(len(upstream_files)==POLICY['upstream_files'] and worktree_bytes==POLICY['upstream_worktree_bytes'],'Upstream exact source inventory differs')
    save(out/'upstream_files.json',upstream_files)
    save(out/'acquisition.json',dict(hf_revision=info.sha,hf_bytes=sum(x['bytes'] for x in inventory),
        upstream_commit=commit,upstream_tree=tree,upstream_worktree_bytes=worktree_bytes,upstream_repository_bytes=total,upstream_tracked_clean=True,
        git_version=subprocess.check_output(['git','--version'],text=True,timeout=10).strip(),
        packages={name:importlib.metadata.version(name) for name in ('datasets','pyarrow','huggingface_hub')}))
    return raw, files, upstream_files


def pilot_role(n,split,qtype):
    if qtype not in TASKS:return None
    if split=='train' and n in POLICY['train_lengths']:return 'train'
    if split=='val' and n in POLICY['val_lengths']:return 'val'
    if split=='test' and n in POLICY['test_lengths']:return 'test'
    return None


def audit_dataset(raw, data, out, started):
    from datasets import Dataset
    from gnnformer.mmred_hf import NIAH_QTYPES,DC_QTYPES,recompute_answer
    qtypes=set(NIAH_QTYPES+DC_QTYPES);need(len(qtypes)==24 and len(NIAH_QTYPES)==15 and len(DC_QTYPES)==9,'Task inventory differs')
    groups=defaultdict(list)
    for path in raw.rglob('*.arrow'):
        match=re.fullmatch(r'seq_len_(\d+)/(train|val|test)/[^/]+\.arrow',path.relative_to(raw).as_posix())
        need(match is not None,'Unknown original Arrow layout');groups[(int(match[1]),match[2])].append(path)
    save(out/'arrow_inventory.json',{f'N{n}_{split}':[str(p) for p in sorted(paths)] for (n,split),paths in groups.items()})
    need(set(groups)==set(CELLS),'Expected all18 official config/split cells')
    errors=[];records=[];cells=[];pilot={};prefixes=defaultdict(list);full_by_length=defaultdict(lambda:defaultdict(list))
    for n,split in CELLS:
        cell=f'seq_len_{n}_{split}';counts=Counter();atypes=Counter();golds=defaultdict(Counter);seen=set();selected=[];cell_records=[];index=0
        for path in sorted(groups[(n,split)]):
            ds=Dataset.from_file(str(path))
            for file_index,row in enumerate(ds):
                if index%100==0:check_time(started)
                sid=f'{cell}/{row.get("qid",index)}';qtype=row.get('qtype');counts[qtype]+=1;atypes[row.get('atype')]+=1
                golds[qtype][str(row.get('answer'))]+=1
                try:
                    need(isinstance(row['qid'],str) and row['qid'] not in seen,'Duplicate/nonstring qid within official cell');seen.add(row['qid'])
                    need(qtype in qtypes,'Unknown qtype')
                    parsed,world,preamble=parse_row(row,n)
                    computed=recompute_answer(qtype,parsed['question'],[{'rooms':x['rooms']} for x in parsed['sequence']])
                    need(computed is not None and str(computed)==str(row['answer']),'Original answer differs from all-task recomputation')
                    role=pilot_role(n,split,qtype);world_sha=object_sha(world)
                    record=dict(sid=sid,cell=cell,n=n,split=split,qtype=qtype,atype=row['atype'],qid=row['qid'],
                        source_file=str(path),source_row=file_index,source_cell_row=index,raw_row_sha256=object_sha(row),world_sha256=world_sha,
                        exact_sequence_sha256=object_sha(parsed['sequence']),pilot_role=role,preamble_lines=preamble)
                    records.append(record);cell_records.append(record);full_by_length[n][world_sha].append(record)
                    for k in LENGTHS:
                        if k<n:prefixes[(k,object_sha(world[:k]))].append(dict(sid=sid,cell=cell,n=n,split=split))
                    if role:selected.append(dict(parsed,**{'source_identity':record}))
                except (ValueError,KeyError,TypeError,SyntaxError,IndexError) as exc:
                    errors.append(dict(sid=sid,cell=cell,source_file=str(path),source_row=file_index,source_cell_row=index,type=type(exc).__name__,message=str(exc)))
                index+=1
        expected=200 if split=='train' else 50
        if dict(counts)!={q:expected for q in qtypes}:errors.append(dict(cell=cell,type='InventoryError',message='Exact24 per-type counts differ',observed=dict(counts)))
        entry=dict(cell=cell,n=n,split=split,rows=index,qtypes=dict(counts),atypes=dict(atypes),gold_counts={q:dict(v) for q,v in golds.items()},
            rows_answer_and_world_passed=len(cell_records),errors=sum(e.get('cell')==cell for e in errors),original_order_preserved=True)
        save(out/f'audit_{cell}.json',entry);cells.append(entry)
        if selected:pilot[cell]=selected
    save(out/'row_identities.json',records);save(out/'answer_and_inventory_audit.json',dict(cells=cells,rows=sum(c['rows'] for c in cells),errors=errors,passed=not errors))
    overlap=overlap_audit(records);save(out/'world_overlap.json',overlap)
    prefix_matches=[]
    for (k,digest),long_rows in sorted(prefixes.items()):
        short_rows=full_by_length[k].get(digest,[])
        if short_rows:
            prefix_matches.append(dict(prefix_length=k,world_sha256=digest,
                shorter_worlds=[dict(sid=r['sid'],cell=r['cell'],split=r['split']) for r in short_rows],longer_worlds=long_rows,
                crosses_official_split=len({a['split'] for a in short_rows}|{b['split'] for b in long_rows})>1))
    save(out/'prefix_overlap.json',dict(groups=prefix_matches,groups_count=len(prefix_matches),
        cross_official_split_groups=sum(x['crosses_official_split'] for x in prefix_matches),exact_leading_prefix_only=True,
        shorter_queries_not_equivalent_to_longer_queries=True,official_splits_unchanged=True))
    need(not errors and sum(c['rows'] for c in cells)==39600,'Complete official inventory/answer/world audit failed; all diagnostic rows retained')
    renderer=data/'renderer_json';renderer.mkdir();manifest=[];files={}
    for n,split in CELLS:
        cell=f'seq_len_{n}_{split}'
        if cell not in pilot:continue
        rows=pilot[cell];path=renderer/f'{cell}.json'
        # Original renderer schema is kept exact; identities live in the separate manifest.
        save(path,[{k:v for k,v in r.items() if k!='source_identity'} for r in rows]);files[str(path)]=sha(path)
        manifest.extend(dict(row['source_identity'],question=row['question'],answer=row['answer'],renderer_file=str(path),renderer_row=i) for i,row in enumerate(rows))
    need(Counter(r['pilot_role'] for r in manifest)=={'train':4000,'val':400,'test':1000},'Selected pilot counts differ')
    for role,lengths,count in (('train',POLICY['train_lengths'],200),('val',POLICY['val_lengths'],50),('test',POLICY['test_lengths'],50)):
        need(Counter((r['n'],r['qtype']) for r in manifest if r['pilot_role']==role)=={(n,q):count for n in lengths for q in TASKS},'Pilot per-task/length balance differs')
    save(out/'pilot_rows.json',manifest)
    return files,dict(total_rows=39600,pilot_rows=5400,pilot_counts={'train':4000,'val':400,'test':1000},
        cells=cells,all24_answer_parity=True,selected_pilot_world_disjoint=overlap['selected_pilot_world_disjoint'],
        selected_overlap={k:{name:value for name,value in v.items() if name!='matching_sids'} for k,v in overlap['selected_pilot'].items()})


def verify_stage(path):
    path=Path(path).resolve();path=path/'summary.json' if path.is_dir() else path
    summary=json.loads(path.read_text())
    need(summary['passed'] is True and summary['completed'] is True and not (path.parent/'failure.json').exists(),'Passed recovery summary required')
    need(summary['protocol']==PROTOCOL and summary['policy']==POLICY and summary['source_sha256']==sources()
         and summary['inherited_source_sha256']==INHERITED,'Recovery source/protocol differs')
    need(sha(summary['plan_file'])==summary['plan_sha256'],'Recovery plan changed');plan=json.loads(Path(summary['plan_file']).read_text())
    need(plan['source_sha256']==sources() and plan['inherited_source_sha256']==INHERITED and plan['no_training_or_inference_release'] is True,'Recovery plan source/gate differs')
    for key in ('input_bindings','artifacts'):
        for file,digest in plan[key].items():need(sha(file)==digest,'Recovery bound artifact changed')
    for name,digest in {**sources(),**INHERITED}.items():need(sha(path.parent/'source'/name.replace('/','_'))==digest,'Recovery archived source changed')
    return plan


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu'
         and not os.environ.get('SLURM_JOB_GPUS') and int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==4,'Only four-core CPU Slurm recovery is allowed')
    started=time.perf_counter();tag=f'recovery_{os.environ["SLURM_JOB_ID"]}'
    out=OUT/tag;data=DATA/tag;out.mkdir(parents=True,exist_ok=False);data.mkdir(parents=True,exist_ok=False)
    os.environ['HF_HOME']=str(data/'hf_home')
    os.environ['HF_DATASETS_CACHE']=str(data/'dataset_cache')
    frozen=sources();(out/'source').mkdir()
    for name,digest in {**frozen,**INHERITED}.items():
        target=out/'source'/name.replace('/','_');target.write_bytes((REPO/name).read_bytes());need(sha(target)==digest,'Source archive copy differs')
    save(out/'request.json',dict(protocol=PROTOCOL,policy=POLICY,hf_repo=HF_REPO,hf_revision=HF_REVISION,upstream=UPSTREAM,
        upstream_commit=UPSTREAM_COMMIT,source_sha256=frozen,inherited_source_sha256=INHERITED,data_directory=str(data)))
    try:
        save(out/'self_tests.json',self_test())
        raw,hf_files,upstream_files=fetch_inputs(data,out,started)
        renderer_files,result=audit_dataset(raw,data,out,started)
        tree=ast.parse((REPO/'scripts/mmred_hf/eval_frozen.py').read_text())
        prompts=[ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id=='SYSTEM_PROMPT' for t in node.targets)]
        need(len(prompts)==1 and isinstance(prompts[0],str),'Original protocol prompt not uniquely literal')
        save(out/'official_protocol.json',dict(system_prompt=prompts[0],images_before_original_question=True,
            primary_original_questions=True,added_queries=False,old_test_results_previously_inspected=True,no_inference_release=True))
        need(sources()==frozen,'Recovery sources changed during run');check_time(started)
        artifacts={str(path):sha(path) for path in out.glob('*.json')};artifacts.update(renderer_files)
        plan=dict(protocol=PROTOCOL,policy=POLICY,source_sha256=frozen,inherited_source_sha256=INHERITED,
            hf_repo=HF_REPO,hf_revision=HF_REVISION,upstream=UPSTREAM,upstream_commit=UPSTREAM_COMMIT,
            input_bindings={**hf_files,**upstream_files},artifacts=artifacts,data_directory=str(data),
            rows_file=str(out/'pilot_rows.json'),overlap_file=str(out/'world_overlap.json'),prefix_overlap_file=str(out/'prefix_overlap.json'),
            inventory_file=str(out/'answer_and_inventory_audit.json'),official_protocol_file=str(out/'official_protocol.json'),
            audit=result,no_training_or_inference_release=True)
        save(out/'plan.json',plan)
        (out/'REPORT.md').write_text('# Original MMReD recovery\n\nAll18 cells and39,600 original rows passed answer/world audits. The original4,000 training,400 validation and1,000 test pilot rows are staged. Official split/world overlap is retained in world_overlap.json and prefix_overlap.json; successful recovery does not establish a world-disjoint assay. No rendering, model call, fitting or inference was released.\n')
        check_time(started)
        save(out/'summary.json',dict(passed=True,completed=True,protocol=PROTOCOL,policy=POLICY,source_sha256=frozen,
            inherited_source_sha256=INHERITED,plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),
            report_file=str(out/'REPORT.md'),report_sha256=sha(out/'REPORT.md'),elapsed_seconds=time.perf_counter()-started,
            audit=result,no_training_or_inference_release=True))
    except BaseException as exc:
        save(out/'failure.json',dict(protocol=PROTOCOL,type=type(exc).__name__,message=str(exc),source_sha256=frozen,
            elapsed_seconds=time.perf_counter()-started,partial_outputs_retained=True,no_training_or_inference_release=True))
        raise


if __name__=='__main__':main()

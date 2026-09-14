"""Stage a fixed MMReD identity join on Slurm CPU only.

Check freezes all symbolic families, tokenizer targets and canonical image reuse.
Render additionally requires the complete passed native inclusion audit. No model,
head, training cache, fit, filtering or answer computation is used at inference.
Existing sources/data are read only; every attempt has a new immutable directory.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import sys
import time

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from scripts.stage_native_vision_pilot import PARK_ROOMS, digest, prepare_renderer, source_states
from scripts.stage_native_vision_v2_clean import CHARS, interleave, stable_seed

DATA_ROOT = Path('/mnt/data/gabriele/gnn_transformer/identity_join')
OUT = REPO / 'outputs/native_aggregation_vlm/identity_join/data_staging'
MODEL = Path('/mnt/ckpts/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5')
REUSE_MANIFEST = Path('/mnt/data/gabriele/gnn_transformer/v18_fresh/main_manifest.json')
PARITY_MANIFEST = Path('/mnt/data/gabriele/gnn_transformer/native_aggregation_vision_pilot/main_manifest.json')
SEED = 20261122
PROTOCOL = 'native_identity_join_data_preparation'
PAIRS = tuple(itertools.combinations(PARK_ROOMS, 2))
HELD_PAIRS = (('Kitchen', 'Bathroom'), ('Garden', 'Office'), ('Bedroom', 'Park'))
TRAIN_PAIRS = tuple(p for p in PAIRS if p not in HELD_PAIRS)
SEEN_TEST_PAIRS = (('Kitchen', 'Garden'), ('Bathroom', 'Bedroom'), ('Office', 'Park'))
DEV_TRIOS = ((0, 1, 2), (3, 4, 5), (6, 7, 8))
TEST_TRIOS = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8),
              (0, 4, 8), (1, 5, 6), (2, 3, 7), (0, 5, 7), (1, 3, 8), (2, 4, 6))
GLOBAL_TEMPLATE = "Consider only images in the {room_a} or the {room_b}. Which person appears in both rooms? Reply with the person's name only."
LOCAL_TEMPLATE = "Collection question: {question}\nDoes this image satisfy the question's inclusion restriction? Reply with exactly 1 for yes or 0 for no."
SPLIT_COUNTS = {'train': 6048, 'dev': 108, 'test_seen': 216, 'test_held': 216}
CELL_COUNTS = {'train_N8': 3024, 'train_N16': 3024, 'dev_N16': 108,
               'test_seen_N32': 108, 'test_seen_N64': 108, 'test_held_N32': 108, 'test_held_N64': 108}
POLICY = dict(protocol=PROTOCOL, seed=SEED, people=CHARS, rooms=PARK_ROOMS, relevant_count=6,
              split_counts=SPLIT_COUNTS, cell_counts=CELL_COUNTS, contexts=6588, contrast_families=1116,
              length_pairs=3240, image_occurrences=95040, train_pairs=TRAIN_PAIRS, held_pairs=HELD_PAIRS,
              seen_test_pairs=SEEN_TEST_PAIRS, dev_trios=DEV_TRIOS, test_trios=TEST_TRIOS,
              global_template=GLOBAL_TEMPLATE, local_template=LOCAL_TEMPLATE,
              target_eos=151645, allowed_eos=[151645,151643], max_answer_tokens=4,
              complete_gate_judgments_required=4860, scalar_count=6, no_training_release=True,
              orientation='(pair_index + trio_index) modulo2; fixed before collision rejection',
              background='IID uniform among all9people and the4 unselected rooms',
              freshness='Complete context/question uniqueness across all new splits; visual atoms reused',
              source_only_before_gate=True, images='Original512RGB canonical Step1..N; no augmentation')
OWN = ('scripts/stage_native_identity_join.py', 'slurm/native_identity_join_check.sbatch',
       'slurm/native_identity_join_render.sbatch', 'scripts/stage_native_vision_pilot.py',
       'scripts/stage_native_vision_v2_clean.py', 'datasets/mmred/render_mmred.py')


def need(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def oid(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def source_hashes():
    return {name: digest(REPO / name) for name in OWN}


def snapshot(out):
    frozen = source_hashes()
    (out / 'source').mkdir()
    for name, expected in frozen.items():
        data = (REPO / name).read_bytes()
        need(hashlib.sha256(data).hexdigest() == expected, 'Source changed during snapshot')
        with (out / 'source' / name.replace('/', '_')).open('xb') as stream:
            stream.write(data)
    save(out / 'source_hashes.json', frozen)
    return frozen


def global_prompt(room_pair):
    need(tuple(room_pair) in PAIRS, 'Require canonical unordered room pair')
    return GLOBAL_TEMPLATE.format(room_a=room_pair[0], room_b=room_pair[1])


def local_prompt(question):
    need(isinstance(question, str) and bool(question), 'Complete global question required')
    return LOCAL_TEMPLATE.format(question=question)


def split_specs():
    return (('train', TRAIN_PAIRS, tuple(itertools.combinations(range(9), 3)), (8, 16)),
            ('dev', TRAIN_PAIRS, DEV_TRIOS, (16,)),
            ('test_seen', SEEN_TEST_PAIRS, TEST_TRIOS, (32, 64)),
            ('test_held', HELD_PAIRS, TEST_TRIOS, (32, 64)))


def states_from_frames(frames):
    return [dict(step_id=i+1, rooms={room: [person]}) for i, (person, room) in enumerate(frames)]


def make_family(split, pair, trio, lengths, pair_index, trio_index, used):
    label = f'{split}/{pair_index}/{trio_index}'
    rng = random.Random(stable_seed(SEED, label))
    orientation = (pair_index + trio_index) % 2
    people = [CHARS[i] for i in trio]
    outside = [room for room in PARK_ROOMS if room not in pair]
    def background():
        return ('negative', rng.choice(CHARS), rng.choice(outside))
    for attempt in range(10000):
        layout = [('positive', i, j) for i in range(3) for j in range(2)]
        layout += [background() for _ in range(lengths[0]-6)]
        rng.shuffle(layout)
        flips = [rng.randrange(2) for _ in range(3)]
        layouts = [(lengths[0], layout, [])]
        if len(lengths) == 2:
            extended, positions = interleave(rng, layout, [background() for _ in range(lengths[1]-lengths[0])])
            layouts.append((lengths[1], extended, positions))
        family = []
        for variant in range(3):
            others = [i for i in range(3) if i != variant]
            for n, tokens, positions in layouts:
                frames = []
                for kind, a, b in tokens:
                    if kind == 'negative':
                        frames.append([a, b])
                    else:
                        room_index = (b ^ flips[a]) if a == variant else (others.index(a) ^ orientation)
                        frames.append([people[a], pair[room_index]])
                states = states_from_frames(frames)
                question = global_prompt(pair)
                content = oid(dict(states=states, question=question))
                family.append(dict(sid='ij_'+content[:24], split=split, contrast_id=label,
                    variant=variant, pair_id=f'{label}/v{variant}', n_frames=n, k=6,
                    room_pair=list(pair), trio=people, orientation=orientation,
                    gold=people[variant], question=question, local_prompt=local_prompt(question),
                    frames=frames, states=states, content_sha256=content,
                    parent_n_frames=lengths[0] if positions else None, parent_positions=positions,
                    generation_attempt=attempt, seed=SEED))
        hashes = [row['content_sha256'] for row in family]
        if len(set(hashes)) == len(hashes) and not set(hashes) & used:
            used.update(hashes)
            return family
    raise ValueError('No fresh complete family found: '+label)


def generate():
    rows, used = [], set()
    for split, pairs, trios, lengths in split_specs():
        for pi, pair in enumerate(pairs):
            for ti, trio in enumerate(trios):
                rows.extend(make_family(split, pair, trio, lengths, pi, ti, used))
    return rows


def audit_row(row):
    frames, pair, trio = row['frames'], row['room_pair'], row['trio']
    need(len(frames) == row['n_frames'] and tuple(pair) in PAIRS and len(set(trio)) == 3
         and all(p in CHARS for p in trio), 'Invalid join frame/pair/trio inventory')
    need(all(p in CHARS and r in PARK_ROOMS for p,r in frames), 'Unknown canonical frame')
    need(row['states'] == states_from_frames(frames), 'Step/state mismatch')
    relevant = [(i,p,r) for i,(p,r) in enumerate(frames) if r in pair]
    need(row['k'] == len(relevant) == 6, 'Relevant count changed')
    need(Counter(p for _,p,_ in relevant) == Counter({p:2 for p in trio}), 'Person marginal changed')
    need(Counter(r for _,_,r in relevant) == Counter({r:3 for r in pair}), 'Room marginal changed')
    joins = [p for p in trio if {r for _,person,r in relevant if person == p} == set(pair)]
    need(joins == [row['gold']] and row['gold'] == trio[row['variant']], 'Unique identity-join answer differs')
    others=[p for p in trio if p!=row['gold']]
    need(type(row['orientation']) is int and row['orientation'] in (0,1), 'Invalid room orientation')
    for index,person in enumerate(others):
        need({room for _,p,room in relevant if p==person}=={pair[index ^ row['orientation']]}, 'Actual nonanswer room orientation differs')
    need(row['question'] == global_prompt(pair) and row['local_prompt'] == local_prompt(row['question']), 'Literal prompt differs')
    content = oid(dict(states=row['states'], question=row['question']))
    need(content == row['content_sha256'] and row['sid'] == 'ij_'+content[:24], 'Content/SID differs')
    need(row['seed'] == SEED, 'Data seed differs')
    return [i for i,_,_ in relevant]


def audit_family(rows):
    by_n = defaultdict(list)
    for row in rows:
        audit_row(row)
        by_n[row['n_frames']].append(row)
    for n, group in by_n.items():
        group.sort(key=lambda row: row['variant'])
        first = group[0]
        need([r['variant'] for r in group] == [0,1,2] and {r['gold'] for r in group} == set(first['trio']), 'All three changing answers required')
        positions = audit_row(first)
        for row in group:
            need(all(row[k] == first[k] for k in ('split','contrast_id','trio','room_pair','orientation','generation_attempt','parent_positions')), 'Contrast metadata differs')
            need(audit_row(row) == positions and [p for p,_ in row['frames']] == [p for p,_ in first['frames']], 'Occurrence slots or person sequence changed')
            need([f for i,f in enumerate(row['frames']) if i not in positions] == [f for i,f in enumerate(first['frames']) if i not in positions], 'Contrast backgrounds changed')
    if len(by_n) == 1:
        need(all(r['split']=='dev' and r['n_frames']==16 and r['parent_n_frames'] is None and r['parent_positions']==[] for r in rows), 'Only dev is unpaired')
    else:
        need(len(by_n)==2, 'Invalid length-pair count')
        low, high = sorted(by_n)
        need((low,high)==((8,16) if rows[0]['split']=='train' else (32,64)), 'Length support differs')
        for parent,child in zip(sorted(by_n[low],key=lambda r:r['variant']), sorted(by_n[high],key=lambda r:r['variant'])):
            positions = child['parent_positions']
            need(parent['parent_n_frames'] is None and parent['parent_positions']==[] and child['parent_n_frames']==low, 'Parent metadata differs')
            need(len(positions)==low and positions==sorted(set(positions)) and all(0<=i<high for i in positions), 'Invalid insertion positions')
            need([child['frames'][i] for i in positions]==parent['frames'], 'Parent semantic subsequence changed')
            need(all(room not in child['room_pair'] for i,(_,room) in enumerate(child['frames']) if i not in set(positions)), 'Insertion added evidence')
            need(child['gold']==parent['gold'] and child['pair_id']==parent['pair_id'], 'Paired answer identity differs')


def audit_rows(rows):
    families = defaultdict(list)
    for row in rows:
        families[row['contrast_id']].append(row)
    need(len(rows)==6588 and len(families)==1116, 'Complete cardinality differs')
    need(Counter(r['split'] for r in rows)==SPLIT_COUNTS, 'Split cardinality differs')
    cells = Counter(f"{r['split']}_N{r['n_frames']}" for r in rows)
    need(cells==CELL_COUNTS, 'Length cardinality differs')
    need(len({r['sid'] for r in rows})==len({r['content_sha256'] for r in rows})==len(rows), 'Complete context collision')
    for group in families.values():
        audit_family(group)
    for split,pairs,trios,lengths in split_specs():
        groups = [g for g in families.values() if g[0]['split']==split]
        actual = Counter((tuple(g[0]['room_pair']),tuple(CHARS.index(p) for p in g[0]['trio'])) for g in groups)
        need(actual==Counter((p,t) for p in pairs for t in trios), 'Pair/trio coverage differs')
        need(Counter(g[0]['orientation'] for g in groups)=={0:len(groups)//2,1:len(groups)//2}, 'Room orientations unbalanced')
        for n in lengths:
            subset=[r for r in rows if r['split']==split and r['n_frames']==n]
            need(Counter(r['gold'] for r in subset)=={person:len(subset)//9 for person in CHARS}, 'Answer names unbalanced')
        for pi,pair in enumerate(pairs):
            for ti,trio in enumerate(trios):
                group=families[f'{split}/{pi}/{ti}']
                need(group[0]['orientation']==(pi+ti)%2, 'Orientation policy differs')
    pairs=sum(r['parent_n_frames'] is not None for r in rows)
    occurrences=sum(r['n_frames'] for r in rows)
    need(pairs==3240 and occurrences==95040, 'Pair/image count differs')
    return dict(passed=True,contexts=len(rows),contrast_families=len(families),length_pairs=pairs,
                image_occurrences=occurrences,split_counts=dict(Counter(r['split'] for r in rows)),
                cell_counts=dict(cells),all_labels_independently_recomputed=True,
                all_contrast_marginals_slots_backgrounds_verified=True,all_insertions_verified=True,
                all_complete_contexts_unique=True,all_answer_names_balanced=True,
                all_orientation_counts_balanced=True,scalar_count_identical_in_every_contrast=True)


def selftest():
    rows=make_family('train',TRAIN_PAIRS[0],(0,1,2),(8,16),0,0,set())
    audit_family(rows)
    failures=0
    def reject(mutator):
        nonlocal failures
        bad=copy.deepcopy(rows);mutator(bad)
        try: audit_family(bad)
        except (ValueError,IndexError): failures+=1
        else: raise AssertionError('Corrupted join family was accepted')
    reject(lambda r:r[0].update(gold=CHARS[8]))
    reject(lambda r:r[0].update(k=5))
    reject(lambda r:r[0]['states'][0].update(step_id=2))
    reject(lambda r:r[1].update(parent_positions=list(reversed(r[1]['parent_positions']))))
    bad=copy.deepcopy(rows);bad[2]['frames'][0]=[CHARS[8],PARK_ROOMS[-1]]
    bad[2]['states']=states_from_frames(bad[2]['frames'])
    bad[2]['content_sha256']=oid(dict(states=bad[2]['states'],question=bad[2]['question']))
    bad[2]['sid']='ij_'+bad[2]['content_sha256'][:24]
    try: audit_family(bad)
    except ValueError: failures+=1
    else: raise AssertionError('Changed contrast occurrence accepted')
    mock=dict(sid='opaque',n_frames=1,question=global_prompt(TRAIN_PAIRS[0]),
              local_prompt=local_prompt(global_prompt(TRAIN_PAIRS[0])),image_files=[dict(path='/tmp/image.png',sha256='a'*64,dimensions=[512,512],mode='RGB')])
    poisoned=dict(mock,gold='Sandra',trio=['Sandra'],variant=0,states=['secret'])
    need(runtime_view(poisoned)==runtime_view(mock), 'Gold leaked into runtime view')
    need(Counter(i for trio in TEST_TRIOS for i in trio)=={i:4 for i in range(9)}, 'Test trio balance differs')
    return dict(passed=True,groups=7,corruptions_rejected=failures,
                tests=['valid_identity_contrast','wrong_answer','wrong_marginal','wrong_Step',
                       'wrong_insertion','changed_occurrence','safe_inputs_and_balanced_trios'])


def tokenizer_contract():
    from transformers import AutoTokenizer
    import importlib.metadata
    names=('tokenizer_config.json','tokenizer.json','vocab.json','merges.txt','special_tokens_map.json','added_tokens.json','config.json','generation_config.json')
    files={str(MODEL/name):digest(MODEL/name) for name in names if (MODEL/name).is_file()}
    need(all(str(MODEL/name) in files for name in ('tokenizer_config.json','vocab.json','merges.txt','config.json','generation_config.json')), 'Tokenizer source files missing')
    tokenizer=AutoTokenizer.from_pretrained(str(MODEL),use_fast=False,local_files_only=True)
    specials=set(tokenizer.all_special_ids)
    need(tokenizer.encode('0',add_special_tokens=False)==[15] and tokenizer.encode('1',add_special_tokens=False)==[16], 'Literal binary tokens differ')
    need(tokenizer.eos_token_id==151645 and 151645 in specials and 151643 in specials, 'Native EOS identities differ')
    targets={}
    for name in CHARS:
        ids=tokenizer.encode(name,add_special_tokens=False)
        need(ids and not set(ids)&specials and tokenizer.decode(ids,skip_special_tokens=False)==name and len(ids)+1<=4, 'Canonical name does not fit native answer budget: '+name)
        targets[name]=dict(name_ids=ids,target_ids=ids+[151645],decoded=tokenizer.decode(ids,skip_special_tokens=False))
    prompts={global_prompt(pair):dict(global_tokens=len(tokenizer.encode(global_prompt(pair),add_special_tokens=False)),
                local_tokens=len(tokenizer.encode(local_prompt(global_prompt(pair)),add_special_tokens=False))) for pair in PAIRS}
    return dict(model_directory=str(MODEL),tokenizer_class=type(tokenizer).__name__,transformers_version=importlib.metadata.version('transformers'),
                tokenizers_version=importlib.metadata.version('tokenizers'),files=files,targets=targets,
                all_special_ids=sorted(specials),count_token_ids={'0':15,'1':16},target_eos=151645,
                allowed_eos=[151645,151643],max_answer_tokens=4,prompt_lengths=prompts,
                all_names_roundtrip=True,all_targets_fit=True,no_model_loaded=True)


def needed_atoms(rows):
    return sorted({(person,room,i+1) for row in rows for i,(person,room) in enumerate(row['frames'])})


def atom_key(atom):
    return '/'.join(map(str,atom))


def reuse_inventory(rows):
    bindings={str(p):digest(p) for p in (REUSE_MANIFEST,PARITY_MANIFEST)}
    reusable={}
    for cell in read(REUSE_MANIFEST)['splits'].values():
        for row in sorted(cell['samples'],key=lambda r:r['sid']):
            directory=Path(row['path']);qa=directory/'qa.txt'
            need(digest(qa)==row['qa_sha256'], 'Canonical reuse QA changed')
            bindings[str(qa)]=row['qa_sha256'];states=source_states(directory)
            need(len(states)==row['n_frames']==len(row['image_files']), 'Reuse frame count differs')
            for i,(state,image) in enumerate(zip(states,row['image_files'])):
                need(state['step_id']==i+1 and len(state['rooms'])==1, 'Invalid canonical reuse state')
                room,persons=next(iter(state['rooms'].items()))
                need(room in PARK_ROOMS and len(persons)==1 and persons[0] in CHARS, 'Invalid canonical reuse atom')
                atom=(persons[0],room,i+1)
                item={k:image[k] for k in ('path','sha256','dimensions','mode')}
                if atom in reusable:
                    need(reusable[atom]['sha256']==item['sha256'], 'One canonical atom has different original PNG bytes')
                else: reusable[atom]=item
    atoms=needed_atoms(rows);entries=[]
    for atom in atoms:
        reference=reusable.get(atom)
        if reference:
            need(digest(Path(reference['path']))==reference['sha256'], 'Canonical reuse PNG changed')
            bindings[reference['path']]=reference['sha256']
        entries.append(dict(atom=list(atom),key=atom_key(atom),reference=reference))
    references=[]
    for row in read(PARITY_MANIFEST)['splits']['train_N8']['samples'][:2]:
        row=dict(row,source_path=row['path']);directory=Path(row['path'])
        bindings[str(directory/'qa.txt')]=digest(directory/'qa.txt')
        for i in (0,row['n_frames']//2,row['n_frames']-1):
            path=directory/f'{i:03d}.png';bindings[str(path)]=digest(path)
        references.append(row)
    need(len(references)==2, 'Two canonical renderer references required')
    return dict(entries=entries,distinct_atoms=len(atoms),reused_atoms=sum(e['reference'] is not None for e in entries),
                new_atoms=sum(e['reference'] is None for e in entries),renderer_references=references,
                input_bindings=bindings,scope='One representative of each canonical(person,room,Step) atom; no old context reused')


def verify_bindings(bindings):
    for path,expected in bindings.items():
        need(digest(Path(path))==expected, 'Bound file changed: '+path)


def check(out,frozen):
    tests=selftest();rows=generate();audit=audit_rows(rows)
    need(rows==generate(), 'Deterministic regeneration differs')
    tokens=tokenizer_contract();inventory=reuse_inventory(rows)
    target_positions={split:sum(len(tokens['targets'][r['gold']]['target_ids']) for r in rows if r['split']==split) for split in SPLIT_COUNTS}
    need(target_positions['train']==13440, 'Balanced canonical training target support differs')
    save(out/'samples.json',rows);save(out/'symbolic_audit.json',audit)
    save(out/'tokenizer.json',tokens);save(out/'reuse_inventory.json',inventory);save(out/'selftests.json',tests)
    files={str(out/name):digest(out/name) for name in ('samples.json','symbolic_audit.json','tokenizer.json','reuse_inventory.json','selftests.json')}
    plan=dict(schema_version=1,protocol=PROTOCOL,policy=POLICY,source_sha256=frozen,
              source_directory=str(out/'source'),files=files,samples_file=str(out/'samples.json'),
              tokenizer_file=str(out/'tokenizer.json'),reuse_inventory_file=str(out/'reuse_inventory.json'),
              symbolic_audit=audit,selftests=tests,data_root=str(DATA_ROOT),target_positions_per_split=target_positions,
              input_bindings={**inventory['input_bindings'],**tokens['files']},
              gate_audit_required_before_render=True,no_model_loaded=True,no_images_rendered=True)
    save(out/'plan.json',plan)
    return dict(passed=True,completed=True,check_only=True,plan_file=str(out/'plan.json'),plan_sha256=digest(out/'plan.json'),
                source_sha256=frozen,counts=audit,distinct_atoms=inventory['distinct_atoms'],reused_atoms=inventory['reused_atoms'],
                new_atoms=inventory['new_atoms'],target_positions_per_split=target_positions,all_tokenizer_targets_fit=True,no_model_loaded=True,no_images_rendered=True)


def verify_plan(path):
    path=Path(path).resolve();plan=read(path);summary=read(path.parent/'summary.json')
    need(summary['passed'] is True and summary['completed'] is True and summary['check_only'] is True
         and summary['plan_file']==str(path) and summary['plan_sha256']==digest(path), 'Passed exact CPU plan required')
    need(plan['protocol']==PROTOCOL and oid(plan['policy'])==oid(POLICY) and plan['source_sha256']==source_hashes()
         and summary['source_sha256']==source_hashes() and plan['data_root']==str(DATA_ROOT), 'Plan source/policy identity differs')
    for name,h in plan['source_sha256'].items():
        need(digest(Path(plan['source_directory'])/name.replace('/','_'))==h, 'CPU source snapshot changed')
    verify_bindings(plan['files']);verify_bindings(plan['input_bindings'])
    rows=read(plan['samples_file']);need(audit_rows(rows)==plan['symbolic_audit'] and rows==generate(), 'Plan symbolic law changed')
    need(tokenizer_contract()==read(plan['tokenizer_file']), 'Actual tokenizer/runtime changed')
    return plan


def runtime_view(record):
    view={k:record[k] for k in ('sid','n_frames','question','local_prompt')}
    view['image_files']=[{k:image[k] for k in ('path','sha256','dimensions','mode')} for image in record['image_files']]
    view['input_sha256']=oid(view)
    return view


def render_atom(entry,root,renderer):
    from PIL import Image
    person,room,step=entry['atom'];target=root/(oid(entry['atom'])+'.png')
    temporary=root/('.'+oid(entry['atom'])+'.png')
    need(not target.exists() and not temporary.exists(), 'Render attempt already exists')
    renderer.render_frame({room:[person]},step,str(temporary))
    rendered_sha=digest(temporary)
    with Image.open(temporary) as picture:
        need(picture.mode=='RGB' and picture.size==(512,512), 'Canonical render format differs')
        pixels=picture.tobytes();rgb_sha=hashlib.sha256(pixels).hexdigest()
    reference=entry['reference']
    if reference:
        source=Path(reference['path']);need(not source.is_symlink() and digest(source)==reference['sha256'], 'Reused image changed')
        with Image.open(source) as picture:
            need(picture.mode=='RGB' and picture.size==(512,512) and picture.tobytes()==pixels, 'Reused canonical RGB parity failed')
        os.link(source,target)
    else: os.link(temporary,target)
    temporary.unlink()
    return atom_key(entry['atom']),dict(atom=entry['atom'],path=str(target),sha256=digest(target),bytes=target.stat().st_size,
        dimensions=[512,512],mode='RGB',rgb_sha256=rgb_sha,rendered_sha256=rendered_sha,
        reused_reference=reference,reused=reference is not None,canonical_rgb_parity=True)


def publish(row,root,cache,targets):
    directory=root/'scenes'/row['sid'];directory.mkdir()
    qa='question:\n'+'\n'.join(repr(state) for state in row['states'])+'\n'+row['question']+'\nanswer:\n'+row['gold']+'\n'
    with (directory/'qa.txt').open('x') as stream:stream.write(qa)
    images=[]
    for i,(person,room) in enumerate(row['frames']):
        atom=cache[atom_key((person,room,i+1))];target=directory/f'{i:03d}.png'
        os.link(atom['path'],target)
        images.append(dict(path=str(target),sha256=atom['sha256'],dimensions=atom['dimensions'],mode=atom['mode']))
    result={k:v for k,v in row.items() if k not in ('frames','states')}
    result.update(path=str(directory),qa_sha256=digest(directory/'qa.txt'),image_files=images,
                  target_ids=targets[row['gold']]['target_ids'],origin='canonical_identity_join')
    return result


def audit_published(rows,records,cache,root):
    from PIL import Image
    for entry in cache.values():
        path=Path(entry['path']);need(path.resolve().is_relative_to(root.resolve()) and not path.is_symlink(), 'Image cache escaped attempt')
        need(digest(path)==entry['sha256'], 'Published PNG bytes differ')
        with Image.open(path) as picture:
            need(picture.mode=='RGB' and picture.size==(512,512) and hashlib.sha256(picture.tobytes()).hexdigest()==entry['rgb_sha256'], 'Published RGB pixels differ')
    need(len(rows)==len(records)==6588, 'Published context count differs')
    for row,record in zip(rows,records):
        directory=Path(record['path']);qa=directory/'qa.txt'
        need(directory.resolve().is_relative_to(root.resolve()) and not directory.is_symlink() and not qa.is_symlink(), 'Scene escaped attempt')
        need(digest(qa)==record['qa_sha256'], 'Published QA bytes differ')
        lines=qa.read_text().splitlines()
        need(lines[0]=='question:' and lines[-2]=='answer:' and lines[-1]==row['gold'] and lines[-3]==row['question'], 'Published native question/name differs')
        need([ast.literal_eval(line) for line in lines[1:-3]]==row['states'], 'Published symbolic states differ')
        need({k:record[k] for k in row if k not in ('frames','states')}=={k:v for k,v in row.items() if k not in ('frames','states')}, 'Manifest semantic row differs')
        need(len(record['image_files'])==row['n_frames'], 'Published image count differs')
        for i,((person,room),image) in enumerate(zip(row['frames'],record['image_files'])):
            path=directory/f'{i:03d}.png';entry=cache[atom_key((person,room,i+1))]
            need(not path.is_symlink() and image==dict(path=str(path),sha256=entry['sha256'],dimensions=[512,512],mode='RGB'), 'Image descriptor/order differs')
            need(os.path.samefile(path,entry['path']), 'Scene image is not the validated immutable canonical inode')
        view=runtime_view(record)
        need(set(view)=={'sid','n_frames','question','local_prompt','image_files','input_sha256'}, 'Runtime inputs carry semantic labels')
    return dict(passed=True,contexts=len(records),image_occurrences=sum(r['n_frames'] for r in rows),
                unique_pngs=len(cache),all_QA_states_names_verified=True,all_RGB_hashes_verified=True,
                all_image_order_inodes_verified=True,runtime_input_allowlist_verified=True)


def render(args,out,frozen):
    need(args.plan is not None and args.gate_report is not None, 'Render requires exact CPU plan and complete passed gate report')
    plan=verify_plan(args.plan)
    from scripts import probe_native_identity_join as gate
    gate_path=args.gate_report.resolve();gate_path=gate_path/'summary.json' if gate_path.is_dir() else gate_path
    analysis=gate.verify_report(gate_path)
    need(all(global_prompt(pair)==gate.global_prompt(pair) and local_prompt(global_prompt(pair))==gate.local_prompt(gate.global_prompt(pair)) for pair in PAIRS), 'Gate and dataset literal prompts differ')
    need(analysis['local_judgments']==4860 and analysis['head_rows']==4950 and analysis['feasibility_passed'] is True, 'Complete local feasibility required')
    gate_summary=read(gate_path)
    gate_plan=read(analysis['plan_file']);tokens=read(plan['tokenizer_file'])
    need(gate_plan['name_target_ids']=={name:row['target_ids'] for name,row in tokens['targets'].items()}
         and analysis['native_identity']['model']['path']==str(MODEL)
         and analysis['native_identity_sha256']==oid(analysis['native_identity']), 'Gate and dataset native name/tokenizer identity differs')
    gate_binding=dict(file=str(gate_path),sha256=digest(gate_path),analysis_file=gate_summary['analysis_file'],
        analysis_sha256=gate_summary['analysis_sha256'],source_sha256=analysis['source_sha256'],
        native_identity=analysis['native_identity'],native_identity_sha256=analysis['native_identity_sha256'],
        local_judgments=4860,included=1620,excluded=3240,head_rows=4950,full_batches=90,
        all_valid_correct=True,full_raw_artifact_hashes_verified=True)
    root=DATA_ROOT/f'stage_{os.environ["SLURM_JOB_ID"]}'
    need(root.resolve().is_relative_to(DATA_ROOT.resolve()) and not root.exists(), 'New authorized data attempt required')
    root.mkdir(parents=True,exist_ok=False)
    save(out/'render_release.json',dict(plan_file=str(args.plan.resolve()),plan_sha256=digest(args.plan),gate_report=gate_binding,data_root=str(root)))
    save(root/'attempt.json',dict(protocol=PROTOCOL,source_sha256=frozen,plan_file=str(args.plan.resolve()),plan_sha256=digest(args.plan),gate_report=gate_binding))
    rows=read(plan['samples_file']);inventory=read(plan['reuse_inventory_file']);tokens=read(plan['tokenizer_file'])
    renderer,provenance=prepare_renderer(inventory['renderer_references'])
    provenance['method']='Unchanged canonical RGB rendering; every reused atom independently rerendered and compared'
    (root/'render_cache').mkdir();(root/'scenes').mkdir()
    with ThreadPoolExecutor(max_workers=4) as pool:
        cache=dict(pool.map(lambda e:render_atom(e,root/'render_cache',renderer),inventory['entries']))
        save(root/'render_cache.json',cache)
        records=list(pool.map(lambda row:publish(row,root,cache,tokens['targets']),rows))
    actual=audit_published(rows,records,cache,root);symbolic=audit_rows(rows)
    views=[runtime_view(record) for record in records]
    save(root/'runtime_inputs.json',dict(schema_version=1,protocol=PROTOCOL,records=views,
        note='Only these prompts and ordered images enter preprocessing; QA, labels and symbolic metadata are offline'))
    save(root/'samples.json',rows);save(root/'audit.json',dict(symbolic=symbolic,published=actual,renderer=provenance))
    splits={}
    for cell,count in CELL_COUNTS.items():
        group=[r for r in records if f"{r['split']}_N{r['n_frames']}"==cell]
        need(len(group)==count, 'Final cell size differs')
        splits[cell]=dict(count=count,samples=group,answer_histogram=dict(Counter(r['gold'] for r in group)))
    manifest=dict(schema_version=1,protocol=PROTOCOL,purpose='identity_join',seed=SEED,policy=POLICY,
        dataset_root=str(root),splits=splits,source_sha256=frozen,plan_file=str(args.plan.resolve()),plan_sha256=digest(args.plan),
        gate_report=gate_binding,tokenizer=tokens,renderer=provenance,runtime_inputs_file=str(root/'runtime_inputs.json'),
        runtime_inputs_sha256=digest(root/'runtime_inputs.json'),samples_file=str(root/'samples.json'),samples_sha256=digest(root/'samples.json'),
        target_positions_per_split=plan['target_positions_per_split'],
        audit_file=str(root/'audit.json'),audit_sha256=digest(root/'audit.json'),render_cache_file=str(root/'render_cache.json'),
        render_cache_sha256=digest(root/'render_cache.json'),no_model_loaded=True,no_fit_released=True)
    save(root/'main_manifest.json',manifest)
    verify_bindings(plan['input_bindings'])
    need(digest(gate_path)==gate_binding['sha256'] and gate.sources()==gate_binding['source_sha256'], 'Gate evidence/source changed during rendering')
    return dict(passed=True,completed=True,check_only=False,source_sha256=frozen,
        manifest_file=str(root/'main_manifest.json'),manifest_sha256=digest(root/'main_manifest.json'),
        plan_file=str(args.plan.resolve()),plan_sha256=digest(args.plan),gate_report=gate_binding,
        dataset_root=str(root),counts=symbolic,published_audit=actual,distinct_atoms=len(cache),
        reused_atoms=sum(e['reused'] for e in cache.values()),new_atoms=sum(not e['reused'] for e in cache.values()),
        input_bindings=plan['input_bindings'],no_model_loaded=True,no_fit_released=True)


def verify_stage(path):
    path=Path(path).resolve();path=path/'summary.json' if path.is_dir() else path
    summary=read(path)
    need(summary['passed'] is True and summary['completed'] is True and summary['check_only'] is False, 'Complete rendered-stage proof required')
    need(summary['source_sha256']==source_hashes(), 'Current dataset sources differ')
    for name,h in summary['source_sha256'].items():
        need(digest(path.parent/'source'/name.replace('/','_'))==h, 'Stage source snapshot differs')
    need(digest(Path(summary['manifest_file']))==summary['manifest_sha256'], 'Published manifest changed')
    manifest=read(summary['manifest_file'])
    need(manifest['source_sha256']==summary['source_sha256'] and manifest['gate_report']==summary['gate_report'], 'Manifest provenance differs')
    need(manifest['plan_file']==summary['plan_file'] and manifest['plan_sha256']==summary['plan_sha256']
         and digest(Path(manifest['plan_file']))==manifest['plan_sha256'], 'Bound CPU plan changed')
    need(read(manifest['plan_file'])['source_sha256']==summary['source_sha256'], 'Bound CPU plan source differs')
    binding=manifest['gate_report']
    need(digest(Path(binding['file']))==binding['sha256']
         and digest(Path(binding['analysis_file']))==binding['analysis_sha256'], 'Bound complete gate report changed')
    gate_summary=read(binding['file']);gate_analysis=read(binding['analysis_file'])
    from scripts import probe_native_identity_join as gate
    need(gate_summary['analysis_file']==binding['analysis_file'] and gate_summary['analysis_sha256']==binding['analysis_sha256']
         and gate_summary['source_sha256']==gate_analysis['source_sha256']==binding['source_sha256']==gate.sources(),
         'Bound complete gate report source differs')
    for key in ('samples','runtime_inputs','audit','render_cache'):
        need(digest(Path(manifest[key+'_file']))==manifest[key+'_sha256'], 'Published '+key+' changed')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check',action='store_true');group.add_argument('--render',action='store_true')
    parser.add_argument('--plan',type=Path);parser.add_argument('--gate-report',type=Path)
    args=parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu'
         and not os.environ.get('SLURM_JOB_GPUS') and int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==4,
         'Only a Slurm CPU allocation with4cores may stage join data')
    need(not args.check or (args.plan is None and args.gate_report is None), 'Check has no mutable plan/gate arguments')
    mode='check' if args.check else 'render';out=OUT/f'{mode}_{os.environ["SLURM_JOB_ID"]}'
    out.mkdir(parents=True,exist_ok=False);frozen=snapshot(out);start=time.perf_counter()
    save(out/'request.json',dict(mode=mode,plan=str(args.plan) if args.plan else None,
        gate_report=str(args.gate_report) if args.gate_report else None,source_sha256=frozen))
    try:
        result=check(out,frozen) if args.check else render(args,out,frozen)
        need(source_hashes()==frozen, 'Sources changed during staging')
        result.update(seconds=time.perf_counter()-start,slurm_job_id=os.environ['SLURM_JOB_ID'])
        save(out/'summary.json',result)
        with (out/'REPORT.md').open('x') as stream:
            stream.write('# Identity-join data preparation\n\n'+('Symbolic/tokenizer check passed; rendering still requires the complete local feasibility audit.' if args.check else 'Canonical rendering and independent data audit passed. No fit is released.')+'\n\n'+json.dumps(result['counts'],indent=2)+'\n')
        print(json.dumps(dict(passed=True,directory=str(out))),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,
            seconds=time.perf_counter()-start,partial_outputs_retained=True))
        raise


if __name__=='__main__':
    main()

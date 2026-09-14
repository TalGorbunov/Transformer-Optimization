"""Build training-only privileged targets; all tensor work is CPU Slurm-only."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
PROTOCOL = 'mmred_prefix_supervision_preparation'
OUT = REPO / 'outputs/native_aggregation_vlm/mmred_prefix_supervision'
DATA = Path('/mnt/data/gabriele/gnn_transformer/mmred_prefix_supervision')
TRAINING_PLAN = REPO / 'outputs/native_aggregation_vlm/mmred_official_native_training/check_444010/plan.json'
TRAINING_PLAN_SHA = '94b0c0eeed2130e256c63a5a4777599562b5be4f9b97b79e6356d21892d0b368'
MODEL_CONFIG = Path('/mnt/ckpts/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5/config.json')
DESIGN = 'docs/paper/NATIVE_AGGREGATION_PRIVILEGED_TRAINING_RESEARCH_DESIGN.md'
OWN = ('gnnformer/native_prefix_supervision.py', 'tests/test_native_prefix_supervision.py',
       'scripts/prepare_mmred_prefix_supervision.py', 'slurm/mmred_prefix_supervision_prepare.sbatch')
POLICY = dict(protocol=PROTOCOL, cpu_seconds=600, cpu_cores=4, memory_gib=16, training_worlds=4000,
              channels=40, hidden_size=3584, projection_seed=20260914, trainable_auxiliary_parameters=0,
              hidden_normalization='none', loss='FP32 mean squared error over N times40',
              normalization='equal_world_then_equal_prefix_population', variance_floor=1e-12,
              image_end_token_id=151653, image_start_token_id=151652, image_token_id=151655,
              no_model_calls=True, no_validation_or_test=True, no_training_release=True)


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, sort_keys=True, allow_nan=False, indent=2)


def descriptor(path):
    return dict(file=str(Path(path).resolve()), sha256=sha(path))


def sources():
    return {name: sha(REPO / name) for name in OWN}


def inherited_sources():
    need(sha(TRAINING_PLAN) == TRAINING_PLAN_SHA, 'Pinned training plan changed')
    parent = read(TRAINING_PLAN)
    result = {}
    for field in ('source_sha256', 'inherited_source_sha256'):
        for name, digest in parent[field].items():
            need(name not in result or result[name] == digest, 'Conflicting source ancestry')
            result[name] = digest
    for name in ('gnnformer/mmred_hf.py', DESIGN):
        digest = sha(REPO / name)
        need(name not in result or result[name] == digest, 'Imported oracle/design source changed')
        result[name] = digest
    need(all(sha(REPO / name) == digest for name, digest in result.items()), 'Frozen ancestor source changed')
    return result


def tensor_info(value):
    import torch
    raw = value.detach().to('cpu').contiguous()
    return dict(shape=list(raw.shape), dtype=str(raw.dtype),
                sha256=hashlib.sha256(raw.view(torch.uint8).numpy().tobytes()).hexdigest())


def image_end_positions(torch, case, n):
    """Locate existing physical image-end rows; never insert an oracle token."""
    prompt = case['inputs']['input_ids']
    teacher = case['teacher_input_ids']
    width = case['metadata']['prompt_width']
    target = case['target_ids']
    need(prompt.dtype == teacher.dtype == torch.int64 and prompt.shape == (1, width)
         and teacher.shape == (1, width + len(target) - 1)
         and torch.equal(teacher[:, :width], prompt)
         and teacher[0, width:].tolist() == target[:-1]
         and case['target_positions'] == list(range(width - 1, width + len(target) - 1)),
         'Native strict teacher prefix/target ownership differs')
    ids = prompt[0]
    starts = (ids == 151652).nonzero().flatten()
    ends = (ids == 151653).nonzero().flatten()
    need(starts.numel() == ends.numel() == n and bool((ends < width - 1).all())
         and int((ids == 151655).sum()) == 196 * n, 'Exactly N existing image blocks before the question required')
    previous = -1
    for start, end in zip(starts.tolist(), ends.tolist()):
        need(previous < start < end and end - start == 197
             and bool((ids[start + 1:end] == 151655).all()), 'Each original image must contain196 native merged tokens')
        previous = end
    coordinates = case['coordinates']
    need(torch.equal(coordinates['frame_index'], torch.arange(n, dtype=torch.int64).repeat_interleave(196)),
         'Image block and original frame coordinates differ')
    return ends.to(dtype=torch.int64, device='cpu')


def position_fixtures(torch):
    ids = [12, 151652] + [151655] * 196 + [151653, 13, 14]
    prompt = torch.tensor([ids], dtype=torch.int64)
    case = dict(inputs=dict(input_ids=prompt), teacher_input_ids=torch.cat((prompt, torch.tensor([[17]])), 1),
                metadata=dict(prompt_width=len(ids)), target_ids=[17, 151645],
                target_positions=[len(ids)-1, len(ids)], coordinates=dict(frame_index=torch.zeros(196, dtype=torch.int64)))
    need(image_end_positions(torch, case, 1).tolist() == [198], 'Image-end location fixture failed')
    bad = dict(case, teacher_input_ids=case['teacher_input_ids'].clone())
    bad['teacher_input_ids'][0, -1] = 18
    try:
        image_end_positions(torch, bad, 1)
    except ValueError:
        pass
    else:
        raise ValueError('Future/changed teacher token was accepted')
    bad = dict(case, coordinates=dict(frame_index=torch.ones(196, dtype=torch.int64)))
    try:
        image_end_positions(torch, bad, 1)
    except ValueError:
        pass
    else:
        raise ValueError('Wrong frame ownership was accepted')
    return dict(passed=True, accepted=1, rejected=2)


def verify_stage(summary_path):
    """JSON/hash handoff, no imports of torch and no model or tensor execution.

    Raw compact packets were checked by the completed preparation. This handoff
    rehashes its output artifacts, binding manifest, source files and parent plan;
    it does not rescan4000 old compact packets or open their tensors.
    """
    summary = read(summary_path)
    need(summary['protocol'] == PROTOCOL and summary['passed'] is True and summary['completed'] is True
         and not (Path(summary_path).parent / 'failure.json').exists(), 'Passed complete target preparation required')
    need(sha(summary['plan_file']) == summary['plan_sha256'], 'Target plan changed')
    plan = read(summary['plan_file'])
    need(plan['protocol'] == PROTOCOL and plan['policy'] == POLICY and plan['rows'] == 4000
         and plan['all_original_oracles_verified'] is True and plan['raw_inputs_verified_in_preparation'] is True
         and plan['raw_inputs_rehashed_in_handoff'] is False, 'Target plan scope differs')
    need(plan['training_plan'] == dict(file=str(TRAINING_PLAN), sha256=TRAINING_PLAN_SHA)
         and sha(TRAINING_PLAN) == TRAINING_PLAN_SHA, 'Target training ancestry changed')
    for field in ('projection', 'targets', 'target_index', 'input_bindings', 'tests', 'model_config'):
        ref = plan[field]
        need(sha(ref['file']) == ref['sha256'], 'Preparation artifact changed: ' + field)
    for name, digest in {**plan['source_sha256'], **plan['inherited_source_sha256']}.items():
        need(sha(REPO / name) == digest, 'Preparation source changed: ' + name)
    return plan


def run(out, data, started, progress):
    import torch
    from gnnformer import native_prefix_supervision as core
    from gnnformer.mmred_hf import recompute_answer
    from scripts import train_mmred_official_native_memory as original_training
    torch.set_num_threads(4)
    own = sources(); inherited = inherited_sources()
    parent = original_training.verify_plan(TRAINING_PLAN)
    bindings = {}
    def bind(path, expected=None):
        path = str(Path(path).resolve()); actual = sha(path)
        need(expected is None or actual == expected, 'Input bytes changed: ' + path)
        need(path not in bindings or bindings[path] == actual, 'Input changed within preparation')
        bindings[path] = actual
        return path
    bind(TRAINING_PLAN, TRAINING_PLAN_SHA)
    bind(parent['cases_file'], parent['artifacts'][parent['cases_file']])
    config = read(bind(MODEL_CONFIG))
    need((config['vision_start_token_id'], config['vision_end_token_id'], config['image_token_id'])
         == (151652, 151653, 151655), 'Native image token configuration changed')
    items = read(parent['cases_file'])
    need(len(items) == 4000, 'Exactly original4000 training examples required')
    spec = importlib.util.spec_from_file_location('prefix_supervision_cpu_fixtures', REPO / OWN[1])
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    tests = module.run_tests(); tests['image_positions'] = position_fixtures(torch)
    save(out / 'tests.json', tests)
    need(tests['passed'] and tests['tests_run'] == 7, 'Prefix supervision CPU fixtures failed')
    projection = core.build_projection()
    records = []; index_rows = []; renderer_cache = {}; counts = Counter(); seen_sids = set()
    for index, item in enumerate(items):
        need(time.perf_counter() - started < 570, 'Preparation reserve reached')
        row = item['row']; n = row['n']; sid = row['sid']
        need(row['split'] == row['pilot_role'] == 'train' and n in (1,2,4,8,16)
             and sid not in seen_sids, 'Unique original training-only source row required')
        seen_sids.add(sid); counts[(n, row['qtype'])] += 1
        renderer = row['renderer_file']
        if renderer not in renderer_cache:
            bind(renderer); renderer_cache[renderer] = read(renderer)
        raw = renderer_cache[renderer][row['renderer_row']]
        need(object_sha(raw['sequence']) == row['exact_sequence_sha256']
             and raw['qid'] == row['qid'], 'Frozen original world/question identity differs')
        need(raw['seq_len'] == n and len(raw['sequence']) == n
             and all(raw[key] == row[key] for key in ('qtype','atype','question'))
             and str(raw['answer']) == str(row['answer']), 'Renderer/source semantic ownership differs')
        need(str(recompute_answer(raw['qtype'], raw['question'], raw['sequence'])) == str(raw['answer']),
             'Original question oracle disagrees with frozen answer')
        values = core.world_targets(raw['sequence'])
        canonical_world = [[core.ROOMS.index(next(room for room, persons in state['rooms'].items() if person in persons))
                            for person in core.PEOPLE] for state in raw['sequence']]
        need(object_sha(canonical_world) == row['world_sha256'], 'Canonical original world identity differs')
        compact = item['compact']
        bind(compact['file'], compact['sha256'])
        packet = torch.load(compact['file'], map_location='cpu', weights_only=True)
        case = packet['case']
        need(case['metadata']['sid'] == sid and case['metadata']['n'] == n
             and case['metadata']['question'] == row['question'], 'Compact native case belongs to another world')
        positions = image_end_positions(torch, case, n)
        local = torch.tensor(values['local'], dtype=torch.int64)
        prefix = torch.tensor(values['prefix'], dtype=torch.int64)
        need(torch.equal(local.cumsum(0), prefix) and bool((prefix >= 0).all()), 'Causal prefix sum differs')
        records.append(dict(index=index, sid=sid, n=n, local=local, prefix=prefix, image_end_positions=positions))
        index_rows.append(dict(index=index, sid=sid, n=n, qtype=row['qtype'],
                               renderer_file=renderer, renderer_row=row['renderer_row'], raw_source_sha256=object_sha(raw),
                               compact=compact, features=item['features'], image_end_positions=positions.tolist(),
                               tensors={key:tensor_info(records[-1][key]) for key in ('local','prefix','image_end_positions')}))
        progress.update(completed_worlds=index+1, original_oracle_checks=index+1)
        del packet, case, values
    need(counts == Counter({(n,q):200 for n in (1,2,4,8,16)
                           for q in ('char_at_frame','steps_in_room','spend_together','where_spend')}),
         'Original balanced4000 training population changed')
    normalization = {mode:core.population_normalization(record[mode] for record in records) for mode in ('local','prefix')}
    projection_packet = dict(projection=projection, normalization=normalization, channel_names=list(core.CHANNEL_NAMES),
                             people=list(core.PEOPLE), rooms=list(core.ROOMS), pairs=[list(x) for x in core.PAIRS],
                             seed=core.PROJECTION_SEED, projection_algorithm='CPU FP64 Gaussian reduced QR; positive R diagonal; FP32 transpose',
                             raw_hidden_projection=True, trainable_parameters=0)
    torch.save(projection_packet, data / 'projection.pt')
    torch.save(dict(records=records), data / 'targets.pt')
    save(data / 'target_index.json', index_rows)
    save(data / 'input_bindings.json', bindings)
    plan = dict(protocol=PROTOCOL, policy=POLICY, rows=4000, total_prefixes=sum(r['n'] for r in records),
                training_plan=dict(file=str(TRAINING_PLAN), sha256=TRAINING_PLAN_SHA),
                cases_file=parent['cases_file'], cases_sha256=bindings[str(Path(parent['cases_file']).resolve())],
                native_identity_sha256=parent['native_identity_sha256'], model_config=descriptor(MODEL_CONFIG),
                projection=descriptor(data/'projection.pt'), targets=descriptor(data/'targets.pt'),
                target_index=descriptor(data/'target_index.json'), input_bindings=descriptor(data/'input_bindings.json'),
                projection_file=str(data/'projection.pt'), targets_file=str(data/'targets.pt'), target_index_file=str(data/'target_index.json'),
                tests=descriptor(out/'tests.json'), source_sha256=own, inherited_source_sha256=inherited,
                all_original_oracles_verified=True, original_oracle_checks=4000,
                raw_inputs_verified_in_preparation=True, raw_inputs_rehashed_in_handoff=False,
                no_model_calls=True, no_training_release=True, elapsed_seconds=time.perf_counter()-started)
    need(sources() == own and inherited_sources() == inherited, 'Source bytes changed during preparation')
    save(out/'plan.json', plan)
    save(out/'summary.json', dict(protocol=PROTOCOL, passed=True, completed=True,
                                 plan_file=str(out/'plan.json'), plan_sha256=sha(out/'plan.json'),
                                 rows=4000, total_prefixes=plan['total_prefixes'], elapsed_seconds=time.perf_counter()-started,
                                 no_training_release=True))


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION') == 'cpu'
         and not os.environ.get('SLURM_JOB_GPUS'), 'CPU Slurm-only target preparation')
    name = 'prepare_' + os.environ['SLURM_JOB_ID']
    out = OUT/name; data = DATA/name
    out.mkdir(parents=True, exist_ok=False); data.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter(); progress = {}
    save(out/'request.json', dict(protocol=PROTOCOL, policy=POLICY, training_plan=str(TRAINING_PLAN), data_directory=str(data)))
    try:
        run(out, data, started, progress)
    except BaseException as exc:
        save(out/'failure.json', dict(protocol=PROTOCOL, passed=False, type=type(exc).__name__, message=str(exc),
                                     progress=progress, elapsed_seconds=time.perf_counter()-started))
        raise


if __name__ == '__main__':
    main()

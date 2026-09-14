"""Training-only descriptive geometry of the frozen uniform join endpoint.

The CPU report supplies torch to analyze(torch, out). No model/head execution,
reader, intervention, outcome rescore, or development/test capture is used.
All numerical recomputations are descriptive; the previous native audit remains
the fidelity evidence. This file intentionally has no executable entry point.
"""
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]
OWN = ('scripts/analyze_native_identity_join_uniform_geometry.py',)
REPORT = ROOT / 'outputs/native_aggregation_vlm/identity_join_uniform/report_443271'
SUMMARY_SHA = '1be7e5c8e5c598efd71d996035985e6f8f797dd749e977f4ac7817fc0df424f0'
ANALYSIS_SHA = 'c2176613b6b1ade6eb4c5a13e7ddfe23cb523b7d7709907a312f834bb5a07056'
PLAN_SHA = 'f87d6d49bab87e3baacb436faed88b6561a1dcd3a71b7ccefdebafe2d79c12d9'
CHECKPOINT_SHA = 'e37197674703128af5ae39ac987b33256fd1c343dbacfb36ae3139b5f07712a3'
SATURATION_THRESHOLD = .01


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def bind(path, digest, ledger):
    path = Path(path).resolve()
    need(sha(path) == digest, 'Bound geometry input changed: ' + str(path))
    need(str(path) not in ledger or ledger[str(path)] == digest, 'Conflicting input binding')
    ledger[str(path)] = digest
    return path


def tensor_info(torch, value):
    value = value.detach().cpu().contiguous()
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest())


def tensor(torch, value, shape, dtype):
    need(isinstance(value, torch.Tensor) and tuple(value.shape) == tuple(shape)
         and value.dtype == dtype and value.device.type == 'cpu'
         and not value.requires_grad and bool(torch.isfinite(value).all()),
         'Malformed CPU geometry tensor')
    return value


def rms(torch, value):
    value = value.float()
    return value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)


def norm(torch, value):
    return float(torch.linalg.vector_norm(value.double()))


def difference(torch, left, right):
    need(left.shape == right.shape, 'Contrast shape differs')
    delta = left.double() - right.double()
    return dict(exact=bool(torch.equal(left, right)), l2=norm(torch, delta),
                rms=float(delta.square().mean().sqrt()), maximum_absolute=float(delta.abs().max()))


def rows_metric(torch, value):
    value = value.double()
    need(value.ndim == 2 and value.shape[0] > 0, 'Expected nonempty occurrence matrix')
    row_norm = torch.linalg.vector_norm(value, dim=-1)
    return dict(frobenius=norm(torch, value), coordinate_rms=float(value.square().mean().sqrt()),
                mean_row_norm=float(row_norm.mean()), row_norms=row_norm.tolist())


def decomposition(torch, x, mu, query, weights):
    """FP32 core-order recomputation with the captured query as the reference."""
    linear = torch.nn.functional.linear
    a = linear(x, weights['local.weight']) + query + weights['local_bias']
    common = linear(mu.unsqueeze(0), weights['local.weight']) + query + weights['local_bias']
    deviation = linear(x - mu, weights['local.weight'])
    return a, common, deviation


def contrast(torch, left, right, aggregate_reference='saved native FP32 aggregates'):
    """Signed right-minus-left physical-slot contrasts, before/after .5 SUM."""
    dx = right['x'].double() - left['x'].double()
    da = right['a'].double() - left['a'].double()
    dp = right['p'].double() - left['p'].double()
    pooled = .5 * dp.sum(0)
    row_total = float(torch.linalg.vector_norm(.5 * dp, dim=-1).sum())
    return dict(rms_local=rows_metric(torch, dx), local_preactivation=rows_metric(torch, da),
                captured_payload=rows_metric(torch, dp), half_sum_payload_delta=pooled.tolist(),
                half_sum_payload_delta_norm=norm(torch, pooled),
                sum_item_delta_norms=row_total,
                pooling_cancellation_ratio=norm(torch, pooled) / row_total if row_total else None,
                aggregate_reference=aggregate_reference,
                aggregate_reference_delta=(right['z'].double() - left['z'].double()).tolist(),
                aggregate_reduction_difference=difference(torch, pooled, right['z'].double() - left['z'].double()),
                captured_query_difference=difference(torch, right['q'], left['q']))


def self_test(torch):
    """Four small algebra fixtures, run only by the Slurm CPU report."""
    x = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
    mu = x.double().mean(0).float()
    w = {'local.weight': torch.tensor([[2., 0.], [0., -1.]]), 'local_bias': torch.tensor([1., 2.])}
    a, common, deviation = decomposition(torch, x, mu, torch.tensor([[.5, -.5]]), w)
    need(torch.equal(a, common + deviation), 'Common/deviation fixture failed')
    unequal = torch.cat((torch.zeros(1, 2), torch.ones(3, 2)), 0).double().mean(0).float()
    need(torch.equal(unequal, torch.full((2,), .75)), 'Occurrence weighting fixture failed')
    p = torch.tensor([[1., 0.], [-1., 0.]])
    left = dict(x=p, a=p, p=p, q=torch.zeros(1, 2), z=.5 * p.sum(0))
    right = dict(x=2*p, a=2*p, p=2*p, q=torch.zeros(1, 2), z=p.sum(0))
    result = contrast(torch, left, right)
    need(result['half_sum_payload_delta_norm'] == 0 and result['sum_item_delta_norms'] == 1
         and result['pooling_cancellation_ratio'] == 0, 'Signed additive contrast fixture failed')
    inserted = torch.tensor([[2., 1.], [0., -1.]])
    child = torch.stack((p[0], inserted[0], p[1], inserted[1]))
    need(torch.equal(.5 * (child.sum(0) - p.sum(0)), .5 * inserted.sum(0))
         and (1 - torch.tensor([0., 1., -1.]).double().square() <= SATURATION_THRESHOLD).tolist()
         == [False, True, True], 'Insertion/saturation fixture failed')
    return dict(passed=True, checks=4, scope='small exact geometry algebra; no fitted tensors')


def inputs(torch):
    ledger = {}
    summary = read(bind(REPORT / 'summary.json', SUMMARY_SHA, ledger))
    analysis = read(bind(REPORT / 'analysis.json', ANALYSIS_SHA, ledger))
    need(summary['passed'] is True and summary['completed'] is True
         and analysis['passed'] is True and analysis['completed'] is True
         and summary['analysis_sha256'] == ANALYSIS_SHA
         and Path(summary['analysis_file']).resolve() == (REPORT / 'analysis.json').resolve(), 'Uniform report is incomplete')
    need(len(analysis['runs']) == 1 and analysis['runs'][0]['arm'] == 'uniform_half_ce'
         and analysis['runs'][0]['trainability']['passed'] is False, 'Uniform training failure trigger absent')
    need(analysis['plan_sha256'] == PLAN_SHA, 'Different uniform plan')
    plan = read(bind(analysis['plan_file'], PLAN_SHA, ledger))
    need(summary['source_sha256'] == analysis['source_sha256'] == plan['source_sha256'], 'Uniform source union differs')
    for name, digest in summary['source_sha256'].items():
        bind(ROOT / name, digest, ledger)
        bind(REPORT / 'source' / name.replace('/', '_'), digest, ledger)
    for key in ('rows_file', 'pairs_file', 'semantic_states_file'):
        bind(plan[key], plan['runtime_bindings'][plan[key]], ledger)
    # The published outcome JSON contains train/dev rows; only train is selected.
    outcomes = read(bind(analysis['outcomes_file'], analysis['outcomes_sha256'], ledger))
    train = outcomes['uniform_half_ce']['train']
    rows = read(plan['rows_file'])['train']
    pairs = read(plan['pairs_file'])
    all_states = read(plan['semantic_states_file'])
    samples = {row['sample']['sid']: row['sample'] for row in rows}
    need(len(samples) == len(rows) == len(train) == 108 and len(pairs) == 54
         and len({r['sid'] for r in train}) == 108 and {r['sid'] for r in train} == set(samples), 'Training capture inventory differs')
    need(all(r['cell'] == 'train_N' + str(r['sample']['n_frames'])
             and r['sample']['split'] == 'train' for r in rows), 'Nontraining scene in geometry')
    states = {sid: all_states[sid] for sid in samples}
    run_ref = analysis['runs'][0]
    run = read(bind(Path(run_ref['directory']) / 'summary.json', run_ref['summary_sha256'], ledger))
    need(run['passed'] is True and run['completed'] is True and run['steps'] == 600
         and run['checkpoint'] == run_ref['checkpoint']
         and run['checkpoint_sha256'] == run_ref['checkpoint_sha256'] == CHECKPOINT_SHA
         and run['plan_sha256'] == PLAN_SHA and run['source_sha256'] == summary['source_sha256'], 'Uniform endpoint join differs')
    endpoint = read(bind(run['final_endpoint_file'], run['final_endpoint_sha256'], ledger))
    checkpoint = bind(run['checkpoint'], CHECKPOINT_SHA, ledger)
    packet = torch.load(checkpoint, map_location='cpu', weights_only=True)
    need(set(packet) == {'branch', 'step', 'config'} and packet['step'] == 600
         and all(run[k] == v for k, v in packet['config'].items()), 'Saved final checkpoint/config differs')
    weights = packet['branch']
    shapes = {'local_bias': (96,), 'selection_weight': (96,), 'selection_bias': (1,),
              'query.weight': (96, 3584), 'local.weight': (96, 3584),
              'aggregate_projection.weight': (96, 96), 'aggregate_projection.bias': (96,), 'up.weight': (3584, 96)}
    need(set(weights) == set(shapes), 'Uniform parameter fields differ')
    table = {key: tensor_info(torch, tensor(torch, weights[key], shape, torch.float32)) for key, shape in shapes.items()}
    need(table == endpoint['before_evaluation'] == endpoint['after_evaluation']
         and object_sha(table) == run['selected_parameter_sha256'] == endpoint['parameter_sha256']
         and bool((weights['selection_weight'] == 0).all()) and bool((weights['selection_bias'] == .5).all()), 'Frozen endpoint/selector differs')
    return plan, train, samples, pairs, states, weights, ledger, table


def load_scene(torch, row, sample, plan, ledger):
    n = sample['n_frames']
    need(n in (8, 16) and all(row[k] == sample[k] for k in ('sid', 'n_frames', 'contrast_id', 'variant', 'pair_id', 'content_sha256')), 'Scene/outcome ownership differs')
    path = bind(row['raw_file'], row['raw_sha256'], ledger)
    need(path.parent.name == 'train_final', 'Nontraining raw capture requested')
    raw = torch.load(path, map_location='cpu', weights_only=True)
    meta = raw['metadata']
    need(raw['schema_version'] == 1 and raw['selection_mode'] == 'sigmoid'
         and meta == row['metadata'] and meta['sid'] == sample['sid'] and meta['n_frames'] == n
         and meta['question'] == sample['question'] == sample['local_prompt']
         and meta['prefix_ids'] == [] and meta['native_identity_sha256'] == plan['native_identity_sha256'], 'First-query/native input identity differs')
    images = sample['image_files']
    need(len(images) == n and meta['image_paths'] == [x['path'] for x in images]
         and meta['image_sha256'] == [x['sha256'] for x in images], 'Original image order differs')
    for image in images:
        if str(Path(image['path']).resolve()) not in ledger:
            bind(image['path'], image['sha256'], ledger)
        else:
            need(ledger[str(Path(image['path']).resolve())] == image['sha256'], 'Conflicting image identity')
    need(bool(raw['captures']), 'Missing first query')
    cap = raw['captures'][0]
    h = tensor(torch, cap['local_states'], (n, 1, 3584), torch.float16)[:, 0].clone()
    g = tensor(torch, cap['global_states'], (1, 3584), torch.float16).clone()
    p = tensor(torch, cap['payload'], (n, 1, 96), torch.float32)[:, 0].clone()
    q = tensor(torch, cap['query'], (1, 96), torch.float32).clone()
    z = tensor(torch, cap['aggregate'], (1, 96), torch.float32)[0].clone()
    gates = tensor(torch, cap['gates'], (n, 1), torch.float32)
    scores = tensor(torch, cap['scores'], (n, 1), torch.float32)
    messages = tensor(torch, cap['messages'], (n, 1, 96), torch.float32)
    need(bool((gates == .5).all()) and bool((scores == .5).all())
         and torch.equal(messages[:, 0], .5 * p) and bool((p.abs() <= 1).all()), 'Captured fixed-uniform arithmetic differs')
    capture_binding = {k: tensor_info(torch, cap[k]) for k in ('local_states', 'global_states', 'query', 'payload', 'scores', 'gates', 'messages', 'aggregate')}
    # Discard the container, all logits and every later prefix immediately.
    del cap, raw
    return dict(x=rms(torch, h), p=p, q=q, z=z, g=g, capture_binding=capture_binding)


def analyze(torch, out):
    """Return JSON-compatible geometry + file ledger; caller binds this result.

    Exclusive JSON writes prevent replacement. Only inherited training input
    provenance uses semantic states; no truth enters any tensor calculation.
    """
    need(os.environ.get('SLURM_JOB_ID') and not os.environ.get('SLURM_JOB_GPUS')
         and int(os.environ.get('SLURM_GPUS_ON_NODE', '0')) == 0, 'Geometry requires a CPU Slurm allocation')
    out = Path(out).resolve()
    need(out.is_dir(), 'The independent report must create its dedicated output directory')
    with torch.no_grad(), torch.autocast(device_type='cpu', enabled=False):
        fixtures = self_test(torch)
        plan, rows, samples, pairs, states, weights, ledger, table = inputs(torch)
        scene = {row['sid']: load_scene(torch, row, samples[row['sid']], plan, ledger) for row in rows}
        by_question, by_family = defaultdict(list), defaultdict(list)
        for sid, sample in samples.items():
            by_question[sample['question']].append(sid)
            by_family[(sample['contrast_id'], sample['n_frames'])].append(sid)
        need(len(by_question) == 6 and len(by_family) == 36
             and Counter(len(v) for v in by_family.values()) == {3: 36}
             and len({f for f, _ in by_family}) == 18
             and all({n for f, n in by_family if f == family} == {8, 16} for family, _ in by_family)
             and sum(samples[s]['n_frames'] for s in samples) == 1296, 'Question/family/occurrence coverage differs')
        question_ids = {q: i for i, q in enumerate(sorted(by_question))}
        question_rows, scene_rows = [], []
        for question in sorted(by_question):
            sids = sorted(by_question[question])
            x = torch.cat([scene[s]['x'] for s in sids], 0)
            need(len(sids) == 18 and x.shape == (216, 3584), 'Question occurrence weighting differs')
            mean64 = x.double().mean(0)
            mu = mean64.float()
            all_a, all_p, all_d, all_common, all_deviation = [], [], [], [], []
            for sid in sids:
                sample, current = samples[sid], scene[sid]
                a, common, deviation = decomposition(torch, current['x'], mu, current['q'], weights)
                current['a'] = a
                d = 1 - current['p'].double().square()
                common_expanded = common.expand_as(a)
                all_a.append(a); all_p.append(current['p']); all_d.append(d)
                all_common.append(common_expanded); all_deviation.append(deviation)
                scene_rows.append(dict(labels=dict(sid=sid, question_id=question_ids[question],
                    contrast_id=sample['contrast_id'], variant=sample['variant'], n_frames=sample['n_frames'], pair_id=sample['pair_id']),
                    raw=dict(file=next(r['raw_file'] for r in rows if r['sid'] == sid),
                             sha256=next(r['raw_sha256'] for r in rows if r['sid'] == sid), prefix_index=0, observed_prefix_ids=[],
                             tensors=current['capture_binding']),
                    metrics=dict(local_preactivation=rows_metric(torch, a), common=rows_metric(torch, common_expanded),
                        common_vector=common[0].tolist(), captured_query_norm=norm(torch, current['q']),
                        deviation=rows_metric(torch, deviation), captured_payload=rows_metric(torch, current['p']),
                        mean_derivative_per_item=d.mean(-1).tolist(), saturation_fraction_per_item=(d <= SATURATION_THRESHOLD).double().mean(-1).tolist(),
                        query_recompute_difference=difference(torch, torch.nn.functional.linear(rms(torch, current['g']), weights['query.weight']), current['q']),
                        split_reconstruction_difference=difference(torch, common_expanded + deviation, a),
                        payload_recompute_difference=difference(torch, a.tanh(), current['p']),
                        aggregate_reduction_difference=difference(torch, .5 * current['p'].double().sum(0), current['z']))))
            a, p, d = torch.cat(all_a).double(), torch.cat(all_p).double(), torch.cat(all_d)
            common, deviation = torch.cat(all_common), torch.cat(all_deviation)
            question_rows.append(dict(labels=dict(question_id=question_ids[question], question=question, sids=sids),
                metrics=dict(contexts=len(sids), occurrences=x.shape[0], rms_local_mean=mu.tolist(),
                    projected_rms_local_mean=torch.nn.functional.linear(mu, weights['local.weight']).tolist(),
                    local_bias=weights['local_bias'].tolist(),
                    mean_cast_difference=difference(torch, mu.double(), mean64),
                    rms_local_population_variance=x.double().var(0, correction=0).tolist(),
                    local_preactivation_population_variance=a.var(0, correction=0).tolist(),
                    payload_population_variance=p.var(0, correction=0).tolist(),
                    mean_derivative_per_coordinate=d.mean(0).tolist(),
                    saturation_fraction_per_coordinate=(d <= SATURATION_THRESHOLD).double().mean(0).tolist(),
                    mean_derivative=float(d.mean()), saturation_fraction=float((d <= SATURATION_THRESHOLD).double().mean()),
                    local_preactivation=rows_metric(torch, a), common=rows_metric(torch, common), deviation=rows_metric(torch, deviation))))
        variants = []
        for (family, n), sids in sorted(by_family.items()):
            sids.sort(key=lambda s: samples[s]['variant'])
            need([samples[s]['variant'] for s in sids] == [0, 1, 2]
                 and len({samples[s]['question'] for s in sids}) == 1
                 and all(samples[s]['parent_positions'] == samples[sids[0]]['parent_positions'] for s in sids), 'Variant/slot coverage differs')
            for left, right in combinations(sids, 2):
                # The original family preserves person order; rooms can change.
                person = lambda frame: next(iter(frame['rooms'].values()))
                need(len(states[left]) == len(states[right]) == n
                     and all(person(a) == person(b) for a, b in zip(states[left], states[right])), 'Original person-slot alignment differs')
                variants.append(dict(labels=dict(contrast_id=family, n_frames=n, left_sid=left, right_sid=right,
                    left_variant=samples[left]['variant'], right_variant=samples[right]['variant'], question_id=question_ids[samples[left]['question']]),
                    provenance=dict(physical_steps=list(range(1, n+1)),
                        left_image_sha256=[v['sha256'] for v in samples[left]['image_files']],
                        right_image_sha256=[v['sha256'] for v in samples[right]['image_files']], alignment='same original physical slot'),
                    metrics=contrast(torch, scene[left], scene[right])))
        lengths = []
        need(len({pair['pair_id'] for pair in pairs}) == 54
             and Counter(s for pair in pairs for s in pair['sids']) == Counter({s: 1 for s in samples}),
             'Duplicate or incomplete length pair')
        for pair in pairs:
            left, right = pair['sids']; small, large = samples[left], samples[right]
            positions = large['parent_positions']
            need(small['n_frames'] == 8 and large['n_frames'] == 16 and small['parent_positions'] == []
                 and large['parent_n_frames'] == 8 and len(positions) == 8 and positions == sorted(set(positions))
                 and all(type(i) is int and 0 <= i < 16 for i in positions)
                 and all(small[k] == large[k] == pair[k] for k in ('pair_id', 'contrast_id', 'variant', 'question')),
                 'Length-pair insertion metadata differs')
            need(len(states[left]) == 8 and len(states[right]) == 16
                 and all(states[left][i]['rooms'] == states[right][j]['rooms'] for i, j in enumerate(positions)), 'Inserted semantic subsequence differs')
            extra = [i for i in range(16) if i not in positions]
            need(all(all(room not in large['room_pair'] for room in states[right][j]['rooms']) for j in extra), 'Original inserted background provenance differs')
            matched = {k: scene[right][k][positions] if k in ('x', 'a', 'p') else scene[right][k] for k in ('x', 'a', 'p', 'q', 'z')}
            matched['z'] = .5 * matched['p'].double().sum(0)
            parent = dict(scene[left], z=.5 * scene[left]['p'].double().sum(0))
            dp = matched['p'].double() - scene[left]['p'].double()
            parent_term = .5 * dp.sum(0)
            insertion_term = .5 * scene[right]['p'][extra].double().sum(0)
            full_delta = scene[right]['z'].double() - scene[left]['z'].double()
            lengths.append(dict(labels=dict(pair_id=pair['pair_id'], contrast_id=pair['contrast_id'], variant=pair['variant'],
                parent_sid=left, child_sid=right, question_id=question_ids[small['question']]),
                provenance=dict(parent_positions_zero_based=positions, inserted_positions_zero_based=extra,
                    parent_image_sha256=[v['sha256'] for v in small['image_files']], child_image_sha256=[v['sha256'] for v in large['image_files']],
                    semantic_subsequence_exact=True, step_labels_may_change_at_moved_slots=True),
                metrics=dict(matched_parent=contrast(torch, parent, matched,
                    aggregate_reference='derived FP64 .5 sums of captured payloads at matched parent slots'),
                    inserted_rms_local=rows_metric(torch, scene[right]['x'][extra]), inserted_payload=rows_metric(torch, scene[right]['p'][extra]),
                    matched_parent_half_sum_delta=parent_term.tolist(), inserted_half_sum=insertion_term.tolist(),
                    total_half_sum_delta=(parent_term + insertion_term).tolist(), captured_aggregate_delta=full_delta.tolist(),
                    matched_parent_half_sum_delta_norm=norm(torch, parent_term), inserted_half_sum_norm=norm(torch, insertion_term),
                    total_half_sum_delta_norm=norm(torch, parent_term + insertion_term),
                    aggregate_reduction_difference=difference(torch, parent_term + insertion_term, full_delta))))
        need(len(variants) == 108 and len(lengths) == 54, 'Incomplete registered contrast inventory')
    files = []
    for name, value in (('questions', question_rows), ('scenes', scene_rows), ('variant_contrasts', variants), ('length_contrasts', lengths)):
        path = out / ('uniform_geometry_' + name + '.json')
        with path.open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
        files.append(dict(kind=name, file=str(path), sha256=sha(path), records=len(value)))
    return dict(passed=True, completed=True, protocol='uniform_training_first_query_geometry',
        source_sha256={name: sha(ROOT / name) for name in OWN}, input_bindings=ledger,
        uniform_summary_sha256=SUMMARY_SHA, uniform_analysis_sha256=ANALYSIS_SHA, uniform_plan_sha256=PLAN_SHA,
        checkpoint_sha256=CHECKPOINT_SHA, checkpoint_tensors=table, self_tests=fixtures,
        counts=dict(training_contexts=108, first_queries=108, local_occurrences=1296, questions=6,
                    variant_contrasts=108, length_contrasts=54, development_captures=0, test_captures=0, model_calls=0, head_calls=0),
        precision=dict(rms='FP32, epsilon 1e-6', mean='occurrence-weighted FP64 mean of FP32 RMS states, cast to FP32',
            linear_recomputation='CPU FP32 with captured FP32 query', reductions='FP64; population variance divisor equals occurrence count',
            payload_reference='captured FP32, promoted to FP64 for derivative and contrasts',
            saturation_rule='1 - captured_payload**2 <= 0.01', differences='descriptive only; no new fidelity threshold'),
        interpretation=dict(no_causal_identification=True, no_centering_intervention=True, no_reader_fitted=True,
                            no_labels_in_tensor_computation=True, no_outcome_rescore=True), files=files)

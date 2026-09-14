"""Bounded V18 saved-state diagnosis after the independent report; CPU Slurm only.

All four fixed models and all272 fresh scenes. No head/model invocation,
probability recomputation, intervention, fitting, or new accuracy criterion.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'outputs/native_aggregation_vlm/v18/diagnosis'
DATA = Path('/mnt/data/gabriele/gnn_transformer/v18_diagnosis')
OWN = ('scripts/analyze_native_vision_v18.py', 'slurm/native_vision_v18_diagnosis.sbatch')
MODES = ('native_gate', 'all_open')
SEEDS = (20, 21)
RUN_KEYS = tuple(f'{mode}_s{seed}' for mode in MODES for seed in SEEDS)
PARTITIONS = {'all': range(17), 'K0_8': range(9), 'K9_15': range(9, 16),
              'K16': (16,), 'K9_16': range(9, 17)}
EOS = (151645, 151643)
POLICY = dict(protocol='v18_all_scenes_saved_state_diagnosis', models=list(RUN_KEYS),
    scenes_per_model=272, families_per_model=136, lengths=[32, 64], counts=list(range(17)),
    families_per_count=8, geometry_position=0, geometry_prefix_ids=[],
    gate_error_definition='positive g==0; negative g>0; full-vocabulary native gates, no new threshold',
    selection_identity='sum(g)-K = sum_negative(g)-sum_positive(1-g)',
    actual_vector_gate='native g in native_gate; one on each actual row in all_open',
    partition_identity='P+Z=sum(saved_FP32_messages promoted to FP64)',
    paired_mapping='semantic parent_positions; retain rerendered physical Step and image identity changes',
    paired_product_identity='g64*u64-g32*u32=(g64-g32)*u32+g64*(u64-u32), FP64 from saved gates/payload',
    paired_message_roundoff='saved FP32 message delta minus FP64 product delta; descriptive',
    algebra_rtol=1e-10, algebra_atol=1e-10,
    native_fp32_differences='Descriptive only; no CPU/GPU kernel-equivalence gate',
    readout_comparison='saved original unfused global logits versus saved fused first-token logits',
    target_position_scope='canonical numeral-prefix compatible executed queries only; either native EOS accepted at terminal position',
    no_fit=True, no_model_calls=True, no_head_calls=True, no_probability_recomputation=True,
    no_intervention=True, no_outcome_selection=True, no_new_accuracy_decision=True)


def need(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def bind(path, expected, bindings):
    path = Path(path).resolve()
    actual = sha(path)
    record = dict(sha256=actual, expected_sha256=expected, bytes=path.stat().st_size)
    previous = bindings.get(str(path))
    need(previous is None or previous['sha256'] == actual, 'Input changed during diagnosis: ' + str(path))
    bindings[str(path)] = record
    need(expected is None or actual == expected, 'Input hash differs: ' + str(path))
    return path


def tensor_info(torch, value):
    value = value.detach().cpu().contiguous()
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest())


def snapshot(out):
    frozen = {name: sha(REPO / name) for name in OWN}
    (out / 'code').mkdir()
    for name, digest in frozen.items():
        target = out / 'code' / name.replace('/', '_')
        target.write_bytes((REPO / name).read_bytes())
        need(sha(target) == digest, 'Diagnostic source changed during snapshot')
    save(out / 'own_source_hashes.json', frozen)
    save(out / 'policy.json', POLICY)
    return frozen


def check_close(torch, actual, expected, message):
    need(torch.allclose(actual, expected, rtol=1e-10, atol=1e-10), message)
    return float((actual - expected).abs().max()) if actual.numel() else 0.


def stats(values):
    valid = [float(v) for v in values if v is not None]
    need(all(math.isfinite(v) for v in valid), 'Nonfinite descriptive statistic')
    return dict(n=len(valid), missing=len(values)-len(valid), mean=math.fsum(valid)/len(valid) if valid else None,
                minimum=min(valid) if valid else None, maximum=max(valid) if valid else None)


def cosine(a, b):
    denominator = float(a.norm()) * float(b.norm())
    return float(a @ b) / denominator if denominator else None


def difference(a, b):
    d = b.double() - a.double()
    return dict(max_abs=float(d.abs().max()) if d.numel() else 0.,
                l2=float(d.norm()), rms=float(d.square().mean().sqrt()) if d.numel() else None,
                exact=bool((a == b).all()))


def selection(torch, gates, positive, p0, p1):
    g = gates.double(); count = len(g); k = int(positive.sum())
    attenuation = (1-g[positive]).sum(); leakage = g[~positive].sum()
    error = check_close(torch, g.sum()-k, leakage-attenuation, 'Selection L-A identity failed')
    return dict(occurrences=count, positive_count=k, negative_count=count-k,
        positive_closed=int((g[positive] == 0).sum()), negative_open=int((g[~positive] > 0).sum()),
        positive_gate=stats(g[positive].tolist()), negative_gate=stats(g[~positive].tolist()),
        positive_attenuation=float(attenuation), negative_leakage=float(leakage), gate_sum=float(g.sum()),
        gate_sum_minus_K=float(g.sum()-k), identity_max_abs=error,
        positive_off_numeric_mass=stats((1-p0[positive]-p1[positive]).tolist()),
        negative_off_numeric_mass=stats((1-p0[~positive]-p1[~positive]).tolist()))


def scene_geometry(torch, cap, origin, positive, w, bias, mode):
    payload = cap['payload'][:, 0].double(); messages = cap['messages'][:, 0].double()
    gates = cap['applied_gates'][:, 0].double(); native_gates = origin['native_gates'].double()
    need(torch.equal(gates, native_gates if mode == 'native_gate' else torch.ones_like(gates)),
         'Applied gate does not match the audited mode')
    p = messages[positive].sum(0); z = messages[~positive].sum(0); total = messages.sum(0)
    identity_error = check_close(torch, p+z, total, 'Positive/negative message partition failed')
    wp, wz = w @ p, w @ z; q = cap['query'][0].double()
    need(not bool((gates[~positive] == 0).all()) or bool((z == 0).all()), 'Closed actual negative rows contributed a message')
    reconstructed = w @ total + q + bias
    product = gates[:, None] * payload
    metrics = dict(P_norm=float(p.norm()), Z_norm=float(z.norm()), PZ_cosine=cosine(p, z),
        projected_P_norm=float(wp.norm()), projected_Z_norm=float(wz.norm()), projected_PZ_cosine=cosine(wp, wz),
        query_norm=float(q.norm()), native_delta_norm=float(cap['delta'].double().norm()),
        partition_identity_max_abs=identity_error,
        all_native_negative_gates_closed=bool((native_gates[~positive] == 0).all()),
        all_applied_negative_gates_closed=bool((gates[~positive] == 0).all()), negative_message_exact_zero=bool((z == 0).all()),
        saved_message_product_roundoff=difference(product, messages),
        saved_fp32_aggregate_difference=difference(total, cap['aggregate'][0]),
        saved_fp32_preactivation_difference=difference(reconstructed, cap['preactivation'][0]))
    vectors = dict(positive=positive, native_gates=native_gates, applied_gates=gates, payload=payload,
        messages=messages, product_messages=product, P=p, Z=z, projected_P=wp, projected_Z=wz, query=q,
        saved_aggregate=cap['aggregate'][0].clone(), saved_preactivation=cap['preactivation'][0].clone(),
        reconstructed_preactivation=reconstructed)
    return metrics, vectors


def item_drift(torch, g32, u32, m32, g64, u64, m64):
    g32, u32, m32, g64, u64, m64 = [x.double() for x in (g32, u32, m32, g64, u64, m64)]
    gate_term = (g64-g32)[:, None] * u32
    payload_term = g64[:, None] * (u64-u32)
    product_delta = g64[:, None]*u64-g32[:, None]*u32
    error = check_close(torch, gate_term+payload_term, product_delta, 'Paired gate/payload product identity failed')
    message_delta = m64-m32
    roundoff = (m64-g64[:, None]*u64)-(m32-g32[:, None]*u32)
    reconciliation = check_close(torch, gate_term+payload_term+roundoff, message_delta,
                                 'Paired product plus saved-multiplication residual failed')
    return dict(gate_term=gate_term, payload_term=payload_term, product_delta=product_delta,
        message_delta=message_delta, multiplication_roundoff_delta=roundoff,
        product_identity_max_abs=error, roundoff_reconciliation_max_abs=reconciliation)


def subset_drift(values, mask, w):
    n = int(mask.sum()); fields = {}
    for key in ('gate_term', 'payload_term', 'product_delta', 'message_delta', 'multiplication_roundoff_delta'):
        rows = values[key][mask]; total = rows.sum(0); projected = rows @ w.T
        fields[key] = dict(count=n, sum=total.tolist(), sum_norm=float(total.norm()), projected_sum=(w @ total).tolist(),
            projected_sum_norm=float((w @ total).norm()),
            mean_item_l2=float(rows.norm(dim=1).mean()) if n else None,
            mean_square_item_shift=float(rows.square().sum(1).mean()) if n else None,
            projected_mean_square_item_shift=float(projected.square().sum(1).mean()) if n else None)
    return fields


def family_geometry(torch, left, right, a, b, w, positive):
    positions = b['parent_positions']; idx = torch.tensor(positions, dtype=torch.long)
    need(len(positions) == len(a['frames']) and positions == sorted(set(positions))
         and [b['frames'][i] for i in positions] == a['frames'], 'Semantic family occurrence mapping differs')
    added = torch.ones(len(b['frames']), dtype=torch.bool); added[idx] = False
    need(not bool(right['positive'][added].any()) and torch.equal(right['positive'][idx], positive),
         'Parent labels or inserted-negative labels differ')
    v = item_drift(torch, left['applied_gates'], left['payload'], left['messages'],
                   right['applied_gates'][idx], right['payload'][idx], right['messages'][idx])
    subsets = {name: subset_drift(v, mask, w) for name, mask in
               (('positive', positive), ('parent_negative', ~positive))}
    added_sum = right['messages'][added].sum(0)
    dq = right['query']-left['query']
    expected_delta = dq+w @ (v['message_delta'].sum(0)+added_sum)
    actual_delta = right['reconstructed_preactivation']-left['reconstructed_preactivation']
    error = check_close(torch, actual_delta, expected_delta, 'Family query/parent/added-negative identity failed')
    metrics = dict(parent_count=len(idx), positive_count=int(positive.sum()), added_negative_count=int(added.sum()),
        query_drift=difference(left['query'], right['query']),
        native_positive_gate_drift=difference(left['native_gates'][positive], right['native_gates'][idx][positive]),
        applied_positive_gate_drift=difference(left['applied_gates'][positive], right['applied_gates'][idx][positive]),
        positive_payload_drift=difference(left['payload'][positive], right['payload'][idx][positive]),
        positive_message_drift=difference(left['messages'][positive], right['messages'][idx][positive]),
        positive_and_parent_negative=subsets,
        added_negative_sum=added_sum.tolist(), added_negative_norm=float(added_sum.norm()),
        added_negative_projected_sum=(w @ added_sum).tolist(), added_negative_projected_norm=float((w @ added_sum).norm()),
        product_identity_max_abs=v['product_identity_max_abs'], roundoff_reconciliation_max_abs=v['roundoff_reconciliation_max_abs'],
        family_identity_max_abs=error, saved_fp32_preactivation_delta_difference=difference(expected_delta,
            right['saved_preactivation'].double()-left['saved_preactivation'].double()))
    return metrics, dict(**v, parent_positions=idx, added_mask=added, query_delta=dq,
                         added_negative_sum=added_sum, reconstructed_preactivation_delta=actual_delta)


def margin(torch, logits, target):
    x = logits.double(); need(x.ndim == 1 and 0 <= target < len(x), 'Malformed saved logit vector')
    competitors = []
    if target: competitors.append(x[:target].max())
    if target+1 < len(x): competitors.append(x[target+1:].max())
    best_other = max(float(v) for v in competitors)
    return dict(argmax=int(x.argmax()), target_token_id=target, target_logit=float(x[target]),
                correct=bool(int(x.argmax()) == target), correct_token_margin=float(x[target])-best_other)


def answer_diagnosis(torch, row, raw, numeral):
    ids = row['generated_ids']; target = numeral+[EOS[0]]; token_rows=[]; compatible=True; first_wrong=None
    canonical_first_difference=next((t for t in range(max(len(ids), len(target)))
        if t >= len(ids) or t >= len(target) or ids[t] != target[t]), None)
    for t, token in enumerate(ids):
        role = 'first_numeral' if t == 0 else ('later_numeral' if t < len(numeral) else ('native_eos' if t == len(numeral) else 'outside_target'))
        expected = numeral[t] if t < len(numeral) else (EOS[0] if t == len(numeral) else None)
        correct = token == expected if t < len(numeral) else (token in EOS if t == len(numeral) else False)
        entry = dict(position=t, observed_prefix_ids=ids[:t], generated_id=token, role=role,
                     canonical_target_prefix_compatible=compatible, expected_target=expected if compatible else None,
                     correct_at_compatible_position=bool(correct) if compatible else None)
        if compatible and expected is not None:
            entry['margin_to_canonical_target'] = margin(torch, raw['raw_logits'][t], expected)
        if compatible and not correct and first_wrong is None: first_wrong=dict(position=t, role=role, generated_id=token, expected_id=expected)
        token_rows.append(entry); compatible=compatible and correct
    before=margin(torch, raw['origin_artifact']['origin_logits'][-1, 0], numeral[0])
    after=margin(torch, raw['raw_logits'][0], numeral[0])
    first_correct=ids[0] == numeral[0]
    category = ('exact' if row['exact'] else 'format_or_termination_failure' if not row['parseable'] or not row['completed']
                else 'wrong_first_numeral' if not first_correct else 'wrong_number_after_correct_first_numeral')
    copied={k:row[k] for k in ('generated_ids','raw_text','text','prediction','parseable','parsed_count_correct','exact',
        'completed','truncated','no_nonterminal_special_tokens','first_token_nll')}
    return dict(**copied, first_token_correct=first_correct, failure_category=category,
        canonical_target_ids=target, first_canonical_difference=canonical_first_difference,
        first_wrong_compatible_position=first_wrong, executed_target_positions=token_rows,
        unfused_first_token=before, fused_first_token=after,
        first_token_correctness_transition=f"{int(before['correct'])}->{int(after['correct'])}",
        first_token_margin_change=after['correct_token_margin']-before['correct_token_margin'],
        canonical_target_mismatch_does_not_override_whole_answer_score=True)


def load_report(report_path, out, frozen, bindings):
    path=bind(report_path, None, bindings); summary=read(path)
    need(path.name == 'summary.json' and path.is_relative_to(REPO/'outputs/native_aggregation_vlm/v18'),
         'Require canonical V18 final report summary')
    need(summary.get('passed') is True and summary.get('completed') is True, 'Independent final report has not passed')
    analysis_path=bind(summary['analysis_file'], summary['analysis_sha256'], bindings); analysis=read(analysis_path)
    need(analysis_path.parent == path.parent and analysis.get('passed') is True and analysis.get('audit_passed') is True
         and analysis['source_sha256'] == summary['source_sha256'] and set(analysis['runs']) == set(RUN_KEYS)
         and analysis['policy']['protocol'] == 'v18_native_semantic_gate_final_only', 'Wrong/incomplete final report')
    need(analysis['policy']['seeds'] == list(SEEDS) and analysis['policy']['conditions'] == list(MODES)
         and analysis['policy']['steps'] == 4590 and analysis['reasoning_objective_achieved'] is False,
         'Wrong final-model protocol')
    # Freeze current sources and the exact report snapshots, without importing
    # or repeating any ancestor data/cache/model audit.
    for name, digest in analysis['source_sha256'].items():
        source=bind(REPO/name, digest, bindings)
        original=bind(path.parent/'code'/name.replace('/', '_'), digest, bindings)
        need(name not in frozen or frozen[name] == digest, 'Conflicting diagnostic/report source identity')
        frozen[name]=digest
        target=out/'code'/name.replace('/', '_')
        if not target.exists(): target.write_bytes(original.read_bytes())
        need(sha(target) == digest, 'Copied inherited source changed')
    fresh=analysis['data']['fresh_data']; manifest_path=bind(fresh['manifest_file'], fresh['manifest_sha256'], bindings)
    manifest=read(manifest_path); data_bindings=analysis['data']['data_bindings']
    need(manifest['fresh_test_seed'] == 20261118 and manifest['data_seed'] == 20261118,
         'Wrong already-audited fresh corpus')
    plan_path=bind(manifest['stage_plan_file'], manifest['stage_plan_sha256'], bindings)
    need(data_bindings[str(plan_path)] == sha(plan_path), 'Fresh plan absent from independent audit')
    plan=read(plan_path); samples_path=bind(plan['samples_file'], plan['samples_sha256'], bindings)
    need(data_bindings[str(samples_path)] == plan['samples_sha256'], 'Semantic labels absent from independent data audit')
    dry=read(samples_path); planned={r['sid']:r for r in dry}
    published=[r for n in (32, 64) for r in manifest['splits'][f'test_N{n}']['samples']]
    need(len(published) == len(planned) == 272 and set(planned) == {r['sid'] for r in published}, 'Fresh scene inventory differs')
    families=defaultdict(dict); by_sid={}
    for sample in published:
        p=planned[sample['sid']]
        for key in ('sid','n_frames','gold','question','content_sha256','target_character','target_room','pair_id','parent_positions'):
            need(sample[key] == p[key], 'Published/semantic label mapping differs: '+key)
        labels=[frame == [p['target_character'], p['target_room']] for frame in p['frames']]
        need(len(labels) == p['n_frames'] and sum(labels) == p['gold'], 'Offline positive labels differ')
        by_sid[p['sid']]=dict(sample=sample, planned=p, labels=labels)
        need(p['n_frames'] not in families[p['pair_id']], 'Duplicate family length')
        families[p['pair_id']][p['n_frames']]=p['sid']
    need(len(families) == 136 and all(set(v) == {32,64} for v in families.values())
         and Counter((p['n_frames'], p['gold']) for p in dry) == Counter({(n,k):8 for n in (32,64) for k in range(17)}),
         'Incomplete fixed N/K/family scope')
    return analysis, by_sid, dict(families)


def endpoint_and_rows(torch, key, audit, bindings):
    directory=Path(audit['directory']); summary_path=bind(directory/'summary.json', audit['summary_sha256'], bindings)
    summary=read(summary_path); endpoint=audit['endpoint']
    proof=read(bind(endpoint['endpoint_file'], endpoint['endpoint_sha256'], bindings))
    checkpoint=read_checkpoint(torch, endpoint, proof, bindings)
    config=checkpoint['config']; config_path=bind(directory/'config.json', None, bindings)
    need(read(config_path) == config and config['run_id'] == audit['run_id']
         and f"{config['condition']}_s{config['seed']}" == key and checkpoint['step'] == 4590
         and audit['selected_step'] == 4590 and audit['native_test_examples'] == 272
         and audit['one_dev_only'] is True and audit['dev_descriptive_only'] is True,
         'Audited fixed final endpoint identity differs')
    need(summary['test_file'] == audit['test_file'] and summary['test_sha256'] == audit['test_sha256'],
         'Run/report test binding differs')
    test=read(bind(audit['test_file'], audit['test_sha256'], bindings))
    need(test['n'] == len(test['rows']) == 272 and test['all_origin_artifacts_and_prefix_captures_retained'] is True,
         'Complete raw test inventory required')
    bind(test['raw_file'], test['raw_sha256'], bindings)  # Already audited global-logit archive: hash only.
    for row in test['rows']: bind(row['raw_file'], row['raw_sha256'], bindings)
    model_path=Path(config['model']['path'])
    for name, digest in config['model']['metadata_sha256'].items(): bind(model_path/name, digest, bindings)
    # No backbone weight shard is read. Only the fitted 96-dimensional branch
    # projection and its small already-audited checkpoint are consumed.
    state=checkpoint['branch']
    return config, test['rows'], state['aggregate_projection.weight'].double(), state['aggregate_projection.bias'].double()


def read_checkpoint(torch, endpoint, proof, bindings):
    path=bind(endpoint['checkpoint'], endpoint['checkpoint_sha256'], bindings)
    need(proof['checkpoint_sha256'] == endpoint['checkpoint_sha256'] and proof['parameter_sha256'] == endpoint['parameter_sha256'],
         'Endpoint proof identity differs')
    checkpoint=torch.load(path, map_location='cpu', weights_only=True)
    need(set(checkpoint) == {'branch','step','config'} and checkpoint['step'] == 4590, 'Wrong final checkpoint schema')
    table={k:tensor_info(torch, v) for k,v in checkpoint['branch'].items()}
    need(table == endpoint['parameter_table'] == proof['before_evaluation'] == proof['after_evaluation']
         and object_sha(table) == endpoint['parameter_sha256'], 'Actual consumed projection/checkpoint differs from deployed endpoint')
    return checkpoint


def summarize_scenes(rows):
    groups={}
    for n in (32,64):
        for partition, counts in PARTITIONS.items():
            subset=[r for r in rows if r['n_frames'] == n and r['gold'] in counts]
            groups[f'N{n}/{partition}']=scene_group(subset)
        for k in range(17): groups[f'N{n}/K{k}']=scene_group([r for r in rows if r['n_frames'] == n and r['gold'] == k])
    return groups


def pooled_stats(groups):
    valid=[x for x in groups if x['n']]
    count=sum(x['n'] for x in valid)
    return dict(n=count,mean=math.fsum(x['mean']*x['n'] for x in valid)/count if count else None,
        minimum=min(x['minimum'] for x in valid) if count else None,
        maximum=max(x['maximum'] for x in valid) if count else None)


def scene_group(rows):
    return dict(n=len(rows), exact=sum(r['answer']['exact'] for r in rows),
        first_token_correct=sum(r['answer']['first_token_correct'] for r in rows),
        failure_categories=dict(Counter(r['answer']['failure_category'] for r in rows)),
        unfused_to_fused_first_token=dict(Counter(r['answer']['first_token_correctness_transition'] for r in rows)),
        first_wrong_compatible_roles=dict(Counter(r['answer']['first_wrong_compatible_position']['role']
             for r in rows if r['answer']['first_wrong_compatible_position'] is not None)),
        native_selection={part:dict(positive_closed=sum(r['selection'][part]['positive_closed'] for r in rows),
            negative_open=sum(r['selection'][part]['negative_open'] for r in rows),
            positive_occurrences=sum(r['selection'][part]['positive_count'] for r in rows),
            negative_occurrences=sum(r['selection'][part]['negative_count'] for r in rows),
            positive_gate=pooled_stats([r['selection'][part]['positive_gate'] for r in rows]),
            negative_gate=pooled_stats([r['selection'][part]['negative_gate'] for r in rows]),
            positive_off_numeric_mass=pooled_stats([r['selection'][part]['positive_off_numeric_mass'] for r in rows]),
            negative_off_numeric_mass=pooled_stats([r['selection'][part]['negative_off_numeric_mass'] for r in rows]),
            total_positive_attenuation=math.fsum(r['selection'][part]['positive_attenuation'] for r in rows),
            total_negative_leakage=math.fsum(r['selection'][part]['negative_leakage'] for r in rows),
            positive_attenuation=stats([r['selection'][part]['positive_attenuation'] for r in rows]),
            negative_leakage=stats([r['selection'][part]['negative_leakage'] for r in rows]),
            gate_sum_minus_K=stats([r['selection'][part]['gate_sum_minus_K'] for r in rows])) for part in ('all','Step_le16','Step_gt16')},
        geometry={name:stats([r['geometry'][name] for r in rows]) for name in
                  ('P_norm','Z_norm','PZ_cosine','projected_P_norm','projected_Z_norm','projected_PZ_cosine','query_norm')},
        first_token_margin_change=stats([r['answer']['first_token_margin_change'] for r in rows]))


def summarize_families(rows):
    groups={}
    selections={**PARTITIONS, **{f'K{k}':(k,) for k in range(17)}}
    for key, counts in selections.items():
        subset=[r for r in rows if r['gold'] in counts]
        groups[key]=dict(n=len(subset), query_drift=stats([r['geometry']['query_drift']['l2'] for r in subset]),
            positive_gate_drift=stats([r['geometry']['native_positive_gate_drift']['l2'] if r['gold'] else None for r in subset]),
            positive_payload_drift=stats([r['geometry']['positive_payload_drift']['l2'] if r['gold'] else None for r in subset]),
            positive_message_drift=stats([r['geometry']['positive_message_drift']['l2'] if r['gold'] else None for r in subset]),
            added_negative_projected_norm=stats([r['geometry']['added_negative_projected_norm'] for r in subset]),
            positive_gate_term_projected_norm=stats([r['geometry']['positive_and_parent_negative']['positive']['gate_term']['projected_sum_norm'] if r['gold'] else None for r in subset]),
            positive_payload_term_projected_norm=stats([r['geometry']['positive_and_parent_negative']['positive']['payload_term']['projected_sum_norm'] if r['gold'] else None for r in subset]),
            positive_roundoff_delta_norm=stats([r['geometry']['positive_and_parent_negative']['positive']['multiplication_roundoff_delta']['sum_norm'] if r['gold'] else None for r in subset]))
    return groups


def self_test(torch):
    tests=[]; positive=torch.tensor([True,False,True,False]); g=torch.tensor([.75,.125,0.,0.])
    p0=torch.tensor([.1,.75,.8,.8],dtype=torch.float64); p1=torch.tensor([.85,.875,.1,.1],dtype=torch.float64)
    selected=selection(torch,g,positive,p0,p1)
    need(selected['positive_closed'] == 1 and selected['negative_open'] == 1
         and selected['positive_attenuation'] == 1.25 and selected['negative_leakage'] == .125
         and selected['gate_sum_minus_K'] == -1.125, 'Known selection leakage/attenuation fixture failed')
    empty=selection(torch,g[:0],positive[:0],p0[:0],p1[:0])
    need(empty['positive_gate']['mean'] is None and empty['gate_sum_minus_K'] == 0., 'Empty selection fabricated an observation')
    tests.append('selection_identity_signs_and_empty_subsets')
    u0=torch.tensor([[.2,.3],[.7,-.3]],dtype=torch.float32); u1=u0+.11
    g0=torch.tensor([.3,.7]);g1=torch.tensor([.8,.2]);m0=g0[:,None]*u0;m1=g1[:,None]*u1
    changed=item_drift(torch,g0,u0,m0,g1,u1,m1)
    need(float(changed['multiplication_roundoff_delta'].abs().max()) > 0., 'FP32 multiplication residual was silently removed')
    same=item_drift(torch,g0,u0,m0,g0,u0,m0)
    need(bool((same['message_delta'] == 0).all()), 'Identical paired items changed')
    tests.append('gate_payload_identity_preserves_saved_fp32_multiplication_roundoff')
    p=torch.tensor([True,False]); w=torch.eye(2,dtype=torch.float64)
    a=dict(frames=[['a','r'],['b','r']]); b=dict(frames=[['x','r'],['a','r'],['b','r']],parent_positions=[1,2])
    left=dict(positive=p,native_gates=g0.double(),applied_gates=g0.double(),payload=u0.double(),messages=m0.double(),query=torch.zeros(2,dtype=torch.float64))
    right=dict(positive=torch.tensor([False,True,False]),native_gates=torch.tensor([0.,.8,.2],dtype=torch.float64),
        applied_gates=torch.tensor([0.,.8,.2],dtype=torch.float64),payload=torch.cat((torch.ones(1,2),u1)).double(),
        messages=torch.cat((torch.zeros(1,2),m1)).double(),query=torch.ones(2,dtype=torch.float64))
    for packet in (left,right):
        packet['reconstructed_preactivation']=packet['query']+packet['messages'].sum(0)
        packet['saved_preactivation']=packet['reconstructed_preactivation'].float()
    paired,_=family_geometry(torch,left,right,a,b,w,p)
    need(paired['added_negative_count'] == 1 and paired['added_negative_norm'] == 0.
         and paired['query_drift']['l2'] > 0., 'Mapped parent/addition/query fixture differs')
    try: family_geometry(torch,left,right,a,dict(b,parent_positions=[0,2]),w,p)
    except ValueError: pass
    else: raise ValueError('Bad semantic parent mapping was accepted')
    tests.append('mapped_parent_added_negative_and_query_partition')
    need(subset_drift(changed,torch.tensor([False,False]),w)['message_delta']['mean_item_l2'] is None,
         'Empty positive subset fabricated a mean')
    need(set(PARTITIONS['K0_8'])|set(PARTITIONS['K9_15'])|set(PARTITIONS['K16']) == set(range(17))
         and not set(PARTITIONS['K0_8'])&set(PARTITIONS['K9_15']) and set(PARTITIONS['K9_16']) == set(range(9,17)),
         'Mandatory K partitions differ')
    tests.append('empty_positive_and_count_partitions')
    actual_gate=torch.tensor([1.,0.]);payload=torch.tensor([[.5,-.25],[.7,.8]])
    messages=actual_gate[:,None]*payload;query=torch.tensor([.1,.2]);total=messages.sum(0)
    cap=dict(payload=payload[:,None],messages=messages[:,None],applied_gates=actual_gate[:,None],
        query=query[None],aggregate=total[None],preactivation=(total+query)[None],delta=torch.zeros(1,2))
    result,_=scene_geometry(torch,cap,dict(native_gates=actual_gate),p,w,torch.zeros(2,dtype=torch.float64),'native_gate')
    need(result['Z_norm'] == 0. and result['PZ_cosine'] is None and result['all_applied_negative_gates_closed'],
         'Closed negative branch or undefined cosine fixture differs')
    cap['messages']=torch.zeros_like(cap['messages']);cap['applied_gates']=torch.zeros_like(cap['applied_gates'])
    cap['aggregate']=torch.zeros_like(cap['aggregate']);cap['preactivation']=cap['query'].clone()
    result,_=scene_geometry(torch,cap,dict(native_gates=torch.zeros(2)),torch.zeros(2,dtype=torch.bool),w,torch.zeros(2,dtype=torch.float64),'native_gate')
    need(result['P_norm'] == result['Z_norm'] == 0. and result['query_norm'] > 0.,
         'K0/all-closed messages incorrectly forced a zero query readout')
    tests.append('actual_saved_message_partition_closed_negative_and_K0_query')
    # Tiny vocabulary, no native head: answer diagnosis operates on saved logits.
    x=torch.tensor([[0.,2.,1.,0.],[0.,1.,3.,0.],[0.,0.,0.,4.]])
    row=dict(generated_ids=[1,2,3],raw_text='fixture',text='fixture',prediction=12,parseable=True,
             parsed_count_correct=False,exact=False,completed=True,truncated=False,no_nonterminal_special_tokens=True,first_token_nll=.5)
    raw=dict(raw_logits=x,origin_artifact=dict(origin_logits=torch.tensor([[[3.,0.,0.,0.]]])))
    # A two-digit wrong continuation uses only positions0/1; later history is incompatible.
    result=answer_diagnosis(torch,row,raw,[1,0])
    need(result['first_wrong_compatible_position']['position'] == 1
         and result['executed_target_positions'][2]['canonical_target_prefix_compatible'] is False
         and result['first_token_correctness_transition'] == '0->1', 'First mismatch/history/readout fixture differs')
    tests.append('saved_logits_readout_and_first_wrong_compatible_prefix')
    return dict(passed=True,tests=len(tests),names=tests,no_model_loaded=True,no_head_calls=True)


def run(torch, report_path, out, frozen, bindings, tests):
    from transformers import AutoTokenizer
    analysis, metadata, families=load_report(report_path, out, frozen, bindings)
    save(out/'source_hashes.json', frozen)
    data=DATA/out.name; data.mkdir(parents=True, exist_ok=False)
    all_rows=[]; all_families=[]; models={}; archives={}; capture_count=0
    with (out/'scene_rows.jsonl').open('x') as scene_stream, (out/'family_rows.jsonl').open('x') as family_stream:
        for key in RUN_KEYS:
            audited=analysis['runs'][key]
            config, predictions, w, bias=endpoint_and_rows(torch, key, audited, bindings)
            save(out/f'input_bindings_before_{key}.json', bindings)
            need(len({r['sid'] for r in predictions}) == 272 and {r['sid'] for r in predictions} == set(metadata),
                 'Model raw scene support differs from the complete fresh manifest')
            tokenizer=AutoTokenizer.from_pretrained(config['model']['path'], use_fast=False, local_files_only=True)
            numerals={k:tokenizer(str(k), add_special_tokens=False)['input_ids'] for k in range(17)}
            need(all(1 <= len(v) <= 2 and not set(v)&set(tokenizer.all_special_ids) for v in numerals.values()),
                 'Unexpected canonical native numeral tokenization')
            rows=[]; vectors_by_sid={}; first_data=[]; family_data=[]
            for index, row in enumerate(predictions):
                spec=metadata[row['sid']]; sample=spec['sample']; p=spec['planned']
                need(all(row[k] == sample[k] for k in ('sid','gold','n_frames','content_sha256','pair_id')),
                     'Audited raw row/semantic scene identity differs')
                # Each raw hash was checked before the first numerical row. The
                # passed report already checked every full probability/capture.
                raw=torch.load(row['raw_file'], map_location='cpu', weights_only=True)
                need(raw['schema_version'] == 1 and raw['metadata'] == row['metadata']
                     and raw['generated_ids'] == row['generated_ids'] and len(raw['captures']) == len(row['generated_ids'])
                     and raw['origin_artifact']['artifact_sha256'] == row['origin_artifact_sha256'],
                     'Consumed raw trajectory differs from the independently audited record')
                cap=raw['captures'][0]; origin=raw['origin_artifact']; positive=torch.tensor(spec['labels'], dtype=torch.bool)
                need(cap['query_indices'] == [row['metadata']['original_prompt_width']-1]
                     and cap['stream_positions'] == cap['query_indices'], 'Geometry is not the original empty prefix')
                selection_rows={}
                for label, mask in (('all',torch.ones(len(positive),dtype=torch.bool)),
                                    ('Step_le16',torch.arange(len(positive))<16), ('Step_gt16',torch.arange(len(positive))>=16)):
                    selection_rows[label]=selection(torch,origin['native_gates'][mask],positive[mask],origin['p0'][mask],origin['p1'][mask])
                geo, vectors=scene_geometry(torch,cap,origin,positive,w,bias,config['condition'])
                answer=answer_diagnosis(torch,row,raw,numerals[row['gold']])
                need(answer['fused_first_token']['argmax'] == row['generated_ids'][0], 'Consumed first-token logits differ from audited generation')
                record=dict(model=key,condition=config['condition'],seed=config['seed'],sid=row['sid'],pair_id=p['pair_id'],
                    gold=row['gold'],n_frames=row['n_frames'],replica=p['replica'],position=0,observed_prefix_ids=[],
                    raw_file=row['raw_file'],raw_sha256=row['raw_sha256'],origin_artifact_sha256=row['origin_artifact_sha256'],
                    source_test_file=audited['test_file'],source_test_sha256=audited['test_sha256'],capture_count=len(raw['captures']),
                    tensor_index=index,selection=selection_rows,geometry=geo,answer=answer)
                rows.append(record); all_rows.append(record); capture_count+=len(raw['captures'])
                scene_stream.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n');scene_stream.flush()
                vectors.update(p0=origin['p0'].clone(),p1=origin['p1'].clone(),sid=row['sid'],pair_id=p['pair_id'])
                first_data.append(vectors);vectors_by_sid[row['sid']]=vectors
                del raw,origin,cap
            need(Counter((r['n_frames'],r['gold']) for r in rows) == Counter({(n,k):8 for n in (32,64) for k in range(17)}),
                 'Model analysis lost an N/K cell')
            paired_rows=[]
            for pair_index, (pair_id, members) in enumerate(sorted(families.items())):
                s32,s64=members[32],members[64];a,b=metadata[s32],metadata[s64]
                left,right=vectors_by_sid[s32],vectors_by_sid[s64]
                need(a['planned']['gold'] == b['planned']['gold'] and a['planned']['question'] == b['planned']['question'],
                     'Family count or question changed')
                result, values=family_geometry(torch,left,right,a['planned'],b['planned'],w,left['positive'])
                positions=b['planned']['parent_positions']; mapping=[]
                for old_index,new_index in enumerate(positions):
                    old_image=a['sample']['image_files'][old_index];new_image=b['sample']['image_files'][new_index]
                    mapping.append(dict(parent_index=old_index,child_index=new_index,parent_step=old_index+1,child_step=new_index+1,
                        positive=bool(left['positive'][old_index]),semantic_frame=a['planned']['frames'][old_index],
                        parent_image_sha256=old_image['sha256'],child_image_sha256=new_image['sha256'],
                        image_exact=old_image['sha256'] == new_image['sha256']))
                record=dict(model=key,condition=config['condition'],seed=config['seed'],pair_id=pair_id,sid32=s32,sid64=s64,
                    gold=a['planned']['gold'],position=0,observed_prefix_ids=[],tensor_index=pair_index,
                    parent_occurrence_mapping=mapping,matched_semantics=True,
                    rerendered_step_changes=sum(x['parent_step'] != x['child_step'] for x in mapping),
                    changed_image_occurrences=sum(not x['image_exact'] for x in mapping),geometry=result)
                paired_rows.append(record);all_families.append(record);values['pair_id']=pair_id;family_data.append(values)
                family_stream.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n');family_stream.flush()
            need(len(paired_rows) == 136 and Counter(r['gold'] for r in paired_rows) == Counter({k:8 for k in range(17)}),
                 'Missing a paired family or K stratum')
            artifact=data/f'{key}.pt'
            with artifact.open('xb') as stream:
                torch.save(dict(schema_version=1,model=key,endpoint=audited['endpoint'],projection=w,projection_bias=bias,
                                scenes=first_data,families=family_data),stream)
            archives[key]=dict(file=str(artifact),sha256=sha(artifact),scenes=len(first_data),families=len(family_data))
            models[key]=dict(run_id=audited['run_id'],directory=audited['directory'],summary_sha256=audited['summary_sha256'],
                endpoint=audited['endpoint'],source_test_file=audited['test_file'],source_test_sha256=audited['test_sha256'],
                reused_independent_gate_audit=audited['test_gate_audit'],original_metrics=audited['metrics'],
                scene_groups=summarize_scenes(rows),family_groups=summarize_families(paired_rows))
            save(out/f'{key}_summary.json',models[key])
            print(json.dumps(dict(model=key,scenes=len(rows),families=len(paired_rows),native_calls=0,head_calls=0)),flush=True)
            del vectors_by_sid,first_data,family_data
    need(len(all_rows) == 1088 and len(all_families) == 544, 'Incomplete fixed all-model diagnosis')
    save(out/'scenes.json',all_rows);save(out/'families.json',all_families)
    identities=dict(selection=max(r['selection']['all']['identity_max_abs'] for r in all_rows),
        message_partition=max(r['geometry']['partition_identity_max_abs'] for r in all_rows),
        paired_product=max(r['geometry']['product_identity_max_abs'] for r in all_families),
        paired_roundoff_reconciliation=max(r['geometry']['roundoff_reconciliation_max_abs'] for r in all_families),
        paired_query_parent_added_negative=max(r['geometry']['family_identity_max_abs'] for r in all_families))
    result=dict(schema_version=1,passed=True,completed=True,policy=POLICY,source_sha256=frozen,
        report_file=str(Path(report_path).resolve()),report_sha256=sha(report_path),models=models,archives=archives,
        scenes=1088,families=544,captures_in_bound_trajectories=capture_count,identity_max_abs=identities,
        tests=tests,original_decisions=analysis['decisions'],original_vision_milestone_gate=analysis['vision_milestone_gate'],
        no_new_accuracy_decision=True,reasoning_objective_achieved=False,
        scope_notes=['Labels are used offline only; all occurrences retain their sampling weights.',
            'Native-gate selection statistics are reported for both modes; actual vector accounting uses the applied mode gates.',
            'Parent mapping preserves semantic frames but may change the rendered Step and therefore the image.',
            'Vector geometry is comparable within a fitted model; different learned coordinate scales are not identified.',
            'Saved FP32 multiplication and reduction differences are descriptive, not new numerical accuracy gates.',
            'Canonical target-position mismatches do not replace the registered whitespace/leading-zero tolerant whole-answer score.',
            'Unfused/fused original logits compare the immediate first-token readout; later counterfactual trajectories are not observed.',
            'Norms and gate/payload decompositions do not identify causal accuracy mediation.'])
    save(out/'analysis.json',result)
    lines=['# V18 saved-state diagnosis','',
        'All four fixed models and all272 fresh scenes per model are retained. Geometry uses only the original empty prefix. No model, head, fitting, or intervention was run.',
        '', '| Model | N | Whole answers | First token | Closed positive gates | Open negative gates | Mean attenuation | Mean leakage | Mean projected P | Mean projected Z | Unfused→fused first-token transitions |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for key in RUN_KEYS:
        for n in (32,64):
            group=models[key]['scene_groups'][f'N{n}/all'];sel=group['native_selection']['all'];geo=group['geometry']
            lines.append(f"| {key} | {n} | {group['exact']}/{group['n']} | {group['first_token_correct']}/{group['n']} | "
                f"{sel['positive_closed']}/{sel['positive_occurrences']} | {sel['negative_open']}/{sel['negative_occurrences']} | "
                f"{sel['positive_attenuation']['mean']:.6g} | {sel['negative_leakage']['mean']:.6g} | "
                f"{geo['projected_P_norm']['mean']:.6g} | {geo['projected_Z_norm']['mean']:.6g} | {group['unfused_to_fused_first_token']} |")
    lines+=['','Selection columns describe the original native gates in both arms. Vector columns use gates actually applied by that arm.',
        '',f'All544 family comparisons retain semantic parent mappings and changed Step/image identities. Maximum FP64 identity residuals: `{identities}`.',
        '', 'Exact identities concern FP64 algebra. The separate saved FP32 multiplication/reduction residuals are descriptive.',
        '', 'If negative gates are closed, their branch messages are absent. Remaining errors can involve positive payloads or native decoding; these norms alone do not identify which intervention would repair them.',
        '', 'The original primary/practical decisions are copied unchanged. This analysis establishes neither task transfer nor composition with reasoning.',
        '', '[All scenes](scenes.json) · [All paired families](families.json) · [Per-N/K and mandatory partition summaries](analysis.json) · [Input identities](input_bindings.json) · [Sources](source_hashes.json).']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return dict(passed=True,completed=True,source_sha256=frozen,policy=POLICY,tests=tests,scenes=1088,families=544,
        captures_in_bound_trajectories=capture_count,identity_max_abs=identities,analysis_file=str(out/'analysis.json'),
        analysis_sha256=sha(out/'analysis.json'),report_file=str(out/'REPORT.md'),archives=archives,
        no_model_calls=True,no_head_calls=True,no_probability_recomputation=True,no_new_accuracy_decision=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true',required=True)
    parser.add_argument('--report',type=Path,required=True,help='Passed V18 independent report summary.json')
    args=parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION') == 'cpu'
         and not os.environ.get('SLURM_JOB_GPUS'), 'Run this bounded analysis on Slurm CPU only')
    out=OUT/f'diagnosis_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False)
    bindings={};frozen=snapshot(out);start=time.monotonic()
    save(out/'invocation.json',dict(argv=sys.argv,report=str(args.report.resolve()),job_id=os.environ['SLURM_JOB_ID'],policy=POLICY))
    result=None
    try:
        bind(args.report,None,bindings)
        import torch
        torch.set_num_threads(4)
        tests=self_test(torch);save(out/'self_tests.json',tests)
        result=run(torch,args.report,out,frozen,bindings,tests)
    except BaseException as exc:
        result=dict(passed=False,completed=False,source_sha256=frozen,policy=POLICY,error_type=type(exc).__name__,
                    error=str(exc),traceback=traceback.format_exc(),all_partial_artifacts_retained=True,
                    no_model_calls=True,no_head_calls=True)
        raise
    finally:
        save(out/'input_bindings.json',bindings)
        result.update(elapsed_seconds=time.monotonic()-start,slurm_job_id=os.environ['SLURM_JOB_ID'],
                      input_bindings_file=str(out/'input_bindings.json'),input_bindings_sha256=sha(out/'input_bindings.json'))
        save(out/'summary.json',result)


if __name__ == '__main__':
    main()

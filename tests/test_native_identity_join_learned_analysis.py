"""Independent saved-capture analysis tests; numerical execution is CPU Slurm only."""
from pathlib import Path
import json
import os
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION') != 'cpu':
    raise SystemExit('CPU Slurm required; no numerical execution on login')

import torch
from scripts.analyze_native_identity_join_learned import analyze_capture


def fixture(payload, scores, mode='clip', *, projection=None):
    """Construct a transparent algebraic capture; no core or runtime is used."""
    payload = torch.tensor(payload, dtype=torch.float32).reshape(-1, 1, 2)
    scores = torch.tensor(scores, dtype=torch.float32).reshape(-1, 1)
    n = len(scores); hidden = 3; rank = 2
    if mode == 'clip':
        gates = scores.clamp(0, 1)
    elif mode == 'sigmoid':
        gates = 1 / (1 + torch.exp(-4*(scores-.5)))
    else:
        values = torch.exp(4*(scores-scores.max())) if n else scores
        gates = values / values.sum(0) if n else scores
    messages = gates.unsqueeze(-1)*payload
    weights = {'query.weight': torch.zeros(rank, hidden), 'local.weight': torch.zeros(rank, hidden),
        'local_bias': torch.zeros(rank), 'aggregate_projection.weight': torch.eye(rank) if projection is None else projection,
        'aggregate_projection.bias': torch.zeros(rank), 'up.weight': torch.zeros(hidden, rank),
        'selection_weight': torch.zeros(rank), 'selection_bias': torch.tensor([.5])}
    origin = torch.zeros(n+1, 1, hidden, dtype=torch.float16)
    aggregate = messages.sum(0)
    cap = dict(query=torch.zeros(1, rank), payload=payload, scores=scores, gates=gates, messages=messages,
        aggregate=aggregate, preactivation=torch.nn.functional.linear(aggregate, weights['aggregate_projection.weight']),
        delta=torch.zeros(1, hidden), local_states=origin[:-1].clone(), global_states=origin[-1].clone(),
        native_query_hidden=origin.clone(), native_delta=torch.zeros(1, hidden, dtype=torch.float16),
        fused_global=origin[-1].clone(), fused_query_hidden=origin.clone())
    return cap, weights


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_equal_gate_mass_can_have_different_payload_and_projected_geometry(self):
        a, weights = fixture([[1., 0.], [0., 1.]], [.5, .5])
        b, _ = fixture([[1., 0.], [-1., 0.]], [.5, .5])
        first = analyze_capture(torch, a, weights, 'clip', [True, False], [1, 17])
        second = analyze_capture(torch, b, weights, 'clip', [True, False], [1, 17])
        self.assertEqual(first['selection']['gate_sum'], second['selection']['gate_sum'])
        self.assertGreater(first['geometry']['sum_norm'], 0.)
        self.assertEqual(second['geometry']['sum_norm'], 0.)
        self.assertEqual(second['geometry']['projected_P_Z_cosine'], -1.)
        self.assertTrue(first['metadata']['no_scalar_ceiling_or_rank_conclusion'])

    def test_closed_positive_and_negative_leakage_are_distinct(self):
        cap, weights = fixture([[1., 0.], [0., 1.], [1., 1.], [-1., 0.]], [-1., .25, 1., 0.])
        result = analyze_capture(torch, cap, weights, 'clip', [True, False, True, False], [1, 2, 17, 18])
        self.assertEqual(result['selection']['closed_relevant_count'], 1)
        self.assertEqual(result['selection']['nonzero_irrelevant_count'], 1)
        self.assertEqual(result['selection']['irrelevant_gate_sum'], .25)
        self.assertEqual(result['selection']['irrelevant_fraction_of_gate_sum'], .2)
        self.assertEqual(result['selection']['closed_message_coordinate_count'], 4)
        self.assertEqual(result['groups']['relevant_step_gt16']['count'], 1)
        self.assertEqual(result['groups']['irrelevant_step_le16']['gate_sum'], .25)
        self.assertEqual(result['geometry']['partition_residual']['maximum_absolute'], 0.)

    def test_softmax_repeat_is_normalized_clip_repeat_is_extensive(self):
        for mode in ('clip', 'softmax'):
            cap, weights = fixture([[1., 0.], [0., 1.]], [.5, .5], mode)
            repeated, _ = fixture([[1., 0.], [0., 1.], [1., 0.], [0., 1.]], [.5]*4, mode)
            a = analyze_capture(torch, cap, weights, mode, [True, False], [1, 17])
            b = analyze_capture(torch, repeated, weights, mode, [True, False]*2, [1, 17, 2, 18])
            self.assertAlmostEqual(b['geometry']['sum_norm'], a['geometry']['sum_norm']*(2 if mode == 'clip' else 1))
            if mode == 'softmax':
                self.assertEqual(a['selection']['gate_sum'], 1.)
                self.assertEqual(b['selection']['softmax_gate_sum_error'], 0.)

    def test_projection_has_no_bias_and_reports_cancellation(self):
        projection = torch.tensor([[2., 0.], [0., 3.]])
        cap, weights = fixture([[1., 0.], [-1., 0.]], [1., 1.], projection=projection)
        weights['aggregate_projection.bias'].fill_(100.)
        result = analyze_capture(torch, cap, weights, 'clip', [True, False], [1, 17])
        self.assertEqual(result['geometry']['P_norm'], 1.)
        self.assertEqual(result['geometry']['projected_P_norm'], 2.)
        self.assertEqual(result['geometry']['projected_Z_norm'], 2.)
        self.assertEqual(result['geometry']['projected_sum_norm'], 0.)
        self.assertGreater(result['cpu_functional_differences']['preactivation']['maximum_absolute'], 0.)

    def test_empty_partitions_and_empty_actuals_are_json_safe(self):
        for n in (0, 2):
            cap, weights = fixture([[.1, .2]]*n, [.5]*n, 'softmax')
            result = analyze_capture(torch, cap, weights, 'softmax', [False]*n, list(range(1, n+1)))
            self.assertEqual(result['geometry']['P_norm'], 0.)
            self.assertIsNone(result['geometry']['P_Z_cosine'])
            self.assertIsNone(result['groups']['relevant']['gates']['mean'])
            self.assertIsNone(result['groups']['step_gt16']['message_mean_norm'])
            json.dumps(result, allow_nan=False)

    def test_product_roundoff_is_separate_from_exact_saved_message_partition(self):
        cap, weights = fixture([[.3, .7], [.2, .9]], [.1, .8])
        result = analyze_capture(torch, cap, weights, 'clip', [True, False], [1, 17])
        self.assertGreater(result['geometry']['fp64_product_vs_saved_fp32_message']['maximum_absolute'], 0.)
        self.assertTrue(result['checks']['partition_identity_passed'])
        self.assertTrue(result['checks']['messages_exact'])

    def test_native_cast_must_precede_add_and_only_global_may_change(self):
        cap, weights = fixture([[.2, .3]], [.5])
        cap['native_query_hidden'][-1].fill_(1.)
        cap['global_states'].fill_(1.)
        cap['delta'].fill_(.0004884)
        cap['native_delta'] = cap['delta'].half()
        cap['fused_global'] = cap['global_states'] + cap['native_delta']
        cap['fused_query_hidden'][-1] = cap['fused_global']
        analyze_capture(torch, cap, weights, 'clip', [True], [1])
        invalid = {k: v.clone() for k, v in cap.items()}
        invalid['fused_global'] = (cap['global_states'].float()+cap['delta']).half()
        invalid['fused_query_hidden'][-1] = invalid['fused_global']
        self.assertFalse(torch.equal(invalid['fused_global'], cap['fused_global']))
        with self.assertRaisesRegex(ValueError, 'Native write'):
            analyze_capture(torch, invalid, weights, 'clip', [True], [1])
        invalid = {k: v.clone() for k, v in cap.items()}
        invalid['fused_query_hidden'][0, 0, 0] = 1.
        with self.assertRaisesRegex(ValueError, 'Native write'):
            analyze_capture(torch, invalid, weights, 'clip', [True], [1])

    def test_rejects_tampered_product_clip_gate_ownership_and_labels(self):
        cap, weights = fixture([[.2, .3], [.1, .7]], [0., .5])
        invalid = {k: v.clone() for k, v in cap.items()}; invalid['messages'][0, 0, 0] = .1
        with self.assertRaisesRegex(ValueError, 'messages'):
            analyze_capture(torch, invalid, weights, 'clip', [True, False], [1, 17])
        invalid = {k: v.clone() for k, v in cap.items()}; invalid['scores'][0] = .5
        with self.assertRaisesRegex(ValueError, 'Clip gate'):
            analyze_capture(torch, invalid, weights, 'clip', [True, False], [1, 17])
        invalid = {k: v.clone() for k, v in cap.items()}; invalid['local_states'][0, 0, 0] = 1.
        with self.assertRaisesRegex(ValueError, 'ownership'):
            analyze_capture(torch, invalid, weights, 'clip', [True, False], [1, 17])
        for labels, steps in (([1, 0], [1, 17]), ([True, False], [1, 1]), ([True], [1, 17]), ([True, False], [1, True])):
            with self.assertRaises(ValueError):
                analyze_capture(torch, cap, weights, 'clip', labels, steps)
        invalid = {k: v.clone() for k, v in cap.items()}; invalid['scores'][0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            analyze_capture(torch, invalid, weights, 'clip', [True, False], [1, 17])

    def test_relevance_changes_only_offline_partitions_not_functional_reconstruction(self):
        cap, weights = fixture([[.2, .3], [.1, .7]], [.2, .8], 'sigmoid')
        a = analyze_capture(torch, cap, weights, 'sigmoid', [True, False], [1, 17])
        b = analyze_capture(torch, cap, weights, 'sigmoid', [False, True], [17, 1])
        self.assertEqual(a['cpu_functional_differences'], b['cpu_functional_differences'])
        self.assertEqual(a['score_gate_recomputation'], b['score_gate_recomputation'])
        self.assertEqual(a['geometry']['P_norm'], b['geometry']['Z_norm'])
        self.assertTrue(a['score_gate_recomputation']['descriptive'])


    def test_zero_functional_fixture_and_input_immutability(self):
        for mode in ('clip', 'sigmoid', 'softmax'):
            cap, weights = fixture([[0., 0.], [0., 0.]], [.5, .5], mode)
            before = {k: v.clone() for k, v in cap.items()}
            before_weights = {k: v.clone() for k, v in weights.items()}
            result = analyze_capture(torch, cap, weights, mode, [True, False], [1, 17])
            self.assertTrue(all(v['exact'] for v in result['cpu_functional_differences'].values()))
            self.assertTrue(all(torch.equal(v, before[k]) for k, v in cap.items()))
            self.assertTrue(all(torch.equal(v, before_weights[k]) for k, v in weights.items()))
            self.assertTrue(result['checks']['global_native_add_exact'])


if __name__ == '__main__':
    unittest.main()

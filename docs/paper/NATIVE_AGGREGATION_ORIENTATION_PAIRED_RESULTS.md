# Paired-minibatch controls: training failure

Pairing original and room-swapped examples within minibatches did not solve the216-context training task. Both independent computational reports passed. All three fixed6000-step models failed the prespecified first-query screen (206/216 pooled,103/108 per orientation and16/18 complete twelve-context families).

| Fixed arm | Original | Flipped | Pooled | Complete families |
|---|---:|---:|---:|---:|
| Exact local codes |32/108|36/108|68/216|0/18|
| Visual product |39/108|38/108|77/216|0/18|
| Visual additive |37/108|36/108|73/216|0/18|

The shared stage443724 preserved the exact48000-entry multiset,96000 scene presentations and213330 training target positions. Its deterministic permutation created5994 orientation-balanced updates and retained six tail updates. Sixteen boundary batches contained repeated base IDs; these were declared and retained. Original unfitted weights, inputs, native FP16 head,6000 learning-rate values and full-name/EOS CE were unchanged. The changed order also changes sample-to-learning-rate assignments, clipping and Adam trajectories, so this is a bounded order control rather than a universal test of balanced training.

Code report443742 and visual report443743 independently reconstructed all6000 logged updates per run, checked66 saved computational captures and replayed all648 final first queries with the actual CPU native head. All CPU/GPU argmax comparisons agreed. Maximum total-variation differences were0.003722652 (code),0.003513833 (product) and0.003611808 (additive), below the unchanged0.02 gate. Earlier strict numerical failures443718 and443682 remain failures.

Jobs443735/443736/443737 completed in64/76/77 allocated GPU-seconds,217 total with maximum three GPUs. Independent reports completed in42/52 CPU-seconds; preflights443728/443734 took18/14 seconds and shared stage443724 took5 seconds. There is no native whole-answer, fresh-data or extrapolation result for these failed models. No further order variant is selected.

The next bounded diagnostic changes only which already-computed positions contribute gradients: preserve full native forward shapes but supervise first queries at their original weights. This addresses the gap between first-query capacity and the full-sequence training objective; it does not establish a useful visual method.

Artifacts: [code report](../../outputs/native_aggregation_vlm/identity_join_orientation_paired_code/report_443742/REPORT.md), [visual report](../../outputs/native_aggregation_vlm/identity_join_orientation_paired_visual/report_443743/REPORT.md), [shared release](../../outputs/native_aggregation_vlm/identity_join_orientation_paired_order/three_arm_release.json).

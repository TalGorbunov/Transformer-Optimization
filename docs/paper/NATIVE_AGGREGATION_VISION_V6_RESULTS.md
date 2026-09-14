# V6: local teacher distillation did not improve native extrapolation

V6 failed both registered efficacy criteria in both seeds. Accurate isolated-image teacher judgments did not transfer into useful native aggregate answers through the rank 96 reader. The small seed 6 improvement over control came entirely from always answering zero on familiar OOD scenes. There is no demonstrated aggregation improvement, reasoning-composition benefit, or new attention mechanism.

The canonical [report](../../outputs/native_aggregation_vlm/v6/REPORT.md), [verified analysis](../../outputs/native_aggregation_vlm/v6/analysis.json), and [figure](../../outputs/native_aggregation_vlm/v6/comparison.png) were produced by CPU job 441682. Unlike V3–V5, this comparison used fresh complete test sequences, generated before model outcomes. Local rendered atoms still recur in the finite benchmark; this is not unseen-perception or natural-video generalization.

Both conditions deployed the unchanged V5 SUM architecture: independent visual memory, a rank 96 branch at layer 14, upper-four-layer LoRA, and 1,762,400 trainable native parameters. Both received the exact V4 refresh schedule: 1,540 distinct scenes, 1,620 presentations, 19,440 presented frames, 405 optimizer updates and nine 72-example development evaluations. Only development exact accuracy, raw first-token NLL and earliest tie-breaking selected checkpoints. All 452 fresh test contexts were retained.

A separate 291-parameter affine head learned three-way probabilities for 0, 1, other from each pre-merge message at the last original prompt token. Both conditions used native answer/EOS CE plus teacher KL averaged over images. Control detached messages for KL; aligned propagated KL into the reader. The head used its own initialization RNG, optimizer and clipping, and was absent from inference. No teacher-forced answer token supplied the auxiliary query.

The frozen isolated-image teacher passed its prospective gate on all 18,800 original training-frame occurrences: 5,840 true positives, 12,960 true negatives, no conditional 0/1 errors, and mean full-vocabulary 0/1 probability mass 0.99983905. All 9,980 deduplicated targets were retained. This establishes training-domain teacher feasibility, not successful student distillation or aggregate counting.

| Training / seed | N16 /108 | N32 /108 | N64 /108 | Familiar OOD /216 | Nonzero OOD /192 | Unseen K9–16 /128 | Selected block |
|---|---:|---:|---:|---:|---:|---:|---:|
| Control /6 |25 (23.1%)|14 (13.0%)|1 (0.9%)|15 (6.9%)|15 (7.8%)|0|7|
| Control /7 |18 (16.7%)|14 (13.0%)|0|14 (6.5%)|14 (7.3%)|0|7|
| Aligned /6 |25 (23.1%)|12 (11.1%)|12 (11.1%)|24 (11.1%)|0|0|9|
| Aligned /7 |24 (22.2%)|12 (11.1%)|0|12 (5.6%)|12 (6.2%)|0|9|

Aligned-minus-control familiar OOD differences were +4.17 percentage points for seed 6 (paired-anchor 95% interval [−2.78,+11.57]) and −0.93 pp for seed 7 ([−2.78,+0.93]); N16 differences were 0 and +5.56 pp. Neither seed met the required +5 pp OOD improvement. The separate practical requirement—at least 70% familiar OOD and 60% nonzero OOD in each aligned seed, alongside the relative criterion—also failed. The pooled difference, +1.62 pp ([−2.08,+5.56]), is descriptive and cannot rescue failed individual seeds. Intervals use 10,000 whole-anchor resamples retaining paired lengths and the two fixed seeds; they do not estimate variation over training seeds.

Aligned/6 answered zero on every 216 familiar OOD example: its 24 correct answers are exactly the zero-count cases, with OOD MAE 4 and bias −4. Control/6 parsed only 1/108 N64 answers; control/7 parsed none. Aligned/7 parsed 107/108 N64 answers, but 93 were `8080`, seven `8008`, and seven `8088`. Its conditional N64 MAE was 8,071.79. High parse rate therefore did not imply meaningful counts. Every invalid answer remained incorrect; conditional MAE must be read with its parse denominator. All four checkpoints scored 0/128 on unseen counts.

The implementation audit reproduced identical native and auxiliary initializations and all ordered training inputs within each seed. First-batch count losses were identical. Native gradient clipping nevertheless differed: control clipped 387/405 and 254/405 updates, versus 399/405 and 276/405 for aligned. Auxiliary clipping was separate, but aligned reader gradients changed native combined clipping and could indirectly rescale LoRA updates. This comparison tests the complete registered auxiliary-training treatment; it does not isolate correspondence-specific learning or a sole architectural cause. Full Adam moments and per-step gradient tensors were not retained.

The separately registered [saved-head audit](../../outputs/native_aggregation_vlm/v6/local_head/run_441687/REPORT.md), CPU 441687, applied every selected head to all 452 contexts/18,240 frame occurrences, without refitting or model calls. Three-way argmax treated “other” as incorrect for either binary truth class; none of these heads selected it. AUROC used unconditional probability of 1, retaining all frames and exact ties.

| Selected head | N16 AUROC | N32 AUROC | N64 AUROC | N64 TPR / FPR |
|---|---:|---:|---:|---:|
| Control /6 |0.4458|0.4441|0.4407|0.0625 /0.0667|
| Control /7 |0.4941|0.4981|0.5064|0 /0|
| Aligned /6 |0.6733|0.6042|0.5439|0 /0|
| Aligned /7 |0.6732|0.5920|0.5276|0.5648 /0.5486|

Alignment improved N16 ranking, but reliable thresholded local transfer did not follow. Aligned/6 labeled every familiar OOD frame negative; aligned/7's N64 positive and false-positive rates were nearly equal. Both aligned N64 balanced accuracies were near 0.5. These observations weaken the account that cleanly readable local evidence was already available and only numeral readout failed. They do not prove information absence, rule out other decoders, or identify the causal bottleneck. Frame observations are correlated within scenes/families; no new significance threshold was introduced.

The [head-subspace audit](../../outputs/native_aggregation_vlm/v6/head_subspace/run_441688/REPORT.md), CPU 441688, used each fixed head's centered row space to split messages into two head-visible and 94 head-invisible dimensions. Retaining the same bias preserved local probabilities numerically on every context: maximum FP64 total variation was 1.28e−15; FP32 replay differences were at most 2.39e−7, with no argmax changes. This preserves the auxiliary head, not native decoding.

Negative-message sums were substantial in both components. At familiar N64, aligned/6 had mean native-space negative-sum norms 2,712 for the head-visible component and 987 for the invisible component; aligned/7 had 3,511 and 1,734. The visible component exceeded the invisible component in every reported arm/cell. Accumulation is therefore not confined to the auxiliary head's nullspace; these norms do not identify which component causes native errors. Native U does not preserve component orthogonality, so these norms are not additive variance fractions. Query states and rendered Step labels can change with length; these summaries are not causal interventions. Head-invisible does not mean irrelevant, and no projection repair was tested.

All four selected checkpoints passed the fixed post-training computational/state audit 441655, including all 20 unchanged strict numerical observations. Production retained native NF4/bf16/SDPA/bitsandbytes. Earlier V5 numerical failures remain preserved; these bounded checks do not prove general cache or reasoning equivalence.

Retained CPU failures were reporting/provenance failures: 441641 encountered floating-order differences in teacher-quality dictionary equality; 441662 passed run audits but failed the shared-runtime caller's dictionary/list interface; 441663 and 441668 stopped before diagnostic outputs because the canonical analysis-code snapshot still held the pre-repair reporter. Repairs preserved original logs/snapshots and changed no models, targets, predictions, selection or scientific criterion. Frozen diagnostics then completed as 441687 and 441688.

The complete V6 cost was **8,320 GPU-seconds (2.31111 GPU-hours)**: 1,104 for teacher profiling/cache generation, 98 for training software profiles, 6,962 for four mains and 156 for selected-checkpoint audits, below the 5-hour cap. CPU diagnostics add no GPU time. This excludes completed V5 work and the separately budgeted parallel-local software oracle. Verified analysis SHA256: `d808a284a8265d50f9fcefd446359965459971129add034ca5eea3f1d4f82685`.

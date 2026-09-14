# V16: cyclic Step labels improve the bank diagnostic, but do not solve the learned method

Replacing displayed frame numbers by a repeating 1–16 cycle substantially improves the two centered models when they use the live empirical null bank. Their N64 complete-answer scores rise from 3/17 and 5/17 to 14/17 each. The learned-mean versions score 9/17 and 10/17 after the same intervention, and their N32 scores fall. This is evidence that the rendered Step labels affect these frozen models; it is not a successful general aggregation method, fresh confirmation, or a reversal of V14's failed practical milestone. [Verified report](../../outputs/native_aggregation_vlm/v16/step_wrap_study/report_442774/REPORT.md) · [All cells, transitions and provenance](../../outputs/native_aggregation_vlm/v16/step_wrap_study/report_442774/analysis.json) · [V15 comparator](NATIVE_AGGREGATION_VISION_V15_RESULTS.md).

The experiment retains all four V14 final models: centered/offset, seeds 18/19, core update 4,590 and student update 8,000. It reuses the same 17 V15 families, one for each K=0–16, at N32 and N64. Each actual image's displayed `Step i` becomes `Step 1+((i-1)%16)`. Character and room contents, physical frame order, question, QA bytes and count labels stay fixed. The 24 training-reference image occurrences remain original. Both learned and bank modes still process N+25 streams and compute both candidate means; only the selected mean differs, with coefficient N for centered and one for offset. There is no fit, checkpoint selection, anchor, vocabulary mask, or forced answer prefix.

The CPU renderer proof covers all 1,632 actual image occurrences. The first 544 occurrences, corresponding to positions 1–16 in every scene, are byte-identical; the remaining 1,088 change only within the bottom strip at y≥476. All 1,272 unique original images are reproduced exactly, and 701 wrapped renders preserve occurrence weighting. Repeated displayed IDs do not merge physical frames. The retry's 36 prepared archive hashes exactly match those from the first CPU preparation. [Renderer proof](../../outputs/native_aggregation_vlm/v16/data_staging/stage_442744/summary.json) · [Final input and model plan](../../outputs/native_aggregation_vlm/v16/step_wrap_study/check_442755/plan.json).

Each model first reruns original K3/N32 and K6/N64 scenes in both modes. All 16 sentinel trajectories match the corresponding V15 token sequences exactly. Their 32 executed raw-logit vectors have maximum absolute difference zero, total variation zero and identical argmax. All four sentinel results are saved before the gate permits wrapped inference. These controls establish observed rerun stability on the sentinel scenes, not equality on every possible input.

Scores below are original→wrapped, each out of 17. Complete correctness requires the full stripped ASCII integer and native EOS within four unrestricted greedy tokens; nonterminal special tokens are invalid. First-token correctness is separate and remains an incomplete proxy for counts 10–16.

| Model | Mean | N32 complete | N64 complete | N32 first token | N64 first token |
|---|---|---:|---:|---:|---:|
| Centered 18 | Learned | 16→13 | 9→9 | 17→16 | 12→13 |
| Centered 18 | Bank | 16→17 | 3→14 | 17→17 | 7→17 |
| Centered 19 | Learned | 13→12 | 7→10 | 17→16 | 13→14 |
| Centered 19 | Bank | 16→17 | 5→14 | 17→17 | 9→17 |
| Offset 18 | Learned | 4→3 | 2→3 | 4→3 | 2→3 |
| Offset 18 | Bank | 4→3 | 2→3 | 4→3 | 2→3 |
| Offset 19 | Learned | 6→6 | 2→3 | 10→10 | 2→3 |
| Offset 19 | Bank | 6→6 | 2→3 | 10→10 | 2→3 |

The bank-mode N64 gains retain every originally correct answer: seed 18 gains eleven and loses none; seed 19 gains nine and loses none. Equal learned-mode totals conceal exchanged successes:

| Centered seed | Mean | N64 both correct | Original only | Wrapped only | Neither |
|---|---|---:|---:|---:|---:|
| 18 | Learned | 4 | 5 | 5 | 3 |
| 18 | Bank | 3 | 0 | 11 | 3 |
| 19 | Learned | 5 | 2 | 5 | 5 |
| 19 | Bank | 5 | 0 | 9 | 3 |

At N32, centered learned seed 18 loses three answers and gains none; seed 19 loses two and gains one. Both centered bank modes gain one with no losses. Offset seed 18 loses one N32 answer in each mode, while offset seed 19 keeps every N32 correctness decision unchanged. Each offset mode gains one N64 answer with no losses.

Both wrapped centered bank modes obtain N64 K0–8=9/9, K9–15=5/7 and K16=0/1, hence K9–16=5/8. Their first tokens are correct in all 17 cases, but three complete answers still fail. The wrapped centered learned partitions are 6/9, 2/7 and 1/1 for seed 18; 7/9, 2/7 and 1/1 for seed 19. All wrapped offset N64 successes lie in K0–8. Every K and all registered partitions remain in the report.

All wrapped N32 outputs, and all wrapped centered N64 outputs, are parseable and EOS-completed. At N64, offset seed 18's learned mode has 12 parseable/completed answers and five truncations; its bank mode has eleven and six. Offset seed 19 has 15 parseable, 16 completed and one truncated answer in each mode. All malformed and truncated trajectories remain in the denominator. The bank improvement is therefore not obtained by filtering failures.

The separately registered CPU decomposition audits all 668 current and 637 original executed-prefix captures, then analyzes only the first prefix of all 272 wrapped/original pairs and 16 sentinel controls. It uses the frozen projection and captured FP32 messages/means promoted to FP64. Let P and Z be the projected positive- and negative-frame message sums after subtracting the live bank mean per occurrence. The offset baseline B and deployed-mean mismatch E complete the exact P+Z+B+E representation. Positive/negative labels are used only offline. [Verified decomposition](../../outputs/native_aggregation_vlm/v16/first_prefix_decomposition/decomposition_442775/summary.json) · [Every paired observation and geometry](../../outputs/native_aggregation_vlm/v16/first_prefix_decomposition/decomposition_442775/geometry.json).

Across all 288 first-prefix comparisons, the global state, query, reference states/messages, bank mean, predicted mean and used mean are exactly unchanged. Thus ΔB, ΔE and Δquery are zero. Every projected per-occurrence change in the first 16 physical frames is zero; every sentinel change is zero. The observed preactivation change is accounted for by changes in actual-frame messages. The maximum FP64 delta-identity residual is 7.50e-13; the maximum discrepancy from native FP32 delta arithmetic is 2.29e-4 and remains descriptive.

The centered bank-mode N64 geometry shows a marked reduction in the negative residual. Means below include all 17 K values; cosine means use the 16 K>0 cases because K0 has no positive mean.

| Seed | Mean P norm, original→wrapped | Mean Z norm, original→wrapped | Mean cos(P,Z), original→wrapped | Mean norm of ΔP | Mean norm of ΔZ |
|---|---:|---:|---:|---:|---:|
| 18 | 1066.02→1073.73 | 99.56→30.38 | −0.931→−0.051 | 10.69 | 93.10 |
| 19 | 1147.19→1154.03 | 92.44→29.12 | −0.932→−0.030 | 9.92 | 87.58 |

These are within-model coordinates, not comparable universal units or additive logit attributions. The sum of negative-message changes is much larger than the positive-message change in these centered N64 cases, while the positive sum's magnitude changes little. This localizes an observed effect of the footer intervention in the message path. It does not establish that Z mediates the accuracy gains, that the bank estimates a population null, or that numeral range alone caused the original failure. Cycling jointly changes displayed range, glyph width and the original unique-ordinal grammar. Later trajectories can diverge and are not matched-state comparisons.

Report442774 passed in 55 allocated CPU-seconds; decomposition442775 passed in 64, with no new native model or head calls. The four successful study allocations used 450, 450, 450 and 448 GPU-seconds, totaling 1,798. The first attempt suffered startup/node failure and cancellations before model outputs, consuming another 1,090 GPU-seconds. Its sources and allocation evidence are preserved. A resource-only amendment raised the campaign cap from 4,200 to 5,400 before retry; the 900-second per-job limit and scientific protocol stayed fixed. Independently checked accounting totals **2,888 GPU-seconds (0.8022 GPU-hours)** across eight allocations, with maximum concurrency four. The reused V15 software profile is bound but not charged twice. [Full allocation ledger](../../outputs/native_aggregation_vlm/v16/step_wrap_study/report_442774/all_user_sacct.psv) · [Preserved infrastructure amendment](../../outputs/native_aggregation_vlm/v16/infrastructure_442749/amendment.json).

V16 motivates addressing nuisance sensitivity and learned-mean deployment together. A future method still needs a separately specified test on fresh data. This diagnostic establishes neither fresh extrapolation success nor reasoning composition; `RESULTS.md` remains unchanged.

# Native value lifting on MMReD Vision: V3 results

2026-09-10. Moving the value nonlinearity before attention pooling **did not
pass the registered two-seed criterion**. PRE improved familiar-count length
extrapolation over POST in seed 0, but matched POST in seed 1. The pooled
difference is smaller than the required gain and cannot replace the per-seed
decision. These results do not establish an overall architectural win or a
general aggregation improvement.

A separate fixed-marginal diagnostic gives a useful positive finding: every
tested model increased its predicted count on all 16 pairs when four matches
were added, despite unchanged character and room histograms. Coarse sensitivity
to the changed joint structure is present; exact count precision remains weak.
The next diagnostic therefore concerns item-level precision and context effects,
not an assumed wholesale absence of binding.

[Verified V3 report](../../outputs/native_aggregation_vlm/v3/REPORT.md) ·
[Canonical analysis and checkpoint provenance](../../outputs/native_aggregation_vlm/v3/analysis.json) ·
[Comparison figure](../../outputs/native_aggregation_vlm/v3/comparison.png) ·
[V2 results and diagnostics](NATIVE_AGGREGATION_VISION_V2_RESULTS.md) ·
[Registered protocols](../../PREREG_AGG.md)

## What V3 compared

Both arms retain the ordinary frozen Qwen2.5-VL-7B attention path. At layer 14,
they add the same learned residual affine lift of the native values, a rank-8
map over the concatenated four KV heads. PRE computes the additional headwise
read as `sum_j alpha_j SiLU(z_j)`; POST computes
`SiLU(sum_j alpha_j z_j)`. Both use the native normalized attention weights and
the same rank-64 outer adapter receiving the read and query hidden state. This
tests a specific nonlinear-value placement hypothesis. It does not introduce a
new sparse-attention family, increase the output dimension, or guarantee
cardinality preservation.

Each arm has **696,896 branch parameters plus 720,896 upper-layer LoRA
parameters: 1,417,792 total**. Paired seeds share initialization and training
order. Training uses the same 180 V2 examples at N8/N16, K0–8, and 72 development
examples. Each run completes nine epochs and 45 optimizer updates per epoch.
Only answer/EOS tokens supervise the native vocabulary head. Evaluation uses
the canonical images-first prompt, 392px images, NF4/bf16/SDPA and unrestricted
greedy cached generation with at most four answer tokens. There are no frame
training labels, external tally or added reasoning tokens.

Checkpoints are selected by pooled development exact accuracy, then lower raw
first-answer-token NLL, then earlier epoch. Selection chooses different epochs
across the four runs; the comparison concerns the registered training-and-selection
procedure, not equal-epoch checkpoints. CPU report 440835 verified the completed
runs against frozen source hashes, manifest identity and selected checkpoint
provenance. The main runs were Slurm array 440830.

## Main results and failed primary criterion

| Arm | Seed | N16, K0–8 | N32, K0–8 | N64, K0–8 | N32/64, K0–8 | Selected epoch |
|---|---:|---:|---:|---:|---:|---:|
| PRE | 0 | 60/108 (55.6%) | 49/108 (45.4%) | 29/108 (26.9%) | 78/216 (36.1%) | 7 |
| POST | 0 | 58/108 (53.7%) | 39/108 (36.1%) | 24/108 (22.2%) | 63/216 (29.2%) | 5 |
| PRE | 1 | 43/108 (39.8%) | 27/108 (25.0%) | 17/108 (15.7%) | 44/216 (20.4%) | 4 |
| POST | 1 | 46/108 (42.6%) | 25/108 (23.1%) | 19/108 (17.6%) | 44/216 (20.4%) | 3 |

V3.1 required PRE to exceed POST by at least five percentage points on pooled
familiar-count N32/N64 accuracy, with no more than five points of N16 loss,
**in each seed**.

| PRE minus POST | Familiar N32/64 gain | Paired-anchor 95% interval | N16 gain | Per-seed screen |
|---|---:|---:|---:|---|
| Seed 0 | +6.94 pp | [−0.46, +14.35] pp | +1.85 pp | Pass |
| Seed 1 | 0.00 pp | [−7.41, +7.41] pp | −2.78 pp | Fail |
| Both fixed seeds, descriptive | +3.47 pp | [−2.31, +9.03] pp | −0.46 pp | Not a decision criterion |

The seed-0 point estimate passes the practical screen while its interval crosses
zero. Seed 1 fails the required effect size. Neither finding establishes
equivalence, and averaging them does not rescue V3.1. The pooled result is
122/432 correct for PRE versus 107/432 for POST: 432 model-example outcomes on
216 unique scenes, not 432 independent scenes.

The bootstrap resamples 108 complete anchors, retaining both lengths and, for
pooled comparisons, both fixed seeds together. It estimates example uncertainty
conditional on these trained models. Two seeds cannot support population
inference about training-seed robustness. Intervals are descriptive and are not
adjusted for the many reported comparisons.

The secondary V3.2 screen passes for **PRE seed 0 versus historical V2 hidden
seed 0**: +6.94 points on familiar OOD, interval [0.00, +13.89], and +9.26 points
at N16. The historical hidden model scored 63/216 and 50/108, respectively.
PRE seed 1 instead scores 44/216, or −8.80 points relative to that historical
model; this comparison is descriptive, not another V3.2 criterion. Selecting
the favorable seed or emphasizing the historical comparison would not establish
a reliable improvement over ordinary hidden-state adaptation.

## Data scope and remaining extrapolation error

V3 reuses the exact V2 clean main and count manifests. The same conditional
generator law supplies all training, development and test examples: exactly K
matching frames, with each negative independently chosen from same-character/
wrong-room, other-character/same-room, or neither, each with probability one
third. This fixes the earlier V1 training/test nuisance mismatch. V3 is a
controlled comparison on this generated evaluation, not the original official
vision split.

The 108 familiar-count anchors contain twelve examples per K0–8 at N16 and are
extended to N32/N64 by interleaving negative frames. The extension changes
distractors, token positions and displayed step indices while preserving the
question and number of matching events. It does not isolate length alone.
The separate 64 count anchors contain eight examples per K9–16 at N32/N64.

V2 test results had already been inspected before V3 was designed. Registering
the V3 comparison does not make those examples untouched: this is **exploratory
test reuse**, and any later efficacy claim needs fresh data and additional
seeds. The four runs each produce 452/452 parseable test answers, so the reported
error is not a consequence of dropping malformed outputs.

Negative-frame extensions still disrupt predictions. PRE seed 0 keeps the same
count on only 32/108 anchors from N16 to N64; 38 of its 60 initially correct
answers become incorrect. PRE seed 1 keeps the same count on 16/108 anchors,
with 32 of 43 initially correct answers becoming incorrect. These are substantial
failures of the desired extension invariance even in the favorable seed.

## Unseen count results

| Arm/seed | K9 correct /16 | K10–16 correct /112 | All K9–16 correct /128 | Count MAE |
|---|---:|---:|---:|---:|
| PRE/0 | 0/16 | 19/112 | 19/128 (14.8%) | 2.094 |
| POST/0 | 0/16 | 4/112 | 4/128 (3.1%) | 3.555 |
| PRE/1 | 0/16 | 17/112 | 17/128 (13.3%) | 3.094 |
| POST/1 | 0/16 | 18/112 | 18/128 (14.1%) | 2.188 |

PRE minus POST is +11.72 points in seed 0, interval [+5.47, +18.75], and −0.78
points in seed 1, interval [−6.25, +4.69]. Pooling the two fixed seeds gives
+5.47 points, interval [+2.34, +8.98], over 64 paired anchors. This positive
conditional pooled contrast is a secondary result; it neither establishes a
benefit across training seeds nor repairs the failed familiar-count primary
criterion. PRE also has worse count MAE than POST in seed 1.

All count outputs parse. K9 uses one answer token and is never answered correctly
here; K10–16 use two numeral tokens and sometimes are correct. Native complete
answers outside adaptation-training support are therefore possible. These
values need not be unseen in backbone pretraining, and occasional success is
not a general counting algorithm. Since all values 10–16 share their first
token, raw first-token NLL is not whole-answer likelihood or count accuracy.

## Fixed-marginal binding diagnostic

[Verified report](../../outputs/native_aggregation_vlm/v3/binding_probe/probe_440886/REPORT.md) ·
[All paired metrics](../../outputs/native_aggregation_vlm/v3/binding_probe/probe_440886/summary.json) ·
[Frozen plan and provenance](../../outputs/native_aggregation_vlm/v3/binding_probe/probe_440886/plan.json)

The post-hoc V2 sampling-law audit showed that global character and room counts
alone predict K statistically. To challenge that explanation, 16 new N32 pairs
fix both queried marginals at C=R=12 and preserve the full character and room
histograms. Four room swaps change eight images, increasing the true conjunction
count from 2 to 6. Every character stays at its original position; the question
and remaining 24 images are unchanged. Data are content-disjoint from the earlier
selected sets. This is an intentional fixed-contingency distribution shift, not
another same-law efficacy test.

GPU 440886 evaluated the six development-selected checkpoints without training.
All 192 model-scene outputs parse, and every model increases its prediction on
**16/16 pairs**, with **0/16 invariant predictions**.

| Selected model | Low K2 correct /16 | High K6 correct /16 | Mean predicted Δ | Mean absolute Δ error vs 4 | Scene MAE /32 | Both answers correct /16 |
|---|---:|---:|---:|---:|---:|---:|
| V2 global/0 | 4/16 | 4/16 | 4.375 | 1.125 | 0.875 | 0/16 |
| V2 hidden/0 | 2/16 | 4/16 | 4.250 | 0.875 | 0.938 | 1/16 |
| V3 PRE/0 | 4/16 | 7/16 | 4.188 | 1.062 | 0.719 | 0/16 |
| V3 POST/0 | 4/16 | 3/16 | 3.750 | 1.125 | 1.000 | 2/16 |
| V3 PRE/1 | 4/16 | 0/16 | 4.812 | 1.312 | 1.531 | 0/16 |
| V3 POST/1 | 1/16 | 2/16 | 4.375 | 1.250 | 1.250 | 0/16 |

Here Δ means the high-K prediction minus the low-K prediction. The responses
cannot be determined only by the unchanged global character/room histograms and
question. The simple oracle marginal moment `0.75(C+R)−0.5N` is always 2: it
gets 16/32 scenes right but 0/16 complete pairs, and is kept outside the model
comparison as an explanatory reference.

The models track the direction and rough magnitude of changed joint structure,
but answering both scenes exactly remains rare. In particular, PRE seed 0 has
the smallest scene MAE while answering both members of zero pairs correctly;
V2 hidden has the smallest absolute Δ error. There is no uniform V3 diagnostic
advantage. These 16 shared pairs give descriptive observations, not 96 independent
pair replications across models. Room swaps also change other associations and
room positions. The diagnostic neither excludes every shortcut nor identifies
an internal binding circuit or an exact aggregation algorithm.

## Implementation cost and attribution limits

Familiar OOD model time is about 1.05–1.07 seconds per example in all four runs;
including preprocessing, about 1.57–1.58 seconds. Peak training allocated memory
is 13.69 GiB for PRE and 13.73 GiB for POST. These are measured implementations
under concurrent cluster conditions, not algorithmic lower bounds.

Equal parameter count does not imply identical activation work. PRE applies the
inner nonlinearity to four KV heads; POST applies it to 28 query-head reads.
PRE recomputes the value transform over cached V at each decode step. The lift
and outer maps run in fp32 while SDPA inputs are bf16, so placement also changes
where rounding occurs. Both arms zero-initialize the outer up map and lift up
map, delaying initial learning of the lift down map. The experiment cannot
attribute every difference to a pure exact-arithmetic ordering effect.

V2's additional-read adapter did not outperform its hidden-only control on the
primary OOD comparison. V3's nonlinear-value placement did not replicate its
seed-0 gain in seed 1. The defensible research question is now which source of
imprecision changes with context and training, rather than assuming that an
extra read or earlier nonlinearity already increases aggregation bandwidth.

## Frame-judgment diagnostic

The [frozen CPU plan](../../outputs/native_aggregation_vlm/v3/frame_judgments/check_440916/plan.json) fixed two models, six selected original frame positions per scene, both binding-pair conditions and isolated/full-context presentations before outcomes. The isolated image retains its original printed Step label and exactly the same indexed question. No training occurs.

[GPU440918](../../outputs/native_aggregation_vlm/v3/frame_judgments/judgments_440918/REPORT.md) completed all768 judgments. Both models achieve192/192 raw first-token Yes/No decisions on isolated images. Full32-frame context reduces this to132/192(68.8%) for frozen Qwen and156/192(81.3%) for V2 hidden. Binary full-vocabulary probability mass exceeds0.999; all top-1 tokens are Yes or No. Pair-bootstrap accuracy losses are31.25pp[23.96,38.54] and18.75pp[14.06,23.44]. Mean gold-signed margins fall7.9643→1.0028 and9.5964→2.7910.

Every decoded output is literally Yes. or No.; the registered strict secondary parser therefore records zero parsed/correct full strings. We retain that result and distinguish it from the prespecified first-token metric. This diagnoses contextual access/reference/interference difficulty on selected indexed questions. It does not identify the computation failing in the original count prompt or justify an inference-time external tally.

## Reasoning-backbone software compatibility

Cosmos-Reason1-7B initially failed offline cache resolution, then lacked two slow-tokenizer assets. [CPU440953](../../outputs/native_aggregation_vlm/v3/reasoning_compatibility/tokenizer_compatibility_440953.json) reconstructed only vocab.json/merges.txt in a new checkpoint-root mirror. Full vocabulary and merge parity,259string/518encoding checks and ordinary processor tensor equivalence passed. Original weights/cache remain unchanged; four shard headers, index and byte sizes were checked, without claiming a full weight checksum.

[GPU440961](../../outputs/native_aggregation_vlm/v3/reasoning_compatibility/lift_pre_seed0_20260910_181849_440961_2877559/summary.json) then completed one tiny training update, checkpoint restoration and eight cached native generations, including N64 and multi-token targets. This is execution compatibility only. A four-token direct-answer software test cannot establish reasoning efficacy or composition; that objective remains open.

V3 consumed7724GPU-seconds(2.14556hours), including profiles, main runs, diagnostics and the failed GPU lookup. The registered main architecture criterion failed. V4 next tests scene diversity with the existing hidden-only control at fixed training steps and inference computation; it does not expand the operator grid.

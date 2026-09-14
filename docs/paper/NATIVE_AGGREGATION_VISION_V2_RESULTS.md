# Native aggregation on MMReD Vision: V2 results

2026-09-10. The clean four-arm comparison did **not** establish a benefit from
giving the middle-layer adapter an additional attention read. On familiar counts
at 32–64 frames, global-read adaptation answered 58/216 examples correctly
(26.9%), versus 63/216 (29.2%) for a nearly parameter-matched hidden-only adapter.
The registered primary attribution criterion failed. Both performed better than
upper-layer LoRA in this run, but this supports ordinary adaptation as an
explanation for the improvement, rather than increased aggregation bandwidth.

[Verified main report](../../outputs/native_aggregation_vlm/v2/REPORT.md) ·
[Full analysis and paired intervals](../../outputs/native_aggregation_vlm/v2/analysis.json) ·
[Comparison figure](../../outputs/native_aggregation_vlm/v2/comparison.png) ·
[Design and interpretation limits](NATIVE_AGGREGATION_VISION_V2_DESIGN.md) ·
[Preregistered protocol](../../PREREG_AGG.md)

## Main comparison and the failed criterion

All four arms adapt frozen Qwen2.5-VL-7B-Instruct in NF4, with the ordinary image
encoder, causal attention, images-first count prompt, native answer/EOS loss and
greedy cached generation. Training uses 180 examples at N8/N16, K0–8; development
uses 72 examples at those same lengths and counts. Each arm trains for nine epochs
with one seed. Checkpoints are selected by development exact accuracy, breaking
ties with raw first-answer-token NLL. There are no per-frame training labels,
external tally or intermediate reasoning tokens in this experiment.

| Arm | N16, K0–8 | N32, K0–8 | N64, K0–8 | N32/64, K0–8 | N32/64, K9–16 | Selected epoch |
|---|---:|---:|---:|---:|---:|---:|
| Global read | 64/108 (59.3%) | 40/108 (37.0%) | 18/108 (16.7%) | 58/216 (26.9%) | 17/128 (13.3%) | 5 |
| Hidden only | 50/108 (46.3%) | 39/108 (36.1%) | 24/108 (22.2%) | 63/216 (29.2%) | 11/128 (8.6%) | 7 |
| Middle + upper LoRA | 43/108 (39.8%) | 30/108 (27.8%) | 17/108 (15.7%) | 47/216 (21.8%) | 9/128 (7.0%) | 5 |
| Upper LoRA | 32/108 (29.6%) | 19/108 (17.6%) | 16/108 (14.8%) | 35/216 (16.2%) | 0/128 (0.0%) | 5 |

The primary V2.1 screen required global to exceed hidden-only by at least five
percentage points on pooled familiar-count N32/N64 accuracy, without losing more
than five points at N16. The observed long-input difference was **−2.3 points**,
with a paired-anchor 95% bootstrap interval of **[−10.2, +5.1]**. The N16 difference
was +13.0 points; that in-range advantage does not rescue the failed primary
long-input criterion. The interval does not establish equivalence or exclude a
modest global advantage: the practical screen failed, while precision remains
limited.

The secondary V2.2 effect-size screen passed: global exceeded middle + upper LoRA
by +5.1 points, interval [−2.3, +12.0], and upper LoRA by +10.6 points, interval
[+4.2, +17.1]. Passing this secondary screen cannot establish the read-specific
mechanism when the hidden-only control matches or exceeds global. These intervals
resample 108 complete anchors, preserving their paired N32/N64 observations.
They describe example uncertainty conditional on one trained model per arm, not
training-seed variability, and are not adjusted for multiple comparisons.

Global and hidden use 1,409,088 and 1,409,120 trainable parameters, respectively,
but different nonlinear widths (64 versus 96). The two LoRA controls each use
1,441,792 parameters. Parameter matching does not match parameterization or input
variance. Adapter learning rate is 1e-3 and LoRA learning rate 1e-4, as registered;
the comparisons include those optimization policies. Global recomputes the
existing dense attention read, so its operation alone adds no new information.

## What the clean data test

V2 uses freshly generated scenes rendered with the canonical MMReD renderer and
one conditional sampling law across train, development and test. For a question
about a character/room pair, exactly K frames match. Each remaining frame is
independently drawn from three equally likely types: same character/wrong room,
other character/same room, or neither. Character and room identities vary. This
is a controlled generated evaluation, rather than reuse of the original vision
test split.

The familiar-count test has 108 anchors at N16, twelve per K0–8, extended to N32
and N64 by interleaving fresh negative frames. Another 64 anchors, eight per
K9–16, form the separate count test at N32/N64. Extensions preserve the question
and matching semantic events; distractor content, token positions and displayed
step indices change. Consequently they measure robustness to this controlled
extension, not an isolated effect of context length or an increased number of
matching items. The count axis supplies the latter distributional challenge.

The [independent data audit](../../outputs/native_aggregation_vlm/v2_clean/data_audit/audit_440739.json)
verified labels, QA/image hashes, content exclusions and extension maps. V1 had
a different same-character distractor distribution in N16 training versus test;
V2 fixes the generator for all arms. V1-to-V2 accuracy changes therefore cannot
be attributed to an architectural change.

Performance still falls sharply when negatives are added. For global, 52 of the
64 anchors answered correctly at N16 become incorrect at N64; six initially
incorrect anchors become correct. Only 19/108 predicted counts remain unchanged
across that extension. Hidden-only also degrades, from 50/108 to 24/108 correct.
All 452 test answers per arm parse as integers, so this failure is not caused by
discarding malformed outputs. Detailed transitions are in the main analysis.

## Unseen counts: native emission is possible, reliable extrapolation is not established

Counts 9–16 were absent from adaptation training. K9 is one answer token; K10–16
require two numeral tokens. The pooled 128-row count test comprises 64 paired
anchors, not 128 independent scenes.

| Arm | K9, N32/64 | K10–16, N32/64 | K9–16 MAE |
|---|---:|---:|---:|
| Global read | 0/16 | 17/112 (15.2%) | 1.83 |
| Hidden only | 0/16 | 11/112 (9.8%) | 2.41 |
| Middle + upper LoRA | 0/16 | 9/112 (8.0%) | 3.57 |
| Upper LoRA | 0/16 | 0/112 | 8.27 |

Global's count-test advantage over hidden-only is +4.7 points, paired-anchor
interval [−1.6, +10.9]. It does not establish a robust count-extrapolation gain.
Global nevertheless correctly emits complete native answers of 10, 11, 13, 14,
15 and 16 on individual examples. The vocabulary decoder is therefore capable
of producing values outside this adaptation set. An absolute claim that it
cannot emit unseen counts would be false here; these values were not necessarily
unseen during backbone pretraining. The unresolved issue is reliable recovery
and use of the count, not a universal prohibition on unseen-number emission.

All count-test answers parse, including the multi-digit outputs. Raw first-token
NLL is insufficient to score these answers: every gold value 10–16 begins with
the same token. Whole-answer exact accuracy and MAE carry the count results above.

## Frozen question-prefix diagnostic

The [prefix diagnostic](../../outputs/native_aggregation_vlm/v2/prefix/20260910_165238_440808_1379431/REPORT.md)
used a predetermined subset of 18 familiar-count anchors, two per K0–8, at N16
and N64. Three conditions each evaluated the same 36 examples: the original
images-first prompt, a correct question prefix plus the unchanged final question,
and a neutral prose prefix plus that final question.

| Condition | N16 | N64 | Both lengths |
|---|---:|---:|---:|
| Original prompt | 4/18 | 1/18 | 5/36 (13.9%) |
| Correct question prefix | 4/18 | 1/18 | 5/36 (13.9%) |
| Neutral prefix | 3/18 | 1/18 | 4/36 (11.1%) |

Correct minus original is 0.0 points, paired-anchor interval [−11.1, +11.1];
correct minus neutral is +2.8 points, interval [−8.3, +13.9]. All 108 outputs parse.
This small fixed diagnostic provides no positive evidence for the tested prompt
repair. It does not show that early query conditioning cannot help a trained
method. Correct and neutral prefixes match actual processor token layout and
image positions, while differing in semantics and repetition. The original
prompt is shorter, so comparisons with it additionally change positions and
length. No prompt was selected or revised from these outcomes.

## Mechanistic diagnostics and remaining scope

The completed [channel-interchange diagnostic](../../outputs/native_aggregation_vlm/v2/channel_probe/channels_20260910_165456_440805/summary.json)
(Slurm 440805) used the development-selected global checkpoint, 16 disjoint N32
pairs with K2 versus K6, both donor directions, and 256 total forwards. Four
frames change within each pair; the other 28 and the question remain identical.
Only the adapter inputs at the last prompt token are interchanged. Ordinary
attention, the residual stream and other branch rows stay on the recipient path.

| Last-query branch intervention | Mean donor-directed logit-margin change | Positive changes /32 |
|---|---:|---:|
| Recipient identity, both channels | 0.0000 | 0/32 |
| Donor h | −0.0020 | 6/32 |
| Donor r | −0.0103 | 7/32 |
| Donor h and r | −0.0029 | 10/32 |
| Donor h, matched to recipient norm | −0.0093 | 8/32 |
| Donor r, matched to recipient norm | −0.0044 | 6/32 |
| Both donor channels, each norm matched | −0.0015 | 8/32 |

The identity intervention preserves the entire vocabulary-logit vector exactly.
The actual swaps yield no positive average shift toward the donor count at this
site. These small effects do not establish that either channel is information
free, or that the branch is unused: earlier query positions and redundant paths
remain intact, and only one recipient row is changed. The 32 directions share
16 pairs, and artificial channel combinations may be off distribution. Input
norm matching also does not match nonlinear branch-output norm. The
[recorded checkpoint and source provenance](../../outputs/native_aggregation_vlm/v2/channel_probe/channels_20260910_165456_440805/config.json)
match the selected global run.

The [frozen recoverability diagnostic](../../outputs/native_aggregation_vlm/v2/recoverability/fit_440792/REPORT.md)
also completed (harvest 440791, CPU fit 440792). It captures h after layer-14 input
normalization and r before the attention output projection at the final prompt
token, using 704 ordinary frozen forwards: 180 training, 72 development and 452
test examples. Each channel has 3,584 features; h+r denotes concatenation.
Training-only feature standardization and target centering precede ridge fitting.
Alpha is chosen separately for each representation by development MAE only.

| Frozen features | Dev MAE /72 | N16 MAE /108 | N32/64 MAE /216 | N32/64 R² | K10–16 MAE /112 |
|---|---:|---:|---:|---:|---:|
| h | 1.042 | 1.425 | 2.189 | −0.111 | 6.562 |
| r | 1.038 | 1.429 | 2.153 | 0.015 | 7.583 |
| h+r | 1.006 | 1.384 | 2.065 | 0.096 | 7.270 |
| N only / training mean | 2.222 | 2.222 | 2.222 | 0.000 | 9.000 |

Both channels contain some linearly accessible count information at familiar
lengths: N16 R² is 0.535 for h and 0.527 for r. That accessibility does not
transfer reliably to longer inputs or larger counts. Concatenating the channels
has long-input rounded exact accuracy 30/216 (13.9%), versus 24/216 (11.1%) for
the constant training-mean control. Every tested ridge representation has 0/128
rounded exact on K9–16. K9 MAE is 3.040, 4.383 and 3.993 for h, r and h+r,
respectively; R² is undefined for that constant-target subset. Adding scalar N
changes the reported MAEs by less than 0.001 and none of the rounded exact
accuracies. Full per-N/per-K metrics are in the
[fit results](../../outputs/native_aggregation_vlm/v2/recoverability/fit_440792/results.json).

All feature-based fits select alpha=1000, the largest preregistered grid value.
With only 180 training examples and thousands of features, this is a limited
linear-probe result, not an optimized bound on decodability. The nearly equal
h/r results provide no strong evidence that r uniquely preserves an accurate
count that h loses. They also cannot locate a loss of character/room binding
between source tokens and the read: source-token representations were not
measured here, and nonlinear codes or other layers may behave differently.
The stored frozen first-token logits are not full-answer generation results.
These diagnostics do not change main checkpoint selection or efficacy criteria.

V2 supports retaining the simpler hidden-only adapter as a serious reference and
withholding the stronger read-bandwidth claim. It has not demonstrated a general
aggregation algorithm, stable count extrapolation, or a benefit that composes
with reasoning tokens. Those remain research objectives rather than established
properties of these runs. All data, checkpoints and execution provenance are
linked from the [V2 execution index](../../outputs/native_aggregation_vlm/v2/INDEX.md).

## Additional sampling-law audit

The subsequent [marginal audit](../../outputs/native_aggregation_vlm/v2/marginal_audit/audit_440844/REPORT.md)
shows why same-law performance alone cannot establish conjunction binding.
Let C and R be the numbers of frames containing the queried character and room
separately. Under the nominal V2 generator, E[C]=E[R]=N/3+2K/3. The fixed oracle
estimate 0.75(C+R)-0.5N therefore predicts K without identifying matching frames.
Its rounded exact is37/216 on familiarOOD and28/128 on unseen counts. These
numbers use oracle marginals and knowledge of the generator; they are not VLM
results or evidence that any model uses this shortcut. Equal marginals can also
correspond to different true conjunction counts, so they do not identify K.

CPU440844 independently checked all704QA labels and semantic counts, with no
image or model-output access. This post-hoc diagnostic leaves the registered
V2/V3 comparisons unchanged and motivates a separate fixed-marginal challenge,
where both-answer correctness and predicted count changes matter more than
pooled example accuracy. The challenge is an intentional sampling-law change.

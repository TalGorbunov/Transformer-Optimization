# Native aggregation on MMReD Vision: V1 results

2026-09-10. The direct vision comparison found a useful native-answer improvement,
but rejected the proposed advantage of processing local reads before merging them.
The simplest global-read adapter reached 40.5% on 32–64 frames, versus 21.5% for
approximately parameter-matched LoRA and 16.0% for the frozen model. The after-merge
mean control reached 40.0%; this one-answer difference does not establish a winner
between those controls. Neither proposed before-merge arm passed the registered
practical screen.

[Full results and paired intervals](../../outputs/native_aggregation_vlm/v1/REPORT.md)
· [Figure](../../outputs/native_aggregation_vlm/v1/comparison.png)
· [Machine-readable analysis](../../outputs/native_aggregation_vlm/v1/analysis.json)
· [Protocol](NATIVE_AGGREGATION_VISION_V1.md)

## What the comparison establishes

All arms use Qwen2.5-VL-7B, actual images through its normal encoder, ordinary
vocabulary answers, the same training examples, nine epochs and development-only
checkpoint selection. There is no external count, per-frame supervision or
intermediate reasoning trace. Train lengths are 8/16 frames; test lengths 8/16/32/64,
100 examples each. Counts remain 0–8.

The matched mean contrast favors processing after merging: 40.0% versus 30.5%, a
9.5-point difference; the paired interval for before-minus-after is [−15.5, −3.5]
points. The sum contrast ties at 6.5%. The originally proposed processing-order
advantage therefore failed in this setup. All failed criteria remain recorded in
[PREREG_AGG.md](../../PREREG_AGG.md). No further training expansion was launched.

Global improves over LoRA by 19 points (supplementary paired 95% interval [12.5, 25.5]),
and from 42.0% to 62.5% within the training length range. Its long-input generation-policy
NLL is 1.783, compared with 2.335 for LoRA and 3.847 for post_mean. Thus the two strongest
controls are close in exact accuracy but differ substantially in policy likelihood.
The experiment has one training seed; the intervals describe paired-example
uncertainty and are not independent-seed confirmation or multiplicity-adjusted.

## Interpretation and the next decisive control

The global adapter computes U*SiLU(W_r*r_SDPA+W_h*h+b) at decoder layer 14. Its read
is the existing globally aggregated attention statistic. Computing the same read
blockwise adds no new information. The gain could come from nonlinear adaptation
at this middle layer, the additional read input, or the placement of trainable
capacity relative to the upper-layer LoRA. It does not establish extra aggregation
bandwidth. The hierarchical control is not a reproduction of HiLS. See the
[literature positioning audit](NATIVE_AGGREGATION_POSITIONING.md).

The smallest useful next attribution experiment keeps upper-four-layer LoRA rank 8
and compares the current global adapter with two ordinary controls:

| Middle-layer intervention | Additional parameters | Question |
|---|---:|---|
| Global-read adapter, rank 64 | 688,192 | Current reference |
| Hidden-only bottleneck, rank 96 | 688,224 | Does access to the extra read matter? |
| Q/K/V/O LoRA at layer 14, rank 32 | 720,896 | Does adaptation placement explain the gain? |

The hidden-only control should use the same normalized hidden input and injection
point. These are proposed controls, not implemented or run results. Match data,
training, selection and seeds. If they explain the global result, retain the useful
engineering improvement and retire the stronger aggregation-mechanism claim.

## What the diagnostics add

The [checkpoint inspection](../../outputs/native_aggregation_vlm/v1/inspection/INDEX.md)
used only four disjoint profile examples per model. AtN64 the sum branch was about
47.7 times the ordinary attention output norm, versus 1.17 for global. Both sum-scaled
arms had large zero-read and opposing read-dependent terms. Normalized branches
remained near the scale of ordinary attention. The first-step gradient norm was
48.27 for sum,47.68 forpost_sum,1.32 for hierarchical and 0.57 for LoRA, before joint clipping.
These observations motivate scale/optimization diagnostics; they do not prove cause.
The zero-read component uses a contextual hidden state that can already carry evidence.

The sum failures include decoding instability: at N64 sum produced no fully parseable
answers andpost_sum only five. A post-hoc leading-character digit check gives 12/100
for both; this neither replaces the exact metric nor supports an aggregation gain.
The normal cache and causal behavior passed unit checks, but reasoning-token
composition was not tested.

## Limits and execution

The [data census](../../outputs/native_aggregation_vlm/v1/data_audit/INDEX.md) verified
all 664 main/profile labels and excluded duplicate content. It also found that N16
training/dev have more same-character distractors than test:10.31/10.28 mean target
character frames versus 6.12. Rendering parity did not imply identical sampling laws.
These are combined length/nuisance-shift results. A clean follow-up should use one
audited generator across splits, paired distractor extensions, and separate larger
N at familiar K from unseen K at feasible fixed N. This is necessary before claiming
improved capacity to aggregate more relevant items.

Generation inherited repetition_penalty=1.05. The saved NLL and dev tie-break are
policy quantities, not raw LM probabilities. The global control currently recomputes
its read redundantly; it could reuse the existing attention result. Concurrent-job
latencies are descriptive, not a compute-optimal comparison.

All 23 native-aggregation checks and existing regression suites passed. Main array
440504 ran up to four B200 s concurrently; all jobs completed. Total GPU allocation
was 3.53 hours, including profiling, the frozen reference and the inspection. Data
and checkpoints use the requested /mnt/data/gabriele/gnn_transformer and
/mnt/ckpts/gabriele/gnn_transformer roots. [Execution record](../../outputs/native_aggregation_vlm/v1/execution.json).

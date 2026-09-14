# Cosmos reasoning on MMReD Vision: 4096-token extension

2026-09-10. **The registered five-percentage-point screen passes by two extra
correct OOD answers: 6/36 with reasoning versus 4/36 with direct answers.**
The longer budget substantially reduces truncation, but exact counting remains
weak. This is a small exploratory policy comparison, not evidence that reasoning
reliably solves aggregation or that our aggregation methods compose with it.

[Verified extension report](../../outputs/native_aggregation_vlm/reasoning_long/report_441457/REPORT.md) ·
[Analysis and provenance](../../outputs/native_aggregation_vlm/reasoning_long/report_441457/analysis.json) ·
[Raw generated trajectories](../../outputs/native_aggregation_vlm/reasoning_long/main/cosmos_441455/predictions.jsonl) ·
[Original baseline report](../../outputs/native_aggregation_vlm/reasoning_baseline/report_441252/REPORT.md) ·
[Registered protocol and amendments](../../PREREG_AGG.md)

## What changed

The original frozen Cosmos-Reason1-7B assay truncated 28/36 N32/N64 reasoning
responses at 512 tokens. This extension changes only the reasoning output cap
to 4096. It preserves the exact model compatibility mirror, NF4/bf16/SDPA
execution, 392px images, native chat template, system/user text, greedy decoding,
repetition penalty 1, and strict answer/EOS rule. The audited direct32 baseline
is reused without regeneration. No adapter, training, external tally, oracle
scratchpad or prompt search is involved.

The chosen cap follows the pinned official model card's recommendation of at
least 4096 output tokens. Its documented BF16 testing and sampled example differ
from our quantized greedy policy, so these results should not be presented as
an assessment of every supported Cosmos configuration.
[Official model card, fixed revision](https://huggingface.co/nvidia/Cosmos-Reason1-7B/blob/3210bec0495fdc7a8d3dbb8d58da5711eab4b423/README.md)

The 54 scenes comprise 18 anchors, two per gold count K0–8, each rendered at
N16/N32/N64 under the clean V2 generator law. Longer versions interleave negative
frames while preserving the queried character/room and number of matches.
Negatives include the target character in a wrong room and other characters in
the queried room. Positions and displayed Step labels change. This is a
controlled generated MMReD-style vision evaluation, not the original official
vision split. These scenes were fresh for the initial assay, but had already
been inspected when this budget extension was registered: **the extension
reuses test data adaptively**.

Each scene receives one natural 4096-cap generation. The 128/512 endpoints are
censored views of that same trajectory. Correctness requires the complete
prescribed think/answer structure, a single integer answer, and native EOS by
the relevant budget. Every malformed or truncated response stays incorrect;
there is no extraction of the last number from an unfinished trace.

## Complete results

| Policy / endpoint | N16 exact | N32 exact | N64 exact | N32/N64 exact | OOD completed and parsed | OOD truncated | Parsed OOD MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Existing direct32 | 4/18 | 2/18 | 2/18 | 4/36 (11.1%) | 36/36 | 0/36 | 5.139 (n=36) |
| Original reasoning512 | 4/18 | 0/18 | 1/18 | 1/36 (2.8%) | 8/36 | 28/36 | 7.000 (n=8) |
| New trace, censored128 | 0/18 | 0/18 | 0/18 | 0/36 | 0/36 | 36/36 | Undefined |
| New trace, censored512 | 4/18 | 0/18 | 1/18 | 1/36 (2.8%) | 8/36 | 28/36 | 7.000 (n=8) |
| New reasoning4096 | 5/18 | 3/18 | 3/18 | 6/36 (16.7%) | 32/36 | 4/36 | 6.781 (n=32) |

Reasoning4096 minus direct32 is **+5.56 percentage points** on familiar-count
N32/N64 accuracy, with a paired-anchor 95% bootstrap interval of
**[0.00, +13.89] points**. The preregistered descriptive screen requires at least
+5 points and therefore passes. The interval touches zero; a threshold screen
on two additional answers is not strong statistical evidence of a general
benefit. The bootstrap uses 10,000 draws of the 18 complete anchors, retaining
both OOD lengths, with seed 20260917. It describes example uncertainty for this
fixed model and policy, not variability across training seeds or models.

N16 improves by one answer, from 4/18 to 5/18: +5.56 points, interval
[−11.11, +22.22]. N16 is reported separately and is not an additional pass/fail
guard in this extension.

The main effect of extending 512 to 4096 is completion: OOD parsed responses
increase from 8 to 32, and correct answers from 1 to 6. All completed responses
parse successfully. Nevertheless, 26/36 OOD responses complete with an incorrect
count, and four remain truncated. Across all lengths, 49/54 complete and parse;
five reach the 4096 cap without completion.

The retained per-K diagnostics constrain the interpretation further. Both
policies correctly answer all four OOD K0 scenes. The two additional exact
reasoning answers are one N32/K4 scene and one N64/K6 scene. Thus reasoning is
correct on only 2/32 nonzero-count OOD examples, versus 0/32 for direct answers.
This is a descriptive support breakdown, not a replacement primary metric.

The reported parsed OOD MAE is higher under reasoning, 6.781 versus 5.139,
although its denominator excludes four truncated cases and is therefore a
different subset. At N64, reasoning's parsed MAE is 10.2 over 15 completed
responses. The small exact-answer gain should not be described as uniformly
better count precision. Extension stability also remains weak: reasoning is
correct at both N16 and N64 on only 2/18 anchors, the same count as direct;
three of its five correct N16 answers become wrong at N64.

## Prefix comparability and verification

**All 54 new trajectories exactly preserve their original reasoning512
prefixes.** Every censored512 endpoint object equals its original counterpart,
and all 22 originally completed trajectories remain identical in full. Therefore
the longer responses empirically continue the same deterministic traces rather
than benefiting from different early generations. No prefix mismatch, scene or
run was removed or selected. This supports the specific interpretation that
allowing these existing trajectories to finish changes their scored outcomes;
it does not show that every added reasoning token improves an internal count.

CPU plan 441453 bound all prepared input IDs, model/processor metadata, native
EOS IDs, original baseline artifacts, source hashes and the 18,000-token total
context ceiling. Software profile 441454 completed naturally after 858 tokens
and passed its source/cache/one-vision-forward checks; it did not exercise a full
4096-token decode. Main 441455 completed all 54 scenes. CPU report 441457
independently rescored raw IDs, checked grammar and EOS, recomputed metrics,
verified every prefix comparison, and checked cache lengths and timing totals.
A separate read-only audit of the completed files confirmed the main
summary/predictions/plan hashes, all original prefixes and endpoint objects, and
the generated-token total. No verification failure was found.

The frozen extension plan SHA256 is
`74f20080553eb59716b2af04a046de45e4f6b120e1fb9b9ec13fabd61106c958`;
all other source, baseline and output hashes are in the linked analysis.

## Cost and implication for aggregation research

The main generated **66,888 tokens**, averaging 1,238.7 per scene, and measured
1,705.8 seconds of generation. Total measured runtime including model loading
was 1,755.3 seconds. The reused direct baseline generated 493 tokens across the
same 54 scenes, with 52.6 seconds of generation. Thus the recorded reasoning
policy uses about 136 times as many output tokens and 32 times as much generation
wall time. The timings come from separate jobs and are descriptive, not a
controlled throughput benchmark. No independent runtime is inferred for the
censored128/512 endpoints.

Slurm allocated 1,769 GPU-seconds to the main plus 49 to the profile:
**1,818 seconds, or 0.505 GPU-hours**, within the separate two-hour extension
budget. The main actually exercised five full 4096-token paths; its maximum
output sequence length was 16,854 tokens, with native cache length 16,853,
within the fixed 18,000 ceiling. Peak allocated GPU memory was approximately
8.3 GiB.

This assay supplies limited positive evidence that allowing a frozen reasoning
policy more time can recover some exact aggregation answers. Its low absolute
accuracy, small effect, remaining truncation and large compute cost leave the
user's broader premise unestablished. The direct-versus-reasoning contrast also
changes the system policy, so it is not the isolated causal effect of adding
tokens. Cross-model differences from the earlier Qwen baseline would additionally
confound pretraining and post-training.

A composition claim requires a separate, fresh, matched comparison of the
proposed aggregation method and its control under both direct and reasoning
policies, with fixed budgets and complete truncation accounting. This frozen
baseline contains no improved aggregation method and cannot establish that
composition on its own.

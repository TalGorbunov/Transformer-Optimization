# Prospective fresh comparison of privileged continuation losses

This protocol is written before any prediction on the new cohort and before
the six continuation fits finish. The scientific question is whether teaching
an existing native state to retain cumulative facts improves aggregation beyond
matched answer training and beyond teaching local facts. It is a diagnostic
study of learnability and use, not a new architecture or a novelty claim for
privileged supervision.

Use the frozen complete cohort from CPU447629 and its main1400 population.
Each main question has a distinct world. Evaluate the original competent
ordinary checkpoint and all final answer/local/prefix continuations, seeds25
and26. No checkpoint, seed, question, loss weight or length selection follows
from these predictions. The two continuation seeds share one original fitted
backbone; they are not independent pretraining or initial finetuning replicates.
Keep all outcomes, including invalid answers and resource/software failures.
Every checkpoint entering a contrast must have complete, numerically verified
predictions for its full fixed population. Missing predictions due to software
or resource failures are unavailable, not incorrect answers. They cannot be
dropped into a smaller complete-case analysis; hold that contrast until its
fixed population is complete under a separately recorded recovery, if any.

The primary outcome is complete native answer correctness on the400 N32
partner/room aggregation questions:100 each in partner-MOST, partner-LEAST,
room-MOST and room-LEAST. Use the unchanged typed JSON answer scorer, greedy
generation, native EOS and50-new-token limit. Malformed or truncated answers
are incorrect. Exact answers count once per question, without partial credit.

The primary contrast is prefix minus answer continuation. For each question,
average the two paired correctness differences across continuation seeds, then
average the400 question values. Report percentage points and each seed's
separate paired difference. Obtain a two-sided95percent percentile interval
from10000 bootstrap draws of worlds within the four fixed100-world strata,
using NumPy default_rng(20260915). A world draw carries all model/seed outcomes
together; do not resample models or treat the two seeds as800 independent
worlds. Use the2.5th and97.5th percentiles with numpy.quantile method='linear'
(type7), and one shared fixed draw inventory for both ordered contrasts.
The interval describes world-sampling uncertainty conditional on these
checkpoints, seeds, strata and cohort generator. It does not measure uncertainty
over training datasets, architectures or unseen tasks.

Only if the primary interval's lower endpoint exceeds zero, test prefix minus
local using the identical paired/stratified procedure and the same bootstrap
index draws. This fixed sequence protects the two ordered superiority claims
at the nominal level under the bootstrap approximation. Report both intervals
regardless, but call the second contrast descriptive unless the first gate
passes. A claim that cumulative teaching helps in both observed continuations
also requires a positive point difference in each seed, checked separately
for each claimed contrast. A statistically positive small effect is reported
at its actual size; no practical-impact threshold is
retroactively selected.

Report local minus answer, all continuations versus the original checkpoint,
N8/N16 results, individual tasks/directions and the200 N32 counting/retrieval
controls descriptively. Publish denominators, correct/invalid/truncated counts,
paired win/loss tables and both-seed values. Do not claim that nonsignificance
establishes equivalence or that high local probe accuracy proves an aggregation
capacity limit. The960 diagnostic questions remain a separate intervention
population and do not enter the primary effect or its interval.

All seven evaluations use the original images and prompts, the full native
decoder and head, and no supervision projection, oracle features, added hidden
states, reasoning tokens or auxiliary modules. There is one native prefill;
each subsequent answer token uses the normal cached decoder. This is unchanged
computation per decoding step, not a claim that an entire multi-token answer
requires only one model call. Record prompt/image/generated-token counts,
actual model/vision/head/layer calls, peak GPU memory and allocation seconds.

The central display is aggregation accuracy against inference work. At fixed
world and generated-token budget, all arms have the same native architecture
and input lengths; the train-only loss adds zero deployed operations. Report
actual token/call work as well because response lengths can differ. Visual
features are computed once and shared operationally across methods. Charge
their native per-world visual workload to each method's logical end-to-end
inference accounting, while recording the actual shared preparation allocation
only once in the experiment's total resource ledger. Keep training and failed
attempt costs visible separately. Instrumented captures and serialized audits
are not production latency measurements; do not turn their elapsed time into
a deployment speedup claim.

A positive result would justify the separately registered causal state-use
study. It would not yet show a structural capacity deficit, generalize the
hand-specified40 fact channels beyond MMReD, or establish composition with
reasoning models. Those require additional evidence and remain open objectives.

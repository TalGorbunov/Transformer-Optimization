# Native aggregation in plain and reasoning models

2026-09-08. Current proposal after the user's clarification. This supersedes the
external numerical-readout and affine-state-machine recommendations in the two
earlier design notes. Implementation and pilot results are recorded in
`NATIVE_AGGREGATION_IMPLEMENTATION.md` and `PREREG_AGG.md` (P0/P1). The initial
P0 sum configuration failed its native extrapolation criteria.


**Positioning update (2026-09-10):** The literature audit in
[NATIVE_AGGREGATION_POSITIONING.md](NATIVE_AGGREGATION_POSITIONING.md) supersedes
the earlier novelty framing below. This is a conditional Deep Sets-style adapter
over grouped attention reads. HiLS and Landmark Attention are direct attention
antecedents; conditional nonlinear summation is established. The next scientific
gate is an advantage over processing-after-merge and learned hierarchical fusion,
with normalization and compute controlled. No vision/reasoning gain is established.

## Objective

Improve the model's own aggregation within each forward pass, through its
existing language head. The same operation must be available at every subsequent
reasoning token. Three claims need separate evidence:

1. Better direct answers with no intermediate reasoning tokens.
2. Better reasoning-model answers at the same reasoning-token budget.
3. Improved aggregation at later reasoning steps, beyond just making the first
   intermediate answer more accurate.

External tally accuracy and decodable latent statistics are diagnostic references,
not success on these criteria. The original working bind-then-count remains a
reference. The user has clarified that a separate numerical head does not meet
the central objective.

## Hypothesis

A decoder may mix several pieces of evidence before performing enough
query-conditioned nonlinear computation on each. Separate local reads, transformed
before being merged, may preserve more usable evidence within one forward pass.

This is an empirical hypothesis, not an impossibility theorem about transformers,
which already have multiple heads, contextual values, and nonlinear layers.
Bandwidth is an operational hypothesis about reliably usable aggregate information,
not an established capacity increase. Compare fixed backbone depth and reasoning
budget, and separately match measured computation. The added update has architectural
rank at most 64; neither more local evaluations nor larger vector norm proves more
information capacity. A count in 0..K only needs log2(K+1) reliable bits.

## Primitive: local read, nonlinear computation, merge

At one middle decoder layer retain ordinary causal attention and add a small
branch. Partition visible keys/values into fixed contiguous token regions B_b.
At every causal position t:

    r_tb = Attention(q_t, K[B_b], V[B_b])
    m_tb = phi_theta(U_h h_t, U_r r_tb)
    delta_h_t = W_up sum_b m_tb
    h_t <- h_t + delta_h_t

The current hidden state supplies the query. The shared nonlinear message
network outputs a vector without prescribed predicate or numeric coordinates.
No question parser, Yes/No prompt, oracle frame labels, or task-specific reducer
is part of this operation. The existing language head generates the output.

First use the layer's ordinary Q/K/V scores, normalized separately within each
region. Concatenate head outputs within a region before the small message
network. Initialize W_up to zero and the message network nonzero, preserving
base-model behavior initially while allowing gradients. Do not zero both a
multiplicative gate and its output projection.

Train the branch with ordinary next-token cross-entropy, with a small identical
upper-layer LoRA allowance in all trained controls so the native decoder can
learn the interface. This does not require full backbone training. Keep the
native vocabulary head and output format. Reasoning-mode training can include
correct reasoning traces; record and match that supervision across controls.

The sum merge is a testable choice, not a conclusion. Include mean merging.
Also account for the fact that an affine local map with a bias can encode the
number of regions without learning nonlinear evidence processing.

For vision, frame-aligned regions provide a favorable diagnostic. Generic claims
require fixed token regions and robustness to block-size and boundary changes.
A block can contain too much evidence, and a boundary can split a useful fact.

## Why the benefit can be reused

Each reasoning token has a new query and makes a fresh collection of local reads.
If a previous step identifies a person, the next query can use that identity
while aggregating evidence about the person's location. Learned messages enter
the residual before later decoder layers, whose ordinary KVs then carry the
modified current-token representation into future computation.

The distinction from the June DeepSets adapter is specific: token-dependent
reads from the actual token cache, local nonlinear processing, and native
integration at every decoding position. The earlier adapter used pooled frame
states and answer-position readout/injection. This difference does not resolve
the native-readout risk by assertion; that remains the first empirical gate.

## Causality and compute

Use Q/K/V already available at the chosen layer in the current traversal. Do not
use final-layer carrier states to modify an earlier layer of the same prefill:
that would silently require another backbone traversal.

Past cache entries remain unchanged. The current query reads them again each
step. Empty or future regions contribute nothing; a partially visible region
contains only positions <= t. Define blocks identically in full prefill and
incremental decoding. Verify numerical agreement of those execution modes with
the branch enabled, and verify that future-token changes cannot affect earlier
logits.

For context length T and region size W, the branch processes about T/W regions
per decoding query. Attention work has the same asymptotic context dependence
as an ordinary attention layer, with extra nonlinear work per region. Retaining
the original path may add another cache read if scores cannot be reused.
Prefill still has quadratic attention work. Materializing every token-by-region
message can consume substantial memory; use tiled execution and profile actual
latency, FLOPs, and memory. One forward does not mean the same amount of compute.

The initial implementation retains ordinary positional scores and contextual
queries. It does not guarantee invariance to distance or input length. The AS3
results already warn that the reader itself can degrade with length. Diagnose
that before adding a positional intervention.

## Exact linear reference

For one attention head let z_tb = logsumexp of its logits within region B_b.
Then, with identical masks and positions:

    global_attention_t = sum_b softmax_b(z_tb) * r_tb.

This identity holds per head. It provides an exact segmented implementation
control: partitioning alone need not change ordinary attention. The treatment
changes local nonlinear processing and/or the merge weights.

## Experimental sequence

Pre-register exact cells, predictions, supervision, selection rules, and resource
budget before GPU work.

### 1. Native-output feasibility

Begin with clearly identifiable symbolic evidence, then MMReD Vision. Train on
N<=32 and test longer N with K still within 0..8. Score ordinary model-generated
answers. The relevant baseline is native VLM accuracy, not the external .97-.99
judge tally. Separately test K beyond training support: numeral extrapolation
and resistance to more distractors are distinct challenges.

The native output requirement is essential. The existing late tally and learned
injection failures show that a correct numeric feature does not ensure the model
can use it. A middle-layer trainable interface is a hypothesis worth testing,
not a guaranteed solution.

### 2. Mechanism controls

Hold local inputs, layer, data, native head, and upper-layer adaptation fixed:

- Linear versus nonlinear local messages, crossed with mean versus sum.
- Nonlinear processing before versus after merging. Report parameter-matched and
  separately compute-matched comparisons: one global MLP naturally executes less
  work than an MLP applied to each region.
- Ordinary added attention heads and LoRA controls with measured parameter and
  compute budgets. Multiple heads already provide parallel reads.
- Exactly reconstructed global attention and the unchanged base model.

Evaluate counts, comparisons of aggregate quantities, and whether all queried
conditions hold. Use one module and native output objective across tasks.
Include held-out entity/room combinations and paraphrases. Native exact match,
MAE/bias for numeric answers, and token likelihood are behavioral metrics;
per-item probes are diagnostics.

### 3. Plain and reasoning modes

Evaluate direct answers and reasoning at several fixed generated-token budgets.
Use the same architecture; weights need not transfer between different backbones.
A model with both thinking and non-thinking modes helps control model differences.
The current local resources provide Qwen2.5-VL for direct vision and Qwen3-8B for
text reasoning. Those results alone would not establish gains in a vision
reasoning model; that requires an actual vision reasoning backbone.

A useful evaluation table is:

    model/mode             base   trained control   aggregation branch
    direct native answer
    reasoning, short budget
    reasoning, long budget
    same correct intermediate prefix, next aggregation

Compare accuracy at fixed tokens and at matched total compute/latency where
possible. The branch adds work per token. Plot accuracy against both budgets.
Do not infer general improvements merely from a shorter trace.

### 4. Reuse and composition

Controlled task: identify which person appears most often in the kitchen, then
count that person's garden appearances. Define ties in advance. Both stages
require aggregation, and the second query depends on the first result.

Compare the same trained model with the branch disabled, prefill-only,
decode-only, and enabled throughout. These are inference ablations, not
substitutes for separately trained controls. Also give both versions the same
correct intermediate text and test the next aggregation. A current-step on/off
intervention with the same inherited KV cache can isolate immediate branch use.

Report stage-conditional accuracy, final accuracy, tokens, latency, and memory.
For a stronger transfer test, train component aggregation skills and withhold
their multi-stage combinations. Report this separately from training directly
on the combined task.

Better conditional performance at later steps supports reuse. Increasing absolute
accuracy gains monotonically with reasoning length is not required or promised:
ceilings and error propagation affect those curves.

## Decision rules and limitations

If only an external probe improves, the objective is unmet. If gains occur only
at the first prediction, do not claim repeated-use benefits. If equal-compute
extra heads match the result, use the simpler intervention and narrow the claim.
If perfect segmentation is necessary, state that scope.

Local means can still dilute evidence, irrelevant-region messages can accumulate,
a sum can dominate the residual, and later normalization can obscure magnitude.
A small local MLP may not replace the full per-frame computation that made the
fenced judge strong. These are actual design risks, not reasons to defer testing.

The method is not a universal replacement for reasoning tokens. The target is
more useful aggregation per token, available repeatedly as reasoning proceeds.

## Prior art and contribution boundary

The [2026-09-10 audit](NATIVE_AGGREGATION_POSITIONING.md) gives the exact operation
comparisons, source sections and missing controls. The closest lineages are:

- [Deep Sets](https://arxiv.org/html/1703.06114v3): conditioned nonlinear instance
  maps followed by summation, including aggregate-supervised visual sum extrapolation.
- [Landmark Attention](https://arxiv.org/html/2305.16300v2) and
  [HiLS](https://arxiv.org/pdf/2607.02980): within-block attention and learned
  normalized merging; native language-model integration is established.
- [Native Sparse Attention](https://arxiv.org/html/2502.11089v1): nonlinear block
  compression and gated combinations of attention branches.
- [Cardinality-Preserved Attention](https://www.ijcai.org/proceedings/2020/0194.pdf)
  and [Sigmoid Self-Attention](https://arxiv.org/html/2409.04431v1): antecedents for
  multiplicity-sensitive or independently weighted aggregation.

Relation Networks, Set Transformer, hierarchical language models, and residual
adapters provide additional context. GATv2's original-GAT theorem does not apply
as a blanket limitation of transformer attention.

The possible contribution is an experimentally established benefit from processing
local evidence before merging it in a pretrained causal decoder, beyond stronger
attention, normalization, capacity and compute controls. Its utility for native
vision extrapolation and later reasoning steps requires separate demonstrations.
The current implementation and text pilot do not yet establish that contribution.

# Native aggregation: prior art and revised research question

> Historical prototype audit. Its implementation equations and rank64 discussion refer to the earlier grouped-attention prototype. The current complete-local-stream rank96 architecture and verified V8/V9 findings are described in [the current research claim](NATIVE_AGGREGATION_WORKING_STORY.md). The prior-art cautions still apply; later results do not make the established pooling or attention ingredients new.

2026-09-10. Literature and implementation audit prompted by Gabriele's comparison
to HiLS. Three independent agent reviews informed this assessment. This is a
research-design correction, not a new experimental result or pre-registration.

The prototype combines established grouped attention and conditional sum pooling
inside a pretrained decoder. Its architectural novelty is currently weak. The
potential contribution is an experimentally supported explanation of aggregation
failure and a useful intervention whose benefit survives stronger controls.
Changing the name or target benchmark does not establish that contribution.

## The closest antecedents

| Prior work | Established ingredient | Consequence for this project |
|---|---|---|
| [Deep Sets, 2017](https://arxiv.org/html/1703.06114v3), §3.1, §4.1.2 | Nonlinear instance maps followed by summation and decoding; explicit conditioning on additional information; text/image digit-sum extrapolation using aggregate labels. | Conditional map-then-sum, answer-only supervision, and visual aggregation extrapolation are not new ideas. Our elements are query-dependent decoder block reads. |
| [Landmark Attention, 2023](https://arxiv.org/html/2305.16300v2), §3.1, Eqs. 1–4 | Separate within-block normalization and learned normalized block weighting. | Independent local attention reads are established. |
| [HiLS, 2026](https://arxiv.org/pdf/2607.02980), §3, Eqs. 7–11; §4.1/4.3 | Learned chunk summaries, top-K selection, locally normalized reads and normalized inter-chunk fusion; low-rank query calibration and frozen-base adaptation. | Hierarchy, native output, LM loss, low-rank adaptation, and one-forward execution cannot carry our novelty claim. |
| [Native Sparse Attention, 2025](https://arxiv.org/html/2502.11089v1), §3.2–3.3, Eqs. 5/7 | Gated compressed, selected, and sliding-window attention branches; nonlinear block compression before attention. | Broad claims about nonlinear local processing plus global attention are already covered. Its block compression is query-independent; our map follows a query-specific read. |
| [Cardinality-Preserved Attention, IJCAI 2020](https://www.ijcai.org/proceedings/2020/0194.pdf), §3–4 | Analyzes normalized-aggregation collisions and adds feature sums or neighborhood-size scaling. | Preserving multiplicity by changing normalization is established and demands a simple scaling/additive control. |
| [Sigmoid Self-Attention, 2024 preprint](https://arxiv.org/html/2409.04431v1), §2–3 | Elementwise sigmoid scores without row-softmax competition; length-dependent calibration. | Independent contribution weights are established. Sigmoid is not automatically length-invariant or count-preserving. |

Deep Sets is especially important: its local map can already be conditioned on
meta-information (§3.1), and its MNIST sum task uses no individual digit labels
(§4.1.2). A successful application to MMReD would still need a substantive native
decoder finding. This is also related to the project's retired Deep Sets adapters;
using current-query cache reads is a concrete implementation change, not proof
that the older research question has been resolved.

Other relevant families are [Relation Networks](https://arxiv.org/abs/1706.01427)
(learned relational messages), [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html)
(attention-based set processing), and [Hierarchical Self-Attention](https://arxiv.org/html/2509.15448v1)
(hierarchy-constrained attention, including pretrained-model use). These should
organize the related-work discussion around representation, selection, and
reduction rather than become a list of allegedly inferior methods.

The empirical and native-output setting has precedents too.
[Can Vision-Language Models Count?](https://arxiv.org/html/2511.17722v3) studies
controlled visual counting and attention interventions; it is relevant to any
claim about diagnosing VLM counting failures.
[Set-LLM](https://arxiv.org/html/2505.15433v1) uses attention masks and positional
changes for permutation-invariant native LLM answers, particularly relevant to
our older fencing/reset construction. The [ILSE work](https://arxiv.org/html/2603.22665v3)
aggregates across layer representations for downstream tasks; it is adjacent to
representation adapters, with a different output interface. None of these
comparisons licenses a claim that this exact decoder integration is first.

## Exact comparison with HiLS

Suppressing heads and treating the local window as another group, its core merge is

    r_tb = softmax(within-block query-key scores) @ V_b
    o_t  = sum_b alpha_tb r_tb

where distant groups use learned mass surrogates and the local window uses exact
attention mass, jointly normalized (paper Eq. 10). HiLS also evaluates mathematics, variable tracking and multi-document tasks; describe its
scope accurately rather than calling it retrieval-only. Its calibration and
positional choices matter to extrapolation. The [official implementation](https://github.com/Tencent-Hunyuan/HiLS-Attention/blob/main/models/FlashHiLS/hils_attention.py)
also supports a zero-sink normalization variant.

Our implementation instead concatenates the head reads within each block:

    a_tb      = W_r concat_heads(r_tb) + W_q h_t + c
    delta_h_t = W_up sum_b SiLU(a_tb)
    output_t  = ordinary_attention_output_t + delta_h_t

The specific changes are a query-conditioned, cross-head nonlinear map before
reduction; an unnormalized sum over every visible block; and retention of the
original dense path. These distinguish the operations but do not establish a new
attention family. The sum map itself has the conditional Deep Sets form.

In this code every query still scores every visible key. Blocks change
normalization and processing order; they do not remove attention edges. Prefill
remains quadratic and the added branch increases work. The text pilot measured
approximately 1.5× long-input generation latency versus its LoRA control.

The existing exact reconstruction reference is useful:

    global_read_t = sum_b softmax_b(log Z_tb) r_tb

This equality holds separately for each attention head with the same masks and
positions. Merely evaluating attention in blocks need not change its function.

## What the current evidence cannot establish

- **More information bandwidth.** Before output dtype casting, the added residual is W_up z
  with z in R^64, so it lies in a subspace of dimension at most 64. Separate local computations
  may provide better task statistics, but do not literally widen that output.
  Measure accuracy against evidence load, generated tokens, and actual compute.
- **A general softmax counting impossibility.** Uniform replication leaves a
  normalized weighted average unchanged only under the appropriate fixed-score,
  fixed-value assumptions. Contextual states, positions, a fixed sink, other
  layers, and generated tokens can carry additional information. The GNN result
  above does not prove a positional transformer cannot count.
- **Recovery of information already lost inside a block.** Equal local reads
  produce equal local messages for a fixed query. The outer network cannot undo
  such collisions, and our own local softmax can discard multiplicity.
- **Reliable selective accumulation.** Summing arbitrary block messages can
  accumulate distractor error or a query-only bias. For an affine local map,
  the sum includes R(W_q h+c), where R is the number of visible blocks. A gain
  could reflect length calibration rather than processing evidence better.
- **A universal set-function theorem for this branch.** Real-valued contextual
  reads, fixed rank, causality and positional effects require their own analysis.
  In particular, [Wagstaff et al.](https://proceedings.mlr.press/v97/wagstaff19a.html)
  study representation of general continuous set functions under specific
  assumptions. Their width bound does not imply that a scalar counting channel
  needs width equal to the largest count. Some historical archive wording makes
  that unjustified inference; it must not be carried into the paper.

The strongest current reviewer objection is fair: this is grouped attention plus
conditional sum pooling and a residual adapter, supported by a small counting
pilot. The observed gain has not been isolated from normalization, extra compute,
or the way training checkpoints are selected.

## Revised research question and paper wording

Question: **Does processing query-conditioned local reads before their merger
improve native aggregation beyond hierarchical attention and simpler adaptation,
at controlled training and computation budgets?**

Suggested current description:

> We study how the order of local computation and global reduction affects
> aggregation in pretrained decoders. Our experimental adapter applies established
> conditional sum pooling to query-dependent chunk attention reads and writes the
> result into the native residual stream. We test whether processing each read
> before merging improves aggregation beyond changes in attention normalization,
> model capacity, and computation. The same intervention is available during
> subsequent decoding, allowing its reuse in reasoning to be tested separately.

This wording describes the investigation, not a proven mechanism or result. A
strong eventual paper needs a causal finding plus a practical consequence:
native multimodal extrapolation and/or improved aggregation during later
reasoning steps. Deployment in another modality alone is insufficient.

## Experiments that distinguish explanations

The following is a proposed sequence, not a preregistered experiment or an
already-run baseline. Register exact samples, predictions, selection rules
and resource limits before execution. All GPU and heavy CPU work remains on Slurm.

**First isolate computation order.** On the same local reads, define

    PRE_s  = W_up * s_R * mean_b SiLU(a_tb)
    POST_s = W_up * s_R * SiLU(mean_b a_tb)
    s_R in {1, R}

This gives a 2×2 comparison of nonlinear computation order and block-count
scaling, with identical learned parameters and access to the same information.
Our original sum is PRE_R. With an identity activation, PRE and POST are exactly
equal. This is a useful algebraic sanity check. POST is cheaper; report that
advantage rather than calling this a compute-matched comparison. If PRE wins,
compare against a useful wider/deeper POST or extra-head control at measured
compute. A deliberately repeated POST computation can serve only as a runtime
diagnostic, not as a competitive use of the budget.

**Then control the attention read and merge.** Use the same native layer,
positions, dense residual path and training allowance for an exactly reconstructed
global read plus adapter, and a learned normalized hierarchical mixture of local
reads plus adapter. Use per-head mixture weights before concatenating heads; a
single shared scalar block weight would weaken the hierarchical control. Include a cheap cardinality-scaled alternative. Mean pooling
alone is insufficient to represent strong hierarchical attention. A sigmoid
attention alternative is relevant if results specifically implicate normalization.

**Distinguish a mechanism control from a HiLS reproduction.** A single-layer,
answer-supervised adaptation of its fusion rule with native RoPE should be named
as such. A faithful reproduction requires the published architectural/training
choices and explicit accounting for differences; changing position encoding,
routing, nonlinearity and training together prevents attribution. Do not compare
our counting accuracy directly to a published retrieval score. If an established
method solves the task at lower cost, adopt it and revise the contribution.

**Vary what must be aggregated.** Independently vary total length N, relevant
evidence, its dispersion across blocks, and answer range. Current K<=8 longer-N
tests measure length/distractor generalization. They do not show generalization
to unseen counts. Compare concentration versus dispersion at fixed N and answer;
test entity/condition binding, block-boundary shifts and distinct tasks. A
selection baseline must receive enough retrieved blocks to access the evidence;
deliberately starving it does not isolate aggregation.

**Confirm useful transfer.** Start with the inexpensive symbolic setting to
identify the mechanism, then run native MMReD Vision against the strongest
surviving controls. Use development-only checkpoint selection, independent seeds
and held-out examples, paired uncertainty, and measured latency/FLOPs. For
reasoning, compare fixed-token and fixed-compute budgets; give controls the same
correct intermediate prefix before a second query-dependent aggregation. This
measures conditional later-step use; also evaluate end-to-end model-generated
reasoning to establish practical composition. Merely enabling the module at all
decoding steps is not evidence of composition.

## Current decision

Keep the implementation as an experimental baseline. Do not launch the previously
suggested broad vision comparison unchanged: sum/mean/LoRA would not settle the
literature objection. Add the missing mechanism and hierarchical controls first.
If only length scaling explains the gain, use the simpler operation and state
that result. If only task-specific segmentation works, narrow the scope. If a
replicated processing-order benefit survives and transfers, that can support a
more substantive contribution.

The [pilot report](NATIVE_AGGREGATION_PILOT_RESULTS.md) remains the empirical
record: fixed-epoch text exact improves over approximately matched LoRA, but the
development-selected gain is smaller, likelihood is worse, and compute is higher.
Vision extrapolation and reasoning composition remain untested. No code, data,
checkpoints, or experiment outcomes were changed by this literature audit.

## 2026-09-10 follow-up: closest antecedents for token-value repairs

The preceding decision records the initial literature audit. The subsequent
[V1 vision results](NATIVE_AGGREGATION_VISION_RESULTS.md) rejected the proposed
local-before-merge advantage; [V2](NATIVE_AGGREGATION_VISION_V2_DESIGN.md) now tests
ordinary adaptation and a separate channel-interchange diagnostic. The following
sources concern possible future token-level repairs, not completed experiments.

**Query-Value Interaction (QVI), Wu et al., 2020.** Section 3, equations 4–6
construct query-conditioned value features with a learned gate and sum them using
ordinary attention weights. This is a direct antecedent of applying a conditional
nonlinear map to individual values before pooling. Equations 7–10 propose a
different efficiency approximation; equation 7 mixes over query vectors. Naively
using all queries would leak future information in a causal decoder and make
stored values depend on later queries. The per-query form remains compatible with
causality when restricted to visible keys. [Primary paper](https://arxiv.org/html/2010.03766).

**Attention as a Hypernetwork, Schug et al., ICLR 2025.** Section 2.2, equations
6–7 place a nonlinear, key-query-specific value computation before token
summation. Its HYLA construction mixes head-specific projections using attention
scores and normalizes scores across heads, unlike our retained native softmax
residual branch. It tests compositional generalization on fuzzy logic and SRAVEN.
Appendix A.1/Table A1 separates normalization, nonlinear values and the full
construction, including a nonlinear-value-only control. These comparisons are
directly relevant; neither a generic nonlinear-value proposal nor the goal of
better compositional generalization is new. [Primary paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/abfa542f546df3c6c35695ec8d5bf4b9-Paper-Conference.pdf).

For a subsequent method study, use an established per-query QVI formulation or
a faithfully specified HYLA baseline when testing their corresponding hypothesis,
and distinguish any causal/native-decoder modification from a full reproduction.
A cacheable nonlinear-value control and a matched post-pooling control are
necessary before paying for query-specific nonlinear evaluation of every cached
value. Preserve the exact normalization in matched contrasts: changing it along
with the value network makes attribution ambiguous.

The strongest possible contribution is an empirical causal diagnosis in
pretrained decoders, a minimal repair that improves controlled aggregation
generalization, and demonstrated usefulness when later reasoning queries reuse
the same stored evidence. Native integration alone does not establish novelty.
An operator rename, a failed linear probe, or a different prompt layout does not
establish information loss. Existing question-first, short-context diagnostics
must not be treated as measurements of the V2 question-last setting.

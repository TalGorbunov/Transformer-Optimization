# Native aggregation: V2 attribution and the next mechanistic question

V2 tested whether the successful V1 global-read adapter adds useful access to
evidence beyond ordinary adaptation at the same decoder layer. Its primary
attribution screen failed. V3 now registers a matched nonlinear-value placement
comparison as exploratory follow-up. Neither study proposes a new attention
family. [`PREREG_AGG.md`](../../PREREG_AGG.md) remains authoritative for protocols,
budgets and decision thresholds; this note preserves their scientific rationale.

## Evidence motivating this decision

In [V1](NATIVE_AGGREGATION_VISION_RESULTS.md), global and POST-mean achieved
40.5% and 40.0% familiar-count accuracy at N32/N64, compared with 30.5% for
PRE-mean and 21.5% for upper-layer LoRA. PRE-sum and POST-sum both achieved 6.5%.
The registered case for local nonlinear processing before block reduction failed.
A four-example checkpoint inspection found a much larger branch relative to
ordinary attention for PRE-sum: mean norm ratio 47.7 at N64 versus 1.17 for global.
This exposes a scale problem, without establishing its cause or a population rate.
The large zero-read component in normalized adapters also motivates a hidden-only
control: their contextual hidden state already contains visual evidence.

V1 additionally changed a nuisance distribution between train and test. V2 uses
one generator law for every split. Its results therefore cannot be interpreted as
a controlled before/after comparison with V1.

## Completed V2 evidence and decision

The [V2 main report](../../outputs/native_aggregation_vlm/v2/REPORT.md) gives
familiar-count N32/N64 exact accuracy of 26.9% for global, 29.2% for hidden-only,
21.8% for middle-plus-upper LoRA and 16.2% for upper LoRA. Global-minus-hidden is
-2.3 percentage points, with paired-anchor 95% interval [-10.2, +5.1].
**V2.1 failed.** V2.2 passed its effect-size threshold against both LoRA controls,
but the middle-LoRA contrast interval crosses zero. This does not establish that
the additional read improves aggregation beyond same-site ordinary adaptation.

The [frozen prefix diagnostic](../../outputs/native_aggregation_vlm/v2/prefix/20260910_165238_440808_1379431/REPORT.md)
found identical baseline and correct-prefix exact accuracy: 4/18 at N16 and
1/18 at N64. Neutral-prefix accuracy was 3/18 and 1/18. The pooled
correct-minus-baseline difference is zero, and correct-minus-neutral is +2.78
points with a descriptive interval spanning zero. No positive prompt-level
repair was established; there is no further prefix search. This does not
disprove query-conditioned value computation.

The last-query-only channel-interchange diagnostic also gave no positive
donor-directed read effect. Its null result does not cover earlier query rows
or prove that the channels carry no usable information. The
[frozen recoverability study](../../outputs/native_aggregation_vlm/v2/recoverability/fit_440792/REPORT.md)
found weak linear count decoding/extrapolation. A failed ridge probe establishes
neither information loss nor intact local binding. Earlier question-first,
short-context relation-AUC claims are not measurements of this V2 setting.

Retire the V2 extra-read mechanism claim. V3 tests a narrower established
operation with a matched control; it does not assume these diagnostics already
identified a compression failure.

## What the four-arm V2 comparison identifies

All models retain frozen Qwen2.5-VL-7B-Instruct in NF4, ordinary image encoding,
native answer/EOS supervision and cached greedy decoding. All use nine training
epochs with development-only checkpoint selection. The middle site is layer 14.

| Arm | Middle adaptation | Upper four attention layers | Active parameters |
| --- | --- | --- | ---: |
| Global | `U SiLU(A h + B r + b)`, rank 64; `r` is the native dense read | LoRA rank 8 | 1,409,088 |
| Hidden | `U SiLU(A h + b)`, rank 96, same input and residual site | LoRA rank 8 | 1,409,120 |
| Middle LoRA | q/k/v/o LoRA rank 32 | LoRA rank 8 | 1,441,792 |
| Upper LoRA | None | LoRA rank 16 | 1,441,792 |

The rank-8 upper groups have matching initialization and training policy. Global
versus hidden compares two nearly parameter-matched architectures; it is not an
input-only intervention. Their nonlinear widths, parameterization and initial
activation distributions differ. Middle LoRA versus an adapter also changes the
operation and learning-rate policy: LoRA uses 1e-4 and the adapters use 1e-3.
These are useful adaptation controls, not proof of a uniquely necessary mechanism.

Training has 90 examples each at N8/N16, ten per K0..8; development has 36 per
length, four per K. Familiar-count tests comprise 108 N16 anchors, twelve per K,
each extended to N32/N64 with independently sampled negative frames. A separate
64 anchors, eight per K9..16, are tested at N32/N64. Negative types are equally
probable: same character/wrong room, other character/same room, or neither.

Paired extensions hold the question and semantic positive evidence fixed while
adding negatives. Positions and printed step numbers change. They identify
robustness to that controlled extension, not a context-length effect independent
of positions or distractor content. Across-K comparisons at fixed N are balanced
distributional comparisons, not paired interventions on the number of positives.
Analyses spanning N must group or bootstrap whole anchor families.

V2.1 screens for a global advantage of at least 5 percentage points over hidden
on pooled familiar-count N32/N64 exact accuracy, with no greater than 5-point loss
at N16. V2.2 requires the same OOD advantage over both LoRA controls. Passing
both justifies replication and testing causal use of the read. Failure leaves
ordinary adaptation sufficient to explain the improvement. Neither outcome
establishes a counting algorithm or composition during reasoning.

Unseen-count results remain separate: K9 versus K10..16 additionally changes
answer length/tokenization. Raw first-answer-token NLL is correctly measured by
V2, but does not score an entire multi-token answer. Whole-answer exact accuracy,
parse rate and tokenization must accompany any count-extrapolation statement.

## The registered causal-interchange diagnostic

Use the selected global checkpoint without updates on 16 new matched pairs at
N32. Each pair begins with K2; uniformly select four negative positions and
replace those sightings with the queried character/room conjunction, producing
K6. The question and all 28 other frames remain identical, including printed
step numbers. The separate causal-probe manifest records changed positions and
image/QA hashes and excludes all selected V1 and V2 main/profile content. These
32 examples are diagnostic, not an efficacy benchmark or model-selection data.

Capture the last prompt token's adapter inputs `(h, r)` in each clean forward.
In a recipient forward, interchange only the branch's `h`, only its fused `r`,
or both, using values from the matched donor. Keep the ordinary attention output,
residual stream, other branch rows and all other computations on the recipient
path. In particular, replacing `h` means changing the adapter input alone, not
the hidden state consumed by the rest of the model.

Measure the change in the raw logit margin `logit(K_donor) - logit(K_recipient)`
for both K2-to-K6 and K6-to-K2 donor directions. In addition to clean forwards,
conditions are identity-both (reinsert the recipient's own two channels), donor
`h` only, donor `r` only, donor both, and recipient-norm-matched versions of each
donor condition. Norm matching rescales each replaced donor channel to the
recipient channel's norm. Inspect channel-input and branch-output norms alongside
the logit shifts. The two directions share an image pair and are not independent
samples.

A donor-directed shift through `r` supports an effect of this branch input in
these paired scenes; a shift through `h` shows another usable contextual route.
It does not isolate count-specific information from other changes in the four
frames. No effect may indicate redundancy, an ineffective intervention or a
bottleneck elsewhere; it does not prove absence of information. Interchanges can
be off-distribution, and input norm matching does not match the nonlinear
branch-output norm or establish a general counting mechanism.

Stronger future controls would add same-count scene donors and norm-matched
directional perturbations, then vary N and the count difference. Those controls
are recommendations, not conditions included in this registered probe.

If hidden-only matches global and the read interchange has little specific
effect, first test where evidence remains recoverable: token representations,
the compressed read, or the later answer state. Small probes should use held-out
analysis labels only, matched capacity and data, and be interpreted as
recoverability rather than evidence that the model itself uses that information.

## Preserved B-versus-POST design and registered V3 interpretation

The minimal cacheable hypothesis learns a query-independent value map and moves
one nonlinearity across attention pooling. In a simplified single-head notation,
let `z_j = A v_j + b` and use an identical post-pooling query-conditioned head:

```text
B_t    = W psi(R sum_j alpha_tj phi(z_j) + B h_t + c)
POST_t = W psi(R phi(sum_j alpha_tj z_j) + B h_t + c)
```

All tensors are trainable in both arms and have identical shapes. The same
native attention weights, output site, training data and upper adaptation are
used. With identity `phi`, the arms coincide. Nonlinear lifting may preserve
useful statistics of the values even when a nonlinearity of their mean does
not. A gain would not alone identify those statistics as entity binding or
cardinality.

The actual registered V3 uses a native-width residual affine lift rather than
the simplified low-width projection: concatenate the four native 128-dimensional
KV heads, then set `z_j = v_j + C(A v_j) + b`, with inner rank 8 and zero-initialized
`C,b`. Unpack native heads and apply PRE
`sum_j alpha_tj^h SiLU(z_j^{g(h)})` or POST
`SiLU(sum_j alpha_tj^h z_j^{g(h)})`, where `g(h)` maps query heads to KV heads.
Both pass concatenated reads through the same rank-64 query-conditioned outer
adapter, whose output projection starts at zero. Both have 1,417,792 active
parameters including identical upper rank-8 LoRA.

The comparison preserves parameters and the common extra native-width SDPA
read; **it is not exactly compute-matched**. In an equal-width conceptual
implementation, one inner activation per source token versus per query token
has equal element counts during full prefill, and transformed-value caching
can make either incremental operation cheap. Actual Qwen GQA instead gives
PRE four KV heads and POST 28 query heads. The current implementation recomputes
the affine lift from all cached native V on every forward; PRE also reapplies
SiLU to all those values. It implements no transformed-value cache. Measure
actual prefill/decode latency and memory rather than claiming the hypothetical
cache advantage. PRE and POST also apply fp32 SiLU on opposite sides of the
bf16 SDPA rounding operation.

Both arms preserve native Q/K, positions, padding, the ordinary residual path,
and exactly one native KV update. The affine lift mixes heads within a token,
not different source positions. CPU tests cover the identity-activation equality,
initial native-output parity, trained-lift gradients, fully masked rows, suffix
perturbation and full/cached parity, including multi-token cache chunks. Equal
outer/lift zero initialization delays learning in the inner lift equally; it
does not establish identical later gradient scales.

V3 is PRE/POST crossed with seeds 0 and 1 under the separately registered
four-GPU-hour cap. Its primary screen requires at least a five-point familiar
OOD PRE advantage, with no greater than five-point N16 loss, in each seed.
Failure rejects that registered screening hypothesis at this setting and
budget; it does not prove that no nonlinear representation could help.

**The V2 test examples have already been inspected. V3 reuses them exploratorily.**
Preregistering the comparison does not restore untouched-test status, and two
seeds on the same examples do not repair adaptive test reuse. A positive result
requires confirmation on freshly generated held-out data and additional seeds.
No reasoning-composition claim follows from direct-answer accuracy.

A query-conditioned alternative `sum_j alpha_tj phi(A v_j + B h_t + b)` is a
prior-art baseline, not a newly invented operator. QVI §3/Eqs. 4–6 already
conditions individual values before pooling; HYLA §2.2/Eqs. 6–7 and Appendix A.1
study nonlinear key-query-specific value computations and a nonlinear-value-only
control. [QVI](https://arxiv.org/html/2010.03766),
[Attention as a Hypernetwork](https://proceedings.iclr.cc/paper_files/paper/2025/file/abfa542f546df3c6c35695ec8d5bf4b9-Paper-Conference.pdf).

B remains query-dependent through its attention weights and outer head, and
could already supply the relevant conjunction features. A must justify fresh
query/key-wise nonlinear work beyond that cacheable representation. Later-query
reuse is a required empirical test, not an automatic novelty claim. A meaningful
contribution would establish a causal failure mechanism, its minimal repair in
pretrained native decoders, and confirmed aggregation/generalization benefits.

## Other conditional hypotheses retained for reference

**1. Compute query-conditioned features before pooling.** The V1 PRE operator
already receives softmax-compressed 64-token reads. If those reads discard
binding information, later processing cannot recover it from the reads alone;
the contextual query may still provide another route. A hypothesis to test is

`m_t = sum_j alpha_tj * phi(U x_j + V h_t + b)`.

Keep normalized native attention weights and add a small residual projection.
This removes fixed block boundaries and makes the nonlinear query/value
interaction precede pooling. It can be evaluated at every causal token without
another backbone pass. However, query-dependent value features cannot generally
be cached once per source token, and prefill cost can be substantial: measured
compute and memory must enter the comparison.

The closest principles are question-conditioned
[Relation Networks](https://arxiv.org/html/1706.01427v1),
[FiLM](https://arxiv.org/html/1709.07871v2), and
[Deep Sets](https://arxiv.org/html/1703.06114v3). This is not a new map-and-pool
principle. The possible contribution is demonstrating and repairing a particular
compression bottleneck inside a pretrained decoder. Proceed only if token-level
evidence survives where current reads fail. A matched post-pooling map that
performs equally well on controlled conjunction-binding tests falsifies the
proposed advantage at the tested budget.

**2. Separate evidence mass from residual amplitude.** A normalized content read
could be accompanied by a calibrated statistic of attention mass, for example a
per-head log partition relative to a learned reference. The residual projection
would have a bounded gain independent of the number of visible blocks. This
tests whether multiplicity can be made accessible without the unnormalized
branch growth observed in V1. Native log partitions are not relevant-item counts;
they also respond to score shifts, irrelevant tokens and context length.

[Cardinality Preserved Attention](https://www.ijcai.org/proceedings/2020/194)
already studies multiplicity lost by normalized aggregation. The public
[Polar Attention implementation](https://github.com/kreasof-ai/atma/blob/main/docs/POLAR_ATTENTION.md)
also describes separating content from a bounded mass statistic; its efficacy
claims have not been independently verified here. Broad novelty claims about
mass/content separation would therefore be unsafe.

This hypothesis fails if the statistic adds no benefit over the same-site hidden
adapter and an explicit length-feature control, or predominantly tracks added
negatives rather than fixed-N changes in matching evidence. Bounded amplitude
alone offers no count-extrapolation guarantee.

Neither hypothesis should be presented as a rank increase, new sparse attention,
or the first hierarchical aggregation method.
[HiLS](https://arxiv.org/pdf/2607.02980) already performs normalized within- and
between-chunk attention and evaluates tasks requiring multiple pieces of evidence.
The next contribution should rest on a demonstrated failure mechanism and a
minimal repair that survives its strongest ordinary-adaptation control.

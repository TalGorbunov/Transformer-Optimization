# Native MMReD Vision: direct mechanism comparison

Started 2026-09-10. The current experiment tests whether processing local evidence
before combining it improves native answers on longer visual sequences. It follows
the [positioning audit](NATIVE_AGGREGATION_POSITIONING.md). The method is a
conditional Deep Sets-style adapter inside a pretrained decoder. A new attention
family, increased output rank, and benefits for reasoning models are not established.

## The comparison

A fixed layer reads every causal 64-token block using the model's existing attention
queries, keys, and values. Concatenating the head reads gives a vector r_b for block b;
h is the current token state, R the number of visible blocks, and phi is a shared
rank-64 SiLU adapter conditioned on h. The ordinary attention path remains active.

| Arm | Added residual, before output projection | What it tests |
|---|---|---|
| Before / sum | sum_b phi(r_b,h) | Local nonlinear processing with extensive scale |
| Before / mean | mean_b phi(r_b,h) | Local nonlinear processing with normalized scale |
| After / sum | R phi(mean_b r_b,h) | Same extensive scale after read compression |
| After / mean | phi(mean_b r_b,h) | Same normalized scale after read compression |
| Global read | phi(global-softmax read,h) | Nonlinear adapter on the ordinary read |
| Hierarchical | phi(learned normalized per-head fusion,h) | Learned block importance before the adapter |
| LoRA rank 16 | No aggregation adapter | Approximately matched trainable parameter budget |

Every added residual is projected back to the existing hidden size through the same
rank-64 output map. The hierarchical scorer starts from exact global fusion and
learns a normalized correction. It is a generic dense hierarchical control, not a
HiLS reproduction. The global control currently reconstructs an already computed
SDPA read, so its implementation is not a compute-optimal global baseline.

The four Before/After arms have exactly 1,409,088 trainable parameters. Their paired
contrasts separate processing order from length-dependent scale. The same learning
rate and clipping protocol do not guarantee equal optimization difficulty across
scales. A win demonstrates effectiveness of a configured adapter, not a theorem
about the capacity of complete pretrained transformers.

## Why processing order could matter

The narrow mathematical distinction is loss of information under averaging. With
scalar reads, q=0, and phi=SiLU, the sets {a,-a} and {0,0} have identical averages.
An after-merge map therefore gives the same result for both. Before-merge summation
gives a*tanh(a/2) for the first set and 0 for the second. It can retain information
about variation that their mean discards. With a linear map, the two orders agree
at either matched scale, which is also checked in the unit tests.

This is an elementary established set-function distinction, not a new expressivity
result for pretrained transformers. Whether that extra statistic helps visual
binding or counting must be demonstrated by the matched trained controls. The
block reads themselves are already compressed and are conditioned by the model's
contextual representations. Also,64-token blocks are not image boundaries: each
392 px image contributes 196 visual tokens, so the current blocks split frames.

## Protocol

- Qwen2.5-VL-7B-Instruct; frozen nf 4 backbone and standard image encoder; 392 px frames.
- Layer 14, block 64, rank 64; upper four attention layers have rank 8 LoRA (rank 16 for LoRA-only).
- Train 180 examples:90 each at 8/16 frames. Dev 72:36 each at 8/16 frames. Test 400:100 each at 8/16/32/64 frames.
- Final answer plus EOS supervision through the native vocabulary. Counts stay in 0..8.
- Nine epochs, seed 0. Select by pooled development exact accuracy, then lower policy NLL.
- Ordinary greedy decoding, no intermediate reasoning tokens. The count is normally one token;
  EOS is also generated. This does not yet test reuse across reasoning steps.
- Seven main arms, up to four concurrent B200 jobs, plus a frozen reference. All heavy work uses Slurm.

The [preregistration](../../PREREG_AGG.md) defines the decision rules. The primary
contrast requires Before/sum to beat After/sum by at least 5 percentage points at
N32/64 with at most 5 points loss at N8/16. Before/mean has the analogous secondary
contrast. The practical screen requires one Before arm to beat every trained control
by those margins. The seed-0 paired bootstrap is exploratory and conditional on the
selected examples; it is not independent-seed confirmation.

## Audit limitations

The exact shared manifest excludes duplicate sequence/question content and prior
smoke examples. Missing N16 training/dev images were rendered with verified pixel
parity to existing samples. All 664 main/profile gold labels were independently
recounted from canonical states; no states or per-frame labels enter the model.

However, N16 training/dev contain more same-character distractors than the existing
test source: target-character occurrence means 10.31/10.28 versus 6.12 in test.
Rendering parity does not imply a common sampling law. N8/16 are in the training
length range, but N16 test is not IID. N32/64 test combined length and nuisance-law
generalization. Clean length interventions require matched distractor extensions.
See the [data census](../../outputs/native_aggregation_vlm/v1/data_audit/INDEX.md).

The cached generation configuration supplies repetition_penalty=1.05. Therefore the
saved first-token NLL is full-vocabulary generation-policy NLL, not raw LM NLL. This
same value breaks dev-accuracy ties for all arms. The inherited setting is preserved
throughout V1 and recorded in generation_settings.json.

## Artifacts

[Execution and outputs](../../outputs/native_aggregation_vlm/v1/INDEX.md).
Main array 440504; frozen reference 440519; CPU reports 440509/440653/440691: completed.
[Results and interpretation](NATIVE_AGGREGATION_VISION_RESULTS.md).
The preregistered checkpoint-scale inspection completed on disjoint profile samples.
It decomposes the branch into zero-read and read-dependent components without
claiming that the contextual hidden state is uninformed or that magnitude proves causality.
Data and checkpoints use the user-designated /mnt/data and /mnt/ckpts project roots.

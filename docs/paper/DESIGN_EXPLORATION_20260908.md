# Design exploration: preserving evidence quantity

Status update: the user clarified that native aggregation reusable at every
reasoning step is the objective. The separate numerical interface proposed here
is outside that objective. See [NATIVE_AGGREGATION_PROPOSAL.md](NATIVE_AGGREGATION_PROPOSAL.md)
for the current direction. This earlier exploration is retained for context.

2026-09-08. Proposal, not a result or a pre-registration. No experiments were run
for this note. Read HANDOFF.md, the paper outline, the relevant PREREG_AGG outcomes,
and the vision implementations before developing this proposal.

The objective is a simple, single-forward method that learns from final answers
and extrapolates on MMReD Vision. The proposed improvement is reduced task-specific
machinery and stronger generalization, not another percentage point on a nearly
saturated restricted test.

## What the existing evidence supports

- The training-free vision results are real in the saved summary: .98/.97/.99/.98
  exact match at N=8/16/32/64, 100 examples each. Source:
  `outputs/judge_fenced_vlm/20260907_111236_4004643/results.json`.
- The current task description explicitly specifies one character per frame and
  gold count K<=8 at every N (`scripts/condmask/vlm_lz.py:13-14`). This is evidence
  of distractor-length robustness, not yet numerical extrapolation. Audit the
  actual data histograms before preparing the new evaluation manifest.
- The training-free method rewrites a fixed question grammar into a local Yes/No
  question, compares selected vocabulary logits, and thresholds and sums the
  verdicts (`outputs/_scratch/dbg/judge_fenced_vlm.py`). It does not have the
  backbone generate the count. Saved `yesno_top1` is zero: these are contrastive
  vocabulary scores, not naturally generated Yes/No responses.
- The trained vision method uses oracle frame labels with BCE, not just final
  count labels (`outputs/_scratch/dbg/train_bindcount_vlm.py:101-106`).
- The existing no-fence vision arms also disable position reset. Their difference
  estimates the joint intervention, not the fence alone.
- Trained carriers receive index-dependent positions after frame position reset
  (`gnnformer/engine.py:121-126`). Exact invariance to frame index therefore does
  not follow from this implementation. The training-free judge reads the token
  before that carrier and avoids this particular issue.

The present approach is a strong control. A matched ordinary batch of independent
frame judges should be tested: under identical visible prefix, local tokens, and
positions, its local computations should reproduce the isolated blocks up to
numerical differences. Packing those computations into a mask is not by itself a
new reasoning algorithm. Dense masks also do not establish sparse compute savings.

## The proposed insight

Semantic processing needs to determine what an observation means. Aggregation
also needs to preserve how many observations support it. A normalized average can
retain the former while losing the latter.

For the restricted case of independently encoded, identically repeated evidence,
normalized pooling returns the same vector after duplication. An additive summary
doubles. This is a revealing example, not an impossibility theorem about all
transformers: positions, contextual values, and fixed reference tokens can encode
cardinality in an otherwise normalized network.

**Hypothesis:** a frozen VLM can learn question-conditioned local evidence writes
from final answers, and a separate additive channel can preserve their quantity
well enough to generalize independently in distractor count and evidence count.

The scientific test is whether this works without hand-written local predicates
or intermediate labels. Merely moving the current Python sum into `forward` is
not the proposed contribution.

## Minimal architecture

1. Encode the original question as a shared prefix. Give each frame one shared
   learned read token; use the same local position for that token in every block.
   Initially retain isolated blocks and local image positions to keep the local
   computation independent of the number of other frames. No regex rewriting and
   no repeated hand-written Yes/No prompt.
2. Read each final local state h_i. Apply a small shared learned write function
   phi to obtain a vector v_i. The local state already has access to the question.
3. Form an additive state m = sum_i v_i. Feed m directly to a small answer head,
   alongside a question representation. Preserve raw m on this path: do not pass
   its only copy through softmax pooling, RMSNorm, LayerNorm, or division by N.
4. Train the read token, write function, and output head with final-answer loss.
   Start with frozen backbone weights. Use a numerical regression output for
   counts, with fixed rounding only at evaluation. No per-frame thresholds.

Equations:

    h_i = F_frozen(frame_i, question; learned_read_token)
    v_i = phi_theta(h_i)
    m   = sum_i v_i
    y   = D_theta(question_state, m)

All local encoding runs in parallel, followed by the small reduction and decoder
within the same forward graph. Each backbone layer is traversed once. The encoder
may be implemented with packed blocks or an ordinary batch. Report vision encoder
work, language model work, latency, peak memory, and total tokens separately.

Use a linear scalar decoder as the simplest count baseline and a small shared
nonlinear decoder only when testing richer tasks. A linear decoder makes the
inductive bias explicit; a nonlinear decoder offers more flexibility but can
saturate on unseen quantities. Neither is evidence of native language generation.

There is no mathematical objection to normalizing local features before writing.
The concern is losing the scale of the accumulated state. Normalizing m alone
approximately erases a positive rescaling; normalizing m plus a fixed residual
can preserve some scale information through its direction. Measure this rather
than claiming all normalization destroys all counts.

## Why this is still a risk

The sum guarantees additivity of m only when local writes do not change with the
rest of the input. It guarantees neither correct writes nor extrapolation of D.
Final-answer supervision may permit incorrect local contributions to cancel.
The current answer range may allow shortcuts unless N and K vary independently.

A scalar additive count head is a modest simplification of bind-then-count. The
more ambitious claim requires the learned vector representation to support at
least one additional aggregation task with the same computational structure and
final-answer supervision. Compare two queried counts, or predict a histogram;
do not implement a separate Python reducer for each task and call that learned
generality. Keep distinct counting and temporal composition outside the first
claim: they require information that a small additive summary may not preserve.

For binary local predictions, expected count bias is

    E[predicted_K - K] = (N-K) * FPR - K * FNR.

Small false-positive rates can therefore dominate at long N. Exact count can
also be correct when false positives and false negatives cancel. The expression
(1-epsilon)^N describes all local predictions being correct only under an
independent equal-error model; it is not generally exact-count accuracy. Report
FPR/FNR, count bias, MAE, and exact match separately.

If the existing language head must emit the answer, treat that as a separate
experimental requirement. The failed tally/helix interventions already show that
recoverable numeric state does not ensure correct digit generation. Multi-token
numbers also require defining whether ordinary answer decoding is allowed under
the phrase single forward. Do not conceal this by restricting the output range.

## Smallest decisive experimental sequence

First do inexpensive checks, then pre-register and budget any GPU work.

1. Audit N/K support and split overlap. Check frame-level duplication between
   splits, not only sequence IDs. Construct a held-out manifest without modifying
   existing datasets. Duplication of one frame is useful as a mechanistic test,
   but separately evaluate distinct rendered matching frames.
2. With oracle local features, train the proposed small decoder only within the
   intended K range and test beyond it. Include repeated-evidence scaling and
   appended-zero tests. Reject a decoder that saturates even with perfect inputs
   before paying for VLM training. This validates a component, not perception.
3. Establish the matched independent-frame judge baseline. Separate mask and
   position reset in a 2x2 control, holding wording and readout fixed. Check local
   states under permutation and appended distractors.
4. Train answer-only variants with identical local encoders, examples, budget,
   and decoder capacity: normalized pooling, raw sum, and normalized raw sum.
   Include mean plus explicit N as a strong simple control: it can reconstruct
   a sum in principle. Include the answer-only scalar sum baseline. Retain the
   current oracle-frame-supervised method as a separately labelled reference.
5. Evaluate N and K independently. For example, train N in {8,16,32}, K<=8;
   test (a) N=64/128 with K<=8; (b) N=32 with K=12/16/24; (c) N=64/128 with
   K=16/32/64 where valid. Longer distinct-frame data need a new artifact and
   rendering/compute budget. Include K=0 and dense-positive cases. Select models
   only on in-range development data; fix extrapolation cells before running.
6. Test unseen question phrasing and held-out entity/attribute combinations.
   If count learning succeeds, add count comparison using the same vector write
   and decoder design, trained with final comparison answers. A second task is
   an independent generality test, not a zero-shot claim unless withheld in training.

Provisional success criteria to settle before the run: retain approximately the
current in-range performance (within 3 percentage points on paired evaluation),
achieve at least 90% exact match in designated unseen-K and long-N cells, and
beat answer-only normalized baselines with paired uncertainty estimates. These
are targets, not predictions supported by existing results. Replicate the winning
configuration with multiple training seeds before claiming an improvement.

Stop or narrow the story if answer-only learning fails, if raw quantity does not
help relative to mean plus N, or if only distractor extrapolation improves. If the
scalar baseline matches the vector design, keep the simpler model. If a second
task needs a hand-coded reducer, do not claim a general learned aggregator.

## Prior art and paper discipline

This architecture belongs to the Deep Sets family. Sum pooling is not novel.
Soft-attention counting limitations were discussed in VQA well before the present
VLM setting. Sigmoid attention and arithmetic modules are also established.

- Deep Sets: https://papers.neurips.cc/paper_files/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html
- Learning to Count Objects in Natural Images for Visual Question Answering:
  https://arxiv.org/abs/1802.05766
- Theory, Analysis, and Best Practices for Sigmoid Self-Attention:
  https://arxiv.org/abs/2409.04431
- Neural Arithmetic Logic Units: https://arxiv.org/abs/1808.00508
- Attention Normalization Impacts Cardinality Generalization in Slot Attention:
  https://arxiv.org/abs/2407.04170

Possible contribution: a causal account of where pretrained VLM aggregation
fails, followed by a minimal answer-supervised repair that survives genuine
cardinality extrapolation. This remains a hypothesis, not an established novelty
claim; a full related-work review is still necessary.

Apply the supplied scientific-writing principles while choosing the research:
one testable hypothesis, one revealing example, the simplest competitive baseline,
equations that match implementation, and experiments that can reject the story.
Source: https://perceiving-systems.blog/en/post/writing-a-good-scientific-paper

Suggested teaser: hold frame appearance fixed and duplicate evidence. Plot the
local representation, aggregated state, and predicted count against multiplicity.
Show exactly where invariance is desirable and where it loses the answer. Then
repeat with new, distinct matching frames to test actual generalization.

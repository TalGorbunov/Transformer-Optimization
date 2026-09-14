# Training-only local teachers for native aggregation

Status updated2026-09-10: V6 is registered and running under this direction; no V6 efficacy result yet. The exact executed protocol is in the append-only PREREG_AGG.md. It fixes the V5 SUM operator, retains the existing V4 training/dev schedule, and evaluates452 fresh complete contexts with seeds6/7. The optional permutation and fresh binding-pair experiments below are proposals, not part of V6. Teacher feasibility passed on all18800 original training-frame occurrences; both computational profiles and the independent release audit passed. Main array441650 is running. See [execution and cost](../../outputs/native_aggregation_vlm/v6/execution.json).

The remaining text preserves the original proposal and its broader alternatives. Completed V5 results are in [the V5 report](NATIVE_AGGREGATION_VISION_V5_RESULTS.md).

## Question and motivation

Can the model's competence on an isolated image teach a small per-image reader to support more accurate native counting over many images, without deploying a local-answer decoder, external tally, or additional reasoning tokens?

The existing independent-image branch asks a randomly initialized rank-96 reader to learn local visual evidence, aggregation, and compatibility with the native answer decoder from count-only supervision. A count error does not identify which individual image was misunderstood. In addition, the zero-initialized outer projection initially prevents count-loss gradients from reaching the reader through that branch. Local supervision could improve this credit assignment while the original answer loss continues to teach the native readout.

The completed indexed-frame diagnostic motivates the teacher but does not establish this proposal: isolated frozen-Qwen Yes/No judgments were strong, whereas full-context indexed judgments were weaker. Those questions differ from ordinary counting, and competence of the whole isolated VLM does not imply competence of one reused FFN or a small random reader. Isolated 0/1 counting must be measured independently on training images before assuming it is a useful teacher.

## Teacher and unchanged inference

For each training scene/image pair, give frozen Qwen the exact original count-question string and that one original image. Use the existing `build_count_prompt(question, 1)` and ordinary processor/chat template, with the original visible Step label preserved. The canonical wrapper changes its frame count and allowed range to 1 and 0–1; the question itself is not rewritten into a predicate or Yes/No task.

Verify that `0` and `1` are single-token continuations at the actual prompt boundary. From one ordinary forward pass, save their raw logits, the full-vocabulary log normalizer, top-one ID/text, and the target

\[
t_i=(p_T(0),\;p_T(1),\;1-p_T(0)-p_T(1)).
\]

The third category represents every other first token. It prevents a teacher with negligible numeric probability mass from appearing certain after conditional 0/1 normalization. These are first-token distributions, not claims about whole-answer generation. Use every target without confidence filtering, hard-label replacement, or per-example reweighting based on correctness. The teacher is fixed; no new generation or teacher inference occurs during deployment.

Audit teacher accuracy, class-conditional errors, calibration, and full-vocabulary 0+1 mass using training-frame semantics only. Report both unique-pair and original-presentation weighting. Frame-level gold labels are audit-only and never enter the auxiliary targets, loss, or sampling. Existing scene-level gold counts continue to train the native answer loss. Freeze any teacher-quality go/no-go rule before inspecting this calibration; an unreliable teacher should stop this proposed experiment, not trigger prompt search or sample selection.

Root's read-only sizing found 1,540 V4 training scenes containing 18,800 frame presentations, 864 distinct rendered image hashes, 52 distinct question strings, and 9,980 distinct image-hash/question pairs. Deduplicate teacher computation by that pair plus the complete prompt, processor, resize, tokenizer, and model identity. Reuse the cached distribution at every original scheduled presentation, preserving the V4 refresh weighting. Its nine blocks contain 1,620 scene presentations and 19,440 frame presentations after saturated-example reuse. This is a finite compositional benchmark whose local rendered atoms recur; fresh sequences do not imply unseen local visual content or real-world generalization.

## Auxiliary objective and leakage prevention

Keep the existing per-image branch, native count head, upper LoRA, and answer/EOS cross-entropy. Attach one shared linear three-way auxiliary decoder to each pre-merge rank-96 message:

\[
L = L_{\mathrm{count}} + \lambda\frac{1}{N}
\sum_{i=1}^{N}\mathrm{KL}\left(\operatorname{stopgrad}(t_i)
\;\|\;\operatorname{softmax}(A m_i+b)\right).
\]

Use a fixed, prospectively registered coefficient and temperature; an initial choice of both equal to 1 avoids a search. Averaging over images prevents a larger scene from automatically receiving a larger auxiliary-loss weight. The decoder has 291 training-only parameters at rank 96. Remove it at inference, leaving the same native count prompt and computation as the matched count-only branch.

Crucially, compute differentiable messages at the **last original prompt token**, before any teacher-forced answer token. The existing exported diagnostic messages are detached and cannot train this objective. Taking the final token of an SFT sequence would instead expose the gold answer/EOS and would mismatch inference. Record and check the original prompt boundary explicitly. The teacher sees only the image and question, while the student's reader sees its normal full-scene count prompt; neither receives local labels or auxiliary questions in its input.

This loss trains local readability, not an explicit count coordinate. It does not enforce zero contribution for negative images, prevent contextual query drift, guarantee survival through normalization, or force the native decoder to use what the auxiliary head can read. Successful transfer to ordinary native counting is therefore essential.

## Matched controls and scope

Select and register **one** existing pooling operator before any new teacher-training outcomes. Use that operator in every arm and seed; do not compare a distilled SUM arm against a count-only MEAN arm. The completed V5 audit may inform this engineering choice, but a choice informed by prior test results must be disclosed as adaptive. Do not rerun a SUM/MEAN search inside the objective experiment or claim that operator choice is independently confirmed by reused V5 tests.

| Arm | Auxiliary target and gradient | What it controls |
|---|---|---|
| Detached control | Correct teacher target; auxiliary decoder receives `stopgrad(m_i)` | Same targets, decoder training, and instrumentation, with count-only gradients to the deployed model |
| Aligned teacher | Correct target for the corresponding image; auxiliary gradient reaches the reader | Effect of the proposed objective |
| Optional permutation control | Fixed within-scene permutation of teacher targets; auxiliary gradient reaches the reader | Whether exact image correspondence matters beyond scene-level target counts and confidence |

The bounded first block is two arms, detached versus aligned, across two fixed seeds: four contemporaneous runs. It can identify the effect of adding this objective under matched training. Adding the permutation arm for both seeds makes six runs and supports the narrower correspondence claim. Aligned versus permuted alone is insufficient because wrong correspondence can actively damage training.

Use identical deployed-model initialization, scene schedule, optimizer settings, count loss, model precision, source files, and evaluation policy. Initialize the auxiliary decoder in an isolated RNG context, so its creation does not perturb model initialization or training order. Train its private parameters with a separate optimizer and private clipping. Exclude auxiliary-head gradients from the branch/LoRA clipping norm in every arm; otherwise the detached control could change deployed-model updates through clipping despite its stop-gradient. The deployed parameters' clipping includes whatever count and auxiliary gradients that arm legitimately receives. Log both groups separately.

Reusing historical V5 baselines would reduce cost, but is weaker than this contemporaneous control given the observed numerical sensitivity and new training instrumentation. Historical results should remain context, not replace the primary matched comparison.

## Evaluation and decisive failures

Keep the V4 refresh training schedule fixed to avoid another data-diversity change. Stage new dev/test sequences, disjoint in full state/question content from all prior selected and software datasets, under the same audited V2 law. Preserve paired N16→N32/N64 negative extensions, K0–8 familiar counts, a separate K9–16 count axis, and fresh fixed-marginal binding pairs. All images, questions, pairing, and exclusions require the existing CPU data audit. The held-out target mixtures and metrics must be frozen before model outcomes.

A reasonable screening criterion is at least +5 percentage points familiar-OOD native exact accuracy over the contemporaneous detached control in **each seed**, with no more than 5 points of N16 loss. Report all-example parsing/exact denominators, MAE, bias, per-K results, nonzero-count accuracy, unseen counts, and binding-pair precision separately. Use whole-anchor bootstrap intervals and do not let a pooled result rescue a failed seed. Register any absolute usefulness threshold before evaluation; a small relative gain from a poor baseline is not itself a working solution.

The outcomes have distinct meanings:

- Unreliable isolated 0/1 predictions invalidate this specific teacher choice; earlier Yes/No success cannot substitute for calibration.
- Better auxiliary prediction without native-count improvement rejects local extraction alone as an adequate repair under this architecture, loss, and budget. It does not prove information is absent.
- Gains confined to K0, worsening nonzero counts, or failure of the registered two-seed criterion do not establish improved aggregation.
- If the permutation arm is included and performs similarly to aligned supervision, a generic regularization or scene-level supervision explanation remains; do not claim image correspondence caused the gain.
- Improvement on fresh native counts and fixed-marginal binding pairs would justify a subsequent matched method/control × direct/reasoning experiment. It would not establish reasoning composition by itself.

## Cost and literature position

Teacher caching needs 9,980 single-image forward passes, not 18,800 independent generations. Their actual cost still requires a bounded Slurm profile with the intended frozen model and processor. A provisional envelope is roughly four GPU-hours for four matched training runs plus up to one GPU-hour for the teacher cache, subject to profile-based revision and an explicit root-approved cap. This document launches nothing. All image work, teacher/model work, and heavy CPU work must run through Slurm; caches belong under `/mnt/data/gabriele/gnn_transformer`, model/auxiliary checkpoints under `/mnt/ckpts/gabriele/gnn_transformer`.

The ingredients are established. [OPSDL, sections 3.1–3.4](https://arxiv.org/html/2604.17535v1) uses short-context teacher behavior to supervise long-context behavior. [CLIPSelf, section 3.2](https://arxiv.org/html/2310.01403v2) transfers isolated-region teacher representations into corresponding contextual features. [PVM](https://arxiv.org/html/2605.00814v1) already provides a close precedent for a native query reading persistent raw visual memory. These comparisons must accompany the existing DeepSets/HiLS positioning; the new loss does not make the underlying structured reader a new attention family.

Nor should this be renamed generic consistency or a sufficient-statistic theorem. Fixed-query SUM partition additivity already holds by construction and provides no training signal. Recomputing the query changes such a loss into context/view invariance. Explicit associativity and commutativity regularization also have precedent in [Learnable Commutative Monoids, section 2.5](https://arxiv.org/html/2212.08541). A count-trained latent representation is not identifiable as a sufficient statistic merely because an auxiliary head decodes local labels.

The possible contribution is the measured transfer: **training-only local competence improves native aggregate precision and length/count extrapolation without local outputs or additional reasoning tokens at deployment**. Exact frame correspondence is privileged training structure, and 0/1 supervision remains a counting instantiation. Reusability across tasks and composition with autonomous reasoning would each need separate evidence.

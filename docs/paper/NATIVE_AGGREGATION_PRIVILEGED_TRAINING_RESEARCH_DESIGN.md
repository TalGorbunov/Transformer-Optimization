# Privileged training as a scientific instrument

Date: 2026-09-14. Status: research design, not an execution release. No finetune or new GPU job has run. This develops the user's proposal to teach the model with privileged information, then study what it learned. The raw-feature summary-residual profile remains held. The same-world behavioral diagnostic remains useful and can share evaluation worlds with this study.

## The question

Can the ordinary MMReD Vision model learn a useful parallel aggregation computation when its intermediate state receives exact supervision, while its deployed architecture, visual input, prompt and decoding procedure remain unchanged?

The hoped-for discovery is the smallest causally useful intermediate computation that ordinary answer training fails to establish. Privileged supervision is an experimental instrument, with substantial prior art. Neither a positive result nor a failed fit automatically identifies a unique algorithm or a fundamental structural limit.

This directly tests the distinction between what the existing model can learn under an informative objective and what a different architecture might make easier to learn. A successful supervised construction would demonstrate capacity for the tested task and lengths. An unsuccessful rank-16/NF4 experiment cannot establish that transformers lack that capacity.

## Exact teacher information

Use original five-person, six-room MMReD worlds. For frame t, define the same 40 query-independent channels in both supervised conditions:

- 30 person-room membership indicators.
- 10 unordered person-pair co-location indicators.

Call this vector F_t. The local teacher supplies F_t. The aggregate teacher supplies S_t = sum_{u <= t} F_u. The teacher is a deterministic calculation from the training world's latent scene, checked against the original task oracles. It has no access to evaluation worlds during fitting.

For this task subset, the final accumulated vector supports room occupancy, partner co-location, and room-step answers. Atomic frame retrieval additionally requires local facts. MOST/LEAST answers depend on relative frequencies; absolute-count loss is not the only possible explanation of their failure. Once accurate local relations are represented, averaging these vectors already preserves the rankings required by the primary questions. This is a task-specific observation, not an expressivity theorem about a full VLM.

The teacher is intentionally richer than one final answer per world. It reveals multiple intermediate facts and implicitly supports additional questions. This extra supervision must be disclosed. It is not a scene-graph input or a hard-coded counting module at inference.

## Three matched finetunes

| Arm | Training supervision | Deployment |
|---|---|---|
| Answer-only continuation | Original complete answer loss | Ordinary native model |
| Local-fact teacher | Answer loss plus F_t targets | Ordinary native model |
| Aggregate-state teacher | Answer loss plus S_t targets | Ordinary native model |

Start all arms from the competent ordinary checkpoint. Match initial trainable state, training worlds and questions, order, optimizer updates, adaptation scope and endpoint. Retain the original checkpoint as a reference, but compare the supervised arms primarily against the equally continued answer-only arm. Replicate the comparison across fit seeds; do not choose a checkpoint on exposed extrapolation accuracy.

Apply the auxiliary loss at existing image-end token states after one prespecified decoder layer. The preferred site is the inspected input to final decoder block 27, where the image states supply keys and values for the later question. Final-block output rows at earlier image positions would not themselves be reread in that same block. Recompute the lower states during fitting so all existing language LoRA receives gradients; the frozen-prefix replay used by the held residual profile is inappropriate for this study. Each target uses only frames visible to that position. In the existing images-before-question prompt, visual states cannot depend on a later question; all 40 channels are therefore query-independent. The later question still has query-dependent attention across layers: this causal layout does not prohibit reasoning. Later frames can depend on preceding images, so these are prefix-contextual states rather than independent frame encodings.

A fixed, shared orthonormal projection P from the native hidden width to 40 coordinates gives a concrete intervention target. The illustrative objective is

    L = L_answer + lambda * mean_world mean_prefix ||P h_t - a(target_t)||^2.

Here a is a per-channel affine normalization derived solely from training worlds. Constants stay fixed at longer evaluation lengths; do not clip to the training count range or normalize with evaluation labels. Equal-world/equal-prefix averaging prevents long scenes automatically receiving greater weight. Degenerate target channels require an explicit fixed treatment before release. Equal dimensions and variance do not equalize semantic difficulty or gradient geometry; report auxiliary and answer losses separately.

The projection is fixed and evaluated only for the training loss and diagnostics. It is absent from ordinary inference. No oracle values are injected into student inputs, no helper tokens are appended, no latent recurrent steps are introduced, and there is no extra forward pass at deployment. Native JSON answer tokens still incur their usual decoding work. Training overhead is measured separately.

The layer, projection seed, normalization specification, auxiliary weight and numerical recipe must be fixed using training-only software checks before any fresh predictions. This document does not choose them through an architecture or layer search and is not a runnable protocol.

## What we would examine

First report unassisted complete-answer accuracy on a fixed fresh N8/N16/N32 cohort, preserving factual retrieval and the original tasks. N32 is beyond the original maximum training length N16. N64 and additional domains follow a convincing first result. Keep the ordinary-distribution cohort distinct from deliberately conditioned counterfactual panels.

Measure intermediate prediction errors across unseen prefix lengths, including whether decoded S_{t+1} minus decoded S_t matches the next frame's F_{t+1}. Use legal extensions and relabelings fixed before outcomes. This is a test of the learned state update, beyond reading a final answer code.

Then test whether the supervised state actually contributes to the answer. A low auxiliary loss is not evidence that the decoder uses that state. Fixed coordinates deliberately engineer a candidate solution; they do not discover the baseline's spontaneous representation.

Use a small prespecified intervention panel drawn by oracle criteria before model outcomes:

1. Choose legal paired scenes with different totals but the same answer to one question, and a changed answer to another predetermined question. Transplant the relevant projected channels from actual donor activations. Test the specific counterfactual answer while retaining the answer that should stay unchanged. This is stronger than swapping a representation between two differently labeled examples.
2. Include equal-total controls. Reversing a legal sequence preserves all 40 final totals and can change atomic frame answers; validate the original movement law and labels explicitly. This tests tolerance to a change in temporal order for the aggregate tasks, not invariance of all hidden states.
3. Include identity swaps, unchanged-answer swaps and equal-rank complementary-subspace perturbations. Preserve ordinary unmodified generations for every scene.

The exact aligned sites and counterfactual program must be specified before interventions. Prefix statistics are redundantly represented at multiple sites; a single-site negative does not establish non-use. Mixed activations can also be off distribution. Coherent donor states and specificity controls strengthen evidence without eliminating this limitation. Arbitrary exact-oracle clamping is a separate privileged readout diagnostic, never a method score.

Only after behavioral gains and causal use are established should we analyze a small number of parameter groups to distinguish learned changes to relational features, their transport, and the final readout. Attention maps or weight-difference norms alone do not establish mechanism.

## Decisions the experiment can support

| Outcome | Interpretation and next step |
|---|---|
| Local-fact supervision closes the gap | Grounding or local relational representation deserves priority; an additional accumulator is not yet motivated. |
| Aggregate-state supervision gives a distinct fresh gain and causal state use | The existing architecture can learn a useful aggregation organization under stronger supervision. Study how to teach that organization more generally. |
| Totals are accurately represented but answers fail | Investigate access to and use of the state, with the privileged readout diagnostic as a control. |
| Both supervised arms fit short examples but fail longer ones | Examine contribution error, positional dependence and readout extrapolation; no structural impossibility conclusion. |
| Neither auxiliary target can be fitted | The alignment, optimization, representation or restricted adaptation setup is unresolved. Do not treat this as evidence that an extra inference module is required. |

The final contribution must go beyond MMReD-specific teacher labels. A promising follow-up would replace exact symbolic state labels with a transferable training signal or introduce the smallest generic mechanism supported by the causal result. That choice is contingent on findings. Earlier local-distillation and summed-message failures must inform it.

## Literature positioning

Primary-source review through 2026-09-14. This is a targeted audit, not a guarantee of exhaustive novelty.

| Prior work | Already established; implication here |
|---|---|
| [LUPI](https://www.jmlr.org/papers/v16/vapnik15b.html); [generalized distillation](https://arxiv.org/abs/1511.03643) | Training-only privileged information and teachers using richer representations. Our oracle-to-visual training belongs to this family. |
| [CLRS](https://arxiv.org/abs/2205.15659) | Intermediate algorithm-state supervision. Its original recurrent hint processing is not evidence of unchanged ordinary-forward work. |
| [ALTA](https://arxiv.org/abs/2410.18077) | Execution-trace supervision studies the gap between representable and learned algorithms. Particularly close to our scientific question. Universal Transformer loops must be charged as computation. |
| [RASP generalization](https://arxiv.org/abs/2310.16028) | Connects length generalization to algorithms suited to transformer computation; scratchpads can hurt when they make the task harder to express. No blanket need for additional serial reasoning. |
| [Distilling Step-by-Step](https://aclanthology.org/2023.findings-acl.507/) | Rationales as auxiliary supervision alongside answer training. Better answers from extra supervision are already established. |
| [iCoT-KD](https://arxiv.org/abs/2311.01460); [iCoT-SI](https://arxiv.org/abs/2405.14838) | Distillation into implicit computation and progressive removal of reasoning traces. iCoT-KD retains a thought emulator/student system; iCoT-SI is closer to an ordinary direct-answer deployment. |
| [IIT](https://proceedings.mlr.press/v162/geiger22a.html); [causal distillation](https://arxiv.org/abs/2112.02505) | Training and testing counterfactual behavior of aligned internal variables. Our first study uses intermediate supervision followed by causal tests; it is not a new intervention framework. |
| [Coconut](https://arxiv.org/abs/2412.06769) | Continuous thoughts consume sequential model evaluations and already have no-thought/pause controls. Fewer visible tokens is not automatically less work. |
| [Mirage](https://arxiv.org/abs/2506.17218); [Mull-Tokens](https://arxiv.org/abs/2512.10941) | Privileged visual/intermediate supervision followed by relaxed training, with latent inference computation. These are close multimodal comparators. |
| [GAS](https://arxiv.org/abs/2608.12209) | Auxiliary visual-generation supervision with a training branch discarded at deployment. Zero inference overhead by itself is not novel even for VLMs. |
| [System-2 counting analysis](https://arxiv.org/abs/2601.02989) | Studies local counts, attention-mediated transfer, and final combination under a test-time decomposition. Close mechanistic precedent; its architectural-limit interpretation is not adopted as a universal theorem here. |
| [When Internalization Fails](https://aclanthology.org/2026.findings-acl.734/) | Reports limitations of implicit-CoT curricula on long mathematical traces. Privileged training does not guarantee successful internalization. |

Sparse retrieval/HiLS and learned visual resampling remain relevant if an inference module is later introduced. The current study changes supervision with the ordinary attention path intact; that difference is scope, not a novelty proof. Summation, cardinality features, auxiliary heads and causal patching are established ingredients.

The candidate contribution is a demonstrated causal explanation of an aggregation failure, followed by a transferable way to repair it at measured inference cost. A successful MMReD oracle finetune alone would be a useful finding, not yet the general method the user seeks. A final plain/reasoning study must use each backbone's native modes and charge all token, latent, vision and prefill work.

## Relation to earlier project experiments

The fixed-summary replacement branch remains closed at N32 61% ordinary versus 32% normalized and 28% mass. V7/V8 used external local-feature summation and consistency losses; V13 attempted background correction. V18's narrow success admits an essentially rank-one weighted-count interpretation. These are not demonstrations of native prefix-state learning on the competent original-MMReD joint model. The V2 question-prefix check was explicitly frozen prompting, not a finetuned test; its negative result remains unchanged and does not establish that causality is the bottleneck.

No old failure is erased or rescued by this proposal. All GPU and heavy CPU execution remains through Slurm, with models and data in the user-designated roots and at most four project GPUs.

# Thesis Proposal — Background and Proposed Approach

*Identifying and Relieving the Aggregation Bottleneck (Over-Squashing) in Vision-Language Transformers*

Tal Gorbunov · MSc Thesis Proposal · July 2026

## 1. Background: the over-squashing problem in vision-language models

Modern vision-language models (VLMs) such as Qwen2.5-VL perform impressively on single-image perception, yet fail at deceptively simple tasks that require *aggregating* evidence across many images or video frames — "how many times does the person pour water?", "how many rooms did the character visit?". On published video benchmarks, state-of-the-art models score 31–42% on action counting, barely above chance, and providing the model with oracle evidence frames does not close the gap: the failure lies in *fusing* the evidence, not in finding it.

This thesis studies that failure through the lens of graph neural networks. Self-attention is message passing on a graph, and the GNN literature identifies a characteristic pathology of message passing called **over-squashing** (Alon & Yahav, 2021): when information from a growing set of source nodes must pass through fixed-size representation bottlenecks, it is compressed beyond recovery. Barbero et al. (2024, arXiv:2406.04267) showed that decoder-only transformers inherit this pathology. In VLMs we find that the aggregation-relevant information flows along one specific, narrow path: frame tokens → a handful of **"carrier" question tokens** (the tokens naming the queried entity) → the final token that emits the answer. Every frame's evidence must squeeze through this carrier bottleneck within a single forward pass — and softmax attention combines the incoming evidence as a *normalized weighted mean*, an aggregator that is provably count-blind (a mean cannot represent "how many").

The behavioral signature is stark. Frozen Qwen2.5-VL-7B answers our controlled counting task at ~85–100% accuracy with 1–2 frames, ~20–30% at 8 frames, and ~2% at 128 frames — while linear probes show that the per-frame evidence is still present and decodable in the model's activations at every length. It is an aggregation failure, not a perception failure, and it appears across tasks (counting, set-cardinality, co-occurrence) and model families.

## 2. Benchmarks: MMRED and HERBench

**MMRED** (our controlled diagnostic) is a self-generated dataset of frame sequences showing characters in rooms, asking "how many frames was character C in room R?" (answer 0–8). Because each frame's evidence status is binary and known exactly, it supports mechanistic measurement: every accuracy number can be decomposed frame by frame. The family extends to sequence lengths up to 128 frames, to sibling tasks probing other aggregation operators (rooms-visited = set cardinality; co-occupancy = a two-character predicate), and to a photographic natural-image variant.

**HERBench** (arXiv:2512.14870) is the real-video anchor: a recent benchmark over HD-EPIC egocentric kitchen videos whose action-counting split provides one annotated timestamp per occurrence — i.e., exact per-frame evidence ground truth on real footage. We ported its action-counting subset into our instrument pipeline, including an evidence-only arm in which every shown frame is a true occurrence, removing retrieval and selection entirely. The frozen model achieves 4.9% exact-match there, undercounts by 3.5 on average, and scores 0% whenever the true count is ≥2 — its prediction saturates at ~2–3 whether 4 or 10 occurrences are shown. This replicates the MMRED wall on data we did not generate, and lands inside HERBench's published 31–42% SOTA band. What HERBench reports as a "fusion deficit" is precisely the aggregation bottleneck we characterize — now with a mechanistic account behind it.

## 3. Our instrument: the carrier separability metric (d′) and the √N law

The core measurement tool of the thesis is a **separability metric on the carrier pathway**. For each frame we extract the attention "message" it writes into the carrier token at a mid-network layer, and measure the whitened separability **d′** between evidence and non-evidence messages (held-out shrinkage-LDA projections, cross-checked against √2·Φ⁻¹(AUC)). Intuitively, d′ asks: *how cleanly does one frame's worth of evidence arrive at the aggregation site?*

The metric earns its place through a parameter-free prediction. If N noisy messages of separability d′ are summed by attention, the best linear readout of the pooled carrier attains accuracy 2Φ(d′/2√N) − 1 (mixed over the count prior). This **√N law** matches measured probe accuracy within ~1 point across tasks, layers, and sequence lengths on MMRED, with zero fitted parameters. An accompanying adequacy battery (Gaussianity and variance diagnostics on the messages) self-diagnoses where the closed form is licensed — on HERBench it correctly *rejects* the closed form, because real-video evidence is graded rather than binary. The framework thus does more than fit one dataset: it identifies which regime a task is in.

## 4. Insights so far

Our measurements decompose the failure into a three-stage account, each stage established causally:

1. **Supply — the joint-context tax.** Per-frame evidence reaches the carrier at d′ ≈ 2 when frames are processed jointly in one context, versus d′ ≈ 7–8 when each frame is processed alone — a ~3.4× tax that is flat from N=8 to N=128 and reproduces across modalities (images, text) and model families (Qwen2.5-VL, InternVL). We localized it: the carrier's *single query vector* must encode frame-specific addressing for all N frames at once, and this addressing degrades from the first companion frame onward. It is not attention-mass competition (causally refuted) and cannot be removed by masking cross-frame attention at any depth — in a frozen model, interference and computation share the same edges.

2. **Aggregation — the √N law.** Given the supplied d′, the law bounds what any linear readout of the pooled carrier can achieve; the frozen model tracks this ceiling.

3. **Readout — the clamp and the token interface.** The model reads the pooled carrier along a fixed, slightly misaligned axis whose emitted range is clamped at ~3–5 regardless of N; composing the law with the measured clamp reproduces the model's entire behavioral collapse with zero fitted parameters. Moreover the readout interface is discrete tokens only: real digit tokens verbalize out-of-distribution counts perfectly, while every learned continuous injection route we trained memorizes its training values and fails out-of-distribution.

Two constructive results confirm the diagnosis. First, replacing the implicit mean with an explicit **unnormalized sum** — a small DeepSets-style adapter on the frozen model — solves evidence-only counting completely (100%, including extrapolation to unseen counts and lengths); ablations isolate normalization as the causal failure (+24 points) and a capacity knee exactly at width = maximum count. Second, a decide-per-frame-then-reduce pipeline (the frozen model verifying each frame separately, then reducing) sustains 0.79–0.88 exact-match up to N=128 where the frozen model scores 0.02 — but costs ~N forward passes, a cost floor that itself follows from the addressing mechanism.

## 5. Proposed approach: a learned carrier per frame

The mechanism dictates the remedy: if one carrier query cannot address N frames, give each frame its own carrier — inside a single forward pass.

**Stage 1 (measured).** Interleaving a replica of the question after every frame, combined with a block-diagonal attention fence and a per-block position reset, makes each (frame + replica) block computationally identical to an isolated forward: supply reaches d′ 6.34 in one forward (isolated-forward parity), and a question-first ordering raises it to 9.2–12.7, flat to N=128. We then *distill the entire replica into a single learned carrier token* — one trained embedding of 3,584 parameters appended after each frame — which recovers 92–93% of the replica's separability, generalizes across lengths (trained at N=8, d′ 9.7 at N=128) and transfers zero-shot to a different task at 88% of its ceiling.

**Stage 2 (the proposed method).** A "carrier layer" architecture: frames and their carrier tokens are fenced up to mid-depth; above it, carriers attend one another and the answer region reads them, with a small zero-initialized LoRA (~2M parameters on the upper layers) trained under a plain language-modeling loss — so the frozen model emits the answer through its own head. Design goals, each pre-registered with evaluation bands: (a) *task-agnostic* — no hand-built gate or tally; the question in context is the only task signal; (b) *length-extrapolating* — train at N=8, evaluate at N=32 and 128; (c) *cross-frame reasoning* — set-union tasks that a per-frame tally provably cannot express. Preliminary runs reach ~0.7–0.85 and are demonstrably data-limited (a scaffold pipeline on identical data reaches 0.998), and a no-LoRA ablation plateaus at ~0.5 — cross-carrier attention alone is insufficient; trained integration is necessary.

The evaluation plan spans the MMRED family under the full count prior, length and task generalization, a second model family (InternVL), and HERBench as the honest boundary case, where the theory predicts a null: on real video the binding constraint moves to per-frame perception, which no aggregation fix should recover.

## 6. Expected contributions

1. A mechanistic, causally-grounded account of why frozen VLMs fail at multi-frame aggregation: over-squashing relocated to carrier addressing, with measured constants and a zero-parameter behavioral closure.
2. The carrier separability (d′) framework — metric, √N law, and adequacy battery — as a portable instrument that also classifies which regime (aggregation-limited vs perception-limited) a benchmark occupies.
3. The MMRED benchmark family and a two-regime transfer study on HERBench.
4. A parameter-efficient architectural fix — a learned carrier token per frame with a light cross-carrier integration layer — that restores aggregation in one forward pass on a frozen backbone.

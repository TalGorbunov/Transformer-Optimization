# Can a reasoning model supply the same local semantic measurement?

The V18 experiment tests a counting-specific gate derived from the plain model's native local judgment. A reasoning model needs a separate check. The existing Cosmos stream asks both local and global rows to begin with a thought, so probabilities of the next tokens `0` and `1` have not been validated as relevance measurements under that prompt. Its saved global trajectories begin with reasoning; those global outputs say nothing decisive about the local digit probabilities.

The proposed first audit compares two local prompt policies while holding the global prompt fixed:

- **Reasoning local:** each image uses the existing Cosmos reasoning system and question prompt.
- **Direct local:** each image uses the unchanged canonical one-image count prompt and ordinary helpful-assistant system.

The text-only global row uses the pinned official Cosmos reasoning system in both cases. Both policies see the same old N16/K3 and N64/K6 software scenes, with canonical pixels and Step labels. They are diagnostic examples already used in software work. No new efficacy test is consumed.

There are exactly four bare prefills, ordered N16 reasoning/direct, then N64 reasoning/direct. No trainable branch, generated token, forced answer boundary, alternative gate, or global-answer score is involved. Save every row's actual native pre-final hidden state, normalized state and full-vocabulary logits. A same-shaped native norm/head replay for each prefill verifies all164 local/global rows against the actual forward; every row must preserve top1 and have total variation at most0.02. Both actual norm and head must be FP16, matching the previously observed Cosmos load, with new model-specific identities. The compatibility config alone is insufficient evidence of actual dtype.

The fixed gate is `max(0, p(1) - p(0))`, using FP64 normalization over the full native vocabulary and then detached FP32 storage. The independent CPU report recomputes probabilities from saved FP16 logits and evaluates all160 local rows. It reports native argmax and binary sign, total digit mass, positive attenuation and negative leakage, split by length, Step<=16/>16 and existing semantic negative categories. Gold labels are used only for these reported checks, never to calculate or change a gate.

The direct-local feasibility criterion is fixed before execution: every observed positive gate is greater than zero, every observed negative gate is exactly zero, and the minimum full-vocabulary digit mass is at least0.95. The reasoning-local condition is descriptive. Computational correctness and semantic feasibility are separate decisions. Passing on these two scenes permits a further software check; it does not establish robust classification, aggregation accuracy or reasoning benefit.

CPU preparation must pin source, model mirror, processor and all prepared inputs, and verify manual mixed-row packing against the ordinary processor. CPU check/report each use four cores,16GB and at most300seconds through Slurm. Each GPU allocation uses one B200 for at most180seconds, with a complete540GPU-second campaign cap including failed and zero allocations. The registered inventory is four VLM forwards, four vision forwards, four ordinary native norm/head calls, and four additional same-state replay norm/head calls. All raw artifacts are retained. No GPU job is submitted while the four V18 main runs occupy the project's four-GPU limit.

A later bounded streaming smoke would check one original gate probe per trajectory, unchanged gate reuse, actual emitted-token broadcasting, mixed-prompt native positions and caches, and global-only writes. Direct local rows would then receive reasoning continuations that differ from their initial output instruction. Correct software execution would not establish useful vector payloads at those positions. V18 Qwen gates and checkpoints cannot substitute for Cosmos-specific evidence.

The eventual composition experiment needs gate/all-open crossed with direct/reasoning on the same reasoning backbone and fresh held-out scenes. A gain within reasoning tests whether aggregation benefits persist. Reasoning improving on the same method's direct policy is additional evidence of useful composition. Final-norm residual writes do not enter decoder KV, so answer-only CE cannot train earlier continuous residuals through future hidden states. Thought-position supervision or a separately validated persistent-memory design would need its own explicit experiment.

This file is a prospective design. Source review, frozen preparation and resource checks precede execution; it releases no training or long reasoning run.

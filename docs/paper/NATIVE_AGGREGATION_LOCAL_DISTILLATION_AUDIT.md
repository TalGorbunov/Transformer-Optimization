# Local-to-context distillation: positioning audit

The proposed teacher-on-an-isolated-frame, student-on-full-context KL objective is an established short-to-long distillation approach. It is a possible baseline, not a new aggregation method.

| Primary source | Closest overlap and relevant distinction |
|---|---|
| [OPSDL (2026), sections 3.1–3.4](https://arxiv.org/html/2604.17535v1) | Uses reliable short-context model behavior to supervise full-context responses to the same question, through on-policy token-level reverse-KL optimization. Frozen versus updated teachers, forward versus reverse KL, and deterministic versus sampled trajectories are ordinary distillation choices. |
| [LongPO (ICLR 2025), sections 3.1–3.3](https://arxiv.org/html/2502.13922v1) | Constructs short/long answer preferences and constrains the long-context policy toward a short-context reference. Both the competence-gap motivation and cross-context alignment precede our proposal. |
| [CLIPSelf, section 3.2](https://arxiv.org/html/2310.01403v2) | A cropped-region teacher representation supervises the corresponding region within full-image student features. Query-conditioned answer distributions across frames differ from its feature-level objective; local-to-context supervision itself is established. |
| [RAFT, sections 4.5 and 5](https://arxiv.org/html/2403.10131v1) | Trains contextual QA with relevant and distracting documents and supervised reasoning. Extra contextual QA is a necessary alternative explanation for any gain from our auxiliary local questions. |
| [EASE-TTT, sections 4.1–4.3](https://arxiv.org/html/2606.06906v1) | Aligns attention with selected evidence positions using test-time updates. This distinguishes attention supervision from answer-distribution supervision, and extra inference optimization from an offline-trained method. |

The teacher's privileged input is a clean localization of facts already present in the student context. Exact frame correspondence is training-time localization information and must be disclosed.

For local truths b_j, indexed QA predicts b_j, existence computes OR_j b_j, and counting computes SUM_j b_j. An isolated local answer can supervise the same indexed question in full context. It cannot in general supervise the global count; a local negative cannot even certify global absence. Consequently better indexed judgments would have to TRANSFER to ordinary unindexed counting before supporting an aggregation claim.

The completed V3 frame diagnostic establishes an isolated-to-context gap on indexed questions, with an additional reference-resolution demand. An index-free existence comparison on fixed K0/K1 negative extensions could test whether that demand explains the gap, but remains a task contrast rather than exclusive circuit localization. No such result is established yet.

A useful paper contribution would require a specific new insight supported by native counting transfer beyond scene diversity, extra contextual QA and existing short-to-long distillation controls, followed by matched reasoning-composition evidence. The normalized adapter and generic KL supply no cardinality-preservation guarantee. Current experiments have not established that contribution.

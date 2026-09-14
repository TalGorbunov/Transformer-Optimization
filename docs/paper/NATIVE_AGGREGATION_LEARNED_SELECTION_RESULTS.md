# Learned selection: verified identity-join failure

The fixed six-fit experiment failed both its primary comparison and practical criteria. The independent [report summary](../../outputs/native_aggregation_vlm/identity_join_learned/reporting/report_443105/summary.json) passed its computational and provenance checks; this is a valid negative efficacy result. The previously conditional joint-control and multiple-request releases remain stopped.

All entries below are complete native name-plus-EOS answers, out of 108. “Held” refers to held-out room pairs; the nine names remain familiar.

| Method | Seed | Final N16 dev | Seen N32 | Seen N64 | Held N32 | Held N64 |
|---|---:|---:|---:|---:|---:|---:|
| Clip | 22 | 12 | 12 | 12 | 12 | 12 |
| Sigmoid | 22 | 15 | 17 | 18 | 15 | 17 |
| Softmax | 22 | 22 | 22 | 21 | 20 | 15 |
| Clip | 23 | 12 | 12 | 12 | 12 | 12 |
| Sigmoid | 23 | 15 | 16 | 16 | 15 | 17 |
| Softmax | 23 | 26 | 31 | 30 | 27 | 27 |

Every test cell has 108 completed, parseable outputs and zero truncations. Every cell also has **0/36 completely correct three-variant families**. Consequently, formatting does not explain the failure, and the experiment does not demonstrate reliable association joining. The weak N16 development results already occur at a training context length: this is broader than failure to extrapolate to N64. Full outcomes and per-name scores remain in the [independent analysis](../../outputs/native_aggregation_vlm/identity_join_learned/reporting/report_443105/analysis.json).

## Training did not establish successful identification

All runs completed the same 4,536 updates, 72,576 scene presentations and 161,280 target positions. Checkpoints were fixed-final, with one descriptive development evaluation and no accuracy-based selection.

The following are **terminal training-batch** losses, not a full training-set evaluation. First-token CE is the mean over that batch's 16 original-prefix queries.

| Method | Seed | Sequence-balanced CE | First-token CE |
|---|---:|---:|---:|
| Clip | 22 | 1.041483 | 2.192788 |
| Sigmoid | 22 | 1.030648 | 2.162638 |
| Softmax | 22 | 0.997869 | 2.088256 |
| Clip | 23 | 1.034275 | 2.164011 |
| Sigmoid | 23 | 1.012644 | 2.102907 |
| Softmax | 23 | 0.999632 | 2.087458 |

Mean CE on subsequent teacher-forced tokens was only approximately 0.000006–0.000027. Thus sequence CE near one conceals poor first-name prediction behind easy continuation/EOS prediction. Terminal residual-consistency losses were small, approximately 4.8e-8–8.4e-5; their small endpoint values do not establish whether the regularizer caused collapse during optimization. Existing [training logs](../../outputs/native_aggregation_vlm/identity_join_learned/training/) supply these scalars; no new evaluation was run.

## Two distinct failure observations

The saved [capture geometry](../../outputs/native_aggregation_vlm/identity_join_learned/reporting/report_443105/geometry.json) establishes exact first-token gate collapse for **both clipped models on all 432 test scenes per model**. All six relevant images are closed in every scene; the positive and irrelevant message sums, and both projected sums, are exactly zero. Since the global stream contains only the question, this removes the image contribution to the first-name prediction. All 16 original-prefix queries in each clipped model's terminal training batch are also closed. Across all executed test prefixes, complete closure occurs in 692/864 records for seed 22 and 856/864 for seed 23; later reopening does not undo an incorrect first emitted name token.

The smooth controls do not share this exact collapse: none of their first-prefix gates is exactly zero. Nevertheless, their development and complete-family results remain poor. Their geometry also retains nuisance contribution or dilution: for sigmoid seed 22, first-prefix projected irrelevant-message norm is approximately 7.745 at N32 and 17.277 at N64 across the scenes; softmax bounds total gate mass but does not guarantee that this mass selects the evidence needed for a join. These observations do not distinguish weak cached local semantics from a deficient aggregate/readout mapping or optimization. They do not prove that clipping alone caused every failure.

## Implication and next diagnostic

V18 counting success was compatible with a question-dependent, nearly constant payload and weighted counting. This identity task supplies no corresponding evidence of useful association bandwidth. Continuous-memory software tests establish a causal path for gradients, not an efficacy repair for these first-token failures.

The chosen next step is a **separate, smaller trainability diagnostic**: clip/sigmoid crossed with CE-only/CE-plus-consistency, seed 24, 18 complete balanced training families comprising 108 contexts, and 600 fixed updates. Final native evaluations cover all 108 training contexts and 54 N16 development contexts. Keeping the cached inputs fixed tests whether either the dead-zone choice or the current regularizer prevents fitting this bounded subset, before paying for a joint-input cache. Native whole-name accuracy, first-token losses and gate/gradient diagnostics must be considered together. One seed and a small subset cannot establish robust generalization, and short/long name tokenization must retain the declared scene-balanced reduction.

A later CE-only joint-native versus parallel-adapter comparison should precede multiple requests if trainability is established. It would need matched parameter and training budgets, actual native norm/head scoring and fresh complete families. Joint inputs change the global state across lengths, so the present identical-global consistency contract is inapplicable there. Joint success with parallel failure would narrow the problem toward the parallel representation/readout interface. Multiple requests would add capacity and an interface change before this simpler issue is resolved. These are separate prospective decisions; this results note releases no jobs and makes no new method or reasoning-composition claim.

The campaign used **11,508 allocated GPU-seconds**, including failures, with maximum concurrency four. Verified analysis SHA256: `938216598c2963458b55b041534d9b1cd8adb27d7f2a29acad21b05aab4f8aea`; geometry SHA256: `9ae358502ab9c6769e529197a825a56ba39c8da887581b5b65940656f38993e1`. Frozen sources and existing results were unchanged.

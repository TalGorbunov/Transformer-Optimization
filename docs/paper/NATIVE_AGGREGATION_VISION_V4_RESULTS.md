# V4 results: matched scene refresh

V4 fails its preregistered two-seed screen. More training scenes substantially help seed 3, but seed 2 gains only 2.31 percentage points on familiar-count N32/N64 and becomes less precise. The pooled positive contrast is descriptive and cannot replace the required success in both seeds. This is evidence about ordinary supervised data diversity under one training law, not an established aggregation method.

Main array **440986** and independent CPU report **441013** completed. Canonical [report](../../outputs/native_aggregation_vlm/v4/REPORT.md), [analysis](../../outputs/native_aggregation_vlm/v4/analysis.json), and [figure](../../outputs/native_aggregation_vlm/v4/comparison.pdf) contain every cell and provenance check. The analysis SHA256 is **2e7b3d0c94ac7ebf4f0c4a6325935bacb5d705047ee7b1cb008bf209cddfef15**. This note adds interpretation and a read-only reproducibility investigation; it changes no source, checkpoint, ledger, parser, selection rule, or result.

## What was compared

Both conditions used the same frozen Qwen2.5-VL-7B NF4/bf16 backbone, hidden-only rank96 adapter at layer14, upper-four-layer LoRA rank8/alpha16, native final-answer/EOS cross-entropy and ordinary inference. There were nine blocks of180 presentations,405 optimizer updates and19,440 training frames. The original V2 training180 formed block0 in both conditions. Repeat reused them; refresh replaced170 nonsaturated slots in each subsequent block. Ten deterministic N8/K8 slots necessarily repeated, giving **1540 distinct refresh scenes versus180**, at1620 presentations in either condition.

Slot N, gold count, question, target character/room, order, answer/EOS IDs and actual processed prompt-token layouts matched. Initial trainable tensor hashes matched within seed. All four runs shared recorded library versions and processor/forward/decoding settings. Development exact accuracy, then first-answer-token NLL, then earliest tie selected one checkpoint per run after each45 updates.

Evaluation reused V2 scenes:108 each at N16/N32/N64 with K0..8, and64 each at N32/N64 with K9..16. These are **exploratory reused tests**, despite freezing this new comparison before V4 outcomes.

## Accuracy and precision

| Condition | Seed | N16 /108 | N32 /108 | N64 /108 | Familiar OOD /216 | Unseen K9–16 /128 | Selected block |
|---|---:|---:|---:|---:|---:|---:|---:|
| Repeat | 2 | 53 (49.07%) | 35 (32.41%) | 24 (22.22%) | 59 (27.31%) | 1 (0.78%) | 7 |
| Refresh | 2 | 63 (58.33%) | 41 (37.96%) | 23 (21.30%) | 64 (29.63%) | 0 (0%) | 7 |
| Repeat | 3 | 51 (47.22%) | 38 (35.19%) | 22 (20.37%) | 60 (27.78%) | 9 (7.03%) | 5 |
| Refresh | 3 | 75 (69.44%) | 60 (55.56%) | 43 (39.81%) | 103 (47.69%) | 12 (9.38%) | 9 |

| Refresh minus repeat | Familiar OOD gain | Paired-anchor95% interval | N16 gain | Registered screen |
|---|---:|---:|---:|---|
| Seed2 | +2.31pp | [−4.63,+9.26]pp | +9.26pp | Fail |
| Seed3 | +19.91pp | [+11.11,+28.24]pp | +22.22pp | Pass |
| Pooled fixed seeds | +11.11pp | [+5.56,+16.67]pp | +15.74pp | Descriptive only |

The criterion required at least5pp familiar OOD improvement and at most5pp N16 loss **in each seed**. All intervals use10,000 whole-anchor draws, seed20260914. The pooled draw preserves both lengths and both fixed seeds. It does not quantify variability over new training executions or training seeds.

Every test output parsed, so the following MAEs cover all selected examples.

| Condition/seed | Familiar OOD MAE | Familiar OOD signed bias | Unseen-count MAE | N16/N64 both correct /108 | Same N16/N64 prediction /108 |
|---|---:|---:|---:|---:|---:|
| Repeat2 | 1.24537 | −0.56019 | 4.32031 | 17 | 35 |
| Refresh2 | 1.58333 | −1.44444 | 5.41406 | 17 | 23 |
| Repeat3 | 1.46759 | −0.87500 | 2.91406 | 17 | 26 |
| Refresh3 | 0.82407 | −0.22222 | 2.38281 | 34 | 42 |

Seed2's additional five exact OOD answers conceal larger errors elsewhere. At N64 its exact count falls24→23, MAE rises1.50926→2.06481 and bias worsens−0.65741→−1.93519. Its N16-to-N64 accuracy gap widens26.85→37.04pp. This does not support improved extrapolation precision.

Seed3 improves exact accuracy and MAE at each familiar length; OOD gains comprise59 refresh-only correct answers against16 repeat-only correct answers. However, N16 improves slightly more than N64: the N16-to-N64 gap widens26.85→29.63pp. Thus it improves the accuracy level without demonstrating a flatter length-degradation curve. Unseen-count performance remains weak: all four runs score0/16 at K9; the nonzero K9–16 scores arise entirely from K10–16. These outcomes do not establish a general counting algorithm.

## Why identical shared-block inputs did not yield identical training

The first-block presentation ledgers show an informative boundary: **presentations1–4 have exactly equal CE within each seed; presentation5 is the first difference in both seeds**. Accumulation4 means these first four forwards precede optimizer update1. All176 later first-block losses differ, although scene IDs, QA/image identities, order and initialization still match.

The step1 logs already report slightly different pre-clipping gradient norms despite identical mean CE:

| Seed | Shared step1 CE | Repeat gradient norm | Refresh gradient norm |
|---|---:|---:|---:|
| 2 | 0.9488917291 | 0.9971750975 | 0.9971373677 |
| 3 | 0.5262378082 | 0.6972808242 | 0.6972901821 |

Both norms are below clip1, so a clipping-threshold change does not explain the first divergence. Across all180 first-block presentations, mean absolute CE differences are0.0163696 and0.00728484 for seeds2/3; maxima are0.149855 and0.0650147. At the end of that still-shared block, pooled development accuracy already differs49/72 versus45/72 in seed2 and39/72 versus37/72 in seed3.

Ordinary stochastic dropout is not supported by the inspected implementation:

- The cached Qwen model config sets language attention dropout to0.0; installed Transformers4.57.6 passes this value to SDPA even in training.
- Vision attention dropout is hardcoded0.0, and the runner explicitly keeps the frozen visual encoder in evaluation mode.
- The [hidden adapter](../../gnnformer/aggregation_controls.py) and [handwritten LoRA](../../gnnformer/carriers.py) contain no dropout. The [runner](../../scripts/native_aggregation_vlm_v4.py) seeds Python and PyTorch, resets the seed before LoRA construction, and uses deterministic slot-order construction.

The pattern is consistent with numerical nondeterminism in backward/reduction/update computations being amplified during training. It is **not a kernel-level diagnosis**: scalar losses and gradient norms do not identify which tensor or operation first differs. The frozen runner and wrapper do not enable deterministic algorithms or pin an SDPA backend; they do not record first-step gradient/tensor hashes, selected CUDA kernels or numerical Adam states. Matching scalar losses also does not establish equality of all hidden activations. PyTorch separates RNG seeding from deterministic-operation controls, and its SDPA documentation allows backend-dependent floating-point results and certain nondeterministic CUDA execution. [PyTorch2.8 reproducibility](https://docs.pytorch.org/docs/2.8/notes/randomness.html), [PyTorch2.8 SDPA](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.scaled_dot_product_attention.html).

Consequently, these are matched nominal-seed runs, **not an exact shared-trajectory intervention**. Execution variability exists before refresh begins, and the current four runs cannot estimate how much of the final contrast it explains. Neither “seed3 proves refresh alone caused19.91pp” nor “nondeterminism explains away the gain” follows. The registered failure remains unchanged.

## What follows

Retain scene diversity as a promising ordinary-training explanation, with substantial execution/seed dependence and no robust unseen-count repair. Do not relabel the pooled result a passed screen, claim architecture novelty, or infer composition with reasoning tokens. Ordinary adaptation and data controls remain necessary baselines for any later operator.

Before attributing another closely matched training comparison to a mechanism, the smallest diagnostic would replay the same first optimizer step from identical initialization in separate executions and record initialized tensors, per-example losses, accumulated gradient hashes, post-update hashes and effective deterministic/backend settings. This is a proposed numerical audit, not an experiment run here; enabling deterministic mode would be a new recorded execution condition and might expose unsupported kernels. No frozen V4 result should be replaced.

Any later efficacy claim still needs fresh evaluation scenes and independent training executions under a frozen protocol. V4 itself provides neither fresh confirmation nor evidence that extra reasoning tokens compose with the training change.

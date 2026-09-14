# Conditional decision after orientation coverage

HELD research design only; written without inspecting the pending orientation-training or confirmation scores. This note does not release data preparation, features, fits or inference, or change the [current study](NATIVE_AGGREGATION_FACTOR_ORIENTATION_PROPOSAL.md).

If native training succeeds and new panel A succeeds but B fails, broader composition coverage is a concrete next hypothesis. A failure as well would leave within-support realization transfer unresolved; a C-only failure concerns question transfer. If A/B succeed, follow the existing length gate instead of adding this study. Training failure cannot establish a generalization failure.

## One broader-support experiment

Keep the original product and additive architectures, original unfitted tensors, global conditioning statistics, fixed half-gates, native norm/head, mean-scene full-name-plus-EOS CE and seed24. Reserve person trios **(0,3,6), (1,4,7), (2,5,8)**, indexed by the frozen nine-person order, from every fitted update. Train on the other **81 trios ×12 original TRAIN_PAIRS**. Each person remains equally represented; every constituent person pair occurs in training, while the three reserved joint trios do not.

The [canonical stager](../../scripts/stage_native_identity_join.py) supplies5832 of the existing6048 training scenes for this split. Include their5832 target-independent requested-room swaps: **11664 scenes,972 abstract families,2916 original N8/N16 base pairs**. Retain both orientations and all three changing answers. Reuse exact native feature rows wherever their image/question/strict-prefix identities match; harvest only independently verified missing rows. Cached rows from excluded trios must never enter updates. The fixed original statistics use three retained trios; do not refit them on held groups or questions.

Keep **6000 updates of eight length pairs**,48000 pair presentations and all6000 original learning rates. Repeatedly shuffle the2916 base-pair IDs with seed24 in canonical source order; alternate original/flipped on successive visits to each base pair. Both arms consume the same list. Targets and CE weighting follow the selected scenes; total vocabulary rows must be derived before release, not asserted identical to the small-support run. This fixes total optimization exposure while spreading it over more support. It deliberately does not promise the same exposure per family or convergence. Use only the final checkpoint; no restart, continuation or horizon rescue.

Before confirmation, audit all cached first queries: require **11124/11664 pooled,5562/5832 per orientation,864/972 complete12-context families**—the current training-gate proportions. Then test native whole answers on a fixed216-scene training sentinel: trios(0,4,8),(1,5,6),(2,3,7), crossed with the current six questions, both orientations and N8/N16. Require the current206/103-each/16-family gate. These trios are newly added training support. This checks sampled native trainability, not native success on all11664 scenes. If either cached arm passes, retain both through the native comparison; if either native arm passes, retain both for confirmation. Profiles, timing forecasts and finite resource caps need a separate reviewed release.

## Independent confirmation and interpretation

Freeze new concrete families before fitting, seed91726343, excluding every previously registered scene/triple, both earlier810-scene fresh manifests and all new training scenes. N16 comes first, with both orientations, three answer-changing variants and shared marginals/backgrounds within each triple:

- A: the three sentinel seen trios ×six current seen questions:108 scenes/36 triples.
- B: the three reserved trios ×those questions:108/36.
- Q: sentinel seen trios ×the original **Kitchen–Bathroom, Garden–Office, Bedroom–Park** held questions:54/18.
- C: reserved trios ×those held questions:54/18.

Require98 answers AND33 complete triples separately in A/B,49 AND16 separately in Q/C. Q isolates question transfer from the combined shift in C. The previous C questions are among the12 fitted questions here and must be relabeled seen. All holdout claims concern this fresh fit; earlier experiments already used some abstract groups, and elementary image atoms remain shared.

Native fit plus A success but B failure would reject the sufficiency of this near-exhaustive support expansion under the fixed recipe, not aggregation learnability in general. Cached/native fit failure instead leaves optimization unresolved. Success by both arms would not establish a product-specific benefit. Even all four panels passing tests compositional generalization with six relevant images; it does not measure general aggregation bandwidth, multiple-request capacity or reasoning composition.

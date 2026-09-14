# Fixed-marginal native vision binding diagnostic

| Selected model | Low K2 exact /16 | High K6 exact /16 | Parsed /32 | MAE (parsed) | Both pair answers correct /16 |
|---|---:|---:|---:|---:|---:|
| v2_global_seed0 | 4/16 | 4/16 | 32/32 | 0.875 | 0/16 |
| v2_hidden_seed0 | 2/16 | 4/16 | 32/32 | 0.938 | 1/16 |
| v3_lift_pre_seed0 | 4/16 | 7/16 | 32/32 | 0.719 | 0/16 |
| v3_lift_post_seed0 | 4/16 | 3/16 | 32/32 | 1.000 | 2/16 |
| v3_lift_pre_seed1 | 4/16 | 0/16 | 32/32 | 1.531 | 0/16 |
| v3_lift_post_seed1 | 1/16 | 2/16 | 32/32 | 1.250 | 0/16 |

| Selected model | Pairs both parsed /16 | Mean prediction Δ | Mean Δ error (target4) | Mean absolute Δ error | Positive Δ fraction | Invariant fraction |
|---|---:|---:|---:|---:|---:|---:|
| v2_global_seed0 | 16/16 | 4.375 | 0.375 | 1.125 | 1.000 | 0.000 |
| v2_hidden_seed0 | 16/16 | 4.250 | 0.250 | 0.875 | 1.000 | 0.000 |
| v3_lift_pre_seed0 | 16/16 | 4.188 | 0.188 | 1.062 | 1.000 | 0.000 |
| v3_lift_post_seed0 | 16/16 | 3.750 | -0.250 | 1.125 | 1.000 | 0.000 |
| v3_lift_pre_seed1 | 16/16 | 4.812 | 0.812 | 1.312 | 1.000 | 0.000 |
| v3_lift_post_seed1 | 16/16 | 4.375 | 0.375 | 1.250 | 1.000 | 0.000 |

Each Δ is high-K prediction minus low-K prediction. Missing parses remain incorrect; Δ summaries condition on complete pairs.

The known-generator oracle marginal moment 0.75(C+R)−0.5N equals2 for every scene here. It gets16/32 examples correct while never answering both members of a pair correctly. This reference is separate from the model table.

## Scope and provenance

- Six development-selected checkpoints evaluated without training or parameter updates; canonical analyses and checkpoints are hashed.
- Sixteen new matched N32 pairs: K2/K6, C=R=12, full character/room marginals fixed, eight room-switched images.
- All32 examples remain in exact-accuracy denominators; contrast means/fractions condition on both outputs parsing and report that count.
- Input-pair contrasts do not identify an internal causal mechanism or establish a new architecture/general aggregation algorithm.
- This fixed-contingency, coupled-pair diagnostic is OOD relative to the V2 training generator; negative outcomes can include distribution shift.
- Room swaps also change other associations; fixed marginals exclude marginal-only shortcuts but do not exclude every alternative shortcut.
- The diagnostic is small and descriptive, with no uncertainty interval or seed-population claim.
- The oracle marginal moment is always2, giving16/32 example exact but0/16 both-pair correct; it is not a competitive VLM baseline.

[Immutable plan](plan.json) · [Summary](summary.json) · [All predictions](predictions.json)

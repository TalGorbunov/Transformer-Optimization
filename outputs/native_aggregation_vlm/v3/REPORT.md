# Native nonlinear-value placement: V3

Exploratory reused-test comparison; development-selected checkpoints and two training seeds.

| Arm | Seed | N16 K0–8 | N32 K0–8 | N64 K0–8 | Familiar OOD | K9–16 | Epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| lift_pre | 0 | 60/108 (55.6%) | 49/108 (45.4%) | 29/108 (26.9%) | 78/216 (36.1%) | 19/128 (14.8%) | 7 |
| lift_pre | 1 | 43/108 (39.8%) | 27/108 (25.0%) | 17/108 (15.7%) | 44/216 (20.4%) | 17/128 (13.3%) | 4 |
| lift_post | 0 | 58/108 (53.7%) | 39/108 (36.1%) | 24/108 (22.2%) | 63/216 (29.2%) | 4/128 (3.1%) | 5 |
| lift_post | 1 | 46/108 (42.6%) | 25/108 (23.1%) | 19/108 (17.6%) | 44/216 (20.4%) | 18/128 (14.1%) | 3 |

Primary passes in both seeds: **False**.
Each familiar-length cell has108 examples; pooled N32/N64 has216 observations from108 paired anchors. The count test has128 observations from64 anchors.

| PRE minus POST | Familiar OOD gain | Paired-anchor 95% interval | N16 gain | Primary passes |
|---|---:|---:|---:|---:|
| seed0 | +6.9pp | [-0.5, +14.4]pp | +1.9pp | True |
| seed1 | +0.0pp | [-7.4, +7.4]pp | -2.8pp | False |
| pooled fixed seeds | +3.5pp | [-2.3, +9.0]pp | -0.5pp | not a decision criterion |

The pooled contrast preserves both seeds and both lengths within each resampled anchor. It cannot replace a failure in either seed.

## Historical reference and unseen counts

- PRE seed0 minus historical V2 hidden seed0: familiar OOD +6.9pp, interval [+0.0, +13.9]pp; N16 +9.3pp; V3.2=True.
- PRE seed1 minus historical V2 hidden seed0: familiar OOD -8.8pp, interval [-15.3, -2.8]pp; N16 -6.5pp; descriptive only; not a registered criterion.

| Arm/seed | K9 exact /16 | K10–16 exact /112 | Count parse rate | Count MAE |
|---|---:|---:|---:|---:|
| lift_pre/seed0 | 0/16 | 19/112 | 100.0% | 2.094 |
| lift_pre/seed1 | 0/16 | 17/112 | 100.0% | 3.094 |
| lift_post/seed0 | 0/16 | 4/112 | 100.0% | 3.555 |
| lift_post/seed1 | 0/16 | 18/112 | 100.0% | 2.188 |

## Measured implementation cost

| Arm/seed | Familiar OOD model sec/example | Total sec/example | Peak training allocated GiB |
|---|---:|---:|---:|
| lift_pre/seed0 | 1.052 | 1.575 | 13.69 |
| lift_pre/seed1 | 1.065 | 1.578 | 13.69 |
| lift_post/seed0 | 1.052 | 1.567 | 13.73 |
| lift_post/seed1 | 1.061 | 1.583 | 13.73 |

Per-K outcomes, parse rates, MAE/bias, likelihoods, memory, paired extension transitions, checkpoint hashes and all contrasts are in [analysis.json](analysis.json).

## Interpretation limits

- Exploratory comparison on reused V2 test data; no new sealed test set.
- Primary criterion must hold separately in each seed: PRE minus POST familiar N32/N64 >=5pp, N16 loss <=5pp.
- Anchor bootstrap conditions on fixed trained models. Both lengths and both seeds stay together in the pooled resample.
- Two seeds are inadequate for population inference over training seeds; no seed bootstrap or independent-seed significance claim.
- Historical V2 hidden seed0 is a reused reference, not a contemporaneous third arm. V3.2 applies only to PRE seed0: familiar OOD gain >=5pp and N16 loss <=5pp; PRE seed1 versus this reference is descriptive only.
- GQA uses 4 KV heads and 28 query heads: PRE applies value SiLU on KV heads, POST applies SiLU on expanded query-head reads. PRE recomputes its lifted value activation over cached V at each decode step; activation work and cache costs are not matched.
- Lift and outer maps use fp32 while SDPA inputs use bf16. PRE and POST activation placement also straddles bf16 SDPA rounding; this is not a pure exact-arithmetic ordering contrast.
- Both arms initialize outer-up and lift-up to zero. This identically delays initial lift_down learning until those paths have nonzero weights; equal parameters do not imply every parameter updates on the first step.
- K9 is a single token; K10..16 are multi-digit. First-token NLL is not whole-answer likelihood.
- All selected examples remain in exact-accuracy denominators. MAE/bias are conditional on parsing and shown with parse rate.
- Timing and memory include concurrent cluster conditions; measured implementation cost is not an optimized algorithmic lower bound.
- Ordinary short native answers are evaluated; neither successful placement nor two-seed replication establishes reasoning composition or a general counting algorithm.

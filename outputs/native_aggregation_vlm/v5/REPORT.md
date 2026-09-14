# Native vision: extensive versus averaged independent evidence (V5)

Exploratory reused V2 tests; identical V4 refresh presentations, two fixed seeds and development-selected checkpoints.

| Pooling | Seed | N16 | N32 | N64 | Familiar OOD | Unseen K9–16 | Selected block |
|---|---:|---:|---:|---:|---:|---:|---:|
| sum | 4 | 33/108 (30.6%) | 17/108 (15.7%) | 13/108 (12.0%) | 30/216 (13.9%) | 0/128 (0.0%) | 6 |
| sum | 5 | 17/108 (15.7%) | 12/108 (11.1%) | 0/108 (0.0%) | 12/216 (5.6%) | 0/128 (0.0%) | 6 |
| mean | 4 | 42/108 (38.9%) | 25/108 (23.1%) | 22/108 (20.4%) | 47/216 (21.8%) | 0/128 (0.0%) | 7 |
| mean | 5 | 45/108 (41.7%) | 17/108 (15.7%) | 16/108 (14.8%) | 33/216 (15.3%) | 1/128 (0.8%) | 8 |

Both-seed primary screen: **False**.

| SUM minus MEAN | Familiar OOD difference | Paired-anchor95% interval | N16 difference | Pass |
|---|---:|---:|---:|---:|
| seed4 | -7.87pp | [-17.59,+1.39]pp | -8.33pp | False |
| seed5 | -9.72pp | [-14.35,-5.56]pp | -25.93pp | False |
| pooled fixed seeds | -8.80pp | [-15.28,-2.78]pp | -17.13pp | descriptive |

Every seed must improve familiar N32/N64 exact by at least5pp and lose at most5pp on N16. Pooling cannot rescue a failed seed.
Each run has216 familiar OOD observations from108 anchors and128 unseen-count observations from64 anchors. Pooled intervals retain both fixed seeds and lengths in each whole-anchor draw.

## Precision and extensions

| Pooling/seed | OOD parse | OOD MAE | OOD bias | Count MAE | Count bias | N16→N64 loss | Both correct /108 | Same parsed prediction /108 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sum/4 | 100.0% | 2.421 | 0.616 | 6.672 | -6.672 | +18.52pp | 3/108 | 15/108 |
| sum/5 | 45.4% | 3.980 | -3.980 | 12.095 | -12.095 | +15.74pp | 0/108 | 0/108 |
| mean/4 | 100.0% | 2.171 | -2.005 | 7.578 | -7.578 | +18.52pp | 15/108 | 25/108 |
| mean/5 | 100.0% | 2.699 | -2.468 | 7.805 | -7.664 | +26.85pp | 14/108 | 27/108 |

MAE/bias condition on parsed outputs; exact and invariance denominators retain invalid outputs. Per-cell and per-K results, count parse denominators, K9 versus K10–16, and all extension transitions are preserved in analysis.json.

## Training and gradient audit

Each run used the same1620 ordered presentations,1540 distinct scenes,19440 image frames,405 updates,37 persistent Adam states and nine72-example development evaluations. All452 selected-checkpoint predictions were independently verified. The branch has1,041,504 parameters plus720,896 upper-LoRA parameters,1,762,400 total.

| Pooling/seed | Clipped updates /405 | Mean branch preclip norm | Mean LoRA preclip norm | First-step branch hash matches /5 | First-step LoRA matches /32 |
|---|---:|---:|---:|---:|---:|
| sum/4 | 405/405 | 12.177 | 0.60098 | 4/5 | 31/32 |
| sum/5 | 405/405 | 13.522 | 1.1609 | 4/5 | 29/32 |
| mean/4 | 401/405 | 5.9008 | 0.98913 | 4/5 | 31/32 |
| mean/5 | 402/405 | 4.9353 | 0.87062 | 4/5 | 29/32 |

Initial parameter bytes, all scene/order/token/target ledgers and recorded runtime settings match within each seed. First-gradient hashes and preclip norms are reproducibility/optimization diagnostics; gradient tensors and Adam moments were not saved for independent numerical replay. All452 first-prefill diagnostic tensor files are hash-bound and independently checked for shapes, visibility, SUM/MEAN reconstruction and residual/native scalar norms; they do not select checkpoints.

## Measured cost

| Pooling/seed | OOD model sec/example | Total sec/example | Training seconds | Peak training allocated GiB |
|---|---:|---:|---:|---:|
| sum/4 | 1.089 | 1.586 | 773.6 | 13.58 |
| sum/5 | 1.135 | 1.630 | 772.3 | 13.58 |
| mean/4 | 1.090 | 1.588 | 774.1 | 13.58 |
| mean/5 | 1.091 | 1.592 | 781.4 | 13.58 |

## Computational and numerical audit

Exploratory-training engineering acceptance: **True**. Registered pre-training strict numerical gate passed: **False**. These are separate decisions. The native NF4/bf16/SDPA/bitsandbytes backend was retained.

| Pooling/seed | State integrity | Strict numerical observations passed | Overall strict result | Failed checks |
|---|---|---:|---|---|
| mean/4 | True | 5/5 | True | none |
| sum/4 | True | 5/5 | True | none |
| mean/5 | True | 5/5 | True | none |
| sum/5 | True | 5/5 | True | none |

All four development-selected checkpoints completed the same five cache comparisons and ten software evaluations. Every numerical failure is retained; audit results do not select checkpoints or alter main predictions. Complete raw/centered/probability metrics, state checks, software predictions and source/checkpoint hashes are in analysis.json. Five two-token probes do not establish general or long-reasoning cache equivalence.


## Interpretation limits

- V2/V3/V4 test scenes have already been inspected. These are reused exploratory evaluations, not fresh confirmation or a general benchmark win.
- At identical parameters/query, delta_SUM=N_visible*delta_MEAN. This tests extensive scaling and its optimization consequences; it does not establish increased information or bandwidth at fixed N.
- SUM/MEAN change branch-gradient scale, global clipping and relative branch/LoRA optimization. Identical nominal learning rates do not remove these confounds.
- Both arms access the same independently encoded visual memory. Their contrast cannot isolate memory source, contextual corruption, or the causal effect of independent encoding.
- The adapter uses established conditional Deep Sets and visual-memory ingredients, with particularly close PVM precedent. No new attention-family or broad persistent-memory novelty is claimed.
- Zero-read centering does not guarantee zero mean distractor messages. Distractor contributions can accumulate with length; query states remain contextual.
- Primary SUM-minus-MEAN familiar OOD >=5pp and N16 loss <=5pp must hold in EACH seed. Pooled effects cannot rescue a failed seed.
- Bootstrap resamples10000 whole anchor families with seed20260916, retaining both lengths and fixed seeds. Two seeds do not establish seed-population variance.
- Both arms use all1620 identical ordered presentations of1540 scenes, including ten deterministic N8/K8 scenes repeated across blocks. This is not a data-diversity contrast.
- All invalid outputs stay incorrect. MAE/bias are conditional on parsing; two invalid answers are not invariant predictions. K9 and K10–16 answer support are reported separately.
- Training initial hashes, first-step gradient hashes and cumulative Adam counters diagnose reproducibility; gradient tensors/momentum states were not retained for independent numerical replay.
- Initial absolute-RMS and subsequent raw native-reference software gates failed; those failures remain recorded and are not accepted by this report. The final registered gate centers cached/full logit differences, applying maximum <=min(0.25,1.5*OFF-centered maximum+0.0625), RMS <=min(0.05,1.5*OFF-centered RMS+0.005), total variation <=0.01 and identical enabled raw top-1. Raw common shifts/KL remain diagnostics. Five checks use two original and two previously unused software scenes plus an exact A/B/C/D/A reset (20 visual forwards); OFF references share the enabled first token. This bounded two-token software check is not exact numerical parity, long-reasoning validation or efficacy evidence.
- Exploratory training was explicitly accepted under schema2 while the strict numerical gate remained FAILED. The native NF4/bf16/SDPA/bitsandbytes backend was unchanged. Completion/state-integrity evidence and kernel localization support this engineering decision; they do not make the failed1%TV criterion pass. All four dev-selected checkpoints receive the same post-training numerical audit; failures remain reported and do not select/drop a model or main prediction.
- First-prefill frame-message and residual norms are detached descriptive diagnostics, not semantic evidence labels or causal attribution. The prototype is batch1, images only, unpadded greedy caching, without beams or mid-cache image insertion.
- Recorded model_seconds includes enabled GPU diagnostic capture; subsequent CPU tensor transfer, serialization and hashing are excluded. SUM/MEAN share instrumentation; cross-version timing against bare inference is not strictly matched. Native attention norm is measured before branch addition, not over the full residual stream.
- No intermediate reasoning tokens, frame-label supervision or external tally are used. This direct-answer experiment does not establish composition with reasoning.

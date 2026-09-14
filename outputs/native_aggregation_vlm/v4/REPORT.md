# Native vision: matched training diversity (V4)

Exploratory reused V2 test; nine equal-budget training blocks, two fixed seeds, development-selected checkpoints.

| Condition | Seed | N16 | N32 | N64 | Familiar OOD | Unseen K9–16 | Selected block |
|---|---:|---:|---:|---:|---:|---:|---:|
| repeat | 2 | 53/108 (49.1%) | 35/108 (32.4%) | 24/108 (22.2%) | 59/216 (27.3%) | 1/128 (0.8%) | 7 |
| repeat | 3 | 51/108 (47.2%) | 38/108 (35.2%) | 22/108 (20.4%) | 60/216 (27.8%) | 9/128 (7.0%) | 5 |
| refresh | 2 | 63/108 (58.3%) | 41/108 (38.0%) | 23/108 (21.3%) | 64/216 (29.6%) | 0/128 (0.0%) | 7 |
| refresh | 3 | 75/108 (69.4%) | 60/108 (55.6%) | 43/108 (39.8%) | 103/216 (47.7%) | 12/128 (9.4%) | 9 |

Both-seed primary screen: **False**.

| Refresh minus repeat | Familiar OOD gain | Paired-anchor 95% interval | N16 gain | Pass |
|---|---:|---:|---:|---:|
| seed2 | +2.31pp | [-4.63, +9.26]pp | +9.26pp | False |
| seed3 | +19.91pp | [+11.11, +28.24]pp | +22.22pp | True |
| pooled fixed seeds | +11.11pp | [+5.56, +16.67]pp | +15.74pp | descriptive |

Each seed must gain at least5pp on familiar N32/N64 and lose no more than5pp on N16. Pooling cannot rescue a failed seed.
Familiar OOD has216 observations from108 anchors per model; unseen counts have128 observations from64 anchors. The pooled interval resamples108 shared anchors while retaining both lengths and both fixed seeds.

## Precision and paired extensions

| Condition/seed | OOD parse | OOD MAE (parsed) | Count parse | Count MAE (parsed) | N16→N64 accuracy loss | N16/N64 both correct | Same parsed prediction /108 |
|---|---:|---:|---:|---:|---:|---:|---:|
| repeat/2 | 100.0% | 1.245 | 100.0% | 4.320 | +26.85pp | 17/108 | 35/108 |
| repeat/3 | 100.0% | 1.468 | 100.0% | 2.914 | +26.85pp | 17/108 | 26/108 |
| refresh/2 | 100.0% | 1.583 | 100.0% | 5.414 | +37.04pp | 17/108 | 23/108 |
| refresh/3 | 100.0% | 0.824 | 100.0% | 2.383 | +29.63pp | 34/108 | 42/108 |

## Training audit

Every run passed checks of1540 available training scenes,1620 actual presentations,19440 training frames,405 updates,35 persistent Adam step counters, all nine development evaluations, checkpoint selection and452 test predictions. Repeat used180 distinct scenes; refresh used1540, including ten saturated N8/K8 scenes repeated in both conditions.
Initial trainable tensor hashes, slot shuffles, ordered input-token hashes, image layouts and final-answer targets match across conditions within each seed. All four runs have identical recorded torch/transformers versions, image-processor settings, quantization, supervision, prompt, image-encoding and decoding configuration. Source manifests, CPU processor audit, copied run manifests, code snapshots and selected checkpoint hashes are bound in analysis.json.

| Seed | First-block mean refresh−repeat CE | Mean absolute CE difference | Maximum absolute CE difference | Exactly equal /180 |
|---|---:|---:|---:|---:|
| 2 | -0.0030960726 | 0.016369585 | 0.14985472 | 4/180 |
| 3 | 0.00090218718 | 0.0072848441 | 0.06501472 | 4/180 |

First-block losses are a descriptive reproducibility diagnostic, not an efficacy selection rule. Adam momentum tensors were not saved; the audit checks cumulative state counters plus the frozen optimizer implementation.

## Measured cost

| Condition/seed | OOD model sec/example | Total sec/example | Peak training allocated GiB |
|---|---:|---:|---:|
| repeat/2 | 1.051 | 1.542 | 13.55 |
| repeat/3 | 1.044 | 1.551 | 13.55 |
| refresh/2 | 1.060 | 1.559 | 13.55 |
| refresh/3 | 1.053 | 1.562 | 13.55 |

All cell/per-K outcomes, likelihoods, K9 versus K10–16 support, extension transitions, tokenization and provenance are in [analysis.json](analysis.json).

## Interpretation limits

- The V2 test scenes have been inspected in earlier blocks. This is exploratory reused-test evaluation; a positive effect needs fresh confirmation without further tuning.
- This isolates scene refresh under the same slot N/K/question, model, initial parameters, targets and405-update budget. It does not isolate diversity from the accompanying reduction in repetition.
- Refresh has1540 distinct scenes, not1620: ten N8/K8 fixed-question slots are deterministic and repeat in both conditions.
- Primary refresh-minus-repeat familiar OOD gain >=5pp and N16 loss <=5pp must hold in each seed. The pooled contrast cannot rescue a failure.
- Bootstrap uses10000 whole-anchor draws with seed20260914, retaining paired lengths and both fixed seeds together. It quantifies conditional test-anchor variation, not population training-seed uncertainty.
- All invalid answers remain incorrect. MAE and signed bias condition on parsing; two invalid answers do not count as prediction invariance.
- K9 is single-token while K10–16 are multi-digit. Raw first-answer-token NLL is not whole-answer likelihood.
- Saved cumulative Adam counters and frozen code establish the observed state-persistence checks; optimizer momentum tensors were not retained for an independent value-by-value audit.
- Timing includes concurrent cluster conditions. The inference architecture is identical across conditions.
- A refresh effect is an ordinary supervised-data result, not a novel aggregation operator or proof of general aggregation/reasoning composition. Failure does not authorize another architecture or learning-rate grid.

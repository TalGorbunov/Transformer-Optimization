# Extended optimization of the unchanged local joint-code oracle

Independent computation and provenance audits passed. The same privileged per-image semantic codes and four unfitted readout tensors were used. Both the optimization horizon and cosine annealing horizon changed from 600 to 6,000 updates.

| Endpoint | Correct /108 | Complete families /18 | Mean first-token NLL | Decision role |
|---|---:|---:|---:|---|
| 600 | 36 | 0 | 1.222148716 | Descriptive only |
| 2000 | 36 | 0 | 1.023820719 | Descriptive only |
| 6000 | 108 | 18 | 0.080282241 | PASS |

Only step 6,000 controls the unchanged 103/108 and 16/18-family screen. Every endpoint includes all 108 training contexts; all three raw archives are retained. Intermediate evaluations preserved parameters, optimizer state, parameter versions, native weights, inputs and training mode. No early score selected a checkpoint, restart or schedule change.

The longer horizon/schedule fits this privileged training screen, so the shorter recipe was insufficient relative to this control. This does not show that native visual states can supply the codes or that the join generalizes.

Full-name-plus-EOS CE uses the original mean-scene weighting and all 48,000 paired presentations. First-token, continuation and EOS losses are reported separately over the full trajectory and each complete cycle; the partial tail is retained. Those trends are descriptive, not convergence certificates. Prior failures and closed architecture branches remain unchanged; no subsequent fit or native/benchmark evaluation is automatically released.

Allocated GPU cost: 69 seconds. GPU work: 6,021 core/norm/head calls each and 213,654 head rows, with zero VLM/vision calls. CPU replay covered 21 endpoint batches / 324 rows; maximum full-vocabulary TV 0.003452907, fixed TV <= 0.02 and exact argmax. All 29 captured computations and semantic-code provenance were independently audited.

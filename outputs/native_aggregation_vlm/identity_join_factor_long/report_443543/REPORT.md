# Extended optimization of the locked factor pair

Independent computation and provenance audits passed. Both original interactions used the exact shared unfitted factor tensors, native cached features and global conditioning statistics. The query remains in both factor preactivations and in the readout. Both optimization and cosine annealing horizons changed from 600 to 6,000 updates.

| Interaction | Endpoint | Correct /108 | Complete families /18 | Mean first-token NLL | Decision role |
|---|---:|---:|---:|---:|---|
| product | 600 | 78 | 6 | 0.786239354 | Descriptive only |
| product | 2000 | 108 | 18 | 0.009728293 | Descriptive only |
| product | 6000 | 108 | 18 | 0.000216824 | PASS |
| additive | 600 | 36 | 0 | 1.153987323 | Descriptive only |
| additive | 2000 | 99 | 14 | 0.244380724 | Descriptive only |
| additive | 6000 | 108 | 18 | 0.004602016 | PASS |

Only each arm's fixed step-6,000 endpoint controls its unchanged 103/108 and 16/18-family screen. All six endpoint archives and all per-name/question/length/family strata remain available. Intermediate evaluations preserved optimizer state, parameter objects/versions, native weights, cached features, conditioning statistics and training mode. No score selected a checkpoint, restart or schedule change. No cyclic factor intervention was run in this calibration.

These are cached training first queries. A pass calibrates trainability of that fixed interaction under the expanded horizon/schedule; a failure remains a bounded incomplete fit. Neither outcome establishes generalization, native whole-answer performance, semantic factor identities, a general bandwidth advantage or reasoning benefit. Product/additive comparisons are descriptive, with one shared seed and the same data and parameter counts. Prior short-run results remain unchanged.

Full-name-plus-EOS CE retains the original mean-scene weighting and all 48,000 pair presentations per arm. Loss descriptions separate first-token, continuation and EOS losses in every complete presentation cycle and the partial tail. Those curves are not convergence certificates. Any later native or fresh-data validation requires a separate prospective release.

Allocated GPU cost: 159 seconds. Each arm used 6,021 core/conditioning/norm/head calls and 213,654 head rows; the combined total is 12,042 calls per module and 427,308 head rows. VLM and vision calls were zero. Independent CPU audit covered 58 captured computations and 42 head batches / 648 rows, with fixed TV <= 0.02 and exact argmax. Maximum TV was 0.003900413.

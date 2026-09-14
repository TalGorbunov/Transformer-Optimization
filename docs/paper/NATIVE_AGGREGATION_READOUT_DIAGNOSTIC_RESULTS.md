# Oracle readout and uniform-payload diagnostic results

The registered ideal-code readout screen passed: **108/108 training first tokens and 18/18 complete families**, exceeding the fixed 103/108 and 16/18 thresholds. The [independent precision report](../../outputs/native_aggregation_vlm/identity_join_readout/report_precision_443304/analysis.json) passed after preserving the original failed report and diagnosing its NLL reduction discrepancy. This is a positive control for readout trainability, not a deployable aggregation result.

The [prospective proposal](NATIVE_AGGREGATION_READOUT_DIAGNOSTIC_PROPOSAL.md) was triggered by the independently verified uniform training failure. The oracle used the same 108 training contexts, seed 24, 600 updates and pair order. It supplied the correct answer's fixed RMS-one, rank-96 code **after SiLU**, directly to the trainable U matrix. Only U was fitted; cached global states and the actual FP16 norm/head stayed frozen. The objective retained each first token's original full-sequence-length weight. Evaluation also supplied the answer code. No continuation, EOS, development or test result was measured; capacity was not matched to the uniform model.

The separate geometry audit covers every uniform training first-query capture: **1,296 local occurrences**, 216 per question. With FP32 RMS states and occurrence-weighted means, it decomposes local preactivation into the shared component `Wl*mu_q + q + b` and deviation `Wl*(x-mu_q)`. The shared component includes the captured content distribution, learned projection/bias and global query; it is **not question-only**. Reductions below are FP64; means are cast back to FP32 before decomposition.

| Question's rooms | Mean shared L2 norm | Mean deviation L2 norm | Mean tanh derivative | Saturated coordinates |
|---|---:|---:|---:|---:|
| Bathroom–Bedroom | 190.168 | 25.411 | 0.005885 | 98.22% |
| Bathroom–Garden | 187.442 | 24.972 | 0.005469 | 98.01% |
| Garden–Bedroom | 187.279 | 23.113 | 0.005655 | 97.87% |
| Kitchen–Office | 187.944 | 24.653 | 0.005527 | 97.59% |
| Kitchen–Park | 188.635 | 24.137 | 0.006935 | 97.62% |
| Office–Park | 181.575 | 23.258 | 0.007578 | 96.93% |

Saturation means `1 - captured_payload² <= .01`. [All six question records](../../outputs/native_aggregation_vlm/identity_join_readout/report_precision_443304/uniform_geometry_questions.json) retain per-coordinate variation as well.

All **108 same-length variant contrasts** use original slot alignment. Across the 54 contrasts at N8/N16 respectively, mean local-RMS row differences were 3.424/1.678, payload row differences 0.1204/0.0577, and pooled `.5*SUM` difference norms 0.3415/0.3304. Across all **54 insertion pairs**, the mean matched-parent pooled change was 0.1117; inserted images contributed norm 38.0848, with total pooled change 38.0802. [Variant](../../outputs/native_aggregation_vlm/identity_join_readout/report_precision_443304/uniform_geometry_variant_contrasts.json) and [insertion](../../outputs/native_aggregation_vlm/identity_join_readout/report_precision_443304/uniform_geometry_length_contrasts.json) records preserve image provenance, including moved Step labels. These statistics neither identify the cause of errors nor demonstrate that centering would repair them.

CPU check **443294** passed; GPU **443295** used **20 allocated GPU-seconds**, with 600 training plus seven evaluation norm/head calls and zero VLM/vision calls. Original report **443297 remains failed**. Diagnosis **443302** found NLL-only discrepancies in all 108 rows, with no metadata, argmax or correctness mismatch. Separate report **443304** used stable FP64 NLL reduction with unchanged `atol=rtol=2e-6`; maximum GPU-FP32 versus reference difference was 4.20e-7. All 108 CPU-replayed rows passed, maximum full-vocabulary TV 4.62e-5 and exact argmax. No refit occurred. [Execution and artifact hashes](../../outputs/native_aggregation_vlm/identity_join_readout/execution.json) retain the complete history.

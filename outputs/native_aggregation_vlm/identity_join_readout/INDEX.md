# Oracle native readout and uniform geometry diagnostic

**Completed and independently audited.** Supplied-answer-code first-token fit passed: **108/108 contexts, 18/18 families**. This offline positive control supplies the correct answer in training and evaluation; it makes no generalization, deployable-method, EOS or whole-answer claim. Uniform geometry is descriptive.

See the [results note](../../../docs/paper/NATIVE_AGGREGATION_READOUT_DIAGNOSTIC_RESULTS.md), [prospective proposal](../../../docs/paper/NATIVE_AGGREGATION_READOUT_DIAGNOSTIC_PROPOSAL.md), [source release](source_release.json), and [complete execution ledger](execution.json).

| Job | Role | Status | Allocated time |
|---|---|---|---:|
| 443294 | CPU source/input check | Passed | 10 CPU-s |
| 443295 | Fixed 600-step oracle fit/evaluation | Passed | 20 GPU-s |
| 443297 | Original independent CPU report | Failed; preserved | 9 CPU-s |
| 443302 | NLL-only rescore diagnosis | Passed | 2 CPU-s |
| 443304 | Separate precision report and uniform geometry | Passed | 14 CPU-s |

The original report remains failed. The separate reporter replaced the CPU FP32 NLL reduction with stable FP64 reduction of the same saved logits, retaining the original tolerances and all other checks. No refit or additional GPU execution occurred. Total GPU usage, including every campaign allocation, is **20/150 GPU-seconds**.

- [Passed CPU check](check_443294/summary.json)
- [GPU run and final endpoint](run_443295/summary.json)
- [Original report failure](report_443297/failure.json)
- [Preserved rescore diagnosis](rescore_diagnosis_443302/summary.json)
- [Passed precision summary](report_precision_443304/summary.json) and [analysis](report_precision_443304/analysis.json)
- [Stable NLL comparisons](report_precision_443304/nll_precision.json) and [native head replay](report_precision_443304/native_replay.json)
- [Uniform geometry bindings](report_precision_443304/uniform_geometry.json): [six questions](report_precision_443304/uniform_geometry_questions.json), [108 scenes](report_precision_443304/uniform_geometry_scenes.json), [108 variant contrasts](report_precision_443304/uniform_geometry_variant_contrasts.json), [54 insertion contrasts](report_precision_443304/uniform_geometry_length_contrasts.json)
- [All campaign GPU allocations](report_precision_443304/all_user_sacct.psv)

Final summary SHA256: `cb280b2bf6d769885ee20f395a6adaeed0aaec09ef00a5899258d6049b5490c7`.
Analysis SHA256: `936c5a9fee35bda4f67df0c6e20e5ef64055dbaea6ed94cd33edfcae1feabe82`.

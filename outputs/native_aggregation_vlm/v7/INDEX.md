# V7: full local computation and native set fusion

No V7 efficacy result yet. Balanced training and fresh evaluation are complete.
The main comparison is parallel-local SUM/SiLU versus matched joint-state adaptation,
seeds8/9, identical1,041,600parameter cores and40epochs. Primary exact requires EOS.

- [Protocol](../../../PREREG_AGG.md) · [Execution](execution.json)
- [Balanced train/dev](balanced_data/stage_441770/summary.json)
- [Fresh test staging](data_staging/INDEX.md)
- [Parallel features](features/INDEX.md) · [Metadata release](features/metadata_release/release_441834/release.json)
- [Original runtime-failed profile](features/profile_441837/summary.json) · [Explicit six-minute release](features/runtime_extension/release_441840/release.json)
- [Joint feature profile](joint_features/profile_441831/summary.json)
- [Native input/source check](runtime/check_441842/plan.json)

All costs count allocated GPU seconds; total reserved4.55GPUh, at most4concurrent project GPUs.
All previous negative results and failed checks remain retained. This is not a new
attention-family claim and has not demonstrated reasoning composition.

Four main runs441885 launched under the [explicit45-minute release](main_runtime_extension_441884/release.json). [Independent data/cache release](data_check_441882/summary.json), [training plan](train_check_441869/plan.json). No efficacy result yet.

The independent report441893 and selected-checkpoint CPU441892 are queued after all four mains. Selected GPU441900 depends on CPU441892 passing. Finalizer selftest441894 passed all six fixed acceptance/plot tests; its source remains frozen. Final acceptance requires both independent verification paths.

All four mains completed (4,325 GPU-seconds). Selected checkpoint CPU441892 passed; GPU441900 passed all computational and 40 native-head replay checks (83 GPU-seconds), while retaining six descriptive cached/full numerical failures. Report441893 stopped at its erroneous FP16 archive-dtype expectation. Installed generation returns lossless FP32 copies; inspection441904 confirmed, explicit storage repair unit441908 passed, complete independent audit441910 running. Total V7 allocated GPU time so far: 6,545 seconds (1.8181 hours). No verified efficacy result yet.

## Verified outcome

[Independent report](report_441910/REPORT.md), [final figure and acceptance](finalization/final_441912/REPORT.md). Both-seed primary passed; practical accuracy failed. Parallel N16=100% both, N32=98.1%/83.3%, N64=14.8%/0%. Familiar OOD56.5%/41.7% versus joint22.2%/26.9%; unseen0/128 all. Full computational/native-head verification passed; six descriptive cache failures remain. Total6,545GPU-seconds. Reasoning composition remains untested.

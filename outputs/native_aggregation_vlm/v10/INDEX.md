# V10 verified: relative improvement, practical failure

Allanswer values0–16 appeared in adaptation. Selectedmodels were64/64 on N16development, but length extrapolation remains poor. Bothregistered relative criteria pass; neither practical criterion passes.

| Condition | Seed | N32 | N64 | N64 K9–16 |
|---|---:|---:|---:|---:|
| CE |14|21/136|8/136|0/64|
| CE |15|26/136|0/136|0/64|
| Consistency |14|43/136|20/136|0/64|
| Consistency |15|52/136|19/136|0/64|

[Independent report442173](report_442173/REPORT.md) · [Selected audit442185](checkpoint_audit/run_442185/REPORT.md) · [Final figure/acceptance442190](finalization/final_442190/REPORT.md).

All40native-head replayspassed; fourknownlocalcache/fullnumericaldifferencesremain descriptive, alltop1same. Total4867allocatedGPU-seconds(1.351944h), includingcache/profiles/mains/selectedaudit; allCPUfailurespreserved. Noacceptedvisionmilestone orreasoningcomposition.

## Prospective protocol and execution history

# V10: supported answers, fourfold length extrapolation

Matched CE versus residual-consistency protocol registered, seeds14/15. Feature CPU stage442121 passed; GPUprofile442123 launched after validation. No main training released yet. Preserve max training N16 and test N64, while representing every tested K0–16 answer during adaptation. Reserved total4.5GPUh, maximum4concurrent projectGPUs. Primary requires at least7/136 additional N64 answers and at most6/136 fewer N32 answers perseed. Practical also requires treatmentN32≥123/136,N64≥109/136,andN64K9–16≥52/64.

Training: 1,782 unique scenes, 1,836 weighted scenes, 918 pair slots. N8/K8 and N16/K16 are the only permitted historical saturated training-context exceptions. Development: 64 fresh N16 scenes, K0–15. Tests: 136 N32→N64 families, 272 contexts, K0–16. All contexts follow the unchanged rendered-world law and the registered exclusion rules.

[Prospective protocol](../../../PREREG_AGG.md) · [Current research question](../../../docs/paper/NATIVE_AGGREGATION_WORKING_STORY.md)

[General sequence software442075](sequence_selftest_442075/summary.json): all6CPUtests passed, including value/gradient compatibility with the prior fixed-two-position losses and equal per-sequence weighting for ragged targets. No GPU run.

[Data stage442095](data_staging/stage_442095/REPORT.md) completed119CPU-seconds. Corrected dry442087 passed; failed dry442083 is preserved.

[Feature stage442121](features/stage_442121/summary.json) passed all inventory, sequence and processor checks:33,658training-only states, including918nonempty global prefixes. GPUprofile442123 uses the registered3minute cap.

Cache profile442123 passed44GPU-seconds, fourshards442125 completed905GPU-seconds combined; [merge442141](features/merge_442141/summary.json) passed13CPU-seconds,33658states. CacheSHA42dd0c3c12bc13326493dce9389656b1c412e9ecfa6dac37ccb45458aa4ede11. Independentmanifest442132failed only record-order equality after JSON roundtrip; preserved. [Corrected manifest442139](manifest_check_442139/summary.json) passed53CPU-seconds with exactrecordequality after restoringregistered split order; selftest442138passed.

[Training CPU442145](train_check_442145/summary.json) and [independent cache audit442146](data_check_442146/summary.json) passed. Trainingprofilearray442148 passedbothconditions,82GPU-seconds total. [Measured release442158](main_release_442158/release.json) projects1900.49/1900.68seconds permain; budget reserves12131/16200GPU-seconds. Mainarray442160 launched four CE/consistency seeds14/15 jobs, each45minute cap. This historical launch entry predates the verified result above.

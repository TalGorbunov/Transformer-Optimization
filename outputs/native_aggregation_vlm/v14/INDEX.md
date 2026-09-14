# V14: exact empirical centering, then fixed-core distillation

Registered before new jobs in [PREREG_AGG](../../../PREREG_AGG.md). Source and fresh-data preparation in progress; no V14 efficacy or accepted objective. V13 verified failure and geometry motivate the separated training phases.

- CPU dry442494 passed1s, planSHA6e8987689363fb6a9a8d3dbc001c2ddd5820506da3330c67dbcd6f14e888fcb6. [Dry plan](data_staging/drycheck_442494/plan.json).
- CPU rendering442496 passed43s;272contexts/136families,3308unique renders. ManifestSHAed900cc26eb99575325583de8ef7d18beeac506c4baedd5a2c3f67ae70adf909. All28prior exclusions and source/QA/image/extension audits passed.

- [Design](../../../docs/paper/NATIVE_AGGREGATION_VISION_V14_DESIGN.md).
- FinalizerCPU442511 stale-seed fixture failure preserved; correctedselftest442512 passed19checks. [Source check](finalization/selftest_442512/summary.json).

- CPU release-arithmetic/accounting selftest442514 passed7checks. [Checks](main_release_442514/self_test.json).
- CPU trainer442520 passed; [plan](train_check_442520/plan.json), SHAb0324838795ce132a785adb3ad4f489e17bb4c7e8451239a7b27f469d434afbf. Full empirical reference gradients, frozen-core student isolation, native target/pair/cache bindings and carried phase orders passed. GPUprofiles pending independent data/cache/source release.

- BeforeGPUexecution, registered overlap of the two training-cache-only profiles with final independent fresh-test/report checks. No profile accuracy or freshdata use; allchecks still required for mains.

- Initial GPUprofiles442526(centered45s)/442525(offset44s) FAILED in conversion-only unique-scene packing; paired-training helper rejected differingtargets. Both32+32fit phases completed, no nativeheadreplay or accuracy. Original source/plan/partials retained. Narrow unpaired diagnostic packer and CPU regression check pending; total89GPU-seconds.

- Corrected conversionpack CPU442540 passed27s (12V14+11null+6ragged+pair tests); [plan](train_check_442540/plan.json), SHAf5a525409f2211c91c71af8b04399e0e6efbdac4b2777a3d8dde149bb2a79dd6. Prior442530 test-only assertion failure retained. Trainer05b1e802..., tests57a2529b... frozen.
- Reporterselftest442531 passed2s; dataaudit442536 failed17s on mistaken282context assertion, correctedto272 withdatasetunchanged; newsourcechecks pending.

- Corrected GPUprofiles442549centered/442548offset PASSED55s each, allnativehead and fullconversion checks. [Centered](profile_centered_s18_442549/summary.json) · [Offset](profile_offset_s18_442548/summary.json). TotalV14GPU199s including89failed. Await independentmainrelease; no efficacyyet.

- Corrected reporter selftest442557/dataaudit442558 and selectedselftest442559 passed. [Data/cache](data_check_442558/summary.json) · [Selected source](checkpoint_audit/selftest_442559/summary.json).
- [Main release442561](main_release_442561/release.json) PASSED33CPU-s;~1806s/model,199GPU-seconds prior includingallfailures;11299s reserved<=16200. Fourmains released; noefficacyyet.

- Mainarray442566 launched fourmatchedruns: centered18/19 jobs442567/442569, offset18/19 jobs442568/442566. [Execution](execution.json). Independentreport442582 andcheckpointCPU442581 are gatedafterallmains; checkpointGPU442583 afterCPUgate; finalizer442584 afterreportandGPUaudit. No efficacyyet.

- Allfourmains complete729/729/727/724GPU-s; main2909+prior199=3108beforeselected. Independentreport/checkpoint verification running. [Prospective next-question memo](../../../docs/paper/NATIVE_AGGREGATION_AFTER_V14.md), written beforeaccuracyinspection.


V14 final verified: report442582,selectedretry442625 andfinalizer442626 passed. Both-seed relative primary TRUE; practical FALSE. Centered N32=109/136,97/136; N64=41/136,43/136. Total3197GPU-seconds including89failedprofiles+3failedselected-invocation,110successfulprofiles,2909mains,86successfulselected. Four descriptive cached/full local numerical failures retained;46nativehead replays pass. [Final artifact](finalization/final_442626/REPORT.md), [results note](../../../docs/paper/NATIVE_AGGREGATION_VISION_V14_RESULTS.md). PNG visually inspected.

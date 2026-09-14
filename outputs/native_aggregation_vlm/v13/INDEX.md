# V13 learned conditional null mean

**Completed: both-seed primary and practical criteria FAILED.** Centered N64=11/136,15/136; offset=20/136,0/136. All four models N64K9–16=0/64. Verified GPU total4,579seconds; no research objective or reasoning composition established. The research objective remains unresolved.

[Proposal](../../../docs/paper/NATIVE_AGGREGATION_LEARNED_NULL_PROPOSAL.md) · [Inventory registration](null_inventory_registration.json) · [Cache registration](null_cache_registration.json).

- [Inventory442353](null_features/check_442353/summary.json): 19,664 missing local features; final union53,322.
- [Feature profile442394](null_cache/profile_442394/summary.json): all8 native-head checks passed,40GPU-seconds.
- Four harvests: [shard0](null_cache/shard0_442405/summary.json), [shard1](null_cache/shard1_442406/summary.json), [shard2](null_cache/shard2_442407/summary.json), [shard3](null_cache/shard3_442404/summary.json);150/151/149/150GPU-seconds. Feature total640.
- Original CPUmerge442411 failed exact scalar equality; immutable tensors and original outputs retained. [Diagnostic442416](null_merge_diagnostic/run_442416/summary.json) verifies all392 comparisons and8 unchanged native-head gates. Seventeen rows differ in descriptive RMS by at most7.45e-9; exact tensor/metadata/max fields agree. No inference or scoring threshold changed.
- [Native softwareCPU442412](null_software/check_442412/summary.json) passed11mean tests and6controllerchecks.
- [Native GPU442413](null_software/profile_442413/summary.json):40VLM/20vision/40head calls;18KV,26fusion,6zeroidentitychecks pass. Two inherited localN64 cached/full numerical differences remain descriptive.68GPU-seconds.
- Fresh272testcontexts at `/mnt/data/gabriele/gnn_transformer/v11_fresh`, first used by V13; [independent manifest442386](../v11/manifest_check_442386/summary.json) passed.

Completed pre-main V13 GPU total801seconds. [Repaired merge442432](null_cache/merge_442432/summary.json), [trainerCPU442436](train_check_442436/summary.json), [independentdata442437](data_check_442437/summary.json), [centeredprofile442442](profile_centered_s16_442442/summary.json), [offsetprofile442441](profile_offset_s16_442441/summary.json), and [selected-source442443](checkpoint_audit/selftest_442443/summary.json) passed.

[Measured release442444](main_release_442444/release.json): each main projected2228–2229seconds<=2700; prioractual801+four2700+selected300=11901<=16200. Four matched main fits are released. Maximum4concurrentprojectGPUs; wholecampaigncap16,200GPU-seconds. V11memorytraining remains held. RESULTS.md untouched.

[Execution ledger](execution.json): array442452, centered16/offset16/centered17/offset17 jobs442453/442454/442455/442452. Main outcomes remain pending.

[Prospective interpretation](../../../docs/paper/NATIVE_AGGREGATION_VISION_V13_INTERPRETATION.md): centering removes a conditional mean; predictor bias and image variance remain distinct extrapolation risks. Current-query reuse does not establish persistent reasoning memory.

[Prospective estimand analysis](../../../docs/paper/NATIVE_AGGREGATION_VISION_V13_ESTIMAND.md) derives how task gradients can pull the learned predictor away from the empirical null mean. No additional experiment is released by that memo. Independent rescoring442466 and selected-checkpoint binding442467 are queued after all four main tasks.

[Independent report442466](report_442466/REPORT.md) · [Selected audit442472](checkpoint_audit/audit_442472/summary.json) · [Final report/figure442481](finalization/final_442481/REPORT.md). The chronological preparation records below/above remain preserved. Registered CPU geometry follows; no additional fit released.

[Calibration geometry442483](calibration_geometry/geometry_442483/summary.json) passed26CPU-seconds: all3888 null groups/17064 ordinary prefix positions across4models. Mean projected bank-fit errors94.98/101.63 in centered models versus6.81/8.87 in controls; these are training-bank geometry, not measured accuracy effects. [Source tests442482](calibration_geometry/selftest_442482/summary.json).

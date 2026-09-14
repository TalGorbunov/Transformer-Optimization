# V9: path bound versus residual consistency

Verified negative result: path N64 18/108 in both seeds versus residual 47/108 and50/108. Both primary and practical criteria failed; all N16/N32 108/108, all unseen0/128. Total4940GPU-seconds. Same native parallel SUM/SiLU inference, paired training, two conditions with coefficient one, seeds12/13. Coefficients do not match effective regularization strength. Reserved cap3.25GPUh, maximum4concurrent projectGPUs.

- [Protocol and limits](../../../docs/paper/NATIVE_AGGREGATION_PATH_BOUND.md)
- [Fresh452contexts,23exclusions](data_staging/stage_442020/REPORT.md): CPU rendering completed62s.
- [Loss software](loss_selftest_442018/summary.json):12path+11consistency tests passed.
- [Training freeze](train_check_442032/summary.json): CPU19s; same1944weighted scenes/972pairs/epoch,40epochs.
- [Independent report tests](self_test_442033/summary.json) and [data/source audit](data_check_442034/summary.json): passed.
- [Residual profile](profile_residual_s12_442036/summary.json) and [path profile](profile_path_s12_442035/summary.json):67GPU-seconds total, no dev/test evaluation, allparametergradients/nativeheadreplay/restore passed.
- [Measured release](main_release_442039/release.json): projections2410.69/2413.75s ≤2700s.
- [Finalizer source check](finalization/selftest_442037/summary.json)
- [Execution ledger](execution.json)

Native prior cache failures stay descriptive and retained. Primary requires path-minus-residual≥11/216correct and N16loss≤5/108 in eachseed. Practical additionally pathOOD≥152/216,nonzeroOOD≥116/192,N64≥76/108. All128unseenK9–16 separate; no reasoning-composition claim.

Selected-checkpoint prospective software test442041 passed. All4main runs use frozen plan442032 and release442039.

[Independent report442051](report_442051/REPORT.md) · [Selected-model audit442081](checkpoint_audit/run_442081/REPORT.md) · [Final artifact442084](finalization/final_442084/REPORT.md). All40 native-head replays passed; five descriptive cache failures retained. The path objective is dropped as the proposed improvement.

[Fixed-state diagnostic442099](response_surface/run_442099/REPORT.md) completed24CPU-seconds after selftest442093. Allmodels54/54syntheticN16; N64path12/13=32/54,24/54 versusresidual28/54,29/54. No consistent synthetic benefit; this is not held-out task accuracy. No VLM calls or fitting.

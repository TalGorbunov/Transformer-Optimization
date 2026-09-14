# Frozen moment readout: linear-separation diagnostic

Neither frozen moment readout yielded a certificate of complete linear recovery. Neither yielded an exact certificate of nonseparability. Both results are **indeterminate**, while the exact dual witnesses tightly bound the maximum possible worst-case margin in the specified raw coordinates.

| Fixed readout | Exact LP margin interval | Upper bound / maximum feature magnitude | Exact candidate correct /216 | Rounded candidate exact-reference correct /216 | Rows certified against FP32 scoring error |
|---|---:|---:|---:|---:|---:|
| Within |[0,1.0718643982e-8]|2.1786065804e-10|49|46|0|
| Cross |[0,2.3275031545e-8]|3.5514344650e-10|97|78|51|

The interval upper endpoints shown above are approximate decimal displays of exact rational bounds stored with the witnesses. The classifier is homogeneous: nine scores from the raw96-dimensional pre-U activation, no intercept or feature normalization, with total coefficient L1 norm at most1. The task is strict separation of **all216 labels**, not the earlier206/103-each/16-family model screen. These candidate accuracies are descriptive and are not maxima over partial-accuracy classifiers.

Both solver invocations reported successful termination, but their returned candidates had negative exact minimum margins (about-8.67e-14 within and-7.94e-10 cross). The independently reconstructed dual residuals were nonzero. Solver status therefore establishes neither exact optimality nor nonseparability. The upper bounds do show that any perfectly separating classifier in this norm ball would have a very small worst-case margin. No tolerance, scaling, alternative solver or feature transformation was tried after observing this result.

The independent certificate distinguishes real arithmetic from finite-precision scoring. It normalizes each candidate into the exact L1 ball, checks every class margin with integer/rational arithmetic, and verifies a dual upper bound. Its separate binary32 candidate is rounded once with exact ties-to-even rules. The scoring certificate accounts for dot-product roundoff, subnormal operands, underflow and possible overflow. The reported rounded-candidate predictions are exact-reference calculations on those rounded coefficients, not observations from a GPU backend. Neither candidate certifies all216 answers under that scoring bound.

All eight mathematical fixture groups passed. The actual SciPy/HiGHS nine-basis-vector fixture certified all nine labels and enclosed the known optimum1/9, with exact interval gap about6.17e-18. The two actual solves used20.06seconds/65,498iterations within and1.09seconds/8,124iterations cross, both within the fixed45second limits. The job used one CPU Slurm allocation,443866, completing in35seconds with four cores and16GiB; no GPU, core, backbone, native head, backward or neural optimizer calls occurred. SciPy1.15.3 and the exact solver binaries are recorded.

[Summary and bound artifacts](../../outputs/native_aggregation_vlm/identity_join_moment_readout_lp/lp_443866/summary.json), SHA`b8d1fa2d349fa5adc1ebc34ad0fa26f7c4473e7810611b103cb8b38b9789d08d`. Independent source/metadata review verified all four current sources and229 inherited sources, ordered216-row ownership for both arms, serialized certificates and solver limits. Matrix computations ran only inside Slurm.

The earlier native-head report443841 remains strictly failed; this diagnostic used its fully audited saved pre-U tensors and preserved that failure. The positive-recovery condition for a possible native-U interface diagnostic was not met. The216-example architecture search is closed. Next is a separately specified adapted ordinary-joint baseline, because no trained identity-join comparator has yet established what standard model adaptation can learn. This does not change the negative results or release fresh evaluation. Useful single-pass aggregation, extrapolation and reasoning gains remain unestablished.

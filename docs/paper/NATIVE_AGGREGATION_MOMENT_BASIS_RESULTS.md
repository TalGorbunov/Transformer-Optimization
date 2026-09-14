# Visual moment-basis result

Both direct-vision moment bases fail to learn the216-context identity join under the fixed recipe. Neither approaches the206/216 pooled,103/108 per-orientation and16/18 complete-family thresholds. This moment-basis branch is closed.

| Readout input | Saved GPU correct /216 | Original /108 | Flipped /108 | Complete families /18 | Optimistic CPU/GPU upper bound |
|---|---:|---:|---:|---:|---:|
| Within-image `[A,B,C]` |77|42|35|0|77|
| Cross-image `[A,B,A*B-C]` |73|40|33|0|74|

The two arms used the same native visual features, fixed conditioning,216 scenes, paired48,000-entry presentation order, fresh seed24 parameter bytes and6,000 full-name-plus-EOS CE updates. Both computed first moments A/B, within-image C and cross-image P; only the selected third readout block differed. Each model has six trainable tensors and1,404,192 parameters. The final first-query mean NLL was1.11163 within and1.10852 cross. These are cached training results, not natural-answer generation or extrapolation.

The strict numerical report remains **FAILED**. All44 captured functional computations, moment identities, native input casts and applicable NLL checks passed; all28 CPU head batches,432 rows, were collected. One cross-arm CPU head argmax differs from GPU, with total variation0.0036886 below the0.02 limit. The affected original-orientation Emma example was already wrong on GPU. The saved metadata therefore bounds CPU cross accuracy at73–74 and the optimistic backend union at74. Within has no argmax disagreement. Maximum total variation is0.00370297 within and0.00378653 cross. No tolerance or acceptance rule was changed, and no passed summary was manufactured.

[Complete analysis](../../outputs/native_aggregation_vlm/identity_join_moment_basis_v2/report_443841/analysis.json), SHA`5b206f806cc3787b4b9cf3601839d1ccb872c53932d1fd4c02b14aa779e7c256`; [preserved failure](../../outputs/native_aggregation_vlm/identity_join_moment_basis_v2/report_443841/failure.json), SHA`63a8a8bdaaefc69ece5c9bacc417c1820cc356f8bcf59dcfce2ea199f535c094`.

CPU preparation443829 initially failed after5seconds because a prior report's protocol name differed from its directory. V2 changed that explicit metadata mapping, preserved the original sources/failure, and passed all numerical fixtures in CPU443836 (14seconds). Both fits443837/443838 completed in72GPU-seconds each,144total; independent report443841 used55CPU-seconds. Each fit used6,014 core/conditioning/norm/head calls over213,546 head rows; there were no backbone or vision calls. See [execution](../../outputs/native_aggregation_vlm/identity_join_moment_basis_v2/execution.json).

This comparison supplies no cross-moment advantage and no evidence of improved aggregation bandwidth. The known moment identity does not add information to `[A,B,C]` in real arithmetic, and finite readout approximation, conditioning and optimization remain distinct. A failed finite-budget fit does not establish architectural incapacity.

The next selected diagnostic freezes the actual96-dimensional pre-U readouts and asks whether all216 answer labels are strictly separable by a homogeneous linear classifier. Its convex maximum-margin formulation is a decoder-accessibility test, not a replacement native model or a206-context screening test. It may use the completely audited readout tensors while explicitly preserving the unrelated strict native-head replay failure. No validation data or additional GPU fit is released. Native whole answers, unused fresh confirmation and reasoning/extrapolation remain held; the user's broader objective is still open.

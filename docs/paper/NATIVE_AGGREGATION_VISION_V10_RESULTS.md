# V10: supported answers still fail at greater lengths

Native residual consistency improves the matched control, but does not solve length extrapolation. All tested answer values0–16 appeared during adaptation. Every selected model scored64/64 on the fresh N16 development set, which covers0–15. AtN32/N64, every model still failed allK9–16 examples.

| Objective | Seed | N32 | N64 | N64 K9–16 |
|---|---:|---:|---:|---:|
| Native CE |14|21/136|8/136|0/64|
| Native CE |15|26/136|0/136|0/64|
| CE + residual consistency |14|43/136|20/136|0/64|
| CE + residual consistency |15|52/136|19/136|0/64|

The N64 gains are8.82percentage points, family-bootstrap95%CI[5.88,11.03], and13.97points,[10.29,17.65]. Both registered relative criteria pass; both practical criteria fail. N32 accuracy is far below123/136, N64 below109/136, and larger-count N64 below52/64. The intervals condition on these two fitted seeds.

Both consistency models produced136/136 parseable, completed numerals atN64. Their poor accuracy therefore includes substantial numerical errors, not merely output formatting or truncation. Answer coverage by itself has not resolved the tested failure. Comparing these values directly with V8 does not isolate the effect of answer support: V10 also changes scene diversity, length/count correlations, pair kinds and weighting.

The experiment held the native parallel SUM/SiLU computation,1,041,600 trainable parameters, initialization within seed, pair schedule, optimizer and inference fixed. Only the residual-consistency coefficient changed from0 to1. Forty epochs comprised73,440 scene presentations and177,120 valid causal targets over4,590 updates. Full ordinary numeral token sequences plusEOS received equal per-scene weighting. All five64-example development sweeps were retained; every model selected step4,590 by the registered rule.

The data comprise1,782 unique training scenes and918 weighted pairs per epoch. K0–8 pairs spanN8/N16, K9–15 use distinctN16 scenes, and54K16 pairs repeat a saturated SID with exactly zero regularizer. Every question/count receives two scene slots. TrainingN8/K8 andN16/K16 are the only historical saturation exceptions. Fresh tests contain136N32→N64 families, eight perK. The independent audit checked all25 prior exclusions, image semantics/hashes, native targets, shared causal cache identities, loss reductions and raw full-vocabulary generation.

Report442173 and selected-checkpoint audit442185 completed; all40 captured-state native-head replays passed. Four of656 cache/full numerical comparisons retain their known descriptive failure, each the localN64 comparison withTV≈.02029646 and unchanged top1. Finalizer442190 independently bound all selected checkpoints, decisions and scheduler accounting; its32-cell figure was visually checked. No vision milestone or reasoning-composition claim was accepted.

Total cost was4,867 allocated GPU-seconds(1.351944hours):949cache,82training profiles,3,752mains and84selected audit. The data dry-run Path adapter failure and independent audit JSON-order failure remain preserved; corrected checks passed without changing data or acceptance criteria.

[Independent report](../../outputs/native_aggregation_vlm/v10/report_442173/REPORT.md) · [Verified figure and acceptance](../../outputs/native_aggregation_vlm/v10/finalization/final_442190/REPORT.md) · [Execution ledger](../../outputs/native_aggregation_vlm/v10/execution.json).

# V8: native residual consistency improves length extrapolation

The matched experiment establishes a substantial training-objective effect with unchanged inference. Requiring equal-answer N8/N16 scenes to produce similar native residuals improves familiar-count N32/N64 exact accuracy by46.76 and43.52percentage points in two seeds. It reaches100% atN32 in both. Reliable64-frame aggregation and unseen-count decoding remain unresolved.

| Training objective | Seed | N16 | N32 | N64 | Familiar OOD | Nonzero OOD | Unseen K9–16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CE |10|108/108|79/108|0/108|79/216|79/192|0/128|
| CE |11|108/108|55/108|12/108|67/216|55/192|0/128|
| CE + residual consistency |10|108/108|108/108|72/108|180/216|156/192|0/128|
| CE + residual consistency |11|108/108|108/108|53/108|161/216|140/192|0/128|

Both conditions used the same1,041,600-parameter parallel core, frozen Qwen backbone and features, paired scene order, initialization within seed, optimizer and4,860updates. The loss compares FP32 residuals separately at the count and EOS positions, divided by the squared norm of the identical frozen global state plus1e-6. Its coefficient is1 for treatment and0 for control. No local oracle, output mask, external tally or additional inference operation is introduced.

All models were selected from five72-example N16 development sweeps by exact answer, raw first-token NLL, then earliest step. CE10 selected step2916; the other three selected4860. Each then evaluated the same452fresh contexts, excluded against21prior manifests. Exact requires a whole ASCII integer equal to gold and native EOS within four tokens. Every malformed and truncated output remains in the denominator.

The paired family-bootstrap95% intervals for OOD gain are[43.98,50.00] and[39.35,48.15]percentage points. They describe uncertainty across test families conditional on these two fitted seeds. Both registered primary criteria pass. Both practical criteria fail specifically because N64 remains below76/108; the OOD and nonzero-OOD thresholds otherwise pass. The weaker historical V7 criterion is not substituted after the fact.

At N64, consistency10 produces valid completed numerals on108/108 examples, and consistency11 on104/108. Their remaining errors therefore include substantial count errors, not just formatting. Correct counts byK0 throughK8 are[12,11,12,12,12,7,2,3,1] and[9,6,10,10,10,2,2,2]. All unseen K9–16 answers remain wrong.

The fixed-state diagnostic holds local visual states to exactly the earlier training prototypes. All four models achieve54/54 synthetic first-token answers atN16. AtN64, CE scores6/54 and11/54; consistency scores32/54 and30/54. This supports improved behavior of the learned fusion on familiar states, while retaining a substantial failure. It does not establish that later Step features are irrelevant to actual MMReD errors. These synthetic repeated-state sets violate the generator's Step law and are not held-out task accuracy.

Independent report441974 verified all1,808test answers and1,440development answers, data/cache ancestry, training targets/loss arithmetic and checkpoint selection. Selected-checkpoint audit441997 verified all40native-head replays and actual execution; five of656cached/full numerical comparisons retain their descriptive failure status, all with unchanged top1. Finalizer441998 binds every selected checkpoint and preserves the failed practical decision. Total V8 cost is4,666allocated GPU-seconds, including profiles and the selected-model audit.

This is a native application of consistency training, with a measured benefit and a clear boundary. It does not establish a new attention family, intrinsic information-capacity increase, a general counting algorithm or reasoning composition. The separate Cosmos streaming smoke passed execution checks but contains no trained-method efficacy result.

[Independent report](../../outputs/native_aggregation_vlm/v8/report_441974/REPORT.md) · [Verified figure and acceptance](../../outputs/native_aggregation_vlm/v8/finalization/final_441998/REPORT.md) · [Fixed-state diagnostic](../../outputs/native_aggregation_vlm/v8/response_surface/run_441992/REPORT.md) · [Transfer criteria](NATIVE_AGGREGATION_TRANSFER_CRITERIA.md).

A subsequent read-only audit of all512 unseen-count traces (CPU442059,12s) found0correct first tokens in every model. Both consistency models output exactly8 on all128unseen examples, at both N32 andN64. This is output saturation at the largest supervised answer, not an observed case of correct first-digit prediction followed by premature EOS. It still does not reveal whether hidden states retain information about larger counts. [Verified natural traces](../../outputs/native_aggregation_vlm/v8/unseen_trace_diagnostic/run_442059/REPORT.md).

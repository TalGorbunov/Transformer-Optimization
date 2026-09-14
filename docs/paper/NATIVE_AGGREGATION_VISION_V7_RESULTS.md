# V7: strong transfer to 32 frames, failure at 64

Preserving full local computation improves native MMReD Vision counting over matched joint-state adaptation. The gain does not yet meet the practical target: the parallel model is perfect at16frames and strong at32, but fails at64. Training used8/16frames and counts0–8.

| Arm / seed | N16 | N32 | N64 | Familiar OOD | Unseen K9–16 |
|---|---:|---:|---:|---:|---:|
| Joint8 |42/108|23/108|25/108|48/216|0/128|
| Parallel8 |108/108|106/108|16/108|122/216|0/128|
| Joint9 |37/108|29/108|29/108|58/216|0/128|
| Parallel9 |108/108|90/108|0/108|90/216|0/128|

The registered primary criterion passed in both seeds: familiar-OOD differences +34.26pp (paired family95%CI28.24–39.81) and+14.81pp (8.33–21.30), with no N16 regression. The practical criterion failed in both. Confidence intervals condition on these two fitted seeds. Pooled+24.54pp is descriptive.

The contrast tests the complete independent local computation plus learned native set fusion, against the same1,041,600-parameter core on the joint hidden state. It does not isolate SUM, equalize inference compute or establish increased information capacity. Balanced training differs from earlier versions; historical runs are not the primary control.

At64frames,65/108 and108/108 parallel outputs are unparseable. This is often a failure to retain integer output, rather than an off-by-one answer. Unseen counts show a separate support limitation: all parseable parallel answers are8. These are exploratory error descriptions; they do not identify the causal source of failure.

All1808 test predictions, raw greedy choices, EOS completion, five-way dev selection, target orders and data/cache provenance were independently verified. Primary exact requires the correct integer and EOS within four tokens. All malformed and truncated outputs remain in the denominator. The selected-checkpoint audit passed computational and40 native-head replay checks. Six cache/full numerical differences exceeded the earlier descriptive threshold, with unchanged top1; the prospective engineering release and original failures remain recorded.

The first reporter failed because it expected FP16 archived logits; installed generation stores lossless FP32 copies. An explicit repair dispatcher preserved frozen original sources, verified exact FP16 roundtrip for every vector, and repeated the independent audit. No model, predictions, selection or scoring changed.

Total V7 cost:6,545 allocated GPU-seconds (1.818hours), below the4.55hour cap. Reasoning composition and reliable64-frame aggregation remain open.

[Independent report](../../outputs/native_aggregation_vlm/v7/report_441910/REPORT.md) · [Full analysis](../../outputs/native_aggregation_vlm/v7/report_441910/analysis.json) · [Figure and bound acceptance](../../outputs/native_aggregation_vlm/v7/finalization/final_441912/REPORT.md) · [Execution ledger](../../outputs/native_aggregation_vlm/v7/execution.json)

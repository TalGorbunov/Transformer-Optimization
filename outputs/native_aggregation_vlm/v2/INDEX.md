# Native aggregation V2: clean vision attribution

Complete. Primary attribution screen FAILED: familiar N32/64 exact global26.9%,
hidden29.2%, middle+upperLoRA21.8%, upperLoRA16.2%. One seed per arm.
Global-minus-hidden -2.3pp, paired-anchor95%CI[-10.2,+5.1].

- [Verified main report](REPORT.md), [analysis](analysis.json), [figure](comparison.png).
- [Scientific interpretation](../../../docs/paper/NATIVE_AGGREGATION_VISION_V2_RESULTS.md).
- [Execution ledger](execution.json): total8299GPU-seconds=2.30528hours, cap4hours.
- Main array440749: global1970s, hidden1708s, middle1691s, upper1622s.
- GPU profiles440741:44+44+44+60=192seconds; disjoint software examples.
- CPU checks440738:33tests passed; independent data audit440739:720samples passed.
- Data: /mnt/data/gabriele/gnn_transformer/v2_clean (stage440730).
- Checkpoints: /mnt/ckpts/gabriele/gnn_transformer/native_aggregation_vlm_v2.
- [Channel diagnostic](channel_probe/channels_20260910_165456_440805/summary.json):
  GPU440805,263s; identityexact0, no donor-directed mean read effect.
- [Prefix diagnostic](prefix/20260910_165238_440808_1379431/REPORT.md):
  GPU440808,158s; baseline=correctprefix5/36,neutral4/36; no positive repair.
- [Frozen recoverability](recoverability/fit_440792/REPORT.md):
  GPUharvest440791,695s; CPUfit440792,2s. Poor long/unseen-count linear recovery.
  Features: /mnt/data/gabriele/gnn_transformer/v2_recoverability/harvest_440791.

All native test answers retained; separate452testrows per arm (324familiar,
128unseencount), with paired anchors. Native global correctly emits17/128
unseen-count answers; no universal unseen-number barrier. No new aggregation
mechanism or reasoning-composition claim. V1 preserved; V3 uses these tests
exploratorily and requires fresh confirmation if positive.

- [Post-hoc marginal law audit](marginal_audit/audit_440844/REPORT.md), CPU440844 (2s).
  Oracle marginals predict some counts; separate from model results and no proof of model use.

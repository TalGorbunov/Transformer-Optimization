# V3 — native-width nonlinear value placement

Main complete. Registered two-seed primary criterion FAILED.

| Arm | Seed0 familiar N32/64 | Seed1 familiar N32/64 |
|---|---:|---:|
| PRE | 78/216 (36.1%) | 44/216 (20.4%) |
| POST | 63/216 (29.2%) | 44/216 (20.4%) |

Seed0 gain+6.9ppCI[-0.5,+14.4], seed1 0ppCI[-7.4,+7.4].
Pooled fixed-seed gain+3.5ppCI[-2.3,+9.0]; cannot override seed1 failure.
The secondary PREseed0 versus historical V2hidden screen passed; no reliable
architecture gain or reasoning-composition claim. Test reuse is exploratory.

[Verified report](REPORT.md) | [Analysis](analysis.json) | [Figure](comparison.png)
[Execution](execution.json) | [Protocol](../../../PREREG_AGG.md)

- CPU440822:30tests passed (13s). GPUprofiles440826:81seconds total.
- Main440830:1682+1679+1711+1711=6783GPU-seconds. Report440835 passed (10s).
- Report-check440834 passed anchor/criterion and historical provenance checks.
- [Binding data](binding_data/INDEX.md): CPU440860 (10s),16audited N32pairs.
- BindingCPUplan440878 passed; GPU440886 complete (345s): allsix models increase predictions on all16pairs;
  meanDelta3.75..4.8125 versus4, butonly0..2both-correctpairs/model.
  [Binding report](binding_probe/probe_440886/REPORT.md).
- Cosmos compatibility440874 failed beforeloading (10GPU-seconds), offlinecache
  lookup. Explicit-snapshot CPU440888 found missing slow-tokenizer BPE files;
  compatibility repair is being checked without packages or model changes.
- Known completed GPU allocation7219seconds (2.0053h), plus planned diagnostics.
  Block cap4GPUh. Checkpoints /mnt/ckpts/gabriele/gnn_transformer/native_aggregation_vlm_v3.
  Main data /mnt/data/gabriele/gnn_transformer/v2_clean, unchanged.

[Next decisions](../../../docs/paper/NATIVE_AGGREGATION_NEXT_DECISIONS.md):
finish diagnostic interpretation before another method grid or fresh confirmation.

Per-frame Yes/No diagnostic is preregistered: frozen and fixed V2hidden,
isolated versus indexedfullcontext, to measure precision without counting.

- Frame diagnostic complete, GPU440918/461s: [report](frame_judgments/judgments_440918/REPORT.md). First-token binary accuracy isolated100%both; full frozen68.8%,hidden81.3%. Strict generation parsing0 because every answer appends a period; retained as registered. Context/access effect, not a proved aggregation mechanism.

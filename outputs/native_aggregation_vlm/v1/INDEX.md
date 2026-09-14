# Native MMReD Vision V1

Completed 2026-09-10. **All preregistered mechanism criteria failed.** The simple global-read adapter and after-merge mean control improved native answers over LoRA, without establishing a new aggregation mechanism.

[Interpretation and next controls](../../../docs/paper/NATIVE_AGGREGATION_VISION_RESULTS.md) · [Full report](REPORT.md) · [Figure](comparison.png) · [Analysis JSON](analysis.json)

| Arm | N32/64 exact |
|---|---:|
| sum | 6.5% |
| mean | 30.5% |
| post_sum | 6.5% |
| post_mean | 40.0% |
| global | 40.5% |
| hierarchical | 36.0% |
| lora | 21.5% |
| Frozen | 16.0% |

Seed 0, train N8/16, counts 0–8, 100 test examples per length. Inherited repetition penalty and N16 nuisance-distribution shift are documented in the report.

- [Main runs](main/INDEX.md)
- [Profiles](profile/INDEX.md)
- [Frozen reference](base/INDEX.md)
- [Data census](data_audit/INDEX.md)
- [Checkpoint inspection](inspection/INDEX.md)
- [Generation settings](generation_settings.json)
- [Execution and cost](execution.json): all jobs completed; 12,698 allocated GPU-seconds = 3.53 GPU-hours; up to four GPUs concurrently.

All GPU/heavy CPU work used Slurm. New models and data use the user-designated /mnt/ckpts/gabriele/gnn_transformer and /mnt/data/gabriele/gnn_transformer roots. No additional training expansion followed the failed screen.

# V18 independent native evaluation

Both-seed primary: **True**. Both-seed practical: **True**. Vision milestone: **True**.

| Mode | Seed | N32 | N64 | N64 K0–8 | N64 K9–15 | N64 K16 | N64 K9–16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| native_gate | 20 | 132/136 | 131/136 | 72/72 | 54/56 | 5/8 | 59/64 |
| native_gate | 21 | 132/136 | 131/136 | 72/72 | 54/56 | 5/8 | 59/64 |
| all_open | 20 | 0/136 | 0/136 | 0/72 | 0/56 | 0/8 | 0/64 |
| all_open | 21 | 0/136 | 0/136 | 0/72 | 0/56 | 0/8 | 0/64 |

All272 fresh examples remain in each denominator. Exact requires the complete correct ASCII integer and native EOS within four tokens.

Both modes use the same original native full-vocabulary probe, parameter count, paired data/order and CE plus residual-consistency objective. Every final core is fixed at4590 updates; its one dev sweep is descriptive.

All V18 GPU allocations, including failures and zero allocations, total 3681 GPU-seconds. Peak concurrency is 4 GPUs.

These results do not establish composition with reasoning tokens. The gate remains fixed to the original question, and answer values0–16 are supported during training.

[Full per-K results, native parsing, capture checks, family intervals and provenance](analysis.json).

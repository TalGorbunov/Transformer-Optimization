# V1 native MMReD Vision results

Seed 0; train N8/16; development-selected checkpoints; 100 test examples per length; K<=8.

| Arm | N8 | N16 | N32 | N64 | Pooled N32/64 | Training parameters |
|---|---:|---:|---:|---:|---:|---:|
| Before / sum | 40/100 | 11/100 | 13/100 | 0/100 | 6.5% | 1,409,088 |
| Before / mean | 71/100 | 54/100 | 36/100 | 25/100 | 30.5% | 1,409,088 |
| After / sum | 46/100 | 12/100 | 12/100 | 1/100 | 6.5% | 1,409,088 |
| After / mean | 75/100 | 62/100 | 47/100 | 33/100 | 40.0% | 1,409,088 |
| Global read | 70/100 | 55/100 | 47/100 | 34/100 | 40.5% | 1,409,088 |
| Hierarchical | 75/100 | 58/100 | 42/100 | 30/100 | 36.0% | 1,411,152 |
| LoRA rank 16 | 52/100 | 32/100 | 21/100 | 22/100 | 21.5% | 1,441,792 |
| Frozen reference | 24/100 | 22/100 | 16/100 | 16/100 | 16.0% | 0 |

## Registered contrasts

| Candidate vs control | OOD gain | Paired 95% interval | In-range gain | OOD model latency ratio |
|---|---:|---:|---:|---:|
| sum_vs_post_sum | +0.0 pp | [-1.5, +1.5] pp | -3.5 pp | 1.19× |
| sum_vs_post_mean | -33.5 pp | [-39.5, -27.5] pp | -43.0 pp | 1.21× |
| sum_vs_global | -34.0 pp | [-39.5, -28.0] pp | -37.0 pp | 1.21× |
| sum_vs_hierarchical | -29.5 pp | [-35.5, -23.5] pp | -41.0 pp | 1.16× |
| sum_vs_lora | -15.0 pp | [-19.5, -10.5] pp | -16.5 pp | 1.71× |
| mean_vs_post_sum | +24.0 pp | [+18.5, +30.0] pp | +33.5 pp | 0.98× |
| mean_vs_post_mean | -9.5 pp | [-15.5, -3.5] pp | -6.0 pp | 1.00× |
| mean_vs_global | -10.0 pp | [-16.0, -4.0] pp | +0.0 pp | 1.00× |
| mean_vs_hierarchical | -5.5 pp | [-11.5, +0.0] pp | -4.0 pp | 0.96× |
| mean_vs_lora | +9.0 pp | [+3.0, +15.0] pp | +20.5 pp | 1.41× |

## Descriptive control comparisons

These supplementary contrasts do not alter the preregistered criteria; seed-conditional, unadjusted intervals.

| Control vs LoRA | OOD gain | Paired 95% interval | In-range gain |
|---|---:|---:|---:|
| post_sum_vs_lora | -15.0 pp | [-19.5, -10.5] pp | -13.0 pp |
| post_mean_vs_lora | +18.5 pp | [+11.5, +25.0] pp | +26.5 pp |
| global_vs_lora | +19.0 pp | [+12.5, +25.5] pp | +20.5 pp |
| hierarchical_vs_lora | +14.5 pp | [+8.5, +20.5] pp | +24.5 pp |

Criteria: `{"V1_1_sum_processing_order": false, "V1_2_mean_processing_order": false, "V1_3_practical_screen": {"mean": false, "sum": false}}`

These intervals describe paired-example uncertainty for one seed. They are not independent-seed confirmation.
N16 train/dev have more same-character distractors than test (mean target-character counts 10.31/10.28 vs 6.12); this is a combined length/nuisance-distribution shift, not IID in-range testing or a clean length-only intervention. See data_audit/INDEX.md.
The saved first-token NLL is generation-policy NLL after inherited repetition_penalty=1.05, not raw LM NLL; this also broke dev-accuracy ties.
Policy likelihood, parsing, absolute/signed error, memory, dev histories and provenance are retained in analysis.json.
The global control redundantly reconstructs the SDPA read; its timing is not an optimized global-attention baseline.
Latency was measured during concurrent GPU jobs and is descriptive; claims of speed require dedicated profiling.

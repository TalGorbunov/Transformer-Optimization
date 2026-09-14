# V14 independent native evaluation

Audit passed. Both-seed primary: **True**. Both-seed practical: **False**.

Both arms train the core against an empirical24-occurrence null mean, then freeze it and distill the mean into the same predictor. Centered subtracts N times the projected mean; offset subtracts it once. Native inference uses N+1 streams without reference images.

| Condition | Seed | N32 | N64 | N64 K0–8 | N64 K9–15 | N64 K16 | N64 K9–16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| centered | 18 | 109/136 (80.1%) | 41/136 (30.1%) | 20/72 (27.8%) | 18/56 (32.1%) | 3/8 (37.5%) | 21/64 (32.8%) |
| centered | 19 | 97/136 (71.3%) | 43/136 (31.6%) | 27/72 (37.5%) | 8/56 (14.3%) | 8/8 (100.0%) | 16/64 (25.0%) |
| offset | 18 | 31/136 (22.8%) | 16/136 (11.8%) | 16/72 (22.2%) | 0/56 (0.0%) | 0/8 (0.0%) | 0/64 (0.0%) |
| offset | 19 | 50/136 (36.8%) | 24/136 (17.6%) | 24/72 (33.3%) | 0/56 (0.0%) | 0/8 (0.0%) | 0/64 (0.0%) |

| Seed | N64 gain | Family bootstrap95%CI | N32 change | Primary | Practical |
|---:|---:|---:|---:|---|---|
| 18 | 18.38pp | [11.76,25.00]pp | 57.35pp | True | False |
| 19 | 13.97pp | [8.09,19.85]pp | 34.56pp | True | False |

Each seed must gain at least7/136 N64 answers and lose at most6/136 N32 answers. Practical additionally requires centered N32≥123/136,N64≥109/136,and N64K9–16≥52/64.

Only the fixed final core4590/student8000 checkpoint is evaluated. Development64 is descriptive. All272 fresh contexts remain in every denominator; all answer values0–16 occur in training.

Independent checks bind28prior exclusions,918pairs/177120target positions,972frozen null groups with24ordered occurrences,512000student presentations,all conversion prefixes,and raw native argmax/EOS/NLL. Initial core/student bytes and data orders match; first core gradients may differ between arms.

Final acceptance requires the separate audit of all four final checkpoints and complete GPU accounting. Empirical bank fit is not independent population calibration. No reasoning-composition or new attention-operator claim follows.

[Every count/length partition, calibration diagnostic and source binding](analysis.json).

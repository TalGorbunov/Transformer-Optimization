# V13 independent native evaluation

Audit passed. Both-seed primary: **False**. Both-seed practical: **False**.

Both methods use the same trainable core and null predictor, full-answer CE, residual consistency and training-only null calibration. Centered subtracts N times the predicted projected null mean; offset subtracts it once. Neither uses reference images at inference.

| Condition | Seed | N32 | N64 | N64 K0–8 | N64 K9–15 | N64 K16 | N64 K9–16 | Selected step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| centered | 16 | 44/136 (32.4%) | 11/136 (8.1%) | 11/72 (15.3%) | 0/56 (0.0%) | 0/8 (0.0%) | 0/64 (0.0%) | 4590 |
| centered | 17 | 48/136 (35.3%) | 15/136 (11.0%) | 15/72 (20.8%) | 0/56 (0.0%) | 0/8 (0.0%) | 0/64 (0.0%) | 4590 |
| offset | 16 | 61/136 (44.9%) | 20/136 (14.7%) | 20/72 (27.8%) | 0/56 (0.0%) | 0/8 (0.0%) | 0/64 (0.0%) | 4590 |
| offset | 17 | 26/136 (19.1%) | 0/136 (0.0%) | 0/72 (0.0%) | 0/56 (0.0%) | 0/8 (0.0%) | 0/64 (0.0%) | 4590 |

| Seed | N64 gain | Family bootstrap95%CI | N32 change | Primary | Practical |
|---:|---:|---:|---:|---|---|
| 16 | -6.62pp | [-9.56,-3.68]pp | -12.50pp | False | False |
| 17 | 11.03pp | [7.35,14.71]pp | 16.18pp | True | False |

Primary requires centered to gain at least 7/136 N64 answers and lose at most 6/136 N32 answers in each seed. Practical additionally requires centered N32 ≥123/136, N64 ≥109/136, and N64 K9–16 ≥52/64.

All 272 fresh contexts per run remain in the denominator, including malformed or truncated outputs. Every answer value 0–16 occurs in training. The test measures length extrapolation.

Independent checks bind 27 prior exclusions, 1,782 ordinary training scenes, 918 pairs, 177,120 valid target positions, 972 null groups with 24 ordered occurrences each, 73,440 auxiliary presentations, five development checkpoints, and raw native argmax/EOS/NLL.

Initial losses and gradients match at zero U/C2; subsequent gradients may differ. The report audits recorded losses and source-bound graph routing, without reconstructing unsaved intermediate messages or targets. Calibration may coadapt with the core.

Final acceptance also requires the separate audit of all selected checkpoints and complete GPU-allocation accounting. No reasoning-composition or new attention-operator claim follows from this comparison.

[Every count/length and numeral-prefix partition, parse/EOS statistics, losses and provenance](analysis.json).

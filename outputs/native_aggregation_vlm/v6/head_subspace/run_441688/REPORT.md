# V6 fixed-head message subspaces

All dev-selected checkpoints and all452contexts each. P preserves the selected local head probabilities mathematically with its bias unchanged; native decoding is not invariant. Q means head-invisible, not irrelevant.

| Checkpoint | Cell | Rank | Mean negative-sum native P norm | Mean negative-sum native Q norm | Mean native P/Q cosine |
|---|---|---:|---:|---:|---:|
| control_seed6 | length_N16 | 2 | 169.81 | 149.44 | 0.22183 |
| control_seed6 | length_N32 | 2 | 401.56 | 340.91 | 0.22627 |
| control_seed6 | length_N64 | 2 | 861.24 | 737.09 | 0.23061 |
| control_seed6 | unseen_count_N32 | 2 | 280.85 | 241.72 | 0.21305 |
| control_seed6 | unseen_count_N64 | 2 | 739.65 | 634.71 | 0.21896 |
| control_seed7 | length_N16 | 2 | 300.35 | 205.49 | 0.34938 |
| control_seed7 | length_N32 | 2 | 694.12 | 470.87 | 0.34605 |
| control_seed7 | length_N64 | 2 | 1464.4 | 994.18 | 0.35009 |
| control_seed7 | unseen_count_N32 | 2 | 484.93 | 328.97 | 0.34636 |
| control_seed7 | unseen_count_N64 | 2 | 1258.6 | 854.2 | 0.35008 |
| aligned_seed6 | length_N16 | 2 | 543.93 | 197.44 | 0.27633 |
| aligned_seed6 | length_N32 | 2 | 1263.2 | 460.27 | 0.2774 |
| aligned_seed6 | length_N64 | 2 | 2712 | 987.41 | 0.27728 |
| aligned_seed6 | unseen_count_N32 | 2 | 862.21 | 320.53 | 0.28267 |
| aligned_seed6 | unseen_count_N64 | 2 | 2311.8 | 846.35 | 0.27837 |
| aligned_seed7 | length_N16 | 2 | 692.36 | 344.04 | 0.65596 |
| aligned_seed7 | length_N32 | 2 | 1639.4 | 810.85 | 0.66875 |
| aligned_seed7 | length_N64 | 2 | 3510.8 | 1733.9 | 0.67279 |
| aligned_seed7 | unseen_count_N32 | 2 | 1156 | 567.74 | 0.67738 |
| aligned_seed7 | unseen_count_N64 | 2 | 3029.1 | 1493.8 | 0.67499 |

FP64 invariance checks passed for every context: True.
Each run retains rank/condition numbers, every precision discrepancy, perN/K/class summaries, and exact96D total/positive/negative/category sum vectors with P/Q parts. No3584D per-image matrices are written.

Head-invisible Q is not nuisance: native count CE also trains these coordinates; the learned head changes during training.
Rank at most2 describes one fixed three-way softmax head, not total representational information or all training-gradient directions.
The same bias is retained in all local probability comparisons; local invariance does not imply native decoder or future-query invariance.
U does not generally preserve orthogonality: P/Q message norms or native norms are not additive explained-variance fractions; native cross-cosines are retained.
P uses the fixed Euclidean message-coordinate metric; dimension-normalized energies are reported because94 versus2 dimensions alone can create norm imbalance.
Question-conditioned hidden states and rendered Step labels can change across paired lengths; observed norm growth is not an intervention holding evidence/query fixed.
Stored native attention-output norms do not establish full residual-stream dominance or a normalization bottleneck.
Gold semantic classes only stratify descriptive reports and sum groups; they never determine P/Q or select examples/checkpoints.
All four dev-selected checkpoints/all452contexts are included; no fitting, efficacy filtering, VLM forward or new external tally is performed.

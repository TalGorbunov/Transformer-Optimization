# Learned selection on the MMReD-derived identity join

| Method | Seed | Seen N32 | Seen N64 | Held N32 | Held N64 |
|---|---:|---:|---:|---:|---:|
| clip | 22 | 12/108 | 12/108 | 12/108 | 12/108 |
| sigmoid | 22 | 17/108 | 18/108 | 15/108 | 17/108 |
| softmax | 22 | 22/108 | 21/108 | 20/108 | 15/108 |
| clip | 23 | 12/108 | 12/108 | 12/108 | 12/108 |
| sigmoid | 23 | 16/108 | 16/108 | 15/108 | 17/108 |
| softmax | 23 | 31/108 | 30/108 | 27/108 | 27/108 |

Primary criterion: **False**. Practical criterion: **False**.

Known learned gated set pooling; contentsensitive MMReD-derived task. Fixed six included images, nine familiar people; not a general relational or reasoning-composition proof.

All malformed and truncated outputs are included. Checkpoint selection was fixed-final only.

Allocated GPU-seconds including failures: 11508.

[Full analysis](analysis.json) · [Every outcome](outcomes.json) · [Every test-prefix diagnostic](geometry.json)

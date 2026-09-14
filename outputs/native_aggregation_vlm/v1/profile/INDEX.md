# V1 vision profiles

Slurm array 440492: all four tasks completed successfully. Each used two disjoint training examples, one optimizer step, and two test examples per length. These are feasibility checks, not efficacy estimates.

| Arm | Train seconds/example | Peak allocated GB | N64 seconds/example |
|---|---:|---:|---:|
| global | 1.71 | 14.59 | 2.88 |
| hierarchical | 1.76 | 19.69 | 2.93 |
| post_sum | 1.71 | 14.59 | 2.86 |
| sum | 1.31 | 16.95 | 2.87 |

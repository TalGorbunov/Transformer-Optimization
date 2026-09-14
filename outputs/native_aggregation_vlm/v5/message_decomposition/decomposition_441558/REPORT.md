# V5 mapped semantic-frame message decomposition

Optional descriptive diagnostic on all108 familiar-count anchor families per model. No correctness filtering.

Across both extensions, 150/3456 mapped image pairs are byte-identical; 3306 have renumbered Step labels.

| Arm | Seed | Extension | Drift norm | Added-negative norm | Normalization norm | Total change norm | Drift / added cosine | Saved-merge reconstruction L2 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| sum | 4 | N16→N32 | 6.87879 | 208.89 | 0 | 209.566 | 0.0578203 | 3.66871e-05 |
| sum | 4 | N16→N64 | 10.5487 | 624.861 | 0 | 625.704 | 0.0615208 | 9.76963e-05 |
| sum | 5 | N16→N32 | 6.78429 | 223.915 | 0 | 224.701 | 0.101733 | 3.50881e-05 |
| sum | 5 | N16→N64 | 8.93113 | 673.87 | 0 | 674.676 | 0.0868997 | 9.2595e-05 |
| mean | 4 | N16→N32 | 0.184465 | 14.662 | 14.6682 | 0.404707 | 0.00545249 | 2.82079e-06 |
| mean | 4 | N16→N64 | 0.148709 | 21.9848 | 22.0023 | 0.632536 | 0.00434255 | 3.46088e-06 |
| mean | 5 | N16→N32 | 0.0549545 | 11.3811 | 11.3696 | 0.144512 | 0.203631 | 2.366e-06 |
| mean | 5 | N16→N64 | 0.0436816 | 17.0809 | 17.0544 | 0.218015 | 0.182927 | 3.07882e-06 |

Each entry is the mean of108 pair-level statistics. Norms of additive vectors do not add. Full per-pair values, cosine denominators, reconstruction errors and separately summarized residual/native ratios are retained in JSON.

Mapped-frame drift combines changed Step pixels with query/context and numerical changes. It does not isolate context-induced corruption or demonstrate a causal mechanism.

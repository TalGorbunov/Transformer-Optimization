# V10 background-direction diagnostic

First-token results on17 fixed families at both lengths, all four selected cores. No fitting or generation.

| Model | N | Intervention | First-token correct | Mean first-token NLL |
|---|---:|---|---:|---:|
| ce_s14 | 32 | base | 2/17 | 21.5952 |
| ce_s14 | 32 | background | 17/17 | 0.0021 |
| ce_s14 | 32 | sham | 4/17 | 23.0992 |
| ce_s14 | 64 | base | 1/17 | 31.7766 |
| ce_s14 | 64 | background | 15/17 | 0.3484 |
| ce_s14 | 64 | sham | 1/17 | 34.0445 |
| ce_s15 | 32 | base | 3/17 | 11.1162 |
| ce_s15 | 32 | background | 17/17 | 0.0010 |
| ce_s15 | 32 | sham | 2/17 | 15.0753 |
| ce_s15 | 64 | base | 0/17 | 35.9322 |
| ce_s15 | 64 | background | 17/17 | 0.0208 |
| ce_s15 | 64 | sham | 3/17 | 19.6942 |
| consistency_s14 | 32 | base | 6/17 | 12.4859 |
| consistency_s14 | 32 | background | 17/17 | 0.0017 |
| consistency_s14 | 32 | sham | 3/17 | 22.1862 |
| consistency_s14 | 64 | base | 4/17 | 33.2814 |
| consistency_s14 | 64 | background | 17/17 | 0.0700 |
| consistency_s14 | 64 | sham | 3/17 | 32.2583 |
| consistency_s15 | 32 | base | 9/17 | 4.3824 |
| consistency_s15 | 32 | background | 17/17 | 0.0014 |
| consistency_s15 | 32 | sham | 5/17 | 13.8851 |
| consistency_s15 | 64 | base | 4/17 | 18.6618 |
| consistency_s15 | 64 | background | 17/17 | 0.0120 |
| consistency_s15 | 64 | sham | 2/17 | 28.2287 |

K10–16 all share first token1; these results do not establish exact count decoding.
Bare native replay checks passed 34/34. Original V10 comparisons are descriptive; original failures remain unchanged.

[All K/length/model/intervention rows, raw tensors and limits](analysis.json).

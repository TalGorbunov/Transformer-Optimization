# V10 independent native evaluation

Audit passed. Both-seed primary: **True**. Both-seed practical: **False**.

All test answer values0–16 occur in adaptation. Training length is at most16; fresh test families extend from32 to64frames. The two conditions use identical native inference and differ only by residual-consistency loss.

| Condition | Seed | N32 | N64 | N64 K0–8 | N64 K9–15 | N64 K16 | N64 K9–16 | Selected step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ce | 14 | 21/136 (15.4%) | 8/136 (5.9%) | 8/72 (11.1%) | 0/56 (0.0%) | 0/8 (0.0%) | 0/64 (0.0%) | 4590 |
| ce | 15 | 26/136 (19.1%) | 0/136 (0.0%) | 0/72 (0.0%) | 0/56 (0.0%) | 0/8 (0.0%) | 0/64 (0.0%) | 4590 |
| consistency | 14 | 43/136 (31.6%) | 20/136 (14.7%) | 20/72 (27.8%) | 0/56 (0.0%) | 0/8 (0.0%) | 0/64 (0.0%) | 4590 |
| consistency | 15 | 52/136 (38.2%) | 19/136 (14.0%) | 19/72 (26.4%) | 0/56 (0.0%) | 0/8 (0.0%) | 0/64 (0.0%) | 4590 |

| Seed | N64 gain | Family bootstrap95%CI | N32 change | Primary | Practical |
|---:|---:|---:|---:|---|---|
| 14 | 8.82pp | [5.88,11.03]pp | 16.18pp | True | False |
| 15 | 13.97pp | [10.29,17.65]pp | 19.12pp | True | False |

Exact requires the complete correct integer plus native EOS within four tokens. All272 examples per run remain in the denominator.

Primary requires at least7/136 additional N64 answers and no more than6/136 fewer N32 answers in each seed. Practical additionally requires consistency N32≥123/136, N64≥109/136, and N64 K9–16≥52/64.

Only K0–8 has cross-length pair supervision. K9–15 has same-length scene pairs;54saturated K16pairs duplicate one SID and contribute exactly zero regularizer while retaining both CE slots. Historical saturated reuse is explicitly recorded.

Independent checks bind25prior exclusions,1,782unique training contexts,918pairs,all177,120valid causal training targets,the shared[1]prefix,complete log reductions,all five dev checkpoints,and raw full-vocabulary native argmax/EOS/NLL.

Final acceptance also requires the separate all-selected-checkpoint audit. This comparison does not establish unseen-answer transfer, reasoning composition, or a new attention operator.

[Every K/length cell, parse/EOS/first-token statistics, loss diagnostics and provenance](analysis.json).

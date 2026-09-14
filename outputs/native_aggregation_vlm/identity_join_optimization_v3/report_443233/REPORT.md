# Fixed training-only optimization pilot

All four fixed seed24 endpoints; no test evaluation or confirmed efficacy.

| Arm | Train first | Train whole | Train families | Dev first | Dev whole | Dev triples | Train/ID pass |
|---|---:|---:|---:|---:|---:|---:|---|
| clip_ce | 12/108 | 12/108 | 0/18 | 6/54 | 6/54 | 0/18 | False/False |
| clip_consistency | 12/108 | 12/108 | 0/18 | 6/54 | 6/54 | 0/18 | False/False |
| sigmoid_ce | 40/108 | 40/108 | 0/18 | 18/54 | 18/54 | 0/18 | False/False |
| sigmoid_consistency | 42/108 | 42/108 | 0/18 | 18/54 | 18/54 | 0/18 | False/False |

Allocated GPU-seconds including failures: 2410.

[Analysis](analysis.json) · [All outcomes](outcomes.json)

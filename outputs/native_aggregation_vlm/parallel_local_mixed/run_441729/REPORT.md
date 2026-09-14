# Mixed local/text-only rows and broadcast cache

Computational integrity: PASS. Registered numerical gate: FAIL.

Two software scenes;80local+2global serial references;92VLM/86visual calls;492rowoutputs.

| Comparison policy | Rows | Violations | Maximum full-vocabulary TV | Top1 mismatches |
|---|---:|---:|---:|---:|
| local_prefill | 80 | 0 | 0.0132978 | 0 |
| full_vocab | 166 | 1 | 0.0202965 | 0 |

Local-prefill TV/top1 are descriptive; that gate uses conditional01/p1/numeric mass.
Global-prefill and cached/full rows use full-vocabulary TV<=.02 and exacttop1.

Software fidelity only: independent local rows never send evidence to the text-only global row.
The Therefore/colon prefix is forced identically, not generated reasoning or a correctness signal.
Hidden/centered-logit differences are descriptive; local output fidelity does not imply hidden equality.
Cache prefix retention and mask/position correspondence are exact; cached/full distributions use fixed .02TV/top1 limits.
Every violation is recorded. No tolerance search, training, aggregation or task-accuracy claim.
State capture and numerical checks add overhead; timing is for this instrumented harness.

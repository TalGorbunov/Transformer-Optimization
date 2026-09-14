# Joint-code oracle: fixed-budget training dynamics

The independent report 443476 passed its computation/provenance checks, while the fixed first-token screen failed at 38/108 and 0/18 complete families. The review below uses only recorded JSON. No tensor, model, native head, optimizer, or new Slurm job was executed.

All 600 logs match the original seed 24 presentation order, targets, strict prefixes and learning rates. The 4,800 length-pair presentations comprise 88 complete cycles of all 54 pairs and 48 pairs in cycle 89. A pair contributes the mean of its N8/N16 scene losses; each scene loss averages its full name-plus-EOS target positions. All 21,334 training target positions and reduction fields were checked. Losses are recorded before each update, so a cycle is an online trajectory summary, not a rescore of one checkpoint.

| Run | Cycle 1 mean CE | Cycle 44 mean CE | Cycle 88 mean CE | Cycles 79–88 relative change |
|---|---:|---:|---:|---:|
| Local joint-code oracle | 2.48615 | 0.85925 | 0.63422 | -0.884% |
| Parent product | 2.47803 | 0.53357 | 0.39414 | -1.701% |
| Readout-only query product | 2.48060 | 0.52607 | 0.39466 | -1.587% |
| Additive | 2.47418 | 0.55222 | 0.47901 | -0.617% |

For the joint-code oracle, the last ten complete cycles fall from 0.63988 to 0.63422, with descriptive OLS slope −0.000672 per cycle and 7/9 downward adjacent changes. Their first-five/last-five means are 0.63732/0.63372. The shared learning rate falls from 0.00005241 to 0.00001029 during this interval, versus the earlier peak 0.001. Thus small remaining improvement and an imposed diminishing update scale coexist; these observations do not demonstrate convergence.

| Joint-code loss | Cycle 1 | Cycle 44 | Cycle 88 |
|---|---:|---:|---:|
| First name token | 4.67659896 | 1.89108331 | 1.39256221 |
| EOS | 0.00292265 | 0.00009622 | 0.00007468 |
| Second name token | 3.68670700 | 0.00003119 | 0.00002737 |

First-token and EOS means each cover 108 scenes per complete cycle; second-name-token means cover 24 Sandra/Noah occurrences. Separate objective contributions retain division by full target length. Almost all final residual loss concerns the first name token, while teacher-forced continuation and EOS losses are tiny. This does not establish free-generation whole-answer accuracy. The 48-pair partial cycle 89 has CE 0.63626 and is retained separately, without treating its different subset as a full-cycle trend.

Perfect local person/room codes did not make this fixed readout-and-optimization recipe fit. Consequently the previous failed screens alone cannot establish that extracting semantics from native local states is the sole obstacle. This oracle changes the representation and removes the learned local encoder; it is not a matched proof of where the original models fail. Its worse loss does not make privileged codes less informative. Readout conditioning, optimization under the decaying 600-step schedule, and adequacy of the selected readout remain unresolved. Neither capacity impossibility nor eventual success with more training follows. No contrastive/encoder fit, continuation or new variant is released.

All 89 cycle records, exact input hashes and prior comparison records are retained in [analysis.json](analysis.json), SHA256 `58549dc0854fce419fe91c93d84e749551e7c8377028a31fcc527c4cbefab269`. Original sources and results are unchanged.

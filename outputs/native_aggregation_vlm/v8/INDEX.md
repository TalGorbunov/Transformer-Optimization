# V8: native residual consistency

**Verified: both-seed relative criterion passed; practical criterion failed at N64.** Same deployed parallel native operator and paired training; only the objective differs.

| Condition | Seed | N16 | N32 | N64 | OOD N32/64 | Unseen K9–16 |
|---|---:|---:|---:|---:|---:|---:|
| CE | 10 |108/108|79/108|0/108|79/216|0/128|
| CE | 11 |108/108|55/108|12/108|67/216|0/128|
| Consistency | 10 |108/108|108/108|72/108|180/216|0/128|
| Consistency | 11 |108/108|108/108|53/108|161/216|0/128|

- [Independent report441974](report_441974/REPORT.md),86CPU-seconds; all1808test answers plus1440dev answers audited.
- [Selected checkpoint audit441997](checkpoint_audit/run_441997/REPORT.md),80GPU-seconds;40native-head replays passed, five cached/full numerical failures remain descriptive, all top1 unchanged.
- [Figure and acceptance441998](finalization/final_441998/REPORT.md),1CPU-second. All four selected checkpoint identities bound; practical gate remains false. Figure visually checked.
- [Fixed-state response comparison441992](response_surface/run_441992/REPORT.md),16CPU-seconds: each model54/54 synthetic N16; CE17/108 versus consistency62/108 synthetic N64. Diagnostic only, no VLM calls or fit.
- [Training plan](train_check_441937/plan.json), [independent data release](data_check_441961/summary.json), [measured resource release](main_release_441963/release.json).
- [Execution/budget](execution.json):4666GPU-seconds(1.2961hours), below3.25hour cap, max4project GPUs.

Sources and predictions remain frozen. Original response unit441984 retained; completion-binding amendment refrozen441989. Pending typo invocation441975 and pending diagnostic441988 cancelled before execution. RESULTS.md unchanged.

[Research interpretation](../../../../docs/paper/NATIVE_AGGREGATION_VISION_V8_RESULTS.md). No reasoning-composition or new-attention claim.

[Unseen-output diagnostic442059](unseen_trace_diagnostic/run_442059/REPORT.md): both consistency seeds output8 on all128unseen counts. Every model has0/128correct first tokens; no immediate-EOS-after-correct-first explanation. CPU12s, no model calls.

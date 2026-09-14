# Fixed-uniform identity-join control

**Completed: computational audits passed; training and development criteria failed.** The fixed seed24, 600-update CE-only run achieved 36/108 whole training answers and 18/54 whole development answers, with 0/18 complete families in each partition. First-token scores were identical. All 162 answers completed and parsed; no outputs were truncated.

The selector stayed exactly 0.5 on every actual item through all training steps and all 324 evaluated prefixes. The model retained 1,041,697 coordinates, of which 1,041,600 were trainable. This failure does not require learned gate closure; it does not distinguish the remaining payload, native readout, and optimization limitations. No test, extrapolation, or reasoning evaluation was performed.

The independent report is **443271**. Allocated GPU cost was **293 seconds**: profile 443261 used 85 and main 443264 used 208, within the registered 990-second cap and one-GPU limit.

[Independent report](report_443271/REPORT.md) · [Verified analysis](report_443271/analysis.json) · [Summary](report_443271/summary.json) · [Execution record](execution.json) · [Results note](../../../docs/paper/NATIVE_AGGREGATION_UNIFORM_JOIN_RESULTS.md)

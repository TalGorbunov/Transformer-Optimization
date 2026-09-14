# Frozen-model MMReD reasoning baseline

Fresh18paired anchors/54main scenes; frozen Qwen and Cosmos; within-model direct32 versus prompted reasoning512 with a128-prefix endpoint. No trained-method or composition claim.

Status: implementation. Registered before staging/outcomes in PREREG_AGG.md. Resource cap1GPU-hour; GPU jobs wait for V4 resources.

- [Final CPU plan](check_441210/plan.json)
- [Throughput calibration](throughput/)
- [Qwen main441249](main/qwen_441249/REPORT.md)
- [Cosmos main441250](main/cosmos_441250/)

Original resource gates failed; operational envelope explicitly amended to1.1GPUh before mains. Both efficacy policies/data remain unchanged. Report441252 waits for both complete runs.

Completed: [verified final report](report_441252/REPORT.md), [analysis](report_441252/analysis.json), [figure](report_441252/comparison.png). Total1204GPU-seconds(0.33444h), including every failed profile and timing calibration. Qwen returned bare integers under both prompts; strict format score0 is not its counting accuracy. Cosmos512 truncated28/36OOD scenes. No general reasoning/composition conclusion.

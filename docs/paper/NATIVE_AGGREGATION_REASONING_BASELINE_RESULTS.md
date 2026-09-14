# Frozen-model reasoning baseline on MMReD Vision

Completed2026-09-10. This compares two prompted output policies within each frozen model, not a new aggregation method. All54 scenes were fresh when registered (18anchors, K0..8, N16/32/64). The direct policy allows32tokens; reasoning allows512 with a censored128-token endpoint. Native NF4/bf16 SDPA,392px, greedy decoding and identical images/questions; exact condition-specific tags and normal EOS are required.

| Fixed model | Policy | N16 exact | N32/N64 exact | OOD parsed | OOD truncated |
|---|---|---:|---:|---:|---:|
| Qwen2.5-VL7B | direct32 |0/18|0/36|0/36|0/36|
| Qwen2.5-VL7B | reason512 |0/18|0/36|0/36|0/36|
| Cosmos-Reason1-7B | direct32 |4/18|4/36|36/36|0/36|
| Cosmos-Reason1-7B | reason512 |4/18|1/36|8/36|28/36|

Every Qwen output was a bare integer under both policies (54/54 each), rather than the required tagged grammar. This post-hoc structural inspection explains its strict zero; it is not evidence of zero counting ability, and no answer parser was retroactively changed. The instructed reasoning policy did not elicit a trace from this model. Its within-policy token totals were111direct and108reasoning.

Cosmos reasoning512-minus-direct OOD difference is−8.33pp, whole-anchor interval[−22.22,0]. At128, every OOD reasoning trajectory was truncated. At512,28/36 were still truncated. The primary effect screen fails, but this mainly exposes a restrictive token budget and does not establish that unbounded reasoning cannot help. The official Cosmos guide recommends at least4096tokens and reports BF16; this assay uses512, NF4/bf16 and a fixed greedy policy. Plain-versus-reasoning model differences are not causal because their training differs.

Measured generation: Qwen43.9s direct/42.2s instructed reasoning; Cosmos52.6s direct/645.9s reasoning, with493/24482 generated tokens respectively. Both native EOS IDs were enforced; every malformed/truncated output remained in exact denominators. The128 endpoint was censored from the same512 trajectory, with no separately measured latency.

The resource envelope was explicitly amended from1 to1.1GPUh before mains after conservative profile estimates failed; all original failures remain recorded. Actual aggregate use was1204GPU-seconds(0.33444h), including failed guards for this assay, software profiles and forced-cap timing-only calibrations. CPU report441252 independently verified prompts, sources, checkpoint metadata, exact IDs/EOS/grammar, costs and paired statistics.

Canonical [report](../../outputs/native_aggregation_vlm/reasoning_baseline/report_441252/REPORT.md), [audited analysis](../../outputs/native_aggregation_vlm/reasoning_baseline/report_441252/analysis.json), and [execution ledger](../../outputs/native_aggregation_vlm/reasoning_baseline/execution.json).

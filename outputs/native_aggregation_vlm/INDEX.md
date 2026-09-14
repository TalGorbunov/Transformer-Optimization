# Native aggregation on MMReD Vision

V0 is a software integration smoke only, with two N8 examples per split.
No extrapolation or efficacy claim. Protocol: PREREG_AGG.md (2026-09-08 V0).
Actual images go through the ordinary VLM forward; no per-frame supervision or external tally.
No results yet. Checkpoints: /mnt/ckpts/gabriele/gnn_transformer/native_aggregation_vlm/.

V0 staging433137 verified six samples and hashes; GPU smoke433138 depends on P1.

V0 completed in35s: actual image forward, finite backward, checkpoint reload and
cached native generation passed. Canonical smoke/sum_seed0_20260908_170316_433138_331049.
Peak training CUDA11.08GB. n=2 per split; no efficacy or extrapolation claim.

## 2026-09-10: direct V1 comparison completed

[Canonical results](v1/INDEX.md). Global-read adapter 40.5%, after-merge mean 40.0%, before-merge mean 30.5%, hierarchical 36.0%, LoRA 21.5%, frozen 16.0% on N32/64; both sum arms 6.5%. All preregistered mechanism criteria failed. Single seed, documented data-law shift and policy-NLL caveat. Total 3.53 GPU-hours; no automatic training expansion.

## 2026-09-10: clean V2 attribution and V3 placement test

[V2 complete](v2/INDEX.md),2.305GPUhours. Global26.9% versus hidden29.2%
familiarOOD; primary attribution criterion failed. Last-query channel swaps,
question-prefix diagnostic and frozen ridge probes do not establish a repair.
An oracle-marginal audit identifies remaining statistical shortcuts in the law.

[V3 in progress](v3/INDEX.md): PRE/POST native-width nonlinear values x seeds0,1,
array440830, matched parameters and final-answer training. CPU/model profiles
passed. Test reuse is exploratory. A separate16-pair input challenge preserves
character/room marginals while changing conjunction count. V3cap4GPUhours.

- V3 complete: two-seed PRE/POST primary failed; fixed-marginal responses remain joint-sensitive; isolated first-token frame judgment100%both, full-context68.8/81.3%. Cosmos software gate passed.7724GPU-seconds. [V3 index](v3/INDEX.md).
- V4 scene-diversity control: four main runs440986 running after matched profiles. [V4 index](v4/INDEX.md).
- Fresh frozen-model direct/reasoning baseline is registered and in preparation. [Assay index](reasoning_baseline/INDEX.md).

V4 complete: [verified report](v4/REPORT.md),6906GPU-seconds; both-seed primary screen failed (+2.31/+19.91pp). Reasoning-baseline CPU validation in progress after complete fresh-data stage441018.

- Paired-minibatch216 controls failed: code68, visual product77, additive73 correct, all0/18families. Reports443742/443743 passed;217GPU-seconds total. [Results](../../docs/paper/NATIVE_AGGREGATION_ORIENTATION_PAIRED_RESULTS.md). Native/fresh held.

- First-query supervision control443775/443779 failed74/216,0families; numericalaudit passed,69GPU-seconds. Saved-code geometry443772 passed4CPU-seconds and shows query dominance/small slopes descriptively. Preparing one post-SiLUquery comparison.

- Post-SiLUquery controls failed84/84/74of216,allzero families; reports443794/443795 passed,245GPU-seconds. Query-placement line closed, native/fresh held. [Results](../../docs/paper/NATIVE_AGGREGATION_POST_QUERY_RESULTS.md).

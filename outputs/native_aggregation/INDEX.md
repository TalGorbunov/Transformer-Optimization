# Native aggregation experiments

Design: docs/paper/NATIVE_AGGREGATION_PROPOSAL.md. Protocols/outcomes: PREREG_AGG.md.
Checkpoints: /mnt/ckpts/gabriele/gnn_transformer/native_aggregation/.
Data: /mnt/data/gabriele/gnn_transformer/.

## P0: native-output feasibility

CPU433097 passed13 new tests and existing core suites. GPU433106 completed in3m53s.
72 examples per length. Exact counts N8/N32/N64/N128:

| Arm | Counts | Canonical run under p0/ |
|---|---|---|
| LoRA |62/29/20/15|lora_seed0_20260908_163350_433106_313348|
| Sum+LoRA |65/32/19/11|sum_seed0_20260908_163437_433106_313473|
| Mean+LoRA |67/46/25/13|mean_seed0_20260908_163543_433106_313685|
| Base strict |0/0/0/0|base_seed0_20260908_163318_433106_313176|

Base continues with explanations: post-hoc leading-integer counts50/20/13/13.
All trained outputs parse correctly. Sum misses all efficacy predictions; do not
claim extrapolation improvement. Sources, predictions and checkpoint paths saved
inside each run. smoke/ is runtime validation only, not an efficacy result.

## P1: exploratory diagnostic

Pre-registered after seeingP0: raw/centered messages × block64/16, plus LoRA,
fixed epochs3/9. Same samples reused; not independent confirmation. No results yet.

P1 execution: CPU433131 passed15 tests; GPU433133; CPU report433142.
First CPU433127 caught/fixed cancellation roundoff before GPU submission.

## Completed P1 and P2

Full interpretation: docs/paper/NATIVE_AGGREGATION_PILOT_RESULTS.md.
P1 fixed9 long-N: LoRA27.8%, raw64 37.5%, centered64 25.0%, raw16 31.9%, centered16 5.6%.
P1 analysis/figure: p1/analysis.json and p1/fixed_epochs.png (all cells retained).
P2 GPU433157 rank16 LoRA26.4% at fixed9; canonical outputs/native_aggregation/p2/rank16/lora_seed0_20260908_170734_433157_333979.
Raw64 +11.11pp over this approximate parameter control at fixed9, with worse NLL
and higher latency. With dev-selected checkpoints: raw64 31.25%, rank8 LoRA28.47%,
rank16 LoRA26.39%. One seed, same samples reused; preliminary gain, not confirmation.
P2 paired outcomes: p2/comparison.json.

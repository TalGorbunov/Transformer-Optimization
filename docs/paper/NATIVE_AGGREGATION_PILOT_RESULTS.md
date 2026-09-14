# Native aggregation: implementation and initial experiments

2026-09-08. The prototype runs through the model's own language head and normal
KV cache. It shows a preliminary text accuracy gain, with substantial remaining
uncertainty about generalization, checkpoint selection and compute efficiency.
It is not yet a demonstrated improvement on MMReD Vision or reasoning models.

A [2026-09-10 literature audit](NATIVE_AGGREGATION_POSITIONING.md) identifies close
HiLS/Landmark and conditional Deep Sets antecedents. The missing processing-order,
learned hierarchical-fusion and compute controls limit interpretation of these
results. This positioning update does not change any reported experiment.

**Metric clarification (2026-09-10):** the saved text P1/P2 source snapshots also
compute NLL from processed generation scores, and cached Qwen2.5-3B uses
repetition_penalty=1.05. The reported NLL values describe the generation policy,
not raw LM probabilities; this also applies to development tie-breaking. Numeric
results and exact-answer comparisons are unchanged.

The subsequent [direct vision comparison](NATIVE_AGGREGATION_VISION_RESULTS.md)
is now complete; its mechanism controls failed despite useful adapter gains.

## What changed

One middle decoder layer retains ordinary attention and adds parallel local reads
from its native Q/K/V. A shared small network processes each query-conditioned
read before sum/mean merging; the result enters the native residual stream.
There are no per-frame verdict labels, parsed question predicates, special numeric
outputs, position resets, fences or additional backbone traversals in this branch.
The operation is available at each cached decoding step. Its reuse during reasoning
is an architectural capability, not a demonstrated accuracy result.

The implementation is in `gnnformer/native_aggregation.py`; text and vision
harnesses are `scripts/native_aggregation.py` and `scripts/native_aggregation_vlm.py`.
All actual model runs require Slurm. New model artifacts and data use the storage
roots requested by Gabriele. Frozen base-model cache files are read in place.

## Text results

Qwen2.5-3B-Instruct, frozen base, answer-token/EOS cross-entropy; 270 training
examples across N8/16/32; K<=8. Test: 72 examples each at N8/32/64/128. Upper four
layers receive LoRA in every trained arm. Counts below are exact native outputs
at the pre-registered fixed epoch nine, with no intermediate reasoning tokens.
All P1/P2 cells reuse the same samples and one seed: these are exploratory results.

| Arm | Trained parameters | N8 | N32 | N64 | N128 | Pooled N64/128 |
|---|---:|---:|---:|---:|---:|---:|
| LoRA rank 8 | 409,600 | 61/72 | 29/72 | 25/72 | 15/72 | 27.8% |
| LoRA rank 16 | 819,200 | 64/72 | 29/72 | 22/72 | 16/72 | 26.4% |
| Original sum, blocks 64 | 802,880 | 69/72 | 41/72 | 31/72 | 23/72 | 37.5% |
| Centered sum, blocks 64 | 802,880 | 66/72 | 45/72 | 26/72 | 10/72 | 25.0% |
| Original sum, blocks 16 | 802,880 | 69/72 | 43/72 | 32/72 | 14/72 | 31.9% |
| Centered sum, blocks 16 | 802,880 | 67/72 | 28/72 | 8/72 | 0/72 | 5.6% |

The original sum gains 9.72 percentage points over rank-8 LoRA and 11.11 over the
approximately matched rank-16 control (the latter has 2.0% more parameters).
This supports further investigation of the operation rather than attributing the
entire fixed-epoch difference to parameter count. It is not independent replication
or a proof that the architecture improves information bandwidth.

Checkpoint selection matters. With the normal checkpoint selected using only
in-range development data, pooled long-N exact is 31.25% for original sum,
28.47% for rank-8 LoRA and 26.39% for rank-16 LoRA. The improvement is considerably
smaller. All these selections and fixed-epoch comparisons were specified before
their respective runs; neither should be silently substituted for the other.

Calibration and cost also matter. At fixed epoch nine, original sum's pooled
long-N first-answer-token NLL is about 2.72 versus 2.36 for rank-16 LoRA, despite
higher exact accuracy. Its batch-one generation latency on the long inputs is
about 1.5 times the LoRA control. One forward does not imply equal compute.

## What the diagnostics established

- P0's short three-epoch pilot missed all efficacy thresholds. The frozen base's
  strict exact score was zero because it continued after the number; post-hoc
  leading-number accuracy was 69.4/27.8/18.1/18.1% across N8/32/64/128. Trained
  controls in P0 parsed correctly, so their differences were not stopping effects.
- Longer training helped original sum: its fixed-epoch in-range exact rose from
  79/144 at epoch three to 110/144 at epoch nine. P1's duration prediction passed.
- Centering subtracts the local message network's zero-read response and adds no
  parameters. Its zero-read cancellation property holds, but it did not improve
  this benchmark. Smaller regions did not rescue extrapolation either. Both P1
  architectural predictions missed their thresholds.
- P1's ten-point long-N gate over rank-8 LoRA narrowly missed by one net example;
  this is a near miss, not proof of no benefit. P2's five-point gate over the
  approximately matched rank-16 control passed at the fixed ninth epoch.
- Centered 16-token blocks also became unstable in output format at long N:
  61/72 parsed at N64 and 0/72 at N128. The complete failure is retained in reports.

![All P1 fixed-epoch cells](../../outputs/native_aggregation/p1/fixed_epochs.png)

## Verification and vision integration

All 15 mathematical/decoder tests passed on CPU Slurm. They cover global-softmax
reconstruction, GQA and masks, partial regions, finite gradients, zero initialization,
future causality and full-prefill/cached-decoding agreement in Qwen2, Qwen3 and the
Qwen2.5-VL language decoder, plus centered-read tests. Existing core suites passed;
optional legacy comparisons requiring nnsight were skipped.

A Qwen2.5-VL-7B nf4 smoke used actual MMReD images and ordinary processor/model
calls. One optimization step, native cached generation, and checkpoint save/reload
passed. Peak training CUDA allocation was 11.08 GB. With two examples per split,
this establishes software integration only. No vision extrapolation or reasoning
performance experiment has been completed.

## Where the project stands

Keep the original uncentered branch as a research baseline. Do not promote the
centered or smaller-block revisions based on these data. The text gain needs
independent seeds and data, and the remaining accuracy is far below the previous
external bind-then-count reference. That reference remains intact.

Before claiming a more general aggregation method, isolate whether local reads
retain the required evidence, verify that the gain persists under practical
checkpoint selection and matched compute, and evaluate native MMReD Vision and
reasoning composition. Current evidence does not identify which remaining
bottleneck is responsible, so another architectural addition should follow a
specific diagnostic rather than an assumed explanation.

## Artifacts and execution

Protocols and append-only outcomes: `PREREG_AGG.md` (P0/P1/P2/V0).
Canonical runs and exact source/checkpoint provenance: `outputs/native_aggregation/INDEX.md`.
All predictions, fixed-epoch checkpoints, configurations and source snapshots are saved.

- P0 CPU 433097; GPU 433106 (3m53s).
- Centered validation CPU 433131 (15 tests); an earlier exact-cancellation roundoff
  failure in CPU 433127 was fixed before GPU work; dependent 433130 was canceled.
- P1 GPU 433133 (13m42s); analysis/figure CPU 433142.
- Vision staging CPU 433137; V0 GPU 433138 (35s).
- P2 GPU 433157; paired comparison `outputs/native_aggregation/p2/comparison.json`.

Checkpoints: `/mnt/ckpts/gabriele/gnn_transformer/native_aggregation` and
`/mnt/ckpts/gabriele/gnn_transformer/native_aggregation_vlm`.
Data: `/mnt/data/gabriele/gnn_transformer`.
No package installations, commits, changes to the existing method, or RESULTS.md
updates were made. All GPU and heavy CPU work used Slurm.

All four GPU jobs completed: total allocated GPU time was 20 minutes 7 seconds
(about 0.34 GPU-hours), including the vision smoke and final capacity control.
P2 took 1 minute 57 seconds. No jobs from this block remain running.

# Native aggregation implementation and pilot

2026-09-08. Design: `NATIVE_AGGREGATION_PROPOSAL.md`. Protocol and predictions:
`PREREG_AGG.md`, P0. This implementation is experimental; benchmark improvement
must be established from the saved native model outputs.

## Code

- `gnnformer/native_aggregation.py`: one trainable branch on a decoder attention
  module, ordinary SDPA retained, native Q/K/V and RoPE reused. The local messages
  enter after attention's output projection and before the decoder residual/MLP.
- `scripts/native_aggregation.py`: answer-only training, ordinary cached greedy
  generation, development-only checkpoint selection, source snapshots, and costs.
- `tests/test_native_aggregation.py`: mathematical and decoder integration checks.
- `slurm/native_aggregation_check.sbatch`: CPU validation and dataset staging.
- `slurm/native_aggregation_pilot.sbatch`: one-GPU smoke and sequential P0 arms.

CPU job **433097** passed all 13 new tests, including Qwen2, Qwen3 and Qwen2.5-VL
language decoder checks, plus the existing fencing, carrier-mask, scratchpad and
data suites. Optional legacy comparisons requiring `nnsight` were skipped. GPU
pilot job **433106** was submitted after this job completed successfully.

## Interface and limits

Freeze the backbone first, then call `attach_native_aggregation(model,
layer_index=18, block_size=64, rank=64)`. The returned module is registered on the
selected attention layer; its `state_dict()` holds only the branch parameters.
`module.remove()` restores the original attention forward. The initial supported
implementations are the pinned transformers 4.57 Qwen2, Qwen3, and Qwen2.5-VL text
attention classes using SDPA with full attention. Unsupported classes or sliding
attention fail explicitly.

The module's `mode` supports `all`, `off`, `prefill`, and `decode`. Only `all` is
used in the initial efficacy pilot. Per-mode interventions need their own checks
and controlled experiments before supporting claims about repeated use.

Blocks are anchored to absolute cache slots. Padding contents are masked, but
changing the amount of left padding can change region boundaries and therefore
predictions. The pilot uses unpadded batch size one. It does not claim invariance
to batch padding, frame boundaries, absolute position or length.

Query tiling limits temporary score tensors. During training, autograd still
saves local read activations across tiles. Measure actual training memory and
latency; this implementation does not claim the same cost as ordinary attention.

P0 uses a 393,280-parameter branch on Qwen2.5-3B and 409,600 upper-layer LoRA
parameters. The LoRA-only control has the same upper adaptation allowance but
fewer total parameters. Parameter- and compute-matched controls remain necessary
for a mechanism claim. Sum versus mean also changes gradient scale; a gain alone
would not establish why the method works.

## Storage and reproducibility

All GPU work and heavy CPU work must run through Slurm. Trainers require
`SLURM_JOB_ID` and CUDA. Data is staged under
`/mnt/data/gabriele/gnn_transformer`. New checkpoints are written only under
`/mnt/ckpts/gabriele/gnn_transformer/native_aggregation`. Existing base-model cache
files are read in place; no new base-model download or installation is required.

Reports are under `outputs/native_aggregation`. Each run records source hashes
and snapshots, model/package versions, sample IDs and gold histograms, selected
checkpoint, native generated tokens, latency, memory and metric denominators.

## Interpretation

At K<=8 the answer digit can be produced in one forward pass; generation allows
up to four output tokens to include termination. No intermediate reasoning tokens
are requested. The registered strict exact metric requires the whole decoded
output to be an integer. The raw-prompt base model sometimes writes a correct
number followed by an explanation; this is a formatting failure under that metric,
not evidence of an aggregation failure. Report numerical-prefix diagnostics
separately as post-hoc analyses and prioritize the equally supervised LoRA control.

A positive text result is a gate toward MMReD Vision and controlled reasoning
experiments. It does not establish those results by itself. If the registered
native-output or extrapolation criteria fail, diagnose the failed gate before
running a larger campaign with the same configuration.


## Initial results and diagnostic revision

P0 completed on one B200 in 3 minutes 53 seconds. Of 144 longer-input examples
(N64 and N128), LoRA answered 35 correctly, the sum branch 30, and mean merging
38. All trained outputs parsed correctly. The sum branch missed every registered
efficacy threshold; no extrapolation improvement is established. See
`outputs/native_aggregation/INDEX.md` and `p0_analysis.json` for the complete cells.

The P1 diagnostic tests optional centering:

    m(q, r) = SiLU(W_r r + W_q q + b) - SiLU(W_q q + b)

This subtracts an exact query-only term repeated by the original sum. Zero-valued
reads become no-ops without adding parameters. Nonmatching evidence need not have
a zero read, so this is a limited structural property, not a counting guarantee.

P1 crosses centering with block sizes 64 and 16 and records fixed epochs 3 and 9,
with the same LoRA control. It reuses P0 data and is exploratory. The registered
protocol requires reporting every cell and does not permit a larger unchanged
vision/reasoning campaign if no native extrapolation gain survives.

Final results and P2 capacity control: `NATIVE_AGGREGATION_PILOT_RESULTS.md`.
Validation now has 15 passing tests and a completed real-image VLM smoke.

# scripts/learnmask/

Code for the learned-discrete-mask campaign (mirrors outputs/learnmask/, see
outputs/learnmask/CAMPAIGN_BRIEF.md). Core machinery lives in gnnformer/learnmask.py
(relation cell map, differentiable mask assembly, MaskGates estimators E1-E4, gated
full-stack forward); parity is pinned bit-for-bit in tests/test_fencing.py +
tests/test_learnmask.py.

- `train_mask_gates.py` — trains gate logits (relation × layer) with DIRECT
  answer-class CE on MMReD-HF (scratchpad/TF metrics deprecated 2026-08-12 — the
  caption tf_acc is a copy detector); frozen backbone + frozen e_c/LoRA from the
  digit-readout ckpt `checkpoints/carrier_layer_digit_p7a_lora_best.pt`;
  `--arm s1|s2|s3`, `--estimator ste|soft|st-gumbel|hard-concrete`,
  `--target class|digit`. Emits the ep0 handfence reference row + fence-init row,
  startup engine-parity (kernel-exact), per-epoch gate heatmaps and hard-mask
  class-acc/CE/EM. Wrapper: slurm/train_mask_gates.sbatch.
- `p0_token_check.py` — CPU data check: tokens/frame at a given resolution + sequence
  geometry projections (verified 324 tok/frame @512, seq 2655 @N=8).
- `eval_mask_transfer.py` — zero-shot length transfer: assemble a trained gate table
  (or hand/init/nofence reference regimes) at arbitrary N and score emitted answers
  on the benchmark's native test splits (class read <=9, digit-sequence decode >9;
  device-side per-layer mask assembly for large N). Wrapper:
  slurm/eval_mask_transfer.sbatch + slurm/lib/roots_learnmask_transfer.txt.

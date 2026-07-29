# PRE-REGISTRATION — P4 readout-simplicity controls (p0p2 campaign)

Written: 2026-07-24 evening, BEFORE any P4 trainer. Bands fixed; never adjusted post-hoc.
Question: was the SCRATCHPAD necessary, or just the carriers, or just in-length data?

## Cells

- **P4.3 no-harm on the plain SFT adapter** (eval-only, first): `noharm_bench.py
  --peft-adapter outputs/ladder/image_longN/sft_control_le8_v2/20260720_191541_lora/adapter`,
  same 500 MME + 500 POPE protocol/seed as job 124508 (carrier-LoRA ref: MME −0.2 /
  POPE −1.4). **Band: |Δ| ≤ 2 pts per bench = no-harm GO.** Plus emission anatomy on ≤20
  failing items (predicted failure mode: digit-on-yes/no). New `--peft-adapter` +
  fail-dump code path; the carrier-ckpt path untouched.
- **P4.1 plain-LoRA SFT, in-length trained (the dangerous cell).** `lora_sft_baseline.py`
  config identical to `sft_control_le8_v2` (task steps_in_room, r8, target both, lr 2e-4)
  EXCEPT: data root = symlink farm `data/mmred_sft_p41/seq_len_{8,16,32,64}` →
  {mmred_images_park seq8, mmred_longN_park 16/32/64}, `--train-seq-lens 8,16,32,64`,
  EPOCHS 4 (best was ep1-3), FLASH wrap on the training forward (grad-ckpt already on via
  prepare_model_for_kbit_training). **Documented mismatches vs the caption trainer's
  mixture (script limits):** single-task steps only (script is one-task by design — no
  cooc/rooms/which/union); no mmred_longN_park2 32/48 (seq_len dir collision in the
  one-root convention); split machinery differs (declare_splits val/test fracs vs
  --split-seed permutation) so per-root counts differ; supervision = answer token only.
  Exams on the ARM-A dirs-files (identical dirs to the caption cells): N=32
  (`eval_dirs_N32all.txt`, LIMIT 150) + N=64 (`eval_dirs_N64.txt`, LIMIT 52) via the new
  `--eval-dirs-file` path (added 2026-07-24; same generate-and-parse scoring).
  Per-count histograms MANDATORY.
- **P4.2 carrier + DIGIT readout, in-length trained (middle rung).**
  `carrier_layer_cached.py` with the caption-winner data/recipe VERBATIM (16 roots,
  LIMIT 900, L12, r8, e_c, jitter, seed 0, split-seed 0, fixed criterion) but NO
  `--running-tally` / `--scratchpad-format poslist` → digit targets (supply via carriers,
  no scratchpad decode). Documented limitation: digit training path skips gold>9 samples
  (single-digit head) → the trainer sees in-length data only at golds ≤9; exams decode
  multi-digit greedily but the training never supervised >9. Exams: N=32 + N=64 on the
  ARM-A dirs-files (identical to the caption cells), DEC=8, per-count mandatory.

## Bands (both P4.1 and P4.2 exams, fixed now)

- **≥0.90 @N=32 → "simple fix wins"** — logged honestly; reframes (not kills) the thesis:
  the scratchpad was unnecessary once data is in-length.
- **≤0.50 @N=32 WITH dead mid-range** (per-count: extremes ≥2× mid-band) →
  theory-confirmed strongest-baseline row.
- Between → partial re-encoding; per-count anatomy is the deliverable.
- Reference cells: caption N=32 0.987 / N=64 0.981 (arm-A dirs); SFT-≤8 ladder
  0.480/0.350/0.220 (stratified-prefix protocol).

## Ops

Trainers on 4d_1g (2 slots; leaves 24h_1g + 4d_1g slots for the seeds/LOTO exam waves —
do-not-starve rule). P4.3 on 2h_2g. Walltimes: P4.1 ~2-4h/ep at mixed lengths → 22h wall;
P4.2 ≈ the caption trainer minus decode ≈ ≤14h.

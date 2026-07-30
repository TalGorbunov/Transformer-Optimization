# P0–P2 consolidation campaign STATE (overwrite at every transition)

Updated: 2026-07-24 ~20:40 (LOTO T1 landed + 6-cell battery launched w/ split-gate amendment; P3a L3 = extremes collapse (3/4 cells); P4.3 GO; P4.1/P4.2 training)

## One-screen summary (final deliverable — fills in as cells land)

| item | status / number | run dir(s) |
|---|---|---|
| **Format winner** | **DONE — C (caption)**; FINAL table: length means B 0.975 / C 0.980 / A 0.786 / D 0.734 (N=48: B 0.982, C 0.972); all 5 bands decided | winner ckpt `carrier_fmt_caption/20260722_222032_L12_r8/carrier_layer_best.pt` |
| **L\* curve** | **DONE — L12 wins, stands** (L8 0.277 · L10 0.373 · **L12 0.443** · L14 0.330 · L17 0.280 · L20 0.273 @N=32 0-shot) | `tallyL{8,10,14,20}_eval_N{32,64}/` (124965-72) |
| **Seeds mean±std (P1.1)** | **DONE — N=32 held-out 0.982 ± 0.007** (seed0 0.987 · seed1 0.987 · seed2 0.973; pf 0 ×3; identical 150 dirs; byte-identity gates PASSED) | `carrier_fmt_caption_seed{1,2}/` + `fmtCseed{1,2}_eval_N32heldout/` |
| **SFT N=64 cell (P1.3)** | **DONE — 0.220** (pf 0, MAE 3.46; test_iid re-anchor 1.0000); SFT ladder 0.480/0.350/0.220 | `sft_control_le8_v2_evalN64/20260723_225940_lora/` (125267) |
| **Measured before-ceiling (P1.2)** | **DONE — 0.317/0.281/0.189/0.183/0.122** @N=8-128 ≈ law-pred; fig regenerated | `measured_ceiling/20260723_222428/` (125259) |
| **LOTO verdict (P2a)** | T1 landed (0.998/tf-ex 0.993 @ep5, n=7872); **split-gate AMENDED** (split-seed pins the permutation, not the split, at different n — Arm 1 airtight: 0 cooc in training; Arm 2 dirs re-drawn from T1's own eval split); 6 exams running/queued (125500-505) | `carrier_loto_nococ/20260724_064756_L12_r8/` |
| **MLVU cell (P2b)** | **DONE — MCQ nearest-option 0.107 ≤ frozen 0.282 → pre-registered DOMAIN-GAP outcome (no transfer)**; open exact 0.000, 161/206 emit "0"; format survives, evidence detector domain-bound | `mlvu_ac/carrier_eval_N32/20260724_064754_L12_r8_evalonly/` (125350) |
| **P3a natural ladder** | **COMPLETE — supply GO (d′ 27.3) · scaffold GO (0.980±0.012) · in-model NO-GO (0.289/0.259 @N=8, 0.155/0.145 @N=16, pooled 0.218 < frozen; pure extremes)** — failure localized to the in-model rung (park e_c / LoRA integration); successor: natural-distilled e_c | `outputs/ladder/natural_mm/` (125486/88/87/92-95 + CPU) |
| **P4.1 SFT in-length** | **DONE — N=32 0.967 (pf 0) → "simple-fix-wins" band, honest log; N=64 extrap 0.787**; retires the ≤8 SFT ladder; surviving contrasts: P4.2 asymmetry, 5-task generality, h200-only training cost (5 attempts documented) | `sft_inlength_p41/20260725_031153_lora/` + `sft_inlength_p41_exams/` (125567/125620) |
| **P4.2 carrier+digit** | **DONE — N=32 0.333 · N=64 0.140, dead ≥g3-g4 → THEORY-CONFIRMED band** (in-dist 0.863; same carriers/data as caption 0.987/0.981 — readout expressivity is the separator); exam dirs rebuilt (split drift) | `carrier_digit_inlength/20260724_202048_L12_r8/` + `p42digit_eval_N{32,64}/` (125498, 125610/11) |
| **P4.3 SFT no-harm** | **DONE — MME −0.6 / POPE +1.2 pts → GO** (band ≤2; fail dumps clean yes/no, digit-failure prediction refuted) | `noharm_bench_sft/20260724_190307/` (125499) |

## Live jobs (2026-07-24 ~20:40)

125347/48 seeds trainers (~13.7h, landing now) · 125494 nat N16far exam · 125497 P4.1 SFT
trainer · 125498 P4.2 digit trainer · 125500-505 LOTO battery (3 running, 3 queued behind
a100 slots) · monitor active.
| **P3b InternVL tally** | **DONE — 0.938 ± 0.031 exact @L16 (band ≥0.90 MET, scaffold ports; multipass-isolated label; gate 0.991 vs digit-readout 0.586)** | `outputs/frame_axis/internvl/gate_tally/20260724_165356/` (CPU on 124280 cache) |

## Live jobs

| job | what | QOS/partition | ETA |
|---|---|---|---|
| 125347 | P1.1 seed1 trainer (caption, --seed 1 --split-seed 0) | 24h_1g a100 | ~14h |
| 125348 | P1.1 seed2 trainer (--seed 2) | 4d_1g a100 | ~14h |
| 125349 | P2a LOTO T1 writer (caption, 15 roots, no cooc) | 4d_1g a100 | ~13h |
| 125485 | P3a L0 frozen baseline ×4 natural eval roots | 2h_2g a100 | ~1h |
| 125486 | P3a L1 supply probe (natural v2 dist_far, n=300, blockfence+posreset+qfirst) | 24h_1g a100 | ~2.5h |
| 125487 | P3a L3 trainer (caption/L12 verbatim, 4 natural train roots, n=890) | 4d_1g a100 | ~4-6h |

Monitor: persistent squeue-diff watcher active.

## On-landing checklists

- **125485 (L0):** read 4 FROZEN BASELINE lines from the log → per-cell numbers → draft.
- **125486 (L1):** read report d′ (replica vs joint anchor, per-copy, adequacy) vs band
  (≥4.0 and ≥2× joint); then L2: `replica_gate_tally.py --cache <run>/messages_cache.pt
  --layer 16 --output <run>/gate_tally` (CPU, local or 4h_0g) vs band ≥0.85.
- **125487 (L3):** verify eval_dirs.txt ⊂ train roots only (image-half A); then 4 exams
  on EVAL roots via of_fmt_eval.sbatch (CKPT=<best>, DIRS_FILE per root — create
  dirs-files by ls, or pass root as dirs? of_fmt_eval needs DIRS_FILE: build
  eval_dirs_nat{8,16}_{far,near}.txt in the run dir), DEC=100 (N8) / 180 (N16),
  LIMIT=300, DUMPS=200. Bands: pooled ≥0.80 GO; ≤ L0+0.10 NO-GO.

- **125347/48 (seeds):** verify `eval_dirs.txt` byte-identical to arm A's
  (`carrier_tally_l12v2/20260721_071710_L12_r8/eval_dirs.txt`) — if not, BLOCKED, no exams.
  Then N=32 exams: of_fmt_eval.sbatch, CKPT=<seed ckpt>, DIRS_FILE=arm A
  `eval_dirs_N32all.txt`, DEC=320, LIMIT=150 (the 0.953-cell dirs). Report mean±std over
  {seed0 0.987, seed1, seed2}. Also record BEST/tf-exact per seed from trainer reports.
- **125349 (LOTO T1):** verify its eval_dirs.txt = arm A's minus cooc rows. Then exams
  (per PREREG): Arm 2 in-dist (dirs `eval_dirs_indist150.txt`, the non-cooc 120 items —
  eval consumes the full file; cooc rows score ~0 for a no-cooc model, so compute the
  4-task subset from per-task lines), Arm 1 cooc N=8 (`eval_dirs_cooc_all.txt`, LIMIT 300)
  + cooc N=32 (`data/mmred_cooc_longN/seq_len_32/all_uniform`, LIMIT 299, DEC 320),
  Arm 3 skyline (winner ckpt, same cells), Arm 4 frozen floor (same cells).
  QOS spread: 24h_1g remaining slots + 4d_1g.

## Landed this campaign (all drafted + INDEXed + migration-drafted)

| cell | number | run dir / job |
|---|---|---|
| E-H L20 N=32/N=64 + curve | 0.273/0.183; L12 peak stands | `tallyL20_eval_*` (124971/72) |
| fmt B N=32 / C N=32 / D N=32/48/64 | 1.000 / 0.987 / 0.907/0.679/0.615 | `fmt{B,C,D}_eval_*` (125194/95, 125185-87) |
| fmt B N=64 / C N=64 → **winner C** | 0.942 (0.956 cap-adj) / **0.981** | `fmt{B,C}_eval_N64heldout/` (125198/99) |
| P1.2 measured ceiling | 0.317/0.281/0.189/0.183/0.122 ≈ law | `measured_ceiling/20260723_222428/` (125259) |
| P1.3 SFT N=64 | **0.220**; ladder 0.480/0.350/0.220 | `sft_control_le8_v2_evalN64/…/` (125267) |

## Ops lessons

- generate+mask-None → enable_gqa → mem-efficient sdpa ineligible → MATH 17GB OOM;
  **FLASH handles GQA+causal 8.3GiB @12.7k** (125263). Fixed in lora_sft_baseline.py.
- 2h_2g per-user MEMORY cap (QOSMaxMemoryPerUser) — right-size --mem on smokes.
- Scan/caption evals on l40s: ~21 min/sample @N=64 dec620 → 18h for 52 dirs. Budget N=48
  pair ≈ 22-27h. a100 ≈ 2× faster (B/C N=32: 3.9 min/sample).
- Stale-STATE job→cell mappings are approximate — always identify cells from report
  headers (ckpt + dirs-file), not the job list.

## Code/infra changes (backward-compatible)

`lora_sft_baseline.py` (+`--eval-only-adapter`, FLASH) · `carrier_layer_cached.py`
(+`--split-seed`) · `carrier_layer_lora.py` (+`--alien-task`) · `of_carrier_fmt.sbatch`
(+SEED/SPLIT_SEED) · new: `convert_ac_to_mmred.py`, `p13_eff_smoke.py`, runners
`p12_measured_ceiling_cpu`, `p13_sft_evalN64`, `p13_eff_smoke`, `p2a_cooc_longn_gen`,
`p2a_loto_trainer`, `p2b_mlvu_convert`, `p2b_mlvu_eval`.

## PREREGs

- P2a: `plans/scratchpad_loto_PREREG.md` (written 2026-07-23, pre-GPU; format slot now
  resolved = caption).
- P2b (fixed 2026-07-23): 32f arm (dense prohibitive — documented); frozen ref 0.282 MCQ;
  band >0.332 GO / ≤0.282 domain gap / between partial. MCQ rule: greedy decode → caption
  parser → emitted count → NEAREST numeric option (ties → smaller); parse-fail = wrong;
  also acc/pf/MAE vs gold count.

## Blockers

None.

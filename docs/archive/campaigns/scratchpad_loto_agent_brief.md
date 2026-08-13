# Agent brief — Scratchpad task-agnosticism: leave-one-task-out (LOTO) zero-shot transfer

**Date issued:** 2026-07-22. **Approved by:** Tal (via Claude session).
**Mission:** measure whether the tally-scratchpad readout generalizes across TASK TYPES, or
whether the trained carrier layer memorized its training question templates.

## 0. Read first (in this order — do not skip)

1. `CLAUDE.md` §3 (SLURM/QOS etiquette), §5 (how to work), §7 (never-do).
2. `plans/carrier_stage4_STATE.md` — current campaign state. Note: jobs 124907/124924
   (N=128) were CANCELLED, queue is empty, E-H exams 124965–72 COMPLETED but uncollected.
3. `plans/carrier_stage2_DRAFT_RESULTS.md` — the campaign draft log (append here, NEVER
   RESULTS.md).
4. `outputs/ladder/INDEX.md` (or `outputs/ladder/image_longN/` INDEX) — canonical run paths.

Key background facts (verified in-session 2026-07-22):
- Campaign-best readout recipe = **l12v2**: L_OPEN=12, LoRA r8, running-tally scratchpad
  format, variable-N (≤64) training data, **fixed save criterion (acc, tf-exact)**.
  Ladder: N=32 held-out 0.953 (pf 0) · N=48 0.878 · N=64 0.678 (cap-adj).
- E-C strong form (question-blind carrier) = NO-GO (d′ 2.40). Question must lead the layout
  (E-C(b) GO). Do NOT test question-blind encoding here; agnosticism = readout-side only.
- Budget override (Tal, permanent): GPU-hours unconstrained; etiquette rules unchanged.

## 1. Phase 0 — collect free results, resolve reusables (CPU only, no new GPU jobs)

- **P0.1** Collect E-H L* sweep exams (jobs 124965–124972; logs in `logs/cl_eval-<id>.out`,
  reports under `outputs/ladder/image_longN/`). Assemble the 6-point L* curve with the
  L12 (0.443) and L17 (0.280) reference cells per stage4 STATE "Next". Log to draft + INDEX.
  **Decision rule:** if a non-L12 arm beats L12's zero-shot N=32 cell by >0.05 with pf≤0.02,
  use that L* below; otherwise L_OPEN=12 stands.
- **P0.2** Resolve from run configs (`config.json`/`run_config.json` in the run dirs; do not
  guess): (a) the exact 5 task names of the scratchpad5 mixture
  (`carrier_layer_scratchpad5/20260719_031356_L17_r8/` and the l12v2 trainer run from job
  124773); (b) the l12v2 checkpoint path (`carrier_layer_best.pt`); (c) which caches
  `carrier_layer_cached.py` can reuse vs regenerate; (d) the frozen carrier ckpt passed as
  `CKPT` (distilled e_c) used by the 124918-class jobs; (e) the eval `--dirs-file` splits
  used by the l12v2 exams (airtight split discipline — reuse, don't re-draw).
- **P0.3** Frozen floor: check whether `experiments/glstm/frozen_baseline_eval.py` already
  has a cell for the held-out task at N=8/N=32; if not, it's one cheap eval arm below.

## 2. Phase 1 — PRE-REGISTER (before any GPU job)

Write `plans/scratchpad_loto_PREREG.md` containing, filled with the Phase-0 resolutions:
- Held-out task: **co-occupancy** if it is one of the 5 (same per-frame facts as steps,
  different aggregation); else pick the mixture task whose aggregation differs most from
  per-(C,R) counting. Record the choice + reason before launch.
- Arms and bands (fixed now, verbatim):
  - **Arm 2 in-dist sanity** (LOTO ckpt, 4 trained tasks, held-out samples): band ≥0.90,
    else trainer broken → STOP, write BLOCKED, do not run Arm 1.
  - **Arm 1 LOTO zero-shot** (held-out task, N=8 and N=32, same dirs-files as skyline):
    GO if ≥0.7 × Arm 3 on the same cells; NO-GO if ≤ frozen floor + 0.10.
    Between: log honestly as "partial transfer", no spin.
  - **Arm 3 skyline** (all-5 ckpt on the same held-out-task items): reuse the existing
    l12v2/scratchpad5 ckpt ONLY if recipe-matched (L*, format, criterion, data mix) —
    else retrain with the identical recipe. Record which.
  - **Arm 4 floor** = frozen baseline on the same items.
  - **Arm 5 (bonus, eval-only, cheap):** LOTO ckpt on OR-union composition
    (`data/mmred_union_or/seq_len_8/all_uniform`, gen via `experiments/glstm/union_gen.py`
    if absent) — never in ANY mixture; compare to the logged scratchpad5 union-0shot 0.321.
- Count-OOD note: keep gold>9 allowed in training (stage3 lesson), tally format mandatory.

## 3. Phase 2 — launch (only after PREREG is written)

Trainers (`runners/of_carrier_cached.sbatch`, env-var driven — CKPT, DATA_ROOT, LIMIT,
EPOCHS, L_OPEN, OUTPUT, EXTRA_FLAGS):
- **T1 LOTO writer:** 4-of-5 mixture via comma-separated `--data_root` (NOTE the sbatch
  `--export` comma-truncation gotcha — pass DATA_ROOT with commas via an env file or wrap
  script, NEVER inline in `--export`). l12v2 recipe, EPOCHS per the l12v2 trainer config,
  fixed save criterion. ~4–6 h, 1 GPU.
- **T2 skyline retrain** only if P0.2 says the existing ckpt is not recipe-matched.
- **Smoke first:** 2-sample, `EPOCHS=1`, on `2h_2g`, output under `outputs/_scratch/` —
  verify `hooks_ok`/`nonzero_gates`/tf loss falls, THEN full runs.

Evals (`carrier_layer_lora.py --eval-only` tooling / the cl_eval runner family used by
124965-72 — copy their flag sets): Arm 2, Arm 1 (N=8 LIMIT 300; N=32 LIMIT 300, dec 240),
Arm 3/4 cells missing after reuse, Arm 5. Exams launch ONLY after their trainer shows
COMPLETED (stage4 ops lesson). Reports write at END — never let walltime kill an eval;
size time limits generously (N=32 exam ≈ 6–7 h at LIMIT 300 per the 124965-class).

SLURM etiquette (mandatory):
- Before EVERY submit: check free GPUs across ALL partitions
  (`sinfo -N -O "Partition:16,NodeHost:12,Gres:26,GresUsed:30,StateLong" -p l40s-shared,h200-shared,rtx6k-shared,a100-public,l40s-public`).
  Submit where there's room; never queue behind a full partition when another is idle.
- Right-size QOS; spread to beat the 3-per-QOS cap: trainers `12h_4g`, evals split across
  `24h_1g` (4 slots) and `4d_1g` (8 slots). Smokes `2h_2g`. CPU work `4h_0g` (mem cap 16G).
- a100-public is 40 GB (fine for all jobs here; nothing at N=128 in this brief).
- rtx6k-shared works (venv fixed 2026-06-23) — overflow only.

## 4. Phase 3 — collect, log, close

- For each landed cell: read the run README/summary + parse-fail; append a draft entry
  (format: what/run-path/number/verdict/caveats) to `plans/carrier_stage2_DRAFT_RESULTS.md`;
  add INDEX rows for canonical cells.
- Maintain `plans/scratchpad_loto_STATE.md` (overwrite-at-transition style, like stage4).
- Verdict paragraph against the PREREG bands — cite exact run dirs. If bands conflict with
  the numbers, the numbers win and the band is reported as refuted; never adjust bands
  post-hoc.
- Blockers → `plans/scratchpad_loto_BLOCKED.md` with a successor-action, then continue with
  whatever arms are unblocked.
- Do NOT edit `RESULTS.md` (Tal logs it explicitly). Do not cancel or resubmit jobs you did
  not launch.

## 5. Hard rules

Never: pip/conda install; delete/overwrite anything under `outputs*/`, `output_*/`, `data/`;
heavy compute on the login node; unverified SLURM flags; a QOS bigger than the job.
Poll cadence: `squeue -u $USER` every ~20–30 min; between polls, prep the next phase's
commands. Every number you log must trace to a run dir on disk.

# Stage-3 STATE (overwrite at every phase transition)

Updated: 2026-07-19 ~15:30 (SESSION END — program complete except N=128 cells)

## Landed this phase (all in carrier_stage2_DRAFT_RESULTS.md + outputs/ladder/INDEX.md)

- **P1 FINAL 0.999** (123741) — `carrier_layer_pooled/20260718_182248_L17_r8/` — in-model
  3-task ≥ scaffold 0.998. THE headline.
- **A3 scratchpad+jitter** (124282, cancelled post-convergence) — TF-count 1.000@ep1;
  ckpt `carrier_layer_scratchpad/20260719_005342_L17_r8/carrier_layer_best.pt`.
  Exams landed: **in-dist greedy 0.953 / parse-fail 0** (124314); NIAH-0shot 0.087 (124316);
  union-0shot **0.321 partial compositionality** (124335).
- **5-task mixture** (124336, RUNNING ~ep2) — TF-count 1.000 on ALL 5 tasks @ep1 (NIAH
  118/118, union 78/78) — "easy once in mixture" confirmed at TF level.
  Ckpt `carrier_layer_scratchpad5/20260719_031356_L17_r8/`. Cancel after ep2 logged.
- **Track B** (124280): InternVL solo-qfirst d′ 6.31/5.11 vs joint 1.79/1.90 — supply
  mechanism PORTS (3.5×); Q-first amplifier does NOT port (plain solo 6.38/6.56).
- **C1/C2 battery** (124300-06): L12 0.941 > r16 0.731 > base 0.698 ≈ r4 0.694 >
  noposreset 0.669 > L22 0.513 > noqfirst 0.378.
- Cached-ckpt exams (earlier): N32 0.097 collapse; rooms-0shot 0.153; drift 0.313 (safe).
- C3 data: `data/mmred_niah_which/` (720) + `data/mmred_union_or/` (540) generated+verified.

## Since last update (all logged + INDEXed)

- A3 N=32 zero-shot **0.215** (in-range 0.311, parse-fail 0) → A4 triggered per bands.
- P1-ckpt: N=32 **0.092** (variable-N digit = same collapse), NIAH 0.117, union 0.150.
- 5-task mixture greedy **0.966** (NIAH 0.992 GO, union 0.910 GO) — 124349.
- **A4 trained** (124362): +longN16(330)+longN32(first-200), grad-ckpt added (2 smoke fixes:
  OOM → --grad-ckpt; sdpa-backend-in-checkpoint mismatch). TF 0.997 @ep3.
  Ckpt: `carrier_layer_scratchpad_longN/20260719_054023_L17_r8/carrier_layer_best.pt`.
- FULL-THESIS TABLE written at end of draft (rows 9/19 pending).

## Running / pending jobs

| job | what |
|---|---|
| 124376 | A4 exam N=32 HELD-OUT complement (190 dirs; ~48s/sample ≈ 2.5h) — decisive |
| 124377 | A4 exam N=128 (h200 PENDING) |
| 124317 | A3 exam N=128 zero-shot (h200 PENDING) |

## Session complete

- 124376 LANDED: A4 held-out N=32 **0.447 (in-range 0.626, parse-fail 0)** — partial band,
  long-N data curve steep. Row 9 filled; one-screen summary written at end of draft.
- ONLY remaining: 124317 (A3-ckpt N=128 zero-shot) + 124377 (A4-ckpt N=128) queued on the
  saturated h200 — they will run unattended and write reports to
  `scratchpad_eval_N128/` and `scratchpadLN_eval_N128/`; successor: read
  logs/cl_eval-{124317,124377}.out, log to draft + INDEX, fill table rows 9/19.
- Budget: Phase-3 ≈ 30 GPU-h (≤35 ✓).

## Budget

Phase-3 spent ≈ 16-18 GPU-h (A3 3, mixture5 ~3, C 7×1, B 1, exams ~4). Remaining exams ~4-5h.

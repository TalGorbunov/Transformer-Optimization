# PRE-REGISTRATION — Scratchpad LOTO (leave-one-task-out) zero-shot transfer

Written: 2026-07-23, BEFORE any LOTO GPU job (p0p2 campaign P2a; brief:
`plans/scratchpad_loto_agent_brief.md` with the two p0p2 amendments). Bands fixed here,
never adjusted post-hoc; between-band results logged honestly as partial.

## Phase-0 resolutions (from P0 of the p0p2 campaign — no re-derivation)

- **Recipe** = the l12v2 recipe verbatim (`plans/scratchpad_format_PREREG.md` "Resolved
  recipe"): `carrier_layer_cached.py --running-tally --jitter-gap 16 --grad-ckpt
  --carrier-ckpt outputs/ladder/image_longN/carrier_token/20260718_130545_distill_room_k1/carrier_best.pt
  --limit 900 --epochs 5 --l-open 12` + defaults, fixed save criterion (acc, tf-exact),
  mem 200G, a100-class GPU. **L\* = 12** (E-H curve complete, `tallyL{8,10,14,20}_eval_*`:
  no arm beats L12's 0.443 → decision rule keeps L12).
- **Format** = the P0.1 format-sweep winner (B/scan leading at time of writing — N=32
  1.000 vs A 0.953; final name pends jobs 125196-99). Whatever wins is used for BOTH the
  LOTO writer (T1) and the skyline (Arm 3), so the contrast is format-matched. Deviation
  from the logged l12v2 poslist reference cells will be stated wherever cited.
- **The 5 tasks of the mixture** (16 data roots): steps (park seq2-8 + longN 16/32/64 +
  park2 32/48), cooc (`mmred_cooc_balanced/seq_len_8`), rooms (`mmred_rooms_balanced/seq_len_8`),
  which (`mmred_niah_which/seq_len_8`), union (`mmred_union_or/seq_len_8`).
- **Held-out task: co-occupancy** (per the brief's first preference: same per-frame facts
  as steps, different aggregation — a relational pairwise predicate).
- **Split discipline:** T1 uses `--split-seed 0` (new flag, 2026-07-23, pins the arm-A
  train/eval permutation) so its train set is a subset of arm A's train dirs and every
  arm-A eval dirs-file remains held-out. Gate: T1's `eval_dirs.txt` (restricted to the 15
  shared roots) must match arm A's minus cooc; else BLOCKED.
- **Eval items (identical dirs for Arms 1/3/4):**
  - cooc N=8: the 432 cooc dirs of arm A's `eval_dirs.txt` (never trained on by ANY arm),
    LIMIT 300 → dirs-file `eval_dirs_cooc_all.txt` written into the arm-A run dir
    (same convention as the other shared dirs-files).
  - cooc N=32: `data/mmred_cooc_longN/seq_len_32/all_uniform` (NEW eval-only data, gen job
    125261, per-count 23 × counts {0..8,12,16,24,32}, n=299, seed 7, same generator +
    defaults as cooc_balanced). No model ever trains on it.
- **Frozen floor (Arm 4):** `frozen_baseline_eval.py` on the same items; the logged frozen
  N=8 full-prior reference is 0.219 (steps; `frozen_baseline/20260718_125303/`) — the cooc
  floor is measured fresh here, not assumed.

## Amendments to the brief (stated before launch)

1. **Trainers on `24h_1g`/`4d_1g`, not 12h_4g** (arm-A elapsed 13h47 — measured; p0p2
   amendment 1).
2. **Arm 5 (union bonus) DROPPED:** the brief's premise ("OR-union … never in ANY
   mixture") is false for the resolved l12v2 recipe — `mmred_union_or` IS a training root.
   No honest "never-trained composition" cell exists in this design; noted, not replaced.
3. **Arm 1 N=32 is doubly zero-shot** (task AND length: cooc training data is N=8-only).
   The skyline Arm 3 is equally length-zero-shot on this cell (its cooc data is also
   N=8-only), so the contrast stays matched; both are labeled cooc@N32-0shot.
4. Count-OOD: gold>9 allowed everywhere (stage-3 lesson); tally/winner format mandatory.

## Arms and bands (fixed now, verbatim from the brief where unchanged)

- **T1 LOTO writer:** 4-of-5 mixture = the 16 roots MINUS `mmred_cooc_balanced` (15 roots,
  n = 8772 − 900×0.5-split… concretely LIMIT=900 over 15 roots; n reported from the run
  header). Recipe/format/L* as above, `--seed 0 --split-seed 0`.
- **Arm 2 in-dist sanity** (T1 ckpt, 4 trained tasks, arm-A held-out samples — the
  `eval_dirs_indist150.txt` file minus its 30 cooc rows = 120 items): band **≥0.90**, else
  trainer broken → STOP, BLOCKED, no Arm 1.
- **Arm 1 LOTO zero-shot** (T1 ckpt, held-out cooc): N=8 (LIMIT 300 of
  `eval_dirs_cooc_all.txt`) and N=32 (LIMIT 299, `mmred_cooc_longN`).
  **GO if Arm1 ≥ 0.7 × Arm3 on the same cells; NO-GO if Arm1 ≤ Arm4 + 0.10; between =
  partial transfer, logged without spin.**
- **Arm 3 skyline** = the P0.1 winning ckpt (all-5, recipe/format-matched by construction)
  on the identical items. Its cooc N=8 reference exists only at in-dist-150 granularity
  (30/30) — the full 300-item cells are run fresh here.
- **Arm 4 floor** = frozen baseline, same items.
- Priors-context reference cells (not bands): in-model zero-shot task transfer without
  mixture training was ~absent (steps→cooc 0.179, `carrier_layer_eval_cooc0shot/`;
  cached→rooms 0.153, `cached_eval_rooms0shot/`) — LOTO asks whether 4-task VARIETY
  builds a task-general readout that one task alone did not.

## Report per cell

acc, parse-fail, MAE, mean decode tokens, run dir + job id. Verdict paragraph against the
bands; numbers win over bands; bands reported as refuted if they conflict.

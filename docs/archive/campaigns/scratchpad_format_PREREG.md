# PRE-REGISTRATION — Scratchpad FORMAT sweep (arms A–D)

Written: 2026-07-22, Phase 1, BEFORE any GPU job of this campaign. Bands are fixed here and
never adjusted post-hoc; between-band results get logged honestly as partial.
Brief: `plans/scratchpad_format_agent_brief.md`. State: `plans/scratchpad_format_STATE.md`.

## Resolved recipe (identical for every trained arm; ONLY the gold text differs)

`experiments/glstm/carrier_layer_cached.py` with: `--running-tally --jitter-gap 16
--grad-ckpt --carrier-ckpt outputs/ladder/image_longN/carrier_token/20260718_130545_distill_room_k1/carrier_best.pt
--limit 900 --epochs 5 --l-open 12` + defaults (train-frac 0.5, rank 8, alpha 16,
lr-lora 1e-4, resize 392, seed 0, shuffle-dirs 0), 16 data roots of the l12v2 mixture,
fixed save criterion (acc, tf-exact) — resolved from the l12v2 run's `report.txt` header +
`logs/carrier_cached-124773.out` + sacct (no config.json exists in that run dir; n=8772
reproduces exactly from LIMIT=900 over the 16 roots). Submit facts from arm A: a100-public,
**mem 200G (MaxRSS was 174G — runner default 120G would OOM)**, elapsed 13h47
→ trainers go to 24h_1g (wall 22h) or 4d_1g, NOT 12h_4g (deviation from brief §4: arm A's
measured 13h47 exceeds the 12h wall).

New arms differ ONLY in `--scratchpad-format {scan,caption,chunked}` (arm A = `poslist`,
existing ckpt reused, NOT retrained).

## Corrections to the brief (stated before launch)

1. **N=64 is IN the training mixture** (`mmred_longN_park/seq_len_64`); the brief's "OOD
   (N=64 for B, trained ≤48)" is inaccurate for the resolved recipe. The N=64 cell is
   held-out-in-length on the identical 52 dirs as arm A's 0.615 cell. True OOD (N=128) is
   out of scope here.
2. Arm A's logged N=64 cap-adj "0.678 (40/59)" is not derivable from its run report
   (per-count sums give 32 hits / 45 non-truncated = 0.711). This prereg fixes a transparent
   cap-adj formula (below) and recomputes arm A from its on-disk report: **A@N=64 raw 0.615,
   cap-adj 0.711; A@N=48 raw 0.789, cap-adj 0.878 (replicates); A@N=32 0.953 (no cap issue).**
3. Arm A additionally gets two cheap eval-only cells (in-dist-150, rooms-100, shared
   dirs-files) so the in-dist and rooms contrasts are same-ckpt. This is evaluation of the
   frozen ckpt, not retraining.

## Format specs (implemented + CPU-verified, 27 samples × 4 arms round-trip/parse/tally OK)

- **A poslist** (control, unchanged): ` frames 2 (1), 5 (2) -> 2` / rooms ` rooms Kitchen (1), Garden (2) -> 2`.
- **B scan**: ` scan: f1:- f2:yes(1) … f8:- | total: 2 END` — every frame a slot in frame
  order, inline tally, explicit END. **Rooms task carries room words in B too** (bare yes
  cannot express distinct-room counting; tally increments on FIRST visit only) — B≡C on rooms.
- **C caption**: as B with room-word slots everywhere: `f2:Kitchen(1)` (room words
  capitalized exactly as in states/questions).
- **D chunked**: 16-frame blocks, global 1-indexed positive lists, ` c1: 2, 5 | sub 2 c2:
  none | sub 0 total: 2+0 = 2 END`; rooms lists NEW rooms per block; `which` reads out the
  frame number (`total: 4 END`, no sum — same convention in B/C: total slot = the answer).
- Parser: poslist keeps last-`->` (backward-compatible); scan/caption/chunked anchor on the
  LAST `total:`, last integer before `END` (or end-of-text). Parse-fail = no such integer.

## Measured token costs → decode budgets (fixed)

Worst case (all-evidence, longest room word): scan/caption N32/48/64 = 279/423/567; chunked
= 146/220/294; poslist = 211/323/435. Budgets (≈ worst + 10%):

| cell (dirs-file, LIMIT) | A (poslist) | B/C (scan/caption) | D (chunked) |
|---|---|---|---|
| in-dist (`eval_dirs_indist150.txt`, 150) | 100 | 100 | 80 |
| rooms (`eval_dirs_rooms100.txt`, 100) | 100 | 100 | 80 |
| N=32 (`eval_dirs_N32all.txt`, 150) | 240 (done, 124904) | 320 | 170 |
| N=48 (`eval_dirs_N48.txt`, 109) | 280 (done, 124905) | 470 | 250 |
| N=64 (`eval_dirs_N64.txt`, 52) | 280 (done, 124906) | 620 | 330 |

All dirs-files live in the arm-A run dir; identical files + LIMIT reproduce arm A's exact
dir sets (dirs-file iteration is order-deterministic; skips are arm-independent).

**Cap-adjustment (fixed formula):** cap-adj acc = hits / (n − #dirs with gold ∈ {48, 64}),
computed from the per-count table of each report — identical exclusion set for every arm
(g48/g64 classes at N=48/64), so arm A's historical 280-token cap cannot advantage the new
arms' bigger budgets in the primary contrast.

## Pre-registered bands (primary contrasts)

1. **Per-arm in-dist sanity:** in-dist-150 greedy ≥ 0.90, else that arm is BLOCKED for
   contrasts (logged, no GPU beyond its already-launched cells).
2. **B vs A, length (N=64, identical 52 dirs):** primary = cap-adj; **B ≥ A + 0.05 →
   B ≥ 0.761 → scan-format GO.** Secondary raw: B ≥ 0.665.
3. **C vs B (agnosticism bet):** |C − B| ≤ 0.03 on BOTH in-dist-150 and rooms-100 →
   agnostic-caption GO; C < B − 0.10 on either → supply-limited, logged as
   carrier-bandwidth evidence.
4. **D vs A/B at N=64:** D cap-adj > max(A, B cap-adj) → chunking GO for the long regime.
5. **Rooms ordering:** B or C rooms-100 greedy ≥ 0.95 (arm A same-ckpt rooms-100 cell is
   the concurrent control; historical L17 reference 0.84–0.85) → ordering-fix confirmed.
6. N=32/N=48 cells are secondary/descriptive: within 0.03 of A = parity; > A + 0.05 = better.

Every cell reports: acc, parse-fail, MAE, mean decode tokens (new metric in the eval
report), run dir + job id. TF in-dist training metrics are NOT verdict metrics (they
saturate); greedy exams decide.

## Launch plan (Phase 2)

Smokes first (LIMIT=2/root, EPOCHS=1, 2h_2g, `outputs/_scratch/fmt_smoke_<arm>/`): verify
[target-debug] roundtrip ok, loss falls, tf metrics present. Then 3 trainers
(`outputs/ladder/image_longN/carrier_fmt_{scan,caption,chunked}/<ts>_L12_r8/`), free-GPU
check across ALL partitions before every submit, QOS spread 24h_1g/4d_1g. **Gate before any
arm's exams: its trainer COMPLETED + its `eval_dirs.txt` byte-identical to arm A's** (same
seed/order ⇒ same split; if not → BLOCKED.md, no exams). Evals per arm: the 5 cells above
(`fmt{B,C,D}_eval_*` siblings + `tallyL12v2_eval_{indist,rooms}` for arm A), each with
`--dump-decodes 200` so transcripts stream to the log (walltime-kill recovery).

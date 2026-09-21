# LORAMECH — campaign state (append-only, newest LAST)

Brief: `outputs/loramech/CAMPAIGN_BRIEF.md` (DRAFT v2). Scripts: `scripts/loramech/`.
Draft RESULTS entries go at the BOTTOM of this file — RESULTS.md is Tal's.

---

## [2026-08-26] CAMPAIGN START (Claude)

Authorization (Tal, 2026-08-26): GPU submissions pre-authorized; discipline rules binding
(smokes in `outputs/_scratch/`, explicit `--time` on every submit, no comma `--export`,
no installs, `gnnformer/` untouched, stop on artifact suspicion). Regime decision:
**FRAMES-FIRST everywhere** — Q-first deprecated for baselines/instruments/new training.
H200 fallback authorized: if h200 busy, ff_le32 may go to l40s.

Order of work (from Tal's start message): (1) this file + `scripts/loramech/`;
(2) `--frames-first` trainer delta (Q-first `build_messages` byte-untouched) + splits
redrawn with P4 contamination discipline + exam dirs files from provably-unseen dirs;
(3) trainer smoke (small LIMIT, 1 ep, `outputs/_scratch/`, 48 GB GPU); (4) P0 submits —
ff_le8 (~3.5 h, 48 GB) and ff_le32 (H200 preferred, `--time=8:00:00`, the long pole,
queue immediately); adapters → `checkpoints/sft_ff_le{8,32}_adapter/` + README row;
(5) exams N=8/16/32 in-length + N=64/128 zero-shot, per-count table, pf, class dist,
majority baseline; L6 rescore from CSVs; (6) H0 verdict here, then **STOP for Tal's
review** before L1–L5 (script deltas for L1–L5 may be PREPARED, not run, while waiting).

Context re-read at start: CAMPAIGN_BRIEF v2, CLAUDE.md §3/§5/§7, RESULTS [2026-07-20] E-B,
[2026-07-25] P4, [2026-08-22] ARMOR A, `scripts/armor/probe_hahn.py`,
`scripts/train_sft_baseline.py`, `checkpoints/README.md`.

### Recipe facts pinned (from the anchor adapters' configs, read today)

- E-B v2 (`sft_control_le8_v2_adapter`) and P4.1 (`sft_inlength_p41_adapter`) are both
  **r=8, alpha=32, target=both (q/k/v/o + gate/up/down)** — trained with the LEGACY
  script pre-refactor. The refactored `train_sft_baseline.py` defaults to r=16, so the
  ff trainers will pass `--lora-r 8` explicitly to mirror the 23.8M-param anchor recipe.
- P4.1 trained on the FULL pools: seq8 900 + longN seq16 330 + seq32 390 = 1620
  (split 1136/242/242). Its clean N=32 exam came from `mmred_longN_park2/seq_len_32`
  (fresh generation, 312 dirs) — same route reused here.

### Data / contamination plan (P4 discipline, by construction not post-hoc)

Pools on disk (counted today): `mmred_images_park/seq_len_8/all_uniform` 900;
`mmred_longN_park/seq_len_{16,32,64,128}/all_uniform` 330/390/450/510;
`mmred_longN_park2/seq_len_32/all_uniform` 312. Class tags run the FULL range
(K0..K8 + K12,K16,…,K=N), 30/class (park2: 24/class).

- Trainer gets a new `--exclude-dirs-file` (repeatable): dirs listed there are skipped
  during data collection → exam sets are provably unseen **by construction**; the run
  dir still writes `train_dirs.txt` for a post-hoc disjointness assert.
- Exam files (built by `scripts/loramech/build_exam_dirs.py`, seed-1 stratified draw,
  validity-gated, class dist reported):
  - `exam_ff_N8.txt` — 150 from seq8 pool; EXCLUDED from both trainers.
  - `exam_ff_N16.txt` — 150 from longN seq16; EXCLUDED from ff_le32 (ff_le8 never sees 16).
  - `exam_ff_N32.txt` — 150 from park2 seq32 (disjoint generation; no exclusion needed;
    ff_le32 trains on all 390 longN seq32).
  - `exam_ff_N64.txt` / `exam_ff_N128.txt` — 150 each from longN seq64/128 (nothing is
    trained at 64/128 — clean for both adapters).
- Train roots: ff_le8 = seq8 minus exam_N8 (~750 → ~525 train after 70/15/15 split);
  ff_le32 = seq8 minus exam_N8 + seq16 minus exam_N16 (180) + seq32 full (390).

### Trainer delta (scripts/train_sft_baseline.py — Q-first path untouched)

`--frames-first`: new message builder = the `build_count_prompt` layout (images, then
question text, then "Answer: ") — token-identical to `gnnformer.data.build_prompt_inputs`,
i.e. the frozen-baseline / ARMOR-A plain-arm prompt. Parity is asserted by
`scripts/loramech/check_ff_template.py` (CPU) before any GPU run. Also added:
`--exclude-dirs-file`, per-sample predictions CSV on the dirs-file eval path (needed for
L6 per-count rescoring), and a `config.json` dump in the run dir.

## [2026-08-26 ~16:10] TRAINER DELTA DONE + EXAM DIRS BUILT + SMOKE & ff_le32 SUBMITTED

- `--frames-first` added to `scripts/train_sft_baseline.py` (+ `--exclude-dirs-file`,
  per-sample `longn_predictions.csv` on the dirs-file eval path, `config.json` dump).
  Q-first `build_messages` untouched. **Parity gate PASSED** (CPU,
  `scripts/loramech/check_ff_template.py`): frames-first template token-identical to
  `build_prompt_inputs`+`build_count_prompt` at N=2 (464 toks) and N=8 (1652 toks).
- Exam dirs files built (`scripts/loramech/build_exam_dirs.py`, seed-1 stratified,
  validity-gated, 0 invalid) → `outputs/loramech/examdirs/exam_ff_N{8,16,32,64,128}.txt`,
  150 each, class-balanced over the FULL gold range (see `examdirs/REPORT.txt`).
  N8/N16 will be trainer-EXCLUDED; N32 = park2 disjoint generation; 64/128 untrained.
- Wrapper `slurm/train_sft_baseline.sbatch` extended: FRAMES_FIRST/EXCLUDES/LORA_R/
  ROOTS_FILE knobs (roots file = `slurm/lib/roots_loramech_le32.txt`; no comma --export).
  DRY_RUN checked for both configs before submitting.
- Cluster at submit time: ALL partitions saturated except 1 free rtx6k GPU (n318).
- **Submitted:** `lm_smoke` 137464 (rtx6k-shared, 2h_2g, --time=1:00:00, mem 48G;
  LIMIT=40 EPOCHS=1 r=8 ff, excl+exam = exam_ff_N8, LONGN_LIMIT=12,
  → `outputs/_scratch/loramech_smoke`) and `lm_ff_le32` 137465 (h200-shared, 24h_1g,
  --time=8:00:00; full recipe EPOCHS=5 r=8 ff, roots seq8+longN16+longN32, excl N8+N16,
  in-job exams = in-length N8/16/32 only → H0 lands with the trainer;
  → `outputs/loramech/p0_ff_le32`). Queued immediately per Tal (long pole); if the
  smoke fails, scancel + fix + resubmit before it can start.
- Zero-shot exams (N=64/128, both adapters) deliberately NOT in the H200 job — they run
  as separate `--eval-only-adapter` jobs on a 48/40 GB GPU (H200 hours are the scarce
  resource; ARMOR ran N=128 plain on a100).
- ff_le8 submission gated on the smoke passing (same code path at scale).
- Interlude (other campaign, on Tal's ping): treefold `tf_armc7b` 137358 was PD-blocked
  on l40s Priority → moved to the free rtx6k GPU, crashed on a missing sys.path entry
  (`ask_compile_execute` lives in scripts/recagg/); one-line path fix in
  `scripts/treefold/armc_7b.py`, resubmitted 137462 = COMPLETED, Tal's chained
  smoke+full job 137461 moved to rtx6k and now RUNNING. No further treefold action.

## [2026-08-26 ~16:40] L1–L5 SCRIPT DELTAS PREPARED (none run); jobs widened across partitions

- Cluster fully saturated (every GPU allocated); both pending jobs widened to
  multi-partition so SLURM takes the first free slot: lm_smoke 137464 →
  rtx6k/a100-public/l40s-public/l40s-shared; lm_ff_le32 137465 → h200 + l40s
  (Tal-authorized fallback). RISK logged: if ff_le32 lands on 48 GB l40s, the P4
  OOM history may reappear as SILENT per-sample "train skip" cascades (the trainer
  catches per-sample exceptions) — will check its log at start; skip-cascade at
  N=32 ⇒ scancel + h200-only resubmit.
- **L1 delta APPLIED** to `scripts/armor/probe_hahn.py`: `--peft-adapter` (frozen
  PeftModel restore) + `--arms` subset selection (default = all three arms = exact
  pre-delta behavior; anchors unaffected). Compile-checked. Not run.
- **L2/L3/L4 adapter loads APPLIED** to `probe_sensitivity.py`, `probe_supply.py`,
  `probe_lasttok_attn.py`: `--peft-adapter` + shared `scripts/loramech/peft_utils.py`.
  Trap closed there: `dequantize_linear_weight(o_proj)` on a LoRA-wrapped Linear4bit
  silently drops the adapter delta — `effective_linear_weight` merges
  dequant(base) + scaling·B@A (identical to before when no adapter). Compile-checked.
  Not run. (The L2 frames-first-layout + answer-position-target rework is NOT done —
  only the adapter load, per the start message.)
- **L5 PREPARED, NOT APPLIED** (`scripts/loramech/PREPARED_L5_attn_logn.md`):
  train_sft_baseline.py is on the PENDING P0 jobs' code path — SLURM reads scripts at
  job start, so the file stays frozen until both trainers are past start. Mechanism
  VERIFIED in this venv (transformers 4.57.6 passes self.scaling into sdpa).
- L6 rescorer written: `scripts/loramech/rescore_percount.py` (per-count recall, pf,
  MAE, class dist, majority baseline, extremes-vs-mid split for H4).

## [2026-08-26 ~16:25] SMOKE PASSED → ff_le8 SUBMITTED AND RUNNING

- lm_smoke 137464 COMPLETED (4m59s, rtx6k n317-pool):
  `outputs/_scratch/loramech_smoke/20260826_161533_lora/`. All mechanical checks green:
  **trainable params 23,794,688 = the E-B/P4.1 anchor count exactly** (r=8 both);
  `[template] frames-first`; `[exclude] 150 dirs` honored (11 skipped in the scanned
  prefix); adapter saved @ep0; dirs-file exam ran (n=12, pf 0); `longn_predictions.csv`
  written; post-hoc `check_disjoint` train_dirs vs exam_ff_N8 = **overlap 0**.
  (acc 0.50 val / 0.33 exam — expected at 28 train samples × 1 ep; not a signal.)
- **lm_ff_le8 137467 submitted → RUNNING on n318 (rtx6k)**, 24h_1g, --time=8:00:00,
  mem 48G. Full recipe (seq8 −exam_N8, 5 ep, r=8 ff); in-job exams N8/16/32
  (LONGN_LIMIT=150) → `outputs/loramech/p0_ff_le8`. Zero-shot 64/128 deferred to a
  separate eval-only job (protects the exam CSVs from a late-generate OOM).
- treefold 137461 (Tal's chained smoke+full) COMPLETED 10m23s — no action taken, per Tal.
- lm_ff_le32 137465 still PD (h200+l40s, Priority) — every GPU allocated again.

## [2026-08-26 ~17:30] ff_le8 DONE — FIRST HEADLINE: frames-first alone breaks the in-length fit

Run: job 137467 (rtx6k n318, 1h07m) → `outputs/loramech/p0_ff_le8/20260826_161912_lora/`
(train 525 / val 112 / test_iid 113; 0 train-skips; best ep2 val 0.633; test_iid 0.593;
adapter → `checkpoints/sft_ff_le8_adapter/` + README row). Contamination: post-hoc
check_disjoint train_dirs vs all three exam files = overlap 0 (on top of by-construction
exclusion).

| exam (150 each, class-balanced, pf 0) | acc | MAE | majority floor |
|---|---|---|---|
| N=8 in-length | **0.540** | 0.48 | 0.113 |
| N=16 zero-shot | 0.333 | 1.12 | 0.093 |
| N=32 zero-shot | 0.180 | 2.73 | 0.080 |

**Reading (draft, honest):** the E-B mirror does NOT mirror. Q-first ≤8-trained LoRA hit
0.998–1.000 in-distribution ([2026-07-20] E-B); the SAME 23.8M recipe trained
FRAMES-FIRST plateaus at ~0.6 val / 0.54 exam AT ITS OWN TRAINING LENGTH. Anatomy at
N=8 is FLAT across g1–g7 (0.47–0.59) with only g8 dead (0.125) — NOT the extremes
shortcut; looks like in-window dilution, not a heuristic. Decay 0.54 → 0.33 → 0.18 is
graceful; every cell ≫ majority floor. Frozen ff baseline for context: 0.219.
This previews the H0 contingency branch (brief §3 H0): without question-conditioned
encoding the LoRA may not fit even short windows — the regime switch itself removes
most of what the Q-first LoRA "learned". Caveats to check before hardening: (1) legacy
trainer hyperparameter parity (lr/grad-accum) — refactored defaults believed identical
but train-path parity was never number-anchored; (2) val still noisy at 112; (3) ff_le32
(3-length mixture, more data) may still fit — its H0 is the pre-registered decision.
- L6 rescore written to run dir (`rescore_percount.py` output above).
- **lm_ff8_zs 137504 submitted → RUNNING n317** (12h_4g, --time=4:00:00): ff_le8
  zero-shot N=64/128, 150 each → `outputs/loramech/zs_ff_le8/`.
- **lm_ff_le32 137465 STARTED on the H200 (n315)** ~17:14 — no l40s OOM concern.

## [2026-08-26 ~17:45] ff_le8 caveat (1) CLOSED — recipe/budget parity with the anchors verified

Read-only check of `legacy/experiments/glstm/lora_sft_baseline.py` + both anchor
run.logs: lr 2e-4, grad_accum 8, dropout 0.05, clip 1.0, AdamW, resize 392 — all
identical to the refactored trainer; anchor runs actually trained 4–5 epochs (E-B five
ep0–4 best ep3; P4.1 four ep0–3 best ep2), same as our 5-ep budget. Anchors reached
train_loss 0.08–0.10 / val ≥0.95 by ep2–3; frames-first plateaus at loss 0.237 /
val 0.633 with the same knobs → the in-length miss is REGIME-driven, not a
hyperparameter or epoch-budget artifact. (Val-split noise caveat (2) and the ff_le32
H0 decision (3) remain open — ff_le32 running on the H200 since ~17:14.)

## [2026-08-26 ~18:05] ff_le8 zero-shot N=64/128 done — extremes relapse OUTSIDE the window

Job 137504 (rtx6k n317, 28m) → `outputs/loramech/zs_ff_le8/20260826_173123_lora/`
(eval-only, `checkpoints/sft_ff_le8_adapter`, frames-first template; the TEST_IID line
in this run is from an unexcluded convenience split — ignore it, the exam files are the
citable cells).

Full ff_le8 ladder (150/cell, pf 0): **N=8 0.540 · N=16 0.333 · N=32 0.180 ·
N=64 0.173 · N=128 0.167** (majority floors 0.113/0.093/0.080/0.067/0.060).

**H4 anatomy (rescore_percount):** inside/near the window the curve is LIVE (N=8
mid-range 0.548 ≈ extremes 0.530); far outside it RELAPSES to the extremes heuristic —
N=64 extremes 0.400 vs mid 0.091; N=128 extremes 0.471 (g0 9/9, g128 6/8) vs mid 0.078.
Same qualitative fingerprint as Q-first E-B beyond ITS window, but shifted down: the
frames-first LoRA never had the in-window fit to lose. Draft H4 call for the ≤8 arm:
replicates the shortcut-relapse structure in the frames-first regime.
Remaining P0 piece: ff_le32 (H200, running since 17:14; H0 verdict when its in-length
exams land).

## [2026-08-26 ~20:30] P0 COMPLETE — H0 VERDICT: REFUTED (contingency branch active). STOPPING FOR TAL'S REVIEW.

ff_le32: job 137465 (H200 n315, 3h08m) → `outputs/loramech/p0_ff_le32/20260826_171532_lora/`
(train 923 / val 198 / test_iid 199; 0 train-skips; best ep4 val 0.617; test_iid 0.533;
adapter → `checkpoints/sft_ff_le32_adapter/` + README row; check_disjoint vs all three
exam files = overlap 0).

### H0 (pre-registered): ff_le32 in-length N=8/16/32 ≥ 0.90 each → **REFUTED**

| cell (150 each, pf 0) | ff_le32 | ff_le8 | Q-first anchor |
|---|---|---|---|
| N=8 in-length | **0.600** | 0.540 | 0.998–1.000 (E-B, iid) |
| N=16 in-length | **0.480** | 0.333 (zs) | — |
| N=32 in-length | **0.313** | 0.180 (zs) | **0.967** (P4.1) |
| N=64 zero-shot | (job 137593 running) | 0.173 | 0.787 (P4.1) |
| N=128 zero-shot | (job 137593 running) | 0.167 | — |

N=32 lands BELOW the 0.80 contingency line, so per the brief's own words: "without
question-conditioned encoding a 24M LoRA cannot fit N=32 in-length — the frames-first
regime is where dilution already binds at 32 — and L1 runs on ff_le8 + the ≤32 window
as measured."

### Anatomy (rescore, run dirs have full tables)

- ff_le32 is EXTREMES-ANCHORED even in-length: N=32 extremes 0.587 vs mid-range 0.192
  (g32 11/11, g0 8/12, g2/g3/g5/g6 = 0); N=16 extremes 0.778 vs mid 0.312. In-length
  training in frames-first buys anchor classes (incl. perfect trained-max recognition),
  NOT the uniform per-count curve Q-first had (P4.1 was per-count uniform at 0.967).
- ff_le8 at its own length is the opposite: FLAT live curve (mid 0.548 ≈ extremes
  0.530) — dilution, not shortcut, inside the window; relapses to extremes at 64/128.

### Honest caveats

1. ff_le32 val was still rising at the last epoch (0.400→0.467→0.617, loss 0.297
   declining) — a longer-budget run might add some points; it cannot plausibly close a
   0.59-gap to the band, but "5 ep = anchor budget" is cleaner for ff_le8 (plateaued)
   than for ff_le32. If Tal wants, a 10-ep confirmation run is ~6h H200.
2. Zero-shot 64/128 for ff_le32 in flight (137593, rtx6k) — H1's α_out read and the
   full §6 row complete when it lands; will append below without starting L1–L5.
3. The [2026-08-22] ARMOR-A frozen frames-first accuracy on the gold≤8 slice (0.28 @N=8)
   and the frozen 0.219 anchor are the frozen references for the drift row.

### What P0 already buys the paper (draft reading, pre-L1)

The peer's objection ("plain LoRA reads N=8 at 1.000 — how can Hahn bind?") is
answered at the BEHAVIORAL level before any probe runs: the 1.000 was the Q-FIRST
regime, where causality lets frame tokens precompute question-conditioned verdicts
(the carrier route in disguise). Deny that route (frames-first, the layout Hahn's
setting actually describes and our frozen baseline/instruments use) and the same
24M-param LoRA cannot even fit its training lengths (0.600/0.480/0.313), decaying
monotonically inside the window with an extremes-anchor signature. The mechanism
cells (L1–L5) now ask a sharper question: is the residual in-window skill a sharpened
diluting sum (GAIN) — and does log-N scaling buy it back (L5)?

**STOP.** Per the mandatory checkpoint, L1–L5 are NOT started. Prepared and idle:
probe_hahn `--peft-adapter/--arms` (applied, compile-checked), adapter loads in the
three L2/L3/L4 probes (applied, compile-checked), L5 patch drafted in
`scripts/loramech/PREPARED_L5_attn_logn.md` (NOT applied — trainer file now safe to
edit once 137593 finishes). Awaiting Tal's review + the L1 window decision.

---

## DRAFT RESULTS ENTRY (for Tal — do not paste into RESULTS.md without review)

## [2026-08-26] ✅📊 LORAMECH P0 — THE REGIME IS THE BASELINE: the "plain LoRA solves it" objection is Q-first-specific — the SAME 24M recipe trained FRAMES-FIRST cannot fit even its own training lengths (in-length exams 0.600 @8 / 0.480 @16 / 0.313 @32, pf 0, vs Q-first P4.1 0.967 @32), with an extremes-anchor anatomy in-window — the frames-first SFT row replaces the Q-first row as the canonical §6 baseline (Q-first numbers stay, regime-labelled)

> Runs: ff_le8 trainer 137467 (rtx6k, 1h07) → `outputs/loramech/p0_ff_le8/20260826_161912_lora/`;
> ff_le32 trainer 137465 (h200, 3h08) → `outputs/loramech/p0_ff_le32/20260826_171532_lora/`;
> ff_le8 zero-shot exams 137504 → `outputs/loramech/zs_ff_le8/20260826_173123_lora/`;
> adapters promoted: `checkpoints/sft_ff_le{8,32}_adapter/`. Template parity gate:
> `scripts/loramech/check_ff_template.py` (frames-first == build_count_prompt layout,
> token-identical). Contamination: exam sets excluded from training BY CONSTRUCTION
> (`--exclude-dirs-file`) + post-hoc overlap 0; N=32 exam from the disjoint park2
> generation; recipe/budget parity with the E-B/P4.1 anchors verified in-log
> (23,794,688 trainable params, lr/accum/dropout/epochs identical).

- ff_le8 full ladder (150/cell, pf 0): 0.540 / 0.333 / 0.180 / 0.173 / 0.167 @
  N=8/16/32/64/128 (majority floors 0.113→0.060): flat live curve at N=8
  (mid 0.548 ≈ extremes 0.530), extremes-heuristic relapse at 64/128
  (0.400/0.471 extremes vs 0.091/0.078 mid) — H4's fingerprint, frames-first edition.
- ff_le32 in-length anatomy: extremes-anchored INSIDE the window (N=32 extremes 0.587
  vs mid 0.192; trained-max g32 11/11) — in-length data buys anchors, not the curve.
- H0 refuted at its own <0.80 line ⇒ pre-registered contingency: L1 mechanism cells
  run on the window as measured. Caveat: ff_le32 val still rising at ep4 (0.617) —
  budget-parity with anchors held, but a longer run could add points (gap 0.59 not
  plausibly closable).

## [2026-08-26 ~21:15] ff_le32 zero-shot done — P0 fully closed; still STOPPED for review

Job 137593 (rtx6k n317, 27m) → `outputs/loramech/zs_ff_le32/` (eval-only, frames-first;
ignore the in-run TEST_IID convenience split as before).

**Complete canonical frames-first SFT rows (150/cell, pf 0):**
- ff_le32: N=8 **0.600** · N=16 **0.480** · N=32 **0.313** · N=64 **0.293** · N=128 **0.160**
- ff_le8:  N=8 **0.540** · N=16 0.333 · N=32 0.180 · N=64 0.173 · N=128 0.167

ff_le32 zero-shot anatomy = the extremes heuristic in pure form: N=128 g0 9/9 +
g128 8/8, all mid classes 0; N=64 g0 10/10 + g64 10/10 + trained-max g32 5/10, mid
dead. The 0.293 @N=64 sits far under the Q-first P4.1 0.787 — the Q-first
extrapolation advantage was also largely the conditioned-encoding route.
All P0 numbers now on disk with run dirs; draft RESULTS entry above updated by these
two cells (see table). L1–L5 remain NOT STARTED — awaiting Tal.

## [2026-08-27 ~09:20] Tal's review question → two P0 controls launched (still no L1–L5)

Tal asks: is Q-first really THE factor; did the ff runs converge / have enough
samples+epochs? Honest state: recipe parity is verified and the training LOSS itself
fails to fit in frames-first (0.24–0.30 plateau vs anchors' 0.08–0.10), but (a) the
Q-first arm never ran through the NEW pipeline (template+script+split changed together
vs the legacy anchors), and (b) ff_le32's best val was its LAST epoch (0.617, still
rising) and both ff runs trained on ~17% fewer samples than anchors (exam exclusions).

- **C1 `lm_c1_qfirst` 137706 (l40s-public, 2h_2g, RUNNING):** Q-first ≤8 control
  through the IDENTICAL refactored pipeline — same seq8 pool, same exam_ff_N8
  exclusion (525 train), same r=8/5-ep recipe; only the template flag differs
  (trains AND evals with its own Q-first template, the E-B convention).
  Expected if regime is the factor: in-length ~1.0. → `outputs/loramech/c1_qfirst_le8`
- **C2 `lm_c2_ep15` 137707 (h200 n315, 24h_1g, --time=10:00:00, RUNNING):** ff_le32
  rerun with EPOCHS=15 (patience 8) — the convergence-ceiling control.
  → `outputs/loramech/c2_ff_le32_ep15`

## [2026-08-27 ~10:00] P1 "fenced-SFT" arm (Tal's proposal) — trainer written, smoke running

Tal proposes the causal test of the P0 explanation: train a plain LoRA under the
ARCHITECTURAL fence — N independent [frame_i + q-replica_i] blocks (replica AFTER its
frame: verdicts computed in replica positions), canonical block fence + per-block
posreset, one final build_count_prompt that sees all blocks; NO Q-first. Decision:
train at N<=8+16 ONLY for now (Tal: "h200s notoriously packed") — the 4-D fence mask
knocks sdpa off FLASH (the documented MATH 45.6GiB@N=64 hazard), but at N<=16
(~1.8k/3.6k tokens) EFFICIENT/MATH + grad-ckpt fits 48 GB, so P1 runs entirely on
rtx6k/l40s/a100.

- New `scripts/loramech/train_sft_fenced.py` (recipe mirrors the baseline trainer:
  r=8/alpha=32 q/k/v/o+MLP, lr 2e-4, accum 8, answer-token loss, exclusion discipline).
  Fencing imported READ-ONLY from gnnformer.fencing: `build_block_mask(hide_cols=[])`
  (replicas VISIBLE to the tail — they carry the verdicts), `reset_positions`,
  `FenceHooks`. Decode = exact cache-free greedy re-forward (mask+pos rebuilt per
  step, <=4 tokens). Per-sample layout invariants hard-checked (N+1 q-spans, replica_i
  inside block_i). tests/test_fencing.py 12/12 PASS (gnnformer untouched).
- New wrapper `slurm/loramech_train_fenced.sbatch` (same knob set as the SFT wrapper).
- **Pre-registered bands (P1, fixed now):** in-length (train N∈{8,16}, exams
  exam_ff_N8/N16) ≥0.90 each → conditioning-is-the-factor (the architectural fence
  recovers what Q-first gave); ≤0.50 with dead mid-range → the readout wall
  (P4.2-style) is a second, separable factor; between → mixed, log as measured.
  Zero-shot N=32 read for the horizon question. Comparators: ff_le8 0.540@8,
  ff_le32 0.480@16, Q-first ~1.0, frozen ff 0.219.
- **lm_p1_smoke 137728 RUNNING (l40s-public, 2h_2g)**: LIMIT=24 EPOCHS=1 → _scratch.
  On pass → P1 trainer at seq 8+16 on a 48 GB card (no H200 dependency).
- LoRA scope documented (read from the adapter safetensors header, applies to ALL
  arms incl. anchors): 28 LM decoder layers × {q,k,v,o,gate,up,down} = 20.19M params
  + 32 VISION blocks × {gate,up,down} (name-collision match) = 3.61M → 23.79M = 0.286%
  of 8.32B; vision attention, embeddings, LM head, norms, and all base weights frozen
  (4-bit). Effective W = W_frozen + (32/8)·B_r8·A_r8 per matched matrix.

## [2026-08-27 ~13:10] P1 smoke #1 FAILED on layout parse → fixed → smoke #2 running

- 137728: mechanically completed but 16/16 train samples + all evals hit
  "layout parse failed" — root cause: `find_question_spans` needs all N+1 question
  copies to tokenize identically, but the final copy sits after "Question: " inside
  build_count_prompt while replicas follow vision tokens (mixed contexts → no single
  needle count matches). The _scratch adapter from this run is untrained garbage.
- Fix: `parse_layout` no longer matches the question at all — blocks need only the N
  vision starts + the final-prompt start, located via the UNIQUE build_count_prompt
  opener "You will be shown" (a question can never contain it). Verified on CPU with
  the real tokenizer at N=2/8, prompt+answer variants: parse OK, block width 208
  (frame 198 + replica ~10), fence invariants hold (cross-block MASK_MIN, own-block
  causal, tail reads all).
- **lm_p1_smoke2 137742 RUNNING** (l40s-public), same knobs → `_scratch/loramech_p1_smoke2`.

## [2026-08-27 ~13:15] P1 smoke2 GREEN → P1 le16 trainer submitted

- 137742 (7m54s): 0 layout skips, fenced train loss 0.828 over 16 samples, cache-free
  fenced decode works end-to-end (exam parse_fail 0.000). Tiny-split accuracies are
  noise by design. Mechanical pass.
- **lm_p1_le16 137744 RUNNING (l40s-public, 24h_1g, --time=6:00:00, mem 48G):** the
  real P1 arm — roots seq8+longN16 (exam_N8/N16 excluded; ~930 samples), 5 ep, r=8,
  fenced [frame+q]-block layout + posreset; in-job exams exam_ff_N8/N16 (in-length)
  + exam_ff_N32 (zero-shot horizon read), 150 each.
  → `outputs/loramech/p1_fenced_le16/`. No H200 used (Tal's constraint).

## [2026-08-27 ~13:40] C1 LANDED — TEMPLATE CONFIRMED AS THE CAUSAL FACTOR (within-pipeline)

Job 137706 (l40s-public, 1h59m) → `outputs/loramech/c1_qfirst_le8/` run dir. Q-first
through the IDENTICAL refactored pipeline (same seq8 pool, same exam_N8 exclusion,
same 525/112/113 split, same r=8 5-ep recipe; trains AND evals its own Q-first
template):

- **In-length exam N=8: 0.867** (vs frames-first ff_le8 0.540 — the template flip
  alone = +0.33 absolute in the SAME pipeline) · test_iid 0.903 · loss 0.070 by ep4
  (anchor-like convergence; ff plateaued at 0.237: the optimization itself unsticks).
- Beyond window (own template): N=16 0.487, N=32 0.327 — classic Q-first anatomy
  (N=16: g0–g3 + g16 perfect, g5–g12 ALL zero; N=32: low band + g32 11/11, middle
  dead) — matches E-B's extremes/anchor fingerprint.
- Residual vs the LEGACY anchor (E-B 0.998 iid): C1 reads 0.867/0.903 — remaining
  ~0.10–0.13 attributable to fewer train samples (525 vs 630) and val still rising at
  ep4 (0.900 best=last). Does NOT change the verdict: the dominant factor is the
  template, measured within-pipeline; the residual is a sample/epoch-budget effect
  (C2's 15-ep run will inform this).
- Draft RESULTS delta: add C1 row — "Q-first control, identical pipeline: 0.867@8
  in-length (+0.33 over frames-first), loss 0.070 vs 0.237 — regime isolated as THE
  learnability factor."
Still running: C2 (h200, 15 ep, ~1h in), P1 le16 (l40s, ~30m in).

## [2026-08-27 ~16:30] P1 LANDED — THE ARCHITECTURAL FENCE RESTORES (AND BEATS Q-FIRST'S) LEARNABILITY

Job 137744 (l40s-public, 3h12m) → `outputs/loramech/p1_fenced_le16/20260827_131154_fenced/`
(train 651 / val 139 / test 140; 0 layout skips; best ep3 val 0.967, test_iid 0.964;
adapter → `checkpoints/sft_fenced_le16_adapter/` + README row; check_disjoint vs all
three exam files = overlap 0; pf 0 everywhere).

| in-length exam | plain ff | Q-first C1 | **P1 fenced (no Q-first)** |
|---|---|---|---|
| N=8  | 0.540 | 0.867 | **0.9933** (mae 0.01; mid-range 0.988) |
| N=16 | 0.480 (ff_le32) | — | **0.8467** (mid-range 0.760) |
| N=32 zero-shot | 0.313 (ff_le32 IN-length) | 0.327 | **0.3867** (mae 0.98; mid 0.337) |

**Band call (pre-registered):** N=8 ≥0.90 MET decisively; N=16 0.847 just under 0.90 →
formally "mixed, logged as measured" — but the qualitative verdict is unambiguous:
**conditioning is the factor, and the ARCHITECTURAL version beats the layout hack.**
The fenced arm's per-count curves are LIVE in-window (no extremes crutch: N=8
extremes 1.000 / mid 0.988; N=16 1.000/0.760) where Q-first C1 at N=8 reads 0.867
with dips and every other arm shows dead-mid anatomy. Even zero-shot at N=32 the
fenced arm keeps a partially live middle (0.337 mid, MAE 0.98) and BEATS the plain
arm that was TRAINED at N=32 (0.313, MAE 2.11).
- The N=32 decay (0.387) is the horizon: the tally read over verdict blocks is still
  one softmax — the law survives, the intercept moved. (The L1 α measurement on this
  adapter is the natural follow-up, same gate as before.)
- Spectrum at N=8 in-length now: frozen 0.219 → ff-LoRA 0.540 → Q-first-LoRA 0.867 →
  **fenced-LoRA 0.993** ≈ caption method. This is the §5/§6 spectrum figure's trained
  middle rungs, measured in one campaign with one pipeline.
- Draft RESULTS entry below updated accordingly. C2 (15-ep h200) still running.

---

## DRAFT RESULTS ENTRY #2 (for Tal — C1 + P1, do not paste without review)

## [2026-08-27] ✅📊 LORAMECH C1+P1 — CONDITIONING ISOLATED AND RE-SUPPLIED ARCHITECTURALLY: within one identical pipeline, the template flip alone buys +0.33 in-length (ff 0.540 → Q-first 0.867 @N=8), and the METHOD'S FENCE (N×[frame+q] blocks + posreset, NO Q-first) buys +0.45 (**0.993 @8, 0.847 @16, live per-count curves**) — the trained LoRA spectrum now brackets the carrier method's mechanism claim

> Runs: C1 Q-first control 137706 → `outputs/loramech/c1_qfirst_le8/`; P1 fenced-SFT
> 137744 → `outputs/loramech/p1_fenced_le16/20260827_131154_fenced/` (new trainer
> `scripts/loramech/train_sft_fenced.py`, canonical gnnformer.fencing mask/posreset
> read-only, cache-free exact fenced decode; adapter →
> `checkpoints/sft_fenced_le16_adapter/`). Same pipeline, splits, exclusions, and
> r=8 recipe as P0 throughout; pf 0, overlap 0, 0 layout skips.

- The three-way contrast at N=8 in-length (identical data budget): plain frames-first
  0.540 (loss stuck 0.237) · Q-first 0.867 (loss 0.070) · fenced frames-first 0.993
  (loss 0.136, val 0.967) — question-conditioned per-frame encoding is THE
  learnability factor, and the explicit architectural supply (fence+replicas+posreset)
  is BETTER at it than the Q-first layout at equal budget.
- Anatomy separates mechanism: fenced curves are LIVE per-count in-window (mid-range
  0.988/0.760 @8/16); Q-first leans on extremes/anchor classes everywhere beyond its
  comfort zone (N=16: g5–g12 all zero). The fence produces uniform verdicts; Q-first
  produces margin-sharpened gist.
- The horizon survives: fenced zero-shot N=32 = 0.387 (yet still > plain trained-at-32
  0.313, with MAE 0.98 vs 2.11) — conditioning moves the intercept, the softmax tally
  read keeps the slope (Hahn); L1 α on the fenced adapter is the quantitative closer.
- Thesis positioning: the LoRA "rediscovering" nothing — we HANDED it the carrier
  structure and it used it to near-ceiling. The method's §5 mechanism story
  (fence = portable conditioning, readout = the remaining bottleneck) now has a
  trained-baseline spectrum measured end-to-end in one pipeline.

## [2026-08-27 ~16:50] N=16 residual DIAGNOSED (+1-overcount bias) → P1b 10-ep control launched

- Error anatomy of P1 @N=16 (from longn_predictions.csv): 23/23 errors are EXACTLY +1
  over-counts, all in g3–g7; zero under-counts. A systematic calibration bias, NOT
  diffuse tally noise (dilution would be ~symmetric ±1) and NOT a dead mid-range.
- Prime suspect: DATA imbalance — only 126 N=16 train samples vs 525 N=8 (pool 330
  − 150 exam = 180 × 0.7). Epochs secondary (val noisy, best ep3/5). The Hahn-margin
  explanation is disfavored for THIS residual by the one-sidedness.
- **lm_p1b_ep10 137778 RUNNING (l40s-public, 24h_1g, --time=10:00:00):** P1 recipe
  ×10 epochs, same data/exclusions → `outputs/loramech/p1b_fenced_ep10/`.
  Read: N=16 cell moves → epochs; doesn't move → data-limited (next lever: generate
  more seq16 MMReD from the park generators, CPU 4h_0g, then P1c).
- NOTE for C2 landing: squeue elapsed-time jumps suggest 137707 may have been
  requeued/restarted mid-run (no resume in the trainer — it would just retrain from
  scratch; results remain valid but check sacct for NODE_FAIL/requeue at landing).

## [2026-08-27 ~17:20] Tal green-lights continuation → P2 (fenced on MMReD-HF) + L1 (Hahn α) launched

Tal: train the fenced LoRAs on MMReD-HF train splits (seq 8+16) + resume the original
mechanism campaign ("find out how it's so good"). Launched:

- **P2 `lm_p2_hf` 137799 (l40s-public, 24h_1g, --time=6:00:00, RUNNING):** fenced-SFT
  on the benchmark's canonical `steps_in_room` train dirs — 200/length (NOT 250 —
  actual materialized count), ALL 400 load via load_mmred_sample (qa.txt format ✓,
  0 invalid), class-skewed as the benchmark ships it (N=8: 125/200 gold-0 — majority
  baseline will be reported per the probe-family lesson). 10 ep (small data),
  train/val 0.8/0.2 (no internal test — the benchmark's own test split is the exam:
  `dirsfiles/seq_len_{8,16,32}_test_steps_shuf.txt`, 50 each, pre-shuffled per the
  K0-trap lesson). Train/test disjointness = the benchmark's own.
  → `outputs/loramech/p2_fenced_hf/`
- **L1 `lm_l1_hahn` 137800 (ff_le32, a100 n310, RUNNING) + 137801 (ff_le8, a100, PD):**
  new chain wrapper `slurm/loramech_l1_hahn.sbatch` — probe_hahn plain arm,
  --peft-adapter, N∈{8,16,32,64,128} chained, SAME pools/limits/seed as ARMOR-A
  jobs 136128–32 → paired vs the frozen α=0.72 curve. 12h_4g, --time=4:00:00.
  → `outputs/loramech/l1_hahn/{ff_le32,ff_le8}_N*/`
- Next prep (no run yet): a P1-layout arm for probe_hahn (fenced adapter's OWN layout:
  no Q-first opener, build_count_prompt tail, hide_cols=[]) so L1 can run on
  `sft_fenced_le16_adapter` — the current fenced arm in probe_hahn is the METHOD's
  probe layout, not P1's.
Still running: C2 (h200, 15 ep), P1b (10-ep P1 rerun), P2, L1×2.

## [2026-08-27 ~17:55] L1 chains 137800/137801 FAILED on a PEFT delegation bug → fixed → resubmitted

- Failure (11 min in, N=8 cell): `get_rope_index_fn` called AFTER the PeftModel wrap —
  `PeftModel.model` is the OUTER base model (not the inner Qwen2_5_VLModel), so every
  `model.model.*` chain captured post-wrap resolves wrong. My earlier "delegation is
  transparent" assumption was wrong for `.model` specifically.
- Fix applied to ALL FOUR probe deltas (probe_hahn, probe_sensitivity, probe_supply,
  probe_lasttok_attn): capture every structural ref (rope_fn, text_model, inner_model
  for get_image_features, vs_id, dims) from the RAW model, wrap with PEFT LAST.
  LoRA injects in-place so pre-wrap refs stay valid and LoRA-active. Compile-checked.
- **Resubmitted: 137821 (ff_le32, RUNNING n310) + 137822 (ff_le8, PD).** Cost of the
  bug: ~23 min a100 time.

## [2026-08-27 ~19:45] L1 LANDED — GAIN CONFIRMED AT THE α LEVEL: the trained LoRA keeps the frozen decay law and just multiplies the signal

Jobs 137821/137822 (a100 n310, ~1h37 each) → `outputs/loramech/l1_hahn/{ff_le32,ff_le8}_N{8..128}/`
+ fitted figs `outputs/loramech/l1_hahn/{ff_le32,ff_le8}_fig/` (fig_hahn.py verdicts) +
in/out-of-window bootstrap (2000 resamples, this session):

| curve (plain arm, L20 final) | α | 95% CI | frozen comparator |
|---|---|---|---|
| ff_le32 IN-window {8,16,32} | **0.710** | [0.36, 0.93] | 0.72 [0.63,0.78] |
| ff_le8 in/near-window {8,16,32} | **0.732** | [0.32, 0.93] | 〃 |
| ff_le32 out {32,64,128} | 0.287 | [0.01, 0.63] | — |
| ff_le8 out {32,64,128} | 0.593 | [0.36, 1.02] | — |
| ff_le32 all-5 (fig fit) | 0.444 | [0.33, 0.54] r²=0.89 | — |
| ff_le8 all-5 (fig fit) | 0.640 | [0.55, 0.73] r²=0.99 | — |

**Readings.** (1) In-window α of BOTH trained adapters is statistically identical to
the frozen model's 0.72 — training did NOT change the sensitivity decay law. (2) What
training changed is AMPLITUDE: LoRA-on median ‖Δh‖ is ~3–6× frozen at every N
(29.35 vs 10.5 @N=8; 7.37 vs 1.3 @N=128 — lifted clear of the bf16 floor, but the
margin still slides: margin medians −0.60 @64 / −2.42 @128 for ff_le32). This is the
GAIN mechanism, measured: an amplified diluting sum. (3) H2's GAIN α-condition MET
for both adapters (in-window α ∈ [0.5,1.0], CI excludes 0.2); the FULL pre-registered
GAIN verdict still wants the L2 Jacobian-share and L3 frame-position-d′ co-conditions
— both probes now have working adapter loads, but the honest L2/L3 for frames-first
need the plain-layout deltas (probe_sensitivity/probe_supply currently build the
Q-first replica layout) — NEXT PREP.
(4) H1 (law survives, out-window α ≥ 0.5): ff_le8 MET (0.593); ff_le32 INDETERMINATE
(0.287 [0.01,0.63], dragged by a 32→64 plateau 10.97→10.73 — logged as-is, wide IQRs;
not the <0.2 refutation either). (5) Skip counts rise with N (8/10/20/30/32 of 50-ish
attempted) — pair-construction skips, same seed/pools as ARMOR-A (paired).
- Draft RESULTS: the peer-objection answer is now complete at both levels —
  behaviorally (P0/C1/P1) and mechanistically (L1): the plain LoRA's in-window skill
  is margin gain on an unchanged 1/N read; the fence changes the SUPPLY structure
  instead, which is why P1 generalizes with live curves.

## [2026-08-27 ~21:00] P2 LANDED — the fenced LoRA transfers to the FAITHFUL benchmark

Job 137799 (l40s-public, 3h42m) → `outputs/loramech/p2_fenced_hf/` run dir (train 320 /
val 80, benchmark's own train/test split; 10 ep, best ep1 val 0.817, 0 skips; adapter →
`checkpoints/sft_fenced_hf_adapter/`).

| MMReD-HF steps_in_room test (50/len, benchmark split) | fenced LoRA | majority floor | frozen anchor |
|---|---|---|---|
| N=8 in-length | **0.900** (mae 0.10) | 0.62 | 0.559 (arm-A, below floor) |
| N=16 in-length | **0.660** (mae 0.54) | 0.38 | — |
| N=32 zero-shot | 0.300 | 0.16 | — |

- NOT majority-riding: non-g0 recall 14/19 @N=8 (extremes 0.949 / mid 0.727); N=16
  mid-range 0.364 — the class skew is the benchmark's own (train 213/400 gold-0).
- Reference ceiling: armB GIN over structured per-frame records reads 0.928–0.952
  length-flat — the fenced END-TO-END model at N=8 (0.900) is ~0.03 under that
  structured ceiling, ON the faithful benchmark, trained with 320 samples on one l40s.
- Same window shape as park-P1 (in-length strong, zero-shot horizon at 32) — the
  regime story replicates off our own generators. Caveats: 50-sample cells; skewed
  classes (both reported); 0.900@8 is a 4-sample-error cell.

## [2026-08-28 ~01:10] P1b LANDED — 10-epoch fenced arm EXTRAPOLATES: zero-shot N=32 0.387 → 0.867

Job 137778 (l40s-public, 5h39m) → `outputs/loramech/p1b_fenced_ep10/20260827_192346_fenced/`
(best ep8 val 0.983, test_iid 0.971, 0 skips, pf 0, check_disjoint ×3 = 0; adapter →
`checkpoints/sft_fenced_le16_ep10_adapter/` — new CANONICAL fenced ckpt).

| exam (150/cell) | P1 5-ep | **P1b 10-ep** |
|---|---|---|
| N=8 in-length | 0.993 | **1.000** (150/150, mae 0.00) |
| N=16 in-length | 0.847 | **0.893** (mae 0.11) |
| N=32 ZERO-SHOT (2× train length) | 0.387 | **0.867** (mae 0.16; g32 11/11, g24 7/11) |

**Readings.** (1) Tal's "not enough epochs?" — answer: both. epochs closed most of the N=16
gap AND transformed extrapolation (0.387→0.867). The fenced arm is the FIRST in the
record to beat the zero-shot length wall (running-tally read 0.280 at 2×; plain/Q-first
LoRAs collapse). Consistent with the mechanism: the fence makes per-frame verdicts
N-invariant (ARMOR-A flat supply), so a longer-trained tally read scales past its
window. (2) Error structure sharpened: N=16 residual = 16/16 exactly-+1 overcounts now
CONFINED to g3/g4 — a narrow systematic band (data artifact candidate, not
resolution); N=32 errors are mostly −1 undercounts (mild dilution at 2×). (3) Caveats:
best-epoch selection rides a noisy 60-sample decode-val (0.483–0.983 swings); exam
n=150 cells are the citable numbers. The N≥64 fenced exam (eval-only, cheap) is the
natural next read — does extrapolation continue or hit the wall at 4×?

## [2026-08-28 ~03:00] P1b zero-shot 64/128 — extrapolation holds to 2×, the read's wall returns at 4× with a DIAGNOSTIC anatomy

Job 137862 (l40s-public, 1h29m) → `outputs/loramech/p1b_zs_64_128/` (eval-only on
`sft_fenced_le16_ep10_adapter`, 100/cell, pf 0).

Fenced (train ≤16) full ladder: **1.000 @8 · 0.893 @16 · 0.867 @32 (2×) · 0.340 @64
(4×) · 0.230 @128 (8×)**.

Anatomy at 64/128 (predictions CSV): three coexisting regimes —
1. LOW BAND (g1–g8): dead by GROWING SYSTEMATIC UNDERCOUNT (N=64 deltas −1/−2/−3;
   N=128 down to −6; e.g. gold 3→1, gold 4→1). The read misses a fraction of evidence
   that grows with N — the softmax-read dilution signature, now on the READ side
   (fenced supply is N-flat; the verdict tokens still compete for attention among
   N blocks). NOT the plain arms' extremes heuristic (their low band was preserved).
2. Trained label-support anchors g12/g16 largely preserved (g12 7/7 + 6/6 at both).
3. All-frames recognition perfect (g64 6/6, g128 5/5) + saturation-adjacent quirks
   (g32 6/6 @N=64 but 0/6 @N=128; g24 6/6 @N=128 but 2/6 @N=64) — logged, not forced.
- Quantitative closer for this story = the P1-layout Hahn arm on the fenced adapter
  (queued prep): prediction = fenced-trained read shows α>0 at the answer position
  while its per-frame supply stays flat.
- C2 note: 137707 ~8h elapsed vs --time=10:00:00; 15 ep at ~37min/ep + exams may
  TIMEOUT before the exam phase — adapter saves incrementally, exams recoverable
  eval-only if cut.

## [2026-08-28 ~03:05] C2 preemption history — twice preempted on h200-shared, attempt 3 running

sacct --duplicates: attempt 1 PREEMPTED after 5h50, attempt 2 PREEMPTED after 7h01,
attempt 3 started 01:14 (fresh run dir each time — the trainer has no resume; earlier
elapsed-time "jumps" in squeue were these restarts). Decision: let attempt 3 ride —
adapter saves every best epoch, and 15 ep (~8.75h) + exams will exceed --time=10:00:00
anyway, so the in-job exams were always going to be cut; plan = harvest the saved
adapter and run the N8/16/32 exams eval-only on a 48 GB card. If preempted a third
time, downscope (10 ep) or close the caveat with C1+P0 budget-parity as-is — the
question C2 answers is a caveat bound, not a headline.

## [2026-08-28 ~09:40] C2 CANCELLED after a 3rd preemption (attempt 4 was at ep0)

~19 H200-hours consumed across 4 attempts with no completed run; h200-shared
preemption pressure makes a 9h job unviable. The caveat closes as: P0 ff_le32 used
the ANCHOR-PARITY budget (5 ep, same as E-B/P4.1 actually trained), C1 isolated the
template as the factor within-pipeline, and the 0.59 gap to the band is not
plausibly an epoch effect (ff_le8 plateaued; C1's Q-first converged anchor-like on
the same split sizes). If a convergence bound is still wanted later: 10-ep rerun
when h200 quiets, or off-peak. RESOLVED-BY-JUDGMENT, logged honestly.

## [2026-08-28 ~10:15] N1 (CPU) — the fenced read's out-of-window failure is LAWFUL: offset first, slope later

Low-band (1≤gold≤8) OLS of pred on gold, fenced ep10 adapter (from the existing
predictions CSVs): **N=32 slope 0.95 · N=64 slope 0.91 with ≈−1.5 constant offset ·
N=128 slope 0.43**. So the read first drops a roughly CONSTANT number of verdicts
(~1.5 at 4×) and only at 8× compresses the slope — while k=N ("all frames",
fraction-1) and the label-support anchors g12/g16 stay exact. Consistent with a
magnitude/fraction-coded tally calibrated in-window (the token-vs-magnitude design
rule, measured). Feeds the Q2/Q3 discussion with Tal; demonstration cells proposed:
N2 = P1-layout Hahn arm on the fenced adapter (read α vs flat supply), N3 = L5 log-N
scaling on the fenced decode at 64/128 (offset-shrink prediction), N4 = P1b adapter
on HF test (P2 gap: domain vs data). Awaiting Tal's go.

## [2026-08-28 ~11:00] Tal go on N2/N3/N4 — all launched

- **N4 137898 (rtx6k n318, 2h_2g):** park-trained `sft_fenced_le16_ep10_adapter` on the
  MMReD-HF test files (8/16/32, 50 each) → `outputs/loramech/n4_park_on_hf/` —
  separates P2's domain gap from its data gap.
- **N3 L5 delta APPLIED** to both trainers (`--attn-logn-sref`; scaling =
  base_scaling·ln(S)/ln(S_ref) on the 28 LM decoder attn modules, captured pre-wrap;
  vision untouched; S measured per sample). S_ref computed with the tokenizer:
  fenced N=16 → **3398**; plain-ff N=32 → **6406**.
  **137899** fenced adapter × logN @ N=32/64/128 → `n3_logn_fenced/` (prediction: the
  −1.5 offset at 64 shrinks; N=32 no-harm check). **137900** ff_le32 × logN @ same
  cells → `n3_logn_plain/` (the original pre-registered L5/H3: N=64 ≥ +0.10 over
  0.293 AND N=32 within −0.02 of 0.313 = GAIN-confirmed rung).
- **N2 p1fence arm ADDED to probe_hahn** (imports build_fenced_messages/parse_layout
  from the P1 trainer — single source; build_block_mask(hide_cols=[]) + posreset;
  loci final + rep_t = room word in flip_t's replica; digit margins when plain arm
  absent; replay+ctrl floors). **137901** trained-adapter chain (a100, RUNNING) +
  **137902** frozen-comparator chain (PD, QOS cap) → `outputs/loramech/n2_hahn/
  {p1fence_ep10,p1fence_frozen}_N*/`. Pre-registered N2 prediction: rep_t (in-block
  verdict) flat in N for BOTH; final (read) decays for both, and the TRAINED read's
  decay aligns with the behavioral undercount onset (fine at 2×, offset at 4×).

## [2026-08-28 ~12:20] N2/N3/N4 LANDED (frozen N2 comparator still running) — all three predictions hit

**N4 (137898): P2's gap was DATA, not domain — and the park adapter sets the HF record.**
Park-trained `sft_fenced_le16_ep10_adapter` on the HF benchmark test, zero HF training:
**1.000 @N=8 (50/50!) · 0.820 @N=16 · 0.540 @N=32** — BEATS the HF-trained P2
(0.900/0.660/0.300) at every length. Domain shift ≈ nil; P2 was limited by its 320
skewed samples. The external-benchmark row for the paper is now the park-trained
fenced adapter with a PERFECT seq8 test score. → `outputs/loramech/n4_park_on_hf/`

**N3 fenced (137899): log-N scaling rescues the fenced read exactly where predicted.**
(100/cell slices of the same exam files)
| N | unscaled | +logN (S_ref 3398) |
|---|---|---|
| 32 | 0.867 (150-slice) | **0.950** (mae 0.06 — no-harm PASSED, actually improves) |
| 64 | 0.340 | **0.470** (+0.13; low band revives: g5–g8 from 0 → 3–5/7) |
| 128 | 0.230 | 0.230 (8× beyond rescue by scaling alone; low band still dead) |

**N3 plain (137900): the pre-registered H3 is NULL for the plain LoRA.** ff_le32+logN
(S_ref 6406): N=32 0.310 vs 0.313 (no harm ✓), N=64 0.250 vs 0.293 (< +0.10 band →
NULL), N=128 0.160. Scaling sharpens a read over clean verdict tokens (fenced) but
cannot rescue a read that must also do conditioned extraction (plain) — the H3 null
and the fenced rescue TOGETHER are the mechanism contrast.

**N2 trained chain (137901): THE mechanism table.** p1fence arm, ep10 adapter, L20:
| locus | N=8 | 16 | 32 | 64 | 128 | α |
|---|---|---|---|---|---|---|
| rep_t (in-block verdict) | 127.0 | 125.5 | 124.6 | 124.3 | 123.7 | **≈0.01 (FLAT)** |
| final (answer read) | 26.4 | 12.4 | 8.7 | 6.4 | 4.5 | **≈0.64 (decays)** |

The trained fence's per-frame verdict supply is exactly N-invariant (and ~4× the
frozen fenced supply's magnitude — training amplified the verdict written IN the
block), while the answer-position read decays Hahn-like; the behavioral undercount
onset (fine @2×, −1.5 offset @4×) coincides with the read signal falling from ~26
(in-window) to ~6. Supply fixed, read bound — in one table.
Frozen p1fence comparator (137902) running; full fig/α-CI analysis when it lands.
Caveat: N3/N4 cells are 50–100-sample slices; N=32 comparisons mix 150- and
100-sample slices of the same file (first-100 is class-stratified — fine, noted).

## [2026-08-28 ~12:45] P3 launched — fenced LoRA trained WITH log-N scaling active (Tal's go)

Delta: `--attn-logn-sref` now also applies in the TRAINING forward (train_loss), so the
readout calibrates to the compensated attention geometry instead of getting scaling
bolted on at eval. Adapter contract: must be evaluated with the same flag (noted in
help + config.json). **lm_p3_logn 137937 RUNNING (rtx6k n318, 24h_1g,
--time=10:00:00):** P1b recipe verbatim (roots le16, 10 ep, exclusions) + EXTRA
"--attn-logn-sref 3398"; in-job exams N8/16/32 (150 each, decode also scaled) →
`outputs/loramech/p3_fenced_logn/`. Question it answers: is the 4×–8× band reachable
without token-coded enumeration when the read TRAINS under N-compensation? Comparators:
P1b unscaled (1.000/0.893/0.867/0.340/0.230) and P1b+eval-only-logN (0.95@32,
0.47@64, 0.23@128). Follow-up on landing: zero-shot 64/128 eval with the flag.

## [2026-08-28 ~13:15] N2 COMPLETE (frozen comparator landed) — the mechanism figure is fully quantified

137902 COMPLETED (a100, 1h04). Fitted α at L20 (bootstrap 2000, this session; medians in
`outputs/loramech/n2_hahn/*_fig/medians.csv`, pairs in the run dirs):

| locus (p1fence layout) | trained ep10 | frozen |
|---|---|---|
| rep_t (in-block verdict) | **+0.009 [0.001,0.016]** — FLAT, magnitude ~125 | **+0.003 [−0.031,0.030]** — FLAT, magnitude ~55 |
| final (answer read), {8,16,32} | **+0.802 [0.66,0.97]** | +0.774 [0.70,0.86] |
| final, all 5 N | +0.603 | +0.322 (floor-limited: frozen read hits the bf16 band ~1.6 by N=32, flattening the fit) |

**The complete sentence, measured:** the fence makes the per-frame verdict channel
exactly N-invariant in BOTH frozen and trained models (α≈0, CIs straddle 0); training
amplifies the verdict content ×2.3 (55→125) and the read gain ×6 (4.5→26 @N=8) but the
READ decays with α≈0.8 in its measurable range in both — the same law as the frozen
plain joint read (0.72). Supply is fixable (fence), gain is trainable (LoRA), the
softmax read's 1/N is neither — it is the invariant of the whole campaign, and only
changing the CODE (token-enumeration readout, or partially the N-compensated logits:
eval-logN +0.13 @4×, P3 pending) moves the behavioral horizon.
- The trained read stays ABOVE the noise floor at N=128 (4.5 vs ~1.6) yet accuracy is
  0.23 — margin, not noise, is binding (matches the L1 margin slide).
- fig_hahn's fitter skips unknown arm names → α computed from pairs.csv directly
  (bootstrap, code in STATE history); medians.csv + figure PNGs archived per chain.
Running: P3 137937 (fenced trained WITH logN, rtx6k, ~ep2).

## [2026-08-28 ~16:20] P3 LANDED — perfect in-window, but compensation-in-training TRADES AWAY extrapolation

Job 137937 (rtx6k n318, 3h19) → `outputs/loramech/p3_fenced_logn/` run dir (best ep7
val 1.000, test_iid 0.993, 0 skips, disjoint ×3 ✓; adapter →
`checkpoints/sft_fenced_logn_adapter/`, CONTRACT: eval only with --attn-logn-sref 3398).

| exam (150/cell) | P1b (no logN) | P1b + eval-logN | **P3 (trained WITH logN)** |
|---|---|---|---|
| N=8 | 1.000 | — | **1.000** |
| N=16 | 0.893 (g3/g4 +1 artifact) | — | **1.000 (150/150 — artifact GONE)** |
| N=32 zero-shot | 0.867 | 0.950 | **0.420** (mid band g2–g8 dead, anchors perfect) |

**Reading:** N-compensation as a TRAINING prior yields perfect in-window calibration
(the g3/g4 wobble was calibration, now resolved) but the readout learns to DEPEND on
the compensated geometry — the analog slack that gave P1b its free octave is spent,
and at 2× the compensation overcorrects (mid-band collapse toward anchors, mae 0.65 =
small local errors). Eval-only bolt-on remains strictly better beyond the window.
**The L5 question is now closed in all three configurations:** plain+evalLogN = null;
fenced+evalLogN = +0.13 @4× only; fenced+trainLogN = in-window win, extrapolation
loss. The 4×–8× band is unreachable by attention re-scaling in any tested form →
token-coded enumeration (the method's scratchpad) stands as the unique measured route
past the read's horizon. Best-of-both recipe note for §6: TRAIN plain (P1b), APPLY
logN at eval (0.950/0.470) — the practical fenced-LoRA deployment config.
- P3 64/128 cells (flag on) submitted for the complete curve: 138006.

## [2026-08-28 ~17:00] CAMPAIGN LOGGED INTO RESULTS.md (Tal's instruction)

Three entries appended to RESULTS.md ([2026-08-26→27] P0+C1 regime baseline;
[2026-08-27→28] P1/P1b/P2/N4 fenced family; [2026-08-27→28] mechanism L1/N2/N3/L5/P3)
— the DRAFT entries earlier in this file are superseded by those. `INDEX.md` created
(canonical-run table). P3 64/128 (138006) still running; its numbers get appended to
the mechanism entry on landing.

## [2026-08-28 ~19:40] P3 64/128 LANDED (attempt 5 of 138006 after 4 preemptions) — CAMPAIGN DATA COMPLETE

0.310 @64 / 0.170 @128 (100/cell, pf 0) → full P3 ladder 1.000/1.000/0.420/0.310/0.170.
Below P1b+eval-logN (0.470/0.230) at every out-of-window length — trained-with-
compensation strictly dominated beyond the window; RESULTS.md addendum appended,
INDEX.md row finalized. No jobs running or queued. Open bench (next wave, needs Tal):
L2/L3 formal co-conditions, L4 attention photograph, H6 Q-first mechanism, g3/g4
generator check (moot for P3 but open for P1b's record).

# REDUX — campaign state (append-only, newest LAST)

Brief: `CAMPAIGN_BRIEF.md` (refined 2026-08-30). Scripts: `scripts/redux/`.
DRAFT RESULTS entries at the BOTTOM of this file — RESULTS.md is Tal's.

---

## [2026-08-30] CAMPAIGN START — brief refined, tasks.py written, C0 datagen submitted; GPU GATED on Tal's OK

Context read per §0: RESULTS [2026-08-22] Hahn + the four LORAMECH entries;
loramech brief/INDEX/STATE tail; probe_hahn.py, train_sft_fenced.py, eval_frozen.py,
gnnformer/data.py (build_count_prompt/parse_task_labels/probe_evidence), examdirs
REPORT. Gotchas honored: stride always, class dist + majority baseline per cell,
measured floors only, frames-first everywhere, count prompt byte-identical.

- `scripts/redux/tasks.py` WRITTEN — the one label module (see brief §2). exists/
  majority templates reuse the count opener line so the fenced parse_layout needle
  works for all tasks unchanged.
- Generator fix (live tree, not legacy): `datasets/mmred/generate_mmred_balanced.py`
  imported the long-gone `evaluations` package unconditionally; made lazy (only
  rooms_visited/co_occupancy asserts need it). 1-sample steps smoke: generated dir
  loads via load_mmred_sample, gold asserted, metadata carries target_character/room.
- Pool audit driving the datagen: N=8 fully covered by park (100/class); N≥16
  exists k∈{0,1} only 30/class (thin) and majority near-boundary counts absent
  (pool classes {0..8,12,16,24,...}) → `data/mmred_redux/` generated per brief §3.
- **C0 submitted (CPU, 4h_0g):** redux_datagen.sbatch × N∈{16,32,64,128} (job ids
  below). Exam builder runs when they land; count exams byte-reuse exam_ff_N*.
- **STOPPED for Tal's OK before any GPU cell** (C1 frozen ladders are the first).
  MANDATORY CHECKPOINT after C0+C1 (H-FROZEN reading) is separately gated per the
  campaign instructions.
  C0 datagen job ids: 138933 (N=16) / 138934 (N=32) / 138935 (N=64) / 138936 (N=128),
  all RUNNING on l40s-shared CPU slots at submit.

## [2026-08-30 ~12:30] C0 COMPLETE — pools + exam files built; still holding at the GPU gate

- Datagen 138933–36 all COMPLETED (14–31 min each): `data/mmred_redux/seq_len_{16,32,64,128}/`
  = 550/700/850/1000 exam dirs + 80 probe_base dirs each (engineered K, asserted).
- `outputs/redux/examdirs/` built (`build_exam_dirs.py`, seed-2 per-class shuffles,
  qa.txt-only loads, gold re-derived+asserted per dir, 0 bad anywhere): count = byte-
  reuse exam_ff; exists 150@8 / 100@N≥16, 50% k=0, k=1 = 1/3 of yes; majority
  150 (144 where strata×per rounds down) over d=±{1,2,4,8,16,32} clipped, all cells
  yes/no-balanced (majority baseline 0.500 exactly). Full tables in examdirs/REPORT.txt.
- N=128 majority spans k∈{32..96} only (spec caps |d|≤32 — no k=0/128 freebies at 8×;
  honesty flag: at N≤64 the d=N/2 strata include k=0/N which are prompt-readable).
- Contamination: N≥16 exam cells are from the FRESH redux pools (disjoint from every
  training root by construction). N=8 exists/majority cells draw from park seq8 (a
  training root): overlap vs P1b's train_dirs recorded (expected nonzero; harmless for
  frozen cells and for P1m, whose C2 exclusion list = ALL C0 exam files + LORAMECH
  exam files, making P1m clean by construction). exam_count_* inherit LORAMECH's
  proven disjointness.
- **GPU gate still closed — awaiting Tal's OK for C1 (two frozen-ladder jobs).**

## [2026-08-30 ~13:40] Tal: "go ahead with everything" — GPU phase OPEN; C1 instruments built + smokes running

Authorization: full pipeline, autonomous; the two rails stay binding (H-CTRL miss
HALTS; checkpoint READINGS pushed as notifications instead of silent stops).
- `scripts/redux/eval_tasks.py` written (plain ladder; greedy generate; per-gold +
  per-k tables; predictions CSV with task+k columns).
- `train_sft_fenced.py --tasks` delta applied (guarded): task-aware fenced messages
  via tasks.py, round-robin balanced task assignment, task-inferring dirs-file eval,
  frozen-eval mode (epochs=0, no adapter, no LoRA object). CPU checks: **count path
  byte-identical** (token-level, N=4 synthetic), exists/majority layouts parse under
  the unchanged "You will be shown" needle. probe_hahn import chain still compiles.
- Wrappers: `slurm/redux_eval_tasks.sbatch` new; fenced wrapper + TASKS knob.
- Smokes RUNNING on a100 n310 (2h_2g): rx_smoke_a 138952 (plain, exists+majority N=8,
  limit 5) and rx_smoke_b 138953 (fenced frozen mode, same cells, LONGN_LIMIT=5).
- C1 full plan on smoke pass: C1a plain ladder split into N≤32 (--time=4h) and
  N∈{64,128} (--time=8h) jobs; C1b fenced-frozen yes/no split likewise with
  LONGN_LIMIT=60 at N≥64 (fenced decode cost; documented cell-size shortfall).

## [2026-08-30 ~13:55] Smokes PASSED → all four C1 ladder jobs submitted

Smokes 138952/138953 (3 min each): both modes end-to-end, parse_fail 0, frozen-eval
mode confirmed (no LoRA object). 5-sample accs are slice noise, not read.
**C1 submitted:** 138954 c1a_short (plain, 9 cells N≤32, RUNNING a100) · 138955
c1a_long (plain, 6 cells N≥64, RUNNING a100) · 138956 c1b_short (fenced-frozen
yes/no, 6 cells N≤32, RUNNING a100) · 138957 c1b_long (fenced-frozen yes/no, 4 cells
N≥64, LONGN_LIMIT=60, PD). → `outputs/redux/c1_*/`. On landing: H-FROZEN reading +
push notification, then C2 proceeds (autonomous per Tal, halt rails armed).

## [2026-08-30 ~21:30] AGENT_PROMPT.md v2 read — C4b + H-SHARP added; probe deltas implemented

Tal's v2 prompt adds C4b (α under N-aware sharpness + decay-in-k, on the P1b adapter,
no training, parallel with C1) with hypothesis H-SHARP, and REINSTATES the mandatory
STOP after C0+C1+C4b (frozen ladder + H-FROZEN + H-SHARP readings) before C2 —
binding over the earlier blanket go. Brief §7 addendum written.
- probe_hahn deltas applied (guarded, no-flag byte-identical): `--gold-set` (explicit
  base-gold selector, overrides --max-gold; digit margins auto-skipped for gold>8)
  and `--attn-logn-sref` (trainer-identical scaling hook, per-forward seq len, attn
  modules captured pre-wrap). Compile OK. Report header carries both flags.
- `slurm/redux_c4b_k.sbatch` written: chain (ii) = 6 k-values × logn{on,off} one-k
  invocations from data/mmred_redux/seq_len_64/all_uniform (50/class incl. k=32).
- Reproduction smoke 138965 submitted (3 pairs, NO flags, p1fence+P1b @N=8, seed 0 —
  must reproduce the first-3 n2_hahn pair dnorms before any full C4b chain).
- C1 ladders still running (138954-57).

## [2026-08-30 ~22:15] C4b reproduction PROVEN → both chains launched

- Smoke 138965 FAILED (my heredoc submit lacked --gres; 3 min CPU-only, no GPU work
  lost); resubmit 138967 COMPLETED: first-3 pair dnorms BYTE-IDENTICAL to the
  n2_hahn/p1fence_ep10_N8 reference (49.4684/25.4728/22.2908, base norms equal) —
  the flagged probe is proven a strict superset of the anchor instrument.
- **Chain (i) 138968 RUNNING** (a100): p1fence+P1b, logN ON (S_ref 3398), N∈{8..128}
  → `outputs/redux/c4b_n/p1fence_ep10_logn_N*/`. Reference = n2_hahn (no rerun).
- **Chain (ii) 138969 RUNNING** (a100): N=64, k∈{1,2,4,8,16,32}→k+1, logn on+off,
  30 pairs/cell, redux pool → `outputs/redux/c4b_k/N64_k*_logn*/`.
- C1: shorts (138954/138956) already COMPLETED; longs (138955/138957) running.
  H-SHARP analysis (α refit + γ,C fit) runs on chain landings; checkpoint after all.

## [2026-08-31 ~00:30] C1 COMPLETE — H-FROZEN READING (H-SHARP pending C4b)

Jobs 138954/55/56/57 all COMPLETED (0h29/1h28/0h35/1h27). Run dirs
`outputs/redux/c1_plain_{short,long}/`, `c1_p1fence_frozen_{short,long}/`.

**Frozen plain ladder (full cells, pf 0 everywhere):**
| task | N=8 | 16 | 32 | 64 | 128 | maj-floor |
|---|---|---|---|---|---|---|
| count | 0.200 | 0.153 | 0.087 | 0.067 | 0.080 | 0.06–0.11 |
| exists | 0.833 | 0.740 | 0.640 | 0.540 | **0.490** | 0.500 |
| majority | 0.547 | 0.528 | 0.573 | 0.542 | 0.521 | 0.500 |

- H-CTRL frozen leg: count @8 = 0.200 vs anchor 0.219 (n=150, ~0.6σ) ✓ HOLDS.
- exists per-k (the retrieval anatomy): k=1 recall 13/25→6/16→1/16→1/16→**0/16**
  across N; k=8 recall 15/16→…→1/11; k=0 stays ~0.94 — the frozen reader can do OR
  in-window and drifts to always-no with N. Retrieval DECAYS with length, frozen.
- majority: chance at every N incl. N=8 — the frozen model has no ratio read at all.

**Frozen reader over the FENCED layout: DEGENERATE always-no at every N** (prediction
tables logged: 100% "no" except 16 yes at N=8) — the raw model cannot operate the
fenced layout zero-shot. INSTRUMENT NOTE: the c1b_long 60-sample cells were
stratum-ordered head-slices (my exam files were written in stratum order — the
K0-trap in new clothes; caught via the prediction tables); their headline accs
(0.833/0.400) are the slice no-fractions of an always-no responder, NOT model skill.
No rerun needed — the degeneracy verdict is slice-independent. All exam files are
now row-shuffled (seed 4; SETS unchanged; full-file C1a/C1b-short cells unaffected).

**H-FROZEN VERDICT (pre-registered, two-sided): frozen plain exists @128 = 0.490 =
chance → the MAX row does NOT come free; fence+TRAINED reader is what's under test
(the fenced-frozen leg shows the fence alone gives nothing without a trained reader).
The C2 mixture proceeds as designed — nothing in the frozen data argues for a recipe
change.** Checkpoint STOP still pending C4b (H-SHARP) per the v2 prompt.

## [2026-08-31 ~01:10] C4b chain (i) COMPLETE — H-SHARP(a) REFUTED as pre-registered, with the margin mechanism measured

Job 138968 (a100, 1h20) → `outputs/redux/c4b_n/p1fence_ep10_logn_N*/`.
Flag-effectiveness VERIFIED before reading the verdict (per the band's failure
protocol): base norms shift ~1.3% vs the OFF reference (scaling reached sdpa;
provenance in report headers), yet:

| L20, p1fence+P1b | α (bootstrap CI) | medians 8→128 |
|---|---|---|
| read (final), logN ON | **+0.605 [0.55,0.69]** | 27.1→4.7 |
| read (final), OFF (n2 ref) | +0.603 [0.53,0.68] | 26.4→4.5 |
| verdict (rep_t), ON | +0.005 [−0.005,0.011] | ~125 flat |

**H-SHARP(a) REFUTED: α does not drop below 0.6 under logN (band demanded <0.3).**
Per the pre-registered protocol, the C4b interpretation stops here; chain (ii)'s
γ fit will be reported as DESCRIPTIVE only.
**Measured margins (the honest "report the margins" clause) — and they explain N3:**
median gold-digit margin @N=64: −2.89 (off) → −0.10 (on); @128: −5.88 → −1.20;
@8/16/32 unchanged-to-slightly-up. logN acts on the DECISION MARGINS, not on the
state-difference geometry — sensitivity-α and margin-calibration are SEPARABLE
quantities, and the behavioral rescue (0.34→0.47 @64, none @128) tracks the margin
sign exactly. The "α is the wrong quantity for SUM-type" thesis survives in an
unexpected form: α was insensitive to the intervention that changed behavior.

## [2026-08-31 ~02:00] C4b chain (ii) COMPLETE → MANDATORY CHECKPOINT (C0+C1+C4b all in). STOPPED FOR TAL'S REVIEW BEFORE C2.

Job 138969 (a100, 3h20) → `outputs/redux/c4b_k/N64_k{1..32}_logn{off,on}/` (30 pairs/cell).

**Decay-in-k (descriptive per the (a)-refutation protocol):** L20 read medians @N=64:
k=1: 10.5 · 2: 7.6 · 4: 5.95 · 8: 4.19 · 16: 3.13 · 32: 2.91 (logn on ≈ off everywhere).
Fit Δ ∝ (k+C)^−γ: **γ = 0.390 [0.375,0.402], C = 0.0** — identical under logN. Had
H-SHARP(b) been evaluated, it would MISS (band γ≥1, CI excluding 0.5 — measured CI
excludes 0.5 from BELOW). The read's one-more-frame sensitivity decays like k^−0.39
(≈1/√k), far slower than the share-code (k+C)^−2 prediction — logged as measured, no
forcing.

### CHECKPOINT SUMMARY (the three readings)
1. **H-FROZEN:** frozen plain exists decays to chance @128 (k=1 recall 0/16); majority
   at chance at every N; fenced-layout frozen reader = degenerate always-no → the MAX
   row requires the trained reader; C2 mixture as designed.
2. **H-SHARP: (a) REFUTED** (α unchanged 0.605 vs 0.603 under verified-active logN);
   margins measured per protocol — logN moves DECISION MARGINS (−2.89→−0.10 @64),
   not Δh geometry; α and margin-calibration are separable. (b) descriptive γ=0.39.
   Interpretation stopped per the pre-registered failure clause.
3. **H-CTRL (frozen leg): HOLDS** (count @8 0.200 vs anchor 0.219, ~0.6σ).
Instrument notes: exam files now row-shuffled (seed 4, sets unchanged); c1b_long
60-slices were stratum-ordered head-slices (always-no degeneracy verdict unaffected);
heredoc-sbatch --gres lesson (3 min lost).
**NO further GPU jobs submitted. Awaiting Tal's review to open C2 (P1m multitask
trainer) → C3 → C4.**

## [2026-09-20] REDUX v3 authorized (Tal, "run all of those") — C2a/C2b multitask readers on the gated trainer; brief addendum written; smokes next
- [21:36] guards relaxed (oracle gate any task; nfree for exists/majority via tasks.build_prompt_nfree, CPU-verified digit-free, count path untouched); REDUX v3 smoke 153875 submitted (anchor + 1-ep multitask smokes).
- [23:21] SMOKE 153875 PASSED: trainer anchor P1b exam_ff_N8 = 1.0000 after the guard/nfree edits; 1-ep multitask smokes (ungated, gated) train and evaluate all three tasks with yes/no parsing (per-gold gno/gyes lines present).
- [23:21] SUBMITTED C2b gated 153961 (24h_1g) and C2a ungated 153962 (4d_1g), l40s-shared, --time 23h: --tasks count,exists,majority --nfree-prompt --limit 2700 --epochs 5, roots seq8+seq16, EXCLUDES = 24 files (verified as 24 separate flags on the A3 command lines), in-job LONGN on the 9 redux exams N8/16/32 (N64/128 eval-only later on >=80 GB). NOTE (review item 11): the two roots hold ~1230 dirs before exclusion, so --limit 2700 = "use all"; each task gets ~1/3. Matched-budget count-only controls submitted: gated 153963, ungated 153964 (--limit 300, same roots/excludes/epochs).

## [2026-09-21 ~10:05] MISHAP + RESUBMIT: C2b/C2a (153961/153962) ran as COUNT-ONLY, N-IN-TEXT readers — `EXTRA="--tasks count,exists,majority --nfree-prompt"` was comma-split by sbatch to `--tasks count` (everything after the first comma, including `--nfree-prompt`, silently dropped; config.json: tasks=count, nfree_prompt=False). The CLAUDE.md footgun, struck through the EXTRA knob this time.

What the two runs ARE (kept, not cited as multitask): 5-ep, 728 samples (seq8+seq16, 24 excludes), count-only,
**N in the prompt text**, in-job ladders on the fresh redux COUNT exams (150/N):
| reader | N=8 | N=16 | N=32 | notes |
|---|---|---|---|---|
| c2b_gated (hard oracle gate, N-in-text) | 0.993 | 0.887 | 0.780 | k≤8 at N=32: 100/107; k16/k24 0/22, k32 11/11; k=4 dips (5/14 @16, 7/12 @32) |
| c2a_ungated (fenced, no gate, N-in-text) | (pending in log) | 0.893 | 0.567 | k≤8 at N=32: 48/107 — the ungated read's 4× collapse; k12/16/32 11/11 each (in-range counts) |
| c2b_countctrl (hard gate, **N-free**, 480 samples, 153963 ✓) | 0.980 | 0.913 | 0.767 | the intended matched-budget gated control |
| c2a_countctrl (no gate, **N-free**, 480 samples, 153964 ✓) | 0.973 | 0.713 | 0.267 | the intended matched-budget ungated control; N-free ungated collapses hardest at 4× |
Their exists/majority LONGN lines (c2b: exists 1.000 at N=8/16/32; majority N=16 0.507 = all "no") are a
COUNT reader answering yes/no prompts — not evidence for H-R1..R6; ignore. Side note for later: the ungated
N-in-text reader (0.567) beats the ungated N-free one (0.267) at N=32 — the S7 prompt-N calibration channel,
again. Recurring **k=4 glitch** in 5-ep hard-gate readers on this 728-sample set (a3_Bhard N32 k4 1/12;
c2b_gated N16 5/14, N32 7/12) — reader-side, two different exam pools; not chased.
**Resubmitted correctly**: `slurm/sparse_train.sbatch` gained `TASKS` (SPACE-separated, comma-joined inside
the script) and `NFREE=1` knobs; submitted C2b_mt **154067** (gated) and C2a_mt **154068** (ungated),
rtx6k-shared n318 (--exclude=n317), 24h_1g, --time 10 h, same roots/limit/epochs/excludes, 9 redux exams;
**verified on the submitted command lines**: `--tasks count,exists,majority --nfree-prompt` present, 24
`--exclude-dirs-file` flags. Outputs: `outputs/redux/c2b_gated_mt/`, `outputs/redux/c2a_ungated_mt/`.
Rule added to the wrapper header and to memory: grep the submitted cmd in `logs/<name>-<id>.out` right after
every submit.
- [~10:40] 153961/153962 finished (3h23 each). Count-only N-in-text pair, complete count ladders: gated
  0.993 / 0.887 / 0.780, ungated 1.000 / 0.893 / 0.567 (N = 8/16/32). Their ZERO-SHOT answers to the
  other two prompts (never trained on them; descriptive only): gated reader — exists 1.000 at every N
  (a block counter gets "any block?" for free), majority 0.49–0.51 (all "no"); ungated N-in-text reader —
  exists 0.79 / 0.70 / 0.60, **majority 0.980 / 0.965 / 0.927**. The ungated reader passes the coarse
  comparison at 4× while failing exact count there (0.567): majority with these exam k-spreads is a
  low-resolution question, count is not. Foreshadows the multitask result — under the N-free prompt the
  hard-gated reader cannot answer majority at all (it sees k blocks and no N), so the N-in-text secondary
  eval from the amendment is the one that matters for majority.
  Per-k of that zero-shot majority: ungated errors sit ONLY at the boundary (N=16: k6 16/18, k7 15/18; N=32:
  k14 12/15, k15 7/15; every other k 15/15 or 18/18) — a resolution pattern, the read separates k from N/2
  except within ~1–2 frames of it. The gated reader's failures are the k > 8 cells (k17–k32: 7/75) —
  the k-wall, not the boundary. The k=4 dip does NOT appear in the N-free count controls (14/14, 12/12).

## [2026-09-21 11:30] C2b_mt / C2a_mt LAND (154067/154068, 1h59 each; 243/243/242 samples per task, 5 ep, N-free) — exists trivial for both; majority: ungated fails only at the boundary, gated N-free breaks at N=32 exactly as pre-registered; COUNT readers are under-trained at one third of the budget (gated one shifted +1 mid-range) — count bands not testable from these readers

| task | reader | N=8 | N=16 | N=32 | pattern |
|---|---|---|---|---|---|
| count | gated_mt | 0.527 | 0.420 | 0.247 | **pred = k+1 for k = 4…7 at N=8 (17/17 at each k), k = 3…7 at N=16; k ≤ 2 and k = 8 exact** — in-length miscalibration (TEST_IID 0.79, val 0.78) |
| count | ungated_mt | 0.840 | 0.740 | 0.173 | k=6 → 5 hole at N=8; k≤8 at N=16 85/124 = 0.69; N=32 k≥2 all wrong |
| count | gated count-only control (480, 153963) | 0.980 | 0.913 | 0.767 | **k≤8: 124/124 at 16, 106/106 at 32** — H-R4 holds at N ≤ 32 on the control |
| count | ungated count-only control (480, 153964) | 0.973 | 0.713 | 0.267 | k≤8: 81/124 = 0.65 at 16, 40/106 = 0.38 at 32 — H-R1's decay, already below 0.9 at 2× |
| exists | both | 1.000 | 1.000 | 1.000 | k=0 recall 50/50, k=1 recall 16/16 at every N (H-R5 MET so far; H-R2's decay clause not yet visible) |
| majority | gated_mt | 0.960 | 0.861 | **0.373** | N=32: says "yes" for k=12–15 (gold no), "no" for k ≥ 20 (gold yes); |d|≤8 strata 41/120 = **0.34 ≤ 0.65 → H-R6 MET** |
| majority | ungated_mt | 0.973 | 0.868 | 0.780 | errors ONLY at the boundary: N=16 k9 3/18, k10 14/18; N=32 k17 2/15, k18 2/15, k20 8/15, everything else 15/15 |
H-R3 at matched ratio r ∈ {0.25, 0.375, 0.625, 0.75} (ungated): **0.96 / 0.94 / 0.88** at N = 8/16/32 — flat within
0.1 so far; the r = 0.625 cell is the one sliding (25/25 → 14/18 → 8/15). The N=64/128 cells decide.
**Count caveat (important):** the multitask readers did NOT reach in-length count competence: the gated one
answers k+1 across the middle of the range at N=8. This is a budget/interference artifact (243 count samples
against 480 in the controls; the gate-trained majority label "yes iff visible k ≥ 5" shares the answer row), not
a mechanism finding; the count rows of H-R1/H-R4 are scored on the count-only controls above. Fixing it needs
either every dir labelled for all three tasks (sampler change: 3 × 728 samples) or more epochs — proposed, not run.
**Submitted (pre-authorized N=64/128 eval-only, rtx6k n318, 24h_1g, --time 4 h):** gated_mt **154212** →
outputs/redux/c2b_gated_mt_eval64_128, ungated_mt **154213** → outputs/redux/c2a_ungated_mt_eval64_128; 6 exams
(count/exists/majority × N64/N128), N-free prompts, --limit 24 scaffold; command lines verified (--tasks
count,exists,majority --nfree-prompt, 6 --eval-dirs-file).

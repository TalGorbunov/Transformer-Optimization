# ARMOR campaign — STATE (append-only; newest last)

- [2026-08-22] CAMPAIGN CREATED. Brief: outputs/armor/CAMPAIGN_BRIEF.md (read first —
  pre-registered bands for A/B/C live there). Authorization: Tal 2026-08-22, GPU
  submissions without per-job asks, discipline rules binding (smokes in _scratch,
  explicit --time, no comma --export, no pip, stop-on-artifact-suspicion).
  Context reviewed: recagg RELATED_WORK (TOP THREATS 1–7), recagg INDEX + STATE full,
  RESULTS.md [2026-07-31b/c]…[2026-08-11c] + [2026-07-24] MLVU entries.
  Cluster at start: a100-public 2 GPUs free (n310), l40s n314 1 free, rtx6k full,
  h200 n315 DOWN. Plan: D warm-up first (validates harness), then A (new instrument),
  B (needs N=128 captures submitted early — long pole), C last.
  Hygiene-D reproduction anchor verified: job 127599 cmd recovered from
  logs/probe_qkv_swap-127599.out (LIMIT=25, L16, seq_len_128/all_uniform, 7:49 elapsed);
  pool has 510 dirs → n=80 well-covered, ~25 min estimated.

- [2026-08-22 ~13:30] D + A LAUNCHED, B captures submitted.
  D: smoke 136123 (a100 2h_2g, LIMIT=5 N=8) PASSED end-to-end; real run 136124
  (a100 2h_2g --time=1:45, LIMIT=80, N=128 longN_park, same protocol/command as 127599)
  -> outputs/armor/qkv128/.
  A: new instrument scripts/armor/probe_hahn.py + slurm/armor_probe_hahn.sbatch.
  Design decisions (recorded before any full run):
   * gold <= 8 slice (longN pools are K-bucketed with K0..8 dense, 30 dirs each = 270/N):
     answers single-digit at EVERY N -> house digit protocol (digit_ids argmax) valid and
     N-comparable; flip k->k+1 stays single-digit. Class dist reported per cell.
   * pairs rendered FRESH from states.json via the deterministic park renderer
     (render_mmred.render_frame is jitter-free); byte-identity of all non-flipped frames
     HARD-ASSERTED per pair; audit renders kept for pair 0. CPU pair-test on N=16 pool:
     5/5 pairs OK (evid k->k+1, ctrl k->k verified).
   * three arms: plain (deployed count prompt, joint causal - A2 margins from same fwd),
     repjoint (Q-first replica layout, base replica-probe mask, no posreset - per-frame
     read loci exist without fence), fenced (blockfence+posreset, A3 deployed).
   * measured noise controls: replay (determinism), matched answer-PRESERVING ctrl flip
     (C moved wrong-room->wrong-room, same frame - Hahn bound applies to it too),
     fenced-perm (non-flip slots permuted = content-preserving under fence+posreset ->
     bf16 reduction-reorder floor, per the canary lesson).
   * hs loci: tuple indices 16/20/28 (28 = post-final-norm); final position + anchor
     (seq-1-ANCHOR_OFFSET) + flipped frame's replica room token.
  A smoke: 136125 (a100 2h_2g, LIMIT=4 CONTROLS=2, N=8 park) -> _scratch.
  B (early, the long pole): N=128 leaf captures submitted - 136126 hf128_512
  (seq_len_128_test, LIMIT=50, @512) + 136127 park128 (longN seq_len_128, LIMIT=100),
  l40s-shared 24h_1g --time=4h, probe_tree_ninv capture-only, byte-same protocol as
  p1_captures (cmds copied from logs 133625/133627) -> outputs/armor/p1_captures_128/.
  Buitrago & Gu mechanics confirmed from the PDF (arXiv:2507.02782): SP = init state <-
  final state of ANOTHER sequence (stop-gradient, most effective variant); noise arms =
  IID N(0,1) / fitted Gaussian matched to real state stats; ~500 post-training steps
  sufficed in their LM setting.

- [2026-08-22 ~21:30] ALL THREE EXPERIMENTS IN FLIGHT + C instrument built.
  D 136124 running (ETA ~25 min). A cells 136128-136132 (N8 DONE in 5:49 —
  50 pairs, skip 8, healthy gold hist g0..g7; joint L20/L28 final dnorm ~10.5/15.2
  vs fenced rep_t ~38-40; N16/32/64 running, N128 queued). Figure/verdict script
  scripts/armor/fig_hahn.py ready (slope fits + bootstrap CI + A2 crossing bands).
  First-capture bonus check from the N8 smoke, worth keeping: fenced rep_t is
  BIT-IDENTICAL (dnorm exactly 0) under permutation of the other frames — the
  fence provably isolates the per-frame channel; replay floor exactly 0; the
  bf16 reduction floor at the fenced final/anchor loci is 0.6-2.9 (vs joint
  signal 10-25 at N=8).
  B: 136139 (hf512) + 136140 (park) running on 4h_0g — 9 arms each
  (R1 + {R2,R3}x{base,sp,noise,spnoise}), exact P2 recipe/seeds/protocol,
  eval @{32,64} zero-shot (the RESCUED band needs only N=64; N=128 eval rerun
  follows captures). Fixed capture no-op: probe_tree_ninv's NS table stops at
  64 (the 21s jobs 136126/136127; dead dirs moved to _scratch/armor_dead_runs);
  armor wrapper scripts/armor/capture_leaf.py registers the 128 row; resubmitted
  136137 (hf128_512) + 136138 (park128).
  SP design decisions (pre-registered): sp = concat-carry with COUNT-CONSISTENT
  labels (borrowed detached state + carried count added to target, rejected if
  total > cls support 16 -> label range UNCHANGED, state coverage extended =
  the clean separation of the two hypotheses); noise = fitted Gaussian on
  running final-state stats with CURRENT-count labels (paper semantics) —
  pre-registered expectation: ill-posed for the pure integrator R3 (carried
  pseudo-count unrecoverable), possibly fine for the gated GRU; a NEGATIVE
  noise effect on R3 is itself mechanism evidence, not a bug.
  C: instruments built — scripts/armor/mlvu_caption.py (per-frame question-
  conditioned records, 32f = round(linspace(0,127,32)) VERIFIED identical to the
  lookagain_N32 judge keys and the [2026-07-11c] frozen protocol; judge-agreement
  reported as fidelity proxy) + scripts/armor/mlvu_compile_execute.py (qwen14b
  venv_arch, armC sandbox imported read-only, P2b nearest-option MCQ rule).
  Honesty notes recorded in the script docstring: single question type ->
  compile near-degenerate here (this cell measures the COMPOSED system, not
  compile generality); baseline saw options in-prompt, composed never does;
  32f evidence-delivery ceiling applies. Caption smoke 136142 queued (LIMIT=3).
  Frozen baseline provenance for the band: docs/archive/RESULTS_pre_fencing.md
  [2026-07-11c] — MCQ 0.282 @32f (0.393 @128f dense), reused not rerun.

- [2026-08-22 ~21:45] D COMPLETE (136124, 31:59, n=80, skip 0) — CANONICAL:
  outputs/armor/qkv128/20260822_210658/. d' CC 6.86+-0.32 / CD 5.41 / DC 3.31 /
  DD 4.09 (ordering identical to the n=25 run); tally CC 0.870 / CD 0.650 /
  DD 0.400 / DC 0.345 — the n=25 non-monotone tally (CD 0.708 > CC 0.523) was
  quantization noise as suspected; at n=80 the staircase is monotone. Gains in
  the [2026-07-31b] convention: q-only +1.32 (stays ~+1 at every N), kv-only
  -0.78 (monotone decline continues: +0.97/+0.64/+0.09/-0.47/-0.78 @8-128),
  interaction +2.23 (~+2 at long N). The N=128 qkv cells are now citable;
  waterfall rungs 2-4 @128 read 0.400/0.650/0.870 (monotone). INDEX.md row added.
  A cells: N8/N16/N32 DONE (joint plain L20 final decays 10.5 -> 4.9 -> 3.2,
  crude alpha ~0.86 — inside the pre-registered band so far; fenced rep_t FLAT
  38.6 -> 39.2). C: caption smoke PASSED (judge-agree 1.0, parse-fails 0,
  3 questions); full 206-question caption run submitted 136145.

- [2026-08-22 ~22:10] C 32f CELL COMPLETE — pre-registered verdict: NO-GO on the
  band, WITH A MEASURED EXPLANATION THAT REFRAMES IT.
  Runs: captions 136145 (22:55, 206 q x 32f, outputs/armor/mlvu/20260822_213354_
  captions/) -> compile+execute 136146 (3:18, outputs/armor/mlvu/20260822_220317_
  compiled/ = CANONICAL 32f cell).
  Numbers: MCQ nearest-option 0.301 (band needed >=0.332; frozen baseline 0.282);
  open exact 0.197, MAE 1.62, mean pred 1.35 vs mean gold 2.93; exec-fail 0.015;
  by-gold g1 0.70 / g2 0.42 / g3 0.18 / g4 0.18 / g5 0.00.
  Perception layer: VLM per-frame recall 0.928 on judge-positive frames
  (precision 0.727), judge-agreement 0.986 overall, 0 parse fails.
  LOCALIZATION (the finding): (1) compile+execute is FAITHFUL - 97.0%
  (197/203) of predictions exactly equal the run-count of the system's own
  delivered evidence; (2) delivery ceiling MEASURED: an ORACLE judge-evidence
  run-counter at 32f scores 0.267 MCQ - BELOW the frozen 0.282 - because 32f
  sampling delivers 282 visible frames vs 603 gold instances (48/206 questions
  get ZERO evidence). At 32f the band was unreachable BY CONSTRUCTION for any
  faithful counter; the frozen baseline's 0.282 sits at the guessing/prior
  level, not above the faithful ceiling by counting.
  => the honest external-validity statement is delivery-limited, matching
  [2026-07-11c] exactly (its cure was density: frozen 0.282 -> 0.393 @128f).
  DIAGNOSTIC EXTENSION (running): composed @128f dense (caption job 136149,
  ~26k frames) vs the archived 0.393 dense baseline - the cell where delivery
  stops binding and aggregation is actually exercised. The 32f band verdict
  stands as pre-registered; the 128f cell is labeled post-hoc extension.

- [2026-08-22 ~22:30] EXPERIMENT A COMPLETE — all 5 cells landed (136128-136132,
  50/50/50/50/40 pairs, total GPU ~1h50m). CANONICAL: outputs/armor/hahn/
  20260822_211457_N{8,16,32,64,128}/ + figure/verdict 20260822_211457_fig/
  (hahn_figure.png = THE theory-to-model figure; hahn_grid.png; medians.csv;
  verdict.json).
  A1 VERDICT vs pre-registered bands:
   * JOINT: plain L20 final-position ||dh|| decays 10.5 -> 4.9 -> 3.2 -> 2.1 ->
     1.3 with alpha = 0.72 [0.63, 0.78], r2 = 0.98 -> INSIDE the [0.7, 1.3]
     band at the gate-landing layer (L20; TRUNC located the per-frame gate
     write at L12-19). L28 (post-final-norm) alpha = 0.58 [0.49, 0.68] - decays
     but sub-band (norm compression); L16 final flat AT floor (0.82 -> 0.69 —
     the flip signal has not reached the final position that early). Verdict:
     BAND MET at L20; the O(1/N)-class decay is measured inside the production
     VLM for the first time.
   * FENCED contrast: rep_t alpha = +0.009 / -0.007 / +0.016 (L16/L20/L28),
     every CI containing 0, all |alpha| < 0.05 -> |alpha| < 0.2 band MET
     decisively. Per-frame supply is N-INVARIANT under fence+posreset.
   * HEADLINE BONUS (stronger than pre-registered): at N=128 the joint
     evid-flip dnorm (1.33) is AT the measured bf16 reduction floor (~1.2-1.4)
     and equal to the answer-PRESERVING ctrl flip (1.1) - the answer-relevant
     single-frame signal is numerically indistinguishable from noise at 128
     frames, while fenced rep_t sits ~30x above floor, flat.
   * Mechanism texture (free intermediate rungs): repjoint final alpha ~0.44;
     per-frame readers over JOINT supply (repjoint rep_t) alpha ~0.21-0.25 —
     per-frame READERS alone recover only part of the invariance; the fence
     (clean supply) is what zeroes the slope. Matches the D/qkv story (query
     repair alone insufficient).
  A2 VERDICT: band FAILED-AS-DEFINED, with the saturation caveat logged: the
  frozen model is already below 50% at N=8 on this slice (acc 0.28), so the
  median margin STARTS negative (-0.53 @8) and the "crosses the floor" event
  fires at the smallest N measured; accuracy reaches chance (1/9) at N~32-64
  (0.18 @32, 0.12 @64, 0.10 @128); crossing ratio 8/32 = 0.25 -> outside the
  factor-2 band. The informative A2 read is the MONOTONE margin slide
  (-0.23 @16 -> -1.91 @128, ctrl-jitter floor 0.02-0.24) alongside the
  accuracy decay - the margin instrument cannot "cross from above" on a task
  the joint model already fails in-length; instrument limitation, not a
  contradiction of the theory.
  Controls (all cells): replay floor EXACTLY 0 everywhere; fenced rep_t
  bit-identical (dnorm 0) under other-frame permutation in every cell = the
  fence provably isolates the per-frame channel; class dists healthy g0-g8.

- [2026-08-22 ~22:50] EXPERIMENT B, HF@512 PRIMARY CELL COMPLETE (136139, 1:33:36;
  outputs/armor/b_statepass/*_hf512/ = CANONICAL). Pre-registered band verdict:
  **NOT RESCUED** — best intervention arm R3_ssm_sp EM_reg 0.800 [0.752,0.848]
  @N64_zs, below the 0.90 band (R2_gru_sp 0.696; bases 0.284/0.260; R1 reference
  1.000). BUT the mechanism DECOMPOSITION is the real result:
   * State-passing WORKS on the leg its theory owns: within-trained-range
     (gold<=16) accuracy @N=64 zero-shot jumps 0.351 -> 0.941 (GRU) and
     0.346 -> 0.962 (SSM) — the Buitrago "unexplored states" drift is real and
     SP fixes it at 4x training length (canary also improves: GRU 0.14 -> 0.06,
     SSM 0.04 -> 0.00).
   * The READOUT-RANGE cap survives untouched: GRU out-of-range (gold>16)
     recall is EXACTLY 0.000 in every arm (hard cliff at the label support;
     max-correct-count@rec>=0.5 = 15-16 = the fit-range max). The SSM's
     near-linear integrator leaks slightly past (out-of-range 0.108 -> 0.338
     with SP, mc50 21). EM stays under the band because 26% of the N=64 pool
     has gold > 16. This is the Yehudai range-cap prediction measured inside
     the SP control: state-coverage interventions cannot extend the emission
     range by construction (labels were held <=16 BY DESIGN — the clean
     separation of the two hypotheses; recorded pre-run).
   * Noise arm: pre-registered ill-posedness CONFIRMED for the pure integrator
     — R3_ssm_noise collapses even in-length (0.984 -> 0.728 @N8_in) and to
     0.080 @N64_zs; GRU tolerates noise (can gate it away) but gains nothing.
   * N32_zs for context: sp arms 0.904/0.936 (>=0.90 at 2x length; the band
     length 4x is where the range cap bites).
  Reviewer answer this buys: "we ran the Buitrago interventions; they rescue
  exactly the state-coverage leg (in-range 0.94-0.96 at 4x length) and leave
  the range cap - the leg our necessity argument rests on - fully intact."
  Caveat noted: for RNN heads, SP-by-concatenation is equivalent to training
  on longer virtual sequences with detached gradients (that IS the paper's
  intervention; label range was held fixed, so the state-coverage vs
  label-range confound is controlled).
  Park twin (136140) still running; N=128-eval rerun awaits captures
  (136137/136138 still queued behind other users on l40s).

- [2026-08-22 ~23:00] B PARK TWIN COMPLETE (136140, 1:34:23; outputs/armor/
  b_statepass/20260822_212000_park/). Replicates the HF decomposition on the
  second domain: N64_zs base 0.401 (GRU) / 0.292 (SSM) -> SP 0.683 / 0.675;
  mc50 pinned at 16 = label support in every SP arm; R1 0.983 (p_frame 0.9997).
  Park pools have uniform golds up to N (more out-of-range mass than HF), so
  the range cap costs more EM — same two-leg structure, NOT RESCUED per band.
  (INDEX dir-name typo fixed: hf512 cell is 20260822_212000_hf512.)

============================================================================
DRAFT RESULTS.md ENTRIES (for Tal to review — NOT appended to RESULTS.md)
============================================================================

--- DRAFT 1 (Hygiene D) ---

## [2026-08-22] ✅📊 N=128 Q/KV SWAP RERUN (n=80) — the [2026-07-31c] parenthesized cells are now citable and MONOTONE: tally CC 0.870 > CD 0.650 > DD 0.400 > DC 0.345; the [2026-07-31b] scaling law extends to N=128 (q-only +1.32, kv-only −0.78, interaction +2.23)

> Job 136124 → `outputs/armor/qkv128/20260822_210658/` (same script/protocol as
> 127599: L16, steps, `mmred_longN_park/seq_len_128/all_uniform`, LIMIT=80, skip 0).

| N=128 (n=80) | qC_kvC | qC_kvD | qD_kvC | qD_kvD |
|---|---|---|---|---|
| d′ | 6.86±0.32 | 5.41±0.09 | 3.31±0.26 | 4.09±0.02 |
| gate→tally | 0.870±0.037 | 0.650±0.055 | 0.345±0.064 | 0.400±0.042 |

**Readings.** (1) The n=25 tally anomaly (CD 0.708 > CC 0.523) was quantization
noise as flagged — at n=80 the staircase is monotone and CC is cleanly on top.
(2) Full gain series across N (8/16/32/64/128): q-only +0.92/+1.01/+1.24/+1.05/
+1.32 (flat ~+1); kv-only +0.97/+0.64/+0.09/−0.47/−0.78 (monotone decline —
value-repair-alone is actively harmful at scale); interaction +1.01/+2.11/
+2.03/+1.87/+2.23 (~+2 at long N). The query side stays the binding constraint
at the largest N; both sides must be repaired jointly — the fence's job.

--- DRAFT 2 (Experiment A) ---

## [2026-08-22] ✅📊 HAHN O(1/N) MEASURED INSIDE THE PRODUCTION VLM — paired one-frame flips: joint answer-position sensitivity decays as N^−0.72 [0.63,0.78] (r²=0.98, band [0.7,1.3] MET at L20) and is AT the measured bf16 noise floor by N=128, while the fenced per-frame supply is exactly flat (α = 0.00 ± 0.02, ~30× above floor) — the theory-to-model contrast figure

> New instrument `scripts/armor/probe_hahn.py`; jobs 136128–136132 →
> `outputs/armor/hahn/20260822_211457_N{8,16,32,64,128}/` + figure/verdict
> `…_fig/hahn_figure.png` (50/50/50/50/40 pairs, gold≤8 slice, ~1h50m GPU).
> Pairs rendered fresh from states via the deterministic park renderer, ONE
> non-evidence frame toggled to evidence (gold k→k+1); byte-identity of all
> other frames hard-asserted per pair. Noise floors MEASURED per the canary
> lesson: replay (exactly 0 everywhere), answer-preserving ctrl flip (same
> frame, wrong-room→wrong-room), fenced block-permutation (bf16 reduction
> floor; and fenced rep_t is BIT-IDENTICAL under it — the fence provably
> isolates the per-frame channel).

| locus | α (Δ ∝ N^−α) | 95% CI | r² | band |
|---|---|---|---|---|
| joint (plain) L20 final | **0.72** | [0.63, 0.78] | 0.98 | [0.7,1.3] **MET** |
| joint (plain) L28 final | 0.58 | [0.49, 0.68] | 0.95 | below (post-norm compression) |
| per-frame readers over JOINT supply (repjoint rep_t L20) | 0.25 | [0.16, 0.30] | 0.97 | — |
| **fenced rep_t L16/L20/L28** | **0.01 / −0.01 / 0.02** | all CIs ∋ 0 | — | \|α\|<0.2 **MET** |

**Readings.** (1) First direct measurement of the Hahn-class single-symbol
sensitivity decay in a real frozen VLM: joint L20 median ‖Δh‖ 10.5 → 1.3
(N=8→128); at N=128 the answer-relevant flip signal (1.33) equals the
answer-preserving ctrl flip (1.1) and the measured bf16 floor (~1.2–1.4) —
flipping the answer-defining frame changes the answer position by no more than
numeric noise. (2) The fenced arm is the money contrast: per-frame supply flat
in N at every layer. (3) The intermediate rungs interpolate (repjoint final
0.44, repjoint rep_t 0.25): per-frame readers alone recover only part of the
invariance; clean supply (the fence) zeroes the slope — consistent with the
q/kv interaction story. (4) A2 margin: the pre-registered crossing band FAILED
AS DEFINED but by instrument saturation, logged honestly: the frozen model is
already below 50% at N=8 on this slice (acc 0.28), so the median margin starts
negative (−0.53) and "crosses" immediately; the informative read is the
monotone margin slide (−0.23 @16 → −1.91 @128, ctrl-jitter floor 0.02–0.24)
with accuracy reaching chance at N≈32–64 (0.18/0.12/0.10). **Caveats:** gold≤8
slice (single-digit protocol, N-comparable by construction); L28 is
post-final-norm; margin conflates multi-digit continuations for non-gold mass.

--- DRAFT 3 (Experiment B) ---

## [2026-08-22] ⚠️📊 STATE-PASSING CONTROL (Buitrago & Gu, arXiv:2507.02782) — NOT RESCUED per the pre-registered band (best arm 0.800 @N=64 zero-shot < 0.90), and the intervention DECOMPOSES the drift: SP fixes the state-coverage leg (in-range 0.35→0.94/0.96 at 4× training length) while the readout-range cap survives exactly intact (GRU gold>16 recall 0.000 — a cliff at the label support)

> `scripts/armor/train_heads_sp.py` (recagg P2 protocol imported unchanged —
> fit @{8,16}, zero-shot @{32,64}, 20k epochs, 5 seeds, canary; recagg
> originals untouched). Jobs 136139/136140 → `outputs/armor/b_statepass/
> 20260822_212000_{hf512,park}/`. SP = detached final state of another
> training sequence + count-consistent carried target, REJECTED if total >16 —
> label range held fixed BY DESIGN so state coverage and label range are not
> confounded. Noise = fitted Gaussian on running final-state stats (paper
> variant), pre-registered as ill-posed for the pure integrator.

EM_reg @zero-shot (HF@512; majority 0.16/0.08; R1 sum-probe 0.996/1.000):

| arm | N32_zs | N64_zs | in-range(≤16) @64 | out(>16) @64 | max-count@rec≥.5 |
|---|---|---|---|---|---|
| R2 GRU base | 0.640 | 0.260 | 0.351 | 0.000 | 10 |
| R2 GRU **+SP** | 0.904 | 0.696 | **0.941** | **0.000** | 15 |
| R3 SSM base | 0.820 | 0.284 | 0.346 | 0.108 | 18 |
| R3 SSM **+SP** | 0.936 | **0.800** | **0.962** | 0.338 | 21 |
| R3 SSM +noise | 0.164 | 0.080 | — (in-length 0.728!) | — | −1 |

**Readings.** (1) The Buitrago objection is answered by RUNNING their
intervention: it works exactly on the leg its theory owns — the "unexplored
states" drift is real, and SP restores within-range accuracy to ~0.95 at 4×
the training length (park twin replicates: 0.40/0.29 → 0.68/0.68). (2) What
it cannot touch is the Yehudai range cap: the GRU emits NOTHING above count
16 in every arm (recall exactly 0.000, mc50 pinned at the fit-range max), and
26% of the N=64 pool lies above it → band NOT met. The necessity argument
rests on this leg, and it is intervention-proof by construction. (3) The
noise variant is catastrophic for the pure integrator even in-length
(0.984→0.728 @N=8) — the pre-registered identifiability prediction (a
borrowed-looking initial state offsets the integral unrecoverably). (4) R1
probe→sum stays exactly invariant through everything. **Caveats:** for RNN
heads SP-by-concatenation ≈ longer virtual training sequences (that IS the
published intervention; gradients detached, labels capped); cls-EM not the
verdict metric (support structurally capped at 16); N=128 zero-shot cells
pending captures.

--- DRAFT 4 (Experiment C) ---

## [2026-08-22] ⚠️📊 MLVU-AC EXTERNAL ANCHOR, 32f — composed frozen system (per-frame VLM records → compiled program → exact executor) reads 0.301 MCQ vs the pre-registered ≥0.332 band: NO-GO as registered — and the miss is LOCALIZED: the executor is 97% faithful to its delivered evidence, per-frame recall is 0.928, and the MEASURED 32f delivery ceiling for ANY faithful counter is 0.267 — below the frozen baseline's 0.282

> Captions 136145 → `outputs/armor/mlvu/20260822_213354_captions/` (206 q ×
> 32f @392, frame selection byte-identical to the [2026-07-11c] baseline
> protocol and the lookagain_N32 judge keys); compile+execute 136146 →
> `outputs/armor/mlvu/20260822_220317_compiled/` (qwen14b, armC v3 sandbox,
> P2b nearest-option MCQ rule; exec-fail 0.015; 0 caption parse-fails).

- MCQ 0.301 (frozen baseline 0.282, band ≥0.332, chance 0.25); open exact
  0.197, MAE 1.62; mean pred 1.35 vs gold 2.93; by-gold g1 0.70 / g2 0.42 /
  g3 0.18 / g4 0.18 / g5 0.00.
- **Localization chain:** VLM per-frame recall on judge-positive frames 0.928
  (precision 0.727) → 97.0% of predictions equal the run-count of the
  system's own delivered evidence (the compile+execute layers are essentially
  exact) → but 32f sampling delivers 282 visible frames against 603 gold
  instances (48/206 questions receive ZERO evidence), and an ORACLE
  judge-evidence run-counter scores 0.267 — **below the 0.282 frozen
  baseline**. At 32f the +0.05 band was unreachable by construction for any
  faithful counter; the frozen baseline's edge over the faithful ceiling is
  option-prior guessing, not counting.
- The honest external-validity statement: on real video the composed system's
  binding error is EVIDENCE DELIVERY (frame sampling), the same conclusion
  [2026-07-11c] reached for the frozen model (0.282 → 0.393 when density
  4×'d). Composed @128f dense (vs the 0.393 dense baseline) measured as a
  post-hoc extension: see addendum.
- **Caveats:** single question type (compile near-degenerate here — this cell
  measures the composed system, not compile generality); baseline had the MCQ
  options in-prompt, composed never sees them (P2b asymmetry, pre-registered);
  judge scores are themselves model-derived (lookagain), used as a proxy.

- [2026-08-22 ~23:59] C 128f DENSE ARM COMPLETE (captions 136149 1:24:13 ->
  outputs/armor/mlvu/20260822_220806_captions128/; compile+execute 136169 3:17
  -> outputs/armor/mlvu/20260822_233323_compiled128/ = CANONICAL dense cell).
  MCQ nearest-option 0.408 vs the [2026-07-11c] DENSE frozen baseline 0.393
  (+0.015) — the composed system EDGES the frozen model at matched 128f budget
  and is the best MLVU-AC number in the project record; far above both 32f
  numbers (0.301/0.282). Open exact 0.241, MAE 2.38; error flips to
  OVERcounting (mean pred 4.76 vs gold 2.93): at 128f delivery is largely
  fixed (1198 visible frames vs 603 instances; zero-evidence questions 48 ->
  10) and the binding error becomes per-frame FALSE POSITIVES (precision
  0.727) fragmenting into spurious runs. By-gold flattens (0.31-0.49, no
  monotone collapse — the delivery gradient is gone). Program-level note:
  at 128f the compiled programs diversify (only 56% match plain run-count
  semantics; index/gap variants otherwise; exec-fail 0.015 unchanged).
  Honest framing vs the pre-registered construction: applying the +0.05 rule
  to the dense baseline would ask >=0.443 — composed 0.408 does NOT clear
  that; it matches (slightly beats) the frozen baseline. The external
  anchor's verdict: on real video, composed EM tracks perception fidelity
  (recall/precision), exactly as Arm B found on MMReD-HF — no evidence of an
  aggregation-side deficit at either budget, and no +0.05 win either. Both
  cells logged; the 32f pre-registered verdict stands.

--- DRAFT 4 ADDENDUM (C, 128f dense) ---

- **128f dense extension (post-hoc, labeled):** composed @128f = **MCQ 0.408**
  vs the dense frozen baseline 0.393 (`[2026-07-11c]`) — the composed system
  slightly edges the frozen model at matched budget (best MLVU-AC number on
  record) but does not clear a +0.05-style margin there either (would need
  0.443). Delivery is largely cured at 128f (zero-evidence questions 48→10;
  1198 visible frames vs 603 instances) and the residual flips to
  OVERcounting from per-frame false positives (mean pred 4.76 vs gold 2.93;
  precision 0.727) — on real video, composed EM tracks perception fidelity in
  both directions, mirroring the Arm B factorization on MMReD-HF. Levers not
  run: per-frame self-consistency, FP-suppressing prompt, judge-style
  thresholding.

- [2026-08-23 ~00:10] N=128 CAPTURES LANDED (moved to a100 after ~2h l40s
  starvation; 136170 hf128 29:13, 136171 park128 31:07 ->
  outputs/armor/p1_captures_128/20260822_233338_{hf128_512,park128}/).
  npz sanity PASSED both: leaf L16+L20, Y-sum==G, hf 50 samples (golds 3-48),
  park 100 samples (class-balanced 6/bucket, golds 0-128). B rerun WITH the
  N128_zs cell submitted: 136186 (hf512_128) + 136187 (park_128), same
  instrument/recipe, 4h_0g.

- [2026-08-23 ~01:40] B N=128 CELL LANDED (136186, 1:33:19; outputs/armor/
  b_statepass/20260823_000332_hf512_128/). N128_zs (8x the fit lengths;
  majority 0.120; 60% of golds > 16):
    R1 1.000->0.980 (flat to 8x) | R2_gru base 0.044 / +SP 0.240 |
    R3_ssm base 0.108 / +SP 0.448 | spnoise arms at/below base.
  In-range(<=16) decomposition @128: GRU +SP 0.600 (was 0.941 @64) / SSM +SP
  0.770 (was 0.962 @64) — SP's state-coverage rescue itself ERODES at 8x (the
  concat-carry visits ~2-3-sequence virtual horizons, not 8x), while the GRU
  out-of-range wall stays EXACTLY 0.000 in every arm. Full drift row for the
  best intervention arm: 0.936 / 0.800 / 0.448 @ 2x/4x/8x vs R1 0.996 / 1.000
  / 0.980. The drift law survives the Buitrago interventions at every length;
  the rescue is partial, length-bounded, and never touches the range cap.
  Note: N<=64 numbers quoted from the main cell (20260822_212000_hf512); the
  rerun's N64 values differ by 0.01-0.03 (cross-node CPU float accumulation
  over 20k epochs = seed-level noise; both runs valid).
  Park+128 twin (136187) still running.

--- DRAFT 3 ADDENDUM (B, N=128 zero-shot) ---

- **N=128 (8×) cell:** R1 probe→sum 0.980 (flat to 8×); best intervention arm
  R3_ssm_sp **0.448** (GRU+SP 0.240; bases 0.108/0.044). SP's within-range
  rescue itself erodes with length (in-range ≤16: 0.96 @4× → 0.77 @8× SSM,
  0.94 → 0.60 GRU) — state passing extends coverage by a bounded factor (the
  virtual concatenation visits ~2–3-sequence horizons), it does not confer
  invariance; the GRU's out-of-range recall stays exactly 0.000 at every
  length. Intervention-arm drift row 0.936/0.800/0.448 @2×/4×/8× vs R1
  0.996/1.000/0.980: the drift law is intervention-robust.

- [2026-08-23 ~01:45] B PARK+128 TWIN LANDED (136187, 1:34:38; outputs/armor/
  b_statepass/20260823_000521_park_128/). N128_zs (majority 0.060): R1 0.994
  — perfect per-class recall at EVERY K bucket including c128 (mc50 = 128);
  R3_ssm_sp 0.500 / R2_gru_sp 0.424 (bases 0.092/0.148), GRU mc50 pinned at
  16. Second-domain replication of the 8x row complete.

============================================================================
CAMPAIGN CLOSE — [2026-08-23] all four experiments landed; per-band verdicts
============================================================================

  D  (N=128 qkv rerun)        DONE. Citable at n=80; staircase monotone;
                              scaling law extends (q-only +1.32 / kv-only
                              -0.78 / interaction +2.23). CANONICAL
                              qkv128/20260822_210658/.
  A  (Hahn instrument)        A1 JOINT BAND MET at L20: alpha 0.72 [0.63,
                              0.78], r2 0.98; FENCED BAND MET decisively
                              (|alpha| < 0.05 all layers); headline: joint
                              flip-signal AT the measured bf16 floor by
                              N=128 while fenced supply is flat ~30x above.
                              A2 band FAILED-AS-DEFINED via instrument
                              saturation (margin starts negative; acc 0.28
                              @N=8) — logged honestly; accuracy hits chance
                              at N~32-64. CANONICAL hahn/20260822_211457_*
                              + hahn_figure.png.
  B  (state-passing control)  NOT RESCUED per band (best 0.800 @N64_zs
                              < 0.90) — with the decomposition that answers
                              the reviewer: SP fixes state coverage
                              (in-range 0.35->0.94/0.96 @4x) but the rescue
                              erodes at 8x (0.60/0.77) and the readout-range
                              cap is untouched (GRU >16 recall exactly 0.000
                              everywhere; mc50 = label support). Drift row
                              of the BEST intervention arm 0.936/0.800/0.448
                              @2x/4x/8x vs R1 0.996/1.000/0.980 (park twin:
                              0.500 vs 0.994 @8x). Both domains, 4 cells.
  C  (MLVU-AC anchor)         32f pre-registered cell: NO-GO on the band
                              (0.301 < 0.332) — localized: executor 97%
                              faithful, VLM frame-recall 0.928, and the
                              MEASURED 32f faithful-counter ceiling (0.267)
                              sits BELOW the frozen baseline (0.282); band
                              unreachable by construction at 32f. 128f dense
                              extension: composed 0.408 vs dense baseline
                              0.393 — best MLVU-AC number on record; error
                              flips to perception-side overcounting. No
                              measurable aggregation-side deficit at either
                              budget.

  Assets: figures+CSVs in the run dirs; INDEX.md rows for all canonical runs;
  DRAFT RESULTS.md entries 1-4 (+ B/C addenda) at the bottom of this file.
  RESULTS.md untouched per instructions. Total footprint: 23 SLURM jobs
  (2 dead 21s no-ops), ~7 GPU-hours + ~9 CPU-core-hours, zero pip installs,
  legacy/ and recagg/ninv scripts untouched (wrappers/imports only).

- [2026-08-23] LOGGED: Tal said "log it" — the four DRAFT entries (with the B/C
  addenda folded in) appended to RESULTS.md as [2026-08-22] D, [2026-08-22] A,
  [2026-08-22→23] B, [2026-08-22→23] C. Drafts above kept for the record.

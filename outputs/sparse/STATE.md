# SPARSE campaign — STATE (append-only, newest last)

Brief: `CAMPAIGN_BRIEF.md` (authorized by Tal 2026-08-30, all cells pre-approved).
Cells: S0 sharpness sweep · S1 P1g oracle-gated fenced SFT · S2 model-gated read ·
S3 α under the gate · S4 capacity in k · S5 HF transfer · S6 conditional.

## [2026-08-30] Campaign created (Claude, with Tal) — no jobs yet

## [2026-08-30 ~21:50] Instruments built; anchors + smokes submitted

Per brief §5, all deltas live in `scripts/sparse/` (loramech/armor originals untouched;
they stay the anchors). REDUX campaign running in parallel (138955/957/968/969 on a100)
— not touched; its C4b no-gate chains will be reused for S3 comparison per brief §2.
- `scripts/sparse/train_sft_gated.py`: copy-extension of `train_sft_fenced.py` with
  `--gate oracle` (hide_cols = spans of non-evidence blocks via probe_evidence,
  |evid|==k asserted, skip+count; train AND eval forwards), `--gate-from-layer` (S2b,
  two FenceHooks over layer subsets — gnnformer untouched), `--attn-sharpen tau` +
  `--sharpen-from-layer` (S0; exclusive with --attn-logn-sref), eval now also reports
  mean signed error + majority baseline + skip counts. No-flag path = P1b anchor.
- `scripts/sparse/probe_hahn_gated.py`: copy-extension of `probe_hahn.py` with
  `--gate oracle` on the p1fence arm (per-member masks: base hides flip block t, evid
  member exposes it; ctrl uses base gate — answer-preserving = same evidence set).
  Pair selection provably unchanged (the added |evid|==gold assert is implied by the
  existing count asserts). No-flag path = N2 anchor.
- CPU checks: py_compile both; gate-mask semantics asserted (tail blind to non-evid
  blocks, sees evid block + prefix sink; own block re-opened; k=0 hides all).
- Wrappers: `slurm/sparse_{train,probe,s3_chain,smoke}.sbatch` (dry-runs clean).
- Smokes/anchors SUBMITTED (l40s-shared, 2h_2g): **138972** trainer chain (ANCHOR
  eval-only P1b exam_ff_N8@150 must read 1.0000; gated-eval smoke; sharpen smoke;
  gated-train 1-ep smoke) + **138973** probe (ANCHOR 3 pairs no-flags must reproduce
  n2_hahn dnorms 49.4684/25.4728/22.2908; 3 gated pairs). → `outputs/_scratch/sparse_smoke/`.
- Plan on pass: S1 (P1g, 10 ep, seq8+16, oracle gate, exclude all 5 exam files,
  24h_1g l40s --time=12h) + S0 6 configs (tau 1/1.5/2/3/4 @L>=12 + tau2 all-layers,
  eval-only P1b, N in {32,64,128} x 100, split across 24h_1g/12h_4g) + S1-control
  (P1b + gate @N=32, 2h_2g). S3/S4 wait for the P1g adapter.

## [2026-08-30 ~22:35] Probe anchor PROVEN byte-identical; gated probe smoke clean

- 138973 (l40s) first-pair dnorms deviated from the n2_hahn reference in the 3rd-4th
  digit — diagnosed as CROSS-ARCH bf16 (reference 137901 + REDUX parity smoke ran on
  a100 n310; the bit-identity canary lesson). Decisive same-arch test 138974: original
  `scripts/armor/probe_hahn.py` vs `scripts/sparse/probe_hahn_gated.py` (no flags),
  same args/seed, same L40S partition → **pairs.csv BYTE-IDENTICAL**. Probe anchor
  contract MET (the shared machinery was already proven vs the a100 reference by
  REDUX smoke 138967).
- Gated probe smoke (3 pairs, gate=oracle, P1b): runs clean; rep_t (verdict, own
  block) unchanged vs ungated as designed; the read locus responds to the gate
  (L16 final 0.54→5.7 — the gate changes what the tail sees, that is the point).
- `scripts/sparse/capture_verdicts.py` written (S2 pass-1 L20 replica-slot capture).
- Awaiting 138972 (trainer anchor: P1b exam_ff_N8@150 must read 1.0000 + gated-eval/
  sharpen/gated-train smokes).

## [2026-08-30 ~22:55] ALL ANCHORS MET + smokes clean → S1/S0/S1-control SUBMITTED

Anchors (brief §5, both required before first full run):
- **Trainer anchor MET**: `train_sft_gated.py` no-flags eval-only P1b on exam_ff_N8
  @150 → **acc 1.0000, pf 0, mae 0** (= the P1b logged number byte-for-byte in
  outcome). `outputs/_scratch/sparse_smoke/anchor_p1b_N8/`.
- **Probe anchor MET**: same-arch pairs.csv byte-identical (138974, see prior entry).
Smokes (all pf 0, no crashes): gated-eval P1b+gate@N8 6 dirs → acc 0.167 **mse +4.17
(gross overcount — previews H-S1-control's prediction)**; sharpen τ=2@L≥12 N32 5 dirs
→ mse +2.20 (overcount — the sweep will locate the break); gated-train 1 ep →
loss 0.721, adapter saves, gated decode works.
S4 dirs files built: `outputs/sparse/s4/dirs_N64.txt` (260: 20/gold × {0,1,2,4,8,12,
16,24,32,48,64} + exam extras), `dirs_N128.txt` (296: 20/gold × 13 anchors) — exam
dirs + strided top-up from the SAME longN_park pools (never-trained lengths).
S2 scripts written: `capture_verdicts.py` (L20 replica-slot pass-1) + `train_gate.py`
(LR gate, per-class + per-N reporting per the probe-family lesson).

**SUBMITTED:**
| job | cell | config | where |
|---|---|---|---|
| 138975 | **S1 P1g train** | oracle gate, 10 ep, seq8+16 ×900, excl all 5 exams, eval N8/16/32@150 gated | l40s 24h_1g 12h |
| 138976 | S0 τ=1.0@L≥12 (ref) | eval-only P1b, N32/64/128 ×100 | l40s 24h_1g 4h |
| 138977 | S0 τ=1.5@L≥12 | 〃 | l40s 24h_1g 4h |
| 138978 | S0 τ=2.0@L≥12 | 〃 | rtx6k 2h_2g |
| 138979 | S0 τ=2.0 all-layers ctrl | 〃 | rtx6k 2h_2g |
| 138980 | S0 τ=3.0@L≥12 | 〃 | l40s 12h_4g (pends behind REDUX) |
| 138981 | S0 τ=4.0@L≥12 | 〃 | l40s 12h_4g (pends) |
| 138982 | **S1-control** | P1b + oracle gate @N=32 ×150 | rtx6k 2h_2g |
[21:48] QOS re-route: 138980/81/82 moved 12h_4g|2h_2g -> 4d_1g (Tal's instruction; idle QOS, 1-GPU jobs)

## [2026-08-30 ~23:15] S1-CONTROL LANDS — H-S1-control CONFIRMED (job 138982)

`outputs/sparse/s1_control/20260830_214857_gated/` — P1b (ungated-trained) + oracle
gate @N=32, 150 exam dirs, pf 0, majority 0.080. **acc 0.080, mean signed error
+20.79** (band predicted mse ≥ +0.5 — met ×40). Per band: g0 0.083 (mse +3.7),
k1-4 0.000 (mse +29.5), k5-8 0.000 (+25.5), anchors 0.250 — g32 "all frames" is
the ONLY intact stratum (11/11). Anatomy: with the competitors hidden the softmax
share saturates and P1b answers ≈N for every k — the competitor-diluted calibration
in its purest form. The gate REQUIRES retraining (S1's premise, measured).

## [2026-08-31 ~00:15] S0 COMPLETE (jobs 138976-138981) — H-S0: NO τ REACHES THE k-REGIME → S6 NOT TRIGGERED

Runs: `outputs/sparse/s0_tau{1.0,1.5,2.0,3.0,4.0}_L12/` + `s0_tau2.0_L0/` (all-layers
ctrl), each N∈{32,64,128}×100 exam dirs, pf 0 everywhere, majority 0.06-0.08.

Per-band acc (mean signed err), bands g0 / k1-4 / k5-8 / anchors / overall:

| τ (layers) | N32 | N64 | N128 |
|---|---|---|---|
| 1.0 ref | 1.00 / .88(-.1) / .72(-.3) / .93 / **.85** | 1.00 / .00(-1.5) / .00(-1.9) / .73 / **.34** | 1.00 / .00(-2.0) / .00(-4.5) / .37 / **.23** |
| 1.5 ≥12 | 1.00 / .16(+1.0) / .22(+1.1) / 1.00 / **.48** | 1.00 / .46(+.2) / .00(+3.1) / .43 / **.36** | 1.00 / .17(-1.3) / .00(+.3) / .28 / **.23** |
| 2.0 ≥12 | 1.00 / .00(+2.1) / .03(+2.8) / .93 / **.35** | 1.00 / .36(+2.3) / .00(+7.1) / .30 / **.28** | 1.00 / .46(-.1) / .00(+4.1) / .17 / **.25** |
| 3.0 ≥12 | 1.00 / .00 / .00 / .54 / **.23** | .57 / .00(+8.1) / .00(+11.1) / .24 / **.13** | .83 / .00 / .00 / .17(-10.4) / **.13** |
| 4.0 ≥12 | .88 / .00 / .00 / .43 / **.19** | .57 / .00(+11.8) / .00(+14.0) / .22 / **.12** | .50 / .04 / .00 / .04(-19.8) / **.06** |
| 2.0 all | 1.00 / .00 / .12 / .82 / **.35** | 1.00 / .25(+2.9) / .00(+7.5) / .30 / **.25** | 1.00 / .25(+.8) / .00(+12.0) / .17 / **.20** |

**Verdicts (pre-registered bands, applied as fixed):**
- **H-S0 → the "needs training" side.** Best k≤8 band @64 ≈ 0.24 (τ1.5) vs the 0.80
  bar; every τ>1 also fails the N=32 side-condition (drop ≥0.37 vs allowed 0.03).
  The mean→sink transition is NOT reachable by eval-time temperature → **S6 not
  triggered** (P3's "compensation is never a training prior" verdict stands closed;
  the fixed factor fails even as an eval patch).
- **Over-sharpening threshold = τ1.5** (smallest tested >1: N32 drop 0.37 ≫ 0.05;
  parse-fail 0 everywhere — the LM never breaks FORMAT, it breaks CALIBRATION).
- Anatomy: sharpening flips the lawful undercount into mid-band OVERcount while g0 and
  in-support anchors survive longest; τ≥3 corrupts even g0/all-frames (τ4 N128 anchors
  mse −19.8). Layer restriction (≥12 vs all) is a wash (τ2 rows near-identical) —
  the failure is not depth-localized.
- τ1.0 ref on the 100-slice = 0.850/0.340/0.230 — reproduces the P1b 150-cell ladder
  (0.867/0.340/0.230); instrument faithful.

## [2026-08-31 ~00:45] S2 model-gate delta implemented; re-anchor v3 running

- `train_sft_gated.py` extended for S2a: `--gate model --gate-npz` (two-forward:
  ungated pass-1 → LR gate on L20 replica-slot states → gated decode; per-sample
  gate_fn/gate_fp columns in longn_predictions.csv + totals in the LONGN line).
  predict() now returns (pred, used_evid) — mechanical change on the default path
  too, so the anchor cell is RE-RUN against the current file (139027, v3; 139025
  cancelled — it had started on the pre-edit source).
- `oracle_evid` made HF-robust: evidence set computed directly from (char,room)
  occupancy; probe_evidence kept as a cross-check where its ROOMS list applies
  (HF pools contain "Hallway" ∉ constants.ROOMS — would have skipped ~1/6 of S5).
- `slurm/sparse_capture.sbatch` + `scripts/sparse/wave2_submit.sh` written: on P1g
  landing, one command submits S1-ladder(64/128) + S4 + S3 vs-N + S3 vs-k +
  S2 captures (train split + 5 exam cells).
- [~01:10] Re-anchor v3 (139027) PASSED on the current trainer file: exam_ff_N8@150 acc 1.0000 pf 0 mae 0 — gate-model plumbing certified default-path-neutral.

## [2026-08-31 ~01:20] S1 DIVERGED at the P1b recipe — killed, restarted at lr 1e-4 (ONE logged deviation)

138975 (`s1_p1g/20260830_214041_gated/`): train_loss 0.618→0.632→0.740→0.794
(ep0→3), val 0.550→0.533→0.167→0.050 — monotone divergence, vs P1b's 0.366→0.188→
0.136 at the identical recipe. Reading: consistent with S1-control — under the gate
the frozen-calibration read answers ≈N, so the gated task starts far from the
ungated init (ep0 loss 1.7× P1b's) and lr 2e-4 is unstable there. Not a data/mask
bug on current evidence (0 layout skips; anchor cell 1.000; gate semantics
CPU-asserted; ep0 val 0.55 shows the gated forward learns something before
diverging). **Deviation from "same recipe as P1b": lr 2e-4 → 1e-4, everything else
identical** (one change at a time; if r2 also diverges → stop and bug-hunt).
Resubmitted as 139033 (`s1_p1g_lr1e4/`, l40s 24h_1g, 12h). The diverged run dir is
kept for the record.

## [2026-08-31 ~02:30] Divergence diagnostic: init is benign — watching r2 through ep3

- diag 139055 (`outputs/_scratch/sparse_smoke/diag_loss_frozen/`): frozen-model
  answer-token loss by k, plain vs gated fence, seq8 ×60. Gated init losses are
  MODERATE (mean 0.38-1.09, max 1.41) and mostly BELOW plain (except k=0: plain
  0.018 — trivially easy ungated — vs gated 0.40, still small). REFUTES the
  "k=0 catastrophic tail at init" hypothesis; the instability develops during
  training, not from the start.
- S1-r2 (139033, lr 1e-4): ep0 loss 0.507 val 0.417; ep1 loss 1.298 val 0.533 —
  spike WITH improving val = possible code-reorganization, not r1's monotone decay.
  Decision: hold through ep3; kill only on sustained rise + val collapse
  (best-epoch selection protects the adapter).

## [2026-08-31 ~03:10] r2 KILLED (val 0.117 @ep2, same pathology at lr 1e-4) — ROOT CAUSE candidate found

Not an LR problem: two LRs, same collapse. **Candidate bug (mechanism-level):** the
trainer clears the hook-injected fence mask BEFORE `.backward()`; gradient
checkpointing (on by default under kbit-training) RE-RUNS the layer forwards during
backward, the mask pre-hook fires with an empty holder → the recompute runs
UNGATED (even FLASH-eligible) while the loss came from the GATED forward →
structurally inconsistent gradients. Explains: benign init losses (diag 139055),
ep0 partial progress, then collapse as gated/ungated activations drift apart; also
explains why P1b survived the same latent pattern (plain-fence vs joint recompute
mismatch is far milder than gated vs joint). NOTE for Tal: the same pattern exists
in `scripts/loramech/train_sft_fenced.py` — P1b's evals are unaffected (inference
only), but fenced training may improve with the same fix; loramech left untouched.
- Verification job: `diag_grad_mask.py` — one gated sample, LoRA grads under
  A) current pattern, B) mask held through backward, C) ground truth (no ckpt).
  139063 OOMed in C (nockpt @392px, 44GB); patched to 256px per-config prints,
  resubmitted as 139064.
- Fix STAGED in `train_sft_gated.py` (masks held through backward; cleared after;
  exception path clears too). S1-r3 will resubmit at the ORIGINAL lr 2e-4 on a
  CONFIRMED verdict (the lr deviation is then withdrawn).

## [2026-08-31 ~03:35] BUG CONFIRMED BIT-EXACT (139064) — S1-r3 submitted with the fix, ORIGINAL recipe restored

`outputs/_scratch/sparse_smoke/diag_grad/`: one gated sample, fresh LoRA, identical
forward losses (0.3140 all three) — gradients: **B(fix)-vs-C(truth) cos 1.000000,
max|Δ| 9e-10 (bit-exact); A(current)-vs-C cos 0.7767** — the clear-before-backward
pattern corrupts ~22° of the gradient on a single sample via the ungated
checkpointing recompute; compounding over steps = the observed collapse. The staged
fix (masks held through backward) is EXACTLY config B → validated by construction.
**S1-r3 = job 139066: `outputs/sparse/s1_p1g_r3/`, lr back to 2e-4
(deviation withdrawn), everything else = P1b recipe.** The r1/r2 run dirs stay as
the record of the bug.

## [2026-08-31 ~06:25] S1 P1g LANDS (139066, 5h07) — k≤8 PERFECT @2× under the gate; mid-k saturates to "all frames"

Run `outputs/sparse/s1_p1g_r3/20260831_011807_gated/` (fixed trainer, original P1b
recipe, lr 2e-4). Convergence textbook: ep0 0.349/0.750 → ep1 0.316/val 1.000
(best), loss →0.047; early stop @ep9; TEST_IID 0.986 (n=140).
**Gated exams (oracle gate, 150/cell, pf 0):**
| N | acc | per-band |
|---|---|---|
| 8 | **1.0000** | all strata perfect |
| 16 | **0.9467** | k≤6 + k12/k16 perfect; only k7 9/13, k8 9/13 (−1 wobbles, mse −0.05) |
| 32 (zero-shot) | 0.7800 | **k≤8 PERFECT 95/95** + g32 11/11; g12/g16/g24 = 0/11 each, EVERY error predicts exactly "32" |
Readings: (1) in-window the gate BEATS P1b (0.947 vs 0.893 @16); (2) the 2×
zero-shot k≤8 band is exactly 1.00 vs P1b's ~0.72-0.88 — the k-regime prediction
in its cleanest form; (3) the failure mode is a clean SNAP-TO-N saturation, and it
is not pure k-capacity: "12 of 16 visible" is perfect while "12 of 32" answers 32
— candidate story: prompt-N-conditioned share saturation. S4 + the 64/128 ladder
will map c* and the saturation boundary. H-S1's "every k≤16 stratum ≥0.97 at every
N" clause is already MISSED at N=32 (k12/16 = 0); the k≤8 clause is alive and
strong. Honest verdict at ladder landing.
- Adapter PROMOTED: `checkpoints/sft_fenced_gated_adapter` (+README row; EVAL
  CONTRACT: gate required).
- **Wave 2 submitted (139105-139114, all 4d_1g):** s1_ladder 64/128 · S4 (dirs
  N64/N128 topped-up) · S3 vs-N chain · S3 vs-k chain · captures train+5 exam cells.

## [2026-08-31 ~08:00] S3 vs-N chain lands (139107) — H-S3 read-flatness clause MET decisively

`outputs/sparse/s3/gate_oracle_N{8,16,32,64,128}/` (p1fence + P1g + oracle gate,
50/50/50/50/40 pairs, ctrl 12, seed 0, pf/skip normal). α = −slope of log median
dnorm vs log N, 2000-fold bootstrap:
| locus | α [95% CI] | medians N=8→128 | ungated ref (N2) |
|---|---|---|---|
| READ L16 final | +0.018 [−0.07,+0.12] | 9.0→8.3 FLAT | — |
| READ L20 final | +0.066 [−0.08,+0.20] | 26.4→21.3 FLAT | **+0.80 [0.66,0.97]**, 26→4.5 |
| READ L28 final | +0.018 [−0.03,+0.08] | 113→106 FLAT | — |
| verdict L20 rep_t | −0.007 [−0.03,+0.01] | ~55 flat | +0.009 (flat, as always) |
Margins (single-digit protocol): median base +5.60→+3.67 (stays positive O(1) at
128; ungated slid to −1.9); one-frame flip CHANGES the emitted answer in 98/86/90/
88/80% of pairs at N=8/16/32/64/128 — single-frame sensitivity at 16× the window.
The (N−k) term is gone from the read: α_N ≈ 0 measured, exactly the k-regime
construction. (γ-in-k clause pends on the s3_k chain 139108.)

## [2026-08-31 ~08:20] S3 vs-k chain lands (139108) — decay-in-k measured; H-S3 γ clause MET at the read locus

`outputs/sparse/s3/gate_oracle_N64_k{1,2,4,8,16,32}/` (30 pairs/k, redux N=64 pool,
gate on, flips k→k+1). Δ ∝ (k+C)^−γ fits (bootstrap CIs; C grid-fit):
| locus | γ [CI] | C | medians k=1→32 |
|---|---|---|---|
| L16 final | +0.77 [0.76,0.79] | 0.5 | 16.1→1.7 |
| **L20 final (read)** | **+1.19 [1.16,1.22]** | 2.0 | 41.9→2.9 |
| L28 final | +2.56 [2.50,2.60] | 8.0 | 119.6→4.4 |
Band "γ ≥ 1, CI excludes 0.5": MET at L20 (the ARMOR-A headline locus); L16 = 0.77
(excludes 0.5, below 1) reported honestly. N-dependence traded for k-dependence,
as the k/(k+C) code predicts. By k≥16 the +1-flip signal (2.9-4.1) nears the bf16
floor (~1.2-1.4) — consistent with the behavioral snap-to-N boundary living at
k≈6-12: the resolution wall is now IN k, not in N. (REDUX C4b no-gate k-chain =
the comparison arm; its runs live in outputs/redux/c4b_k/ — cite at write-up.)

## [2026-08-31 ~08:50] S2 GATE TRAINED — near-perfect and FLAT in N; model-gated evals submitted

Captures (139109-14): train split 651 samples/6216 frames (pos 0.469, 0 skips) +
five exam cells (1200/2400/4800/9600/19200 frames). LR gate on L20 replica-slot
states (`outputs/sparse/s2/gate/`): train/val acc 1.0000/1.0000, d' ~7.
**Per-N exam gate: acc 0.9992/1.0000/0.9990/1.0000/0.9997 @8/16/32/64/128, recall
1.0000 everywhere except N128 0.9997 (1 miss in 19,200 frames), d' 6.4-8.4 — FLAT
(the ARMOR-A flat-supply prediction carried through the trained adapter).**
H-S2 gate clause (≥0.99/frame every N) MET. Submitted: 139125 S2 model-gated
exams N8/16/32 · 139126 N64/128 · 139127 S5 HF seq8/16/32 (model gate, 50/cell,
N4 protocol) — all with per-sample gate_fn/gate_fp accounting.

## [2026-08-31 ~09:05] S5 lands (139127) — H-S5 NOT MET; the mid-k snap dominates the HF loss

`outputs/sparse/s5_hf/` (model gate, N4 protocol, 50/len, pf 0): **0.900 @8 /
0.700 @16 / 0.540 @32** vs P1b's 1.000/0.820/0.540 → the ≥+0.10 band is MISSED
(−0.10/−0.12/0.00). Gate transfer itself is clean (fn/fp per cell: 0/0, 3/2, 3/0
of 400/800/1600 frames ≈ 0.4%). Anatomy = the park story: k ≤ 5 near-perfect
(seq32: 25/26), all losses in the mid-k snap band (seq16 k6-10 ≈ 0, seq32 k≥7 ≈ 0,
mse +3.34). Honesty: benchmark cells are 50/len with tiny k-strata (1-7/cell);
majority floors 0.62/0.38/0.16.

## [2026-08-31 ~09:30] S1 ladder N=64/128 lands (139105) — H-S1 headline formally REFUTED; the partial result is the story

`outputs/sparse/s1_ladder_64_128/` (oracle gate, 150/cell):
- **N=64: 0.5533** (P1b 0.340) — k≤4 PERFECT 50/50; k5 10/10, k6 5/10, k7 8/10,
  k8 0/10; k12-48 all → "64"; g64 all-frames 10/10.
- **N=128: 0.4400** (P1b 0.230), pf 0.053 — k≤4 PERFECT 45/45; k5-k8 = 7/7/4/3 of
  9; k12-64 ALL predict exactly "128" (53/53); k96/k128 BREAK DOWN (parse-fails
  + absurd 1/8 answers — all 8 parse-fails live here); g128 0/8 (P1b ungated got
  this anchor right — gated training loses the all-frames case at 16×).
**H-S1 verdict (pre-registered): REFUTED** — k≤8 band @128 = 0.79 (excl. g0) /
0.81 (incl.), below the 0.85 refutation line; the ≥0.97-every-stratum clause fails
from N=32 up. What stands: k≤4 EXACT at EVERY N (8→128, 100%), and the failure is
a single lawful mode — a snap-to-N boundary that CONTRACTS with N (c*≈8-12 @32,
≈6 @64, ≈5 @128) plus an overflow regime near k≈N at 16×. The N-dependence the
gate removed from the read (S3: α≈0) survives in the CALIBRATION of the magnitude
code — diagnosis per the brief's H-S1 escape clause: a positional story is excluded
(posreset + flat verdict channel), leaving the sink/prompt-mass side: with (N−k)
competitors gone, k·e^s competes only against the FIXED prompt mass C, and the
trained thresholds place the "all" decision at k·e^s/C ratios that shift with the
prompt's declared N. S4 will chart the exact c*(N) curve.

## [2026-08-31 ~11:35] S4 lands (139106) — c*_trained(64)=8, c*_trained(128)=7; H-S4 MISSED; below c* the gate is EXACT at 16×

`outputs/sparse/s4/eval/` (oracle gate, topped-up 20/gold cells, 260+296 dirs):
- N=64: k≤5 **100/100**, k6 5/10, k7 8/10, k8 0/20, k12–48 all→"64", k64 20/20.
  acc 0.512, mse +15.9. **c* (first acc<0.5) = 8.**
- N=128: k≤4 **89/89**, k5 7/9, k6 7/9, k7 4/9, k8 9/20, k≥12 0/140 (all→"128"
  or parse-fail), k128 0/20. acc 0.392, pf 0.105 (parse-fails concentrated in
  k≥96). **c* = 7.**
**H-S4 verdict: MISSED** (expected c* 24–48; measured 7–8 ≈ the frozen c(fan)
crossing). The read: the gate buys EXACTNESS below c* at any N (frozen fan-8 was
0.65 at its own length; gated k≤5 is 1.00 at 16×), but does NOT extend the
capacity boundary itself — c* is a property of the magnitude code's resolution,
not of the competitor mass. Combined with S3's γ≈1.2 decay-in-k and the snap
anatomy: the k-wall is real and the code needs >log-resolution in k to pass it
(the token-coded route remains the only measured way past — consistent with the
LORAMECH closing claim).

## [2026-08-31 ~12:20] S2-long lands (139126) — campaign compute COMPLETE; H-S2 verdict mixed, accounting airtight

`outputs/sparse/s2/eval_long/`: model-gated N=64 **0.5533 — IDENTICAL to oracle in
every stratum, 0 gate errors on 9,600 frames**. N=128 **0.4267** vs oracle 0.4400,
gate errors = 5 frames/19,200 (fn 1, fp 4) across exactly 4 samples — and the
model-vs-oracle gap is those samples: g0→1 (1 fp), g1→3 (2 fp); the other two
(g12→128, g128→PF) fail identically under oracle. **H-S2: gate clause MET
(≥0.999/frame, flat); parity clause MET (gap = measured gate errors, per-sample);
the ≥0.90 @128 k≤8 clause NOT MET (0.79)** — bounded by the S1 snap, not the gate.
Full model-gated ladder: 0.993 / 0.960 / 0.753 / 0.553 / 0.427.

═══════════════════════════════════════════════════════════════════════════════
# CAMPAIGN VERDICT (2026-08-30→31, all cells closed) — self-contained summary

**Question.** LORAMECH showed the fine-tuned count read is a softmax SHARE
m = k·eˢ/(k·eˢ+(N−k)+C): N-dependent, hence the length wall. SPARSE asked: if the
read attends ONLY to evidence blocks (the gate = hide_cols over non-evidence
block spans, no new mask code), does the code become k/(k+C) — N gone, exact
counting at any length for k ≤ c*, one forward, no scratchpad?

**Answer: the N-mechanism is removed exactly as predicted — and the wall
reappears in k, smaller than hoped, plus a residual prompt-N calibration effect.**

What was ESTABLISHED (each traces to a run dir; see INDEX.md):
1. **The gate removes the read's N-decay** (S3, `s3/`): read α +0.07 [−0.08,+0.20]
   vs ungated +0.80 [0.66,0.97]; margins positive O(1) at N=128; a single-frame
   flip changes the emitted answer 80–98% of the time at EVERY N ∈ {8..128}.
   Sensitivity now decays in k: Δ ∝ (k+2)^−1.19 [1.16,1.22] @L20. **H-S3 MET.**
2. **Gated retraining gives a length-independent exact band** (S1+ladder,
   `s1_p1g_r3/`, `s1_ladder_64_128/`): k ≤ 4 EXACT at every N 8→128 (140/140,
   8–16× beyond training); in-window 1.000 @8 / 0.947 @16 (beats P1b's 0.893).
   Overall gated ladder 1.000/0.947/0.780/0.553/0.440 vs P1b 1.000/0.893/0.867/
   0.340/0.230. **H-S1 headline REFUTED as pre-registered** (k≤8 band @128 = 0.79
   < 0.85): beyond a boundary the model snaps to answering exactly "N" (all
   mid-k), and near k≈N at 16× it breaks down (parse-fails). The boundary
   CONTRACTS with N (≈8–12 @32, 6 @64, 5 @128) → a residual prompt-N calibration
   effect in the magnitude code (read-mechanism N-dependence excluded by S3).
3. **The capacity wall is in k and it is small** (S4, `s4/eval/`): c* = 8 @N=64,
   7 @N=128 (**H-S4 MISSED**; expected 24–48; ≈ the frozen c(fan) crossing). The
   gate buys EXACTNESS below c* at any length — not a bigger c*. Token-coded
   readout (the method's caption scan) remains the only measured route past k~8.
4. **The gate is deployable** (S2, `s2/`): logistic gate on L20 replica-slot
   states = per-frame acc ≥0.999 at every N, FLAT (1 miss/19,200 frames @128;
   d′ 6.4–8.4) — the ARMOR-A flat supply carried into deployment. Model-gated ==
   oracle at N=64 exactly (0 gate errors/9,600 frames); full model ladder
   0.993/0.960/0.753/0.553/0.427; every model-vs-oracle gap accounted per-sample
   by named gate errors. **H-S2 gate+parity clauses MET; the ≥0.90 @128 k≤8
   clause NOT MET (0.79, bounded by the snap, not the gate).**
5. **Eval-time alternatives fail** (S0, `s0_*/`): no sharpening τ reaches the
   k-regime (best k≤8 @64 = 0.24 vs bar 0.80); threshold τ1.5; calibration breaks
   before format ever does. **H-S0 → needs-training side; S6 not triggered.**
   P1b+gate without retraining answers ≈N (mse +20.79 @32) — **H-S1-control
   CONFIRMED**; the share calibration is competitor-mass-specific in both
   directions.
6. **Transfer** (S5, `s5_hf/`): model-gated on MMReD-HF 0.900/0.700/0.540 vs P1b
   1.000/0.820/0.540 — **H-S5 MISSED**; k≤5 near-perfect, the loss is the same
   snap band; complementary anatomies at seq32 (P1b keeps label-support anchors,
   loses low-k; gated keeps ALL k≤5, loses mid-k).
7. **Methodological find (repo-relevant):** hook-injected masks + gradient
   checkpointing = the recompute during .backward() runs UNGATED if the mask is
   cleared pre-backward. Gradients 22° off truth (cos 0.777); two divergent runs;
   fix (hold mask through backward) validated bit-exact (cos 1.000000, 139064).
   The same latent pattern exists in `scripts/loramech/train_sft_fenced.py`
   (its evals unaffected; P1b converged through the milder plain-fence mismatch).

Bands scorecard: H-S3 MET · H-S2 2/3 MET · H-S0 resolved(training side) ·
H-S1-control MET · H-S1 REFUTED · H-S4 MISSED · H-S5 MISSED.

OPEN after this campaign: (a) the snap-to-N calibration — can training on a
declared-N-randomized prompt (or N-free prompt) decouple the "all" decision from
prompt N? (b) c* is code-resolution-bound: does a coarse token-coded intermediate
(e.g. carrier digit) under the gate lift c* without the full scratchpad?
(c) the k≈N overflow regime at 16×. Honesty flags: single task family + model;
oracle vs model gate labeled in every number; k-strata <10 marked in run logs;
N=16-eval wobble (k7/k8) is margin-thin (cross-node bf16 flips it).

Figures: `fig/F1_flatline.png` (headline), `F2_capacity.png`, `F3_alpha.png`,
`F4_s0_sweep.png` (+CSV tables). Every number above traces to the run dirs in
INDEX.md. Campaign compute: ~30 GPU-jobs, all logged in this file, newest last.
═══════════════════════════════════════════════════════════════════════════════

## [2026-08-31] WAVE 3 AUTHORIZED (Tal) — S7 prompt-lie probe + S8 virtual-N training

Brief extended (CAMPAIGN_BRIEF.md "WAVE 3" section): S7 = --declare-n eval probe of the
snap's text channel (two-sided bands); S8 = virtual-N mixture training (equivalence smoke
REQUIRED before use; fall-back defined), adapter → sft_fenced_gated_vn_adapter. Same
permissions/discipline as waves 1-2. Agent prompt: AGENT_PROMPT_WAVE3.md. No jobs yet.

═══ WAVE 3 (authorized Tal 2026-08-31) — kill the snap: S7 prompt-lie + S8 virtual-N ═══

## [2026-08-31 ~14:10] Wave-3 instruments built; smoke + equivalence gate submitted

- Trainer deltas (`train_sft_gated.py`, guarded, no-flag path unchanged):
  `--declare-n Ñ` (prompt TEXT only — frames/mask/gate/positions untouched) and
  `--virtual-n` (S8 mixture: 50% real gated + 50% synthetic evidence-only samples,
  k∈0..16 stratified × Ñ∈{8,16,32,64,128}, k≤Ñ, answer=k; sources = k-subsets of
  real training samples' evidence frames — self-certifying; k=0 = text-only step,
  plain causal, position caveat logged: real gated k=0 tails sit at blk0_max+1,
  synthetic k=0 tails are contiguous — both teach "no evidence → 0").
- `scripts/sparse/diag_equiv.py`: the S8 HARD GATE. A = true gated N-frame forward,
  B = evidence-only twin declaring N. Positions provably N-independent by
  construction (posreset: all blocks wear block-0's ids; tail starts at blk0_max+1)
  — the smoke tests the bf16 reality. PASS pre-registered: ALL greedy decodes
  identical AND median rel L2 diff @L20 answer position < 0.02 (10 samples each
  at N=16 and N=32, P1g adapter).
- S7 dirs: `outputs/sparse/s7/dirs_N{32,64}.txt` — 20/stratum strided from
  longN_park pools (fresh, never-trained; k∈{4,8,12,16}@32, k∈{4,6,8,12}@64).
- SUBMITTED: 139239 sp_w3_smoke (ANCHOR eval-only P1b exam_ff_N8@150 must read
  1.0000 with the wave-3 deltas + declare-n smoke + virtual-n 1-ep smoke) ·
  139240 sp_w3_equiv (the S8 gate). S7 full chains submit on anchor pass; S8
  training only on equivalence PASS (fall-back per brief on FAIL).

## [2026-08-31 ~15:10] EQUIVALENCE SMOKE PASS (139240) — S8 GO; the twin REPRODUCES the snap with nothing hidden

`outputs/_scratch/sparse_w3/equiv/`: 20/20 greedy decodes IDENTICAL (A gated
N-frame vs B evidence-only twin declaring N), median rel L2 @answer position
L16/L20/L28 = 0.0104/0.0102/0.0068 — well under the 0.02 pre-registered bar.
The gated forward IS the evidence-only forward (bf16-tight), so S8's synthetic
coverage is licensed. **Bonus pre-confirmation of the text channel:** the k=12
and k=16 twins — which contain NO hidden blocks whatsoever — also decode exactly
"32": the snap needs only the prompt's declared N to appear. S8 training submits
when 139239's anchor + virtual-n smoke land.

## [2026-08-31 ~15:25] ALL WAVE-3 GATES PASSED → S7 chains + S8 training SUBMITTED

Smoke chain 139239: **ANCHOR MET again** (eval-only P1b exam_ff_N8@150 = 1.0000
with --declare-n/--virtual-n in the file); declare-n smoke clean (8-dir slice
already previews H-S7(a): declared-16 on true-32 recovers g12 2/2, g16 2/2);
virtual-n 1-ep smoke trains without skips (mixture builder verified; small-limit
grid truncation is expected j%77 behavior — full run covers all Ñ).
SUBMITTED: 139245 S7 N32 chain (ref/d16/d64, 80 dirs each) · 139246 S7 N64 chain
(ref/d16/d32) · **139247 S8 virtual-N training** (S1-r3 recipe verbatim + the
mixture; excludes = 5 exam files + S4 + S7 dirs; 24h_1g, 14h; ~1302 steps/ep).

## [2026-08-31 ~17:30] S7 N=32 chain lands (139245) — BOTH H-S7 CLAUSES MET: the snap IS prompt-N text calibration, causally

`outputs/sparse/s7/N32_{ref,d16,d64}/` (P1g + oracle gate, 80 dirs = 20/stratum
k∈{4,8,12,16}, majority 0.25, pf 0):
| declared Ñ | acc | k4 | k8 | k12 | k16 | snap target |
|---|---|---|---|---|---|---|
| 32 (ref) | 0.500 | 1.00 | 1.00 | 0.00 | 0.00 | "32" (40/40) |
| **16** | **0.900** | 1.00 | 0.60 | **1.00** | **1.00** | — (k12→"12", k16→"16" EXACT) |
| **64** | 0.250 | 1.00 | **0.00** | 0.00 | 0.00 | **"64" (40/40)** |
- **H-S7(a) MET** (k12 ≥0.8 bar → measured 1.00; snapped answers relocate to the
  declared value and become CORRECT). **H-S7(b) MET** (k8 ≤0.5 bar → measured
  0.00). **Side condition MET** (k4 = 1.00 under every lie).
- Fine structure: a −1 undercount regime precedes the snap (declared-64 k8 → all
  "7"; declared-16 k8 → 12/20 with 8ד7") — the boundary tracks the DECLARED N
  and the whole wave-2 contraction pattern re-parameterizes in Ñ, not true N.
**The text channel is the snap. S8's mechanism confirmed causally before S8
trains** (consistent with the equivalence smoke's k12/k16 twins answering "32"
with nothing hidden). N64 chain (139246) still queued/running; S8 (139247) training.

## [2026-08-31 ~19:50] S7 N=64 chain lands (139246) — replication complete; H-S7 CONFIRMED overall

`outputs/sparse/s7/N64_{ref,d16,d32}/` (20/stratum k∈{4,6,8,12}, majority 0.25, pf 0):
| declared Ñ | acc | k4 | k6 | k8 | k12 | notes |
|---|---|---|---|---|---|---|
| 64 (ref) | 0.400 | 1.00 | 0.60 | 0.00 (all "7") | 0.00 (all "64") | −1 ring at k8, snap at k12 |
| 16 | **0.900** | 1.00 | 1.00 | 0.60 | **1.00 (all "12")** | recovery |
| 32 | **0.738** | 1.00 | 1.00 | **0.95** | 0.00 (all "32") | boundary moved UP with smaller Ñ |
**Combined S7 verdict: the snap is prompt-Ñ text calibration, CONFIRMED at both
true-N values with 6 lies.** Every stratum's outcome is a function of (k,
declared Ñ) only: snap target = the declared string, always; a −1-undercount ring
precedes the snap (k≈Ñ/2±); recovery is EXACT when the lie brings k inside the
declared window; k4 = 1.00 in all 6 lies (side condition MET, 120/120). True N
does not appear anywhere in the behavior once the gate is on — converging with
the equivalence smoke. S8's mixture (which trains the full (k,Ñ) grid) is the
directly-indicated fix; H-S8-main now has a confirmed mechanism behind it.

## [2026-08-31] S9 AUTHORIZED (Tal) — N-free prompt cell added to wave 3

Brief extended with S9 (N-free prompt; upstream-protocol fact verified: the original
benchmark never states N — our build_count_prompt N-injection is legacy-local). S1-r3
recipe + oracle gate + minimal N-free prompt, no virtual-N mixture needed; bands
H-S9-main / -capacity / -vs-S8 / -HF pre-registered in the brief. Adapter target:
sft_fenced_gated_nfree_adapter. Same permissions/discipline as the rest of wave 3.

## [2026-08-31 ~20:20] S9 implemented (N-free prompt) — gate smokes submitted

Per Tal's S9 authorization: `--nfree-prompt` added to `train_sft_gated.py` (N-free
count prompt, same "You will be shown" opener so parse_layout is unchanged;
exclusive with --declare-n/--virtual-n; train AND eval; adapter contract carries
the flag) + `capture_verdicts.py --nfree-prompt` (S9 re-capture through the
matching layout). CPU check: prompt text + needle verified. SUBMITTED 139300
(smoke9): ANCHOR re-run (P1b exam_ff_N8 must read 1.0000 with the nfree delta in
the file) + 1-ep nfree train smoke. S9 full training submits on pass (S1-r3
recipe verbatim + nfree, NO mixture — with no N in the text, real gated samples
cover every N by the equivalence-smoke logic). S8 (139247) unaffected, training.

## [2026-08-31 ~21:30] S9 gates PASSED → training submitted (139318)

139300 (smoke9, rtx6k, 6m51): **ANCHOR MET** — eval-only P1b exam_ff_N8@150 =
1.0000/pf 0 with the --nfree-prompt delta in the file; nfree 1-ep train smoke
clean (mixture-free gated nfree path trains and decodes, 0 skips). **S9 training
139318** → `outputs/sparse/s9_nfree/` (rtx6k 24h_1g 14h; S1-r3 recipe verbatim +
--nfree-prompt, NO mixture; excludes = 5 exams + S4 + S7 dirs). S8 (139247) at
ep2 0.149/0.983, healthy. Note: earlier "one job visible" moment was smoke9
completing faster than estimated (rtx6k N=8 eval ~2s/sample), not a loss.

## [2026-09-01 ~00:45] S8 LANDS (139247, 9h04) — the snap is DEAD where coverage exists

`outputs/sparse/s8_vn/20260831_134617_gated/` (virtual-N mixture, fixed trainer,
best ep3 val 1.000, TEST_IID 1.000). **Gated exams (oracle gate, 150/cell, pf 0):**
| N | acc | vs P1g | anatomy |
|---|---|---|---|
| 8 | **1.0000** | = | perfect (H-S8-noharm N8 ≥0.99 **MET**) |
| 16 | **1.0000** | 0.947 | perfect — P1g's k7/k8 wobble GONE (noharm N16 ≥0.94 **MET**, exceeded) |
| 32 zs | **0.8533** | 0.780 | k≤8 perfect 95/95; **k12 11/11, k16 11/11 (P1g: 0/11 both — SNAP ELIMINATED)**; k24 0/11 + k32 0/11 (beyond the k≤16 trained grid: honest undercount mse −1.76, no snap crutch; g32 "all frames" now fails as predicted N/A-by-coverage) |
Cascade fired (139451-58): oracle ladder 64/128 · S4 dirs · captures (train + 5
exam cells) — all pending behind the busy cluster. Adapter PROMOTED:
`checkpoints/sft_fenced_gated_vn_adapter`. H-S8-main decides on the ladder.
S9 (139318) finished ep9 (best ep3 val 1.000) — running its in-window exams now.

## [2026-09-01 ~01:20] S9 LANDS (139318) — in-window strong, NO snap signature; cascade fired

`outputs/sparse/s9_nfree/20260831_174021_gated/` (N-free prompt, no mixture, best
ep3 val 1.000, TEST_IID 0.986). Gated exams (oracle gate + nfree prompt, 150/cell,
pf 0): **1.0000 @8 · 0.9733 @16** (k12 9/13 wobble) **· 0.7933 @32 zs** — and the
anatomy is NEW: k≤8 perfect 95/95, **k16 10/11**, but k12 2/11, k24 1/11, k32
0/11 with mse +0.29 (mae 0.99 — SMALL misses, no snap-to-anything; the prompt
carries no N to snap to, as designed). S8-vs-S9 @32: S8 0.853 (k12+k16 perfect,
k24/32 dead by coverage) vs S9 0.793 (k16 alive, k12 weak, k24 near-dead) —
different mid-k profiles, ladder + S4 will resolve. Adapter PROMOTED:
`checkpoints/sft_fenced_gated_nfree_adapter` (contract: gate + nfree prompt).
**S9 cascade fired (139461-68):** oracle ladder 64/128 + S4 dirs (both with
--nfree-prompt) + captures (train + 5 exam cells, nfree layout; capture wrapper
gained an EXTRA knob).
- [~02:00] S8 gate refit (): acc 0.9973-1.0000 per N, recall 1.0000 everywhere, flat (d' 6.9-7.9). Model-gated ladder submitted: 139471 (N8/16/32) + 139472 (N64/128).
- [~02:00] S8 gate refit (outputs/sparse/s8_gate/): acc 0.9973-1.0000 per N,
  recall 1.0000 everywhere, flat (d' 6.9-7.9). S8 model-gated ladder submitted:
  139471 (N8/16/32) + 139472 (N64/128). Ten wave-3 eval jobs now in flight.
- [~02:40] S9 gate refit (outputs/sparse/s9_gate/): acc 0.9983-0.9998 per N, flat. S9 model-gated ladder (sp9_mg 139475/139476-era ids above) + S5-HF submitted.

## [2026-09-01 ~03:00] S8 ORACLE LADDER LANDS (139451) — **H-S8-MAIN MET: k≤8 = 1.000 AT EVERY N; THE SNAP IS DEAD**

`outputs/sparse/s8_ladder_64_128/` (+ the in-window cells from the train job):
| N | acc | k≤8 band | k12 | k16 | k≥24 |
|---|---|---|---|---|---|
| 8 | 1.0000 | **1.000** | — | — | — |
| 16 | 1.0000 | **1.000** | 13/13 | 13/13 | — |
| 32 zs | 0.8533 | **1.000** | 11/11 | 11/11 | 0 (coverage) |
| 64 zs | 0.7333 | **1.000** (80/80) | **10/10** | **10/10** | 0 |
| 128 zs | 0.6067 | **1.000** (72/72) | **9/9** | 1/9 | 0 |
**H-S8-main (k≤8 ≥0.95 every N): MET at 1.000/1.000/1.000/1.000/1.000** — wave-2
values were 1.00/0.95/1.00/0.73/0.63. pf 0 at both lengths (the k≈N overflow
parse-fails are GONE too — mid-k no longer routes through the "all frames"
answer). H-S8-cap (descriptive): k12 = perfect at EVERY N and k16 through N=64 —
the wave-2 c*=7-8 reading is hereby annotated CALIBRATION-CONFOUNDED; the true
trained-capacity edge sits at k≈16 (N≤64) / k≈12-16 (@128), consistent with S3's
resolution physics (γ≈1.2, floor at k≈16). Failures at k≥24 are the stated
coverage hole (k>16 never trained), mse mixed ±, not snap-shaped.

## [2026-09-01 ~04:10] S9 ORACLE LADDER LANDS (139461) — H-S9-main MET TOO: k≤8 = 1.000 at every N, with NO N in the prompt

`outputs/sparse/s9_ladder_64_128/`: N=64 acc 0.700 (pf 0.067) — k0-k8 PERFECT
(90/90), k16 10/10, k12 5/10, k≥24 = 0; N=128 acc 0.600 (pf 0.147) — k0-k8
PERFECT (81/81), k12 6/9, k16 3/9, k≥24 = 0. **k≤8 band = 1.000/1.000 @64/128 →
H-S9-main MET at every N.** Parse-fails live in the un-trained high-k strata (no
answer-range hint in the N-free prompt). Both wave-3 adapters now hold the exact
flat line; S8 is cleaner (pf 0, k12 perfect at every N) — three-way call at close.

## [2026-09-01 ~05:20] BOTH S4 RUNS LAND (139452/139462) — c* SUPERSESSION CONFIRMED: the capacity edge is k≈16, not 7-8

Oracle-gated capacity (20/gold cells, `outputs/sparse/s8_s4/` + `s9_s4/`):
| arm | N=64 | N=128 | c* |
|---|---|---|---|
| S8 vn | **k0-k16 PERFECT (170/170)**, k≥24 = 0, pf 0 | k0-k12 perfect, k16 5/20, k≥24 = 0 | 16-24 @64, **16 @128** |
| S9 nfree | k≤8 perfect, k12 11/20, **k16 20/20**, k≥24 = 0, pf 0.085 | k≤8 perfect, k12 11/20, k16 7/20 | ~16 (non-monotone k12 dip) |
**Wave-2's S4 (c* = 7-8) is hereby SUPERSEDED as calibration-confounded** — with
the snap dead the same dirs read k12/k16 at up to 100%; the true edge (≈16) lands
exactly in S3's resolution-physics window (γ≈1.2 fit + flip-signal-at-floor by
k≈16). H-S9-capacity: MET in range. S9's k12 anomaly anatomy (k12@128: 11×"12",
**7×"128"**, 2×"16" — "128" emitted with NO N in any text): candidate channel =
the step-id burned into the frame PIXELS (park renderer stamps step numbers up to
N on the images; S8's mixture — small-step evidence frames paired with large
declared Ñ — taught prompt-dominance, S9 has no text N to dominate). Logged as an
interpretation with its evidence, not a claim.

## [2026-09-01 ~06:00] S9 mg-long lands (139487) — ALL WAVE-3 CELLS CLOSED

S9 model-gated: 0.6933 @64 (1 fp; k≤8 band **1.000**) · 0.5667 @128 (1 fn + 7 fp;
k≤8 band 0.958 — the 3 lost samples are the gate-error samples). Parity holds for
both adapters.

═══════════════════════════════════════════════════════════════════════════════
# WAVE-3 VERDICT (2026-08-31→09-01, all cells closed) — self-contained summary

**Question.** Wave 2 left one wall inside the resolution range: mid-k snaps to
answering exactly "N", boundary contracting with N. Wave 3 asked: is that snap the
PROMPT TEXT's declared N (calibration), and do (a) full (k,Ñ) coverage (S8) or
(b) removing N from the text entirely (S9) eliminate it?

**Answer: yes and yes — the snap was text calibration, and BOTH fixes produce a
PERFECT k≤8 flat line at every N ∈ {8..128}, 8-16× beyond training.**

1. **S7 (causal, 6 lies, 2 true-N values): the snap IS prompt-Ñ calibration.**
   Declared-16 on true-32 recovers k12/k16 from 0.00 to 1.00 WITH CORRECT answers;
   declared-64 collapses k8 1.00→0.00; snap target always = the declared string;
   k4 = 120/120 under every lie; a −1-undercount ring precedes the snap. Both
   pre-registered clauses MET. (`s7/`)
2. **Equivalence smoke (S8's hard gate): PASS** — gated N-frame forward ≡
   evidence-only twin (20/20 identical decodes, rel L20 0.0102); the k12/16 twins
   answer "32" with NOTHING hidden — independent text-channel confirmation. Also
   licenses the synthetic mixture. (`_scratch/sparse_w3/equiv/`)
3. **S8 virtual-N (CANONICAL): H-S8-main MET — oracle k≤8 = 1.000/1.000/1.000/
   1.000/1.000; noharm MET (1.000 @8 AND @16)**; k12 perfect at every N; k16
   perfect through N=64 (@128: 5/20); c*(128) = 16 — wave-2's S4 c*=7-8 SUPERSEDED
   as calibration-confounded; the true edge matches S3's resolution physics.
   pf = 0 everywhere (the k≈N overflow is gone too). Failures only at k≥24 = the
   stated coverage hole. Deployed (refit model gate, ≥0.997/frame flat): k≤8 band
   0.99/1.00/0.92/0.96/0.94, every gap = named gate errors. (`s8_*`)
4. **S9 N-free (protocol-fidelity variant): H-S9-main MET — oracle k≤8 = 1.000 at
   every N with NO N ANYWHERE IN THE TEXT** (upstream MMReD never states N; our
   N-injection was legacy-local). In-window 1.000/0.973; k16 20/20 @64; k12
   non-monotone dip (11/20 @64/128) with a candidate pixel-channel explanation
   (step-ids burned into frames; "128" emitted at k12@128 7/20 with no N in text)
   — logged as interpretation. High-k parse-fails (no range hint). H-S9-capacity
   MET in range (c*≈16). **H-S9-HF MISSED** (0.86/0.78/0.58 vs ≥0.95/0.90/0.85;
   driver = gate recall on HF, fn 6-7/cell, + the k12+ edge; still ≥ wave-2 S5 at
   16/32). Deployed k≤8: 1.000 @64, 0.958 @128. (`s9_*`)
5. **Three-way call (H-S9-vs-S8): S8 = CANONICAL** (k≤8 tie at 1.000; S8 wins k12
   everywhere, pf 0 vs 0.07-0.19, and the HF comparison is prompt-variant-
   confounded in S9's favor-case anyway); **S9 = the protocol-fidelity variant**
   proving the mechanism needs no N: text, virtual or otherwise. S7+S8+S9 agree —
   no S7/S8 disagreement to report.

Scorecard: H-S7 both clauses MET · equiv PASS · H-S8-main MET · H-S8-noharm MET ·
H-S8-cap: edge at k≈16 · H-S8-model MET · H-S9-main MET · H-S9-capacity MET ·
H-S9-vs-S8 → S8 canonical · H-S9-HF MISSED.

OPEN: (a) k>16 coverage (extend the synthetic grid or curriculum — c* is now the
only wall below k≈N); (b) the k≈N regime at large N (unknowable without N by
design in S9; coverage-hole in S8); (c) HF gate recall (6-7 fn/cell — gate
retraining on HF captures or threshold tuning); (d) the step-id pixel channel
(verify by re-rendering an S9 eval cell with step-ids stripped). Honesty flags:
single task family/model; S9 numbers are a NEW prompt anchor (labeled); k-strata
<10 marked; S8's k>16 zeros are coverage, not capacity, through k≈16-24.

Figures: `fig/F5_flatline_w3.png` (+CSV) — P1b crash / P1g 0.79 tail / S8+S9
EXACT flat lines / S8-model deployed. All numbers trace to run dirs in INDEX.md.
═══════════════════════════════════════════════════════════════════════════════

## [2026-09-01 ~09:40] CORRECTION/refinement — S9's k12 dent is a TRAINED-CEILING attractor, not (primarily) pixels

Error-histogram analysis (Tal's morning question): S9 k12 errors predict **"16"**
— 4/4 @N=16, 9/9 @32, 14/14 @64 — the trained ANSWER CEILING, not N. Root cause:
the real training mixture's gold support is g0-g8 dense + g12 (17) + g16 (17)
ONLY — answers 9-11/13-15 never seen; 12 is weakly trained against a strong 16.
S8 avoids this exactly because its synthetic grid trains k=0..16 DENSELY. Not an
epoch-budget issue (same budget as S8; best ep3, six further improving-loss
epochs never beat it). The step-id-pixel candidate now applies ONLY to the @128
second attractor ("128" ×9 at k12, ×16 at k16 — no text contains 128). Follow-up
levers: S9b = k-balanced evidence-subset oversampling (S8's coverage without Ñ
text); the strip-step-ids re-render test for the @128 attractor. Also noted: under
the gate, k≤16 cells at any N have ≤16 VISIBLE blocks (equivalence) — the gate
converts length-generalization into k-generalization; genuinely-longer-than-
trained visible inputs begin at k≥24, where both arms are also outside answer
support (the k≥24 zeros stay coverage-confounded, per the caveats).

## [2026-09-01 ~10:20] S9b AUTHORIZED (Tal) — k-balanced evidence-subset oversampling under the N-free prompt

Design: S9 recipe verbatim + the S8 mixture machinery with the Ñ dimension
collapsed (grid = k∈{0..16} only, ~38 dense exposures/answer-value/epoch; fixes
the answer-support hole 9-15 that causes the k12→"16" attractor). Delta: the
nfree×virtual-n exclusivity assert relaxed (declare-n still excluded); nfree
synthetic branch builds N-free prompts. Also recorded: S9 trained-support table
(gold 0-8) = **oracle 827/827 = 1.0000 at every N**, model gate 0.93-0.99.
Gate smokes 139713 (anchor + 1-ep 9b train). Training submits on pass →
`outputs/sparse/s9b_vn_nfree/`.
- [~10:50] Smoke9b PASSED (anchor 1.0000; k-only grid verified). **S9b training 139714** → outputs/sparse/s9b_vn_nfree/ (24h_1g, 14h; ~1302 steps/ep like S8).

## [2026-09-01 ~11:30] Tal: 5 EPOCHS IS THE DEFAULT GOING FORWARD; S9b-ep5 launched in parallel

Evidence basis: every wave-3 run's best adapter came at ep1-3 (S8 ep3, S9 ep3,
S1-r3 ep1); epochs 6-10 never won val. **139718 sp_s9b_ep5** → `outputs/sparse/
s9b_ep5/` (rtx6k, same recipe, EPOCHS=5). The 10-ep run 139714 CONTINUES (Tal:
keep unless GPUs are scarce — one rtx6k slot was free, so both run). The ep5-vs-
ep10 pair doubles as a budget ablation for the record. Future submits in this
line: EPOCHS=5 (logged deviation from the S1-r3 anchor recipe, evidence above).

## [2026-08-31] S10 + S11 AUTHORIZED (Tal) — attention photograph + text-controlled S9 α chain

Brief extended: S10 = direct observation of the share formula (manual attention weights
from FenceHooks q/k at the answer row; ungated (N−k) fit vs gated k-only invariance;
self-gating head readout; optional Jacobian sub-cell; F7). S11 = probe_hahn_gated +
--nfree-prompt on the S9 adapter (falsification-grade: byte-identical prompt across N;
|α| ≥ 0.3 = non-text N-channel, stop-and-investigate); F6 three-panel figure. Both
measurement-only. No jobs yet.

## [2026-09-01 ~12:10] S10 + S11 AUTHORIZED (Tal) — instruments built, gates running

- **S11**: `probe_hahn_gated.py --nfree-prompt` added (single-source: the sparse
  trainer's builder; report header carries the flag; chain wrapper gained EXTRA).
  Anchor re-verify 139735 (3 no-flag pairs must byte-match the L40S reference
  `outputs/_scratch/sparse_smoke/probe_anchor/pairs.csv`). Chains (vs-N + vs-k,
  S9 nfree adapter, oracle gate, nfree prompt — text byte-identical across N)
  submit on anchor pass. H-S11a refutation branch armed: |α|≥0.3 ⇒ STOP write-up.
- **S10**: `scripts/sparse/probe_attn_photo.py` written — manual attention at the
  last prompt row from FenceHooks qkv+rotary (adapter loaded BEFORE capture-hook
  install so LoRA q/k are captured; per-row softmax-sums-to-1 asserted; per-head
  per-block masses + evid/nonevid/other split, layers {12,16,20,24,27}).
  `slurm/sparse_s10.sbatch` (smoke/p1b/gated/frozen chains). Smoke 139736 (3
  samples, p1b + gated arms). Full cells on smoke pass: p1b N{8,16,32,64}×k{2,4,8},
  gated(S8) N{8,32,128}, frozen N32 — 30/cell strided from pools.
- Meanwhile: S9b twins training healthy (139714 ep10-arm, 139718 ep5-arm).
- [~12:40] S11 anchor BYTE-IDENTICAL (139735) → chains submitted: 139752 (vs-N) + 139753 (vs-k), S9 nfree adapter, oracle gate, nfree prompt.
- [~13:00] S10 smoke PASSED (softmax sums OK; gated nonevid mass EXACTLY 0 — mask verified in the photograph itself). Full arms submitted: p1b/gated/frozen.

## [2026-09-01 ~14:30] S10 P1b ARM LANDS (139760) — H-S10a MET: THE FORMULA IS PHOTOGRAPHED

`outputs/sparse/s10/p1b_N{8,16,32,64}/` (30/k-stratum, k∈{2,4,8}, head-mean over
28 heads, per-row softmax sums asserted). L20 evidence-mass table (evid/nonevid/
other): k=4: 0.31/0.31/0.38 @N=8 → 0.21/0.49/0.30 @16 → 0.14/0.64/0.22 @32 →
0.09/0.74/0.17 @64 — dilution watched happening. Share-law fit per layer:
| L | R² | s | C |
|---|---|---|---|
| 12 | 0.989 | −0.90 | 10 |
| 16 | 0.992 | −0.65 | 8 |
| **20** | **0.969** | **+0.30** | **8** |
| 24 | 0.992 | −0.95 | 8 |
| 27 | 0.933 | −1.00 | 19 |
N-doubling sign check: **9/9 decreasing at every layer** (H-S10a's clause).
Reading: the trained read's L20 advantage is a mere e^0.30 ≈ 1.35× per-block edge
over competitors — γ-gain (LORAMECH ×3-6 amplitude) lives in VALUE content, not
attention selectivity; with (N−k) competitors present the share must dilute
exactly as the formula says. m = k·eˢ/(k·eˢ+(N−k)+C): observed, not inferred.

## [2026-09-01 ~16:20] S9b-ep5 LANDS (139718) — k12 dent CLOSED; new low-k wobble to disambiguate via the ep10 twin

`outputs/sparse/s9b_ep5/20260901_142255_gated/` (5-ep budget, best ep3 val 1.000,
TEST_IID 1.000): **1.0000 @8 · 1.0000 @16** (S9's k12@16 dent GONE) · 0.7533 @32
(pf 0.06). @32 anatomy: **k12 11/11 (was 2/11 in S9)** — the k-balanced coverage
did its job — but k7 5/11 + k6 9/12 regressed (S9 had them perfect); k16 5/11.
Candidates: ep5 best-epoch selection noise (val rides 60 samples) vs a mixture
calibration shift; the ep10 twin (139714, training) disambiguates. Ladder+S4
evals fired: 139774/139775 (multi-partition, nfree). 

## [2026-09-01 ~17:00] S11 vs-N LANDS (139752) — H-S11a MET; NO non-text N-channel; refutation branch dead

`outputs/sparse/s11/gate_oracle_N{8..128}/` (S9 nfree adapter, oracle gate,
N-free prompt = byte-identical text at every N; 50×4+40 pairs, ctrl 12, seed 0):
| locus | α [95% CI] | medians N=8→128 |
|---|---|---|
| read L20 | **+0.035 [−0.043,+0.123]** | 28.5→25.7 FLAT |
| read L16 / L28 | +0.015 / +0.004 (CIs ∋ 0) | flat |
| verdict L20 | −0.016 [−0.048,+0.005] | ~52 flat |
Margins: median **+11.7 CONSTANT at every N** (P1g chain drifted +5.6→+3.7 — the
N-free margins are both larger and flatter). Flip-changes-answer: 1.00/0.90/0.90/
0.90/0.90 (bar ≥0.7). **|α| ≥ 0.3 refutation branch NOT triggered** — with zero
text variation the read shows no N-dependence: the equivalence smoke's claim now
holds at the α level, closing the S7-(ii) channel question completely. γ (vs-k,
H-S11b) pends on 139753.

## [2026-09-01 ~17:30] S10 gated+frozen arms land (139761/62) — H-S10b MET at the read locus; H-S10c: THE INTERNAL GATE EXISTS (L24h20, AUC ≥0.993 flat, trained-in)

- **H-S10b**: gated (S8) L20 evidence share at fixed k across N∈{8,32,128}:
  k2 4.5% / k4 6.2% / k8 2.0% relative range — ≤10% band MET at the read locus
  (worst across ALL (k,L) = 16.3%, in low-mass off-read layers; reported).
  The share tracks k/(k+C′), N absent — the photograph agrees with S3/S11.
- **H-S10c (prominent per the band)**: in the UNGATED P1b arm a single FIXED head
  — **L24 h20 — separates evidence vs non-evidence blocks at AUC 0.9997/0.9927/
  0.9999/1.0000 @N=8/16/32/64**. Frozen best head: 0.848 → the selector is
  TRAINED-IN. Reading: the model already computes a near-perfect per-frame
  evidence verdict internally, but the read cannot exploit it — the head's mass
  obeys the same share dilution (H-S10a). The external gate (mask or the LR
  readout at ≥0.999/frame) is precisely the read-out of this internal signal.
  Self-gating exists at the representation level; the failure was always the
  softmax aggregation, never detection.

## [2026-09-01 ~18:10] S11 vs-k lands (139753) — H-S11b MET: γ(L20) = +1.19 [1.17,1.22], C=5 — the SAME γ as the P1g chain

`outputs/sparse/s11/gate_oracle_N64_k{1..32}/`. Decay-in-k through the S9 read is
γ 1.41/1.19/2.79 at L16/L20/L28 (all CIs exclude 0.5) — L20 numerically identical
to the P1g-chain fit (1.19): the k-resolution law is a property of the gated
softmax read itself, not of any adapter's calibration. **S11 CLOSED: both bands
MET.** F6 next.

## [2026-09-01 ~20:45] S9b ladder lands (139774) + wobble anatomy — interim, twin pending

`outputs/sparse/s9b_ladder/`: N=64 0.680 (k≤8 0.975, **k12 9/10**, k16 5/10, pf
0.193) · N=128 0.607 (k≤8 0.986, **k12 7/9**, k16 4/9, pf 0.267; pf concentrated
in untrained k≥24). The k12 recovery holds at every N (S9: 0.18-0.59 → S9b:
0.9-1.0). Wobble anatomy: ALL 12 low-k errors are EXACTLY +1 (k6→"7" ×3, k7→"8"
×9) — calibration crowding, not noise: densely training answers 9-16 packs 17
levels into a code whose per-unit signal shrinks as (k+C)^-1.19 (S11), and the
6-8 boundaries compressed. S8 (same dense grid, declared-N prompt, NO wobble)
suggests the prompt-N text also FUNDS code capacity as a context register — the
channel that snaps P1g also helps S8. ep10 twin (139714, ep7) decides the budget-
noise component. Also recorded for the write-up: the park seq16 pool is
anchor-classed (K0-8, K12, K16 ONLY — 9-11/13-15 don't exist in any real
training data); Tal's question prompted the S9c option (regenerate a uniform-gold
seq16 pool — the data-side fix; NOT launched, awaiting authorization).

## [2026-09-01 ~23:20] S9b ep10 TWIN LANDS (139714) — WOBBLE = CHECKPOINT-DRAW NOISE; S9b(ep10) MATCHES S8'S PROFILE WITH NO N IN THE TEXT

`outputs/sparse/s9b_vn_nfree/20260901_140231_gated/` (best ep2 val 1.000,
TEST_IID 1.000): **1.0000 @8 · 1.0000 @16 · 0.8467 @32** — N=32 anatomy k6 12/12,
k7 11/11, k8 11/11, **k12 10/11, k16 11/11**, k≥24 = 0 (coverage), pf 0.073
(untrained high-k). The ep5 arm's +1 wobble does NOT reproduce → it was
CHECKPOINT-DRAW noise (both arms' best-val epochs tied at 1.000 on the 60-sample
val; different draws, ±3-5% in-window calibration variance invisible to
val-by-decode). My earlier "code-crowding/context-register" reading is DOWNGRADED
to a rejected hypothesis for the wobble (the +1 structure was real but its cause
was the draw, not capacity; the crowding story survives only as an unproven
conjecture — logged honestly). 5-ep default caveat recorded: same-val checkpoints
differ by a few % in calibration; prefer the cheaper budget but expect draw
variance at margin-thin strata. Twin adapter PROMOTED:
`checkpoints/sft_fenced_gated_vn_nfree_adapter`. Twin long-N ladder + S4
submitted (139977/139978) — the last two cells of the extended wave.

## [2026-09-02 ~03:20] TWIN LONG-N LANDS (139977/78) — S9b(ep10) IS THE BEST-MEASURED ADAPTER OF THE CAMPAIGN

`outputs/sparse/s9bT_ladder/` + `s9bT_s4/`:
| N | overall (exam) | k0-k16 (trained support) | k≥24 |
|---|---|---|---|
| 64 | 0.7333 (pf 0.200) | **110/110 exam · 170/170 S4 = PERFECT** | 0 |
| 128 | 0.6600 (pf 0.280) | **99/99 exam · k12 20/20 + k16 20/20 S4 = PERFECT** | 0 |
**k16@128 = 9/9 AND 20/20 — the cell S8 read 1/9 / 5/20.** Every failure in both
cells is an untrained answer (k≥24; pf concentrated there — no range hint). Full
twin ladder: 1.000/1.000/0.847/0.733/0.660 with the ENTIRE loss = untrained
strata.

═══ WAVE-3 EXTENSION VERDICT (S9b/S10/S11, all cells closed 2026-09-02) ═══
The N-free + k-balanced adapter (S9b ep10, `sft_fenced_gated_vn_nfree_adapter`)
is EXACT on every trained answer value (0-16) at every N ∈ {8..128} — the
strongest profile measured in this campaign, on the upstream-faithful prompt
(labeled: N-free prompt anchor; S8 remains the N-prompt canonical). The ep5 twin
= budget ablation (same recipe: −4h, +draw-variance at margin-thin strata). S10
photographed the share law (R² .93-.99) and found the trained-in internal
evidence head (L24h20, AUC ≥0.993 flat — detection was never the problem);
S11 closed the N-channel taxonomy (α +0.035 with byte-identical text; γ 1.19
adapter-independent). OPEN after the extension: k>16 coverage (extend the grid),
the k≈N regime, HF gate recall, the S9c data-side variant (uniform-gold seq16
regeneration — specced, unlaunched). Every number traces to INDEX.md run dirs.
═══════════════════════════════════════════════════════════════════════════════

## [2026-09-02 ~09:40] Post-close verification (Tal's confirm request) + one correction

Consolidated S9b-ep10 trained-support tally (gold 0-16, oracle gate, ALL cells,
recomputed from prediction CSVs): 150/150 · 150/150 · 127/128 · 290/290 · 275/275
@N=8/16/32/64/128 = **992/993 (0.9990)**. The single miss:
steps_in_room_N32_K12_0017 → "13" (+1). NOT literally 100% — the write-up claim
is "exact on 992 of 993 trained-support evaluations, perfect at 4 of 5 lengths
incl. 8× and 16×". Corrected in the RESULTS S9b entry: the "(32-cell: 2 misses
in 22)" phrase was wrong (it was 1 miss in 128); fixed in place in the fresh
entry, correction logged here.

## [2026-09-15 ~02:00] GATED layout twins launched (Tal's ask, via the SELFGATE session)

The 2026-09-14 query-side layout ablation (outputs/sparse/layout/, jobs 149076-78)
ran UNGATED; the gate × no-replica quadrant was empty. Launched now, byte-matching
the ablation's recipe + `--gate oracle`: **150433 qlast-once-gated (n308)** +
**150434 qfirst-once-gated (n310)** → `outputs/sparse/layout/{qlast,qfirst}_once_ep5_gated/`.
Prediction (logged pre-result): with the oracle gate carrying the question-dependence,
gated qlast-once ≈ gated replica (count = count-the-visible-blocks) — if MET,
question-BLIND per-frame encodings suffice under external selection (cacheable
frame blocks; conditioning is only needed when the read must select). If gated
qlast-once still collapses, conditioning is load-bearing even for the gated read.

## [2026-09-15 ~05:00] Budget-matched gated-replica 5-ep control launched (Tal)

**150572 (n317)** → `outputs/sparse/layout/replica_ep5_gated/` — identical recipe to
the gated twins (--gate oracle, 5 ep, same roots/exclusions/exams), --layout replica.
Completes the 2×3 factorial {replica, qfirst-once, qlast-once} × {ungated, gated}
with every gated cell budget-matched. Note for the table: ungated replica_ep5's
N=128 cell was skipped=150 in the 149076 run (instrument note, cell missing).

## [2026-09-19 ~18:50] S10b — photograph fill-in cells LAND (frozen ladder + P1b@128): the frozen read has NO evidence preference at any N; both arms' per-frame mass falls as ~1/N; fine-tuning's edge is a constant ×1.07

Tal asked for the "softmax is not enough" companion figure to the Hahn panel. New wrapper
`slurm/sparse_s10b.sbatch` (ARM/NLIST dash-separated; `--attn-sharpen` support added to
`scripts/sparse/probe_attn_photo.py`, default 0 = unchanged path). Jobs 153242 (frozen N=8/16/64,
rtx6k n318, 14m), 153243 (frozen N=128, h200, 17m), 153244 (P1b N=128, h200, 17m); 90 samples
(30 × k∈{2,4,8}) per cell → `outputs/sparse/s10/frozen_N{8,16,64,128}/`, `p1b_N128/`.
Per-FRAME mass at the answer row, L20, k=4 (median over samples × heads; evidence / non-evidence):
| arm | N=8 | 16 | 32 | 64 | 128 | slope (evid) | ratio evid/nonevid |
|---|---|---|---|---|---|---|---|
| frozen | .049/.046 | .028/.028 | .014/.014 | .0073/.0075 | .0038/.0039 | 0.94 | 1.08, 1.00, 0.97, 0.95, 0.96 |
| P1b | .073/.069 | .047/.043 | .026/.024 | .014/.013 | .0074/.0071 | 0.83 | 1.07, 1.10, 1.06, 1.07, 1.04 |
| gated (S8) | .016/0 | — | .0165/0 | — | .0165/0 | −0.02 | ∞ |
Selector head L24h20 (P1b): evid/frame slope 0.63; evid:nonevid ratio 1.6/1.7/2.0/2.6/3.2 at
N=8..128 vs ≈N/k = 2/4/8/16/32 needed to hold its share; frozen L24h20 ratio ≈1.0 at every N
(selector entirely trained in). Readings: (1) the frozen read gives an evidence frame no
preference beyond N=8 — dispersion is uniform; (2) fine-tuning buys a constant ×1.07 per-frame
edge and does not change the ~1/N slope (weights-level rigidity, matching the flip exponents);
(3) the gate holds per-frame evidence mass flat at 0.016 from 8 to 128. Figure + CSV:
`outputs/sparse/fig/F10_dispersion_hahn.{png,csv}` (script `scripts/sparse/fig_s10b.py`):
a) per-frame mass vs N, b) flip response vs N (ARMOR/N2/S11 medians), c) L24h20.
Smoke note: tau=1.0 vs the S10 smoke on a different GPU differs per head row by ≤1.25e-2 (bf16
across hardware); the tau sweep job 153246 self-anchors tau-off vs tau=1.0 on the SAME GPU (<1e-4
required) before running. S0 accuracies per tau (N=32/64/128): 1.0: .85/.34/.23 · 1.5: .48/.36/.23
· 2.0: .35/.28/.25 · 3.0: .23/.13/.13 · 4.0: .19/.12/.06 (mean signed error turns positive).

## [2026-09-19 ~20:40] S10b TAU SWEEP LANDS (153246, 1h27, h200) — sharpening the read moves mass to the SINK and away from the evidence; at N=128 the evidence share FALLS with τ; the coupling photographed

Self-anchor PASS (same GPU, tau-off vs tau=1.0: 840 rows, max|diff| 0.00e+00). Cells
`outputs/sparse/s10/p1b_N{32,128}_tau{1.5,2,3,4}/` (90 samples each; τ on layers ≥12 via
module.scaling, photographed scores scaled identically = the S0 intervention). L20, k=2,
head-median; masses are TOTALS at the answer row (evidence = 2 frames, competitors = N−2):
| N | τ | competitor | evidence | prompt+sink | max/mean over evidence | S0 exact |
|---|---|---|---|---|---|---|
| 32 | 1 | 0.738 | 0.053 | 0.148 | 1.04 | 0.85 |
| 32 | 1.5 | 0.672 | 0.046 | 0.172 | 1.06 | 0.48 |
| 32 | 2 | 0.619 | 0.039 | 0.198 | 1.10 | 0.35 |
| 32 | 3 | 0.474 | 0.023 | 0.291 | 1.21 | 0.23 |
| 32 | 4 | 0.292 | 0.012 | 0.413 | 1.35 | 0.19 |
| 128 | 1 | 0.893 | 0.0146 | 0.053 | 1.03 | 0.23 |
| 128 | 2 | 0.840 | 0.0124 | 0.081 | 1.10 | 0.25 |
| 128 | 4 | 0.720 | 0.0043 | 0.062 | 1.36 | 0.06 |
Readings: (1) temperature does remove competitor mass (0.74→0.29 at N=32) but the freed mass
goes to the prompt/sink tokens, NOT to the evidence — evidence mass falls monotonically with τ
at both N (×4.5 at N=32, ×3.4 at N=128); (2) at N=128 the evidence per-frame mass drops BELOW
the non-evidence per-frame mass under τ (0.0022 vs 0.0057 at τ=4): the ×1.04 edge is smaller
than the within-frame logit spread, so sharpening amplifies the wrong frames; (3) uniformity
over the evidence degrades (max/mean 1.03→1.36) while accuracy collapses in lock-step. This is
the two-requirement coupling measured on the production read: one temperature cannot zero the
competitors without un-flattening (and here even shrinking) the evidence. Figure + CSV:
`outputs/sparse/fig/F11_temperature_coupling.{png,csv}` (fig_s10b.py). Small-N cells (153259,
153260) still queued.

## [2026-09-19 ~21:30] S10b SMALL-N LANDS (153279 photo N=2/4 × 3 arms; 153280 flip fill) — at FIXED base count the gated read is flat from N=2 to 128; the pooled small-N decline was k-composition

Tal asked to extend the plots to N=2,4. Photograph pools 2/4 added to `probe_attn_photo.py`
(`data/mmred_images_park/seq_len_{2,4}/all_uniform`); cells `s10/{frozen,p1b,gated}_N{2,4}/`
(60/90 samples). Flip fill (`slurm/hahn_small_fill.sbatch`, h200 10m): `armor/hahn/short_N2/`
(plain/repjoint/fenced, ARMOR protocol; the 09-14 short chain had left N=2 empty),
`loramech/n2_hahn/p1fence_ep10_N{2,4}/` (P1b read, N2 protocol), `sparse/s11/gate_oracle_N2/`.
Per-frame mass L20 (k=1 at N=2, k=2 at N=4/8): frozen .082/.069/.048 (ratio 1.04/1.06/1.04),
P1b .172/.117/.087 (0.96/1.05/1.09), gated evid .027/.029/.029 — the 1/N trend and the constant
gap hold down to N=2. Flip response, POOLED medians: gated 72.5 (N=2), 40.4 (4), 28.5 (8) — a
decline that is NOT N: small-N pools contain only gold 0–1 (N=2: g0 31, g1 19) vs g0–g8 at N≥8,
and the gated response depends on k. **At fixed base count (flip 0→1), L20:**
| N | frozen read | fenced fact | trained read | gated read |
|---|---|---|---|---|
| 2 | 34.7 | 40.4 | 98.4 | 74.3 |
| 4 | 28.1 | 37.6 | 69.0 | 74.5 |
| 8 | 14.6 | 32.4 | 52.8 | 75.7 |
| 16 | 7.7 | 41.0 | 39.7 | 75.8 |
| 32 | 6.3 | 36.1 | 31.1 | 73.6 |
| 64 | 2.0 | 39.2 | 22.5 | 74.6 |
| 128 | 1.2 | 42.2 | 15.5 | 77.2 |
(n = 31/14/7/6/6/6/5 per cell; gold-1 and gold-2 strata agree: gated 40.7→43.6, 37.7→36.8.)
Readings: (1) the gated read is flat in N at every fixed k from N=2; (2) the frozen and trained
reads decay from N=2 onward — no small-N plateau; (3) the fenced fact is flat from N=2. Figure
F10 updated: panel b keeps the pooled 8–128 curves (the α protocol) and gains an inset with the
fixed-count series over 2–128; panels a/c extend to N=2. CSVs: `fig/F10_flip_stratified_gold0.csv`,
`fig/F10_flip_medians.csv`, `fig/F10_dispersion_hahn.csv`. Caveat: small-N strata n=5–14.

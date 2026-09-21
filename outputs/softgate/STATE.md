# SOFTGATE — STATE (append-only, newest last)

## [2026-09-20] Campaign created (Claude, authorized by Tal "run all of those") — brief written, patches next
- [21:10] smoke/anchor job 153852 submitted (probe byte-identity, B=1≡no-gate, B=8 smoke, trainer anchor, eval+train smokes incl. multitask).
- [21:32] SMOKE 153852: probe anchor BYTE-IDENTICAL; B=1 ≡ no-gate (18 rows, 0.000e+00); B=8 live (max Δ 145.9); trainer anchor P1b exam_ff_N8 = 1.0000; eval+train soft-gate smokes ran. Multitask smokes stopped at the trainer's "--nfree-prompt is count-only" guard (REDUX v3 fix pending).
- [21:32] SUBMITTED A1 alpha(B) chains (a100-public, 4d_1g, --time 5h): B1=153859 B2=153860 B8=153861 B32=153862 B128=153863 ; A2 no-retrain accuracy sweep 153865 (B=2,8,32,128 @N=32/128).
- [21:33] SUBMITTED A3 trainers (l40s-shared, 24h_1g, --time 14h; S9 recipe: nfree, oracle gate, 5 ep, roots seq8+seq16, excludes 24 files; in-job LONGN N8/16/32 — N64/128 ladders later on h200): B8=153866 B128=153867 Bhard=153868 (B=hard is the matched 5-ep hard-gate control).
- [21:34] REVIEW item 4: alpha(B) chains run on S9b while S11 (B=inf) ran S9 → added BONUS=0 (hard-gate) chain on S9b 153874 so the B-curve carries its own endpoint; adapters to be named in every caption. #SBATCH --time defaults added to the new wrappers (item 5).
- [23:22] A3 trainers' [data] line (all three arms): 728 samples after exclusion (excluded 502 — the count the review panel predicted), gold-hist g0:34 g1:80 g2:80 g3:79 g4:82 g5:79 g6:79 g7:101 g8:80 g12:17 g16:17. Note g0 is thin (34) — relevant to REDUX exists 'no' labels (review item 12).

## [2026-09-21 00:18] A3 LANDS (153866/67/68, 2h40 each, 728 samples, 5 ep) — the SOFT gate is WORSE than the law's competitor-equivalent bookkeeping: B=8 collapses to 0/107 on k≤8 at N=32 (2×) while the hard twin holds; both soft arms OVERcount

| arm | N=8 | N=16 | N=32 overall | N=32 k≤8 | N=32 k12 / k16 | mean signed err @32 | best ep |
|---|---|---|---|---|---|---|---|
| B=8 (ε(32)≈3.5) | 0.973 | 0.860 | 0.147 | **0/107** | 10/11 / 0/11 | **+2.04** | 4 |
| B=128 (ε(32)≈0.23) | 1.000 | 0.913 | 0.633 | 95/107 (0.89) | 0/11 / 0/11 | +1.98 | 3 |
| hard (ε=0) | 1.000 | 0.933 | 0.767 | 95/107 (0.89; k4 = 1/12 wobble) | 11/11 / 9/11 | −1.74 | 4 |
Readings (pre-band, descriptive): (1) H-A3's N_eff prediction ("B=8 passes at 2×") is REFUTED at N=32
already — the soft-gated reader does not degrade gracefully with ε; it snaps UP (B=8 answers land on
12 and 32; mse +2.0). Interpretation to test: the residual competitor mass (N−k)/B is an N-proxy
available during training (ε ≈ 0.9–2 at N=8–16 for B=8); the reader calibrates on it like it did on
the prompt-declared N (S7), and the calibration breaks the moment N leaves the window — a leaky gate
recreates the length dependence through CALIBRATION, not through dilution. (2) B=128 keeps k≤8 at 2×
(ε≈0.23 too small to serve as a proxy?) but loses k12/k16 entirely where the hard twin keeps them —
the same +2 bias. (3) The hard twin at 5 ep shows a k4 = 1/12 hole at N=32 (S9b-ep5-style margin-thin
wobble; checkpoint-draw variance). Next: N=64/128 ladders (h200, job 154014) and N=32 cross-evals
(each arm under the other gates, job 154015): if the B=8 arm evaluated under the HARD gate recovers,
the failure is calibration to ε, not capacity.

## [2026-09-21 ~09:30] A1 α(B) LANDS (153859–153863 + hard chain 153874; S9b reader, N-free text, oracle soft gate; 50 pairs, 40 @128) — the exponent slides with the penalty; law within 0.07 for B ≥ 8; the gate-removed reader is CONTENT-BLIND

α = −slope of log median ‖Δh‖ (L20, answer row, evidence flips) vs log N over 8..128; 2000-pair bootstrap.
| B | α (95% CI) | law (s=0.30, C=8, k=4) | medians N=8/16/32/64/128 |
|---|---|---|---|
| 1 (gate removed) | −0.02 [−0.04, 0.02] | 0.72 | 2.25 / 2.07 / 2.24 / 2.15 / 2.34 — **at the bf16 floor at every N** |
| 2 | 0.66 [0.64, 0.67] | 0.48 | 12.7 / 11.0 / 10.1 / 3.3 / 2.4 — plateau to 32, then to the floor |
| 8 | 0.17 [0.13, 0.22] | 0.15 | 24.6 / 21.0 / 20.0 / 19.3 / 14.0 |
| 32 | 0.10 [0.06, 0.16] | 0.03 | 28.6 / 24.6 / 23.5 / 22.8 / 21.3 |
| 128 | 0.07 [0.00, 0.15] | 0.01 | 29.7 / 26.1 / 25.9 / 24.5 / 24.1 |
| ∞ (hard, 153874) | 0.03 [−0.06, 0.12] | 0.00 | 29.9 / 26.5 / 26.5 / 26.1 / 26.9 — reproduces S11's 0.035 on a fresh chain |
Bands (§5 within-chain H-A1): **monotone slide MET for B ∈ {2, 8, 32, 128, ∞}** (0.66 > 0.17 > 0.10 > 0.07 > 0.03,
CIs of 2 / 8 / 128 disjoint); α(8) = 0.17 vs law 0.15, α(32) = 0.10 and α(128) = 0.07 sit ≤ 0.07 above the law
(the law's near-zero tail is flattened by a residual the fit does not model — see A2 below: the reader adds the
residual mass to its count, so its response never fully reaches the hard-gate value). **Clause "α(B=1) ≥ 0.5" NOT
MET, for an informative reason: with the gate removed the S9b reader's answer row does not respond to a
content flip at all** — median 2.2 at every N versus 10.4 (frozen) and 26 (P1b) at N=8 — while the same
frame's in-block fact locus responds normally (L20 rep_t ≈ 50). The gate-trained reader is a
visible-block counter: it reads "how many blocks reach me", not "what the blocks say". That is the selector
atrophy of the gate panel measured at the reader itself, and it is what makes the exponent in the B=1 cell
undefined rather than 0.72. **B = 2 is not a power law**: flat 12.7→10.1 over 8..32, then the response falls
to the floor at 64/128 (the residual (N−k)/2 ≈ 30–60 block-equivalents swamps one block); the 0.66 fit is
the floor's doing — report the shape, not the number. Law refit over B ∈ {2..128}: C = 67 (RMSE 0.27) —
i.e., the law's exponent is right but its amplitude bookkeeping is not what sets the slide's tail; the
reader's arithmetic is (next entry).
Figure: `outputs/softgate/fig/F12_alpha_vs_B.{png,csv}` (scripts/softgate/fig_alpha.py; panel a raw curves
per B, panel b α vs B with the law curve). Per-locus fits also in the JSON `outputs/softgate/a1_alpha_L20final.json`.

## [2026-09-21 ~09:40] A2 no-retrain sweep (153865, TIMEOUT at 5 h after B=32 @N=32): the gated reader's answer IS the share law's numerator — pred = k + (N−k)/B, to the unit

S9b reader, no retraining, oracle soft gate B, N-free prompt, 150 exam samples per N (k ∈ {0..8, 12, 16, 24, 32, …}):
| B | N | exact | k≤8 exact | parse-fail | median(pred − gold) at k = 0..8 | (N−k)/B at k = 0..8 |
|---|---|---|---|---|---|---|
| 2 | 32 | 0.073 | 0.000 | 0.15 | 16 15 14 13 12 11 10 9 8 (= 16 for all k, the count vocabulary's cap) | 16.0 … 12.0 |
| 2 | 128 | 0.000 | 0.000 | 1.00 | — (no parsable number at any k) | 64 … 60 |
| 8 | 32 | 0.073 | 0.000 | 0.07 | **4 4 3 4 4 3 4 3 3** | **4.0 3.9 3.8 3.6 3.5 3.4 3.2 3.1 3.0** |
| 8 | 128 | 0.000 | 0.000 | 0.41 | **16 15 14 13 12 11 10 9 8** (pred = 16 at k = 0…8; 40% emit 160–166) | **16.0 … 15.0** |
| 32 | 32 | 0.073 | 0.000 | 0.07 | run.log only (mae 1.55, signed +0.28; k=16 11/11) — predictions lost to the timeout | 1.0 … 0.75 |
| 32 | 128 · 128 | 32 · 128 | — | — | — | **refilled: 154064 (B=32 @128), 154065 (B=128 @32+128)**, a100-public, 4d_1g, --time 3 h / 4 h | 4 / 0.25 / 1 |
Reading: with B = 8 the reader reports **k + (N−k)/B with an effective evidence edge E ≈ 1.0** (fit of
pred = k + (N−k)/(B·E): E = 1.00 at N=32, 1.29 at N=128 where the cap intervenes). The residual competitor
mass is not "noise the reader tolerates"; it is **added to the count as block-equivalents** — 24 blocks at
weight 1/8 read as 3 frames. E ≈ 1 is the same content-blindness A1 measured (α(B=1) at the floor): the
reader assigns evidence and non-evidence blocks the same weight, so it can only count what the gate lets
through. Where (N−k)/B exceeds the trained count range (B=2, or N=128), the answer saturates at the
vocabulary cap 16 or leaves the vocabulary (160–166 = "16" followed by a digit; blank). So the no-retrain
finite gate is not "slightly worse than hard": it is exactly wrong by the residual, and the error grows
linearly in N. H-A2 (ε-bookkeeping, "accuracy ≈ (1−ε)^N of the hard gate") is the wrong model: the
failure is a bias of (N−k)/B, not a per-frame error rate.

## [2026-09-21 ~09:45] A3 cross-gate evaluations at N=32 (154015; ladders 154014 running on h200) — calibration, not capacity: the B=128 reader is exact under the hard gate; the hard reader under a finite gate adds the residual; the B=8 reader has learned the residual as its N-proxy

Retrained 5-ep arms (728 samples, seq8+seq16 roots) evaluated under each other's gate, N=32, 150 samples:
| reader \ eval gate | hard | B=128 | B=8 |
|---|---|---|---|
| hard (a3_Bhard) | 0.767 (own; k≤8 95/107, k4 glitch) | **0.513** — pred = k or k+1 (residual 0.25 rounds up at k=0,3,4,5) | **0.120** — pred = k + 4 exactly for k≤5 (0→4, 1→5, 2→6, 3→7, 5→8), then snaps to {8, 12, 16} |
| B=128 (a3_B128) | **0.707 — k≤8: 107/107 exact** (4× training length; own gate gave 0.633 = 12 residual-rounding errors) | 0.633 (own) | — |
| B=8 (a3_B8) | **0.100** — pred ≈ 2k for k≤5 (1→3, 2→3/4, 3→5, 4→7, 5→8), k≥6 → 12…32 | 0.187 — signed +3.8 | 0.147 (own; k≤8 0/107) |
Readings. (1) The **B=128 reader is a hard-gate reader in disguise**: trained where the residual was ≤ 0.125
block-equivalents, it transfers to the hard gate at N=32 with k≤8 perfect; its own-gate loss at N=32 is the
0.25-block residual tipping ~1 in 9 answers by one. **Capacity is not the issue.** (2) The **hard-trained
reader under a finite gate does the A2 arithmetic** (k + (N−k)/B, quantized to its count vocabulary {0..8, 12, 16}):
the read is a magnitude sum whatever the reader was trained on. (3) The **B=8 reader is the calibration case**:
at training (N ≤ 16) the residual (N−k)/8 ≤ 2 co-varied with N and k, and the reader learned to read the
count from the joint picture (evidence mass, residual mass). Remove the residual (hard gate) and it infers
"k ≈ N" (pred ≈ 2k, then 32); enlarge it (its own gate at N=32) and it overcounts. The residual mass is the
N-proxy predicted on 2026-09-21 00:18 — confirmed by the cross-evals. **Verdict on H-A3 (two-sided):** the
soft gate fails at 2× because of calibration to the residual, not because the reader lacks capacity; the
only N-invariant training signal is a residual of zero (hard gate) or one that is provably N-invariant.
Pending from 154014: B8 @ B8 N=64 landed at 0.020 (parse-fail 0.11, signed +10.4 — the residual (64−k)/8 ≈ 8
added); the rest of the N=64/128 ladders come with the waiter.

## [2026-09-21 11:20] SOFTGATE CLOSES — A2 refill (154064/154069) and A3 N=64/128 ladders (154014) landed; the law holds to the unit at N=128; no finite penalty extrapolates

A2 (S9b, no retrain), predictions at N=128 for k = 0…8: **B=32 → 4 5 6 7 8 9 10 10 12** (law k + (128−k)/32 = k+4:
exact at 8 of 9 k); **B=128 → 1 2 3 4 5 6 7 8 9** (law k+1: exact at 9 of 9). B=128 at N=32: **0.833 exact,
k≤8 107/107** (residual 0.25 rounds away). Final A2 table (exact / k≤8 exact):
| B | N=32 | N=128 |
|---|---|---|
| 2 | 0.073 / 0.000 (all "16") | 0.000 / 0.000 (nothing parsable) |
| 8 | 0.073 / 0.000 (k+4) | 0.000 / 0.000 (k+16, 41% out of vocabulary) |
| 32 | 0.073 / 0.000 (k+1; run.log only) | 0.027 / 0.000 (k+4) |
| 128 | **0.833 / 1.000** (k+0.25) | 0.073 / 0.000 (k+1) |
| ∞ (S9b, S11 protocol) | 1.000 on k≤8 | 1.000 on k≤8 |
A3 own-gate ladders at N=64 / 128: B=8 reader 0.020 / 0.020 (signed +10 / +15); B=128 reader **0.493 / 0.067**
(N=64: k≤8 74/90 with the 0.5-block residual tipping 16 answers by one; N=128: residual ≈ 1 → k+1 everywhere,
20% out of vocabulary); hard 5-ep control **0.693 / 0.547** (k≤8: 90/90 at 64; 76/81 at 128 — the 5-ep,
728-sample reader is a touch below S9's 1.000 at 16×, its misses are digit run-ons "72", "82", "128").
**Verdicts.** H-A1 slide MET for B ≥ 2 (B=1 undefined: content-blind reader). H-A2 ε-bookkeeping REPLACED by
the bias law pred = k + (N−k)/B (E ≈ 1), confirmed at N = 32 and 128, B = 8/32/128. H-A3 two-sided: CALIBRATION
(B=128 reader exact under the hard gate; B=8 reader learned the residual as its N-proxy). Campaign closed;
INDEX updated. Paper use: F12 + the A2 prediction table are the "magnitude read" exhibit — the gated reader
counts block-equivalents of attention mass, and only the hard gate makes that quantity N-invariant.

# SOFTGATE campaign — the gate as the limit of boosting the evidence frames (2026-09-20)

**Status: AUTHORIZED (Tal, 2026-09-20, "run all of those").** Scripts: the existing gated probe and
trainer gain one flag (`--gate-bonus B`); no new mask code in gnnformer/. Runs: `outputs/softgate/`.
Log: `STATE.md` (append-only). Sister campaigns launched together: REDUX v3 (`outputs/redux/`) and
FIXEDK (`outputs/fixedk/`).

## 0. Question
Tal: "why can't we just scale the attention of the evidence frames?" Answer to test: a uniform logit
bonus on the evidence keys (equivalently a finite penalty −log B on the non-evidence keys) gives an
evidence share k·e^s/(k·e^s + (N−k)/B + C). N stays in the denominator unless B grows like N; the hard
gate is the B→∞ limit. Prediction: the read's length exponent α slides continuously from the ungated
value (≈0.8) to the gated value (≈0.04) as B grows, and the B needed for a fixed accuracy at N scales
with N. A uniform bonus does NOT disturb uniformity over the evidence (unlike temperature), so the
"sink capture" seen under sharpening should be absent: evidence mass should RISE with B.

## 1. Implementation (one flag, two scripts)
`--gate-bonus B` (float, default 0 = hard gate, byte-identical path): entries that the oracle gate
would set to MASK_MIN but the plain fence leaves open get −log B instead of MASK_MIN. Added to
`scripts/sparse/probe_hahn_gated.py` (gated_mask) and `scripts/sparse/train_sft_gated.py`
(fenced_setup). Requires `--gate oracle`. B=1 ⇒ penalty 0 ⇒ ungated mask through the gated path.

## 2. Cells
| cell | what | cost |
|---|---|---|
| A0 anchor | probe_hahn_gated 3-pair no-flag run must byte-match `outputs/_scratch/sparse_smoke/probe_anchor/pairs.csv`; trainer eval-only P1b exam_ff_N8 must read 1.0000 | minutes |
| A1 α(B) chains | probe_hahn_gated, S9b nfree adapter (`checkpoints/sft_fenced_gated_vn_nfree_adapter`), `--gate oracle --nfree-prompt --gate-bonus B`, B ∈ {1, 2, 8, 32, 128}, N ∈ {8,16,32,64,128}, 50 pairs (40 @128), controls 12, seed 0 — the S11 protocol; S11 itself is B=∞ | 5 chains × ~2.5 h |
| A2 accuracy, no retrain | canonical gated adapter evaluated with `--gate-bonus B` at N ∈ {32,128}, 150/cell, B ∈ {2,8,32,128} | ~2 h |
| A3 accuracy, retrained | S9 recipe (nfree, oracle gate, 5 ep, roots seq8+seq16, exclusions as S9) trained WITH `--gate-bonus B` for B ∈ {8, 128}; in-job ladder N=8..128 | 2 × ~7 h + evals |
| A4 photograph under bonus | probe_attn_photo at B ∈ {8,128} (needs the flag there too; only if time) | optional |

## 3. Pre-registered bands (fixed now)
- **H-A1 (monotone slide):** α_read(L20) is non-increasing in B with α(B=1) within the ungated CI
  [0.66, 0.97] and α(B=128) ≤ 0.2; α(B=8) between them with CI excluding both endpoints.
  Refuted if α(B=128) > 0.4 or the ordering is non-monotone beyond CI overlap.
- **H-A2 (calibration is competitor-mass specific):** the hard-gate adapter under B ∈ {2,8} at N=128
  loses ≥0.3 exact vs its hard-gate ladder (analogue of S1-control); under B=128 it loses ≤0.1.
- **H-A3 (retrained soft gate):** the B=128 retrained adapter reaches k≤8 exact ≥0.95 at N=128; the
  B=8 adapter does not (≤0.8) — the residual (N−k)/B term at N=128 is 15 competitors-equivalent.
- **H-A4 (no sink capture):** under the bonus, evidence mass at the answer row rises with B at fixed N
  (opposite of the temperature sweep). Descriptive if A4 runs.

## 4. Discipline
Smokes in `outputs/_scratch/softgate/`; anchors before full runs; explicit `--time`; no comma lists
in `--export`; partitions checked; QOS caps (12h_4g 3 jobs, 24h_1g 4 jobs) respected; STATE.md
append-only; RESULTS.md only on "log this". Every number traces to a run dir.

## 5. AMENDMENTS after the pre-launch review panel (2026-09-20, before any A1/A2/A3 result was read)
- **Adapters named:** A1/A2/A3 use S9b `sft_fenced_gated_vn_nfree_adapter` (S11's B=∞ chain used S9
  `sft_fenced_gated_nfree_adapter`); a BONUS=0 hard-gate chain on S9b is added so the α(B) curve carries
  its own endpoint. Every caption states the adapter.
- **H-A1 rewritten within-chain (S9b, N-free text):** α(B=1) ≥ 0.5 (CI excludes 0.2) > α(8) > α(128) ≤ 0.15,
  non-increasing with non-overlapping CIs between 1, 8 and 128; α(B=1) vs P1b's 0.80 is descriptive only
  (a gate-trained reader with the gate removed need not match P1b). Law prediction computed 2026-09-20 (m = k·e^s/(k·e^s + (N−k)/B + C), s=0.30, C=8, k=4, slope of
  log Δm vs log N over N=8..128): α ≈ 0.72 / 0.48 / 0.15 / 0.03 / 0.01 for B = 1 / 2 / 8 / 32 / 128
  (ranges over C∈{2,5,8}, k∈{2,4,8}: 0.67–0.87 / 0.37–0.67 / −0.11–0.25 / −0.18–0.07 / −0.08–0.02; the
  negative values are finite-N artefacts of the fit when (N−k)/B ≪ C). So the law says B=8 already
  flattens the read over 8..128 and the informative cells are B ∈ {1, 2, 8}. Test: one-parameter fit of C
  to the six α's (B = 1, 2, 8, 32, 128, hard) with RMSE ≤ 0.10. Refuters: α(2) outside [0.30, 0.70]
  (bonus does nothing or already gates); α(8) > 0.35; α(32) or α(128) > 0.15. 0.15 < α(8) ≤ 0.35 =
  "slower slide than the law with C=8; fit C and report", not a refutation.
- **H-A2 rewritten as ε-bookkeeping** (ε = (N−k)/B competitor-equivalents; hard-gate adapter's
  calibration assumes ε ≈ 0): reference = `outputs/sparse/s9bT_ladder` (S9b, nfree) per-k on k ≤ 8.
  Bands on the k≤8 stratum: B=32 @128 (ε≈3.9): loss ≥ 0.3; B=128 @128 (ε≈0.97): loss ≤ 0.15; B=32 @32
  (ε≈0.9) must match B=128 @128 per k (the ε-collapse test); B∈{2,8} @128 are sanity controls (near-total
  loss expected). Pre-registered signed error: mean signed error at N=128 linear in k with slope
  −ε/(C+ε) (C=8 ⇒ −0.11/unit at ε=0.97). N=64 cells added as A2b.
- **H-A3 two-sided:** the retrained B=128 arm at N=128 either reaches k≤8 exact ≥ 0.95 (C_eff ≥ 20 branch)
  or reads k ≤ 4 exact with k = 6–8 undercounted by one (C ≈ 8 branch); the B=8 arm (N_eff ≈ 23 at 128,
  inside the 2× regime) is predicted to PASS k≤8 ≥ 0.9 at N=64 and to sit between the B=128 arm and the
  hard arm at 128. Recipe named: nfree, oracle gate, 5 ep, replica, roots seq8+seq16, excludes the
  24-file list; the matched B=∞ twin is the `a3_Bhard` arm trained in the same batch. Cross-evals: each
  arm at its own B and at B=∞ (eval-only). N=64/128 ladders run eval-only on ≥80 GB after training.

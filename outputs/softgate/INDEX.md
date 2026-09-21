# outputs/softgate — INDEX (SOFTGATE campaign, 2026-09-20 → ; brief: CAMPAIGN_BRIEF.md, log: STATE.md)

Finite-penalty gate: gated columns get logit −log B (B = 1 no gate … ∞ hard mask). Reader = S9b
(`checkpoints/sft_fenced_gated_vn_nfree_adapter`) unless stated; N-free prompt; oracle gate.

| Experiment | Canonical run(s) | Headline |
|---|---|---|
| A1 α(B) flip chains | `a1_B{1,2,8,32,128,0}_N{8..128}/pairs.csv` (153859–63, 153874) | α = −0.02 (floor) / 0.66 (plateau→floor) / 0.17 / 0.10 / 0.07 / 0.03 for B = 1/2/8/32/128/∞; law 0.72/0.48/0.15/0.03/0.01/0; gate-removed reader content-blind (response 2.2 at all N) |
| F12 figure | `fig/F12_alpha_vs_B.{png,csv}` (scripts/softgate/fig_alpha.py) | α vs B with the share-law curve |
| A2 no-retrain accuracy | `a2_B{2,8}/`, `a2_B32/` (N=32, run.log only), `a2_B32_N128/`, `a2_B128/` (153865 TIMEOUT; refills 154064/154069) | pred = k + (N−k)/B to the unit at N=32 AND 128 (B=8 +4/+16, B=32 +1/+4, B=128 +0.25/+1); B=128 @N=32 0.833 with k≤8 107/107; every other cell 0 on k≤8 |
| A3 retrained arms (own gate) | `a3_B8/`, `a3_B128/`, `a3_Bhard/` (153866–68; 5 ep, 728 samples) | N=32: 0.147 / 0.633 / 0.767; B=8 k≤8 0/107 |
| A3 cross-gate evals N=32 | `a3eval_{hard_at_B8,hard_at_B128,B8_at_hard,B128_at_hard,B8_at_B128}_N32/` (154015) | B128@hard 0.707 with k≤8 107/107; hard@B8 0.120 (pred = k+4); B8@hard 0.100 (pred ≈ 2k) — calibration, not capacity |
| A3 ladders N=64/128 (own gate) | `a3eval_{B8_at_B8,B128_at_B128,hard_at_hard}_N64_128/` (154014) | B8 0.020/0.020; B128 0.493/0.067 (residual 0.5 → 1 block); hard 5-ep 0.693/0.547 (k≤8 90/90, 76/81) |

# outputs/sparse/ — INDEX (experiment → canonical run → headline)

Campaign: SPARSE, 2026-08-30→31 (brief `CAMPAIGN_BRIEF.md`, full log `STATE.md`,
figures `fig/F{1,2,3,4}*.png` + CSVs). Trainer/probe/gate: `scripts/sparse/`.
Adapter promoted: `checkpoints/sft_fenced_gated_adapter` (eval contract: gate REQUIRED).

| experiment | canonical run | headline |
|---|---|---|
| S0 sharpness sweep (eval-only P1b) | `s0_tau{1.0..4.0}_L12/`, `s0_tau2.0_L0/` (138976–81) | **no τ reaches the k-regime** (best k≤8 @64 ≈ 0.24 vs bar 0.80); threshold τ1.5; S6 not triggered |
| S1-control (P1b + gate, no retrain) | `s1_control/` (138982) | **CONFIRMED**: answers ≈N, mse **+20.79** @N=32 |
| S1 diverged runs (the gradient bug) | `s1_p1g/` (138975), `s1_p1g_lr1e4/` (139033) | ckpt recompute dropped the hook mask → grads cos 0.777 vs truth; fix = hold mask through backward (cos 1.000000, diag 139064) |
| **S1 P1g gated fenced-SFT (CANONICAL)** | `s1_p1g_r3/20260831_011807_gated/` (139066) | gated exams **1.000 @8 / 0.947 @16 / 0.780 @32-zs**; TEST_IID 0.986 |
| S1 ladder 64/128 (oracle gate) | `s1_ladder_64_128/` (139105) | 0.553 @64 / 0.440 @128; **k≤4 EXACT at every N (140/140)**; mid-k snaps to "N"; H-S1 headline refuted (k≤8 @128 = 0.79 < 0.85) |
| S3 α under the gate, vs-N | `s3/gate_oracle_N{8..128}/` (139107) | read α **+0.07 [−0.08,+0.20]** (ungated +0.80) — FLAT; margins stay positive; flip changes answer 80–98% at every N |
| S3 decay-in-k, vs-k @N=64 | `s3/gate_oracle_N64_k{1..32}/` (139108) | Δ ∝ (k+2)^−**1.19** [1.16,1.22] at L20 — the wall moved from N to k |
| S4 capacity in k (topped-up 20/gold) | `s4/eval/` (139106) | **c\*(64)=8, c\*(128)=7** (expected 24–48 → H-S4 missed); below c\*: exact at 16×; k≥96 @128 = parse-fail overflow |
| S2 gate (LR on L20 replica slots) | `s2/gate/` (captures 139109–14) | per-frame acc ≥**0.999 at every N, FLAT** (1 miss / 19,200 frames @128), d′ 6.4–8.4 |
| S2 model-gated exams | `s2/eval_short/` (139125), `s2/eval_long/` (139126) | 0.993/0.960/0.753 @8/16/32; @64 **identical to oracle** (0 gate errors on 9,600 frames); N=128 in `eval_long` |
| S5 MMReD-HF transfer (model gate) | `s5_hf/` (139127) | 0.900/0.700/0.540 vs P1b 1.000/0.820/0.540 → **H-S5 missed**; k≤5 near-perfect, loss = the same mid-k snap |

## Wave 3 (2026-08-31→09-01) — kill the snap

| experiment | canonical run | headline |
|---|---|---|
| Equivalence smoke (S8 hard gate) | `outputs/_scratch/sparse_w3/equiv/` (139240) | PASS 20/20 decodes; twins reproduce the snap with nothing hidden |
| S7 prompt-lie N=32 | `s7/N32_{ref,d16,d64}/` (139245) | **both clauses MET**: k12 0→1.00 under declared-16 (answers correct); snap target = declared string |
| S7 prompt-lie N=64 | `s7/N64_{ref,d16,d32}/` (139246) | replicates; k4 = 120/120 across all 6 lies; −1 ring precedes snap |
| **S8 virtual-N train (CANONICAL)** | `s8_vn/20260831_134617_gated/` (139247) | 1.000/1.000/0.853 in-window+2×; ep3 val 1.000 |
| S8 oracle ladder | `s8_ladder_64_128/` (139451) | **H-S8-main MET: k≤8 = 1.000 at EVERY N**; k12 perfect ×5; pf 0 |
| S8 capacity (S4 dirs) | `s8_s4/` (139452) | @64 k0-k16 PERFECT (170/170); **c*(128) = 16** — wave-2 S4 superseded |
| S8 gate + model ladder | `s8_gate/`, `s8_mg/eval_{short,long}/` (139471/72) | gate ≥0.997 flat; deployed k≤8 0.99/1.00/0.92/0.96/0.94 = oracle − named gate errors |
| S9 N-free train | `s9_nfree/20260831_174021_gated/` (139318) | 1.000/0.973/0.793, NO snap signature, no N in text |
| S9 oracle ladder | `s9_ladder_64_128/` (139461) | **H-S9-main MET: k≤8 = 1.000 at every N** (new prompt anchor) |
| S9 capacity (S4 dirs) | `s9_s4/` (139462) | k16 20/20 @64; k12 dip 11/20 (candidate step-id pixel channel); c*≈16 |
| S9 gate + model ladder | `s9_gate/`, `s9_mg/eval_{short,long}/` (139486/87) | deployed k≤8 1.000 @64 / 0.958 @128 |
| S9 S5-HF | `s9_s5_hf/` (139488) | 0.86/0.78/0.58 — **H-S9-HF missed** (gate recall on HF); ≥ wave-2 S5 @16/32 |

Adapters: `checkpoints/sft_fenced_gated_vn_adapter` (**CANONICAL**, gate required),
`sft_fenced_gated_nfree_adapter` (gate + `--nfree-prompt` required; new prompt anchor).
Figure: `fig/F5_flatline_w3.png`.

## Wave 3 extension (2026-09-01) — S9b coverage fix · S10 photograph · S11 text-controlled α

| experiment | canonical run | headline |
|---|---|---|
| S9b ep5 arm | `s9b_ep5/20260901_142255_gated/` (139718) + `s9b_ladder/`, `s9b_s4/` (139774/75) | k12 dent CLOSED (0.9-1.0 every N); +1 wobble at k6-7 → shown to be checkpoint-draw noise |
| **S9b ep10 twin (CANONICAL N-free)** | `s9b_vn_nfree/20260901_140231_gated/` (139714) + `s9bT_ladder/`, `s9bT_s4/` (139977/78) | **1.000/1.000/0.847/0.733/0.660 — k0-k16 PERFECT at EVERY N incl. k16@128 20/20 (S8: 5/20); all loss = untrained k≥24**; ep5-vs-ep10 = budget ablation |
| S10 attention photograph | `s10/{p1b_N*,gated_N*,frozen_N32}/` (139736,139760-62) | **share law OBSERVED** (R² .93-.99, 9/9 sign); gated mass N-invariant ≤6%; **L24h20 internal evidence head AUC ≥0.993, trained-in** |
| S11 text-controlled α | `s11/gate_oracle_N*/`, `_N64_k*/` (139735,139752/53) | α +0.035 [CI ∋ 0] with byte-identical text; margins +11.7 const; γ 1.19 = P1g's; **no non-text N-channel** |

Adapters: `checkpoints/sft_fenced_gated_vn_nfree_adapter` (S9b ep10 twin; gate +
`--nfree-prompt` required). Figures: `fig/F6_s9_alpha.png`, `fig/F7_attn_photo.png`.

## S10b (2026-09-19) — photograph fill-in + temperature sweep

| experiment | canonical run | headline |
|---|---|---|
| S10b frozen ladder | `s10/frozen_N{8,16,64,128}/` (153242/153243) | frozen per-frame mass falls as N^-0.94 with evid:nonevid ratio ≈1.0 at every N — no evidence preference |
| S10b P1b @128 | `s10/p1b_N128/` (153244) | P1b ratio 1.04 at 128; per-frame slope 0.83 across 8..128; L24h20 ratio 3.2 (needs ~32) |
| S10b figure | `fig/F10_dispersion_hahn.{png,csv}` (`scripts/sparse/fig_s10b.py`) | weights (a) beside flip response (b) beside selector head (c) |
| S10b tau sweep | `s10/p1b_N{32,128}_tau{1.5,2,3,4}/` (153246, anchor 0.00e+00) | sharpening frees competitor mass (0.74→0.29 @32) but sends it to the SINK; evidence mass falls ×4.5 @32, ×3.4 @128; max/mean over evidence 1.03→1.36; S0 exact 0.85→0.19 — the two-requirement coupling photographed; `fig/F11_temperature_coupling` |
| S10b small-N photo | `s10/{frozen,p1b,gated}_N{2,4}/` (153279) | 1/N trend + constant gap hold down to N=2; gated evid/frame .027–.029 |
| S10b small-N flip fill | `armor/hahn/short_N2/`, `loramech/n2_hahn/p1fence_ep10_N{2,4}/`, `sparse/s11/gate_oracle_N2/` (153280) | at fixed count 0→1 the gated read is FLAT 74→77 from N=2 to 128; frozen 34.7→1.2; trained 98→15.5; pooled small-N decline = k-composition |
| FIXEDK (2026-09-21) | `outputs/fixedk/{frozen,trained,gated}_N{2..128}_g{0,1,2}/` (153850/51 N≤8; 153869–73) | fixed-count flips, 30 pairs/cell: gated read FLAT (|α|≤0.03) N=2..128; fact flat; frozen 0.76–0.88; trained 0.44–0.62 (×2–12 amplitude); `outputs/fixedk/fig/fixedk_medians.csv` |

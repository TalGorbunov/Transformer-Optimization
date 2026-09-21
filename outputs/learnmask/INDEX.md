# outputs/learnmask — INDEX (LEARNMASK campaign, 2026-08-12 → 08-15; brief: CAMPAIGN_BRIEF.md, log: STATE.md)

Learned discrete attention mask from the fence init on the frozen 4-bit 7B: (relation × layer) gate logits
(S1 56 / S2 392 / S3 616) or a per-cell free table (S0); estimators E1 ste / E2 soft-anneal / E3 ST-Gumbel;
task = MMReD-HF `steps_in_room` @512 (native), replica scaffold (zero trained components) from wave 1 on.
Scripts frozen at `legacy/v1/scripts/learnmask/` (README there); figures `figures/acc_vs_n.{png,pdf}`,
`figures/survivor_heatmap.{png,pdf}` (`plot_results.py`). Machinery pinned: engine parity 0.00e+00, mask parity bit-for-bit.

| experiment | canonical run | headline |
|---|---|---|
| P1 smoke, S1 on the carrier scaffold (P7a digit ckpt) | `p1_smoke/20260812_174838_s1_st-gumbel_class/` (131337, TIMEOUT ep3/6; the three earlier `p1_smoke/` dirs = parity-gate failures 131330/32/34) | handfence 0.940 = fence-init 0.940 @seq8 → 0/56 flips (no CE pressure toward aggregation at N=8); 14.6 s/step |
| Wave-1 S2 estimator sweep, replica scaffold (ABORTED by the abort rule) | `sweep_s2_{ste,soft,st-gumbel}/` (131378–80; ep0 rows in `sweep_s2_soft/`), `sweep_s0_{ste,soft}/` (131381/82 guard-fail → 131384/85) | ep0 rows at FLOOR: handfence 0.267 / init 0.300 / nofence 0.133, em 0.000 — the frozen model never emits a digit; all six jobs cancelled |
| Readout-position probes (canonical tail; supply) | `probe_readout/canonical/` (131393), `probe_readout/supply/` (131402); `hint/`, `prime/` (131388/89) cancelled unused | digits emitted but pure majority ("0"): CE init 2.17 < nofence 2.53 < handfence 2.93 < handsupply 3.67 < supply 4.02 — frozen 7B cannot count under ANY topology |
| Existence test, penalty OFF (unconstrained search) | `exist_s2/20260812_221106_replica_s2_st-gumbel_class/` (131403, TIMEOUT 3/4 ep), `exist_s0/20260812_221051_replica_s0_st-gumbel_class/` (131404) | S0 oracle: 0/6,698,496 flips, eval 0.600 = majority; S2: 1/392 flips, calibration shuffle only — MEASURED negative |
| S0 20-ep, lr 5e-2 (definitive frozen-scaffold negative) | `exist_s0_long/20260813_181916_replica_s0_st-gumbel_class/` (132431) | 947,575/6.7M cells flipped, train CE 1.58→0.32 (position memorization), HARD eval 0.600→0.575 (best 0.650 @ep14 = noise), eval CE 1.31→1.94 |
| S0 layer-shared (professor's /28 variant) | `exist_s0_shared/20260815_205741_replica_s0_st-gumbel_class/` (132928; 132926 cancelled at ep0, first subdir) | per report.txt (not in STATE.md): 36,997/239,232 flips, train CE →0.86, HARD eval 0.600 (best 0.625 @ep12 = nofence row) — same negative |
| Fence-agnostic LoRA readout, s2open-pinned | v1 `readout_s2open/` (132465, undertrained 0.680); **v2 `readout_s2open_v2/20260813_194351_replica_lorareadout_s2open_class/` (132498, CANONICAL)**; v3 multi-task `readout_s2open_v3/` (132500) | v2: first emitted counting above majority (0.550 vs 0.283 on the 8+16 mix); v3 dilutes steps_in_room (0.64 vs 0.82 @8) |
| Readout validation × topology (50/split) | `readout_v2_val/20260813_210108/` (132501), `readout_v3_val/20260814_113853/` (132549) | v2 em @8/16/32: s2open 0.82/0.58/0.30 · hand 0.78/0.54/0.26 · nofence 0.62/0.42/0.22 · init 0.60/0.42/0.16 — fence family wins increasingly with N |
| Estimator sweep from the closed fence init (v2 readout frozen) | `sweep2_s2_{ste,soft,st-gumbel}/` (132503/04/05; soft TIMEOUT ep2, a100 1.5× slower) | 0/392 flips in all three; p_open R4/R5/R7 0.12 → 0.02–0.10 — COORDINATION BARRIER (no single edge pays for itself) |
| **Prune-from-open (the campaign's payoff)** | `prune_s2/20260814_121039_replica_s2_st-gumbel_class/` (132567), `gates_best.pt` @ep2, `heatmap_ep4.csv` | 139/392 edges pruned at ZERO cost (0.550 = full-open row); survivors: R4+R5 aggregation kept L0–17, pruned L18–27; R6 28/28, R7 26/28 — cutoff at the READ_LAYER 16 locus |
| Pruned-mask length transfer (money figure) | `pruned_transfer/20260815_013151/` (132671; `results.csv`) | learned 0.820/0.580/0.260/0.040 @8/16/32/64 == full-open at 8/16, == hand fence at 32/64; nofence MAE 10.4 @64 |

Status: CLOSED 2026-08-15 with four deliverables (frozen-scaffold oracle negative; coordination barrier; necessity structure via prune-from-open; fence-family length ordering). No RESULTS.md entry was ever logged. Superseded by the gated fenced read (SPARSE, 2026-08-30 →); listed RETIRED in the 2026-09-21 rewrite (paper plan 09-14: "what does not work" paragraph); code frozen under `legacy/v1/`.

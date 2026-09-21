# outputs/superquery — INDEX (SUPERQUERY / repeater campaign, 2026-08-05 → 08-13, retired; brief: none (STATE.md header), log: STATE.md)

Divide-and-conquer readout on the frozen 4-bit 7B: fenced [frame + question-replica] blocks (posreset) + b-ary
"superquery" nodes that attend only their children; per-node linear probes quantize counts into digit tokens
("repeater"). Park steps task, N=8 unless noted; every cell frozen, no LoRA. STATE's logged entries span
08-05→06 (the Tal-authorized night run). Scripts frozen at `legacy/v1/scripts/superquery/`; every run dir
carries its `runner-<jobid>.log`; fitted tables in `fits_all/`, per-run CSVs named in the table.

| experiment | canonical run | headline |
|---|---|---|
| Probe 1+2 flat + trees, first N=8 captures | `20260805_122449_tree_128758/` (log only), `20260805_142644_tree_128773/` (feats_N8) | flat L20 count 0.24 vs b=2 level-1 (fan-2) 0.94 — tree level-1 reads work frozen |
| Full frozen sweep N=8/16/32/64 + fits | `capture_16_64_128790/` (feats_N{8,16,32,64}.npz), `fits_all/{nodes,top}.csv` (128806); `fits_n8/` (128787/95) = N=8-only precursor | capacity set by READ FAN-IN, not length: fan-2 0.93–0.97, fan-4 0.62–0.71, fan-8 0.24–0.46, flat 0.14–0.24; hop fidelity ~0.4–0.5, levels ≥3 chance; approximate (MAE) flat is best |
| Ridge→round reanalysis (Tal's challenge) | `fits_all/reanalysis_rr.csv` (128845), `deconfound/deconfound.csv` (128846) | corrected c(fan) @L20: fan-2 0.98 / 4 0.87 / 8 0.67 / 16 0.44 / 32 0.21 / 64 0.12 — "counts to ~4, halves per doubling"; LR probes understated exactness (P2 two-hop 0.29→0.97) |
| PATCH — hop mechanism | `patch_n8/patch.csv` (128816; 128815 precursor log) | lvl2 raw 0.43 / denoised 0.47 / DIGIT-TOKEN embeddings 1.000 / text ceiling 0.98 → hops need vocab RE-QUANTIZATION |
| Repeater tree v1/v2/v2b (probe-quantized, one forward) | `repeater_n8/` (128824), `repeater2_n8/` (128829), `repeater2b_n8/repeater2.csv` (128831, FINAL) | v2b: Q1 0.983/node → sum 0.938 lossless ×2 → ROOT 0.897 exact (±1 0.992, MAE 0.12) vs flat 0.239; hard > soft quantization |
| Self-quantization + answer lens (task-agnostic quantizer) | `selfq_n8/` (128844), `selfq_v3_n8/` (128848; `selfq_v2_n8/` 128847 void), `lens2_n8/lens.csv` (128850; `lens_n8/` 128849 void teacher) | raw logit lens pairs 0.535/0.519, top-1 never a digit → parameter-free route CLOSED; distilled lens pair-acc 0.786 > teacher 0.71, frame transfer 0.964 — ceilinged by native competence |
| Native flat text counting of clean symbols | `textcount2/textcount.csv` (128855; `textcount/` 128853 void) | 0.892/0.550/0.325/0.317/0.233/0.183 @N=4/8/16/32/64/128 — pasting codes + one read fails at high N; staged fan-in is NECESSARY |
| Emission mechanism (v4b/v4c) + pass-2 template law | `repeater4b_n8/` (128854), `repeater4c_n8/` (128856), `addsanity/addsanity.csv` (128857); `repeater4_n8/` (128852) cancelled mid-calibration | emit-EM ~0.13 even with GOLD mid-layer codes: emission copies LAYER-0 token identity → two-pass; "Two partial counts are {a} and {b}. What is the total count?" + "Answer: " = 1.000 |
| **★ v4d two-pass (HEADLINE)** | `repeater4d_n8/repeater4d.csv` (128858) | MODEL-EMITTED exact count **0.980 @N=8** (MAE 0.02, cond-EM 1.000, n=300): pass 1 = fenced b=2 tree, Q1@L20 0.994 / Q2@L24 0.990; pass 2 = ~50 text tokens |
| Emission deadline curve v1 / v2 (same-sequence control) | `deadline_n8/deadline.csv` (128878), `deadline2_n8/deadline2.csv` (128883) | v1 depth-matched codes 1.000@L4 → 0.24@L20; v2: mouth open through L16 (1.00), true deadline ~L20–24 — v1 was harvest-context mismatch |
| Compressed single-forward schedule | `compressed_n8/compressed.csv` (128885) | Q1@16→Q2@17→reg@18: EM 0.635 (relay is 1-layer, 0.949); capped by maturity-vs-acceptance collision — thesis rows 0.635 single-forward vs 0.980 two-pass |
| Layer looping (window stretch) | `layerloop_n8/layerloop.csv` (128888; 128887 precursor) | L16×2/×3/[13–16]×2: chain 0.69→0.85 but e2e saturates 0.673 — the deadline follows PROCESSING COUNT, not layer index |
| N=8 feature re-capture, L20/24/27, no fits | `20260806_123222_tree_128914/` (feats_N8.npz) | not referenced in STATE.md |

Status: RETIRED. Laws carried forward: c(fan) length-invariant; hops need vocab re-quantization; emission = layer-0 identity → two-pass. Ported to the official benchmark by NINV (`outputs/ninv/`, RESULTS [2026-08-10b]: two-pass EMIT-EM 0.960 @512 on HF seq8; ninv INDEX cites 128773+128790 as its pre-fix baseline); the in-model executor route was closed by TREEFOLD (RESULTS [2026-08-24→25]); superquery/ninv/treefold all retired in the 2026-09-21 rewrite for fence+gate+read. No RESULTS.md entry of its own.

# FIXEDK — STATE (append-only, newest last)

## [2026-09-20] Campaign created (Claude, authorized by Tal) — chains to submit after the SOFTGATE anchor
- [21:09] submitted FIXEDK frozen (153850) and trained (153851) chains on a100-public, 12h_4g, --time 6h; gated chain waits for the SOFTGATE probe anchor (patched file).
- [21:32] SUBMITTED gated chain 153864 (a100-public, 4d_1g) after the probe anchor passed.
- [21:34] REVIEW PANEL (blocking item 6): longN_park N>=16 pools hold exactly 30 dirs/gold and seq_len_16 is a TRAINING root of P1b and S9/S9b. CANCELLED 153850/153851/153864 at their N=16 cells; kept the completed N<=8 cells (park pools, 100/class). Wrapper now uses data/mmred_redux/seq_len_N (fresh, 50/class at k=0,1,2) for N>=16, GLIST 0-1-2, S9 adapter for the gated arm (= the S11 endpoint; item 4). RESUBMITTED: frozen N16-128 153869, trained N16-128 153870, gated N2-128 153871, gold-2 top-ups N4/8 frozen 153872 trained 153873 (a100-public, 4d_1g).

## [2026-09-21] FIXEDK LANDS (153869–153873 + the N≤8 cells of 153850/153851): 60 cells × 30 pairs, matched samples — the gated read is FLAT at fixed count from N=2 to 128; frozen and trained reads decay from N=2; fenced fact flat

Medians of ‖Δh‖ at L20 (final row for reads, rep_t for the fact), 30 pairs per cell, N = 2/4/8/16/32/64/128:
| flip | frozen read | fenced fact | trained read (P1b) | gated read (S9) |
|---|---|---|---|---|
| 0→1 | 34.4/28.6/17.2/4.2/2.6/2.0/1.3 (slope 0.88) | 39.6…35.8 (0.04) | 97.0/67.7/52.2/40.4/30.2/23.5/14.0 (0.44) | 74.2/74.6/75.0/75.4/74.8/74.7/74.6 (−0.00) |
| 1→2 | 23.9/21.1/11.6/5.0/2.7/2.0/1.4 (0.76) | 39.5…35.9 (0.02) | 145.5/40.9/26.2/18.8/13.7/10.5/8.1 (0.62) | 41.0/42.5/43.1/44.5/45.1/45.1/46.0 (−0.03) |
| 2→3 | —/17.2/11.5/5.6/2.8/1.6/1.3 (0.80) | —/41.4…35.7 (0.05) | —/51.2/22.4/14.9/10.4/7.6/6.2 (0.58) | —/38.0/38.0/37.5/37.8/37.2/37.0 (0.01) |
Bands: **H-F1 MET** (gated |α| ≤ 0.03 at every stratum, N=2..128). **H-F3 MET** (fact |α| ≤ 0.05).
**H-F2: frozen MET** (0.76–0.88); **trained: amplitude clause MET** (×2.4–3 over frozen at N≤8, ×10–12
at N≥16 where the frozen response sits at the floor) but the **α ≥ 0.5 clause MISSED at the 0→1
stratum (0.44)** and met at 1→2 / 2→3 (0.62 / 0.58). Reading: at fixed count the trained read's
effective exponent (0.44–0.62) is LOWER than the pooled 0.80 [0.66,0.97] and lower than the frozen
read's (0.76–0.88) — consistent with the share law when training raises the edge e^s slightly (see
the fit below), NOT with a removed N-dependence: the trained response still falls ×7 from 2 to 128
where the gated read moves by <3%. Per-gold protocol ≠ the paper's pooled α protocol; both reported.
Pools: park (100/class) N≤8, mmred_redux (fresh) N≥16; adapters P1b (trained), S9 nfree (gated).
Figure input: outputs/fixedk/fig/fixedk_medians.csv → F10 panel b′ (redrawn from these cells).

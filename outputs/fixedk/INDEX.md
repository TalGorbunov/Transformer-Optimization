# outputs/fixedk — INDEX (FIXEDK campaign, 2026-09-20/21; brief: CAMPAIGN_BRIEF.md, log: STATE.md)

Fixed-count flip responses: ‖Δh‖ at layer 20 when one frame flips the count g → g+1, N = 2..128,
30 matched pairs per cell. Pools: park (100/class) for N ≤ 8, mmred_redux (fresh) for N ≥ 16.

| Experiment | Canonical run(s) | Headline |
|---|---|---|
| frozen read | `frozen_N{2..128}_g{0,1,2}/` (153850 N≤8; 153869 N≥16) | 34→1.3 (0→1), slopes 0.76–0.88 |
| fenced per-frame fact | same cells, locus rep_t | flat 36–41, |slope| ≤ 0.05 |
| trained read (P1b LoRA) | `trained_N{2..128}_g{0,1,2}/` (153851 N≤8; 153870 N≥16; top-ups 153872/73) | 97→14 (0→1), slopes 0.44–0.62; ×2–12 the frozen amplitude |
| gated read (S9 nfree adapter, oracle gate) | `gated_N{2..128}_g{0,1,2}/` (153871) | 74–75 flat at 0→1, 41–46 at 1→2, 37–38 at 2→3; |slope| ≤ 0.03 |
| medians table | `fig/fixedk_medians.csv` | feeds F10 panel b′ (`outputs/sparse/fig/F10_dispersion_hahn.png`) |

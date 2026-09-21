# FIXEDK — flip response at FIXED base count, N = 2…128, all four conditions (2026-09-20)

**Status: AUTHORIZED (Tal, 2026-09-20).** Why: the pooled ARMOR/N2/S11 protocol draws pairs over gold
≤ 8, but N=2/4 pools only contain gold 0–1, so pooled medians confound k-composition with N (found
2026-09-19: the gated read looked like it fell 72→28 over N=2..8 and was flat at fixed count). The
2026-09-19 fixed-count series rests on 5–14 pairs per cell at small N. This campaign re-measures the
four conditions at fixed base count with ≥30 pairs per cell.

## Cells (all eval-only; existing scripts, `--gold-set g --limit 30 --controls 12 --seed 0`)
| condition | script / arms / adapter | N | golds |
|---|---|---|---|
| frozen read + fenced fact | `scripts/armor/probe_hahn.py --arms plain,fenced` | 2,4,8,16,32,64,128 | 0, 1 (and 2 where N≥4) |
| trained read | `probe_hahn.py --arms p1fence --peft-adapter checkpoints/sft_fenced_le16_ep10_adapter --max-gold 8` | same | same |
| gated read | `scripts/sparse/probe_hahn_gated.py --arms p1fence --gate oracle --nfree-prompt --peft-adapter checkpoints/sft_fenced_gated_vn_nfree_adapter` | same | same |
Pools: `data/mmred_images_park/seq_len_{2,4,8}/all_uniform`, `data/mmred_longN_park/seq_len_{16,32,64,128}/all_uniform`.
Outputs: `outputs/fixedk/{frozen,trained,gated}_N{N}_g{g}/`.

## Pre-registered bands
- **H-F1:** gated read at gold 0→1 flat: log-log slope |α| < 0.1 over N=2..128 with CI ∋ 0; same at gold 1→2.
- **H-F2:** frozen and trained reads at fixed gold decay with α ≥ 0.5 over N=2..128 (CI excludes 0.2);
  the trained read's amplitude exceeds the frozen read's at every N (×2–6).
- **H-F3:** fenced fact at fixed gold flat (|α| < 0.1).
- Descriptive: the per-gold exponents vs the pooled 0.72/0.80 — report the spread; do not re-fit the
  paper's headline α from these cells (different protocol); F10 panel b′ is redrawn from them.

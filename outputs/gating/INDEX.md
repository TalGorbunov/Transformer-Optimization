# outputs/gating — gated attention (arXiv:2505.06708) vs the aggregation bottleneck

Campaign brief: [CAMPAIGN_BRIEF.md](CAMPAIGN_BRIEF.md) · live log: [STATE.md](STATE.md)
(the STATE file is the handoff; this index lists only canonical runs and headlines).

Question: does gating relieve the aggregation (over-squashing) bottleneck, and if so at
which position — `G1` (after the attention sum, query-side) or `G2` (inside the sum, on
the message/value path)? Gating is multiplicative masking in (0,1): it can attenuate,
never amplify, so a gain is evidence for **interference**, not **capacity**.

| experiment | canonical run | headline |
|---|---|---|
| P0 · 1B G1 triple (baseline / headwise / elementwise), text MMReD tally probe | `p0_text_triple/20260807_212129_129124` | G1-headwise raises ridge R² at 48/64 layer-matched cells (+0.119 at N=16 → +0.059 at N=40); **the gain does not grow with N**; sink killed (F-Attn 0.74 → 0.006, M-Act 1.4e4 → 621) |
| P1 · sink diagnostic, frozen Qwen2.5-VL-7B, plain vs deployed | `p1_sink_7b/20260807_214956_129133` | **gate NOT met** — our 7B has no token-0 sink (F-Attn ≤0.05 vs 0.74 for the 1B); massive activations WITHOUT a sink |
| P3 · main ablation, 5 arms (caption scratchpad) | `p3_arms/`, `p3_arms_v2/` | anchor reproduced (0.999/0.996) — but the readout turned out copyable, see below |
| ⚠ metric audit · tally-copy probe | `p35_tallycopy/` | the caption `tf_acc` is largely a COPY DETECTOR: shift the tally +3 and the model follows it 85% of the time at N=32, counts 0% |
| P7 · **digit readout, no scratchpad** (the campaign's real result) | `p7_digit/`, `p7_digit_extrap14/` | LoRA control 0.955±0.012 vs best gate+LoRA 0.941±0.010 (3 seeds); all 5 gate positions below plain LoRA |
| P8 · accuracy per sequence length | `p8_digit_seqlen/`, `p8_digit_seqlen_extrap/` | the wall: 1.000 → 0.123 as N goes 8 → 128; no gate position beats LoRA above N=8 |
| P0 addendum · what the 1B ckpts EMIT | `p0_emitted/`, `p0_emitted_short_full/` | at/near chance from N=8 up, all three checkpoints; probe says the count IS decodable (R² 0.58–0.70) |

Code: `scripts/gating/` (`probe_text_triple.py`, `probe_sink_7b.py`, `eval_gated.py`),
module `gnnformer/gating.py`, tests `tests/test_gating.py`, wrappers
`slurm/gating_*.sbatch`, root lists `slurm/lib/roots_text_tally.txt`,
`roots_gating_grid.txt`, `roots_gating_capacity.txt`.

Sizing smokes (throwaway, not canonical) live in `outputs/_scratch/gating_smoke/`.

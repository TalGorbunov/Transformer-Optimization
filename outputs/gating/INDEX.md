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
| P1 · sink diagnostic, frozen Qwen2.5-VL-7B, plain vs deployed | `p1_sink_7b/` | _running_ |
| P3 · main ablation, 5 arms | `p3_arms/` | _sizing smokes running_ |
| P3.5 · capacity-vs-interference discriminator | `p35_discriminator/` | _pending_ |

Code: `scripts/gating/` (`probe_text_triple.py`, `probe_sink_7b.py`, `eval_gated.py`),
module `gnnformer/gating.py`, tests `tests/test_gating.py`, wrappers
`slurm/gating_*.sbatch`, root lists `slurm/lib/roots_text_tally.txt`,
`roots_gating_grid.txt`, `roots_gating_capacity.txt`.

Sizing smokes (throwaway, not canonical) live in `outputs/_scratch/gating_smoke/`.

# outputs/presentation — cache-only figures for the peer-presentation narrative

All CPU jobs over frozen caches (`checkpoints/` symlinks + `outputs_legacy/`); no new
model forwards. Each run dir has `ABOUT.md` (plain-language summary + provenance),
the figure (png+pdf), and a CSV of every plotted number. NOTE: figure d′ values are
the held-out logistic-axis estimator (conservative), NOT the probe's d′_w that
RESULTS.md headlines — same ordering, different scale (stated in each ABOUT.md).

| experiment | canonical run | headline |
|---|---|---|
| pca — per-frame messages @L16 N=8: PCA scatter + discriminant-axis histograms, joint vs fenced vs carrier | `pca/20260731_202940` (job 127490) | joint d′ 1.9 (classes overlap) vs fenced 5.6 / carrier 5.1 (fully separated) |
| curves — SAME gate→tally sum-readout vs N, three graphs | `curves/20260731_202940` (job 127491) | joint 0.468→0.077 @N=8→128; fenced/carrier 0.98–1.00 flat to N=128 (fenced N=8 = 0.998, matches the logged anchor) |
| saturation — per-layer gate on the deployed-stack depth dump (N=32) | `saturation/20260731_202507` (job 127489) | gate err flat ~0.33 through L12, drops in the write window (L12–19), saturates 0.008 @L20–24 (anchor [2026-07-25]) |

| attnmap — head-averaged attention @L16, joint vs fenced, segment-averaged (n=8) | `attnmap/20260731_211330` (job 127555) | joint = dense lower triangle (cross-frame paths); fenced = block-diagonal islands + Q column. NOTE probe layout: replicas hidden from the final question (leak-proof probe mask), so `fin` reads frames, not carriers |
| sensitivity — Jacobian ‖∂(read-out signal for f)/∂(frame f′ emb)‖ share, L≤16 (fit n=64, grad n=12) | `sensitivity/20260731_211330` (job 127556) | joint: own-frame share 0.480 — **52% of sensitivity flows through OTHER frames** (uniform=0.125); fenced: 1.000, cross-frame **exactly 0.00**. Caveat: joint direction fit d′ 0.60 in THIS (Q-first replica, unmasked) layout — weaker locus than the historical plain-joint anchor (~2.0) |

| anchor_sweep — where to read the replica message, 5 variants @L16, teacher config (n=150) | `anchor_sweep/20260731_213558` (job 127560) | **room WINS**: d′ 8.93 / tally 0.997 vs mean 7.00/0.963, last 4.80/0.837, char 4.39/0.813, first 3.31/0.432 — the distillation anchor is now an ablation winner, not a guess; also reproduces the A3 teacher band (~9 @n=150) |

| attnmap_deployed — DEPLOYED stack (e_c + LoRA, L\*=12), attention @L8 vs @L20 (n=8) | `attnmap_deployed/20260731_215003` (job 127571) | the two-level hierarchy photographed: L8 = islands (f_i↔c_i + Q; carriers hidden from the tail); L20 = coarse graph (carriers attend earlier carriers, `fin` reads carriers) |

| qkv_swap — q/kv activation-patching 2×2 @L16, mask-only fence vs unmasked, shared positions, own-frame softmax keys (n=150) | `qkv_swap/20260731_220735` (job 127582) | **the tax splits ~50/50**: total 2.90 d′; query half 1.93, value half 1.98 (sub-additive → ~1.0 d′ overlap); corners externally consistent (dirty-dirty 3.74 ≈ unmasked band 3.56; clean-clean 6.64 mask-only band). Direct paired measurement of the [2026-07-14] inferred dissociation |

| qkv_swap N-scaling — same 2×2 at N=16/32/64 (longN_park; n=150/100/50, jobs 127586/7/8) | N=32 → `qkv_swap/20260731_221639` · N=64 → `qkv_swap/20260731_222127` · N=16 → `qkv_swap/20260731_222936_127589` (deterministic rerun; 127586's artifacts were clobbered by a same-second run-dir collision with 127587 — wrappers now suffix `_${SLURM_JOB_ID}`) | **kv-alone repair collapses with N**: q-alone +0.92/+1.01/+1.24/+1.05 vs kv-alone +0.97/+0.64/+0.09/**−0.47** @N=8/16/32/64; interaction ~2 at long N; joint-corner d′ flat ~3.7–4.1 while its tally decays 0.78→0.55 |

| qkv_scaling — repair-gains-vs-N figures (CPU, parses the 4 qkv_swap reports) | `qkv_scaling/20260731_224350_local` (d′ gains + ACCURACY version `qkv_scaling_acc`; supersedes `_223504_local`) | d′: value-only +0.97→−0.47 @N=8→64, query-only ~+1 flat, both ~+3; acc: values-only DROPS BELOW joint at N≥32 (0.66 vs 0.72; 0.46 vs 0.55), fence 0.94–0.99; logged as RESULTS [2026-07-31b] |

| waterfall — joint failure → method, one named repair per bar @N=32 (stitched from curves + qkv cells; protocol caveat in ABOUT) | `waterfall/20260731_230425_local` | 0.15 (joint readout) → 0.72 (+reader per frame) → 0.84 (+clean readers) → 0.94 (+clean frames) → 1.00 (+posreset & Q-first); carrier matches 1.00 — THE summary slide |

| fenced_supply — A3 probe (fence+posreset+Q-first) at N=16 (n=150) / N=64 (n=50), longN_park | `fenced_supply/N16/…` (job 127597) · `fenced_supply/N64/…` (job 127598) | replica d′ 10.67 / 10.23 @L16 (joint anchors 3.91 / 2.64) — the fence band holds at length; rung-5 caches for the waterfall grid |
| qkv_swap N=128 (n=25 — NOISY, ~13 eval samples/seed) | `qkv_swap/…_127599` (job 127599) | d′ ordering sane (CC 5.14 > CD 3.89 > DD 3.07 > DC 2.57) but tally cells non-monotone (CD 0.71 > CC 0.52) — quantization noise at n=25; rerun bigger before citing |
| waterfall_grid — the 5-rung staircase at every N=8..128 | `waterfall_grid/20260731_234426_local` | staircase monotone and complete N=8–64; N=128 panel noisy (see above), fence rung 0.98 solid at every N — THE summary slide, length edition |

| supercarrier — ONE frozen reader over the N coarse nodes, coarse-keys vs full-causal arms, L16/L20 (job 127738) | `supercarrier/…_127738` | **NO-GO**: tally 0.25 @N=8 → 0.07 @N=64, ferr ~0.21–0.29, d′ 1.3–2.1 in BOTH arms — restricting the softmax to carrier keys does NOT rescue a single reader; one frozen query can't address N items even over the coarse graph. Kills the "one aggregator token" shortcut zero-shot; per-frame readers (0.72–0.85) or serial readout remain the only working reads |

| posreset_sweep — 10 fresh A3 cells, N=4..64 × {reset, noreset} (jobs 127736/127737) | `posreset_sweep/N{4,8,16,32,64}_{reset,noreset}/…` | gate-acc dose-response caches; reset flat 0.996–1.000 to N=32, 0.95 @64; no-reset drifts 0.98→0.83 |
| posreset_dose — the position-tax figure, accuracy edition | `posreset_dose/20260801_164109_local` | widening gap with N (0.013 @4 → 0.12 @64) at the SUPPLY level; the deployed decode amplifies it to collapse (0.987→0.313 @N=32, [2026-07-27]) — supply tax modest, trained readout multiplies it |

| attnmap_plain — VANILLA forward (frames + fin only, no carriers/fence/reset) @L16, N=8 vs 32 (n=8 each, job 127814) | `attnmap_plain/20260801_193854_127814` | the BEFORE picture, both failures quantified: **query dilution** — fin gives frames a fixed ~0.31–0.33 total budget at any N → per-frame share ~1/N (0.034 → 0.010), with a primacy+recency U at N=32 (middle frames starved to ~0.006); **value pollution** — cross-frame mass rises with index and saturates ~0.56/0.64: from frame ~5 on, frames spend MORE attention on other frames than on themselves (own 0.30/0.20 mean @N=8/32) |

Superseded: `pca/20260731_202507`, `curves/20260731_202507` (v1 — label collision;
no discriminant row). Failed (mask-dtype bugs, no artifacts): `attnmap/20260731_211027`,
`sensitivity/20260731_211027` (jobs 127553/127554), `attnmap_deployed/20260731_214717`
and `_214836` (jobs 127565/127566 — fixed in v3 by engine-style fp32 masks +
EFFICIENT→MATH sdpa backends). Produced by `scripts/presentation_figs.py`,
`scripts/probe_attention_map.py`, `scripts/probe_sensitivity.py` via the matching
`slurm/*.sbatch` wrappers (figures QOS 4h_0g CPU; probes QOS 2h_2g 1 GPU).

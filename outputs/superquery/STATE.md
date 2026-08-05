# superquery — divide-and-conquer tree readout (MAIN DIRECTION, 2026-08-05)

Professor's proposal: fenced [frame+q-replica] blocks (posreset, fence NEVER lifts) +
"superquery" question replicas after the blocks that attend only their children's spans
at all layers. Flat version = star graph fan-in N (predicted to fail counting: supercarrier
probe Aug-1, coarse≈full, tally 0.25@8→0.07@64). Our extension: bounded fan-in TREE
(b-ary superquery hierarchy) — over-squashing theory prescribes bounded-degree readout;
in-model tally works at fan-in ~8 (0.87). Sweep measures the capacity curve c(b).

## Probe 1+2 (flat + trees in one script) — scripts/superquery/probe_tree.py
- One forward/sample carries ALL arms (flat, b=2,4,8): blocks shared, per-SQ-row masks.
- Park steps task, N=8/16/32/64 (120/120/80/40), read layers 12/16/20, feats mean|last.
- Failure-signal ladder: per-level subtree-count probes (acc/±1/R²/gate-d′/tally),
  hop survival implicit in per-level rows, top-node gold regression + composed tally.
- Job 128758 (l40s-public, 2h_2g). Output: outputs/superquery/<ts>_tree_128758/
  {nodes.csv, top.csv}.

## Interpretation guide
- level-1 count acc vs fan_in = the aggregation-capacity curve (the headline).
- level-l acc >> flat at same N = divide-and-conquer works frozen.
- If level-1 good but level-2 drops: hop-survival failure (info dies in transit).
- If all levels fail even b=2: replica loci don't expose verdicts to SQ reads
  (routing/read failure, not aggregation) — check gate d′ first.

Related in-flight: v7 arm-C train (128757, L16 full-fence + v5 counter scans).

## FULL FROZEN SWEEP COMPLETE (capture 128790 + fits 128806, 2026-08-05)
CSVs: fits_all/{nodes,top}.csv; features: capture_16_64_128790/feats_N{8,16,32,64}.npz
(read layers 12/16/20/24/27; N8 npz has 12/16/20 only).

### Headline: aggregation capacity is set by READ FAN-IN, not context length
Level-1 exact subtree-count acc (best of L20-27, mean|last):
  fan-2 : 0.949 / 0.974 / 0.968 / 0.927   @ N=8/16/32/64
  fan-4 : 0.619 / 0.643 / 0.668 / 0.706
  fan-8 : (0.239) / 0.344 / 0.385 / 0.460   (fan-8@N16-64 = b8 lvl1)
  fan-N : 0.239 / 0.206 / 0.142 / 0.150     (flat)
One frozen softmax read counts to ~2 (0.95) regardless of 8 vs 64 blocks in context.
(Caveat: probe-row volume differs per cell — flat/deep cells have fewest rows.)

### Hop fidelity ~0.4-0.5, length-stable; levels >=3 -> chance
b2 lvl2: 0.414/0.435/0.464/0.521; depth-maturation RULED OUT (L24/27 flat).
MLP probe check: does NOT beat linear on hop cells (0.25 vs 0.39 lvl2) — hop loss is
genuine accessibility loss, not linear-probe blindness.

### Best exact end-to-end frozen: pairs + external sum
b2 lvl1 composed tally: 0.806/0.822/0.725/0.383 vs flat 0.239/0.206/0.142/0.150.

### Approximate counting: flat is actually best (analog average survives)
flat TOP: MAE 0.43/0.79/1.92/4.76, R2 0.84-0.95. Deep trees' top MAE explodes
(b2@64: 18.7) — noisy hops compound; trees pay only at level 1 (frozen).

### Diagnosis: hops = analog relay without re-quantization (scan tokens requantize
every step; tree states don't). PATCH experiment (probe_patch.py, job 128816) live:
P0 control / P1 count-centroid (denoise, same subspace) / P2 digit-token embedding
norm-matched @L20 (vocab-code test) / P3 input-level text patch (ceiling). Whichever
arm rescues lvl2 names the minimal repeater: P1->linear denoiser, P2->quantizer to
token space, P3->carrier-style distilled SQ. All-fail -> LoRA the hop read.

## PATCH EXPERIMENT (128816) — MECHANISM NAILED: hops need RE-QUANTIZATION
patch_n8/patch.csv. Children patched @L20 w/ GOLD pair counts; lvl2/lvl3 probed @L24/27.
  P0 raw-state control:      lvl2 0.43, lvl3 0.08
  P1 denoised centroids:     lvl2 0.47          -> NOISE IS NOT THE PROBLEM
  P2 digit-token embeddings: lvl2 1.000 (!),    -> VOCAB CODE TRANSFERS PERFECTLY
                             lvl3 0.29 R2 0.99  -> but its OUTPUT is state-code again
  P3 input-text ceiling:     lvl2 0.98, lvl3 0.41
LAW: frozen attention moves vocab-coded content losslessly, state-coded content only
as analog (R2~0.99, exactness ~0.3) — at EVERY hop. Minimal repeater = per-level
linear probe -> write digit embedding into the SQ span (no LoRA, no backprop).
NEXT: repeater-tree probe (N=8 b=2, quantize lvl1@~L16, lvl2@~L24, root read @27;
probes calibrated on quantized-input states). Depth budget: ~7 layers/level -> ~3
levels/forward; deeper N needs fan-4 or multi-pass — real architectural constraint.

## REPEATER TREE (128824) — FIRST FULLY IN-MODEL EXACT COUNT
repeater_n8/repeater.csv. N=8 b=2, ONE frozen forward, quantize @L20 (Q1, predicted
counts -> digit embeddings) and @L24 (Q2), root read @L27:
  Q1 0.887/node -> sum(Q1) bound 0.550 -> sum(Q2) 0.550 (HOP 1 LOSSLESS)
  -> ROOT exact 0.517 (+-1 0.928, MAE 0.59) — at the bound within probe noise.
  vs flat frozen 0.239. Trained params: two linear heads. No LoRA, no generation.
Loss fully attributed to Q1 miscalibration (head fit on multi-arm capture layout,
applied to b2-only layout; unregularized). Known fixes -> expected ~0.75-0.8.
NEXT: (a) in-layout calibrated + regularized Q1 (+soft quantization: sum_k p_k e_k);
(b) scale to N=16-64 (fan-4 stages or multi-pass — depth budget ~7 layers/stage);
(c) benchmark task port (MMReD-HF steps_in_room via mmred evidence labels).

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

## REPEATER v2 (128829) — CALIBRATION WORKS; HARD BEATS SOFT
repeater2_n8/repeater2.csv. In-layout calibrated Q1 (C=0.5): 0.887 -> 0.967/node.
  hard: Q1 0.967, sum(Q1) 0.883 = sum(Q2) 0.883 (hop lossless AGAIN),
        ROOT last 0.778 (+-1 0.939, MAE 0.32) — 3.3x flat frozen (0.239).
  soft (sum_k p_k e_k): relay slightly LOSSIER (sum(Q2) 0.858), root 0.772 —
        soft quantization adds nothing; the clean discrete symbol is the point.
Losslessness explained: per-hop error on vocab codes at fan-2 ~ 0 -> nothing to
compound; total error = Q1's 0.967^4 ~ 0.874 (cancellation -> 0.883 measured).
Root-read gap (0.778 vs 0.883): probe-budget suspect -> repeater2b (128831,
150 calib / 480 eval) in flight. Scaling stresses identified for N=32/64: depth
budget (~3 quantize stages/forward -> multi-pass or fan-4), multi-digit codes,
fan-2 ADDITION vs operand magnitude (needs its own capacity curve); stage-1
perception compounds 0.967^(N/2) regardless (exact-metric ceiling ~0.34@64).

## REPEATER v2b (128831, 150 calib / 480 eval) — N=8 RESULT FINAL
repeater2b_n8/repeater2.csv.
  hard: Q1 0.983/node -> sum(Q1) 0.938 = sum(Q2) 0.938 (lossless x2)
        ROOT last 0.897 exact / +-1 0.992 / MAE 0.12   (mean 0.892)
  soft: 0.893 root, relay 0.931 — soft confirmed inferior; HARD is canonical.
v1's root gap was probe budget (60->240 fit rows: 0.639->0.892). Bottom line @N=8:
FROZEN MODEL + 3 LINEAR PROBES = 0.897 exact in-model counting vs 0.239 flat (3.8x),
5 pts off the external-compose bound; residual = Q1 perception ceiling.
Next levers (kept seq8): level-0 VERDICT quantizers (frame verdicts 0.985-0.999
decodable -> quantize before the pair read; expected sum bound 0.92-0.95+),
512px capture (+3 measured), 300-calib; then distilled SQ embedding (trained).
Scaling (N=32/64): multi-pass or fan-4 staging + multi-digit codes + fan-2
addition-vs-magnitude curve.

## RIDGE-ROUND REANALYSIS (128845) — CAPACITY LAW CORRECTED (Tal's challenge)
reanalysis_rr.csv. LR probes systematically understated exactness (deconfound.py:
ridge->round >> logistic on every noisy cell; P2 two-hop 0.294->0.967(!)).
CORRECTED frozen level-1 capacity c(fan) @L20 (rr, N=8/16/32/64):
  fan-2 : 0.983/0.990/0.984/0.968   (length-invariant ~0.98)
  fan-4 : 0.872/0.918/0.903/0.798
  fan-8 : 0.672/0.675/0.633/0.562
  fan-16: 0.444   fan-32: 0.208   fan-64: 0.117
=> "counts to ~4 nearly exactly, halves per doubling beyond." Flat@N=8 was 0.67
not 0.24. b2 lvl1 + external sum (rr-tally): 0.933/0.922/0.892/0.667.
Natural hop (lvl2 fan-2, rr): 0.65/0.67/0.61/0.54 — still THE bottleneck vs 0.98
leaf reads; quantized hops (P2 1.000, two-hop 0.967) still THE fix. Global-gain
jitter falsified (dirnorm no-op); residual hop noise mechanism open (per-read).
Repeater v2b (0.897 in-model @N=8) ~ matches external compose 0.933.

## SELF-QUANTIZATION PROBE (128844) — RAW LOGIT LENS INSUFFICIENT
selfq_n8/selfq.csv. Model's own final-norm+LM-head applied mid-forward at the
" Answer:"-cued node positions: frame verdicts 0.752@L27 (linear probe: 0.985),
lvl1 pairs 0.535 max (ridge: 0.983), lvl2 0.258, root 0.117; unrestricted top-1
is NEVER a digit -> mid-layer states are not emission-ready under this layout
(the classic raw-logit-lens failure). Parameter-free quantization does NOT work
off the shelf. Task-agnostic ladder for the quantizer decision:
  raw LM head 0.535 (free) < tuned-lens translator (affine/layer, trained on
  GENERIC text, no task labels — untested, the principled middle) < linear
  count-head 0.983 (tiny but task-specific — current repeater).
Day closed. Next session: tuned-lens quantizer test; N=16-64 repeater staging
(one quantize per TWO levels now justified); MMReD-HF port.

## SELFQ v3 (128848, merge-safe ' Answer: (' cue) — PARAMETER-FREE ROUTE CLOSED
selfq_v3_n8/selfq.csv. Fixed readout position + digit/word candidates + top-1 log:
frame verdicts 0.844 max (+words @L20), pairs 0.519 @L27 (top-1 digit only 48%),
lvl2/root ~chance. Top-1 narrative: mid-layers = logit-lens noise; L24 =
<|object_ref_start|>; L27 = <|im_end|> — the model wants to END THE TURN at inner
node positions, never enters answering mode. (v2 run 128847 was void: trailing-space
suffix merged into next question's first token -> all samples skipped; tokenizer
merge-safety now checked in-script.)
=> Quantizer decision function requires training. NEXT BUILD: self-distilled answer
lens W (affine, vocab codomain via frozen LM head): teacher = model's own alone-pass
answer distribution; loss = KL + aux-CE on TEACHER-ARGMAX (Tal: plain distillation
would let W absorb bulk gradients — stage3 lesson). Steps task first, no gold labels,
task-agnostic by construction.

## ANSWER LENS (128849 void teacher / 128850 forced-prefix teacher) — WORKS, TEACHER-LIMITED
lens2_n8/lens.csv (+lens_data.npz, per-layer ckpts). Teacher = model's own answer
distribution at a FORCED assistant prefix "Answer: (" (first attempt without forcing:
teacher top1==digit 0.000 — chat prose; run 128849 void). Teacher sanity: 0.710.
Lens (affine per layer, KL + aux-CE-on-teacher-argmax, pair-trained, no gold labels):
  pair-acc 0.786 @L27 (kl+ce; kl 0.740) — STUDENT EXCEEDS TEACHER (0.71);
  top1-is-digit ~1.0 (answer-mode restored); FRAME TRANSFER 0.964 zero-shot;
  lvl2 transfer 0.375 (support mismatch — trained digits 0-2 only).
Aux-CE > plain KL again (Tal's stage-3 lesson generalizes). CONCLUSION: the
task-agnostic quantizer exists and is ceilinged by the MODEL'S OWN native competence
(0.71 pair counting), not by our machinery; task-specific ridge (0.983) remains the
accuracy route. Open: better teachers (restricted/ensembled), support-complete
training mix, lens-based repeater v3 vs probe-based v2b (0.897).

## NIGHT RUN 2026-08-05/06 (Tal-authorized autonomous)
### textcount (128853 void: emission artifact — model says '( ' before digit;
### fixed via forced prefix "Answer: ( "; 128855 canonical)
textcount2/textcount.csv: NATIVE flat counting of PERFECT clean symbols (text bits,
gold<=9): 0.892/0.550/0.325/0.317/0.233/0.183 @ N=4/8/16/32/64/128.
=> "paste codes + aggregate at last token" FAILS at high N even in the model's best
modality. Staged bounded-fan-in aggregation is NECESSARY. Repeater tree @N=8 (0.897)
already beats native flat text counting @N=8 (0.550).

### repeater v4b/v4c (128854/128856) — EMISSION MECHANISM ISOLATED
v4b (far-span codes): quantizer chain best-ever IN-FORWARD (A_late Q1 0.994,
sum bounds 0.967-0.980) but EMIT-EM ~0.13 all arms — decode position doesn't
read mid-layer codes in far spans. v4c tail registers: even GOLD codes 3-8 tokens
before the reader emit at 0.113 — model emits '0', THE REGISTER'S ORIGINAL TOKEN.
=> MECHANISM: the copy/emission circuit keys on the source token's EARLY-LAYER
identity (embedding-level), not its overwritten mid-layer state. Probes read late
states; the model's own copying reads layer-0 identity. Single-forward mid-layer-
write emission is dead ARCHITECTURALLY; computed values must become REAL INPUT
TOKENS => TWO-PASS is the principled emission design (pass 2 cost: ~50 text tokens).
TP_pred arm scored 0.213 vs sum2 0.780 (conditional ~0.27 — template suspect);
addsanity sweep (128857) hunting the pass-2 phrasing.

### addsanity (128857) — pass-2 template law
addsanity/addsanity.csv. Force "Answer: ( " POISONS text arithmetic (model echoes
operand: tp_v1 0.24, plain 0.20); no force = no digit within 4 steps (prose).
WINNER: "Two partial counts are {a} and {b}. What is the total count?" + "Answer: "
= 1.000 (100/100). tp_v1/plain 0.88, story/plain 0.88, plain-arith/plain 0.60(!).
v4d (128858) = late chain (Q1@20 0.994, sum2 bound 0.980) + winning pass-2 IN FLIGHT.

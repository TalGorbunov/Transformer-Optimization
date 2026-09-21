# LEARNMASK — campaign state

Newest last. Brief: CAMPAIGN_BRIEF.md.

## 2026-08-12 — campaign opened (proposal stage)
- Peer meeting output: learned discrete mask from fence init, Gumbel/STE machinery.
- Brief drafted; NO code, NO jobs. Waiting on Tal: arms OK, P1 smoke budget OK.
- Next concrete step: P0 (CPU) gate module + bit-for-bit fence-init parity test.

## 2026-08-12 — data spec set by Tal
- Dataset = data/mmred_hf (original Fr0do benchmark), resolution 512 (not FRAME_RESIZE
  392). Native seq_len splits 2..128 → P4 transfer runs on the benchmark's own tests.
- Consequences recorded in brief: ~324 tok/frame expected (verify), K0-sorted-trap
  handling mandatory on every split, carrier-ckpt@392 distribution-shift confound →
  P1 logs frozen hand-fence baseline on hf@512 first. Nothing launched.

## 2026-08-12 — Tal GO: arms + estimator + budget confirmed
- Estimator default E3 (ST-Gumbel); arm order S1 → S2 → S3 + S0 (S0 diagnostic
  approved); gates-only training (carriers+LoRA frozen at the canonical ckpt, joint
  gate+LoRA OUT of scope); data = mmred_hf @512 per the brief. Group name `learnmask`.
- Phase gates: P0 CPU-only now; P1 smoke pre-approved (one 2h_2g job); HARD STOP
  after P1 — results in tables, no P2+ submission without Tal's OK.

## 2026-08-12 — P0 done (CPU): gate module + parity + trainer; all tests green
- **gnnformer/learnmask.py**: relation vocabulary as 22 channels/layer (R1, R2×6Δ,
  R3, R4×6Δ, R5×6Δ, R6, R7; R8+tail→tail causal = anchor, never learnable; Δ-buckets
  {1,2,3-4,5-8,9-16,17+}). `relation_cell_map` classifies every (q,k) cell (row-chunked
  for P4 scale); `mask_parts`+`assemble_mask` = differentiable per-layer assembly
  (base + (1-g)[cell]·K, one gather+FMA); `MaskGates` carries all four estimators
  (E1 det-STE / E2 soft-anneal / E3 ST-Gumbel default / E4 hard-concrete), deviation
  penalty, p_open heatmap, flips/stats instrumentation. ST forwards emit EXACT 0/1
  (Sterbenz-ordered straight-through). Train-mode closed learnable cells use
  SOFT_FORBID=-30; frozen relations + hard-frozen eval always MASK_MIN (brief gotcha).
- **Per-layer mask path** = `gated_stack_logits` (full-stack TF forward mirroring
  CarrierEngine.forward_logits geometry). engine.py/fencing.py UNTOUCHED — gates train
  at all 28 layers so the cached-lo-phase trainer path can't apply anyway, and the
  anchored code carries zero regression risk. In-run parity instrument compares
  handfence-as-gates vs engine.forward_logits on real samples at startup (fail >1e-3).
- **Parity pinned bit-for-bit** (tests/test_fencing.py extended): fence-init hard
  assembly == build_block_mask(hide_cols=carriers) == make_masks lo at every layer;
  deployed hand design (R4+R7 open ≥ L_OPEN) == make_masks lo/hi exactly (the baseline
  is a POINT in gate space); both also with appended TF/decode rows (== ext_mask); on
  two layouts incl. a 20-block one covering all Δ-buckets. + tests/test_learnmask.py
  (13 tests: partition totality, cell classes, arm sizes 2/14/22 = brief's 56/392/616
  logits, estimator ranges/hardness/gradients, penalty values, state roundtrip).
  FULL tests/ suite green (9 files).
- **Trainer**: scripts/learnmask/train_mask_gates.py — gates-only Adam, τ anneal
  geometric τ0→τ1 over steps, CE + λ_open·Σp_open (+λ_close·Σ(1-p) for S3), eval and
  ckpt selection always on the HARD-frozen mask, ep0 emits BOTH the handfence
  reference row and the fence-init row, per-epoch gate stats + heatmap_ep*.csv +
  gates_last/best.pt + incremental report.txt (walltime-kill-safe). Class distribution
  reported on every split (K0 trap); stratified shuffled sampling via
  iter_sample_dirs_shuffled. slurm/train_mask_gates.sbatch added (env-driven, 2h_2g).
- **P0 data checks closed**: counting variant = `steps_in_room` (exact headline task
  wording "How many steps did C spend in the R"). Train splits exist at seq_len
  {2,4,8,16} (json + materialized dirs, 200 dirs each for qtype-filtered train;
  seq_len_8_val_steps_in_room = 50 dirs; test/headfit mixed-qtype 50/qtype). Class
  dist seq8 train steps_in_room = 125/200 K0 (62.5% majority-0) → trap confirmed live,
  mitigations wired in. **tokens/frame @512 VERIFIED = 324** (512→504 snap, grid
  36×36; scripts/learnmask/p0_token_check.py): block = 327 tok (324 img + 2 markers +
  1 carrier), seq @N=8 = 2655; projected N=128 ≈ 41.9k tokens → 7.0 GB fp32 mask per
  layer, confirming per-layer in-hook assembly (never 28× buffers). mmred_hf renders
  are natively 512×512 (resolution 512 = no resize at all).

## 2026-08-12 — P1 smoke SUBMITTED (job 131330, l40s-shared, 2h_2g, --time 02:00)
- One job = both P1 deliverables: ep0 handfence row (frozen hand-fence baseline on
  mmred_hf@512, the reference row) + fence-init row, then S1 (R6/R7, 56 logits,
  ST-Gumbel) training: train = seq_len_8_train_steps_in_room limit 120 (stratified
  shuffle), eval = seq_len_8_val_steps_in_room (50), 6 epochs, lr 3e-2, λ_open 1e-2,
  τ 2.0→0.5, grad-ckpt on. Output → outputs/learnmask/p1_smoke/.
- GPU check at submit: h200 full 8/8, a100-public 1 free, l40s-shared n314 5 free,
  rtx6k n318 5 free → l40s-shared. Queue empty for user.
- Will record: min/step (recalibrate the brief's cost table), parity result,
  baseline-vs-init-vs-trained table. HARD STOP after this job: tables to Tal, no P2+.

## 2026-08-12 — job 131330 FAILED at its own parity gate (4 min); diagnosed + fixed
- The gate worked as designed: max|Δlogit| 1.56e-02 > 1e-3 threshold → trainer refused
  to train. Diagnosis: the two compared forwards ran DIFFERENT sequence lengths (gated
  TF appends all e target rows; the engine comparison got extra=tgt[:-1] to align rows)
  — different row counts change attention-kernel tiling, and bf16 reduction-order noise
  through 28 layers alone accounts for ~1.6e-2. NOT a mask defect: verified by a CPU
  real-layout check (exact tokenized geometry @512, scratchpad script): cell-map fence
  == make_masks lo, hand gates == hi, both also with 33 appended rows — ALL bit-for-bit
  on 3 real samples; carrier confirmed block-last; blocks[0]=(24,351), seq 2655.
  (Same lesson family as [[bf16-bit-identity-canary]]: bit-identity only under
  identical shapes/kernels.)
- Fixes: gated_stack_logits now also returns the post-target continuation row (e+1
  rows; CE callers slice [:-1]) so parity compares the SAME row of SAME-shape forwards
  vs forward_logits(extra=tgt); trainer additionally asserts real-layout lo/hi mask
  parity bit-for-bit at startup (mask-parity line), and rewrites report.txt every
  epoch (walltime-kill safety). Tests + compile re-run green.
- Useful facts from the failed run: prep 170 samples = 82s; train class-dist after
  stratified shuffle = majority 0.38 (0:45 1:21 2:16 3:7 4:7 5:9 6:8 7:5 8:2); eval
  (val split) majority 0.62 (0:31) — skewed, per-gold table will carry it. MaxRSS
  5.9G. GPU cost of the failure ~4 min.
- **Resubmitted as job 131332** (same config, l40s-shared 2h_2g --time 02:00).

## 2026-08-12 — job 131332 failed the gate too (7.81e-03); head-kernel isolated → 131334
- Same-shape fix halved the drift (1.56e-2 → 7.81e-3) and the NEW in-run real-layout
  mask parity printed bit-for-bit OK — so masks/geometry are exact and the residual is
  again kernel-shaped: engine heads its last row as a 1-D GEMV (h[0,-1] @ lm_head),
  the gated path heads all TF rows as one GEMM — different matmul kernel, different
  bf16 reduction order. With bit-equal masks/emb/pos through identically-shaped layer
  calls the hidden states should be bit-identical; the head is the only remaining
  structural difference.
- Fix: gated_stack_logits gained return_h; the parity instrument now re-heads the
  last row 1-D (the engine's exact call) and HARD-FAILS only on that kernel-matched
  comparison (still 1e-3); the row-GEMM diff + argmax agreement stay as info lines.
  If the 1-D number is not ~0 the gate still refuses to train — that would mean a
  real stack divergence, not head noise. Tests + compile green.
- **Resubmitted as job 131334.** GPU cost of 131332 ~4 min.

## 2026-08-12 — job 131334: PARITY EXACT (0.00e+00); then killed for the metric pivot
- The kernel-matched comparison returned **max|Δlogit| 0.00e+00 on both samples**
  (row-GEMM info number 7.81e-03, argmax agreed) — the gated per-layer forward is
  BIT-IDENTICAL to the anchored engine; both prior failures were kernel-shape numerics
  as diagnosed. The per-layer gate machinery is proven on live geometry.
- ep0 rows landed before the kill (caption ckpt, TF metrics — recorded for the
  archaeology only, these metrics are now deprecated): handfence TF-count 1.000 /
  tf-exact 0.620 / CE 0.1099; fence-init 1.000 / 0.620 / 0.1100 — TF-count saturated,
  tf-exact == the val majority-0 fraction (31/50): the TF readout has no resolution
  here, consistent with the gating campaign's copy-detector finding.
- Killed at ~18 min (mid-epoch-1) on Tal's call — see next entry.

## 2026-08-12 — Tal: DEPRECATE the TF/scratchpad metric everywhere; direct answer-class CE
- New metric policy (brief updated with a dated addendum in Losses & schedule):
  objective + metric = the ANSWER directly. `--target class` (default): CE over the
  10 digit-token logits at the answer position (= last prompt row, e=0 — NO appended
  rows, nothing copyable); headline = class acc (0-9-restricted argmax) + class CE;
  unrestricted-argmax EM reported as the mass-drift canary (restricted CE cannot see
  probability mass leaving the digit set). Golds >9 → `--target digit` (digit-sequence
  CE + greedy emitted EM, gating-P7 convention; gated_greedy_digits added).
- Frozen carrier stack switched to the DIGIT-readout ckpt: gating P7a LoRA control
  (job 129918, `--digit-multi`, no scratchpad, BEST 0.924 @ep5) — promoted as
  `checkpoints/carrier_layer_digit_p7a_lora_best.pt` (+README row). Confound note:
  it was trained on the park 16-root mixture @392 → park→hf + 392→512 shift is IN
  the baseline row, which is exactly what the handfence ep0 row measures.
- Trainer rewritten around class CE (TF metrics removed); MaskGates.hard_open_table
  added (hard-frozen eval/decode via memoized distinct layer masks); tests extended
  (14 green) + full fencing suite green. Also caught: a `2>/dev/null | tail` pipe was
  masking a test ImportError (typo hand_open_table/hard_open_table) — fixed; note
  pipelines hide exit codes.
- **P1 smoke resubmitted as job 131337** (class CE, digit ckpt, same budget:
  l40s-shared 2h_2g, S1, limit 120/50, 6 ep).

## 2026-08-12 — P1 smoke DONE (job 131337, TIMEOUT @2h after ep3/6 — report intact)
Run: outputs/learnmask/p1_smoke/20260812_174838_s1_st-gumbel_class (gates_last @ep3,
heatmaps ep1-3; incremental report.txt saved the run through the walltime kill).
- **Machinery: fully validated.** Engine parity EXACT (max|Δlogit| 0.00e+00, both
  samples, 1-D head AND row-GEMM); real-layout mask parity bit-for-bit; gates receive
  gradients (max|Δlogit| grew 0.44→1.08); stats/heatmaps/ckpts all land.
- **Baseline rows (carrier scaffold = P7a digit ckpt, mmred_hf@512 seq8 val n=50):**
  handfence class_acc 0.940 / class_ce 0.2201; fence-init 0.940 / 0.2293. The
  park@392→hf@512 shift costs little (0.94); the handfence-vs-init gap at N=8 is
  Δce 0.009, Δacc 0 — closing ALL aggregation is nearly free at N=8 under the digit
  metric (tail reads frames directly; consistent with TRUNC P0.1 + token-necessity).
- **S1 verdict at N=8: gates correctly learn to change nothing.** 0/56 flips over 3
  epochs; R7 p_open ~0.10 (stays closed — opening cannot pay for its λ=0.01 penalty
  when the CE gain is ~0), R6 ~0.89 (stays open). HARD metrics pinned at the init row.
  This is the deviation penalty working as designed, and it means N=8 has no CE
  pressure toward aggregation — **the informative training regime is seq_len ≥ 16**,
  where the handfence-vs-init gap must first be measured (ep0 rows do it for free).
- **Measured cost (recalibrates the brief's table):** 1900 s/epoch = 120 train steps
  + 50 eval forwards → **~14.6 s/train-step** at seq 2655 (L40S, grad-ckpt, ST-Gumbel,
  class CE e=0). Brief's P2 estimate holds at N=8 (500 ex × 5 ep ≈ 10 h, 24h_1g);
  seq16 ≈ 2.5-3×/step → right-size to ~300 ex × 4 ep ≈ 13 h or use A100/H200.
- HARD STOP honored: no further submissions. Pending Tal: replica-first scaffold
  redesign (replicas = zero trained components, removes the LoRA-trained-under-the-
  fence circularity + the @392 ckpt confound), per-arm zero-shot transfer evals,
  seq16 training regime for S2.

## 2026-08-12 — Tal: REPLICA scaffold + NO leading question; implemented (CPU, green)
- Decisions: question replicas per frame instead of the distilled carrier (zero
  trained components — gate logits become the ONLY parameters in the system; kills
  the LoRA-trained-under-the-fence circularity and the @392 ckpt confound), and the
  leading question is REMOVED (replicas carry the conditioning; prefix = 14-token
  chat preamble).
- Implementation: relation_cell_map + masks generalized to reader SPANS (carrier =
  the single-token special case; carriers.make_masks reproduced bit-for-bit by
  make_masks_spans — pinned); prepare_sample_replicas (replica prompt, posreset
  verbatim, no e_c); trainer --scaffold replica|carrier (replica default) with a
  third ep0 row `nofence` (all-open = plain causal); engine.forward_logits got a
  behavior-neutral empty-cpos guard so the parity instrument runs on replica records
  (span lo/hi injected via d["lo"]/d["hi"]).
- Tokenizer gotcha (measured): a "\n" separator before the final question FUSES with
  the last replica's '?' into one token and breaks span location; a SPACE separator
  keeps '?' intact → locate_replica_layout uses the bare-question needle (NF replica
  hits after <|vision_end|>) + the " "+question needle (exactly one hit = tail start).
- Verified on real mmred_hf@512 seq8 data: 4/4 samples, replica spans inside blocks,
  fence/hand masks bit-for-bit vs make_masks_spans; layouts UNIFORM (seq=2717 all —
  relevant for the S0 per-cell variant). Full tests green (fencing 12, learnmask 14,
  carrier_masks 9, rest unchanged). Nothing launched.

## 2026-08-12 — Tal's SWEEP design adopted; all build items done (CPU, green)
- Sweep = the campaign's P2+P3+P4 compressed into one grid on the replica scaffold:
  **S2 relations (R4/R5/R6/R7, 392 logits) × estimators {E1 ste, E2 soft-anneal,
  E3 st-gumbel} × mixed train seq{8,16} → zero-shot transfer eval at seq{32,64}**,
  plus the **S0 per-cell sweep × the same 3 estimators @seq8 only** (no transfer —
  layout-bound by construction). Clarified with Tal: S2 scope confirmed; init keeps
  the strict fence (tail does NOT see replicas at init — opening must pay); class CE
  trains on golds<=9 only (~7% of seq16 skipped; topology doesn't care), evals score
  gold>9 per-sample via greedy digit-sequence decode so EM covers everything.
- Built since: trainer multi-root roots (path=LIMIT, per-root prep lines) +
  per-sample eval rule + --qtype-filter (mixed test splits carry all 24 qtypes;
  steps_in_room = 50/dir in seq{16,32,64}_test) + --arm s0; FreeTableGates (per-cell
  logits over the S2-scope cells, ~1e5-1e6 cells x 28 layers, fence-init bit-for-bit,
  scatter-assembled differentiable masks, relation-grouped audit heatmap);
  hard_mask_lut (device-side per-layer mask assembly for N=64 where one fp32 mask is
  ~1.8 GB — never 28 at once); scripts/learnmask/eval_mask_transfer.py (+
  slurm/eval_mask_transfer.sbatch, slurm/lib/roots_learnmask_{816,transfer}.txt —
  comma lists live in FILES, never --export). Tests: 16 learnmask + 12 fencing green;
  everything compiles; wrapper dry-runs assemble the mixed-root command correctly.
- Cost plan (measured 14.6 s/step @seq8; ~2.5x est. @seq16): 3 gate-sweep jobs
  ~9 h each (24h_1g, spread partitions); 3 S0 jobs ~3 h each; transfer evals ~1-2 h
  each on 2h_2g after training lands. NOTHING LAUNCHED — awaiting Tal's GO.
- Plots (deferred until there are learned masks to plot): (i) relation x layer
  heatmap init-vs-learned per estimator; (ii) acc-vs-N lines handfence/init/nofence/
  learned; (iii) S0-vs-relation-gate audit figure; (iv) p_open trajectories.

## 2026-08-12 — Tal GO → SWEEP LAUNCHED (6 jobs)
- Clarifications closed before launch: S2 gates are bidirectional (R6 fence-ON can
  CLOSE per layer — readout suppression is in scope; full suppression = S3 later);
  granularity worry → S0 is the audit, per-head gates the ready escalation; eval
  never filters gold>9 (class-CE restriction is train-loss-only); transfer-eval EM
  for gold<=9 fixed to UNRESTRICTED emission (restricted argmax was slightly
  generous; restricted stays as the labeled class-acc diagnostic).
- Wave 1 (S2 replica, class CE, train 150@seq8 + 150@seq16 via
  roots_learnmask_816.txt, eval 30@8val + 30@16test steps_in_room via
  roots_learnmask_eval816.txt, 4 ep, 24h_1g --time 16:00):
  **131378** ste @l40s-shared · **131379** soft @rtx6k-shared ·
  **131380** st-gumbel @a100-public.
- Wave 2 (S0 per-cell @seq8 only, 150 train / 50 val, 4 ep, 12h_4g --time 06:00):
  **131381** ste @l40s-shared · **131382** soft @rtx6k-shared ·
  **131383** st-gumbel @l40s-public.
- Abort rule armed: wave-1 ep0 rows (handfence/init/nofence at 8 AND 16) print in
  the first ~15-20 min; if handfence≈init at seq16 (no CE pressure), stop the sweep
  and reconsider (train at 32 / carrier scaffold) instead of burning 40 GPU-h.
- Transfer evals (wave 3) launch AFTER wave-1 training lands, per estimator:
  eval_mask_transfer with roots_learnmask_transfer.txt (seq32_test=100, seq64_test=60).

## 2026-08-12 — ABORT RULE FIRED: replica baselines at FLOOR → sweep stopped, readout probes out
- ep0 rows (job 131379, eval 30@8val+30@16test): handfence class_acc 0.267 / ce 2.93,
  init 0.300 / 2.19, nofence 0.133 / 3.54 — and **em 0.000 in every regime**: the
  PLAIN frozen model never emits a digit at the answer position (it opens a sentence);
  the digit-restricted read is majority-class collapse (handfence/init → "0",
  nofence → "1"; uniform CE = 2.30 for reference). No topology signal in that
  objective — gates would learn calibration of a non-answer. The carrier stack's
  0.94 came exactly from its digit-multi LoRA emitting the digit at that position.
- ALL SIX sweep jobs cancelled (~30 GPU-min total). Parity remained exact (0.00e+00)
  and mask parity bit-for-bit — the machinery is fine; the READOUT POSITION is the
  problem on the untrained scaffold.
- Fix under test (prompting, still zero trained components): --answer-hint (user-side
  "Answer with the number only." after the final question) and --answer-prime
  (assistant-side "Answer:" after the generation prompt; read position follows it).
  Probes (eval-only, --epochs 0, ep0 rows @8+16): **131388** hint · **131389** prime ·
  **131390** hint+prime (2h_2g, ~20 min each). Relaunch the sweep with the winning
  variant; if all stay at floor → back to Tal (options: train at 32, carrier
  scaffold with its bias made explicit, or a minimal untrained readout).
## 2026-08-12 — canonical tail FIXES emission but exposes the real wall (probe 131393)
- Tal's catch was right: the codebase already had the fix — data.build_count_prompt
  (PROMPT-CRITICAL wording). Wired as the replica tail (--tail-style canonical,
  default; fin now = last-replica-end so the instruction lives in the tail; verified
  bit-for-bit on real data; improvised hint/prime probes cancelled unused).
- Probe rows (eval 30@8val+30@16test): em == class_acc now (digits emitted ✓):
  handfence 0.283/ce 2.93 · init 0.333/2.17 · nofence 0.333/2.53. BUT per-gold shows
  pure majority-class behavior: every regime answers "0" (~all gold-0 correct, 1-2
  hits elsewhere), MAE 2.5-3.2. **The frozen 7B cannot count on mmred_hf@512 at
  seq8/16 under ANY attention topology** — nofence (plain model) included; the fence
  even has slightly WORSE answer-CE than init/nofence on a frozen model (it hides
  context a model that never learned carriers might use). Consistent with the thesis
  story: the method's 0.94 = fence + TRAINED carriers/LoRA readout; the A3 replica
  d′ 6.34 was about measurable SUPPLY, not end-to-end frozen emission.
- Consequence for the campaign premise: end-task CE on a 100%-frozen model carries
  ~no topology signal at these lengths (gates route information; they cannot teach
  the frozen readout arithmetic). Gates trained on this objective would tune the
  majority posterior. SWEEP STAYS DOWN. Options for Tal (recommendation = C):
  A. carrier-scaffold sweep with the existing P7a digit ckpt (signal exists — 0.94
     baseline; bias stated: LoRA was trained UNDER the hand fence @392; S0 audit and
     suppression results remain clean; fastest);
  C. train ONE fence-agnostic digit readout on mmred_hf@512 first — LoRA on all
     layers, replica scaffold, mask FIXED at nofence (never sees any fence), digit
     target; freeze it; run the sweep on top. Clean rediscovery claim: if gates then
     converge fence-like, topology genuinely helps a readout that never saw the
     fence; if they converge nofence-like, the fence is NOT locally optimal for a
     fence-agnostic readout — honest either way. Costs one training job + a modest
     trainer extension (--train lora --fixed-regime nofence).
  (B probe-style supply objectives and D train-at-32 considered and rejected: B
  abandons the direct-answer metric policy; D makes the frozen floor worse.)

## 2026-08-12 — Tal's challenge accepted: EXISTENCE TEST instead of asserted negative
- Tal (correct): 3 probed topologies ≠ proof that no mask unlocks frozen emission;
  the "missing readout circuit" was inference, not measurement. The campaign's own
  instruments answer it: S0 = the mask ORACLE at N=8; S2+SGD = the transferable
  version. Staged plan: probe the untested readout-through-supply topologies → 1xS2
  + 1xS0 existence test (penalty OFF — unconstrained search) → full estimator sweep
  on whichever scaffold the result dictates; fence-agnostic LoRA readout (--train
  lora, nofence-pinned) if the negative is confirmed.
- Supply probes (job 131402, canonical tail): supply (tail reads ONLY replicas)
  class_acc 0.283 / CE **4.02**; handsupply 0.283 / 3.67. Ordering: init 2.17 <
  nofence 2.53 < handfence 2.93 < handsupply 3.67 < supply 4.02 — the harder the
  readout is forced through the supply positions, the WORSE the frozen answer
  distribution. The frozen model cannot decode replica states into digits (probe-
  decodable ≠ model-readable). Strengthens the missing-readout story; oracle closes it.
- **Existence test launched**: 131403 S2 st-gumbel (8+16, LAM_OPEN=0, 24h_1g/16h,
  l40s-shared) · 131404 S0 st-gumbel (seq8, LAM_OPEN=0, 12h_4g/6h, rtx6k-shared).
  Success criterion: eval accuracy meaningfully above majority (0.33/0.62-adjusted)
  → frozen-scaffold sweep resumes. CE-only movement at majority accuracy → measured
  negative (topology cannot cash supplied info into emission; the wall is the
  readout — consistent with the superquery capacity law) → LoRA-readout arm.
- Probe footgun for the record: '+'-separated EP0 list initially fell through the
  regime parser silently (comma can't ride --export); parser now splits on '+' and
  REJECTS unknown regime names. 131401 cancelled pre-rows; rerun as 131402.

## 2026-08-13 — EXISTENCE TEST: NEGATIVE, measured. No mask makes the frozen 7B count
- **S0 oracle (131404, COMPLETED 13 min, 139 s/ep)**: 6.7M per-cell logits, penalty
  OFF, 4 epochs: train CE fell 1.63→1.22 (sampled-mask calibration) but the HARD
  mask **never flipped a single cell (0/6,698,496)** and eval stayed bit-identical
  to init every epoch (class_acc 0.600 = majority, CE 1.3103, gold-0 only).
- **S2 (131403, TIMEOUT@16h, 3/4 epochs, 16,019 s/ep)**: gradients real
  (max|Δlogit| 2.28) yet 1/392 flips by ep3; ep1-2 eval bit-identical to init; ep3's
  single flip traded gold-0 hits for gold-2 (acc 0.333→0.267, CE 2.17→2.05) —
  calibration shuffle, not counting.
- Verdict (with the supply probes): **topology cannot cash supplied information into
  emission on the frozen model** — unconstrained search over both the transferable
  gate space and the per-cell oracle found nothing above majority. This is now a
  MEASURED result (thesis-grade negative: the wall is the readout, consistent with
  the superquery capacity law), not an inference. Frozen-scaffold sweep is moot.
- Cost lesson: seq16 replica steps ≈ 88 s (grad-ckpt, L40S) → 16k s/epoch for the
  8+16 mix; future 8+16 jobs need seq16 caps (~100) / 3 epochs / bigger walltime.
- **--train lora implemented** (trainer): fence-agnostic digit readout — fresh LoRA
  on ALL 28 layers, mask PINNED at --fixed-regime (nofence default: never sees any
  fence), class CE, saves readout_last/best.pt; gate runs then load it FROZEN via
  --readout-ckpt (replica scaffold). Wrapper knobs TRAIN/FIXED_REGIME/READOUT_CKPT.
  Compiles, 16 tests green, dry-run OK. AWAITING Tal's GO for the readout job
  (proposal: seq8-only first, 150 train / 6 ep ≈ 3.5-4 h single GPU).

## 2026-08-13 — Tal GO on the readout pipeline; s2open pin; 20-ep S0 update
- Readout pin decision (Tal asked for recommendation, accepted): **s2open** — train
  the readout under "everything the S2 sweep can REACH, open" (frames isolated,
  R4-R7 open at all layers) instead of pure nofence: no train/deploy shift inside
  the searchable family, so the sweep measures pure edge NECESSITY ("which edges
  pay for themselves from the fence init"). nofence-pinned readout kept as a
  one-job reviewer ablation. Implemented as --fixed-regime s2open (default).
- Expectations recorded (pre-registration flavor): H2 likely at train lengths (a
  trained readout may not need cross-frame edges at N=8/16); the DECISIVE
  instrument is the acc-vs-N transfer at 32/64 (length generalization is the
  fence's actual selling point); H1 signature = R4/R7 banded opening; estimators
  expected to agree in conclusion, differ in dynamics (E1 may stick).
- **Readout job launched: 132465** (TRAIN=lora, s2open, seq8 150 train / 50 val,
  6 ep, lr 1e-4, 12h_4g l40s-shared, --time 08:00). Sanity gate before the sweep:
  eval acc >> 0.62 majority at seq8. Pipeline after the gate (GO'd): promote ckpt →
  3-estimator S2 sweep from strict fence init (λ_open on) via --readout-ckpt →
  transfer evals 32/64 → heatmap + acc-vs-N figures.
- 20-ep S0 extension (132431) mid-run finding: with lr 5e-2 the search now CROSSES
  boundaries (6,300 flips by ep5) and hard eval STILL does not move (0.600, CE
  marginally worse) — the strongest form of the frozen-scaffold negative. Final
  table when it completes. Peer-share script scripts/learnmask/s0_free_table_exp.py
  smoke-verified end-to-end (job 132432, 2m35s).

## 2026-08-13 — 20-ep S0 FINAL (132431, 50 min): the definitive frozen-scaffold negative
- Flips 0 → 6.3k (ep5) → 219k (ep10) → 603k (ep15) → **947,575/6.7M (ep20, 14%)**;
  train CE 1.58 → **0.32** (near-perfect train fit — per-cell position memorization,
  exactly the S0 failure mode the brief predicted); HARD eval acc 0.600 → 0.575
  (BEST 0.650 @ep14 = +2 samples over the 0.60 majority on n=40 — noise); eval CE
  1.31 → **1.94**, worsening monotonically as memorization deepens.
- Verdict, final form: the mask search has ABUNDANT capacity and gradient signal on
  the frozen model (it can rewire 14% of edges and memorize the train set through
  the mask alone) yet produces ZERO generalizing counting — the oracle bound on
  generalizable frozen-model counting via attention topology is the majority class.
  Kills "search too weak" conclusively; run dir outputs/learnmask/exist_s0_long/.
  Candidate RESULTS.md entry — awaiting Tal's "log this".

## 2026-08-13 — readout v1 FAILED the sanity gate (undertrained); v2 launched
- 132465 (s2open LoRA, 150 samples @seq8, 6 ep, lr 1e-4, 35 min): BEST eval 0.680
  vs 0.62 majority (+3 samples = noise); train CE still falling (1.69→0.72) at the
  end. Diagnosis: SIZING — 900 steps / 150 samples vs P7a's ~22k steps / ~4.4k
  pooled samples. Not a design verdict.
- **v2 = 132498** (12h_4g l40s, --time 11:00): ALL steps_in_room train data at the
  cheap lengths (seq 2+4+8, 200 each = 600 samples via roots_learnmask_readout248),
  lr_lora 3e-4 (LR_LORA env added to wrapper), 5 ep ≈ 3k steps, eval 30@8val +
  30@16test. Sanity gate unchanged: seq8 eval acc >> 0.62; also read the seq16
  half (readout never trained there — its zero-shot behavior shapes whether the
  sweep trains at 8 only or 8+16). Escalation if v2 also stalls: pool the other
  numeric qtypes (crowd_count, n_empty, n_char_at_frame, rooms_visited — P7a-style
  multi-task mixture) before questioning the approach.

## 2026-08-13 — readout v2: FIRST counting signal; validation + multi-task v3 out
- 132498 (600 samples seq2+4+8, lr 3e-4, 5 ep, 66 min): **BEST 0.550 vs 0.283
  majority on the mixed 8+16 eval** — first model-EMITTED counting above majority in
  the campaign (per-gold: 1:7/8, 2:4/7, 4:3/8, one gold-14). But unstable
  (0.37↔0.55 across epochs; train CE 0.275) — single-task overfit variance.
- **132501** validation: v2-best (readout_best.pt @ep4) on FULL splits 8val=50 /
  16test=50 / 32test=50 × regimes s2open/hand/init/nofence via the extended
  eval_mask_transfer (--readout-ckpt + s2open regime added). Answers: is 0.55 real,
  and how does the readout decay with length under each topology?
- **132500** readout v3: P7a-style multi-task numeric mixture — 4 qtypes
  (steps_in_room, crowd_count, n_empty, rooms_visited) × seq{2,4,8} × 100 = 1200
  samples (roots_learnmask_readout_multi), lr 2e-4, 4 ep, 24h_1g rtx6k, ~12 h.
  Trainer --qtype-filter now takes +/comma lists (QTYPES env).
- Footgun caught pre-crash this time: REGIMES='+'-list would have fallen through
  the transfer script's comma-only parser to gates=None; parser now splits on '+'
  and validates names (same fix as EP0 earlier — pattern: EVERY list-valued env
  knob must '+'-split and validate).

## 2026-08-13 — READOUT VALIDATED (gate PASSED) + the topology ordering IS the fence story
- 132501 validation of v2-best on FULL 50-sample splits (em):
  | topology | seq8 | seq16 (zero-shot) | seq32 (zero-shot) |
  | s2open   | 0.820 | 0.580 | 0.300 |
  | hand     | 0.780 | 0.540 | 0.260 |
  | nofence  | 0.620 | 0.420 | 0.220 (MAE 4.88 — worst blowups) |
  | init     | 0.600 | 0.420 | 0.160 |
  Sanity gate PASSED (0.82 >> 0.62 majority @8). The readout USES aggregation edges
  (init/nofence each cost ~20 pts @8) and **fence-family topologies win increasingly
  with length** (+16/+12 pts over nofence @16, graceful MAE @32) — a miniature of
  the campaign's headline acc-vs-N figure, on a readout trained only at <=8.
- **Estimator sweep LAUNCHED on v2-best** (gates only, strict fence init, lam_open
  1e-2 back on, --readout-ckpt frozen): **132503** ste @l40s-shared · **132504**
  soft @a100-public · **132505** st-gumbel @l40s-public; 150@8+100@16 (new
  roots_learnmask_sweep816), 3 ep, 24h_1g --time 14:00 (~10 h each). Question:
  starting from init (0.60/0.42), which edges does CE re-open under the penalty —
  sparse hand-like structure (0.78/0.54) or full s2open (0.82/0.58)?
- v3 multi-task readout (132500) still training; if it validates much better than
  v2, the estimator-winner gets one confirmation run on it.

## 2026-08-14 — SWEEP VERDICT: coordination barrier — gates RETREAT into the fence
- All three estimators (132503 ste 9.8h · 132504 soft TIMEOUT@14h ep2 (a100 is
  ~1.5x slower: 17.3ks/ep — avoid a100 for these) · 132505 st-gumbel 9.8h), from
  the strict fence init with lam_open=1e-2: **0/392 flips everywhere and p_open of
  R4/R5/R7 DROPPED 0.12 → 0.02-0.10** — the gates dig deeper into the fence while
  hard eval stays frozen at the init row (0.367/0.350; ep0 rows confirm the
  validation ordering: handfence 0.517 > nofence 0.4 > init 0.35 on the mixed eval).
- Reading: a **coordination barrier**, not an absence of value — full opening is
  WORTH +22 pts (validated), but gates sample independently and opening any single
  (channel, layer) edge alone buys ~nothing, so near the closed init the CE
  gradient is flat and the penalty dominates. Real optimization-landscape finding;
  ST/Gumbel per-gate exploration cannot discover jointly-valuable openings from a
  closed init.
- Fix (L0-pruning direction): **prune-from-open** — init ALL learnable gates open
  (= s2open, the readout's native regime) and let lam_open prune; surviving edges =
  the necessity answer, approached from the side where gradients exist. Implemented
  as --gate-init open (MaskGates init_open; flips counted vs the open init; 17 tests
  green). v3 apples-to-apples validation running (132549) → prune arm launches on
  the better readout (st-gumbel first, single job).
- v3 validation (132549): confirms the topology ordering (s2open/hand > nofence at
  every length; nofence MAE 5.04 @32) but v2 dominates on steps_in_room (0.82 vs
  0.64 @8, 0.58 vs 0.40 @16) — multi-task diluted the target task. **Readout pick:
  v2-best.** v3 kept as the multi-task robustness ckpt.
- **Prune-from-open launched: 132567** (st-gumbel, GATE_INIT=open, lam_open 1e-2,
  v2 readout frozen, 150@8+100@16, 4 ep, 24h_1g l40s --time 16:00, ~13.5 h).
  At init the hard mask == s2open (0.82/0.58 validated); the question: which edges
  survive the penalty — sparse hand-like band (rediscovery) or everything (fence
  not locally optimal from the open side)?

## 2026-08-15 — PRUNE-FROM-OPEN: clean structure found (the campaign's payoff)
- 132567 (13 h, 4 ep): pruning WORKS where opening could not — flips 0→139(ep2)
  →179(ep4)/392; 139 edges removed at ZERO accuracy cost (ep2 BEST 0.550 == the
  full-open row), 179 at −0.033.
- **Survivor structure (heatmap_ep4, hard P>0.5):** R4+R5 (aggregation) kept at
  layers 0–17, pruned ENTIRELY at 18–27 — sharp cutoff, consistent across all five
  Δ-buckets of both families; R6 kept 28/28 layers, R7 26/28 (readout edges needed
  everywhere). The cutoff (~17-18) sits at the hand-identified message locus
  (READ_LAYER=16). Rarest buckets (Δ9-16) prune hardest.
- Relation to the hand design: PARTIAL rediscovery — agreement on the 12–17
  aggregation band + tail-reads-summaries; disagreement below 12 (necessity extends
  down — the fence's ≥12 rule protected carrier supply, moot for always-visible
  replicas) and above 17 (hand leaves open, pruning kills). All statements are
  relative to the v2 readout (trained under full openness) — say so in the writeup.
- **132671 launched**: pruned-mask (gates_best @ep2) transfer on full splits
  8/16/32/64 × {s2open, hand, init, nofence, gates} — the acc-vs-N money figure
  (roots file gained seq_len_64_test=50).

## 2026-08-15 — TRANSFER RESULT + FIGURES: the campaign has its closing package
- 132671 (em; 50/split): learned(pruned) **0.820 / 0.580 / 0.260 / 0.040** ==
  full-open EXACTLY at the training lengths (8/16 — identical MAE too) and == hand
  fence at zero-shot 32/64; fence-family > nofence/init everywhere; nofence MAE
  10.4 @64. Honest caveat recorded: the pruned upper-layer aggregation edges may
  carry a little value at 32+ (pruned < s2open by 0.04 there) — necessity was
  measured at the training lengths.
- **Figures rendered** (dataviz-skill procedure, eyeballed + fixed label collisions):
  outputs/learnmask/figures/acc_vs_n.{png,pdf} (headline: learned dashed line rides
  full-open at 8/16, matches hand at 32/64) and survivor_heatmap.{png,pdf}
  (aggregation kept 0-17 / pruned 18-27, hand design's >=12 opening outlined;
  Δ17+ rows dropped as structurally empty). scripts/learnmask/plot_results.py.
- Campaign deliverables now in hand: (1) frozen-scaffold oracle negative (S0:
  masks memorize, never generalize), (2) coordination barrier (opening cannot be
  discovered from closed init), (3) necessity structure via prune-from-open
  (aggregation 0-17, readout everywhere, cutoff at the READ_LAYER locus),
  (4) length-transfer ordering (fence-family >> no-fence, growing with N).
  Remaining optional arms: prune-arm estimator robustness (E1/E2), S3, carrier-
  scaffold confirmation, nofence-readout reviewer ablation. RESULTS.md entries
  await Tal's "log this".

## 2026-08-15 — professor's S0 variant: LAYER-SHARED mask (one mask for all 28 layers)
- Peer feedback on the shared s0 script: identical masking scheme at every layer —
  "natural design choice, /28 trainable params". Implemented as
  FreeTableGates(share_layers=True) (--s0-share-layers / S0_SHARE env; also
  --share-layers in the peer-share script so it's reproducible from the commented
  file). One logit per cell (239,232 total), ONE Gumbel draw per cell per step,
  identical hard mask at every depth; 18 tests green.
- Note for the discussion (verify, don't assume): our prune result found the
  learned solution IS depth-dependent (aggregation 0-17, pruned 18-27), so layer-
  sharing is a real constraint — but for the frozen-scaffold oracle question it is
  a clean, better-conditioned test (28x fewer params, 28x gradient accumulation
  per logit, far less memorization capacity).
- **Launched: 132926** — layer-shared S0, definitive existence config (frozen
  replica scaffold, seq8 150/50, lam=0, lr 5e-2, st-gumbel, 20 ep, rtx6k 12h_4g).
  Expected ~50 min.

- S0 jobs 131381/131382 FAILED fast (4 min) on their own guard: seq8 layouts are NOT
  all identical — 2 distinct layouts across 150 samples (question token length varies
  with name/room; the earlier 4-sample census was lucky). 131383 cancelled pre-fail.
  Fix: S0 keeps the DOMINANT layout and reports the drop + filtered class dists
  (fixed-layout diagnostic by definition). Resubmitted: **131384** ste @l40s-shared ·
  **131385** soft @rtx6k-shared · **131386** st-gumbel @l40s-public. Wave 1 unaffected
  (relation gates are layout-independent). Note train class-dist at limit 150/200 =
  0.50 majority-0 (stratified round-robin exhausts non-K0 at 75) — reported per split.

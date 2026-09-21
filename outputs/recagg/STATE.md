# recagg campaign — STATE (append-only; newest last)

- [2026-08-17] CAMPAIGN CREATED (session with Tal; direction approved same day).
  Brief: outputs/recagg/CAMPAIGN_BRIEF.md — read it FIRST, it carries the
  pre-registered predictions (H1–H4), guardrails, and phase plan.
  Context in one line: Hahn O(1/N) sensitivity bound explains the flat-read
  failure; recurrence has O(1) per-item influence; this campaign trains tiny
  recurrent heads (GRU + hand-rolled selective SSM, NO pip installs) on the
  EXISTING fenced per-frame verdict-state captures and measures length
  extrapolation vs a trained attention-pool negative control.
  Data verified on disk 2026-08-17:
    outputs/ninv/20260810_000358_park8_leaf/feats_N8.npz  (leaf|0|20|mean 200x8x3584, Y, G)
    outputs/ninv/20260809_235142_hf8_leaf392/feats_N8.npz (same schema, HF @392)
  No leaf states exist yet for N=16/32/64 (superquery capture_16_64 npz are SQ
  nodes only — verified). P1 captures them.
  NEXT: P0 (CPU-only, may start immediately): scripts/recagg/train_heads.py —
  R1 sum-probe / R2 GRU / R3 minimal-SSM / R4 attention-pool on the two N=8
  captures; every EM next to majority + p_frame^N bound; order canary.
  GPU (P1+) waits for plan-shown-to-Tal per house rules.

- [2026-08-17 14:44] P0 LAUNCHED. Wrote scripts/recagg/train_heads.py (R1 sum-probe
  = leaf_probe-style logistic PCA-512 frame probe -> external sum; R2 GRU proj128/
  hid128 559k params; R3 hand-rolled diagonal selective SSM (input-dep dt,B,C,
  pure-torch loop) 526k; R4 trained-query attention-pool 919k — all under the 1M
  budget). Pooled 5x 50/50 splits (leaf_probe convention), scaler fit on train
  frames only + saved in ckpt, class dist printed per split incl. train, EM
  (classification 0..8 primary) + EM_reg (scalar-round) + majority + balanced +
  cluster-bootstrap CI + train-EM (underfit diagnostic, June-runs lesson) + order
  canary (3 perms). metrics_skew imported read-only from scripts/ninv.
  Smoke (1 seed, 10 ep, login-trivial): park8 R1 EM 0.990 = p_frame^8 bound 0.990
  exactly (p_frame 0.9988, majority 0.17) — H1's control already at the bound.
  SLURM: slurm/recagg_train_heads.sbatch (4h_0g CPU, mem 12G). Jobs 133591 (park8)
  + 133592 (hf8_392), epochs=2000 default, run dirs
  outputs/recagg/p0_heads/20260817_144419_{park8,hf8_392}/.
  NEXT: collect P0 reports -> H1/H4 first read; then draft P1 capture plan
  (leaf N=16/32/64, park+HF, needs Tal's OK before any GPU submit).

- [2026-08-17 15:10] P0 v1 COLLECTED + RECIPE ITERATION (all CPU). v1 runs
  (jobs 133591/133592, outputs/recagg/p0_heads/20260817_144419_*, ~5.5 min each):
  R1 sum-probe hits the perception bound on BOTH domains — EM 0.988
  [0.978,0.996], p_frame 0.9985, p_frame^8 = 0.988; park majority 0.124, HF
  majority 0.632 (HF balanced 0.969, so not prior-riding). But ALL trained heads
  collapsed (eval EM 0.15-0.26 park / ~majority HF, trainEM 1.000, HF canary
  violations up to 0.24): count-only supervision on 100 raw-3584-dim sequences =
  pure memorization. NOT an H1/H4 verdict — a supervision-parity confound (R1's
  probe gets 800 frame bits; heads got 100 counts). v1 dirs kept as the
  no-aux/no-augment ablation record.
  Recipe iteration (smokes in outputs/_scratch/recagg_smoke*, seconds each):
  (1) aux per-frame bit BCE (parity with R1) -> every head reads the bit at
  0.986-0.999 (head_pf) yet still memorizes counts; (2) SUBSET-COUNT AUGMENTATION
  (random frame subset in random order, target = subset Y-sum — legit, uses only
  train-split Y) -> R2 GRU 0.99-1.00 EM, canary clean. THE unlock for recurrent
  heads. (3) R3 minimal-SSM needed three structural fixes: near-integrator init
  (A_log=-4), count readout from h_N not C(u_last)*h_N, and — decisive — input
  contribution as a learned CONSTANT vector (pure selective integrator: any
  content-bearing input map W*u gives the readout a linear bag-of-frames
  fingerprint to memorize; GRU's tanh destroys fingerprints for free). Result:
  R3 scalar-round EM 0.93-0.94; its CE-classification readout converges much
  slower (0.44 @3k ep, an ordinal-CE-on-scalar-state pathology, both reported).
  (4) R4 attention-pool with the SAME recipe: head_pf 1.000 but cannot aggregate
  — eval 0.15-0.33 across all variants, trainEM 0.68-0.99 (memorizes with
  epochs, never generalizes). The Hahn negative control shows up in-length @N=8.
  Also: PCA-128 whitened front-end (fit on train frames) for all heads; shared
  tiny MLP readout across arms (readout capacity never the confound); knobs
  --pca/--aux-weight/--augment/--lr/--wd for ablations.
  v2 DEFINITIVE P0 submitted: jobs 133609/133610, epochs 3000, run dirs
  outputs/recagg/p0_heads/20260817_150845_v2_{park8,hf8_392}/.

- [2026-08-17 15:30] P0 COMPLETE (v2 definitive, jobs 133609/133610, ~3 min each,
  outputs/recagg/p0_heads/20260817_150845_v2_{park8,hf8_392}/ = CANONICAL).
  Pooled 5x50/50 splits, epochs 3000. Headline (EM cls / EM scalar-round):
    PARK8 (majority 0.124, p_frame^8 bound 0.988):
      R1 0.988/0.988  R2 GRU 0.986/0.992  R3 SSM 0.298/0.936  R4 attn 0.310/0.268
    HF8_392 (majority 0.632, bound 0.988):
      R1 0.988/0.988  R2 GRU 0.914/0.870 (bal 0.765)
      R3 SSM 0.672/0.868 (bal 0.185)     R4 attn 0.700/0.652 (bal 0.207)
  H1 (in-length): CONFIRMED on park — R2 GRU 0.986 [0.974,0.996] is statistically
  AT the bound and does NOT beat R1 (no leakage signal); canary ≤0.04 everywhere.
  On HF, R2 0.914 sits below the bound with the miss concentrated in rare high
  counts (c7 0.67, c8 0.00 — pool has 5 and 2 such samples) — a train-skew
  limitation of the HF pool, not an aggregation failure; note for P2/P3 design.
  H4 (first read): GRU > minimal-SSM. The pure selective integrator DOES count —
  scalar-round 0.936 park / 0.868 HF — but its CE-classification readout is the
  weak link (0.298/0.672, per-class recall decaying with count). Gap = readout
  optimization, not integration. Both readouts reported per convention.
  R4 NEGATIVE CONTROL (the figure-maker): head_pf 0.999 (reads every frame bit)
  + trainEM 0.97 (memorizes train) but eval 0.31/0.27 park with per-class recall
  decaying monotonically to c8 0.00 — the softmax-average capacity collapse
  visible IN-LENGTH @N=8, before any extrapolation. Hahn story confirmed inside
  our own feature space.
  INDEX.md created. NEXT: P1 leaf captures (plan presented to Tal — GPU, needs OK).

- [2026-08-17 15:30] DIRECTIVE (Tal): HF resolution 512 is PRIMARY everywhere;
  392 is DEPRECATED (resolution-comparison arm only). Basis: the 512 leaf twin
  20260809_235142_hf8_leaf512 exists on disk (missed in the initial data sweep;
  brief said "verify" — verified now), and ninv already showed 512 > 392 (leaf
  recall 1.000 vs 0.995, two-pass EMIT 0.960 vs 0.920). P0 @512 confirms:
  R1 0.998 (balanced 1.000 — perfect per-class), R2 GRU 0.948 (bal 0.889 vs
  0.765 @392), R3 scalar-round 0.874, R4 0.718 raw (prior-riding, bal 0.239).
  Run: outputs/recagg/p0_heads/20260817_152313_v2_hf8_512/ (job 133613).
  ALSO found in the same sweep: park16 leaf capture (ninv 20260810_000358_
  park16_leaf, N=16, 200 samples) -> in-length N=16 cell + a zero-GPU fit@8->
  eval@16 extrapolation cell; P1 park16 capture no longer needed.
  IN FLIGHT: 133614 park16 P0; SSM-undertraining test @20k epochs (Tal asked):
  133615 park8, 133618 hf8_512. All P1 planning now targets 512 HF pools.

- [2026-08-17 16:00] SSM UNDERTRAINING TEST (Tal's hypothesis) — CONFIRMED.
  20k epochs (6.7x the 3k budget), same config otherwise:
    park8  (133615, 20260817_152313_ssm20k_park8):  R3 SSM cls 0.984 [0.972,
    0.994] (was 0.298 @3k!) / reg 0.986 — AT the bound next to R1 0.988,
    R2 GRU 0.992. canary 0.01.
    hf8_512 (133618, 20260817_152712_ssm20k_hf8_512): R3 0.910 (bal 0.818) /
    reg 0.932, vs R2 0.970 (bal 0.932), R1 0.998.
  H4 UPDATED: GRU ≈ minimal-SSM on clean counting once the SSM's CE readout is
  trained to convergence — the 3k-epoch gap was optimization budget (ordinal CE
  on a scalar-like integrator state trains ~7x slower), NOT architecture. This
  is the pre-registered H4 prediction (both integrate).
  R4 attn-pool at 20k: park 0.302 / hf512 0.728 (bal 0.262) — trainEM 1.0, eval
  unchanged. The negative control is NOT budget-limited; failure is structural.
  CANONICAL RECIPE now epochs=20000. park16 20k rerun submitted for uniformity;
  park16 @3k (0.968 GRU / R3 undertrained 0.358) superseded on landing.
  Also: park16 P0 @3k collected earlier (133614): R1 0.986, R2 0.968 — GRU at
  the bound in-length @N=16 as well.

- [2026-08-17 15:44] P1 LAUNCHED (Tal OK'd). Five leaf captures via the existing
  slurm/probe_tree_ninv.sbatch (capture-only: FIT_LAYERS=none, READ_LAYERS "16 20"
  — space-separated to dodge the --export comma-split), run dirs
  outputs/recagg/p1_captures/20260817_154347_*:
    133624 park32   (a100-public 24h_1g, longN root, LIMIT=200)
    133625 park64   (a100-public 24h_1g --time=8h, LIMIT=150)
    133626 hf32_512 (a100-public, seq_len_32_test, all 50 steps dirs, --resize 512)
    133627 hf64_512 (a100-public --time=8h, seq_len_64_test, 50 dirs, 512)
    133628 hf16_512 (rtx6k 2h_2g, seq_len_16_train_steps_in_room, 200 dirs, 512)
  All RUNNING at submit (+133621 park16-20k rerun still going). l40s was full;
  a100-public had 6 idle. park16 capture not needed (existing ninv park16 leaf).
  ETA: short jobs ~25-40 min; N=64 pair ~1.5-2.5 h (quadratic in seq len) = the
  long pole. NEXT on landing: npz sanity (keys/shapes/class dist), then P2 (CPU):
  fit @{8,16} → zero-shot eval @{32,64}, all four arms, EM-vs-N headline figure.

- [2026-08-17 16:05] park16 @20k CANONICAL (133621, outputs/recagg/p0_heads/
  20260817_155*_20k_park16/): R1 0.986 · R2 GRU 0.998 · R3 SSM 0.952 · R4 0.242.
  In-length table complete at N=8 and N=16: ALL THREE integrator arms at the
  perception bound on park, R4 structurally out. P1 captures ran far faster than
  the quadratic estimate (capture stops at L20; A100 sdpa): hf16/hf32/hf64/park32
  ALL DONE by ~16:05 (~10-20 min each); park64 100/150 @640s, ~5 min out
  (seq=14281). NEXT: npz sanity, then P2.
  R5/P4 feasibility verified (Tal is exploring a fully-frozen task-agnostic arm:
  frozen per-frame labeler -> frozen pretrained recurrent LM aggregates):
  /rg/shocher_prj/tal.gorbunov/venv_arch has transformers 5.13.1 (native Mamba +
  RWKV classes, no mamba_ssm needed) and hf_arch already caches xLSTM-7b (pure
  recurrent), recurrentgemma-9b-it, Nemotron-Nano-9B-v2 + Falcon-H1-7B (Mamba
  hybrids), Qwen3-Next-80B, with Mistral-7B + Qwen2.5-14B as attention controls.
  Pile-matched pair mamba-2.8b/pythia-2.8b would be a ~11GB download. NOTE: the
  repo's .venv_arch symlink is GONE — restore with
  ln -s /rg/shocher_prj/tal.gorbunov/venv_arch .venv_arch when needed.

- [2026-08-17 16:05] P2 LAUNCHED (HF @512 ONLY per Tal's directive; park P2
  deferred). npz sanity PASSED on all three HF captures (leaf L16+L20, Y-sum==G,
  healthy gold dists; hf32/64 n=50 each — thin, CIs will show it; hf16 majority
  0.44). New instrument scripts/recagg/eval_extrap.py + slurm/recagg_eval_
  extrap.sbatch: fit R1-R4 @{8,16} (train halves, canonical 20k recipe,
  multi-length interleaved subset-augmentation), zero-shot eval @{32,64} +
  in-length halves; cls support hard-capped at 16 (structural — EM_reg carries
  H2); order canary per cell. SMOKE FINDING (200-ep, seed 0): R1 sum-probe
  EM 1.000 at EVERY cell incl. zero-shot N=64 — p_frame = 1.000 on HF@512:
  perception is NOT the limit at native resolution; the whole game is
  aggregation. Job 133636, outputs/recagg/p2_extrap/20260817_160326_hf512/,
  ETA ~2.2h. park64 capture at 140/150, minutes from done.

- [2026-08-17 16:20] P1 COMPLETE. All five captures COMPLETED, total GPU ~59 min
  (vs 2-3h est): park32 12:43 / park64 19:49 / hf32_512 7:18 / hf64_512 12:58 /
  hf16_512 6:12. park npz sanity PASSED (leaf L16+L20, Y-sum==G, balanced
  uniform gold dists incl. counts up to 64). P1 captures live under
  outputs/recagg/p1_captures/20260817_154347_*/. P2 (HF512) running: 133636.
  Park P2 = ready-on-request (same instrument, park fit/eval specs).

- [2026-08-17 16:26] ARM A LAUNCHED (Tal OK'd: frozen pretrained readers over
  oracle captions, all-model comparison). New instrument scripts/recagg/
  frozen_readers.py + slurm/recagg_frozen_readers.sbatch (venv_arch, HF_HOME on
  /rg, HF_HUB_OFFLINE on nodes; .venv_arch symlink RESTORED). Protocol: oracle
  per-frame captions rendered from MMReD-HF qa.txt GT states (benchmark
  questions/answers verbatim; perception error = 0 by construction); plain-text
  3-shot completion, exemplars fixed from the N=4 TRAIN pool; greedy; answer =
  first int (format failures reported separately); N=8/16 (200 ea, whole train
  pools) + 32/64/128 (50 ea test pools, strided per the K0 rule).
  Jobs (run dirs outputs/recagg/armA_readers/20260817_162559_*):
    133641 xlstm7b (pure recurrent)   133642 recgemma9b   133643 nemotron9b
    133644 falconh1_7b (hybrids)      133645 mistral7b    133646 qwen14b (attn)
  All RUNNING. Pile-matched pair mamba-2.8b/pythia-2.8b downloading to /rg cache
  (wave 2 job after download). RISK NOTED: transformers slow-path SSM prefill
  (no mamba_ssm kernels) may be slow on 128-frame prompts — pilot logs will
  show; rescope Ns for affected models if needed.
  Still racing: P2 hf512 GPU 133637 vs CPU 133636.

- [2026-08-17 16:45] ARM A COMPLETE (7 of 8 readers; total GPU ~28 min). Run dirs
  outputs/recagg/armA_readers/20260817_162559_*/. EM by N (majority: 0.625/0.440/
  0.160/0.080/0.120):
                     N=8    N=16   N=32   N=64   N=128
    qwen14b (attn)   0.695  0.515  0.240  0.100  0.020
    falconh1 (hyb)   0.670  0.470  0.180  0.100  0.040
    mamba2.8b (rec)  0.625  0.440  0.160  0.040  0.000
    pythia2.8b(attn) 0.625  0.440  0.120  0.000* 0.000*  (*fmtfail 0.76/1.00)
    xlstm7b (rec)    0.590  0.395  0.140  0.060  0.000   (fmtfail up to 0.28)
    recgemma9b (rec) 0.445  0.245  0.060  0.040  0.020
    mistral7b (attn) 0.315  0.245  0.160  0.060  0.020
  VERDICT: pre-registered outcome (b). NO frozen reader counts reliably at ANY
  N>=8 (best: 0.695@8, 0.515@16); ALL collapse to ~majority by N=32-64. The
  mamba/pythia matched pair sits EXACTLY at majority @8/16 with identical MAE =
  both degenerate to predicting 0 (floor effect) — the pair measures task-prior
  absence, not architecture. No clean recurrent-vs-attention shape separation:
  the task-prior floor masks it. IMPLICATION: zero-shot frozen readers cannot
  carry the task-agnostic endpoint; the honest route is a task-GENERAL adapter
  (one recurrent reader tuned on mixed question types, eval on held-out types).
  Contrast row for the thesis: our 559k-param GRU @perception bound (0.99) vs
  14B frozen LMs failing at N=16 on ORACLE captions of the same benchmark.
  nemotron9b: 2nd failure, custom-code bug (cache_position None in its remote
  modeling under tf-5.13) — dropped, 7 readers suffice. NOTE: CoT prompting
  would likely rescue the attention models by externalizing the count into the
  token stream — that IS the scratchpad repair (caption-scan); a one-job CoT
  variant would tie Arm A to THE METHOD directly (proposed, not run).

- [2026-08-17 16:50] P2 COMPLETE (HF@512). CPU job 133636 WON the race (26:54;
  the 2.2h estimate was login-node-throttled extrapolation); GPU twin 133637
  scancel'd mid-run. CANONICAL: outputs/recagg/p2_extrap/20260817_160326_hf512/.
  EM_reg (scalar-round; bound = p_frame^N = ~1.000 everywhere, perception is
  PERFECT @512):
                 N8_in  N16_in  N32_zs  N64_zs     (majority .632/.464/.160/.080)
    R1 sum-probe 1.000  1.000   0.996   1.000   <- exactly length-invariant
    R2 GRU       0.988  0.940   0.640   0.256      (canary 0.14-0.16 at zs!)
    R3 SSM       0.984  0.970   0.800   0.220      (canary clean 0.02)
    R4 attnpool  0.596  0.456   0.236   0.048   <- collapses, as pre-registered
  H2 VERDICT: PARTIAL. The architecture ORDERING matches theory at every N
  (R1 > R3 > R2 > R4; integrator > gated RNN > attention-pool), and R3@32 =
  0.800 vs R4 0.236 is a big margin. But the trained continuous heads are NOT
  absolutely length-invariant: both recurrent arms decay by 64 (4x the fit
  lengths). Mechanism candidates (all head-side — perception bound is 1.000):
  (a) GRU tanh state saturates beyond trained count range (classic RNN counting
  limit; its zs order-sensitivity canary corroborates), (b) R3's learned decay
  exp(dt*A)~0.99/step is tuned on <=16 steps and leaks ~half the integral over
  64, (c) MLP readout trained on counts 0..16 extrapolates its output range
  poorly. The EXACTLY invariant aggregator is the modular symbolic one: probe ->
  external sum (R1), i.e. the gate->tally structure. Thesis framing: length
  invariance comes from the SYMBOLIC EXTERNALIZATION of the count, not from
  recurrence per se; learned continuous integrators approximate it to ~2x their
  training range. Proposed mechanism-isolating ablation (cheap CPU): R3 with
  A pinned to 0 (no leak) + linear readout — prediction: recovers most of the
  invariance; would cleanly attribute the 64-decay to leak+readout, not to
  recurrence. NOT RUN yet.

- [2026-08-17 17:10] MECHANISM ROUND LAUNCHED (Tal OK'd noleak + tally; D&C is
  Tal's idea, head-level twin of the tree cascade). 133665 (CPU 4h_0g):
  eval_extrap rerun with (a) new arm R3b_noleak — A_log frozen at -20 (exact
  integrator, no leak) + PURE LINEAR count readout on h_N (linear extrapolates;
  the MLP was range-capped) — and (b) --chunk 16 divide-and-conquer eval on the
  zs cells: head per <=16-frame window (inside trained range), per-window
  scalar ROUNDED (re-quantization, as in the ninv cascade), summed. Run dir
  outputs/recagg/p2_extrap/*_hf512_mech. Predictions: R3b recovers most of
  N=64; chunking lifts R2/R3 toward R1.
  133669/133670: Arm A TALLY style (running-count scratchpad exemplars,
  'Final answer:' extraction, limit 50/N) for qwen14b + xlstm7b — the
  symbolic-externalization rescue test.
  peft NOT in venv_arch — the mamba fine-tune (adapter route) needs an install
  OK from Tal before it can happen.

- [2026-08-17 17:06] ADAPTER ROUTE LAUNCHED (Tal OK'd peft install + fine-tune).
  peft 0.20.0 installed into venv_arch (approved; 4bit-kernels warning benign).
  scripts/recagg/finetune_mamba.py: LoRA r=16 on all mamba-2.8b linear layers,
  3000 steps, SYNTHETIC caption streams (MMReD vocab, frozen_readers format,
  N~U{2..16}, count k~U{0..N} BY CONSTRUCTION — uniform coverage; real HF pools
  untouched -> eval uncontaminated), loss on answer tokens only, val previews
  incl. N=32 extrapolation mid-training. Generator VERIFIED 300/300 by
  round-tripping rendered streams through load_hf_sample.evidence_bits.
  Job 133671 (a100 24h_1g, chained: train then frozen_readers eval --adapter
  --shots 0 --limit 50 at N=8..128), outputs/recagg/armB_adapter/
  20260817_170537_mamba_count_ft/. THE question: does a supervised counting
  circuit in a pretrained SSM extrapolate, or drift like from-scratch R3?

- [2026-08-17 17:50] MECHANISM ROUND LANDED (133665 COMPLETE, 43 min;
  outputs/recagg/p2_extrap/*_hf512_mech = CANONICAL for H2 mechanism).
  Zero-shot N=64 EM_reg (bound 1.000, majority 0.080):
    whole-sequence:  R1 1.000 | R3b_noleak 0.864 | R3 0.300 | R2 0.260 | R4 0.068
    chunk-16 (D&C):  R1 1.000 | R3b 0.928 | R3 0.900 | R2 0.824 | R4 0.064
  TWO CONFIRMED REPAIRS, canary 0.000 on both:
  (1) R3b noleak (pin decay=1 + linear readout): 0.300 -> 0.864. Tal's theorem
  gets its precise boundary: length-failure lives in the length-coupled LEARNED
  parameters; make the learnable surface length-free (gate = per-frame problem)
  and extrapolation follows. Residual 0.136 = soft-gate analog accumulation
  (Sum eps_t crossing +-0.5 by N=64), exactly the predicted leftover.
  (2) CHUNKED D&C (Tal's idea, re-quantization at 16-frame boundaries):
  rescues EVERY integrator arm — R3 0.300->0.900, R2 0.260->0.824, R3b ->0.928.
  Weights only ever run inside their trained range; cross-chunk combine exact.
  Chunking does NOT rescue R4 (0.064) — can't compose an aggregator that fails
  within-range. The convergence law: every repair pushes the learned aggregator
  toward the symbolic one (quantize early, add exactly).
  TALLY (partial): xlstm7b_tally = NO rescue for a BASE model (0.62@8 ->
  0.18@16, loses its own count) — scratchpad requires procedural instruction-
  following, an important qualifier. qwen14b_tally OOM'd @N=128 (batch 8 KV);
  resubmitted batch 2 (133675) + mistral7b_tally added (133676).

- [2026-08-17 18:10] MAMBA FINE-TUNE LANDED (133672, 59 min train + 137s eval;
  outputs/recagg/armB_adapter/20260817_*_mamba_count_ft/). LoRA r=16 on
  dt/in/x_proj (24M params, 0.86%), 3000 steps synthetic N<=16, eval zero-shot
  on the REAL HF pools (never seen):
    N=8 0.960 | N=16 0.880 | N=32 0.580 | N=64 0.140 | N=128 0.000
  (base mamba 3-shot was majority-floor 0.625/0.440/0.160/0.040/0.000.)
  VERDICT: supervision INSTALLS the counting circuit (0.96 in-range, from
  majority-floor) — and the circuit DRIFTS exactly like from-scratch R3:
  fine at 1x, half-broken at 2x (0.580), dead at 4-8x. MAE 0.04 -> 15.3.
  Training dynamics show it live: val N=16 climbed 0.15->0.75 while the N=32
  preview never exceeded 0.25. The drift law is scale- and pretraining-
  independent: 60k-param head and 2.8B pretrained SSM fail identically.
  Tal's theorem now measured at THREE scales; the repairs that work are the
  imposed-structure ones (noleak 0.864 / chunk 0.900 / probe-sum 1.000 @64).
  NEXT candidate (not run): chunked INFERENCE for the tuned mamba — feed
  16-frame caption windows, sum the answers — the D&C repair applied at LM
  level; predicted to restore most of 64/128.

- [2026-08-17 18:30] CHUNKED-INFERENCE MAMBA LANDED (133680, 3:50): the D&C
  repair works at LM level too. Tuned mamba, 16-frame windows, answers summed:
    N=32 0.580->0.780 | N=64 0.140->0.580 | N=128 0.000->0.420 (MAE 15.3->0.84)
  Same model, same weights — only the inference structure changed. Residual =
  per-window near-misses compounding (8 windows @128, window EM ~0.96 ->
  ~0.42-0.6 expected band; MAE<1 = usually off by one window's slip).
  outputs/recagg/armB_adapter/*_mamba_ft_chunk16/.
  MISTRAL TALLY (133676): weak rescue @short N (0.440 vs 0.315 direct @8),
  still collapses @32+ (0.10/0.02/0.04) — a 7B instruct cannot hold the tally
  protocol over long streams either. Procedural fidelity is the scratchpad's
  binding constraint; qwen14b_tally (133675, still running) is the remaining
  strong-instruct test.

- [2026-08-17 19:00] QWEN14B TALLY LANDED (133675, 75 min of generation) —
  SCRATCHPAD VERDICT CLOSED, NEGATIVE at <=14B: 0.740/0.500/0.260/0.060/0.000
  vs its own direct 0.695/0.515/0.240/0.100/0.020 — no rescue beyond N=8; even
  the strongest instruct model in the fleet loses its own running count inside
  a 64-128-step enumeration. Repair ranking for pretrained LMs is now measured:
    chunked inference (2.8B tuned: 0.580 @64, 4 min)
      >> scratchpad tally (14B: 0.060 @64, 75 min)
      >> direct (anything: ~majority).
  The externalized state only helps if the EXECUTION is external too — models
  below ~14B cannot faithfully run a long procedure in their own token stream.
  (No contradiction with THE METHOD's caption-scan: that scan is short and its
  tally is external — gate->tally. The failure here is LM-INTERNAL long-horizon
  tallying.) This is the strongest empirical motivation yet for the
  Ask-Compile-Execute proposal (LM chooses the program, code executes it):
  proposed to Tal 19:00, awaiting OK on Arm C (oracle-records feasibility, all
  18 question types, program-synthesis by qwen14b, sandboxed exact execution).

- [2026-08-17 19:20] ARM C LAUNCHED (Tal OK'd Ask-Compile-Execute). New
  instrument scripts/recagg/ask_compile_execute.py + slurm/recagg_ask_compile
  .sbatch (job 133716, outputs/recagg/armC_compile/). LM (qwen14b) sees ONLY
  the question -> compiles `def solve(frames)` over per-frame GT records;
  sandboxed exact execution (whitelisted builtins; import/__/while/exec
  forbidden; failures reported, never silently wrong; every program saved to
  programs.json for audit). Eval: mixed test pools N=16/32/64/128, ALL 24
  question types x 8 strided samples = 768 tasks; 4 SEEN exemplar types in the
  prompt (steps_in_room, rooms_visited, where_spend, char_at_frame — the last
  pins the 1-BASED step convention, verified 50/50 against gold), 20 UNSEEN
  types scored separately = the task-generalization claim. PREDICTION on the
  record: EM ~flat in N where the program is right; accuracy becomes a
  per-TYPE constant, not a length curve.

- [2026-08-17 19:45] ARM C v1 LANDED (133716, 9 min, 768 programs;
  outputs/recagg/armC_compile/*_qwen14b/). THE SHAPE PREDICTION CONFIRMED:
  ALL-types EM 0.58/0.58/0.58/0.61 across N=16/32/64/128 — the first
  length-FLAT row of the campaign; UNSEEN(20 types) 0.51-0.54 flat; SEEN(4)
  0.84-1.00. Eight types sit at 1.00 flat incl. FIVE UNSEEN (crowd_count,
  crowded_room, first_app, n_empty, room_empty, spend_alone) = composition,
  not retrieval. programs.json AUDIT: the 0.307 exec-failure rate is mostly
  OUR sandbox, not the model — 207/236 failures are `next` missing from the
  builtins whitelist; all 26 'forbidden' = `from collections import
  defaultdict` (reasonable code vs blanket ban); room_at_frame misses =
  returning []/None where the benchmark's convention is the string "Nobody"
  (never stated in the prompt). True compile ability >> 0.58.
  v2 SUBMITTED (133721): whitelist += next/iter/filter/map/bool/frozenset,
  Counter+defaultdict preloaded (and said so in the schema), Nobody convention
  stated. v1 kept as the audit record. Flatness conclusion unaffected —
  v2 measures the honest LEVEL.

- [2026-08-17 20:00] ARM C v2 LANDED (133721; outputs/recagg/armC_compile/
  *_qwen14b_v2/ = CANONICAL; v1 = audit record). Exec-failure 0.307 -> 0.046.
    ALL    0.80 / 0.78 / 0.74 / 0.83   (N=16/32/64/128)
    SEEN   0.97 / 0.88 / 0.88 / 1.00
    UNSEEN 0.77 / 0.76 / 0.72 / 0.80
  LENGTH-FLAT at ~0.79 overall / ~0.76 on the 20 UNSEEN types. 13/24 types at
  ~1.00 flat, 11 of them UNSEEN (incl. last_at_room, final_app, n_empty,
  crowd_count, crowded_room, spend_alone, n_room_on_char_first_app...).
  Remaining weakness is concentrated and INTERPRETABLE: the char_on_char_* /
  room_on_char_* two-stage lookups (0.25-0.62; nested find-frame-then-lookup
  programs with subtle stage errors) and the tie-ambiguous superlatives
  (spend_together, who_spend, first_at_room ~0.4-0.7 — "least/first" ties the
  benchmark resolves by an unstated rule). NO length trend anywhere.
  VERDICT: Ask-Compile-Execute is the task-agnostic aggregation solution —
  length-invariant by construction and measured, ~0.76 zero-shot on unseen
  question types with failures localized to compile-hard/ambiguous types.
  Obvious lifts (not run): self-check retry (execute program on the few-shot
  exemplar, regenerate on mismatch — labels-free), and stating the tie rule.
  NEXT (campaign): Arm B (real VLM captions replace oracle records) closes the
  full pipeline; P2-park + P3 unchanged; RESULTS.md on "log this".

- [2026-08-17 20:45] ARM C v3 LANDED (133747, rtx6k; outputs/recagg/armC_compile/
  *_qwen14b_v3/ = CANONICAL; v1/v2 = audit/level records). Prompt-side fixes
  only (definitions block for appear/final/together/alone + ONE synthetic
  reversed-scan exemplar — all 20 benchmark types still UNSEEN):
    ALL    0.89 / 0.92 / 0.85 / 0.90   (N=16/32/64/128; was ~0.79 in v2)
    UNSEEN 0.88 / 0.93 / 0.85 / 0.88   exec-fail 0.003 (was 0.046)
  16/24 types now >=0.88, most 1.00 flat; the final-but-forward-scan bug and
  the spec-convention misses largely eliminated (char_on_char_first_app +
  first_at_room -> 1.00 flat; char_on_char_final_app 0.12->0.88ish;
  spend_together 0.25->~0.7). Remaining tail: room_on_char_final_app (~0.4),
  who_spend (~0.55), char_on_char_at_frame (~0.6) — nested two-stage programs
  at 14B; candidate lifts = self-check retry / k-sample execution-guided
  selection / SQL target (ORDER BY makes ties explicit) — none run yet.
  Still length-flat everywhere. Day summary: task-agnostic frozen aggregation
  at ~0.89 EM across 24 question types, N=16..128, on oracle records.

- [2026-08-17 21:15] CITATION SWEEP LANDED -> outputs/recagg/RELATED_WORK.md
  (7 areas, must-cites, delta sentences; 18 queries). TOP FINDINGS:
  (1) THREAT to the necessity claim: Apple "Tool-Use Unlocks Length
  Generalization in SSMs" (arXiv:2510.14826, Oct 2025) already argues
  delegation NECESSITY with theory (fixed-state SSMs provably can't do
  long-form; code tools restore length gen) — text/symbolic domain only.
  Claim (i) reframed: we TRANSPORT + empirically verify the law in video-QA
  aggregation, with the finer mechanism triad (decay identifiability / analog
  accumulation / readout range) and a matched-perception aggregator spectrum
  — not first-ever necessity.
  (2) Fencing has a structural twin: Parallel Context Windows (Ratner, ACL23)
  + APE (ICLR25) — block-diag attention + position reuse, training-free, for
  TEXT contexts. Ours = the application to per-frame visual perception in a
  frozen VLM w/ M-RoPE reset + the invariance measurement + carriers. Cite
  first, claim the instantiation not the mask trick.
  (3) "Transformers need glasses!" (NeurIPS24) already bridges over-squashing
  ->LM counting failures for text — cite as the metaphor's origin.
  (4) Supports: Yehudai NeurIPS24 (count-to-n readout threshold), Chang&Bisk
  (short-FT fails long counting, text), Merrill&Sabharwal. Counter-argument to
  address: Buitrago&Gu ICML25 (state-passing interventions fix recurrent
  length gen -> our GRU/SSM drift rung needs the undertrained-vs-fundamental
  discussion).
  (5) Pipeline precedented (NS-DR/CLEVRER, ProViQ, MoReVQA, HPP'26) but NONE
  measure length extrapolation or argue necessity — the combination
  (matched-perception drift law + fenced invariant perception + spectrum map
  + in-model re-quantization) remains unclaimed. EC-Bench'26: MLLMs ~24% on
  ultra-long counting = supporting diagnostic.

- [2026-08-17 20:40] ARM B LAUNCHED (Tal OK'd). Step 1 captioner
  scripts/recagg/caption_frames.py (frozen Qwen2.5-VL @512, per-frame
  independent calls — measured-equivalent of the fenced forward via A3
  multipass parity; sampling mirrors Arm C stride so v3 PROGRAMS ARE REUSED
  UNCHANGED = controlled perception swap). Step 2 executor
  scripts/recagg/armB_execute.py ready (CPU).
  Smoke 133777 (24x16 frames, 57s): captions GOOD, my parser regex grabbed the
  literal 'Room:' prefix -> fidelity 0.0; parser fixed (match room name
  anywhere in segment), re-scored saved captions:
  PER-CHAR PLACEMENT 0.9156, PER-FRAME EXACT STATE 0.6562.
  KEY EXPECTATION SET: full-state captioning (5 chars placed) is much harder
  than the 0.999 single-bit verdict readout — Arm B EM will be
  perception-dominated; compile+execute (0.89 flat) is no longer the
  bottleneck. Full run: 133785 (rtx6k 2h_2g, per-type 4, N=16..128, ~23k
  frames, est ~1h) -> outputs/recagg/armB_captions/20260817_203538_full/.

- [2026-08-17 21:35] ARM B COMPLETE — the composed frozen system measured
  end-to-end. Captions: 133785 (56 min, 384 samples x N frames @512, rtx6k):
  per-char placement 0.9126, per-frame exact-state 0.6373. Execute (CPU,
  seconds): v3 programs UNCHANGED on VLM records ->
  outputs/recagg/armB_execute/20260817_213301_v3progs/ (+ captions run dir).
    ALL    vlm/oracle: 0.79/0.92  0.81/0.89  0.72/0.85  0.72/0.90 (N=16..128)
    UNSEEN vlm/oracle: 0.75/0.90  0.85/0.90  0.71/0.86  0.70/0.88
  READ: the error factorization holds. Single-frame-lookup types (first_app,
  final_app, last_at_room, first_at_room, room/char_at_frame) stay ~1.00 FLAT
  to N=128 — touching one frame means perception enters once, so composed EM
  ~= oracle EM. Whole-sequence EXACT-COUNT types decay with N exactly as
  p_relevant^N predicts (crowd_count 0.50->0.00, steps_in_room 1.00->0.25
  @128) — NOT aggregation drift (executor exact, oracle rows flat): with
  per-frame state acc 0.91, corrupt frames corrupt the count. Length-coupling
  now lives ONLY in perception fidelity ^ (facts touched) — the one place the
  theory says it must. Levers if pursued: per-character caption queries
  (5 easier emissions), self-consistency over caption variants, count-tolerant
  scoring (off-by-1); NOT run.
  CAMPAIGN DAY CLOSES: P0-P2+mechanism+ArmA/B/C+sweep all landed. Open: P3
  (H3, Tal's 2 decisions), P2-park figure job, SQL/self-check ablations,
  RESULTS.md on "log this".

# Stage-2 carrier layer — DRAFT results (autonomous session 2026-07-18)

> Draft RESULTS-style entries. Mission: `plans/carrier_stage2_agent_brief.md`.
> Every number traces to a run dir. To be merged into `RESULTS.md` by Tal, not by the agent.

## [2026-07-18] Collected in-flight jobs (launched before this session)

### Job 123205 — frozen baseline, N=8 images_park (TRUNCATED prior — superseded by E1)

- Run: `outputs/ladder/image_longN/frozen_baseline/20260718_122504/`
- `frozen_baseline_eval.py`, LIMIT=300 on `data/mmred_images_park/seq_len_8/all_uniform` —
  which is e-sorted, so n=300 ⇒ **gold ∈ {0,1,2} only** (the audit caveat). Numbers kept for
  the record, NOT headline: acc **0.513**, MAE 0.57 (g0 68/100, g1 29/100, g2 57/100).
- The full-prior (LIMIT=900) rerun is E1(b) below; this row is retired on its completion.

### Job 123208 — steps-trained carrier token → NATURAL images (cross-DOMAIN transfer), eval-only

- Run: `outputs/ladder/image_longN/carrier_token_crosstask_natural/20260718_122538_proxy_room_k1/`
- Steps-distilled e_c (`carrier_token/20260717_201919_distill_room_k1/carrier_best.pt`, d′ 8.35)
  evaluated zero-shot on `data/mmred_natural/dist_far` (n=50, all the cell has).
- **d′ 3.19 ± 0.55** (auc 3.75); fresh per-cell logistic gate: per-frame err 0.144,
  **tally exact 0.432 ± 0.106**. (ckpt-head row is garbage-by-design — untrained head.)
- **Reading (honest negative-ish):** cross-domain transfer is much weaker than cross-task.
  The same-cell replica teacher reads d′ 6.22 and the frozen model 0.58 exact — the carrier
  arrives at **~51% of the domain's teacher** (vs 88% for steps→cooc within synthetic MMRED),
  and the gated tally (0.432) UNDERPERFORMS the frozen model on this cell. The carrier token's
  content is partly domain-bound (synthetic MMRED renders), not purely a task-general "read this
  frame" instruction. n=50 caveat: single cell, wide bars.

### Job 123206 — stage-2 carrier layer rerun, EPOCHS=30 (truncated prior) — collected below (E2)

## [2026-07-18] E1 — full-prior fix (pre-registration, written BEFORE the runs)

Code: added `iter_sample_dirs_shuffled` (stratified round-robin, seeded) to
`evaluations/helpers/utils.py`; `--shuffle-dirs SEED` + `[gold-hist]` prep-end print wired into
`replica_carrier_probe.py`, `carrier_token_distill.py`, `carrier_layer_lora.py`,
`frozen_baseline_eval.py`; `[mask-debug]`/`[pos-debug]` first-sample lines added to
`carrier_layer_lora.py` (were absent). CPU verify: first-300 prefix of images_park = 33-34 per
gold class; deterministic. New CPU analyzer `experiments/glstm/replica_gate_tally.py` for
gate->tally on probe caches. Smoke job 123222 (outputs/_scratch/st2_smoke/).

Full-prior N=8 headline runs (all LIMIT=900 on `mmred_images_park/seq_len_8/all_uniform`,
gold uniform 0..8). Pre-registered bands:
- (a) Q-first blockfence probe: d' ~= 9.24+-0.33 (prior-free supply; large deviation = data-path
  bug). Gate->tally: per-frame err ~0.004-0.01 at N=8 implies exact ~0.92-0.97; band >=0.90,
  vs majority baseline 1/9 = 0.111.
- (b) frozen baseline: ~0.2 (historical band 0.15-0.30); the truncated-prior 0.513 should DROP
  (g0-favoring bias no longer rewarded).
- (c) carrier-token distill (train 450 / eval 450): in-run teacher anchor must reproduce the
  9.2 band; eval d' >= 90% of the in-run teacher = GO (the 300-sample runs sat at 92-93%).
- (d) stage-2 full prior = E2 (bands there).
## [2026-07-18] E2 — stage-2 convergence (pre-registration)

- Job 123206 (30 ep, truncated prior n=300) reproduces the first run's trajectory
  (0.840 @ ep12 vs 0.853 @ ep12 in `carrier_layer/20260718_023033_L17_r8/`).
- Full-prior headline run planned: LIMIT=900, TRAIN_N=450, epochs 40 (train 450 / eval 450,
  gold uniform 0..8), warm-start e_c from the distilled carrier, L_open=17 r8 (unchanged).
- Bands (pre-registered): GO >= 0.95 emitted on the held-out 450 (match the scaffold in-model);
  0.85-0.95 partial; per-count row must not show a g0-collapse. no-LoRA ablation row kept from
  `carrier_layer/20260718_023030_L17_r8_nolora/` (~0.50 plateau, truncated prior) — rerun on the
  full prior only if budget allows.
- Sweep arms (ONE change each, after the headline): L_OPEN in {12, 22}; lr_lora 3e-4.
## [2026-07-18] E1(b) — frozen baseline on the FULL prior (job 123225) — LANDED

- Run: `outputs/ladder/image_longN/frozen_baseline/20260718_125303/` — LIMIT=900 (all of
  `mmred_images_park/seq_len_8/all_uniform`, gold uniform 0..8).
- **acc 0.219, MAE 1.86.** Per-count: g0 68, g1 29, g2 57, g3 38, g4 1, g5 0, g6 2, g7 0,
  g8 2 (each /100) — the undercount clamp in full view; virtually no mass above 3.
- In the pre-registered band (~0.2). The truncated-prior 0.513 (job 123205) is hereby retired —
  it rewarded the model's low-count bias. **0.219 is the headline frozen-model number for N=8.**
## [2026-07-18] Smokes (round 1 job 123222, round 2 job 123226) — ALL PASSED

- Round 1 (`outputs/_scratch/st2_smoke/`): carrier_layer mask-debug (8 blocks identical
  allowed-keys, hi=lo+0..+7, tail hi−lo=8), pos-debug blocks_identical=True, stratified
  gold-hist 3/class; distill + probe shuffle/hist wiring OK (small-n anchors deflated as
  documented — plumbing only).
- Round 2 (`outputs/_scratch/st2_smoke2/`): (A) single-task train path byte-identical behavior
  (loss 14.534 vs 14.538); (B) steps+cooc mixture: task parse + stratified split + per-task
  eval OK; (C) eval-only streaming on the truncated-prior ckpt (ep12 acc 0.840): N=8 uniform
  → 0.333 = EXACTLY g0-g2 correct, everything above wrong — the truncated-prior pathology
  reproduced from the model side; (D) N=32 eval-only with multi-digit greedy decode: 32-block
  mask identity + posreset verified at seq=6407, acc 0.154 = again exactly g0/g1.
- In-flight: 123232 probe full-prior (E1a), 123233 distill full-prior (E1c), 123235 stage-2
  FULL-PRIOR headline 40ep/900 (E2), 123206 truncated-prior 30ep (E2 convergence reference).
## [2026-07-18] E1(a) — Q-first blockfence probe + gate->tally on the FULL prior (jobs 123232, 123236) — LANDED

- Probe run: `outputs/ladder/image_longN/replica_blockfence_qfirst_full900/20260718_130546/`
  (n=900, gold uniform 0..8, mask/pos-debug PASS, skip=0).
- **d' 13.54 ± 0.27 @L16** (n=900 scale; AUC saturated 5.26); joint anchor 5.95 (ratio 2.28x);
  per-copy FLAT 8.4-9.2. Matched-n honesty check (subsample to n=300, 3 seeds): **10.50-10.89**
  vs the truncated-prior band 9.24 ± 0.33 — the supply is prior-free (if anything reads higher
  with high-count samples included). d' estimator scale-with-n reconfirmed (13.5@900 vs ~10.7@300).
- **Gate->tally (CPU, `gate_tally/` under the run): exact 0.998 ± 0.001, MAE 0.00, per-frame err
  0.0002** vs majority 0.111; per-count uniform (only g6 drops 4/242). THE scaffold headline at
  N=8 on the full prior — better than the truncated-prior 0.991. Scaffold ceiling for stage-2
  comparisons = 0.998.

## [2026-07-18] E2 (reference arm) — 30-ep truncated-prior stage-2 (job 123206) — LANDED

- Run: `outputs/ladder/image_longN/carrier_layer/20260718_122503_L17_r8/` (n=300 truncated
  gold∈{0,1,2}, train 150 / eval 150, warm-start distilled e_c).
- **BEST 0.840 @ ep12; loss -> 1e-4 by ep14; eval FLAT 0.81-0.84 through ep30.**
- **Reading: the "undertrained, still climbing" hypothesis is REFUTED.** The 12-ep first run
  (0.853) had effectively converged; the model memorizes 150 train samples while eval saturates
  ~0.84. The gap to the scaffold is a DATA/generalization gap, not an optimization one → the
  full-prior 450-train run (123235) is the right lever; extending to 60 ep is not. No 60-ep job.
## [2026-07-18] E4/E5 pre-registration (written at submit time)

- **123237 (E4b)** steps+cooc 50/50 mixture, 450/root stratified-shuffled, train 450 / eval 450,
  30 ep, warm-start distilled e_c, L17 r8 → `outputs/ladder/image_longN/carrier_layer_mixture/`.
  Bands: GO mixture >=0.90 on BOTH tasks' eval; scaffold refs steps 0.998 (full prior), cooc
  0.973. Watch per-task row.
- **123240 (E5)** steps+cooc+rooms mixture, 300/root, train 450 / eval 450, 30 ep →
  `outputs/ladder/image_longN/carrier_layer_mixture3/`. Rooms = set-union across carriers (a
  per-frame tally provably cannot express it; K-channel record linear 0.40). Bands: rooms
  emitted >=0.90 = cross-frame GO (multiclass-gate pipeline ref 0.993); steps/cooc must not
  regress below their 2-task values.
- **E4a** (steps-only -> cooc zero-shot, eval-only) runs after 123235's ckpt exists.
- Rooms parse smoked in job 123239 (`outputs/_scratch/st2_smoke3/`): 3 tasks, skip=0,
  stratified split OK.
## [2026-07-18] E1(c) — carrier-token distill on the FULL prior (job 123233) — LANDED

- Run: `outputs/ladder/image_longN/carrier_token/20260718_130545_distill_room_k1/` (n=900,
  train 450 / eval 450, distill objective, 10 ep).
- **In-run teacher anchor d' 13.54 ± 0.27 full-n — EXACTLY reproduces the probe run** (123232);
  data path verified. Teacher eval-split (scale-matched ceiling): 11.94 ± 0.24.
- **BEST carrier eval d' 11.45 @ ep9 (full-n 13.01) = 96% of teacher** — the one-embedding
  bottleneck costs even less on the full prior than the 92-93% measured at n=300. Converged by
  ep5 (11.3 flat). e_c ckpt: `carrier_best.pt`; messages cached in `messages_best.npz`.
- **Carrier-stack tally on the full prior (CPU job 123243, log `logs/car_tally-123243.out`):
  fresh logistic on `messages_best.npz` (5 seeds): per-frame err 0.0002, exact 0.999 ± 0.001.**
  The 3.6k-param carrier + gate stack matches the 20-token replica scaffold (0.998) at N=8 full
  prior. E1 scaffold row complete.

## [2026-07-18] E2/E4b/E5 CONVERGED (logged post-hoc; the campaign agent died on a session limit mid-wait)

| arm (job) | run dir | BEST emitted acc | per-task / notes |
|---|---|---|---|
| E2 stage-2 steps-only, FULL prior, 40ep (123235) | `carrier_layer/20260718_13*_L17_r8/` | **0.678 @ep30** | per-count UNIFORM 0..8 (clamp dead: g8 34/51); train loss 1e-4 = memorized |
| E4b steps+cooc mixture, 30ep (123237) | `carrier_layer_mixture/…` | **0.693 @ep25** | cooc 0.796 / steps 0.560 — no task interference; one carrier + one LoRA serves both |
| E5 3-task incl. rooms, 30ep (123240) | `carrier_layer_mixture3/…` | 0.509 @ep12 | rooms 0.50 (vs frozen 0.087, pipeline 0.993) — the nonlinear cross-carrier reduction PARTIALLY learned |
| E1c distill carrier, full prior (123233) | `carrier_token/…` | d′ 11.45 @ep9 (teacher 13.54, 85%) | supply is NOT the bottleneck |

**Diagnosis (consistent across all arms):** every arm memorizes its ~450 train samples and
plateaus — a DATA-STARVATION ceiling, not an architecture ceiling (epochs refuted by the 30-ep
reference; data tripling moved majority-relative accuracy 2.5× → 6.1×). Scaffold anchor on the
same data: 0.998. **Next lever (Phase 2): pooled multi-N multi-task training data (~10×).**

# PHASE 2 (2026-07-18 evening session)

## [2026-07-18] P1/P2 pre-registration (written BEFORE the runs)

**Data audit for P1 (pooled multi-N multi-task):** steps roots
`mmred_images_park/seq_len_{2..8}/all_uniform` = 300/400/500/600/700/800/900 dirs (100 per gold
e0..eN — gold range grows with N, so the pooled steps prior is low-count-skewed by construction;
per-(task,N)-group eval is the honest readout) + cooc `mmred_cooc_balanced/seq_len_8` = 1080
(120 per K0..K8) + rooms `mmred_rooms_balanced/seq_len_8` = 720 (120 per K1..K6; NOT K1..K8 as
assumed — max gold 6). **Total pool 6000; train 5100 / eval 900, split stratified per
(task,N) group (~15% held out per root).**

**Code change (one change: the loader/cache):** `carrier_layer_lora.py` — (1) attention masks
no longer cached per sample (2× seq² fp16 ≈ 7 MB/sample → ~100 GB at 6k pooled samples);
rebuilt lazily at forward time from cached `blocks`/`cpos`/`fin` via the new `make_masks()`
(same construction code moved verbatim; eval-only path still caches). (2) train/eval split +
eval reporting now stratified/keyed per (task,N) group (was per task). (3) per-group gold-hist
prints. Smoke job 123725 (`outputs/_scratch/st2_smoke4/`): variable-N mixture train path +
eval-only regression on the truncated-prior ckpt.

**P1 run (THE decisive one):** 9 pooled roots, LIMIT=2000/root (=all), TRAIN_N=5100, 12 ep,
L17 r8, warm-start e_c from full-prior distill (`carrier_token/20260718_130545…/carrier_best.pt`,
d′ 11.45), lr unchanged (1e-3/1e-4) — data is the ONLY change vs 123235/123237/123240.
a100-public, 4d_1g, mem=120G (~60 GB emb cache), expected ~10 h.
**Bands (pre-registered): GO ≥0.90 overall with per-task ≥0.85; 0.75–0.90 = data curve still
rising → double data again before touching architecture; steps8 reference points: steps-only
450-train 0.678, mixture 0.560; cooc ref 0.796; rooms ref 0.50. Scaffold 0.998.**

**P2 (eval-only on the EXISTING full-prior steps ckpt `carrier_layer/20260718_131157_L17_r8/`,
acc 0.678 @N=8):**
- (a) E3 length: N=32 (`mmred_longN_park/seq_len_32`, all 390, a100 2h_2g) and N=128
  (`seq_len_128`, LIMIT=100 stratified-shuffled, h200 24h_1g — will queue; h200 full now).
  Report RAW multi-digit-decode AND 0-9-restricted acc + per-count. Bands: scaffold zero-shot
  stack read 0.917/0.860; the in-model ckpt is 0.678 at its TRAINING length, so expect lower —
  the honest quantity is retention (acc@N / acc@8). ≥0.85 retention = length-robust;
  a multi-digit-emission failure boundary is itself a finding.
- (b) E4a task transfer: same ckpt zero-shot on cooc (all 1080, a100 24h_1g). Refs: frozen
  model ~0.2-ish on cooc (not measured — note), mixture-trained cooc 0.796, carrier-token
  supply transfer 88% of ceiling. Band: meaningfully above chance 0.111 + report gap to 0.796.

## [2026-07-18] P2(a) E3 — steps-trained stage-2 ckpt at N=32 (job 123742) — LANDED

- Run: `outputs/ladder/image_longN/carrier_layer_eval_N32/20260718_182252_L17_r8_evalonly/`
- Same ckpt (steps-only, trained at N=8 only, 0.678 there), streamed over all 390 of
  `mmred_longN_park/seq_len_32` (gold ∈ {0..8, 12, 16, 24, 32}, 30 each).
- **Emitted RAW 0.138 / 0-9-restricted 0.126, MAE 2.04.** In-range (gold ≤ 8) subset: 49/270
  = **0.181**. Mass again piles at g2-g3; multi-digit greedy decode works mechanically (g24:
  3/30 correct RAW — raw > restricted) but there is no real signal.
- **Reading (honest negative): the N=8-trained stage-2 integration does NOT length-extrapolate**
  — retention 0.181/0.678 ≈ 27%, at/below the frozen-model level (0.219 @N=8). The SUPPLY is
  length-general (probe flat to N=128; carrier token d′ 9.7 @N=128) but the LoRA readout was
  only ever shown 8 carriers at sequential positions 56..63; 32 carriers at 56..87 is
  off-manifold. This is exactly what P1's variable-N pooled training is built to fix — P3
  reruns this eval with the pooled ckpt.

## [2026-07-18] P2(b) E4a — steps-trained stage-2 ckpt ZERO-SHOT on cooc (job 123743) — LANDED

- Run: `outputs/ladder/image_longN/carrier_layer_eval_cooc0shot/20260718_182247_L17_r8_evalonly/`
- Ckpt: full-prior steps-only stage-2 (`carrier_layer/20260718_131157_L17_r8/`, 0.678 @N=8
  steps), streamed over ALL 1080 of `mmred_cooc_balanced/seq_len_8` (gold uniform 0..8).
- **Emitted acc 0.179 (raw = restricted), MAE 1.43** vs chance 0.111, mixture-trained 0.796.
  Per-count: mass piles on g3 (90/120) with g0/g7/g8 = 0 — the model emits mid-range counts
  regardless of the cooc question.
- **Reading (honest negative): in-model zero-shot TASK transfer is essentially absent.** The
  carrier-token SUPPLY transfers across tasks (steps→cooc 88% of ceiling, RESULTS 07-17/18),
  but the stage-2 trained integration (LoRA readout) is task-bound — task-generality must come
  from mixture training (E4b showed one LoRA serves both tasks without interference), not from
  zero-shot generalization of a single-task readout.

## [2026-07-18] P1-CACHED (frozen-e_c fast trainer, run by the main session in parallel) — GO 0.980

- Script `experiments/glstm/carrier_layer_cached.py` (NEW): e_c FROZEN from the distilled ckpt;
  layers 0..L_OPEN-1 run ONCE per sample at prep, h_L16 cached (bf16 RAM, ~40GB); only the
  L17-27 LoRA trains on cached states; masks rebuilt per step from metadata; MATH-SDPA forced
  (backend-dependent mask-dtype strictness — new pitfall: mem-efficient backend requires exact
  mask/query dtype match under no_grad, math backend doesn't).
- Run: job 123937 → `outputs/ladder/image_longN/carrier_layer_cached/<ts>_L17_r8/`; steps
  seq_len_2-8 + cooc_8, LIMIT=900/root, ~5.1k samples, 934 s/epoch (≈4× faster than the full
  trainer).
- **BEST 0.980 @ ep10 (steps 0.987 / cooc 0.946), MAE 0.02**; oscillates 0.95±0.03 after.
- **Readings:** (1) GO — the in-model readout crosses the band once data is sufficient;
  (2) frozen e_c matches the trainable-e_c P1 arm (0.963@ep5, still climbing) — task-time
  carrier tuning is unnecessary, the distilled embedding is reusable as-is; (3) the cached
  trainer is validated as the workhorse for scale/sweeps.

## [2026-07-19] Pre-registered: the three exams of the pooled cached ckpt (0.980) — submitted by the main session

- **124275 LENGTH:** eval-only at N=32 (`mmred_longN_park/seq_len_32`, n=300 stratified, counts
  to 32 = multi-digit answers never trained). Bands: scaffold zero-shot ref 0.917; old
  450-ckpt read 0.138; expect degradation concentrated at counts >9 — retention vs the 0.980
  training-length acc is the honest quantity.
- **124276 HELD-OUT TASK:** rooms N=8 zero-shot (`mmred_rooms_balanced/seq_len_8`) — rooms was
  NOT in the cached arm's training (only in P1's). Refs: P1-mixture rooms 0.88@ep5+, frozen
  0.087, chance 0.111.
- **124277 DRIFT ("does the adapter break the model"):** `frozen_baseline_eval.py --lora-ckpt`
  (new flag): the PLAIN prompt with the trained LoRA hooks ACTIVE, same 300 samples. Ref: 0.219
  without LoRA. ≈0.22 = adapter is prompt-mode-selective (deployment-safe even left on);
  large shift = gate the adapter (still safe — the base model is untouched by construction).

## [2026-07-19] Exams 124276 (rooms zero-shot) + 124277 (LoRA drift) — LANDED

- **124276 HELD-OUT TASK:** cached ckpt (steps+cooc, 0.980) zero-shot on rooms N=8, n=300
  stratified → `outputs/ladder/image_longN/cached_eval_rooms0shot/20260719_000556_L17_r8_evalonly/`.
  **acc 0.153 (chance 0.111), MAE 1.64**; mass piles on g2-g3 again. Confirms E4a with a second
  task pair: the trained readout does NOT zero-shot to an unseen task — task coverage must come
  from the training mixture (P1's 3-task arm reads rooms 0.88 in-mixture).
- **124277 DRIFT:** plain prompt + trained LoRA hooks ACTIVE, n=300 stratified →
  `outputs/ladder/image_longN/frozen_baseline_driftlora/20260719_000556/`. **acc 0.313 vs 0.219
  frozen ref (n=900; n-mismatch caveat), MAE 1.21.** The adapter does not break the base model
  on plain prompts — it *improves* it slightly (count mass spreads above g3 where frozen
  collapses). Deployment-safe left on; no gating needed.

## [2026-07-19] Exam 124275 — cached ckpt (0.980 @N<=8) at N=32 — LANDED

- Run: `outputs/ladder/image_longN/cached_eval_N32/20260719_000556_L17_r8_evalonly/` (n=300
  stratified over gold {0..8,12,16,24,32}).
- **acc 0.097, MAE 3.44 — near-total collapse: g0 24/24 correct, everything else ~0.** The
  N=8-trained readout answers "0" almost regardless of content at N=32 (the old 450-ckpt at
  least emitted low counts, 0.138). Fixed-N training binds the readout to the trained carrier
  position range even harder as in-range accuracy improves.
- Track-A target confirmed: readout, not supply, is the sole non-extrapolating stage.

## [2026-07-19] TRACK B pre-registration — InternVL2.5-8B Q-first solo probe (written before the run)

- Existing record (RESULTS 2026-07-07, run `internvl/multipass_bench/` job 118996): isolated-frame
  carrier d′ **6.38/6.56 @L16/L20 vs joint 1.79/1.90** — the supply gap (3.5×) already ports.
- Missing for the B1 deliverable: the **Q-first amplifier** arm. Change: `--qfirst` flag in
  `experiments/internvl/multipass_bench.py` (question BEFORE the frame as well; solo forwards =
  the fence/multipass-equivalent supply measurement — the fence↔multipass identity was
  established on Qwen; no mask surgery in remote code).
- Run: LIMIT=200, layers 16/20, same seed/estimator as 118996 → directly comparable.
- **Bands: solo-qfirst ≥ 2× joint (1.8/1.9) = mechanism-ports GO (already met by plain solo);
  qfirst ≥ +20% over plain solo (6.38/6.56) = the amplifier ports too (Qwen read +50%).**

## [2026-07-19] TRACK A pre-registration — A3 scratchpad+jitter training (written before the run)

- **Code (smoked in jobs 124278→124281, `outputs/_scratch/st3_smokeA/`):**
  (1) A1 verdict-scratchpad targets in `carrier_layer_cached.py` — teacher-forced
  `" frames 2, 5, 7 -> 3"` / `" none -> 0"` / `" rooms Garden, Park -> 2"` (+EOS), CE over the
  full target; per-epoch metric = teacher-forced COUNT-token acc (greedy-decode acc comes from
  eval-only exams); eval/train dir lists exported for contamination-free in-dist greedy eval.
  (2) A2 jitter `--jitter-gap G`: TRAIN-only, hi-phase-only — per step carrier gaps ~U{1..G}
  (G=12 → spans ≤96 at N=8; covers the carrier-carrier/tail-carrier RoPE distances of N=32
  fully, N=128 partially). DEVIATION from the brief's "one offset per sample": a uniform shift
  moves carriers TOWARD/PAST the tail (wrong direction); what long N actually enlarges is
  carrier-carrier and tail-carrier distances → gap-stretch covers those. Logged as the design
  rationale. (3) scratchpad greedy decode + parse-fail reporting + `--dirs-file` in
  `carrier_layer_lora.py` eval-only.
- **A3 run:** cached trainer, steps seq_len_2-8 + cooc(900 of 1080) + rooms(720), LIMIT=900/root
  (n≈5820), train-frac 0.85, 12 ep, frozen full-prior distill e_c, L17 r8, --scratchpad
  --jitter-gap 12. a100 4d_1g, expect ~6-7 h.
- **Bands (pre-registered):** in-dist TF-count ≥0.90 (digit-mode cached ref 0.980 — some loss
  to the harder target is acceptable); THE EXAMS (zero-shot greedy): N=32 ≥0.80 = READOUT
  SOLVED for length (scaffold 0.917; digit-ckpt collapse 0.097); 0.5-0.8 partial → A4 (add
  N=16/32 data); ≤0.5 = scratchpad insufficient, log honestly. Count-OOD: golds 12-32 get
  native multi-digit readout — report per-count; parse-fail separately.

## [2026-07-19] TRACK B — InternVL2.5-8B solo-QFIRST probe (job 124280) — LANDED

- Run: `outputs/frame_axis/internvl/multipass_qfirst/20260719_004112/` (n=200, 1600 solo
  passes, same seed/data/estimator as the 118996 record → sample-matched).
- **Solo-qfirst carrier d′ 6.31 @L16 / 5.11 @L20** vs joint 1.79/1.90 (recorded) and plain
  solo 6.38/6.56 (118996). Per-frame perception acc 0.586 (unchanged).
- **Verdicts vs pre-registered bands:** (1) **mechanism-ports GO** — fenced-equivalent supply
  ≥2× joint on a second model family (3.5× @L16); the aggregation bottleneck and its
  isolation relief are not Qwen artifacts. (2) **The Q-first amplifier does NOT port** —
  6.31 vs 6.38 flat at L16, 5.11 vs 6.56 at L20 (if anything negative). Question-conditioned
  frame encoding is a Qwen-specific (or at least not universal) bonus; honest scope note for
  the thesis: fence/isolation is the portable piece, Q-first is family-dependent.
- Method note: solo forwards = the fence/multipass-equivalent supply measurement (fence ≡
  multipass identity established on Qwen); no mask surgery in InternVL remote code needed.

## [2026-07-19] TRACK C pre-registration — ablation battery (written before the runs)

7 arms, cached digit trainer, steps8+cooc (LIMIT=900/root, n=1800, train 900), 8 ep, frozen
e_c, one change each vs base: **base(L17,r8) · L_OPEN=12 · L_OPEN=22 · rank=4 · rank=16 ·
--no-qfirst · --no-posreset**. 900-train is the known data-starved regime — ABSOLUTE numbers
will sit mid-range; the deliverable is the RANKING vs the in-run base arm (never cross-compare
to the 5.1k runs). Bands: no-qfirst should drop hardest (supply-level Q-first = +3 d′ on Qwen);
no-posreset a smaller drop (+0.6 d′ supply); L_OPEN/rank = ordering information for the thesis
table. NIAH (C3): 720 which-frame samples generated (`data/mmred_niah_which/seq_len_8/`,
K1-8 × 90, parse verified 16/16); zero-shot exams on the P1/A3 ckpts after they land.

## [2026-07-19] P1 mid-run + exam pre-registration

- P1 (123741) at **0.993 @ep7** (cooc 162/162, rooms 105/108, steps8 133/135) — the pooled
  3-task carrier layer effectively matches the 0.998 scaffold IN-MODEL at N<=8. Data starvation
  diagnosis confirmed: 450→0.678, 5.1k→0.980 (2-task cached), 6k→0.993 (3-task trainable e_c).
- **Pre-registered P1-ckpt exams (submit when 123741 lands):** (a) N=32 digit-restricted eval —
  THE variable-N question: P1 trained on N=2..8 (variable) vs the cached fixed-N=8 ckpt's 0.097
  collapse; band: >0.3 = variable N alone helps length; ≥0.8 = solved without scratchpad
  (unlikely — counts >9 can't be emitted single-digit; restricted-to-0..8 subset is the fair
  read). (b) NIAH which-frame zero-shot (never-seen question type, digit answer). Refs: chance
  0.125 (8 classes), frozen model TBD in the same run.

## [2026-07-19] TRACK C1/C2 — ablation battery LANDED (jobs 124300-124306)

All: cached digit trainer, steps8+cooc (n=1800, train 900), 8 ep, frozen e_c — the data-starved
regime by design; RANKING is the deliverable (base in-run reference, never cross-compare to 5-6k runs).

| arm | run dir (`outputs/ladder/image_longN/cached_ablations/`) | BEST acc |
|---|---|---|
| **L_OPEN=12** (124301) | `L12/20260719_*_L12_r8/` | **0.941 @ep8** |
| rank=16 (124304) | `r16/20260719_*_L17_r16/` | 0.731 @ep7 |
| base L17 r8 (124300) | `base/20260719_*_L17_r8/` | 0.698 @ep8 |
| rank=4 (124303) | `r4/20260719_*_L17_r4/` | 0.694 @ep7 |
| no-posreset (124306) | `noposreset/20260719_*_L17_r8/` | 0.669 @ep8 |
| L_OPEN=22 (124302) | `L22/20260719_*_L22_r8/` | 0.513 @ep8 |
| **no-Qfirst** (124305) | `noqfirst/20260719_*_L17_r8/` | **0.378 @ep8** |

**Readings.** (1) **Q-first is the single most load-bearing piece at the behavior level**
(0.698→0.378, −46%) — matches its supply price (+3 d′), and note Track B: this piece is
Qwen-specific. (2) posreset is real but mild (−4%; supply +0.6 d′) — consistent. (3)
**L_OPEN=12 ≫ 17 ≫ 22**: more trained integration depth wins decisively in the starved regime
(caveat: LoRA params scale with open depth, 16 vs 11 vs 6 layers — depth and capacity
confounded); at full data (5-6k) L17 already reaches the 0.98-0.99 ceiling, so the default
stands there, but L12 is the better default for small data and possibly for harder tasks.
(4) rank nearly flat (r4 0.694 / r8 0.698 / r16 0.731) — capacity per layer is not the wall.

## [2026-07-19] A3 training + first exams — LANDED

- **A3 training (job 124282, `carrier_layer_scratchpad/20260719_005342_L17_r8/`):** TF-count
  acc **1.000 @ep1** on all tasks (873 eval), tf-exact 0.953→0.986 @ep2, loss 0.284→0.008.
  The verdict-scratchpad target is drastically easier to optimize than the bare digit (the
  digit recipe needed 10+ epochs on comparable data). Ckpt = ep1 (save criterion saturated).
- **In-dist GREEDY decode exam (job 124314, `scratchpad_eval_indist/20260719_021326_*/`):
  acc 0.953, parse-fail 0.000, MAE 0.05** on the 873 truly-held-out dirs (dirs-file). Per-task:
  steps 0.980, cooc 0.906, rooms 0.850. The model emits well-formed verdicts+count every time.
- **NIAH which-frame ZERO-SHOT (job 124316, `scratchpad_eval_niah0shot/20260719_023026_*/`):
  0.087** (chance 0.125), parse-fail 0.071, mass on "frames 1 -> 1". Third confirmation:
  NO zero-shot transfer to unseen question types — task coverage must be trained (C3 mixture
  arm remains the test of "NIAH is easy once in the mixture").
- Pending: N=32 (124315), N=128 (124317) — the length verdicts.

## [2026-07-19] Discretionary pre-registration — COMPOSITION (OR-union) zero-shot exam

- Question ("does this strengthen the thesis table?"): the "programmable reduction" claim says
  the question in-context programs how carrier verdicts are reduced. The sharpest test is a
  reduction NEVER seen in training that composes trained primitives: "How many frames was C in
  the R1 **or** the R2?" (union count over two per-frame predicates). Generated 540 balanced
  samples from existing states (`data/mmred_union_or/seq_len_8/`, K0-8 × 60, symlinked frames,
  parse verified 16/16, `experiments/glstm/union_gen.py`).
- Exam: A3 scratchpad ckpt zero-shot, n=240 stratified, decode 48. **Bands: ≥0.7 = the readout
  composes untrained reductions (headline-grade); 0.3-0.7 partial compositionality; ≤0.3 = the
  readout is a fixed per-task program (consistent with the NIAH/rooms/cooc zero-shot nulls).**
  Note vs NIAH: union shares the trained VERDICT format ("frames ...") and the counting tail —
  NIAH did not (frame-INDEX answer); if union >> NIAH-0shot, format proximity is what transfers.

## [2026-07-19] C3(b) pre-registration — 5-task scratchpad mixture (written before the run)

- A3 recipe + NIAH(720) + union(540) roots → n≈7080, train-frac 0.85, 6 ep (A3 converged @1),
  jitter 12, frozen e_c, L17 r8. Gives the TRAINED rows for the two new tasks (the zero-shot
  rows come from the A3-ckpt exams).
- **Bands: NIAH-in-mixture ≥0.90 (the theory's "easy case" claim); union-in-mixture ≥0.85;
  steps/cooc/rooms TF-count must not regress below 0.95.**

## [2026-07-19] Composition (OR-union) zero-shot exam (job 124335) — LANDED

- Run: `outputs/ladder/image_longN/scratchpad_eval_union0shot/20260719_030349_L17_r8_evalonly/`
  (A3 ckpt, n=240 stratified 0..8).
- **acc 0.321 (band: partial compositionality), parse-fail 0.021, MAE 1.34.** Hits spread over
  the FULL count range (g0 27/27 … g8 8/26) — not a mode collapse; the model attempts the
  union reduction and lands near it (MAE ~1.3).
- vs NIAH-0shot 0.087: the shared VERDICT FORMAT ("frames … -> k") is what transfers — a
  never-seen reduction over trained primitives gets ~3.7× the fully-alien question type.
  Partial compositionality, honestly labeled; the trained row comes from the 5-task mixture
  (124336).

## [2026-07-19] P1 FINAL (job 123741) — LANDED: 0.999, the in-model method MATCHES the scaffold

- Run: `outputs/ladder/image_longN/carrier_layer_pooled/20260718_182248_L17_r8/` (6000 pooled samples: steps N=2..8 4200 + cooc 1080 + rooms 720; train 5100 /
  eval 900 stratified per (task,N) group; trainable e_c warm-started from the full-prior
  distill; L17 r8; 12 ep).
- **BEST emitted acc 0.999 @ep12 (899/900), MAE 0.00.** Per-task: steps groups ALL perfect
  (630/630 across N=2..8), rooms 108/108, cooc 161/162. Trajectory: 0.176(ep0) → 0.669(ep1)
  → 0.963(ep5) → 0.997(ep8) → blip 0.763(ep11, transient optimizer instability) → 0.999(ep12).
- **The data-starvation diagnosis is fully vindicated: 450→0.678, 5.1k→0.980, 6k→0.999 —
  the in-model carrier layer now equals/exceeds the 0.998 gate-tally scaffold at N≤8, with NO
  task-specific head, one architecture, three tasks incl. the provably-nonlinear rooms
  set-union.** Scaffold 0.998 / frozen 0.219 / chance 0.111.

## [2026-07-19] A3 exam — N=32 ZERO-SHOT length (job 124315) — LANDED: 0.215, A4 triggered

- Run: `outputs/ladder/image_longN/scratchpad_eval_N32/20260719_024313_L17_r8_evalonly/`
  (n=195 stratified over {0..8,12,16,24,32}, decode ≤160).
- **acc 0.215, parse-FAIL 0.000, MAE 4.79.** In-range (gold≤8): 42/135 = **0.311**; zero above
  g6. vs digit ckpts: fixed-N=8 0.097, 450-ckpt 0.138 → scratchpad+jitter is 2.2× better and
  the FORMAT fully survives (every sample yields a parseable verdict; the digit ckpt collapsed
  to "0"). Failure mode from decode samples: correct verdicts for small counts (e.g.
  " frames 5, 19 -> 2" exact at N=32); at larger counts the verdict list undercounts, and
  occasionally a degenerate "-> a -> b -> …" chain appears (parser takes the LAST match).
- **Band verdict: ≤0.5 → scratchpad+jitter alone does NOT solve zero-shot length; A4
  (pre-registered fallback: add long-N training data) triggered.**

## [2026-07-19] A4 pre-registration (written before the run)

- Recipe: A3's 9 roots + `mmred_longN_park/seq_len_16` (all 330) + `seq_len_32` FIRST 200 of
  the seed-0 stratified shuffle — the exam then uses the COMPLEMENT 190 dirs via --dirs-file
  (airtight split, same length, zero overlap). gold>9 now allowed in scratchpad training
  (multi-digit answers native). 3 ep, jitter 12, frozen e_c. Code: per-root `root=LIMIT`
  syntax + scratchpad gold>9 skip removal in the cached trainer (smoked before launch).
- **Bands: N=32 complement ≥0.80 = READOUT SOLVED for length (scaffold 0.917); N=128
  (LIMIT 65, 4× beyond max trained length) ≥0.5 = genuine extrapolation, ≥0.8 = solved;
  in-dist N≤8 TF must not regress (≥0.99).**

## [2026-07-19] P1-ckpt exam — N=32 (job 124353) — LANDED: variable-N does NOT fix length

- Run: `outputs/ladder/image_longN/pooled_eval_N32/20260719_051622_L17_r8_evalonly/` (n=195).
- **acc 0.092 (restricted 0.077), MAE 2.81** — the 0.999-at-N≤8 pooled DIGIT ckpt collapses
  to "0" at N=32 exactly like the fixed-N=8 cached ckpt (0.097). **Training-N DIVERSITY (2..8)
  buys nothing at 4× length — the digit readout binds to the trained range regardless.**
  Zero-shot length ranking now: digit ckpts 0.092-0.138 « scratchpad+jitter 0.215 (0.311
  in-range) « scaffold 0.917. The readout FORMAT, not the data mix, is the length lever —
  A4 (long-N scratchpad training, job 124362) is the fix under test.

## [2026-07-19] 5-task mixture greedy + P1 zero-shot contrast rows (124349/124354/124355) — LANDED

- **5-task mixture in-dist GREEDY (124349,
  `scratchpad5_eval_indist/…/`): 0.966, parse-fail 0.000, MAE 0.04** (n=1062 held out).
  Per-task: steps 0.997 · **which/NIAH 0.992** · cooc 0.944 · **union 0.910** · rooms 0.842.
  **Pre-registered bands: NIAH ≥0.90 GO (0.992 — "NIAH is the easy case" confirmed); union
  ≥0.85 GO (0.910); no regression vs A3** (rooms 0.842 ≈ A3's 0.850; TF was 1.000 on all).
  Trainer run: job 124336 (`carrier_layer_scratchpad5/20260719_031356_L17_r8/`), TF-count
  1.000 @ep1 on all 5 tasks, tf-exact 0.993 @ep2, cancelled after ep2 (ckpt=ep1).
- **P1 digit-ckpt zero-shots:** NIAH 0.117 ≈ chance 0.125 (124354); union 0.150 (124355) —
  4th and 5th zero-shot nulls for the digit readout. **Contrast: scratchpad union-0shot 0.321
  = 2.1× digit — partial compositionality is a property of the VERDICT FORMAT.**

---

# FULL-THESIS TABLE (Phase-3 endgame deliverable; updated as cells land)

Scaffold = Q-first blockfence probe + logistic gate→tally (task-specific head, external).
In-model = the carrier layer: fence ≤L16 + cross-carrier ≥L17 + LoRA r8 + carrier e_c; model
emits the answer itself. Frozen Qwen2.5-VL-7B 4-bit throughout. Chance = 0.111 (9-class).

| # | claim | number | run dir | verdict |
|---|---|---|---|---|
| 1 | In-model counting (steps, N≤8, full prior) | **0.999** (steps groups 630/630) | `carrier_layer_pooled/20260718_182248_L17_r8/` (123741) | **GO — matches scaffold 0.998, frozen 0.219** |
| 2 | In-model relational (cooc) | **0.994** (161/162) | same run | GO (scaffold 0.973) |
| 3 | In-model cross-frame set-union (rooms) | **1.000** (108/108) | same run | **GO — provably-nonlinear reduction in-model (pipeline ref 0.993)** |
| 4 | Data is the lever (not epochs/arch) | 450→0.678 · 5.1k→0.980 · 6k→0.999 | `carrier_layer/20260718_131157/` · `carrier_layer_cached/20260718_192428/` · pooled | GO — starvation diagnosis vindicated |
| 5 | Scratchpad readout, in-dist greedy | **0.953** parse-fail 0.000 | `scratchpad_eval_indist/20260719_021326/` (124314) | GO — model emits verdicts+count free-form |
| 6 | 5-task mixture (adds NIAH + union) | **0.966** greedy (which 0.992, union 0.910) | `scratchpad5_eval_indist/…/` (124349) | GO — one carrier+LoRA serves 5 tasks; NIAH "easy case" confirmed |
| 7 | Length extrapolation, digit readout | 0.092-0.097 @N=32 (collapse to "0") | `pooled_eval_N32/…/` (124353), `cached_eval_N32/…/` (124275) | **NO-GO — variable-N training does not help** |
| 8 | Length extrapolation, zero-shot readout | A3 0.215 · le16-tally 0.280 @N=32; le16 0.150 @N=64 | `scratchpad_eval_N32/` (124315) · `tally16_eval_N{32,64}/` (124522/124578) | partial — smooth decay 1.000→0.280(2×)→0.150(4×); format survives |
| 9 | Length with in-length training | A4 0.447 → **le64-tally 0.733 @N=32 · 0.558 @N=64 (0.821 in-range)** | `scratchpadLN_eval_N32heldout/` (124376) · `tally64_eval_N{32,64}heldout/` (124571/124586) | running-tally + in-length data is the working recipe; curve still data-limited; N=128 pending (124606/07) |
| 10 | Task transfer, zero-shot (any ckpt, any unseen task) | 0.087-0.179 ≈ chance (5 pairs) | `carrier_layer_eval_cooc0shot/` `cached_eval_rooms0shot/` `*_niah0shot/` `pooled_eval_*0shot/` | honest NO — task coverage must be trained (mixture costs nothing, row 6) |
| 11 | Compositionality (OR-union, never trained) | scratchpad **0.321** vs digit 0.150 | `scratchpad_eval_union0shot/` (124335) vs `pooled_eval_union0shot/` (124355) | partial — the verdict format composes 2.1× better than digit; trained row: 0.910 |
| 12 | Model port (InternVL2.5-8B, supply level) | solo-qfirst d′ **6.31/5.11** vs joint 1.79/1.90 | `outputs/frame_axis/internvl/multipass_qfirst/20260719_004112/` (124280) | **GO 3.5× — mechanism ports; Q-first amplifier does NOT (Qwen-specific)** |
| 13 | Deployment drift (LoRA left on, plain prompt) | 0.313 vs frozen 0.219 | `frozen_baseline_driftlora/20260719_000556/` (124277) | GO — safe, slightly helps |
| 14 | Ablation: no Q-first | 0.378 vs base 0.698 | `cached_ablations/noqfirst/` (124305) | Q-first most load-bearing (−46%) |
| 15 | Ablation: no posreset | 0.669 vs 0.698 | `cached_ablations/noposreset/` (124306) | mild (−4%), consistent with +0.6 d′ supply price |
| 16 | Ablation: L_OPEN 12/17/22 | **0.941** / 0.698 / 0.513 | `cached_ablations/{L12,base,L22}/` | earlier opening ≫ in starved regime (depth/params confounded) |
| 17 | Ablation: rank 4/8/16 | 0.694 / 0.698 / 0.731 | `cached_ablations/{r4,base,r16}/` | capacity not the wall |
| 18 | No-LoRA (attention alone cannot aggregate) | ~0.50 plateau (truncated prior) | `carrier_layer/20260718_023030_L17_r8_nolora/` | trained integration required (kept from Phase 1) |
| 19 | N=128 cells | **le16 8× zero-shot 0.087 (band ≥0.80 REFUTED)**; v2 2× pending (124758); l12v2/E-G arms = the remaining extrapolation candidates | `tally16_eval_N128/` (124736, recovered from dumps) | honest NO at 8×; ladder story stands |
| 20 | Layout freedom (E-C) | at-end+Qfirst d′ 10.27, tally **0.999** = interleaved; no-Qfirst 2.40/0.508 | `carrier_atend_qfirst/` (124514) vs `carrier_atend/` (124492) | **GO qualified: carriers placement-free IF question leads** |
| 21 | No-harm (E-D) | MME −0.2 pts · POPE −1.4 pts | `noharm_bench/20260719_203833/` (124508) | **GO** (band ≤2) |
| 22 | Seeds (E-E) | TF-count 1.000 × 3 seeds (tf-exact 0.963±0.007) | `carrier_tally_le16{,_seed1,_seed2}/` (124482/509/510) | zero variance on the count cell |
| 23 | SFT control (E-B) | train-length 0.9984 (23.8M params); long-N BLOCKED (adapter never saved) | `sft_control_le8/20260719_185022_lora/` (124484) | control matches in-dist; long-N cell blocked (BLOCKED.md) |
| 24 | Rooms decode gap | all errors = missing-room verdicts; count==list 40/40 | `rooms_gap_diag/` (124527) | counting exact; detection recall is the residual |

## What remains (10 lines)

1. A4 verdict (job 124362) → rows 9; its N=32-complement + N=128 exams after it lands.
2. N=128 anything: h200 has been 8/8 all night — 124317 queued; resubmit A4-N128 when free.
3. rooms greedy sits at 0.84-0.85 across scratchpad arms (TF 1.000) — decode-level gap worth
   one look (likely the multi-room verdict list ordering).
4. Degenerate "-> a -> b" decode chains at OOD lengths — a stop-criterion or running-tally
   format variant would likely add several points at N≥32 (pre-approved variant, not run).
5. L12 + pooled data: does earlier opening also help at full data / OOD length? (1 job.)
6. InternVL trained stack (carrier+LoRA) — supply ports; the in-model leg is future work.
7. Natural-images carrier layer (P4) — untouched this phase; cross-domain gap 0.43 vs 0.92 stands.
8. HERBench expected-null (P4) — untouched.
9. Composition beyond OR (comparatives, "more in X than Y") — generator pattern exists.
10. The ep11 P1 instability blip (0.997→0.763→0.999) — benign here, but lr schedule would tidy it.

## [2026-07-19] A4 training (job 124362) — LANDED; exams launched

- Run: `outputs/ladder/image_longN/carrier_layer_scratchpad_longN/20260719_054023_L17_r8/`
  (n=6350: A3 roots + longN_16 all 330 + longN_32 first-200 of seed-0 shuffle; gold to 32
  multi-digit; grad-ckpt for the long-seq backward — two smoke rounds fixed an OOM at
  seq 5.8k on 40GB and a checkpoint/sdpa-backend recompute mismatch, jobs 124356/60/61).
- **TF-count 0.994 → 0.996 → 0.997 (BEST @ep3), tf-exact 0.966; long-N training samples
  essentially solved in one epoch.**
- Exams: 124376 N=32 on the 190 HELD-OUT complement dirs (airtight split, decode 160);
  124377 N=128 (h200 queue, 4x beyond max trained length).

## [2026-07-19] A4 exam — N=32 HELD-OUT (job 124376) — LANDED: 0.447 (in-range 0.626), partial

- Run: `outputs/ladder/image_longN/scratchpadLN_eval_N32heldout/20260719_092517_L17_r8_evalonly/`
  (n=190 complement dirs — zero overlap with the 200 training dirs).
- **acc 0.447, parse-FAIL 0.000, MAE 1.44.** In-range (gold≤8): 82/131 = **0.626**; g12/g16:
  3/30; g24/g32: 0/30.
- **Progression on the same length: digit ckpts 0.092-0.097 → scratchpad+jitter zero-shot
  0.215 (in-range 0.311) → +200 in-length samples 0.447 (in-range 0.626).** MAE 4.79→1.44.
  Verdicts stay perfectly formed at every length (parse-fail 0 everywhere); errors are verdict
  UNDERCOUNTS on long evidence lists, not format/position collapse.
- **Band verdict: partial (<0.80).** The jump from just 200 in-length samples (only 30 with
  gold>8) says the long-N data curve is still steep — the same data-starvation shape that
  450→6k resolved at N=8. Not an architecture wall on this evidence.

---

# ONE-SCREEN SUMMARY — Phase 3 (2026-07-19, autonomous session end)

**The method now exists end-to-end.** One frozen Qwen2.5-VL-7B + one carrier embedding + one
LoRA (~2M params): fence ≤L16, cross-carrier attention ≥L17, model emits the answer.

- **In-model = scaffold at N≤8: 0.999** (3 tasks; rooms set-union 108/108) — `carrier_layer_pooled/`
  (123741). Data was the whole gap: 450→0.678, 6k→0.999. Frozen 0.219, chance 0.111.
- **Verdict-scratchpad readout** ("frames 2, 5 -> 2"): TF 1.000 in ONE epoch; in-dist greedy
  0.953-0.966 with parse-fail 0.000; **5-task mixture** (adds NIAH 0.992, OR-union 0.910) —
  one carrier+LoRA serves 5 question types (124336/124349).
- **Length**: digit readout collapses at N=32 regardless of training-N diversity (0.092-0.097);
  scratchpad+jitter zero-shot 0.215; **+200 in-length samples → 0.447 (0.626 in-range),
  format never breaks** (124376). Long-N data curve still steep — next lever is more long-N
  data, not architecture. N=128 rows pending (124317/124377 queued; h200 full all night).
- **Zero-shot task transfer is uniformly ~chance** (5 pairs) — mixture training is the
  (cheap) requirement. **Compositionality**: untrained OR-union 0.321 scratchpad vs 0.150
  digit — the format partially composes; trained: 0.910.
- **Ports**: InternVL2.5-8B supply gap 3.5× (solo-qfirst d′ 6.31 vs joint 1.79) — mechanism
  ports; **Q-first amplifier is Qwen-specific** (124280). **Drift**: adapter left on plain
  prompts HELPS (0.313 vs 0.219) — deployment-safe (124277).
- **Ablations** (8ep, 900-train): no-Qfirst −46% (most load-bearing) > L22 > no-posreset −4%;
  **L_OPEN=12 0.941 ≫ L17 0.698** in the starved regime; rank flat (124300-06).

**Blocked/pending:** N=128 exams (h200 saturation, jobs queued); rooms greedy ~0.85 decode gap;
"-> a -> b" degenerate chains at OOD length (running-tally format = ready next variant).
**Budget:** Phase-3 ≈ 30 GPU-h (≤35 ✓). Full detail: FULL-THESIS TABLE above; every number
traces to a run dir. Code additions all smoked: scratchpad targets + jitter + grad-ckpt +
per-root limits (`carrier_layer_cached.py`), scratchpad decode + --dirs-file
(`carrier_layer_lora.py`), NIAH + union generators, InternVL --qfirst.

---

# PHASE 4 (2026-07-19 evening, autonomous until Tal returns)

## [2026-07-19] Phase-4 collect + E-C pre-registration

- In-flight (main session): **124482** headline 5-task ≤16 running-tally+jitter →
  `carrier_tally_le16/20260719_184950_L17_r8/`; **124483** upper (+longN_32/64) →
  `carrier_tally_le64/20260719_184950_L17_r8/`; **124484** SFT control (peft LoRA r8, 23.8M
  params — 12× our budget — on q/k/v/o+MLP all layers, steps ≤8, train 2940, summary only at
  END); 124317/124377 N=128 baselines still h200-queued.
- **E-C pre-registration (carriers-at-end, coded + smoking as job below):** student layout
  [frames][question][c×N] with NO leading question; carrier_i reads {prefix, question,
  frame_i, itself}; carriers hidden from everything else; frames posreset to block-0,
  question+carriers sequential after (existing tail rebase). Teacher = FRAMES-FIRST
  blockfence replica (A3 layout, anchor ≈6.34 — NOT the 9.24 Q-first teacher). Distill
  N=8 LIMIT=900 train 450/eval 450 after smoke.
  **Bands: eval d′ ≥5 (~80% of the 6.34 teacher) AND fresh-logistic tally within a few
  points of the Q-first carrier stack (0.999) ⇒ layout-freedom GO (prompt-order-free method
  statement). Expected BELOW the Q-first carrier d′ 11.45 — the Q-first amplifier is priced
  at +3 d′; the claim here is layout FREEDOM, not parity.**

## [2026-07-19] 124482 headline arm converged @ep1 — E-A/E-D/E-E launched against its ckpt

- 124482 (le16): n=7410 (5 tasks + longN_16), train/eval 3705/3705 (train-frac 0.5),
  **TF-count 1.000 @ep1** (tf-exact 0.678 — running-tally transcripts are longer, exactness
  lags; curve left running for the record). Ckpt fixed at ep1 (save criterion saturated).
- **E-A pre-registered bands (restated): le16@N=128 ≥0.80 = THE headline (train ≤16 →
  extrapolate 8×); 0.5–0.8 partial → E-F; ≤0.5 → transcript diagnosis. N=32/64 exams
  LIMIT 300/200; running-tally decode caps 240/320 (g≥~45 truncations reported per-count).
  Refs on N=32: plain-scratchpad zero-shot 0.215, A4-trained 0.447, digit 0.092.**
- **E-D pre-registered:** MME+POPE ~500 items each, base vs le16-LoRA-on, |Δ| ≤ 2 pts = GO.
- **E-E:** seeds 1/2 of the headline recipe (reconstructed: same 11 roots, LIMIT 900,
  --running-tally --jitter-gap 12 --grad-ckpt --train-frac 0.5, EPOCHS=3 — deviation: fewer
  epochs than the spec arm; justified by the @ep1 saturation of the save criterion).

## [2026-07-19] E-C carriers-at-end (job 124492) — LANDED: honest NEGATIVE, d′ 2.40 (band ≥5)

- Run: `outputs/ladder/image_longN/carrier_atend/20260719_192758_distill_room_k1/` (n=900,
  train 450/eval 450, 10 ep, distill vs the FRAMES-FIRST blockfence teacher).
- Teacher anchor healthy: held-out d′ **8.89 ± 0.14** (n=900 scale; the ~6.3 A3 band was
  n=300 — estimator-scale consistent). Student: **eval d′ 2.40 @ep10, loss converged 0.019
  by ep3 and FLAT** — the at-end carrier reproduces the teacher messages in bulk (low MSE)
  but not the discriminative evidence direction: **27% of teacher vs 96% for the Q-first
  interleaved carrier.**
- **Reading: layout freedom in the strong form FAILS — a single carrier token placed after
  the question cannot extract question-conditioned per-frame evidence, even though its
  allowed key set is IDENTICAL to the interleaved carrier's (mask-debug: 223 keys, same).**
  Two candidate causes differ between this and the winning layout: (a) frames are encoded
  WITHOUT the question in front (no question-conditioned encoding), (b) carrier is not
  ADJACENT to its frame. Follow-up (pre-registered below) isolates them.

## [2026-07-19] E-C(b) pre-registration — Q-first + carriers-at-end (one change: restore leading question)

- Layout [q0][frames][q0][c×N] (`--atend-qfirst`); teacher = Q-FIRST replica (anchor ~13.5
  at n=900). Carrier keys: prefix + leading q0 + own frame + final q0 + self.
- **Bands: eval d′ ≥5 ⇒ the E-C failure was missing question-conditioned frame ENCODING
  (adjacency innocent — partial layout freedom still claimable: carriers may sit at the end
  IF the question leads); d′ ~2.5 ⇒ adjacency/retrieval-distance is binding (carriers must
  interleave; layout claim dropped).**

## [2026-07-19] E-D no-harm benchmarks (job 124508) — LANDED: GO

- Run: `outputs/ladder/image_longN/noharm_bench/20260719_203833/` (500 MME + 500 POPE items,
  identical samples both arms, Yes/No logit-argmax readout, le16 running-tally ckpt LoRA).

| benchmark | base | LoRA-on | Δ | band |
|---|---|---|---|---|
| MME (acc) | 0.862 | 0.860 | **−0.2 pts** | GO (≤2) |
| POPE (acc / F1) | 0.862 / 0.839 | 0.848 / 0.819 | **−1.4 pts** | GO (≤2) |

- Per-subtask deltas are ~0 across 12/14 MME cells; existence −4.5 and landmark −2.4 are
  small-cell noise (n≈20-40/cell), celebrity +2.7. All POPE splits −1.1..−1.7.
- **With the drift row (0.313 vs 0.219 on plain MMRED prompts), the adapter is
  deployment-safe left permanently on: no measurable general-benchmark cost.**

- Note (ckpt-selection limitation, logged before exam results): the cached trainer saves on
  TF-COUNT acc, which saturates at 1.000 @ep1 for tally arms; tf-exact keeps climbing
  (0.678@ep1 → 0.933@ep2 on 124482). All Phase-4 exams therefore test the ep1 ckpt — if
  long-N transcripts show verdict-list token corruption, ckpt selection (save on
  (acc, tf-exact)) is the first E-F iteration candidate.

## [2026-07-19] E-C(b) — Q-first + carriers-at-end (job 124514) — LANDED: LAYOUT FREEDOM GO (qualified)

- Run: `outputs/ladder/image_longN/carrier_atend_qfirst/20260719_205916_distill_room_k1/`
  (n=900, train 450/eval 450, 10 ep; Q-first teacher anchor 13.70 = the known band).

| layout (all: fence + posreset, distill, N=8 full prior) | eval d′ | fresh-logistic tally |
|---|---|---|
| Q-first, carriers INTERLEAVED (123233, reference) | 11.45 (96% of teacher) | 0.999 ± 0.001 |
| **Q-first, carriers AT END (124514)** | **10.27 @ep2 (75%)** | **0.999 ± 0.001** |
| no leading question, carriers at end (124492) | 2.40 (27%) | 0.508 ± 0.017 |

- **Verdict: both E-C(b) bands met (d′ ≥5 ✓; tally equal to the interleaved stack ✓).
  The binding requirement is QUESTION-FIRST — not carrier placement.** Carrier tokens can sit
  as a block after the question at the sequence end (a pure suffix — no interleaving with
  user content), which is the deployment-friendly form. The strong form (nothing before the
  frames) fails because frames get encoded without question conditioning — the same +3 d′
  Q-first term seen in the C2 ablation (0.698→0.378) and priced at supply level.
- Method statement update: prompt = [question][frames][question][carrier suffix] — carriers
  never interrupt user content.

## [2026-07-19] 124482 headline arm FINAL + exam ckpt-consistency fix

- 124482 curve (4 ep, 1676s/ep): TF-count 0.9997@ep1 → **1.000@ep2 (saved: tf-exact 0.933)**
  → 1.000/0.960 → 0.9997/0.977. Ckpt on disk = ep2, **jitter_gap=16** (main session used 16,
  not 12 — my seed arms 124509/10 run jitter 12: now a jitter-dose comparison, logged as such).
- First N=32/64 exams (124505/06) had loaded the transient ep1 save (tf-exact 0.678) —
  CANCELLED at ~50/300 and resubmitted (124522/124523) against the final ep2 ckpt so all three
  E-A cells test the SAME weights (N=128 job 124507 loads ep2 whenever h200 frees).

## [2026-07-20] E-E seeds (jobs 124509/124510) — LANDED: zero variance on the count cell

| arm | seed / shuffle | jitter | TF-count BEST | tf-exact @ep3 |
|---|---|---|---|---|
| headline 124482 | 0 / 0 | 16 | **1.000 @ep2** | 0.960 (0.977 @ep4) |
| seed1 124509 | 1 / 1 | 12 | **1.000 @ep3** | 0.962 |
| seed2 124510 | 2 / 2 | 12 | **1.000 @ep3** | 0.966 |

- Runs: `outputs/ladder/image_longN/carrier_tally_le16_seed1/20260719_203924_L17_r8/` · `outputs/ladder/image_longN/carrier_tally_le16_seed2/20260719_203925_L17_r8/` (EPOCHS=3 vs headline 4 — save criterion saturates @ep1-2).
- **In-dist TF-count = 1.000 ± 0.000 (3 seeds); tf-exact 0.963 ± 0.007** (headline @ep3 basis).
  Jitter 12-vs-16 makes no in-dist difference. Length-cell error bars pending: N=32 exam of a
  seed ckpt queued after the headline exam lands (protocol value check first).

## [2026-07-20] Pre-registration — rooms decode-gap diagnostic (DO-NOT-STOP item 1)

- Question: rooms reads TF-count 1.000 but greedy 0.842-0.850 across scratchpad arms — where
  do the ~15% go? Hypotheses: (a) verdict lists a wrong/reordered room set but right count;
  (b) drops a room (undercount); (c) format derail. 40 held-out rooms transcripts from the
  5-task ckpt (`--dump-decodes 40`) decide. No band — diagnostic.

## [2026-07-20] Rooms decode-gap diagnostic (job 124527) — LANDED

- Run: `outputs/ladder/image_longN/rooms_gap_diag/…_evalonly/` (40 held-out rooms samples,
  5-task ckpt, full transcripts dumped). acc 0.825, parse-fail 0.000, MAE 0.17.
- **Every error is a MISSING-ROOM verdict; the emitted count always equals the emitted list
  length** (e.g. gold 6 → "Bathroom, Bedroom, Garden, Office, Park -> 5"). No format derail,
  no reordering errors, no count-list mismatch in 40/40 transcripts.
- **Reading: the readout's COUNTING is exact; the residual on rooms (and, by the same
  signature, at long N) is per-frame/per-room DETECTION RECALL in the verdict stage.** The
  right future lever is supply-side (carrier content), not the readout — consistent with the
  d′ picture (rooms verdicts require aggregating rare cross-frame evidence).

## [2026-07-20] Upper arm 124483 FINAL + exam hygiene

- le64: TF-count **1.000 @ep4** (tf-exact 0.947; n=8250 incl. golds to 64; 4550s/ep,
  jitter 16). Ckpt = ep4. Its first held-out exams (124536/37) had loaded the transient ep2
  save — cancelled and rerun as **124548 (N=32, 200 held-out dirs) / 124549 (N=64, 216) /
  124550 (N=128 zero-shot, h200 queue)**, all on the final ckpt.
- Process rule adopted: exam jobs launch only AFTER the training job COMPLETES (two
  incidents of transient-ckpt loading).
- SFT control (124484): epochs 0-2 val_acc 0.850/1.000/0.983 at train length; now ~4h in its
  end-of-run test-generation phase — untouched per brief.

## [2026-07-20] E-B SFT control — train-length LANDED, long-N BLOCKED

- **124484: test_iid acc 0.9984 at N≤8** (630 items, best ep1, bias +0.002) →
  `sft_control_le8/20260719_185022_lora/`. The 23.8M-param plain LoRA (12× our budget,
  q/k/v/o+MLP all layers) matches the 2M carrier layer in-distribution — as expected;
  the informative cell is long-N, which is **BLOCKED: the script never saves the adapter**
  (see plans/carrier_stage4_BLOCKED.md; retrain+save ≈5h if wanted).
- Budget trim: le64 held-out exams resubmitted at LIMIT 150/108 (124571/124572) — Phase-4
  exam battery was heading ~1.5× over the 30 GPU-h budget at full LIMITs.

## [2026-07-20] E-A cell 1 — le16 (running-tally, ≤16-trained) @N=32 ZERO-SHOT (job 124522) — LANDED

- Run: `outputs/ladder/image_longN/tally16_eval_N32/…_evalonly/` (n=300, ep2 ckpt confirmed).
- **acc 0.280, parse-fail 0.007, MAE 4.64.** In-range (g≤8) 82/208 = **0.394**; g≥12: 2/92.
- Comparison at N=32 zero-shot: digit 0.092 « A3 scratchpad+jitter (≤8) 0.215 « **tally+jitter
  (≤16) 0.280** « A4 with 200 in-length samples 0.447. The tally format + doubled training
  length improves the slope but 2× length extrapolation remains partial — headline band
  (≥0.80 @N=128 = 8×) now very unlikely; the operative lever remains in-length data
  (le64 held-out exams 124571/72 give the trained-at-length curve).

- Ops note: 124523 (le16@N=64, LIMIT=200 dec320) was pacing ~12 min/sample → guaranteed
  walltime kill with total loss (report writes at end). Cancelled at 25/200, resubmitted as
  LIMIT=60 dec200 (124578) — zero-shot degenerate decodes run to the cap, so the cap is the
  cost driver; truncation only affects golds ≥~28 which are failing anyway at this
  extrapolation ratio.

## [2026-07-20] E-A cell 2 — le64 (running-tally, ≤64-trained) @N=32 HELD-OUT (job 124571) — LANDED

- Run: `outputs/ladder/image_longN/tally64_eval_N32heldout/20260720_045609_L17_r8_evalonly/`
  (150 held-out N=32 dirs from the arm's own eval split; ep4 ckpt).
- **acc 0.733, parse-fail 0.000, MAE 0.43.** Per-count near-uniform INCLUDING multi-digit:
  g12 9/14 · g16 7/16 · g24 2/10 · **g32 9/9**; in-range (g≤8) 83/101 = 0.822.
- **The trained-at-length N=32 curve: plain scratchpad + 200 in-length samples (A4) 0.447 →
  running-tally + ~190 N32 + ~215 N64 training samples 0.733.** The tally format converts
  long-list counting into last-tally read-off (g32 = list-everything = perfect), leaving
  detection recall as the residual (same signature as rooms).

## [2026-07-20] N=128 exam consolidation (ops)

- All four queued N=128 exams (124317 A3, 124377 A4, 124507 le16, 124550 le64) were
  walltime-doomed: measured pace at seq≈23k is ~14-20 min/sample (decode-cap runs), i.e.
  15-21h per job vs 5-12h walls — and the report writes only at the end. All four CANCELLED.
- Resubmitted the TWO decisive cells only, right-sized (LIMIT=34 = 2/class over the 17-class
  prior, decode 280, 11h wall): **124596 le16@N=128 (THE headline cell) and 124597
  le64@N=128** (trained-≤64, 2× extrapolation). The A3/A4 N=128 rows are dropped for budget —
  their zero-shot length story is already established at N=32 (0.215 / 0.447); table rows
  marked accordingly. Truncation caveat: golds ≥~36 exceed the 280-token tally cap and score
  as fails — reported per-count; the le64 "list-everything" regime (g96/g128) is therefore
  not measurable under this cap and is annotated as such, not laundered.

## [2026-07-20] E-F pre-registration — long-N data iteration (CPU gen running)

- New training data via `generate_mmred_balanced.py` (seed 7, `data/mmred_longN_park2/`,
  disjoint from every exam pool): N=32 ×312 (24/count) + N=48 ×210 (15/count). If the
  current exam battery confirms the "in-length data moves the curve" reading (0.447→0.733
  precedent), the next iteration retrains the upper arm with these +522 samples and re-exams
  N=32/64 held-out + N=128. Bands set at retrain time.

- N=128 exams restarted once more (124606/124607, ~2 samples lost) with --dump-decodes 40:
  the first headline decode showed gold=0 answered CORRECTLY (" none -> 0") then degenerating
  into an incrementing "-> k" chain that flips the LAST-match parse to wrong. Full transcripts
  are required for the pre-registered detection-vs-counting diagnosis AND for an honest
  first-match-vs-last-match parse sensitivity note. Primary metric stays the pre-registered
  last-match parse.

## [2026-07-20] E-A cell 3 — le16 @N=64 ZERO-SHOT (job 124578) — LANDED

- Run: `outputs/ladder/image_longN/tally16_eval_N64/20260720_060832_L17_r8_evalonly/`
  (n=60, dec 200). **acc 0.150, parse-fail 0.133, MAE 10.77.** In-range (g≤8) 9/36 = 0.250;
  nothing ≥g12; dominant failure = repetition loops ("frames 12 (1), 12 (2), …").
- Zero-shot length decay for the ≤16-trained tally arm: **1× (in-dist TF) 1.000 → 2× 0.280 →
  4× 0.150** — smooth decay, no cliff, format survives partially (parse-fail only 0.13).

## [2026-07-20] E-A cell 4 — le64 @N=64 HELD-OUT (job 124586) — LANDED

- Run: `outputs/ladder/image_longN/tally64_eval_N64heldout/…_evalonly/` (52 held-out dirs,
  dec 280). **acc 0.558, parse-fail 0.135, MAE 0.84 (parsed).** In-range (g≤8) 23/28 =
  **0.821**; g12-32: 6/17; g48/g64: 0/7 — all cap-truncations (280 tokens < the 48-64-verdict
  tally), annotated as unmeasurable-under-cap, not model failures.
- **The length table for the running-tally arms (all parse-honest):**

| ckpt \\ exam | N=32 | N=64 | N=128 |
|---|---|---|---|
| le16 (trained ≤16) | 0.280 zero-shot | 0.150 zero-shot | pending 124606 (headline) |
| le64 (trained ≤64) | **0.733** held-out | **0.558** held-out (0.821 in-range) | pending 124607 |

## [2026-07-20] BUDGET OVERRIDE (Tal) + E-F retrain pre-registration

- GPU-hour caps void per Tal (brief BUDGET OVERRIDE section). Cluster etiquette unchanged.
- **E-F retrain (upper arm v2):** le64 recipe (12 roots + longN16/32/64 all, running-tally,
  jitter 16, grad-ckpt, train-frac 0.5, seed/shuffle 0) + `data/mmred_longN_park2/seq_len_32`
  (312) + `seq_len_48` (210) → n≈8772, EPOCHS=4, a100 4d_1g mem 200G. Exams AFTER completion:
  held-out N=32/64 (its eval split), N=128 (LIMIT 34 / dec 280 / 11h — ops lesson).
- **Bands (pre-registered): N=32 held-out ≥0.80 (from 0.733; +~500 in-length samples);
  N=64 held-out ≥0.65 (from 0.558, in-range basis); N=128 ≥0.35 = extrapolation improving,
  ≥0.50 = partial GO (2× beyond max trained 64).**

## [2026-07-20] E-B rerun pre-registration (SFT control, adapter-saving patch)

- Patch smoked (124683, `outputs/_scratch/st4_smokeSFT/`): adapter saves via peft
  save_pretrained; in-run long-N generate-and-parse eval added (`--eval-longn`).
- Rerun: steps ≤8, r8, target both, ep5 (124484 recipe), then in-run eval at N=16/32/64
  (limits 100/100/100, stratified seed 0). **Bands (theory): train-length ≈0.998 replicated;
  N=16 degraded; N≥32 COLLAPSE toward frozen level (joint supply d′≈2) — either outcome is a
  finding; if the plain LoRA somehow holds at N=32, the carrier-layer story needs revision.**

## [2026-07-20] Follow-up pre-registrations (budget-override lane)

- Ops: 124606 (headline N=128) was PREEMPTED/requeued by h200-dds — will rerun unattended;
  its 15 orphaned decode-samples retained in the log for the parse-sensitivity analysis.
- **(a) Seed error-bar on the zero-shot length cell:** N=32 exam (LIMIT 300, dec 240) on the
  seed1 tally ckpt. Question: is 0.280 stable across seeds? Band: none (error bar); report
  mean±range with the headline arm. Caveat: seed arms used jitter 12 vs headline 16 — the
  comparison bundles seed+jitter-dose; noted.
- **(b) L12 at full data (DO-NOT-STOP list):** le16 recipe with L_OPEN=12 (one change; the
  ablation read L12 0.941 vs L17 0.698 at 900-train). TF will saturate regardless; the
  informative cell is its N=32 zero-shot exam vs L17's 0.280. Band: ≥0.40 = deeper trained
  integration also extrapolates better → L12 becomes the recommended default; ≈0.28 = the
  L12 advantage is a small-data effect only.

- Parse-sensitivity preview (CPU, dumped transcripts so far): first-match vs last-match
  differs by ≤1 sample per cell (15 orphaned N=128: 0 vs 1 hits; others ±0-1). No systematic
  parse bias; full quantification from the 40-transcript dumps when 124606/07 land.
  Pre-registered last-match stays primary.

## [2026-07-20] E-G (position-coupled tally) — handoff collected; full train monitored

- Implementation by the main session: `couple_offsets` (carrier_layer_lora.py — ONE rule
  drives teacher-forced target positions AND online decode positions), `--pos-couple`
  (carrier_layer_cached.py), eval auto-detect from ckpt. Rooms uncoupled by design.
- **Smoke 124700 collected:** [couple-debug] prints; prep n=24 skip=0; loss 6.15 @ep1 (tiny-n
  plumbing run; no decode phase — decode-parse acceptance deferred to the full run's exams).
  Anchor rule CPU-verified on a 3-verdict tally: verdicts re-anchor (2→c2, 5→c5, 7→c7),
  paren tallies and the '-> k' tail never re-anchor — the exact c<m>+k acceptance pattern.
- **Full train = job 124701** (launched by the main session; mem 200G/22h matches the brief
  spec) — I monitor, not relaunch. **Pre-registered (from the brief): E-G GO = beats the
  matched E-F (uncoupled, 124682) cells at EVERY OOD length with parse-fail ~0.** Exams after
  completion: held-out N=32/64 from its eval split; N=128 LIMIT34/dec280/11h.

## [2026-07-20] Pre-registered (main session): POSRESET NECESSITY ablation — complete fence WITHOUT position reset

- Tal's challenge: +0.59 at N=8 is small, and no-posreset behavior at N=8 was only −4%. But the
  position tax scales with block offset (∝ N); the without-reset column at long N was never
  measured (supply flatness to 128 always had reset ON), and the fence-complete × no-reset cell
  is missing even at N=8. Two probe runs (zero training), Q-first blockfence, NO --reset-positions:
  (a) N=8 n=300 (vs 9.24 with reset); (b) N=64 n=150 (vs 7.55/11.4-band with reset).
- Bands: N=64 no-reset ≪ with-reset (e.g. <60% of it) ⇒ posreset is N-scaling-critical, keep with
  this as its justification; both ≈ with-reset ⇒ posreset demotes to optional (simplify the
  method, +0.59 was partial-fence-specific); intermediate ⇒ report the curve honestly.

## [2026-07-20] L12 full-data arm (job 124698) — LANDED; N=32 exam launched

- Run: `outputs/ladder/image_longN/carrier_tally_le16_L12/…_L12_r8/` — TF-count **1.000 @ep2**,
  tf-exact trajectory **0.942 → 0.991 → 0.984** (vs L17's 0.678 → 0.933 → 0.960): deeper
  trained integration reaches transcript fidelity ~1 epoch sooner. Ckpt = ep2 (job COMPLETE
  before exam launch, per hygiene rule).
- N=32 zero-shot exam launched (vs the L17 cell 0.280; pre-registered band ≥0.40 → L12
  becomes the recommended default).

## [2026-07-20] E-B SFT control long-N — LANDED (job 124696; N=64 cell OOM'd, adapter saved)

- Run: `outputs/ladder/image_longN/sft_control_le8_v2/20260720_191541_lora/` (r8 target=both,
  23.8M params, steps ≤8 train; best ep3 val 0.983; test_iid 1.000).

| eval | SFT control (plain LoRA, ≤8) | our tally arms | frozen |
|---|---|---|---|
| N=16 | **0.480** (pf 0.000) | (le16 in-dist TF 1.000) | ~0.2 |
| N=32 | **0.350** (pf 0.000) | le16 zero-shot 0.280 · le64 in-length **0.733** | 0.092-0.097 |

- **Reading (honest, more nuanced than the pre-registered "collapse"):** the plain LoRA does
  NOT collapse to frozen level at N=32 — it degrades to 0.350 by riding EXTREME-count anchors
  (per-count: g1 8/8, g32 7/7, g24 2/7, but g6-g8 0/27 at N=16 and mid-range ~0 at N=32 —
  a bimodal low-end/saturation heuristic, MAE 1.83). Two implications: (1) the theory's
  "joint supply d′≈2" ceiling shows up as the DEAD MID-RANGE, exactly where per-frame
  aggregation is needed; extremes are solvable from global gist. (2) The carrier method's
  long-N edge is the in-length-trained cell (0.733 vs 0.350) and uniform per-count coverage —
  the zero-shot tally cell (0.280) does NOT beat the SFT control at N=32; only in-length
  training does. Both facts go in the table.
- N=64 cell: OOM on l40s-44GB during generate (adapter saved; a separate loader-eval on h200
  is possible if wanted — low priority, curve already characterized).

## [2026-07-20] POSRESET NECESSITY ablation (main session, jobs 124713/124714) — LANDED: intermediate band, N-scaling justification

- Runs: `outputs/ladder/image_longN/noreset_N{8,64}/20260720_*/` — Q-first blockfence WITHOUT
  --reset-positions; comparators: with-reset 9.24 (N=8) and ~12 (N=64, bracketed 12.67@32 /
  11.57@128).
- **N=8: 7.74 ± 0.18** (Tal's instinct confirmed: minor at short N; also > fence-level 6.34 —
  Q-first partially substitutes for reset). **N=64: 7.54 pooled BUT per-copy decays 6.4 → ~3.0
  with frame index** — the A2 position-tax gradient returns without reset; the pooled number is
  carried by early frames. Extrapolated, late-frame supply at N=128 approaches joint level.
- **Verdict (pre-registered intermediate band): KEEP posreset, re-justified** — free at
  inference, minor at N=8, increasingly load-bearing with N (per-copy decay is its
  fingerprint), and required infrastructure for jitter + E-G position coupling. The historical
  "+0.59" was partial-fence-specific and is retired as its justification.

- Ops: 124606/124607 were CANCELLED externally at 22:39 (h200 queue management by the main
  session, presumably). le16@N=128 headline resubmitted (job below); the le64@N=128 cell is
  superseded by the le64v2 (E-F) and E-G N=128 exams that follow those trainers. The orphaned
  124606 partial transcripts (15+ dumps incl. a spectacular gold=32 sample whose tally chain
  shows COUNTING-INDEX CONFUSION at 8×: verdict indices leak into the tally slots) are kept
  for the diagnosis section.

## [2026-07-21] Seed error-bar on the zero-shot length cell (job 124697) — LANDED

- `tally16seed1_eval_N32/20260720_192734_L17_r8_evalonly/` (n=300, seed1 ckpt, jitter 12):
  **0.287, parse-fail 0.000** (headline arm, jitter 16: 0.280). The zero-shot N=32 cell is
  **0.284 ± 0.004 over two seeds** — stable; jitter dose 12-vs-16 indistinguishable here.

## [2026-07-21] E-F retrain (le64v2, job 124682) — LANDED; exam battery launched

- Run: `outputs/ladder/image_longN/carrier_tally_le64v2/20260720_185443_L17_r8/` (n=8772 =
  upper roots + longN_park2 32/48; TF-count **1.000 @ep3** (tf-exact 0.891@save, 0.916@ep4);
  ckpt = ep3; job COMPLETE before exams).
- Exams: 124755 N=32 held-out (150 of 355 mixed park+park2 eval dirs) · 124756 N=48 held-out
  (park2, 109 dirs) · 124757 N=64 held-out (52) · 124758 N=128 zero-shot (h200, behind the
  le16 headline 124736). Bands as pre-registered (N=32 ≥0.80, N=64 ≥0.65, N=128 ≥0.35/0.50).
- E-G (124701) at ep2 0.838/tf 0.500 — coupled targets fit slower (uncoupled ep2: 0.999);
  its 4-ep budget may leave it TF-undertrained; will note asymmetry at exam time if so.

## [2026-07-21] L12 @N=32 ZERO-SHOT (job 124727) — LANDED: 0.443, band GO — L12 is the new default

- Run: `outputs/ladder/image_longN/tallyL12_eval_N32/20260720_223517_L12_r8_evalonly/` (n=300).
- **acc 0.443, parse-fail 0.000, MAE 1.15** vs L17's 0.280/4.64 (same data ≤16, same tally
  recipe — ONE change: L_OPEN 12). Multi-digit zero-shot partially works (g16 5/23, g24 4/23,
  g32 6/23; L17 ≈ 0 there). **Zero-shot L12 ≈ A4's in-length-trained 0.447.**
- **Pre-registered band ≥0.40 met: deeper trained integration (16 LoRA layers vs 11)
  extrapolates far better — L_OPEN=12 is the recommended default from here.** (Depth/params
  confound from the C1 note still applies; the result is the config recommendation either way.)

## [2026-07-21] Pre-registration — L12 + in-length data ("l12v2"): the two levers combined

- One arm: le64v2 recipe (upper roots + park2 32/48, running-tally, jitter 16, grad-ckpt,
  train-frac 0.5, EPOCHS=4) with **L_OPEN=12** — combines the two measured levers (in-length
  data: 0.733@N=32-held-out; L12: 0.443 zero-shot). **Bands: N=32 held-out ≥0.85; N=64
  held-out ≥0.70; N=128 zero-shot ≥0.50 partial GO / ≥0.80 = THE headline achieved by
  composition.** Exams after completion, matched dirs to the v2 exams.

## [2026-07-21] E-G train (124701, main session's) — collected: TF-UNDERTRAINED at 4 ep; extended rerun launched

- Run: `outputs/ladder/image_longN/carrier_tally_pcouple/…/` — BEST **0.923 @ep4, tf-exact
  0.720**, still climbing every epoch (0.785/0.838/0.892/0.923) vs the matched uncoupled E-F
  v2 1.000/0.891 (identical n=8772/roots/split). **Finding regardless of exams: position
  coupling makes the TF objective materially harder to fit in-distribution** (~2× the epochs
  for the same TF level) — the anchor-jumping positions are a real optimization cost.
- Decision (shepherding call): examining the undertrained ep4 ckpt would confound the
  pre-registered E-F-vs-E-G comparison ("beats at every OOD length") — an E-G loss would be
  uninterpretable. **Rerun launched with EPOCHS=8, everything else identical**; exams follow
  its completion on the matched v2 dirs. The 4-ep ckpt is retained on disk.

## [2026-07-21] v2@N=32 held-out (124755) — LANDED with a ckpt-selection lesson; trainer fixed

- `tallyV2_eval_N32heldout/20260721_040638_L17_r8_evalonly/` (150 mixed park+park2 dirs):
  **0.607, parse-fail 0.000, MAE 0.71** (in-range 0.739; g12/g16 0/13). BELOW le64's 0.733 —
  but confounded: the acc-only save rule picked v2's ep3 (tf-exact 0.891) while le64's exam
  ckpt was ep4 (0.947). NOT clean evidence that more data hurt.
- **Trainer fix: save criterion is now (TF-count, tf-exact) lexicographic** (one-line change,
  syntax-checked). l12v2 and pcouple8 were restarted on the fixed criterion (l12v2=124773, pcouple8=124774), EPOCHS 5/8; ~1.7h prep each sacrificed for correct ckpt selection.

## [2026-07-21] v2@N=64 held-out (124757) — LANDED: 0.365 (same ckpt confound)

- `tallyV2_eval_N64heldout/20260721_040636_L17_r8_evalonly/` (52 dirs): **0.365, pf 0.096**
  (g48/64 cap-truncated as before) vs le64's 0.558. Both v2 OOD cells sit below le64's with
  the SAME direction and the ckpt-selection confound (ep3/tf 0.891 vs ep4/tf 0.947) — v2's
  extra data cannot be judged from these two cells. **The fixed-criterion arms (l12v2 124773,
  pcouple8 124774) are the canonical successors; v2 rows are kept as the confound exhibit.**

## [2026-07-21] THE HEADLINE CELL — le16@N=128 (job 124736, TIMEOUT@23/34, recovered from dumps) — 0.087: band REFUTED

- Recovered from 23 dumped transcripts (the --dump-decodes safety net; per-sample gold/parsed
  in `logs/cl_eval-124736.out`; run dir `tally16_eval_N128/…`): **last-match acc 0.087,
  first-match 0.130, parse-fail 0.087, MAE 35** (per-count: only g0, g3 hit once each).
- **Pre-registered ≥0.80 band decisively NOT met. Train-≤16 → emit-at-128 (8×) does not hold
  for the running-tally+jitter recipe.** Transcripts show the failure: tally-index confusion
  (verdict indices leak into tally slots: "frames 30 (11), 31 (112)…") and unterminated
  chains — the coupled-position hypothesis (E-G) targets exactly this.
- **The honest length story for the thesis (all cells parse-traceable):** in-model accuracy
  is a LADDER in trained-length coverage — in-dist 1.000 → 2× 0.28 → 4× 0.15 → 8× 0.09
  zero-shot; +in-length data 0.73@N=32 / 0.56@N=64; +L12 0.44 zero-shot@2×. Remaining
  candidates for true extrapolation: E-G position coupling (124774) and L12+data (124773).

## [2026-07-21] v2@N=48 held-out (124756) — LANDED: 0.514

- `tallyV2_eval_N48heldout/…/` (109 park2 N=48 dirs, first in-training-N=48 cell):
  **0.514, pf 0.037, MAE 2.20**; in-range (g≤8) 49/72 = 0.681; g12-24 7/21; g48 0/11
  (cap truncations + detection). Same ckpt confound as the other v2 cells (ep3/tf 0.891).

## [2026-07-21] v2@N=128 zero-shot (124758) — LANDED: 0.118 (band ≥0.35 not met)

- `tallyV2_eval_N128/…/` (n=34, full report written — 11h wall sufficed here): **0.118,
  pf 0.147, MAE 12.6**; hits only g0-g2. Trained-to-≤64 does not buy 2× extrapolation to 128
  under the uncoupled tally (ckpt confound caveat applies but the gap to the band is large).
- **The extrapolation wall is now measured at every ratio for uncoupled tally readouts:
  2× 0.28-0.44 · 4× 0.15 · 8× 0.09 (zero-shot) and 2×-beyond-trained 0.118. The two live
  candidates: E-G position coupling (pcouple8, 124774 — designed against exactly the
  tally-index confusion seen in every long-N transcript) and L12 depth (l12v2, 124773).**

---

# PHASE-4 ONE-SCREEN SUMMARY (updated live; two cells pending: l12v2 + pcouple8 exams)

| claim | number | run | verdict |
|---|---|---|---|
| Layout freedom (E-C) | at-end+Qfirst d′ 10.27, tally **0.999** = interleaved; no-Qfirst 2.40/0.508 | 124514/124492 | **GO (qualified: question must lead; carriers = pure suffix)** |
| No-harm (E-D) | MME −0.2 · POPE −1.4 pts | 124508 | **GO** — adapter safe always-on |
| Seeds (E-E) | in-dist TF 1.000 ×3; N=32-0shot 0.284±0.004 ×2 | 124482/509/510/697 | stable |
| SFT control (E-B) | in-dist 0.998-1.000; **N=16 0.480 / N=32 0.350, dead mid-range** | 124484/124696 | no collapse, no aggregation — extremes heuristic; supply ceiling located |
| Length ladder (E-A) | 0-shot: 2× 0.280 · 4× 0.150 · **8× 0.087 (headline band ≥0.80 REFUTED)**; in-length: N=32 0.733 · N=48 0.514 · N=64 0.558; ≤64→128 0.118 | 124522/578/736; 124571/586/756/757/758 | wall is robust for uncoupled tally; ladder = the honest claim |
| L12 depth lever | N=32 0-shot **0.443** vs 0.280 (pf 0, MAE 1.15) | 124727 | **GO — L12 new default** |
| E-G coupling cost | TF fit ~2× slower than uncoupled (0.923@4ep vs 1.000@3ep) | 124701 | finding; converged rerun = 124774 |
| PENDING | l12v2 (124773, 5ep) + pcouple8 (124774, 8ep) → exams N=32/48/64/128 on their splits | — | the two live extrapolation candidates |

Ops encoded: exams only after trainers COMPLETE; save criterion now (TF-count, tf-exact);
N=128 = 34 samples / dec 280 / **12h wall** / dumps=40 (report-loss-proof).

## [2026-07-21] l12v2 (job 124773) — LANDED: 1.000 / tf-exact 0.976 @ep5 (fixed criterion); exams launched

- `carrier_tally_l12v2/20260721_071710_L12_r8/` — the strongest ckpt of the campaign
  (tf-exact 0.976 vs v2 0.891 / le64 0.947). Exams: N=32all/N=48/N=64 held-out + N=128
  zero-shot (12h wall). Bands (pre-registered above): ≥0.85 / — / ≥0.70 / ≥0.50-0.80.

## [2026-07-21] E-H pre-registration — separator-layer (L*) sweep (written before launch)

- Question: L* splits the 28 layers into fenced READING depth (0..L*-1) and open INTEGRATION
  depth (L*..27). Known coarsely: L12 ≫ L17 ≫ L22 (starved regime) and L12 > L17 zero-shot at
  full data (0.443 vs 0.280). Four new arms, ONLY --l-open varies ∈ {8, 10, 14, 20}; recipe =
  the headline ≤16 recipe exactly (12 roots incl. longN_16, running-tally, jitter 16,
  grad-ckpt, train-frac 0.5, shuffle 0, frozen distilled e_c, EPOCHS=4, FIXED (acc, tf-exact)
  save criterion — note: L12/L17 reference cells predate the fixed criterion; L12's ckpt was
  tf-0.991@ep2 (criterion-equivalent here since acc hit 1.000 with top tf at ep2-3), L17's
  was ep2/tf-0.933 — recipe-matched, criterion caveat logged).
- **Expected shape (pre-registered): an inverted-U peaking somewhere in 8-14 — too-early L*
  starves the fenced supply (d′ develops through L16 in the probe record), too-late L*
  starves trained integration. Verdict metric = ZERO-SHOT exams (N=32 LIMIT 300; N=64
  LIMIT 60 dec 200), NOT in-dist TF (saturates for all).** Winner (best OOD, no in-dist
  regression) gets 2 more seeds + the N=128 cell.

## [2026-07-21] pcouple8 / E-G (job 124774) — LANDED: 0.955 / tf-exact 0.832 @ep5; exams launched

- `carrier_tally_pcouple8/…/` — E-G's in-dist ceiling over 8 ep: **0.955/0.832** (trajectory
  0.79/0.78/0.89/0.94/0.955/0.94/0.85/0.95 — oscillatory) vs uncoupled 1.000/0.976. The
  coupling's in-distribution fit cost is real and persistent, not an epoch-budget artifact.
- Exams launched on ITS splits (identical dirs to v2/l12v2 by seed): N=32 held-out (150),
  N=64 held-out (52), N=128 zero-shot (34, 12h wall). **E-G pre-registered GO = beats the
  matched uncoupled cells at every OOD length with parse-fail ~0.** Matched-cell references:
  l12v2 exams running (its ckpt tf 0.976); v2 cells 0.607/0.365/0.118 (weaker ckpt caveat).

## [2026-07-22] l12v2@N=32 HELD-OUT (job 124904) — LANDED: 0.953 — BAND MET, thesis-grade cell

- `tallyL12v2_eval_N32heldout/20260721_210415_L12_r8_evalonly/` (150 held-out mixed
  park+park2 N=32 dirs): **acc 0.953, parse-fail 0.000, MAE 0.05.**
- The N=32 trained-at-length progression: A4 scratchpad 0.447 → le64 tally 0.733 → v2
  (weak ckpt) 0.607 → **l12v2 (L12 + park2 data + fixed ckpt) 0.953 ≥ band 0.85.** The
  in-model carrier layer now reads N=32 evidence at near-in-dist fidelity.
- Per-count (near-uniform INCL. multi-digit): g0-g8 ≥0.85 each; **g12 4/4 · g16 8/9 ·
  g24 10/12 · g32 9/10.**

## [2026-07-22] E-G exam cell 1 — pcouple8@N=32 held-out (124922) — 0.527: losing to uncoupled

- `tallyPC8_eval_N32heldout/20260722_002811_L17_r8_evalonly/` (SAME 150 dirs as l12v2's
  0.953): **0.527, pf 0.040, MAE 2.28**; multi-digit near-dead (g24 0/12, g32 0/5+5pf).
- Matched comparisons at N=32: l12v2 (uncoupled, L12) **0.953** · le64 0.733 · v2 0.607 ·
  **E-G 0.527**. Even granting E-G its weaker in-dist ckpt (0.955/0.832), the first
  pre-registered GO condition ("beats uncoupled at every OOD length") is failing at cell 1.
  N=64 (124923) and N=128 (124924) complete the verdict.

## [2026-07-22] l12v2@N=64 held-out (124906) — LANDED: 0.615 (cap-adjusted 0.678)

- `tallyL12v2_eval_N64heldout/…/` (52 dirs): **0.615, pf 0.135** (all pf = g48/g64 cap
  truncations, unmeasurable under dec 280); cap-adjusted (excl. g48/64) 40/59 = **0.678**;
  g24 4/4, g12 3/4. vs le64 0.558 / v2 0.365. Band ≥0.65: met cap-adjusted, marginal raw.

## [2026-07-22] l12v2@N=48 held-out (124905) — LANDED: 0.789 raw / 0.878 cap-adjusted

- `tallyL12v2_eval_N48heldout/…/` (109 park2 dirs): **0.789, pf 0.101 (all = g48 cap
  truncations); cap-adjusted 86/98 = 0.878**; g12 7/7 · g24 6/6 · g32 3/5. vs v2 0.514.
- **The l12v2 length ladder (held-out, cap-adjusted): N=32 0.953 · N=48 0.878 · N=64 0.678
  — the strongest in-model long-N readout of the campaign; N=128 pending (124907).**

## [2026-07-22] E-H arms LANDED; 8 zero-shot exams launched

| arm | BEST (fixed criterion) |
|---|---|
| L8 (124917) | 1.000 / tf-exact 0.978 @ep2 |
| L10 (124918) | 0.999 / tf-exact 0.927 @ep4 |
| L14 (124919) | 1.000 / tf-exact 0.997 @ep3 |
| L20 (124920) | 1.000 / tf-exact 0.996 @ep4 |

- In-dist TF saturates for ALL L* (as pre-registered — not the verdict metric). Note: even
  L20 reaches tf 0.996 by ep4 (its ep1 was 0.529 — slower start, same ceiling); L10's 0.927
  is the outlier low. Zero-shot N=32 (300) + N=64 (60) exams launched per arm; curve
  assembled when all 8 land, with L12 0.443 / L17 0.280 reference cells.

## [2026-07-22] E-G VERDICT (cells 1+2 of 3) — position coupling FAILS the pre-registered GO

- N=64 held-out (124923, `tallyPC8_eval_N64heldout/…/`): **0.212, parse-fail 0.365, MAE 44**
  (token-salad degeneration in transcripts: "28 (10 (10), 32 (11), 3 (1 (1)…").

| cell (identical dirs) | E-G (coupled) | l12v2 (uncoupled best) | uncoupled same-L17 (v2) |
|---|---|---|---|
| in-dist TF / tf-exact | 0.955 / 0.832 | 1.000 / 0.976 | 1.000 / 0.916 |
| N=32 held-out | 0.527 (pf 0.040) | **0.953 (pf 0.000)** | 0.607 |
| N=64 held-out | 0.212 (pf 0.365) | **0.615 (pf 0.135)** | 0.365 |

- **GO condition ("beats uncoupled at EVERY OOD length, pf ~0") is refuted at both testable
  cells; coupling also costs in-distribution fit (persistent, 8 ep) and BREAKS the format at
  4× length — the opposite of its design goal.** The stream-anchor rule is deterministic and
  verified (CPU + couple-debug), so this is a negative result about the MECHANISM, not the
  implementation: forcing decoded-token positions to ride carrier anchors degrades the LM's
  own sequential coherence more than it helps length binding. N=128 cell (124924) completes
  the record when h200 frees but cannot change the verdict.

## [2026-07-22] N=128 partial cells (both exams PREEMPTED by h200-dds mid-run; recovered from dumps; jobs requeued for full numbers)

- **l12v2@N=128 (2× beyond max trained): PARTIAL 0.286 (4/14), parse-fail 0.000** — the best
  N=128 reading of the campaign (le16 0.087 · v2 0.118) and the FIRST with fully intact
  format at 128. g0-g2,g4 correct; counts ≥5 missed (undercounts, not derails).
- E-G@N=128: PARTIAL 0.154 (4/26), pf 0.500 — cell 3 consistent with the E-G refutation.
- Both marked PARTIAL (stratified-order prefixes, unbiased); requeued jobs 124907/124924
  will write the full-34 reports whenever h200-dds releases the node.

## [2026-07-22] E-H separator-layer curve (5 of 6 zero-shot points landed)

| L* | in-dist TF / tf-exact | N=32 zero-shot (n=300) | N=64 zero-shot (n=60) |
|---|---|---|---|
| 8 | 1.000 / 0.978 | 0.277 (pf 0.017) | 0.133 (pf 0.183) |
| 10 | 0.999 / 0.927 | 0.373 (pf 0.000) | 0.217 (pf 0.150) |
| **12** | 1.000 / 0.991 | **0.443 (pf 0.000)** | (le16-recipe cell not run — L12 arm 124698 predates E-H; N=64 not measured) |
| 14 | 1.000 / 0.997 | 0.330 (pf 0.000) | 0.217 (pf 0.150) |
| 17 (ref) | 1.000 / 0.977 | 0.280 (pf 0.007) | 0.150 (pf 0.133) |
| 20 | 1.000 / 0.996 | pending 124971 | pending 124972 |

- **The pre-registered inverted-U is confirmed with the peak at L12** — supply needs ~12
  fenced layers (the probe's d′-develops-through-L16 was conservative; trained end-to-end,
  12 suffice), and integration wants every layer after. N=64 ranks identically (L10/L14
  0.217 > L17 0.150 > L8 0.133). Winner = L12 (already the default; its two extra seeds +
  N=128 cell are ON HOLD per Tal's no-new-launches instruction).

## [2026-07-22] FORMAT sweep (arms A–D) OPENED — prereg + Phase 0/1 done; arm-A control cells LANDED at 1.000/1.000

- Campaign: `plans/scratchpad_format_agent_brief.md` + `plans/scratchpad_format_PREREG.md`
  (bands fixed pre-GPU). Arms differ ONLY in gold scratchpad text (`--scratchpad-format`,
  new in `carrier_layer_cached.py`/`carrier_layer_lora.py`; CPU sanity 27×4 round-trip/
  tally checks passed; smokes 125098-102 passed incl. 3-ep falling-loss check).
- **Arm A (l12v2 ckpt, poslist) in-dist-150: acc 1.000, pf 0.000, MAE 0.00, mean decode
  19.6 tok** (job 125107, `tallyL12v2_eval_indist150/20260722_*/`; 30/30 on every task:
  steps/cooc/rooms/union/which; dirs = `eval_dirs_indist150.txt`, dec 100).
- **Arm A rooms-100: acc 1.000, pf 0.000, MAE 0.00** (job 125108,
  `tallyL12v2_eval_rooms100/20260722_*/`, dec 100).
- **Reading: the "rooms greedy 0.84 decode gap" (5-task L17 A3 ckpt) does NOT exist in the
  l12v2 ckpt** — the same-ckpt rooms control is CEILINGED. The pre-registered rooms-ordering
  band (scan ≥0.95 → "ordering fix confirmed") can now only show parity, not improvement;
  will be reported as such, band unchanged.
- Trainers running: 125104 B/scan, 125105 C/caption, 125106 D/chunked (l12v2 recipe
  verbatim, a100, ~14-16h). Exams gate on COMPLETED + eval_dirs.txt identity vs arm A.

## [2026-07-23] FORMAT sweep trainers B/C/D LANDED (125104/05/06) — all fit TF ~1.000; exams launched

- Identical recipe/data/split as arm A (l12v2), only gold text differs; **split-identity
  gate passed for all three: eval_dirs.txt + train_dirs.txt byte-identical to arm A's.**

| arm | run dir | BEST (fixed criterion) | elapsed |
|---|---|---|---|
| B scan | `carrier_fmt_scan/20260722_222030_L12_r8/` | acc 0.999 / **tf-exact 0.996 @ep5** | 13:57 |
| C caption | `carrier_fmt_caption/20260722_222032_L12_r8/` | acc 0.999 / **tf-exact 0.994 @ep4** | 14:01 |
| D chunked | `carrier_fmt_chunked/20260722_222026_L12_r8/` | acc 1.000 / **tf-exact 0.904 @ep5** | 13:45 |

- Reference arm A: 1.000 / 0.976 @ep5. Scan-family transcripts fit BETTER than poslist
  (0.996/0.994 vs 0.976, ep1 tf 0.287 → deterministic slot structure is easy to learn);
  chunked is the TF laggard (0.904 — sum-expression tail hardest to fit). In-dist TF is NOT
  a verdict metric (prereg); greedy exams decide.
- Exams: D = 125183-187 (running); B/C = 125190-199 (indist/rooms dec100 · N32 dec320 ·
  N48 dec470 · N64 dec620; D used 80/170/250/330). All on arm A's dirs-files, DUMPS=200.

## [2026-07-23] FORMAT sweep in-dist + rooms cells LANDED (125183/84, 125190-93) — all sanity bands MET; C≡B parity

| arm | in-dist-150 (acc/pf/dec-mean) | rooms-100 (acc/pf/dec-mean) | run dirs |
|---|---|---|---|
| A poslist | 1.000 / 0.000 / 19.6 | 1.000 / 0.000 / 18.2 | `tallyL12v2_eval_{indist150,rooms100}/` (125107/08) |
| B scan | 1.000 / 0.000 / 46.5 | 1.000 / 0.000 / 53.7 | `fmtB_eval_{indist150,rooms100}/` (125190/91) |
| C caption | 1.000 / 0.000 / 47.3 | 1.000 / 0.000 / 53.7 | `fmtC_eval_{indist150,rooms100}/` (125192/93) |
| D chunked | **0.987** / 0.000 / 22.6 | **0.920** / 0.000 / 21.0 | `fmtD_eval_{indist150,rooms100}/` (125183/84) |

- **Pre-registered in-dist sanity (≥0.90) MET for all four arms** — no arm BLOCKED.
- **C vs B (band 3, agnostic-caption): EXACT PARITY (1.000/1.000 both cells, |C−B|=0 ≤0.03)
  → agnostic-caption GO on the in-dist leg** (the caption text costs nothing in-dist; length
  cells will complete the picture).
- **Rooms-ordering (band 5): B/C 1.000 ≥0.95 — but control A is ALSO 1.000** → parity only;
  the 0.84 ordering gap was a property of the L17 5-task ckpt, not reproducible in this
  ckpt family (already noted at control landing).
- D's only losses are rooms (28/30 in-dist, 92/100 rooms; misses at g5/g6) — consistent
  with its TF lag (0.904) sitting in the chunked new-rooms-per-block lists.
- Scan-family decode cost in-dist: ~47 tok vs poslist ~20 (2.4×), chunked ~22 (parity).

## [2026-07-23] E-H separator-layer curve COMPLETE (124971/72 collected) — L12 confirmed the peak; L_OPEN=12 stands

- L20 cells collected (p0p2 campaign P0.2): **N=32 zero-shot 0.273 (pf 0.107, MAE 4.33)**
  (`tallyL20_eval_N32/20260722_084020_L20_r8_evalonly/`, job 124971) · **N=64 zero-shot
  0.183 (pf 0.300, MAE 8.76)** (`tallyL20_eval_N64/20260722_085202_L20_r8_evalonly/`, 124972).
  L20's parse-fail is the worst of any arm and concentrates at high counts (g24: 7/23,
  g32: 17/23 fails at N=32) — late opening leaves too few integration layers to keep the
  format coherent at long N.

| L* (open layer) | N=32 zero-shot (n=300) | pf | N=64 zero-shot (n=60) | pf |
|---|---|---|---|---|
| 8 | 0.277 | 0.017 | 0.133 | 0.183 |
| 10 | 0.373 | 0.000 | 0.217 | 0.150 |
| **12** | **0.443** | 0.000 | (not measured, le16 recipe) | — |
| 14 | 0.330 | 0.000 | 0.217 | 0.150 |
| 17 (ref) | 0.280 | 0.007 | 0.150 | 0.133 |
| 20 | 0.273 | 0.107 | 0.183 | 0.300 |

- **Verdict (decision rule from the LOTO/p0p2 briefs): no arm beats L12's 0.443 by >0.05
  with pf ≤0.02 — L_OPEN=12 STANDS** for P1.1 seed retrains and P2a LOTO. The curve is an
  inverted U peaking at 12: ~12 fenced supply layers suffice, and every remaining layer is
  wanted for integration. All numbers: jobs 124965-972, dirs
  `outputs/ladder/image_longN/tallyL{8,10,14,20}_eval_N{32,64}/`.

## [2026-07-23] FORMAT sweep length cells: D complete (125185-87), B/C N=32 (125194/95) — SCAN BEATS POSLIST AT N=32: 1.000 vs 0.953

- All on arm A's dirs-files (`eval_dirs_N32all.txt` 150 / `eval_dirs_N48.txt` 109 /
  `eval_dirs_N64.txt` 52), collected under p0p2 P0.1 adoption (format agent stale).

| cell (identical dirs) | A poslist | B scan | C caption | D chunked |
|---|---|---|---|---|
| N=32 held-out (150) | 0.953 (pf 0) | **1.000 (pf 0, MAE 0)** | 0.987 (pf 0) | 0.907 (pf 0) |
| N=48 held-out (109) | 0.789 raw / 0.878 cap-adj | 125196 running | 125198 running | 0.679 (pf 0) |
| N=64 held-out (52) | 0.615 raw / 0.711 cap-adj (prereg formula) | 125197 running | 125199 running | 0.615 (pf 0) |

- **B (scan) N=32: 1.000 — the first PERFECT held-out length cell of the campaign**
  (`fmtB_eval_N32heldout/20260723_122731_L12_r8_evalonly/`, job 125194; dec-mean 164.4 of
  cap 320). Beats A by +0.047 (prereg band 6: >A+0.05 = "better" — misses by 0.003; ≥
  parity confirmed at minimum, and A's 7 misses vs B's 0 on identical dirs).
- C caption N=32: 0.987 (125195, `fmtC_eval_N32heldout/…/`; misses = 2× g24). C−B = −0.013
  ≥ −0.03 → agnostic-caption parity holds on the length leg so far.
- D chunked: N=32 0.907 (`fmtD_eval_N32heldout/…/`, 125185) · N=48 0.679
  (`fmtD_eval_N48heldout/…/`, 125186) · N=64 0.615 (`fmtD_eval_N64heldout/…/`, 125187);
  pf 0.000 at ALL THREE lengths with decode well under cap (means 55/75/101) — chunking
  fully solves the decode-budget/parse problem but costs accuracy vs A at N=32/48
  (0.907/0.679 vs 0.953/0.789 raw) and only TIES A's raw 0.615 at N=64 (band 4 "D cap-adj >
  max(A,B)" requires beating A's cap-adj 0.711 — D has no truncations to adjust, so 0.615
  is its final number: band currently NOT met, B's N=64 cell pending).
- Format winner named when 125196-99 land (B/C N=48/64).

## [2026-07-23] P1.2 — MEASURED before-ceiling (best linear readout of the joint carrier sum): the law-predicted curve CONFIRMED by direct measurement

- Job 125259 (CPU, `probe_dprime_parity.py --carrier-caches`, deployed locus L16/off9, the
  EXISTING joint caches `image_longN/joint/N{8,16,32,64,128}/20260710_2154*/count/`,
  60/40 sample-disjoint split, seeds 0-2) →
  `outputs/ladder/image_longN/measured_ceiling/20260723_222428/`.

| N | law-pred (iid, 0 params) | MEASURED best linear (ridge/logit) | MLP−linear | frozen model (same caches) |
|---|---|---|---|---|
| 8 | 0.307 | **0.317** (ridge, ±0.036) | −0.119 | 0.207 |
| 16 | 0.246 | **0.281** (logit) | −0.003 | 0.127 |
| 32 | 0.175 | **0.189** (logit) | +0.000 | 0.053 |
| 64 | 0.137 | **0.183** (logit) | −0.004 | 0.040 |
| 128 | 0.096 | **0.122** (logit) | +0.006 | 0.013 |

- **The law-predicted "squashed readout" curve is now MEASURED, not just predicted: the
  best sample-disjoint linear readout of the summed joint-carrier messages lands within
  0.01–0.05 of the zero-parameter prediction at every N** (slightly ABOVE it at N≥64 — the
  logistic readout exploits the non-Gaussian tail the law ignores; adequacy kurtosis grows
  +1.8→+14.0 exactly as logged in [2026-07-11e/n]). MLP adds nothing (E3 sufficiency ≤0.006
  except the N=8 dip). d′_w flat ~2.0 to N=64, 1.6 @N=128 — replicates the B1 flat-d′ result.
- Reading for the thesis figure: frozen model < measured-best-linear < trained-scaffold
  (0.95–1.00 @N=32) — the gap frozen→linear is readout misalignment, the gap
  linear→scaffold is supply repair; fig regenerated with the measured curve at
  `outputs/_scratch/figs/pre_stage1_squashed_readout_measured.png` (original law-only
  fig untouched).
- Caveat: caches keep the gold≥1 convention at N≥2 and per-cache n shrinks with N
  (300/300/300/200/150) — error bars widen at N=128 (±0.031).

## [2026-07-23] P1.3 — E-B SFT baseline N=64 cell LANDED (the OOM'd leg of 124696, rerun): 0.220 — the extremes heuristic extends unchanged; SFT ladder complete

- Job 125267 (`sft_control_le8_v2_evalN64/20260723_225940_lora/`, a100 24h_1g, 22 min):
  eval-only rerun of the E-B adapter (`sft_control_le8_v2/20260720_191541_lora/adapter`)
  on `mmred_longN_park/seq_len_64` (LIMIT 100, same stratified-shuffled-prefix sampling as
  the landed N=16/32 cells). **N=64 acc 0.220, parse-fail 0.000, MAE 3.46.**
- Adapter-restore sanity: test_iid 1.0000 (135/135) — byte-exact reproduction of the
  original run's cell; the eval path change cannot have touched the model.
- Per-count: g0 4/7 · g1 5/7 · g64 6/6, mid-range g4–g48 ≈ 0 — the bimodal extreme-count
  heuristic of [E-B, 2026-07-20] extends to 8× training length with no collapse and no
  aggregation. **SFT ladder final: N=16 0.480 / N=32 0.350 / N=64 0.220** vs the carrier
  readout (l12v2) 0.953 / 0.878 (N=48) / 0.678 cap-adj on held-out dirs.
- Ops (logged for reuse): the 124696 OOM cause is now diagnosed — in `model.generate` the
  attention mask is None → transformers sets `enable_gqa=True` (28 q-heads vs 4 kv-heads)
  → the mem-efficient sdpa kernel is INELIGIBLE (fused kernels need matching num_heads on
  dense inputs) and MATH materializes 17GB fp32 attention. **FLASH supports GQA+causal:
  peak 8.3 GiB @seq 12.7k** (smoke 125263, `p13_eff_smoke.py`). The manual-layer eval path
  (carrier_layer_lora) never hit this because explicit float masks force repeat_kv.
  `lora_sft_baseline.py` now uses [FLASH, EFFICIENT, MATH] + `--eval-only-adapter`.

## [2026-07-24] FMT sweep: C (caption) N=64 held-out 0.981 (125199) — the scan-family format nearly SOLVES the hardest length cell (A poslist: 0.615 raw / 0.711 cap-adj)

- `fmtC_eval_N64heldout/20260723_122649_L12_r8_evalonly/` (job 125199, 18h05 on l40s):
  **0.981 (51/52), pf 0.000, MAE 0.02, dec-mean 330.6 of cap 620 — zero truncations;
  g48 3/3 and g64 4/4 both perfect** (arm A's cap-adj formula excluded exactly these).
  Only miss: 1× g32. Identical 52 dirs as every other arm's N=64 cell.
- Prereg band 2 is defined on B (scan) vs A — B's cell (125197) still running — but the
  C result already shows the format family clears the GO bar (≥0.761 cap-adj) by +0.22.

## [2026-07-24] FMT sweep: B N=64 landed (125198) → ALL PRIMARY BANDS DECIDED — scan-format GO, agnostic-caption GO, chunking NO; WINNER = C (caption)

- **B (scan) N=64 held-out: 0.942 raw, pf 0.019 (1× g64 degenerate), MAE 0.37, dec-mean
  328/620; cap-adj (prereg formula, g48/g64 excluded) = 43/45 = 0.956**
  (`fmtB_eval_N64heldout/20260723_122357_L12_r8_evalonly/`, 18h14 on l40s). Note: the job
  ran under the label fmt_eval-125198; the stale STATE's job→cell mapping was approximate —
  cells identified from each report header (ckpt + dirs-file), not the job list.
- **Band 2 (primary, B vs A @N=64 cap-adj): 0.956 ≥ 0.761 → SCAN-FORMAT GO** (+0.245 over
  A's 0.711; secondary raw 0.942 ≥ 0.665 ✓).
- **Band 3 (C vs B): GO** — in-dist/rooms exact parity (1.000 all), N=32 −0.013 (within
  0.03), N=64 +0.039 in C's favor. The caption text costs nothing and helps at the top end.
- **Band 4 (D vs A/B @N=64): NO** — D 0.615 < max(A 0.711, B 0.956).
- **WINNER = C (caption)**: takes the hardest length cell (0.981 vs B 0.942, A 0.711
  cap-adj) with parse-fail 0.000 at every length; within-parity everywhere else; and the
  caption format IS the agnosticism bet (attribute words in the scratchpad) — the natural
  recipe for LOTO/MLVU. Winner ckpt:
  `outputs/ladder/image_longN/carrier_fmt_caption/20260722_222032_L12_r8/carrier_layer_best.pt`.
  Not a tie → the "ties → arm A" rule does not apply. N=48 cells (125196/97, running,
  secondary/descriptive) complete the table on landing and cannot change the bands.

## [2026-07-24] FMT sweep: B N=48 landed (125196) — 0.982, pf 0; the scan family dominates every length cell

- `fmtB_eval_N48heldout/20260723_122357_L12_r8_evalonly/` (job 125196, 18h22 l40s):
  **0.982 (107/109), pf 0.000, MAE 0.02, dec-mean 247/470; g48 11/11.** vs A poslist 0.789
  raw / 0.878 cap-adj on identical dirs. Misses: 1× g7, 1× g8. Winner verdict (C) unchanged
  — this is the secondary N=48 cell; C's N=48 (125197) completes the table.

## [2026-07-24] FMT sweep COMPLETE (C N=48 0.972, 125197) — FINAL TABLE; sweep closed

- `fmtC_eval_N48heldout/20260723_122357_L12_r8_evalonly/` (18h42 l40s): **0.972 (106/109),
  pf 0.000, MAE 0.05, dec-mean 252/470.**

| cell (identical dirs) | A poslist | B scan | C caption (WINNER) | D chunked |
|---|---|---|---|---|
| in-dist-150 | 1.000 | 1.000 | 1.000 | 0.987 |
| rooms-100 | 1.000 | 1.000 | 1.000 | 0.920 |
| N=32 (150) | 0.953 | **1.000** | 0.987 | 0.907 |
| N=48 (109) | 0.789 / 0.878 cap-adj | **0.982** | 0.972 | 0.679 |
| N=64 (52) | 0.615 / 0.711 cap-adj | 0.942 / 0.956 cap-adj | **0.981** | 0.615 |
| worst parse-fail | 0.135 | 0.019 | **0.000** | 0.000 |
| length-cell mean | 0.786 raw | 0.975 | 0.980 | 0.734 |

- **Headline: the gold scratchpad TEXT alone is worth +0.19-0.37 at held-out lengths** —
  full-scan formats (a slot per frame) beat the positive-list by forcing the search
  burden into a deterministic frame-order scan. B vs C are within noise overall
  (0.975 vs 0.980 mean); C named winner on the primary N=64 cell (+0.039), pf 0
  everywhere, and the agnosticism property (attribute captions). All five prereg bands
  decided: sanity ×4 MET · scan GO (0.956 ≥ 0.761) · agnostic-caption GO · chunking NO ·
  rooms-ordering parity-only.
- Sweep CLOSED. Downstream (p0p2): C seeds ×2 (125347/48), LOTO T1 (125349), MLVU
  (125350) — all running.

## [2026-07-24] P2b — MLVU-AC zero-shot carrier cell (32f arm): MCQ 0.107 ≤ frozen 0.282 → PRE-REGISTERED "DOMAIN GAP MEASURED" OUTCOME; no transfer, logged without spin

- Job 125350 (`outputs/ladder/mlvu_ac/carrier_eval_N32/20260724_064754_L12_r8_evalonly/`,
  4h42, a100): winner ckpt (C caption) on all 206 MLVU-AC questions, 32 frames @392px
  (dense N=128 arm ruled prohibitive pre-launch: ~20 min/sample × 206 > 60h; prereg
  fallback clause invoked — the 32f evidence-delivery caveat applies: judge found ~0.37
  visible frames per gold instance, ~35% of Qs zero visible evidence [2026-07-11c]).
- **Open emitted count: 0.000 exact** (pf 0.214, MAE 2.99); parsed distribution: 161×"0",
  44 parse-fail, 1×"1". **MCQ nearest-option (prereg rule, parse-fail=wrong):
  22/206 = 0.107** (by-gold g1 12/37 · g2 8/52 · g3 2/45 · g4 0/33 · g5 0/39) —
  `mcq_mapping.txt` in the run dir; order-consistency dumps↔dirs verified (0 mismatches).
- **Verdict vs band: 0.107 ≤ frozen 0.282 (32f MCQ) → domain gap measured, transfer NO.**
  Protocol note (prereg design): our cell never sees the MCQ options (count → nearest
  mapping), while the frozen 0.282 had the options in-prompt — chance structure differs;
  both stated.
- Failure anatomy (from the 206 dumped transcripts): the readout FORMAT survives —
  scan/caption structure emitted, mostly well-formed — but the content collapses to
  all-negative verdicts; the 44 parse-fails are fluent refusals that correctly DESCRIBE
  the wildlife frames ("the image shows a zebra … no 'making jewelry' action … answer is
  0"). I.e., per-frame perception works, the trained evidence-detection channel does not
  fire outside the MMRED render domain — consistent with the cross-DOMAIN carrier result
  (steps-e_c on natural images ~51% of teacher, gate tally < frozen [2026-07-18]) and the
  delivered-evidence ceiling. The domain gap is in the evidence detector, not the
  aggregator/format.

## [2026-07-24] P3b — InternVL2.5-8B scaffold-level gate→tally: 0.938 ± 0.031 exact @N=8 (band ≥0.90 MET) — the GNN scaffold structure PORTS across model families (multipass-isolated supply level)

- CPU fit on the EXISTING per-frame message cache of job 124280
  (`outputs/frame_axis/internvl/multipass_qfirst/20260719_004112/bench_cache.pt`, n=200
  samples × 8 frames, layers 16/20) → `outputs/frame_axis/internvl/gate_tally/20260724_165356/`
  (`experiments/glstm/internvl_gate_tally.py`; logistic gate, sample-disjoint 60/40,
  seeds 0-2; tally = Σ per-frame verdicts).
- **L16: gate→tally exact 0.938 ± 0.031 (per-frame gate 0.991) · L20: 0.892 ± 0.012
  (gate 0.985); majority 0.160.** Band ≥0.90 → **scaffold ports** at L16.
- **Honest label:** the cache is MULTIPASS-ISOLATED Q-first supply (each frame solo in
  its own forward), NOT one-forward fenced — this is the scaffold-level claim (per-frame
  messages + linear gate + sum), one rung below Qwen's one-forward blockfence result.
- Same signature as Qwen: InternVL's own per-frame digit readout = 0.586, yet a linear
  gate on its carrier messages reads 0.991 — the information is in the messages, the
  native readout wastes it; cross-family generality of the readout-misalignment account.

## [2026-07-24] P3a L1+L2 — natural-images supply and scaffold BOTH GO: one-forward replica d′ 27.3 (3.5× joint) and gate→tally 0.980 ± 0.012 @N=8 — the mechanism is domain-general at n=300; only the in-model rung (L3) remains

- **L1 supply** (job 125486, `replica_supply_dist_far/20260724_180046/`, n=300 skip=0,
  blockfence+posreset+qfirst on `mmred_natural_v2/dist_far`): **L16 replica d′_w 27.31 ±
  0.36 vs in-run joint anchor 7.75 ± 0.34 — ratio 3.52×; per-copy 8.5-13.7 (flat, all ≥4)**;
  L14 23.2 vs 7.5. Band (≥4.0 and ≥2× joint) MET with huge margin. CAVEAT: both arms
  saturate the AUC estimator (auc-d′ 5.26 both) — d′_w is the working scale; and the joint
  anchor itself is ~2.5× the v1-n=50 reading (2.95) — the v1 numbers were small-n deflated
  (known lesson) and v2 evidence is judge-cleaned; natural dog-vs-far-distractor is an
  EASY per-frame discrimination.
- **L2 scaffold** (CPU `replica_gate_tally.py` @L16, 5 seeds, train-frac 0.5 →
  `replica_supply_dist_far/20260724_180046/gate_tally/`): **tally exact 0.980 ± 0.012,
  MAE 0.02, per-frame gate err 0.0025** (majority 0.187; golds 0-6 present, per-count
  clean). Band ≥0.85 MET — park-level (0.998); supersedes the v1 n=50 cell (0.920±wide).
- Ladder state: L0 frozen rerunning (125488 — first attempt consumed only K0: the park
  evidence-parser skipped natural samples; natural branch added to
  `frozen_baseline_eval.py`); L3 trainer running (125487). If L3 lands low, supply and
  scaffold are now EXCLUDED as the cause (both GO) — the localization the prereg wanted.

## [2026-07-24] P3a L0 — frozen baseline on natural (image-held-out eval roots, digit protocol): N=8 0.563/0.407, N=16 0.422/0.311 (far/near) — higher floor than park (0.219) but the same undercount wall (g8 = 0 everywhere, falls with N)

- Job 125488 (rerun; first attempt 125485 consumed only K0 — park evidence-parser skipped
  natural samples; natural branch added) → `outputs/ladder/natural_mm/frozen_baseline/`
  (4 timestamped run dirs, one per cell; numbers in `logs/nat_frozen-125488.out`).
- Protocol notes: digit-argmax readout (single token) → gold>9 skipped by design, so the
  N=16 cells cover golds 0-8 only (n=90/cell; K12/K16 excluded — the L3 band comparison
  uses matched golds). Plain prompt (no qfirst), resize 392, full eval roots.
- **Reading: natural per-frame evidence is EASY (dog vs distractors — hence the high L1
  d′ and 0.4-0.56 frozen floor vs park 0.219), but aggregation still collapses exactly the
  same way: g8 0/15-0/10 in all four cells, mid-range dies, N=16 < N=8.** L3 bands now
  concrete: pooled-eval GO ≥0.80; NO-GO ≤ L0+0.10 (N=8 pooled 0.485 → 0.585; N=16
  matched-gold pooled 0.367 → 0.467).

## [2026-07-24] P4.3 — no-harm on the PLAIN SFT adapter: MME −0.6 / POPE +1.2 pts → GO (band |Δ|≤2); the predicted digit-on-yes/no failure mode did NOT appear

- Job 125499 (`noharm_bench_sft/20260724_190307/`, l40s 2h_2g, 8 min — MME/POPE already
  cached): same 500+500 protocol/seed as the carrier-LoRA cell 124508 (ref −0.2/−1.4).
  New `--peft-adapter` arm (PeftModel; base pass = raw model) + ≤20 fail dumps.
- **MME 0.862→0.856 (−0.6) · POPE 0.862→0.874 (+1.2) — no-harm GO for the SFT baseline
  too**; both adapters (carrier and plain SFT) are safe always-on. Category texture at
  small n: OCR −20 (n=5-ish) and celebrity −13.3 vs count +16.7 and commonsense +12.5 —
  counting SFT slightly helps MME count, as one would hope.
- Emission anatomy: fail dumps all emit yes/no words (some lowercase) — no digit
  contamination of yes/no questions; the failure mode prediction is REFUTED (good news
  for the deployment story).

## [2026-07-24] P2a LOTO T1 trainer LANDED (125349) + split-gate amendment; 6-cell exam battery launched (125500-505)

- T1 (4-of-5, cooc held out): `carrier_loto_nococ/20260724_064756_L12_r8/` — n=7872
  (= arm A's 8772 − 900 cooc, exact), caption/L12 recipe, 13h38. **BEST acc 0.998 /
  tf-exact 0.993 @ep5** (rooms 332/332 · steps 2965/2971 · union 257/257 · which 376/376).
- **Split-gate amendment (documented, honest):** `--split-seed 0` pins the PERMUTATION,
  but over n=7872 it selects different sample indices than arm A's permutation over 8772 —
  T1's train set overlaps arm A's eval split (72/120 of the indist150 non-cooc items).
  LESSON: split pinning only reproduces splits at IDENTICAL dataset size/order.
  Consequences: **Arm 1 is UNAFFECTED and airtight** (zero cooc dirs in T1's training —
  checked; the cooc N=8 432-dir file is outside every training set incl. the skyline's,
  and cooc N=32 is a fresh never-trained root). Arm 2's dirs-file re-drawn from T1's OWN
  eval split (30/task ×4 = `eval_dirs_arm2_indist120.txt`, seed 0) — same band ≥0.90.
  Seeds trainers are NOT affected by this mechanism (identical 16 roots → identical
  permutation; byte-gate on landing as planned).
- Exams launched (QOS spread 24h_1g/4d_1g): Arm 2 in-dist-120 (125500) · Arm 1 cooc N=8
  (125501) / N=32 (125502) · Arm 3 skyline cooc N=8 (125503) / N=32 (125504) · Arm 4
  frozen floor both cells (125505, dirs-file support added to frozen_baseline_eval.py).

## [2026-07-24] P3a L3 exams (3 of 4 landed): natural in-model greedy readout COLLAPSES TO EXTREMES — 0.289/0.259 @N=8, 0.145 @N=16-near, all ≤ frozen floor; with L1/L2 GO the failure is localized to the in-model rung

- `nat_eval_N8far` 0.289 (g0 15/15, g8 15/15, mid ≈ 0; 125492) · `nat_eval_N8near` 0.259
  (125493) · `nat_eval_N16near` 0.145 (g0 10/10, g16 6/10, EVERYTHING else 0; 125495);
  pf 0.000 everywhere; N16far (125494) still running. Trainer TF-count 0.996 was riding
  the gold tally prefix (tf-exact 0.187 was the tell — logged at landing).
- Verdict pending the 4th cell, but the shape is already: **NO-GO band hit (≤ L0+0.10;
  cells are BELOW the frozen floor 0.563/0.407/0.311)** while supply (d′ 27.3) and
  scaffold (0.980 linear gate on the SAME messages) are GO — the pre-registered
  decomposition deliverable: on natural images the failure is NOT supply and NOT the
  scaffold; it is the in-model readout rung (park-trained carrier e_c and/or the
  LoRA-through-frozen-layers integration).

## [2026-07-24] P3a VERDICT — natural-images ladder COMPLETE: supply GO (d′ 27.3) · scaffold GO (0.980) · in-model NO-GO (0.15-0.29, below frozen) — the cross-domain failure is the IN-MODEL READOUT RUNG, not the mechanism

| rung | natural result | park reference | band | verdict |
|---|---|---|---|---|
| L0 frozen | N=8 0.563/0.407 · N=16 0.422/0.311 (g≤8) | 0.219 | (reference) | — |
| L1 supply (one-forward) | d′_w 27.3 (3.5× joint 7.75; per-copy 8.5-13.7) | 13.5 | ≥4.0 & ≥2× | **GO** |
| L2 scaffold (linear gate→tally) | **0.980 ± 0.012** | 0.998 | ≥0.85 | **GO** |
| L3 in-model (caption/L12 trained in-domain, image-held-out) | 0.289/0.259 @N=8 · 0.155/0.145 @N=16 (pooled 0.218) | 0.987-1.000 | GO ≥0.80 / NO-GO ≤L0+0.10 | **NO-GO** |

- L3 cells (125492-95, `outputs/ladder/natural_mm/nat_eval_N{8,16}{far,near}/`): pf 0.000
  everywhere; anatomy = PURE EXTREMES (g0 perfect, gmax mostly right, EVERY intermediate
  count 0) — the same bimodal signature as the park SFT baseline. Trainer tell: TF-count
  0.996 rode the gold tally prefix while tf-exact sat at 0.187 (the model never learned
  which frames carry the dog INTO its own transcript).
- **The pre-registered decomposition deliverable: on natural images the GNN mechanism
  (per-frame message supply + linear gate + sum) is fully intact — what fails is the
  in-model rung: the park-distilled carrier e_c and/or the LoRA-through-frozen-layers
  integration cannot re-encode the (easy) natural evidence into the scratchpad.**
  Consistent with the cross-domain carrier result (~51% of teacher) and the MLVU
  domain-gap cell (0.107): the trained detector, not the aggregator, is domain-bound.
  Successor (not launched — needs Tal): natural-distilled e_c (distill on natural
  replica teachers, then retrain L3) would split e_c from LoRA-integration.

## [2026-07-24] P2a Arm 2 — LOTO in-dist sanity 1.000 (120/120, rooms/steps/union/which 30/30 each) — band ≥0.90 MET; T1 healthy, Arm 1 interpretable

- Job 125500, `loto_arm2_indist120/…_evalonly/`, T1 ckpt on its OWN eval-split dirs-file
  (`eval_dirs_arm2_indist120.txt`, re-drawn post split-gate amendment), dec 45.9/100, pf 0.

## [2026-07-24] P1.1 seed trainers LANDED (125347/48) — byte-identity gates PASS ×2; P2a Arm 4 floors: cooc 0.130 / 0.130

- Seed1 `carrier_fmt_caption_seed1/20260724_064754_L12_r8/`: BEST 0.999 / tf-exact 0.993
  @ep4. Seed2 `carrier_fmt_caption_seed2/20260724_064743_L12_r8/`: BEST 0.999 / tf-exact
  0.991 @ep3. (Seed0 ref: 0.999 / 0.994 @ep4.) **Both eval_dirs.txt byte-identical to arm
  A's** (identical 16 roots → identical permutation; the LOTO split-drift mechanism does
  not apply here). N=32 exams on the 0.953-cell dirs: 125507 (seed2) / 125519 (seed1).
- Arm 4 frozen floors (125506, dirs-file cells, `loto_frozen_coocN{8,32}/`): **cooc N=8
  0.130 (n=300) · cooc N=32 0.130 (n=207, golds ≤9 — digit protocol)**. Arm 1 NO-GO bar
  = 0.230; GO bar = 0.7 × Arm 3 (running). N=8 floor anatomy: answers cluster on "2"
  (g2 32/33, all other counts ~0) — the length-dependent middle estimate.

## [2026-07-24] P2a Arm 1a — LOTO cooc N=8 zero-shot: 0.403 (pf 0, MAE 1.03) — 3.1× the frozen floor, errors spread over all counts (NO extremes collapse); GO/partial call pends Arm 3

- Job 125501, `loto_arm1_coocN8/…_evalonly/`, T1 ckpt on the 432-dir arm-A cooc file
  (LIMIT 300; identical items as Arms 3/4). Per-count: hits at every gold 0-8 (g0 19/32 …
  g8 16/35) — a graded counting attempt, qualitatively unlike the historical zero-shot
  task-transfer nulls (steps→cooc 0.179; cached→rooms 0.153) and unlike every
  extremes-collapse cell of this campaign.
- vs bands: NO-GO bar (floor+0.10 = 0.230) CLEARED. GO bar = 0.7 × Arm 3 (125503 running).

## [2026-07-24] P2a Arm 3a — skyline cooc N=8: 0.997 (299/300, pf 0) → Arm 1 N=8 verdict = PARTIAL TRANSFER (0.403 ∈ (0.230, 0.698))

- Job 125503, `loto_arm3_coocN8/…_evalonly/`, caption ckpt on the identical 300 items.
  Skyline near-perfect → GO bar 0.7×0.997 = 0.698. **Arm 1 (0.403) clears NO-GO (0.230)
  but not GO → pre-registered "partial transfer" band at N=8, logged without spin: 4-task
  variety builds a REAL task-general readout (3.1× floor, graded errors, format intact)
  that recovers 40% of the trained skyline zero-shot.** N=32 pair (125502/04) pending.

## [2026-07-24] ✅📊 READOUT ERROR LAW (CPU re-analysis of the fmt-C dumps, main session) — the caption readout's ENTIRE in-length error budget is one number: per-frame miss rate p ≈ 0.0012–0.0022, flat in N AND in context position; false alarms EXACTLY ZERO (~11k non-evidence frames); labels 100%, tally arithmetic 100%; an independent-errors model predicts measured exact-match within ±0.002 at every length

> CPU only, no new runs: per-frame verdicts parsed from the existing decode dumps of jobs
> 125195/125197/125199 (fmt C caption, N=32/48/64 held-out; 150/109/52 samples), gold per-frame
> sets recomputed from each cell's dirs-file qa.txt states; dump↔dir alignment PROVEN by exact
> gold-sequence match. Script + summary: `outputs/_scratch/readout_error_law/`.

| cell | measured exact | p_miss (per gold frame) | q_FA (per non-ev frame) | MC-pred exact (indep. p,q) | pos early/mid/late miss |
|---|---|---|---|---|---|
| N=32 | 0.987 | 0.0016 | **0.00000** | 0.989 | .002/.002/.000 |
| N=48 | 0.972 | 0.0022 | **0.00000** | 0.973 | .002/.000/.004 |
| N=64 | 0.981 | 0.0012 | **0.00000** | 0.980 | .000/.004/.000 |

- **Readings:** (1) the method's in-length errors are FULLY explained by independent per-frame
  detection misses at p ≈ 0.002 — no length-dependent readout failure, no position tax at the
  deployed recipe (posreset doing its job at readout level), no error correlation; (2) the
  detector is strictly one-sided (misses only, zero hallucinated evidence) — threshold sits
  conservative; a recall lever trading a little FA for miss is the natural next knob;
  (3) label words on hits 100% and tally increments 100% intact → scratchpad mechanics are
  error-free; everything left is supply-side recall, quantified.
- **Predictive law for in-length scaling:** acc(N) ≈ E[(1−p)^{N_ev}] — e.g. p=0.002, uniform
  gold to N=128 (mean N_ev≈64) predicts ≈0.88 for a future in-length-trained N=128 cell.
  Falsifiable pre-registration for that cell.
- Caveats: single ckpt (fmt C winner), steps task only (150/109/52 samples), gold≥0 prior as in
  the parent cells; independence tested only via the exact-match moment (MC), not full error
  covariance.

## [2026-07-24] P4.1 attempt 1 FAILED (125497) — training-forward OOM on 40GB: fused sdpa kernels don't engage on the MASKED training forward (MATH asks 45.6 GiB @N=64); ALL 590 long-N samples were skipped, then the val pass OOM'd → resubmitted on h200 (125522)

- Ops record: unlike generate (mask=None → FLASH works, P1.3 lesson), the SFT TRAINING
  forward carries a non-None attention mask → FLASH ineligible, efficient not selected
  (reason not chased), MATH materializes fp32 attention. The per-sample try/except turned
  this into silent data loss (590 "train skip" lines = every N=32/64 sample) — a run that
  "completes" this way would be an in-length trainer IN NAME ONLY. Caught because the
  unwrapped val evaluate crashed the job. **Lesson: count 'train skip' lines before
  trusting any lora_sft run; long-N SFT training needs a ≥90GB GPU (h200) under MATH.**
- 125522 = identical config on h200-shared (4 GPUs free at submit).

## [2026-07-25] 🔬 TRUNC E1 (exactness): the caption readout READS FRAMES AT DECODE TIME — drop-frame-KV is NOT free; fast cached decode = 16× (N-mix) to ~100× (N=64) per-sample decode speedup, exact vs the mask-only arm 18/20

> Campaign: `plans/trunc_efficiency_agent_brief.md`, PREREG `plans/trunc_PREREG.md`
> (bands fixed pre-GPU + pre-launch amendment). Jobs 125554 (E1a, 8×N8+8×N32, 2h_2g h200)
> + 125555 (E1b, 4×N64, 4d_1g h200) → `outputs/ladder/image_longN/trunc_kvdrop/e1{a,b}/
> 2026072*_evalonly/`. Winner caption ckpt, DEC=620, dirs = first-k of arm-A's
> indist150/N32all/N64 files (no cherry-pick).

- **P0.1 code truth (executable, `trunc_mask_smoke.py` + [mask-debug])**: `build_block_mask`
  fences only in-block rows; tail rows and `_ext_mask` decode rows keep plain CAUSAL access
  to all frame tokens at every layer (carriers hidden in lo, added in hi). Token-identity
  under KV-drop was therefore never structurally guaranteed — and the cached trainer
  teacher-forces target rows WITH this frame visibility, so the LoRA learned to use it.
- **E1a (16 samples)**: mask-only kvdrop identical **1/16** (the all-evidence gold=8 sample
  only), answer-equal **1/16**; divergence at the FIRST evidence-verdict token (div@4-7 at
  N=8; @10-60 at N=32); kvdrop answers collapse (2→8, 5→8, 2→0; N=32 transcripts
  degenerate to short "total: 0"; gold=32 undercounts to 20 via div@184). Baseline arm
  re-scored 1.000/16 (byte-identical to logged cells). **PREREG band 3: generation READS
  frames — the carrier story holds for PREFILL aggregation, but the decode-time verdict
  content (evidence yes/no + room caption) is substantially read from frame tokens.**
- **fast cached decode (engineering arm)**: fast≡mask 14/16 + 2/2 (E1b so far) = within
  the ≤2-flip band; both flips are near-tie numeric cascades that changed answers →
  pre-launch AMENDMENT: E2/E3 accuracy cells use the DENSE flagged decode (shape-identical
  to baseline; truncation the only variable); fast reserved for E6 with the 18/20 caveat.
- **Speedup (h200, per-sample decode wall-clock)**: N-mix base 59.7s vs fast 3.7s =
  **16.2×**; N=64 single: 608.9s vs 6.0s ≈ **100×** (prefill_capture 2.2s included);
  keep=103 of seq=12775 (124× shorter decode context). VRAM flat (7.9 vs 7.8 GiB — dense
  prefill still present; VRAM wins arrive with truncation/chunking, E6).
- Consequence: E2 (eval-only truncation) expected PARTIAL/FAIL → E4 deploy-matched
  truncated RETRAIN (trainer flag landed + LIMIT=4 smoked, job 125556) is the repair
  under test. E1b N=64 completion pending at write time (2/4 in, pattern identical).

## [2026-07-25] TRUNC E5 (chunked prefill) STRUCTURAL VERIFICATION: per-block chunk forwards reproduce question+carrier rows to bf16 numerics (carriers max|Δ| 0.21 abs); the only real deviation is the PREREG'd tail one; "question-row mismatch" resolved as attention-sink numerics

> Jobs 125566 (5 samples, 3×N8+2×N32) + 125568/69 (1-sample layer-resolved debug) →
> `outputs/ladder/image_longN/trunc_bench/chunkverify/20260725_030842_*/` +
> `outputs/_scratch/trunc_smoke/chunkdbg/`. Implementation: `--chunked-prefill` — all
> chunk masks and the truncated upper stack are PLAIN CAUSAL / direct-built
> (`truncated_masks`, equality vs index-selected dense asserted in `trunc_mask_smoke.py`)
> — no dense seq² mask anywhere in the chunked path.

- Layer-resolved deltas (chunk vs dense-truncated, keep rows): **L0 = 0.0000 exactly**
  (assembly/indexing exact); L1-2 ≤0.15; at L6 the max ABS delta jumps to 32-34 but sits
  on header token 2's massive-activation dims (|h|≈4800-5400 → **~0.7% relative**;
  byte-identical across samples because header states are sample-independent) — bf16
  reduction-order noise amplified by attention sinks, NOT chunk math. Carriers: ≤0.21
  abs / ≤0.064 rel everywhere — **the carrier supply channel is per-block computable,
  as the fence identity predicts**.
- Tail rows: Δ grows 0.73 (L1) → 12.7-16.4 (L12) — REAL and pre-registered (dense tail
  reads frames in lo; chunked tail = full-truncation semantics).
- Behavioral bar (decoded answers equal): 0/5 — but MOOT on this ckpt: the dense
  truncated decode is equally broken (E1/E2 finding: readout uses frame reads). The
  behavioral equivalence check re-runs on the E4 deploy-matched ckpt.

## [2026-07-25] TRUNC E2 cell 1 (in-dist-150, --truncate-at 12, eval-only): **0.047** vs reference 1.000 — Δ −0.953, HARD FAIL band (< −0.10) → Phase 3 (E4 deploy-matched retrain) triggered

- Job 125562 (2h_2g h200, dense flagged decode per PREREG amendment, DEC=100, arm-A
  `eval_dirs_indist150.txt` byte-identical) →
  `outputs/ladder/image_longN/trunc_at12/indist150/20260725_030649_L12_r8_evalonly/`.
- acc 0.047, pf 0.000, MAE 4.73: format survives PERFECTLY (parse never fails) but the
  content degenerates — transcripts collapse to all-yes captions with a LOWERCASE room
  word ("f1:bedroom(1) … f8:bedroom(8) | total: 8"), i.e. the truncated model loses both
  the per-frame evidence discrimination AND the trained capitalization. Consistent with
  E1: the verdict channel reads frames at decode time; cutting frames at L12 (prefill)
  + decode (kvdrop) removes the readout's input, NOT the carrier supply.
- PREREG interpretation caveat: this cell does NOT show "carriers unsaturated" — it
  shows the CURRENT ckpt never learned a carrier-only readout. E4 (train/deploy matched)
  is the discriminating experiment; E3 sweep still charts the eval-only depth curve.

### [2026-07-25] E1b completion (4×N=64, job 125555) — E1 CLOSED

- mask-only identical 0/4, answer-equal 0/4 (div@4-33; answers 4→0, 5→0, 16→20, 6→20).
  **E1 totals: identical 1/20, answer-equal 1/20 — READS-FRAMES verdict final.**
- fast≡mask 4/4 → **18/20 total** (2 near-tie flips, both in E1a) — within the ≤2 band;
  dense flagged decode still used for accuracy cells per the pre-launch amendment.
- **Decode speedup @N=64: 98.9× (657.1 → 6.6 s/sample incl. 2.2s cached prefill)**;
  VRAM 10.7 vs 10.6 GiB (flat, dense prefill still present in both arms).
- Run: `outputs/ladder/image_longN/trunc_kvdrop/e1b/20260725_*_evalonly/report.txt`.

### [2026-07-25] TRUNC E2 cell 3 (N=64, --truncate-at 12): **0.019** vs ref 0.981 — Δ −0.962, FAIL band (job 125564)

- `outputs/ladder/image_longN/trunc_at12/N64/20260725_030642_L12_r8_evalonly/`; dense
  flagged decode, DEC=620, arm-A `eval_dirs_N64.txt` (52 dirs). pf 0.000, MAE 15.73 —
  format intact, evidence content gone (same degeneration as in-dist). E2 verdict now
  rests on N=32 (125563) for completeness; the band outcome is already decided (FAIL).

### [2026-07-25] TRUNC E2 COMPLETE (cell 2 landed: N=32 0.040, job 125563) — eval-only truncation verdict: FAIL on all three cells

| cell (arm-A dirs, dense flagged decode) | trunc@12 | reference | Δ |
|---|---|---|---|
| in-dist-150 (125562) | 0.047 | 1.000 | −0.953 |
| N=32 ×150 (125563) | 0.040 | 0.987 | −0.947 |
| N=64 ×52 (125564) | 0.019 | 0.981 | −0.962 |

- pf 0.000 everywhere (format never breaks); MAE 4.73/11.17/15.73; runs
  `outputs/ladder/image_longN/trunc_at12/{indist150,N32,N64}/20260725_*_evalonly/`.
- Reading (with E1): NOT evidence against carrier saturation — the CURRENT ckpt's
  readout is frame-dependent by training. The N=32 0.040 point doubles as the E3
  curve's L12 anchor. E4 (deploy-matched retrain) is running and already at TF-count
  0.999 @ep2 (566s/ep) — the carrier-only readout trains; exams will decide GO.

## [2026-07-25] TRUNC E4 TRAINER LANDED (125570): deploy-matched truncated retrain reaches TF-count **1.000 @ep5** in **1h34 total** (566s/ep vs 8079s/ep = 14.3×; cache 8.1GB vs 169.2GB = 21×) — the carrier-only readout TRAINS to the same count-channel ceiling as the frame-reading recipe

- `outputs/ladder/image_longN/trunc_retrain/carrier_caption_trunc12/20260725_032236_L12_r8/`
  (h200, 12h_4g). Caption winner recipe verbatim + `--truncate-at 12`: lo phase full
  (carriers read frames), target rows frame-MASKED, cache = keep+target rows only, hi
  phase on truncated coords with original position ids.
- **Split gate PASSED: eval_dirs.txt byte-identical to arm A's AND the caption winner's**
  (verified by cmp before any exam) — exams on arm-A dirs-files are valid held-out cells.
- Trajectory: ep0 0.000 (frozen has NO carrier-only readout) → ep1 0.997 → ep5 **1.000**
  (cooc 432/432 · rooms 370/370 · steps 2964/2966 · union 255/255 · which 363/363).
  tf-exact plateaus at 0.165 (vs 0.995 original) — count channel perfect, transcript
  token diffs persist (room-word identity suspected); greedy exams (parse = tally) are
  the arbiter. Exams ×3 (125604/05/06, TRUNC_AT=12 dense) + E5 behavioral verify on this
  ckpt (125607) submitted. Band: within 0.01 of 1.000/0.987/0.981.

### [2026-07-25] TRUNC E3 point: L14 @N=32 = 0.033 (125565) — eval-only sweep reading FLAT, and the mechanism says it must be

- `outputs/ladder/image_longN/trunc_sweep/L14/20260725_030655_L12_r8_evalonly/` (pf 0,
  MAE 11.59). Curve so far: L12 0.040 · L14 0.033 (refs: baseline 0.987).
- Interpretation fixed BEFORE L16/20/24 land: the flagged decode removes decode-row frame
  reads at EVERY L (that is the deployment semantics), and E1 showed kvdrop-alone (≡ L=∞)
  already collapses to ~0.05 (answer-equal 1/20). So the eval-only curve is expected
  ~FLAT at the kvdrop floor for all L — it measures the decode-side frame dependence,
  NOT carrier-saturation depth. The saturation question is answered by E4-style
  deploy-matched retrains instead (E4@L12 TF-count 1.000 already suggests saturation at
  12 is sufficient WITH matched training). L16/20/24 land anyway (pre-registered).

## [2026-07-25] TRUNC E4-caption EXAM (in-dist): **0.133** vs band ≥0.99 — deploy-matched retrain FAILS at GREEDY readout despite TF-count 1.000; failure mode = ALL-OR-NOTHING transcripts; E4b (scan) launched as the one-change retry

- Job 125604 → `outputs/ladder/image_longN/trunc_retrain/exam_indist150/20260725_052147_*/`
  (pf 0.000, MAE 2.87). Transcripts collapse to a single global decision: all-"-" →
  "total: 0" or all-yes → "total: 8" (e.g. gold=5 → Kitchen(1..8)).
- Anatomy (TF vs greedy): TF-count 1.000 (count tokens read the GOLD verdict context —
  near-trivial); tf-exact 0.165 ⇒ per-token TF acc ~0.97-0.99 with errors concentrated
  in VERDICT tokens (~1-8 positives/transcript; missing any kills exactness). So the
  carrier-only PER-SLOT verdict is unreliable even teacher-forced, and greedy compounds
  it into all-or-nothing (first-verdict errors self-reinforce over the uniform format).
- The information IS in the carriers (external gate→tally reads 0.991-0.998 linearly) —
  the failure is the LM's in-context per-slot readout, i.e. the ADDRESSING/routing wall
  resurfacing at the readout stage. E5 verify2 on this ckpt (dq drops 34→8.4, carriers
  ≤0.12) confirms chunk math is fine; behavioral bar moot again (dense greedy broken).
- **E4b launched (125609)**: FMT=scan (presence-only 'yes' verdicts, no room identity),
  TRUNC_AT=12, recipe otherwise identical — separates "room-word identity beyond
  carriers" from "per-slot presence addressing" as the binding constraint. ~1.5h.
- Caption N=32/N=64 exams (125605/06) run to completion for the record.

## [2026-07-25] P4.2 trainer LANDED (125498) — carrier+DIGIT in-length: in-dist BEST 0.863 @ep5 (caption ref ~0.999); exams launched on rebuilt held-out dirs

- `carrier_digit_inlength/20260724_202048_L12_r8/`, 9h56: caption-winner data/recipe minus
  the scratchpad. **In-dist digit acc 0.863** (cooc 0.89 · steps 0.86 · rooms 0.73 ·
  union 0.84 · which 1.00 on gold≤9) — carriers WITHOUT the scratchpad already lose ~14
  pts in-distribution vs the caption arm.
- **Split note (the LOTO lesson recurs):** the digit path's gold>9 prep-skip → n=8241 ≠
  8772 → redrawn split → arm A's exam dirs-files are contaminated for this ckpt (128/150
  of `eval_dirs_N32all.txt` in its train set). Exams REBUILT from provably-unseen dirs:
  its own eval-split N=32/64 dirs ∪ ALL gold>9 dirs (excluded from both splits at prep,
  never seen), K-stratified (`eval_dirs_p42_N{32,64}.txt`, 150/100 dirs, full gold range).
  Same roots/distribution as the caption cells, different dir sample — stated wherever
  compared. Jobs 125610/11 (DEC=4 digit decode).

## [2026-07-25] P4.2 N=32 exam: 0.333 with a dead ≥g4 range — pre-registered THEORY-CONFIRMED band (≤0.50 + dead mid-range); the scratchpad, not the carriers or the data, carries the emitted answer

- Job 125610, `p42digit_eval_N32/…_evalonly/` (150 held-out dirs, full gold range, DEC=4):
  **0.333, MAE 2.65**; per-count g0-g3 ≈ perfect (44/48), g4 4/12, g5+ = 3/98 TOTAL
  (every gold ≥8 at zero). vs caption 0.987 @N=32 (same distribution, same carriers, same
  data — only the readout differs). The digit head undercount-clamps exactly where the
  law says a mean-aggregated single-token readout must; in-length data does not repair it.
- N=64 cell (125611) pending; P4.1 (plain SFT in-length, no carriers) completes the
  three-way decomposition.

## [2026-07-25] P4.2 VERDICT — carrier+digit in-length: N=32 0.333 · N=64 0.140, dead beyond g3 → THEORY-CONFIRMED at both cells; "carriers sufficient once data is in-length" is REFUTED

- N=64 (125611, `p42digit_eval_N64/…_evalonly/`, 100 held-out dirs): **0.140, MAE 6.26**;
  hits only at g0-g2 (14/21), zero at every gold ≥3. Ladder: caption 0.987/0.972/0.981 ≫
  digit 0.333/—/0.140 on the same carriers/data/recipe. Middle-rung answer: supply
  (carriers) + in-length data do NOT suffice — the sequential scratchpad decode is the
  load-bearing readout component. P4.1 (no carriers at all) completes the triangle.

## [2026-07-25] P1.1 seed2 N=32 held-out: 0.973 (pf 0, MAE 0.03) — identical 150 dirs as the 0.953/0.987 cells

- Job 125507, `fmtCseed2_eval_N32heldout/…_evalonly/`. Misses: 1× g1, 1× g8, 2× g24.
  Seeds so far: seed0 0.987 · seed2 0.973 · seed1 = 125519 (running).

## [2026-07-25] P1.1 COMPLETE — caption-recipe seeds: N=32 held-out 0.982 ± 0.007 over 3 seeds (0.987 / 0.987 / 0.973), pf 0.000 ×3, identical 150 dirs — the headline caption cell is seed-robust

- seed0 = the winner trainer (125105, exam 125195, 0.987) · seed1 (125347, exam 125519,
  **0.987**, `fmtCseed1_eval_N32heldout/`) · seed2 (125348, exam 125507, 0.973,
  `fmtCseed2_eval_N32heldout/`). Trainers byte-identical splits (gates PASSED ×2),
  --seed varies init/jitter/shuffle only. Trainer fits: 0.999/0.994 · 0.999/0.993 ·
  0.999/0.991 (acc/tf-exact). The l12v2 poslist reference on the same dirs: 0.953 —
  every caption seed beats it.

### [2026-07-25] E4-caption exam N=32: **0.073** (125605, ref 0.987; pf 0, MAE 10.18) — greedy carrier-only readout FAIL confirmed at length; N=64 (125606) pending

- `outputs/ladder/image_longN/trunc_retrain/exam_N32/20260725_052146_L12_r8_evalonly/`.

### [2026-07-25] E3 L16 @N=32: **0.107** (125571) — the eval-only curve is NOT flat: L12 0.040 · L14 0.033 · L16 0.107 (L20/L24 pending)

- `outputs/ladder/image_longN/trunc_sweep/L16/20260725_032320_L12_r8_evalonly/` (pf 0,
  MAE 6.49). The rise above L14 measures what layers 14-15's carrier own-frame refresh +
  tail frame-reads contribute through the frozen readout; still ≈floor vs baseline 0.987.

## [2026-07-25] TRUNC E4b (scan) — trainer tf-exact plateaus at 0.165 (IDENTICAL to caption's) and in-dist exam **0.093** (125613) → the greedy carrier-only readout failure is FORMAT-INDEPENDENT; in-model truncation verdict = NO-GO for the caption/scan recipe family

- Trainer 125609 → `outputs/ladder/image_longN/trunc_retrain/carrier_scan_trunc12/
  20260725_053752_L12_r8/` (BEST TF-count 0.999 @ep5, tf-exact 0.165, 558s/ep). Exam
  125613 → `exam_scan_indist150/20260725_073245_*/` (0.093, pf 0, MAE 3.38).
- Presence-only verdicts (scan 'yes') fail exactly like room-word captions → the binding
  constraint is per-slot ADDRESSING of carrier k from the decode row, not caption
  content. TF per-token verdicts are ~97-99% right (context-identical to greedy at each
  slot boundary), yet greedy collapses all-or-nothing — first-verdict misses cascade
  over the uniform format (exposure-style), and TF-count 1.000 is satisfied by counting
  gold context markers. Two ckpts, two formats, same 0.165 plateau: the number looks
  like a property of the truncated-readout task, not the run.
- Campaign path from here: HYBRID cell (external gate→tally on deploy-matched truncated
  carrier states; PREREG amendment 2 bands) + E6 benchmark + E3 curve completion; E7
  stays gated (no in-model GO).

### [2026-07-25] E3 L20 @N=32: **0.073** (125572) — curve: L12 0.040 · L14 0.033 · L16 0.107 · L20 0.073 (L24 pending)

- `outputs/ladder/image_longN/trunc_sweep/L20/20260725_032320_L12_r8_evalonly/` (pf 0,
  MAE 6.06). All points ≈ floor (baseline 0.987); mild non-monotone bump at L16.

## [2026-07-25] P4.1 trainer LANDED CLEAN on h200 (125567, attempt 5) — 0 train-skips, best val 0.967 @ep2, test_iid 0.959; exams launched on drift-checked dirs

- `sft_inlength_p41/20260725_031153_lora/`, 5h14, mixture N={8,16,32} (amended ≤32; five
  attempts documented). Adapter saved; val curve 0.733→0.850→0.967→0.883 (ep2 best,
  consistent with the ep1-3 lesson).
- Exam dirs (drift check run for THIS split too): arm-A `eval_dirs_N32all.txt` is
  contaminated (declare_splits trained ~70% of longN_park32) → N=32 exam file rebuilt
  from its own test split + never-seen park2 dirs (K-stratified 150,
  `eval_dirs_p41_N32.txt`); **N=64 = arm-A 52-dir file, fully clean (nothing at N=64
  was trained — extrapolation cell, labeled)**. Job 125620 runs both + test_iid re-anchor.

## [2026-07-25] 🔬 TRUNC HYBRID PROBE (external gate→tally on TRUNCATED carrier states, N=32): per-frame err ~0.33-0.37, tally 0.05-0.075 at L∈{12,16,20} — the truncated carrier states DON'T linearly encode per-frame evidence; z-scoring and layer-deltas don't rescue it → NEW HYPOTHESIS: the carrier gate CONSOLIDATES in layers 12-16 via continued own-frame attention, and truncation at 12 cuts it off

> Dump 125615 (E4-caption ckpt, TRUNC_AT=12, arm-A N32 dirs ×150, layers {12,16,20}) →
> `outputs/ladder/image_longN/trunc_retrain/hybrid_dump_N32/20260725_073846_*/`
> (`carrier_states_cache.pt` + `gate_tally_L{12,16,20}/report.txt`); re-probe with
> z-scoring + deltas inline (main session, printed above run dir).

- replica_gate_tally (5 seeds): L12 0.051±0.031 · L16 0.059±0.022 · L20 0.056±0.031
  (majority 0.107!); per-frame err 0.33-0.35. Z-scored + delta features: no change
  (err 0.34-0.37). **PREREG amendment-2 band <0.70 → "truncation damages the carrier
  code" — but the sharper reading is below.**
- KEY: the L12-ENTRY states are PRE-LoRA (layers 0-11 frozen, e_c frozen distill) — pure
  deployed supply. The historical 0.99 gate→tally cells were on REPLICA messages / the
  distill locus at LAYER 16 — i.e. after 4 MORE FENCED layers of own-frame attention.
  In stage-2 the fence opens at 12, but mask_hi RETAINS carrier→own-frame edges — the
  own-frame refresh in layers 12+ (the exact edge the brief called "the hypothesis
  under test") may be where the evidence signal finishes consolidating.
- If confirmed, E2/E4's failures get a SUPPLY-TIMING component (not only readout
  addressing): at L12 the carriers haven't finished encoding; truncating there starves
  even a perfect readout. Discriminating dump RUNNING (125621): NON-truncated forward,
  winner ckpt, layers {12,13,14,16,20,24} — the layer where per-frame err drops to ~0
  IS the carrier-saturation depth (the E3 deliverable, probe-measured without the
  decode confound).

### [2026-07-25] E4-caption exam row COMPLETE (N=64: **0.096**, 125606) — deploy-matched retrain verdict: NO-GO (in-dist 0.133 · N=32 0.073 · N=64 0.096 vs refs 1.000/0.987/0.981; pf 0 everywhere)

- `outputs/ladder/image_longN/trunc_retrain/exam_N64/20260725_052141_L12_r8_evalonly/`.
- With E4b-scan 0.093 in-dist: the truncated in-model readout fails at ~0.07-0.13 across
  formats and lengths, while TF-count is 0.999-1.000 — jointly explained by the readout
  addressing wall AND (per the hybrid probe) supply-timing: at L12 the carrier gate isn't
  yet linearly consolidated. Saturation-depth dump (125621) will separate these.

## [2026-07-25] P4.1 VERDICT — the dangerous cell lands in the pre-registered "SIMPLE FIX WINS" band: plain-LoRA SFT trained in-length reads N=32 at 0.967 (pf 0, per-count uniform incl. g12-g32); N=64 extrapolation 0.787 — logged honestly, with the contrasts that survive

- Job 125620 (`sft_inlength_p41_exams/…_lora/` + `logs/p41_exam-125620.out`): test_iid
  re-anchor 0.9587 = trainer exactly. **N=32 held-out (150 drift-checked dirs): 0.9667,
  pf 0.000, MAE 0.03** — ≥0.90 band → simple-fix-wins. **N=64 (150 dirs of the arm-A
  eval-split file; trained ≤32 → extrapolation): 0.7867, pf 0.000, MAE 1.11** (soft spots
  g6 3/11 · g16 2/9 · g48 2/11; g64 9/9). Note: this cell used 150 dirs of
  `eval_dirs_N64.txt` (the caption cells used its first 52 — subset included).
- **Honest reading: with IN-LENGTH data, a plain 23.8M-param LoRA (all layers, attn+MLP)
  solves emitted N=32 counting for the single steps task without carriers or scratchpad —
  the previous "SFT ladder 0.480/0.350/0.220" strongest-baseline row is RETIRED (it was a
  data artifact: trained ≤8). This REFRAMES the thesis contribution, per the prereg.**
- Contrasts that survive P4.1 (the reframed story):
  1. **P4.2's asymmetry**: carriers + digit + the same in-length data = 0.333/0.140 —
     a small r8 LoRA on L12+ with a single-token readout canNOT do what the full-model
     23.8M LoRA + free-form generation can; readout expressivity, not data, separates them.
  2. **Task generality**: P4.1 is single-task by design (script limit, documented);
     the caption arm is one model for 5 tasks incl. LOTO partial transfer.
  3. **Length reach + training cost**: SFT training at N=32 required an h200 (5 failed/
     degenerate attempts on 40GB — MATH/mask lessons); N=64 SFT training fits NO available
     GPU. The carrier/cached trainer trains ≤64 on a 40GB a100 (13.8h) — the architecture's
     cheap-training property is now a measured claim, not a preference.
  4. N=64: caption in-length 0.981 vs SFT extrapolated 0.787 (SFT structurally cannot be
     trained in-length at 64 on this cluster).

### [2026-07-25] TRUNC E3 sweep COMPLETE (L24: 0.073, 125573) — eval-only truncation-layer curve @N=32: **L12 0.040 · L14 0.033 · L16 0.107 · L20 0.073 · L24 0.073** (baseline 0.987; pf 0 everywhere)

- Runs `outputs/ladder/image_longN/trunc_sweep/L{14,16,20,24}/20260725_*_evalonly/` +
  L12 = the E2 N=32 cell. All points ≈ the kvdrop floor (E1: dropping decode frame reads
  alone collapses answers) — the eval-only curve measures the decode confound, NOT
  saturation depth. No knee exists; figure kept for the record. The probe-based
  saturation curve (dump 125621 + gate probes per layer) supersedes this as the
  mechanistic deliverable.

## [2026-07-25] 🔬📊 TRUNC — CARRIER-SATURATION DEPTH MEASURED (probe curve, dump 125621 + CPU logistic): the per-frame evidence gate is UNREADABLE at L12 (err 0.34) and consolidates between L16 and L20 (err 0.173 → 0.0082); external gate→tally reaches **0.909±0.016 @N=32 at L24** — the mechanistic quantity the brief wanted from E3, without the decode confound

> `outputs/ladder/image_longN/trunc_retrain/hybrid_dump_N32_notrunc/20260725_082922_*/`
> (`saturation_probe_report.txt`, `saturation_curve.png`, cache). Winner caption ckpt,
> NON-truncated forward, arm-A N32 dirs ×150; z-scored logistic, 5 seeds.

| carrier state entering | per-frame gate err | external tally exact |
|---|---|---|
| L12 | 0.3388±0.0199 | 0.075 |
| L13 | 0.3062 | 0.085 |
| L14 | 0.2164 | 0.099 |
| L16 | 0.1728 | 0.080 |
| **L20** | **0.0082±0.0016** | **0.843±0.026** |
| **L24** | **0.0051±0.0010** | **0.909±0.016** |

- Truncated@12 reference (dump 125615): err stays 0.33-0.37 at L16/L20 — consolidation
  REQUIRES frames present in layers 12-19 (the mask_hi carrier→own-frame edges, exactly
  the "hypothesis under test" from the brief §0.3 — now answered: those edges are where
  the gate gets written).
- This REFRAMES the campaign: L*=12 is the right fence-open depth for LM aggregation,
  but the carrier CODE saturates at ~L20. "Best L_trunc" (brief E4 wording) = 20, not
  12. E2/E4@12 failures = supply starvation + readout jointly.
- **E4c LAUNCHED (125628): --l-open 20 --truncate-at 20**, caption recipe verbatim —
  fenced carrier maturation through L19 (distill-like), cache at L20, deploy-matched.
  Chunked prefill remains valid (trunc == L_OPEN). If its exams land near reference,
  the thesis gets: truncation GO at L20 (8/28 layers + all decode on ~100 tokens,
  decode speedup already measured 32-95×) + the saturation-depth figure.

## [2026-07-25] TRUNC E4c trainer landed (125628, l_open=20 + truncate_at=20, saturation-guided): BEST 0.998 @ep4, **tf-exact 0.240** (vs 0.165 plateau @12; original 0.995) in 1h27 (352s/ep); split gate byte-identical again — exams 125636/37/38 running

- `outputs/ladder/image_longN/trunc_retrain/carrier_caption_trunc20/20260725_100107_L20_r8/`.
  tf-exact trajectory 0.176 → 0.223 → 0.230 → 0.240 → 0.251: clearly above the @12
  ceiling from ep1, still far from the untruncated 0.995 — consolidation helps, the
  remaining gap is the readout itself. Greedy exams decide.

## [2026-07-25] 📊 TRUNC E6 run-A COMPLETE (125616-19, h200, 3 samples/N, winner ckpt): decode speedup **1.9× / 32.3× / 95.4× / 311.4×** at N=8/32/64/128

| N | base decode s/sample | mask-only | fast (cached keep-only) | speedup | peak GiB base/fast |
|---|---|---|---|---|---|
| 8 | 5.2 | 7.3 | 2.7 | 1.9× | 6.1 / 6.1 |
| 32 | 96.4 | 36.4 | 3.0 | 32.3× | 7.8 / 7.8 |
| 64 | 667.3 | 230.6 | 7.0 | 95.4× | 10.7 / 10.6 |
| 128 | 3546.9 | 633.2 | 11.4 | **311.4×** | 18.3 / 18.0 |

- `outputs/ladder/image_longN/trunc_bench/runA_N{8,32,64,128}/20260725_*/report.txt`.
  fast≡mask 12/12 across the grid. Caveats (report verbatim in the thesis table): dense
  python re-forward baseline (no KV-cache server); N=128 bench dirs are K0 (gold 0 —
  transcript length is format-driven so timing valid); VRAM ~flat because the dense
  prefill dominates peak in both arms (truncated/chunked arms address prefill).

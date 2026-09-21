# RESULTS.md — Research progress log (live: the fencing / gate / read era)

> **Purpose:** the run-by-run research record of the current method. Append-only; every
> number traces to a real run dir. New entries go at the END with `## [YYYY-MM-DD]` headers.
> **Status flags:** ✅ done & trusted · ⚠️ partial/suspect · ❌ failed/no-gain · 📊 probe · ▶ running.
>
> - Pre-fencing record (Feb 2026 → [2026-07-13]: AF1 lineage, frame-axis/DeepSets, gLSTM,
>   d′ framework construction): frozen at [docs/archive/RESULTS_pre_fencing.md](docs/archive/RESULTS_pre_fencing.md)
>   — glossary and the GNN references live there too. Dated cross-references earlier than
>   [2026-07-14] point into that file.
> - Cited output paths that no longer resolve → [legacy/v1/ARCHIVE_MAP.md](legacy/v1/ARCHIVE_MAP.md).
>   Cited code paths (`experiments/…`, `evaluations/…`, `models/…`, `runners/…`) → `legacy/<same path>`;
>   (`gnnformer/…`, `scripts/…`, `slurm/…`, `datasets/…`) → `legacy/v1/<same path>`.
> - The July method statement cited by the 2026-07 entries: [docs/archive/METHOD_2026-07.md](docs/archive/METHOD_2026-07.md).

## Status (2026-09-21) — data regimes, read this before citing any number below

**Two data regimes appear in this log, and a table row must say which one it is.**

| regime | what | entries | reproduce with |
|---|---|---|---|
| **park data (legacy generator)** | our own MMRED re-implementation: `data/mmred_images_park` (seq 1–8), `data/mmred_longN_park*` (N=16…128), `data/mmred_redux`, the balanced/cooc/rooms pools; count prompt with or without N in the text | every entry from [2026-07-14] to [2026-09-21] unless marked HF | `legacy/v1/` + the archived roots (`/rg/shocher_prj/tal.gorbunov/archive/data/`) |
| **official MMReD** | HF `ef1e43ce/mmred` rows + the authors' renderer (native 512 px), 24 question types | [2026-08-01→13] MMReD-HF campaign; LORAMECH P2 / N4; SPARSE S5 / S9-HF | `data/mmred_hf` → `/rg/shocher_prj/lab_data/mmred` |

**From 2026-09-21 the repo works ONLY on the official benchmark** (paper prompt verbatim,
train seq_len ≤ 16, test all lengths; long-N counting cells reported split by answer ≤ 16 / > 16).
The park-data results stay valid as mechanism evidence and are labelled "legacy data" when cited.

**Current canonical numbers (park data unless marked):** frozen baseline 0.219 @N=8 · fenced
supply d′ 13.54 vs joint 5.95 · gated fenced-SFT S8 (virtual-N) 1.000/1.000/0.853 @8/16/32
· S9b (N-free) exact on every trained answer value at every N ∈ {8..128} · deployable gate
(logistic on the L20 replica slot) ≥ 0.999/frame flat in N · share law s = 0.30, C = 8 at L20 ·
decay-in-k γ = 1.19 · **official MMReD (steps_in_room test):** P2 fenced-SFT 0.900/0.660/0.300
@8/16/32, N4 park→HF 1.000 @8, S5 model gate 0.900/0.700/0.540, S9-HF 0.86/0.78/0.58; frozen
24-type grid 0.533/0.422/0.361/0.304/0.247 @8…128.

## Executive summary (updated 2026-07-30 — HISTORICAL: describes the July carrier/scratchpad method, superseded 2026-08-12 by the gated fenced read; see the Status block above and the entries from [2026-08-26] on)

**Problem.** A frozen VLM (Qwen2.5-VL-7B) collapses on multi-frame counting (MMRED: "how many
frames was C in room R?", answer ∈ 0..N): joint-attention supply dilutes per-frame evidence
(d′/√N law; measured squashed-readout curve P1.2) and the single-token readout can't use what
does arrive. Frozen baseline: **0.219** exact @N=8 full prior.

**Method (three stacked repairs, all on the frozen 4-bit backbone):**
1. **Fencing (supply):** per-frame question replicas / carrier tokens behind a block-diagonal
   attention fence + per-block M-RoPE position reset, one forward. Supply d′ **13.54 ± 0.27**
   @n900 vs joint anchor 5.95 ([2026-07-18] E1); posreset advantage grows with N ([2026-07-27]).
2. **Learned carriers (in-model aggregation):** one distilled carrier embedding e_c
   (96% of teacher d′) + a ~2M-param LoRA on layers ≥ L\*=12 (E-H inverted-U) integrating
   carrier messages.
3. **Scratchpad readout (expressivity):** caption-format scan decode; the answer is a read-off
   of the inline tally (format sweep winner C, [2026-07-24]).

**Headline numbers (all in this file; run dirs cited per entry):**

| cell | number | entry |
|---|---|---|
| frozen baseline, N=8 full prior | 0.219 | [2026-07-18] E1 |
| supply d′ (blockfence+posreset+qfirst, n900) | 13.54 ± 0.27 (joint 5.95) | [2026-07-18] E1 |
| gate→tally scaffold ceiling, N=8 | 0.998 ± 0.001 | [2026-07-18] E1 |
| in-model in-dist (5-task, one carrier + one LoRA) | 0.999 / greedy 0.953–0.966 | [2026-07-19] |
| **in-model held-out N=32** (caption, in-length-trained) | **0.987** (seeds 0.982 ± 0.007, pf 0) | [2026-07-24] FORMAT SWEEP / P1.1 |
| in-model held-out N=48 / N=64 | 0.972 / **0.981** (pf 0) | [2026-07-24] FORMAT SWEEP |
| N=128 (2× beyond max trained) | 0.235 full-34 (format intact) | [2026-07-29] migration note |
| plain-LoRA SFT control, in-length | 0.967 @N=32 but 0.787 @N=64-extrapolated; h200-only training; single-task | [2026-07-25] P4 |
| no-harm (MME/POPE), both adapter families | \|Δ\| ≤ 1.4 pts → GO | [2026-07-19] E-D / [2026-07-24] P4.3 |
| task transfer (LOTO, leave-cooc-out) | partial @N=8 (0.403), NO-GO @N=32 | [2026-07-24] P2a |
| cross-domain (natural images) | supply GO (d′ 27.3) · scaffold GO (0.980) · in-model NO-GO | [2026-07-24] P3a |
| cross-family (InternVL2.5-8B) | supply ports (3.5×) · scaffold 0.938 · Q-first is Qwen-specific | [2026-07-19] / [2026-07-24] P3b |
| efficiency | exact cached-fast decode 16–311×; KV-truncation refuted (decode reads frames) | [2026-07-25] TRUNC |

**Standing honesty flags.** (1) All headline in-model numbers are trained-on-clean synthetic
MMRED; the natural-images in-model rung is a NO-GO (mechanism ports, trained readout is
domain-bound). (2) The N=32/48/64 cells are in-length-trained — zero-shot length extrapolation
still fails (E-A ladder); the honest claim is a ladder in trained-length coverage.
(3) The old "SFT ladder 0.480/0.350/0.220" baseline row is RETIRED (P4.1: it measured missing
in-length data); the surviving SFT contrasts are readout expressivity (P4.2), generality, and
training cost. (4) d′ estimates scale with n — compare like-for-like n only.

---

# Experiment log (append-only, newest last)

## [2026-07-14] ✅📊 REPLICA CARRIERS — per-frame question replicas in ONE forward: the UNMASKED (plain interleaved prompt) arm reads d′ 3.56 @L16 (+81% over joint 1.97, above the pre-registered GO bar ≥3) and BEATS the masked-attention arm (2.52); the masked arm's per-copy ladder exposes a design subtlety (the mask cleans queries everywhere but values only for frame 0); the predicted contamination decay in the unmasked arm did NOT appear at N=8

> **Motivation:** the query half of the joint-context tax is architecturally irreducible
> ([2026-07-13] trained-query NO-GO: shared q* 0.36–0.48). Replica carriers test per-source
> addressing via TOKEN TOPOLOGY instead of weights: insert a copy of the question after every
> frame; each copy's room token is a per-frame local carrier. Two arms: MASKED (custom all-layer
> 4D mask — each replica attends only {prefix, its own frame, itself} and is INVISIBLE to all
> other tokens, so the original computation is undisturbed) and UNMASKED (plain interleaved
> prompt, everything visible — the prompt-engineering control, proposed by Tal).
> **Script:** `experiments/glstm/replica_carrier_probe.py` (new; reuses the fence 4D-mask
> injection, the probe qkv/rotary capture, and the 2×2 o_proj dequantization). **Runs:** job
> **121431** (a100, 88 min: masked smoke n=8 → masked full n=300 → unmasked full n=300; job
> 121401 = earlier attempt, failed on an image_token_groups signature); steps task,
> mmred_images_park seq_len_8, 392px, forward-only. Dirs:
> `outputs/ladder/image_longN/replica_carrier/20260714_214534/` (masked) and
> `…/replica_carrier_nomask/20260714_221634/` (unmasked); held-out shrinkage-LDA d′
> (dprime_pair, 3 sample-disjoint seeds), n=300, skip=0.

| read @L16 (n=300) | mean d′ | per-copy d′ (index 0→7) |
|---|---|---|
| **UNMASKED interleaved** | **3.56±0.14** | 3.73 4.00 2.88 2.65 3.52 2.68 2.59 2.97 (≈flat) |
| MASKED replicas | 2.52±0.11 | **3.70** 2.33 1.75 2.01 1.99 1.82 1.99 2.18 (ladder) |
| external joint anchor (B1 N=8, same data/layer/px) | 1.97 | — |
| in-run "off−9 anchor" | 0.15–0.73 | **INVALID** — the interleaved chat template shifts the final room token off the off−9 position; do not use |

(L14: unmasked 2.72, masked 1.52 — same ordering, lower level, consistent with L14 being the
weaker carrier layer.)

**Readings.**
1. **The unmasked interleave is the finding: d′ 3.56 in one forward, zero training, zero
   architecture change** — +81% over the joint carrier, above the pre-registered GO bar (≥3)
   that the masked arm missed. Gate-law pricing at this supply (p=0.038): tally exact ≈ 0.75 /
   0.37 / 0.10 at N=8/32/128 vs 0.33/0.09/0.02 at joint supply — IF the level holds at larger N
   (untested; the decisive follow-up).
2. **The predicted per-copy contamination decay (≈6 at copy 0 → ≈2.3 at copy 7, from the
   chunk-size curve) did NOT appear** — the unmasked ladder is ≈flat at ~2.6–4.0. Best current
   account: question-conditioned frame encoding (frames attend earlier question copies, the
   Q-first effect) offsets the growing cross-frame contamination, at least to N=8.
3. **The masked arm underperformed its 2×2 prediction (3.3–4.0) at 2.52, and its per-copy ladder
   explains why:** the mask restricted REPLICA rows only, so frame encodings still attend all
   earlier frames — frame 0 is effectively isolation-encoded (copy 0 reads 3.70 ✓ the predicted
   band) while frame 7 is fully joint (2.18). The mask delivered clean queries everywhere but
   clean values only at frame 0. The arms dissociate the two ingredients: clean queries alone
   ≈ +28%; visibility (question conditioning) ≈ +81%.
4. **Both in-run anchors are mislocated** (off−9 convention breaks under the interleaved
   template); the external B1 anchor is the valid baseline. Fix for reruns: locate the final
   question's room token by word match like the replicas.

**Caveats.** Single seed, N=8 only, steps task only, n=300; per-copy d′ from n=300×1 frames
(±~0.2); masked-arm smoke n=8 prints nan (too small for held-out folds — expected); the two
arms share frames/questions so their difference is paired but no paired test was run;
behavioral (emitted-answer) effect of the interleaved prompt NOT measured — messages only.

**Next (registered):** (a) unmasked interleave at N=32/128 — does the flat ladder hold (the
gate-law prize at N=32 is 0.37 vs joint 0.09)? (b) replicas + frame-fencing combined (clean
queries AND clean values per frame → 2×2 predicts ~6, multipass-in-one-forward; the B1
fence-alone null does not preclude it — that null had a joint query); (c) behavioral EM with
the interleaved prompt; (d) gate→tally on the unmasked replica messages (supply 3.56 is above
the k=2 chunk supply 3.37 that retrieve-v2 uses — a ONE-forward shortlist candidate).

## [2026-07-17] ✅📊 ONE-FORWARD SUPPLY CLOSED — replicas + FULL block-diagonal fence + per-block M-RoPE reset reach the multipass band (d′ 6.34 = solo anchor 6.01) in ONE forward; Q-FIRST amplifies to 9.24; supply FLAT to N=128; gate→tally beats every N-forward system at every N

> **Motivation:** complete the [2026-07-15] fence PARTIAL (4.07). Two rungs: A2 = per-block
> position reset (PCW-style reuse — legal because fenced blocks are mutually invisible); A3 =
> seal the marker leak found analyzing A2 (`vision_start/end` tokens are neither visual nor
> replica tokens, so per-token-class fencing leaves them visible and their residuals carry
> earlier frames' content — worth +1.7 d′). A4 = the question ALSO in the shared prefix.
> **Script:** `replica_carrier_probe.py` flags `--reset-positions --fence-blocks
> --question-first --task cooc --natural`. **Runs (2026-07-17/18, jobs 122739–123137):**
> `outputs/ladder/image_longN/replica_{posreset,blockfence*,cooc_qfirst*,natural_*}/`,
> anchors `replica_posreset_N1anchor*/`. Draft detail: `plans/oneforward_DRAFT_RESULTS.md`.

| rung (N=8, L16, n=300, held-out LDA) | pooled d′ | per-copy |
|---|---|---|
| joint (external anchor) | 1.97 | — |
| unmasked replicas [2026-07-14] | 3.56 | ≈flat |
| + frame fence [2026-07-15] | 4.07 | ladder 3.8→2.0 |
| + position reset (A2, job 122744) | 4.66 | flatter, frame0 unchanged |
| **+ marker seal = block-diagonal (A3, job 122809)** | **6.34±0.11** | **flat 3.6–4.6** |
| solo anchor, same instrument (N=1, n=1200, job 122782) | 6.01±0.26 | — |
| **+ question-first (A4, job 122888)** | **9.24±0.33** | flat 5.7–7.6 |

**N-sweep (blockfence+posreset, standard layout):** d′ 7.62/7.81/7.55/7.24 at N=16/32/64/128 —
flat at the multipass band (mp 7.18–8.08); joint stays ~2.0. Q-first long-N: 12.67 @N=32,
11.57 @N=128. Cooc (`--task cooc`, relational predicate): 6.36/8.45/7.58 @N=8/32/128. Natural
photos (`--natural`, n=50/cell): dist_far 6.22 (in-run joint 3.12), dist_near 5.69 (3.61).

**Behavior (held-out logistic gate → tally, CPU on the message caches, 5 seeds):**

| N | steps 1-fwd | steps Q-first | cooc | retrieve-then-verify (~N fwd) | mp tally (N fwd) |
|---|---|---|---|---|---|
| 8 | 0.960 | 0.991 | 0.973 | — | 0.910 |
| 16 | 0.976 | — | — | — | 0.793 |
| 32 | 0.960 | 1.000 | 0.932 | 0.862 | 0.680 |
| 64 | 0.952 | — | — | 0.853 | 0.580 |
| 128 | 0.876 | 0.984 | 0.680 | 0.791 | 0.420 |

**Readings.** (1) Multipass is compiled into one forward by token topology + mask + positions —
zero training; the residual decomposes as replicas +1.6 / fence +0.5 / posreset +0.6 / marker
seal +1.7. (2) Q-first beats even isolated forwards: multipass encodes frames question-blind;
Q-first conditions the ENCODING — the first arm in the record to do so. (3) One forward + a
logistic gate beats the ~N-forward pipelines everywhere measured. (4) Attention is block-sparse
(O(N) in frames); no OOM at seq≈28k.

**Caveats.** AUC estimator pegs at 5.26 in the high-supply regimes (same caveat as all mp-band
numbers; E4 saturation signature — quote measured accuracies, not Φ(d′/2), there). **Count-prior
audit:** the N=8 caches contain gold∈{0,1,2} only (K-sorted dirs; majority 0.333) — N=8
exact-count rows are NOT comparable to the historical frozen 0.207 (full 0–8 prior) without the
same-samples frozen baseline (job 123205, pending); long-N caches have wide priors (majority
0.10–0.30) and stand as-is. Per-frame errors and d′ are prior-free. Gate trained per-N/task on
half the cache (~150 labeled samples). Cross-task gate transfer is partial (steps→cooc 0.460).

## [2026-07-17→18] ✅📊 LEARNED CARRIER TOKEN — ONE trainable embedding (3,584 params, frozen backbone) replaces the 20-token question replica at 92–93% of teacher; task-general (steps→cooc zero-shot 88% of ceiling) and length-general (N=8-trained → d′ 9.7 @N=128); random init converges to the same endpoint; the trained-query-at-L16 floor (0.4) beaten ~20×

> **Motivation:** distill the replica into a clean, task-agnostic, model-agnostic primitive; the
> controlled contrast with the [2026-07-13] trained-query NO-GO (same-size vector, input-with-
> context vs L16-without). **Script:** `experiments/glstm/carrier_token_distill.py` (placeholder
> `<|box_start|>` id 151648; Q-first blockfence+posreset; truncated differentiable forward to
> L16; objectives: proxy=BCE on evidence label, distill=cosine to in-run replica-teacher
> messages, label-free; in-run teacher anchor reproduces the probe exactly, 9.24). **Runs:**
> jobs 122938/39 (arms), 123124–26 (ablations), 123128/29 (length-gen), 123145 (cross-task) →
> `outputs/ladder/image_longN/carrier_token*/`.

| measurement | value |
|---|---|
| ep-0 warm start (UNTRAINED) | d′ 5.23 eval |
| proxy arm | 6.46 eval |
| **distill arm (label-free)** | **8.35 eval / ~9.0 full-n** (teacher eval-split 8.95) |
| random init / k=2 / k=4 | 8.25 / 8.14 / 8.38 — init irrelevant, k=1 suffices |
| gate→tally @N=8 (carrier messages) | **0.997±0.003** (per-frame err 0.0003) |
| zero-shot N=32 / N=128 (trained @N=8 only) | d′ 11.40 / 9.71; refit gate 1.000 / 0.988; FULL N=8-trained stack zero-shot **0.917 / 0.860** |
| zero-shot cooc (task never seen) | d′ 5.58 = 88% of cooc teacher 6.36; + per-task gate 0.880 |

**Reading:** per-source addressing must be computed in-context and CAN be learned into one input
token; task-conditioning arrives via attention to the question (label-free objective, cross-task
transfer). Token overhead 1/frame. **Caveats:** same count-prior note as above at N=8; both
tasks share the MMRED visual world (natural-image carrier transfer = job 123208 pending); the
per-task gate (3.6k params, ~150 samples) is the remaining task-specific piece.

## [2026-07-18] 📊 STAGE-2 "CARRIER LAYER" (all-in-model) — fenced extraction ≤L16 + cross-carrier attention ≥L17 + LoRA(r8, ~2M params) + plain LM loss: emitted answer 0.353→0.853 in 12 ep (undertrained, still climbing); the no-LoRA ablation plateaus at ~0.50 — cross-carrier attention alone canNOT aggregate, trained integration is required (the text-MMRED prediction, confirmed in-model)

> **Script:** `experiments/glstm/carrier_layer_lora.py` (mask schedule lo/hi; carriers get
> sequential positions; tail attends carriers ≥L17; hand-rolled zero-init LoRA on q/k/v/o of
> layers 17–27; model answers via its own lm_head — no gate/tally/render). **Runs:** jobs
> 123149 (LoRA, best 0.853 @ep12) / 123150 (no-LoRA, best 0.507) →
> `outputs/ladder/image_longN/carrier_layer/20260718_0230*/`; ep-0 = 0.353 ≈ the gold∈{0,1,2}
> majority (0.333). Extended 30-ep run job 123206 + same-samples frozen baseline job 123205
> pending. Scaffold ceiling on this prior: 0.991–0.997.

## [2026-07-15] ❌📊 ONE-FORWARD REPAIR ARMS (oneforward Exp B/C, never logged) — the offline encoding un-mixer does NOT transfer to the replica forward (3.56 → **1.44**, destructive), and a CoGNN-style content-side broadcast gate cannot repair routing (1.80 ≈ the 2.09 floor); the "qcond GO" is INVALID by q_pad feature leak (q_pad itself carries d′ 8.63)

- **B1** un-mixer retrained + weights SAVED (121919, `unmixer_saved/20260715_194450/`, weights
  `unmixer_saved/weights/unmixer_L16.pt`): offline mp-q × un-mixed-kv 5.94 vs joint-kv 3.82 /
  ceiling 6.33 = **84% of the encoding gap** (prior 93% ≈ retrain variance).
- **B2 deployed composition** (121928, `replica_unmix/20260715_204158/`; g_k/g_v hooked on
  frame-token k/v_proj @L16 in the unmasked replica forward): **3.56 → 1.44 @L16 — NO TRANSFER,
  actively destructive.** Frame 0 alone improves (3.73→4.17, in-distribution); L14 control
  unchanged (2.70 vs 2.72) — the replica layout's question-conditioned k/v are off the
  un-mixer's training distribution. (Moot after A3 blockfence: prevention beats repair.)
- **C broadcast gate** (121918, `broadcast_gate/20260715_194451/`; in-run anchors PASS 2.09/3.82):
  content arm ([k_j,v_j]) eval **1.80** (trajectory max 2.13 ≈ the 2.09 floor) — pre-registered
  "routing NOT repairable from content" outcome, closing addressing from a third direction
  (trained shared query NO-GO · content gate NO-GO · only architectural frame identity works).
  **qcond arm 30.69 = INVALID (do NOT cite):** q_pad is captured after the question attends the
  frame — the leak probe (121927, `broadcast_gate/qpad_leak/20260715_201750/`) reads q_pad alone
  at d′ **8.63** (q_mp 10.32) ≥ the mp×mp ceiling, while pooled content k/v-means carry ~0.65 —
  the gate broadcasts label information, and computing q_pad at inference = multipass anyway.
  The script's auto-VERDICT "GO" line is superseded by this reading.

## [2026-07-18] ⚠️📊 STAGE-2 AT 450 TRAIN SAMPLES — convergence hypothesis REFUTED (30-ep ref flat at 0.840 from ep12), full-prior steps 0.678 with the clamp DEAD, mixtures 0.693 (2-task) / 0.509 (3-task, rooms 0.50): a DATA-STARVATION ceiling, not architecture — sets up the pooled P1; cross-DOMAIN carrier transfer is only ~51% of teacher

- **30-ep truncated-prior ref** (123206, `carrier_layer/20260718_122503_L17_r8/`): BEST 0.840
  @ep12, loss→1e-4 by ep14, eval FLAT to ep30 — "undertrained, still climbing" ([2026-07-18]
  entry) is REFUTED; the gap is data/generalization, not optimization.
- **Full-prior steps-only 40ep** (123235, `carrier_layer/20260718_131157_L17_r8/`, train 450/eval
  450): **0.678 @ep30**, per-count uniform incl. g8 (clamp dead); train loss→0 = memorizes 450.
- **Mixtures** (warm-start distilled e_c, 30 ep): steps+cooc **0.693 @ep25** (cooc 0.796 / steps
  0.560 — no task interference; 123237, `carrier_layer_mixture/20260718_133821_L17_r8/`);
  +rooms 3-task 0.509 @ep12, rooms 0.50 vs frozen 0.087 / pipeline 0.993 — cross-carrier
  set-union PARTIALLY learned (123240, `carrier_layer_mixture3/20260718_134209_L17_r8/`).
- **Cross-DOMAIN carrier** (123208, `carrier_token_crosstask_natural/20260718_122538_proxy_room_k1/`,
  n=50 — wide bars): steps-distilled e_c on natural dist_far zero-shot d′ **3.19 ± 0.55** = ~51%
  of the cell's replica teacher (6.22); fresh-gate tally 0.432 UNDERPERFORMS the frozen model
  (0.58) — the carrier is partly domain-bound (vs 88% cross-task within synthetic MMRED).


## [2026-07-18] ✅📊 E1 FULL-PRIOR RECALIBRATION (N=8, LIMIT=900, gold uniform 0..8) — frozen baseline **0.219** (retires the truncated-prior 0.513); Q-first blockfence probe d′ 13.54 @n900 with gate→tally **0.998**; carrier-token distill 11.45 = 96% of teacher, carrier-stack tally 0.999 — the scaffold ceiling for all stage-2 comparisons

- **Audit caveat driving this entry:** the old N=8 caches were e-sorted → n=300 meant gold∈{0,1,2}
  only. Fix: stratified `iter_sample_dirs_shuffled` (+`--shuffle-dirs`, gold-hist prints) wired
  into probe/distill/trainer/baseline scripts; smokes 123222/123226 (`outputs/_scratch/st2_smoke*/`).
- **Frozen baseline** (123225, `frozen_baseline/20260718_125303/`): **0.219, MAE 1.86** — the
  undercount clamp in full view (g4+ ≈ 0). Truncated-prior 0.513 (123205) RETIRED (it rewarded
  the low-count bias).
- **Probe** (123232, `replica_blockfence_qfirst_full900/20260718_130546/`): **d′ 13.54 ± 0.27**
  @n900 (joint anchor 5.95; per-copy flat 8.4–9.2); matched-n300 subsample 10.50–10.89 vs the
  truncated band 9.24 ± 0.33 — supply is prior-free; d′ estimator scales with n (caveat for all
  cross-n comparisons). **Gate→tally (123236): exact 0.998 ± 0.001, MAE 0.00** vs majority 0.111.
- **Distill** (123233, `carrier_token/20260718_130545_distill_room_k1/`): in-run teacher anchor
  reproduces 13.54 exactly; **carrier eval d′ 11.45 @ep9 = 96% of the scale-matched teacher
  11.94**; carrier-stack fresh-logistic tally **0.999 ± 0.001** (CPU job 123243).


## [2026-07-19] ✅📊 C1/C2 ABLATION BATTERY (7 arms, 900-train starved regime, RANKING is the deliverable) — Q-first is the single most load-bearing piece (−46%); earlier opening ≫ (L12 0.941 vs L17 0.698 vs L22 0.513); posreset mild (−4%); LoRA rank flat

> Cached digit trainer, steps8+cooc (n=1800, train 900), 8 ep, frozen e_c, one change per arm
> (jobs 124300–06) → `outputs/ladder/image_longN/cached_ablations/{base,L12,L22,r4,r16,noqfirst,noposreset}/`.
> ABSOLUTE numbers are data-starved by design — never cross-compare to the 5–6k runs.

| arm | BEST acc | arm | BEST acc |
|---|---|---|---|
| **L_OPEN=12** | **0.941 @ep8** | rank=4 | 0.694 @ep7 |
| rank=16 | 0.731 @ep7 | no-posreset | 0.669 @ep8 |
| base (L17 r8) | 0.698 @ep8 | L_OPEN=22 | 0.513 @ep8 |
| | | **no-Q-first** | **0.378 @ep8** |

- Q-first −46% matches its +3 d′ supply price (and Track B: this piece is Qwen-specific);
  posreset −4% consistent with +0.6 d′. L12 ≫ L17 ≫ L22 — caveat: LoRA params scale with open
  depth (16/11/6 layers), depth and capacity confounded; at full data L17 already reaches
  0.98–0.99, so L12 is the better default for small data (later confirmed OOD — see E-H).


## [2026-07-19] ✅📊 TRACK B — InternVL2.5-8B PORT: solo-Q-first carrier d′ **6.31/5.11 @L16/L20 vs joint 1.79/1.90 — the supply mechanism ports (3.5×)**; but vs plain solo 6.38/6.56 the Q-first AMPLIFIER does NOT port — fence/isolation is the portable piece, Q-first is Qwen-specific

- Run: `outputs/frame_axis/internvl/multipass_qfirst/20260719_004112/` (job 124280; n=200, 1600
  solo passes, same seed/data/estimator as the 118996 record — sample-matched; `--qfirst` flag in
  `experiments/internvl/multipass_bench.py`; per-frame perception acc 0.586 unchanged).
- Verdicts vs pre-registered bands: mechanism-ports GO (≥2× joint ✓ at 3.5×); amplifier band
  (≥ +20% over plain solo) FAILED — 6.31 vs 6.38 flat at L16, 5.11 vs 6.56 negative at L20.
  Honest thesis scope note: question-conditioned frame encoding is family-dependent.
- Method note: solo forwards = the fence/multipass-equivalent supply measurement (fence ≡
  multipass identity established on Qwen); no mask surgery in InternVL remote code.


## [2026-07-19] ❌📊 DIGIT-READOUT EXAMS — length collapse and zero-shot task transfer are both robust NULLS: N=32 0.092–0.097 (collapse to "0" regardless of training-N diversity); five unseen-task pairs all ≈ chance (0.087–0.179); the LoRA left on plain prompts is SAFE (0.313 vs frozen 0.219)

- **Length:** cached ckpt (0.980 @N≤8) @N=32 → **0.097** (g0 24/24, else ~0)
  (`cached_eval_N32/20260719_000556_…/`, 124275); pooled ckpt (0.999, variable-N 2..8 trained)
  @N=32 → **0.092** (`pooled_eval_N32/20260719_051622_…/`, 124353) — training-N diversity buys
  nothing; the digit readout binds to the trained carrier-position range. Steps-450 ckpt: 0.138
  (`carrier_layer_eval_N32/20260718_182252_…/`, 123742).
- **Task transfer (zero-shot, all ≈ chance 0.111–0.125):** steps→cooc 0.179 (123743) ·
  cached→rooms 0.153 (124276) · A3→NIAH 0.087 (124316) · pooled→NIAH 0.117 (124354) ·
  pooled→union 0.150 (124355). Task coverage must be trained — and the mixture rows show it
  costs nothing (no interference).
- **Drift** (`frozen_baseline_driftlora/20260719_000556/`, 124277): plain prompt + LoRA hooks ON
  = **0.313 vs 0.219** frozen (n-mismatch caveat: 300 vs 900) — slightly HELPS; no gating needed.


## [2026-07-19] ✅ STAGE-2 POOLED GO — data was the whole gap: the in-model carrier layer at 6k pooled samples reads **0.999** (steps 630/630 · rooms 108/108 · cooc 161/162) = the 0.998 scaffold, one architecture, three tasks incl. the provably-nonlinear rooms set-union; frozen-e_c cached trainer hits 0.980 at 4× speed

- **P1 pooled 3-task** (job 123741, `carrier_layer_pooled/20260718_182248_L17_r8/`; steps N=2..8
  4200 + cooc 1080 + rooms 720, train 5100/eval 900 stratified per (task,N), trainable e_c
  warm-started from the full-prior distill, L17 r8, 12 ep): **BEST 0.999 @ep12, MAE 0.00.**
  Trajectory 0.176 → 0.669(ep1) → 0.963(ep5) → 0.997(ep8) → 0.763(ep11, transient optimizer
  blip) → 0.999(ep12).
- **P1-CACHED** (job 123937, `carrier_layer_cached/20260718_192428_L17_r8/`; e_c FROZEN, layers
  ≤L16 run once and cached, steps 2–8 + cooc, ~5.1k, 934 s/ep ≈ 4× faster): **0.980 @ep10**
  (steps 0.987 / cooc 0.946), MAE 0.02 — frozen e_c matches trainable; validated workhorse.
- **Data-starvation diagnosis vindicated: 450→0.678 · 5.1k→0.980 · 6k→0.999** (scaffold 0.998,
  frozen 0.219, chance 0.111). Caveat: all trained-on-clean synthetic MMRED; scaffold ceiling
  comparison is same-data.


## [2026-07-19] ✅📊 SCRATCHPAD READOUT (A3/A4/5-task) — verdict-scratchpad targets fit in ONE epoch (TF-count 1.000), in-dist greedy 0.953–0.966 with parse-fail 0.000; a 5-task mixture (adds NIAH 0.992 + OR-union 0.910) runs on ONE carrier + ONE LoRA; untrained OR-union composes PARTIALLY (0.321 = 2.1× digit); zero-shot length 0.215 → +200 in-length samples 0.447

- **A3 train** (job 124282, `carrier_layer_scratchpad/20260719_005342_L17_r8/`, 5.8k, jitter 12):
  TF-count **1.000 @ep1**; **in-dist greedy 0.953, pf 0.000, MAE 0.05** (124314; steps 0.980 /
  cooc 0.906 / rooms 0.850). **NIAH which-frame zero-shot 0.087** ≈ chance 0.125 (124316).
- **Composition (OR-union, never trained): 0.321** (pf 0.021, MAE 1.34; hits over the FULL count
  range — not mode collapse) vs digit-ckpt 0.150 → the shared VERDICT FORMAT is what transfers
  (NIAH, fully alien, gets 0.087). Run `scratchpad_eval_union0shot/20260719_030349_…/` (124335).
- **5-task mixture** (124336, `carrier_layer_scratchpad5/20260719_031356_L17_r8/`, +NIAH 720
  +union 540): TF-count 1.000 on all 5 @ep1; **in-dist greedy 0.966** (steps 0.997 · which 0.992 ·
  cooc 0.944 · union 0.910 · rooms 0.842) (124349). Bands NIAH ≥0.90 ✓ ("easy once in mixture"),
  union ≥0.85 ✓, no regression ✓.
- **Length:** A3 N=32 zero-shot **0.215** (in-range 0.311, pf 0.000 — format fully survives where
  digit ckpts collapse to "0") (124315) → A4 fallback triggered. **A4** (+longN_16 all 330 +
  longN_32 first-200; 124362, TF 0.997 @ep3) → **N=32 held-out complement 0.447 (in-range 0.626,
  pf 0.000, MAE 1.44)** (124376, `scratchpadLN_eval_N32heldout/20260719_092517_…/`) — band
  partial (<0.80); errors are verdict undercounts, never format collapse; long-N data curve
  still steep. (A3/A4 N=128 rows dropped for budget; the wall was later measured on the tally
  arms — see the N=128 entry.)


## [2026-07-19] ✅ E-D NO-HARM — the carrier-layer LoRA left permanently ON costs nothing on general benchmarks: MME −0.2 pts, POPE −1.4 pts (band ≤2 = GO); with the drift row (plain-MMRED 0.313 vs frozen 0.219) the adapter is deployment-safe always-on

- Run: `outputs/ladder/image_longN/noharm_bench/20260719_203833/` (job 124508; 500 MME + 500 POPE
  items, identical samples both arms, Yes/No logit-argmax, le16 running-tally ckpt LoRA).

| benchmark | base | LoRA-on | Δ | band |
|---|---|---|---|---|
| MME (acc) | 0.862 | 0.860 | **−0.2 pts** | GO (≤2) |
| POPE (acc / F1) | 0.862 / 0.839 | 0.848 / 0.819 | **−1.4 pts** | GO (≤2) |

- Per-subtask deltas ~0 across 12/14 MME cells (existence −4.5 / landmark −2.4 are small-cell
  noise n≈20–40, celebrity +2.7); all POPE splits −1.1..−1.7.


## [2026-07-19] ✅📊 E-C/E-C(b) LAYOUT FREEDOM (qualified GO) — carrier tokens can sit as a PURE SUFFIX after the question (d′ 10.27, tally **0.999** = the interleaved stack), but the strong form fails: with no leading question, at-end carriers read d′ 2.40 / tally 0.508 — the binding requirement is QUESTION-FIRST, not carrier placement

> E-C strong form (job 124492, `carrier_atend/20260719_192758_distill_room_k1/`, n=900,
> frames-first teacher anchor 8.89 ± 0.14): student d′ **2.40** (27% of teacher), tally 0.508 —
> band ≥5 missed; the at-end carrier reproduces messages in bulk (MSE converged) but not the
> discriminative direction, despite an IDENTICAL allowed-key set (mask-debug: 223 keys, same).
> E-C(b) restores ONE thing — the leading question (`--atend-qfirst`, job 124514,
> `carrier_atend_qfirst/20260719_205916_distill_room_k1/`, Q-first teacher anchor 13.70).

| layout (fence+posreset, distill, N=8 full prior) | eval d′ | fresh-logistic tally |
|---|---|---|
| Q-first, carriers INTERLEAVED (123233, reference) | 11.45 (96% of teacher) | 0.999 ± 0.001 |
| **Q-first, carriers AT END (124514)** | **10.27 @ep2 (75%)** | **0.999 ± 0.001** |
| no leading question, carriers at end (124492) | 2.40 (27%) | 0.508 ± 0.017 |

- Both E-C(b) bands met (d′ ≥5 ✓, tally = interleaved ✓): the E-C failure was missing
  question-conditioned frame ENCODING (the same +3 d′ Q-first term as the C2 ablation), adjacency
  innocent. Method statement: prompt = [question][frames][question][carrier suffix] — carriers
  never interrupt user content (the deployment-friendly form).


## [2026-07-20] ✅ E-E SEEDS — headline running-tally recipe at 3 seeds: in-dist TF-count **1.000 ± 0.000** (tf-exact 0.963 ± 0.007); the zero-shot N=32 length cell 0.284 ± 0.004 over 2 seeds — the recipe is seed-stable

- Runs: headline 124482 (seed/shuffle 0, jitter 16, 1.000 @ep2) ·
  `carrier_tally_le16_seed1/20260719_203924_L17_r8/` (124509, jitter 12, 1.000 @ep3) ·
  `carrier_tally_le16_seed2/20260719_203925_L17_r8/` (124510, jitter 12, 1.000 @ep3).
- Caveat: seed arms ran jitter 12 vs headline 16 (seed+jitter-dose bundled) — indistinguishable
  both in-dist and at the N=32 cell (0.287 vs 0.280, job 124697), so jitter dose 12-vs-16 is a
  no-op here.


## [2026-07-20] 📊 ROOMS DECODE-GAP DIAGNOSTIC — every error in 40/40 held-out transcripts is a MISSING-ROOM verdict; emitted count ALWAYS equals emitted list length: the readout's counting is exact, the residual is per-frame DETECTION RECALL (supply-side), not the readout

- Run: `outputs/ladder/image_longN/rooms_gap_diag/…_evalonly/` (job 124527; 40 held-out rooms
  samples, 5-task L17 ckpt, full transcripts): acc 0.825, parse-fail 0.000, MAE 0.17. No format
  derail, no reordering, no count-list mismatch (e.g. gold 6 → "Bathroom, Bedroom, Garden,
  Office, Park -> 5").
- Same signature as the long-N misses → the future lever is carrier content, not the readout.
- Follow-up caveat (2026-07-22): the l12v2 ckpt reads rooms-100 at **1.000** (job 125108) — the
  0.84–0.85 gap was a property of the L17 5-task ckpt family, not of the method.


## [2026-07-20] ✅📊 E-B SFT CONTROL — a 23.8M-param plain LoRA (12× our budget, q/k/v/o+MLP all layers, trained N≤8) matches in-distribution (0.998–1.000) but does NOT aggregate at length: N=16 **0.480** / N=32 **0.350** with a DEAD MID-RANGE — the theory's joint-supply ceiling located behaviorally

- First run 124484 (`sft_control_le8/20260719_185022_lora/`): test_iid **0.9984** at N≤8 (best ep1)
  — but the long-N cell was BLOCKED (script never saved the adapter; `plans/carrier_stage4_BLOCKED.md`).
  Rerun with adapter-saving patch: job 124696 → `sft_control_le8_v2/20260720_191541_lora/`
  (best ep3 val 0.983, test_iid 1.000).
- **N=16 0.480 / N=32 0.350 (pf 0.000 both)** — more nuanced than the pre-registered "collapse":
  the LoRA rides EXTREME-count anchors (g1 8/8, g32 7/7 but g6–g8 0/27 at N=16; mid-range ~0 at
  N=32; MAE 1.83) — a bimodal low-end/saturation heuristic. The dead mid-range is exactly where
  per-frame aggregation is needed (joint supply d′≈2); extremes are solvable from global gist.
- Honest packaging: the carrier method's long-N edge is the in-length-trained cell (0.733 vs
  0.350 @N=32) and uniform per-count coverage; the zero-shot tally cell (0.280) does NOT beat the
  SFT control. N=64 cell OOM'd during generate (adapter saved; low priority).


## [2026-07-20] ✅📊 POSRESET NECESSITY (Tal's challenge) — minor at N=8 (no-reset d′ 7.74 vs 9.24) but the per-copy position-tax gradient RETURNS at N=64 (pooled 7.54, per-copy decays 6.4→3.0): KEEP posreset, re-justified as N-scaling-critical; the historical "+0.59" justification retired

- Runs: `outputs/ladder/image_longN/noreset_N{8,64}/20260720_*/` (jobs 124713/124714) — Q-first
  blockfence WITHOUT `--reset-positions`; comparators with-reset 9.24 (N=8) and ~12 @N=64
  (bracketed 12.67@32 / 11.57@128).
- **N=8: 7.74 ± 0.18** — minor cost, and > fence-level 6.34 (Q-first partially substitutes for
  reset). **N=64: pooled 7.54 is carried by early frames — per-copy decays 6.4 → ~3.0** with frame
  index (the A2 position-tax fingerprint); extrapolated, late-frame supply at N=128 approaches
  joint level.
- Verdict (pre-registered intermediate band): keep — free at inference, increasingly load-bearing
  with N, and required infrastructure for jitter + position-coupling experiments.


## [2026-07-20→21] ⚠️📊 E-A LENGTH LADDER + THE N=128 WALL — running-tally readout decays smoothly zero-shot (1× 1.000 → 2× 0.280 → 4× 0.150 → **8× 0.087: headline band ≥0.80 REFUTED**); in-length training is the working lever (le64: N=32 **0.733**, N=64 0.558/0.821-in-range) — the honest claim is a LADDER in trained-length coverage, not extrapolation

> Arms: le16 (124482, `carrier_tally_le16/20260719_184950_L17_r8/`, 11 roots + longN_16,
> running-tally + jitter 16, TF-count 1.000 @ep2) · le64 (124483, `carrier_tally_le64/…/`,
> +longN_32/64, TF 1.000 @ep4). Exams: 124522/578/736 (le16), 124571/586 (le64), 124758 (v2→128).

| ckpt \ exam | N=32 | N=64 | N=128 |
|---|---|---|---|
| le16 (trained ≤16), zero-shot | 0.280 (pf 0.007; in-range 0.394) | 0.150 (pf 0.133) | **0.087** (first-match 0.130, pf 0.087) |
| le64 (trained ≤64), held-out | **0.733** (pf 0, MAE 0.43, g32 9/9) | 0.558 (in-range 0.821) | 0.118 (v2 ckpt, 124758) |

- **The headline cell** (`tally16_eval_N128/`, job 124736, TIMEOUT @23/34 — recovered from the
  --dump-decodes safety net, per-sample gold/parsed in `logs/cl_eval-124736.out`): train-≤16 →
  emit-at-128 (8×) does NOT hold. Transcripts show tally-index confusion (verdict indices leak
  into tally slots: "frames 30 (11), 31 (112)…") and unterminated chains. Trained-to-≤64 → 128
  (2×) also fails the ≥0.35 band (0.118; ckpt-selection confound caveat, gap to band large).
- le64's N=64 g48/g64 cells are cap-truncations (280-token decode < the 48–64-verdict tally) —
  annotated as unmeasurable-under-cap, not model failures. le16 N=64 failure mode = repetition
  loops. Parse sensitivity: first-vs-last-match differs ≤1 sample/cell (last-match stays primary).
- Zero-shot N=32 stable across seeds: **0.284 ± 0.004** (124522 headline jitter-16 0.280;
  124697 seed1 jitter-12 0.287). Successor arms (l12v2 0.953 in-length / 0.286@128, E-G refuted)
  logged above.


## [2026-07-22] ❌📊 E-G POSITION-COUPLED TALLY REFUTED — coupling decoded-token positions to carrier anchors costs in-distribution fit (persistent over 8 ep) and BREAKS the format at 4× length: N=32 0.527 / N=64 0.212 (pf 0.365) vs uncoupled l12v2 0.953 / 0.615 on IDENTICAL dirs

> `couple_offsets` in `carrier_layer_lora.py` + `--pos-couple` (one rule drives teacher-forced AND
> online decode positions; CPU-verified anchor rule + couple-debug — this is a MECHANISM negative,
> not an implementation bug). First train 124701 (`carrier_tally_pcouple/`) TF-undertrained at 4 ep
> (0.923/0.720, still climbing); converged rerun 124774 → `carrier_tally_pcouple8/20260721_071710_L17_r8/`,
> in-dist ceiling **0.955 / tf-exact 0.832 @ep5** (oscillatory) vs uncoupled 1.000/0.976 — ~2× the
> epochs for the same TF level. Exams 124922/23/24 → `tallyPC8_eval_N{32,64,128}heldout/`.

| cell (identical dirs) | E-G (coupled) | l12v2 (uncoupled best) | v2 (uncoupled same-L17) |
|---|---|---|---|
| in-dist TF / tf-exact | 0.955 / 0.832 | 1.000 / 0.976 | 1.000 / 0.916 |
| N=32 held-out | 0.527 (pf 0.040) | **0.953 (pf 0.000)** | 0.607 |
| N=64 held-out | 0.212 (pf 0.365, MAE 44) | **0.615 (pf 0.135)** | 0.365 |
| N=128 | PARTIAL 0.154 (4/26, pf 0.500) | PARTIAL 0.286 (pf 0.000) | 0.118 |

- **Pre-registered GO ("beats uncoupled at EVERY OOD length, pf ~0") refuted at every testable
  cell**; N=64 transcripts show token-salad degeneration ("28 (10 (10), 32 (11)…") — the opposite
  of the design goal (it was built against the tally-index confusion seen at 8×). Forcing decoded
  positions to ride carrier anchors degrades the LM's own sequential coherence more than it helps
  length binding. N=128 full cell (requeued 124924) cannot change the verdict.


## [2026-07-22] ✅📊 l12v2 — L12 + in-length data + FIXED ckpt criterion = the campaign-best in-model long-N readout: held-out **N=32 0.953 · N=48 0.878 cap-adj · N=64 0.678 cap-adj**; N=128 PARTIAL 0.286 with format fully intact (best 128 reading so far)

> Two measured levers composed: (1) **L12 depth** — le16-recipe L12 arm (124698,
> `carrier_tally_le16_L12/20260720_192738_L12_r8/`) reads N=32 zero-shot **0.443** (pf 0, MAE 1.15)
> vs L17's 0.280 (job 124727, `tallyL12_eval_N32/20260720_223517_L12_r8_evalonly/`; band ≥0.40 GO →
> L12 = new default); (2) **in-length data** (le64v2 roots + `mmred_longN_park2` N=32×312 + N=48×210).
> Trainer 124773 → `carrier_tally_l12v2/20260721_071710_L12_r8/`: TF 1.000 / tf-exact 0.976 @ep5,
> the strongest ckpt of the campaign. Exams 124904/05/06 → `tallyL12v2_eval_N{32,48,64}heldout/`.

- **N=32 held-out (150 dirs): 0.953, pf 0.000, MAE 0.05** — band ≥0.85 MET, thesis-grade; per-count
  near-uniform incl. multi-digit (g12 4/4 · g16 8/9 · g24 10/12 · g32 9/10). Trained-at-length
  progression: A4 scratchpad 0.447 → le64 tally 0.733 → **l12v2 0.953**.
- **N=48 (109): 0.789 raw / 0.878 cap-adjusted** (all pf = g48 cap truncations). **N=64 (52):
  0.615 raw / 0.678 cap-adjusted** (excl. g48/64, unmeasurable under dec 280) — met vs the v2 band
  ≥0.65 cap-adjusted, just under the stricter l12v2 prereg ≥0.70.
- **N=128 (2× beyond max trained): PARTIAL 0.286 (4/14), parse-fail 0.000** — job preempted by
  h200-dds mid-run, recovered from dumps (stratified-order prefix, unbiased); best N=128 of the
  campaign (le16 0.087 · v2 0.118) and the FIRST with intact format at 128; misses are undercounts
  at g≥5, not derails. Full-34 report = requeued job 124907.
- Caveat / confound exhibit: the earlier v2 arm (124682, old acc-only save criterion, ckpt ep3
  tf 0.891) read 0.607/0.514/0.365/0.118 at N=32/48/64/128 (124755/56/57/58) — kept as the
  ckpt-selection confound exhibit; trainer save criterion is now (TF-count, tf-exact) lexicographic.


## [2026-07-23] ✅📊 E-H SEPARATOR-LAYER (L*) CURVE COMPLETE — inverted-U confirmed, peak at L_OPEN=12: zero-shot N=32 0.277/0.373/**0.443**/0.330/0.280/0.273 for L*=8/10/12/14/17/20 — ~12 fenced supply layers suffice, every remaining layer is wanted for trained integration; L12 STANDS as the default

> Four new arms (ONLY `--l-open` varies), headline ≤16 running-tally recipe, fixed (acc, tf-exact)
> save criterion. Trainers: L8 124917 / L10 124918 / L14 124919 / L20 124920; zero-shot exams
> jobs 124965–972 → `outputs/ladder/image_longN/tallyL{8,10,14,20}_eval_N{32,64}/`. Reference
> cells: L12 (124698/124727), L17 (124482/124522) — recipe-matched, but predate the fixed save
> criterion (caveat logged).

| L* (open layer) | in-dist TF / tf-exact | N=32 zero-shot (n=300) | pf | N=64 zero-shot (n=60) | pf |
|---|---|---|---|---|---|
| 8 | 1.000 / 0.978 | 0.277 | 0.017 | 0.133 | 0.183 |
| 10 | 0.999 / 0.927 | 0.373 | 0.000 | 0.217 | 0.150 |
| **12** | 1.000 / 0.991 | **0.443** | 0.000 | (not measured, le16 recipe) | — |
| 14 | 1.000 / 0.997 | 0.330 | 0.000 | 0.217 | 0.150 |
| 17 (ref) | 1.000 / 0.977 | 0.280 | 0.007 | 0.150 | 0.133 |
| 20 | 1.000 / 0.996 | 0.273 | 0.107 | 0.183 | 0.300 |

- In-dist TF saturates for ALL L* (pre-registered: not the verdict metric). L20's parse-fail is
  the worst of any arm and concentrates at high counts (N=32: g24 7/23, g32 17/23 fails) — late
  opening leaves too few integration layers to keep the format coherent at long N.
- **Decision rule (LOTO/p0p2 briefs): no arm beats L12's 0.443 by >0.05 with pf ≤0.02 →
  L_OPEN=12 stands** for the seed retrains and LOTO. L12's N=64 cell and 2 extra seeds ON HOLD
  per Tal's no-new-launches instruction.


## [2026-07-24] ✅📊 FORMAT SWEEP COMPLETE (arms A–D) — the gold scratchpad TEXT alone is worth +0.19–0.37 at held-out lengths: full-scan formats crush the positive-list (B scan N=32 **1.000**, N=48 **0.982**; C caption N=64 **0.981**, pf 0.000 everywhere); WINNER = C (caption); chunking NO

> Prereg `plans/scratchpad_format_PREREG.md` (bands fixed pre-GPU); arms differ ONLY in gold
> scratchpad text (`--scratchpad-format`), l12v2 recipe/data/split verbatim — eval/train dirs-files
> byte-identical across arms. Trainers 125104/05/06 →
> `outputs/ladder/image_longN/carrier_fmt_{scan,caption,chunked}/20260722_2220*_L12_r8/`; exams
> 125107/08 + 125183–99 on arm A's dirs-files (`fmt{B,C,D}_eval_*` + `tallyL12v2_eval_*` siblings).

| cell (identical dirs) | A poslist | B scan | C caption (WINNER) | D chunked |
|---|---|---|---|---|
| TF-fit (acc / tf-exact) | 1.000 / 0.976 | 0.999 / 0.996 | 0.999 / 0.994 | 1.000 / 0.904 |
| in-dist-150 | 1.000 | 1.000 | 1.000 | 0.987 |
| rooms-100 | 1.000 | 1.000 | 1.000 | 0.920 |
| N=32 held-out (150) | 0.953 | **1.000** (125194) | 0.987 (125195) | 0.907 (125185) |
| N=48 held-out (109) | 0.789 / 0.878 cap-adj | **0.982** (125196) | 0.972 (125197) | 0.679 (125186) |
| N=64 held-out (52) | 0.615 / 0.711 cap-adj | 0.942 / 0.956 cap-adj (125198) | **0.981** (125199) | 0.615 (125187) |
| worst parse-fail | 0.135 (N=64) | 0.019 | **0.000** | 0.000 |

- **All five prereg bands decided:** in-dist sanity ≥0.90 MET ×4 · **scan GO** (B N=64 cap-adj 0.956
  ≥ 0.761) · **agnostic-caption GO** (C≡B: in-dist/rooms exact parity, N=32 −0.013, N=64 +0.039 in
  C's favor) · **chunking NO** (D 0.615 < max(A,B)) · rooms-ordering parity-only (control A also
  1.000 — the 0.84 gap was the L17 5-task ckpt, not reproducible here).
- **Winner = C (caption)**, ckpt `carrier_fmt_caption/20260722_222032_L12_r8/carrier_layer_best.pt`:
  takes the hardest cell (0.981 vs B 0.942) with pf 0 at every length; B within noise overall
  (length-cell mean 0.975 vs 0.980) — C chosen on the primary N=64 cell + parse robustness + the
  agnosticism property (attribute words in the scratchpad). Mechanism reading: a slot per frame
  converts the long-N search burden into a deterministic frame-order scan; D's subtotals kill
  truncation (dec-means 55/75/101, pf 0) but cost accuracy at every length.
- Cost note: scan-family decode ≈ 2.4× poslist in-dist (~47 vs ~20 tok); N=48/64 exams ≈ 18h each
  on l40s (~21 min/sample @dec620) — a100 ≈ 2× faster.


## [2026-07-23] ✅📊 P1.2 — MEASURED BEFORE-CEILING: the best sample-disjoint linear readout of the summed joint-carrier messages lands within 0.01–0.05 of the zero-parameter law prediction at every N — the "squashed readout" curve is now measured, not just predicted

> Job 125259 (CPU, `probe_dprime_parity.py --carrier-caches`, deployed locus L16/off9, existing joint
> caches `image_longN/joint/N{8,16,32,64,128}/20260710_2154*/count/`, 60/40 sample-disjoint split,
> seeds 0–2) → `outputs/ladder/image_longN/measured_ceiling/20260723_222428/`.

| N | law-pred (iid) | MEASURED best linear | frozen model (same caches) |
|---|---|---|---|
| 8 | 0.307 | **0.317** (ridge, ±0.036) | 0.207 |
| 16 | 0.246 | **0.281** (logit) | 0.127 |
| 32 | 0.175 | **0.189** (logit) | 0.053 |
| 64 | 0.137 | **0.183** (logit) | 0.040 |
| 128 | 0.096 | **0.122** (logit) | 0.013 |

- Slightly ABOVE the law at N≥64 — the logistic readout exploits the non-Gaussian tail the law ignores
  (adequacy kurtosis +1.8→+14.0, as in [2026-07-11e/n]); MLP−linear ≤0.006 (E3 sufficiency); d′_w flat
  ~2.0 to N=64 (1.6 @128), replicating B1. Thesis reading: frozen < measured-linear < trained scaffold
  (0.95–1.00 @N=32) — readout misalignment vs supply repair, both gaps now measured.
- Fig: `outputs/_scratch/figs/pre_stage1_squashed_readout_measured.png`. Caveat: gold≥1 convention at
  N≥2; n = 300/300/300/200/150 → wider bars at N=128 (±0.031).


## [2026-07-23] ✅📊 P1.3 — E-B SFT BASELINE N=64 CELL (the OOM'd leg, rerun): 0.220 — the bimodal extremes heuristic extends to 8× training length; the SFT ladder is complete at N=16 0.480 / N=32 0.350 / N=64 0.220 vs the carrier readout's 0.953/0.878/0.678

> Job 125267 → `outputs/ladder/image_longN/sft_control_le8_v2_evalN64/20260723_225940_lora/`;
> eval-only restore of `sft_control_le8_v2/20260720_191541_lora/adapter`, LIMIT 100, same
> stratified-prefix sampling as the N=16/32 cells. Adapter-restore sanity: test_iid 1.0000 = the
> original run byte-exact.

- **N=64: 0.220, parse-fail 0.000, MAE 3.46**; per-count g0 4/7 · g1 5/7 · g64 6/6, mid-range
  g4–g48 ≈ 0 — no collapse, no aggregation, the dead-mid-range supply ceiling unchanged at 8×.
- Ops lesson (new): the 124696 OOM = mask-None generate → `enable_gqa=True` → mem-efficient sdpa
  ineligible (num_heads mismatch on dense inputs) → MATH materializes 17GB. **FLASH handles
  GQA+causal at 8.3 GiB peak @seq 12.7k** (smoke 125263). `lora_sft_baseline.py` eval path now
  [FLASH, EFFICIENT, MATH] + `--eval-only-adapter`.


## [2026-07-24] ✅📊 P2b — MLVU-AC ZERO-SHOT CARRIER CELL (32-frame arm): MCQ nearest-option 0.107 ≤ frozen 0.282 — the pre-registered "domain gap measured" outcome; the readout FORMAT transfers, the trained evidence detector does not fire outside the MMRED render domain

> Job 125350 → `outputs/ladder/mlvu_ac/carrier_eval_N32/20260724_064754_L12_r8_evalonly/` (+
> `mcq_mapping.txt`); winner ckpt (C caption) on all 206 MLVU-AC questions, 32f @392px. Dense N=128
> ruled prohibitive pre-launch (~20 min/sample × 206 > 60h) → prereg 32f fallback with the
> evidence-delivery caveat ([2026-07-11c]: ~0.37 visible frames per gold instance at N=32).

- **Open emitted count 0.000 exact** (pf 0.214, MAE 2.99); 161/206 emit "0", 44 parse-fail, 1×"1".
  **MCQ nearest-option (prereg rule, parse-fail = wrong): 22/206 = 0.107** (g1 12/37 · g2 8/52 ·
  g3 2/45 · g4 0/33 · g5 0/39). Band: ≤ frozen 0.282 → **transfer NO, domain gap measured**.
  Protocol note (by design): this cell never sees the MCQ options; the frozen 0.282 had them
  in-prompt — chance structure differs.
- Failure anatomy (206 dumped transcripts): scan/caption structure emitted and mostly well-formed;
  content collapses to all-negative verdicts, and the parse-fails are fluent refusals that correctly
  DESCRIBE the frames ("the image shows a zebra … no 'making jewelry' action"). Perception works;
  the trained evidence-detection channel is domain-bound — consistent with the cross-domain carrier
  result (~51% of teacher, [2026-07-18]) and the delivered-evidence ceiling.

## [2026-07-24] ✅📊 P3b — InternVL2.5-8B SCAFFOLD-LEVEL GATE→TALLY: 0.938 ± 0.031 exact @N=8 (L16; per-frame gate 0.991, majority 0.160) — the GNN scaffold (per-frame messages + linear gate + sum) PORTS across model families, at the multipass-isolated supply level

> CPU fit (`experiments/glstm/internvl_gate_tally.py`) on the existing cache of job 124280
> (`outputs/frame_axis/internvl/multipass_qfirst/20260719_004112/bench_cache.pt`) →
> `outputs/frame_axis/internvl/gate_tally/20260724_165356/`; logistic gate, sample-disjoint 60/40,
> seeds 0–2, tally = Σ verdicts. L20: 0.892 ± 0.012.

- Band ≥0.90 MET at L16 → scaffold ports. **Honest label: multipass-isolated Q-first supply** (each
  frame solo in its own forward), one rung below Qwen's one-forward blockfence cell.
- Cross-family replication of the readout-misalignment signature: InternVL's own per-frame digit
  readout is 0.586, while the linear gate on its carrier messages reads 0.991.

## [2026-07-24] ✅📊 P4.3 — NO-HARM ON THE PLAIN SFT ADAPTER: MME −0.6 / POPE +1.2 pts (band |Δ|≤2 → GO); predicted digit-on-yes/no failure REFUTED — both adapter families are safe always-on

> Job 125499 → `outputs/ladder/image_longN/noharm_bench_sft/20260724_190307/`; same 500+500
> MME/POPE protocol/seed as the carrier cell (124508, ref −0.2/−1.4); new `--peft-adapter` arm +
> ≤20 failure dumps in `logs/p43_noharm-125499.out`.

- MME 0.862→0.856, POPE 0.862→0.874. Small-n category texture: OCR −20 / celebrity −13.3 vs
  count +16.7 / commonsense +12.5. Fail dumps emit clean yes/no words — no digit contamination.

## [2026-07-24] ✅📊 P3a — NATURAL-IMAGES MMRED FULL LADDER: supply GO (one-forward d′ 27.3 = 3.5× joint) · scaffold GO (linear gate→tally 0.980 ± 0.012) · IN-MODEL NO-GO (0.145–0.289, below the frozen floor, pure-extremes anatomy) — the cross-domain failure is localized to the in-model readout rung, not the mechanism

> Prereg `plans/p3a_natural_PREREG.md`. Data: `data/mmred_natural_mm` composed from the judge-gated
> `mmred_natural_v2` pools (global image-half split — train/eval image-disjoint; builder
> `natural_compose_mmred.py`, BUILD_INFO in the root). Runs: L0 125488 (`natural_mm/frozen_baseline/`),
> L1 125486 (`natural_mm/replica_supply_dist_far/20260724_180046/`), L2 CPU (`…/gate_tally/`),
> L3 trainer 125487 (`natural_mm/carrier_caption_nat/20260724_180058_L12_r8/`) + exams 125492–95
> (`natural_mm/nat_eval_N{8,16}{far,near}/`).

| rung | natural | park ref | band | verdict |
|---|---|---|---|---|
| L0 frozen | 0.563/0.407 @N=8 · 0.422/0.311 @N=16 (g≤8, digit protocol) | 0.219 | ref | — |
| L1 supply | d′_w 27.3 (joint anchor 7.75; per-copy flat 8.5–13.7; AUC-cap caveat) | 13.5 | ≥4.0 & ≥2× joint | **GO** |
| L2 scaffold | **0.980 ± 0.012** (gate err 0.0025, majority 0.187, n=300) | 0.998 | ≥0.85 | **GO** |
| L3 in-model | 0.289/0.259 @N=8 · 0.155/0.145 @N=16 (pooled 0.218, pf 0) | 0.987–1.000 | GO ≥0.80 / NO-GO ≤L0+0.10 | **NO-GO** |

- L3 anatomy is PURE EXTREMES (g0 perfect, gmax mostly right, every intermediate 0) — the bimodal
  signature; the trainer's TF-count 0.996 rode the gold tally prefix (tf-exact 0.187 was the tell).
- Reading: natural per-frame evidence is EASY (high frozen floor, huge d′) and the GNN mechanism is
  fully intact off-domain — what fails is the in-model rung (park-distilled carrier e_c and/or
  LoRA-through-frozen-layers), consistent with the ~51% cross-domain carrier and the MLVU 0.107 cell.
  Successor experiment (not launched): natural-distilled e_c to split e_c from the integration.

## [2026-07-24] ✅ P1.1 SEEDS — the caption-winner recipe at 3 seeds: N=32 held-out **0.982 ± 0.007** (seed0 0.987 · seed1 0.987 · seed2 0.973; parse-fail 0 ×3; identical 150 dirs) — the headline in-model long-N readout is seed-stable

> Seed trainers 125347/48 → `outputs/ladder/image_longN/carrier_fmt_caption_seed1/20260724_064754_L12_r8/`
> (BEST 0.999 / tf-exact 0.993 @ep4) and `carrier_fmt_caption_seed2/20260724_064743_L12_r8/`
> (0.999 / 0.991 @ep3); N=32 exams on arm-A's `eval_dirs_N32all.txt` (byte-identity split gates
> passed, per `plans/p0p2_STATE.md`) → `fmtCseed1_eval_N32heldout/20260724_213236_L12_r8_evalonly/`
> **0.987** (pf 0.000, MAE 0.01) and `fmtCseed2_eval_N32heldout/20260724_211258_L12_r8_evalonly/`
> **0.973** (pf 0.000, MAE 0.03). Seed0 = the format-sweep winner cell (125195, 0.987).

- Logged during the 2026-07-29 migration from `plans/p0p2_STATE.md` (campaign ended before logging);
  this is the P1.1 cell referenced by the P4 entry's "seeds 0.982±0.007" row.

## [2026-07-24] ⚠️📊 P2a — LOTO (leave-cooc-out) ZERO-SHOT TASK TRANSFER: 4-task variety buys PARTIAL transfer at N=8 (0.403 = 2.3× the best no-variety prior, pf 0) but NO-GO at the doubly-zero-shot N=32 cell (0.107 ≤ frozen+0.10); the all-5 skyline reads the identical cells at 0.997 / 0.629 — task coverage must still be trained; held-in tasks fully intact (arm-2 gate 1.000)

> Prereg `plans/scratchpad_loto_PREREG.md` (bands fixed 2026-07-23 pre-GPU; Arm 5 dropped by
> amendment — union is a training root of the resolved recipe; Arm-1 N=32 is doubly zero-shot, task
> AND length, matched with the skyline). T1 writer job 125349 →
> `outputs/ladder/image_longN/carrier_loto_nococ/20260724_064756_L12_r8/` (15 roots = the 16-root
> mixture minus `mmred_cooc_balanced`, n=7872, BEST 0.998 / tf-exact 0.993 @ep5; split gate amended +
> passed per `plans/p0p2_STATE.md`). Exams jobs 125500–505 (2026-07-24); frozen floors via
> `runners/p2a_frozen_cooc.sbatch`. Logged during the 2026-07-29 migration — reports on disk, never drafted.

| arm (identical dirs per cell) | cooc N=8 (n=300) | cooc N=32 (n=299) |
|---|---|---|
| Arm 1 — T1 LOTO (never saw cooc) | **0.403** (pf 0.000, MAE 1.03) | **0.107** (pf 0.000, MAE 4.78) |
| Arm 3 — skyline (all-5 winner ckpt) | 0.997 (pf 0.000, MAE 0.00) | 0.629 (pf 0.003, MAE 0.70) |
| Arm 4 — frozen floor | 0.130 | 0.130 (n=207, g≤8 digit-protocol) |
| prereg bands | GO ≥0.7×skyline = 0.698 · NO-GO ≤ floor+0.10 = 0.230 → **PARTIAL** | GO ≥0.440 · NO-GO ≤0.230 → **NO-GO** |

- Arm 2 in-dist gate (T1 ckpt, the 120 non-cooc rows of arm-A's held-out `eval_dirs_indist150`):
  **1.000, pf 0.000** (rooms 30/30 · steps 30/30 · union 30/30 · which 30/30) →
  `loto_arm2_indist120/20260724_203510_L12_r8_evalonly/` — the writer is intact; the transfer gap is
  real, not trainer breakage.
- Priors context: without mixture variety, in-model zero-shot task transfer was ≈ chance
  (steps→cooc 0.179 · cached→rooms 0.153, [2026-07-19]) — 4-task variety more than doubles the N=8
  cell (0.403) but stays far from the trained skyline; the relational pairwise predicate does not
  come free. Run dirs: `loto_arm1_coocN{8,32}/`, `loto_arm3_coocN{8,32}/`, `loto_frozen_coocN{8,32}/`.
- Caveats: the frozen reports' header `data=` string prints the script default — the runner passes
  the shared cooc dirs-files (`--dirs-file`, verified); frozen N=32 covers the g≤8 subset (207/299,
  gold>9 skipped by the digit protocol, documented in-runner) while Arms 1/3 answer all 299; first
  frozen attempts (n=47/23, same dirs) superseded by the full reruns.

## [2026-07-25] ✅📊 P4 — READOUT-SIMPLICITY CONTROLS (3 pre-registered cells): the ≤8-trained SFT baseline was a DATA artifact — plain LoRA SFT trained in-length reads N=32 at 0.967 ("simple-fix-wins" band, logged honestly); but carriers+digit with the same data collapse (0.333/0.140, theory-confirmed), and the SFT path needed an h200 to train at all

> Prereg `plans/p4_PREREG.md` (bands fixed pre-trainer; P4.1 amended to trained-≤32 after 4
> documented OOM/skip-cascade attempts). Runs: P4.1 trainer 125567 (h200, 0 skips) →
> `sft_inlength_p41/20260725_031153_lora/` + exams 125620 (`sft_inlength_p41_exams/`); P4.2 trainer
> 125498 → `carrier_digit_inlength/20260724_202048_L12_r8/` + exams 125610/11
> (`p42digit_eval_N{32,64}/`); P4.3 = 125499 (`noharm_bench_sft/20260724_190307/`).

| cell | N=32 | N=64 | verdict vs prereg band |
|---|---|---|---|
| P4.1 plain SFT, in-length (≤32) | **0.967** (pf 0, per-count uniform) | 0.787 (extrapolation — N=64 training fits no available GPU) | **≥0.90 → simple-fix-wins, logged honestly** |
| P4.2 carriers + digit, in-length | **0.333** (dead ≥g4) | **0.140** (dead ≥g3) | **≤0.50 + dead mid-range → theory-confirmed** |
| caption winner (ref) | 0.987 (seeds 0.982±0.007) | 0.981 (in-length) | — |
| P4.3 SFT no-harm | MME −0.6 / POPE +1.2 pts | — | **GO** (≤2 pts; digit-on-yes/no failure refuted) |

- **Reframing (per the prereg's own terms):** the old "SFT ladder 0.480/0.350/0.220" strongest-baseline
  row is RETIRED — it measured missing in-length data, not a readout limit. What survives as the
  thesis contribution: (1) the P4.2 asymmetry — a 2M-param carrier LoRA with a single-token readout
  cannot use the same data (readout expressivity is the separator); (2) one caption model serves 5
  tasks + partial LOTO transfer vs single-task SFT (script-limited by design); (3) measured training
  cost — SFT@N=32 trains only on a 140GB h200 (5 attempts documented: masked-forward→MATH 45.6GiB,
  ckpt-recompute OOM, skip-cascade silent data loss), while the cached carrier trainer does ≤64 on a
  40GB a100; (4) N=64: caption 0.981 in-length vs SFT 0.787 structurally-extrapolated.
- Split-drift discipline (new standing lesson): BOTH P4 trainers redraw splits (gold>9 prep-skip /
  declare_splits) — every exam dirs-file was contamination-checked and rebuilt from provably-unseen
  dirs where needed (`eval_dirs_p42_N{32,64}.txt`, `eval_dirs_p41_N32.txt`); the arm-A N=64 file is
  clean for P4.1 (nothing trained at 64).

## [2026-07-25] ⚠️📊 TRUNC CAMPAIGN (E1–E7) — the caption winner's decode READS FRAMES (kvdrop changes 15/16 transcripts at the first verdict token): prefill aggregation and decode readout are SEPARATE channels; truncation cannot be bolted on — not eval-only (0.047/0.040/0.019), not retrained (greedy 0.093–0.153, the per-slot ADDRESSING wall), not via external readout on truncated carriers (~0.05); measured cause: the per-frame gate is WRITTEN in layers 12–19 (err 0.339@L12 → 0.0051@L24). What SURVIVES: the EXACT cached-fast decode, 16.2×–98.9× per-sample

> Prereg `plans/trunc_PREREG.md` + amendment; campaign state `plans/trunc_STATE.md` (2026-07-25).
> Root cause (P0.1, code truth + executable smoke): `build_block_mask` never hid frames from
> tail/decode rows, and the cached trainer teacher-forced targets WITH that visibility — the LoRA
> learned decode-time frame reads. Logged during the 2026-07-29 migration (campaign ended without
> logging; only E1 existed as a draft candidate).

| cell | number | run dir (jobs) |
|---|---|---|
| E1 exactness / kvdrop (winner ckpt) | mask-only identical **1/16**, answer-equal 1/16 (N-mix, base acc 1.000); 0/4 @N=64; **fast≡mask 18/20**; decode 59.7→3.7 s/sample (**16.2×**) · 657.1→6.6 (**98.9×**); keep 103/12,775 tok @N=64 | `trunc_kvdrop/e1{a,b}/20260725_*_evalonly/` (125554/55) |
| E2 eval-only truncate@12 | **FAIL ×3: 0.047 / 0.040 / 0.019** (in-dist150 / N32 / N64; pf 0 — format survives, evidence gone; band Δ vs 1.000/0.987/0.981) | `trunc_at12/{indist150,N32,N64}/` (125562/63/64) |
| E3 eval-only truncate@L, N=32 | flat-low: 0.033 (L14) / 0.107 (L16) / 0.073 (L20) / 0.073 (L24) | `trunc_sweep/L{14,16,20,24}/` (125565/71/72/73) |
| E4 deploy-matched retrain, caption t12 | TF-count 0.999 but tf-exact plateaus ≤0.165; exams **in-dist 0.133 / N32 0.073 / N64 0.096 = FAIL** (greedy all-or-nothing) | `trunc_retrain/carrier_caption_trunc12/` + `exam_{indist150,N32,N64}/` (125570, 125605/06) |
| E4b scan retry, t12 | tf-exact ~0.10–0.12; exam **in-dist 0.093 = FAIL** — presence-only verdicts don't fix addressing | `trunc_retrain/carrier_scan_trunc12/` + `exam_scan_indist150/` (125609) |
| E4c L*=20 + truncate@20 | tf-exact 0.223 by ep2; exams **0.153 / 0.140 / 0.115 = FAIL** | `trunc_retrain/carrier_caption_trunc20/` + `exam_t20_{indist150,N32,N64}/` (125628) |
| Fallback: truncated carriers + EXTERNAL gate→tally | **FAILS: per-frame err 0.33–0.35, tally exact 0.051–0.059** @L12/16/20 (truncated cache) | `trunc_retrain/hybrid_dump_N32/…/gate_tally_L{12,16,20}/` |
| Saturation depth (NON-truncated cache) | per-frame gate err **0.339 (L12) → 0.306 (L13) → 0.216 (L14) → 0.173 (L16) → 0.0082 (L20) → 0.0051 (L24)**; external tally 0.843 @L20 · **0.909 ± 0.016 @L24**; truncated@12 reference stays ~0.33–0.37 at all layers | `trunc_retrain/hybrid_dump_N32_notrunc/…/saturation_probe_report.txt` + `saturation_curve.png` (dumps 125615/21) |
| E5 chunked prefill | **STRUCTURAL PASS**: L0 delta exactly 0; carriers ≤0.21 abs; question dq=34 = bf16 noise on attention-sink dims (~0.7% rel); tail Δ real (prereg'd); E4-ckpt rerun: question 8.36 / carriers 0.126 | `trunc_bench/chunkverify{,_e4}/` (125566, 125568/69 dbg) |
| E6 bench (winner ckpt, cached-fast EXACT path) | speedup base/fast **1.9× (N=8) · 32.3× (N=32) · 95.4× (N=64) · 311.4× (N=128)** (decode s/sample 5.2→2.7 · 96.4→3.0 · 667.3→7.0 · 3546.9→11.4); VRAM ≈ base; n=3/cell — engineering numbers | `trunc_bench/runA_N{8,32,64,128}/` |

- **Verdict chain:** frames were never hidden from decode rows → the winner reads frames at decode →
  dropping frame KV after L* fails every route: eval-only (E2/E3), retrained in-model (E4/E4b/E4c —
  per-slot carrier verdicts unreliable even teacher-forced, tf-exact ≤0.223, although the info is
  linearly present in carriers pre-truncation: gate→tally 0.99 non-truncated), and external readout
  on truncated carriers (err ~0.35). Measured reason: per-frame evidence consolidates into carriers
  through layers 12–19 via the retained carrier→own-frame edges — truncating at 12 removes frames
  before the gate is written; the earliest viable L_trunc is ≥20, where E4c still fails the readout.
- **E7** (N=256 supply + the 0.88 prediction) was gated on E4 GO → **gated-out, not run**. runB/C t20
  deploy benches recorded for timing only (decode 13.9 / 95.2 s/sample @N=32/128; chunked arm 14.1
  vs 18.0 GiB VRAM) — their acc cells reflect the E4c FAIL and are not additional evidence.
- **What survives for the thesis:** (1) the two-channel mechanism finding (prefill aggregation vs
  decode readout); (2) the saturation-depth curve — where the gate gets written; (3) the exact
  cached-fast decode (fast≡mask, no accuracy trade) with 16–311× decode speedups and keep=103/12,775
  tokens @N=64; (4) the honest negative: one-token-per-frame KV compression is not reachable for
  this ckpt family by truncation.

## [2026-07-27] ✅📊 POSRESET DOSE-RESPONSE (N=2→32) + BEHAVIORAL NECESSITY — the posreset supply advantage GROWS with N (replica d′ @L16: 8.05 / 9.70 / 12.06 / 12.57 with reset vs 7.77 / 8.45 / 9.05 / 8.51 without; late-copy decay is the no-reset fingerprint), and the caption winner evaluated WITHOUT reset COLLAPSES: N=32 0.313 (pf 0.207) vs 0.987 · N=64 0.000 (pf 0.923) vs 0.981

> Supply probes (Q-first blockfence, in-run joint anchors, L14/L16):
> `outputs/ladder/image_longN/posreset_qf_N{2,4,16,32}/20260727_*/` vs `noreset_N{2,4,16,32}/20260727_*/`.
> Behavioral: `fmtC_noreset_eval_N{32,64}/20260727_*_evalonly/` — the `carrier_fmt_caption` winner
> ckpt with reset OFF at eval, identical arm-A dirs-files. Also produced: a carrier-states depth dump
> `carrier_depth_L12_N32/20260727_220340_L12_r8_evalonly/` (n=150, layers 2–24,
> replica_gate_tally-compatible; decode≤48 dump artifact — its acc row is meaningless by design).
> Logged during the 2026-07-29 migration — this wave (post-STATE, 2026-07-27) was recorded nowhere.

| N | with posreset d′ (L16) | without (L16) | Δ | no-reset per-copy tail (L16) |
|---|---|---|---|---|
| 2 | 8.05 ± 0.35 | 7.77 ± 0.02 | +0.3 | flat |
| 4 | 9.70 ± 0.16 | 8.45 ± 0.16 | +1.2 | mild decay |
| 16 | 12.06 ± 0.09 | 9.05 ± 0.19 | +3.0 | 7.9 → 5.0–5.9 |
| 32 | 12.57 ± 0.52 | 8.51 ± 0.23 | +4.1 | 7.8 → 4.1–5.6 |

- Extends [2026-07-20] POSRESET NECESSITY (N=8/64 spot cells) into a monotone dose-response with
  matched in-run joint anchors; per-copy decay under no-reset reproduces the A2 position-tax
  fingerprint at every N, while with-reset per-copy stays flat (~7–8 @N=32).
- The behavioral rows are the train-eval consistency requirement made visible: the winner was
  trained WITH reset; disabling it at eval alone destroys the readout (N=64 pf 0.923 = format
  collapse) — posreset is part of the deployed contract, not an optional nicety.

## [2026-07-29] 🔧 MIGRATION NOTE — plans/results_migration_DRAFT.md merged; its pending list resolved; N=128 full-34 addendum

- Provenance: the [2026-07-24] P3b/P4.3/P3a and [2026-07-25] P4 entries above are pasted verbatim
  from `plans/results_migration_DRAFT.md` (p0p2 campaign staging file). P1.1, P2a-LOTO, the TRUNC
  campaign, and the [2026-07-27] posreset wave were written during this migration directly from
  run-dir reports + campaign STATE files (`plans/p0p2_STATE.md`, `plans/trunc_STATE.md`) — those
  campaigns ended without logging. Every number above was re-read from the cited `report.txt` files.
- Draft "still pending" list, resolved: format-sweep B/C N=48/64 cells → landed in [2026-07-24]
  FORMAT SWEEP (jobs 125194–99) · L12 extra seeds → P1.1 above · LOTO → P2a above · MLVU leg →
  [2026-07-24] P2b · N=128 full-34 reports: **l12v2 landed** —
  `tallyL12v2_eval_N128/20260721_235843_L12_r8_evalonly/report.txt`: **acc 0.235 (8/34), parse-fail
  0.176, MAE 3.64** (g0–g2 6/6 · g4 1/2 · g32 1/2 · g48+ mostly parse-fail; consistent with, and
  superseding, the PARTIAL 0.286@n=14 reading in the l12v2 entry). **E-G full-34 (job 124924) never
  landed** — both `tallyPC8_eval_N128` run dirs are empty; the E-G refutation rests on the PARTIAL
  0.154 + the N=32/64 cells (verdict unchanged).
- The one-screen campaign summaries stay in `plans/` (to be archived intact, never deleted, in the
  docs restructure); run dirs + SLURM logs (being copied into run dirs) are the primary provenance.


## [2026-07-31] ✅📊 PRESENTATION INSTRUMENT BATTERY (GNN framing, 7 cells) — same sum-readout before/after: joint 0.468→0.077 @N=8→128 vs fenced/carrier 0.98–1.00 FLAT to 128; Jacobian: 52% of the joint read-out's sensitivity flows through OTHER frames, fenced cross-frame EXACTLY 0.00; q/kv swap 2×2: the joint-context tax splits ~50/50 (query/value) and the two repairs SYNERGIZE (+1.0 d′ interaction); anchor sweep: the room-token distillation target WINS (8.93 vs mean 7.00 / last 4.80 / char 4.39 / first 3.31)

> **Motivation:** peer-presentation instruments for the GNN narrative — render logged findings
> in the field-standard vocabulary (rewiring seen as attention maps; over-squashing as Jacobian
> sensitivity; the external readout as a DeepSets/GIN sum), plus two never-measured cells: the
> distillation-anchor choice (was a hand-picked heuristic) and the direct q-vs-kv decomposition
> of the joint-context tax (inferred, not measured, in [2026-07-14]).
> **Scripts:** `scripts/presentation_diagnostics/` (presentation_figs, probe_attention_map,
> probe_attnmap_deployed, probe_sensitivity, probe_anchor_sweep, probe_qkv_swap) + matching
> `slurm/*.sbatch`. All figure cells are CPU refits on frozen caches (`checkpoints/` symlinks +
> `outputs_legacy/`); probe cells are 1-GPU 2h_2g jobs, park seq8, 392px, steps task.
> **Runs (all `outputs/presentation/`, per-group INDEX.md):** figures jobs 127489–91 →
> `pca/20260731_202940` · `curves/20260731_202940` · `saturation/20260731_202507`; probes
> jobs 127555/127556/127560/127571/127582 → `attnmap/20260731_211330` ·
> `sensitivity/20260731_211330` · `anchor_sweep/20260731_213558` ·
> `attnmap_deployed/20260731_215003` · `qkv_swap/20260731_220735`. (Failed, no artifacts:
> 127553/4 + first attnmap_deployed pair + 127580 — mask-dtype and cell-indexing bugs.)

| cell | headline |
|---|---|
| curves — SAME gate→tally on 3 graphs vs N | joint 0.468/0.229/0.152/0.088/0.077 @N=8..128; fenced 0.998/1.000/0.984; carrier 0.976/1.000/0.988 @N=8/32/128 |
| pca + discriminant histograms @L16 N=8 | joint d′ 1.90 (clouds overlap) vs fenced 5.65 / carrier 5.06 (separated; 1-D discriminant view added because PCA understates carrier separation) |
| saturation (deployed stack, N=32, 10 layers) | gate err ~0.28–0.33 flat L2–L12 → 0.0087 @L20 / 0.0080 @L24; tally 0.08 → 0.82/0.85; write window = open+LoRA phase, reproduces [2026-07-25] |
| attnmap @L16 (probe layout) | joint = dense lower triangle; fenced = f_i↔c_i islands + Q column (probe mask hides replicas from fin by design) |
| attnmap deployed (e_c+LoRA, L\*=12) @L8 vs @L20 | the two-level hierarchy: islands below L\*, coarse carrier graph above (fin reads carriers) |
| sensitivity (Jacobian, L≤16, fit n=64 / grad n=12) | joint own-frame share 0.480 (uniform 0.125) — 52% cross-frame; fenced 1.000, cross EXACTLY 0.00 (structural) |
| anchor sweep (teacher config, 5 variants, n=150) | room d′ 8.93 / tally 0.997 ≫ mean 7.00/0.963 > last 4.80 > char 4.39 > first 3.31/0.432 |
| q/kv swap 2×2 (mask-only fence vs unmasked, shared positions, own-frame softmax keys, n=150) | qC_kvC 6.64 (0.981) · qC_kvD 4.66 (0.845) · qD_kvC 4.71 (0.893) · qD_kvD 3.74 (0.781) |

**Readings.**
1. **The curves figure is the controlled before/after**: readout held fixed (logistic gate + sum,
   the gate_tally protocol), only the graph changes. Joint messages cap it at any N; rewired
   graphs stay at ceiling to N=128. Fenced N=8 reproduces the 0.998 anchor exactly.
2. **Sensitivity is the GNN-literature over-squashing measure we lacked**: in joint attention,
   52% of ∂(read-out signal for frame f)/∂(embeddings) lies on OTHER frames (causal → earlier
   frames); the fence zeroes cross-frame Jacobians structurally (measured 0.00, not small).
   Caveat: the joint direction was fit at the final-question anchor of the Q-first replica
   layout (d′ 0.60 there) — weaker locus than the plain-joint anchor (~2.0); shares are
   row-normalized so the interference reading stands.
3. **The 2×2 turns the [2026-07-14] inference into a paired measurement**: from the joint corner,
   clean queries alone +0.97 d′, clean values alone +0.92 — the tax is HALF reader, HALF read —
   and the interaction is +1.01 (6.64−4.66−4.71+3.74): each repair alone recovers ~⅓, both
   together the full 2.90. The fence is the one intervention that does both at once. External
   consistency: dirty-dirty 3.74 ≈ the unmasked band 3.56; clean-clean 6.64 = mask-only band
   (posreset deliberately OFF for position-consistent q/k mixing).
4. **The room-token anchor is now an ablation winner, not a guess** (it was the one hand-picked
   knob never swept): room ≫ span-mean > last > char > first. The per-frame verdict binds at the
   question's content word; span-mean dilutes it, span-first barely carries it. Also reproduces
   the teacher band (~9 @n=150) as its top cell.
5. **Estimator discipline:** all d′ in these figures are held-out logistic-axis d′ (conservative);
   RESULTS headline d′ values elsewhere are the probe's d′_w — same ordering, different scale,
   do not cross-quote. Fresh joint fits also show a mild d′ decay with N (1.90→1.12 @8→128)
   where the older d′_w band read ≈flat — same caveat class.

**Caveats.** Steps task, park domain only; attn/sens/2×2/anchor cells are N=8 single-batch
(q/kv swap at N=16/32/64 launched as follow-up); carrier long-N cells in `curves` are the
proxy_room_k1 chain (only chain with long-N caches; distill N=8 cell in curves.csv: 0.976);
sensitivity grad phase n=12; no paired tests run on the 2×2 (same-sample by construction).

## [2026-07-31b] ✅📊 Q/KV SWAP vs N (16/32/64) — the VALUE-only repair collapses with graph size (+0.97 → +0.64 → +0.09 → −0.47 d′ @N=8/16/32/64) while QUERY-only stays ~+1 everywhere; interaction ~+2 at long N: at scale, the tax is only recoverable by repairing BOTH sides — which is what the fence does

> Follow-up to the [2026-07-31] battery's 2×2 cell. Same script/protocol
> (`scripts/presentation_diagnostics/probe_qkv_swap.py`: mask-only fence vs unmasked, shared
> base positions, own-frame softmax keys, L16), data `mmred_longN_park/seq_len_{16,32,64}`,
> n=150/100/50. **Runs:** jobs 127586/127587/127588 → N=32 `qkv_swap/20260731_221639` ·
> N=64 `qkv_swap/20260731_222127`; 127586's artifacts (N=16) were clobbered by a same-second
> run-dir collision with 127587 — deterministic rerun job 127589 →
> `qkv_swap/20260731_222936_127589` (numbers identical to 127586's log). Wrappers now suffix
> `_${SLURM_JOB_ID}` to the default OUTPUT (collision closed). Scaling figure + per-N CSV:
> `qkv_scaling/20260731_223504_local` (CPU, parses the four reports).

| N | qC_kvC | qC_kvD | qD_kvC | qD_kvD | q-only gain | kv-only gain | total | interaction |
|---|---|---|---|---|---|---|---|---|
| 8 | 6.64 (0.98) | 4.66 (0.85) | 4.71 (0.89) | 3.74 (0.78) | +0.92 | +0.97 | +2.90 | +1.01 |
| 16 | 7.84 (0.99) | 5.09 (0.91) | 4.72 (0.84) | 4.08 (0.85) | +1.01 | +0.64 | +3.76 | +2.11 |
| 32 | 7.39 (0.94) | 5.27 (0.84) | 4.12 (0.66) | 4.03 (0.72) | +1.24 | +0.09 | +3.36 | +2.03 |
| 64 | 6.49 (0.86) | 5.09 (0.67) | 3.57 (0.46) | 4.04 (0.55) | +1.05 | **−0.47** | +2.45 | +1.87 |

**Readings.** (1) The N=8 ~50/50 split was small-graph behavior: as N grows, cleaning the frame
values without cleaning the read-position query buys nothing (N=32) then hurts (N=64) — the
QUERY side is the binding constraint at scale, independently corroborating the [2026-07-13]
trained-query NO-GO ("the query half is architecturally irreducible") with a patching
instrument, and explaining why Q-first + fence (the two query-side repairs) is the winning
recipe. (2) Interaction stabilizes ~+2 at long N: most of the tax is recoverable only jointly.
(3) The joint corner's d′ stays flat (~3.7–4.1) across N while its tally decays 0.78→0.55 —
per-frame supply flat, task accuracy compounding — consistent with the flat-d′-vs-N verdict.
**Caveats:** n shrinks with N (150/100/50; N=64 noisiest, clean-corner tally ±0.15); single
seed batches; steps task, longN_park domain; no posreset anywhere by design.

## [2026-07-31c] ✅📊 WATERFALL GRID (N=8..128) + fenced-supply length cells — the A3 fence band HOLDS at length (replica d′ 10.67 @N=16 / 10.23 @N=64, joint anchors 3.91 / 2.64); the repair staircase is monotone N=8–64 with the addressing rung growing into the dominant step (+0.31 @N=8 → +0.46 @N=64); fence rung 0.96–1.00 at every N

> Completes the [2026-07-31b] decomposition into the presentation summary figure: one
> 5-rung staircase per length (joint single-locus readout → +reader per frame → +clean
> readers → +clean frames → full fence), same held-out gate→tally readout everywhere.
> **Runs:** A3 supply probes (fence+posreset+Q-first, `probe_supply`, L16) jobs 127597/127598
> → `outputs/presentation/fenced_supply/N16/…` (n=150) · `…/N64/…` (n=50); q/kv swap N=128
> job 127599 → `qkv_swap/…_127599` (n=25); grid figure + per-cell CSV →
> `waterfall_grid/20260731_234426_local` (auto-assembles curves + qkv + fenced cells;
> missing cells rendered n/a, never interpolated).

| N | joint | +reader/frame | +clean readers | +clean frames | full fence | carrier |
|---|---|---|---|---|---|---|
| 8 | 0.47 | 0.78 | 0.84 | 0.98 | 1.00 | 0.98 |
| 16 | 0.23 | 0.85 | 0.91 | 0.99 | 0.98 | — |
| 32 | 0.15 | 0.72 | 0.84 | 0.94 | 1.00 | 1.00 |
| 64 | 0.09 | 0.55 | 0.67 | 0.86 | 0.96 | — |
| 128 | 0.08 | (0.41) | (0.71) | (0.52) | 0.98 | 0.99 |

**Readings.** (1) The fence rung is flat 0.96–1.00 at every length while every partial
configuration decays — supply repair does not bend with N. (2) The per-frame-reader rung's
share of the total climb grows with N (+0.31 of 0.53 @N=8 → +0.46 of 0.87 @N=64): the
single-locus addressing/competition failure becomes the dominant component at scale,
consistent with [2026-07-31b]. (3) N=16/64 fenced cells are NEW A3 probe runs on longN_park
(d′ 10.67/10.23 — squarely in the fence band; also fresh joint anchors 3.91/2.64 for this
layout). **Caveats:** N=128 qkv rungs (parenthesized) are n=25 (~13 eval samples/seed) —
tally cells non-monotone (CD 0.71 > CC 0.52) from quantization noise though the d′ ordering
is sane (5.14 > 3.89 > 3.07 > 2.57); DO NOT cite the parenthesized cells without the n≈80
rerun; grid stitches two instruments (qkv probe protocol for rungs 2–4), stated on-figure.

## [2026-08-01] ❌📊 SUPER-CARRIER NO-GO — ONE frozen reader over the N coarse nodes fails at every N (tally 0.25 @N=8 → 0.07 @N=64, ferr 0.21–0.29, d′ 1.3–2.1), and restricting its softmax to carrier keys only changes NOTHING (coarse ≈ full at every cell): one query cannot address N items regardless of how few keys compete

> **Motivation:** Tal's two-level-hierarchy proposal — after the fence builds N carrier
> nodes, can a single aggregator token read them (competition over N instead of N·m)?
> **Design:** `scripts/presentation_diagnostics/probe_supercarrier.py` (job 127738, run
> `outputs/presentation/supercarrier/…_127738`): Q-first replica layout, deployed phase-1
> (full fence + posreset) below L*=12; above L* the fence lifts and the FINAL question's
> room token acts as the reader. Two arms differ only in the reader rows' keys above L*:
> `coarse` = {question + replica/carrier tokens} only; `full` = plain causal (control).
> Reader message decomposed per source carrier at L16/L20 → held-out gate→tally.
> Zero training. N=8/16/32/64 (n=120/120/80/40), park + longN_park.

| N | coarse reader | full reader | per-frame readers (rung-2 ref) |
|---|---|---|---|
| 8 | 0.25 | 0.25–0.28 | 0.78 |
| 16 | 0.13 | 0.10–0.14 | 0.85 |
| 32 | 0.09 | 0.09 | 0.72 |
| 64 | 0.07 | 0.07 | 0.55 |

**Readings.** (1) The single-reader failure is NOT about softmax key count: with only ~N
carrier keys available the reader does no better than over the full N·m context (cells
numerically identical at N=32/64) — the binding constraint is ONE QUERY addressing N
distinguishable items. (2) Completes the query-side triad: joint anchor (N·m keys, 1
reader) 0.09 · coarse super-reader (N keys, 1 reader) 0.07 · per-frame readers (N
readers) 0.55 — every working configuration in the method (replicas, carriers, serial
scratchpad steps) has ONE READER PER ITEM, in space or in time. (3) Kills the "one
aggregator token" shortcut zero-shot; consistent with the [2026-07-13] trained-query
NO-GO (0.36–0.48). **Caveats:** frozen zero-shot reader (final-question room token); a
TRAINED super-carrier embedding is untested (prognosis poor per [2026-07-13]); d′ at L20
lower than L16 in both arms (reader-locus property, not investigated).

## [2026-08-01b] ✅📊 POSRESET DOSE-RESPONSE, ACCURACY EDITION — 10 fresh matched A3 cells (only --reset-positions differs): reset flat 0.996–1.000 to N=32 / 0.950 @64 vs no-reset drifting 0.984 → 0.830; the supply-level position tax is MODEST and monotone (gap 0.013 @N=4 → 0.120 @N=64) — the deployed decode's collapse without reset (0.313/0.000 @N=32/64, [2026-07-27]) is train-eval inconsistency AMPLIFYING it, not raw supply loss

> **Motivation:** Tal wanted the position tax in task currency (gate→tally acc, not d′)
> across N. **Runs:** `slurm/posreset_sweep.sbatch` groups A/B (jobs 127736/127737) →
> `outputs/presentation/posreset_sweep/N{4,8,16,32,64}_{reset,noreset}/…` (probe_supply,
> Q-first + block fence in BOTH arms, L16, n=150/150/120/100/40, park + longN_park);
> figure + CSV `outputs/presentation/posreset_dose/20260801_164109_local/`
> (`plot_posreset_dose.py`).

| N | with reset | without | gap |
|---|---|---|---|
| 4 | 0.997 | 0.984 | 0.013 |
| 8 | 0.997 | 0.992 | 0.005 |
| 16 | 1.000 | 0.963 | 0.037 |
| 32 | 0.996 | 0.944 | 0.052 |
| 64 | 0.950 | 0.830 | 0.120 |

**Readings.** (1) Monotone widening, measurable from N=16 — matches the d′ dose-response
[2026-07-27] in a second currency with matched fresh cells. (2) The two-number story to
present TOGETHER: supply gap @N=32 is ~0.05 while the caption winner evaluated without
reset scores 0.313 — the trained readout's addressing assumes positionally identical
islands, so no-reset at eval is a distribution break, not just weaker supply; posreset is
part of the deployed contract. **Caveats:** N=64 cells n=40 (largest error bars, reset
0.950±0.077); single seed batch per cell; steps task only.

## [2026-08-09→11] ✅📊 NINV CAMPAIGN, PHASE 0 — the cross-N head-transfer leak is POSITIONAL and the node-posreset closes it: N=8→64 transfer 0.320 → 0.998 (park, same generator family) / 0.990 (MMReD-HF benchmark, all six N-pair cells pass the gap-based gate)

> **Motivation:** tree-register states must carry an N-invariant count code before any
> training. Baseline leak (cached features, runs 128773/128790): ridge head fit on N=8
> pair-node states transferred to N=16/32/64 at 0.643/0.445/0.320 (in-N 0.979).
> **Fix:** node posreset — every SQ node span gets identical canonical positions
> (`scripts/ninv/probe_tree_ninv.py`, copy-only). **Instruments:**
> `scripts/ninv/transfer_test.py`, `transfer_matrix.py` (+`metrics_skew.py`: every gate
> reports raw/majority/c2/balanced after the ridge-under-skew artifact below).
> **Runs:** park captures `outputs/ninv/20260809_231331_cap{8,64}`,
> `20260809_231854_cap{16,32}`, `20260809_232635_cap64_n200` (n=200 recapture),
> `20260809_232635_cap32_park2` (third-generator control); HF captures
> `outputs/ninv/20260809_234057_hf{8,16,64}` (train roots ×200, seq_len_64_test ×50).

| cell (park, longN family) | cross-N | in-N(test) | gap |
|---|---|---|---|
| 16→32 / 16→64 | 0.998 / 0.998 | 0.989 / 0.999 | −0.009 / +0.001 |
| 32→16 / 32→64 | 0.996 / 0.997 | 0.994 / 0.999 | −0.002 / +0.002 |
| 64→16 / 64→32 | 1.000 / 1.000 | 0.994 / 0.989 | −0.006 / −0.011 |

| cell (MMReD-HF) | raw | in-N(test) | balanced | c2 |
|---|---|---|---|---|
| 8→16 / 8→64 | 0.974 / 0.971 | 0.973 / 0.956 | 0.960 / 0.939 | 0.910 / 0.861 |
| 16→8 / **16→64** | 0.995 / **0.990** | 0.961 / 0.956 | 0.989 / 0.981 | 0.968 / 0.947 |
| 64→8 / 64→16 | 0.989 / 0.981 | 0.961 / 0.973 | 0.990 / 0.978 | 1.000 / 0.959 |

**Readings.** (1) Headline park cell N=16→64 = 0.9978, cluster-bootstrap CI
[0.9966, 0.9989] @n=200. (2) The literal original gate (N=8 images_park → N=64 longN)
reads 0.940 and FAILS 0.95 — cross-DOMAIN, not cross-length (equal 4× ratio:
within-domain 0.998 vs cross-domain 0.959; park2 fixed-N control: cross-root costs
~0.02 while the images_park N=8 head is erratic, 0.959→park / 0.597→park2).
(3) MMReD-HF adapter `scripts/ninv/load_hf_sample.py` self-check 320/320 across 13
pools. **Caveats:** HF N=64 pool is the full 50 test dirs; c2 columns here are ridge
(see the readout-artifact entry below for why balanced/c2 must accompany raw).

## [2026-08-10a] ⚠️📊 READOUT-ARTIFACT RETRACTION + THE MARGIN FINDING — the "HF merge anomaly" (c2 recall 0.82 vs park 0.98) was mostly MY INSTRUMENT: ridge+round shrinks toward the 0.8 zero-prior; a logistic readout on identical states recovers c2 0.822→0.950 with park untouched; what SURVIVES is a verdict-MARGIN gap (failures at min-child margin +3.98 vs +7.96) that 512px largely closes

> **Runs:** leaf captures `outputs/ninv/20260809_235142_hf8_leaf{392,512}` + park leaf
> `outputs/ninv/20260810_000358_park{8,16}_leaf`; instruments `leaf_probe.py`,
> `merge_anatomy.py`. Leaf evidence-recall 0.995 @392 / 1.000 @512 (perception ruled
> out; sqrt-law prediction of leaf ~0.90 refuted). OR-vs-SUM refuted ("exactly two"
> decodes ≥ "at least one" everywhere). Ridge→logistic on identical states: HF c2
> +0.058…+0.128, park +0.000/+0.001. Park margins +10.0/+10.4 with ZERO c2 failures
> (0/681, 0/727); HF failures are its low-margin tail; 392→512 cuts c2 failures 46→6
> of 258. **Caveat:** logistic is NOT uniformly better (buys c2, drops c1/balanced) —
> the binary decomposition is the trustworthy read; campaign instrument unchanged.

## [2026-08-10b] ✅📊 FROZEN TWO-PASS ON THE MMReD-HF BENCHMARK — EMIT-EM 0.960 @512px / 0.920 @392px on the untouched seq_len_8_test pool (park anchor 0.980; cond-EM 1.000 both arms: pass-2 never fails, ALL error is pass-1 quantization) + THE CAPACITY LAW OVER CODES: fan-8 merge 0.527 (raw analog) → 0.887 (quantized leaves)

> **Runs:** `outputs/ninv/20260810_151238_twopass_hf{392,512}` (calib = 200 train dirs
> strided, eval = the 50-dir benchmark test pool, zero training;
> `scripts/ninv/frozen_twopass_hf.py`, byte-faithful v4d port). Majority(predict-0)
> 0.620; EM-on-gold>0 0.842/0.947; MAE 0.12/0.04. Capacity-over-codes:
> `outputs/ninv/20260810_151952_hf8_quantcap` vs `..._hf8_rawL14` (same 200 dirs, codes
> written at L14 from a held-half-fit probe): fan-4 0.857→0.917, fan-8 (flat arm)
> 0.527→0.887 (+0.360), acc-on-count>0 0.433→0.750. Quantize-at-L14 does NOT fix the
> b2 margin tail (codes inherit the L14 probe's own errors: L14 evidence-recall 0.952
> vs L20 0.995 — quantize where the decision is ripe). **Caveats:** n=50 eval pool
> (±0.02/sample); pre-registered "fan-4 ≥0.95" not met (0.917).

## [2026-08-10→11a] ✅📊 THE FIVE-ARM TABLE — WHY the trained register tree fails or works: stability of the leaf interface fixes the MERGE (C), learned token-quantization completes the READOUT and eliminates per-level decay (D flat at 1.000), and the recipe fixes (N-mixture + answer-CE balance) buy ~2× training efficiency (V2: EM 0.920, the ≥0.90 gate PASSED)

> **Runs (all 512px, HF roots, identical seed/val, majority-emission baseline 0.510
> unless noted):** A/B `outputs/ninv/20260810_150229_p1_arm{A,B}` + evals
> `20260810_1658…/20260810_1702…`; C/D `20260810_201213_p1_arm{C,D}` + evals
> `20260810_230152_evalC`/`20260810_231226_evalD`; V2 `20260810_215334_p1_v2` + eval
> `20260811_013156_evalV2` (val majority 0.427). Trainer
> `scripts/ninv/train_registers.py` (row-masked LoRA `attach_masked_lora`;
> `--token-anchor`; `ROOT_PRESETS v2mix`; `--ans-balance`).

| arm | mechanism | lv1/lv2/lv3/root probes (in-length) | emitted EM | N=64 transfer (lv1…root) | lv5/6 @N=64 |
|---|---|---|---|---|---|
| A | none | 1.000/0.890/dead/dead | 0.510 (=maj) | 1.000/0.921/dead/dead | dead |
| B | injected leaf codes | 0.989/0.981/0.968/0.926 | 0.730 | 0.976/0.960/0.932/0.880 | dead |
| C | frozen leaves (row-masked LoRA) | 0.998/0.994/0.961/0.778 | 0.510 (=maj) | 0.999/0.995/0.970/0.700 | dead |
| D | learned token-anchor | 1.000/1.000/1.000/1.000 | 0.870 | 1.000/1.000/0.998/0.975 | dead |
| **V2** | D + N-mix + ans-balance | **1.000 flat** | **0.920** | 0.999/0.999/0.998/0.990 | dead |

**Readings.** (1) LENGTH-invariance of trained components: proven (every trained level
transfers to 4× length ≥0.97 in D/V2). (2) DEPTH does not extrapolate — never-trained
levels are dead under every mechanism incl. shared alphabet + 4-depth role mixing; the
emitted number at N=64 stays at majority for all arms (0.03–0.08). (3) Per-level
re-quantization eliminates error compounding (B decays ×0.99/×0.99/×0.96 per level;
D/V2 flat). (4) The A/C EM sits at the majority-emission attractor — EM must ALWAYS be
read against the val gold-0 fraction (a smoke-scale "EM 0.562" equalled its val
baseline exactly). **Caveats:** smoke-scale readings (≤18 optimizer steps) predicted
BOTH C and D wrongly — full-scale only; C's readout never flipped within 15 epochs
(root crossed the CE~1.8 breakout threshold one epoch before budget end).

## [2026-08-11a] ✅ CANARY, CORRECTED — no leak: mask CPU-verified, positions symmetric in ALL THREE M-RoPE channels, T0 identity replay bit-exact (0.000e+00), 0/10 answer flips under content-preserving swaps; the pre-registered BIT-IDENTITY canary demanded an invariant unachievable in bf16 (content permutation reorders reductions; ~1e-3 amplifies over 28 layers to O(0.5) logits) and is retired

> **Runs:** `outputs/ninv/20260810_174323_possym` + `20260810_173729_canaryB_T0`.
> Corrected canary of record: structural checks + T0 replay + answer stability.

## [2026-08-11b] ✅📊 P1.5 + THE INTERFACE LAW (LAW 7) — synthetic digit-leaf training brings levels 5–6 ALIVE on synthetic trees (lv5 CE 11.2→1.36) at zero cost in-length (EM 0.925, probes 1.000 flat, N=64 lv1–4 transfer 1.000/1.000/0.998/1.000 — best of campaign) BUT they stay DEAD on real trees (emitted 0.054 ≈ majority): states that DECODE to the same digit are NOT interchangeable inputs — soft token-alignment ≠ substitutability

> **Runs:** trainer `outputs/ninv/20260811_105241_p15` (`--synthetic-n 400
> --init-ckpt` v2; image-free samples, literal '0'/'1' single-token leaves,
> bit-order verified per sample), eval `20260811_132203_evalP15`. Both regimes' lv4
> children emit their digits (node_tok CE ~0.03) yet lv5 accepts only the synthetic
> ones — the merge reads more of the child state than its token projection; the other
> ~3,583 dimensions carry regime-specific residue that SGD exploited. **Caveat:**
> lv5/6 probes reuse the lv4 head (range-clamped); the emitted collapse is
> head-independent.

## [2026-08-11c] ✅📊 THE FLAT-IN-N EMITTED CURVE — hard-requant cascade (one fenced forward → probe-read the depth-4 subtree counts → validated text adder, all stages at measured ≥0.99): EM(GT≤16) 0.925 (N≤16) / 0.875 (N=32) / 0.811 (N=64) vs majority 0.42/0.16/0.04, every trained component blind to N>16 — LAW 7 CONFIRMED ON THE TRAINED SYSTEM (same ckpt: 0.054 in-embedding vs 0.811 as decoded numbers at N=64)

> **Runs:** `outputs/ninv/20260811_145257_cascade_p15c` (`scripts/ninv/eval_cascade.py`,
> P1.5 ckpt, benchmark test pools n=50/N). lv4 subtree fidelity 0.990/0.995 with the
> probe head HELDOUT-validated at 1.000 (160/40); pairwise-composed v4d adder.
> **Instrument history logged in STATE:** run A died on a stale row-mask; run B's head
> was silently clipped to 0..2 by a reused pair-count helper (also invalidates the
> capped-eval probe numbers, `20260811_112520_capped_v2c`) and under-fit at n=120
> (heldout resolves 160 suffice); the zero-training capped/flat-top route measured
> DIRECT 0.42 @N=32 but dead @N=64 — bounded, superseded by the cascade.
> **Caveats:** the interface head is a fitted 3584→17 linear component (same class as
> the frozen-twopass quantizers; all-model-emitted variant needs node_tok hardening at
> long context — open); GT≤16 primary per pre-registration; 6/50 and 9/50 samples lost
> to adder-chain decode -1s (each an automatic miss) — harness, not model; N=128
> untested.

## [2026-08-22] ✅📊 N=128 Q/KV SWAP RERUN (n=80) — the [2026-07-31c] parenthesized cells are now citable and MONOTONE: tally CC 0.870 > CD 0.650 > DD 0.400 > DC 0.345; the [2026-07-31b] scaling law extends to N=128 (q-only +1.32, kv-only −0.78, interaction +2.23)

> ARMOR campaign hygiene cell (brief/log: `outputs/armor/`). Job 136124 →
> `outputs/armor/qkv128/20260822_210658/` (same script/protocol as 127599: L16, steps,
> `mmred_longN_park/seq_len_128/all_uniform`, LIMIT=80, skip 0).

| N=128 (n=80) | qC_kvC | qC_kvD | qD_kvC | qD_kvD |
|---|---|---|---|---|
| d′ | 6.86±0.32 | 5.41±0.09 | 3.31±0.26 | 4.09±0.02 |
| gate→tally | 0.870±0.037 | 0.650±0.055 | 0.345±0.064 | 0.400±0.042 |

**Readings.** (1) The n=25 tally anomaly (CD 0.708 > CC 0.523) was quantization noise as
flagged — at n=80 the staircase is monotone and CC is cleanly on top; the waterfall's
N=128 rungs 2–4 now read 0.400/0.650/0.870. (2) Full gain series across N (8/16/32/64/
128): q-only +0.92/+1.01/+1.24/+1.05/+1.32 (flat ~+1); kv-only +0.97/+0.64/+0.09/−0.47/
−0.78 (monotone decline — value-repair-alone is actively harmful at scale); interaction
+1.01/+2.11/+2.03/+1.87/+2.23 (~+2 at long N). The query side stays the binding
constraint at the largest N; both sides must be repaired jointly — the fence's job.

## [2026-08-22] ✅📊 HAHN O(1/N) MEASURED INSIDE THE PRODUCTION VLM — paired one-frame flips: joint answer-position sensitivity decays as N^−0.72 [0.63,0.78] (r²=0.98, band [0.7,1.3] MET at L20) and is AT the measured bf16 noise floor by N=128, while the fenced per-frame supply is exactly flat (α = 0.00 ± 0.02, ~30× above floor) — the theory-to-model contrast figure

> ARMOR Experiment A. New instrument `scripts/armor/probe_hahn.py`; jobs 136128–136132 →
> `outputs/armor/hahn/20260822_211457_N{8,16,32,64,128}/` + figure/verdict
> `…_fig/hahn_figure.png` (50/50/50/50/40 pairs, gold≤8 slice, ~1h50m GPU total). Pairs
> rendered fresh from states via the deterministic park renderer, ONE non-evidence frame
> toggled to evidence (gold k→k+1); byte-identity of all other frames hard-asserted per
> pair. Noise floors MEASURED per the [2026-08-11a] canary lesson: replay (exactly 0
> everywhere), matched answer-preserving ctrl flip (same frame, wrong-room→wrong-room),
> fenced block-permutation (bf16 reduction floor — and the fenced per-frame locus is
> BIT-IDENTICAL under it: the fence provably isolates the per-frame channel).

| locus | α (Δ ∝ N^−α) | 95% CI | r² | band |
|---|---|---|---|---|
| joint (plain) L20 final | **0.72** | [0.63, 0.78] | 0.98 | [0.7,1.3] **MET** |
| joint (plain) L28 final | 0.58 | [0.49, 0.68] | 0.95 | below (post-norm compression) |
| per-frame readers over JOINT supply (repjoint rep_t L20) | 0.25 | [0.16, 0.30] | 0.97 | — |
| **fenced rep_t L16/L20/L28** | **0.01 / −0.01 / 0.02** | all CIs ∋ 0 | — | \|α\|<0.2 **MET** |

**Readings.** (1) First direct measurement of the Hahn-class single-symbol sensitivity
decay in the frozen production VLM: joint L20 median ‖Δh‖ 10.5 → 1.3 (N=8→128); at
N=128 the answer-relevant flip signal (1.33) equals the answer-preserving ctrl flip
(1.1) and the measured bf16 floor (~1.2–1.4) — flipping the answer-defining frame moves
the answer position by no more than numeric noise. (2) The fenced arm is the money
contrast: per-frame supply flat in N at every layer. (3) The intermediate rungs
interpolate (repjoint final 0.44, repjoint rep_t 0.25): per-frame readers alone recover
only part of the invariance; clean supply (the fence) zeroes the slope — consistent
with the q/kv interaction story. (4) A2 margin: the pre-registered crossing band FAILED
AS DEFINED but by instrument saturation, logged honestly: the frozen model is already
below 50% at N=8 on this slice (acc 0.28), so the median margin starts negative (−0.53)
and "crosses" at the smallest N measured; the informative read is the monotone margin
slide (−0.23 @16 → −1.91 @128, ctrl-jitter floor 0.02–0.24) with accuracy reaching
chance at N≈32–64 (0.18/0.12/0.10). **Caveats:** gold≤8 slice (single-digit protocol,
N-comparable by construction); L28 is post-final-norm; margin conflates multi-digit
continuations for non-gold mass.

## [2026-08-22→23] ⚠️📊 STATE-PASSING CONTROL (Buitrago & Gu, arXiv:2507.02782) — NOT RESCUED per the pre-registered band (best arm 0.800 @N=64 zero-shot < 0.90), and the intervention DECOMPOSES the drift: SP fixes the state-coverage leg (in-range 0.35→0.94/0.96 at 4×) while the readout-range cap survives exactly intact (GRU gold>16 recall 0.000 — a cliff at the label support); at 8× even the rescued leg erodes (best arm 0.448 vs R1 0.980)

> ARMOR Experiment B (closes RELATED_WORK Top Threat #7). `scripts/armor/train_heads_sp.py`
> (recagg P2 protocol imported unchanged — fit @{8,16}, zero-shot @{32,64,128}, 20k
> epochs, 5 seeds, canary; recagg originals untouched). Jobs 136139/136140 (main cells)
> + 136186/136187 (+N=128 eval) → `outputs/armor/b_statepass/20260822_212000_{hf512,park}/`,
> `20260823_000332_hf512_128/`, `20260823_000521_park_128/`; N=128 leaf captures
> `outputs/armor/p1_captures_128/20260822_233338_*/` (p1 protocol, npz sanity passed).
> SP = detached final state of another training sequence + count-consistent carried
> target, REJECTED if total >16 — label range held fixed BY DESIGN so state coverage and
> label range are not confounded. Noise = fitted Gaussian on running final-state stats
> (paper variant), pre-registered as ill-posed for the pure integrator.

EM_reg @zero-shot (HF@512; majority 0.16/0.08/0.12; R1 sum-probe 0.996/1.000/0.980):

| arm | N32_zs | N64_zs | N128_zs | in≤16 @64 | out>16 @64 | mc@rec≥.5 |
|---|---|---|---|---|---|---|
| R2 GRU base | 0.640 | 0.260 | 0.044 | 0.351 | 0.000 | 10 |
| R2 GRU **+SP** | 0.904 | 0.696 | 0.240 | **0.941** | **0.000** | 15 |
| R3 SSM base | 0.820 | 0.284 | 0.108 | 0.346 | 0.108 | 18 |
| R3 SSM **+SP** | 0.936 | **0.800** | 0.448 | **0.962** | 0.338 | 21 |
| R3 SSM +noise | 0.164 | 0.080 | — | (in-length 0.728!) | — | −1 |

**Readings.** (1) The Buitrago objection is answered by RUNNING their intervention: it
works exactly on the leg its theory owns — the "unexplored states" drift is real, and SP
restores within-range accuracy to ~0.95 at 4× the training length (park twin replicates:
bases 0.40/0.29 → SP 0.68/0.68 @64). (2) What it cannot touch is the Yehudai range cap:
the GRU emits NOTHING above count 16 in every arm (recall exactly 0.000, max-correct
pinned at the fit-range max), and the out-of-range mass grows with N (26% @64, 60% @128)
→ band NOT met. (3) The rescue itself is length-bounded: in-range accuracy falls back to
0.60/0.77 at 8× (the concat-carry visits ~2–3-sequence virtual horizons, not 8×) — the
best intervention arm drifts 0.936/0.800/0.448 @2×/4×/8× while R1 probe→sum reads
0.996/1.000/0.980 (park: 0.994 @8× with perfect recall to gold=128). The drift law is
intervention-robust; the necessity argument rests on the range leg, which is
intervention-proof by construction. (4) The noise variant is catastrophic for the pure
integrator even in-length (0.984→0.728 @N=8) — the pre-registered identifiability
prediction (a borrowed-looking initial state offsets the integral unrecoverably).
**Caveats:** for RNN heads SP-by-concatenation ≈ longer virtual training sequences with
detached gradients (that IS the published intervention; labels capped, so coverage vs
range is controlled); cls-EM not the verdict metric (support structurally capped at 16);
N≤64 numbers quoted from the main cells — the +128 reruns differ by 0.01–0.03 there
(cross-node CPU float accumulation over 20k epochs, seed-level noise).

## [2026-08-22→23] ⚠️📊 MLVU-AC EXTERNAL ANCHOR — composed frozen system (per-frame VLM records → compiled program → exact executor): 32f pre-registered cell 0.301 vs band ≥0.332 = NO-GO, with the miss LOCALIZED (executor 97% faithful, frame-recall 0.928, and the MEASURED 32f faithful-counter ceiling is 0.267 — below the frozen 0.282); at 128f dense the composed system reads 0.408 vs the 0.393 dense baseline — the best MLVU-AC number on record, residual = perception-side overcounting

> ARMOR Experiment C. Captions 136145/136149 → `outputs/armor/mlvu/20260822_213354_captions/`
> (32f) + `20260822_220806_captions128/` (128f dense); compile+execute 136146/136169 →
> `20260822_220317_compiled/` (CANONICAL 32f) + `20260822_233323_compiled128/` (dense).
> Frame selection byte-identical to the [2026-07-11c] frozen protocol and the
> lookagain_N32 judge keys (verified); qwen14b armC-v3 sandbox (read-only import), P2b
> nearest-option MCQ rule; exec-fail 0.015 both cells; 0/2 caption parse-fails (32f/128f).

- **32f (pre-registered):** MCQ 0.301 (frozen 0.282, band ≥0.332, chance 0.25); open
  exact 0.197, MAE 1.62; mean pred 1.35 vs gold 2.93; by-gold g1 0.70 → g5 0.00.
  **Localization chain:** VLM per-frame recall on judge-positive frames 0.928 (precision
  0.727) → 97.0% of predictions equal the run-count of the system's own delivered
  evidence (compile+execute essentially exact) → but 32f sampling delivers 282 visible
  frames against 603 gold instances (48/206 questions receive ZERO evidence), and an
  ORACLE judge-evidence run-counter scores 0.267 — **below the frozen baseline**. At 32f
  the +0.05 band was unreachable by construction for any faithful counter; the frozen
  model's edge over the faithful ceiling is option-prior guessing, not counting.
- **128f dense (post-hoc extension, labeled):** composed **0.408** vs dense frozen
  baseline 0.393 — slightly ahead at matched budget, best MLVU-AC number on record, but
  short of a +0.05-style margin there too (would need 0.443). Delivery largely cured
  (zero-evidence questions 48→10; 1198 visible frames) and the residual flips to
  OVERcounting from per-frame false positives (mean pred 4.76 vs gold 2.93; precision
  0.727 fragments instances into spurious runs); by-gold flattens to 0.31–0.49 (the
  delivery gradient is gone). Program note: at 128f the compiled programs diversify
  (~56% plain run-count semantics, index/gap variants otherwise).
- The external-validity statement: on real video the composed system's EM tracks
  perception fidelity in both directions (undercount when starved, overcount from FPs) —
  no measurable aggregation-side deficit at either budget, mirroring the Arm B
  factorization on MMReD-HF. Levers not run: per-frame self-consistency, FP-suppressing
  prompt, judge-style thresholding.
- **Caveats:** single question type (compile near-degenerate here — this measures the
  COMPOSED system, not compile generality; armC measured that across 24 types); the
  baseline had the MCQ options in-prompt, the composed route never sees them (P2b
  asymmetry, pre-registered); judge scores are themselves model-derived (lookagain),
  used as a proxy.


<!-- ══════════════ TREEFOLD campaign (2026-08-24→25) — logged on Tal's 'log this' 2026-08-26; drafts verbatim from outputs/treefold/STATE.md ══════════════ -->

## [2026-08-24→25] ❌📊 TREEFOLD — the frozen 7B as its own tree-fold executor FAILS the executor swap on every rung: oracle-leaf fan-2 tree ALL ~0.20 flat-below-majority vs armC v3's 0.89 on identical samples (exact execution is worth ~0.68 EM), and the per-level fidelity instrument localizes it: p_merge 0.20–0.70 per op, decaying with TREE DEPTH ~independent of N

> Campaign `outputs/treefold/` (brief + STATE + INDEX). One frozen Qwen2.5-VL-7B nf4 for
> every stage (Ask/leaf/merge/answer); few-shot Ask primary (Tal's checkpoint decision,
> armC-matched convention); samples/scoring byte-identical to armC v3 / Arm B.
> T1 `t1_oracle/20260824_fs/` · T4 `t4_fullstate/20260824_fs/` · figures `figs_20260825/`.

| ALL EM (N=16/32/64/128) | 16 | 32 | 64 | 128 |
|---|---|---|---|---|
| T1 tree, oracle records, 7B executor | 0.23 | 0.20 | 0.18 | 0.21 |
| armC v3: same records, Python executor | 0.89 | 0.92 | 0.85 | 0.90 |
| T4 tree, Arm B caption records | 0.25 | 0.24 | 0.18 | 0.23 |
| Arm B: same captions, Python executor | 0.79 | 0.81 | 0.72 | 0.72 |
| majority | 0.38 | 0.36 | 0.36 | 0.39 |

**Readings.** (1) H1/H4 decisively not met: swapping exact execution for model
execution costs ~0.5–0.68 EM on identical inputs. (2) The failure is LOCALIZED at the
interfaces, not the fold: conditional merge fidelity (both children correct) is 0.995
(2005/2016, ≥0.98 at every level) — the 7B executes left+right essentially perfectly,
and the per-level fidelity decay (F-C) is pure error propagation. What breaks is
(a) MAP — per-frame predicate evaluation on clean GT text fails 10–40%/frame (p_leaf
0.56–0.91; e.g. record "Office: Daniel; Bedroom: Sandra" + rule "count=1 if Daniel is
in the Bedroom" → count=1), and p_leaf^N alone (0.8^16 ≈ 0.03) explains the collapse;
(b) ANSWER read-off — 5 of the 8 samples with an exactly-correct root note still
answered wrong (root {"count": 8} → "Daniel"). (3) The composition check closes:
p_leaf^N·(cumulative p_merge)^(N−1) ≈ 0 ≈ measured EM on instrumented types.
(4) Null-propagation programs survive (single-frame lookups 0.75–1.00 flat to N=128) —
they are single-interface: one predicate evaluation, no accumulation. (5) T1 EM is
~length-flat — but flat below majority: flatness without competence. **Caveats:**
few-shot Ask (4 SEEN exemplars, armC parity); 8/cell per type; conditional-fidelity
n=2016 pooled over count-like types with an identified field; the pre-registered 14B
control (now aimed at leaf+answer stages, not merge) is proposed, not run.

## [2026-08-25] ❌📊 TREEFOLD fan-in law: EM FLAT in k∈{2,4,8,16,N} at N=128 (0.16–0.21, all at the executor-noise floor) — no over-squashing signature AND no noisy-op inversion; per-level p_merge decays with composition depth ~N-invariantly (L1 0.59–0.69 → L4+ ≤0.29 at every N) — the wall is per-op semantic fidelity, which fan-in cannot route around

> T2 `t2_fan/20260824_fs_k{4,8,16,N}/` (k=2 = T1; leaves reused, k-independent);
> headline figure F-A + mechanism figure F-C in `figs_20260825/`. k=N ran at BATCH=4
> after an OOM at 48 (prompts carry 64–128 notes).

- k=N does ONE merge call and suffers ~zero parse deaths (1–2 vs 22–32 at k≤4) yet
  lands at the same EM — depth mortality and breadth squashing exactly net out.
- Pre-registered fallback stands: the fan-in/over-squashing link is NOT supported at
  this scale; the thesis' mechanism evidence remains the in-forward measurements
  (Hahn α=0.72 joint vs 0.00 fenced; capacity law c(fan)).
- H5 addendum: zero-shot Ask collapses harder with depth (0.073 vs few-shot 0.208
  @N=128; merge-parse 70/192 vs 25/192) — program quality depends on demonstrations.

## [2026-08-25] ⚠️📊 TREEFOLD T3 — question-conditioned per-frame JSON notes cost perception: count-like leaf accuracy 0.921 @512 (vs 0.995–1.000 for the 1-bit conditioned verdict, ≈0.913 for Arm B's full-state caption) — structured note EMISSION, not seeing, is the lossy step; end-to-end tree 0.31→0.16 with merge-parse mortality growing with depth (1/9/22/36 per 96)

> Captions `t3_leaves/20260824_fs/` (382 samples, 23,008 frames @512, parse 0.990);
> tree `t3_tree/20260824_fs/`. H3 not met on both rungs (0.921 < 0.98; ALL ≪ ArmB−0.03).
> The 0.995→0.92 drop vs the verdict-bit record isolates note-emission cost — relevant
> to any design that asks the VLM for structured per-frame state instead of a bit.

## [2026-08-25] ✅📊 TREEFOLD EXTENSION — the failure LOCALIZES to the two interfaces and the composition law becomes PREDICTIVE: merge is sound at 7B (conditional 0.995); leaf binding = margin×clutter (7B 0.500→0.984 by prompt-stripping; 14B 1.000 regardless); with repaired interfaces the tree is real at short N (steps_in_room 1.00 @N=16, 7B-ONLY) and decays as p^N — ctrl-B's pre-registered ceiling 0.995^(N−1) predicted 0.93/0.53 @16/128, measured 1.00/0.50

> Stage-swap grid `ctrl14b/` (14B = Qwen2.5-14B venv_arch at LEAF/ANSWER only; merge
> always 7B) + prompt ablation `leaf_ablate/` (V0 canon 0.500 / V1 minimal 0.984 /
> 14B both 1.000; FP|room-occupied 0.86→0.02) + repaired arm `t1_minimal/20260825_fs/`
> (v1 read-off bug — a stray Nobody clause turned correct numeric roots into "Nobody"
> — caught by the fidelity instrument, artifacts preserved, v2 rescored).

| steps_in_room EM | N=16 | 32 | 64 | 128 |
|---|---|---|---|---|
| canonical 7B tree | 0.00 | 0.00 | 0.00 | 0.00 |
| minimal prompts, 7B-only | 1.00 | 0.75 | 0.38 | 0.50 |
| 14B interfaces (ctrl-B) | 1.00 | 1.00 | 0.88 | 0.50 |
| predicted ceiling 0.995^(N−1) | 0.93 | 0.86 | 0.73 | 0.53 |

**Readings.** (1) The 7B *can* divide-and-merge — at short N, with clutter-free
interfaces; the negative headline becomes a boundary condition, not an impossibility.
(2) No fixed per-op fidelity survives long N (both repaired arms → 0.50 @128, on the
predicted merge-noise ceiling); scale only moves the crossover. (3) Answer-stage
fragility is real and instrument-detectable (Nobody-clause incident: correct roots,
wrong answers). (4) ALL EM stays ≈majority in every arm — residual is Ask quality on
the 20 UNSEEN types, not interface fidelity. **Caveats:** 2×2 ablation is
steps_in_room-only (384 frames); ctrl grids share the 7B merge; first_at_room
anomaly at 14B leaves (null-propagation schema filled with present-characters) noted.


## [2026-08-26→27] ✅📊 LORAMECH P0+C1 — THE REGIME IS THE BASELINE: the peer's "plain LoRA reads N=8 at 1.000" is Q-first-specific — the SAME 23.8M recipe trained FRAMES-FIRST cannot fit even its training lengths (0.600 @8 / 0.480 @16 / 0.313 @32 vs Q-first P4.1 0.967 @32), and a within-pipeline Q-first control isolates the TEMPLATE as the causal factor (+0.33 @8, loss 0.237→0.070)

> LORAMECH campaign (brief `outputs/loramech/CAMPAIGN_BRIEF.md`, log `outputs/loramech/STATE.md`).
> Trainers: ff_le8 137467 → `outputs/loramech/p0_ff_le8/20260826_161912_lora/`; ff_le32
> 137465 (h200, 3h08) → `outputs/loramech/p0_ff_le32/20260826_171532_lora/`; C1 Q-first
> control 137706 → `outputs/loramech/c1_qfirst_le8/`; zero-shot exams 137504/137593.
> Adapters: `checkpoints/sft_ff_le{8,32}_adapter/`. Trainer delta: `--frames-first`
> (build_count_prompt layout, token-parity-gated vs build_prompt_inputs by
> `scripts/loramech/check_ff_template.py`) + `--exclude-dirs-file`. Contamination BY
> CONSTRUCTION: exam sets (`outputs/loramech/examdirs/`, 150/cell, class-balanced,
> stratified seed-1) excluded from training + post-hoc overlap 0; N=32 exam = disjoint
> park2 generation; recipe parity with the E-B/P4.1 anchors verified in-log (23,794,688
> trainable params = 28 LM layers × q/k/v/o+MLP r=8 + 32 vision-block MLPs via name
> match; lr/accum/dropout/epoch budget identical to what the anchors actually ran).

| exam (150/cell, pf 0) | ff_le32 (≤32) | ff_le8 (≤8) | Q-first |
|---|---|---|---|
| N=8 | 0.600 | 0.540 | C1 **0.867** / E-B 0.998–1.000 |
| N=16 | 0.480 | 0.333 | — |
| N=32 | **0.313** | 0.180 | P4.1 **0.967** |
| N=64 | 0.293 | 0.173 | P4.1 0.787 |
| N=128 | 0.160 | 0.167 | — |

- **H0 (pre-registered, ≥0.90 in-length) REFUTED below its own 0.80 contingency line** →
  without question-conditioned encoding, dilution binds already inside the window.
  Causality: frames-first forbids frame tokens from precomputing question-conditioned
  verdicts; the read at the answer position over N question-blind frames is the clean
  Hahn setting. The Q-first 1.000 was the carrier route in disguise.
- Anatomy separates the arms: ff_le8 at its own length is FLAT per-count (mid 0.548 ≈
  extremes 0.530 — dilution, not shortcut); far out-of-window both plain arms relapse
  to the extremes heuristic (N=128 ff_le32: g0 9/9 + g128 8/8, mid 3/116 = 0.026).
  ff_le32 is extremes-anchored even in-length (N=32 mid 0.192, trained-max g32 11/11).
- C1 (identical pipeline/split/recipe, template flipped): loss converges anchor-like
  (0.070) and in-length jumps to 0.867 — the regime isolated as THE learnability
  factor within one pipeline. Residual 0.13 vs the legacy anchor ≈ sample count
  (525 vs 630) + val-noise; C2 (15-ep bound) abandoned after 3× h200 preemption
  (~19 GPU-h), closed by budget-parity + C1.
- §6 consequence: the canonical SFT baseline row is now FRAMES-FIRST (these numbers);
  the Q-first 0.967/0.787 stays in the record regime-labelled.


## [2026-08-27→28] ✅📊 LORAMECH P1/P1b/P2/N4 — THE ARCHITECTURAL FENCE RESTORES (AND BEATS) Q-FIRST LEARNABILITY, EXTRAPOLATES 2× ZERO-SHOT, AND TRANSFERS TO THE FAITHFUL BENCHMARK AT 1.000: plain LoRA trained under N×[frame+q] blocks + posreset (NO Q-first) reads **1.000 @8 / 0.893 @16 in-length and 0.867 @32 zero-shot** — the first arm on record past the zero-shot length wall — and the park-trained adapter scores **1.000 on the MMReD-HF seq8 test** with zero HF training

> New trainer `scripts/loramech/train_sft_fenced.py` (fencing imported READ-ONLY from
> gnnformer.fencing: build_block_mask(hide_cols=[]) — replicas visible to the tail —
> + reset_positions + FenceHooks; exact cache-free greedy decode; 4-D mask ⇒ EFFICIENT/
> MATH sdpa, N≤16 trains on 48 GB — no H200 anywhere). Runs: P1 137744 (5 ep) →
> `outputs/loramech/p1_fenced_le16/20260827_131154_fenced/`; P1b 137778 (10 ep, CANONICAL)
> → `outputs/loramech/p1b_fenced_ep10/20260827_192346_fenced/` + zs 137862; P2 (HF train
> splits, 320 samples) 137799 → `outputs/loramech/p2_fenced_hf/20260827_193851_fenced/`;
> N4 cross-domain 137898 → `outputs/loramech/n4_park_on_hf/`. Adapters:
> `checkpoints/sft_fenced_le16_ep10_adapter/` (canonical), `sft_fenced_le16_adapter/`,
> `sft_fenced_hf_adapter/`. Same exam files/discipline as P0 (disjoint ×3 each, pf 0).

| N=8-in-length spectrum (one pipeline) | acc |
|---|---|
| frozen frames-first | 0.219 |
| plain ff-LoRA (P0) | 0.540 |
| Q-first LoRA (C1) | 0.867 |
| **fenced ff-LoRA (P1b)** | **1.000** |
| caption method (ref) | 0.987 |

- **Fenced ladder (P1b, train ≤16):** 1.000 @8 · 0.893 @16 · **0.867 @32 (2×, zero-shot)**
  · 0.340 @64 · 0.230 @128 — LIVE per-count curves in-window (P1 5-ep: mid-range
  0.988/0.760 @8/16; no extremes crutch), and at 2× the fenced zero-shot BEATS the
  plain arm trained AT 32 (0.867 vs 0.313; mae 0.16 vs 2.11). Explicit structure beats
  the layout hack at equal budget (0.993–1.000 vs 0.867): isolated, position-normalized
  verdict blocks > interfering Q-first stream.
- **The 4×/8× wall is LAWFUL, not chaotic:** low band fails by growing systematic
  undercount (unit slope with ≈−1.5 offset @64; slope 0.43 @128; all 23 N=16-residual
  errors in the 5-ep run were exactly +1) while fraction-1 ("all frames") and trained
  label-support anchors (g12/g16) stay exact — an analog magnitude code going out of
  calibration, with ~one octave of slack. (The +1 wobble later vanished under
  calibrated training — see the L5/P3 entry.)
- **Benchmark transfer (N4):** the park-trained fenced adapter on the HF test, zero HF
  training: **1.000 @8 (50/50) · 0.820 @16 · 0.540 @32** — beats the HF-trained P2
  (0.900/0.660/0.300; its 320-sample skewed split was the limiter — majority floors
  0.62/0.38, frozen anchor 0.559 is BELOW floor) and lands over the armB structured-GIN
  ceiling (0.928) at seq8. What transfers is the aggregation program, not the pixels;
  balanced generator data + structure > in-domain skewed data.
- Caveats: HF cells are 50/len (benchmark's own splits); N-cells 100–150; single task
  family + model; val-by-decode is noisy (best-epoch selection rides 60 samples).


## [2026-08-27→28] ✅📊 LORAMECH MECHANISM (L1/N2/N3/L5/P3) — GAIN vs STRUCTURE, QUANTIFIED: trained LoRAs keep the frozen sensitivity law (in-window α 0.71/0.73 ≈ frozen 0.72) and only multiply amplitude ×3–6; through the trained fenced layout the per-frame verdict channel is EXACTLY flat (α +0.009 [0.001,0.016]) while the answer read decays (α +0.80 [0.66,0.97]) — supply fixable, gain trainable, the softmax read's 1/N is neither; log-N logit scaling rescues ONLY the fenced read (+0.13 @4×) and FAILS as a training prior (in-window 1.000/1.000 but 2× collapses 0.867→0.420) — token-coded enumeration stands as the unique measured route past the read's horizon

> Instruments: `probe_hahn.py` + `--peft-adapter`/`--arms`/p1fence arm (trained-fenced
> layout imported from the P1 trainer — single source; no-flag behavior = the ARMOR-A
> anchors); `--attn-logn-sref` on both trainers (S_ref measured: 3398 fenced-N16 /
> 6406 plain-N32; scaling verified to reach sdpa in transformers 4.57.6). Runs: L1
> chains 137821/137822 → `outputs/loramech/l1_hahn/{ff_le32,ff_le8}_N*/` (+_fig);
> N2 chains 137901/137902 → `outputs/loramech/n2_hahn/{p1fence_ep10,p1fence_frozen}_N*/`
> (+medians/figs); N3 137899/137900 → `outputs/loramech/n3_logn_{fenced,plain}/`;
> P3 trained-with-logN 137937 → `outputs/loramech/p3_fenced_logn/` (adapter
> `checkpoints/sft_fenced_logn_adapter/`, eval-contract: flag required). Same
> pools/seed/pairs as ARMOR-A (paired vs frozen α=0.72 [0.63,0.78]).

| α (L20, paired one-frame flips, bootstrap CIs) | value |
|---|---|
| ff_le32 in-window {8,16,32} | 0.710 [0.36,0.93] |
| ff_le8 {8,16,32} | 0.732 [0.32,0.93] |
| trained-fenced VERDICT locus (rep_t), all N | **+0.009 [0.001,0.016] — FLAT**, magnitude ~125 (frozen ~55) |
| trained-fenced READ (final), {8,16,32} | **+0.802 [0.66,0.97]**, magnitude 26→4.5 (above floor at 128) |
| frozen-fenced read {8,16,32} | +0.774 [0.70,0.86] (floors ~1.6 by N=32) |

- **L1 (H2):** GAIN confirmed at the α level for both plain adapters — training does
  not change the decay law, it multiplies the per-frame contribution ×3–6 (29.4 vs
  10.5 @N=8; 7.4 vs 1.3 @N=128, lifted clear of the bf16 floor while margins still
  slide −0.60 @64 → −2.42 @128). H1 out-of-window: ff_le8 met (0.593), ff_le32
  indeterminate (0.287 [0.01,0.63], a 32→64 plateau, logged as-is). Formal L2/L3
  co-conditions not run (plain-layout probe deltas pending) — α + anatomy carry the
  verdict.
- **N2 (the mechanism table):** the fence's supply invariance holds trained AND frozen;
  training amplifies verdict content ×2.3 and read gain ×6; the read decays Hahn-like
  in both, and the behavioral undercount onset (fine @2×, −1.5 offset @4×) coincides
  with the read signal falling 26→6. At N=128 the trained read is far above noise yet
  acc = 0.23: MARGIN-bound, not noise-bound.
- **L5 closed in all three configurations:** plain + eval-logN = NULL per the
  pre-registered H3 (0.250 @64 vs 0.293, no-harm intact) — sharpening cannot rescue a
  read that must also extract; fenced + eval-logN = real bounded rescue (0.950 @32,
  0.470 @64 with the low band reviving, 0.230 @128 unchanged); fenced trained WITH
  logN (P3) = perfect in-window calibration (1.000 @8 AND @16 — the +1 artifact was
  calibration and is GONE) but the readout learns to depend on compensation and the
  free octave is spent (0.420 @32; 64/128 cells job 138006 pending, appended on
  landing). Practical §6 recipe: TRAIN plain (P1b), APPLY logN at eval.
- **The campaign's closing claim:** counting-over-frames decomposes into conditioning
  (where verdicts may form — the fence's job), gain (what LoRA training does — margins
  on an unchanged read), and code (magnitude vs token). The first is fixable, the
  second trainable, the third is the wall: the count is magnitude-coded in a softmax
  read and no re-scaling made 4×–8× reachable — the method's caption-scan enumeration
  (token-coded readout) is the unique measured route past it, which is why the full
  stack holds 0.981 @64. Every component of the method now traces to a measured
  failure of a cheaper alternative.


## [2026-08-28] 📊 LORAMECH addendum — P3 zero-shot cells land (job 138006, 5 preemption-restarts): trained-with-logN reads **0.310 @64 / 0.170 @128** — below eval-only scaling (0.470/0.230) at every out-of-window length; the mechanism entry's P3 verdict is final: compensation is an eval-time patch, never a training prior

> `outputs/loramech/p3_zs_64_128/` (100/cell, pf 0, flag on per the adapter contract).
> Complete P3 ladder: 1.000 / 1.000 / 0.420 / 0.310 / 0.170 @ N=8/16/32/64/128.
> Same anchor anatomy as all fenced arms out-of-window (g0, g12/g16 label-support,
> and all-frames survive; low band dead). Closes the LORAMECH campaign's last open cell.

<!-- ══════════════ SPARSE campaign (2026-08-30→31) — entries appended at campaign close per Tal's 2026-08-30 pre-authorization; full log outputs/sparse/STATE.md ══════════════ -->

## [2026-08-30→31] ❌📊 SPARSE S0+S1-control — THE k-REGIME IS NOT REACHABLE AT EVAL TIME: no fixed sharpening τ ∈ {1.5,2,3,4} lifts the k≤8 band past ~0.24 @N=64 (bar 0.80) while every τ>1 breaks N=32 by ≥0.37 with mid-band OVERcount; and P1b under the oracle gate without retraining answers ≈N for every k (mse +20.79) — the softmax read's calibration is competitor-mass-specific in BOTH directions

> SPARSE campaign (brief `outputs/sparse/CAMPAIGN_BRIEF.md`, log/verdict
> `outputs/sparse/STATE.md`, index `outputs/sparse/INDEX.md`). S0 jobs 138976–81 →
> `outputs/sparse/s0_tau{1.0,1.5,2.0,3.0,4.0}_L12/` + `s0_tau2.0_L0/` (all-layers
> ctrl); N∈{32,64,128}×100 exam dirs each, pf 0, majority 0.06–0.08. S1-control
> 138982 → `outputs/sparse/s1_control/` (P1b + oracle gate @N=32, 150 dirs).
> Instrument: `scripts/sparse/train_sft_gated.py --attn-sharpen τ` (τ=1.0 ref
> reproduces the P1b ladder 0.850/0.340/0.230 on the 100-slice). Figure
> `outputs/sparse/fig/F4_s0_sweep.png`.

- **H-S0 → the needs-training side; S6 not triggered.** Best k≤8 band @64 ≈ 0.24
  (τ1.5: k1-4 0.46, k5-8 0.00) vs the 0.80 bar; every τ>1 fails the N=32
  side-condition (drop ≥0.37 vs allowed 0.03). Over-sharpening threshold = τ1.5,
  and sharpening never breaks FORMAT (parse-fail 0 through τ4) — it breaks
  CALIBRATION: the lawful undercount flips to mid-band overcount (τ2 @64: k5-8
  mse +7.1) while g0 and in-support anchors survive longest. Layer restriction
  (≥12 vs all layers) is a wash. A fixed factor fails two lengths at once
  (τ1.5: +0.67 @32 while still −1.73 @128) — the (N−k) term needs a structural
  fix, not a constant.
- **H-S1-control CONFIRMED ×40 over its band** (predicted mse ≥ +0.5): P1b +
  oracle gate @N=32 reads 0.080 with mse **+20.79**; k1-4 0.000 (+29.5), k5-8
  0.000 (+25.5); the ONLY intact stratum is g32 "all frames" (11/11) — with the
  (N−k) competitors hidden, the frozen-calibrated share saturates and the read
  emits ≈N. The m = k·eˢ/(k·eˢ+(N−k)+C) code, measured from its other side.

## [2026-08-30→31] ✅📊 SPARSE S1 (P1g) + THE MASK/CHECKPOINTING BUG — gated fenced-SFT under the oracle evidence gate: k≤4 EXACT AT EVERY N 8→128 (140/140, 8–16× past training) and in-window 1.000/0.947 (beats P1b's 0.893 @16); H-S1's ≥0.97-every-stratum headline REFUTED by ONE lawful mode — beyond a boundary that CONTRACTS with N (≈8–12 @32 → 6 @64 → 5 @128) the model snaps to answering exactly "N". Found en route: clearing a hook-injected attention mask before .backward() lets gradient checkpointing recompute UNGATED — gradients 22° off truth, two divergent trainings; holding the mask through backward is bit-exact

> Trainer `scripts/sparse/train_sft_gated.py` (`--gate oracle`: hide_cols = union
> of non-evidence block spans via the canonical build_block_mask, train AND eval
> forwards; copy-extension per brief §5, loramech untouched; no-flag path
> re-anchored at exam_ff_N8 = 1.0000 after every delta). THE BUG: diverged runs
> 138975 (lr 2e-4, val 0.55→0.05) + 139033 (lr 1e-4, val 0.53→0.12) kept at
> `outputs/sparse/s1_p1g{,_lr1e4}/`; diagnosis chain `outputs/_scratch/
> sparse_smoke/diag_loss_frozen/` (init losses benign — 139055) → `diag_grad/`
> (139064: LoRA grads, one gated sample — current pattern vs ground truth cos
> 0.7767, mask-held vs truth cos 1.000000/max|Δ| 9e-10). ⚠ The same latent
> pattern exists in `scripts/loramech/train_sft_fenced.py` (evals inference-only
> = unaffected; P1b converged through the milder plain-fence mismatch); left
> untouched per scope. CANONICAL fixed run 139066 → `outputs/sparse/s1_p1g_r3/
> 20260831_011807_gated/` (P1b recipe verbatim, lr 2e-4, ep1 val 1.000, TEST_IID
> 0.986, 5h07 l40s); ladder 139105 → `outputs/sparse/s1_ladder_64_128/`. Adapter
> `checkpoints/sft_fenced_gated_adapter` — EVAL CONTRACT: gate required. Figure
> `outputs/sparse/fig/F1_flatline.png`.

| N (oracle gate, 150/cell) | acc | vs P1b | anatomy |
|---|---|---|---|
| 8 | **1.0000** | = | all strata perfect |
| 16 | **0.9467** | 0.893 | k7/k8 wobble 9/13 (margin-thin), k12/k16 perfect |
| 32 zs | 0.7800 | 0.867 | **k≤8 PERFECT 95/95** + g32; g12/16/24 = 0, ALL → "32" |
| 64 zs | **0.5533** | 0.340 | k≤5 perfect 70/70; k8 0; k12-48 → "64"; g64 10/10 |
| 128 zs | **0.4400** | 0.230 | k≤4 perfect 45/45; k12-64 → "128" (53/53); k96/128 parse-fail/absurd; pf 0.053 |

- **H-S1 headline REFUTED as pre-registered** (k≤8 band @128 = 0.79 < 0.85; the
  every-stratum clause fails from N=32). What stands: the first arm on record
  with a **length-independent exact band** (k≤4, 100% at 8–16× zero-shot), and
  the failure is a single snap-to-N mode + an overflow regime near k≈N at 16×
  (g128 0/8 where P1b kept it). Diagnosis per the pre-registered escape clause:
  positional story excluded (posreset; S3 verdict channel flat) — the snap tracks
  the PROMPT's declared N (every mid-k error is the exact string "N"), i.e. with
  the competitors gone, k·eˢ competes against fixed prompt mass C and the trained
  "all" threshold shifts with declared N. Candidate next lever: N-randomized/
  N-free prompts at training (logged as open).
- Convergence note: under the gate (with correct gradients) the task is EASIER
  than ungated — val 1.000 by ep1 vs P1b's ep8 peak.

## [2026-08-31] ✅📊 SPARSE S3 — α_N ≈ 0 UNDER THE GATE, MEASURED: through the gated read the answer-locus sensitivity is FLAT in N (α +0.07 [−0.08,+0.20] vs +0.80 [0.66,0.97] ungated; margins positive O(1) at N=128; a one-frame flip changes the emitted answer 80–98% of the time at every N ∈ {8..128}) and the decay moves to k: Δ ∝ (k+2)^−1.19 [1.16,1.22] — the (N−k) term is gone from the read and the resolution wall is in k

> Instrument `scripts/sparse/probe_hahn_gated.py` (`--gate oracle`, p1fence arm,
> per-member masks: base hides flip block t, evid member exposes it; no-flag path
> proven BYTE-IDENTICAL to `scripts/armor/probe_hahn.py` same-arch — 138974; the
> flagged-superset lineage to the a100 n2_hahn reference was proven by REDUX
> 138967). Chains 139107/139108 → `outputs/sparse/s3/gate_oracle_N{8..128}/`
> (N2 pools/limits: 50×4+40 pairs, ctrl 12) + `gate_oracle_N64_k{1,2,4,8,16,32}/`
> (30/k, redux N=64 pool). P1g adapter throughout. Figure `fig/F3_alpha.png`.

| locus | gated α [95% CI] | medians N=8→128 | ungated (N2) |
|---|---|---|---|
| READ L20 final | **+0.066 [−0.076,+0.195]** | 26.4→21.3 FLAT | **+0.802 [0.66,0.97]** (26→4.5) |
| READ L16 / L28 | +0.018 / +0.018 (CIs ∋ 0) | flat | — |
| verdict L20 rep_t | −0.007 [−0.030,+0.014] | ~55 flat | +0.009 (flat) |

- **H-S3 MET** on flatness (CIs ∋ 0, |α| ≤ 0.07 point at every layer) and on
  decay-in-k at the read locus (γ 1.19 [1.16,1.22] ≥ 1, CI excludes 0.5; L16 0.77,
  L28 2.56 post-norm — reported per-layer). Margins: median base +5.60 @8 →
  +3.67 @128 (stays positive; ungated slid to −1.91). By k≥16 the +1-flip signal
  (2.9–4.1) nears the measured bf16 floor (~1.2–1.4) — the S4 behavioral
  boundary sits where the flip signal meets noise. The k-regime construction
  k/(k+C): both halves now measured (N removed, k inherited).

## [2026-08-31] ✅⚠📊 SPARSE S2+S4+S5 — THE GATE IS DEPLOYABLE AND THE WALL IS IN k: a logistic gate on L20 replica-slot states reads per-frame evidence at ≥0.999 accuracy FLAT in N (1 miss/19,200 frames @N=128, d′ 6.4–8.4) and the model-gated system equals the oracle to within per-sample-named gate errors (identical at N=64: 0 errors/9,600 frames); trained capacity c* = 8 @N=64 / 7 @N=128 (H-S4's 24–48 MISSED — c* ≈ the frozen c(fan) crossing; the gate buys EXACTNESS below c* at 16×, not a larger c*); MMReD-HF transfer 0.900/0.700/0.540 (H-S5 missed; the loss is the same snap band)

> S2: captures 139109–14 (`scripts/sparse/capture_verdicts.py`, L20 room-word
> loci) → `outputs/sparse/s2/cap_train/` (651 samples/6,216 frames, pos 0.469) +
> `cap_N{8..128}/`; gate `scripts/sparse/train_gate.py` → `outputs/sparse/s2/
> gate/` (LR, standardized, sample-held-out val 1.0000); model-gated evals
> 139125/26 → `s2/eval_{short,long}/` (two-forward: ungated pass-1 → gate →
> gated decode; gate_fn/gate_fp per sample). S4: 139106 → `outputs/sparse/s4/
> eval/` (topped-up 20/gold dirs files `s4/dirs_N{64,128}.txt`, strided from the
> same longN_park pools). S5: 139127 → `outputs/sparse/s5_hf/` (N4 protocol,
> 50/len). Figure `fig/F2_capacity.png`.

| ladder (150/cell) | N8 | N16 | N32 | N64 | N128 |
|---|---|---|---|---|---|
| P1g oracle gate | 1.000 | 0.947 | 0.780 | 0.553 | 0.440 |
| **P1g model gate** | 0.993 | 0.960 | 0.753 | **0.553** | 0.427 |
| gate frame-errors | 1 fp | 0 | 5 fp | **0** | 1 fn + 4 fp |

- **H-S2: gate clause MET** (≥0.99/frame at every N — measured 0.999–1.000,
  flat: the ARMOR-A flat supply carried through the trained adapter into
  deployment); **parity clause MET** (every oracle−model gap traces to named
  gate-error samples; N=16's +0.013 flip is margin-thin bf16 jitter on the k7/k8
  wobble samples, gate errors 0); the **≥0.90 @128 k≤8 clause NOT MET** (0.79 —
  bounded by the S1 snap, not by the gate).
- **S4 (H-S4 MISSED):** exactness vs k at fixed N crosses 0.5 at **k=8 (N=64) /
  k=7 (N=128)** vs expected 24–48 — ≈ the frozen c(fan) crossing (~8). k≤5 =
  100/100 @64, k≤4 = 89/89 @128; k≥12 all → "N"; parse-fails live entirely in
  k≥64 @N=128 (4/12/15 at k=64/96/128). The gate converts the sub-c* regime to
  EXACT at 16× (frozen fan-8 was 0.65 at its own length) but does not move c*:
  capacity is code-resolution-bound (S3's γ≈1.2 + bf16 floor), and token-coded
  readout (the method's caption scan, 0.981 @64) remains the only measured route
  past k≈8.
- **S5 (H-S5 MISSED):** model-gated HF 0.900/0.700/0.540 vs P1b's 1.000/0.820/
  0.540 (band wanted ≥+0.10 on 16/32). Gate transfers (frame-error rate ≈0.4%);
  k≤5 near-perfect (seq32 25/26); the entire loss is the snap band. Anatomies at
  seq32 are complementary (same 0.540): P1b keeps label-support anchors
  (g12/g16 3/3) but drops g2 0/4; gated keeps ALL k≤5 and drops mid-k.
- **Caveats (campaign-wide):** single task family + model; oracle vs model gate
  labeled everywhere; HF cells 50/len with k-strata of 1–7; k-strata <10 marked;
  the gate presupposes the fence (flat supply — ARMOR-A) and per-frame verdict
  slots; c* numbers are at 20/gold resolution.

<!-- ══════════════ SPARSE WAVE 3 (2026-08-31→09-01) — appended at wave close per Tal's 2026-08-31 pre-authorization; full log outputs/sparse/STATE.md (wave-3 verdict), index outputs/sparse/INDEX.md ══════════════ -->

## [2026-08-31] ✅📊 SPARSE S7 + EQUIVALENCE — THE SNAP IS THE PROMPT'S DECLARED N, CAUSALLY: lying to the model about N relocates the snap to the declared string, exactly (k12/k16 on true-32 inputs recover 0.00 → 1.00 WITH CORRECT ANSWERS under "16 frames"; k8 collapses 1.00 → 0.00 under "64"; k4 = 120/120 across all six lies) — and the gated N-frame forward is proven ≡ the evidence-only forward (20/20 identical decodes), so the twins reproduce the snap with NOTHING hidden

> Wave-3 instruments: `train_sft_gated.py --declare-n` (prompt TEXT only; frames/
> mask/gate/positions untouched) + `scripts/sparse/diag_equiv.py`. Anchor re-held
> (P1b exam_ff_N8 = 1.0000 after every delta). Equivalence 139240 →
> `outputs/_scratch/sparse_w3/equiv/`: gated N-frame forward vs evidence-only twin
> declaring N — 20/20 greedy decodes identical, median rel L2 @answer L20 = 0.0102
> (< the 0.02 pre-registered bar); the k=12/16 twins answer "32" with no hidden
> blocks at all. S7 chains 139245/139246 → `outputs/sparse/s7/N{32,64}_*/` (20/
> stratum, fresh longN_park dirs, majority 0.25, pf 0; P1g + oracle gate).

| true N=32 | acc | k4 | k8 | k12 | k16 | snap target |
|---|---|---|---|---|---|---|
| declared 32 (ref) | 0.500 | 1.00 | 1.00 | 0.00 | 0.00 | "32" 40/40 |
| declared 16 | **0.900** | 1.00 | 0.60 | **1.00** | **1.00** | — (answers correct) |
| declared 64 | 0.250 | 1.00 | **0.00** | 0.00 | 0.00 | **"64" 40/40** |

- N=64 replicates (declared-32 recovers k8 0.00 → 0.95; k12 snaps to "32").
  **Both H-S7 clauses MET; side condition MET** (k4 never drops). Fine structure:
  a −1-undercount ring precedes the snap (k ≈ Ñ/2: all-"7" at k8/declared-64).
  Every stratum's fate is a function of (k, declared Ñ) alone — true N appears
  nowhere once the gate is on. The wave-2 "boundary contracts with N" law
  re-parameterizes entirely in the DECLARED N: pure text calibration, the
  mechanism behind S8/S9 confirmed before either trained.

## [2026-08-31→09-01] ✅📊 SPARSE S8 (VIRTUAL-N) — THE SNAP IS DEAD: full (k, declared-Ñ) training coverage makes the k≤8 band EXACTLY 1.000 AT EVERY N ∈ {8..128} (wave-2: 1.00/0.95/1.00/0.73/0.63), k12 perfect at every N, in-window PERFECT (1.000 @8 AND @16), parse-fails gone — and the trained capacity edge is c* ≈ 16 (wave-2's c* = 7-8 SUPERSEDED as calibration-confounded), exactly where S3's resolution physics put the floor; the deployable model-gated system tracks oracle to named per-sample gate errors

> Trainer delta `--virtual-n` (50% real gated + 50% synthetic evidence-only:
> k∈0..16 stratified × declared Ñ∈{8,16,32,64,128}, k≤Ñ, answer=k — licensed by
> the equivalence smoke; k=0 = text-only step). Training 139247 (S1-r3 recipe
> verbatim otherwise, best ep3 val 1.000, 9h04) → `outputs/sparse/s8_vn/
> 20260831_134617_gated/`; ladder 139451 → `s8_ladder_64_128/`; capacity 139452 →
> `s8_s4/` (20/gold); gate refit + model ladder 139471/72 → `s8_gate/`,
> `s8_mg/`. Adapter `checkpoints/sft_fenced_gated_vn_adapter` (**CANONICAL** gated
> config; eval contract: gate required). Figure `outputs/sparse/fig/F5_flatline_w3.png`.

| N (oracle gate, 150/cell, pf 0) | acc | k≤8 | k12 | k16 | k≥24 |
|---|---|---|---|---|---|
| 8 / 16 | **1.0000 / 1.0000** | 1.000 | — / 13/13 | — / 13/13 | — |
| 32 zs | 0.8533 | **1.000** | 11/11 | 11/11 | 0 |
| 64 zs | 0.7333 | **1.000** | 10/10 | 10/10 | 0 |
| 128 zs | 0.6067 | **1.000** | **9/9** | 1/9 | 0 |

- **H-S8-main MET (bar ≥0.95): measured 1.000 ×5. H-S8-noharm MET and exceeded**
  (in-window PERFECT — P1g's k7/k8 wobble gone). Capacity (S4 dirs, 20/gold):
  @64 k0-k16 = 170/170; @128 k12 = 20/20, k16 = 5/20 → **c*(128) = 16**, double
  the wave-2 reading, in S3's predicted 16-20 window (γ≈1.2, flip signal at floor
  by k≈16) — the wave-2 S4 entry is annotated calibration-confounded. k≥24 = 0
  with MIXED signed error (no snap): the stated k>16 coverage hole, now the only
  wall below k≈N. **H-S8-model MET**: refit gate ≥0.9973/frame flat (recall
  1.0000 at every N); deployed k≤8 band 0.99/1.00/0.92/0.96/0.94 with every gap
  = counted gate errors (e.g. @64: 3 fp = the 3 lost samples).
- The convergence note: with correct coverage the gated task trains EASIER than
  ungated (val 1.000 @ep3 vs P1b's ep8 peak) — calibration, not capacity, was
  consuming the optimization.

## [2026-08-31→09-01] ✅⚠📊 SPARSE S9 (N-FREE PROMPT) — THE MECHANISM NEEDS NO N ANYWHERE: with the frame count and answer range REMOVED from the prompt (matching the upstream benchmark, which never states N — our N-injection was legacy-local), the same recipe with no synthetic mixture reads k≤8 at EXACTLY 1.000 at every N ∈ {8..128}; S8 stays canonical (k12 + cleanliness), S9 is the protocol-fidelity variant; HF transfer misses its band (gate recall on HF is the driver)

> Trainer delta `--nfree-prompt` (same "You will be shown" opener — parse
> unchanged; adapter contract carries the flag; S9 numbers = a NEW prompt anchor,
> never tabulated against N-prompt arms unlabeled). Training 139318 (S1-r3 recipe,
> no mixture, best ep3 val 1.000) → `outputs/sparse/s9_nfree/20260831_174021_gated/`;
> ladder 139461 → `s9_ladder_64_128/`; capacity 139462 → `s9_s4/`; gate + model
> ladder 139486/87 → `s9_gate/`, `s9_mg/`; HF 139488 → `s9_s5_hf/`. Adapter
> `checkpoints/sft_fenced_gated_nfree_adapter` (gate + `--nfree-prompt` required).

- **H-S9-main MET: oracle k≤8 = 1.000/1.000/1.000/1.000/1.000** — no N in the
  text, no virtual N, real gated samples only: the flat line is a property of the
  gate + coverage-free honesty of the N-free decoder. In-window 1.000/0.973.
  **H-S9-capacity MET in range**: k16 20/20 @64, c* ≈ 16 @128 — confirms the
  supersession independently. k12 is non-monotonically dented (11/20 @64/128;
  @128 the errors emit "128" 7/20 WITH NO N IN ANY TEXT — candidate channel: the
  step-ids burned into the frame PIXELS, which S8's mixture taught the model to
  ignore in favor of the prompt; logged as interpretation, testable by re-rendering
  with step-ids stripped). High-k parse-fails (pf 0.15-0.19 @128, no range hint).
  Deployed: k≤8 = 1.000 @64, 0.958 @128 (= oracle − 8 named gate errors).
- **H-S9-vs-S8 → S8 CANONICAL** (k≤8 tie; S8 wins k12 everywhere + pf 0), S9 =
  the fidelity proof. **H-S9-HF MISSED**: 0.860/0.780/0.580 vs ≥0.95/0.90/0.85 —
  though ≥ wave-2's S5 at seq16/32 with clean errors (mse ≈ 0); the measured
  driver is gate RECALL on HF (fn 6-7/cell vs ~0 on park) + the k12+ edge; open
  lever: gate retraining on HF captures.
- **Wave-3 caveats:** single task family/model; S8's k≥24 zeros are coverage
  (grid k≤16), not measured capacity, through k≈16-24; the k≈N regime at large N
  is unknowable-by-design in S9 and untrained in S8; S9's k12 dent awaits the
  pixel-channel test; k-strata <10 marked in run logs.

## [2026-09-01] 📊 SPARSE wave-3 addendum — S9's k12 dent decomposed: the dominant attractor is the TRAINED ANSWER CEILING "16" (4/4, 9/9, 14/14 of the k12 errors at N=16/32/64), not the pixel channel; root cause = answer-support sparsity (the real mixture's gold support is 0-8 dense + {12,16} thin — values 9-11/13-15 never trained), which S8's dense synthetic k-grid fixes by construction — the "128" attractor at N=128 (9+16 errors at k12/k16) remains the pixel-channel candidate

> Analysis over `s9_{nfree,ladder_64_128,s4}` prediction CSVs (STATE 2026-09-01
> ~09:40). Consequence for §6: the N-free config needs k-BALANCED evidence-subset
> oversampling (S9b, = S8's coverage trick without the Ñ text) rather than more
> epochs (same budget as S8; best ep3 val 1.000, later lower-loss epochs never
> beat it). Mechanics note: under the gate, k≤16 cells at ANY N present ≤16
> visible blocks (equivalence smoke) — the gate converts length-generalization
> into k-generalization; genuinely longer-than-trained visible inputs begin at
> k≥24, which is also outside trained answer support (those zeros stay
> coverage-confounded).

## [2026-09-01] ✅📊 SPARSE S10 — THE ATTENTION PHOTOGRAPH: the share law m = k·eˢ/(k·eˢ+(N−k)+C) is OBSERVED (R² 0.93–0.99 per layer, 9/9 N-doubling sign checks; L20: s=0.30, C=8 — the trained read's attention edge over a competitor block is only ×1.35), under the gate the mass is N-invariant (≤6% across N=8→128 at the read locus) — and a single trained-in head (L24 h20) separates evidence from non-evidence at AUC ≥0.993 at every N in the UNGATED model: detection was never the problem, aggregation was

> New instrument `scripts/sparse/probe_attn_photo.py` (manual attention at the
> last prompt row from FenceHooks q/k captures + rotary — sdpa exposes no weights;
> per-row softmax-sums-to-1 asserted; adapter loaded BEFORE capture hooks so LoRA
> q/k are measured). Jobs 139736 (smoke: gated non-evidence mass EXACTLY 0.0000 —
> the mask verified in the photograph itself) + 139760-62 →
> `outputs/sparse/s10/{p1b_N8..64,gated_N8/32/128,frozen_N32}/` (30/k-stratum,
> k∈{2,4,8}, layers {12,16,20,24,27}, head-mean over 28 heads). Figure
> `outputs/sparse/fig/F7_attn_photo.png` (+CSVs).

- **H-S10a MET:** P1b-arm evidence mass at L20, k=4: 0.31 → 0.21 → 0.14 → 0.09
  across N=8→64 — dilution watched happening; one-(s,C)-per-layer fits R² =
  0.989/0.992/0.969/0.992/0.933 at L12/16/20/24/27; sign check 9/9 decreasing at
  every layer. The formula every SPARSE/LORAMECH intervention presupposed is now
  a measured object.
- **H-S10b MET at the read locus:** gated (S8) L20 evidence share at fixed k
  varies 2.0–6.2% across N∈{8,32,128} (its own fit k·eˢ′/(k·eˢ′+C′), s′=0.26,
  C′=46 — the gated sink holds ~5× more mass than P1b's C=8); worst off-read
  layer 16.3%, reported.
- **H-S10c (prominent):** best FIXED head **L24 h20 separates evidence vs
  non-evidence blocks at AUC 0.9997/0.9927/0.9999/1.0000 @N=8/16/32/64** in the
  UNGATED P1b model; frozen best head 0.848 → the near-perfect internal evidence
  selector is TRAINED-IN, yet its mass obeys the same share dilution — the model
  knows which frames matter and cannot cash it in through softmax aggregation.
  The external gate (mask, or the LR readout at ≥0.999/frame) is the read-out of
  this internal signal. Caveats: head-mean summaries pool heterogeneous heads;
  k∈{2,4,8} only; single task family/model.

## [2026-09-01] ✅📊 SPARSE S11 — α WITH BYTE-IDENTICAL TEXT AT EVERY N (the S9 N-free adapter): read α = +0.035 [−0.043,+0.123] (medians 28.5→25.7, FLAT), margins CONSTANT at +11.7 nats from N=8 to N=128, one-frame flips change the emitted answer 90–100% of the time at every N, and decay-in-k is γ = 1.19 [1.17,1.22] — numerically the P1g chain's γ: the k-resolution law belongs to the gated softmax read, not to any adapter — NO non-text N-channel exists; the |α| ≥ 0.3 refutation branch is dead

> Instrument: `probe_hahn_gated.py --nfree-prompt` (single-source S9 builder;
> no-flag anchor re-proven BYTE-IDENTICAL — 139735). Chains 139752/139753 →
> `outputs/sparse/s11/gate_oracle_N{8..128}/` (50×4+40 pairs, ctrl 12, seed 0)
> + `gate_oracle_N64_k{1,2,4,8,16,32}/` (30/k). Figure
> `outputs/sparse/fig/F6_s9_alpha.png` (+CSV).

- **H-S11a MET** (CI ∋ 0, |α| < 0.15; margins median positive at 128; flip-rate
  ≥ 0.7): this is the S3 measurement with the last confound removed — in the
  P1g chain the prompt text varied with N ("You will be shown {N}…"); here every
  cell is byte-identical text, so flatness is attributable to the mechanism
  alone. Verdict channel flat (−0.016 [−0.048,+0.005]); N-free margins (+11.7)
  are both LARGER and FLATTER than the P1g chain's (+5.6→+3.7).
- **H-S11b MET:** γ(L20) = +1.19 [1.17,1.22] with C=5 — matching the P1g-adapter
  chain to two decimals (L16 1.41, L28 2.79 post-norm). Cross-adapter agreement
  pins the decay-in-k as a property of the gated read itself.
- Together with the equivalence smoke and S7: the N-channel taxonomy is closed —
  text was the whole story; positions/sink carry nothing measurable.

## [2026-09-01→02] ✅📊 SPARSE S9b — k-BALANCED EVIDENCE-SUBSET OVERSAMPLING UNDER THE N-FREE PROMPT: the ep10 arm is EXACT ON EVERY TRAINED ANSWER VALUE (k = 0..16) AT EVERY N ∈ {8..128} — including k16@128 at 20/20, the cell the S8 N-prompt arm read at 5/20 — making it the strongest measured adapter of the campaign on the upstream-faithful prompt; the ep5/ep10 twin pair doubles as a budget ablation (same recipe, half cost, ±3–5% checkpoint-draw variance at margin-thin strata, invisible to val-by-decode)

> Root cause being fixed (recorded for the write-up): the park seq16 training pool
> is ANCHOR-CLASSED — its gold support is K0–K8, K12, K16 ONLY (30/class; answers
> 9–11 and 13–15 do not exist in any real training data; g12/g16 reach the mixture
> at 17 samples each vs ~100/class for 0–8). S9's k12→"16" rounding was this
> support hole (2026-09-01 addendum). S9b = the S9 recipe + the S8 mixture
> machinery with the Ñ dimension collapsed (`--nfree-prompt --virtual-n`: grid =
> k∈{0..16} only, ~38 dense exposures/answer/epoch; evidence-subset construction,
> licensed by the equivalence smoke). Runs: ep5 arm 139718 →
> `outputs/sparse/s9b_ep5/` (+`s9b_ladder/`, `s9b_s4/` 139774/75); **ep10 twin
> (CANONICAL) 139714 → `outputs/sparse/s9b_vn_nfree/20260901_140231_gated/`**
> (+`s9bT_ladder/`, `s9bT_s4/` 139977/78). Adapter
> `checkpoints/sft_fenced_gated_vn_nfree_adapter` (contract: gate +
> `--nfree-prompt`; N-free prompt anchor — label the variant in any comparison).

| N (ep10 twin, oracle gate) | overall exam | k0–k16 (= trained support) | k≥24 (untrained) |
|---|---|---|---|
| 8 | **1.0000** | perfect | — |
| 16 | **1.0000** | perfect | — |
| 32 zs | 0.8467 | k12 10/11, k16 11/11, rest perfect | 0 |
| 64 zs | 0.7333 | **exam 110/110 + S4 170/170 PERFECT** | 0 |
| 128 zs | 0.6600 | **exam 99/99 + S4 k12 20/20, k16 20/20 PERFECT** | 0 |

- **The coverage fix works completely at ep10**: S9's k12 (0.18–0.69) → 0.9–1.0
  everywhere; k16@128 (0.34) → **1.00 at both cell sizes** — BEATING the S8
  N-prompt arm's 1/9 + 5/20 there. Every remaining error in the ladder is an
  untrained answer (k ≥ 24: 0/191 across cells, parse-fails concentrated there —
  the N-free prompt gives no range hint, so beyond-support reads emit junk rather
  than a snapped number; mae on PARSED answers 0.68 @64).
- **Budget ablation (pre-authorized 5-ep default, evidence refined):** the ep5
  arm closed k12 identically but carried a +1 wobble at k6–k7 (all 12 errors
  exactly +1) that the ep10 twin does NOT reproduce → checkpoint-draw variance
  between equal-val-1.000 checkpoints (60-sample val-by-decode cannot separate
  them), not a capacity or crowding effect (that hypothesis was raised and is
  hereby withdrawn for the wobble). Practical: keep the 5-ep default for
  iteration; train 10 ep (or add a k-stratified val) before promoting an adapter.
- Trained-support view across the N-free family (gold ∈ trained support, oracle):
  S9 (no mixture) = 1.0000 on gold 0–8 at every N (827/827) but 0.89–0.97 with
  the thin g12/g16 included; S9b-ep10 = 1.0000 on the FULL 0–16 support at N ≥
  64 (32-cell: ONE miss in 128 — a single k12→"13"). Consolidated trained-support
  tally, verified from prediction CSVs across all ep10 cells: **992/993 = 0.9990**
  (perfect at N=8/16/64/128; the sole miss = steps_in_room_N32_K12_0017).
  **Canonical calls: S8 = N-prompt canonical;
  S9b-ep10 = N-free canonical and the campaign's best long-N profile** (labeled
  prompt variants, never one table row unlabeled).
- Caveats: single task family/model; k-strata 9–11 at N=32 exam cells; pf at
  untrained strata inflates MAE-style summaries (parsed-only mae reported);
  the S9c data-side variant (uniform-gold seq16 regeneration) is specced but
  unlaunched — the conventional-fix comparison remains open.

## [2026-09-14→16] ⚠️📊 SPARSE LAYOUT × GATE FACTORIAL + MINIMAL-FIX LADDER — the per-block question replicas are a CONDITIONING device that the oracle gate replaces (qlast-once, question-blind blocks: ungated 0.447/0.380/0.260/0.240/0.147 @N=8..128 → oracle-gated 0.967/0.900/0.687/0.527; N-free twins gated qlast ≈ gated replica 1.000/0.913/0.827/0.607 vs 1.000/0.913/0.813/0.667), and the retrained part shrinks 5–8× without losing the k≤8 band (A2 r1-on-everything 2.97M and A3 attention-only r8 5.05M = k≤8 1.000 at N=64 AND 128; A1 r1 q,v L20–27 90k = 0.975 @64 / 0.802 @128; readout-only A0 caps at 0.57–0.74)

> Instruments: `train_sft_gated.py --layout {replica,qfirst-once,qlast-once}` (+ `--lora-r/--lora-targets/--lora-min-layer` for the ladder; A0 = `calib_digits.py`), now under `legacy/v1/scripts/sparse/`; anchors 149068 / 148722 = eval-only P1b exam_ff_N8 1.0000 with the edited trainer (`outputs/_scratch/{layout,minfix}/anchor_p1b_N8/`). Regime: park data (legacy generator: `data/mmred_images_park/seq_len_8` + `data/mmred_longN_park/seq_len_16`, limit 900, 5 ep, lr 2e-4, r8-all unless stated), count task; layout arms = declared-N prompt WITHOUT virtual-N (P1b-style; `*_gated_nfree` cells = N-free prompt); minfix arms = S8 recipe (declared-N + `--virtual-n`), oracle gate. Exams `outputs/loramech/examdirs/exam_ff_N{8..128}.txt`, 150/file (gated N=128 reruns 100/cell). Runs: `outputs/sparse/layout/{replica,qfirst_once,qlast_once}_ep5/` (149076/77/78; N=128 eval-only reruns `*_N128/` 150264/149378/150265 after the a100-40GB OOM-as-skip), `{qlast,qfirst}_once_ep5_gated/` (150433/34; `eval_N128/` 150717/18), `replica_ep5_gated/20260916_020357_gated/` (150583 after the n317 stall kill of 150572; `eval_N128/` 150770), `{replica,qlast_once}_ep5_gated_nfree/` (150581/150580; `eval_N128/` 150769/150768); `outputs/sparse/minfix/{a0_calib,a1_qv_r1_L20,a2_all_r1,a3_attn_r8}/` (148725–28) + `*_N128/` (150263/149541/149540). READMEs in both dirs; 2×3 factorial table also in `docs/PRIOR_ART_MASKGATE_2026-09-16.md`.

- **Ungated layouts (declared-N, no virtual-N):** replica 1.000/0.987/0.493/0.300/0.173, qfirst-once 0.947/0.880/0.460/0.220/0.200, qlast-once 0.447/0.380/0.260/0.240/0.147 @N=8/16/32/64/128 (10-ep replica reference P1b 1.000/0.893/0.867/0.340/0.230, README). qfirst-once ≈ replica in-window (−0.05/−0.11) → N query copies were mostly conditioning, not "one reader per item"; question-BLIND blocks collapse already at N=8 (0.447) — the ungated tail read cannot condition-and-select by itself.
- **Oracle-gated layouts (the empty quadrant; pre-registered prediction "gated qlast ≈ gated replica", STATE 09-15 02:00 — MET within a row):** replica 1.000/0.953/0.753/0.540, qfirst 1.000/0.913/0.700/0.600, qlast 0.967/0.900/0.687/0.527 @8–64 (N=128, 100/cell: 0.22/0.46/0.54). With the gate carrying selection, question-blind per-frame encodings suffice in-window (0.967/0.900 vs 0.447/0.380): conditioning and selection are one job, done inside each block (replica) or at the read (gate). All three N-prompt gated cells hit the snap-to-declared-N wall from N=32 (mse +3.3 @32, +13.8 @64, +38…+79 @128; e.g. qfirst-gated @64 k8 0/10 while k64 10/10) because no virtual-N was trained — read WITHIN a row, never against the S8/S9b headlines.
- **N-free twins:** gated replica 1.000/0.913/0.813/0.667/0.600, gated qlast 1.000/0.913/0.827/0.607/0.440; k≤8 band @32/64/128: replica 1.000/1.000/1.000 (94/94, 90/90, 54/54; k12 also perfect), qlast 1.000/0.989/0.815 (94/94, 89/90, 44/54) — blind blocks cost the k≤8 band only at 16× the training length.
- **Minimal-fix ladder (oracle gate, S8 recipe at 5 ep; reference S8 10-ep 23.8M: k≤8 1.000 every N, overall 0.853/0.733/0.607 @32/64/128):** A0 readout-only (N-free; lm_head digit rows 35,850 p) gold≤9 acc 0.773/0.661/0.623/0.622/0.617, band-1..8 0.744/0.618/0.575/0.575/0.569 (`a0_calib/*/results.csv`); its 10-param digit-bias arm is numerically identical to frozen (val 0.222 = frozen — fit did not move the readout, suspect). A1 (r1 q,v, L20–27, 90,112 p) 1.000/1.000/0.780/0.633/0.507, k≤8 1.000/1.000/0.991/0.975/0.802 (k8 0/9 @128, pf 0.107). A2 (r1 all 7 projections, 2.97M) 0.980/1.000/0.927/0.733/0.573, k≤8 **1.000 at every N** (80/80 @64, 81/81 @128), k12+k16 10/10 @64 = S8's @64 profile exactly. A3 (r8 q,k,v,o only, 5.05M) 1.000/0.973/0.820/0.713/0.600, k≤8 1.000 @64/128, k12 10/10 + 9/9, k16 0.
- **Pre-registered dichotomy did NOT resolve as written:** neither A0 nor A1 reaches k≤8 ≥0.95 @128 (so the LoRA is not a pure share→digit recalibration), but the "only MLP-bearing arms" branch is false too — attention-only A3 reaches it. Reading: the aggregation fix needs the attention path retrained across layers; ~3–5M params (5–8× below S8, half the epochs) suffice; the tiny arm's miss is a k8@128 read failure with parse-fails, not calibration.
- Caveats: 5-ep arms vs 10-ep references; layout arms lack virtual-N (long-N confounded with the snap); N=128 cells are separate eval-only reruns on 48 GB nodes, gated twins at 100/cell; A0 readout single-digit (gold ≤ 9) by construction; single task family, park data.

## [2026-09-14→16] ❌📊 SELFGATE — the answer-loss-only straight-through gate NEVER CLOSES: mean_bit 1.000 from ep1 at λ ∈ {0, 1e-3, 1e-2} (G1a exams 1.000/1.000/0.360 @N=8/16/32 = the ungated snap; G1b 0.927–1.000 / 0.273 / 0.160); root cause = hard-closed gates are gradient black holes (diag4 148538: gmask.grad 2.3, bits.grad exactly 0) plus no in-window incentive; G0a: the DEPLOYED S9b gated adapter has NO internal selector (best head 0.52–0.75 — oracle-gated training atrophied it) while P1b carries a population (≥10 heads ≥0.998 @N=128); G0c: the P1b selector population is QUESTION-GENERAL (1.000 under exists and majority at every N; L24h20 itself 0.9998–1.000 in 5/6 cells, 0.789 at majority@N=8)

> Instruments (now `legacy/v1/scripts/selfgate/`): `probe_headscan.py` (per-(layer,head) rank-AUC of answer-row block mass, ungated forwards; S10 anchor reproduced: P1b L24h20 0.9997/0.9998/1.0000 @N=8/32/128), `train_sft_selfgate.py` (two-forward step: L12 replica-slot linear gate head → ST Gumbel-sigmoid bits, τ 5→0.5 → mask held through backward, grad pass on the MATH sdpa kernel), `diag_grad_selfgate.py` / `diag_maskgrad_micro.py` (H-SAFETY A/B/C gradient checks). Regime: park data (legacy generator; seq8 + longN16 roots, exam dirs excluded), count task, N-free prompt; substrates `checkpoints/sft_fenced_gated_vn_nfree_adapter` (S9b) and `sft_fenced_le16_ep10_adapter` (P1b). Runs: `outputs/selfgate/g0a/{p1b,s9b}_N{8,32,128}/` (148381; 30/k × k∈{2,4,8}), `g0c/p1b_{exists,majority}_N{8,32,128}/` (148593), `g1a/20260914_145650_selfgate/` (148575, 3h29, rtx6k n318 96 GB; the `20260914_143253_selfgate/` dir = crashed attempt 148543), `g1b_l1e-3/20260915_201150_selfgate/` (150418), `g1b_l1e-2/20260915_201205_selfgate/` (150419); diag chain `outputs/_scratch/selfgate_{diag,diag2,…,diag6,micro,g1a_smoke,g1a_smoke2}/` + `logs/sg_*.out` (148493/148501/148505/148522/148538/148542/148554/148563/148574). Brief `outputs/selfgate/CAMPAIGN_BRIEF.md`, log `STATE.md`.

- **H-G0 clause 1 MISSED as pre-registered:** no head ≥0.98 in the deployed gated adapter at any N (best per layer 0.52–0.75; @128 best 0.586, `g0a/s9b_N*/report.txt`). The oracle mask did the rejecting during training, so evidence separation is gone from the ungated attention geometry — the oracle gate is a crutch. P1b (`g0a/p1b_N*/`): L24 h6/h17/h20 ≥0.9997 @N=8; L20 h1/h3/h15 + L24 h4/h5/h6/h17/h20 ≥0.998 @N=32/128. G0b (θ-threshold self-gate) was never run (the share-law argument already predicts a fixed θ fails at N≥32; QGATE took over the deployable-gate question).
- **H-G0c (≥0.9) MET at the population level:** under exists and majority questions ≥7 heads per cell reach ≥0.98 and the best is 1.0000 at every N (`g0c/*/report.txt`); L24h20 = 0.9998/1.000/1.000 (exists) and 1.000/1.000 (majority @32/128) but 0.789 at majority@N=8 (`p1b_majority_N8/auc.json`), where L20h27, L24h21, L24h22 sit at 1.000 — STATE's "L24h20 ≥0.9998 in all six cells" is corrected here; the relevance signal is question-general, the single head is not.
- **H-SAFETY earned its keep (6 diag jobs, ~19 min GPU, before any training):** EFFICIENT sdpa cannot differentiate w.r.t. attn_mask at all (hard error, micro 148505) → MATH kernel on the grad pass; then diag4: with bits all-0, gmask.grad = 2.296 but bits.grad = 0.000 — a hard-closed block's softmax weight is exactly 0 and ∂loss/∂mask ∝ that weight, so the REOPEN direction never gets signal (closed-forever). Amendment 1: multiplicative soft train mask log(clamp(p,1e-6)), hard bits at eval (diag5 148542 PASS: gate cos(B,C) 1.000000, open/closed grads 3.2e-1/1.8e-1). Then gradient checkpointing × grad-carrying hook mask crashes in BOTH reentrant and non-reentrant modes (148543, 148563) and GC-off all-layer training OOMs @N=16 (~100 GB, 148554) → Amendment 2: LoRA on decoder layers ≥12 only (11.53M p), GC off.
- **H-G1a MISSED — degenerate-open (the pre-registered G1b trigger, measured):** mean_bit 1.000 from ep1 (fn 0; fp = every non-evidence block 609/1547/3459 @N=8/16/32; gate_bit_acc = the evidence fraction 0.49/0.36/0.28), val 1.000 @ep7, TEST_IID 1.000, exams 1.000/1.000/0.360 with the ungated snap anatomy @32 (k0–k2 + k12/k32 perfect, k3–k8 0/12) vs 0.847 @32 for the same substrate under the oracle gate (S9b). G1b: mean_bit 1.000 at every epoch for both λ; N=8 0.927 (1e-3) / 1.000 (1e-2) but N=16 0.273 and N=32 0.160 in both — the L1-dampened soft train mask creates a train/eval mismatch when the gate never commits.
- **Reading:** the answer loss is satisfiable in-window without gating (seq ≤16 is inside the ungated read's competence), so closing never pays; the out-window collapse re-demonstrates the campaign's theorem on its own adapter. The incentive fix (virtual-N pressure during gate training) is deferred (QGATE C3, unrun); G1-multi never ran.
- Caveats: gate head reads the L12 slot only; G1 arms carry two forced recipe deviations (≥L12 LoRA scope, soft train mask) so oracle-parity comparisons are approximate; G1 exams N ≤ 32 only; single task, park data.

## [2026-09-15→16] ❌📊 QGATE — a question-agnostic gate over question-BLIND (qlast) block encodings is DEAD at the linear level: the "internal selector" is a VERDICT-READER (under qlast no head separates evidence — best AUC 0.53–0.68 in the qlast-gated adapter, 0.56–0.79 in P1b, vs 1.000 with replicas; neutral fillers in the replica slots collapse P1b to 0.62–0.78, D1b), question-blind block states do not linearly expose occupancy (D3 = 0.70–0.75 AUC with a PERFECT symbolic (C,R) selector; B1 model-q_vec gate = chance 0.52; A2 state-only = base rate 0.802/0.519, passes as the control), a cacheable generic-filler slot exposes no more (E1: D3 0.767/0.748, B1 0.802/0.520), and the O1 neutral-replica trainer produced NO trained adapter (builder kwarg mismatch skipped all 651 train + 600 exam samples)

> Instruments (now `legacy/v1/scripts/qgate/`): `qgate_common.build_task_messages_layout` (replica path byte-identical to the sparse builder, CPU-verified for count/exists/majority), `probe_headscan.py --layout` (selfgate), `capture_qb.py` + `fit_qb.py` (span-mean/last/max block states at L12/L20, chat-templated q_vecs, ALT-(C,R) relabelings; CPU LR/bilinear fits), `train_neutral.py` (thin launcher over the sparse trainer). Regime: park data (legacy generator; seq8 + longN16 roots, 150 dirs/root for captures, exam dirs excluded), count/majority/exists templates, declared-N prompt (qlast-gated cells scanned N-free per their contract). Substrates: `outputs/sparse/layout/qlast_once_ep5_gated/20260915_203606_gated/adapter` (150433) and P1b. Runs: `outputs/qgate/a1/{p1b,qlastg}_{count,majority}_N{8,32,128}/` (150577 / 150719; 90 samples/cell), `d1/p1b_neutral_N{8,32}/` (150841; v1 150827 died on a tokenizer boundary merge), `bcap/` (150720, UNUSABLE: 0 q_vecs), `bcap2/fit_report.json` (150826), `bcap3/fit_report.json` (150848, pooling sweep), `e1cap/fit_report.json` (150851, layout=replica-neutral), `o1_train/20260916_101845_gated/` (150856). Brief + STATE in `outputs/qgate/`.

- **H-QA1 MISSED decisively:** under the qlast layout no head reaches 0.98 in any of 12 cells — qlast-gated best per cell 0.63/0.55/0.54 (count) and 0.68/0.59/0.53 (majority) @N=8/32/128; P1b-under-qlast 0.79/0.68/0.63 and 0.76/0.60/0.56 (`a1/*/report.txt`). L24h20 reads the replica-borne VERDICT computed in-block by the question copy, not frame content. **D1b proves it causally:** replica positions kept, question text replaced by a fixed filler → P1b's best head 0.78 @N=8 / 0.62 @N=32 (`d1/p1b_neutral_N*/report.txt`; L24h20 off the board). B2 (label-free internal-head threshold) is therefore dead on qlast and was not run.
- **H-QA2 MET as the negative control:** block state alone → evidence bit = base rate exactly (acc 0.802, AUC 0.519; `bcap2/fit_report.json`). **D3:** a bilinear probe with the PERFECT symbolic (C,R) selector reads occupancy from the blind L20 states at only 0.766/0.747 (acc/AUC); mean/last/max pooling (`bcap3`) 0.733/0.697 — not gate-grade on any channel. **H-QB1 MISSED catastrophically:** the (q_state, b_state, q⊙b) gate = majority on held-out dirs (0.802/0.520, n=10,464) and chance on LOO-majority (0.204/0.523, n=3,488). The bottleneck is the STATE side, not the question vector.
- **E1 / Option 1 (cacheable generic-filler slot "What is shown in this frame?"):** the slot states expose no more than raw blind states — A2 0.802/0.519, D3 0.767/0.748, B1 held-out 0.802/0.520, LOO 0.204/0.527 (`e1cap/fit_report.json`) vs the ≥~0.99 that would have kept Option 1 alive. Verdict: a cacheable question-agnostic slot gate is dead at the linear level.
- **O1 neutral-replica trainer (150856) = instrument failure, no result:** `train_neutral.py`'s rebound builder does not accept the trainer's `declare_n` kwarg; every train sample was skipped in all 5 epochs (3,255 skips = 651 × 5), val 0/139, TEST_IID 0/140, all four exams n=0/skipped=150 (`o1_train/*/longn_eval.csv`); the saved adapter is the untrained init. Not logged in STATE; recorded here so the cell is not mistaken for a null RESULT.
- **Synthesis:** the frame facts ARE in the blind blocks (oracle-gated qlast reads 0.967/0.900 in-window, `outputs/sparse/layout/qlast_once_ep5_gated/`) but only the model's own question-conditioned compute extracts them — the cacheable-blind-encoding premise fails at EXTRACTION, not selection. C2 (deployed B1/B2 ladder) and C3 (virtual-N learned gate on qlast) never ran. Routes left: a micro cross-attention reader over cached blocks, or replica replay over cached KV (the proven fallback). **Rewrite note (2026-09-21):** the variant NOT tested here is question-FIRST (question in the shared prefix, so blocks are question-conditioned without replicas) with the `<|vision_end|>` slot — that is the first experiment of the rewrite, not a known result.
- Caveats: linear/bilinear probes only (a nonlinear reader is untested); captures at L12/L20 span statistics; 150 dirs/root, single task family, park data; A1 at 90 samples/cell.

## [2026-09-20→21] ✅⚠📊 FIXEDK — FLIP RESPONSE AT FIXED BASE COUNT, N = 2…128 (30 matched pairs/cell, 60 cells): the GATED read is FLAT from N=2 to 128 (0→1: 74.2→74.6, |slope| ≤ 0.03 at every stratum), the frozen read decays 34→1.3 (slopes 0.76–0.88), the trained P1b read decays 97→14 (slopes 0.44–0.62, ×2–12 the frozen amplitude), the fenced per-frame fact is flat 36–41 — the 2026-09-19 "gated read falls 72→28 over N=2..8" was k-composition, not N

> Instruments: `scripts/armor/probe_hahn.py --arms plain,fenced` (frozen read + fenced fact) and `--arms p1fence --peft-adapter checkpoints/sft_fenced_le16_ep10_adapter` (trained, P1b); `scripts/sparse/probe_hahn_gated.py --arms p1fence --gate oracle --nfree-prompt --peft-adapter checkpoints/sft_fenced_gated_nfree_adapter` (gated, S9 = the S11 endpoint; both scripts now under `legacy/v1/scripts/`). All cells `--gold-set g --limit 30 --controls 12 --seed 0`; ‖Δh‖ at L20, answer row for reads, rep_t for the fact. Runs `outputs/fixedk/{frozen,trained,gated}_N{2..128}_g{0,1,2}/`: N≤8 cells of 153850 (frozen) / 153851 (trained) (both CANCELLED at their N=16 cell, see review fix), 153869 frozen N16–128, 153870 trained N16–128, 153871 gated N2–128, 153872/153873 gold-2 top-ups N4/8 (a100-public). Medians → `outputs/fixedk/fig/fixedk_medians.csv` → F10 panel b′. Regime: park data (legacy generator, 100/class) for N ≤ 8 / mmred_redux (legacy balanced generator, 50/class at k=0,1,2) for N ≥ 16; frozen + trained arms under the frames-first count prompt with N in text, gated arm under the N-free prompt.

- **H-F1 MET (gated flat, every stratum):** 0→1 = 74.2/74.6/75.0/75.4/74.8/74.7/74.6 (N=2/4/8/16/32/64/128; slope −0.00); 1→2 = 41.0→46.0 (−0.03); 2→3 = 38.0→37.0 (0.01). The gated read moves < 3% over a 64× range of N; the flat line of S11 (pooled) is now flat at fixed k as well.
- **H-F3 MET (fenced fact flat):** rep_t medians 39.6→35.8 (0.04) / 39.5→35.9 (0.02) / 41.4→35.7 (0.05) — the per-frame fact is written N-invariantly; only the read dilutes.
- **H-F2 frozen MET:** 34.4/28.6/17.2/4.2/2.6/2.0/1.3 (slope 0.88), 23.9→1.4 (0.76), 17.2→1.3 (0.80) — at the bf16 floor from N=32. **Trained (P1b): amplitude clause MET** — 97.0/67.7/52.2/40.4/30.2/23.5/14.0 is ×2.4–3 the frozen response at N≤8 and ×10–12 at N≥16 — **but the α ≥ 0.5 clause MISSED at 0→1 (0.44)**; met at 1→2 (0.62; 145.5→8.1, the N=2 cell is an outlier where one flip is half the context) and 2→3 (0.58; 51.2→6.2). Per-gold exponents 0.44–0.62 sit BELOW the pooled 0.80 [0.66,0.97] and below the frozen read's — consistent with the share law when training raises the edge eˢ slightly, NOT with a removed N-dependence: the trained response still falls ×7 from N=2 to 128 where the gated read moves < 3% (STATE 2026-09-21).
- **Review-panel fix on record (STATE 2026-09-20 21:34):** the brief's N≥16 pools (`data/mmred_longN_park/seq_len_N`) hold only 30 dirs/gold and seq_len_16 is a TRAINING root of P1b and S9/S9b → jobs 153850/153851/153864 cancelled at N=16, every N≥16 cell re-run on the fresh `data/mmred_redux/seq_len_N` pools (153869–153873); gated arm switched to S9 so the campaign shares S11's endpoint. N≤8 cells were kept on park pools (note: park seq8 is also a P1b/S9 training root; these are state-geometry cells, no decode accuracy is claimed from them).
- Caveats: per-gold protocol ≠ the paper's pooled-α protocol — do not refit the headline α from these cells (both reported); no gold-2 cell at N=2; slopes are point fits (no bootstrap CI in STATE); single task family/model; L16/L28 loci in the report.txt files, not read here.

## [2026-09-20→21] ✅❌📊 SOFTGATE — THE GATED READER IS A CONTENT-BLIND BLOCK COUNTER: with a finite gate penalty (−log B on non-evidence keys) the read's exponent slides with B — α = 0.66/0.17/0.10/0.07/0.03 for B = 2/8/32/128/∞ (law 0.48/0.15/0.03/0.01/0) — with the gate removed (B=1) the S9b answer row does NOT respond to a content flip at all (2.2 = bf16 floor at every N), the no-retrain answer is k + (N−k)/B TO THE UNIT at N=32 and 128, and retraining under B=8/128 fails by CALIBRATION to the residual mass (the B=128 reader is 107/107 under the hard gate), not capacity — no finite penalty extrapolates; only a residual of exactly zero is N-invariant

> Instrument: one flag `--gate-bonus B` on `scripts/sparse/probe_hahn_gated.py` and `train_sft_gated.py` (now `legacy/v1/scripts/sparse/`; B=0 = hard gate, byte-identical path; smoke 153852: probe anchor BYTE-IDENTICAL, B=1 ≡ no-gate at 0.000e+00, trainer anchor P1b exam_ff_N8 = 1.0000). Reader S9b `checkpoints/sft_fenced_gated_vn_nfree_adapter`, N-free prompt, oracle soft gate. A1 chains 153859–153863 + hard chain 153874 → `outputs/softgate/a1_B{1,2,8,32,128,0}_N{8..128}/` (S11 protocol: 50 pairs, 40 @128, ctrl 12, seed 0); F12 `outputs/softgate/fig/F12_alpha_vs_B.{png,csv}` (`scripts/softgate/fig_alpha.py`, now under legacy/v1; α fits also in `a1_alpha_L20final.json`). A2 153865 (TIMEOUT at 5 h after B=32 @N=32) + refills 154064/154069 → `a2_B{2,8,32,128}/`, `a2_B32_N128/`. A3 trainers 153866/67/68 → `a3_B8/`, `a3_B128/`, `a3_Bhard/` (S9 recipe: nfree, oracle gate, 5 ep, 728 samples, roots seq8+seq16, 24 excludes); cross-gate evals 154015 → `a3eval_{hard_at_B8,hard_at_B128,B8_at_hard,B128_at_hard,B8_at_B128}_N32/`; own-gate ladders 154014 → `a3eval_{B8_at_B8,B128_at_B128,hard_at_hard}_N64_128/`. Regime: park data (legacy generator), exams `outputs/loramech/examdirs/exam_ff_N*` (150/cell), N-free prompt.

- **H-A1 monotone slide MET for B ∈ {2, 8, 32, 128, ∞}** (0.66 [0.64,0.67] > 0.17 [0.13,0.22] > 0.10 [0.06,0.16] > 0.07 [0.00,0.15] > 0.03 [−0.06,0.12]; CIs of 2/8/128 disjoint); the hard chain reproduces S11's 0.035 on a fresh S9b chain. **Clause α(B=1) ≥ 0.5 NOT MET, for an informative reason:** medians 2.25/2.07/2.24/2.15/2.34 at N=8..128 (vs 10.4 frozen, 26 P1b at N=8) while the same frame's in-block locus responds normally (rep_t ≈ 50) — the gate-trained reader reads "how many blocks reach me", not what they say (selector atrophy measured at the reader). B=2 is not a power law (12.7→10.1 flat over 8..32, then the floor at 64/128; the 0.66 is the floor's doing). Law refit C = 67 (RMSE 0.27): the exponent is right, the amplitude bookkeeping is not.
- **A2 (no retrain) — H-A2's ε-bookkeeping REPLACED by a bias law, pred = k + (N−k)/B with E ≈ 1:** B=8 @32 → k+4 exactly (0.073 exact, k≤8 0/107); @128 → k+16 saturating at the vocab cap "16" (41% out of vocabulary); B=32 @128 → k+4 (exact at 8 of 9 k); B=128 @32 → **0.833 exact, k≤8 107/107** (residual 0.25 rounds away), @128 → k+1 (9/9 k). Residual competitor mass is ADDED to the count as block-equivalents (24 blocks at weight 1/8 = 3 frames); the error is linear in N, not a per-frame rate.
- **A3 (retrained, own gate), N = 8/16/32/64/128:** B=8 0.973/0.860/0.147/0.020/0.020 (k≤8 0/107 @32; signed +2.0/+10/+15); B=128 1.000/0.913/0.633/0.493/0.067 (0.5-block residual tips 16 answers by one @64; k+1 everywhere @128); hard 5-ep twin 1.000/0.933/0.767/0.693/0.547 (k≤8 90/90 @64, 76/81 @128, misses = digit run-ons "72"/"82"/"128"). H-A3's "B=8 passes at 2×" REFUTED at N=32 already — both soft arms OVERcount (mse +2.0).
- **Cross-gate at N=32 (154015) = the two-sided H-A3 verdict: CALIBRATION, not capacity.** B=128 reader under the HARD gate 0.707 with **k≤8 107/107** (own gate 0.633 = 12 residual-rounding errors); hard reader under B=8 0.120 = k+4 exactly for k≤5 (a magnitude sum whatever it was trained on); B=8 reader under the hard gate 0.100, pred ≈ 2k (in-window the residual (N−k)/8 ≤ 2 co-varied with N and was learned as the N-proxy — the S7 prompt-N channel in a new guise).
- Paper use: F12 + the A2 prediction table = the "magnitude read" exhibit — the gated reader counts block-equivalents of attention mass and only the hard gate makes that quantity N-invariant. Caveats: A1/A2/A3 on S9b while S11 ran S9 (named per caption); single seed, 5-ep/728-sample A3 readers with the recurring k=4 wobble (a3_Bhard k4 1/12 @32); `a2_B32` @N=32 predictions lost to the timeout (run.log only: mae 1.55, k16 11/11); A4 (photograph under bonus) not run.

## [2026-09-20→21] ✅⚠📊 REDUX v3 — THE REDUCTION-TYPE LAW, GATED vs UNGATED (multitask count/exists/majority readers, N-free prompt): EXISTS is 1.000 for BOTH readers at every N ∈ {8..128} (OR does not dilute once fenced + trained — H-R5 MET, H-R2's decay never appears); MAJORITY is N-flat at matched ratio for the ungated reader (0.960/0.944/0.883/0.854/0.875 at N=8..128, all errors within a fixed FRACTION of N/2 on the yes side) and breaks to chance under the hard gate at N ≥ 32 (0.373/0.431/0.563 — the gate deletes the denominator, H-R6 MET); COUNT bands scored on the count-only controls (N ≤ 32 only): gated k≤8 exact 124/124 @16, 106/106 @32 vs ungated 0.65/0.38 — the multitask count rows are UNDER-TRAINED (k+1 at N=8) and H-R1/H-R4 at N=64/128 are UNSCORED

> Trainer `scripts/sparse/train_sft_gated.py --tasks count,exists,majority --nfree-prompt --gate {oracle|none}` (labels `scripts/redux/tasks.py`; both now under `legacy/v1/scripts/`), 5 ep, roots seq8+seq16 (728 dirs after the 24-file exclusion; 243/243/242 per task), in-job LONGN on `outputs/redux/examdirs/exam_{count,exists,majority}_N*.txt` (150/100/144–150 per cell; yes/no cells balanced, majority baseline 0.500). Runs: C2b_mt gated 154067 → `outputs/redux/c2b_gated_mt/20260921_024331_gated/`, C2a_mt ungated 154068 → `c2a_ungated_mt/20260921_024331_gated/` (rtx6k n318, 1h59 each); N=64/128 eval-only 154212/154213 → `c2b_gated_mt_eval64_128/20260921_112301_gated/`, `c2a_ungated_mt_eval64_128/20260921_112302_gated/` — **COMPLETED 15:02/15:09 on 09-21, all 12 cells fully scored (n = 150/100/144, no n=0 cells), NOT yet read into STATE (numbers below are from their longn_eval/longn_predictions.csv)**; count-only matched-budget controls (480 samples) 153963/153964 → `c2b_countctrl/20260921_001408_gated/`, `c2a_countctrl/20260921_001506_gated/`; mis-split count-only N-in-text pair 153961/153962 → `c2b_gated/20260920_232350_gated/`, `c2a_ungated/20260921_001236_gated/`. Pools: `data/mmred_redux/seq_len_{16,32,64,128}` (C0 datagen 138933–36 → `outputs/redux/datagen_N*/`; 550–1000 exam + 80 probe-base dirs per N, engineered K), N=8 exams from park seq8. Regime: park data (legacy generator) training roots + mmred_redux (legacy balanced generator) exams at N ≥ 16; N-free prompt (primary); the N-in-text pair is count-only and labeled. Earlier phases C0/C1/C4b (2026-08-30/31: frozen plain exists 0.833→0.490 = chance @128, majority at chance ∀N, fenced-frozen degenerate always-no, H-SHARP(a) refuted α 0.605 vs 0.603 under verified logN) are STATE-only.

- **exists — H-R5 MET at every N, H-R2's decay clause NOT observed:** both readers 1.000 at N=8/16/32/64/128 (gno 50/50, gyes 50/50 at 64/128; k=1 ⊂ the yes half). The two-sided branch fires: under the fence + trained reader the OR read holds to 16× UNGATED, where the frozen plain model (C1, 138954/55) had drifted to always-no (k=1 recall 0/16 @128). Caveats: training labels ~89% yes; the gated reader gets exists free (any visible block → yes).
- **majority, ungated — H-R3 MET at matched ratio r ∈ {0.25, 0.375, 0.625, 0.75}: 0.960/0.944/0.883/0.854/0.875** (N=8/16/32/64/128; 100/72/60/48/48 items) — flat within 0.1 and ≥ 0.8 through 128. Pooled cells 0.973/0.868/0.780/0.708/0.674 fall only because the yes-side boundary band WIDENS with N: gno 72/72 at every N ≥ 16; misses at N=16 d=+1,+2 (k9 3/18, k10 14/18); N=32 d ≤ +4 (k17 2/15, k18 2/15, k20 8/15); N=64 d ≤ +8 (k33/34/36 1/36, k40 5/12); N=128 d ≤ +16 (k65–72 7/48, k80 6/12); r=0.75 is 12/12 at every N. Second clause (fixed |d| decays with N) MET: k=N/2+1 → 3/18, 2/15, 0/12, 0/12. Reading: the ungated reader resolves the RATIO to a fixed fraction of N (a "yes" needs r ≳ 0.7) — Weber-like, as the share-is-the-quantity prediction requires; it is biased conservative (never a false yes).
- **majority, gated — H-R6 MET at N ≥ 32:** 0.373/0.431/0.563 at N=32/64/128 with |d|≤8 strata 0.342/0.500/0.521 (band ≤ 0.65); gyes 2/72 @64, 14/72 @128 — the hard-gated reader sees k blocks and no N and answers "no" to nearly everything, while in-window it was 0.960/0.861. The gate that makes COUNT N-invariant destroys the RATIO read: the reduction type decides.
- **count — bands scored on the count-only controls (N ≤ 32):** H-R4 gated k≤8 124/124 @16, 106/106 @32 (0.980/0.913/0.767 overall); H-R1 ungated k≤8 81/124 = 0.65 @16, 40/106 = 0.38 @32 (0.973/0.713/0.267) — already below the ≥0.9 @16 clause. The multitask readers did NOT reach in-length competence (gated_mt 0.527/0.420/0.247 with pred = k+1 for k=4..7 at N=8, 17/17 at each k; ungated_mt 0.840/0.740/0.173): a budget/interference artifact (243 count samples vs 480; the majority label shares the answer row), not a mechanism finding. Their N=64/128 count cells (gated 0.233/0.213, signed +17.7/+38.7; ungated 0.127/0.120) are descriptive only — **H-R1/H-R4 at N=64/128 UNSCORED (no control cells there)**.
- Mishap on record: the first C2 pair (153961/153962) ran count-only with N in text — `EXTRA="--tasks count,exists,majority --nfree-prompt"` comma-split by sbatch (the CLAUDE.md footgun) — kept as a labeled side result: gated 0.993/0.887/0.780, ungated 1.000/0.893/0.567; the ungated N-in-text reader beats its N-free twin at N=32 (0.567 vs 0.267 — the S7 prompt-N channel) and its zero-shot majority is 0.980/0.965/0.927 with boundary-only errors. Wrapper gained space-separated `TASKS`/`NFREE` knobs; submitted command lines now grep-verified.
- Still open: Tal's reading of the N=64/128 cells into STATE; C4 per-task flip chains DEFERRED (probe_hahn_gated has no `--task`); an all-task-labelled sampler (3 × 728) or more epochs to fix the count rows (proposed, not run); single seed, 5 ep; N=8 exists/majority exams draw from park seq8 (a training root — overlap recorded, exam files excluded from training); recurring k=4 wobble in 5-ep hard readers.

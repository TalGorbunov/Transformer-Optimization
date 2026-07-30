# DRAFT results — the single-forward supply fixes (oneforward brief, 2026-07-15)

> Draft RESULTS.md-style entries per `plans/oneforward_agent_brief.md`. **Not** merged into
> RESULTS.md. Every number traces to a run dir listed with it. Protocol for all: steps task,
> `data/mmred_images_park/seq_len_8/all_uniform`, 392px, Qwen2.5-VL-7B 4-bit frozen, held-out
> shrinkage-LDA d′ (`dprime_pair`), layers 14/16 (probes) or L16 (capture-based).

## [2026-07-15] Exp A — replicas + FULL frame fencing (multipass-in-one-forward)

- **Script:** `experiments/glstm/replica_carrier_probe.py --fence-frames` (new flag): on top of the
  masked-replica 4D mask, each frame's visual-token rows are forbidden from all OTHER frames'
  visual tokens → every (frame + its replica) block is an isolated forward inside one sequence.
- **Smoke (n=8, job 121917):** `outputs/_scratch/rep_fence_smoke/20260715_201140/` — end-to-end OK,
  d′ nan by design at n=8. Mask sanity (seq=1693): frame0-row 16 allowed keys, frameLast-row 30
  (prefix + inter-frame markers + self — no other frames' visual content), replica rows 213/227
  (prefix + own frame 196 + own span), final-q row 1604 (everything minus 8×~11 replica tokens).
- **Full (n=300, job 121925):** `outputs/ladder/image_longN/replica_fence/20260715_203619/`.

| replica read (n=300, same samples) | L14 d′ | L16 d′ | L16 per-copy (0..7) |
|---|---|---|---|
| masked replicas ([2026-07-14] baseline) | 1.52 ± 0.16 | 2.52 ± 0.11 | 3.70 2.33 1.75 2.01 1.99 1.82 1.99 2.18 |
| unmasked replicas ([2026-07-14] baseline) | 2.72 ± 0.11 | 3.56 ± 0.14 | 3.73 4.00 2.88 2.65 3.52 2.68 2.59 2.97 |
| **fenced (this run)** | 2.45 ± 0.10 | **4.07 ± 0.08** | 3.81 3.26 1.97 2.20 2.30 2.01 2.37 2.85 |

  (In-run "JOINT anchor" column d′ ~0.1 = the known-invalid off−9 anchor under the interleaved
  template; compare against the external joint anchor 1.97–2.09 instead.)
- **Pre-registered bands:** GO ≥ 5.5 (expected 6.3–7.3 = pad/mp anchor band); 3–5.5 partial
  (position/RoPE effects); ≤3 = suspect mask bug (mask-debug above argues against).
- **Reading: PARTIAL (4.07, mid-band).** Fencing beats both replica baselines at L16
  (2.52 masked / 3.56 unmasked → 4.07) — best one-forward supply measured so far — but lands far
  below the 6.3–7.3 "multipass in one forward" expectation, and per-copy d′ is NOT flat
  (3.81 → ~2.0–2.9 for later frames; frame 0 is still special). Since fencing makes each
  (frame+replica) block computationally isolated in *content*, the residual per-copy ladder and
  the ~2.3-point shortfall vs true multipass point at the only thing fencing cannot remove:
  RoPE/position — later frames sit at large position offsets (block at ~pos 1400 vs ~15 in a
  real isolated forward). This cleanly identifies the long-context/position term as the next
  wall for one-forward supply, exactly the 3–5.5 band's pre-registered interpretation.
  Curious coincidence to note: 4.07 ≈ the priced clean-q × joint-v cell (3.97–4.47).

## [2026-07-15] Exp B — composition: replica queries × un-mixed values (offline anchor 6.17)

- **B1 — un-mixer retrained + weights saved (job 121919, CPU):**
  run `outputs/ladder/image_longN/unmixer_saved/20260715_194450/`, weights
  `outputs/ladder/image_longN/unmixer_saved/weights/unmixer_L16.pt` (g_k, g_v MLP 512→1024→512,
  trained per `encoding_unmixer.py` recipe on the n500 capture
  `outputs/ladder/image_longN/qkv_2x2/20260712_n500/`, split seed 0, train 300).

  | condition (mp query × …) | L16 d′ (eval n=200) |
  |---|---|
  | joint k/v | 3.82 ± 0.07 |
  | **un-mixed k/v** | **5.94 ± 0.13** |
  | mp k/v (ceiling) | 6.33 ± 0.09 |

  Un-mixer recovers **84%** of the joint→clean encoding gap (prior run reported 93%/6.17 — retrain
  variance; direction identical: encoding damage is RECOVERABLE by a learned per-token map).
- **B2 — deployed composition (unmasked replica forward + g_k/g_v applied to frame-token
  k_proj/v_proj outputs at L16 via forward hooks, pre-rotary):** implemented as
  `replica_carrier_probe.py --no-mask --unmix-dir …` (hooks registered before the qkv capture
  hooks, so the message recompute reads the un-mixed k/v and the live forward deploys them).
  Smoke n=8 job 121926 (`outputs/_scratch/rep_unmix_smoke/`, `[unmix] loaded` + variant
  `unmasked+unmix_L16` verified); full n=300 job 121928:
  `outputs/ladder/image_longN/replica_unmix/20260715_204158/`.

  | replica read (n=300, same samples) | L14 d′ | L16 d′ | L16 per-copy (0..7) |
  |---|---|---|---|
  | unmasked replicas ([2026-07-14] baseline) | 2.72 ± 0.11 | 3.56 ± 0.14 | 3.73 4.00 2.88 2.65 3.52 2.68 2.59 2.97 |
  | **unmasked + un-mix L16 (this run)** | 2.70 ± 0.09 | **1.44 ± 0.07** | **4.17** 1.60 1.00 1.46 1.22 1.43 1.49 1.23 |

- **Reading: NO TRANSFER — actively destructive (1.44, below even the ≤3.56 negative band).**
  The un-mixer corrupts every frame's L16 values EXCEPT frame 0, which improves (3.73 → 4.17,
  the best single-copy number in any replica arm). The mechanism is exactly the pre-registered
  transfer risk: in the unmasked replica layout, later frames attend the preceding question
  replicas, so their joint k/v are question-conditioned — off the un-mixer's standard-layout
  training distribution — while frame 0 (no preceding replicas) matches the training
  distribution and gains. Internal control: L14 (below the edit) is unchanged (2.70 vs 2.72),
  so the collapse is attributable to the L16 k/v edit alone. The offline anchor (5.94 with the
  mp query on standard-layout k/v) therefore does NOT price the deployable composition; a
  replica-layout-trained un-mixer (retrain g_k/g_v on a replica-prompt capture — frame-0's +0.44
  shows in-distribution un-mixing helps even deployed) is the obvious next rung.

## [2026-07-15] Exp C — CoGNN-style broadcast gate (content-side routing repair)

- **Script:** `experiments/glstm/broadcast_gate_probe.py` (new; differentiable message path of
  `trained_query_ceiling.py`). Per-token ADDITIVE logit offset b_j = MLP(features) added to the
  within-frame attention under the contaminated JOINT query; zero-init last layer (ep 0 == joint
  anchor by construction); logistic proxy loss; split seed 0 train 300 / eval 200.
- **Run (job 121918, L16, 400 ep):** `outputs/ladder/image_longN/broadcast_gate/20260715_194451/`
  (`20260715_192911/` is an empty dir from a cancelled duplicate on n310 — NFS stall, resubmitted).
- **Anchor gate PASS (in-run):** joint-q 2.09 eval (want ~2.09), mp-q 3.82 eval (want ~3.82).

| arm (features) | eval d′ | eval trajectory max | full500 d′ |
|---|---|---|---|
| content ([k_j, v_j], 1024-d) | 1.80 ± 0.16 | 2.13 (ep 20) | 2.88 ± 0.38 |
| qcond ([k_j, v_j, q_pad], 4608-d) | 30.69 ± 4.01 (**INVALID — leak**) | 30.74 | 34.30 |

- **content arm — the honest number:** never beats the joint floor (2.09); the cherry-picked
  trajectory max (2.13, ep 20) is within noise of the floor, then monotonically decays to 1.80 as
  train loss → 0 (overfit). Per the pre-registered bands this is the "≈2.1 = routing is NOT
  repairable from content" outcome — the addressing story now closed from a third direction
  (trained shared query NO-GO; content-side gate NO-GO; only architectural frame identity works).
- **qcond arm — invalid by feature leak, do NOT cite as GO:** eval d′ 30.69 is ~4× beyond the
  physical mp×mp ceiling (6.3–7.3) and the AUC term saturates its cap (5.26) — impossible for a
  routing repair; the gate must be broadcasting label information carried by its q_pad input.
  q_pad is captured AFTER the question attends the frame in the pad forward, so it is
  evidence-bearing, and computing it at inference requires a per-frame forward (= multipass) —
  neither clean nor one-forward-deployable. The script's auto-VERDICT line ("GO") keys on
  max(arms) and is therefore wrong; superseded by this reading. Quantified by the q_pad leak
  probe (job 121927, `experiments/glstm/qpad_leak_probe.py`,
  `outputs/ladder/image_longN/broadcast_gate/qpad_leak/20260715_201750/`) — held-out LDA d′
  directly on the raw features:

  | feature | eval d′ | full d′ |
  |---|---|---|
  | q_pad (the qcond gate input) | **8.63 ± 0.20** | 10.06 |
  | q_mp | 10.32 ± 0.12 | 11.98 |
  | k_mean (joint, within-frame mean) | 0.66 ± 0.09 | 0.87 |
  | v_mean (joint, within-frame mean) | 0.65 ± 0.03 | 0.81 |

  One table, both conclusions: the per-frame clean queries are themselves evidence-bearing
  (d′ 8.6–12 ≥ the mp×mp message ceiling — the question read the frame before the query was
  captured), so any gate fed q_pad can broadcast the label without repairing anything; and the
  pooled content features carry almost nothing linearly (d′ ~0.65), so the content arm had
  nothing to exploit — its floor-level result is a genuine property of the k/v content at L16,
  not an optimization failure.
- **Pre-registered bands:** floor 2.09 · GO ≥ 3.0 · ≈2.1 = not repairable from content. Result:
  **≈2.1 outcome** (content arm), qcond arm excluded for leakage.

## [2026-07-17] Exp A2 — replicas + fencing + per-block POSITION RESET (PCW-style)

- **Script:** `experiments/glstm/replica_carrier_probe.py --fence-frames --reset-positions` (new
  flag): on top of full fencing, every (frame + replica) block's M-RoPE position ids are remapped
  onto block 0's range (`get_rope_index` → per-block constant shift; final question re-based to
  follow block 0). Legal because fenced blocks are mutually invisible (the Parallel-Context-Windows
  trick, Ratner et al. 2023). Each block is now content-isolated AND position-identical to an
  isolated forward.
- **Smoke (n=8, job 122739):** `outputs/_scratch/rep_posreset_smoke/` — `[pos-debug]` verified:
  block starts [14,40,…,196] → all 14, `blocks_identical=True`, max_pos 236→54, final-q 222→40;
  mask identical to the fence run (seq=1693, same allowed-key counts).
- **Full (n=300, job 122744, 19 min a100):**
  `outputs/ladder/image_longN/replica_posreset/20260717_181630/`.

| replica read (n=300, same samples/protocol) | L14 d′ | L16 d′ | L16 per-copy (0..7) |
|---|---|---|---|
| fenced ([2026-07-15] Exp A) | 2.45 ± 0.10 | 4.07 ± 0.08 | 3.81 3.26 1.97 2.20 2.30 2.01 2.37 2.85 |
| **fenced + posreset (this run)** | 2.68 ± 0.09 | **4.66 ± 0.05** | 3.70 3.57 3.17 3.00 3.09 2.30 2.84 3.49 |

- **Reading: PARTIAL, improved (4.66; position term confirmed real but not the whole residual).**
  +0.59 over fencing at L16 — new one-forward supply best — and the per-copy ladder largely
  FLATTENS (later copies 2.0–2.9 → 2.3–3.6; frame 0 unchanged 3.81→3.70, as predicted — it already
  sat at block-0 positions). So position explains the per-copy gradient but every copy now reads at
  ≈ frame-0's level (~3.0–3.7), NOT the mp band (6.3–7.3): the residual gap is whatever makes
  block 0 itself read below a true isolated forward, and it is shared by all copies. Suspects:
  template/locus mismatch vs the mp anchors (off−9 on a standard prompt vs word-matched replica
  room token mid-sequence), estimator scale (per-copy d′ is finite-sample-deflated at n=300 —
  cf. the 2×2's 4.79→7.81 at n=150→500), or a real remaining companionship effect. In-run JOINT
  anchor 0.06–0.09 = the known-invalid interleaved off−9 read; compare against external joint 1.97.
- **N=1 anchor control (job 122764, pending):** same script/flags/locus/estimator on
  `data/mmred_smallN_park/seq_len_1/all_uniform` n=300 — a truly-alone frame through identical
  machinery. If it reads ≈3.7 (≈ per-copy block 0), the remaining gap is protocol/estimator and
  the posreset architecture is already AT the isolated-forward bound as measured by this
  instrument; if it reads ≈6–7, a real one-forward residual remains.

## [2026-07-17] Exp A3 — full block-diagonal fence (marker leak sealed) — **GO 6.34**

- **The leak (found analyzing A2):** under `--fence-frames`, the inter-frame `vision_start`/
  `vision_end` MARKER tokens are neither visual nor replica tokens, so later blocks could still
  attend them — and each marker's residual is computed attending all earlier frames' content: a
  real cross-frame channel into every block except block 0 (matching A2's per-copy pattern,
  frame 0 highest).
- **Script:** `--fence-blocks` (new flag, on top of `--fence-frames --reset-positions`): every
  token of block i (markers + frame + replica) forbidden from every token of every other block —
  true block-diagonal attention. Mask debug confirms structural solo-equivalence: frameLast-row
  allowed keys 30 → 16 (= frame0-row), replicaLast-row 227 → 213 (= replica0-row).
- **Runs (a100):** A3 full n=300 job 122809 →
  `outputs/ladder/image_longN/replica_blockfence/20260717_190158/`. Anchors: N=1 solo through the
  SAME instrument — smallN data n=1200 job 122782 → `replica_posreset_N1anchor/20260717_184440/`
  (**6.01 ± 0.26**, AUC 5.13); images_park-matched n=200 job 122810 →
  `replica_posreset_N1anchor_parkimg/20260717_190153/` (4.55 ± 0.27 — n=200 estimator-deflated;
  smallN cache subsampled to n=300 reads 4.76, so matched); first N=1 attempt job 122764 n=300 was
  single-class NaN (sorted K0 dirs — use full balanced set).

| arm (n=300 unless noted, same protocol) | L14 d′ | L16 d′ | L16 per-copy (0..7) |
|---|---|---|---|
| A (fenced, [2026-07-15]) | 2.45 ± 0.10 | 4.07 ± 0.08 | 3.81 3.26 1.97 2.20 2.30 2.01 2.37 2.85 |
| A2 (+ posreset, [2026-07-17]) | 2.68 ± 0.09 | 4.66 ± 0.05 | 3.70 3.57 3.17 3.00 3.09 2.30 2.84 3.49 |
| **A3 (+ blockfence, this run)** | 3.41 ± 0.08 | **6.34 ± 0.11** | **3.70 3.97 3.84 3.76 4.56 3.70 4.09 3.64 — FLAT** |
| solo anchor (N=1, same instrument, n=1200) | 2.91 ± 0.02 | 6.01 ± 0.26 | — |

- **Reading: GO (pre-registered band ≥5.5 = "multipass in one forward confirmed").** Pooled L16
  6.34 lands IN the mp anchor band (6.3–7.3) and AT/above the same-instrument solo anchor (6.01);
  per-copy is flat with frame 0 no longer special (per-copy absolute values ~3.6–4.6 are
  n=300-deflated — the solo anchor itself reads 4.76 when subsampled to n=300; pooled is the valid
  scale). Same AUC-cap caveat as all mp-band numbers (rank estimator pegged at 5.26). The
  single-forward supply story is closed constructively: **question replicas + full block-diagonal
  fence + per-block M-RoPE reset = isolated-forward supply (d′ ≈ 6.3) at one-forward cost.** The
  three ingredients decompose the joint tax: replicas (query) +1.6, content fence (values) +0.5,
  position reset +0.6, marker seal +1.7 (interaction terms included in each step's measured gain).

## [2026-07-17] Exp A4 — Q-FIRST + blockfence + posreset, and gate→tally behavior

- **Q-first control (job 122888, n=300, `--question-first --fence-blocks --reset-positions`):**
  `outputs/ladder/image_longN/replica_blockfence_qfirst/20260717_195501/` — **L16 pooled d′
  9.24 ± 0.33** (AUC estimator pegged at its 5.26 cap; same caveat as all mp-band numbers),
  per-copy 5.66–7.58 (n=300 scale, where A3 reads 3.6–4.6). Question-first is not just layout
  prep for the learned carrier — it is a supply AMPLIFIER: the question sits in the shared
  prefix, frame tokens attend it during encoding, so the k/v arriving at each carrier are
  already task-conditioned. Consistent with the historical Q-first probe advantage (3.35 vs
  2.47), now compounded with clean blocks. In-run final-q off−9 anchor reads 4.40 (Q-first
  partially rescues even the deployed-locus read; secondary observation).
- **Gate→tally behavior (CPU, held-out logistic gate on the message caches, count = #positives,
  5 seeds):** A3 standard layout per-frame err 0.0050 → **exact-count 0.960 ± 0.013** @N=8 ·
  A4 Q-first per-frame err 0.0012 → **exact-count 0.991 ± 0.008** @N=8. (Vs frozen model 0.207,
  multipass tally 0.910, joint-sum law ceiling 0.314 — the one-forward architecture beats the
  ~N-forward pipeline it compiles.)
- **N-sweep (jobs 122889–92, N=16/32/64/128, standard layout blockfence+posreset,
  `data/mmred_longN_park`): SUPPLY IS FLAT TO N=128.** Runs
  `outputs/ladder/image_longN/replica_blockfence_N{16,32,64,128}/20260717_*/`:

  | N | n | L16 pooled d′ | per-copy range | joint ref (Table 1) |
  |---|---|---|---|---|
  | 8 | 300 | 6.34 ± 0.11 | 3.6–4.6 | 1.97 |
  | 16 | 300 | 7.62 ± 0.11 | 4.1–5.2 | 2.12 |
  | 32 | 300 | 7.81 ± 0.08 | 4.1–5.0 | 1.98 |
  | 64 | 200 | 7.55 ± 0.11 | 3.1–5.1 | 1.93 |
  | 128 | 100 | 7.24 ± 0.14 | 2.5–5.9 | 1.59 |

  (AUC estimator pegged at 5.26 throughout — same caveat as the multipass column it now matches:
  mp read 7.18–8.08 flat. The in-run "JOINT anchor" column is the known-invalid interleaved
  off−9 read; compare against Table 1's joint column.) Per-copy flat within every N; no OOM at
  N=128 (seq ≈ 28k, h200, 75 min total).
- **Gate→tally vs N (CPU, held-out logistic on the sweep caches, 50/50 split, 5 seeds) — the
  wall-ladder table, rewritten by one forward:**

  | N | per-frame err | one-forward exact | retrieve-then-verify | multipass tally | frozen |
  |---|---|---|---|---|---|
  | 8 | 0.0050 | **0.960 ± 0.013** | — | 0.910 | 0.207 |
  | 16 | 0.0015 | **0.976 ± 0.009** | — | 0.793 | 0.127 |
  | 32 | 0.0012 | **0.960 ± 0.008** | 0.862 | 0.680 | 0.053 |
  | 64 | 0.0009 | **0.952 ± 0.020** | 0.853 | 0.580 | 0.040 |
  | 128 | 0.0015 | **0.876 ± 0.056** | 0.791 | 0.420 | 0.013–0.020 |

  One forward + a logistic gate beats the ~N-forward retrieve-then-verify pipeline and the full
  multipass solution at every N — 44× the frozen model at N=128. (Gate trained per-N on half the
  cache; the reference columns are the RESULTS.md / artifact numbers. A Q-first behavioral sweep
  at N=32/128 is queued — its N=8 per-frame error 0.0012 predicts ~0.95 at N=128.)

## [2026-07-17] Exp A5 — Q-first at scale + the learned carrier token (both landed same evening)

- **Q-first long-N (jobs 123027/123028):**
  `outputs/ladder/image_longN/replica_blockfence_qfirst_N{32,128}/20260717_2105*/` — N=32 d′
  12.67 ± 0.38 (per-copy 6.7–8.6), N=128 d′ 11.57 ± 0.50 (per-copy 4.8–8.8). The amplifier holds
  (grows) at scale. **Gate→tally: N=32 per-frame err 0.0000 → exact 1.000 ± 0.000; N=128 err
  0.0001 → exact 0.984 ± 0.008** (50× frozen 0.020; retrieve-then-verify 0.791). MMRED steps
  counting is saturated at every tested N, in ONE forward.
- **Learned carrier token (jobs 122938 proxy / 122939 distill; script
  `experiments/glstm/carrier_token_distill.py`):**
  `outputs/ladder/image_longN/carrier_token/20260717_201919_{proxy,distill}_room_k1/` — ONE
  trainable embedding (3,584 params) after each frame replaces the ~20-token question replica;
  Q-first layout, blockfence+posreset; backbone frozen; in-run teacher anchor reproduces the
  independent probe exactly (9.24 ± 0.33 / eval-split 8.95 ± 0.46).

  | arm | eval d′ (held-out split) | full-n | notes |
  |---|---|---|---|
  | ep-0 warm start (mean room embedding, UNTRAINED) | 5.23 ± 0.04 | 6.22 | in-context does most of the work |
  | proxy (BCE on evidence label) | best 6.46 @ ep10, stable ~6.4 | ~7.4 | ≥ A3 replica level |
  | **distill (cosine to replica messages, label-free)** | **best 8.35 @ ep10, stable ~8.2** | **~9.0–9.2** | **93% of the scale-matched teacher (8.95)** |
  | trained-query floor (fixed vector AT L16, [2026-07-13]) | 0.36–0.51 | — | the controlled contrast |

  Reading: the same-sized learned vector goes from 0.4 (injected at L16, context-free) to 8.35
  (inserted at the input, query computed in-context) — a ~20× controlled demonstration that
  per-source addressing must be in-context, and that it can be learned into ONE token.
  Token overhead drops 20/frame → 1/frame. The distill objective (no task labels) beats the
  supervised proxy — the task-agnostic form is the stronger one.

## [2026-07-18] Overnight — E4 adequacy on the new architecture's messages

CPU battery (held-out shrinkage-LDA projections, 3 seeds) on every new cache: **blockfence N=16
PASS clean** (|skew|≤.13, kurt≤.46, std-ratio 0.97); N=32 near-pass (kurtN 0.50); the high-supply
regimes (A3/A4 N=8, qfirst long-N, carrier distill) show the **saturation signature** already
documented in the verdict map for near-perfect probes — mild skew (−0.5..+0.5), kurt up to ~1.2,
stdE/N 1.1–1.5, d′_AUC pegged at its cap. Reading per the E4 protocol: in saturated regimes the
closed form is not licensed — quote the DIRECTLY MEASURED exact-count accuracies (which is what
all headline claims use); ordering claims unaffected. Nothing load-bearing rests on Φ(d′/2) here.

## [2026-07-18] Overnight — carrier LENGTH GENERALIZATION (train N=8 → deploy N=32/128)

- **Eval-only mode** (`carrier_token_distill.py --eval-only --carrier-ckpt …`, streaming, no RAM
  cache): the distill carrier (trained ONLY at N=8) evaluated zero-shot. Jobs 123128/123129 →
  `outputs/ladder/image_longN/carrier_token_lengen_N{32,128}/20260718_*/`.

  | N | carrier d′ (zero-shot) | replica teacher at this N | gate refit per-N: exact | gate trained @N=8, zero-shot: exact |
  |---|---|---|---|---|
  | 32 | 11.40 ± 0.30 | 12.67 | 1.000 ± 0.000 | **0.917** (err 0.0028) |
  | 128 | 9.71 ± 0.61 | 11.57 | 0.988 ± 0.024 | **0.860** (err 0.0023) |

- **Reading: the mechanism is length-agnostic.** One embedding trained at N=8 supplies
  near-teacher d′ at 16× its training length; the ENTIRE stack (carrier + logistic gate, both
  N=8-trained, zero new parameters at deployment) reads 0.92/0.86 exact at N=32/128 — above
  retrieve-then-verify (0.86/0.79) with ~N× less compute; a 30-sample per-N gate recalibration
  (the same supervision retrieve-then-verify used) recovers 1.000/0.988. Note: the in-run
  "ckpt-head zero-shot" rows in the job reports are garbage-by-design for the distill ckpt (its
  head never enters the distill loss — zeros); the meaningful zero-shot head test is the
  N=8-fit logistic above (CPU, caches `messages_best.npz` + `messages_eval.npz`).

## [2026-07-18] Overnight — carrier ablations (init and bandwidth)

Jobs 123124–26 → `outputs/ladder/image_longN/carrier_token/20260718_0058*_distill_{random_k1,room_k2,room_k4}/`
(distill objective, same protocol/teacher as the baseline arm; teacher anchor 9.24 reproduced in
all three):

| arm | ep-0 (untrained) eval d′ | BEST eval d′ |
|---|---|---|
| room-init k=1 (baseline, [2026-07-17]) | 5.23 | 8.35 @ ep10 |
| **random-init k=1** | 3.60 | **8.25 @ ep3** |
| room-init k=2 | 4.02 | 8.14 @ ep7 |
| room-init k=4 | 3.40 | 8.38 @ ep5 |

**Readings:** (1) init is irrelevant to the endpoint — random converges to the same ~8.3 (faster,
even) — the mechanism is learned + in-context, not inherited from the room embedding; (2) k=1 is
enough — no bandwidth gain from 2 or 4 carrier tokens per frame at this task; (3) all arms sit at
92±1% of the scale-matched teacher (8.95) — a consistent small distillation cost, candidate
explanation: the replica's multi-token read vs one token's write budget.

## [2026-07-18] Overnight — NATURAL images (real photos), Q-first blockfence

Jobs 123136/37 (`--natural`, new meta.json loader, carrier locus = concept word) →
`outputs/ladder/image_longN/replica_natural_{dist_far,dist_near}/20260718_*/`, n=50/cell (all the
cell has):

| cell | replica d′ | in-run joint anchor | gate→tally exact | model alone (historical) |
|---|---|---|---|---|
| dist_far | 6.22 ± 1.11 | 3.12 | **0.920 ± 0.036** | 0.580 |
| dist_near | 5.69 ± 0.33 | 3.61 | 0.760 ± 0.098 | 0.607 |

**Reading:** the architecture transfers to real photos — 1.6–2.0× the joint anchor within-run and
+0.15–0.34 exact over the frozen model — with smaller amplification than synthetic MMRED (these
cells never had the binding pathology; joint supply was already ~4.3). n=50 bars are wide;
per-copy values unstable at this n. Realism is not the boundary, consistent with the E4 verdict
map's dist_far PASS.

## [2026-07-18] Overnight — SECOND TASK FAMILY: co-occupancy (relational per-frame predicate)

- **Probe port:** `--task cooc` (evidence = both named characters share a room per states, sanity-
  checked against the answer; carrier locus = second name token). Q-first blockfence+posreset.
- **N=8 (job 123132, n=300, skip=0):** `outputs/ladder/image_longN/replica_cooc_qfirst/20260718_011918/`
  — **d′ 6.36 ± 0.27** (in-run joint anchor 4.40, ratio 1.45×); per-copy flat 3.6–4.8.
  **Gate→tally: 0.973 ± 0.011 exact** — vs frozen model 0.155, Σ-gate block read 0.74–0.77
  (the previous best), retrieve-then-verify ~0.66 (pair-verifier-limited). New cooc record.
- **Cross-task GATE (steps-trained logistic applied to cooc reads zero-shot):** per-frame err
  0.089, exact 0.460 — partial transfer; the gate direction is task-specific. Honest packaging:
  the carrier mechanism is shared, the gate costs 3.6k params + ~150 labeled samples per task.
  (The carrier-level cross-task eval — steps-trained e_c on cooc — job 123145, pending.)
- **Cooc long-N (jobs 123133/34):** `replica_cooc_qfirst_N{32,128}/20260718_*/` — supply holds:
  d′ 8.45 ± 0.16 @N=32 (anchor 5.67), 7.58 ± 0.25 @N=128 (anchor 5.72), per-copy flat.
  Gate→tally: **0.932 ± 0.015 @N=32** (err 0.0021), **0.680 ± 0.068 @N=128** (err 0.0042 — the
  relational predicate's ~3× higher per-frame error than steps starts to bite at 128 boundaries;
  still ~4× the frozen cooc model and above every prior cooc system).
- **CROSS-TASK CARRIER (job 123145):** the steps-distilled e_c evaluated zero-shot on cooc —
  `outputs/ladder/image_longN/carrier_token_crosstask_cooc/20260718_020945/` — **d′ 5.58 ± 0.09**
  (cooc replica teacher 6.36 → ~88% of ceiling on a task the carrier NEVER saw), fresh per-task
  gate: 0.880 ± 0.024 exact. The carrier mechanism is task-general (conditioning is in-context);
  only the 3.6k-param gate is per-task. Train-once claim: carrier ✓, gate = cheap per-task head.

## Single-forward supply ladder (updated)

joint 1.97–2.09 → replicas masked 2.52 / unmasked 3.56 → A (fenced): 4.07 → A2 (+posreset): 4.66
→ **A3 (+blockfence): 6.34 — GO, in the mp band (solo anchor same-instrument 6.01; per-copy
flat)**. One-forward supply CLOSED: replicas (clean query) + block-diagonal fence (clean values,
marker leak sealed) + per-block position reset (clean geometry) = multipass supply in one forward.
· B (unmix deployed): 1.44 — destructive, NO TRANSFER — and now moot (prevention beats repair) ·
C (broadcast gate): ~2.1 — NO-GO, routing not repairable from content. · **A4 (+Q-first): 9.24 —
supply amplifier; gate→tally 0.991 exact @N=8 in one forward.** Next rungs: N-sweep results
(16–128, in flight), learned 1-token carrier distillation (in flight), trained in-model readout.

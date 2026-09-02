# PAPER SKELETON — "Where Over-Squashing Lives"

> Drafted 2026-08-22 (Claude, reviewed by: —). Working doc for the thesis/paper story.
> Every claim is pinned to its RESULTS.md entry + canonical run dir, so this doubles as
> the verifiability audit. Status tags: **SOLID** (logged, seed/prereg-backed) ·
> **SOLID-CAVEAT** (logged, honesty flag applies) · **PENDING-ARMOR** (needs the armor
> campaign cell). Related-work positioning lives in
> [outputs/recagg/RELATED_WORK.md](../outputs/recagg/RELATED_WORK.md) — this file only
> summarizes the deltas.

---

## 0. Title candidates

1. *Where Over-Squashing Lives: Localizing the Aggregation Bottleneck in a Frozen
   Vision-Language Transformer*
2. *Perception Is Not the Bottleneck: A Drift Law for Learned Aggregation in Video QA*
3. *One Reader Per Item: Diagnosing and Evacuating the Aggregation Bottleneck in VLMs*

## 1. The story in one paragraph

A frozen production VLM (Qwen2.5-VL-7B) collapses on multi-frame counting as the number
of frames N grows. We localize the failure by construction: a ladder of interventions,
each isolating one pipeline stage (perception → aggregation → readout), shows that
per-frame **perception is repairable in-model and length-invariant** (block-diagonal
fencing + position reset: flat supply to N=128, zero training), that the per-frame facts
are **perfectly decodable yet unusable** by the model itself, and that the residual
failure is the **aggregation read**: one softmax query cannot address N items regardless
of key competition. Holding perception fixed, we then measure a **drift law** across the
spectrum of learned aggregators — attention pooling, GRU and SSM heads, a fine-tuned
2.8B Mamba, frozen LMs to 14B, CoT tally scratchpads — all fit training lengths and
decay beyond them, each for a theoretically predicted reason (Hahn 1/N sensitivity,
analog state drift / unexplored states, bounded readout range), while **symbolic
execution over the same per-frame records is length-invariant by construction**
(flat ≈0.89 from N=16 to 128, unseen task types included). The composed frozen system
(fenced VLM perception → program execution) retains length-coupling *only* in
perception fidelity (p^N), completing the localization. Conclusion: for exact
aggregation over growing visual inputs, delegation to an executor is not an
optimization but the only known length-invariant option.

## 2. Section outline with claims → evidence

### §1 Introduction

Framing: over-squashing (GNNs → "Transformers need glasses") predicts the aggregation
read is the bottleneck; nobody has *localized* it in a real VLM or measured which
repairs bend the length curve. Contribution list:

- **C1 (localization):** the failure is the aggregation read, not perception, not
  decodability, not readout format — shown by elimination.
- **C2 (fencing):** per-frame perception inside one frozen forward can be made
  length-invariant with a mask + position reset, zero training.
- **C3 (drift law):** under matched perception, every learned aggregator family drifts
  out-of-length; per-family mechanism given.
- **C4 (spectrum map):** the more exact/symbolic the aggregation substrate, the flatter
  the length curve — with in-model re-quantization (register trees) as a measured
  intermediate rung.
- **C5 (practical necessity):** symbolic execution is the only length-invariant rung;
  stated in the "within known trainable families" form, never as a theorem.

### §2 Related work

Use RELATED_WORK.md verbatim (must-cite ~15, delta sentences, top-threats). The one-line
positioning: *pipeline shape is precedented (NS-VQA→ViperGPT→HPP; PCW/APE for the mask);
the localization, the matched-perception drift law, and the necessity framing for
video-QA aggregation are unclaimed.*

### §3 Setup

MMRED (self-generated frames; "how many frames was C in room R?", answer ∈ 0..N) +
MMReD-HF benchmark port; Qwen2.5-VL-7B frozen 4-bit; metrics: exact-match, held-out
linear-probe d′, gate→tally accuracy. Honesty flag up front: synthetic-first design is
what makes matched perception possible (gold per-frame facts exist).

### §4 The failure and its locus (elimination, part I)

| claim | number | entry / run |
|---|---|---|
| Frozen baseline collapses (full prior) | 0.219 exact @N=8 | [2026-07-18] E1 · `frozen_baseline/20260718_125303` — SOLID |
| Same readout, joint graph: decays with N | 0.468→0.077 @N=8→128 | [2026-07-31] curves · `outputs/presentation/curves/` — SOLID |
| **Not perception:** fenced supply flat in N | fence rung 0.96–1.00 @N=8..128; d′ 10.67/10.23 @N=16/64; supply d′ 13.54 vs joint 5.95 @N=8 | [2026-07-31c] waterfall · `outputs/presentation/waterfall_grid/` + [2026-07-18] E1 — SOLID |
| Fencing zeroes cross-frame interference structurally | joint Jacobian 52% cross-frame → fenced exactly 0.00 | [2026-07-31] sensitivity · `outputs/presentation/sensitivity/` — SOLID (grad n=12 caveat) |
| Posreset is load-bearing and grows with N | supply gap +0.3→+4.1 @N=2→32; eval-without-reset collapse 0.987→0.313 @N=32 | [2026-07-27] + [2026-08-01b] · `posreset_sweep/` — SOLID |
| **Not decodability:** facts linearly present | gate→tally 0.998 @N=8; carrier probes ~1.00; perception bound 1.000 at every recagg cell | [2026-07-18] E1 + recagg P2 — SOLID |
| **Not key competition:** one reader can't address N items even over only N keys | super-carrier coarse ≈ full at every N (0.07 @N=64 both) | [2026-08-01] SUPER-CARRIER NO-GO · `outputs/presentation/supercarrier/` — SOLID |
| Query side is the binding constraint at scale | value-only repair +0.97→−0.47 @N=8→64; query-only ~+1 everywhere; interaction ~+2 | [2026-07-31b] q/kv swap · `outputs/presentation/qkv_swap/` — SOLID (N=128 cells PENDING-ARMOR D) |
| Squashed-readout ceiling matches the 1/√N law | measured best-linear within 0.01–0.05 of law at every N | [2026-07-23] P1.2 · `measured_ceiling/20260723_222428` — SOLID |
| Direct Hahn sensitivity/margin measurement | — | **PENDING-ARMOR A** (the theory-to-model figure) |

Mechanism summary sentence: *every working configuration in the record has one reader
per item, in space or in time* (replicas, carriers, serial scratchpad steps —
[2026-08-01] reading 2).

### §5 In-model repairs and their ceiling (elimination, part II)

The strongest possible in-model system (fence + learned carrier + LoRA + caption
scratchpad) is excellent **in trained length coverage** and still not length-general:

| claim | number | entry / run |
|---|---|---|
| In-model headline (in-length-trained) | N=32 0.987 (seeds 0.982±0.007) · N=64 0.981, pf 0 | [2026-07-24] FORMAT SWEEP / P1.1 · `carrier_fmt_caption/` — SOLID-CAVEAT (in-length-trained, synthetic) |
| Zero-shot length extrapolation fails for every in-model readout | le16→N=128 0.087; l12v2 N=128 full-34 0.235; digit readouts collapse to "0" | [2026-07-20→21] E-A + [2026-07-29] migration note — SOLID |
| Plain-LoRA SFT control: same in-length pattern, worse extrapolation | 0.967 @N=32 in-length · 0.787 @N=64-extrapolated · h200-only training | [2026-07-25] P4 — SOLID-CAVEAT (single-task) |
| Readout expressivity is a real separator | carriers+digit with identical in-length data: 0.333/0.140 (dead mid-range) | [2026-07-25] P4.2 — SOLID |
| Register trees: trained components ARE length-invariant, depth is not | V2 EM 0.920; every trained level transfers 4× at ≥0.97; never-trained depths dead under every mechanism | [2026-08-10→11a] FIVE-ARM TABLE · `outputs/ninv/` — SOLID |
| Per-level re-quantization eliminates error compounding | B decays ×0.96–0.99/level; D/V2 flat 1.000 | [2026-08-10→11a] — SOLID |
| LAW 7 (interface law): decode-equal ≠ substitutable | synthetic-trained lv5–6 alive on synthetic, dead on real (0.054); same ckpt as decoded numbers: 0.811 @N=64 | [2026-08-11b/c] · `eval_cascade` — SOLID |
| Hard-requant cascade is near-flat in N | EM(GT≤16) 0.925/0.875/0.811 @N≤16/32/64 vs majority 0.42/0.16/0.04 | [2026-08-11c] · `20260811_145257_cascade_p15c` — SOLID-CAVEAT (fitted 3584→17 interface head) |

Reading for the spectrum map (C4): the ninv arc shows *partial* symbolization
(quantize early, re-quantize per level, read out as decoded numbers) buys *partial→full*
flatness — the intermediate rungs between "all-learned" and "all-symbolic" behave
exactly as the axis predicts.

### §6 The drift law (matched perception, the headline table)

All rows consume identical per-frame inputs (verified decodable at ~1.00) — any
difference is attributable to aggregation alone. Fit @N={8,16}, zero-shot @{32,64}
(recagg P2, HF@512 · `outputs/recagg/p2_extrap/20260817_160326_hf512/` — SOLID):

| aggregator | N=32 zs | N=64 zs | mechanism of failure |
|---|---|---|---|
| attention pool (R4) | 0.236 | 0.048 | Hahn 1/N (reproduces the c(fan) collapse in our own feature space) |
| GRU head (R2) | 0.640 | 0.256 | analog drift + unexplored states |
| SSM head (R3) | 0.800 | 0.220 | same (R3b-noleak 0.864 @64: drift = length-coupled learned params) |
| fine-tuned Mamba-2.8b | 0.58 | 0.14 (0.00 @128) | state collapse past train length |
| frozen LMs ≤14B over captions | ~majority by N=32–64 (best 0.695 @N=8) | — | Arm A: no frozen LM counts |
| CoT tally scratchpad (Qwen14B) | — | 0.060 | loses its own running count — token-externalization is NOT enough |
| chunk-16 external composition | 0.78 | 0.58 (0.42 @128) | partial rescue = partial externalization (mid-spectrum rung) |
| **symbolic probe→sum (R1)** | **0.996** | **1.000** | exact by construction |

- Attn-pool failure is structural, not budget (trainEM 1.0 → 0.302 in-length park) —
  and the SSM 3k-vs-20k ablation shows we did rule out undertraining for the
  integrators (0.298→0.984 @20k). SOLID.
- Buitrago state-passing control: **PENDING-ARMOR B** (the one open counter-argument).

### §7 The endpoint: externalized aggregation + the composed system

| claim | number | entry / run |
|---|---|---|
| Ask–Compile–Execute is length-FLAT, task-agnostic | ~0.89 @N=16→128, unseen types ~0.88, exec-fail 0.3% | recagg Arm C v3 · `armC_compile/*_qwen14b_v3/` — SOLID |
| Composed frozen system: residual length-coupling lives ONLY in perception | 0.79/0.81/0.72/0.72 @N=16→128 vs oracle 0.92/0.89/0.85/0.90; single-frame types ~1.00 flat; exact-count types decay as p^N | Arm B · `armB_captions/…full` + `armB_execute/…v3progs` — SOLID |
| Frozen two-pass on the untouched HF benchmark | EMIT-EM 0.960 @512px; cond-EM 1.000 (all error is pass-1 quantization) | [2026-08-10b] · `20260810_151238_twopass_hf512` — SOLID |
| External benchmark anchor (non-synthetic) | — | **PENDING-ARMOR C** (MLVU-AC composed-system cell; the in-model rung already measured NO-GO 0.107 there, [2026-07-24] P2b) |

Necessity claim, final wording (C5): *within every trainable aggregator family tested,
under matched perception, length generalization fails as theory predicts; symbolic
execution is the only length-invariant rung, and it is invariant by construction.*
Pair with Apple 2510.14826 (SSM theorem), Merrill–Sabharwal (CoT), Yehudai (range) —
we transport and verify in the VLM/video setting and add the mechanism triad. Never
claim a theorem; Chiang & Cholak expressibility and Buitrago interventions are cited
as the honest boundary.

### §8 Secondary results (own subsection or appendix)

- **Cross-family port:** InternVL2.5-8B — supply mechanism ports (3.5× joint), scaffold
  0.938; Q-first amplifier is Qwen-specific ([2026-07-19]/[2026-07-24] P3b). SOLID.
- **Cross-domain:** natural images — supply GO (d′ 27.3), scaffold GO (0.980), in-model
  NO-GO (domain-bound trained readout) ([2026-07-24] P3a). Corroborates the
  localization: the mechanism is architecture-level, the learned rung is domain-bound.
- **No-harm:** both adapter families |Δ| ≤ 1.4 pts on MME/POPE, always-on safe
  ([2026-07-19] E-D / [2026-07-24] P4.3). SOLID.
- **Efficiency:** exact cached-fast decode 16–311×; KV-truncation refuted — decode
  reads frames; two-channel (prefill-aggregation vs decode-readout) mechanism finding
  ([2026-07-25] TRUNC). SOLID.
- **Anchor/ablation battery:** Q-first −46%; L\*=12 inverted-U; room-token anchor wins
  the sweep ([2026-07-19] C1/C2, [2026-07-23] E-H, [2026-07-31]). SOLID.

### §9 Limitations & honesty flags (verbatim from the record)

1. Headline in-model numbers are in-length-trained on clean synthetic MMRED; zero-shot
   length extrapolation of in-model readouts fails — the in-model claim is a *ladder in
   trained-length coverage*, and that is precisely the point the drift law generalizes.
2. Necessity is practical/empirical, not a theorem (see §7 wording).
3. The cascade's interface head is a fitted linear component (same class as the
   two-pass quantizers); all-model-emitted long-context variant open.
4. N=128 qkv cells uncitable until the n≈80 rerun (ARMOR D).
5. d′ estimates scale with n — like-for-like comparisons only.

## 3. Figure plan

- **F1 (headline): the spectrum figure.** Accuracy vs N (log-x): one decaying curve per
  learned aggregator (§6 table), the flat symbolic line on top, chunk-16 as the
  mid-spectrum curve; inset: fenced supply d′ flat to 128. Source: recagg P2 CSVs +
  presentation curves. *(Instantly-readable; this is the one-image thesis.)*
- **F2: the repair staircase** (waterfall grid, N=8..128) — which repairs bend the
  curve. Exists: `outputs/presentation/waterfall_grid/`.
- **F3: mechanism panel** — Jacobian cross-frame shares (52% vs 0.00), attention-map
  islands, q/kv 2×2 with N-scaling. Exists: `outputs/presentation/`.
- **F4: the Hahn figure** — Δlogits and margin vs N with measured bf16 noise floor and
  the accuracy-collapse N marked. PENDING-ARMOR A.
- **F5: the in-model spectrum** — five-arm register-tree table as a depth×length
  invariance heatmap + cascade flat-in-N curve. Exists: `outputs/ninv/`.

## 4. What must land before submission (armor campaign)

| item | closes | status |
|---|---|---|
| A: Hahn sensitivity/margin instrument | the theory-to-model gap; F4 | queued |
| B: state-passing control (GRU/SSM) | Buitrago objection on §6 | queued |
| C: MLVU-AC composed-system cell | external validity of §7 | queued |
| D: N=128 qkv rerun n≈80 | §4 citability at 128 | queued |

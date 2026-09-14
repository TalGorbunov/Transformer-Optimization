# condmask — run report (2026-08-17/18)

**Question**: does training a Transformer with a learnable attention mask at short
sequence length improve exact-match at longer lengths? Mask = binary block-vs-full per
(layer, head), decided by a network conditioned on the whole input (mean-pooled
embeddings → MLP), optimized with hard ST-Gumbel, trained with CE on the answer digit.
Task: text-MMRED steps_in_room (Qwen2.5-1.5B-Instruct, LoRA r8 all layers) + vision
confirm (frozen 4-bit Qwen2.5-VL-7B, park renders, replica scaffold, native positions).
All arms fit the train set to EM ≈ 1.000 before any comparison (Stage-0 guarantee).
Every number = free greedy decode EM vs per-N majority, shuffled slices, n=360–400/cell,
3 seeds (vision: 1 seed). Run dirs: `outputs/condmask*` on this machine.

## Main table — text, EM (mean of 3 seeds)

Majority baselines: N8 .12 · N16 .07 · N32 .08 · N64 .07 · N128 .07.
Frozen model: full .100/.092/.005/.000/.007 · blockwise .115/.055/.075/.055/.062.

| grid (train data) | arm | N8 | N16 | N32 | N64 | N128 |
|---|---|---|---|---|---|---|
| single-N=8, 5.4k | A: LoRA+full | .998 | .286 | .140 | .128 | .107 |
| | B: LoRA+fixed blockwise | 1.00 | .322 | .118 | .151 | .095 |
| | C: +learned mask (block init, v1) | 1.00 | .271 | **.185** | **.179** | .115 |
| | D: +learned mask (rand init, v1) | 1.00 | **.352** | .148 | .166 | .115 |
| mixture {2,4,8}, 5.4k | **A: LoRA+full** | .999 | **.523** | **.395** | **.258** | **.142** |
| | B | 1.00 | .487 | .131 | .121 | .122 |
| | C | 1.00 | .406 | .124 | .124 | .100 |
| | D | 1.00 | .445 | .182 | .162 | .127 |
| | E: gates-only + frozen readout | .883 | .362 | .122 | .102 | .085 |
| | F: gates-only, rand init | .808 | .414 | .198 | .127 | .105 |
| mixture, 18k (XL) | A | .998 | .490 | .364 | .237 | .139 |
| | B | 1.00 | .487 | .135 | .172 | .101 |
| | C | 1.00 | .447 | .271 | .220 | .122 |
| | D | 1.00 | .462 | .233 | .158 | .110 |

Gold≤8 control slices (output-range confound removed): mixture-A .628/.501/.311 at
N=16/32/64 (vs .523/.395/.258 full-range) — the range confound is real (~+0.1) but the
in-range collapse with N remains. Fixed blockwise B hits .63–.65 at N=16 in every grid,
then dies at N≥32.

Vision (7B, trained N=8, 1 seed): A 1.00/.510/.230 at N=8/16/32 > D .430/.185 >
B .420/.170 > C .300/.100 — full attention wins on the 7B even at single-N training.

## Findings

1. **The claim holds only in the length-impoverished regime.** With single-N training
   on the 1.5B, learned masks genuinely beat full attention at N≥32 (C .185 vs A .140,
   non-overlapping seeds) and fixed blockwise beats it at N=16/64. But **a length
   mixture {2,4,8} with plain full attention dominates every masked arm at every N**
   (.395 vs ≤.182 at N=32) — the mask was a substitute for length diversity, and once
   the training data supplies that diversity, the mask only subtracts information. On
   the 7B (vision) full attention wins even under single-N training.
2. **The mixture, not data volume, is the active ingredient**: 5.4k mixture ≈ 18k
   mixture for arm A (.395 vs .364 at N=32); tripling data bought nothing further.
3. **CE chooses blockwise voluntarily but never uses the input-conditioning.** From
   both inits, gates converge to P(block) ≈ 0.58–0.70 (random init *rises* from 0.5) —
   yet cross-input variance is 0.000 in every run, including mixed-N training where
   different inputs genuinely warrant different masks. The "input-conditioned" pathway
   (zero-init weights) is never left by SGD; the learned object is a static mask.
4. **Gates flip decisively when the instrument is sound** (anti-S0): canary converges
   in 3 steps; real runs flip 90–370 of 336–784 gates. The S0 reward-blindness
   diagnosis is confirmed twice over: with a trained readout attached, gates-only
   optimization is productive (E reaches dev EM 0.94 with 336+MLP trainable params),
   while S0's frozen-readout objective never flipped one gate of 6.7M.
5. **Mask-decided-early beats mask-co-trained** (v1 .185 vs v2 .105 at N=32): letting
   the readout co-adapt with a moving mask overfits jointly; the early-frozen mask
   generalizes better. (The principled two-phase variant is untested on vision.)
6. **The residual long-N collapse (in-range, mixture, full attention: .63→.50→.31 at
   N=16/32/64) is the genuine aggregation wall** — consistent with the repo's fan-in
   law; masks and data diversity both leave it standing. That is the target for
   in-forward aggregation architecture (register trees, per-head position regimes).

## Gate-dynamics recipe (hard-won)

lr 3e-2 (S0's direct-logit value) explodes MLP-parameterized gate logits (|Δ| > 2000
in a smoke epoch) and freezes them by saturation within ~1 epoch. Working recipe:
**lr 1e-3 + grad-norm clip 1.0 + LayerNorm on the pooled input**; barrier |init| = 1.0;
τ 2.0 held 30% → 0.5. Telemetry that catches everything: per-epoch flips vs init,
mean/max |Δlogit|, P(block) heatmap, cross-input variance, plus the planted-target
canary test (tests/test_condmask.py).

## Instrument bugs caught (all fixed in-repo)

- SDPA mem-efficient backward hard-errors ("LSE is not correctly aligned") when the
  attention mask is the ONLY gradient path → force MATH backend for gates-only arms.
- Fixed-arm eval used the wrong mask regime for arm B (full instead of blockwise).
- sbatch --export silently splits comma-lists (twice!) → '+' separators everywhere.
- Per-second-timestamped eval dirs collide under simultaneous jobs → pid-suffixed.
- Frozen 1.5B failed the E/F preflight (≤ majority+2pp) → pre-registered fallback
  (frozen arm-A readout) used throughout.

## Where this leaves the thesis direction

Attention-topology interventions (hand fences, learned masks, gating) now have three
independent negative-boundary results on trained-readout models: they help only when
the model is small AND training lacks length diversity. The surviving lever for
single-forward-pass aggregation is the one the audit pointed at: in-model aggregation
*structure* (bounded-fan-in register trees in the mask vocabulary, per-head N-invariant
position regimes) attacking the in-range collapse that neither masks nor data diversity
fixed. The condmask machinery (per-head 3D masks, HeadGates, template system) is the
scaffolding for exactly that next step.

---

# Addendum: the relmask line (2026-08-18) — expressiveness ladder, oracle bound, and the geometry verdict

**Setup**: filtered-counting task (each frame one sighting; "How many frames show C in
the R?"; distractor frames = other characters, so relevance is question-dependent and no
static mask can be optimal; gold <= 8 at EVERY N, so length transfer is free of the
answer-range confound). Arms (all with co-trained LoRA, 3 seeds, train N=8):
A2 full attention · B2 static-isolate-everything · G relational per-frame MLP gates ·
Gx relational transformer gates · Gx@k14/k28 (gate conditioned on layer-14 / layer-28
states) · Gx@k28-grad (trained-through full-depth emitter) · Dyn (per-layer, per-token,
per-head gates from h^{l-1} — the maximal mask family member) · O oracle (isolate
exactly the distractors).

**Transfer EM (mean of 3 seeds; majority ~0.11):**

| arm | N8 | N16 | N32 | N64 |
|---|---|---|---|---|
| A2 full attention | .999 | **.629** | **.468** | **.283** |
| O oracle mask | .917 | .426 | .293 | .253 |
| Gx@k28-grad (best learned) | .998 | .315 | .187 | .175 |
| G / Gx / Gx@k14 / Gx@k28 / Dyn | ~1.0 | .19-.29 | .14-.20 | .13-.15 |
| B2 isolate-everything | .497 | .180 | .003 | .000 |

**Findings:**
1. **Incentive, not expressiveness, is the binding constraint on learned relevance.**
   Every learned emitter — up to the full-depth trained-through model and the layerwise
   h-conditioned per-token gates — converged to indiscriminate cutting:
   P(cut|distractor) - P(cut|relevant) = 0.000 in every run. At train N=8 the model
   reaches CE~0 with full attention, so nothing pays for relational selectivity.
2. **The oracle bound is NEGATIVE**: perfect relevance-based masking transfers worse
   than no masking at every length (0.293 vs 0.468 @N=32). No mask emitter, however
   expressive, can beat "do nothing" on this task — the family is bounded below its own
   control.
3. **The geometry probe pins the length collapse on POSITIONS, not content** (the
   decisive experiment): with oracle-SELECTED content at N=64, masked-but-present
   distractors give EM 0.253-0.283, physically DELETED distractors (relevant frames
   repacked to compact positions, same information) give EM **0.998-1.000** (n=400).
   75 EM points from geometry alone. Interference removal is worth ~nothing; position
   canonicalization is worth ~everything. (Consistent with ninv's posreset result and
   the fence's design; explains why masks — which cannot touch positions — kept losing
   to full attention across condmask AND relmask.)

**Where this leaves the program**: the winning shape is not "emit a mask" but
**select-and-repack** — an input-conditioned selector chooses the relevant units and a
compact second pass reads them at in-distribution positions (oracle version measured at
~1.0 @N=64 here; the repo's retrieve-then-verify is the multi-pass ancestor). The
relational-emitter machinery (RelGateTF/DynGate, oracle bounds, gap telemetry) carries
over with the interface changed from edge-deletion to token-selection + position
repacking; alternatively, in-pass per-head canonical position regimes (ASSESSMENT.md
improvement #1) target the same geometry without a second pass.

---

# Addendum 2: Co-Former (2026-08-18) — the Co-GNN adaptation that works

**Design** (from the Co-GNN discussion): two-pass select-and-repack. A selector reads the
input and emits per-frame KEEP/DROP; the answer pass runs with dropped frames cut from
the attention graph in both directions AND kept frames at canonical COMPACTED positions
(rank-among-kept) — actions rewire edges *and positions*, the piece GNNs get for free
(no absolute positions) and transformers must add explicitly.

**Final table** (filtered-counting, trained N<=32, 3 seeds, gold<=8 at all N; majority ~.11):

| arm | N8 | N16 | N32 | N64 | N128 |
|---|---|---|---|---|---|
| A2 short-trained, full attention | .998 | .728 | .513 | .327 | .217 |
| A2mix in-length {8,32} (strong baseline) | 1.00 | .982 | .999 | .560 | .290 |
| **LAF-final: learned selector + repack** | **.999** | **.998** | **.998** | **.932** | **.857** |
| OR: oracle selector + repack (ceiling) | .997 | .952 | .983 | .997 | .976 |

LAF-final = pairwise N-invariant selector (per-frame question x frame MLP on layer-0
embeddings, no cross-frame attention), warm-trained with class-balanced BCE on
task-derivable relevance labels covering the deployment number range, recall-calibrated
keep margin, selector FROZEN, LoRA trained under it at N<=32. keep(rel) ~0.99 at every N.
**First fully-learned configuration to beat the in-length baseline at long N: 1.7x at
N=64, 3.0x at N=128, ~88% of the oracle ceiling.**

**The five structural lessons of the Co-Former line:**
1. **Positions, not edges, carry the value** (established by the geometry probe; the
   trained-OR arm is essentially length-invariant, 0.976 @N=128).
2. **ST gradients cannot learn selection here**: the payoff of dropping a frame routes
   through the non-differentiable position recompaction, so gradient masks see only the
   cost of lost edges — LS stayed at keep-all, joint LA collapsed to drop-all.
   Supervised-then-frozen selection is not a hack; it is forced by the estimator.
3. **The selection problem inherits the length-generalization problem** if the selector
   is itself a transformer over N native-positioned summaries (TF selector: keep(rel)
   0.99 -> 0.66 as N grew). Fix: match the selector's structure to the relevance
   structure — pairwise (q x frame), position-free features -> N-invariant by
   construction.
4. **Watch content-side length leaks**: frame-number tokens ("Frame 87:") drift out of
   the warm range and blew keep(dist) to 0.83 at N=128; covering the number range in
   warm data fixed it (0.857 final). The same leak explains masked-repack (0.875)
   vs physical deletion (0.998) — text the mask cannot rewrite.
5. **Recall is the only expensive error**: dropping a relevant frame is fatal
   ((1-p)^N); keeping distractors is nearly free. Calibrate the operating point to
   keep(rel) >= 0.99, not to balanced accuracy.

**Relation to Co-GNN**: the action space (per-node keep/drop = ISOLATE vs STANDARD),
straight-through optimization, and joint action/environment training were all ported —
and the two failures Co-GNN does not face were both measured and both fixed:
(a) transformers need action-consistent positions (GNN isolation is size-invariant for
free; RoPE isolation is not), and (b) Co-GNN's benchmarks make actions matter in
training, whereas here task CE is satisfiable without actions at trainable lengths,
so the action network needs its own supervision. With both fixes, the Co-GNN idea
delivers 3x the strongest baseline at 16x the training length.

---

# Addendum 3: real-world applicability (2026-08-18)

**Context window**: not the issue. Qwen2.5-1.5B supports 32,768 tokens; our longest eval
is ~1,600 (5% of window). The collapse is skill-distribution OOD (counting behavior
trained at short layouts), not context capacity.

**Label-free arms (3 seeds each):**

| arm | labels | passes | N8 | N16 | N32 | N64 | N128 |
|---|---|---|---|---|---|---|---|
| **LZ: self-judged select+repack** | **none** | 2 | .999 | .998 | .994 | **.970** | **.912** |
| SP: shared positions (all frames as-if-first) | none | 1 | 1.00 | .304 | .153 | .150 | .098 |
| LAF (gold-label reference) | task-derived | 2 | .999 | .998 | .998 | .932 | .857 |
| A2mix in-length baseline | — | 1 | 1.00 | .982 | .999 | .560 | .290 |
| OR oracle ceiling | oracle | 2 | .997 | .952 | .983 | .997 | .976 |

LZ = identical pipeline to LAF, but selector warm labels come from the FROZEN model's
own zero-shot judgment ("Is this frame relevant to the question? yes/no"; 93-96%
agreement with oracle, cached). **The label-free version BEATS the gold-label version**
(0.912 vs 0.857 @N=128) — likely because "relevant to the question" (which names both
the character AND the room) tracks task-relevance more tightly than the entity-match
gold labels: harmless relevant-but-non-evidence frames get dropped, the repacked
geometry gets cleaner, and evidence-recall is what actually bounds EM.

**SP (positions-only, the single-pass variant) fails**: normalizing every frame to
as-if-first positions makes per-frame ENCODING length-invariant, but the readout still
aggregates N positionally-indistinguishable items — the read fan-in limit reasserts
itself and grows with N. Position normalization alone is not enough; the token-count
REDUCTION (selection) is what defuses the aggregation limit. Both halves are needed:
select (fewer items) AND repack (familiar geometry).

**Bottom line for real settings**: the deployable recipe is fully label-free —
(1) the model zero-shot-judges per-unit relevance to the question, (2) a tiny pairwise
N-invariant selector distills those judgments (recall-tilted threshold), (3) irrelevant
units are dropped and the survivors repacked at compact positions, (4) the model
answers on the short, familiar-looking input. Trained at N<=32, it holds 0.91 EM at
N=128 — 3.1x the strongest conventional baseline — with every component's failure mode
measured along the way.

---

# Addendum 4: single-forward attention rewiring — the elegance/performance frontier (2026-08-19)

**Probe first**: the model's own answer-row attention IS a relevance detector, peaking at
L16 (AUC 0.83 frozen, 0.91 after task training) — but raw recall ~0.85, far below the
~0.99 exact counting needs. So the attention was trained to BE the selector: aux loss
distilling the model's own zero-shot yes/no judgments into the L16 attention pattern
(teacher used at training time only; deployment = one forward, one threshold).

**The rewiring family** (all single-forward, selection at L16 by own attention mass,
survivors position-compacted; 3 seeds):

| arm | N8 | N16 | N32 | N64 | N128 |
|---|---|---|---|---|---|
| AS: frame-level rewiring | 1.00 | .996 | .997 | **.888** | .438 |
| AST: token-level rewiring | 1.00 | .888 | .997 | .779 | **.530** |
| ASTH: token-level + per-head masks | 1.00 | .872 | .999 | .772 | .518 |
| (LZ two-pass, self-judged) | 1.00 | .998 | .994 | .970 | .912 |
| (A2mix baseline) | 1.00 | .982 | .999 | .560 | .290 |
| (oracle ceiling) | 1.00 | .952 | .983 | .997 | .976 |

**Findings:**
1. **The fully-internal version works**: one model, no prompts, no selector network —
   1.6x baseline @N=64, 1.5-1.8x @N=128. The transformer rewires its own graph.
2. **But it pays a measured price at extreme length** (.44-.53 vs LZ's .91 @N=128), and
   the reason is the RECURSION PRINCIPLE, now measured for the third time: whatever
   performs the selection must itself be length-invariant. The L16 attention doing the
   selecting runs on layers 0-15 at native long positions — so the selector degrades
   exactly like the model it is trying to save. LZ's judge reads each frame in a tiny
   fixed-size prompt and never degrades. (Same lesson as the TF-selector's keep(rel)
   0.99 -> 0.66 and the frame-number leak.)
3. **Token granularity trades mid-range for extreme length** (AST beats AS @N=128,
   loses @N=16/64): finer cuts = better geometry at 128, worse recall mid-range.
   **Per-head refinement adds nothing** (ASTH == AST) — mask-only refinements on
   already-fixed geometry are marginal, consistent with every prior mask result.
4. **SP (shared as-if-first positions, no selection): fails** (.153 @N=32) — position
   normalization without token-count reduction leaves the read fan-in limit standing.

**The frontier, stated plainly**: elegance (one pass, fully internal) currently costs
~0.4-0.5 EM at 16x training length versus the two-pass self-judged pipeline, and the
cost is attributable to one thing — the internal selector reads the long input at
native geometry. Closing that gap = making the in-forward selection length-invariant
(e.g., blockwise-encoded layers 0..k at canonical per-frame positions BEFORE the
selecting layer — the fence returns, one level down), which is the natural next
experiment.

---

# Addendum 5: AS3 — the recursion principle resolved (2026-08-19)

The single-forward variant, iterated three times, each failure landing exactly where
the recursion principle points, one level deeper:

- **AS**: selection by L16 attention, layers 0-15 at NATIVE geometry -> the selecting
  attention degrades with N like the model it rescues (.888@64, .438@128, ~.23@1024).
- **AS2**: frame KEYS made invariant (blockwise + shared positions) but the selection
  QUERY (answer row) still built from an N-diluted read of all frames -> dev32 1.0,
  transfer collapse (.226@64). Invariant keys, degrading query.
- **AS3**: BOTH sides invariant — pre-segment isolates every frame at as-if-first
  positions AND blinds the tail to frames, so the L16 selection is a question-pure
  query against independently-encoded frame keys: an internal retrieval head,
  length-invariant by construction. Then repack, then answer.

**AS3 results (3 seeds, trained N<=32, one forward, zero deploy scaffolding):**
EM .999/.958/1.000/.995/.919/.813/.756/.633 at N=8/16/32/64/128/256/512/1024 —
matches the two-pass LZ at N<=128 and dominates it at extreme length (.633 vs .190
@N=1024; 6x baseline; ~2/3 of the oracle's 1.000). The attention-based selection also
proves more drift-robust than LZ's distilled MLP selector (no frame-number-range leak).

**The design law, four measurements deep**: every component that participates in
selection — keys, query, features, thresholds — must itself be length-invariant, or it
re-imports the pathology it exists to cure. In GNN terms: the action network must not
over-squash. AS3 is what that law, applied exhaustively, produces.

---

# Addendum 6: vision leg — cross-substrate consistency (2026-08-19)

Visual filtered-counting (park renders, one character per frame, gold<=8 at all N),
frozen 4-bit Qwen2.5-VL-7B, LoRA trained N<=32, 2 seeds. The VLM judges its own frames
zero-shot ("Is {C} in this image? yes/no") — measured judge quality: agreement 1.000,
recall 1.000, false-keeps 0.000 over 86,400 frame judgments (prompt matters: the
meta-question "is this frame relevant?" scored only 0.73; the direct visual question is
perfect even at 224px). Physical repack (kept frames re-fed as a short input) — NO
custom attention machinery anywhere.

| selection | N8 | N16 | N32 | N64 |
|---|---|---|---|---|
| none (baseline) | .988 | .880 | .802 | .657 |
| judge (label-free) | 1.000 | 1.000 | .998 | **1.000** |
| oracle | .998 | 1.000 | .996 | .998 |

The label-free system sits AT the oracle ceiling at every length — same method, same
mechanism, same result shape as text. Cross-substrate consistency: established.
(Note the 7B's baseline degrades slower than the 1.5B's — bigger models length-
generalize better natively — yet the intervention still buys +0.34 EM at N=64.)

---

# Addendum 7: the full validation matrix (2026-08-20) — tasks x substrates x models x lengths

**Task battery** (text 1.5B, ONE multi-task model over 4 question types, trained N<=32,
2 seeds; EM at N=8/16/32/64/128):

| arm | steps | entity | distinct (nonlinear) | cooc (relational) |
|---|---|---|---|---|
| baseline | 1.0/.91/1.0/.79/.52 | 1.0/.90/.97/.71/.42 | 1.0/.98/.99/.98/.77 | 1.0/.91/.95/.63/.36 |
| **AS3** | 1.0/.99/.97/**.96/.83** | 1.0/1.0/1.0/**1.0/.93** | 1.0/.99/1.0/**.98/.83** | 1.0/1.0/1.0/**.96/.81** |
| oracle | .99/.97/.80/.86/.86 | 1.0/.99/.98/.97/.99 | 1.0/.98/1.0/.99/.99 | 1.0/.98/.93/.92/.94 |

AS3 generalizes across ALL four reductions — including nonlinear distinct-counting and
pair-relational co-occurrence — with ZERO per-task engineering: the internal attention
selection adapts to each question by itself. (LZ's generic pseudo-prompt does not
transfer as well across tasks: .41-.44 @128 — per-task judge prompts would be needed.)

**Cross-model, text** (steps suite; AS3 layer by the ~57%-depth rule):

| model | arm | N64 | N128 | N256 |
|---|---|---|---|---|
| Qwen2.5-3B (L20) | AS3 | **.982** | **.981** | **.950** |
| Qwen2.5-3B | LZ | .895 | .976 | .473 |
| Qwen2.5-3B | baseline | .887 | .518 | .343 |
| Qwen2.5-1.5B (L16) | AS3 | .995 | .919 | .813 |
| Qwen2.5-0.5B (L14) | AS3 | .320 | .181 | — |
| Qwen2.5-0.5B | LZ / baseline | .386 / .402 | .236 / .250 | — |

The depth rule (~57%) works out of the box on 1.5B and 3B (the 3B's AS3 is the best
text result of the program: .95 @N=256). On the 0.5B EVERY mechanism fails — the
method's power tracks the model's own per-unit competence (selection cannot outrun the
selector's understanding).

**Cross-model, vision** (judge-selected vs none; judge recall = 1.000 on all three
VLMs, false-keeps 5-14% on the smaller two):

| model | selection | N16 | N32 | N64 |
|---|---|---|---|---|
| Qwen2.5-VL-7B | judge | 1.0 | .998 | **1.000** (vs none .657) |
| Qwen2.5-VL-3B | judge | .976 | .956 | **.900** (vs none .547) |
| Qwen2-VL-2B | judge | .966 | .916 | **.883** (vs none .260) |

Smaller VLMs collapse harder natively and gain more from selection (2B: 3.4x at N=64),
while their zero-shot judging stays reliable. Consistency across substrates, models,
and tasks: established.

**Campaign integrity notes**: battery/vision data seeds partially unrecorded
($RANDOM — regenerate with fixed seeds for the camera-ready data); LZ per-task judge
prompts unexplored (AS3 made them unnecessary); mmred-HF's original 24-qtype benchmark
still untested (needs the upstream renderer); N>64 vision untested.

---

# Appendix B: team FAQ (2026-08-24)

**Q1 Tasks & lengths.** Filtered-counting family (text + park-render vision): one
sighting per frame; qtypes steps-in-room / entity count / distinct rooms (nonlinear) /
co-occurrence (pair-relational). Gold <= 8 at EVERY N (longer N = more distractors
only). Trained N in {8,32} (oracle arm: N=8 only); evaluated N=8..128 everywhere, text
to 1024, vision to 64. 2.7-5.4k train samples/task, 80-450 eval/cell, 2-3 seeds.
Precursor phase: crowded steps-in-room (source of the mask-family negative results).

**Q2 Models.** Text: Qwen2.5-Instruct 0.5B/1.5B(main)/3B, frozen bf16 + LoRA r8 a16 on
q/k/v/o all layers. Vision: Qwen2.5-VL-7B (4-bit), Qwen2.5-VL-3B, Qwen2-VL-2B. The
relevance judge is the same frozen model (zero-shot yes/no logit comparison) — no
external models, no labels.

**Q3 Layer choice.** Per-layer probe: answer-row attention mass per frame scored as a
relevance classifier (AUC), minutes of GPU. Unimodal curve, peak L16 on the 1.5B
(AUC .83 frozen / .91 trained) = the verdict-formation depth. Derived rule: ~57% of
depth; validated untouched on the 3B (L20 -> .95 @N=256).

**Q4 Attention -> keep; frames vs tokens.** Recompute the answer row's attention at the
selection layer manually (layer's LN + q/k + RoPE; SDPA hides weights; one row only);
sum mass per frame span, mean over heads; keep iff mass >= 0.5/n_frames. The row means
relevance by construction (AS3 pre-segment: isolated as-if-first frames + question-only
tail => question-pure query x independent frame keys = retrieval head) and by training
(aux BCE distilling the model's own zero-shot judgments; teacher discarded at deploy).
Frames not tokens: token variants (AST/ASTH) traded mid-range for marginal extreme-N
gains; recall asymmetry decides (a dropped evidence token ruins the count, (1-p)^k; a
kept junk frame costs ~nothing) -> select at the natural relevance unit, tilt to recall.

**Q5 Why repack.** Measured: oracle-selected content at N=64, masked-in-place 0.25 vs
deleted/repacked 1.00 — identical information, 75 points from geometry; oracle MASKS
lose to no masking at all. Transformer graphs are edge-LABELED (RoPE relative
positions); rewiring edges without renormalizing labels runs the graph through a
miscalibrated function. Repack = contiguous renumbering => the answer computation is
identical at every N (oracle flat to 1024).

**Q6 Sensitivity.** S(i) = ||d logit(gold) / d h_L16(frame_i)|| (at the rewiring layer,
isolating the aggregation segment); reported as evidence influence share
sum_rel/sum_all (baseline .79->.33 with N; oracle flat 1.0). Companion: effective
fan-in 1/sum(alpha^2) of the mask-aware answer-row attention (~0.4N baseline; ~5-10
rewired). Instrument pitfalls fixed: use shares not raw norms (activation-scale
confound); include the active mask in recomputed attention (phantom mass otherwise).

**Q1 addendum — minimal training lengths (measured 2026-08-24):** AS3 trained at N=8
ONLY: .68/.65/.54/.28 at N=16/32/64/128 (vs .96/1.00/.995/.92 for {8,32}-trained). With
the oracle arm (N=8-trained, flat to 1024) this pins the attribution: the ANSWER pass
needs only N=8; the SELECTOR's attention supervision needs distractor-rich examples
(one moderately long length — N=32 — suffices for generalization to N=1024). Minimal
recipe: readout short + selection supervision at one long length.

## Addendum 8 — Dense counting at the token level (2026-08-24)

**Question:** does select-and-repack extend to DENSE tasks where every frame is
relevant and evidence is a token-level clause? New qtype: crowded steps_in_room —
every frame contains all 5 characters ("Frame k: A is in R1. B is in R2. ..."),
so frame-level selection is useless (all frames kept = baseline); the evidence
unit is the single clause mentioning the queried character in the queried room.
Data: `data/mmred_crowded` (seeds 101-108, fixed; gold <= 8; train N in {8,32};
tests N=8..128). Runs: `outputs/crowded*` (2 seeds each unless noted).

| arm | N8 | N16 | N32 | N64 | N128 |
|---|---|---|---|---|---|
| A2mix baseline (LoRA, mixed-N) | 1.000 | .932 | .920 | .356 | .285 |
| AS3T learned, aux=evidence-clause | 1.000 | .958 | 1.000 | .512 | .265 |
| AS3T learned, aux=entity-mention (superset) | 1.000 | .814 | .972 | .246 | .205 |
| OTOK oracle: evidence-clause tokens only | 1.000 | .554 | .556 | .556 | .552 |
| OTOK-H oracle + frame headers kept | 1.000 | .980 | .938 | .890 | .883 |
| AS3T-H learned + headers (2 seeds: high var) | .998 | .588 | .810 | .388 | .235 |

**The headline mechanistic finding — the identical-item counting wall.** OTOK
(perfect token selection) is perfectly FLAT N16-128 (~0.55): geometry is fully
solved by token repack, exactly as designed. But the plateau is LOW because what
survives is g copies of the *identical* clause ("C is in the R." x g) with the
frame headers dropped — counting g indistinguishable adjacent items is the
counting problem in its purest form, and no amount of selection or geometry fixes
it. Keeping each kept frame's "Frame k:" header as a *distinct anchor* (OTOK-H)
lifts the ceiling to .98/.94/.89/.88. Dense counting therefore decomposes into
THREE separable, individually measured requirements:
1. **geometry** (token repack -> flat curves; OTOK flat vs A2mix decaying),
2. **distinctness** (anchors -> .55 to .89 at N=64),
3. **selection** (the remaining gap: learned AS3T .51@64 vs .89 ceiling).

**Learned arms.** AS3T with evidence-clause aux beats baseline at N=64 (.512 vs
.356) and dominates through N=32; the entity-mention aux (all 8 clauses naming C,
room-agnostic superset) fails at length as predicted — recall-safe but precision-
starved, repacked length grows ~ N. Attaching headers to LEARNED keeps (AS3T-H)
HURTS (.38@64, seed var .32-.85 at N16): the anchor trick amplifies false keeps
(each false clause drags a header in; e·N compounding), so it only pays when
selection is already precise. The open item for dense tasks is selection
precision at the token level, not geometry and not readout.

Ops notes: crowded loader in eval-only path fixed (was hardcoded prep_root_rel ->
KeyError 'clauses'); `--keep-heads` flag in train_coformer.py (OTOK keep + AS3T
learned-keep header attachment); clause spans recorded at build time in
prep_root_crowded (rec["clauses"]).

## Addendum 9 — Can selection be learned END-TO-END from task CE? An anatomy of no
### (differentiable position-repacking, arms AS4/AS4M/AS5; 2026-08-24)

**Motivation.** AS3's selector is trained by self-distilled attention supervision
(the model's own zero-shot yes/no judgments — label-free but a second signal).
The elegant target: CE alone. The historical blocker was gradient blindness —
masks are differentiable but POSITIONS are not, and positions carry the payoff
(.25 masked-in-place vs 1.00 repacked). So we built the missing gradient path:

**The estimator (new, verified):** `soft_repack_positions` — float positions =
prefix + cumsum of gate-weighted chunk widths (d pos / d gate != 0); `soft_rope`
— cos/sin computed WITH gradient (HF's rotary is @torch.no_grad; ours is
bit-parity-tested against it); `log_gate_mask` — log(g) column bias (the
-K*(1-g) OR-field is binary below g~0.9). Exact at the vertices g in {0,1}
(tested, all 8 keep patterns). Arms: AS4 = gates -> mask+positions (soft);
AS4M = gates -> mask only, positions detached (the blindness control);
AS5 = gates from an external N-invariant PairSelector, loss = CE + lam*mean(g).

**The failure ladder (each rung measured, one change at a time):**

| # | configuration | outcome | mechanism |
|---|---|---|---|
| i | AS4, gate init at AS3's threshold | CE pinned at ln9 forever; dev8 .09 | at INIT the L16 answer-row mass is UNIFORM (~0.16/n, rel==junk): the "retrieval head" is CREATED by AS3's aux, not found. All gates ~0.2 -> tail blind -> majority-only -> closed-gate self-reinforcing optimum (probe 398003) |
| ii | AS4 open-init (a0=-1) | dev8/32 = 1.0; gates NEVER close (a -> -1.5); transfer .16@64 | incentive starvation: keep-all achieves CE=0 at train lengths; the gradient path exists but carries nothing. AS4 == AS4M exactly |
| iii | + N=128 in the train stream | dev128 .988 UNDER KEEP-ALL; N16/32 collapse; .24@256 | shortcut absorption (readout): the LoRA absorbs any in-distribution length; selection only pays at absent lengths = no gradient by definition |
| iv | freeze readout (layers >= 16), body trainable | dev128 1.0, gates open, test == keep-all | shortcut absorption (body): layers < 16 adapt representations so the frozen readout works at N=128 keep-all — absorption capacity just moves |
| v | AS5: ALL LoRA frozen, external selector, CE + lam, lam in {.05,.2,3,6} | bit-identical to frozen keep-all at every lam | selector saturates open in ep1: at the parameter level CE's keep-more pull (head grad norm 73) dwarfs lam (0.15); per-GATE forces DO separate (below) but shared weights entangle them |
| vi | FREE per-(sample,frame) gates, lam in-band, 300 steps, N=128 | g(rel) .71 vs g(junk) .73 — NO separation; CE at soft init = 12.1 (uniform = 2.2) | **the interior of the relaxation is OOD**: a model trained on integer RoPE positions cannot decode fractional-position geometry; the soft interior is a loss plateau ABOVE uniform, not a bridge between vertices |

**Force-band measurement** (per-frame |dCE/dg| through the soft path, at init,
probe): N=128 evidence median 8.2 / junk median 0.63 (13x, bands disjoint at
p90); N=8 evid p90 3.6 / junk p90 0.14. The selection signal EXISTS in the CE
gradient — rungs v-vi show why descent still cannot use it: parameter
entanglement, and above all the interior-OOD barrier (vi).

**Conclusion.** Under a frozen(-ish) backbone, select-and-repack CANNOT be
learned by gradient descent through a continuous relaxation of the rewiring:
(a) any trainable capacity in the answer path absorbs the training length
first (iii, iv); (b) with absorption removed, the relaxation interior is
out-of-distribution for integer-position-trained models, so the two decodable
vertices (keep-all / select-and-repack) are separated by a measured loss
barrier (vi: CE 12.1 > uniform 2.2). Discrete-search or distilled supervision
is NECESSARY, not convenient — AS3's self-distilled aux (the model's own
zero-shot judgments; no gold labels) stands as the method. The estimator
remains valuable as an INSTRUMENT: the per-frame force measurement (only
definable through soft positions) is how we can see the selection signal that
descent cannot follow.

Escape routes for future work (not tried): (1) position-augmented training so
fractional geometry is in-distribution (make the interior decodable, then the
band argument applies); (2) discrete search — REINFORCE/Gumbel on HARD
configurations (forward always at a vertex); (3) ST-positions (hard forward,
soft backward at the vertex) — the backward is then a linearization AT a
decodable point, dodging the interior entirely.

Ops: arms AS4/AS4M/AS5 + SoftGate/soft_repack_positions/soft_rope/log_gate_mask
in scripts/condmask/{train_coformer,coformer}.py; probes AS4_PROBE/AS5_PROBE;
3 new CPU tests (vertex parity, gradient-through-positions, HF-rope parity).
Runs: outputs/as4, as4v2, as4v3, as4p2, as5_lam*, as5b_lam*, as5c_lam*,
_scratch/as4_probe, _scratch/as5_probe. Frozen bases: as4v2 (AS3-regime) and
outputs/coformer A2mix ckpts. NOTE: AS3-regime frozen base + no-pre-segment
forward is itself broken (dev8 .55) — regime consistency matters when freezing.

### Addendum 8b — AS3T ablations (2026-08-26; runs outputs/crowded_{half,heads,n8only})

| variant | N16 | N32 | N64 | N128 |
|---|---|---|---|---|
| reference {8,32}, full data | .958 | 1.000 | .512 | .265 |
| half data (1350+340) | .950 | 1.000 | .512 | .235 |
| + per-head column cuts (asth-style) | .996 | .970 | .308 | .213 |
| trained N<=8 only | .502 | .384 | .160 | .130 |

Three verdicts: (1) DATA IS NOT THE CONSTRAINT — half data is bit-identical at
N=64; AS3T sits at 92% of its measured no-anchor ceiling (.512 vs .556), the
rest is structural (anchors: ceiling .89). (2) HEAD GRANULARITY IS DEAD on both
task families — positions are per-token so heads can only mask, and per-head
cuts remove evidence from some heads' view (recall asymmetry): -20 pts @64.
(3) The minimal-training law is uniform across granularities: local clause-level
distractor richness at N=8 does NOT substitute for length — token selection
trained N<=8 collapses (.16@64, below baseline); one distractor-rich length
(N=32) is necessary and sufficient, same as frame-level.

## Addendum 10 — Distill-and-stop: the minimal learnable selector (AS6/AS7; 2026-08-26)

**AS6 (drift-back, mechanism vii).** Warm the PairSelector on pseudo-labels
(agreement with oracle 0.941; keep(dist) -> 0.000), then fine-tune end-to-end
with ST-at-vertex CE: by ep3 the selector is BACK AT KEEP-ALL and evals equal
frozen keep-all. Both vertices fit train CE exactly, the valley between them is
flat, and the residual tilt points open; mean-normalized sparsity (lam/F per
gate) cannot hold. End-to-end fine-tuning of a distilled selector is not just
useless — it is destructive. (Also: --lr-sel 0 kills the warm too — same
optimizer; use --sel-freeze-after-warm.)

**AS7 (the method that falls out).** Frozen A2mix backbone + 2-layer pairwise
MLP on layer-0 embeddings, trained ONLY by 3 warm epochs (~7 s each on cached
features) of BCE against the model's own zero-shot yes/no judgments + hard
repack at deploy. No gold labels, no CE through the model, no thresholds, no
pre-segment, no LoRA training.

| variant | N8 | N16 | N32 | N64 | N128 | N256 | N1024 |
|---|---|---|---|---|---|---|---|
| frozen keep-all | 1.00 | .97 | 1.00 | .56 | .27 | .18 | .12 |
| AS7b (warm data N<=32) | .90 | .88 | .91 | .87 | .83 | .63 | .12 |
| AS7c (+N=128 warm slice) | .93 | .87 | .91 | .78 | .80 | **.84** | .24 |
| AS3 (LoRA + persistent aux) | 1.00 | — | 1.00 | .995 | .92 | .81 | .63 |

Reading: distill-and-stop recovers most of AS3 at ~1/1000 the training cost and
even beats it at N=256 with number-range coverage; the ~.9 plateau is the
pseudo-judge's ~6% label noise (constant recall tax); the N=1024 gap is the
frame-number extrapolation limit of embedding-level features (coverage to 128
fixes 256, not 1024) — AS3's in-model attention selector extrapolates further.
The AS4-AS6 anatomy (Addendum 9 + mechanism vii here) is the justification:
under a frozen backbone, distillation is the ONLY stable way to place the
selector, and stopping there is not a compromise but the optimum of the
measured dynamics. Runs: outputs/as6_lam*, as7, as7b, as7c.

### Addendum 10b — Everything trained at N<=16 (2026-08-26; outputs/cap16/*, data seeds 301/302/311/312)

| arm (trained {8,16}) | N32 | N64 | N128 | N256 | N1024 |
|---|---|---|---|---|---|
| A2mix baseline | .64 | .30 | .20 | .20 | .13 |
| AS3 (frame-attention sel.) | .97 | .83 | .60 | .40 | .35 |
| AS7 (judge-distilled MLP) | .94 | .86 | .90 | .46 | .12 |
| AS3T dense (token-attention) | .24 | .13 | .13 | — | — |

THE SUPERVISION-LENGTH LAW ORDERS BY SELECTOR GRANULARITY:
judge-MLP (length-free, coverage-limited) < frame-attention (graded: ~linear
in supervision-length doublings; .83@64 between the .54 N<=8 floor and .995
N=32 ref) < token-attention (needs the full N=32; 16-cap ~= the N<=8 floor).
Mechanism: the zero-shot judge factorizes N away (per-frame yes/no, labels
N-independent; only frame-NUMBER token coverage binds — AS7@16 matches
AS7@32 through N=128, drops only past coverage); in-model attention selectors
need softmax calibration at long fan-in, and finer units need it more.

## Addendum 11 — What length-finetuning actually learns (frozen vs FT probe; 2026-08-26)

Probe: scripts/condmask/probe_ft_effects.py (job in logs/, json at
outputs/_scratch/ft_effects.json). Models: frozen / A2mix-{8,16} / A2mix-{8,32};
N=8..256; layer-16 answer row, native full attention. Metrics: EM, evidence
attention-mass share, effective fan-in 1/sum(a^2), PRE-softmax evidence-junk
score gap ("margin"), Jacobian evidence-influence share.

| model | N16 gap | N32 gap | N64 gap | N128 gap | N256 gap | EM knee |
|---|---|---|---|---|---|---|
| frozen | .17 | .47 | .80 | .85 | 1.22 | ~8-16 |
| ft-{8,16} | .44 | .85 | 1.09 | 1.02 | 1.15 | ~32 |
| ft-{8,32} | .46 | .95 | 1.33 | 1.41 | 1.57 | ~64 |

Findings: (1) K_eff (fan-in) is UNCHANGED by FT — dilution structure identical
frozen vs finetuned (~60->200 over N=16->256, all models); FT does not touch
over-squashing structurally. (2) FT learns a FIXED anti-dilution margin
(+0.3-0.5 pre-softmax inside the window, extending ~one octave past the cap);
the EM wall sits at ~2x the training cap for every model. (3) Constant evidence
mass against ~N junk needs margin ~ log N (+2.1 from 32->256); the learned
margin grows +0.6 — a constant chasing an unbounded requirement. In-range
success and out-of-range failure are the same phenomenon: a fixed intercept on
a growing demand curve. (4) Influence share improves multiplicatively and
persists past the window (ft32 .32 vs frozen .12 @256) yet EM collapses —
influence without mass concentration is insufficient. (5) EXTRAPOLATION LESSON:
score interventions by the N-SCALING of (required - available) margin, not by
in-range EM. Finetuning moves the intercept; select-and-repack zeroes the slope
(fan-in flat 5-10, required margin N-independent) — why FT extrapolates 2x and
repack 30x. Also explains: length-mixture dominance capped at its own window;
AS3@{8,16} landing midway between its 8- and 32-trained versions.

## Addendum 12 — Training AT N=1024: the calibration-budget law (2026-08-26)

A2mix {8,1024} (data seeds 321/322; outputs/ft1024; batch 2, 706 s/epoch = ~8x
the {8,32} cost): test EM 1.000/.594/.572/.692/.700/.753/.765 at N=8..1024.
Margin probe incl. ft1024 (probe_ft_effects.py; ft_effects_1024.json):

| model | gap@32 | gap@1024 | keff@1024 | mass@1024 |
|---|---|---|---|---|
| frozen | .47 | 1.65 | 857 | .055 |
| ft-{8,32} | .95 | 1.69 | 972 | .070 |
| ft-{8,1024} | .78 | 2.79 | 410 | .156 |

(1) The anti-dilution margin IS trainable to the trained length — gap 2.79@1024,
no ceiling yet; absorption reaches 1024 (.765, dev .90) though a generalization
gap opens (train CE ~0 vs test .77; at 128 absorption gave .99). (2) FIRST
structural movement from FT ever observed: ft1024 HALVES effective fan-in at
1024 (410 vs 857) — heavy length pressure concentrates attention, not just
margins; still 50-80x short of repack (5-10). (3) It is REALLOCATION, not
growth: ft1024's gap@32 (.78) < ft32's (.95), keff@32 worse than frozen ->
interpolation hole (.57@32, monotone rise toward the trained end), zero
extrapolation. THE LAW: finetuning aims a shared, finite calibration budget at
a window; the requirement grows with N while the budget does not stretch across
windows. A method extrapolates iff it makes the required calibration
N-independent — repack does by construction (oracle 1.00@1024 trained at N=8);
window-training cannot. In-range success at any single length (incl. 1024, at
quadratic cost) and out-of-range failure are the same phenomenon.

## Addendum 12b — CORRECTION to Addenda 11-12 attention metrics (2026-08-26)

**Bug**: pad_batch pads S to %8, so "row -1" is a PAD token whenever padding
occurred; the Add.11/12 mass/gap/keff numbers were the pad row's (maskless)
view. EMs, EM knees, and influence shares were always computed at the true
answer position and STAND. Corrected probe (true answer row, ans=seq-1):
outputs/_scratch/ft_effects_fixed.json.

| model | keff@32 | keff@128 | keff@256 | keff@1024 | mass@128 | EM@128 |
|---|---|---|---|---|---|---|
| frozen | 31 | 99 | 159 | 179 | .17 | .12 |
| ft-{8,16} | 15 | 24 | 50 | 54 | .30 | .23 |
| ft-{8,32} | 14 | 16 | 22 | 30 | .45 | .27 |
| ft-{8,1024} | 20 | 15 | 15 | 19 | .49 | .75 |

**REVISED MECHANISM**: finetuning DOES concentrate the answer row's attention
(fan-in ~13-30 nearly flat, vs frozen growth to 179) and puts high mass on
evidence — yet EM still collapses at ~2x the training window: ft32@128 has
fan-in 16 and evidence mass .45 and scores .27, while ft1024@128 with the SAME
fan-in/mass scores .75. The attention is equally good; only the POSITIONAL
RANGE the readout was trained on differs. The binding constraint is therefore
NOT answer-row attention dilution (Add.11's margin-intercept framing is
demoted to secondary) but DECODING THE AGGREGATE AT OUT-OF-WINDOW POSITIONS —
the geometry thesis isolated even more sharply: FT can fix the attention; it
cannot fix positional coverage; repack removes the need for coverage.

**Second consequence of the bug (important)**: answer_row_frame_mass — AS3's
own selection read — also reads row -1. Trained AS arms are SELF-CONSISTENT
(aux and eval read the same row: a learned end-position read, functionally a
CLS-at-end); all AS3 results stand, but the method description should say
"end-row read" not "answer-row read". The TRUE answer row turns out to be a
far better ZERO-SHOT selector (Addendum 13). answer_row_frame_mass now takes
`row=` (default -1 = legacy/trained; probes and training-free regimes pass
the true answer position).

## Addendum 13 — Training-free selection (2026-08-26; outputs/tfree/*, eval-regimes pseudo/attn)

**Question (user)**: can selection be effective with NO training? Answer: yes,
two ways, and the second rewrites the headline.

**(a) Zero-shot judge + repack** (eval-regime pseudo; the vision result's text
twin): frozen 1.5B readout .71/.71/.78/.73/.59 at N=8..1024 — length-FLAT,
capped by frozen short competence; with the existing {8,32} readout LoRA:
.79/.88/.87/.79/.66 — matches AS3 at 1024 with zero selector training.

**(b) Native-attention top-k (the user's thresholded-attention proposal,
done right)**: run the model to L16 under NATIVE attention, read the TRUE
answer row's (ans=seq-1, NOT the legacy pad row) per-frame attention mass,
keep the top k=16 frames (k = 2x the gold bound — N-independent, recall-safe),
repack, answer with the {8,32} readout. NO selector, NO aux, NO extra model:

| rule | N8 | N32 | N64 | N128 | N1024 |
|---|---|---|---|---|---|
| threshold a=0.5 | .26 | .39 | .73 | .96 | .33 (rec 1.0, 11% junk) |
| threshold a=2.0 | — | — | .29 | .53 | .93 (1% junk) |
| **top-16** | 1.00 | .89 | .91 | .95 | .80 |
| AS3 (trained) | 1.00 | 1.00 | .995 | .92 | .63 |

Threshold optima DRIFT with N (0.5@128 -> 2.0@1024: mass*n is not
length-invariant); top-k is the clean rule for bounded-evidence tasks.
Recall .97-1.00 through 128, .92@1024.

**The mechanistic punchline** (with the corrected probe, Add.12b): the
length-finetuned model ALREADY computes an excellent selection signal at its
true answer row (fan-in ~16, evidence mass .45 @N=128) — it simply cannot
DECODE the aggregate at out-of-window positions. Reading its own attention and
repacking closes the loop with zero additional training: the model knows what
to read; it can't read it where it is; move it to where it can.

**Literature position** (agent survey, 2026-08-26): closest prior = SnapKV
(question-window attention keeps, KV eviction) — known to need retrieval-head
restriction/value-weighting/sink-stripping and per-head budgets; our top-k at
the TRUE answer row + physical repack (not eviction) + measured recall against
known evidence is the differentiator. Genuinely-different portable signals for
future work: Quest-style pre-softmax upper bounds (recall-safe by
construction) and restart-at-question PPR over chunk-pooled attention (the
DIGL/GDC port; apparently an open novelty slot). Effective resistance and
curvature (Black et al., SDRF/BORF) port as bottleneck DIAGNOSTICS, not
selectors (their de-duplication prior is wrong for counting).

**Cross-model caveat (competence floor, now instrumented)**: SmolLM2-1.7B has
NO native retrieval signal at ANY layer (attn-regime scan: rec ~= dist
everywhere) AND a broken zero-shot judge (rec .19-.26) — AS3 trained on it
fails beyond N=16 accordingly. The layer scan is a cheap PREDICTOR of method
transferability before any training. Qwen family (0.5B excepted) passes;
battery cap16: 3B@cap16 ~= 1.5B@cap32 (bigger model needs less supervision
length).

### Addendum 13b — corrected OSQ probe on AS3 (cap-16 battery model; outputs/bat16/osq2)

AS3's own selective forward, true answer row, N=8..128 (steps):
kept 4.2-6.9 frames (flat); **fan-in 5.3/7.9/6.1/6.8/6.1 (FLAT)**;
**evidence-influence share 1.00/1.00/1.00/.94/.94 (FLAT)**.

The over-squashing triptych, all on corrected instruments (cf. Add.12b):
frozen fan-in 21->179 with influence .49->.10 (structural dilution);
finetuned fan-in ~14-30 concentrated but positions out-of-window -> EM fails
at 2x cap anyway; AS3 fan-in ~6 flat + influence ~1 flat + EM extrapolates.
Attention quality, positional coverage, and sensitivity are now separately
measured, and only the repacked model holds all three constant in N.

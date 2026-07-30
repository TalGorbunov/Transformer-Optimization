# Why vision-language models can't count, and how to fix it

> Whole-story framing + per-claim evidence table for the thesis / advisor conversation.
> Companion to `RESULTS.md` (run-by-run log). Literature placeholder at the end — a
> `deep-research` lit search is running; slot verified citations in when it returns.
> Drafted 2026-06-30.

---

## Abstract (one paragraph)

Decoder-only vision-language models perceive each frame fine but **aggregate them linearly**, so a
count drowns in **√N-accumulated per-frame noise**. The count is *present* in the representation
(linearly decodable, R²≈0.85) yet the model emits the wrong number — an **aggregation/readout failure,
not perception**. The governing variable is the **per-frame discriminability d′**
(signal-detection theory): the measured d′ and N *predict* the pooled-readout ceiling in closed form
(`2Φ(d′/2√N)−1`), with zero fitted parameters. The fix is a **per-frame nonlinearity (a sharp threshold/gate) applied
*before* the sum**, so per-frame errors cancel; paired with a **parameter-free extensive reduction (Σσ /
soft-OR)** it counts exactly and **extrapolates to unseen counts**, where every *learned* readout caps.
We demonstrate two working forms — a frozen model + attention-isolation mask + Σσ readout (0.95–1.00,
extrapolates), and a native LoRA that verbalizes the count (0.99 in-distribution) — and show the residual
OOD wall is the **learned number head**, not the aggregation. Framed through the GNN lens, this is the
**over-squashing** bottleneck (attention ≈ message passing into a fixed-width carrier), and the remedy is
a DeepSets-shaped aggregator read *directly* rather than through the model's saturating number geometry.

---

## 1. The phenomenon
Qwen2.5-VL collapses on simple visual aggregation — "how many frames was character C in room R?" (answer
0–8) drops from ~85% at 1–2 frames to ~20–30% at 8, and the *same* length-collapse appears on
distinct-rooms and co-occupancy. But the **per-frame evidence is extractable**: a linear probe reads "is
C in R in this frame?" at ~0.96. So the model *sees* each frame; it fails to *combine* them.

## 2. Diagnosis: the per-frame code is strong; the pooled readout is what caps
The count is **linearly decodable** from the last-layer rep (R²≈0.85) — yet the model says the wrong
number. Model each frame's rep/message as **two Gaussian clouds** (evidence vs non-evidence):
`m = μ + e·δ + ε`, `ε ~ N(0, Σ)`, count `g = Σe`. Two measured facts:
- **The naive axis understates the code.** Along the difference-of-means axis δ̂ the class separation is
  only ~0.33σ (the old "SNR 0.33") — but a linear classifier is free to *whiten* first (Fisher 1936), and
  its measured ~0.94–0.96 per-frame accuracy implies a whitened (Mahalanobis) discriminability
  **d′ = √(δᵀΣ⁻¹δ) = 2Φ⁻¹(acc) ≈ 3.1–3.5**. The old "0.96-vs-0.33 paradox" was an artifact of measuring
  along the wrong axis (Cohen's d on δ̂ vs Fisher d′); there is no paradox: **the per-frame code is strong.**
- **The count direction is tiny relative to the shared component** (‖δ‖ ≈ 1% of ‖μ‖). This explains why
  *magnitude/norm* readouts and the model's own (misaligned, cos ≤ 0.01) head miss it —
  `corr(‖S_evid‖, count)=+1.0` but `corr(‖S_all‖, count)=−0.14` — but it is **not** why a trained linear
  readout of the sum fails. That reason is §3.

## 3. Mechanism: the √N law, derived rather than fitted
**Linearity distributes over addition** → a linear readout of a sum *is* the sum of per-frame linear
scores. Each evidence frame moves the score by one fixed gap (∝ wᵀδ); the N per-frame noises add like a
random walk (std ∝ √N). So the **count-level discriminability is d′/√N**, and the accuracy of the *best
possible* readout of the pooled statistic (matched filter `w ∝ Σ⁻¹δ` + round-to-nearest, optimal for
Gaussian classes) is **closed-form**: `P(exact | interior count) = 2Φ(d′/2√N) − 1` (boundary counts are
one-sided, slightly higher). Two measured parameters (d′, N) fix the whole curve — no fitted collapse
curve, no "SNR ≳ 2" threshold.

**Zero-free-parameter checks (derived from logged runs; dedicated validation run pending — see E1–E5):**
- Question-first N-sweep (S_all linear decode): predicted **0.86 / 0.62–0.69 / 0.53–0.59 / 0.46–0.52**
  for N=2/4/6/8 vs measured **0.85 / 0.63 / 0.58 / 0.45**.
- Deployed room-carrier decode-then-count: per-frame AUROC 0.955 ⇒ p_err ≈ 0.115; independent ±1 errors
  with cancellation predict **≈0.47** vs measured **0.48**.
- One miss: predicted optimal-linear-on-sum at the carrier ≈0.40 vs measured 0.28–0.30 — consistent with
  **correlated cross-frame noise** (breaks the iid √N step) and/or probe power (n=400); probe E2 decides.
- The **old δ̂-axis story is refuted by its own numbers**: SNR/√N = 0.12 predicts ~0.15 accuracy, 3× below
  the measured 0.45; the whitened d′ predicts 0.46–0.52. The restatement is a correction, not a relabel.

The earlier collapse curve (acc rises 0.20→0.79 as δ̂-SNR 0.15→1.77 across crowding/depth/masking, fixed
N=8) stands as *consistent* evidence, but the load-bearing claim is now the parameter-free prediction.

**N-scaling revision (2026-07-11, supersedes the "d′ dilutes with N" reading — RESULTS [2026-07-11],
[2026-07-10f,i]):** measured to N=128, the joint carrier d′ does NOT dilute — it is FLAT at ~2.0
(1.97/2.12/1.98/1.93/1.59 for N=8→128) while behavior collapses 0.173→0.020. The collapse is carried
by two measured downstream mechanisms: (i) an **emission-range clamp** — the answer distribution is
N-invariant (mean ≈3) while ordinal signal survives (corr 0.75 to N=64); (ii) **decision-threshold
drift** — a frozen per-frame gate's FN inflates 0.09→0.99 with N from 1/N attention-mass dilution of
message MAGNITUDES (direction intact: AUC stays 0.88+; mass-normalization cuts the drift ~10×, and the
tally-bias identity bias≈N·FP−g·(FN+FP) is exact). Composing the closed form with the measured emission
clamp (rank remap, zero fitted parameters) reproduces the behavioral curve almost exactly
(law+clamp 0.186/0.106/0.069/0.054/0.039 vs measured 0.173/0.127/0.073/0.053/0.020), where the plain
law is 2–5× high. The √N law still governs what a linear readout of the pooled statistic CAN extract;
what the frozen model actually loses at scale is calibration and emission range, not supply.

**Correction to the naive over-squashing story (unified):** attention output is a convex combination — an
*intensive* statistic (a function of the empirical distribution of messages), which can encode the
fraction k/N but never the count. At fixed N that's a bijection, so **mean ≡ sum** (measured: both 0.44)
and the same d′/√N ceiling applies; across varying N it **confounds count with length** (measured: +24pp
OOD for sum vs mean; GAIN>1 helps at sl8). Evidence-only is easy because there the per-count gap is ‖μ‖
(huge), not ‖δ‖: `d′_evid ≈ ‖μ‖/(σ√g) ≈ 5 at g=8` — decodes at 1.000. One formula covers all regimes.

**Why no post-sum readout helps (sufficiency, not just DPI):** for Gaussian classes the matched filter is
a *sufficient statistic* of the sum — no nonlinearity after aggregation can beat linear + rounding.
Measured, as predicted: MLP ≈ linear on the aggregate (0.437 vs 0.458, readout_ceiling); MLP-after-sum
caps at 0.34 while the same nonlinearity per-frame *before* the sum reaches ~1.0.

## 4. The fix: a sharp per-frame gate *before* the sum
Threshold each frame's evidence to a clean 0/1 before summing → each frame commits to a decision at the
**full per-frame d′** (error ~4–11%) *before* noise can accumulate, and **per-frame errors cancel** (a
wrong-yes cancels a wrong-no). Evidence:
- soft gate `Σσ`: 0.45 → **0.73**; **sharp** gate: → **~1.0**;
- linear "amplifier" can't (caps **0.52**); **MLP-*after*-sum fails (0.34)** — the threshold must be
  per-frame, *before* aggregation;
- causal: a LoRA trained to sharpen drove **per-frame SNR 0.40→2.76** and accuracy with it; the
  decomposition proved the win came *through* the gate (`Σσ` 0.44→1.0), not a black box.

## 5. Three working solutions
- **Frozen + frame-isolation mask + fixed Σσ readout.** A block-diagonal attention mask (each frame
  attends only to itself + the query) recovers multipass-clean per-frame reps **in one forward**
  (extraction 0.94→0.99); a parameter-free Σσ counts. **1.00 IID / 1.00 OOD on clean extraction; 0.95 /
  0.82 crowded** — it *extrapolates*. (Crowded 0.82-OOD gap = extraction ceiling, not aggregation.)
- **Native LoRA (MLP + per-frame BCE).** The model itself learns the gate and **verbalizes at 0.99**
  crowded IID, mechanism-confirmed by the decomposition.
- **Retrieve-then-verify pipeline (frozen end-to-end, 2026-07-11 — RESULTS [2026-07-11l]).** One
  joint pass shortlists frames by a mass-normalized gate margin (the gate: one logistic trained
  once on the N=8 cache); isolated yes/no look-again passes verify the shortlist; the tally is
  rendered as a predicate-matched fact and the frozen model verbalizes it. **Exact-match
  0.880 / 0.807 / 0.707 at N=32/64/128** — 17–35× the frozen model, above full multipass, above
  the law ceiling for any linear read of the joint carrier (it never sums noisy messages: it is
  the gate-before-sum principle deployed at system level). Cost ≈ N+1 forwards (the diluted joint
  margins shortlist ~85% of frames — the honest miss; a k=2-chunk shortlist is the untested v2).

(The isolation mask *helps* the frozen-probe path but *hurts* the native LoRA — 0.99→0.89 — because
isolated reps are off-distribution for the frozen downstream. Two different tools.)

**Scope correction (2026-07-11, supersedes an earlier conflation — RESULTS [2026-07-11d]):** the
isolation-mask win is a QUESTION-FIRST, frame-rep result. At the *deployed frames-first carrier* the
same fence is null-to-negative (fenced message AUROC 0.920 < joint 0.956; carrier d′ 2.47→1.93; fenced ≤
joint at every N to 128). What DOES lift the deployed supply is full per-frame isolation — separate
forwards — which raises carrier d′ from ~2 to 7–8 in both modalities and both model families tested
(the **joint-context tax**, a ~3.2–3.6× d′ factor: Qwen image 3.6×, Qwen text 3.2×, InternVL 3.4×).
The tax is NOT carried by cross-frame attention edges (fence null), NOT binding format (Char@Room
rendering changes nothing, [2026-07-10c]), NOT legibility ([2026-07-08c]): it is a property of
processing N frames in one shared forward. Two mechanism probes then pinned it further
([2026-07-11g,j]): equal-mass attention renormalization during the joint pass recovers ~1–8% of the
gap (mass competition at the carrier hop REFUTED as the main cause), and the chunk-size sweep shows
the tax ONSETS at the first companion frame (d′ 8.08 alone → 3.37 with one companion → 1.98 at 32)
— the damage is done during in-context ENCODING of the frames, before any carrier read.
The campaign-3 dissection ([2026-07-11r,s]) then DECOMPOSED the tax: (1) a saturating long-context
component ≈1.9 d′ (an isolated frame text-padded to joint positions reads 5.3, independent of
padding length), and (2) a content-similarity interference component graded by the companion's
domain (gray/noise neighbors ≈ free at 6.4; patch-shuffled 4.4; other-scene 3.6; same-scene 3.3).
**Correction ([2026-07-12]) and final localization ([2026-07-12b]):** the "encoding interference"
reading was overstated — in the companion pairs the real frame came FIRST, so its k/v never saw
the companion; the content-graded drop had to enter through the carrier's READ. The query/encoding
2×2 (swap q_c and k/v independently between clean and joint forwards, one fixed rope geometry,
within-frame softmax so the mass term is excluded) settles it: a JOINT-context query collapses
per-frame d′ to the joint level even on clean values (1.59 vs clean×clean 4.79 @L16), while a
clean query reading JOINT-encoded values keeps most separability (3.14). **The joint-context tax
is chiefly QUERY-ROUTING CONTAMINATION**: q_c, shaped by the whole multi-frame context, mis-aims
the within-frame attention ŵ_j — with a real but secondary value-encoding component (~half the
gap, overlapping). This unifies every prior null: renorm (mass ≠ routing), fence (the corrupted
read travels legal edges), first-frame causality. The only working supply lever remains separate
or near-separate forwards; a clean-query re-read of one joint forward delivers k=2-level margins
(≈3.4) at ~2 forwards — an economics option, below the d′≥5 wiring bar.

**Final localization — the tax is an ADDRESSING-capacity limit, and its two halves are opposite in
kind ([2026-07-12g–i]).** Three further probes closed the mechanism. (1) *The query is frame-specific,
not merely "contaminated."* A per-frame query reading a **different** frame scores d′ 1.02 — WORSE
than the generic joint query (1.71) and far below its own frame (3.48); a content-free (frameless)
query scores 0.53. Monotone in frame-conditioning (frameless < joint < own): the query is a
"where-to-look-in-THIS-arrangement" program that is actively wrong on other frames, so no single
query serves many — the deployable "one probe query → read all N" fix is dead. (2) *The
value-encoding half is REVERSIBLE.* A small label-free adapter trained to map joint-encoded k/v →
isolated (multipass) k/v recovers ~98–100% of the clean-encoding d′ gap (3.13 → 4.4 ≈ the mp
ceiling): joint processing does not destroy per-token evidence, it applies an invertible mixing.
(3) *The query half is IRREDUCIBLE.* The analogous query un-mixer — reconstruct q^(f) from the joint
forward's state — FAILS: its output reads **below** the joint query (1.19 vs 1.71, −37%), and it fits
its training queries fine (R²≈0.65–0.85), so a query MSE-close to the target still routes wrongly.
**Synthesis: the joint-context tax = a reversible value-mixing + an irreducible query-capacity limit.**
A single fixed-width query vector cannot encode N frame-specific attention patterns, and that missing
addressing cannot be transferred, faked, or reconstructed after the fact — it can only be *created* by
attending each frame (a per-frame forward). This is **over-squashing at the query**: the bottleneck
squeezes the *addressing* (which tokens to attend), not the evidence content — the values are fine.
It proves the per-frame (multipass/chunk) forward is not a workaround but the floor: cleaning values
is free, but there is no one-forward supply fix, because the one thing joint processing cannot
preserve is which query reads which frame.

**Why exact-match still caps at large N (the crush line, [2026-07-11]).** Even at full supply, exact
counting needs the summed per-frame decisions to have net-zero error, so per-frame error ε ≲ 1/N:
at N=128 the ~1.5% verifier error gives √(Nε)≈1.4 counts of spread → exact-match ≈0.71 (retrieve-then-
verify), not ~1.0 — the tally-bias law N·FP − g·(FN+FP) biting as registered. This is intrinsic to
exact integer counting at scale (MAE stays small: within ~1–2 of gold), not a defect of the gate.

## 6. The remaining wall is the readout, not the aggregation
The native counter **memorizes**: train on counts 0–4 → **0.97 IID but 0.035 OOD** (never emits 7/8). Yet
its internal gate is *perfect and extensive* (`Σσ` 0.44→1.0). **The brain has the count; the mouth caps**
— the learned number head can't emit an unseen value, and the model's native number representation
saturates ~5. Exact split: read via the **learned head → 0.035 OOD**; via **direct Σσ → 0.82–1.0 OOD**.

**The token interface is necessary, not just sufficient (2026-07 campaign, RESULTS [2026-07-11b]).**
Writing the count into the prompt as a predicate-matched fact sentence verbalizes at 1.000 including
held-out two-digit values with zero parameters — but EVERY learned activation-level injection route
fails OOD at the same site: per-digit soft-token codebook 0.098, per-count codebook (+interpolation)
0.000, a continuous Fourier basis 0.000, and a Fourier basis anchored to the model's own digit-embedding
geometry 0.000 — all at 0.81–1.00 on trained counts. The frozen readout treats injected vectors as
memorized opaque symbols; it decodes no geometry. The interface itself is gated: the fact must use the
question's own predicate and precede the question (paraphrase or post-question placement → 0.00,
RESULTS [2026-07-10d]) — consistent with causal question-carrier aggregation. Emission-side, the same
wall appears as a **range clamp**: the frozen model's answer distribution is N-invariant (mean ≈3,
support ≲7) from N=8 to 128 while ordinal information survives to N=64 (corr 0.75, [2026-07-10f]).

## 7. Task-agnostic, and it generalizes
The **gate is universal** — the query supplies the predicate; per-frame labels are self-generatable
(look-again, 0.96–0.99). The **reduction is a small fixed choice** covering the family: **sum**
(occurrence), **soft-OR** (distinct), **position-weighted sum** (temporal — order rides in via the
positional encoding, so sum ≈ a sequence model). Validated OOD: **sum/soft-OR extrapolate
(steps 0.996 / rooms 1.000 / co-occ 0.974); every learned readout collapses (0.00–0.59).**

**And the wall is binding-specific — natural images without binding don't have it (RESULTS
[2026-07-10h]).** mmred_natural (COCO needles, per-frame GT by construction, judge-gated 0.998–1.00)
gives a d′ dial on natural images: needle diversity moves carrier d′ 6.2→4.3 and distractor similarity
6.2→5.4. On this rung — "does a dog appear?", no character→room binding — the frozen model RIDES the
closed-form law (measured 0.57–0.66 vs predicted 0.60–0.76): content addressing reaches the evidence,
and the MMRED-style gap between model and achievable vanishes. The wall lives where evidence is
relational, not where it is merely visual.

## 8. The thesis lens (GNN over-squashing)
Attention is message-passing; squashing N frame-messages into a fixed-width carrier is the
**over-squashing** bottleneck. The remedy — a **structured extensive aggregator with a per-element
nonlinearity (DeepSets `Σσ`)** — fixes counting, and reading the tally *directly* (not through the
saturating number head) is what makes it **length/count-extrapolate**, where learned aggregators fail.

**One line:** *the model perceives each frame but aggregates them linearly, so the count drowns in
√N per-frame noise; a sharp per-frame gate before an extensive sum recovers it exactly and
task-agnostically, and reading that sum directly (not through the saturating number head) is what makes
it generalize to unseen counts.*

---

## Per-claim evidence table

| # | Claim | Key number(s) | Where (script / output) |
|---|---|---|---|
| 1 | Aggregation, not perception | per-frame extract 0.96; base 85→20–30 | `probe_evidence_selection_*`; RESULTS exec summary |
| 2 | Count decodable but unread | R² 0.85 present; align to digit rows 0.008 | `probe_count_readout_alignment.py` → `count_readout_alignment/` |
| 3 | Count in a ~1% direction, low SNR | ‖δ‖/‖μ_all‖=0.011; SNR 0.33; corr(‖S_evid‖,g)=+1.0 vs (‖S_all‖,g)=−0.14 | `probe_aggregation_decomposition.py` |
| 4 | Linear sum can't threshold; ÷N benign | mean≡sum 0.439; linear-amp caps 0.52; sharp gate 0.09→1.0 | `probe_amplification.py` |
| 5 | SNR is the governing variable | acc 0.20→0.79 vs SNR 0.15→1.77 (fixed N) | `probe_snr_collapse.py` → `snr_collapse/collapse.png` |
| 6 | N-sweep: acc falls with N | S_all acc 0.85/0.63/0.58/0.45 (N=2/4/6/8) | `probe_message_sum_decodability.py` on `cache_ns_*` |
| 7 | Per-frame nonlinearity is the fix | sigmoid-then-sum 0.73; MLP-after-sum 0.34; linear 0.45 | `probe_nonlinearity_ceiling.py` |
| 8 | Isolation mask = multipass in 1 forward | per-frame 0.94→0.99; count 0.66→0.95 (1-char 1.0/1.0) | `frame_isolation_diagnostic.py` |
| 9 | Frozen + mask + Σσ extrapolates | 1.00/1.00 (1-char), 0.95/0.82 (crowded); learned-sum 0.11 OOD | `probe_mask_sigma_ood.py` |
| 10 | Native LoRA verbalizes; mechanism-confirmed | TEST 0.993; SNR 0.40→2.76; last-tok count 0.21→0.99 | `lora_sft_baseline.py` (mlp+BCE) + `decompose_reps.py` |
| 11 | OOD wall = the learned readout | native OOD 0.035 yet internal Σσ→1.0 | `lora_sft_baseline.py --holdout-counts 5,6,7,8` |
| 12 | Extensive readout extrapolates across tasks | sum/soft-OR OOD 0.996/1.000/0.974; learned 0.00–0.59 | `readout_benchmark/` (RESULTS 2026-06-23) |
| 13 | No in-place attention fix | GAIN (attention×N) null / hurts at high N | `denom_gain_vs_temp.py` |

> ⚠ Rows 3–5 quote the δ̂-axis SNR (0.33) and the fitted collapse curve; superseded by the whitened-d′
> restatement in §2–3 (RESULTS [2026-07-03b]). The *measurements* stand, but the operative per-frame
> quantity is d′ = √(δᵀΣ⁻¹δ) ≈ 3.1–3.5 and the law is the closed form, not a fitted monotone curve.
> Rows 1–6 are also **question-first** probe-layout numbers; the deployed frames-first analogs live at the
> room-token carrier (RESULTS [2026-07-03]) and the deployed-locus parity validation is pending (E1).

## Key figures to produce for the writeup
- **SNR collapse** (`snr_collapse/collapse.png`) — count acc vs per-frame SNR at fixed N (the law).
- **Extensive-vs-learned OOD** — bar chart: Σσ vs learned readout, IID vs OOD, per task.
- **Decomposition before/after** — SNR / δ-magnitude under the LoRA (causal confirmation).
- **Isolation-mask staircase** — per-frame extraction & count, joint vs masked vs multipass.

## Honest caveats (state these — they strengthen credibility)
- The decomposition/d′ account is **representational** (probing) with **causal support** (the LoRA
  intervention moves d′ and accuracy together); the closed-form law is derived under a stated model, and
  the model's assumptions are checkable (Gaussianity, iid noise across frames — E4/E2), not proven a priori.
- **Retracted earlier caveat:** "SNR is measured along the class-mean-difference axis — the axis a linear
  sum is restricted to" was **wrong** — a linear readout of the sum can use *any* direction, and the
  optimal one whitens (w ∝ Σ⁻¹δ). The δ̂-axis 0.33 understated the code; the operative d′ is ≈3.1–3.5.
- **Headline decomposition numbers are question-first**, not the deployed frames-first layout. The
  deployed locus is the room-token carrier (d′ map peaks L14–20); preliminary deployed checks obey the
  same law (dtc 0.48 vs predicted ≈0.47) but the dedicated deployed parity run (E1) is pending.
- d′ values inverted from probe accuracies assume Gaussian equal-covariance classes; the validation
  measures d′ directly (Ledoit–Wolf shrinkage Mahalanobis) on the caches.
- **Independence of per-frame noise (the √N step) is untested**; correlated noise gives
  √(N(1+(N−1)ρ)) — the "information-limiting correlations" refinement (Moreno-Bote 2014) — and is the
  leading suspect for the one parity miss (carrier sum-decode 0.28–0.30 vs predicted 0.40).
- The **answer-level count** (small N has fewer classes) is handled inside the closed form (boundary
  terms + level mixture), no longer a confound left to a fitted curve.
- High **IID** accuracies (native LoRA 0.99) are **memorization-suspect** — only the OOD/extensive results
  certify generalization.
- One task family (MMRED, synthetic). Cross-architecture / real-data replication is future work.

## Literature (verified via deep-research, 2026-06-30; 108-agent adversarial search)

### Must-cite, by pillar (arXiv IDs verified)
**Pillar 0 — signal detection / population decoding (the formal machinery behind d′; replaces ad-hoc "SNR"):**
- Green & Swets 1966, *Signal Detection Theory and Psychophysics* — d′ and its exact accuracy/AUC
  identities (bal-acc = Φ(d′/2), AUC = Φ(d′/√2))
- Fisher 1936, *The Use of Multiple Measurements in Taxonomic Problems* — whitened linear discriminant;
  Mahalanobis d′ = √(δᵀΣ⁻¹δ)
- Macmillan & Creelman, *Detection Theory: A User's Guide* — standard d′ practice
- Averbeck, Latham & Pouget 2006, Nat Rev Neurosci — linear Fisher information / population decoding
  (treating frame messages as a noisy population code is this, verbatim)
- Moreno-Bote et al. 2014, Nat Neurosci — information-limiting (correlated) noise: the ρ>0 refinement of √N

**Pillar 1 — GNN over-squashing (motivation; *formalized via Jacobian/curvature, NOT SNR*):**
- Alon & Yahav 2021, *On the Bottleneck of GNNs* — arXiv:2006.05205
- Topping et al. 2022, *…via Curvature* — arXiv:2111.14522
- Di Giovanni et al. 2023, *…width/depth/topology* — arXiv:2302.02941
- (cardinality-blind GNN pooling — arXiv:1907.02204)

**Pillar 2 — DeepSets & set capacity / extrapolation (supports the fix, claims 4–5):**
- Zaheer et al. 2017, *Deep Sets* — arXiv:1703.06114
- Wagstaff et al. 2019, *Limitations of Representing Functions on Sets* (latent-dim ≥ set-size) — arXiv:1901.09006
- Pooling-for-extrapolation: Ko 2021 (arXiv:2106.06210); Xu 2021 (arXiv:2009.11848) — **sum extrapolates, learned pooling fails OOD**

**Pillar 3 — transformer counting, softmax limits, readout, superposition:**
- Yehudai 2024 (arXiv:2407.15160), Behrens 2024 (arXiv:2407.11542) — counting capacity vs embed-dim/vocab (**different mechanism**)
- Veličković 2025, *Softmax is not Enough* — arXiv:2410.01104 (dispersion for **sharp/selection**, not extensive counting)
- Brothers 2025 — arXiv:2506.09215 (SNR-fragile pooling, but **distractor-fraction** SNR, not √N)
- arithmetic length-gen over operand count — arXiv:2410.15787
- Garcia 2026, *The Right Answer, the Wrong Direction* — arXiv:2605.03258 (**count decodable, ⊥ output head**; 2026 preprint, not peer-reviewed; head-fix fails under free generation)
- Elazar et al. 2021, *Amnesic Probing* (TACL) — decodable ≠ used/steerable
- Elhage et al. 2022, *Toy Models of Superposition* (transformer-circuits.pub; arXiv:2209.10652) — low-norm interfering directions

### What is genuinely ours (claim as the thesis's contribution)
- The **closed-form d′/√N ceiling for pooled linear readouts inside a frozen VLM**, with its
  zero-fitted-parameter empirical validation (predicted-vs-measured parity across N/crowding/tasks).
  The machinery is textbook signal detection / population decoding; applying it to transformer
  aggregation and *predicting* the measured ceiling is ours (over-squashing lit uses Jacobian
  sensitivity, Brothers uses distractor-fraction SNR — neither derives an accuracy law).
- The **sign-vs-magnitude / error-cancellation** argument (per-element threshold *before* the sum).
- **OOD-via-direct-extensive-read vs learned-head-capping**, demonstrated for a **frozen VLM**.
- The **cross-domain synthesis** (over-squashing ↔ cardinality-blindness ↔ readout-misalignment) under one SNR account.
- **The query/encoding dissection of the joint-context aggregation tax** ([2026-07-12]): decomposing the
  per-frame attention message into routing (query) vs values (encoding), and showing by independent
  interventions (2×2 swap, frameless/donor query transfer, and joint→clean un-mixers for each half)
  that the tax is a **reversible value-mixing plus an irreducible query-addressing-capacity limit** —
  "over-squashing at the query." The value half is recoverable by a learned map (~100%); the query half
  cannot be transferred, faked, or reconstructed from the joint state — a one-query-cannot-address-N-sources
  result localized inside a deployed VLM, proving per-frame forwards are the supply floor. (Lit check
  for this specific mechanism in flight — position against attention-capacity / associative-memory and
  retrieval-head work.)
- **VLM / multi-frame visual counting failure as an empirical regime is UNSOURCED** — the 85%→20–30% length-collapse and per-frame d′≈0.33 appear genuinely uncovered; claim as our empirical contribution (and do a dedicated CVPR/ICCV/NeurIPS video-VLM benchmark search before final write-up).

### Caveats from verification (do NOT over-claim)
- **Don't launder** Yehudai/Behrens "transformers can't count" as SNR evidence — different (embed-dim-vs-vocab) mechanism; cite for the general theme only.
- Veličković dispersion = **sharp tasks**; the "→ count-blind" step is *our* bridge.
- Garcia & the steering preprint are **2026, not peer-reviewed** — supporting, not load-bearing.
- **Three claims were refuted/over-reaching:** (1) the universal `ρ(Σφ(x))` *structural* form — cite Wagstaff's bound + permutation conditions instead; (2) "R²>0.99 ⇒ readout failure" — *our* R² is **0.85**, the 0.99 is Garcia's *text* case; (3) the single-source medical-LLM uncorrectable claim. Keep wording careful around these.

### Open question for the write-up
No prior work found on **video/multi-frame VLM counting or temporal-aggregation failure** specifically — needs a dedicated benchmark search; if none exists, this regime + the SNR/√N law are the thesis's distinct contributions.

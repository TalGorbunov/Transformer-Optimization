# Independent assessment of this codebase (2026-08-17)

> Written by Claude Code after a full read of the repo: `METHOD.md`, `RESULTS.md` (all 1103
> lines), `docs/archive/RESULTS_pre_fencing.md` (arc), the three `docs/theory/` pages, the
> Aug-06 lit sweep, the `gnnformer/` package, all `scripts/` workstreams, every tracked
> `outputs/*/{INDEX,STATE,CAMPAIGN_BRIEF}.md`, and the git history on both branches
> (`master`, merged `serious-refactor`). All numbers below trace to those files.
> Purpose: an outside view on (a) whether the project's "the bottleneck is the readout"
> framing is earned, (b) whether the Gaussian/d′ theory is fit for purpose, and (c) what a
> method for **single-forward-pass evidence aggregation** should build on from here.

---

## 0. Verdict in five sentences

1. The experimental record is unusually honest and self-correcting, but its **headline
   framing ("supply fixed, aggregation solved externally, the frozen READOUT is the wall")
   is not supported by its own strongest evidence** — it rests on oracle readouts
   (supervised sklearn probes) that quietly bundle the aggregation computation into the
   "readout" they claim to be testing.
2. The **Gaussian/d′ model is descriptively useful as a separability meter but fails as a
   quantitative law**: its independence assumption is contradicted by its own measured
   ρ≈0.1, its "ceiling" is violated by the model itself at N=2, its behavioral "closures"
   predict numbers a majority-class emitter also produces, and by August the project was
   quietly operating on a *different, incompatible* law (fan-in ~4, accuracy halves per
   doubling ∝ 1/N, vs the d′ law's ∝ N^(−1/2)) without ever reconciling the two.
3. There is a traceable causal chain from the theory's over-claims (post-sum sufficiency,
   the readout wall, the requirement curve) to the project's worst practical failures —
   most notably the caption scratchpad, whose teacher-forced metric turned out to be a
   **copy detector** (0% true counts at N=32) and which by design **bypasses the very
   bottleneck the thesis is about**.
4. The genuinely load-bearing discoveries in this repo are **not** the Gaussian law and not
   the scratchpad: they are the **fence/supply repair** (real, causal, cross-family), the
   **read fan-in capacity law** (~4 items per softmax read, length-invariant), the
   **re-quantization law** (frozen attention moves vocab-coded content losslessly, analog
   state codes lossily), the **emission deadline** (~L20–24; compute and emission share one
   clock), and **sum-vs-mean extensivity** (+24 pts OOD) — all of which are aggregation
   mechanics, not readout mechanics, and all of which map cleanly onto over-squashing
   ideas (bounded-degree rewiring, virtual nodes, discrete message alphabets).
5. The current active direction (learnmask) is, by the repo's own S0 result, searching a
   space (attention topology) that provably contains no solution; the right target is a
   **bounded-fan-in, staged, in-model aggregation architecture with learned
   re-quantization at internal nodes** — which the repo's own evidence base already
   specifies almost completely, and which is exactly the "single forward pass" goal.

---

## 1. What this repo is (state, branches, chronology)

**State of this checkout.** Code-only clone (all mtimes 2026-08-17). No `.venv`, no
`data/`, no run dirs; **all 22 `checkpoints/` symlinks dangle** (they point into absent
`outputs/`/`outputs_legacy/` trees). Nothing here can re-run or be independently verified
until those are restored. `outputs/learnmask/STATE.md` and `CAMPAIGN_BRIEF.md` are cited
by code and sbatch headers but **do not exist in the repo** — the only record of the
current campaign's results is the `scripts/learnmask/s0_free_table_exp.py` docstring.

**Branches.** `master` and `origin/serious-refactor` (fully merged 2026-08-13; nothing
stranded). The real "branches" are research workstreams:

| workstream | dates | status | one-line verdict |
|---|---|---|---|
| carrier/fencing (THE method) | Jul 14–30 | substrate | headline 0.987 @N=32; but see §3 |
| presentation_diagnostics | Jul 27–Aug 1 | closed | fence before/after instruments; super-carrier NO-GO |
| mmred_hf (published benchmark) | Aug 1–5 | closed, v8 unlaunched | frozen < generative < external heads; "carriers transport, readout doesn't extract" |
| superquery (prof's tree readout) | Aug 5–6 | closed | fan-in law, requant law, deadline; two-pass 0.980 |
| gating (arXiv:2505.06708) | Aug 7–11 | closed, negative | no sink in the 7B; plain LoRA beats all 5 gate positions; **copy-detector found here** |
| ninv (N-invariance, register tree) | Aug 9–11 | closed | posreset closes cross-N leak; depth does not extrapolate; LAW 7 |
| herbench_retrieve_opt | Aug 2 | NULL, phase 3 gated | real-video perception ceiling, not aggregation |
| **learnmask** | Aug 12–15 | **ACTIVE** | S0: 0 flips of 6.7M gates; latest figure title: learned mask ≡ full-open |

**Chronology in one paragraph.** Feb–Apr: bottleneck located (last-token collapse,
"more true evidence hurts"). June: gLSTM → DeepSets → two causal knobs (sum-vs-mean +24
pts OOD; width knee at d_mem = max count); decrowding dominates. July: the d′ framework;
the "readout wall" battery; then the fencing turn — supply d′ 2→13.5, external gate→tally
0.998, scratchpad readout, headline 0.987 @N=32. August: every attempt to make the
*model itself* aggregate (gating, tree registers, learned masks) returns null or
plain-LoRA-wins; the headline metric is exposed as a copy detector; the project pivots to
"prove no mask helps → therefore readout" (learnmask), which is where it stands at HEAD.

---

## 2. The method as deployed (what actually happens)

Recipe (METHOD.md): `[question][frame_1][carrier_1]…[frame_N][carrier_N][question]`,
block-diagonal fence on layers 0–11 with per-block M-RoPE reset, one distilled 3,584-param
carrier embedding, LoRA r8 (~2M params) on layers 12–27, and a **caption-scan scratchpad**
readout (`scan: f1:- f2:Kitchen(1) … | total: 2 END`) decoded greedily and parsed by regex.

Three facts about the deployed system that the headline story omits:

1. **The fence never fences the readout.** `fencing.build_block_mask` leaves tail/decode
   rows attending **all raw frame tokens** (`fencing.py:36-38`, pinned by
   `tests/test_fencing.py:103-111`). The TRUNC campaign proved the decode uses them:
   dropping frame KV changes 15/16 transcripts; eval-time truncation to carriers-only
   collapses 0.987 → 0.040, and deploy-matched retraining also fails (≤0.153)
   (RESULTS.md 2026-07-25). So the headline numbers are produced by a model that at every
   decode step re-reads all ~12,775 frame tokens. The carrier channel and the decode
   channel are, in the repo's own words, "separate channels" — but METHOD.md §2 and the
   README still describe the carriers as the delivery mechanism.
2. **The aggregation in the headline numbers is serial text generation, not a forward
   pass.** The running tally means the model never holds a count in its residual stream;
   it increments a number that is visible in its own context window. METHOD.md §6 says
   this openly ("the aggregation happens serially in the model's own generated text").
   The gating campaign's postmortem says it more sharply: *"the caption scratchpad exists
   precisely to BYPASS in-model aggregation … we were testing whether gating relieves an
   aggregation bottleneck using a method engineered to avoid that bottleneck"*
   (outputs/gating/STATE.md, 2026-08-09).
3. **The scaffold "ceiling" (0.998) is sklearn.** `scripts/gate_tally.py:52-56`: a
   supervised `LogisticRegression` per-frame gate (trained on gold per-frame labels,
   50/50 split *within* the eval cache) plus `numpy` sum. This is the number most often
   quoted as "aggregation is solved once supply is clean." It shows the per-frame
   information is linearly present; it does **not** show the model can aggregate it —
   the threshold-then-sum program is executed entirely outside the transformer.

Baseline context the exec summary underplays: at N=32, caption scratchpad 0.987 vs
**plain in-length LoRA SFT 0.967** (RESULTS.md P4.1, "simple-fix-wins band") vs
**plain LoRA + multi-digit readout 0.942** (gating P8). The method's margin over the
simplest trainable baseline, measured where it's best measured, is ~2 points.

---

## 3. The "readout is the wall" thesis, audited

### 3a. What the record actually shows, sorted by what it tests

Separate three failure sites that the repo's binary ("aggregation vs readout") conflates:

- **(i) Representation** — is the per-frame evidence present in the states?
- **(ii) Computation** — can the frozen circuit *reduce* N pieces of evidence to a count
  somewhere in the network?
- **(iii) Interface** — can a count that has been computed be *emitted* as tokens?

| evidence | tests | result |
|---|---|---|
| fence/carrier d′ 13.5, probes 0.99+ | (i) | solved by fencing — real |
| gate_tally 0.998; mmred_hf arm B 0.678 @N=128 flat | (i) + oracle (ii)+(iii) | supervised external program solves it — says nothing about in-model (ii) |
| dose-response: ×64 amplified signal decodable 0.78, emitted 0.13 | (iii) given oracle (ii) | frozen head can't read a *novel, probe-defined* axis — expected, see 3b |
| digit-LoRA readout (gating P7/P8): 0.942 @N=32, 0.767 @N=64, 0.157 @N=128 | (ii)+(iii) with trained interface | **interface is cheap to fix; the residual N-collapse is (ii)** |
| textcount: perfect clean digit tokens, no vision — 0.550 @N=8 → 0.183 @N=128 | pure (ii) | frozen model cannot tally even in its best modality |
| superquery fan-in law: exact to fan~4, halves per doubling, length-invariant | pure (ii) | a genuine capacity law on one softmax read |
| super-carrier NO-GO (0.25→0.07, unchanged when softmax restricted to carrier keys) | pure (ii) | one query cannot address N items |
| emission deadline ~L20–24; per-frame gate *written* in L12–24 | (ii) timing | compute and crystallization share one clock — a depth budget |
| S0 learnmask: 0 flips of 6.7M gates | (ii) via routing only | routing alone cannot create the missing computation |

Reading the table: (i) is solved; (iii) is solved by a 2M-param LoRA whenever the training
data covers the range (the P7/P8 rows); the thing that has *never* been solved in this
repo — and collapses every time it is honestly measured — is **(ii): in-model reduction of
N items in one forward pass**. That is the aggregation bottleneck, and it is exactly the
thesis topic. The "readout wall" label survives because the instruments that "solve
aggregation externally" (gate_tally, ridge/logistic heads, the tally text) are all
supervised programs that perform (ii) outside the model, making the residual failure look
like interface-only.

### 3b. Two specific over-readings

**The misalignment "wall" is chance geometry.** The celebrated cos ≈ 0.005 between the
model's native reading axis and w* (and cos ≤ 0.01 to all digit unembedding rows) is
statistically indistinguishable from two random directions in d=3584 (E|cos| ≈ 0.013,
σ ≈ 0.017). There is no anomalous misalignment to explain: a frozen model has no reason to
read a code it never learned. Also note the "glasses" arithmetic doesn't reproduce its own
number (2.47 × 0.005 = 0.012, not the quoted d′ 0.51 — the 0.51 comes from a separate
held-out fit). "The readout is misaligned" reduces to "the model was never trained on this
code" — true, shallow, and fixable with a LoRA, as P7/P8 then demonstrated.

**S0's conclusion doesn't follow from S0.** The docstring concludes the wall is "the
frozen READOUT, not the attention topology." But an attention mask is a routing change; the
operation the repo's own theory says is required (R1: a per-item *nonlinearity* before
pooling — the threshold gate) cannot be expressed by any mask. S0's null is therefore
predicted by the project's own T4/DeepSets analysis and localizes nothing between
"readout" and "missing per-node computation." What S0 does establish, usefully: pure
topology search is a dead end, and the S1/S2/S3 sweeps now running are searching a space
known to contain no win — the latest committed figure title already concedes it
("Learned mask transfers like the full-open topology it was pruned from", i.e. the learned
mask converges to *no fence at all*).

### 3c. Evidence on the other side (kept honest)

The readout story is not empty: the dose-response battery (decode ↑ while behavior ↓, with
clean scrub controls), the token-interface necessity result (activation injections 0.75–1.0
in-range / ~0.00 held-out vs digit tokens 1.000 zero-shot), and the InternVL replication
are real and causal. But note what they jointly establish: *a frozen model cannot exploit a
novel activation-space code without training.* They do not establish that fixing the
interface unlocks aggregation — and when the interface *was* fixed (digit-LoRA), the
N-collapse remained (1.000 @N=8 → 0.123 @N=128). The wall that stays after the readout is
repaired is the wall.

---

## 4. The Gaussian/d′ model, audited

### 4a. Where it fails on its own data

1. **Independence is assumed against the evidence, post-hoc.** The √N step needs ρ=0;
   measured joint-pass ρ ≈ +0.09–0.13. The correlated-noise correction (variance ×13.7 at
   N=128) exists in the docs and is *declined* because "measured accuracies track the iid
   prediction better." That is a model choice made by looking at the target — a hidden
   fitted parameter in a framework whose selling point is "zero fitted parameters." Every
   long-N prediction, and every architecture "priced from the law," depends on this choice.
2. **The "ceiling" is not a ceiling.** Table 1, N=2: law 0.747, model 0.889 — a 14-point
   violation of "the ceiling of what any readout could extract," never confronted. The
   model is not a linear functional of a single layer's message sum (12 more layers of
   MLPs and full KV access), so the bound doesn't apply to it — which also undermines every
   "model < ceiling ⇒ separate readout wall" inference.
3. **The behavioral closures are at the prior.** The flagship parity checks (native axis
   0.17 pred vs 0.21–0.24 meas; InternVL 0.135 vs 0.117–0.137; text 0.163 vs 0.165) all
   sit at/near the majority baselines printed on the same pages (0.13–0.24; "prior-locked
   from N=16 on"). A constant emitter reproduces these numbers; predicting them is not a
   test. And the native axis d′ is itself behavior-derived, so the closure runs from
   behavior to behavior.
4. **The CLT self-defense is refuted by the project's own audit.** objections_qa argues
   large N Gaussianizes the sum "precisely where the law matters"; the E4 map measures the
   opposite (excess kurtosis +0.52 @N=8 → +25 @N=128, FAIL). The long-N rows quoted for
   architecture pricing come from the regime where the model's own adequacy audit fails.
5. **The heteroskedastic case is never written.** Their own measurements give class-std
   ratios 1.4–2.2 at long N (10.8 on text-CWE); under unequal variance the count bumps
   have g-dependent widths and the midpoint-rounding decoder — hence the accuracy formula
   — is simply wrong. E4 gates which rows may be quoted, and the gate is outcome-correlated
   ("the regimes that miss are exactly those where the probe saturates").
6. **Key structural assumptions are false for a transformer.** Additive identical δ per
   item (softmax mass makes per-item contributions depend on all other items — their own
   "1/N dilution" finding contradicts the shift-family model); exchangeable ε across slots
   (frame position is 0.988-decodable *from the same messages*, so a large slot-structured
   component lives inside "noise"; lost-in-the-middle ⇒ δ varies by slot); linear readout
   of S (everything downstream reads RMSNorm(h+S), a ratio — the docs use normalization
   for the magnitude-code argument and then ignore it in the ceiling arithmetic);
   "slope 0.19 ⇒ fraction reader" (slope attenuation toward the prior mean is what *any*
   shrinkage estimator produces under low SNR, including their own ridge decoder — this
   fingerprint has ~zero evidential value).
7. **Two incompatible laws now coexist.** July: acc ≈ 2Φ(d′/2√N)−1 ∝ N^(−1/2). August
   (lit sweep + superquery, the operative one): exact to fan ~4, then **halves per
   doubling** ∝ N^(−1). At N=32–128 these disagree badly; both were used to price
   architectures; no document reconciles them. Meanwhile the exec summary's mechanism
   sentence ("supply dilutes like d′/√N") is contradicted by B1's measured *flat* joint d′
   ≈2.0 across N — the project's own log flags this (item 22 of its contradiction list).
8. **Where it was tested out-of-family it lost.** Refuted leaf prediction (ninv: predicted
   ~0.90, measured 0.995–1.000); rejected on HERBench, text-CWE, VNBench, MLVU (honestly);
   the cross-architecture "clamp" prediction — half of the zero-parameter behavioral
   closure — is logged as **REFUTED as-released** (the clamp is a Qwen-VL-specific
   pathology, not a law).

### 4b. What it was good for

Fairness requires saying this: d′ as a *descriptive separability meter* did real work — it
detected the supply repair (2→13.5, cross-family), ranked interventions, and the E4
adequacy audit caught its own failures loudly. The R1–R3 "requirements for any cure"
(per-item nonlinearity before pooling; extensive not intensive reduction; a trained
readout) are correct and important — but note they follow from the DeepSets/GIN analysis
and the June ablations, not from Gaussianity. The Gaussian superstructure — the accuracy
formulas, ceilings, sufficiency slogans, requirement curves — is the part that failed, and
it is exactly the part that drove strategy.

### 4c. The causal chain from the theory to the practical damage

1. Gaussian sufficiency ("no post-sum function can help") + the readout-wall arithmetic
   ⇒ declared in-model aggregation futile and the token interface the only viable channel
   ⇒ **the caption scratchpad**.
2. The scratchpad's running tally ⇒ the TF metric became a **copy detector** (at N=32 the
   control emits the true count 0% of the time and follows a corrupted tally 85%;
   verified 120/120 at N=128) ⇒ the entire P3/P3.5 gating comparison void; the same
   artifact had appeared twice before (P3a natural TF 0.996 vs tf-exact 0.187; mmred_hf
   TF 0.941 vs 0.094) without being generalized.
3. The requirement curve ("no post-processing of a d′=2.4 gate can cross the 128-crush
   line; any long-sequence claim requires the multipass read") ⇒ licensed multi-pass and
   serial designs, de-prioritizing single-forward architectures — the thesis goal.
4. The P4.2 "single-token readout cannot use the same data" result (0.333 @N=32) — a
   pillar of "readout expressivity is the separator" — was an artifact of a digit path
   that **skipped gold > 9**; the corrected readout scores 0.942. The scratchpad's
   measured marginal value over a digit readout is ~+0.05 @N=32, not +0.65.
5. The d′-gated diagnostic ladder routed campaign after campaign into
   supply-vs-readout dichotomies, so the genuinely new aggregation laws (fan-in,
   requantization, deadline) arrived late, from a side campaign (superquery), and still
   haven't displaced the framing in METHOD.md/README.

So the user-hypothesis under review — *"the Gaussian model is overly simplistic and the
root cause of many issues"* — is **substantially confirmed**, with one amendment: the root
cause is not Gaussianity per se but the promotion of a descriptive separability statistic
into a prescriptive law (ceilings, sufficiency, requirement curves) that then dictated
method design and metric design. And the twin hypothesis — the repo is biased toward
"readout" — is confirmed in the specific sense of §3: the oracle-readout instruments
guarantee that whatever the model can't do gets booked as "readout."

---

## 5. What survives — the real findings to build on

These are the results I would treat as load-bearing for a new method. Each is causal or
replicated, and none depends on the Gaussian law:

1. **Isolation/fencing repairs supply** (d′ 2→13.5 park; 3.5× InternVL; Jacobian
   cross-frame 0.52→0.00; same-readout before/after 0.47→1.00 @N=8, 0.08→0.99 @N=128).
   GNN reading: cut interfering edges; PCW is the published ancestor.
2. **One softmax read has a hard fan-in capacity ~4** (exact to fan 2–4, halves per
   doubling, *length-invariant*; super-carrier NO-GO shows restricting keys doesn't help —
   it's the single fixed-width query, not key competition). GNN reading: bounded degree.
3. **Frozen attention transports discrete/vocab-coded content losslessly and analog state
   codes lossily** (patch experiment: digit-token embeddings 1.000 vs raw states 0.43 per
   hop; LAW 7: states that decode to the same digit are not interchangeable). GNN reading:
   messages need a discrete alphabet; re-discretize at every hop.
4. **Emission deadline ~L20–24, and it follows processing count, not layer index** —
   a frozen forward has a fixed serial-compute budget; values computed too late can't be
   emitted. This, not the interface, is the real "single forward pass" constraint.
5. **Sum vs mean is causal** (+24 pts OOD; GAIN helps, temperature doesn't; intensive
   normalization end-to-end). GIN's sum-aggregator lesson, demonstrated in a frozen VLM.
6. **Position carries N-dependence; canonical per-node position reset confers length
   invariance** (ninv: cross-N transfer 0.32→0.998; trained components length-invariant
   under it) — but **depth does not extrapolate** (untrained tree levels are dead under
   every mechanism tried).
7. **Per-frame detection is not the limit on synthetic data; it is the limit on real
   video** (HERBench NULL) — domain transfer is a perception problem, out of scope for
   aggregation work.

## 6. Cheap falsifications and missing baselines (run these before building anything)

1. **The digit-LoRA control is the true baseline** for every claim: plain LoRA +
   multi-digit answer-position readout, in-length. It already scores 0.942/0.767 @N=32/64.
   Any proposed aggregation architecture must beat *this*, not the frozen 0.219.
2. **Never-run 5/5-rated baselines from the project's own lit sweep** (all cheap):
   **APE** (parallel-encoding attention rescale — the closest published fix to the
   one-reader bottleneck), **PCW** as the no-tree fence baseline, **adaptive softmax
   temperature** (Veličković et al. — a one-file sdpa patch aimed exactly at dispersion),
   and Barbero's **separator-token insertion**. If any of these moves the N-curve, the
   fence/carrier machinery is over-built.
3. **Run `probe_tally_copy.py` on the headline caption exam** (free decode). The TF copy
   result doesn't automatically transfer to free decode, but the l12v2→caption jump
   (+0.19–0.37 from changing only the gold *text*) is suspicious in the copy direction;
   nobody has measured how much of 0.987 is increment-from-visible-prefix.
4. **A `budget=∞` decode control** for the cap-adjusted cells (the adjustment is worth
   ~+0.09 and always favors the method).
5. **Reconcile the two laws**: fit acc(N) at fixed trained readout across N∈{8..128} and
   test N^(−1/2) vs N^(−1) vs gate-law forms. This decides whether anything of the d′
   superstructure is worth keeping quantitatively.
6. **Re-derive the "ceiling" with the correct object**: the model's usable information at
   the answer position is not a linear functional of one layer's message sum; if a bound
   is wanted, it must be measured (trained-readout family, held-out), not derived from
   two-Gaussian arithmetic.

## 7. Direction for the actual goal: single-forward-pass aggregation

The goal ("improve Transformers at aggregating many pieces of evidence in one forward
pass, inspired by over-squashing") is precisely failure site (ii) of §3a, and the repo's
surviving findings dictate the design almost uniquely:

**Design implied by findings 1–6:** a *staged, bounded-fan-in aggregation tree inside one
forward pass* — fenced per-unit encoding (finding 1); merge nodes with fan ≤4 (finding 2);
a **learned re-quantization at each merge** so inter-node messages live in a discrete,
input-embedding-like alphabet (finding 3 — this is the component that has never been
properly trained in-model; the repeater tree did it with sklearn+Python, LAW 7 says naive
distillation of it fails because merges read the full state, not the token projection);
canonical per-node positions for N-invariance (finding 6); all merges scheduled to finish
before ~L20 (finding 4); an extensive (sum-like, unnormalized) combine at each node
(finding 5); and a trained digit readout at the answer position (P7/P8 show this part is
cheap). Depth-generalization is the known open problem (ninv: untrained levels dead) —
the layer-looping result (+17 pts, "deadline follows processing count") suggests weight
sharing across tree levels as the natural attack, mirroring how GNNs share the same
message function across hops.

**In over-squashing vocabulary** this is: graph rewiring from the complete graph to a
bounded-degree tree (Alon–Yahav), virtual nodes as aggregation registers, GIN-style sum
aggregation with per-node nonlinearity, and a discrete message alphabet to survive
multi-hop relay — i.e., the thesis's GNN bridge, executed *in* the forward pass instead of
in generated text.

**What to stop doing:** topology-only search (S0 already answered it); gating (closed,
negative); pricing designs with the d′ requirement curve; quoting TF metrics or the
sklearn scaffold as evidence about the model; treating the scratchpad as an aggregation
method rather than as the serial-fallback baseline it is.

**The honest thesis reframe available here:** "we located the aggregation bottleneck as a
read-fan-in capacity limit with an emission deadline, showed topology and gating cannot
fix it, and built the minimal in-forward staged aggregator that does" — every clause
already has evidence in this repo except the last, which is the work.

---

## Appendix A. Audit of S0 (`scripts/learnmask/s0_free_table_exp.py`), 2026-08-17

Code-level audit of the "free-table mask oracle" whose result ("0 flips of 6.7M gates")
anchors the current READOUT verdict. The actual protocol vs the docstring's claims:

**Actual protocol.** mmred_hf seq8 steps_in_room @512px, ≤150 train / ≤50 eval samples,
one dominant token layout, gold ≤ 9. Replica scaffold, zero trained components. Per-cell
Bernoulli gates **only on the S2 relation families {R4,R5,R6,R7}** (`arm_learn_mask("s2")`,
line 104), init ±2.0 logits at the fence. Loss = CE over the frozen model's **10
digit-token logits** at the answer position; one hard ST-Gumbel sample of all gates per
step; Adam lr 3e-2, grad-accum 8 → **~76 optimizer steps** total (600 forwards) at
defaults, ~32 under the docstring's own example line (`--limit 60`).

**Mismatches found:**
1. **Scope over-claim.** "EVERY individually toggleable attention edge" / "the bottleneck
   is not attention routing": R2 (frame-content → other-frame content/replicas — all
   direct cross-frame communication) is frozen closed and never searchable; ~93% of the
   causal edge table is outside the search space. Only routes *into replicas* and *into
   the tail* were searchable.
2. **Underpowered against its own flip criterion.** Flip barrier = 2.0 logits; Adam
   travel ceiling ≈ lr × steps ≈ 2.3 at defaults (0.96 — flips mathematically impossible —
   under the example config). Observed max |Δlogit| 1.64 ≈ 70–80% of ceiling. "0 flips"
   is what this budget produces on any loss landscape; the "~100× too faintly" gloss
   doesn't match the file's own numbers (mean shortfall 7×, max 1.2×).
3. **Objective circularity.** The loss composes the frozen digit unembedding — the search
   can only reward masks that improve *frozen emission*, which the repo's own results
   (dose-response, P7/P8) show requires training. S0 has no readout option; the
   `--readout-ckpt` fix exists only in `train_mask_gates.py` (added Aug 15, no recorded
   S0 re-run).
4. **Position confound.** `prepare_sample_replicas` applies `reset_positions`
   unconditionally and omits sequential reader positions: all blocks and all replicas
   share identical position ids, so every openable cross-block edge reads keys at
   relative position ≈ 0. Masks cannot touch positions; a joint mask+position search
   never ran. Corollary: the "no fence (plain causal attention)" baseline row is plain
   attention over 8 position-superimposed frames — not the frozen model's actual
   plain-attention behavior.
5. **Pre-written verdict + tautology.** Lines 273–274 print "…there is nothing in this
   space for it to find" unconditionally, whatever the results; "eval bit-identical to
   fence init" is a restatement of 0 flips, not corroboration.
6. **Drift gloss inconsistent.** "Drifting toward the hand design (… tail→frames −)":
   R6 is open in both the fence init and the hand design, so downward R6 drift is *away*
   from the hand design — toward the supply regime (readout forced through replicas).
7. **Provenance.** No sbatch wrapper; example command contradicts defaults; run dirs
   absent; two job ids with unknown config differences; single seed; eval is a 10-way
   digit-restricted forced choice on ≤50 samples of one qtype/layout/N. Results were
   committed inside the docstring the same day the jobs ran.

**Fair reading of what S0 shows:** within the S2 edge families, under aliased positions
and a frozen digit readout, a ~600-forward single-sample Gumbel search moved 96.5% of
logits slightly (toward opening replica-aggregation edges, toward *closing* tail→frames)
and flipped none at a budget where flipping required near-oracle gradient consistency.
It does not support "no mask can help" and cannot distinguish "topology useless" from
"topology useful but invisible to a frozen readout objective."

## 8. Repo hygiene notes (for whoever works here next)

- Restore `outputs/`, `outputs_legacy/`, `.venv`, `data/` before trusting or re-running
  anything; all 22 checkpoint symlinks dangle, including
  `carrier_layer_digit_p7a_lora_best.pt`, the default for every learnmask entrypoint.
- `RESULTS.md` stops at 2026-08-11; the gating/mmred_hf/superquery/learnmask records live
  only in `outputs/*/STATE.md` and code docstrings; `outputs/learnmask/STATE.md` +
  `CAMPAIGN_BRIEF.md` are referenced but untracked — recover them from the working
  cluster checkout or the run dirs. `ninv` and `learnmask` violate the INDEX.md
  convention (no INDEX at all).
- METHOD.md/README headline blocks predate the copy-detector, tail-sees-frames, and
  P4.2-artifact findings and should be revised before anything is cited in a thesis:
  0.999 is a TF (copy-inflated) number; "held-out N=32/48/64" is sample-held-out,
  in-length-trained; the carriers are not the decode's evidence channel.
- Metric policy that the record itself converged on (worth promoting to CLAUDE.md):
  never `tf_acc`; report against the majority/prior baseline always; shuffle eval slices;
  free-decode or answer-position-CE only; run the copy probe next to any scratchpad
  metric; single-run margins <0.07 are noise for gated arms.

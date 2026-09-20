# Brutal review + paper strategy for the SPARSE line (2026-09-02)

> Written by Claude after the 2026-09-02 peer meeting, from RESULTS.md, METHOD.md,
> outputs/{sparse,loramech,redux,armor}/ and the memory notes. Every number below traces
> to an INDEX.md run dir. Status: DRAFT for Tal + peers; nothing here is logged as a result.

## 0. Verdict in one paragraph

The SPARSE finding is real, clean and mechanistically the best thing in the record, but as
currently packaged it is a method paper about a synthetic counting task on one model, whose
headline number ("exact at 16x length") is true by construction once the gate is exact. A
strong reviewer will say: you trained a supervised per-frame classifier on gold labels, used
it to delete every non-evidence frame from attention, and then trained a LoRA on the target
task under that mask; of course it counts the survivors. What survives that attack, and
what no one has published, is the MECHANISM: the length coupling of counting in a production
VLM is entirely the competitor term (N-k) of the softmax normalizer, photographed in the
attention, removed causally by exactly one intervention, with detection shown to be already
solved inside the model. Sell the mechanism, with the gate as the causal instrument and the
deployable variant as a corollary. Then buy generality: self-derived gate, text modality,
second model, a differential prediction across reduction types, one non-synthetic number.

## 1. What the record establishes, ranked by strength

| # | claim | evidence | strength |
|---|---|---|---|
| A | The ungated count read is a softmax share m = k e^s / (k e^s + (N-k) + C); dilution by (N-k) is the length coupling | S10 photograph R2 0.93-0.99, 9/9 sign checks; L20 s=0.30, C=8 | SOLID, novel |
| B | Removing the competitor mass removes the N-dependence of the read | alpha 0.80 -> 0.07 (S3) -> 0.035 with byte-identical text (S11); margins constant | SOLID, novel |
| C | Detection is already solved inside the ungated model; aggregation is not | L24h20 AUC >= 0.993 flat in N, frozen best head 0.848 | SOLID, novel, quotable |
| D | Under the gate, exact counting holds for every trained k at every N 8..128 | S8 k<=8 = 1.000 all N; S9b 992/993 on k<=16 | SOLID but circular as "length generalization" (see R2) |
| E | The deployable gate reads per-frame evidence at >= 0.999 flat in N and matches oracle to named errors | S2 1 miss / 19,200 frames @128 | SOLID, but supervised on gold labels (see R1) |
| F | Calibration wall: the read snaps to the prompt-declared N; causal | S7 prompt-lie relocates the snap; equivalence twins | SOLID, novel, quotable |
| G | Resolution wall in k: gamma 1.19, adapter-independent | S3 + S11 | SOLID |
| H | Capacity c* ~ 16 | S8/S9b S4 cells | COVERAGE-CONFOUNDED (k > 16 never trained) |
| I | Transfer to MMReD-HF | 0.900/0.700/0.540 and 0.86/0.78/0.58 vs P1b 1.000/0.820/0.540 | MISSED its band; gate recall on HF |
| J | No eval-time temperature reaches the k-regime; retraining is needed | S0, S1-control (P1b + gate answers ~N) | SOLID, supports "calibration is competitor-mass-specific" |

## 2. The attacks (in the order a reviewer will make them)

**R1. "The gate is the counter."** The model-gate is a logistic regression trained on gold
per-frame evidence labels; once you have a 0.999 per-frame detector, counting is a sum over
its outputs, and the LoRA only maps "k visible blocks" to a digit. Answer available in the
record but NOT yet run: the internal head L24h20 separates evidence at AUC >= 0.993 in the
ungated model, so the mask can be derived from the model's own attention with no labels
beyond one threshold. Until a self-derived gate is measured, R1 stands. This is the single
most important missing experiment.

**R2. "Length generalization is by construction."** Under the gate the visible context is k
blocks whatever N is, so "16x beyond training" is a statement about the gate, not the read.
STATE.md already says it: the gate converts length generalization into k generalization.
Do not headline 16x. Headline the mechanism and the invariance of the read (alpha), then say
exactness follows for k <= c*.

**R3. "One synthetic task, one model, one modality."** MMRED park renders, one question
family, Qwen2.5-VL-7B only, HF transfer missed. For a top venue: a text-only port (the
fence and gate live on the LM, the vision tower is irrelevant), a second model, and at least
one non-synthetic number where the gated system beats ungated fenced-SFT.

**R4. "c* = 16 is your training grid."** k > 16 was never trained; the S4 zeros are coverage.
Either train the k-grid to 64 or 128 and report the measured break, or drop c* as a number
and keep only the resolution law gamma.

**R5. "Oracle numbers dominate the tables."** Every headline table leads with the oracle
gate. Lead with the deployable (self or model) gate; oracle as a dashed upper bound.

**R6. Prior art.** They will cite: PCW / APE (block attention + position reuse, text);
AHAT theory (hard attention over the argmax set counts: Hao/Angluin/Frank, Barcelo, Merrill);
ASEntmax / Selective Attention (learned zero-attention gives 1000x length extrapolation on
retrieval); Yehudai 2024 (counting capacity, the resolution wall); Barbero 2024 (over-
squashing in decoder LMs, text); Chang and Bisk (in-range counting fails OOD); NoLiMa /
context rot (distractors, not length, drive degradation); retrieve-then-read pipelines. What
is unclaimed: the share law measured on a production VLM with (N-k) identified as the whole
length coupling; the causal removal; detection/aggregation dissociation via a single head;
the prompt-N calibration attractor; the k-resolution law in bf16 on a real model.

**R7. Missing baselines.** (a) per-frame classifier + sum, the ceiling you must show the
in-model read reaches; (b) evidence-only frames fed to the vanilla frozen model with no
LoRA: does it count k identical-verdict frames at all? Either outcome is informative and a
reviewer will ask; (c) the METHOD.md caption-scan reader zero-shot at 64/128 (exists: 0.087
at 8x); (d) a trained soft gate (learned per-block temperature or entmax-style) vs the hard
mask; (e) retrieve-then-read with re-encoding (the equivalence twin is this by construction,
say so).

**R8. Statistics.** Single training seed per adapter; ep5 vs ep10 twins differ by 3-5% at
thin strata; k-strata of 9-20 samples; HF cells of 50. Headline adapter needs 3 seeds and
strata pooled to >= 50.

**R9. "Two forwards plus a task-trained 2M-param LoRA plus a classifier is a pipeline."**
True for the deployable variant. The defense is the analysis framing: the intervention is
minimal and surgical (one mask term), and every other knob was held at the fenced-SFT
baseline. Do not oversell deployability.

**R10. The thesis story conflicts with itself.** PAPER_SKELETON.md argues "symbolic
execution is the only length-invariant rung". SPARSE shows an in-model, one-read fix that is
exact in N for k <= c*. Reconcile explicitly: the necessity claim moves from N to k. Beyond
the read's resolution in k, only token-coded readout passes. Say this once, clearly, or the
reviewers of either paper will find the other in the thesis.

## 3. The story that survives: one mechanism, five walls

The narrative device the peers asked for ("divide the failure points into a story") is
already in the data. Every wall gets a causal test and either a fix or an honest boundary.

| wall | is it the failure? | test | fix / boundary |
|---|---|---|---|
| 1 Perception | NO | fenced supply alpha 0.00 (ARMOR-A); L24h20 detects evidence at AUC 0.993 flat | none needed |
| 2 Aggregation share | YES | photograph of m = k e^s/(k e^s+(N-k)+C); alpha 0.72-0.80; margins slide | delete (N-k): alpha -> 0.035, exact for trained k at all N |
| 3 Calibration to prompt text | YES, second-order | prompt-lie relocates the snap; twins reproduce with nothing hidden | coverage mixture or N-free prompt |
| 4 Resolution in k | YES, the remaining wall | gamma 1.19 in k, adapter-independent; bf16 floor | measure c* honestly; beyond it token-coded readout |
| 5 Detector transfer | domain-bound | HF gate recall fn 6-7/cell | retrain detector in-domain; self-gate may help |

One sentence for the abstract: counting over N frames fails not because the model cannot see
the evidence but because softmax aggregation divides it by everything else; removing the
"everything else" term makes the read length-invariant, and what remains is a resolution
limit in the count itself.

The unifying variable is COMPETITOR MASS, not length. That reframing answers the peers'
"target more problems than length generalization": distractor sensitivity (NoLiMa, "More
Images More Problems") and length are the same term. A fixed-N hard-negative experiment
makes that a measured claim (E7).

## 4. What to claim

- C1 Mechanism (causal): the length coupling of counting in a frozen VLM is the competitor
  term of the softmax normalizer. Evidence: share law photographed; alpha under gate;
  byte-identical-text control; prompt-lie.
- C2 Dissociation: detection is solved internally, aggregation is not; a single trained-in
  head detects evidence flat in N while the answer collapses.
- C3 Intervention: evidence-gated attention (self-derived if E1 lands) + a plain LoRA gives
  exact counting for every trained k at every N up to 16x training length, with 2M params.
- C4 Laws: calibration to declared N; resolution law in k; reduction-type differential
  (count needs the gate, ratio needs the competitor term).
- C5 Generality: text and vision, two models, one non-synthetic benchmark. PENDING.

What to drop from THIS paper: carriers, register trees, scratchpad readout, ACE executor,
treefold, the recagg drift table. One paragraph of "what does not work" plus an appendix
pointer; they are a second paper or thesis chapters.

## 5. Experiments that close the attacks, ranked by leverage

Costs use the campaign's own numbers (5-ep trainer ~7 h on 24h_1g; ladder evals ~2 h; probe
chains ~2-3 h). Bands are proposals to pre-register, not results.

| # | experiment | kills | cost | pre-registered band |
|---|---|---|---|---|
| E1 | Self-derived gate: binarize L24h20 answer-row attention per block (one threshold fit on train), eval S8/S9b adapters; then retrain under the self-gate | R1, R9 | 3 h eval + 7 h train + 2 h | k<=8 >= 0.95 at N=128 with zero gold labels at eval; parity with model-gate to named errors |
| E2 | Baseline pair: per-frame-classifier + sum ceiling; evidence-only frames to the vanilla frozen model | R7, frames R1 | 1-2 h | report, no band |
| E3 | REDUX C2/C3: one multitask fenced reader for count / exists / majority, gated and ungated, ladder 8..128 | mechanism credibility, "other problems" | 7 h + 7 h | ungated majority N-flat at fixed ratio while ungated count decays; gated majority fails; gated count exact |
| E4 | Text port: Qwen2.5-7B-Instruct, N passages as blocks, same fence + gate + probe_hahn | R3 modality | 2-3 days eng + 7 h + 4 h | share law fits (R2 >= 0.9); alpha 0.6-0.8 ungated, CI contains 0 gated; k<=8 exact at 8x |
| E5 | True c*: virtual-N grid to k = 64 or 128, eval at N=128 | R4 | 7-14 h + 2 h | report c* with CI; keep gamma |
| E6 | 3 seeds of the headline adapter, strata pooled to >= 50 | R8 | 3 x 7 h | mean +- sd on every headline cell |
| E7 | Competitor mass at fixed N: hard negatives (same character, other room) vs easy negatives (empty frames) | "beyond length"; unifies with distractor literature | datagen + 4 h | ungated accuracy tracks fitted competitor mass; gated invariant |
| E8 | Second model: Qwen2.5-VL-3B and 32B (size sweep of s, C, gamma) or InternVL2.5-8B | R3 | ~1 week | share law holds; c* vs size reported |
| E9 | Non-synthetic: MMReD-HF with an HF-trained gate; MLVU-AC or EC-Bench counting subset with the model gate | R3 external validity | 1-2 days | gated > ungated fenced-SFT at seq32 by >= 0.10 |
| E10 | No-harm MME/POPE with the gated adapter | completeness | 2 h | <= 2 pts |

Minimum set for a credible submission: E1, E2, E3, E6, plus E4 or E9. E5 and E7 make the
laws section honest and broad. E8 is the "scaling" panel reviewers like.

## 6. Paper shape

Titles (mechanism first, method second):
1. Where Over-Squashing Lives: Softmax Competitor Mass, Not Perception, Bounds Counting in
   Vision-Language Models
2. The Model Sees the Evidence but Cannot Count It: Removing the Length Coupling of
   Attention Aggregation
3. Counting Under a Mask: Evidence-Gated Attention Makes a Frozen VLM Length-Invariant Up
   to Its Resolution Limit

Figures:
- F1 (mechanism): attention share vs N at fixed k with the fitted law, ungated vs gated
  (S10 data). The one-image thesis.
- F2 (flat line): accuracy vs N, frozen / fenced-SFT / gated self-derived (solid) / gated
  oracle (dashed).
- F3 (dissociation): L24h20 AUC flat in N beside the collapsing answer accuracy.
- F4 (the two remaining walls): prompt-lie relocation panel; accuracy vs k at N=128 with
  the gamma fit and measured c*.
- F5 (generality grid): task x modality x model, one cell per E3/E4/E8/E9 result.

Venue: an analysis-plus-intervention paper fits ICLR or NeurIPS main track; a text port
opens ACL/EMNLP. Check the ICLR 2027 and CVPR 2027 deadlines against the minimum set above
before choosing.

## 7. Wording rules for the draft

- Never write "length generalization" as the headline; write "the read is invariant to
  competitor mass" and let exactness follow.
- Every table labels oracle vs model vs self gate.
- c* appears only with the coverage it was measured under, until E5.
- The MMReD-HF miss is reported with its anatomy (gate recall), not hidden.
- Cite PCW/APE for the fence, AHAT for the hard-attention idea, Yehudai for the resolution
  wall, Barbero for the framing, ASEntmax for learned sparsity. Claim the combination and
  the measurements, never the mask.

---

## 8. Paper options (2026-09-02, second pass: framing only, no new experiments)

Four candidate papers the record could support, each with its thesis sentence, what
carries it today, its killer figure, and the objection that decides it.

| option | thesis sentence | carried by | killer figure | deciding objection | verdict |
|---|---|---|---|---|---|
| M: method | "Evidence-gated attention gives a frozen VLM exact counting at 16x training length" | S8/S9b ladders, S2 gate | flat line | the gate is a supervised detector; length invariance is by construction; one task/model | NOT writable now |
| S: spectrum / necessity (PAPER_SKELETON) | "Every learned aggregator drifts; only symbolic execution is length-invariant" | recagg drift table, ACE, cascade | spectrum figure | SPARSE contradicts the headline: an in-model read IS invariant in N for k <= c*; would need the necessity claim moved from N to k | second paper or thesis chapter |
| T: toolkit / ladder | "A diagnostic ladder localizes aggregation bottlenecks in frozen models" | METHOD.md ladder, probe_hahn, photograph, gate | ladder table | tool papers need several case studies; one task | NOT now |
| **W: mechanism, GNN-bridged** | "Over-squashing in transformer aggregation is a degree effect in the softmax denominator; rewiring to evidence-only degree removes it, and the residual wall is the aggregator's resolution in k" | ARMOR-A, LORAMECH N2/N3/P3, REDUX C4b, S0-S11, Jacobian cross-frame share | share-law photograph + the alpha rewiring ladder | "Barbero did over-squashing in LMs" and "the gate needs labels"; both answerable by scoping to localization | **WRITABLE NOW; recommended** |

### Why W is the impactful one

- It gives the field a LAW with a photograph, not another degradation curve. The
  long-context failure literature (context rot, NoLiMa, "More Images More Problems",
  EC-Bench) reports decay; none supplies a quantitative mechanism with a causal test.
- It has a hook that survives one sentence: the model sees the evidence but cannot count
  it, because attention divides evidence by everything else.
- It bridges two communities honestly. To the graph community: your over-squashing
  bound's degree term predicts a production VLM's failure exponent, and your remedy,
  rewiring, works inside it. To the LLM community: masks are rewiring, retrieval is
  denominator repair, length and distractors are one variable.
- It closes with a second, different wall predicted by a second piece of GNN theory:
  after rewiring, the aggregator is still a normalized mean, and "mean cannot count, sum
  can" (Xu et al., GIN) appears as the measured resolution law in k. Two walls, two
  theories, one aggregator. That is a story, not a table.

### The story spine of W (the "five walls" told as a ladder of rewirings)

| rung | graph operation | measured sensitivity exponent alpha in N | accuracy consequence |
|---|---|---|---|
| frozen joint attention | complete graph over N frames + question | 0.72 [0.63,0.78] | collapse 0.22 -> 0.01 |
| fence | cut frame-frame edges; islands + question; per-block position reset | per-frame supply 0.00 +- 0.02 | supply flat, read still decays (0.80 trained) |
| gate | cut non-evidence -> readout edges; readout degree = k | 0.07 [-0.08,0.20]; 0.035 with byte-identical text | exact for every trained k at every N |
| (residual) | normalized mean over k | decay in k: gamma 1.19 [1.17,1.22] | resolution wall; c* coverage-caveated |

Side walls that the spine passes through with a causal test each: calibration to the
prompt-declared N (S7 prompt-lie, equivalence twins; fixed by coverage or N-free prompt)
and the calibration-specificity of the share code (P1b + gate answers N without
retraining; no temperature reaches the regime at eval; logN moves margins not alpha).

### Section outline for W

1. Introduction: hook, framing, contributions C1 mechanism / C2 dissociation / C3 causal
   rewiring / C4 the two laws (calibration, resolution).
2. Background: over-squashing bound and its degree term; Hahn 1/N; GIN mean vs sum;
   Barbero; PCW/APE as the fence precedent; AHAT as the hard-attention precedent.
3. Setup and instruments: MMRED, frozen Qwen2.5-VL-7B, fenced-SFT read as the trained
   baseline; instruments = paired one-frame-flip sensitivity per locus, attention
   photograph, oracle/model gate as intervention.
4. Not perception: fenced supply flat; L24h20 detects evidence at AUC >= 0.993 flat in N;
   per-frame states decodable at 0.999.
5. The denominator: share law photographed; fitted s and C; alpha of the read; margins
   slide; logN and temperature controls.
6. Rewiring the readout: alpha -> 0; exactness for trained k; the no-retrain control.
7. Calibration wall: prompt-lie; virtual-N; N-free.
8. Resolution wall: gamma; the GIN reading; c* with its coverage caveat; token-coded
   readout as the only measured route past it (one paragraph, cite the record).
9. Discussion: design principle (evidence-only denominators: hard/sparse attention,
   entmax, retrieval); the competitor-mass reframing of length vs distractors as a stated
   prediction; limitations (one task, one model, one modality; gate supervised at deploy;
   oracle vs model labeled); future (self-derived gate, text port).

### Figures for W

- F1 the photograph: evidence share vs N at fixed k, ungated with the fitted law, gated
  flat. One image, the thesis.
- F2 the rewiring ladder: alpha per locus with CIs, four bars, one panel.
- F3 the dissociation: L24h20 AUC vs N beside answer accuracy vs N.
- F4 the flat line: frozen / fenced-SFT / gated model (solid) / gated oracle (dashed).
- F5 the two side walls: prompt-lie relocation; accuracy vs k at N=128 with the gamma fit.

### Abstract draft (W)

A frozen vision-language model that counts frames correctly at eight frames fails by
one hundred twenty-eight, while a single attention head inside it flags every evidence
frame at every length. We localize the failure with a ladder of graph rewirings inside
one forward pass. Per-frame perception is length-invariant once frames are isolated;
the answer's sensitivity to one frame decays as a power of N, as the over-squashing
bound's degree term predicts, and the attention weights obey a photographed share law
in which the evidence competes with every non-evidence frame in the softmax denominator.
Deleting exactly that competitor term makes the read invariant to length and the count
exact for every trained value up to sixteen times the training length. Two walls
remain and both are measured: the read is calibrated to the frame count the prompt
declares, and the rewired aggregator resolves counts only up to a resolution limit in
the count itself, the transformer instance of "mean cannot count". Aggregation, not
perception, is where over-squashing lives.

### Claims that are safe today vs claims that must wait

Safe now: C1-C4 as stated, scoped to "a frozen production VLM on a controlled counting
task", oracle vs model gate labeled, c* with its coverage caveat, HF miss reported.
Must wait: any "general", "modality-robust", "self-gated", "deployable" wording, and the
distractor unification as a measurement rather than a prediction.

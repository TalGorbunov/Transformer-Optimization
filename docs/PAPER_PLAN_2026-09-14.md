# Paper plan for the peer meeting (2026-09-14)

> Written by Claude for Tal's peer meeting, from RESULTS.md (through S9b/S10/S11),
> docs/PAPER_REVIEW_2026-09-02.md, outputs/{sparse,redux,selfgate,armor,loramech}/,
> and a fresh prior-art check (2026-09-14, sources in §4). Every number traces to a
> RESULTS.md entry (date in brackets) and its INDEX.md run dir. Nothing here is a result.

## 0. The answer in five sentences

The gate is not the paper. Masking non-evidence frames is precedented six ways
(PCW, S2A, ASEntmax, Screening, KV eviction, frame selection) and a reviewer will say
"you deleted the distractors and counted what was left". What nobody has is the
MEASUREMENT: in a frozen production VLM the whole length coupling of counting is the
competitor term (N-k) of one softmax denominator, photographed in the attention,
removed causally by one intervention with byte-identical text, while a single internal
head already detects the evidence at every length. The paper is a mechanism paper with
the gate as its knockout instrument, closed by two residual walls that two pieces of
GNN theory predict (calibration; "mean cannot count" in k). Novelty is then a set of
five measured claims, each with a named closest prior and a stated delta (§2), and
four risky differential predictions that competing explanations do not make (§3).

## 1. What the paper is

**Working title.** *The Model Sees the Evidence but Cannot Count It: Softmax Competitor
Mass Is the Length Wall of Aggregation in a Vision-Language Model.*
(Alt: *Where Over-Squashing Lives: the Denominator, Not Perception.*)

**Thesis sentence.** Counting over N frames in a frozen VLM fails not because the model
cannot see the evidence but because softmax aggregation divides it by everything else;
deleting exactly that term makes the read invariant to length, and what remains is a
resolution limit in the count itself.

**Genre.** Analysis + causal intervention (ICLR/NeurIPS/ICML "science of deep learning"
track). Not a method paper. The deployable gated adapter is a corollary in one subsection.

**Contributions (each one measured, each with a closest-prior line in §2).**

| # | claim | evidence (RESULTS.md) | status |
|---|---|---|---|
| C1 Law | Read attention share obeys m = k·e^s / (k·e^s + (N-k) + C); R² 0.93-0.99 per layer, 9/9 N-doubling sign checks; L20 s = 0.30, C = 8. One-frame sensitivity decays N^-0.72 [0.63, 0.78] in the frozen model, +0.80 [0.66, 0.97] through the trained read; margins slide monotonically | [08-22] Hahn, [09-01] S10, [08-27→28] LORAMECH | SOLID |
| C2 Dissociation | Per-frame supply is flat under the fence (α 0.00 ± 0.02); a single trained-in head L24h20 separates evidence blocks at AUC 0.993-1.000 flat in N in the UNGATED model (frozen best head 0.848); an LR probe on per-frame states reads evidence at ≥ 0.999/frame flat to N = 128 | [08-22] ARMOR-A, [09-01] S10, [08-31] S2 | SOLID |
| C3 Knockout | Deleting the (N-k) term (mask only; positions reset; byte-identical text) → read α +0.035 [-0.043, +0.123], margins constant +11.7 nats N = 8→128, one-frame flips change the answer 90-100 % at every N; count exact on 992/993 trained-support cells at N ∈ {8..128}, 16× past training | [09-01] S11, [08-31] S3, [09-01→02] S9b | SOLID |
| C4 Remedy taxonomy | Temperature/log-N class cannot reach the regime (no τ lifts k ≤ 8 @64 past 0.24; α 0.605 vs 0.603 under log-N; log-N as training prior collapses 2× from 0.867 to 0.420); exact-zero class works (C3); retrieve-then-read is provably the same forward (equivalence twin, 20/20 identical decodes); the read without retraining answers ≈ N under the gate (calibration is competitor-mass-specific) | [08-30→31] S0, REDUX C4b, [08-28] P3, [08-31] S7/equiv, S1-control | SOLID |
| C5 Two residual walls | (a) Calibration to the prompt-declared N: lying about N relocates the snap exactly (k12/k16 0→1.00 under "16 frames"; k8 1.00→0.00 under "64"); fixed by coverage (virtual-N) or an N-free prompt. (b) Resolution in k: Δ ∝ (k+C)^-1.19 [1.17, 1.22], adapter-independent; capacity edge c* ≈ 16 under trained coverage | [08-31] S7, [09-01] S8/S9, S3/S11 | SOLID; c* coverage-caveated |

**What to drop from this paper.** Carriers, register trees, scratchpad readout, ACE
executor, treefold, recagg drift table, learnmask sweep. One "what does not work"
paragraph + appendix pointer. They are the thesis's other chapters.

**Wording rules.** Never headline "16× length generalization" (true by construction once
the gate is exact); headline α and the photograph, let exactness follow. Every table
labels oracle / model / self gate. c* appears only with its coverage. The MMReD-HF miss
is reported with its anatomy (gate recall). "Frozen production VLM, controlled counting
task" is the scope until E4/E8/E9 land.

## 2. Novelty defense: closest prior per claim, and the delta

Verified 2026-07-24 (docs/citations_verified.md), 2026-08-17 (outputs/recagg/RELATED_WORK.md),
2026-08-30/31 (memory), and today (§4). "Not novel" is only true of the mask; state that
in the paper's first paragraph of related work and move on.

| our claim | closest prior (verified) | what they have | what they do not have = our delta |
|---|---|---|---|
| C1 law | Veličković et al. 2024, *softmax is not enough* (2410.01104) | theorem: attention coefficients disperse as N grows; adaptive temperature remedy | measured law with the (N-k) term identified and fitted (s, C) in a production VLM; and their remedy class fails here (C4) |
| C1 law | Barbero et al. NeurIPS 2024, *Transformers need glasses* (2406.04267) | over-squashing + representational collapse in decoder LMs; counting/copying failures; separator-token mitigations; text only | localization to one denominator term; knockout; VLM; dissociation from perception; the two residual walls |
| C1 law | Hahn TACL 2020 | O(1/n) sensitivity bound | first measurement of the exponent inside a production model, at the bf16 floor by N = 128 |
| C1 law | Levy et al. ACL 2024 *Same task, more tokens*; Du et al. EMNLP 2025 *Context length alone hurts* (2510.05381); NoLiMa; *More Images More Problems* (2601.07812); EC-Bench | degradation curves with length/distractors; Du et al.: degradation persists when irrelevant tokens are attention-masked | a quantitative mechanism, not a curve. Du et al. masked WITHOUT resetting positions; our fence resets per-block positions and the no-reset arm drifts (0.996 → 0.830 supply; deployed decode 0.987 → 0.313 @32) — their residual is the positional channel, ours removes both, and S11 shows nothing is left |
| C2 dissociation | Orgad 2024 *LLMs know more than they show*; arXiv:2605.09239 (count decodable internally, overwritten late); Sengupta 2511.17722 (see-but-not-enumerate) | "information present, output wrong" | a single named attention head that is the selector, flat in N, whose mass still obeys the share law: the model knows WHICH frames and cannot cash it through softmax |
| C3 knockout | ASEntmax, ICLR 2026 (2506.16640); *Screening Is Enough* (2604.01178); Selective Attention (2410.02703) | architectures trained from scratch with exact zeros → 1000× extrapolation on synthetic retrieval | same principle realized as a mask in a FROZEN production model; the law it removes is measured; counting (not retrieval); the k-wall they never see |
| C3 knockout | PCW ACL 2023 / APE ICLR 2025 (block attention + position reuse) | the fence's structural twin, text, no selection | fence + per-block reset as the perception half; selection as the aggregation half; each measured separately (α per rung) |
| C3 knockout | S2A (Weston & Sukhbaatar 2023); RAG; DAFS Jul 2026 (2607.15689, attention-based frame selector); KV eviction (H2O, SnapKV) | remove irrelevant context, accuracy or efficiency improves | we prove the gated forward IS the retrieve-then-read forward (equivalence twin) and say WHY it works (denominator), WHEN it cannot (majority/ratio, k > c*), and that the selector is already inside the model (L24h20) |
| C3 knockout | AHAT theory (Hao/Angluin/Frank TACL 2022; Barceló ICLR 2024; Merrill) | uniform hard attention over the argmax set counts | AHAT realized by mask in a frozen VLM; its bf16 capacity and calibration failure measured |
| C4 taxonomy | SSMax (2501.19399); adaptive temperature (Veličković); log-N scaling | sharpening remedies | tested in one model: eval-only partial, training prior harmful, α unchanged; only exact zeros move α |
| C5a calibration | none found (prompt-declared N as an attractor) | — | causal prompt-lie experiment; coverage and N-free fixes |
| C5b k-wall | Yehudai et al. NeurIPS 2024 *When can transformers count to n*; Xu et al. GIN (mean cannot count) | capacity phase transition; theory | γ = 1.19 measured, adapter-independent, in bf16 on a production model, after the N term is removed |
| framing | Lee 2512.09182 (transformer failures through the GNN lens); Alon & Yahav; Topping; Di Giovanni | framing / theory | an instantiated ladder of rewirings with a measured exponent per rung |

One-sentence version for the meeting: *"Everyone in this table owns one adjacent piece
(a theorem, a from-scratch architecture, a degradation curve, a frame selector); nobody
has the measured law, its knockout, the dissociation head, or the k-wall in a deployed
model, and no paper combines them."*

## 3. Risky predictions (pre-register these; they are what makes the mechanism claim reviewer-proof)

A reviewer can always say "of course fewer frames is easier". The mechanism is defended
by predictions that the "easier task" story does NOT make.

| # | prediction (only the denominator story makes it) | test | cost | band |
|---|---|---|---|---|
| P1 | Accuracy at FIXED N tracks fitted competitor mass, not frame count: hard negatives (same character, other room) carry higher e^s than empty frames, so swapping negative type at fixed N, k moves accuracy by the amount the fitted (s, C) predicts; gated arm invariant | E7 competitor-composition cell: datagen + ungated/gated ladders | datagen + ~4 h | ungated accuracy ordered by fitted mass; gated ± 0.03 |
| P2 | The detector is already inside: a gate derived from the model's OWN head (one threshold, no labels at eval) reaches oracle parity to named errors | E1 = selfgate G0a/G0b (brief authorized 09-01, not run) | ~3 h eval; + 7 h retrain under self-gate | k ≤ 8 ≥ 0.95 @128 with zero gold labels; gaps named per-sample |
| P3 | Reduction-type differential: gating HELPS count and HURTS majority/ratio (the share IS the quantity there), and ungated majority is N-flat at fixed ratio while ungated count decays | E3 = REDUX C2/C3 (one multitask fenced reader, count/exists/majority, gated + ungated ladder) | 7 h + 7 h | gated majority < ungated majority @128; ungated majority flat ± 0.1 at |d|/N = 1/8 |
| P4 | Position and denominator are separable channels: gate WITHOUT per-block reset leaves a residual α > 0 (Du et al.'s residual); gate + reset → 0 | one α chain, no-reset arm through the gate | ~2 h | residual α ∈ [0.2, 0.5] no-reset; CI ∋ 0 with reset |
| P5 | Sharpening cannot substitute for zeros | already measured (S0, C4b, P3) | 0 | done |

## 4. Experiments before submission (ranked; costs from the campaigns' own numbers)

| # | experiment | kills | cost | status |
|---|---|---|---|---|
| E1 | Self-derived gate (G0), then retrain under it (G1a) | "the gate is a supervised counter" | 3 h + 7 h + 2 h | brief written outputs/selfgate/, authorized, NOT RUN |
| E2 | Baseline pair: per-frame classifier + sum (ceiling); evidence-only frames to the vanilla frozen model | missing baselines | 1-2 h | not run |
| E3 | REDUX C2/C3 multitask gated/ungated ladder (P3) | "one task"; mechanism credibility | 14 h | C0/C1/C4b done; C2 gated on review |
| E6 | 3 seeds of the headline adapter; strata pooled to ≥ 50 | statistics | 3 × 7 h | not run |
| E7 | Competitor composition at fixed N (P1) | "fewer frames is easier"; unifies length with distractors | datagen + 4 h | not run |
| P4 | No-reset gated α chain | Du et al. | 2 h | not run |
| E5 | c* with k-grid to 64/128 | "c* is your training grid" | 7-14 h + 2 h | not run |
| E9 | Non-synthetic: MMReD-HF with an in-domain gate; MLVU-AC or EC-Bench counting subset | external validity | 1-2 days | HF cells exist (missed band; gate recall) |
| E4 | Text port (Qwen2.5-7B-Instruct, passages as blocks) | modality | 2-3 days eng + 11 h | not started |
| E8 | Second model (Qwen2.5-VL-3B; 32B if H200) | model | ~1 week | not started |
| E10 | No-harm MME/POPE with the gated adapter | completeness | 2 h | not run |

Minimum credible set: E1, E2, E3, E6, E7, P4 (≈ 4-5 GPU-days, all on 48 GB cards) plus one of
E4/E9. E8 is the scaling panel.

**Venues (verified 2026-09-14).** ICLR 2027 full paper 2026-09-25 (abstract 09-18): 11 days,
not feasible for the minimum set — do not chase it. CVPR 2027 2026-11-13 (abstract 11-07):
8 weeks; feasible with E1/E2/E3/E6/E7/P4 + E9. NeurIPS 2027 2027-05-21. ICML 2027 (late
January 2027, unverified) is the natural home for a mechanism paper and leaves room for E4/E8.
Recommendation: write for CVPR as the forcing function; if E9 misses its band, submit the
same paper with E4 to ICML.

## 5. Paper shape

1. **Introduction.** Hook (sees but cannot count); star-graph framing (softmax read = normalized
   aggregator over a degree-N star; Hahn's 1/N is the degree term of the over-squashing bound);
   contributions C1-C5; scope statement.
2. **Background.** Over-squashing bound and its degree term; Hahn; GIN mean vs sum; Barbero;
   Veličković dispersion; PCW/APE as the fence precedent; AHAT as the hard-attention precedent;
   ASEntmax/Screening as the exact-zero architectures.
3. **Setup and instruments.** MMRED; frozen Qwen2.5-VL-7B 4-bit; fenced-SFT read as the trained
   baseline; instruments: paired one-frame-flip sensitivity per locus (α), attention photograph
   (share law fit), gate as knockout (oracle / model / self, labeled).
4. **Not perception.** Fenced supply α 0.00; L24h20 AUC ≥ 0.993 flat; per-frame states 0.999.
5. **The denominator.** Photograph; s, C; read α 0.72 frozen / 0.80 trained; margin slide;
   temperature and log-N controls (C4).
6. **Knockout.** α → 0.035 with byte-identical text; margins constant; exactness for trained k;
   no-retrain control; equivalence twin (= retrieve-then-read); self-gate (E1) and the no-reset
   arm (P4).
7. **Beyond length: competitor mass.** E7 fixed-N composition; E3 reduction-type differential.
8. **Two walls.** Prompt-N calibration (prompt-lie; virtual-N; N-free). Resolution in k (γ;
   GIN reading; c* with coverage; token-coded readout as the only measured route past it, one
   paragraph citing the record).
9. **Discussion.** Design principle: evidence-only denominators (hard/sparse attention,
   screening, retrieval are one family); length and distractors are one variable; limitations
   (one task family, one model, one modality until E4/E8; gate supervised unless E1 lands);
   deployable corollary (23.8M-param LoRA (r8, all projections, all layers; same recipe as the ungated baseline) + a ≥ 0.999 gate, 16×).

**Figures.**
- F1 The photograph: evidence share vs N at fixed k with the fitted law; gated flat (exists:
  outputs/sparse/fig/F7_attn_photo.png).
- F2 The rewiring ladder: α per locus with CIs (frozen joint 0.72 / fenced supply 0.00 /
  trained read 0.80 / gated read 0.035), four bars.
- F3 The dissociation: L24h20 AUC vs N beside answer accuracy vs N.
- F4 The flat line: frozen / fenced-SFT / gated self (solid) / gated oracle (dashed)
  (exists as F5_flatline_w3.png; add the self-gate curve after E1).
- F5 The two walls: prompt-lie relocation panel; accuracy vs k at N = 128 with the γ fit and c*.
- F6 (after E7/E3): accuracy vs fitted competitor mass at fixed N; count vs majority under gate.

**Abstract draft.** A frozen vision-language model that counts correctly over eight frames
fails by one hundred twenty-eight, while a single attention head inside it flags every
evidence frame at every length. We localize the failure with a ladder of graph rewirings
inside one forward pass. Per-frame perception is length-invariant once frames are isolated;
the answer's sensitivity to one frame decays as a power of N, as the degree term of the
over-squashing bound predicts; and the attention weights obey a measured share law in which
the evidence competes with every non-evidence frame in the softmax denominator. Deleting
exactly that competitor term, with the prompt held byte-identical, makes the read invariant
to length and the count exact for every trained value up to sixteen times the training
length; sharpening the softmax cannot do this, and retrieval-then-read is the same forward.
Two walls remain and both are measured: the read is calibrated to the frame count the prompt
declares, and the rewired aggregator resolves counts only up to a limit in the count itself,
the transformer instance of "mean cannot count". Aggregation, not perception, is where
over-squashing lives.

## 6. The presentation (12 slides, ~20 min + discussion)

| # | slide | what is on it | say |
|---|---|---|---|
| 1 | Sees but cannot count | F3 sketch: head AUC flat at 0.99 vs answer accuracy collapsing | "Last time we ended on the gate. The gate is an instrument. This is the finding." |
| 2 | Why the gate felt un-novel, and why that is fine | the six precedents for masking (PCW, S2A, ASEntmax, Screening, KV eviction, DAFS) | "Masking is not the claim. Nobody in this list measured what the mask removes." |
| 3 | The graph picture | star graph; normalized aggregator; Hahn 1/N = degree term | "Over-squashing predicts a sensitivity exponent. We measured it." |
| 4 | The exponent | Hahn figure: joint N^-0.72 at the bf16 floor by 128; fenced supply flat | ARMOR-A |
| 5 | The photograph | F7a: share law, s = 0.30, C = 8, R² 0.97 | "The formula every intervention presupposed is now a measured object." |
| 6 | The model already knows which frames | L24h20 AUC ≥ 0.993 flat; LR gate 0.999/frame | "Detection is solved. Aggregation is not." |
| 7 | The knockout | F7b + α bar: 0.80 → 0.035, byte-identical text; margins constant | "One term, one intervention, one channel left: none." |
| 8 | The flat line | F5: 992/993 trained-support cells exact 8..128; oracle vs model labeled | "Exactness follows; it is not the headline." |
| 9 | What cannot do it | S0 temperature sweep; log-N α unchanged; P3 training prior; no-retrain control | "Sharpening is the wrong family. Zeros are the right family. Retrieval is the same forward." |
| 10 | The two walls | prompt-lie panel; γ = 1.19 in k; c* ≈ 16 with coverage caveat | "Calibration and resolution. Both predicted, both measured." |
| 11 | Novelty map + risky predictions | §2 table compressed to 8 rows; P1-P4 with bands | "Each row owns one piece; none has the law. These four predictions are ours alone." |
| 12 | Plan and asks | E1/E2/E3/E6/E7/P4 (+E9 or E4); CVPR 11-13 vs ICML; asks: venue, second model, text port, seeds | end on the asks |

Rehearsal note: if the peers push "in-model GNN aggregator" again, the answer is slide 10:
after rewiring, the aggregator is still a normalized mean and GIN says it cannot count
beyond its resolution; that is the measured γ, and it is the paper's closing wall, not a
gap to fill with another architecture.

## 7. Objections the peers will raise, with the answer in the record

| objection | answer |
|---|---|
| "The gate is the counter." | E1 self-gate (no labels at eval) + E2 classifier+sum ceiling + vanilla evidence-only control. Until E1: the S2 gate reads the per-frame channel that the fence already made flat; the LoRA under it fits k, not N. |
| "16× is by construction." | Agreed; we headline α, the photograph, and margins, never 16×. |
| "Barbero did over-squashing in LMs." | They proved collapse and proposed separators, text only. We localize to one denominator term, photograph it, knock it out, in a VLM, and dissociate it from perception. |
| "ASEntmax / Screening already give exact zeros and extrapolate." | From scratch, synthetic retrieval, no law, no k-wall, no production model. Cite as convergent evidence for the design principle. |
| "Du et al. show masking does not help." | They kept positions; the residual they see is the positional channel. Our fence resets it; P4 tests the separation directly. |
| "DAFS already selects frames from attention." | Efficiency under a frame budget, no mechanism, answerer's accuracy not the point. We say why it works and when it cannot. |
| "It is retrieve-then-read." | Yes: the equivalence twin proves the gated forward is that forward. The contribution is the mechanism and its limits (majority, k > c*). |
| "One task, one model, one modality." | Stated scope; E3 adds tasks, E4 modality, E8 model, E9 external. |
| "c* = 16 is your training grid." | Reported with its coverage; E5 measures it; γ is coverage-free. |
| "Single seed." | E6. |
| "The thesis's other chapter says only symbolic execution is length-invariant." | Reconciled: necessity moves from N to k. Below c* the in-model read is exact in N; beyond it only token-coded readout passes. Say it once. |

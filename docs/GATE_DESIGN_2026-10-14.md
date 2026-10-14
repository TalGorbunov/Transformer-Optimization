# Gate design after the 2026-10-14 peer meeting

Written for Tal and the peer group, 2026-10-14 (branch `theory-2`). Inputs: the three VNBench weeks
(`outputs/scaleup/STATE.md` entries 2026-10-11 to 2026-10-13), the peer page "Gated aggregation in a
frozen VLM: the main results", eight web-verified literature lenses (`docs/gate_design_2026-10-14/research_digest.md`,
176 papers, most read at full text), a six-proposal design panel with a hostile critique per proposal
(`docs/gate_design_2026-10-14/design_panel_critiques.md`), and three CPU transfer fits run today (jobs 181599-181601).

**Decision.** Keep the gate as it is: a linear probe on each fenced, question-conditioned block's layer-20
end-of-vision state, hard threshold, frozen packed re-read, the one-frame-per-needle merge. Do not build a
relational module, a label-free threshold, or a reader LoRA for the headline. Change what is claimed and what
is measured: (1) the label-source table, (2) the kept-frame accounting (false frames per question, exact-set rate,
leak curve) instead of per-frame accuracy, (3) the per-task table showing where cross-frame relevance binds, and
(4) three cheap controls the record lacks (question-blind capture on VNBench, a 3B fenced-slot probe, reader-derived
labels on the same probe). Every proposal the panel generated to replace the architecture died on a number already
on record; the ones worth keeping are controls and cells, listed in section 4.

## 1. Answers to the four questions

### Q1. The gate is supervised. Is that a problem? How does Sparse RAG train its gate?

Not by the field's standard, and the honest sentence is "supervised by the benchmark's construction, not by a
stronger model". Every thresholded keep/drop gate that works in print is trained on per-item labels: Sparse RAG
(arXiv 2405.16178, ICLR 2025) reads P("yes") after a control token appended to each independently encoded
context, drops below sigma 0.15 to 0.25, trains a LoRA r4 on labels from a two-round Gemini Ultra rater plus
PaLM2 XL critic (F1 0.821 against humans on 50 x 10 NQ contexts), and co-trains the reader; Self-RAG's critic is
GPT-4-labelled; VideoITG (2507.13353, CVPR 2026) trains on 500K temporal-grounding annotations from a GPT-4o-driven pipeline; TSPO (2508.04369, AAAI 2026)
rewards a selector with needle positions from its own synthetic splicing, which is exactly our label source.
Among the 40 papers citing Sparse RAG none removes the supervision from a parallel-encoded gate. Every
label-free method ends in a budget: Superposition Prompting (2404.06910) is greedy top-k by query likelihood
(k=1 best, 0.465 falling to 0.423 at k=8; its attention-based score 0.218 is no better than no pruning 0.224);
DAFS (2607.15689) is top-K; SeViLA's zero-shot P("yes") localizer gains 0.5 points where the one self-refined on
the answerer's labels gains 1.9 (NExT-QA uniform 62.4, zero-shot 62.9, self-refined 64.3, with human pretraining 64.6). Our own record says the same thing twice: the model's verbal yes/no ranks
needles at AUC 0.92 to 1.0 per task but the literal "yes" keeps 13 percent of them, and a within-question
relative rule reaches exact set on 78 percent of questions; the evidence head L24h20 ranks at AUC 0.99 and
cannot threshold because its mass dilutes as 1/N.

Today's measurement sharpens this. A probe fitted on VNBench's construction labels transfers to HERBench's
human evidence intervals (real footage, no splices) at L20 AUC 0.873 (TSO) and 0.879 (SVA), flat from N=8 to 128,
against the in-domain 5-fold 0.91 and the question-blind arm at 0.53 to 0.57; its specificity at the trained
threshold is 0.77 to 0.79. The reverse probe (HERBench human labels, 12.6k frames) reads VNBench at pooled AUC
0.821 with subtitle needles at 0.989 to 1.000 and inserted pictures at 0.66 to 0.83, but specificity between
0.63 and 0.98 depending on the task. A probe from MMReD renders reads VNBench at 0.587. So the direction the probe
learns is relevance, not an artifact of the splice (an artifact detector would sit at chance on HERBench), it
transfers across footage and across needle kinds for ranking, and the threshold does not transfer across needle
kinds. That is the same pattern as the label-free signals: ranking is cheap and general, calibration is what the
labels buy. The paper should say "label-free ranking, supervised threshold", report the gate in kept-frame units,
and show the label-source table of section 4.

### Q2. Does the gate need the whole 7B as its feature extractor?

No, and the claim should not be made. What it needs is a question-conditioned, per-block state; the 7B's is free
because the fenced scan runs anyway. The record's D4 cell on MMReD fixed this: the same logistic head on the
end-of-vision state reads 0.51 (chance) from a joint forward in the paper layout, 0.84 from a joint forward with
the question first, and 0.975 (AUC 0.998) from the fenced question-first scan, holding 0.973 at N=128; QGATE showed
question-blind encodings expose nothing linear (AUC 0.5 to 0.7). In print, small question-conditioned selectors rank
as well as big ones: TSPO's 3.5M-parameter head on frozen CLIP-L, Sentinel's logistic regression on a 0.5B proxy's
attention (proxy size 0.5B to 7B makes no systematic difference, 2505.23277), DAFS's training-free 2-3B mid-layer
attention beating trained selectors with a 7B answerer. All of them run under a top-K budget, none reports a
threshold across N. The cheap row that settles it for us: the same probe on a Qwen2.5-VL-3B fenced slot and on
SigLIP question-frame cosine, reported as AUC and false frames per question at N=16/64/128 next to the 7B's 0.989
(about 2 GPU-hours). Expected outcome from the literature: ranking parity, calibration gap. The defensible sentence
is "the gate needs the question-conditioned block, not a second model; on the 7B it costs 10^-6 of a scan that is
paid for the read".

### Q3. A per-frame gate cannot see relevance that depends on another frame. What handles that?

First the taxonomy, checked against our own label rules (`core/mmred.py::evidence_frames`, the VNBench task
table). Six classes; a question-conditioned per-frame gate already covers four of them in principle:

| class | example | per-frame gate | who resolves it |
|---|---|---|---|
| anchor-relative, anchor visible per frame | "the frame after the red car" | fires on both referents (both are in the prefix) | the packed re-read orders them (ord 0.99 / 0.99 / 0.96 at N=128 behind the fitted gate) |
| absence / negation | "which did NOT happen" | fires per true option | the re-read eliminates |
| order-relative | VNBench ord_*, HERBench TSO | every needle self-evident | the re-read |
| composite needles | cnt_edit2: count inside 4 images, sum | every needle self-evident | the reader (its 0.21 wall is perception) |
| duplicates / "how many distinct" | rooms_visited | keeps every copy; must not de-duplicate | the reader; a k-wall, not a gating problem |
| every-frame aggregates | where_spend, MPDR, AC | relevance is not sparse; gate is a no-op | the aggregation k-wall |
| target identifiable only through an anchor's content | "what the clue frame names" | blind by construction | needs conditioning on the anchor |

The last class is the only genuinely relational-selection case, and no task in VNBench, HERBench lite_v2 or MMReD
binds it (HERBench's AG* rows are text-answerable). The panel's two relational designs, a 2.5M-parameter set
encoder over the N block states and a second fenced scan whose prefix holds the first scan's kept frames, both
failed the same test: on the benchmarks we evaluate they can only reproduce the pointwise gate (the critics'
simulated gain from a perfect false-positive filter is about 1.5 points pooled at N=128, inside the fold-to-fold
spread), and the cheapest mechanisms for temporal relations are zero-forward rules on the existing states (HiMu's
sequence rule, a predicate's score times the running max of the preceding predicate's score with a symmetric factor for
predicates still to come, arXiv 2603.18558; timestamp dilation of the kept set). Precedents
that do condition selection on earlier picks are all supervised with gold chains (MDR 2009.12756, Beam Retrieval
2308.08973, whose two-head hop-2 scorer beats a shared head 74.18 vs 73.11) or are budgeted set selectors (Moment-DETR
2107.09609, QD-DETR 2303.13874, VideoITG Variant C, HFS 2512.11534). Relational gains in print are thin: GenS +9.6
on action order but -2.9 on action count; Logic-in-Frames +0.3 precision; HiMu +0.9.

Recommendation: state the table in the paper, concede the last class as untested, and build a relational cell
only as a time-boxed rebuttal experiment (section 4, week 3): anchored VNBench questions with decoys of the target's
type, pointwise gate vs HiMu composition vs an anchored second scan, with the decoy rate set so the pointwise kept
set sits inside the measured collapse regime (about 48 to 64 frames at N=128). If the pointwise-gated re-read is
within 3 points of the conditioned one, the paper says relational selection is not needed on any benchmark we
have and names the content-hop case as future work.

### Q4. After gating, rely on the frozen backbone or fine-tune a reader?

Freeze the reader for the method; report the LoRA reader as a labelled extra row; raise the counting ceiling by
changing the answer format, not by training a longer-range reader.

The frozen packed re-read already is the headline (0.87 to 0.88 flat from N=8 to 128 vs plain 0.84 to 0.69;
circular 0.677 at N=64 vs 0.556 plain; 0.821 at N=256 behind the merge rule, published Qwen2-VL-7B 0.338). It is
also the field's deployment norm (Frame-Voyager, TSPO, VideoITG, GenS keep the answering LLM frozen) and a real
difference from Sparse RAG, which co-trains the reader. Fine-tuning is necessary in print when the reader is
damaged by what it receives (one in-topic distractor costs GSM-IC 95.0 to 72.4; RAFT 35.28 vs 4.41 on HotpotQA) and
harmful when the SFT distribution is narrow (SFT -5.6 to -79.5 points on rule-OOD variants vs RL +3.0 to +11.0, 2501.17161;
SFT on Qwen2.5-VL-7B lowers TempCompass by 3.0 before RL recovers it, Video-R1 2503.21776; MMReD's own LoRA r32
SFT on N <= 16 "is not enough for generalization" and their GRPO, on a different base model, is worse than SFT). RAFT-style exposure to the
gate's false positives is standard (RAFT, RetRobust 2310.01558, RA-DIT) and buys robustness to random
distractors but not to in-topic near-duplicates, which is what our false frames are (adjacent frames); a lower
false-positive rate is worth more than reader hardening.

The generalization limit in k is real and documented: no published direct-answer reader trained to count to 8
counts to 32 (Golowich 2502.16792, measured on sparse parity: k_test > K_train "deteriorates rapidly"; Yehudai 2407.15160: the 1/count readout
needs width proportional to n; Chang and Bisk 2405.20131: RoPE falls to about 53 percent on selective counting).
Our own gated LoRA readers (trained on k <= 8) are exact to k about 16 at every N and then wall, which is the
predicted 2x. The only documented 2 to 3x count extrapolations use serial emission: Pointing (2603.21746: LoRA r32
on Qwen2.5-VL-7B, trained counts 1 to 9, OOD 10 to 18, direct 32.0 percent vs enumerated 94.96 percent) and
scratchpad arithmetic. So the in-scope lever for counting beyond the frozen ceiling (inserted images beyond k=4
at 0.58 to 0.69; objects inside images 0.21) is the output format on the packed re-read: enumerate the matching
kept frames, then state the count; per-kept-frame identifiers (NumPro-style pixel numbers, 2411.10332,
training-free on Qwen2-VL-7B; or "Frame i" captions, Chain-of-Frames 2506.00318) give order and de-duplication a
handle. Training-free first; if a LoRA is needed, train it on enumeration traces at k <= 8, N <= 16, from the gate's
actual kept sets, and label the row a budget deviation. Never a soft gate (SOFTGATE), never GRPO within this deadline.

Scope verdict: reader fine-tuning is in scope only as the "gate converts the N-wall into a k-wall" row (already
measured) and, if time permits, the enumeration-format counting cell; it is not part of the method and not part
of the generalization claim.

## 2. The professor's proposal (one joint pass emits a 0-to-1 significance vector; pass two sees frames plus vector)

Steelman: produce the vector with a per-frame head rather than generated text (text ranks but never thresholds),
compute it from a joint view so a frame's score can depend on other frames, inject it into pass two as a hard
keep/drop. Each of the three ingredients has a measured bound:

| ingredient | record cell | number |
|---|---|---|
| per-frame significance from a joint forward over all N frames | DIAG D4, same probe, four layouts | joint, paper layout 0.51; joint, question first 0.84; fenced question-first 0.975 (AUC 0.998), 0.973 at N=128 |
| soft 0-to-1 weights instead of a hard drop | SOFTGATE | count = k + (N-k)/B; a reader retrained under the soft gate calibrates to the residual; only exact zeros are N-invariant |
| all N frames kept in pass two | week-3 positions ablation | same kept frames, zero fillers, original positions: k=5-6 counting 0.79 to 0.17 at N=128 |

The joint 7B forward also costs about 77x the attention FLOPs of the fenced scan at N=256 and is the regime the
share law describes. What survives is the joint view as a small set scorer over the fenced block states, and the
panel's version of exactly that (2.5M parameters, residual on the probe, thresholded at 0.5, feeding the packed
re-read) was judged by its critic to have no benchmark cell where it can win (section 1, Q3). The cheap settling
experiment is the anchored cell of week 3. For the meeting: the proposal is the soft, single-context version of
what already works, and each softening costs something already measured; its one good idea, a frame score that
sees other frames, belongs in the gate and only on a task that needs it.

## 3. Recommended design

```
frames f_1..f_N ──► fenced scan (question in the shared prefix, one block per frame,
                     per-block position reset; N independent forwards of prefix + one frame)
                         │ per block: layer-20 <|vision_end|> state h_t   (free; the scan is paid for the read)
                         ▼
                 gate: keep_t = [ w·z(h_t) + b > 0 ]      logistic probe, 3,584 weights, hard threshold,
                         │                                 trained by BCE on construction labels; no gradient
                         │                                 ever crosses the gate; no top-k, no budget
                         ▼
                 merge_kept_runs (one frame per 1 s window of video time)   zero parameters
                         │
                         ▼
                 frozen packed re-read: kept frames in time order, official layout, consecutive positions,
                 the model generates the answer (letters readout where the backbone answers in prose)
```

What changes relative to the three weeks on record is the evidence around the gate, not the gate:

| piece | why | cost |
|---|---|---|
| label-source table on ONE gate: construction timestamps / reader-derived labels (in-scan verdict rank binarised by the answer's cardinality) / the model's own yes/no with the within-question rule / cross-dataset probes (today's fits) | the peers' Q1; no selector paper has this table | judge recapture at planted N=16, about 0.5 GPU-h; the rest CPU |
| kept-frame accounting: false frames per question vs N, exact-set rate vs N, leak curve (re-read accuracy vs number of leaked frames) | per-frame 0.989 compounds to 0.90 exact sets at N=128; a reviewer who knows Sparse RAG's gold-label row will ask | CPU on existing eval.csv files |
| question-blind (fenced_qlast) capture on VNBench planted N=16, 5-fold by haystack | the only direct test that the 0.989 gate is not a splice/overlay detector; today's HERBench transfer argues it is not, the control is cheap | about 0.5 GPU-h |
| gate-size row: same probe on Qwen2.5-VL-3B fenced slot and on SigLIP cosine | the peers' Q2 | about 2 GPU-h |
| per-task relational table (section 1) plus the time-boxed anchored cell | the peers' Q3 | CPU; anchored cell about 3 GPU-h |
| enumeration-format counting on the packed re-read, training-free | the counting ceiling without a new generalization limit | about 2 GPU-h |

How it avoids each negative result: (a) SELFGATE and (b) SOFTGATE do not arise because no gradient ever crosses the
gate and the reader sees exact drops; (c) the probe is thresholded on a block-local state, which cannot dilute, and
the record shows the threshold flat from N=16 to 64 (0.989 / 0.990) with false frames linear in N as a constant
specificity predicts; (d) the block is question-conditioned (prefix); (e) the read is the packed re-read, never the
fenced read; (f) the wiring is cited as Sparse RAG's, the claims are the measurements.

Cost at N=128: 128 block forwards of about 182 tokens plus one short re-read (about 3 to 6 frames); at N=256 under the
official grid, 256 block forwards plus a re-read of about 3 frames after merging. The gate is 3,585 parameters.

Why the panel's alternatives lost (each verdict is traceable in `design_panel_critiques.md`):

| proposal | what killed it |
|---|---|
| null-anchored verdict gate (zero parameters, the model's own yes/no calibrated by a null frame and the question's peak) | re-simulated today on the 9,034 stored verdict scores (`outputs/_scratch/gate_design_1014/nav_sim.py`): pooled recall 0.89 / rejection 0.95 / exact set 0.77, false k=0 on 8 percent of questions, ordering-picture tasks at recall 0.69 with 26 to 28 percent false k=0; 2.5 to 6 leaked frames per question at N=128. Same ROC as the already-failed max-2 rule. Prior art: SeViLA's zero-shot localizer plus contextual calibration |
| 2.5M-parameter relational set gate over block states | its real-data win (needle runs under the dense grid) is already delivered by the zero-parameter merge rule (circular 0.821); no benchmark task binds the relational part; architecturally QD-DETR / VideoITG Variant C on fenced states |
| reader-derived labels (in-scan verdict rank x cardinality from the answer, round-trip checked) | sound and cheap, but on VNBench the answer fixes k for every task, so it can only show that timestamps are reconstructible from self-evident needles; kept as the label-source ablation row, not as a claim |
| anchored second scan (prefix = question + first-round kept frames) | the frozen re-read already orders four kept needles at 0.96 to 0.99, so the constructed "right after X" cell cannot show a downstream gain; 2x scan cost on every ordinary row; the content-hop case where it would bind was never scheduled |
| synthetic needles on disjoint footage (leave-every-benchmark-out gate) | within VNBench itself the threshold moves with the needle kind at AUC 1.0 (leave-one-task-out specificity 0.66 to 0.90); a VNBench-recipe generator tested on VNBench is in-distribution by construction; today's cross-dataset fits show the same ranking-transfers / threshold-does-not pattern |
| professor's soft vector, steelmanned as a set scorer | section 2 |

## 4. Experiment ladder (four weeks to 2026-11-13)

Cells are ordered by cost and by what they decide. GPU cells go through the sbatch-submit loop after Tal's OK;
CPU cells on `4h_0g` need none.

| week | cell | decides | GPU-h | success band | kill / fallback |
|---|---|---|---|---|---|
| 1 | label-source table from existing captures and today's transfer fits (CPU): timestamps / HERBench-label / MMReD-label / model's own yes-no + within-question rule; per task, per N | Q1 wording | 0 | table exists; ranking AUC >= 0.85 on point-evidence tasks for every source | none; it is a report |
| 1 | kept-frame accounting and leak curve from `s4_method` eval.csv files (CPU) | the pipeline-level N-invariance claim | 0 | false frames per question linear in N at the measured specificity; rows with exact set 0.89 at every N | none |
| 1 | fenced_qlast capture, VNBench planted N=16, 5-fold by haystack | artifact control | 0.5 | L20 AUC <= 0.70 (HERBench 0.53-0.57, MMReD 0.49) | AUC >= 0.90: every VNBench gate claim is rescoped to "needle detection"; the HERBench transfer rows become the relevance evidence |
| 1 | judge recapture at planted N=16 with `--judge` (verdict logits beside the slot states), fit the same probe on reader-derived labels (verdict rank binarised by the answer's cardinality) | Q1 ablation row | 0.5 | per-frame >= 0.98, AUC >= 0.995 held out by haystack at N=16 and on the existing N=64 capture | < 0.95: report as the measured gap between construction labels and reader-derived labels |
| 2 | gate-size row: Qwen2.5-VL-3B fenced slot (capture N=16+64) and SigLIP cosine (CPU) with the same probe | Q2 | 2 | AUC within 0.02 of the 7B; false frames per question reported | any outcome is reportable |
| 2 | 1 fps rows (published setting), plain and two-pass, circular | the leaderboard footnote the peer page lists as next | 12 | two-pass above plain by >= 10 points circular | none; it is the comparability row |
| 2 | gate fit + two-pass on Gemma-3-12B and InternVL3.5-8B (planted N=16 fit, ladder) | backbone independence | 10 | E-level flat in N on both, as on Qwen | a backbone that fails is reported as such |
| 3 | anchored relational cell (constructed on the 450 ord_* videos: anchor + target + decoys of the target's type, decoy rate so the pointwise kept set is 48 to 64 frames at N=128, k=0 arm): pointwise gate vs HiMu composition vs anchored second scan, two-pass read behind each | Q3 | 3 | conditioned gate beats pointwise by >= 3 points at N=128 on the anchored rows AND keeps the kept set O(k) | within 3 points: relational selection dropped from the paper with the per-task table as the argument; hard stop after 10 days |
| 3 | enumeration-format counting on the packed re-read (list matching kept frames, then count; with and without frame identifiers), VNBench cnt_* k=1..6 and MMReD counting at N=32/64/128 | the counting ceiling lever | 2 | k=5-6 inserted images from 0.58-0.69 toward >= 0.85 | no gain: the ceiling is stated as the reader's |
| 4 | writing: commit the three weeks, log the results, collapse figure with the two-pass curve, prior-art citations | the paper | 0 | | |

Total about 30 GPU-hours. The N=256 circular row behind the merge rule (0.821) already exists; the positions
follow-up and the 32B reader ceiling probe named on the peer page continue as scheduled.

## 5. Papers that give us exactly what we want

| paper | what it gives us | how we use it | caveat |
|---|---|---|---|
| Sparse RAG, Zhu et al., arXiv 2405.16178, ICLR 2025 (cite the camera-ready: the arXiv v1 has PopQA and QMSum only) | the structural twin: block encoding, per-context positions, in-model P("yes") gate after a control token, fixed sigma; labels from Gemini Ultra + PaLM2 XL (F1 0.821 vs humans); gold labels at inference lift QMSum 23.96 to 26.76 | cite as prior wiring in the method section; the label-source argument; "gate quality bounds the read" | 20 contexts, no gate precision/recall, no N sweep |
| Superposition Prompting, Merth et al., 2404.06910, ICML 2024 | the training-free twin; label-free path pruning ends in top-k; attention-based scoring no better than none | cite as the label-free alternative and why it needs a budget | k chosen on the evaluation sets |
| SeViLA, Yu et al., 2305.06988, NeurIPS 2023 | NExT-QA with 4 frames: uniform 62.4, untrained zero-shot P("yes") localizer 62.9, human-moment pretraining 63.6, self-refined on answerer labels 64.3, both 64.6 | the precedent for "untrained self-gate is weak, reader-derived labels work"; the ancestor of our verdict prompt | 4 frames, 44 s clips |
| VideoITG, 2507.13353, CVPR 2026 | per-frame classification head on pooled Video-LLM frame tokens (Variant C); 500K temporal-grounding annotations from a GPT-4o-driven pipeline; LLaVA-Video-7B 58.4 to 62.9 | the structural twin on video; cite before any reviewer does | fixed budget, no per-frame numbers vs N |
| TSPO, 2508.04369, AAAI 2026 | 3.5M-parameter selector on frozen CLIP-L, GRPO, reward includes needle positions from its own synthetic splicing; MLVU 70.3 to 76.3 | same label source as ours; the small-selector precedent for Q2 | fixed top-K |
| M-LLM based frame selection, 2502.19680, CVPR 2025 | a fixed Qwen2-VL-7B labeler's True/False probability, per-video normalised, as spatial selector labels (GPT-4o-mini temporal labels); LongVideoBench 50.9 to 56.0 at 8 frames with Qwen2-VL | the recipe for the reader-derived label row | separate selector (Qwen2.5-1.5B + SigLIP) |
| DAFS, 2607.15689 | training-free 2-3B MLLM mid-layer attention, top-K (32 in the main table), beats the authors' BOLT and AKS reproductions with a Qwen2.5-VL-7B answerer (61.1 to 67.4 Video-MME); mid layers rank best (Hit@32 0.943 at L15 vs 0.847 at the last layer, Qwen3.5-2B) | Q2 (small proxy ranks), layer choice support for L20 | top-K, no threshold |
| Sentinel, 2505.23277 | logistic regression on a 0.5B proxy's attention; proxy 0.5B to 7B no systematic difference | Q2 | budget-ranked |
| CSM, 2509.21359, NeurIPS 2025 | reader leave-one-out influence CI = loss(list minus c_i) - loss(full list), keep CI > 0 with no tuned threshold; a list-wise surrogate predicts CI | the reader-counterfactual label precedent; its additivity caveat (pointwise decomposition; duplicates get CI about 0) | text QA |
| ReQuest, 2607.01737 | pseudo-labels P(gold given segment) - P(gold given blank visual) from the MLLM itself | the null-visual baseline for any likelihood label | top-K |
| FrameOracle, 2510.03584 | leave-one-out VLM loss labels; "how many" learned only with a labelled per-k sweep and a cost penalty | why a learned count is not label-free | |
| Found in the Middle, 2406.16008, ACL 2024 Findings; ICR, 2410.02642; Calibrate Before Use, 2102.09690 | null-reference calibration (dummy document, content-free query, "N/A"): Recall@3 0.2052 to 0.6832 at K=20 | precedents if a null-frame k=0 rule is ever tested (MMReD answer-0 cells only) | rank, never threshold |
| HiMu, 2603.18558 | zero-forward composition of per-frame predicate scores (sequence rule with running maxima of preceding and following predicates; a decayed-kernel right-after rule, kappa 2); +6.7 over T* on LongVideoBench with Qwen3-VL-8B | the first relational baseline in the week-3 cell | training-free, external experts |
| Beam Retrieval, 2308.08973, NAACL 2024; MDR, 2009.12756, ICLR 2021 | conditioning hop 2 on hop-1 picks; two heads beat a shared head 74.18 vs 73.11 | the anchored-second-scan arm, if the week-3 cell shows a gap | supervised gold chains |
| GenS, 2503.09146, ACL 2025 Findings; Frame-Voyager, 2410.03226, ICLR 2025 | the prior-art null on counting: GenS Action Order +9.6 but Action Count -2.9; Frame-Voyager no counting gain | cite when presenting counting behind the gate | |
| From Artificial Needles to Real Haystacks, 2406.19292 | synthetic-needle fine-tuning transfers to real long-context tasks (+10.5 percent MDQA) without hurting TriviaQA | the citation for training on construction labels | text only |
| Visual Haystacks / MIRAGE, 2407.13766, ICLR 2025 | a BCE relevance head trained on 600K synthetic composites with distractors | the distractor rule if a synthetic set is ever built | fixed budget, images |
| RAFT, 2403.10131; RetRobust, 2310.01558; The Power of Noise, 2401.14887 | training the reader with distractors is standard and buys random-distractor robustness, not in-topic robustness | Q4: why a lower false-positive rate beats reader hardening | |
| LoRA Learns Less and Forgets Less, 2405.09673; SFT Memorizes, RL Generalizes, 2501.17161; Video-R1, 2503.21776 | narrow SFT collapses rule-OOD (-5.6 to -79.5 vs RL +3.0 to +11.0); LoRA limits the damage; SFT on Qwen2.5-VL-7B lowers TempCompass by about 3 points relative to its CoT row before RL | Q4: keep the reader frozen for the headline | |
| Golowich et al., 2502.16792; Yehudai et al., 2407.15160; Chang and Bisk, 2405.20131 | the k-wall is predicted: generalization only for k_test <= K_train; the 1/count readout needs width proportional to n | explain the gated LoRA reader's wall at about 2x the trained k | toy / text transformers |
| Pointing, 2603.21746; NumPro, 2411.10332, CVPR 2025; Chain-of-Frames, 2506.00318, CVPR 2026 | serial emission extrapolates counts (1-9 to 10-18: 32.0 vs 94.96 percent); pixel frame numbers are training-free on Qwen2-VL-7B | the enumeration-format counting cell | single image / grounding |
| Expire-Span, 2105.06548; hard concrete L0, 1712.01312; SeerAttention, 2410.13276; sigmoid-attention eviction study, 2608.23296 | why answer-loss-only gates never close and soft gates calibrate to the residual; every working learned gate has a budget or a closing force and a stochastic or multiplicative path | the paragraph that retires SELFGATE and SOFTGATE as design options | |
| Das et al., 2601.07812; Block-Attention, 2409.15355; Anil et al., 2207.04901; PEVLM, 2506.19651 | the mask, the block encoding, the oracle distractor mask, the parallel video encoding | prior wiring, already in the contributions ledger | |

## 6. Paper claim and limitations

Claim sentence: "a thresholded per-frame gate on the frozen model's fenced, question-conditioned block state holds
its per-frame accuracy from N=16 to 256 while every label-free ranker fails on calibration; behind it a frozen
packed re-read recovers the perception ceiling at every length, and the gate's ranking transfers to human evidence on
real footage while its threshold does not." Prior wiring credited to Sparse RAG, Superposition Prompting, Anil et al.
and Das et al.; the structural twin on video credited to VideoITG. Conceded: the gate is supervised by construction
labels (as every working thresholded gate is); no benchmark we evaluate binds relational selection; counting beyond
the frozen reader's ceiling is the reader's wall, reported as such; cross-dataset deployment needs one scalar refit
per needle kind.

## 7. What we did not resolve

- Whether the question-blind arm on VNBench sits near chance (the direct artifact control; week-1 cell).
- Whether reader-derived labels reproduce 0.989 on the same probe (week-1 cell; on VNBench the answer fixes k, so
  the row is a robustness check, not a label-independence claim).
- The content-hop relational case has no benchmark and no cell; the week-3 anchored cell tests temporal anchoring only.
- The merge rule is VNBench-exact (one-second needles); its general form (non-maximum suppression on the gate score)
  is written but not run.
- The HERBench captures are at native resolution and VNBench's at max_pixels 151,200; today's transfer rows carry
  that confound (it did not prevent 0.87).

## Addendum, evening of 2026-10-14: two cells that landed after the panel

**Read geometry (jobs 181245-181254; same fitted gate and kept frames; folds pooled, planted N=16 / 64 / 128).**

| read of the kept frames | N=16 | N=64 | N=128 | ordering at N=128 | counting at N=128 (edit1 / insert1) |
|---|---|---|---|---|---|
| packed, full attention (the method) | 0.873 | 0.866 | 0.880 | 0.99 / 0.99 / 0.96 | 0.98 / 0.83 |
| packed positions, fenced (kept frames cannot see each other) | 0.473 | 0.470 | 0.507 | 0.26 / 0.17 / 0.25 | 0.28 / 0.28 |
| block-0 positions, fenced, plus timestamp captions | 0.491 | 0.491 | 0.525 | 0.25 / 0.25 / 0.28 | 0.28 / 0.31 |
| packed frames, question at its far original distance | 0.882 | 0.859 | 0.858 | 0.98 / 0.95 / 0.92 | 0.91 / 0.73 |
| frames spread to original positions, question adjacent | 0.858 | 0.827 | 0.819 | 0.99 / 0.97 / 0.89 | 0.79 / 0.53 |

Ordering and counting need the kept frames to attend to each other and to sit close together; the question's distance costs little,
and timestamps do not rescue a fenced read. This corrects the week-1 reading (position reset alone): the fence is the first cause,
positional spread the second. It also bounds the design space: any method must end in a joint read of the kept frames at packed
positions, so the two-pass form (N question-conditioned block forwards, one joint O(k) read) is the minimal one, not a workaround.

**Qwen2.5-VL-32B counting ceiling (job 181256; needle frames only, 450 rows):** cnt_edit1 0.993, cnt_insert1 0.953 (7B 0.81),
cnt_edit2 0.340 (7B 0.18 to 0.21); k >= 5 at 0.971 (7B 0.58 to 0.69). The inserted-image counting wall moves with reader size;
counting objects inside images barely does.

## Provenance

Transfer fits: `outputs/scaleup/herbench/qwen2.5-vl-7b/s2_fit/vnbench_to_herbench.json` (181599),
`outputs/scaleup/vnbench/qwen2.5-vl-7b/s2_gate/fits/herbench_to_vnbench.json` (181600),
`outputs/scaleup/vnbench/qwen2.5-vl-7b/s2_gate/fits/mmred_to_vnbench.json` (181601); null-anchored verdict simulation
`outputs/_scratch/gate_design_1014/nav_sim.py` on `s1u_unit/D_frame_lo/20261010_225007_D-frame_reslo/units.csv`.
Literature: `docs/gate_design_2026-10-14/research_digest.md` (eight lenses, ids and verification level per paper);
panel: `docs/gate_design_2026-10-14/design_panel_critiques.md` and `design_panel_full.md`. Record numbers: STATE entries
2026-10-11 01:40 / 15:45, 2026-10-12 09:30, 2026-10-13 12:30; DIAG D4 `outputs/diag/INDEX.md`; SELFGATE / QGATE /
SOFTGATE / FIXEDK entries in `RESULTS.md` (2026-09-14 to 09-21).

# What "Beneath the Surface of Chains-of-Thought" offers our single-pass aggregation question

Reading note, 2026-09-07. Paper: Jeong, Hwang, Han, Gu, Oh, Kim (KAIST / NAVER AI Lab),
*Beneath the Surface of Chains-of-Thought: A Mechanistic Interpretation of Reasoning Operations in
LLMs*, arXiv 2609.04753v1 (4 Sep 2026), code github.com/naver-ai/beneath-cot.
Status of everything in §4: **proposal, nothing run**; predictions written before any run.

---

## 1. What the paper establishes

Setting: step-by-step solutions on DAPO-Math-17K and TheoremQA from Qwen2.5-7B (base), Qwen3-8B,
Gemma4-31B, plus Llama-3-8B; 500–700 correct traces per model. GPT-5 segments each trace into
non-overlapping operation spans from a Pólya taxonomy (eight main labels: extraction, direct mapping,
decomposition, recall, deduction, algebraic manipulation, arithmetic computation, final answer).
Human check on 84 spans: κ = .715 vs. the human majority.

Findings, with the numbers that matter:

1. **Operations are linearly separable in hidden states, peaking in middle layers.** L2-norm → PCA-128
   → LDA → one-vs-rest centroid direction d_c; held-out AUROC .85–.95 per operation, layers 16–22
   for six of eight operations on Qwen3-8B. Survives without LDA (PCA-only .94/.72 AUROC/AUPRC),
   beats text-only (.85/.55) and position-only (.72/.28) classifiers, holds within position bins
   (.92–.97), on spans full of competing-operation vocabulary (.92/.85), and transfers to GPQA and
   MATH-500 without refitting (.94/.76, .95/.80).
2. **The signal is span-distributed in the middle layers.** Intra-span variance of token alignment
   scores is high in early layers (cue-token-local) and falls toward the middle layers.
3. **Identical tokens take operation-specific representations.** Shared function words ("is", "the",
   "so") separate along operation directions from the middle layers on.
4. **The chunk-onset representation depends causally on the immediately preceding local context.**
   Masking the onset token's attention to the preceding 30 tokens lowers alignment with the chunk's
   operation (Δ = −0.44, p < .0001); ordering of effect sizes: random earlier block < preceding chunk
   < immediately preceding tokens. Recall is the exception (depends on broad context). A literary
   discourse control shows no such drop (Δ = +0.04).
5. **Operation identity persists under factual errors** (AUROC .955 vs .971 mean-pooled), attenuated
   most for deduction and arithmetic.

Their own annotation example 3 is a chain-of-thought *count*: "for n = 1…9 check divisibility …
count it" is one long LOGICAL-EVALUATION span (repeated per-item check) followed by FINAL-ANSWER
("Total qualifying: 3"). Example 2 is decomposition → per-item arithmetic ×3 → combine → final answer.

## 2. What the paper does not establish

- No aggregation mechanism. It never asks how information from many steps is combined; the only
  cross-step result is (4), a local dependency of each chunk on the previous ~30 tokens.
- Diagnostic only. Probes are correlational; the single causal result is on probe alignment under
  attention masking, not on behaviour. No steering, no intervention that changes an answer.
- "Reasoning models" here means step-by-step generation; Qwen2.5-7B base is included. Thinking
  mode was left at checkpoint default.
- Labels are GPT-5 annotations of *textually expressed* operations, not latent states (their words).

So it cannot tell us that reasoning models have a stronger aggregation operator. It tells us how the
serial program is organised in the residual stream.

## 3. What it means for our question

**Reasoning models do not aggregate better in one pass. They serialize into per-item evaluation
chunks whose operation state is set by the immediately preceding text, and they keep the tally on
the tape.** In operation terms, a chain-of-thought count is

    DECOMPOSITION ("check each item")
    → for each item: INSTANTIATION ("n = 3:") → LOGICAL-EVALUATION ("divisible, count it")
    → FINAL-ANSWER (read the tally: "Total qualifying: 3").

Each per-item chunk is a fresh evaluation whose onset is controlled by the ~30 tokens before it
(finding 4), executed with the network's full depth, writing a verdict token into the context. The
model never evaluates many items inside one softmax read; that is exactly the read our analysis shows
cannot bind and cannot count (selectivity and readout bottlenecks, PREREG_AGG bundles C–E). Our
chain-of-thought test agrees: Qwen3-8B thinking counts to K = 32 with ~2.2k tokens and collapses
with 64 confusable distractors, because each chunk's single-target retrieval fails, not its evaluation.

**Bind-then-count is the parallel form of that serial program.**

| CoT operation | Serial mechanism (paper) | Bind-then-count |
|---|---|---|
| decomposition | one chunk per item, in sequence | fence: one block per unit, in parallel |
| instantiation + onset | preceding ~30 tokens set the operation state | unit tokens immediately precede the carrier; question in the prefix |
| logical evaluation | full-depth computation inside the chunk | carrier verdict, full depth, no competitors |
| "count it" on the tape | tally written as tokens | z_i > 0 |
| final answer | reads the tally | sum of verdicts, linear by construction |

This is a framing gain for the paper (Sec. 8.1 of the outline can say the construction parallelises
the operation sequence reasoning models execute serially) and it points to three things the
construction does not yet do.

## 4. Design implications, as pre-registered proposals

### A. Training-free fenced judge (the operation cue makes the verdict the model's own)

In the serial program the verdict token is produced by the model itself ("count it"). Under the fence
with position reset each block is an independent prompt, so the model's own zero-shot yes/no is
computable for all units in **one forward**: block_i = "\n" + u_i + " Is C in R here? Answer:",
verdict z_i = logit(" Yes") − logit(" No") at the last block position, count = Σ 1[z_i > 0].
No trained parameters at all. August evidence (CONDMASK_REPORT.md Addenda 3, 6): the direct visual
question "Is {C} in this image?" judged 86,400 frames with agreement 1.000, recall 1.000, 0 false
keeps (the meta-question "is this frame relevant?" only .73); the text judge on the 1.5B pilot agreed
with oracle labels 93–96% **[check model]**. Those were N separate forwards; the fence makes it one.

Predictions. (A1) Vision, Qwen2.5-VL-7B, rendered MMRED N = 8…64: per-frame verdict ≥ .99, exact
count ≥ .90 at every N, training-free, one forward. Falsifier: exact < .70 at N = 32. (A2) Text,
Qwen2.5-7B, MMRED N = 8…1024: per-unit verdict .93–.98 → exact count below .5 by N = 32 (the
(1−ε)^N law); the trained 2d+1 head stays necessary for text. Falsifier for the *reading*: text
per-unit ≥ .995 (then the trained head is not needed anywhere). (A3) Cue wording matters as in
August: the direct question beats the meta-question by ≥ .05 per-unit accuracy.

What it buys: a positive training-free, single-forward aggregation result on vision (the mandate
that has so far only produced negatives), and the cleanest statement of where training is needed.
Cost: ~20 GPU-min text, ~1 GPU-h vision (reuse `train_bindcount*.py` layout code; no training loop).

### B. Cue adjacency (finding 4 applied to the carrier)

The paper says the chunk-onset state is set by the ~30 tokens right before it. Our carrier sits after
the unit; the question sits in the prefix, up to thousands of tokens away under native positions and
at a fixed offset under the reset. Test: question in prefix only vs. a short per-block cue right before
the carrier, for (i) the training-free verdict of A and (ii) the trained head, on a predicate family
where the trained verdict is not already saturated (needles HF at K, J large; or a two-condition
predicate).

Predictions. (B1) Untrained: adjacent cue > prefix-only by ≥ .05 per-unit accuracy on text.
(B2) Trained: difference ≤ .01 on MMRED (already .999) but ≥ .02 on the harder family.
Falsifier: adjacent cue ≤ prefix-only in both. Cost: ~30 GPU-min.

### C. Width per unit: parallel latent chain-of-thought

A CoT chunk spends several tokens per item (instantiate, compute, evaluate, record). Our carrier is
one token; it has depth but no width. Give each block k latent carriers (chained causally inside the
block, behind the fence), read the verdict from the last. Test on a per-unit predicate ladder needing
1 / 2 / 3 operations (e.g. "Mary in the kitchen"; "Mary in the kitchen and Bob not"; "the number of
people in the room is odd"), k ∈ {1, 2, 4}, trained head, N ≤ 32 train, test to 1024.

Predictions. (C1) k has no effect on the 1-operation predicate (verdict already ≈ 1). (C2) On the
2–3-operation predicates per-unit accuracy rises with k by ≥ .03 at k = 4 vs k = 1, and exact count
at N = 1024 rises accordingly. Falsifier: no k effect anywhere (depth suffices; width is not the
limit at 7B). Prior art to cite and differentiate: pause tokens (Goyal et al. 2024), Coconut (Hao et
al. 2025), CODI (Shen et al. 2025), SoftCoT (Xu et al. 2025): all serial latent thoughts; Multiverse
and ParaThinker (2025): parallel generated branches with a reduce step, O(N·L) generated tokens.
Ours: O(N·k) latent positions, one forward, sum reduce. Cost: ~1 GPU-h.

### D. Mechanistic bridge (optional, after the deadline unless cheap)

Extract the LOGICAL-EVALUATION / "count it" direction from Qwen3-8B thinking traces on our own
counting stories (structured traces, regex segmentation, no GPT-5), then (D1) measure alignment of
fenced-carrier states (same backbone, trained head) with that direction, verdict-positive vs
verdict-negative carriers, layer-wise; (D2) steer: add the direction to the carrier residual at the
peak layers and measure the training-free verdict of A. Prediction: alignment AUROC ≥ .8 at the
paper's peak layers if the carrier reuses the model's evaluation machinery; steering raises the
untrained verdict. Falsifier: alignment at chance. What it buys: a mechanistic statement that the
single-pass operator runs on the same axis the model uses when it reasons serially.
Cost: ~2 GPU-h plus tooling; lowest priority for 25 Sept.

## 5. Where the paper enters the manuscript

- Related work, theme "chain of thought as serial computation": add representation-level evidence
  that the serial program is organised as operation chunks with locally set state (Jeong et al.
  2026; Thought Anchors, Bogdan et al. 2025; ReasoningFlow, Lee et al. 2026).
- Method, Sec. 8.1: one sentence framing the construction as the parallel form of the serial
  operation sequence.
- Experiments: A and B if run (A1 would be the training-free headline; B a design-principle result).
- Limitations: per-unit width (C) if not run.

---

## 6. The shortcut program (added 2026-09-07 after Gabriele's question: can a single pass shortcut what reasoning models do, and compound?)

**Thesis.** A single forward pass can shortcut exactly the *map-reduce* part of a reasoning trace:
independent per-item evaluations (map) combined by an associative operation (count, sum, max, any,
first/last, argmax). The serial residue is composition depth: state tracking and long dependency
chains, which bounded-depth attention cannot fold for arbitrary length. Theory anchors:
constant-depth log-precision transformers sit in TC0 (Merrill & Sabharwal), and majority/counting is
in TC0, so the frozen counting failure is a learned-read failure, not an expressivity one (our two
bottlenecks); chain of thought buys serial power (Merrill & Sabharwal ICLR 2024; Li, Liu, Zhou, Ma
ICLR 2024, 2402.12875); "Transformers Learn Shortcuts to Automata" (Liu, Ash, Goel, Krishnamurthy,
Zhang, 2210.10749, ICLR 2023): O(log T)-depth simulators always exist and O(1)-depth ones are
common, but learned shortcuts are brittle off-distribution.

**What chain of thought does on an aggregation task (three moves).** (i) It re-addresses one item
per step, turning a K-of-N aggregation softmax into N separate 1-of-N retrieval softmaxes. (ii) It
externalizes the accumulator into digits, so the count is never a share to be decoded; the increment
is a copy-and-add on small integers, which the model does exactly. (iii) It pays O(N) generated tokens
and O(N·L) attention, and each retrieval step inherits the softmax's selectivity limit (confusable
distractors → loops; our Qwen3-8B test). The structural shortcut reproduces (i) and (ii) without
the tape: the fence makes every softmax see one item; the sum read keeps the accumulator outside the
softmax. Everything else about the pass is unchanged, which is why it composes with the frozen model.

**Differentiation.** Learned shortcuts exist: implicit / internalized chain of thought (Deng et al.
2023, 2024, 2405.14838), distilling System 2 into System 1 (Yu et al. 2024, 2407.06023), Coconut (Hao
et al. 2025), CODI (Shen et al. 2025), Quiet-STaR (Zelikman et al. 2024), looped and recurrent-depth
models (Saunshi et al. 2025, 2502.17416; Geiping et al. 2025), and parallel generation with a merge
step (Multiverse, Yang et al. 2025, 2506.09991; ParaThinker 2025 **[verify]**; Hogwild! **[verify]**).
All of them learn the shortcut into weights or generate parallel tokens; the in-house evidence (digit
LoRA .92 → .25, mean read 1.00 → .09 at N = 1024) is that learned shortcuts fit the training range.
Ours is structural (length invariance by construction) and mechanistically grounded in which
operations are map-reduce shaped. Citations marked [verify] before use.

### Stages (pre-registered proposals; nothing run)

**S1. Trace anatomy of a reasoning model on our task** (~2 GPU-h; ICLR-feasible as one figure).
Qwen3-8B thinking traces on needle counting (K ≤ 32, J ∈ {0, 64}); regex segmentation into
plan / per-item chunks / answer (structured traces, no LLM annotator).
- S1a tally location: corrupt the digit the model wrote for the running count and re-run from
  there; prediction: the next chunk's count follows the corrupted digit (tally is on the tape, not
  latent) in ≥ .8 of cases; a residual probe for the running count at chunk ends extrapolates no
  better than ±2 beyond the probe's training range (the latent counter is depth-limited).
- S1b per-item retrieval: attention share from a chunk's instantiation tokens onto its target item
  vs. distractors; prediction: share on the target falls with J, and chunks whose target share is
  below the median precede the loop/hallucination failures with AUROC ≥ .7.
- S1c operation directions (paper's method on our labels): plan / evaluate / answer directions at
  the paper's peak layers; used by D above.
What it buys: the figure "what chain of thought does that one pass does not", the motivation for
the operator as its parallel form.

**S2. In-model tally through the model's own number geometry** (~1 GPU-h feasibility; the piece
that makes benefits compound). Numbers live on helix/Fourier features (Kantamneni & Tegmark 2025;
Levy & Geva 2024). If the sum of verdicts is written into the answer position in that basis, the
frozen model reads the count with its own digit head, and the count is a residual state a reasoning
model can continue from.
- H1 read test: patch the answer-row residual with the model's own mean state for "the answer is k"
  contexts; prediction: emitted digit = k for k ≤ 32 with accuracy ≥ .9.
- H2 geometry: those states are well fit (R² ≥ .9) by a low-frequency helix in k; so Σ verdicts →
  helix → residual is a fixed map with no learned parameters.
- H3 extrapolation: k beyond the range used to fit the helix still decodes (falsifier: accuracy
  < .5 at k ∈ [33, 64]).
What it buys: a fully in-model tally (no external Σ), and the concrete mechanism for compounding.

**S3. Operator beyond counting: parallel case analysis** (~3 GPU-h; post-deadline unless S1
lands early). Branching in the taxonomy ("x² = 4 → x = 2 or −2; check each") is a map over
hypotheses. Task family: k candidate answers, each verified in its own fenced block by a carrier
verdict, reduce by argmax/any. Prediction: exact accuracy flat in k to 256 with the fence, falling
without it; the frozen model in one pass degrades past k ≈ 4–8 like counting.

**S4. Compounding / self-improvement** (post-deadline; hypothesis, not a claim). If each pass can
aggregate, a reasoning model needs O(1) steps where it needed O(N), and traces shorten. Minimal
test: a small model with and without the operator, RL or SFT on counting-with-CoT, measuring trace
length and accuracy at unseen N; the self-improvement loop (STaR-style retraining on its own shorter
traces) only after that.

**S5. How much of real reasoning is shortcut-able** (post-deadline). On released or self-generated
traces (MATH-500, GPQA), label chunk pairs as independent (map) vs dependent, and report the fraction
of tokens in map-reduce-shaped regions; the operation taxonomy of Jeong et al. gives the chunking.
This turns "shortcut" into a measurable quantity per task family.

### What fits before 25 Sept
A and B (Sec. 4), S1 (one figure), S2 feasibility (H1–H3, one afternoon). S3–S5 are the follow-up
paper. The ICLR story stays: two bottlenecks → bind-then-count → boundary, with the reasoning-model
trace as motivation and the in-model tally as the door to compounding.

---

## 7. Outcomes (2026-09-07; full record in PREREG_AGG.md, same date)

- **A, training-free fenced judge: positive on both modalities.** Text MMRED exact 1.00 and per-unit 1.000 at every N to 1024; rendered frames
  .98 / .97 / .99 / .98 at N = 8–64 (per-frame ≥ .996); stories with confusable distractors .89–.92 with per-unit .995–.998 and AUC ≈ 1.00.
  The 2d + 1 head is a threshold calibration of an already perfect ranking. Runs: `outputs/judge_fenced`, `outputs/judge_fenced_vlm`.
- **Controls.** Fence removed, same cue: text .65 → .00, vision .31–.41, ranking degrades with N (AUC .99 → .88 text, .90 → .71 needles).
  Cue removed (question only in the prefix): text .49–.80 with AUC ≥ .999 (calibration lost, B1 supported). Vague cue: MMRED unchanged,
  confusable stories .09–.19 with over-counting, vision .62–.74 (A3 holds where selectivity is hard).
- **S1, trace anatomy.** Written tally in .97 / 1.00 of traces; corruption followed by the next numeral in .63 / .86, repaired 0 / 71; final
  answer shift +1 only .06 (redundant tallies on the tape). Chunk-onset target share .42–.45 → .07–.10 with distractors (4.5–6×); share does
  not predict which distractor traces fail (AUROC .545; 28 / 36 fail, mostly loops). Run: `outputs/cot_anatomy`.
- **S2, in-model tally.** A patched answer row is read as one token: single digits 1.00 at layer 24, 9–32 .26, 33–64 .01; helix + linear
  basis R² .66–.72 (linear .38–.44); H3 extrapolation 0.00. Boundary recorded; the sum stays outside the softmax. Run: `outputs/tally_helix`.
- **Consequence for the paper.** The method statement becomes: fence + per-unit question + the model's own verdict + sum, no parameters;
  optional 2d + 1 calibration for hard families. Outline updated (⟨TF⟩ marks). S3–S5 unchanged as follow-up work.

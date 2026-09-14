> Historical outline: this describes the earlier per-unit verdict and tally pipeline. For the current native aggregation direction, claims and unresolved tests, use [the current working story](NATIVE_AGGREGATION_WORKING_STORY.md). The length guarantees and earlier headline results below do not apply to the learned native V7–V9 method.

# Paper outline — Bind, Then Count

Working outline for the ICLR 2027 submission. Abstract deadline 18 Sept 2026, paper 25 Sept 2026
(AoE). Main text 9 pages; references and appendices unlimited; an AI-use statement is required.
Every number below traces to a dated entry in `PREREG_AGG.md` / `PREREG_OSQ.md` and a run dir; items
marked **[check]** must be re-read from the run dir before they enter the manuscript.

**Update 2026-09-07 (PREREG_AGG outcomes of the same date).** The construction is training-free where
the units are cleanly judged: with a per-block question and the frozen model's own yes/no behind the
fence, MMRED text is exact at every N to 1024 and rendered frames reach .97–.99 to N = 64, with zero
trained parameters; the 2d + 1 head is a calibration that matters only on stories with confusable
distractors (.9 → 1.0). Removing the fence collapses the zero-shot judge (text .65 → .00, vision
.31–.41), removing the cue costs calibration, a vague cue costs selectivity on the hard family. The
sections below carry the pre-update wording; adjust where marked ⟨TF⟩.

---

## 0. The story in one breath

**Goal.** Answer questions that need several pieces of evidence from one long input, in one forward
pass of a frozen model.

**Problem.** Frozen transformers stop aggregating after a handful of items. Every remedy in use
sidesteps the aggregation instead of performing it.

**Nugget.** The count travels as a small excess of attention on the matching items, read into one
node by a softmax. That read is additive in features (it cannot bind *Mary* to *kitchen*) and its
output is a share, which the model maps to a number compressively. Both limits are properties of a
one-hop mean read into a single node, not of the model's knowledge.

**Solution.** Bind first, count second: give each unit its own carrier behind an attention fence so
the full depth computes a per-unit verdict without competitors, then aggregate the verdicts with a
sum. The second stage is linear in the count by construction, so length generalization is not
something we hope for, it is something the architecture guarantees.

**Result.** ⟨TF⟩ With no trained parameters, the frozen model's own per-unit yes/no behind the fence
counts exactly at every N to 1024 on text and at .97–.99 to N = 64 on rendered frames; a 2d + 1
calibration head trained at N ≤ 32 makes it exact on stories with confusable distractors and on three
backbones. Each ingredient has an ablation that fails in the way the diagnosis predicts (no fence, no
cue, no reset, mean read). On real egocentric video the per-frame judgment is the ceiling, which
marks the scope.

One story, three parts: diagnosis → method that follows from it → boundary. The BABILong retrieval
program (distance axis) is a subplot with one figure; select-and-repack (our August program) is a
baseline and a related-work paragraph; the learned-mask negative goes to the appendix.

---

## 1. Title

Recommended: **Bind, Then Count: Single-Pass Evidence Aggregation in Frozen Transformers**

Why: three words name the method and the two-verb rhythm mirrors the two ingredients; the subtitle
says the setting. Alternatives, in decreasing preference:

- *Why Transformers Cannot Count in One Pass, and How to Make Them* (analysis-first; stronger hook, weaker recall of the method)
- *Read-Node Dispersion: Two Bottlenecks of Single-Pass Aggregation* (names the analysis; drops the method)

No acronym: the method name is already two words and searchable. "GNN" stays out of the title; the
graph lineage lives in the text where it does work.

---

## 2. Figure 1 (teaser)

Form: the problem, and how the solution changes it. Two panels, full column width.

- **(a) Emitted count vs. true count K**, three frozen models (Qwen2.5-3B, Qwen2.5-7B, Llama-3.1-8B),
  clean context and with same-domain distractors. What to notice: exact to K = 4, then a constant
  (Qwen emits 4; Llama 12); with distractors the curves start *above* the diagonal (over-count at
  K ≤ 3). Dashed diagonal = correct.
- **(b) Exact match vs. N (log axis, 8 → 1024)** for bind-then-count and its three ablations, plus
  frozen and digit-LoRA, all trained at N ≤ 32 (shaded training range). What to notice: one flat line
  at 1.00; the others fall in the order the diagnosis predicts (mean read first, then no-reset, then
  no-fence).

Caption must be readable without the text. Axis labels with units ("exact-match accuracy", "true
count K", "units N").

---

## 3. Abstract (draft, ~210 words)

Transition words are the scaffold; keep them in the final unless a sentence reads better without.

> Many questions about a long input are answered by aggregating several pieces of evidence: how many
> frames show a person in a room, whether two events co-occur, which of them comes first. Doing this
> in one forward pass is the cheapest route and the one where the model's knowledge lives.
> **Unfortunately**, frozen transformers stop aggregating after a handful of items, and every current
> remedy sidesteps the aggregation instead of performing it: chain of thought walks the items one at a
> time, pruning and retrieval hand the model fewer of them. **In contrast**, we ask what caps
> aggregation inside one pass. On three frozen 7–8B models we locate the count in a small excess of
> attention on the matching items at the question tokens, spread over many heads and layers, and
> show causally that it is the channel: clamping it fixes the answer, amplifying it raises the count.
> **However**, the channel has two bottlenecks. The read is a dot product, additive in features, so
> it matches the person but not the person-in-room conjunction and leaks partial matches; and the map
> from attention share to number is compressive. Precision, the softmax normalizer, competitor count
> and single heads are not the cause. **Consequently**, we make the per-unit judgment nonlinear and
> the cross-unit aggregation linear: each unit receives a carrier token behind a block-diagonal
> attention fence with a per-block position reset, a linear head reads a verdict off each carrier,
> and the answer is the sum of verdicts. ⟨TF⟩ With no trained parameters, the model's own yes/no to a
> per-unit question behind the fence counts exactly at every length to 1024 on text and at .97–.99 on
> up to 64 rendered video frames, where the same frozen model scores .2–.4 in one pass; a 2d + 1
> calibration head trained at N ≤ 32 makes it exact on stories with confusable distractors and on
> three backbones. Removing the fence, the cue, the reset or the sum read fails in the way the
> diagnosis predicts. On real egocentric video the per-frame judgment, not aggregation, is the
> ceiling, which marks the method's scope.

---

## 4. Introduction (≈1.25 pages incl. Fig. 1)

Rhythm: goal → problem → solution → problem → solution. One paragraph each. The reader should know
by the end why to care, what is unsolved, and what we do.

1. **Goal.** The aggregation question, with the running example ("how many frames was Mary in the
   kitchen?"). Where it appears: long-document QA (multi-fact bAbI tasks), video QA (HERBench action
   counting), agents reading logs. Why one pass: no O(N) generated tokens, no recall-limited
   retrieval stage, and the model's knowledge is applied to every unit.
2. **Problem.** Frozen models answer exactly to 4–8 items and then emit a constant (Fig. 1a).
   Chain of thought, pruning and retrieval each serialize or shrink the problem; chain of thought
   costs O(N) tokens and loops when items are confusable (our test: with 64 distractors every arm
   ≤ .38 at K ≥ 2). The question stands: what caps aggregation inside the pass?
3. **What is believed, and what it gets right.** Attention is a normalized mean, so evidence mass
   disperses as Θ(1/n) with competitors; over-squashing in graph networks describes the same
   bottleneck. This predicts a failure that grows with competitor count. Two facts do not fit:
   63 unrelated competitors give the same counting curve as none on Qwen, and the models
   *over*-count when distractors share a feature with the evidence, which dilution cannot produce.
4. **The nugget.** The count travels as a small excess of attention share on true items at the
   question row (2–13% of the weight), spread over many heads and layers; this is causal. Two
   properties of a one-hop softmax read cap it. The score is additive in features, so the read
   matches *Mary* (.8 of the full margin) far more than *kitchen* (.25) and cannot implement the
   conjunction; partial matches leak in proportion. The output is a share, and the model's map from
   share to number is compressive (×4 the share still reads "4"–"6" on Qwen). Both are limits of the
   read, not of the representation: the per-unit facts are decodable, the aggregation is not.
5. **The method that follows.** If the read cannot bind, bind before reading: one carrier per unit
   behind a block-diagonal fence, so the network's full depth computes a per-unit verdict with no
   competitors in any softmax; a per-block position reset removes distance; a sum over verdicts
   replaces the compressive map. Three ingredients, each tied to one measured cause, each with its
   own ablation.
6. **Results and scope.** 1.00 exact at every N from 8 to 1024 on Qwen2.5-3B/7B and Llama-3.1-8B,
   1.00 on needle stories with distractors, .99–1.00 on rendered frames to N = 64 (frozen .20–.40),
   from 7.2k parameters and N ≤ 32 training. Ablations: no fence .30, no reset .23, mean read .09 at
   N = 1024. On HERBench action counting the per-frame verdict AUC is .78 for every architecture and
   the fence hurts: the method converts aggregation error into per-unit error and cannot lower the
   latter.
7. **Contributions.** Write the reviewer's summary for them: "Our key ideas are …" then "Our
   contributions are: (i) a causal localization of the counting channel in frozen LMs; (ii) a
   two-bottleneck diagnosis with pre-registered tests that exclude precision, normalizer, competitor
   count, single heads and running counters; (iii) bind-then-count, a 7.2k-parameter construction
   whose length invariance follows from the architecture, validated across three backbones and two
   modalities with cause-matched ablations; (iv) a scope result on real video."

---

## 5. Related work (≈0.75 page)

By theme, present tense, each paragraph says what the line gets right and what it misses. Never a
list of names.

- **Dispersion and over-squashing.** Softmax mass on a target falls as Θ(1/n) with competitors
  (Hahn 2020; Veličković et al. 2025); entropy grows Ω(log n) (Han et al. 2024); the same bottleneck
  is over-squashing in message passing (Alon & Yahav 2021; Topping et al. 2022; Di Giovanni et al.
  2023) and appears in decoder LMs (Barbero et al. 2024); attention dilution with a mean-margin law
  (Gollapudi et al. 2026; Nakanishi 2025). Right: the mechanism of mass loss. Missed: which quantity
  carries the answer. We measure it causally, find the mean-margin law off by 1–6 nats on frozen
  models, and show that competitor count is not the cap for counting.
- **Counting and cardinality in transformers.** Counting fails without inductive bias (Chang & Bisk
  2024; Barbero et al. 2024; Yehudai et al. 2024); a depth-limited running counter lives in item
  tokens on haystack-free inputs (Hasani et al. 2026a,b); many-needle benchmarks report listing
  recall (RULER; Needle Threading; NeedleChain). Right: the failure and one mechanism. Missed: with a
  haystack the count is read at the question tokens (later items carry ≤ .37 of the signal, earlier
  items none), and two bottlenecks, not one, explain the curves.
- **Chain of thought as serial computation.** Expressivity results (Merrill & Sabharwal; Feng et al.;
  Li et al. 2024) explain why a tape helps. Right: CoT extends what one pass cannot do. Missed: it
  is a serial retrieval plus a counter in digits, O(N) tokens, and it fails when items are confusable.
- **Selection and compression for long context.** SnapKV, GemFilter, InfiniRetri, LongLLMLingua,
  EHPC, and our own select-and-repack: score, keep the top-k, read again. Right: fewer competitors
  restore retrieval. Missed: the read over survivors is still a mean, and a count needs recall 1.0
  per sample; on rendered frames a frozen VLM given only the gold frames scores .65–.80.
- **Independent context encoding and register tokens.** Parallel Context Windows (Ratner et al.
  2023), CEPE (Yen et al. 2024), APE (Yang et al. 2025), landmark/memory/gist/beacon tokens, register
  tokens (Darcet et al. 2024), virtual nodes (Southern et al. 2025). Right: encoding blocks
  independently at shared positions is the fence. Missed: all of them are read by the same additive
  softmax mean, and the mean is the second bottleneck. Our mean-read ablation is exactly this family
  and falls to .09 at N = 1024; the sum read is the difference.
- **Sum versus mean aggregators.** DeepSets, GIN's sum over mean, PNA degree scalers, Set
  Transformer. Right: cardinality needs a sum. Missed: inside a frozen LM the sum must be applied to
  bound per-unit verdicts, not to token states; we show the correspondence is load-bearing because
  each ablation fails as the graph account predicts.

---

## 6. Notation (table in the appendix, symbols used consistently)

| Symbol | Meaning |
|---|---|
| N | number of units (sentences, frames) |
| K, J | matching items, non-matching items (K + J = N) |
| u_i, q | unit i, question |
| c_i | carrier token of unit i |
| h_i^{(L)} | final-layer state of c_i |
| M | attention mask (fence); π | position map (per-block reset) |
| s | attention share on matching items at the read row; g | needle-vs-junk logit margin |
| k_eff | effective fan-in 1 / Σ a² |
| z_i | verdict logit of unit i; ĉ = Σ_i 1[z_i > 0] |
| H0 / HF / HC / HU | needle regimes: clean / same-domain filler / length-controlled / unrelated filler |

---

## 7. Anatomy of the failure (≈3 pages, the analysis half)

Each subsection: claim → instrument → figure → what it means. Text, equation, figure for every
important concept. All tests pre-registered; falsifiers reported as fired.

- **7.1 Instruments.** Needle stories (bAbI-style; PEOPLE × ROOMS × VERBS; K = 1…64; regimes H0/HF/
  HC/HU; minimal pairs K vs. K+1 differing in one same-length sentence), MMRED text (filtered
  grammar, N = 8…1024). Three frozen models. Metrics: exact match, emitted count, share s, k_eff,
  margins g, per-needle d′. Cell sizes 24–50 stories in the analysis grid **[replicate the figure
  cells at n ≥ 100]**.
- **7.2 The failure.** (Fig. 1a, Fig. 2: emitted vs. true per model and regime.) Exact to K = 4 on
  all three, then constant (Qwen2.5-7B .86 → .02 at K = 6, emits 4; Llama exact to 12). With
  distractors: K = 1 → "2"–"4", K-independent ≈ 4 to K = 8, then compressed (K = 32 → 8, K = 64 →
  14). Chain of thought in a paragraph, with the trace anatomy (S1, `outputs/cot_anatomy`): Qwen3-8B
  thinking counts .92–1.00 at K ≤ 16 in clean context and .33 / .42 / .00 with 64 same-domain
  distractors, mostly by looping; .97 of traces write a running tally, a corrupted numeral is followed
  by the next one in .63–.86 of cases and never repaired by a latent counter, yet the final answer
  survives because the tape holds redundant tallies (index and count); the chunk-onset attention on
  the target sentence drops 4.5–6× with distractors (.42–.45 → .07–.10). The serial program is
  per-item retrieval plus a written tally, and it fails where retrieval does.
- **7.3 Where the count lives.** (Fig. 3: patching path + clamp/amplify.) Minimal-pair activation
  patching: the K → K+1 signal rides the flipped sentence's own tokens to mid-depth, hands off to
  the question tokens in a band (Qwen-3B L18–24, Qwen-7B L15–18, Llama L11–15), then to the answer
  row; later items ≤ .37, earlier exactly 0 (no running counter). Clamping the question-row share on
  needles at all layers to the K = 4 reference makes Qwen answer 4–6 at every K and Llama 12;
  amplifying raises the count; a single-layer clamp is inert. Equation: s = Σ_{i∈needles} a_i at the
  read rows; the channel is s, distributed over layers.
- **7.4 Bottleneck 1, selectivity.** (Fig. 4: margins by competitor type; half-match bars.) At the
  band, a true item outscores a same-domain non-match by .5–.8 nats (3–4 nats over unrelated text).
  Half-matches by person keep .77–.85 of the full margin, by room .21–.28: the read matches the
  person, not the conjunction. Leakage predicts over-count within stories (ρ = +.25…+.46). Graph
  reading, one paragraph: a one-hop additive read is 1-WL-like; evidence defined by a relation
  between two features needs a nonlinear per-unit computation.
- **7.5 Bottleneck 2, readout.** (Fig. 5: emitted number vs. imposed share, per model.) Multiplying
  the evidence share by 4 at K = 4 still yields "4"–"6" on Qwen; Llama's map is near-linear (×2 → 8),
  which is why Llama counts to 12 in clean context and over-counts in leaky context. Per added
  needle d′ ≈ .5 (resolvable, not resolved). Sharpening the read trades leakage for compression
  (helps Llama at K = 8 only).
- **7.6 What is not the cause.** (Table 1: hypothesis, pre-registered falsifier, outcome.) fp32 =
  bf16; running counter in items (patching, 7.3); softmax normalizer log Z decodes K worse than the
  residual and adapts upstream; single head or layer (distributed, 7.3); label-free pattern
  statistics (no head is sentence-selective, k_eff 21–33 of 64 for all K); competitor count per se
  (HU ≈ H0 on Qwen); individuation prompts and numbering (no effect). Two lines each.
- **7.7 The distance axis.** (Fig. 6, one panel; details in App. C.) BABILong factorial
  full / same-tokens-at-original-positions / repacked: up to 32k, removing competitors recovers most
  of the loss (Llama qa1 32k .735 → .86 → .875); beyond it positions dominate (128k .29 → .40 → .85)
  and the 1M-context model removes the distance term while dispersion alone still costs 5–7pp. Two
  axes → two ingredients (fence, reset).

---

## 8. Bind, then count (≈1.25 pages, the method half)

- **8.1 From diagnosis to design.** Three sentences: selectivity needs a nonlinear per-unit
  computation with no competitors in the softmax → fence; distance degrades the per-unit verdict →
  per-block position reset; the compressive share-to-number map → sum over bounded verdicts.
- **8.2 Construction.** (Fig. 7: layout, mask, read.) Layout: prefix (system + question), N blocks
  each "\n" + u_i + c_i, tail "\nQuestion: q\nAnswer:". Mask M: causal within the prefix and
  within each block; cross-block cells closed at every layer; tail rows see prefix and carriers.
  Positions π: each block restarts at prefix_end; the tail follows the longest block. Carrier
  embedding e_c is trained; verdict z_i = wᵀh_i^{(L)} + b; count ĉ = Σ_i 1[z_i > 0]; exists = max,
  first/last = extreme index with z_i > 0; distinct/mode from a per-unit class head. Loss:
  BCE(z_i, y_i) + λ (Σ_i σ(z_i) − y)² / N. Parameters: e_c and (w, b), i.e. 2d + 1 (7.2k on the 7B)
  **[state per backbone: 3B and Llama differ in d]**. Trained 3–4 epochs on N ≤ 32 (360 stories per
  N). ⟨TF⟩ Training-free form (`judge_fenced.py`, `judge_fenced_vlm.py`): block_i ends with the
  per-unit question and " Answer:"; z_i = logit(Yes) − logit(No) at that token; no e_c, no head; the
  trained head is the calibrated version of the same read (AUC ≈ 1.00 in both, threshold differs). Equations must match `outputs/_scratch/dbg/train_bindcount.py` **[move to scripts/ before
  release]**.
- **8.3 Why it generalizes.** One proposition, two lines: z_i depends only on (prefix, u_i, q), so
  its distribution is independent of N; exact-count accuracy is the probability that all N verdicts
  are correct. Consequence stated as a design rule: per-unit error must be ≪ 1/N, which the
  count-aware loss delivers for the verdict head and which the 7-way room head does not (Sec. 9.5).
- **8.4 Cost.** One forward; block-diagonal attention costs no more than full attention; the mask is
  the only added machinery. Vision: carrier per frame via the VL model's box token, M-RoPE reset per
  frame; no other change.

---

## 9. Experiments (≈2.25 pages)

Order: simplest baseline first; change one thing at a time; explain what each number means.

- **9.1 Setup.** Data: MMRED text (filtered grammar), needle stories, MMRED rendered frames, HERBench
  action counting (HD-EPIC egocentric video, N = 16/32 frames). Backbones: Qwen2.5-3B, Qwen2.5-7B,
  Llama-3.1-8B, Qwen2.5-VL-7B (4-bit, frozen). Metrics: exact match; per-unit verdict accuracy and
  AUC; MAE. Training N ≤ 32 everywhere; n = 100 per test cell **[seeds: record and report]**.
- **9.2 Baselines.** Frozen model (Qwen2.5-7B text .59 → .15 from N = 8 to 1024); frozen model
  given only the gold units (vision .65–.80: it cannot count even what it is shown); digit-LoRA at
  the answer row (.92 in-length → .25); chain of thought (Sec. 7.2); select-and-repack, our own
  two-pass method **[re-run on the same backbone and data for a fair row; August numbers are on a
  different split]**.
- **9.3 Main result.** (Table 2, Fig. 1b.) ⟨TF⟩ Training-free: MMRED text 1.00 at every N from 8 to
  1024 (per-unit 1.000), rendered frames .98 / .97 / .99 / .98 (per-frame ≥ .996), needles with
  distractors .89 / .92 / .91 (per-unit .995–.998, AUC ≈ 1.00). Calibrated (2d + 1 head): 1.00 at every
  N on all three text backbones, 1.00 on needles at N = 64/128/256 with K ≤ 64, .99–1.00 on frames.
  Narrative: the number is not a fit to a range, it is the proposition of 8.3 holding; the head buys
  the threshold, not the ranking.
- **9.3b Training-free ablations.** (Table 3a.) No fence (same cue, causal, native positions): text
  .65 / .14 / .18 / .01 / .00 … with AUC falling .99 → .88, needles AUC .90 → .71, vision .33 / .41 /
  .31 / .31 → isolation. No cue (question only in the prefix): text .49 / .16 / .07 / .22 / .41 / .66 /
  .66 / .80 with AUC ≥ .999 → calibration. Vague cue ("relevant to the question?"): MMRED ≈ direct,
  needles .19 / .09 / .10 with over-counting, vision .62–.74 → selectivity on partial matches.
- **9.4 Ablations, one ingredient at a time.** (Table 3.) No fence: 1.00 → .81/.62/.56/.44/.30 at
  N = 64…1024 with per-unit accuracy .997 (one wrong verdict in N kills the count) → selectivity.
  No reset: 1.00 to N = 64, then .98/.63/.32/.23; verdict accuracy 1.00 → .984; worst-case margin
  14 → 5.8 → distance. Mean read (fenced carriers, digit LoRA at the answer row): 1.00/.98/.94
  in-length → .60/.26/.10/.09/.09 → readout. Each ablation is the cause-matched control, and the
  mean-read arm is also the PCW/APE-style control from related work.
- **9.5 Set functions from one forward.** (Table 4.) count / exists / first / last 1.00 at every N;
  distinct rooms 1.00 → .71, most-visited room 1.00 → .82 at N = 1024 with a .998-accurate room
  head: (1 − ε)^N. Explain the number: this is the design rule of 8.3, not a failure of the idea.
- **9.6 Vision.** (Table 5.) Rendered frames, N = 8/16/32/64: .99/.99/1.00/.99, per-frame verdict
  .999; no fence .60/.33/.33/.27 (verdict .91–.96); frozen .20–.40; oracle frames .65–.80. Frames
  interfere more than sentences.
- **9.7 Scope: real video.** (Table 6.) HERBench action counting: per-frame verdict AUC .77–.80 for
  fenced, unfenced and frozen; exact counts .04–.15 for every arm; frozen full .12/.05; oracle .02.
  The fence hurts (no-fence verdict .75/.82 vs. fenced .68/.69): neighbouring frames help a frozen
  VLM judge one frame when the action recurs. Reported as the limit of the method: it converts
  aggregation error into per-unit error and cannot lower the backbone's perception error.
- **9.8 Mechanistic check on the carriers.** (Fig. 8.) Verdict-margin distribution vs. N is flat
  under the full construction (median worst-case margin 19–25 at every N) and decays without the
  reset (14 → 5.8): the same instrument that showed the frozen read has no peaks shows the carriers
  hold theirs. Numbers exist in the run dirs **[make the figure]**.

---

## 10. Limitations and future work (≈0.35 page)

Open, and each framed with its path forward.

- Units must be pre-segmented (sentences, frames). Path: learned segmentation or overlapping windows.
- The verdict head is trained (2d + 1 parameters); fully training-free variants were negative.
- The verdict is a per-unit predicate. Relations between two units (were A and B ever together
  across frames) need a second stage over carriers; exists/first/last already fall out.
- Isolation costs context: on real video the fence hurts because neighbouring frames inform a
  frame. Path: fence with a local window; learned masks did not help length generalization in our
  earlier study (App. H), so the window should be fixed, not learned.
- Per-unit error must be ≪ 1/N; multi-class heads need margin-aware losses.
- Synthetic aggregation data; real-data gains are shown only where per-unit judgment is reliable.

## 11. Conclusion (≈0.15 page)

Present tense. Restate the nugget with the numbers: a one-hop softmax read into one node cannot bind
and cannot count; binding each unit behind a fence and summing bounded verdicts makes the
aggregation exact at any length; the remaining error is the model's per-unit judgment.

---

## 12. Figures and tables

| # | Content | What the caption tells the reader to notice |
|---|---|---|
| Fig. 1 | teaser (a) emitted vs. true count, (b) EM vs. N with ablations | constant plateau at 4–12; one flat line at 1.00; ablations fall in predicted order |
| Fig. 2 | emitted vs. true per model × regime | over-count above the diagonal only with same-domain distractors |
| Fig. 3 | patching path heatmap (token group × layer) + clamp/amplify curves | hand-off band; clamp pins the answer, amplify raises it |
| Fig. 4 | margins by competitor type; half-match bars (person vs. room) | .5–.8 vs. 3–4 nats; person ≫ room |
| Fig. 5 | emitted number vs. imposed share, Qwen vs. Llama | compressive vs. near-linear map |
| Fig. 6 | BABILong factorial vs. length (one panel) | competitor axis ≤ 32k, distance axis beyond |
| Fig. 7 | method layout: blocks, carriers, fence mask, position reset, sum read | which cells are closed; where the verdict is read |
| Fig. 8 | verdict margin vs. N, full vs. no-reset | flat vs. decaying |
| Table 1 | excluded hypotheses: falsifier → outcome | each row one line |
| Table 2 | main result: EM vs. N, three backbones, MMRED + needles | all 1.00 |
| Table 3 | ablations vs. N with per-unit accuracy and worst-case margin | which ingredient buys what |
| Table 4 | set functions vs. N | exact ones vs. (1 − ε)^N ones |
| Table 5 | vision: method / no fence / frozen / oracle vs. N | frames interfere more than sentences |
| Table 6 | HERBench: exact, verdict acc, AUC per arm | perception ceiling; fence hurts |

Every figure referenced in the text; text readable without zoom; colour for information, not
decoration; `[t]` placement.

---

## 13. Appendices

- A. Pre-registration documents with predictions and dated outcomes (verbatim from `PREREG_AGG.md`,
  `PREREG_OSQ.md`), including fired falsifiers and post-hoc corrections labelled as such.
- B. Needle instrument: grammar, regimes, minimal pairs, patching and clamp procedures.
- C. Retrieval program: BABILong grids, LSE decomposition (three regimes), 1M-context test, k-identity.
- D. Softmax-normalizer probe (falsified). E. Attention-pattern statistics (falsified).
- F. Chain-of-thought test. G. Training-free selection on MMRED (text and vision; negative).
- H. Learned attention masks (negative; length-mixture with full attention dominates).
- I. What fine-tuning buys: a fixed margin, one octave of length; required margin grows as log N.
- J. HERBench preparation (frame sampling, splits by video). K. Compute and cost.
- L. AI-use statement (required). Reproducibility statement (recommended): run dirs, seeds, code.

---

## 14. Page budget (9 pages)

| Section | Pages |
|---|---|
| Introduction + Fig. 1 | 1.25 |
| Related work | 0.75 |
| Anatomy of the failure | 3.00 |
| Bind, then count | 1.25 |
| Experiments | 2.25 |
| Limitations + conclusion | 0.50 |

Fill the nine pages; a short paper reads as unfinished.

---

## 15. House style

- Present tense throughout, including related work ("Barbero et al. show"). Past tense only for how
  data were generated.
- Active voice; the method does things, "we" only for choices we made. No "allows to", "enables",
  "provides", "showcases", "delve", "novel", "aims to", "may". No contractions, no exclamation marks.
- Every comparative names its comparison ("more accurate than the frozen model at N = 1024").
- No over-claiming: the method is exact where the per-unit verdict is reliable; say where it is not.
- Every important concept three ways: one sentence of gist, one equation, one figure.
- Equations match the code; notation table kept from day one; no overloaded symbols.
- No sentence starts with a citation; citation numbers sorted; cite the published version.
- Captions say what to notice. "Figure 3 shows", not "In Figure 3 we show".
- Fewer words. No "in this section we". No paragraph ending with one or two words on its last line.

---

## 16. Evidence gaps to close before the text is final

1. Replicate the analysis cells that carry figures 2–5 at n ≥ 100 (current 24–50; differences under
   .15 are noise).
2. Seeds for every method cell; report mean over 3 seeds where cheap (text arms are minutes).
3. Select-and-repack baseline row on the same backbone and split as Table 2.
4. Parameter count per backbone (2d + 1; 7.2k holds for d = 3584 only).
5. Fig. 8 from existing run dirs (verdict margins vs. N).
6. Move `train_bindcount.py` / `train_bindcount_vlm.py` from `outputs/_scratch/dbg/` into
   `scripts/` with a CPU test, so equations and code can be checked side by side.
7. Compute table (GPU-hours per arm).
8. Optional: rendered-frame N > 64; HERBench 5-way mapping and the training-free judge on HERBench;
   a second real benchmark where the per-unit verdict is reliable (the main reviewer gap).
10. ⟨TF⟩ Decide the headline: training-free exactness (text to 1024, frames to 64) with the 2d + 1
    head as calibration for hard families, or the trained construction with the training-free result
    as an ablation. Recommendation: training-free headline; it is the stronger and simpler claim.
11. In-model tally (S2) stays in the appendix as a boundary: a patched answer row is read as one
    token (single digits 1.00 at layer 24, two digits ≤ .26); helix + linear basis explains R² ≈ .7 of
    the number vectors.
9. Draft in this order: Fig. 1 and Table 2 → abstract → introduction → method → experiments →
   analysis → related work → limitations. First full draft by 14 Sept, abstract registered 18 Sept.

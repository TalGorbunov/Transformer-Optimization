# Fenced-Carrier Scratchpad Readout — method statement & adoption guide

> A recipe for making a **frozen** decoder VLM/LLM aggregate information across many
> context units (frames, documents, turns) that it otherwise over-squashes. Every default
> below is the winner of a logged ablation in `RESULTS.md`; run dirs are cited there.
> Status 2026-07-24: validated on Qwen2.5-VL-7B (4-bit) / synthetic MMRED; supply-level
> ported to InternVL2.5-8B; natural-image and task-transfer cells in flight.

## 1. When this method applies

- The task decomposes as **per-unit extraction + reduction**: each context unit carries a
  small fact (a predicate verdict, an attribute), and the answer is a reduction over
  units (count, distinct-set, co-occurrence, retrieval, union).
- The backbone is **frozen** but you control inference: custom attention masks + position
  IDs (nnsight/hooks). No mask control → fall back to per-unit forwards (multipass); the
  supply result still holds (InternVL).
- **Gold per-unit labels** exist or are generatable for a training set (they become the
  scratchpad targets). ~5k samples pooled over tasks is the measured comfortable regime
  (450 → 0.68, 5.1k → 0.98, 6k → 0.999).
- Deployment lengths are covered by training lengths (extrapolation holds only ~1.3–2×;
  beyond that is an open research problem — see §5).

## 2. The recipe (defaults = ablation winners)

1. **Layout:** `[question] [unit_1][carrier_1] … [unit_N][carrier_N]` — or all carriers
   as a suffix block. **The question must lead** (−46% without; question-conditioned
   encoding is the single most load-bearing piece). Carrier placement itself is free.
2. **Fence (graph rewiring):** block-diagonal attention mask on layers `0..L*-1`; each
   unit+carrier is an island that also sees the question; per-block position reset.
   Removes the joint-context interference (cross-frame noise ρ 0.09–0.13 → ~0.01).
3. **Carrier (virtual node):** ONE learned embedding (~3.5k params), one copy per unit.
   Train by label-free distillation against a question-replica teacher (96% of teacher
   d′), then freeze. Random init converges to the same endpoint.
4. **Open phase + reader:** from layer `L*` up, the fence lifts (carriers see each other,
   generation sees carriers) and a **LoRA r8 (~2M params)** is trained there. `L* = 12`
   (inverted-U ablation, peak 0.443 vs 0.280 @L17 zero-shot N=32). Rank is flat 4–16.
5. **Readout target — caption full-scan running-tally scratchpad** (format-sweep winner):
   `scan: f1:- f2:kitchen(1) f3:- … | total: 2 END`
   One slot per unit in fixed order (no search burden), attribute words not bare
   verdicts (costs nothing, buys task-generality), running tally (never count a long
   list), explicit anchor + terminator. Plain teacher-forced CE on this text; converges
   in 1–5 epochs. Save criterion: (TF-count acc, tf-exact) lexicographic — acc-only
   ckpt selection cost up to 0.35 at length in a logged confound.
6. **Include every deployment length in the training mix.** In-length is the working
   regime (0.98–0.99 through N=64); zero-shot length decays smoothly (2× ≈ 0.28).
7. **Eval discipline:** greedy decode; decode budget ≥ ~6 tokens/unit (cap truncation is
   otherwise misread as model failure); parse on the anchor; report parse-fail and
   per-count histograms (failure anatomy is diagnostic: undercounts = detection recall;
   constant answer = readout collapse; extremes-only = supply ceiling heuristic).

**Measured properties on the reference setup:** 0.999 @N≤8 (3 tasks incl. nonlinear
set-union), 0.987/0.972/0.981 @N=32/48/64 held-out; one carrier+LoRA serves 5 question
types (0.966); seed-stable (TF 1.000×3); **no-harm**: MME −0.2 / POPE −1.4 pts with the
adapter always-on, and plain-prompt drift is mildly positive (0.313 vs 0.219) — the
LoRA's behavior is gated on the carrier context, which normal prompts never contain.

## 3. The diagnostic ladder (run this BEFORE training, on any new model/task/domain)

Each rung is cheaper than the next and localizes the failure if a later rung is low.

| Rung | Question it answers | Cost | Reference bands |
|---|---|---|---|
| **L0** frozen baseline | is there a problem? | eval-only | park: 0.219 @N=8, →0.013 @128 |
| **L1** supply probe: d′ of fenced-carrier vs joint messages | does the intervention repair supply here? | cache job + CPU | fenced ≥4 and ≥2× joint (park: 9–13 vs 2) |
| **L2** scaffold: logistic gate→tally on the cached messages | do the node states solve the task externally? | CPU | ≥0.9 (park 0.998; InternVL 0.938) |
| **L3** in-model: recipe §2 trained, in-length | does the model itself say it? | 1 trainer + evals | ≥0.8 of L2 |

Reading the ladder: L1 low → the encoder can't extract the per-unit fact in this domain
(MLVU/natural-images failure mode — a perception/domain problem, not architecture);
L1 high + L2 low → aggregation/label structure problem; L2 high + L3 low → readout
problem (data, format, or L* — consult the ablation table). This ladder, not the recipe,
is the part that transfers to ANY over-squashing suspicion in a frozen model.

## 3b. Porting checklist — your model, your task

What you bring, what you reuse, what you re-derive:

1. **Your task → the schema.** Write your task as *per-unit fact + textual reduction*.
   Define the unit (frame / document / table row / dialogue turn), the per-unit fact
   (a word or short phrase — not a vector), and the reduction the LM will do in text
   (count / distinct / lookup / compare). If you cannot phrase the per-unit fact as
   tokens, stop — the method's readout is the token interface.
2. **Your model → three capabilities to verify** (an afternoon): (a) can you inject a
   custom attention mask at inference (hooks/nnsight)? If not, you can still run rungs
   L0–L2 with per-unit forwards. (b) can you reset/control position IDs per block?
   (c) does a leading question measurably condition the unit encoding? — this is
   family-dependent (strong on Qwen, absent on InternVL), so measure it at rung L1
   with and without the question leading; if absent, expect to rely on the fence alone
   and validate L2 before proceeding.
3. **Labels:** you need gold per-unit facts for a training set (thousands, not
   hundreds — our data curve: 450 → 0.68, 5–6k → 0.98+). Synthetic generation with
   known ground truth is the cheap route; a judge/pseudo-labeler is untested here.
4. **Run the ladder (§3) before any training.** L0–L2 cost no reader training and tell
   you whether your model over-squashes your task at all (L0 vs L2 gap), and whether
   supply repair works in your domain (L1). Only then pay for L3.
5. **Re-derive two knobs on your setup**, reuse the rest: the open layer `L*` (small
   sweep, 3–4 arms — our inverted-U peaked at 12/28 layers ≈ 40% depth) and the decode
   budget (≥ ~6 tokens × your N). Format (caption scan + tally), rank (8), save
   criterion, and eval discipline transfer as-is.
6. **Close with the two safety cells:** a no-harm benchmark pair with the adapter on
   (our band: ≤2 pts), and a plain-prompt drift check. The gating argument (LoRA
   behavior triggered only by the carrier context) is generic, but measure it.

All numbers in this file are reference bands from our setup (Qwen2.5-VL-7B, synthetic
MMRED) — treat them as what "working" looks like, not as guarantees.

## 4. Costs

Reference hardware: one 40–48 GB GPU, 4-bit 7B backbone. Carrier distill ~hours; cached
trainer ~14 h at the full 8.8k-sample mixture (grad-ckpt, bf16 cache of the fenced
prefix makes steps ~3–4× cheaper); evals minutes/sample at long N (decode-bound; scan
formats ≈ 2.4× poslist decode length). Trained parameters total ≈ 2M (LoRA) + 3.5k
(carrier) on a frozen 7B.

## 5. Known limits (all measured — cite, don't hide)

- **Length extrapolation:** ~1.3–2× beyond training lengths, then tally-index confusion
  / repetition (8×: 0.087). Train at deployment length instead; this is a field-wide
  open problem, not method-specific.
- **Domain-bound evidence detection:** carrier trained on one visual domain does not
  fire on another (MLVU zero-shot 0.107 ≤ frozen 0.282; cross-domain carrier ~51% of
  teacher). The format transfers; the detector must be trained in-domain (ladder rung
  L1 predicts this before you pay for L3).
- **Question-first is family-dependent:** the fence/isolation piece ports (InternVL
  3.5×); the Q-first amplifier is Qwen-specific (flat-to-negative on InternVL).
- **Task coverage must be trained:** zero-shot transfer to an unseen question type is
  ≈ chance for single-task checkpoints; mixture training is cheap (no interference) and
  is the current requirement (mixture-minus-one transfer under test).
- **Exact-match at length is bounded by per-unit detection recall** (errors are missing
  verdicts, compounding ≈ (1−p)^N; the counting itself is exact once verdicts are
  written).

## 6. Why it works (one paragraph, GNN framing)

Joint attention over N units is a complete graph whose evidence competes through a
softmax bottleneck — over-squashing in the Alon–Yahav sense; measured as a d′≈2 supply
ceiling that caps ANY readout of the joint states (best linear: 0.19 @N=32). The fence
is graph rewiring (cut the interfering edges), the carrier is a per-unit virtual node —
after them, N clean node embeddings exist and one sum-aggregator pass solves the task
externally (GIN/DeepSets-equivalent; 0.998). The remaining failure is the frozen
readout: activation-space codes are memorized symbols to a frozen decoder (in-range
only, zero value-extrapolation), while its token interface decodes systematically. The
scratchpad routes the answer through that interface — the aggregation happens serially
in the model's own generated text, the one place a frozen LM can always compute.

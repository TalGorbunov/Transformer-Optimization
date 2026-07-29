# RESULTS.md — Research progress log

> **Purpose:** the single overall view of the thesis that no individual output dir gives — what we've
> found, what works, what's open. **Read top-to-bottom:** the narrative (At a glance → Synthesis →
> Method comparison) comes first; the detailed run-by-run **Experiment Log** is the appendix below it,
> and maintenance/provenance notes are at the very end.
>
> **Status flags used throughout:** ✅ done & trusted · ⚠️ done but partial/suspect · ❌ failed/no-gain ·
> 📊 characterization/probe only · ▶ running.
---
## Executive summary

We study **why decoder only vision-language models fail at simple visual *aggregation* tasks**,
and how to fix it with a small trainable adapter — framed through the graph-neural-network lens of
**over-squashing** (attention ≈ message passing).

- **Diagnosis.** The model fails because attention aggregates the per-frame evidence as a *normalized
  mean*, which provably cannot represent "how many" (a mean is count-blind). Baseline accuracy collapses
  from ~85% at 1–2 frames to ~20–30% at 8 frames, even though the per-frame evidence is present and
  decodable — it's an aggregation failure, not a perception failure. **And it's not counting-specific
 :** two *other* MMReD tasks — rooms-visited (set-cardinality) and co-occupancy (a 2-character
  predicate) — show the same seq_len collapse, with the output regressing toward a length-dependent
  *middle* estimate (undershooting high-answer tasks, overshooting low-answer ones). So the bottleneck is
  a general aggregation/representational-collapse failure, invariant to both the aggregation type and the
  evidence predicate.
- **The fix.** Replacing that implicit mean with an explicit **unnormalized sum** — a *DeepSets* readout
  (`ρ(Σ φ(messageᵢ))`) injected into the frozen residual stream — **solves the pure-counting task: 100%,
  including perfect extrapolation to unseen counts and lengths.** It's a one-layer adapter; the heavy
  gLSTM memory we started with turns out to be unnecessary. We causally isolate the two ingredients that
  matter: (1) the aggregator must be a **sum**, not a mean/softmax (normalizing costs ~24 pts on
  extrapolation, n=3 seeds); and (2) its width must be **≥ the maximum count** (a measured capacity knee
  exactly at width = 8 = max count).
- **Open frontier.** When irrelevant **distractor** frames are added, the problem stops being aggregation
  and becomes **selection**. An oracle that knows which frames are evidence reaches **96%**, but every
  *learned* selector plateaus at **~63–68%** (memory adapters with a LoRA; ~47% for the bare gated posneg
  adapter). We now know this plateau is **robust to the selection mechanism**: per-frame gating, count-level
  supervision, *and* competitive slot-attention routing all land in it; only richer *aggregation*
  (PNA) nudges it (+4pp). And per-frame gating is *structurally* capped — errors compound with count, and
  the per-frame evidence-detection ceiling is ~0.96, short of the ~0.99 exact counting needs. So
  **the distractor gap is not closeable by better selection** — the remaining levers are richer aggregation
  and/or count-level (distributed) supervision, but neither has crossed ~68% yet. This is the open problem.

**One-sentence takeaway:** the visual-counting bottleneck is a normalized, fixed-width *mean*; an
unnormalized sum of width ≥ max-count fixes the aggregation half outright, and the remaining gap is a
distractor-*selection* problem that per-frame methods provably can't close.

---
## Glossary

| Term | Meaning |
|------|---------|
| **MMRED** | The task/dataset: short sequences of *frames* (images) of characters in rooms. Question: *"how many frames was character C in room R?"*, answer ∈ {0..8}. Headline metric = exact-match accuracy of the predicted count. |
| **seq_len / frames** | Number of images in one sample (1–8). |
| **evidence frame** | A frame where the queried character **is** in the queried room — counts toward the answer. |
| **neutral frame** | Non-evidence, *easy*: queried character absent **and** queried room empty. |
| **distractor frame** | Non-evidence, *hard*: queried character is elsewhere, or the queried room is occupied by **other** characters. The distractor task is the open frontier. |
| **carrier tokens** | The question/answer text tokens that mechanistically absorb each frame's evidence and relay it to the last token (the **frame→carrier→last** information pathway). |
| **adapter** | A small trainable module added to the **frozen** Qwen residual stream (Qwen stays frozen + 4-bit). All "fixes" here are adapters, not fine-tuning. |
| **DeepSets / sum readout** | The proposed fix: encode each frame's message (φ), **sum** over frames (unnormalized), decode (ρ), inject. The sum is the count-preserving aggregator. |
| **gLSTM** | The associative-memory adapter this project started from; our controls show its memory-addressing machinery is **dispensable** — a plain sum matches it. |
| **IID / length-OOD / count-OOD** | Test splits: in-distribution; sequences **longer** than trained on; counts **higher** than trained on. |
| **oracle** | A diagnostic **upper bound** that uses gold (ground-truth) labels — not a deployable method, used to bound what's achievable. |
| **AUC** | Ranking quality of a per-frame evidence detector (1.0 perfect, 0.5 chance); see §Synthesis for why ~0.96 still isn't enough for exact counting. |

---
## At a glance

- **Task / headline metric:** MMRED — *"how many frames was character C in room R?"*, answer ∈ {0..8}.
  Metric = **exact-match accuracy** of the predicted count (argmax over candidate count tokens),
  reported **overall** and **stratified by `evidence_count`** (the gold count). Secondary signals:
  `gold_margin` (gold-token logit/logprob minus best competitor), predicted-count **MAE**, and
  `fix_rate`/`break_rate` of an intervention vs the frozen base.
- **The bottleneck (well-established):** baseline Qwen2.5-VL accuracy collapses as the number of
  frames/evidence grows — ~85–100% at seq_len 1–2 down to ~19–30% at seq_len 8. The collapse is an
  **aggregation/over-squashing** failure: per-frame evidence is present and decodable, but the model
  cannot combine many frames in a single forward pass. Last-token representations and the gold margin
  collapse sharply once **2–4 frames** must be aggregated.
- **Current leading approach:** a frozen-Qwen **DeepSets readout** — `ρ(Σ φ(message_i))`:
  project each frame's attention "message", **unnormalized sum** over frames, inject at carrier/last-token
  residuals (~L14–17). gLSTM's associative memory-addressing is **dispensable** (sum ties/beats it on
  matched controls); the load-bearing ingredients are **unnormalized aggregation + width ≥ max-count**.
- **Status:**
  - ✅ **Evidence-only counting is solved** — sum / layer-local / raw-matrix / PNA adapters hit **100%**
    at seq_len 1–8 (vs ~39% base), incl. **100% count/length OOD** (train 1–4 → eval 5–8). Minimal form:
    a single layer, one shared φ/ρ, one inject at last token.
  - ✅ **Two causal knobs isolated:** unnormalized sum vs mean/softmax = +24pp OOD
    (normalization is the failure); d_mem sweep saturates at **width = max-count N** (capacity bound).
  - ❌ **Distractors are the open frontier** — learned adapters plateau ~40–64%; only oracle-masked
    upper bounds (96%) recover. The gap is **selection** (per-frame evidence detection), not aggregation;
    the per-frame gating interface is falsified, pointing to count-level stream supervision.

---

---
## Weekly progress

### Earlier work (Feb 2026 → early June)  ·  pre-weekly-cadence summary

Locating and diagnosing the bottleneck (full detail in the Experiment Log, Feb–May rows): baseline Qwen2.5-VL
collapses from ~85% (1–2 frames) to ~20–30% (8 frames). Probes show the per-frame evidence is **decodable**
(count probe ~95% seq2 / ~40% seq8 while the model gets ~30%) and causal ablations show individual-frame
influence vanishes by seq8 → an **aggregation / over-squashing** failure, not perception. Last-token
representational collapse (cosine 0.060→0.019, seq1→8) matches [Barbero et al. 2024](https://arxiv.org/abs/2406.04267).
Evidence-only counting was first **solved** in this period by a sum/DeepSets adapter (~100%).

### Week 2026.06.02-09

**The story.** Pinned the *counting* fix down to two causally-isolated knobs; the minimal adapter is tiny and
the heavy gLSTM memory turns out to be unnecessary.

**Results.**
- **Counting is solved and the fix is a 2-knob DeepSets sum.** Matched-harness ablations (n=3 seeds on the two
  headline curves): (1) **unnormalized sum vs mean/softmax = +24pp on count/length-OOD** (sum/PNA 1.00 vs
  mean/softmax 0.76) — normalization is the causal failure; (2) **capacity knee exactly at width = max-count**
  (d_mem sweep saturates at 8 = N). Minimal deployable form: one mid-layer, one shared φ/ρ, inject once at the
  last token → **100% incl. count/length OOD** (train 1–4 → eval 5–8). gLSTM's associative addressing is
  **dispensable** (plain sum ties/beats it); PNA = sum (safe default).
- **Aggregator-ingredient decomposition:** of the three candidate ingredients only **unnormalized aggregation**
  and **width ≥ max-count** are load-bearing; the associative q·k addressing adds nothing over a plain sum.

**Insight.** The counting bottleneck reduces to "attention is a normalized, bounded-width *mean*"; an
unnormalized **sum** of width ≥ max-count fixes it as a single-layer adapter.

### Week 2026.06.10-16

We (a) showed the bottleneck **generalizes** to two new aggregation tasks, and (b) in
a focused sprint traced *where* those new tasks fail — correcting our own first guess: the failure is
**aggregation, not vision extraction**, and the **frame→carrier→last "stage" pathway is a general,
task-agnostic property** of the model.

**Results.**
- **The bottleneck generalizes beyond counting.** Frozen-Qwen baselines on two *new* MMReD tasks —
  **rooms-visited** (set-cardinality) and **co-occupancy** (2-character predicate) — show the same seq_len
  collapse, regressing toward a length-dependent *middle* estimate (undershoot high-answer tasks, overshoot
  low-answer ones). So the over-squashing failure is invariant to aggregation *type* and *predicate*.
- **For the new tasks the wall is AGGREGATION, not vision** (this *revised* our mid-week "vision-extraction"
  guess). Message probes that recompute each frame's attention-routed message to the carrier show the per-frame
  evidence **reaches the carrier with high fidelity** (rooms-visited room 0.85; co-occupancy same/diff **AUROC
  0.97**). Yet **decode-then-count** (classify each frame, then count — no learned count fn) far beats the model:
  rooms-visited **0.63 vs model 0.10**, co-occupancy **0.52 vs 0.27**. Evidence is extracted+routed; the model
  fails to *combine* it. Attention-LoRA (`global_lora`, ~0.58) ≈ the decode-then-count ceiling → it helps by
  improving **re-attention/aggregation**. **Vision-encoder LoRA does not help** (ViT-only 0.50, ViT+LM 0.596 ≈
  LM-only 0.58) — confirming the lever is LM aggregation, not the vision tower.
- **The frame→carrier→last STAGE pathway is task-general.** Layerwise token-group restoration (n=40, same
  method for all three tasks) gives the identical staircase: frames rescue **early** (L0–12) → question/carrier
  **mid** (peak L14–16) → last token **late** (L18–26), crossover ~L14–16. Invariant across three predicates →
  a **general, training-induced routing scaffold**, not a counting artifact.
  (Plot: `outputs/stages_7b_plots/stages_restoration_by_group_7b_n40.png`.)
- **Distractor selection still plateaus.** New selection mechanisms (count-level supervision, competitive
  slot-attention routing) all land in the ~63–68% band; only richer *aggregation* (PNA) nudges it (+4pp).

**Insights.**
- The staged frames→carrier→last flow is **how this model routes evidence in general** (reproduces on 3 tasks)
  — likely a training-induced computational scaffold worth treating as a fixed substrate for interventions.
- For set-cardinality tasks the evidence is present and routed; the model just doesn't aggregate it — the same
  diagnosis as counting, now shown one task-family wider.
- Methodological self-correction: the symbolic text-oracle and pooled linear probes *over-stated* an
  "extraction" story; the message-level + decode-then-count probes are the trustworthy test.

**Caveat to carry.** rooms-visited's symbolic oracle is only **0.76** (vs co-occ 0.98) → its dedup-distinct-count
*computation* is itself partly lossy, stacking an internal-computation limit on top of the aggregation gap.

**In progress (this week).**
- **DeepSets aggregator sweep** (count-only, running now): does an explicit operator recover the new-task
  count? Sum for counting/co-occupancy (additive); a **union/dedup** operator (noisy-OR per room → count) for
  rooms-visited, since plain sum can't dedup distinct rooms. Tests operator-must-match-algebra; if union works,
  next is a width / injection-layer sweep.
- Distractor selection remains the standing open problem (richer aggregation or count-level supervision).

### [2026-06-16] The bottleneck is a bandwidth-limited, saturating aggregation cut — not the operator or the aggregator

**Diagnosis (task-general).** All three MMReD tasks require aggregating N frames' evidence into the
fixed-capacity residual stream of a few carrier/last tokens — a thin bottleneck cut that *saturates*.
Evidence: last-token cosine collapse 0.060→0.019 (seq1→8); hidden-state norm 0.347→0.181; and the
nested-evidence margin collapses at **k=2–4 frames even when the added frames are true evidence**
(+6.66 → +0.15 → −1.03 → −2.23 for k=1,2,4,8). The channel is full by ~2 frames, so *more* information
*hurts* — the clean over-squashing/capacity signature ([Di Giovanni 2023](https://arxiv.org/abs/2302.02941)
width; [Alon & Yahav 2021](https://arxiv.org/abs/2006.05205) bottleneck; [Barbero 2024](https://arxiv.org/abs/2406.04267)
last-token collapse).

**What this rules out.** (a) *Operator swaps are dead* — maxmix, PNA-mix, native-attention factorization,
and oracle routing all reweight the **same saturated channel** and plateau/fail (maxmix distractor ~47%,
count-OOD 0%; native-attention factorization ~0% smoke; oracle routing ±2%). (b) *A better single-channel
aggregator is not the fix* — unnormalized **sum solves counting (100%, incl. OOD)** because counting is a
scalar (sum is a sufficient statistic), but sum/PNA only reach **~48% on rooms-visited** (sum 48.2%, PNA
46.5%) and **~44% on co-occupancy** (baseline ~10%). Those are *set functions* (distinct-count /
predicate-count) that need the carrier to hold the *set* of rooms/pairs, which a saturating scalar-ish
channel cannot.


### [2026-06-17] Week recap (06-12 → 06-17)


**Aggregation operator + data sweeps (`agg_sweep` / `agg_moredata` / `agg_ood`).**
- On rooms-visited, single-channel operators cluster **~46% iid** (carrier-sum 45.7%, slot/union/gLSTM/max within noise) — operator choice ≈ noise.
- Doubling data lifts carrier-sum to **58.1% iid** → capacity/data, not the operator, moves the needle.

**Read-mechanism / softmax ablation (evidence-only, neutral).**
- len-OOD: gLSTM **0.778** / sum **0.762** / softmax **0.680** → normalization costs ~−10pp; associative addressing adds nothing over plain sum (= GIN sum>mean inside the frozen VLM).

**Counting task status.**
- Evidence-only sum adapter = **100%** (baseline 23%).
- Distractor counting: baseline **24.4%**; best learned **PNA 68.1%** (> sum 63.8 / gLSTM 61.7); oracle-posneg ceiling **91.1%** (this run) → the residual gap is *selection*, not aggregation capacity.

**Two new tasks (set-cardinality / predicate-count).**
- **rooms_visited** (distinct rooms visited): frozen baseline **35.8%** → best adapter **63.9%** (global LM-attn LoRA); symbolic-text ceiling **75.8%**.
- **co_occupancy** (frames two characters share a room): frozen baseline **31.9%** → best adapter **59.5%** (carrier-sum); symbolic-text ceiling **98.3%** → its headroom is in the vision→carrier path, not the arithmetic.
- The evidence-only sum adapter does **not** apply to either — neither has an evidence-only variant (both are inherently full/distractor tasks).

**Diagnosis on the new tasks (probe battery, both tasks).**
- Per-frame evidence is well extracted **and routed** to the carrier: room decode **0.84–0.89**, co-occ same/diff **0.93** (AUROC 0.97).
- But count is **not recoverable downstream**: count-from-messages ~**0.43–0.48** (≈ majority), last-token count decode **≤0.475** (≈ majority), frame-token-state pooling ≈ blind (~0.43).
- decode-then-count oracle ≫ model (rooms **0.675 vs 0.10**, co-occ **0.55 vs 0.27**) → the bottleneck is **aggregation/dedup, not extraction** — revises the earlier "vision extraction" framing.
- token-group corruption shows an identical frames→carrier→last staircase across all 3 tasks → the stage phenomenon is **task-general**.

**Insights**
- **One bottleneck explains all three MMReD tasks**: a bandwidth-limited carrier cut that saturates (~2 frames). Extraction is fine; aggregation is the cut.
- The fix is **DeepSets-shaped** (unnormalized sum, width ≥ max-count) for *scalar* counting — solves clean counting (100%, incl. count-OOD) — but *set-functions* (distinct-room, co-occ) need the carrier to hold the **set**, which a saturating scalar channel can't → direction is a **multi-slot external set-memory**.
- Distractor **selection** (~63–68% plateau) is robust to every selection mechanism tried; only richer aggregation nudges it (+4pp).


### [2026-06-19] Text-frames + diagnostic decomposition — the bottleneck is single-pass *extract→aggregate composition*, not aggregation capacity and not selection-signal availability

> **Refines the 06-16/06-17 "aggregation/dedup is the cut" framing.** Aggregation alone turns out to be
> *easy* once the per-frame evidence is handed over; the evidence is *linearly present* in both the text
> and the vision representations; the failure is the model's inability to **compose** per-frame
> extract-and-bind with cross-frame aggregation **in a single forward pass**. Maps onto
> [HERBench (arXiv 2512.14870)](https://arxiv.org/abs/2512.14870)'s two deficits (retrieval vs fusion).

**Method — three diagnostics (all text-side unless noted).** Scripts: `eval_mmred_text_frames_acc.py`,
`probe_evidence_selection_linear.py`, `probe_evidence_selection_image.py`.
- **Frames-as-text.** Feed the per-frame room→occupant state as *text* instead of rendered images — removes
  vision/perception entirely, leaving pure language-side aggregation.
- **CoT (chain-of-thought).** Let the model write out per-frame reasoning before the answer (vs forcing a
  single integer). Adds *serial computation steps* — externalises a running tally into the token stream
  ([Merrill & Sabharwal ICLR'24](https://arxiv.org/abs/2310.07923); [Li et al. 2402.12875](https://arxiv.org/html/2402.12875):
  CoT lets constant-depth transformers do inherently-serial work they can't do in one pass).
- **Oracle-list.** Feed *only the queried entity's per-frame rooms* (no scene to scan) — removes
  retrieval/selection, leaving pure cross-frame fusion. This is [HERBench]'s oracle-frame study, in text.
- **Linear selection probe.** Logistic regression on the frozen *per-frame* representation predicting
  `is_evidence` (C in R this frame), per layer, sample-disjoint split → is selection *linearly available*?

**Findings.**
- **Plain single-token, text frames** (`eval_mmred_text_frames_acc/`): steps_in_room **0.470**, rooms_visited
  **0.389**, co_occupancy **0.338** — identical collapse to the image pipeline ⇒ **not a vision problem**.
  Predictions are range-compressed toward a narrow prior band: steps/rooms **undercount** (bias →−1.7 at
  sl8), co-occupancy **overcounts** (bias +0.8) — i.e. "guess a plausible small number," not "count low."
- **CoT** (`_cot/`): steps **0.695**, rooms **0.570**, co-occ **0.623**; steps_in_room bias collapses to ≈0
  at every length (the systematic undercount was a single-pass artifact). Residual decay at long len remains.
- **Oracle-list** (`_oracle/`, `_oracle_cot/`): plain **0.841 / 0.883 / 0.662**; **CoT 0.985 / 1.000 / 0.940**.
  rooms_visited oracle-CoT is **100% at every seq_len incl. 8, bias 0.00** — overturning the 06-17 claim that
  the rooms_visited long-len saturation was a set-accumulation limit. **Aggregation/dedup is easy once
  evidence is pre-extracted; the saturation was the per-frame extract-and-bind step.**
- **Precision** (`_prec_nf4` vs `_prec_bf16`): nf4 ≈ full bf16 (0.470/0.389/0.338 vs 0.470/0.423/0.325); the
  saturation is unchanged ⇒ 4-bit weight quant is **not** the cause (rules out Barbero's precision lever here).
- **Linear selection probe** — evidence-vs-distractor is **linearly decodable** from the frozen per-frame rep,
  peaking mid-late (layers ~18–21): **text AUC 0.997** (`probe_evidence_selection_linear/`, layer 21, bal-acc
  0.965), **vision AUC 0.984** (`probe_evidence_selection_image/`, layer 19, bal-acc 0.939). The signal
  survives into the vision tokens too ⇒ the distractor gap is **not** vision-side perception.

**Insights / revised picture.**
- The bottleneck is the **single-pass composition** "for each frame: locate the queried entity, decide
  relevance, fold into a running count." Each piece is easy in isolation (probe: evidence present at AUC
  0.98–0.997; oracle-list: aggregation → 88–100%), but the frozen model can't chain them in one forward pass.
  **CoT** (serial steps) and **oracle-list** (pre-extraction) each relieve it; together → ~94–100%.
- **Why the prior selection experiments plateaued (≈47–68%) is now a *supervision* problem, not signal
  absence.** The evidence direction is sitting at AUC 0.98 (layer ~19), but gates trained only on the final
  count loss never discover it. (Also: aggregator richness — the PNA line — was optimising the part that
  *wasn't* broken, hence "PNA = sum" / only +4pp.)
- Caveats: probe is *supervised* (proves linear separability/availability, not that the model uses it);
  question placed first for query-conditioning (non-standard order); single seed; `steps_in_room` evidence
  label; oracle-list/CoT aggregation results are text-side (vision-side oracle aggregation untested).

**Next.** (1) **Per-frame-evidence-supervised adapter**: auxiliary per-frame `is_evidence` loss and/or
initialise the gate toward the probe direction at layer ~19 (vs end-to-end count loss alone); move
injection/readout from L14–17 to **~18–21** where the signal peaks. (2) Image-side **oracle-aggregation**
to confirm fusion is easy on vision too. (3) Count-magnitude OOD readout remains a separate unsolved
problem (PNA runs ~0% at unseen counts) — likely needs a regression/structured readout, not the frozen
token head.


### [2026-06-20] Frame-axis aggregator adapter — a task-agnostic one-pass fix; the residual is *extraction*, not the aggregator

> Built the adapter the 06-19 diagnosis implied (read clean per-frame reps at L19 → aggregate → inject)
> and swept it hard. **Plain DeepSets wins; PNA/codebook/balanced-loss don't help (PNA's count-scaler
> *hurts* OOD); the ~75% IID ceiling is per-frame extraction error compounding, not aggregation.**
> Scripts: `experiments/glstm/frame_axis_aggregator_adapter.py` (live, LM-injection readout),
> `experiments/glstm/frame_axis_aggregator_cached.py` (cached, count-head readout). Shared L19 rep cache
> at `outputs/frame_axis_cache/L19.pt`.

**Architecture (task-agnostic, one forward pass).** A forward-pre-hook on decoder layer **L19** reads each
frame's mean-pooled VISION-token rep, a learned per-frame φ → a fixed pool → side-channel; **two readouts**:
(a) **live**: inject `ρ(agg)·gate` into the residual at the answer position, the frozen-LM head emits the
count (deployable); (b) **cached**: a count-head on the aggregate (fast proxy — caches the 7B forward once,
trains in minutes). Trained **jointly on all 3 tasks**. Qwen frozen+4-bit.

**Methodology fixes worth keeping.** Disjoint **train/val/test split** stratified by seq_len (the IID eval
was previously train-overlapping); per-epoch **val** with best-epoch checkpoint; `val_cap` must sample the
**shuffled** split or it's short-seq-biased (val 0.84 vs true test 0.65 before the fix). Cached val ≈ test
after the fix.

**Results.**
- **deepsets > seqmodel** in both readouts (live val 0.889 vs 0.852; cached 0.844 vs 0.713). The order-aware
  transformer aggregator overfits these (permutation-invariant) tasks.
- **Cached aggregator sweep** (train+test 1–8, 120 ep, `outputs/frame_axis_sweep/`): mean test_iid —
  **deepsets 0.720**, pna_cb_balanced 0.714, pna_balanced 0.698, pna 0.690, pna_codebook 0.638,
  deepsets_codebook 0.611. **No config beats plain deepsets**; **codebook-φ falsified** (meant to fix
  rooms_visited dedup, made it worse); PNA narrow (only lifts additive tasks).
- **Live 4-way head-to-head** (train 1–6, test IID 1–6 + OOD 7–8, LM-injection, `outputs/frame_axis_live_h2h/`):
  | config | IID mean | OOD mean | note |
  |---|---|---|---|
  | **deepsets** | 0.754 | **0.558** | best OOD, smallest OOD bias (rooms −0.06) |
  | pna_balanced | **0.766** | 0.528 | best IID but **worst OOD bias** (steps −0.34, rooms −0.33) |
  | deepsets_balanced | 0.737 | 0.547 | balanced-loss marginal/negative |
  | pna_cb_balanced | 0.686 | 0.442 | codebook dead |
  **PNA's count-scaler HURTS OOD** (refuting the "scaler helps extrapolation" hypothesis): at unseen N the
  scaler value `log(N+1)` is itself OOD → it *introduces* a length-extrapolation failure.
- **Bias is solved two ways.** (1) Train on the **full length range (1–8)** → deepsets unbiased on all 3
  (rooms bias −0.04 vs −1.80 when trained 1–6 only) → the rooms_visited "saturation" was an **OOD-length
  artifact**, not an aggregation limit (`outputs/frame_axis_aggregator_cached_1to8/`). (2) The **live
  LM-injection extrapolates** to unseen lengths where the **cached count-head saturates** (OOD rooms bias
  −0.03 vs −1.80) — same aggregate, different readout → don't let cached OOD stand in for the model.

**The ceiling, explained.** IID ~0.75 (steps 0.78 / rooms 0.68 / co-occ 0.80, LM readout) is **per-frame
extraction error compounding**: the probe measured ~0.94 per-frame evidence accuracy, and exact-match count
needs every frame right → 0.94⁶ ≈ 0.69, matching observed. Oracle-list (clean per-frame facts) → 0.84–1.0,
so the 75→95 gap **is** extraction. **The aggregator is solved (sum of clean indicators is exact); the
remaining lever is per-frame extraction**, e.g. **query-conditioned attention pooling** over a frame's
tokens (vs the current mean-pool that dilutes the relevant token) — running now (`outputs/frame_axis_live_attnpool/`).

**Conclusion.** The **task-agnostic backbone is the plain DeepSets frame-axis adapter** (read@L19 → sum/mean/max
→ inject), trained on the right length range — simplest, best-generalizing, smallest; it relieves the
aggregation bottleneck across counting/set-cardinality/relational and is unbiased. **The aggregator search
is done** (PNA/codebook/balanced don't help; PNA hurts OOD). Remaining gains are in **per-frame extraction**
(esp. rooms_visited ~0.68) and length-robust readout — not the aggregator.

**Caveats.** Single seed; live runs ≤20 epochs (a deepsets +10-epoch continuation nudged val 0.748→0.763,
so not fully peaked); OOD entangles length+count; cached count-head under-reports OOD vs the live readout.

---
### [2026-06-20b] Localizing the "extraction ceiling": it's single-pass *superposition*, not perception — and rooms is the genuine aggregation residual

> Five diagnostics to pin down *what* the ~0.94/frame extraction ceiling actually is and whether we're
> stuck at it: (1) 32B backbone, (2) resolution sweep, (3) crowding bucketing, (4) per-frame "look-again"
> verification, (5) single-task rooms training to convergence. **Headline: the per-frame info is fully
> present (isolated-frame readout → AUC 1.0); the 0.94 is a single-pass *superposition* artifact, not a
> perception limit — recoverable by per-frame passes (steps count 0.79→0.93) but only when the task is
> extraction-limited. rooms is aggregation-limited (plateaus ≪ its bound); co-occ is already soft-agg-optimal.**
> Scripts: `evaluations/scripts/eval_per_frame_verification.py` (new), `probe_evidence_selection_image.py`
> (extended: `--image-sizes` resolution sweep + crowding bucketing). Runs under `outputs/frame_axis/probes/`.

**1. Bigger backbone does NOT raise perception (32B vs 7B, peak-over-layers, same probe).** steps is-evidence
**0.892/0.951** (32B L43) vs 0.939/0.984 (7B L19); co-occ same-room **AUC 0.999** (32B) vs 0.996 (7B); rooms
room-decode **0.858** (32B) vs 0.915 (7B). **No lift** — co-occ already saturated at 7B; steps/rooms flat.
(32B came in slightly *lower*, but on 60 vs 90 samples and dim 5120 vs 3584 with the same linear probe → that
biases the probe down; the defensible claim is "no increase," not a decrease.) → perception is **not
capacity-bound**; scaling won't fix it.

**2. Resolution plateaus at native (not a lever).** Per-frame is-evidence AUC by frame size: 224px 0.873 →
336 0.945 → 448 0.971 → **native-512 0.969** → 672(upscaled) 0.982. Below ~448px it degrades, but we already
feed native 512 and upscaling buys only ~+1pt. **The 0.94 is not resolution-limited at the resolution we use.**

**3. Crowding (underpowered, directional).** is-evidence acc by #characters-in-frame, native res: 4ch 0.962 vs
5ch 0.926 (gap widens at low res). Directionally **more entities → lower acc = binding/superposition**, but the
dataset only ever has 4–5 chars/frame so the axis barely varies (crowd-4 n=26) — *suggestive, not decisive*.

**4. Per-frame "look-again" verification — the key result.** Ask one focused single-frame question
("is C in R here?" / "are C and D in the same room here?"), read P(yes), sum across frames.
- **steps:** per-frame **bal-acc 0.987 / AUC 1.000** (≫ joint 0.94); **count 0.79 (single-pass adapter) → 0.928**.
  Isolated frames decode ~perfectly ⇒ **the 0.94 ceiling is the single joint pass having to hold all N frames
  at once (superposition), not perception.** The **extraction-axis analog of CoT** — relieves it at N× compute.
- **co-occ:** per-frame 0.909/0.981 but **count 0.722 — *worse* than the single-pass adapter (0.867)**, and it
  *decays* with length (0.867→0.717→0.583 over sl 4/6/8). co-occ's single-pass adapter wins via **soft
  error-cancellation** (it already beat its hard extraction bound, 0.867 > 0.815); per-frame *hard*-sum throws
  that away and compounds. **co-occ is not extraction-limited.**
- Method note: **hard-sum > soft-sum at long sequences** (steps sl8: 0.883 vs 0.700) — summing soft P(yes)
  accumulates per-frame leakage → overcounts (noisy-OR-style saturation).

**5. Single-task rooms to convergence (rooms30, deepsets, 30ep, patience-8).** Val plateaued at ~0.77 from
epoch 6 (early-stopped ep14) while train loss kept falling; **final test_iid 0.691, ~17pt under the 0.865
extraction bound**, g5 collapses to **0.184**. **rooms is aggregation-limited (structural) — more training does
not reach the bound.** (Test 0.691 < multi-task 0.748, but rooms30 trained on less data — cap 1140 vs 1890 —
so this is *not* clean evidence that single-task hurts; the robust claim is only "plateaus short of bound.")

**Synthesis — recognition vs binding, and option-4.** The extraction ceiling is **binding/superposition, not
recognition**: resolution is maxed at native (2), perception doesn't scale with model size (1), and isolated
per-frame extraction is ~perfect (4, AUC 1.0). The single joint pass simply can't read all N frames cleanly at
once — **the same over-squashing as the aggregation thesis, one level up the pipeline.** The blanket "we're
extraction-bounded, this is the best possible" (option-4) is **refuted**: per-frame passes exceed it for steps,
and soft aggregation already exceeds the hard bound for co-occ. The defensible claim is *single-pass*
optimality. **Per-task verdict:** steps = single-pass-extraction-limited (multi-pass fixes); co-occ = already
soft-agg-optimal; **rooms = the genuine residual aggregation bottleneck** (one-pass distinct-count plateaus
below its bound). *(Open: whether the per-frame extraction tax exists on the **text** versions — measured only
for steps text, where it's ~absent at 0.997; not yet run for rooms/co-occ text.)*

---
### [2026-06-21] The extraction ceiling IS entity-crowding (superposition), measured directly; and rooms is aggregation-bound even with perfect extraction

> Built count-balanced datasets with a **controllable entity count** (`generate_mmred_balanced.py --n-chars`)
> and isolated extraction from aggregation. **Headline: per-frame extraction at the deployed L19 declines
> monotonically with crowding (rooms 0.996→0.915→0.835 for 1→2→5 chars; steps 0.984→0.970→0.939) and is
> ~1.0 with one entity — so the ~0.92 real-task ceiling is *superposition*, not perception. But removing the
> extraction bound does NOT fix rooms: with perfect per-frame extraction (target-token read, AUC 1.0) the
> adapter still only reaches 0.41, and the frozen model only 0.29 — distinct-count is aggregation-bound.**
> Scripts: `generate_mmred_balanced.py` (now rooms/co_occ/steps + `--n-chars`), `probe_text_pooling_sweep.py`,
> `probe_pertask_extraction.py` (+per-layer CSV, empty-list guard), `frame_axis_aggregator_adapter.py`
> (+`--text`, +`frame_pool=target`). Datasets: `data/mmred_{rooms,steps,cooc}_balanced` (5-char, uniform
> counts), `data/mmred_{rooms_1char,steps_1char,cooc_2char,cooc_3char,rooms_2char,steps_2char}`.

**1. Text pooling sweep (`outputs/frame_axis/probes/text_pooling_sweep/`).** mean/last/max/target × layer, on
the balanced text frames. **rooms:** mean 0.767 / last 0.733 / max 0.764 (all ~L19–26) vs **target-token L1 =
1.000** — no content-pooling at the working layers recovers it; only reading the queried entity's own token at
the embedding layer does (positional/parsing-trivial). **co-occ:** mean 0.964 is already best; pooling
barely matters → co-occ is NOT pooling-handicapped. So "image>text" for rooms was a recipe artifact (mean-pool
superposes the multi-room text block); for co-occ it isn't.

**2. Count-balanced image-vs-text deepsets (`outputs/frame_axis/balanced/`, live 7B, N=8, uniform counts).**
test_iid LM acc — **rooms** image 0.500 / text 0.398; **co-occ** image 0.586 / text 0.438 (0.519 at 35ep);
**steps** text 0.704. Image≥text on both (rooms +10pt, co-occ +15pt); image near-unbiased, text biased
(rooms +0.32, co-occ −0.34). Balanced data removes the old skew — co-occ's honest number is 0.586 (vs the
distribution-inflated 0.867). Caveats: small per-count support (~17–24), noisy val, single seed.

**3. target@L1 upper bound (`outputs/frame_axis/balanced/rooms_text_targetL1/`).** Read the queried token at
L1 (where rooms extraction = 1.0) → deepsets. **lm 0.435 / aux 0.407.** Perfect per-frame extraction does NOT
solve rooms — the aux count-head (bypasses the LM) agrees at 0.41 → it's the **aggregation** (distinct-count),
not the readout. The earlier "soft-OR fails rooms" was confounded by extraction noise (saturation); clean
extraction is the untested rescue.

**4. Minimal-crowding extraction (`outputs/frame_axis/probes/crowding_min/`, image, L19).** rooms 1ch **0.996**
/ 2ch **0.915** / 5ch **0.835**; steps 1ch **0.984** / 2ch **0.970** / 5ch **0.939**; co-occ 3ch **0.999**,
5ch 0.996 (1–2ch = AUC 1.0 but **density-confounded**: with ≤2 chars "same room" ⟺ "1 occupied room", trivially
separable — discard those, use ≥3ch). **Monotonic decline with crowding at the deployed layer = superposition,
measured directly.** rooms/steps are extraction-(crowding-)limited; co-occ extraction is ~0.99 at every
crowding → never extraction-bound.

**5. Frozen base acc at minimal crowding (`outputs/frame_axis/probes/base_acc/`, image, no adapter).** steps
1ch **0.583** (U-shaped per count: 1.0 at gold 0 and 8, 0.25 at gold 3), rooms 1ch **0.289** (undercounts
3.5→2.7), co-occ 2ch **0.204** (undercounts 4.0→1.9). **Even with one entity and ~perfect extraction, the
frozen 7B fails — worst on distinct-count/relational, best on sum** — confirming the bottleneck is single-pass
*aggregation*, present already at minimal crowding.

**Synthesis (task split, now nailed).** (a) **Extraction** is bounded by **entity superposition** at the
pooling step — provably (1 entity → ~1.0 at L19; monotonic decline with crowding). The agnostic, non-bounding
fix is to stop collapsing each frame to one mean vector (slot-attention / per-token aggregation that separates
entities); L19 is the right task-agnostic layer (it *holds* the ~1.0 signal — 1-char proves it). (b) **steps /
co-occ are extraction-(crowding-)limited** → clean extraction lifts them (steps text 0.704). (c) **rooms is
aggregation-limited** → perfect extraction still caps it at 0.41; it needs a better *aggregator*
(union-then-count), not a better extractor. The open test: soft-OR with clean (minimal-crowding) extraction.

---
### [2026-06-23] Minimal-crowding aggregator experiment — decrowding + DeepSets solves the count/set family; soft-OR perfects distinct-count

> The experiment the 06-21 work set up (`outputs/frame_axis/agg_min/`, spec in `outputs/frame_axis/SPEC_minimal_crowding_aggregator.md`):
> live 7B adapter, image, L19, mean-pool, **DeepSets vs `logic` (sum+soft-OR+soft-AND)**, one task-agnostic
> module per task, on **count-balanced minimal-crowding** datasets (rooms·1char, steps·1char, co-occ·3char;
> 250/count, `data/mmred_agg/`). **Headline: clean extraction (decrowding) is the dominant lever — it lifts
> every task from ~0.2–0.5 to ~0.9–1.0 with PLAIN DeepSets; soft-OR perfects distinct-count (rooms 1.000).
> So the "rooms aggregation wall" (target@L1 0.41) was a layer/setup confound — rooms was crowding-limited.**

**Results (test_iid LM acc; frozen base / hard-compounding ceiling in parens).**
| task | frozen base | **DeepSets** | **logic (soft-OR/sum)** | ceiling |
|---|---|---|---|---|
| rooms_visited | 0.289 | **0.973** (all counts ≥0.91) | **1.000** (perfect every count) | ≈1.0 (agg) |
| steps_in_room | 0.583 | **0.950** (≥0.85 all counts) | **0.961** | 0.86 |
| co_occupancy | 0.204 | **0.899** (counts 2,4 dip to 0.76–0.79) | **0.938** (all counts ≥0.86; soft-sum +0.039) | 0.78 |

**Reads.** (1) **Decrowding is the lever:** rooms 0.50 (balanced-5char) → 0.973 (1char, DeepSets); steps 0.58→0.95;
co-occ 0.20→0.90. (2) **DeepSets suffices** for the sum tasks and gets rooms to 0.973; **soft-OR (logic) is the
correct distinct-count operator** and perfects rooms (1.000), fixing exactly the high-count dips — vindicating
soft-OR, which had failed earlier only because *extraction noise* made noisy-OR saturate. (3) **Both sum tasks
beat their hard-compounding ceiling** via soft summation (steps 0.95>0.86, co-occ 0.90>0.78). (4) On *clean*
data logic ≥ DeepSets everywhere (it only *hurt* under crowded/noisy extraction) — but it's more complex/fragile,
so **DeepSets is the recommended robust default; soft-OR is the principled refinement for set-cardinality.**

**Confound check (honest).** The 0.50→0.973 *adapter* comparison is confounded (the minimal run also had more
epochs 35 vs 20, more data 1050 vs 504 train, balanced-loss on). The *clean* isolation is the **extraction
probe** (linear, no training, only #chars varies): rooms L19 hard-acc **0.835 (5ch) → 0.915 (2ch) → 0.996 (1ch)**;
steps 0.939→0.970→0.984; co-occ ~0.97 at 3ch — pure crowding. **Caveat:** minimal-crowding tasks are perceptually
trivial; the claim is about *aggregation*, and the realistic crowded task stays extraction-bounded (which a 4×
backbone, 32B, does NOT fix). Frame as a controlled **aggregation-isolation** experiment, not a benchmark headline.

**Operational.** Runner hard-codes `--time=12h`; these slow image runs (~20 min/epoch) hit the wall mid-final-eval
and were killed (no checkpoint) → relaunched with `--time=18h`, 30 epochs. **Fix the runner default.** A **plain-LoRA
SFT baseline** (`experiments/glstm/lora_sft_baseline.py`, peft 0.17.1) — "does fine-tuning native softmax suffice
vs the explicit aggregator?" — is *running* (the apparent hang was ~18-min cold-cache model-load on n315, not a bug);
results pending.

**Scope (full MMRED task taxonomy).** The approach (decrowd + permutation-invariant sum/soft-OR) covers the entire
**LC family**: the count tasks (`steps_in_room`, `rooms_visited`=soft-OR, `crowd_count`, `room_busy`,
`char_accompanied`, `char_alone`) directly; the "which/who" argmax tasks (`where_spend`, `crowded_room`, `who_spend`,
`spend_alone`, `spend_together`, `room_empty`) need a **comparative/argmax readout** but the aggregation is in-scope.
It is **structurally wrong for the NIAH family** (FA/FI/FX: first/last/step-X) — those are order/position retrieval,
which permutation-invariant pooling discards; they'd need the **order-aware `seqmodel` aggregator**. Relational tasks
(co-occ, accompanied, alone) can't decrowd below ~2–3 chars.

**Related work / novelty (lit search 2026-06-23; sources in References).** Every *ingredient* is published —
query-conditioned visual pooling (QG-VTC, PARCEL; for efficiency), the image-before-question ordering effect
(observed in MLLM probing, arXiv:2508.20279), permutation-invariant *counting* (Set Transformer counts unique
elements), DeepSets on *frozen-LLM* states (ILSE), activation read+inject (steering). **The specific combination —
reading question-conditioned per-frame states from a frozen autoregressive VLM, DeepSets/soft-OR aggregating across
frames, injecting back, framed as over-squashing relief — has no direct precedent found.** Notably, **arXiv:2511.17722
(Nov 2025) independently diagnoses VLM counting as "locate-but-not-enumerate under cognitive load"** = our
extraction-fine/crowding-kills-aggregation result; they intervene via attention reweighting, we provide an explicit
permutation-invariant aggregator + the extraction-vs-aggregation decomposition. Cite as concurrent corroboration.

---
### [2026-06-23] Count-extrapolation OOD + injection-direction failure — the readout, not the aggregator, is the OOD wall

> Question: does the structured adapter beat LoRA where it *should* — on **counts never seen in training**?
> (IID both tie, so only extrapolation can separate them.) Setup: minimal-crowding `mmred_agg/*`, fixed
> 8 frames, **count-holdout** split (`--holdout-counts`): train on low counts, test on held-out high counts.

**Head-to-head, train low / test held-out high (`outputs/frame_axis/ood_holdout/`):**
| task (held-out) | deepsets | logic | **LoRA** |
|---|---|---|---|
| rooms (5,6) | 0.98→**0.000** | 0.96→**0.000** | 0.99→**0.000** |
| steps (7,8) | 0.97→**0.000** | 0.95→**0.000** | 0.98→**0.452** |
| co-occ (7,8) | 0.93→**0.000** | 0.94→**0.000** | 0.99→**0.094** |

- **All methods cap at the top trained label OOD.** `mean_pred` pins at exactly 4.0 (rooms) / 6.0 (steps,cooc); accuracy on held-out counts = 0. LoRA's only "win" is **count 8 = "all 8 frames"**, copied from the prompt's stated frame count (nails 8, fails 7) — a cue, not counting. So on this axis LoRA ≥ adapter, and **neither extrapolates intermediate counts.**
- **Root cause = the readout, not the aggregation.** The aux count head is a **9-way CE classifier** (`aux_head`, `frame_axis_aggregator_adapter.py:138`); held-out count classes get **zero gradient** → can never be argmax'd. The LM-injection readout (CE on output tokens) has the same disease. The aggregate sum keeps growing; the closed-label readout throws that away.

**Can we instead INJECT a count direction so the frozen LM verbalizes it (Solution 3)?** Two probes
(`probe_count_direction_extrapolation.py`, `probe_generic_number_direction.py`):
- **READ — number axis compresses.** Fit a linear count direction on counts 0–4, read 5–8.
  *Task-count axis* (`probes/count_direction_extrap/`): monotonic but heavily compressed — count 8 reads ~5.3, held-out acc ≈ 0.
  *Generic LM-numeracy axis* (arithmetic prompts, `probes/generic_number_direction_L{16,19}/`): **perfect in-range (acc 1.0 on 0–4)** and better at 5 (~0.55–0.65), but **still saturates** (8 → ~5.5; 7,8 acc 0). So the LM encodes magnitude linearly **only up to ~5, then saturates**; the frame-aggregated count is *even more* collapsed (over-squashing signature).
- **WRITE/steer — non-causal.** Forcing the residual along the direction, and a dose-response sweep to **±16×residual-rms**, left `emit_mean` **perfectly flat** at *both* L16 and L19 for *both* directions. Injecting a direction at the answer position has **zero causal control** over the emitted number. (The only injection that *does* control output is a learned per-count **codebook** — `translator_*` 100% in-range — which **cannot extrapolate** by construction.)

**Verdict: injection-for-extrapolation is a dead end.** Read-axis saturation + non-causal steering, robust across directions/layers/magnitudes. You can make the frozen model count *in-range* (codebook=100%) but not *extrapolate* via injection. **Pivot: compute the count externally with an extensive additive head** (`count = Σ σ(w·φ(rᵢ))`) and **read the scalar directly** — extensive by construction, bypasses the LM's saturating number geometry. Mechanistic finding worth citing regardless: **frozen Qwen's number representation is linear only to ~5 then saturates.**

**Additive-readout extrapolation — CONFIRMED (fast CPU probe on cached L19 per-frame reps, `cache/L19.pt`; crowded data, per-frame AUC~0.94).** Same reps, only the readout differs; trained on counts ≤4, tested on held-out 5–8:
| task | additive `Σσ(w·repᵢ)` on held-out counts | 9-way classifier |
|---|---|---|
| **steps** (occurrence) | 5→4.9(**.82**) 6→5.8(.83) 7→6.8(.88) 8→7.8(**.93**) — **extrapolates** | caps ~4, **acc 0.00** |
| **rooms** (distinct) | caps ~3.9 (acc ~0) — **plain sum is the wrong operator** | caps, acc 0.00 |
| co-occ (occurrence) | inconclusive — crowded cache has only 4 examples with gold≥5 | — |
- **The readout, not the aggregation, was the OOD wall.** A sum-of-per-frame-probabilities extrapolates to counts never trained (steps 0.82–0.93 on 5–8) *even on crowded data*; the closed-label softmax caps at the top trained count (0.00). This is the cleanest isolation — identical reps, readout swapped.
- **Operator must match the task:** `Σσ` (sum) extrapolates for *occurrence*-counting; *distinct*-count (rooms) needs the **soft-OR extensive readout** `Σ_room[1−Π(1−p)]` (not yet probed — needs a per-frame per-room head; agg_min already showed soft-OR perfects rooms IID).
- On *minimal-crowding* data (per-frame AUC 0.98–0.996) the additive accuracy would be higher; full-adapter minimal-crowding confirmation: `ood_holdout/{steps,rooms}_additive` (`--count-readout additive`).

---
### [2026-06-23] OOD count-extrapolation benchmark — what aggregator extrapolates, and *why* (the readout principle)

> Setup: cached per-frame **L19** reps for minimal-crowding `mmred_agg/{steps_1char,rooms_1char,cooc_3char}`
> (one frozen pass, `cache/minimal_L19_*.pt`), then **count-holdout** (train counts <5, test held-out ≥5)
> readout experiments on CPU — fast, multi-seed, no LM injection (count read directly from the head).
> All in `outputs/frame_axis/readout_benchmark/`.

**Headline benchmark (IID → OOD exact-count accuracy; OOD = counts never trained):**
| method | steps IID→OOD | rooms IID→OOD | co-occ IID→OOD |
|---|---|---|---|
| base Qwen (0-shot) | 0.23→0.11 | 0.31→**0.00** | 0.00→0.00 |
| CoT Qwen | 0.30→0.63† | 0.33→**0.18** | 0.21→**0.06** |
| LoRA Qwen | 0.98→0.45* | 0.99→**0.00** | 0.99→0.09 |
| classifier (9-way CE) | 0.65→**0.00** | 0.96→**0.00** | 0.54→**0.00** |
| **sum** (per-frame-sup) | **0.996** | 0.966–0.974 | **0.974** |
| **soft-OR** (per-frame-sup) | — | **1.000** | — |

\*LoRA's steps-OOD 0.45 is the count-8="all 8 frames" prompt cue (nails 8, fails 7), not real counting.
†CoT steps-OOD 0.63 > its IID 0.30: CoT is well-calibrated (mean_pred≈gold both splits) but imprecise — it makes ±1
errors on mid counts (low IID exact-match) yet nails the *saturated* high counts 7,8 ("all/almost-all frames"),
inflating OOD exact-match. Still far below sum (0.996), and CoT collapses on rooms (0.18) / co-occ (0.06).

**The central finding — extrapolation requires a FIXED extensive readout; learned readouts provably don't.**
Across every configuration tried, the *only* readouts that extrapolate are **parameter-free** reductions of the
**supervised per-frame quantity**: `count = Σᵢ pᵢ` (sum, occurrence) and `Σ_slot[1−Πᵢ(1−p)]` (soft-OR, distinct).
*Any* learned readout re-introduces non-extrapolating solutions (it fits counts ≤4 with a combination that caps
beyond). Evidence (multi-seed OOD mean±std, `stability.csv`/`deepsets_*.csv`):
- **classifier (9-way CE):** 0.00 OOD everywhere — closed-label cap (`mean_pred` pins at top trained count).
- **count-only scalar sum:** *unstable* — steps **0.52±0.38** (bimodal 0.99/0.21), co-occ 0.79±0.27, rooms 0.97±0.01
  (rooms stable because the causal reps strongly encode a "new-room" signal). Under-determined by aggregate-only loss.
- **canonical DeepSets** `ρ(Σφ)`, multi-dim φ-MLP, **count-only:** 0.56/0.54/0.25, unstable — *more capacity made it
  worse* (more shortcut basins).
- **canonical DeepSets + per-frame aux supervision:** still fails (steps 0.54±0.31, co-occ 0.22–0.38) — the per-frame
  loss shapes φ but the **separate learned ρ stays decoupled** and doesn't extrapolate. **ρ-MLP caps OOD more than
  ρ-linear** (nonlinear readout saturates at large sums).
- **"universal" DeepSets** (multi-dim φ + fixed soft-sum/soft-OR channels + **linear** readout + per-frame sup):
  **still fails** — steps 0.13, rooms 0.20, co-occ 0.08. **Even a *linear* learned readout over extensive channels
  breaks extrapolation.** → The readout must be *parameter-free*, not just linear.
- **Latent-dim control (`dimsweep.csv`):** canonical DeepSets count-only, ρ=linear, d ∈ {64,256,512,1024}.
  OOD *monotonically decreases* with dim (steps 0.42→0.06, rooms 0.58→0.10, co-occ 0.25→0.005). **More capacity
  makes it WORSE** → the failure is *not* a representational-capacity / Wagstaff-dim issue (already satisfied at
  small d); it's learned-readout over-parameterization overfitting the bounded training-count range.
- **Query-routed bank (`benchmark_query_router.py`):** one module = query-conditioned detector + router selecting
  {sum, soft-OR}. **Failed** (OOD 0.24–0.46, mis-routed) — a *single shared* detector can't serve conflicting
  per-task slot semantics, and a soft blend doesn't extrapolate. Auto-routing across operator *types* remains open;
  the working task-agnostic method is the single-operator **marginal-contribution sum** (additive counting family).
- **per-frame-supervised sum / soft-OR (fixed readout):** **stable & near-perfect** — steps 0.996±0.001, co-occ
  0.974±0.003, rooms 1.000±0.000 (soft-OR) / 0.974 (sum, first-visit label).

**Definitive readout ablation (`benchmark_readout_ablation.py`) — per-frame detector held FIXED, vary ONLY the pooling:**
| readout (same pᵢ) | steps IID→OOD | co-occ IID→OOD | verdict |
|---|---|---|---|
| **sum** `Σpᵢ` (fixed, parameter-free) | 0.97→**0.997** | 0.90→**0.974** | **fully extrapolates** |
| mean `Σpᵢ/N` | 0.20→0.00 | 0.20→0.00 | count-blind (read direct)* |
| max `maxᵢ pᵢ` | 0.39→0.00 | 0.37→0.00 | count-blind (saturates) |
| **learned ρ(Σpᵢ)** (MLP on the *correct* sum) | 0.98→**0.744** | 0.92→**0.739** | fits 0–4, **degrades OOD** |
- **The headline figure:** same detector, swap only the readout. **Only the fixed parameter-free `sum` is lossless OOD.**
  A learned decoder *on the correct sum* fits IID (0.98) but **drops ~25 pts OOD** — nothing should be learned between the
  sum and the answer. (Degradation *scales with the readout's input dim*: 1-D scalar-sum → mild 0.74; high-dim `Σφ`
  canonical-DeepSets ρ → hard fail ~0.5. Fixed sum wins in every case.)
- \*fixed N=8 here, so `mean = sum/N` is a *rescaled* sum (recoverable by ×N); read directly it predicts the fraction →
  count-blind. Mean's failure is *fundamental* only when N varies; `max` and `learned ρ` are count-blind/non-extrapolating
  even at fixed N.

**Why sum doesn't overcount distinct (rooms):** it sums **first-visit** indicators, not occupancy — `[A,A,B]→[1,0,1]→2`.
Causal attention lets frame *i*'s rep encode "seen this room before?", so the detector suppresses repeats.

**Verification (`verify.csv`):** leak probes (regressors on mean/last/first-pooled reps) **fail** to extrapolate
(0.0–0.1) while the sum succeeds; `sum_shuffle_diff = 0.000` (permutation-invariant). So the sum genuinely aggregates
per-frame evidence — not a count leaked into a single pooled vector.

**How much supervision (`auxloss.csv`, count-MSE + λ·per-frame-BCE):** a token λ does nothing; λ must be large
(steps stabilizes only at λ≈1.0; co-occ at λ≳0.2). The per-frame objective must genuinely *steer* the detector.

**Task-agnostic method (no operator selection):** `count = Σᵢ (per-frame "+1 marginal contribution")`, one fixed-sum
readout. The per-frame label is a single unified concept — "does this frame add 1 to the answer" — auto-derived per
task (occurrence: evidence; distinct: first-visit). Covers the whole *additive counting* family with **one operator**;
validated: steps 0.996 / rooms 0.974 / co-occ 0.974. **Scope:** additive "how-many" tasks. Genuinely non-additive
permutation-invariant functions (max, variance, threshold "crowded ≥3") need their *own* fixed reduction — a property
of the function class (DeepSets hides it in a learned ρ that doesn't extrapolate), not a flaw in the method.

**The read → reduce → decode template (the generalization, and where task-knowledge enters):** the method is not a
counting trick — it's a 4-stage template, and *each stage has a defined agnosticism level*:
```
[A] read  per-frame state x_i from the frozen VLM      <- TASK-AGNOSTIC (always identical)
[B] phi:  x_i -> per-frame state p_i                   <- form agnostic; SUPERVISION TARGET is the task knob
[C] reduce: combine p_1..p_N  -> z (fixed-size summary) <- task-specific CHOICE from a small FIXED menu
[D] head:  decode z -> answer                          <- task-specific, but tiny & swappable
```
- **Is φ's loss task-specific?** The *architecture* of φ (small MLP + nonlinearity) is identical across tasks. Only the
  per-frame **target** `label_i` changes ("is-evidence" / "which room" / …) — a **one-line declaration, not a learned
  black box**. So φ's *form* is agnostic; the supervision label is the task knob. (You *can* drop the per-frame label
  and train φ end-to-end from the final answer — but that is exactly where extrapolation got fragile: itself a finding.)
- **The reduction `z = REDUCE_i φ(x_i)` is a task-agnostic set summary** — a fixed-size summary of the whole set — and
  you bolt **any fixed head** on it. The last step is the only thing that changes by question type:
```
which room did C spend most time in?  ->  [B] p_{i,r}=prob C in room r at frame i  (same per-frame state)
                                            [C] score_r = SUM_i p_{i,r}              (per-room occupancy)
                                            [D] answer  = name( argmax_r score_r )   (categorical decode)
```
  - **count** question → `[D]` = `sum → scalar`;  **name** question → `[D]` = `argmax → label`;  **yes/no** → `soft-OR
    → threshold`. Same skeleton (read+reduce identical), different *last step*. So text-answer tasks (room/char name)
    are covered by swapping only `[D]` to a categorical (argmax/softmax) decode — `[A][B][C]` are unchanged, and argmax
    is parameter-free so it extrapolates trivially (the name vocabulary is a closed known set, no magnitude OOD).
- **Why it stays interpretable & extrapolates:** "what question is this" lives in **three small declarative knobs** —
  (i) what φ extracts, (ii) which reduction, (iii) which head — *not* in a big learned decoder. The generalization
  pitch: **a read → reduce → decode template where counting is one instantiation** (scalar/count, argmax/name,
  soft-OR/yes-no, regressor, … all bolt onto the same agnostic set summary `z`).

**Net thesis claim:** *a per-frame-supervised, permutation-invariant sum with a **fixed extensive readout** extrapolates
to unseen counts on all three tasks (0.97–1.0) where base Qwen, CoT, LoRA fine-tuning, and a closed-label classifier
all collapse (≤0.45, mostly 0). Learned readouts — including the canonical DeepSets ρ — provably do not extrapolate,
regardless of φ capacity or per-frame supervision.* (Lit grounding: DeepSets/Set-Transformer, Wagstaff set-function
limits, PNA degree-scalers — see References, pending verified citations.)

---
## Standing reference (cross-week synthesis)

### What's working
- **The diagnosis is solid and convergent.** Probes (count decodable at ~95% seq2 / 40% seq8 while the
  model gets 30%), causal ablations (evidence-frame influence vanishes by seq8), last-token cosine
  collapse, and the nested-growth margin curves all say the same thing: **per-frame evidence is present
  and recoverable; the model cannot aggregate many frames in one forward pass.** This is an
  over-squashing / bottleneck story ([Alon & Yahav 2021](https://arxiv.org/abs/2006.05205)), which is
  exactly the GNN-message-passing framing the thesis wants.
  - **Direct prior for the last-token signature:** [Barbero et al. 2024, *Transformers need glasses!*](https://arxiv.org/abs/2406.04267)
    prove decoder-only transformers suffer **last-token representational collapse** — distinct input
    sequences map to arbitrarily close final-token representations — explicitly connect it to GNN
    over-squashing, and show it produces errors *specifically in counting and copying*, **worsened by
    low-precision floats**. Our early-diagnosis last-token cosine-collapse metric (0.060→0.019, seq1→8) is the
    empirical signature of exactly this; their bf16 caveat is also a flag for our 4-bit/bf16 runs. This
    is the single closest transformer-side precedent — cite it as the mechanism behind the diagnosis.
- **Evidence-only counting is fully solved** by a small frozen-Qwen adapter that **sums per-frame
  attention messages** and injects them at carrier/question tokens (L14–17): 100% at seq 1–8 vs ~39%
  base, MAE→0. Sum, layer-local, and raw-matrix readouts all reach 100% — confirming the
  **additive-aggregation hypothesis** (sum of per-frame signals ≫ joint MLP, 63.7% vs ~22% on the probe).
- **And it extrapolates perfectly (2026-06-12):** trained on seq/count 1–4 only, the converged (3-ep)
  sum adapter scores **100% on seq/count 5–8** (n=100/seq) with exactly calibrated mean predictions —
  lengths, counts, and answer labels all unseen in training. The additive mechanism *computes* the
  count rather than memorizing the mapping (contrast: LoRA/maxmix get 0% count-OOD). This is the
  "simple solution that generalizes" for the evidence-only task; the open problems remain distractors
  and count-extrapolation on the distractor task.
- **gLSTM beats everything simpler on clean data.** Layerwise *persistent* gLSTM ≈ 98–99% IID and
  88–89% length-OOD, and in the head-to-head `final_glstm_aggregation_comparison` it leads on the hard
  high-aggregation-extrapolation split (70.5% vs sum 46% vs LoRA 48.5%). Cross-layer state persistence
  is the ingredient that buys length extrapolation (+18.7pp). This is the leading approach.
- **Injection geometry is understood:** write early (L14–17), read later (L18–27); multi-layer >
  single-layer; carrier/question tokens > last-token-only (which caps ~46%); discrete slots (k=8) beat
  continuous mixing. Oracle count injection → 100%, so **the frozen model can use a clean count signal**
  — the problem is producing one, not consuming it.

### Deep dive — aggregator ingredients: only two of three are load-bearing (counting + distractors)
- **The aggregator decomposes into three ingredients; only two are load-bearing.** Matched-harness
  controls (same data/LoRA/training, only the read changes) isolate each: **(1) unnormalized read is
  necessary** — softmax-normalizing the memory read costs −9.8pp length-OOD (77.8→68.0) with IID
  unchanged, the predicted over-squashing signature; **(2) persistent cross-layer slot memory is
  necessary** — prior +18.7pp ablation; **(3) associative q·k addressing is *not* necessary** — a plain
  sum read ties or beats the gLSTM's associative read on clean (76.2 vs 77.8 len-OOD) *and* distractor
  (63.8 vs 61.7 IID) data. **Net: the bottleneck is relieved by a persistent, unnormalized, additive
  virtual-node memory — a sum aggregator. The gLSTM's ([arXiv:2510.08450](https://arxiv.org/abs/2510.08450))
  distinctive memory-addressing machinery is dispensable on MMRED.** (Earlier gLSTM > sum results came
  from clean/extrapolation splits with a *fresh*-query or final-only sum, not this matched persistent-sum control.)
- **A plain sum memory is the new best learned distractor method** (63.8% IID, distractor fillers) —
  above codebook (60.7%), LoRA-attn (52.6%), gated mixer (51%) — using no gating at all.
- **The 96.3% oracle bound is decomposed:** the **negative/absence stream is the dominant ingredient
  (~+20pp)**, late read only ~+5pp. The neg stream works by calibrating *low* counts (cnt0 60→100,
  cnt1 7→93): counting needs evidence for the count *and* for its complement (signed aggregation), not
  a filter-then-sum.
- **The learned-vs-oracle gap is an optimization problem, not a detection problem.** Across 5 learned-gate
  variants, task accuracy is pinned at 39–47% **independent of gate AUC (0.61↔0.91)**, training length,
  gate layer, and hardness; per-sample gate-count error is uncorrelated with prediction error (r=−0.18).
  46.7% is exactly the old gateless-adapter score → joint training reaches the ungated-aggregate optimum
  and the gate stays decorative. Detection, hardness, epochs, and gate placement are each falsified as
  the cause.

### Deep dive — the counting fix is DeepSets, with two measured causal knobs
- **The whole story collapses to one sentence:** the model cannot count because attention is a
  **normalized, bounded-width mean**, and the minimal fix is an **unnormalized sum (DeepSets,
  `ρ(Σ φ(message_i))`) with width ≥ max-count** — injectable as a *single-layer* adapter, not gLSTM.
  (Grounding: DeepSets [1703.06114], GIN sum>mean [1810.00826], width≥N [1901.09006]; see References.)
- **Knob 1 — operation (sum vs normalized), now a clean causal ablation.** Matched harness, only the
  pool changes: on IID everything ties at 100%, but on count/length-OOD the normalized pools collapse
  (mean 0.76, softmax 0.76, identical curves → it is the Σ=1 constraint) while sum and PNA hold 1.00.
  This is the direct softmax-vs-sum baseline the project previously lacked, and it is the
  [GIN](https://arxiv.org/abs/1810.00826) sum>mean theorem ([DeepSets](https://arxiv.org/abs/1703.06114),
  `ρ(Σ φ)`) instantiated inside a frozen VLM.
- **Knob 2 — capacity (width), measured.** Sweeping d_mem on the sum readout (IID, counts 0–8) gives a
  monotonic curve that **saturates exactly at d_mem = 8 = max count N**, with failures concentrated at
  high counts. This is the [Wagstaff et al. 2019](https://arxiv.org/abs/1901.09006) set-representation
  **width ≥ N** bound measured directly (and aligns with [Di Giovanni et al. 2023](https://arxiv.org/abs/2302.02941),
  where width mitigates over-squashing); it refutes the tempting "a scalar count needs no width" intuition.
  Prescription: d_mem ≥ max expected count. Full n=3 grid (overall acc, IID counts 0–8):

  | d_mem | seed0 | seed1 | seed2 | mean |
  |------:|------:|------:|------:|-----:|
  | 1  | 0.53 | 0.64 | 0.49 | 0.55 |
  | 2  | 0.60 | 0.73 | 0.60 | 0.64 |
  | 4  | 0.78 | 0.89 | 0.81 | 0.83 |
  | 8  | 0.99 | 1.00 | 0.97 | **0.99** |
  | 16 | 1.00 | 1.00 | 1.00 | 1.00 |
  | 64 | 1.00 | 1.00 | 1.00 | 1.00 |
- **The deployable baseline is tiny.** A single mid-layer, one shared φ/ρ, injected once at the last token
  → 100% incl. count/length OOD. Inject site is irrelevant (carriers = last-token); weights are shareable.
- **PNA = sum (safe task-agnostic default).** On evidence-only [PNA](https://arxiv.org/abs/2004.05718)
  exactly matches sum (it contains sum via its degree-scaler×mean), so it is a no-regression generalization
  for tasks whose right aggregator is unknown.

### What isn't / dead ends
- **Distractors are the unsolved frontier.** With non-evidence frames present, learned adapters plateau:
  message-memory 46.7%, gated token mixer 51%, LoRA-attn 52.6%, answer-aligned codebook 60.7%. The
  **oracle pos/neg write-read upper bound is 96.3%**, and even oracle *selection*-then-sum is only 57% —
  so the gap is partly suppression *and* partly that distractors carry **negative evidence** that must be
  routed through a separate stream and read at later layers. Closing 60→96 without oracle masks is the
  next milestone.
- **Prompt / attention-bias / hand-crafted-summary interventions all failed** (frame labels, semantic
  carrier expansion, soft room bias, mean-pool frame summaries, fresh-query aggregation, native-attention
  factorization) — ≤ base. The carrier bottleneck is **not** about question-text capacity.
- **Oracle attention routing did not help** (cognn_oracle_routing, ~±2%): perfect routing to evidence
  keys isn't enough, confirming the bottleneck is upstream of routing.
- **Count extrapolation is a distinct, unsolved problem.** Train on counts 0–5, test on 7 → ~0% for every
  method; even all-counts-IID ceilings ~72%. Don't conflate "aggregation" wins with "count generalization".
- **The per-frame gating interface is the wrong abstraction for closing the distractor gap**.
  Every route through it fails for an understood reason: soft gates get bypassed by joint training (47%);
  a learned gate feeding a frozen exact-mask readout compounds errors (12–16%, below base); and
  training the readout to tolerate mask noise destroys the count signal (96→62 even at clean eval, since
  15%/frame ≈ count-label noise over 8 frames).
  - **Detection-ceiling measured:** even a *pure* per-frame detector (λ_ce=0, 10 ep) ceilings at
    **AUC ≈ 0.96, at layer 19** (w14–17 only reached 0.93 — the L14–17 write window every prior gate used
    was the wrong layer; evidence is most decodable at L19). This is higher than the earlier ~0.9 estimate
    but still short of the ~0.99/frame exact counting needs.
  - **Why per-frame gating can't scale (structural, not engineering):** exact count needs net error
    `#FP − #FN = 0`. Any per-frame *bias* is amplified ×n (linear in count); even unbiased, the net error
    is a random walk with √n spread, so exact-match probability falls with n. So hard per-frame gating
    degrades with count and is hopeless past ~10–12 regardless of detector quality — a better gate does not
    save it. The escape is **not** a better per-element gate but either (a) **count-level / distributed
    supervision** (calibrated `count = Σ P(evidence_i)`, where unbiased noise averages instead of
    compounding — though it too drifts at very high counts), or (b) **global/competitive selection** rather
    than independent per-element decisions (see Open questions).
- **Aggregator richness recovers *some* distractor signal (mild).** A clean PNA readout beats pure
  sum on distractors (iid 68.1% vs 63.8%, +4–6pp across splits), so the distractor gap is **not 100%
  selection** — distributional features (max/std of the frame messages) that pure sum discards carry real
  evidence-vs-distractor signal. But it's marginal (single-seed, still ~28pp below oracle), so it's a
  one-line nuance, not a research direction; the bulk of the gap remains selection.

### Honesty flags (verify before thesis)
- Several **near-100% numbers are trained-on-clean or oracle-masked** (`pnamix_clean` 97%,
  `layerwise_glstm` 98–99%, all `evidence_only` 100%, all `*oracle*` rows). They are upper bounds / clean
  conditions, **not** distractor-robust deployable results.
- For the gLSTM "full-model" variants (`layerwise_frame_message_glstm`), confirm exactly which params are
  trainable — "memory-disabled" collapses to ~58%, so the adapter is doing heavy lifting; check for any
  train/eval leakage on the synthetic splits.
- `memory_carrier_site_layer_norm_sweep` showed the earlier all-question injection edge **disappears
  after proper √|S| normalization** → some prior "gains" may have been answer-steering, not cleaner
  aggregation. Re-audit injection-site claims with normalization held fixed.
- **All 06-12→13 ablation numbers are single-seed.** The matched read-mechanism control (sum 76.2 vs gLSTM 77.8
  len-OOD) is a tie *within seed noise* — the "associative read is dispensable" claim should be confirmed
  with ≥3 seeds before it goes in the thesis. The softmax −9.8pp effect is larger and more likely robust,
  but also single-seed and at short train lengths (degradation, not collapse). Those gLSTM runs still
  carry the unresolved "memory-disabled ~88–95%" flag (LoRA does much of the work at short lengths).
- **The DeepSets ablation — two core curves now n=3 (2026-06-13), the rest still single-seed.** The two *headline*
  curves are confirmed at 3 seeds: pooling (sum/pna 1.00 vs mean/softmax 0.76, ±0.01) and the width sweep
  (knee at d_mem=8=N; per-seed d8 = 0.99/1.00/0.97). These are no longer single-seed. Still single-seed
  and to confirm before thesis: the **layer ablation** (esp. the single-layer L14/L16=1.00 vs L15/L17
  0.86–0.91 split — could be seed noise, don't claim a *specific* layer), shared-weights, and inject-site
  rows. Those runs use `--load-in-4bit` default per the runner; confirm consistent with other rows.

### Open questions / next experiments
1. ~~**gLSTM on the distractor task**~~ — **done:** gLSTM ties/loses to plain sum on distractors
   (61.7 vs 63.8 IID); associative read is dispensable.
2. ~~**Why oracle-selection-then-sum is only 57%**~~ — **answered:** it's the missing negative
   stream (≈+20pp) plus read depth (≈+5pp), not an injection/representation incompatibility.
3. **Count-level / distributed stream supervision** for distractors — the per-frame gating interface is
   falsified; supervise the *summed* pos/neg streams or the count directly instead of per-frame
   masks. This is now the headline open problem for closing 47→96.
4. **Global/competitive selection instead of independent per-element gates (the GNN-method direction).**
   The diagnosis says per-element-independent decisions are structurally fatal (errors compound ×n / √n),
   which rules out whole families and points at methods where selection is *joint/competitive*:
   - **Slot Attention (Locatello 2020) / Set-Transformer PMA (Lee 2019)** — competitive routing (attention
     normalized across slots, not frames) splits evidence vs distractor *jointly*; feed the evidence slot
     into the existing **unnormalized sum** readout (competition for selection, sum for counting). Strongest
     fit; empirically prefigured by the discrete carrier-slots result (k=8 → 82.8%). Top candidate.
   - **Signed / relational message passing (Signed-GCN, Derr 2018; R-GCN, Schlichtkrull 2018)** — the
     pos/neg two-stream that hits 96% oracle *is* signed message passing; open Q is learning the
     edge-type (evidence vs distractor) assignment without per-frame labels.
   - Pair either with #3 (count-level supervision). Rule out: rewiring (complete graph; oracle-routing
     null), plain GAT (reintroduces softmax normalization), more virtual-node/width tricks (done).
5. **Re-run the matched read-mechanism control with ≥3 seeds** to firm up sum-vs-associative-read (and
   ideally extend to longer train-length ranges where the softmax/normalization gap should widen). Note the
   memory-only control (no carrier-LoRA) interim hints gLSTM-memory > sum-memory on distractors once the
   LoRA isn't masking it — confirm when those runs finish.
6. **Decouple count generalization** from aggregation (the 0% count-OOD result) — possibly a count-direction
   /codebook that extrapolates beyond trained counts; note any per-frame *or* soft-sum approach also drifts
   at very high counts, so a true high-count solution may need a non-accumulation mechanism.

### Interesting insights & surprises
- **Hierarchical slicing → ~98–100%**: just processing frames in small chunks and summing nearly solves
  the task, the single most striking "the bottleneck is simultaneity" demonstration.
- **Binary per-frame oracle hint → 79% at seq8** (from 28%): the model can count if told per-frame yes/no.
- **Additive ≫ joint**: the model's evidence integration is genuinely a *sum*, which is why
  sum/PNA-mean style readouts and the gLSTM additive memory work and joint-MLP readouts don't.
- **Margin collapses at k=2–4 frames regardless of frame type** — even adding *true* evidence past ~2
  frames hurts the margin, which is the clean over-squashing signature.

---

---
## Method comparison

| Method / family | Best metric | Setting | Verdict | Evidence (dirs) |
|-----------------|-------------|---------|---------|-----------------|
| Frozen baseline | seq8 ~19–30% | as-is | reference | `output_less_less_less_old/mmred_new_accuracy_all_uniform`, `outputs_oh_man/qwen7b_accuracy_heatmap` |
| Prompt / semantic carrier / soft-bias | ≤ base | 32B | ❌ dead end | `outputs_kitkat/mmred_{frame_carrier_prompt,semantic_carrier_expansion,frame_summary_routing}`, `entity_room_soft_bias` |
| Oracle attention routing | ±2% | 32B | ❌ (routing not the bottleneck) | `outputs_least_oldest/cognn_oracle_routing*` |
| Compact context / hierarchical slicing | seq8 → 98–100% | oracle-ish | ✅ (proves bottleneck) | `outputs_kitkat/mmred_hierarchical_slicing*`, `outputs_least_oldest/selective_routing_oracle` |
| Oracle count / per-frame hints | 79–100% | oracle | ✅ upper bound | `outputs_kitkat/mmred_oracle_frame_hints`, `outputs_oreo/translator_*`, `oracle_count_*` |
| LoRA + carrier mixing (maxmix/slots) | clean high-count 82–97% | trained-clean | ✅ (clean only) | `outputs/pnamix_clean_aggregation_lora`, `visual_fixed8_iid_carrier_slots_lora` |
| Memory adapter / count-direction | 45–61% | distractor task | ⚠️ partial | `outputs_oreo/{shared_count_direction*,answer_aligned*}`, `outputs_no_train/message_memory_carrier_update` |
| Learned token mixers (GTM/PNA/LoRA-attn) | 50–53% | distractor task | ⚠️ partial | `outputs_no_train/{gated_token_mixer_adapter,pna_gated_token_mixer_adapter}` |
| **Evidence-only sum/layer-local/raw-matrix** | **100%** (incl. **100% OOD**: train 1–4 → eval 5–8) | evidence-only | ✅ solved + extrapolates | `outputs_no_train/evidence_only_*`, `outputs/evidence_only_sum_adapter_train14_eval58_7b` |
| **gLSTM memory adapter** | IID 98–99% / length-OOD 88% / hi-extrap 70.5% | clean splits | ✅ **leading** | `outputs/{final_glstm_aggregation_comparison,layerwise_*glstm*}`, `outputs_kitkat/glstm_memory_adapter_7b_seq8` |
| Oracle pos/neg write-read (distractor) | **96.3%** | oracle mask | 📊 target upper bound | `outputs_no_train/distractor_oracle_posneg_write_read_adapter_seq8_7b` |
| Read-mechanism control: sum vs associative vs softmax (matched) | len-OOD: sum 76.2 / assoc 77.8 / **softmax 68.0** | neutral, train 1–4→5–8 | ✅ unnormalized necessary; assoc dispensable (single-seed) | `outputs/layerwise_glstm_train14_ood58_7b/20260612_155614_*`, `outputs/layerwise_frame_message_glstm/{20260612_185319_sum_ctrl_*,20260612_175221_softmax_ctrl_*}` |
| Sum vs gLSTM memory on **distractors** | sum **63.8%** IID ≥ gLSTM 61.7% | distractor task | ✅ sum = new best learned distractor method | `outputs/layerwise_frame_message_glstm/{20260612_175227_distractor_sum_*,20260612_175225_distractor_glstm_*}` |
| Learned posneg gate (all variants) | 39–47% (invariant to gate AUC) | distractor task | ⚠️ gap is optimization, not detection | `outputs/distractor_posneg_write_read_adapter_seq8_7b/learned_posneg_*` |
| Two-stage (frozen oracle readout + learned gate) | 12–16% (below base) | distractor task | ❌ per-frame interface compounds errors ≈pⁿ | `outputs/distractor_posneg_write_read_adapter_seq8_7b/learned_posneg_frozen*` |
| **Pooling ablation: sum/pna vs mean/softmax** | OOD: sum/pna **1.00** vs mean/softmax 0.76 | evidence-only, matched | ✅ normalization is the causal failure | `outputs/evidence_only_sum_evidence_adapter_seq1_8_7b/20260613_141248_{sum,mean,softmax,pna}_L14_17` |
| **DeepSets width (d_mem) sweep** | saturates at **d_mem=8=N**; d2 0.60 / d4 0.78 / d8 0.99 | evidence-only IID 0–8 | ✅ width≥max-count bound measured | `outputs/evidence_only_sum_evidence_adapter_seq1_8_7b/2026*_dmem*_sum_L14_17_iid` |
| **Minimal DeepSets baseline** (1 layer / shared φ,ρ / last-token) | **100%** IID + OOD | evidence-only | ✅ cleanest deployable fix | `outputs/evidence_only_sum_evidence_adapter_seq1_8_7b/20260613_141248_sum_{L14,L16,L14_17_shared}` |

---

---
## Experiment Log (appendix — full run-by-run record)

> Append-only. One row per experiment family (tight group of runs). Dirs are relative to repo root.
> Status: ✅ done & trusted · ⚠️ done but suspect/partial · ❌ failed/no-gain · 📊 characterization/probe only · ▶ running

### Feb–Apr 2026 — Locating the bottleneck (32B/7B)

| Date | Output dir | Method / change | Key config | Metric | Status | Notes |
|------|-----------|-----------------|-----------|--------|--------|-------|
| 2026-03 | `output_old/seq_len_*/LD_*`, `find_af1_transition` | Layer-decomposition (CAMA/AF1-style) of per-layer info contribution | 32B/7B, seq 2/4/8/16 | — | 📊 | Info concentrates through mid layers (~10–27) into last token; ~0 contribution past L32. Bottleneck is seq-length-agnostic. |
| 2026-03 | `output_less_old/transfer_bottleneck_scaling/*` | Frame-token rescue/restoration curves | 32B, seq 2/4/8 | max_rescue ~0.99 (seq2 success) | ✅/⚠️ | >98% of signal is recoverable from the last token via late layers; failures show *misrouting*, not loss. |
| 2026-04-08 | `output_less_less_less_old/mmred_new_accuracy_all_uniform` | Baseline accuracy by seq_len | 7B, all_uniform, 900 samp | seq2 62.7% / seq4 42.3% / seq8 19.3% | 📊 | Canonical baseline degradation curve. |
| 2026-04-09 | `output_less_less_less_old/seq_len_*/unified_bottleneck_analysis` | Layer-wise clean-ablation + restoration | 7B, seq 2/4/8 | norm. damage peaks L10–27 | ✅ | Evidence frames carry more causal damage (~0.36) than question tokens (~0.23); recoverable → routing, not capacity. |
| 2026-04-11→16 | `outputs_least_oldest/mmred_accuracy_all_uniform`, `mmred_basic_acc_mosaic` | Baseline accuracy mosaics | 7B/3B, seq 2/4/8/9 | seq2 85% → seq8 ~29% | 📊 | Confirms steep seq-length sensitivity. |
| 2026-04-14 | `outputs_least_oldest/mmred_image_size_sweep` | Image-resolution sweep | seq 2/4/8, 66–504 px | seq8 11%→18% (66→504px) | ✅ | Resolution helps a little at low seq, negligibly at seq8 → vision detail is *not* the bottleneck. |
| 2026-04-18 | `outputs_least_oldest/mmred_frame_causal_separation_heatmaps` | Per-frame causal ablation | 32B, seq 2/4/8 | evidence drop 6.5(seq2)→0.64(seq8) @count1 | ✅ | Individual frame influence vanishes at seq8 → aggregation, not per-frame signal. |
| 2026-04-19 | `outputs_least_oldest/mmred_non_evidence_empty_context` | Remove non-evidence frames | 32B, seq 2/4/8 | ~neutral | ⚠️ | Removing distractors alone doesn't fix it — bottleneck is in aggregating evidence. |
| 2026-04-25 | `outputs_least_oldest/stage34_count_probe[_all]` | Linear/MLP probe of count by layer | 32B, seq 2/4/8, L0–63 | probe seq2 95.6% / seq8 40.7% (vs model 30.4%) | ✅ | **Key:** count info is decodable (probe ≫ model); failure is in the model *using* it. Info stabilizes after L40. |
| 2026-04-25 | `outputs_least_oldest/last_token_rep_collapse_evidence1` | Last-token representation collapse metrics | 32B, seq 1–8 | cosine-dist 0.060→0.019 (seq1→8) | ✅ | Last-token reps homogenize as seq grows — mechanistic signature of over-squashing. |
| 2026-04-25 | `outputs_least_oldest/mmred_evidence_index_attention`, `_sweep`, `evidence_count_seq_len_heatmap` | Attention to evidence vs non-evidence by layer | 32B | question-tokens > last-token evidence focus mid-layers | 📊 | Last token under-attends evidence; carrier (question) tokens hold more. Motivates "carrier" framing. |
| 2026-04-26 | `outputs_least_oldest/wrong_routing_decoys` | Semantic decoys (swapped char–room bindings) | 32B, seq8, L24/48 | decoy +29 pp swing | ✅/❌ | Model follows spurious char–room associations rather than robustly counting — a real failure mode. |
| 2026-04-29 | `outputs_least_oldest/stage34_nonlinear_probe_control` | Linear vs MLP/KNN probes | 32B | nonlinear +3–5pp early layers only | ⚠️ | Bottleneck isn't probe complexity; nonlinearity only matters in early layers. |
| 2026-04-29 | `outputs_least_oldest/carrier_capacity_probe` | Can carrier tokens encode count? | 32B, seq 1–8, slot prompts | probe≈model+~10pp; gap grows with seq | ✅/📊 | Carriers retain count info but the model's *use* degrades with length. |
| 2026-04-29→30 | `outputs_least_oldest/cognn_oracle_routing[_new*]` | Oracle: question attends ONLY to evidence keys | 32B, seq 2/4/8, T=30–42 | ~−2% to +2% vs base | ⚠️/❌ | **Perfect routing does NOT help** → bottleneck is upstream of routing (encoding/recognizing evidence). |
| 2026-04-29 | `outputs_least_oldest/selective_routing_oracle` | Keep only evidence frames (compact context) | 32B, seq 2/4/8 | seq8 28.7%→59.7% (compact) | ✅ | Shrinking the context to evidence frames recovers a lot → dilution/over-squashing from many frames. |
| 2026-05-08 | `outputs_least_oldest/evidence_count_ablation` | Frame influence + entropy by count | 32B, seq8 | total influence 0.48(c1)→1.38(c8) | ✅ | Each frame's signal is weak; aggregate grows but model can't sum many weak signals. |

### May 2026 — Probing the frame→carrier→last pathway (7B)

| Date | Output dir | Method / change | Key config | Metric | Status | Notes |
|------|-----------|-----------------|-----------|--------|--------|-------|
| 2026-05-02→03 | `outputs_kitkat/last_question_text_mixed_counts_T30`, `mmred_frame_carrier_prompt`, `mmred_semantic_carrier_expansion*` | Prompt-level carrier interventions (frame labels, semantic expansion, late question text) | 32B, seq 2–8, T=30 | all ≤ base (e.g. seq8 28.1%, no gain) | ❌ | Prompt/text manipulation of the carrier doesn't help; late question-text injection biases toward predict-1. Bottleneck isn't carrier text capacity. |
| 2026-05-03 | `outputs_kitkat/entity_room_soft_bias_T30` | Soft visual attention bias to target room | 32B, seq 2–8 | marginal/negative | ⚠️ | Hard attention bias alone insufficient. |
| 2026-05-03 | `outputs_kitkat/mmred_additivity_saturation` | Are token-group reps additive vs saturating? | 32B, seq 4/8, L36–63 | additivity ratio ≈1.0, residual ≈0 | 📊 | Reps combine **additively, no saturation** → motivates additive (sum) aggregation design. |
| 2026-05-05 | `outputs_kitkat/mmred_oracle_frame_hints`, `outputs_best/oracle*` | Oracle/binary frame hints in prompt | 32B, seq 1–8 | seq8 27.8% → 79.2% (binary yes/no hint) | ✅(oracle) | If you tell the model per-frame yes/no, counting is near-solved → the bottleneck is **extracting per-frame evidence**, not final summation. |
| 2026-05-06 | `outputs_kitkat/mmred_bottleneck_by_evidence_count*` | Token-group causal importance by count | 32B, seq 1/2/4/8, L30–62 | — | 📊 | Evidence frames + last token most critical at high counts. |
| 2026-05-08 | `outputs_kitkat/outputs_best/mmred_nested_evidence_growth` | Add evidence frames k=1→16, track margin | 32B, L40/63 | acc 100%(k1)→57%(k2)→12%(k8); margin +6.7→−2.2 | 📊 | **Margin collapses by k=2–4** even for *genuine* evidence — the core failure curve. |
| 2026-05-08 | `outputs_kitkat/outputs_best/mmred_nested_distractor_drift` | Add distractor frames k=1→16 | 32B, L40/63 | acc 100%→25%(k8); margin +6.4→−1.0 | 📊 | Distractors collapse margin similarly fast; mid-layer representation drift (r≈0.5–0.7 @L40). |
| 2026-05-10 | `outputs_kitkat/mmred_hierarchical_slicing_seq8_park_room` | Decompose 8 frames into chunks, count, sum | 32B, seq8 | flat 28.1% → slice_2 75.6% → slice_1 97.8% → gold-text 100% | ✅ | **Strongest signal:** per-frame decomposition ≈ solves it. Bottleneck = simultaneous aggregation, not the count itself. |
| 2026-05-10→11 | `outputs_kitkat/mmred_frame_summary_routing_seq8_park_room` | Mean-pool each frame into first token | 32B, seq8, L32/40/48 | ≤ base; suppress variant 14.8% | ❌ | Hand-crafted frame summaries don't help (too lossy / slot saturated). |
| 2026-05-12 | `outputs_kitkat/attention_rollout_group_heatmaps_seq8*` | Attention-rollout maps by token group | 32B, seq8 | — | 📊 | Baseline attention-flow maps for interpreting interventions. |
| 2026-05-15→19 | `outputs_oh_man/chefer_*` (frame patch / question-token / second-hop / step-marker) | Chefer relevance attribution of frames→tokens | 7B, seq 2–8, L12–24 | evidence relevance peaks L14–17; question tokens 2–5% of total relevance | 📊/⚠️ | Evidence relevance is real but **suppressed** between vision and language; step markers help ~5–7pp. |
| 2026-05-15 | `outputs_oh_man/qwen7b_accuracy_heatmap` | 7B baseline accuracy grid | 7B, seq 2/4/8 × count 0–8 | degrades with seq & count | 📊 | The 7B reference surface the interventions target. |
| 2026-05-17 | `outputs_oh_man/perm_bias_seq8` | Frame-order permutation sensitivity | 7B, seq8 | count2 −3pp, count3 −11pp, count4 −7pp | ✅ | Order matters more at higher counts → recency/order-dependent (not order-invariant) aggregation. |
| 2026-05-18→19 | `outputs_oh_man/evidence_to_carrier_routing*`, `two_stage_question_answer_routing*`, `additive_evidence_aggregation*` | Attention-logit boosting of evidence→carrier→answer | 7B, seq8, L14–17/20–21, γ=1–4 | best ~7.3% (stage A+B); additive interventions 0% | ⚠️/❌ | Direct attention boosting yields only small gains; raw additive boosts fail → problem is post-routing, at answer generation. |
| 2026-05-21 | `outputs_oh_man/frame_to_carrier_evidence_sum_probe_seq8_7b_20260521_164621` | Learn per-frame binary evidence, sum to count | 7B, seq8, L14–17, d256, 40 ep | target char×room MLP **88.1%** (MAE 0.12) | ✅ | **Reused downstream as a "source run".** Evidence is linearly present and additively integrable at carriers. |
| 2026-05-21 | `outputs_oh_man/frame_to_carrier_message_memory_probe_*` (multilayer/linear/mlp) | Detect frame→carrier "message memory" by layer | 7B, seq8, L14–17 | actual-message MLP up to 57.8% count acc | ✅ | Message-memory > content-only > raw; MLP needed. Multilayer run reused as a source run. |
| 2026-05-21 | `outputs_oh_man/message_probe_sum_vs_joint_mlp_*`, `outputs_oreo/message_probe_base_vs_per_frame_*` | Per-frame **sum** vs joint MLP readout | 7B, seq8, counts 0–8 | per-frame sum **63.7%** vs joint MLP ~22%; binary per-frame **93%** | ✅ | **Aggregation is additive, not joint** — sum of per-frame signals wins decisively. Per-frame evidence ≈93% recoverable vs 16% base. |

### Late May 2026 — Memory-adapter / count-direction line (7B)

| Date | Output dir | Method / change | Key config | Metric (overall acc) | Status | Notes |
|------|-----------|-----------------|-----------|--------|--------|-------|
| 2026-05-26 | `outputs_kitkat/glstm_memory_adapter_7b_seq8/20260526_123006` | Additive **gLSTM memory adapter** on frozen Qwen (memory_q vs memory_rc) | 7B, seq8, L14–17, d_mem 64, lr 3e-4, 3 ep | base 25.2% → **memory_q 31.1%** (memory_rc 9.6%, breaks) | ⚠️ | First trainable adapter; memory_q helps at extreme counts (0→86.7%, 8→93.3%) but breaks mid-counts (break_rate 56%). |
| 2026-05-26 | `outputs_oreo/message_memory_adapter_stage1_stage3_seq8_7b_*` | Stage1 count head on messages → Stage3 inject to carriers | 7B, seq8, L14–21 | base 24.4% → Stage3 **45.9%** | ✅ | Per-frame message sum probe 68%; Stage3 injection roughly doubles base. |
| 2026-05-27 | `outputs_oreo/shared_count_direction_memory_seq8_7b_20260527_203756` | Learn a shared **count-direction** vector + small residual | 7B, seq8, L14–21 | base 45% → **54.8%** (shared+residual) | ✅ | **Stage3 checkpoint reused downstream.** Pure direction alone fails (11%); residual essential. Count corr 0.996. |
| 2026-05-27 | `outputs_oreo/shared_count_direction_calibrated_seq8_7b_*` | Add calibration (λ_count, λ_res) | 7B, seq8 | **48.9%** overall, mid-count 43.3% | ✅ | Calibration controls residual norm (48→9.6) and lifts mid-counts +10pp; corr preserved 0.948. |
| 2026-05-27 | `outputs_oreo/message_memory_count_channel_ablation_seq8_7b_*` | memory-only vs gate-sum-only vs combined readout | 7B, seq8 | gate-sum corr 0.70 (stage1); memory-only best stage3 mid | ⚠️ | Different channels win at different stages; combining doesn't help stage1. |
| 2026-05-28 | `outputs_oreo/memory_injection_site_sweep_seq8_7b_20260528_000508` | Sweep injection site × layer window (on the 54.8% ckpt) | 7B, seq8, L14–24 | best **56.3%** (all_question, L18–21 multi) | ✅ | Marginal but consistent: multi-layer L18–21 > single; all_question ≳ room_char; last-token-only caps at 45.9%. |
| 2026-05-28 | `outputs_oreo/memory_carrier_site_layer_norm_sweep_seq8_7b_*` | Raw α vs √|S| token-count normalization | 7B, seq8, L12–21 | room_char L16–19 **28.2%** | ⚠️ | After proper normalization the all_question advantage disappears → earlier gain was answer-steering, not cleaner aggregation. **Honesty flag.** |
| 2026-05-28 | `outputs_oreo/evidence_gate_to_codebook_seq8_7b_20260528_204057` | Sigmoid evidence gate → gated count codebook | 7B, seq8, L14–21 | gated best 26.7% vs **oracle codebook 48.9%**, base 24.4% | ❌ | Evidence detection AUC only 0.687 → learned gating can't beat oracle; detection is the weak link here. |
| 2026-05-28 | `outputs_oreo/answer_aligned_count_codebook_memory_seq8_7b_20260528_000551` (also in `outputs_no_train`) | Count-aware codebook aligned to answer tokens | 7B, seq8, L14–17, inject L18, 3 ep | base 24.4% → **60.7%** (Qwen-init, τ=2) | ⚠️ | Best learned distractor-task adapter so far; per-layer count score corr 0.94. Still ~35pp below oracle posneg. |
| 2026-05-28→29 | `outputs_oreo/oracle_count_multilayer_injection`, `oracle_count_injection_site_sweep_seq8_7b` | Inject **oracle gold count** into residuals | 7B, seq8, L14–24 | (ceiling probe) | 📊 | Designed to measure the upper bound of count injection. |
| 2026-05-29 | `outputs_oreo/translator_*` (codebook / layer-suffix / gold-count ablation) | Inject oracle count via codebook vs learned translator | 7B, seq8, L14–17…17–17 | static/energy-norm L14–17 **100%**; state-conditioned 51.9% | ✅(oracle) | Oracle count injection → 100% (model *can* use a clean count). Static/low-rank translators work; learned state-conditioned translators are fragile. Single late layer insufficient. |

### Late May – early Jun 2026 — Evidence-only counting solved; distractor frontier (7B)

| Date | Output dir | Method / change | Key config | Metric | Status | Notes |
|------|-----------|-----------------|-----------|--------|--------|-------|
| 2026-05-29 | `outputs_no_train/evidence_only_layer_local_seq1_8_7b` | Layer-local frame aggregation, **all frames are evidence** | 7B, seq 1–8, L14–17, 1 ep | base 39.2% → **100%** (high counts 20%→100%) | ✅ | Pure-aggregation task is **solved**; MAE 1.26→0.00. |
| 2026-05-29 | `outputs_no_train/evidence_only_all_question_to_last_seq1_8_7b` | Raw-matrix (QK) readout, last token reads all-question messages | 7B, seq 1–8, L14–17 | base 39.2% → **100%** | ✅ | Content-addressed mixing also fully solves evidence-only. |
| 2026-05-30 | `outputs_no_train/evidence_only_sum_evidence_adapter_seq1_8_7b` (`experiments/...py`) | Pure additive sum of frame messages → last token | 7B, seq 1–8, L14–17, d256 | base 39.2% → **100%** | ✅ | No gates/queries needed when every frame is evidence — confirms additive hypothesis. |
| 2026-05-30 | `outputs_no_train/distractor_oracle_mask_sum_adapter_seq8_7b` | Oracle mask selects evidence, then sum (with distractors) | 7B, seq8, L14–17 | base 24.4% → **57.0%** | ⚠️📊 | **Surprise:** perfect selection + sum only reaches 57% → injection/representation compat is *also* limiting, not just selection. |
| 2026-05-30 | `outputs_no_train/distractor_oracle_posneg_write_read_adapter_seq8_7b` | Oracle pos/neg streams, write L14–17, **read L20–27** | 7B, seq8, 5 ep | base 24.4% → **96.3%** | ✅📊 | Upper bound: distractors carry essential *negative* signal; separate streams + delayed read nearly recover evidence-only. The 57→96 gap = stream separation + late read. |
| 2026-05-30 | `outputs_no_train/distractor_oracle_posneg_{sum,concat}_adapter_seq8_7b` | Oracle pos/neg sum vs concat variants | 7B, seq8 | (variants of above) | 📊 | Ablations of the pos/neg upper-bound mechanism. |
| 2026-05-30 | `outputs_no_train/distractor_supervised_gated_sum_adapter_seq8_7b` | Supervised (non-oracle) gated sum | 7B, seq8 | — | ⚠️ | Learned version of the oracle gate; the gap to 96% is the open problem. |
| 2026-05-29 | `outputs_no_train/message_memory_carrier_update_seq8_7b` | Layer-local vs cumulative-sum memory update (distractor task) | 7B, seq8, L14–17, d256 | base 24.4% → layer-local **46.7%**, cumulative 45.2% | ⚠️ | Helps, but gate-sum barely correlates with evidence count (0.04–0.16) → mixing is noisy. |
| 2026-06-02 | `outputs_no_train/gated_token_mixer_adapter` (`experiments/gated_token_mixer_adapter.py`) | MLP vs LoRA-attn vs gated token mixer (distractor) | 7B, seq8, L14–17, 3 ep | base 24.4% → LoRA-attn **52.6%**, GTM 51.1%, MLP 38.5% | ⚠️ | Learned mixers plateau ~50% on distractors (high-count ~48–53%); far from oracle 96%. |
| 2026-06-03 | `outputs_no_train/pna_gated_token_mixer_adapter`, `evidence_only_pna_gated_token_mixer_adapter` | PNA-style (multi-aggregator) gated mixers | 7B, seq8 | softmax 40%, sigmoid-gated-sum 42.2% | ⚠️ | PNA variants underperform LoRA-attn; aggregator choice matters but doesn't close the gap. |
| 2026-06-05 | `outputs/pna_carrier_mixing_lora[_diagnostics]`, `pnamix_layer_sweep_lora` | LoRA + carrier mixing (maxmix/pnamix), layer sweep | 7B, 12 LoRA layers | base→ maxmix high_count +5.5pp, long +6pp; best layers L14–17 | ✅/⚠️ | maxmix ≳ pnamix; **early-layer (14–17) injection is most effective**; alpha·v gating best on high counts. |
| 2026-06-06 | `outputs/pnamix_clean_aggregation_lora` | maxmix vs pnamix on **clean** data (no distractors) | 7B, LoRA 12 layers, 2400 train | maxmix high_count **97.3%**, long **95.5%** vs LoRA base 64.8%/77.8% | ✅ | On clean data maxmix nearly solves high-count & length OOD. (Clean-trained — not distractor-robust.) |
| 2026-06-06 | `outputs/visual_fixed8_iid_carrier_slots_lora` | Discrete carrier **slots** (k=1/4/8) | 7B, L14–17, fixed-8 | base 69.4% → **k8 82.8%** (high-count 52.5%→81.3%) | ✅ | Explicit slot memory beats continuous mixing; more slots → better high-count. |
| 2026-06-06 | `outputs/visual_fixed8_count_sweep_lora[_label_control]`, `visual_fixed8_iid_all_counts_lora` | LoRA/maxmix/pnamix under true count-OOD vs all-counts-IID | 7B, fixed-8 | OOD high-count ~0%; all-counts-IID plateau ~72% | ⚠️ | **Honesty flag:** true count-extrapolation (train 0–5, test 7) fails ~0% for all methods; even all-counts-IID ceilings ~72%. Count generalization is a separate hard problem from routing. |
| 2026-06-07 | `outputs/frame_sigmoid_sum_attention_patch` | Frame sigmoid-sum attention patch + LoRA | 7B, L14–17 | iid 37.8%→70.0% (patch+LoRA) | ✅/⚠️ | Sigmoid-sum aggregation helps IID; doesn't fix high-count OOD. |
| 2026-06-07→09 | `outputs/layerwise_frame_message_glstm`, `layerwise_glstm_mechanism_ablation` | gLSTM mechanism ablation: persistent vs fresh, layerwise vs final-only | 7B, L14–17, d_mem 64, 3.3M params | layerwise-persistent **98.1% IID / 88.1% length-OOD** vs direct-sum 94.5/73.9 | ✅ | Cross-layer **persistent state +18.7pp length-OOD**; layerwise injection +~1pp IID. ⚠️ verify what's frozen — "memory-disabled" drops to ~58%, so a lot rides on the adapter. |
| 2026-06-08 | `outputs/layerwise_fresh_query_aggregation_ablation`, `native_frame_factorized_attention` | Fresh-query / native-attention factorization | 7B, smoke | ~0% | ❌ | Generating fresh queries / factorizing native attention fails; explicit messaging or memory is needed. |
| 2026-06-09→10 | `outputs/final_glstm_aggregation_comparison` | **Head-to-head:** gLSTM (final-only) vs sum vs LoRA baseline | 7B, L14–17, lr 1e-4, 3 ep | gLSTM **70.5%** high-aggregation-extrap vs sum 46.0% / LoRA 48.5%; IID 98.9% | ✅ | **Current leading comparison:** gLSTM memory > sum > LoRA on the hard extrapolation split. |
| 2026-06-12 | `outputs/evidence_only_sum_adapter_train14_eval58_7b/{20260612_160218,20260612_161616}` | Sum adapter **OOD extrapolation**: train seq 1–4 only, eval seq 5–8 (new `--train-seq-lens` split; OOD test n=100/seq) | 7B, evidence-only, L14–17, d256, lr 1e-4, 1 vs 3 ep | 1 ep (val 96.7%): OOD 60–96%, undercounts; **3 ep (val 100%): OOD 100%** (460/460, mean pred = gold exactly) | ✅ | **Perfect length+count extrapolation** once converged — counts 5–8 never seen as labels. 1-ep degradation was undertraining, not an extrapolation limit. Diagnostics clean (frozen Qwen, exact messages, disjoint splits). Caveat: in evidence-only, count ≡ seq_len (axes confounded); jobs 93447/93452. |

### Week of 06-10→16 — Aggregator-ingredient decomposition + distractor-gap mechanism (06-12→13, 7B)

> All single-seed. Three first-submission gLSTM jobs (93513–15) ran ~10 min with mangled `TRAIN_LENGTHS`
> (sbatch `--export` comma-splitting); cancelled, dirs marked `ABORTED.md`, resubmitted correctly. The
> new posneg script (`experiments/distractor/distractor_posneg_write_read_adapter_seq8_7b.py`) was
> recovered from an orphaned git blob (`2e75c2c`) and extended (gate mode/source/hardness, stream
> ablation, frozen/oracle-init readout, mask noise, gate-AUC diagnostics).

| Date | Output dir | Method / change | Key config | Metric | Status | Notes |
|------|-----------|-----------------|-----------|--------|--------|-------|
| 2026-06-12 | `outputs/layerwise_glstm_train14_ood58_7b/20260612_155614_carrier_glstm_layerwise`; `outputs/layerwise_frame_message_glstm/{20260612_185319_sum_ctrl_*,20260612_175221_softmax_ctrl_*}` | **Read-mechanism control** (matched harness/data): unnormalized associative (gLSTM) vs plain sum vs softmax-normalized read. Train seq 1–4 → length-OOD 5–8, neutral fillers | 7B, L14–17, d_mem 64, 3 ep | IID: gLSTM 1.000 / sum 0.982 / softmax 0.993. **len-OOD: gLSTM 0.778 / sum 0.762 / softmax 0.680** (len8 .583/.633/.517) | ✅ | **Normalization is the causal variable.** Two unnormalized reads tie (sum even wins at len8); softmax −9.8pp len-OOD, IID unchanged → predicted over-squashing signature. Associative q·k addressing adds nothing over sum. Caveat: degradation not collapse at short train lengths; memory-disabled 88.6/94.3/94.6%. |
| 2026-06-12 | `outputs/layerwise_frame_message_glstm/{20260612_175227_distractor_sum_carrier_direct_sum,20260612_175225_distractor_glstm_carrier_glstm_layerwise}` | **gLSTM vs sum read on distractor fillers** (new `--filler-kind distractor`). Train seq 4,6,8 → OOD 5,7,10 | 7B, L14–17, d_mem 64, 3 ep | sum: IID **63.8%** / len-OOD 60.0% / comp-OOD 59.5%. gLSTM: IID 61.7% / 58.6% / 57.6% | ✅ | Sum ties/beats associative read **with distractors too** → q·k addressing dispensable even under selection pressure. Sum's 63.8% IID > all prior learned distractor methods (codebook 60.7%, LoRA-attn 52.6%). U-shaped by-count (extremes easy, mid ~50%). Still ~32pp below oracle 96.3% → gap is signed-stream separation, not aggregation capacity. |
| 2026-06-12 | `outputs/distractor_posneg_write_read_adapter_seq8_7b/{oracle_pos_only_w14_17_r20_27_5ep,oracle_posneg_w14_17_r18_19_5ep,oracle_posneg_noise015_w14_17_r20_27_5ep}` | **Oracle posneg decomposition**: pos-only late-read; posneg early-read (L18–19); posneg readout trained with ε=0.15 mask-flip noise | 7B, seq8, base 24.4%, 5 ep | pos-only late **74.8%**; posneg early-read **91.1%**; noise-robust readout **61.5%** (eval w/ clean mask) | ✅📊 | Decomposes the 96.3% bound: **negative stream ≈ +20pp (dominant)**, read depth ≈ +5pp. Neg stream fixes low counts (cnt0 60→100, cnt1 7→93) = explicit absence signal. Noise training craters 96→62 even at clean eval → 15%/frame ≈ count-label noise (corrupts ~73% of 8-frame samples). |
| 2026-06-12→13 | `outputs/distractor_posneg_write_read_adapter_seq8_7b/{learned_posneg_w14_17_r20_27_5ep,...10ep,learned_posneg_w18_21_r22_27_5ep,learned_posneg_lategate_w14_17_r20_27_5ep,learned_posneg_hardgate_w14_17_r20_27_5ep}` | **Learned-gate plateau** (sigmoid gate + aux mask BCE/count loss): epochs, gate layer (14–17 vs 18–21), late-gate-on-early-messages, straight-through hard gate | 7B, seq8, base 24.4%, 5–10 ep | **39–47% regardless of gate AUC (0.61↔0.91)**: 46.7/47.4/39.3/43.0/46.7%; gate AUC 0.81/0.84/0.87/0.87/0.61 | ⚠️ | **Accuracy invariant to gate quality** → learned-vs-oracle gap is *not* detection. Per-sample: |gate-count-err| uncorrelated with |pred-err| (r=−0.18). 46.7% = old gateless adapter score → joint training collapses to ungated-aggregate optimum; gate is decorative. Detection/hardness/epochs/layer all falsified. |
| 2026-06-13 | `outputs/distractor_posneg_write_read_adapter_seq8_7b/{learned_posneg_frozenreadout_w14_17_r20_27_5ep,learned_posneg_frozenro_hardgate_noce_5ep}` | **Two-stage** (frozen oracle-trained 96.3% readout, train gate only): soft gate; hard gate + λ_ce=0 (pure detector → hard mask at interface) | 7B, seq8, base 24.4%, 5 ep, `--init-streams-from`/`--freeze-streams` | soft **12.6%**, hard-detector **16.3%** (both **below base**) | ❌ | Frozen exact-mask readout is brittle off the binary manifold: soft α≈0.5 halves streams → decodes "≈4"; hard mask compounds errors ≈ p⁸ (0.8⁸≈0.17). **Per-frame gating interface is the wrong abstraction** for closing 47→96 — needs ≥0.99/frame detection (unreached) or count-level stream supervision. |

### Week of 06-10→16 — DeepSets baseline: the two causal knobs (06-13, 7B)

> The aggregation fix reframed as **DeepSets** (`ρ(Σ φ(message_i))`). New flags on
> `experiments/evidence_only/evidence_only_sum_evidence_adapter_seq1_8_7b.py`: `--pool {sum,mean,softmax,pna}`,
> `--share-weights`; `CARRIER_PNA` read variant added to the gLSTM harness. Runner
> `runners/evidence_only_sum_ablation.sbatch`. All evidence-only, train seq 1–4 → eval OOD 5–8, **single-seed**.

| Date | Output dir | Method / change | Key config | Metric | Status | Notes |
|------|-----------|-----------------|-----------|--------|--------|-------|
| 2026-06-13 | `outputs/evidence_only_sum_evidence_adapter_seq1_8_7b/20260613_141248_{sum,mean,softmax,pna}_L14_17` | **Pooling ablation** (matched harness/data, only the aggregator changes): sum vs mean vs softmax vs PNA readout | 7B, L14–17, d256, 3 ep | **IID all 1.00**; OOD: sum **1.00** / pna **1.00** / mean **0.76** / softmax **0.76** (s5–8 mean 1.00/0.85/0.61/0.57) | ✅ | **Normalization is the causal variable.** IID hides it (all tie); count/length OOD exposes it. mean≈softmax exactly → it's the Σ=1 constraint, not the weighting. PNA=sum (its degree-scaler×mean reproduces sum). = GIN sum>mean inside a frozen VLM, as a causal ablation. The direct softmax-vs-sum baseline the project previously lacked. |
| 2026-06-13 | `outputs/evidence_only_sum_evidence_adapter_seq1_8_7b/20260613_141248_sum_{L14,L15,L16,L17,L14_15,L15_16,L16_17,L15_17,L14_17,L14_17_shared,L14_17_carriers}` | **Cleanest-baseline ablation**: layer window (singles/pairs/triples), shared vs per-layer φ/ρ, inject at last-token vs carriers | 7B, sum, d256, 3 ep | single L14 **1.00** / L16 **1.00** (L15 0.91, L17 0.86); any 2+ window 1.00; **shared-weights 1.00**; carriers = last-token 1.00 (all OOD) | ✅ | **One DeepSets block suffices**: a single mid-layer (L14 or L16), one shared φ/ρ, inject once at last token → 100% incl. OOD. Inject site irrelevant; weights shareable. Single-layer is mildly layer-dependent (L14/L16 perfect, L15/L17 weaker) — single-seed, don't over-read *which* layer. |
| 2026-06-13 | `outputs/evidence_only_sum_evidence_adapter_seq1_8_7b/2026*_dmem{1,2,4,8,16,64}_sum_L14_17_iid` | **Capacity (width) sweep**: vary d_mem on the sum readout, evidence-only **IID counts 0–8** (isolate width from extrapolation) | 7B, sum, L14–17, 3 ep | overall acc d1 **0.53** / d2 0.60 / d4 0.78 / **d8 0.99** / d16 1.00 / d64 1.00; high-count(6,7,8) 0.29/0.40/0.64/**1.00**/1.00/1.00 | ✅ | **Saturates exactly at d_mem=8 = max count N**; monotonic, failures concentrate at high counts. = DeepSets **width ≥ N** bound measured inside a VLM. Prescription: d_mem ≥ max expected count. (Refutes "scalar count needs no width" — width is a genuine second constraint.) |
| 2026-06-13 | `outputs/layerwise_frame_message_glstm/20260613_142804_distractor_pna_carrier_pna` | **PNA on distractors** (falsification: does aggregator richness close the distractor gap?) vs sum 63.8% / gLSTM 61.7% | 7B, distractor fillers, train 4,6,8 → OOD 5,7,10, 3 ep | PNA **iid 68.1%** / len-OOD 62.6% / comp-OOD 65.7% (memory-disabled 50.0%) | ✅(surprise) | **Prediction falsified: PNA beats sum** (+4.3 iid, +2.6 len, +6.2 comp). PNA memory contributes ~+18pp over LoRA vs sum's ~+12pp → distributional aggregators (max/std) carry real distractor-discriminative signal pure sum discards. Still ~28pp below oracle 96% → narrows but doesn't close the selection gap. Single-seed. |
| 2026-06-13 | `outputs/evidence_only_sum_evidence_adapter_seq1_8_7b/{20260613_153743_*,2026*_seed{1,2}_*}` (seeds 0/1/2) | **Multi-seed confirmation of both knobs** (n=3): pooling ablation (OOD seq5–8) + d_mem width sweep (IID 0–8) re-run at seeds 1,2 | 7B, L14–17, 3 ep | **Pooling OOD**: sum 1.00/1.00/1.00, pna 1.00/1.00/1.00, mean 0.76/0.76/0.77, softmax 0.76/0.76/0.77 (s0/s1/s2). **Width overall** (mean over seeds): d1 0.55 / d2 0.64 / d4 0.83 / **d8 0.99** / d16 1.00 / d64 1.00 | ✅ | Both thesis-core curves now n=3. Pooling gap is seed-invariant (±0.01; mean≈softmax every seed). Width knee replicates at **d_mem=8=N** across all seeds (per-seed d8 = 0.99/1.00/0.97). Operation + capacity are no longer single-seed claims. |
| 2026-06-13 | `outputs/distractor_posneg_write_read_adapter_seq8_7b/detceiling_w{14_17,18_21,14_21}_10ep` | **Per-frame detection ceiling**: gate trained as pure evidence detector (λ_ce=0, λ_count=0, λ_mask=1, 10 ep), measure max gate-AUC vs gold mask across write windows L14–17 / L18–21 / L14–21 | 7B, seq8 distractor | best-layer AUC: w14–17 **0.93 (L16)**; w18–21 **0.96 (L19)**; w14–21 **0.96 (L19)** | ✅📊 | **Detection ceilings ~0.96, at L19** (consistent across windows). Evidence is most linearly decodable at **L19**, not the L14–17 write window all prior gates used — that artifact was dragging earlier estimates to ~0.9. But 0.96/frame is still short of the ~0.99 needed: per-frame errors compound (bias×n + √n variance) so hard gating can't reach 96% and degrades with count. Concrete lever: gate from L19. |
| 2026-06-13 | `outputs/layerwise_frame_message_glstm/2026*_distractor_{sum,glstm}_memonly_*` | **Memory-only control** (`--no-carrier-lora`): isolate the memory adapter's contribution from the carrier-LoRA on the distractor task | 7B, distractor, train 4,6,8, 3 ep | **gLSTM-memory iid 0.674 > sum-memory iid 0.605** (len-OOD 0.62 vs 0.57; comp-OOD 0.62 vs 0.59) | ✅(nuance) | **Reverses the with-LoRA finding:** without the carrier-LoRA, the gLSTM associative read genuinely beats plain sum on distractors (+7pp iid). The LoRA was masking the memory's contribution. So "associative read is dispensable" holds *with* the LoRA but **not** without it — on distractors the richer read carries selection-relevant signal (consistent with PNA 68.1 > sum 63.8). Single-seed. |

### Week of 06-10→16 — Generalization beyond counting (06-14, 7B)

| Date | Output dir | Method / change | Key config | Metric | Status | Notes |
|------|-----------|-----------------|-----------|--------|--------|-------|
| 2026-06-14 | `outputs/eval_mmred_rooms_visited_baseline/full` (`evaluations/scripts/eval_mmred_rooms_visited_baseline.py`) | **Does the bottleneck generalize to a non-counting task?** Frozen-Qwen baseline on **rooms-visited** ("how many distinct rooms did C visit?", ans 0..6) — a **set-cardinality** aggregation (OR-within-room, then sum-over-rooms), labels recomputed from existing states, **no re-render**. 120 samples/seq_len | 7B, 4-bit, seq 1–8 | acc **0.87(s1) → 0.29 → 0.51 → 0.40 → 0.28 → 0.25 → 0.18 → 0.09(s8)**; undercounts (s8 mean pred 2.99 vs gold 4.34) | ✅📊 | **The over-squashing bottleneck is NOT counting-specific** — a different aggregation (set-cardinality) collapses the same way with seq_len, with the same under-aggregation signature. Generalizes the diagnosis the professor/peer asked about. Next: does the sum/DeepSets adapter fix it, or does set-cardinality need a different aggregator (OR/max-within-room)? Single-seed-equivalent (1 char sampled/sample). |
| 2026-06-14 | `outputs/eval_mmred_cooccupancy_baseline/full` | **Second generalization task** (different *predicate*): frozen-Qwen baseline on **co-occupancy** ("in how many frames were C and D in the same room?", ans 0..seq_len) — a frame-count with a 2-character predicate. 120/seq_len | 7B, 4-bit, seq 1–8 | acc **0.93(s1) → 0.28 → 0.23 → 0.21 → 0.18 → 0.25 → 0.18 → 0.29(s8)**; **over**counts (s8 mean pred 2.23 vs gold 1.43) | ✅📊 | Same collapse with seq_len → bottleneck is invariant to **predicate** too (not just aggregation type). **Cross-task insight:** error *direction* flips — rooms-visited (high golds) **undershoots**, co-occupancy (low golds) **overshoots** → under over-squashing the output **regresses toward a length-dependent middle estimate** regardless of the true value. This is the representational-collapse signature ([Barbero 2024](https://arxiv.org/abs/2406.04267)) seen behaviorally across two new tasks. |

### Week of 06-10→16 — New distractor-selection mechanisms: all plateau (06-14, 7B)

> Tests whether a *selection* mechanism beyond per-frame gating can close the distractor gap. Baselines:
> sum 63.8 / gLSTM 61.7 / PNA 68.1 (iid). All distractor, train 4,6,8 → OOD 5,7,10, single-seed.

| Date | Output dir | Method / change | Key config | Metric | Status | Notes |
|------|-----------|-----------------|-----------|--------|--------|-------|
| 2026-06-14 | `outputs/distractor_posneg_write_read_adapter_seq8_7b/b2_countsup_nomask_{w14_17,lategate_w18_21}_8ep` | **B2 — count-level supervision, no per-frame mask** (gate trained only via answer-CE + Σα≈count; λ_mask=0); also a lategate-at-L19 variant | 7B, seq8, 8 ep | **43.0%** (write 14–17) / **45.2%** (lategate L19); gate AUC 0.77 / 0.86 | ❌ | Training the gate on the count alone does **not** beat the ~47% plateau, and learns a *weaker* detector than per-frame-mask supervision. Selection-via-count-supervision is not the lever. |
| 2026-06-14 | `outputs/layerwise_frame_message_glstm/2026*_distractor_slot_noprobe_carrier_slot` (`CARRIER_SLOT`, new) | **B1 — slot-attention competitive routing + sum** (4 slots, softmax-across-slots assignment → per-slot unnormalized sum). The diagnosis-driven "global/competitive selection escapes per-frame compounding" hypothesis | 7B, L14–17, NUM_SLOTS=4, 3 ep, `--no-probes` | iid **64.0%** / len-OOD 60.8 / comp-OOD 61.4 (mem-disabled 53.6) | ❌(falsified) | Competitive routing lands **in the plateau** (≈ sum 63.8), not above it. **Conclusion across B1+B2:** the ~63–68% distractor plateau is **robust to every selection mechanism** (per-frame gate, count-supervision, competitive slots); only richer *aggregation* (PNA 68.1) nudges it, and only +4pp. The distractor gap is not closeable by better *selection*. (⚠️ first slot run 93951 hung in the sklearn probe phase — rerun with `--no-probes`; the variant's probe diagnostics are unvalidated.) |

### Week of 06-10→16 — Stage pathway task-general + aggregation diagnosis (06-16, 7B)

> Why it matters → Weekly progress → "Week of 06-10→16". Runs below.

| Date | Output dir | Method / change | Key config | Metric | Status | Notes |
|------|-----------|-----------------|-----------|--------|--------|-------|
| 2026-06-16 | `outputs/token_group_corruption_new_tasks/{count,rooms_visited,co_occupancy}_7b_n40` (`token_group_corruption_new_tasks.py`; plot `outputs/stages_7b_plots/stages_restoration_by_group_7b_n40.png`) | **Layerwise token-group restoration** (blank corruption): patch clean **frames / question / last-token** activations at each layer into the corrupted run; median normalized rescue. Same method for all 3 tasks (count now apples-to-apples) | 7B, seq8, layers 0–26, n=40/task, `--corruption_mode evidence` | **All 3 tasks**: frames rescue EARLY (L0–12 ≈1.0) → question/carrier MID (peak L14–16) → last-token LATE (L18–26 →1.0). Crossover ~L14–16 | ✅ | **Stage phenomenon is task-general** — identical frames→carrier→last staircase across counting + the 2 new set-cardinality tasks → a general (training-induced) model behavior, not task-specific. |
| 2026-06-16 | `outputs/probe_frame_to_carrier_message/{rooms_visited,co_occupancy,count}` (`probe_frame_to_carrier_message.py`, sanctioned SDPA attn recompute) | **Per-frame evidence + decode-then-count from the attention-routed frame→carrier message** (capture q/k/v + position_embeddings, reuse library rotary, recompute carrier-row softmax offline; model stays SDPA) | 7B, n=150, L16/18/19 | per-frame evidence decodes: rooms_visited room **0.85**, co_occ same/diff **AUROC 0.97** (shuffle ≈chance); **decode-then-count ≫ frozen model**: rooms **0.63 vs 0.10**, co_occ **0.52 vs 0.27** | ✅ | Evidence is extracted **and routed** to the carrier; the model fails to **aggregate** it → revises sprint-2 "vision extraction" framing to **AGGREGATION**. global_lora ~0.58 ≈ decode-then-count ceiling. |
| 2026-06-16 | `outputs/rooms_visited_adapter/*rv7b_ev_vit{lora_late16,lmlora}*` (`layerwise_frame_message_glstm.py` `vision_lora`/`vision_lm_lora`) | **ViT / vision-encoder LoRA** (wrap `model.visual.blocks[*].attn`) — does adapting the frozen vision tower help? | 7B, rooms_visited evidence | vision_lora (ViT-only) **0.50**, vision_lm_lora (ViT+LM) **0.596** vs global_lora (LM-only) **0.58** | ✅ | Vision-encoder adaptation does **not** help beyond LM-attention LoRA → confirms the lever is LM re-attention (aggregation), not vision extraction. |

### Week recap diagnostics (06-16→17, 7B) — symbolic ceiling, stage localization, operator/data sweeps

| Date | Output dir | Method / change | Key config | Metric | Status | Notes |
|------|-----------|-----------------|-----------|--------|--------|-------|
| 2026-06-16 | `outputs/oracle_text_distinct_count/{rooms_visited,co_occupancy}_result.txt` (`oracle_text_distinct_count.py`) | **Symbolic-input ceiling**: feed frozen 7B the clean per-frame room/occupancy sequence as **text** (no vision) → isolates dedup/count arithmetic from extraction | 7B, n=120/task | rooms_visited **0.758**, co_occupancy **0.983** | ✅📊 | Count arithmetic is near-solved for co-occ (98%) and only mid for distinct-room dedup (76%) → most rooms-visited headroom is in the vision→carrier→aggregation path, not the arithmetic. |
| 2026-06-16 | `outputs/probe_aggregation_stages/{rooms_visited,co_occupancy}` (`probe_aggregation_stages.py`); `outputs/probe_frame_token_states/{rooms_visited,co_occupancy}` (`probe_frame_token_states.py`) | **Where does count die?** per-frame evidence decode vs last-token count decode by layer; plus frame-token-state mean/sum pooling probe | 7B, n=150 | per-frame room decode **0.84** (L4); last-token count decode **≤0.475** ≈ maj 0.434 (all layers); frame-token mean/sum pool ≈ blind (~0.37–0.43) | ✅📊 | Evidence present at frames; **count absent at the last token and not in pooled frame states** → loss is in the aggregation step, corroborating the frame→carrier message probe. |
| 2026-06-16→17 | `outputs/agg_sweep/*rv*`; `outputs/agg_moredata/20260617_040354_md_rv_sum_ml_carrier_direct_sum` | **Operator + data sweep on rooms-visited**: carrier sum/slot/union/gLSTM/max at L12–17, then 2× data on carrier-sum | 7B, single-seed | operators cluster **~46% iid** (carrier-sum 45.7%, mem-disabled 40.4%); 2× data → carrier-sum **iid 58.1%** (comp-OOD 41.7 / len-OOD 49.0) | ✅ | Operator choice ≈ noise; **data/capacity moves rooms-visited** (46→58). Best overall remains global LM-attn LoRA 63.9%. |

### Text-frames + diagnostic decomposition (06-17→19, 7B) — composition bottleneck, selection-signal availability

All text-side unless noted. Heatmaps per (gold-count × seq_len) saved per run; predictions logged. Scripts:
`eval_mmred_text_frames_acc.py` (`--cot`, `--oracle-list`, precision flags), `probe_evidence_selection_linear.py` (text),
`probe_evidence_selection_image.py` (vision tokens). Runners: `eval_text_frames_*.sbatch`, `probe_evidence_selection*.sbatch`.

| Date | Output dir | Method / change | Key config | Metric (steps / rooms / co-occ) | Status | Notes |
|------|-----------|-----------------|-----------|--------|--------|-------|
| 2026-06-17 | `outputs/eval_mmred_text_frames_acc/` | **Frames-as-text, plain single-token** (no vision) | 7B nf4, 80/seq, sl1–8 | **0.470 / 0.389 / 0.338** | ✅📊 | Same collapse as images ⇒ not vision. Range-compressed prior: steps/rooms undercount (bias→−1.7 @sl8), co-occ overcounts (+0.8). |
| 2026-06-17 | `outputs/eval_mmred_text_frames_acc_cot/` | **+ chain-of-thought** (reason then `Answer:`) | 7B nf4, 50/seq, 512 tok | **0.695 / 0.570 / 0.623** | ✅ | steps bias →≈0 at every len (single-pass artifact removed); residual long-len decay remains. |
| 2026-06-17 | `outputs/eval_mmred_text_frames_acc_oracle/` · `_oracle_cot/` | **Oracle-list** (only queried entity's per-frame rooms; no scene) plain · CoT | 7B nf4, 80/50/seq | plain **0.841 / 0.883 / 0.662**; CoT **0.985 / 1.000 / 0.940** | ✅ | rooms_visited oracle-CoT **100% @ every sl incl 8, bias 0.00** → aggregation/dedup easy once extracted; revises 06-17 "set-accumulation limit". |
| 2026-06-17 | `outputs/eval_mmred_text_frames_acc_prec_nf4/` · `_prec_bf16/` | **Precision lever**: 4-bit nf4 vs full bf16 weights (bf16 compute both) | 7B, 80/seq, plain | nf4 0.470/0.389/0.338 · bf16 0.470/0.423/0.325 | ✅ | nf4 ≈ bf16; saturation unchanged ⇒ quantization is **not** the cause. |
| 2026-06-18 | `outputs/probe_evidence_selection_linear/` | **Linear selection probe (text tokens)**: per-layer logreg for `is_evidence` | 7B, n=2160 frames (sl4,6,8), 50.5% pos | best **layer 21 AUC 0.997**, bal-acc 0.965 (chance 0.5) | ✅📊 | Evidence/distractor **linearly present** in the frozen text rep, peaks mid-late. |
| 2026-06-19 | `outputs/probe_evidence_selection_image/` | **Linear selection probe (vision tokens)**: same, over `<\|image_pad\|>` per-frame spans, question-first | 7B nf4, n=1800 frames, 50.6% pos | best **layer 19 AUC 0.984**, bal-acc 0.939 | ✅📊 | Signal **survives into vision tokens** (≈ text) ⇒ distractor gap is **not** vision-side perception; the gating plateau is a *supervision* gap (signal present, not learned from count loss). |

### Frame-axis aggregator adapter (06-19→20, 7B) — read@L19 → aggregate → inject; deepsets wins

One-pass adapter; reads per-frame vision reps at L19, aggregates, readout = LM-injection (live) or count-head
(cached, fast proxy on a shared rep cache `outputs/frame_axis_cache/L19.pt`). Joint 3-task training, disjoint
stratified splits, per-epoch val + best-epoch ckpt. Scripts: `frame_axis_aggregator_adapter.py` (live),
`frame_axis_aggregator_cached.py` (cached). Metric = exact-match count; bias = mean_pred − mean_gold.

| Date | Output dir | Method / change | Key config | Metric | Status | Notes |
|------|-----------|-----------------|-----------|--------|--------|-------|
| 2026-06-19 | `outputs/frame_axis_aggregator_cached/` (deepsets, seqmodel) | **cached count-head**, deepsets vs seqmodel, train 1–6 / OOD 7–8 | 7B nf4, 40 ep | deepsets val **0.844** > seqmodel 0.713; IID 0.73/0.54/0.80; OOD count-head **saturates** (rooms bias −1.80) | ✅ | deepsets > seqmodel; count-head under-reports OOD. |
| 2026-06-19 | `outputs/frame_axis_aggregator_cached_1to8/` | **train+test 1–8** (no OOD), deepsets count-head | cap 400/seq, 40 ep | IID 0.667/0.510/0.772, **bias ≈0** (rooms −0.04) | ✅ | Full length range → **bias gone**; rooms saturation was an OOD-length artifact. |
| 2026-06-19 | `outputs/frame_axis_live_h2h/{deepsets,deepsets_balanced,pna_balanced,pna_cb_balanced}` | **live LM-injection 4-way**, train 1–6 / OOD 7–8 | 7B nf4, 20 ep | IID mean 0.754 / 0.737 / 0.766 / 0.686; **OOD mean 0.558 / 0.547 / 0.528 / 0.442** | ✅ | **deepsets best OOD**; PNA scaler **hurts OOD** (bias −0.33); codebook dead; balanced marginal. |
| 2026-06-19 | `outputs/frame_axis_sweep/{6 configs}` | **cached aggregator sweep** (train+test 1–8, converged) | 120 ep | mean test_iid: deepsets **0.720**, pna_cb_bal 0.714, pna_bal 0.698, pna 0.690, pna_cb 0.638, ds_cb 0.611 | ✅ | No config beats plain deepsets; codebook-φ falsified for dedup. |
| 2026-06-20 | `outputs/frame_axis_live_deepsets_eval/` | **live-readout IID+OOD diagnostic plots** (deepsets winner, eval-only from ckpt) | EPOCHS=0 + `--init-from` | IID mean_pred tracks y=x to ~6 (vs count-head saturating) | ✅📊 | LM-injection readout well-calibrated; per-split plots: acc/mean-pred/confusion. |
| 2026-06-20 | `outputs/frame_axis_live_attnpool/` | **query-conditioned attention pool** per frame (vs mean-pool) — *the extraction fix* | deepsets, train 1–6 / OOD 7–8, 20 ep | ▶ running | ▶ | Targets the per-frame extraction ceiling (0.94/frame compounding); needs live (raw tokens). |

### Extraction-ceiling probes (06-20, 7B) — the read side is maxed at ~0.94/frame; *blame not yet fully localized*

Three probes of per-frame **is-evidence** (C in R) decodability, all on **mean-pooled** L19 vision reps with a
**linear** logreg (steps_in_room, n=1800 frames, sl 4/6/8, sample-disjoint split).

| Date | Output dir | Probe | Result | Status | Notes |
|------|-----------|-------|--------|--------|-------|
| 2026-06-20 | `outputs/probe_adapter_messages/` (`probe_adapter_messages.py`) | does the **trained adapter's φ(rep)** still decode evidence vs the raw rep? | raw **0.939** bal-acc / 0.984 AUC; φ-message **0.931** / 0.977 (Δ ≈ 0) | ✅📊 | φ **preserves** per-frame evidence → φ/aggregation/readout near-optimal; adapter (count-loss-trained) extracts **as well as a supervised probe** → no supervision gap. |
| 2026-06-20 | `outputs/probe_multilayer_evidence/` (`probe_multilayer_evidence.py`) | does **concatenating layers** beat the best single layer? | best single L19 0.939/0.984; concat band {14,17,19,22,25} 0.946/0.983; concat ALL 0.935/0.978 | ✅📊 | Multi-layer **redundant** (AUC gain ≈ 0; residual-stream layers correlated). Single mid-late layer suffices. |
| (06-19) | `outputs/probe_evidence_selection_image/` | per-**single-layer** sweep | broad plateau L18–24 (~0.93–0.94 bal-acc / ~0.98 AUC); peak L20 0.944 | ✅📊 | No layer beats ~0.94; **layer choice robust** (no per-model tuning — "read ~0.7·depth"). |

**Conclusion (scoped).** For **[mean-pool + linear probe + single/multi-layer]**, per-frame evidence caps at
**~0.94 bal-acc / ~0.98 AUC**, and the adapter sits at that ceiling → the ~75% count accuracy is
**compounding-limited (0.94⁶≈0.69)**, with the read side (layer/pooling/φ/aggregation) near-optimal.

| 2026-06-20 | `outputs/probe_token_extraction/` (`probe_token_extraction.py`) | **Probe A — best read over RAW frame tokens** for is-evidence: mean/max/attn-pool+MLP | 7B, n=1260 frames | **mean+linear 0.897** (best); max+linear 0.827; attn-pool+MLP 0.873 | ✅📊 | **Mean+linear is best — token-level & non-linear reads do NOT beat it** → pooling/non-linearity is NOT the bottleneck; **attn-pool won't help.** |
| 2026-06-20 | `outputs/probe_perception_binding/` (`probe_perception_binding.py`) | **Probe B — perception vs binding** (query-cond vs query-indep, binding-aware probe) | 7B, 14k triples | both ~**0.51 (chance)** | ⚠️ FLAWED | Confounded: probes *arbitrary* (X,Y) while reps encode only the *target*, and mean-pool destroys per-char binding → **does NOT localize vision-encoder vs LM-binding.** Discarded. |
| 2026-06-20 | `outputs/probe_pertask_extraction/` (`probe_pertask_extraction.py`) | **Per-task per-layer extraction sweep** — all layers, rooms (7-way room-of-C) & co-occ (same-room) | 7B, n≈1620/task | rooms peaks **L21 0.925** (L19 ~0.915); co-occ peaks **L19 0.996**; broad plateau L18–24 | ✅📊 | **Read layer L19 is within ~1pt of the best layer for ALL 3 tasks** (steps L19–20, rooms L21, co-occ L19) → the read layer is **not** limiting any task; "read ~0.7·depth" is robust, no per-model/per-task tuning. |
| 2026-06-21 | `outputs/frame_axis/balanced/{rooms,cooc}_{image,text}/` (`frame_axis_aggregator_adapter.py`) | **count-balanced deepsets, image vs text** (N=8, uniform counts, live 7B) | deepsets, 20ep | rooms img **0.500**/txt 0.398; co-occ img **0.586**/txt 0.438 | ✅📊 | image≥text both; image unbiased, text biased; balanced data → co-occ honest 0.586 (vs skew-inflated 0.867). Small per-count support, single seed. |
| 2026-06-21 | `outputs/frame_axis/balanced/{steps_text,cooc_text_long,rooms_text_targetL1}/` | **text controls** — steps; co-occ 35ep; rooms target@L1 | live 7B | steps·text **0.704**; co-occ·text **0.519** (35ep, was 0.438); **rooms target@L1 lm 0.435 / aux 0.407** | ✅📊 | steps (clean extraction 0.997)→0.70; **perfect extraction (target@L1=1.0) still → 0.41 on rooms** ⇒ rooms aggregation-bound, not extraction. |
| 2026-06-21 | `outputs/frame_axis/probes/crowding_min/` (`probe_pertask_extraction.py`, `probe_evidence_selection_image.py`) | **L19 extraction vs entity count** (image) | 7B, n≈90–135/cfg | rooms 1ch **0.996**/2ch 0.915/5ch 0.835; steps 1ch **0.984**/2ch 0.970/5ch 0.939; co-occ 3ch **0.999**/5ch 0.996 | ✅📊 | **monotonic decline with crowding at deployed L19 = superposition, measured.** co-occ ~0.99 always (not extraction-bound). 1–2ch co-occ = density-confounded (discard). |
| 2026-06-21 | `outputs/frame_axis/probes/base_acc/` + `steps_base` log (`eval_mmred_rooms_visited_baseline.py`, `eval_mmred_qwen25_vl_accuracy.py`) | **frozen base acc at minimal crowding** (image, no adapter) | 7B, n=90–108 | steps 1ch **0.583** (U-shape), rooms 1ch **0.289**, co-occ 2ch **0.204**; all undercount | ✅📊 | **frozen fails even with 1 entity + perfect extraction** — worst on distinct-count/relational → single-pass *aggregation* is the bottleneck, not perception. Sets the adapter's bar to beat. |
| 2026-06-21 | `outputs/frame_axis/probes/text_pooling_sweep/` (`probe_text_pooling_sweep.py`) | **text pooling × layer** (mean/last/max/target) | 7B, balanced N=8 | rooms: target@L1 **1.000** vs mean 0.767/last 0.733/max 0.764; co-occ: mean 0.964 best | ✅📊 | text rooms recoverable only via queried-token@L1 (positional/trivial); co-occ pooling-insensitive → "image>text" for rooms was a mean-pool artifact. |
| 2026-06-20 | `outputs/frame_axis/probes/evidence_selection_image_32b/` | **32B steps is-evidence** (image probe, all-layer) | 32B nf4, n=60/seq sl4/6/8 | best **L43 bal-acc 0.892 / AUC 0.951** (vs 7B L19 0.939/0.984) | ✅📊 | Bigger backbone does **not** raise the perception ceiling; slightly-lower likely a probe-data artifact (60 vs 90 samples, dim 5120 vs 3584). Claim = "no lift," not decrease. |
| 2026-06-20 | `outputs/frame_axis/probes/pertask_extraction_32b/` | **32B co-occ same-room + rooms room-decode** | 32B nf4, n=60/seq | co-occ **L44 AUC 0.999** (7B 0.996); rooms **L48 0.858** (7B 0.915) | ✅📊 | co-occ **already saturated** at 7B → no headroom; rooms no lift → perception not capacity-bound. |
| 2026-06-20 | `outputs/frame_axis/probes/extraction_resolution_crowding/` (`probe_evidence_selection_image.py --image-sizes`) | **Exp1 resolution sweep + Exp3 crowding** (steps is-evidence, mean-pool) | 7B, n=40/seq, sizes 224–672px | AUC: 224 .873 / 336 .945 / 448 .971 / **native512 .969** / 672 .982; crowd@native 4ch .962 / 5ch .926 | ✅📊 | **Resolution plateaus at native** (upscale +1pt only) → not a lever. Crowding **underpowered** (data only 4–5 chars/frame) but directional (5<4) = binding. |
| 2026-06-20 | `outputs/frame_axis/probes/per_frame_verify_steps/` (`eval_per_frame_verification.py`) | **Exp2 per-frame "look-again" verification**, steps | 7B, n=60/seq sl4/6/8 | per-frame **bal-acc 0.987 / AUC 1.000**; **count hard 0.928** / soft 0.850 | ✅📊 | Isolated frame ~perfect (≫ joint 0.94) ⇒ extraction ceiling is **single-pass superposition, not perception**; count **0.79→0.928** (extraction-axis CoT, N× compute). |
| 2026-06-20 | `outputs/frame_axis/probes/per_frame_verify_cooc/` (`eval_per_frame_verification.py`) | **Exp2 per-frame verification**, co-occ | 7B, n=60/seq | per-frame 0.909/0.981; **count hard 0.722** / soft 0.717 (0.867→0.717→0.583 by sl) | ✅📊 | **Worse than single-pass adapter (0.867)**: co-occ wins via soft error-cancellation; per-frame hard-sum loses it & compounds → **not extraction-limited**. |
| 2026-06-20 | `outputs/frame_axis/adapter_live/rooms30/20260620_182313_deepsets/` | **rooms-only deepsets, 30ep (patience-8 early-stop @ep14)**, train+test sl1–6 | 7B nf4, train cap 1140, no OOD | best val **0.778 (ep6)**; **test_iid 0.691**, bias +0.00; g5 collapse **0.184** | ✅ | **Plateau ≪ 0.865 extraction bound → rooms aggregation-limited (structural), not optimization.** vs multi-task 0.748 confounded by data cap (1140 vs 1890) — *not* "single-task hurts." |
| 2026-06-20 | `outputs/frame_axis/adapter_live/h2h_cont_evalplot/` | **acc-per-count ceiling plot regen** (eval-only from h2h_cont best deepsets ckpt) | EPOCHS=0 + `--init-from`, train1–6/ood7,8 | steps 0.790 (**on** ceiling), rooms 0.748 (**below**), co-occ 0.867 (**above** hard bound) | ✅📊 | Canonical "acc per count + extraction-ceiling line" figure; rooms peels off ceiling at g≥3, g5 0.34. |
| 2026-06-23 | `outputs/frame_axis/agg_min/rooms_visited_{deepsets,logic}/` | **minimal-crowding (1char) rooms: DeepSets vs soft-OR** | live 7B, 250/count, 30–35ep | DeepSets **0.973** (counts 1/1/1/.91/.97/.95); **logic 1.000** (perfect all counts) | ✅📊 | decrowd lifts rooms 0.50→0.97 (extraction-limited, not agg); **soft-OR perfects distinct-count**. |
| 2026-06-23 | `outputs/frame_axis/agg_min/steps_in_room_{deepsets,logic}/` | **minimal-crowding (1char) steps** | live 7B, 250/count | DeepSets **0.9496**, logic **0.9614** (both ≥0.85 all counts) | ✅📊 | decrowd 0.58→0.95; **beats 0.86 hard-compounding ceiling** via soft sum; logic≈DeepSets. |
| 2026-06-23 | `outputs/frame_axis/agg_min/co_occupancy_{deepsets,logic}/` | **minimal-crowding (3char) co-occ** | live 7B, 250/count | DeepSets **0.8991** (counts 2,4 dip 0.76–0.79); **logic 0.9377** (per-count 0:.97 1:.94 2:.86 3:.90 4:.90 5:.95 6:.97 7:.95 8:1.00) | ✅📊 | decrowd 0.20→0.90; beats 0.78 ceiling; **soft-sum (logic) +0.039 over DeepSets, lifts the count-2/4 dips to ≥0.86.** |
| 2026-06-23 | `outputs/frame_axis/probes/{phimsg_min,crowding_min}/` | **L19 extraction (hard bal-acc/7-way) vs #chars** (probe, no training) | 7B image | rooms 0.835(5ch)→0.915(2ch)→0.996(1ch); steps 0.939→0.970→0.984; co-occ ~0.97@3ch; **φ preserves all** | ✅📊 | clean isolation: crowding (superposition) is the extraction lever; φ not lossy. |
| 2026-06-23 | `outputs/frame_axis/ood_holdout/{steps,rooms,co_occupancy}_{deepsets,logic,lora}/` | **count-extrapolation OOD** (train low counts, test held-out high) | live 7B, minimal-crowding | OOD acc: deepsets/logic **0.000** all tasks; LoRA steps **0.452** (only count8="all frames"), cooc 0.094, rooms 0.000 | ✅📊 | **readout cap**: 9-way CE classifier can't emit unseen labels; mean_pred pins at top trained count. Neither extrapolates intermediate counts. |
| 2026-06-23 | `outputs/frame_axis/probes/count_direction_extrap*/`, `generic_number_direction_L{16,19}/` | **cd-injection (Solution 3) probe**: linear count-direction read + steer | 7B; task-axis + generic arithmetic-axis | READ: task 8→5.3(acc0); generic perfect 0–4, 5→0.55, saturates 7,8. STEER/dose: emit_mean **flat to ±16×rms**, both layers | ✅📊 | **injection-for-extrapolation dead**: number axis saturates past ~5; direction non-causal at answer site. Only learned codebook controls output (in-range, no extrapolation). |
| 2026-06-23 | `outputs/frame_axis/ood_holdout/{steps,rooms}_additive/` | **Solution 1: extensive additive count head** (`Σσ(·)`, `--count-readout additive`), count-holdout | live 7B, minimal-crowding | *running* (confirmatory) | ⏳ | full-adapter clean-data confirmation of the CPU-probe result below. |
| 2026-06-23 | `cache/L19.pt` (CPU probe, no training of the VLM) | **additive readout vs 9-way classifier, count-holdout** (train ≤4, test 5–8); same reps, readout swapped | cached L19 per-frame reps, crowded | **steps additive extrapolates: 5–8 acc 0.82/0.83/0.88/0.93; classifier 0.00**. rooms: sum caps (wrong op→needs soft-OR). co-occ inconclusive (sparse) | ✅📊 | **readout, not aggregation, is the OOD wall**: extensive sum extrapolates, softmax caps. Operator must match task. |
| 2026-06-23 | `outputs/frame_axis/readout_benchmark/{benchmark,stability}.csv`, `cache/minimal_L19_*.pt` | **OOD count-extrapolation benchmark** on minimal-crowding cached reps (count-holdout, multi-seed): base/CoT/LoRA/classifier vs sum/soft-OR | frozen 7B reps, CPU readouts | **per-frame-sup sum/soft-OR: steps 0.996 / rooms 1.000 / co-occ 0.974 (stable)**; base/CoT/LoRA/classifier OOD ≤0.45 (mostly 0); count-only sum unstable (steps 0.52±0.38) | ✅📊 | the verified main result; fixed-extensive readout extrapolates, generation baselines + classifier collapse. |
| 2026-06-23 | `outputs/frame_axis/readout_benchmark/{deepsets_proper,deepsets_framesup,deepsets_universal,auxloss}.csv` | **why learned readouts fail**: canonical DeepSets `ρ(Σφ)` (count-only, +per-frame-sup, +fixed-extensive-channels+linear ρ); aux-loss λ sweep | cached reps, multi-seed CPU | canonical DeepSets fails all configs (0.08–0.59, unstable); ρ-MLP caps OOD; universal-linear-readout 0.13/0.20/0.08; aux λ must be ≈1 to stabilize | ✅📊 | **principle: extrapolation needs a parameter-free fixed extensive readout on the supervised per-frame quantity; any learned ρ (even linear) breaks it.** |
| 2026-06-24 | `experiments/glstm/benchmark_readout_ablation.py` (+ `dimsweep.csv`) | **definitive readout ablation**: per-frame detector FIXED, vary only pooling (sum/mean/max/learned-ρ); + latent-dim sweep {64..1024} | cached reps, multi-seed CPU | sum IID→OOD 0.97→**0.997** (steps), 0.90→**0.974** (cooc); mean/max OOD **0.00**; **learned ρ(Σ) 0.98→0.74** (degrades). dim-sweep: more dim → *worse* OOD | ✅📊 | the headline figure: only the fixed parameter-free sum is lossless OOD; learned decoder on the correct sum drops ~25pts; capacity makes it worse not better. |
| 2026-06-23 | `outputs/frame_axis/agg_min/lora_{rooms,steps,cooc}/` (`lora_sft_baseline.py`, peft) | **plain-LoRA SFT baseline** (native softmax, no aggregator) | h200, r=16 | ▶ running (slow n315 model-load ~18min; not a bug) | ▶ | baseline: does fine-tuning native softmax match the explicit aggregator. |

**Conclusion (scoped).** For **[mean-pool + linear/MLP + single/multi-layer + token-level]**, per-frame evidence
caps at **~0.94 bal-acc / ~0.98 AUC** — the **read side is exhausted**; *no* pooling/layer/aggregator/φ choice
beats it (Probe A: mean+linear is the best read). The adapter sits at that ceiling. **The ~0.94/frame is the
frozen 4-bit Qwen's per-frame information content; everything the adapter does on top is near-optimal.**

**Still open (Probe B was the wrong instrument):** whether the ~0.94 is **vision-encoder perception** vs **LM
query-conditioned binding** is *unresolved* — a correct probe needs a **raw-token, set-aware, query-independent
occupancy** decode (no mean-pool). Practically it may not matter: both are upstream of the frozen reps, so the
only lever past ~0.94 is **unfreezing (LoRA on vision encoder / early LM)**. Read-side options are done.

**Why count accuracy (~0.80 IID) >> 0.94⁶≈0.69** (not a contradiction): (1) IID averages over **seq 1–6**, and
0.94ⁿ for n=1..6 averages ≈ **0.81** (short seqs are easy) — 0.94⁶ is only the *seq-6* worst case; (2) **error
cancellation** — count is a sum, so opposite per-frame errors cancel → P(count right) > P(all frames right);
(3) **soft aggregation** — the adapter sums *continuous* evidence, more robust than hard-threshold compounding.
deepsets +10ep continuation hits **IID 0.802** (steps 0.79 / rooms 0.75 / co-occ 0.87), right at the per-frame
ceiling → the adapter is near-optimal given frozen perception.

**✅ MILESTONE — adapter ceiling reached.** The frame-axis DeepSets adapter (read@L19 → φ → sum/mean/max →
inject) is **near-optimal given the frozen 4-bit Qwen**: ~0.80 IID across the 3 tasks, unbiased, sitting at
the averaged per-frame-info ceiling. The read side is exhausted (no pooling/layer/aggregator/φ beats it).
The adapter line is **done** as a no-touch fix.

**Per-task extraction probes (06-20, `outputs/probe_pertask_extraction/`) — the residual is TWO different
bottlenecks.** Per-frame extraction is **good for all three** tasks: steps is-evidence **0.94**, co-occ
same-room **0.94** (AUC 0.996), rooms-visited room-of-C **0.915** (7-way; majority 0.21). So extraction is NOT
what separates them:
- **steps_in_room & co_occupancy = extraction-bound** — aggregation is a *sum* (error-cancelling, easy), so
  they sit at the 0.94/frame compounding limit (0.79 / 0.87; co-occ higher = low counts).
- **rooms_visited = aggregation-bound, NOT extraction-bound** (corrects earlier ~0.85 guess). Extraction is
  fine (0.915); the limiter is **single-shot distinct-count / dedup**: the symbolic clean-input ceiling is only
  **0.758** (distinct-count has *no error cancellation* + must hold the *set*), and the adapter (0.748) sits at
  it. **CoT → ~1.0** (serial dedup) → it's the over-squashing/aggregation bottleneck, not perception or the
  aggregator (sweep tied; codebook-φ failed).

**Net framing:** steps & co-occ are **frozen-perception-bound** (lever = vision LoRA); **rooms_visited is the
clean one-pass set-aggregation bottleneck** (lever = serial computation / set-memory, e.g. distilling CoT).

**Extraction-bound ceiling (task-generic optimality test, 06-20).** Method: corrupt the *true* per-frame
quantities with the *measured* per-frame extraction error, apply **perfect aggregation**, Monte-Carlo the
answer accuracy = "best achievable given current extraction." Adapter ≈ ceiling ⇒ aggregation near-optimal;
adapter ≪ ceiling ⇒ aggregation has headroom. (Distinct from the **symbolic clean ceiling** = perfect
extraction, *frozen-LM* aggregation.)

| task | extraction-bound ceiling (perfect agg) | symbolic clean ceiling (frozen-LM agg) | adapter IID | read |
|------|------|------|------|------|
| steps_in_room | 0.816 | ~1.0 | 0.790 | **adapter ≈ ceiling → aggregation optimal** |
| co_occupancy | 0.815 (hard; true higher, AUC 0.996) | 0.98 | 0.867 | adapter ≥ ceiling → optimal (uses soft evidence) |
| rooms_visited | **0.865** | 0.758 | 0.748 | **adapter ≪ 0.865 → ~12pt AGGREGATION headroom** |

**Correction to the milestone:** steps & co-occ aggregation is near-perfect (adapter at the extraction-bound
ceiling). **rooms_visited is NOT at its aggregation ceiling** — a *perfect dedup* would reach **0.865** even at
today's 0.915 extraction, but the adapter only reaches 0.748 (≈ frozen-LM single-shot dedup). So the DeepSets
max-union is **not** deduping optimally; ~12 pts remain for a better one-pass set-aggregation or CoT-distillation.
This is the genuine remaining over-squashing target (codebook-φ failed; CoT→~1.0). `cf. inline Monte-Carlo
06-20; per-frame extraction steps 0.94 / co-occ 0.94 / rooms 0.915.`

---

### Diff Transformer & Mamba operator bake-off (2026-06-21)

> New aggregator variants in `experiments/glstm/layerwise_frame_message_glstm.py`: **`carrier_mamba`**
> (selective diagonal SSM scan over the frame axis; pure-torch, no mamba-ssm dep; `--mamba-decay-init/-readout/
> -order-aug/-eval-permute`) and **`carrier_diff`** (difference-of-two-softmax read over frames; `--[no-]diff-output-norm`).
> Controlled comparison: identical config across variants, 7B nf4, 6 ep, train-per-count 35, **candidate-max 10**
> (so length-OOD len-10 = genuine count-extrapolation to unseen counts 9–10), **single-seed** (multi-seed confirm
> in progress). Operators unit-tested (shapes/grads/counting-monotonicity). Two infra bugs fixed en route:
> runner `--candidate-max 8` → KeyError on count len-OOD (override to 10); shared-NFS checkpoint-dir race
> (defensive mkdir-before-save + unique OUTPUT_ROOT per job).
> **Caveat:** absolute levels here are *below* the frame-axis DeepSets adapter (e.g. co-occ 0.867) — different
> read window (L14–17 vs L19), 6 vs 30 ep, train sl4–8 vs sl1–6. These runs isolate the **operator** (ranking is
> valid); they are not the project's SOTA absolute numbers.

| Date | Output dir | Method / change | Key config | Metric | Status | Notes |
|------|------------|-----------------|------------|--------|--------|-------|
| 2026-06-21 | `outputs/dm4_count_{base,mamba,diffoff,diffon}/` | **carrier_mamba & carrier_diff(±output-norm) vs sum** — counting, neutral fillers | 7B nf4, 6ep, n=420, single-seed; len-OOD = counts 9–10 unseen | iid/lenOOD — **mamba 0.998/0.937** · sum 0.981/0.880 · diff-noNorm 0.931/0.840 · diff-norm 0.881/0.767 | ✅ | Mamba ≥ sum incl. extrapolation (memdis 0.55→0.998 ⇒ operator does the work). Diff output-norm caps counting; norm-off recovers but still < sum/mamba. Single-seed. |
| 2026-06-21 | `outputs/dm4_distract_{base,mamba,diffon}/` | same operators — counting with **distractor** fillers (selection) | 7B nf4, 6ep, n=420; oracle pos/neg 0.963 | iid — mamba 0.652 ≈ sum 0.643 > diff 0.607 (lenOOD ~0.57–0.60) | ✅ | All ≪ oracle 0.96. **Diff WORSE than sum** & its memory adds ~0 (memdis 0.579→0.607). Neither operator closes the selection gap. |
| 2026-06-21 | `outputs/diffmamba2_coocc/` | operators — **co_occupancy** evidence-only | 7B nf4, 6ep, n=106; oracle 0.98 | iid — glstm 0.613 · diff 0.604 · mamba 0.557 · sum 0.538 | ✅ | All ≪ 0.98 oracle; operator barely matters, memory adds little → routing/aggregation-limited, not operator. (Absolute < frame-axis adapter; see caveat.) |
| 2026-06-21 | `outputs/dm4_{count_mamba,order_permEval,order_aug}/` | **mamba order-sensitivity** on counting | 7B nf4, 6ep, n=420 | iid 0.998 (normal) → 0.971 (permuted eval) → 0.979 (order-aug) | ✅ | Mamba **order-robust** on counting (near-sum decay init ⇒ ~perm-invariant); −2.7pp under permuted frames, aug recovers. |
| 2026-06-22 | `outputs/dm5_count_{sum,mamba}_s{1,2}/` + dm4 seed0 | **3-seed confirm: mamba vs sum on counting** (seeds 0/1/2, data+init varied) | 7B nf4, 6ep, n=420 | IID **mamba 0.988** [.981,.998] > sum 0.972 [.950,.986]; **len-OOD mamba 0.929** [.920,.937] ≫ sum 0.877 [.873,.880] (non-overlapping); comp-OOD 0.986 vs 0.956 | ✅ | Headline robust across seeds; mamba's **count-extrapolation edge is the clean win** (len-OOD ranges don't overlap). IID edge real but small (ranges overlap). |

**Conclusion (Diff vs Mamba).** **Mamba is a modest win** — matches/beats sum on counting *and* extrapolates
better to unseen high counts (0.937 vs 0.880 len-OOD), is order-robust in practice, and contributes the most of
any operator (largest mem-disabled→IID lift). It behaves as "a sum that can also gate," and is the natural choice
for future order/sequential tasks where sum provably can't go. **Differential Transformer is not worth pursuing**
as the aggregator: its output normalization caps counting (norm-off helps but still < sum/mamba), and on the
distractor task — where its signed attention was *predicted* to win — it was **worse than sum**. **Most important:
neither operator breaks the hard-task ceilings** (distractor ~0.65 vs 0.96, co-occ ~0.61 vs 0.98) — re-confirming
the frame-axis finding that the residual bottleneck is extraction/routing & one-pass set-aggregation, **not the
mixing operator**. Mamba only helps where the bottleneck genuinely *is* aggregation (counting). *3-seed confirmed (06-22):
mamba ≥ sum on counting holds across seeds; the count-extrapolation edge (len-OOD 0.929 vs 0.877) has
non-overlapping ranges. The other tasks remain single-seed.*

---

## 2026-06-27 session — temporal tasks, causal dispersion test, extraction re-check, per-frame-sup reconciliation

> A diagnosis-tightening session. Net: we **causally ruled out** two candidate bottlenecks (attention
> dispersion; aggregator choice), **corrected** a wrong "extraction is the wall" claim, and **walked back**
> an over-stated "per-frame supervision → ~1.0" claim. The surviving mechanism is the one in the Executive
> Summary: attention is a **normalized mean (not a sum)** + **over-squashing during consolidation**.

**Findings (why each matters):**
1. **Temporal MMRED tasks added** (`first/last/span_in_room`, gold = frame index; in
   `evaluations/scripts/eval_mmred_text_frames_acc.py`). **mamba ≈ sum on all three, IID *and* OOD** — no
   order-modelling advantage. Cause: **frame position leaks into every per-frame rep** (positional encoding
   + a visible step marker), so an order-blind sum recovers first/last/span without sequential modelling.
   ⇒ MMRED temporal does **not** motivate mamba over sum.
2. **Position-leakage probe.** Frame-index is linearly decodable from *one* L19 rep at **1.000** (joint).
   Removing the step marker collapses *multipass* (single-frame) decodability to **0.145 ≈ chance** but
   **joint stays 1.000** ⇒ in a normal forward, position comes from the **positional encoding**, not just the
   marker. No clean MMRED setup makes order-modelling *necessary* (sum always has position).
3. **Extraction is NOT the rooms/co-occ bottleneck (corrects earlier claim).** The adapter's built-in
   `extraction_p`≈0.558 is a **buggy metric** — it labels a *random* pair while the question conditions on the
   *queried* pair. A correct probe (queried pair) gives **co-occ 0.98 / rooms 0.98 (joint) → ~1.0 (multipass)**.
   Per-frame info is clean; low task acc is downstream (readout/training), not perception.
4. **★ Softmax dispersion is causally NOT the bottleneck.** Rescaling attention temperature — globally **and**
   surgically on just the (question→frame) block — is **null**: β=1 baseline is best, sharpening hurts, and the
   seq8 collapse (~0.0–0.12) is **immune** to temperature. Reason: temperature changes attention *sharpness*,
   but counting fails because softmax is *normalized* (a convex **mean**, not a **sum**) — no temperature turns a
   mean into a sum, and sharpening attends to *fewer* frames (opposite of summing all). ⇒ **no zero-shot
   attention-temperature fix exists**; the fix must be an external *extensive* aggregator that bypasses the
   normalized, over-squashing consolidation.
5. **Per-frame supervision: real but modest — earlier "→~1.0" over-stated.** Correct pairing is the
   **additive (Σσ) readout** (a CE pairing was a mis-test that *hurt*). With additive: co-occ add
   **0.154→0.346**, steps add **0.154→0.216** (lm 0.333→0.463) — fsw helps consistently but **does not reach
   ~1.0** in this **seq8-only** (max-crowding) regime, and the **steps positive control did not reproduce** the
   SPEC's 0.52→0.996. ⇒ that 0.996 is **regime-specific** (easier seq-lens / multipass), not robust at high N.

**Lit sweep (4 parallel agents; arXiv ids verified by the agents, re-verify before thesis):** two 2026 papers
match our phenomenon. **Garcia, "The Right Answer, the Wrong Direction"** ([arXiv:2605.03258](https://arxiv.org/abs/2605.03258)):
count linearly decodable (R²>0.99) but ~orthogonal to the digit unembedding (|cos|≤0.032); readout-only patch
→ **0% in free generation**, LoRA Q/V → **83%** ⇒ must realign *upstream routing*, not just readout (validates the
adapter direction; explains our +0.1 digit-fix ceiling). **Liu** ([arXiv:2605.05715](https://arxiv.org/abs/2605.05715)):
**decodable ≠ correctable** — fixed residual steering gives Δ≈0 (error dir 85–88% overlaps task computation) ⇒
**zero-shot steering cannot set a precise count**. Working zero-shot path: **decompose-and-aggregate** (per-frame
query → external reduce; **System-2 Counting** [arXiv:2601.02989](https://arxiv.org/abs/2601.02989)). Most promising
*no-per-frame-label* lever: **additivity / subset-sum self-supervision** (`count(A∪B)=count(A)+count(B)`; MATT
[arXiv:2003.00164](https://arxiv.org/abs/2003.00164), Noroozi [arXiv:1708.06734](https://arxiv.org/abs/1708.06734)) — a literature gap.

| Date | Output dir | Method / change | Key config | Metric | Status | Notes |
|------|------------|-----------------|------------|--------|--------|-------|
| 2026-06-27 | `outputs/frame_axis/adapter_live/temporal/{first,last,span}_in_room_{mamba,sum}_20260627_153222/` | **temporal tasks: mamba vs sum** (first/last/span, gold=frame index) | 7B nf4, frame-pool mean, LM-CE, train seq5–8, IID + OOD seq4, marked data (mmred_images_park) | macro-acc IID/OOD: first .71/.91(m) ≈ .70/.96(s); last .75/.81(m) ≈ .78/.75(s); span .54/.63(m) ≈ .55/.60(s) | ✅ | mamba≈sum on all 3 (no order edge); both ≫ majority floor. Position leaks via PE+marker ⇒ sum suffices. raw-acc inflated by prior skew (first~.48@1, last~.50@8) → macro reported. |
| 2026-06-27 | `outputs/frame_axis/cache_{mp_compare,nosm}/` (`position_leakage_probe.py`) | **position-leakage probe** (frame-index from one rep) | linear probe, seq8, steps reps, joint vs multipass | marked: joint 1.000 / mp 0.999; **no-marker: joint 1.000 / mp 0.145** | 📊 | Position is in the rep via PE (joint) regardless of marker; marker only load-bearing for single-frame/multipass. |
| 2026-06-27 | `outputs/frame_axis/cache_mp_compare/` (`mp_extraction_rooms_cooc.py`) | **per-frame extraction re-check** (queried pair) | linear probe, seq8 | co-occ 0.983/AUC.998 (joint)→0.998 (mp); rooms 0.976→1.000 | 📊 | **Corrects** adapter's buggy `extraction_p`=0.558 (it labels a *random* pair). Extraction clean ⇒ NOT the bottleneck. |
| 2026-06-27 | `logs/attn_temp_sweep-109314.out`, `logs/attn_temp_frame-109332.out` | **★ attention-temperature causal test (dispersion)** | frozen 7B, digit-logit readout, β∈{0.5,1,2,3,4}, blunt(all-attn) + targeted(q→frame), seq1–8 | **null**: β=1 best (blunt .325 / targeted .205); seq8 ~0.0–0.12 for all β | ✅ | Sharpening (global *or* just q→frame) does NOT recover aggregation ⇒ dispersion not causal. Root = normalized-mean + over-squashing. Also gives native digit-readout ceiling (~.33 overall, ~.12@seq8). |
| 2026-06-27 | `outputs/frame_axis/adapter_live/framesup_additive/{co_occupancy,steps_in_room}_add_fsw{0,1}_*/` | **per-frame supervision (additive readout)** | 7B nf4, sum agg, additive Σσ readout, seq8-only IID, balanced data | add acc: co-occ 0.154→**0.346**; steps 0.154→0.216 (lm .333→.463) | ⚠️ | fsw helps consistently but **modest**; does NOT reach SPEC's ~1.0; steps control failed to reproduce ⇒ 0.996 is regime-specific. Earlier **CE-pairing** mis-test (`adapter_live/cooc_framesup/*`, 109163/164) *hurt* (0.52→0.38) — wrong readout, discard. |

---

## 2026-06-28 session — message-sum decodability (Stage 1): evidence↔non-evidence interference and normalization are *each* sufficient to squash the count

> **Setup.** We construct the aggregate *ourselves* (S = Σ per-frame messages) over cached **L19** per-frame
> reps (joint pass, `steps_in_room`, seq8), so this is independent of the model's attention routing
> (that's the Stage-2 flow diagnostic). Two datasets: **crowded** = `mmred_images_park` (4–5 char/frame),
> **decrowded** = `mmred_steps_balanced` (1 char/frame); 800 ex each, gold ≈ uniform 0–8. Linear probe =
> StandardScaler→Ridge, round-to-int acc + R² (5-seed CV). Script: `experiments/glstm/probe_message_sum_decodability.py`;
> caches: `outputs/frame_axis/probes/message_sum/cache_{crowded,decrowded}/minimal_L19_steps_in_room.pt`.
>
> **Headline.** (1) Sum of **evidence-only** frames decodes the count **perfectly** (1.000), sum of
> **non-evidence-only** frames decodes *its* count **perfectly** (1.000), but the **mixed** sum of all 8
> collapses to **0.45** — the two count-subspaces are not linearly separable once superposed in one channel.
> (2) **SUM** of evidence frames decodes count 1.000 vs **MEAN** 0.42 — normalization (÷count) destroys the
> extensive signal; the count lives in the **magnitude**. Both crowded≈decrowded → at L19, per-frame
> crowding is *not* the sum-decodability limiter. The model's U-shaped *answer* accuracy did **not**
> reproduce in linear sum-decodability (S_all per-count is flat ~0.4–0.55) → U-shape is a downstream
> readout/boundary effect, not sum interference. *Caveat:* S_evid=1.000 is partly the DeepSets extensive-count
> property (sum of homogeneous evidence vectors → ‖·‖∝g); the informative result is S_all's collapse.

| date | path | experiment | config | result | flag | notes |
|---|---|---|---|---|---|---|
| 2026-06-28 | `outputs/frame_axis/probes/message_sum/20260628_193258/` (`probe_message_sum_decodability.py`) | **Exp1 prefix superposition curve** (running count from Σ first j frames) | L19 reps, j=1..8, crowded+decrowded | round-acc **0.99→0.45** (j 1→8); **R² flat ~0.88**; MAE 0.13→0.67; crowded≈decrowded | 📊 | Count stays *linearly present* (R² const); only discretization precision degrades (~0.08 MAE/frame). |
| 2026-06-28 | same run | **Exp2a evidence/non-evidence interference** (decode gold g, seq8) | S_all vs S_evid vs S_nonev (sum) | crowd: S_all **0.448** / S_evid **1.000** / S_nonev→(8−g) **1.000**; decrowd 0.465 / 1.000 / 1.000 | 📊 | Each population's count is perfect in its *own* sum; superposing them in one channel destroys separability. The mixing bottleneck, measured. |
| 2026-06-28 | same run | **Exp2b normalization test** (decode g, g≥1) | SUM_evid vs MEAN_evid | crowd SUM **1.000** / MEAN **0.422**; decrowd 1.000 / 0.425 | 📊 | Dividing by the (per-ex varying) count squashes the extensive signal → softmax-style average is *sufficient* to lose "how many". |
| 2026-06-28 | same run | **per-count U-shape** (S_all, S_evid by true g) | seq8 | S_all per-g **flat ~0.40–0.55** (no U); S_evid **1.00** all g | 📊 | Model's U-shaped accuracy is **not** explained by linear sum interference → downstream readout/boundary effect. |
| 2026-06-28 | same run | **reference: real last-token L19 rep → gold** | StandardScaler→Ridge | crowd acc **0.30** / R² 0.78; decrowd 0.33 / 0.80 | 📊 | Count present in graded form (R²~0.8) but poorly discretized — mirrors the model's own weak accuracy. |

---

---
## 2026-06-28b — CORRECTED mechanism: the count is read off a ~1%-magnitude direction; **fixed set-size + per-frame noise + normalization** are three stacked causes (refines the "interference" framing above)

> The [2026-06-28] entry called S_all's collapse evidence/non-evidence "interference". Follow-ups
> (`probe_message_sum_mechanism.py`, `probe_nonlinearity_ceiling.py`, `probe_aggregation_decomposition.py`,
> + the Garcia readout-alignment and the now-bug-fixed GAIN causal test) **correct and sharpen** it.

**Decomposition** (L19 crowded; per-frame rep mₖ = μ_all + sₖ·δ + εₖ, sₖ=+1 evidence/−1 not; `probe_aggregation_decomposition.py`):

| ‖μ_all‖ | ‖δ‖ | ‖δ‖/‖μ_all‖ | σ_within (count axis) | per-frame SNR | corr(‖S_evid‖,g) | corr(‖S_all‖,g) | corr(proj_δ S_all,g) | pred S_all SNR/count |
|---|---|---|---|---|---|---|---|---|
| 87.0 | 1.0 | **0.011** | 6.0 | **0.33** | **+1.000** | **−0.14** | +0.80 | **0.12** |

**The one correct story.** Two ways to read a count: (1) **magnitude** — ‖Σ of g near-identical frame vectors‖ ∝ g (what S_evid uses; corr +1.000; huge SNR); (2) a **tiny direction δ** (1.1% of the shared mean). The real task sums a **fixed 8 frames**, pinning the magnitude (corr ‖S_all‖,g = **−0.14**, blind to g) → the count is forced onto δ: S_all = 8μ_all + **(2g−8)·δ** + Σεₖ. That term is clean & linear (corr 0.80, R² 0.89) but SNR 0.12 (δ=1 vs σ=6 over 8 frames) → can't round → acc 0.45. **Distractors do NOT cancel** (the −δ gives clean (2g−8)δ); test A (add a *fixed* m distractors → acc stays **1.000**) proves distractors are harmless — only **fixing the set size** (constant magnitude) hurts. Three stacked causes: (i) fixed cardinality kills the magnitude readout; (ii) per-frame noise σ≫δ kills the direction's SNR; (iii) the model's *mean* (Σα=1) further divides δ by N — but even a perfect *sum* only reaches 0.45, so normalization is **one of three**, not the whole.

| date | experiment | result | flag | notes |
|---|---|---|---|---|
| 2026-06-28 | **nonlinearity ceiling** (`probe_nonlinearity_ceiling.py`) | per-frame **sigmoid-then-sum 0.73** vs linear-on-S_all 0.45 vs **MLP-after-sum 0.34** vs last-token 0.30 | ✅📊 | fix = per-frame nonlinearity **before** the sum (restores count-as-magnitude); after summing, per-frame identity is gone (MLP can't recover). |
| 2026-06-28 | **K-hub / distractor dilution** (`probe_message_sum_mechanism.py`) | dilution acc **1.000** for m=0..6; K-hub flat (K8 0.44, p≫n confound) | 📊 | adding fixed distractors harmless → confirms fixed-set-size, not conflict, is the cut. |
| 2026-06-28 | **layer sweep** (`probe_message_sum_layersweep.py`) | S_all R² peaks **L19 0.89**, **L27 0.86**; last-token R² peaks L21 0.82, **always below the sum** through L27 | 📊 | model's aggregate never forms as clean a count as a plain sum → normalization+dilution gap, not incomplete-aggregation. |
| 2026-06-28 | **Garcia readout-alignment** (image, n150; `probe_count_readout_alignment.py`) | reg_r2 L19 **0.85** but align_max **0.008** (~orthogonal to digit rows); realign ceiling cls 0.37 | ✅📊 | "right answer, wrong direction" reproduced on our task; readout misaligned AND SNR-capped (two stacked problems). |
| 2026-06-28 | **GAIN causal test** (`denom_gain_vs_temp.py`, frames-first; earlier crash was a bug) | baseline 0.65→0.25 (sl2→8); **gain×N → 0.00 at sl6/8**; temp only hurts; dilution slope ~0.4 | ✅ | un-normalizing attention (×N) does NOT fix counting and hurts at high N → no in-place attention fix; need external extensive aggregator. |

---

## 2026-06-28c — N-sweep over-squashing curve, δ rotates across depth, and TWO corrections (decrowded-mislabel; frames-first relocates the bottleneck to the carrier)

**N-sweep (S_all decode vs N, crowded L19):** acc **0.85 / 0.63 / 0.58 / 0.45** for N=2/4/6/8, R² flat **~0.85** throughout; the model's own last-token tracks it (0.75/0.53/0.42/0.30). Over-squashing as a measured curve — signal stays present (R²), precision degrades with N. (`probe_message_sum_decodability.py` on `cache_ns_seq{2,4,6}`.)

**δ rotates across depth (`probe_delta_stability.py`, layersweep cache):** the count axis is NOT fixed — L19↔L21 cos 0.80 but **L13↔L27 cos 0.12** (≈orthogonal). ‖δ‖/‖μ_all‖ and per-frame SNR **peak at L19–21 (0.33)** and *decline* to L27 (0.18) → deep layers make the count *less* accessible; inject/read at **L19–21**.

**⚠️ Correction 1 — "decrowded" was never decrowded.** `mmred_steps_balanced` is **5 chars/frame**, same as `mmred_images_park` (`probe_sigma_vs_crowding.py`: both show crowding=5 only). So **crowded≈decrowded in every Stage-1 result because both are 5-char** — Stage 1 **never varied crowding**, and "crowding isn't the limiter" is unsupported by our data. A real 1-char cache (`mmred_steps_1char`) is needed.

**⚠️ Correction 2 — frames-first (deployed) relocates the bottleneck to the carrier.** Re-extracting reps in the deployed frames-first layout (`cache_frames_first_reps.py`):

| metric | question-first (Stage 1) | frames-first (deployed) |
|---|---|---|
| corr(‖S_all‖,g) | −0.14 | −0.20 (magnitude-blind **replicates**) |
| ‖δ‖/‖μ_all‖ | 0.011 | 0.007 (tiny **replicates**) |
| MEAN_evid vs S_evid | 0.42 vs 1.0 | 0.22 vs 1.0 (normalization **replicates**) |
| sigmoid-then-sum (per-frame) | **0.73** | **0.20** |
| model last-token acc | 0.30 | **0.46** |

The over-squashing mechanism (magnitude-blind, tiny-δ, normalization) **replicates** on frame-token reps. BUT in frames-first the frame tokens precede the question → **query-blind**, so per-frame extraction from frame tokens collapses (0.73→0.20), while the **last token is better** (0.46) because query→frame binding + aggregation happen at/after the question. **⇒ in the deployed model, aggregation is at the CARRIER (question tokens after the frames), not the frame-token reps Stage 1 probed.** Stage-1's "clean 0.96 per-frame extraction, aggregation is the wall" is **specific to the query-conditioned (question-first) probing layout**; the carrier aggregation in deployment is not yet characterized (next: carrier-message decomposition in frames-first).

---

## 2026-06-30 — the fix is a SHARP per-frame gate + a frame-isolation mask; "÷N suppresses count" was wrong; one masked forward ≈ multipass

**Amplification experiment** (`probe_amplification.py`, crowded L19, fixed N=8) — corrects the normalization story:
| read | acc | takeaway |
|---|---|---|
| sum_all → count | 0.439 | — |
| **mean_all → count** | **0.439** | **≡ sum** → dividing by N is **benign at fixed N** (the earlier "0.42" was MEAN_evid=÷g, a different case) |
| linear amp along evidence dir (→mean) | 0.44→**0.52** (λ=4..20) | linear "louder" plateaus — can't beat the noise |
| **nonlinear gate** Σσ(κ·score), κ=0→20 | 0.09 → **1.00** | sharp per-frame commitment is THE lever; **SUM≡MEAN×N at every κ** |

- **At fixed N the bottleneck is per-frame SNR, NOT normalization.** mean≡sum; the softmax mean is fine. ÷N only hurts for *varying* N (length-OOD), where it conflates count with length.
- **Sign vs magnitude resolves the "0.96 per-frame but 0.33 SNR" paradox:** per-frame *classification* needs only the SIGN (reliable, 0.96); the linear SUM uses the noisy *magnitudes* (SNR 0.33) → fails. The sigmoid threshold converts magnitude→sign before summing, and **per-frame errors cancel in the sum** (a sharp gate beats the soft one; κ=20 → ~1.0 on the 800-cache).
- **Why sigmoid (not MLP / not linear):** per-frame evidence is *linearly separable* (0.96), so a linear score + a fixed threshold suffices (MLP = overkill); a linear probe (no threshold) caps at 0.45. The sigmoid is the *operation*; the model's MLP is the *machinery* for it.

**Frame-isolation mask** (`frame_isolation_diagnostic.py`, crowded, question-first; block-diagonal so each frame attends only to itself + the question, monkeypatched SDPA) — the structural extraction fix, **ONE forward, no training**:
| 1 forward (n=300) | per-frame extraction | sum-count |
|---|---|---|
| joint (normal attn) | 0.942 | 0.657 |
| **masked (isolated)** | **0.995** | **0.952** |

Cross-frame attention *was* contaminating per-frame reps; isolating frames recovers **multipass quality (≈0.93) in a single forward**. Per-frame extraction → 0.995. So "look at each frame by itself → sum" is structurally validated; the mask cleans the per-frame *signs*, the sum reads the count.

| date | experiment | result | flag | notes |
|---|---|---|---|---|
| 2026-06-29 | **LoRA fix** (mlp/attn/both/V2, L8-18, question-first; `lora_sft_baseline.py`) | **1-char: all arms → TEST 1.000** (flat per-count). crowded val: attn 0.98 / both 0.98 / **V2(mlp+BCE) 0.97** / **pure-mlp 0.77** | ⚠️ | decrowded counting solved; crowded high but **val=1.0-after-1-epoch ⇒ memorization-suspect** (no OOD run). attn≥mlp (recruits frozen MLPs by routing); per-frame BCE rescues mlp (0.77→0.97). |
| 2026-06-29 | **OOD count-holdout** (train ≤4 / test 5-8) | **cancelled externally before results** | — | expectation: native answer-CE head caps → ~0 OOD (learned readout can't emit unseen counts). Re-run needed. |
| 2026-06-30 | **in-model solution** = isolation mask + MLP-LoRA + per-frame BCE (`lora_sft_baseline.py --frame-isolation`) | smoke (1-char, 2 ep): **TEST 0.125** (undertrained, n=76); **proper crowded 5-ep run + no-mask control RUNNING** | ▶ | mask makes the frozen downstream off-distribution → native verbalization is the hard part; structural mask+Σσ already gives 0.95 without training. |

**TWO validated solutions (both confirm the per-frame-gate mechanism causally):**

| solution | IID | OOD (train ≤4 / test 5–8) | training | notes |
|---|---|---|---|---|
| **frozen + isolation mask + soft Σσ readout** (`probe_mask_sigma_ood.py`) | **1-char 1.00 / crowded 0.95** | **1-char 1.00 / crowded 0.82** | tiny head only | **1-char: perfect count AND perfect OOD extrapolation** (masked per-frame=1.000, count=1.000); learned linear-sum collapses OOD (1-char 0.21 / crowded 0.11). crowded 0.82-OOD gap = extraction ceiling (5-char superposition), not aggregation. masked≫joint. |
| **native LoRA: MLP+BCE, no mask** (`lora_sft_baseline.py`, crowded, 5ep) | **0.993** | (untested) | LoRA 14M | model **verbalizes natively**; decomposition off→on: SNR **0.40→2.26**, sigmoid-then-sum **0.56→0.98**, last-token count **0.21→0.99** → won *through* the per-frame gate, not a black box. |

- **The isolation mask HELPS the frozen+probe path but HURTS the native LoRA** (crowded 0.993 no-mask → 0.889 with mask; the mask makes the frozen downstream off-distribution, so the LoRA sharpens less: SNR →1.02 vs →2.26). ⇒ mask for the probe path; no mask for native.
- **Per-frame BCE is load-bearing for native:** pure-mlp (answer-CE) lagged 0.77; +BCE → 0.99, and the decomposition shows it sharpened the gate (causal confirmation of the whole decomposition story).
- **Open:** native-LoRA OOD (the gate it learns is extensive, but the verbalization head may cap). Frozen+Σσ already extrapolates (0.82); the native-OOD path if it caps is an **iterated/scratchpad decode**.

---

## 2026-06-30 — Method comparison: per-frame SNR vs count accuracy (the consolidated table)

> per-frame SNR = Fisher/d′ of evidence-vs-non-evidence frame reps along the count axis (`probe_aggregation_decomposition.py`);
> for LoRA rows it's adapter-OFF→ON. "native" acc = the model generating the answer; "readout" acc = a head on
> frozen reps. ⚠️ regimes differ (crowded 5-char vs clean 1-char/filler) — compare *within* a regime; SNR is "—"
> where a method reads frozen reps without changing them. **Jacobian control:** per-frame ∂(gold-logit)/∂(frame)
> is uniform ~1/8 and >0 for every frame (`frame_jacobian_sensitivity.py`) → info propagates; the wall is SNR, not over-squashing.

| method | regime | per-frame SNR | count acc IID | OOD (unseen counts) | mechanism |
|---|---|---|---|---|---|
| **frozen base (native readout)** | crowded 5ch | **0.33** | ~0.30 | caps | the failure |
| frozen base (native) | clean 1ch | **0.74** | 0.58 | — | decrowding ≈ 2× SNR |
| *— readouts on frozen reps —* | | | | | |
| linear sum (Ridge) | crowded | 0.33 | 0.45 | 0.11 | can't threshold |
| sigmoid-then-sum (gate) | crowded | 0.33 | 0.73 | — | per-frame nonlinearity |
| sum aggregator | filler-count | — | 0.98 | 0.88 (len) | extensive |
| **mamba aggregator** | filler-count | — | **0.99** | **0.93 (len)** | extensive, order-robust |
| mamba (w/ distractors) | selection | — | 0.65 | — | selection-limited |
| **DeepSets Σσ (per-frame-sup)** | clean | — | 0.95 | **0.996 / 1.000 / 0.974** | the extrapolating fix (steps/rooms/cooc) |
| **frozen + mask + Σσ** | crowded | 0.99 extract | **0.95** | **0.82** | structural, 1 forward |
| frozen + mask + Σσ | clean 1ch | 1.0 extract | **1.00** | **1.00** | — |
| *— LoRA on the model (native verbalize) —* | | | | | |
| LoRA attn | crowded | 0.41→**2.70** | 0.96 | caps | recruits frozen MLPs |
| LoRA mlp (answer-CE) | crowded | 0.40→2.1–2.4 | 0.87–0.96 | caps | |
| **LoRA mlp + per-frame BCE** | crowded | 0.40→**2.26** | **0.993** | **0.035** | native fix; mechanism-confirmed (last-tok count 0.21→0.99) |
| LoRA mlp + BCE + mask | crowded | 0.40→1.02 | 0.89 | — | mask hurts native verbalization |

**Reading:** (1) every working fix **raises per-frame SNR ~6–7×** (0.4→~2.7) — SNR is the lever, and both MLP and attn LoRA reach it (attn by routing through frozen MLPs). (2) **IID and OOD dissociate at the readout:** native LoRA gets 0.99 IID but **0.035 OOD** (capped head), while a fixed extensive Σσ read gets 0.82–1.0 OOD. (3) **decrowding (0.33→0.74 SNR) and the isolation mask** both lift the frozen-readout count without touching weights. (4) mamba/sum extrapolate on *clean filler-counting* but collapse with distractors (selection, not aggregation).

---

## 2026-07-01 — Consolidated formal account: the decomposition, the SNR, and why no post-sum readout recovers the count

> This is the write-up of the **mechanism math** behind the numbers above (the "why", not new runs). Every
> empirical value cited traces to a run already in this log: the decomposition/SNR to `probe_aggregation_decomposition.py`
> (2026-06-28), the SNR-vs-acc rows to the 2026-06-30 method-comparison table, the SNR/√N collapse to
> `probe_snr_collapse.py`, the Jacobian control to `frame_jacobian_sensitivity.py`, and the Phase-2 co-occ/rooms
> generalization below to the LoRA sweep (`lora_sft_baseline.py --task {co_occupancy,rooms_visited}`, 5-char balanced).

**The decomposition (standard LDA / signal-detection form).** For frame `k` with rep `m_k ∈ R^d` (mean-pooled
visual tokens, L19) and evidence indicator `e_k ∈ {0,1}`:
`m_k = μ + e_k·δ + ε_k`, where `μ` = shared "a-frame" component, `δ = μ_E − μ_N` = evidence-direction (difference
of class means), `ε_k` = zero-mean within-class noise. The answer is the count `g = Σ_k e_k`. This is the
class-conditional Gaussian model underlying **Fisher LDA (1936)** and **signal-detection d′ (Green & Swets 1966)** —
the *form* is textbook; what is ours is the *measurement* that `μ` dominates (‖μ‖≈87) while `δ` is tiny (‖δ‖≈1,
≈1% of ‖μ‖) with per-frame spread σ≈6.

**SNR, defined + measured.** Along the unit separating direction `û = δ/‖δ‖`, project each frame `p_k = m_kᵀû`:
`SNR = ‖δ‖/σ = |μ_Eᵀû − μ_Nᵀû| / [½(std(p_k|e=1)+std(p_k|e=0))]` = distance between the two class means ÷
within-class std, along the axis that separates them. This is exactly Cohen's d / d′ / √(1-D Fisher ratio).
Measured (steps, crowded 5ch, L19): **SNR ≈ 0.33** — evidence and non-evidence frames overlap by ~⅔σ.

**Why a linear sum/mean fails (the SNR/√N law).** The model reads the count from an aggregate with a linear
read-out. Linearity distributes: `wᵀ(Σ_k m_k) = Σ_k wᵀm_k`, so a linear read-out of a sum is *forced* to be a sum
of per-frame scores. Substituting the decomposition: `wᵀS = N·(wᵀμ) + g·(wᵀδ) + Σ_k wᵀε_k`. The conditional mean
is affine in `g` (count is linearly *present* → R²≈0.85), but the noise is a sum of N independent terms → std
grows as σ_w·√N. Signal-per-count = ‖δ‖, noise = σ√N, so **SNR_per_count = ‖δ‖/(σ√N) = SNR/√N ≈ 0.33/√8 ≈ 0.12**.
Reliable integer rounding needs ≳2 (P(correct round) ≈ erf(½·SNR_per_count/√2)); at 0.12 it is near chance →
count acc 0.45. `probe_snr_collapse.py` confirms acc collapses onto one curve vs **SNR/√N** (not raw SNR). The
**mean is not different**: mean = S/N, a constant rescale at fixed N → identical SNR_per_count (measured mean≡sum
= 0.44) — normalization is *not* the fixed-N cause; ÷N only hurts across *varying* N (length-OOD).

**Why 1–2 frames don't feel it.** (i) No √N: a single frame needs one decision, made from the *full* d-dim rep
(classifier 0.96), not the 1-D δ axis; the δ-SNR only binds once you sum. (ii) Fewer levels: N=2 has 3 possible
counts → high acc even at low SNR. Measured: count acc 0.85 (N=2) → 0.45 (N=8).

**Why evidence-only ≫ with distractors (magnitude vs direction).** `‖S_evid‖ ≈ g·‖μ‖` (g near-identical big
vectors) → count = **length of the sum**, huge SNR (corr(‖S_evid‖,g) = **+1.00**). `‖S_all‖ ≈ N·‖μ‖ = constant`
(N fixed at 8) → length carries no count (corr = **−0.14**); the count is forced onto the tiny δ residual
`S_all = 8μ + (2g−8)·δ + Σε` (clean, corr 0.80, but SNR 0.12). Distractors are *not* adversarial: adding a *fixed*
m of them keeps acc 1.000 (length still ∝ g+const); only **fixing the total set size** (constant magnitude) hurts.

**Why no post-sum readout recovers it (data-processing inequality).** Any read-out is a function `f(S)` of the
aggregate `S = Σ_k m_k`; by the data-processing inequality `I(g; f(S)) ≤ I(g; S)`, and `I(g; S)` is small because
summing (a) superimposes each frame's signal and its own noise into the *same* scalar (no coordinate left to
separate them) and (b) erases *which* frames contributed — exactly what counting needs. So a nonlinear MLP placed
*after* the sum cannot help: measured **MLP-on-sum = 0.34**, vs the same nonlinearity placed *per-frame before* the
sum `ĝ = Σ_k σ(wᵀm_k)` = 0.73 (soft) → ~1.0 (sharp). This is the **Deep Sets** requirement (Zaheer 2017): a
permutation-invariant set-function must be `ρ(Σ_k φ(m_k))` with a *per-element* nonlinearity `φ` **before** the
sum; a plain sum is φ=id and represents only linearly-separable set-functions. Wagstaff (2019): exact sum/count
needs latent width ≥ set size.

**Corollary — retrain the compressor, not the decompressor.** You *can* fix this by retraining, but only
*upstream* of the sum (raise per-frame SNR): LoRA on the MLP/attention that *produces* the reps moves
**SNR 0.40→2.26 (mlp+BCE) / 0.41→2.70 (attn)** → acc 0.99/0.96. Retraining a read-out on the frozen sum cannot
(MLP-on-sum 0.34). A **mamba** aggregator over *frozen* low-SNR reps is likewise limited (co-occ 0.34, rooms 0.51) —
a smarter aggregator can't un-sum information the low-SNR reps never separated.

**Control — this is aggregation-SNR, not classical over-squashing.** `frame_jacobian_sensitivity.py`: per-frame
∂(gold-logit)/∂(frame-input) is uniform ~1/8 and >0 for every frame → information *propagates* fine; the wall is
the readability of the sum (SNR), not connectivity/Jacobian attenuation.

**Generalization (Phase-2, 5-char "crowded" = 5 chars/frame; `mmred_{cooc,rooms}_balanced`).** The SNR bottleneck
and the LoRA fix reproduce on both other task families:

| task | frozen SNR (5ch) | frozen count-sum acc | LoRA mlp+BCE | LoRA attn | LoRA both |
|---|---|---|---|---|---|
| co-occupancy | **0.66** | 0.487 | 0.975 (SNR 0.46→**4.10**) | 0.982 (SNR 0.46→**4.55**) | 0.982 (SNR not logged) |
| rooms-visited | **0.41** (per-room avg) | 0.366 | 0.982 | 0.898 | 0.991 |

mamba over frozen reps: co-occ lm 0.34, rooms lm 0.51 (limited by frozen SNR, as predicted). **Takeaway:** frozen
5-char SNR is low on all three tasks; LoRA raises it ~9× on co-occ (0.46→4.1) → 0.90–0.99; "fix the reps (LoRA)
beats a better aggregator on noisy reps (mamba)"; attn ≥ mlp again.

**One-paragraph summary.** The model perceives each frame (0.96) but reads the count from a linear sum, which is
algebraically forced to add per-frame scores whose signal (δ) is ~1% of the per-frame noise (σ) and is further
attenuated by √N → SNR_per_count ≈ 0.12 (need ~2). It is not the softmax mean (mean≡sum at fixed N) and not
propagation (Jacobian uniform). Evidence-only is easy because there the count is the *length* of the sum; the full
set pins the length at 8, hiding the count in a tiny direction. No post-sum readout can recover it (data-processing
inequality; MLP-on-sum 0.34); the fix must act per-frame *before* the sum — a threshold `Σσ(w·m)`, or LoRA that
raises the per-frame SNR (0.4→2.3–2.7 → 0.96–0.99).

---

## [2026-07-03] Deployed frames-first carrier localization — the evidence carrier is the ROOM token; the deployed bottleneck is aggregation-limited (decode-then-count 2× the model) on top of a *moderate* carrier-extraction ceiling

> **Closes the biggest hole in the SNR story:** the decomposition/SNR (0.33, [2026-06-28]) was measured
> **question-first on frame-token reps**, but the deployed model is **frames-first and aggregates at the
> carrier** (the [2026-06-28c] correction). This session measures the decomposition *where the model
> actually aggregates*. Scripts: `experiments/glstm/cache_message_sum_layersweep.py --frames-first`
> (frame-token sweep) + `experiments/glstm/probe_delta_stability.py` (extended: prints ‖μ‖,‖δ‖,σ,SNR);
> `evaluations/scripts/patch_importence/probe_frame_to_carrier_message.py` (extended: `--carrier
> {last,all_question,per_token}`, `--max-offset`, `--decode-offsets`, `--sample-seed`; new sections
> **(D)** message-decomposition, **(E)** per-carrier-token SNR sweep, **(F)** direct count-decode, **(G)**
> decode-then-count). Data: `mmred_images_park/seq_len_8/all_uniform` (steps_in_room, crowded 5-char), 7B nf4.

**1. Frame-token FF decomposition sweep (L0–27, mean-pooled; `outputs/frame_axis/probes/message_sum/cache_layersweep_ff_early/` + `…_ff_crowded/`, job 115572).**
Per-frame SNR is **low everywhere in the deployed layout — 0.10–0.18 across mid/late layers, peak only 0.175 @L19** — about **half** the question-first 0.33. ‖δ‖/‖μ‖ stays ~0.5–0.9% (the tiny-direction claim replicates). Frame tokens are **query-blind** frames-first, so their "evidence direction" is only a query-independent confound (L0 spikes to 0.67 with tiny σ — a visual shortcut, control still pending). ⇒ **the frame-token reps are NOT the deployed aggregation locus; move to the carrier.**

| layer | 0 | 8 | 12 | 16 | **19** | 20 | 24 | 27 |
|---|---|---|---|---|---|---|---|---|
| per-frame SNR | 0.67⚠ | 0.11 | 0.14 | 0.13 | **0.175** | 0.17 | 0.10 | 0.10 |

**2. Frame→carrier message decomposition (`outputs/frame_axis/probes/carrier_message/count_last`, `…/count_allq`; jobs 115579/115580).**
The per-frame *message* into the carrier (`msg_{f→c}=o_proj(Σ_{j∈f} A[c,j]v_j)`) is far more separable than the frame tokens. carrier=**last** peaks **0.42 @L20**; carrier=**all_question** peaks **0.79 @L14–16** (averaging over question tokens *denoises* → evidence is distributed across several tokens). But even the best SNR/√N ≈ 0.28 ≪ 2, and the count from the message-sum sits at majority ⇒ aggregation bottleneck reproduces **at the carrier**. Per-frame evidence itself is decodable from messages (AUROC 0.73–0.90).

**3. Per-carrier-token SNR sweep (E), MAXOFF=20 (`…/count_per_token_wide/`, job 115639).**
**The carrier is the queried ROOM token** (offset −9): per-frame SNR **1.46 / 1.58 / 1.43 @L14/16/20** — 3–7× the last token and every other question token. The **character token** (offset −13) is a **weak secondary** (~0.51); `?` (−8) and `<|im_start|>` (−2) are deep-layer sinks. Reading the single room token beats both the all-question mean (0.79) and the last token (0.42). Question decodes as *"How many steps did &lt;Char&gt; spend in the &lt;Room&gt;?"*

**4. ⚠️ Sampling bug caught + fixed.** `iter_sample_dirs` returns samples **count-grouped**; taking the first `--limit` **unshuffled** gave a degenerate **{0,1}-only** count subset (majority 0.67), which invalidated the first count-decode (its "0.70" was a majority artifact, not signal). Fixed with a seeded shuffle (`--sample-seed`) → **balanced 0–8** (majority 0.13). All numbers below are post-fix.

**5. Balanced n=400 count-decode (`…/count_per_token_balanced/`, job 115655).** model own-answer **0.207**, per-frame evidence AUROC **0.955**. Room SNR is **1.16** (honest; the skewed {0,1} subset had inflated it to 1.58); char SNR **0.67**.
- **(F) direct decode:** room-token `_sum` decodes the count **0.28–0.30 @L16–20** (>2× majority 0.13; > model 0.21) — the count *is* present in the summed room messages. ⚠ `sum > concat` everywhere is a **probe-capacity artifact** (concat is 8× the dim, underpowered @n=400), **not** an information difference — do not read the sum-vs-concat contrast here.
- **(G) decode-then-count** (per-frame classifier at the room carrier → sum predictions; well-powered, ~3200 per-frame examples):

| layer | 8 | 12 | 14 | **16** | 18 | 20 | 22 | 24 |
|---|---|---|---|---|---|---|---|---|
| dtc_acc (room) | 0.331 | 0.281 | 0.366 | **0.480** | 0.271 | 0.419 | 0.307 | 0.247 |
| dtc_MAE | 1.11 | 1.23 | 1.02 | **0.65** | 1.16 | 0.79 | 1.14 | 1.32 |

  **dtc @L16 = 0.480 (MAE 0.65) vs model 0.207 — more than 2×.** room+char ≈ room (0.476) → the **room token carries essentially all the count-relevant evidence**; the char adds ~0 to *counting* despite its 0.67 SNR (char matters for per-frame evidence *presence*, not for the count).

**Synthesis — a two-part ceiling at the room carrier (stronger than the pure-aggregation story).**
- **Aggregation headroom (model 0.21 → dtc 0.48):** a trivial "classify each frame at the room carrier, then sum" **doubles** the model's accuracy *on the model's own internal messages* ⇒ a **real aggregation gap** the model fails to close. This is the thesis, now localized to the deployed room carrier @L16–20.
- **Extraction ceiling (dtc 0.48 → 1.0):** dtc peaks at 0.48, not ~1.0, because the room-carrier per-frame SNR is only **moderate** (d′≈1.16 → per-frame AUC ~0.80). So the deployed bottleneck is **aggregation-limited AND extraction-moderate**, not "evidence perfect, only aggregation fails."
- **Read/inject site = the room token, L16–20** (where both SNR and dtc peak).

**Methodological corrections logged.** (i) The question-first frame-token SNR **0.33 is not the deployed number** — frame tokens are query-blind frames-first; the deployed carrier SNR is **1.16 (room token)**. (ii) The first count-decode was a **sampling-skew artifact** (shuffle fix). (iii) `all_question > last` is real but reflects **denoising** (signal distributed across question tokens), not that the mean is faithful — the single room token is the true carrier.

**Caveats.** dtc's per-frame classifier is *supervised* (upper bound on linearly-extractable-then-summed count); sum-vs-concat is dim-confounded; 5 seeds @ n=400; L0-confound control and a **sigmoid-gate dtc** (how much of 0.48→1.0 is extraction vs a sharper per-frame readout) not yet run; measured on crowded 5-char steps_in_room only (co-occupancy / rooms_visited frames-first not yet done). Formal decomposition/SNR definitions: see the [2026-07-01] entry (Fisher 1936 / Green & Swets 1966 / Cohen 1988 / DeepSets–Wagstaff).

**Runs.** frame-token FF sweep `outputs/frame_axis/probes/message_sum/cache_layersweep_ff_early/` (job 115572); carrier messages `outputs/frame_axis/probes/carrier_message/{count_last,count_allq,count_per_token,count_per_token_wide,count_per_token_balanced}/` (jobs 115579/115580/115581/115639/**115655** canonical).

---

## [2026-07-03b] ⚠️📊 Restatement of the formal account: ad-hoc "SNR" → signal-detection d′ (derivation only, NO new runs)

> Supersedes the *framing* of [2026-07-01] (every number logged there remains valid as measured; what
> changes is which quantity is operative and how the accuracy law is obtained). Motivation — three flaws
> in the SNR story as written: **(a)** SNR was measured along the naive difference-of-means axis δ̂
> (Cohen's d), which *understates* what a linear readout can do — a linear readout is free to whiten
> (Fisher 1936), and the earlier caveat claiming "the sum is restricted to the δ̂ axis" was simply wrong;
> **(b)** the accuracy law was a *fitted* monotone collapse curve gated by an arbitrary "SNR ≳ 2"
> threshold; **(c)** the old account is quantitatively inconsistent with its own numbers: SNR/√N = 0.12
> predicts ~0.15 S_all accuracy (interior 2Φ(0.06)−1 ≈ 0.05, boundary-mixed ≈ 0.15), 3× below the
> measured 0.45 — the discrepancy was papered over by the fitted curve.

**The correction.** Operative per-frame quantity = the whitened (Fisher/Mahalanobis) discriminability
**d′ = √(δᵀΣ⁻¹δ)** — what a trained linear probe actually attains — implied by the measured per-frame
probe: bal-acc 0.94–0.96 ⇒ d′ = 2Φ⁻¹(acc) ≈ **3.1–3.5** (vs the δ̂-axis 0.33). Under the already-stated
class-conditional Gaussian model (`m_k = μ + e_k·δ + ε_k`, `ε_k ~ N(0,Σ)` iid over frames), the best
possible readout of any pooled linear statistic is the matched filter `w ∝ Σ⁻¹δ` + nearest-integer, with
closed-form exact-match accuracy **`2Φ(d′/2√N) − 1`** for interior counts (boundary counts one-sided;
overall = mixture over the count prior). Two measured parameters (d′, N); zero fitted parameters, no
threshold. Because the matched filter is a **sufficient statistic** of the sum for this family, *no*
post-sum nonlinearity can beat it — a theorem under the model, upgrading the [2026-07-01] DPI hand-wave,
and retro-dicting two logged results: MLP ≈ linear on the aggregate (0.437 vs 0.458, readout_ceiling
[2026-06-26]) and MLP-after-sum 0.34 vs per-frame-gate ~1.0 (probe_nonlinearity_ceiling).

**Zero-free-parameter checks against already-logged numbers (derivations — ⚠ pending dedicated runs, E1):**

| check | predicted (d′ from per-frame probe) | measured (logged) | measurement source |
|---|---|---|---|
| Q-first S_all N-sweep, N=2/4/6/8 | 0.86 / 0.62–0.69 / 0.53–0.59 / 0.46–0.52 | 0.85 / 0.63 / 0.58 / 0.45 | `probe_message_sum_decodability` cache_ns (row 6, claim table) |
| deployed dtc @ room carrier L16 (AUROC 0.955 ⇒ p_err ≈ 0.115; iid ±1 errors, cancellation) | ≈ 0.47 | 0.480 | [2026-07-03] §(G), job 115655 |
| deployed optimal-linear-on-summed-messages @ room carrier | ≈ 0.40 | 0.28–0.30 | [2026-07-03] §(F) — **MISS**: suggests correlated cross-frame noise (ρ>0, breaks iid √N) and/or n=400 probe power → E2 |

⚠ Regime/layer matching between the per-frame probe and the N-sweep was not verified when deriving row 1
— E1 must recompute d′ and the sum-decode **on the same cache**. Prediction ranges reflect per-frame
acc 0.94–0.96 and interior-only vs uniform-count-mixture bounds.

**Layout caveat made explicit.** The headline decomposition numbers (0.33 δ̂-SNR, 0.96 per-frame,
S_all N-sweep) are **question-first** and not the deployed pipeline; the deployed frames-first locus is
the **room-token carrier, L14–20** ([2026-07-03]). The theory is layout-agnostic — what changes is where
d′ is measured (frame reps vs carrier messages) and its value; the deployed preliminary checks (rows 2–3)
are the reason to believe the law transfers, and E1 runs the parity test at the deployed locus.

**Lit anchoring (kills the "SNR is arbitrary/unseen in lit" objection).** The machinery is standard:
d′/signal detection (Green & Swets 1966; Macmillan & Creelman), Fisher LDA / Mahalanobis (Fisher 1936),
and **linear Fisher information for population decoding** in computational neuroscience (Averbeck, Latham
& Pouget 2006, Nat Rev Neurosci; Moreno-Bote et al. 2014, Nat Neurosci — information-limiting correlated
noise = the ρ>0 refinement of √N). Our contribution is applying it to frozen-VLM aggregation and
**predicting** the measured ceiling with no fitted parameters.

**Planned validation (all CPU on existing caches unless noted):**
- **E1 — parity plot (the law test):** per regime (task × crowding × layer × carrier token × N), measure
  d′ = Mahalanobis with Ledoit–Wolf shrinkage on cached per-frame messages, predict `2Φ(d′/2√N)−1`
  (count-prior-mixed), independently measure linear-probe-on-sum accuracy on the same cache; one
  predicted-vs-observed scatter. Include weighted-attention refinement (effective N from attention
  weights). Primary locus: deployed room-carrier messages; secondary: question-first frame reps
  (layout-invariance claim).
- **E2 — cross-frame noise correlation ρ:** residuals after class-mean removal, mean pairwise cross-frame
  correlation along w; joint vs isolation-mask caches (does masking shrink ρ? unifies the "single-pass
  superposition" story with the law); refit parity with √(N(1+(N−1)ρ)).
- **E3 — sufficiency check, well-powered:** linear vs MLP probe head-to-head on summed carrier messages;
  prediction: no MLP headroom.
- **E4 — model adequacy:** QQ plots of matched-filter projections per class (Gaussianity), per-class
  variance equality.
- **E5 — causal d′ intervention (GPU, small n):** scale the δ component of room-token messages by λ
  during real forward passes; probe-accuracy should track `2Φ(λd′/2√N)−1`; the model's *emitted* answer
  should lag until the readout is realigned — cleanly separating the aggregation wall from the
  readout-misalignment wall in one experiment.

---

## [2026-07-03c] ✅📊 d′-parity validation RUNS — the law holds (16 question-first regimes AND the deployed room-carrier locus); "superposition" is measurable noise correlation

> Executes E1–E4 of [2026-07-03b]. **Script:** `experiments/glstm/probe_dprime_parity.py` (new; CPU-only
> on existing caches; 60/40 sample-disjoint split × 3 seeds; d′ = gap/width of held-out projections onto a
> shrinkage-LDA (Ledoit–Wolf) direction, cross-checked by √2·Φ⁻¹(AUC); prediction = count-prior-mixed
> `2Φ(d′/2√N)−1` with one-sided boundary counts; measured = held-out RidgeCV-on-sum → round).
> **Runs:** `outputs/frame_axis/probes/dprime_parity/20260703_162729/` (16 Q-first regimes, parity.png)
> and `…/20260703_170355/` (deployed carrier). **Deployed message cache:** job **116866** (31 min, L40S)
> → `outputs/frame_axis/probes/carrier_message/count_msgcache/count/messages_cache.pt` (n=600 balanced,
> L14/16/18/20, offsets 9=room / 13=char; `probe_frame_to_carrier_message.py` gained `--save-messages`).

**E1 — the parity law holds wherever d′ is measurable.** Zero fitted parameters throughout.
Question-first steps N-sweep (d′ constant ≈3.2–3.5 while accuracy collapses exactly as √N dictates):

| regime | N | d′ naive (old "SNR") | d′ whitened | predicted | measured (ridge±std) |
|---|---|---|---|---|---|
| ns2 | 2 | 0.21 | 3.46 | 0.849 | **0.853** ±.037 |
| ns4 | 4 | 0.24 | 3.21 | 0.662 | **0.650** ±.016 |
| ns6 | 6 | 0.64 | 3.34 | 0.575 | **0.593** ±.018 |
| crowd8 | 8 | 0.47 | 3.35 | 0.504 | **0.471** ±.023 |

Task-general: co-occ 5char pred 0.552 / meas 0.475; co-occ balanced 0.814 / 0.798; steps joint (mp-pair)
0.514 / 0.514; frames-first query-blind frame tokens (weak code, d′_w=0.57) pred 0.176 / meas 0.245.
**Deployed locus** (room-token carrier messages, frames-first): room@L14 pred .346/meas .296;
**room@L16 pred .375 / meas .363** (peak carrier — the law lands within ~1pt); room@L20 .350/.396;
char@L16 .212/.311. This **resolves the [2026-07-03b] "MISS"** (0.40-vs-0.28): it was naive-axis d′ +
n=400 probe power, not the law. Full deployed ladder now measured in one place: **model 0.236 <
linear-on-sum 0.36–0.40 (≈ predicted d′/√N ceiling) < decode-then-count 0.47 (report.txt, job 116866) <
1.0** — readout-misalignment wall, then √N aggregation tax, then extraction bound.

**E2 — cross-frame noise correlation: the superposition story IS the noise story.** ρ along the readout
direction: **joint-pass caches ρ ≈ +0.085…+0.13 in every regime; matched multipass (isolated-frame)
caches ρ ≈ +0.004…+0.014** (steps: .085→.014; co-occ: .096→.004; cache_mp_compare pairs). Single-pass
processing measurably correlates per-frame noise; isolation removes it — the quantitative form of the
[2026-06-20b] "single-pass superposition" finding ("information-limiting correlations", Moreno-Bote 2014).
Subtlety: measured accs track the **iid** prediction better than the ρ-penalized one — a full-dim readout
can project out shared-direction correlated noise, so only the matched-filter component is irreducible;
refine ρ measurement along the sum-probe's own direction before using the ρ-corrected law.

**E3 — sufficiency: confirmed 21/21 regimes.** MLP-on-sum never beats linear (Δ = −0.04…−0.58).
Bonus: ridge→round beats multinomial logistic by 10–25pts everywhere — the decoder matching the
equispaced-means geometry wins, as the model demands.

**E4 — model adequacy: good exactly where the law fits.** Trusted regimes: |skew| ≤ 0.15, excess kurtosis
≤ 0.2, class-std ratio 0.89–0.92, d′_gap ≈ d′_AUC (±few %). **Known failure mode:** regimes with per-frame
probe saturated at ~1.0 (char1, cooc de/crowded small-n, multipass) — there d′ is unmeasurable from finite
data (AUC→d′ pegs at its 6.72 cap), predictions are extrapolations and overshoot by 0.1–0.2. The law's
domain of validity is self-diagnosing: quote it only where the per-frame probe is off ceiling.

**Caveats.** Single probe family (ridge/logistic/MLP); Q-first crowd8 vs decrowd8 showed no d′ difference
(both ≈3.3 — at odds with the earlier naive-SNR crowding story; recheck what cache_decrowded actually
contains before citing a crowding contrast from this run); deployed L18/char measured > predicted →
count information in auxiliary channels beyond the single evidence axis (the prediction is a
single-channel account, so measured ≥ predicted flags extra channels, not refutation).

---

## [2026-07-03d] ✅📊 E5 causal dose-response — the readout-misalignment wall shown CAUSALLY: an amplified, last-token-decodable count signal leaves the emitted answer unchanged

> **Script:** `evaluations/scripts/patch_importence/dprime_dose_response.py` (new) + `runners/dprime_dose_response.sbatch`.
> **Run:** `outputs/frame_axis/probes/dprime_dose/20260703_182429/` (job **116926**, A100, 13 min, n=150).
> Intervention: during real frames-first forwards, add `(λ−1)·g·δ` to the residual at the **room-token**
> position entering **L17** (δ = measured evidence direction of the L16 room-carrier messages, ‖δ‖=1.41);
> arms λ=0 (ablate) / 1 (baseline) / 2 / 4 + same-magnitude **random-direction** control (λ=4). Emitted
> answer = candidate-digit argmax; last-token L24 rep cached per arm and probed post-hoc (ridge, 60/40).

| arm | emitted acc | last-token L24 count decode |
|---|---|---|
| oracle λ=0 (ablate) | 0.233 | 0.367 |
| oracle λ=1 (baseline) | 0.227 | 0.367 |
| oracle λ=2 | 0.213 | 0.400 |
| oracle λ=4 | 0.200 | **0.533** |
| random λ=4 (control) | 0.200 | 0.400 |

**Readings.** (1) **The readout wall is causal and severe:** the λ=4 dose propagates room-token → last
token along the δ channel specifically (decode 0.367→**0.533**, +17pts; random control ≈ baseline) —
count information amplified, delivered, and *linearly readable at the last token* — yet the frozen
model's emitted answer does not move (0.23→0.20, within noise at n=150). "Decodable ≠ used" (Elazar
amnesic-probing; Garcia right-answer-wrong-direction), now shown **causally at the deployed locus**: no
amount of aggregation-side signal rescues the answer while the readout is misaligned — fixes must
include R3 (direct/realigned readout), exactly as the [2026-07-03b] framework requires.
(2) **The λ=0 ablation is a leaky null, not evidence the channel is unused:** downstream decodability was
unchanged (0.367) because later-layer messages (L18/20 attention) re-deliver the evidence — a single-site
subtraction can't scrub a channel that is re-written at every layer. A conclusive "load-bearing" test
needs a **multi-layer projection scrub** (remove the δ̂ component of the room-token state at L14–21
continuously, LEACE-style) — logged as the E5b follow-up, not yet run.
(3) Dose–response on the probe side is attenuated by the carrier→last hop (0.53 ≪ the ~1.0 an
uncorrupted injected signal would support), consistent with the weak last-token message path (SNR 0.42,
[2026-07-03]). **Caveats:** n=150 (±~0.07 emitted, ±~0.09 probe); single site/layer; oracle-dosed (uses g)
— a mechanism experiment, not a method.

---

## [2026-07-03e] ✅📊 E5b multi-layer dose + continuous scrub — the δ channel IS load-bearing (scrub ⇒ undercount collapse), yet amplifying it ×16 makes the count 0.70-decodable at the last token with ZERO behavioral gain

> **Run:** `outputs/frame_axis/probes/dprime_dose/20260703_185555/` (job **116938**, A100, 19 min, n=150;
> job 116937 died on an argparse flag collision — runner fixed). Same script, v2: multi-layer arms use
> each layer's own δ_L (L14/16/18/20, dose enters L+1); **scrub** = label-free removal of the δ̂_L
> component of the room-token state at every carrier layer (continuous, so later layers cannot
> re-deliver — closes the [2026-07-03d] leaky-ablation hole); single-site ladder extended to λ=8/16;
> random-direction multi control at λ=8. Last-token L24 reps probed post-hoc (ridge, 60/40).

| arm | emitted acc | MAE | last-token count decode |
|---|---|---|---|
| base | 0.227 | 1.51 | 0.367 |
| single λ=8 / λ=16 | 0.187 / 0.193 | 1.90 / 2.07 | 0.650 / **0.700** |
| multi λ=2 / 4 / 8 / 16 | 0.213 / 0.207 / 0.193 / 0.140 | 1.56→2.19 | 0.433 / 0.517 / 0.650 / 0.650 |
| **scrub (all layers, label-free)** | **0.180** | **2.47** | 0.317 |
| random-dir multi λ=8 (control) | 0.213 | 1.96 | 0.600 (see caveat) |

**Readings.**
1. **The δ channel is load-bearing — scrub verdict.** Continuous removal of the room-token δ̂ component
   collapses the answer distribution into systematic **undercounting**: g≥3 accuracy → ~0 (g3 0.42→0.00,
   g4 0.33→0.04), "g0 accuracy" jumps to 0.86 because the model now always answers low; MAE 1.51→2.47.
   The model behaves as if the evidence mass is gone ⇒ its emitted count **does** transit the room-token
   δ channel. Resolves [2026-07-03d]'s open question: the channel is USED — just read very badly.
2. **The readout wall widens with dose.** Emitted accuracy is flat-to-falling across every dose arm while
   last-token decodability climbs 0.367→0.533→0.650→**0.700** (λ=4→8→16). At high λ behavior *degrades*
   (multi λ=16: 0.140; by-gold mass collapses toward middle/low bins, g7/g8→0) — large doses push the
   carrier off-distribution for the frozen downstream. So the answer to "can we push decodability higher?"
   is yes (0.70 and still rising slowly), and it buys **zero** emitted accuracy — the cleanest possible
   statement of "the remaining wall is the readout."
3. **Multi-layer ≈ single-site at matched λ for decodability** (0.650 both at λ=8) — the last hop, not
   accumulation depth, limits what reaches the last token; multi hurts *behavior* more (bigger drift).
4. **Even a random-axis g-signal is decodable-but-unused:** the λ=8 random control carries g by
   construction (magnitude ∝ g ⇒ probe reads it at 0.600) yet behavior stays at baseline (0.213) — the
   frozen model exploits **no** novel linear g-channel, on any axis, without retraining. (This also means
   the random arm controls direction-specificity of *behavior*, not of decodability.)

**Caveats.** No random-direction **scrub** control yet (scrubbing a random axis should be behaviorally
inert — cheap follow-up); the scrub also removes the constant μ·δ̂ offset (operating-point shift can't be
fully excluded, though the g-dependent collapse pattern is exactly the evidence-removal signature);
n=150; high-λ arms are off-distribution by design.

---

## [2026-07-03f] ✅📊 E5c — random-scrub control (load-bearing verdict now airtight) + the "repaired readout" column: dosed count survives all 28 layers and dies at the unembedding

> **Run:** `outputs/frame_axis/probes/dprime_dose/20260703_192542/` (job **116943**, A100, 19 min, n=150).
> Script v3 additions: `--scrub-random` control (remove a random unit axis instead of δ̂ at every carrier
> layer) and **final-layer last-token capture** (`reps_final`, post-L27, one linear map from the digit
> logits) — a ridge head on it = the accuracy the model would emit with a **repaired (realigned) readout**.
> ⚠ The intended λ=16/32/64/128 ladder did not run in this job — `sbatch --export` splits on commas, so
> `LAMS_SINGLE="8,16,…"` silently became `8` (lesson: export env vars through the shell, never inline
> comma-valued `--export`). High-λ ladder re-submitted as job **116945** (▶ running); append on landing.

| arm | emitted acc (MAE) | decode@L24 | **decode@FINAL = repaired-readout acc** |
|---|---|---|---|
| base | 0.227 (1.51) | 0.367 | 0.283 |
| single dose λ=8 | 0.187 (1.90) | 0.650 | **0.683** |
| multi dose λ=8 | 0.193 (1.99) | 0.650 | 0.667 |
| multi dose λ=16 | 0.140 (2.19) | 0.650 | 0.617 |
| **scrub δ̂ (all layers)** | **0.180 (2.47)** | 0.317 | 0.317 |
| **scrub random axis (control)** | **0.247 (1.50)** | 0.333 | 0.300 |
| random-dir dose λ=8 | 0.213 (1.96) | 0.600 | 0.517 |

**Readings.**
1. **Load-bearing verdict airtight.** Random-axis scrub ≈ baseline (0.247, MAE 1.50 vs 0.227/1.51);
   δ̂-scrub collapses into undercounting (0.180, MAE 2.47, g≥3 ≈ 0). It is *that specific direction*,
   not projection-removal per se, that carries the model's count.
2. **The dosed count survives to the unembedding's doorstep and dies there.** decode@FINAL ≈ decode@L24
   at every dose (λ=8: 0.683 vs 0.650) — no attenuation across the remaining 20+ layers; the frozen head
   then emits 0.187. **With a linear count head the λ=8 model would score 0.68 — a 3.6× gap located
   entirely in the readout.** At base the repaired head is worth only +0.06 (0.283 vs 0.227, ≈ the
   un-dosed linear ceiling) — head-repair alone is modest; dose+repair reveals how much signal the head
   discards.
3. **Multi-site injection adds nothing to decodability (0.650 = 0.650 at λ=8) while hurting behavior
   more** (0.140 at multi-16). Why, in theory terms: (i) the dose is a *noise-free* deterministic
   function of g — one copy is information-saturated, replicas add zero; (ii) RMSNorm + linear probing
   are scale-insensitive — more amplitude along the same axis is the same 1-D signal; (iii) the binding
   constraint is the room→last-token transmission through frozen attention (the SNR-0.42 hop), which
   same-position copies don't widen. Each extra site only pays a fresh off-distribution penalty.
   **Prediction registered in advance for job 116945:** decode saturates across λ=16→128 (pipe full)
   while emitted accuracy keeps degrading.

**Caveats.** n=150 (decode columns n_test=60, ±~0.09); repaired-readout head is fit per-arm on 90
samples (upper-bound flavor, not a deployed artifact); μ·δ̂-offset caveat from [2026-07-03e] still
applies to the scrub arm.

**High-λ ladder landed (job 116945, run `…/dprime_dose/20260703_194553/`, 21 min) — the registered
prediction confirmed.** Single-site λ = 16/32/64/128:

| λ | emitted (MAE) | decode@L24 | repaired readout (@FINAL) |
|---|---|---|---|
| 16 | 0.193 (2.07) | 0.700 | 0.650 |
| 32 | 0.147 (2.25) | 0.717 | 0.700 |
| 64 | 0.133 (2.31) | 0.750 | **0.783** |
| 128 | 0.167 (2.28) | 0.633 | 0.667 |

(1) **Saturation as predicted in advance:** decode plateaus at ~0.70–0.78 across λ=16–64 — a 127×
amplified, noise-free count signal cannot push past ~0.78 at the output position. The plateau *is* the
measured transmission capacity of the frozen room→last-token attention hop (the SNR-0.42 path) — the
pipe, not the payload, is the limit. At λ=128 decode *declines* (0.633/0.667): the dose is so large it
crushes the token's remaining content and drifts the whole forward further off-distribution.
(2) **Emitted accuracy degrades monotonically to 0.133** (by-gold mass collapses onto g≈4; g≥5 → 0) —
the frozen head never benefits at any dose. Peak dissociation at λ=64: repaired readout **0.783** vs
emitted **0.133 — a 5.9× gap, entirely in the unembedding.** (3) The behavioral degradation pattern
(regress-to-middle) matches the carrier going off-distribution, not information loss (decode stays high).

---

## [2026-07-04] ✅📊 d′-theory rollout to CO-OCCUPANCY and ROOMS — distributed carrier found & causally mapped (double dissociation); rooms' structural-vs-statistical split measured; the law + readout wall replicate on both tasks

> Overnight autonomous rollout of the [2026-07-03b–f] program to the two other MMReD tasks, on the
> balanced 5-char datasets. All runs single-GPU (l40s), ~2.2 h total.
> **Jobs:** localization 116997 (cooc, 18 min) / 116998 (rooms, 30 min); message caches 117005 (cooc,
> 616 MB) / 117014 (rooms, 411 MB — first attempt 117006 lost 40 min to a section-(G) single-class crash;
> probe now guards it AND saves the cache *before* any analysis); E5-cooc + H3 117015 (12 min).
> **Script upgrades:** probe gained a multiclass (mean one-vs-rest) per-token d′ map + raw-label saving;
> `probe_dprime_parity.py` gained the **K-channel mode**; dose script v4 (multi-offset scrub sets, --task).

**1 · Carrier localization (per-token × layer d′ maps, n=400).**
- **co-occupancy: the carrier is DISTRIBUTED over the two character-name tokens** — char1 (off−15)
  d′ 1.49@L14, char2 (off−13) 1.29@L14/1.23@L16, "and" between them 1.33@L18, "same" (off−10) 1.15@L16;
  no single dominant token. Model own-answer **0.155**; per-frame same/diff from messages AUROC **0.982**;
  decode-then-count **0.601** (3.6× model). `carrier_message/cooc_locmap/`.
- **rooms-visited: the carrier is the character-name token** (off−10, 1.19@L14/1.22@L16; "visit" 0.82–0.96
  secondary; deep-layer sinks only at L20). Model own-answer **0.087**; per-frame room decode (multiclass)
  **0.947**; decode-then-count **0.803** (**10× model** — the biggest aggregation gap of the three tasks).
  `carrier_message/rooms_locmap/`.

**2 · Deployed parity, co-occupancy (`dprime_parity/20260704_011808/`, 3 seeds).** Single-token loci
UNDER-predict everywhere (char2@L16 pred .412/meas .497; char1@L18 pred .259/meas .489; same@L16 pred
.354/meas .500) — the distributed-carrier signature. **The block readout closes it: two-name concat @L14
has block-d′ 3.19 → pred 0.456 vs meas 0.484** (3-carrier block @L16: d′ 2.90, pred .423/meas .540 —
residual ~+.11, probe-power on the 10.7k-dim concat suspected, flagged). d′_w ≈ d′_auc everywhere
(adequacy ✓), ρ ≈ 0–.05, MLP ≤ linear in all 7 loci (sufficiency ✓). Ladder: model .155 < single-carrier
linear .46–.53 ≈ block .48–.54 < dtc .601 < per-frame .937.

**3 · Rooms K-channel parity (`dprime_parity/20260704_015706/`, 3 seeds) — the structural split, measured.**
Per-room one-vs-rest d′_r at the char-token carrier: **4.5–5.9 (mean 5.0) @L14**, decaying 4.0@L16 →
2.4@L18. At L14, on the SAME pooled sum: **linear readout 0.400 < per-channel-threshold readout 0.650
(closed-form prediction 0.51) ≪ decode-then-count 0.988** (per-frame room decisions are ~perfect at
d′≈5, then union). Readings: (i) **structural failure demonstrated** — a per-channel nonlinearity on the
*same* pooled vector beats the best linear readout by +0.25 (support-size is nonlinear in the tallies;
no linear functional can express it); (ii) **the √N statistical tax on presence detection is real** —
per-frame decisions 0.99 collapse to 0.65 when forced through the pooled state; (iii) layer gradient
tracks the theory (d′_r 5.0→2.4 drives hard .65→.41 and dtc .99→.71 in lockstep, linear flat at its
~.40 ceiling). Closed-form is conservative (+.14 low vs measured threshold readout — independence +
midpoint-threshold approximations); MLP-on-sum (0.34) failed to beat linear — underpowered at n_tr=360,
NOT a theory pass/fail (the hard-threshold column is the structural test and it passed). Full ladder:
model .087 < linear .40 < Σ-threshold .65 < dtc .99.

**4 · E5-cooc + H3 causal carrier map (`dprime_dose_cooc/`, job 117015, n=150) — DOUBLE DISSOCIATION.**

| arm | emitted acc (MAE) | L24 decode | final decode |
|---|---|---|---|
| base | 0.187 (2.03) | 0.350 | 0.317 |
| dose char1 ×4 / ×16 (multi-layer) | 0.193 / 0.133 | 0.383 / **0.767** | 0.350 / 0.683 |
| **scrub char1 (off15)** | **0.193 — NULL** | 0.317 | 0.350 |
| **scrub char2 (off13)** | **0.100, collapse** (g2→1.00, g4→0.03) | 0.317 | 0.283 |
| scrub "same" (off10) | 0.127, partial collapse | 0.367 | 0.367 |
| scrub char1+char2 | 0.100 (= char2 alone) | 0.350 | 0.250 |
| scrub random axis (control) | 0.187 ≈ base | 0.333 | 0.317 |

Readings: (i) **the model's co-occ count causally transits char2 (the second-mentioned name) and
partially "same" — NOT char1**, despite char1's messages being decodable (d′ 1.8): decodable ≠ consumed,
now shown *between carriers* of one task. The d′ map's argmax (char2, 2.8) = the causal carrier → **H3
validates the map's peak as a cheap causal proxy** (but secondary d′ loci are not necessarily used).
(ii) **The readout wall replicates on co-occ:** dosing char1 ×16 delivers +0.42 of last-token
decodability (0.35→0.767) while emitted *falls* (0.187→0.133). (iii) Scrubs crush behavior without
denting probe decodability (0.32–0.37) — the model consumes one channel; probes read them all.

**Cross-task synthesis.** All three tasks now show, at their deployed carriers: (a) the parity law
(exact at steps' single carrier; lower-bound-per-channel + block-closure at co-occ's distributed
carrier; conservative-but-ordered at rooms' K channels); (b) per-frame evidence strong (d′ 2.4–5.9) with
the pooled linear readout capped as predicted; (c) an enormous causal readout wall (model 0.09–0.24 vs
information present 0.5–0.99); (d) scrub-identified load-bearing carriers with inert random controls.
Task-dependent structure discovered along the way: steps = one carrier (room token); co-occ = two
distributed name-carriers, only one causally consumed; rooms = K channels through one name-carrier
with structural linear-failure on top.

**Caveats.** E5-cooc gold distribution is skewed (pick_pair maximizes co-occurrence → g≥2, model's
by-gold shows total failure at g≥5 regardless of arm); n=150 single seed for E5; rooms closed-form
under-predicts its own threshold readout by ~.14 (approximations listed above — refine before quoting
as "the" rooms prediction); cooc 3-block residual +.11 unexplained; E5-rooms (subspace scrub over
span{δ_r}) designed but NOT run — do after the K-channel model is accepted; MLP-on-sum for rooms needs
n≥1500 to be a fair structural test. Everything on balanced 5-char datasets, 7B nf4.

**Addendum (same day, CPU diagnostics on the existing caches):**
- **Rooms structural sign-flip CONFIRMED in tally space.** Fair test (project the pooled sum onto the 6
  per-room whitened axes → 6-dim features, no dimensionality excuse for the MLP): **linear 0.325–0.329
  vs MLP(32) 0.537–0.588, MLP−linear = +0.21…+0.26** (3 seeds). Counting: MLP never beats linear
  (21/21, sufficiency). Distinct-count: MLP must and does — the sign flips exactly where the algebra
  says support-size is nonlinear in the tallies. This supersedes the underpowered raw-3584-dim MLP null.
- **Cooc 3-block residual is NOT d′-estimation error:** probe-accuracy-based block d′ (2.94–2.96, from
  CV bal-acc 0.929–0.930) agrees with the LDA-projection estimate (2.90–3.19). New leading hypothesis:
  co-occ per-frame labels are truly **ternary** (same / diff / not-both-present); the binary two-cloud
  model lumps {diff, absent}, undercounting available channels (the "both-present" channel correlates
  with the count via pick_pair's construction). To be tested on the n=1500 multi-offset cache (job 117102).
- **In flight:** E5-rooms subspace-scrub/tally-dose (job 117100: scrub span{δ_r} at char token, QR-orthonormal,
  random-subspace control); rooms n=1500 cache (117101, raw-MLP rerun); cooc n=1500 6-offset cache (117102).

---

## [2026-07-04b] ✅📊 E5-rooms: the K-subspace is load-bearing (dimensionality-matched control inert) + the FIRST dose that moves behavior (a crude magnitude coupling); cooc ternary hypothesis refuted; big-cache replications

> **Runs:** E5-rooms `outputs/frame_axis/probes/dprime_dose_rooms/` (job **117100**, 14 min, n=150);
> big caches job **117101** (rooms — dataset exhausted at **n=720**, not 1500) and **117102** (cooc,
> n=1080, 6 offsets ×L14/16, 1.1 GB); CPU analyses appended to `dprime_parity/20260704_015706/`.

**E5-rooms (char-token carrier; scrub = remove the whole span{δ_room1…6}, QR-orthonormal, L15–21):**

| arm | emitted acc (MAE) | L24 decode | final decode |
|---|---|---|---|
| base | 0.133 (1.45) — model answers "3" almost always | 0.300 | 0.433 |
| tally dose ×4 / ×16 (multi-layer) | 0.133 / **0.233** | 0.367 / **0.517** | 0.417 / 0.467 |
| **scrub span{δ_r} @ char token** | **0.053 (2.29)** — answers collapse to 1–2 | 0.267 | 0.333 |
| scrub span{δ_r} @ "visit" token | 0.073 (1.81) | 0.433 | 0.433 |
| scrub random 6-dim subspace (control) | 0.120 ≈ base | 0.333 | 0.417 |

Readings: (i) **the K-channel subspace at the char token is causally load-bearing** — scrubbing it
collapses the model into low answers (the evidence-removed signature), while a *dimensionality-matched*
random subspace is inert. The rooms causal story now matches steps and co-occ. (ii) **First
behavior-moving dose in the program:** tally dose ×16 lifts emitted 0.133→0.233 — but MAE is flat
(1.45→1.43) and by-gold shows a pure **upward distribution shift** (g3-answers migrate to 4): the
readout is coupled to the **total tally magnitude** (≈ Σ n_r, a crude scalar) and gains no per-sample
discrimination. A magnitude-coupled-but-structure-blind readout — consistent with everything else.
(iii) Probe side as always: dose ×16 delivers +0.22 of decodability (0.30→0.517) far beyond the
behavioral response.

**Cooc ternary-channel hypothesis: REFUTED** (big cache, char2@L16, n=1080, 2 seeds): the 1-D same-axis
projection of the sum decodes 0.477–0.500 ≈ full-3584-dim ridge 0.491–0.523; adding the both-present
axis adds nothing (0.481–0.488). So (a) **sufficiency holds at the deployed cooc carrier** — one matched
filter carries all linearly-readable count; (b) the n=600 3-block "+0.11 residual" was probe variance,
now largely closed at n=1080 (residual vs single-channel prediction shrinks to ~+0.07); (c) the
[2026-07-04] "ternary channels" caveat is retired.

**Rooms replication at full dataset (n=720 — the balanced set is exhausted; "n=1500" impossible):**
d′_r ≈ 5.1, linear 0.375–0.420 < hard-threshold 0.656–0.688 (pred 0.52) ≪ dtc 0.993–1.000 — all
[2026-07-04] numbers replicate. Raw-3584-dim MLP-on-sum still ≈ linear (0.375–0.382) even at n=720 —
**permanently underpowered at this dataset size; the 6-dim tally-space test (+0.21…+0.26) is the
definitive structural evidence**, not the raw-dim null.

**Also measured (temporal-readiness probe):** frame POSITION is ~losslessly decodable from the per-frame
carrier messages (steps room@L16: 0.988; cooc char2@L16: 0.989; within-1: 0.999) — the positional
channel temporal tasks need exists at the deployed locus; temporal answers being nonlinear in
(evidence × position) puts them in the rooms (structural) class — prediction registered for a future
temporal rollout.

---

## [2026-07-04c] ✅📊 Carrier anatomy: char1 matters EARLY, char2 matters LATE (relay-then-bind confirmed); dose at the causal carrier moves behavior; last-token δ-channel null; distinct_* reveals the gate's low-d′ regime

> **Run:** `outputs/frame_axis/probes/dprime_dose_cooc_anatomy/` (job **117213**, 21 min, n=150; job 117197
> died on an unbound-var bug in the scramble guard — fixed; 117198 was a mis-parameterized resubmit,
> cancelled at 0 cost). New intervention: **token scramble** = add a fixed matched-norm direction to one
> token's state at L2–12 (δ-free early-layer disruption) + neutral-token control. distinct_* analysis on
> the June caches appended to `dprime_parity/20260704_015706/` console log.

| arm (co-occ) | emitted acc (MAE) | by-gold signature |
|---|---|---|
| base | 0.187 (2.03) | g2:.79 g4:.52 |
| dose ×4 / ×16 at **char2 = the causal carrier** (L14+16) | **0.240 (1.68)** / 0.220 (**1.50**) | mass shifts up: ×16 → g4:.94, g2:.07 |
| scrub last-token δ̂ (off0) | 0.187 (2.07) — **null** | unchanged |
| scrub char1 δ̂ @L14–16 (replication) | 0.200 — null | unchanged |
| **scramble char1 EARLY (L2–12)** | **0.113 (2.45)** | g4 .52→**.12**, pushed low |
| scramble char2 early | 0.180 ≈ base | mild redistribution |
| scramble 'the' early (control) | 0.167 | mild nonspecific (g4 .39) |

**Readings.**
1. **Relay-then-bind confirmed — a double dissociation across DEPTH:** char1 is causally needed EARLY
   (L2–12 scramble → 0.113, g4 collapses to 0.12; specific margin vs the neutral-token control ~.05–.07
   in acc but far sharper in the by-gold signature) and inert LATE (δ̂-scrub null, twice); char2 is inert
   early (0.180) and load-bearing late ([2026-07-04] scrub → 0.100). The [2026-07-04] "char1 ignored"
   reading is CORRECTED: char1's identity is consumed during early question self-attention and relayed
   into the binding site; by L14 its token is causally spent. **The refined single-carrier picture: one
   BINDING SITE per task — the last query-completing entity token — where upstream slots deposit their
   identity early and the mid-layer count evidence then accumulates and is consumed.**
2. **Dose–response is carrier-specific:** dosing char1 did nothing ([2026-07-04]); dosing char2 — the
   causal carrier — moves behavior (0.187→0.240 at ×4; MAE 2.03→1.50 at ×16 with the same crude upward
   magnitude-shift as rooms, g2→g4 overshoot). The dose arm now doubles as a carrier-identification test.
3. **Last-token δ̂ channel is null:** the count does not ride the last token's own δ-content through the
   mid layers — it is read off the carrier by late attention. Consistent with the stage staircase
   (carrier peaks L14–16, last token only L18–26) and with why last-token d′ was always weak (0.4).
4. **distinct_visitors / distinct_companions (June caches, K=9 char channels, CPU):** per-channel
   d′_c only 1.07–1.46 at the Q-first locus ⇒ the hard per-channel threshold readout **collapses**
   (0.13–0.14) while linear-on-sum gets 0.51–0.52 — the exact mirror of rooms (d′_r≈5 ⇒ threshold wins
   +0.25). **The gate has a d′-crossover, and the theory contains it:** per-channel thresholds beat soft
   pooling only when per-channel decisions are reliable; at d′_c/√N ≈ 0.4, premature hard decisions
   destroy graded information. Prescription refined: "gate before sum, IF d′ suffices — else raise d′
   first." (Caveat: deployed carriers for distinct_* not yet localized; a better locus likely has
   higher d′_c.)

**Caveats.** Scramble is a fixed-direction, matched-norm perturbation — a robustness probe, not a
surgical ablation (the model may partially absorb a shared offset; char2's early-scramble null could
partly reflect this); n=150 single seed; the early-char1 specific margin over the control is modest in
raw accuracy (0.113 vs 0.167) and rests mainly on the by-gold signature; last-token null is specific to
the δ̂ axis of its own messages at L15/17 entry, not a full last-token ablation.

---

## [2026-07-04d] ✅📊 Anatomy at n=400 (all claims replicate; carrier→last transfer completes by ~L17) · first temporal task localized · BATCH-0 of the tally-register solution: rooms 1.00/0.97-OOD, cooc 0.75, steps capped at its carrier d′ exactly as predicted

> **Runs:** steps anatomy `dprime_dose_steps_anatomy/` (job 117351, n=400); cooc anatomy replication
> `dprime_dose_cooc_anatomy400/` (job 117352, n=400); first_occurrence localization
> `carrier_message/firstocc_locmap/` (job 117354, A100); batch-0 solution
> `outputs/frame_axis/probes/tally_solution/20260704_150209/` + v2-fix console (logistic block gates).
> New: `experiments/glstm/tally_register_solution.py`; `first_occurrence` task added to the carrier probe.

**1 · Anatomy, now at n=400 with clean controls.**
- **The carrier→last transfer completes by ~L17 (steps):** mid-window scrub of the room token (L14–16)
  collapses the count (undercount signature, g≥3→~0) while the **late-window scrub (L18–20) is exactly
  null** (0.242 vs base 0.235). Combined with the restoration staircase: aggregate at carrier L14–16 →
  hand-off ≈L17 → last token carries it L18–26 → head. The last token is the *prediction* hub, never the
  *aggregation* site — [2026-07-04c]'s picture, now time-stamped by intervention.
- Steps also shows **relay-then-bind** (early char scramble 0.113 vs neutral control 0.255; early room
  scramble collapses onto the prior, g4=0.91) and a **partial secondary carrier** (char-token mid scrub
  0.170 — steps' binding is more shared than cooc's).
- **Cooc: all four anatomy claims replicate at n=400** — char2 scrub 0.085 (base 0.155); char1-early
  scramble 0.100 vs control 0.142; char2-early null (0.155); dose at the causal carrier +0.065 (0.220);
  random-axis scrub 0.160 ≈ base.
- Steps dose (multi ×4 @room) is null (0.230) — unlike rooms (+0.10) and cooc (+0.065): the
  magnitude-coupling of the frozen readout is task-dependent.

**2 · First temporal task (first_occurrence) localized.** Model own-answer **0.432 — below the majority
baseline 0.513** (worse than always answering "frame 1"). Carrier map: peak d′ 1.27 @L16 (off−13
region), per-frame evidence AUROC 0.948. ⚠ the report's dtc row (0.090) is meaningless — the probe's
sum-reduction is the wrong algebra for argmin-over-positions; correct temporal reduction needs the
message cache (not yet saved for this task).

**3 · BATCH-0 of the tally-register solution (deployed carrier messages; gate→tally→task algebra; the
tally IS the answer — no frozen head).** v1 = LDA gates on the single peak carrier; v2 = balanced
logistic gates on multi-token/layer block features:

| task | model | v1 IID | **v2 IID** | v2/v1 count-OOD | distill-proxy (v1) |
|---|---|---|---|---|---|
| **rooms** (multiclass gate → union) | 0.087 | **0.993–1.000** | — | **0.960–0.980** | not measured (flag) |
| cooc (2-token × 2-layer block) | 0.155 | 0.632–0.664 | **0.736–0.766** | 0.439–0.469 (v2; v1 0.35) | 0.514–0.537 |
| steps (room, 4-layer block) | 0.207 | 0.475–0.525 | 0.517–0.521 | 0.118–0.175 | ≈ gt |

**Readings.** (i) **rooms: 11× the model with near-perfect count-OOD** — gate at d′_r≈5 + parameter-free
union = the flagship demonstration of the architecture. (ii) **cooc: 4.8× the model** (block gate 0.75);
distributed carrier handled by the block read. (iii) **steps is the theory grading its own solution:**
0.52 is what the closed form predicts for a gate at the deployed carrier's d′=2.44 (p_err≈0.11 →
P(cancel)≈0.5); the 4-layer block added nothing (layers are correlated copies, not independent
channels). Raising steps needs a better read locus (query-conditioned/isolation-mask reads measured at
0.94–0.99 per-frame in June — the known remedy), not a better aggregator. (iv) **steps count-OOD failure
is NOT just prior calibration** — class-balanced gates didn't fix it (0.12–0.18); systematic undercount
(MAE≈1.9) persists. Hypothesis for batch-1: frame-content distribution shift between low- and high-count
samples (queried-room crowding co-varies with count) — a generator-confound-flavored issue; test by
auditing per-frame gate error vs gold. (v) distill-proxy: steps unaffected (flip 1.3%); cooc drops
0.66→0.53 (flip 9.1%) — look-again quality is the deployability bottleneck for cooc, as the gate-quality
theory says.

**Per-count audit (`tally_solution/20260704_150209/percount_audit.{png,json}`).** (i) The IID gate's
per-count accuracy tracks the cancellation formula (dotted curve) through g=1–6 and dips under it at
g=7–8; (ii) **the tally bias is the predicted straight line** bias(g) ≈ N·FP − g·(FN+FP): steps runs
+1.0 at g=1 → −1.4 at g=8 (slope ≈ −(FN+FP) = −0.28 as predicted from FN=.143/FP=.132); (iii) the
OOD-trained gate's failure is quantified as **FN inflation**: training on gold≤4 doubles-to-triples the
false-negative rate (steps .143→.308, cooc .075→.220) with FP unchanged — threshold drift against
positives, stacking on the structural one-sidedness at high counts (bias −3.6 at g=8).
**Bias-correction null:** the closed-form adjusted count k̂′=(k̂−N·FP)/(1−FN−FP) with train-estimated
rates does NOT rescue exact-match (IID it *hurts*: rescaling amplifies noise 1/(1−FN−FP)≈1.4× — rounding
loses more to variance than it gains from de-biasing; OOD it under-corrects because the error rates
themselves shift with the count distribution — train-side calibration can't see it). Conclusion stands:
**the fix for steps/cooc exact-match and OOD is d′ at the read locus (Tier-2 reads), not post-hoc
calibration.**

**Batch-0 verdict for the 24-task campaign:** the architecture works end-to-end and extrapolates where
per-frame d′ suffices (rooms), the block-read handles distributed carriers (cooc), and the failure modes
are exactly the theory's own predictions with known remedies (steps: read locus; cooc: look-again
quality; OOD: audit the generator confound). Pipeline is campaign-ready; the two carried caveats go into
the agent's stop-and-ask list.

---

## [2026-07-04e] 📊 ρ is axis-dependent: two distinct sources of cross-frame correlation dissociated (content vs processing); δ̂ and w* are nearly orthogonal

> CPU, mp_compare caches, console (steps/cooc, joint vs multipass, n=1080 each). Native-axis (E6)
> gradient job 117789 submitted (grad of answer-logit margin at the room token, L17/L21 entries) —
> ρ along the model's own readout axis + cos comparisons to follow.

| cache | ρ along naive δ̂ | ρ along whitened w* | cos(δ̂, w*) |
|---|---|---|---|
| steps joint | +0.054 | +0.104 | +0.07 |
| steps multipass | **+0.303** | +0.016 | +0.13 |
| cooc joint | −0.034 | +0.099 | +0.07 |
| cooc multipass | +0.013 | +0.017 | +0.11 |

**Readings.** (1) The whitened-axis ρ isolates **shared-forward-pass interference** (joint ~0.10 → multipass
~0.016, both tasks — the [2026-07-04] finding, now shown axis-specific). (2) The naive-axis ρ measures
something else entirely: **shared-scene content correlation** — steps-multipass frames correlate at +0.30
along δ̂ despite fully independent forwards (impossible for processing noise; the frames share a video's
scene statistics, and δ̂ is entangled with them). The whitened axis has projected that content away —
which is *why* it is the good axis. (3) **cos(δ̂, w*) ≈ 0.07–0.13**: the mean-difference direction and the
optimal discriminant are nearly orthogonal — an extreme-anisotropy statement that retroactively explains
the size of the 0.33-vs-3.4 gap. Open puzzle (flagged, not explained): joint processing *suppresses* the
δ̂-content correlation (0.054 vs 0.303) — candidate mechanisms: cross-frame attention homogenization, or
zero-sum attention competition anti-correlating frame representations; untested.

---

## [2026-07-05] ✅📊 E6 — the model's NATIVE reading axis extracted: nearly orthogonal to both good axes (cos ≈ 0.01–0.06), d′ = 0.51 along it — and the law evaluated on the model's own axis predicts the model's own accuracy

> **Runs:** `outputs/frame_axis/probes/native_axis/20260705_153147/` (job **117809**, rtx6k, 6 min, n=100:
> gradient of the answer-digit logit margin w.r.t. the room-token residual entering L17/L21; unit grads
> sign-aligned and averaged) + CPU comparison vs the count_msgcache room messages. New:
> `native_axis_probe.py` + runner. Ops note: multi-partition submission (`-p l40s,rtx6k,a100,l40s-public`)
> started instantly on a saturated cluster where single-partition queued indefinitely — adopt as default.

| layer | grad coherence | cos(native, δ̂) | cos(native, w*) | d′ along native (δ̂ / w* for ref) | ρ along native |
|---|---|---|---|---|---|
| L16 | 0.658 | +0.056 | **+0.005** | **0.51** (1.12 / 2.47) | +0.052 |
| L20 | 0.437 | +0.016 | +0.033 | 0.17 (0.79 / 2.26) | **+0.400** |

**Readings.**
1. **The readout wall now has a direction and a number.** The model's effective reading axis at the peak
   carrier is essentially orthogonal to the whitened discriminant (cos 0.005) AND to the naive axis
   (0.056), and carries only d′ = 0.51 — ~20% of the available separation, worse than even δ̂. "Right
   token, wrong axis," measured.
2. **Behavioral closure of the law (headline):** the closed form evaluated along the model's own axis —
   d′ 0.51, N=8, prior-mixed — predicts **≈ 0.17**; the model's measured own-answer accuracy is
   **0.21–0.24**. One law, three axes, three regimes: native 0.51 → model ~0.2; whitened 2.47 → probe
   ceiling ~0.47 (parity); multipass ≥ ~6.6 → solution 0.95+. The entire accuracy hierarchy of this
   project is one formula evaluated at three read qualities.
3. **L20's native axis is noise- and correlation-dominated:** d′ 0.17 with ρ = +0.40 along it — the
   late-layer reading direction sits in a strongly cross-frame-correlated subspace (sink-like), carrying
   almost no evidence signal. Consistent with deep-layer sinks in the carrier maps.
**Caveats.** The gradient axis is a first-order (locally linear) summary of a nonlinear readout —
coherence 0.658 says one direction captures much but not all of it; the behavioral-closure match
(0.17 vs 0.21) is therefore corroboration, not exact accounting; n=100, single task (steps), messages
space vs residual space treated as the shared 3584-dim basis (exact for o_proj contributions).

**Addendum — d′ estimator convergence (CPU, console).** Held-out d′ vs n_train frames (shrinkage-LDA):
steps bench L19: 2.15 (n=250) → 2.94 (1k) → 3.20 (3k) → 3.34 (3.8k, max); deployed room L16: 1.74 (250)
→ 2.14 (1k) → 2.47 (2.9k, max). **Still rising at max n on both loci** ⇒ every quoted d′ is a
*converging lower bound*; increment decay suggests asymptotes ≈3.5–3.7 (bench) / ≈2.6–2.8 (deployed) —
we sit ~5–10% below truth. Parity was unaffected because both sides share the finite-data handicap (the
sum-side ridge had the same samples). Direction of the bias is conservative for all Tier-2/threshold
claims; steps' single-pass verdict unchanged (even 2.8 ≪ 4.8). If an asymptote is ever needed: fit
d′(n)=d∞−c/n, or cache more frames (128-length samples donate 16× frames each).

---

## [2026-07-06] ✅📊 OVERNIGHT CAUSAL BATTERY — native-port doses (first behavior-moving dose; knob-vs-line double dissociation), real-state transplant, attention-edge patching, three-way window confirmation; write-into-any-activation-channel refuted; two instrument nulls flagged

> **Runs:** `outputs/frame_axis/probes/dprime_dose/native_port/20260706_225256/` (job 118594, n=250),
> `…/dprime_dose/causal_night/` (118588, n=250), `…/dprime_dose/rescue_early/` (118608, n=250),
> `…/carrier_transplant/` (118589, 100 diff-gold + 25 same-gold pairs), `…/attn_edge_patch/` (118590,
> n=150), `…/pixel_minimal_pair/` (118607, n=150). New scripts: `carrier_transplant.py`,
> `attn_edge_patch.py`, `pixel_minimal_pair.py`; `dprime_dose_response.py` gained rescue/setg/noisew/nat* arms.
> All steps task, 7B nf4, carrier = room token (off9), layers 14–20 unless noted.

**1. Native-port doses (E6c — injections along the model's OWN reading axis, L16+L20).** natdose×2:
emitted **0.240 → 0.300, MAE 1.48 → 1.06** — the FIRST dose in the program to move behavior upward — by
unlocking high counts the base model never emits (g5: 0.00 → 0.58, g7: 0.10 → 0.28); overshoots at λ≥4
(monotone collapse toward high answers = causal direction confirmed). natscrub: emitted 0.240 → 0.196
with decode@L24 intact (0.22 ≈ base 0.23); random-axis scrub control clean (0.228). **Double
dissociation at one token: δ̂/w* arms move content-not-behavior; native arms move behavior-not-content.**
natset (write a chosen g′ along the port): follows at 0.176 < coincidence 0.240 — the port is a
**confidence knob, not a number line**.

**2. Real-state carrier transplant (no synthetic directions).** Same-question donor with |Δgold| ≥ 2,
carrier state swapped at L15/17/19/21 entries: emitted follows the DONOR 0.04 → 0.18 and repaired-read
0.05 → 0.20, both **monotone in blend α** (0.25/0.5/0.75/1.0); same-gold control transplant harmless
(0.28, MAE 1.16). The count is a transferable object in the model's own state space.

**3. Attention-edge path patching (E5b).** Cutting evidence-frame→carrier edges at L14–20: 0.353 →
**0.207** (undercount, mean_pred 2.47 → 1.92); equal-count random-frame cuts: 0.380 (≈ base); all-frame
cuts: 0.253 (prior fallback — no signal beats confidently-wrong signal). Route exclusivity confirmed.

**4. Transfer window ≤ L17 is now CAUSAL, three independent ways:** cutlate (last←carrier edges cut at
L17–20) = null (0.373); rescue-late (re-inject after scrub at L21-entry) = exact scrub replica (0.196);
rescue-early (inject L15-entry under continuous scrub) also fails (0.204) because **attention re-delivers
messages every layer** — removal must be continuous, leaving no restoration slot. Scrub-then-rescue is
infeasible-by-mechanism at this locus; the infeasibility itself measures the re-delivery dynamics.

**5. No activation channel is dictatable.** setg along δ̂: 0.068 vs 0.24 coincidence (anti-follows);
natset 0.176 < 0.24. Liu's decodable ≠ correctable reproduced at our locus on BOTH axes ⇒ the tally
register's symbolic readout wins **by exhaustion** — token space is the only graded, dictatable interface.

**6. Pixel minimal pairs (input-space causality) — half-confirmed, instrument honest-failed.** One-frame
re-render moving C out of R: 56% of emissions move, 53% move DOWN (vs 23% control churn), MAE-to-new-gold
1.37 → 1.24. Up-edits register at only 19% — blocked by the undershoot bias wall, as the softmax story
predicts. The carrier-state w*-projection instrument FAILED its own control (control edits shift −0.034 ≈
down-edits −0.038): state-level projection measures edit churn, not signed evidence. Clean version needs
message-level capture. **noisew (graded w*-noise at carrier):** flat on both channels — inconclusive
(decode instrument ceiling 0.25; injection partly post-window). Not evidence against the law.

**Synthesis.** The causal chapter is closed on all four sides: **content** (scrub kills it, transplant
carries it, specific edges route it), **access** (native-axis doses open it, native scrub narrows it,
cos(native, w*) = 0.005 and length-invariant), **timing** (≤L17, three nulls), **interface** (nothing
dictatable but tokens). The base-model per-gold rows also directly exhibit the fraction-reader signature
(g1/g5/g6 at 0.00 across all 250-sample arms). Follow-ups parked: message-level pixel-pair capture;
pre-window noise dose; per-digit native ports (deprioritized — natset below coincidence).

---

## [2026-07-07] 📊 CONSOLIDATION: every Mamba result in one place — frozen Falcon-Mamba-LM vs frozen Qwen (text), and the carrier_mamba adapter operator vs sum, per input type (messages vs cached reps) and per task

> No new runs — this section tabulates existing logged results so the Mamba story is readable in one
> place ahead of the Falcon-Mamba mechanistic track. Sources named per table; original rows unchanged.

**A. Frozen Mamba-LM vs frozen Qwen-LM — text-MMRED, zero-shot** (run
`outputs/frame_axis/mamba_lm_text/20260627_125502/` (+3 sibling ts dirs same day), model
`tiiuae/falcon-mamba-7b-instruct`, n=45/task; Qwen text refs from the text-frames baseline evals):

| task (text frames, N=8) | Falcon-Mamba-7B | frozen Qwen-7B (text) | winner |
|---|---|---|---|
| steps count | 0.244 (MAE 1.36) | **0.47** | Qwen |
| rooms visited | 0.244 (MAE 1.02) | **0.39** | Qwen |
| co-occupancy | **0.400** (MAE 0.60) | 0.34 | Mamba |

*Caveats: n=45/task (±0.07 SE); pretraining/instruction-tuning confound (script docstring). The
task-dependent flip (Mamba wins only co-occ) is unexplained — a candidate signature of recency-weighted
accumulation (evidence-position profiles differ per task); pre-registered for the mechanistic track.*

**B. carrier_mamba adapter operator vs sum — PER-MESSAGE input (frame→carrier attention messages,
IMAGE pipeline, frozen Qwen-VL 7B nf4;** `layerwise_frame_message_glstm.py`**):**

| task / condition | mamba IID | sum IID | mamba len-OOD | sum len-OOD | run |
|---|---|---|---|---|---|
| counting, neutral fillers (3-seed) | **0.988** [.981,.998] | 0.972 [.950,.986] | **0.929** [.920,.937] | 0.877 [.873,.880] | `outputs/dm5_count_{sum,mamba}_s{1,2}/`+dm4 |
| counting + distractors (selection) | 0.652 | 0.643 | ~0.57–0.60 | ~0.57–0.60 | `outputs/dm4_distract_*/` (oracle 0.963) |
| co-occupancy (evidence-only) | 0.557 | 0.538 | — | — | `outputs/diffmamba2_coocc/` (glstm 0.613; oracle 0.98) |
| order-sensitivity (counting) | 0.998 → 0.971 permuted → 0.979 order-aug | | | | `outputs/dm4_{count_mamba,order_permEval,order_aug}/` |

**C. mamba operator over PER-TOKEN cached reps (frozen minimal frame reps, IMAGE pipeline, LM readout):**
co-occ **0.34**, rooms **0.51** — quoted in the [2026-06-25] Phase-2 synthesis (SNR-limited, as the
theory predicts: a smarter aggregator cannot un-sum what low-SNR reps never separated). Temporal tasks
(image, `outputs/frame_axis/adapter_live/temporal/*_20260627_153222/`): mamba ≈ sum on first/last/span
(position leaks into every rep, so an order-blind sum suffices).

**Reading, in the theory's terms.** The operator only wins where aggregation is genuinely the binding
constraint AND the reps supply d′ (clean counting: extrapolation edge 0.929 vs 0.877, non-overlapping
seeds); it ties or trails wherever supply binds (distractors ≪ oracle; per-token low-SNR reps; co-occ).
Frozen Falcon-Mamba behaviorally underperforms the frozen transformer on 2/3 text tasks at small n.
Nothing in the Mamba record exceeds a gate-law ceiling; the retro-audit (measured acc vs gate-law(d′)
per run) is parked as CPU work.

---

## [2026-07-07b] ✅📊 TRACK B — CROSS-FAMILY ROLLOUT to InternVL2.5-8B: the fraction-reader signature, the two-site carrier anatomy, and the ZERO-PARAMETER LAW PARITY all replicate on a model sharing no weights with Qwen; readout fraction ≈ family-invariant

> **Runs:** `outputs/frame_axis/internvl/baseline/` (job 118959, n=250/task),
> `…/internvl/carrier_map/` (118962, n=300, MAXOFF=13) + `…/carrier_map_ext/` (118968, MAXOFF=20),
> `…/internvl/parity/20260707_202109` (CPU, full parity engine, 3 seeds × 2 layers).
> New: `experiments/internvl/{baseline_eval,carrier_map,native_axis,dose_scrub}.py` + runners.
> Model: OpenGVLab/InternVL2_5-8B (InternLM2 LM, 32 layers, fused wqkv attention, dynamic-tile ViT,
> 256 vis tokens/frame), 4-bit nf4, protocol identical to Qwen (single-forward digit argmax).
> Env: +sentencepiece, +einops, +timm(--no-deps) installed with approval; torch/transformers unchanged.

**1. Behavioral baseline — the fraction-reader signature replicates, exaggerated.** steps 0.124
(majority 0.132), slope **0.08**, answers "2" for nearly everything (g2:0.90, all other golds ≤0.16);
rooms 0.088 (majority 0.392), slope 0.10, collapsed onto 2–3. Worse than Qwen (0.21/0.09) — matches
MMReD's own family ranking.

**2. Carrier anatomy replicates — two sites, same relative depth.** Fused-wqkv message capture
verified EXACT (reconstruction vs the model's own wo-input: cos = 1.0000, first-sample hard guard).
Peak carrier = the ROOM token, d′ = **1.90 @ L20/32** (~62% depth; Qwen: room token 2.47 @ L16/28,
~57%); the CHARACTER token is the secondary site (1.0 @ L12) — Qwen's room-primary/char-secondary
structure. Sweep extended to offset 20: nothing larger beyond.

**3. LAW PARITY, zero parameters, on a foreign family.** Full parity engine on the carrier cache:
d′_w 1.69–1.96 with d′_auc agreeing (1.77–2.07 ⇒ Gaussian adequacy holds); ρ = +0.04–0.13 with the
iid prediction fitting better than ρ-penalized (Qwen's pattern); naive d′ 0.5–1.1 ≪ whitened (the
whitening gap); MLP ≤ ridge (sufficiency echo). **pred_iid 0.270–0.304 vs measured ridge 0.225–0.367
— means 0.288 vs 0.292: on the diagonal.** Model emits 0.137 ≈ **47% of its probe ceiling (Qwen:
45%)** — the readout-wall FRACTION looks family-invariant; the d′ supply difference (1.9 vs 2.5)
alone explains the model ranking.

**Methodology note.** A quick single-split ridge (alpha fixed, no protocol) had first suggested a
parity gap (measured 0.183 vs pred 0.344) — estimator starvation, resolved by the engine. Standing
rule for the campaign: parity verdicts only from `probe_dprime_parity.py`, never from ad-hoc decoders.
Adaptation friction log (all fail-loud, ≤2 min each): 3 missing deps, vision-dtype mismatch,
GenerationMixin API break (transformers ≥4.50 vs remote code) → replaced generation with the
single-forward digit protocol; native-axis OOM under eager-attention backward → gradient graph now
starts at the first probed layer.

**4. [appended 2026-07-07 late] E5 first pass "failed" — diagnosed exactly; law closure completed.**
Runs: `internvl/dose_scrub/` (118977), `internvl/e5_diagnose/` (119000), `internvl/multipass_bench/`
(118996), `internvl/native_axis_N4/` (119001, L20, N=4 seqs after 3× OOM — a100-public nodes are
**40GB**, not 80; eager-attention backward needs short seqs or H200).
**(a) Multipass bench:** isolated-frame carrier d′ = **6.38/6.56** (L16/L20) vs joint 1.8/1.9 — the low
joint d′ is ~all JOINT-PASS INTERFERENCE (3.5× collapse vs Qwen's 1.4×); querying exonerated
(mass-d′ 0.68 = Qwen 0.67); perception-at-rep-level excellent. Per-frame BEHAVIORAL acc only 0.675
(Qwen 0.96–0.99): a readout wall at N=1. InternVL's multipass supply clears the 128-crush line (6.5 ≥ 6.3).
**(b) Why the E5 scrub was null — wrong axis (H-axis):** per-layer carrier-STATE count decode is
IDENTICAL under scruball (0.21–0.23 both arms) — in InternVL the accumulated state's count lives off
the message-δ̂ axis (Qwen's message-δ̂ was load-bearing; InternVL rotates message→state more).
State-derived δ̂ scrub = the fix, parked.
**(c) Why the dose "didn't deliver" — it DID; the instrument lied:** per-arm logit lens shows doseMid
lifts last-token count decode **0.23 → 0.33** from L14 on (the earlier base-trained repaired head was
off-distribution on dosed states). Emission still 0.133 ⇒ the classic readout wall, properly measured.
**(d) The pipe exists — H-nopipe rejected:** last→carrier attention window **L8–L19, peak L13–L16**
(mass 0.047 vs 0.005 background), closing after L19 — mirroring Qwen's ≤L17 window. dose8 (pre-peak,
single site) delivered little; doseMid (in-window) delivered.
**(e) MODEL-LEVEL LAW CLOSURE ON A FOREIGN FAMILY:** native axis (grad extraction, coherence 0.47,
length-transferred from N=4) has cos(native, w*) = **0.020** and carries d′_native = **0.19**; the law
at that d′ predicts accuracy **0.135**; the model measures **0.117–0.137**. One formula, two families,
each model's behavior predicted from its own reading axis. Correction to §3's framing: InternVL's
emissions are prior-like because its native axis carries almost nothing — the readout wall is nearly
total there, and the law says exactly that.

---

## [2026-07-07c] 📊 InternVL "is the carrier actually read?" thread closed: dose delivers end-to-end at λ≥32 (under-dosing + off-distribution head explained the earlier null); NO single attention route into the last token is individually necessary — the count content is redundantly routed

> **Runs:** `internvl/dose_ladder/` (job 119009, n=200, λ=2..64), `internvl/edge_cut/` (119010, window
> L8–19) + `internvl/edge_cut_late/` (119011, window L20–31), both n=150.

**λ-ladder (final-layer decode, per-arm heads / emitted):** base 0.163/0.120 · ×2 0.163/0.120 ·
×4 0.188/0.120 · ×8 0.150/0.120 · ×16 0.175/0.125 · **×32 0.300/0.120 · ×64 0.375/0.115** · scrub
0.125/0.115. Delivery threshold λ≥32 at the final layer (Qwen: λ≈4–8) — quantifying the **late-layer
attrition** (λ16 reads 0.33 at L14–16, decays to ~0.18 by L31); still unsaturated at λ64 (0.375 vs
Qwen's ~0.75 pipe) — a narrower, lossier pipe. At λ≥32 the model becomes perfectly g2-fixated
(g2 = 1.00): **information more than doubles at the answer's doorstep; behavior never moves at any λ.**
C6's dissociation on family #2. Ops lesson: dose ladders are per-model; the first-run "null" was
Qwen-calibrated λ + a base-trained head applied off-distribution.

**Edge-cut necessity, both windows (outcome = final-layer decode; emission prior-locked):** early
window — base 0.267, cut last←carrier 0.233, cut last←question 0.217, cut last←images 0.250 (all
within ~1 SE); late window — cut last←question 0.317, cut last←images 0.300, cut last←carrier 0.250
(≥ base!). **No single-category cut in either window removes the last-token count content** ⇒ the
content is redundantly routed (and/or partially pre-L8); the relay-late hypothesis also falls.
Contrast: Qwen's frame→carrier evidence edges WERE individually necessary (cutevid 0.353→0.207).

**Thread verdict.** "Is the carrier read by the last token for the answer?" — three-part answer:
(1) for the ANSWER: effectively no — the native axis carries d′ 0.19 and the law converts that to the
observed 0.117–0.137 (prior-locked); (2) for the deliverable CONTENT: yes causally — doses injected at
the carrier propagate to the final state (λ-ladder), so the carrier is a genuine source; (3) by WHICH
route: diffusely — no individually-necessary edge, unlike the concentrated Qwen anatomy upstream.
Follow-ups parked: union cuts (car+q+img simultaneously); pre-L8 window; state-derived δ̂ scrub.

---

## [2026-07-07d] ✅📊 REAL-BENCHMARK TRANSFER — HERBench Action Counting (HD-EPIC video): the undercount wall + carrier architecture + ladder ordering REPLICATE on data we didn't generate; adequacy correctly REJECTS the Gaussian closed form (graded-evidence regime); frame-isolation d′ gain is NULL on real video

> First test of the d′ framework outside MMRED. **Task:** HERBench-lite Action Counting
> ([arXiv:2512.14870](https://arxiv.org/abs/2512.14870); 144 questions / 28 HD-EPIC egocentric kitchen
> videos, 5-way MCQ "how many times does action-object pair X occur", `required_timestamps` = one
> timestamp per occurrence ⇒ exact per-frame evidence gold, verified `len(ts)==true_count` on all rows).
> **Prep** (`experiments/herbench/prep_ac_frames.py` → `data/herbench_ac/`, 448px frames): **armA**
> evidence-only (N=true_count, every frame an occurrence; retrieval AND selection removed); **armB**
> k evidence + same-video fillers ≥5 s from any occurrence, N=16, count ≤12 (n=134). Videos fetched by
> walking the HF tar-part headers with HTTP ranges (22.7 GB of 161; login-scratch only; re-fetch script
> in session scratchpad). PyAV wheel unpacked to `~/.local/pyav-py39` (no pip, .venv untouched).
> **Jobs (all 2026-07-07, 7B nf4, 1 GPU):** own-answer **119014**; locmap **119013** (probe gained
> `--task herbench_ac`); message caches **119017** (joint) / **119018** (fenced) + CPU analyses (console).

**1 · Evidence-only counting collapses — the MMRED wall, verbatim, on real video (armA, n=144,
`outputs/herbench/own_answer/armA_20260707_213726/`).** Exact-match **0.049**, bias **−3.5**; by-gold:
g1 0.47, **g≥2 = 0.00 everywhere**, mean_pred saturates at ~2–3 whether 4 or 10 occurrences are shown.
Frozen Qwen cannot accumulate >~2–3 evidence items even with retrieval and selection removed — the
carrier-saturation signature from MMRED (model 0.21 / rooms "answers 3") on video we did not generate.
MCQ-mapped 0.27 (chance 0.20) — inside HERBench's own 31–42% SOTA band, and oracle frames don't fix it
⇒ their "fusion deficit" is our aggregation+readout wall, now with a mechanistic account. armB EM 0.17
(gold-2 collapse), bias −2.5.

**2 · Carrier architecture transfers (locmap, n=134, `outputs/herbench/probes/ac_locmap/20260707_213723/`).**
Per-token×layer map finds the carrier at the **quoted action-pair tokens** (offsets −9/−10, L12–16,
naive SNR 0.86) — the real-video analogue of the room token: frame evidence again flows into the
query's key content tokens. Model digit-argmax own-answer 0.082.

**3 · The honest divergence: extraction is weak at the source, and the Gaussian model is REJECTED.**
Per-frame evidence at the carrier: AUROC ≈ 0.80, held-out whitened **d′_AUC ≈ 0.98–1.10** (vs MMRED
deployed 2.4–2.8). **E4 adequacy fails decisively** (joint cache, CPU): skew → +3.5, excess kurtosis
**+2.6…+35** (MMRED threshold 0.2), class-std ratio 1.4–2.2, d′_gap vs d′_AUC disagree 20–40% —
the class-conditional Gaussian is the wrong model here. Mechanism: **graded evidence** — real actions
range from unmistakable to marginal mid-action/occluded frames ⇒ a mixture along the evidence axis
(binary-by-construction MMRED never had this). Pooled sum partially re-Gaussianizes (residual exkurt
+4.7) and **corr(sum-projection, gold) = 0.73** — the count information is present but graded.

**4 · Frame-isolation (fence) d′ gain is NULL on real video (`ac_msgcache_fenced/`, fence verified
Δv@L12 = 5.75).** Fenced d′ 0.86–1.09 ≈ joint 0.98–1.10 — no recovery, vs MMRED's 3.1→5.2 from the
same mask. Cross-frame superposition is NOT the binding constraint on HD-EPIC at 448px; the per-frame
signal itself is. Revises the supply-side story: querying/interference dominates in the synthetic
regime; on real video the extraction ceiling moves to the pixels/encoder side.

**5 · Ladder + law (armB, N=16; registered before the runs).**
model **0.082** < law-predicted linear ceiling **0.158** (d′_AUC 0.98, prior-mixed, zero fitted params)
≈ measured linear-on-sum **0.196±0.041** (maj 0.191) ≤ dtc **0.226** (maj 0.217).
Scorecard: (a) ladder ordering ✓ (compressed exactly as d′≈1 dictates); (b) prediction within ~1σ ✓
but the regime is weakly discriminating (prediction ≈ prior) AND adequacy fails → quote as
**ordering + magnitude**, not exact parity; (c) fence raises d′ ✗ NULL (the informative surprise);
(d) evidence-only undercount ✓✓.

**Two-regime synthesis (the thesis-level claim).** MMRED: binary evidence, extraction-strong
(d′ 2.4–5+), Gaussian adequacy passes ⇒ law exact; bottleneck = aggregation + readout; isolation lifts
d′. Real video: graded evidence (mixture), extraction-weak (d′≈1), isolation null ⇒ everything
downstream compressed toward the prior; the E4 battery correctly refuses the closed form. Same
instruments diagnose both regimes — the framework identifies *which* regime a task is in, which is a
stronger generality result than the law holding everywhere.

**Caveats.** 448px frames (native 1408) — resolution follow-up pending; evidence labels binary while
content is graded (label noise biases d′ down = conservative); fillers may contain related actions
(same direction of bias); n=134 single seed for caches; MCQ mapping via nearest-choice; armA variable-N
prevents message caching (behavioral only); videos are session-scratch, frames in `data/herbench_ac/`
are the durable artifact. RLPC (148 people-counting rows, same lite set) untouched — pipeline ready.
Index: `outputs/herbench/INDEX.md`.

---

## [2026-07-07e] ✅📊 HERBench follow-ups: forced-binary curation QUANTIFIES the graded-evidence mixture (~1/4 of real evidence is binary-groundable, d′ 0.98→2.1 on it); resolution 448→896 is a NULL (judge +0.03 AUROC, carrier d′ flat, fence-null replicates) — the extraction ceiling is intrinsic perception, not pixels or interference

> **Jobs:** look-again 448 **119020** (7 min; `experiments/herbench/lookagain_frames.py`, new — per-frame
> single-image yes/no judge, P(yes) from yes/no logits); 896px re-prep (login CPU, `data/herbench_ac_hi/`,
> 343 MB); hi-res caches **119027** (joint) / **119028** (fenced, self-check Δv@L12=7.4) ×
> look-again-hi **119029**. CPU analyses on console; curation is by the look-again judge, NEVER the
> probe axis (no selection on the measured direction; held-out probe throughout).

**1 · Forced-binary curation (448 cache × judge, thresholds P(yes)≥0.7 evidence / ≤0.3 filler).**
Judge itself: AUROC **0.832**, median P(yes) on TRUE occurrences only **0.40** (fillers 0.01) — the
model cannot verify ~3/4 of annotated occurrences even one frame at a time. Curation keeps **26% of
evidence frames** (85% of fillers); on the surviving subset **d′_AUC 0.98 → 2.10**, evidence-class
kurtosis +25 → +3.3. Reading: real-video evidence ≈ **1/4 crisp (MMRED-regime, d′≈2.1 ≈ deployed
steps 2.4) + 3/4 marginal** (the mixture that broke E4). The graded-evidence account is now measured,
not hypothesized. (Std-ratio worsens on the small survivor set; stricter thresholds run out of data —
"regime shift", not "Gaussianity restored".)

**2 · Resolution 448→896 (4× visual tokens): NULL on every channel.** Judge 0.832→**0.860**; joint
carrier d′ 0.98–1.10 → **0.90–1.07**; fenced-hi 1.01–1.16 ≈ joint (isolation null replicates at high
res); curated-hi keeps 32%, d′ 1.64 (≈ 448-curated within small-n error); model own-answer 0.052,
dtc 0.200 (maj 0.217) — unchanged. **The per-frame perception ceiling on egocentric video is intrinsic
(motion/occlusion/ambiguity of mid-action frames), not resolution and not cross-frame interference.**
Consistency: judge 0.83–0.86 (single-frame path) ≈ carrier-message per-frame AUROC 0.80 (attention
path) — two independent readout routes hit the same extraction cap.

**Synthesis addendum.** The real-video regime is now fully characterized: supply-capped at d′≈1 by
intrinsic perception, with a binary-groundable quarter where the synthetic regime's numbers reappear.
Prescription order for real video inverts MMRED's: raise per-frame evidence quality FIRST (better
perception / multi-frame-per-occurrence looks / stronger encoder), then the aggregation+readout fixes
apply on top. **Caveats:** curation judge = same model (measures "frames the model can perceive" —
the intended question, but not an external ground truth); survivor-set d′ has wide error bars
(n_ev ≈ 110); single seed; 896 < native 1408 (a native-res run would close the resolution question
completely, ~4× cost).

---

## [2026-07-08] ✅📊 B0a resolution smoke — deployed carrier d′ vs frame resolution: 392px keeps 87.5% of the 512-baseline d′ (within the pre-set 15% rule) → **392px chosen for all long-N work**; 336px drops 20%

> Plan 2026-07-08 workstream B, step B0a [preapproved-smoke]. **Question:** what does downscaling
> frames cost in *deployed carrier d′* (frames-first room-token messages) — the number the prior
> behavioral (2026-04-14) and is-evidence-AUC (2026-06-20) sweeps never measured. **Run:**
> `outputs/_scratch/b0a_res_smoke/20260708_195354/` (job **119516**, rtx6k, 25 min; probe gained
> `--resize`, tokens/frame=(px/28)²: 512→324, 392→196, 336→144). n=150, steps crowded 5-char N=8,
> carrier=room token (off−9), L14/16/20, joint pass. **d′ via the parity engine** (3 seeds,
> held-out shrinkage-LDA; `outputs/_scratch/b0a_res_smoke/parity/{20260708_201235,20260708_201658}/`
> + console.log).

| room@layer | 512px (native) | 392px | 336px |
|---|---|---|---|
| **L16 (peak)** | **d′_w 1.99±0.11** | **1.74±0.05 (−12.5%)** | 1.60±0.06 (−20%) |
| L14 | 1.69±0.05 | 1.49±0.08 (−12%) | 1.30±0.03 (−23%) |
| L20 | 1.83±0.07 | 1.51±0.03 (−17%) | 1.65±0.04 (−10%) |
| per-frame balacc @L16 | 0.864 | 0.829 | 0.827 |
| model own-answer | 0.200 | 0.200 | 0.173 |

**Decision (per the plan's pre-registered rule):** 392 ≈ within 15% of baseline AND N=128@392px ≈
25.4k tokens fits context → **long-N (B0b/B1/B2/B3) runs at 392px**; 336 exceeds the drop threshold
and is not needed. No stop-and-ask triggered. Data strategy: render long-N sets at native 512 and
downscale at load (`--resize`) so any res remains measurable later.
**Consistency checks:** naive-axis SNR at 512 reproduces the known 1.16@L16 ([2026-07-03], 1.156
here); behavior is res-flat (the 2026-04-14 result) while d′ is not — resolution taxes the supply
side well before it shows behaviorally at N=8. Parity holds at every res (pred vs ridge within
~1–4 pts). **Caveats:** n=150 (d′ values are finite-n lower bounds — compare across res at matched
n only); joint-pass only; steps task only; L20 non-monotonicity (1.51 vs 1.65) within seed noise.

---

## [2026-07-08b] ✅📊 C1 token-interface smoke — the registered "≈1.0 trivially" prediction is WRONG in the most useful way: the token interface hits 0.95–0.99 (incl. OOD multi-digit counts) ONLY under semantic-fact phrasing; explicit answer-directives and tally sentences are IGNORED (0.00)

> Plan 2026-07-08 workstream C, step C1 [preapproved-smoke]. Write an oracle count into the prompt
> as digits, zero training; deployed visual context (steps, N=8 images, crowded 5-char). Answer
> read two ways: candidate-digit argmax (0–8, comparable to all prior runs) AND greedy generation +
> first-integer parse (multi-digit capable — the B3/C answer reader, new). **Script:**
> `experiments/readout/c1_token_interface.py` + runner. **Runs:**
> `outputs/_scratch/c1_token_interface/20260708_202759/` (job **119524**, n=150, v1 arms) and
> `…/20260708_204306/` (job **119528**, n=120, phrasing variants). Counterfactual arms write a
> WRONG count k≠gold (in-range 0–8) or an out-of-range k ∈ {11,13,17,23,29,34,40} — target = k
> (does the model repeat the written count?).

| phrasing (arm) | tally=gold | counterfactual in-range | counterfactual OOD (11–40) |
|---|---|---|---|
| "Counted occurrences so far: k." (before instr) | 0.013 | 0.000 | 0.000 |
| same, right before "Answer:" | 0.042 | — | 0.000 |
| **"The correct answer is k."** | **0.008** | 0.000 | 0.000 |
| **"Note: C spent exactly k steps in the R."** | **0.992** | **0.950** | **0.992** |

(accuracies = generation vs written target; base arm reproduces the 0.200 baseline. In the failing
arms the model just does its own counting — acc-vs-gold stays ≈0.19 with the usual undercount.)

**Readings.**
1. **The token interface is real and composes out-of-range** — but only as a *world-fact in the
   question's semantic frame*: "Note: Michael spent exactly 17 steps in the Park." → the model
   answers 17 at 0.99, two-digit targets included, zero training. The C2 comparison bar is set:
   **0.95–0.99 in-range and OOD**, params = 0.
2. **The refutation half (registered prediction C-1 ✗/partial):** "digits compose so anything ≈1.0"
   was wrong — a frozen VLM does NOT obey "The correct answer is k." (0.008!) nor a tally sentence
   even when adjacent to the answer slot. The interface is *phrasing-gated*: the count must enter
   as evidence (a fact it can treat as world state), not as instruction. For the tally-register
   architecture this fixes the design: **the adapter's symbolic output must be rendered as a fact
   statement**, not as an answer hint.
3. **The model prefers the written fact over its own perception** (fact counterfactual arms:
   acc-vs-gold drops to 0.008–0.033 — it follows k, not the frames) — clean dictatability, in
   token space, with the right packaging.
4. Instrument note: the digit-argmax reader shows first-token logit nudges the generation path
   doesn't follow (e.g. afterq_std argmax 0.625 vs gen 0.042) — first-position logits over-report
   compliance; generation is the honest channel for interface tests (and the argmax reader is
   blind >9 by construction).

**Caveats.** n=120–150 single seed; steps task only; the fact arm's phrasing matches the question
template exactly (paraphrase robustness untested); "honest version" (trained tally-adapter output →
text) not yet run — this was the oracle-written-prompt bar.

---

## [2026-07-08c] ✅📊 A1 — text-MMRED full instrument battery: perfect input legibility does NOT raise carrier d′ (1.8 ≈ image) — the supply cap is the model's carrier-write process, not perception and not interference (fence NULL on text); behavior, ladder, and the native-axis law closure all replicate without vision

> Plan 2026-07-08 workstream A rung 1. Frames rendered as TEXT (states → words, wording identical
> to `eval_mmred_text_frames_acc.py`), frames-first, frozen Qwen-7B nf4 — the same instrument
> battery as the deployed image pipeline, vision tower removed entirely. **New instrument code:**
> `probe_frame_to_carrier_message.py --text-frames` (text-block token groups via fast-tokenizer
> offsets, self-checked == chat-template ids), `native_axis_probe.py --text-frames`,
> `experiments/glstm/native_axis_compare.py` (new CPU E6 comparator). **Runs:** behavior
> `outputs/ladder/text_mmred/behavior/20260708_200014/` (job 119517, n=250); locmap+cache
> `…/locmap_cache/20260708_200751/` (job **119518**, n=400, L4–24, offsets −9 room / −13 char);
> fenced cache `…/locmap_cache_fenced/20260708_202818/` (job **119525**, fence L0–13, Δv verified);
> parity `…/parity/` + `…/parity_fenced/` (3 seeds); native axis `…/native_axis/20260708_200014/`
> (job 119517, n=100) + `…/compare/`. Index: `outputs/ladder/INDEX.md`.

**1 · Behavior: the wall is vision-independent.** Text steps N=8: **0.196** (generation, n=250;
bias −1.95, 74% undercounts) / 0.165 (digit-argmax, n=400) — statistically the image pipeline's
0.21/0.24. Removing extraction entirely does not move the frozen model. *(Registered prediction
A-2 ✓.)* NB the historical "frozen Qwen text 0.47" row averaged seq_len 1–8; at seq8-only it was
never high.

**2 · The headline refutation: carrier d′ does NOT rise on text.** *(Registered prediction A-1 ✗
for this rung — the most informative miss of the day.)* Room-token carrier messages: **d′_w
1.65–1.88 (L14–L20)**, per-frame probe AUROC 0.94 @L16 — the same numbers as the image deployed
locus (1.99 @ n=150 / 2.47 @ n=600), nowhere near the "extraction ≈ perfect ⇒ d′ high ⇒ probe
ceiling ≈ 1" expectation. Input legibility and carrier-message separability are different
quantities: **what caps d′ is the frame→carrier attention write itself.**

**3 · Fence NULL on text kills the interference account for this cap.** Cross-frame attention
blocked L0–13 (Δv self-check passed): fenced room d′_w **1.84 @L16 vs joint 1.76**; per-frame
AUROC 0.917 vs 0.938. No recovery (images: 3.1→5.2 from the same mask). Text-MMRED behaves like
HERBench (fence-null) while being synthetic, binary and perfectly legible ⇒ the supply ceiling
family now has two members: *interference-limited* (image-MMRED) and **write-limited** (text-MMRED,
and plausibly the "intrinsic perception" part of HERBench is partly this too). Side-effect worth
knowing: the fence collapses text *behavior* (own-answer 0.070) — the model's own computation uses
cross-frame text attention even though the carrier messages don't gain from isolation.

**4 · Ladder + parity on text (zero fitted params).** model **0.165** < law ceiling at the peak
carrier (d′ 1.76, N=8, prior-mixed) **0.280** ≈ measured ridge-on-sum **0.317** < dtc (room+char
@L16) **0.514** < per-frame 0.94. Parity is on-diagonal at L14–16; **measured > predicted at
L18/L20 and at the char token (0.41 vs 0.27; 0.38 vs 0.22)** — the locmap explains why: on text
the evidence is *distributed* across many question tokens (offsets −2/−5/−8/−9/−11/−13 all carry
naive SNR 0.8–1.0 @L16–18; char > room, unlike image), so single-token predictions are lower
bounds, the co-occ distributed-carrier signature. Adequacy: skew ≤0.7 / exkurt ≤2.8 / std-ratio
0.7–1.2 at L14–16 (law's domain); L20 fails kurtosis (+7.7) — quote L14–16 only.

**5 · E6 law closure on text (third modality-regime).** Native axis @L16: coherence 0.67,
**cos(native, w*) = −0.004, cos(native, δ̂) = +0.13, d′_native = 0.80** → law prediction **0.163**
vs measured own-answer **0.165** (same cache; behavior run 0.196). "Right token, wrong axis"
replicates verbatim without vision; the readout wall is a property of the frozen LM stack, not of
the visual interface.

**Caveats.** Steps task only; text carrier is more distributed than image (single-token d′
understates the pooled supply — block-read parity like cooc's not yet run); native-axis coherence
0.67 = first-order summary; fenced-behavior collapse (0.070) makes the fenced own-answer column
non-comparable; n=400 d′ values are finite-n lower bounds (image comparison quoted at both n=150
and n=600 for fairness).

---

## [2026-07-08d] ✅📊 A2 — text-CWE rung (word-in-frame counting, extraction perfect by construction): the high-d′ rung confirmed (d′_w 4–5, per-frame ≈ 1.0) — and the frozen model largely ESCAPES the wall on it (0.79 @ N=8), decaying far slower than √N; E4 correctly rejects the closed form in a THIRD way (variance-ratio explosion)

> Plan 2026-07-08 workstream A rung 2. Synthetic "Frame i: <word>" text frames; question =
> `how many frames contain the word "X"` (targets verified single-token in quotes); gold uniform
> 0–8 at every N; frozen Qwen-7B nf4, same battery. **Code:** `--task text_cwe` in the carrier
> probe + native-axis probe (generator `gen_cwe_sample`). **Runs:** N=8/16/32
> `outputs/ladder/text_cwe/locmap_cache/20260708_201605/N{8,16,32}/` (job **119522**; its N=64
> pass OOM'd at 48G — RAM, not GPU) + N=64 rerun (job **119529**, 96G, 4 layers)
> `…/N64/`; parity `…/parity/` + `…/parity_n64/` (3 seeds); native axis
> `…/native_axis/20260708_201605/{20260708_205647,compare}/`. Index: `outputs/ladder/INDEX.md`.

**1 · Behavior vs N (own-answer, digit-argmax, n=150/N):**

| N | 8 | 16 | 32 | 64 |
|---|---|---|---|---|
| model | **0.793** | 0.547 | 0.540 | 0.473 |
| law at the single-carrier d′ (pred_iid) | 0.685 | 0.430 | 0.307 | 0.209 |
| measured ridge-on-sum | 0.722 | 0.583 | 0.572 | 0.583 |
| decode-then-count @L16 | 0.94 | 0.93 | 0.86 | 0.87 |

**2 · The rung IS high-d′** *(registered A-1: d′-high part ✓ on this rung)*: carrier = the quoted
word token (off −11), **d′_w 4.0–5.1** roughly flat in N; per-frame evidence AUROC ≈ 0.998–1.0
(probe acc 0.99). This is what "extraction removed" actually looks like at the carrier — text-MMRED
([2026-07-08c]) showed relational room/char evidence does NOT reach this even as text; CWE's
literal-token evidence does.

**3 · The frozen model largely escapes the wall here** *(registered A-2 ✗ on this rung — honest
refutation)*: 0.79 at N=8 (vs 0.20 on MMRED image/text), and the decay to N=64 (0.47) is FAR
slower than the fixed-d′ √N law (which dictates 0.69→0.21 over this range). The model also beats
the single-carrier probe ceiling at every N ≥ 16 (0.55 vs 0.43 @N16). Mechanism (from the
instruments): (i) the answer token can address evidence **by literal content match** — attention
selection with a literal key removes the effective-N dilution (counts stay ≤ 8 while fillers grow);
(ii) the native single-axis account breaks exactly here: d′_native 1.4 @L16, law-at-native predicts
0.30 ≪ measured 0.79, cos(native, w*) = −0.006, coherence only 0.60 — **the model's CWE readout is
not a single linear direction at one carrier** (contrast: text-MMRED closure 0.163 vs 0.165).
`native_axis/…/compare/native_compare.txt`.

**4 · E4 adequacy: rejected, in a NEW direction — the instrument's third regime.** Class-std ratio
explodes with N (**1.2 → 2.9 → 4.7 → 7.8–10.8** at L14/16), evidence-class kurtosis +2 to +12:
filler messages are near-deterministic while evidence messages vary ⇒ equal-covariance Gaussianity
fails ⇒ pred_iid *under*-predicts (measured ridge 0.58 vs pred 0.21 at N=64). Taxonomy now:
image-MMRED = adequacy ✓ law exact · HERBench = ✗ graded evidence (heavy right tail) · **CWE = ✗
degenerate noise class (variance ratio)**. The battery self-diagnoses all three correctly — quote
the closed form only where E4 passes, as standing policy.

**Ladder placement (thesis figure):** CWE sits ABOVE image/text-MMRED on both axes (d′ 4–5,
model 0.47–0.79): extraction perfect + content-addressable readout ⇒ smallest wall; MMRED
(relational binding through a carrier) ⇒ full wall at matched legibility; HERBench ⇒ wall +
supply-cap. The thesis claim sharpens: **the aggregation wall is specifically a property of
carrier-mediated relational evidence, not of counting per se.**

**Caveats.** Counts capped at 0–8 for all N (answers single-token; N-scaling of the *count range*
is a different axis, see B0b/C-range); digit-argmax reader (counts ≤ 8 only); n=150/N single seed;
N=64 cache is 4-layer (L14–20); adequacy failure means the pred columns are lower bounds by
diagnosed violation, not law failures; per-frame probe saturation caps measurable d′ (values are
floor estimates).

---

## [2026-07-08e] ✅📊 Text-MMRED N-scaling preview (N=16/24/40, states-only data) + C-range: the law tracks the collapse into the prior (pred 0.14/0.12/0.08 vs ridge 0.14/0.08/0.03; model at majority from N=16); the fact-phrased token interface verbalizes EVERY count 0–40 at 0.99–1.00 in-task

> Cheap unblocked preview of workstream B's math on text (no image data gen needed) + the plan's
> C-range design requirement (counts ≥10). **Data:** `data/mmred_text_longN/` (states-only,
> `generate_mmred_balanced.py --no-render`, 15 MB, N ∈ {16,24,40}, counts uniform 0–N, ~200–490
> samples/N; generator gained `--counts`). **Runs:** caches+locmap `outputs/ladder/text_mmred/
> longN/20260708_214148/N{16,24,40}/` (job **119532**; first attempt 119530 discarded to
> `outputs/_scratch/broken_runs/` — states-only loading left `frame_targets` empty via a
> `len(frames)`→`len(states)` bug, now fixed for count/first_occurrence/co_occupancy); parity
> `…/parity_longN/` (3 seeds, room+char @L16); C-range fact test
> `…/c_range_n40/20260708_213957/` (job **119531**, n=150, N=40 text, generation reader).

**1 · d′ vs N (joint pass, room@L16):** d′_AUC **1.91 (N=8, [2026-07-08c]) → 1.39 → 1.39 → 1.40**
(N=16/24/40) — a real ~25% supply drop from 8→16, then a PLATEAU (adequacy holds, std-ratio
0.9–1.1; d′_w declines 1.23→1.09 with frame-count-growing caches, d′_AUC is the stabler estimator
here). For B1's registered prediction this is the text preview: **joint d′ does not keep falling
with N — dilution of the pooled readout, not per-frame message quality, drives the collapse.**

**2 · The collapse itself follows the law with zero fitted parameters:**

| N (counts 0–N uniform) | 16 | 24 | 40 |
|---|---|---|---|
| law pred (d′_w, prior-mixed) | 0.141 | 0.116 | 0.078 |
| measured ridge-on-sum | 0.139 | 0.083 | 0.033 |
| model own-answer (argmax≤8 floor) | 0.062 | 0.035 | 0.020 (gen reader: **0.027**) |
| majority | 0.059 | 0.040 | 0.024 |

Model ≈ majority from N=16 on — fully prior-locked (the B-3 signature, on text, before any image
long-N data exists).

**3 · C-range established (the C2 contrast column's foundation):** on N=40 TEXT frames in-task,
writing "Note: C spent exactly k steps in the R." yields emitted **k at 0.987 (k=gold) / 1.000
(counterfactual k, every value 0–40 incl. all two-digit)** while the base model scores 0.027
(MAE 13.4). Multi-digit verbalization via the token interface is unconditionally available across
the full C2 test range; the C2/C3 activation-injection routes now have a hard 1.00-vs-0.03
contrast pair to beat/match. (Fact-phrasing dependence logged in [2026-07-08b] still applies.)

**Caveats.** Text only (image long-N gated on B0b/Q1); the argmax own-answer column reads only
digits ≤8 (floor for counts >8 — the generation reader is the honest behavioral number);
d′-vs-N confounds n_samples (200 vs 400 at N=8) — frame counts (3.2k–8k) keep the per-frame
estimate well-powered, and d′_AUC agreement across N=16/24/40 supports the plateau reading;
counts-prior differs from the planned B0b low-band design (uniform 0–N here).

---

## [2026-07-09] ✅📊 A1 addendum — LOCUS-COMPLETENESS / block read: text-MMRED's carrier is DISTRIBUTED, so the [2026-07-08c] single-token d′≈1.8 was a ~0.5 underestimate; the complete text supply is d′≈2.45 @L16 ≈ image-MMRED's 2.47 — "legibility ≠ higher d′" survives but "text is write-limited/lower-supply" is REVISED to "same supply, spread over more tokens"

> **Motivation:** [2026-07-08c] measured text carrier d′ at the single room token (image's locus) and
> got 1.8, concluding "write-limited." But the text locmap showed evidence *distributed* across ~6
> question tokens (char > room, unlike image) → any single-token d′ is a LOWER BOUND (the cooc
> distributed-carrier signature). This runs the block read to measure the pooled supply.
> **New code:** `experiments/glstm/block_read_completeness.py` (CPU; per-locus shrinkage-LDA d′,
> two block estimators [score-concat two-stage + PCA256-concat], incremental d′-vs-K curve, block
> dtc, zero-param law parity, E4 on the block; 3 sample-disjoint seeds, group-split by sample).
> **Cache:** job **119762** (rtx6k, n=800 [seq8 all_uniform pool ran dry at LIMIT=1200], 7 offsets
> {0,−2,−5,−8,−9,−11,−13} × L14/16/18, save-messages, 1.4 GB) →
> `outputs/ladder/text_mmred/locmap_cache/blockcache_20260709_191746/count/messages_cache.pt`.
> **Analysis:** `outputs/ladder/text_mmred/block_read/20260709_192440/` (report.txt, results.csv,
> incremental.csv, block_read.png).

**1 · Distributed carrier CONFIRMED — single-token underestimated the supply by ~0.5 d′.** At the
peak transfer layer L16, whitened d′ rises **1.97 (best single locus) → 2.45 (block score-concat,
d′_auc 2.49)** — gain **+0.47** (> the 0.4 complementary threshold). PCA256-concat agrees (2.29;
conservative by truncation). The incremental curve (greedy add by single-locus d′) **plateaus at
K≈4–5**: 1.99 → 2.25 → 2.36 → 2.45 → 2.47 → 2.46 → 2.45 — ~4 question tokens carry the supply
(room off−9, plus the naive-hot off−5/−2 and char off−13), the rest redundant. L14/L18 blocks are
lower (2.00–2.02, small +0.11–0.18 gains) — the distributed lift is specific to the peak layer.

**2 · Corrected text supply ≈ image supply.** Complete text carrier d′ **2.45 @L16 ≈ image-MMRED
deployed room-token 2.47** ([2026-07-03], n=600). ⇒ **REVISION of [2026-07-08c]:** text is NOT
uniquely low-supply/"write-limited"; it has the *same* carrier supply as image, merely spread over
more tokens (image concentrates on the room token → single-token ≈ complete there; text distributes
→ single-token undercounts). The "perfect legibility does NOT raise d′" headline SURVIVES (text
2.45 ≯ image 2.47), but the specific "text 1.8 < image 2.5" gap was a **locus-completeness
artifact**, exactly as [2026-07-08c]'s own caveat flagged. The fence-null-on-text result
([2026-07-08c] §3) is untouched — interference still isn't the text binder.

**3 · Last-token whitened d′ measured (answers the standing question): near-dead in the transfer
window.** Per-frame evidence d′ at offset 0 = **0.49 @L14/L16**, rising only to 0.95 @L18 — far
below room/char, confirming the last token is the prediction hub, not the aggregation recipient
(the late-layer naive-SNR bump 0.81@L24 does NOT correspond to strong *whitened per-frame* evidence
in-window). Adding the last token to the block changes nothing (K=7 = K=4).

**4 · The aggregation gap is larger than logged.** Block dtc (charitable per-frame-classifier →
sum) = **0.589 @L16** (vs 0.514 room+char only, [2026-07-03] era) — the extra loci carry real
complementary count evidence. Full ladder @L16: **model 0.158 < law-pred(block d′) 0.376 ≈
ridge-on-sum 0.460 < dtc(block) 0.589 < per-frame ~0.94.** Model-vs-achievable gap ~3.7×.

**5 · E4 at the peak layer: mild FAIL — separability valid, closed-form accuracy slightly
optimistic.** L16 block excess kurtosis **+1.07** (just over the 1.0 flag; HERBench was +2.6…+35),
skew +0.18, std-ratio 1.07. **Does NOT invalidate the d′/distributed-carrier finding:** (i) d′ is
descriptive (gap/width separability), Gaussian-free; (ii) the internal self-check PASSES — d′_w
2.449 ≈ d′_auc 2.488 (1.6% apart; the exact test HERBench failed by 20–40%), so the classes ARE
approximately Gaussian, just mildly heavy-tailed; (iii) the only consequence is the closed-form
accuracy runs slightly low (pred 0.376 < measured ridge 0.460 at L16). L14/L18 pass E4 outright
(kurt +0.53/+0.74) at their lower block d′ (2.00). Quote 2.45 as separability; quote the accuracy
law with the adequacy caveat at L16.

**Registered-prediction verdict (plan A1-followup).** "Block read either recovers to ~4 (distributed,
mismeasured) or saturates ~2 (write-capped)": **DISTRIBUTED at the peak layer** (2.45, +0.47) — the
first branch, but landing at ~2.45 not ~4: text supply equals image, not more. The write-limited
label is downgraded to "distributed-carrier, supply ≈ image."

**Caveats.** n=800 single cache (6.4k frame examples — well-powered; d′ is a converging lower bound);
block d′ uses a two-stage estimator (stage-1 locus selection is held-out and PCA-concat corroborates,
but a small optimism can't be fully excluded); E4 mild-fail at L16 (accuracy law caveated there);
steps task only. Instruments: [2026-07-03b/c] parity engine conventions reused (shrinkage-LDA d′,
√2·Φ⁻¹(AUC) cross-check, prior-mixed law).

**Addendum (same day) — SYMMETRIC IMAGE CONTROL: concentrated vs distributed carrier, cleanly
dissociated.** Ran the identical 7-offset block read on a fresh IMAGE deployed cache (job **119781**,
rtx6k, n=800, same offsets/layers) → `outputs/frame_axis/probes/carrier_message/count_msgcache_blockctrl/20260709_220729/`
+ analysis `.../block_read_20260709_2231*/`. **Image L16: single room token d′ 2.57 → block 2.71,
gain +0.14 (REDUNDANT/concentrated)** — the room token alone is ~95% of the pooled supply, vs text's
+0.47 (distributed). Incremental curves both plateau by K≈4 but from opposite starts (image flat-high,
text rising). **The dissociation is the clean result:** image writes the relational evidence into ONE
carrier token; text spreads it over ~4. Two further reads: (i) **complete supplies are comparable,
image slightly higher (2.71 vs text 2.45)** — so "perfect legibility doesn't raise d′" is now stronger
than survival: the *visual* pipeline binds character→room into its carrier at least as well as text,
killing any "text should be higher because it's legible" intuition; (ii) **E4 mild-fails at L16 in
BOTH modalities** (image kurt +1.05, text +1.07) with L14/L18 passing — the peak-layer adequacy wobble
is a block-score-estimator/layer property, not modality-specific; d′_w≈d′_auc in both (image 2.57≈2.63
single). Image dtc(block) 0.566, model 0.219 — same ~2.6× aggregation gap. **Verdict:** the [2026-07-08c]
"text write-limited" revision stands and is now symmetric — image=concentrated-carrier, text=distributed-
carrier, comparable supply; single-token reads are complete for image, ~0.5 d′ low for text.

---

## References (GNN / set-aggregation grounding)

> External literature this work builds on. Relevance noted per entry; arXiv IDs verified 2026-06-13.

- **DeepSets** — Zaheer et al., *Deep Sets*, NeurIPS 2017. [arXiv:1703.06114](https://arxiv.org/abs/1703.06114).
  The `ρ(Σ φ(x_i))` form our sum adapter instantiates; sum is the universal permutation-invariant pool.
- **Width ≥ set-size bound** — Wagstaff et al., *On the Limitations of Representing Functions on Sets*,
  ICML 2019. [arXiv:1901.09006](https://arxiv.org/abs/1901.09006). Proves continuous set-function
  representation needs latent dim ≥ max #elements — **exactly our d_mem-saturates-at-N=8 result**.
- **GIN (sum > mean > max)** — Xu et al., *How Powerful are Graph Neural Networks?*, ICLR 2019.
  [arXiv:1810.00826](https://arxiv.org/abs/1810.00826). Counting/multiplicity is the canonical task
  separating sum from mean — our pooling ablation (sum/pna 1.00 vs mean/softmax 0.76) inside a VLM.
- **PNA** — Corso et al., *Principal Neighbourhood Aggregation for Graph Nets*, NeurIPS 2020.
  [arXiv:2004.05718](https://arxiv.org/abs/2004.05718). Multiple aggregators + degree scalers; its
  degree-scaler×mean reproduces sum, so PNA = sum on our task (safe task-agnostic readout).
- **Over-squashing: width/depth/topology** — Di Giovanni et al., ICML 2023.
  [arXiv:2302.02941](https://arxiv.org/abs/2302.02941). Width mitigates over-squashing (matches our
  capacity knob); depth does not. Frames the two-knob (operation + width) story.
- **Over-squashing (origin)** — Alon & Yahav, *On the Bottleneck of GNNs…*, ICLR 2021.
  [arXiv:2006.05205](https://arxiv.org/abs/2006.05205). Defines the bottleneck the thesis ports to VLM attention.
- **Transformer last-token collapse** — Barbero et al., *Transformers need glasses! Information
  over-squashing in language tasks*, NeurIPS 2024. [arXiv:2406.04267](https://arxiv.org/abs/2406.04267).
  Proves decoder last-token **representational collapse**, ties it to GNN over-squashing, and shows it
  causes failures *specifically in counting/copying* (exacerbated by low-precision FP) — the theory behind
  our early-diagnosis last-token cosine-collapse diagnostic and the bf16 caveat.
- **gLSTM** — *Mitigating Over-Squashing by Increasing Storage Capacity*, 2025.
  [arXiv:2510.08450](https://arxiv.org/abs/2510.08450). The capacity/associative-memory approach our
  matched controls show is over-engineered for MMRED (its addressing is dispensable; sum suffices).

### Count/size-EXTRAPOLATION grounding (added 2026-06-23; verified via lit-search subagent)
- **★ Universal Approximation of Functions on Sets** — Wagstaff et al., JMLR 2022, v23(151).
  [arXiv:2107.01959](https://arxiv.org/abs/2107.01959). Sharpens the 2019 bound: if latent dim is even *one*
  below max set size, Deep Sets does **no better than a constant** on worst-case piecewise-affine targets.
  Theoretical backbone for "an additive readout must scale width with the count range or provably breaks."
- **★ When Can Transformers Count to n?** — Yehudai et al., 2024.
  [arXiv:2407.15160](https://arxiv.org/abs/2407.15160). Sharp phase transition: counting is learnable iff
  **embedding dim ≥ vocabulary**, else numerically unstable + catastrophic OOD. The transformer echo of the
  Wagstaff dim-bound and of our "readout caps beyond the trained count range."
- **★ Unveiling the Visual Counting Bottleneck in VLMs ("fractured magnitude hypothesis")** — Pang et al., 2026.
  [arXiv:2605.30170](https://arxiv.org/abs/2605.30170) *(⚠ verify the 2605 listing resolves)*. **Direct
  corroboration of our result**: VLMs fail to *extrapolate* counts **not at perception** (magnitude reps stay
  linearly separable into the OOD regime) **but at the symbolic-mapping/readout stage** — exactly our "the readout,
  not the aggregation, is the OOD wall." Strongly motivates a structured/additive readout over a learned one.
- **Size generalization in GNNs** — Yehudai et al., *From Local Structures to Size Generalization in GNNs*, ICML 2021.
  [arXiv:2010.08853](https://arxiv.org/abs/2010.08853). "Bad global minima" fit small graphs but fail on larger —
  the GNN analogue of our count-only-sum instability (fits ≤4, caps ≥5).
- **Set Transformer** — Lee et al., ICML 2019. [arXiv:1810.00825](https://arxiv.org/abs/1810.00825) *(verify ID)*.
  Attention-based *learned* permutation-invariant pooling — the "learned aggregator/readout" foil our fixed-extensive
  readout beats on extrapolation.

**How these frame our contribution:** the lit establishes (a) representational dim bounds (Wagstaff; Transformers-count-to-n)
and (b) that VLM count failure is a *readout/symbolic* problem, not perception (Pang). Our empirical contribution is the
*learnability/extrapolation* counterpart: among readouts that all satisfy the representational bounds, **only a
parameter-free fixed extensive reduction (sum / soft-OR) of a per-frame-supervised quantity extrapolates; every learned ρ
(canonical DeepSets, even linear-over-extensive-channels) does not** — across 3 tasks, multi-seed, on a frozen VLM.

### Related work / novelty positioning (lit search 2026-06-23)
> Verdict: every *ingredient* is published; the *combination* (read question-conditioned per-frame states from a
> frozen autoregressive VLM → DeepSets/soft-OR aggregate → inject-back, as over-squashing relief) has no direct
> precedent found. These are snippet-level finds — verify the starred ones before citing as closest prior.

- **★ Can VLMs Count? Attention-Based Interventions** — [arXiv:2511.17722](https://arxiv.org/abs/2511.17722).
  **Concurrent corroboration of our diagnosis**: VLM counting = "locate-but-not-enumerate under cognitive load"
  (= our extraction-fine / crowding-kills-aggregation). They fix via attention reweighting; we use an explicit
  permutation-invariant aggregator + extraction-vs-aggregation decomposition. **Must cite.**
- **VLM Can't Even Count to 20** — [arXiv:2510.04401](https://arxiv.org/abs/2510.04401). Compositional counting failures.
- **GroundCount** — [arXiv:2603.10978](https://arxiv.org/pdf/2603.10978). Counting-hallucination mitigation via detection grounding.
- **Set Transformer** — Lee et al. 2019, [arXiv:1810.00825](https://arxiv.org/pdf/1810.00825). Permutation-invariant
  pooling; **canonical demo counts unique elements in an image** (= distinct-count / rooms_visited). Cite as the
  perm-invariant-counting precedent + alternative aggregator.
- **★ ILSE (Inter-Layer Structural Encoders)** — [arXiv:2603.22665](https://arxiv.org/html/2603.22665v1). DeepSets
  over a **frozen LLM's layer** representations — closest "DeepSets-adapter-on-frozen-states" prior (over layers, not frames).
- **Set-LLM: A Permutation-Invariant LLM** — [arXiv:2505.15433](https://arxiv.org/pdf/2505.15433).
- **How Multimodal LLMs Solve Image Tasks** — [arXiv:2508.20279](https://arxiv.org/pdf/2508.20279). Documents the
  **image-before-question ordering effect** (image tokens query-blind ⇒ lower probe acc; last token most decodable) —
  the mechanism our question-first design exploits.
- **QG-VTC** ([arXiv:2504.00654](https://arxiv.org/pdf/2504.00654)), **PARCEL** ([arXiv:2605.30126](https://arxiv.org/abs/2605.30126)) —
  query-conditioned visual-token pooling, but for **efficiency/compression**, not aggregation-for-counting.
- **Activation steering / read-inject lineage** — Activation Engineering ([arXiv:2308.10248](https://arxiv.org/html/2308.10248v5)),
  REAL ([arXiv:2506.08359](https://arxiv.org/abs/2506.08359)), InversionView ([arXiv:2405.17653](https://arxiv.org/pdf/2405.17653)).
  Our read-aggregate-**inject-back** resembles this.

### Candidate methods for the distractor-selection problem (proposed next directions, not yet tried)
- **Slot Attention** — Locatello et al., *Object-Centric Learning with Slot Attention*, NeurIPS 2020.
  [arXiv:2006.15055](https://arxiv.org/abs/2006.15055). Competitive routing (softmax across slots, not
  inputs) → joint evidence/distractor separation that escapes per-element compounding. Top candidate (pair
  with sum readout).
- **Set Transformer (ISAB/PMA)** — Lee et al., ICML 2019. [arXiv:1810.00825](https://arxiv.org/abs/1810.00825).
  Learnable permutation-invariant pooling via attention over the whole set — global, not per-element.
- **Signed-GCN** — Derr et al., *Signed Graph Convolutional Network*, ICDM 2018.
  [arXiv:1808.06354](https://arxiv.org/abs/1808.06354). Positive/negative relations aggregated separately —
  the learnable form of the oracle pos/neg two-stream (96% bound).
- **R-GCN** — Schlichtkrull et al., *Modeling Relational Data with GCNs*, ESWC 2018.
  [arXiv:1703.06103](https://arxiv.org/abs/1703.06103). Multi-relational aggregation (evidence vs distractor
  as edge types).
- **GAT** — Veličković et al., ICLR 2018. [arXiv:1710.10903](https://arxiv.org/abs/1710.10903). Listed as a
  **ruled-out** baseline: its attention is softmax-normalized, reintroducing the count-blind mean.

---
## Maintenance & provenance notes

> **How to update (Claude, read this):**
> - Only update when I explicitly ask ("log this run", "update results").
> - Append to the Experiment Log table — never rewrite past rows.
> - Every row's metric must come from a real output dir, named in the row.
> - Put *why it matters* in the Synthesis section, not the table.
> - If a result is uncertain or suspicious, say so in the Notes — don't launder it into a clean number.

> **⚠️ Backfill provenance (2026-06-12):** the Experiment Log was reconstructed by scanning the existing
> `outputs_*/` / `output_*/` trees (summary / eval_metrics / accuracy_by_* / README per run). Numbers trace
> to the named dir but were **not all hand-verified** — spot-check any number before it goes in the thesis.
> Several headline accuracies are **trained-on-clean** or **oracle-masked upper bounds**, not deployable
> results; these are flagged ⚠️ / 📊 and explained in Synthesis.

### Backfill checklist
- [x] Infer common run structure (config.json / summary.* / eval_metrics.csv / accuracy_by_*.csv / README.md).
- [x] Define THE metric (exact-match count accuracy, by evidence_count; + gold_margin, MAE, fix/break).
- [x] Backfill the Experiment Log from existing dirs (oldest → newest).
- [x] Write first-pass Synthesis.
- [ ] **Spot-check the flagged numbers** (trained-on-clean / oracle / normalization) before any go in the thesis.

## [2026-07-10] ✅📊 A1-fu3 — COOC block read (locus completeness on the third task): block d′ 3.43 @L14 vs best single (char2, off−13) 3.05 — gain +0.38/+0.42, BORDERLINE at the 0.4 threshold; co-occupancy sits BETWEEN image-MMRED (concentrated, +0.14) and text-MMRED (distributed, +0.47) on the carrier-concentration axis

> **Motivation:** [2026-07-09] dissociated concentrated (image, block gain +0.14) vs distributed
> (text, +0.47) carriers at equal supply. Cooc's locmap was known-distributed over the two NAME
> tokens ([2026-07-04] 2-name block d′ 3.19). This runs the full 6-locus block read on the existing
> big cooc cache to complete the three-task picture. Pure CPU, job **119985** (4h_0g, 5 min).
> **Cache:** `outputs/frame_axis/probes/carrier_message/cooc_msgcache_big/co_occupancy/`
> (n=1080, offsets {0,10,11,13,14,15}, L14/16 — note: memo said "n=1500"; the real cache is n=1080).
> **Analysis:** `outputs/frame_axis/probes/carrier_message/cooc_block_read/20260710_210215/`
> (report.txt, results.csv, incremental.csv, block_read.png).

1. **Complete cooc supply d′ ≈ 3.43 @L14 (score-concat; d′_auc 3.24; PCA256 3.30), 3.41 @L16.**
   Best single locus is the char2 name token (off−13): 3.05 @L14 / 2.99 @L16. Block gain **+0.38
   (L14) / +0.42 (L16)** — straddles the pre-set 0.4 "complementary" line: the second name region
   (off−15/−10/−11) adds a real but modest ~0.4 d′; the incremental curve plateaus at K≈2 (L14:
   3.05→3.42 with ONE added locus, then flat; L16 needs K≈4 to plateau at 3.42).
2. **Three-task carrier-concentration ordering (same instrument, same estimator):** image-MMRED
   +0.14 (one room token ≈ complete) < cooc +0.38/+0.42 (two name tokens carry it) < text-MMRED
   +0.47 (~4 tokens). Cooc's "distributed" is really *two-locus*: exactly the two entities the
   question binds — consistent with the [2026-07-04] 2-name block (3.19; the 6-locus block adds
   +0.24 over it).
3. **Ladder replicates on the corrected supply:** model 0.138 < law-pred(3.43) 0.476 ≈ ridge-on-sum
   0.534±0.028 (L14; L16 0.566±0.004) < dtc(block) 0.648±0.010 < per-frame ≈ [2026-07-04] values.
   Model-vs-achievable gap ~4.7×. **E4 PASSES at both layers** (skewE +0.12/+0.16, kurtE
   +0.38/+0.35, std-ratio 0.83/0.80) — unlike the L16 mild-fails on text/image blocks, the cooc
   block projections are clean-Gaussian, so the law is licensed without caveat here.
4. **Verdict vs the informal expectation** ("cooc is 2-name distributed; block should recover
   ~3.2+"): ✓ — block 3.43 ≥ the 2-name 3.19, gain concentrated in the name loci; single-token
   reads of cooc undercount supply by ~0.4 d′ (quote 3.4, not 3.0).

**Caveats.** Single cache (n=1080, one seed pool), L14/16 only (no L18 in this cache); offsets are
the cooc locmap's picks, not a dense sweep — a wider sweep could find small extra loci; same
two-stage estimator optimism caveat as [2026-07-09] (PCA-concat corroborates at 3.30).

## [2026-07-10b] ✅📊 A1-fu1 — TEXT MULTIPASS (isolated per-frame forwards): the text carrier "write cap" is NOT a write cap — isolation lifts room-token d′ 1.97→7.9 @L16 (block 2.45→7.9; d′_auc saturates its 5.26 estimator ceiling = perfect separation), and the multipass+sum solution scores 0.965 behaviorally; the joint-pass supply loss is SHARED-FORWARD-induced, not attention-edge-carried (fence was NULL)

> **Motivation:** [2026-07-08c] measured joint text carrier d′ 1.8 (block-corrected 2.45,
> [2026-07-09]) and called text "write-limited"; the definitive test is whether the model's
> carrier-write process stays capped when each frame is processed ALONE (the image/InternVL
> multipass unlock: joint 1.9 → multipass 6.4–6.6). **New code:**
> `experiments/glstm/text_multipass_cache.py` (isolated single-frame text forwards, identical
> question suffix, same 7 carrier offsets/layers as the joint blockcache; schema-compatible
> messages_cache.pt + per-frame perception + multipass-sum behavioral score).
> **Runs:** cache job **119991** (l40s, 15 min, n=400 × 8 frames, L14/16/18) →
> `outputs/ladder/text_mmred/multipass_cache/20260710_211249/`; block-read job **120010** →
> `outputs/ladder/text_mmred/block_read_multipass/20260710_215350/` (report.txt, results.csv).

1. **Isolation unlocks the text supply ~3×:** room token (off−9) d′_w **7.86 @L16** (joint block
   2.45); L14 3.20; L18 11.2 (ceiling-inflated). d′_auc pegs at **5.26 = the estimator's ceiling**
   (AUC ≈ 1 exactly at n=3200 frame examples) — quote "d′ ≥ 5.3, d′_w ≈ 7.9". Block gain over the
   best single locus is **+0.00–0.04 @L14/16** — under isolation the carrier RE-CONCENTRATES into
   the room token (the joint-pass distribution over ~4 tokens was itself a joint-processing
   artifact; L18 shows a +0.49 tail).
2. **Behavioral closure:** per-frame single-frame perception = **0.995**; multipass-sum solution
   (sum of clipped per-frame answers) = **0.965** vs joint model 0.158–0.196. dtc(block) 0.987,
   ridge-on-sum 0.938, law-pred 0.872 @L18 — the whole ladder moves to the high-d′ regime.
3. **Mechanism dissociation completed:** fence (blocking cross-frame attention edges) was NULL on
   text ([2026-07-08c] §3) and HURTS on image — yet full isolation is a 3× d′ unlock in both
   modalities. ⇒ the joint-pass supply loss is a property of the shared forward pass (context
   dilution / attention-mass competition / in-context binding load), NOT of direct cross-frame
   attention edges. "Write-limited" is retired; the cap is **joint-context-induced**, and it is
   modality-general (text replicates image).
4. **Registered-framing verdict (A1-fu1 "definitive write-cap test"):** ✗ for the write-cap
   branch — the carrier-write process is nearly noiseless in isolation; the supply cap lives in
   joint processing. (This is the informative refutation the test was designed to force.)

**Caveats.** E4 FAILS at all layers (kurtE +0.8…+2.6, std-ratio 0.53–0.67 — evidence-class
projections are tighter than no-evidence): quote the law rows with the adequacy caveat; the d′
numbers are separability statements (d′_w vs d′_auc diverge at ceiling — report both). L18's 11.2
is finite-n inflated (perfect separation). Single-frame prompts say "1 frames" (template kept
identical to joint up to the frame count) — an in-distribution-ness gap vs joint remains possible
in principle, but the image multipass shares it and the per-frame perception 0.995 bounds it.
Steps task only, n=400.

## [2026-07-10c] ✅📊 A1-fu2 — EASY-TEXT minimal pair ("Frame i: Michael@Park, Sara@Kitchen."): putting the character→room binding INTO THE SURFACE FORM does NOT unlock the supply — block d′ 2.89 @L16 (standard text 2.45), model 0.215 (unchanged); the binding-format account of the text cap is REFUTED

> **Motivation:** text-CWE (surface literal match) escaped the wall with d′ 4–5 while text-MMRED
> (binding required: room→occupants lists) capped at ~2.45. If in-context BINDING is the cap,
> rendering each frame as pre-bound "Char@Room" pairs should push d′ toward CWE. Registered
> expectation (approved item A1-fu2): "expect d′ → CWE-like if the binding account is right."
> **New code:** `--text-style compact` in `probe_frame_to_carrier_message.py`.
> **Runs:** cache job **119992** (n=400, L14/16/18/20, 7 offsets; job died at the report stage in
> the 21:29 quota incident but the 957MB messages_cache.pt is complete and verified) →
> `outputs/ladder/text_mmred_easy/locmap_cache/20260710_211249/`; block-read job **120011** →
> `outputs/ladder/text_mmred_easy/block_read/20260710_215350/`.

1. **d′ barely moves:** best single locus = room token (off−9) **2.51 @L16** (d′_auc 2.67); block
   score-concat **2.89±0.06** vs standard-text block 2.45 — a ~+0.4 improvement, far from the
   CWE regime (4–5). L14 block 2.41. The carrier also RE-CONCENTRATES mildly (block gain +0.38
   vs standard text's +0.47; room token dominant, char token 1.45).
2. **Behavior unchanged:** model own-answer 0.215 (from the cache's digit-argmax field; standard
   text 0.158–0.196, image 0.219) with the classic undercount profile (1.00 at K=2, 0.00 at K≥5).
3. **Verdict on the registered expectation: ✗ REFUTED** — surface-form binding does not produce
   CWE-like d′. Combined with [2026-07-10b] (multipass unlocks text to d′≈7.9) and the fence null,
   the three-way dissociation is now: NOT attention edges (fence null), NOT binding format (this),
   NOT extraction/legibility ([2026-07-08c]) — the cap is **joint-context processing of N frames
   per se** (attention-mass competition / context dilution in a shared forward). What separates
   CWE is presumably its literal-match ADDRESSING (the target word is in the question), not its
   easier binding.
4. Ladder @L16: model 0.215 < law-pred 0.416 ≈ ridge 0.512 < dtc(block) 0.665.

**Caveats.** E4 FAILs at L14/16 (kurtE +1.1/+1.3) — law rows caveated; d′ comparison (2.89 vs
2.45) is cross-cache (same n=400, same seed pool, same estimator — but different prompts, so
token-position offsets differ slightly in what they capture); in-run locmap sweep (MAXOFF 20)
was lost to the quota kill — the 7 cached offsets cover the known carrier region; the compact
rendering sorts pairs by character name (canonical order), which if anything should HELP.

## [2026-07-10d] ✅📊 C1b — FACT-PHRASING ROBUSTNESS GRID: the token interface is gated on (a) PREDICATE MATCH with the question and (b) PRE-QUESTION POSITION — given those it is fully robust (words≈digits 1.00, source-attributed 0.99, max-distance 0.99, distractor-fact 0.97); paraphrased predicates and post-question placement collapse to 0.00 OOD

> **Runs:** job **120012** (rerun after the quota kill; the 21:15 attempt's partial rows are in
> `outputs/readout/c1b_phrasing/20260710_211500/`) →
> `outputs/readout/c1b_phrasing/20260710_215647/` (report.txt, summary.json, rows.json).
> n=120 image samples (steps N=8, deployed visual context), counterfactual targets (in-range
> and OOD {11,13,17,23,29,34,40}) so the score isolates fact-READING, not counting. Generation
> reader. 12 arms. New phrasings/positions/distractor logic added to
> `experiments/readout/c1_token_interface.py`.

| arm | phrasing / manipulation | acc vs target (in / OOD) |
|---|---|---|
| fact (canonical) | "Note: C spent exactly K steps in the R." | 0.942 / 0.967 (replicates C1) |
| **words** | count as an English word ("seventeen") | **0.992 / 1.000** |
| src | "An automated frame counter reports: …" | — / 0.992 |
| top | fact BEFORE everything (max distance from answer) | — / 0.992 |
| dis | + same-form distractor fact (other char+room, other count) | 0.925 / 0.967 |
| para1 | "For reference, C was in the R in K of the frames." | 0.225 / **0.000** |
| para2 | "It is known that C appears in the R exactly K times." | — / **0.000** |
| factq | canonical fact AFTER the question | — / **0.000** |

**Readings.**
1. **The gate is predicate match, not "fact-ness":** every working arm phrases the count with the
   QUESTION'S OWN predicate ("spent … steps in the R" ← "How many steps did C spend in the R?").
   Both failing paraphrases state the same quantity under a different predicate ("in K of the
   frames" / "appears K times") and are IGNORED (model falls back toward counting: acc-vs-gold
   0.13–0.18, MAE-vs-target ≈ 21). The interface does lexical/semantic retrieval keyed on the
   question, not general fact integration.
2. **Position: the fact must precede the question.** Identical canonical sentence placed after the
   question = 0.00. Mechanistically consistent with the carrier architecture: question-token
   carriers aggregate evidence via CAUSAL attention (they cannot see later tokens), and the
   answer readout reads carriers — content the carriers never saw effectively doesn't exist.
   (Distance is irrelevant — 'top' at maximum distance scores 0.992 — only ORDER matters.)
3. **Within the gate, the interface is format-agnostic and selective:** number-words = digits
   (1.000 OOD — C2's digit-token assumption is not load-bearing for the bar), source attribution
   free, and the model BINDS the right fact under a competing same-form distractor (0.967 OOD;
   only weak spot: target K=0 with a distractor present → 0.58 — the distractor's nonzero count
   leaks when the true fact says zero).
4. **C2 design confirmed:** the C2 fact-slot site uses the canonical predicate-matched sentence
   pre-question — inside the working regime. (And the tally-adapter→text route must render
   predicate-matched facts, per Q5 amendment (b) — now with direct evidence.)

**Caveats.** n=120/arm, single task/model; argmax column is 0.00 on OOD by construction (single
digit head) — generation reader is the metric; paraphrase set is 2 items (predicate-match reading
inferred from a consistent pattern, not a parametric sweep).

## [2026-07-10e] ✅📊 Evidence-only behavioral number (flagged-gap closer): with EVERY shown frame being evidence (gold = N, zero selection load), the frozen model still CRUSHES — 1.00 @N=1, 0.05–0.26 @N=2–5, 0.00 @N≥6 (answers ~4.6 when shown 8 evidence frames); the wall is aggregation/readout, not distractor selection

> **Run:** job **120029** → `outputs/ladder/evidence_only_behavior/20260710_221821/N{1..8}/`
> (report.txt, rows.json, summary.json per N). Data `data/mmred_images_park_evidence_only_seq1_8`
> (all frames evidence, gold = seq_len), n=19/N (n=152 total), native res, generation reader
> (`experiments/readout/behavior_vs_n.py`), seed 0.

| N | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| exact | 1.000 | 0.053 | 0.263 | 0.105 | 0.158 | 0.000 | 0.000 | 0.000 |
| MAE | 0.00 | 0.95 | 0.79 | 1.05 | 1.16 | 1.74 | 3.16 | 3.42 |

At N=8 the prediction distribution is {3–7}, mode 4, mean 4.58 — the canonical undercount/crush
(≈ the "number axis saturates ~5" readout wall) with the count signal *maximal* (every frame is
evidence, extraction validated ≥0.99 elsewhere). Selection/distractor load is not required to
produce the wall; pure aggregation+readout reproduces it. (N=2's 0.053 dip below N=3: model says
1 or 3, rarely 2 — a known digit-prior quirk, small n.) Closes the gap flagged in the 2026-07-09
approvals ("frozen evidence-only behavioral number at N=8").

**Caveats.** n=19/N (the approved cheap read); gold is constant within each N cell so "exact"
is per-cell hit rate on a degenerate target — the MAE row and the prediction distribution carry
the signal; native-res (not 392) by design for comparability with historical N=8 numbers.

## [2026-07-10f] ✅📊 B3 — frozen-model behavior vs N (images @392px, generation reader): exact-match 0.173 → 0.127 → 0.073 → 0.053 → 0.020 for N=8→128; the emitted range is CLAMPED at mean ≈3 regardless of N (gold mean grows 4→27.5) while rank correlation with gold HOLDS at ~0.75 up to N=64 — the readout-range wall, not signal loss, drives the collapse

> **Runs:** jobs **120030** (N=8/16/32) / **120031** (N=64) / **120032** (N=128) →
> `outputs/ladder/image_longN/behavior/N{8,16,32,64,128}/<ts>/` (report.txt, rows.json,
> summary.json). n=150/N, seed 1 (disjoint from the cache draw), data `mmred_longN_park`
> (N=8: `mmred_images_park`), resize 392, greedy generation + first-integer parse
> (`experiments/readout/behavior_vs_n.py` — the B3/C multi-digit reader; parse-fail 0.000
> everywhere, N=128 ≈ 25k tokens fits and runs 7.7 s/sample).

| N | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| exact-match | 0.173 | 0.127 | 0.073 | 0.053 | 0.020 |
| MAE | 1.73 | 3.28 | 6.37 | 12.35 | 24.57 |
| mean pred (mean gold) | 3.0 (4.1) | 3.3 (6.1) | 3.5 (9.5) | 2.8 (15.0) | 3.2 (27.5) |
| corr(pred, gold) | 0.74 | 0.70 | 0.77 | 0.78 | 0.33 |

**Readings.** (1) MAE ∝ N (ratio ≈ 0.20±0.02) because the answer DISTRIBUTION is N-invariant:
the model emits ~{1..7} (mode 3–4) at every N — the [2026-07-10e] evidence-only crush and the
"number axis saturates ~5" readout wall, now measured to N=128. (2) The ordinal signal SURVIVES
to N=64 (corr ~0.75): relative count information reaches the readout but cannot be verbalized
outside the clamped range — readout-limited, not extraction- or transport-limited (multipass
per-frame perception stays 0.985+ at all N, [2026-07-10 mp caches]). At N=128 even the ordinal
signal thins (0.33; residual exact hits only at K∈{0,128}). (3) Registered prediction B-3
("collapse toward prior as the law dictates"): collapse ✓ and monotone; the quantitative
2Φ(d′/2√N)−1 closure will be evaluated against the B1 joint-cache d′ when those land — note the
clamp means the mechanism is range-truncation on top of d′ dilution, so expect the law (which
models dilution only) to sit ABOVE the measured exact-match at large N.

**Caveats.** n=150/N; counts span {0..8}∪spread (not uniform 0..N) — MAE mixes bands; corr at
N=128 computed over the wider gold range (0–128) so its drop partly reflects range, not only
signal; seed-1 sample draw.

## [2026-07-10g] ✅ A4 pilot gate — mmred_natural passes the pre-registered extraction gate after one judge-curation round: per-frame look-again accuracy 0.998–1.000 in all 4 (diversity × similarity) cells

> **Pipeline:** COCO val2017 needles (dog) + distractors (near = cat-no-dog, far = no-animal),
> per-frame GT by construction; 4 cells ident/dist × far/near, N=8, counts 0–8.
> **Round 1** (uncurated, 50 samples/cell, job 120028; judge 120034): AUROC 0.986–0.997 but
> accuracy 0.922–0.960 — 3/4 cells FAIL the ≥0.95 gate via FN 8–15% (a tail of small/occluded
> COCO dogs; FP ≈ 0). **Curation:** dropped 23/170 needle images with judge P(yes)<0.9 and 9/548
> distractors with P(yes)>0.1 (`data/mmred_natural/judge_curation.json`).
> **Round 2** (curated pools, `data/mmred_natural_v2/`, job 120038; judge **120039**):
> **ident_far 0.998 / dist_far 1.000 / ident_near 1.000 / dist_near 1.000** (AUROC 1.000) → GATE
> PASSED; full 375/cell build proceeding. Judge = the plan-designated look-again instrument
> (single-frame yes/no logit read), independent of the probe axis; curation is on DATA
> construction only. v1 pilot kept at `data/mmred_natural/` (superseded).

## [2026-07-10h] ✅📊 A4 — mmred_natural LANDED (the d′ dial on natural images works): needle-diversity moves carrier d′ 6.2→4.3, distractor-similarity moves it 6.2→5.4 under identical needles (3 of 4 registered monotone edges ✓); and on this no-binding rung the MODEL RIDES THE LAW (law-pred 0.60–0.76 vs measured 0.57–0.66) — the MMRED-style wall largely disappears when counting needs no relational binding

> **Dataset:** `data/mmred_natural_v2/` — 4 cells × ~421 samples, N=8, counts 0–8, COCO val2017
> needles (dog) with judge-curated pools ([2026-07-10g]: gate 0.998–1.000), distractors near =
> cat-no-dog / far = no-animal, per-frame GT by construction, HERBench-style meta.
> **Runs:** caches+behavior jobs **120042/120046/120047/120048** (n=300/cell, L14/16, offsets
> 8–12, digit-argmax reader) → `outputs/ladder/natural/<cell>/<ts>/herbench_ac/`; per-cell d′
> (shrinkage-LDA held-out, 3 seeds) jobs **120056–58** → `outputs/ladder/natural/dprime_cells/
> 20260710_231039_off{9,10,11}/`. Probe gained meta-question support (`question`/head override).

| cell (diversity_similarity) | carrier d′_w @L16 (off−10) | model acc | law-pred 2Φ(d′_auc/2√8)−1* | dtc |
|---|---|---|---|---|
| ident_far | **6.21** (auc 4.83) | 0.657 | 0.761 | 0.983 |
| ident_near | 5.41 (auc 4.39) | 0.567 | 0.704 | 1.000 |
| dist_far | 4.30 (auc 3.73) | 0.580 | 0.595 | 0.949 |
| dist_near | 4.28 (auc 4.05) | 0.607 | 0.596 | 0.941 |
*prior-mixed, boundary-aware, from the run CSVs.

**Readings.**
1. **The dial works (registered A-3 mostly ✓):** diversity (ident→dist) lowers d′ by ~1.9/1.1
   (far/near rows); similarity (far→near) lowers it by 0.8 under identical needles but is FLAT
   under distinct needles (4.30≈4.28) — once instance variability dominates the evidence noise,
   distractor proximity adds nothing measurable. 3/4 edges monotone as registered; the 4th
   saturates rather than reverses.
2. **The natural rung behaves like CWE, not like MMRED:** model 0.57–0.66 ≈ law 0.60–0.76 (gap
   ≤0.11) vs image-MMRED's model 0.21 ≪ law 0.31. Counting "does a dog appear" needs no
   character→room binding — content addressing reaches the evidence, and the model spends most
   of the available d′. Behaviorally the cells order 0.657/0.607/0.580/0.567 — NOT the d′ order;
   two visible biases: identical-needle repeats slightly depress behavior vs their d′ (dedup
   bias — counting distinct dogs, not dog-frames: ident_near 0.567 < dist_near 0.607 despite
   +1.1 d′), and law-model gaps grow with d′ (readout saturation).
3. **Extraction is not the constraint anywhere on this rung:** dtc 0.94–1.00 in every cell.
4. Registered A-4 (adequacy passes while evidence binary-groundable) — E4 not yet computed on
   these caches; block-read/E4 pass queued as follow-up analysis.

**Caveats.** d′ at the single best offset (−10; −9/−11 within ±0.4 — see the off9/off11 CSVs);
digit-argmax behavioral reader (counts ≤8, adequate here); needle pool judge-curated (14% of
COCO dog images dropped — the rung measures unambiguous-evidence counting by design); one seed
pool, n=300/cell.

## [2026-07-10i] ✅📊 B2 — GATE CALIBRATION vs N (train@N=8, frozen; the dilution test): raw-message gate FN explodes 0.09→0.99 by N=128 exactly as registered (threshold drift; AUC stays 0.88–0.94 — the DIRECTION survives, the threshold breaks); mass-normalization cuts the drift ~10× (FN 0.11→0.26 ✓) but its FP floor then over-counts (+13 bias @N=128); FENCED messages dilute like raw (✗) — and the bias law bias≈N·FP−g·(FN+FP) matches measured bias everywhere

> **Run:** job **120059** (CPU) → `outputs/ladder/image_longN/gate_calibration/20260710_231635/`
> (report.txt, results.csv, fig_b2.png). Gate = logistic per-frame classifier on the room-token
> (off−9) carrier message, fit on 70% of the N=8 joint cache, DIRECTION+THRESHOLD frozen, applied
> to the joint caches at N∈{8holdout,16,32,64,128}; inputs (a) raw msg_f, (b) msg_f / attention
> mass m_f (the new `mass` cache field), (c) fenced caches (fenced-trained gate). New script
> `experiments/glstm/gate_calibration_vs_n.py`.

| arm @L16 | FN @N=8→128 | FP @N=8→128 | AUC @8→128 | tally bias @128 |
|---|---|---|---|---|
| raw | 0.09 → **0.99** | 0.15 → 0.00 | 0.94 → 0.88 | **−29.9** (all-miss) |
| mass-norm | 0.11 → **0.26** | 0.15 → 0.22 | 0.94 → 0.83 | **+13.1** (FP-driven) |
| fenced (to N=64) | 0.11 → 0.75 | 0.29 → 0.04 | 0.88 → 0.83 | −10.6 @64 |

**Registered verdicts (plan B):** raw FN inflates ✓ (1/N dilution pushes every message below the
frozen threshold; per-frame magnitudes shrink while the frozen intercept stays); mass-normalized
~flat **partial ✓** (drift cut ~10×, FN plateaus ~0.2 — magnitude dilution WAS the main driver —
but exact tallies then need FP ≲ 1/N, and the constant ~0.2 FP floor makes the Σ-gate over-count
at scale: bias +N·FP); fenced ~flat **✗** (fencing does not stop softmax renormalization over
N·196 visual tokens — fenced messages dilute like joint ones); bias identity ✓ (pred == measured
throughout, i.e., per-frame errors aggregate independently — no error correlation rescue).

**Design consequence for the tally register:** an N-robust deployed gate needs BOTH
mass-normalization (threshold stability) AND an N-scaled decision margin / FP control
(e.g., threshold on the mass-normalized margin calibrated per-N, or a top-g selection rather
than a fixed threshold). Neither raw Σσ nor fenced reps survive N-scaling as-is.

**Caveats.** Single offset (room −9); gate = plain logistic (no recalibration allowed by design
— that IS the test); fenced arm missing N=128 (job 120017 still running at analysis time; the
0.11→0.75 trend is already decisive); AUC decline (0.94→0.83-0.88) shows dilution also costs
some separability, consistent with joint d′ 1.97→1.59 ([dprime_vs_n run]).

## [2026-07-11] ✅📊 B1 — d′ vs N COMPLETE (images @392px, N=8→128, three arms): joint carrier d′ is FLAT ≈2.0 out to N=64 (1.97/2.12/1.98/1.93, dip 1.59 @128) — never near the 6.3 crush line; multipass is N-INVARIANT at 7.2–8.1 (≥ crush line at every N); fenced sits consistently BELOW joint (1.6–1.8) — the fence is not a lever at any N. The model's collapse (0.207→0.013) therefore is NOT d′ collapse: supply stays ~2 while behavior dies — dilution + the range clamp sit between carrier and answer

> **Runs:** caches jobs **120013–120022** (+ wave-1 survivor 120004; wave 1 killed by quota,
> [PROGRESS incident note]) → `outputs/ladder/image_longN/{joint,fenced,multipass}/N{8..128}/`;
> n=300/300/300/200/150 per N; layers 14/16, offsets 9/13, mass field included. Analysis job
> **120065** (shrinkage-LDA held-out, 3 sample-disjoint seeds, room token off−9) →
> `outputs/ladder/image_longN/dprime_vs_n/20260711_003548/` (report.txt, results.csv,
> **fig_b1.png** — the pre-registered Fig B1). Multipass = isolated per-frame forwards
> (`text_multipass_cache.py --modality image`).

| arm @L16 (d′_w) | N=8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| joint | 1.97 | 2.12 | 1.98 | 1.93 | 1.59 |
| fenced | 1.80 | 1.84 | 1.79 | 1.67 | 1.62 |
| multipass | 7.18 | 7.91 | 8.08 | 7.74 | 7.82 (auc-ceiling 5.26 ∀N) |
| model (joint) / mp-sum | 0.207/0.910 | 0.127/0.793 | 0.053/0.680 | 0.040/0.580 | 0.013/0.420 |
| law-pred(joint d′) | 0.314 | 0.248 | 0.175 | 0.130 | 0.084 |

**Registered-prediction verdicts (plan B):**
1. "Joint d′ stays ≪ 6.3" — **✓** (max 2.12; at N=128 the per-frame supply is 4× short of what
   ~90% exact-match needs).
2. "Fenced/multipass approach the crush line" — **✓ multipass / ✗ fenced.** Multipass clears 6.3
   at EVERY N with no N-trend (per-frame writes are N-independent when frames are processed
   alone — the supply degradation is entirely a joint-context effect, replicating [2026-07-10b]
   on images at scale). Fenced is a null-to-negative lever at all N (extends the N=8 fence
   refutation).
3. "Model collapses as the law dictates" — **partial**: collapse ✓ and monotone, but the law
   over-predicts increasingly with N (0.084 vs 0.013 @128) — because d′ does NOT dilute as
   1/√N-of-supply (it's flat); the failure concentrates downstream: the [2026-07-10f] emitted
   range clamp + the B2 threshold drift. The "d′ dilution" picture is REVISED: joint attention
   costs a fixed ~4–6× d′ factor vs isolated processing at every N, and N-scaling failures ride
   on readout/decision calibration, not on further supply loss.
4. mp-sum behavior tracks per-frame error accumulation (0.985 flat perception → 0.91→0.42),
   staying 4–30× above the joint model at every N.

**Caveats.** Single offset (room −9; off−13 in results.csv), L14 lower everywhere (0.9–1.1 @128);
multipass d′_w is above the AUC estimator ceiling (perfect separation; quote "≥5.3, d′_w≈7–8");
E4 not evaluated per-N here (law rows indicative); n shrinks with N (150 @128); fenced N=128 ran
in the same job as joint (shared sample draw).

## [2026-07-11b] ✅📊 C2 + C3 — THE TOKEN INTERFACE IS NECESSARY, NOT JUST SUFFICIENT: every learned embedding-level injection route verbalizes trained counts (0.81–1.00) and extrapolates to ZERO (0.000–0.098 held-out), while real digit tokens in the same slot score 1.000 everywhere with zero parameters — including a continuous Fourier basis and one ANCHORED to the model's own digit-embedding geometry (both 1.000 in-range / 0.000 held-out)

> **Runs:** jobs **120027** (digit routes; crashed on an OOD-interp bug after digit arms
> completed) / **120040** (digit_fact rerun, 5 epochs) / **120060** (count/fourier/fourierE/token)
> → `outputs/readout/c2_digit_codebook/{20260710_220702,225300,233141}/` (results.json,
> recs_*.json, report.txt). New: `experiments/readout/c2_digit_codebook.py`. Setup: text-MMRED
> N=40 context, fact-slot injection ("Note: C spent exactly ⟨K⟩ steps in the R." — the C1b-validated
> predicate-matched pre-question site), embedding-level soft tokens, frozen 7B, CE on the answer
> digits, train counts {0–9,12,25,30}, held-out = other two-digit ≤40, NO trained readout head.

| route | trained params | in-range | held-out two-digit |
|---|---|---|---|
| **token_fact** (real digit tokens = C1 bar) | **0** | 1.000 | **1.000** |
| digit_fact (per-digit codebook, compositional in principle) | 35,840 | 0.808 | 0.098 |
| count_fact (C-CONTROL per-count + charitable interpolation) | 46,592 | 0.808 | **0.000** |
| fourier_fact (continuous basis φ(K), extrapolates mathematically) | 28,672 | 1.000 | **0.000** |
| fourierE_fact (φ(K)→W anchored to E(digits) geometry) | 28,672 | 1.000 | **0.000** |

**Registered-prediction verdicts (plan C):**
1. C-control fails held-out (~0): **✓ exactly** (0.000, even with interpolated vectors).
2. C2 digit-compositional extrapolates (the bet): **✗** — 0.098, and its nonzero cells are only
   counts adjacent to the trained envelope (24/28/29); cos(v_d, E(d)) ≈ 0–0.11: SGD finds
   arbitrary memorized symbols, never the digit tokens, and single digits SHARED with trained
   two-digit counts degrade (2:0.00, 9:0.00 in-range) — direct evidence of compositional tension,
   not composition.
3. **The pre-registered alternative branch LANDS as the thesis claim:** since token_fact = 1.000
   OOD under identical conditions, the failure is not the site, phrasing, or range — **the frozen
   readout can verbalize activation-level quantities only through the token embedding lookup;
   learned vectors are read as opaque trained symbols, not decoded from any geometry** — even a
   basis anchored to the model's own number geometry (fourierE) is not decoded for unseen values.
   Token interface: sufficient AND necessary.
4. C3 (native/continuous geometry, secondary): **✗ resolved negative** at embedding level —
   fourier/fourierE are exactly the "inject count as a point in a continuous basis" test.

**Caveats.** No trained readout head anywhere (the plan allowed LoRA-r4/unembedding; adding one
could only blur the token-necessity contrast — the clean negative stands without it and the
"too-good" gate never triggered); embedding-level injection only (residual-level per-count was
already ruled out pre-plan; residual-level compositional injection remains untested); train set
13 count values × 60 reps × 5 epochs, loss plateaued ~0.15 (converged in-range — 0.81–1.00);
single seed; eval 4 samples/count (52 in-range, 112 held-out per route).

## [2026-07-11c] ✅📊 A3 — MLVU Action-Count PORTED (206 Qs, frames-only, `data/mlvu_ac/`): at the standard 32-frame budget the benchmark is EVIDENCE-DELIVERY-limited, not counting-limited — the judge finds 0 visible evidence frames for 35% of questions and only ~0.37 frames per gold instance, and the model's mean answer (1.02) is roughly calibrated to what it sees; 4× denser sampling (N=128 @392px) causally lifts mean-pred to 2.45 (gold 2.93), MCQ 0.282→0.393

> **Port:** ungated mirror `sy1998/MLVU` (the official repo is HF-gated); 206 count questions,
> 206 videos (~26 GB transient, batched ≤7 GB with a quota guard, all deleted after prep) →
> 128 uniform 448px frames/question + meta (`experiments/mlvu/prep_ac_frames.py`, jobs
> 120037/120049-OOM→120051/120068; OOM fixed by a packbits/chunked hamming matrix). **The released
> jsons carry NO insertion timestamps** — the plan's "exact insertion GT" assumption fails;
> duplicate-detection recovers GT for only 7/206 (insertions are DIFFERENT clips of the same
> Kinetics action, not repeats). Per-frame labels therefore come from the look-again judge
> (`lookagain_ac.py`, job 120070 → `lookagain_N32.json` per question).
> **Behavior:** `eval_ac_behavior.py` — N=32: job 120069 → `outputs/ladder/mlvu_ac/behavior_N32/`;
> N=128: job 120071 → `outputs/ladder/mlvu_ac/behavior_N128/`.

| protocol | N=32 | N=128 |
|---|---|---|
| MCQ (4-choice; chance 0.25) | 0.282 | **0.393** |
| open integer (exact) | 0.112 | 0.175 |
| open MAE / mean-pred (gold 2.93) | 1.91 / 1.02 | **1.24 / 2.45** |

**Readings.**
1. **The rung's difficulty is dominated by sampling coverage:** videos average 782 s (max 2.2 h)
   and insertions are short; 32 uniform frames deliver ~0.37 evidence frames per gold instance
   (judge-measured; corr(visible, gold)=0.48), and 35% of questions show ZERO evidence. The
   model's near-constant answer "1" at N=32 is approximately optimal for its observation.
   Quadrupling frames moves every metric in lockstep — behavior tracks delivered evidence, not
   count difficulty. **For the thesis ladder: MLVU-AC sits BELOW HERBench in per-frame evidence
   availability; it measures the sampling+perception pipeline, not aggregation, at standard
   budgets.** (This answers the peer "benchmark ladder" request with a measured caveat.)
2. gold=5 stays ≈0 in both protocols and budgets — the emitted-range clamp (mean-pred ≤2.5)
   compounds with coverage.
3. Instruments beyond behavior: judge per-frame labels are in place for a d′/parity pass on
   sampled-frame caches (judge-noise caveat applies); dup-detect GT exists for 7 questions
   (exact); both left as follow-up analysis — behavioral + coverage results above are complete.

**Caveats.** Judge labels are model-derived (same 7B family as the subject — shared blind spots
possible; HERBench precedent applies); MCQ letter-reader; N=128 open accuracy is still
range-clamped; mirror provenance (sy1998/MLVU) rather than the gated official repo — spot-check
against the official release before thesis citation.

## [2026-07-11d] 📊 RECONCILIATION + REGISTRATION — the fence-layout split is now understood (fence helps QUESTION-FIRST frame reps; at the DEPLOYED frames-first carrier it is null-to-negative), and MASS COMPETITION in the shared softmax is registered as the candidate mechanism for the joint-context tax; the tax constant is family-general (Qwen 3.2–3.6×, InternVL 3.4×)

> Bookkeeping entry (no new runs — every number below traces to an existing run dir).

**1 · Fence-layout reconciliation.** Two apparently contradictory fence results coexist in this
log and are now explicitly re-scoped:
- **Question-first layout: fence WORKS.** [2026-06-30] block-diagonal isolation mask (each frame
  attends only to itself + the question) lifts per-frame extraction 0.942→0.995 and sum-count
  0.657→0.952 in one forward — the "3.1→5.2" d′ lever cited in the 2026-07-08 plan is THIS
  (question-first per-frame reps).
- **Deployed frames-first carrier: fence HURTS.** `count_msgcache_fenced` (job 118557): best
  per-frame message AUROC **0.920 < joint 0.956** (matched-spec logistic L14: 0.889 < 0.905),
  carrier d′ 2.47→1.93, model 0.215→0.120; text fence NULL ([2026-07-08c]); and B1 fenced ≤
  joint at EVERY N ([2026-07-11]: 1.80/1.84/1.79/1.67/1.62 vs joint 1.97/2.12/1.98/1.93/1.59).
- **Re-scoped claim:** frame isolation by attention masking repairs the frames→frame-rep path
  (question-first reps read the question; cutting cross-frame edges removes contamination), but
  the deployed frames-first *carrier write* is not limited by cross-frame ATTENTION EDGES —
  cutting them only removes useful context. The joint→multipass gap survives fencing entirely.

**2 · Registered hypothesis (frozen before the P1 mechanism probes): MASS COMPETITION.** The
joint-context tax — carrier d′ ~2 in any joint pass vs 7–8 when frames are processed alone,
N-invariant on both sides ([2026-07-11] Fig B1) — is caused by the carrier's softmax attention
being SHARED across all N frames' tokens: each frame's message enters with attention mass ~V/N
instead of ~V, and the induced magnitude/mixing noise (competition varying per frame with chunk
content) costs the fixed ~3–4× d′ factor. Predictions for the P1 probes (registered now):
- (e) chunk-size sweep: d′ decreases monotonically in chunk size k with the sharpest drop from
  k=1 to small k (competition onset), saturating toward the joint value — anchors k=1: 8.08,
  k=32: 1.98 at N=32 (both measured).
- (f) within-frame attention renormalization during a joint pass (equal per-frame mass at the
  carrier hop, L14–17): if mass competition is the tax, patched d′ recovers MOST of the
  joint→multipass gap; if patched ≈ joint, the tax is upstream in-context frame encoding and
  the hypothesis is REFUTED for the carrier hop.
- (g, conditional on f-recovery) a frame's message shifts with its chunk-mates' salience through
  the shared denominator.

**3 · Cross-family tax constant (spot-check satisfied from existing anchors — no new runs).**
Qwen image: joint 1.97 → multipass 7.18 @L16 (**3.6×**, [2026-07-11]); Qwen text: joint block
2.45 → multipass 7.9 (**3.2×**, [2026-07-10b]); InternVL2.5-8B: joint 1.9 → multipass 6.4–6.6
(**3.4×**, [2026-07-07c], `internvl/multipass_bench/`). Three pipelines, two model families,
two modalities: the joint-context tax is a ~3.2–3.6× d′ factor everywhere measured.

## [2026-07-11e] ✅📊 P0b — E4 ADEQUACY SWEEP: registered A-4 lands PARTIAL with a sharper reading — adequacy passes exactly where the evidence class is DIVERSE (natural distinct-needle cells: kurt +0.5–0.8 PASS) and fails via kurtosis where evidence is degenerate-identical (ident cells +1.6…+5.4) or where joint context is long (long-N joint caches: kurt +0.5 @N=8 → +25 @N=128); binary-groundability alone does NOT buy adequacy

> **Runs:** CPU block-read/E4 jobs **120130–120138** →
> `outputs/ladder/natural/block_read/20260711_125059_<cell>/` and
> `outputs/ladder/image_longN/block_read/20260711_125059_N{8,16,32,64,128}/` (report.txt,
> results.csv each). Estimator identical to all prior block reads (score-concat matched filter,
> 3 sample-disjoint seeds).

**1 · Natural cells (scores registered A-4 "adequacy passes while evidence stays
binary-groundable"): PARTIAL.**
| cell | E4 @L14 / @L16 (kurtE) | block d′ @L14 | best single |
|---|---|---|---|
| dist_far | **PASS / PASS** (+0.49 / +0.78) | 6.87 | 5.24 |
| dist_near | FAIL / **PASS** (+2.09 / +0.77) | 7.20 | 5.53 |
| ident_far | FAIL / FAIL (+5.43 / +1.56) | 8.12 | 5.97 |
| ident_near | FAIL / FAIL (+4.06 / +2.32) | 8.50 | 6.09 |
All four cells are binary-groundable by construction, yet only the DISTINCT-needle cells pass —
identical repeated needles make the evidence-message distribution near-degenerate (point-mass ×
chunk-context mixture → heavy tails). **A-4 revised:** adequacy tracks evidence-class diversity.
The law is licensed on the naturalistic cells (which are also the thesis-relevant ones); the
ident cells carry the d′-dial anchor but get the caveat. (Block reads also lift the natural
carrier ~+1–2 d′ over the best single offset — the multi-token concept region is a distributed
carrier; auc-ceiling caveat applies at these magnitudes.)

**2 · Long-N joint caches (qualifies B1's law rows): E4 fails progressively with N.**
kurtE @L14/L16: N=8 **+0.52 PASS**/+1.61 → N=16 +2.9/+1.8 → N=32 +5.2/+3.0 → N=64 +16.8/+7.9 →
N=128 **+25.2/+19.2**; std-ratio drifts 0.97→1.4–2.0. The joint-context tax is not only a mean
d′ factor — long joint contexts make per-frame message distributions heavy-tailed (magnitude
depends on chunk-mates → mixture noise), exactly the signature the registered mass-competition
mechanism predicts ([2026-07-11d] §2). Consequence: B1's law-pred column is licensed at N=8
and indicative-only beyond; the [2026-07-11] B1 caveat is now quantitative.

## [2026-07-11f] ✅📊 P4k — DEDUP SEMANTICS on mmred_natural: the registered dissociation CONFIRMED — with identical repeated needles, "in how many FRAMES does a dog appear?" is answered with the frame count (mean 3.90 vs gold 4.17) while "how many DOGS are there in total?" is answered with the SUPPORT SIZE (mean 1.13–1.37 vs unique 0.9; exact-vs-unique 0.707/0.827); distinct-needle cells show no dissociation — the frozen model selects the aggregation operator (Σ vs distinct-count) from question wording

> **Run:** job **120147** → `outputs/ladder/natural/dedup_semantics/<ts>/` (report.txt,
> summary.json, rows.json). n=150/cell/question, mmred_natural_v2, generation reader,
> `experiments/natural/dedup_semantics.py`.

| cell | question | acc vs frame-gold | acc vs unique | mean pred (frame-gold / unique) |
|---|---|---|---|---|
| ident_far | frames | **0.687** | 0.220 | 3.90 (4.17 / 0.91) |
| ident_far | count | 0.260 | **0.707** | 1.37 |
| ident_near | frames | **0.567** | 0.367 | 3.31 (4.09 / 0.89) |
| ident_near | count | 0.280 | **0.827** | 1.13 |
| dist_far | frames / count | 0.613 / 0.487 | (= frame-gold) | 3.93 / 3.38 (3.86) |
| dist_near | frames / count | 0.553 / 0.260 | (= frame-gold) | 3.84 / 3.17 (4.21) |

**Readings.** (1) **Registered prediction ✓** — dissociation exactly on ident cells, agreement on
dist cells. The model implicitly DEDUPLICATES under entity-count wording: same visual stream, two
different aggregation operators selected by the question (frame-count Σ vs support-size distinct
count). (2) This retro-explains the [2026-07-10h] ident-cell behavioral dip (the "dedup bias" —
part of the frames-question probability mass leaks to the entity reading) and ties directly to
the rooms_visited support-size semantics: distinct-counting is a NATIVE operator, available
in-context. (3) The clean 0.827 exact at ident_near/count is also a positive control that the
model can count near-perfectly when the effective target is small — consistent with the range
clamp (support size ≤1 here).

**Caveats.** "How many dogs are there in total?" is ill-posed on dist cells (a COCO frame may
contain several dogs; unique-image count ≠ dog-instance count) — the dist count-question rows
are reported for symmetry but their "gold" is approximate; the ident-cell dissociation (where
one image = one dog instance repeated) carries the claim. Single model, n=150/cell.

## [2026-07-11g] ✅📊 P1f — MECHANISM VERDICT: the registered MASS-COMPETITION hypothesis is REFUTED at the carrier hop — renormalizing every frame to an EQUAL share of the carrier's attention mass during a joint pass (forward patch, L14–17) recovers essentially nothing of the joint→multipass gap (N=8: renorm d′ 2.03 vs joint 1.97 vs multipass 7.18; N=32: 2.44 vs 1.98 vs 8.08) — the joint-context tax is UPSTREAM, in the in-context encoding of the frames themselves

> **Runs:** new `experiments/glstm/attn_renorm_patch.py` — SDPA forward patch: question-token
> rows' attention recomputed with per-frame mass equalization (within-frame relative weights and
> text columns untouched; self-check |Δ question-row output| = 0.72) in layers 14–17; carrier
> messages read WITH the renormalized weights. Jobs **120145/120149** (N=8, n=300) and
> **120146/120154** (N=32, n=200) → `outputs/ladder/image_longN/renorm/{N8,N32}/` +
> `renorm/dprime_N{8,32}_*/`. Anchors: B1 joint/multipass caches (same data, same seed).

| N | joint d′ @L16 | renorm d′ @L16 | multipass d′ @L16 | gap recovered |
|---|---|---|---|---|
| 8 | 1.97 | 2.03 | 7.18 | **~1%** |
| 32 | 1.98 | 2.44 | 8.08 | **~8%** |

**Registered branches ([2026-07-11d] §2):** "patched d′ recovers most of the gap → mass
competition" vs "patched ≈ joint → tax is upstream in-context frame encoding." **The second
branch lands.** Equal-mass delivery at the carrier hop (and through 4 patched layers feeding
it) leaves d′ within ~0.5 of joint. The per-frame information that the carrier COULD collect is
already degraded by the time frames are encoded in a shared context — consistent with the fence
result (cutting cross-frame attention edges doesn't help either) and with B2/E4 (heavy-tailed
message distributions at long N: the damage is in the message CONTENT, not its mass).
Behavioral note: the patch leaves the model's answers roughly intact at N=8 (0.173 vs 0.207)
and unchanged-bad at N=32 (0.020) — renormalized streams are mildly off-distribution downstream.
**P1g (chunk-mate composition probe) SKIPPED per its pre-registered condition** (no recovery).
What remains as candidate mechanisms for the upstream tax: shared-context encoding interference
in the visual/early-LM layers (not attention-edge-carried — possibly normalization/gain effects
or position-conditioned encoding), to be constrained by the P1e chunk-size curve (in flight).

**Caveats.** Patch covers L14–17 question rows only (frame-token rows unpatched — a full-row
renorm was not tested; the carrier's supply is what the hypothesis addressed); N=32 renorm shows
a real but small +0.46 lift, so mass competition is a MINOR contributor, not zero; single seed,
n=200–300.

## [2026-07-11h] ✅📊 P0c — MLVU-AC d′/parity on judge-labeled frames: the carrier is the ACTION-DESCRIPTION token region (peak off−16, d′_w 2.60 @L14; block 2.80, d′_auc 3.24), placing MLVU-AC-at-N=32 between image-MMRED (~2.0–2.5) and mmred_natural (4.3+) on the ladder; E4 fails via a 6.6–7.9× STD-RATIO — the judge-label-noise signature; and the model's answer tracks the judge-visible count at 0.583

> **Runs:** derived dataset `data/mlvu_ac_n32judge/` (206 questions × 32 uniform frames,
> is_evidence = look-again judge P(yes)>0.5, question normalized to a fixed template so carrier
> offsets align); cache job **120139** → `outputs/ladder/mlvu_ac/msgcache_n32judge/20260711_125240/`;
> block-read job **120150** → `outputs/ladder/mlvu_ac/block_read/20260711_130831/`.

Single-locus d′_w @L14: off−16 (last action token) 2.60 > off−15 2.23 > off−14 2.20 > off−0 1.91;
block 2.80±0.17. Ladder vs judge-count gold: model 0.583, ridge-on-sum 0.518, dtc(block) 0.610 —
the frozen model is essentially AT its achievable ceiling for the evidence actually delivered
(the failure vs true gold is sampling, [2026-07-11c]). The carrier location (needle-description
tokens in the question) matches the CWE/content-addressing pattern, not MMRED's relational-carrier
pattern. **Caveats:** labels are judge-derived (noise attenuates d′ — quote as lower bound; the
std-ratio E4 failure is exactly what label noise predicts); evidence base rate ~3.4% (nE≈220);
"model acc 0.583" is vs the judge-visible count, not the 1–5 insertion gold.

## [2026-07-11i] ✅📊 P0d — LAW + CLAMP composed closure (zero fitted parameters): rank-remapping the d′-limited latent estimate through the MEASURED emission marginal reproduces the B3 behavioral curve almost exactly where the plain law is 2–5× high — and the composition transfers directionally to both real-video rungs

> **Script:** `experiments/glstm/law_clamp_closure.py` (composition = latent count estimate
> ~ N(gold, N/d′²) pushed through F_emitted∘F_latent⁻¹ — a monotone rank remap onto the measured
> answer marginal; predicts the PAIRING, the marginal is measured). **Run:**
> `outputs/ladder/image_longN/law_clamp/<ts>/` (results.csv, report.txt, **fig_b3_clamp.png**).
> Inputs: B1 joint d′ @L16 per N, B3/HERBench/MLVU behavioral rows.

| cell | measured | plain law | **law+clamp** |
|---|---|---|---|
| MMRED N=8 | 0.173 | 0.350 | **0.186** |
| MMRED N=16 | 0.127 | 0.291 | **0.106** |
| MMRED N=32 | 0.073 | 0.188 | **0.069** |
| MMRED N=64 | 0.053 | 0.147 | **0.054** |
| MMRED N=128 | 0.020 | 0.100 | **0.039** |
| HERBench armB (N=16, d′ 1.04) | 0.172 | 0.103 | 0.137 |
| MLVU-AC N=32 (d′ 2.80, visible-count latent) | 0.112 | 0.124 | 0.140 |

**Readings.** (1) On MMRED the composition is a near-exact ZERO-parameter account (mean |err|
0.013 vs the plain law's 0.13): flat supply + measured emission clamp + ordinal-preserving squash
= the entire N-collapse. The two mechanisms measured separately ([2026-07-11] d′ flat;
[2026-07-10f] clamp) COMPOSE. (2) Transfer is directional, not exact: HERBench moves from 40%
under-prediction to 20% under (graded-evidence adequacy failure documented there); MLVU lands
within 0.03 when the latent is centered on the judge-VISIBLE count (the delivered evidence).
One quantitative backbone — d′ supply → rank-preserving clamped emission — now spans synthetic
and natural rungs, with honest residuals where E4 already flags inadequacy.

**Caveats.** The emitted marginal is measured from the same behavioral runs whose exact-match is
predicted — the composition predicts the gold-pairing given the marginal (non-circular in the
pairing, circular in the support; stated in the figure caption); MLVU cell uses judge-visible
latents (adds judge noise); HERBench golds are 1–5 so the clamp barely binds there.

## [2026-07-11j] ✅📊 P1e — CHUNK-SIZE SWEEP (N=32 @392px, k frames per forward): the joint-context tax is paid at the FIRST companion frame — d′ 8.08 (k=1) → 3.37 (k=2) → 2.54 (k=4) → 2.31 (k=8) → 2.15 (k=16) → 1.98 (k=32); the registered "sharp drop at small k, saturating toward joint" shape lands EXACTLY, and with the renorm null it pins the tax to in-context encoding interference that switches on as soon as a frame is not alone

> **Runs:** `--chunk-k` mode added to `experiments/glstm/text_multipass_cache.py` (image chunked
> passes; per-chunk digit answers give a chunked-tally behavioral read for free); job **120143**
> (k∈{2,4,8,16}, n=200 each; smoke 120140) → `outputs/ladder/image_longN/chunk_sweep/k{2,4,8,16}/`;
> anchors k=1 (multipass N32) and k=32 (joint N32) from B1. d′ analysis job **120166** →
> `outputs/ladder/image_longN/chunk_sweep/dprime_20260711_134108/` (report.txt, results.csv,
> fig_b1.png = the d′-vs-k curve).

| k (frames/forward) | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| carrier d′_w @L16 | 8.08 | **3.37** | 2.54 | 2.31 | 2.15 | 1.98 |
| chunked-tally behavior | 0.680 | 0.140 | 0.105 | 0.135 | 0.140 | 0.053 |
| per-chunk count exact | 0.985* | 0.833 | 0.57 | ~0.35 | 0.223 | — |
*(k=1 = per-frame binary perception.)

**Readings.**
1. **58% of the total log-d′ drop happens between k=1 and k=2.** One companion frame in the
   forward costs more than the next thirty combined. The tax is an ONSET effect of shared
   processing, not a gradual per-frame crowding accumulation.
2. **Mechanism triangulation (with [2026-07-11g]):** not carrier-mass competition (renorm null),
   not cross-frame attention edges (fence null/negative), onset at the first companion —
   the per-frame representations are computed differently the moment context exists
   (candidates: visual-encoder/early-LM contextual normalization, cross-frame feature binding
   during encoding, position-conditioned encoding). This is the sharpest constraint the campaign
   has on WHERE the joint-context tax lives.
3. **Pipeline consequence (explains the P2h result):** k=8 chunks retain only d′ 2.31 ≈ joint —
   chunking with k≥4 buys almost nothing; the accuracy-per-forward frontier is at k=1–2
   (k=2: d′ 3.37 at half the forwards of multipass). The e2e chunked pipeline's N=64 collapse
   ([P2h, this session]) is now explained: its k=8 messages were barely better than joint.
4. Chunked-tally behavior is non-monotone in k (0.14 at k=2 vs 0.105 at k=4): per-chunk answers
   at k=2 are near-perfect (0.833) but 16 chunk-answers accumulate sum errors; at k=16 fewer,
   noisier chunks — two error sources trade off. The gate-based pipeline (not digit answers)
   remains the right reader.

**Caveats.** Single N (32), steps task, n=200/k, single seed; k=1/k=32 anchors are n=300 from B1
(same data/seed pool); per-chunk "exact" for k=16 read from the running log at n≈200.

## [2026-07-11k] ✅📊 P3 — VNBench COUNTING PORTED (450 videos, exact needle-time GT for edit1/insert1): registered scoring — (1) E4 fails via VARIANCE-RATIO ✓ exactly as predicted (std-ratio 3.2–3.5); (2) "high d′, model between CWE and MMRED" ✗ — block d′ is only 2.50 (MMRED-level; no content-addressing escape); (3) "carrier = needle-description tokens" ✗ — the carrier is FLAT across question offsets (no needle-token peak); behaviorally the emission clamp reappears (open 0.193→0.229 at N=32→64, mean-pred 3.4–4.6 vs gold 5.66, zero exact above ~8)

> **Port:** official Google-Drive archive (10.9 GB zip, videos deleted post-extraction; jsons from
> `videoniah/VNBench`) → `data/vnbench_cnt/` (450 questions × 128 uniform 448px frames, 2.0 GB
> durable; `experiments/vnbench/prep_cnt_frames.py`). **GT structure finding:** `needle_time` =
> one start-time per occurrence; gold == len(needle_time) for ALL 300 edit1/insert1 questions
> (exact per-frame GT with a 1.5 s window) — but cnt_edit2's gold counts OBJECTS WITHIN needles
> (gold ≠ #events for 149/150; metas annotated `needle_time_events_only`). Instruments use the
> exact-GT 300 (`data/vnbench_cnt_n32exact/`).
> **Runs:** behavior jobs **120158** (N=32) / **120159** (N=64) →
> `outputs/ladder/vnbench/behavior_N{32,64}/`; cache **120160** + block-read **120170** →
> `outputs/ladder/vnbench/{msgcache_n32exact/<ts>,block_read/20260711_135010}/`.

| instrument | value |
|---|---|
| behavior MCQ (chance 0.25) | 0.369 @N=32 → 0.416 @N=64 |
| behavior open | 0.193 → 0.229 (MAE ~3.0; mean-pred 3.44→4.57 vs gold 5.66; ~0 above gold 8) |
| carrier d′ (N=32 frames, exact labels) | best single 2.02 (off−12), **block 2.50±0.05 @L14** |
| carrier profile | FLAT: off−12/−16/−0 ≈ 2.0 each — no needle-token peak |
| E4 | **FAIL via std-ratio 3.21/3.46** (skew +0.35/+0.43, kurt +1.3/+0.4) |
| ladder (vs visible-count gold) | model 0.250, ridge 0.144, law-pred 0.179, dtc 0.192 |

**Registered verdicts:** (1) ✓ exact — the synthetic-needle variance-ratio failure mode predicted
from CWE's precedent; (2) ✗ — d′ 2.50 sits at MMRED level, NOT the CWE regime; the anticipated
content-addressable escape did not materialize (candidate cause: VNBench needles are small
edited patches / 1-frame inserts — at 392px×32-frame sampling the per-frame evidence is
perception-thin, unlike CWE's whole-token match); (3) ✗ — carrier distributed/flat across the
question span. Net ladder placement: VNBench-cnt lands NEXT TO image-MMRED (d′ ~2–2.5, model at
law), not next to CWE — synthetic needles do not by themselves buy content addressing.

**Caveats.** d′ from N=32 subsampled frames (needle visibility ~1–2 frames per occurrence at
this rate — delivery-limited like MLVU, quantified by visible_count in the metas); 392px resize
may under-render small needles (not swept); edit2 excluded from instruments (compound gold);
judge-free exact GT is the port's strength — numbers are lower bounds only via sampling, not
label noise.

## [2026-07-11l] ✅📊 P2 — END-TO-END TALLY PIPELINE (frozen model as its own extractor, verifier, and verbalizer): retrieve-then-verify recovers exact-match 0.880 / 0.807 / 0.707 at N=32/64/128 — 17–35× the frozen model (0.053/0.053/0.020), above the full multipass solution (0.680/0.580/0.420), and above the law ceiling for any linear read of the joint carrier — while the cheap chunked variant beats the frozen model 2.6–3.5× at N≤32 for 2–5 forwards but dies at N≥64 exactly as the chunk-size curve predicts

> **Script:** `experiments/pipeline/e2e_tally.py` — every stage is a measured campaign result
> deployed: (1) per-frame carrier margins from a mass-normalized logistic gate trained ONCE on
> the N=8 joint cache ([2026-07-10i]); (2) per-N calibrated threshold (30 labeled calib samples,
> N known at inference); (3) tally; (4) the tally rendered as the predicate-matched fact sentence
> pre-question ([2026-07-10d]) and verbalized by the frozen model. `--mode retrieve` (P2i): ONE
> joint pass → high-recall shortlist (FN≤2% calibrated) → isolated yes/no look-again passes on
> the shortlist → tally = #yes. **Runs:** chunked (k=8) job **120151** →
> `outputs/pipeline/e2e_tally_chunked/`; retrieve jobs **120162/120163/120164** →
> `outputs/pipeline/e2e_retrieve_N{32,64,128}/`. Eval seed 2 (disjoint from cache/behavior
> draws), n=150/N, calib disjoint from eval.

| N | frozen (B3) | law ceiling (joint d′) | mp-sum (N fwd) | chunked k=8 (fwd) | **retrieve (fwd)** |
|---|---|---|---|---|---|
| 8 | 0.173 | 0.314 | 0.910 (8) | **0.600** (2) | — |
| 16 | 0.127 | 0.248 | 0.793 (16) | 0.280 (3) | — |
| 32 | 0.073 | 0.175 | 0.680 (32) | 0.193 (5) | **0.880** (26.6) |
| 64 | 0.053 | 0.130 | 0.580 (64) | 0.033 (9) | **0.807** (56.7) |
| 128 | 0.020 | 0.084 | 0.420 (128) | 0.027 (17) | **0.707** (113) |

**Readings.**
1. **Retrieve-then-verify is the headline constructive result:** a fully frozen system (gate =
   one logistic on N=8 messages; everything else is the model itself) sustains 0.71–0.88 exact
   to N=128, above every baseline including full multipass — the isolated yes/no verifier
   (per-frame perception 0.985) is a cleaner reader than per-frame digit answers, and errors
   that survive it are nearly unbiased (MAE 0.15/0.23/0.55). It also sits ABOVE the closed-form
   ceiling for any linear readout of the summed joint carrier — by construction: it never sums
   noisy messages; it decides per frame first (the thesis's gate-before-sum principle, deployed).
2. **Cost verdict (registered P2i "cost scales with evidence count"): ✗ honest miss** — the
   shortlist calibrated to FN≤2% on DILUTED joint margins keeps ~80–90% of frames (B2's
   threshold-drift again), so cost ≈ N+1 forwards, not O(evidence). Accuracy goal ✓✓, cost goal
   ✗. (Obvious v2: shortlist from k=2 chunk margins — d′ 3.37 — instead of the joint pass;
   untested, noted as future work.)
3. **Chunked k=8 (registered P2h): ✓ at N≤32, ✗ beyond** — 0.600 at N=8 for 2 forwards is the
   budget option (3.5× frozen), but k=8 chunks retain only d′ 2.31 ≈ joint ([2026-07-11j]), so
   the gate has nothing to work with at N≥64 and MAE-calibration collapses the tally. The
   chunk-size curve, the gate calibration, and the pipeline results now form one consistent
   quantitative picture.

**Caveats.** Per-N threshold calibration uses 30 labeled samples/N (legitimate at inference —
N is known — but it is supervision the frozen baseline lacks; the gate itself is trained on the
pre-existing N=8 cache only); render stage is text-only (no frames at answer time; C1 cf-arms
showed the fact wins even against conflicting visuals); steps task only; n=150/N, single seed.

## [2026-07-11m] ✅📊 P4j (partial) — NATIVE READING AXIS vs N: the axis is STABLE, not rotating — |cos(axis_N=16, axis_N=8)| = 0.82–0.86 at L14/L16 (gradient axes, coherence 0.64–0.69), with d′ along the native axis flat at 0.55–0.77 across N — the emission clamp is "same axis, saturating magnitude," not context-driven axis rotation

> **Runs:** `native_axis_probe.py` (+ new `--resize`) @392px, offset −9, n=100 (N=8) / 55 (N=16)
> — job **120144** → `outputs/ladder/image_longN/native_axis/N{8,16}/20260711_130100/`; axis
> comparison computed from the saved `native_axes.pt` files against the joint caches (console,
> this session). Global gradient-axis sign is a convention (|cos| is the statistic).
> **N=32/64 legs:** the answer-margin backward at ≥6.5k tokens OOMs on 48GB L40S (588 OOM
> retries, then empty-grad crash) AND on the 40GB A100 slices — attempts 120144/120161; third
> attempt 120169 pends on the fully-occupied H200s. Blocked-with-note if no H200 frees before
> campaign close; the N≤16 stability result stands on its own.

| | L14 | L16 |
|---|---|---|
| \|cos(axis_16, axis_8)\| | 0.817 | 0.864 |
| d′ along native axis (N=8 / N=16 caches) | 0.72 / 0.66 | 0.55 / 0.77 |

**Reading:** between N=8 and N=16 the model's reading direction barely moves while its emitted
range stays clamped ([2026-07-10f]) — the clamp is a saturating MAGNITUDE/decoding phenomenon on
a fixed axis, not the readout re-aiming as context grows. Combined with C2/C3 (the readout only
decodes token-embedding lookups), the readout picture is: one fixed, poorly-aligned axis
(cos to w* ≈ 0.005, [2026-07-05]), read through a saturating value range.

## [2026-07-11n] ✅📊 E4 tags for the constructive chapter: multipass caches PASS adequacy at EVERY N (kurtE −0.1…+0.5, std-ratio 0.92–1.07 for N=32/64/128) and so does chunk-k=2 — while the joint caches fail progressively (kurt +25 @N=128, [2026-07-11e]); the joint-context tax is ALSO the adequacy-breaker: isolated/near-isolated processing yields clean equal-covariance Gaussian messages, licensing the law for every pipeline number

> **Runs:** CPU block-read jobs **120225–120230** →
> `outputs/ladder/image_longN/block_read/e4tags_*_{mp_N8,mp_N32,mp_N64,mp_N128,chunk_k2,chunk_k8}/`.
> Exceptions worth noting: mp_N8 fails only via std-ratio 0.63 (evidence class TIGHTER than
> noise — the near-ceiling separation regime, quote d′ as separability); chunk_k8 marginal
> (kurt +0.85/+1.10) — the first traces of joint-style corruption already visible at k=8,
> consistent with the k-curve. Every retrieve-pipeline supply number (k=1/k=2 messages) is now
> E4-licensed; the composed law∘clamp closure inherits the caveat only on its JOINT-d′ input.

## [2026-07-11o] ✅📊 P4b — HERBench armB retrieve-then-verify: NULL (honest refutation of even the "modest lift" expectation) — judge-verified tally 0.157 = rendered 0.157 vs frozen open 0.172 (MCQ 0.351); on graded-evidence real video the per-frame verifier IS the wall (MAE 2.87, systematic undercount of marginal evidence), and no aggregation machinery can add what perception doesn't supply — the regime-2 prescription ("fix perception, not aggregation") confirmed constructively

> **Run:** job **120223** → `outputs/pipeline/e2e_herbench/<ts>/` (report.txt, rows.json);
> `experiments/pipeline/e2e_herbench.py`. Verifier = the existing 2026-07-07 look-again scores
> (independent of any probe axis), thr 0.5, all 134 armB samples, tally → predicate-matched fact
> → frozen render. **Registered verdict: ✗-as-stated** — the prediction said "modest lift,
> bounded by d′≈1"; measured is NO lift (−0.015). Two clean sub-findings: (1) the render stage
> is perfectly faithful (rendered == tally on every sample — the token interface transports the
> tally losslessly even when it's wrong); (2) the pipeline's failure decomposes entirely onto
> the verifier: HERBench's forced-binary curation ([2026-07-07e]) showed only ~¼ of evidence is
> binary-groundable — the yes/no read misses the graded ¾, hence MAE 2.87. Contrast: MMRED
> (binary evidence) retrieve-verify = 0.71–0.88. The two regimes now have constructive
> demonstrations on both sides.

**Caveats.** thr=0.5 unswept (a calibrated threshold could trade FN/FP but cannot exceed the
judge's AUROC ceiling ~0.79 on graded frames); armA (evidence-only) not run; n=134.

## [2026-07-11p] ✅📊 P3 (algebra half) — THE PIPELINE GENERALIZES ACROSS THE TASK ALGEBRA: rooms_visited 0.993 (frozen 0.193 — effectively SOLVED by per-frame room-read → SUPPORT SIZE → support-size fact), co-occupancy 0.513 (frozen 0.127), both at 10 forwards/sample; the registered ordering rooms ≥ steps > cooc lands exactly, and the reduction operator is swapped per task with zero retraining

> **Runs:** `experiments/pipeline/e2e_task_algebra.py`, jobs **120215** (rooms) / **120216**
> (cooc) → `outputs/pipeline/e2e_algebra_{rooms,cooc}/<ts>/` (report.txt, rows.json). N=8,
> n=150, verify-all (no shortlist at this N), seed 3. Verifiers: rooms = per-frame "Which room
> is C in?" first-token logit read over the 6 room names → tally = |{answers}| (the distinct/
> union operator — the [2026-07-11f] dedup semantics, deployed); cooc = per-frame same-room
> yes/no → Σ. Render: predicate-matched fact per task ("C visited exactly K different rooms." /
> "C1 and C2 were in the same room in exactly K of the 8 frames.").

| task | pipeline (reduction / rendered) | frozen joint | lift | registered |
|---|---|---|---|---|
| rooms_visited | **0.993 / 0.993** (MAE 0.01) | 0.193 (MAE 1.24) | 5.1× | rooms ≥ steps ✓ (it TOPS the table) |
| steps (N=8, retrieve family) | 0.600–0.910 by mode | 0.173 | 3.5–5.3× | — |
| co_occupancy | 0.513 / 0.513 (MAE 0.70) | 0.127 (MAE 2.10) | 4.0× | cooc lower ✓ (pair-visibility perception is the verifier's known bottleneck, [2026-07-04-era per-frame verify 0.867]) |

**Readings.** (1) The constructive claim is now task-general: ONE skeleton (per-frame verify →
fixed extensive reduction → fact render), THREE reduction operators (Σ, distinct-count, Σ over
pair-predicate), all frozen, no training anywhere — rooms even demonstrates the support-size
operator that the learned adapters historically struggled with (union is exact once per-frame
room reads are right: per-frame room decode ≈0.99 at 392px). (2) Renders are again perfectly
faithful (rendered == tally on every sample in both tasks). (3) cooc's 0.513 decomposes onto
per-frame same-room verification (~0.92/frame → 0.92⁸ ≈ 0.51 with cancellation) — same
per-frame-supply bound as everywhere; better pair-perception is the only lever.

**Caveats.** N=8 only (long-N algebra untested); rooms first-token logit read assumes the six
park rooms (task-specific but query-independent); n=150, seed 3, single model.

## [2026-07-11q] ✅📊 P4j COMPLETE (memory fix + N=32/64 legs): the native reading axis DRIFTS MILDLY but does not rotate away — |cos(axis_N, axis_8)| @L16 = 0.86/0.72/0.72 for N=16/32/64 (coherence stable 0.62–0.74), and d′ along the model's own axis is FLAT in N (0.55/0.77/0.73/0.66) — the emission clamp is downstream saturation on a stable read, not context-driven re-aiming

> **Ops:** the [2026-07-11m] OOM blocker was 125 float params (norms/embeddings) left with
> requires_grad=True — they built the autograd graph from layer 0 regardless of the graph-start
> hook. Fix: explicit `model.requires_grad_(False)` + graph start at the first probed layer +
> `num_logits_to_keep=1` (`native_axis_probe.py`); N=32 now runs in 3 min on a 48G L40S (was
> OOM on 48G/40G, needed H200). H200 job 120169 cancelled; legs run as job **120238** →
> `outputs/ladder/image_longN/native_axis/N{32,64}/`, n=60/N, @392px, offset −9.

| L16 | N=8 | 16 | 32 | 64 |
|---|---|---|---|---|
| \|cos(axis_N, axis_8)\| | 1 | 0.864 | 0.717 | 0.723 |
| d′ along native axis_N | 0.55 | 0.77 | 0.73 | 0.66 |
| grad coherence | 0.69 | 0.67 | 0.70 | 0.74 |

The [2026-07-11m] partial verdict extends: the reading direction shifts ~25–30% of its angle by
N=32 then stabilizes, and reads a constant ~0.6 d′ at every N — neither rotation-collapse nor
read-collapse tracks the behavioral collapse (0.173→0.053 over this range). The failure remains
in emission (the [2026-07-10f] clamp), which law∘clamp already accounts for. L14 shows the same
pattern with a mild native-read decline (0.72→0.41).

## [2026-07-11r] ✅📊 P1a+P1b — THE k=2 TAX DISSECTED: it is TWO measured components — a saturating LONG-CONTEXT tax (isolated frame at joint positions via text padding: d′ 7.18→5.37/5.30 for 16/32-frame-equivalent padding — position NOT innocent, but pad16≈pad32) and a CONTENT-SIMILARITY interference tax graded exactly as registered (gray 6.40 / noise 6.35 / patch-shuffled 4.39 / other-scene 3.63 / same-scene 3.27 — matching the P1e k=2 anchor 3.37); a contentless visual neighbor costs almost nothing (−0.8), a same-domain scene costs −3.9

> **Runs:** P1a jobs **120235** (pad16/pad32, n=200; `--pad-to-frames`, question-neutral filler
> text before the image, prompt suffix identical) → `outputs/ladder/image_longN/position_ctrl/`
> + d′ `dprime_20260711_191524/`; P1b jobs **120236/120237** (5 companion types ×n=200;
> `--companion`, real frame always FIRST, messages captured for the real frame only) →
> `outputs/ladder/image_longN/companion/` + d′ `dprime_20260711_185105/` (fig_b1.png = the
> ladder). Anchor "alone" = multipass N8 (n=300). All @L16, room-token carrier, N=8 park data
> @392px. Smokes 120211.

| condition (one evidence frame, read @L16) | d′_w |
|---|---|
| alone (k=1 multipass) | 7.18 |
| + 15/31-frame-equivalent TEXT padding (position/context only) | **5.37 / 5.30** |
| + gray companion | 6.40 |
| + uniform-noise companion | 6.35 |
| + patch-shuffled scene companion | 4.39 |
| + real frame, DIFFERENT scene | 3.63 |
| + real frame, SAME scene | **3.27** |
| joint N=8 (7 same-scene companions) | 1.97 |

**Registered verdicts.**
1. P1a ("d′ falls toward 2–3.4 with no companion → positional; stays 7–8 → position innocent"):
   **neither branch cleanly — position costs a real, SATURATING ~1.9 d′** (5.3, identical at
   pad16/pad32) but cannot produce the content ladder. Behavioral per-frame perception under
   padding is intact (0.989; mp-sum 0.93).
2. P1b ("flat across companions → any-visual-neighbor; graded → cross-frame feature
   interference"): **GRADED — cross-frame feature interference confirmed.** The cost tracks
   content-domain similarity monotonically; a contentless neighbor is nearly free (−0.8),
   killing the token-budget/attention-allocation account.
3. Synthesis with [2026-07-11g] (renorm null): the interference is content-carried but NOT
   via carrier-mass competition, and [2026-07-08c]-era fence results say cutting cross-frame
   visual edges wholesale doesn't recover it either (off-distribution confound documented).
   The remaining consistent picture: same-domain companions corrupt the frame's in-context
   ENCODING through cross-frame attention content (V-mixing), in a way that naive edge-cutting
   breaks rather than repairs. P1c (partial-depth fence sweep, in flight) probes the depth.
4. Pipeline note: gray-padding companions ≈ free → a "pad-to-k" trick canNOT recover multipass
   d′ in fewer forwards (padding with content-free frames keeps d′ 6.4 but wastes the slot);
   the accuracy-per-forward frontier stays at k=1–2 with real frames.

**Caveats.** Real frame always first (order effect unswept); text padding ≠ visual-token padding
(M-RoPE positions differ by modality — the 5.3 is the TEXT-context tax; a gray-frames padding
arm would give the visual-position variant and its number is ~6.4 from the companion gray arm at
k=2); n=200/arm, single seed, steps task.

## [2026-07-11s] ✅📊 P1c — DEPTH LOCALIZATION NULL: partial-depth fencing hurts at EVERY depth (joint 1.97 → fence[0,4) 1.77 → fence[0,8) 1.63 → fence[0,12) 1.56 → fence[0,14) 1.80 @L16) — no depth recovers any of the content-interference tax; the registered "d′-vs-L curve names the depth" instrument fails INFORMATIVELY: cross-frame attention edges cannot be removed at any depth without net signal loss in the frozen model

> **Runs:** job **120244** (fence-upto ∈ {4,8,12}, n=150, N=8 @392px) →
> `outputs/ladder/image_longN/depth_fence/upto{4,8,12}/` + d′ **120256** →
> `depth_fence/dprime_20260711_192154/`; anchors joint + fence[0,14) from B1 (n=300).

**Reading.** With P1b showing content-GRADED interference and P1a showing a separate saturating
context tax, the natural expectation was that cutting cross-frame visual edges below some depth
would recover part of the 3–4 d′ content tax. It recovers none, anywhere — the fence curve is
monotone-worse through L12 (and the fuller [0,14) fence sits at 1.80, still below joint). Two
readings compatible with all data: (i) the interference travels through the SAME edges that
carry useful cross-frame computation the frozen model depends on (norm statistics, shared-scene
context) — cutting removes both, net negative at every depth; (ii) part of the tax enters through
non-edge channels (the P1a context tax is ~1.9 of it by construction). Either way, the
constructive consequence is unchanged and now fully justified: the ONLY working lever on the
supply is separate/near-separate forwards (k≤2), which is exactly what the retrieve pipeline
deploys. Depth localization of the content component needs a finer instrument (e.g., per-layer
activation patching of companion content, not edge removal) — noted as future work.

**Caveats.** n=150/arm; fence blocks visual→visual edges only (text-relay edges intact); the
behavioral column degrades with fence depth (0.21→0.13) consistent with off-distribution damage.

## [2026-07-11t] ✅📊 P2 — RETRIEVE-THEN-VERIFY v2 (k=2 chunk shortlist) + 3-SEED ERROR BARS: accuracy improves at scale over v1 (N=128: 0.791±0.055 vs v1 0.707; N=64: 0.853 vs 0.807; N=32: 0.862±0.014 ≈ v1 0.880) — but the registered COST prediction is REFUTED: keep-rate stays ~65–70% (registered 25–35%), so v2 costs ≈1.15N forwards (fig_frontier.png written)

> **Runs:** jobs **120212/120213/120214** (seeds 3, N=32/64/128) + **120217/120219** (N=32 seeds
> 4/5) + **120218/120220** (N=128 seeds 4/5) → `outputs/pipeline/e2e_retrieve2_N*/` (reports,
> rows). Same protocol as v1 (FN≤2% calibration, 30 samples/N, eval seed-disjoint), shortlist
> margins from k=2 chunk passes (d′ 3.37) instead of the joint pass.

| N | v2 exact (seeds) | v1 (seed 2) | mp-sum | frozen | v2 fwd/sample |
|---|---|---|---|---|---|
| 32 | **0.862 ± 0.014** (.860/.880/.847) | 0.880 | 0.680 | 0.073 | 36.9 |
| 64 | 0.853 (single seed) | 0.807 | 0.580 | 0.053 | 76.8 |
| 128 | **0.791 ± 0.055** (.833/.713/.827) | 0.707 | 0.420 | 0.020 | 150.1 |

**Registered verdicts.** Accuracy ≈ v1 at small N and BETTER at large N ✓ (the k=2 margins
don't dilute with N, so the shortlist quality is N-invariant — v1's joint margins degraded
exactly where it mattered); keep-rate 25–35% **✗ refuted** — measured 65–70%: even d′-3.37
margins under an FN≤2% recall constraint must keep two-thirds of frames (the gate's ROC at 98%
recall allows ~35% rejection, not 70%). Consequently cost ≈ N/2 chunk-forwards + 0.65N verifies
≈ 1.15N — v2 buys accuracy, not cost. In TOKEN terms v2 is still much cheaper than v1 at N=128
(k=2 forwards are ~500 tokens vs the 25k joint pass). The 3-seed spread at N=128 (±0.055,
driven by seed 4's 0.713) is the honest error bar on the headline; N=32 is tight (±0.014).
**fig_frontier.png** (accuracy vs forwards/sample; frozen / chunked-k8 / mp-sum / v1 / v2 at
N=32/64/128) → `outputs/ladder_report/fig_frontier.png`. Steps-task 3-seed rerun requirement
(P3c) is satisfied by the same runs.

**Caveats.** fwd/sample counts forwards, not tokens (footnoted in the figure); N=64 single-seed;
per-N calibration uses 30 labeled samples as before.

## [2026-07-11u] ✅📊 P4a — CROSS-FAMILY PORTABILITY: verify-then-tally on InternVL2.5-8B (steps N=32, verify-all, zero Qwen components, zero training) scores 0.690 exact vs its frozen joint 0.090 — a 7.7× lift; the rendered answer equals the tally on every sample (the fact interface transports losslessly in the second family too)

> **Run:** job **120245** (smoke 120232→120243; two fixes: InternVL forward requires
> pixel_values — text-only renders get a gray dummy image + explicit ignore instruction; manual
> greedy decode since the remote code lacks .generate) → `outputs/pipeline/e2e_internvl/<ts>/`.
> `experiments/pipeline/e2e_internvl.py`: per-frame yes/no logit verify → Σ → fact render,
> n=100, seed 3. Registered ("works — multipass supply 6.5 clears the crush line"): **✓**.
> Design note: no gate/shortlist (no InternVL mass-cache; verify-all at 33 forwards/sample) —
> the one-number portability claim per the charter. Tally MAE 0.51: InternVL's per-frame yes/no
> is slightly noisier than Qwen's (its 0.690 vs Qwen-N=32-verify-all-equivalent ~0.86), matching
> its slightly lower multipass supply (6.5 vs 7.2).

## [2026-07-11v] ✅📊 P5 — RESIDUAL-LEVEL COMPOSITIONAL INJECTION (L14–17 fact slot): the registered prediction lands — every learned route trains in-range (digit 0.750 / count 1.000 / Fourier 0.923 / native-anchored Fourier 1.000) and FAILS held-out (0.080 / 0.018 / 0.000 / 0.000), matching the embedding level exactly; "the token interface is necessary" is now airtight at BOTH injection levels

> **Runs:** `--inject-level residual` added to `experiments/readout/c2_digit_codebook.py`
> (count slot holds same-length dummy digits so tokenization/positions are unchanged; learned
> vectors REPLACE the residual stream at the slot positions at layers 14–17 entries via
> pre-hooks; init rescaled to the measured residual norm ≈96 at the slot; generation-safe
> hook guard). Smoke **120221** → full job **120233** →
> `outputs/readout/c2_residual/20260711_181926/` (results.json, recs, report). Same splits,
> no trained head, 5 epochs × 60 reps.

| route (residual, L14–17) | params | in-range | held-out |
|---|---|---|---|
| digit codebook | 35,840 | 0.750 | 0.080 |
| count codebook (+interp) | 46,592 | 1.000 | **0.018** |
| Fourier basis | 28,672 | 0.923 | **0.000** |
| native-anchored Fourier | 28,672 | 1.000 | **0.000** |
| (embed-level token reference, [2026-07-11b]) | 0 | 1.000 | **1.000** |

**Notes.** (1) The residual-run's own token_fact row (0.077/0.000) is NOT the C1 bar — in
residual mode the prompt's count slot carries dummy digits by design, so the untrained token
route reads the dummy (its only hit is K=5 == the 1-digit dummy "5"); the valid token reference
is the embedding-level 1.000/1.000. (2) In-range trainability at the residual level is even
EASIER than embed level (count/fourierE reach 1.000) — the model will happily read arbitrary
residual vectors it was trained to map — and OOD failure is equally total: the readout performs
value lookup, not geometry decoding, at every level we can inject. (3) The too-good gate never
triggered (no positive to audit; no head used).

**Caveats.** One layer window (14–17); replacement (not additive) injection; single seed;
dummy-digit slot design means residual routes also carry a (constant) dummy-token signal —
the learned vectors must OVERRIDE it, which they do in-range, confirming the injection is
causally read.

## [2026-07-12] ⚠️ CORRECTION to [2026-07-11r] — the companion-ladder claim "same-domain companions corrupt the frame's in-context ENCODING" was OVERSTATED: in every P1b pair the real frame came FIRST, so under causal attention its keys/values were computed BEFORE the companion existed — the content-graded d′ drop (gray 6.40 → same-scene 3.27) can only have entered through the CARRIER'S READ of the frame, i.e. the within-frame routing ŵ_j set by the carrier query q_c (which forms after the companion), not through the frame's own k/v

> Correction credit: Tal (2026-07-11 session review). Three consequences, logged before the
> P1 (2026-07-12) experiment that separates the survivors:
> 1. **Re-scope:** [2026-07-11r]'s mechanism label becomes "READ-SIDE (query-routing)
>    contamination, demonstrated for frame 1; encode-side untested for frames 2..N" — for the
>    first frame of a causal pair, k_j and v_j are companion-independent by construction, so
>    the entire measured content-graded effect on THAT frame is carried by q_c.
> 2. **Factorization to carry forward:** msg_f = m_f · o_proj(Σ_j ŵ_j v_j) — scale m_f
>    (algebraically EXCLUDED from d′, which is per-frame-scale-invariant; also the refuted
>    renorm target), within-frame routing ŵ_j (q-side), values v_j (encode-side). The 2×2
>    swap experiment (joint vs clean q × joint vs clean k/v) separates the last two.
> 3. **Instrument note:** masking the carrier's softmax to one frame is ALGEBRAICALLY the
>    refuted renorm variant — within-frame softmax shape is unchanged by removing other
>    frames' logits (softmax ratios are subset-independent); only m_f changes, and d′ doesn't
>    see m_f. That variant tests nothing new and is not run.
> Downstream text updated with supersession notes (STORY.md §supply, theory_background.html
> Wall Ⅰ) after the P1 verdict lands. [2026-07-11r]'s MEASUREMENTS are untouched — only the
> mechanism attribution is narrowed.

## [2026-07-12b] ✅📊 P1 — THE QUERY/ENCODING 2×2 LANDS: QUERY CONTAMINATION IS DOMINANT — with the carrier query taken from the JOINT forward, per-frame d′ collapses to the joint level REGARDLESS of whose values are read (joint-q × clean-kv = 1.59 ≈ joint×joint 1.71 @L16), while a CLEAN query reading the JOINT-encoded values retains most separability (clean-q × joint-kv = 3.14 of clean×clean 4.79); Tal's session prediction confirmed — the joint-context tax lives in the carrier's mis-routed within-frame attention ŵ_j, with a real but secondary encoding component

> **Runs:** capture job **120529** (n=150, 3 forwards/sample: joint / multipass / text-padded
> single; pre-rotary q_c, k_j, v_j at L14/16 + joint rope slices + dequantized o_proj; 3.3 GB) →
> `outputs/ladder/image_longN/qkv_2x2/20260712_full/`; analysis job **120530** (within-frame
> softmax, ONE consistent joint geometry, o_proj applied, held-out LDA ×3 seeds) →
> `.../20260712_full/analysis/` (report.txt, results.csv). New code:
> `experiments/glstm/qkv_2x2_{capture,analysis}.py` (smokes 120453 + login n=4).

**ANCHOR GATE (charter: hard):** the literal band FAILS — pad×pad 4.79 < 5.3 @L16 (L14 2.68) —
and the required debug attributes it to a UNIFORM reconstruction compression: (i) all arms are
read under the JOINT rope geometry (the controlled-mixing choice; it moves the clean arms' native
carrier↔frame relative phases), (ii) fp16 capture + dequantized-o_proj path. The compression is
structure-preserving: mp×mp / pad×pad / joint×joint = 5.55 / 4.79 / 1.71 reproduces the native
7.18 / 5.37 / 1.97 ordering with matched RATIOS (joint/mp: 0.31 vs native 0.27; pad/mp: 0.86 vs
0.75). Off-diagonals are therefore interpreted on the reconstruction's own scale, with this
caveat attached to every number below.

| L16 (d′_w, rows = q-arm, cols = kv-arm) | pad-kv (clean) | joint-kv | mp-kv |
|---|---|---|---|
| pad-q (clean) | **4.79** | 3.14 | 4.87 |
| joint-q | **1.59** | 1.71 | 1.60 |
| mp-q | 5.33 | 3.48 | **5.55** |

**Registered verdicts.**
1. "joint-q × clean-kv ≪ clean×clean ⇒ query contamination dominant" — **✓ LANDS, maximally**:
   the joint query flattens EVERY kv arm to 1.59–1.71 (share of the clean−joint gap ≈ 104%).
2. "clean-q × joint-kv ≪ clean×clean ⇒ encoding dominant" — partial: 3.14 < 4.79 (share ≈ 54%,
   overlapping with 1) — the joint-encoded values DO carry most of the frame's evidence; the
   encoding component is real but secondary.
3. Mechanism statement (supersedes [2026-07-11r] per the [2026-07-12] correction): the
   joint-context tax is carried chiefly by the CARRIER QUERY — q_c, formed after attending the
   whole multi-frame context, mis-routes within each frame's 196 tokens (ŵ_j points at the
   wrong patches), even when the values it mixes are clean. Consistent with everything prior:
   renorm null (mass ≠ routing), fence null (the query reads through legal edges), companion
   grading (q_c's corruption scales with how much same-domain content it absorbed), first-frame
   causality (k,v companion-free yet d′ drops — because the READ drops).

**P1d gate (registered "wire the clean-query joint read into the pipeline if d′ ≥ 5"):** the
grid's clean-q × joint-kv cell IS that fix's d′ — **3.1–3.5, below the ≥5 wiring bar** → not
wired. Noted for the thesis: a clean-query read of one joint forward delivers k=2-chunk-level
margins (≈3.4) at ~2 forwards for ANY N — the economics beat k=2 (N/2 forwards) even though
the supply is unchanged; a deployable frameless-q variant is measured next.

**Caveats.** Reconstruction scale ~0.85× native (documented above; all comparisons within-grid);
n=150, L14 shows the same pattern compressed (joint-q row 1.38–1.63 vs pad×pad 2.68); q for the
joint arm is the single carrier query shared across frames (as deployed); fp16 capture.

## [2026-07-12c] ✅📊 P1d CLOSED (both branches negative, mechanistically decisive): the deployable "frameless clean query" reads joint-encoded frames at d′ 0.53–1.07 — WORSE than the joint query itself (1.71) and far below the per-frame clean query (3.14) — so the query is not "clean vs contaminated" as a static object: q_c carries FRAME-CONDITIONED routing (where to look inside THIS frame), acquired only by attending the frame; a single query attending 8 frames holds a compromise routing program for all of them — the joint-context tax is a ONE-QUERY-MANY-FRAMES binding/capacity limit

> **Runs:** frameless-q capture job **120533** (one gray-image, filler-padded forward per sample,
> question identical; q_c at L14/16) + CPU analysis **120539** →
> `outputs/ladder/image_longN/qkv_2x2/20260712_full/{frameless_q.pt,frameless_analysis/}`.
> `experiments/glstm/frameless_q_{capture,analysis}.py`.

| query variant (× joint-kv, @L16) | d′_w |
|---|---|
| per-frame clean q (saw frame f alone, padded) | 3.14 |
| joint q (saw all 8 frames) | 1.71 |
| **frameless q (saw no frame)** | **0.53** |

**Readings.** (1) The monotonicity frameless < joint < per-frame in frames-seen-specificity
nails the mechanism: within-frame routing quality tracks how specifically q_c was conditioned
on the frame being read. The joint query is not "noise-corrupted" — it is a rate-limited
compromise: one 3584-dim vector cannot hold 8 frame-specific routing programs (the thesis's
over-squashing lens, now at the QUERY: the bottleneck squeezes not the evidence but the
addressing). (2) **P1d wiring: dead on both counts** — the ≥5 bar unmet (3.1–3.5 ceiling even
with per-frame clean queries) and the 2-forward economics variant fails outright (0.53). The
only clean queries are per-frame ones, i.e., exactly the k=1/2 forwards the pipeline already
uses. P2 (adaptive verification) proceeds as the cost lever.

**Caveats.** Frameless q built with a gray image in the slot (the model may treat "no real
frame" pathologically; a text-only-question q was not tested — but the direction of the result
makes the deployable variant moot regardless); same reconstruction-scale caveat as
[2026-07-12b]; n=150.

## [2026-07-12d] ✅📊 P3 — LONG-N TASK ALGEBRA: the support-size pipeline holds at scale — rooms_visited 0.993 @N=32 and 0.967 @N=128 (frozen: 0.300 / 0.213; registered "≥0.9 both N" ✓✓); co_occupancy degrades gracefully 0.367 → 0.233 (frozen 0.073 / 0.060) — just BELOW the registered 0.3–0.5 band at N=128 (scored partial): the pair-visibility verifier's ~0.92/frame error compounds over 128 frames exactly as the per-frame-supply bound dictates

> **Data:** `data/mmred_longN_{rooms_visited,co_occupancy}/seq_len_{32,128}/` (jobs 120449–52;
> render 512, 5-char, per-count 75/35/27; rooms counts native 1–6 — the support-size target
> does NOT grow with N, isolating the operator from count-range effects).
> **Runs:** jobs **120503/120505** (N=32) and **120504/120506** (N=128) →
> `outputs/pipeline/e2e_algebra_{rooms,cooc}_N{32,128}/<ts>/` (n=150, verify-all, seed 3;
> report headers say "N=8" — a cosmetic label bug; forwards/sample 34/130 confirm the N).

| task | N=32 | N=128 | frozen (32/128) | registered |
|---|---|---|---|---|
| rooms_visited (distinct-count operator) | **0.993** | **0.967** | 0.300 / 0.213 | ≥0.9 both ✓✓ |
| co_occupancy (pair-Σ operator) | 0.367 | 0.233 | 0.073 / 0.060 | 0.3–0.5: ✓ @32, **✗ @128 (0.233)** |

**Readings.** (1) The distinct-count operator is nearly N-INVARIANT (0.993→0.967 over 4× frames)
— set-union absorbs per-frame read errors unless a NEW room is hallucinated; with per-frame room
reads at ~0.99 the union stays clean. This is the strongest long-N number in the project and the
cleanest demonstration that the operator choice, not N, governs pipeline scaling. (2) cooc's
graceful decay tracks its verifier: ~0.92 per-frame same-room accuracy → sum over 128 frames →
MAE 5.3 (vs the frozen model's catastrophic 43.7 — it answers small numbers while gold reaches
128). The honest miss vs the 0.3 band-edge at N=128 re-confirms the universal bound: pipeline
accuracy = f(per-frame verifier quality, operator error-absorption), and pair-predicates are the
weakest verifier in the family. (3) Renders lossless in all four cells (rendered == tally).

**Caveats.** verify-all (no shortlist) — N+2 forwards; cooc gold band at N=128 includes counts
up to 128 (exact-match is harsh; MAE tells the graceful-degradation story); seed 3, n=150/cell.

## [2026-07-12e] ✅📊 P2 — RETRIEVE COST FIX: both registered variants REFUTED, and P1 explains why — adaptive early-stop loses 0.10–0.13 accuracy for ≤13% cost savings (0.733/0.740/0.693 @32/64/132 fwd); two-stage joint-prefilter is within 0.033 of v2 only at N=64 (0.773/0.820/0.680 @31/72/142 fwd); the verification cost is CALIBRATION-BOUND: with d′≈3.4 margins under a recall-safe constraint, ~⅔ of frames must be verified regardless of scheduling — and [2026-07-12c] shows margins can't get better without per-frame-conditioned queries, i.e., the k=1/2 forwards themselves

> **Runs:** adaptive jobs **120534/120535/120536**, two-stage **120556/120557/120558** →
> `outputs/pipeline/e2e_{adaptive,twostage}_N{32,64,128}/<ts>/` (n=150, seed 4). Adaptive =
> descending-margin verify with a ceil(0.15N) consecutive-no stop (no threshold); two-stage =
> joint pre-filter (FN≤1%) → k=2 margins on survivors (FN≤2%) → verify.

| N | retrieve-v2 (ref) | adaptive | two-stage |
|---|---|---|---|
| 32 | 0.862 @36.9 | 0.733 @32.0 | 0.773 @31.1 |
| 64 | 0.853 @76.8 | 0.740 @64.0 | **0.820 @72.4** |
| 128 | 0.791 @150.8 | 0.693 @132.5 | 0.680 @142.1 |

**Registered verdicts.** P2a (cost O(evidence+window), accuracy within 0.03): **✗ both counts** —
the stability window rarely triggers early because evidence sits throughout the 65%-quality
margin ordering. P2b fallback: within-band at N=64 only (−0.033), **✗** at 32/128; savings ≤15%.
**Closing synthesis (the campaign's cost story):** the ~N verification floor is not a scheduling
failure but a margin-quality consequence — d′ 3.4 margins at 98% recall keep ~⅔ of frames, and
the P1 mechanism ([2026-07-12b,c]) proves better margins need frame-conditioned queries, which
cost one forward per frame(-pair) by construction. Accuracy and cost are two faces of the same
supply bound. v2 (0.79–0.88, ~1.15N forwards) stands as the deployable configuration; below-N
cost at matched accuracy would require breaking the one-query-many-frames limit inside the
model — a training-time question, out of frozen-model scope.

**Caveats.** Window/thresholds per registered protocol (0.15N, FN≤2%/1%) — other operating
points trade differently but cannot beat the margin ROC; seed 4, n=150/N.

## [2026-07-12f] 📊 Session note — parallel threads by Tal (same account, 2026-07-12): the P1 capture (job 120529) and grid analysis (120530) that this campaign's [2026-07-12b] uses were submitted from Tal's side using the campaign's scripts; his follow-up own-geometry anchor variant (`qkv_2x2/20260712_geomfix`, jobs 120537/120566, n=8) is exploratory-scale (LDA noise floor at n=8 — not interpretable yet) and remains HIS open thread: a full-n own-geometry rerun would test whether native-geometry anchors land in the 5.3–7.2 band and firm up the [2026-07-12b] reconstruction-compression account.

## [2026-07-12g] ✅📊 QUERY-GENERALIZATION test (does one frame's query read others? → the deployable-query-fix test): DECISIVELY NEGATIVE and stronger than "no" — a per-frame query reading a DIFFERENT frame scores d′ 1.02 @L16, WORSE than the generic joint query (1.71) and far below its own frame (3.48); carrier queries are frame-SPECIFIC routing programs that are ACTIVELY WRONG on other frames, so no single-query "one probe → read all N" fix exists
> **Motivation:** [2026-07-12b/c] showed query-side dominance; the untested question for a DEPLOYABLE
> fix was whether a query from ONE real-frame forward generalizes to read OTHER frames (transfer =
> 1-forward fix; frame-specific = no fix). CPU recompute on the existing n=150 2×2 capture
> (`qkv_2x2/20260712_full`), joint rope geometry, job **120609** →
> `.../query_gen/`. New: `experiments/glstm/query_generalization.py`.
>
> | read frame f with… (L16) | d′_w |
> |---|---|
> | its OWN clean query | 3.48 |
> | the JOINT query (compromise over all 8) | 1.71 |
> | a DONOR query (from a different frame) | **1.02** |
>
> **Readings.** (1) donor < joint < own: a query tuned to frame A mis-routes on frame B (worse than
> the compromise joint query, which at least attended B) — even though all 8 MMRED frames are the
> SAME scene (per-frame character arrangement differs enough to break transfer). Queries are not
> sharp-vs-blurry detectors; they are frame-specific "where to look in THIS arrangement" programs.
> (2) **Kills the query-transfer fix** ("one probe forward → reuse its query to read all N"): ruled
> out. The masked-carrier-probe alternative (N question-probes, one forward) still gives N
> frame-specific queries but reads JOINT-encoded frames (cap ≈3.1) at ≈multipass compute while
> multipass reads CLEAN frames (cap ≈7) — dominated. **No deployed-query trick improves aggregation
> on the frozen model.** (3) Confirms the one-query-many-frames capacity limit at its strongest: even
> one frame's *perfect* query is worse than the generic query on any other frame. The per-frame tally
> pipeline is necessary; the cost lever is shortlisting (fewer per-frame reads), not query cleverness.
> **Caveats:** reconstruction-scale caveat as [2026-07-12b] (relative comparison, joint geometry); n=150,
> steps, L14/16; donor set = other same-scene frames (cross-scene could differ but only strengthens the point).

## [2026-07-12h] ✅📊 ENCODING UN-MIXER — the encoding half of the joint-context tax is REVERSIBLE: a small learned map (trained joint-kv→mp-kv, label-free) recovers ~98–100% of the clean-encoding d′ gap, while the query half is a capacity limit — the two halves of the tax are mechanistically OPPOSITE
> **Motivation:** the 2×2 ([2026-07-12b]) split the joint tax into query (dominant) + encoding
> (secondary, clean-q×joint-kv 3.1 vs clean-q×mp-kv 4.4). This tests whether the ENCODING gap is
> entangled-but-recoverable (learned map cleans it) or destroyed. CPU, existing 2×2 capture
> (`qkv_2x2/20260712_full`); train g_k,g_v per-token pre-rotary joint→mp on 90 samples (label-free
> reconstruction), apply to 60 held-out, read with the PERFECT (mp) query, d′ vs joint/mp anchors.
> Job **120621**; new `experiments/glstm/encoding_unmixer.py`.
>
> | read joint-encoded frame, perfect query (L16, recon units) | d′_w |
> |---|---|
> | joint values (baseline) | 3.13 |
> | **un-mixed values (ridge)** | **4.37** (98% of gap) |
> | **un-mixed values (MLP)** | **4.67** (≈ceiling, within n=60 noise ±0.3) |
> | clean/mp values (ceiling) | 4.39 |
>
> **Readings.** (1) A label-free learned map recovers essentially ALL the encoding-degradation gap
> → the joint context does NOT destroy per-token evidence; it applies an invertible mixing an
> adapter undoes. **Encoding degradation is REVERSIBLE.** (L14 same pattern: 1.48→1.92/2.41,
> ceiling 2.21.) No label leak (trained on rep-reconstruction; d′ on disjoint eval); legitimately
> can exceed the 3.1 pooled-message figure because it acts on the 196 per-token reps before pooling.
> (2) **The two halves of the joint-context tax are mechanistically OPPOSITE:** encoding =
> reversible mixing (recoverable); query = capacity limit (a wrong query ACTIVELY destroys the read
> — donor 1.02 < joint 1.71 < own 3.1, [2026-07-12g]). The frozen joint VALUES are fine; the
> irreducible wall is the QUERY's addressing capacity. (3) **Deployability caveat:** used the
> perfect per-frame query, so not a standalone deployed fix (with the joint query the read is
> query-limited). The decisive follow-up is a QUERY un-mixer (reconstruct q^(f) from the joint
> state): if it works, one joint forward + two small adapters = full recovery with no per-frame
> forwards (trained, cheap at inference); if not, the query capacity limit is the floor.
> **Caveats:** recon-scale (recovers the clean-q encoding gap 3.1→4.4, not the recon→direct 4.4→7.2
> deflation); n_eval=60 (d′ ±0.3, MLP-over-ceiling within noise → read "≈100%"); steps, L14/16.

## [2026-07-12i] ✅📊 QUERY UN-MIXER — the query half is IRREDUCIBLE: it cannot be reconstructed from the joint forward (recovers −37%, reads WORSE than the joint query), so the two halves of the tax are asymmetric — encoding reversible ([2026-07-12h]), query a hard capacity limit → NO one-forward deployable fix
> **Motivation:** [2026-07-12h] showed encoding degradation is reversible; the decisive follow-up is
> whether the per-frame query q^(f) is reconstructable from the SINGLE joint forward (→ one-forward
> fix with two adapters). Learned map (ridge) from [joint carrier q + mean-pooled frame-f joint k,v]
> → mp query; value un-mixers (MLP) reused for the combined condition. CPU, existing capture; job
> **120622**; new `experiments/glstm/query_unmixer.py`. **Data-limited: only ~720 query examples
> (1/frame) for a 3584-dim target vs 150k tokens for values — recovery is a lower bound.**
>
> | read frame f from ONE joint forward (L16, recon units) | d′_w |
> |---|---|
> | joint-q × joint-kv (deployed baseline) | 1.71 |
> | **unmix-q × joint-kv** (reconstructed query) | **1.19** (−37%: WORSE than baseline) |
> | mp-q × joint-kv (query ceiling) | 3.13 |
> | unmix-q × unmix-kv (BOTH adapters) | 1.48 |
> | mp-q × mp-kv (full ceiling) | 4.39 |
>
> **Readings.** (1) The reconstructed query reads BELOW the joint query (1.19<1.71) — a data-starved
> underfit would regress toward the mean query (≈baseline), not produce a worse-than-baseline
> mis-route; the map produces donor-like frame-specific-but-wrong queries. **The per-frame routing
> program is not present in the one-query-many-frames joint state; it can only be CREATED by
> attending the frame (a per-frame forward).** (2) Converges with two training-free results:
> frameless q 0.53, donor q 1.02 ([2026-07-12g]) — three independent angles, same verdict. (3)
> **The two halves of the joint-context tax are ASYMMETRIC:** encoding = reversible mixing
> (un-mixer ~100%, [2026-07-12h]); query = irreducible capacity limit (this). Cleaning values is
> free but wasted behind a broken query (both-adapters 1.71→1.48, no gain). **⇒ NO one-forward
> deployable supply fix; per-frame (multipass/chunk) forwards are provably the floor — the one thing
> joint processing cannot preserve is the addressing.** (4) L14 identical pattern (query −5%,
> both +13% — within noise of null).
> **Caveats:** query un-mixer data-limited (720 ex, ridge; a larger capture could firm the null —
> but negative-not-weak + frameless/donor convergence make "irreducible" the strong reading);
> recon-scale; n_eval=60 (±0.2–0.3); steps, L14/16. A definitive version = capture ~1k+ samples for
> the query map (parked).

## [2026-07-13] ✅📊 n=500 FIRM-UP of the query/encoding dissection — the reconstruction ANCHOR now VALIDATES (mp×mp own-geometry 7.81 ≈ direct 7.18), so the 2×2 is on the REAL d′ scale; value un-mixer 93% at scale; and query-irreducibility is AIRTIGHT — the query un-mixer still fails NEGATIVELY (−27%) at 3–4× the data
> **Motivation:** the [2026-07-12b–i] dissection was n=150, single-seed, and the reconstruction anchor
> FAILED at small n (deflated absolutes). Re-capture at n=500 (job **120639**, 13.4 GB) + analysis
> battery (job **120645**, 180 GB — the first pass OOM'd on the 4h_0g 16 GB cap; note: 13.4 GB capture
> needs a GPU-QOS high-mem node). Dir `outputs/ladder/image_longN/qkv_2x2/20260712_n500/`.
>
> **1 · ANCHOR VALIDATES (the key methodological win).** L16 fixed-geometry pad×pad **6.29** (band
> 5.3–7.2 ✓), joint×joint **2.33** (2.0±0.5 ✓); own-geometry mp×mp **7.81** ≈ **direct 7.18** ✓,
> pad×pad(own) 6.84 ✓. ⇒ the earlier "reconstruction deflation" was just **finite-sample d′
> under-estimation**, not a recompute bug; at n=500 the CPU reconstruction matches the real forward.
> **The 2×2 numbers are now quotable on the true d′ scale.** (L14 anchor still low — intrinsically
> lower-d′ layer; quote L16.)
>
> **2 · The query/encoding 2×2, real scale (L16, d′_w):**
> | q \ kv | pad(clean) | joint | mp(clean) |
> |---|---|---|---|
> | pad(clean) | 6.29 | 3.97 | 6.13 |
> | **joint** | 1.97 | **2.33** | 1.91 |
> | mp(clean) | 6.92 | 4.47 | **7.33** |
>
> Joint query collapses the read to ~2 regardless of encoding (1.91 even on clean values); a clean
> query recovers to 4–4.5 on joint values, 6–7 on clean. QUERY-contamination share **1.09**,
> ENCODING share 0.59 — query dominant, confirmed at scale.
>
> **3 · Value un-mixer: RECOVERABLE at scale** (L16, MLP): joint 3.82 → un-mixed **6.17** ≈ ceiling
> 6.33 (**93%**); ridge 54%. Encoding degradation is reversible — confirmed n=500.
>
> **4 · Query un-mixer: IRREDUCIBLE, now AIRTIGHT** (L16): joint 2.09 → un-mixed **1.63** (−27%,
> WORSE than baseline) with ~2,400 training examples (3–4× [2026-07-12i]'s 720). The negative result
> replicates at scale — NOT a data-limitation artifact. Both-adapters recovers −6% (the value gain is
> wasted behind a broken query). **The joint-context tax is confirmed: reversible value-mixing +
> irreducible query-addressing-capacity limit; per-frame forwards are the supply floor.**
> **Caveats:** n_eval=200, single capture seed (7); L16 (peak carrier); the [2026-07-12b–i] entries'
> relative conclusions all hold — this entry puts them on the validated absolute scale and removes the
> data-limitation caveat on query-irreducibility.

## [2026-07-14] ✅📊 REPLICA CARRIERS — per-frame question replicas in ONE forward: the UNMASKED (plain interleaved prompt) arm reads d′ 3.56 @L16 (+81% over joint 1.97, above the pre-registered GO bar ≥3) and BEATS the masked-attention arm (2.52); the masked arm's per-copy ladder exposes a design subtlety (the mask cleans queries everywhere but values only for frame 0); the predicted contamination decay in the unmasked arm did NOT appear at N=8

> **Motivation:** the query half of the joint-context tax is architecturally irreducible
> ([2026-07-13] trained-query NO-GO: shared q* 0.36–0.48). Replica carriers test per-source
> addressing via TOKEN TOPOLOGY instead of weights: insert a copy of the question after every
> frame; each copy's room token is a per-frame local carrier. Two arms: MASKED (custom all-layer
> 4D mask — each replica attends only {prefix, its own frame, itself} and is INVISIBLE to all
> other tokens, so the original computation is undisturbed) and UNMASKED (plain interleaved
> prompt, everything visible — the prompt-engineering control, proposed by Tal).
> **Script:** `experiments/glstm/replica_carrier_probe.py` (new; reuses the fence 4D-mask
> injection, the probe qkv/rotary capture, and the 2×2 o_proj dequantization). **Runs:** job
> **121431** (a100, 88 min: masked smoke n=8 → masked full n=300 → unmasked full n=300; job
> 121401 = earlier attempt, failed on an image_token_groups signature); steps task,
> mmred_images_park seq_len_8, 392px, forward-only. Dirs:
> `outputs/ladder/image_longN/replica_carrier/20260714_214534/` (masked) and
> `…/replica_carrier_nomask/20260714_221634/` (unmasked); held-out shrinkage-LDA d′
> (dprime_pair, 3 sample-disjoint seeds), n=300, skip=0.

| read @L16 (n=300) | mean d′ | per-copy d′ (index 0→7) |
|---|---|---|
| **UNMASKED interleaved** | **3.56±0.14** | 3.73 4.00 2.88 2.65 3.52 2.68 2.59 2.97 (≈flat) |
| MASKED replicas | 2.52±0.11 | **3.70** 2.33 1.75 2.01 1.99 1.82 1.99 2.18 (ladder) |
| external joint anchor (B1 N=8, same data/layer/px) | 1.97 | — |
| in-run "off−9 anchor" | 0.15–0.73 | **INVALID** — the interleaved chat template shifts the final room token off the off−9 position; do not use |

(L14: unmasked 2.72, masked 1.52 — same ordering, lower level, consistent with L14 being the
weaker carrier layer.)

**Readings.**
1. **The unmasked interleave is the finding: d′ 3.56 in one forward, zero training, zero
   architecture change** — +81% over the joint carrier, above the pre-registered GO bar (≥3)
   that the masked arm missed. Gate-law pricing at this supply (p=0.038): tally exact ≈ 0.75 /
   0.37 / 0.10 at N=8/32/128 vs 0.33/0.09/0.02 at joint supply — IF the level holds at larger N
   (untested; the decisive follow-up).
2. **The predicted per-copy contamination decay (≈6 at copy 0 → ≈2.3 at copy 7, from the
   chunk-size curve) did NOT appear** — the unmasked ladder is ≈flat at ~2.6–4.0. Best current
   account: question-conditioned frame encoding (frames attend earlier question copies, the
   Q-first effect) offsets the growing cross-frame contamination, at least to N=8.
3. **The masked arm underperformed its 2×2 prediction (3.3–4.0) at 2.52, and its per-copy ladder
   explains why:** the mask restricted REPLICA rows only, so frame encodings still attend all
   earlier frames — frame 0 is effectively isolation-encoded (copy 0 reads 3.70 ✓ the predicted
   band) while frame 7 is fully joint (2.18). The mask delivered clean queries everywhere but
   clean values only at frame 0. The arms dissociate the two ingredients: clean queries alone
   ≈ +28%; visibility (question conditioning) ≈ +81%.
4. **Both in-run anchors are mislocated** (off−9 convention breaks under the interleaved
   template); the external B1 anchor is the valid baseline. Fix for reruns: locate the final
   question's room token by word match like the replicas.

**Caveats.** Single seed, N=8 only, steps task only, n=300; per-copy d′ from n=300×1 frames
(±~0.2); masked-arm smoke n=8 prints nan (too small for held-out folds — expected); the two
arms share frames/questions so their difference is paired but no paired test was run;
behavioral (emitted-answer) effect of the interleaved prompt NOT measured — messages only.

**Next (registered):** (a) unmasked interleave at N=32/128 — does the flat ladder hold (the
gate-law prize at N=32 is 0.37 vs joint 0.09)? (b) replicas + frame-fencing combined (clean
queries AND clean values per frame → 2×2 predicts ~6, multipass-in-one-forward; the B1
fence-alone null does not preclude it — that null had a joint query); (c) behavioral EM with
the interleaved prompt; (d) gate→tally on the unmasked replica messages (supply 3.56 is above
the k=2 chunk supply 3.37 that retrieve-v2 uses — a ONE-forward shortlist candidate).

## [2026-07-17] ✅📊 ONE-FORWARD SUPPLY CLOSED — replicas + FULL block-diagonal fence + per-block M-RoPE reset reach the multipass band (d′ 6.34 = solo anchor 6.01) in ONE forward; Q-FIRST amplifies to 9.24; supply FLAT to N=128; gate→tally beats every N-forward system at every N

> **Motivation:** complete the [2026-07-15] fence PARTIAL (4.07). Two rungs: A2 = per-block
> position reset (PCW-style reuse — legal because fenced blocks are mutually invisible); A3 =
> seal the marker leak found analyzing A2 (`vision_start/end` tokens are neither visual nor
> replica tokens, so per-token-class fencing leaves them visible and their residuals carry
> earlier frames' content — worth +1.7 d′). A4 = the question ALSO in the shared prefix.
> **Script:** `replica_carrier_probe.py` flags `--reset-positions --fence-blocks
> --question-first --task cooc --natural`. **Runs (2026-07-17/18, jobs 122739–123137):**
> `outputs/ladder/image_longN/replica_{posreset,blockfence*,cooc_qfirst*,natural_*}/`,
> anchors `replica_posreset_N1anchor*/`. Draft detail: `plans/oneforward_DRAFT_RESULTS.md`.

| rung (N=8, L16, n=300, held-out LDA) | pooled d′ | per-copy |
|---|---|---|
| joint (external anchor) | 1.97 | — |
| unmasked replicas [2026-07-14] | 3.56 | ≈flat |
| + frame fence [2026-07-15] | 4.07 | ladder 3.8→2.0 |
| + position reset (A2, job 122744) | 4.66 | flatter, frame0 unchanged |
| **+ marker seal = block-diagonal (A3, job 122809)** | **6.34±0.11** | **flat 3.6–4.6** |
| solo anchor, same instrument (N=1, n=1200, job 122782) | 6.01±0.26 | — |
| **+ question-first (A4, job 122888)** | **9.24±0.33** | flat 5.7–7.6 |

**N-sweep (blockfence+posreset, standard layout):** d′ 7.62/7.81/7.55/7.24 at N=16/32/64/128 —
flat at the multipass band (mp 7.18–8.08); joint stays ~2.0. Q-first long-N: 12.67 @N=32,
11.57 @N=128. Cooc (`--task cooc`, relational predicate): 6.36/8.45/7.58 @N=8/32/128. Natural
photos (`--natural`, n=50/cell): dist_far 6.22 (in-run joint 3.12), dist_near 5.69 (3.61).

**Behavior (held-out logistic gate → tally, CPU on the message caches, 5 seeds):**

| N | steps 1-fwd | steps Q-first | cooc | retrieve-then-verify (~N fwd) | mp tally (N fwd) |
|---|---|---|---|---|---|
| 8 | 0.960 | 0.991 | 0.973 | — | 0.910 |
| 16 | 0.976 | — | — | — | 0.793 |
| 32 | 0.960 | 1.000 | 0.932 | 0.862 | 0.680 |
| 64 | 0.952 | — | — | 0.853 | 0.580 |
| 128 | 0.876 | 0.984 | 0.680 | 0.791 | 0.420 |

**Readings.** (1) Multipass is compiled into one forward by token topology + mask + positions —
zero training; the residual decomposes as replicas +1.6 / fence +0.5 / posreset +0.6 / marker
seal +1.7. (2) Q-first beats even isolated forwards: multipass encodes frames question-blind;
Q-first conditions the ENCODING — the first arm in the record to do so. (3) One forward + a
logistic gate beats the ~N-forward pipelines everywhere measured. (4) Attention is block-sparse
(O(N) in frames); no OOM at seq≈28k.

**Caveats.** AUC estimator pegs at 5.26 in the high-supply regimes (same caveat as all mp-band
numbers; E4 saturation signature — quote measured accuracies, not Φ(d′/2), there). **Count-prior
audit:** the N=8 caches contain gold∈{0,1,2} only (K-sorted dirs; majority 0.333) — N=8
exact-count rows are NOT comparable to the historical frozen 0.207 (full 0–8 prior) without the
same-samples frozen baseline (job 123205, pending); long-N caches have wide priors (majority
0.10–0.30) and stand as-is. Per-frame errors and d′ are prior-free. Gate trained per-N/task on
half the cache (~150 labeled samples). Cross-task gate transfer is partial (steps→cooc 0.460).

## [2026-07-17→18] ✅📊 LEARNED CARRIER TOKEN — ONE trainable embedding (3,584 params, frozen backbone) replaces the 20-token question replica at 92–93% of teacher; task-general (steps→cooc zero-shot 88% of ceiling) and length-general (N=8-trained → d′ 9.7 @N=128); random init converges to the same endpoint; the trained-query-at-L16 floor (0.4) beaten ~20×

> **Motivation:** distill the replica into a clean, task-agnostic, model-agnostic primitive; the
> controlled contrast with the [2026-07-13] trained-query NO-GO (same-size vector, input-with-
> context vs L16-without). **Script:** `experiments/glstm/carrier_token_distill.py` (placeholder
> `<|box_start|>` id 151648; Q-first blockfence+posreset; truncated differentiable forward to
> L16; objectives: proxy=BCE on evidence label, distill=cosine to in-run replica-teacher
> messages, label-free; in-run teacher anchor reproduces the probe exactly, 9.24). **Runs:**
> jobs 122938/39 (arms), 123124–26 (ablations), 123128/29 (length-gen), 123145 (cross-task) →
> `outputs/ladder/image_longN/carrier_token*/`.

| measurement | value |
|---|---|
| ep-0 warm start (UNTRAINED) | d′ 5.23 eval |
| proxy arm | 6.46 eval |
| **distill arm (label-free)** | **8.35 eval / ~9.0 full-n** (teacher eval-split 8.95) |
| random init / k=2 / k=4 | 8.25 / 8.14 / 8.38 — init irrelevant, k=1 suffices |
| gate→tally @N=8 (carrier messages) | **0.997±0.003** (per-frame err 0.0003) |
| zero-shot N=32 / N=128 (trained @N=8 only) | d′ 11.40 / 9.71; refit gate 1.000 / 0.988; FULL N=8-trained stack zero-shot **0.917 / 0.860** |
| zero-shot cooc (task never seen) | d′ 5.58 = 88% of cooc teacher 6.36; + per-task gate 0.880 |

**Reading:** per-source addressing must be computed in-context and CAN be learned into one input
token; task-conditioning arrives via attention to the question (label-free objective, cross-task
transfer). Token overhead 1/frame. **Caveats:** same count-prior note as above at N=8; both
tasks share the MMRED visual world (natural-image carrier transfer = job 123208 pending); the
per-task gate (3.6k params, ~150 samples) is the remaining task-specific piece.

## [2026-07-18] 📊 STAGE-2 "CARRIER LAYER" (all-in-model) — fenced extraction ≤L16 + cross-carrier attention ≥L17 + LoRA(r8, ~2M params) + plain LM loss: emitted answer 0.353→0.853 in 12 ep (undertrained, still climbing); the no-LoRA ablation plateaus at ~0.50 — cross-carrier attention alone canNOT aggregate, trained integration is required (the text-MMRED prediction, confirmed in-model)

> **Script:** `experiments/glstm/carrier_layer_lora.py` (mask schedule lo/hi; carriers get
> sequential positions; tail attends carriers ≥L17; hand-rolled zero-init LoRA on q/k/v/o of
> layers 17–27; model answers via its own lm_head — no gate/tally/render). **Runs:** jobs
> 123149 (LoRA, best 0.853 @ep12) / 123150 (no-LoRA, best 0.507) →
> `outputs/ladder/image_longN/carrier_layer/20260718_0230*/`; ep-0 = 0.353 ≈ the gold∈{0,1,2}
> majority (0.333). Extended 30-ep run job 123206 + same-samples frozen baseline job 123205
> pending. Scaffold ceiling on this prior: 0.991–0.997.

## [2026-07-15] ❌📊 ONE-FORWARD REPAIR ARMS (oneforward Exp B/C, never logged) — the offline encoding un-mixer does NOT transfer to the replica forward (3.56 → **1.44**, destructive), and a CoGNN-style content-side broadcast gate cannot repair routing (1.80 ≈ the 2.09 floor); the "qcond GO" is INVALID by q_pad feature leak (q_pad itself carries d′ 8.63)

- **B1** un-mixer retrained + weights SAVED (121919, `unmixer_saved/20260715_194450/`, weights
  `unmixer_saved/weights/unmixer_L16.pt`): offline mp-q × un-mixed-kv 5.94 vs joint-kv 3.82 /
  ceiling 6.33 = **84% of the encoding gap** (prior 93% ≈ retrain variance).
- **B2 deployed composition** (121928, `replica_unmix/20260715_204158/`; g_k/g_v hooked on
  frame-token k/v_proj @L16 in the unmasked replica forward): **3.56 → 1.44 @L16 — NO TRANSFER,
  actively destructive.** Frame 0 alone improves (3.73→4.17, in-distribution); L14 control
  unchanged (2.70 vs 2.72) — the replica layout's question-conditioned k/v are off the
  un-mixer's training distribution. (Moot after A3 blockfence: prevention beats repair.)
- **C broadcast gate** (121918, `broadcast_gate/20260715_194451/`; in-run anchors PASS 2.09/3.82):
  content arm ([k_j,v_j]) eval **1.80** (trajectory max 2.13 ≈ the 2.09 floor) — pre-registered
  "routing NOT repairable from content" outcome, closing addressing from a third direction
  (trained shared query NO-GO · content gate NO-GO · only architectural frame identity works).
  **qcond arm 30.69 = INVALID (do NOT cite):** q_pad is captured after the question attends the
  frame — the leak probe (121927, `broadcast_gate/qpad_leak/20260715_201750/`) reads q_pad alone
  at d′ **8.63** (q_mp 10.32) ≥ the mp×mp ceiling, while pooled content k/v-means carry ~0.65 —
  the gate broadcasts label information, and computing q_pad at inference = multipass anyway.
  The script's auto-VERDICT "GO" line is superseded by this reading.

## [2026-07-18] ⚠️📊 STAGE-2 AT 450 TRAIN SAMPLES — convergence hypothesis REFUTED (30-ep ref flat at 0.840 from ep12), full-prior steps 0.678 with the clamp DEAD, mixtures 0.693 (2-task) / 0.509 (3-task, rooms 0.50): a DATA-STARVATION ceiling, not architecture — sets up the pooled P1; cross-DOMAIN carrier transfer is only ~51% of teacher

- **30-ep truncated-prior ref** (123206, `carrier_layer/20260718_122503_L17_r8/`): BEST 0.840
  @ep12, loss→1e-4 by ep14, eval FLAT to ep30 — "undertrained, still climbing" ([2026-07-18]
  entry) is REFUTED; the gap is data/generalization, not optimization.
- **Full-prior steps-only 40ep** (123235, `carrier_layer/20260718_131157_L17_r8/`, train 450/eval
  450): **0.678 @ep30**, per-count uniform incl. g8 (clamp dead); train loss→0 = memorizes 450.
- **Mixtures** (warm-start distilled e_c, 30 ep): steps+cooc **0.693 @ep25** (cooc 0.796 / steps
  0.560 — no task interference; 123237, `carrier_layer_mixture/20260718_133821_L17_r8/`);
  +rooms 3-task 0.509 @ep12, rooms 0.50 vs frozen 0.087 / pipeline 0.993 — cross-carrier
  set-union PARTIALLY learned (123240, `carrier_layer_mixture3/20260718_134209_L17_r8/`).
- **Cross-DOMAIN carrier** (123208, `carrier_token_crosstask_natural/20260718_122538_proxy_room_k1/`,
  n=50 — wide bars): steps-distilled e_c on natural dist_far zero-shot d′ **3.19 ± 0.55** = ~51%
  of the cell's replica teacher (6.22); fresh-gate tally 0.432 UNDERPERFORMS the frozen model
  (0.58) — the carrier is partly domain-bound (vs 88% cross-task within synthetic MMRED).


## [2026-07-18] ✅📊 E1 FULL-PRIOR RECALIBRATION (N=8, LIMIT=900, gold uniform 0..8) — frozen baseline **0.219** (retires the truncated-prior 0.513); Q-first blockfence probe d′ 13.54 @n900 with gate→tally **0.998**; carrier-token distill 11.45 = 96% of teacher, carrier-stack tally 0.999 — the scaffold ceiling for all stage-2 comparisons

- **Audit caveat driving this entry:** the old N=8 caches were e-sorted → n=300 meant gold∈{0,1,2}
  only. Fix: stratified `iter_sample_dirs_shuffled` (+`--shuffle-dirs`, gold-hist prints) wired
  into probe/distill/trainer/baseline scripts; smokes 123222/123226 (`outputs/_scratch/st2_smoke*/`).
- **Frozen baseline** (123225, `frozen_baseline/20260718_125303/`): **0.219, MAE 1.86** — the
  undercount clamp in full view (g4+ ≈ 0). Truncated-prior 0.513 (123205) RETIRED (it rewarded
  the low-count bias).
- **Probe** (123232, `replica_blockfence_qfirst_full900/20260718_130546/`): **d′ 13.54 ± 0.27**
  @n900 (joint anchor 5.95; per-copy flat 8.4–9.2); matched-n300 subsample 10.50–10.89 vs the
  truncated band 9.24 ± 0.33 — supply is prior-free; d′ estimator scales with n (caveat for all
  cross-n comparisons). **Gate→tally (123236): exact 0.998 ± 0.001, MAE 0.00** vs majority 0.111.
- **Distill** (123233, `carrier_token/20260718_130545_distill_room_k1/`): in-run teacher anchor
  reproduces 13.54 exactly; **carrier eval d′ 11.45 @ep9 = 96% of the scale-matched teacher
  11.94**; carrier-stack fresh-logistic tally **0.999 ± 0.001** (CPU job 123243).


## [2026-07-19] ✅📊 C1/C2 ABLATION BATTERY (7 arms, 900-train starved regime, RANKING is the deliverable) — Q-first is the single most load-bearing piece (−46%); earlier opening ≫ (L12 0.941 vs L17 0.698 vs L22 0.513); posreset mild (−4%); LoRA rank flat

> Cached digit trainer, steps8+cooc (n=1800, train 900), 8 ep, frozen e_c, one change per arm
> (jobs 124300–06) → `outputs/ladder/image_longN/cached_ablations/{base,L12,L22,r4,r16,noqfirst,noposreset}/`.
> ABSOLUTE numbers are data-starved by design — never cross-compare to the 5–6k runs.

| arm | BEST acc | arm | BEST acc |
|---|---|---|---|
| **L_OPEN=12** | **0.941 @ep8** | rank=4 | 0.694 @ep7 |
| rank=16 | 0.731 @ep7 | no-posreset | 0.669 @ep8 |
| base (L17 r8) | 0.698 @ep8 | L_OPEN=22 | 0.513 @ep8 |
| | | **no-Q-first** | **0.378 @ep8** |

- Q-first −46% matches its +3 d′ supply price (and Track B: this piece is Qwen-specific);
  posreset −4% consistent with +0.6 d′. L12 ≫ L17 ≫ L22 — caveat: LoRA params scale with open
  depth (16/11/6 layers), depth and capacity confounded; at full data L17 already reaches
  0.98–0.99, so L12 is the better default for small data (later confirmed OOD — see E-H).


## [2026-07-19] ✅📊 TRACK B — InternVL2.5-8B PORT: solo-Q-first carrier d′ **6.31/5.11 @L16/L20 vs joint 1.79/1.90 — the supply mechanism ports (3.5×)**; but vs plain solo 6.38/6.56 the Q-first AMPLIFIER does NOT port — fence/isolation is the portable piece, Q-first is Qwen-specific

- Run: `outputs/frame_axis/internvl/multipass_qfirst/20260719_004112/` (job 124280; n=200, 1600
  solo passes, same seed/data/estimator as the 118996 record — sample-matched; `--qfirst` flag in
  `experiments/internvl/multipass_bench.py`; per-frame perception acc 0.586 unchanged).
- Verdicts vs pre-registered bands: mechanism-ports GO (≥2× joint ✓ at 3.5×); amplifier band
  (≥ +20% over plain solo) FAILED — 6.31 vs 6.38 flat at L16, 5.11 vs 6.56 negative at L20.
  Honest thesis scope note: question-conditioned frame encoding is family-dependent.
- Method note: solo forwards = the fence/multipass-equivalent supply measurement (fence ≡
  multipass identity established on Qwen); no mask surgery in InternVL remote code.


## [2026-07-19] ❌📊 DIGIT-READOUT EXAMS — length collapse and zero-shot task transfer are both robust NULLS: N=32 0.092–0.097 (collapse to "0" regardless of training-N diversity); five unseen-task pairs all ≈ chance (0.087–0.179); the LoRA left on plain prompts is SAFE (0.313 vs frozen 0.219)

- **Length:** cached ckpt (0.980 @N≤8) @N=32 → **0.097** (g0 24/24, else ~0)
  (`cached_eval_N32/20260719_000556_…/`, 124275); pooled ckpt (0.999, variable-N 2..8 trained)
  @N=32 → **0.092** (`pooled_eval_N32/20260719_051622_…/`, 124353) — training-N diversity buys
  nothing; the digit readout binds to the trained carrier-position range. Steps-450 ckpt: 0.138
  (`carrier_layer_eval_N32/20260718_182252_…/`, 123742).
- **Task transfer (zero-shot, all ≈ chance 0.111–0.125):** steps→cooc 0.179 (123743) ·
  cached→rooms 0.153 (124276) · A3→NIAH 0.087 (124316) · pooled→NIAH 0.117 (124354) ·
  pooled→union 0.150 (124355). Task coverage must be trained — and the mixture rows show it
  costs nothing (no interference).
- **Drift** (`frozen_baseline_driftlora/20260719_000556/`, 124277): plain prompt + LoRA hooks ON
  = **0.313 vs 0.219** frozen (n-mismatch caveat: 300 vs 900) — slightly HELPS; no gating needed.


## [2026-07-19] ✅ STAGE-2 POOLED GO — data was the whole gap: the in-model carrier layer at 6k pooled samples reads **0.999** (steps 630/630 · rooms 108/108 · cooc 161/162) = the 0.998 scaffold, one architecture, three tasks incl. the provably-nonlinear rooms set-union; frozen-e_c cached trainer hits 0.980 at 4× speed

- **P1 pooled 3-task** (job 123741, `carrier_layer_pooled/20260718_182248_L17_r8/`; steps N=2..8
  4200 + cooc 1080 + rooms 720, train 5100/eval 900 stratified per (task,N), trainable e_c
  warm-started from the full-prior distill, L17 r8, 12 ep): **BEST 0.999 @ep12, MAE 0.00.**
  Trajectory 0.176 → 0.669(ep1) → 0.963(ep5) → 0.997(ep8) → 0.763(ep11, transient optimizer
  blip) → 0.999(ep12).
- **P1-CACHED** (job 123937, `carrier_layer_cached/20260718_192428_L17_r8/`; e_c FROZEN, layers
  ≤L16 run once and cached, steps 2–8 + cooc, ~5.1k, 934 s/ep ≈ 4× faster): **0.980 @ep10**
  (steps 0.987 / cooc 0.946), MAE 0.02 — frozen e_c matches trainable; validated workhorse.
- **Data-starvation diagnosis vindicated: 450→0.678 · 5.1k→0.980 · 6k→0.999** (scaffold 0.998,
  frozen 0.219, chance 0.111). Caveat: all trained-on-clean synthetic MMRED; scaffold ceiling
  comparison is same-data.


## [2026-07-19] ✅📊 SCRATCHPAD READOUT (A3/A4/5-task) — verdict-scratchpad targets fit in ONE epoch (TF-count 1.000), in-dist greedy 0.953–0.966 with parse-fail 0.000; a 5-task mixture (adds NIAH 0.992 + OR-union 0.910) runs on ONE carrier + ONE LoRA; untrained OR-union composes PARTIALLY (0.321 = 2.1× digit); zero-shot length 0.215 → +200 in-length samples 0.447

- **A3 train** (job 124282, `carrier_layer_scratchpad/20260719_005342_L17_r8/`, 5.8k, jitter 12):
  TF-count **1.000 @ep1**; **in-dist greedy 0.953, pf 0.000, MAE 0.05** (124314; steps 0.980 /
  cooc 0.906 / rooms 0.850). **NIAH which-frame zero-shot 0.087** ≈ chance 0.125 (124316).
- **Composition (OR-union, never trained): 0.321** (pf 0.021, MAE 1.34; hits over the FULL count
  range — not mode collapse) vs digit-ckpt 0.150 → the shared VERDICT FORMAT is what transfers
  (NIAH, fully alien, gets 0.087). Run `scratchpad_eval_union0shot/20260719_030349_…/` (124335).
- **5-task mixture** (124336, `carrier_layer_scratchpad5/20260719_031356_L17_r8/`, +NIAH 720
  +union 540): TF-count 1.000 on all 5 @ep1; **in-dist greedy 0.966** (steps 0.997 · which 0.992 ·
  cooc 0.944 · union 0.910 · rooms 0.842) (124349). Bands NIAH ≥0.90 ✓ ("easy once in mixture"),
  union ≥0.85 ✓, no regression ✓.
- **Length:** A3 N=32 zero-shot **0.215** (in-range 0.311, pf 0.000 — format fully survives where
  digit ckpts collapse to "0") (124315) → A4 fallback triggered. **A4** (+longN_16 all 330 +
  longN_32 first-200; 124362, TF 0.997 @ep3) → **N=32 held-out complement 0.447 (in-range 0.626,
  pf 0.000, MAE 1.44)** (124376, `scratchpadLN_eval_N32heldout/20260719_092517_…/`) — band
  partial (<0.80); errors are verdict undercounts, never format collapse; long-N data curve
  still steep. (A3/A4 N=128 rows dropped for budget; the wall was later measured on the tally
  arms — see the N=128 entry.)


## [2026-07-19] ✅ E-D NO-HARM — the carrier-layer LoRA left permanently ON costs nothing on general benchmarks: MME −0.2 pts, POPE −1.4 pts (band ≤2 = GO); with the drift row (plain-MMRED 0.313 vs frozen 0.219) the adapter is deployment-safe always-on

- Run: `outputs/ladder/image_longN/noharm_bench/20260719_203833/` (job 124508; 500 MME + 500 POPE
  items, identical samples both arms, Yes/No logit-argmax, le16 running-tally ckpt LoRA).

| benchmark | base | LoRA-on | Δ | band |
|---|---|---|---|---|
| MME (acc) | 0.862 | 0.860 | **−0.2 pts** | GO (≤2) |
| POPE (acc / F1) | 0.862 / 0.839 | 0.848 / 0.819 | **−1.4 pts** | GO (≤2) |

- Per-subtask deltas ~0 across 12/14 MME cells (existence −4.5 / landmark −2.4 are small-cell
  noise n≈20–40, celebrity +2.7); all POPE splits −1.1..−1.7.


## [2026-07-19] ✅📊 E-C/E-C(b) LAYOUT FREEDOM (qualified GO) — carrier tokens can sit as a PURE SUFFIX after the question (d′ 10.27, tally **0.999** = the interleaved stack), but the strong form fails: with no leading question, at-end carriers read d′ 2.40 / tally 0.508 — the binding requirement is QUESTION-FIRST, not carrier placement

> E-C strong form (job 124492, `carrier_atend/20260719_192758_distill_room_k1/`, n=900,
> frames-first teacher anchor 8.89 ± 0.14): student d′ **2.40** (27% of teacher), tally 0.508 —
> band ≥5 missed; the at-end carrier reproduces messages in bulk (MSE converged) but not the
> discriminative direction, despite an IDENTICAL allowed-key set (mask-debug: 223 keys, same).
> E-C(b) restores ONE thing — the leading question (`--atend-qfirst`, job 124514,
> `carrier_atend_qfirst/20260719_205916_distill_room_k1/`, Q-first teacher anchor 13.70).

| layout (fence+posreset, distill, N=8 full prior) | eval d′ | fresh-logistic tally |
|---|---|---|
| Q-first, carriers INTERLEAVED (123233, reference) | 11.45 (96% of teacher) | 0.999 ± 0.001 |
| **Q-first, carriers AT END (124514)** | **10.27 @ep2 (75%)** | **0.999 ± 0.001** |
| no leading question, carriers at end (124492) | 2.40 (27%) | 0.508 ± 0.017 |

- Both E-C(b) bands met (d′ ≥5 ✓, tally = interleaved ✓): the E-C failure was missing
  question-conditioned frame ENCODING (the same +3 d′ Q-first term as the C2 ablation), adjacency
  innocent. Method statement: prompt = [question][frames][question][carrier suffix] — carriers
  never interrupt user content (the deployment-friendly form).


## [2026-07-20] ✅ E-E SEEDS — headline running-tally recipe at 3 seeds: in-dist TF-count **1.000 ± 0.000** (tf-exact 0.963 ± 0.007); the zero-shot N=32 length cell 0.284 ± 0.004 over 2 seeds — the recipe is seed-stable

- Runs: headline 124482 (seed/shuffle 0, jitter 16, 1.000 @ep2) ·
  `carrier_tally_le16_seed1/20260719_203924_L17_r8/` (124509, jitter 12, 1.000 @ep3) ·
  `carrier_tally_le16_seed2/20260719_203925_L17_r8/` (124510, jitter 12, 1.000 @ep3).
- Caveat: seed arms ran jitter 12 vs headline 16 (seed+jitter-dose bundled) — indistinguishable
  both in-dist and at the N=32 cell (0.287 vs 0.280, job 124697), so jitter dose 12-vs-16 is a
  no-op here.


## [2026-07-20] 📊 ROOMS DECODE-GAP DIAGNOSTIC — every error in 40/40 held-out transcripts is a MISSING-ROOM verdict; emitted count ALWAYS equals emitted list length: the readout's counting is exact, the residual is per-frame DETECTION RECALL (supply-side), not the readout

- Run: `outputs/ladder/image_longN/rooms_gap_diag/…_evalonly/` (job 124527; 40 held-out rooms
  samples, 5-task L17 ckpt, full transcripts): acc 0.825, parse-fail 0.000, MAE 0.17. No format
  derail, no reordering, no count-list mismatch (e.g. gold 6 → "Bathroom, Bedroom, Garden,
  Office, Park -> 5").
- Same signature as the long-N misses → the future lever is carrier content, not the readout.
- Follow-up caveat (2026-07-22): the l12v2 ckpt reads rooms-100 at **1.000** (job 125108) — the
  0.84–0.85 gap was a property of the L17 5-task ckpt family, not of the method.


## [2026-07-20] ✅📊 E-B SFT CONTROL — a 23.8M-param plain LoRA (12× our budget, q/k/v/o+MLP all layers, trained N≤8) matches in-distribution (0.998–1.000) but does NOT aggregate at length: N=16 **0.480** / N=32 **0.350** with a DEAD MID-RANGE — the theory's joint-supply ceiling located behaviorally

- First run 124484 (`sft_control_le8/20260719_185022_lora/`): test_iid **0.9984** at N≤8 (best ep1)
  — but the long-N cell was BLOCKED (script never saved the adapter; `plans/carrier_stage4_BLOCKED.md`).
  Rerun with adapter-saving patch: job 124696 → `sft_control_le8_v2/20260720_191541_lora/`
  (best ep3 val 0.983, test_iid 1.000).
- **N=16 0.480 / N=32 0.350 (pf 0.000 both)** — more nuanced than the pre-registered "collapse":
  the LoRA rides EXTREME-count anchors (g1 8/8, g32 7/7 but g6–g8 0/27 at N=16; mid-range ~0 at
  N=32; MAE 1.83) — a bimodal low-end/saturation heuristic. The dead mid-range is exactly where
  per-frame aggregation is needed (joint supply d′≈2); extremes are solvable from global gist.
- Honest packaging: the carrier method's long-N edge is the in-length-trained cell (0.733 vs
  0.350 @N=32) and uniform per-count coverage; the zero-shot tally cell (0.280) does NOT beat the
  SFT control. N=64 cell OOM'd during generate (adapter saved; low priority).


## [2026-07-20] ✅📊 POSRESET NECESSITY (Tal's challenge) — minor at N=8 (no-reset d′ 7.74 vs 9.24) but the per-copy position-tax gradient RETURNS at N=64 (pooled 7.54, per-copy decays 6.4→3.0): KEEP posreset, re-justified as N-scaling-critical; the historical "+0.59" justification retired

- Runs: `outputs/ladder/image_longN/noreset_N{8,64}/20260720_*/` (jobs 124713/124714) — Q-first
  blockfence WITHOUT `--reset-positions`; comparators with-reset 9.24 (N=8) and ~12 @N=64
  (bracketed 12.67@32 / 11.57@128).
- **N=8: 7.74 ± 0.18** — minor cost, and > fence-level 6.34 (Q-first partially substitutes for
  reset). **N=64: pooled 7.54 is carried by early frames — per-copy decays 6.4 → ~3.0** with frame
  index (the A2 position-tax fingerprint); extrapolated, late-frame supply at N=128 approaches
  joint level.
- Verdict (pre-registered intermediate band): keep — free at inference, increasingly load-bearing
  with N, and required infrastructure for jitter + position-coupling experiments.


## [2026-07-20→21] ⚠️📊 E-A LENGTH LADDER + THE N=128 WALL — running-tally readout decays smoothly zero-shot (1× 1.000 → 2× 0.280 → 4× 0.150 → **8× 0.087: headline band ≥0.80 REFUTED**); in-length training is the working lever (le64: N=32 **0.733**, N=64 0.558/0.821-in-range) — the honest claim is a LADDER in trained-length coverage, not extrapolation

> Arms: le16 (124482, `carrier_tally_le16/20260719_184950_L17_r8/`, 11 roots + longN_16,
> running-tally + jitter 16, TF-count 1.000 @ep2) · le64 (124483, `carrier_tally_le64/…/`,
> +longN_32/64, TF 1.000 @ep4). Exams: 124522/578/736 (le16), 124571/586 (le64), 124758 (v2→128).

| ckpt \ exam | N=32 | N=64 | N=128 |
|---|---|---|---|
| le16 (trained ≤16), zero-shot | 0.280 (pf 0.007; in-range 0.394) | 0.150 (pf 0.133) | **0.087** (first-match 0.130, pf 0.087) |
| le64 (trained ≤64), held-out | **0.733** (pf 0, MAE 0.43, g32 9/9) | 0.558 (in-range 0.821) | 0.118 (v2 ckpt, 124758) |

- **The headline cell** (`tally16_eval_N128/`, job 124736, TIMEOUT @23/34 — recovered from the
  --dump-decodes safety net, per-sample gold/parsed in `logs/cl_eval-124736.out`): train-≤16 →
  emit-at-128 (8×) does NOT hold. Transcripts show tally-index confusion (verdict indices leak
  into tally slots: "frames 30 (11), 31 (112)…") and unterminated chains. Trained-to-≤64 → 128
  (2×) also fails the ≥0.35 band (0.118; ckpt-selection confound caveat, gap to band large).
- le64's N=64 g48/g64 cells are cap-truncations (280-token decode < the 48–64-verdict tally) —
  annotated as unmeasurable-under-cap, not model failures. le16 N=64 failure mode = repetition
  loops. Parse sensitivity: first-vs-last-match differs ≤1 sample/cell (last-match stays primary).
- Zero-shot N=32 stable across seeds: **0.284 ± 0.004** (124522 headline jitter-16 0.280;
  124697 seed1 jitter-12 0.287). Successor arms (l12v2 0.953 in-length / 0.286@128, E-G refuted)
  logged above.


## [2026-07-22] ❌📊 E-G POSITION-COUPLED TALLY REFUTED — coupling decoded-token positions to carrier anchors costs in-distribution fit (persistent over 8 ep) and BREAKS the format at 4× length: N=32 0.527 / N=64 0.212 (pf 0.365) vs uncoupled l12v2 0.953 / 0.615 on IDENTICAL dirs

> `couple_offsets` in `carrier_layer_lora.py` + `--pos-couple` (one rule drives teacher-forced AND
> online decode positions; CPU-verified anchor rule + couple-debug — this is a MECHANISM negative,
> not an implementation bug). First train 124701 (`carrier_tally_pcouple/`) TF-undertrained at 4 ep
> (0.923/0.720, still climbing); converged rerun 124774 → `carrier_tally_pcouple8/20260721_071710_L17_r8/`,
> in-dist ceiling **0.955 / tf-exact 0.832 @ep5** (oscillatory) vs uncoupled 1.000/0.976 — ~2× the
> epochs for the same TF level. Exams 124922/23/24 → `tallyPC8_eval_N{32,64,128}heldout/`.

| cell (identical dirs) | E-G (coupled) | l12v2 (uncoupled best) | v2 (uncoupled same-L17) |
|---|---|---|---|
| in-dist TF / tf-exact | 0.955 / 0.832 | 1.000 / 0.976 | 1.000 / 0.916 |
| N=32 held-out | 0.527 (pf 0.040) | **0.953 (pf 0.000)** | 0.607 |
| N=64 held-out | 0.212 (pf 0.365, MAE 44) | **0.615 (pf 0.135)** | 0.365 |
| N=128 | PARTIAL 0.154 (4/26, pf 0.500) | PARTIAL 0.286 (pf 0.000) | 0.118 |

- **Pre-registered GO ("beats uncoupled at EVERY OOD length, pf ~0") refuted at every testable
  cell**; N=64 transcripts show token-salad degeneration ("28 (10 (10), 32 (11)…") — the opposite
  of the design goal (it was built against the tally-index confusion seen at 8×). Forcing decoded
  positions to ride carrier anchors degrades the LM's own sequential coherence more than it helps
  length binding. N=128 full cell (requeued 124924) cannot change the verdict.


## [2026-07-22] ✅📊 l12v2 — L12 + in-length data + FIXED ckpt criterion = the campaign-best in-model long-N readout: held-out **N=32 0.953 · N=48 0.878 cap-adj · N=64 0.678 cap-adj**; N=128 PARTIAL 0.286 with format fully intact (best 128 reading so far)

> Two measured levers composed: (1) **L12 depth** — le16-recipe L12 arm (124698,
> `carrier_tally_le16_L12/20260720_192738_L12_r8/`) reads N=32 zero-shot **0.443** (pf 0, MAE 1.15)
> vs L17's 0.280 (job 124727, `tallyL12_eval_N32/20260720_223517_L12_r8_evalonly/`; band ≥0.40 GO →
> L12 = new default); (2) **in-length data** (le64v2 roots + `mmred_longN_park2` N=32×312 + N=48×210).
> Trainer 124773 → `carrier_tally_l12v2/20260721_071710_L12_r8/`: TF 1.000 / tf-exact 0.976 @ep5,
> the strongest ckpt of the campaign. Exams 124904/05/06 → `tallyL12v2_eval_N{32,48,64}heldout/`.

- **N=32 held-out (150 dirs): 0.953, pf 0.000, MAE 0.05** — band ≥0.85 MET, thesis-grade; per-count
  near-uniform incl. multi-digit (g12 4/4 · g16 8/9 · g24 10/12 · g32 9/10). Trained-at-length
  progression: A4 scratchpad 0.447 → le64 tally 0.733 → **l12v2 0.953**.
- **N=48 (109): 0.789 raw / 0.878 cap-adjusted** (all pf = g48 cap truncations). **N=64 (52):
  0.615 raw / 0.678 cap-adjusted** (excl. g48/64, unmeasurable under dec 280) — met vs the v2 band
  ≥0.65 cap-adjusted, just under the stricter l12v2 prereg ≥0.70.
- **N=128 (2× beyond max trained): PARTIAL 0.286 (4/14), parse-fail 0.000** — job preempted by
  h200-dds mid-run, recovered from dumps (stratified-order prefix, unbiased); best N=128 of the
  campaign (le16 0.087 · v2 0.118) and the FIRST with intact format at 128; misses are undercounts
  at g≥5, not derails. Full-34 report = requeued job 124907.
- Caveat / confound exhibit: the earlier v2 arm (124682, old acc-only save criterion, ckpt ep3
  tf 0.891) read 0.607/0.514/0.365/0.118 at N=32/48/64/128 (124755/56/57/58) — kept as the
  ckpt-selection confound exhibit; trainer save criterion is now (TF-count, tf-exact) lexicographic.


## [2026-07-23] ✅📊 E-H SEPARATOR-LAYER (L*) CURVE COMPLETE — inverted-U confirmed, peak at L_OPEN=12: zero-shot N=32 0.277/0.373/**0.443**/0.330/0.280/0.273 for L*=8/10/12/14/17/20 — ~12 fenced supply layers suffice, every remaining layer is wanted for trained integration; L12 STANDS as the default

> Four new arms (ONLY `--l-open` varies), headline ≤16 running-tally recipe, fixed (acc, tf-exact)
> save criterion. Trainers: L8 124917 / L10 124918 / L14 124919 / L20 124920; zero-shot exams
> jobs 124965–972 → `outputs/ladder/image_longN/tallyL{8,10,14,20}_eval_N{32,64}/`. Reference
> cells: L12 (124698/124727), L17 (124482/124522) — recipe-matched, but predate the fixed save
> criterion (caveat logged).

| L* (open layer) | in-dist TF / tf-exact | N=32 zero-shot (n=300) | pf | N=64 zero-shot (n=60) | pf |
|---|---|---|---|---|---|
| 8 | 1.000 / 0.978 | 0.277 | 0.017 | 0.133 | 0.183 |
| 10 | 0.999 / 0.927 | 0.373 | 0.000 | 0.217 | 0.150 |
| **12** | 1.000 / 0.991 | **0.443** | 0.000 | (not measured, le16 recipe) | — |
| 14 | 1.000 / 0.997 | 0.330 | 0.000 | 0.217 | 0.150 |
| 17 (ref) | 1.000 / 0.977 | 0.280 | 0.007 | 0.150 | 0.133 |
| 20 | 1.000 / 0.996 | 0.273 | 0.107 | 0.183 | 0.300 |

- In-dist TF saturates for ALL L* (pre-registered: not the verdict metric). L20's parse-fail is
  the worst of any arm and concentrates at high counts (N=32: g24 7/23, g32 17/23 fails) — late
  opening leaves too few integration layers to keep the format coherent at long N.
- **Decision rule (LOTO/p0p2 briefs): no arm beats L12's 0.443 by >0.05 with pf ≤0.02 →
  L_OPEN=12 stands** for the seed retrains and LOTO. L12's N=64 cell and 2 extra seeds ON HOLD
  per Tal's no-new-launches instruction.


## [2026-07-24] ✅📊 FORMAT SWEEP COMPLETE (arms A–D) — the gold scratchpad TEXT alone is worth +0.19–0.37 at held-out lengths: full-scan formats crush the positive-list (B scan N=32 **1.000**, N=48 **0.982**; C caption N=64 **0.981**, pf 0.000 everywhere); WINNER = C (caption); chunking NO

> Prereg `plans/scratchpad_format_PREREG.md` (bands fixed pre-GPU); arms differ ONLY in gold
> scratchpad text (`--scratchpad-format`), l12v2 recipe/data/split verbatim — eval/train dirs-files
> byte-identical across arms. Trainers 125104/05/06 →
> `outputs/ladder/image_longN/carrier_fmt_{scan,caption,chunked}/20260722_2220*_L12_r8/`; exams
> 125107/08 + 125183–99 on arm A's dirs-files (`fmt{B,C,D}_eval_*` + `tallyL12v2_eval_*` siblings).

| cell (identical dirs) | A poslist | B scan | C caption (WINNER) | D chunked |
|---|---|---|---|---|
| TF-fit (acc / tf-exact) | 1.000 / 0.976 | 0.999 / 0.996 | 0.999 / 0.994 | 1.000 / 0.904 |
| in-dist-150 | 1.000 | 1.000 | 1.000 | 0.987 |
| rooms-100 | 1.000 | 1.000 | 1.000 | 0.920 |
| N=32 held-out (150) | 0.953 | **1.000** (125194) | 0.987 (125195) | 0.907 (125185) |
| N=48 held-out (109) | 0.789 / 0.878 cap-adj | **0.982** (125196) | 0.972 (125197) | 0.679 (125186) |
| N=64 held-out (52) | 0.615 / 0.711 cap-adj | 0.942 / 0.956 cap-adj (125198) | **0.981** (125199) | 0.615 (125187) |
| worst parse-fail | 0.135 (N=64) | 0.019 | **0.000** | 0.000 |

- **All five prereg bands decided:** in-dist sanity ≥0.90 MET ×4 · **scan GO** (B N=64 cap-adj 0.956
  ≥ 0.761) · **agnostic-caption GO** (C≡B: in-dist/rooms exact parity, N=32 −0.013, N=64 +0.039 in
  C's favor) · **chunking NO** (D 0.615 < max(A,B)) · rooms-ordering parity-only (control A also
  1.000 — the 0.84 gap was the L17 5-task ckpt, not reproducible here).
- **Winner = C (caption)**, ckpt `carrier_fmt_caption/20260722_222032_L12_r8/carrier_layer_best.pt`:
  takes the hardest cell (0.981 vs B 0.942) with pf 0 at every length; B within noise overall
  (length-cell mean 0.975 vs 0.980) — C chosen on the primary N=64 cell + parse robustness + the
  agnosticism property (attribute words in the scratchpad). Mechanism reading: a slot per frame
  converts the long-N search burden into a deterministic frame-order scan; D's subtotals kill
  truncation (dec-means 55/75/101, pf 0) but cost accuracy at every length.
- Cost note: scan-family decode ≈ 2.4× poslist in-dist (~47 vs ~20 tok); N=48/64 exams ≈ 18h each
  on l40s (~21 min/sample @dec620) — a100 ≈ 2× faster.


## [2026-07-23] ✅📊 P1.2 — MEASURED BEFORE-CEILING: the best sample-disjoint linear readout of the summed joint-carrier messages lands within 0.01–0.05 of the zero-parameter law prediction at every N — the "squashed readout" curve is now measured, not just predicted

> Job 125259 (CPU, `probe_dprime_parity.py --carrier-caches`, deployed locus L16/off9, existing joint
> caches `image_longN/joint/N{8,16,32,64,128}/20260710_2154*/count/`, 60/40 sample-disjoint split,
> seeds 0–2) → `outputs/ladder/image_longN/measured_ceiling/20260723_222428/`.

| N | law-pred (iid) | MEASURED best linear | frozen model (same caches) |
|---|---|---|---|
| 8 | 0.307 | **0.317** (ridge, ±0.036) | 0.207 |
| 16 | 0.246 | **0.281** (logit) | 0.127 |
| 32 | 0.175 | **0.189** (logit) | 0.053 |
| 64 | 0.137 | **0.183** (logit) | 0.040 |
| 128 | 0.096 | **0.122** (logit) | 0.013 |

- Slightly ABOVE the law at N≥64 — the logistic readout exploits the non-Gaussian tail the law ignores
  (adequacy kurtosis +1.8→+14.0, as in [2026-07-11e/n]); MLP−linear ≤0.006 (E3 sufficiency); d′_w flat
  ~2.0 to N=64 (1.6 @128), replicating B1. Thesis reading: frozen < measured-linear < trained scaffold
  (0.95–1.00 @N=32) — readout misalignment vs supply repair, both gaps now measured.
- Fig: `outputs/_scratch/figs/pre_stage1_squashed_readout_measured.png`. Caveat: gold≥1 convention at
  N≥2; n = 300/300/300/200/150 → wider bars at N=128 (±0.031).


## [2026-07-23] ✅📊 P1.3 — E-B SFT BASELINE N=64 CELL (the OOM'd leg, rerun): 0.220 — the bimodal extremes heuristic extends to 8× training length; the SFT ladder is complete at N=16 0.480 / N=32 0.350 / N=64 0.220 vs the carrier readout's 0.953/0.878/0.678

> Job 125267 → `outputs/ladder/image_longN/sft_control_le8_v2_evalN64/20260723_225940_lora/`;
> eval-only restore of `sft_control_le8_v2/20260720_191541_lora/adapter`, LIMIT 100, same
> stratified-prefix sampling as the N=16/32 cells. Adapter-restore sanity: test_iid 1.0000 = the
> original run byte-exact.

- **N=64: 0.220, parse-fail 0.000, MAE 3.46**; per-count g0 4/7 · g1 5/7 · g64 6/6, mid-range
  g4–g48 ≈ 0 — no collapse, no aggregation, the dead-mid-range supply ceiling unchanged at 8×.
- Ops lesson (new): the 124696 OOM = mask-None generate → `enable_gqa=True` → mem-efficient sdpa
  ineligible (num_heads mismatch on dense inputs) → MATH materializes 17GB. **FLASH handles
  GQA+causal at 8.3 GiB peak @seq 12.7k** (smoke 125263). `lora_sft_baseline.py` eval path now
  [FLASH, EFFICIENT, MATH] + `--eval-only-adapter`.


## [2026-07-24] ✅📊 P2b — MLVU-AC ZERO-SHOT CARRIER CELL (32-frame arm): MCQ nearest-option 0.107 ≤ frozen 0.282 — the pre-registered "domain gap measured" outcome; the readout FORMAT transfers, the trained evidence detector does not fire outside the MMRED render domain

> Job 125350 → `outputs/ladder/mlvu_ac/carrier_eval_N32/20260724_064754_L12_r8_evalonly/` (+
> `mcq_mapping.txt`); winner ckpt (C caption) on all 206 MLVU-AC questions, 32f @392px. Dense N=128
> ruled prohibitive pre-launch (~20 min/sample × 206 > 60h) → prereg 32f fallback with the
> evidence-delivery caveat ([2026-07-11c]: ~0.37 visible frames per gold instance at N=32).

- **Open emitted count 0.000 exact** (pf 0.214, MAE 2.99); 161/206 emit "0", 44 parse-fail, 1×"1".
  **MCQ nearest-option (prereg rule, parse-fail = wrong): 22/206 = 0.107** (g1 12/37 · g2 8/52 ·
  g3 2/45 · g4 0/33 · g5 0/39). Band: ≤ frozen 0.282 → **transfer NO, domain gap measured**.
  Protocol note (by design): this cell never sees the MCQ options; the frozen 0.282 had them
  in-prompt — chance structure differs.
- Failure anatomy (206 dumped transcripts): scan/caption structure emitted and mostly well-formed;
  content collapses to all-negative verdicts, and the parse-fails are fluent refusals that correctly
  DESCRIBE the frames ("the image shows a zebra … no 'making jewelry' action"). Perception works;
  the trained evidence-detection channel is domain-bound — consistent with the cross-domain carrier
  result (~51% of teacher, [2026-07-18]) and the delivered-evidence ceiling.


# Pre-registered predictions — many-needle aggregation at the read node (written 2026-09-02, before any run)

Purpose: the BABILong program measured how the read node's aggregation degrades as the
HAYSTACK grows (degree axis = dispersion, distance axis = horizon). This program holds the
haystack fixed, small or absent and grows the number K of NEEDLES that must be aggregated
into one answer (a count). It asks what breaks when the evidence itself is many-fold, which
is the over-squashing question proper: a fixed-width read node summarising a growing set.

Instrument: `scripts/condmask/needles.py` (written 2026-09-02; reuses `eval_babilong.build/decode`,
`osq.osq_probe` and the model loader; `--analyze` runs the offline analysis). CPU-tested
(`tests/test_needles.py`, tiny Llama + Qwen tokenizer) before any GPU run.
Models: Qwen2.5-3B (probe layer L27), Qwen2.5-7B (L22), Llama-3.1-8B (L20) — the BABILong
selection layers, so numbers are comparable across the two programs; plus the final layer for
the read-node state. Frozen, bf16, no training anywhere.

## Design

Story = one sentence per line, bAbI style ("Mary went to the kitchen."). K NEEDLES = sentences
matching the (person, room) of the question; DISTRACTORS = sentences about other (person, room)
pairs including the same person elsewhere and other people in the same room. Question:
"How many times did Mary go to the kitchen?" with a two-example prompt asking for the number in
digits. Gold = K. Needle verbs rotate over 6 bAbI verbs; order is a random interleaving; n = 50
stories per cell, seed 0.

K ∈ {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64}. Three haystack regimes:

- **H0** — no distractors (context = the K needles). Mass on needles ≡ 1; the answer token's own
  position grows with K (a positional counter is available).
- **HF** — 64 distractors, fixed. Mass on needles is the only content channel that grows with
  K; position grows with K as well.
- **HC** — 64 sentences in total (64 − K distractors, so K ≤ 48 here). Position is constant;
  mass is the only channel. The cleanest regime.

Per story and regime, at the read node (the answer row): emission (greedy decode ≤ 4 tokens →
first integer; exact match, |error|), the logit margin between K and its neighbours K±1,
attention metrics at the probe layer (`mass`, `keff_sent`, `lse_e/lse_j`, the per-needle mass
vector), the Jacobian of the gold logit w.r.t. the probe layer's input on each needle
(per-needle norm, sum, share), the read-node hidden state at the probe layer and the final layer
(stored, fp16), and the mean pairwise cosine between the K needle-block states at the probe
layer. Analysis is offline (`needles.py --analyze`): per-K means with bootstrap CIs; linear
decodability of K from the stored states by ridge regression trained on half the stories and
tested on the other half (exact match after rounding, and R²); resolution
d′(K) = |μ_K − μ_{K+1}| / pooled sd along the μ_K→μ_{K+1} direction.

## Theory that dictates the predictions

Softmax attention is a MEAN aggregator: the read node receives Σ_i α_i v_i with Σ α_i = 1.
A mean aggregator cannot distinguish multisets that differ only in multiplicity (Xu et al.
2019, GIN, Fig. 3 — mean vs sum). So if the K needle values are near-identical (v_i ≈ v),
the read of the needle set is ≈ v regardless of K, and the ONLY K-dependent channel is the
share of mass the needles win against the junk: m(K) = K e^g / (K e^g + J) (the dispersion
law with the roles reversed). Its resolution per added needle is dm/dK = m(1 − m)/K, which
vanishes both as m → 1 (few distractors) and as K grows. Over-squashing enters as the
sensitivity bound: the gold logit's Jacobian on any one needle scales like α_i ∝ 1/K
(Di Giovanni et al. 2023's degree factor), so no single needle can move the answer once K
is large. A transformer can escape the mean-aggregator limit only through position (H0/HF)
or through non-uniform, content-dependent reads (e.g. attending to the LAST needle whose
representation may carry a running count — Barbero et al. 2024's counting analysis).

## Predictions (numeric; falsifiers named)

**A1 — dose-response exists without a haystack.** Exact-match accuracy falls with K in every
regime: for all three models, acc(K ≤ 4) ≥ .90 and acc(K = 32) ≤ .50 in HF and HC, and
acc(K = 32) ≤ .70 in H0 (position helps but does not rescue). FALSIFIED if any model keeps
acc ≥ .80 at K = 32 in HC (then aggregation of 32 identical-role facts is not a bottleneck
for a 7–8B frozen model and the BABILong failures are haystack effects only).

**A2 — the mass channel follows the reversed dispersion law.** In HF, needle mass m(K) is
increasing and concave in K; fitting the single parameter g at K = 1 predicts m(K) for all
other K within ±.05 (mean absolute error over K). FALSIFIED if the MAE exceeds .10 — then the
read is not a fixed-margin softmax over the needle set (e.g. the model attends to a few
needles only, which A2' tests).

**A2' — the read is uniform over needles.** In HF and HC, the per-needle mass vector has
effective size k_eff,needles = 1/Σ p_i² ≥ .7 K for K ≤ 16 (p = normalised needle masses).
FALSIFIED if k_eff,needles ≤ .3 K at K ≤ 16 — a sparse read (a few needles get most of the
mass), which would mean the read node is NOT a mean aggregator here and the GIN argument
does not apply as stated.

**A3 — resolution predicts emission.** Across K in HF and HC, the K-vs-(K±1) logit margin at
the read node correlates with the mass resolution m(1 − m)/K: Spearman ≥ .7 over the 12 (HF)
/ 11 (HC) K-levels, per model. FALSIFIED if Spearman ≤ .3 in both regimes.

**A4 — sensitivity is shared out, not added.** The mean per-needle Jacobian norm of the gold
logit falls with K with log–log slope in [−1.3, −0.7] (∝ 1/K) in HF and HC, while the SUM
over needles stays within a factor 2 across K ∈ [1, 32]. FALSIFIED if the per-needle slope is
≥ −0.3 (the read node keeps full sensitivity to each of many needles — no squashing).

**A5 — the count is decodable but not emitted (the readout fork).** In HC, a ridge probe on the
FINAL-layer read-node state decodes K with exact match ≥ .80 for K ≤ 16 on all three models,
while emission exact match at K = 16 is ≤ .60 on at least two of them. Anchor: the MMRED text
dose-response found a 0.78-decodable count at the answer position with frozen emission at
.13–.23 (see the Anchors section). FALSIFIED if probe accuracy tracks emission within ±.10 at
every K — then the information is lost at the read itself and a readout fix cannot help.

**A6 — representational resolution collapses with K.** d′(K, K+1) of the final-layer read-node
state falls with K (log–log slope ≤ −0.5 in HC) and tracks per-K emission accuracy
(Spearman ≥ .7 over K). FALSIFIED if d′ is flat (slope ≥ −0.2) while emission falls — then the
failure is not in the read node's state but downstream of it (the unembedding), which also
points at the readout, but for a different reason than A5.

**A7 — the needle set is over-smoothed at the probe layer.** Mean pairwise cosine between the
K needle-block mean states at the probe layer is ≥ .90 for every K ≥ 4 in all regimes (the
premise of the mean-aggregator argument: near-identical values). FALSIFIED if ≤ .70 — needles
carry individuating information (e.g. an in-context ordinal), and the model could in principle
count by content.

**A8 — the "ballast" prediction (non-monotone in the haystack).** At K = 8 and K = 16, the HF
regime (64 distractors) is NOT worse than H0 by more than .10 exact match for any model —
distractors give the mass channel something to be measured against. FALSIFIED if
acc(H0) − acc(HF) ≥ .20 at both K (then the haystack only ever hurts and the mass channel is
not what the model counts with).

### Added 2026-09-02, after the CPU dry run and before any GPU run of the grid

Two regimes and two instruments were added while the instrument was being written; the
predictions above are untouched.

- **H32** — 32 sentences in total (D = 32 − K, K ≤ 16): the base stories for DUP.
- **DUP** — the H32 story with K/2 needles concatenated with itself (gold K; even K ≤ 32).
  This is the GIN mean-vs-sum test transposed to the read node (Xu et al. 2019, Fig. 3: a
  mean aggregator cannot tell {a, b} from {a, a, b, b}). A read node that counts through the
  mass fraction, or through collapsed needle representations, sees S‖S exactly as S.
- **Layer scan** — the read-node state after every layer (ridge decodability of K per layer)
  and the needle cosine per layer: where the count is linearly present, where it disappears,
  and where the needles smooth out.

**A9 — duplication blindness sets in where counting fails.** For K ≥ 8 in DUP, on all three
models the mean emitted count divided by K is ≤ .75 while the un-duplicated base story
H32(K/2) is answered correctly (acc ≥ .80 for K/2 ≤ 4): the second copy of the evidence is
mostly not counted. FALSIFIED if acc(DUP, K) ≥ .80 at both K = 8 and K = 16 — then duplicated
evidence is counted as distinct, the counter is positional/ordinal rather than a mean read,
and the method must NOT destroy absolute position (fencing's per-block position reset would
be harmful for counting; the carriers would have to carry ordinals instead).

**A10 — the count lives in the middle and is lost at the top (layer scan).** In HC, ridge
exact match of K from the read-node state, as a function of layer, peaks at a layer in the
middle third of the stack with exact ≥ .80 (K ≤ 16) and is lower at the final normed state
by ≥ .15 on at least two models. FALSIFIED if decodability is monotone non-decreasing to the
last layer (then nothing is lost downstream of the read and A5's fork resolves toward the
aggregation side).

### Added 2026-09-02, after the literature sweep (`lit_D_aggregation.md`) and before any GPU run of the grid

Positioning that the sweep forces, recorded before the data so it cannot be shaped by them:

- *Needle Threading* (Roberts et al., ICLR 2025, 2411.05000) finds that the NUMBER of needles
  matters much less than context length — for a LISTING readout on frontier models, where each
  emitted needle is its own read with one target. Our claim is scoped to one-forward AGGREGATION
  into a single read node (a count), where the K targets share one softmax budget. The mass
  identity predicts exactly this split (listing easy, counting hard) on the same contexts.
- *Hasani et al.* (CVPR 2026, 2511.17699; ACL 2026, 2601.02989) report the closest existing
  accuracy-vs-K curve (K = 10…50 identical adjacent items, no haystack, Qwen2.5-7B / Llama-3-8B:
  collapse beyond ≈ 30 items) and explain it by a RIVAL MECHANISM: a depth-limited running counter
  carried in the item tokens and read from the LAST item through transfer heads (layers 19–23 of
  Qwen2.5-7B — our 7B probe layer L22 sits inside that band). This is the account named in
  "Theory" as the model's escape from the mean aggregator; A11 below is its direct test. The
  regime comparison at fixed K (H0 / H32 / HC / HF = J ∈ {0, 32−K, 64−K, 64}) is the K × J
  factorial: a running counter predicts a J-independent ceiling, dispersion predicts a ceiling
  that moves with J through K e^g / (K e^g + J).
- Nobody has measured attention mass, k_eff or per-needle Jacobian share as a function of K at
  fixed n in a frozen LLM, and no count-decodable-but-not-emitted result exists for tallies
  (nearest: Orgad 2410.02707, Biran 2406.12775). Rediscoveries we cite, not claim: softmax
  dispersion Θ(1/n) (Veličković 2410.01104), the mass bound 1/(1 + J e^{−Δ}) (Bansal
  2512.13898), counting as 1/c inversion (Yehudai 2407.15160), collapse of identical-token
  contexts under finite precision (Barbero 2406.04267 Cor 6.2), one virtual node ⇒ uniform
  sensitivity (Southern 2405.13526 Prop 4.1).
- Tool verdicts for the analysis (128k, frozen decoder): attention row / k_eff trivial; one-backward
  Jacobian feasible (what `osq_probe` does); attention rollout streamable but only sensible
  ≤ 16k; effective resistance and Forman curvature degenerate to (weighted) degree on the complete
  causal graph — cite, do not compute. Alon–Yahav capacity is trivially satisfied
  (log₂(K+1) bits ≪ d), so ours is a normalization/precision bottleneck, not a capacity one.
- Stored per-layer states allow any later probe; a helix probe (linear + Fourier features, Levy &
  Geva 2410.11781, Kantamneni 2502.00873) is a possible post-hoc addition and will be labelled so.

**A11 — the read is a mean over the needles, not a read of the last one.** In HC and HF, for
K ≥ 8 on all three models, the last needle's share of the needle attention mass and of the needle
Jacobian, multiplied by K, lies in [0.5, 2] (≈ uniform). FALSIFIED if last-needle share × K ≥ 4
for both mass and Jacobian at K ≥ 8 on ≥ 2 models — then the read node reads a running counter
off the last item (Hasani), the dispersion account is wrong for counting, and the method should
protect the item-token counter path (chunked partial counts, Hasani's System-2) rather than the
softmax budget. Instrument: `mass_last_share`, `jac_last_share` in `needle_metrics`
(story-order last needle), added before any GPU run.

## What each outcome buys the method

- A5 holds (decodable, not emitted) → a readout repair (linear head / small digit-LoRA on the
  read node, or the thesis' scratchpad tally) is the right fix; judge any such fix against the
  digit-LoRA baseline, not against the frozen model.
- A5 fails but A4/A6 hold (information squashed at the read) → the fix must be on the
  aggregation side: a hierarchical tally (aggregate within windows of ≤ K* needles, where
  emission is still ≥ .9, then combine — turning one mean read into a sum of means; the
  virtual-node/hierarchical-pooling construction), which is what fencing + carriers already do
  structurally.
- A2' fails (sparse read) → the model counts by attending to a few needles; the dispersion
  framing must be dropped for counting and replaced by an ordinal/positional account.

## Anchors (existing logged numbers this design must reproduce or be reconciled with)

Filled 2026-09-02 from the repo audit, before the first GPU run. All are frozen-model,
MMRED-text or MMRED-vision numbers; none is a many-needle count at fixed haystack, which is
why this program exists.

1. **Fan-in capacity law** (`outputs/superquery/STATE.md:114-122`): superquery accuracy vs
   fan-in — fan-2 ≈ .98, fan-4 .87–.92, fan-8 .63–.67, fan-16 .444, fan-32 .208, fan-64 .117
   ("counts to ~4, halves per doubling"). A1 must reproduce this shape: acc(K) roughly halving
   per doubling beyond K ≈ 4 in HC. If needles.py finds a much slower decay, the superquery
   number was haystack-limited, not aggregation-limited.
2. **Frozen text bit-counting EM** (`outputs/superquery/STATE.md:170-176`, forced prefix
   "Answer: ( "): .892 / .550 / .325 / .317 / .233 / .183 at N = 4 / 8 / 16 / 32 / 64 / 128
   frames. Same halving shape; the plateau at .23–.32 for N ≥ 16 is the per-N majority floor,
   so A1's ≤ .50 at K = 32 is conservative.
3. **×64 dose (decodable vs emitted)** (`docs/archive/RESULTS_pre_fencing.md:1794-1806`,
   `legacy/dprime_dose_response.py`): emitted .133 vs decode-at-L24 .750 vs repaired readout
   .783 — the anchor behind A5's "decodable ≥ .80 while emitted ≤ .60".
4. **Decodable-but-not-emitted at matched N** (`outputs/gating/STATE.md:700-720`): emitted
   ≈ chance while a ridge probe on the answer position reaches R² .58–.70 — A5's ridge
   R² should land in or above this band for K ≤ 16 in HC.
5. **probe_ft_effects frozen row** (`CONDMASK_REPORT.md:698-711`): keff@N=32/128/256/1024 =
   31/99/159/179; mass@128 .17; EM@128 .12; ft32@128 vs ft1024@128 have the same fan-in 16
   and mass .45–.49 but EM .27 vs .75 — i.e. the same attention statistics can hide very
   different emission, which is what A5/A6 separate (state resolution vs readout).
6. **Representational-collapse metric** (`docs/aggregation_lit_sweep_2026-08-06.md:147`):
   planned in the lit sweep, never implemented; A6/A7 are its first implementation here.

## Cost

Per model: (12 + 12 + 11 + 8 + 8) cells × 50 stories = 2,550 stories, each ≤ 1.6k tokens:
decode prefill + probe forward/backward + 3 candidate passes + one hidden-states forward
≈ 1–1.5 s on a B200 → ~1 GPU-hour per model. Grid = 3 models, one job each (`slurm/needles.sbatch`).

## Outcomes (dated; predictions above are never edited)

- **2026-09-02 — grid launched (user OK).** Code smoke 416307 (`gpu-short`, Qwen2.5-3B, N=5,
  K ∈ {1, 8, 32}, `outputs/_scratch/needles/`, a code check, not a study run). Grid jobs
  `slurm/needles.sbatch`, `--dependency=afterok:416307`: 416427 Qwen2.5-3B L27, 416428 Qwen2.5-7B
  L22, 416429 Llama-3.1-8B L20; roots `outputs/needles/<model>/<stamp>_<pid>/` (filled in when
  the jobs start). Defaults: n = 50, all K, regimes H0+HF+HC+H32+DUP, seed 0. Predictions
  A1–A11 as written above; no threshold has been touched since.
- **2026-09-02 — grid landed.** Roots: 416427 Qwen2.5-3B `outputs/needles/Qwen2.5-3B-Instruct/
  20260902_172402_2295342`, 416428 Qwen2.5-7B `outputs/needles/Qwen2.5-7B-Instruct/20260902_173017_2301190`,
  416429 Llama-3.1-8B `outputs/needles/Llama-3.1-8B-Instruct/20260902_173055_2302310`; n=50 per
  (regime, K), 2,600 rows each; `needles.py --analyze <root>`. Distractors verified to exclude the
  queried (person, room) pair (`story()`); gold K is exact.
  Emission (exact match) by K = 1 2 3 4 6 8 12 16 24 32 48 64 —
  Q3 H0 .84 1.00 1.00 .92 .44 .50 .38 .04 .02 0 0 0 · HF .02 .28 .12 .64 .26 .10 .04 0 0 0 0 0;
  Q7 H0 1.00 1.00 .98 .86 .02 .06 0 0 0 0 0 0 · HF 0 .20 .42 .58 0 0 0 0 0 0 0 0;
  LL H0 1.00 1.00 1.00 .92 .44 .68 .54 .02 0 .02 0 0 · HF .22 .08 .10 .06 0 .12 .50 0 .26 .06 0 0;
  HC ≈ HF throughout. What is emitted: WITHOUT distractors the models count exactly to K = 4 and
  then under-count (Q7 emits 4 for K = 6 in 43/50 stories, 8 for K = 16, 14 for K = 64; Q3 and LL
  emit ≈ .6–.7 K at K ≥ 32). WITH distractors they OVER-count at small K (K = 1: Q7 emits 2–4 in
  49/50, LL 1–5; K = 2: Qwen emits 4 in half the stories) and the emitted count is nearly flat in K
  up to K = 8 (Q7: 3–5), then compressed (Q7 HF K = 32 → 8, K = 64 → 14; LL HF K = 8 → 12, K = 32 →
  34). The K = 4 accuracy peak in HF/HC (.58–.66 on Qwen) is the coincidence of the gold with the
  models' default answer, not counting.
  * **A1 — mixed.** acc(K = 32) ≤ .10 in every regime (falsifier does not fire); acc(K ≤ 4) ≥ .90
    holds in H0 only and FAILS in HF / HC / H32 at K = 1–3 (.00–.48) through over-counting. The
    anchor shape (counts to ~4, then degrades) reproduces in H0 on all three; the 7B has a cliff,
    not a halving (.86 → .02 between K = 4 and 6).
  * **A2 — falsified on Q3 (MAE .148 > .10); unsupported on Q7 (.064) and LL (.092)** (support
    needs ≤ .05). The shape holds (m(K) increasing, concave; g(K=1) = 2.30 / 1.75 / 1.22) but the
    single-g law over-predicts the mass from K ≥ 8 (Q3 K = 32: .83 predicted vs .62 measured; LL .74
    vs .65) — the needles compete with each other more than a fixed margin allows.
  * **A2′ — near-support.** keff_needles / K at K ≤ 16 is .64–1.00 (HF) and .65–1.00 (HC); the ≥ .7
    line is missed narrowly at K = 12–16 on Q7 HF (.69 / .64) and LL HF/HC (.62–.69); the falsifier
    (≤ .3) is far from firing. The read is a slightly uneven mean over the needles.
  * **A3 — FALSIFIED, with reversed sign.** Spearman(resolution m(1−m)/K, margin) = −.43 / −.39
    (Q3 HF/HC), −.62 / −.62 (Q7), −.71 / −.45 (LL). Where the mass resolution is highest (small K)
    the emitted answer is the wrong prior (margin −4.2 nats at K = 1 on Q3 HF).
  * **A4 — FALSIFIED on Q7 and LL, unsupported on Q3.** Per-needle Jacobian log–log slope (K ≤ 32)
    HF / HC: Q3 −.52 / −.41, Q7 −.33 / −.28, LL −.04 / +.04 (the ≥ −.3 falsifier fires on Q7 HC and
    on both Llama cells; the [−1.3, −.7] band is met nowhere). H0 slopes −.59 / −.30 / −.04: even
    without distractors the per-needle sensitivity does not fall as 1/K. The SUM over needles is not
    bounded: ×5–10 (Q3), ×9–17 (Q7), ×21–35 (LL) across K ∈ [1, 64] against the predicted factor 2.
    Sensitivity is ADDED, not shared out — the read node keeps (near-)full sensitivity to each of
    many needles. Post-hoc: with J distractors the identity gives per-needle mass ∝ 1/(K + J e^{−g}),
    i.e. slopes of −.27 (LL, g = 1.2) … −.47 (Q3, g = 2.3) over this K range, which the Qwen
    Jacobians match and Llama undershoots; the pre-registered band assumed J e^{−g} ≪ K.
  * **A5 — support fails.** Ridge on the final read-node state (HC) decodes K with exact .18 / .19 /
    .20 (chance .08), within-1 .48 / .52 / .54, R² .98–.99; emission at K = 16 is 0 on all three. The
    named falsifier (probe tracks emission within ±.10 at every K) does not fire (K = 4: probe
    .12–.24 vs emission .58–.66). Reading: a coarse magnitude of K (±2) is linearly present at the
    read node; an exact count is not.
  * **A6 — FALSIFIED in the haystack regimes (falsifier fires); holds in H0.** d′(K, K+) RISES with
    K under distractors: HC slopes +.47 (Q3), +.43 (Q7), +.26 (LL), Spearman(d′, acc) −.65 / −.49 /
    −.06, while emission is 0; in H0 the prediction holds (slopes −.48 / −.54 / −.52, Spearman .90 /
    .79 / .77). With distractors the read-node state separates neighbouring K better the larger K is
    and the emission still fails → the failure is downstream of the read-node state.
  * **A7 — near-support.** Min pairwise needle cosine over K ≥ 4 is .84–.92 in every regime (≥ .90
    met only in HF: .901 / .922 / .900; the ≤ .70 falsifier nowhere).
  * **A8 — mixed / vacuous.** acc(H0) − acc(HF) at K = 8 = .40 / .06 / .56 (Q3 / Q7 / LL); at K = 16
    = .04 / 0 / .02 (both arms at the floor). The falsifier (≥ .20 at both K) does not fire only
    because of the K = 16 floor; there is no evidence for the ballast effect — the haystack costs
    40–56pp at K = 8 on Q3 and Llama.
  * **A9 — premise fails.** H32(K/2) is not answered correctly (acc at K = 2: .28 / .28 / .30; K = 4:
    .50 / .66 / .10), so duplication blindness cannot be read off. Descriptively, mean pred/K at
    K = 8 / 16 / 32 = .78 / .63 / .52 (Q3), .55 / .40 / .25 (Q7), 1.19 / 1.06 / .65 (LL); the
    falsifier (acc(DUP) ≥ .80 at K = 8 and 16) does not fire (0–.22).
  * **A10 — falsified in substance.** In HC the per-layer ridge exact never exceeds .24 / .26 / .27
    (peaks at L25–27 of 36, L21 of 28, L17 of 32) and ends at .18 / .19 / .20 — no layer holds the
    exact count (support needs ≥ .80); the literal falsifier (monotone non-decreasing to the top) does
    not fire. CAVEAT (instrument): in H0 and HF the early layers decode K at .8–1.0 (LL L1–13, Q7
    L1–16, Q3 L1–21) — there the total length grows with K, so the read node's position leaks K; HC
    (64 sentences at every K) is the only length-controlled regime and there the number is .20–.27.
  * **A11 — SUPPORTED.** Last-needle share × K for K ≥ 8 in HF / HC is .7–1.7 (mass) and .6–1.8
    (Jacobian) on all three models (only Q3 at K = 48–64 reaches 2.0 / 2.9); the ≥ 4 falsifier is
    nowhere near. The read node reads a near-uniform mean over the needles, not a running counter
    off the last item (Hasani). In H0 the Jacobian tilts to the last needle ×1.7–2.8 at K = 4–16 on
    Qwen, still inside the band.
  **Net (dated).** The mean-aggregator premise holds (A2′, A7, A11) but its squashing consequences do
  not: sensitivity is not shared out (A4), the state's K-resolution grows rather than collapses under
  distractors (A6), and no exact count is linearly present at any layer in the length-controlled
  regime (A5, A10) — only a ±2 magnitude. Emission fails by a K-independent prior with distractors
  (over-counting at K ≤ 3) and by saturation at 4 without. For counting, the data point at a
  precision/readout failure of a mean read, not at a sensitivity (over-squashing) failure of the
  read node. Per "What each outcome buys": A2′ holds → keep the mean-read framing; A4/A6 fail →
  the aggregation-side repair is not indicated; A5 fails in exact form → any readout repair must be
  judged against the digit-LoRA baseline and cannot claim the count was "there" — only its magnitude.
- **2026-09-02 — post-hoc correction to the A4 / A6 reading (labelled post-hoc; the verdicts above
  stand as written).** (1) A6's instrument computes d′ between ADJACENT GRID POINTS (1→2, 2→3, 3→4,
  4→6, 6→8, 8→12, 12→16, 16→24, 24→32, 32→48, 48→64), not K vs K+1 as the prediction is phrased; the
  spacing grows with K, so a rising raw d′ is expected even while resolution collapses. Per unit ΔK
  (HC, K = 1…32; the 48→64 pair excluded because HC at K = 64 has D = 0, a different regime):
  Q7 1.01 .72 .63 .50 .53 .52 .25 .24 .17 .19 · Q3 .88 .70 .61 .51 .47 .50 .31 .32 .21 .12 ·
  LL 1.34 1.17 .87 .71 .59 .46 .37 .25 .19 .14. Resolution per added needle falls ~7× from K = 1
  to 32 on all three — the direction A6 predicted. (2) A4's band [−1.3, −.7] assumed the needles'
  share is far from saturation. With the share m(K) = K e^g / (K e^g + J), the per-needle
  sensitivity through the share is ∝ e^g J / (K e^g + J)²: flat for K e^g ≪ J, ∝ 1/K² beyond, knee at
  K* = J e^{−g} ≈ 6 (Q3), 11 (Q7), 19 (LL) for J = 64; the summed sensitivity should peak near K*.
  Observed sums (HF, relative to K = 1): Q7 rises to 9.4 at K = 16 then 8.6–13 flat; LL rises to
  25.6 at K = 32 then 25–28 flat; Q3 peaks at 6.2 (K = 24). Direction matches; magnitude does not:
  per-needle sensitivity at K = 64 is 1/5 (Q7), 1/2.3 (LL), 1/14 (Q3) of its K = 1 value against
  1/30, 1/17, 1/90 predicted by the single-softmax share alone — 3–8× more sensitivity to each
  needle reaches the answer than the one-head share path carries (other layers / paths). (3) Net
  reading, post-hoc: the mean read compresses cardinality into one saturating share, and the answer
  token's residual stream never holds more than a ±2 magnitude at any layer (A5 / A10) — consistent
  with squashing AT the softmax normalization, not downstream of it; the "readout failure" phrasing
  in the 2026-09-02 net above is withdrawn. Still open: the over-counting with distractors (a pure
  share reading predicts under-counting) and A8's failure (distractors should have improved
  resolution near K*; they did not).

### Added 2026-09-03, before the run — the normalizer channel ("log Z probe", Stage 0 of the set-read plan)

Motivation (from the landed data): the read is a mean over near-identical needles, so cardinality
survives the head only as the needles' share of one softmax. The softmax DENOMINATOR
Z_h = Σ_j exp(s_hj) is computed by every head and then divided away; for K needles of margin g over
the rest J, log Z_h ≈ log(K·e^g + J). If the count is recoverable from {log Z_h} where it is not
recoverable from the residual-stream state, an adapter that exposes the normalizers is the method.
Instrument: `outputs/_scratch/dbg/needles_logz.py` — same stories as needles.py (seeded per K, so
identical across models/arms), regimes H0 / HF / HC, K ∈ {1…64}, n = 50; for every layer and head,
the last row's log Z over the context columns (and over all columns), the needle and junk
log-sum-exps and the needle share; three rows per story: (A) the real question, (B) a CONTROL
question in the same prompt about a (person, room) pair absent from the story (its Z is junk only,
same context, same prefix), (C) the COUNTERFACTUAL context with the needles removed (distractors
only; two forwards, diagnostic only). Read-node residual states at the probe layer, six layers
above and the final norm are stored for the baseline probe. Analysis: ridge regression on log K,
5-fold CV, prediction rounded to an integer; "exact" and "within-1".

**N1 — the count is in the normalizer (H0).** A ridge probe on the per-head log Z_ctx at the read
layer (Q3 L27, Q7 L22, LL L20; 16–28 features) decodes K with exact ≥ .80 for K ≤ 16 and within-1
≥ .80 over all K ≤ 64, on all three models, and beats the residual-state probe at the same layer
by ≥ .30 exact. FALSIFIED if the log-Z probe's exact ≤ residual-probe exact + .10 on ≥ 2 models —
then the normalizer is as blurred as the state and the un-normalized channel is not the leap.

**N2 — the junk term can be cancelled without knowing the needles (HF, HC).** The raw log Z probe
degrades with 64 distractors (J unknown, query-dependent); the control-row difference
log Z(A) − log Z(B), or the subset of heads whose needle share exceeds .8 at K ≥ 12 (selected inside
each training fold), restores exact ≥ .60 for K ≤ 16 on all three models. FALSIFIED if neither
variant exceeds the residual probe by .10 on ≥ 2 models — then the channel exists only when the
needles already dominate the head, and the adapter would need the carriers to make them dominate.

**N3 — the channel is log-linear, not saturating.** For the three most needle-selective heads at
the read layer, mean log Z_ctx regressed on log K has slope in [0.8, 1.2] in H0 and ≥ 0.6 in HF
(the head-mean lse − max seen so far has slope ≈ 0.5). FALSIFIED if the slope is < 0.5 in every
regime on ≥ 2 models — the normalizer would then compress like the share and add nothing.

Post-hoc (allowed, labelled): all-layer features vs the read layer alone; the counterfactual
difference (C) as the ceiling for (B); which layers carry the cleanest count.
- 2026-09-03 — **analysis addition before the full runs report (labelled): length extrapolation at the
  probe level.** Every feature set is also fit on K ≤ 16 only and tested on K ∈ {24, 32, 48, 64}
  (exact, within-1, mean relative error, median K̂ at K = 64). Prediction (added, not part of N1–N3):
  the log Z features extrapolate (relative error ≤ .25 at K = 64) and the residual-state features do
  not (K̂ saturates near the training range) — the probe-level form of "a sum computed is a sum at
  any N".
- 2026-09-03 — **log Z probe landed: N1, N2 FALSIFIED; N3 mostly falsified; extrapolation addition
  fails.** Roots `outputs/needles_logz/{Qwen2.5-3B,Qwen2.5-7B,Llama-3.1-8B}-Instruct/20260903_1235*`
  (jobs 418298 / 418299 / 418300, smoke 418297; n = 50 per K, 600 stories per regime; ANALYSIS.md in
  each root). N1 (H0): ridge on the read layer's per-head log Z_ctx decodes K with exact .30 / .31 /
  .45 (Q3 / Q7 / LL; exact≤16 .43 / .44 / .62) — BELOW the residual-state probe at the same layer
  (.69 / .70 / .67; exact≤16 .89 / .92 / .85). Falsifier fires on all three. All-layer log Z reaches
  exact .83 / .84 / .85 in H0 and HF but only .12–.21 in HC — the length-controlled regime — so the
  in-distribution success is the sequence-length leak (the log-linear normalizers sit in layers 0–1:
  L0H14 slope .95 R² 1.00 on Q7, L0H22 slope 1.02 R² 1.00 on Llama — token counters, not evidence
  counters). N2 (HC / HF): every normalizer variant is at or below the state probe (raw .13–.23,
  control-row difference .11–.17, counterfactual .14–.20, selective-head subset .09–.13 vs state
  .22–.35); falsifier fires on all three. N3: at the read layer the three most needle-selective heads
  have slope d log Z / d log K = .77 / .55 (Q3 HC / HF), .61 / .35 (Q7), .39 / .38 (LL); across all
  (layer, head) cells only 2–8 of 576–1024 are log-linear (slope > .8, R² > .9) with distractors, and
  those are noisy (within-K sd .4–1.1 against a range of 2–7). Extrapolation (fit K ≤ 16, test 24–64):
  no feature set extrapolates; median K̂ at K = 64 is 31–41 for all-layer log Z vs 17–21 for the state
  (relative error .23–.37 vs .45–.53) — better, not log-linear. Head-mean log Z_ctx at the read layer
  grows by only ≈ .5·log K (Q7, LL) or FALLS (Q3, −.21·log K) in H0: **the per-needle scores adapt to
  K** — the network suppresses each needle's logit as more needles are present, so Z ≠ K·e^g + J. The
  context share of the row rises only mildly with K (Q7 .04 → .21, LL .01 → .05; the sink holds the
  rest). Reading: the count is not sitting in the denominators; the assumption of K-independent
  per-needle scores was wrong. The normalizer adapter (Stage 1) is NOT pursued. Positive residue: a
  cheap, pre-registered negative that rules out the simplest "un-normalize" fix and shows an upstream
  regulation of attention logits with K.

### Added 2026-09-03 (after the log Z falsification, before the run) — two diagnostic bundles

Bundle A (`outputs/_scratch/dbg/needles_diag.py`): where in the sequence is the count, and what
moves the scores. Bundle B (`outputs/_scratch/dbg/needles_causal.py`): causal tracing that does not
presuppose the mean-read account. Same stories as needles.py (seeded per K). HC is the
length-controlled regime and carries the verdicts; H0 is shown alongside.

**A1 — running count in the items?** Ridge probes on per-token states (layers L/2, L, L+6, final):
needle token → its rank among the needles so far; distractor token → needles before it; last
needle → K; a tail token and the answer token → K. Mean-read account predicts rank exact ≤ .40 for
ranks ≤ 16 at every layer (the items do not count); a running-counter account (Hasani) predicts
≥ .80. FALSIFIER of ours: rank exact ≥ .80 at some layer on ≥ 2 models.
**A2 — individuation.** Sentences prefixed with a unique clock time ("At 09:41, …") or numbered.
Prediction: numbering in H0 is read as an explicit ordinal (exact ≥ .80 at K ≤ 32: control);
time stamps change emission by < .10 at every K in HC (a frozen model has no decoder for a new
norm channel). Exploratory: the answer-state probe under each variant.
**A3 — precision.** Same stories, fp32 weights and activations. Prediction: exact changes by < .10
at every K; FALSIFIED (precision is part of the cap) if the largest K with exact ≥ .80 in H0
doubles.
**A4 — what adapts.** At the read layer, per head and per K: |q|, mean |k| and mean cos(q, k) over
needle tokens and over junk tokens. Prediction: the per-needle score falls by ≥ 1 nat from K = 1
to 64 in H0 and the drop sits in cos (the query rotates), not in the norms. Exploratory.
**B1 — minimal-pair patching (HC, K ∈ {2, 4, 6, 8}, n = 24 pairs).** K vs K+1 stories that differ in
ONE sentence at the same position with the same token count. Effect of patching the K+1 run's
states into the K run at layer ℓ for a token group = (Δ digit log-odds)/(full effect). Groups: the
flipped sentence, the other needles, the needles after the flip, the distractors, the tail, the
answer row. Mean-read account predicts the effect travels in the flipped sentence's own tokens
(patch 'flip' at ℓ ≤ L ≥ .5 of the full effect; 'later needles' and 'tail' < .2). Running-counter
account predicts ≥ .5 through 'later needles' or 'tail'. FALSIFIER of ours: 'later needles' or
'tail' ≥ .5 while 'flip' < .3 on ≥ 2 models.
**B2 — share clamp at the read layer (HC and H0, K ≤ 32).** At layer L, for the answer row only,
the needle share of every head is reset to that head's mean share at K = 4 (or amplified ×1.5),
then the count is emitted. Prediction: under the clamp the median emitted count changes by ≤ 1
between K = 4 and K = 16 (the share IS the channel); amplifying raises it. FALSIFIED if the median
emitted count at K = 16 is ≥ 8 under the clamp on ≥ 2 models — the count reaches the answer by
another path.
**B3 — direct logit attribution per head** for the (K+1 − K) digit direction on the minimal pairs:
which heads carry the difference. Exploratory (a map, no threshold).
- 2026-09-03 — **Bundles A and B landed** (`outputs/needles_diag/<model>/20260903_13*`, jobs
  418373 / 418375 / 418377, n = 40; `outputs/needles_causal/<model>/20260903_1329*`, jobs 418374 /
  418376 / 418378, 96 minimal pairs per model; ANALYSIS.md in each root).
  * **B2 — FALSIFIED on all three models.** Clamping every head's needle share at the read layer's
    answer row to its K = 4 value leaves the emitted count unchanged at every K in H0 and HC (Q7 H0
    median 1 2 4 4 5 7 8 10 12 vs clamp 1 2 4 4 5 7 8 10 14; Q3 HC 2 4 4 6 6 10 11 14 18 identical;
    LL identical), and amplifying ×1.5 changes nothing either. The needle share at layer L is
    causally irrelevant to the count. Our probe layer was chosen for the answer row's selection
    recall; it is not where the count is read.
  * **B1 — the count is read by the QUESTION tokens in a mid-depth band, then handed to the answer
    row; the items carry no running count.** Patching the flipped sentence's own tokens carries the
    full K→K+1 effect through the early and middle layers (≈ 1.0 at L0–4, .7 by mid-depth); the
    effect transfers to the tail (question) tokens in a band — Q3 L18–24 (peak .71 at L23–24), Q7
    L15–18 (.70 at L17), LL L11–15 (.71 at L14–15) — and then to the answer row (Q3 .62 at L25 → ≈ 1.0
    from L28; Q7 .60 at L19 → 1.0 by L24; LL .65 at L16 → .83 from L18). Later needles carry ≤ .37
    at any layer (LL up to .56–.62 in two K cells), distractors ≤ .33, needles before the flip
    exactly 0 (causality check). The band lies BELOW the probe layer on every model (L27 / L22 / L20).
    Falsifier of the mean-read account ('later needles' or 'tail' ≥ .5 while 'flip' < .3): the tail
    reaches .7 while flip is ≤ .15 only above the band, i.e. after the hand-off — the effect is not
    a running counter but a relocation of the read. Digit log-odds K+1 vs K move by only +.8 / +1.1
    / +.3 nats (Q3 / Q7 / LL) for one more needle at K ≤ 8 with distractors; predA = K in .32 / .29
    / .09 of pairs.
  * **B3** (exploratory): per-head direct-logit attributions are tiny (|·| ≤ .12) and confined to the
    last two layers; the digit is decided by MLPs / the residual, not by a head's direct write.
  * **A1 — running counter rejected in the length-controlled regime.** HC needle → rank exact
    .25–.35 (ranks ≤ 16: .29–.41; within-1 .59–.70), distractor → needles-before .17–.21, last needle
    → K .10–.25, tail → K .11–.18, answer → K .15–.33, at every depth on all three models (Hasani's
    account predicts ≥ .80). In H0 all of these are .7–.97 — the position leak, since rank = position
    there. Our falsifier does not fire.
  * **A2 — individuation changes nothing systematic.** Time stamps move HC emission by −.32…+.35 in
    single cells (n = 40) with no consistent sign (mean change < .05). Numbering as an explicit
    ordinal is used only partially: Q3 H0 .88 at K = 24 but .38 at K = 32, LL .68 / .42; Q7 ignores
    it (0 from K = 6). The pre-registered control (numbered H0 exact ≥ .80 at K ≤ 32) fails on all
    three.
  * **A3 — precision is not the cap.** fp32 emission equals bf16 within noise at every K on all three
    models (largest cell difference .11). Prediction holds.
  * **A4 — the adaptation is in the angle, and much of it is a global shift.** |q| flat (Q3 31.8 →
    31.3; Q7 17.5 → 17.8; LL 12.0 → 12.4); |k_needle| flat or rising (Q7 19.6 → 22.7; LL 22.6 →
    25.4); cos(q, k_needle) falls (Q3 .20 → .13, Q7 −.14 → −.18, LL −.20 → −.25), so the mean needle
    score falls by 7.3 / 1.9 / 2.1 nats in H0 from K = 1 to 32. In HC the junk scores fall in
    parallel (Q3 −12.8 → −15.6): a softmax-invariant global shift; the needle−junk margin moves only
    2.5 → 1.65 (Q3), 1.63 → 1.54 (Q7), .95 → .48 (LL). This global drift is why log Z was never a
    count (the shift-invariance caveat).
  **Net (dated).** The aggregation of "one more needle" happens over a band of layers, by the
  question tokens and then the answer row, several layers below the layer we had been probing; the
  answer row's attention share at that layer is causally inert. Whether the share at the band is the
  channel is untested → bundle C below. Precision, individuation and a running counter are excluded.

### Added 2026-09-03, before the run — bundle C: the read at the band (`outputs/_scratch/dbg/needles_band.py`)

B1 located the hand-off of the K→K+1 signal from the items to the question tokens in a band of
layers (Q3 L18–24, Q7 L15–18, LL L11–15). Bundle C asks whether the attention share of the TAIL rows
(question tokens + answer row) at those layers is the channel, and which heads carry it. HC regime.
**C1 — the read relocates.** At the band layers the tail rows' needle share is concave-saturating in
K and its fan-in over needles is near-uniform (k_eff ≥ .6 K), as the answer row's was at L.
Exploratory map: share per (layer, tail row) by K.
**C2 — share clamp at the band and everywhere.** Clamping the needle share of every tail row at the
band layers to its K = 4 reference flattens the emitted count (median change ≤ 1 between K = 4 and
K = 16); clamping at ALL layers flattens it fully (median at K = 16 ≤ 6). FALSIFIER: with the clamp
at all layers the median emitted count at K = 16 is still ≥ 8 on ≥ 2 models — then the count is not
carried by attention weights on the needles anywhere, and the mean-read account is wrong as a
mechanism, not just misplaced.
**C3 — counting heads.** Patching single heads' outputs at the tail rows (K+1 run into the K run) at
the band layers: the five strongest heads jointly carry ≥ .5 of the K→K+1 effect on each model
(few heads aggregate). Exploratory if they do not.
- 2026-09-03 — **Bundle C landed** (`outputs/needles_band/<model>/20260903_1348*`, jobs 418400 / 418401 /
  418402 after smoke 418399; n = 30 per K, 12 minimal pairs per K for C3; ANALYSIS.md in each root).
  * **C2 — SUPPORTED (Qwen fully, Llama partially); falsifier does not fire.** With the needle share of
    every tail row reset to its K = 4 reference at ALL layers, the emitted count stops tracking K:
    Qwen-3B median 4 4 4 6 6 6 6 6 5 (K = 1…32; unclamped 2 4 4 6 6 10 11 14 18), Qwen-7B 3 3 4 4 4 4 4 5 4
    (unclamped 2 3 4 4 5 5 7 8 8), Llama 4 5 7 8 9 12 12 12 12 (unclamped 2 3 7 8 12 12 17 24 32). The
    K = 1 and K = 2 cells RISE to 3–4: the model reads the clamped share as "about four". The clamp at
    the band layers alone flattens only partly (Qwen-3B K = 32: 18 → 11; Llama 32 → 16), so the channel
    is spread over layers beyond the hand-off band. Amplifying the band shares ×1.5 raises the count
    (Llama K = 16: 17 → 24, K = 24: 24 → 34; Qwen-7B K = 12: 5 → 7). The attention share on the needles
    at the question/answer rows is the causal channel of the count — distributed over layers, absent
    from the answer row's single-layer read (B2).
  * **C1 — the share is small and near-LINEAR in K, not saturating.** Head-mean needle share of the
    tail rows (mean over all layers): Qwen-7B .006 .010 .019 .027 .034 .048 .058 .078 .097 for K = 1 … 32
    (±.002–.005 across stories); Qwen-3B .005 → .087; Llama .006 → .067. Mildly concave (K 1 → 2 doubles
    it, K 16 → 32 multiplies by 1.6–1.7). Fan-in over needles at the band, answer row: k_eff/K = .83 /
    .75 / .69 / .67 (Qwen-7B, K = 4 / 8 / 16 / 32), .75 / .62 / .53 / .43 (Qwen-3B), .73 / .60 / .49 /
    .46 (Llama). The ≥ .6 K prediction holds for Qwen-7B at every K and for the others to K ≈ 8–12.
    The saturating-share explanation of the K ≈ 4–8 cliff is therefore NOT what the data show: the
    channel keeps growing (Qwen-7B share ×2.9 from K = 8 to 32 while the emitted count goes 5 → 8).
  * **C3 — NOT supported: no small set of counting heads.** Strongest single heads carry .21–.28 of
    the K→K+1 effect; each pair's top-5 heads jointly carry ≤ .55 (Qwen-3B L23), .27 (Qwen-7B), .44
    (Llama); per-layer sums of head effects reach 3.5 on Llama (redundant, non-additive). The
    aggregation is distributed over many heads at every band layer.
  * **Signal-to-noise of the channel (post-hoc, from sharemap.pt).** d′ of the tail-row share between
    adjacent grid values of K is 1.1–4.4 (Qwen-7B: 2.1, 3.0, 1.9, 1.3, 2.8, 2.1, 4.4, 4.4 for 1→2 …
    24→32); per added needle it falls from ≈ 2 at K = 1 to ≈ .5 at K ≥ 12 (Qwen-7B) and ≈ .2–.3
    (Qwen-3B, Llama). Within a fixed K, the story-to-story variation of the head-mean share does not
    predict the emitted number (Spearman −.5 … +.4, no consistent sign), while pooled over K both
    track K (ρ .91–.99). Reading: the share is resolvable per needle well beyond K = 8 by an ideal
    reader; what the frozen models lack is a share → integer map beyond ≈ 4–8. The bottleneck is the
    DECODING of a small, growing, distributed share into a number — not the saturation of the share
    and not the normalization of a single softmax.
  **Net (dated).** Causal account after bundles A–C: "one more needle" changes the flipped sentence's
  representation (early layers), the question tokens and then the answer row pick it up through
  their attention share on the needles, spread over many layers and heads (band hand-off, then
  accumulation); fixing that share fixes the answer, amplifying it raises the count. The share is
  small, near-linear and resolvable in K; the emitted count compresses beyond 4–8 because the model's
  map from share to number does. Precision, item-level running counters, single-layer read, single
  heads, and the softmax normalizer as a hidden count are all excluded.

### Added 2026-09-03, before the run — bundle D: label-free pattern statistics as the count channel (`outputs/_scratch/dbg/needles_pattern.py`)

Bundle C showed the count is carried by the needles' attention share at the question/answer rows,
distributed over layers and heads, small and near-linear in K. The share needs needle labels; the
attention PATTERN does not. For every head at the band layers (and the probe layer), tail rows: the
per-sentence weights (summed over each sentence's tokens, renormalized over the context), and from
them the participation ratio k_eff = 1/Σ p_s² (the read node's effective degree), exp-entropy, and
peak counts #{s : p_s > τ/F} for τ ∈ {2, 4, 8}. Features are means over the last 8 tail rows and the
answer row. HC (fixed length) carries the verdict; HF (length grows with K) is the robustness check.
Baselines: the answer-row residual state at the same layers, and the ORACLE needle share (labelled).
**D1 — a linear reader of pattern statistics counts.** Ridge on the label-free statistics decodes K
in HC with exact ≥ .80 for K ≤ 16 and within-1 ≥ .80 over K ≤ 48, beating the residual-state probe by
≥ .30 exact, on ≥ 2 models. FALSIFIER: exact ≤ state probe + .10.
**D2 — it extrapolates.** Fit on K ≤ 16, test on K ∈ {24, 32, 48}: mean relative error ≤ .20 and the
median K̂ at K = 48 ≥ 36 (the state probe saturates near the training range). FALSIFIER: relative
error ≥ .40.
**D3 — a single head's degree is a count.** At least one (layer, head) with k_eff_sent linear in K:
slope ∈ [.6, 1.1], R² ≥ .95 (means over stories), on ≥ 2 models; its 2-parameter calibration fit on
K ≤ 16 extrapolates to K = 48 within 25 %. Exploratory otherwise.
- 2026-09-03 — **Bundle D landed: D1, D2, D3 FALSIFIED on all three models.** Roots
  `outputs/needles_pattern_v2/<model>/20260903_14*` (jobs 418481 / 418482 / 418483 after smoke 418480;
  a first pass, `outputs/needles_pattern/`, jobs 418473–418475, had an overflow in k_eff for heads
  with no weight on the context; the corrected pass agrees on every unaffected row). n = 40 per K,
  K ∈ {1 … 48}, HC (verdict) and HF (check).
  * D1: in HC the ridge on label-free pattern statistics decodes K with exact .12 / .18 / .23 (all
    stats, band layers; Q3 / Q7 / LL), .23 / .20 / .25 (k_eff only), .07 / .13 / .15 (peak counts) —
    none above the residual-state probe (.24 / .21 / .26 at L; .27 / .22 / .28 final). Falsifier fires
    on all three.
  * D2: fit K ≤ 16 → test 24–48: relative error .29–.43 (all stats), .60–.92 (k_eff only); the state
    probe .30–.31. Nothing extrapolates; falsifier (≥ .40) fires for k_eff and peaks, and the
    all-stats sets do not beat the state. In HF the pattern features "extrapolate" to K̂ = 500–1000 —
    they read the growing sentence count, not the needles.
  * D3: no (layer, head) has k_eff over sentences linear in K in HC (best slopes .03 / −.02 / .04;
    k_eff flat at ≈ 32 / 21 / 23 sentences for every K). The single HF heads that pass (Q3 L13H14
    slope .79, LL L24H14 .62, R² .99) track the number of sentences (64 + K): the length leak.
  * The ORACLE (labelled) needle share at the band decodes K only to exact .25 / .25 / .29, within-1
    .54 / .56 / .58 — the causal channel itself is coarse story by story, consistent with the
    per-needle d′ ≈ .5 of bundle C.
  **Reading.** The frozen models' heads are NOT sentence-selective: at the band the tail rows spread
  each head over 20–33 of the 64 sentences whatever K is, and the needles receive only a small excess
  of weight (the .02–.13 share of bundle C) summed over many heads and layers. There are no "K
  peaks" to count. Label-free statistics of diffuse patterns cannot recover the count; the count
  channel exists only relative to the needle labels. The attention-pattern aggregation head, as a
  readout of the frozen model's patterns, is dead. What would restore a countable pattern is
  selectivity that the frozen model does not have: learned (an adapter that sharpens a few heads'
  queries toward the question's predicate, then reads their degree) or structural (per-unit carriers
  whose read is selective by construction — the fence). Both are training-time interventions.

### Added 2026-09-03, before the run — bundle E: is the cause the SIMILARITY of competitors? (`outputs/_scratch/dbg/needles_types.py`)

Theory under test: dot-product attention scores conjunctive evidence additively, so distractors that
share one feature with the evidence (same person / same room) take a share proportional to their
partial match; the true items' excess is small and noisy; the readout shrinks it toward small counts.
Regimes at F = 64 sentences: HC (mixed, as before), HN (no-match distractors only), HP (same person,
other rooms), HR (other people, same room), HU (unrelated PG-19 sentences, no person/room/verb words).
**E1 — competitor type, not competitor count.** At K ∈ {6, 8}, exact match in HU and HN exceeds HC by
≥ .15 on ≥ 2 models, and the median emitted count at K = 1 is ≤ 2 in HU/HN while ≥ 3 in HP/HR.
FALSIFIER: HU is not better than HC by .10 at K ∈ {6, 8} on ≥ 2 models (similarity is not the cause).
**E2 — additive scoring.** At the band layers, tail rows, the head-mean logit of half-match sentences
sits at a fraction ∈ [.35, .75] of the way from no-match to full-match. FALSIFIER: ≤ .15 (the scores
are already conjunctive; leakage is not from additivity).
**E3 — test-time sharpening.** Multiplying the tail rows' logits at the band by β ∈ {1.5, 2, 3} raises
exact match at K ∈ {6, 8, 12} in HC by ≥ .15 for some β on ≥ 2 models, with an interior optimum.
FALSIFIER: no β improves any of those cells by ≥ .10 (selectivity at the band is not the test-time limit).
**E4 — compressive readout.** Scaling every tail row's needle share at all layers by m ∈ {.5, 1, 2, 3, 4}
on K = 4 stories: the emitted number is monotone in m and compressive — at m = 2 the median is ≤ 7
(linear would give 8), at m = 4 ≤ 12. FALSIFIER: linear or super-linear response.
**E5 — leakage (CPU, existing runs).** Within each K ≤ 8 in HF/HC, Spearman between the number of
half-match distractors in the story and the emitted count is > .2 on average on ≥ 2 models.
FALSIFIER: ≤ 0.
- 2026-09-03 — **E5 landed (CPU, before the GPU cells): SUPPORTED on all three models.** Within each K ≤ 8,
  Spearman between the number of half-match distractors in the story (same person OR same room) and the
  emitted count, on the original needles runs: Qwen-3B HF/HC mean +.32 / +.34, Qwen-7B +.32 / +.25,
  Llama +.46 / +.40 (per-K values +.03 … +.63; both the same-person and the same-room components
  positive in most cells). Stories with more half-match distractors get larger emitted counts at the
  same true K: the over-count is leakage from partial matches. (`outputs/_scratch/dbg/E5_leakage.md`.)
- 2026-09-03 — **Bundle E landed** (`outputs/needles_types/<model>/20260903_1552*`, jobs 418635 / 418636 /
  418637 after smoke 418634; n = 30 per cell; F = 64 sentences). Two bottlenecks, not one chain.
  * **E1 — FALSIFIED as stated; the leakage half holds.** Unrelated filler (HU) removes the small-K
    over-count on every model (median emitted at K = 1 / 2: HU 1 / 2 on all three vs HC 2 / 3–4) and
    lifts exact at K ≤ 2 to .80–1.00, but it does NOT restore counting at K = 6–8 on Qwen (Q3 HU .00 /
    .00 vs HC .20 / .07; Q7 .00 / .00 vs .00 / .00); only Llama gains (K = 8: .47 vs .07; K = 12: .73 vs
    .50; median tracks K to 12). HU's emitted curve on Qwen equals H0's (Q7: 1 2 4 4 5 7 7 for K = 1 …
    16): the cap at ≈ 4–8 is independent of competitor TYPE and of competitor COUNT (63 unrelated
    competitors vs none). No-match same-domain distractors (HN) make the models UNDER-count (Q7 K = 4
    → 2, Q3 K = 1 → 0); same-person distractors (HP) make Llama over-count massively (K = 8 → 12, K = 16
    → 20). The HC "counts to 4" is partly leakage compensating under-count. Falsifier fires (2 / 3).
  * **E2 — mixed; sharper than predicted.** At the band, the head-mean logit margin of a needle over a
    same-domain no-match sentence is only .54 / .48 / .79 nats (Q3 / Q7 / LL) against 3.3 / 4.4 / 4.1
    nats over unrelated filler. Within that small margin, same-PERSON half-matches sit at .85 / .77 /
    .77 of the way to a full match and same-ROOM half-matches at .26 / .28 / .21: the read matches on
    the person and barely on the room — the conjunction is essentially not implemented by the
    attention scores. Prediction band [.35, .75] holds for half_r on none, half_p exceeds it on all.
  * **E3 — FALSIFIED on Qwen, supported on Llama with a trade-off.** Sharpening the tail rows' logits at
    the band (β = 1.5 / 2) lifts Llama at K = 8 from .07 to .40 / .50 (median 12 → 9 / 8: it removes the
    over-count) but lowers K = 12–16 (median 12 → 8, 17 → 12); on Qwen no β improves any cell by ≥ .10
    and all β compress the emitted numbers (Q3 K = 12: 10 → 7 / 6 / 4). Sharpening trades leakage for
    compression; it is not a remedy.
  * **E4 — supported on Qwen, FALSIFIED on Llama: the readouts differ in kind.** Needle share × m at all
    layers, K = 4: Qwen-3B emits 3 4 5 6 6 (m = .5, 1, 2, 3, 4), Qwen-7B 3 4 4 4 4 (insensitive beyond
    the K = 4 level), Llama 4 6 8 9 11 (near-linear). K = 8: Q3 4 6 10 10 10, Q7 4 5 5 7 7, LL 8 12 17 17
    17. Qwen's share → number map is compressive / saturating; Llama's is faithful.
  **Net (dated).** Counting in one forward pass is limited by two separable factors whose weights differ
  by model: (i) SELECTIVITY of the read for the conjunction — a .5-nat margin over same-domain
  competitors, carried by the person feature alone; partial matches leak in proportion (E5) and, on a
  faithful readout (Llama), produce over-counts that sharpening or unrelated filler remove; (ii) the
  READOUT's map from share to number — compressive on Qwen (flat beyond ≈ 4), so even a selective read
  (unrelated filler) cannot count past 4–8 there. Competitor count per se (dilution) is not the cap for
  counting. Both factors are what the bind-then-count construction changes: the fence turns the
  per-unit conjunction into a full-depth nonlinear verdict (selectivity), and the sum read replaces the
  compressive map (readout).

### 2026-09-03 — Method run 1: bind, then count (`outputs/_scratch/dbg/train_bindcount.py`, `outputs/bindcount/`)

Qwen2.5-7B frozen (bf16); carrier token per unit (learned embedding, 3.6k params) + linear verdict head
(3.6k params); trained 3 epochs on 1,200 samples (MMRED filtered text N ∈ {8, 16, 32} from
`data/mmred_filtered_train`, needle stories N = 32, K ≤ 16, mixed distractor types); count = number
of carriers with verdict > 0. Evaluation n = 100 per cell. Stated before the run (message of
2026-09-03): the sum read reaches exact ≥ .9 at every N to 1024 and the mean-read baseline degrades
with N; the no-fence ablation degrades with N.
- **sum (fence at all layers + per-block position reset): exact 1.00 at N = 8, 16, 32, 64, 128, 256,
  512, 1024 on MMRED text and 1.00 on needle stories at N = 64 / 128 / 256 (K ≤ 32 / 64 / 64);
  per-carrier verdict accuracy 1.00; median worst-case verdict margin 9–14 logits.** Job 418651,
  `outputs/bindcount/sum/20260903_160058_2610047`.
- **sum_nofence (same carriers and head, full attention, native positions): exact 1.00 → .96 (N = 8)
  → .81 (64) → .62 (128) → .56 (256) → .44 (512) → .30 (1024); needles .37 / .10 / .05.** Verdict
  accuracy stays .996–.998 on MMRED, but the worst verdict per story degrades (median min-margin 14 →
  −.08 at N = 1024): with the other units visible, one verdict in N fails, and a count is exact only
  if all N are. Job 418652, `outputs/bindcount/sum_nofence/20260903_160058_2610312`.
- mean_lora (fence below L* = 12, open above, LoRA r8 on layers ≥ 12, digit CE at the answer row):
  did NOT train (loss ≈ ln 10 throughout, exact 0 even at N = 8) — lr 2e-3 shared with the carrier /
  head is far above the repo's LoRA recipe (1e-4); re-run pending with the repo recipe and a plain
  digit-LoRA arm (no carriers, no fence) as the mandated baseline. Job 418653 (invalid as a baseline).
- 2026-09-03 — **Method run 1, baselines (re-run with the repo LoRA recipe, lr 1e-4, 4 epochs, same 1,200
  training samples).** mean_lora (fence below L* = 12, open above, LoRA r8 on layers ≥ 12, digit CE at the
  answer row — the mean read over carriers): MMRED exact 1.00 / .98 / .94 (N = 8 / 16 / 32, in-length),
  then .60 / .26 / .10 / .09 / .09 (N = 64 … 1024); needles .00 / .02 / .02. Job 418683,
  `outputs/bindcount/mean_lora/20260903_161756_2615783`. digit_lora (plain prompt, no carriers, no fence,
  LoRA r8 ≥ 12, digit CE): 1.00 / .99 / .92, then .63 / .32 / .37 / .28 / .25; needles .11 / .05 / .02.
  Job 418684, `outputs/bindcount/digit_lora/20260903_161755_2615784`. Both mean reads fit in-length and
  decay past the training range, as the repo's earlier length results did; the sum read (previous
  entry) stays at 1.00 to N = 1024 with 7.2k parameters. The stated prediction (sum ≥ .9 everywhere,
  mean reads degrade) holds.
- 2026-09-03 — **Method run 1, multi-head arm (sum_multi): six set functions from one forward.** Same
  carriers and training data; a 7-way room-of-C head added beside the verdict head (7.2k + 25k params).
  Derived at evaluation without further training: count (sum of verdicts), exists (max), first / last
  (position of the first / last positive verdict), distinct rooms visited (|{rooms predicted for C}|),
  most-visited room (mode of room predictions). MMRED text N = 8 … 1024, n = 100: count 1.00 everywhere;
  exists / first / last 1.00 everywhere; per-frame room accuracy .995–1.00; distinct rooms 1.00 / 1.00 /
  1.00 / .99 / .97 / .86 / .85 / .71; most-visited room 1.00 / 1.00 / 1.00 / 1.00 / .99 / .87 / .91 / .82.
  Reading: statistics that tolerate no single-unit error (distinct, mode) decay with N as
  (per-unit accuracy)^N even at .998 per unit; the count head, trained with a count-aware loss and
  margins ≈ 11 logits, does not. Length-invariance of the construction is exact; its realized accuracy
  is set by the per-unit error rate raised to N — which is the design requirement for any per-unit
  head. Job 418709, `outputs/bindcount/sum_multi/20260903_162835_2621052`.
- 2026-09-03 — **Method run 1, VISION (Qwen2.5-VL-7B-Instruct, 4-bit, frozen; `train_bindcount_vlm.py`).**
  CarrierEngine layout (question first, one carrier token per frame, block fence, M-RoPE reset), the
  fence at ALL layers, linear verdict head on the carriers' final states, count = #(verdict > 0);
  trained 2 epochs on 450 samples (300 from `data/mmred_vfiltered/seq_len_8/train`, 150 from
  `seq_len_32/train`), 7.2k trainable parameters, frames at 392 px; test splits, n = 100 per N.
  * **sum (fenced): exact .99 / .99 / 1.00 / .99 at N = 8 / 16 / 32 / 64; per-frame verdict accuracy
    .9988 / .9994 / 1.00 / .9998; median worst-case margin 10.9 / 9.7 / 8.8 / 6.6.** Job 418711,
    `outputs/bindcount_vlm/sum/20260903_162936_2622612`. Reference frozen VLM on the same test splits
    (`outputs/mmred_vfrozen/`): full .40 / .28 / .20 / .215, oracle (gold frames only) .80 / .725 / .65 /
    .765, per-frame judge (N forwards) .45 / .41 / .385 / .36.
  * **sum_nofence (same carriers and head, full attention, native positions): exact .60 / .33 / .33 /
    .27; verdict accuracy .906 / .918 / .947 / .962; worst-case margin negative from N = 16.** Job 418712,
    `outputs/bindcount_vlm/sum_nofence/20260903_162936_2622613`. In vision the fence is worth 4–9 points
    of per-frame verdict accuracy (vs .3 in text): frames interfere with each other far more than
    sentences do, and a count multiplies that per-frame loss N times.
  Reading: the "bind" half holds when the per-unit verdict is a perception problem — the fenced VLM
  reads one frame and the question with the whole stack and is ~.999 right per frame, from 450
  examples. Beyond N = 64 no local vision data exist (N = 128 would need a new render).
- 2026-09-03 — **Method run 2: the two axes, separately; a second backbone.**
  * **sum_noreset (fence at all layers, NATIVE positions; Qwen2.5-7B):** exact 1.00 / 1.00 / 1.00 / 1.00 /
    .98 / .63 / .32 / .23 at N = 8 … 1024; verdict accuracy 1.00 → .984; median worst-case margin 14 → 5.8;
    needles .95 / .90 / .70. With competition removed by the fence but distance left in place, the
    per-unit verdict degrades as its unit moves away from the question (13k tokens at N = 1024): the
    distance axis of the BABILong analysis re-appears inside the construction, and the per-block
    position reset removes it (sum: 1.00 everywhere). Each ingredient of the fence-plus-reset now has
    its own ablation: fence ↔ competition, reset ↔ distance. Job 418739,
    `outputs/bindcount/sum_noreset/20260903_170524_2630656`.
  * **sum on Qwen2.5-3B:** 1.00 at every N to 1024 and .99–1.00 on needles (margins 14–21). Job 418740,
    `outputs/bindcount_q3/sum/20260903_170522_2631186`. Llama-3.1-8B: first attempt failed (no
    `<|fim_pad|>` token in its vocabulary → no carriers); re-run with a reserved special token as carrier.
- 2026-09-03 — **sum on Llama-3.1-8B (carrier = `<|reserved_special_token_10|>`): exact 1.00 at every N
  8 … 1024 on MMRED text and 1.00 on needles N = 64 / 128 / 256; verdict accuracy 1.00; margins 19–25
  (MMRED), 11–16 (needles).** Job 418744, `outputs/bindcount_ll/sum/20260903_171237_2633566`. The
  construction holds on all three backbones (Qwen-3B, Qwen-7B, Llama-8B) with the same 7.2k parameters.

### Added 2026-09-03, before the run — HERBench Action Counting (real video), bind-then-count vs frozen

Data: HERBench (DanBenAmi/HERBench) Action-Counting split, 2,040 questions on 136 HD-EPIC videos
("How many times does the action-object pair '<verb> <noun>' occur?", true_count 1–63, one annotated
timestamp per occurrence). Frames at t + 0.3 s (mid-action) plus fillers ≥ 3 s from every
occurrence, N = 16 and 32 per question, chronological, per-frame labels (port of the legacy prep to
ffmpeg). Split by VIDEO (no video in both train and test). Frozen Qwen2.5-VL-7B (4-bit).
**H1 — the per-frame verdict is a perception problem with a real ceiling.** Fenced carrier verdict
(as in MMRED vision) on held-out videos: AUC ≥ .80 but accuracy < .97 at N = 16. FALSIFIER of the
method's usefulness here: AUC ≤ .65 (the frozen VLM cannot see the action in one mid-action frame;
report as the perception boundary, not as an aggregation result).
**H2 — the count follows the verdict.** Exact count at N = 16 ≈ (per-frame accuracy)^16 within ±.15,
and ≥ 2× the frozen model's exact count on the same 16 frames (full prompt) and ≥ the frozen model
given only the evidence frames (oracle), if H1's AUC ≥ .8. Exploratory otherwise.
**H3 — the fence still matters on real frames.** Fenced ≥ no-fence by ≥ .10 exact at N = 16 and by
≥ .05 per-frame accuracy. FALSIFIER: no-fence ≥ fenced (real frames interfere less than rendered).
Also reported (no prediction): the 5-way MC accuracy obtained by mapping the count to the nearest
option, against HERBench's published 31–42 % for frontier Video-LLMs.
- 2026-09-03 — **HERBench Action Counting landed** (`outputs/bindcount_herbench/{sum,sum_nofence}/20260903_1940*`,
  jobs 419082 / 419083; Qwen2.5-VL-7B 4-bit frozen; 443 training samples from 84 videos, 3 epochs; test =
  150 samples from 51 held-out videos per N; frames 392 px; frozen baselines on the same frames).
  | N = 16 | exact | per-frame verdict acc | AUC |  | N = 32 | exact | acc | AUC |
  | fenced + sum | .12 | .683 | .774 |  | fenced + sum | .04 | .685 | .788 |
  | no fence + sum | .153 | .749 | .791 |  | no fence + sum | .127 | .817 | .799 |
  | frozen, full prompt | .12 (MAE 3.6) | — | — |  | frozen, full | .047 (MAE 4.7) | — | — |
  | frozen, oracle frames only | .02 | — | — |  | frozen, oracle | .027 | — | — |
  * **H1 — the perception ceiling is real and binding.** Verdict AUC .77–.80 (below the .80 support line,
    above the .65 floor): one mid-action egocentric frame carries only partial evidence for
    '<verb> <noun>' to this backbone. Per-frame accuracy .68–.82 makes an exact count over 16–32 frames
    nearly unreachable for any aggregator ((.75)^16 ≈ .01; observed .12–.15 because errors partly cancel).
  * **H2 — the count follows the verdict.** Exact ≈ verdict-limited; ≥ 2× the frozen full prompt only at
    N = 32 (no-fence .127 vs .047), not at N = 16 (.153 vs .12). Both arms beat the frozen oracle
    (.02–.03): the frozen model cannot even count the evidence frames it is handed.
  * **H3 — FALSIFIED: on real frames the fence HURTS.** No-fence verdict accuracy .749 / .817 vs fenced
    .683 / .685; exact .153 / .127 vs .12 / .04. Seeing the other frames helps a frozen VLM judge one
    frame here (the same action recurs; the camera is continuous), the opposite of rendered MMRED frames,
    where neighbours only interfere. Selectivity by isolation is not free when the per-unit evidence is
    weak and the units are correlated.
  **Net (dated).** Bind-then-count converts aggregation error into per-unit error exactly as designed;
  on HERBench the per-unit error is the backbone's perception ceiling, which isolation cannot lower and
  context slightly raises. Reported as the boundary of the method, not as a gain.

### Added 2026-09-07, before the run — reasoning-model shortcut program, stage 1 (docs/cot_operations_bridge.md §4 A/B, §6 S1/S2)

Scripts: `outputs/_scratch/dbg/judge_fenced.py` (A/B text), `judge_fenced_vlm.py` (A vision),
`cot_anatomy.py` (S1), `tally_helix.py` (S2); wrapper `run.sbatch`. Approved by Gabriele 2026-09-07
("Go ahead with the plan"). Predictions below are final before any GPU run.

- **A1** vision training-free fenced judge (Qwen2.5-VL-7B 4-bit, rendered MMRED N = 8/16/32/64, n = 100,
  one forward; per-frame cue "Is {C} in this image? Answer:" + own Yes/No logit; count = #yes): per-frame
  verdict acc ≥ .99 and exact ≥ .90 at every N. Falsifier: exact < .70 at N = 32.
- **A2** text (Qwen2.5-7B-Instruct, MMRED N = 8…1024, n = 100): per-unit verdict acc in [.93, .98] →
  exact < .5 from N = 32 on ((1−ε)^N). Falsifier of the reading: per-unit ≥ .995 at N ≤ 64 (then the
  trained head is unnecessary for text).
- **A3** cue wording: the direct question beats the meta-question ("Is this frame relevant to the
  question?") by ≥ .05 per-unit acc, text and vision.
- **B1** untrained verdict, cue adjacent to the read position vs. question only in the prefix (block ends
  with " Answer:"): adjacent > prefix-only by ≥ .05 per-unit acc on MMRED text and on needles HC.
  Falsifier: adjacent ≤ prefix-only on both. (B2, trained head with/without cue, is run 2 if B1 holds.)
- **S1a** tally on the tape (Qwen3-8B thinking, needle stories K ∈ {4, 8, 16} × J ∈ {0, 64}, n = 12,
  max 3072 new tokens): ≥ .6 of the J = 0 traces contain a written running tally (a chain of ≥ 3 numerals
  increasing by one); corrupting one tally numeral by +1 and regenerating makes the next written tally
  numeral follow the corruption (= corrupted + 1) in ≥ .8 of cases and shifts the final answer by +1 in
  ≥ .6. Falsifier: follow rate ≤ .5 (a latent counter overrides the tape).
- **S1b** per-item retrieval: the attention share from a chunk's first tokens onto its target sentence
  (mean over sampled layers) has a median at J = 64 at most half the J = 0 median, and the per-trace mean
  target share predicts failure (wrong answer or truncation) within J = 64 with AUROC ≥ .7.
  Falsifier: AUROC ≤ .55.
- **S2 H1** read test (Qwen2.5-7B-Instruct; number vectors v_k = mean answer-row residual over copy
  templates "… the answer is", k = 0…64; patch the answer row of plain MMRED prompts, N ∈ {64, 128, 256},
  gold k, at layers {8, 12, 16, 20, 24}, re-applied at each decode step): emitted = k in ≥ .9 of prompts
  at the best layer. Falsifier: < .5 at every layer.
- **S2 H2** geometry: v_k over k = 0…64 is fit by [1, k, cos(2πk/T), sin(2πk/T)], T ∈ {2, 5, 10, 100},
  with R² ≥ .9 at the best layer (linear-only R² reported alongside). Falsifier: R² < .7 at every layer.
- **S2 H3** extrapolation: helix fit on k ≤ 32 only; reconstructed v̂_k for k ∈ 33…64 patched as in H1:
  emitted = k in ≥ .6. Falsifier: < .3.

Cost ≈ 4 GPU-h (text judge ~20 min, vision judge ~1 h, S1 ~40 min, S2 ~30 min). Outputs:
`outputs/judge_fenced/`, `outputs/judge_fenced_vlm/`, `outputs/cot_anatomy/`, `outputs/tally_helix/`.

### 2026-09-07 — Outcomes of the shortcut program, stage 1 (jobs 428210 text judge, 428234 vision judge, 428212 trace anatomy, 428235 tally; first vision/tally submissions 428211/428213 died on a dtype cast and a tokenization bug before producing results and were fixed and resubmitted)

Run dirs: `outputs/judge_fenced/Qwen2.5-7B-Instruct/20260907_111006_4003053`, `outputs/judge_fenced_vlm/20260907_111236_4004643`,
`outputs/cot_anatomy/Qwen3-8B/20260907_110935_4003321`, `outputs/tally_helix/Qwen2.5-7B-Instruct/20260907_111230_1973535`. n = 100 per cell
(judges), 12 per cell (traces), 20 prompts per N (tally).

- **A1 SUPPORTED.** Vision, fenced, direct cue, no trained parameters, one forward: exact .98 / .97 / .99 / .98 at N = 8 / 16 / 32 / 64;
  per-frame verdict acc .9975 / .9963 / .9994 / .9997; AUC 1.00 everywhere. (Trained head, same data: .99 / .99 / 1.00 / .99.)
- **A2 FALSIFIED in the favourable direction.** Text MMRED, fenced, direct cue: per-unit acc 1.000 and exact 1.00 at every N from 8 to 1024
  (24.5k tokens), against the predicted per-unit .93–.98. The stated "falsifier of the reading" (per-unit ≥ .995 at N ≤ 64) fired: on MMRED
  text the trained head is unnecessary. On needle stories with same-domain distractors (HC, N = 64 / 128 / 256) the zero-shot judge reaches
  per-unit .995 / .995 / .998 with AUC ≈ 1.00 but exact .89 / .92 / .91: the ranking is perfect, the zero threshold is slightly off, and one
  miscalibrated verdict in N breaks the count. The trained 2d + 1 head (1.00 on the same cells) is a calibration of an already perfect ranking.
- **A3 MIXED.** Direct question vs. "is this frame relevant to the question?": the direction holds in every cell; the ≥ .05 per-unit gap holds
  on the hard text family (needles: .995–.998 vs .86–.93, exact .89–.92 vs .09–.19 with heavy over-counting) and at N ≤ 16 on vision
  (.9975 vs .951, .996 vs .947; exact .98 / .97 vs .74 / .62); it is .01 or less on MMRED text (meta exact 1.00 / .79 / .98 / .99 / .97 / .99 /
  1.00 / 1.00). Cue wording matters where selectivity is hard (partial matches), as the two-bottleneck account says.
- **B1 SUPPORTED at the count level, mixed at the per-unit level.** Cue adjacent to the read position vs. question only in the prefix, fenced,
  MMRED: exact 1.00 vs .49 / .16 / .07 / .22 / .41 / .66 / .66 / .80 (N = 8 … 1024); per-unit 1.000 vs .914 / .795 / .894 / .958 / .985 /
  .998 / .998 / .9996 (gap ≥ .05 at N ≤ 64, shrinking with N as the prefix-only calibration improves); AUC ≥ .9994 in both. Needles: exact
  .89–.92 vs .43–.55, per-unit .995–.998 vs .970–.984. The adjacent cue calibrates a signal that is already almost perfectly ranked.
- **Unplanned control that carries the result: the fence.** Same per-block direct cue, causal attention and native positions (fence 0):
  text MMRED exact .65 / .14 / .18 / .01 / .00 / .00 / .00 / .00, per-unit .94 → .54, AUC .99 → .88 (N = 8 … 1024); needles exact .04 / .01 /
  .00, per-unit .67 / .62 / .63, AUC .90 / .83 / .71; vision exact .33 / .41 / .31 / .31, per-frame .77 / .92 / .955 / .974, AUC .92–.99.
  Without the fence the per-unit ranking itself degrades with N. The training-free bind-then-count = fence + per-block question + the model's
  own yes/no + sum; the cue calibrates, the fence isolates, and both are needed.
- **S1a MIXED, with a sharper reading.** Qwen3-8B thinking, 71 traces. A written running tally (numerals 1, 2, 3, … in order) appears in .97
  of clean traces and 1.00 of distractor traces (chain length = K in 34 / 36 clean traces). Corrupting the middle numeral by +1 and regenerating:
  the next written numeral follows the corruption (v + 2) in .63 of clean and .86 of distractor traces, pooled .75 (predicted ≥ .8, falsifier ≤
  .5: not fired); the "repair" outcome (v + 1 written right after) occurred 0 times in 71: no latent counter overrides the tape. The final
  answer shifted by +1 in only .06 of clean traces (predicted ≥ .6: FALSIFIED). Reading from the continuations: the traces carry redundant
  written tallies (an item index "9." and a running count "– 9", or ordinals "Seventh:" and words "Seven"); a corrupted numeral propagates along
  its own chain while the other chain, and the final answer read from it, stay correct. The count is on the tape, in more than one copy.
- **S1b SUPPORTED on the drop, FALSIFIED on the predictor.** Chunk-onset attention on the target sentence as a fraction of context attention:
  median .45 / .43 / .42 (clean, K = 4 / 8 / 16) vs .082 / .073 / .095 with 64 same-domain distractors, a 4.5–6× drop (predicted ≥ 2×).
  Within J = 64 the per-trace share does not predict failure (AUROC .545; 28 of 36 traces fail, mostly by looping: chunks per trace 132–154 vs
  17–36 clean, item indices running past the number of sentences). Accuracy: clean .92 / 1.00 / .92, distractors .33 / .42 / .00.
- **S2 H1 SUPPORTED for one token, not beyond.** Patching the answer row at layer 24 (of 28) with the number vector v_k makes the model emit
  k for every single-digit k (1.00 at gold, which is ≤ 8; frozen no-patch .30–.45 on the same prompts) but .26 for k = 9–32 and .01 for 33–64;
  earlier layers (8–20) read the patched vector poorly even for single digits (.16–.33). Template copy accuracy of the vectors 1.00.
- **S2 H2 MIXED.** Helix + linear basis explains R² .698 / .718 / .674 / .663 / .401 of v_k at layers 8 / 12 / 16 / 20 / 24 (linear-only .38 /
  .44 / .41 / .41 / .13): above the .7 falsifier at the best layer, below the predicted .9.
- **S2 H3 FALSIFIED.** k ∈ {33 … 64} patched from the true vector, the helix fit on k ≤ 32, or the linear fit: 0.00 emitted correctly in all
  three. With digit-by-digit tokenization a single patched row carries one token; the second digit is not recoverable from it. An in-model
  tally needs a different write (per-step state, or a single-token count vocabulary); recorded as the boundary of S2.

### 2026-09-08 — Added control, NOT pre-registered (run at Gabriele's question "do we have a version with only the mask and the repack?"): fence + per-block position reset only, frozen model writes the count itself (jobs 432240 text, 432241 vision; `outputs/_scratch/dbg/fence_only.py`)

Runs: `outputs/fence_only/text/20260908_104326_139582` and `outputs/fence_only/vision/20260908_104326_139580` (n = 100 per cell). Exact match, fenced+reset / plain causal on the same prompt:
- Text (Qwen2.5-7B-Instruct), N = 8 … 1024: .16 / .07 / .07 / .07 / .07 / .10 / .03 / .05 vs .72 / .57 / .53 / .43 / .27 / .30 / .23 / .24.
  Under the fence the model answers 0 or 1 up to N = 128 (median prediction) and garbage beyond (median 123 at N = 512, 73 at N = 1024).
- Vision (Qwen2.5-VL-7B 4-bit, question + "Answer with the number only."), N = 8 … 64: .14 / .10 / .07 / .14 vs .16 / .18 / .19 / .19
  (the canonical count prompt without the appended instruction gives the frozen model .40 / .28 / .20 / .215, outputs/mmred_vfrozen).
Reading: isolating and repacking the frames without a per-frame verdict leaves one softmax read over N frames at the answer token and
makes the frozen model worse than with no fence at all. The per-block question + external sum is the mechanism (judge_fenced: 1.00 / .98).


### 2026-09-08 — Native aggregation pilot P0 (added before GPU runs)

Authorized by Gabriele's “Let's go ahead”; all GPU and heavy CPU work goes through
Slurm. This tests the native-output feasibility gate in
`docs/paper/NATIVE_AGGREGATION_PROPOSAL.md`. It does not establish vision or
reasoning improvements. CPU staging/correctness job: 433097.

**Construction.** Frozen bf16 Qwen/Qwen2.5-3B-Instruct, ordinary SDPA and vocabulary
head. Add one branch at zero-based layer 18 of 36: native Q/K/V, fixed contiguous
64-token blocks in absolute cache slots, per-block softmax, shared rank-64
query-plus-read projection followed by SiLU, then sum or mean, zero-initialized
output projection into the residual before the remaining layers. Applied at every
causal token. No labels for individual frames, external tally, position reset,
new token, or additional backbone traversal. Query tile size 32 is an implementation
choice; it bounds temporary scores but not total saved training activations.

**Four arms:** base, upper LoRA alone, nonlinear-sum branch plus upper LoRA,
nonlinear-mean branch plus upper LoRA. LoRA is rank 8, alpha 16, q/k/v/o of last
4 layers, same initialization seed and supervision. These are matched for the
upper adaptation allowance, NOT total trainable parameters or compute; report
both costs. Linear message and additional-attention controls remain later work.

**Data and training.** Copy existing text datasets without changing sources to
`/mnt/data/gabriele/gnn_transformer`. Canonical raw text prompt from
`scripts/condmask/textdata.py`; question in prefix and tail; region boundaries do
not use frame annotations. Select 90 examples per N in {8,16,32} from train,
36 per N in {8,32} from dev, 72 per N in {8,32,64,128} from test. Seeded shuffle
before limiting, data seed 1234+N, same samples across arms; record IDs and K
histograms. All K<=8. This is distractor/length extrapolation with familiar outputs,
not numerical extrapolation. Train 3 epochs, accumulation 4, seed 0, AdamW with
weight decay 0, branch LR .001, LoRA LR .0001, gradient clipping 1. Full-vocabulary
next-token CE on complete final answer plus EOS only. Select epoch on in-range
dev exact match, break ties with lower first-answer-token NLL; never select on test.

**Evaluation.** Native greedy cached generation, at most 4 new tokens, strict
whole-integer parse; all unparsable samples count as wrong. Primary metric exact
match, secondary first-token native NLL, MAE/bias with parse coverage, generation
latency and peak memory. Report all four lengths and paired outcomes. A later
larger/seeded study is necessary for a reliable improvement claim.

**Predictions and decision rules.** P0.1: numerical reconstruction/causality/cache
checks pass and branch output projection gets finite nonzero gradient at zero
initialization; failure blocks training until repaired. P0.2: nonlinear sum reaches
>=.80 pooled in-range test exact on N8,32. P0.3: nonlinear sum exceeds LoRA-alone
by >=.10 pooled exact on N64,128 without losing >.05 pooled in-range exact.
P0.4: sum exceeds mean by >=.05 pooled out-of-range exact. A threshold miss is a
miss for this pilot, not evidence of a universal impossibility. If P0.2 fails,
native interface learning remains unresolved. If P0.3 fails, do not scale this
unchanged configuration into a large vision/reasoning campaign. Post-hoc diagnosis
may motivate a separately registered revision. No result from a probe substitutes
for the native-output metrics.

**Resources.** One B200 at a time on `gpu`, 4 CPUs and 96 GiB requested RAM, 2-hour
hard GPU wall limit covering a short smoke and all four arms (<=2 GPU-hours).
Smoke: sum, 2 optimizer steps, N8 only, 9 train/dev/test records; checks runtime,
finite loss, checkpoint reload and cached generation, not an efficacy result.
CPU validation job cap15 min, 4 CPUs,16GiB. Checkpoints strictly under
`/mnt/ckpts/gabriele/gnn_transformer/native_aggregation`; reports and source hashes
under `outputs/native_aggregation/`; job scripts under `slurm/`. No package installs.


### 2026-09-08 — P0 outcomes (job 433106, completed; 3m53s on one B200)

All 13 new CPU tests passed in job433097; four existing core suites also passed
(optional legacy nnsight comparisons skipped). GPU smoke passed finite backward,
checkpoint reload and cached native generation. P0.1 supported.

72 test examples per N; exact counts at N8/32/64/128:
- Base strict: 0/0/0/0. This is a termination-format failure: the raw prompt often
  produces a number followed by an explanation. It is NOT zero aggregation skill.
- Base leading integer, **post-hoc diagnostic**: 50/20/13/13.
- LoRA: 62/29/20/15.
- Nonlinear sum + LoRA: 65/32/19/11.
- Nonlinear mean + LoRA: 67/46/25/13.

Every trained-arm output parsed as a whole integer. Thus their comparisons are
not caused by stopping format. P0.2 MISSED: sum in-range97/144=.6736, below .80.
P0.3 MISSED: sum out-of-range30/144=.2083 versus LoRA35/144=.2431; paired sum wins7,
loses12. P0.4 MISSED in the opposite direction: mean38/144=.2639, sum30/144=.2083.
At N128, native first-token NLL is5.186(sum) versus2.892(LoRA), consistent with the
behavioral failure. These are single-seed pilot results, not a broad negative claim.
Do not scale the unchanged sum configuration into a vision/reasoning campaign.

Canonical runs: `outputs/native_aggregation/p0/{base_seed0_20260908_163318_433106_313176,
lora_seed0_20260908_163350_433106_313348,sum_seed0_20260908_163437_433106_313473,
mean_seed0_20260908_163543_433106_313685}`. Exact sources and checkpoints are recorded
inside each config; `scripts/analyze_native_aggregation.py` reproduces the paired
and formatting diagnostics. No RESULTS.md update was made.

### 2026-09-08 — P1 diagnostic (added before its GPU run; motivated post hoc by P0)

Purpose: distinguish a repeated query-only contribution, insufficient local-read
resolution, and insufficient training. This is exploratory development on the
same samples already used in P0. It cannot independently confirm a benchmark gain.

One parameter-free change: optionally center the message network at a zero read:
`m(q,r)=SiLU(W_r r+W_q q+b)-SiLU(W_q q+b)`. For R visible regions the existing raw
sum includes exactly R copies of the query-only response. Centering removes these
copies and makes a zero-valued read an exact no-op. It does NOT make arbitrary
irrelevant nonzero reads zero. No new losses, labels, output decoder or positions.

Factorial: raw/centered messages crossed with block size64/16, all using sum.
Include the same upper-LoRA-only arm. Train each of five arms from scratch for
9epochs on the same270 examples, seeds, optimizer, rates and upperLoRA as P0.
Record fixed-epoch test metrics at epochs3 and9 without selecting epochs on test;
separately retain ordinary in-range-dev-selected results. All test cells and
strict parsing rules remain P0's (72 perN8/32/64/128, K<=8). No warm starts and no
hyperparameter selection from this factorial.

Predictions, evaluated on fixed epochs: P1.1 centering at size64 improves pooled
N64/128 exact by>=.10 over raw64 and reduces pooled native first-token NLL by>=.30
at epoch9, with <=.05 pooled in-range exact loss. P1.2 reducing size64 to16 gives
>=.10 pooled out-of-range exact gain within at least one of raw/centered pairs at
epoch9. P1.3 extending raw64 from3 to9epochs improves its pooled in-range exact
by>=.10; also report this duration effect on the LoRA-only arm. Report all cells,
including missed thresholds and interactions. A centered main effect supports
that specific intervention; it does not prove irrelevant-block noise is eliminated.
A size effect motivates more resolution/boundary controls; it is not invariance.
An extension effect in every arm suggests undertraining rather than a specific
architecture advantage. If no branch has >=.10 long-N gain over epoch9 LoRA with
<=.05 short-N loss, native extrapolation remains unresolved and stop this block.

Resource cap: one B200,4CPUs,96GiB,30minutes on gpu; expected minutes based on
P0's measured3m53s total. CPU correctness cap10minutes,4CPUs,16GiB before GPU.
Together with P0 this stays below the previously stated2GPU-hour budget. Same
checkpoint/data directories; results under `outputs/native_aggregation/p1`.


### 2026-09-08 — V0 software smoke (added before GPU execution)

Purpose is integration verification, NOT a vision efficacy experiment or scaling
P0. Use ordinary Qwen2.5-VL-7B-Instruct in nf4 through load_runtime, actual images
resized392x392 and the existing canonical chat prompt, zero-reset-free native
positions, ordinary cache. Stage two preselected N8 samples per train/dev/test
split on a CPU allocation without modifying the source. Train raw sum plus last
4-layer LoRA for one optimizer step on two examples; generate on two dev and two
test examples, save/reload a checkpoint. Layer defaults to the middle language
layer; rank64, block64; same learning rates as P0. Finite nonzero gradients,
checkpoint restoration, ordinary image forward and cached generation must work.
No accuracy hypothesis, no extrapolation or vision-improvement claim from n=2.
One B200/4CPUs/96GiB/15min cap, scheduled after P1 so only one GPU is used for this
work. Data /mnt/data/gabriele/gnn_transformer/mmred_vfiltered; checkpoints
/mnt/ckpts/gabriele/gnn_transformer/native_aggregation_vlm; reports
outputs/native_aggregation_vlm/smoke. P0+P1+V0 hard limits total under2GPU-hours.


### 2026-09-08 — P1/V0 execution record (before their outcomes)

P1 first CPU check433127 caught tiny shape-dependent SiLU roundoff in an exact
zero-read cancellation test. No GPU work ran: dependent job433130 was canceled.
Computing the baseline activation with the same shape/layout fixed cancellation;
CPU433131 passed all15 tests in5.43s. Final P1 GPU job433133 runs the registered
five arms. V0 CPU staging433137 verified six samples and byte hashes, preserving
sources; GPU433138 waits for P1 and staging. CPU433142 generates the complete P1
report/figure after P1 succeeds. This implementation correction does not change
P1's parameters, data, objectives or comparisons. All code snapshots are saved.


### 2026-09-08 — P1 outcomes and P2 capacity control (P2 added before its run)

P1 GPU433133 completed in13m42s; report CPU433142 completed. At fixed epoch9,
exact counts N8/32/64/128 (72 each): LoRA61/29/25/15; raw64 69/41/31/23;
centered64 66/45/26/10; raw16 69/43/32/14; centered16 67/28/8/0.
P1.1 centering MISSED; P1.2 smaller regions MISSED; P1.3 duration SUPPORTED:
raw64 in-range exact rises79/144=.5486 at epoch3 to110/144=.7639 at epoch9.
Best long-N raw64 is54/144=.375 versus LoRA40/144=.2778, a +9.72pp difference,
slightly below the pre-set10pp gate (one additional paired net win would cross it).
This is a near miss, not proof of no benefit. However its long-N first-token NLL
is worse, and the other revisions are less successful. Centered16 also loses
output formatting at long N (parse61/72 atN64,0/72 atN128). P1's stopping criterion
fired: stop architectural variations in this block. All cells and paired outcomes:
`outputs/native_aggregation/p1/analysis.json`; figure `fixed_epochs.png`/`.pdf`.

V0 GPU433138 completed in35s. Actual images, finite backward, native cached
generation and checkpoint reload succeeded; training peak CUDA11,079,996,416bytes.
Run `outputs/native_aggregation_vlm/smoke/sum_seed0_20260908_170316_433138_331049`.
Its n=2 per split results are software checks, not vision efficacy evidence.

**P2 closes an outstanding comparison, not additional architecture search.** P1
branch+LoRA has802,880 trained parameters versus409,600 for LoRA. Run one LoRA-only
control of rank16/alpha32 on the same last4 layers (819,200 parameters,2.0% above
branch+LoRA), keeping alpha/rank=2 and all other P1 settings/data/seed fixed.
Train9epochs from scratch; record fixed3/9 and separate dev-selected metrics.
Compare fixed9 to P1 raw64. P2 prediction: raw64 retains >=.05 pooled N64/128 exact
advantage with <=.05 pooled N8/32 loss. If missed, no claim that this branch is a
better use of trainable capacity on this pilot. Report NLL and latency alongside
exact match. This is exploratory, same data reused, not independent confirmation.
One B200,4CPUs,96GiB,5-minute cap; reports `outputs/native_aggregation/p2/rank16`.
Same model/data storage roots. No additional architectural variants or vision
training campaign. All GPU work including P2 remains far below the2GPU-hour cap.


### 2026-09-08 — P2 outcome (GPU433157 completed)

Rank16 LoRA, fixed epoch9: exact counts64/29/22/16 atN8/32/64/128, long-N38/144=.2639.
Original sum64 at fixed9:69/41/31/23, long-N54/144=.375. P2 prediction SUPPORTED:
+11.11pp long-N and +11.81pp in-range. The control has819,200 parameters versus
802,880 for branch+LoRA. This is exploratory, not independent confirmation.
Original sum has worse long-N first-token NLL (~2.72 vs~2.36) and about1.5x long-N
batch-one generation latency. With dev-selected checkpoints the long-N comparison
shrinks to45/144=.3125 (sum) versus38/144=.2639 (rank16 LoRA); rank8 LoRA is41/144.
Thus a fixed-epoch gain survives the approximate parameter control, but practical
selection, calibration, compute and replication remain unresolved. P0/P1 failed
thresholds remain recorded unchanged. Full report:
`docs/paper/NATIVE_AGGREGATION_PILOT_RESULTS.md`.


### 2026-09-10 — V1 direct MMReD Vision mechanism comparison (added before all V1 runs)

**Authorization and scope.** Gabriele requested direct MMReD Vision experiments
and maximal practical parallelism, replacing the proposed text-first screen.
Run a new block addressing the literature audit in
`docs/paper/NATIVE_AGGREGATION_POSITIONING.md`. P0/P1 misses and the P2 limitations
remain unchanged. This is a single-seed vision screen, not confirmation of a
new attention family or reasoning composition.

**Data.** Stage source `data/mmred_vfiltered` without editing it, under
`/mnt/data/gabriele/gnn_transformer`. Main: train90/N8,N16; dev36/N8,N16;
test100/N8,N16,N32,N64. K in0..8 throughout. Deterministic data seed20260910,
round-robin across available gold categories, exact common manifest acrossarms.
Audit duplicate sequence+question content across splits, distinguish reused SIDs
from duplicate content, and exclude six prior V0 examples. Source availability
or content-duplicate shortages will be recorded before model training and may
require an explicit protocol addendum. Separate small profiling samples do not
select models or architectural parameters. Profile test examples are disjoint
from main tests. No state/per-frame labels enter the model or training loss.

**Model/training.** Qwen2.5-VL-7B-Instruct, frozen nf4 backbone, bf16 compute,
ordinary image encoder and native vocabulary output, images resized392x392,
canonical existing count prompt. Middle decoder layer14, block64, message rank64,
query tile32. Upper4-layer q/k/v/o LoRA rank8 alpha16 for adapter arms. AdamW
branchlr0.001, LoRAlr0.0001, weightdecay0, accumulation4, gradientclip1, seed0,
nine epochs. Same example order and LoRA initialization acrossarms. All use
answer+EOS full-vocabulary CE and ordinary greedy cached generation(max4tokens).
No reasoning tokens, fences, position resets, numeric head, or external tally.

**Seven arms.** `sum`=local PRE_R; `mean`=local PRE_1;
`post_sum`=local POST_R; `post_mean`=local POST_1;
`global`=per-head exact global-softmax read followed by the same nonlinear adapter;
`hierarchical`=learned per-head normalized fusion of local reads then the adapter;
`lora`=no branch, LoRA rank16 alpha32 (approximately matches totalparameters).
For local arms a_tb=Wr*r_tb+Wq*h_t+b, PRE_s=Wup*s_R*mean_b SiLU(a_tb),
POST_s=Wup*s_R*SiLU(mean_b a_tb), s_R=1 orR with causal validregioncountR.
Centering remains off. Hierarchical logweights start at exact per-head logZ and
add a learned rank8 query/read scorer shared acrossheads but evaluated separately;
its outputprojection is zero-initialized. Normalize acrossvisibleblocks perhead.
This is a generic dense hierarchical-fusion control, NOT a HiLS reproduction.
The learned scorer adds a small recorded parameter overhead. Every adapter keeps
ordinary dense attention. Compute is measured, not claimed to match.

**Selection and metrics.** Select best epoch only by pooled in-range dev exact,
then lower first-answer-token NLL. Test only the selected checkpoint, after training.
Primary exact includes all selected examples, with noninteger outputs incorrect.
Report perN and pooled in-range(N8,N16)/out-of-range(N32,N64), parsed MAE/bias,
parse rate, first-answer-token native NLL, latency(includevision/preprocessing),
peakCUDAmemory, trainableparameters, sampleIDs and source/checkpoint provenance.
Store all dev epochs and the final checkpoint so selection sensitivity is visible.

**Predictions and decision criteria.** V1.1 primary mechanistic contrast:
PRE_R (`sum`) beats POST_R (`post_sum`) by>=0.05 pooled N32/N64 exact, with<=0.05
pooled N8/N16 loss. V1.2 secondary analogous PRE_1 versusPOST_1, same thresholds.
V1.3 practical screen: at least one PRE arm beats EACH of post_sum,post_mean,
global,hierarchical,lora by>=0.05 pooled OOD exact with<=0.05 in-range loss against
each. Report every contrast and failures, not just the best cell. Paired bootstrap
95% intervals (10000replicates, resample within N andgold where possible) are
exploratory single-seed uncertainty, not seed-generalization evidence or adjusted
multiple-comparison tests. If practicalscreen fails, do not expand the same
configuration automatically; diagnose the observed failure. If it passes,
pre-register seed1/2 confirmation of the candidate and strongest controls before
additional runs, subject to the remaining announced compute budget. A positive
screen alone does not establish the five-point replicated advancement criterion.

**Execution budget and correctness.** AllGPU/heavyCPUwork viaSlurm. CPU staging
and mathematical/decoder tests may run concurrently. Validate PRE=POST forlinear
activation, mask/causality, zero-initialization and fullprefill/cached parity in
Qwen2/Qwen3/VL. Profile real N16 train/backward and N8/16/32/64 generation for
sum,post_sum,global,hierarchical on separate samples, at most4B200 concurrently,
15min each. Launch main seven arms as a Slurm array throttled to4, oneB200,
4CPUs,96GiB RAM each, at most60min perarm after profiling feasibility. Total
initial allocation caps<=8GPUhours. No cancellation of unrelatedjobs. If profiling
requires a settings change, document it before main model results. Reports in
`outputs/native_aggregation_vlm/v1`, models under
`/mnt/ckpts/gabriele/gnn_transformer/native_aggregation_vlm`.


### 2026-09-10 — V1 data availability correction (before model runs)

CPUstaging440389 found that rendered N16 has only test, N32 only train/test,
and N64 onlytest; the handoff's general split notation was insufficiently precise.
It failed before publishing manifests or running any model. Keep the registered
train/devN8,N16 and testN8,N16,N32,N64: render the missing N16 train/dev images
from the existing canonical filtered text training/development sequences, using
`scripts/condmask/gen_filtered_counting.py`'s renderer configuration
(`datasets/mmred/render_mmred.py`, sixrooms ending inPark). Copy existing QA
unchanged, retain provenance, and verify pixel parity with an existing N8 image
before rendering. All generation/audit/copywork runs on CPU Slurm and writes only
under /mnt/data/gabriele/gnn_transformer. The source sequences and existing vision
tests are unchanged. Missing unused split/length groups are reported, not required.
The main/profile sets remain disjoint; source shortages are checked explicitly.


### 2026-09-10 — V1 data and CPU validation outcomes (before GPU profiles)

Staging440440 passed in1m17s. All664 selected main/profile samples have distinct
normalized sequence+question content, despite42 cross-split duplicate-content
groups in the full source population. All main training/dev cells are exactly
balanced overK0..8; each testN has11/count plusoneadditionalK6. MissingN16train/dev
rendering passed six existing-image RGB parity comparisons. Total stagedimages
136,985,133bytes, copiedfiles137,761,376bytes. MainmanifestSHA256
321512f2a29209a44f91a301164e72db91d0727320086a0f1d8759d5db384ae4;
profilemanifestSHA256 fe467facd8d2a797a6ba563e2eedc663cd1db786abd6f26c44a0e85f3b0858a5.
CPU440487 passed mathematical/decoder variant checks and existing fencing,
carrier-mask,scratchpad,data suites in27s. Optionallegacyparity skipped where
nnsight unavailable. No efficacy outcomes yet. The generic hierarchical scorer
adds2064 parameters atVLMhead_dim128 andrank8; actual totals logged atmodel load.


### 2026-09-10 — V1 implementation review (before GPU profiles)

Independent review found POST's affine projection could be moved after averaging
without changing its mathematical function. Applied that optimization so POST
gets its natural lower compute cost, then re-ran CPU correctness checks. The
global mechanism control still reconstructs its read from segmented attention;
it is not an optimized reuse of the existing dense SDPA output. Timings of this
screen are descriptive and cannot establish a compute-optimal baseline frontier.


### 2026-09-10 — V1 profile outcomes and main launch (before main training)

Final CPU recheck440491 passed all23 native aggregation tests plus existing
regression checks. Profile array440492 completed all4 tasks in49–61 seconds
each (230 allocated GPU-seconds total). Training used14.59–19.69GB peak allocated
memory; N64 evaluation used8.90GB and2.86–2.93seconds/example including preprocessing.
The cold two-example N16 training measurements were1.31–1.76seconds/example.
A pessimistic all-N16 extrapolation is near the60-minute cap; half of main
training is N8, and startup overhead is amortized. Keep the registered60-minute
main limits and unchanged scientific configuration. All outputs, including
profile predictions, are retained; no efficacy-based changes were made.
Proceed with the seven-arm main array at most4 concurrent GPUs, followed by
the registered CPU paired analysis. Actual profiles plus maximum main runtime
are7.064GPU-hours, within the8GPU-hour initial cap.


### 2026-09-10 — V1 frozen reference and data-law audit (during main training, before main test results)

Add a descriptive frozen-backbone reference on the exact400main test examples,
using the existing base arm (no adapters, no training, identical image/prompt/
decoding settings). This does not change the seven trained-arm contrasts or
selection rules. One GPU,15minute cap; worst-case cumulative profile+main+base
allocation is7.314GPU-hours, below8. Prediction: adaptation should improve
in-range exact over this reference; no new success threshold is introduced.
Run after the first main wave starts freeing capacity; account cap remains4GPUs.

The independent data audit identified possible train/test nuisance-law shifts
in target-character occurrences in N16, despite renderer parity and correct
conjunction labels. A CPU metadata-only census is being run across all selected
examples. No data or running configuration changes. Paired comparisons remain
on identical samples, but the length-only interpretation will be qualified
according to the census. This concern was raised before main test outcomes.


### 2026-09-10 — V1 likelihood-label correction (during main training, before main test outcomes)

Independent evaluation audit found that the cached Qwen2.5-VL generation config
sets repetition_penalty=1.05. The harness explicitly makes generation greedy
but inherits this penalty. Its generated.scores-based first-token NLL is thus
full-vocabulary generation-policy NLL after repetition processing, not raw LM
NLL as initially described. This same policy NLL breaks development-accuracy
ties. All arms share these settings and retain them; no running model/harness
code or selection rule is changed. Report labels are corrected. Prompt digits
can be penalized, potentially in a length-dependent way. Raw-LM calibration is
not established by this experiment. Cached config and hash retained in
outputs/native_aggregation_vlm/v1/generation_settings.json.


### 2026-09-10 — V1 metadata census outcome (CPU440524, before main test outcomes)

All664selected QA hashes and final-gold recounts pass; every frame contains one
character. N16training/dev target-character occurrences average10.311/10.278
versus6.12in N16test.75/90train and27/36dev exceed8target-character frames;
0/100test do. Same-character/wrong-room distractors average6.311/6.278 versus
2.10in test. These N16splits are not IID; pixel parity confirms rendering
consistency, not the sampling law. N8target means are5.978/5.972/5.78for
train/dev/test, with no analogous support violation. TestN8/16/32/64target
means5.78/6.12/5.97/6.07, while same-room other-character distractor means
0.41/1.70/4.35/9.85. The registered comparison proceeds unchanged on common
examples; results test a combined length/nuisance distribution shift. A future
clean length-causal claim requires matched distractor extensions, not these
independent source splits. Full census: outputs/native_aggregation_vlm/v1/
data_audit/audit_440524.json.


### 2026-09-10 — V1 post-training scale inspection (added during training, before main test outcomes)

Development underfitting and first-step pre-clipping gradient norms motivate a
small descriptive inspection, not a new trained arm. Sum/post_sum first norms
48.268/47.684 versus1.323hierarchical and0.571LoRA; these do not prove clipping
is causal. After all main runs finish, inspect every selected checkpoint plus
the frozen reference on the same4disjoint profile examples:2at N16,2at N64.
At the final prompt position record branch delta, ordinary attention output,
and exact decomposition delta=z+(delta-z), with
z=s_R W_up SiLU(W_q h+b), where s_R=R for sum-scaled arms and1otherwise.
Report norms, alignments, relative sizes, and raw first-token logits/NLL as
diagnostics only. The current hidden h already contains contextual evidence;
zero-read does not mean uninformed. No activation intervention or training is
performed, and tiny profiling data cannot establish accuracy/calibration.
A larger zero-read contribution in sum-scaled arms is a tentative diagnostic
prediction; magnitude alone does not establish harm or causality.

One B200,4CPUs,96GB,15minute cap, sequential model inspections, through Slurm.
Nominal maximum profile+sevenmain+frozen+inspection allocation7.564GPU-hours
remains below8. Outputs outputs/native_aggregation_vlm/v1/inspection.


### 2026-09-10 — retrospective P1/P2 likelihood-label clarification

The V1 scoring audit also checked the saved text P1/P2 source snapshots and
cached Qwen2.5-3B generation config. Those runs likewise used generated.scores
with inherited repetition_penalty1.05. Previously reported first-token NLL
values and NLL-based dev tie-breaking are generation-policy quantities, not
raw LM likelihoods. Exact predictions, selection decisions and numeric values
are unchanged. The text pilot report now states this distinction explicitly.


### 2026-09-10 — V1 first-wave completed outcomes (remaining normalized arms still training)

All four first-wave tasks completed. Sum exact atN8/16/32/64 is40/11/13/0
out of100each; post_sum46/12/12/1; hierarchical75/58/42/30; LoRA52/32/21/22.
V1.1 primary processing-order threshold FAILED: sum andpost_sum both13/200
=6.5%atN32/64, so the required5pp advantage is absent. In-range sum is25.5%
versuspost_sum29.0%. Hierarchical is72/200=36.0%atN32/64 versusLoRA43/200
=21.5%, with in-range66.5%versus42.0%. These latter values do not establish
a unique mechanism; global, mean andpost_mean are still running. No protocol
changes or training expansion follow this partial result. Full paired intervals
and V1.2/V1.3 await all seven arms.


### 2026-09-10 — inspection measurement refinement (before inspection execution)

Capture the ordinary attention output directly at o_proj before branch addition,
rather than subtracting branch output from the rounded patched output. This
avoids cancellation/rounding error in the norm denominator. Record the branch
addition rounding error separately. Hooks are observational and do not modify
model tensors, weights, inputs, or selected checkpoints. Scientific contrasts
and the four diagnostic samples are unchanged.


### 2026-09-10 — V1 practical screen outcome (mean complete; final two controls pending)

Mean exact is71/54/36/25out of100atN8/16/32/64:61/200=30.5%OOD,
versus hierarchical72/200=36.0%. Therefore V1.3 practical screen FAILED for
both candidates: sum already failed against post_sum; mean does not beat
hierarchical by5pp (it trails by5.5pp). Per the registered rule, do not expand
this configured recipe into additional training seeds. Complete post_mean,
global, the full CPU paired report and the already registered diagnostic.
The mean-vs-post_mean secondary contrast still awaits post_mean completion.


### 2026-09-10 — V1 final outcomes, diagnostics and stop decision

All seven main tasks440504 completed, frozen reference440519 completed, CPU
report440509/440653 completed, and observational inspection440563 completed.
Final exact N8/N16/N32/N64 per100: sum40/11/13/0; mean71/54/36/25;
post_sum46/12/12/1; post_mean75/62/47/33; global70/55/47/34;
hierarchical75/58/42/30; LoRA52/32/21/22; frozen24/22/16/16.
All three criteria FAILED. V1.2 mean-post_mean OOD difference=-9.5pp,
paired95%[-15.5,-3.5], in-range difference=-6.0pp. V1.1 sums tie6.5%OOD;
V1.3 fails for both proposed candidates. Do not expand this recipe into more
training runs. The supplementary global-vs-LoRA difference is+19.0pp,
paired95%[12.5,25.5], but global40.5%andpost_mean40.0% differ by one answer.
Intervals remain conditional on one seed and unadjusted.

Global policy NLL is1.783OOD versuspost_mean3.847 andLoRA2.335. This is the
processed generation-policy quantity previously clarified, not raw LM NLL.
All normalized controls andLoRA parse every example. AtN64 sum parses0/100
andpost_sum5/100; the failures include malformed continuation strings. A
post-hoc first-character digit check yields12/100for both, which does not
replace registered exact metrics or rescue the processing-order claim.

The four-example checkpoint inspection found mean branch/ordinary norm ratios
atN64:sum47.67,mean1.40,post_sum4.13,post_mean1.13,global1.17,hierarchical1.25.
Sum-scaled zero-read and opposing read-dependent components are large; normalized
branches remain near ordinary-attention scale. This supports a scale/cancellation
concern, not a causal conclusion. Contextual h carries evidence; the zero-read
term is not uninformed. Raw forward logits in this diagnostic are distinct from
main generation-policy scores, and the tiny sample is not an efficacy estimate.

Total actual GPU allocation:12698seconds=3.5272GPU-hours (four profiles, seven
main arms, frozen reference and inspection), below8. All GPU/heavy CPU work used
Slurm. Final report: outputs/native_aggregation_vlm/v1/REPORT.md; interpretation:
docs/paper/NATIVE_AGGREGATION_VISION_RESULTS.md. No reasoning-model result,
unseen-count result, or new aggregation-bandwidth mechanism is established.


V1 report presentation update: CPU440653 and440691 regenerated the same deterministic analysis while correcting overlapping labels in the cost figure. Scientific values and decisions are unchanged; report provenance hashes were updated.


### 2026-09-10 — V2 clean data protocol (before generation or V2 model runs)

Gabriele authorized continuing toward the aggregation objective. The V1 data
semantics audit found a nuisance-distribution shift in canonical N16 train/dev
versus existing vision tests. V2 generates new data under
`/mnt/data/gabriele/gnn_transformer/v2_clean`; V1 and original data stay intact.
This entry specifies the data protocol only; model comparisons are registered
separately. Data seed20260911 is an identifier, not a generation date.

**One conditional generator law.** Uniformly choose a target from9characters and
6Park rooms. At every split, length N, and final count K, place exactly K matching
character/room sightings. Each of the N-K distractors independently chooses one
of three equally probable types: same character/wrong room, other character/same
room, or neither. Choose identities uniformly within the selected type. Uniformly
shuffle standalone sequences. This law keeps realistic conjunction confounds and
does not cap target-character occurrences at eight. Only final answer/EOS labels
supervise models; state metadata and nuisance labels serve data generation/audit.

**Fixed sizes.** Train90examples per N8,N16 (10 per K0..8), development36per N8,N16
(4per K). Familiar-count tests comprise108anchors atN16 (12per K0..8), each extended
toN32 andN64:324test sequences. Unseen-count tests comprise64anchors atN32 (8per
K9..16), each extended toN64:128additional tests. Main familiar/count manifests
therefore contain704total sequences, including452test sequences. A separate12-
example software profile contains2N16train,2N16dev,2N8test and2paired test anchors
atN16/N32/N64. All profile sequence/question content is disjoint from main/count.

**Paired extensions.** Draw new distractors from the exact same IID mixture and
uniformly interleave them into the shorter semantic sequence while preserving
its order. No new matching evidence is added. Record shared pair_id/anchor_id,
parent_n_frames, parent_positions, anchor_positions, and test_family in manifests.
Rendered step indices are renumbered, so the controlled addition also changes
positions and printed step numbers. Samples from one anchor are correlated;
analyses spanning lengths must resample or group whole anchor families rather
than treating their lengths as independent observations.

**Exclusions and integrity.** Reject complete generated families if any normalized
ordered-state-plus-question content duplicates a selected V1 main/profile sample
or another selected V2 sample. This excludes the664selected V1 sequences, not
all historical source populations; the all-matchN8 case has only54possible
character/room combinations. Retain the canonical512x512 renderer, sixPark-room
layout, fonts and character colors; require existing-image RGB pixel parity and
record renderer/helper/generator hashes, font paths/checksums and Pillow version.
Recount every generated gold and verify paired semantic subsequences before
publishing fixed manifests. QA files and PNGs carry SHA256 hashes. Repeated
character/room/step graphics use a checked immutable render cache with hardlinks;
this changes storage only, not model inputs. Existing files are never overwritten.

**Execution.** New script `scripts/stage_native_vision_v2_clean.py`, CPU Slurm
wrapper `slurm/native_aggregation_vision_v2_clean_stage.sbatch`:4CPUs,8GiB,10min cap
(within authorized30min maximum), no GPU. Expected716sequences/21,568logicalPNG
images and at most3,456unique rendered graphics, substantially below1GB. Publish
main_manifest.json, count_manifest.json, profile_manifest.json and audit.json
under the V2 root. Count examples retain split=test in separate exact manifests;
model analyses label them as a separate unseen-count axis and do not select
checkpoints or architectures using their outcomes.


## 2026-09-10 — V2 clean-vision attribution block (registered before GPU runs)

Motivation: V1 rejected the local-before-merge mechanism; global/post-mean gains
may come from ordinary middle-layer adaptation. V2 changes the dataset protocol
for all arms together; comparisons with V1 are descriptive, not controlled.
The clean generator and paired test construction are registered in the preceding
V2 data entry. No V2 GPU outcomes have been inspected at registration.

Four seed-0 arms: global rank64 plus upper4 q/k/v/o LoRA rank8/alpha16;
hidden-only U SiLU(Ah+b) rank96 at the same layer14 normalized input and output
residual site plus identical upper LoRA; layer14 q/k/v/o LoRA rank32/alpha64
plus identical upper LoRA; upper4 q/k/v/o LoRA rank16/alpha32 alone.
Expected active parameters: 1409088,1409120,1441792,1441792 respectively.
The global read is the existing dense softmax read, recomputed by the legacy
adapter; this is an adaptation control and does not establish new aggregation.

Common: frozen Qwen2.5-VL-7B-Instruct NF4/bf16/SDPA, 392px, layer14, block64,
querytile32, no fences/frame labels/external tally, native answer+EOS CE,
9 epochs on180 examples, accumulation4, branchlr1e-3, LoRAlr1e-4,
AdamW wd0, joint gradient clip1, seed0 and exact staged lists. All arms see the
same shuffled order and seed-reset upper LoRA initialization (rank8 arms).
Selection: pooled72-example in-range dev exact, tie lower raw first-token NLL;
no test selection, all examples remain in denominator, strict integer parse.
Inference: greedy native cached generate max4 tokens, explicitly repetition
penalty1.0; record raw first-token logits NLL (not full-answer likelihood).
This corrects the V1 inherited penalty/processed-score ambiguity prospectively.

Primary attribution criterion V2.1: global exceeds hidden-only by >=5pp pooled
N32/N64 familiar-count exact, with no >5pp loss at N16. Report paired bootstrap
95% intervals by shared anchor (resample 108 anchors, preserve both lengths;
10000 draws, analysis seed20260910). This is a screening effect threshold,
not a significance declaration. V2.2: global also exceeds each LoRA control
by >=5pp on the same OOD metric. Failure leaves ordinary adaptation as a
sufficient explanation; a winner is not by itself an aggregation mechanism.

Secondary: separate N16/32/64 exact, parse rate, parsed MAE/bias, perK outcomes,
paired prediction stability and correct-to-incorrect transitions under distractor
extension. Report unseen K9..16 separately at N32/N64 (64 paired anchors),
including K9 versus multi-digit K10..16 and output tokenization. Do not combine
these with familiar-count primary accuracy. No numerical extrapolation claim
unless unseen-count gains survive appropriate controls and replication.

Decision: if global clears both screens, replicate independent seeds and test
causal use of its read before calling it an aggregation improvement. If either
screen fails, prioritize channel-specific activation interchange and token-to-read
information probes before another operator grid. Reasoning composition remains
untested and is a later required objective, never inferred from plain accuracy.

Cost cap: this new block <=4 GPU-hours including profiles and retries; initial
four main tasks max50min each (<=3h20 reserved), four software profiles max5min
each (<=20min reserved). CPU staging/tests/reporting through Slurm. No unrelated
jobs are modified. Subsequent blocks require a new recorded scientific decision
and bounded budget within the user's continuing research authorization.


### 2026-09-10 — V2 multi-digit software profile (before V2 model profiles)

Add a separate profile_count_manifest.json containing two newly generated paired
anchors with K10 and K16 atN32, extended toN64 using the same V2 generator law.
These four examples are content-disjoint from all716previously generated V2
main/count/profile samples and all664selected V1 samples. Main/count/profile
manifests are unchanged. Purpose: verify multi-digit native decoding and scoring
without reading main held-out count-test outcomes. CPU Slurm only, then an
independent CPU audit of all generated QA labels, pair mappings and image hashes.


V2 pre-main software gate, 2026-09-10: CPU440738 passed all33tests; CPU440739 independently audited720samples. GPUprofiles440741_0..3 passed allfour arms (44,44,44,60s), finite native-loss updates, selected checkpoint restore, long inputs and multi-digit gold handling. No profile efficacy claim. Reviewer caveats: rank96 hidden and rank64 global match parameter count, not nonlinear width/input variance; LoRA and adapters retain different registered learning rates, so optimization-policy differences remain. All main settings unchanged.


### 2026-09-10 — V2 causal-interchange diagnostic data (before generation/probe outcomes)

Generate16new matched pairs atN32 under the V2 law. Each pair begins with a
K2sequence, then uniformly selects4negative positions and replaces only those
sightings by the queried character/room conjunction, producing K6. Keep the same
question/target and all28other frames exactly unchanged, including their rendered
step labels. This coupling preserves uniform positive positions and IID remaining
negative sightings in both marginals. Record four zero-based changed_positions,
pair_id, and low/high standard image/QA-hashed records in a separate
`v2_clean/causal_probe_manifest.json`; directories use seq_len_32/probe.

Exclude normalized sequence/question content from all664selected V1 examples and
all720V2 main/count/profile/profile_count examples, and across the new32samples.
Use the unchanged canonical renderer. Recount every gold, verify exactlyfour
semantic and pixel differences per pair, and confirm existing main/count/profile
manifest hashes remain unchanged. CPU Slurm4CPUs/8GiB/5min maximum, no GPU.

These examples are diagnostic only, for descriptive artificial branch-channel
interchange in both donor directions with identity and norm-matched controls.
They are not training data, model-selection data, or additions to the registered
performance-test sets. Probe-specific model methods are documented separately.


### 2026-09-10 — V2 frozen native-read recoverability diagnostic (before feature harvesting)

Diagnostic only; this does not modify or select the four running V2 models. Harvest exactly the existing main/count manifests: 180 training, 72 development, 324 familiar-count test, and 128 unseen-count test samples (704 forwards). Use frozen Qwen2.5-VL-7B-Instruct, nf4/bf16, ordinary causal SDPA, canonical images-first count prompt, 392px images, native positions, no branch, no LoRA, no carrier or extra prompt tokens. Capture only the final prompt token at decoder layer index 14: h is the attention-input state after input layer normalization; r is the concatenated SDPA head output immediately before o_proj. These correspond to the frozen inputs of the V2 hidden/global branches, subject to arithmetic differences between native SDPA and the global branch's segmented fp32 reconstruction. Store both 3584-dimensional vectors as fp32 under /mnt/data/gabriele/gnn_transformer/v2_recoverability with exact record ordering, prompt-layout metadata, source/manifest hashes, and code snapshots. No generated answer or test-selected feature is required.

CPU ridge fits use the 180 training rows only, training-only feature centering/scaling, target centering with an unpenalized intercept, and dual ridge. Representations: h, r, concatenated [h,r], each with and without scalar N (feature-standardized by the same training-only rule); N-only and constant training-mean controls. Select alpha independently per fitted representation from [0.01,0.1,1,10,100,1000] by pooled 72-row development MAE only, breaking ties toward larger alpha. No clipping to the training count support, test refitting, or test-driven choice of representation. Primary descriptive metrics are continuous MAE and R-squared, with rounded-integer exact accuracy secondary. Report each original split/N, familiar-count versus unseen-count families, and unseen K=9 separately from multi-digit K=10..16, plus per-K results. Paired N16/32/64 or N32/64 rows remain identifiable by pair_id; pooled rows are descriptive correlated observations, not independent inferential replicates. Save predictions/configuration/metrics, not fitted weights.

This addresses linear count accessibility in two frozen ordinary-causal VLM channels on the clean V2 distribution. Success is not a native-output algorithm; failure does not establish information destruction or absence of nonlinear codes. A read-state gap does not by itself establish intact token-level character/room binding. Earlier text-needle count probes and historical VLM frame-message probes concern different inputs/layers/data and are supporting context, not this measurement. GPU harvest is capped at 20 minutes on one GPU; fit and synthetic hook/ridge checks use CPU Slurm (4 CPUs/16GB, 5 minutes), within the already authorized V2 4-GPU-hour block. No additional GPU profile is requested.

Recoverability diagnostic addition before harvesting: retain the already-computed frozen native first-token top-1 ID/text and raw logits for verified single-token numerals 0..9. This costs no extra forward or generation and is descriptive first-token evidence only, not whole-answer exact accuracy; K>=10 requires further digits.

### 2026-09-10 — V2 final-query channel interchange diagnostic (before probe execution)

Diagnostic only, not training or a proposed inference method. Use the V2 global arm's development-selected checkpoint, with strict architecture, selected-epoch, source-hash and LoRA-state checks. Dedicated data manifest causal_probe_manifest.json SHA256 ca76155dbc5b300398b9380a07ba400126ba00f2c300403b892cbc8b8aee78f5 contains 16 N32 pairs, same question, K2/K6, exactly 4 replaced images and 28 identical images. No model-outcome-based selection. Evaluate both directions of every pair; --limit-pairs 16 is the primary run, smaller manifest-order prefixes are correctness-only smoke runs.

Capture the inputs to the global branch's query_down (normalized hidden input) and read_down (existing global attention read) at the final prompt query. Pre-hooks track each projection's query-chunk offsets independently and replace only that row. Ordinary image encoding, QKV, SDPA, residual path and all other queries remain unchanged. Main diagnostic uses uncached full forwards and raw native vocabulary logits, with no generation penalties or answer tokens in the input. CPU tests verify query-row locality, hook cleanup, identity and full/cached final-query parity in Qwen2/Qwen3/Qwen2.5-VL.

For each direction run identity-both, donor-hidden, donor-read, donor-both, and recipient-norm-matched versions of each donor condition. Norm matching scales each donor channel vector to its recipient norm; reject zero/nonfinite norms. Two unmodified context captures plus 14 interventions per pair gives 256 forwards. One B200, 15 minute cap; no optimizer/checkpoint writes. Store all 32 directional baselines and 224 intervention rows.

Estimand: within-recipient change in raw logit margin z(donor answer)-z(recipient answer), alongside probabilities, argmax, norms, full-context contrast and maximum change over the entire vocabulary. Identity must pass full-vocabulary allclose(atol 0.03125, rtol 0.001), not merely preserve this two-token margin; record maximum identity difference and abort interpretation on failure. Two directions within one pair are dependent; summaries are descriptive and no significance or seed-generalization claim is made.

Falsifiers/interpretation: a read-channel swap that fails to shift the margin toward the donor, including after norm matching, gives no positive evidence that this branch read mediates the tested count contrast. A larger hidden-only effect is compatible with ordinary adaptation of contextual hidden states. Effects only under joint swaps suggest channel interaction. Even a positive read effect demonstrates use of an existing attention statistic, not additional aggregation bandwidth. Artificial mixed-channel states may lie outside the training distribution; these interventions do not establish a new counting algorithm or improvement in native accuracy.


### 2026-09-10 — Frozen vision question-prefix diagnostic (before CPU plans or GPU outcomes)

Select the first two lexicographically sorted V2 familiar-test pair_id anchors
for each K0..8, using both N16 and N64:18 anchors,36 examples. This is a fixed
diagnostic subset of the main test and does not select models, methods or prompts.
Three conditions: original images-first + canonical final question; correct
question prefix before images + identical final question; neutral prose prefix
before images + identical final question. The correct prefix is exactly
"Question: {question}\n". Neutral prose comes from the fixed script constant;
CPU tokenization truncates at whole-word boundaries and adjusts punctuation to
match the correct prefix's actual chat-template token layout, before generation.
All36 full processor runs must match correct/neutral total tokens, image-token
spans and vision-marker positions. Image grids and processed pixels must match
across allthree conditions. Prefixes/layouts and hashes are frozen in a CPU plan
which the GPU run verifies exactly; no response-based adjustment is allowed.

Frozen Qwen2.5-VL-7B-Instruct NF4/bf16/SDPA,392px, native cached greedy generation,
max4 tokens, repetition penalty1.0, unrestricted vocabulary, strict whole-integer
exact with every example in the denominator. Fixed rotating condition order.
Report all108 outputs; perN exact/parse and raw first-token NLL; correct-prefix
minus neutral and baseline, perN and pooled, with10000 paired-anchor descriptive
bootstrap draws (seed20260910). No formal success threshold or inference of
reasoning composition: this tests whether early availability of the question
helps the frozen direct-answer model. Correct/neutral semantics and repetition
differ; baseline also differs in length/positions. A null contrast provides no
positive evidence for this prompt-level repair, not proof against query-value
computation. No further prefix search from these outcomes.

GPU cap1 B200×10minutes. Fits within the existing4GPU-hour block: main tasks2/3
completed in1691/1622seconds; even remainingtasks0/1 at their full50minute caps,
plus192profile seconds,15min channel probe,20min recoverability, and10min prefix,
reserve at most12205 GPU-seconds (3.40h). CPU layout checks use Slurm. No new
models/checkpoints or data modifications.


### 2026-09-10 — V2 completed outcomes and next decision (after V2 outcomes, before V3 runs)

V2 main array440749 completed allfour arms. Familiar-count N32/64 exact:
global58/216=26.85%, hidden63/216=29.17%, middle-LoRA47/216=21.76%,
upper-LoRA35/216=16.20%. Global-minus-hidden=-2.31pp, paired-anchor95%CI
[-10.19,+5.09]pp. V2.1 FAILED. V2.2 passed its >=5pp effect-size screen against
both LoRA controls, but the middle-LoRA contrast CI crosseszero; this is not a
significance claim. Exact perN, perK, checkpoint/source verification and caveats:
outputs/native_aggregation_vlm/v2/REPORT.md and analysis.json.
Native global correctly emitted17/128 unseen-count answers, all multi-digit
(17/112 K10..16; 0/16 K9). This contradicts a universal unseen-number emission
barrier, without establishing accurate numerical extrapolation.

Causal last-query-only branch interchange440805:16pairs,32dependent directions,
256forwards; entire-vocabulary identity error exactly0. Mean donor-read margin
shift -0.01025, norm-matched read -0.00439; donor-hidden -0.00195, both -0.00293.
No positive evidence of the preregistered donor-directed read effect. This does
not test all earlier prompt queries or prove that the channels are unused.

Prefix diagnostic440808: correctprefix and baseline each5/36 exact; neutral4/36.
Correct-minus-baseline0ppCI[-11.11,+11.11]; correct-minus-neutral+2.78pp
CI[-8.33,+13.89]. Null prompt-level repair; no further prefix search.

Frozen feature harvest440791/CPUfit440792: all704rows complete. Ridge dev-MAE
selection chose alpha1000 for every nonconstant representation. FamiliarOOD
MAE h2.1886, r2.1535, h+r2.0652, constant2.2222. K10..16 MAE h6.5616,
r7.5832, h+r7.2701, constant9.0. Nfeatures do not materially help. This is
poor linear count extrapolation, not proof of lost information or intact local
binding. Reports under outputs/native_aggregation_vlm/v2/recoverability.
Total V2 GPU allocation8299seconds=2.30528hours, within4hourcap:
profiles192 + main6991 + channel263 + prefix158 + harvest695.

Decision: retire the V2 extra-read mechanism claim. Test one prior-art-inspired
change in what values are pooled, with a matched nonlinear-placement control.
No further V2 operator or prefix tuning. This is a new exploratory block under
the user's continuing authorization, not a claim that the objective is met.

### 2026-09-10 — V3 native-width value-lifting comparison (before CPU/GPU outcomes)

Hypothesis: a nonlinear map of individual native values before normalized
attention pooling preserves task-relevant statistics better than the same
nonlinearity applied after pooling, at equal trainable parameter count.
This is inspired by nonlinear-value attention/Deep Sets, including QVI and
HYLA; it is not a novel attention family, a literal output-rank increase, or
a cardinality-preservation guarantee. Softmax normalization remains unchanged.

At layer14 of frozen Qwen2.5-VL-7B, let v_j concatenate the four native128D KV
heads. Both arms learn the same residual affine lift z_j=v_j+C(A v_j)+b,
A:512->8, C:8->512, with C and b zero-initialized. Unpack z into native heads.
PRE reads each head with existing native attention weights alpha using
sum_j alpha_tj SiLU(z_j); POST uses SiLU(sum_j alpha_tj z_j). Both feed the
same outer residual U SiLU(W_r read + W_h h + b_r), rank64, U zero-initialized,
added after native o_proj. Each arm has696896branch parameters and720896upper
LoRA parameters, total1417792. Same tensors and initialization per pairedseed;
only inner SiLU placement differs. The common outer initialization order matches
V2global; upper LoRA initialization resets the same seed. The frozen/native
attention path is retained. One extra full-width SDPA read per selected layer;
native QKV/RoPE/KV-cache update only once. Native cached V is re-lifted each
forward; no transformed-feature cache is implemented or claimed.

Four main runs: PRE/POST x seeds0,1, parallel up to4B200 GPUs. Same V2 clean
main/count manifests, images-first canonical prompt,392px, nf4/bf16/SDPA.
Train180examples atN8/16,K0..8; dev72; paired familiar test108anchors atN16/32/64;
separate unseen64anchors atN32/64,K9..16. Eachseed changes initial weights and
training shuffle, paired across PRE/POST; fixed data remain identical.
Nine epochs, accumulation4 (45steps/epoch), AdamWwd0, branchlr.001,
upper4-layer LoRA rank8alpha16 lr.0001, clip1. Answer+EOS native CE only.
Greedy unrestricted cached generation, max4tokens, repetition_penalty1.0.
Select highest pooled dev exact, ties lower raw first-answer-token NLL,
then earliest epoch. No test selection, frame supervision, fences, carriers,
external tally, added inference tokens, or count-dependent rescaling.

Primary screen V3.1: PRE exceeds POST by >=5percentagepoints on pooled familiar
N32/64 exact, with no >5pp loss atN16, IN EACH of the two seeds. Report actual
perseed differences and10000paired-anchor bootstrap95%intervals(seed20260910).
A pooled descriptive estimate resamples the same anchors across bothseeds
jointly; two seeds are not sufficient for population inference over seeds.
Secondary screen V3.2: PREseed0 exceeds the historical same-data V2hidden seed0
by>=5pp on familiarOOD, with no>5ppN16loss. Historical implementation/run-time
comparisons have limitations; same-size ordinary adaptation remains essential.
Report every original cell, perK, parse rate, MAE/bias, rawfirsttokenNLL,
paired extension stability, latency/memory and exact parameter counts.
UnseenK9 and K10..16 remain separate secondary outcomes, no count claim from
familiar-count length extrapolation. Profile examples are disjoint software
checks, not selection data. Do not expand activation/rank/layer grids after
outcomes. Failure directs renewed measurement, not another arbitrary variant.

All V2 test outcomes have already been examined. V3 is exploratory reuse of
these test examples; preregistration does not make them untouched. A positive
screen must be confirmed on freshly generated held-out data and additional
seeds before broader efficacy claims. Reasoning composition remains an explicit
later objective and cannot be inferred from this direct-answer experiment.

Software gate: CPU Slurm tests for linear-placement equivalence, zero-output
parity, causality/padding/fullymasked rows, native CE gradients, full/cached
parity and exactlyoneKVupdate on Qwen2/Qwen3/Qwen2.5-VL. Then twoGPUprofiles,
one perarm, on V2's separate profile/profile_count manifests, oneupdate plus
checkpointrestore/longinputs/multidigit gold handling. Only proceed if all pass.
Cost cap: NEW block <=4GPUhours, including profiles and retries. Main fourtasks
max50min each + twoprofiles max5min each reserve3.5GPUh. CPU tests/reporting
throughSlurm. Models /mnt/ckpts/gabriele/gnn_transformer; data unchanged under
/mnt/data/gabriele/gnn_transformer. No unrelated jobs touched.

V3 software gate before GPU profiles: CPU440822 passed7new value-lifting test methods plus23native tests in13s. Independent code review found no blocking correctness issue. Parameter-matched arms have different activation work under GQA: PRE applies SiLU to KV values, POST to expanded query-head reads; PRE redoes the value SiLU over cache during decode. Latency must be measured; no exact FLOP matching claimed. Full-width SDPA uses bf16 inputs while lift/outer layers are fp32, so activation placement also changes where rounding occurs. Profiles proceed on disjoint software examples.

V3 GPU software profiles440826_0/1 completed38/43seconds. Both passed native-loss update, checkpoint restore, N64 and multi-digit gold evaluation;1,417,792trainable parameters each. No efficacy conclusion from these disjoint software examples. Main training launches under unchanged registered protocol and frozen source hashes.

### 2026-09-10 — V2 marginal-shortcut audit (post-hoc design, before audit outcomes)

While V3 trains, audit the clean generator's remaining statistical shortcut.
This is post-hoc after examining V2 model outcomes; it does not change V3 or
supply a competitive model. Use all704existing main/count records, independently
QA-hash-check/recount their semantic categories, without loading images or model
predictions. C=#target-character frames, R=#target-room frames, N=totalframes.
Under the nominal IID-negative law, E[C]=E[R]=N/3+2K/3. Fixed oracle references:
0.75(C+R)-0.5N, 1.5C-0.5N, and1.5R-0.5N. No fitting, clipping, or tuning.
Their nominal conditional variances are(N-K)/8,(N-K)/2,(N-K)/2; finite content
exclusion/rejection conditions can slightly change these nominal statements.

Report continuous MAE/bias/MSE and secondary floor(x+0.5) exact per original
cell/perK, familiarOOD and separate unseenK9/K10..16. Enumerate small synthetic
category sequences to verify formulas, and show equal marginal counts can have
different conjunction counts. Do not infer that a VLM uses this shortcut from
oracle-reference accuracy, or conflate oracle count access with visual inference.
Paired anchors remain dependent; descriptive audit only. CPU Slurm2CPUs/4GiB,
5minutes max, zero GPUs. Source snapshots and new outputs only under
outputs/native_aggregation_vlm/v2/marginal_audit. Existing data stay unchanged.

2026-09-10 marginal audit outcomes: CPU440844 completed2seconds, all704QA/gold/
semantic checks and3279synthetic category configurations passed. Joint oracle
reference rounded exact37/216=17.13% on familiarOOD,28/128=21.875% on unseenK;
MAE2.115 on familiarOOD,1.176/2.078 at unseenN32/N64. This quantifies a data
shortcut using oracle marginals and the known generator, not VLM accuracy or
proof of model reliance. Detailed separate report: v2/marginal_audit/audit_440844.

### 2026-09-10 — Marginal-matched visual binding diagnostic (before staging/model outcomes)

Purpose: distinguish sensitivity to conjunction changes from dependence only on
individual character/room totals. This is a new input-counterfactual diagnostic,
not training, model selection, or an untouched same-law confirmation. It follows
the post-hoc sampling-law audit; V3 main protocol/checkpoints remain unchanged.

Generate16new N32 pairs with K2(low) andK6(high), fixed target-character count
C=12 and target-room countR=12 in both. Low contingency counts are
(matches,C-only,R-only,neither)=(2,10,10,10), uniformly shuffled, with uniform
valid non-target identities and target uniformly from9characters x6rooms.
Select4C-only and4R-only positions, pair them and swap room assignments:
(C,R')+(C',R) becomes(C,R)+(C',R'). Character identity at EACH position and the
FULL room-frequency vector remain identical. High contingency is(6,6,6,14).
Exactly8renderedframes differ;24pixel-identical frames remain. Source positions,
question, step labels, full character/room marginals and sequence length match.

New root /mnt/data/gabriele/gnn_transformer/v3_binding_probe. Exclude full
sequence/question content from V1/V2selected main/count/profile/profile_count/
causal data and within these32new examples. Use canonical renderer/new cache;
existing data unchanged. Manifest records pair/condition/changedpositions and
allQA/image hashes. CPU Slurm verifies labels, margins, exactchangedpositions,
image hashes and8pixel differences before any model runs. This fixed-marginal
law is intentionally different from V2's IID-negative mixture; do not pool it
with the main efficacy test or interpret any loss as pure distribution-free
binding incapacity.

Evaluate selected V2global/hidden seed0 and V3PRE/POST seeds0/1, after their
registered reports verify development-selected checkpoint provenance. Six
sequential frozen evaluations on oneGPU,32examples each, using the unchanged
respective native harnesses in eval-only mode. NF4/bf16/SDPA,392px, images-first
canonical question, layer14, same branch/LoRA settings, cachedgreedymax4,
repetition_penalty1.0, unrestrictednativevocabulary. No fitting, output correction,
new prompts, answer-prefix hints, or choice of condition from outcomes. Snapshot
plan/code/data/checkpoint hashes and verify checkpoint bytes before/after.

Descriptive outcomes permodel: low/highexact andMAE with parse counts; both
answers correct perpair(/16); mean prediction(high)-prediction(low), absolute
contrast error againstgoldDelta4, fractionpositive contrasts and identical
predictions. Unparsable outputs countincorrect; numericcontrast metrics report
their validpair denominator. No significance or universalmechanismclaim from
16pairs. A fixed-marginal oracle estimate2 gets50%exampleexact but0%both-pair
exact and contrast0; pooledexampleaccuracy alone is therefore insufficient.

Resource allocation: CPU staging/audit <=5minutes,2-4CPUs and<=8GiB; GPU probe
<=10minutes on oneB200, after V3 main/report and data audit. This remains INSIDE
V3's4GPUhourblock: four main50mincaps=12000seconds +81completed profile seconds
+600bindingprobe seconds=12681seconds(3.5225h), leaving1719seconds for retries.
All models/data use user-requested roots; reports outputs/native_aggregation_vlm/v3.

Binding diagnostic data implementation before staging: seed20260913, SHA-derived with label v3_binding_fixed_marginals_K2_K6;16pairs/32examples/1024frames. Four uniformly paired room swaps, existing content excluded. Independent publication audit rehashes QA, PNG bytes and RGB pixels, recounts categories and verifies character identity at every position/full room marginals. CPU4/8GiB/5minutes.

Binding data CPU440860 completed10seconds;16pairs/32samples/1024images audited. Exactly8imagechanges perpair, positionwisecharacteridentity and fullroommarginals preserved, allcontentexclusions passed. Frozenmanifest SHA256 276492ee34b2622a402e7ada0a06b4732451a4f963ae1f4441f3a20508303c93.

### 2026-09-10 — Cosmos reasoning-VLM compatibility check (before loading/outcomes)

A software gate only, not reasoning efficacy or composition. Local cache metadata
and allfour shard names for nvidia/Cosmos-Reason1-7B snapshot
3210bec0495fdc7a8d3dbb8d58da5711eab4b423 were independently verified. Its official
model card identifies it as a reasoning VLM; config uses Qwen2_5_VL architecture
with the same language dimensions/M-RoPE as the current implementation. Shard
loading and actual execution remain to be checked.

Use the unchanged V3 harness with model=Cosmos, PRErank64/liftrank8 plus upper4
LoRArank8alpha16. One answer+EOS-loss update on the two disjoint V2 N16 software
training examples, two software dev examples, select/restore that only checkpoint,
then native cached generation on the four two-example long/multidigit software
cells. NF4/bf16/SDPA,392px,canonical images-first prompt,max4tokens and
repetition_penalty1.0 as before. Save tiny adapter checkpoints under the requested
/mnt/ckpts/gabriele/gnn_transformer/native_aggregation_vlm_v3 root, reports under
v3/reasoning_compatibility, sources and model identity explicit. Do not merge with
Qwen efficacy results. Truncated short output/format correctness is not an
estimate of reasoning quality; this gate does not use a CoT prompting protocol.

OneGPU,max5minutes, after V3main software/data gates, can overlap binding eval.
Within V3cap4hours: priorworstcase12681seconds +300=12981seconds(3.6058hours).
No model download or installation; existing cached base weights stay unchanged.
Only if this passes is native execution on this reasoning backbone established;
composition and measured reasoning benefits still require separate experiments.

Cosmos software job440874 failed before model loading after10seconds: offline repo-ID resolution did not find its config despite locally present snapshot files. Retry uses the explicit inspected local snapshot path, preserving existing weights and all experiment settings; CPU config/processor gate first. No model efficacy outcome exists. Original failed run/log preserved; count10seconds against V3budget.


### 2026-09-10 — V3 main outcomes (after registered main/report)

Array440830 allfour tasks complete, GPU seconds1682/1679/1711/1711(total6783).
CPUreport440835 independently verified all sources/data/checkpoints/selection and
paired452testexamples/model. PRE/POST familiarOOD seed0:78/216 vs63/216,
gain+6.944pp95%CI[-0.46,+14.35]; N16gain+1.85pp, perseed screen passes.
Seed1:44/216 vs44/216,gain0ppCI[-7.41,+7.41],N16gain-2.78pp, screenfails.
V3.1 FAILED. Pooledfixed-seedgain+3.47ppCI[-2.31,+9.03] does not rescue it.
V3.2 passes PREseed0 versus historicalhidden: +6.94ppOOD,+9.26ppN16. Historical
reference and reusedtest caveats remain; no seed-robust newarchitectureclaim.
UnseenK9..16 correct PREseed0/1=19/17 of128,POSTseed0/1=4/18; allK9wrong.
No general count algorithm or reasoningcomposition established.

Both methods have1417792trainable parameters; measured long-input model time
about1.05-1.07s/example, trainingpeak13.69/13.73GiB. Canonicalreport and PNG/PDF
under outputs/native_aggregation_vlm/v3; figure visuallychecked without changes.
Continue the registered fixed-marginal diagnostic, not fresh efficacy expansion.

Cosmos explicit-snapshot CPU440888 failed7seconds: slow Qwen2 tokenizer lacks
vocab.json/merges.txt; its exception additionally mentions missingprotobuf.
No installation is requested or performed. A CPU-only compatibility mirror using
exact tokenizer.json BPE assets and fast/slow parity is being prepared; existing
weights, sourcecache and mainexperiment code stay unchanged.

### 2026-09-10 — Fixed-marginal diagnostic outcomes and precision question

GPU440886 completed345seconds, all192outputs parsed, six selected checkpoints,
source/data/processor/package checks passed. Every model increased its prediction
on all16pairs; no invariant predictions. Mean high-minus-low changes:
V2global4.375,hidden4.250; V3PREseed0/1=4.1875/4.8125,
POSTseed0/1=3.750/4.375, versusgold4. SceneMAEs respectively
.875,.9375,.71875,1.53125,1.0,1.25; both-pair-correct0,1,0,0,2,0 outof16.
The models respond to changed joint structure despite fixed complete marginals;
a purely marginal-only explanation is incompatible with these outputs. This is
not a clean absence-of-binding result. Exact count precision remains weak, and
input swaps do not isolate a particular internal circuit. Canonical report:
v3/binding_probe/probe_440886/REPORT.md. No fresh efficacy expansion triggered.

### 2026-09-10 — Per-frame judgment versus aggregation diagnostic (before CPU/GPU outcomes)

Follow the observed coarse count sensitivity with a narrower behavioral question:
are individual conjunction judgments precise when counting is removed, both in
isolation and amid the same full context? This is diagnostic only, no fitting or
new method output. Use frozen Qwen2.5-VL and the FIXED V2hidden seed0 checkpoint,
not a post-hoc V3winner. Canonical data are the16audited fixed-marginal pairs.

For each pair select the firsttwo entries of room_swap_pairs: their two C-only
source positions and two R-only source positions. Also select the two unchanged
original-positive positions in original frame order. Evaluate allsixpositions in
bothlow/high scenes:12judgments/pair, with exactly6Yes/6No. C-switchpositions
are No->Yes; R-switchpositions No->No; originalpositives Yes->Yes. Do not replace
positions based on model behavior. Two presentations perjudgment: original
selected image alone, or all32originalimages. Preserve each original printed
Step label; do not renumber the isolated image. Same finalquestion byte-for-byte:
"In frame {one_based_index}, is {character} in the {room}? Answer Yes or No."
No gold categories/counts/pair IDs in prompts. Assert imageSHA/reference-index
identity and low/high question identity. Total192queries/presentation/model,
384/model,768across frozen+hidden. Repeated unchanged isolated images are planned
controls within clusters, not independent evidence.

CPU Slurm freezes selection, allprompt/template metadata, sources, bindingdata
and hiddencheckpoint hashes before outputs. Verify exact canonical continuations
"Yes" and"No" are one token each under the actual chat-template suffix and
preserve prefix-token boundaries. Abort on unsupported tokenization; do not
select spellings after model outputs. Frozen/native processor,392px, NF4/bf16,
SDPA,unrestrictedcachedgreedy generationmax4tokens,repetition_penalty1.0.
No model weights or inference logits are changed. Reuse first-generation-step
rawlogits for diagnostics, avoiding a duplicate fullforward.

Primary descriptive statistics: m=z(Yes)-z(No); sign correctness with tieswrong;
gold-signedmargin; unconditionalgoldtokenprobability; A=P(Yes)+P(No) over the
fullvocabulary. Conditional sigmoid(m) alone can conceal negligible answer mass.
Also retain unrestricted top1token and rawgeneration/IDs. Secondary exactparser
accepts stripped case-insensitive whole Yes/No only; explanations, otheranswers
or truncatedoutput remainincorrect, reportedseparately with all denominators.
Count-SFT can affectanswerformat; poor generationalone is not failedbinding.
Likelihood discrimination with tiny answer mass is not a repaired nativesuccess.

Report permodel/presentation/stratum signaccuracy, generationexact/parse,
signedmargin andanswermass. Pair full-minus-isolated for thesameimage/question.
For Cswitches report m_high-m_low; for R-switches andoriginalpositives report
label/prediction stability. Bootstrap10000whole pair clusters(seed20260913),
retaining all24queries/pair and crosspresentation/model correspondence. This
is descriptive uncertainty conditional on fixedmodels/16pairs, not a new
confirmatory inference or seed-population claim.

Interpretation: weak isolatedjudgments implicate perception/binding or interface;
strongisolated andweakindexedfullcontext implicate contextualaccess/selection or
referencehandling; strongcontextjudgments withnoisycounts motivate aggregation/
readout-precision measurement. These are not mutuallyexclusive causal diagnoses,
and indexed access is easier than simultaneous aggregation. No externaltally
will be presented as method output; no frame labels supervise model training.

CPU planning/tests <=5minutes,2-4CPUs/8GiB; GPU <=15minutes ononeB200, two models
loaded sequentially, actualvisionforwards onlyinsideSlurm. Within V3cap4GPUh:
completedprofiles81+main6783+failedCosmos10+binding345=7219seconds. Add atmost
300secondsCosmosretry and900secondsframejudgments =>8419seconds(2.3386h),
leaving substantial retry allowance. Record all real GPU allocation time.


### V3 frame-judgment operational correction — 2026-09-10 17:59 UTC, before CPU/GPU launch

The CPU planning wrapper requests four CPUs and 16 GiB for the ordinary full-image processor calibration, with the same five-minute cap. This replaces the preliminary 8 GiB estimate; GPU allocation and all scientific protocol choices are unchanged. Code review corrected the bootstrap implementation seed to the registered 20260913 before source freeze or any outcomes.


### V4 training-diversity control — 2026-09-10 18:05 UTC, before staging or training

Motivation: V2's extra global read failed against hidden-only adaptation, and V3 PRE versus POST failed its two-seed criterion. Every tested model increases its prediction on all sixteen fixed-marginal binding pairs, so wholesale missing conjunction sensitivity is not supported. Limited training diversity is a competing explanation worth testing before adding another operator. This is an ordinary supervised-training control, not a proposed novel aggregation method.

Four runs cross conditions `repeat` and `refresh` with fixed seeds 2 and 3. Use the exact V2 hidden-only rank96 adapter at layer14 plus upper-four-layer LoRA rank8/alpha16, frozen Qwen2.5-VL-7B NF4/bf16, ordinary SDPA and processor, 392px images, canonical images-first counting prompt, native final-answer/EOS CE, branch LR0.001/LoRA LR0.0001, AdamW weight decay0, gradient clipping1, accumulation4. No extra reasoning tokens, frame labels, fences, external tallies, or inference operations beyond this existing control.

Both conditions have nine blocks of180 presentations (90N8/90N16, ten per K0..8 per length), 45 updates per block,405 total updates,1620 presentations and19440 training frames. Retain optimizer state throughout. Shuffle the same180 slot indices with Random(seed+block) in both arms. A slot's N, K, question text and target character/room are identical across arms and blocks. Block0 is the original V2 training set in both arms; repeat reuses those180 scenes in all blocks. Refresh replaces every nonsaturated slot with a new scene in each later block, from the exact V2 conditional distractor law. The ten N8/K8 slots are deterministic at fixed question and necessarily repeat in BOTH conditions. Thus refresh has1540 distinct whole sequences, not1620, versus180 for repeat. No repeated content in the1360 newly generated nonsaturated scenes, and no collision with any prior V1/V2/profile/causal/binding or current dev/test content. Existing original training content is an intentional shared control. Preserve original renderer and final target support; never overwrite prior data.

Stage a new dataset under /mnt/data/gabriele/gnn_transformer/v4_diversity, hardlinking immutable source images where appropriate and copying/auditing QA. Use data seed20260914 and slot/block-derived stable seeds. Copy existing V2 dev/test and unseen-count sets byte-for-byte into the new root. This retains their status as reused exploratory evaluation; no result from this block is fresh confirmation. Freeze a schedule manifest enumerating all slot/scene assignments and source/content/image checksums. A CPU Slurm audit independently recounts all examples and verifies schedules, original-source identity, all disjointness exceptions, and every image hash. Check actual processed prompt-token equality for matched training slots; image pixels change but N/question/template/token layout must not.

Keep the original dev exact / raw first-answer-token NLL / earliest tie selection after every45 updates. Evaluate selected checkpoints on all216 familiar-count N32/N64 scenes; N16/108 and unseen K9..16/128 remain separate. Main criterion: refresh minus repeat familiar OOD exact at least5 percentage points in EACH seed, with no more than5 points N16 loss in either seed. Failing either seed fails the primary screen. Report each seed, pooled fixed-seed contrast, all invalid answers as wrong, paired10,000 family bootstrap intervals seed20260914, count MAE, N16-to-long deterioration and both-members-correct/invariance. Two seeds do not estimate population seed variance. Historical V2/V3 rankings are descriptive, not matched controls. Gains establish an effect of scene refresh under this training law and budget; they do not establish a new aggregation architecture, a general ability gain, or reasoning composition.

Software: retain frozen V2/V3/runtime sources. A separate V4 runner may extend V2 only with explicit block schedules, initial-parameter hashes and training-order/token ledgers; no optimizer, loss or forward-path changes. Compare initial parameter hashes across conditions within each seed. A small two-block GPU profile must exercise refresh, state persistence, native cached generation, N64 and multi-token target paths before the main array. CPU validation and main report independently check nine complete blocks,45 updates each, schedule hash, parameter counts, checkpoint metadata and all452 prediction identities. Checkpoints exclusively under /mnt/ckpts/gabriele/gnn_transformer/native_aggregation_vlm_v4.

Resource limit: new V4 block at most4 GPU-hours, at most4 B200 allocations concurrently across all ongoing research jobs, main four tasks capped50minutes each and profiles at most5minutes each. Expected main cost about1.9 GPU-hours from the prior measured runtime, plus modest profile headroom. All rendering, heavy audit/reporting and every model operation run through Slurm. Do not expand this into a layer/rank/learning-rate grid if the primary screen fails. Counterfactual CE and consistency losses remain contingent, separately registered proposals.


### Cosmos tokenizer compatibility mirror — 2026-09-10 18:08 UTC, before repair job

The cached original Cosmos-Reason1-7B snapshot has tokenizer.json but lacks vocab.json/merges.txt required by the unchanged runtime's slow Qwen2 tokenizer. CPU440888 failed before model weights loaded. Create a NEW compatibility mirror under /mnt/ckpts/gabriele/gnn_transformer/cosmos_reason1_compat_JOB, copy the original processor/tokenizer/model metadata, symlink the four original immutable weight shards and preserve the original shard index. Derive the missing vocabulary and ordered merge files exactly from the ordinary byte-level BPE tokenizer.json; do not change token identities or model weights, install packages, download replacements, or overwrite the original cache.

CPU Slurm gate (2 CPUs,8 GiB,5 minutes) must validate BPE format, full vocabulary and added/special-token identities, deterministic ordinary/unicode/count/chat-prompt tokenization equivalence between original fast and mirror slow tokenizer, ordinary AutoProcessor(use_fast=False) loading, and original/mirror chat-template identities. Record source and generated metadata hashes and verify weight shard presence/header/index consistency. Header/index checks are not a full weight-content checksum; the subsequent model load is still a software gate. Abort on any parity failure. Only after a passing gate may the existing V3 compatibility wrapper use the published mirror path, with all other arguments and its one-update/N64/multi-token software tests unchanged. The five-minute GPU retry remains within the existing V3 cap. This test says nothing about reasoning efficacy or composition.


### V3 frame-judgment outcomes — 2026-09-10 18:12 UTC

CPU440916 passed in9seconds; GPU440918 completed in461allocated GPU-seconds,768judgments. All frozen-plan/template/processor/data/checkpoint validations passed. Canonical report outputs/native_aggregation_vlm/v3/frame_judgments/judgments_440918/REPORT.md and summary.json.

Primary raw-first-token binary accuracy: frozen isolated192/192,full132/192(68.75%); V2hidden seed0 isolated192/192,full156/192(81.25%). Full-minus-isolated accuracy is -31.25pp for frozen (pair bootstrap[-38.54,-23.96]) and -18.75pp for hidden([-23.44,-14.06]). Mean gold-signed margins fall7.9643→1.0028 and9.5964→2.7910; mean full-vocabulary P(Yes)+P(No) remains above0.999 in every aggregate. Thus the binary measurement is not selecting between two vanishingly unlikely tokens. Inspection of stored top-1 IDs confirms every top-1 token is Yes or No.

Secondary strict generation exact and parse rates are0/192 in all four cells: every generated output is literally Yes. or No., whereas the registered parser permits only stripped/casefolded Yes or No. Preserve that outcome and parser; do not silently re-score punctuation away. The raw first-token metric was primary before outcomes. This is a clear formatting failure of that secondary criterion, not evidence that neither model can answer binary questions.

Contextual degradation affects selected indexed-frame questions despite perfect isolated responses under this diagnostic. It may reflect evidence access, reference/index resolution, or interference; an indexed prompt does not isolate the computation used by a whole-sequence count query. The preceding count probe also showed joint-structure sensitivity, so wholesale missing binding remains unsupported. The fixed16pair/six-position design does not establish general visual perception accuracy or exact aggregation. No method-generated external tally or training change was introduced. V3knowncompleted GPU cost is now7680seconds(2.13333hours), including failed reasoning lookup10seconds, main/profiles,binding and this diagnostic; tokenizer repair and one five-minute GPU compatibility retry remain separately tracked.


### Cosmos compatibility complete and V4 CPU gates — 2026-09-10 18:22 UTC

CPU440953 passed the tokenizer repair gate in10allocated seconds:151665vocabulary entries,151387merges,259strings/518encoding comparisons, exact slow/fast multimodal input tensor parity. New mirror /mnt/ckpts/gabriele/gnn_transformer/cosmos_reason1_compat_440953; original metadata and weights are preserved. GPU440961 completed in44seconds using only the mirror-path replacement in the unchanged V3 compatibility harness: one update, checkpoint restore and eight native cached outputs including N64 and multi-token gold targets. All eight outputs parse. This establishes software compatibility, not performance, reasoning behavior or composition. V3 is complete at7724GPU-seconds(2.14556hours), below4hours including every failed allocation.

V4 stage440963 is running on CPU4/16GiB/10minutes. CPU440965 passed all9 schedule/provenance tests. A direct actual-QA-question equality check was added before staging to catch reconstructed-metadata disagreement earlier. The software profile uses two original V2 test_N8 examples copied into new V4 dev_N8 paths because no V2 profile dev_N8 exists; original sources remain unchanged, main data are disjoint, and the exception is independently audited. No V4 GPU jobs have been submitted yet.


### V4 data stage and launch-check correction — 2026-09-10 18:24 UTC

Stage440963 passed in145allocated CPU-seconds:2078unique published samples,1360new draws,1540distinct training scenes, all source/copy/semantic/image/exclusion checks and every actual training processor layout. Main schedule hash c83781a8794dbd662d84cff4e18bc95e554905dd755b66ec4201f3a16f54518c. CPU440965 passed9tests. An additional root-authored real-schedule launch wrapper initially unpacked an outdated return signature (440971,440974); both CPU checks failed before any model work. Profile array440973 was canceled while pending its failed dependency and used zero GPU time. The helper returns five values including raw/parsed CPU token ledgers; only this wrapper was corrected. Runner/data/core source and scientific choices are unchanged.


### V4 profiles passed; main launched — 2026-09-10 18:31 UTC

CPU440975 passed both actual schedules and the frozen processor ledger. GPUprofiles440978_0/_1 completed47/46seconds. Both share exactly the initial parameter hash, block0 order and loss0.7779754996, and every slot token layout. Block1 refresh changes the intended scenes; Adam state persists through steps1 and2. Ten test predictions per arm include N64 and multi-token targets, all parsed; selected checkpoint restoration completed. Profile verification saved in outputs/native_aggregation_vlm/v4/profile/verification.json.

Main array440986 maps task0repeatseed2,1refreshseed2,2repeatseed3,3refreshseed3. Four concurrent B200s,50minute caps, identical frozen source and training budget. No scientific protocol changes. V4GPUcost beforemain is93seconds; even full3000second allocations for eachmain task plusprofiles wouldremainbelow4GPUhours.


### Frozen-model reasoning baseline assay — 2026-09-10 18:36:15 UTC, before staging/outcomes

Purpose: test whether extra unassisted reasoning tokens improve native MMReD Vision aggregation, establishing a baseline for later composition tests. This does not train/evaluate a newly improved method. The old V2 correct-prefix diagnostic supplied an oracle intermediate count and is not an end-to-end reasoning comparison.

Data:18 fresh anchors, two per K0..8, generated at N16 then extended to N32/N64 with the V2 negative-frame law and uniqueness rejection.54 main examples/18 families; data seed20260915. Exclude all selected V1–V4 main/profile/causal/binding content. Two separate software anchors K3,6 use the same lengths;60 total published examples. GPU profiles use their N16/N64 scenes. Freeze manifests and source/QA/image/semantic/extension audits under /mnt/data/gabriele/gnn_transformer/reasoning_baseline. No source dataset changes.

Models fixed in advance: frozen Qwen2.5-VL-7B-Instruct and frozen Cosmos-Reason1-7B from token-equivalent mirror cosmos_reason1_compat_440953. Native NF4/bf16 SDPA,392px images, ordinary processor/chat templates, cached generation. No adapter, optimizer, teacher, oracle prefix, external evidence tally, frame labels or best-of-many traces. Cross-model differences are descriptive because their training differs; primary contrasts are WITHIN each model.

Both conditions use exactly the same images-first user text: You will be shown N frames describing steps in a house. Question: Q. The final answer must be a single integer from0 toN. Replace the old integer-only prompt in BOTH conditions because it conflicts with reasoning. The reasoning system uses the documented Cosmos format: a nonempty <think> block followed by one <answer> integer block. The direct system requests just the answer block without reasoning. Use the same two system strings across models, each native template, no invented enable_thinking option. Freeze exact rendered prompts/token IDs/processor settings/native EOS IDs in the CPU plan before outcomes; no prompt search.

Greedy decoding, repetition_penalty1, unrestricted vocabulary and validated native EOS stopping explicitly depart from the Cosmos example's sampling policy. Direct cap32 new tokens; reasoning cap512, with a128-token endpoint computed from the SAME trajectory, never a duplicate run. Completion requires a declared normal EOS by the budget; EOS exactly at the budget counts. At128, later-EOS runs are truncated/incorrect even if a prefix contains gold. Preserve all raw generated IDs/text. Both cached model generation configs specify EOS IDs151645 and151643; verify and pass both rather than replacing this with tokenizer.eos alone.

Parsing accepts only the complete condition-specific grammar: exactly one terminal <answer> block containing an unsigned integer and surrounding whitespace. Reasoning additionally requires exactly one nonempty <think> block before it. Reject missing/multiple answer blocks, extra text outside the allowed blocks, noninteger answers and budget truncation. Never extract the last numeral of a trace. CPU synthetic tests cover embedded trace numbers, duplicate tags, whitespace, missing EOS, exact-budget EOS and invalid denominators. All examples stay in exact accuracy; show parse/completion/truncation separately and MAE only with its parsed denominator.

Primary descriptive contrast per fixed model: reasoning512 minus direct N32/N64 exact,36 observations/18 anchors, paired10000 whole-anchor bootstrap seed20260915. The prespecified effect screen is at least5 percentage points, not a significance test or comparison between model families. Report N16/18, reasoning128, perK and extension stability regardless. No selecting the best prompt/model/budget. Record prompt/generated tokens and measured time/memory;128 outcomes are explicitly censored views of a512-cap run, with no fabricated separately measured128 latency. A positive result supports this prompted-reasoning policy on these models/scenes, not all reasoning models. A negative result weakens the premise in this setting, not the possibility of useful reasoning tokens.

Resource block: at most1 GPU-hour. Two software profiles capped2minutes each; two main model jobs capped26minutes each,108 generations/model. Total maximum56 GPU-minutes leaves4minutes headroom. Profile throughput must support bounded main execution; otherwise record resource failure without silently dropping conditions. New GPU work waits for V4 allocations to free, at most4 concurrent B200s across this program. CPU stage4CPUs/16GiB/5minutes; planning/reporting through bounded CPU Slurm. No checkpoints are created; all data/model roots remain as requested.


### Reasoning-baseline data stage complete — 2026-09-10 18:50 UTC

CPU441018 completed in17seconds.60fresh examples/20families/2240frame references passed independent QA/gold/image/renderer/exclusion and40paired-extension checks. Main54examples are18anchors(two per K0..8), software6examples use two separate K3/K6anchors. Main manifest SHA7625e912d229e39006506cb060c34c9ea8bc5d154ab61c22f14a82a8ba010cdd; profile SHA88f59df9a856c4a966035c7abba1353f7362cb0ff1f55421b2911edb12d9c338. No model outcomes yet. The official Cosmos guide recommends a larger output budget; the registered512-token assay is deliberately a bounded-compute policy comparison and cannot establish unrestricted reasoning failure, especially if truncation is frequent.


### V4 matched scene-diversity outcomes — 2026-09-10 19:10 UTC

All four main tasks440986 completed: repeat2/refresh2/repeat3/refresh3 allocated1696/1709/1696/1712 GPU-seconds,6813 total. Together with93 profile seconds, V4 used6906seconds(1.91833GPU-hours), below4hours. Independent report441013 passed in11CPU-seconds after report-check441005. Canonical outputs/native_aggregation_vlm/v4/REPORT.md and analysis.json. All data/source/initialization/order/processor/runtime/checkpoint/405-update/1620-presentation/452-prediction audits passed.

Familiar N32/N64 exact: repeat seed2 59/216(27.31%), refresh2 64/216(29.63%); repeat3 60/216(27.78%), refresh3 103/216(47.69%). Refresh-minus-repeat seed2 +2.31pp, paired-anchor interval[-4.63,+9.26]; seed3 +19.91pp,[+11.11,+28.24]. N16 gains+9.26pp/+22.22pp. Primary both-seed >=5pp screen FAILED because seed2 falls short. Pooled fixed-seed+11.11pp,[+5.56,+16.67] is descriptive and cannot rescue it. Selected blocks repeat2/3=7/5,refresh2/3=7/9.

All outputs parsed. Unseen K9–16 correct repeat2/3=1/9 of128,refresh2/3=0/12. Seed2 OOD MAE worsened1.245→1.583 despite exact gain; seed3 improved1.468→0.824. N16→N64 accuracy losses remain26.85/26.85pp repeat and37.04/29.63pp refresh. Scene diversity helps the mean of these fixed seeds, but has not established reliable length/count extrapolation. This is ordinary supervised training on reused exploratory tests, not a novel operator or reasoning-composition result.

Reproducibility caveat: within each seed the initial trainable hashes, slot order, exact input-token hashes and image layouts match. First four training CE values match exactly; losses diverge from presentation5, immediately after update1, even though all180 first-block examples are shared. No nonzero attention/adapter dropout was found. Backward/update numerical nondeterminism is a possible explanation, not a localized finding; deterministic kernels and gradient/momentum tensor hashes were not recorded. Identical nominal initialization/order therefore did not yield bitwise-identical training trajectories. No criterion or checkpoint choice was changed after observing this.


### Reasoning assay CPU validation and first profile failure — 2026-09-10 19:19 UTC

CPU441193 failed in10seconds because this processor requires typed-list system content. Only that serialization was corrected before any plan/outcome; system text and scientific choices were unchanged. CPU441194 passed in68seconds: all60scenes/two processors, exact prompts/input IDs, native EOS metadata, QA/image audits and8parser tests. Plan SHA3c7629c322217d3b8f8f1ccf9452001750a492dcc8c59c1bafad59e4ff3f9e1b.

GPUprofiles441198/441199 failed after1/2allocated seconds in the model metadata equality guard, before model loading or predictions. Total failed GPUcost3seconds. Investigate exact metadata-field difference before any repair; preserve this plan and failed logs. No performance-dependent prompt, cap, parsing or dataset changes.


### Reasoning assay resource-gate outcome and separate throughput calibration — 2026-09-10 19:36 UTC, before calibration jobs

The metadata mismatch was exclusively node-local filesystem st_dev; all nine shard paths/inodes/sizes/mtime and small metadata hashes matched. Only st_dev was removed from portable identity. CPU441210 passed in69seconds, including verification after saved-JSON roundtrip; new plan SHA6de7e0c1610c4ab928fec6b513fc5ef70afb2d90477c7a1b0a88c37812c3373f. Old sources/plans/logs are preserved.

Software profiles441212(Qwen)/441213(Cosmos) completed39/83allocated GPU-seconds. Including the failed first profiles, assay expenditure is125GPU-seconds. Both original cost gates FAILED: projected5505.32/1697.43seconds exceed the1560-second main cap. No main was launched from those gates. Qwen produced two-token native-EOS trajectories under both policies on all four software scenes; Cosmos reasoning averaged472.25tokens. Software scenes do not enter the main efficacy analysis, and no prompt/budget/model will be selected from these outcomes.

The original maximum per-token wall estimator attributes short-generation setup overhead to every subsequent token and applies the short direct-policy rate to all511 reasoning decode steps. Register a separate timing-only calibration, preserving the failed estimate. Use exactly software K3/N64 and K6/N16, both original prompts and models, native NF4/bf16/SDPA/greedy settings. For timing only force the complete32/512-token paths with min_new_tokens equal to max_new_tokens, suppressing EOS until the cap. These artificial continuations are not efficacy trajectories and must never be scored as reasoning or aggregation results; retain raw IDs/configuration for accounting.

Measure both policies separately at both lengths. New conservative projection: model-load time +1.25*54*(maximum shared preparation +2*maximum measured prefill +31*maximum direct-policy per-decode-token wall time +511*maximum reasoning-policy per-decode-token wall time)+30seconds. Use full-cap trajectories so fixed setup is amortized over measured decode length. Passing requires <=1560seconds independently for each model. This changes the operational estimator, not the primary efficacy protocol. A timing failure remains a resource failure; do not shorten/drop main scenes or conditions. Original frozen assay modules/wrappers/plan remain unchanged; new timing script/wrappers get their own CPU source ledger.

Each calibration GPU job has a two-minute cap. Known125seconds + at most240seconds calibration + at most3120seconds for both26-minute mains =3485seconds, below the original one-GPU-hour cap, leaving115seconds headroom. CPU validation/reporting remains on Slurm. Main54scenes/108trajectories per model and all inference/scoring decisions remain unchanged.


### V5 independent visual evidence: extensive versus averaged messages — 2026-09-10 19:36 UTC, before model profiles/training

Motivation: V1–V3 did not establish a robust aggregation-operator benefit, V4's data-refresh screen failed across two seeds, and indexed conjunction judgments degrade in full context. The latter does not isolate visual-feature corruption. Test one narrower hypothesis: after accessing each image independently, preserving the total magnitude of learned query-conditioned evidence helps native length extrapolation. This is a conditional Deep Sets adapter built from established visual-memory ideas (especially PVM), not a new attention family or a literal bandwidth/capacity theorem.

Use ordinary Qwen2.5-VL7B NF4/bf16 SDPA and one native vision-encoder forward. Retain its final per-image visual output after window-order restoration; do not hook the earlier still-packed merger tensor. At language layer14, query this memory separately within each already completed visible image. Normalize each input hidden/visual vector by fixed RMS(epsilon1e-6); learn Wq:hidden→96, Wmem:hidden→96(shared keys/values), Wr:96→96 with bias, and U:96→hidden without bias, initialized to zero. For query q=Wq(norm(h)), image read r_f=softmax(q*M_f^T/sqrt96)M_f, message m_f=SiLU(Wr(r_f)+q)-SiLU(b+q). SUM adds U sum_f m_f; MEAN adds U(mean_f m_f). Empty memory contributes zero. Both use identical parameter construction/initialization, centering, data, optimizer and native output head.

Only language positions query; image content and vision boundary tokens do not receive a branch update. A query sees only images whose closing boundary is strictly earlier than it. Preserve native attention, positions and KV cache; the branch adds parallel computation without serial reasoning tokens or external predictions. Prototype is explicitly batch1, images only, unpadded inputs, ordinary greedy cache, no beams or mid-cache image insertion. New examples/errors reset retained memory. Learned memory projections must not survive an optimizer update. The vision encoding is reused during native decode, while each new language state issues a new branch read.

SUM and MEAN have identical parameters and at fixed weights/query satisfy delta_SUM=N_visible*delta_MEAN. Their contrast tests extensive scaling and its optimization consequences; it does not establish more information at fixed N. Learned negative-frame messages may accumulate despite zero-read centering. Keep these limitations visible in every report.

Both arms also train the same upper-four-layer LoRA rank8/alpha16. Same native count-answer/EOS CE only, no local labels/curriculum/teacher/fences/numeric coordinates or arithmetic head. Branch LR0.001, LoRA LR0.0001, AdamWwd0, clip1, accumulation4. Fixed seeds4 and5. Reuse the exact immutable V4 refresh schedule and staged data under /mnt/data/gabriele/gnn_transformer/v4_diversity: nine180-presentation blocks,1540distinct scenes,1620presentations,19440frames,405updates. Both arms use every same scene in each corresponding optimizer batch. The deterministic N8/K8 repeats remain as documented. Same392px/native images-first counting prompt, train N8/N16 and K0..8. Checkpoints under /mnt/ckpts/gabriele/gnn_transformer/native_aggregation_vlm_v5.

Same nine dev evaluations/72scenes; choose highest dev exact, then lowest raw first-answer-token NLL, then earliest. Evaluate the selected checkpoint with cached unrestricted greedy max4tokens/repetition_penalty1 on all original108 N16,216 familiar N32/N64 and128 unseen K9..16 examples. These are reused exploratory tests, not fresh confirmation. Main criterion: SUM-minus-MEAN familiarOOD exact >=5pp in EACH fixed seed and N16 loss<=5pp in each. Failure in either fails the screen; pooling cannot rescue it. Report each seed, pooled fixed-seed contrast,10000 whole-anchor bootstrap(seed20260916), all parse/MAE/bias/count-support/extension outcomes and measured inference/training cost. Do not compare historical parameter-unmatched baselines as a causal estimate.

Software gate before training: CPU tests for null contribution, duplicate/SUM and duplicate/MEAN behavior, image-boundary causal masking, arbitrary patch counts, reset/error/unsupported inputs and meaningful nonzero cached-versus-full execution. Freeze new source ledger and validate both actual V4 schedules without modifying earlier sources. Bounded GPU profiles for both arms must verify zero-initialization native parity, nonzero-branch cached/full parity, fresh-example reset, finite updates and restored checkpoints, N64 and multi-token targets. Main runs require passing implementation gates and measured feasibility. Record initial parameter hashes and actual presentations/orders/processor layouts. Track first-step gradients as a reproducibility diagnostic; nominally matched seeds are not a claim of bitwise-identical training.

New V5 resource block at most4GPU-hours, at most4B200s concurrently across all research jobs. Two profiles at most5minutes each and four main tasks at most50minutes each; projected full reservation210minutes fits. All CPU-heavy checks/reporting and GPU work use Slurm. No layer/rank/LR search or automatic expansion to memory-source, pooling-order or local-supervision arms. Any later confirmation/composition claim requires separately registered evidence.


### Timestamp correction and reasoning resource amendment — 2026-09-10 19:27:17 UTC, before main jobs

The preceding calibration and V5 protocol headings were mistakenly entered as19:36UTC. They were actually appended before the independent clock read19:19:55UTC and before the associated model jobs. This is a heading-timestamp error; tool-call order, immutable artifacts and job records preserve the chronology. No V5 outcomes or reasoning-main outcomes existed when those protocols were appended.

Timing CPU441223 passed in1second, plan SHAb69833771da563ea30406ea9166835ded522afecb6131e82bc2e9c4a04df5cee. Full-cap calibration441240(Qwen)/441241(Cosmos) completed99/56allocated GPU-seconds. All forced32/512token/native-forward/source checks passed; these traces are not efficacy outcomes. Projections2116.56/1250.35seconds: Qwen AGAIN FAILS the original26-minute gate, Cosmos passes. Qwen's maximum first N64 prefill7.954seconds enters the conservative bound for every prefill; sustained reasoning decode was0.02487seconds/token. Preserve both failed original gates and this second Qwen failure. Do not retune the timing formula or run another calibration.

Explicitly amend the resource envelope, before any main predictions, to at most1.1GPU-hours total(3960seconds), with main Slurm caps36minutes for Qwen and22minutes for Cosmos. The unchanged projections fit these1560→2160 and1560→1320second allocation limits. Known prior expenditure280seconds plus maximum2160+1320=3760seconds(1.04444GPU-hours), leaving200seconds within the amended envelope. This is an operational budget increase, not a passing result under the original1-hour reservation; actual total use remains to be measured.

Submit the exact unchanged frozen main wrapper with Slurm --time overrides; no source/plan/prompt/model/data/decoding/scoring changes. Both models still receive all54main scenes and both policies, including the strict128-prefix endpoint. Main comparisons remain descriptive and within-model.


### V5 CPU checks and reasoning main launch — 2026-09-10 19:32:44 UTC

V5 CPU441256 passed15 focused unit tests and both conditions on both actual V4 main/profile schedules in2seconds. Frozen source ledger outputs/native_aggregation_vlm/v5/check_441256/source_hashes.json. GPUprofile array441258 launched two arms, within the five-minute cap each. Actual GPU parity helper checks zero-U exact equality, nonzero-U full/cache next-token logits at identical prefixes (maximum absolute difference<=0.25 and vocabulary RMS<=0.025, allowing ordinary bf16 kernel variation), one native visual forward per cached generation, and exact A/B/A fresh-example repeated IDs/logits. These tolerances were implemented and CPU-source-frozen before model-profile outcomes.

Same-forward V5 diagnostics are explicitly observational: store detached final-prompt per-image messages, merged message, branch residual and receiving native attention-output norm under /mnt/data/gabriele/gnn_transformer/v5_diagnostics/RUN; prediction rows bind paths/hashes. Capture occurs on the first native generation forward, before decode overwrites stats. No annotations enter the branch or its loss. The receiving attention-output norm is not the full language residual-stream norm. Main backward logging records separate pre-clip branch/LoRA norms, clipping and first-update gradient hashes. These instruments do not change the architecture/loss/selection criterion.

Reasoning mains441249(Qwen36min)/441250(Cosmos22min) launched under the explicitly amended1.1GPUh allocation envelope. Qwen completed161allocated seconds; Cosmos still running. Independent report441252 is queued after both. No efficacy interpretation until the complete paired report.


### V5 actual-model parity gate failure and native-reference check — 2026-09-10 19:36:17 UTC, before retry

Both first GPUprofiles failed the registered absolute RMS cache/full tolerance after passing zero-U exact parity, two finite updates and checkpoint restoration. Array441258 SUM/MEAN allocated39/38seconds,77total. First N16 next-token logit maximum differences were0.140625 in both; vocabulary RMS differences0.0379130/0.0313141 exceeded0.025. No V5 main training was launched. Preserve failed runs and the initial source ledger/checkpoint artifacts. This is a software-verification failure, not an efficacy result.

The initial tolerance lacked a measured ordinary-native cache/full reference. Before any main outcomes, add a disabled-branch reference for each exact tested prefix. A software-only LogitsProcessor forces only the disabled run's first generated token to the enabled run's first token, so second-step raw native logits condition on identical tokens; it does not modify the stored raw logits or any scientific generation. Compare disabled cached versus disabled full at that prefix, alongside the enabled pair. Keep zero-U equality, CPU active-U masking/cache tests, native visual-call counts and exact A/B/A reset checks.

Revised numerical engineering gate, fixed before retry: enabled maximum absolute logit difference <=1.5 times disabled reference maximum +0.0625; enabled vocabulary RMS <=1.5 times disabled reference RMS +0.005. Save both reference differences and computed tolerances for every N16/N64/N16 observation. This checks whether the added branch substantially worsens the existing numerical discrepancy; it is not mathematical equality or a calibrated confidence interval. The failed absolute criterion stays failed. No architecture, weights initialization, optimizer, data, loss, dev selection or primary scientific criterion changes. CPU revalidation/source freeze precedes the two profile retries. V5 remains within4GPUhours including77failed seconds.


### Frozen reasoning-baseline outcomes — 2026-09-10 19:43:49 UTC

Both mains complete: Qwen441249 used161GPU-seconds, Cosmos441250 used763. CPUreport441252 passed independent raw-token/provenance/cost checks in13seconds. Total assay1204GPU-seconds(0.33444hours), including280prior failed/software/calibration seconds, below both original and amended envelopes in actual use. Canonical report outputs/native_aggregation_vlm/reasoning_baseline/report_441252/REPORT.md.

Qwen strict exact0/18N16 and0/36OOD under direct32,reason128,reason512; every main output was an untagged bare integer (54/54perpolicy, post-hoc structural inspection only). This is failure to follow the required format/reasoning policy, not zero numerical accuracy. No alternative parser or efficacy rescore was substituted.

Cosmos direct32 N16=4/18,OOD=4/36; reason512 N16=4/18,OOD=1/36. Reason-minus-direct OOD−8.33pp, paired interval[−22.22,0]; primary fails. Direct all36OODparsed, reason512 only8parsed and28truncated; reason128 all36truncated. Native reasoning did not improve this constrained policy, but heavy truncation prevents a general inference about reasoning effectiveness. Models/prompts/budgets were not selected for success, and software timing continuations never entered efficacy data. This is not a composition test of an improved method.


### V5 second integration-gate failure and localization diagnostic — 2026-09-10 19:48:03 UTC, before diagnostic jobs

CPU441410 again passed15tests and all schedules in2seconds. Second profiles441416 SUM/MEAN used45/39GPU-seconds and failed the native-reference RMS comparison: SUM enabled max/RMS0.171875/0.0710197 versus disabled0.109375/0.0295943 (RMSlimit0.0493915); MEAN enabled0.1640625/0.0730174 versus disabled0.15234375/0.0434522 (limit0.0701783). No V5 main launched. Failed GPUcost now161seconds. Report-check441419 passed its synthetic/statistical/data tests in1CPU-second; no failed profile can qualify as a passing gate.

Do not increase tolerance again without locating the discrepancy. Add a separate observational diagnostic on exactly four fixed cases: selected second-failed-profile SUM/MEAN checkpoints crossed with the first existing profile N16/N64 test examples. Use identical first-token prefixes for enabled and disabled cache/full comparisons. Save raw logits, global-shift-centered logits, log-softmax differences/KL/top1, final-query layer input/projection, per-image reads/messages/residual, position/visibility metadata and raw visual-memory equality/difference. No efficacy classification or new acceptance threshold is selected in this diagnostic. The purpose is to locate a cache/memory/interface bug or characterize ordinary numerical sensitivity/softmax-invariant shifts. Preserve all existing core/harness/checkpoint files. New script/wrappers and source/plan hashes before Slurm execution; CPUgate<=5minutes and one GPU diagnostic<=5minutes. This remains within V5's4GPU-hour block.

### Cosmos reasoning4096 extension — 2026-09-10 19:48:03 UTC, before new profile/main predictions

Motivation: the registered512-token policy truncated28/36OOD scenes, so it is insufficient to test the user's premise about longer reasoning. Add one explicit exploratory budget extension, keeping the original results unchanged. This is not a new aggregation method or confirmation on fresh test data; these54scenes are now reused. Cosmos's documented recommendation of at least4096 output tokens motivates this single budget; do not search prompts/budgets/models for success.

Frozen Cosmos-Reason1-7B from compatibility mirror440953, same NF4/bf16SDPA,392px processor, exact54main records from baselineplan441210 SHA6de7e0c1610c4ab928fec6b513fc5ef70afb2d90477c7a1b0a88c37812c3373f, same images-first user question and reasoning system, unrestricted greedy/repetition_penalty1/native dual EOS. Only the reasoning maximum becomes4096. Total prompt+generation ceiling18000, below the cached model's128000position limit. No min_new_tokens, forced continuations, frame labels, oracle prefix or external tally in efficacy.

Reuse the already independently audited direct32 predictions from Cosmos main441250 as the paired baseline; do not regenerate/select them. Generate exactly one4096-cap natural reasoning trajectory per scene and score128/512/4096 endpoints using the unchanged strict full-trace grammar and EOS completion rule. Record raw IDs/text, time/memory, completion/truncation, perK/N, MAE with denominator and extension stability. Compare each new trace's first512 tokens to the original512 trajectory and report every mismatch without dropping or choosing runs. Shorter-budget endpoints of the new trace are censored views, not separately measured runtimes.

Primary descriptive contrast reason4096 minus existing direct32 N32/N64 exact(36observations/18whole anchors),10000 paired-anchor bootstrapseed20260917; >=5pp descriptive effect screen, not significance. Report N16 and all endpoints regardless. Both condition policy and output budget still differ from direct; the within-reasoning512→4096 contrast is additionally shown. This is one quantized model and a reused synthetic benchmark, not all reasoning models or proof of composition.

CPU freeze validates exact baseline/report/prediction/source hashes, all prepared prompt IDs, native EOS/configuration and 18000-token budget. One natural software profile uses the existing separate K3/N64 example at4096cap, at most4GPUminutes. If it stops earlier, disclose the actually exercised cache length; do not call it a full4k-depth test. The profile is a software/resource check, never used to tune the policy. Reuse prior sustained full-cap timing (~0.025seconds/decode token); nominal all54at4096cost is about95GPUminutes, with115minutes main reservation. One main Slurm job<=115minutes and profile<=4minutes make a new2GPU-hour block, at most4B200s across all research jobs. No new checkpoints/data images; reports under outputs/native_aggregation_vlm/reasoning_long, all existing model/data roots preserved. CPU-heavy validation/reporting through Slurm.


### V5 numerical localization and translation-invariant integration gate — 2026-09-10 19:57:43 UTC, before revised profiles

CPU441428 passed in2seconds; diagnostic441439 completed43GPU-seconds, bringing V5 GPU use to204seconds. All four fixed arm/length cases have bit-exact raw visual memory, image boundaries and rotary positions. Identical-input branch replay is exact or within6.11e-5 maximum/1.06e-6 RMS; fixed-query memory swaps are exact. Native hidden states already differ before the branch(RMS0.00427–0.00565), and ON/OFF upstream discrepancy summaries are identical. This finds no cache-state/interface error in these cases, but does not identify the underlying CUDA kernel or prove global numerical equivalence.

Correction: the second profile failures quoted earlier occurred at N64, after N16 passed; the prior description incorrectly called these first-N16 failures. The saved logs and diagnostic records retain actual identities. N64 enabled raw-logit squared differences are92.40%/91.80% vocabulary-wide constant shifts for SUM/MEAN. Such shifts leave softmax probabilities invariant. Centered RMS is0.01957/0.02091 versus disabled0.01805/0.02169; all four next-token argmaxes match. Three of four enabled distributions are EOS-dominated, so their tiny KL/TV is not sufficient evidence alone. Original raw gates remain failed.

Before any main training, replace the raw-logit engineering criterion with a translation-invariant one while continuing to record raw differences. For each cached/full difference subtract its vocabulary mean; compare enabled centered maximum/RMS to1.5 times the disabled centered reference plus0.0625/0.005, retaining the same relative factors/floors. Additionally require enabled centered maximum<=0.25, centered RMS<=0.05, total-variation distance<=0.01 and equal raw next-token argmax. These are explicit engineering tolerances, not statistical confidence intervals or proof of cache equivalence. Use FP64 softmax for probability diagnostics. No further tolerance relaxation is authorized merely to pass.

Exercise both original first N16/N64 software examples AND the previously unused second N16/N64 software examples, followed by the first N16 again. Five observations,20native visual calls, exact A/B/C/D/A reset checks. The latter two examples were not used for localization/calibration. Preserve zero-U exact equality,15 focused CPU tests, nonzero-U/finite updates/restored checkpoint and identical forced-prefix native reference. The core architecture, training and efficacy protocol are unchanged. Freeze the new helper/runner sources and update independent report validation before the next two<=5minute profiles. Main training remains blocked until these checks and feasibility pass. The4GPU-hour block still includes all failures and diagnostic cost.


### V5 third profile outcome and quantized-kernel diagnostic — 2026-09-10 20:07:14 UTC, before diagnostic jobs

CPU441447 passed15tests/schedules in2seconds. Third profiles441450 SUM/MEAN completed64/48GPU-seconds. SUM passed allfive software checks, reset and ten final predictions. MEAN failed the previously unused N64/K6 case solely on total variation0.0118166>0.01; centered RMS0.0164927 and maximum0.0756970 and top1 agreement passed. The same-prefix disabled native TV was0.0148787, with larger centered RMS0.0246590. MEAN's final repeat/reset and ten final predictions were not reached. This remains a failed registered integration gate, and no V5 main is launched. V5 cumulative316GPU-seconds. Do not loosen the TV threshold or call this passed.

Installed bitsandbytes autograd/_functions.py selects gemv_4bit for a single no-gradient input vector, whereas multi-token prefill invokes MatMul4Bit.apply, which dequantizes and uses torch linear. This source inspection identifies a plausible numerical difference but does not establish causation. Add a separate causal software diagnostic on exactly the third-profile MEAN selected checkpoint and the unused second N16/N64 software records. Compare ordinary execution with a temporary wrapper that routes bnb.matmul_4bit through its existing MatMul4Bit.apply path for both cached and full forwards; restore the function afterward. Keep identical quantized weights, bf16 activations, prefixes, prompts, branch, native attention and all other settings. No training or efficacy scoring. Record upstream layer14 hidden differences, image memory/positions, branch replay and FP64 probability/centered-logit metrics for ON and same-prefix OFF.

CPU plan freezes checkpoint/source/input hashes before one diagnostic GPU job<=3minutes. There is no new threshold or automatic permission to alter the main inference backend. Any main backend change requires an explicit prospective amendment and the unchanged numerical gate. All failures remain preserved; the4GPU-hour block includes this diagnostic.

### Cosmos4096 CPU freeze and profile launch — 2026-09-10 20:07:14 UTC

CPU441453 passed the exact55input/source/baseline/profile audits in87seconds. Plan outputs/native_aggregation_vlm/reasoning_long/check_441453/plan.json SHA74f20080553eb59716b2af04a046de45e4f6b120e1fb9b9ec13fabd61106c958. Software profile441454 launched under the4minute cap. A main must bind this plan and a matching successful profile; neither answer accuracy nor old-prefix agreement selects eligibility.


### V5 paired-message decomposition — 2026-09-10 20:10:23 UTC, before main predictions

Prepare a separate descriptive analysis of already planned final-prompt per-image messages on all paired N16→N32/N64 familiar-count tests, for each fixed seed/arm. Bind exact manifests, image hashes, selected checkpoint and diagnostic artifacts. The extended scene contains the original anchor images plus negative frames; audit their indices before matching. At fixed trained weights decompose the change in summed messages into the change on those original images and the contribution from appended negatives. For MEAN retain the exact N-dependent normalization, reporting original-message drift/N, new-negative contribution/N and the dilution term from1/N−1/16 separately; verify vector reconstruction.

Report norms, cosines, reconstruction errors and perN/perseed/arm summaries over all complete pairs, without selecting correct examples. Report saved branch-to-native-attention-output ratios separately. Changes in original-image messages can reflect the contextual language query and numerical effects; this is not a causal isolation of query drift. Negative-message contributions are not necessarily semantically harmful merely because their norm is nonzero. These diagnostics explain possible failure modes and do not replace the registered efficacy screen or establish information absence. No fitted classifier, local labels in the model, new optimization or method arm. Analysis CPU jobs through Slurm after complete bound artifacts are available.

### Cosmos4096 main launch — 2026-09-10 20:10:23 UTC

Software441454 completed49GPU-seconds, natural EOS after858tokens, all native cache/one-vision-forward/source checks passed and first512tokens exactly matched the original trace. It did not exercise4096decode steps. Main441455 launched with all54scenes and the unchanged4096cap, at most115minutes. Independent report441457 queued after successful completion. V5 remains blocked on its separate numerical check; future overlap must keep total research concurrency<=4GPUs.


### Paired-message image-identity correction — 2026-09-10 20:10:54 UTC, before analysis/main outcomes

Source review of stage_native_vision_v2_clean.py shows that extensions preserve semantic character/room frames but renumber rendered Step labels. The preceding decomposition description incorrectly assumed all mapped original images were byte-identical. Correct the planned analysis before execution: audit semantic correspondence and every actual image hash against its own manifest, and report mapped-anchor image hash match counts. Retain every pair; do not select an exact-image subset. Call the first component mapped original semantic-frame message drift. It combines changed Step pixels, query/context changes and numerical effects; it cannot isolate context alone. Algebraic reconstruction remains exact at the saved-vector level. No dataset or method change.


### V5 kernel finding and conditional engineering acceptance — 2026-09-10 20:15:25 UTC, before any main training

CPU441465 passed4seconds; kernel diagnostic441467 completed41GPU-seconds. V5 total357GPU-seconds. Both full-prefill logits and upstream full-forward hidden states are bit-exact across native versus forced MatMul4Bit routes. Cached states change, and the failing N64 MEAN enabled TV falls0.0118166→0.0077706; disabled0.0148787→0.0097143. This causally identifies quantized linear dispatch as one numerical contributor. Remaining differences persist, and changing dispatch moves cached predictions' distributions itself; no production kernel patch is adopted. Image/position identities and fixed-input branch replay remain correct within the recorded tiny arithmetic differences.

The absolute1% numerical gate remains FAILED. This is an explicit change to the engineering release decision, not a passing result under that gate and not a fourth adjusted tolerance. Conditional acceptance for exploratory native-backend training is based on separate evidence of branch-state/operator integrity and a reproduced native numerical limitation. No V5 main efficacy outputs exist. Keep the standard NF4/bf16/SDPA/native bitsandbytes dispatch for BOTH arms and all fixed seeds. The scientific training, selection, test, effect criterion and old-data exploratory status remain unchanged. No claim of exact full/cache equality or general long-reasoning numerical robustness is warranted.

Before release, use a separate source-frozen audit to load the SAME third-failed MEAN selected checkpoint, without retraining. Complete the five fixed A/B/C/D/A cache comparisons and exact reset, memory/position/native-visual-call checks, then all ten ordinary software evaluations with final-prompt message capture. Record each unchanged centered/TV/top1 criterion and overall strict_numerical_gate_passed=false if any fail. Numerical failures are retained as observations; computational integrity failures still block release. Bind a separate explicit engineering-acceptance artifact to the failed profile, diagnostic hashes, completion audit, exact main source ledger and the rationale/limitations above. The independent reporter must distinguish engineering acceptance from passing the numerical gate.

After all four main selected checkpoints exist, run the same fixed five cache comparisons and ten software evaluations per checkpoint, with every numerical failure reported and no threshold adjustment, rerun selection or model selection from this audit. This diagnoses amplification after training; two-update checks cannot bound it. No main test sample is dropped or rescored. The CPU plan binds the already dev-selected checkpoint identities before this post-training audit.

All new audit compute remains through Slurm: CPU freeze<=5min, pre-release oneGPU<=3min, post-training oneGPU<=5min. A conservative resource reservation357+180+300+4*3000=12837GPU-seconds(3.566h) remains below V5's4hour cap; actual accounting includes any other V5 jobs. Runtime from completed SUM software projects~37.6minutes with25% work margin, or~41minutes with doubled evaluation model time, supporting the50minute main caps subject to shared-resource variability. Max4researchGPUs; overlap with Cosmos means at most3V5 mains until it finishes.


### V5 engineering release verified and main launch — 2026-09-10 20:27:57 UTC

CPU pre-audit441486 passed4seconds, planSHA0c3118da68d6306257836d1c2f7d172ff4c04f2c111f241becc33d8ab88ec1e9. Precompletion441489 completed56GPU-seconds from the exact failed MEAN checkpoint, without training: allfive state/cache/reset cases and allten ordinary software evaluations completed; computational_integrity_passed=true, strict_numerical_gate_passed=false. The original D64 TV failure was reproduced, retained, and not relaxed. Cumulative V5 GPUuse413seconds.

Schema2 engineering release artifact outputs/native_aggregation_vlm/v5/implementation_gates.json binds CPU441447, strict SUM441450, failed MEAN441450/log, both diagnostics and the same-checkpoint completion. It explicitly says accepted_for_exploratory_training=true and strict_numerical_gates_passed=false. Main source ledger is unchanged441447; independent reporter/release validator sources are frozen in analysis_source_hashes.json and analysis_code. CPU441493 passed statistical/adversarial/data checks in1second; full release audit441495 passed in10seconds while explicitly reporting the strict failure.

Main array441498 submitted SUMseed4,MEANseed4,SUMseed5,MEANseed5 with at most3 concurrent while Cosmos441455 uses one project GPU. Each task<=50minutes, unchanged registered schedule/loss/selection/decoding. Raise this project's array throttle to4 only after Cosmos completes, if useful. The pre-existing unrelated diffdist task427854 is outside this MMReD experiment's accounting and is left untouched. The project resource cap concerns the jobs launched for this investigation. Post-training fixed-checkpoint audit and complete independent efficacy report remain required; no positive method claim yet.


### Cosmos4096 complete; fourth V5 training slot released — 2026-09-10 20:39:53 UTC

Main441455 completed1769GPU-seconds; independent report441457 verified in11CPU-seconds. Including49profile seconds, this extension used1818GPU-seconds(0.505hours), under its2hour cap. All54new trajectories match the original512 prefixes exactly; all54remain included. Actual4096decode depth was exercised in the main. Total66888generated tokens,1705.8seconds generation and1755.3seconds including model load/preparation.

Cosmos reasoning4096 N16=5/18 versus direct4/18; familiarOOD=6/36 versus direct4/36. Descriptive +5.56pp, paired interval[0,+13.89], passes the prespecified descriptive5pp screen by two additional correct scenes. This is weak absolute performance(16.67%), not a robust aggregation solution or composition evidence. Reasoning512 remains1/36OOD with28truncated;4096 has32completed/parsed and4truncated, conditional parsed MAE6.781 versus direct5.139. Additional completion largely did not yield correct counts. This adaptive extension reuses inspected scenes; no fresh confirmation claim.

After Cosmos finished, V5 array441498 throttle increased3→4. Allfour own MMReD project GPU slots now train the fixed SUM/MEAN×seeds4/5 runs. No V5 scientific setting or selection criterion changed.


### Conditional V5 frozen-message readability audit — 2026-09-10 20:43:53 UTC, before selected-checkpoint main test outcomes

Prepare the failed-screen branch of the documented decision tree while training continues; development logs have been observed, but no selected-checkpoint V5 main test outcomes are available. Execute this separate CPU diagnostic only if the complete registered two-seed V5 efficacy screen fails. It is not another method arm, a replacement score or a claim of external-head aggregation success. All four dev-selected checkpoints remain included.

Use only existing final-prompt per-frame96-dimensional messages and audited QA character/room labels. Deterministically sort108familiar anchor families, shuffle with seed20260918, assign54fitting and54held-out families, identical across all arms/seeds. Fit using only the54N16fitting contexts. Hold out all lengths of the other54families. Also report all128unseen-count contexts separately; none enters fitting. Verify complete family separation, canonical main analysis/checkpoint hashes, semantic targets and every used diagnostic tensor hash.

One fixed probe per checkpoint: fit-only feature means and standard deviations(floor1e-6), intercept plus96standardized features, class-balanced weighted squared loss on targets−1/+1 and ridge coefficient0.001 on the96weights only. Normalize the weighted loss by total fit weight; leave intercept unpenalized. Solve in float64, use threshold0. No hyperparameter, threshold, feature or checkpoint search. Save coefficients under /mnt/ckpts/gabriele/gnn_transformer/v5_local_readability and reports under outputs/native_aggregation_vlm/v5/local_readability.

Report all held familiar N16/N32/N64 and all unseen N32/N64 with confusion counts, positive recall, false-positive rate, balanced accuracy, tie-correct AUROC and explicit denominators; distinguish positive, character-only negative, room-only negative and neither, and report perK. Primary descriptive interest is held-family N32/N64local AUROC and TPR/FPR at the fixed threshold, with no invented success cutoff. A high score demonstrates accessibility to this fitted local probe on this exploratory split; a low linear score does not establish information absence. No classifier prediction enters native inference or its accuracy table, and no external tally is presented as the method. Further method changes require separate evidence and registration.

All computation through Slurm, CPU4/16GB<=5minutes; no new GPU forward or model training beyond fitting these explicitly diagnostic linear probes. Source freeze and synthetic split/weighted-ridge/zero-variance/tie checks before execution. The existing paired-message decomposition remains a separate algebraic analysis without fitted classifiers.


### V5 complete, conditional readability and decomposition schema repair — 2026-09-10 21:14:32 UTC

Allfour mains441498 completed6868GPU-seconds. CPU441515 bound every dev-selected checkpoint in5seconds; GPU441517 completed155seconds with computational integrity and all20unchanged strict numerical observations passing. This does not erase earlier failed profile gates. Independent final report441547 passed18CPU-seconds and all1808main predictions/diagnostics. TotalV5 GPUcost7436seconds(2.06556hours), including failed profiles and diagnostics.

SUM familiarOOD30/216(seed4),12/216(seed5); MEAN47/216,33/216. SUM-minus-MEAN −7.87pp[−17.59,+1.39] and−9.72pp[−14.35,−5.56]; both-seed primary FAILED. N16 SUM33/108,17/108 versusMEAN42/108,45/108. SUMseed5 all108N64answers are invalid; allinvalidoutputs remain incorrect. UnseenK9–16 SUM0/128both,MEAN0/128and1/128. Pooled−8.80pp[−15.28,−2.78] is descriptive. All405SUMupdates and401/402MEANupdates clipped; the treatment includes optimization scale, not pure representational capacity.

Conditional readability441550 completed5CPU-seconds under its frozen441532source ledger. HeldN64 AUROC SUM4=.5108,SUM5=.4425,MEAN4=.5391,MEAN5=.5348; fixed-threshold probes forSUM5/MEAN4 predict allnegative atN32/N64. These weak linear results do not establish absent information, but do not support a cleanly readable local-message explanation either. No native model is updated by this diagnostic.

Paired decomposition441549 failed7CPU-seconds before results: it incorrectly required anchor_positions/parent_positions fields in canonical predictions, which never serialize those fields. Correct only this diagnostic schema check: obtain mappings from the alreadyhash-bound independently audited manifests; validate actual prediction identity fields, and reject conflicting optional mappings if present. Keep allpairs, algorithms, thresholds, scientific settings and failed artifacts. Add missing-versus-conflicting mapping tests and rerunCPUselftest before actualanalysis. This is a software repair, not a changed decomposition or selected-data analysis.


### V6: training-only local competence for native SUM aggregation — 2026-09-10 21:16:14 UTC, before teacher outputs or V6 runs

Motivation: V5 SUM/MEAN both extrapolate poorly, and fixed held-family local probes show weak conjunction readability. This does not prove missing information or establish a sole cause. Test whether local supervision can train the existing small reader to expose useful evidence to native count prediction. The inference architecture is unchanged V5 SUM, chosen prospectively for this new block because it retains extensive scaling; its poor V5 results and invalid long outputs are known. This is an established distillation intervention, not a new attention family, cardinality theorem or reasoning-composition result. Alternative native parallel-query-stream ideas remain untested.

Four contemporaneous main runs: control versus aligned local distillation, seeds6/7. Both use the same V5 independent per-image rank96 SUM branch atlayer14 and upper4-layer LoRA,1,762,400deployed trainable parameters, frozen Qwen2.5-VL7B NF4/bf16/SDPA/native bitsandbytes. Keep V4 refresh training:1540distinct scenes,1620presentations,19440presented frames,405AdamW updates,accumulation4, branchLR.001/LoRALR.0001,wd0,combined native-parameter clip1. Keep72V4dev examples aftereach45updates; select exact, then raw gold-first-tokenNLL, then earliest. Normal images-first count prompt,392resize,native unrestricted greedymax4/repetitionpenalty1, no intermediate reasoning tokens or external tally.

Teacher: exact same frozen Qwen snapshot/runtime and canonical build_count_prompt(original_question,1), using each unchanged original training image including renderedStep label. Deduplicate exact(imageSHA,question,fullprompt/processoridentity): currently9980unique image/question pairs for18800frames across1540distinct training scenes. Reuse each target at every original training presentation; no changed sampling weights. One ordinary isolated-image prefill, no generation/search. Cache full-vocabulary-normalized first-token probabilities for0,1,and all other tokens together, plus raw0/1logits and full-vocabulary lognormalizer. Verify0/1 are single tokens. Temperature1. Gold frame semantics are used only for training-domain teacher-quality auditing and software coverage, never as KD targets. No teacher targets from dev/test; no target filtering or confidence weighting. Rawteacher data lives under/mnt/data/gabriele/gnn_transformer/v6_local_teacher.

Teacher quality is a prospective feasibility gate using all original18800training frame occurrences: conditional0/1 balanced accuracy>=.95, accuracy>=.90 separately forpositive/character-only/room-only/neither, and mean full-vocabulary probability mass p0+p1>=.90. Report calibration and denominators, both deduplicated and original-weighted. If the frozen policy fails, stop this distillation block; do not search prompts/temperatures or silently convert togoldlabels. Initial16software inputs coverfourdeterministicallychosen training pairs per semanticcategory; profile eligibility is computational integrity, never their accuracy.

Auxiliary objective: a shared trainable affine96-to-3 decoder A,b predicts the teacher distribution from each differentiable pre-merge message at the LAST ORIGINAL PROMPT TOKEN. Never use the final teacher-forced answer token. Loss=native answer/EOSCE + mean_over_images KL(teacher||softmax(A*m_i+b)), coefficient1,temperature1. In the control replace m_i by stopgrad(m_i) ONLY for this auxiliary decoder; native count CE remains unchanged. The aligned arm propagates KD into the existing reader. The head is initialized from an independent RNG stream(seed+1000003), Gaussian weightsstd.01 and zerobias, without changing model/branch/LoRA RNG or dataorder. Both arms train it identically using its ownAdamW LR.001/wd0 and separate clip1, excluding head gradients from the37native-parameter clipping calculation. Both optimizers step405times and persist acrossall9blocks. Save native and auxiliary checkpoints under/mnt/ckpts/gabriele/gnn_transformer; the293parameter auxiliary head and teacher are discarded at inference.

Implementation may observe the existing branch inputs and recompute ONLY this one prompt-query message with the exact same projection/read/centering formula, retaining autograd. This training-only recomputation must match the actual branch calculation and preserves one native VLM/vision forward per training example. No frozenV5source changes. Explicit tests cover prompt-index/answer leakage, aligned-versus-detached reader gradients, main RNG preservation, unchanged native inference after observer removal, same imagevisibility, and auxiliary-loss weighting. Teacher inference, all model training, heavy CPU staging/analysis and tests useSlurm.

Generate fresh evaluation sequences BEFORE model outcomes, data seed20260919, using unchanged V2/V4 generator laws:108N16anchors(12perK0..8) with matched negative extensions toN32/N64, plus64N32anchors(8perK9..16) and theirN64extensions,452tests total. Reject complete semantic contexts already present in the training/dev data or prior inspected evaluation manifests, binding the explicit exclusion-manifest inventory in the CPU stageplan. Local rendered atoms may repeat, as in the finite compositional benchmark; this is disclosed rather than called unseen local perception. Existing V4dev staysfixed. All four runs share the fresh frozen test set; software profiles use existing separate software scenes. No V6test evaluation until completed training/dev selection.

Primary aligned-minus-control familiarOOD exact>=5pp in EACH seed, withN16loss<=5pp each. Retain everyinvalidanswer asincorrect. Bootstrap10000wholeanchor draws withseed20260921, retaining bothlengths andfixedseeds; pooledresults are descriptive. Report nonzeroK separately, allperK, unseenK9..16, parse, MAE/bias, localauxiliary quality, clipping and full teacher/training/inferencecost. Practical milestone is separate and stronger:>=70%familiarOODexact and>=60%nonzero-K OODexact ineachalignedseed, in addition to primary. A small relative win below that is not the user objective. No automatic reasoning-composition claim; a useful native result must later support a separately registered direct/reasoning comparison and dependent-query reuse test. The fourrun comparison tests the auxiliary-objective effect; it does NOT establish correspondence-specific attribution without a separate same-target/permuted-image control.

Budget5allocated GPU-hours for this block, maximum4projectGPUs concurrently. Reservation: teacher4shards<=15min each(1GPUh), two software training profiles<=3min each(.10GPUh), fourmains<=50min each(3.333GPUh), fixed selected-checkpoint numerical audit<=5min(.083GPUh); total4.517GPUh leaving boundedsoftware-repair reserve. Profiletimings must support caps before launchingmains. CPU staging/checks/analysis capped and right-sized separately. KeepV5's7436GPU-seconds underitsseparatecompleted4hour cap. Failedjobs/qualitygates remain recorded, sourcefreeze precedesoutputs, no parameter/prompt/seed sweeps.


### V6 parameter arithmetic correction — 2026-09-10 21:17:38 UTC, before implementation/teacher outputs

The affine96-to-3 auxiliary head has96*3+3=291parameters, not293as mistyped above. Deployed nativeparameters remain1,762,400; totaltrainable including theprivatehead is1,762,691. No architecture, objective or resource change.


### V5 decomposition complete; V6 data-binding clarification — 2026-09-10 21:20:37 UTC

Corrected diagnostic CPUselftest441556 passed; decomposition441558 completed6CPU-seconds, all864pairs included. Added-negative message norms dominate mapped-original drift forSUM (N16→N64 means624.86vs10.55 inseed4,673.87vs8.93 inseed5). SUM branch/native-attention ratios approximately double withN, whereasMEANadded/normalizationterms largelycancel. This is descriptive, not a causal proof or full-residual normalization test. Only150/3456mappedimagepairsarebyteidentical;3306rerenderStep labels. Oldfailed441549 remains preserved. No methoddata/prediction changed.

For V6, the unchangedV4 training schedule is copied with identical slot metadata, condition SID sequences and prompt-token audit. The newcombinedmanifest contains the exactoldtrain/devsample records and fresh tests. Schedule metadata hashes/datasetroot therefore bind thisnewmanifest, while preserving the originalV4schedule hash separately. This necessary provenance rebinding does not alter anytraining presentation or selectiondata. Trainingdata_seed stays20260914; fresh_test_seed is20260919.


### V6 manifest interface and numerical checks — 2026-09-10 21:23:12 UTC, before implementation freeze or anyV6model output

Use separatefresh test-manifest/test-count-manifest arguments. The originalV4 train/devmanifests, schedule andtoken-audit files remainliterallyunchanged; no metadata rebinding or combinedmanifest is needed. This replaces only the preceding implementation-binding note and better preservestrainingprovenance. Per-split roots derive from eachmanifest. Freshsample counts/laws/seed/exclusions andalltrainingpresentations are unchanged.

TheV6 inferenceoperator is exactlyV5SUM. RetainV5's explicitlyaccepted unchangednativeNF4/bf16/SDPA/bitsandbytesbackend andits documentednumericalfloor. Softwareprofiles block on zero-initialized identity, finiteupdates/restores, correctprompt-index live messages, aligned/detachedgradientrouting, RNGpreservation and exactnativeinferenceafterobserverremoval. Numericalcached/full observations mayberecorded descriptively; they do not reopen or relax the failedpre-V5strictgate and are not an efficacy/model-selectioncriterion. Allfourdev-selectedV6checkpoints receive the same fixedpost-trainingnumericalaudit, with unchangedcentered/TV/top1thresholds, allfailuresretained. Computationalintegrityfailure stillblocks. No furthernumericalthresholdsearch.


### V6 teacher software launch specification — 2026-09-10 21:29:16 UTC, before anyteacher outputs

TeacherCPUstage source review/syntax passed. The16profilepairs interleave the first4sorted pairs from each semanticcategory. Profile is oneGPU<=2minutes. Prospective runtimeeligibility for each2495-pair shard: measuredmodel-load time + first4pairwalltimes +1.25*2495*max(last12pairwalltimes)+30seconds <=900seconds. This separates a coldfirstprefill fromsteadywork; accuracyneverselects profileeligibility. Fourshards use sortedpairIDs modulo4,2495each, each<=15minutes. CPUmerge recomputesprobabilities/quality and publishes a canonicaleligiblecache onlyif the registeredall-training qualitygatepasses. A failedqualitygate preservesallrawtargets andmergedreport butstopsthetrainingblock. AllCPUstage/mergechecks runthroughSlurm.


### V6 teacher/fresh data complete and training software launch — 2026-09-10 21:42:04 UTC

TeacherCPU441590 passed25seconds, planSHAca04535fb67e065fcde0408fb23036eef51439d9188340199dc81167fe418b19. Profile441592 completed26GPU-seconds; conservativeprojectedshardtime356.17seconds<900cap. Fourshards441596 completed1078GPU-seconds total, all9980targets. CPUmerge441606 verifiedalltargets andpublishedcanonicalcache: original18800frame conditional0/1balancedaccuracy1.0, everypositive/negativecategoryaccuracy1.0, meanfull-vocabularynumericmass~.999839. This is training-domain teacherfeasibility, notnativecountingefficacy. Totalteacherblock1104GPU-seconds.

NativeCPU441601 passed15existing+8newtests andallfourcondition/main-profileschedulechecks in4seconds, freezingthetraining sourceledger. Independentreviewcorrected matchedgradmode forobserver-removalidentityBEFOREthisfreeze. Freshdatadry441605 passed; actualCPU441607 completed43seconds with452uniquecontexts/172families/18240imagereferences/280negativeextensions and16manifestexclusioninventory. Trainingsoftwareprofilearray441613launched withbothcontrol/alignedseed6; freshheldoutdataisnotusedinsoftwareprofiles. NoV6mainresults.

### V6 local-head diagnostic specification — 2026-09-10 21:42:04 UTC, before anyV6main training/outcomes

Clarify the registered localauxiliary-quality report: after completecanonicalmainanalysis, evaluateeachdev-selected291-parameterauxiliaryhead onthatmodel's already-saved final-original-prompt per-frame96-dimensionalmessages forALL452freshtestcontexts. No newfit, thresholdchoice, teacherforward, VLMforward ormodelselection. UseitsordinaryFP32three-waysoftmax andargmax0/1/other; reportfull2-by-3confusions against independentlyrecountedconjunctiontruth, positive recall, falsepositiverate, explicitother/abstentionrate, three-waycorrectness andbalancedclassaccuracy(otheriswrong), plus tie-correctAUROC ofp1. IncludeallN/K andnegativecategories; noexternalcountorclassifierpredictionentersnativeinference. This testswhetheranytrainedlocalcompetencetransfer accompaniesnativecountchanges. SeparateCPU4/16G<=5min, sourcefrozenbeforeuse, allcheckpoint/message/manifest/analysis hashesbound.


### V6 prospective main timing release (2026-09-10, before main launch)

Both computational profiles passed (441613 array, 49 allocated GPU seconds each). The independent runtime projection uses1620 times the profile mean training-presentation time, nine72-item dev sweeps using the larger observed time for each dev length, and452 test items with profile per-cell timing, multiplied by1.25 plus120 seconds. Projected control2327.095s and aligned2339.551s both fit the unchanged3000s cap. Accuracy on the ten software cases is not an eligibility criterion. All V6 GPU consumption so far1202s, including teacher profile and all four cache shards.


### V6 reporting check repair before main launch (2026-09-10)

CPU report gate441641 verified all fresh data and the complete teacher raw-shard chain, then failed exact dictionary equality of teacher quality: archived quality accumulated pairs in shard order whereas the independent check uses sorted published-index order. The floating numeric-mass mean is now compared at1e-12 tolerance; eligibility booleans, thresholds, class accuracies and all raw target bytes remain exact. This is a reporting arithmetic repair, with no teacher/data/training change and no gate threshold change. The failed CPU log is retained.


### V6 main release passed and sources frozen (2026-09-10, before main submission)

Independent CPU441647 passed unchanged V4 train/dev, all452fresh test QA/image/exclusion records,9980raw teacher target reconstruction and teacher eligibility, and both computational profiles. CPU441633passed report decision/auxiliary self-tests;441635passed saved-head diagnostic self-tests. Analysis sources and fixed post-checkpoint audit implementation are frozen in v6/analysis_source_hashes.json with code snapshots. Four mains now released under the existing5GPUh cap; no architecture, optimizer, data, selection, or decision criterion has changed.


### V6 prospective head-subspace diagnostic (2026-09-10, mains running; before fresh test outputs)

After all four canonical native V6 runs, analyze every saved final-original-prompt message and every dev-selected auxiliary head. This is a CPU-only descriptive analysis; no new model outputs, fitting, threshold optimization, checkpoint choice, or changes to native predictions. All452contexts/18240frame occurrences per checkpoint remain included. Head A has shape3×96; define C=I−11ᵀ/3, R=CA, P=R⁺R and Q=I−P by FP64 SVD with relative singular-value tolerance1e−10, and verify rank≤2. Record singular values/rank/conditioning and actual numerical errors. The real-arithmetic identity softmax(APm+b)=softmax(Am+b) follows because CAQ=0; retain the original bias. Verify this numerically, with separate FP64 algebra and actual FP32-head errors, without claiming machine-bit equality.

Report per-N/K and semantic category message norms, native write norms through the fixed trained U, positive/negative/total SUM components, P/Q decomposition, and relevant vector alignments. Use UᵀU for small exact-equivalent norm computations where appropriate. Independently hash-bound checkpoints, all452message artifacts, source/analysis/data manifests and QA recounts. Local labels are descriptive audits only. Call Q head-invisible, not nuisance: native CE trains these directions, the head changes throughout training, and the instantaneous ≤2-dimensional KL gradient is not a claim that only two representation dimensions were ever trained. Head-visible negative logit margins can also accumulate with length; probability preservation is not count preservation. No diagnostic outcome establishes a causal repair, and a GPU projection intervention would require its own prospective fixed design. Freeze diagnostic source and CPU self-tests before reading its results; five CPU-minute cap, zero GPU allocation.


### Parallel native local computation: prospective software oracle (2026-09-10)

Before any oracle outputs, register a bounded implementation check independent of V6 efficacy. Motivation: isolated frozen Qwen already passed the V6 teacher feasibility test, whereas a learned rank96 reader need not reproduce full pretrained local computation. This check asks only whether ordinary native batched N1 teacher prompts retain that computation as batch size grows. It does not add an aggregation operator, train a model, measure native aggregate counts, or establish reasoning composition.

Use the exact immutable V6 teacher plan/cache, Qwen2.5-VL7B NF4/bfloat16 SDPA, complete N1 count/chat/assistant-prefix prompt, original rendered image and processor. Select the first16 sorted teacher pair IDs within each of positive/char_only/room_only/neither, interleave categories in that order, giving64 fixed pairs. Evaluate nested batch sizes1,8,16,64 and reverse-order64. Also run the first four pairs separately on the same loaded model to distinguish archived-source discrepancies from batch numerical effects. No confidence/outcome filtering or alternative prompt search.

Left-padding must preserve every row's actual unpadded token IDs, original processed pixel tensor/grid, and all three native isolated mRoPE axes. One frozen native VLM and visual invocation per batch, with independent batch rows. Save each0/1 raw logit, full-vocabulary log normalizer and numeric/other probabilities, calls, elapsed time and peak memory. Eligibility requires exact token/pixel/grid/position correspondence, exact conditional0/1 predictions versus the archived teacher on every row, maximum conditional-p1 difference≤.02 and maximum numeric-mass difference≤.02. Raw logits and ordering effects are descriptive; no post-hoc tolerance revision. Every failure stays recorded.

Freeze CPU input/source plan before GPU execution. Wrapper limit3GPUminutes; this oracle has a separate total0.1GPUh envelope including any separately documented computational repair, and it must not increase concurrent project GPU use above4. All GPU/heavy CPU work is submitted through Slurm; new tensor artifacts use /mnt/data/gabriele/gnn_transformer. No checkpoint is required. The initial oracle does not test generation or cache parity. A subsequent packed global-output architecture and broadcast-token cache test would require their own exact protocol. Known relation: full local query processing followed by learned set fusion is FiD/DeepSets-like and related to the earlier fenced replicas; no new attention family is claimed.


### Parallel local oracle CPU release (2026-09-10, before GPU outputs)

CPU441679 passed the fixed64-pair selection, all five ordinary batched-processor parity checks, every isolated token/pixel/grid/mRoPE identity, and synthetic numerical/packing self-tests. Frozen planSHA c33c48dce427bf097b727e8322441aa391ced35bce2ab8f9aea7309988a86aeb. Numerical eligibility uses archived-teacher comparisons exactly as registered; freshserial overlap differences and truth checks are retained descriptively, without an additional acceptance criterion. GPU oracle released at the existing3-minute job limit and separate0.1GPUh envelope.


### V6 final-report interface repair (2026-09-10, after completed mains)

CPU441662 independently verified all four full native runs, auxiliary ledgers and fixed post-checkpoint audits, then failed before producing analysis.json at the shared-runtime helper: its caller supplied a keyed dictionary, whereas the frozen V4 helper expects a list of four provenance records. Change only that caller to the same list interface used by V5. No model, predictions, data, checkpoint selection, criterion or arithmetic changed. Original source snapshot and analysis hash ledger retained as analysis_source_hashes_before_441662_fix.json; replacement snapshot analysis_code_after_441662_fix. Report source hash 225f6777e5280376686897e537f74571ef5843e0ce105542d523372af192f2c2 -> 9f5835a37c9f8e754de4aee3d469452859a914e8e3b40b1ab5c4c6795e110bd6. The24CPU-second failed job/log remains in the record.


### V6 dependent diagnostic snapshot binding repair (2026-09-10)

Final independent report441682 completed and both primary/practical criteria failed. The pending local-head441663 and subspace441668 CPUjobs each stopped after2seconds before diagnostic outputs because their canonical analysis_code snapshot still contained the pre441662 runtime-interface source, whereas analysis.json correctly bound the repaired reporter. Preserve that complete old snapshot as analysis_code_before_441662_fix; update canonical analysis_code from the already archived/hash-verified analysis_code_after_441662_fix. No diagnostic code, source ledger, numerical rule, model, predictions or analysis changed. Rerun the same two frozen diagnostics. Both failed logs remain.


### V6 complete and parallel-local oracle outcomes — 2026-09-10T22:45:23.154819+00:00

Independent final report441682 verified all1808native predictions. FamiliarOOD aligned/control:24/15 of216(seed6),12/14(seed7). Both-seed primary and practical criteria FAILED. Aligned6 answeredzero throughout all216familiarOOD; allfourmodels got0/128unseen-count exact. Seed7aligned N64 parsed107/108, mostly8080/8008/8088, so parseability is not meaningful counting. V6 total8320GPU-seconds(2.31111h), including1104teacher,98profiles,6962mains,156selected-checkpoint audit. All20post-training strict/state observations passed; historical V5 failures remain.

Frozen local-head diagnostic441687 completed9CPU-seconds: aligned N64 AUROC.5439/.5276, balancedaccuracy near.5; aligned6 labels every familiarOOD frame negative. Frozen head-subspace441688 completed15CPU-seconds: allheads rank2, preserved probabilities (maxFP64TV1.28e-15,FP32TV2.39e-7,noargmaxflips); substantial negative sums in bothvisible/invisiblecomponents, withvisiblecomponentlarger in everyarm/cell. This does not establish cleanextraction, nuisance-nullspace, causalnormalization, or an aggregationgain. No projectionintervention launched.

Separate parallel-local CPU441679/GPU441680 oracle passed. All157conditionalbinary predictions matched archivedteacher; maximumconditionalp1drift.01479938945 andnumericmassdrift.000127339606, within unchanged.02limits. Batch64forward1.305768s,peak8.192GiB;30allocatedGPU-seconds total. Allselectedpromptslength265, so actualGPUbatcheshadzeroleftpadding; no fullhiddenstate, fullvocabulary, fusion, cache, or reasoning equivalence claim. Canonical artifacts and independent result note retain allscalarcomparisons/sourcebindings.

### Parallel local plus global/cache software check — 2026-09-10T22:45:23.154819+00:00, before outputs

Separate architecture-independent software check, not a training or efficacy experiment. Use two fixed old V4 software scenes from v4_diversity/profile_manifest.json: v2c_profile_test_software_N16_K3_0000_N16 (Emma/Kitchen) and v2c_profile_test_software_N16_K6_0000_N64 (Peter/Garden). Form N unchanged complete native N1 image/question/assistant-prefix prompts, plus a final text-only build_count_prompt(question,N) global row with no image placeholders. Left pad native token rows; preserve pixels/grid and isolated logical mRoPE. Genuine heterogeneous padding must occur. CPU compares ordinary native processor packing and freezes source/data/tensor plan before GPU.

On one frozen native Qwen2.5-VL7B NF4/bf16/SDPA instance, run fresh serial references for all80localrows and2text-onlyglobals. Each mixedscene runs cachedprefill, then forced common literal single ordinary nonEOS token 'Therefore' followed by ':' broadcast to everyrow; compare each cachedprefix with a fresh uncached fullprefix. Require both literalstrings to encode as one ordinary nonEOStoken, or stop without searching alternatives. Total92VLMcalls and86visualcalls; actual invocation ledger must match. Capture pre-final-RMS states and rawfull-vocabulary outputs for comparisons, with source/input/cache-state provenance underdata root.

Computational gate: exact native token/pixel/grid ordering, per-row unpadded masks, logicalmRoPE and cache correspondence, finite tensors, genuine padding. Local mixedprefill compared to corresponding freshserial requires exact conditional0/1prediction plus maximumconditionalp1 andnumericmassdifferences<=.02. Global mixedprefill versusserial and everyrow cached/full-prefix comparison require full-vocabularyTV<=.02 andexacttop1. Hiddenrelative and centeredlogit errors are descriptive; no hiddenequality claim. Everyfailure retained, no post-hoc threshold changes. Forcedtokens onlyteststate mechanics, notgeneratedreasoningorcomposition.

Separate total0.1allocatedGPU-hour envelope, initial3-minuteGPUlimit, max4concurrentprojectGPUs. CPUstaging/checks viaSlurm; no checkpoint/training. Newsource files preserve the completed parallel_local oracle. The core learnedfusion/readout can be implemented/tested independently onCPU, but any actualnativecounttraining or syntheticperfect-evidence readout diagnostic requires a separate fixedprotocol before outputs.


### Synthetic perfect-evidence native first-token readout capacity — 2026-09-10T22:48:01.302852+00:00, before outputs

This diagnostic supplies the correct count as a synthetic latent z=K e1; it is explicitly an oracle and cannot count as MMReD method accuracy. It asks whether the proposed small readout can fit native count-token outputs at all. No image judgments, learned local extractor, generated complete answer, EOS, multi-digit count, or reasoning benefit is tested.

Use the first8 lexicographically sorted distinct questionstrings in the unchanged V6 training teacher plan (no test/outcome-based selection). First4 are fitquestions, next4 heldquestions. For eachquestion, construct native text-only build_count_prompt(question,N) chat/assistant-prefix prompts for N8,16,32,64, with no imageplaceholder. Extract each of32frozen native pre-final-RMS globalstates serially through unchanged Qwen2.5-VL7B NF4/bf16/SDPA. Repeat eachstate across K0..8; verify all9correct first-token numerals are distinct ordinary single tokens. Fit4questions×2lengths(N8/16)×9counts=72rows. Finalheldcells: newquestions atN8/16, fitquestions atN32/64, and newquestions atN32/64,72rowseach. K9+ is excluded because this first-token check would conflate multi-digit answers.

Instantiate identical seed20260923 ParallelLocalAggregation cores (H3584,r96); bypass encode/aggregate and feed z=K e1 into their actual decode(z,h_g). Freeze unused local/local_bias; train query, aggregate_projection andup. Both arms have the same learned globaloffset, matrices andparametercount; only post_activation differs (identity versusSiLU). Thus the affinecontrol can change its intercepts and is not structurally forced to retain the frozen K0 answer. U startszero. Fusedstate=(h_g+delta) in nativehidden dtype, then actual frozen native finalRMSNorm andunrestrictedvocabularyhead. No numeric-vocabulary mask, teacherclassificationhead, outputbias, or arithmeticdecoder.

Fit eacharm500full-batch AdamWsteps onall72fitrows,LR.001,weightdecay0,clipnorm1. No dropout, earlystop, held-selection, alternateinitialization or hyperparametersearch. Sameinitialparameterbytes andfitorder. Save trainloss/clipping/parameterdigests andfinalcheckpoint underckptroot. Onlyfinalmodel evaluatesheldcells. Prospective fitcapacity criterion=100%nativefull-vocabularytop1 onall72fitrows; heldcellsdescriptive. Failure means this optimization/configuration didnotfit; it isnotanimpossibilityproof. Trainingandheld results remain oracle readout results, never nativevision accuracy.

Beforetraining, verify zero-U cachedhead replay againstactualnativeoutputs withfull-vocabularyTV<=.02 andexacttop1 for all32globalprompts; require exactcapturedstate/input/source/nativeheadbindings andfinitegradients. CPU freezesquestion/token/sourceplan and self-tests; allheavywork viaSlurm. OneinitialGPUjob<=5minutes, separatetotal0.2allocatedGPU-hour envelope includingboundedsoftware-repairs, max4projectGPUs. This diagnostic is independent of mixedlocal/cache acceptance: it uses text-only serial states. Fullnativeaggregation training requires itsownprotocol after these bounded checks.


### Parallel core and synthetic readout CPU release — 2026-09-10T22:56:29.145806+00:00

Standalonecore441718 passed all10tests,2CPU-seconds; coreSHAed600b0ae865108093b74bb1bcf30f905dc38a8a67cb7708cb9f1688f2df7b0f. CapacityCPU441719 passed9CPU-seconds, exactnativeprompt-boundary numerals and source/question/model bindings; frozenplanSHAbafd2d7ef06435b78d5b5c01a7b8212dbe8ba985a22d1be9fc7ddcad01d93c63. Matched500-step oracle readouts released underexisting5-minutejoblimit/separate0.2GPUhcap. No heldoutcapacity or modelaccuracy outputs were available atrelease.


### Synthetic readout initial outcomes — 2026-09-10T23:01:03.291825+00:00

GPU441720 completed29allocatedGPU-seconds. Exactnativeheadreplay(TV0/top1exact) onall32serialtext-onlystates. Actualnativehidden/norm/headdtype isFP16;4-bitcompute remainsBF16, as inunchangedruntime. Bothfit-capacitycriteriaFAILED: affine13/72,SiLU62/72. Heldoraclecells(newquestions,newN,both) were13/21/21 and41/30/17 of72 respectively; theseare notvisionaccuracy, first-tokenonly, anddidnotselectmodels. Fixed500-stepoptimizer clipped451/439updates; affinelastfivefitlosses.802,1.168,1.558,1.034,.878 followedfinalfitNLL1.110, indicatingunstableoptimization. No architectureimpossibilityorusefulnativeaggregation claim. Originalcheckpoints/predictions/frozen sources retained. Furtheroptimizationneedsnewprospectiveregistration, andheldoraclecellswouldbeexplicitreuse.


### Synthetic readout optimization sequel — 2026-09-10T23:04:03.941474+00:00, before sequel outputs

Retain initial441720failedfit result unchanged. This separately identified software optimizer diagnostic addresses incomplete/oscillatoryfit, not MMReDefficacy or an independent test. Use the exact32cached pre-final-RMS states andnative reference logits fromreadout_capacity/run_441720, hashbound byits summary/stateidentity. No newVLM/vision/languageforward isrequired. Reload the identicalfrozen native norm/head and verify cachedheadreplay under the existingTV.02/exacttop1rule plus exactnorm/head tensorhashes.

From the same freshseed20260923initialized core, fitbothidentity/SiLU arms for5000full-batchupdates on the exactsame72fitrows. KeepAdamW/wd0/clip1 andnativeFP16 residual/norm/head. Changeonly the registered optimization procedure: learningrate atupdate t=1..50 is.001*t/50; t=51..5000 uses1e-5 +(.001-1e-5)*(1+cos(pi*(t-50)/4950))/2. Thus50-stepwarmup, cosine decay to1e-5. This jointlychangesdurationandschedule; it doesnotisolatewhichcausedanydifference. No earlystop, alternate seeds, hyperparametergrid orheld-selection. Finalcheckpoint only; fitcriterion100%of72 unchanged. Previouslyobservedthreeheldoraclecells areexplicitdescriptivereuse; no newextrapolation claim orvisionaccuracy. Preserve rawfulltrainingcurves/margins/norms andallfailures.

NEW source/wrappers/outputsubgroup readout_capacity/decay, leavinginitialsourcesimmutable. CPUsource/state/protocolfreeze precedesGPU. Initialsequeljob<=5minutes, within existingtotal0.2GPUhcapacityenvelope (29GPU-seconds alreadyspent,691remaining), max4concurrentprojectGPUs. ActualnativeMMReDtraining remainsunreleased pendingsoftware/readoutfeasibility.


### Mixed local/global cache CPU release — 2026-09-10T23:05:23.391405+00:00

CPU441727 passed15CPU-seconds; frozenplanSHA0dd28471db37493db053977821db203d589440801345a899a9a3554e804d5675. Bothmixedcasesexercisenonzero globalrowpadding; nativeprocessor/token/pixel/mRoPEpreservationchecks pass. Fullnativevocabularylogits andpre-finalRMSstates willbesaved forindependentnumericalaudit, withactualresidual/norm/head dtypeobservations. Initial3-minuteGPUprobe released underexistingseparate0.1GPUhcap; noaggregationtrainingoraccuracyclaim.


### Synthetic optimizer sequel CPU release — 2026-09-10T23:06:15.107650+00:00

CPU441730 passed9CPU-seconds; exactparentstate/source/teacherbindings, promptboundary andfixedscheduleendpoint/monotonicitychecks. FrozenplanSHA272ffad98c2387564f2a7f5f106d049cadc9717de0a788b424adc84c740dcf38. Rootreviewed source diff beforeGPU; registered5000-step sequel released at5minutes withinparent0.2GPUhcap. Original441720failure unchanged.


### Readout sequel and mixed/cache initial outcomes — 2026-09-10T23:09:46.250615+00:00

Readoutdecay441733completed42GPU-seconds,bothfinalfit72/72. Minimumfitgoldmargins1.46875affine/2.578125SiLU; finalfitNLL.18320/.07067. Reusedheldnewquestions67/62,newN49/49,both45/37 of72. CorrectKoracle bypassesevidence, so no nativevisionextrapolation/completeanswer/reasoningclaim. Initial500-stepfailurepreserved; totalcapacity71GPU-secondsunder.2hcap.

Mixed441729completed42GPU-seconds. Allcomputationalcacheprefix/mask/mRoPE/token/pixelchecks pass. StrictnumericalgateFAILED1/246: N64local062cached_vs_full_step2 fullvocabularyTV.0202964635694>.02,top1same. All246top1equal; all80localprefillconditional01checks pass(maxp1drift.0132944,mass.00002786). Bothrealglobalrowshave196leftpads. Nativehidden/norm/head/logitsactualFP16; NF4BF16computeunchanged. No gate/toleranceretroactivelychanged. Standalone nativehookCPU441731passed21tests(10core+11integration),3CPU-seconds; actualHFlearnedfusiongenerationnotyettested.

### Mixed/cache discrepancy localization — 2026-09-10T23:09:46.250615+00:00, before outputs

Retain441729strictfailure. Independentlyrecomputeallsaved full-logit comparisons/statebindings viaSlurmCPU, with1e-10agreement for archivedFP64metrics andunchanged.02/top1decisions. Actualfirstlayercausalmask andK/Vtensorswerenotsaved, so those remain source-boundGPUassertions; do notcalltheCPUauditaK/Vreplay.

SeparatefixedGPUlocalizationwithinexistingmixed.1GPUhenvelope: exactN64local062completeN1prompt serialprefill, thenforcedThereforeandcolon cached/fullprefix atbothsteps; thenexactoriginal65-rowmixedscene performsidenticalprefill/two-stepcached/fullcheck. TenVLM/sixvisualcalls,330rowoutputs. Samefrozenweights/runtime/processor/tokens, no precision/kernelpatch, nochangedthreshold. Comparefailedrow'sserialandmixedcached/fullTV/top1/hiddenerrors, all65repeatedmixedrows, andrepeat-versus-original outputs. Thisdescribesbatch/cache/nativenumericalbehavior; cannotretroactivelypass441729orproveall drift harmless. CPUsource/inputfreeze first;initialGPUjob<=3minutes,42GPU-secondsalreadyspentso318remainunderoriginal360secondcap. No aggregationefficacyoutputs.


### V7 preparation: frozen local features and fresh evaluation — 2026-09-10T23:12:47.548024+00:00, before outputs

This registers data/feature preparation only; fullnativeaggregation training/comparison remains a separate forthcoming protocol. PreserveallV1–V6results andsources. Motivation: fullfrozenN1localcomputation isavailable, and bothnativepostmergereadouts canfittheperfect-evidence trainingoracle afterfixedoptimization. Neither provesvisionaggregationperformance.

Fix theglobalstream prompt uniformly acrossN: 'You will be shown frames describing steps in a house.\nRespond with a single non-negative integer (0 is allowed). Output only the integer.\nQuestion: {original_question}\nAnswer: '. Localstreams retain exactcanonicalN1teacherprompts. Theglobalstream omits explicitset-size metadata; N enters via evidence aggregation. Forfixedquestion/generatedprefix, globalprompttokens areidenticalacrossN. This invariant isbyconstruction, not an extrapolationresult. No promptsearch. Originalquestion semantics, imagepixels/Step labels, processor, frozenQwen NF4/BF16compute/nativeFP16residual/head andnativevocabulary remainunchanged.

Prepare training-only pre-final-RMS features fromliteral V4refresh1540scenes:9980uniqueempty-prefixlocal(imageSHA,question) entries plusonlyactuallyoccurring(pair,precedinggoldcounttoken) entries (atmost18800), anduniqueglobal(question,empty or precedinggoldtoken) entries. For eachscene, cachemaps local[N,2,H] andglobal[2,H] tocount/EOS targets. Empty-prefixfeatures areextractedindependentlyofgold andneverchosenfromgold-specificruns; goldprefixfeatures areusedonlytopredictEOS. Verify alltrainingcounts0..8 areexactsingletokencontinuations. No teacherauxiliaryloss/goldlocalclassifier/tally. Fourdeterministicshards partitioneachlocal-empty/local-prefix/global-empty/global-prefixgroup separately; no empty/gold-prefixmixture duringextraction. Freeze exactfeatureIDs, tokens, processedpixels/grids, isolatedmRoPE, model/backend/source/tensorhashes andscenemaps onSlurmCPU. Reconstruct864uniqueprocessedimagesonceandverifyagainstimmutableV6teacherpixelSHA; pixels/featuresstoredin/mnt/data/gabriele/gnn_transformer/v7_parallel_local.

Fixedcacheprofile32distinctfeatures:12localempty(first3sortedpersemanticcategory),12observedlocalprefix coveringcategories andavailabledigits0/1/8 bydeterministicCPUselection,4globalempty,4globalprefix. Freeze IDs beforeoutputs. Eachoffourgroups runsits smallbatch anditscyclicallyrepeated64-rowstressbatch (8nativecalls). Require exacttoken/pixel/grid/mRoPE/callcorrespondence, finitestates, andcapturedhidden→actualfrozennorm/headreplay fullvocabularyTV<=.02/exacttop1; rawfailures retained. No taskaccuracyeligibility. CPUcode/sourcefreeze first, GPUprofile<=3minutes. Projectshardtime=measuredmodelload +1.5*sum(featurecount_bygroup*B64groupseconds_perrow)+30seconds; allshardsmustfit300secondscaps before4shardmaincachelaunch. Cachebudgetseparate0.4allocatedGPUh(1440s), max4projectGPUs; initial4shardseach<=5min plusprofile<=3min,CPUmergehash/auditallfeatures. Trainingfeaturecache doesnotreplace eventualnativeevaluation.

Prepare fresh V7tests before modeloutcomes usingseed20260924 andunchanged V2/V4generatorlaws:108N16anchors(12perK0..8), matchednegativeextensionstoN32/N64, plus64N32anchors(8perK9..16)andN64extensions;452contexts/172families/18240imagereferences/280extensions. Excludeall16priorcanonicalinventorymanifests PLUSbothV6freshmain/countmanifests (18total) atcompletecontext/questionlevel. Finiterenderedatoms mayrecur; Step labelsrerenderonextension. Keeptraining/dev/schedulesliterallyunchanged. Newroot/mnt/data/gabriele/gnn_transformer/v7_fresh, SlurmCPU4/16G<=5minstage andindependentsemantic/image/extensionaudit. Noheldoutfeaturecache or modeltestevaluationinthispreparation.


### V7 preparation amendment: balance question–count support before mains — 2026-09-10T23:18:16.082913+00:00, before new training/data outputs

Metadata inspection found V4refresh1540scenes coveronly149question×Kpairs across52questions; countsupporthistogram is{1:9,2:14,3:13,4:10,5:3,6:3}. Refreshchangedscenecontentswhilepreservingbasequestion/countslots. This offersquestion→answer shortcuts and motivates changingtrainingdata, not claiming an architecture-only comparison againstoldV6. No oldfeaturecacheGPU waslaunched. EarlierV7preparationregistration remainshistorical; amend trainingfeature source to the balanced corpus below, keepingits frozenmodel/localprompt/globalprompt/empty-vs-goldprefix principles. The oldfresh-data drycheck441754passed butdidnotpublishdata; retainits source/plan snapshot and repeatfreeze after the exclusioninventory amendment.

Newtrainingroot /mnt/data/gabriele/gnn_transformer/v7_balanced, seed20260925. Crossall9canonicalcharacters×6rooms×K0..8×N8/16. Drawtwoindependentdistinctcontexts percell usingunchangedV2conjunctivedistractor/shufflelaw, except saturatedN8K8hasonepossiblecontext perquestion:54*(8*2+1+9*2)=1890uniquecontexts. Onebalanced presentationepochcontains1944slots, usingeachordinarycontextonceandrepeatingeachof54saturatedcontexts twice. Thereforeeachquestion/NhasauniformK0..8presentationdistribution; no imageoutcomesorconfidence filtering.

For nonsaturatedtrainingcontexts rejectall18priorcanonicalcontext/questionhashes andwithin-corpusduplicates. SaturatedN8K8necessarilyusesitsuniquecontext; historicalreuse isexplicitlyallowedandindividuallyrecorded (no novelty/freshnessclaimforthose54contexts). This exception appliesonlytotrainingN8K8. Newdevcontains72freshN16contexts(8perK0..8), drawnfromuniformcanonicaltargets withindependentseed20260926; rejectallpriorandnewtrainingcontexts. ThereisnoN8developmentcell, avoidinginevitablesaturatedtrain/devduplicates. Count/target/category/image/QAhashaudits precede models. Olddatasets/dev/schedulesremainunchangedandarenotusedforV7selection.

V7freshtestseed20260924,laws/452contexts/172families remainfixed. Its exclusioninventory now adds the newbalancedtrain/devmanifest to the18historicalmanifests (19total). Re-run sourcefreeze/dryauditbeforepublication; this supersedes only the unpublished441754dryplan. Allcurrenttrain/dev/testcontexts mustbedisjoint; finiteimageatoms stillrecur. Traininggenerator exceptions do not relax heldoutfreshness.

The intendedfourmainruns are fullparallel-localSUM+SiLU versus ordinaryjoint-stateadaptation with the identical1,041,600parametercore, seeds8/9. Thejointarm passesitsfinalnativequeryh asbothglobalstateandonesinglelocalelementto the samecore; allparametersareactive. Bothuse thesameN-agnosticglobalinstruction, originalimages, frozenmodel and nativevocabularyhead. Thiscompareswholelocal-computation/fusion againstmatchedordinaryadaptation, notSUMalone. AffinevsSiLU isdeferredbecausebothalreadyfit thecapacityoracle. Fulltrainingbudget/nativecacheprofiles/exactselection/efficacycriteria willbe separatelyregistered before mains. Trainingfeaturecacheforjointpath alsoneeds itsownprofile/reservation.


### V7 balanced-data CPU dry release — 2026-09-10T23:23:08.802807+00:00

CPU441764passed deterministicgenerationof1890train+72dev and1944balancedslots, fixedglobalprompttext, exclusions anduniform972question/N/Kcells. PlanSHA1e9bd9a17f0cc2ed1e374295c0b05ff98d731b311cf528f28bf8fc9aa41855e2. ActualCPUrender/publicationreleasedat5minutes,withindependentpublishedQA/image/categoryaudits. No modeloutputs.


### V7 balanced corpus published and mixed-localization release — 2026-09-10T23:25:55.829212+00:00

BalancedCPU441770completed31seconds:1890train/72dev,1944balancedslots,all972question/N/Kcellsweight2,864uniqueimages/24048references. Exactly39of54saturatedN8K8contextsreusedhistoricalcontentasprospectivelypermitted; allnonsaturated/newdevcontextsfreshandcurrenttraindevdisjoint. ManifestSHA37228e51c4cd95985931a14e391cd5024de0ca31b923ed20e998e79b80cb52b3; scheduleSHA4a15a63a050b601031f25a73ca84270be5a8e0de3d6e6f92ed346bb2bccdcdaf. FreshV7teststager nowbindsthis19thmanifest andbalanced schedule; old441754drysource/planretained.

IndependentmixedCPUaudit441766completed10seconds,reproducedall246comparisonsfrom492full-logitvectors/92states,retainingtheoneTVfailure. Selftest441762used52CPU-seconds. LocalizationCPU441767passed9seconds, frozenplanSHA693d59842d82413a6a24ac7e3ca29690a8db57b7774379a1e2f2ae0e5a8360e5; independentstaticreviewfoundnoblocker. Ten-forwardlocalizationreleasedat3minuteswithunchangednumericalpolicy.


### V7 joint-feature profile and native software release — 2026-09-10T23:39:36.030920+00:00, before outputs

Joint CPU staging binds all1890 balanced training scenes and3780 empty/count-prefix states, balanced semantic/staging/prior audit ancestors, exact shared pixels, native prompt/token/mRoPE/source/model identities. Fixed profile scenes are the first sorted SID for each K0,1,4,8 at each N8/16. Four groups N8empty,N8prefix,N16empty,N16prefix each run B1 first row then B4 all:8nativeVLM/vision calls,16distinct states,20row outputs. Computational correspondence and captured-hidden→actual native norm/head replay TV<=.02/exacttop1 gate; B1/B4 differences are descriptive. No count accuracy scored. Initial CPU10min and GPU3min, separate0.1GPUh prep/profile envelope. Full joint harvest requires measured timing and its own release; projected four-shard cost uses load+pixelverify+1.5*sum(ceil(group rows/4)*B4time)+30.

Mixed localization441772 completed33GPU-seconds: all325 repeated mixed hidden/logit vectors bitexact to441729; sameone TV.02029646 failure. Failed row alone cached/full TV.01743662; single-versus-mixed full-prefix TV.02190883, alltop1same. Native batch/cache floating effects occur without learnedfusion. Original strict.02 gate remains FAILED, independently audited. This prospective engineering release permits V7 integration/training on the unchanged native backend if exact token/pixel/logicalposition/cache/call assertions pass, finite updates/restore pass, and actual active-branch generation is audited. Cached/full numerical TV/top1 and all failures remain reported descriptively, without threshold search or claims of exact arithmetic equivalence. New captured-state norm/head replay gates remain binding. Total mixed75GPU-seconds under360cap.

Fresh test dry441773 passed19-exclusion inventory and balanced current train/dev protection; planSHAdc637c9e83adf3fd5c7fdf37623835f0fd77fa7ed89bd36eced2e2aeaf53a03f. Actual rendering released atCPU5min. Parallel feature CPU441775 completed40seconds:32229training states,864processed images, frozenplanSHA5a30b412504d8cd8636c4446a2d83a812b60c37d30e65fc32b4480c1e8ceab65. Supplemental ancestor release precedes parallel GPUprofile; no frozen source edits.


### V7 native integration and matched main experiment — 2026-09-10T23:46:27.327961+00:00, before any V7 training outputs

Nativeintegration: fourfixedarm×scene cases (parallel/joint, oldsoftwareN16K3/N64K6 frommixedcheck), newsharedN-agnosticglobalprompt. Eachruns unmodifiednativebaseline, identicalzero-Ubranch, andseed20260927corewithfixednormalUstd.001;12greedygenerations,max4tokens. Exactzero-Uglobalrawlogit/IDidentityrequired. Replayeverycandidate generatedprefix throughfullnativeforward (<=32replays;<=80VLM/44visiontotal). No goldprefix, fitting, accuracygate oroutcomeselection. Exactnativeinputs/positions/causalvalidquerymasks/cacheprefix/callcounts andfinitecomputationsgate; cached/fullTV.02/top1reporteddescriptivelyunderexplicitengineeringrelease. CPUsource/inputfreeze first, initialGPU5min,separate.2GPUhcap. Recordtimingbyarm/Nbefore mainruntimeeligibility.

Mains: parallel completeN1localstreams+text-onlyglobal versusordinaryallimagejointqueryadaptation; same1041600parameterFP32SUM/SiLUcore, allfrozenQwenbackbone, nativeFP16carry/finalnorm/unrestrictedhead. Joint suppliesfinalh asglobalandonesinglelocalelement. Seeds8/9, sameinitialparameterbytes andtrainingorders withinseed. Balanced1890scenes/1944weightedslots,40independentlyshuffledcompleteepochs, RNG=random.Random(seed) advancedacrossepochs, carrybatchesacrossepochboundaries:77760presentations,4860batch16updates. Counttoken+EOSCE equallyweightedover32targetpositions/batch, noauxiliaryloss. ParallelN8rawstatespadwithzeroonlyinthebranch toN16; noextraVLMimages. Cachedempty-prefixstate isindependentofgold; goldcountprefix isusedonlytopredictEOS.

AdamWlr.001,wd0,clipnorm1; steps1..50lr=.001*t/50; thereafter1e-5+(.001-1e-5)*(1+cos(pi*(t-50)/4810))/2. Noearlystop/alternatefitsearch. Nativegenerationdev72newN16 atsteps972/1944/2916/3888/4860, retainall5checkpoints. Selecthighestexactcount,thenlowestmeanrawfirst-tokenNLL,thenearlieststep. Evaluateall452freshtestcontexts onceusingselectedcheckpoint. Greedy1beam,max4tokens,repetitionpenalty1,nooutputmask; nativeEOSlist[151645,151643], trainingEOS151645. ThisEOSpolicyisidenticalacrossV7arms butdiffersfromhistoricalsingleEOSV4–V6. Parser isfullstrippedASCII[0-9]+; leadingzerosparseint; retainmalformed/truncated/incompleteanswers anddenominators. Completionreportedseparately; exactisparsedcount==gold. No gold-informeddev/testprefix. Saveallglobalrawlogits and executedIDs/call/layoutledgers.

Primary: parallel-joint familiarOOD(N32/64,K0..8) gain>=5pp ANDN16loss<=5pp in EACHseed (>=11additionalof216OOD,<=5fewerof108N16). Practical: primaryplusparallel>=70%OOD(>=152/216) and>=60%nonzeroOOD(>=116/192), EACHseed. Both-seedfailureneverrescuedbypooling. UnseenK9..16(all128)separatewhole-answersecondary. PairedK-stratifiedfamilybootstrap10000replicates,seed20260928,percentile95%CI; resample12familiesperK0..8 retainingalllengths/arms/seeds; unseen8familiesperK9..16separate. PooledCIonlytest-familyuncertainty,nottraining-seeduncertainty; CIisdescriptivenotanaddedgate.

Beforemains: independentCPUdata/source/reportgate plus32-update trainingsoftwareprofilesonseed8botharms(3GPUmin each), finiteallparametergradients/backbonefreeze/save-restore. OnfirstsortedsidatN8/N16, profilefinalbranch comparescache-derivednativeheadoutputsagainstactualnativeemptyandgoldcount-prefixforwards(4/arm), allTV/top1andhiddenerrorsretaineddescriptively; capturedactualh→samebranch/norm/headreplayretainsTV<=.02/exacttop1gate. No dev/testevaluationinthesetrainingprofiles. Mainruntimeprojection uses load+1.25*(4860*maxsteadyprofileupdateseconds +360*maxnativeN16generationseconds +108*N16+172*N32+172*N64generationseconds)+120; N32usesN64timing ifnotmeasured. Nativeprofilepreparationwalltime included. Eachmainmustfit1800secondsbeforelaunch. Separatemainblock2.5GPUhcap9000s includes2profiles,four30minmains andboundedselected-checkpointaudits/repairs. Max4concurrentprojectGPUs. Noarchitecture/optimizerchangeafterprofileswithoutseparateamendment.

Jointfeatureprofile441831passed8calls/20rows,38GPU-seconds, projected403.60seconds/shard. ReleasefourB4jointcacheharvestshardsat8min each, fixedalready-stagedfeaturepartition, CPUsource/profilefreezeandmerge. Totaljointfeatureenvelope.65GPUh2340s INCLUDINGthe38spentprofile seconds(supersedespreparation-only.1h, notaddedtwice). Parallelcache.4h andnativeintegration.2hremainseparate; totalnewV7reservations3.75GPUh.


### V7 parallel-cache runtime extension — 2026-09-10T23:49:13.131043+00:00, before harvesting

Profile441837 passed computational and numerical replay checks but FAILED original runtimeprojection:318.6322secondsper shard>300secondcap. Preserveoriginalsummary/source/rawfailure. New explicitdispatcher/release changesonly walltimeeligibility to360seconds; originalharvest/merge computations andfeaturepartitionremainimmutable. Four6minuteGPUshards, newtotalparallelcachecap.45GPUh1620secondsincludinginitialprofile(increasesprevious.4by.05; V7total3.80h). Dispatcher/source/releaseledger accompaniesallartifacts; no fabricated passing originalprofile. CPUrelease precedeslaunch. No taskaccuracyorarchitecturechange.


### V7 review corrections before main outputs — 2026-09-10T23:50:16.881400+00:00

Clarify primarynativeexact requires both parsedgoldinteger ANDnativeEOScompletion; truncatednumericprefixes areincorrect. Retainparsed-count correctness separately. This supersedesthepreviousparse-onlyexactdefinition beforeanyV7training/modeltestoutputs; allthresholds/denominators unchanged. Leadingzerosremainparseableint; malformedanswerswrong. Reviewalsofoundexclusive-create savehelper incompatiblewithrollingtraininglogs, correctedtolocalatomicreplacement beforetrainfreeze. Runtimeprofileeligibility usescomputational_integrity_passed andzero_identity_passed separatelyfromdescriptivenumericalfailures.

Cache runtimeextension initialCPU441839 retained; beforeanyGPUharvesting, revieweridentifiedexclusive-save mismatch inplannedmerge metadataaddition. Newdispatcher decoratescachemetadata atfirstpublication, preservingappend-onlysourceartifacts. RepeatCPUrelease freezes thisimplementation-only repair, originalrelease retained.


### V7 native profile invocation repair — 2026-09-10T23:53:03.163815+00:00

CPU441841/441842passedruntimeunit/inputfreeze. GPU441843stoppedbeforemodelload after8allocatedsecondsbecause root passed relativeplanpath whereas frozenverifierrequiresabsolute descendantpath. Resubmit same immutableplan/sourcewithabsolute argument; no model/configurationchange. Failedlog retained andchargedtonativeintegrationbudget.


### V7 all-selected-checkpoint native audit — 2026-09-10T23:54:35.682217+00:00, before main outcomes

Audit ALL four dev-selectedV7checkpoints onthetwofixedoldN16K3/N64K6softwarescenes, withtheirregisteredarm/newsetprompt. CPUbindscheckpointstep/hashagainstfive-waydevselection, training/source/cache/runtime releaseidentities, exactinputs andordinarysingle-token Therefore/colon. Eachof8cases runs cachedprefill,forcedThereforethencolon withnativecached/fullprefixcomparisons:40VLM/24vision. Addoneordinarygreedymax4generationpercase:<=32VLM/8vision, totaltoplevel<=72VLM/32vision. Preserveall-rowrawlogits/prenormstates/actualmasks/positions andKVprefixretentionassertions. Actualh plusselectedbranch/norm/headreconstruction retainsTV<=.02/exacttop1gate; nativecached/full numericalerrorsdescriptivewithallfailuresretainedunderexistingengineeringrelease. Computationalintegritymandatory; no checkpointdropping,selection,thresholdsearch orretry-to-pass. Oldsoftwareaccuracy/forcedtokensarenotreasoningefficacy. Newsourcefreeze/CPUtests beforemains; selectedbindingCPUaftermains, initialpostGPU5minchargedtomain2.5hbudget.


### V7 runtime-cost measurement clarification — 2026-09-10T23:59:49.114540+00:00, before training profiles

Nativeprofile441845passedzeroidentityandcomputationalchecks,40VLM/28visioncalls,all336numericalcomparisonsalso passed;64GPU-seconds plus8failedinvocation=72integrationtotal. Allgeneratedsoftwareanswersused2tokens. Forprospectivemainruntimeprojection, usemaximuminstrumentedgenerationsecondsperarm/N fromzero/active/nativeobservations PLUS2*maximumobservedcached-generationnativeforwardsecondsforthatarm/N(fromforwards.json,visual=0/query_tokens=1/phase=generation), allowingup to4tokens. AddCPUtraincheckmeasuredprepare_scene walltimeforeachfixedarm/Ninput. N32usesN64bound. Excludefullprefixdiagnosticforwards. Thisrefinescostaccountingonly, not accuracy/protocol.


### V7 cache merge invocation correction — 2026-09-11T00:01:32.582455+00:00

Allparallelshards441846completed834allocatedGPU-seconds. CPUmerge441855receivedincorrectroot-suppliedSlurmarraytaskdirectorymapping; correctactualpublishedpaths are shard0_441847,shard1_441848,shard2_441849,shard3_441846. Preservefailedlogandresubmittheunchangedmerge/source/releasewiththoseverifiedpaths. No featuresorGPUcomputationschanged.

The next invocation441859 also stopped at an incorrectly transcribed release path. The corrected submission constructs all arguments from verified existing artifact paths and shard metadata. Both root invocation failures remain; no source/model/data change.


### V7 selected audit provenance repair — 2026-09-11T00:07:45.870007+00:00, before mains

Initialselected-auditCPUselftest441862passed2seconds. Independentreviewrequestedexplicitjointcache release binding, supplementingexistingparallel runtimeextensionbinding. Newauditcodeverifiesjointreleasehash/sidecar/fullharvestauthorization/nativeplan/parent sourcechain. Old441862snapshotretained; repeatCPUselftestfreezestheprovenance-onlyrepair beforemodelselection.


### V7 independent release scheduling — 2026-09-11T00:08:28.825155+00:00

Freeze native training/profiling sources separately fromthe independentreporter, allowingthe registeredtraining-only32-stepsoftwareprofileswhileindependentreport/dataaudit implementationfinishes. No dev/testevaluationisallowedintheseprofiles. Mainresource release still requires independentdata/reportpassedrelease and freezesreporterhashes beforemains; mainrunchecksallbindings. This changes source-ledger scheduling only; architecture,data,optimization,selection,efficacycriteria andresourcecaps unchanged.


### V7 training software CPU release — 2026-09-11T00:09:33.805806+00:00

Jointcache441854completed1101GPU-seconds; CPUmerge441867verifiedall3780features. ParallelcacheCPU441860verifiedall32229features, total869GPU-secondsincludingprofile. Nativeintegration72GPU-secondsincludingfailedrelative-pathinvocation. TrainingCPU441869passed nativefeature/target/model/sourcebindings, matchedorders/padding/lrtests andtimedoldsoftwareinputpreparation. Frozenplanattrain_check_441869/plan.json. Two32-update profilesseed8 releasedat3mineach, training-only; independentdata/reportreleaseandmeasuredruntimegate remainrequiredforfullmains. Selectedauditprovenance-repairselftest441868passed; old441862snapshotretained.


### V7 main prospective runtime extension — 2026-09-11T00:12:55.633384+00:00, before any main training

Trainingprofiles441870bothpassed57GPU-seconds total; activecached/native4comparisonsperarmalltop1sameandTV<=.02; actualcapturedh nativeheadreplaypassed. Maxsteadyupdates.0062055sparallel/.0053635sjoint. Registered conservativegeneration+preparationcost impliesoriginal30-minutemainprojectionexceeds1800seconds. Preserveits explicitCPUfailedreleaseonceindependentdata/reportgatepasses. A NEWCPUresourcerelease mayextendonlyeachmaincapto2700seconds ifalloriginal computational/data/sourcegatespassandbothprojections<=2700; no model, data, optimizer, checkpointselection, scoringorcriterionchange. Mainblockcapincreases2.5to3.25GPUh11700s, including57profileGPU-seconds andpostaudit/repairs; V7totalreserved4.55GPUh. Submitunchangedmainwrapper with explicit--time=00:45:00 override andnewreleasehash. Original1800-secondfailure remains individuallyrecorded; independentreport bindsbothresourceledgers.


### V7 independent data-report serialization repair — 2026-09-11T00:22:35.556123+00:00, before mains

Reporterunit441879passed; independentdata441880verifiedpublishedQA/images/counts/exclusions/extensions thenfailed archivedauditdictionaryequality. Freshauditreturnsinteger-keyhistograms inPython; archivedJSONhasstringkeys. NormalizeonlytherecomputedfreshsummarythroughJSONroundtripbeforeexactcomparison. No numericaltolerance, data, labels, generation, cache, model, selectionorcriterionchange. Originalfailedlog/source snapshotretained; repeatfinalreport/datafreeze.


### V7 complete main release — 2026-09-11T00:26:16.049387+00:00, before main outputs

IndependentCPU441882passed36seconds: allbalanced/freshQA/image/semantic/exclusion/extension audits andbothfulltraining-featurecaches includingcausalcount/EOSprefixbindings. Reporter69b5129b2b4743cf1f55c1fef057c725efdfdc575c119d9bd40202d267a8ad78 frozen; unit441881passed. MainCPU441883retainedoriginal1800-secondtimingfailure(projected2390.726parallel/1837.591joint). SeparateCPU441884passedregistered2700-secondresourceextension, preservingoriginalfailedreleasehash. Native/training/data/sourcegatespassed; fourmainsreleasedwithunchangedCPUplanSHAa87d40a9379c919aca72d16b83107316d961b93e2100b394621f593df75bf14c and--time=00:45:00. Allfourseeds/armsuseidenticalresource/data/reportrelease. FinalscientificacceptancerequiresbothindependentnativeefficacyreportANDall-selected-checkpoint computational/native-head replayaudit; no reasoningcompositionclaim.


### V7 final artifact and acceptance binding — 2026-09-11T00:29:06.071094+00:00, while mains run; before test outputs

Prepareafixed2x2barfigurefromverifiedreportcounts: columnsseeds8/9, familiarrowN16/32/64, unseenrowN32/64; fixedjoint/parallelcolors, completecorrect/denominatorlabels, PNG/PDF. No newfit, sampling, metric, modelselectionorconfidenceinterval. SeparateCPUfinalizerbindseachreport-selectedstep/checkpoint/parameterdigesttoallfourpost-auditmodelsandsource/artifactchains. Copyallregistered efficacybooleansunchanged. accepted_vision_milestone requiresreportauditpass ANDpostcomputational/native-head replaygatepass ANDexistingboth-seedprimary/practicalvisiongate. Cache/fullnumericalfailuresretaindescriptiveoriginalstatus. reasoning_composition_established=False. Finalizersource/unitfreezebeforetestoutputs; numerical/dataworkCPU Slurm<=5min. Thisdoesnotreplaceultimateplain/reasoningaggregationobjectivewithalocalvisionmilestone.


### V7 independent report storage-schema repair — 2026-09-11T00:47:10.518501+00:00

All four mains completed without training/evaluation changes. Independent report441893 stopped on its first development raw-logit dtype assertion: it incorrectly expected FP16 storage. Native FP16 model outputs are promoted losslessly to FP32 by installed transformers/generation/utils.py before output.logits archival. CPU441904 confirms first tensor [2,152064], FP32, finite, exact FP16→FP32 roundtrip. Preserve the failed original source/output. A new explicit repair dispatcher leaves frozen reporter and model files untouched, changes only this assertion to require FP32 plus exact FP16 roundtrip for EVERY archived vector, and retains all native greedy/EOS/NLL/data/checkpoint checks. Its source ledger includes original and dispatcher; main/report/data ancestor bindings still use the original frozen report hashes. Unit CPU first, then complete independent CPU re-audit. No predictions, thresholds, model selection or numerical tolerance change. Post-selected GPU441900 passed computational and40 native-head replay checks; six cache/full numerical failures remain descriptive under the prospective engineering release. No verified efficacy result yet.


### V7 verified outcomes — 2026-09-11T00:55:10.211194+00:00

Independent repaired CPU441910 completed28seconds and verified all1808 native test predictions plus five dev selections per run, target orders, raw full-vocabulary logits and all data/cache ancestry. Explicit FP32 archive repair leaves original frozen reporter untouched; all archived vectors are exact FP16-promoted values. Finalizer441912 completed2seconds and binds all four selected models to post441900. Computational/native-head replay verification PASSED; all6 descriptive cached/full failures retain same top1 and original statuses.

Parallel seed8: N16 108/108, N32 106/108, N64 16/108; familiar OOD122/216, nonzero110/192. Joint8:42/108,23/108,25/108; OOD48/216. Parallel seed9:108/108,90/108,0/108; OOD90/216,nonzero90/192. Joint9:37/108,29/108,29/108; OOD58/216. Primary PASS in both seeds (+34.26pp CI[28.24,39.81] and+14.81pp CI[8.33,21.30]); practical target FAIL in both. All four unseen-count evaluations0/128. Pooled+24.54pp descriptive only. Final accepted_vision_milestone=False; no reasoning-composition claim. N64 parallel often exits integer format (65/108 unparseable seed8,108/108seed9); separate unseen-count saturation at8 is exploratory error inspection.

Four mains4325GPU-seconds; selected post83; total V7=6545GPU-seconds(1.81806h), under4.55h cap. Model/data sources unchanged. Full report outputs/native_aggregation_vlm/v7/report_441910/REPORT.md; fixedfigure/acceptance finalization/final_441912/. Relative improvement is established for this new matched comparison, but reliable64-frame or unseen-count aggregation is not.


### V7 frozen response-surface diagnostic — 2026-09-11T00:56:18.137860+00:00; before diagnostic outputs

Following practical failure, inspect the two unchanged dev-selected parallel checkpoints from independently verified report441910. Fix lexicographic first two training questions and lowest pair_id for one positive and each of character-only, room-only, neither negative categories, reconstructed from training QA semantics. Using existing empty-prefix cached states and fixed corresponding global state, evaluate z(K,N)=K*m_positive+(N-K)*m_negative for K0..8,N16/64,three negative categories,two questions,two checkpoints:216 synthetic first-token points. These repeated-state multisets are deliberately outside the original Step/rendering law, not held-out accuracy. No new VLM forwards, optimization, checkpoint selection or output-policy repair. Record local positive-minus-negative separation, accumulated negatives, z/carry/normalized-carry geometry, unrestricted native-head top1/numeral margins, and genuine training-sum reference ranges. Raw-zero state padding must be an exact no-op. Native safetensors norm/head cast to observed FP16 and bound to previous capacity441720 tensor hashes; CPU-versus-archived GPU readout drift retained explicitly. At most4CPU/16G/10min via Slurm; zeroGPUcost.

If familiar-state syntheticN64 fails despite syntheticN16success, unseen Step labels are not necessary for that synthetic failure. If it succeeds, this prototype does not substantiate a fusion-length repair. Negative cancellation restoring z is an algebraic localization control, not deployable inference or independent evidence of generalization. Nonzero positive-minus-negative direction preserves K in exact-arithmetic z; wrong numeral readout does not prove intrinsic native-head capacity failure. Preserve all prototypes and outcomes; any method change needs its own prospective comparison.


### V8 fresh evaluation preparation — 2026-09-11T01:01:26.365211+00:00; before generation or V8 model outcomes

Prepare a fresh fixed evaluation for the next native aggregation comparison. This does not yet release a training intervention. Seed20260929; same452-example family design asV7:108N16 anchors(K0..8,12each) extended toN32/N64,64N32anchors(K9..16,8each) extended toN64. Exclude exactV7prior19 manifests plus bothV7freshmain/count=21; protect allV7balancedtrain/dev/schedule and prior artifacts. New data /mnt/data/gabriele/gnn_transformer/v8_fresh. V7 results motivated this research direction; freshness is complete-context/question, not new visual atoms. NewstagerSHA4bd1587c35c62aabd8fbe783167aa1e05a4b59d2d62db8487619ac08615641b1, wrapperSHA6debf3582ac79c6171f8b8cf64a0208cde354f594130a88b1ad4791e2bcc8af5. CPUdryfirstthenverifiedrender/audit,4CPU/16G/5min perjob; zeroGPUwork. Fixeddatawillnotbeselectedbasedonmodeloutcomes.


### V7 response-surface numerical contract clarification — 2026-09-11T01:08:07.080369+00:00; before any diagnostic test/output

Raw-zero local messages must equal zero bitwise. Appending zeros changes the reduction shape, so retain exact FP16 residual equality and FP32 error separately; software gate for the aggregate/readout is maxabs<=1e-6+1e-5*max(1,maxabs(originaldelta)), matching the explicit algebra-versus-expanded-multiset comparison. This is a prospective floating-point software rule, not a claim that arithmetic padding is bitwise invariant. No previously observed failure or prediction influenced it. Native CPU/GPU head replay remains descriptive with exact weight fingerprints.


### V7 frozen-state diagnostic outcome and V8 matched consistency experiment — 2026-09-11T01:11:18.492732+00:00; before any V8 GPU training

CPU response surface441927 completed20allocated seconds; all software checks passed. Across216 fixed synthetic first-token points, N16 correctness107/108 versus N64 20/108; per seed/question N16=27,27,26,27 of27 andN64=8,8,1,3. All12 positive-minus-negative message directions are nonzero; no adjacent-count FP16 carries or logits are identical. Negative messages have substantial nonzero norm; later-Step local states are unnecessary for these synthetic failures. This does not identify every actualMMReD failure or establish a deployable cancellation rule. CPU/native head reference replay top1same4/4, maximumTV1.55e-7, rawmaxlogitdifference.015625; descriptive only. Exact restoration after subtracting48negative messages holds for11/108FP16 carries; full errors retained, no bitwise cancellation claim. Unit441926 passed. Raw source states are training-only; no held-out diagnostic accuracy claim.

V8 tests ONE training objective with no deployed architecture change. Both conditions use the same frozen V7 complete-local parallel native operator and 1,041,600-parameter SUM/SiLU core, trained from matched seed10 or11 initialization. Conditions ce and consistency. Pair same-question/same-count N8 andN16 scenes using sorted SIDs: replica0/1 for each of486question/K cells, repeating the sole saturatedN8K8SID only as already weighted. Exactly972pairs/1944scene slots per epoch, preserving the V7 weighted multiset. One persistent random.Random(seed) shuffles a fresh canonical pair-slot list in each of40epochs; each pair remains N8thenN16, batch8pairs=16scenes, carry batches across epochs. Both conditions share exactly77760scene presentations/4860updates, targets, order and initialization.

Native count/EOS CE remains equally weighted across32positions. Compute delta in FP32 before its native FP16 cast. For each adjacent pair and each causal position separately, consistency=||delta8-delta16||²/(||frozen_global||²+1e-6); frozen global states must be bitwise identical within each pair. Mean acrosspairs/positions. CE condition coefficient0; consistency coefficient1. No local labels, teacher head, output mask, learned denominator, additional inference state, or architectural change. This is consistency training, not a novel attention family. Its use of equal final-answer labels is explicit; no guarantee of exact neutrality or unseen-count transfer.

Same40epochs/4860updates, AdamWlr.001,wd0,clip1; warmup50 thencosine1e-5, asV7. Same72V7N16dev scenes, native greedy max4/EOS[151645,151643] at972/1944/2916/3888/4860. Select highest exact, lowest rawfirsttokenNLL, earlieststep. Keepall5checkpoints; evaluate selectedcheckpointonly once on new452V8fresh tests, seed20260929,21manifestexclusions. Exact integer+EOS scoring unchanged; malformed/truncated remain. Native rawgenerationlogits archivedFP32 must roundtrip exactlythroughFP16. Both conditions use unmodified native runtime and onlyglobalgeneratedtoken broadcast; oneVLMcall/token andonevisualprefill. Store rawfullvocabulary, actualIDs, masks/positions/callledgers.

Primary, required in each seed: consistency−CE familiarOOD>=11/216additional correct, N16loss<=5/108. Practical additionally requires consistency>=152/216OOD, >=116/192nonzeroOOD, and now>=76/108N64 (70%). The new N64 requirement addresses the observed64-frame failure prospectively; V7 criteria/outcomes remainunchanged. All128unseenK9–16 tests separate secondary. PairedK-stratifiedfamilybootstrap10000seed20260930, same draws across conditions/lengths/fixedseeds; pooledCI descriptive, no rescue of failedperseedcriteria. A strongCE-only paired-data result remains useful but would not establish a consistency-objective benefit.

CPU source/data/loss/pair checks precede GPU. Loss11tests441924passed. V8 freshdry441922passed; render/audit441923 completed53CPU-seconds, all452contexts/280extensions/21exclusions verified, V7train/dev unchanged. Two32-update seed10 training softwareprofiles (one each condition), no dev/test outputs; finiteallgrad/frozenbackbone/save-restore and actualcapturedh→nativeheadreplay gate asV7. Fullmains need independent report/data freeze and measured runtime release. GPU partition, max4concurrentprojectGPUs. Profiles initial3min each; four mains max45min each; selected-checkpoint audit initial5min; total V8 GPU envelope3.25hours11700seconds including allprofiles/mains/audits/failures. No GPU spent yet; expected actual basedV7 around1.5hours. Allcheckpoint/data outputs remain prescribedmntroots. No reasoning efficacy claim from this experiment.


### V8 CPU release and timing reuse — 2026-09-11T01:18:28.760315+00:00

CPU441937passed16seconds and freezes trainerSHA0493db926ca2803edc42b97ad2ea0937e17172ca4082dd12478755a97b41dc00, pair/loss/tests, exact1944weightedslots/77760pairedpresentations, featurestates andfresh/native/diagnostic bindings. Two registered32-update GPUprofiles released at3min each, no dev/test predictions. Mainresource formula remains profileload+1.25*(4860*maxsteadynewupdatetime+360*T16+108*T16+344*T64)+120. T16=1.4562452072277665,T64=3.1757718743756413 from unchanged V7native integration four-token+CPUpreparation measurements, bound via oldrelease441884 and sameprofile441845/nativeoperator/runtime/GPU identity; N32usesT64. Require<=2700each and independentV8report/data frozenrelease beforemains. This is measured timing reuse, no efficacy transfer or source mutation.


### V8 all-selected-checkpoint native audit — 2026-09-11T01:21:13.399320+00:00; before main outcomes

Audit all four dev-selected CE/consistency checkpoints, seeds10/11, with unchanged native parallel runtime on the same two oldN16K3/N64K6software scenes per model. Eight ordinary greedy max4 generations plus five forced-prefix forwards percase (prefill,Therefore cached/full,colon cached/full): maximum72VLM/32vision,40nativecaptured-state norm/head replay checks. Because ALLfourmodelsnowuseparallelrows, cached/full comparisons total656 (=4*(17+65)*2),164permodel; no changed sample selection or numerical tolerance. Exact computational masks/positions/actualsharedprefix/nativecache ownership mandatory. Capturedh→samebranch/nativehead TV<=.02/exacttop1 binding; cached/full errors retained descriptively under existing unchanged-backend release. Freeze source viaCPUunit beforemains; aftermains CPU reconstructs all77760pairedpresentations/4860updates, validatescondition/loss/source/cache/release and allfive dev selections, thenfreezes selected checkpoints forGPU5min. No audit output selects/drops a checkpoint. Charge GPU to existingV8 3.25hcap; reasoningcomposition remains untested.

Trainingprofiles441939bothpassed33allocatedGPU-seconds each: allparametergradients exercised, backbonefrozen, restoreexact, fouractualh→nativehead replays percondition passed. Maxsteadyupdate.0071750sce/.0072640sconsistency. V8GPUspent66seconds; no mainoutputs yet.


### V8 fixed artifact and acceptance binding — 2026-09-11T01:22:58.918901+00:00; before main outcomes

Use the same fixed20-cell plot design asV7, now CE versusconsistency andseeds10/11:2x2figure, familiarN16/32/64 above andunseenN32/64below, unchangeddenominators, PNG/PDF. No new estimates, confidenceintervals, fitting ormodelselection. Finalizerbinds report-selected checkpointstep/fileSHA/parameterdigest to allfourpost-auditmodels,40replaysand656cache/fullcomparisons. Finalvisionacceptance requires independentreportaudit, postcomputational/native-headreplaychecks, andboth-seedregisteredprimary/practicalV8criteria. All descriptivecachefailures remain. Reasoningcompositionestablished=False. Newfinalizersource/unit beforemains; CPU2/4G5min.


### V8 complete main release — 2026-09-11T01:45:04.340005+00:00; before main training

Independent CPU reporter unit441960 passed2seconds; full data/cache/pair audit441961 passed23seconds. Reporter66debff0affd88bf232b68f3964c5e3e8116be8ff6ed0b004caac85eae4fbc16 frozen with its complete reused-source ledger. CPU resource441963 passed: projected2397.0171seconds CE and2397.4783seconds consistency, each below2700. Frozen training plan142c04be2f5ceda64a930a51a6c650663200d4b422ebcff1141a6230bff7b065. Selected-audit unit441942 and finalizer unit441944 passed; independent finalizer/schema review found no blocker. Four unchanged registered mains released on GPU partition, array throttle4,45min each. All sources/data/objective/selection/scoring frozen;66GPU-seconds spent in profiles. Final efficacy acceptance still requires independent report and all-selected-checkpoint native replay verification.


### Natural reasoning native streaming software pilot — 2026-09-11T01:47:05.644167+00:00; before implementation or outputs

Prepare NEW wrapper/source files for the previously verified CosmosReason1 Qwen2.5VL-compatible local mirror, without editing any frozen V7/V8 runtime/training/report files. Reuse parallel local/native hooks and shared-global-token broadcast. Use the existing official Cosmos natural reasoning system template and separately bound parallel image/global prompts, no forced gold answers. The current Qwen []/[gold count] training cache is ineligible for Cosmos reasoning training. This pilot only verifies software execution and supplies measured cost; no task accuracy eligibility, optimization or efficacy claim.

Use the same two fixed old software scenes, N16K3/N64K6. For each run unchanged native, exactly zero-U, and fixed seed20261001 active core with U normal std.001, native greedy max8, no score mask except shared-row global-token broadcast. Six trajectories, at most48 VLM/6 visual calls. Replay every executed active prefix from scratch, at most16 VLM/16 visual calls; total at most64/22. Exact native/zero global raw logits and IDs required; all rows share the executed suffix, one batched model call/token and one visual prefill/trajectory, exact masks/logicalmRoPE/KV ownership and hook cleanup required. Actual capturedh to same branch/native norm/head retains TV<=.02/exacttop1 replay gate. Cached/full native numerical drift remains descriptive under the existing unchanged-backend convention, without retry-to-pass. Actual dtypes measured; original unmodified model ancestry bound. Preserve any early EOS and actual cache coverage rather than claiming eight tokens always executed.

Disable generation output_logits/output_scores; a nonmutating forward hook streams only the global raw-logit row. Retain full global vectors for this bounded smoke, with a separately testable compact token/top1/lognormalizer recording mode for eventual long traces. No all-row trajectory logit retention. CPU unit/input/source freeze precedesGPU. CPU4/16G<=10min; GPU initial3min, separate total.1GPUh360seconds including any failures. At most4concurrentprojectGPUs; do not launch this GPU while all four V8 mains run. Passing establishes mechanics only, not4096-token feasibility, benefit, or reasoning composition. Any natural-prefix training/evaluation needs a separate prospective protocol.


### V8 fixed synthetic response comparison — 2026-09-11T01:52:54.476856+00:00; before V8 test outcomes are inspected

After the independent V8 report, evaluate all four unchanged dev-selected CE/consistency models on exactly the V7 diagnostic training-only prototypes, two questions, three negative categories, K0..8,N16/64:432 synthetic first-token points. Reuse the immutable V7 surface algebra/native FP16 head replay and state-selection functions; verify the V8 cache is exactly the same. Bind all four selected checkpoint identities to the V8 independent report and retain every group. No VLM calls, fit, checkpoint selection or efficacy acceptance gate. Record the same message/background/signal, aggregate, pre/postRMS, native full-vocabulary outputs, padding/algebra checks and CPU/GPU reference differences. This diagnoses whether a change survives fixed familiar local states; successful synthetic N64 is not actual MMReD generalization. CPU4/16G<=5min, new source/unit before execution, data under prescribed mntroot. V8 primary/practical definitions remain unchanged.


### V8 synthetic diagnostic completion binding — 2026-09-11T01:55:49.377827+00:00; before diagnostic execution

Independent review found no arithmetic/coverage/checkpoint defect, but requested explicit sibling report summary binding. Preserve original unit441984 source snapshot. Before any diagnostic execution, add verification that completed report summary binds exact analysis hash/path/source ledger and its archived code copies. Pending diagnostic441988 cancelled before execution; repeat source unit and resubmit. No model/report/data/prediction/metric or432-point protocol change.


### Cosmos streaming software CPU release — 2026-09-11T01:58:49.244585+00:00

New recorder/native wrapper f576bbf5f9c716d56f025ca80e934f343884527945bd47dc88aa7aa7591e7376 andprofiler e5ef5c5265be32aaaba3ea2b5e726d32d841f0cd14876b2b2c9d16e0b5b5e1af independently reviewed with no blocker. CPU441990 passed7unit tests,11allocatedseconds; CPU441991 passed fullprompt/processor/mRoPE/model-mirror/input/source freeze,16allocatedseconds. Planecb3546fd6b1f1d5b3c8c1d003e0e8d5366ba81f74d3dbd0a277e777ae865be0. Registered3minute GPU smoke may start only after allfour V8 mains terminate successfully, preserving4GPUprojectcap. No V8 efficacy or reasoning benefit is assumed.


### V8 verified outcome and Cosmos software completion — 2026-09-11T02:13:07.472243+00:00

Independent441974 passed86CPU-seconds; all1808 native test and1440dev answers verified. CE10:108/79/0 atN16/32/64, OOD79/216; CE11:108/55/12,OOD67/216. Consistency10:108/108/72,OOD180/216,nonzero156/192; consistency11:108/108/53,OOD161/216,nonzero140/192. OOD gains+46.76ppCI[43.98,50.00] and+43.52ppCI[39.35,48.15]; N16unchanged. PrimaryPASSboth; practicalFAILboth, specificallyN64<76/108. AllfourunseenK9–16=0/128. This isolates the objective effect for the same native inference computation.

SelectedCPU441976 passed81seconds; GPU441997 passed80seconds,57model/32vision,40native-head replays,656cache/fullcomparisons. Five descriptive numerical failures remain: the known localN64row62 TV.02029646 oncepermodel, plusconsistency11globalN64forcedstep2 TV.02749113; alltop1same. Finalizer441998 passed1CPU-second, boundallfour selected checkpoint identities; accepted_vision_milestone=False, reasoningcomposition=False. Fourmains1130seconds each=4520; totalV8=66profiles+4520mains+80post=4666GPU-seconds(1.29611hours).

Fixedstatecomparison441992 passed16CPU-seconds: everymodel54/54syntheticN16; syntheticN64 CE10=6/54,CE11=11/54,consistency10=32/54,consistency11=30/54. Same training-only prototypes/head,432firsttokenpoints, noVLMforward/fit. Remaining syntheticfailure means laterStepnovelty is not necessary for that failure. No original efficacygate changed.

Cosmosstreaming441994 passed121allocatedGPU-seconds underseparate360cap:48natural tokens,16activeprefixfullreplays,64model/22visualcalls,32same-stateheadreplays. Allnative-zero identity/computational anddescriptivecache/fullchecks passed; actualnativeFP16norm/head. Software-only, noaccuracy/training/longcache/composition claim. ExistingQwencount-prefixcachesremainineligibleforCosmosreasoningtraining.


### V9 fresh evaluation preparation — 2026-09-11T02:15:38.267704+00:00; before data generation

Prepare the next fixed452-context native vision evaluation with the identical V8 family law, seed20261002. New root /mnt/data/gabriele/gnn_transformer/v9_fresh. Exclusions are exactly all21V8 prior manifests plus V8freshmain/count=23; bind the previous21hash inventory, protected V7balanced train/dev/schedule and all prior artifacts unchanged. Freshness means complete-context/question, not new visual atoms. New stager/wrapper only; CPU4/16G<=5min dry first, then frozen reviewed rendering/audit. This releases data preparation only, not a training intervention. NoV9GPU yet.


### V9 directional sensitivity objective — 2026-09-11T02:25:00.214881+00:00; before V9 GPU training

V8's relative improvement is verified, but both selected models still fail the prospective N64 practical target and the fixed familiar-state diagnostic. Test ONE new training constraint while preserving the entire deployed parallel SUM/SiLU operator, native FP16 carry/norm/full-vocabulary head,1,041,600FP32parameters and one batched VLM call/output token. Conditions residual and path, fresh seeds12/13, matching initial parameter bytes and scene order within seed. No affine decoder ablation: it would restrict recurring native digit outputs along an affine count ray. No claim of a new attention operator, path norm, or Lipschitz theorem.

For each adjacent same-question/count N8/N16 pair, separately at count/EOS positions, z is the unchanged learned SUM statistic and d=Wagg(z16-z8), excluding the shared query and aggregate bias algebraically. With output columns u_j, define B=sum_j ||u_j||_2*abs(d_j). Lpath=mean B^2/(||frozen_global||^2+1e-6). Lresidual is the exact V8 mean squared endpoint residual difference with the same denominator. Native CE uses the original branch.forward, unchanged in both conditions. Both compute/log both regularizers and all position components; objective is CE+Lresidual for residual, CE+Lpath for path. Coefficient1 fixed prospectively in both; equal coefficients do not match effective regularization strength. No coefficient search or profile-based tuning.

The mathematical implication is ||U[SiLU(r+t*d)-SiLU(r)]|| <=1.1*abs(t)*B for any base r and real t; squared normalized change <=1.21*t^2*Lpath per pair. This controls observed latent directions and removes output-channel/activation endpoint cancellation; cancellation inside Wagg remains. Reciprocal per-channel scale changes leave B unchanged but need not preserve the SiLU network function. Future directions, Step shifts, nonlinear native normalization, rounding, vocabulary margins and unseen numerals are not certified. A small mean loss is not a worst-case guarantee. Potential overconstraint is part of the experiment.

Full V7 balanced1890training scenes and972pairedslots/1944scene slots per epoch reused unchanged;40epochs, persistent random.Random(seed), N8thenN16 inside each pair, batch8pairs, carry across epochs:77760scenes/38880pairs/4860updates. New seed-aware helper only, original V8 pairing source immutable. Same AdamWlr.001,wd0,clip1,warmup50 and cosine to1e-5. Same72V7N16dev at steps972/1944/2916/3888/4860. Select exact then raw first-token NLL then earliest; retain all five checkpoints. Evaluate selected once on452V9fresh tests,seed20261002,23exclusions. Native greedy max4, EOS[151645,151643], strict full ASCII integer+EOS, no repetition penalty/output mask. Keep malformed/truncated and all128K9–16 examples as separate whole-answer secondary.

Primary in EACH seed: path-minus-residual familiarOOD>=11/216correct and N16loss<=5/108. Practical additionally requires path>=152/216OOD,>=116/192nonzeroOOD and>=76/108N64. Both-seed failure is not rescued by pooling or historical criteria. Paired K-stratified complete-family10000bootstrap,seed20261004; same draws across lengths/conditions/fixedseeds, pooled intervals conditional on fitted seeds. Report per-K, parse/EOS, both loss scales and clipping. Logged comparison Lresidual versus1.21Lpath and native residual magnitudes are descriptive numerical diagnostics, not an added accuracy/certification gate.

Loss prototype unit442013 passed10new+11oldtests in4CPU-seconds; preserve its snapshot. Before training, independent review requested computing d directly from projected SUM differences instead of subtracting endpoint preactivations with large shared offsets. Final unit442018 passed12new+11oldtests in5CPU-seconds, including zero-U/zero-column finite gradients, bound along extrapolation rays, endpoint-cancellation counterexample, direct projection, inverse-scale penalty invariance and actual unchanged core reconstruction. torch.linalg.vector_norm uses its tested zero subgradient; no epsilon added to output-column norms. No V9 fitting occurred.

CPU source/cache/pair/input checks first; two32-update seed12 profiles3min each, no dev/test predictions, allparametergradients/frozenbackbone/exactrestore and captured-state native-head replay required. Independent report/data/source release and measured2700second main projections before four45min mains. Reuse unchanged V7 native inference timing/hardware bound and new update/load times. Selected-checkpoint CPU binding then GPU<=5min on all four models;40headreplays/656cached-fullcomparisons, same old software scenes/prefixes and preserved numerical statuses. Separate CPU finalizer binds all four models and unchanged efficacy decisions. Total V9 envelope3.25GPUhours11700seconds including profiles/mains/audits/failures; GPU partition,max4concurrentprojectGPUs. Expected actual around1.4hours basedV8. Alldata/checkpoints stay in prescribedmntroots. Reasoning composition and answer-value holdout are not released by this experiment.


### V9 software and data release outcomes —2026-09-11, before main efficacy

Fresh stage442020 passed452contexts/23exclusions in62CPU-seconds; mainSHA6100894312e384e5bac6966c84d9d7c5ff84e72d06ba416a8697d0b2dbc70f35, countSHA4d4ebedf497f031ebeff1ca6e3242bbac815566c714904c5e840e7fcabf4de6a. TrainingCPU442032 passed19s, planSHA cda9479abc0112a44ccf37b0f951ad7729c61c960901bad39e810eac6b5c6f8a. Independent reporter tests442033/data442034 passed3/28s. Profilearray442035 passed both32-update software cases,67GPU-seconds total; zero-U bothauxiliarylosses/gradients zero, allparametergradients exercised, frozenbackbone/restore/nativeheadreplay passed. No dev/test evaluated. Resource442039 projects2410.69s residual and2413.75s path, both≤2700s. Finalizer selftest442037 passed. Reserved3.25GPUh/max4concurrent remains unchanged. Sources, testthresholds and native inference remain frozen; no V9 main outcome yet.

V9 main array442042 submitted after data/profile/release checks and selected-audit prospective selftest442041 passed. Four tasks residual/path seeds12/13, four concurrent B200s, each45min cap. No configuration or criterion changed. Profiles used67GPU-seconds; total blockcap remains11700.

### V8 unseen-output description —2026-09-11

Read only the existing verified V8 outputs to separate K9 (single-token answer, unseen label) from K10–16 (unseen values and continuations). Retain all512 recorded sequences, all four selected models and both N32/N64 cells. Count first generated token correctness, whole strict answer plus EOS, correct first token immediately followed by EOS, wrong first token, malformed/truncated outputs and parsed/completed numeral histograms. First-token comparison never scans for a later numeral. These overlapping descriptive counts do not identify whether a latent count is present. No new predictions, model calls, prefixes, fitting, selection or acceptance criterion. Bind report441974 and selected checkpoint/output hashes. Source audit_native_vision_v8_unseen_traces.py SHA1bbdfcc44c1756c9217bf114553aa36fec242958019a523c84cf77fb4891adaa, CPU-only Slurm up to5min.

V8 unseen-output audit442059 completed12CPU-seconds and verified all512 stored natural generations. Every model has0/128correct first tokens. Both consistency10/11 answer8 on all128unseen-count contexts each, with all outputs parseable/completed. Thus actual failures begin at the first numeral token; immediate EOS after a correct first token explains0cases. The output saturation does not prove absent latent count information. No new predictions or intervention. Canonical v8/unseen_trace_diagnostic/run_442059.

### V10 data and general sequence software preparation —2026-09-11, V9 efficacy pending

This releases only CPU data preparation and software tests, not new GPU training. The question is length extrapolation when every tested native answer sequence is represented in adaptation, retaining maximum training length16 and testlength64 (4×). It does not test unseen answer values or free reasoning.

Use54questions and uniform K0..16 weighted support. For each question/K0..8, pair one N8 and one N16 context. For K9..15, pair two distinct N16 contexts. For K16, one saturated N16 context appears twice as the same SID pair; its regularizer is exactly zero and remains in the denominator. This yields918pair slots,1836weighted scene slots,1782unique training contexts. The finite-world saturation exceptions permit historical training-content reuse ONLY for N8/K8 and N16/K16. The former has one slot per question, the latter two slots of one SID. Every reuse must be recorded; no claim those contexts are novel. All other training, all development and all test contexts must exclude25prior manifests (the exact V9 exclusion inventory plus both V9 fresh manifests) and other current selected contexts.

Training drawseed20261005; developmentseed20261006. Development has64fresh N16 contexts,4for each K0..15; no falsely fresh saturated K16 case. Fresh testseed20261007 has136families (8per K0..16), each N32 parent extended to N64 using the inherited random insertion/nuisance law,272contexts total. Train/data roots /mnt/data/gabriele/gnn_transformer/v10_balanced and v10_fresh; new sources only. Dry generation, frozen plan/inventory, independent semantic/hash review precede image rendering. CPU Slurm only, at most4CPUs and15minutes perstage.

Required interpretation: training length and count are correlated; only K0..8 has cross-length pair supervision. K9..15 has same-length scene consistency and K16 no nonzero consistency term. Report K0..8, K9..15 and K16 separately at each test length, alongside the uniform-K total. No N16 full-support test is feasible after all saturated K16 contexts enter training.

The sequence software prototype must represent targets as arbitrary ordinary native token sequences plus EOS, harvesting only strict prefixes y[:t]. The same [1] prefix serves K1 and K10..16; cache keys cannot encode gold, total target length or N. Pack valid causal positions with explicit scene/pair offsets. CE is the mean of per-scene token means; regularizers are the mean of per-pair corresponding-prefix means. Padded output positions contribute nothing. Test exact two-position compatibility, ragged-prefix alignment and sequence weighting under CPU Slurm before any new training release. Existing V7–V9 sources and outcomes remain frozen.

V10 dry442083 failed after2CPU-seconds because the inherited file digest helper accepts Path objects, while bound plan filenames are strings. No images or data roots were created. Preserve its source/sample/plan artifacts and failed log. The new stager alone normalizes digest arguments through Path; seeds, generated semantics, pairing and exclusion rules are unchanged. New sourceSHA 6a1fa9a1e4e16fea2d42615ea5733f6348c47f4d79764c6f92f76ec1a504f9e7. Rerun a fresh CPUdry freeze before rendering.

### V9 fixed-state diagnostic —2026-09-11, after the failed main comparison

Reuse the exact V7 training-state prototypes, native head, grid and surface evaluator already applied to V8. Evaluate all four V9 selected cores (residual/path, seeds12/13), bound to completed independent report442051 and its companion summary. The fixed432points comprise two training questions, three negative categories, N16/N64 and K0..8. No new model calls, predictions on new images, fitting, selection, thresholds or prototype choices. This synthetic diagnostic tests whether the V9 degradation also appears on familiar local states; it is not held-out task accuracy and cannot rescue the failed V9 comparison. CPU self-test precedes full execution. SourceSHA d83e17136d94721be588e1add905bf3dfc146bc8137099905ac8e74080e87bbe.

V9 verified complete: report442051, selectedCPU442052, selectedGPU442081 andfinalizer442084 passed verification. Residual12/13 N64=47/108,50/108; path12/13=18/108each. AllN16/N32=108/108; allunseen0/128. OOD differences−29/216 and−32/216, both primary/practicalfailed. Forty nativehead replays passed; five descriptivecachefailures retained, alltop1equal. Total4940GPU-seconds=1.372222h (67profiles+4788mains+85postaudit). No acceptedvisionmilestone, no reasoningcomposition. Pathobjective dropped as improvement; no coefficient tuning or retrospective threshold change. TheV10data-only preparation remains asregistered.

### V10 native supported-answer length experiment —2026-09-11, before features/training

After V9's independently verified negative result, retain ordinary native residual consistency as the sole treatment. The comparison is CE versus CE+residual consistency, coefficient0 versus1, both using the same unchanged native parallel SUM/SiLU core and the already registered expanded-answer data. Do not tune or retain the path objective as treatment. Both objectives may compute its training-only diagnostic via the generic sequence helper; it has no loss weight or inference role.

Conditions ce/consistency; new seeds14/15. Same initialization within seed, identical persistent random.Random(seed) permutations of a fresh canonical918-pair list each epoch, adjacent pair members and intact8-pair/batch groups carried across epoch boundaries. Forty epochs give36,720pair presentations,73,440scene presentations,177,120valid causal target positions,4,590updates. Native answer sequences include every ordinary numeral token and exactly one tokenizer EOS. CE averages tokens within each scene, then scenes; residual consistency averages corresponding prefixes within each pair, then pairs. Retain all54saturated identity pairs with exactly zero regularizer and their intended CE weights. Metadata independently binds questions, actual strict prefixes, sequence lengths, feature identities and target tokens. No padded output position, full-answer feature at an earlier prefix, or gold/target-length/N-dependent empty-prefix lookup is permitted.

Core remains1,041,600FP32parameters, rank96,H3584; frozen/eval Qwen2.5VL7B NF4 with BF16compute, actual native residual/norm/head FP16. One batched N+1-stream VLM invocation per output token, one visual prefill, ordinary independent KV and native positions. No local labels, auxiliary binary teacher, external tally, logit mask, forced numeral, repetition penalty, or added inference operation. AdamWlr0.001,wd0,clip1,warmup50,cosine to0.00001 atstep4590. Evaluate the fixed64N16 K0..15 development scenes at918/1836/2754/3672/4590updates. Select maximum whole-answer exact, then minimum raw first-token NLL, then earliest step. First-token NLL is explicitly only a tie-break and cannot distinguish the shared first digit of10–16. Keep all five checkpoints. K16 has no purported fresh N16 dev case.

Evaluate all272fresh contexts only after selection, natural greedy up to4new tokens, native EOS; exact requires the complete stripped ASCII integer equal to gold and completion within the budget. All malformed/truncated examples remain in the denominator. Counts K0..16 are supported during adaptation; no unseen-count extrapolation claim follows.

Primary EACH seed: consistency-minus-CE N64 at least7/136additional correct (at least5percentage points), and N32 loss at most6/136 (no more than5percentage points). Practical success additionally requires consistency N32≥123/136, N64≥109/136 and N64 K9..16≥52/64. Both seeds must pass every required condition; pooling cannot rescue a failure. Report all17K cells, nonzero counts, K0..8, K9..15 and saturatedK16 separately at N32/N64. Report strict whole answer and native EOS/parse/truncation plus first-token diagnostics. Family bootstrap uses10,000 K-stratified draws with seed20261008, intact N32→N64 families and identical family draws across conditions and fixed fitted seeds. Intervals condition on these seeds, not a training-seed population.

Resource release: maximum4concurrent projectGPUs; total V10 cap4.5GPU-hours (16,200allocatedGPU-seconds), including failed/profile/cache/training/post jobs. CPU source/data/inventory/tokenization/sequence checks precede GPUs. Cache inventory is derived solely from1782training scenes, with local key(imageSHA,question,strict prefixIDs) and global key(question,strict prefixIDs), independently verifying processor layout/masks/native positions. Freeze four deterministic disjoint shards. A single ≤3min GPUcache profile uses the inherited four phase groups (local/global × empty/nonempty), small and repeated64-row stress batches,8model/4visionforwards total. It must include one- and two-token prefixes, native head replays, frozen model/source identity and finite features. Small-versus-stress numerical differences are descriptive and retained. Project each shard using measured per-phase64-row cost, model/pixel load,1.5safetyfactor and30s allowance; all must fit600s. Then four cache jobs each≤10min, CPU merge with complete feature/tensor/source checks. No cache prediction/accuracy gate.

Two seed14 training profiles,32updates each,≤3min each, no dev/test, exercise gradients, zero-initial residual loss, frozen backbone, checkpoint restoration and native head replay. CPU/short native software checks must cover all strict prefixes of0,9,10,16, including shared[1] and complete two-digit prefixes; original old scene software tests may be reused with fixed identities. Main release requires independent report/data/source freeze and measured projected per-main cost≤2700s, with the unchanged native four-token timing bound and new update timing. Four mains each≤45min. The CPU selected-checkpoint audit binds all four selected models and every source, followed by a≤5min GPU audit preserving all40native-head replays/656cache comparisons on the same two fixed old software cases; all original descriptive numerical failures remain recorded. Final verification must bind independent scoring and allselectedcheckpoint identities. No reasoning efficacy or long-generation run is released by this block.

V9 fixed-state diagnostic442099 completed24CPU-seconds: allfourmodels54/54syntheticN16; N64path12/13=32/54,24/54 versusresidual12/13=28/54,29/54. No consistent synthetic benefit and no rescue of the failed main experiment. All432points retained, zeroVLM calls/fit.

V10 featureCPU442121 passed in36allocatedseconds, all1782training scenes/4428weighted causal positions and33658unique native features; planSHA03c92ba5e563a2b5b22dbcaa41e42db6fe5714a803fe88a8ef94fc12b3257093. The four frozen shard inventories are fully recorded in the plan. GPUcacheprofile442123 launched atregistered3minute cap; no new training outcome.

V10 featureCPU442121 scheduler accounting correction:45allocatedCPU-seconds,35.73measured script seconds (the previous36figure rounded script time). CacheGPUprofile442123 passed allsoftware/nativehead/prefix/timing gates in44allocatedGPU-seconds; allfourshardprojections340.50s≤600. Cachearray442125 launched fourregistered10minute jobs. No efficacy predictions or fitting.

V10 independent manifest audit442132 failed18CPU-seconds at exact record-list equality: JSON sorted split mapping keys while original stage used train_N8,train_N16,dev_N16 insertion order. Earlier semantic/image checks passed. Preservefailedsnapshot/log. Reporter alone normalizes the private audit copy to the registered split order, retaining exact equality for every reconstructed record and all other fields. No data/source-generation/metric/criterion change. RevisedreporterSHA7e2409674980fc73074e657c903bec0b1e11051e9d25f6704740e968d8c45a13; selftest442138 andmanifestcheck442139 submittedbeforetrainingrelease. Rootfirstmerge-submission helper usedincorrect directoryglob and stoppedbeforeany merge submission; no artifactsmodified.

V10 correctedindependentmanifest442139 passed53CPU-seconds and selftest442138passed2s, exactrecord/field equality retained. Featurecachemerge442141 passed13CPU-seconds with33658states,cacheSHA42dd0c3c12bc13326493dce9389656b1c412e9ecfa6dac37ccb45458aa4ede11. Fourcacheallocations442126/442127/442128/442125 used232/222/230/221GPU-seconds; totalV10GPUspent949includingprofile44. TrainingCPU/sourcefreeze andindependentfullcacheaudit next.

V10 trainingCPU442145 passed17CPU-seconds,planSHAd8e1aebaefa97b3b61d3e03d371666bd4c74034293d377d2803adb4ca18d32b9; independentfullcache/data442146 passed23s. Selectedauditselftest442134 andfinalizerselftest442144 passed. Two32update trainingprofiles442149/442148 completed42/40GPU-seconds, allparametergradients/frozenbackbone/zero-Ulossandgrad/restore/tenstrictprefixnativeheadreplayspassed. No dev/test predictions. TotalV10spent1031GPU-seconds; measured main release next under unchanged16200cap.

V10 measuredmainrelease442158 passed: sharedslower updatebound0.01796548s, projected1900.49/1900.68s percondition≤2700. Nativefourtoken+preparation timingunchanged; actualprior1031+four2700mains+300postaudit=12131GPU-seconds≤16200. Independentreportercode/data andtraining sourcesfrozen. Fourregistered CE/consistency seeds14/15mains mayexecute inparallel; nooutcomesyet.

### V10 verified outcome —2026-09-11

Independentreport442173 passed35CPU-seconds; selectedCPU442174 passed36s. CE14 N32/N64=21/136,8/136; CE15=26/136,0/136. Consistency14=43/136,20/136; consistency15=52/136,19/136. N64gains+8.82ppCI[5.88,11.03] and+13.97ppCI[10.29,17.65], bothprimarypass. Bothpracticalfail allthreeabsolute requirements; allfourmodelsK9–16zeroatbothN32/N64. AllselectedN16dev64/64. BothconsistencyN64 outputs136/136parseable/completed; countfailureisnotjustformat. All17answer valueswereprovided duringtraining; crosscampaignV8/V10differencesalsochange data/pairingdistribution anddo notcausallyisolatetheeffectofanswercoverage.

SelectedGPU442185 passed84GPU-seconds,56model/32vision,40nativeheadreplays,656cachecomparisons; fourknownlocalN64TV.02029646descriptivefailuresretainedalltop1same. Finalizer442190 passed2CPU-seconds,figurevisuallychecked; accepted_vision_milestoneFalse/reasoningcompositionFalse. Mains970+912+958+912=3752GPU-seconds. FullV10=949cache+82trainprofiles+3752mains+84post=4867GPU-seconds(1.351944h),below16200. No retrospectivecriterion/source/selectionchange.

### V11 persistent native write: software preparation —2026-09-11, before prototype execution

V10's fully supported-answer test remains poor at longer lengths despite perfectselectedN16development. This does not identify its cause. Test a structural capability missing from the final-readout adapter: allowing an aggregate to enter the model's own future attention memory and receive later-token gradient credit at fixed teacher-forced inputs.

Both softwareconditions use the same existing1,041,600FP32parameter rank96SUM/SiLU core and common frozenlocal/globalhiddenstates AFTER the penultimate decoderblock (D−2 zero-based; Qwen28blocks:index26). pre_last addsδ to theglobalstate there, BEFORE finalblockindex27; post_last computes the sameδ there andholds it until beforefinalRMS/head. Preserve native addition h+castFP16(δ). No middle-layer search or changedmessagefunction. NativeevalisoneN+1streammodelinvocation/outputtoken andonevisualprefill. The globalrowremains text-onlybeforeaggregation; bothconditions have the same evidence path.

pre_last changes final-layerglobalK/V at written querypositions; prior27layers andalllocalrows remainunchanged forfixedprefixes. post_last never changesbackboneK/V. Thisplacementcontrast also changes how thecurrentδ passes throughonefrozennonlinearblock; a lateraccuracydifferencealone cannotidentify a persistencecause. A future selectivepast-K/V restoration experiment would be needed for thatattribution and is not yetreleased.

Prototype trainingreplay reconstructsthe WHOLEglobalprompt plus target[:-1] fromnewfrozenpenultimatestates, writesatP−1+t for everyvalid assistantquery, andrunsthe actual finalfrozenblock withnativecausalmask/rotary, use_cache=False andpast=None. Normalization/head remainnative. NoEOS target orfuturetoken mayenterthe input. Directpairsgivenidenticalquestion/prefix retainidenticalfrozenglobalreads; oldresidualconsistencysemanticscouldapply but noefficacyfitisreleasedhere. Existingfinalstatecachescannotserveaspenultimatefeatures. Last-blockbackwardrequiresnewtiming/memoryprofiles; oldhead-onlytimingisineligible.

CPU-onlySlurmchecksfirst, upto4CPUs/16G/5min percheck: source/model/runtime/processor/layoutfreeze andboundedtests of explicitquerypositions, hooks, causality, gradients andcleanup. Fixed GPUsoftwarecases use theunchanged oldN16K3/N64K6nativeprofileexamples plus a deterministic V10trainingN16K10example for threepositions. NoheldoutV10testselection. ZeroU mustpreservenativeoutputs; deterministicnonzeroU testsbothsites. Cacheddecodewritesonlynewqueries; full-prefixreferenceMUSTrepeatallhistoricalassistantquerywrites. Require nativecaptured-statehead/last-blockreplay, finitetensors, modelweightgradientstateunchanged, lower/localKVunchanged, andonlyearlywrite'sfinal-layerglobalKVmodified. Keep rawdifferences andexplicitcounters.

Causalcreditcheck uses independent liveδ leaves atfixedK10prefixpositions and CE at thesecondnumeral(token0afterprefix1), avoiding a saturatedEOSobjective. Itsgradienttoearlierδ0 must be nonzeroonlyforpre_last, andgradienttofutureδ2 mustbezero inboth. Futurewritesmustnotchangeearlierlogits. Capture andverifycasting order. Samebatch/samehidden lastblockreplayis a required computationalcheck; global-onlyreplaybatch-route differencesare separatelydescriptive, never assumedexact underNF4.

OneGPUsoftwareprofile upto5minutes afterCPUreview; totalV11softwareenvelope0.15GPUh(540allocatedseconds), includingfailedattempts, maximum4concurrentprojectGPUs. Freezeanexplicitcallinventorybeforelaunch. Thisreleases nofeaturecorpusharvest, modeltraining, newtestdata orreasoning efficacy. Data/models remain in theprescribed mntroots; allV10artifactsandfailuresremainimmutable.

V11 fixedsoftwarecallinventorybeforeexecution: oldN16/N64cases eachuse barecached3calls,zero/pre_lastcached3,zero/post_lastcached3,activepre_lastcached3plusfull-prefixreplaysatt1/t2,activepost_lastcached3plusfull-prefixreplaysatt1/t2=19VLM/9visionpercase. Inheritedtwo-tokenforcedsoftwareprefix(Therefore,colon) is unchanged andnotanaccuracytask. Addonefullprefix V10trainN16K10capture foractualnumeral1,0 prefix: total39fullVLMcalls/19visualforwards. Atmost16additionalfinal-block-only replays:12samebatchactivecontextcomparisons,2global-onlytrainingbatchdescriptivecomparisons,and2three-positiondifferentiablecreditchecks. Countthese separately. Allnative-zero identity,lower/localKVpreservation,write/historypositions,finite/frozenweightstate andsame-statehead/samebatchlastblockreplaysarebinding. Cached-versus-full numericalTV/top1differencesremainexplicitlydescriptiveunderunchangedNF4nativebackend; structuralhistorychecksremainbinding. No hiddenmodelpasses orpredictionselection.

V11 independent CPU selftest442222 passed21 tests (controller11, actual tiny-Qwen final-block10); no GPU use. Source snapshots preserve native-zero identity, full/cached historical writes, lower/local preservation, live temporal credit only before final block, future causality, casting and cleanup checks. Native full-model profile still awaits its complete source/input freeze.

### V10 training-estimated background direction diagnostic — 2026-09-11, before execution

Separate from V11 memory software, test whether a first-moment irrelevant-evidence drift is correctable in the four already selected V10 models. This is a fixed diagnostic on existing test contexts, not fresh confirmation, a new training condition or a deployed method. At fixed empty answer prefix, estimate each core's query-dependent mean local message mu0(q) using all image occurrences in its V10 K0 training scenes (N8 and N16 occurrence-weighted), with the shared frozen training global query. No development/test labels, hidden states or predictions enter this calibration.

Select one complete N32-to-N64 family per K0..16 by lexicographically lowest family ID in the existing V10 fresh manifest, both lengths: 34 scenes total, all four selected CE/consistency seeds14/15. Bind exact IDs before prediction. At the decoder preactivation r=Wagg*z+query+bias, compare unchanged r, r-(N-16)*Wagg*mu0(q), and r-(N-16)*s(q), where s is a deterministic direction orthogonal to Wagg*mu0 with equal L2 norm. Seed20261101 and a fixed model/question ordering define the sham; when the estimated direction is exactly zero both interventions are zero. No scalar search, direction selection, checkpoint selection or outcome filtering. The intervention is exactly zero at N16 by definition, which does not constitute an additional accuracy result. Keep the nonlinear SiLU decoder, original native norm/head and h+castFP16(delta).

One shared branch-free native N+1 forward per scene captures first-query local/global states. Four cores and three interventions reuse those same states. Native norm/head replay keeps the full N+1 by1 hidden shape; one additional bare captured-state head replay per scene verifies the readout binding. Fixed GPU inventory34fullVLM/34vision calls,408core/intervention head replays plus34bare head replays=442 additional head calls, no lastblock-only calls or generated continuations. Preserve full raw states/logits, calibration tensors, first-token IDs/NLL, per-K/length/model results and all source/model/selected-checkpoint identities. Comparison with the original stored first-token vectors is descriptive under native numerical variations; captured-state bare-head replay is binding at TV<=.02 and exact top1. All finite/output/state checks are binding.

A direction-specific rescue over the norm-matched sham would support this limited correctable-background account. Failure rejects that simple intervention, not the presence of count information. First-token accuracy cannot establish integer-plus-EOS performance, especially because10–16 share first digit1. Report those groups explicitly; no change to V10 acceptance. CPU source/calibration checks first, up to4CPUs/16G/5min. Then one GPU diagnostic at most3min, separate total cap0.1GPUh(360allocatedseconds), including failures; max4concurrentprojectGPUs. No parameter fitting or benchmark expansion. Data and checkpoint artifacts remain under the prescribed roots.

V10 background diagnostic schema resolution before execution: the manifest stores the complete-family identifier as pair_id (equal to anchor_id), not a field named family_id. Select the lexicographically lowest pair_id per K and require exactly one N32 and one N64 member. The442 head count refers to additional replay calls; the34 full native forwards also each invoke their own head. No input-selection or budget change.

V11 software refinement before CPU source freeze: use the same16 last-block-only calls to check future-write value causality as well as gradients. Each global-only reference uses zero independent FP32 delta leaves with autograd enabled. Its paired credit replay keeps delta0/delta1 zero and assigns delta2=0.1*linspace(-1,1,H), then differentiates CE at t1. Require earlier t0/t1 logits to match the zero reference exactly and the t2 logits to change, with future-row CE gradient zero. Both replays use the same gradient-enabled backend route; compare their full global-only zero-reference logits with native full-batch logits only descriptively. The maximum39VLM/19vision/16last-block inventory and540GPU-second software cap are unchanged. No fitting or efficacy selection.

V11 CPU source/input freeze442238 passed17allocatedCPU-seconds, planSHAddfbc6c23174d0a415a9f22b662878d898bfbb656366f31831c86c332125c430. All21 tests and exact model/runtime/processor/oldsoftware/K10-input binding passed. Native GPU software profile may now execute under the unchanged300-second job/540-second total envelope and39/19/16 call inventory; no training released.

V11 native GPU attempt442246 failed31allocatedGPU-seconds after two native forwards: the exclusive-create JSON helper was called twice for the same partial-progress filename. Preserve all original inputs, source snapshots, raw captures, logs and counters. Repair only the profiler artifact writer: numbered per-forward progress files and distinct per-case checkpoints, canonical final outputs written once in finally. No model computation, input, criterion or call-budget change. RevisedprofilerSHAa849eaab19363bd373d6ec24b604aa51a3ebab23b388e87747a646ac25f051f5; repeatCPUfreeze before retry. Total31/540GPU-seconds spent; one300-second retry fits the existing envelope.

V10 background diagnostic CPU442244 passed76allocatedCPU-seconds: all54 questions/fourcores calibrated from1296 occurrence-weighted K0 image slots,34 fixed native input bundles and complete source/selected-model bindings. PlanSHAb3cadb60190546f2aa64ed704dc6fd7ee7dc5550cb1ffdb75536d29d0e45e6e2. The registered34-forward GPU diagnostic is released under its180-second job/360-second total cap. No accuracy result yet.

V10 background submission442251 failed argument parsing before model load or a forward (0allocatedGPU-seconds): root passed a positional plan to a wrapper that requires --plan. Preserve scheduler log; retry the unchanged source and exact CPU plan with --plan. No resource/selection/computation change.

V11 corrected progress-writer CPU442253 passed19allocatedseconds, planSHA2e0e1b05d025662ea16ecf3d82067f9342d7fefe137498824830d96ec852450d. Repeat the exact registered GPU software protocol, preserving failed442246 and31GPU-seconds spent.

### V11 common-read native write-placement experiment — 2026-09-11, before efficacy fitting

Prospective specification for one matched comparison; main GPU training still requires a measured release. Both conditions read the same frozen local/global states after decoder block26 and compute the unchanged rank96,1,041,600-parameter SUM/SiLU core. pre_last writes before block27; post_last writes before final norm using the same common read. Native h+castFP16(delta), NF4 backbone, ordinary head and oneN+1-stream model call per output token remain. Both optimize full native answer CE plus residual consistency with coefficient1 and the unchanged squared common-global-norm+1e-6 denominator. Path loss is diagnostic only. Same initial parameter bytes and presentation order within seeds16/17; initial gradients may differ because the write Jacobians differ. This comparison includes current-write processing through a pretrained block and persistent attention memory; an accuracy difference alone will not identify memory causation.

Reuse exact V10 train/dev/pairs/schedule:1782 unique train scenes,918pairs/1836weighted scene slots, K0..16 support, maximum trainN16. Keep K0..8 cross-length, K9..15 same-length and K16 same-SID identity pairs with zero penalty in the denominator. Same40epochs, persistent seeded pair shuffles, carried8-pair batches,4590updates/73440scene presentations/177120valid target positions. AdamWlr.001,wd0,clip1,warmup50,cosine to1e-5. Same64N16 dev K0..15 at steps918/1836/2754/3672/4590; select exact, rawfirst-tokenNLL, earliest. No supposedly fresh saturatedK16 dev. Retain all checkpoints and malformed/truncated generations.

New testseed20261010:272 contexts,8families perK0..16, eachN32 extended toN64 with the unchanged insertion/nuisance law. Exclude exact27 manifests: V10prior25 plus V10balanced main and V10fresh main. No new train/dev. Dry semantic/source/input plan before rendering; independently audit every QA/image/extension and all complete-context overlaps. Data rootv11_fresh. Bootstrapseed20261011,10000K-stratified complete-family resamples shared across lengths/conditions/fittedseeds; conditional-on-fitted-seeds intervals only. Primary in EACHseed:pre_last-minus-post_lastN64>=7/136 and N32loss<=6/136. Practical additionally pre_lastN32>=123/136,N64>=109/136,N64K9..16>=52/64. Pooling or historical results cannot rescue either-seed failure. Report K0..8,K9..15,K16 and single-digit/shared-first-digit diagnostics separately. Ordinary native greedy max4 with EOS151645/151643 and completeASCIIinteger+EOS exact; no output masks, penalties, forced digits or external tally in evaluation. No unseen-value or free-reasoning claim.

After a completed passing native V11 software profile, freeze and harvest the same33658 V10 training-only actual input identities at the new penultimate boundary, plus54 complete global prompts. Keep oldpixels/inputlayouts read-only; new tensors rootv11_parallel_local. Require prompt[-1]exactly equal its corresponding empty-prefix feature. CPUmetadata/tests/stage/merge each<=4CPUs/16G/10min. Cache GPUprofile<=3min with8VLM/4vision plus8actual final-block replays and16 standalone head evaluations; same-captured native full-batch/mask/position replay TV<=.02/exacttop1 binding, raw small/stress differences descriptive. Only if projected<=600s per shard, run4shards each<=10min, maximum4concurrentprojectGPUs; merge independently checks every tensor/ID/prompt/input/source. No test features enter training.

Training replays WHOLE globalprompt+target[:-1], with all live delta writes atP-1+t and exact native causal/mRoPE layouts. No targetEOS input, detached history, or isolated per-prefix training. Both run the actual frozen finalblock; oldhead-only training timings are ineligible. Two32-update profiles,seed16, each<=5min and no dev/test scoring:32live training final-block forwards;4 complete training-cache K0/9/10/16 sequence replays;10nativeN16 full-prefix model calls and10same-captured full-batch final-block replays;20 additional native-shaped norm/head projections (10capturedcarry,10blockreplay); two natural oldsoftware N16K3/N64K6 timing generations capped4tokens. Thus eachprofile<=18VLM/12vision,46standalone final-block calls/64totalfinal-blockcalls; the46replay helpers also each call the queriedhead. Independent K10 laterCE/past/current/future/other-scene delta gradients, zero-U auxiliaries, allparameter finite gradients, frozenbackbone andcheckpoint restoration are required. Reconstructed-cache versus native differences remain descriptive; samecapturedstate/nativebatch replay TV<=.02/exacttop1 binding. Measure new preprocessing/generation/load/update/peakmemory; four-token bound scales each full natural generation elapsed by4/observedtokens, then adds measured preparation. Use shared slower per-condition timing bounds for main projections.

Total V11 campaign cap4.5GPUhours(16200allocatedseconds), including software attempts, cache, profiles, mains, selected audit and failures. Existing software subcap540seconds remains. Four prospective main jobs each<=45min may launch only after both profiles, independent fresh-data/source/report checks and measured projected per-main<=2700s plus whole-campaign budget check. A selected-checkpoint/native verification audit<=5min follows, with independent CPU rescoring/final accounting before any success claim. The exact selected-audit call inventory must be frozen before main outcomes. This entry permits bounded CPU data/cache preparation and gated cache/profile jobs, not unreviewed main execution. AllGPU/heavyCPU useSlurm; models/data use the user-specified roots; RESULTS.md untouched.

V11 native software retry442259 passed83allocatedGPU-seconds, all39VLM/19vision/16extra-final-block calls completed. All41same-captured head positions and12same-batch final-block replays passed. The two known localN64 cached/fullTV.02029646 failures remain descriptive; allglobal-only replay comparisons passed. At fixed second-numeral CE, earlierdelta gradient norm.0789695 forpre_last and0 forpost_last, futuregradient0both; fixed future perturbation preserved earlier logits bitexact. Zero-U identity, common reads, lower/local KV, frozenweights andexactwrite/historypositions passed. TotalV11software31+83=114GPU-seconds. Feature-schemaCPU442262 passed2s. No V11 fitting or feature corpus harvest occurred.

V10 background diagnostic442260 completed88allocatedGPU-seconds,34VLM/34vision/442extrahead calls. All34bare native head bindings and136baseline decompositions passed; all136baseline first-token argmaxes matched the original saved V10 outputs. Background correction gaveN32=17/17 for allfourmodels; N64=15/17 CE14 and17/17 CE15/consistency14/consistency15. N64base=1/17,0/17,4/17,4/17; sham=1/17,3/17,3/17,2/17. CE14 remaining failuresK1->0 andK10->9. Every408registered row retained. This strongly supports the limited correctable-background account on these fixedfirst-token contexts, but K10..16sharefirst1 and no integer-plus-EOS generation was tested. No change to V10 efficacy decisions. CPUcalibration76s, failedCLIlaunch0GPU-s preserved. AnalysisSHA0b978114afa4d5520e7bb59fe0c3427a85be142b89e3a2be7e005129f375dd29.

Prospective priority change after this diagnostic: hold V11 main training and large cache harvesting while testing native full-answer background centering. V11 software/source preparation remains available, with no memory-efficacy claim or accepted objective. A new null-reference controller prototype may compute24 training K0 reference messages at the actual generated prefix in the SAME native batch as the actual images, enabling arbitrary prefixes without a lookup table or extra sequential model pass. This is source/CPU-prototype preparation only; a separate fixed native software/generation inventory and budget must precede GPU execution. The anchor16, question-specific known-null reference bank, extra24streams, and distinction from a retrained zero-centered general method must remain explicit.

### V12 null-reference native software preparation — 2026-09-11, before prototype execution

The first-token background rescue motivates testing complete-prefix behavior without a finite answer-prefix lookup. This software block does not release a new accuracy experiment. At each actual generated prefix, use one batched native model forward containing N actual N1-image streams,24 known-null reference N1-image streams, and one text-only global stream. References are exactly the occurrence-weighted N8K0 plusN16K0 training image banks for that question (24rows); preserve their original pixels/Step labels. Actual benchmark N excludes these reference rows. All modes receive the same augmented native batch and compute both actual/reference messages. Raw global greedy tokens are broadcast unchanged to every stream; no reference gold/0 token enters a prompt. This costs24extra parallel streams and still requires question-specific valid null examples; it is not yet an unrestricted general method.

Use the frozen original SUM/SiLU core. r=Wagg(sum actual messages)+query+bias; b0=Wagg(mean reference messages). Base leavesr unchanged; background subtracts(N−16)b0; sham subtracts(N−16)times a fixed-seed direction orthogonal to b0 with matched norm. Keep the V10 diagnostic seed20261101/question/core-key rule, recomputing orthogonality for the current b0; no coefficient or direction search. Native addition h+castFP16(delta), original norm/head and all actual/reference/local KV remain unchanged at fixed prefixes because fusion is only before finalnorm. Anchor16 and reference supervision are explicit limitations. No output vocabulary restriction, external tally or nonnative stopping. Whole-answer generation always uses ordinary greedy max4 and native EOS.

CPU4CPUs/16G/5min first: controller tests, source/model/processor/oldsoftware/input and null-bank semantic freeze. Fixed GPU software uses the same oldN16K3/N64K6 scenes and their training reference banks, with unchanged V7 fixed untrained zero/active branch states. Each scene: barecached3; zero-U/backgroundcached3; activebase/background/sham eachcached3 plusfullprefix t1/t2 (all historical query writes)=5 each. Thus42fixedVLM/22vision calls total. Add exactly two natural active-background timing generations, one perN, each capped4, for at most50VLM/24vision calls overall. One additional same-captured norm/head replay for every forward, no standalone decoder blocks. Native-zero identity, unchanged all-layer/actual/reference KV, actual/reference row ownership, correct prefix/mRoPE/current/history positions, finite/frozen weights and same-captured native-shaped head TV<=.02/exacttop1 are binding. Cached/full numerical comparisons remain descriptive and retain all failures. CPU tests must cover exact N16 anchor behavior, empty actual sets where allowed, projected-space algebra, shared current query, zero-U and nonzero gradients, sham norm/orthogonality/determinism, casting and cleanup.

One GPU software profile<=4min, totalV12softwarecap480allocatedGPU-seconds including failures; max4concurrentprojectGPUs. Raw tensors/models remain in prescribed roots under newv12 references paths. The new augmented native batch and preparation/generation timings must be measured; oldN+1 timings cannot be inherited. This releases source/CPU/software only, no selected-model generation campaign or parameter fitting. A full-answer diagnostic inventory and resource gate will be frozen separately after software passes.

V12 software arithmetic detail before source freeze: compute reference mean as FP64 mean followed by FP32 conversion, matching the V10 diagnostic calibration reduction. The remaining branch arithmetic staysFP32 and residual addition stays nativeFP16. Sham key is [core_key, exact_question], using core_key=software for fixed zero/active software states and original selected-model keys for any later registered generation diagnostic; it does not depend on N, prefix, mode or case order.

V12 reference unit442270 ran10tests successfully in2CPU-seconds, but root found its copied wrapper still snapshotted V11 files instead of the new V12 controller/tests. Treat the source-binding gate as incomplete and preserve the attempt. Correct only the explicit wrapper source-file list, then rerun before native source/input freezing. No tensor logic, test, model or GPU computation changed.

V12 corrected source-bound controller selftest442273 passed all10 tests in3allocatedCPU-seconds. CoreSHA6ee0acbb750a164cc66b78e8dff326a67fb1146b492778e1cbd782425f4b51b9; testSHA2199537c05d7ea44e27d7bb08a895d2aae0a937b77c6584ad46a410217e27ef1; wrapperSHA6659c5dfe7d6cd1b3126e1f9d094f3743254e75b5eaf237ee8d6e34529831fc4. Original442270 binding failure remains preserved. No V12 GPU execution yet.

### V12 complete native null-reference answer diagnostic — 2026-09-11, before selected-model outputs

After the registered native software gate passes, test allfour unchanged selected V10 cores (CE14/15, consistency14/15), exactly the34 identifier-selected V10 diagnostic scenes (one lowest pair_id per K0..16, N32 andN64), allthree base/background/sham modes. This is408 complete independent generated trajectories, not fitting or fresh confirmation. Keep frozen checkpoints,24 training K0 occurrence references per question, anchor16, coefficient1, fixed sham seed/key, original native decoder, ordinary raw global greedy argmax and native EOS/max4. At diverged prefixes, every mode recomputes its own actual/reference/global states; no later forwards or generated tokens are shared across trajectories. Report complete ASCII integer+EOS correctness, parse failures and truncation for every model/N/K, with K0..9 andK10..16 partitions and first-token accuracy separately. Every outcome remains exploratory and cannot alter V10's failed practical milestone.

CPU source/input freeze up to4CPUs/16G/5min binds passingV12software, selectedV10checkpoint/finalreport and background-calibration ancestry, all34 prepared augmented bundles, exact ordered reference-bank occurrence identities and every executable source. The same augmented N+25 native batch is used by allthree modes, including baseline, to match native numerical routes. No static first-token calibration table is used after generated prefixes. Bind all original images/QA, source parameters, prompts/masks/mRoPE and bank roles. CPU preparation may begin after software passes; no selected-model GPU inference before source and measured-resource gates.

One Slurm array of four tasks, eachonefrozenmodel and102generations (34scenes x3modes), at most408VLM/102vision calls per task; wholeblock at most1632VLM/408vision, no extra decoder-block or head replay calls. Raw per-step native logits are retained as lossless FP16-to-FP32 vectors with tokens, input/reference provenance, counters, timing, parameter identities and atomic progress. All modes include reference processing cost. Each task at most600allocatedGPU-seconds; total diagnostic envelope2700allocatedGPU-seconds including failures, separate from480software. Maximum4concurrentprojectGPUs. Before launch, require measured load +1.25*102*T64 +30 <=600 per task, where T64=augmentedN64software preparation seconds + complete natural generation seconds*4/observedgeneratedtokens; charge all34scenes at that N64bound. No reuse of old N+1 timings. Reserve four600-second tasks within the2700total, and retain actual accounting for all failures. No reasoningevaluation or newfitting released.

An independent CPU report must bind all408 raw trajectories/checkpoints/source/plan hashes and rederive token argmax, EOS stopping, integer parsing and every aggregate. No threshold tuning, bank pruning or choosing a winning seed after outputs. A consistent full-answer improvement motivates an unchanged fresh-family confirmation with separately frozen criteria and budget; it does not yet establish a general bank-free method or reasoning composition. AllGPU/heavyCPU runs useSlurm, allnew data/models use prescribed roots, RESULTS.md remains untouched.

V12 native software source freeze preparation: runtimeSHA6e8c8b93d57f246b2831d56ae9fc77197588687b3f00af3b187c7e1ef837044b; profilerSHA2a8dc04ca7dda50762be6e3228fdb2d226e1df0cfdefec1906d2fe7571fc52a6; CPUwrapper6daa98e5620a46b055feb914b840cf1e9c398915417ab34374f41c3efb503ef2; GPUwrapper782b1472349f5b41af9db8cb4961110ea583ac2a8c6be95b579e501b9947ac4a. An additional prospective empty-actual controller case completes the registered edge-case coverage:11tests now, testSHA6de630b51bbea4c65785142ef9acab62b298d8a22f9a849e5694a23ff2e84368. No controller tensor logic changed and the prior44227310-test snapshot remains valid for its recorded source. The full CPU freeze will execute all11 before GPU. Natural software generation timing includes all active capture/copy/artifact overhead, making the measured study projection conservative; no timing subtraction is permitted.

V12 independent runtime/profile review found no blocking computation, prefix, reference-order or provenance issue. FullCPUfreeze442309 submitted under the registered11-test/input/model/source gate,4CPUs/16G/5min. No V12 GPU execution yet.

V12 fullCPU442309 passed17allocatedseconds and all11controller tests; exact model/processor/runtime/reference/input/source bindings passed, planSHA63db8d44a5b1fbecda27a4d21d49adb95182e8da39b9aa5b9ffbb6c70a580be8. Native GPUsoftware442314 submitted under the unchanged240-second job/480-second total envelope, maximum50VLM/24vision plus50nativehead replay calls. No selected-model study execution yet.

V12 whole-answer parser clarification before any selected outputs: decoding with skip_special_tokens must not silently make a malformed answer valid. Explicitly flag any special token before the final EOS (or anywhere in a truncated trace); such an output is unparseable/incorrect even if skipping that token would leave digits. Preserve both raw/clean text and the flag, and independently test/rescore this condition. Native generation and EOS stopping do not change. This prospective clarification enforces the registered complete ASCII integer+EOS rule; no V10 result is rescored or replaced.

V12 native software442314 passed130allocatedGPU-seconds:46VLM/24vision/46additionalnativehead calls,24all-layer fixed-prefix KV comparisons,36fusion checks,6zero-U identity checks and780cached/full row comparisons. All binding checks passed; this augmented batch also had zero descriptive numerical failures. Software generated2tokens in each fixed natural timing case; no efficacy scoring. Peak allocated15,420,464,640bytes. TotalV12software130/480GPU-seconds.

V12 original study timing gate FAILED before any selected-model inference: measured augmentedN64 four-token bound7.831716092070565s,load22.890363042941317s imply22.890363042941317+1.25*102*7.831716092070565+30=1051.4341647819383s per model, above600. Preserve this original gate in outputs/native_aggregation_vlm/v12/original_study_timing_gate.json. The bound retains all software capture/copy/artifact overhead and charges every scene atN64; it is not an uninstrumented inference latency claim.

Prospective resource-only amendment before study source freeze or selected outcomes: extend each of the four study allocations from600 to1200seconds, total study cap2700to5100allocatedGPU-seconds including failures (four1200reservations plus300reserve). Keep software cap480 and actual130separate; maximum4projectGPUs. Projection1051.434<=1200 under the identical formula; no work is removed from the estimate. All408 trajectories, three interventions, four frozen models,34fixedscenes, raw native generation, reference-bank rule, anchor/sham/arithmetic, scoring, source/input audit and no-selection protocol remain unchanged. CPU study preparation/report remains4CPUs/16G/5min. This amendment changes only resource reservation; the original ten-minute timing failure remains reported.

V12 studyCPU442328 passed116allocatedseconds: all34augmented native input bundles,1296reference occurrences,54banks,allfourselectedmodels,whole-answerparser checks and source/software/timing bindings. Its originalsourceplan is retained. Before any selected GPU launch, independent review found its run guard incorrectly requiredFP16 while native generation returns an exact FP32 promotion. BoundedCPU442339 confirmed both saved software natural raw tensors areFP32, finite and exactly equal to FP16roundtrip;2allocatedCPU-seconds, wrapperSHA64434cb48edb4c2afb075b978ce85a384b49fb113230dc5d6cd64677f167c027.

Repair only the study run raw-dtype assertion to requireFP32 plus exactFP16roundtrip, matching its already-correct archive/report contract. No logits, inference arithmetic, scoring, generation, reference, resource, source-ancestor or selection changes. RevisedstudyscriptSHA4cfb37a75a1782e92451a2c126da688dddda76d951317642cfd8581cfff2ba53. CPUwrapper3547db2de7a365992eb52e278b68ad012434e932b1765c33c6bd77ebbf27cdb6; GPUwrapper56deb211dbf7ac15ac01e289b7ffcf5494b966fdb2052bff0964c1e6a08d65cf. RepeatfullCPUfreeze before selected GPU execution. No study GPUseconds spent.

V12 correctedstudyCPU442347 passed118allocatedseconds, complete34-scene/four-core/reference/software/source audit and prospective strictparser tests. PlanSHA9a59d45222dccb6319313256b56da7bbf5ea3348ae10062e580897a4608eff06; projection1051.4341647819383s/model<=1200, priorstudyGPUspend0, four1200reservations<=5100. Finalindependent source/schema review found no remaining blocker. The fixed four-task selected-model study is released; no outcomes observed yet.

### V13 learned null mean — CPU inventory only,2026-09-11 before execution

While the fixed V12 complete-answer diagnostic executes unchanged, prepare a training-only input inventory for the small conditional-null predictor proposal. This releases no feature GPU harvest, predictor fit, model selection, inference or new test. Retain all108K0 training scenes/1296ordered occurrences,54questions,18strict native prefixes from V10answer0..16support. Assign prefixes independently for null calibration only; never apply incorrect zero-answer CE to positive-prefix auxiliary views. Deduplicate extraction by exactimage/question/prefix while preserving all24occurrence weights in each of972auxiliary groups.

Metadata audit predicts22,932unique local features:3,268reused,19,664missing; all972globalstates reused. ParentV10cache33,658 states unchanged; union53,322; missingFP16payload140,951,552bytes. Fourprospective4916-row shard inventories imply77batchesof64 each, but no execution of them is released. CPU4cores/16G/5min verifies parent source/model/runtime/layout/QA/image/provenance, exact descriptors and no-future-EOS prefixes, then writes newplan only under prescribed data rootv13_null_features. Existing tensors/pixels remain immutable; no tensor/model loading is claimed by this metadata stage. Exact source registration in outputs/native_aggregation_vlm/v13/null_inventory_registration.json, with stagerSHA3c41fbca44e3c5b429dbd658a7131d4f03df7e120d31febeb31e8bd67b89df63 and wrapperSHA05c4a0db15eea9894a913b05510aee2343bbd586a34cc929edd6fca28da0defb. The conceptual18,624-parameter conditional mean predictor remains a proposal pending V12 complete-answer evidence, native feature checks, explicit fit design and resource registration.

V12 selected-model study array442349 launched four parallel tasks under the frozen correctedCPU442347 plan, all408trajectories unchanged. V13 CPUinventory442353 passed6allocatedseconds: allpredicted22,932requiredlocal/3,268reuse/19,664missing and972reusedglobalcounts verified, planSHAfa6e3014dff2a083cf45dab1888c065bb52e55244e06379dbca44ae1e03a88ef. V13 remains inventory-only; no GPUharvest or predictor fit released.

### V12 complete-answer diagnostic verified outcome —2026-09-11

Allfourselected-model jobs completed (CE14/15 291/290GPU-s, consistency14/15 290/291); totalstudy1162GPU-seconds. Independentreport442359 passed22CPU-seconds, all408rawFP32promotednative trajectories fully rescored/bound. Total1006VLM/408vision/fusion1006, withinregisteredlimits. No study failures. Including native software130, totalV12=1292GPU-seconds, preservingoriginal600-second projectionfailure andprospective20-minuteextension (actualjobs~5min each).

N32 wholeinteger+EOS base2/17,3/17,5/17,6/17 versusbackground14/17,14/17,14/17,16/17 (CE14,CE15,consistency14,consistency15 order); sham3/17,2/17,0/17,0/17. N64base1/17,0/17,3/17,4/17 versusbackground10/17,12/17,12/17,12/17; sham1/17,1/17,0/17,0/17. EverybackgroundgenerationparseableandEOScompleted, nottruncated. N64backgroundK0..9=9/10,10/10,10/10,10/10, butK10..16=1/7,2/7,2/7,2/7. The first-token rescue therefore extends to many complete answers but doesnot solve multi-digitprecision. CE14alsoemitted130for15 and140for16; these remain wrong validintegers. Pooled39/40lowcounts and7/28highercounts are descriptive, not independent-seedreplication or a revised criterion.

CanonicalanalysisSHA6ef7b842f0da4f35b526af2c2870a634086e5c16347842527b4437302b3e855b. The17familiesare reused exploratory contexts; allfourcores,24training-referenceoccurrences,anchor16 andextra24native streams remain explicit. V10practicalfailureunchanged, nofreshconfirmation orreasoningcomposition, noacceptedresearchobjective.

Priority afterV12: develop a learned conditional null mean and train from initialization in centered coordinates, with a same-capacity query-offset control. Hold the optional3264-trajectory fresh confirmation of the incomplete frozenreference procedure. The untouched V11fresh-data stager remains available for the next properlyregisteredmethodcomparison. V13 featureinventorypreparation is alreadycomplete; GPUfeature/profile andfitting protocols still require prospective source/call/resource registration. This decision is based on the nowobserved strong directional benefit andremainingmulti-digit failure, and is explicitly postdiagnostic.

### V13 learned conditional null mean: native software and matched training —2026-09-11, before execution

V12's fixed correction improves complete answers but leaves N64K10..16 only7/28 across repeated fitted models. Test learning the expected irrelevant message and training directly in centered coordinates. Keep the original1,041,600-parameter rank96SUM/SiLU core and native final-norm residual. Add a separate18624-parameter predictor c(q)=Linear96→SiLU→Linear96, q=WqRMS(g), signedFP32output, finalweight/biasinitiallyzero. Total1,060,224 trainable parameters. The predictor receives no N, queryID, tokenID or prefix table; current nativeg carries the generated prefix. Each inference token uses exactly N actualN1image streams plus oneglobaltext stream, no runtime reference bank.

Matched conditions offset andcentered use exactly the same parameters, training data and auxiliary supervision. r=Wagg(sumactualmessages)+q including its originalaffinebias; b0=linear(c(q),Wagg.weight) withoutbias. Offset usesU*SiLU(r-b0); centered usesU*SiLU(r-N*b0). NoN16anchor in these trained models. Preserve projected-space subtraction order, FP32branch/predictor and nativeh+castFP16(delta), originalRMS/head and unrestricted greedyinteger+EOSmax4. SUM's centeredstatistic is additive atcommonq; the finalnonlinearresidual is not assertedadditive. No newattention/rank/bandwidththeorem claim.

Both arms optimize ordinaryfullsequenceCE plus coefficient1 residualconsistency using unchanged common-globalnorm denominator, and coefficient1 null-mean calibration. MainCE AND consistency differentiate fully throughc(q),q andthecore. Auxiliaryonly: computecorequery and24orderednullmessages underno_grad; target=FP64meanmessages→FP32, prediction=c(detachedq), pergroup squared error meanover96 divided by detachedFP32message squaredmeanover24×96 plus1e-6. Average16groups. Thus only predictorparameters receive auxiliarygradients; the auxiliary does notdirectly force localfeatures tozero. This doesnotprevent coadaptation/rescaling; report rawandprojectedcalibrationerrors, null/predictor/readoutnorms. NoEMA, addedvariancepenalty, largerpredictor, full3584globalinput or loss-weight search.

UseexactV10ordinarytrain/dev/pairs:1782uniquetrainscenes,918pairs/1836weightedscenes,answers0..16,maxtrainN16;40epochs, carried8-pair batches,4590updates/73440scene presentations/177120validtargets. Samefive64-sceneN16devsweeps at918/1836/2754/3672/4590, selectcompleteexact thenrawfirsttokenNLL thenearliest; retainallcheckpoints. Seeds16/17, corethenpredictor initialization fromsameperseedRNG, identicalbytes/order withinarms. AdamWlr.001,wd0,warmup50,cosine to1e-5; two core/predictor parameter groups with separateclipnorm1 each, identically inbotharms. Thisnewoptimizergrouping is explicit. Auxiliary groups are all972question/strictprefixcombinations from54questions and18prefixes, each24occurrenceweightedK0trainingimages. Independent seededshuffle(seed+20261105), carried16-groupbatches overcycles for4590updates/73440auxiliarygrouppresentations. No wrongzero-answerCE on assignedpositiveprefix views. AllstrictprefixesexcludetargetEOS; no development/testimages entertrainingcache.

Reassign the stillunexecuted V11fresh-data stager tothis V13comparison withoutchangingitsbytes,naming,seed orlaw: destinationv11_fresh, seed20261010,272contexts/136N32→N64families,eightperK0..16. Excludeexact27manifests(V10prior25+V10balanced+V10fresh); allV12studycontextsarealreadycontained inV10fresh. PreserveV10train/dev/schedule/pairing andeveryearlierdatafile. CPUdrysemantic/sourceplan before rendering,4cores16G10minstagerwrapper; independentpublishedQA/image/extension/nuisance/source audit before any mainfit. Newmethod source/supportunchangedbytestlabels.

Both-seed primary: centered-minus-offsetN64>=7/136 AND centeredN32loss<=6/136. Practical additionally inEACHcenteredseed N32>=123/136,N64>=109/136,andN64K9..16>=52/64. ReportK0..8,K9..15,K16 andsingle/shared-first-digitpartitions, allmalformed/truncatedoutputs, rawexactcounts andinferencecost. Bootstrap20261106,10000K-stratified complete-family pairedresamples, sharedacrosslengths/conditions/seeds; uncertaintyconditionalonfittedseeds. Pooling/historicalcontrols cannotrescueeitherseedfailure. No reasoningcompositionclaim orreasoningefficacyexecution released.

Native software before fitting: originaloldN16K3/N64K6cases, fixedunfittedV7zero/activecoreweights; newpredictorinitialstate seed20261104 with C2 made deterministicallynonzero(std.001) foractivechecks, exactsavedstates. Eachcasebarecached3,zeroUcenteredcached3,activeoffsetcached3+fullprefix t1/t2,activecenteredsame5=16VLM/8vision. Total32fixedVLM/16vision. Addexactlyfourmode×N natural timinggenerations max4 each =>max48VLM/20vision total, oneadditional samecaptured native-fullbatch lastquerynorm/head replay peractualforward(max48), zero standalone decoderblocks. Fixedprefixall-layer/all-rowKV and nativehiddenunchanged; explicitquery/historypositions,zeroUidentity,onlyglobalwrite/nativecast,frozenbackbone andnativeheadTV<=.02/exacttop1 binding. Cached/full differencesdescriptive, allfailuresretained. Naturaltiming observes onlyfullbatchlastquerynorminput/globalrawlogits, withoutfullhidden orKV CPUcopyoverhead; actualnativegeneration stillvalidatesmasks/mRoPE/cachelength/broadcast/rawargmax/EOS/counts. Thisinstrumentationdifference fromV12 is explicit and not a retrospective timingadjustment. CPU4/16G5min; softwareGPUjob<=300s,totalsoftwarecap600includingfailures.

Feature preparationuses verifiedV13inventory442353:19664missinglocalprefixstates, all972globalstates and3268requiredlocalstatesreused, allold33658states retainedbyteexact; finalunion53322. Profilefixedprefixstrata1,9,10,16: eachtwo lowest-IDmissingnullfeatures andtwo lowest-IDexistingordinaryV10localcontrols, one4-row nativebatch andone64-rowcyclicrepetition. Exactly8VLM/8vision/8standalone samecapturedheadprojections, eightdistinctmissing plus eightreusedcontrols. Prefix16reusedcontrolsneednotbenull, explicitlymarked; nativehidden4vs64/oldcache discrepanciesdescriptive. No extraoldstatevocabularyprojections. Fourharvestshards4916rows each,77batchesof64 each, total308VLM/308vision, noheadreplays inharvest; totalprofile+harvest316VLM/316vision. CPUfreezeselects allIDsbeforeoutputs. Profile<=180GPU-s; measuredprojecteachshard<=600s beforelaunch; fourshards<=600s each,featureenvelope2700GPU-sincludingfailures. IndependentCPUmerge bindsallreused/newfeatures,source/pixel/prefix/shardidentities; no data deletion or ancestor edits.

Two trainingprofiles,seed16,onepercondition,32updates each,<=300GPU-s each. Then10fixednativeN16prefills coveringallstrictprefixesofK0,9,10,16,10vision; eachonebindingcapturedhead andonedescriptivecached-statehead projection=20standaloneheadcalls. No additional naturalgeneration ifsuccessfulV13software suppliesnewmode×Ntimings. VerifyinitialzeroU/zeroC2lossandgradparity, mainversusauxgradientrouting atsteps1/2/32, allrequiredfiniteparametergradients, separateclipnorms, frozenbackbone andcore+predictorcheckpoint restoration; log16pergroupnull losses/denominators andexactauxiliaryorder eachstep.

WholeV13campaigncap16200allocatedGPU-seconds includesallsoftware,features,trainingprofiles,failures,fourmains andselectedaudit; maximum4concurrentprojectGPUs. Fourprospectivemains<=2700s each onlyafterbothprofiles,independentfresh-data/cache/source/reportchecks,sharedslowerperarmupdate/newnativegenerationprojections<=2700 andwholecampaignreservecheck. Selectedcheckpointaudit<=300s afterwards: sameoldtwo fixedsoftwarecases,onebarecached3sequence percase plus eachof4selectedmodels cached3/fullt1,t2=23VLM/13visionpercase, total46VLM/26vision,46samecapturednativeheadreplays,656cached/full rowcomparisons, all24selectedfixedprefixKV comparisonsagainstbare. Keepallpriornative numericalfailuresdescriptive; software/head/sourcechecksremainbinding. Finalindependentscoring/selectedbinding/resourceaccounting requiredbeforeefficacyclaims. Thisregisters boundedpreparation/software/feature/profile work subjecttoitsgates, notunreviewedmainexecution. AllGPU/heavyCPU useSlurm; models/data stay in prescribedroots; RESULTS.md untouched.

V13 standalone conditional-mean CPU442380 passed11tests in2allocatedseconds; helperSHA95bbfc2091493bfe113032ffcd009aa3e2082fb432ca8df524f92739231a923f, testSHA780f0313ff98f95c16a948851ace7f8a4baa32df67553a47f4389340ebcf0688. Freshdata dry442378 passed1s, planSHA40314056f84a1b5177d99bd0361bb3da69d8f56ed81a6f5c04ae2d6451bab28c; first rendering442381 passed42CPU-seconds,272contexts/13056imageoccurrences/3309uniquerenders. ExactV11stager/source/root reusedforV13, no V11memorytraining. IndependentmanifestCPU442386 submitted.

V13 cacheworker source reviewed before execution; corrected a dict-iteration error in the future merge before freezing. FinalworkerSHA756a2b213867327dab1808b0e991f92361244e9821abe3b98e3b0c659c927f75, exact source/policyregistration outputs/native_aggregation_vlm/v13/null_cache_registration.json. The8VLM/8vision/8headprofile,308harvestcalls,19664missing/53322union,180sprofile/600s-shards/2700total resources remain asregistered. Exact sharedshard projection is model_load + pixel_load_and_verify +1.5*77*max(fourstress64harvest_seconds)+30 <=600. Profile staging fixes all16distinctIDs independentlyofoutputs. No V13GPUwork yet.

V13 independentfreshmanifest442386 passed18CPU-seconds. CacheworkerCPU442388 passed7s, planSHA3987c2001c0e168e7584377bfdc8a6dfd37c701823ce708e75a295e6d3010aa1. FeatureGPUprofile442394 passed40allocatedGPU-seconds,8VLM/8vision/8samecapturedheads, allbindingchecks; projectedshards[219.04904262011405, 219.04904262011405, 219.04904262011405, 219.04904262011405]seconds<=600. PriorfeatureGPU40 +four600reservations=2440<=2700; fourregisteredfeatureharvesttasks released. No fitting or benchmarkpredictions.

V13 four featureharvests442405/442406/442407/442404 completed150/151/149/150GPU-seconds, totalfeature640including40profile. CPUmerge442411 failed exactdict equality for saved descriptive hidden reductions. The originalworker/source/artifacts andfailure are preserved. Both original and rederived reductions execute onCPU; host/layout reduction differences are a hypothesis, not yet established. New boundedCPUdiagnostic will compare every raw feature/control identity, exactflags/maxima, old/stress hidden metrics and allunchanged nativehead gates before any tolerance-only dispatcher is considered. No tensor replacement, extractionrerun, model/loss/scoring change is authorized by this repair. NativeCPU442412 passed11mean tests plus6controllerchecks, planSHA56613183c3452df0e8bfa3fb05e425b4222dcaeabdd4468147a7b1c7b0140d47; GPUsoftware442413 executes the unchangedregistered inventory.

V13 mainprojection clarification before trainingprofiles oroutcomes: shared maximum step_seconds_max_steady acrossboth32updateprofiles, eachcomputedoversteps5..32; sharedT16/T64=maxacrossnativeoffset/centeredsoftware four-tokenbounds. Perarmprojection=itsmeasuredmodel+featuresload +1.25*(4590*sharedstep+320*T16+272*T64)+120 <=2700; N32chargedN64. Fullcampaign Slurm discovery includesalluserGPUallocations whoseJobNamestarts v13_ since2026-09-01, includingfailures; no mainrelease untilallpriorGPUallocations terminal andprioractual+10800main+300selected<=16200. Newselected-audit CPUselftest/source freeze binds before mainoutcomes. Auxiliaryprojectedcalibration/query/readoutnorm scalarlogs now implementedasdetacheddiagnostics only, leavingalllosses/modelcalls unchanged; trainerSHA84b33cc0b2f8802816fdd3e411c2412481b9cda4010c306c1992e0a06bb668d8.

V13 native software442413 passed68allocatedGPU-seconds,40VLM/20vision/40samecapturedheads,18KV/26fusion/6zeroidentitychecks and328cached/full comparisons. Two knownlocalN64row62t2TV.020296463569398154 differences remain descriptive, top1equal; allbindingchecks passed. Sharedfour-tokenboundsT16=1.301686738152057,T64=4.345250430051237seconds. TotalV13GPU708(640features+68software).

CPUdiagnostic442416 passed4.26scriptseconds, exact immutable tensors/metadata/maxfields andall8nativeheadgates;392descriptivecomparisons,17RMSdifferencesmax7.450580596923828e-9 and17relativeRMSdifferencesmax9.116197380309998e-12, noL2differences. This localizes the mergefailure to tiny descriptive scalar recomputation differences; it doesnotidentifya specifichostkernelcause. Permit a separatelysourceboundCPUdispatcher withmath.isclose rtol1e-6/atol1e-8 ONLY for descriptive rms/l2/relative_rms fields. Alltensors, identities, flags/maxima andnativehead .02/exacttop1 checksstayunchanged. Preservefrozenworker756a2b..., original442411failure, alloriginalscalarrecords anddiagnostic442416. No featureextractionrerun, model/loss/generation/scoring change.

V13 explicit precision adapter source6a51e9e23e34dfcc5bdd2b26d27bb4262c842ae4a5fb3d5ce2c14f8b0c405852 andwrappercda745fa0def079a8d7a7d7810d1a6738cc78cb83668699a3df772e190caf476 independentlyreviewed: no monkeypatching, frozenworkerunchanged, fullmerge/tensor/headconditionspreserved. Adapterselftests execute beforemerge; cache/summarybindnewsource snapshots, original442411failurelog/source anddiagnostic442416. Unfrozentrainersourceupdatedonlyto requirethisexplicitcacheadapter/provenance, SHAb2946658a7fcbefe5c29e8b5a6ad800c0a39ed491bb2a5782b9eb9bb69b0c993; no trainingmathchange.

V13 correctedCPUmerge442432 passed12allocatedseconds, all53322nativefeaturevectors exact andall8headcomparisons verified; unionSHA4ce0cfdd71f44e4f03f34673fe6c91b89a7594889587f5328dcd0c62b7e9f46b. Original442411scalarfailure retained. IndependentreporterCPUselftest442433passed2s; fullfreshdata/cachecheck442437passed42s. TrainerCPU442436passed26s, planSHAf30e03028e19d9720fab2a2301e455fea01e2b96745038618b9cc311422ceaaf. Two32updateprofiles442442centered/442441offset passed46/47allocatedGPU-seconds; alllearnedtensors exercised, mainpredictorgradientslive/auxiliarycoregradientsNone,10nativeprefixheadreplayspassperarm,0descriptivecachefailures. Sharedworststeadyupdate0.016490699956193566s. V13priorGPUtotal801(640features+68native+93profiles). Selected-audit source independentlyreviewed andCPUselftest442443passed; no mainoutcomes yet.

V13 mainrelease442444 passed33CPU-seconds. Independentfulltrainingprofile/rawnative timing/source/data/selected-source audits pass; Slurmallocationdiscovery verifiesall8priorGPUjobs801GPU-seconds, includingallpriorfailures(noneGPU). Measuredmainprojections2228.044472413021scentered and2228.630835599848soffset<=2700. Prior801+four2700+selected300=11901<=16200. Fourregisteredmainfits nowreleased underunchangedobjective/model/data/seed/selection/scoring; no outcomes observed.

V13 finalizer/plot source prepared beforetestoutcomes: scripts/finalize_native_vision_v13.py SHA8b7a6a705a32115904ce633a3b2249bdec6663c25e1fcefeefa44ccb71d25865, wrapper8b495a72b66e9c7a092e640ce78e5c78bdaeab032473923e73cbc5e34bb9b191. Recomputesregisteredintegerdecisions fromexistingreport, bindsall4combinedselectedmodels/46postheads/656descriptivecachecomparisons, discoversallv13_GPUallocationssinceSept1includingfailures, checks13successfulartifacts/perjobcaps/subcaps/16200total/max4. Copies32fixedexistingmetricbars; no newestimates,predictions,selection orintervals. CPUselftests beforefinalization.

### V13 selected calibration geometry — prospective CPU diagnostic, before V13 test outcomes inspected

Training logs show larger normalized calibration loss in centered runs than offset controls, while allfour reach64/64dev. To distinguish estimator error from message spread, register one descriptive CPU-only diagnostic on allfour immutable dev-selected checkpoints after the existing selected-checkpoint CPUaudit binds them. This diagnostic cannot change checkpoints, scoring, release, primary/practical decisions or native outputs. Use canonical V13union and all972question/strictprefixgroups with24orderednulloccurrences, no group selection. Compute currentFP32messages and FP64mean→FP32, raw/predicted/projectedmean errors andnorms, empiricalprojectedcovariancetrace withpopulationnormalization1/24, and separateN8/N16sourcebankmeancontrast. Preserve duplicatedoccurrences. These are training-bank diagnostics, not independentnullcalibration.

At every ordinaryunique training context/prefix (1782scenes,4266validpositions), compare deployed preactivation/residual with replacing onlyc(q) bythatgroup's empiricalmean, retainingthearm'sregisteredcoefficient1oractualN. Report exactpreactivationdifference, residualdifference and firstorderlocalreadoutresponse coefficient*U*diag(SiLU'(t))*W(c−mean); compute empiricalnullspread throughthesameJacobian withoutformingfullH×Hmatrices. No vocabularyhead, VLM, nativeaccuracy or proposedinferenceoracle. Group-levelN²projectedbias versus(N−K)projectedspread atN16/32/64 may be reportedonlyasfixed-bank IIDreference quantities withKspecified; no generated-answerclaims. Reportallquestions/prefixes/K/N andeverymodel, nofavorablecellselection.

OneCPU Slurm job4cores/16G/5min, noGPUallocation; models/cache readfromprescribedroots, JSONdiagnosticsinnewv13/calibration_geometrygroup withsource/input/selected-plan/checkpointbindings. Source and synthetic finite-differenceJacobian/occurrence-weight/algebra selftests freezebeforeexecution. No backbone/headloaded, no additionalfit, no branch/meanpredictorupdate. Thisdiagnostic is motivated by alreadyobservedtraininglogs andtheprospectiveestimand memo; V13testoutcomes have notbeen inspected atregistration.

V13 finalizer pre-executionreview caught actualKVschema mismatch afterselftest442468: selectedauditor stores28per-layerbooleans per cachedcheck, whilefinalizer/fixture expectedscalarTrue. Preserveoriginalsource/selftest. CorrectONLY finalizer/schemafixture toexactly24lists of28literalTrue and16fullchecksNone; rejectmissing/falsenumeric/scalarflags. Revisedsourcee99383f4390b5293c032f407fcb029b6be2de6e09661118720e52f85a22ce0e2 independentlyreviewed, correctedCPUselftest442477passed. No selected/model/numerical/scoringchange.

V13 fourmains completed921/921/921/932GPU-seconds (centered16/offset16/centered17/offset17), all4selectedstep4590 and64/64dev. Independentreport442466 passed74.35scriptseconds; centeredN32=44/136,48/136 andN64=11/136,15/136; offsetN32=61/136,26/136 andN64=20/136,0/136. All4modelsN64K9..16=0/64. Seed16primaryfails, seed17relativeprimarypasses; bothseedprimary/practicalFAIL. No successfulmethod orreasoningclaim. SelectedCPU442467passed33s; GPU442472passed83s,46VLM/26vision/46heads,24KV/40fusion checks;4descriptivecached/fullfailuresretained. ExpectedtotalV13=4579GPU-seconds pendingindependentfinalaccounting. Registeredcalibrationgeometrydiagnostic unchanged afteroutcomes.

V13 calibrationgeometrysourcef099b18c327d5e36451104ad121d8042b63fca022c40d7d45c2ed18e11d63c4e andwrapper34c8295ef1d1a21d4478ebf02c69b4d922e45b4aeb51518616ff3ee90a62d4c1 independentlyreviewed; metadataexplicitmixedFP32/FP64, nomathchange. CPUselftest442482passed2s; fullgeometry442483passed26s,3888groups/17064uniqueordinarypositions, noVLM/head/GPU/fitting. Mean normalizedbankMSE centered.19736/.25303 versusoffset.01184/.00924; meanprojectederrors94.978/101.632 versus6.808/8.873. Centered projectednullcovtraces28.539/24.455, control25.076/46.408. Meanordinaryresidualchange underempiricalmeansubstitution758.60/895.30centered versus7.62/5.44offset. Thisestablishespoorbankcalibrationandlargefrozen-readoutsensitivity; it doesnotestablishnativeaccuracycausality orpopulationnullcalibration. V13failureunchanged. A separatenextprotocol must preventtaskgradientsfromrepurposingthemean ratherthan merelyincreasejointtraining.


### V14 exact empirical centering followed by frozen-core distillation — 2026-09-11, before new data/jobs

V13 is verified failed, with final accounting4579GPU-seconds. Independent geometry audit corrects a units ambiguity: projected error norms and covariance traces cannot be compared directly. The within-model ratio mean squared projected error / mean projected covariance trace is326.82/443.99 for centered16/17 and2.85/2.42 for offset16/17. These training-bank, coordinate-dependent diagnostics establish a large empirical mismatch, not its causal contribution to native accuracy. No V13 output, source or decision changes.

V14 tests one cleaner training principle: learn the core with exact empirical centering, then freeze the core before distilling its null mean. During every ordinary training prefix, compute the original rank96 messages for the24 registered occurrence-weighted K0 reference streams at the same current global query. The empirical mean uses FP64 accumulation then FP32, with FULL autograd through the reference messages and query. Retain duplicates. Form original r=Wagg(sum actual messages)+q+b_rho and subtract a*Wagg(mean) without projection bias, a=actualN for centered and1 for offset. Original U/SiLU readout, raw-zero message subtraction, native norm/head and frozen backbone remain unchanged. Core phase contains no learned mean predictor. Both arms use complete-answer CE plus coefficient1 paired residual consistency; no loss on assigned null prefixes. Existing V13 union53322/972groups and V10 training1782unique scenes/918pairs are reused byte-for-byte.

Matched seeds18/19, same persistent Random(seed) ordinary order, 40epochs/4590updates/batch16 (8intactpairs),73440scene slots,177120native target positions, AdamW lr.001/warmup50/cosine final1e-5/wd0/clip1. Both arms core1041600parameters. Save the fixed final core at4590 and freeze every parameter. Precompute all972 continuous rank96 queries, means and detached FP32 mean(message squared over24*96)+1e-6 denominators. Only then initialize the separate18624parameter ConditionalNullMean using independent torch seed(seed+20261110), and train it solely with uniform-group normalized mean96 squared error. Fixed8000updates, batch64,512000group presentations; persistent Random(seed+20261111) permutations of sorted972groups, carrying cycle tails. Student AdamW lr.001/warmup100/cosine final1e-5/wd0/clip1. No main-loss gradients to student, no later core update, no adaptive extension or error-based selection. Final combined model1060224parameters. Match initial states, data orders and budgets across arms; first core gradients need not match because exact-bank subtraction already changes the initial readout inputs.

After student8000, use that final checkpoint only. Evaluate one64-scene V10N16 dev sweep descriptively; it cannot select a checkpoint, extend training or exclude a model. Evaluate every model on fresh272contexts/136independent N32-to-N64 families,8perK0..16, seed20261108, destinationv14_fresh, excluding all28exact prior manifest paths (the frozen V11stager's27 plus now-usedv11_fresh). Protect original training/dev/pair/schedule and allparent audits. Freshness concerns complete context/question combinations, not new questions, answer values or visual atoms. Native deployment uses unchanged V13N+1 batched streams, actualimages plusglobaltext, continuous query-to-mean predictor, original native greedy max4/EOS151645or151643, strict ASCIIinteger-plus-EOS and rejection of nonterminalspecialtokens. No reference images or N16anchor at inference. No new VLMfeature harvest or runtime operator change.

Primary perseed: centered N64 gain at least7/136 and N32 loss at most6/136 versus matchedoffset; bothseeds required. Practical additionally centeredN32>=123/136,N64>=109/136 andN64K9..16>=52/64 for bothseeds. Report all integer counts, K0..8/K9..15/K16, parse/EOS/truncation/firsttoken and seed-specific differences. Bootstrap10000, seed20261109, K-stratified intact paired families shared acrossarms/seeds; no pooling as independent observations. Reasoning composition remains untested even if the vision milestone passes.

At the fixed final model, record all972 groups' normalizedMSE, projected-error norms and squared norms, projected null covariance trace, maxima and prefix breakdowns. On all1782unique ordinary scenes/4266prefixpositions record bank-versus-distilled preactivation and nonlinear residual changes and the algebra t_distilled-t_bank=-a*W(c-mean). These are complete cached training-state conversion diagnostics, with no VLM/head/accuracy oracle and no selection role. Verify fixed core/query/target equality throughout student fitting, student-only gradient flow and differentiable reference mean during core fitting. Exact empirical centering still has finite-bank/position-shift/variance limitations; the compact query/predictor need not represent every mean. Neither small regression loss nor residual norms establish accuracy or capacity claims.

Execution: source/CPU tests and independent data/cache audit first. Two simultaneous seed18 profiles, centered/offset, each32core+32student updates and all972 target precomputation plus complete conversion diagnostics; unchanged10nativeprefills/10vision/20standaloneheadchecks perprofile. Each<=300allocatedGPU-seconds. Reuse fullybound V13native software442413 (same inference sources/hardware), including known descriptive local numerical differences. Measure shared max core steps5..32, student steps5..32, full972precomputation and fullconversiondiagnostics acrossprofiles. Perarm projection=its measured load+1.25*(4590*shared_core_step+8000*shared_student_step+shared_precompute+shared_conversion+64*T16+272*T64)+120<=2700; N32chargedN64. No accuracy is evaluated during profiling. Freeze independent reporter and final-checkpoint audit before mains. The selected audit retains V13 inventory46VLM/26vision/46head,24cachedKV lists of28literalTrue,40fusion,656descriptivecached/full comparisons; <=300GPU-seconds. CPUrelease checks source/data/caches/profiles and alluser v14_ GPU Slurm allocations since2026-09-01 including failedattempts. Prioractual+4*2700+300<=16200campaign cap, max4concurrentprojectGPUs. Reused V13costs remain explicitly accounted in V13 rather than double-counted in V14. Only a passed measured release authorizes fourmains. Independent CPUrescoring, checkpoint/nativeaudit and finalaccounting precede any success claim. AllGPU/heavyCPU viaSlurm, models/data inuserroots. RESULTS.md untouched.


V14 CPUfresh dry442494 passed1s; bound rendering442496 passed43s,272contexts/136families/3308unique renders, all28exclusions/QA/images/semanticextension checks passed. ManifestSHAed900cc26eb99575325583de8ef7d18beeac506c4baedd5a2c3f67ae70adf909. NoV14GPUwork oroutcomes. Before trainer/profile freeze, clarify diagnostic artifact portability: retain alreadycomputed final-core rawmessages and projectedmessages, both972x24x96FP32, alongside972q/mean/denominator/variance targets in the user data root, with exact tensor/file hashes. Independent CPU recomputation of descriptive FP32 reductions and FP64mean-castFP32 may use rtol1e-6/atol1e-8 while recording exact equality and maximum deviation; FP64covariance reductions from saved projectedmessages use rtol1e-9/atol1e-10. Do not require cross-device F.linear bitidentity or silently rewrite any fitted target. These tolerances concern audit recomputation only; trainingmath/targets, tensoridentity, nativehead gates, checkpoints and accuracy/scoring remain unchanged.

V14 prospective finalizerCPUselftest442511 failed on a stale fixture seed key16 after changing registeredseeds to18/19. Preserve failedsource/log; correct only thatfixture keyto18 before retry. No V14GPUjobs, models, predictions or decisions exist.

V14 corrected finalizerCPUselftest442512 passed19checks, source4e7d1d0dc45999c90ba599e621ab81d91baa126df8ac4c6038e69605c007312b; explicit fixedcore4590/student8000/512000presentations, allfour models, strict28-layerKVschema,7newsuccessfulGPUroles, preservedfailedattempts and unchangedefficacy integergates. Sourcefreeze precedes V14profiles/outcomes.

V14 CPU release selftest442514 passed7checks. TrainerCPU442520passed; planSHAb0324838795ce132a785adb3ad4f489e17bb4c7e8451239a7b27f469d434afbf, source789d8cbbf3bcab1eed0360b1eb37b64ad0175803bda057065dbcd0204023b260. Recorded setup/logIO/finalvalidation walltimes are descriptive resource metadata; projectedformula unchanged. Both fixedphasegradients/nativecast/raggedpairs/carriedorders and unchangeddata/cache/priors bind. NoV14GPUwork oraccuracyyet.


V14 prospective scheduling clarification before any GPU profile or V14 accuracy: permit the two registered32core/32student profiles to overlap completion of the new independent fresh-test/report audit. Profiles consume only the unchanged V13training union, never fresh/dev/test images or native accuracy. Independent full-cache442437 plus current exactcache/source/trainCPU442520 already bind that unchanged input, and both root/hils reviewed the final live-gradient/phase runner. The new V14data/cache/report audit remains mandatory before the measured four-main release; no check, source freeze, computational inventory, timing formula, budget, fit schedule, scoring or decision is removed. This relaxes only the unnecessarily serial fresh-test-audit dependency to honor the user's parallelization request. All later audits and failed attempts remain preserved.


V14 initial profilearray442525 failed in botharms: centered44252645GPU-seconds, offset44252544GPU-seconds, total89. Both completed32core+32student updates and frozen-target precomputation, then conversion_diagnostics called the ordinary paired batch_states helper on sorted unique scene IDs; sequence_layout correctly rejected unequal adjacent target sequences. No registered native head replay, dev/test generation or V14efficacy completed. Preserve original789d8c...source in bothprofile andCPU snapshots, planb03248..., allpartialartifacts/logs and89GPU-seconds. Repair only the conversion input packer: gather the same unique scenes/features with an unpaired target_sequences/offsets layout, keeping all1782scenes/4266positions and same32-scene diagnostic batches. Ordinary paired training, target means, student updates, native inference, diagnostic estimands, call inventory, timing formula and scoring remain unchanged. Add a CPU ownership test with mixed target lengths/sequences and a partial batch, repeat trainerCPU/source freeze before either GPUprofile retry. Reporter/selected-source unions held until repaired trainer is frozen; no reporter checks yet submitted.


V14 conversion-only repair independentlyreviewed: productiontrainer05b1e802c718a25c8c19435c6f037573f8e0391711636261a16aa484a63772ad differs from original789d8c... only by unpaired conversion_states and its diagnosticcall. CPU442530failed10s in a new expected-value assertion that used nested Python-list tensor indexing; productiongather alreadyused explicittorch.tensor. Preservefailedsnapshot; correctedtests57a2529b1529210bd37966d2912296dc577fb821a2aa464464b9bc735193a933. CPU442540passed27s,12V14+11null+6raggedtests andpairhelper, planSHAf5a525409f2211c91c71af8b04399e0e6efbdac4b2777a3d8dde149bb2a79dd6. Full1782/4266 ownership, unequal targets/lengths, odd/single/final22-scene batches andmissingprefix rejection pass. GPUretry underthisnewplan is released with unchanged32+32/noaccuracy inventory andcaps; previous89GPU-seconds count toward16200.

Independent reporterCPUselftest442531passed2s. Dataaudit442536failed17s beforecache verification because the newreporter's literal context-count assertion was accidentally282 rather than272 during27-to28exclusion adaptation. The dataset/dryplan remain correct272 andunchanged. Correct onlythat independentassertion, inspect numericadaptations, andrefreshreporter/selectedsource unions against finaltest57a252... beforemains. PreservebotholdCPUresults. No threshold, model, dataset, accuracy ornativeexecution change.


V14 corrected profilearray442548 PASSED: centered44254955GPU-seconds andoffset44254855GPU-seconds. Both32core/32student updates, full972targetprecomputation,972/4266conversion diagnostics and10nativeprefills/10vision/20headcalls completed with bindinggatespassed. Maxsteadycore.011385391931980848s, student.0018260111100971699s, precompute.1627144308295101s, conversion1.4166493059601635s; CPUindependent mainrelease will rederive the registered projection from rawlogs. CurrentV14GPU199s=89failed+110successfulprofiles. No dev/test accuracy or mainfit released yet. Repairedtrainer05b1e802.../tests57a252... remainfrozen; originalfailedsource preserved.


V14 correctedreporter814769895133f1e2f6b0ef96b23cc7f442e84464833830e38e46f403113b48ca: CPUselftest442557passed2s, fullfreshdata/cache442558passed33s, all272contexts/136families/28exclusions and53322nativecachedvectors/972groups verified. SelectedsourceCPU442559passed7checks3s, exactfinalreporter/trainer/testunion frozen. CPUmainrelease442561passed33s; independent fullprofile/phase/target/conversion/head/source/data audits pass. Sharedphase measurements give1805.6937589022564s centered and1805.910962096299s offset (read exactrelease JSON for canonical precision), both<=2700. Alluser V14GPUallocation discovery includesfailed89+successful110=199priorGPU-seconds. Prior199+10800mains+300selected=11299<=16200. Fourregisteredmains are nowreleased under unchanged data/objective/core/student schedules/finalcheckpoint/scoring; noV14efficacy observed.

V14 fourmains completed successfully: centered18/offset18/centered19/offset19 jobs442567/442568/442569/442566 used729/729/727/724GPU-seconds, totalmain2909; prior199 gives3108beforeselectedaudit. Independent report442582 and final-checkpointCPU442581 run afterallmains; GPU442583 andfinalizer442584 remain success-gated. NoV14accuracy inspected at this entry. A separate prospective decisionmemo/source-only augmented-null comparison prototype is being prepared; no additionalexperiment orGPUwork released.


V14 independent report442582 passed: centered18/19 N32=109/136,97/136 andN64=41/136,43/136 versus offset31/136,50/136 and16/136,24/136. Both relative primary pass; both practical fail. Centered projected mean-error-squared/null-variance ratios1.756/1.996 and ordinary residual substitution79.399/83.202 show conversion fidelity remains unresolved. SelectedCPU442581 passed39s; GPU442583 failed3s before loading because submission supplied --plan to a wrapper already supplying it, causing argparse duplicateflag. No source, model, math or audit change. Retry unchanged wrapper with positional plan; preserve failed3GPU-seconds and cancel dependency-never-satisfied finalizer442584, then submit unchanged finalizer with successful retry path/dependency. No accepted objective or reasoning claim.


### V15 frozen native learned-versus-bank comparison — 2026-09-11, before diagnostic jobs or outputs

V14 independent accuracy report passes provenance but fails practical extrapolation: centered N64 41/136 and43/136, with training-bank conversion still functionally inaccurate. Register a bounded attribution diagnostic, not another fit or fresh efficacy campaign. Use all four fixed final V14 models (centered/offset seeds18/19, core4590/student8000) after selected-checkpoint and finalization pass. Select the lexicographically smallest canonical V14 fresh family_id separately for each K0..16 and include both its N32 andN64 contexts:34scenes/model, identical across models, identifiers fixed in the CPU plan before outputs. These contexts were already scored in V14; explicitly exploratory reuse.

For each model andscene independently generate two native trajectories, learned andbank. Both execute identical Nactual+24ordered question-specific K0 training reference occurrences+1global streams, computing the same actual messages, reference messages and compact predictor. Change ONLY the mean supplied to the original projected readout. Learned uses c(q); bank uses FP64 mean of current FP32 reference messages followed byFP32cast. Coefficient is actualN for centered and1 for offset; no anchor, scaling search, bank selection, parameter change or fitting. Preserve all24occurrences including duplicates and their exact training question/image provenance. Each mode follows its own global raw-argmax generated history after divergence; all streams share that mode's generated tokens. Max4 native greedy, EOS151645/151643, strict ASCIIinteger-plus-EOS with nonterminalspecialtokens invalid. Every trajectory stays in the denominator. No filtering or forced answer tokens. Report complete answers, K0..8/K9..15/K16 and K9..16, parse/completion/truncation/first-token and paired learned-to-bank transitions, allfourmodels. Compare augmented learned to saved original N+1 outputs only descriptively because batch kernels differ. No confirmatory thresholds or bootstrap claims for this17-family reuse diagnostic.

All272trajectories (4models*34scenes*2modes), at most1088model invocations and272vision prefills, must be retained with raw native logits, generated IDs, timing, model/source/data/layout/reference identities and strict raw-argmax checks. Both modes have matched work. A bank rescue supports a mean-source effect in these frozen native augmented executions; it cannot establish population calibration or attribute all of the effect specifically to predictor capacity rather than cached-versus-native reference states. No rescue means this finite-bank substitution is insufficient; it does not prove aggregation information absent. Final-norm writes still leave fixed-prefix nativeKV unchanged and reasoning composition remains untested.

Software first: separate controller/runtime/profiler, immutable prior source untouched. Seven CPU tests in Slurm check coefficients, zero-U identity, occurrence weighting, current/history reads, native-only global write, cleanup and unfiltered special IDs. GPU profile uses the two fixed earlier N16/N64 software cases and unfitted active/zero original cores plus a fixed deterministic unfitted predictor. For each case3barecached calls+3zero-U bank cached+2active mean modes*(3cached+2fullprefix)=16, total32fixedVLM/16vision. Four natural generations (bothmeans atbothN), max4tokens each, add<=16VLM/4vision: total<=48VLM/20vision/48standalone native-shaped head replays. Exactly18fixed cachedKV checks,26fixedfusionchecks,6zeroidentities,520cached/full row comparisons. Every existing layerKV, native hidden/mask/mRoPE/currenthistory and unchanged parameter/cleanup checks bind. Native captured norm/head replay TV<=.02 and exacttop1 bind; cached/full numerical differences remain descriptive and preserved. Arbitrary structural-special full-prefix replay is not claimed from normal forced-prefix checks; no failed observed replay may be silently repaired by removing tokens.

Resources: gpu partition B200, maximum4concurrent projectGPUs; eachsoftwareattempt<=300allocatedGPU-s, cumulative software (including failures)<=600; each of4studyjobs<=900GPU-s; V15campaign cap4200GPU-s including everyfailedallocation. CPUpreparation/validation/report/plot via cpuSlurm. Measured studyrelease requires load_seconds +1.25*(68*T64)+120<=900, where T64 is the maximum overbothN64 mean modes of preprocessing_seconds+generation_seconds*4/actual_generated_tokens; charge everyN32trajectory asN64. Independently bind native profile/source, final V14 models, allscene/reference provenance and rawtimings before fourparallelstudies. CPUrelease discovers alluser v15_ GPUallocations and requires actualprior+3600<=4200. No other GPUwork or training released by this protocol. Models stay in /mnt/ckpts/gabriele/gnn_transformer and data/rawtensors in /mnt/data/gabriele/gnn_transformer; lightweight reports in repository outputs. RESULTS.md untouched.


V14 selected retry442625 passed86GPU-seconds with unchanged source,46model/26vision/46headcalls,24cachedKVchecks/40fusionchecks and656cached/full comparisons. Allbindinggatespass; fourknown descriptive local numerical failures retained. Finalizer442626 passed10CPU-seconds, verifies3197GPU-seconds including allfailedattempts, finalfixedmodels andallcounts; bothrelativeprimaryTRUE, practicalFALSE, acceptedvisionmilestoneFALSE. FinalacceptanceSHAb9a53e39fc41fa05e90c17fd727e75b3943db9846802a5b1cad6f1429e68fd27. PNG/PDF generated fromexisting32metricbars; rootvisuallyinspectedPNG. V15may bindthisverifiedfailure, not success.


V15 prospective capture/offline attribution clarification before any V15job/output: retain capture=True for every study trajectory and its natural software timing, including current-query actual/reference/global states, FP32 messages, aggregate, predicted/bank/used means, query, projected correction, preactivation and residual. This adds no native forwards; measured projection must include capture overhead. Register one CPU-only4core/16G/5min descriptive decomposition after independent study rescoring: all272trajectories/alltheiractualgeneratedprefixes, no subset or recomputation of model answers. SourceQA semantics partition actual messages into positive/negative only offline. With a=N(centered) or1(offset), mu=bank mean, nu=used mean, report projected96-space vectors P=sum_positive W(m-mu), Z=sum_negative W(m-mu), B=(N-a)Wmu, E=aW(mu-nu). Their sum equals Wsum_actual(m)-aWnu; native q and projection bias are separate unchanged terms. K0 has an exactzero positive sum; do not invent a positive mean. N32/N64,K<=16 ensures negative frames exist. Retain vector Gram matrices/squarednorms and exactoccurrencecounts, actual-negative mean and covariance, bank/predictor errors, captured native residualnorms and reconstruction differences. Compute algebra from savedFP32tensors in FP64 and require synthetic/algebra identity rtol1e-10/atol1e-10; comparisons to captured nativeFP32 reduction order are descriptive, never overwrite source tensors or fit targets. Allmodel/N/K/prefix summaries includeeveryobservation. Each condition follows its own prefix; later values are not same-state causal comparisons after token divergence. Z is actual-negative residual relative to this finite bank, not identified population bias, image-position effect or sampling variance. No VLM/head/newfit/parameterchange, no accuracyoracle, no criterion or checkpoint change. CPUsource/algebra selftests freeze before execution; allraw/model/QA/report provenance binds. RESULTS.md untouched.


V15 CPUsoftware442638 passed23CPU-seconds, allsevencontroller tests and inheritedfixednative/referenceprovenance checks. Source83763c589e1a690440bf144d7dc203db6e4bf8076e63b4c0f55dc4a3361f4e07; planSHAdef2f7451ac855a81c9b3e523bc3eb4594b2a5c45c01e661d74a7699883cef84. Rootread-onlyreviewandexactsource/snapshotchecks passed. Registered300GPU-second softwareprofile is released with capture-enablednatural timings andunchangedinventory; no V15efficacyyet.


V15 native software442641 passed92GPU-seconds.40model/20vision/40headcalls,18fixedKV/26fixedfusion/6zero and520cached/full rows allpass; no binding or descriptive numerical failures. Fourcapture-enablednatural generations each2tokens. MeasuredN64maxbound6.528569211950526s, load13.666174425976351s; registered studyprojection688.594557441771s/model<900. Sources andCPUplan unchanged. NewindependentCPUstudyplan/source/reporter freeze stillrequired beforefourstudies; no V15efficacy yet.


V15 studyCPU442669 exited0 in0seconds without executing: source92be0e1e... definedmain but omitted the __main__ invocation. No tests, snapshot, plan, data preparation or release occurred; schedulerCOMPLETED is not a passed gate. Preserve exactoriginalsource undernull_comparison_study/no_execution_442669 andemptylog. Add only `if __name__=="__main__":main()`; repairedsource579ac554675aaeee58a970689dbff98f84b9cfbf2bb49306dbe12e403f349781. Repeat fullCPUcheck with unchangedsoftware/profile/data/scoring/diagnostic/caps. No V15studyGPU or outcomes yet.


V15 fullstudyCPU442678 failed34s after selftests/model/source/reference checks, before34input preparation, on a native API provenance schema mismatch. The imported V10 feature helper returns the same native owner/ropefunction plus two extra RMSNorm/bitsandbytes source bindings; the frozen softwareplan uses the original three-source native helper. PreserveCPU442678 source/logs. Change only that import to scripts.probe_native_vision_parallel_local.native_api, exactly the helper used by the softwareprofile, retaining strict whole-dict API equality. Training source/config hashes remain bound independently. No underlying installed source, numerical API, data, models, method or scoring changes. Newstudy source4ee71c4b4c6709e8d9b15158cb82ceadaea53c044921360aceef886c1bc05702; repeat fullCPUcheck.


V15 fullstudyCPU442691 passed105CPU-seconds under import-only repairedsource4ee71c4b4c6709e8d9b15158cb82ceadaea53c044921360aceef886c1bc05702. Independentexactdiffreviewpassed; allselftests,4finalV14modelidentities,54exactreferencebanks/1296occurrences,34identifier-selectedcontextQA/images/processor/mRoPE andallfrozenancestor/source hashes bind. PlanSHA37ffc4b9d69ba94487273eb5c07a730c0b1fda57ddadd6ba7d8e79cfeeb84512. Measuredprojection688.594557441771s/model<900; alluserrawSlurmledger prior92GPU-seconds,92+3600=3692<=4200. Rootexactsource/planchecks pass. Fourregistered parallelstudies nowreleased, no V15accuracyseen. OfflinegeometryCPU442702 passed2s with8algebra checks andexactcurrentstudy-sourceunion; noVLM/head/GPU.


V15 verified: fourstudies442708/442709/442710/442707 completed224/225/225/224GPU-seconds,898total; plus92software=990. Report442711 passed21CPU-s, all272trajectories/637nativecalls/272visioncalls andall637captures verified, maximum4GPUs. Centered18 learned→bank N32 16→16/17,N64 9→3/17; centered19 13→16and7→5. Offset18 4→4and2→2;offset19 6→6and2→2. All136augmentedlearned token sequences matchpriorN+1descriptively. This doesnot support a compact-mean substitution rescue atN64; projected-distillation proposal remains held. Geometry442712passed15CPU-s, all637prefixes, noVLM/head; maximumFP64vectorreconstruction9.09e-13, nativeFP32difference3.63e-4descriptive. Centeredfirstprefix Pnormnear1075/1155unchangedN32→64, Zgrows22.23→99.56/20.43→92.44, stronglyopposesP atN64(meanK>0cos≈−.93). Thisfiniteactual-negative/referencecontrast is not evidenceof a specificinputcause. Figure442713passed1CPU-s, existingmetricsonly, PNGvisuallyinspected. No acceptedobjective orreasoningclaim.


### V16 rendered Step-label counterfactual — 2026-09-11, before rendering or new inference

V15 does not support fixing only mean compression: exactbank substitution worsens centeredN64 native accuracy. First-query decomposition shows a growing coherent negative-frame residual relative to the training bank, but doesnotidentifycause. Register one narrow input intervention before another fit: within the exact34V15selected scenes, rerender every actual frame with displayed `Step ((i-1)%16)+1` for original one-based frame indexi. Keep character/room contents, physical frame order, exactquestion, originalQAbytes/physicalstep_id, N and frame-occurrence gold fixed. References remain byte-identical original24trainingoccurrences. This tests dependence on these changed rendered labels in frozen models. Repeated displayed ordinals differ from the original unique-ordinal grammar; it doesnot uniquely identify numeral-range effects or establish a general aggregation method. No cropping, newfit, capacity expansion, bank selection or countoracle at inference. A cyclic17..32control, nuisanceaugmentationfit and varianceobjective are held pending this result.

CPU stager uses immutable render_frame and sourceV15plan442691/report442711. For every unique originalimage, rerender originali and require exact originalRGBpixels (recordPNGbyteidentity); then render wrappedlabel. Require512x512RGB, exactpixels outside fixedbottomstrip y>=476, exact first16images, changed footer fori>16, unchangedQA/character-roomrecount/order/caseidentities andreferencehashes. All1632actualimageoccurrences retained:544first16unchanged and1088laterlabelchanges. Preserve originaldata; publish separate v16_step_wrap root with displayed-versus-physicalindices, source/wrappedPNGhashes, dimensions, pixel-differenceboundingboxes anddeduplicatedrenderlinks preserving occurrences. No fresh-test claim: these are counterfactuals of already observed families. Snapshot/testsource and deterministicrender/ownership checksbeforeallstaging, CPU4cores/16G/5minutes; data inuserprescribedroot.

FourfixedV14models, centered/offsetseeds18/19, same final4590core/8000studentcheckpoints. For each wrappedscene, independentlygenerate learned andbank using unchanged V15 nativecontroller/runtime, allN+25rows/bothmeanscomputed, originalN/1coefficient, noanchor. Same strict raw-argmax/max4/nativeEOS/scoring, allquerycaptures andeveryfailure retained. Compare against the frozenoriginalV15outputs aspaired rendered-label interventions atfixedN/model/mean. Report allK, K0..8/K9..15/K16/K9..16, wholeanswers, firsttokens, parse/EOS/truncation and original→wrapped exacttransitions. These comparisonsareexploratory, with no freshconfirmation, practicalacceptance orreasoningclaim.

Before wrappedinference, eachmodel executes twofixedoriginalsentinel scenes fromtheV15selection: K3N32 andK6N64, bothmeans, total4sentineltrajectories/model. Require generated IDs exactlyequal correspondingV15original IDs, then same-prefix full-vocabularyTV<=.02 andexacttop1 on everyrawlogit vector; no filtering or changing prefixes. Saveallsentinelrawcaptures/comparisons beforeenforcinggate; ifanyfail, no wrappedinference forthatmodel. These16originaltrajectories checkrerun drift separatelyfromthepixelintervention. Totalnewinference288trajectories=272wrapped+16sentinel, atmost1152modelcalls/288visionprefills;72trajectories/model,<=288modelcalls/72visionprefills. AllcaptureTrueasprofiled. No additionalnativeheadreplays needed: actualnativeoutputscomparetoexistingV15rawoutputs andunchanged runtimealreadypassed software442641. Existing40head/520cache-rowsoftwareproof and92GPU-seconds stay bound andchargedtoV15, notdouble-counted.

Resources: fourparallel gpu/B200 jobs, each<=900allocatedGPU-seconds, V16cap4200GPU-seconds including allfailed/zeroallocationattempts, maximum4projectGPUs. ReuseV15capture-enabledmeasuredT64=max6.528569211950526 andload13.666174425976351; registeredprojection load+1.25*72*T64+120 (allN32chargedN64) must<=900 before release. CPUstudyplan binds source/fullrendererproof/34preparedinputs/allfixedmodels/frozenoriginalreport/rawsentinels/nativeprofile andalluser v16_ Slurmallocationledger; requireprioractual+3600<=4200. CPUreport independentlyrescoredall288newtrajectories and paired originaldata, allcapture/sentinel/provenance/resourcechecks. Data/rawtensors under /mnt/data/gabriele/gnn_transformer; checkpointsread-onlyin /mnt/ckpts/gabriele/gnn_transformer. NoV16model fittingornewsoftwareGPU jobreleased. ExistingmetricsmayberenderedintoastandalonefigureinCPUjob withinputhashes andvisualinspection. RESULTS.md untouched.


V16 CPU staging442744 passed34CPU-seconds under independently reviewed source4aa662c250359534590a76c492c097777c4f5e97daba5b1814aa5874354f8eeb. Allfive executable checks passed. Exact34cases/1632occurrences:544first16unchanged and1088footerchanges;1272unique originalimages and701wrappedrender specs. ManifestSHAf28d86b497a85c72d39826eda01c532d3af8bb15db91dac8895c0c1be089b590. AlloriginalRGB rerenders, unchangednonfooterpixels, sourceQA/physicalorder/gold/referencebank hashes bind. NoGPU/modelcalls. Study-driver review andCPUplan gate still required.


V16 CPU studycheck442748 passed133CPU-seconds. Final driver407e9bbabe05850ce603ddc30a82a9764c5243f2b1fececc447a0ee393fac8d6 independently reviewed, including successful oneGPU/exit0 allocation evidence; no prior jobs under earlier driver. PlanSHA598ae80dd14fcd293e1b26a9dd4361efb582423835e821766bcd40247e9192a1 binds all34wrapped/2originalsentinel inputs, allfixedmodels, unchangednativeprocessor/mRoPE/reference inputs, alloriginaloutputs and exactstageproof. Allselftests passed. Projected721.2374035015237s/model<900; prior0+3600<=4200. Rootcurrent/snapshot sourcecheck andsamplevisualinspectionpassed. Fourparallel registeredstudies released; noV16accuracy inspected.


V16 prospective saved-state diagnostic, registered while four native jobs are initializing and before any wrapped outcomes are inspected: after a passed independent V16 report, one CPU4core/16G/5min job may compare all272wrapped/original first-query captures plus16originalsentinel controls. New source adapter binds allV15/V16reports, allfourfixedcheckpoints, original/wrappedQA and rawhashes, audits every saved prefix, and reuses immutable V15 FP64 P/Z/B/E decomposition and selftests. No new VLM/head calls or fitting. Retain signed vector changes and Gram/norm terms, not just differences of norms. For positive/negative actual occurrences retain W*delta_message and report sum/count, mean-shift norm and mean squared occurrence shift, split physicali<=16 versusi>16 as unchanged-image internal control. Empty-class sums zero; means undefined. Retain exact equality and max/RMS difference checks for global/query/reference messages/bank/predicted/used means. Verify delta_preactivation decomposition including query and chosen-mean changes in FP64 at1e-10; native FP32 residual/reconstruction discrepancies descriptive. Allmodel/N/mode/K records included, with all17K summaries; no correctness selection. Later generated histories are audited but do not enter first-prefix causal comparisons. This is an input-response decomposition, not identified mediation of accuracy or numeral-range-specific causality. Rawvectors under prescribed data root. New source frozen and independently reviewed before CPU execution; no extra GPU allocation released.


V16 infrastructure amendment before inference/outcomes: allfour array442749 allocations shared slurm-b200-8x-gpu-131-255 and stalled in startup. Root task442749 ended NODE_FAIL(1:0) after235s; child442750/442751/442752 were cancelled after285s each when the node failure was observed. Total1090allocatedGPU-seconds, all charged. Originalsource snapshots/logs/partialfolders preserved. No config, sentinel_gate, raw trajectory or prediction file was produced; logs empty. A lightweight process-status Slurmstep also failed to start and is not a separateGPU allocation. Retry will exclude that node. Explicitly amend only campaign cap4200→5400GPU-seconds to accommodate this external failure; retain900s/job, maximum4concurrentGPUs, exactdata/models/scoring/software/projection/sentinel and allcapture gates. Prior1090+3600=4690<=5400. Newsource differs only in4200budget literals, with prior407e9bb snapshot, independentdiffreview andfullCPUplan rerun before retry. No scientific criterion, selection, model or numerical tolerance changes.


V16 retry CPU442755 passed135CPU-seconds, sameprepared scientificinputs/tests. Newsource8f2e93f0fe80438abc41202f5ec09730b69973a5f967516b92de972168c6530b independently verified as exactlyfive4200→5400budget-literal replacements; previous407e9bb preserved. PlanSHAeaa21bc7c6825a12912bbc49471bc3ef54f483767a5b2a42039a305e747d57dc. Allfourfailedallocations/1090GPU-s included, reserve4690<=5400, unchangedprojection721.2374<=900. Rootcurrent/snapshot hashes bind. Healthy-node retry released with explicit exclusion131-255; no inference outcomes observed.


V16 healthy-node retry442762/442763/442764/442761 completed450/450/450/448GPU-seconds, total1798 plus failed1090=2888. All288 registeredtrajectories retained, all16originalsentinel gatespass with exacttokens/logits(TV0). IndependentCPUreport442774 runs afterallmains; reviewedCPUdecomposition442775 andplot442776 depend onits success. NoV16accuracy inspected atthisentry. Decomposition source d748922f3cd444e39c672cc85cceb334bf1aa378bcfb1f1e38dbdacca2c85bb0 /wrapper8a5f38ee63adc5ba1e20115ebe51445dc27edba93c596ad9b167e4b827d78926 independentlyreviewed; allprior sourcesimmutable.


V16 verified diagnostic: report442774 passed55CPU-s, all288trajectories/668nativecalls/288visioncalls and668captures. Centered18 original→wrapped N32learned16→13/bank16→17; N64learned9→9/bank3→14. Centered19 N32learned13→12/bank16→17; N64learned7→10/bank5→14. Offset18 N32both4→3,N64both2→3;offset19 N32both6→6,N64both2→3. All16sentinel tokens/rawlogits identical. Both wrapped centeredbank N64 firsttokens17/17, full14/17; exactremainingerrorsK12→11,K13→12,K16→15. Geometry442775passed64CPU-s: all288firstprefixpairs plus668current/637originalprefix audits; maxFP64deltaidentity7.50e-13. CenteredN64firstquery Znorm99.56→30.38/92.44→29.12, cos(P,Z)−.931→−.051/−.932→−.030; positive normchanges<1%, q/global/reference/bank/predicted/usedmean controls exact. Thisdecomposesobservedinputresponses, notlogitmediation. Figure442776passed2CPU-s andvisuallyinspected. Cost2888GPU-s includesfailed1090 andsuccessful1798. Strongcyclic-footer effect withbank; no unique numeral-range attribution, generalmethod, freshconfirmationorreasoningclaim. Noadditionalfitreleased.


### V17: native local-readout audit — registered before any V17 job or output

The audit was chosen before V16 outcomes were inspected. Keep its original canonical V15 scope: determine whether the frozen model already has reliable isolated 0/1 evidence in the exact local states used by the current aggregation system. This is a no-fit attribution audit, not aggregate answer generation, a new method, fresh confirmation, or a training release. The V16 label intervention motivates a simpler eventual per-item interface but does not change the selected audit cases.

Bind the immutable V10 cache and native profile442123, V15 independent report442711 and all its four fixed V14 models/cases/first-query captures, and the actual native FP16 norm/head identity. Score only original empty-answer-prefix local states. Later continuation logits are not relevance labels. “Other” means all vocabulary mass outside digit0/1, not a semantic unknown class. QA supplies offline truth/ownership only; GPU head computation consumes hidden states without labels.

Fixed inventory: eight original V10 profile replays (288 head rows);152 original local_empty training batches,148 of64 and four of10 (9,512 distinct image/question states); all272 V15 first-query batches with N+25 rows (19,856 head rows, of which13,056 are actual images). Total432 norm/head calls,29,656 rows,22,568 scored local rows. No backbone/VLM or vision calls. Training occurrence weights are24,624 across1,782 unique scenes and25,488 across1,836 epoch slots. V15 retains all1,632 physical scene/image occurrences in34 contexts across four models and two mean modes, with1,574 distinct image/question identities. Repeated executions are not independent observations. Do not substitute old V6 probabilities for the exact current states.

A CPU4core/16G/10min preparation job snapshots source, runs arithmetic/scoring/resource selftests without a head, verifies full upstream source/data/model/state/QA provenance, and writes432 hash-bound hidden-state input bundles plus a separate offline ownership/label artifact. Preserve original head batch shape [B,1,3584]. V15 bundles contain all actual rows,24 reference rows and the captured fused global row in original order. Actual native FP16 norm/head weights and installed normalization source must bind. GPU validation checks the exact CPU plan/source snapshot, consumed bundles and actual loaded weights; it need not repeatedly read unrelated upstream archives whose proof is already bound in the CPU plan. Independent CPU reporting verifies the upstream chain again.

One B200 GPU job is limited to600 allocated seconds; V17 total is limited to900 seconds including every failed or zero-allocation attempt, with at most one V17 GPU at a time and four project GPUs overall. Exclude the previously failed node131-255 for this job. The first eight profile replays must all have full-vocabulary FP64 softmax TV<=0.02 and identical raw argmax against archived native outputs. Record complete timing including input reads, transfers, probability checks, full raw output serialization and hashing. Before the remaining424 calls, require elapsed_to_gate +1.25*29,368*max(first8 seconds/row)+30<=600. Save all eight raw outputs and gate evidence before stopping on any numerical/timing failure; no fallback or relaxed threshold. Every one of the272 V15 fused-global replays uses the same TV/top1 gate. Retain every violation and all raw outputs; a numerical failure cannot yield an accepted audit.

Save every full-vocabulary output losslessly in native FP16 under /mnt/data/gabriele/gnn_transformer/v17_local_readout (about9GB), together with input hashes, row ownership and timing. No checkpoint is fitted. Independent CPU4core/16G/5min reporting recomputes FP64 full-vocabulary log normalizers, p0/p1/other, numeric mass, conditional p1, full raw top1, three-category correctness, confusion and calibration error. Conditional ties predict0. Report every identity/occurrence with no probability-based filtering. Use distinct-identity, unique-scene and epoch weights for training; separate all four models, both means, N32/N64, all K, semantic negative categories and Step<=16/>16 for V15. Within repeated identities, average per-observation metrics with equal total identity weight; do not average probabilities before classifying. Preserve state/prediction differences across executions.

No sum of isolated probabilities or decisions is presented as a native aggregate answer. Robust local evidence alongside failed canonical aggregate answers would support investigating the interface or decoder, without proving a particular gate or training recipe. Poor native local evidence would invalidate that premise for this readout, without proving information absent. First-query success cannot establish later-prefix relevance or reasoning composition. A future fit requires a separate prospective protocol. Models remain read-only in /mnt/ckpts/gabriele/gnn_transformer; reports remain in repository outputs. RESULTS.md is unchanged.


V17 source review passed before CPU submission: audit source114dd78b0588a3efe356db70c0e84a3033a2ec3b97a4ee7a4fa133a76d682bfa, GPU wrappera494be632361bd5968bf5c355f189a3477f1645dbed1b77777ae730bfc23ef13 with registered node exclusion and actual b200-8x-gpu feature. Explicit executable entrypoint, fixed inventory, native APIs, mass/weighting logic and retained-failure string handling reviewed. CPU preparation442786 submitted under the registered protocol. No V17 outputs or GPU job yet.


V17 CPU preparation442786 passed46CPU-seconds, all nine arithmetic/record/timing/accounting/weighting tests and full frozen source/state/QA/shape provenance. PlanSHA8c5ac67f41cc24043720896b62a28a404c79a3b66496eac6b68ef523a69d7a4d binds432 prepared head inputs/29,656 rows; root verifies current/snapshot sources and zero prior V17GPU cost. The registered600-second head-only job442787 is released with its first-eight native replay/timing gate. No V17local-readout outcomes have been inspected.


V17 timing-model amendment before any new local-state scores: GPU442787 stopped after the eight original profile calls, using15 allocated GPU-seconds. All288 native reference rows pass TV<=.02 and exacttop1 (maximumTV2.09e-9). The original per-row timing gate fails at1972.6279s because the first8-row call costs0.4211255s, whereas the other8-row calls cost0.02936–0.05055s and64-row calls0.12816–0.14763s. No training or V15 audit state was scored; all8raw outputs, oldsource114dd78 and failedgate remain preserved. This is a failed runtime estimate, not a numerical or efficacy failure.

Register a new conservative affine timing model before retry: c=max complete seconds over ALL eight pilot calls, s=max seconds/row over ALL four largest (B64) pilot calls. Project elapsed_to_gate +1.25*(424*c+29,368*s)+30<=600. Charge the largest observed whole-call cost, including the cold call, for EVERY remaining call, plus the measured large-batch row cost. Require this affine model to upper-bound every observed pilot call; retain all measurements. This separates fixed overhead from row processing without omitting the cold observation or relaxing the numerical/resource limits. On the preservedpilot it projects348.45s approximately (exact calculation recorded separately); newretry must recompute its own gate. Source changes are limited to timing-policy metadata, the projection/selftest and its explanatory documentation, with independentdiffreview andfullCPUplan rerun. Inputs, probability calculation, batch shapes,call inventory and numerical thresholds unchanged: eachsuccessfulaudit remains432calls/29,656rows; with the preserved8calls, a successfulretry would bring campaignheadcalls to440. No VLM/vision calls. Keep600seconds/job,900campaignincludingfailed15; reserve15+600=615<=900. No newfit or gate-choice tuning is released.

V17 timing amendment exact arithmetic clarification: the affine projection on preservedpilot442787 is348.0671877012355seconds (c=0.4211254920810461seconds/call, s=0.002306792710442096seconds/row, elapsed=10.188316497951746), superseding the preceding rough348.45notation. Allobservedpilotcalls lie below the registered affine envelope. This clarification changes no gate or resource limit.


V17 timing-only source amendment independently reviewed by root: newscript00d378f6f39c87c5699d67acd678a52e1d971b0023571bf582407b7eec54a578, doc2712d3eea23e7e522682fb70c0df4fe7cb83689b90b793f7f00cc75bcb1da340. Exactdiff changes only timing POLICY metadata, timing_projection, its synthetic tests and documentation; all other Python AST nodes and every wrapper are unchanged. Originalsource114dd/doc/wrappers/failure/rawbindings are preserved under timing_failure_442787. Repeat fullCPUpreparation now; no local-readout outcomes observed.


V17 CPUretry442792 passed43CPU-seconds and all11selftests. PlanSHA3e10ef42ade680f4ff59ab77ca11af8031085f429aa8eb0822dcb2a43ea8e3d4. Root verifies all432 prepared archive hashes and scientific call metadata match442786 exactly; only source/timing metadata and directory paths change. Priorfailed15GPU-seconds included, reserve615<=900. RetryGPU442796 released under unchanged600-second cap and numerical gates; its own affine timing gate still required. No newlocal outcomes inspected.


V17 independently verified: GPU442796 completed60GPU-seconds; CPUreport442798 passed44.02scriptseconds. AnalysisSHA c59a178d6192623d6c277e8d32cfc51a9fad8f6782c8471ad82fd00f411b06cb; reportsummarySHA1c780d874707e9a18bb3b9969f97083f6d5fb76b260ae82580c4461128c15836. All432successfulheadcalls/29656rows,22568scoredlocals and560nativebindings reconcile; maximumTV2.08776209e-9, exacttop1 throughout, actualhead/norm weights unchanged. Retry own affine projection347.2037975266867<=600. All9512trainingidentities correct (864positive,8648negative), all24624unique-scene and25488epochoccurrences correct. All13056V15executionrows correct, representing1632physicaloccurrences/1574distinctidentities/226sharedtraining. Every model/mode/N/K/Step/negativecategory slice retained and correct. Minimum numericmass .997079training/.997637V15; every negative gap gate is exactly0 and every positive nonzero. Total75GPU-seconds includesfailed15; total440headcalls includesold8; zeroVLM/vision/no fit. This supports the prospectively fixed native gate but is neither aggregate accuracy nor a fresh efficacy/reasoning result. Resultnote and source/provenance independently reviewed. RESULTS.md untouched.


### V18: matched native semantic-gate experiment — prospective preparation

V17 verifies the premise for the gate rule fixed in the semantic-gate proposal before those outputs: original empty-prefix native g=max(0,p1-p0), with full-vocabulary FP64 normalization and detached FP32 storage. All coordinates of the rank96 tanh payload are gated before SUM and the existing learned native SiLU residual readout. The native_gate versus all_open comparison keeps1041600FP32parameters, zero-U start, canonical images, originalN1local/globalprompts, ordinaryEOS generation and identical probe/payload work. Every trajectory computes one extra native norm/head/probability call at prefill and reuses that original gate for later tokens; no additional VLM/vision call, referencebank, mean predictor, scalar tally or teacher replica. This is a controlled semantic-selection baseline, not a novel attention claim or general adaptive reasoning solution.

Prepare fresh test data now, without fitting: seed20261118, exactly272complete contexts in136families, eightfamilies per K0..16, eachN32extendedtoN64 by the unchanged canonical negative-insertion law. Preserve semantic parent order while rerendering original physicalStep labels; no cyclic-label intervention or augmentation. Exclude all30canonical prior manifests: the immutableV14prior28, completedV14fresh andV16wrap (whose semantic contents are already inV14). Exclusion is at completecontext/question level, not independent visualatoms. LiteralV10training/dev/schedule/pairing remain unchanged. Allnewcontexts nonsaturated, no overlap exceptions. CPU4cores16G10min drycheck freezes exactsamples, allsource/exclusion/protectedhashes, deterministic regeneration and collision rejection before any rendering. A secondCPUjob withsameboundsource/plan renders exact512RGB originals, runs independentQA/count/parent/imagebyte/link audits, verifies all272publishedrecords equal frozenplan and retains any partialfailure. Data root /mnt/data/gabriele/gnn_transformer/v18_fresh; repository reports outputs/native_aggregation_vlm/v18/data_staging. NoGPU/model/headcalls or fit release. Newstager independently diff-reviewed againstV14; frozenhashes below. AfterpassedCPUdrycheck, unchangedsource rendering is authorized without a newscientificchoice.

{"scripts/stage_native_vision_v18_test.py": "298f0aabc3431cc51c9d3dad2195eac51c54d38d288f2474e2590476ae588585", "slurm/native_vision_v18_test_stage.sbatch": "100ed01483f2d2de645e17fb7750046bb266d0b3a74c9e4e2412b85e4db62541"}


V18 fresh CPUdrycheck442827 passed deterministic generation, complete30manifest exclusions, exact272contexts/136families and collision-rejection tests. PlanSHA08faed89d9e28569c87027779d473237f040173565396ebf12b658f6f3df31dc, seed20261118. Root verifies every current/snapshot sourcehash and frozen specification; unchangedsource rendering is released under the registered CPU4/16G10min limit. No modelcalls/fits.


V18 native software profile is registered before execution. New core, controller and runtime independently source-reviewed; no fit or accuracy scoring. CPU4cores/16G/300s runs11core and9controller tests, verifies completeV17source/data/head ancestry, old native backend and model/processor/API identity, prepares the unchanged oldN16K3/N64K6cases with fixed two-token software continuation ('Therefore',':'), and saves only consumed input/initial tensors. PrivateCPU seed20261118 initializes zero-U semantic core and an active version with U Normal(0,.001); all other parameters identical, no learning. GPU requires the passed exactCPUplan, actualB200/FP16head+norm identities and source/consumedtensorhashes. It avoids repeated ancestor tensor reads on GPU.

Per fixed case: barecached3, zero-U semanticcached3, activeall-opencached3 plus two full-prefix replays, activesemanticcached3 plus two full-prefix replays. Total32VLM/16vision,26fusions,18all-layerfixed-historyKV comparisons,6exactzero-Uidentities. Each of the six origin trajectories probes the complete original [N+1,1,3584] query once; all246originheadrows must match exact barehidden/normalized states and native rawlogitTV<=.02 with exacttop1. Both active modes must have identical actualprobe logits/probabilities/gates before applying the all-open control. Full-prefix replays use a fresh controller bound to the exported originalartifact, with zero new probes. All328cached/fullrowcomparisons retained as descriptive under the prior explicit native kernel limitation; they cannot redefine the hard same-forward/probe gates. Preserve original failed mixed-cache evidence. Local/pastnonquery states and every fixed-history layerKV stay native; same-device core algebra is exact.

Then four natural trajectories (N16/N64 x bothmodes), rawglobalargmax/max4/nativeEOS, one vision prefill each, no masks/localtokens, allorigin/query captures enabled. At most16additionalVLMcalls and4extraoriginprobes. Total<=48VLM/20vision, exactly10originalnorm/head/probabilityprobes, and one standalone native-shaped captured-state norm/head replay perVLMcall (<=48), for <=58extraheadcalls. All replaynativeglobalvectors must passTV<=.02/exacttop1. Timings include runtimecapture/origin export; retain preprocessing and scale measuredwholegeneration*4/T pluspreprocessing for a four-token bound. No efficacy score or reasoning claim. EachGPUjob<=300allocatedseconds, software reserve<=900including failed/zero attempts, counted within prospectiveV18campaigncap14400; maximumone softwareGPU/fourprojectGPUs, failednode131-255excluded. Preserveallfailures before stopping; no silent threshold/inventorychange. Sourcehashes:

{"gnnformer/parallel_local_semantic_aggregation.py": "e51ab3859812b4d9c6680f2343883e4e3a00498df99fba74e0591d2bc3994f5a", "gnnformer/parallel_local_semantic_gate.py": "d115044a2f21dea777001838fc3bb3dc64282ea819e2ed048f0626b3584b73a0", "scripts/native_vision_semantic_gate_runtime.py": "9f934bf3c98a8a7fd1d7a94a90cb6d476133b0d222048bb39c4fd6b742efc074", "scripts/profile_native_vision_v18_semantic_gate.py": "99a4a8e0cb95d3b681034fa9a1c2f1a0ebc9e8202bb6bddeb2ed570ac73c88a8", "slurm/native_vision_v18_semantic_gate_check.sbatch": "05f2526aee9269e6bdb70af0eb05c8c35783c40876dc13fff309afc637f81799", "slurm/native_vision_v18_semantic_gate_profile.sbatch": "db235ccf86ab65f24dbd9808d7bf41d423646bb727f83fe83b1652343182a598", "tests/test_parallel_local_semantic_aggregation.py": "eb1bc2b6b9120b23ad7b6246e2c70a6d9458c660c95c4cb43c205d6fd39d2170", "tests/test_parallel_local_semantic_gate.py": "cf460c5c363c1e7643d96fe69b2c385e33152d4826d1d05ff7f9a40f070d04e3"}


V18 CPU softwarecheck442830 passed33allocatedCPU-seconds, all20core/controller tests and exactoldN16/N64preparedlayout/origin/nativeidentity proof. PlanSHA45921222241820700a8e3cbd6637c819f331c4c9f27348b8896e1a233655b8aa. Root verifies everycurrent/snapshot source binding and20testcounts; GPU300-second nativeprofile is released with registered48VLM/20vision/10origin/48replay maxima. V18fresh staging442828 independentlypassed38CPU-seconds, all272contexts/13056imageoccurrences/3288unique canonicalrenders and30prior exclusions; manifestSHAb153b1dfea5c11d4fd92b1109ffd235898a32559d3db45aeb4b20b768a164abe. No training or evaluation outcome exists.


V18 gate cache is registered for CPU-only preparation before fitting. Reuse every one of the152alreadycomputed V17trainingheadbatches (9512distinctoriginalempty-prefiximage/questionstates); no newhead/model/visioncalls. CPU4cores16G300s runs7precision/origin/padding/fullmass tests, verifies frozenV17report andactualFP16head/hidden/rawlogit provenance, recomputesfull-vocabularyFP64probabilities fromalloriginalFP16outputs and requiresagreementwiththeindependentreport. Applyonlythefixedg=max(0,p1-p0) then detachedFP32; no gold, totalcount, prefixlength or alternativegate affects itsvalue. PreserveFP64p0/p1 and allnativeweight/input/hash metadata. Key by original local_emptyfeatureID; verify all32686localprefixfeatures mapto their own emptyfeature andall1782training scenes preserveimage/orderoccurrences. Attrainingbroadcastsamegateovereveryvalidtargetposition; actualall-openrows1afterthesamegategather, paddedrows0bothmodes. Save canonical /mnt/data/gabriele/gnn_transformer/v18_semantic_gates/{feature_cache.json,gates.pt}, proof andsource snapshots; nevermodifyparentV10cache. The gate nativeidentity mustequal the separately frozenV18softwareidentity. Current readout batches are the originalcachedfeature route, not new deployedN+1 execution; software/head/cachedtrainingprofiles keep that distinction. Anyfailure isretained; no standalonetest rerun isneededifembedded7testspass. Root independentlyreviewed source/schema/precision/ownership andentrypoint; hashes:

{"scripts/stage_native_vision_v18_gates.py": "33b9556b8ead0e7811a91c5a364410e99c007d70757dc4f439f9aefc174b6406", "slurm/native_vision_v18_gate_stage.sbatch": "a584c2cab248f958e6b7c3bf05469ee91c972ff9031eb2aa412574446f76ed42", "tests/test_native_vision_v18_gates.py": "27ab86de211698b174af08fb05ea9ab927abf31ff005d77e069f5ba1e42cb1af"}


V18 native software442831 completed74GPU-seconds. All40actualVLMcalls/20vision,10originprobes and40standaloneheadreplays verified; all246originrows and40nativeglobalreplays haveTV0/exacttop1. All18x28layerKV,26fusion,6zero-U and8naturalquery checks pass. All328cached/fullrows retained; exactlytwo descriptive violations (sameactualrow62,N64,t2,bothmodes,TV.02029646357,top1equal). Gates are reused exactly across cached/fullprefix calls. Software summarySHA212567976c10b4c998b8368411a9694ffa66e410db03860f860a653c264510b1. Pooled four-token bounds T16=1.3968497309833765s,T64=5.163364219479263s. Gate CPU442832 passed30allocatedseconds and7tests; all9512originstates/32686localprefixes/1782scenes bind, maxreportprobabilitydifference0. GatecacheSHA1f9259a292db503f89d5cec793318afa7a27aa019446977ecb0021e22c3941bc; proofSHA284d582cb69e76c42f475e56ab241f62b8be77bba4656543c2566eb0ca19adda. Cache andsoftware nativeidentity agree at e1d1ea91ed83c6fa57e2a991e1bcfb485007eb98895a0fa217871212dc9fb0f8. Independent source/JSON/provenance review passed. No efficacy established.

V18 matched training and evaluation protocol is fixed before any training profile or main outcome. Conditions native_gate and all_open; seeds20/21; same privateinitialization/persistentpairedorder per seed;1041600FP32parameters, rank96 tanhpayload, SUM/SiLUreadout, zeroU. Frozen native NF4Qwen2.5VL7B with actualFP16norm/head, canonicalV10images/features/918pairs/1782unique scenes, supportedK0..16, maxtrainingN16. Fortyepochs,4590updates,16scenes/8intactpairs perbatch,73440scenepresentations,36720pairpresentations,177120validnativetargetpositions. Preserve54sameSIDK16pairs perepoch, exactzero consistency. Both objectives equal per-scene mean complete numeral+EOS CE plus per-pair correspondingprefix residual consistency coefficient1; fixed frozen-global squared-norm denominator+1e-6. Path statistic isdiagnostic only. AdamW lr.001,50warmup,cosine to1e-5,weightdecay0,globalclip1. No gategradient, auxiliaryteacher, nullpredictor, reference stream, augmentation or directscalar answer. Original cached gate repeated at everytargetprefix, paddedimagesgate0; botharmsgatheridenticalcachedgates beforeall-openreplacement.

Endpoint isfixedfinal4590, with checkpointsave/reset/reload and exact before/afterevaluation parameter tables bindingactualdeployedweights. Onefinal64N16development sweep isdescriptive and cannot selectepochs. Then evaluateall272freshV18N32/N64contexts; N32 andN64each136, eight perK0..16. Everytrajectory usescanonicalimagesand fullnativeglobalrawargmax, four-token budget,nativeEOS151645/151643 and no masks/penalties; exact requires cleanASCIIintegerbody andterminalEOS, retainallinvalid/truncatedoutputs. Save allrawglobalFP32losslessFP16logits, originalfullnativeFP16probeartifact andallquerycaptures; independently verifyoriginalgate/fullmass/detachment/closedcoordinates/prefixownership andactualnormal/directprobe counters. Everytrajectory hasonevisionprefill,Tnativeforwards,Tnormalheads andexactlyoneextraoriginalnorm/head/probabilitycall, evenallopen. No reasoningclaim.

Primary criterion: in BOTHseeds, native_gate N64 gainsatleast7/136overmatchedall_open, withN32loss atmost6/136. Practicalcriterion in BOTHgatefits: N32>=123/136,N64>=109/136,N64K9..16>=52/64. Report allN/K andK0..8/K9..15/K16/K9..16, wholeanswer/firsttoken/parse/EOS/truncation, pairedfamilydifferences and10000bootstrapdraws(seed20261118) stratifiedbyK withsharedfamilydrawsacrossN/modes/fixedseeds. Thismeasureslength extrapolation withtrainedanswersupport; no unseenanswer/generalreasoning/newattention claim. These criteria definea visionmilestone, notthecomplete researchobjective.

CPUtrainingcheck4cores16G600s requirespassedsoftware,gatecacheandfreshstaging, runsnewtrainingintegrationtests andboundhelperchecks, validates everyactualpair/globalprefix/gatepadding andfreezes allsources/artifacts/orders. Then two parallelGPUtrainingprofiles, eachseed20/mode,32updates,<=300allocatedseconds, no dev/test scoring. Eachprofile usesold fixedN16K0/9/10/16 examples (10strictprefixpositions),10VLM/10vision,4originalprobesreusedforcontinuations,10complete[N+1,1,H]capturedstatereplayheads and10cachedglobalprojections. All170same-state nativeheadrowsmustpassTV<=.02/exacttop1; allcachedcomparisondifferencesdescriptive/preserved. Zero-U gradient, gateisolation, frozenbackbone andcheckpointrestore proofs required.

Fourmainjobs stay held untilbothprofiles andtheindependentreporter/data/source/releasechecks pass. Resourceprojectionfixednow: setup_before_training +1.25*(sumfirst4profileupdateseconds +(4590-4)*maxsteps5..32 +64*T16 +272*T64)+120 <=2700seconds/mode; usepooledworstmodeT16/T64frompassednative software. CPUrelease4cores16G600s independentlyauditsprofiles, exactdata/source/nativeidentity andallSlurmfailed/zeroallocations; reserve4*2700plusallpriorV18GPUcost<=14400campaign. Eachmain<=2700allocatedseconds, maximumfourprojectGPUs; softwareonlyreserve<=900,failednode131-255excluded. CompleteindependentreportCPU4/16G600s mustrescoreallfourfinalendpoints. RESULTS.md unchanged. Trainer/source/tests independently reviewed beforeCPUrelease; hashes:

{"scripts/train_native_vision_v18.py": "8d39ad9cd98cfb543e9318b3e09a4b12e753c289766aa2087c05b6974fcaa6f4", "slurm/native_vision_v18_train.sbatch": "a01e8cfdc78b558753c6d20cc6cf5a6eaa65a75b75fb519361109b4ae69ddfda", "slurm/native_vision_v18_train_check.sbatch": "31a538f8a4ef79535f3a286f993baeef7f15d77e93f119fe03f2906541c5e29c", "slurm/native_vision_v18_train_profile.sbatch": "74a07d1892129731629924ac68e7cb3eddc12f2d19905a0dc6e92dd5d33b62df", "tests/test_native_vision_v18_training.py": "7255db207ef454c9518aa07800ef2193bfa4b11f96ced8908f06e9d5441fbcf5"}


V18 CPUtrainingcheck442845 passed35allocatedCPU-seconds: allregisteredhelpertests,918actualpairedqueries andallorigin/prefix/paddingchecks, completefrozenfeature/gate/fresh/nativeidentity provenance. PlanSHA1016ce503265ee7eb2f1cdf08d792b6cfaa9960b60e4364f8b0fab6dc4e4bbbc. Root verifies allcurrent/code sourcehashes, exactnativeidentity andbothindependentseedorders. The two300-second seed20trainingprofiles are released inparallel, native_gate/all_open, under the fixed32step/10VLM/10vision/4probe/20head inventory. Mains remainheld for independentdata/report/source andmeasuredresource release; priorV18GPUcost74seconds.


V18 both training profiles completed: native_gate_s20 job44284942GPU-seconds and all_open_s20 job44284841GPU-seconds. All32updates,10native/visionforwards,4originalprobes and20standaloneheads perprofile pass; all340same-state headrows have TV0/exacttop1. Cached-versus-live originalgate maximum difference .0019540786743164062 is descriptive. Gradient isolation and checkpoint binding pass. PriorV18GPUcost157seconds. No development/test outcome exists.

Independent reporter and main-release sources are prospectively frozen after independent reviews. Before numerical execution, original full-vocabulary FP64 probability replay tolerance is rtol1e-12/atol1e-15; retain actual maximum error. Exact saved nativegate reconstruction uses saved FP64 probabilities followed by FP32 cast. FP32 saved aggregate versus FP64 sum of saved gated messages is descriptive, with its discrepancy recorded; neither relaxes native head replay TV<=.02/exacttop1. Reporter --check-data runs its six meaningful selftest groups followed by full fresh-data/cache/gate audit on CPU4cores16G600s. Root release then runs eight resource/ledger selftests and independently rechecks source, data, profiles and registered timing formula before releasing any main fit. Source hashes:

{"scripts/prepare_native_vision_v18_release.py": "db276888cf0f30854253a99107f4f2c05e13c7f2d869698689d0ff78fb4e29ca", "scripts/report_native_vision_v18.py": "29c7fce8870a4a38d5dfc88e77a7a54682fa16967c95d4128a10615affc64280", "slurm/native_vision_v18_release.sbatch": "5d36120b0dceff394aa0a9cbe1f49645ac9346229f59e2810f4ac3627cdd34a7", "slurm/native_vision_v18_report.sbatch": "6109bd4ddbdb7fe5a055385c1eaa4e9a6f767642ccfbb647418a7418cbec0917"}


V18 independent complete data audit442877 passed38CPU-seconds, six selftest groups and147sourcebindings; auditSHA77cda98e3c6320bcd0ffd51431a06ec44cd221e238fbf5e42c23e72bbd3b2752. Independent main release442888 passed51CPU-seconds, eight resource tests, all training/native profile replay, exact final-state/source/cache/data/native identities. Four final-only mainfits are now released, conditionsnative_gate/all_open andseeds20/21, maximumfourparallelGPUs and2700seconds/allocation. Measured projections {"all_open": 2108.771396338707, "native_gate": 2112.4107485709246} seconds; priorallocated157 andreserved10800 yield10957<=14400campaign. ExactreleaseSHAc0c941ac71bcc4c5a3674701f63728f19e05a50d35f30b9044d001b323fed7cb. No development or test outcomes have been observed before release.


V18 mainarray442892 is evaluating allfourfixedfinalendpoints. Independent frozenreport442913 is queued with afterok dependency on thewholearray, usingexactfourrundirectories. Optional CPU2cores4G300s figures are released only afterthatpassedreport; they copy existingwhole-answer/first-token/per-K metrics withallfourmodels andfixeddenominators, no new scoring/statistics/decisions. Sixsmall consistency selftests run in the plotting job. Plot source reviewed before outcomes; hashes {"scripts/plot_native_vision_v18.py": "4d759880c5a5839a80110d478d45ea2cab8524571d4b55e77bdb0c582d19ace8", "slurm/native_vision_v18_plot.sbatch": "a5bd0e2539215174e75865f5074ea6a647ff16b0815ea8ecb303c79dac865936"}.


### Cosmos original local gate audit: prospective release

Source-reviewed four-prefill audit, independent of V18 efficacy. Compare reasoning-local versus canonical direct-N1-local prompts with an identical official reasoning global prompt, on the old N16/K3 and N64/K6 software scenes only. Exactly four bare VLM/vision prefills, four ordinary native heads and four same-shaped replay heads; all164 native rows replay-bound, all160 local rows reported. No branch, fit, generated token or global-answer scoring. New actual Cosmos FP16 norm/head/source/mirror/processor identities, native SDPA/NF4 backend, ordinary mixed-processor parity and exact positions/masks are mandatory. No Qwen gates/features are reused.

Gate rule stays max(0,p1-p0), FP64 full-vocabulary normalization then FP32; CPU replay probabilities use rtol1e-12/atol1e-15, while actual saved gates must equal the saved probability gap exactly. Native same-state head replay requires TV<=.02/exacttop1 for every row. Replayed-normalization equality is independently recomputed and descriptive. Report all native argmax, binary signs, digit mass, positive attenuation and negative leakage by length, Step and semantic negative category. Fixed direct-local feasibility requires every positive gate>0, every negative gate=0, and minimum digit mass>=.95; reasoning-local descriptive. Passing this old-example prerequisite establishes no aggregate or reasoning efficacy.

CPU check/report each4cores16G300seconds. GPU oneB200180seconds maximum, campaign540GPU-seconds including every failed/zero allocation. The GPU script checks allfourV18main allocations terminal, then reserves180seconds plus allprior auditGPUcost<=540 before model load; maximum one auditGPU and four projectGPUs. CPU preparation runs seven meaningful probability, boundary, denominator and allocation test groups. Preserve exact source and all available failed artifacts. Design: docs/paper/NATIVE_AGGREGATION_REASONING_GATE_AUDIT.md. Frozen source hashes {"scripts/probe_reasoning_semantic_gate.py": "e6e92026a43a297b41e971f5d0222f374902b17518e9827b4ebacfa750073560", "slurm/reasoning_semantic_gate_check.sbatch": "ffa347c72ee1a920fec05c1c2467a3e19a66131b795ea58935bc223813c16758", "slurm/reasoning_semantic_gate_report.sbatch": "08cf1d991b86a3a59908a06f3a6add44d203259609517a3b343377f4e911864e", "slurm/reasoning_semantic_gate_run.sbatch": "a44fd0d3df29283cd598722e43e5b6bcfeb66c6cc8a7aa41f6edda03768ce6a8"}.


Cosmos localgate CPUcheck442924 passed14allocatedCPU-seconds, all seven registered selftest groups, exactmixedprocessor/positions/layout and nativeactualsource/runtime/mirror preparation. Rootverifiedallcurrent/source-snapshotbindings; planSHAcf142722644d6bd1d384790aec9391a45f7de34457822c27ac2a71f1347fe0d6. AllfourV18main allocations completed successfully; the unchanged four-prefill180-second GPU audit is released, followed by the frozen CPUreport. No V18 efficacy result is yet available from the independent report.


V18 independently reported main outcome: both seeds132/136N32 and131/136N64 native-gate whole answers,59/64N64K9..16; all-open bothseeds0/136atbothlengths. Allfour64/64N16dev. Both primary/practical criteria pass. Independent report442913 analysisSHAbe875fdd02389847630c8927c594eb37d2c229697f1e344e090bbc7d598588be, summarySHA7baff4cc1d4d304fe2ca403ed9b084c89860fe958508413643c8e99e9f6b1837. Allfourmain allocations878/879/876/891seconds, totalcampaign3681GPU-seconds. Distinct saved checkpoint hashes and before/after deployedparameter tables verified. Native-gate all272firstnumeraltokens permodel correct; remainingerrors are valid multi-digit undercounts, notEOS failures. All-open cached/devfit succeeds but freshlength outputs fail in rawnumeral/content/termination, allretained. This is a native counting length-extrapolation milestone withtrainedanswer support, notgeneralvectoraggregation orreasoningcomposition.

The saved-state diagnosis designed and held before reading the outcomes is now released after independent report verification and independent source review. CPU4cores16G600seconds, all1088scenes/544pairedfamilies, t0P/Z andWagg projections, nativegateL-A identity, pairedsemanticgate/payload decomposition with explicit storedFP32 multiplicationroundoff reconciliation, unchanged firsttoken/fullanswer/targetcompatibleprefix scoring. Saveunfused/fusedoriginalhead comparisons fromalreadyretainedvectors, withoutheadcalls/probabilityrecalculation/fit/intervention. FP64identitiesrtol1e-10atol1e-10; nativeFP32differencesdescriptive. Sixembeddedtests, unchangedprimary/practicaldecisions, allfailures retained. Source hashes {"scripts/analyze_native_vision_v18.py": "34e8e0b132e051a1cd4c5723801305bed117176a3c3e64f28421e0f840767209", "slurm/native_vision_v18_diagnosis.sbatch": "f8e020ad86131ba8be453c51ff4f538ad837adc3ef6b413bb4822e9ebff0912a"}.


Cosmos original localgate GPU442925 passed33allocatedGPU-seconds, fourVLM/fourvision/fourordinary+fourreplayheads; all164same-state nativeheadrowsTV0/exacttop1. IndependentCPUreport442926 passed3seconds, analysisSHA4d0bfbb9b414d48f5062a3111172be8b88c171b3adfbff569f9aa7c2372dfd79. Direct-local80/80native andbinarycorrect,9positivegates0.997967422..0.999605596,71negativegatesexact0,minimumnumericmass0.999998809; directfeasibilitypasses. Reasoning-local0/80nativecorrect,numericmassmean8.3139e-17,all71negativegapsgreaterthanzero. Fullprobabilitymaxroundoff7.105427357601002e-15. No fitting, generated/globalanswer accuracy orreasoningcomposition. The nextsource-only boundedstreamingdesign is docs/paper/NATIVE_AGGREGATION_REASONING_SEMANTIC_STREAM.md; noGPUstreamingreleaseyet.


V18 saved-state diagnosis442927 passed38allocatedCPU-seconds, all1088scenes/544families andsixembeddedtests. AnalysisSHAb25d365119879a5cdecfa19325df85d67441637d6e9e5b084a10c53c158767f5. Selection/partition/product/roundoff identities have0maxerror; pairedquery/parent/added-negative reconstructionmax1.9184653865522705e-13. No nativefalsepositive/falsenegative gate acrossfreshdata, gatednegativeZexact0. Queryexactacrossallpairedlengths; positivepayloaddriftmean~5.5e-5, projectedgatevariationdominateswithin-modelpairedmessagechange. These aredescriptivestatistics, notproofthatconfidencecausesremainingerrorsorvectorvaluesarenecessary. Plots442918 passed3CPU-seconds andsixconsistencychecks; bothPNGfiguresvisuallyinspected, PDF/PNGhashesverified. Resultnote docs/paper/NATIVE_AGGREGATION_VISION_V18_RESULTS.md SHA76802ca3164177d6b9bbfb1714c5f9572ec5fae7d9f8bfa09e2e9dbb0c15252a. RESULTS.md remainsuntouched.


### V18 exploratory positive-payload description after the verified result

The all-case diagnosis showed small paired positive-payload drift. Before executing any new statistic, fix a CPU-only description of whether positive payloads are approximately constant within the original question. Use only the completed442927compactarchives, allfourmodels/all272scenes/2176positiveoccurrences permodel, originalt0, exactquestion+targetgrouping. Actualappliedweightsnativeg/all-open1, retainedsamplingduplicates andzero-weightpositivecompanions. Compute anchored FP64 weighted/unweightedquestionmeans, coordinateRMS/maxitemspread, all96singularvalues/energyfractions of pooled uncentered andquestion-centeredweighted/unweightedpayloads; no numericalrankthreshold. Emptygroups/K0 andzeroenergyfractions remain explicit/undefined.

Per-scene P-minus-sumgate*questionmean decomposes into FP64 weightedpayloadheterogeneity plus actualsavedFP32multiplicationresidual, withrtol1e-10atol1e-10reconciliation. Questionmeans are descriptive statistics estimated fromthefreshtest sample and NEVER deployed; no model/head/probability/optimizer/intervention/accuracy computation orchangeddecision. This exploratoryanalysis cannotprove causalnecessity ofvectorvalues. CPU4cores16G300s; sevenembeddedconstant/twodirection/zero/empty/ragged/invalidweightchecks. Rootreviewedsource/provenance/algebra/coveragebeforeexecution. Frozenhashes {"scripts/analyze_native_vision_v18_payload_rank.py": "89a870229d376bea1e0486a0eef549438da73707d32f6381b253d8f5952ec5dc", "slurm/native_vision_v18_payload_rank.sbatch": "3ac3d49f82bb11bcc3da1ccbea4b11149cc280f387ed9bcb2d496bdb18ab43ed"}.


V18 exploratory positive-payload CPU442933 passed5allocatedseconds/sevenembeddedtests, all1088scenes/8704positiveoccurrences/64K0scenes. AnalysisSHAfe7b5a86d7494ae35b350a9c102a1c085603cd6c182978667a7b82254c9c9e34. Gatedseeds20/21 uncenteredlargest-singular-energyfractions0.9999999999295294/0.9999999999337671; within-questioncenteredenergyfractions5.948203937823727e-12/6.097670468803335e-12. PerN64positive-scene meanrelativePdeviationfromquestionmean*sumgate8.053087469843275e-7/6.214110882697013e-7. Allmodelidentityerrors<=8.881784197001252e-16. This supports a nearlyweighted-count description at theoriginalquery; no replacement/nativeaccuracy intervention wasexecuted andtinycomponentscouldstillmattertodecoder. Do notclaimhigher-dimensionalaggregationfromV18.


### Cosmos semantic aggregation through reasoning prefixes: prospective software release

Root and independent source reviews passed. Freeze direct local measurement with the unchanged official Cosmos reasoning global prompt, on the two old N16/K3 and N64/K6 software scenes. Use a new untrained rank96 core, seed20261118, with identical parameters except Uzero versus Normal(0,.001); no V18 checkpoint transfer. Six native greedy trajectories (bare/zero/active at each N), maximum8tokens, unmasked original EOS. Let G<=48 be all generated tokens and A<=16 active tokens: full-prefix replay at every active prefix including empty t0 gives VLM/ordinaryheads G+A<=64, vision6+A<=22, originalprobes exactly4/all164originrows, replayheads G+A<=64, extraheads<=68, all28-layer KV transitions G-6<=42, gatedforwards<=48. Full replay uses the already-bound original artifact and performs no extra origin probe. Original direct-local gates remain fixed while payloads follow the common raw global prefix.

Exact bare/zero-U tokens and raw logits required. All164 original hidden/normalized states equal bare, all origin/current-head native replays TV<=.02/exacttop1, full-vocabulary FP64 probabilities replay rtol1e-12/atol1e-15 and actual gates exact. Cached/full active global divergence is descriptive, with every comparison retained. Four-axis native generation and three-axis full-prefix positions, actual masks, all28KV layers, raw argmax/EOS, source/model/runtime/processor/native FP16 norm/head identities, frozen parameter versions and actual checkpoint tensor tables are checked. No fit, answer score or reasoning efficacy claim.

CPUcheck/report4cores16G300seconds; GPU oneB200180seconds, campaign540GPU-seconds including failed/zero allocations, prior+180 reservation before load, maximumone streamGPU andfour projectGPUs, knownbadnode131-255excluded. CPUcheck runs the existing seven probe groups, twenty semantic-core/controller tests, exact max/early-stop inventory and nonmutating recorder fixtures. Initial core tensors are saved under the authorized checkpoint root, data under the authorized data root. Design docs/paper/NATIVE_AGGREGATION_REASONING_SEMANTIC_STREAM.md. Frozen hashes {"scripts/native_vision_reasoning_semantic_stream.py": "eb45f55b066a76232fa8c750aeb22d4acb037f4d2b70d2a50a13fa8d2680a494", "scripts/profile_native_vision_reasoning_semantic_stream.py": "1ea9311c8e3884aa04fe64d2cfc732938dfd7e495179dda83bc598453c3630ef", "slurm/reasoning_semantic_stream_check.sbatch": "7501ca4d6e85647f6476c38a2751b8558cdd0ea5e77691d5c18c31ce68010530", "slurm/reasoning_semantic_stream_report.sbatch": "9078fc37e9a2e943cf77e7a8ce9d29275169e66938d6c6dce9c7df836b744cbf", "slurm/reasoning_semantic_stream_run.sbatch": "f1520bf7246daf72c20c886fb830529d5dfa23347b0d3ff3b03706d608e23a4f"}.


Cosmos semantic stream CPUcheck442952 passed allregistered tests/config-only preparation; planSHA22923543a416fc7ddddefeaab8841be09bab8cae470b8fa9f514e2259261f93c, summarySHA501c426637286caa4b941799cc8d7dd531a8f360f0a64b4dc4addb3f394babeb. Rootverifiedall35current/sourcecopybindings. The unchanged180-secondGPUsoftwarecheck and dependent independentCPUreport are released; no efficacy evaluation.


Cosmos semantic stream GPU442954 passed74allocatedGPU-seconds; independentCPUreport442955 passed18allocatedseconds. AnalysisSHA41031ed5a104925a212445fe84b88d5660ff07dc2d625b639f50e9b393422b83, summarySHAf64e961bea0ce1ae62fbfc61d58c0034bee9da14c8898e1cd1df0e1f76cd6e03. All48generatedtokens/16activeprefixreplays/64VLM/22vision/4originprobes/164originrows/64headreplays/68extraheads/42all28-layerKVtransitions/48gatedforwards verified. Maximumsame-stateheadTV0, exactbare/zeroU identity, fixedoriginalgate reuse, no cached/full comparison failures. Rootverifiedall35source/current/snapshotsatallthreeexecutionstages. No fitting/accuracy orcomposition efficacy.


### Identity-join task after the V18 weighted-count finding: prospective design

V18 practical counting success does not establish content-sensitive aggregation. Freeze a new MMReD-derived two-room identity join with canonical9people/6rooms: three people each appear twice among six included images, each room appears three times, and exactlyoneperson appears in both rooms. Each contrast family contains allthree answer-changing variants with identical included-count, person/room marginals, occurrence slots and irrelevant backgrounds. Randomize placements and counterbalance the two nonanswer room orientations. Global question and the generic local inclusion wrapper are exactly docs/paper/NATIVE_AGGREGATION_IDENTITY_JOIN_PROPOSAL.md; no question/identity/room parser is used during inference. This is a new derived task, not a claim about the canonical counting benchmark.

New measurement interface: original complete-unmasked-native-vocabulary argmax must be literal1/0, giving detachedbinary gate1/0; otherargmax invalid, never silentlyclosed/oraclecorrected. Gates stayfixedacross nativeanswerprefixes. Before any fit, audit all9people*15roompairs*6depictedrooms*Steps[1,8,16,17,32,64]=4860judgments:1620included,3240excluded. Every case must be valid/correct under the one frozen wrapper. A failure stops this proposal; no filtering or promptsearch on these outcomes. CPUprepare4cores16G600s with324unique canonical atoms/sharedpixels. Fixed timingpilot batches0and89 each54locals+1global, then full90batchaudit repeats allcases. Eachbatch onebareVLM/vision/nativehead plusonesame-shapedheadreplay; full4950nativerows, campaign5060includingpilot. CPUindependentreport4cores16G300s. GPUprofile180s, full600s only if measured load+1.5*90*max(two fullprefill+headreplay+save timings)+30<=600; campaign900GPU-seconds including failed/zero, nextfull600reservation beforeload, maxonecampaign/fourprojectGPUs. Fullfeasibility outcomes only after fixedfullaudit, pilot timingonly. Executable source stillunderreview, noGPUreleaseyet.

Matched branch comparison: same1041600parameter rank96 tanh/SUM/SiLU residual. Vector uses binarygated localstatepayloads. Scalar control replaceslocalstates withcurrentglobalstate for denseprojectionwork, but computes exactFP32count s=sum(binarygates) first, then aggregate=s*onefixedquery-onlypayload. It mustnever sum interspersed repeatedFP32payloads to form itsaggregate. Thus identicalcapturedglobalinputsandcorrectbinarygates giveidenticalcontrolinputs across every3variantcontrast and atmostonecorrectnamepertriple fordeterministicgeneration. Verifyactualgate/global/controlinputidentities andreportallviolations withoutdroppingcases. Nativeglobal name+EOS unmaskedmax4; exactcanonicalname aftersurroundingwhitespacestrip, noalias/partial/truncatedcredit, tokenizerbudgetcheckedbeforeexecution. Training residualconsistency onlysamevariantN8/N16pairs.

Finite data design seed20261122: all12nonheldroompairs*all84unorderedpersontrios=1008trainfamilies*3variants*2lengths(N8,N16)=6048traincontexts. Dev12pairs*trios012/345/678*3variants=108freshN16contexts. Seen-pairtest usesKitchen/Garden,Bathroom/Bedroom,Office/Park; heldtestKitchen/Bathroom,Garden/Office,Bedroom/Park. Eachregime3pairs*12balancedtrios*3variants*2lengths(N32,N64)=216contexts. Testtrios012,345,678,036,147,258,048,156,237,057,138,246 incanonicalCHARSorder. Everytestperson4occurrencesamongtrios,12answersperN/regime. Freshcompletecontexts/backgrounds/placementsacrosssplits, canonicalStepindices; lengthpairsonlyinsertirrelevantimagesoutsidechosenrooms. Eachcompletefamily is the uncertainty/contrastunit. Source/render/feature/fitting releases remainheldpendingtheirchecks andbinarymeasurementfeasibility. Proposedfitseeds22/23 and12epochs4536updates are not yetreleased; quantitativecriteria andmeasuredcostmustbefrozenbeforetraining.


Identity-join binary aggregation core passes independent static review. Release only its seven CPU algebra/gradient/control-boundary tests, twoCPUcores4G120s, no model/gate/data/fit. Scalar must consume captured aggregate directly, never sum descriptive messages. Exact source copies outputs/native_aggregation_vlm/identity_join/core_source_20261122. Hashes {"gnnformer/parallel_local_aggregation.py": "ed600b0ae865108093b74bb1bcf30f905dc38a8a67cb7708cb9f1688f2df7b0f", "gnnformer/parallel_local_binary_aggregation.py": "4c68919d001ec1081efe2634d166ddf067b56ca43b35a25ec0e12c5093e6a459", "gnnformer/parallel_local_semantic_aggregation.py": "e51ab3859812b4d9c6680f2343883e4e3a00498df99fba74e0591d2bc3994f5a", "slurm/identity_join_core_check.sbatch": "fc9d384a70ed444d4497ef6ca637672f6769f73d69c4b4ad1ed5cc72f8ff747e", "tests/test_parallel_local_binary_aggregation.py": "13d54d6e569cc9eebab931752124c0e686723d57fde62906352670173bd7ed95"}.


Identity-join binary-core CPU442960 passed seven algebra/gradient/control-boundary tests in1allocatedCPU-second. Allfivefrozen source/current/copies match. Saved outputs/native_aggregation_vlm/identity_join/core_source_20261122/test_result.json. Scalarcontent/gate-position invariance exact; vector content sensitivity and binary invalid rejection pass. No native interface/fit release.


Identity-join generic inclusion audit passes root and independent source reviews. Freeze the registered90batch/4860judgment/324atom protocol, exact literal prompts and allresource/measurementcriteria. CPUcheck600s is released to run seven meaningful denominator/invalid/fullmass/replay/resource/timing groups and fullprocessor/canonicalrender/nativeidentity preparation. GPUprofile remainshelduntilCPUproofchecked; fullaudit onlyafterfixedtwo-prefilltiminggate, independentreport includesallcases. ActualQwenFP16norm/head,NF4doubleBF16,SDPA,hardwareancestor andarithmeticdefaultbindings required; allactuallyexecutedlocalhelpers/sourcecopiesbound. Frozen hashes {"scripts/probe_native_identity_join.py": "e6a717f9c81bca50d676bf92f032c8a3e36608370c4cf775c6b832bdaadf1f6b", "slurm/identity_join_gate_check.sbatch": "11a67d04832f794b5fb755897fb6baafd868ca453bc165bb03c6b207e4c3a6d3", "slurm/identity_join_gate_profile.sbatch": "01e3588110484e9e78acc4a7c6bea8040390ae0493c18c729961cc06f8b49b51", "slurm/identity_join_gate_report.sbatch": "bda1174243a47c89af41a37e8f66329c720efa4b9ad81e3216120d9f7850d094", "slurm/identity_join_gate_run.sbatch": "43e23689ac10711f57674685bcdc9c982e4d73bb56c67975f2e591d8baec6e96"}.


Identity-join inclusion CPU442963 passed97allocatedCPU-seconds, sevenregisteredtestgroups, all324canonicalrenders(317existingatomsreused),90mixedHF/nativepositionlayouts and18sourcebindings. Rootverifiedallcurrent/snapshotbindings. PlanSHA87cc1d9dff4b761ad6608bfda8996946f628b25c86e9b4aaa633c3857d960758; summarySHAdae5dcec255ee7749a2aa62165363a240b7642389d82ff3f6ba78b3d66778b27; nativeidentitySHAac76aa2d7ca542972d42ac2b548adbefd675f87e8ca8cb2078c77ed405a7c5c3. Sandra/Noah requiretwoletter-piece tokensplusEOS; sevenothernamesonetokenplusEOS, allwithin4budget. Unchangedfixedtwo-prefill180-secondGPU timingprofile isreleased; no measurementoutcomesread.


Identity-join fixedpilot442966 passed32GPU-seconds: exactlytwoB55prefills/tworeplayheads,all110replayrowsTV0/exacttop1. Timing1.8471519751474261/1.492935943417251seconds,load14.508896105922759; registeredprojection293.8744127508253<=600. Rootverified18current/sourcebindings, counters andprojection; pilotmeasurementoutcomesnotread. SummarySHA1d0f815337c6de7e8cf1eb6be5252e8aecfeeef13d3bc391e8e99243f2590b7c. Release unchangedfull90prefills(4860local/4950nativerows)600s andindependentCPUreport. Actualprior32+fullreservation600=632<=900. No branchtrainingreleased.


Identity-join dataset source passes root and independent review; rootreviewedthefinal11-lineverify_stagebindingclosure withunchangedscientificsettings. Freeze exact6588contexts/1116contrastfamilies/3240lengthpairs/95040imageoccurrences andbalanced6048/108/216/216splitmatrix. Release CPUcheck4cores16G600s for sevenmeaningfulsymbolic/input-corruptiongroups, deterministicregeneration, allname+EOS inventory(13440trainpositions), canonicalreuseinventory andsourcebindings. RenderingseparateCPU600s remainshelduntilcompletefixed4860nativegateproofpasses. EveryrenderattemptnewDATA/identity_join/stage_JOB, everyreusedatomindependentlyrerendered/RGBcompared, allQA/frameorder/hardlinkinodes audited. Runtimeinputsallowlist excludesgold/trio/variant/states. Frozen hashes {"scripts/stage_native_identity_join.py": "1ccbb48dc28bb4e4e11a182760d497057f056e2237143465eebbf4110e20b287", "slurm/native_identity_join_check.sbatch": "bb54fa10119b4097c81bc9ebdad91806e3d98bcc1c4b71f57f0fb7eccb625815", "slurm/native_identity_join_render.sbatch": "e107e18e392798a0e64b9ed2ba6fd91e28bb3d2cda8c8fefee495827ee0cb722"}.


Identity-join generic inclusion fullaudit442968 completed167GPU-seconds; independentreport442973 passed49CPU-seconds butFIXEDFEASIBILITYFAILED. All4860judgmentsretained:1122valid/correct(allnegative0),3738invalid,0/1620positive literal1,negativeaccuracy1122/3240=34.6296%,overall23.0864%,equalclassmacro17.3148%. Nativeoutputsfrequentlypersonnames instead ofbinarydecisions. Fullnativecomputational/headreplayintegritypasses;FP64probabilitymaxroundoff3.552713678800501e-15. AnalysisSHAf769bf3dc39c4472acce79d55143ba6e40448fae3d97cdb6248f8b79a773df20, summarySHA4cfd178ec4cbe06f514ace2b69c30d4c197f1a0c3e14b7322fd1b6e7fefe8643. Total199GPU-secondsincluding32pilot. Rootverifiedall18source/current/snapshotbindingsatallfourstages. Asregistered, STOP thisbinarypromptjoinroutebefore rendering/cache/fit; no promptsearch/filter/oracleconversion. Newbinarycontroller/runtimeheldsourceisnotreleased.

Independent join symbolic/tokenizer CPU442974 passed13allocatedCPU-seconds with exact6588contexts/1116families/3240pairs/95040imageoccurrences and13440trainingtargetpositions. Source1ccbb48 remainsfrozen. This data-law check doesnot overridefailednativefeasibility. Existing --render continues to requirethefailedgateproof; no imagespublished. Any alternative method requiresaseparateprospectivedesign/source/data release, withouteditingthefailedinterface oritscriteria.


### New learned-selection join experiment after the stopped binary interface

This is a separate method and release; it does not change the failed generic binary-prompt criterion or authorize its old renderer/runtime. The complete failed result remains final. Reuse the independently checked, still-unrendered semantic dataset law from CPU442974 (planSHAcf1c153d399f73c9a5614f1cc3ee7e4efb5df8256216420d23b70d831d21f668), with new DATA/identity_join_learned attempts and a new four-field native input interface: sid, n_frames, question, ordered image_files. Local image rows and the text-only global row receive the SAME UNALTERED global task question. No binary wrapper, original head probe, probability gap, per-image target, answer parser or external relevance rule is used by the method. Frozen old --render still refuses the failed gate. New source will separately verify semantic contexts, names, canonical pixels and the changed model-input contract.

Use three learned-selection arms, each the same1041697parameter rank96 conditional tanh payload and native SiLU residual readout. For current prefix t, q=WqRMS(h_global), u_i=tanh(WlocalRMS(h_i)+q+b_local), s_i=w_gate^T u_i+b_gate. Allarms initialize w_gate=0,b_gate=.5,U=0. Clip: g_i=clamp(s_i,0,1). Matchedsmooth: g_i=sigmoid(4*(s_i-.5)). Standardnormalizedattention: g_i=softmax_i(4*(s_i-.5)). Aggregate SUM_i g_i*u_i, followedbyunchangedWagg/SiLU/U residual andnativeFP16castbeforeadd/norm/head. Allgateparameters receive only native set-answer CE andsame-variant length consistency. Clip/sigmoid have identical initial values and first derivatives; this is not a pure causal intervention on zero support. Softmaxscorebias cancels, so its identical parameter table contains one redundant scalar. Softsigmoid can numerically saturate to0/1inFP32; report actualexecutedvalues. No sparsity penalty, local supervision or tuned threshold. Selection recomputes fromcurrentcausal states at every token, with no extrahead/model/vision call. Paddingmask concerns only actual-row existence.

Fixeddata6048train,108dev,216seen-test,216held-test; seed20261122 andallsemanticroom/person/contrast/insertion invariants remain asregisteredabove. K6andall9familiaranswer names; trainN8/N16,testN32/N64; threeheldroompairs. Eachtestcell108contexts/36completeanswer-changingtriples. The old scalar-count ceiling is NOT a claim or endpoint for any learnedgatearm, because continuousgatevaluescanencodecontents.

Sixmatchedfits, clip/sigmoid/softmax x seeds22/23, maximumfourconcurrentGPUs. Sameexactparameterinitialization (includingredundantsoftmaxbias), samepairedshuffle/order, same6048trainingcontexts andnative strict-prefix name+EOS targets. Twelvecompleteepochs4536batch16updates, eightlengthpairs/batch,72576scenepresentations and161280targetpositions perfit. Loss=scene-mean nativeCE + coefficient-one same-variant N8/N16 residualconsistency, denominator frozenidenticalglobalstatesquarednorm+1e-6. AdamWlr.001,warm50,cosine1e-5,wd0,clip1; no otherregularizer. Fixedfinalcheckpointonly; reset/reloadactualsavedtensors, one108devsweepdescriptive thenall432testcontexts. No accuracy-dependent stopping/selection ortestcalibration. Native unmaskedgreedymax4tokens, completecanonicalnameplusnativeEOS; surroundingwhitespaceonly, noaliases/extra/nonterminalspecial/truncatedcredit.

Primaryscreen requires in EACHseed andEACHseen/heldregime atN64: clip exceeds EACH sigmoid/softmax control byatleast6/108wholeanswers. AtN32 clip mayloseatmost3/108againstEACHcontrol ineachregime/seed. Practicaltarget requiresEACHclipseed >=98/108seen and>=87/108held atBOTHlengths, inadditiontoprimary. Reportallarms regardless, fullthree-variantfamily correctness, pername, N, regime, truncation/format andallpairedtransitions. Bootstrap10000draws seed20261124, whole3variant*2Nfamilies stratifiedroompair, samedrawsallarms/fixedtwoseeds; theseareconditionalexampleintervals,nottraining-seedpopulationinference.

After finalweights/outcomes freeze, retain actualcurrent-prefix scores/gates/payloads, offline relevant/irrelevant P/Z decomposition, closedpositive counts, nonzeronegatives/leakage andcanonicalStepstrata onALLtestcontexts. No per-image labels influence training/inference. Exactclosure is a possible learnedbehavior, not a guarantee from ReLU/clipping orpairedendpointconsistency. Cancellation/readoutnullspaces, wronglyclosedrelevantimages, saturatedgradients, score-codedcontents, Stepdistribution shifts andsoftmaxdilution remain live failuremodes. A clip gain without the corresponding measured behavior doesnot establish that exact suppression caused it. Known gated set pooling andstandardattention are the primitives; no newattention/bandwidththeoremclaim.

Prospectivehardcampaigncap24000GPU-seconds includingfailures/zeroallocations: sixmains2700s=16200, fourfeaturecacheshards1200s=4800, featureprofile300, threetrainingprofiles300s=900, nativesoftware<=900, contingency900. Priorbinaryaudit199GPU-seconds remainschargedtoitsstopped900-secondcampaignandreportedseparately; no doublecounting/newbudgetresetofsamejobs. Newdata/featuresauthorizeddataroot; allnewhead/norm/coremodeltensorsauthorizedcheckpointroot. Everyrelease requiresactualpriorcostplusfullremainingreservations<=24000. CPUandGPUjobsallSlurm. Featureinventory exactdeduplicatedstrictprefixinputs only; noglobalteststate enterscache. Resource/formula/source gates are stillpending; noGPUfit/cache isreleasedbythisdesign.


New learned-selection dataset source passes root/independent review. CPUcheck4cores16G600s isreleased for sixtransformation/input-boundarytests andfullparentsemantic/tokenizer/sourceverification. Only local_prompt metadata changes to question; contentstates/answers/slots/backgrounds/insertionlaw unchanged. Four-key modelinputs sid/n_frames/question/image_files; oldbinarywrapper discarded. New --render source usescanonicalpixel/QA helpers only and has no failedbinarygate dependency; its own CPUproof stillrequired. Full6588contexts,3016uniqueatoms/95040imageoccurrences,13440trainpositions; renderseparate600sCPUaftercheck. Frozen hashes {"scripts/stage_native_identity_join_learned.py": "cb9552a083f01aee52cf6c8a5a7ee8a4ae908ce027d68994b49e7c74a8223280", "slurm/native_identity_join_learned_check.sbatch": "dc6f213f3e8ec3a04c410f9ad03ae38eb41240cd511d4484527df5c9247b96e4", "slurm/native_identity_join_learned_render.sbatch": "605bc18a5c2611053c1d6265b4a0c1ee292983528d0b4a562570bd3b5e9f1034"}.


New learned-selection data CPU442986 passed12allocatedCPU-seconds, sixinput-transformationgroups andcompleteparentsemantic/tokenizer verification. Rootverifiedallninecurrent/sourcecopybindings. PlanSHA1f5a404e54d14ff69dcee51f66bd09a07239170f691a34c93e679a6a12f5dfc2, summarySHAa8b2bbdaca16603718aea02cd18731a41991353d90a9cd02abba5381f418bd7e. All6588semanticcontexts/3016canonicalatoms/95040imageoccurrences unchanged; everymodelstreamreceivesunalteredquestion. Release newindependentrender4cores16G600s toDATA/identity_join_learned/stage_JOB, everyatomcanonicalRGBreplayandallimage-order/QA/sourcechecks. Oldbinaryroute staysstopped. No cache/fitrelease.


New learned-selection render442988 passed116allocatedCPU-seconds. All3016canonicalatoms independently rendered/RGBchecked (2886reused,130new), all95040imageoccurrences/QA/frameorder/hardlinkinodes and6588four-key input views audited. Rootverifiedninecurrent/sourcecopybindings andmanifest/plan hashes. ManifestSHAdb2045c772a475eb36562c70018f0aa7641efcd6d71a5e3201c7da1bf1a0d94f; summarySHAd893bf436db50fdb98c25c45e841b01c209d8f77cd7d744b69ef68044bf3f2b3. No binarygate dependency or hiddenfeature reuse. Software/cache/fit resource gates remainpending; no learned-selection GPUjob released.


New learned-selection feature stage/worker passes root+independent staticreview. Freeze fourphase causalfeature inventory, canonicaloriginalpairorder, no local supervision/probe oroldhiddenreuse. Release CPUstage4cores16G600s onpassedrender442988, checking6048trainingcontexts/13440targetpositions/161280localfeatureoccurrences/72576empty-prefixlocaloccurrences. Forty-twodistinctprofileinputs plus4B64stressbatches and8N16+1nativeprefix calls total16VLM/12vision/16same-shapedheadreplays/434bindingrows. AllN32/N64testfeatures excluded. LaterfeatureGPUprofilerequirespassednewsoftwareindependentreport; four1200sGPUshards require measured load+pixels+1.5*sum(ceil(phase_rows/64)*max_complete_small_stress_phase_seconds)+30<=1200 andwholecampaignreservation. Allrawreplay/tensorbindings independentlyverifiedatmerge. Frozen hashes {"scripts/cache_native_identity_join_features.py": "1042ce2c0f80c7c459cfc522548f9611c119c612bd899c156f4254d8bda19295", "scripts/stage_native_identity_join_features.py": "80226c3d521e35e64ae22f94f93b6f3fc183cdb4ab621dfed5fa074692ae362b", "slurm/native_identity_join_feature_merge.sbatch": "be3f31f7c9e370488e57b256409caaeca383e7e254cc06026e17f40aaafaafbe", "slurm/native_identity_join_feature_profile.sbatch": "3ead7e8550cc61d49b2333a757c89b5b18426765817b78bfb683eccf4fa448cb", "slurm/native_identity_join_feature_shard.sbatch": "880c872b6ea8771eebfad59450d7e1b905c14516ff84eb7b21b7d761cb57477a", "slurm/native_identity_join_feature_stage.sbatch": "57e108e1000af866954c3f7eb3678b18f40921bf244b9d4d89b29b54a2d2d765"}.


Learned-selection native software source passes root+independentreview. Release CPUcheck4cores16G600s includingtennewcore/runtime tests, inventory/resourcefixtures, unchanged-question nativeprocessor andfour-axisgeneration/three-axisfull-prefixlayout checks. Cases lowestSIDtrain_N16 anditsfixedseed20261124 outside-roomcanonicalStep17..64extension; no freshtesttrajectory. Freeze14naturalmax4trajectories:bare,zeroall3modes,activeall3modes atbothN. Privateuntrainedseed20261124,identicalparamsUzero/Normal(0,.001); exactbare/zero andactiveinitialclip/sigmoid. Everyactiveprefixincludingt0 getsfull-prefixreplay. IfGgenerated/Aactiveprefixes:VLM+ordinaryheadsG+A<=80,vision14+A<=38,same-shapednativeheadreplaysG+A<=80,all28-layerKVtransitionsG-14; originprobes0. OnecorecallperbranchactiveVLM, global-onlyFP16castbeforeadd,noKVwrite; fullrawhead/fusion/mask/position evidence savedbeforelatergates. Cached/full differencesdescriptive. IndependentCPUreportrequiredforfeatureGPUs. GPUprofile300s/softwarecampaign900inclfailedzeroallocations,whole24000capandmax4projectremain. Freeze {"scripts/profile_native_learned_selection.py": "b646ca610de525c96a13ad53bc46079f72db7c875543a6a4292fa63f679d3b5d", "slurm/native_learned_selection_check.sbatch": "d4ce538c98da8a90ee540752c7bf6927ab4540ce0fa1a94f19fde23bc967e010", "slurm/native_learned_selection_profile.sbatch": "b11317d56de00ca2674fdd93538b40f7e3bbdb8b80c92e89d321c06fe37a2858", "slurm/native_learned_selection_report.sbatch": "78f1d52a964a589db6e6b1dcb28b45e88066bf64d160436d754059ada7118ec8"}.


Learned-selection feature CPU443005 passed53allocatedCPU-seconds:64847distinctcausalstates (9007localempty,55696localprefix,12globalempty,132globalprefix),864canonicaltrainingimages,6048scenes. Rootverified35current/sourcecopies. PlanSHAcd8f92fdceb23f69bf52be3c8487a1c722780b2b1444435cd019ae8fcd3514bd; summarySHAafdf9c8b38517cc0eb4314e989e6968f48f4af6d429a2c721d2e5f1c7312bc9e. SoftwareCPU443013 passed16allocatedCPU-seconds,10core/runtimetests+inventory/accountingfixtures+actualprocessor/layout. Rootverified39current/sourcecopies. PlanSHAc1814de78824c9562c4b38a213b87da3ec9a22aebc813148fb507c4a7454386c; summarySHA8e49fdd04c537052fd63f8317cbe12b11953d4d299fe848d5fb7baa3d5610f47; nativeidentitySHAebb81bbfa7873389e0734cf860e5cb3dbb2639f605a5f42e967bc982dcb405f3. Release unchanged300sB200 native software profile. Newcampaignspent0;300software+300featureprofile+4800cache+900trainprofiles+16200mains=22500<=24000 (600additionalsoftware+900contingencyremain). No fitrelease.


Learned-selection softwareGPU443015 completed80GPU-seconds; rawprofilepassesall40nativecalls/26vision/40same-shapedheadreplays,zerooriginprobes,initialclip/sigmoid andzeroUidentities. Rootverified39current/sourcebindings. Release unchangedindependentCPUreport4cores16G300s. Exactsoftwareverdict andcacheGPUrelease awaitthatreport; nofit.


Learned-selection independentsoftwareCPU443016 passed; all40nativecalls/26vision/40same-shapedheadreplays and28naturaltokens verified,36selectioncalls/14all28-layerKVtransitions,zerooriginprobes. AllnativeheadTV0; no descriptivecached/fullfailures. SummarySHA2a5e6427d8340034e3c1a795932bda15b8bdb1ec6e2da6f74e24c4cc9d93b881; analysisSHAc762d2276b75a5c655785d5bfe85c5cd266f5dc181b9801ea0ca03628d046a04. Rootverified39current/sourcecopies andanalysisbinding. Release featureprofile300sB200 usingCPU443005 exact64847-stateinventory andpassedsoftwareproof, fixed16VLM/12vision/16headreplays/434rows. Actualprior80+300featureprofile+4800cache+900trainprofiles+16200mains=22280<=24000; softwareunspentbudgetstillcountedwithinremainingallowance. No fitrelease.


Learned-selection featureprofile443017 passed50GPU-seconds; all16VLM/12vision/16same-shapedheadreplays/434nativerows pass. Rootverified35current/sourcecopies andobservationbinding. SummarySHA97ad8928e820d1b3ac2e27746a99a28fa57ec1a1b72cc855a1b27452c2624708; projectedshardseconds [695.4501609667204, 695.4501609667204, 695.4501609667204, 695.4501609667204]. Releasefourunchanged1200sB200shardsinparallel(max4GPU). Wholeactualprior130+4800shards+900trainingprofiles+16200mains=22030<=24000,leaving1970sforunusedsoftware/contingency; no mainsreleased.


Learned-selection trainer passes root+independent staticreview: native actualFP16norm/headCE+same-variantresidual1,12epochs4536updates,canonicaloriginalgenerationorderbeforepairedseed22/23shuffles. Sixmeaningfulnewtrainingfixtures pluspaired-sequence tests atCPUcheck; no repeatedcoretest. Fixedfinalonly checkpointreset/reload andfullrawnaturalarchivebeforeevidencevalidation/scoring. Software/cache/exactmodelcopies bound; fullbackbone frozen. CPUcheck4cores16G600s releasedonce independentcachemergepasses. Threeprofiles300s each,32updatesseed22 andninefixedtraining-derivednaturalcases (all3variants atN16/N32/N64),max36VLM/9vision/36same-shapedheadreplays, allcompleteprepare/generate/replay/save/hash timed. Mainrelease requiresindependentdata/source/log/checkpoint/replayaudit andsetup+1.25*(first4+4532*steady+108*T16+216*T32+216*T64)+120<=2700perarm plusactualprior+16200sixreservations<=24000. Node131-255 excluded/max4projectGPU. Frozen hashes {"scripts/train_native_identity_join_learned.py": "ef95207222c3769d9245155d7ff78678d949534521b5b503119b8a348c89c077", "slurm/native_identity_join_learned_train.sbatch": "f759db5ac7dbea51ac98dfdaa89057215395e4ef28cc34a2eb8839e308b55274", "slurm/native_identity_join_learned_train_check.sbatch": "c62a25f26d6929eae4dfa29af4902237c4401247f4d8a11b95e98392ca2803fb", "slurm/native_identity_join_learned_train_profile.sbatch": "ed2835ac25d5830ec2972710493e408d0d0ff347c9b0d435f17c56f45f382ad4", "tests/test_native_identity_join_learned_training.py": "6601d0762164c686886859e11a06be003b31a7ca387ce6defa9ed44c21a251a0"}.


Allfourlearned-selection featureGPUs443019/443020/443021/443018 completed462/456/457/454seconds (1829GPU-seconds),withcompleteper-shardnativeFP16files/rows and35sourcebindingsrootverified. Release unchangedCPUmerge4cores16G600s:verify64847unionstates,allrawbatch/final/per-rowhashes,434profileheadrows,136descriptivecache/nativehiddencomparisons. Actualnewcampaign1959GPU-secondsincluding80software+50profile. No fitreleaseduntilcompletecacheaudit.


Learned-selection independentfeaturemerge443031 passed23CPU-seconds; all64847nativeFP16states,6048scenes/13440targets,434headreplayrows and136descriptivebatch-routehiddencomparisons independentlyverified. Rootverified35current/sourcecopies andcachebinding. CacheSHAdb060aed422aa05459d3fde9f512daf44bcab447c1426ff8c5be111764190d46; summarySHAa5b3bdc6fca74d423dc0327b913d0601c98619fb0195b2c19999321022b71d04. Release frozenef952072trainerCPUcheck4cores16G600s withthiscompletecache/newsoftwareCPUreport443016/render442988. Alltraining/pixel/modeltensorwork remainsSlurm; noGPUsfituntilCPUcheck.


Independent learned-selection reporter/analysis passes root+independentreview; release CPUcheck4cores24G900s forfiveevidence/criteria/whole-familybootstrap/resourcegroups+tengeometrytests andall6588contextpublishedQA/RGB/order/semantic/inputverification. ExactcontrollingPREREGpracticalcriterion remainsprimaryANDabsoluteaccuracy; proposalword"independently" isexposed separatelyasabsolute_accuracy_target, neverusedtochangeacceptance. Reportall6fixedfinalmodeloutcomes andalltestprefixes; devcapturesalsointegritychecked. IndependentnativeFP16rawEOS/argmax/text/inputdigest, actualfinaltensor/initial/gradient/log/reductionchecks; everyloggedconsistencydenominator replayedfromhashednativeglobalstates atFP32crossdevice rtol1e-6atol1e-4. CPUfunctional q/payload/gate/aggregation driftisdescriptive; exactelementwisemessage/clip/nativeFP16write enforced; P/Z andWaggP/Z inFP64withrtol/atol1e-10. Poolmaxsetup/first4/steady/T16/T32/T64 acrossALL3profilesbeforemainprojection; joinconsumedsoftware,featureprofile,4shard,3trainprofile IDs to successfulschedulerallocations. Wholecampaign<=24000/max4/software<=900includingfailedzeroallocations. Fitting/testrelease stillseparateafterallprofiles. Frozen hashes {"scripts/analyze_native_identity_join_learned.py": "86a99532d917e851a9fadee29d85175621936bd7f58ef9d50600b935a8153728", "scripts/report_native_identity_join_learned.py": "7516a560327e1e6672a8f87186da370ac735fc9fe7d923010cdd9b9a2b24deb0", "slurm/native_identity_join_learned_report.sbatch": "73769cf38e4f3f74d891745a9a4ba1fb690c927c97ee81667b22f356942673f6", "tests/test_native_identity_join_learned_analysis.py": "edf68e1ade3263bd6d6a8ef9c49020838ac8093ac8f1f89adaa458f1e599db70", "tests/test_native_identity_join_learned_report.py": "8d7b064c036b125132bd083886d8d75689aa4db17a4b688d67d9ddc224a07f7d"}.


Learned-selection trainingCPU443032 passed45allocatedCPU-seconds:actualnativecache/globalpairidentity,strictname/EOS/padding/live-gradient/objectivereductionfixtures,canonicalorders and9training-derivedtiminginputs verified. IndependentreporterCPU443034 passed25allocatedCPU-seconds:all6588publishedcontexts and15report/geometrytestgroups. Rootverifiedallcurrent/archivedsourcebindingsandplans. Release threeunchanged32-stepseed22trainingprofilesinparallel(clip/sigmoid/softmax),300seach. Prioractual1959+900profile+16200sixmainreservations=19059<=24000; no mainreleasedbeforepooledtiming/nativereplayindependentcheck.


Allthreelearned-selection trainingprofiles443037/443038/443036 failed48GPU-secondseachAFTER32updates,butBEFOREnative-timing/dev/testexecution: weights_onlytorch.load refusedtheinstalledTransformers QuantizationMethod(str,Enum) incheckpointconfig.hardware.quantization. Actualmodelweights/objectives/gradients werefinite andsaved; allfailedcheckpoints/logs/sourcecopies retained. Totalfailed144,newcampaign2103GPU-seconds. This ismetadata serializationcompatibility, notanaccuracy orcost-screenfailure. Frozenef952072trainer and7516a560reporter remainUNCHANGED. Prepareseparatenarrowscopedserializationlauncher allowingonlytheboundinstalledQuantizationMethod class under torch.serialization.safe_globals, retainingweights_only=True. CPUcompatibilitygate mustverifyallthreeactualfailedcheckpoint/tensor/configidentities andcontextrestoration; wrapper/enum/source/proof sidecars mustbeboundateverynewprofile/main/release/reportinvocation. No broadpicklefallback/no scientific sweep; unchanged3profile retry onlyafterthisCPUproof,144failurecost retained.


Separatemetadata compatibilitylauncher passes rootreview. Release CPUcheck4cores16G300s onthreeactualfailed32-stepcheckpoints, config/tensor/enum/source bindings andsuccess+exception safe_globals/argv restoration. AllowonlyinstalledQuantizationMethod(str,Enum) inscopedtorch.serialization.safe_globals; everyoldloadstillweights_only=True. Originalef952072trainer,7516a560reporter,andalloldchecks/checkpoints staybyteidentical. Newsidecarsbindwrapper/enum/serializationmoduleversions+sources,originalledgers,exactCPUproof,normalreturn/restoredglobals,eachconsumedprofile/release/run sidecarandtargetoutputhashes. Alllaterprofiles/mains/releases/reportsuseonlythese wrappers; oldscientific/data/softwareproofs remainvalid. SameJobNames/resourcecaps/arrayspreservewholeaccounting. Freeze {"scripts/native_identity_join_serialization_compat.py": "7cdc2b6f5def54bfbe74b485b160d1089e3adaad79131f8a8758f90facffcca9", "slurm/native_identity_join_serialization_check.sbatch": "7cb8efe506541834a12fc3eae692798819f06634d2b45ae823f1fff04e44ac5b", "slurm/native_identity_join_serialization_report.sbatch": "06e742c9ab4ea9ef76b0b9aa4ffe02c873b12e2b48abc4367d216c7f864396e6", "slurm/native_identity_join_serialization_train.sbatch": "cc53687a47ae5deca4f1475a2b103e8856ecbde1c99fad6c79c297703e1886fd", "slurm/native_identity_join_serialization_train_profile.sbatch": "f755e55b7b2e26f4508b58aee5d03a3c64bb8a74fa7e6985e88ce5c7681c8024"}.


ScopedserializationCPU443044 passed4allocatedCPU-seconds:allthreeactual32-step1041697parametercheckpoints safelyloadedwithweights_only=True andONLYboundQuantizationMethod,config/tensorsverified; originalsafe_globalsandargvrestoredonsuccessandexception. Rootverified5launcher/sourcecopies+all54/59unchangedoriginalsourcesandinstalledmodulebindings. ProofSHA24b0dd5442f3ca51253aafcc080c5638660e4ffe0189458ce73b07155e5600e7. Release unchanged3seed22trainingprofile retrythroughtheboundlauncher300s/arm; oldfailures144GPU-seconds remaincharged. Actualprior2103+900profilere servation+16200sixmainreservations=19203<=24000. No mainreleaseuntilcompleteindependentpooledtiming/replay/source/sidecaraudit.


Serializationlauncher retryprofiles443046/443047/443045 failed69/69/68allocatedGPU-seconds AFTERoriginaltrainercompleted32updatesandallnine native timingtrajectories,withallnativeheadgatespassed. Failurewaslaunchercleanup: PyTorch lazilyregistersitsownsafeclasses (installedtorch/_dynamo/__init__.py adds_DimRange; distributed/tensor registersitsclasses), invalidatingthelaunchersassumption thattargetimportsleaveglobalregistryunchanged. Preservealloriginalpassedcomputation summariesANDfailedwrapper/Slurmstatuses;206additionalGPU-seconds,wholecampaign2309. No dev/testoutcomesran. Fixedpooledconservativetiming fromthesetraining-onlyprofiles3177.6640598215163>2700:theoriginal2700-secondresourceprotocolisSTOPPED,notaccepted. No furtherv1compatibilityretryorrelease.

Prospective resource/serialization-only V2 beforeanynewfinalfit/unseentestoutcome: preserveALLscientificdata/6arm-seedmatrix/initialization/12epochs4536updates/loss/scheduler/gates/criteria/analysis/4GPUmax. Setpermainallocation3300s(55min),keepingwholecampaign24000sINCLUDINGallv1failures; sourcecandidate hardwaremetadata isJSON-normalized beforecheckpointserialization, soordinaryweights_only=True workswithno customserializationallowlist. Newtrainer/reporter/tests/wrappers getnew filenames; allold frozen sources/checkpoints/proofs remainunchanged. FreshV2CPU/regression/profile/independentrelease gatesrequired; noinheritanceofv1passedSlurmstatusornativeefficiencyclaimunder45min. Reserve2309actual+900threeprofiles+19800sixmains=23009<=24000,leaving991contingency. Newpooledmaxsetup/first4/steady/T16/T32/T64 projectionmust<=3300. No numericalcriteria/targetcoverage/seed/hyperparameter changesorprofile-time cherry-picking.


V2 resource/metadata source freeze passes root and independent cross-reviews: trainer/test by attention_prior_art; reporter/test by hils_audit. Diff is only new filenames, plain JSON hardware metadata, 3300-second main cap and 19800-second six-main reservation, plus actual QuantizationMethod serialization regression and3300/3301 accounting boundary. Ordinary weights_only loading, scientific protocol, immutable V1 sources, whole24000 cap and all failures preserved. Release fresh CPU train-check4cores16G600s and independent report-check4cores24G900s against render442988/cachemerge443031/software443016. No GPU release until both pass. Frozen hashes {"scripts/report_native_identity_join_learned_v2.py": "9a2ccbbe83e85c9a186c83ed18bb36ce0124194898d736d977dc7f5b357f59e1", "scripts/train_native_identity_join_learned_v2.py": "d5095269f7e2c6f4a244a8e2a52ea51fbaa9c6f72c789f1e75d18a3ba80aaedb", "slurm/native_identity_join_learned_v2_report.sbatch": "0be306edd33cead914dac7539412d79e81740580f331645220ea7e502d2013a5", "slurm/native_identity_join_learned_v2_train.sbatch": "7cd5323e98405a9a5bd8d3240baa2618a1044968999bdc6412c343d92a6fd5a3", "slurm/native_identity_join_learned_v2_train_check.sbatch": "65f91cd75ebfd2a3dfb259033640c47753d3269efa6f346e440bbf41c8a5ec88", "slurm/native_identity_join_learned_v2_train_profile.sbatch": "e309be6e6d2330d7635709d75c3a2509008001a01ecac3685b013249ab386d89", "tests/test_native_identity_join_learned_report_v2.py": "6537d74e94da444065825155e6921ccf7dbaf31ad8cc5e9509658e9720aa772e", "tests/test_native_identity_join_learned_training_v2.py": "c764572ac8c96030d5f085b3c2e63d8971d609180f293c98afc485934aa2df03"}.


V2 CPUtraining443056 passed42allocatedCPU-seconds including7trainer+6sequence tests and actual QuantizationMethod safe-load regression; independentCPUreport443057 passed22seconds withall6588contexts/15report-geometrytests. FreshV2trainingprofiles array443058 released300s/arm afterbothCPUchecks; identical32-stepseed22science andtraining-derived9timingcases; no customserializationlauncher. Root subsequentlyverifiedall54/59currentandarchivedsources andnewplan739477cc4a9b5e8278d8e632f0691543fd02ce42673585a349015970c251d79f after correcting a login hash-check archive filename assumption; no experiment source or setting changed. Wholebudget2309actual+900profiles+19800mains=23009<=24000. Allprofilesneed successfulschedulerstatus andpooledprojection<=3300beforemains.


FreshV2profiles443059clip/443060sigmoid/443058softmax completed71/71/70GPU-seconds, all32updates/checkpoint restricted reload/native timing headchecks passed. Actualcampaign2521GPU-seconds includingallfailedV1attempts. Release frozenV2independentCPUmain-release4cores24G900s: auditall3actualprofiles/pairorders/gradients/checkpoints/rawtiming/nativeheads andsource/data/software/cachebindings, poolalltiminginputs, joinsuccessfulSlurmprerequisites andretainallfailedallocations. Reserve19800sixmains,total22321<=24000. No mains untilrelease passes.


IndependentV2mainrelease443066 passed37CPU-seconds. Allprofile/source/data/cache/nativeweights/gradient/order/raw-replay andsuccessfulSlurmallocation checks pass; pooledforecast 3227.959978815168<=3300s. Wholeactual2521 includesallfailedattempts; plus19800reservedmains=22321<=24000,max4. Rootverifiedallcurrenttraining/reportsourcehashes,profilebindings, releaseSHA43e330a2c6d756d14dd5c1f9fa3f9d2c8c04a54518a47580b3806c7afb89a101. Release unchangedsixfixedfinalfits:clip/sigmoid/softmax seeds22/23,array0-5%4,each3300s. No interimcheckpointselection or early test inspection; all6completion thenfrozenindependentreport.


Separate learned native-memory SOFTWARE preparation while the immutable six identity-join fits run. No efficacy or model-cache/fit release. New replay keeps every historical delta live through the frozen causal final block, while explicit same-scene subset loss positions alone reach native norm/head. Common block26 reads for pre_last/post_last. Root and independent test review pass. Release one CPU Slurm4cores8G300s check: nine tiny actual Qwen SDPA tests for unchanged V11 all-write logits/gradients, selected later-loss temporal credit, zero future/crossscene influence, detached-history negative control, ragged padding/indices, selected-only head shape, native FP16 cast-before-add, and frozen parameter/input preservation. No pretrained model, image, GPU, optimizer or accuracy scoring. All outcomes and source snapshots retained. Frozen hashes {"scripts/native_learned_memory_training.py": "8bc7b41a669eed2cbcbaa58f4b434cd8697f6b5b1b41098ee4bfa11e406358ff", "slurm/native_learned_memory_training_check.sbatch": "e01690bd93acddf8cdfa348292b30c3a643c72932ec7d930d6ce4aef3aa20929", "tests/test_native_learned_memory_training.py": "a673ef8b6a9c33031a4e8f02b7499d0ec3ca055255cfb287b1c36864a0d396f2"}. Current identity-join protocols/criteria and24000GPU-secondcap untouched.


Learned-memory selected-loss CPU443081 passed all9tests/no skips in8allocatedCPU-seconds, including actualtinyQwenSDPA causal gradients and FP16 cast fixture. Rootverifiedcurrent/archived5sources,4installednative sources andtestlog. SummarySHA6a5d2ca9c88c1423d395c89caed797cb55ca31fa5bed495c2cef7eb7307e1b4f. This establishes toy/native-API software behavior only; no pretrainedquantizedmodel/GPU/longreasoning/efficacy claim. Controller/native runtime remain held for their ownchecks.


Freeze descriptive all-model learned-selection plot consumer before final outcomes are inspected: script8c6ff144e627984d7b0ec94325bfd8572c132c3249ef5158f4d21225c702a93c, CPUwrapper9fa5d360a81361d0825e284f46fd4d4749a12d942dd8f8b02d58cf405c66b1fc. Requires completed bound independent report/analysis/outcomes, recounts2592outcomes/24points, all3modes/2seeds inseen/heldN32/N64 panels. Labels uniformover9names11.1% (notoptimalchancebound) and absolute98/108,87/108targetlines; absolute linesdo notimplyrelativecriterion. PNG/PDF,CPU4cores4G300s onlyafterverifiedreport. No result/criterion changes.


Learned-memory native controller CPU gate: root+independent hils review passed. Release one CPU4cores8G300s job with8tinycausal fixtures for all3selectionmodes/pre-post, onecorecallperforward, actualmultiquerycaptures, exactzeroU/localstateidentity, liveFP32delta/nativeFP16castnode, earlier/current/futurelossgradient paths, full-versus-cachedhistory, mutation/overlap/exceptioncleanup andnativeoverflowrejection. No pretrained model/GPU/data/fit/evaluation. ReuseimmutableV11fixtureclassesonly; preserve allsources/results. Freeze {"gnnformer/parallel_local_learned_memory.py": "c74d5d065d59d594f43c920f3f05f25b429e70a3e277311bb0440eb4941eb054", "slurm/native_learned_memory_controller_check.sbatch": "2f51037534efc8138b66caf49a5a7870c13a4fb7259c0b03dbc286d72504895d", "tests/test_parallel_local_learned_memory.py": "b1ade47ef5ac352426803eb05f5c65a6384259c051bad5680841679ec12cc6c2"}. NativeQwen/Cosmos profile remainsheld andseparatefromcurrent6fits.


Learned-memory controller CPU443092 passedall8tests/noskips in2allocatedCPU-seconds. Rootverifiedall8current/archivedsources, ledgerandtestlog. SummarySHAe2bea79cbdeccad92939f45c798ef6aa4bb0b5517de391ab50839358ec5ddad1. Togetherwith9selected-loss replaytests443081,thisestablishesCPUfixturebehavioronly; no actualpretrainedQwen/CosmosnativeKV equivalence ormemory efficacy. Independent nativeprofile sourceisbeingprepared; current6identity-joinfits/criteria unchanged.


Release frozenV2 final independent CPUreport4cores24G900s via afterok dependency onentiremainarray443068. All6run/configidentities checked, including finaltwo443094sigmoid23 and443068softmax23. Itmustverifyeveryfinalcheckpoint/4536updatelog/nativeargmax+EOS/fullrawcapture andall2592test+648dev outcomes, applyunchangedwholefamilybootstrap/criteria, andretainallcosts/failures. No finalscores inspected beforeall6complete. Schedulefrozenall-modelPNG/PDF consumer onlyafterthisreportpasses. No GPU/fit/criterion change.


Native learned-memory software profile prospective freeze: root and independent v10_evaluation final source reviews pass. Common block26 reads, clipped learned selection with a fixed unfitted core, matched pre-block27 versus pre-final-norm writes. Qwen and Cosmos are separately prepared with their own native weights/tokenizers/mRoPE. Use only original training timing cases at N16 and its predetermined N64 extension; no dev/test, fit, accuracy scoring or layer/perturbation search. Seed20261124; zero fixture U/selector weights zero; active fixture U then selector weights Normal(0,.001), other constructor tensors unchanged. Fixed continuations Therefore and colon, plus two natural max8 runs per model. Native forward API remains explicitly unexecuted; native generate API is covered.

Per model:43 fixed VLM/20 vision calls, up to16 additional VLM and2 vision calls,2 differentiable standalone final-block replays, up to63 extra native-head calls. Verify exact zero identity, common lower reads, all28KV layers, local and historical prefix preservation, all-row same-shaped native-head TV<=.02 and exact argmax; cached/full differences remain descriptive. Selected middle-query CE must credit historical pre-block writes, with post-block historical/future/local gradients zero. Fixed sinusoidal perturbation is .1*RMS(common global first query), all later writes/tokens held fixed; restoring only layer27 global first-query K/V must restore later logits exactly. Archive original/applied writes, all native logits, baseline/altered/restored KV banks and independent model-specific position evidence.

Separate software ceiling1200 allocated GPU-seconds INCLUDING failures, at most600/model, two memory GPUs and four combined project GPUs. Prospectively replace the earlier held wait-for-all-six queue suggestion: memory jobs may overlap the final two identity-join fits after the CPU gate passes, provided combined project concurrency stays<=4. This scheduling-only change does not affect either scientific protocol or the identity-join24000-second ceiling. CPU preparation4cores16G600s first; then two GPU profiles only after source/proof checks; independent CPU report4cores16G600s after both successful profiles. Preserve failures; no automatic retry. No efficacy or reasoning-composition claim. Frozen sources: {"gnnformer/parallel_local_learned_memory.py": "c74d5d065d59d594f43c920f3f05f25b429e70a3e277311bb0440eb4941eb054", "scripts/native_learned_memory_runtime.py": "31d269d403b9f67fbe64daf534b6537c9d30fd511c6752ed0916b4e5a2c1aaba", "scripts/native_learned_memory_training.py": "8bc7b41a669eed2cbcbaa58f4b434cd8697f6b5b1b41098ee4bfa11e406358ff", "scripts/profile_native_learned_memory.py": "74f92f1f916e4573e00e3b17b5b241984b3ee41266992e541df98436ac0f3898", "slurm/native_learned_memory_check.sbatch": "1a8efe4d6b81b82911a1ac3f2ec6ebca2022a92617efd55d149b4e48982d9362", "slurm/native_learned_memory_profile.sbatch": "420c4111e1fd3d6a5ba5909ab94677dcf21fc831e044b13823886c27cb9060a7", "slurm/native_learned_memory_report.sbatch": "9abc66472275c196cc7297f9295f611ef4cc929169f008276fd15d7c0228d1b9"}.


Native-memory software CPU443132 passed20 allocated CPU-seconds. Root verified65 current/archived sources,88 input/proof bindings, installed native sources, initial and prepared packets. PlanSHAef9497d4c2bc5af02302b5969fc530b23b77e4baf1c8c32e30f844f83ce6ab18, summarySHA5d76b86b547942b050793ebaac6b77a8087f528d4b3917d7f4b25486dbc7eed1. All six identity-join mains are now complete; release two unchanged Qwen/Cosmos software profiles under600seconds/model and1200 whole software ceiling, maximum2 memory/4 project GPUs. Native generation, persistence and gradients remain unverified until profiles and independent report succeed.


Native-memory software Qwen443135 and Cosmos443134 each completed131 allocated GPU-seconds and passed their fixed runtime checks. Total262GPU-seconds. Release unchanged independent CPU report over both actual profile directories and plan443132; no accuracy claim before/after this software report.


Final learned-selection report443105 passed131CPU-seconds; root verified59 current/archived sources and bound analysis/outcomes/geometry/accounting. SummarySHA6ff5ccedef0f39cb8a6282a6ecc3c0a441b8961544d4209541b1d2eac069ecdf, analysisSHA938216598c2963458b55b041534d9b1cd8adb27d7f2a29acad21b05aab4f8aea. All criteria FAILED. Clip12/108 every cell/both seeds; sigmoid15–18/108; softmax15–31/108. Dev12,15,22 forseed22 and12,15,26 forseed23. Total11508GPU-seconds including all failed attempts, max4. Plot443108 passed2CPU-seconds, all bindings verified and PNG visually inspected. Conditional joint and multiple-request efficacy releases stop as specified. No outcome-based operator selection or model tuning is retroactively accepted. Bounded training-only diagnosis is the next exploratory step; no new fit released yet.


Independent native-memory report443139 FAILED at source line780, Fixed prefix/write coverage differs, after~33workseconds. FailureSHA39b8527d29baf62753a09995ab6fb69269147535726060030993aa1d96f68253. Both GPU profile successes remain but do not constitute accepted independent software verification. Diagnose exact mismatch before any new separately named reporter. Preserve original report/source and allraws. No GPU retry or fit release.


Memory report443139 source/JSON diagnosis rules out missing coverage: eachmodel328 comparisons,43 fixedcalls,37fusion records. Release CPU-only4cores16G600s diagnostic to recompute all saved cached/full metrics, preserve every field discrepancy and count/flag identity, with no tolerances changed. This tests whether host FP64 reductions caused the exact-dictionary assertion. No raw/native/GPU rerun, no profile acceptance. Freeze {"scripts/diagnose_native_learned_memory_report.py": "69a5c48b98a81d1c9bf5cac3eab430026dbe7938ba7f4de66eb0fbd107dd666f", "slurm/native_learned_memory_report_diagnosis.sbatch": "4ec24fcbd48675d9adadbb3d41d79eca0f2e68f01b8e438a921cc3db389c6b81"}.


Prospective training-only optimization diagnostic after the fully reported failed learned-selector campaign. This is exploratory method development, not a retroactive test success or a new extrapolation claim. Fixed 2x2: clip/sigmoid selection crossed with scene-mean native answer CE alone/CE plus unchanged residual consistency coefficient1. Shared fresh seed24, unchanged1,041,697parameter core and native FP16 residual/norm/head,600fixedupdates, AdamW lr.001/wd0,50warmup then cosine to1e-5, gradclip1. All4conditions retained; no hyperparameter/seed/endpoint sweep.

Use18existing TRAINING complete families: six room-pair questions forming the lexicographically first allowed degree2 graph over all six rooms, three disjoint canonical person trios covering nine people, three answer-changing variants, paired N8/N16=108contexts/54pairs. The six selected questions are Bathroom–Bedroom, Bathroom–Garden, Garden–Bedroom, Kitchen–Office, Kitchen–Park, Office–Park (canonical room order retained in actual prompts). First canonical disjoint trio partition; every name12trainingcontexts. Same deterministic shuffled repeated54pair order,8pairs/16scenes perstep crossing epochs until600updates. Existing matching54N16developmentcontexts evaluated once after finalfit; no test trajectory/score/feature lookup. Training full108 and dev54 both use native unmasked max4tokens and exact name+EOS scoring. Short/long-name scene-mean weighting is unchanged and reported.

Diagnostic training_association_fit requires at least103/108 native first-token AND complete-answer correct plus16/18 complete six-context families (allthree variants atbothlengths). ID_generalization separately requires49/54first-token AND complete-answer correct plus16/18complete three-variant devfamilies. Report every denominator/outcome regardless; one small seed is not confirmation. Preplanned gradient diagnostics atsteps1,2,32,128,300,600 record separate CE/consistency gradients/norms/cosines and first/continuation/EOS losses; they do not alter optimizer gradients. Within-mode differences address the objective; within-objective differences address clipping/smoothness. All-arm failure does not demonstrate missing information.

Separate whole ceiling3000 allocated GPU-seconds including allfailures,4projectGPUsmax. Four timingprofiles capped150s each,32updates and9unchanged originaltraining timingcases only; four mains capped600s each. Eachprofile must roundtrip its finalcheckpoint and pass native replay; independent CPUrelease must verify allprofiles and a pooled conservative fit/evaluation projection<=600. If resource gates fail, preserve/stop before mains. Exact source freeze, subset/schedule proofs and fresh CPUchecks are required before profile release. No scientific source is released by this protocol paragraph alone. No joint-control or multiple-request followup is released.


Memory CPU diagnostic443153 passed6allocatedseconds; all656cached/full rows recomputed. ONLY centered_rms differed:2Qwen and4Cosmos values, maximum absolute6.938893903907228e-18, alloneULP. AllTV/top1/boolean/ownership fields exact; existing14Qwen and74Cosmos descriptive failures unchanged (Cosmos includes2top1 differences). DiagnosisSHA597dd9af14ceb5149379caf7cb9acf48cc334b8a284ac0333518de4df6539275; original report443139 failed36allocatedCPU-seconds and remains preserved.

Release a separate independent CPU4cores16G600s precision reporter. Only cached/full descriptive centered_rms comparison allows<=2maxFP64ULPs; all other fields/types exact and every native/head/causal/coverage check unchanged. Root verified copied audit AST is exact apart from that comparison and returned precision evidence. New reporter binds failedreport, diagnosis, all raw/source ancestry and snapshots its own sources separately; embedded13-case boundary/corruption tests run before fullaudit. No GPU/model/head/rerun/fit, and no numerical threshold change. Sources {"scripts/report_native_learned_memory_precision.py": "02d83b8948816bd9b21f285f1c7d18fb76bce83b38a9e378f480314f138ab3c0", "slurm/native_learned_memory_precision_report.sbatch": "a3b58cd1f27e3c48c436583c6211284b8bbaecd8b8eb48ea25abc9f42f6c68a6"}.


Independent native-memory precision report443156 PASSED. SummarySHAf25d86ec48c24943a2a305bbdf9f619ca651339084c2036cb2773e8132b024c7, analysisSHAf25a497da5eb91642cd4c25907088199784e4106c3f0a90f2a185648aa8cdaf4. Root verified65 original and2 reporter current/archivedsources, analysis/resources andbothprofile bindings. All13precision/copy-contract fixtures pass; only6oneULP descriptive RMS differences admitted. Same-state native heads pass, zero/common-read/KV ownership and fixed-token persistence/restoration and actual quantized temporal gradients pass onbothmodels. Total262GPU-seconds;14Qwen/74Cosmos descriptive cached/full failures remain, including2Cosmosargmaxdifferences. This establishes a software mechanism only, with no fitted accuracy/longreasoning/composition claim.


Fixed optimization diagnostic source freeze after root and independent attention reviews. Unchanged prior prospective2x2science. Pooled main projection fixed as maxsetup+1.25*(maxfirst4+596*maxsteady+162*maxT16)+120, eachmaximum acrossall4profiles; maxsteady includes diagnosticstep32 overhead. N16timing covers smallerN8 in this engineering projection, not a physical worst-case guarantee. Allgradientnorm/dot comparisons use savedFP32tensors withCPUFP64reduction reltol1e-6/abstol1e-10; no nativeheadthreshold change. Finalreport independently checks fullnativepredictions and first/allprefixgateclosure. No test outcomes used. Release one CPU4cores16G300s preparation with subset/schedule/resource tests, tiny livegradient/querysegmentation and actualmetadata restricted-load regression. Exact sources {"scripts/diagnose_native_identity_join_optimization.py": "058aedf5863f6c0180f931565a9eb88137b6e83c0aa30a0f7965df95e6a514d9", "scripts/report_native_identity_join_optimization.py": "56eef4deb2781ac565f94b705ff629efd78a152145f34d643b5bd6c76d6edaf6", "slurm/native_identity_join_optimization_check.sbatch": "eb7d4d124a935706885d24a2511d5e26c72b4ce23bfc973c7c2a2fe7a0e3839c", "slurm/native_identity_join_optimization_main.sbatch": "53d3c437c520d72b1d73906a1d367172d2078d68711389fc2275571b87374629", "slurm/native_identity_join_optimization_profile.sbatch": "57c2f2145c2b00da0a4a4ad548926b533bc14789570adaad123c50bb77b6db56", "slurm/native_identity_join_optimization_report.sbatch": "f3df3729c29bf178fc3394c7a60875c891d4766aae81dd4ef3e8945b1f039209"}. Four150secondprofiles remainheld untilCPUproof andsnapshot verification.


Optimization CPU443176 PASSED227allocatedCPU-seconds, all162preparedscene packets and fixedsubset/schedule/resource/gradient/serialization checks. PlanSHAf4b756df418c7f54eae447a7fa9ffbc9f35b56544b1616113ef306d1b1efc275, summarySHA152f19703ff9c1687c1e2382069dae35457d2dd82b1b05cae128eab6954328d0. Root verifiedcurrent/archivedsources andsmallsubset/proofbindings; scheduledworkers rehash every preparedtensor/nativeweight input. Release four unchanged32-step profiles, all4arms seed24,150seconds each,max4projectGPUs. Whole ceiling3000GPU-seconds includes600reservedprofiles+2400mains; no main untilindependentpooledrelease passes. No dev/test generation inprofiles.


Allfour fixed optimization profiles443180clipCE/443181clipR/443182sigmoidCE/443179sigmoidR completed and passed32updates, nativeheadchecks and restrictedcheckpointreload. No dev/testoutcomes. Release unchanged independent CPU4cores16G300s audit ofallfour savedgradient packets/logs/finalweights/nativeninerowtimings andpooledruntime+wholebudget, beforeany600stepmain. Original source/criteria unchanged.


Optimization V1 independent release443185 passed allfour actualinitial/finalcheckpoint/log/objectivegradient/native-timing audits but FAILED the resource gate: pooled751.1270590291824seconds>600. Inputs maxsetup67.2462724884972,first4.6926146941259503,steady.05237823259085417,T16=2.587620913051069. Failure and releaseSHA31663af241d1c0a7e7c1102e30dc6b0807a2ea710e11d8da6d4a9603968d9e9a preserved; no600stepmain attempted. Fourprofile allocations114/113/113/112=452GPU-seconds. The original600second/3000whole resource protocol is stopped.

Prospective resource-only V2 BEFORE any new finalfit/development outcome: same2x2architecture/seed24/108training/54dev/600updates/loss/optimizer/schedule/criteria/gradientsteps, same native evaluations andfixed pooledformula. Setpermain840seconds(14min),whole4500GPU-seconds INCLUDING original452 andallfuturefailures; max4projectGPUs. Reserve452prior+600freshprofiles+3360fourmains=4412<=4500. New source/report/wrapper filenames and output/data/checkpoint roots; preserve allV1sources/results. FreshCPUpreparation andfour150secondprofiles followedbyfresh independentrelease mustpass<=840 beforemains. No scientificcriterion, timingformula or convergence schedule change.


Optimization resource-only V2 exact-diff review passed root and independent attention. Original600update/32profile/seed24/subset/loss/gradientsteps/criteria and596steady-stepprojection allunchanged; onlynewfilenames/roots/imports,840main/4500whole,3360reservation and14minwrapper. GPU JobNames unchanged sooriginal452remaincharged. Freeze {"scripts/diagnose_native_identity_join_optimization_v2.py": "0b3ab2d88eed8d4af59824be5cbebc18da25cef475c72307be44903c31d4a95a", "scripts/report_native_identity_join_optimization_v2.py": "e64bc90b72494239f7bcaef2525b5ecd2541b7af7346028e274dcaa0be1ef3ae", "slurm/native_identity_join_optimization_v2_check.sbatch": "0491cdc27c7f92fac4953286a9f668c25979b1307556b70b3fabe5ac91d71ce3", "slurm/native_identity_join_optimization_v2_main.sbatch": "9e8d70d76dfcd8f933a736319fb89ac771f4d9fcb81de580cf59aa7a32bc1d0a", "slurm/native_identity_join_optimization_v2_profile.sbatch": "df0dc1c9360eb098f70a93af3a6249e1d120103cfee6c593506e478328d2837f", "slurm/native_identity_join_optimization_v2_report.sbatch": "076afaa781cbf15daa6159273450eb429d862b41082236ebc573438a92f9218e"}. Release freshCPU4cores16G300s preparation. NoGPUbeforefreshCPU/source gate. V1release443185 FAILED61allocatedCPU-seconds, preserved.


### Optimization resource-only V2 CPU pass and profile release — 2026-09-11

CPU443189 completed229allocatedseconds, all checks passed. Plan SHA256 f093dfa8798015f473558c64f9be7d37e866ca89ad338c8e1e68da44c217b02e; summary21663475021809cce0f6ca970cceecbb45672b806023643c10393c83680b4c56. Root independently verified all64 current and archived source hashes and all small runtime JSON bindings. Release four training-only32-step profiles, one per fixed arm, maximum4GPUs and150seconds each. PriorV1cost452GPU-seconds remains charged; reservation452+600+3360=4412≤4500. No main fit is released until the independent V2 resource/audit gate passes.


### Optimization V2 profiles complete; independent release audit — 2026-09-11

Fresh profiles443201/443202/443203/443200 completed114/114/113/114GPU-seconds, all32-step computational summaries passed. V2profilecost455; cumulativeincludingV1=907GPU-seconds. IndependentCPUrelease443204 submitted using allfour. No main fit or dev outcome has run.
profile_clip_ce_s24_443201 summarySHA256 7a4e99e9de799b3a83f684a1c0a50c609b95092790fec2d17e8021d181f30e98
profile_clip_consistency_s24_443202 summarySHA256 3bf9fc90ca8977cbde7fa1d4c6549c46561edc0883db1829ad6abc2047df6fc6
profile_sigmoid_ce_s24_443203 summarySHA256 c9815ff42da5c0d2be5c219c655bfd761ab1b4b5f777521e8141f89becbd014f
profile_sigmoid_consistency_s24_443200 summarySHA256 77525832a6567eb4fa358270649e438e478cc153f4451d67bd29792ea2b9b453


### Optimization V2 independent release PASS; four600-step mains released — 2026-09-11

CPU443204 completed60seconds and independently passed allfour native/training/gradient/endpoint audits. Release SHA256 ec8a2ec8abbbdb4828b31383b0130dde9bae6a9ade6982de2f72177e0cc48d0b; summary 1d9caa4ac79b2d5691cc385882780955c96f3f7e0475903ff609080e9f4b51fa. Root verified current/archive sources. Pooled measuredprojection695.7116531033535seconds≤840; cumulative907+3360reserved=4267≤4500GPU-seconds. Release exactlyfour600-step seed24main fits (clip/sigmoid ×CE/CE+consistency),108training and54development final native evaluations each; no test trajectories. Allfour outcomes will be reported regardless criterion; no endpoint selection.


### Optimization V2 main logging failure — 2026-09-11

Allfour mains443208/443209/443210/443207 failed49/50/50/48GPU-seconds (197total) atstep200: the immutable JSON writer rejected a second training.json afterstep100. No finalcheckpoint/nativeevaluation was reached. Profile32 never exercised thisbranch. Preserveallpartialartifacts; cumulativecampaign1104GPU-seconds. SeparateV3 will name each100step progresssnapshot uniquely, retainfinalfilenames, and add a realexclusive-write fixture across100..600plusfinal. No scientificchange; prospective cap5500 includesallpriorfailures andfresh600profile+3360mainreservations, totaling5064≤5500. FreshCPU/profiles/independentrelease required; priorV2failedmains must be exactlybound exceptions to the no-prior-main resourcegate.
run_clip_ce_s24_443208 failureSHA256 bf391c8c915778b2ecfaa7929135e8d921b5d7f7680c25b846b544d526cc214b
run_clip_consistency_s24_443209 failureSHA256 02e749f1e11815ad78ab1232fad5eb48d88afbe6ad7aff8d0d626c6e822683b0
run_sigmoid_ce_s24_443210 failureSHA256 8460f494f9d0eaff308fcf5b3fc2b242b6b2451747d16660faf0e6f3b2e59cd6
run_sigmoid_consistency_s24_443207 failureSHA256 5e48465ef358510d4e7db267328c3b13db1b6e9ab1318f0165a58ea78157973a


### Optimization V3 logging-only repair source freeze and CPU release — 2026-09-11

Root and independent reviewer approved exactdiff fromV2: unique progress snapshots at100..600plusfinal; realexclusivewriter14-file/overwrite-rejection CPUfixture; reporter compares allprogress snapshots with finalprefixes; exacthash-bound fourV2loggingfailures are the only permitted prior mains. Roots/entrypointsv3; inclusivecampaign5500, main840 unchanged. All600steps/seed24/data/optimizer/CE+consistency/nativecriteria unchanged. Freeze below and release oneCPUcheck (4cores16GB300seconds). No freshGPU is released yet.
scripts/diagnose_native_identity_join_optimization_v3.py SHA256 86d60fc47cae05ebc26051fd6732f1fb0be640ac3215093477b2cb1fa9a9eadf
scripts/report_native_identity_join_optimization_v3.py SHA256 676b43e9b3ccdd62aee2a0142c3ad01bf32ac5580dc662a41010578d8c3b6360
slurm/native_identity_join_optimization_v3_check.sbatch SHA256 f7ef47f32c6d0bb7c2c48ef3a379998f43c3261a0fcd6d3b76a8a531297c96a7
slurm/native_identity_join_optimization_v3_profile.sbatch SHA256 87727ac44972bb763d154c73dcc653f13979f746602444027b51a81ce2a17c18
slurm/native_identity_join_optimization_v3_main.sbatch SHA256 962ec75e216e6b95fd0d7274bb449243639426ffef86c229a7cbd618c25174bf
slurm/native_identity_join_optimization_v3_report.sbatch SHA256 f0eadfb06f0cfa7acfe78b93fcaad3717ebd24d9dbe83cec2ac60aae3733f948


### Optimization V3 CPU pass; four profiles released — 2026-09-11

CPU443214 completed226seconds, allchecks including14uniqueprogress/finalfiles andexclusive-overwrite rejection passed. Plan0a3cb1b364e69e8cfaf534eaa5118508d7353610e3c9cf01aba0cbb9e4e28326; summarya618def5fe3d2e7364acdef8eca725b5bb9501b0c32efb5ef50eaa54360d1fba. Root verified64current/archive sources andallsmallruntimeJSONbindings. Releasefour32-step profiles maximum150seconds each,max4GPUs, samefrozenconditions. Inclusiveprior1104+600profiles+3360mains=5064≤5500. MainfitsheldpendingindependentV3release.


### Optimization V3 profiles complete and release audit — 2026-09-11

Profiles443218/443219/443220/443217 completed112/112/111/111GPU-seconds, all32step/nativeprofiles passed. Freshcost446; inclusivecampaign1550GPU-seconds. IndependentCPUrelease443225 submitted onallfour. NoV3mainreleased or pilotnativeefficacyoutcomeobserved.
profile_clip_ce_s24_443218 summarySHA256 ec1698969621308c8743c3a94479beed2f3719a02e3cb0024f80149824dff446
profile_clip_consistency_s24_443219 summarySHA256 4cfd244b504cb2e8c153eb9c7938b0bd3462297b3d5bd27be4b3224c0b4d242e
profile_sigmoid_ce_s24_443220 summarySHA256 f62327040f57ddc825a77d4adfd49ca99ab42bde1952a5968550594f05fb1f82
profile_sigmoid_consistency_s24_443217 summarySHA256 89849d9aded06788caca2afa1681b360cccd21ca9f577c971b70d61c8d8af1d2


### Optimization V3 independent release PASS; main release — 2026-09-11

IndependentCPU443225 passed computational/native/gradient audits, exactfourV2failurebindings andresourcegate. Root verified64current/archive sources. ReleaseSHA256 858641d56eb4eec0207dc7acc5dffcda0dc6ff9ea7f6f1f760ecdc69663bae1e; summary62262015582e2386ccdadaea4122c4e365abe631bf28a0246607c1603fcba98d. Pooledforecast681.9578904360533≤840; inclusive1550+3360reservation=4910≤5500GPU-seconds. Releaseexactlyfour600-stepmains withoriginalseed24/scienceand108train+54devfinalnativeevaluations. No testtrajectory/checkpointselection.


### Training-only feature linear-accessibility assay: source freeze and single CPU release — 2026-09-11

Root and independent source/metadata audit passed the fixedproposal. ReleaseoneCPUjob4cores16GB600seconds, zeroGPU/model/vision/headcalls, nohyperparametersearch. Exactly9007traininglocalemptystates,4472odd-Step fit/4535even-Step evaluation,432disjointimages/side, all648question-person-room cells/side; eachside36288occurrencesreportedseparately. FP32coreRMS thenFP64 centeredridge lambda1 onmean-row/sum15columns loss, unpenalizedintercept, oneactualdatasetCholeskysolve. Fit-onlymajoritycontrols; allpredictionsfromexactrestrictedcheckpointreload. Bothnormal-equationrelative residuals≤1e-8; allinput/label/split inventorieswrittenbeforedatafit; rawstates/multisetcollisionsconfirmedbytensorequality. Strongdiagnosticcriterion: person/room/joint each≥4490/4535, everyquestionjoint≥98%, everyperson-roomjoint≥95%, jointadvantage≥50pp overquestion-onlyfrequency. Allmistakes/strata/ties/costsretained; failedlinearprobe withoutconflictingcollisions neverprovesinformationabsence. Diagnosticlabels/coefficientsneverenterdeployedmodel, and thisreleasesnofurthernativefit. The protocol was chosenbefore anycompletedfour-arm nativeevaluation wasread. Failurepreservesevidenceandstops; noautomaticpenalty/split/solverretry.
scripts/diagnose_native_identity_join_feature_sufficiency.py SHA256 816a16c2fdc1147140582d3ee13e687076d17d9ebeef90a9b9d8ef1cb6733ce8
slurm/native_identity_join_feature_sufficiency.sbatch SHA256 c249b4d78599129acce1b98f3d9710fc5b777dcfcb06fd2fb5c56dd373257d06
docs/paper/NATIVE_AGGREGATION_FEATURE_SUFFICIENCY_PROPOSAL.md SHA256 bfc7ed371e803d230cde4166d90aab33d95afabf58a5f112e2067ce1bdcdb6ec


### Optimization V3 mains complete; independent final audit — 2026-09-11

Mains443227/443228/443229/443226 completed216/215/214/215GPU-seconds, all600updates and108train+54devnativeevaluations passed computationalsummaries. Maincost860; cumulative2410GPU-seconds includingV1/V2profilesandfailedV2mains. Alloutcomeshelduntilindependentreport443233 auditsallfour.
run_clip_ce_s24_443227 summarySHA256 4ecdb1cbeb64d877f5e7f841ce2bfb63c7be409948851a32e4e5b2f49ce7137d
run_clip_consistency_s24_443228 summarySHA256 ba27900e9f521ee65cf021a7fb019c87cc59b5a75c00f0064713f1298ff49ff0
run_sigmoid_ce_s24_443229 summarySHA256 a277d7ad3b15df0f598348b61d53b1d788cfaa95c9793535cb50a939b1a5ca27
run_sigmoid_consistency_s24_443226 summarySHA256 a1a470c222dd7cb8da8aaa064beebdece097a189d00fddc417689dc799fe6e61


### Feature assay completed; numeric outcome and label-table repair — 2026-09-11

CPU443231 completed9seconds;38sourcecopies,12outputsandallsmallinputbindingsverifiedbyroot. Summary01f5ebbbb991c7169f3c23e5a8acaf4014d936beed0e63e692157cc4a1b597e1; metrics9b3ce53b5a303d3fc0a5de6ae4201e111a24c92c370d148dfb9ac1492825d201. Heldperson4535/4535,room/joint4379/4535; strictstrongdiagnosticFAILED. Fitperson4472/4472,room/joint4330/4472. Question-majorityjoint95/4535. Noexactconflictinglocal/scenemultisetgroups. Normal-equationrelative residuals1.77e-15/2.03e-15. Reporttablebug: person/roomstratumlabelkeys overwrittenbymetricdicts; allpredictionsandcoverage retaincorrectlabels. Preserveoriginalfit/source/results. PrepareseparateCPU-onlyreportrepair torederiveallnumericcountsfromboundpredictions, requireexactoriginalmetricagreement, andpublisheverystratumwithseparatelabel/metricnamespaces. No refit, penalty/split/thresholdchange ornative-trainingrelease.


### Feature label-table repair frozen; CPU report released — 2026-09-11

Root and independentreviewpassed pureJSONlabelreportrepair. Bindoriginalsummary01f5eb.../metrics9b3ce.../all12outputsandcheckpointproof; independentlyrecompute9007storedscorepredictions/ties,72576occurrenceweights, fit-onlymajorities,4422stratumrows andalloldnumericmetrics/decisions exactly. Publishseparatelabel/metricnamespaces, preserveoldoutputs. OneCPU4cores16GB60seconds; no torch/tensorload/solver/refit/model/head/GPU; no scientificthresholdchange.
scripts/report_native_identity_join_feature_sufficiency_labels.py SHA256 caeab35bd33a0878f690f9d5d7b97e28d9785414b70ee04541da710cfb26bf98
slurm/native_identity_join_feature_sufficiency_labels.sbatch SHA256 175183f1602bb491645b7319a42aa65a792c0d3fb157c638101a1ce28ca95a2b


### Optimization V3 verified outcome: allfour trainability/ID criteria FAIL — 2026-09-11

Independentreport443233 completed78CPU-seconds, all64sources/currentarchives/outcome/geometrybindingsverified. Summary6c0232cd70908d13872d8b59f22cae3ab1ef7e4ca79b5373e6c030f39d12dcb6; analysis999028973907dab197f46ca5177ecddf89849b33c740bf7998fbaa8971a22fc9; outcomesf2edc0f826e405686e25e1c66d4503a47d7a74cdf23ccb2a9aa825538de3d70b; geometry53a1189768accf2a22407aba1a68ee8c0d9a939f10a37e505b1b31429c7be16e. ClipCE/R12/108train6/54dev,firstallclosed108+54each. SigmoidCE40/108train18/54dev;sigmoidR42/108train18/54dev. Everyfirst-tokencount equalswholecount, everyarm0/18completefamilies bothsplits. AllthresholdsFAIL; cumulative2410GPU-seconds. Fixeduniformcontrolprotocolconditionaltrigger nowmet, sourceimplementationauthorizedforreview; noGPUreleaseuntilfrozenCPU/profile/independentrelease.

Featurelabelreport443240 passed1CPU-second;40sources/outputsverified. Summaryf1ff72eebb87efc58920a3cd07d2067dbd1dcdd6234a165983aab597f72b17bc;labelmetrics48042871a8524852829b5d2e0f02b97684f6cc6198c3c2b54dfe094159d15c43. All9007predictionrows/72576occurrenceowners/4422labelledstrata andoldnumericmetrics/decisionsexact. Originallabelcollisionreport preserved. Norefit/thresholdchange.


### Fixed-uniform join control: source freeze and CPU release — 2026-09-11

The predeclared conditional trigger is met by the completed, independently audited four-arm training failures in report443233. Root and independent source review passed the separate uniform driver, reporter and wrappers. Freeze the sources below. Release one CPU check (four cores,16GB,300seconds), reusing all162 bound V3 prepared scenes and the exact feature/initial/order artifacts; no re-rendering or new model input. The sigmoid runtime has selector weight zero and bias0.5 frozen:97 coordinates remain in the state table but only1,041,600 parameters are trainable. Every valid gate is exactly0.5 and padding is zero. Initial aggregates, preactivations and U gradients must match; the independent reporter also checks all six active first-gradient tensor hashes against the original CE-only run.

Keep seed24,600 updates,108 training/54 development contexts, paired order, native mean-scene/token CE, optimizer/schedule and final unmasked name+EOS generation unchanged. Consistency is diagnostic only with coefficient zero. No local labels, probe readout, new test trajectory, checkpoint selection or reasoning claim. CPU fixtures verify the actual frozen-selector optimizer/autograd mask, nonzero matching initial U gradients through a frozen FP16 toy head, persistent fixed gates, and exclusive progress serialization. One fresh32-step profile at150seconds must pass independent CPU release before the sole840-second main. Separate campaign ceiling990GPU-seconds including failures; maximum one GPU for this control and four project GPUs overall. No GPU is released by this CPU submission.
scripts/diagnose_native_identity_join_uniform.py SHA256 3fae2fd6ad2c3bb60ba72681179b7bc188ca0b23e329696f3af60445799e6330
scripts/report_native_identity_join_uniform.py SHA256 e5b1658c0017fd6ba239a59861e2a52537ea24aadb3fa0433f4b83245fa49fea
slurm/native_identity_join_uniform_check.sbatch SHA256 4b1ccc2377fdb2d9ae6dc91c1041be6c0900ccbe918cc8a89e566861e84177aa
slurm/native_identity_join_uniform_profile.sbatch SHA256 14e6f24e7244c23481824ef851c02717db9c17131e93103d28fe472e5743e66a
slurm/native_identity_join_uniform_main.sbatch SHA256 4c5a63ceb678a87fa97d3ed9deba78f0753f23c789fb078dcd6a3af2ce26801c
slurm/native_identity_join_uniform_report.sbatch SHA256 83580914f4921f4472522972c0688bfe17745a59f51c0e5b4c42af2b2f71e9f3


### Fixed-uniform CPU pass; one profile released — 2026-09-11

CPU443258 passed all checks, including exact initial aggregate/preactivation/nonzero U-gradient equality and the actual frozen-selector optimizer mask. Root verified all70 current/archive sources and small runtime JSON bindings. Planf87d6d49bab87e3baacb436faed88b6561a1dcd3a71b7ccefdebafe2d79c12d9; summaryd01d32e8dc9aba6736dba45e7c2c201f8f41edcfeb6d6cfa16e4f73ce1a7c7fd. Release one32-step GPU profile with the nine original training-derived timing cases,150seconds maximum. Main remains held pending the independent native/training/resource audit. Single-profile plus main reservation150+840=990GPU-seconds; maximum one GPU.


### Fixed-uniform profile complete; independent release audit — 2026-09-11

GPU443261 completed85seconds; the32-step fixed-selector training and nine native timing cases passed. Profile summary75e8e3575f68d36339337fce6d319319bdfe5979b321f7dcc93206a0433ab9af. Independent CPU release443263 submitted. Current85+840mainreservation=925≤990GPU-seconds. No main or development evaluation has run for this control.


### Fixed-uniform independent release PASS; sole main released — 2026-09-11

CPU443263 completed25seconds and passed all native, loss, gradient, frozen-selector, initial-reference and resource audits. Root verified70 current/archive sources. Release3dfe7f8bd2ee3755a49dd8421ca6e9df4b0c2bce4e1dbd740c4c3bcf56450b22; summary971fb62f5d10a58e0f0b2fdf28e5b4586e50dfbca96cfd32ee4cf4d10f0b8439. Actual initial gradients of all six active tensors match the original CE-only reference. Pooled forecast667.1558101777919≤840seconds;85+840reservation=925≤990GPU-seconds. Release exactly one600-step seed24 fixed-half-weight CE-only fit and its final108training+54development native evaluations. No test or reasoning evaluation, selection or extension.


### Fixed-uniform main complete; independent final audit — 2026-09-11

Main443264 completed208GPU-seconds, all600updates and final108training/54development native evaluations passed the computational summary (cb6986bd9c8a4b570c6939242d14c11fa4cfd39253039a23b9fcc959785e55a9). Wholecontrol293GPU-seconds including its85secondprofile. IndependentCPUreport443271 submitted; no outcome claim until audit.


### 2026-09-11 — Uniform control independently completed; oracle diagnostic trigger met

Uniform report443271 completed and passed computational audit (34CPU-seconds): summary SHA1be7e5c8e5c598efd71d996035985e6f8f797dd749e977f4ac7817fc0df424f0; analysis SHAc2176613b6b1ade6eb4c5a13e7ddfe23cb523b7d7709907a312f834bb5a07056; outcomes SHAee292b616649e2623a3a5cbfe3596c558a46e2d921ba78187275dc5431ba3686. Fixed uniform-half CE scores36/108 first/whole training answers and18/54 development answers, zero completefamilies inboth. Both registered scientific criteria FAIL. All600 steps and native captures retainfixed.5gates; allsix initial active gradients match the original clipped CE reference. Inclusive GPUcost293seconds (profile44326185 +main443264208), maxoneGPU, cap990. No extrapolation or reasoning efficacy follows.

The completed valid training failure meets the exact conditional trigger in NATIVE_AGGREGATION_READOUT_DIAGNOSTIC_PROPOSAL.md (SHA15aafed17f2815fc7544581735ca73a8288df97cb38b4fac74f9489a71ce18ec), written before uniform outcomes. Implementation is now released for its oracle native-readout and descriptive uniform geometry diagnostic; nofit until allsources are frozen and CPUcheck passes. Keep its600update/RMS-one post-SiLU code/U-only/nativeFP16/CE-per-full-target-length/final108 first-token criteria and150GPU-second/607norm-headcall limits unchanged. Capture existing final delta, FP16 cast-before-add residual and normalized state without extra GPUcalls. Before fitting, specify an independent CPU final-logit replay ofall108 rows in the same seven batch sizes; full-vocabulary TV<=.02 and exact argmax are software consistency gates, while norm/FP32 delta differences remain descriptive. This cached head diagnostic makes no fullVLM numerical-equivalence claim. No readout/geometry result automatically releases a new architecture or efficacy claim.

Before source freeze or any oracle GPU fit: strengthen the U-to-capture software binding using the oracle code's single active coordinate. Require CUDA matmul TF32 disabled. Independently compute each delta as the saved U column times the saved FP32 code amplitude, using FP64 product then FP32 cast, and require abs(error)<=1e-7+1e-6*abs(reference) coordinatewise. This fixed software tolerance does not change the native .02-TV/exact-argmax gates, fit criteria, amplitudes, objective, parameters, calls, or budget. General FP32 GEMM/norm recomputation discrepancies remain descriptive.

### 2026-09-11 — Oracle readout diagnostic source freeze and CPU release

Two independent source/API reviews pass. Freeze the following new sources (the proposal remains unchanged); the CPU plan binds all70uniform ancestors plus these7entries. CPUcheck60s/4cores/16G nowreleased; GPU150s remains conditional on a passed sourcecheck. No retry/profile or extra GPUattempt is released. Reporter validates600logs,607observed norm/headcalls,108 finalfirsttokens,all18families,actualnativeweights and saved U/code/cast path. Separate7CPUheadreplays use fixed .02TV/exactargmax; geometry uses108trainingcaptures only.

- `scripts/diagnose_native_identity_join_readout.py` SHA256 `48d3679227ee16c735a3ba253df6554bd6f77f5e1babedb70866e30adb4752cf`
- `scripts/report_native_identity_join_readout.py` SHA256 `d120a9422bb96c230c625c6e5fd3c2383888575afe3f6e610d820b61bded1196`
- `scripts/analyze_native_identity_join_uniform_geometry.py` SHA256 `7b3c4f90f4f3573b8487242c2c47dd1f6f68709264befce4a8ff25cb9baccf3c`
- `slurm/native_identity_join_readout_check.sbatch` SHA256 `c8ea3b7b7b949daabcca73e5ddc5da45e471e2f8666a0bb0bd3839399e87bd7c`
- `slurm/native_identity_join_readout_run.sbatch` SHA256 `e195ebbcbc356a9c1f63ffca0a41af3dd62713ec478fe81c56e4cdfdc1f00b0d`
- `slurm/native_identity_join_readout_report.sbatch` SHA256 `211cb9b8e0f85762e1f841dd96f0f24ead488bb214805b6b76cc32942265eaad`
- `docs/paper/NATIVE_AGGREGATION_READOUT_DIAGNOSTIC_PROPOSAL.md` SHA256 `15aafed17f2815fc7544581735ca73a8288df97cb38b4fac74f9489a71ce18ec`

CPUcheck 443294 submitted.

CPUcheck443294 PASSED10allocatedCPU-seconds; all14small softwarefixturegroups pass. Plan SHAcb805a97882e7cd97b18993dde742bee82031ae90f1f10fde1a31ca4867e7edc; summary SHAa1e9b1cd92f64145f238d087d3147a0a853e161982e71320eafb6d7daab49d13. Root verifiedall77current/archive sources and consumed-input bindings. Release the one150second head-onlyGPUattempt, unchanged600updates/final108 firsttokens; GPUcalls607norm/head andzeroVLM/vision.

OracleGPUjob443295 submitted.

OracleGPU443295 completed20allocatedGPU-seconds (15.98softwareseconds),600updates and607norm/headcalls,0VLM/vision. Preliminarysaved first-token screen108/108 and18/18families; independentreport notyetcomplete. Runsummary SHA60b8fe719b71d3c3cd8774abc4d360acf407345a8aa98ed7adad01d45de71bfe. ReleaseoneCPUreport<=300s/4cores/16G for allinput/endpoint/logit/replay audits and fulluniformgeometry; no additionalGPUattempt.

Independentoracle/geometryreport443297 submitted.

Independentreport443297 FAILED9CPU-seconds at the saved-prediction rescore assertion, before nativeCPUreplay/geometry. FailureSHA6f06a5c8653a385bb8bf36d3792cc757f622d3732b1079eb0525ccd824d37f78. AllrawGPUresults/source/checkpoints remainimmutable; noacceptedreadoutresult yet. Separatelyrelease a boundedCPU-only rescore diagnosis (<=60s,4cores,16G) to distinguish exactmetadata/argmax discrepancies from FP32 cross-entropy reduction differences using FP64reference, preservingall108rows andoriginalcheck. Nohead/VLM/vision/optimizer calls, noGPUattempt, noautomaticreportretry or tolerancechange. Newdiagnostic source mustbefrozen/reviewedbefore submission.

Rescore diagnostic source frozen after root review, no change to original comparator. All108 metadata/argmax and FP32full108/original7batch/FP64loss comparisons retained, zero norm/head/model/optimizer calls, CPU60s/4cores/16G.
- `scripts/diagnose_native_identity_join_readout_rescore.py` SHA256 `ce8b42b7559c513cd994df7eebe3c30ad975416669758e7b0ba5d2bfc2fd5e94`
- `slurm/native_identity_join_readout_rescore.sbatch` SHA256 `c698668086620dd993dd60fcdf041537277c5e3680e1c5874f9151c95bde8f4f`

RescoreCPUjob443302 submitted.

### 2026-09-11 — Readout NLL precision diagnosis and separate report repair

CPUdiagnosis443302 PASSED2allocatedCPU-seconds; summary SHAdbe0c1e367ccd24cc7188a4ad8298ee7f1c0be97fcdb0abc514aca7d87b6e5f9. All108metadata/argmax/correctness records and7rawbatches agreeexactly. All108 originalCPUFP32 NLL comparisons fail; CPUfull108/original7batch results agreeexactly. MaxCPUFP32-vsGPUFP32 error1.5065074e-5; GPUFP32-vs-stableFP64 error4.2013668e-7; CPUFP32-vs-FP64 error1.5362049e-5. This identifies reporting reduction precision, not a changed target/prediction. Preserve failedreport443297 andallsources/results.

Release a separate CPUprecision report (300s/4cores/16G), using stableFP64 logsumexp ofexactrawFP16logits forNLL. Crucially, retain the originalNLL comparison atol2e-6/rtol2e-6 unchanged, along with everytraininglog/label/nativeTV/.02/top1/Ucolumn/cast/endpoint/source/resourcegate. NoGPUfit/inference or datachange. RetainoriginalCPUFP32/GPUFP32/FP64 NLLvalues andfailedflags. The new81entry reporting union is separate from the immutable77entry originalscientific source. The correctedreportmustpass independently beforeoracleacceptance.
- `scripts/report_native_identity_join_readout_precision.py` SHA256 `eaada3e9682a6bece5b0bbb8e1373ddc46d4df319f53a18173df1bd1e359c2fa`
- `slurm/native_identity_join_readout_precision_report.sbatch` SHA256 `dd8059c2c93ddb5d34f24752eefd0368f324801c4cae97e88bc608cb53d3465b`

SeparateprecisionCPUreport443304 submitted.

### 2026-09-11 — Oracle readout independently accepted; conditioning design

Separateprecisionreport443304 PASSED14CPU-seconds: summary SHA cb280b2bf6d769885ee20f395a6adaeed0aaec09ef00a5899258d6049b5490c7, analysis936c5a9fee35bda4f67df0c6e20e5ef64055dbaea6ed94cd33edfcae1feabe82. Finalsupplied-codefirsttokens108/108,18/18families; allunchangedgates passed, CPUheadmaxTV4.6212335e-5/top1108, Ucolumnproduct/cast exact. Originalfailedreport443297 anddiagnosis443302 preserved; NLLreferenceonly changedtoFP64, originaltoleranceunchanged. Total20GPU-seconds,607norm/headcalls,0VLM/vision. Rootverifiedall81current/archive reporting sources and outputbindings.

Uniformgeometry83d1a711660bc3e522daba3f0d5557df0e9882eba02843b1c9ad3df3ad088a5e covers1296occurrences,108variantcontrasts,54lengthpairs. Across6questions saturation96.93–98.22%; commonmeanrownorm181.57–190.17, deviation23.11–25.41. Medianpooled answer-changingcontrast.21343 versus medianinsertedhalf-sumnorm38.00585; descriptive, notcausal or directlyidenticalperturbations. Noinformationceiling.

Explicitnewdecision: implement exactlyone fixed combinedinput-centering/scaling control as specifiedinNATIVE_AGGREGATION_CONDITIONING_PROPOSAL.md. Its600stepfullsequenceCE andinitialparameterbytes matchuniform; onlylocalalreadyRMSinputschange. Head-onlyfirststage avoidsnewVLMexecutionforanuntrainablefit; itsfixedfirsttokenscreen is a resourcegate, notnativewholeanswerevidence. NoGPUfituntilnewsourcefreeze+CPUcheck. No architecture/mainstudy automatically follows.

### 2026-09-11 — Pre-freeze amendment: parallel global/question conditioning

Before source freeze, CPU statistics or either fit, replace the initial single question-mean arm with exactly two parallel arms: global versus question-specific mean subtraction, using the SAME global pooled scale in both. The global arm has no deployment question lookup and its affine operation is mathematically foldable into existing local W/b, although no conversion is performed yet. This directly tests a deployable standard-conditioning counterpart alongside the richer lookup diagnostic. Shared scale isolates mean granularity; all original training/data/parameter/600step details and first-token resource screen remain. Two90s GPU attempts, atmost2projectGPUs, inclusive180GPU-second ceiling; oneCPU90s check and onejointCPU300s report. No further normalization variants. Revised proposal SHA4e638f3473b4f4eab55291493a1ebd835014657ee5c3bd1175c27d291998f49d. The earlier one-arm decision is superseded before any experimental outcome or frozen source.

### 2026-09-11 — Two-arm conditioning source freeze and CPU release

Root and independent agent reviews pass for the completed driver/reporter/wrappers. Freeze these6newentries plus81immutableancestors; oneCPUcheck<=90s/4cores/16G nowreleased. GPUattempts remain conditional onpassedCPUcheck. Global andquestionmeans shareglobalpooledscale; actualfunctionallocalcapturechecks1e-4, nativeCPUheadreplayonly7finalbatches/108rowsperarm withTV.02andexactargmax;6trainingcaptures retainfunctionalandFP64NLLcheckswithoutadditionalheads. Nootherprotocolchanges.
- `scripts/diagnose_native_identity_join_conditioning.py` SHA256 `97663e1eef3e5a56bd2b6919abf13083f9d73ae10c9a9610b9d453e86301c38f`
- `scripts/report_native_identity_join_conditioning.py` SHA256 `aa28b28426fe895ce22c5aadcdc2727893cb65c8236eaed088df737ea6aba8af`
- `slurm/native_identity_join_conditioning_check.sbatch` SHA256 `feee38ab1151ea8c884b72913cbfa7150a377f6932d8d2a01d2cd7203c791f7c`
- `slurm/native_identity_join_conditioning_run.sbatch` SHA256 `d6607f290b2911370389fc3b0cf4ea78b83a96d2c0908038a49b972d86eed5f2`
- `slurm/native_identity_join_conditioning_report.sbatch` SHA256 `3f3a8864b0cfd1fd80ff4624f8988c944503efa7300e53da07b9980ba5b33f3c`
- `docs/paper/NATIVE_AGGREGATION_CONDITIONING_PROPOSAL.md` SHA256 `4e638f3473b4f4eab55291493a1ebd835014657ee5c3bd1175c27d291998f49d`

ConditioningCPUcheck443321 submitted.

### 2026-09-11 — Preserved conditioning CPU fingerprint failure; software-only V2

CPUcheck443321 FAILED10allocatedCPU-seconds while constructing statistical tensor metadata: the frozen shared fingerprint helper cannot reinterpret a zero-dimensional floating tensor as byte elements of a different size. FailureSHAe270ccf7df158e2acbe1b63d694f6173f2e68ce7d6b54a6fcc0f888922d8ba95. No plan, GPUfit, outcome, or scientific screen was produced. Originalstatistics/files/source remainimmutable.

Release preparation of a separateV2 driver/reporter and threewrappers, binding thisfailure andtheoriginal87sources. Onlyfingerprinting changes: preserveactualoriginalshape/dtype, flattenacontiguouscopy before byte reinterpretation/hash. Add scalarFP32/FP64 fixtures and non-scalar identity versusoriginalhelper. Statisticsremain0Dscale withidenticalmean/varianceformula, inputs/order/init/optimizer/criteria/calls/budgets unchanged. New _conditioning_v2 data/checkpoint/output roots preserve the failedattempt. Require freshCPUcheck90s andsourceaudit before eitherGPUfit; totalGPUcap180 unchanged (zeroGPUspent). No additionalscientific variant.

### 2026-09-11 — Conditioning V2 software repair freeze

Root and two independent targeted reviews passed. All87 original sources match; new5 produce92-source union. Only scalar-aware hashing, fixtures, versioned references/paths/job names, and original443321 failure binding differ. Science and180GPU-second budget unchanged. Fresh CPU90s check released; GPU conditional on pass.
- `scripts/diagnose_native_identity_join_conditioning_v2.py` SHA256 `aaf55a976732b5c81ea9fa1897a63e17fac3ead3167bb666dcd2034e51b81050`
- `scripts/report_native_identity_join_conditioning_v2.py` SHA256 `eb1134278848dab49b11d698b95a03c1880358389eee0324cd83255a9c5aa60e`
- `slurm/native_identity_join_conditioning_v2_check.sbatch` SHA256 `d5ceedf31c67a9d2ccc5563067cfb445f563d9010dab5a2501ec116f4e796591`
- `slurm/native_identity_join_conditioning_v2_run.sbatch` SHA256 `67544eef024d37be7b6a9c58cb90f843d961f72139c5ca4afb0100e4187d4ff5`
- `slurm/native_identity_join_conditioning_v2_report.sbatch` SHA256 `32d11d743f7ebafe2d11bd1b409bd6f288177de772e787b673aae2c357d7675b`

ConditioningV2 CPUcheck443335 submitted.

ConditioningV2 CPU443335 PASSED10allocatedseconds; planSHA5fd647817756f643ccd2b265de2b44dbc2c43d90c43d6902ecd4d14b73fc644d. All92 source/current/archive bindings verified by root. Release exactlytwo parallel90sGPU fits global/question, total180GPU-s, unchanged fixed600steps; no extensions/checkpoint selection.
ConditioningV2 global GPUjob443340 submitted.
ConditioningV2 question GPUjob443341 submitted.

Bothfixedconditioningfitscompleted29GPU-secondseach(58total); pendingindependentaudit rawfirsttokens global64/108/question56/108, both0/18completefamilies; bothbelowcachedscreen. CPUjointreport443351 submitted. No native evaluations or extensions released.

### 2026-09-11 — Conditioning comparison independently completed

CPUreport443351 PASSED19s; summary134af48d0e3be92d8fd542fa01c6eea0da4cc9fd85bd03e688ab87cf16b69012, analysis37904d35314ab4cf04afdf41241c9cf03fc0bff91e220966d565ae624c65c77c. BothscreensFAIL(global64/108,question56/108,both0/18families). Rootverifiedall92current/archive sources and outcome/NLL/geometry/strata bindings. MaxCPUheadTV.003256/.003819,allargmaxexact.58GPU-s,max2. No native or fit extension. Stop normalization variants; release preparation of one matched product/additive binding comparison, source/CPU checks required before fitting.

### 2026-09-11 — Matched within-image factor interaction diagnostic

Before any new fit, prepare exactly two arms: coordinate-wise product versus nonlinear additive factor messages. Proposal SHA2565932b65f4e80ebb878be28e5dcf1ce7cc1a991b47b8319ca53598031a335aa93. Each uses two learned96-dimensional tanh factors, identical newly initialized parameter bytes,1385760 trainable/1385857 retained parameters, and96 pooled coordinates. Both reuse the exact fixed GLOBAL conditioning tensors and original108 training contexts/order from443335. No new normalization or semantic factor labels. Original600-step full-name-plus-EOS CE recipe and103/108-plus16family cached screen unchanged. One predetermined final product diagnostic rolls factor_b byone amongvaliditems perquery; additive algebraic invariance is independently checked. This intervention never selects checkpoints or changes the paired screen. GPU caps90s/arm,180total,max2; CPUcheck90s/report300s. New99-source union requires review andCPU pass before GPU release. Neither fit implies extrapolation, newattention, increasedrepresentationwidth or reasoningcomposition. Bilinear pooling is explicitly acknowledged as priorart. No further work on the failed conditioning fits is released.

### 2026-09-11 — Matched factor-binding source freeze and CPU release

Root reviewed completed core/driver/reporter/wrappers; independent cross-review passed core/driver and reporting computational path. Freeze99sourceunion:92unchangedancestors plus7newfiles. CPU90scheck nowreleased; two90sGPUfits remain conditional on CPU pass. No science changes from frozenproposal.
- `gnnformer/parallel_local_factor_binding.py` SHA256 `7061f833e0292925ee298e6474fde4c3e6af1e49b7222dbc992fde722e78eb0a`
- `scripts/diagnose_native_identity_join_factor_binding.py` SHA256 `ccd83447f9fac3f4079b6223489b8322f07263e1e4296133a9b7556a426803d0`
- `scripts/report_native_identity_join_factor_binding.py` SHA256 `75ea93d2210ea675fe117c629f87eef08bf2e6c372b3571e1ad91ee0b0c25540`
- `slurm/native_identity_join_factor_binding_check.sbatch` SHA256 `4d3895b3666916e6528c02909bb4e8e8b331d396237c250212b57176de547f01`
- `slurm/native_identity_join_factor_binding_run.sbatch` SHA256 `4d7d7839759f4171fccb912a2fa8643ff292dcfebfc2db2fb07f44dc02f7d449`
- `slurm/native_identity_join_factor_binding_report.sbatch` SHA256 `5372eb876e892bf3277658c367524e40b2ac81b51341b32fba61d35d6d3a0f5d`
- `docs/paper/NATIVE_AGGREGATION_FACTOR_BINDING_PROPOSAL.md` SHA256 `5932b65f4e80ebb878be28e5dcf1ce7cc1a991b47b8319ca53598031a335aa93`

Factor-binding CPUcheck443373 submitted.

Factor CPU443373 PASSED12allocatedseconds; planSHAaa9a44067f86ce6340d56419b5efe9411ccd84cb52115f0b31d10745b4e68a07. Rootverifiedall99current/archive sources. Two independentfinalreportreviews alsoPASS75ea93d2. Release exactlytwo90sGPU fits product/additive, fixed600steps,total180GPU-s,max2, no extensions/checkpointselection.
Factor product GPUjob443375 submitted.
Factor additive GPUjob443376 submitted.

Factor fits completed: product44337528GPU-s/additive44337627GPU-s,total55. Pendingindependentaudit, paired product79/108with5/18families versus additive58/108with1/18; bothscreensFAIL. JointCPUreport443383 submitted. No native evaluation or extension released.

### 2026-09-11 — Factor binding independently completed; geometry only

CPUreport443383 PASSED24s; summary de38a1498115bc835955cf264a609ead1a095bacb0c3faa99eba10a1ae962238, analysis326db6392f648a517e169b206ca2c46b25b28f1d7f2faf3da31a314afcb74f94. Rootverifiedall99current/archive sources pluspaired/permutedoutcome/strata bindings. Product79/108and5/18families versusadditive58/108and1/18; bothpairedscreensFAIL. Fixedproductfactorpermutation38/108and0/18families; meanNLL.868483→1.402758. All324CPUheadargmaxmatch, maxTV.003827726; GPU55s,max2. This supports dependence on within-item pairing on these training examples, not semanticfactorization, adequateaggregation, extrapolation or reasoning. No native evaluation or fitted-checkpoint continuation.

Release implementation of one CPU-only query-contribution geometry diagnostic, followed bysource review/freeze andoneCPU60s/4cores/16G allocation. Consume26existing pairedcaptures (6training+7final perarm), savedweights andfrozenconditionedinputs; no new core/norm/head/VLM/vision/optimizer execution. Recompute eachfactor's affinepreactivation, decompose explicitq,localbias andmeanlocalprojection, retain signed dot/cosine terms, centeredlocalvariation andactual versus pre−q tanhderivative/saturation at fixed.01 threshold. Retain perquery/factorrows and finalquestion/N/family summaries; sixtrainingbatches are descriptive samples, notcontrolledlearning trajectories. This does not measure accuracy after removingq or provecausality. A furtherfit is notyetreleased. Largeqnormalonewillnotjustify a newablation; anydecisionmust use specificsigned/saturationevidence. New2files plus99ancestors=101sources; no furthernormalizationvariants.

### 2026-09-11 — Query geometry source freeze and CPU release

Rootreview passes completed101sourceunion. Directquery-removal uses independentlyrecomputedFP32W*x+b; bothactual andcounterfactualtanh/derivatives computedinFP64. CapturedFP32factor proxy and roundedpre−q difference retaineddescriptively. Includes per-factorpayloadpartials (product(1−u²)v, additive.5(1−u²)) toavoidconflatingtanhederivativewithfullproductJacobian; excludesouter.5SUMweight, no lossgradientclaim. Allvalidqueries/prefixes/targetroles retained, fourcancellation/saturation/Jacobian fixtures. CPU60s/4cores/16Gnowreleased; noGPU/newfit.
- `scripts/analyze_native_identity_join_factor_query_geometry.py` SHA256 `5d094c3b158eba5775ee76b82ba7fbaae3c6c00d01f6ff138877faa93035cc2c`
- `slurm/native_identity_join_factor_query_geometry.sbatch` SHA256 `0e9fc07074b58ff8cff1830616ab3cf870aa8dfa9f143cdde346c1247d3b219a`

Factorquerygeometry CPUjob443401 submitted.

### 2026-09-11 — Query geometry accepted; one query-placement ablation

CPUgeometry443401 PASSED4s; summary7f2b8b6592c840c459f1edb97a97b257a1cd96339e03334da74c03518c1a7147, rowsb74f1c061155768bdbaaa2cbbb9e10963e10db20e7dfa8a7ef831a6dc79f2da5, strata13281ced5900d3a8d6ac08ee3de217df1d4185aad9de9f04dfb94d5a7e805b81. Rootverified101current/archive sources andalloutputfiles; independentagent reproducedfinalscalarstrata andconfirmedall216productscene/factorrows saturation decreases andpayloadpartialmean/RMSincreases withoutq. All24question×N×factor cellsagree. Productfinalsaturation94.5/94.4%→72.6/76.6%; partialmean.0112/.0117→.0881/.0710. Signedq contributionpositiveeveryrow. Earlysampledsteps1/2 mostlyunsaturated andpartials slightlydecrease(.92–.96x) withoutq; no claiminitialsaturationorcausaltrajectory. Nohead/core/VLM/optimizer calls.

Release preparation of exactlyone product query-placement ablation: removeexplicitqinsidebothfactors,retainqafterSUM. ReuseexactUNFITTEDinitialpacketfrom443373, fixedglobalstats/order/600CE and103/108+16familyscreen. Equalparameters/zeroUinitialoutput; internalaggregates/firstUgradients differ. One90sGPU attempt afterCPU/sourcechecks; same7cyclicdiagnosticbatches; no otherarchitecturevariants ornormalizers. Comparefrozenoriginalproduct443375 descriptively. A failedscreenendsthisarchitecturebranch; passing releasesnative/fresh-family preparation. No furtherfit is yetlaunched. Fullproposal NATIVE_AGGREGATION_READOUT_QUERY_PROPOSAL.md.

### 2026-09-11 — Readout-only-query source freeze and CPU release

Root and independent reviews pass completed core/driver/reporter/wrappers. All101ancestors unchanged; freeze108sourceunion with7newfiles below. CPU90scheck nowreleased. The single90sGPU attempt remains conditional onpass. Unfittedinitialbytes, dataset/order/globalstats/fullCE/614calls21550rows unchanged fromparentproduct; explicitqinsidefactors isonlycomputationalchange. Originalparentisbound withoutrefitting orheadrerun.
- `gnnformer/parallel_local_readout_query_binding.py` SHA256 `e1259ed6214336561e68f6060846a61d830433bcf0deadde0fb63b3a6fb9e956`
- `scripts/diagnose_native_identity_join_readout_query.py` SHA256 `50b863fbdce14e22a1ef2abd87903744f95e1118ecce0add46f2a0c7ea27c8cb`
- `scripts/report_native_identity_join_readout_query.py` SHA256 `3de6ef3529ce079a4eaab4d9395e79fadc12f5bb28ddc8045f6a47b1a4a20de2`
- `slurm/native_identity_join_readout_query_check.sbatch` SHA256 `91b65344ef6d6ebaf118fbf66d46b228499b632d153cf8b619b6536768a30206`
- `slurm/native_identity_join_readout_query_run.sbatch` SHA256 `b334583fbd85f615ea8a4a338a9ac20dc0447b71440b93dcf475f449bef6123a`
- `slurm/native_identity_join_readout_query_report.sbatch` SHA256 `d7c1b47e4bf567e60bc1c92e6412ff1c8a41f8fb10088f0ac078cf618d628725`
- `docs/paper/NATIVE_AGGREGATION_READOUT_QUERY_PROPOSAL.md` SHA256 `04f45b2748d9150e9d94594bb2ceb6510b49d63f6191b17b4c56c43ca774ef02`

ReadoutqueryCPUcheck443413 submitted.

ReadoutqueryCPU443413 PASSED13s; planSHA92b71da44c0918848fecc16d84573050d71ae1b8e59257353d13478c534b8b6e. Rootverifiedall108current/archive sources; completeindependentreviewspass. Releaseexactlyone90sGPUattempt,600updates,614norm/headcalls21550rows. No extension/checkpointselection.

ReadoutqueryGPUjob443416 submitted.

ReadoutqueryGPU443416completed28s; provisional80/108and5/18families versusfrozenparent79/108and5/18; screenFAIL. CPUreport443418 submitted. No native evaluation, continuation, or furtherarchitecturevariant released.

### 2026-09-11 — readout-query negative accepted; architecture branch closed

Independent CPUreport443418 COMPLETED0:0 in20seconds. SummarySHA `fea7ecb1a2f71a943aa126e1f395163a91e2f7ca8bc189c8a2d4ebb1374733a2`; analysisSHA `06f544d0e046be13dda1dd13aa2c41f2cafd4e91f707858bac2f687dd9cc2235`. Root verified108 current/archive source hashes and24 small output bindings. Paired80/108 and5/18families; cyclic11/108 and0/18; parentpaired79/108 and5/18. No screen pass. GPU443416 consumed28seconds, maximum1GPU;14 independentCPUhead batches216rows have exacttop1 and maxTV.003819364123046398. Prospective stop applies: close factor/query-placement architecture branch; no further variants, failed-checkpoint continuation or native evaluation. All failures retained. Training JSONreview SHA `65757042bd6716627c115f712a607979476c5134b138d73524c890ca52dea07f` finds small residual decline in last10full cycles under decayedLR; resource-screen failure is not convergence or a capacity ceiling. A local semantic joint-code positive control is separately being specified, not a continuation or efficacy result.

### 2026-09-11 — prospective local joint-code positive control

Separate diagnostic protocol: `docs/paper/NATIVE_AGGREGATION_JOINT_CODE_ORACLE_PROPOSAL.md`, SHA `1db9fbd3928b0c076643347d0ac05a65f361bb5765f62192054dd1f33bc7811d`. Each image supplies raw one-hot person-by-requested-room semantics in18of96coordinates, orzero outside requested rooms. No whole-scene intersection or answer enters local codes; the readout must learn the cross-image decision. Fixed half-SUM and the same query/SiLU/native norm/head readout; exactly697440 trainable parameters copied from four original UNFITTED443373 tensors. Same108training/54pairs/600CE/order/seed24,607GPUheadcalls21442rows; six training captures andseven final first-query batches. Unchanged103/108and16/18family screen. This is privileged local-semantic training diagnosis, not a vision/generalization/reasoning result. Source review pending before CPU preparation; CPUcheck90s, oneGPUattempt90s/max1, CPUreport300s. No follow-on fits or continuation of failed branches are released. Numerical/tensor/image work remains exclusively Slurm.

### 2026-09-11 — local joint-code source release

All seven new files frozen in `outputs/native_aggregation_vlm/identity_join_joint_code/source_release.json`;99 inherited current/archive sources verified against plan443373. Core `3ebcebb62df088bfdbe07042e9b1bd5b87c003dd7108c8c0b506469237794690`, driver `8266e207f181b8aa046b43e7e6fc4686a2a340c19a35d65f82caaecc9f2b8630`, reporter `6fec9122f7f5377d2c94f21fc3434381615d07eb6cfd25d2ca92f4435cdef2da`. Root full-source review and independent core/driver/reporter reviews passed; AST/bash syntax checks passed. Release one90secondCPU preparation. GPU fit remains conditional on passed preparation and root artifact verification; fixed90second single attempt/max1GPU, then300secondCPU report. No new efficacy claim.

### 2026-09-11 — local joint-code CPU gate passed; single fit released

CPU443472 COMPLETED0:0 in12seconds; planSHA `f8d11406cbf1e2766bbde03859b45ba15148b5429804257b060f28c4a5bdce63`, summarySHA `d8beabbc354d99311fed967d8501bf3d4624483cbacf99f43b045cd159018a93`. All core/reporter and semantic independence/prefix fixtures passed;108contexts,1296localoccurrences,72frozen global states. Root verified7 new and99 inherited current/archive sources plus6 small runtime bindings. Release exactly one90second GPU fit, seed24/600CE/four unfitted readout tensors; no scene-intersection input and no native inference claim.

### 2026-09-11 — local joint-code run complete; provisional screen failure

GPU443474 COMPLETED0:0 in24seconds. Software summary passed, but fixed first-token screen FAILED38/108and0/18families. All600 CE updates and607core/norm/head calls completed; no VLM/vision calls. This sufficient per-image-code control did not fit within the original recipe, so local representation learning cannot yet be singled out as the bottleneck. Submit the preregistered independent300secondCPU report; no contrastive or encoder experiment is released by this outcome.

### 2026-09-11 — local joint-code negative independently accepted

CPUreport443476 COMPLETED0:0 in15seconds; summarySHA `a742e98875b6f48a23ea5dba147b22c326bb25450df7542a883f1d5c23870b45`, analysisSHA `5fff5aff6e8ee4bdbd3ff920b3db4490c5c6f5536f3fd04ac3694f14ece0fc74`. Root verified7 new and99 inherited current/archive sources and8 small result bindings. Screen38/108,0/18; mean first-query NLL1.38860763. CPU21? No: exactly7 final native head batches/108rows, exactargmax and maxTV.003443524707. All13 captures and1296 local-code occurrences passed. GPU24seconds,607core/norm/head calls/21442rows, zeroVLM/vision. Loss review58549dc0854fce419fe91c93d84e749551e7c8377028a31fcc527c4cbefab269 shows first-token loss remains dominant and no convergence certificate. Do not blame only local native representation access. No contrastive/encoder/native follow-on is released. A fresh, separately preregistered extended horizon and annealing diagnostic is being prepared, with the exact same oracle architecture and inputs.

### 2026-09-11 — prospective unchanged-oracle extended optimization control

Protocol `docs/paper/NATIVE_AGGREGATION_JOINT_CODE_LONG_PROPOSAL.md`, SHA `e0f37bab51d6ace9882093c0efe26d69e06353e407adbb43ab40dd87429b4ff8`. Fresh original unfitted4-tensor oracle, exactly the same inputs/readout/CE/optimizer/batch8/seed24; only horizon6000 and its cosine annealing endpoint change.48000 persistent-shuffled pair presentations with exactold4800prefix. Fixed after-update endpoint observations600/2000/6000; only6000screen is decisive. Eight pre-update captures plus21 endpoint batches;6021 core/norm/head calls and CPU-derived training head rows+324. CPU90/GPU180singleattemptmax1/CPUreport300. Source review pending before preparation. No extra model, native inference, contrastive fit, encoder change, or failed-checkpoint continuation. Failure closes further budget/schedule sweeps for this control.

### 2026-09-11 — algebraic capacity observation, no numerical experiment

Root derived and hils independently checked the construction in `docs/paper/NATIVE_AGGREGATION_JOINT_CODE_CAPACITY_NOTE.md` (SHA `a65dc9db0a47087c2a45659daa59bd011849c00b76982c03aedf48354cccc625`). For each valid pair of half-counts, four SiLU coordinates implement the requested-room conjunction exactly in real arithmetic; nine people require36 of96 coordinates. Using the prior answer-code oracle up-projection gives an algebraically equal residual on this restricted domain. This establishes an abstract width witness for the privileged readout, not optimization accessibility, finite-precision equivalence, native generation, learned visual coding or generalization. No tensor/head call, optimizer, job or constructed initialization is released. The6000-step optimization control still starts from the original unfitted random readout.

### 2026-09-11 — unchanged-oracle long-control source release

Six new sources frozen in `outputs/native_aggregation_vlm/identity_join_joint_code_long/source_release.json`;106 inherited current/archive sources verified. Driver `b92881eeee15ec40141d66b0e63ea4370e9f1ba4cb3df483c10b753623b0245d`; reporter `a225569d624226a67997a490db6003aeb73b2610baa2eaeca11ec44329211876`. Root full-source review, hils protocol/driver review and v10 independent reporter review passed; AST/bash checks passed. Release one90secondCPU preparation. GPU180second single attempt remains conditional on passed CPU artifacts; unchanged model and original unfitted initialization only.

### 2026-09-11 — long-oracle CPU gate passed; fresh fit released

CPU443496 COMPLETED0:0 in10seconds; planSHA `3e692f8fe3f834fecb59407807f99b7c8b533ece25497c08a1f2c2853ac6f308`, summarySHA `5d033a59ebd924e973a5c3e4f014dbc246e148d26dc3427c736f835f35dea5dc`. Core, schedule, original-order-prefix, final-only decision and actual-Adam/reset fixtures passed. Exact workload213330 training head rows plus324 endpoint rows=213654; maximum44 rows/batch;6021 calls. Root verified6 new/106 inherited current/archive sources, original unfitted checkpoint bytes and7 small runtime bindings. Release one180second GPU allocation/max1, with all three fixed observations and only6000 as the decision endpoint. No architecture or input changes.

### 2026-09-11 — long-oracle completed with provisional primary pass

GPU443498 COMPLETED0:0 in69seconds. Fixed6000 endpoint108/108 first tokens and18/18complete families; earlier fixed600/2000 observations both36/108and0/18. All6000 updates/6021 core,norm,head calls completed;213654head rows and zeroVLM/vision. Horizon and annealing both differ from the600-step control; no original result is revised. Submit the preregistered independent300secondCPU report. No learned-visual or benchmark experiment is automatically released by this provisional result.

### 2026-09-11 — longer privileged readout pass independently accepted

CPUreport443511 COMPLETED0:0 in26seconds. SummarySHA `b702fa30ab354708f3d82192c7e54ba83af491ebf5010ca880b4df6996718fed`; analysisSHA `d0938deeb7212851b6f044ee88d7d09803c17535894c77db4951c2a8ef636ef5`. Root verified6new/106inherited current/archive sources and15 small bound outputs. Final6000screen108/108and18/18families; fixed600/2000both36/108and0/18. All29 captures/21CPUheads324rows passed; maxTV 0.003452907083556056. GPU44349869seconds,6021core/norm/head calls,213654head rows; zeroVLM/vision. The horizon andannealing intervention establishes learnability of this privileged training screen under the larger recipe. It neither revises the short negative nor establishes learned visual or reasoning efficacy. A separate prospectively specified calibration of the locked learned-feature pair is being designed; no fittedcheckpoint continuation or newarchitecture variant.

### 2026-09-11 — prospective calibrated locked visual factor comparison

Protocol `docs/paper/NATIVE_AGGREGATION_FACTOR_LONG_PROPOSAL.md`, SHA `e65cda4d9a856fee7d8f7aba39c9304e2ecebff5bf6527f9afe8b49a350f6547`. The passed long-oracle control motivates a fresh6000-step horizon/cosine calibration of the original locked product/additive pair, from exact original unfitted weights/statistics/features. Both paired-only; no new architecture, auxiliary loss, semantic-code input or failed-checkpoint continuation. Per arm6021core/norm/head calls,213654rows,29captures; only6000screendecisive. CPU90, twoGPUattempts240seconds each/480total/max2, jointCPUreport300. Ifeitherpasses, conditional verified native image-to-answer evaluation ofBOTH endpoints precedes freshvalidation. Native profile120/full108train360seconds perarm requiresseparateCPUintegration release. Freshvalidation panels/seed/criteria are fixed prospectively in the protocol, with manifest/resource release still held. Old failures remain unchanged and matchedjoint/reasoning baselines remain required for broaderclaims. Source review pending.

### 2026-09-11 — pre-execution native-resource correction to factor-long protocol

Preserve draft SHAe65cda4d9a856fee7d8f7aba39c9304e2ecebff5bf6527f9afe8b49a350f6547. Before source release or either new cached fit, replace it with `docs/paper/NATIVE_AGGREGATION_FACTOR_LONG_PROPOSAL_V2.md`, SHA `3c4af1bd12ae328c83764bd7fa949d1453d84dade61ca612451bbc7a8c14c875`. Independent review found the prior measured native setup/four-token timing projects above the draft360second main cap. V2 holds480seconds/native main and1200GPU-seconds for both profile+main pairs, with conservative measured projection required. Cached240second fits/480total, data, initialization, models, losses, criteria, native decoding and fresh validation definition are unchanged. No native job is released now.


### 2026-09-11 — locked visual factor calibration source release

Six new sources frozen in `outputs/native_aggregation_vlm/identity_join_factor_long/source_release.json`; all 99 inherited current/archive sources verified. Driver `07b5ea065b97b6fc34dbfa4323fca4317a4771d0d572196d2443c2f11db06228`; reporter `4bebdae13acfdaa521c0b71a4412cd958338bc21c90a628fff5fa0a089627eb1`. Root and two independent reviews pass; AST/bash checks pass. Release one 90-second CPU preparation. The two 240-second GPU attempts remain conditional on verified CPU preparation; total 480 GPU-seconds and maximum two GPUs. No new architecture or fitted-checkpoint continuation. Native software preparation may proceed concurrently but no native jobs are released.

Before either cached outcome, clarify the held fresh-validation wording without changing thresholds: 33/36 complete triples implies at least 99/108 correct, so the separate 98/108 condition is redundant. Family-instance exclusion is based on concrete canonical three-variant realizations, not abstract question/trio/orientation cells, seeds or paths. Panel A will report orientation strata because it includes both orientations while training had one per cell.

Factor-long CPU preparation 443538 submitted.

### 2026-09-11 — factor-long CPU preparation passed; paired fits released

CPU443538 COMPLETED0:0 in10seconds; planSHA `3442ba95f15bcb833291f71d078832e7e95b7e0137e423696ff6ea535fd018e2`; summarySHA `04dbab209e7aa29a6a2aa2d40a5511913b6a2f86698af4b1b8d8244e2c539e8b`. All inherited factor/conditioning and new schedule/optimizer fixtures passed. Root verified six new and99 inherited current/archive sources plus 5 small runtime bindings. Per arm213330 training rows+324 endpoint rows, max44 per update,6021 core/norm/head calls. Release exactly one240second GPU attempt per arm product/additive,480total/max2, both original unfitted checkpoints. Only6000 decides.

Factor-long product GPUjob 443540 submitted.

Factor-long additive GPUjob 443541 submitted.

### 2026-09-11 — locked visual factors complete with provisional training passes

Product443540 and additive443541 both COMPLETED0:0, allocated seconds {"443540": 80, "443541": 79}, 159 total. Both fixed6000 endpoints achieve108/108 cached first tokens and18/18 complete families. Product fixed600/2000 observations78/108,6families and108/108,18families; additive36/108,0families and99/108,14families. Only6000 decides. These results do not establish a necessary product interaction or native/fresh-data efficacy. Submit the registered independent300secondCPU audit. Conditional native software preparation continues; no native job released yet.

Factor-long independent CPU report 443543 submitted.

### 2026-09-11 — both locked visual factors independently pass calibrated training

CPUreport443543 COMPLETED0:0 in39seconds; summarySHA `705fef511c80c11bfd54fa0ce2adda426462c395345f15ab7a676fc49a6f8a43`; analysisSHA `c6f8db9921ab30d6200ef5fd362ee277d82c6ea8f9130ec807dfd0c333336b98`. Root verified six new and99 inherited current/archive sources plus 24 small output bindings. Both fixed6000 endpoints achieve108/108 first tokens and18/18 complete families. All58 captures/42CPUhead batches648rows passed; GPU159seconds total. The earlier600-step failures remain unchanged; multiplication is not necessary for fitting this dataset under the larger recipe. This is cached training, not native whole-answer or generalization evidence. Conditional native integration preparation now proceeds for BOTH fixed endpoints under the predeclared120profile+480main seconds per arm; new source review and CPU gate required before jobs.

### 2026-09-11 — concrete conditional native evaluation protocol

Protocol `docs/paper/NATIVE_AGGREGATION_FACTOR_NATIVE_PROPOSAL.md`, SHA `e228fc4a52f8409bfc7443beede0730908751132840a1cb039d2db53c43338e5`, passes root and independent review. Cached report443543 satisfies the conditional gate. Both fixed6000 endpoints are evaluated, with only up.weight zeroed for separately saved software copies. Four natural profile trajectories per arm, maximum16model/head calls and4vision prefills, precede an independent CPU release. All profile prefixes receive CPU head replay at the actual N+1 batch shape. Full108-scene main evaluation retains raw logits and independently audited factor/conditioning/cast/history/position captures, without exhaustive main head replay. CPU90/300/300seconds; GPU120profile+480main per arm,1200total/max2. Both profiles and the pooled conservative forecast must pass before mains. Source completion and final review remain pending; no native job released now.

### 2026-09-11 — prospective fresh-data realization and exclusion law

Held data proposal `docs/paper/NATIVE_AGGREGATION_FACTOR_FRESH_DATA_PROPOSAL.md`, SHA `4b47f0653581ed9e9a5602eba1f88c66e845ed4ff0e9f7e97c7e0f10b702a5a4`; symbolic helper SHA `77fc5ca203a9c6ac40b42cee0eaade8c167b1681e6615ef17d03f68518e34041`. Root and both independent readers pass the combinatorial law. The previously fixed810-scene/90-family panels now have explicit deterministic seed/collision handling and canonical held question order. Include all32 canonical prior manifests plus exposed derived software/timing contexts and stopped symbolic ancestry; concrete scene/triple exclusions are reconstructed from actual states and questions. CPU preparation300seconds/render600seconds, four cores/16GiB, remain conditional on independently passed native whole-answer training. No data generation, rendering or fresh prediction is released yet; final stager/source review and exact native report binding remain required. Thresholds and N16-first stop remain unchanged.

### 2026-09-11 — native factor source release and CPU preparation

Ten new files frozen in `outputs/native_aggregation_vlm/identity_join_factor_native/source_release.json`, with105 inherited current/archive sources verified. Driver `20651a26937d8b47f0ca0e796a878686bc45a14740a423c78b4fd8d18a8ff9d8`; reporter `b444ab6adc21c7fc6f66de36f451f4ac3aa6ef1ba97d40dac7ac233deb0f0e9a`; runtime `c5bbfbec739c019ff75344cafcece7a0b9238e25f353db131d985ce71333e173`. Root full-source review and independent final review passed. The head-replay list-schema bug was corrected before execution and an actual-helper CPU fixture added. AST/bash checks pass. Release one90second CPU preparation; native profiles remain conditional on its verified pass. No model fit or fresh prediction.

Native factor CPU preparation 443567 submitted.

### 2026-09-11 — native preparation failure retained; minimal compatibility repair

CPU443567 FAILED1:0 after14allocatedseconds, before runtime fixtures or any GPU work. FailureSHA `742887a78d7d597fa96feb9cf5ab12d92f80e681de9562fe23ede8a37e024760`: unbound `Qwen2_5_VLModel.get_rope_index` was called without its owner. The immutable learned-selection preparation already calls the returned function as `fn(owner, **kwargs)`. Preserve all original ten sources and failed artifacts. Prepare a separate V2 driver/reporter/wrapper release that binds this existing native API correctly; original runtime, model, checkpoints, statistics, cohorts, losses, criteria, native decoding and GPU budgets remain unchanged. No GPU retry or resource extension is involved; no GPU attempt has occurred. New90secondCPU preparation remains conditional on V2 source review.

### 2026-09-11 — native V2 owner-binding repair source release

Eight new files frozen in `outputs/native_aggregation_vlm/identity_join_factor_native_v2/source_release.json`, with115 original sources unchanged. Driver `a49aae13f702de44488e11d2f41b1ac93a72eecc00824a946e9606a3f6ae1a6d`; reporter `3e5ed6e280fa56f5cb4c9a6a9e7cb161eabdb6a36964054117b449b827fa256a`; repairnote `4747a8a69aef4b21e8cac159251c6804c51a816820c7020ff25ebfcfba6ea220`. Root and independent exact-diff reviews pass: the sole native computation repair binds the returned owner in the existing position function call; other edits provide versioned paths and preserved failure/source/accounting. Full108 original layouts provide the regression. Runtime/tests/core/checkpoints and all native budgets/criteria unchanged. Release fresh90second CPU preparation; no GPU job yet. Prior443567 failure14CPU-seconds remains retained, with0priorGPU-seconds.

Native V2 CPU preparation 443574 submitted.

### 2026-09-11 — native V2 CPU gate passed; paired profiles released

CPU443574 COMPLETED0:0 in30seconds; planSHA `894f4c6e17c71b4b4ddbc9184d27e9bca5ef90f3501ddbad78497e126b8286a6`; summarySHA `a0286045ebc3411ff9e28a671d51ef810a9d49581b53ae5764b6da8894784db9`. All108 actual owner-bound native layouts, seven runtime fixtures and four reporter fixtures pass. Root verified eight new archived/current and115 inherited current sources plus 3 small runtime bindings. Original CPU failure preserved;0 priorGPU-seconds. Release one120second profile per arm, four fixed natural trajectories each, maximum16model/head calls4visionprefills; total240GPU-seconds/max2. Full mains remain conditional on independent profile audit and pooled conservative forecast, with unchanged1200totalGPU-seconds.

Native V2 product profile 443576 submitted.

Native V2 additive profile 443577 submitted.

### 2026-09-11 — native V2 profiles complete; independent release audit

Profiles443576/443577 both COMPLETED0:0 in56GPU-seconds each,112 total. Each completed four natural trajectories and10 native model/norm/head/factor calls,4vision prefills,0extraGPUheads. Submit the prescribed300second independent CPU profile replay/timing release audit. No main or fresh-data job is released until that audit and pooled forecast pass.

Native V2 independent release 443578 submitted.


### Native factor V2 full training evaluation release — 2026-09-11

Independent CPU release443578 passed in33seconds; both profiles completed56GPU-seconds each. Root verified123 current source hashes and21 small output bindings. Release SHA154d0c11824799c2a380da20a76c20186968e34e050b48de3838f272ff80e8fd; summary SHA9967fb8b008ad428ef671b8b2f50dc9df1c71913ea4e768ea9211a7d97813f17. All20 CPU heads/260 rows have exact argmax; maxTV5.023167432227638e-06. Pooled forecast416.750231327489s passes480s per arm, total reservation1072/1200GPU-s including profiles. Release BOTH fixed6000 endpoints for108 natural whole-answer training trajectories each, max4tokens, no fitting or extra heads. Whole-answer gate103/108 and16/18 complete families; both arms evaluated regardless of relative outcome. Fresh extrapolation remains unreleased.


Native mains443588/443589 both completed108 natural trajectories,240 model/head calls and108 vision prefills each. Allocated mainGPU-seconds: 230; profiles included 342. Submit independent whole-answer and all-prefix audit 443592, CPU300seconds. No fresh data released before verified report.


### Fresh N16 inference source freeze — 2026-09-11

Root and three independent agents reviewed the fresh staging and inference sources. Frozen ledgers are identity_join_factor_fresh/data_staging/source_release.json and evaluation/source_release.json. Staging11 sources bind the123 native ancestors; evaluation6 new sources bind the134-source native/staging union. Inference proposal SHA9124ce838f635df191949c9e924a7d6b11f6ac3923206420957a81f6e04c4eb0 clarifies that all224 prior timings and all54 prior/270 fresh N16 envelope checks occur inside the SAME600secondCPU preparation. Both270-scene GPU runs have1200seconds each,2400total/max2. Fixed whole-answer/EOS A98/108+33/36triples, B98/108+33/36, C49/54+16/18 criteria retained. N32/N64 require later separate release. All remain held until completed native full-report and fresh-data proofs; no fresh predictions yet.


Correction to the immediately preceding fresh source-freeze inventory: the evaluation inherited union has128 unique sources (123native plus11staging, with6shared sources), not134. With6new evaluation files, its total closure is134. The source_release.json already records the correct128-entry map; no executed source or policy changed.


### Native whole-answer training pass and fresh CPU release — 2026-09-11

Independent report443592 COMPLETED0:0 in47CPU-seconds; summarySHA d2f7a6ebe32020960be9e6811fb622f42c8effc3e35cd30e0578a419103d5a59; analysisSHA ccf811475981073703af7a172abafcbc1cdfeccad08b29a6f46045e876bb5720. Both fixed6000 endpoints achieve108/108 whole answers includingEOS and18/18 complete families. All480 main native calls/216vision prefills and20 inherited CPU profile heads are audited; no extraGPUheads. Native profiles+mains342GPU-seconds total. Root checked123current sources,8current/archive and244 small output bindings. This establishes training-only native execution, not fresh/generalization performance. Release the reviewed300second fresh-data CPU preparation, then conditional600second rendering. Data-only; inference remains dependent on those proofs and its separate CPU preparation.


Fresh CPU preparation443594 COMPLETED0:0 in55seconds. PlanSHA957611990d113efd43537c07154ab5ee28e9665768d3a3c035a43b3a85c439b0; summarySHA6fa577a19c15e7de1dfffe2ac70570912ee7c2b9ba6248711959263b1e36f848. All810scenes/90families/270triples and30240image occurrences pass;16074scene and2198concrete-triple exclusions verified. Canonical render inventory3155atoms:3007reused,148new. Root verified11current/archive sources and9output bindings. Release one600secondCPU rendering job; frozen images/QA/semantic outputs audited before inference. No fresh predictions yet.


Fresh canonical render443595 COMPLETED0:0 in91CPU-seconds; summarySHAaed7387e9fff96b68e0d04d26a3584ca93c335fc9e264d093b123a56cd206754; manifestSHAc40dcd05cee34ac7232c089b788b340a7e13344bb662abe2704e8deb2f999859. All810QA/image sequences,3155RGB atoms and30240ordered image occurrences independently audited. Release the reviewed600secondCPU preparation for all270N16contexts and both fixed6000 endpoints. That same job verifies all224native profile/main timings and compares all270fresh/54original prepared envelopes before any GPU inference release. No fitting or new statistics; N32/N64 still held.


### Fresh N16 paired GPU release — 2026-09-11

CPU443598 COMPLETED0:0 in219seconds. PlanSHA761d5e01753e2f80e49f06bd2d1d63ccafff23d0931bcf70a6d8296af3c89793; summarySHA60dd48c04bb58f29489249280a25f0828c1d001f336d61f387f10f4c9475a210. All270fresh inputs remain inside the54original N16 envelope; driver4/reporter4 fixture groups pass. All4setup/224trajectory timings yield884.1979018114507seconds <=1200perarm. Root verified134current sources,6archives and5small runtime bindings. Release BOTH fixed step6000 endpoints for all270A/B/C natural trajectories each, max4tokens, one1200secondGPU attempt per arm/2400total/max2. No fitting/new statistics/extraheads. OriginalA/B/C and N16-first criteria fixed; no N32/N64 release.


Fresh N16 GPU443605/443606 both completed all270 scenes. Inclusive allocatedGPU-seconds680. Submit independent600secondCPU report443611; whole-answer/family outcomes and every native prefix remain to be audited. Neither fresh success nor longer-length release is assumed.


### Fresh N16 failure retained; longer lengths stopped — 2026-09-11

Independent report443611 COMPLETED0:0 in162CPU-seconds. SummarySHAc10913e34b67fd0623f87997f6193b8da20b37abe7e624d2c8ea4721c2b6f00d; analysisSHAcb4c210cc1b3eb289c33d6b8c2419f1f752ab9bc76834acda05ca96b725baffe. Product A64/108+15/36triples, B14/108+0/36, C7/54+0/18; additive A49/108+11/36, B14/108+0/36, C7/54+0/18. Both fail every registered panel criterion and the A/B qualification. Both270-scene runs completed340GPU-seconds each,680total;1240native calls/captures,540vision prefills,0extraGPUorCPUheads. Root verified134current sources,6archives and all540trajectory audit/output bindings. No N32/N64 evaluation of these checkpoints is released. Preserve these fresh panels as failed confirmation, not tuning data. The separate frozen joint reference remains useful for interpretation and cannot reopen this length gate. Any training-coverage expansion requires a new prospective experiment and new independent fresh confirmation.


### Ordinary frozen joint reference source release — 2026-09-11

Eight new files frozen at identity_join_joint_baseline/source_release.json SHAace46e637f006c4f73bb8ef85ff5ffe04760b0b19d76a1138a1fc6ee209f0de2, with128unchanged native/staging ancestors. DriverSHAae41f267d8d6051c2bc3731e36ce0e06443ca6dee23c650915d4f292e38a1681; reporterSHA2c2502959b1a0ee191f0aa83f5099ad622806db7a014558532f16edf04d421c6; proposalSHA891851f5ac930b09b9ba207a7aed476dad70b558129e64ffcaa630608baef94f. Root/full independent source/API/schema review passes. Same270freshN16scenes and exact name question, all16images in one ordinary native sequence, frozenQwen, nofactor/broadcast/countwrapper. CPU300prep,120GPUprofile withthree input-selected widest-panel cases,300CPUrelease, conditional1200GPUmain,300CPUreport;1320totalGPU-seconds/max1. Observed native norm coversfullprompt, selected actualheadonequery; only those <=12profilequeries receive CPUheadreplay. GPU verifies consumed bundle bytes within scene timing; CPU verifiesall270. Profile native integrity and measured joint-only forecast required beforemain. Release only CPUpreparation now. Frozen-reference result cannot reopen failed factorN16 gate or establish adaptation-matched superiority.


Ordinary joint CPU443619 COMPLETED0:0 in213seconds. PlanSHAef0ff4a4412f638d11ece4109402d3e2297359b6d30bd4bca829f334ccecd66c; summarySHAe629d76a6ae53607ce2ab41fd31ddb43d0fef30d400d0773880be56bf2073ce3. All270native layouts and4driver/4reporter fixture groups pass. Three fixed input-only profiles select firstwidest A/B/C rows, width3212 each. Root verified136current sources,8archives and3small runtime bindings. Release one120secondGPU profile, maximum12nativecalls/3visionprefills. Fullmain remains conditional on independent selected-head fidelity and measured joint-only projection; totalstage1320GPU-s/max1.


Ordinary joint profile443629 COMPLETED0:0 in33GPU-seconds;3natural trajectories,6model/nativehead calls,3visionprefills,0extraheads. Submit independent300secondCPU replay/timing release 443631. Fullmain still held.


### Prospective fixed-budget orientation coverage — 2026-09-11

Protocol docs/paper/NATIVE_AGGREGATION_FACTOR_ORIENTATION_PROPOSAL.md SHAe7b0bc378ca7115a71bc48f033acef693eb1586871bb4ab585bcf0e028fa128e passed root and independent scientific review. Add exact target-independent requested-room swaps to original108 training scenes;216contexts/18complete12-context families. Keep both locked architectures, unfitted initialization, original conditioning and6000CE schedule. Alternating per-base-pair visits selectsoneorientation per originalpresentation, retaining48000pair slots/16scenes perupdate/target sequence/headcounts. Gate206/216 pooled,103/108eachorientation and16/18completefamilies at fixed6000; native wholeanswers decisive aftercached feasibility. New seed91726342 independent810sceneconfirmation must be frozen before either fit, excluding oldfailedfresh810 andallnewtrainingcounterparts; sameN16-first A/B/C thresholds. CPUtraining staging300s is being implemented/reviewed, missingfeatureharvest requiresseparatemeasuredrelease; no newdataorfit is released now. One180sfit/core360totalmax2 conditionally proposed. Native/freshGPU work still requiresseparatereviewed budgets. No changes topriorfailures/stops.


Ordinary joint CPU release443631 COMPLETED0:0 in34seconds. SummarySHA919feda3d87d7a84b12e0707ad917574cba7b75e289bb5cd6712dd80e25588b0; releaseSHAe22a54484870c32c6087c09f50b8d6fc41590b068bbfe393889f24d126c44b77. All6actual selected-head queries replayed with exactargmax,maxTV3.6183232055009285e-07. Joint-onlyforecast862.7519357004203<=1200s; profile33+mainreserved1200=1233<=1320GPU-s. Root136current/8archive sources andprofile/outputbindingsverified. Release exactlyone1200secondordinary joint main onall270freshN16scenes, max1080model/headcalls270visionprefills. Bothfailedparallelendpointsremaincomparators; nofactorlonger-lengthrelease.


Ordinary joint main443635 COMPLETED0:0 in247GPU-seconds,270natural trajectories/639nativecalls/270visionprefills. Profile included280GPU-seconds total. Submit independent300secondCPU report443643, including descriptive paired comparison against BOTHfailedfactorendpoints fromsame270freshscenes. No change tofactorN16 stop.


Orientation training source release SHA1256d1c29acce571b6f5a10bbd0edcbd2f8de1be4671d934615ec770d6210366: 4new+104inherited sources verified. Root/v10/attention code and independent verifier review PASS; hils scientific review PASS. Release one300secondCPU stage for216canonical scenes, target-independent room swaps, exact48000presentation mapping, old+failedfresh exclusion audit and strict native-feature missing-key inventory only. No model/head/tensor loads, harvest or fit.


Ordinary joint report443643 COMPLETED0:0 in70CPU-seconds. SummarySHAa4cb29de4d5a08e8d962f48c8f45fe1e1d081b444fcb2c20aa509840e074a3dc; analysisSHA247afdc265a8473e4bbe87888a6000047563c7a162e67618d7efd5c5f8d9137a. All270trajectories/639nativecalls verified. Frozen joint scores A42/108, B36/108, C18/54; complete triples1/36,0/36,0/18. Both fitted parallel methods exceed joint on familiar-groupA but lose on novel-groupB/C. Comparison is not adaptation matched and does not establish equal-compute performance; setup/hash overhead differs. Profile/main cost280GPU-seconds. BothfailedfactorN16 stops remain; room-orientation training coverage experiment proceeds separately.


Orientation stage443650 FAILED1:0 in55CPU-seconds before feature inventory: general input binder rejected canonical HuggingFace config.json symlink. FailureSHA3e1f12be71eca8837344bcab543556664121059a516d04ece8158ccaa572d04f. No model/head/tensor load or fit. Preserve source/data/failure unchanged. Separate minimal V2 read-only input binding repair is being prepared; all216semantic scenes/order/targets and prospective criteria unchanged.


Orientation stage V2 source release SHAbcfb7a993cfd61131c16e00500428870c7738595f1b4e0c86fd4f7336074974e:3new+108inherited verified and root/attention/v10 exact-diff review PASS. Only expected HuggingFace snapshot-to-own-blob symlinks may be read, with snapshot and resolved-target hashes both retained; all other input bindings unchanged. Actual config/wrong-hash/unrelated-symlink fixtures will run inCPU. Preservefailed44365055s; release one300sCPU V2attempt, same216semantic scenes/order/targets/featureinventory, noGPU/model/head/fit.


Orientation V2 stage443660 COMPLETED0:0 in37CPU-seconds; summarySHAac2fba1ab16e49e0c879f3f11261ba4fd746cc39b2732a387ce60f081beb26d4; planSHA159be56f60e4ec2614397d6915d0310dd26e15133cbbf51156287a650e8bfc32. All216training scenes/2592images/48000mapped presentations pass independent semantics/feature-key/exclusion checks. Feature inventory {'features': 2940, 'missing': 220, 'missing_by_phase': {'local_empty': 2, 'local_prefix': 218}, 'reused': 2720}. No model/head/tensor loads. Root111current/3archive sources andplan/auditbindingsverified. PreservedV1failure55s remains. Newconfirmation and native feature-bank preparation next; nofitreleased.


Orientation confirmation sources frozen before fitting: releaseSHA45eb201bba0fe9c5ca3c0318f994e79004f2712e929a9983d98ddf33dc25dad6, 140total current sources verified. Root/attention/hils source/law review PASS; final import-only adaptation consumes passedV2training443660. Newseed91726342, all810newscenes/90families, exactoriginal108support, old+810failedfresh+216training semantic/concrete-triple exclusions; bothorientations nowtrained. Release300sCPUcheck.600srender remains conditional onpass. No model/head/tensor loads, fit or inference.


Newconfirmation check443663 COMPLETED0:0 in47CPU-seconds. SummarySHA86ed3de6ccca18247e6aed6cc8bfe1a0645304015ce04597983a469587c56c4d; planSHA3cd946f038982a6f48527438d1dc939bae33e4b355b864e3d46c6c8a9e6ebd39. All810semantic scenes/90families/270triples pass exclusions andnestedcanonical law; excludes16992sceneidentities2504triples;3156atoms3014reused142new. Root140current/140archive sources andallsmall outputbindingsverified. Release600sCPU canonicalrender usingexactplan/trainingstage. No fit/inference.


Independent orientation confirmation render443666 COMPLETED0:0 in98CPU-seconds. SummarySHA4aa67212336048575dc958e5d7b82990fedcdf1f395b3a74359c0f7c082913a9; manifestSHA279193c4f6b9861aa4bca1ee88b011241a87c57705ddf9e38347c59f2df80544. All810scenes/30240imageoccurrences passcanonicalrender/QA/hash/inode/order audit;3156atoms3014reused142new. Root140current/140archive source hashes andmanifest/outputbindingsverified. Complete newseed91726342manifest is frozen BEFOREeitherfit; no fit or inference yet.


Orientation native-feature bridge source releaseSHAef1f1791193466d82da13672be35b9b245de804ba33d6c54575817626770f86d:4new+111inherited frozen/verified; root/attention/v10 source/APIreviewPASS. Release one300sCPU preparation for exact220missing keys (2empty,218strictprefix),5batches B2/64/64/64/26, copied original native pixels/layouts and owner-awarepositions. GPU harvest requires passed source/input/measurement preparation; cap150s,oneattempt/oneB200,no extraGPUheads; pooled original localphase complete-batch timing predicts about62.3s. CPUmerge300s will replay all220nativeFP16headrows atactualB×1shape (exactargmax,TV<=.02), retain2720oldrowhashes andgatherexact2940statebank. No gate/local labels/fit, zero-missingpathforbidsGPU.


Feature CPUprepare443667 COMPLETED0:0 in32seconds. SummarySHA4113f2ff2fd89ebd304c164401af64826b6a4c36e3652f439834bbf56c8c1970; planSHA4cdd2cd026c4857b20ac485e1833acaeb049b828df9db166c8014910b1bf2aad. Actual220strict-prefix layouts in5B2/64/64/64/26 batches and5fixtures pass; measuredforecast62.31513432227075<=150s. Root115current/4archive sources and11small inputbindingsverified. Release exactlyone150secondB200 native harvest for220missingstates,5actualmodel/vision/norm/headcalls,0extraGPUheads/0cores. No fit; completeCPU nativeheadreplay/merge remainsrequired.


Orientation training implementation frozen:6new+144inherited sources; releaseSHAebefc0bb076d89316955ffa83b4dfe1e4f5beaff1ca6a299ac3bf421eae68f89. Root/attention/hils driver/report/source/APIreviewPASS. Original question_index retained as offline metadata; no modelinputchange. CPUcheck300s waits complete exactnativebank. Bothfits remain original unfittedweights/globalstats,6000CEupdates/same213330trainingrows, sole6000endpoint216rows/14heads. Nativehead6000+14=6014calls/arm. Newconfirmation443666summary4aa67212336048575dc958e5d7b82990fedcdf1f395b3a74359c0f7c082913a9 alreadycompletebeforefits. Two180sGPUfits/360total/max2 conditional onCPUcheck; noimplicitretry. NativeproposalSHA8ea55afc7d42f1b25c229c63b1e83e42744309236a37facfb54c5666049b5cb8 independentlyreviewed andHELD conditional oncachedsuccess.


Missing-native-state harvest443671 COMPLETED0:0 in39GPU-seconds; summarySHA870479dc95c23d2d7774774f8c3a81d98ded024442c9f0ddb0e7019ac4e19050. Exactly5native model/language/vision/norm/headcalls,220rows,0extraheads/cores. Allactualinput/mRoPE/nativeweight checks passed. Root115current/4archive sources andplan/observationsbindingsverified. Submit300sCPU fullnative replay/merge, preservingall2720reusedrowhashes and220newcaptures. No fituntilpassedmerge+trainingCPUcheck.


Feature CPUmerge443672 COMPLETED0:0 in31seconds; summarySHA8adbff2573e5d64be0b91b50f7ca20aeee4efef5763bbd370fc6a6bfd37cc8e1; cacheSHAa4b7ea5975d7cdd32a5456949417a148e05e348ec249ff2ffd407a6e1fc08feb. All2940nativeFP16states exact,2720reused+220actualnew; all5CPUheads/220rows exactargmax,maxTV0.0003415265959701606. GPUcost39s/5calls/0extraheads. Root115current/4archive sources and5replay/outputbindingsverified. Release300secondorientation training CPUcheck against thiscompletebank andalreadyfrozenconfirmation443666, noGPUfituntilcheckpasses.


Orientation training CPUcheck443676 COMPLETED0:0 in14seconds; summarySHA72e678300833b1e470e7d6b0aff2d1f8b98b92d393233ce986c08d422ac92a03; planSHAd299d42a4444520f148f86252e2931ad599e03afac9edadbe9201836229bf097. Complete2940statebank,216rows,6000unchangedtraining-headrowbatches213330targets,originalunfittedstate/statistics, driver4criterion/unchangedfactor andreporter7fixturegroups PASS. Root150current/6archivesources and7small runtimebindingsverified; confirmation443666 wascompleted beforeCPUcheck. Releaseboth180secondB200 fits concurrently with exactregisteredproduct/additivejobnames,360GPU-scombined/max2. Onlyfixed6000endpoint, no retry/continuation/native release.


Bothorientationfits443677(product)/443678(additive) COMPLETED0:0 in73GPU-seconds each,146total. Each6000updates/6014core,norm,headcalls/213546headrows,0VLMvision. Allfixedtraining/endpointrawoutputs retained. Submit300sindependentCPUreport against bothmodels; accuracyinterpretationwaitsfullaudit. No nativejob released.


Native orientation implementation frozen HELD:8new+150inheritedsources releaseSHA459025a3048cca811f59dacfb24162664b03469fb66edbc8296fb12221d4e949. Root/attention/v10/hils fullsource/APIreviewPASS. Sameoriginalunfittedstats/nativecore, bothfixed6000checkpoints, all216trainingcontexts,8profiles/arm×zero/fitted+twoorientations+twolengths, pooled16timings;120sprofiles900smains2040total/max2;300sCPUcheck/release/report. Native206pooled103each16complete12-contextfamily gate. Onlypositivecachedreport443682 canreleaseCPUpreparation; no nativejob launched now.


Orientation report443682 FAILED1:0 in38CPU-seconds atproductfinalbatch10: oneCPUFP16native-head argmax differsfromsavedGPU, TV0.0036688551772385836<.02; allcapturedfunctional/NLLchecks pass throughthispoint. FailureSHA4b113df062f18c49973864849378608678264a035e30e25935ab111df525be1c. Preservefailedhardargmaxrule andallfrozenfiles. Provisional GPU-saved endpointcounts areproduct79/216(42original,37flipped),additive72/216(37,35),both0/18families: neitherapproaches206/103each/16. No validatedtraining/nativeclaim. SeparateCPU-only complete28batchprecision diagnostic isbeingprepared toreportallCPU/GPU disagreements andworst/bestcriterionbounds withoutrelaxingacceptance. Native/confirmation inference remainsHELD; no reason tospendGPUonfailedfits.


### 2026-09-11 — Orientation failure CPU diagnostics

Freeze the five source hashes in outputs/native_aggregation_vlm/identity_join_factor_orientation_training/diagnostic_source_release.json. One CPU300s/4cores/16GiB replay of28norm/29head calls432/448rows, plus one CPU60s/4cores/16GiB four-log streaming comparison. Preserve report443682 failure and all original models. Descriptive sensitivity/dynamics only, no relaxed argmax rule and no native/fresh release.


### 2026-09-11 — Finite-precision symmetric-join capacity witness

Freeze3 new sources plus inherited closure at outputs/native_aggregation_vlm/identity_join_orientation_capacity/source_release.json. One CPU300s/4cores/16GiB witness,216 first queries,14 constructed-core/28 norm/28 head calls432headrows. Compare the existing36-unit construction to the bound earlier answer-code oracle; require216/216 bothroutes,18/18families and declared residual/TV/exactargmax agreement. No fit, GPU, backbone or native/fresh release. Constructed weights never initialize the subsequent learned control.


### 2026-09-11 — Orientation-complete learned local-code control

Freeze6 new sources plus inherited closure in outputs/native_aggregation_vlm/identity_join_orientation_joint_code/source_release.json. Exact216-scene orientation plan,2592 independent raw per-image codes, original unfitted4 readout tensors and frozen globals/native head. Same6000 CE updates,48000 mapped pair slots and213330 target positions. First require passed finite-precision capacity witness443708. Then CPUcheck90s/4cores/16GiB, singleGPU120s, independentCPUreport300s. Only final6000 first-query216endpoint,6014core/norm/headcalls213546rows. Numerical failures retain all22captures/14CPUreplaybatches before failing unchanged thresholds. Criterion206/103each/16families. Neither success nor failure releases vision or native/fresh evaluation.


### 2026-09-11 — Shared paired-minibatch presentation permutation

Freeze3 sources and ancestry in outputs/native_aggregation_vlm/identity_join_orientation_paired_order/source_release.json. CPU90s/4cores/16GiB, no tensors/heads/fit. Preserve48,000 original full entry dictionaries; pair original/flipped counterpart visits within each of444 complete two-cycle groups, flatten globally, four bundles per update; preserve48-entry tail. Exact independent permutation/bijection, complete N8/N16 pairs,213330targetrows, derived max<=48. Fixed code negative audit remainsFAILED butall22core/NLLchecks andall216outcomes retained; optimistic80<206. Shared visual79/72negativebound and four-log comparison fixed. No fit automatically released.


### 2026-09-11 — Paired-minibatch local-code learning control

Freeze6 new sources and closure at outputs/native_aggregation_vlm/identity_join_orientation_paired_code/source_release.json. Sharedorderstage443724 passed beforepreflight. Exact prior443711 inputs/init/nativehead, onlynewpairedorder. Same6000CE/LR and213330targetrows; maxderived48. CPUcheck90s, singleGPU120s, CPUreport300s,4cores/16GiB. Final216firstqueries/206pooled/103each/16families,22capture/14CPUheadbatchcollection. CodeGPU held untilvisualproduct/additive sources/preflightsready, thenall3parallel atmax3GPUs360total seconds. No automaticnative/freshrelease.


### 2026-09-11 — Paired-minibatch visual product/additive control

Freeze6 sources and closure in outputs/native_aggregation_vlm/identity_join_orientation_paired_visual/source_release.json. Sharedorderstage443724 and unchangedoriginal216 data/features/stats/fullunfittedcheckpoint. Onlypresentationpermutation changes. Same6000LR/CE,213330train+216finalheadrowsperarm, max48/6014calls. CPUcheck90s, one120sGPUperarm240total, CPUreport300s; all4cores/16GiB. Collect44core/native/NLLcaptures and28CPUreplaybatchesbeforeunchangedfidelityfailure. Final206pooled/103each/16families. Releasebothvisualfitswithpaired-code controlonlyafterbothpassedpreparations;3GPUsmax360totalseconds. No automaticnative/freshrelease.


### 2026-09-11 — Saved local-code readout geometry

Freeze3 own sources and ancestry in outputs/native_aggregation_vlm/identity_join_code_readout_geometry/source_release.json. One90-second/four-core/16GiB CPU allocation covers only runs443716/443735,44 saved captures and432 final first queries. Saved actual preactivation SiLU derivatives and immutable saved-weight projections only; no core/head/backbone, gradient or fit. Fixed descriptive bins|slope|<=.01,|curvature|<=.01,|slope-1|<=.01. Preserve strict numerical failure443718. No scientific or downstream release criterion changes.

### 2026-09-11 — First-query supervision local-code control

Freeze6 own sources and ancestry in outputs/native_aggregation_vlm/identity_join_orientation_first_query_code/source_release.json. Bind passed paired-code preparation443728 and negative computational report443742. Original code/input/init/order/6000optimizer settings and all213330+216 full-prefix native head rows unchanged. Only sum_s CE[first_s]/(16*L_s) drives backward; original first-position coefficients retained, full CE diagnostic only. CPU preflight90s includes actual-loss gradient/full-forward fixture; one GPU120s; independent CPU report300s. Final216firstqueries/206pooled/103each/16families, all22captures/14CPU replay batches retained before strict numerical gate. No visual/native/fresh release from either outcome.


### 2026-09-11 — Post-SiLU query code placement control

Freeze7 own sources and ancestry at outputs/native_aggregation_vlm/identity_join_orientation_post_query_code/source_release.json. Bound passedpreparation443773 and negativecomputationalreport443779. Change only finalreadout to U[SiLU(Wagg*z+b)+q], preserve4 originalunfittedtensors697440, exactcodes/global/native/order/firstqueryloss/6000LR and all213330+216headrows. CPUcheck90s includes newcore/formula/gradient fixtures; one120sGPU; independentCPUreport300s,4cores16GiB. Final206/103each/16family screen,22captures14nativeCPUreplays, allnumericalfailuresretainedbeforestrictgate. HoldGPU untilvisualpairedpreflightready, then3armsmax360totalseconds. No native/freshrelease. Failure ends this placement line; no scale/activation sweep.


### 2026-09-11 — Post-SiLU query visual placement control

Freeze7 own sources and ancestry in outputs/native_aggregation_vlm/identity_join_orientation_post_query_visual/source_release.json. Bind pairedvisualpreparation443734, negative443743, first-querynegative443779 and descriptivegeometry443772. Change only finalqueryplacement to U[SiLU(Wagg*z+b)+q]; actualq remainsinbothlocalfactors. Sameoriginalunfitted8tensors1385857retained/1385760trainable, nativeinputs/stats/pairedorder/6000LR/fullCE. Same213330+216headrows6014callsperarm. CPUcheck90s,newcore/independentfixtures;GPU120sperarm240total;jointCPUreport300s. Both44captures/28CPUreplayscollectedbeforestrictgate. Final206/103each/16families. CodeCPU443785alreadyPASS19s; releaseall3onlyaftervisualCPU passes,max3GPUs360totalseconds. No automaticnative/freshrelease andno furtherplacementscale/activation sweep.


### 2026-09-12 — Matched visual moment-basis comparison

Freeze7 own sources plus ancestry at outputs/native_aggregation_vlm/identity_join_moment_basis/source_release.json. Direct visual learned factors; both arms compute A=sum(.5a), B=sum(.5b), C=sum(.25ab), P=A*B-C; readout receives [A,B,C] or [A,B,P]. Six trainable tensors1404192parameters, fresh shared seed24 initialization, pre-SiLU query. Same pairedvisual443734 nativeinputs/globalstats/48000pairedorder/6000fullCE/213330train+216finalrows6014calls perarm. CPUcheck90s, one120sB200 perarm240total/max2, jointCPUreport300s,4cores16GiB. Explicit pair values/gradients/merge/duplicate/masking fixtures; all44captured audits and28CPUhead batches collected before unchanged strict fidelity gates. Final206/103each/16families. Bothfailed ends branch; no outcome-dependent scaling or recipe sweep. Real-arithmetic invertibility is not equal finite-width expressivity/FP32conditioning; known moment primitive, no information-capacity claim. Native whole answers and fresh91726342 require further reviewed implementations.


### 2026-09-12 — Moment-basis metadata compatibility repair V2

Original CPU443829 failed5seconds before fixtures/initialization/GPU: postquery-code report protocol differs from folder name. Preserve7 frozen sources/release/failure/request. Freeze6 new V2 sources plus223 inherited at outputs/native_aggregation_vlm/identity_join_moment_basis_v2/source_release.json. Explicit expected protocol mapping only; original moment core/formulas/freshinit/input/order/6000training/evaluation/gates/caps unchanged. Root executed smallmetadata guards and verified scientificpolicy equality; independentreviewsPASS. One90secondCPUpreflight now; GPU remains conditional onpassedCPU. Originalfailure5CPU-seconds/zeroGPU reported separately.


Moment V2 CPU443836 COMPLETED0:0 in14seconds; summarya205e2100714242d5352bc3da9a5ec02710f8ce34c8fa4599680f3928bd7209b,plan88ae90c63fd0bc08a6e8c3cadec9da46b34f2da3c0214cfa2bba88a884dc1d30. All actual-core and independent moment/gradient/mask/merge/native-cast/criterion fixtures passed; exact original inputs and fresh6-tensor initializer passed. Root229current/6archive sourcehashes and smallruntimebindings verified. Release both120secondB200 fits concurrently,240total/max2,exactper-arm jobnames. No native/freshrelease.


Moment fits443837within/443838cross bothCOMPLETED0:0 in72GPU-seconds each144total. Bothfixed6000endpoints retained,6014core/norm/headcalls each,0VLM/vision. SavedGPUfirstquery77/216(42/35) and73/216(40/33),0completefamilies; provisionalfailure. Submit independent300secondCPUreport collectingall44captures/28nativeheadreplays. No native/freshrelease.


### 2026-09-12 — Frozen moment-readout exact linear-separation diagnostic

Freeze4 own sources plus229 inherited in outputs/native_aggregation_vlm/identity_join_moment_readout_lp/source_release.json. Bind strictfailed443841 analysis5b206f80/failure63a8a8bd, all44functional/cast/NLLpassed andcomplete432headrows; soleheadtie leavesrobustnegativewithin77/crossupper74. ExactrawsavedFP32 pre-U features216x96perarm, nointercept/preprocessing/aggregation update. One5secondninebasisfixture (knownmargin1/9, exactintervalwidth<=1e-8) then2actual45secondHiGHS-ds L1-bounded max-margin solves, fixed1e-9 tolerances. Independentexactrational primal/dual bounds andFP32roundingcertificate; noapproximatezerononseparabilityclaim. Full216strictseparation distinctfrom206methodgate; no native/fresh/architecture release. AllcoefficientsCKPT,matricesDATA; one300secondCPU4cores16GiB,0GPU/core/nativehead/backbone. Root/3peer source/mathematicalreviewPASS.


### 2026-09-12 — Ordinary joint-image LoRA competence baseline CPU/profile

Freeze4 own sources plus158 inherited in outputs/native_aggregation_vlm/identity_join_joint_lora/source_release.json. Root and independent reviewsPASS. Exact216 orientation443676 inputs; ordinary images-first native stream and complete-name+EOS loss, no decoder-feature cache. Language28-layer q/k/v/o LoRA rank16 alpha32 dropout.05 seed24,224FP32 tensors10092544params; all original base storage/dtypes frozen. Prospective12epochs2592sceneF/B,324AdamWupdates8accum,constant2e-4; only CPU300s and one B200240s profile released now. Profile8naturalzero-adapter comparisons,16teacherF/B2updates,4roundtripforwards; caps52nativecalls28vision, real kernel/recompute/gradient/initial-state restoration evidence. Fixed inclusive-cost forecast must be<=3600s; main implementation/release and native whole-answer/fresh evaluation remain separate. Standard competence baseline changes adaptation scope/exposure/compute; not a matched causal comparison with failed adapters.


Joint-LoRA CPU443875 COMPLETED0:0 in145CPU-seconds, summary88fc29a38f66762ac09d47de8b5a6b7adf1c987c9363e311bcf73a2a39b071cd, plan886da0916e78ab72e0aa1428fe058ab1f465db009c70bdb912b19adaadcacf58. All216 image/QA/teacher layouts and fourCPU fixture groupsPASS. Full12epoch targetrows5760; widestN8/N16 widths1630/3214, bothSandra3targets; profileheadcap92. Root independent smallJSON/current-source auditPASS; GPU443880 single240secondB200 software/costprofile submitted. No main or fresh release.


### 2026-09-12 — Joint-LoRA effective-target metadata repair V2

Preserve original443880 failure34GPU-s/10nativecalls/4vision, with no zero-LoRA or backward/update. Freeze4new sources+162inherited at identity_join_joint_lora_v2/source_release.json. InstalledPEFT condenseslarge targetlists; require effective installed-matcher selection overALLlive modules==exact112, retainingactual112/224FP32/count/scalarguards. New112tinyCPUfixture actuallycondenses and rejects widenedvisualselector. Bind passed443875 and recomputeall216layouts/teacheridentities/order exactly; retainall originalscience/nativeparity/loss/gradient/costgates. Saveactualconfigbeforeguard andrawsuccessfulteacherbeforegradientchecks. CPU300s thenonlyoneconditional240sB200profile; explicitcumulative34+240=274GPU-s. Root+independentreviewsPASS,19scientific/runtimefunctions/originalpolicyfieldsASTunchanged. No main/freshrelease or automaticretry.


Joint-LoRA V2CPU443888 COMPLETED0:0 in150seconds; summaryd1002b3e3bdb9608567eee09b57741dabe064ddd7dac9e5c5e0c55fe29d725cb,planf2314c7c689fb489c1efea43c7da625cf53a2822ae1d8467e5208c784010f430. Actual112modulefixturecondenses to4suffixes whilepreserving224adapters andrejectingwidenedvisualselector; all216rebuiltmetadata/orderexactV1. Rootindependent4+162source/CPUproof/smallruntimeauditPASS. GPU443890 single240secondB200profile released; totalcap274includingfailed34. Main/extrapolationexecutionheld.


## 2026-09-12 — independent joint-LoRA audit and separate main cost accounting

Freeze the new two-source CPU auditor plus166 immutable ancestors before numerical replay. Profile cap300 seconds, main cap1800 seconds, four CPU cores16GiB, no VLM or backward. Require exact GPU base/zero and trained serialization parity, native CPU full-vocabulary TV<=.02 and full target CE error<=2e-6. CPU/GPU argmax differences are descriptive; actual native GPU IDs including canonicalEOS define performance. Earlier strict reports stay unchanged. Main CPU replay will validate all2592 teacher traces/5760 target positions and replay first+last scene in each epoch plus every final generation query.

Preserve the original failed28545.207520-second profile forecast. For the forthcoming main, which never enables profiler tracing, classify only fixed(step1,micro1/2) instrumented calls as once-only additional setup; maxima over ALL remaining seven ordinary calls at eachN determine recurring training cost. Keep both original natural allowances, optimizer/checkpoint costs,1.25multiplier,60-secondreserve and3600-secondcap. Root estimate3029.584956seconds; require independent CPU verification before separate main release. No full fit or fresh inference has run.


## 2026-09-12 — fixed ordinary joint-LoRA main release

Independent profile audit443909 COMPLETED0:0 in43CPU-job-seconds: all80native head rows PASS, maximumTV1.25256e-6, zeroCPU/GPUargmax differences, allsevenfixturesPASS. Independently verifies original28545.207520forecastFAIL and distinct uninstrumented-main3029.584956forecastPASS. Freeze main OWN3 plus168 ancestors and release ONE B200 job capped3600seconds. Same unfittedseed24 LoRA bytes, fresh optimizer,12completeepochs/2592presentations/324updates/accum8, final checkpoint only then all216 natural answers. Main uses no profiler, no prompt wrapper, no candidate mask and no extra decoder streams. Complete answer+canonicalEOS criteria remain206pooled/103eachorientation/16families. Computational success can contain negative competence. Require independent completed CPU main audit before unused91726342 inference; no confirmation release here.


## 2026-09-12 — outcome-independent joint-LoRA N16 confirmation preparation

Freeze new stager/proposal/CPUwrapper plus169ancestors. One600secondCPUallocation,4cores16GiB. Bind untouched91726342manifest810records and original order; prepare only270N16 native image prompts with no model/head/checkpoint or predictions. Preserve all810targets/270+540partitions. Record measured N16 profile envelope comparison and descriptive timing estimate; neither implies inference release. Main443917 continues unchanged. Fresh N16 inference still requires independently verified complete training competence; longer540input preparation/inference remain separate.


## 2026-09-12 — prospective N16 confirmation resource choice

CPU preparation443925 completed0:0 in214job-seconds. All270N16 inputs fit the measured3214-position N16 teacher envelope; independent-profile-derived natural inference estimate575.194030seconds. Before any full-training or fresh inference outcome, select ONE900-secondB200N16 allocation, allowing provenance/setup overhead beyond the estimate, plus one1800-second4core16GiB independentCPUreport. This is a prospective cap selection only: code review/freeze and passed independent main competence still precede actual inference release. No new GPU calibration or N32/N64 release.


## 2026-09-12 — ordinary visual-prefix reuse software check

Freeze OWN5 plus166V2ancestors. CPU300s prepares two exactN8/N16 sentinelimagecollections and firstfour canonicalquestions, noanswerlookup, plus native28layercachefixtures. Conditional singleB200GPU240s:8ordinaryfull trajectories,8freshcachebranches,8reverseorderbranches,2headlessimageprefixprefills,10visioncalls,atmost96headcalls/98decodercalls,zeroLoRA/compression/fitting. Require identicalfull-v-split IDs andfullvocabularyTV<=.02; branchorder IDs/logitsexact; originalprefixbytesalwaysunchanged andquestionKVindependent. Preserveall24outcomesbeforecomparisonfailure. This softwaretestdoesnotclaimaccuracyornewmemorybenefit.


## 2026-09-12 — conditional N16 evaluator source freeze

Freeze reviewed evaluator/reporter/proposal/twowrappers OWN5 plus175ancestors before the main efficacy result. Exactactualfinal324checkpoint and independentcomplete216competence required. Reserved270N16nativegenerations, allqueriesCPUheadreplayTV<=.02; actualGPUwholeIDs+EOS define panel/familymetrics. Prospectivecaps900GPU/1800CPU unchanged. Sourcesconcrete/reviewed; actual inference still held until mainaudit passes206/103/16. NoN32/N64release.


## 2026-09-12 — pinned original MMReD recovery

Freeze new CPUstager/proposal/wrapper plus2existingofficialhelpers. One600secondCPUjob4cores16GiB recovers fixedHF d7963ebac13621dd1105e7596d8ed4a4c442119a (66files82,014,317bytes) and originalrenderer source56c6ee7041d539c7273d42d5a6c4c5e922015e40. All18cells/39,600rows/all24answerfunctions andone-person-transition/five-person/six-room laws audited. Preserve/reportfullworld andexactprefixoverlaps without silentlychangingofficialsplits. Prepare four-task original4000train/400val/1000test metadata; primarytasks spend_together/where_spend, controlssteps_in_room/char_at_frame. N8/N16/N32primarytest,N64/N128held. This is anofficial4taskpilot, not24taskresult; originaltests historicallyexposed, notfreshconfirmation. Dataall under/mnt/data/gabriele/gnn_transformer. No downloadedsourceexecution/rendering/model/fitting/inference inthisrelease.


## Original MMReD renderer profile release (2026-09-12)

Freeze three new sources plus five recovered ancestors, source release `8d9b5781e04d2009ce675c0baa0af693d62063c8156ba33e8be46c4826870ccf`. One CPU Slurm allocation capped300seconds,4cores,16GiB, exactly64 upstream frame renders (56unique+6repeat+2sequence reference). Require original512x512 geometry and exact PNG/sequence identity, inventory5000worlds/40800frame occurrences, publish descriptive4-shard forecast. No model/fitting/full render/N64/N128 release. Root and independent attention review passed.


## Fixed native-value memory core release (2026-09-12)

OWN4+one frozen stable accumulator, releaseSHA`f62b316469d520eeef4eae35f7faf98413d9e5fb84e0885327dfdaa77a982f26`. One90second4core16GiBCPU Slurm check,8fixture groups. Fixed32slots/key128/native3584identityvalues,24global sine/cosine position features, private seed24,469504FP32params; mean versusmean+logZ*d withzeroinitiald. Known attention/multiplicity primitive, no sufficiency/efficacyclaim, noimage/model/training/nativeintegration release. Root source review passed.


## Original MMReD four-shard render release (2026-09-12)

OWN4+8 frozen releaseSHA`1db1855d0ff84b4c4ec7a0ccb4b4288d1f0660447d488e9cde6249b27a7ca587`. Passed64frame profile443954 used7CPU-job-seconds; forecasts3789.610s/shard. One4-taskCPUarray,4500seconds4cores16GiB per task,34708totalkeys8677each; one600secondCPUmerge. Originalupstreamrenderer/env namespace retained, rerenderallkeys, preserveStep/order/rawPNG.5000worlds40800occurrences (4000train400val600primarytest). NoN64/128/model/fittingrelease. Root reviewedcompletecandidatebeforeexecution.


## Original MMReD native input/runtime preparation release (2026-09-12)

OWN4+179 frozen releaseSHA`3730eaaff2e291cc14394f968013da31de29c54de7247431ba4116c1312d8147`. One600second4core16GiBCPU Slurm preparation; eightoriginaltraining-onlycases(N1/2/4/8/16char_at_frame+other3N16tasks), exactupstreamsystem/unprefixedQ,392RGB commonprocessor, typedJSON+EOS targetinventory across5000rows, nativeimage/teacherpositions, strictparser10invalidfixtures. Runtime coldcontinuousprefix/teacher/forked50tokenfreegreedy source reviewed butNOmodelcallinthisstage. Fullrender443959/443964 passed1089+26CPU-job-seconds,34708PNGs/5000worlds40800occurrences. NoGPU/fullfit/testinference released.


## Original MMReD native preparation V2 compatibility release (2026-09-12)

CPU443965 FAILED31seconds beforeprocessorconstruction: upstream Literal[*ROOMS] syntax unsupportedbyjobPython3.10. Preserveoriginal4sources/failure. NewOWN3+183 frozen releaseSHA`2b8c1aeddc20dacecd05e3afeb5e01548ed38a8d7be9665823fd358cd0b25c86`; extractonlyuniqueanchoredtriplequotedSYSTEM_PROMPT andliteral_eval, samecompleteupstreamhash andsystemequality; threeinvalidfixtures. Allscientifichelpers directoriginalaliases, sameeightoriginaltraincases/targets/pixels/positions/core/runtime. One600second4core16GiBCPUretry (cumulativecap631). NoGPU/fitting/testrelease. Root and independent hils reviewpassed, Python3.10AST/bashpassed.


## Original MMReD native preparation V3 content-container release (2026-09-12)

CPU443970 FAILED42seconds afterliteralrepair/targetcompilation, beforefirstimagecasepublication: multimodalprocessorrequiresstructuredassistantcontent. Preservebothpriorfailures31+42secondsandallsources. NewOWN3+186 frozen releaseSHA`8ed154d9bc65049c94d9a39b3edb11abd941700952410f725465bb50bf7db0f8`; copiedprepare_joint changesonlycontent=target tolistoftextblock, exacttarget/EOSproofretained; allothersciencealiasesunchanged. One600second4core16GiBCPUretry(cumulativecap673). NoGPU/fittingrelease. Root/hils reviewed ASTone-expressiondiff andidenticalV2main; Python3.10/bashsyntaxPASS.


## Original MMReD three-arm native software/cost profile release (2026-09-12)

NativeV3prep443974 PASS38CPU-job-s, totalpreparation111including73priorfailures, exact8traincases/maxJSON+EOS8tokens/29555trainingtargetrowsper4000epoch. FreezeproducerOWN3+194, releaseSHA`f3fefd045dc58d337ef05ff20e9cdaf314270b9315b0bffa29163935577a80f8`. One900secondB200,3armsordinary/normalized/mass each2updatesaccum8=48FB6updates;8visionfeatureextractions+2pixelteacherchecks=10visioncalls;4parityteachers12checkpointteachers29naturalsoftwaretrajectories+2headlessprefixes;<=1514nativehead/model1516decoder/norm. All24numericalcomparisonsretainedbeforegate, exactGPU pixel/feature+reload+reverseorder, cold/splitIDs+TV.02; no extraGPUheads, no efficacyor fullfit. IndependentCPU22teacher/allnaturalsTV.02, coreFP64rtol/atol2e-4/CE2e-6 preregistered; reporterfrozen separately. Trainingforecasts empiricalandexclude100diagnostic/val/testgeneration andPNGrecooking. Root/V10fullproducerreviewpassed.


### 2026-09-12: original MMReD compact training input release

Freeze four sources plus194 ancestors, root/hils/attention reviewed, releaseSHA 6d18f9d03f44cd2e0a3d6ce6fbee3c2db30ea2f195aeb6b9db78b42b6092885c. Run one CPU array4x900seconds and CPU merge600seconds,4cores16GiB each. Preserve4000original train rows/24800frame occurrences and fixed100diagnostic IDs. Omit only pixel_values while binding exact tensor identity. Full input widths and reprocessing times retained; width excess descriptive and holds future fit pending boundaryprofile. No model/feature extraction, GPU, fit or inference release.


### 2026-09-12: independent CPU audit of complete failed443978 profile

Freeze OWN3+197ancestors root/attention/V10 reviewed, releaseSHA 925ed826060baf5167ef4e861caa263b525a9b9e8a66ddda2ddf048615b4c3d6. CPU1800seconds4cores16GiB. Audit all64teacherCE, fixed22teacherheadexamples,29naturaltrajectories,42core reconstructions and all24original comparisons. CPUheadTV.02/coreatol=rtol2e-4/CEabs2e-6 unchanged. Top PASS means independentCPU numerical/provenance audit only; reproduce originalFAILED cachegate explicitly. Cold/training component flag separate, no fullfitrelease.


### 2026-09-12: shared original MMReD frozen native feature harvest

Freeze OWN5+204ancestors root/hils reviewed, source releaseSHA e8f77a972d25783e2def7d8fc34391635fe98c49c5640e458dbf50562acf0019. CPU resource/input check120seconds, conditional4GPUshards1800seconds each, CPUmerge600seconds;4CPU16GiB/task. Exactly4000per-world frozen native visioncalls, zero decoder/model/head calls. Same V3 reprocessing verified against allcompactpixel/grid/coordinate identities beforefeaturecall; nativeFP16common35GBfeaturecache usedoncebyall3arms. Resource pricing includes fullCPUstagingtime and profilefeaturemax with25percent margin. GPUconditional on passedCPUcheck AND everyforecast<=1800. No mainfit, validation/test or cachedprefixreuse release. Originalcachefailurepreserved.


### 2026-09-12: actual training-width boundary profile

Freeze OWN4+209ancestors, root/attention reviewed, releaseSHA b199e2e898933df43551054e621fa2c173a9b4c0d2725425fae302fa2b823047. One900secondB200 profile afterpassedfeaturemerge, then1800secondCPUaudit. Deterministicfirst-tie width maxima union yields12actualtrainingworlds,36fresh-initializationFB withzero gradsbetweenrows/zerooptimizerupdates,15coldnaturaltrajectories forsoftwarecostonly. Zero visioncalls, no efficacy score. Auditallcapturedteacher/core/head andnaturalhistories withunchangednumericalpolicy. Retainmax(old,boundary)perNtimings and originaloptimizer/checkpointcost, explicitestimatedAdamWmemoryreserve. No fullfitrelease or cachefailureoverride.


### 2026-09-12: original MMReD three-arm cold training release

Freeze OWN5+213ancestors, root/hils/V10 reviewed, releaseSHA f9078bf290049cb970d54913375fb792b3d5563d5c3a29064722269340ab19cd. CPUcheck600seconds thenconditional one3taskGPUarray7200seconds/arm;4CPU16GiB each. Original4000train,3epochs12000FB1500updates perarm; ordinary/normalized/mass sameunfittedinitialization+advancingseed24order+fullJSONEOSCE. Finalcheckpointonly then100fixedtrainingdiagnostics; no val/test inference. Sharedfeaturemerge444004PASS, boundary444007/444008PASS59GPU+31CPU-s, originalcacheFAILpreserved. Measured complete forecasts ordinary7040.786/norm5198.932/mass4575.277seconds include100diag and checkpointwork; devicepeak28.17/7.73/7.79GB<90%191.5GB. EverymainrequirespassedCPUcheck, no automaticretry/resume/selection. Independent per-armCPUaudit remains mandatorybefore accepting training results.


### 2026-09-12: original MMReD independent main-fit CPU audit

Freeze OWN3+218ancestors, root/V10/attention reviewed, releaseSHA a3afe0815bdbf2aedf8fb8e0410bbdd6db4ac8c73ddffaf64f54325ab7a09cb8. One3600second4core16GiB CPUreport perarm afterfull fit. Replay6teacher examples+all100natural heads, audit capturedcore andCE withunchangedtolerances, exact12000scalar/1500optimizer/order/state/native/feature/strictJSON joins; do notclaimallscalarCEindependentlyrecomputed. No extraGPU, backward, optimizer or val/test inference. Originalcachefailureunchanged.


### 2026-09-12: original validation/test inputs and shared native features

Freeze OWN4+209ancestors root/hils reviewed, releaseSHA 84255d028d827bc949fb95182b629ae224a6aa78771e57b506e0c0abeaa06883. Exact400val+600testN8/16/32,1000worlds16000frames. CPU600s thenconditionaloneGPU1800s4CPU16GiB. FullV3prepare/pixel-onlycompact proof andsamefrozenperworldnativefeaturecall. InitialN32visioncost estimate2xN16 measured, firstsourcecaseeachN8/16/32 comprises3of1000visionoutputs beforeexplicitremaining-workreforecastgate. No model/decoder/head/prediction, noN64/128. Common22.48GBFP16featurecache. Inputwidths/descriptivedecoderenvelope unvalidatedforN32; laterinference requiresseparaterelease.


### 2026-09-12: fixed final-checkpoint original MMReD evaluation and paired statistics

Before validation/test predictions, freeze reviewed evaluator OWN5+225 ancestors (releaseSHA 1f047c45163c3dcdffc493c8216dfc0268ca6c0b8c618256b7e0dde1d43f5aa5), independent auditor OWN3+230 (releaseSHA de6083c8d426a380c09c801f8eba9de99c3deaf27fc591373366e6c37aa741a7), and combined statistics OWN3+232 (releaseSHA 20c35509a21682fb9b56060c75fa2d8174bb6a20e42814ef81d27e7ce17f083a). Shared protocol e22eae44ada348fc919830524b86d5cbea755a9a17b851b02e0e049d1e0cd409. One CPU600-second evaluator check after shared features and all three independent main audits; one conditional three-arm B200 array10800seconds/arm, all1000 original rows/final1500 checkpoints irrespective of accuracy. Fixed firstN8/N16/N32 are included outputs and precede a timing-only remaining-cost gate. Strict typedJSON+nativeEOS50, cold fullprefix, no extra heads/vision/fit; all raw evidence preserved. Independent CPU7200seconds/arm audits all1000 trajectories/head rows and all1000 FP64 memory reconstructions in compressed arms. Combined CPU1800seconds,30 analyses x10000 paired full-world/bootstrap replicates, exact protocol/heuristic conditions and descriptive cost/strata/plots. FourCPU16GiB each; root limits concurrent projectGPUs to4. Failed/negative outcomes retained; no automatic N64/128, follow-up fits or reasoning release.


### 2026-09-12: pre-outcome reporting fixtures

One additional60-second four-core4GiB CPU Slurm allocation runs only the seven synthetic fixture groups in frozen combined reporter7f32ab89, before any benchmark evaluation outputs. No actual predictions, model calls or numerical policy change. Test wrapper SHA252f8475c912f61eb3c7c46f548cc788695e964ffd9c12ebb689a2f022ae8316. The fixtures also remain scheduled inside the final reporter; this isolated early software test does not replace those checks.


### 2026-09-12: posthoc scalar-only training dynamics

After observing the fixed training diagnostics94/82/75 of100, freeze a separate two-source descriptive reporter, releaseSHA6ab7379e1c5df9297e3e57e468c1f5e4d952399c5baf7726683855f5215f9c5f. OneCPU60-second4core4GiB job reads only36,000 existing sceneCE records and4,500 logged accumulated optimizer norms, with audit/row/order/hash joins. FullJSON/EOS scene and token-weighted CE byepoch/N/task, fixed finalepoch halves, loggednorm>1 fractions and type7 quantiles. No tensors/models/newaccuracy, no convergence/capacity conclusion, and no alteration of ongoing evaluation or later release.


### 2026-09-12: resource-only continuation of the fixed original evaluation

OriginalCPU444051PASS38s, originalGPU444052 remainsFAILED at its first3 timingcases perarm: forecasts11111.062/10983.993/10977.820 exceed10800; allocations33/32/33seconds98total. Freeze newproducerOWN4+230, releaseSHA21431c9f99f7f717940448e4c2b6f9f80304291672a31eaa51f8eb45575b97b8. OneCPU600second4core16GiB check, thenoneconditional3armGPUarray14340seconds each withold+new≤14400perarm. Preserve the exactoriginal3 outputs/rawfiles/IDs/scores/timings; generateonlyremaining997, sourceordered1000combined, no reruns/checkpointcopies/training changes. Samefitted1500states/nativeinputs/32slots/precision/greedy50/parser/statisticalprotocole22eae44. Retainoldconservativetimingbounds; actualnewsetup reforecast before997. Importedmodel/headcounters retained; originalper-layercountervector wasnotpersisted, so do notinvent it; new997 hasall28actualcounters. All1000head/core audits remainrequired in separatelyfrozenreport. No accuracy-dependentrelease. GPUsubmissionwaitspassedCPUcheck (do notprequeue).


### 2026-09-12: independent continuation audit and unchanged statistical report

Before anyremaining997 predictions, freeze independentall1000auditor OWN3+237, releaseSHA7f298b5cad78e3fc6ced9c0e8e31a347c67511a69675ed8a6c40b75adbcc77f9, CPU7200seconds/arm4cores16GiB. Exactoriginal3 plusnew997 samefinalstates/inputs, fullheadTV.02/all1000FP64memoryatol=rtol2e-4, same strictnativeJSONEOS scores; new997 measuredlayercounters, old3vectorunavailable explicitly. Originalfailed98GPU-s andnew/cumulativeallocationchargedseparately; originalrawanswers notrerun. Freeze combinedstatistics OWN3+242, releaseSHA8849cef886509952b5400aad3c7b92e8385d43097eb5a53ee1a3a7a36d6e0ca9, CPU1800seconds4cores16GiB; directlycallsfrozenoriginalbootstrap/qualification/plots/fixtures unchanged under e22eae44, exact1000×3cohortand30×10000replicates. Newonlylineage/accountingadapter plusmutationfixtures. No efficacygateortrainingchange. CPUcontinuation444096PASS7s, forecastcumulativeordinary11151.921/normalized11024.324/mass11019.507<14400.


## 2026-09-12: original-MMReD fixed-summary pilot closure

All scheduled 3,000 outcomes, including the nine unchanged outputs from resource-failed array 444052, were independently audited by 444108/444109/444110 and analyzed by 444111 with the unchanged e22eae44 statistical protocol. N32 primary ordinary/normalized/mass = 61/32/28%; mass−ordinary −33pp, 95% paired-world interval [−46,−19]. Ordinary validation competence passed; all outputs complete/valid. Close this fixed-summary configuration; no hyperparameter rescue or further-length/reasoning efficacy run. The distinct intact-evidence hypothesis and fresh suffix-pair feasibility work are prospective new studies, with no fitted benefit yet. See docs/paper/NATIVE_AGGREGATION_MMRED_OFFICIAL_PILOT_RESULTS.md.


## 2026-09-12: fresh suffix-counterfactual feasibility release

Freeze standalone driver748187f9, protocol2d634136, CPUwrapper34719ab2. Exactly20,000 fresh original-law worlds each N16/N32, fixed separate world/choice seeds; one random cut/actor/person-pair/room-pair per world. Test legal/non-global/changed/unchanged/equal-marginal/third-competitor subsets with independent integer/original oracles. CPU300s, sampling270s; DATA stores all generated worlds. Acceptance is not a completion gate. No model outputs, images, fit or final benchmark cohort. Original exposure overlap explicitly deferred. Root reviewed and releases this one CPU feasibility job.


## 2026-09-12: intact-evidence residual core CPU release

Freeze coref1a0f5d5, testsb85501eb, wrapper4369ae58, designef108bea: one32slotnormalized128wideQKVO readout versusrank321SiLU adapter, exactly2,300,929live FP32 parameters, matchedzero scalar gates, freshpool24 andreader25/26. Only8 algebra/gradient/interface fixturegroups in90sCPU Slurm are released. No native/model/head/image/optimizer calls; no fit or efficacy result. New architecture retains all original evidence and is standard cross-attention machinery, with empirical claims pending native checks and controlled fresh-world results.


## 2026-09-14: privileged training implementation and fresh cohort preparation

User authorization: proceed with three matched ordinary/local40/prefix40 continuations, using existing native image-end states and no additional inference computation. Fixed target source release `outputs/native_aggregation_vlm/mmred_prefix_supervision/target_source_release.json` SHA256 4bc99e53334a365ba9ef91b745f3fd3c514a129c0edd870057d8c264ae3f5656 was frozen before CPU447609. That job passed in19 allocated CPU-job seconds:4000 original training entries,24800 image boundaries,7 core test groups and3 position fixtures. Target plan SHA256 8538f51580a95ec9183fe59faae6897cb43ec4945251255e372d5fd3fb2b2d00. No efficacy result.

Fresh metadata release `outputs/native_aggregation_vlm/mmred_fresh_evaluation/source_release.json` SHA256 07f142b3514e9854ff875cdb4110435181ab690c1dfa88dac5e7dd43e875af31 was frozen before CPU447629 (600seconds,4cores,16GiB). Fixed2040world/2360question target, main1400 processed first; no predictions and no automatic quota or seed rescue. Both release archives preserve exact preexecution sources; the append records already submitted job identities. The raw summary-residual method remains held.

Fresh metadata447629 completed in487 allocated CPU-job seconds, full fixed2040world/2360question quotas,160diagnostic families,47232frame occurrences. All2360 serialized answers independently recomputed; plan SHA256 dfc58ee1daebf38d45e0c1deab54406a3e179aef646113eeb3270936d3687642. No model predictions.

Fresh native main1400 source release frozen before execution: OWN6+222 ancestors, SHA256 2f464f6daacef1c0f433f8f93df86494350145a28edb4aa6852980e93b4def85. CPU prepare447665 submitted600seconds,4cores,16GiB. Four conditional CPU render/compact shards3000seconds each, merge600seconds, then one measured-gate B200 harvest3600seconds. Exact unaltered frozen renderer and V3 processor/vision math, main1400 only, no adapter or predictions. See docs/paper/MMRED_FRESH_NATIVE_INPUTS_PROTOCOL.md.

Native privileged GPU source release frozen OWN10 and all ancestry,241 total sources, SHA256 a066ef7fb567765fb91345dc9a6f9a4e30cde1ed9e6281cc5b26d504961dd31f. Independent source reviews passed. GPU software profile447668 submitted900seconds/oneB200; CPU independent audit447669 queued afterok900seconds/4cores. Exact92teachers/72backwards/2auxiliary-gradient probes/6updates/8native responses, fixed training witnesses only. Main six-fit release remains conditional on exact bound profile/audit/forecast. Fresh native CPU447665 passed7seconds, plan5268da02ca6093850ad110d5940c7336e97b7273a8042fdb5ae893ddb141269f; four render+compact workers447670 submitted3000seconds each.

Software GPU447668 passed100 allocated GPU-seconds. Exactly156native model/head calls (92teachers+64generatedtokens),72ordinary backwards,2auxiliary probes,6updates,0visioncalls. Forecasts answer5163.99/local4349.62/prefix4483.09seconds fit7200cap; independent audit447669 running. No efficacy gate or extrapolation predictions. Fresh CPUmerge447674 queued afterok447670.

Independent profile CPU447669 passed48seconds:156head calls,788actual head rows,90auxiliary FP64 captures,9states/6gradient packets,all numerical checks and forecasts pass. Main release SHA256 8cce361c374efb6b0a221c5da0696f69496223ec819864f8a9e867572ebd95aa. Six matched continuation fits array447689 submitted (0answer25,1local25,2prefix25,3answer26,4local26,5prefix26), max3GPUs,7200seconds each, fixed final1500update state. No checkpoints selected by accuracy. Fresh statistics source SHA256 687b1bf5eb192a204aaa5da10b6f0527c2a9e16feecc9bed94610a2c254f4e9e frozen before any fresh prediction; primary paired seed-mean prefix-minus-answer on400N32aggregation worlds, stratified bootstrap with conditional seed scope and ordered prefix-minus-local claim.

Fresh native render/compact array447670 completed all1400 questions in four allocations381/381/380/380seconds; merge447674 passed23seconds. Merge plan SHA256 af96681fd6180467d2ce463e3115630ccdfaaada882e7431727260d36e600c1b, initial shared-vision forecast2080.83seconds fits3600cap. One B200 harvest447702 submitted, running alongside at most3model-fit GPUs; no decoder/head/model calls or adapter changes allowed, exactly1400visual calls.

Main independent CPU audit OWN3+241 frozen ancestor release SHA256 38f4e57a3b9a4f87db307ce3131c335c3d6427ad9485c97e6b1203fc37e2c4b7. Six3600second/4core audits447704 through447709 queued after matching mainarray447689 tasks0through5. Full12000scalar exposures/1500updates/3states/6teachers/100training diagnostics, all native heads, exact initialseed/RNG and fixedtarget bindings. Final checkpoints retained irrespective of training accuracy; fresh evaluation still requires its separate execution protocol.

Fresh seven-checkpoint evaluator OWN4+270 ancestors frozen before any fresh prediction, release SHA256 624b8a9996086af28c896aa2479bc659780040ab9dabf0207f024e9687b9011a. CPUcheck447723 queued600seconds after feature447702 plus all6main audits447704–447709. CPU plan must bind exact fixed1400worlds, common checkpoint/initialRNG training proofs, original ordinary anchor and measured prior timing. GPUevaluation is not yet released; prospective7task cap3600seconds each/max3concurrent, unchanged ordinary native generation, complete1400outcomes and independent numerical audit required before statistics.

Complete fresh-evaluation/reporting pipeline conditionally queued before any predictions. A separate CPU-only release gate447757 (300seconds/4cores) runs afterCPUcheck447723 and calls the frozen verify_plan, requiring resource eligibility before exit0. Its OWN2+274ancestor release SHA256 d6ce3d0b422865b2c89cfd1c11125fc0e6162c77ad2edf528eb43f3f539d61d9. GPUarray447759 depends afterok447757; task order anchor,answer25,answer26,local25,local26,prefix25,prefix26, each3600seconds/max3 concurrent. Thus an ineligible CPU plan cannot allocate downstream evaluation GPUs. Seven independent CPU audits447762/447763/447764/447765/447766/447767/447769 depend on corresponding GPUtasks, each7200seconds/4cores, frozen OWN2+274 release8cfe8242e69817005932c86a05f1832401674c6bd3969566d16e7e7fee9984bc. Final paired statistics447770 (600seconds/4cores) depends on all seven audits, frozen OWN3+276 release69db1d4083ae0ee8579be6d787a1e6219d6766ecdde6370a8db5653c820d814b. No accuracy-dependent selection or automatic retry; all failures preserved. Full source/AST/wrapper and independent statistical/source reviews passed; numerical fresh outcomes remain unavailable.

Shared fresh visual harvest447702 completed1435 allocated GPU-seconds. All1400worlds/28800frames/5644800nativeFP16 feature rows verified; exactly1400vision and0model/decoder/head calls. Final feature plan SHA256 d24d2221893fb2c94238cf4a698a1a6b9ba616c3e955d253bbf67afeaf5a8b80. Metadata/ownership/forecast handoff verified; actual payload hashes will be checked again by consumers. This is shared input preparation, not model accuracy.

# Pre-registered predictions — over-squashing at the read node (written 2026-09-02, before the confirm runs)

Purpose: make the GNN over-squashing framing load-bearing. Every prediction below is
numeric, was written before the corresponding runs were launched, and names what result
would falsify it. Results are appended to CONDMASK_REPORT.md when in; nothing here is
edited after the fact (corrections go in a dated "outcome" column).

Instruments: `scripts/condmask/osq.py` (answer-row fan-in `keff_sent`/`keff_tok`,
evidence mass, pre-softmax margin `gap`, dispersion prediction `pred_mass`, Jacobian
influence share `infl`) run through `eval_babilong.py --probe`; CPU-validated on planted
scores (`tests/test_babilong_osq.py`). Selection layers: Qwen2.5-3B L27, 7B L22,
Llama-3.1-8B L20 (BABILong scans). k = 64 sentences. n = 200 (<=32k) / 100 (64k, 128k).

Existing numbers referenced (native RoPE, one-forward selector unless stated):
Llama 128k recall .61/.61/.43/.94/.70 (qa1-5), chunked(32k) .65/.61/.59/.96/.78;
Llama 32k one-forward recall .95/.87/.67/-/.90. Qwen-7B 128k chunked recall
.82/.76/.76/.97/.92; Qwen-3B .83/.76/.73/.97/.89 (YaRN .92/.92/.78/.97/.95).

## Definitions

- Degree axis = the read node's effective fan-in grows with N (over-squashing proper:
  Alon-Yahav's fixed-width squeeze / the normalisation term of Di Giovanni et al.'s
  Jacobian bound). Position axis = relative-position (edge-length) OOD. The theory
  claims the first and is silent on the second; the method fixes both (select = degree,
  repack = edges).
- A full-arm cell "succeeds" if acc(full) >= acc(oracle) - .10.

## P1 — Tournament selection (theory-dictated design: every comparison inside one bounded softmax)

Llama-3.1-8B, 128k, chunk 32k, k_local 64: fact recall >= one-forward recall at 32k
minus .05, i.e. qa1 >= .90, qa2 >= .82, qa3 >= .62, qa5 >= .85 (from .65/.61/.59/.78
chunked). Repack accuracy follows: qa1 >= .85, qa2 >= .50.
Qwen-7B and 3B at 128k: tournament recall >= chunked recall - .03 on every task.
FALSIFIED IF Llama recall stays within +-.05 of the chunked values on qa1/qa2: then the
selector's failure is not the cross-window comparison / dilution the theory names.

## P2 — Dispersion law (no free parameter)

Full arm, frozen, all 3 models x 5 tasks x {4k..128k}: `pred_mass` (from the mean
margin) tracks measured `mass`: Spearman > .8 over cells; |pred - mass| <= .10 in >= 70%
of cells. Margin premise: the frozen `gap` does NOT compensate for length — gap(128k) -
gap(4k) < 1.0 for every model/task (full compensation would need log(32) = 3.5).
FALSIFIED IF gap grows by >= 2.0 from 4k to 128k (the model re-calibrates the margin
with length; over-squashing would then not be structural) or Spearman < .5.

## P3 — Fan-in growth and the factorial separation

Full arm: `keff_sent`(128k) / `keff_sent`(4k) >= 4 and mass(128k) <= .5 x mass(4k), every
model, every task. Repack (attn): `keff_sent` varies < 2x between 4k and 128k. No-repack
(attn_norp): `keff_sent` within 1.5x of attn's at the same cell, while accuracy differs
by >= 12pp on qa1/qa2 at 128k (Llama native, Qwen native) — the metric is silent on the
position axis by construction. Under YaRN x4 (Qwen): full still has growing fan-in and
trails oracle by >= 15pp on qa1/qa2 at 128k (degree axis alone).
FALSIFIED IF full's fan-in is flat in N on any model (then the read is not diluting) or
norp's fan-in differs from attn's by > 2x (then repack changes the read, not just the edges).

## P4 — Sensitivity chain and the necessity invariant

Per cell (full arm): Spearman(infl, mass) > .7 within each model. Per example: pooled
logistic regression acc ~ infl has a positive coefficient, p < .001, within each model.
Necessity: evidence `mass` of the full arm separates succeeding from failing cells with
AUC > .85 (90 cells); no succeeding cell has `keff_sent` above 3x its own 4k value.
Norp cells with high mass but low accuracy are expected (bounded degree is necessary,
not sufficient) and do not count against this.
FALSIFIED IF a full-arm cell succeeds with mass below the failing cells' median, or AUC < .7.

## P5 — Position dose-response (degree fixed at the oracle set, edge length P)

0k split, n=200, oracle_gap@P for P in {0, 2k, 4k, 8k, 16k, 24k, 32k, 40k, 48k, 64k, 96k, 120k}.
Qwen native: accuracy within .05 of P=0 for P <= 24k; at P >= 40k accuracy <= .5 x acc(P=0)
on qa1/qa2. Qwen YaRN x4: within .10 of P=0 through 96k. Llama: monotone decline, at
P=120k >= 10pp below P=0 on qa1/qa2 (its in-window norp deficit reproduced without junk).
oracle_pgap@P (gap between prefix and evidence): effect smaller than oracle_gap at every
P for both families. Read-node metrics under the gap: if mass/keff stay within 1.5x of
P=0 while accuracy collapses, the axes are independent (claimed); if they collapse too,
position acts through the read's attention — reported either way.
FALSIFIED IF Llama is flat to 120k (then its 128k in-window failure is pure dilution and
repack's gain over norp on Llama needs another explanation).

## P6 — Rounds = problem radius (Alon-Yahav)

Two-round selection (copies of round-1 sentences appended to the tail), Qwen-7B and
Llama, 64k/128k, tournament selector: qa2 and qa3 recall >= one-round + .05; qa1 within
+-.03; qa2 accuracy >= one-round + 5pp.
FALSIFIED IF qa2/qa3 recall change < .02 (the second hop is not what limits them).

## P7 — k-identity (repack is degree control, not magic)

Qwen-7B and Llama, 128k, chunked selector, k in {16,32,64,128,256,512,1024,2048}:
accuracy rises with k while recall is the limit, then falls once recall >= .9; on the
falling branch acc(attn@k) is within 10pp of acc(full) at the length whose sentence
count matches k (k=1024 ~ 16k-context, k=2048 ~ 32k). `keff_sent` of the repacked read
grows with k like full's grows with N.
FALSIFIED IF accuracy is flat in k beyond recall saturation (then the repacked read does
not dilute and the dispersion law does not apply inside the prompt).

## Analysis plan (fixed)

Cells = (model, task, length); statistics as stated; per-example joins on (model, task,
length, i) within a run. Bootstrap 95% CIs (1000 resamples) on every accuracy and recall
reported. Runs: `outputs/babilong_osq/` (probe grid), `outputs/babilong_tourn/`,
`outputs/babilong_rounds/`, `outputs/babilong_gap/`, `outputs/babilong_ksweep/`.

## Outcomes (dated; predictions above are never edited)

- 2026-09-02 (before any grid landed) — **P6 mechanism failed its design smoke; ROUNDS grid not launched.**
  Two-round selection (round-1 kept sentences copied into the tail; copy rows read the context)
  reduced recall on Qwen-3B 4k k=16 n=10 in both variants (v1 tail-mean rows, v2 copy rows only,
  self-hits zeroed): one- vs two-round recall qa1 1.00/.86, qa2 .71/.47, qa3 .24/.15
  (logs/dbg_sel-415422.out). A per-layer scan of the copy rows found no layer with hop-2 signal
  (best qa2 .54 @L33 vs .71; qa3 .19 @L25 vs .24; top-8 precision of the copy-row scores never above
  chance; logs/dbg_rscan-415456.out). P6 is therefore UNTESTED by this instrument — not falsified.
  The analysis reports the observational stand-in (selector recall by hop count) labelled as such.
- 2026-09-02 — run roots added to the analysis plan at launch time: `outputs/babilong_osq_yarn4/`
  (Qwen YaRN x4, 64k/128k), `outputs/babilong_tourn8k/` (Llama tournament at 8k windows),
  `outputs/babilong_gap_yarn4/` (P5 under YaRN). `outputs/babilong_tourn/` is subsumed by the
  probe grid (its attn arm IS the tournament selector). Analysis code: `scripts/condmask/analyze_osq.py`.
- 2026-09-02 — **T8K landed** (`outputs/babilong_tourn8k/`, Llama, tournament windows 8k instead of
  32k, k=64, 64k/128k, n=100; logs/osq_t8k-415441.out). Recall 8k vs 32k windows: qa1 .903/.784 vs
  .859/.759; qa2 .744/.648 vs .744/.686; qa3 .583/.537 vs .588/.589; qa4 .980/.970 vs .980/.970;
  qa5 .811/.767 vs .837/.814. Accuracy 8k vs 32k: qa1 .90/.84 vs .85/.85; qa2 .44/.44 vs .50/.32;
  qa3 .37/.40 vs .44/.38; qa4 .55/.57 vs .58/.55; qa5 .77/.78 vs .79/.77. Post-hoc reading: a smaller
  window does not raise recall in general (qa1 +2–4pp, qa2/qa3/qa5 −4–5pp at 128k), so the residual
  selector loss is not within-window dilution but the fixed k=64 budget against a pool that grows with
  N (P7 k-sweep measures this). The LL qa2 128k tournament-vs-chunked accuracy anomaly (.32 vs .43)
  is not robust: .44 with 8k windows at similar recall.
- 2026-09-02 — **two instruments added, pre-registered before their runs land.**
  (a) *LSE split* (`outputs/babilong_lse/`; osq.py now stores `lse_e`, `lse_j`, `smax_e` with
  logit(mass) = lse_e − lse_j; arms full/attn/attn_norp, qa1+qa2 on Q3/Q7/LL + LL qa5, all six
  lengths, n=100, attention probe only). Predictions: (i) in `full`, 4k→128k, the junk side rises by
  LESS than the uniform-dilution value log(N128/N4) ≈ 3.47 (heavy-tailed junk scores: the LSE tracks
  the top junk scores, not the count) AND the evidence side falls (smax_e and lse_e decrease) — i.e.
  Gollapudi et al.'s Table-9 pattern (gold-score erosion > junk-LSE rise) also holds on frozen general
  LLMs; (ii) attn_norp vs attn at 64k/128k (same 64 sentences, only positions differ): more than half
  of the logit(mass) deficit of norp is carried by Δlse_e (far evidence scores lower), not by Δlse_j.
  Falsifier of (ii): Δlse_j ≥ Δlse_e in magnitude — the distance axis would then act by raising junk
  (sink-like) rather than eroding evidence. (b) *Stage-loss trace* (`outputs/babilong_stage/`; Llama,
  qa1/qa2/qa3/qa5 at 64k/128k, n=100, tournament 32k/64; per pruning stage: fact recall and each
  fact's within-window rank). Prediction: recall is lost mostly at stage 1 (local windows) rather than
  at the final softmax, and the lost facts sit at within-window ranks just past the k_local cut
  (ranks 64–128), i.e. the loss is budget, not blindness. Falsifier: lost facts ranked deep (>256)
  inside their window → the selector does not see them at all.
- 2026-09-02 — **horizon test with a 1M-context model, pre-registered before the run** (`outputs/
  babilong_1m/`; Qwen/Qwen2.5-7B-Instruct-1M — same architecture as Q7, continued-pretrained to 1M
  with rope_theta 1e7; selection layer chosen by `--scan` at 4k/qa1 BEFORE looking at any long-N
  result; arms full/oracle/attn/attn_norp with the attention probe, qa1+qa2 at 4k/32k/64k/128k, n=100).
  The two-axis reading says: the distance axis (norp ≪ attn beyond ~32k) is the model's *trained
  horizon*, the degree axis (full ≪ attn at every N) is dispersion and is horizon-independent.
  Predictions: (i) at 64k/128k, attn_norp accuracy is within 5pp of attn on qa1 and qa2 (Q7 gap at
  128k: qa1 .90 vs ≤ .50, qa2 .45 vs ≤ .20 in the probe grid), and lse_e(norp) − lse_e(attn) ≥ −0.5
  nats; (ii) full at 128k still trails attn by ≥ 15pp on qa1 and mass(full,128k) ≤ .5·mass(full,4k)
  — the horizon fix does not touch the degree axis. Falsifiers: (i) fails if the norp deficit at 128k
  stays ≥ 15pp (then "distance" is not trained-horizon but something in the position code itself);
  (ii) fails if full ≈ attn at 128k (then long training also repairs dispersion, and the degree axis
  is not a frozen-model constant but a training artefact).
- 2026-09-02 — **STAGE landed** (`outputs/babilong_stage/Llama-3.1-8B-Instruct/`, logs/osq_stage-416245.out;
  n=100 per cell, 32k windows × k_local 64, final k=64; the empty `qa1_64k/20260902_140841_*` dir is an
  aborted first attempt, the canonical one is `qa1_64k/20260902_151947_2222493`). Share of the lost
  facts that were already lost at stage 1 (local windows), 64k / 128k: qa1 .80 / .86, qa2 .52 / .65,
  qa3 .40 / .21, qa5 .32 / .34. Within-window rank of the stage-1-lost facts (k_local cut = 64):
  median qa1 130 / 176, qa2 140 / 190, qa3 100 / 96, qa5 125 / 114; fraction in [64,128): qa1 .49 /
  .36, qa2 .46 / .36, qa3 .67 / .72, qa5 .52 / .54; fraction deeper than 256: qa1 .30 / .35, qa2 .26 /
  .40, qa3 .04 / .07, qa5 .19 / .21. Reading: **mixed.** "Lost at stage 1" holds only for qa1 (and
  half for qa2); for the many-fact tasks (qa3 ≈ 46, qa5 ≈ 16 vocab sentences per sample) most loss is
  at the final k=64 cut — a pure budget effect (the pool of finalists still holds 85–94% of the facts).
  "Just past the cut" holds for qa3/qa5 (52–72% in [64,128), ≤ 21% deep) but the falsifier fires
  partially on qa1/qa2: 26–40% of their stage-1-lost facts rank deeper than 256 inside a 32k window.
  Caveat (instrument): "facts" = every sentence in the task's fact vocabulary, so on qa1/qa2 the
  deep-ranked losses include superseded location sentences that the answer does not need; the trace
  cannot separate blindness from justified down-weighting there. Net: on Llama the residual selector
  loss is budget-dominated (consistent with T8K and the k-sweep), with a single-fact-task minority that
  the read node does not see even inside a bounded window.
- 2026-09-02 — **1M-model selection layer: tie-break added BEFORE any long-N run of that model.** The
  pre-registered 4k/qa1 scan (k=64, n=100; `outputs/babilong_1m/Qwen2.5-7B-Instruct-1M/qa1_4k/20260902_163131_2236352`,
  logs/osq_scan1m-416254.out) saturated: recall 1.000 at every layer 11–27, so it cannot choose a layer.
  Tie broken with the same unsaturated protocol that chose L27/L22/L20 for Q3/Q7/LL — `--scan` on
  qa3+qa1 at 16k with k=32, n=100 (`qa3_16k/20260902_163514_2239342`, `qa1_16k/20260902_163613_2240940`,
  logs/osq_scan1m_qa3-416640.out), picking the argmax of the qa3 scan: **L22** (qa3 recall .577; next
  L19 .564, L24 .534; qa1_16k is again near-saturated, .997 at L14/L19, .995 at L22). L22 is also the
  base-Q7 layer, so the 1M factorial is a same-layer comparison. Predictions above unchanged; the
  factorial (`bash slurm/osq_grids.sh M1M` with LAYER_1M=22) is launched after this note.
- 2026-09-02 — **P1–P5 verdicts** (all pre-registered runs of these landed: probe grid `outputs/babilong_osq/`
  jobs 415424–415438; YaRN `outputs/babilong_osq_yarn4/` 415439–415440; GAP `outputs/babilong_gap/`
  415442–415444 and `outputs/babilong_gap_yarn4/` 415445–415446; statistics exactly as in the analysis
  plan, `scripts/condmask/analyze_osq.py` → `outputs/_scratch/osq_figs/ANALYSIS_partial.md` 16:41).
  **P1 mixed.** Llama 128k tournament recall qa1/qa2/qa3/qa5 = .759/.686/.589/.814 — all four targets
  (.90/.82/.62/.85) missed; but paired Δ(tournament − chunked) = +.113 [.07,.15], +.074 [.04,.10],
  +.004, +.038 [.01,.06], so the falsifier (within ±.05 of chunked on qa1/qa2) does not fire. Repack acc
  qa1 .85 (target met), qa2 .32 (target .50 missed; paired Δacc vs chunked −.11 [−.21,−.02], the anomaly
  T8K showed is not robust). Qwen: tournament ≥ chunked − .03 on every task (Q3 +.105/+.132/+.031/0/+.073,
  Q7 +.099/+.101/−.003/0/+.021). Direction confirmed, Llama magnitudes were set from its 32k one-forward
  recall and are not reached.
  **P2 FALSIFIED as stated; ordinal law holds.** Spearman(pred_mass, mass) over 90 cells = .876 (> .8);
  |pred − mass| ≤ .10 in 2% of cells (target 70%) — pred_mass from the *mean* margin overestimates
  mass everywhere; post hoc the mean margin exceeds the mass-implied margin m_eff by 3.0 nats on average
  (0.4–6.9): junk scores are heavy-tailed, dilution is not uniform. Margin premise: gap(128k) − gap(4k)
  ≥ 2.0 on 8/15 model×task (Q3 qa2/3/5 +2.1/+3.6/+3.8, Q7 qa2/3 +2.6/+3.3, LL qa2/3/5 +2.2/+3.1/+2.4)
  → the falsifier fires. Post hoc: the mass-governing m_eff moves far less (−3.4..+2.7; holding mass
  would need +3.5) and mass(128k) ≤ .5·mass(4k) in 14/15 cells — the model re-calibrates the mean
  score, not the mass. The law survives only as an ordering (which cells disperse more), not as a
  no-free-parameter prediction of the mass value.
  **P3 FALSIFIED in 7/30 falsifier checks; the factorial separation holds.** full keff(128k)/keff(4k) ≥ 4
  in 6/15 (Q3 qa1/qa4 4.5/4.5; Q7 qa1/qa2/qa4/qa5 4.9/4.3/8.7/6.5); flat (< 1.5×) on Q3 qa3 1.47 and
  LL qa2/qa3/qa5 1.38/1.06/1.34 — Llama's read is already wide at 4k (keff 24–50 vs Qwen 5–16) and
  saturates near 42–61, where Qwen-7B arrives at 128k (66–78). mass halves 4k→128k in 14/15 (Q7 qa3
  .31→.20 the exception). attn keff varies < 2× in 15/15. norp keff within 1.5× of attn fails in
  9/15 and > 2× on Q3 qa3, LL qa1, LL qa3: **removing the repack contracts the read** (LL qa1 128k keff
  14.9 vs attn 30.1; Q7 qa1 9.9 vs 17.4) — the far evidence is not merely unread, the whole read
  narrows onto near/prefix tokens, so the position axis does touch fan-in, in the direction opposite
  to dilution. acc(attn) − acc(norp) ≥ 12pp at 128k on qa1/qa2 in 6/6 (Q3 +48/+26, Q7 +36/+22, LL
  +45/+14). YaRN ×4 (128k): full trails oracle by 34/20 (Q3 qa1/qa2) and 27/23pp (Q7) — all ≥ 15pp —
  with full's fan-in still growing 64k→128k (Q3 qa1 20.4→24.2, Q7 qa1 42.2→49.6, qa2 43.9→50.8; Q3
  qa2 18.0→13.8 the exception) and full mass .09–.36: the degree axis is untouched by the position fix.
  **P4 FALSIFIED as pooled; holds within model.** Spearman(infl, mass) Q3 .968, Q7 .948, LL .904 (all
  > .7). Logistic acc ~ infl: Q3 +.34/SD p = 6e-12, Q7 +.51 p = 9e-22, LL +.08 p = .10 (fails; post hoc
  LL qa1 +.58 p = 2e-5 and qa2 +.57 p = 8e-7 positive, qa4/qa5 negative — the counting/listing tasks do
  not reward influence on the gold sentences). Necessity: pooled AUC(mass) = .772 (target .85; the
  falsifier's < .7 clause does not fire) but 10 succeeding cells sit below the failing cells' median
  mass (.154), all Llama (e.g. LL qa1 8k mass .133, acc .89 vs oracle .90) → that falsifier fires; 4
  succeeding cells have keff > 3× their own 4k value. Post hoc per model: AUC .904 / .901 / .925 — the
  mass scale is not comparable across models (Llama reads flatter at every N), so necessity is a
  within-model statement.
  **P5 mixed (falsifier does not fire); two regimes on the position axis.** Position-id gap P between
  evidence and tail, degree fixed at the oracle set (n = 200; `mass` ≡ 1 by construction, `keff_sent`
  within 1.5× of P = 0 at every P for every model — the read's *distribution over evidence* does not
  move while accuracy does: axes independent at that level). Qwen native, qa1 acc by P: Q3 .90 → .82
  (2k) → .72 (16k) → .46 (24k) → .53 (40k) → .20 (48k) → ≤ .01 (≥ 64k); Q7 .98 → .82 (2k) → .84–.81
  (4–24k) → .67 (32k) → .54 (40k) → .09 (48k) → .05 (96k). "Within .05 through 24k" fails — a step of
  8–16pp appears already at P = 2k and holds as a plateau to ~32k; the "≤ .5× at ≥ 40k" collapse lands
  at 48k, not 40k (40k: Q3 .53, Q7 .54, Q7 qa2 .46 vs .59). The collapse is visible in the read as
  `ctx_share` (weight the tail row gives the context at all): Q3 qa1 .24 → .00, Q7 qa2 .21 → .03 — the
  read withdraws from the context altogether (prefix/tail capture), not from particular sentences.
  Qwen YaRN ×4: no collapse (Q3 qa1 .52–.59 from 32k to 120k, Q7 qa1 .86 at 40–64k, .74 at 120k;
  ctx_share stays .04–.11) but "within .10 of P = 0 through 96k" fails — the early step is larger under
  YaRN (Q3 .87 → .60 at 4k). pgap (gap between prefix and evidence, evidence→answer edges short) is
  flat for Qwen at every P (Q7 qa1 .94–.99), so the effect is edge length answer→evidence, not absolute
  position; pgap < gap at every P for both families. Llama: qa1 .99 → .94 (4–24k) → .80 (40k) → .69
  (64k) → .19 (96k) → .69 (120k); qa2 .52 → .47 (64k) → .26 (96k) → .38 (120k): ≥ 10pp below P = 0 at
  120k on qa1 (30pp) and qa2 (14pp) as predicted, but **not monotone** — a trough at P = 96k across
  tasks (qa4 .16 vs .53 at 120k, qa5 .09 vs .23; also in pgap qa1 .70 vs .84) that a smooth
  horizon story does not predict (position-code specific; flagged, unexplained). Llama's ctx_share is
  .01–.03 throughout and does not track its decline. Net: for Qwen the position axis = a small
  edge-length cost from 2k plus a trained-horizon cliff at ~48k that YaRN removes; for Llama a slow
  decline with a resonance-like trough — in both, the degree metrics of the read are unchanged.
- 2026-09-02 — **P7 landed: mixed, falsifier does not fire; the k-identity holds at the token level.**
  (`outputs/babilong_ksweep/{Qwen2.5-7B-Instruct,Llama-3.1-8B-Instruct}/qa{1,2,3}_128k/`, jobs 415447 /
  415448, n = 100, chunked 32k selector, k ∈ {16…2048}, one scoring pass.) Disclosure first: the
  mapping written above ("k=1024 ~ 16k-context, k=2048 ~ 32k") assumed ~16 tokens per sentence; the
  contexts have 33 (16k ≈ 438 sentences, 32k ≈ 935, 64k ≈ 1760, 128k ≈ 3527), so k=1024 ≈ 40k and
  k=2048 ≈ 75k by sentence count — which coincides with the analysis plan's token-matched rule
  (repacked prompts: k=1024 → 45.5k tokens, k=2048 → 79.6k; selected sentences run 44 tok, 1.3× the
  average). Shape: Q7 qa2 rises .38 → .50 (k=128, recall .84) then falls to .15 (k=2048, recall .99);
  LL qa2 .32 → .43 (k=64) → .19; LL qa3 .34 → .45 (k=64) → .29–.34 — the predicted rise-then-fall.
  Q7 qa1 (.91 → .39) and LL qa1 (.84 → .50) only fall: at k=16 they are already at their ceiling
  although fact recall is .63/.47 (qa1 needs the LAST location sentence; "recall" counts superseded
  ones too — same caveat as STAGE). qa3 is flat on both models (.33–.40 / .29–.45) with oracle-level
  accuracy ≈ .3–.4: a task floor, not a test of dilution. Falling-branch identity (acc(attn@k) within
  10pp of full at the token-matched natural length, nearest of 4k…64k): 16/20 comparisons pass; the
  four misses are all k=2048 (80k tokens matched to 64k: Q7 qa1 .39 vs .71, qa2 .15 vs .31; LL qa1 .50
  vs .67, qa2 .19 vs .27) and all four pass against the post-hoc 64k/128k interpolation (full at 128k:
  Q7 .24/.16, LL .29/.16). Read metrics: keff_sent of the repacked read grows with k (Q7 qa1 6.5 →
  80.8, ≈ full's 78 at 128k; LL 8.7 → 66 by k=256 then saturates 48–66 — the same ceiling as its full
  arm) and mass falls (Q7 qa1 .26 → .08, LL .12 → .03), i.e. the dilution law acts inside the
  repacked prompt exactly as in the natural one. Net: repack helps only through k (degree control);
  a 2048-sentence repack is as bad as an 80k natural context.
- 2026-09-02 — **LSE split landed** (`outputs/babilong_lse/{Qwen2.5-3B,Qwen2.5-7B,Llama-3.1-8B}-Instruct/`,
  jobs 416246 / 416247 / 416248; n=100; qa1+qa2 (Llama also qa5) × 4k–128k; arms full / attn /
  attn_norp; probe at the selection layer with the three new columns lse_e / lse_j / smax_e).
  (a)(i) full 4k→128k. Δlse_j: LL +2.96 / +2.63 / +2.96 (qa1/qa2/qa5), Q7 +0.95 / +1.42, Q3 +5.78 /
  +6.26 — "< 3.47" holds on Llama and Qwen-7B, FAILS on Qwen-3B (its junk side rises MORE than
  uniform dilution: −1.63 → +4.15 nats on qa1). Δlse_e: LL −1.04 / −0.54 / −0.32, Q7 −3.64 / −1.06,
  Q3 −1.68 / +1.69 — falls in 6/7 (Q3 qa2 rises). Δsmax_e: LL −0.49 / −0.07 / +0.10, Q7 −3.19 /
  −0.73, Q3 −1.44 / +1.88. Verdict (i): holds on 4/7 (LL qa1, LL qa2, Q7 qa1, Q7 qa2), marginal on
  LL qa5 (evidence max flat), fails on Q3 (both tasks). Post-hoc reading — three regimes, one per
  model: Qwen-7B loses mass by EVIDENCE EROSION (qa1: lse_e −3.6 vs lse_j +1.0 = Gollapudi's Table-9
  pattern), Llama by near-uniform DILUTION (lse_j +2.6–3.0 ≈ log 32, lse_e −0.3…−1.0), Qwen-3B by a
  junk tail that grows FASTER than the count (lse_j +5.8–6.3; the Chen et al. heavy-key-tail regime).
  (ii) attn vs attn_norp. At 128k a norp mass deficit exists in 7/7 cells (Δlogit(mass) +.03…+.80)
  and its evidence-side share Δlse_e / (Δlse_e + Δlse_j) is .78 / .97 / .66 (LL qa1/qa2/qa5), .93 /
  1.46 (Q3 qa1/qa2; >1 = norp's junk LSE falls too), .77 / .50 (Q7 qa1/qa2; the .50 sits on a
  negligible +.03 deficit) → prediction holds in 6/7, borderline 1/7; falsifier (|Δlse_j| ≥ |Δlse_e|)
  fires nowhere at 128k. At 64k the premise is ABSENT on Llama and Qwen-7B: norp has equal or MORE
  probe-layer mass than attn (LL qa1 .103 vs .093, LL qa2 .212 vs .216, Q7 qa1 .208 vs .208, Q7 qa2
  .364 vs .347) while its accuracy is 15 / 9 / 14 / 23pp lower — the 64k distance penalty is not
  visible in the read's attention mass at this layer at all (new, flagged; smax_e(norp) is even
  higher). On Qwen-3B the deficit is present from 32k on and is evidence-side with both sides falling
  (share 1.4–2.0: far scores are globally lower, evidence more). keff(norp) < keff(attn) in every
  cell (the fan-in contraction of P3).
- 2026-09-02 — **1M horizon test landed** (`outputs/babilong_1m/Qwen2.5-7B-Instruct-1M/`, job
  416645; L22; n=100; qa1+qa2 × 4k / 32k / 64k / 128k; tournament 32k/64; recall .96–1.00).
  Accuracy full / oracle / attn / attn_norp — qa1: 4k .91/.82/.90/.91, 32k .90/.88/.88/.83, 64k
  .91/.85/.87/.89, 128k .81/.88/.86/.83; qa2: 4k .54/.68/.55/.56, 32k .61/.63/.60/.56, 64k
  .48/.61/.55/.48, 128k .49/.67/.55/.47.
  (i) norp within 5pp of attn at 64k/128k: qa1 ✓ (+2 / −3), qa2 ✗ by 2–3pp (−7 / −8) — against the
  base Qwen-7B deficits of −14 / −36 (qa1) and −23 / −22 (qa2). lse_e(norp) − lse_e(attn) = −0.15 /
  −0.20 (qa1), −0.29 / −0.32 (qa2), all ≥ −0.5 ✓ (the same offset, −0.22…−0.53, is there at 4k and
  32k: it is not distance-specific). Falsifier (i) (norp deficit ≥ 15pp) does not fire →
  **distance axis = trained horizon, SUPPORTED.**
  (ii) full trails attn by ≥ 15pp on qa1 at 128k: ✗ — .81 vs .86 (5pp; qa2 .49 vs .55, 64k qa2 .48
  vs .55). mass(full,128k) / mass(full,4k) = .34 (qa1 ✓) / .61 (qa2 ✗). keff(full) 16.0 → 75.6 (qa1),
  16.9 → 66.7 (qa2) — identical to the base model (16.0 → 78.3). **Falsifier (ii) FIRES on accuracy:
  full ≈ attn at 128k (within noise, n=100).** But its stated reading ("long training repairs
  dispersion") is wrong in mechanism: dispersion is NOT repaired — fan-in and mass fall exactly as on
  the base model (mass .296 → .100 vs base .299 → .042). What the 1M training removed is the
  evidence-score erosion: Δlse_e(4k→128k) = −0.94 vs base −3.64, Δlse_j +1.05 vs +0.95, Δsmax_e
  −0.66 vs −3.19. So mass lands at .10 instead of .04, and the read still answers (base Qwen-7B
  answers at mass .14 / .136 [32k / 64k], fails at .042). Consequence for the framing: dispersion
  alone (keff ×4.7, mass ÷3) costs 5–7pp on qa1/qa2; the base model's 60pp collapse is dispersion
  × margin erosion. The degree axis is horizon-independent as a *mechanism*, not as an *accuracy
  cost*; the load-bearing quantity is the evidence mass crossing the model's threshold, and the
  count of competitors alone does not push it there when the margin holds.

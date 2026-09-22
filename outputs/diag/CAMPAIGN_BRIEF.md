# DIAG — aggregation diagnostics on the official benchmark: flip influence, attention photograph, temperature dissociation (2026-09-22)

**Status: plan APPROVED (Tal, 2026-09-22; `docs/DIAGNOSTICS_2026-09-22.md` = reference copy of
`~/.claude/plans/ok-so-i-want-sprightly-liskov.md`). Branch `theory-2`. Code being written; nothing run.**
Every GPU submission goes through the `sbatch-submit` loop (plan → OK → preflight → dry-run → submit →
verify); no cell is pre-authorized for submission. Diagnostic-only: **no training in DIAG** (Tier 1
consumes the rewrite's B3 adapter, it does not train one).
Scripts: `experiments/` (one entrypoint per instrument). Runs: `outputs/diag/<instrument>/…`.
Log: `STATE.md` (append-only, newest last). Index: `INDEX.md`. Wrappers: `sbatch/probe_hahn.sbatch`,
`sbatch/evaluate.sbatch`, `sbatch/probe_attention.sbatch` (to write).

## 0. One paragraph

Everything the record says about the aggregation bottleneck was measured on park data, the legacy
count prompt and the replica layout (`docs/DIAGNOSTICS_2026-09-22.md` §1). The chain: the per-frame
fact is N-invariant once frames are isolated (fenced slot α = 0.00); the model already ranks the
evidence (L24h20 AUC ≥ 0.993, ungated); the answer row cannot cash that in because its attention over
frames is a normalized mean, m = k·eˢ/(k·eˢ + (N−k) + C), so one frame's influence decays as N^−0.72
and the decision margin crosses zero where accuracy dies; LoRA multiplies the gain, not the law;
temperature and log-N move margins and free competitor mass but send it to the sink; deleting the
(N−k) term (the gate) makes the read N-invariant (α 0.035). DIAG re-measures that chain on the
OFFICIAL benchmark (paper prompt byte for byte, `<|vision_end|>` slot, no replica), frozen model
first, with three instruments (flip influence, attention photograph, accuracy/margin) organized by
**qtype class × intervention × N** — and adds the one prediction the record states but never tested:
**one temperature rescues a needle (k = 1) and cannot rescue a count (k equal units)** (D3, figure F3).
The official benchmark hands us that contrast for free: 15 NIAH types (k = 1) vs 9 DC types (k grows
with N).

## 1. Fixed inputs

### 1.1 Data — three regimes; every number carries its label

| label | what | rows | use |
|---|---|---|---|
| **official test** | HF `ef1e43ce/mmred` test split, Fr0do renders, native 512 px, paper prompt verbatim (`data/mmred_hf/json/seq_len_{8,16,32,64,128}_test.json`) | exactly **50 rows per qtype per N** | every headline cell |
| **official headfit** | `seq_len_{32,64,128}_headfit` — the official generator run with our seed (`data/mmred_hf/json/*_headfit.json`; `headfit_raw.json` = 1800 rows) | 600 rows per length = **25 per qtype** | pooled into thin strata only, always labelled "headfit (official generator, our seed)"; never alone in a headline row |
| **legacy park** | `data/mmred_*park*`, `mmred_redux` (generators frozen under `legacy/v1/datasets/`) | — | D5's re-fit of the legacy flip CSVs (ARMOR / FIXEDK / S3 / S11) and the F5 / F8 panels drawable today; "legacy park" in every caption |

Facts that shape the cells (verified 2026-09-22):
- `steps_in_room` gold at N=8 is **31/50 zeros**; at N=128 the **median gold is 18 and 60 % of golds
  exceed 16**. Consequences: (i) the flip response ‖Δh‖ uses **ALL golds** (k → k+1 is defined for
  every row with at least one non-evidence frame) — pairs ≤ 50 per cell, typically 40–50; (ii) the
  **digit-margin instrument is valid only where gold+1 ≤ 9** (digits 0–9 are single tokens) and is
  reported on that subset with its n — at N=128 that subset is a minority of the cell; (iii)
  **stratified-by-gold fits need ≥ 5 pairs per gold and will be thin at N ≥ 64** — reported with n per
  stratum, headfit rows pooled in (labelled) where they exist, strata with n < 5 shown but not fitted;
  (iv) the photograph's k ≥ 1 strata at N=8 hold at most 19 of the 50 rows.
- Count cells at N ≥ 32 report accuracy split by answer ≤ 16 / > 16 (CLAUDE.md §5).
- Room-name answers have **distinct first tokens**, so a first-token margin is well defined for
  `char_at_frame` (6 rooms); `n_char_at_frame` and the count types use the digit margin.
- Token budget: 324 image tokens per 512-px frame; **N=128 fenced/gated ≈ 41.8k tokens → h200
  (140 GB) or the batched-block path**. `experiments/evaluate.py` refuses the dense fenced path above
  24k tokens by default. Plain-arm N=128 runs on 40–48 GB.

### 1.2 Qtype cells (from the 24-type map in `core/mmred.py`; `--qtypes` on every script)

| class | qtype | why | frozen (legacy grid) |
|---|---|---|---|
| k=1 needle | `char_at_frame` | 6-room answer, no Nobody mass; slot-linear 0.98–1.00 flat | 0.82 → 0.32 |
| k=1 needle, numeric | `n_char_at_frame` | numeric k=1 with an N-flat answer range | — |
| k grows | `steps_in_room` | gold = \|evidence\|, k ≈ N/6; the workhorse | 0.56 → 0.00 |
| k grows | `crowd_count` | k ≈ 0.21 N | 0.28 → 0.00 |
| all frames evidence | `where_spend` | argmax over counters; slot-linear ≈ 1.00 flat; the gate is a no-op here (nothing to hide) — the control for "the gate is the counter" | 0.34 → 0.20 |

Not used as diagnostics: order-dependent first/last types, `rooms_visited` at N ≥ 64 (saturated),
Nobody-heavy person types. D4 alone runs all 24 at N=8/16 and 9 DC + 5 NIAH at N ≥ 32.
Lists ride in files: `sbatch/lib/splits/qtypes_{all,dc,niah,numeric,steps,anchor3}.txt` (never a
comma string in `--export`).

### 1.3 Arms

| arm | layout | loci |
|---|---|---|
| `plain` | paper prompt, images then question — the deployed model | answer row |
| `qfirst` | paper prompt with `--prefix_question` (the method's prefix, no fence) | answer row |
| `fenced` | qfirst + every frame in its own attention block + per-block position reset | answer row + the flipped frame's `<|vision_end|>` slot |
| `gated` | fenced + oracle gate (per-member evidence set: base hides block t, evid member exposes it) | answer row + slot |
| Tier 1 adds | trained read (B3 = P2-recipe fenced-SFT on the paper prompt) and the gated read on that adapter | as above |

Frozen for every Tier 0 cell. Layer convention: `hidden_states[L]` = output of decoder module L−1;
read layers L16/L20/L28, slot layers L12/L20/L24.

### 1.4 Instruments and run dirs

| instrument | entrypoint | knobs | run dir |
|---|---|---|---|
| flip influence | `experiments/probe_hahn.py` | arms plain/qfirst/fenced/gated; `--qtypes steps_in_room char_at_frame n_char_at_frame`; `--attn-sharpen τ --sharpen-from-layer L`; `--attn-logn-sref S` | `outputs/diag/hahn/<qtype>/N<N>[_tau<t>][_logn<sref>]/<stamp>_<jobid>/` |
| attention photograph | `experiments/probe_attention.py` | mass per block at the answer row, per head × layer; headscan AUC; same τ / log-N knobs; arms plain/qfirst/fenced/gated | `outputs/diag/photo/<qtype>/<arm>/N<N>[_tau<t>][_logn<sref>]/<stamp>_<jobid>/` |
| accuracy + margin | `experiments/evaluate.py` | + τ / log-N knobs | `outputs/diag/eval/<qtype>/<arm>/N<N>[_tau<t>][_logn<sref>]/<stamp>_<jobid>/` (mirrors `photo/`) |
| slot decodability (D4) | slot capture + CPU LR/AUC fit (`experiments/gate_capture.py` / `gate_fit.py` per CLAUDE.md §4; the entrypoint is confirmed in STATE when it lands) | arms × layers L12/L20/L24 | `outputs/diag/slot/<arm>/N<N>/<stamp>_<jobid>/` |
| CPU fits | `experiments/diag_fit.py` | `alpha`, `margins`, `sharelaw`, `headscan`, `gamma`, `mechform` | `outputs/diag/fits/` |
| figures | `experiments/figs/diag_fig.py` | F1–F10, from CSVs only | `outputs/diag/fig/` |

Flip definition per class: count types keep k → k+1 (move the asked character INTO the asked room in
one non-evidence frame; ctrl = move it to a third room, k unchanged). Needle types: edit the needle
frame so the answer changes (move the asked character to another room at step X); ctrl = the same edit
in a non-needle frame (answer preserved). Both members differ from base in exactly one frame, so the
byte-identity check carries over. Floors every cell: replay (must be 0), answer-preserving control,
block permutation. Renders md5-checked against the stored official frames (canary `n_same/n = n`).

### 1.5 Gotchas (memory-grade)

Official dirs sort K0-first — always stride; report class distribution + majority baseline per cell.
No comma in any `--export` value. Explicit `--time` on every GPU job (DefaultTime 2 h everywhere).
`--exclude=n317` on rtx6k. a100-public is 40 GB. 4-D mask ⇒ sdpa EFFICIENT/MATH only. PeftModel
wraps LAST (Tier 1). `plain` (frames-first) is the deployed baseline; `qfirst` numbers carry their
regime label. The paper prompt is byte-identical (`tests/test_prompt.py`). Every τ / log-N hook is
anchored τ-off vs τ=1 = 0.00e+00 on the same GPU before its cells are read.

## 2. Cells

Design principle: **qtype class × intervention × N**, three instruments, frozen model first.
N grid 8/16/32/64/128.

### Tier 0 — frozen model, no training (runnable once port gate C1 passes)

| id | experiment | design | reads | cost / partition |
|---|---|---|---|---|
| **D4** | fence × question 2×2 at the slot | arms {plain, qfirst, fenced-qlast, fenced-qfirst}; LR/AUC decodability of the per-frame fact at each frame's `<|vision_end|>` (L12/L20/L24) + the flip response at that slot; all 24 qtypes at N=8/16 (train split, fits only), 9 DC + 5 NIAH at N=32–128 (test) | "selection requires conditioning"; the gate-feasibility table the rewrite plan calls the first experiment | ≈ 6 GPU-h; `a100-public` `12h_4g` `--time 06:00:00` `--mem 48G`; fits on `4h_0g` |
| **D3** | temperature / log-N dissociation (the new cell) | frozen `qfirst`; qtypes {`char_at_frame` (k=1), `steps_in_room` (k ≥ 2)}; N ∈ {32, 128}; settings τ ∈ {1, 1.5, 2, 3, 4} on L ≥ 12 + log-N (sref = tokens at N=8); photograph AND accuracy/margin per setting = 24 + 24 cells; `where_spend` control at τ ∈ {1, τ*} × both N; the fenced (Tier 1: gated) accuracy ceiling at N=32/128 for both qtypes | can one τ rescue the needle and the count? where does the freed mass go? | ≈ 20 GPU-h (+ the where_spend control); `a100-public` / `l40s-shared` `12h_4g` `--mem 48G`; the N=128 fenced ceiling on `h200-shared` `24h_1g` |
| **D1** | flip-influence ladder, official | qtypes {`steps_in_room`, `char_at_frame`, `n_char_at_frame`} × arms {plain, qfirst, fenced} × N = 45 cells; ≤ 50 pairs/cell (all golds), 12 controls, seed 0; pooled α (L16/L20/L28) with bootstrap CI, per-gold α where n ≥ 5, margins on the valid subset, three floors | the task-blind weight law; fenced slot invariance; needle vs count margin survival | ≈ 40 GPU-h; one N per job (arms may share a job — `--arms` is a list); N ≤ 64 on `a100-public` / `l40s-shared` `12h_4g --time 04:00:00 --mem 48G`; N=128 on `h200-shared` `24h_1g` |
| **D2** | attention photograph, official, per class | same qtypes × {plain, qfirst, fenced} × N = 45 single-forward cells × 30–90 samples, k-strata; NLS fit of (s, C) per layer with bootstrap CI (`diag_fit.py sharelaw`); per-head AUC scan (`headscan`) | does the share law hold on official data at the read layer, and does it predict D1's α? | ≈ 25 GPU-h; same partitions as D1 |
| **D5** | mechanistic-form fit | fit every D1 flip curve (and the legacy ARMOR / FIXEDK / S3 / S11 CSVs, labelled legacy park) to Δ = G·[m(k+1, N) − m(k, N)] with (G, s, C) (`diag_fit.py mechform`); predict α from D2's (s, C) per layer | which parameter moves frozen → trained → gated; whether "α" is two fits with different C | CPU, `4h_0g` |
| **D6** | flip position + frame permutation | stratify D1 pairs by flip_t ∈ {first third, middle, last third} (free); permute frame order on order-invariant DC types (frozen, official): answer stability + ‖Δh‖ vs the perm floor | distance decay vs mean-read | free + ≈ 3 GPU-h (`2h_2g` / `12h_4g`) |

### Tier 1 — adds the method model (needs the rewrite's B3 adapter; not before)

| id | experiment | design | cost |
|---|---|---|---|
| D1b / D2b | D1 / D2 with the fenced-SFT adapter + oracle gate; arms {trained read, gated read} | trained α vs frozen α with gain; gated α; gated share N-invariance; α(B) slide on official data | ≈ 40 GPU-h |
| D3b | τ / log-N on the trained reader | as D3 | ≈ 15 GPU-h |
| D8 | headscan on official data, trained vs frozen, per class | selector population AUC; atrophy under gate training | ≈ 5 GPU-h |

### Optional / later

- **D7** unquantized bf16 control at N ∈ {8, 32} (80 GB for N=32 → `h200-shared`) to state the nf4 floor.
- **D9** model-gate ladder (LR on the slot, from D4) — method work, NOT in DIAG.

## 3. Pre-registered hypotheses and bands (fixed now; report every one, met or not)

Each band names the denominator theory's prediction, the rival, and what refutes what. Bands are
never re-tuned after seeing data; a miss is reported as measured and the theory clause it hits is
named in STATE.

- **H-D1 — flip influence, task-blind law.** On official test rows the `plain` AND `qfirst`
  answer-row α (L20, pooled over all golds, log-log fit of median ‖Δh‖ vs N ∈ {8..128}) lies in
  **[0.5, 1.0] for BOTH `steps_in_room` and `char_at_frame`** — the weight law does not know the task.
  The `fenced` slot α satisfies **|α| ≤ 0.05** (CI ∋ 0). Tier 1: the `gated` answer-row α has
  **CI ∋ 0**. The **replay floor is exactly 0** in every cell.
  Rival 1 (distance decay): α is a token-distance effect — refuted if D6's position strata agree
  (H-D6). Rival 2 (readout geometry): the slot is N-invariant AND the answer row is too (answer
  α ≈ 0 while accuracy still falls) — refuted by answer α ≥ 0.5.
  What refutes H-D1: `char_at_frame` α < 0.5 while `steps_in_room` is in band → the weight law is
  task-dependent (the needle escapes dilution in the frozen model): the "task-blind" clause falls,
  the count clause stands. Fenced slot |α| > 0.05 → the official layout's fence does not isolate the
  fact — stop the D1/D2 fenced and gated arms, diagnose the block mask. `n_char_at_frame` is reported
  alongside as the numeric-needle replicate; it does not gate the band.

- **H-D2 — the share law on official data.** For the frozen `qfirst` arm the per-layer NLS fit
  m = k·eˢ/(k·eˢ + (N−k) + C) reaches **R² ≥ 0.9 at the read layer (L20)**, and **α_pred from the
  fitted (s, C) via dm/dk lies inside D1's plain/qfirst α CI**. Descriptive (reported, not banded):
  frozen per-frame mass ∝ N^−(≈1) with evidence:non-evidence ratio ≈ 1.0; the k=1 needle mass obeys
  m = eˢ/(eˢ + (N−1) + C) with the same C; a head population with AUC ≥ 0.99 appears in the trained
  arm only (Tier 1), frozen best head ≈ 0.85.
  Rival: "0.72 vs 0.80 are two fits with different C" (D5 separates G, s, C). What refutes: R² < 0.9
  at every layer → the mean-field summary breaks on official data — report per-head heterogeneity,
  not a law; α_pred outside the D1 CI → the weight law and the flip response are two phenomena and
  plan §1.3-1 ("one denominator, three observables") is withdrawn.

- **H-D3 — one temperature rescues a needle and breaks a count.** Photograph: at N=32 AND N=128 the
  **needle (`char_at_frame`) evidence mass rises monotonically with τ** over {1, 1.5, 2, 3, 4} while
  the **count (`steps_in_room`) evidence mass falls** (competitor mass falls, the sink absorbs,
  max/mean over evidence rises). Accuracy: under the best τ* or under log-N the **needle accuracy
  rises by ≥ 0.15 toward the fenced/gated ceiling** (measured in this campaign at the same N) while
  **count accuracy does not rise** (Δ ≤ +0.05 at every τ and under log-N). Log-N: margins move,
  α does not, for both classes. `where_spend` (all frames evidence): no τ helps — nothing to select.
  Rival ("sharpening is enough": Veličković / SSMax / InfoScale / DySCO): both rise. What refutes
  H-D3: count accuracy rises by ≥ 0.15 too → the two-constraint claim is wrong on this backbone.
  Needle accuracy does not rise AND needle mass does not climb → temperature is not even a needle fix
  here: report the measured gap g and redo the ln N arithmetic from it (both gaps quoted — block-mass
  s ≈ 0.30 and per-frame ×1.04–1.10 — plan §1.3-3). The over-sharpening threshold (first τ with
  parse-fail > 0.05) is reported for both classes.

- **H-D4 — selection requires conditioning.** For the **9 DC types**, the per-frame fact at the
  `<|vision_end|>` slot is decodable at **LR/AUC ≥ 0.99 under `qfirst` and `fenced-qfirst`** and at
  **≤ 0.85 under question-last** (`plain`, `fenced-qlast`), at N=8/16 (train-split fits) and holding
  at N=32–128 (test). NIAH types are reported per qtype (positional types are expected near-flat
  under every arm — descriptive).
  Rival (Block-Attention counter-precedent): question-blind blocks recover after fine-tuning — out
  of scope for the frozen model; the claim is written "not in the frozen model". What refutes:
  fenced-qfirst < 0.99 on DC types → the fence design (question in the shared prefix) does not
  deliver the per-frame fact — STOP the D1/D2 fenced arms and report before any further fenced
  spend. qlast ≥ 0.95 → conditioning is not required and the prefix-question rationale weakens to
  "convenience".

- **H-D5 — mechanistic form.** Verification first: the fitter recovers (G, s, C) on synthetic curves
  before touching real CSVs. On real curves: frozen → trained moves **G** (and s slightly), **not C**;
  gated is the **C-only limit** (m = k/(k+C)); D2's per-layer (s, C) predicts D1's α within its CI
  (shared with H-D2). Tier 0 delivers the frozen fit + the legacy re-fit (labelled legacy park); the
  trained/gated columns wait for Tier 1.
  Rival: the frozen 0.72 and trained pooled 0.80 are two fits with different C. What refutes: C moves
  by more than its bootstrap CI frozen → trained while G is flat → fine-tuning changes the law, not the
  gain; the sentence "LoRA buys gain, not a new law" is withdrawn.

- **H-D6 — position and order.** The per-position α (flip_t in first third / middle / last third)
  is **within ±0.15 of the pooled α** for `plain` and `qfirst` at every stratum with ≥ 10 pairs.
  Descriptive: frame permutation on order-invariant DC types changes answers little and ‖Δh‖ under
  permutation sits at the perm floor (rates reported; no band).
  Rival (Brändel distance decay): last-third flips decay slower — **refutes H-D6 if the last-third α
  is smaller than the pooled α by > 0.2** with non-overlapping CIs. Then the N-decay has a distance
  component and F4's caption says so.

Tier 1 bands (fixed now, run later): **H-D1b** trained α within the frozen α CI with amplitude gain
≥ ×2; gated α CI ∋ 0. **H-D2b** gated evidence share N-invariant within ±10 % at fixed k. **H-D3b**
the D3 dissociation reproduces on the trained reader with usable count accuracy at τ = 1. **H-D8**
trained selector AUC ≥ 0.99 vs frozen ≈ 0.85 (legacy 0.848); atrophy under gate training reported.

## 4. Protocol and verification (plan §5)

1. **Port gate first.** C1 reproduces the frozen grid **0.533@8** (640/1200) and C2 reproduces P2
   0.900/0.660/0.300 before any DIAG cell (`outputs/port/INDEX.md`).
2. **Every DIAG cell prints:** replay floor = 0; render canary `n_same/n = n`; class distribution +
   majority baseline; pairs n and per-gold n; τ-off vs τ=1 = 0.00e+00 on the same GPU for every
   photograph/eval cell that carries the τ hook.
3. **Small N before N=128.** D1: fenced-slot |α| ≤ 0.05 and plain α ∈ [0.5, 1.0] at N=8..64 before
   any h200 hour is spent on N=128.
4. **D5's fitter** recovers (G, s, C) on synthetic curves before real CSVs (a test in `tests/`).
5. **Figures regenerate from CSVs by one script** (`experiments/figs/diag_fig.py`); no number typed
   into a figure.
6. **Labels.** Every number: arm (plain / qfirst / fenced / gated), prompt (paper images-first vs
   question-first vs fenced), data regime (official test / official headfit / legacy park), model
   (frozen / B3 adapter), gate (oracle / none). Never one table row unlabeled. Count cells at N ≥ 32:
   answer ≤ 16 / > 16 split.
7. **Train split** is touched only by D4's fits; no model training in DIAG.
8. Every cell: a `--limit 5` smoke on `2h_2g` into `outputs/_scratch/diag/` before the full cell.

## 5. Kill / stop criteria (stop-and-diagnose, never "tune and rerun")

- Port gate C1 fails → no DIAG submission until the port is fixed.
- A cell's replay floor ≠ 0, or the render canary `n_same/n < n`, or the τ anchor ≠ 0.00e+00 → the
  cell is INVALID (kept under `_scratch/`, never in INDEX); fix the instrument before the next
  submission.
- D4 fenced-qfirst DC decodability < 0.99 → stop the fenced/gated arms of D1/D2 (the layout is in
  question) and report to Tal.
- D1 at N ≤ 64: fenced slot |α| > 0.05 or plain α ∉ [0.5, 1.0] → stop before N=128; diagnose (mask,
  positions, pair construction) and re-run the small-N cell; the h200 hour is never spent on an
  unexplained band miss.
- N=128 fenced/gated OOM on 48 GB → do not retry on 48 GB; h200 or the batched-block path only.
- D5 fitter fails synthetic recovery → no real-CSV fit is reported.
- Anything that needs training to proceed → out of scope; goes to a method campaign, not DIAG.
- A band miss is a finding, not a kill: reported as measured, theory clause named.

## 6. Costs and partitions (no hour caps; discipline rules only)

| family | cells | GPU-h | partition / QOS / `--time` / `--mem` |
|---|---|---|---|
| D4 | 4 arms × (24 qtypes × N 8/16 + 14 qtypes × N 32–128) captures | ≈ 6 | `a100-public` `12h_4g` `--time 06:00:00` `--mem 48G`; fits `l40s-shared` `4h_0g` `--time 04:00:00` `--mem 16G` `--cpus-per-task 8` |
| D3 | 24 photo + 24 eval (+ where_spend control) | ≈ 20 | `a100-public` / `l40s-shared` `12h_4g` `--time 04:00:00` `--mem 48G`; N=128 fenced ceiling `h200-shared` `24h_1g` `--time 12:00:00` `--mem 64G` (host; confirm at dry-run) |
| D1 | 45 (one N per job) | ≈ 40 | N ≤ 64: `a100-public` / `l40s-shared` `12h_4g --time 04:00:00 --mem 48G`; N=128: `h200-shared` `24h_1g --time 12:00:00 --mem 64G` (confirm with the smoke) |
| D2 | 45 | ≈ 25 | as D1 |
| D6 | permutation cells | ≈ 3 | `2h_2g --time 02:00:00 --mem 48G` / `12h_4g` |
| D5, D6 strata, fits, figures | — | CPU | `l40s-shared` `4h_0g --time 04:00:00 --mem 16G` (cpus ≤ 8) |
| **Tier 0 total** | | **≈ 94 GPU-h** | |
| D1b/D2b, D3b, D8 (Tier 1) | | ≈ 40 + 15 + 5 | as above; after B3 |
| D7 (optional) | bf16, N 8/32 | — | `h200-shared` |

Rules: `--limit` smoke on `2h_2g` first; `--time` on every submission; no comma in `--export`
(qtype lists via `QTYPES_FILE`); `--exclude=n317` if rtx6k is used; long single-shot N=128 jobs
prefer a100/l40s/h200 over rtx6k (a NODE_FAIL loses a run whose summary is written at the end);
overflow to `24h_1g` / `4d_1g` when `12h_4g` slots are full; `unset DRY_RUN` before a real submit;
verify every job's command line with `verify_submit.sh` before walking away.

## 7. Order

port gate C1 (0.533@8 reproduced) → **D4** (cheapest; gates the fence design) → **D3** (the new
claim) → **D1 / D2** ladders (N ≤ 64 first, N=128 last) → **D5 / D6** on the CSVs → figures → STATE
verdict → **Tier 1** (D1b/D2b, D3b, D8) after B3 lands. F5 and F8 are drawable today from legacy CSVs
(CPU) and are labelled legacy until D3/D4 redraw them.

## 8. Deliverables

- `STATE.md`: every job id, run dir, class distribution, floors, band verdict; final section = the
  campaign verdict in the house style. `INDEX.md`: cell family → canonical run → headline.
- Figures in `outputs/diag/fig/` (design rules: N on a log₂ x-axis; one legend; ≤ 4 series; fixed
  colours — frozen grey, fenced/fact green, trained read orange, gated read blue; bf16 floor as a grey
  band; train window shaded; the exponent printed on the line; data regime in every caption; every
  title is the claim):
  F1 α ladder bars (frozen read · fenced fact · trained read · gated read) — D1 (+D1b);
  F2 evidence mass vs N with fitted curves, gated overlay — D2 (+D2b);
  **F3 "One temperature rescues a needle and breaks a count"** — τ vs (a) evidence mass, (b) accuracy;
  needle vs count; N=32 and 128 — D3 (the headline);
  F4 median ‖Δh‖ vs N, three lines + floor band + flip-position inset — D1, D6;
  F5 where the mass goes when you sharpen (stacked evidence / competitors / sink, τ = 1 vs 4) — D3
  (legacy CSV drawable today);
  F6 fitted (G, s, C) table + overlay — D5;
  F7 head AUC vs accuracy on one x — D2 / D8;
  F8 per-qtype heatmap, slot-linear decodability vs frozen accuracy, 24 rows — drawable today (legacy
  CSVs), redrawn from D4;
  F9 ‖Δh‖ vs k with the γ fit — D1b k-chain (later);
  F10 margin vs N — D1.
- RESULTS.md: appended only on Tal's "log this"; one entry per cell family, house style (headline
  claim → run dirs → table → readings → caveats), newest last, nothing that isn't backed by a run dir.
- Honesty flags in every entry: cell sizes and per-gold n; valid-margin subset n; official test vs
  headfit vs legacy park; frozen vs adapter; oracle gate stated wherever it is on; single seed, one
  backbone, nf4 (D7 states the floor if run); γ / the k-wall is "measured, not predicted"; MultiMax
  (2406.01189) cited for the two-constraint statement, never presented as new.

## 9. Discipline

CLAUDE.md §5/§7 in full: read before write; plan before GPU hours (partition / QOS / `--time` / cost
named, Tal OKs); right-size (`--limit` smokes first); one change at a time; anchors are law
(`tests/` after any `core/` touch); prompt fidelity byte for byte; no installs; `legacy/` read-only;
never re-tune a band after seeing data; never a metric in RESULTS.md without a run dir; submit→poll
loops in tmux; the `sbatch-submit` skill for every submission.

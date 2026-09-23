# DIAGNOSTICS — what the record says about the failure, the DIAG campaign, and the figure set (2026-09-22)

> Reference copy of the plan approved by Tal on 2026-09-22 (`~/.claude/plans/ok-so-i-want-sprightly-liskov.md`).
> Published page: see the "One Denominator" artifact. Nothing here is a logged result; RESULTS.md stays the record.


## 0. TL;DR

**What we know (park data, legacy prompt, replica layout — none of it yet on the official benchmark):**

| stage | measured fact | number | source |
|---|---|---|---|
| perception | the per-frame fact is flat in N once frames are isolated | fenced slot α = 0.00 ± 0.02; fixed-count slope 0.02–0.05; LR probe ≥ 0.999/frame to N=128; HF leaf recall 0.995–1.000 | [2026-08-22] ARMOR-A; FIXEDK; S2; [2026-08-10a] |
| selection | the model already *ranks* evidence frames | L24h20 AUC 0.9997/0.9927/0.9999/1.0000 at N=8/16/32/64 (P1b, ungated); frozen best head 0.848 | [2026-09-01] S10 |
| the read | one frame's influence on the answer decays as a power of N | frozen α = 0.72 [0.63, 0.78], r² 0.98, at the bf16 floor by N=128; margins slide −0.23 → −1.91 | ARMOR-A |
| the read | attention obeys the share law m = k·eˢ/(k·eˢ + (N−k) + C) | L20: s = 0.30, C = 8, R² 0.97 (P1b); frozen per-frame mass ∝ N^−0.94 with evidence:non-evidence ratio ≈ 1.0 (no preference at all); P1b ×1.07 constant edge | S10, S10b **(per-frame ladders unlogged)** |
| the read | the law predicts the exponent | dm/dk at (s=0.30, C=8, k=4) gives α_pred = 0.72; s=0 gives 0.78 (recomputed today) | memory share-law-predicts-alpha **(unlogged)** |
| fine-tuning | LoRA multiplies amplitude, not the law | ×3–6 gain; pooled α 0.80 [0.66, 0.97]; fixed-count 0.44–0.62 | LORAMECH N2; FIXEDK |
| temperature | sharpening cannot do it, and the photograph says why | no τ lifts k≤8@64 past 0.24; competitor mass 0.74→0.29 at N=32 but evidence mass 0.053→0.012 and sink 0.15→0.41; at N=128 τ≥3 evidence/frame < non-evidence/frame | S0; S10b τ sweep **(photograph unlogged)** |
| log-N | scaling moves margins, not the geometry | margin −2.89→−0.10 @64; α 0.605 vs 0.603; eval-only +0.13 @64 fenced, NULL plain; as training prior 0.867→0.420 | REDUX C4b; LORAMECH N3/P3 |
| knockout | deleting the (N−k) term makes the read N-invariant | gated α 0.035 [−0.043, 0.123] with byte-identical text; margins +11.7 constant; gated share N-invariant ≤ 6%; α(B) slides 0.66/0.17/0.10/0.07/0.03 for B = 2/8/32/128/∞ against the law's 0.48/0.15/0.03/0.01/0 | S11; S10; SOFTGATE |
| walls | after the knockout the decay moves to k, plus prompt-N calibration | γ = 1.19 [1.16, 1.22] gated (adapter-independent) vs 0.39 ungated; snap relocates with a declared-N lie | S3/S11; C4b; S7 |
| retrieval vs count | the record has ONE controlled family (REDUX) and it says: frozen single-needle retrieval also decays (exists k=1 recall 13/25 → 0/16 at N=128); a trained reader keeps exists = 1.000 at every N ≤ 128 while count dies | REDUX C1; C2 mt **(N=64/128 cells unlogged)** |

**The open prediction Tal named** ("temperature fixes a needle but cannot fix k equal units") is stated
in the record and precedented (MultiMax, Chiang & Cholak, Yang & Chiang) but **never measured here**:
no τ or log-N cell was ever run on a k=1 task. That is the cleanest new diagnostic and the
official benchmark hands it to us for free: 15 NIAH qtypes (k=1) vs 9 DC qtypes (k grows with N).

**The plan:** a diagnostic-only campaign (DIAG) on the official benchmark, frozen model first, three
instruments (flip influence, attention photograph, margin/accuracy), organized by qtype class ×
intervention × N, plus a CPU-only mechanistic-form fit. Eight clean figures, four drawable today.

---

## 1. What we know — the mechanistic account, diagnostics only

### 1.1 The chain (one sentence per stage, then the evidence)

1. **Frames are read correctly one at a time.** Under the fence the per-frame fact is N-invariant and
   linearly decodable; on the official benchmark the per-frame linear readout is flat for most qtypes.
2. **The model knows which frames matter.** A trained-in head population (L24h20 and ~10 others)
   separates evidence from non-evidence at AUC ≥ 0.993 at every N, in the *ungated* model.
3. **The answer row cannot cash that in.** Its attention over frames is a normalized mean: evidence
   share m = k·eˢ/(k·eˢ + (N−k) + C). The per-frame edge is tiny (×1.35 as block mass, ×1.04–1.10
   per frame, ≈1.0 frozen), so the (N−k) competitors take the mass, one frame's influence on the
   answer decays as N^−0.72, and the decision margin crosses zero where accuracy dies.
4. **Fine-tuning buys gain, not a new law.** LoRA multiplies the flip response ×3–6 and adds a
   constant ×1.07 edge; the exponent is unchanged within CI.
5. **Sharpening is the wrong family.** One temperature must (i) beat N−k competitors (τ·g ≈ ln N;
   with g = 0.30 that is τ ≈ 12 at N=32, 16 at N=128) and (ii) keep the k evidence frames equal
   (τ·δ ≈ 0). Both hold only if δ = 0, i.e. k = 1. Measured: raising τ frees competitor mass but
   sends it to the prompt/sink, not the evidence; evidence concentration max/mean rises 1.04 → 1.35;
   accuracy collapses. Log-N scaling is the same family: margins move, α does not.
6. **Deleting the (N−k) term is the right family.** With the gate the read's α is 0.035 with
   byte-identical text, margins are constant, the gated share is N-invariant, and a leaky gate
   (weight 1/B on hidden blocks) slides α along the law's curve. The gated forward is the
   evidence-only forward (20/20 identical decodes).
7. **Two walls remain.** (a) Calibration to the prompt-declared N (causal, prompt-lie); cured by an
   N-free prompt (the official prompt never states N). (b) Resolution in k: γ = 1.19, adapter-
   independent, exact only within trained answer coverage (c* ≈ 16 = coverage).

### 1.2 Evidence table (diagnostics only; regime = park unless stated)

| # | quantity | condition | locus | value | run dir / source | status |
|---|---|---|---|---|---|---|
| E1 | α, frozen joint read | paper-style plain prompt (legacy count prompt) | L20 answer row | **0.72 [0.63, 0.78]**, r² 0.98; L28 0.58 [0.49, 0.68]; L16 at floor | `outputs/armor/hahn/20260822_211457_N*` | logged [2026-08-22] |
| E2 | α, fenced per-frame fact | fence + posreset + replica | rep_t L16/L20/L28 | +0.01 / −0.01 / +0.02, CIs ∋ 0 | same | logged |
| E3 | floors | replay / answer-preserving ctrl / block-perm | L20 | 0 / 1.1 @128 / 1.2–1.4 (fenced rep_t bit-identical under perm) | same | logged |
| E4 | margin slide, frozen | joint | digit logits | −0.53 @8 (saturated), −0.23 @16 → −1.91 @128; acc 0.28 → 0.10 (chance 1/9) | same | logged; crossing band failed-as-defined |
| E5 | Jacobian cross-frame share | joint vs fenced | L≤16 | joint own-frame 0.480 (52% cross-frame); fenced 1.000, cross exactly 0.00 | `outputs/presentation/sensitivity/20260731_211330` | logged [2026-07-31] |
| E6 | q/kv patching | 2×2, N=8..128 | L16 | q-only +0.92…+1.32 flat; kv-only +0.97 → −0.78; interaction +1.0 → +2.2 | `outputs/presentation/qkv_swap/*`, `outputs/armor/qkv128` | logged |
| E7 | share law fit | P1b, k∈{2,4,8}, N=8..64 | L20 (per layer 12–27) | s = 0.30, C = 8, R² 0.969 (R² 0.93–0.99 per layer; 9/9 sign checks) | `outputs/sparse/s10/p1b_N*` | logged [2026-09-01]; **fit code does not exist in repo** (constants are literals in `mk_figs.py`) |
| E8 | per-frame mass ladder | frozen / P1b / gated, k=4 | L20 | frozen .049→.0038 (slope 0.94, ratio ≈1.0); P1b .073→.0074 (0.83, ratio 1.07); gated .016 flat, non-evidence 0 | `outputs/sparse/s10/{frozen,p1b,gated}_N*`, `fig/F10_dispersion_hahn.csv` | **unlogged** (STATE only) |
| E9 | α predicted from (s, C) | CPU | — | 0.74 / 0.72 / 0.67 at k = 2/4/8; s=0 → 0.78; C=0 → 0.88–0.93 | memory | **unlogged** |
| E10 | LoRA gain vs exponent | ff_le32 / P1b | L20 answer row | ×3–6 amplitude; α 0.71 [0.36, 0.93] (plain) / 0.80 [0.66, 0.97] (fenced read) | `outputs/loramech/{l1,n2}_hahn` | logged [2026-08-27→28] |
| E11 | fixed-count α | frozen / trained / gated, flip 0→1, N=2..128 | L20 | 0.88 / 0.44 / −0.00 (1→2: 0.76 / 0.62 / −0.03) | `outputs/fixedk/*`, `fig/fixedk_medians.csv` | logged (rewrite RESULTS) |
| E12 | τ sweep, accuracy | P1b, τ on L≥12, N=32/64/128 | decode | τ1 .85/.34/.23 → τ4 .19/.12/.06; best k≤8@64 = 0.24 | `outputs/sparse/s0_tau*` | logged [2026-08-30→31] |
| E13 | τ sweep, photograph | P1b, k=2 | L20 head-median | N=32: competitor .738→.292, evidence .053→.012, sink .148→.413, max/mean 1.04→1.35; N=128: .893→.720 / .0146→.0043 / conc 1.03→1.37; evidence/frame < non-evidence/frame at τ≥3 | `outputs/sparse/s10/p1b_N{32,128}_tau*`, `fig/F11_temperature_coupling.csv` | **unlogged** |
| E14 | log-N, α | P1b + sref 3398 | L20 read | 0.605 [0.55, 0.69] on vs 0.603 [0.53, 0.68] off; margins −2.89 → −0.10 @64 | `outputs/redux/c4b_n/*` | logged (STATE) |
| E15 | log-N, accuracy | eval-only / training prior | decode | fenced 0.340→0.470 @64, plain NULL; prior 1.000/1.000/0.420/0.310/0.170 | `outputs/loramech/n3_logn_*`, `p3_*` | logged |
| E16 | gated α, byte-identical text | S9 + oracle gate | L20 read | **0.035 [−0.043, 0.123]**; margins +11.7 constant; flips change answer 0.9–1.0 | `outputs/sparse/s11/*` | logged [2026-09-01] |
| E17 | gated share | S8 + gate | L20 | N-invariant ≤ 6.2% at fixed k; s′ = 0.26, C′ = 46 | `outputs/sparse/s10/gated_N*` | logged |
| E18 | α(B) | S9b + soft gate | L20 | 0.66 / 0.17 / 0.10 / 0.07 / 0.03 (B = 2/8/32/128/∞); law 0.48/0.15/0.03/0.01/0; B=1 = floor (content-blind reader) | `outputs/softgate/a1_*`, `a1_alpha_L20final.json` | logged (rewrite RESULTS) |
| E19 | gate without retraining | P1b + gate, N=32 | decode | acc 0.080, signed error +20.8 (answers ≈ N) | `outputs/sparse/s1_control` | logged |
| E20 | γ in k | gated (P1g, S9) / ungated (P1b) | L20, N=64 | 1.19 [1.16, 1.22] / 1.19 [1.17, 1.22] / **0.39 [0.375, 0.402]** | `outputs/sparse/s3/…k*`, `s11/…k*`, `outputs/redux/c4b_k` | logged (S3/S11); C4b STATE only |
| E21 | prompt-N snap | P1g + gate, declared-N lies | decode | k12/k16: 0 → 1.00 under "16"; k8: 1.00 → 0 under "64"; k4 120/120 | `outputs/sparse/s7/*` | logged |
| E22 | equivalence twin | P1g | L16/20/28 | 20/20 identical decodes; rel L2 0.010 | `outputs/_scratch/sparse_w3/equiv` | logged |
| E23 | selector head | P1b ungated / frozen / S9b | L24h20 mass AUC | 0.9997–1.0000 flat / 0.848 / atrophied 0.52–0.75 | `outputs/sparse/s10/*`, `outputs/selfgate/g0a/*` | logged |
| E24 | selector mass ratio | P1b | L24h20 | evid:nonevid 1.6→3.2 vs N/k = 2→32 needed; frozen ≈ 1.0 | `fig/F10_dispersion_hahn.csv` | **unlogged** |
| E25 | exists / count / majority, frozen | REDUX C1, plain | decode | exists 0.833→0.490 (k=1 recall 13/25→0/16); majority ≈ chance ∀N; count 0.20→0.08 | `outputs/redux/c1_plain_*` | logged (STATE) |
| E26 | exists / count / majority, trained multitask | REDUX C2 mt, N-free, ungated / gated | decode | exists **1.000 at N=8…128 both**; majority ungated flat at matched ratio 0.96→0.875, gated 0.37 @32; count dies | `outputs/redux/c2{a,b}_*_mt*`, `*_eval64_128` | N ≤ 32 STATE; **N=64/128 unlogged** (jobs 154212/154213) |

### 1.3 Theory insights (what the numbers add up to)

1. **One denominator, three observables.** The share law explains the weights (E7/E8), the answer's
   flip response (E1 via dm/dk, E9), and the gate ladder (E18: the leaky gate puts (N−k)/B back in
   the denominator and α follows the law within 0.07 for B ≥ 8). Recomputed today:
   α_pred(B) = 0.716 / 0.484 / 0.154 / 0.033 / 0.007 / 0 for B = 1/2/8/32/128/∞ at (s=0.30, C=8, k=4).
2. **The two-constraint statement** (precedented by MultiMax 2406.01189 — must cite, never present as
   new): τ·g ≳ ln N to remove competitors and τ·δ ≈ 0 to keep the k evidence equal; compatible only
   when δ = 0, which is k = 1. This is why the literature's temperature fixes (Veličković, Chiang &
   Cholak, SSMax, DySCO, InfoScale) are all needle/retrieval results, and why our count read breaks
   under every τ. Our additions: the length axis (the required gap grows like ln N), the measured
   gap on a production VLM, and where the freed mass goes (the sink, E13) — no paper reports that.
3. **Number hazard to state, not hide:** the record has two gaps — s = 0.30 from the block-mass fit
   (×1.35) and ×1.04–1.10 from the direct per-frame photograph (ln = 0.07). The τ ≈ 16 arithmetic
   uses g = 0.30; with the per-frame gap it would be τ ≈ 70. Quote both instruments.
4. **The k-wall is NOT derivable from the photograph.** Today's check: the gated fit (s′=0.26, C′=46)
   gives a dm/dk slope in (k+2) of only ≈ 0.5, not the measured γ = 1.19; Yehudai gives 1/k (weight)
   or 1/k² (gap). γ sits between and is unexplained — say "measured, not predicted", never "confirms".
5. **Retrieval is not free either.** Frozen exists collapses to chance (E25), so "NIAH is solved by
   the frozen model" is false on our data; what is true is that a *trained* reader keeps k=1 exact
   at every N (E26) while count dies. The frozen official grid shows the softer version: NIAH types
   fall 0.90→0.60 (first_app) / 0.82→0.32 (char_at_frame) while steps_in_room falls 0.56→0.00.
6. **Two separable short-N failures.** The N=8 failure of the frozen model is a decoder/conditioning
   failure cured by LoRA (0.22 → 1.000); the long-N failure is dilution and is not cured. Never write
   "even short videos".

### 1.4 Confounds the record itself names (each is closed by a cell in §2)

| confound | why it matters | closed by |
|---|---|---|
| everything is park data + legacy count prompt + replica layout | the paper's numbers must come from the official benchmark under the paper prompt; the fence/gate layout changed (question in shared prefix, `<|vision_end|>` slot, no replica) | D1–D4 re-measure on official data |
| fence × question-copy 2×2 never run at the same locus | "each frame understandable" may be a property of the replica, not the fence | D4 |
| (s, C) fit on the P1b adapter, α = 0.72 on the frozen model | the law–exponent agreement is a consistency check, not a fit | D2 fits frozen and trained separately; D5 fits the flip curves to the form |
| finite-range effective exponent of a rational form (5 points) | "α" may be two fits with different C | D5 reports (gain, s, C) instead of α, plus R² |
| flipped-frame position vs distance decay (Brändel) | N-decay could be token-distance decay | D6 stratifies by flip position; frame-permutation on order-invariant DC types |
| pooled-k vs fixed-k α (0.80 vs 0.44–0.62) | protocol changes the number | D1 reports both, stratified by base k |
| no τ/log-N cell on a k=1 task | the two-constraint claim is untested here | D3 |
| γ is a coverage cliff or a resolution law | contested (2605.03258) | out of scope for DIAG; note only |
| single seed, one backbone, nf4 | scope | state scope; D7 optional bf16 control |

---

## 2. The diagnostic campaign: DIAG (official MMReD, deployed model)

Design principle: **qtype class × intervention × N**, three instruments, frozen model first.

- **Qtype cells** (from the 24-type map; official data, test split, 50/qtype/N):
  - k=1 needle: `char_at_frame` (6-room answer, no Nobody mass; frozen 0.82→0.32; slot-linear 0.98–1.00 flat); `n_char_at_frame` (numeric k=1, N-flat answer range).
  - k grows: `steps_in_room` (gold = |evidence|, k ≈ N/6; frozen 0.56→0.00; the workhorse), `crowd_count` (k ≈ 0.21N; frozen 0.28→0.00).
  - argmax-over-counters, all frames evidence: `where_spend` (slot-linear ≈1.00 flat vs frozen 0.34→0.20) — the sharpest supply/readout dissociation; gate is a no-op here (nothing to hide), so it is the control for "the gate is the counter".
  - Avoid as diagnostics: order-dependent first/last types, `rooms_visited` at N ≥ 64 (saturated), Nobody-heavy person types.
- **Arms:** `plain` = paper prompt, images-then-question (the deployed model); `qfirst` = paper prompt with `--prefix_question` (the method's prefix, no fence); `fenced` = qfirst + fence + posreset (loci: answer row + flipped frame's `<|vision_end|>` slot); `gated` = fenced + oracle gate (per-member evidence set). Frozen for all; P2-recipe fenced-SFT adapter as a second pass once the rewrite's B3 gate exists.
- **N grid:** 8/16/32/64/128; 50 pairs (flip) or 30 samples per k-stratum (photograph); floors replay/ctrl/perm every cell; renders md5-checked against the stored official frames.
- **Flip definition per class:** count types keep the k→k+1 protocol (move the asked character into the asked room in one non-evidence frame; ctrl = move to a third room). Needle types: edit the needle frame so the answer changes (move the asked character to another room at step X); ctrl = same edit in a non-needle frame (answer preserved). Both members differ from base in exactly one frame, so the byte-identity check carries over.

### Tier 0 — frozen model only, no training (runnable once the port's C1/C2 gates pass)

| id | experiment | design | prediction (denominator theory) | rival prediction | cost | headline plot |
|---|---|---|---|---|---|---|
| **D1** | flip-influence ladder, official | qtypes {steps_in_room, char_at_frame, n_char_at_frame} × arms {plain, qfirst, fenced} × N; `experiments/probe_hahn.py` (in stash; extend to needle flips + `--qtypes`) | plain/qfirst answer-row α ≈ 0.7 for both classes (the weight law is task-blind); fenced slot α ≈ 0; needle margins survive longer than count margins | distance decay: α depends on flip position (D6 test); readout geometry: slot α ≈ 0 but answer α ≈ 0 too | 3 qtypes × 3 arms × 5 N = 45 cells; N ≤ 64 on 48 GB, N=128 needs h200 or the batched-block path; ≈ 40 GPU-h | F1 ladder bars + F4 lines |
| **D2** | attention photograph, official, per class | same qtypes × {plain, qfirst, fenced} × N, k-strata; new `experiments/probe_attention.py` (port of `probe_attn_photo.py` with `layout_blocks`, no replica, `<|vision_end|>` blocks) + **new NLS fit of (s, C) per layer with bootstrap CI** (no fit code exists today) + per-head AUC scan | frozen per-frame mass ∝ N^−(≈1), evidence ratio ≈ 1.0; k=1 needle mass obeys m = eˢ/(eˢ + (N−1) + C) with the same C; head population AUC ≥ 0.99 for the trained arm only | — | 3 × 3 × 5 = 45 single-forward cells × 30–90 samples; ≈ 25 GPU-h | F2 photograph; F7 dissociation |
| **D3** | **the temperature / log-N dissociation** (the new cell) | frozen `qfirst` arm; qtypes {char_at_frame (k=1), steps_in_room (k≥2)}; N ∈ {32, 128}; τ ∈ {1, 1.5, 2, 3, 4} on L ≥ 12 and log-N (sref = tokens at N=8) — both photograph and accuracy/margin; also `where_spend` as the "all frames are evidence" control | k=1: evidence (needle) mass rises monotonically with τ, accuracy/margin recover toward the fenced ceiling; k≥2: competitor mass falls but evidence mass falls faster, sink absorbs, max/mean rises, count breaks; log-N: margins move, α does not, for both | "sharpening is enough" (Veličković/SSMax/InfoScale): count recovers too | 2 qtypes × 2 N × 6 settings = 24 photograph cells + 24 accuracy cells; ≈ 20 GPU-h | **F3** (the new headline) + F5 |
| **D4** | fence × question 2×2 at the slot, per qtype | arms {plain, qfirst, fenced-qlast, fenced-qfirst}; LR/AUC decodability of the fact at each frame's `<|vision_end|>` (L12/L20/L24) and the flip response at that slot; all 24 qtypes at N=8/16 (train split), 9 DC + 5 NIAH at N=32–128 (test) | fact decodable only when the question is co-located (qfirst or fenced-qfirst ≥ 0.99; qlast ≈ 0.75) — "selection requires conditioning"; this is also the gate-feasibility table the rewrite plan calls the first experiment | Block-Attention counter-precedent: question-blind blocks recover after fine-tuning (say "not in the frozen model") | 1 forward per sample; ≈ 6 GPU-h + CPU fits | F8 per-qtype heatmap |
| **D5** | mechanistic-form fit (CPU) | fit every flip curve from D1 (and the legacy ARMOR/FIXEDK/S3/S11 CSVs) to Δ = G·[m(k+1,N) − m(k,N)] with (G, s, C); report which parameter moves frozen → trained → gated; predict α from D2's (s, C) per layer and compare | trained moves G (and s slightly), not C; gated is the C-only limit | two fits with different C explain 0.72 vs 0.80 | CPU only | F6 (parameter table) |
| **D6** | position of the flipped frame + frame permutation | stratify D1 pairs by flip_t ∈ {first third, middle, last third}; permute frame order on order-invariant DC types (frozen, official) and measure answer stability + ‖Δh‖ | α independent of position; permutation changes answers little for DC types (the read is a mean) | Brändel distance decay: last-third flips decay slower | free (re-analysis of D1) + ≈ 3 GPU-h | inset on F4 |

### Tier 1 — adds the method model (needs the rewrite's B3 adapter = P2 recipe on the paper prompt)

| id | experiment | design | prediction | cost | plot |
|---|---|---|---|---|---|
| D1b/D2b | repeat D1/D2 with the fenced-SFT adapter and the oracle gate | adds arms {trained read, gated read} | trained α ≈ frozen α with ×3–6 gain; gated α ≈ 0; gated share N-invariant; α(B) slide on official data | ≈ 40 GPU-h | F1 (all four bars), F2b, F6 |
| D3b | τ / log-N on the trained reader | as D3 | same dissociation with usable count accuracy | ≈ 15 GPU-h | F3 (second row) |
| D8 | headscan on official data | trained vs frozen, per class | selector population AUC ≥ 0.99 trained, ≈ 0.85 frozen; atrophy under gate-training | ≈ 5 GPU-h | F7 |

### Optional / later

- **D7** unquantized bf16 control at N ∈ {8, 32} (needs 80 GB for N=32) to state the nf4 floor.
- **D9** model-gate (LR on the slot, from D4) ladder — this crosses into method work; not in DIAG.

### Dependencies and order

1. Tal reviews/commits the stashed port; `pip install pydantic`; run B1 → C1 (frozen grid port check) → C2 (P2 port check) per `docs/REWRITE_EXECUTION_2026-09-21.md`.
2. Extend `experiments/probe_hahn.py` (needle flips, `--qtypes`, τ and log-N flags carried over from `probe_hahn_gated.py`); write `experiments/probe_attention.py` (+ the (s, C) fitter and headscan) and `experiments/figs/diag_*.py`; wrappers `sbatch/probe_attention.sbatch`; `outputs/diag/{CAMPAIGN_BRIEF,STATE,INDEX}.md` per the campaign convention.
3. D4 first (cheapest, gates the fence design), then D3 (the new claim), then D1/D2 ladders, then D5/D6 on the CSVs. Tier 1 when B3 lands.
4. Everything via the `sbatch-submit` skill; N=128 at 512 px (≈ 41k tokens) goes to h200 or waits for the batched-block path; `--exclude=n317`; explicit `--time`.

### Protocol notes

- Official data only; train split touched only by D4's fits (no model training in DIAG).
- Report count cells split by answer ≤ 16 / > 16 where relevant (steps_in_room N=128 has 60% > 16).
- Every flip cell prints its three floors; every photograph cell anchors τ-off vs τ=1 on the same GPU.
- Label every number by arm and prompt (paper images-first vs question-first vs fenced); never one table row unlabeled.

---

## 3. Plots — the clean set

Design rules: N on a log₂ x-axis; one legend; ≤ 4 series; colors fixed across the set (frozen grey,
fenced/fact green, trained read orange, gated read blue); bf16 floor as a grey band; train window
shaded; the exponent printed on the line; official vs legacy data always in the caption; every title
is the claim.

| # | claim-title | x | y | series | exists today? | produced by |
|---|---|---|---|---|---|---|
| F1 | "One frame's influence on the answer: the read dilutes, the fact does not, the gate flattens" | rung (4 bars) | α with 95% CI | frozen read 0.72 · fenced fact 0.00 · trained read 0.80 · gated read 0.035 | legacy: `outputs/sparse/fig/F3_alpha.png` (2 of 4), `F10_dispersion_hahn.png` legend | D1 (+D1b) |
| F2 | "Attention obeys the share law; the gate removes N from it" | N | evidence mass at L20 | measured k=2/4/8 + fitted curves; gated flat overlay | `outputs/sparse/fig/F7_attn_photo.png` (two panels; merge) | D2 (+D2b) |
| **F3** | "One temperature rescues a needle and breaks a count" | τ | (a) mass on evidence, (b) accuracy | k=1 needle vs k≥2 count, N=32 and 128 | no | D3 |
| F4 | "Single-frame influence vs N, three lines" | N | median ‖Δh‖ (log) | frozen joint · fenced fact · gated; floor band; inset: flip position | `outputs/armor/hahn/…_fig/hahn_figure.png` panel a (7 lines → 3) | D1, D6 |
| F5 | "Where the mass goes when you sharpen" | τ = 1 vs 4 | stacked mass: evidence / competitors / sink | N=32, N=128 | `fig/F11_temperature_coupling.png` (3 panels, overlapping legend → 1 stacked-bar panel) | D3 (legacy CSV drawable today) |
| F6 | "Fine-tuning moves the gain, the gate moves C" | rung | fitted (G, s, C) table + overlay of fitted vs measured curves | frozen / trained / gated | no (CSVs exist for the legacy fit) | D5 |
| F7 | "The model ranks the evidence and still cannot count it" | N | left: head AUC; right: accuracy | trained-in head vs answer accuracy, same x | `F10` panel c (partial) | D2/D8 |
| F8 | "Per-frame readout is flat, the answer collapses — across the whole benchmark" | N (columns) | 24 qtypes (rows), two heatmaps: slot-linear decodability vs frozen accuracy, grouped NIAH / DC | **yes**: `outputs/mmred_hf/armB_grid_v2_perlen/armB_grid_linear_ALL.csv` + `outputs/mmred_hf/frozen/grid_seq*_test/*/report.txt` | drawable today (CPU) |
| F9 | "After the knockout the decay is in k" | k | median ‖Δh‖ (log) | gated read with γ fit | `F3_alpha.png` panel b | D1b k-chain (later) |
| F10 | "The decision margin crosses zero where accuracy dies" | N | logit margin | frozen · trained · gated | `F9_margin_3variants.png` | D1 |

Existing figures to retire or simplify: F10 (four panels, three legends → split into F1/F4/F7);
F11 (legend overlaps the axis → F5); F5_flatline (mixes N-prompt and N-free anchors → one prompt per
figure); F8_sens_3variants (three trained variants nobody needs → F4).

---

## 4. Implementation after approval (what gets written, where)

1. `docs/DIAGNOSTICS_2026-09-22.md` — §1 of this plan as the reference document (tables + the
   confidence ledger + the unlogged list), committed.
2. `outputs/diag/CAMPAIGN_BRIEF.md` (cells D1–D8 with pre-registered bands), `STATE.md`, `INDEX.md`.
3. Code on `rewrite` after the port lands: `experiments/probe_hahn.py` extensions,
   `experiments/probe_attention.py` (photograph + (s, C) NLS fit + headscan), `experiments/figs/diag_fig.py`
   (F1–F10 from CSVs), `sbatch/probe_attention.sbatch`; tests for `layout_blocks`/slot positions and
   for the fitter on synthetic data.
4. Draw F8 and F5 today from existing CSVs (CPU, `4h_0g`), so the story has two official/legacy
   panels before any GPU hour.
5. Unlogged numbers to append to RESULTS.md on Tal's "log this": S10b per-frame ladders and τ
   photograph (E8, E13, E24), REDUX N=64/128 exists/majority (E26), the share-law α prediction (E9),
   FIXEDK stratified flips.

## 5. Verification

- Port gates C1/C2 reproduce 0.533@8 frozen grid and P2 0.900/0.660/0.300 before any DIAG cell.
- Every DIAG cell: replay floor = 0; render canary n_same/n = n; τ-off vs τ=1 = 0.00e+00 on the same GPU.
- D1 fenced-slot α within |0.05| and plain α within [0.5, 1.0] at N=8..64 before N=128 is spent.
- D5's fitter recovers (G, s, C) on synthetic curves before touching real CSVs.
- Figures regenerate from CSVs by one script; no number typed into a figure.

## 6. Open decisions for Tal

1. Branch for the DIAG code: `rewrite` (assumed) or `theory`.
2. Whether Tier 1 waits for the B3 adapter or DIAG ships as frozen-only first (recommended: frozen-only first; it is the "deployed model" question).
3. Whether the qtype set above (3 + 1 control) is enough for the paper or the full 24 go into D4 only.

---

## 7. Results — Tier 0, run 2026-09-22 (frozen Qwen2.5-VL-7B nf4, official MMReD test rows, 50 per type per N)

Branch `theory-2`; run dirs and job ids in `outputs/diag/INDEX.md`, the dated log in `outputs/diag/STATE.md`,
fits in `outputs/diag/fits/`, figures in `outputs/diag/fig/`. Every number below is on the official
benchmark under the paper's prompt (images then question = "deployed"; their `--prefix_question` order =
"question-first"), native 512 px, upstream parser. Nothing here is on park data. RESULTS.md untouched.

### 7.1 Port gates and the faithful frozen baseline

| gate | result |
|---|---|
| B1 prepare_data verify (seq_len_8_val, 50 qids) | JSON bytes equal; gold parity 1200/1200; 400/400 frames pixel-identical |
| C1 port check (legacy 392 px preset) | 637/1200 = 0.531 vs anchor 640/1200 = 0.533; per-type identical to the legacy grid |
| Faithful frozen grid, 24 types | **0.515 / 0.415 / 0.348 / 0.285 / 0.220** at N = 8/16/32/64/128 (CI ±0.03); steps_in_room 0.56 → 0.42 → 0.14 → 0.04 → 0.00; char_at_frame 0.86 → 0.64 → 0.54 → 0.30 → 0.24; first_app 0.80 → 0.80 → 0.80 → 0.58 → 0.50 |

### 7.2 D4 — the per-frame slot carries the fact only when the question precedes the frames (H-D4 MET)

Logistic readout of each frame's `<|vision_end|>` state, held out by sample, all 24 types.

| arm (layout × fence) | N = 8+16 train, L12 / L20 / L24 (acc / AUC) | trained N ≤ 16 → tested N = 32 (14 types, 22,400 frames), L20 |
|---|---|---|
| deployed (images then question), no fence | 0.48 / 0.51 / 0.49 (chance) | — |
| question-first, no fence | 0.76 / 0.84 / 0.84 (AUC 0.88 / 0.93 / 0.93) | — |
| fenced, question after the frames (question-blind blocks) | 0.50 / 0.49 / 0.48 (chance) | 0.49 / AUC 0.51 |
| **fenced, question-first** | 0.89 / **0.975 / 0.998** / 0.969 | **0.982 / 0.9987** (recall 0.988, specificity 0.975) |

Per type at L20 under fenced question-first: every needle and count type ≥ 0.96 (char_at_frame 1.00,
steps_in_room 1.00, who_spend 1.00, crowd_count 0.99, first/last_at_room 1.00); the "when B first/last
appeared" family 0.82–0.95 (its predicate is partly global, the readout's job). Question-blind blocks read
only question-independent visual facts (crowd present AUC 0.96–0.97, first/last frame 1.00). Transfer to
N = 32 / 64 / 128 (gate fitted at N ≤ 16): pooled 0.982 / 0.980 / 0.9725 (22,400 / 44,800 / 89,600 frames); per
type at N = 128: char_at_frame 0.986, steps_in_room 0.976, first_at_room 0.986, crowd_count 0.970. The
1.8–2.7 %/frame error still compounds at exact match ((0.98)^64 ≈ 0.27).

### 7.3 D2 — the attention photograph: a constant per-frame edge, 1/N dilution, a constant sink share

Layer 20, answer row (last token of the teacher-forced `{ "answer": "` prefix), head-mean then sample-mean.

| arm | prompt+sink share (N = 8…128) | needle edge e^s = evid/frame ÷ non-evid/frame | count (steps_in_room) edge | evidence mass, needle, N = 8…128 |
|---|---|---|---|---|
| deployed | 0.63 / 0.60 / 0.60 / 0.60 / 0.57 | 2.2 / 1.8 / 2.2 / 2.1 / 1.6 | 1.4 / 1.4 / 1.2 / — / — | 0.083 / 0.040 / 0.024 / 0.013 / 0.006 |
| question-first | 0.43 / 0.42 / 0.42 / 0.41 / 0.39 | 2.4 / 2.6 / 2.9 / 2.6 / 3.1 | 1.1 / 1.2 / 1.1 / — / — | 0.140 / 0.083 / 0.046 / 0.025 / 0.013 |
| fenced (frozen) | 0.25 / 0.18 / 0.11 / 0.08 / — | 1.15 / 1.17 / 1.24 / 1.08 / 1.03 | 1.06 / 0.98 / 0.99 / 0.98 / — | 0.106 / 0.060 / 0.034 / 0.017 / 0.009 |
| gated (evidence only) | 0.52 / 0.52 / 0.53 / 0.53 / — | — | — | **0.477 / 0.475 / 0.474 / 0.473** (flat) |

What this says. (i) The sink takes a constant fraction of the answer row's mass per layout, not a constant
number of frame-equivalents; the two-parameter fit m = k·eˢ/(k·eˢ + (N−k) + C) is therefore not identifiable
on this prompt (it returns s = −0.34 for a needle whose measured edge is ×2). The identifiable form is
(1 − sink) × k·eˢ/(k·eˢ + (N−k)) with the edge read directly from the photograph. (ii) The edge is constant
in N and is set by the task: ×2 for a needle, ×1.1–1.4 for a count-evidence frame, ≈1 under the frozen
fence. (iii) Deleting the (N−k) term makes the mass N-invariant with nothing trained. (iv) A per-head
median shows the deployed needle edge below 1 at N ≥ 64: a minority of heads carries it. (v) No frozen
head separates evidence at AUC ≥ 0.98 at any N (best 0.86 → 0.78); the selector head of the record is
trained-in.

### 7.4 D3 — sharpening raises the needle's edge, the sink takes the mass, both classes lose accuracy (H-D3 accuracy clause REFUTED)

N = 32, question-first, τ multiplies the logits of decoder modules ≥ 12 at eval time; log-N: scaling =
ln(seq)/ln(3000). Layer 20; EM = exact match with greedy decoding and the upstream parser.

| condition | needle char_at_frame: evidence mass · competitor mass · sink · edge · EM (margin) | count steps_in_room: edge · EM | where_spend EM |
|---|---|---|---|
| τ = 1 | 0.046 · 0.535 · 0.419 · ×2.67 · **0.60** (+0.30) | ×1.10 · **0.22** | 0.24 |
| τ = 1.5 | 0.042 · 0.416 · 0.542 · ×3.17 · 0.54 (+0.29) | ×1.11 · 0.18 | 0.16 |
| τ = 2 | 0.036 · 0.319 · 0.645 · ×3.50 · 0.44 (−0.20) | ×1.05 · 0.16 | 0.14 |
| τ = 3 | 0.030 · 0.265 · 0.705 · ×3.53 · 0.34 (−0.77) | ×1.05 · 0.02 | 0.10 |
| τ = 4 | 0.026 · 0.281 · 0.694 · ×2.86 · 0.22 (−1.70) | ×1.10 · 0.12 | 0.04 |
| log-N (sref 3000) | 0.044 · 0.513 · 0.443 · ×2.68 · 0.54 (+0.02) | ×1.14 · 0.18 | 0.26 |

The k = 1 half of the two-constraint argument holds in the weights (the needle's relative edge rises, the
count's stays flat) but the freed competitor mass goes to the prompt/sink, the needle's absolute mass
falls monotonically, and exact match falls for both classes. A global eval-time temperature does not
rescue even a needle on the frozen read; log-N scaling is a null at this N.

N = 128 (same protocol): needle EM 0.24 / 0.26 / 0.24 / 0.14 / 0.10 for τ = 1 / 1.5 / 2 / 3 / 4 with the sink
share 0.39 → 0.50 → 0.60 → 0.71 → 0.73 and the needle's mass 0.0145 → 0.0050 (edge ×3.1 → ×3.6 → ×2.4);
count EM 0.00 under every condition; log-N: needle 0.22, count 0.00, where_spend 0.30 (vs 0.18 at τ = 1;
±0.13 on 50 rows). The picture is the same at 4× and 16× the training length.

### 7.5 D1 — single-frame influence at the answer, N = 8…128 (H-D1: slot clause MET; deployed needle α below the [0.5, 1.0] band)

Median ‖Δh‖ at the layer-20 answer row over 50 pairs per cell; floors: replay exactly 0 in every cell,
permutation 0.6–0.7, answer-preserving control edit elsewhere 0.7–5.8 (deployed), 0 under the gate.

| edit | deployed read | question-first read | frozen fence, answer row | gated read (evidence blocks only) | the frame's own slot |
|---|---|---|---|---|---|
| needle char_at_frame | 19.6 / 14.1 / 10.8 / 7.6 / 5.3 — α 0.46 [0.36, 0.59] | 14.3 / 10.7 / 5.0 / 2.5 / 2.0 — α 0.79 [0.67, 0.89] | 4.7 / 2.9 / 1.7 / 1.3 / 1.5 (floor) | **27.9 / 28.0 / 27.0 / 28.8 / 26.6 — α 0.01 [−0.02, 0.03]** | 8.9 / 9.9 / 9.2 / — / 9.4 — α −0.02 |
| needle n_char_at_frame | 11.3 / 10.9 / 7.8 / 4.9 / 3.6 — α 0.44 | 14.4 / 11.8 / 8.2 / 3.2 / 1.7 — α 0.80 | at floor | 18.8 / 16.5 / 19.5 / 17.1 / 16.7 — α 0.03 | flat |
| count steps_in_room, fixed count 0 → 1 | α 0.75 [0.40, 1.44] | α 0.89 [0.70, 1.22] | 0.74 | **0.03** | flat (0.05) |
| count, pooled (k grows with N) | 16.8 / 7.8 / 4.1 / 1.9 / 1.3 — 0.94 | 8.8 / 4.6 / 1.8 / 0.9 / 0.8 — 0.93 | — | 38.5 / 12.8 / 5.1 / 3.0 / 2.1 — 1.05 (k-composition) | — |

(α ranges above are from the final fit over the N values present per cell; the N = 8–32 fits give the
same picture: deployed 0.43 [0.20, 0.61], question-first 0.77 [0.48, 0.96], gated 0.02 [−0.03, 0.09].)
Flip position (terciles, ~17 pairs each): no monotone distance effect for the needle (0.57 / 0.22 / 0.34).
Reading: removing the (N−k) term makes the read's response N-invariant over 16× with nothing trained,
for needle and (at fixed count) count edits alike; the deployed needle read decays with a sub-unit
exponent, shallower than the needle's own attention share (≈0.9 from the frame law), because the answer
row's response is not proportional to the share alone and the control edit already moves it by 0.7–5.8.
The frozen fence without a trained read is at the floor: isolation supplies the fact to the slot (flat,
decodable at 0.97) but does not route it to the answer.

### 7.6 Band verdicts and what changed

| band | verdict |
|---|---|
| H-D1 slot |α| ≤ 0.05 | MET (−0.02 / 0.05 / 0.05, CIs ∋ 0) |
| H-D1 plain/qfirst α ∈ [0.5, 1.0] both classes | question-first MET (0.79 / 0.80 / 0.89); deployed needle 0.46 BELOW (the sink holds 0.60 of the mass at every N); count fixed-count 0.75 MET |
| H-D1 gated α CI ∋ 0 (Tier 1 clause) | already MET on the frozen model |
| H-D2 frame-law R² ≥ 0.9, α_pred within D1 CI | the 2-parameter share fit is not identifiable on this prompt; the frame-only law with a constant sink fits the mass ladders; its predicted share exponent (≈0.9) exceeds the measured answer-row α (0.46) — not within CI |
| H-D3 needle accuracy +0.15 under τ | REFUTED (0.60 → 0.22); weight-level dissociation MET (needle edge ×2.7 → ×3.5, count flat) |
| H-D4 fenced-qfirst ≥ 0.99, qlast ≤ 0.85 | MET (0.975 pooled; needles/counts 1.00; qlast 0.49) |
| H-D6 position within ±0.15 | descriptive only (17 pairs per tercile) |

Caveats: one backbone, nf4, frozen only (Tier 1 = the trained read + the trained gate is next), 50 rows
per cell, τ applied globally at eval time (a per-head or trained temperature is a different experiment),
the wave-1 photograph chains did not store per-block CSVs (no within-evidence concentration at N ≤ 32).

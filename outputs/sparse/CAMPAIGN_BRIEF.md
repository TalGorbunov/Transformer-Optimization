# SPARSE campaign — the k-regime: a gated read makes counting decay in k, not N

**Status: AUTHORIZED (Tal, 2026-08-30) — all cells pre-approved, no stop-for-OK checkpoints.**
Scripts: `scripts/sparse/`. Runs: `outputs/sparse/<cell>/`. Log: `STATE.md` (append-only,
newest last). Index: `INDEX.md`. Sbatch: `slurm/sparse_*.sbatch`. Sister campaign: REDUX
(`outputs/redux/AGENT_PROMPT.md`, the reduction-type law; REDUX C4b measures α under logN
scaling — reuse its chains if they exist, never duplicate).

## 0. One paragraph

LORAMECH showed the fine-tuned count read carries k as a softmax SHARE
`m = k·eˢ / (k·eˢ + (N−k) + C)` — N-dependent (lawful undercount out-of-window), saturating,
1/N-resolved (read α = 0.80 vs verdict-slot α = 0.01). The `(N−k)` term is the N-dependence.
If the read attends ONLY to evidence blocks (+ the sink), the code becomes `k/(k+C)`: N never
enters, sensitivity depends on k alone, α_N = 0 by construction, and the only remaining wall is
resolution in k (capacity c*). Since realistic counts are small even at N=128 (our dense exam
band is k=1..8; video counts are ≤10), this is the regime that matters: exact counting at ANY
length for k ≤ c*, in ONE forward, with a plain LoRA and a mask — no scratchpad. The gate is
the per-frame verdict, which the fence already delivers at α = 0 with d′ ≈ 9–13. Eval-time
log-N scaling could not reach this regime because at our token counts (S_ref = 3398) the
factor ln S / ln S_ref is only 1.17 @4× and 1.26 @8× — far below the +1.4/+2.1 nats the
`(N−k)` term needs; N3/P3 were half-sharpened reads, which reopens P3's verdict.

## 1. Fixed inputs (all exist)

- Backbone: Qwen2.5-VL-7B, 4-bit, bf16, sdpa (`gnnformer.runtime.load_runtime`, 7B default).
- Trainer to extend: `scripts/loramech/train_sft_fenced.py` (layout `[frame_i, q]×N + count
  prompt`, `build_block_mask(hide_cols=[])`, posreset, cache-free greedy decode, r8 α32
  q/k/v/o+MLP, lr 2e-4, accum 8, answer-token loss, `--exclude-dirs-file`, `--eval-dirs-file`,
  `--eval-only-adapter`, `--attn-logn-sref` hook that sets `module.scaling` on the LM attention
  modules). P1b = `checkpoints/sft_fenced_le16_ep10_adapter/` (1.000/0.893/0.867/0.340/0.230
  @8/16/32/64/128; +eval-logN 0.950/0.470/0.230 @32/64/128, `outputs/loramech/n3_logn_fenced/`).
- Mask primitive (READ-ONLY, `gnnformer/fencing.py`): `build_block_mask(seq, blocks, hide_cols)`
  hides `hide_cols` from every row outside their own block, incl. the final prompt/decode tail.
  **The gate = `hide_cols` := union of the spans of all NON-evidence blocks.** No new mask code.
- Evidence labels: `gnnformer.data.parse_task_labels` / `probe_evidence` (per-frame evidence
  set from states; assert |set| == gold, skip+count mismatches).
- Probe: `scripts/armor/probe_hahn.py` (arms plain/fenced/p1fence, `--peft-adapter`,
  `--max-gold`); the p1fence arm imports the trained-fenced layout from the P1 trainer.
- Gate instrument: the supply-probe / gate→tally protocol (`scripts/probe_supply.py`,
  `scripts/gate_tally.py`: logistic gate on cached per-frame messages, 0.998 external).
- Pools: `data/mmred_images_park/seq_len_8/all_uniform`, `data/mmred_longN_park{,2}/seq_len_
  {16,32,64,128}/all_uniform`; exam files `outputs/loramech/examdirs/exam_ff_N{8..128}.txt`
  (150/cell, gold {0..8,12,16,24,32,48,64,96,128} anchors, contamination-proof; REPORT.txt).
  MMReD-HF loader/splits: `gnnformer/mmred_hf.py` (N4 protocol in `outputs/loramech/n4_park_on_hf/`).
- Gotchas (memory-grade): dirs sort K0-first — ALWAYS stride; report class distribution +
  majority baseline per cell; no comma-lists in `sbatch --export`; explicit `--time`; ALL
  partitions default 2 h walltime; check every partition before submitting; 4-D mask ⇒ sdpa
  EFFICIENT/MATH only (N≤16 trains on 48 GB); PeftModel wraps LAST (capture structural refs
  pre-wrap); frames-first only; the count prompt is byte-identical (`build_count_prompt`).

## 2. Cells

| cell | what | reads | GPU |
|---|---|---|---|
| **S0 sharpness sweep** (eval-only, P1b) | replace the log-ratio with a fixed factor τ ∈ {1.0 ref, 1.5, 2, 3, 4} on the LM attention `scaling`, restricted to layers ≥ L*=12 (flag `--attn-sharpen τ --sharpen-from-layer 12`; τ=2 all-layers as one control); count exams N ∈ {32, 64, 128}, 100/cell; report per gold band (0 / 1–4 / 5–8 / anchors), parse-fail, mean signed error | is the k-regime reachable by temperature at eval? where does over-sharpening break the LM? | ~1.5 h |
| **S1 P1g — oracle-gated fenced SFT** | trainer flag `--gate oracle`: `hide_cols` = non-evidence block spans, in TRAINING and EVAL forwards (all layers). Same recipe/budget as P1b: N ≤ 16 (seq8 + seq16 pools), class-balanced, 10 ep, `--exclude-dirs-file` = all exam files, val-by-decode. Eval with the oracle gate at N ∈ {8,16,32,64,128} on the exam files (in-support gold ≤ 16 is the headline slice; out-of-support anchors reported too). Adapter → `checkpoints/sft_fenced_gated_adapter/` (+ README row; eval contract: gate required). Control: **P1b + oracle gate at N=32** without retraining (prediction: overcounts — its thresholds were fit to the mean-ish code) | the flat-in-N line for k ≤ 16; the calibration argument | ~3.5 h train + ~2 h eval, 48 GB |
| **S2 model-gated read** (deployable) | (a) two-forward: pass 1 = fenced forward (P1g adapter, no gate), capture replica-slot states at L20 (FenceHooks capture or `output_hidden_states`), logistic gate (train on the S1 training split's cached messages with gold evidence labels — CPU; report gate acc/d′ per N on exam dirs), build `hide_cols`, pass 2 = gated decode. (b) optional one-forward: `--gate-from-layer 12` variant (fence everywhere, non-evidence hidden from the tail only at layers ≥ 12) trained as a second adapter only if (a) lands and time allows. Eval N ∈ {8,…,128} | oracle→model gap = gate recall; the deployable number | CPU + ~2 h eval |
| **S3 α under the gate** | `probe_hahn.py --arms p1fence --peft-adapter <P1g> --gate oracle` (the evid member's hidden set excludes block t, the base member's includes it — that is the point): count flips k→k+1, gold ≤ 8, N ∈ {8,…,128}, 50 pairs/N (40 @128), controls 12; plus a vs-k chain at N=64, base k ∈ {1,2,4,8,16,32} (`--gold-set`), ~30 pairs/k. Reuse REDUX C4b's no-gate chains if present. Report read α_N (L16/20/28) with bootstrap CI, verdict-slot α, margin vs N, and `Δ ∝ (k+C)^−γ` fit | α_N ≈ 0 exactly under the gate; decay-in-k measured | ~2.5 h |
| **S4 capacity in k** | P1g + oracle gate at N=128 (and N=64), gold ∈ {0,1,2,4,8,12,16,24,32,48,64,96,128} from the exam files (≥ 9/cell; top up from pools to ≥ 20/cell where possible, strided); accuracy and signed error vs k | the trained c* curve: where exactness dies in k with N fixed; overlay the frozen `c(fan)` curve (superquery rr: 0.98/0.90/0.65/0.44/0.21/0.12 at fan 2/4/8/16/32/64) | ~1 h |
| **S5 transfer** | model-gated P1g on the MMReD-HF test splits seq8/16/32 (N4 protocol, 50/len); report vs P1b's 1.000/0.820/0.540 | does the gated read transfer? | ~1 h |
| **S6 (conditional)** | only if S0 finds a τ that reaches the k-regime: retrain P1b's recipe WITH that τ (P3-redux) and run the P3 ladder; the P3 verdict ("compensation as a training prior fails") was measured with a 17% compensation and is reopened by this cell only | — | ~3.5 h |

Order: S0 ∥ S1 (S1 is the long pole — submit first) → S3 + S4 + S1-control as soon as the
P1g adapter lands → S2 (gate training on the S1 split, then evals) → S5 → S6 if triggered →
figures → RESULTS.md entries.

## 3. Pre-registered hypotheses and bands (fixed now; report every one, met or not)

- **H-S1 (headline).** P1g + oracle gate: accuracy ≥ 0.97 for every k ≤ 16 stratum at EVERY
  N ∈ {8,…,128}, and the k ≤ 8 band drops by ≤ 0.05 from N=16 to N=128 (flat in N). Refuted
  if the k ≤ 8 band at N=128 is < 0.85. Any N-dependence that survives the gate is a
  positional effect (posreset immunity claimed) or a sink-mass effect — diagnose with S3.
- **H-S1-control.** P1b + oracle gate at N=32 shows a systematic OVERcount on k ≤ 8 (mean
  signed error ≥ +0.5) — the thresholds were fit to the competitor-diluted code.
- **H-S3.** Under the gate the read α_N over {8,…,128} has CI ∋ 0 and |α| < 0.15 (vs 0.80
  [0.66,0.97] ungated), the verdict slot stays ≈ 0, the margin does not slide with N; at fixed
  N=64 sensitivity decays in k with γ ≥ 1 (CI excludes 0.5).
- **H-S4.** Exactness vs k at N=128 is ≥ 0.97 through k=16 and falls beyond; the k where it
  crosses 0.5 is reported as c*_trained (expected 24–48); the frozen curve crosses 0.5 at ~8.
- **H-S2.** Gate accuracy ≥ 0.99 per frame at every N (supply is flat); model-gated exact
  match ≥ 0.90 at N=128 for k ≤ 8; oracle − model gap explained by the measured gate errors
  (report per-sample: was every error a gate miss?).
- **H-S0 (two-sided).** If some τ gives the k ≤ 8 band ≥ 0.80 @64 (P1b+logN: 0.07/0.50 for
  1–4/5–8) with parse-fail ≤ 0.02 and N=32 within −0.03 → the k-regime is reachable at eval
  (trigger S6). If no τ does → the mean→sink transition needs training (S1 is the route).
  Over-sharpening threshold (first τ with parse-fail > 0.05 or N=32 drop > 0.05) reported.
- **H-S5.** Model-gated P1g ≥ P1b on HF seq16 and seq32 (0.820 / 0.540) by ≥ 0.10.

## 4. Deliverables

- `STATE.md`: every job id, run dir, class distribution, band verdict; final section = the
  campaign verdict in the house style. `INDEX.md`: cell → canonical run → headline.
- Figures in `outputs/sparse/fig/`: (F1) accuracy vs N on the k ≤ 8 band — frozen / P1b /
  P1b+logN / P1g-oracle / P1g-model-gated (the flat-line figure); (F2) accuracy vs k at N=128,
  P1g-oracle with the frozen c(fan) curve overlaid (the capacity figure); (F3) α panel — read
  α with gate vs without, sensitivity vs k; (F4) the S0 sweep. Presentation-grade, tables
  alongside (Tal's standard).
- `checkpoints/README.md` rows for every promoted adapter (with eval contracts).
- RESULTS.md: append the final entries at campaign close (pre-authorized 2026-08-30) — one
  per cell family, house style (headline claim → run dirs → table → readings → caveats),
  newest last, nothing that isn't backed by a run dir.
- Honesty flags: cell sizes; oracle vs model gate stated in every number; single task
  family/model; the gate's dependence on the fence (the supply must be flat for the gate to
  be N-invariant — cite ARMOR-A); k-strata with < 10 samples marked.

## 5. Discipline

CLAUDE.md §5/§7 in full: smokes (limit 5, 1 ep) in `outputs/_scratch/` before every full
cell; explicit `--time`; right-sized QOS (`12h_4g`/`24h_1g` for trainers, `2h_2g` for evals,
`4h_0g` for CPU); no installs; `gnnformer/` and `legacy/` untouched (all deltas in
`scripts/sparse/`; the no-flag behavior of every extended script must reproduce its anchors —
verify one P1b exam cell and one N2 median before the first full run); one change at a time;
never re-tune a band after seeing data; run submit→poll loops in tmux.

---

# WAVE 3 (authorized by Tal, 2026-08-31) — kill the snap: S7 prompt-lie probe + S8 virtual-N training

Same permissions and discipline as waves 1–2 (all cells pre-approved, no stop-for-OK;
anchors before full runs; append to STATE.md; RESULTS.md entries at close).

## Context (from the wave-1/2 verdict)

The gate removed the read's N-decay (S3: α ≈ 0) and k ≤ 4 is exact at every N, but mid-k
snaps to answering exactly "N", with the boundary CONTRACTING with N (≈8–12 @32, 6 @64,
5 @128), while the resolution wall (S3 γ ≈ 1.2, flip signal near floor by k ≈ 12–16) sits
higher. Diagnosis to test: under the gate the read sees k evidence blocks + prompt either
way, so the only N-dependent channels left are (i) the PROMPT TEXT ("You will be shown {N}
frames… integer from 0 to {N}") and (ii) any residual position/length effect in the tail.
If (i) drives the snap, it is calibration, and full (k, declared-N) training coverage
removes it — which the fence makes CHEAP: a gated N-frame forward is computationally
equivalent to "k evidence frames + a prompt declaring N" (hidden blocks contribute exactly
zero to visible rows; per-block posreset makes block positions N-independent — the perm
canary's bit-identity). That equivalence must be PROVEN before it is used (S8 smoke).

## S7 — prompt-lie probe (eval-only, ~1 h, decisive)

`train_sft_gated.py` eval path + `--declare-n Ñ`: overrides ONLY the num_frames used in
`build_count_prompt` text; frames, mask, gate, positions untouched. P1g adapter
(`checkpoints/sft_fenced_gated_adapter`), oracle gate. Cells (dirs from the exam/S4 files,
≥ 20/stratum where available, strided):
- true N=32, declared Ñ ∈ {16, 32 ref, 64}, strata k ∈ {4, 8, 12, 16}
- true N=64, declared Ñ ∈ {16, 32, 64 ref}, strata k ∈ {4, 6, 8, 12}
Report per stratum: acc, prediction histogram (is the snap target the DECLARED Ñ?), mse.
**H-S7 (two-sided, both clauses required for "calibration confirmed"):**
(a) true-32 declared-16: k12 recovers to ≥ 0.8 (from 0.00) — and snapped answers relocate
    to "16" where they occur;
(b) true-32 declared-64: k8 drops to ≤ 0.5 (from 1.00).
Both met → the snap is prompt-N calibration (text channel) → S8's mechanism is confirmed.
Neither moves → the text channel is excluded; the residual is position/length-side —
S8 still runs (its mixture also randomizes the true-N/declared-N pairing) but its
interpretation shifts; report honestly. k4 must stay 1.00 under every lie (side condition:
any k4 drop > 0.05 = the lie itself harms the read — report and take it into account).

## S8 — virtual-N training (the fix, same budget as S1)

**Step 0 — equivalence smoke (REQUIRED GATE for the cell):** for ~10 real seq16/seq32
samples, build (A) the true gated forward and (B) the synthetic twin: only the evidence
frames as blocks + prompt declaring the ORIGINAL N, positions constructed to match A's
visible rows (check `reset_positions` for the tail offset; add a tail-offset knob if the
tail's position ids depend on N). Assert: answer-position hidden states equal within bf16
tolerance AND greedy decodes identical, A vs B, at N=16 and N=32. If the smoke fails,
STOP S8, log the divergence channel (that is itself the S7-(ii) answer), and fall back to
declared-N randomization on REAL gated samples only (k ≤ 16 × true N ≤ 16 — weaker
coverage, still runs).
**Trainer delta:** `--virtual-n` mixture mode: 50% real gated samples (= S1) + 50%
synthetic (k ∈ 0..16 stratified × declared Ñ ∈ {8, 16, 32, 64, 128}, k ≤ Ñ; k=0 cells at
every Ñ included — the read sees only the prompt and must answer 0). Everything else =
the S1-r3 recipe verbatim (fixed trainer — mask held through backward, lr 2e-4, 10 ep,
seq8+16 pools, exclude all exam files + S4 top-ups). Known coverage hole, stated up
front: k = Ñ ("all frames") is only trainable for Ñ ≤ 16 — the k≈N overflow regime at
large N stays out of scope (wave-2 open item (c)).
**Evals:** oracle-gated full ladder (exam N ∈ {8..128} @150 + S4 dirs N64/128); then the
S2 pipeline for the model-gated ladder (re-capture replica states under the NEW adapter,
re-fit the LR gate on the train split, per-N gate accounting as in wave 2). Adapter →
`checkpoints/sft_fenced_gated_vn_adapter` (+README row, eval contract: gate required).
**Bands (fixed now):**
- **H-S8-main:** oracle-gated k ≤ 8 band ≥ 0.95 at EVERY N ∈ {8,…,128} (wave-2 values:
  1.00/0.95/1.00/0.73/0.63) — the snap eliminated inside the resolution range. Refuted
  if k ≤ 8 @128 < 0.85 with the equivalence smoke passing (then the snap was not text
  calibration — revisit S7's (ii) channel).
- **H-S8-noharm:** N=8 ≥ 0.99 and N=16 ≥ 0.94 (no in-window cost for the coverage).
- **H-S8-cap (descriptive):** k12/k16 strata at N ≥ 32 reported as the measured
  calibrated-capacity edge; ≥ 0.8 at k12 would beat the resolution estimate — report,
  don't require. k ≥ 16 expected ~0 (capacity, not an S8 failure).
- **H-S8-model:** model-gated == oracle within per-sample gate-error accounting.
**Figure:** F5 = the wave-2 F1 flat-line redrawn with the S8 row (frozen / P1b / P1g /
P1g-vn on the k ≤ 8 band vs N) — the headline if H-S8-main lands.

Order: S7 immediately (eval-only) ∥ S8 step-0 smoke; S8 training on smoke pass; evals;
captures+gate; figures; STATE verdict + RESULTS.md entries (wave-3 addendum). If S7 and
S8 disagree (S7 null but S8 flat), say so loudly — that is a finding, not an
embarrassment.

## S9 — N-free prompt (authorized by Tal, 2026-08-31; run alongside/after S8)

**Motivation + protocol fact (verified in `data/mmred_hf/upstream_repo/`):** the ORIGINAL
MMReD benchmark never states N — its count question is bare ("How many steps did {char}
spend in the {room}?") and its system prompt (scripts/openai_server_inference.py:68,
mirrored verbatim in `scripts/mmred_hf/eval_frozen.py`) contains no frame count and no
answer range. Our `build_count_prompt`'s "You will be shown {N} frames … integer from 0 to
{N}" is a legacy-local wrapper — the prompt-N channel the snap latched onto (S7, causal) is
OUR addition, not the benchmark's. S9 removes it at the source.

**Design:**
- **Prompt (the ONE change vs S1-r3):** "You will be shown a sequence of frames describing
  steps in a house.\nRespond with a single integer (0 is allowed). Output only the
  integer.\nQuestion: {q}\nAnswer: " — minimal N-free edit of the anchor prompt (same
  structure, same parse). Implement as `--nfree-prompt` in `train_sft_gated.py` (guarded;
  no-flag path byte-identical — re-verify the P1b N8 anchor after the edit). Do NOT adopt
  the upstream JSON protocol here (that changes parsing and decode budget; it stays the
  S5-side protocol note).
- **Training:** S1-r3 recipe VERBATIM otherwise (oracle gate, seq8+16, 10 ep, lr 2e-4,
  fixed trainer, same exclusions incl. S4/S7 dirs). NO virtual-N mixture — with no N in
  the text, the gated forward is the evidence-only forward identically, so real gated
  samples already cover every N by construction (the equivalence smoke's logic, now with
  zero residual channel). Adapter → `checkpoints/sft_fenced_gated_nfree_adapter` (+README
  row; eval contract: gate + nfree prompt).
- **Evals:** oracle-gated ladder N ∈ {8,…,128} (exam files @150) + the S4 dirs (the
  capacity curve under the clean decoder) + model-gated ladder via the S2 pipeline
  (re-capture under this adapter, re-fit gate) + S5-HF (model gate, N4 protocol, N-free
  prompt).

**Pre-registered bands:**
- **H-S9-main** (= H-S8-main): oracle-gated k ≤ 8 band ≥ 0.95 at EVERY N ∈ {8,…,128}.
  Refuted if k ≤ 8 @128 < 0.85 — which, given the equivalence smoke, would mean a
  non-text N-channel exists after all: investigate before any write-up.
- **H-S9-capacity:** the k12/k16 strata at every N ≥ 32 are THE clean resolution
  measurement (no snap, no prompt channel): report acc vs k; c* = first stratum < 0.5.
  S3's physics predicts c* ≈ 16–20; wave-2's S4 (c*=7–8) gets a "calibration-confounded"
  annotation superseded by this cell.
- **H-S9-vs-S8:** per-stratum |S9 − S8| reported; S9 ≥ S8 − 0.03 on every k ≤ 8 stratum →
  N-free becomes the CANONICAL config and S8 the robustness variant; if S8 > S9
  anywhere material, say where and why (the k=N anchors are the expected place: with no
  declared N the "all frames" answer for N > 16 is unknowable — report those strata as
  N/A-by-design, not failures).
- **H-S9-HF:** model-gated MMReD-HF ≥ 0.95 / 0.90 / 0.85 @ seq8/16/32 (gold ≤ 16 ceilings
  measured from the S5 splits: 1.00 / 1.00 / 0.96; majority floors 0.62/0.38/0.16).
**Comparison discipline:** S9 numbers are a NEW prompt anchor — never put them in the same
table row as P1b/frozen without the prompt-variant label; the upstream-fidelity fact above
goes in the write-up as the justification.

## S10 + S11 (authorized by Tal, 2026-08-31): the attention photograph, and the text-controlled α chain

Same permissions/discipline as all of wave 3. Both cells are measurement-only (no training).

### S10 — attention photograph (the formula's direct test; LORAMECH's unrun L4)

**Goal:** stop inferring `m = k·eˢ/(k·eˢ + (N−k) + C)` from interventions and OBSERVE it:
measure the attention mass the answer position actually places on evidence blocks,
non-evidence blocks, and everything else, as k and N vary.
**Instrument:** new `scripts/sparse/probe_attn_photo.py`. sdpa exposes no weights — compute
them manually from FenceHooks captures (`.qkv[L]` q/k projections + `.cos/.sin` rotary),
at the LAST PROMPT ROW: apply rotary, scores = q·k/√d + the injected mask row, softmax,
then sum weight per block (block spans from the trainer's parse; evidence labels from
`oracle_evid`). Per head and per layer L ∈ {12,16,20,24,27}. Report per cell: mass on
{evidence blocks, non-evidence blocks, prompt/sink}, per-head distribution, and the
best-single-head evidence-vs-non-evidence separation (the SELF-GATING question: do trained
read heads already gate internally? report that head's ROC/acc per N).
**Cells (forward-only, ~30 samples/cell, strided, from exam/S4 dirs):**
- P1b (ungated), N ∈ {8,16,32,64} × k ∈ {2,4,8} — the (N−k) term visible: fit
  `share = k·eˢ/(k·eˢ+(N−k)+C)` per layer (s, C fitted; report R² and the fitted values).
- P1g or S8 adapter + oracle gate, same k strata × N ∈ {8,32,128} — prediction: evidence
  share depends on k only (N-invariance of the measured mass, ±10% across N at fixed k),
  tracking `k/(k+C′)`.
- frozen fenced (no adapter), one cell N=32 — the untrained reference.
**Optional sub-cell (Jacobian corroboration, only if time):** directional gradient of the
gold-digit logit wrt each block's input embeddings (one backward per sample; reuse
`scripts/presentation_diagnostics/probe_sensitivity.py` machinery read-only), 20 samples
at N=64 (P1b and gated): per-block saliency share vs the measured attention share on one
axis. First-order corroboration; the flip probe stays canonical.
**Pre-registered bands:** H-S10a: in the P1b arm, evidence share at fixed k decreases when
N doubles, consistent in SIGN at every (k, N-pair) measured, and the one-(s,C)-per-layer
fit reaches R² ≥ 0.8 at the best layer. H-S10b: under the gate, evidence share at fixed k
varies ≤ 10% across N ∈ {8,32,128}. H-S10c (descriptive): the self-gating head's
separation reported per N — if a single head separates at d′ ≥ 3 flat in N, say so
prominently (the internal gate exists). Misses reported as measured; a FAILED fit is a
finding about where the mean-field summary breaks — report per-head heterogeneity.
**Figure F7:** measured evidence-mass vs k and vs N, overlaid with the fitted formula
curves, ungated vs gated panels (+CSV). ~2–3 h on any 48 GB GPU.

### S11 — Hahn at the answer position of the S9 (N-free-prompt) gated model, text-controlled

**Goal:** the α-vs-N measurement with ZERO text variation across N — in the S3/P1g chain
the prompt declared the true N per cell; under S9's prompt every cell is byte-identical
text, so any residual α ≠ 0 is a genuine non-text N-channel (and would contradict the
equivalence smoke — that is the falsification value).
**Depends on:** `checkpoints/sft_fenced_gated_nfree_adapter` (S9). If S9 has not landed,
queue S11 to fire on its landing; if S9 failed to train, run on the S8 adapter instead and
label the chain accordingly (the text-control claim then weakens to "declared-N constant
within cell" — say so).
**Instrument:** `probe_hahn_gated.py` + a guarded `--nfree-prompt` (import the S9 prompt
builder from the trainer — single source; after adding the flag, re-verify the no-flag
anchor: 3 pairs must reproduce the n2_hahn dnorms on the same partition).
**Chains:** (i) vs-N: gate oracle, count flips k→k+1 gold ≤ 8, N ∈ {8,16,32,64,128},
50/50/50/50/40 pairs, controls 12, seed 0 — same pools/pairs discipline as S3.
(ii) vs-k at N=64: base k ∈ {1,2,4,8,16,32}, 30 pairs/k, gate on.
**Pre-registered bands:** H-S11a: read α (L20 final) CI ∋ 0 and |α| < 0.15; margins median
positive at N=128; flip-changes-answer rate ≥ 0.7 at every N. REFUTED if |α| ≥ 0.3 →
non-text N-channel exists: STOP the wave-3 write-up and investigate (positions of the
tail row / sink mass are the suspects; log before proceeding). H-S11b: vs-k fit
`Δ ∝ (k+C)^−γ` with γ ≥ 1 (CI excludes 0.5), C reported.
**Figure F6 (the deliverable, presentation-grade + CSVs):** three panels —
(a) log-log median ‖Δh‖ at the read vs N: S9-gated chain WITH bootstrap CI band, next to
the ungated-P1b reference (reuse `outputs/loramech/n2_hahn` medians) and the frozen joint
reference (ARMOR-A medians), measured floors (replay/ctrl/perm) as a shaded band;
(b) median Δ vs k with the (k+C)^−γ fit line; (c) margin vs N and flip-changes-answer
rate. Axis labels with α values and CIs printed on-plot. → `outputs/sparse/fig/F6_s9_alpha.png`.
~3 h total on any 48 GB GPU.

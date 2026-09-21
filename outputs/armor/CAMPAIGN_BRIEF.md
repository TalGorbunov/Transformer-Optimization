# ARMOR campaign — closing the reviewer holes (localization + drift law + necessity)

**Status: AUTHORIZED (Tal, 2026-08-22) — GPU submissions pre-authorized, no per-job asks,
no GPU-hour cap; discipline rules below are binding.**
Scripts: `scripts/armor/`. Runs: `outputs/armor/`. Log: `STATE.md` (append-only, newest last).
RESULTS.md is Tal's — DRAFT entries go at the bottom of STATE.md, never appended to RESULTS.md.

Context read before start: `outputs/recagg/RELATED_WORK.md` (TOP THREATS), `outputs/recagg/INDEX.md`,
RESULTS.md tail ([2026-07-31b/c] … [2026-08-11c]), recagg STATE.md in full.

## Why (the three reviewer holes)

1. **Hahn's O(1/N) bound has never been measured inside a real frozen VLM** — the thesis
   leans on it as mechanism but cites only theory (Hahn 2020; Barbero et al. 2024 prove
   text-decoder analogs). → Experiment A instruments it directly.
2. **Buitrago & Gu (ICML 2025, arXiv:2507.02782)**: state-passing training interventions
   largely fix recurrent length generalization — a reviewer can claim our R2/R3 drift rung
   is under-trained, not fundamental (RELATED_WORK Top Threat #7). → Experiment B runs the
   state-passing control.
3. **External validity**: every drift/repair number is synthetic (MMReD family). One
   non-synthetic anchor for the composed system is missing. → Experiment C (MLVU-AC).

Plus Hygiene D: the [2026-07-31c] N=128 qkv cells are n=25 and flagged DO-NOT-CITE —
rerun at n≈80 (also validates the harness end-to-end before A–C).

## Discipline rules (Tal, 2026-08-22 — binding)

- Smoke every instrument at tiny scale first (LIMIT≤10, N=8) in `outputs/_scratch/` before
  any full run.
- Prefer a100-public / l40s-public / rtx6k-shared; check `sinfo` across ALL partitions first.
- Right-size QOS: 2h_2g smokes, 12h_4g / 24h_1g real cells; **explicit `--time` always**
  ([[slurm-default-time-2h]]).
- Never comma-lists in `--export` (use files). No pip installs ever — missing dep ⇒ STOP,
  note in STATE.md.
- After touching anything in `gnnformer/`: run `tests/` before submitting.
- Leakage/artifact suspicion ⇒ stop that arm, write it in STATE.md, don't push through.
- House guardrails inherited: K0-sorted trap (stride + report class dist), majority baseline
  next to every EM, raw+balanced+per-class (probe-family rule), bf16 canary lessons
  (measured noise floors, no bit-identity demands).

## HYGIENE D — N=128 qkv swap rerun (warm-up)

Same script (`scripts/presentation_diagnostics/probe_qkv_swap.py`), same protocol as job
127599 (L16, steps task, `data/mmred_longN_park/seq_len_128/all_uniform`, mask-only fence,
shared base positions), LIMIT=80 (pool has 510 dirs; 127599 @n=25 took 7:49 → ~25 min).
Output: `outputs/armor/qkv128/`. Deliverable: citable N=128 row for the [2026-07-31b] table
+ waterfall grid cells de-parenthesized. No pre-registered band (hygiene, not hypothesis);
sanity expectation: d′ ordering preserved (CC > CD > DD > DC as at n=25), tally cells
monotone-ized or honestly still noisy.

## EXPERIMENT A — the Hahn margin/sensitivity instrument (highest priority)

**Goal:** directly verify Hahn's O(1/N) single-symbol sensitivity bound inside the frozen
production VLM (7B, 4-bit, sdpa, joint unfenced forward) — theory-to-model figure.

**Script:** `scripts/armor/probe_hahn.py` (new; reuses gnnformer runtime/fencing/data).
**Data:** paired samples emitted by a new pair-generator mode over the datasets/mmred park
generators: same seed, ONE frame's room assignment toggled for the asked character
(evidence↔non-evidence, gold k vs k+1); **byte-identity of all other frames verified per
pair** (hard assert). N ∈ {8,16,32,64,128}; n≥50 pairs per N (n≥25 OK at N=128).

**A1 SENSITIVITY (joint arm):** per pair, ‖Δ hidden state‖ at the final (answer) position
at L16/L20/L28, plus ‖Δ answer logits‖ (digit-token logits). Curve vs N, log-log fit.
**Noise floor (measured, not theoretical):** per N, (i) exact replay of the same input
(expect ~0; job-level determinism check) and (ii) content-preserving permutation of
non-evidence frames — the reduction-reorder bf16 floor ([[bf16-bit-identity-canary]]).
The floor line drawn on the figure is the permutation floor.
**A1 contrast arm (fence+posreset):** same pairs, fenced forward; measure Δ at the flipped
frame's own per-frame read position (replica room token, probe_supply convention) — the
per-frame supply channel. Prediction: joint final-position Δ decays ~1/N; fenced per-frame
Δ is flat in N. Two-panel money figure.

**A2 MARGIN (same pools, joint arm):** margin = logit(gold count) − max logit(other counts
0..N) at the answer position, vs N; plus joint accuracy-vs-N on the same samples.

**Pre-registered bands (write the verdict against these, whatever it is):**
- A1: joint slope α ∈ [0.7, 1.3] for Δ ∝ N^−α (log-log fit over N=8..128, above the floor).
- A2: the margin crosses the measured noise floor within a factor of 2 of the N where joint
  accuracy reaches chance.
- Contrast: fenced per-frame Δ slope statistically indistinguishable from 0 (|α| < 0.2).

**Deliverable:** one figure (log-log Δ vs N, fitted slope, noise band; margin vs N with the
accuracy-collapse N marked; fenced contrast panel) + CSV per cell.

## EXPERIMENT B — state-passing control (closes the Buitrago objection)

**Goal:** do the Buitrago & Gu interventions rescue the recurrent rungs of the P2 spectrum?

**Script:** `scripts/armor/train_heads_sp.py` — copy of `scripts/recagg/train_heads.py` +
`eval_extrap.py` protocol (recagg originals untouched — their anchors are law). Add:
(i) **state-passing**: carry final hidden state across concatenated training sequences so
long-horizon states are visited during training; (ii) **random initial-state sampling**.
Train R2 (GRU) and R3 (SSM) × {baseline, +state-passing, +random-init, both} on the
existing p1_captures (park + HF@512), fit @N={8,16}, canonical 20000-epoch recipe.
Eval: EXACT P2 extrapolation protocol zero-shot @N={32,64,128}. N=128 leaf captures do not
exist yet → submit them (same protocol as p1_captures: probe_tree_ninv capture-only,
hf `seq_len_128_test` @512 + park longN seq_len_128). Report raw + balanced + per-class +
EM_cls and EM_reg + order canary + max-correct-count per arm (readout-range probe).

**Pre-registered outcomes (either is fine, written down):**
- RESCUED: EM_reg ≥ 0.90 @N=64 zero-shot on HF@512 for any intervention arm.
- NOT RESCUED: below that. Either way, compare against R1 sum-probe 0.996/1.000 and report
  whether max-correct-count stays capped at the trained range (readout-range prediction).

## EXPERIMENT C — external benchmark anchor (MLVU-AC)

**Goal:** one non-synthetic number for the composed frozen system.

**Data:** MLVU-AC counting subset, 206 questions, 32f @392px (`data/mlvu_ac`; mcq_mapping +
prior evals under RESULTS.md [2026-07-24] entries; frozen baseline 0.282 — REUSED, not rerun).
**Route (Arm B / armC v3):** port into `scripts/armor/` (legacy/ read-only): per-frame
captioning (`caption_frames.py` pattern, question-conditioned per-frame records) →
`ask_compile_execute.py` v3-style program compilation (qwen14b, venv_arch) → exact sandboxed
executor. Report EM + per-question-type breakdown + exec-fail rate + MCQ nearest-option
mapping (the prereg rule from [2026-07-24]: parse-fail = wrong).

**Pre-registered band:** composed ≥ baseline + 0.05 (≥ 0.332) ⇒ GO; anything else is logged
honestly as the external-validity caveat.

## Order & pooling

D → A → B → C; A and B may overlap across QOS pools once their smokes pass (spread across
QOS, most cap 3 running jobs/user). End state: STATE.md verdicts vs bands, figures+CSVs in
run dirs, INDEX.md rows for canonical runs, DRAFT RESULTS.md entries at the bottom of STATE.md.

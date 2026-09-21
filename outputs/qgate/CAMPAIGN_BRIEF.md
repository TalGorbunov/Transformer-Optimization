# QGATE — the question-agnostic gate over question-blind (qlast) encodings

**AUTHORIZED (Tal, 2026-09-15, "all done in the morning") — all cells pre-approved.**
Scripts `scripts/qgate/`, runs `outputs/qgate/`, log STATE.md (append-only). RESULTS
drafts at the bottom of STATE. Substrates: `outputs/sparse/layout/qlast_once_ep5_gated/`
adapter (lands from job 150433), P1b, frozen. Prior facts: gated-qlast in-window
0.967/0.900 (oracle); L24h20 task-agnostic AUC ≥0.9998 (G0c, replica layout); S2 LR
gate ≥0.999/frame (label-fit, replica layout); G1 answer-loss gate saturates open.

**Design principle:** the gate is a (question × block) FUNCTION over question-blind
block encodings. Agnostic = generalizes across questions/tasks, not question-blind.

## Cells & bands (fixed now)

- **A1 — answer-row selector scan under qlast.** probe_headscan + --layout, adapters
  {qlast-gated, P1b} × tasks {count, majority} × N∈{8,32,128}, 30/k.
  **H-QA1**: some head ≥0.98 AUC per N per task in the qlast-GATED adapter (P1b arm
  = substrate-dependence control).
- **A2 — question-necessity control.** From the B-capture: LR from block state ALONE
  → evidence bit, with ALT-(C,R) relabelings (same state, different label).
  **H-QA2**: MUST fail (≤ base rate + 0.10) — evidence is question-relative.
- **B-capture** (one job): qlast forwards over train-split dirs (park seq8 + longN16,
  non-exam), per-block span-mean states at L12/L20 + tail question-state, for the
  dir's own (C,R) AND ≥3 ALT (C,R) pairs per dir (questions templated per task via
  tasks.py; evidence per pair derived from states, CPU-asserted).
- **B1 — bilinear/logistic gate**: features [q_state, b_state, q⊙b] → evidence bit.
  Fit: count+exists phrasings, train dirs, incl. ALT pairs. Test: held-out dirs ×
  majority phrasing × unseen (C,R). **H-QB1**: per-block acc ≥0.99 trained-task
  held-out AND ≥0.98 LOO. Labels at fit only; none at inference.
- **B2 — label-free**: Otsu threshold on A1's best-head answer-row mass (no labels
  anywhere). **H-QB2**: within 0.02 per-block acc of B1 on the LOO set.
- **C2 — deployed ladder**: eval_qgate.py two-forward decode on the qlast-gated
  adapter with B1 and B2 gates, exams N∈{8..128} (100/cell N≥64), per-sample
  gate fn/fp parity vs oracle. **H-QC2**: k≤8 acc ≥0.95 @64 AND @128 with B1;
  B2 within 0.05 of B1. C1 (oracle rows) = the layout twins + 5-ep control.
- **C3 (conditional)**: virtual-N answer-loss learned gate on qlast — ONLY if B1/B2
  miss their bands (the incentive-fix hypothesis is otherwise deferred to a later
  campaign; G1b post-mortem feeds it).

Discipline as always: smokes in _scratch (instruments are eval-only + one CPU fit),
explicit --time, no comma --export, class dist + majority baselines, bands never
re-tuned. Oracle evidence labels appear in FIT (B1) and EVAL accounting only.

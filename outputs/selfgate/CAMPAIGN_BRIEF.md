# SELFGATE campaign — from oracle gate to a self-supervised, task-agnostic learned gate

**Status: AUTHORIZED (Tal, 2026-09-01) — all cells pre-approved, no stop-for-OK checkpoints.**
Scripts: `scripts/selfgate/`. Runs: `outputs/selfgate/<cell>/`. Log: `STATE.md` (append-only,
newest last). Index: `INDEX.md`. Sbatch: `slurm/selfgate_*.sbatch`. Parent campaign: SPARSE
(`outputs/sparse/` — its verdict, adapters, probes and discipline are the substrate).
Sibling precedent: learnmask (`outputs/learnmask/`, peer-endorsed 2026-08-12) — this campaign
is that direction, now grounded by SPARSE's two facts below.

## 0. One paragraph

SPARSE proved the gated read works (k-regime: α_N 0.80→0.07, k≤4 exact at every N, capacity
c* in k) but its gate is an ORACLE: gold per-frame evidence labels build the mask at train
AND test (deployable S2 variant = LR probe on L20 replica states, ≥0.999/frame, but still
label-supervised and single-task). Two S10 facts say a better gate is latent: (1) a single
trained head — L24 h20 in the P1b adapter — separates evidence from non-evidence blocks at
AUC 0.993–1.000, FLAT in N (frozen best head: 0.848 → the selector is trained-in); (2) the
read's softmax can't exploit it because softmax cannot assign exact zeros (per-block edge
e^s ≈ 1.35×). SELFGATE replaces the oracle in two steps: G0 — gate directly off the model's
own internal head, zero training, no labels; G1 — a CoGNN-style learned discrete gate
(ST-Gumbel bit per block, trained END-TO-END BY THE ANSWER LOSS ONLY, no evidence labels),
which is task-agnostic by construction because the bit is computed from the [frame+question]
block state. Evidence labels are used for EVALUATING gate bits, never for training them.
α-entmax read layers (G2) are the architectural endpoint — out of scope here, one line in
the write-up.

## 1. Fixed inputs (all exist)

- Adapters: `checkpoints/sft_fenced_gated_adapter` (P1g), `sft_fenced_gated_vn_adapter`
  (S8), `sft_fenced_gated_nfree_adapter` (S9; eval contract: gate + nfree prompt),
  `sft_fenced_le16_ep10_adapter` (P1b, ungated — where L24h20 was found). Check
  `checkpoints/README.md` for any newer canonical rows (S9b twins) before choosing; use the
  CURRENT canonical gated adapter for decoding cells and note the choice in STATE.
- Trainer: `scripts/sparse/train_sft_gated.py` (gate plumbing, `--gate-from-layer` two-hook
  layer-split machinery, nfree prompt, MASK HELD THROUGH BACKWARD — the cos-0.777 lesson;
  any new sample-dependent mask MUST follow the same hold-through-backward pattern).
- Attention photograph: `scripts/sparse/probe_attn_photo.py` (manual per-head per-block
  masses from FenceHooks q/k + rotary at the answer row; per-row softmax-sum asserted).
- Gate pipeline: `scripts/sparse/capture_verdicts.py` + `train_gate.py` (S2 two-forward).
- Probes: `scripts/sparse/probe_hahn_gated.py` (α chains, `--nfree-prompt`).
- Exams: `outputs/loramech/examdirs/exam_ff_N{8..128}.txt` + `outputs/sparse/s4/dirs_*.txt`
  (+ S7 dirs excluded from any training). Evidence labels for EVAL: `oracle_evid`.
- Discipline gotchas: stride K0-sorted dirs; class distribution + majority baseline per
  cell; explicit `--time`; check all partitions; no comma `--export`; no installs (sparsemax/
  entmax, if ever touched, is pure-torch — but G2 is OUT of this campaign); smokes in
  `outputs/_scratch/`; anchors before full runs (eval-only P1b exam_ff_N8 = 1.0000 after any
  trainer edit; 3-pair probe byte-match after any probe edit).

## 2. Cells

### G0 — self-gate from the internal head (no training, no labels at eval)

- **G0a — locate the selector in the deployed adapter.** Rerun the photograph's per-head
  separation readout under the CURRENT canonical gated adapter (and P1b for reference),
  ungated forwards, N ∈ {8,32,128}, 30/cell: report every head with evidence/non-evidence
  AUC ≥ 0.98 per N. (L24h20 was found in P1b; the gated adapter may have moved or sharpened
  it.) CPU-light GPU pass, ~1 h.
- **G0b — threshold-gate decode.** Two-forward eval: pass 1 ungated, compute the chosen
  head's per-block mass at the answer row, threshold θ (ONE scalar, chosen on the train
  split by Youden/accuracy — no exam data), build `hide_cols`, pass 2 gated decode. Exam
  ladder N ∈ {8..128} count; per-sample gate_fn/gate_fp accounting exactly as S2. ~2 h.
- **G0c — cross-question relevance (the agnosticity preview).** Capture ungated block
  masses for the SAME frames under exists/majority question texts (templates from the REDUX
  design; build a minimal `scripts/selfgate/tasks.py` if `scripts/redux/` lacks one) and
  report the head's AUC against each question's own evidence labels. No decoding — AUC
  only. ~1 h.

### G1 — learned discrete gate (CoGNN-style, answer-loss-only)

- **Mechanism.** Gate head = linear probe (+bias) on each block's replica-slot state at
  layer L_g = 12 (position: same slot the S2 capture uses, at the layer the fence's
  verdict is formed and `--gate-from-layer` already splits on) → logit per block →
  straight-through Gumbel-sigmoid bit (hard forward, soft backward; temperature annealed
  5 → 0.5 over epochs; seedable) → the bit builds `hide_cols` applied at layers ≥ L_g via
  the existing two-hook machinery. Gate head params + LoRA train jointly on the ANSWER
  LOSS ONLY. The mask tensor (and the soft relaxation) is HELD THROUGH BACKWARD.
- **G1a — count-only.** S1-r3 recipe otherwise (seq8+16, 10 ep, lr 2e-4, nfree prompt per
  the S9 canonical config, same exclusions). NO sparsity penalty, NO evidence supervision.
  Adapter+gate → `checkpoints/selfgate_g1a/` (save the gate head with the adapter; eval
  contract: learned gate + nfree prompt). Evals: exam ladder N ∈ {8..128} + S4 dirs;
  report per stratum vs the oracle-gated reference; gate-bit AUC/acc vs `oracle_evid`
  per N (evaluation only); α chain (probe with the learned gate: per-member bits computed
  by the model, both members) if time.
- **G1b (conditional).** ONLY if G1a's gate saturates open (bit ≈ 1 everywhere, measured):
  add an L1 penalty λ·mean(bits) with λ swept {1e-3, 1e-2} — one change, two arms. If
  G1a's gate collapses closed (answers degenerate), retry once with open-init bias +2 and
  slower anneal; log both outcomes as measured.
- **G1-multi (the task-agnostic claim).** Mixture training {count, exists, majority}
  (labels/templates via `tasks.py`; yes/no answer tokens for the binary tasks; majority
  keeps declared-N in ITS prompt — the denominator channel; count stays nfree). Two arms:
  (i) all three tasks; (ii) LEAVE-ONE-OUT: train {count, exists}, test the gate's bits AND
  accuracy on majority. Gate-bit AUC on the held-out task = the agnosticity number.
- Cost: G1a ~5 h train + ~3 h eval (48 GB); G1-multi ~5 h + data gen (CPU).

## 3. Pre-registered hypotheses and bands (fixed now; report every one)

- **H-G0.** Some internal head in the deployed adapter reaches AUC ≥ 0.99 per N (G0a), and
  the G0b self-gate ladder is within 0.05 of the oracle-gated reference on every k ≤ 8
  stratum at every N, with every gap accounted per-sample by gate errors. If no head
  clears 0.99 in the gated adapter but L24h20 does in P1b, run G0b through P1b's pass-1
  (capture arm) + gated-adapter pass-2 and say so. G0 null (no head ≥ 0.95) → report; G1
  becomes the only route.
- **H-G0c (descriptive).** The selector head's AUC under exists/majority questions
  reported per task; ≥ 0.9 = the internal relevance signal is already question-general.
- **H-G1a (headline).** The label-free learned gate reaches: gate-bit AUC vs gold evidence
  ≥ 0.99 at every N (bits evaluated, never trained, on labels); k ≤ 8 band within 0.05 of
  oracle at every N ∈ {8..128}; flat in N (≤ 0.05 drop 16→128). Refuted if the k ≤ 8 band
  @128 trails oracle by > 0.15. Degenerate-gate outcomes (open/closed) are measured
  failure modes with the pre-registered G1b responses — not band misses.
- **H-G1-multi.** Leave-one-out gate-bit AUC on the held-out task ≥ 0.90, and held-out
  task accuracy ≥ 0.8× its in-mixture value. This is the task-agnosticity claim; a miss
  bounds it honestly ("relevance transfers within trained families only").
- **H-SAFETY.** Every trained arm re-verifies the trainer anchor first; any gate mechanism
  interacting with gradient checkpointing runs the diag_grad-style A/B/C gradient check
  (cos ≥ 0.999 vs no-ckpt truth) BEFORE its full training run.

## 4. Deliverables

STATE.md log + verdict section; INDEX.md; figures in `outputs/selfgate/fig/`:
(FG1) oracle vs S2-LR vs G0 vs G1a ladders on the k ≤ 8 band (the "no labels needed"
figure); (FG2) gate-bit AUC per N per mechanism incl. G0c/G1-multi transfer bars;
(FG3, if the α chain runs) read α under the learned gate. `checkpoints/README.md` rows
with eval contracts. RESULTS.md entries at close (pre-authorized). Honesty flags: which
adapter served as substrate per cell; oracle/model/learned gate labeled in every number;
the ratio-task denominator caveat stated wherever majority appears.

## 5. Order

G0a → G0b ∥ G0c → (checkpointless report in STATE) → G1a (submit its training as soon as
G0a picks the substrate adapter — G0b/G0c don't block it) → G1a evals → G1b only on its
trigger → G1-multi → figures → verdict → RESULTS.md. Smokes before every full cell; tmux.

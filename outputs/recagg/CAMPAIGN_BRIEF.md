# RECAGG campaign — recurrent aggregator head over fenced per-frame verdict states

**Status: APPROVED DIRECTION (Tal, 2026-08-17) — P0 is CPU-only and may start immediately;
GPU phases (P1+) follow the normal show-plan-then-submit rule.**
Origin: Hahn-bound analysis (2026-08-17 session) — attention has O(1/N) per-item output
sensitivity (softmax = average), recurrence has O(1); the two architectures fail on each
other's strengths. tmux session: `recagg`. Scripts: `scripts/recagg/`. Runs: `outputs/recagg/`.

## Goal & thesis framing

Test the **sensitivity-complementarity hypothesis** directly: keep the frozen fenced
perception (parallel, question-conditioned, per-frame verdict states decodable at
0.985–0.999) and swap ONLY the aggregator. A tiny trained recurrent head (GRU /
minimal selective-SSM) consumes the sequence of per-frame verdict states and emits the
count. Theory prediction: a recurrence is an integrator (per-item influence O(1),
independent of N), so — unlike the flat softmax read (capacity law c(fan), halves per
doubling) and unlike the tree (depth-extrapolation death at untrained levels lv5/6) —
the recurrent head should be **length-invariant by construction**, bounded only by
per-frame perception fidelity.

Thesis row this produces: "same frozen perception, three aggregation mechanisms" —
flat read (theory says impossible), in-model fan-2 tree (works, needs re-quantization,
depth-capped per forward), external recurrent scan (works, length-invariant, but
external module). This is the ablation the Hahn analysis says the thesis must contain.

Honest scope: part of this story is already told by gate→tally (external counter 0.96
@N=8). The recurrent head's MARGINAL contribution is (a) learned, (b) order-aware,
(c) handles distractors/routing (the wall that stopped the June Mamba-operator runs)
— so the hard/distractor cells (P3) are where its added value is judged, not clean
counting.

## Pre-registered predictions (score at campaign close)

- H1 (in-length): recurrent head EM @N=8 lands at the per-frame-fidelity bound
  (~0.89–0.99 = p_frame^8 for p_frame 0.985–0.999), i.e. aggregation adds ≈no loss.
  A plain sum-over-probe-verdicts (R1 control) should land in the SAME band — if the
  recurrent head BEATS R1 in clean counting, suspect leakage/overfit, not magic.
- H2 (extrapolation, THE headline): trained @N∈{8,16}, evaluated zero-shot @N∈{32,64}:
  EM tracks p_frame^N with no additional N-dependent penalty. Compare: flat probe
  collapses (0.67→0.12), tree emitted needs the cascade (0.875/0.811), lv5/6 dead.
- H3 (routing/distractors): on hard variants (irrelevant frames / other characters),
  the selective head beats sum-over-verdicts fit on the same features, because
  selection is learned jointly with aggregation. This is the cell that failed in the
  June Mamba-operator runs ("routing is the wall") — those runs never converged
  ([[hard-task-ceilings-undertrained]]), so this is a genuine open question, not a rerun.
- H4 (architecture): minimal selective-SSM ≈ GRU on clean counting (both integrate);
  any gap appears only on H3 cells (input-dependent gating = selection).

## Data (all existing or cheap to capture)

**Already on disk (P0 uses these, zero GPU):**
- `outputs/ninv/20260810_000358_park8_leaf/feats_N8.npz` — park N=8: `leaf|0|20|mean`
  (200, 8, 3584) fp16 per-frame replica states @L20, `Y` (200,8) per-frame evidence
  bits, `G` (200,) gold counts. Layout: fenced blocks + posreset (probe_tree_ninv).
- `outputs/ninv/20260809_235142_hf8_leaf392/feats_N8.npz` — same for MMReD-HF @392
  (+ `..._hf8_leaf512` twin if present — verify).
- Node/SQ features (NOT leaves) for N=16–64: `outputs/superquery/capture_16_64_128790/`
  — useful only for cross-checks; leaf keys absent there (verified 2026-08-17).

**Needs capture (P1, small 2h_2g jobs):** per-frame leaf states @L20 (+L16 for a
layer check) at N=16/32/64 — park longN roots (`slurm/lib/roots_inlength.txt` /
`data/mmred_longN_park`) AND MMReD-HF `seq_len_{16,32,64}_test` steps_in_room pools.
The capture path exists: `scripts/ninv/probe_tree_ninv.py` leaf-read mode (jobs
130087/130089 precedent) — VERIFY the flag set before submitting; if leaf mode was a
one-off edit, port it properly into `scripts/recagg/capture_leaf.py` (copy, don't
touch ninv scripts mid-campaign).

**P3 (hard/distractor) data decision for Tal:** MMReD-HF task variants with distractor
structure (e.g. char_on_char_*, aug_dense_qa) vs our text-MMReD hard variants. Needs a
per-frame-verdict label definition per variant (evidence bits derivable from per-frame
GT room states — mmred metadata route, see ninv Phase-0 precedent).

## Guardrails (house lessons that WILL fire here)

- **K0-sorted trap** ([[mmred-hf-k0-sorted-trap]]): stride every HF pool slice; print
  class distribution for EVERY split, train included.
- **Majority baseline next to every EM** (the smoke-EM mirage, ninv 15:3x): every
  reported EM/acc ships with its majority/prior on the same split; per-class + balanced
  per metrics_skew ([[probe-family-artifact]]).
- **Per-frame fidelity bound reported next to every aggregate EM** (attribution: is the
  miss perception or aggregation?). Fit the frame-verdict probe on a held split; report
  p_frame and p_frame^N alongside.
- **No pip installs** — no mamba_ssm: implement a minimal diagonal selective SSM in
  pure torch (sequential loop is fine — sequences are ≤128 steps, hidden ≤3584-dim
  projected to ≤256); GRU via torch.nn. Heads ≤ ~1M params.
- **fp16 features**: cast to fp32 before head training; standardize per-dim on train
  split only (fit-on-train scaler, saved with ckpt).
- **Train/eval split hygiene**: park and HF are different domains — never mix in one
  fit; extrapolation heads are fit at N∈{8,16} ONLY and never see longer captures
  (the ninv N-invariance gate discipline).
- **Order canary**: clean counting is permutation-invariant — evaluate the trained
  head under frame-order permutation (answer must be stable); a big order sensitivity
  on clean counting = the head is keying on position artifacts.
- RESULTS.md only on explicit "log this". INDEX.md updated when a run becomes canonical.

## Arms

| arm | aggregator over the same leaf states | question |
|-----|--------------------------------------|----------|
| R1 sum-probe (control) | logistic verdict probe per frame → external sum (gate→tally analog on THESE captures) | the trivial-aggregation floor; H1's twin |
| R2 GRU head | 1–2 layer GRU (proj 3584→128/256) → count logits (support 0..N_max) or scalar+round (report both) | does ANY recurrence suffice? |
| R3 minimal selective SSM | hand-rolled diagonal SSM, input-dependent (Δ,B,C) gating, pure torch | Mamba-style gating; H4 contrast vs R2 |
| R4 attention-pool control | single softmax attention read over leaf states (trained query) | the Hahn bound INSIDE our own feature space — should reproduce the c(fan) collapse in-length AND fail extrapolation; the negative control that makes the figure |
| R5 (later, own approval) | task-agnostic: frozen VLM emits per-frame verdict TEXT → pretrained Mamba LM reads + answers (venv_arch, installs need OK) | the architecture demo; out of P0–P3 scope |

## Phases & cost

| Phase | What | Compute | Deliverable |
|-------|------|---------|-------------|
| P0 | R1–R4 on existing N=8 captures (park + HF392); splits, scaler, probes, heads; order canary | CPU (4h_0g, mem ≤16G) — zero GPU | in-length EM table w/ majority + p_frame^N bounds; H1/H4 first read |
| P1 | leaf captures N=16/32/64, park + HF; verify/port capture script; layer check L16 vs L20 | ~4–6× 2h_2g GPU jobs (~15–40 min each by ninv capture timing) | feats npz per N per domain |
| P2 | extrapolation: fit @{8,16}, eval @{32,64}; all four arms; headline figure | CPU | **EM-vs-N figure** + H2 verdict |
| P3 | distractor/hard variant captures + rerun arms (H3) | ~2–4× 2h_2g + CPU | routing verdict: learned selection vs sum |
| P4 | (optional, needs separate OK) R5 task-agnostic text-interface arm | venv_arch, TBD | architecture demo |

Campaign wall-clock: P0 same-day; P1+P2 ~1 day incl. queue; P3 ~1 day. Tiny GPU
footprint (~2–4 GPU-hours total through P3).

## Headline figure (decided up front, per presentation policy)

**"Same perception, different aggregators": EM vs N (8/16/32/64), one line per arm** —
R4 attention-pool collapsing along ~c(fan), R1 sum and R2/R3 recurrent flat along the
p_frame^N bound (drawn as a dashed envelope), tree-cascade record numbers
(0.925/0.875/0.811) as reference marks. Table twin with majority baselines per cell.

## Relation to record (baselines quoted, not rerun)

- Flat frozen probe capacity: 0.67/…/0.12 (fan law, superquery 128845).
- gate→tally external counter: 0.96 @N=8 ([[oneforward-a3-go]]).
- In-model tree emitted: v2 0.920 in-length; cascade 0.875 @32 / 0.811 @64 (ninv).
- Native flat text-count: 0.55@8 → 0.18@128 (textcount 128855).
- June Mamba-operator result: Mamba ≥ sum on clean counting, ceilinged by routing on
  hard tasks (undertrained — [[diff-mamba-results]], [[hard-task-ceilings-undertrained]]).

## Open decisions for Tal

1. P1 capture scope: park longN + HF test pools both, or HF-only (benchmark-first)?
2. P3 data: which hard/distractor variant is the canonical routing cell?
3. Count readout: classification over 0..N_max vs scalar regression + round — default
   is report both, classification primary (matches record convention).
4. R5 (pretrained Mamba LM, text interface): park until P2 lands?

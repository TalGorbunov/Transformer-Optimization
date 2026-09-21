# SELFGATE campaign — STATE (append-only, newest last)

Brief: `CAMPAIGN_BRIEF.md` (authorized by Tal 2026-09-01, all cells pre-approved).
Cells: G0a selector-in-deployed-adapter · G0b threshold self-gate · G0c cross-question
AUC · G1a learned gate (answer-loss-only) · G1b conditional · G1-multi leave-one-out.

## [2026-09-01] Campaign created (Claude, with Tal) — no jobs yet

## [2026-09-14 ~12:40] CAMPAIGN STARTED (Tal's go) — context re-read, G0a instrument built, smoke running

Context read per the agent prompt: sparse STATE (full section map + the two load-bearing
lessons re-read verbatim: the clear-before-backward gradient bug + its diag_grad
B-vs-C cos-1.000000 verification pattern; the S10 L24h20 discovery via block_mass.csv
AUC), sparse INDEX, checkpoints/README (canonical gated adapter =
**`sft_fenced_gated_vn_nfree_adapter`** — S9b ep10 twin, contract: gate +
`--nfree-prompt`; chosen as the G0 substrate, P1b as the reference arm),
probe_attn_photo.py (hook-order lesson: capture hooks POST-wrap), train_sft_gated.py
(builders: nfree keeps the parse_layout needle), learnmask brief.
- `scripts/selfgate/probe_headscan.py` WRITTEN (copy-extended from probe_attn_photo;
  original untouched): --nfree-prompt, ungated-by-default forwards, in-script
  per-(layer,head) rank-AUC report (the L24h20 readout) + auc.json.
- `slurm/selfgate_g0a.sbatch`: chains S9b(nfree)+P1b × N∈{8,32,128}, 30/k-stratum.
- **Smoke 148359 RUNNING** (limit 2, k=2 only → _scratch): mechanical pass + the P1b
  N=8 cell must show L24h20 near the top of the AUC table (the S10 anchor reproduced
  through the new script) before the full G0a submits.
- Next after G0a submit: G1a trainer (`scripts/selfgate/train_sft_selfgate.py`,
  copy-extend of train_sft_gated.py — ST-Gumbel bit per block from the L12 replica
  slot, answer-loss-only, mask+soft-relaxation HELD THROUGH BACKWARD) + its
  MANDATORY diag_grad A/B/C check before any full training (H-SAFETY).

## [2026-09-14 ~13:30] G0a smoke green → full G0a running (one self-caught submit bug)

- Smoke 148359 (17 min, all 6 cells): AUC machinery + both adapters + nfree layout all
  work. n=2 AUCs are noise for head-ranking (full run is the L24h20 anchor arbiter) —
  but one PRELIMINARY signal worth pre-registering an eye on: the S9b gated adapter's
  best head reads only 0.79–0.94 (P1b shows saturated separators). If it survives
  n=30: oracle-gated training ATROPHIED the internal selector (it never had to reject
  non-evidence — the mask did it), and G0b's pass-1 head should come from P1b.
- Submit bug SELF-CAUGHT: first full submit (148380) carried KLIST=2,4,8 through
  --export — the comma-split trap; CANCELLED before any GPU time, wrapper now takes
  dashes (KLIST=2-4-8 → commas internally), dry-run verified.
- **Full G0a = 148381 RUNNING** (a100 n310, 12h_4g, --time=5:00:00, 30/stratum).
- Next block: G1a trainer build (train_sft_selfgate.py) + diag_grad A/B/C gate.

## [2026-09-14 ~15:00] G0a LANDS (148381, 1h23) — anchor reproduced; the DEPLOYED adapter has NO internal selector (H-G0 clause 1 formally MISSED as written); P1b carries a POPULATION of selectors

`outputs/selfgate/g0a/{p1b,s9b}_N{8,32,128}/` (90 samples/cell = 30×k∈{2,4,8}).

- **S10 anchor REPRODUCED through the new instrument:** P1b L24h20 AUC
  0.9997/0.9998/1.0000 @N=8/32/128 (S10: 0.9997/0.9927–1.0000) ✓. And the selector
  is not one lucky head — P1b has a POPULATION: L24 h6/h17/h4/h20 and L20 h1/h3/h15
  all ≥0.998 at N≥32; ~10 heads ≥0.98 at N=128.
- **The smoke's atrophy signal CONFIRMED at n=90:** the CANONICAL gated adapter
  (S9b, scanned under its own nfree contract, ungated forwards) has NO head ≥0.98
  at ANY N — best per layer 0.52–0.75, near-chance at N=128 (best 0.586).
  Oracle-gated training removed the need to separate evidence (the mask always did
  it) and the separation is GONE from the attention geometry under ungated forwards.
  **H-G0 clause 1 ("some head in the DEPLOYED adapter ≥0.99 per N") is MISSED as
  pre-registered** — the selector lives in P1b, not in the gated adapter. Logged as
  measured; no band re-tuning.
- **Consequence for G0b (design adaptation, logged):** pass 1 = P1b ungated forward
  → chosen-head mass (head fixed on TRAIN-split data only, from the P1b population)
  → threshold θ → hide_cols; pass 2 = S9b gated decode (its contract). Two-adapter
  two-pass; the brief's per-sample gate_fn/gate_fp accounting unchanged.
- Narrative note for the close: the oracle gate is a CRUTCH — it atrophies the
  model's own selector. The G1 answer-loss-only gate must keep the selector alive
  because gating errors hurt its own loss. G0a strengthens G1's motivation.

## [2026-09-14 ~16:10] G1a trainer + H-SAFETY diag WRITTEN; diag running

- `scripts/selfgate/train_sft_selfgate.py`: the learned-gate trainer. Two-forward
  step: pass A no_grad plain-fence → L12 replica-slot states (detached; the head's
  gradient arrives via the MASK path) → linear GateHead → ST Gumbel-sigmoid bits
  (τ annealed 5→0.5) → differentiable tail-row column penalty (`gated_tail_mask`,
  functional build, grad flows bits→mask) applied at layers ≥12 via the two-hook
  split → pass B answer loss, masks HELD THROUGH BACKWARD, cleared after. ANSWER
  LOSS ONLY (`--l1-bits` exists but is 0 for G1a; oracle_evid appears ONLY in the
  eval accounting — logged per H-SAFETY rule 2). Eval: deterministic hard bits
  (sigmoid>0.5, no noise), per-sample gate_fn/fp vs oracle, per-k tables,
  adapter+gate_head.pt saved together.
- `scripts/selfgate/diag_grad_selfgate.py` (H-SAFETY gate): (1) oracle-bit mask
  EQUIVALENCE vs build_block_mask (tail rows exact; non-tail diffs only on
  base-forbidden entries); (2) A/B/C gradient check for BOTH param groups (LoRA +
  gate head), PASS iff cos(B,C) ≥ 0.999 both; (3) gate-grad liveness. Shared
  fixed noise across configs. **Diag = job 148493 RUNNING.** G1a full training
  submits only on PASS.

## [2026-09-14 ~16:40] H-SAFETY diag FAILED as designed-to-catch — zero gate gradient; MATH-backend hypothesis under test

- Diag 148493 (3m45s): CHECK1 mask-equivalence PASS (tail max|diff| 0.0; non-tail
  diffs only on base-forbidden entries). CHECK2/3 FAIL: gate-head gradient EXACTLY
  0 — the mask path does not reach the head. No training was submitted (the gate
  worked exactly as H-SAFETY intends).
- Root-cause hypothesis: the EFFICIENT sdpa kernel does not differentiate w.r.t.
  attn_mask (MATH does by construction). Secondary diag weakness fixed at the same
  time: the seed-0 first sample is gold=0/evid=∅ (weak mask coupling) — diag now
  requires gold ≥ 2. Diag2 = 148501 (--sdpa math) RUNNING; if gate grads go live,
  the trainer's pass B pins MATH (memory OK at N≤16; ~2× step cost, logged).

## [2026-09-14 ~17:30] Micro-diag 148505 PINPOINTS the break — working pattern verified; trainer patched; diag3 running

- STAGE 1 (torch-only): MATH computes attn-mask grads (fp32+bf16, norm ~3.0);
  **EFFICIENT hard-errors on mask-grad backward** ("LSE is not correctly aligned
  (strideH)") — the kernel cannot do it, ever.
- STAGE 2 (model path, MATH + DIRECT 4-D holder injection): loss depends on the
  gate (finite-diff +0.065 hiding one evidence block), backward REACHES the mask
  tensor (register_hook fired), **gate-head grad norm 1.5e+04 — LIVE**.
- Adopted verbatim in the trainer: `set_grad_mask` (direct 4-D holder injection)
  + `GRAD_SDPA=[MATH]` for pass B only (pass A/eval stay EFFICIENT+MATH — no mask
  grads needed there). The set_mask-chain zero-grad config is noted, not
  re-litigated (working pattern > archaeology; one-line suspect: the view/to
  re-chain + in-hook dtype cache).
- diag3 = 148522 (full A/B/C with the verified pattern). G1a training submits on
  PASS only.

## [2026-09-14 ~18:30] ROOT CAUSE FOUND (diag4 148538) — hard-closed gates are gradient black holes; MECHANISM AMENDED before any training

- Instrumented diag: bits all-0 in this head draw; gmask.grad LIVE (2.3) but
  bits.grad EXACTLY 0. Mechanism: a hard-closed block's softmax weight is exactly 0
  (MASK_MIN saturates), and ∂loss/∂mask_entry ∝ that weight → the REOPEN direction
  carries no signal, ever. Micro-148505 "worked" only because its bits were all-1
  (the close direction is live; the reopen direction is the black hole). Under
  training this is closed-forever — a mechanism flaw, not plumbing.
- **AMENDMENT (pre-training, measured-failure-driven, logged as a brief deviation):**
  TRAIN-mode mask = MULTIPLICATIVE soft gating, penalty log(clamp(p,1e-6)) with
  p = Gumbel-sigmoid soft prob (τ annealed as specced) — closed blocks keep ~1e-6
  mass so both gradient directions stay alive; sigmoid saturation anneals toward
  hard. EVAL-mode mask unchanged: hard bits × MASK_MIN = exact oracle-mask
  semantics (check1 equivalence untouched). Hard bits remain the REPORTED gate.
  The answer-loss-only and no-evidence-labels contracts are unchanged.
- Diag CHECK3 extended: reopen-liveness — a deeply-closed head (bias −6) must also
  receive gradient. diag5 = 148542 RUNNING. Training launches on PASS.
- GPU archaeology bill for the whole hunt: 5 diagnostic jobs ≈ 16 min total.

## [2026-09-14 ~19:10] H-SAFETY PASS (diag5 148542) → G1a TRAINING LAUNCHED

- diag5: check1 mask-equiv ✓; check2 cos(B,C) LoRA ≈1.0, GATE = 1.000000 ✓;
  check3 BOTH directions live (open 3.2e-1, closed-bias−6 1.8e-1) ✓. The amended
  mechanism (soft-log train mask / hard eval mask) is validated end-to-end.
- **G1a = job 148543 RUNNING (rtx6k n318, 24h_1g, --time=20:00:00):** roots
  seq8+longN16 (exam_ff_N8/N16 excluded), 10 ep per the brief, nfree prompt,
  answer-loss-only (no l1, no labels in loss — H-SAFETY rule 2 restated here);
  two-forward step (pass A EFFICIENT no-grad, pass B MATH grad); in-job exams
  exam_ff_N8/16/32 with the LEARNED gate + per-sample gate_fn/fp vs oracle
  (EVAL-ONLY reference) + per-k tables. → `outputs/selfgate/g1a/`.
  Cost note: two-forward + MATH ⇒ ~2-3× P1b step cost; 20h budget is generous.
- Diagnostics bill for the whole gradient hunt: 6 jobs, ~19 min GPU.

## [2026-09-14 ~20:20] G1a attempt 1 CRASHED (reentrant GC × grad-mask) → GC-off OOMs @N=16 → non-reentrant GC under test

- 148543 FAILED 2m43s: "backward through the graph a second time" — REENTRANT
  gradient checkpointing recompute re-consumes the grad-carrying mask's single-use
  upstream graph. Diag blind spot identified: no diag config ever called
  model.train(), so GC stayed dormant everywhere — diag B "matched" C trivially.
- GC-off smoke 148554: N=8 steps TRAIN CLEAN end-to-end (loss/bits/eval accounting
  live) but N=16 OOMs even on a ~95GB card — MATH-no-GC stores ~28 fp32 attention
  tensors ≈ 100GB at seq 3.4k. GC-off cannot cover the brief's seq8+16 recipe.
- **diag6 = 148563:** CHECK4 — model.train() + GC(use_reentrant=False) + held
  masks: must not crash and must match config C (cos ≥ 0.999). If PASS → trainer
  uses non-reentrant GC (memory of GC, exactness vs C verified). If FAIL →
  fallback decision: selective GC or a logged recipe deviation (train ≤8-only).

## [2026-09-14 ~21:00] CHECK4: non-reentrant GC ALSO crashes → AMENDMENT 2: LoRA ≥ L_g only, GC off

- diag6 148563: use_reentrant=False + train() + held grad-mask → CheckpointError
  ("recomputed values have different metadata") — BOTH GC flavors are incompatible
  with hook-injected grad-carrying masks. Fact chain now closed: GC×grad-mask
  impossible (148543, 148563); no-GC all-layer @N=16 = ~100GB OOM even on n318's
  NEW 96GB cards (148554 — note: rtx6k n318 has been upgraded, 94.97GiB visible).
- **AMENDMENT 2 (measured-constraint-driven, pre-training, logged):** LoRA scope =
  decoder layers ≥ L_g=12 only (`layers_to_transform`; vision untouched), GC off,
  input-grads off → below-L12 runs grad-free, stored graph ≈ 58GB @N=16 → fits
  80GB a100-public + 96GB n318 (l40s/48GB EXCLUDED from partitions). Precedent:
  the method's own carrier-layer L*=12 stack. Bonus: stationary L12 input for the
  gate head. Deviations vs the S1-r3 recipe (all-layer+vision LoRA) are now TWO —
  both forced by measurement, both logged; oracle-parity comparisons will carry
  the caveat.
- Smoke2 = 148574 (rtx6k/a100 only): N=16 must TRAIN without OOM → then G1a v2.

## [2026-09-14 ~21:30] Smoke2 PASS (N=16 trains, 11.53M params = exact ≥L12 scope) → G1a v2 RUNNING

148574: no OOM at any length, scope exactly as computed (16 layers × 720,896, vision
excluded), end-to-end machinery live. **G1a v2 = job 148575 RUNNING (rtx6k n318 96GB,
24h_1g, --time=16:00:00)** — 10 ep, nfree, answer-loss-only, amendments 1 (soft-log
train mask) + 2 (LoRA ≥ L12, GC off) in effect; in-job exams exam_ff_N8/16/32 with
learned-gate decode + gate-bit accounting. → `outputs/selfgate/g1a/`. ETA ≈ 9-11h.
While it trains: G0b/G0c next.

## [2026-09-14 ~22:00] G0c launched alongside G1a training

- probe_headscan + `--task {count,exists,majority}` (count = byte-identical original
  path; templates via the sparse trainer's build_task_messages = the redux single
  source; AUC labels = the same occupancy set — cross-QUESTION is the variable).
- **G0c = 148593 RUNNING (a100 n307):** P1b × {exists,majority} × N∈{8,32,128},
  30/k-stratum → `outputs/selfgate/g0c/`. Count reference = the G0a p1b cells.
  (One self-caught submit slip: dash k-list passed to the python CLI — dashes are
  the --export convention only; cancelled 148592 pre-start, zero cost.)
- G1a v2 (148575) training on n318. G0b (θ-threshold deployable gate) next build.

## [2026-09-15 ~01:20] G1a COMPLETE (open-saturation branch, measured) + G0c COMPLETE (agnosticity proven) → G1b λ arms LAUNCHED

**G1a (148575, 3h29, n318)** → `outputs/selfgate/g1a/` run dir. mean_bit 1.000 from
ep1 (all-ones bits: fn=0, fp=all non-evidence — the gate opened everything);
best ep7 val 1.000, TEST_IID 1.000. Exams (learned gate = effectively UNGATED):
**N=8 1.000 · N=16 1.000 · N=32 0.360** with the ungated snap anatomy (k0–k2 +
k12/k32 anchors perfect, mid dead) — vs 0.847@32 for the same substrate under the
ORACLE gate. Readings: (1) the ≥L12 scope trains excellently in-window — the
amendments cost nothing there; (2) answer loss ALONE is satisfiable in-window
without gating, so the gate has no in-window incentive — and the out-window
collapse re-demonstrates the campaign's theorem on the campaign's own adapter.
**The pre-registered G1b trigger ("gate saturates open, bit ≈ 1 everywhere,
measured") is MET** — this run is the trigger's clean documentation, not a failure.
**G0c (148593, ~2h)** → `outputs/selfgate/g0c/`: L24h20 AUC ≥0.9998 under exists
AND majority at N∈{8,32,128} (+ per-task populations at L20/L24). The internal
selector is QUESTION-GENERAL before any gate training — H-G0's agnosticity clause
answered at the representation level.
**G1b = jobs 150418 (λ=1e-3, n308) + 150419 (λ=1e-2, n309), RUNNING in parallel**
(one change vs G1a: --l1-bits λ on soft.mean()). ETA ~3.5h each.

## [2026-09-16 ~02:10] G1b BOTH ARMS COMPLETE — saturated open at every λ; the G1 line closes as measured

Jobs 150418 (λ=1e-3) / 150419 (λ=1e-2), ~6h11 each → `outputs/selfgate/g1b_l*/`.
mean_bit 1.000 at every epoch in BOTH arms (fp counts identical to G1a's all-ones
gate); exam N=8 0.927 / 1.000, but N=16 COLLAPSED to 0.273 in both (vs G1a 1.000)
— the L1-perturbed soft train-mask creates a train/eval mismatch when the gate
never commits (trained under partially-dampened attention, evaluated all-open).
**Verdict (three runs, consistent, logged as measured per the pre-registered
protocol): the answer-loss-only ST-gate does NOT emerge from in-window training,
λ ∈ {0, 1e-3, 1e-2}. The incentive is structurally absent in-window (gating never
pays where the ungated read already suffices) — matching the G1a analysis. The
incentive-fix route (virtual-N pressure during training) is deferred to QGATE C3
(conditional). The deployable-agnostic-gate line continues via QGATE B1/B2/C2.**

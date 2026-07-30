# Agent brief — CARRIER TRUNCATION: one-token-per-frame inference ("frames end at L12")

**Date issued:** 2026-07-24. **Approved by:** Tal (via Claude session).
**Mission:** determine — decisively, with pre-registered bands — whether frame tokens can be
physically dropped after the fenced phase, so layers ≥L* and the entire decode run on
[question]+[N carriers]+[tail] only. If yes: ~150× shorter sequences on 16/28 layers,
frame-KV-free decode, near-linear-in-N prefill, N≫128 becomes reachable, and the thesis
gains a "compression as a corollary of capability repair" section that inverts the
VoCo-LLaMA/Victor comparison. If no: the truncation-layer sweep measures carrier-saturation
depth — a new mechanistic quantity. Either way, a result.

## 0. Read first
1. `CLAUDE.md` §3/§5/§7. Budget override (Tal, permanent): GPU-hours unconstrained.
2. `plans/p0p2_STATE.md` — the OTHER campaign (seeds/LOTO/P3/P4) is live. Coexist: never
   cancel its jobs, never take a partition's last free GPU, spread QOS around its slots.
3. Key architecture facts (verified in-session 2026-07-24): `make_masks` in
   `carrier_layer_lora.py:50` — the tail is given carriers explicitly (`mask_hi[fin_start:,
   ct]=0`) and mask_hi is mask_lo.clone()+carrier edges, so whether the tail can EVER see
   frame tokens depends on `build_block_mask`'s treatment of tail rows — VERIFY, do not
   assume. Carriers DO keep attending their own frame in layers ≥L* (mask_hi retains lo
   edges) — that is exactly the hypothesis under test.
4. Winner ckpt (use everywhere): `outputs/ladder/image_longN/carrier_fmt_caption/
   20260722_222032_L12_r8/carrier_layer_best.pt` (caption, L_OPEN=12). Eval dirs-files:
   the arm-A byte-identical set (`carrier_tally_l12v2/.../eval_dirs_*.txt`).
5. Known equivalence to exploit: fence + posreset ⇒ per-frame blocks are independent given
   the question (fence ≡ multipass identity, RESULTS [2026-07-17]) — the basis of E5's
   chunked prefill.

## Phase 0 — code truth (CPU, no jobs)
- P0.1 Read `build_block_mask` + the eval/decode path (`--eval-only`, `ext_mask` in
  `carrier_layer_cached.py`): establish exactly which columns tail rows may attend, in lo
  and hi, prefill and decode. Write the answer into the PREREG (it decides E1's expected
  outcome).
- P0.2 Position handling plan for physical truncation: blocks are position-reset; the
  truncated sequence must keep every surviving token's ORIGINAL position ids (pass explicit
  position_ids; never renumber). Write the plan before coding.
- P0.3 New flags, backward-compatible: `--drop-frame-kv` (prune frame rows from KV after
  prefill, decode-only change) and `--truncate-at L` (physically shorten the hidden states
  entering layer L to prefix+question+carriers[+tail]). Smoke-test mask/position plumbing
  on CPU shapes first.

## Phase 1 — exactness (E1, one GPU smoke, `2h_2g`)
- E1: `--drop-frame-kv` vs baseline on 20 samples spanning N=8/32/64: if P0.1 found tail
  never sees frames, outputs must be TOKEN-IDENTICAL (assert), and the run reports decode
  wall-clock + peak VRAM both arms. Band: identical → free-speedup GO, log the factor.
  NOT identical → a mask edge exists somewhere; find it (that is itself a logged finding —
  it would mean generation currently reads frames, revising the architecture story).
  Do NOT proceed to Phase 2 until E1's outcome is understood.

## Phase 2 — the truncation hypothesis (eval-only, the core)
PREREG bands in `plans/trunc_PREREG.md` BEFORE launching (per-cell): Δacc vs the logged
caption cells (in-dist 1.000 · N=32 0.987 · N=64 0.981):
  ≥ −0.02 everywhere = **TRUNCATION GO** (carriers saturated by L12; headline).
  −0.02 … −0.10 = PARTIAL → Phase 3 retrain is the fix under test.
  < −0.10 = carriers NOT saturated at L12 → the sweep (E3) becomes the deliverable.
- E2: `--truncate-at 12`, winner ckpt unchanged: exams on in-dist-150, N=32 (150 dirs),
  N=64 (52 dirs) — byte-identical dirs-files, same decode budgets as the fmt sweep.
- E3 (runs regardless of E2's band): truncation-layer sweep, eval-only, L_trunc ∈
  {12, 14, 16, 20, 24} on the N=32 cell → the carrier-saturation depth curve. Cheap: same
  ckpt, one flag varies. Plot acc vs L_trunc; the knee is a thesis figure.

## Phase 3 — repair (only if E2 partial/fail)
- E4: retrain the LoRA WITH truncation at the best L_trunc (upper layers never see frames
  — train/deploy matched). Natural fit for the cached trainer: cache h_{L*} for
  question+carriers+tail rows ONLY (cache shrinks ~100×; note the epoch-time change).
  Recipe otherwise = caption winner verbatim. Exams as E2. Band: within 0.01 of the
  non-truncated cells = GO-after-retrain.

## Phase 4 — the efficiency numbers (what the paper/thesis section quotes)
- E5 chunked prefill: implement layers 0..L*-1 as INDEPENDENT per-block short forwards
  ([question][frame][carrier], ~250 tokens each, batched) — mathematically equivalent
  under the fence (verify once against the dense-mask path on 5 samples, assert close),
  then hidden states concatenated for the truncated upper stack. This removes the dense
  seq² mask entirely.
- E6 benchmark table (the deliverable): baseline vs drop-frame-kv vs truncate vs
  truncate+chunked, at N=8/32/64/128: prefill time, decode tok/s, peak VRAM, KV bytes,
  min/sample end-to-end on one a100. Report honest caveats (dense-mask baseline; python
  overhead).
- E7 N-scaling push (GATED on E2/E4 GO + E5 working):
  (a) generate N=256 eval data (extend the longN generator; `4h_0g` CPU);
  (b) EXTERNAL tally at N=256 (supply scaling beyond 128 — cache + CPU fit);
  (c) **the pre-registered prediction test**: an in-length N=128-trained caption arm is
  now affordable — the readout error law (draft entry 2026-07-24) predicts **≈0.88**
  (p=0.002, mean N_ev≈64). Train with N=128 in the mix (truncated trainer), eval the
  N=128 held-out cell, and judge the prediction: |measured − 0.88| ≤ 0.05 = law holds at
  128. This single cell tests efficiency, capability, and the error law at once.

## AMENDMENT (2026-07-25, Tal-approved) — two arms the campaign is missing

Rationale (from the campaign's own measurements): the saturation curve says the per-frame
code consolidates at **~L20** (gate err 0.173@L16 → 0.0082@L20), but the E-H L* curve says
opening at 20 is the WORST arm for trained integration (N=32 zero-shot 0.273, pf 0.107 vs
L12's 0.443). **E4c sets l_open=20 AND truncate_at=20 — it confounds those two variables**,
so neither a pass nor a fail is attributable. Also: the E-H L20 cell used the OLD poslist
format, and its failure mode was parse-fail/format coherence — exactly what the caption
format fixed (pf 0.000 at every length). Verified prior art (2026-07-25): VoCo-LLaMA does
NOT truncate mid-stack — vision runs full-depth prefill and only GENERATION reads the
compressed KV; Victor truncates at layer 3/32 but fine-tunes the whole LM.

- **E4d — decode-only restriction (VoCo-shaped). RUN FIRST; this is the headline route.**
  Change ONLY the tail/decode mask: scratchpad+answer rows attend question + carriers +
  their own generated tokens, with frame columns masked out. **Frames stay full-depth**
  (carriers keep writing to saturation; L_OPEN stays 12 = the integration optimum). New
  flag, default off; smoke the mask at LIMIT=2 before real jobs. Then:
  (a) deploy-matched retrain, caption recipe verbatim, in-length data, fixed save criterion;
  (b) exams in-dist-150 / N=32 (150 dirs) / N=64 (52 dirs), byte-identical dirs-files;
  (c) with THAT ckpt, re-run the E1 exactness test — frame-KV-drop at decode must now be
      TOKEN-IDENTICAL (assert on 20 samples); then log the decode speedup (E1 reference:
      keep 103/12775 = 124× fewer tokens; N=64 657.1 → 6.6 s/sample).
  Bands: within 0.02 of the caption cells (1.000 / 0.987 / 0.981) = **GO — full accuracy
  with ~100× decode**, the compression-as-corollary headline; 0.02–0.10 below = partial,
  report per-count anatomy; >0.10 below = the tail genuinely needs frame tokens for
  per-slot addressing (a mechanistic finding — log it as the addressing-wall confirmation).
- **E4e — caption @ L_OPEN=20, NON-truncated (attribution control for E4c).**
  Caption format, L_OPEN=20, frames retained, no truncation, in-length data, recipe
  otherwise verbatim. Exams on the same three cells/dirs. Bands: within 0.02 of caption@L12
  = late opening is free under the winner format ⇒ any E4c failure is attributable to
  TRUNCATION, and the E-H inverted-U was partly a format artifact (log that finding and
  flag the L* curve for re-measurement at the winner format); ≥0.10 below caption@L12 =
  late opening itself is the cost ⇒ E4c is confounded, its truncation verdict is
  inconclusive, and E4d is the only viable compression route.
- Ordering: E4d before E4e. Neither may starve E4b/E4c exams or the p0p2 agent of QOS
  slots. Trainer walltime reference: E4c trained in 1h27 (352 s/ep).

## Logging & rules
- STATE: `plans/trunc_STATE.md` (overwrite-at-transition; live jobs + landed cells).
  Blockers → `plans/trunc_BLOCKED.md` + successor actions. Every cell → draft
  (`plans/carrier_stage2_DRAFT_RESULTS.md`) + `outputs/ladder/INDEX.md` + migration draft.
- Output layout: `outputs/ladder/image_longN/trunc_{kvdrop,at12,sweep,retrain,bench}/<ts>/`;
  smokes in `outputs/_scratch/`.
- SLURM: free-GPU check across ALL partitions before every submit; smokes `2h_2g`;
  eval-only exams `24h_1g`/`4d_1g`; retrain `24h_1g`-class (~14h measured for this
  recipe); spread QOS around the p0p2 agent's slots. `--export` comma gotcha (env files).
- Code: new flags only, never change default behavior of existing paths; SDPA backend
  lessons apply (EFFICIENT/FLASH lists, mask-dtype match); smoke every code change with
  LIMIT=2 before real jobs.
- Never: edit RESULTS.md; pip/conda install; delete outputs/data; cancel jobs you didn't
  launch. Every number traces to a run dir. Poll squeue every ~20-30 min.

## Success picture
One screen at the top of trunc_STATE.md: E1 exactness verdict + speedup factor · E2/E3
truncation verdict or saturation curve · E4 retrain cell (if run) · E6 benchmark table ·
E7 N=256 supply + the 0.88-prediction verdict. That screen is simultaneously a thesis
section, a defense against the compression papers, and the cost model for every future
experiment in this repo.

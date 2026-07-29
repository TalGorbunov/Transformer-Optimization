# Agent brief — P0–P2 consolidation campaign (post-framing priorities)

**Date issued:** 2026-07-23. **Approved by:** Tal (via Claude session).
**Mission:** collect everything already paid for, harden the headline cells with seeds and
missing baselines, then answer the two open questions (task-agnosticism, external validity).
After this campaign every number the thesis needs should exist.

## 0. Read first
1. `CLAUDE.md` §3/§5/§7. Budget override (Tal, permanent): GPU-hours unconstrained,
   etiquette unchanged.
2. `plans/scratchpad_format_STATE.md` + `plans/scratchpad_format_agent_brief.md` — the
   format sweep (arms B/C/D, jobs 125104-06 + evals). **If its agent is still active
   (STATE updated <2h ago or its jobs visibly being collected), do NOT touch its jobs —
   only read its results. If stale, ADOPT collection per its brief §5.**
3. `plans/scratchpad_loto_agent_brief.md` — the LOTO experiment (P2a below runs it).
4. `plans/carrier_stage4_STATE.md`, `plans/carrier_stage2_DRAFT_RESULTS.md`,
   `outputs/ladder/INDEX.md` — current numbers. Session context 2026-07-23: thesis scoped
   to core N≤16 + in-length N=32/48 headline; far-length extrapolation = limitation, not
   a campaign. Do not start N=128 work.

## P0 — collect (no new GPU jobs)
- **P0.1 Format sweep:** wait for / collect trainers 125104-06 and their eval ladder
  (per its brief §4-5): assemble the arm×cell table, log to draft + INDEX, name the
  **winning format + checkpoint** (ties → arm A/poslist stands). This gates P1.1, P2a, P2b.
- **P0.2 E-H L\* curve:** jobs 124965-972 COMPLETED, uncollected. Read their reports
  (`outputs/ladder/image_longN/`, logs `cl_eval-<id>.out`), assemble the 6-point L\* curve
  with L12 (0.443) / L17 (0.280) reference cells, log draft + INDEX. If some arm beats L12
  (zero-shot N=32, >0.05, pf≤0.02), flag it — it changes the recipe for P1.1/P2a.
- **P0.3 RESULTS.md migration prep:** compile the landed campaign results into
  `plans/results_migration_DRAFT.md`, formatted exactly as RESULTS.md entries, newest
  first, every number with its run dir. **DO NOT edit RESULTS.md itself — Tal logs it.**

## P1 — harden headline cells
- **P1.1 Seeds (GATED on P0.1/P0.2):** retrain the winning recipe ×2 seeds (`--seed 1/2`,
  everything else identical) + each seed's N=32 held-out exam (same dirs-file as the 0.953
  cell). Report mean±std across the 3 seeds. Trainers ~14h → `24h_1g`/`4d_1g` (12h_4g
  TIMES OUT — measured). Evals ~3.5h.
- **P1.2 Measured before-ceiling (CPU, start immediately):** ridge/logistic-on-sum probe on
  the EXISTING joint image caches (`outputs/ladder/image_longN/joint/N{8..128}/`, cache
  format of `probe_dprime_parity.py` — reuse its loaders; sample-disjoint split, 3 seeds).
  Output: measured best-linear-readout acc per N to replace the law-predicted curve in
  `outputs/_scratch/figs/pre_stage1_squashed_readout.png` (regen script in the session
  scratchpad; recreate under `outputs/_scratch/figs/` if inaccessible). `4h_0g` (mem ≤16G)
  or a `2h_2g` CPU-style slot if RAM demands.
- **P1.3 SFT baseline N=64 cell:** the E-B adapter IS saved (see stage4 STATE late
  additions; only the N=64 eval OOM'd). Rerun the N=64 generate-and-parse eval using the
  EFFICIENT-attention eval path (the lesson that unlocked N=128 on 40GB A100s —
  carrier_layer_lora.py EFF_SDPA pattern). One job, `24h_1g`. If the SFT eval script lacks
  the EFF path, port it (small); timebox 3h, else BLOCKED note.

## P2 — the two open questions
- **P2a LOTO (GATED on P0.1/P0.2):** execute `plans/scratchpad_loto_agent_brief.md` in
  full (its Phase 0 overlaps P0 here — dedupe, don't redo), with two amendments:
  (1) trainers on `24h_1g`/`4d_1g` (measured ~14h); (2) use the P0.1 winning format +
  P0.2 L\* in the recipe, noting any deviation from the logged l12v2 reference cells.
- **P2b MLVU-AC zero-shot (GATED on P0.1):** eval-only, winning ckpt on `data/mlvu_ac/`
  at the dense-sampling setting of RESULTS [2026-07-11c] (N=128@392px arm — if decode
  cost is prohibitive, the 32-frame arm with its evidence-delivery caveat quoted).
  Pre-register first in `plans/p0p2_STATE.md`: frozen refs 0.282 (32f) / 0.393 (dense);
  band: > frozen+0.05 on the same setting = transfer GO; ≤ frozen = domain gap measured,
  log honestly. MCQ mapping: emitted count → nearest option (state the rule before
  running). Carrier eval tooling may not consume MLVU format — timebox porting to 3h,
  else BLOCKED with successor note.

## Ordering & parallelism
Start immediately: P0.2, P0.3, P1.2, P1.3. Then P0.1 as the sweep lands → gates open for
P1.1, P2a, P2b (run in parallel, spread QOS: trainers `24h_1g`×2 + `4d_1g`, evals across
`24h_1g`/`4d_1g` remaining slots). Before EVERY submit: free-GPU check across ALL
partitions (CLAUDE.md §3 sinfo pattern). Never take a partition's last free GPU while the
format agent's jobs are queued.

## Logging & state
- Maintain `plans/p0p2_STATE.md` (overwrite-at-transition; list live jobs + landed cells).
- Every landed cell → draft entry (`plans/carrier_stage2_DRAFT_RESULTS.md`) + INDEX row +
  a line in `plans/results_migration_DRAFT.md`.
- Blockers → `plans/p0p2_BLOCKED.md` with successor actions; continue unblocked arms.
- Final deliverable: a one-screen summary at the top of `plans/p0p2_STATE.md` — seeds
  mean±std, completed baseline column, L\* curve, format winner, LOTO verdict vs its
  PREREG bands, MLVU cell — each with run dirs.

## P3 — external validity (APPENDED 2026-07-24, Tal-approved; start after P2a exams are
## launched — do not delay the seeds/LOTO checklists for this)

Motivation: P2b landed the pre-registered DOMAIN-GAP outcome (MLVU 0.107 ≤ frozen 0.282;
evidence detector domain-bound). P3 tests whether the method — not just the mechanism —
works off the synthetic park renders.

- **P3a Natural-images MMRED, full protocol @N≤16 (the headline of this tier).**
  Resolve the natural-images MMRED variant root first (RESULTS references the P4
  cross-domain item, transfer gap 0.43 vs 0.92 in-domain; find the actual data root +
  its labels/loaders — do not guess; if per-frame gold supports caption targets, the
  full ladder applies). Then, pre-registered per level BEFORE the trainer:
  1. Frozen baseline on natural @N=8/16 (eval-only).
  2. L1 supply: cache fenced-carrier vs joint messages, d′ + adequacy check; compare
     against park values (d′ 8-11).
  3. L2 scaffold: gate→tally on the cached messages (CPU).
  4. L3 in-model: WINNING recipe verbatim (caption format, L12, r8, distilled e_c,
     fixed save criterion) trained on natural roots @≤16, tested in-length on held-out
     natural @8/16. Zero-shot park→natural ref = the known 0.43.
  Bands (fix exact numbers in the prereg): L3 in-length ≥0.80 = cross-domain
  trainability GO; ≤ frozen+0.1 = NO-GO (log honestly); L1/L2 localize any failure
  (supply vs readout) — that decomposition IS the deliverable if L3 lands low.
  Trainer ~14h class → `24h_1g`/`4d_1g`.
- **P3b InternVL2.5-8B scaffold-level tally.** Check what the supply run cached
  (`outputs/frame_axis/internvl/multipass_qfirst/20260719_004112/`, job 124280). If
  per-frame messages are cached: CPU logistic gate→tally → exact acc @N=8 (band ≥0.90 =
  "the GNN structure ports", scaffold level). If not cached: ONE cache job via the
  InternVL runner (uses the /rg venv_arch environment — see memory/CLAUDE conventions;
  do not touch the main .venv), then the CPU fit. Label the claim honestly per what the
  cache is (multipass-isolated vs one-forward fenced).
- Log P3 rows into the one-screen summary + draft + INDEX + migration draft like all
  other cells.

## P4 — readout-simplicity controls (APPENDED 2026-07-24 evening, Tal-approved; runs
## alongside/after P3 — do not starve the seeds/LOTO/P3a checklists of QOS slots)

Motivation: with extrapolation de-scoped (in-length training accepted), the committee-grade
question is "was the scratchpad necessary, or just the carriers, or just in-length data?"
The existing SFT ladder (0.480/0.350/0.220) is trained ≤8 only — NOT matched to the
caption cells. Three cells close the branch. Pre-register all bands in
`plans/p0p2_STATE.md` (or a P4 PREREG file) BEFORE the trainers.

- **P4.1 Plain-LoRA SFT, in-length trained (the dangerous cell — run it first).**
  `lora_sft_baseline.py` config identical to `sft_control_le8_v2` EXCEPT the data mixture
  gains the same in-length long-N roots the caption trainer used (longN_16/32/64 +
  longN_park2 32/48; match counts to the caption mixture as closely as the script allows;
  document any mismatch). Save adapter (patch exists). EPOCHS small (best was ep1-3);
  long-seq training may need grad-ckpt/FLASH (lessons already in lora_sft_baseline.py).
  Exams: N=32 (arm-A `eval_dirs_N32all.txt`, LIMIT 150) + N=64 (52-dir file), generate-
  and-parse, per-count histograms MANDATORY (the dead-mid-range signature is the reading).
  Bands: ≥0.90 @N=32 = "simple fix wins" (log honestly — this reframes, not kills, the
  thesis); ≤0.50 with dead mid-range = theory-confirmed strongest-baseline row; between =
  partial re-encoding, report per-count anatomy.
- **P4.2 Carrier + DIGIT readout, in-length trained (the middle rung).** Caption-winner
  recipe/data verbatim but `--scratchpad-format` off / digit targets (the pre-scratchpad
  target builder) — carriers+fence for supply, plain digit emission, no decode cost.
  Same two exams + bands as P4.1. Distinguishes "scratchpad needed" from "carriers
  sufficient once data is in-length."
- **P4.3 No-harm on the plain SFT adapter (eval-only, cheap, run immediately).**
  `noharm_bench` harness, existing adapter `sft_control_le8_v2/20260720_191541_lora/`,
  same 500 MME + 500 POPE protocol as job 124508. Band ≤2 pts as before. Also log WHAT it
  emits on failure (digit-on-yes/no is the predicted failure mode). This cell is
  informative regardless of P4.1's outcome (always-on deployment story).
- Trainers `24h_1g`/`4d_1g` per measured walltimes; P4.3 fits `2h_2g`. Log everything to
  draft + INDEX + migration draft + the one-screen summary (add P4 rows).

## Hard rules
Never: edit RESULTS.md; pip/conda install; delete/modify `outputs*/`/`data/`; heavy
compute on login node; oversized QOS; cancel/resubmit jobs you didn't launch (the format
agent's jobs included). Every logged number traces to a run dir. `--export` comma gotcha:
multi-root DATA_ROOT via env file/wrapper. Poll `squeue -u $USER` every ~20-30 min.

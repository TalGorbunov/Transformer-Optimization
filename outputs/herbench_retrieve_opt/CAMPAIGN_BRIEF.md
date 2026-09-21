# HERBench retrieve-opt campaign — agent brief

> **You are executing a pre-approved campaign** (Tal, 2026-08-01). Purpose: determine
> whether the per-unit perception ceiling on HERBench action counting (measured d′ ≈ 1
> on single frames) can be lifted by giving the model ±δ temporal context around the
> oracle frames (clip-units) and/or higher resolution — WITHOUT training anything.
> This is a probe campaign: its deliverable is a measured VERDICT, not a trained model.
> Work phase by phase with gates. Read CLAUDE.md first; its rules apply except where
> this brief pre-authorizes. Keep `outputs/herbench_retrieve_opt/STATE.md` current
> after every phase — it is the handoff file.

## Background (read these before anything)

- `docs/proposal_background.md` §2 — HERBench (arXiv:2512.14870, HD-EPIC egocentric
  kitchen video; action-counting split has one annotated timestamp per occurrence).
  Frozen armA (evidence-only) EM = 4.9%, saturates at count ~2–3: the pure-aggregation
  exposition cell.
- `docs/archive/RESULTS_pre_fencing.md` (~lines 2360–2430) — the measured per-frame
  ceiling: carrier-level AUROC ≈ 0.80, whitened d′ ≈ 0.98–1.10 @448px single frames;
  look-again solo-frame detection 0.200 vs majority 0.217; verdict "per-frame perception
  ceiling is intrinsic". These are the ANCHORS: your single-frame baseline arm must
  land in this band or your pipeline is suspect.
- The decision bar: per-unit d′ ≥ ~2.5–3 (recall p ≥ ~0.9) is what leaves room for a
  meaningful exact-match after p^N compounding at N=8–16. Below that, the method's
  honest null on HERBench stands (and this campaign strengthens it: "we tried the
  perception levers"). Per-VERB verdicts matter as much as the pooled one — a
  transition-verb subset clearing the bar is a usable partial-scope result.

## Standing rules

- NEVER `pip install` into the shared `.venv`; this campaign should need no new deps —
  if something is missing, STOP AND ASK.
- `legacy/` is READ-ONLY. Port what you need (prep_ac_frames.py, lookagain_frames.py
  patterns) into new scripts; never edit or execute legacy in place without reading it.
- ALL new scripts go in `scripts/herbench_retrieve_opt/` (create it). Reuse
  `gnnformer/` as a library; do NOT modify gnnformer core — if you believe you must,
  STOP AND ASK. (If you only ADD a standalone module, run the full CPU `tests/` suite
  after.)
- Run dirs: `outputs/herbench_retrieve_opt/<name>/<YYYYMMDD_HHMMSS>_<jobid>/` with
  report.txt + ABOUT.md each; maintain `outputs/herbench_retrieve_opt/INDEX.md`.
- Do NOT write RESULTS.md (Tal logs explicitly). STATE.md is your log.
- SLURM: check free GPUs across all partitions before submitting
  (`sinfo -p l40s-shared,h200-shared,rtx6k-shared,a100-public,l40s-public -N -O "Partition:16,NodeHost:12,Gres:26,GresUsed:30,StateLong"`);
  probes → `2h_2g` (≤2 GPUs/user), CPU → `4h_0g` (mem ≤16G); no comma-lists in
  `--export`; DRY_RUN=1 any new wrapper once; OUTPUT dirs suffixed `_${SLURM_JOB_ID}`.
- Committing new files (scripts, wrappers, INDEX, this dir's docs) is pre-authorized;
  never push; never commit data.
- STOP AND ASK if: source video/frame material for ±δ is not on disk and fetching it
  exceeds 30 GB or needs credentials/licenses; any anchor reproduction fails; you
  need a trainer (this campaign trains NOTHING without an explicit go); disk > 60 GB.

## Phase 0 — inventory & feasibility (CPU only)

1. Map what exists: `data/herbench_ac/` (armA_evidence_only, armB_ev_fill16,
   manifest.json — read the manifest fully), any per-sample metadata (timestamps, verb
   classes, video ids, fps). Read `legacy/experiments/herbench/prep_ac_frames.py` to
   learn how frames were extracted and from WHERE (source videos? pre-extracted frame
   dirs? paths may point to /rg or scratch). Record everything in STATE.
2. **The critical question: can ±δ neighbor frames be obtained?** Either the source
   videos / dense frame dumps are reachable on disk, or they must be fetched
   (STOP-AND-ASK conditions above). Do not proceed to Phase 1 without a confirmed
   route to neighbor frames.
3. CPU gap analysis: from the occurrence timestamps, compute the inter-occurrence gap
   distribution per video and per verb → choose the SAFE δ grid (clips must not
   swallow neighboring occurrences; expect δ ∈ {0.5, 1, 2}s to be the candidate set,
   trimmed by the data). Also extract the verb-class distribution and per-class sample
   counts. Write both tables to STATE.
4. Extraction plan + disk estimate for the clip frames actually needed (positives at
   timestamps, negatives away from all timestamps incl. HARD negatives just outside
   occurrence windows). Estimate before extracting; then extract (CPU jobs).

## Phase 1 — the probe sweep (GPU, eval-only, THE deliverable)

Units = clips; per-unit binary gold (occurrence vs not). Build a supply-probe script
in `scripts/herbench_retrieve_opt/` reusing the gnnformer probe machinery
(`probe_supply.py` and `scripts/presentation_diagnostics/probe_*.py` are the
patterns): Q-first + fence + posreset layout over units, replica read at L16,
messages → held-out d′/AUROC + logistic gate. Arms:

| arm | unit | res | note |
|---|---|---|---|
| A0 | single frame @ timestamp | 448 | ANCHOR — must reproduce d′ ≈ 1 band |
| A1 | single frame | 672 | resolution axis alone |
| B-δ | clip of ±δ (3–5 frames) | 448 | temporal axis, δ from Phase-0 grid |
| C-δ | clip of ±δ | 672 | both axes |
| (opt) V | native-video clip input | 448/672 | only if the engine path supports it cleanly; skip rather than hack |

n per arm: as much as the data allows (target ≥100 units/class; report actual).
Negatives: matched count, half random-away, half hard (adjacent to occurrence
boundaries). Metrics per arm: pooled d′/AUROC + gate acc, AND per-verb-class d′
(report classes with n≥20 only). One or two 2h_2g jobs; batch arms inside a job
like slurm/posreset_sweep.sbatch does.

**Gate:** A0 inside the archived band (else stop — pipeline bug). Then verdict per
arm vs the d′ ≥ 2.5–3 bar, pooled and per verb.

## Phase 2 — verdict & report

- Figure 1: d′ vs δ, one line per resolution, A0/A1 as δ=0 points; bar at 2.5.
- Figure 2: per-verb d′ bars for the best arm, sorted, bar at 2.5 — the
  "which verbs the method can serve" chart.
- STATE verdict, one of: (a) GO — pooled clears the bar (name the config);
  (b) PARTIAL — a verb subset clears (name the subset + its share of AC questions);
  (c) NULL CONFIRMED — nothing clears; the honest-null chapter gains "perception
  levers tried: δ up to Xs, res up to 672, gains from/to".
- Update INDEX.md; leave RESULTS to Tal.

## Phase 3 — conditional stretch (ONLY on GO/PARTIAL, and ASK FIRST)

If a viable config exists: L2 gate→tally on mixed sequences (armB-style: evidence
clips + filler clips) at the winning config, and an in-domain carrier distillation
plan (costed, NOT launched). Present both to Tal in STATE and stop.

## Reporting

STATE.md after every phase: what ran (job IDs, run dirs), numbers vs bars/anchors,
decisions, next step. Final deliverable: the two figures + verdict + (if any) the
Phase-3 proposal.

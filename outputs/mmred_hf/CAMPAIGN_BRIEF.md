# MMReD faithful-benchmark campaign — agent brief

> **You are executing a pre-approved campaign.** Tal approved this plan on 2026-08-01.
> Work phase by phase; each phase has a GO/NO-GO gate — do not spend GPU past a failed
> gate, report instead. Run everything from the repo root inside tmux. Read CLAUDE.md
> first; its rules apply except where this brief explicitly pre-authorizes something.

## Goal

Run our method (fenced carriers + caption-scan scratchpad, the METHOD.md recipe) on the
ORIGINAL MMReD benchmark — HF dataset `ef1e43ce/mmred` + images rendered by the authors'
generator `github.com/Fr0do/mmred` (verified alive 2026-08-01) — using the benchmark's own
protocol: train on seq_len ≤ 16 splits, evaluate on all lengths to 128. Deliverable: a
three-arm comparison per task and task-averaged, per seq_len:
  (A) frozen model accuracy (`scripts/eval_frozen.py`)
  (B) external gate→tally on carrier states (the "GIN ceiling": dump + `scripts/gate_tally.py`)
  (C) deployed method accuracy (`scripts/eval_carrier.py`, caption format)
compared against the paper's published baselines where available.

## Standing rules (verbatim, non-negotiable)

- NEVER `pip install` into the shared `.venv`. PRE-AUTHORIZED instead: create a dedicated
  `venv` at `.venv_mmred/` (python3.9 -m venv) and install the upstream repo + `datasets`
  there. Only data-generation/rendering runs in it; all training/eval uses the shared `.venv`.
- NEVER edit `legacy/`, `outputs_*` trees, or any existing data. New data → `data/mmred_hf/`.
- New results → `outputs/mmred_hf/<subgroup>/<name>/<YYYYMMDD_HHMMSS>_<jobid>/` with
  report.txt + ABOUT.md per run dir; maintain `outputs/mmred_hf/INDEX.md` (git-tracked).
- Do NOT write to RESULTS.md — Tal logs explicitly. Keep a running campaign log at
  `outputs/mmred_hf/STATE.md` instead (update after every phase/gate; this is the handoff file).
- After ANY edit to `gnnformer/` core: run the CPU test suite (`python tests/test_*.py`) first.
- SLURM: check free GPUs across ALL partitions before each submit
  (`sinfo -p l40s-shared,h200-shared,rtx6k-shared,a100-public,l40s-public -N -O "Partition:16,NodeHost:12,Gres:26,GresUsed:30,StateLong"`);
  right-size QOS (CPU→4h_0g mem≤16G; short GPU→2h_2g, ≤2 GPUs/user; trainer→24h_1g);
  NEVER put comma-lists in `sbatch --export` values (use files); always DRY_RUN=1 a new
  wrapper once before real submit. Default OUTPUT dirs must include `_${SLURM_JOB_ID}`.
- Committing NEW files (adapter code, scripts, INDEX, plans) with clear messages is
  pre-authorized; never push, never commit data or outputs (except INDEX.md files).
- STOP AND ASK TAL if: rendering needs > 60 GB disk; any GO gate fails; the trainer fails
  twice; upstream code wants system packages; anything requires touching the shared venv;
  or their JSON schema can't be mapped cleanly to our sample interface.

## Phase 0 — acquisition, fidelity, triage (CPU only)

1. `git clone https://github.com/Fr0do/mmred data/mmred_hf/upstream_repo` (untracked).
   Record the commit hash in outputs/mmred_hf/STATE.md. Check LICENSE; note it.
2. Create `.venv_mmred/`, `pip install -e data/mmred_hf/upstream_repo` + `datasets` there.
3. Download `ef1e43ce/mmred` (Arrow, text-only) to `data/mmred_hf/hf/`. Enumerate configs
   (seq_len_1..128), the 24 question types, split sizes. Write the inventory to STATE.
4. Render a SMALL probe first (one config, ~20 samples) with their `scripts/` renderer;
   measure per-sample disk; extrapolate total for the needed configs
   (train: seq 2,4,8,16; eval: 8,16,32,64,128 — val/test splits only for eval lengths).
   If projected total > 60 GB → stop and ask (candidate overflow: /rg).
   Determinism check: render one batch twice, diff hashes — record result.
5. Render the needed configs (4h_0g CPU jobs; parallelize per config; each job sources
   `.venv_mmred` explicitly — do NOT rely on slurm/lib/common.sh's venv activation here).
6. **Task triage:** for each of the 24 question types, decide schema fit
   (per-unit fact expressible as a short token phrase + textual reduction).
   Expected: the 14 NIAH-style types fit as-is; of the 10 dense-LC types, keep those
   reducible to running per-room/per-char counters (+argmax at the end); declare the rest
   out-of-schema with one sentence each. Write the triage table to STATE. Target scope:
   14 + compliant LC subset.
7. **Adapter:** write `gnnformer/data.py::load_mmred_hf(sample)` (or a thin adapter module)
   mapping their JSON/rendered layout → (sample_id, frames[PIL], question, states, answer),
   plus per-task `probe_evidence` extensions where needed. Add a CPU parity test
   (`tests/test_mmred_hf_adapter.py`): fields present, evidence set matches gold answer on
   ≥50 samples per task. Run the full tests/ suite.
8. **Fidelity anchor:** frozen L0 eval (arm A) on seq_len_8, 2–3 tasks, limit 100, and
   compare against the paper's reported frozen-model numbers (fetch from the OpenReview
   PDF H6fM44DOHP). Within a plausible band (model/prompt differences allowed) → GO.
   Wildly off (e.g., near-zero where paper reports 0.4) → suspect the adapter; stop.

## Phase 1 — diagnostic ladder on their data (small GPU)

On 2 representative tasks (the steps-analog counting task + one more NIAH-style), seq 8:
1. L1 supply probe: `scripts/probe_supply.py` full A3 arms, limit 150, LAYERS=16, on their
   renders. GO bar: fenced d′ ≥ 4 AND ≥ 2× the joint anchor.
2. Renderer-transfer cell (free, informative): existing carrier
   `checkpoints/carrier_token_room_k1_best.pt` eval-only (`train_carrier_token --eval-only`)
   on their renders — measures the domain-bound-detection prediction. Record %-of-teacher.
3. L2: `gate_tally.py` on the L1 caches. GO bar: ≥ 0.9.
Both bars pass → Phase 2. Either fails → stop, write diagnosis to STATE, ask Tal.

## Phase 2 — carrier (only if needed)

If the transfer cell shows < ~70% of the in-domain teacher: re-distill e_c on their renders
(`train_carrier_token --objective distill`, n≈900, seq8, room-token anchor — the anchor
sweep validated it), 2h_2g or 24h_1g, ~2–4 h. Freeze the winner; symlink under checkpoints/
per the README convention. Otherwise reuse the existing carrier and say so in STATE.

## Phase 3 — scan formats + trainer

1. Write the gold-scan format spec per task family into `outputs/mmred_hf/formats.md`:
   caption full-scan running-tally (METHOD §2.5) for counting/NIAH; extended per-frame
   state scans with running per-room counters + final argmax line for the compliant LC
   tasks. One slot per unit, fixed order, explicit anchor + END.
2. Build the training mixture: their train splits, seq ≤ 16, all in-scope tasks; cap total
   ≈ 5–6k samples (the measured data curve's comfortable regime); gold scans generated
   from their JSON states (never from model output).
3. Train: `scripts/train_carrier_layer.py` caption recipe verbatim (L*=12, r8, frozen e_c,
   lexicographic save criterion), prefix cache ON. QOS 24h_1g. Expect 6–10 h.
   One retry on infra failure; two failures → stop and ask.

## Phase 4 — the eval grid

Arms A/B/C × in-scope tasks × seq_len {8,16,32,64,128} on their val (tuning) then test
(reported) splits. Limits: 100 (N≤32), 80 (N=64), 50 (N=128). Decode budget ≥ 6 tokens ×
N; greedy; parse on the anchor; per-count histograms + parse-fail rates in every report.
Arm B = `eval_carrier --dump-carrier-states` + `gate_tally.py` (L16 read).
Produce: per-task CSV, task-averaged curves per arm, and the headline figure (3 lines vs N,
train/eval boundary marked at 16). Compare vs paper baselines in a table.
Everything into `outputs/mmred_hf/` with INDEX rows.

## Phase 5 — safety cells

No-harm pair (`scripts/bench_noharm.py`, MME+POPE band ≤ 2 pts) and plain-prompt drift
with the new adapter always-on. Record in INDEX + STATE.

## Reporting

At every phase boundary update `outputs/mmred_hf/STATE.md` with: what ran (job IDs, run
dirs), the gate numbers vs bars, decisions taken, and what's next. Final deliverable:
STATE points to the grid CSV + figure + the comparison-vs-paper table. Do not summarize
into RESULTS.md — Tal does that.

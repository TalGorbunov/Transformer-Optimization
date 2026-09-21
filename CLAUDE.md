# CLAUDE.md — Project operating guide

> Read automatically at the start of every Claude Code session. Keep it **operational**: how to
> work in this repo and on this cluster. Research findings go in `RESULTS.md`, not here.
> **2026-09-21: the repo is being rewritten** (branch `rewrite`; plan in
> `~/.claude/plans/ok-i-want-to-nifty-brook.md`). `core/` is a skeleton Tal fills in himself —
> do not implement its bodies unasked.

---

## 1. What this project is

- **Thesis topic:** relieving the **aggregation (over-squashing) bottleneck** in a frozen
  vision-language transformer with GNN-style message passing (message passing ≈ attention).
- **Author:** Tal Gorbunov (MSc). Feb–Jul 2026 data/probes; paper target CVPR (Nov 2026).
- **Backbone:** `Qwen/Qwen2.5-VL-7B-Instruct`, always **frozen + 4-bit nf4**, bf16 compute, `sdpa`.
- **Benchmark:** the OFFICIAL MMReD (HF `ef1e43ce/mmred` + Fr0do/mmred renders, native 512 px,
  the paper's prompt verbatim). 24 question types, exact match. Train on seq_len ≤ 16, test on
  all lengths (32/64/128 = extrapolation). **Nothing else is a valid dataset for new work**;
  the custom park generators are frozen under `legacy/v1/datasets/`.
- **THE METHOD (current):** (1) **fence** — every frame in its own attention block with a
  per-block position reset, the question visible in the shared prefix (their `--prefix_question`
  order); (2) **gate** — a per-frame classifier on each block's `<|vision_end|>` state hides
  non-evidence blocks; (3) **read** — a small LoRA decodes the answer from prefix + kept blocks.
  Learned carriers, scratchpads, learnmask, gating adapters, superquery/ninv/recagg/treefold are
  RETIRED (all reproducible from `legacy/v1/`).

## 2. Environment setup

```bash
cd /home/tal.gorbunov/projects/Transformer-Optimization
source .venv/bin/activate          # Python 3.11: model stack + the official mmred package + core
```

- ONE environment. `.venv` = Python 3.11 (torch 2.8.0, transformers 4.57.6, bitsandbytes 0.48.2,
  peft 0.17.1, `mmred` editable from `data/mmred_hf/upstream_repo`, `core` editable).
  `.venv39` = the pre-freeze 3.9 interpreter, kept only until legacy anchors are re-verified.
- **Never `pip install` / `conda install` without asking first** (shared env).
- Model weights: `~/.cache/huggingface` (Qwen2.5-VL-7B 16 G; MME/POPE). `HF_HUB_OFFLINE=1` in jobs.

## 3. Cluster & SLURM

**Scheduler:** SLURM. No `--account` line (the default account works; an explicit
`--account=shocher_partition` is REJECTED on l40s-shared).

### Partitions

| Partition | GPUs | Notes |
|-----------|------|-------|
| `l40s-shared` | L40S 48 GB ×2 nodes | default (`*`); also the proven partition for CPU jobs (`--qos=4h_0g`) |
| `h200-shared` | H200 ~140 GB | biggest; usually busiest |
| `rtx6k-shared` | RTX6000 ~48 GB | overflow; **`--exclude=n317`** (silent stalls) |
| `a100-public` | A100 **40 GB** (not 80) | usually idle; 4-bit 7B fits |
| `l40s-public` | L40S | usually idle |

**ALWAYS check free GPUs across ALL partitions before submitting:**
```bash
sinfo -p l40s-shared,h200-shared,rtx6k-shared,a100-public,l40s-public \
  -N -O "Partition:16,NodeHost:12,Gres:26,GresUsed:30,StateLong"
```

### QOS rules (most cap at 3 running jobs/user; 4h_0g caps at 10 in queue)

| QOS | Wall | Max GPU | Jobs/user | Use for |
|-----|------|---------|-----------|---------|
| `4h_0g` | 4 h | 0 | 10 | CPU: data prep, copies, fits, plots (**mem cap ~16G**) |
| `2h_2g` | 2 h | 2 | 3 | smokes, short evals (per-user MEMORY cap — keep `--mem` modest) |
| `12h_4g` | 12 h | 4 | 3 | standard GPU runs |
| `24h_1g` | 24 h | 1 | 4 | long single-GPU; good overflow |
| `24h_4g` / `72h_8g` / `4d_1g` | — | — | 3/1/8 | long multi-GPU / huge / many-parallel-slots |
| `contrib` | 7 d | — | — | ask first |

**DefaultTime is 2 h on every partition** — long jobs need an explicit `--time`.

### Submitting

Wrappers live in [sbatch/](sbatch/) — env-var driven, one per `experiments/` entrypoint, all
source `sbatch/lib/common.sh` (repo-root cd, venv, `run_logged` tee into the run dir, `DRY_RUN=1`,
`stage_split` = per-split tarball → node-local NVMe `$TMPDIR`):
```bash
SLURM_SUBMIT_DIR=$PWD DRY_RUN=1 bash sbatch/<wrapper>.sbatch      # check the assembled command
sbatch -p a100-public --qos=24h_1g --time=12:00:00 --export=ALL,QTYPES=steps_in_room,... sbatch/<wrapper>.sbatch
```
**NEVER put comma-lists in `--export` values** — sbatch silently splits them; use files.

### Monitoring

```bash
squeue -u $USER
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,MaxRSS
tail -f logs/<jobname>-<jobid>.out      # run_logged also tees into the run dir
```

### Run conventions

- Every run gets a timestamped dir under `outputs/<group>/<name>/<YYYYMMDD_HHMMSS>*/` with
  `report.txt` (+ config/eval csvs). Smokes → `outputs/_scratch/`.
- **Every `outputs/<group>/` has `INDEX.md` (experiment → canonical run → headline) and, for
  campaigns, `CAMPAIGN_BRIEF.md` + `STATE.md`. All three are git-tracked** — update INDEX.md
  when a run becomes canonical.
- Canonical checkpoints get stable symlinks under [checkpoints/](checkpoints/README.md); the
  README table is the tracked record, the symlinks are local.
- Frozen trees (`outputs_legacy/`, custom `data/mmred_*`) live on `/rg` behind symlinks. Never
  delete; never move cited paths without leaving a symlink.

## 4. Repo layout

```
core/               THE method package (Tal writes it): constants · model · fence · mmred ·
                    prompt · metrics — each with a legacy twin in legacy/v1/gnnformer/ and a
                    test in tests/ that pins its contract
experiments/        one file per experiment (prepare_data, train, evaluate, probe_hahn,
                    probe_attention, gate_capture/gate_fit, baselines, figs/); every script
                    takes --qtypes
sbatch/             wrappers + lib/common.sh + migrate/ (the 2026-09-21 /rg migration)
tests/              CPU tests — run after ANY change to core/: python tests/test_<module>.py
checkpoints/        README.md (tracked) + local symlinks to canonical adapters
data/               symlinks: mmred_hf -> /rg/shocher_prj/lab_data/mmred (official benchmark);
                    everything else -> /rg/shocher_prj/tal.gorbunov/archive/data/ (legacy only)
outputs/            run dirs (untracked) + tracked INDEX/STATE/CAMPAIGN_BRIEF per group
docs/               paper plan, prior art, framing audits, theory/ (HTML), archive/ (incl.
                    METHOD_2026-07.md = the retired July method)
legacy/             pre-July-2026 code (top level) + v1/ (July–Sept 2026: gnnformer, scripts,
                    slurm, tests, datasets, ARCHIVE_MAP.md). FROZEN. Run: PYTHONPATH=legacy/v1
RESULTS.md          append-only research log     CLAUDE.md   this file
```

## 5. How I want you to work in this repo

- **Read before you write.** Read the relevant files (and `RESULTS.md`) and tell me your plan first.
- **Show the plan before spending GPU hours** (what, which partition/QOS, expected cost). Wait for OK.
- **Right-size experiments** — small `--limit`, few epochs first; scale up only to confirm.
- **One change at a time when debugging.**
- **Anchors are law:** each script's docstring names the logged numbers it must reproduce.
  After touching `core/`, run `tests/` (CPU, seconds) before anything else.
- **Prompt fidelity:** the system prompt and question text are the paper's, byte for byte
  (`tests/test_prompt.py`); layouts only move the words. Native resolution; no resize code.
- **Be honest about uncertainty**; flag anything that looks like leakage/contamination
  (train ≤ 16 only; long-N counting cells reported split by answer ≤ 16 / > 16).
- **Verifiability:** any number that might reach the thesis traces to a real run dir.
- **Updating RESULTS.md is an explicit step** — append-only, newest last, only when I say "log this".
- **legacy/ is read-only.** Port with a parity test; never edit.

## 6. Long unattended sessions

- For submit → poll → collect loops, run inside `tmux` (`tmux new -s thesis`).

## 7. Things to never do

- Never `pip`/`conda install` into the shared env without asking.
- Never delete or edit `legacy/`, `outputs_*`/`output_*` trees, or any data (moves leave symlinks).
- Never submit with unconfirmed SLURM flags or an oversized QOS.
- Never run heavy compute on the login node (copies go through `sbatch/migrate/`).
- Never put a metric in `RESULTS.md` that isn't backed by a real run.
- Never let a comma-list ride in `sbatch --export` (silent truncation).
- Never train on seq_len > 16 or on non-official data for a headline row.

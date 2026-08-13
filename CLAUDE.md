# CLAUDE.md — Project operating guide

> This file is read automatically at the start of every Claude Code session.
> Keep it **operational**: how to work in this repo and on this cluster.
> Research findings do NOT go here — they go in `RESULTS.md`.

---

## 1. What this project is

- **Thesis topic:** Identifying and relieving the **aggregation (over-squashing) bottleneck** in
  vision-language transformers, using ideas from graph neural networks (message passing ≈ attention).
- **Author:** Tal Gorbunov (MSc student). Feb–Jul 2026.
- **Framework:** PyTorch + HuggingFace `transformers` (+ `peft` for the SFT baseline; `nnsight`
  only in legacy code). Model: `Qwen/Qwen2.5-VL-7B-Instruct`, always **frozen + 4-bit (nf4)**,
  bf16 compute, `sdpa` attention — loaded via [gnnformer/runtime.py](gnnformer/runtime.py)
  (7B is the default; the 32B is opt-in — the old silent-32B footgun is fixed).
- **Task (MMRED):** self-generated frame datasets (characters in rooms); "how many frames was
  character C in room R?", answer ∈ 0..N. Headline metric: exact-match accuracy.
- **THE METHOD (current, see METHOD.md):** three stacked repairs on the frozen backbone —
  (1) **fencing**: per-frame carrier tokens behind a block-diagonal attention fence +
  per-block M-RoPE position reset (one-forward isolated supply); (2) **learned carriers**:
  distilled carrier embedding e_c + LoRA on layers ≥ L\*=12 (in-model aggregation);
  (3) **caption-scan scratchpad readout** (the answer is a tally read-off).
  The gLSTM/DeepSets/frame-axis approaches are RETIRED (see docs/archive/).
- The "All for One" paper (AF1 last-token aggregation) is the methodological ancestor; its
  repo clone was removed 2026-07-30 (re-clone from github.com/siddarth-pm/all-for-one if needed).

## 2. Environment setup

```bash
cd /home/tal.gorbunov/projects/Transformer-Optimization
source .venv/bin/activate          # slurm/lib/common.sh does this automatically
```

- **Python:** 3.9. Deps pinned in [pyproject.toml](pyproject.toml) (nnsight = the `[legacy]` extra).
- **Never run `pip install` / `conda install` without asking me first** — shared env.
- A second venv for non-Qwen models lives at `.venv_arch -> /rg/...` (untracked symlink).

## 3. Cluster & SLURM

**Scheduler:** SLURM. **Account:** `shocher_partition`.

### Partitions

| Partition | GPUs | Notes |
|-----------|------|-------|
| `l40s-shared` | L40S 48 GB ×2 nodes | default (`*`) |
| `h200-shared` | H200 ~140 GB | biggest; usually busiest; needed for in-length SFT training |
| `rtx6k-shared` | RTX6000 ~48 GB | often least loaded; overflow (one past NODE_FAIL on n318) |
| `a100-public` | A100 **40 GB** (not 80!) | usually idle; 4-bit 7B fits fine |
| `l40s-public` | L40S | usually idle |

**ALWAYS check free GPUs across ALL partitions before submitting** — don't queue behind a full
partition while a public one idles:
```bash
sinfo -p l40s-shared,h200-shared,rtx6k-shared,a100-public,l40s-public \
  -N -O "Partition:16,NodeHost:12,Gres:26,GresUsed:30,StateLong"
```
Override partition/QOS at submit: `sbatch -p <partition> --qos=<qos> ...` (beats the `#SBATCH` lines).

### QOS rules (right-size; most cap at 3 running jobs/user — spread big batches across QOS)

| QOS | Wall | Max GPU | Jobs/user | Use for |
|-----|------|---------|-----------|---------|
| `4h_0g` | 4 h | 0 | 10 | CPU: gate_tally, datagen, plotting (**mem cap ~16G**) |
| `2h_2g` | 2 h | 2 | 3 | smokes, short evals (**per-user MEMORY cap — keep `--mem` modest**) |
| `12h_4g` | 12 h | 4 | 3 | standard GPU runs |
| `24h_1g` | 24 h | 1 | 4 | long single-GPU (trainers ~14 h); good overflow |
| `24h_4g` / `72h_8g` / `4d_1g` | — | — | 3/1/8 | long multi-GPU / huge / many-parallel-slots |
| `contrib` | 7 d | — | — | ask me first |

### Submitting

Wrappers live in [slurm/](slurm/) — env-var driven, one per entrypoint, all source
`slurm/lib/common.sh` (repo-root cd, venv, `run_logged` tee into the run dir, `DRY_RUN=1`):
```bash
# check the assembled command without running anything:
SLURM_SUBMIT_DIR=$PWD DRY_RUN=1 bash slurm/eval_carrier.sbatch

# real submit (knobs via --export; NEVER put comma-lists in --export values — sbatch
# silently splits them; use files like slurm/lib/roots_inlength.txt instead):
sbatch -p a100-public --qos=24h_1g \
  --export=ALL,CKPT=checkpoints/carrier_layer_fmt_caption_best.pt,DIRS_FILE=<dirs.txt>,LIMIT=150,DEC=320,OUTPUT=outputs/carrier/exam \
  slurm/eval_carrier.sbatch
```

### Monitoring

```bash
squeue -u $USER
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,MaxRSS
tail -f logs/<jobname>-<jobid>.out      # run_logged also tees into the run dir
```

### Run conventions

- Every run gets a timestamped dir under `outputs/<group>/<name>/<YYYYMMDD_HHMMSS>*/` with
  `report.txt` (+ config/eval csvs). Smokes/throwaways → `outputs/_scratch/`.
- **Every `outputs/<group>/` has a hand-maintained `INDEX.md`** (experiment → canonical run →
  headline number). INDEX.md files are **git-tracked** — update them when a run becomes canonical.
- Canonical checkpoints get stable symlinks under [checkpoints/](checkpoints/README.md).
- Old trees (`outputs_*`, `output_*_old`) are frozen archives; `ARCHIVE_MAP.md` maps the
  RESULTS.md citations. Never delete; never move cited paths.

## 4. Repo layout

```
gnnformer/          the method package: runtime, fencing (THE mask, one copy), carriers,
                    engine (all forwards/decodes), data, scratchpad, metrics, constants
scripts/            thin entrypoints: probe_supply, train_carrier_token, train_carrier_layer,
                    eval_carrier (exam + TRUNC instruments), eval_frozen, gate_tally,
                    train_sft_baseline, bench_noharm, e2e_pipeline
slurm/              sbatch wrappers + lib/common.sh + lib/roots_inlength.txt
tests/              CPU tests — run them after ANY change to gnnformer/ core modules:
                    python tests/test_fencing.py etc. (mask parity vs legacy is bit-for-bit)
checkpoints/        stable symlinks to canonical ckpts/caches + README table
datasets/mmred/     MMRED generators (incl. the recovered park generators)
data/               generated datasets (untracked; per-root metadata.json is the record)
outputs/            live run dirs (only INDEX.md files tracked)
legacy/             ENTIRE pre-refactor tree, frozen — see legacy/README.md for how to run it;
                    NEVER edit; benchmark preps (mlvu/herbench/vnbench/internvl) still live here
docs/               citations, prior art, proposal, theory/ (HTML explainers),
                    archive/ (pre-fencing RESULTS, STORY, campaign files)
RESULTS.md          append-only live research log     METHOD.md   the method recipe
ARCHIVE_MAP.md      old cited output path → real location
```

## 5. How I want you to work in this repo

- **Read before you write.** Read the relevant files (and `RESULTS.md`) and tell me your plan first.
- **Show the plan before spending GPU hours** (what, which partition/QOS, expected cost). Wait for OK.
- **Right-size experiments** — small `--limit`, few epochs first; scale up only to confirm.
- **One change at a time when debugging.**
- **Anchors are law:** each script's docstring names the logged numbers it must reproduce.
  After touching `gnnformer/` core, run `tests/` (CPU, seconds) before anything else.
- **Be honest about uncertainty**; flag anything that looks like leakage/contamination.
- **Verifiability:** any number that might reach the thesis traces to a real run dir.
- **Updating RESULTS.md is an explicit step** — append-only, newest last, only when I say "log this".
- **legacy/ is read-only.** If a legacy behavior is needed, port it (with a parity test), don't edit it.

## 6. Long unattended sessions

- For submit → poll → collect loops, run inside `tmux` (`tmux new -s thesis`).

## 7. Things to never do

- Never `pip`/`conda install` into the shared env without asking.
- Never delete or edit `legacy/`, `outputs_*`/`output_*` trees, or any data.
- Never submit with unconfirmed SLURM flags or an oversized QOS.
- Never run heavy compute on the login node.
- Never put a metric in `RESULTS.md` that isn't backed by a real run.
- Never let a comma-list ride in `sbatch --export` (silent truncation).

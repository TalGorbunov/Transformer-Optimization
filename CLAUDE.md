# CLAUDE.md — Project operating guide

> This file is read automatically at the start of every Claude Code session.
> Keep it **operational**: how to work in this repo and on this cluster.
> Research findings do NOT go here — they go in `RESULTS.md`.

---

## 1. What this project is

- **Thesis topic:** Identifying and relieving the **aggregation (over-squashing) bottleneck** in
  vision-language transformers, using ideas from graph neural networks (message passing ≈ attention).
- **Author:** Tal Gorbunov (MSc student). Worked here ~Feb–Jun 2026 (much of it with Codex).
- **Framework:** PyTorch + HuggingFace `transformers` + `nnsight` (activation hooks/patching).
- **Model under study:** `Qwen/Qwen2.5-VL-7B-Instruct` (main), with `Qwen2.5-VL-32B-Instruct`
  for heavier mechanistic characterization and `3B` referenced in older runs. Always loaded
  **frozen + 4-bit (nf4)**, bf16 compute, `sdpa` attention. See [models/model.py](models/model.py).
- **Task (MMRED):** a self-generated dataset of *frames* showing characters in rooms. Question:
  *"how many frames was character C in room R?"* Answer ∈ {0..8}. The headline metric is exact-match
  answer accuracy (see `RESULTS.md` for the precise definition and what we've found).
- **Core finding driving the work:** information flows **frame tokens → "carrier" question tokens →
  last token**. The current leading approach is a **gLSTM-style memory adapter** that sums each
  frame's attention "messages" and injects them back into the frozen Qwen residual stream.
- **`all-for-one/`** is a *separate, self-contained* sub-repo: the "All for One" mental-math paper
  (AF1 last-token aggregation in LLMs). It is the methodological ancestor of this work
  (`evaluations/helpers/af1_utils.py` reuses its ideas). Don't confuse it with the MMRED pipeline.

## 2. Environment setup

```bash
cd /home/tal.gorbunov/projects/Transformer-Optimization
source .venv/bin/activate          # runners do this automatically if .venv exists
```

- **Python:** 3.9 (compiled artifacts are `cpython-39`).
- **Key deps:** `torch`, `transformers`, `nnsight`, `datasets`, `pillow`, `torchvision`,
  `num2words`, `huggingface-hub>=0.34`, `matplotlib`, plus `bitsandbytes` (4-bit quant).
  Installed into `.venv`. Package itself is `pip install -e .` (see [pyproject.toml](pyproject.toml)).
- **Never run `pip install` / `conda install` without asking me first** — it can break the shared env.

## 3. Cluster & SLURM

**Scheduler:** SLURM. **Account/association:** `shocher_partition`. Confirmed working values below
(verified live via `sinfo` / `sacctmgr`).

### Partitions (the 3 shared ones I use)

| Partition | GPUs | Nodes | Notes |
|-----------|------|-------|-------|
| `l40s-shared` | NVIDIA L40S (48 GB) | 2 | **Default** (the `*` partition); what every runner currently hardcodes |
| `h200-shared` | NVIDIA H200 (~140 GB) | 1 | Biggest GPUs; usually the **busiest** |
| `rtx6k-shared` | NVIDIA RTX 6000 (~48 GB) | 2 | Often the least loaded |

Also available but not normally used: `a100-public`, `l40s-public`, `l40s-benisty`, `h200-dds`, `rtx6k-shocher`.

**Launch to the least-loaded partition.** Before submitting, check load and pick the emptiest:
```bash
sinfo -p l40s-shared,h200-shared,rtx6k-shared -o "%P %t %C %G"   # %C = CPUs A/I/O/T per node
squeue -p l40s-shared,h200-shared,rtx6k-shared -o "%P %t %u" | sort | uniq -c
```
Override a runner's hardcoded partition at submit time with `sbatch -p <partition>` (it wins over the
`#SBATCH -p` line). A 4-bit 7B job fits comfortably on any of L40S / RTX6000 / H200 / A100.

**ALWAYS check for free nodes across ALL partitions before submitting — never let a job sit blocked
when another partition has idle GPUs.** The 3 "shared" partitions are often 8/8 full while the
`*-public` ones (esp. `a100-public`, `l40s-public`) sit nearly idle. Check actual free GPUs (not just
CPU load) and submit where there's room:
```bash
# free GPUs per node = GRES total minus GRES_USED; pick a node with spare GPUs
sinfo -p l40s-shared,h200-shared,rtx6k-shared,a100-public,l40s-public \
  -N -O "Partition:16,NodeHost:12,Gres:26,GresUsed:30,StateLong"
```
A100s are 80 GB so a 4-bit 7B fits trivially. `a100-public`/`l40s-public` use the same `.venv` and
account; just pass `-p a100-public` (avoid `rtx6k-*` — its `.venv` python symlink is broken). If your
job shows `PENDING (Priority/Resources)` and a public partition has free GPUs, cancel and resubmit
there rather than waiting.

### QOS rules (IMPORTANT — read before submitting)

QOS names encode `<walltime>_<max_gpus>g`. **Right-size the QOS to the job** — don't use a bigger one
than needed. **Max 3 running jobs per QOS** for most of them, so if I'm launching many jobs at once,
**spread them across different QOS** to get past the per-QOS cap.

| QOS | Wall | Max GPU | Max jobs / user | Use for |
|-----|------|---------|-----------------|---------|
| `4h_0g` | 4 h | 0 (CPU only) | 10 | dataset generation, plotting, CPU post-processing |
| `2h_2g` | 2 h | 2 | 3 | small/smoke GPU runs (≤2 h) |
| `12h_4g` | 12 h | 4 | 3 | **standard training run** (what all current runners use) |
| `24h_1g` | 24 h | 1 (single GPU) | 4 | long single-GPU run; good overflow when `12h_4g` is full |
| `24h_4g` | 24 h | 4 | 3 | long multi-GPU run |
| `72h_8g` | 72 h | 8 | 1 | only for genuinely huge jobs |
| `4d_1g` | 4 d | 1 | 8 | very long single-GPU; lots of parallel slots |
| `contrib` | 7 d | — | — | contributor priority; ask me before using |

Set QOS with `sbatch --qos=<qos>`. Rule of thumb: a 2-hour script → `2h_2g`; a normal 12 h training
job → `12h_4g`; need >3 jobs at once → push extras to `24h_1g` (4 slots) or `4d_1g` (8 slots).

### Submitting a job

Runners live in [runners/](runners/) and are **env-var driven** (sane defaults, override at submit):
```bash
# Standard pattern: submit a runner, override knobs via --export, pick partition/qos explicitly
sbatch -p l40s-shared --qos=12h_4g \
  --export=ALL,SEQ_LEN=8,EPOCHS=3,LR=1e-4,OUTPUT_DIR=$PWD/outputs/my_run \
  runners/run_glstm_memory_adapter_7b_seq8.sbatch

# Many runners also support DRY_RUN=1 (assemble command, don't execute) for a safety check:
sbatch --export=ALL,DRY_RUN=1 runners/run_message_memory_carrier_update_seq8_7b.sbatch
```
Common override vars: `MODEL_NAME`, `DATASET_ROOT`, `OUTPUT_DIR`/`OUTPUT_ROOT`, `SEQ_LEN`, `LAYERS`,
`EPOCHS`, `LR`, `D_MEM`, `LOAD_IN_4BIT`, `NO_PLOTS`, `RUN_SMOKE`/`RUN_FULL`. Some experiment scripts
can also be run directly, e.g. `python -u experiments/<subject>/<name>.py --run-all`.

Runners auto: locate `REPO_ROOT`, `source .venv/bin/activate`, set `PYTHONUNBUFFERED=1` and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `mkdir` the output dir, tee a `runner-<jobid>.log`,
and copy the SLURM log into the output dir on exit.

### Monitoring jobs

```bash
squeue -u $USER                                              # my queued/running jobs
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,MaxRSS # finished-job accounting
tail -f logs/<jobname>-<jobid>.out                           # live SLURM log (also copied into the run dir)
```

### Run conventions

- Each run writes its own output dir, timestamped: `RUN_NAME=$(date +%Y%m%d_%H%M%S)`.
- Output trees are named like `outputs_<codename>/` (e.g. `outputs_kitkat`, `outputs_oreo`,
  `outputs_no_train`) and `output_*_old/` for archived earlier work. Inside each, one subdir per
  experiment family, then timestamped run dirs.
- **Per-run files (the common structure):** `config.json`/`run_config.json` (exact config),
  `summary.md`/`summary.csv`/`eval_metrics.csv`/`accuracy_by_*.csv`/`metrics*.csv` (the headline
  metric + by-evidence-count breakdown), `train_history.csv`, `README.md` (auto-generated
  interpretation / "Readout" — **read this first** to see what a run concluded), `diagnostics.json`
  (smoke sanity flags like `hooks_ok`, `nonzero_gates`), `run_done.json` (completion marker),
  `checkpoints/*.pt` / `*_adapter.pt`, `plots/*.png`.
- **Output dir naming rule:** a run's output tree must be named after the Python script that
  produced it: `outputs/<script_basename_without_.py>/<YYYYMMDD_HHMMSS>/` (e.g.
  `evidence_only_sum_evidence_adapter_seq1_8_7b.py` → `outputs/evidence_only_sum_evidence_adapter_seq1_8_7b/20260612_161616/`).
  Keep the timestamped run subdirs. Variant/config suffixes go on the timestamped subdir
  (e.g. `20260608_101718_carrier_glstm_layerwise`), never on the root. This keeps every metric
  traceable to the exact script that produced it.

## 4. Repo layout

```
models/model.py            Qwen2.5-VL loading (4-bit nf4, sdpa), ModelRuntime
evaluations/
  helpers/                 af1_utils, patching_core, sdpa_attention, plots, utils  (shared infra)
  experiments/mmred/       MMRED task definitions
  scripts/                 eval scripts (accuracy mosaics, image-size sweeps, af1, attention, patching)
experiments/               ALL experiment entrypoints, one subject per subdir (a real package —
                           import as experiments.<subject>.<module>):
  evidence_only/           evidence-only adapters (sum / layer-local / all-question-to-last)
  distractor/              distractor-task adapters (oracle-mask sum, supervised gated sum)
  oracle_bounds/           oracle count injection / translator upper-bound probes + diagnostics
  carrier_probes/          message-memory, count-direction, codebook, injection-sweep probes
  carrier_mixing/          pna/pnamix LoRA carrier mixing, visual_fixed8 sweeps, frame sigmoid-sum
  glstm/                   gLSTM memory adapters (layerwise, mechanism ablation, final comparison,
                           gated token mixer, native factorized attention)
runners/                   sbatch + .sh submit wrappers (env-var driven); runners/archive/ = retired
data/                      MMRED dataset variants (mmred_images_park = main; *_evidence_only_seq1_8,
                           *_no_step_marker, *_perm_bias, *_corrupted* = ablation variants)
datasets/                  raw/source MMRED assets
outputs_*/ , output_*_old/ experiment outputs (one dir per run); see RESULTS.md for what's in them
logs/                      SLURM .out logs (also copied into each run dir)
all-for-one/               SEPARATE sub-repo: "All for One" mental-math paper (methodological ancestor)
RESULTS.md                 research progress log — read this for context on what's been tried
```

## 5. How I want you to work in this repo

- **Read before you write.** Read the relevant files (and `RESULTS.md`) and tell me your plan first.
- **Show the plan before spending GPU hours.** For anything that submits jobs or launches training,
  describe what you'll run, on which partition/QOS, and the expected cost. Wait for my OK.
- **Pick the least-loaded partition and the smallest sufficient QOS** (see §3). Spread many jobs
  across QOS to beat the 3-jobs-per-QOS cap. Always check free GPUs across *all* partitions
  (incl. `a100-public`/`l40s-public`) before submitting so jobs don't sit blocked.
- **Right-size experiments — enough to show a direction, not exhaustive.** Don't run 3000-sample
  sweeps when a few hundred (or fewer) settle the question. Prefer small `--limit`/`per-count`/`per-label`,
  fewer epochs, and `--no-plots`/`--no-probes` for quick reads; scale up only once a direction looks
  worth confirming. Cheap, fast, decisive beats big and slow.
- **One change at a time when debugging.** Don't bundle a config change, a code change, and a new run.
- **Be honest about uncertainty.** If a result looks too good or a config looks off, say so. Catching
  data leakage / wrong normalization / train-eval overlap is more valuable than agreeing with me.
  (Several "near-100%" rows in `RESULTS.md` are trained-on-clean or oracle-masked — flag, don't launder.)
- **Verifiability.** Any number that might end up in my thesis must trace back to a real run in a real
  output dir. No remembered or estimated metrics.
- **Updating RESULTS.md is an explicit step.** When I say "log this", append to `RESULTS.md` using its
  format. Don't silently edit it mid-task.

## 6. Long unattended sessions

- For submit → poll `squeue` → collect loops, run inside `tmux` so it survives disconnects:
  ```bash
  tmux new -s thesis    # run `claude` inside; detach Ctrl-b d; reattach: tmux attach -t thesis
  ```
- You act while a session is live and tasked — not a background daemon. If waiting on my approval, wait.

## 7. Things to never do

- Never `pip`/`conda install` into the shared env without asking.
- Never delete `outputs_*/` / `output_*/` directories or any data.
- Never submit jobs with unconfirmed SLURM flags, or a QOS larger than the job needs.
- Never run heavy compute on the login node.
- Never put a metric in `RESULTS.md` that isn't backed by a real run.

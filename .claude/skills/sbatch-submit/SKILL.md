---
name: sbatch-submit
description: >-
  Submit, dry-run, verify and monitor SLURM jobs on the lab cluster for this repo: choosing
  partition / QOS / --time / --mem, the sbatch/ wrapper and --export conventions, the post-submit
  command-line check, and diagnosing jobs that pend, time out, OOM or stall. Use it whenever the user
  wants to run, launch, submit, queue, kick off, smoke-test, train, evaluate or re-run anything on a
  GPU or "on the cluster", mentions sbatch / slurm / squeue / sacct / QOS / partition / walltime /
  node, asks why a job is pending or died, or wants a new sbatch wrapper written. Also load it when
  proposing any GPU experiment, because the plan Tal OKs must name partition, QOS, walltime and cost.
---

# sbatch-submit: running jobs on the lab SLURM cluster

Everything in this repo that touches a GPU goes through `sbatch` with a wrapper from `sbatch/`.
This skill is the submit loop that avoids the failures the project has already paid for: a
forgotten `--time` killed the first full trainer at 2 h, a comma in `--export` ran three campaigns
on partial inputs, and one node silently ate 16 GPU-hours. Run the loop end to end. The verify
step is the only cheap moment to catch a wrong command line.

## The loop

1. **Plan, then wait for OK.** Before a GPU submission, say what runs, on which partition + QOS,
   how long (`--time`), how much memory, and the expected GPU-hours. Tal OKs it (CLAUDE.md §5).
   Right-size first: a `--limit` smoke on `2h_2g` before the real run. CPU jobs on `4h_0g` need no OK.
2. **Preflight.** `bash .claude/skills/sbatch-submit/scripts/preflight.sh` prints free GPUs per
   node, your queue and a per-QOS table CAP / RUN / PEND / FREE. Pick a partition with free GPUs
   and a QOS with a FREE slot whose walltime cap fits. **For a batch, count first**: never queue
   more jobs on one QOS than its FREE column (the caps are per user, running + pending); spread
   single-GPU jobs over `12h_4g` → `24h_1g` → `24h_4g` → `4d_1g` → `72h_8g` (all accept 1-GPU jobs),
   or chain several cells inside one job (`sbatch/diag_d2.sbatch` / `diag_d3.sbatch` are the
   pattern). Total single-GPU concurrency is about 19 jobs; a bigger batch waits no matter what.
3. **Dry-run the wrapper.** `SLURM_SUBMIT_DIR=$PWD DRY_RUN=1 bash sbatch/<wrapper>.sbatch` prints
   the assembled command without running it. Read the flags, paths, `--limit` and output dir.
4. **Submit.**
   ```bash
   sbatch -p <partition> --qos=<qos> --time=HH:MM:SS [--exclude=n317] \
     --export=ALL,SPLIT=seq_len_8_train,LIMIT=50,QTYPES_FILE=path/to/list.txt \
     sbatch/<wrapper>.sbatch
   ```
   Always pass `--time`. Never a comma inside an `--export` value. Note the job id.
5. **Verify.** `bash .claude/skills/sbatch-submit/scripts/verify_submit.sh <jobid> "--limit 50" "--epochs 5"`
   waits for the log and checks that each quoted substring appears in the executed command line and
   that DRY_RUN did not leak in. A MISSING line means the job is running the wrong thing: `scancel` it.
6. **Monitor.** `squeue -u $USER`; `sacct -j <id> --format=JobID,JobName,State,Elapsed,MaxRSS,NodeList`;
   `tail -f logs/<jobname>-<id>.out` (run_logged also tees into `<run dir>/runner-<id>.log`).
   No output for over an hour after model load means check the node before extending anything.
7. **Collect.** Results live in `outputs/<group>/<name>/<YYYYMMDD_HHMMSS>*/report.txt`. Smokes go to
   `outputs/_scratch/`. When a run becomes canonical, update `outputs/<group>/INDEX.md`. RESULTS.md
   is appended only when Tal says "log this".

## Choosing partition, QOS, time, memory

| Partition | GPUs | Notes |
|-----------|------|-------|
| `l40s-shared` | L40S 48 GB, 2 nodes | default; also the proven partition for CPU jobs |
| `l40s-public` | L40S 48 GB | usually idle |
| `a100-public` | A100 **40 GB** (not 80), 9 nodes | usually idle; the 4-bit 7B fits |
| `h200-shared` | H200 140 GB, 1 node | biggest; usually busiest |
| `rtx6k-shared` | RTX6000 48 GB, 2 nodes | overflow; **`--exclude=n317`** (silent stalls) |

| QOS | Wall | GPU cap | Jobs/user | Per-job caps | Use for |
|-----|------|---------|-----------|--------------|---------|
| `4h_0g` | 4 h | 0 | 10 queued | cpu ≤ 8, mem ≤ 16G | data prep, tarballs, copies, fits, plots |
| `2h_2g` | 2 h | 2 per user total | 3 | user-wide mem 240G | smokes, short evals |
| `12h_4g` | 12 h | 4 | 3 | | standard GPU runs |
| `24h_1g` | 24 h | 1 | 4 | cpu ≤ 32, mem ≤ 275G | long single-GPU; first overflow |
| `4d_1g` | 4 d | 1 | 8 | cpu ≤ 32, mem ≤ 275G | many parallel single-GPU slots |
| `24h_4g` / `72h_8g` | 24 h / 72 h | 4 / 8 | 3 / 1 | | multi-GPU **or** overflow for 1-GPU jobs (verified 2026-09-22) |
| `contrib` | 7 d | | | | ask Tal first |

Decision guide:
- **CPU job:** `-p l40s-shared --qos=4h_0g --time=04:00:00` with `--mem` ≤ 16G and
  `--cpus-per-task` ≤ 8. sbatch rejects anything above the per-job caps.
- **GPU smoke under 2 h:** `--qos=2h_2g`. The cap is 2 GPUs total per user, so a third 1-GPU smoke
  pends on QOSMaxGRESPerUser even with idle nodes. Overflow to `24h_1g`.
- **Standard run under 12 h:** `--qos=12h_4g --time=<needed>`.
- **Long single-GPU (trainers, N=128 ladders):** `--qos=24h_1g` (4 slots), then `4d_1g` (8 slots),
  then `24h_4g` (3) and `72h_8g` (1) — the multi-GPU QOS take 1-GPU jobs and are usually empty.
- **A pending job stuck on a cap** (reason `QOSMaxJobsPerUserLimit` / `QOSMaxGRESPerUser`) is moved,
  not resubmitted: `scontrol update JobId=<id> QOS=<qos>` (+ `TimeLimit=HH:MM:SS` when the new QOS
  wall is shorter than the request, + `Partition=<p>` to change partition). Works on PENDING jobs
  only; a running job keeps its QOS. Reasons `Resources` / `Priority` mean the GPUs themselves are
  busy: nothing to fix on our side except a different partition.
- **N=128 at 512 px (≈ 41.8k tokens):** a masked (fenced/gated) forward needs h200 (`--mem 96G`,
  the [S,S] mask alone is 7 GB fp32 + 3.5 GB bf16); mask-free arms (paper / question-first layouts,
  `model.generate`) run on the 40 GB A100s. Split the arms across jobs so the h200's single free
  GPU is not a serial bottleneck for cells that do not need it.
- **Partition:** the 4-bit 7B fits every GPU here including the 40 GB A100s. Prefer whatever
  preflight shows idle. Listing several (`-p a100-public,l40s-public,rtx6k-shared`) starts fastest
  on a busy day. Long single-shot runs prefer l40s/a100/h200 over rtx6k because a NODE_FAIL loses a
  run whose summary is written only at the end.
- **Memory:** 48G covers the 4-bit 7B trainer and evals. Ask for more only with a reason, since
  `2h_2g` also has a per-user memory cap and a fat smoke blocks the second slot.

## Rules and the incident behind each

- **`--time` on every GPU submit.** Every partition's DefaultTime is 2 h and a QOS only caps walltime,
  never raises it. Job 127802, the first full trainer, hit TIMEOUT at 2:00:18 mid-epoch.
- **No comma inside any `--export` value, `EXTRA` included.** sbatch splits on commas with no escape:
  `EXTRA="--tasks count,exists"` arrives as `--tasks count` and the job "succeeds" on partial
  inputs. Lists go in a file (`*_FILE=path`) or space-separated; the wrapper joins them. Struck
  three times on 2026-07-10 and twice on 2026-09-21.
- **No `--wrap`, no `--account`.** The cli_filter plugin rejects `--wrap`, and an explicit
  `--account=shocher_partition` is rejected on l40s-shared. Submit wrapper files under the default account.
- **`--exclude=n317` on rtx6k-shared.** It hangs GPU jobs silently after model load (2026-09-15,
  about 16 GPU-hours lost across three ladders). n318 is fine.
- **Heavy work never on the login node.** Copies and tarballs go through `sbatch/migrate/`.
- **Headline training data is official MMReD at seq_len ≤ 16 only.** Longer splits are test-only.
- **Check `DRY_RUN` is not exported in your shell** before a real submit; `--export=ALL` carries it
  into the job and run_logged turns the job into a no-op.

## Failure decoder

| Symptom | Cause | Do |
|---|---|---|
| `TIMEOUT` with Elapsed ≈ 2:00:xx | no `--time` | resubmit with `--time` |
| pending, reason `QOSMaxGRESPerUser` or `QOSMaxJobsPerUserLimit` | QOS slot cap (per user, running + pending) | `scontrol update JobId=<id> QOS=<qos with FREE>0>` (preflight table); next time count slots before the batch |
| pending, reason `Resources` or `Priority` | the GPUs are busy, not a cap | wait, or `scontrol update JobId=<id> Partition=<p>` to a partition with free GPUs |
| `RuntimeError: No CUDA GPUs are available` within a minute of start, sibling jobs on the same node fine | node GRES hiccup (n314, 2026-09-22, two jobs) | resubmit (a different partition is safest); nothing to debug in the code |
| CUDA OOM "Tried to allocate 40–50 GiB" inside `scaled_dot_product_attention` on a mask-free long prompt | `sdpa_kernel([EFFICIENT, MATH])` wrapped a forward that has no 4-D mask, so MATH materialised the full [H,S,S] score matrix | restrict the backend context to masked forwards; mask-free forwards use the default (FLASH) path — fixed in probe_hahn / probe_attention / gate_capture 2026-09-22 |
| `cli_filter plugin terminated with error` at submit | `--wrap` or `--account` | wrapper file, no account line |
| rejected on `4h_0g` (`QOSMaxMemoryPerJob` / cpu) | `--mem` > 16G or cpus > 8 | lower it |
| CUDA OOM "total capacity 39.49 GiB" | a100-public is 40 GB | fine for 4-bit 7B; bigger models go to h200 |
| log silent for > 1 h after model load, no traceback | node stall (n317) | `sacct … NodeList`; scancel; resubmit with `--exclude` |
| executed cmd shorter than intended, flags missing | comma in `--export` | scancel; move the list to a `*_FILE` |
| `[DRY_RUN] not executing` in a real job's log | `DRY_RUN=1` exported in the shell | `unset DRY_RUN`; resubmit |
| `NODE_FAIL` mid-run | node blip | resubmit; nothing to debug |
| pending, reason `ReqNodeNotAvail, May be reserved or down` / `Nodes required for job are DOWN` | the partition's nodes are draining (`sinfo -N -O NodeHost,StateLong,Reason` shows e.g. `admin_maint`; l40s 2026-09-23) | `scontrol update JobId=<id> Partition=<other>`; submit with a partition list (`-p a100-public,l40s-shared`) so the scheduler picks whatever is up |
| `MISMATCH` in copy_tree / tar_split logs | partial copy | resubmit the same SRC/DST (idempotent) |
| completed job, degenerate decodes | eval-mode / checkpoint mismatch, not hardware | spot-check decode samples early |

## Writing a new wrapper

Copy `.claude/skills/sbatch-submit/assets/wrapper_template.sbatch` to `sbatch/<entrypoint>.sbatch`,
one wrapper per `experiments/` file with the same name. The conventions the template encodes:

- The `#SBATCH` header carries a sane default partition, QOS, **`--time`** and memory. The CLI
  overrides all of them, so the header is the safety net, not the plan.
- `source "${SLURM_SUBMIT_DIR:-.}/sbatch/lib/common.sh"` comes first: repo-root cd, `.venv`,
  `HF_HUB_OFFLINE=1`, and the two helpers. sbatch copies the script elsewhere, so every path is
  relative to `$SLURM_SUBMIT_DIR`, never to `$0`.
- Every knob is an env var with a default. Lists come from `*_FILE` files, never comma strings.
- `stage_split <split>` before the python call when the job reads frames: it unpacks the per-split
  tarball onto the node's NVMe and exports `DATA_ROOT`. `STAGE=0` skips it.
- `OUTPUT` defaults to a timestamped dir under `outputs/<group>/<name>/`, and
  `run_logged "$OUTPUT" python -u …` gives `DRY_RUN=1` support plus the tee'd log.
- The docstring lists the knobs, one example submit line, and the anchor numbers the script must
  reproduce. Flag names must match the entrypoint's argparse; check with `--help` before the dry-run.

Then dry-run it, run a `--limit` smoke, then the real thing.

## Batch submissions (the 2026-09-22 incident)

A 17-job wave queued 4 jobs on `12h_4g` (cap 3) and 5 on `24h_1g` (cap 4), and the N=128 wave
put five jobs behind the h200's one free GPU. The rules above encode the fix: read the FREE column
before a batch, spread across QOS by remaining slots, chain cells inside one job when the batch is
bigger than ~19, move stuck pending jobs with `scontrol update`, and send only masked N=128 forwards
to the h200. `sbatch/diag_submit_tier0.sh` is an example of a wave script that records every job id
in `outputs/<group>/jobs.tsv` so a monitor can follow them.

## Provenance

The tables were re-verified on 2026-09-21 with `sacctmgr show qos` and `scontrol show partition`.
sinfo also lists `b200-shared`, `rtx6k-shocher` and `h200-dds`; nothing from this repo has run on
them, so ask Tal before using one. When a cap looks wrong, re-run
`sacctmgr show qos format=Name,MaxWall,MaxTRESPU%30,MaxTRESPJ%30,MaxJobsPU` instead of trusting the table.

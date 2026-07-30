# GNN-Transformer (`gnnformer`)

**Relieving the aggregation (over-squashing) bottleneck in frozen vision-language
transformers with GNN-style message passing** — MSc thesis code, Tal Gorbunov, 2026.

A frozen VLM (Qwen2.5-VL-7B, 4-bit) collapses on multi-frame counting: joint attention
dilutes per-frame evidence like an over-squashed GNN (a measured d′/√N law), and the
single-token readout cannot use what does arrive. This repo implements and evaluates a
three-part repair, all on the frozen backbone (~2M trainable parameters total):

1. **Fencing (supply).** Per-frame carrier tokens behind a block-diagonal attention
   fence with per-block M-RoPE position reset — every frame is read as if in isolation,
   in ONE forward. Supply d′: 5.95 (joint) → **13.54** ([RESULTS.md](RESULTS.md) E1).
2. **Learned carriers (aggregation).** A distilled carrier embedding `e_c` + a small
   LoRA above the separator layer L\*=12 integrates the per-frame messages in-model.
3. **Scratchpad readout (expressivity).** The answer is decoded as a caption-format
   frame scan with an inline tally — a read-off, not a squashed single token.

**Headline** (MMRED counting, exact match; frozen baseline 0.219 @N=8):
in-distribution **0.999** · held-out N=32 **0.987** (3 seeds: 0.982 ± 0.007) ·
N=64 in-length **0.981**, parse-fail 0 · no-harm on MME/POPE (|Δ| ≤ 1.4 pts) ·
exact cached decode **16–311×** faster. Full record + honesty flags: [RESULTS.md](RESULTS.md).

## Repo map

```
gnnformer/        the method as a small package
  fencing.py        block-fence masks + position reset + hooks + message recompute
  carriers.py       lo/hi masks, LoRA, checkpoint I/O        engine.py   all forwards/decodes
  data.py           MMRED loading + task parsing             runtime.py  model loading (7B, 4-bit, sdpa)
  scratchpad.py     readout targets + parser                 metrics.py  d' estimator, √N law
scripts/          one entrypoint per concern (probe, trainers, exam, baselines, pipeline)
slurm/            sbatch wrappers + lib/common.sh (env-var driven; DRY_RUN=1 supported)
tests/            CPU tests: mask parity vs legacy (bit-for-bit), invariants, round-trips
checkpoints/      stable names for canonical checkpoints/caches (symlinks into outputs/)
datasets/mmred/   MMRED dataset generators (park renders + corruptions)
data/             generated datasets (untracked; see the generators + per-root metadata)
outputs/          run dirs (untracked except INDEX.md per group)
legacy/           the ENTIRE pre-refactor code world, frozen (see legacy/README.md)
docs/             method/theory docs, citations, archives (incl. the pre-fencing RESULTS)
RESULTS.md        append-only research log (live era)      METHOD.md  the method recipe
ARCHIVE_MAP.md    old cited output path -> real location
```

## Quickstart

```bash
cd <repo-root> && source .venv/bin/activate     # python 3.9; deps pinned in pyproject.toml

# CPU tests (mask parity vs frozen legacy, scratchpad round-trips, d' estimator):
python tests/test_fencing.py && python tests/test_carrier_masks.py

# Supply probe (A3: blockfence + posreset + qfirst), N=8:
python scripts/probe_supply.py --question-first --fence-frames --fence-blocks \
    --reset-positions --limit 300 --shuffle-dirs 0 --output outputs/carrier/probe

# The exam: caption-winner checkpoint on the pinned N=32 held-out dirs (anchor: 0.987):
python scripts/eval_carrier.py --ckpt checkpoints/carrier_layer_fmt_caption_best.pt \
    --dirs-file checkpoints/carrier_tally_l12v2_run/eval_dirs_N32all.txt \
    --limit 150 --decode-tokens 320 --output outputs/carrier/exam

# Train the production carrier layer (caption recipe; ~14 h on one A100):
sbatch slurm/train_carrier_layer.sbatch          # knobs via --export, see the wrapper
```

On the cluster, submit through `slurm/` (partitions/QOS rules: [CLAUDE.md](CLAUDE.md) §3).

## Reproducibility contract

- Every number in [RESULTS.md](RESULTS.md) traces to a run dir on disk; canonical
  checkpoints have stable names under [checkpoints/](checkpoints/README.md).
- Each script's docstring names the logged anchor it must reproduce; `tests/` pin the
  mask/position/target semantics bit-for-bit against the frozen legacy implementations.
- `legacy/` is the complete pre-refactor tree and still runs
  (`PYTHONPATH=legacy python legacy/experiments/...`) — never edited, never deleted.
- `all-for-one/` is an unrelated nested sub-repo (the AF1 paper — methodological ancestor).

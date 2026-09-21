# legacy/ — frozen snapshot of the pre-refactor repo (2026-07-29)

This directory is the **entire old code world**, moved here wholesale on branch
`serious-refactor` with its internal layout intact:

```
legacy/
├── experiments/    all experiment entrypoints (glstm/ = the pre-port thesis core,
│                   carrier_probes/, carrier_mixing/, distractor/, evidence_only/,
│                   oracle_bounds/, readout/, pipeline/, natural/, arch_battery/,
│                   herbench/, mlvu/, vnbench/, internvl/)
├── evaluations/    helpers (af1_utils, patching_core, sdpa_attention, utils) + eval scripts
│                   (incl. scripts/patch_importence/ — the old de-facto model runtime "gri")
├── models/         model.py (old ModelRuntime; DEFAULT_MODEL_ID is the 32B — known footgun)
├── runners/        all 254 sbatch/sh submit wrappers, every campaign era
├── scripts/        pyc-only residue of the Jun-10 deletion (sources live in experiments/)
└── scratch_*.py    root one-off analyses (Jun 2026)
```

## Rules

- **Frozen.** Never edit, "fix", or deduplicate anything here — these files are the
  reproducibility record behind `RESULTS.md`. New work happens in the `gnnformer/`
  package + `scripts/` at the repo root.
- **Never delete.** Same policy as `outputs_*/`.

## Running legacy code

Cross-imports (`from experiments.… import …`, `import evaluations.…`) resolve against
this directory, while data/output paths in the scripts are relative to the repo root —
so run from the **repo root** with `legacy/` on `PYTHONPATH`:

```bash
cd /home/tal.gorbunov/projects/Transformer-Optimization
PYTHONPATH=legacy python -u legacy/experiments/glstm/<script>.py …
```

Notes:
- The `.venv` editable install still maps the old top-level `models`/`evaluations`
  paths; `PYTHONPATH=legacy` shadows that correctly.
- Legacy runners (`legacy/runners/*.sbatch`) still reference pre-move paths
  (`experiments/...`); prepend `legacy/` and set `PYTHONPATH` if you ever need to
  resubmit one. The live method has ported wrappers under `slurm/` instead.
- Some scripts compute the repo root from `__file__` (`parents[2]`-style); under
  `legacy/` that resolves to `legacy/` itself — pass explicit `--output`/data args
  if you rerun such a script (or read it first).

## Where things went

- Path translation for **outputs** cited in RESULTS.md: see `../ARCHIVE_MAP.md`.
- Code paths cited in RESULTS.md/STORY.md (`experiments/…`, `evaluations/…`,
  `models/…`, `runners/…`) → same path prefixed with `legacy/`.
- Clean ports of the live method (fencing, carriers, readout, baselines):
  `../gnnformer/` + `../scripts/` (see the root README for the mapping).
- Canonical checkpoints: `../checkpoints/` (stable symlinks).

---

# legacy/v1/ — frozen snapshot of the July–September 2026 code (2026-09-21)

The second freeze. `v1/` holds, unchanged, the `gnnformer` package + `scripts/` + `slurm/` +
`tests/` + `datasets/` + `ARCHIVE_MAP.md` that produced every RESULTS.md entry from
2026-07-30 to 2026-09-21 (fencing, learned carriers, scratchpad, gating, learnmask, superquery,
ninv, recagg, treefold, loramech, sparse, softgate, redux, fixedk, qgate, selfgate). The July
method statement is at `../docs/archive/METHOD_2026-07.md`.

```
legacy/v1/
├── gnnformer/    the package (runtime, fencing, carriers, engine, data, scratchpad, metrics,
│                 mmred_hf, scan_grammar, learnmask, gating, constants)
├── scripts/      every entrypoint, campaign subdirs mirror outputs/<group>/
├── slurm/        the sbatch wrappers + lib/common.sh + lib/roots_*.txt
├── tests/        the v1 CPU tests (mask parity vs the pre-July code, etc.)
├── datasets/     the 7 custom MMRED generators (park renders, balanced pools, corruptions)
└── ARCHIVE_MAP.md
```

## Rules

Same as above: **frozen, never edit, never delete.** New work happens in `core/` +
`experiments/` + `sbatch/` at the repo root, which work ONLY on the official MMReD benchmark.

## Running v1 code

Scripts locate the package via `Path(__file__).parents[k]`, which now resolves to `legacy/v1`,
and use `data/...` / `outputs/...` paths relative to the repo root — so run from the **repo root**
with `legacy/v1` on `PYTHONPATH`:

```bash
cd /home/tal.gorbunov/projects/Transformer-Optimization
PYTHONPATH=legacy/v1 python legacy/v1/scripts/sparse/train_sft_gated.py ...
PYTHONPATH=legacy/v1 python legacy/v1/tests/test_fencing.py          # CPU parity tests
```

- Environment: the pinned model stack is identical in the new 3.11 `.venv`; the exact
  pre-freeze interpreter is `.venv39` (Python 3.9) if a number must be reproduced to the bit.
- The v1 wrappers under `legacy/v1/slurm/` reference `scripts/...` and `slurm/lib/common.sh`
  relative to the repo root; prepend `legacy/v1/` and set `PYTHONPATH` if you resubmit one.
- Custom `data/mmred_*` roots the v1 code reads live in `/rg/shocher_prj/tal.gorbunov/archive/data/`
  (symlinked from `data/` after the 2026-09-21 migration); the official benchmark is in
  `/rg/shocher_prj/lab_data/mmred` (`data/mmred_hf`).
- `outputs_legacy/` moved to `/rg/shocher_prj/tal.gorbunov/archive/outputs/outputs_legacy`
  (symlink left in place); `checkpoints/` links keep resolving.

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

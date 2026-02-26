# Transformer-Optimization

This repo's main analysis entrypoint is `compute_entropy.py`.

## `compute_entropy.py`
- Loads MMRed clean and corrupted samples.
- Runs clean, corrupted, and patched forward passes.
- Computes `LD = logit(a*) - logit(a^-)` per sample/frame/layer.
- Computes per-layer importance entropy from those LD values.
- Writes results and a cache so interrupted jobs can resume.

## Inputs
- `--data_root`: clean samples directory
- `--corrupted_data_root`: corrupted samples directory
- `--limit`: target number of accepted samples
- `--min_clean_ld`: minimum clean LD required
- `--min_corrupted_diff`: minimum `(clean_ld - corrupted_ld)` required for patched runs
- `--output`: output directory
- `--computed_lds_dir`: optional cache directory to reuse

## Outputs
In `--output`:
- `sample_metrics.txt`: per-sample layer metrics
- `computed_lds.txt`: cached clean/corrupted/patched LD results
- `entropy_summary.png`: layer entropy summary plot
- `run-<jobid>.log`: execution log (when run via `scripts/main.sh`)

## Typical run
Use the SLURM launcher:

```bash
sbatch scripts/main.sh
```

The script activates `.venv`, sets seq length / thresholds, and runs:

```bash
python compute_entropy.py ...
```

# Transformer-Optimization

This repo's main analysis entrypoint is `compute_entropy.py`.

## `compute_entropy.py`
- Loads MMRed clean samples.
- Extracts evidence frames from each sample's `question` and `states`.
- Computes `LD = logit(a*) - logit(a^)` on clean runs.
- Zeroes evidence-frame token spans per layer and computes per-layer importances.
- Computes per-layer importance entropy and writes summary artifacts.

## Inputs
- `--data_root`: clean samples directory
- `--limit`: target number of accepted samples
- `--min_clean_ld`: minimum clean LD required
- `--output`: output directory

## Outputs
In `--output`:
- `sample_metrics.txt`: per-sample layer metrics
- `entropy_summary.png`: layer entropy summary plot
- `run-<jobid>.log`: execution log (when run via `scripts/compute_entropy.sh`)

## Typical run
Use the SLURM launcher:

```bash
sbatch scripts/compute_entropy.sh
```

The script activates `.venv`, sets seq length / thresholds, and runs:

```bash
python compute_entropy.py ...
```

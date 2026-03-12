# Transformer-Optimization

Main analysis entrypoint: `compute_entropy.py`.

## `compute_entropy.py` 
- Loads clean MMRed samples from `--data_root`.
- Detects evidence frames from the question/state traces (pattern: `How many steps did <Name> spend in the <Room>`).
- Scores full answer sequences (not single-step logits):
  - `a*`: ground-truth numeric answer
  - `a^`: best competing numeric answer from `0..num_frames`, excluding `a*`
  - `LD_clean = score(a*) - score(a^)`
- Skips samples with `< 2` evidence frames or `LD_clean < lambda`.
- For each layer, patches corrupted evidence-frame activations into the clean run and computes:
  - `LD_corrupted`
  - `r_i(l) = max(LD_clean - LD_corrupted_i(l), 0)`
  - normalized entropy over evidence-frame importances.

## Required input layout
- Clean root (`--data_root`): directories that contain `qa.txt` and frame images `000.png`, `001.png`, ...
- Corrupted root (`--corrupted_root`): for each clean sample id and each evidence frame index, expects:
  - `<corrupted_root>/<sample_id>/corrupted_frame_<frame_idx>/`
  - each corrupted frame directory must also be a valid sample directory (`qa.txt` + images).

If `--corrupted_root` is omitted, it is inferred from `--data_root` by replacing `mmred_images`/`mmred` with `mmred_corrupted`.

## CLI
```bash
python -u compute_entropy.py \
  --data_root data/mmred_images_generated/seq_len_16/all \
  --corrupted_root data/mmred_corrupted_generated/seq_len_16/all \
  --limit 10 \
  --lambda 1 \
  --batch_size 8 \
  --output output/seq_len_16/lambda_1_generated_mmred \
  --clean_ld_cache_dir output/seq_len_16/lambda_1_generated_mmred
```

Arguments:
- `--data_root` (required): clean sample root.
- `--corrupted_root` (optional): corrupted sample root.
- `--limit` (default `1`): number of accepted samples to process.
- `--output` (default `outputs`): output directory.
- `--clean_ld_cache_dir` (optional): directory where `clean_lds.json` is read/written. Defaults to `--output`.
- `--batch_size` (default `8`): number of evidence-frame corruptions evaluated per forward pass chunk.
- `--lambda` (optional): clean LD threshold.
- `--min_clean_ld` (optional): alias for `--lambda` (backward compatibility). If both are given, values must match.

If neither `--lambda` nor `--min_clean_ld` is provided, threshold defaults to `0.0`.

## Outputs
Written under `--output`:
- `sample_metrics.txt`: per-sample metrics including:
  - clean scores (`clean_ld`, `clean_answer_score`, `clean_competing_score`)
  - selected answers/tokens (`a_star_*`, `a_hat_*`)
  - per-layer arrays (`r`, `p`, `H_norm`, `R_total`)
- `clean_lds.json`: cache of `sample_id -> clean_ld` used to skip low-LD samples early on reruns.
- `entropy_summary[_seq_len_X].png`: mean/median layer entropy with bootstrap CIs.
- `total_importance_summary[_seq_len_X].png`: mean total importance per layer with bootstrap CI.
- `layer_invalidity_rate[_seq_len_X].png`: fraction of samples where a layer had zero total importance.
- `run-<jobid>.log`: when using `scripts/compute_entropy.sh`.

## Typical run (SLURM)
```bash
sbatch scripts/compute_entropy.sh
```

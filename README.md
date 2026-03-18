# Transformer-Optimization

Main analysis entrypoint: `evaluations/frames_entropy.py`.

## `evaluations/text_group_importance.py`
- Loads clean MMRed image-text samples from `--data_root`.
- Keeps only samples where the clean model:
  - ranks the correct numeric answer as top-1 among valid answers `0..num_frames`
  - and assigns the correct answer probability at least `--min_clean_correct_prob`
- Builds a matched control prompt with:
  - the same chat template
  - the same number of frames
  - changed question semantics, preferring character/room replacements with matching tokenized span lengths
- For each layer, patches matched-control hidden states into one non-image token group at a time and scores the full correct answer sequence with teacher forcing.
- Computes:
  - `score_clean`: full-answer log-prob of the correct answer
  - `score_corrupt(group, layer)`: same score after matched-control patching
  - `importance(group, layer) = max(0, score_clean - score_corrupt(group, layer))`

Token groups:
- `character`
- `room`
- `question_operator`
- `question_relation`
- `question_marker`
- `answer_marker`
- `question_punct`
- `instruction_context`
- `instruction_output_rule`
- `assistant_prefix`

Patchability rules:
- clean/control prompt sequence lengths must match
- clean/control group token counts must match exactly
- mismatched groups are skipped per sample with logged reasons

CLI example:
```bash
python -u evaluations/text_group_importance.py \
  --data_root data/mmred_images_generated/seq_len_16/all \
  --limit 10 \
  --batch_size 8 \
  --min_clean_correct_prob 0.4 \
  --include_groups character,room,question_operator,question_relation,question_marker,answer_marker,question_punct,instruction_context,instruction_output_rule,assistant_prefix \
  --output output/seq_len_16/matched_control_text_group_importance_generated \
  --clean_score_cache_dir output/seq_len_16/matched_control_text_group_importance_generated
```

Outputs:
- `sample_metrics.txt`
- `sample_metrics.json`
- `clean_scores.json`
- `total_importance_summary[_seq_len_X].png`
- `layer_invalidity_rate[_seq_len_X].png`
- `group_importance_heatmap[_seq_len_X].png`
- `group_importance_lines[_seq_len_X].png`

Typical run:
```bash
sbatch scripts/text_importances.sh
```

## `evaluations/frames_entropy.py` 
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
python -u evaluations/frames_entropy.py \
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
- `run-<jobid>.log`: when using `scripts/entropy.sh`.

## Typical run (SLURM)
```bash
sbatch scripts/entropy.sh
```

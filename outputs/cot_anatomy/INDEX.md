# outputs/cot_anatomy — reasoning-trace anatomy on needle counting (PREREG_AGG 2026-09-07 S1a/S1b)

Script `outputs/_scratch/dbg/cot_anatomy.py`; Qwen3-8B thinking mode; needle stories K ∈ {4, 8, 16} × J ∈ {0, 64 same-domain distractors},
n = 12 per cell, max 3072 new tokens. Canonical run: `Qwen3-8B/20260907_110935_4003321` (job 428212); `ANALYSIS.md` in the run dir.

- Accuracy: clean .92 / 1.00 / .92; distractors .33 / .42 / .00 (most traces loop to the token limit, item indices run past the context).
- Written running tally in .97 (clean) / 1.00 (distractors) of traces; corrupting a middle numeral by +1: the next written numeral follows in
  .63 / .86, "repairs" 0 / 71; final answer shifts +1 in .06 of clean traces (redundant written tallies: index and count, ordinal and word).
- Chunk-onset attention on the target sentence (fraction of context attention): .42–.45 clean vs .07–.10 with distractors; AUROC of the share
  for failure within J = 64: .545.

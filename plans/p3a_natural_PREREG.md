# PRE-REGISTRATION — P3a natural-images MMRED full ladder (p0p2 campaign)

Written: 2026-07-24, BEFORE any P3a GPU job. Bands fixed here; never adjusted post-hoc.

## Data (resolved, not guessed)

- Source: `data/mmred_natural_v2/{dist_far,dist_near}` — the judge-gated A4 dataset
  (RESULTS [2026-07-10g,h]; per-frame `is_evidence` curated 0.998–1.00; one concept per
  cell = "dog"; 154 distinct evidence images, dist pools 1363/166). `ident_*` cells
  EXCLUDED (identical-evidence pathology rungs). The prior cross-domain references:
  carrier-token transfer tally 0.432 (n=50, v1 `data/mmred_natural/dist_far`,
  `carrier_token_crosstask_natural/`); in-domain replica gate→tally 0.920 (n=50).
- Composed roots (`experiments/glstm/natural_compose_mmred.py`, seed 11, build info in
  `data/mmred_natural_mm/BUILD_INFO.txt`): **global image-half split** (train samples draw
  only from half A, eval only from half B — image-held-out exams, stricter than
  sample-held-out; the v2 pools reuse images heavily so this matters). Dual-format dirs
  (qa.txt+NNN.png for trainer/frozen; meta.json+frame_XX.jpg for the probe).
  - Train: `seq_len_8/{cell}_train` 225×2 · `seq_len_16/{cell}_train` 220×2 → n=890.
  - Eval: `seq_len_8/{cell}_eval` 135×2 · `seq_len_16/{cell}_eval` 110×2.
  - N=16 golds {0..8,12,16}; N=8 golds {0..8}; per-count balanced.
- Template code: `parse_task_labels` natural branch + `frame_attr_labels` concept
  captions (CPU sanity: roundtrip OK, caption slots carry "dog(k)").

## Ladder + bands (fixed now)

1. **L0 frozen baseline** (eval-only, plain prompt, `frozen_baseline_eval.py`) on the 4
   eval roots (N=8/16 × dist_far/near). Reference measurement, no band. Park ref 0.219.
2. **L1 supply**: `replica_carrier_probe.py --natural --fence-frames --fence-blocks
   --reset-positions --question-first` on `mmred_natural_v2/dist_far` (LIMIT 300,
   shuffle 0) — replica d′ + in-run joint anchor + per-copy profile + adequacy.
   **Band: replica d′ ≥ 4.0 AND ≥ 2× the in-run joint anchor → supply available.**
   (Park @n900: 13.5; natural v1 @n50: 6.22 vs joint 3.12 — estimator-noisy.)
3. **L2 scaffold**: `replica_gate_tally.py` (CPU) on the L1 cache @L16.
   **Band: gate→tally exact ≥ 0.85 → scaffold works on natural** (v1 n=50 ref 0.920±wide;
   park 0.998).
4. **L3 in-model**: WINNING recipe verbatim (caption format, L12, r8, park-distilled
   carrier e_c — noted: e_c itself is park-trained, cross-domain carrier transfers ~51%
   d′; if L3 fails with L1/L2 GO, the e_c is a candidate culprit and L1/L2 localize it),
   `carrier_layer_cached.py --running-tally --jitter-gap 16 --grad-ckpt --limit 900
   --epochs 5 --l-open 12 --seed 0 --split-seed 0` on the 4 TRAIN roots; exams on the 4
   EVAL roots (in-length, image-held-out), dec 100 (N=8) / 180 (N=16), DUMPS=200.
   **Bands: pooled in-length exam ≥ 0.80 → cross-domain trainability GO;
   ≤ frozen(L0) + 0.10 → NO-GO (logged honestly); between → partial.**
   If L3 < GO: the L1/L2 vs L3 decomposition (supply vs readout vs e_c) IS the deliverable.
5. Zero-shot park→natural reference: the logged 0.432 (n=50) stands as the no-training
   floor for the transfer story; not re-run.

## Report per cell

acc, parse-fail, MAE, mean decode tokens, run dir + job id; per-count lines for the L3
exams. Verdicts vs these bands; numbers win; refuted bands reported as refuted.

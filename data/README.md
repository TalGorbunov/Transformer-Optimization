# data/ — generated datasets (untracked; this manifest is the record)

Nothing under `data/` is in git. Reproducibility = the generators in `datasets/mmred/`
(+ each root's own `metadata.json`/`generation_summary.json`/`BUILD_INFO`, written by the
generator with the exact args/seed). The main park generator was recovered byte-exact and
committed 2026-07-29 (`datasets/mmred/generate_mmred_park_dataset.py`).

## Canonical (referenced by live code / RESULTS.md entries / the training mixture)

| root | what | source |
|---|---|---|
| `mmred_images_park` | THE main dataset (park renders, seq 1–8, 4400 samples, seed 0) | `generate_mmred_park_dataset.py` (metadata.json = full record) |
| `mmred_longN_park`, `mmred_longN_park2` | long-N ladder (N=16..128; park2 = N=32×312 + N=48×210 in-length) | park generator, longN configs |
| `mmred_cooc_balanced`, `mmred_rooms_balanced`, `mmred_niah_which`, `mmred_union_or` | the 5-task mixture roots | `generate_mmred_balanced.py` family |
| `mmred_cooc_longN` | LOTO N=32 eval-only cooc (n=299, seed 7, gen job 125261) | balanced generator |
| `mmred_steps_balanced`, `mmred_sft_p41` | balanced steps / P4.1 SFT training set | balanced generator |
| `mmred_natural`, `mmred_natural_v2`, `mmred_natural_mm` | natural-image cells (judge-gated pools; _mm = composed, image-half split, BUILD_INFO in root) | `legacy/experiments/natural/` builders |
| `mlvu_ac`, `mlvu_ac_mmred` | MLVU-AC benchmark leg (converted) | `legacy/experiments/mlvu/convert_ac_to_mmred.py` |
| `herbench_ac`, `vnbench_cnt`, `vnbench_cnt_n32exact` | benchmark-ladder legs | `legacy/experiments/{herbench,vnbench}/prep_*.py` |
| `mmred_text_longN`, `mmred_text_arch` | text-MMRED (arch battery / text anchors) | text generators (legacy) |
| `coco_val2017` | natural-image source pool | download |

## Archive candidates (no references found by the 2026-07-29 audit — kept in place, never delete)

`mmred_longN_co_occupancy` · `mmred_longN_rooms_visited` · `herbench_ac_hi` ·
`mmred_smallN_park` · `mmred_perm_bias_seq8` · `mmred_corrupted_park_rooms_visited` ·
`mmred_corrupted_park_co_occupancy` · `mmred_park` · `mmred_cooc_2char/3char` ·
`mmred_rooms_1char*/2char` · `_smoke` — plus legacy-era roots used only by `legacy/` code:
`mmred_corrupted`, `mmred_agg`, `mmred_images`, `mmred`,
`mmred_images_park_no_step_marker`, `mmred_images_park_evidence_only_seq1_8`.
`oxford_pets`: probably a natural-pool source — verify before archiving.

If disk pressure ever demands it: move candidates into `data/_archive/` (move, never
delete) and update this manifest.

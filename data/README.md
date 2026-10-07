# data/ — where the datasets live (2026-09-21 layout)

Nothing under `data/` is in git, and since the 2026-09-21 migration nothing under `data/` is
even on `/home`: every entry is a **symlink** to `/rg`. This file is the manifest.

## The official benchmark (the ONLY dataset the current code reads)

| entry | target | contents |
|---|---|---|
| `data/mmred_hf` | `/rg/shocher_prj/lab_data/mmred` (lab-shared, permanent) | `hf/` Arrow cache of HF `ef1e43ce/mmred` · `json/<config>_<split>.json` prepped rows (`legacy/v1/scripts/mmred_hf/prep.py`, to be replaced by `experiments/prepare_data.py`) · `images/<config>_<split>/<qid>/frame_%04d.png` official renders at native 512 px (Fr0do/mmred renderer, commit 56c6ee7, deterministic) · `upstream_repo/` the pinned generator clone (no LICENSE file — internal use) · `dirsfiles/`, `headfit_raw.json`, `*_probe/` small helpers |

Splits present: train 2/4/8/16 · val 8/16 · test 8/16/32/64/128 · headfit 32/64/128
(1,200 rows per test/val config, 4,800 per train config, 24 qtypes balanced). 605 K frames, 13 G.
NOT copied: `dirs/` (1 M hardlinks — the legacy `qa.txt` bridge); it exists only in the retired
original `data/mmred_hf.moved-20260921` on /home; regenerate on /rg with
`legacy/v1/scripts/mmred_hf/materialize_dirs.py` if a legacy/v1 run ever needs it.

## External video benchmarks (not regenerable — source videos are gone; kept for a possible paper row)

| entry | target |
|---|---|
| `data/mlvu_ac`, `data/mlvu_ac_mmred`, `data/mlvu_ac_n32judge` | `/rg/shocher_prj/lab_data/mlvu/<name>` |
| `data/herbench_ac`, `data/herbench_ac_hi`, `data/herbench_retrieve_opt_clips` | `/rg/shocher_prj/lab_data/herbench/<name>` |
| `data/vnbench_cnt`, `data/vnbench_cnt_n32exact` | `/rg/shocher_prj/lab_data/vnbench/<name>` |

Built by `legacy/experiments/{mlvu,herbench,vnbench}/prep_*.py` (PyAV; `~/.local/pyav-py39`).

## The two external video benchmarks (SCALEUP, 2026-09-23; `docs/SCALEUP_2026-09-23.md`)

| entry | target | contents |
|---|---|---|
| `data/herbench_v2` | `/rg/shocher_prj/lab_data/herbench/lite_v2` | HF `DanBenAmi/HERBench` config `lite_v2`: `hf/` (annotations, README; tar parts deleted after extraction) · `videos/<source>/…mp4` (34.3 GB, KEPT) · `json/test.json` (1,971 rows, evidence intervals parsed per task) · `pool/<video_id>/` 256 frames on the official grid + `times.json` · `evidence/<qid>/` one frame per evidence unit · `pool.tar`, `evidence.tar` (`stage_video`) · `verify_report.txt` |
| `data/minerva` | `/rg/shocher_prj/lab_data/minerva/lmms_v1` | MINERVA (ICCV 2025) via the lmms `minerva.json` + `lmms-lab-eval/minerva` Lance video blobs: `hf/minerva.json` · `videos/<youtube_id>.mp4` (8.85 GB, 197 videos, KEPT) · `json/test.json` (1,207 rows; Listening dropped; evidence = timestamps regex-parsed from the reasoning) · `pool/`, `evidence/`, tarballs as above |

Built by `experiments/prepare_video.py` (layout spec in `core/data/videoqa.py`); `rows` and the MINERVA
`videos` stage need the network (login node), the rest runs under `4h_0g` (`sbatch/prepare_video.sbatch`).
Frames: long side 512 px, never upscaled, JPEG q95 — a prep-time choice, not model-path resize code.

## Custom-generator roots (legacy only — the rewrite does not read them)

Every other `data/mmred_*` root, `coco_val2017`, `oxford_pets`, `_smoke`, `_scratch_herb_smoke`
→ `/rg/shocher_prj/tal.gorbunov/archive/data/<name>` (personal archive; moved, never deleted).
Generators: `legacy/v1/datasets/mmred/*.py`; each root carries `metadata.json` (the three that
lacked one — `mmred_longN_park`, `mmred_longN_park2`, `mmred_redux` — got a reconstructed one on
2026-09-21 with the runner args and per-count histograms read back from the directory names).
The main park dataset is `mmred_images_park` (seq 1–8, 4,400 samples, seed 0).

## Migration bookkeeping

- Copies: `sbatch/migrate/copy_tree.sbatch` via `submit.sh` + the two manifests; every job log
  (`logs/copy_tree-<jobid>.out`) ends with `OK (rsync clean)` and a file-count MATCH.
- Switch: `sbatch/migrate/switch_symlinks.sh <manifest>` re-verifies counts + sampled md5s, then
  renames the original to `<name>.moved-20260921` and puts the symlink in place. The originals
  are deleted by hand, never by a script (`cleanup_2026-09-21.sh`).
- Job-time staging: `sbatch/lib/common.sh:stage_split` copies a per-split tarball
  (`images/<split>.tar`, written by `experiments/prepare_data.py`) to the node-local NVMe.

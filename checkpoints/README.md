# checkpoints/ — stable names for the live checkpoint chain

> **2026-09-21 (rewrite):** this README is the tracked record; the symlinks themselves are now
> git-ignored (local). Paths below of the form `outputs_legacy/...` resolve through a symlink to
> `/rg/shocher_prj/tal.gorbunov/archive/outputs/outputs_legacy/...` after the data migration, and
> every runner/script cited as `slurm/...` or `scripts/...` now lives under `legacy/v1/`.
> Nothing was re-pointed; every link still resolves (checked 2026-09-21).

Created 2026-07-29 (phase 0 of the `serious-refactor` reorg). Every entry is a **relative
symlink** into `outputs*/` — nothing was moved or copied. Purpose: give the checkpoints and
caches that live runners hardcode a stable path, so reorganizing `outputs/` cannot break the
live chain. **If a target run dir ever moves, re-point the symlink here — never re-edit ten
runners again.** (Runners still hardcode the raw paths as of phase 0; a later phase will switch
them to these names.)

All targets verified to exist on 2026-07-29 (`test -e` on every link). Sizes/mtimes below are
from that check.

## Model checkpoints (the deployed carrier stack)

| Stable name | Real target (repo-relative) | Producing run | Role in the method | Hardcoded by (raw path) |
|---|---|---|---|---|
| `carrier_token_room_k1_best.pt` (30 KB, Jul 18) | `outputs/ladder/image_longN/carrier_token/20260718_130545_distill_room_k1/carrier_best.pt` | Stage-1 carrier-token distillation, obj=distill init=room k=1, L=16, n=900 seq8 park (`report.txt`: BEST d' 11.45 @ep9, teacher 13.54) | The distilled carrier-token embedding e_c; warm-starts / is loaded frozen by every stage-2/3 `carrier_layer_*` trainer (`--carrier-ckpt` / `--carrier-init`) | `of_carrier_fmt`, `of_trunc_trainer`, `of_stage2_smoke4`, `of_stage3_smokeA`, `of_stage3_smokeA4`, `of_stage3_smokeC`, `p2a_loto_trainer`, `p3a_trainer`, `p42_carrier_digit` (9 .sbatch) |
| `carrier_token_room_k1_messages_best.npz` (103 MB, Jul 18) | `outputs/ladder/image_longN/carrier_token/20260718_130545_distill_room_k1/messages_best.npz` | Same stage-1 run (per-sample carrier messages at best epoch) | Message dump for the CPU gate→tally readout of the distilled carrier | `of_carrier_tally_cpu` (inline `np.load`) — the 10th runner referencing this run dir |
| `carrier_layer_l17_r8_best.pt` (8.0 MB, Jul 18) | `outputs/ladder/image_longN/carrier_layer/20260718_122503_L17_r8/carrier_layer_best.pt` | First stage-2 LoRA, L_open=17 r=8 init=distill, n=300 (`report.txt`: best emitted 0.840 @ep12; INDEX "Stage-2 30-ep convergence ref", job 123206) | Legacy stage-2 reference ckpt used by smoke tests (eval-only decode path) | `of_stage2_smoke2`, `of_stage2_smoke4` |
| `carrier_layer_l12v2_best.pt` (11.6 MB, Jul 21) | `outputs/ladder/image_longN/carrier_tally_l12v2/20260721_071710_L12_r8/carrier_layer_best.pt` | **l12v2** (job 124773): L_open=12 r=8 frozen e_c, scratchpad tally, 15-root pooled data n=8772, fixed ckpt criterion (`report.txt`: TF 1.000 / tf-exact 0.976 @ep5; exams: N=32 0.953 · N=48 0.878 · N=64 0.678 cap-adj — campaign-best long-N readout, INDEX line "l12v2") | The deployed L\*=12 carrier-layer LoRA (pre-caption default); the ckpt normally passed via `CKPT=` to eval runners (`of_trunc_eval`, `of_noharm`, `of_carrier_depth_dump`, `p2b_mlvu_eval`) | no runner hardcodes the .pt itself (env-passed); run dir hardcoded by `p41_exams`, `p2a_frozen_cooc` (see run-dir links below) |
| `carrier_layer_fmt_caption_best.pt` (11.6 MB, Jul 23) | `outputs/ladder/image_longN/carrier_fmt_caption/20260722_222032_L12_r8/carrier_layer_best.pt` | FMT sweep arm C = **caption format WINNER** (INDEX "FMT sweep DECIDED": N=64 0.981 pf 0; recipe = l12v2 verbatim, gold format = caption; trained by `of_carrier_fmt`) | The current deployed caption-format L\*=12 ckpt (what `of_carrier_depth_dump` calls "the fmt-caption ckpt"); used for MLVU-AC transfer cell (INDEX P2b) | none (env-passed via `CKPT=`) — promoted because it is the current method winner |
| `carrier_layer_loto_nococ_best.pt` (11.6 MB, Jul 24) | `outputs/ladder/image_longN/carrier_loto_nococ/20260724_064756_L12_r8/carrier_layer_best.pt` | P2a LOTO arm (leave-cooc-out): caption recipe, 11-root n=7872 no-cooc mixture (`report.txt`: BEST 0.998 @ep5; PREREG `plans/scratchpad_loto_PREREG.md`) | LOTO transfer-test ckpt (does an unseen task transfer in-model) | run dir hardcoded by `p2a_frozen_cooc` (eval_dirs; see below) |
| `carrier_layer_digit_p7a_lora_best.pt` (Aug 9) | `outputs/gating/p7_digit/arm1_lora/20260809_162439_129918/20260809_162514_L12_r8/carrier_layer_best.pt` | Gating P7a LoRA control (job 129918): `--digit-multi`, NO scratchpad, full 16-root mixture, 5 ep (`report.txt`: BEST 0.924 @ep5; INDEX P7a table) | The canonical DIGIT-readout carrier stack (answer emitted directly — not copyable); frozen e_c+LoRA for the learnmask gate arms after the 2026-08-12 scratchpad deprecation | `scripts/learnmask/train_mask_gates.py` + `slurm/train_mask_gates.sbatch` (default `--carrier-ckpt`) |

## PEFT adapter dirs (LoRA SFT baseline)

| Stable name | Real target | Producing run | Role | Hardcoded by |
|---|---|---|---|---|
| `sft_control_le8_v2_adapter/` | `outputs/ladder/image_longN/sft_control_le8_v2/20260720_191541_lora/adapter` | Plain LoRA-SFT control, train ≤8 (v2, Jul 20) | SFT baseline arm for P1.3/P4.3 (no-harm + eval-N64 comparisons) | `p13_sft_evalN64` (`--eval-only-adapter`), `p43_noharm_sft` (`--peft-adapter`) |
| `sft_inlength_p41_adapter/` | `outputs/ladder/image_longN/sft_inlength_p41/20260725_031153_lora/adapter` | P4.1 in-length SFT, train-seq-lens 8,16,32 (Jul 25; PREREG `plans/p4_PREREG.md`) | In-length SFT arm for the P4.1 exam cells | `p41_exams` (`--eval-only-adapter $D/adapter`) |
| `sft_ff_le8_adapter/` | `outputs/loramech/p0_ff_le8/20260826_161912_lora/adapter` | LORAMECH P0 ff_le8 (job 137467, Aug 26): FRAMES-FIRST regime, train N≤8, r=8 recipe = E-B mirror | Canonical frames-first SFT baseline, ≤8 window (exam N8/16/32 = 0.540/0.333/0.180, pf 0 — does NOT fit in-length; see `outputs/loramech/STATE.md`) | eval via `train_sft_baseline.sbatch` `ADAPTER=` + `FRAMES_FIRST=1` |
| `sft_ff_le32_adapter/` | `outputs/loramech/p0_ff_le32/20260826_171532_lora/adapter` | LORAMECH P0 ff_le32 (job 137465, h200, Aug 26): FRAMES-FIRST, train N∈{8,16,32}, r=8 recipe = P4.1 mirror | Canonical frames-first in-length SFT baseline (exam N8/16/32 = 0.600/0.480/0.313, pf 0 — **H0 refuted**, vs Q-first P4.1 0.967@32; see `outputs/loramech/STATE.md`) | eval via `train_sft_baseline.sbatch` `ADAPTER=` + `FRAMES_FIRST=1` |
| `sft_fenced_le16_adapter/` | `outputs/loramech/p1_fenced_le16/20260827_131154_fenced/adapter` | LORAMECH P1 fenced-SFT (job 137744, l40s, Aug 27): plain LoRA trained under the ARCHITECTURAL fence — N×[frame+q] blocks + posreset + one count prompt, NO Q-first, train N∈{8,16} | The conditioning-is-the-factor arm: in-length **0.993@8 / 0.847@16** (live per-count curves), zero-shot 0.387@32 — beats plain-ff (0.540@8) AND Q-first C1 (0.867@8) | eval via `slurm/loramech_train_fenced.sbatch` `ADAPTER=` |
| `sft_fenced_gated_vn_adapter/` | `outputs/sparse/s8_vn/20260831_134617_gated/adapter` | SPARSE S8 (job 139247, Aug 31): P1g recipe + **virtual-N mixture** (50% real gated + 50% synthetic evidence-only, k∈0..16 × declared Ñ∈{8..128}) — kills the prompt-N snap inside the trained grid | Gated exams **1.000/1.000/0.853** @8/16/32 (k12/k16 @32 = 11/11 vs P1g 0/11); k>16 = coverage edge (undercount, no snap). **EVAL CONTRACT: gate REQUIRED** | eval via `slurm/sparse_train.sbatch` `ADAPTER=` `GATE=` |
| `sft_fenced_gated_vn_nfree_adapter/` | `outputs/sparse/s9b_vn_nfree/20260901_140231_gated/adapter` | SPARSE S9b ep10 twin (job 139714, Sep 1): N-free prompt + k-balanced evidence-subset mixture (k∈0..16 dense — fixes the park-seq16 answer-support hole K0-8/12/16-only) | **CANONICAL N-free arm**: 1.000/1.000/0.847 @8/16/32, k12+k16 ≈ perfect @32; ep5 arm = budget-ablation twin | eval via `slurm/sparse_train.sbatch` `ADAPTER=` `GATE=` `EXTRA=--nfree-prompt` |
| `sft_fenced_gated_nfree_adapter/` | `outputs/sparse/s9_nfree/20260831_174021_gated/adapter` | SPARSE S9 (job 139318, Aug 31): P1g recipe + **N-free prompt** (no frame count/answer range in text — matches upstream MMReD protocol; no mixture needed) | Gated exams **1.000/0.973/0.793** @8/16/32, NO snap signature (mse +0.29 @32; k16 10/11). **EVAL CONTRACT: gate + `--nfree-prompt` REQUIRED**; numbers are a NEW prompt anchor — label the prompt variant in any comparison | eval via `slurm/sparse_train.sbatch` `ADAPTER=` `GATE=` `EXTRA=--nfree-prompt` |
| `sft_fenced_gated_adapter/` | `outputs/sparse/s1_p1g_r3/20260831_011807_gated/adapter` | SPARSE S1 P1g (job 139066, l40s, Aug 31): fenced-SFT recipe + **oracle evidence gate** (hide_cols = non-evidence block spans, train AND eval; trainer `scripts/sparse/train_sft_gated.py` with the mask-held-through-backward fix — see `outputs/sparse/STATE.md` 2026-08-31) | The k-regime arm: gated exams **1.000 @8 / 0.947 @16 / 0.780 @32 zero-shot** with k≤8 PERFECT (95/95) @32 and all errors = mid-k→"32" saturation. **EVAL CONTRACT: the gate is REQUIRED** (`--gate oracle` or `--gate model --gate-npz`; ungated eval of this adapter is meaningless) | eval via `slurm/sparse_train.sbatch` `ADAPTER=` `GATE=` |
| `sft_fenced_hf_adapter/` | `outputs/loramech/p2_fenced_hf/20260827_193851_fenced/adapter` | LORAMECH P2 (job 137799, l40s, Aug 27): fenced-SFT on the MMReD-HF benchmark's own `steps_in_room` train splits, seq 8+16 (320 train) | Faithful-benchmark fenced row: HF test **0.900@8 / 0.660@16** (majority 0.62/0.38, frozen anchor 0.559@8), zero-shot 0.300@32 | eval via `slurm/loramech_train_fenced.sbatch` `ADAPTER=` |

## Run-dir links (runners read `eval_dirs*.txt` and `adapter/` out of these)

| Stable name | Real target | Why a dir link | Hardcoded by |
|---|---|---|---|
| `carrier_tally_l12v2_run/` | `outputs/ladder/image_longN/carrier_tally_l12v2/20260721_071710_L12_r8` | `eval_dirs_N64.txt`, `eval_dirs_cooc_all.txt`, … define the frozen exam splits | `p41_exams` (`$A`), `p2a_frozen_cooc` (`$A`) |
| `carrier_loto_nococ_run/` | `outputs/ladder/image_longN/carrier_loto_nococ/20260724_064756_L12_r8` | `eval_dirs_coocN32.txt` (LOTO exam split) | `p2a_frozen_cooc` (`$L`) |
| `sft_inlength_p41_run/` | `outputs/ladder/image_longN/sft_inlength_p41/20260725_031153_lora` | `adapter/` + `eval_dirs_p41_N32.txt` | `p41_exams` (`$D`) |

## Data caches (.pt) hardcoded by live runners / experiment defaults

Not model weights, but part of the live chain (CPU tally/probe runners re-read them).

| Stable name | Real target | Producing run | Role | Hardcoded by |
|---|---|---|---|---|
| `msgcache_count_seq8.pt` (411 MB, Jul 3) | `outputs/frame_axis/probes/carrier_message/count_msgcache/count/messages_cache.pt` | frame→carrier message cache, count task, seq8 park | Default `MSG_CACHE` for d′ dose-response; "steps" cache for tally-register | `dprime_dose_response` (default), `experiments/glstm/tally_register_solution.py` |
| `msgcache_cooc_seq8.pt` (1.1 GB, Jul 4) | `outputs/frame_axis/probes/carrier_message/cooc_msgcache_big/co_occupancy/messages_cache.pt` | same, co-occupancy task | "cooc" cache for tally-register | `experiments/glstm/tally_register_solution.py` |
| `msgcache_rooms_seq8.pt` (247 MB, Jul 4) | `outputs/frame_axis/probes/carrier_message/rooms_msgcache_big/rooms_visited/messages_cache.pt` | same, rooms-visited task | "rooms" cache for tally-register | `experiments/glstm/tally_register_solution.py` |
| `msgcache_replica_blockfence_qfirst_full900.pt` (310 MB, Jul 18) | `outputs/ladder/image_longN/replica_blockfence_qfirst_full900/20260718_130546/messages_cache.pt` | A3 replica+blockfence+qfirst full-900 capture (one-forward d′ 6.34 line) | Gate→tally CPU readout input | `of_gate_tally_cpu` |
| `msgcache_joint_N8.pt` (103 MB, Jul 10) | `outputs/ladder/image_longN/joint/N8/20260710_215405/count/messages_cache.pt` | joint-context image caches, deployed locus L16/off9 (Jul 10) | Measured best-linear-readout ceiling per N | `p12_measured_ceiling_cpu` |
| `msgcache_joint_N16.pt` (205 MB, Jul 10) | `outputs/ladder/image_longN/joint/N16/20260710_215406/count/messages_cache.pt` | 〃 | 〃 | 〃 |
| `msgcache_joint_N32.pt` (411 MB, Jul 10) | `outputs/ladder/image_longN/joint/N32/20260710_215405/count/messages_cache.pt` | 〃 | 〃 | 〃 |
| `msgcache_joint_N64.pt` (548 MB, Jul 10) | `outputs/ladder/image_longN/joint/N64/20260710_215405/count/messages_cache.pt` | 〃 | 〃 | 〃 |
| `msgcache_joint_N128.pt` (822 MB, Jul 10) | `outputs/ladder/image_longN/joint/N128/20260710_215411/count/messages_cache.pt` | 〃 | 〃 | 〃 |
| `bench_cache_internvl_multipass_qfirst.pt` (39 MB, Jul 19) | `outputs/frame_axis/internvl/multipass_qfirst/20260719_004112/bench_cache.pt` | InternVL multipass qfirst bench capture (produced by `internvl_multipass`) | Cross-model gate→tally replication input | `experiments/glstm/internvl_gate_tally.py` (module-level `CACHE`) |

## Dead references found (NOT promoted — targets no longer exist)

| Raw path referenced | Referenced by | Status on disk |
|---|---|---|
| `outputs_no_train/distractor_oracle_posneg_write_read_adapter_seq8_7b/oracle_posneg_write_read_5epochs/checkpoints/oracle_posneg_write_read_adapter_best.pt` | `distractor_posneg_frozenreadout_smoke.sbatch` | run dir exists, but `checkpoints/` is **empty** (purged ~Jun 15); runner is already broken |
| `outputs/message_memory_adapter_stage1_stage3_seq8_7b_20260526_212606/checkpoints/stage3_best.pt` | `run_memory_carrier_site_layer_norm_sweep_seq8.sbatch` (default `MESSAGE_MEMORY_RUN`) | run dir moved to `outputs_oreo/…` in an earlier reorg AND its `checkpoints/` is empty — doubly dead |
| `outputs/shared_count_direction_memory_seq8_7b_20260527_203756/checkpoints/stage3_shared_count_direction_plus_small_residual_best.pt` | `run_memory_injection_site_sweep_seq8.sbatch` (default `BASELINE_RUN`) | same: dir now at `outputs_oreo/…`, `checkpoints/` empty |

## Deliberately not promoted

- `of_stage3_smokeA` line 31 (`${RUN}carrier_layer_best.pt`) and `of_posreset_sweep_tally`
  (`${d}messages_cache.pt`): dynamic paths into dirs the same job just created/globbed — nothing
  stable to pin.
- `of_trunc_eval` / `of_noharm` / `of_carrier_depth_dump` / `p2b_mlvu_eval` take `CKPT=` via env
  (no hardcoded default). The values normally passed are already promoted above
  (`carrier_layer_l12v2_best.pt`, `carrier_layer_fmt_caption_best.pt`).
- TRUNC retrain candidates (`trunc_retrain/carrier_caption_trunc12/…`, E4c l_open=20): line still
  in flux as of 2026-07-25 (INDEX), no runner hardcodes them; promote once a winner is declared.

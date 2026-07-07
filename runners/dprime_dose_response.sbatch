#!/bin/bash
#SBATCH --job-name=ddose
#SBATCH --partition=l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:45:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/tal.gorbunov/projects/Transformer-Optimization}"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python -u evaluations/scripts/patch_importence/dprime_dose_response.py \
  --task "${TASK:-count}" \
  --data_root "${DATA_ROOT:-data/mmred_images_park/seq_len_8/all_uniform}" \
  --messages-cache "${MSG_CACHE:-outputs/frame_axis/probes/carrier_message/count_msgcache/count/messages_cache.pt}" \
  --layer "${LAYER:-16}" --multi-layers "${MULTI_LAYERS:-14,16,18,20}" \
  --offset "${OFFSET:-9}" --probe-layer "${PROBE_LAYER:-24}" \
  --limit "${LIMIT:-150}" --lams-single "${LAMS_SINGLE:-}" --lams-multi "${LAMS_MULTI:-4,16}" \
  --scrub-offsets "${SCRUB_OFFSETS:-}" --scrub-offsets2 "${SCRUB_OFFSETS2:-}" --multi-layers2 "${MULTI_LAYERS2:-18,20}" --scramble-offsets "${SCRAMBLE_OFFSETS:-}" --scramble-layers "${SCRAMBLE_LAYERS:-2,4,6,8,10,12}" \
  --rand-lam "${RAND_LAM:-0}" ${RESCUE_LAMS:+--rescue-lams "$RESCUE_LAMS"} ${SETG:+--setg} ${NOISE_SIGMAS:+--noise-sigmas "$NOISE_SIGMAS"} ${NATIVE_AXES:+--native-axes "$NATIVE_AXES"} ${NAT_LAMS:+--nat-lams "$NAT_LAMS"} ${NAT_SCRUB:+--nat-scrub} ${NAT_SET:+--nat-set} ${SCRUB_RANDOM:+--scrub-random} ${KCHANNEL:+--kchannel} --sample-seed "${SAMPLE_SEED:-0}" \
  --output "${OUTPUT:-outputs/frame_axis/probes/dprime_dose}"

#!/bin/bash
#SBATCH -J netprobe
#SBATCH -p cpu
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 00:05:00
#SBATCH -o logs/%x-%j.out
timeout 20 curl -sI https://huggingface.co | head -1 || echo "NO-NET"
timeout 60 curl -s -r 0-104857599 -o /dev/null -w "100MB in %{time_total}s\n" "$(cd $SLURM_SUBMIT_DIR && .venv/bin/python -c "from huggingface_hub import hf_hub_url;print(hf_hub_url('DanBenAmi/HERBench','videos/videos.tar.part.16',repo_type='dataset'))")" -L || echo "range-dl failed"

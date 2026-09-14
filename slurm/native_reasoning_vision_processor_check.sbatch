#!/bin/bash
#SBATCH -J native_reason_processor
#SBATCH -p cpu
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:05:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export HF_HOME="$HOME/.cache/huggingface" HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
.venv/bin/python - <<'PY'
from pathlib import Path
import hashlib,json,os
from transformers import AutoConfig,AutoProcessor
p=Path('/mnt/home/gabriele.serussi/.cache/huggingface/hub/models--nvidia--Cosmos-Reason1-7B/snapshots/3210bec0495fdc7a8d3dbb8d58da5711eab4b423')
c=AutoConfig.from_pretrained(p,local_files_only=True)
processor=AutoProcessor.from_pretrained(p,local_files_only=True,use_fast=False)
assert c.model_type=='qwen2_5_vl'
index=p/'model.safetensors.index.json';ix=json.loads(index.read_text());names=sorted(set(ix['weight_map'].values()))
assert len(names)==4 and all((p/n).is_file() for n in names)
r=dict(model='nvidia/Cosmos-Reason1-7B',snapshot=str(p),model_type=c.model_type,processor_type=type(processor).__name__,tokenizer_type=type(processor.tokenizer).__name__,shards={n:(p/n).stat().st_size for n in names},metadata_sha256={n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in ['config.json','model.safetensors.index.json','tokenizer_config.json','preprocessor_config.json']},slurm_job_id=os.environ['SLURM_JOB_ID'],scope='CPU config/processor load and shard presence; no model weights loaded')
out=Path('outputs/native_aggregation_vlm/v3/reasoning_compatibility');out.mkdir(parents=True,exist_ok=True)
(out/f'cpu_check_{os.environ["SLURM_JOB_ID"]}.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
PY

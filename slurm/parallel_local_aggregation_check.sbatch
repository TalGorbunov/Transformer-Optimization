#!/bin/bash
#SBATCH -J parallel_local_core
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:05:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
.venv/bin/python - <<'PY'
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

assert os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_JOB_PARTITION") == "cpu"
assert not os.environ.get("SLURM_JOB_GPUS")
started = time.monotonic()
names = ("gnnformer/parallel_local_aggregation.py", "tests/test_parallel_local_aggregation.py",
         "slurm/parallel_local_aggregation_check.sbatch")
payloads = {name: Path(name).read_bytes() for name in names}
hashes = {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()}
root = Path("outputs/native_aggregation_vlm/parallel_local/core_checks")
output = root / f"check_{os.environ['SLURM_JOB_ID']}"
output.mkdir(parents=True, exist_ok=False)
for name, payload in payloads.items():
    target = output / "code" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
result = subprocess.run([sys.executable, names[1]], capture_output=True, text=True)
(output / "tests.stdout.txt").write_text(result.stdout)
(output / "tests.stderr.txt").write_text(result.stderr)
print(result.stdout, end="")
print(result.stderr, end="", file=sys.stderr)
unchanged = all(Path(name).read_bytes() == payload for name, payload in payloads.items())
test_count = sum(isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
                 for node in ast.walk(ast.parse(payloads[names[1]])))
report = dict(schema_version=1, passed=result.returncode == 0 and unchanged,
              slurm_job_id=os.environ["SLURM_JOB_ID"], source_sha256=hashes,
              source_unchanged_during_checks=unchanged, unit_test_count=test_count,
              test_exit_code=result.returncode, elapsed_seconds=time.monotonic() - started,
              scope="Standalone software math only; no model integration or efficacy result")
(output / "core.json").write_text(json.dumps(report, indent=2) + "\n")
(output / "source_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
(output / "INDEX.md").write_text("# Parallel local aggregation CPU checks\n\n"
    "- [Results](core.json)\n- [Frozen source hashes](source_hashes.json)\n"
    "- [Test output](tests.stderr.txt)\n")
index = root / "INDEX.md"
if not index.exists():
    index.write_text("# Parallel local aggregation software checks\n\n")
with index.open("a") as stream:
    stream.write(f"- Slurm {os.environ['SLURM_JOB_ID']}: [results]({output.name}/core.json).\n")
print(json.dumps(dict(output=str(output), passed=report["passed"])))
if not report["passed"]:
    raise SystemExit("Standalone aggregation CPU checks failed")

PY


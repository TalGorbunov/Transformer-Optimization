#!/bin/bash
#SBATCH -J parallel_local_native
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
tests = ("tests/test_parallel_local_aggregation.py", "tests/test_parallel_local_native.py")
names = ("gnnformer/parallel_local_aggregation.py", "gnnformer/parallel_local_native.py",
         *tests, "slurm/parallel_local_native_check.sbatch")
payloads = {name: Path(name).read_bytes() for name in names}
hashes = {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()}
root = Path("outputs/native_aggregation_vlm/parallel_local/native_checks")
output = root / f"check_{os.environ['SLURM_JOB_ID']}"
output.mkdir(parents=True, exist_ok=False)
for name, payload in payloads.items():
    target = output / "code" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
results = []
for name in tests:
    begin = time.monotonic()
    result = subprocess.run([sys.executable, name], capture_output=True, text=True)
    stem = Path(name).stem
    (output / f"{stem}.stdout.txt").write_text(result.stdout)
    (output / f"{stem}.stderr.txt").write_text(result.stderr)
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    count = sum(isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
                for node in ast.walk(ast.parse(payloads[name])))
    results.append(dict(script=name, test_exit_code=result.returncode,
                        unit_test_count=count, seconds=time.monotonic() - begin))
unchanged = all(Path(name).read_bytes() == payload for name, payload in payloads.items())
report = dict(schema_version=1, passed=unchanged and all(row["test_exit_code"] == 0 for row in results),
              slurm_job_id=os.environ["SLURM_JOB_ID"], source_sha256=hashes,
              source_unchanged_during_checks=unchanged, tests=results,
              unit_test_count=sum(row["unit_test_count"] for row in results),
              elapsed_seconds=time.monotonic() - started,
              scope="CPU standalone math/native-hook software only; no pretrained HF or efficacy result")
(output / "core.json").write_text(json.dumps(report, indent=2) + "\n")
(output / "source_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
(output / "INDEX.md").write_text("# Parallel local native integration CPU checks\n\n"
    "- [Results](core.json)\n- [Frozen source hashes](source_hashes.json)\n"
    "- [Native-hook test output](test_parallel_local_native.stderr.txt)\n"
    "- [Aggregation test output](test_parallel_local_aggregation.stderr.txt)\n")
index = root / "INDEX.md"
if not index.exists():
    index.write_text("# Parallel local native software checks\n\n")
with index.open("a") as stream:
    stream.write(f"- Slurm {os.environ['SLURM_JOB_ID']}: [results]({output.name}/core.json).\n")
print(json.dumps(dict(output=str(output), passed=report["passed"])))
if not report["passed"]:
    raise SystemExit("Parallel local native CPU checks failed")

PY


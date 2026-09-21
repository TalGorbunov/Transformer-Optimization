# Transformer-Optimization — relieving over-squashing in a frozen VLM

MSc thesis code, Tal Gorbunov, 2026. **Rewrite in progress (branch `rewrite`, since
2026-09-21):** the repo is being rebuilt by hand around the official MMReD benchmark; the
July–September code that produced the current results is frozen under `legacy/v1/`.

## The problem and the method

A frozen Qwen2.5-VL-7B (4-bit) must answer questions about a sequence of N frames (characters
moving between rooms). Joint attention over all frames dilutes each frame's evidence like an
over-squashed GNN, and accuracy collapses with N. The method keeps the backbone frozen and
changes only how information flows:

1. **Fence.** Every frame is encoded in its own attention block (block-diagonal mask + per-block
   position reset) with the question visible in the shared prefix. No frame sees another frame,
   so per-frame evidence is never diluted, in one forward.
2. **Gate.** A small classifier on each block's `<|vision_end|>` state decides whether the frame
   is evidence for the question; non-evidence blocks are hidden from the readout. The gate is
   N-independent by construction (a block's state is the same at N = 8 and N = 128).
3. **Read.** A small LoRA decodes the answer from the prefix plus the kept blocks only. With the
   blocks batched over a shared prefix, encode cost is linear in N and the read costs k blocks.

Everything is measured on the **official MMReD benchmark** (HF `ef1e43ce/mmred`, frames from the
authors' renderer at native resolution, their prompt verbatim), with the paper's protocol: train
on sequence lengths ≤ 16, test on all lengths up to 128.

## Layout

```
core/          the method: constants · model · fence · mmred · prompt · metrics   (tests/ pin each)
experiments/   one file per experiment (prepare_data, train, evaluate, probes, baselines, figs/)
sbatch/        SLURM wrappers + lib/common.sh + migrate/
tests/         CPU tests, seconds:  python tests/test_fence.py  (etc.)
checkpoints/   README.md = the stable-name → run-dir record; symlinks are local
data/          symlinks to /rg (mmred_hf = the official benchmark; the rest = legacy archives)
outputs/       run dirs; INDEX.md / STATE.md / CAMPAIGN_BRIEF.md per group are tracked
docs/          paper plan, prior art, framing audits, theory explainers, archive/
legacy/        pre-July-2026 code + v1/ (July–Sept 2026), frozen; run with PYTHONPATH=legacy/v1
RESULTS.md     append-only research log
```

## Quickstart

```bash
source .venv/bin/activate                    # Python 3.11: model stack + official `mmred` + `core`
python tests/test_fence.py                   # CPU; every core module has one
python experiments/prepare_data.py --help    # HF rows -> json, official renderer -> frames
```
On the cluster submit through `sbatch/` ([CLAUDE.md](CLAUDE.md) §3 has the partition/QOS rules).

## Reproducibility contract

- Every number in [RESULTS.md](RESULTS.md) traces to a run dir on disk; canonical adapters have
  stable names in [checkpoints/README.md](checkpoints/README.md).
- Each experiment's docstring names the logged anchor it must reproduce.
- `tests/test_fence.py` pins the mask and position reset bit-for-bit against both frozen
  implementations; `tests/test_prompt.py` pins the prompt byte-for-byte against the benchmark
  authors' inference script; `tests/test_mmred.py` demands 100 % gold parity over all 24 question
  types.
- Results from before the rewrite (park-generator data, the carrier/scratchpad method) are
  reproducible from `legacy/v1/` and labelled as legacy data in the log.

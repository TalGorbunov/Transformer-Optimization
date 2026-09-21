"""gnnformer — GNN-style message passing in a frozen vision-language transformer.

Method (thesis: "relieving the aggregation bottleneck in VLMs", Tal Gorbunov 2026):
  1. FENCING  — one-forward per-frame question replicas behind a block-diagonal
                attention fence with per-block M-RoPE position reset (supply repair).
  2. CARRIERS — a learned carrier token per frame + a small LoRA above the
                separator layer L* that integrates carrier messages (in-model readout).
  3. READOUT  — caption-scan scratchpad decoding (readout expressivity fix).

Modules: runtime (model loading), fencing (masks/positions/hooks), carriers
(e_c + LoRA + checkpoints), data (MMRED samples), scratchpad (targets/parser),
metrics (d', law, accuracy), constants.

Legacy pre-refactor implementations live in ../legacy/ (frozen, reproducibility
record for RESULTS.md). Ports here are verified against legacy anchor numbers.
"""

__version__ = "1.0.0"

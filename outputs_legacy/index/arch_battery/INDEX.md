# arch_battery — cross-architecture text-MMRED battery (Exp 1 of plans/arch_experiments_agent_brief.md)

Task: steps_in_room, states-as-text, EXACT prompt of `eval_mmred_text_frames_acc.py`, greedy,
max_new_tokens=12, generation reader (first integer), n=150/N, seed 2.
Data: N=8 `data/mmred_images_park` · N=16/24/40 `data/mmred_text_longN` · N=64/128 `data/mmred_text_arch` (new, job 120907).
Env: `.venv_arch` → /rg/shocher_prj/tal.gorbunov/venv_arch (py3.11, torch 2.7.1→2.8.0+cu128, transformers 5.13.1); HF cache /rg/shocher_prj/tal.gorbunov/hf_arch.
Anchor (measured earlier, not re-run): Qwen2.5-VL-7B text EM 0.196/0.062/0.035/0.020 @ N=8/16/24/40.

| experiment | canonical run | headline |
|---|---|---|
| **battery, 6 models × N∈{8,16,24,40,64,128}** (jobs 120918/120922/120923/120928/120947/120948/120967/120968/120971; a100+l40s, ~3.5 GPU-h total) | `outputs/arch_battery/20260713_215812/` (`battery_all.csv`, `fig_arch_battery.png`; per-model subdirs with results.csv + predictions.csv) | NO architecture escapes: all models ≤0.09 EM ≈ majority by N≥40. Softmax (Mistral-7B, Qwen2.5-14B) collapse with undercount (bias → −16); hybrids (Falcon-H1, Nemotron-Nano) and Griffin (recurrentgemma) collapse too — Falcon-H1 only slightly flatter at N=16–24 (0.127/0.120 vs Mistral 0.127/0.040). xLSTM-7b base-only → 40–80% parse fails, uninformative. Qwen3-Next-80B dropped (bnb can't quantize packed-Parameter MoE experts; 3 attempts) |

Caveats: Falcon-H1/Nemotron N≤40 ran on the naive mamba2 path (torch 2.7.1), N=64/128 with mamba-ssm kernels (torch 2.8.0) after OOM repairs — same weights/greedy protocol, split noted in run_config.json of the `*_n64128` dirs. Nemotron used system=/no_think (it is a reasoning model by default). a100-public GPUs are 40GB.

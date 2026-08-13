# Mission brief — architectural-solution experiments (Exp 2 → Exp 1)

> **Agent instructions.** You are a long-running Claude Code session in tmux executing two
> pre-registered experiments. Work autonomously: submit jobs, poll `squeue`, collect results,
> write reports. Read `CLAUDE.md` first and obey it — especially: **never `pip install` into the
> shared `.venv`** (Exp 1 requires a NEW venv, see below), never run heavy compute on the login
> node, right-size QOS, check idle GPUs across ALL partitions before submitting
> (`a100-public` is often idle), smokes go to `outputs/_scratch/`.
> **Do NOT edit `RESULTS.md` or any docs/ artifact.** Instead write draft entries (see
> Deliverables). If truly blocked (missing data, license wall, repeated OOM), write
> `plans/arch_experiments_BLOCKED.md` explaining what and why, and stop that experiment.

## Context (2 paragraphs, read once)

The d′ theory of this repo: per-frame evidence messages at a carrier token (room token, off −9,
L16) follow `m = μ + e·δ + ε`; held-out shrinkage-LDA d′ measures per-frame supply; the law
`2Φ(d′/2√N)−1` (count-prior-mixed) prices any linear read of the summed messages, and the
gate-law (per-frame error p = 1−Φ(d′/2), exact = P(errors cancel)) prices any hard-gate tally.
Three walls: Ⅰ supply (joint-context tax — one query cannot address N sources; joint d′ ~2 vs
isolated ~7; the value half is repairable, the query half is not), Ⅱ pooling (√N), Ⅲ readout
(native axis / clamp / token-interface necessity).

These experiments test **architectural** escape routes. Exp 2 asks whether a *trained* per-frame
query can recover supply from joint-encoded features (the ceiling for any DETR/slot-style head).
Exp 1 asks whether released **extensive-state architectures** (Gated-DeltaNet/Mamba/RWKV hybrids)
escape the law's collapse on a text counting battery that softmax models provably fail.

---

## Experiment 2 first — trained-query ceiling test (cheap, go/no-go for the slot head)

**Question.** Train a query vector to read evidence from JOINT-encoded k/v. What d′ does it reach?

**Pre-registered anchors (n=500 capture, L16, all already measured — reproduce before trusting
anything):** joint-q × joint-kv **2.09–2.33** · mp-q × joint-kv **4.47** (the plausible ceiling —
a "perfect" frame-specific query on joint values) · clean-q(pad) × joint-kv 3.97 ·
mp-q × mp-kv 6.33–7.33.
**Interpretation bands (pre-registered):** trained-q d′ ≥ 4 → slot-head architecture is GO
(gate-law then promises ≈0.68/0.49/0.29 exact at N=32/64/128); d′ ≈ 2 → requirement-1 confirmed
(learned queries cannot beat the joint query; per-frame addressing must be architectural);
in between → partial recovery, report as measured.

**Data & machinery (all exists):**
- Capture: `outputs/ladder/image_longN/qkv_2x2/20260712_n500/{qkv_capture.pt (13.4GB), oproj_dense.pt}`.
  Structure: `blob["samples"][i]["arms"][{"joint","mp","pad"}][L]` with `q` (joint: key `"all"`,
  one [3584] carrier query; mp/pad: per-frame index), `k`/`v` per frame `[196, 512]` (pre-rotary,
  fp16); `rope_cos/rope_sin`, `joint_fg_sizes`, `labels`.
- Reusable helpers (`rot_q`, `rot_k`, `vrep`, `msg`, mrope config, split convention seed 0,
  train 300 / eval 200): `experiments/glstm/unmixer_jointq_cell.py` and
  `experiments/glstm/encoding_unmixer.py`. Evaluation MUST use `dprime_pair` from
  `experiments/glstm/dprime_vs_n.py` (held-out, sample-disjoint, 3 seeds) on messages shaped
  `[n_eval, 8, 3584]`.

**Design (implement `experiments/glstm/trained_query_ceiling.py`):**
1. Load capture; same train/eval split (RandomState(0), 60/40) as the un-mixers.
2. Train, on TRAIN samples only, a query that reads evidence from joint k/v. Two arms:
   (a) **shared learned query** `q* ∈ R^3584` (init = the joint carrier query mean), and
   (b) **query = small MLP of the mp query's input**? NO — keep it honest: arm (b) is
   `q* per head` with the same shared-across-frames constraint. (A frame-specific query would
   need frame identity at inference — that is the thing the 2×2 says doesn't exist; the shared
   q* is exactly what a trained head could deploy.)
   Objective: differentiable proxy for separability — logistic loss of `w·msg(q*, k_f, v_f) + b`
   on the per-frame evidence label, optimizing `q*, w, b` jointly (torch, CPU or single GPU;
   messages recomputed per batch through the SAME rotary/within-frame-softmax/o_proj path as the
   helpers — port `msg()` to torch with gradients w.r.t. q*).
3. Evaluate: freeze q*, rebuild eval-split messages `msg(q*, joint k/v)`, run `dprime_pair`.
   Report alongside the four anchors (recompute the joint-q and mp-q anchors in the same run as
   cross-checks — they must land on 2.09–2.33 / 4.47 or the run is invalid).
4. Ablation (cheap, same run): q* initialized from mp-query-mean vs from joint query vs random —
   does the optimum depend on init (local minima)?

**Compute:** ONE job, `a100-public`, `--qos=2h_2g`, `--mem=160G`, `--cpus-per-task=16`, 1 GPU
(training q* on GPU is fine; the 160G is for the capture blob — this is the proven footprint of
job 120864). If 2h is tight for the optimization, checkpoint q* and resubmit; do NOT raise QOS
above `12h_4g`.

**Output:** `outputs/ladder/image_longN/qkv_2x2/20260712_n500/trained_query/` (report.txt,
results.csv, the trained q*.pt) + append one row to `outputs/ladder/INDEX.md`.

---

## Experiment 1 — cross-architecture text-MMRED battery (the natural experiment)

**Question.** Do released extensive-state architectures escape the softmax counting collapse?

**Pre-registered predictions (write them in the report BEFORE running):** softmax-attention
models collapse on the law schedule with the undercount/clamp signature (emitted range compresses,
bias grows negative, model → majority by N≈16 — the measured Qwen2.5 anchor: EM 0.196/0.062/
0.035/0.020 at N=8/16/24/40, majority-locked from N=16); write-gated extensive-state models show a
flatter EM-vs-N curve, wider emitted range, ordinal correlation surviving to larger N. Either
outcome is a finding; a hybrid that collapses identically is evidence that pretraining, not the
primitive, dominates.

**Step 0 — model inventory (login node, no GPU).** Candidates, in priority order — verify open
weights on HF, license, loadability, and 4/8-bit footprint ≤ 80GB (a100):
- softmax controls: `meta-llama/Llama-3.1-8B-Instruct` (gated license — if blocked, use
  `mistralai/Mistral-7B-Instruct-v0.3` or Qwen2.5-14B as a second softmax point). Qwen2.5-7B
  anchor numbers already exist (do not re-run unless the harness changes the prompt).
- hybrids/recurrent (pick 2–3 that actually load): Qwen3-Next (Gated-DeltaNet hybrid; check size —
  MoE variants may fit in 4-bit on a100 80GB, verify), `ai21labs/AI21-Jamba-Mini-1.5` (or newer
  mini), NVIDIA `Nemotron-H-8B` variants, `tiiuae/Falcon-H1-7B-Instruct`, `Zyphra/Zamba2-7B`,
  RWKV-7 7B ("world"), `NX-AI/xLSTM-7b`, `google/recurrentgemma-9b-it` (Griffin). Prefer
  instruction-tuned variants; note each model's aggregation primitive in the report.
- Record for each: params, active params, attention/state layout, transformers version needed.

**Step 1 — NEW venv (mandatory).** `python3 -m venv /home/tal.gorbunov/projects/Transformer-Optimization/.venv_arch`
(or under /scratch if quota-tight); install: recent `torch` (match cluster CUDA), latest
`transformers`, `accelerate`, `bitsandbytes`, plus per-model extras (`mamba-ssm`,
`causal-conv1d`, `flash-linear-attention`, `rwkv` — only what the chosen models need). Document
every install + version in the report. NEVER activate the shared `.venv` in these jobs.
Smoke-test each model loads + generates 10 tokens (a100 job, `2h_2g`, one job for all smokes,
`outputs/_scratch/arch_smoke/`).

**Step 2 — data.** Text states-only MMRED exists for N=8/16/24/40 (see
`outputs/ladder/text_mmred/` runs and their `run_config.json` for the exact data roots).
Generate N=64 and N=128 with `datasets/mmred/generate_mmred_balanced.py --task steps_in_room
--seq-len {64,128} --per-count <so that ~150 samples> --n-chars 5 --no-render --seed 7
--out-root data/mmred_text_arch` (CPU, `4h_0g`, mem ≤16G). n=150/N is enough.

**Step 3 — the battery.** Reuse the existing text behavior evaluation
(`evaluations/scripts/eval_mmred_text_frames_acc.py` renders states → text frames; the ladder
runs' `run_config.json` show the working invocation) with the SAME prompt template and the
generation reader (not digit-argmax — counts exceed 9). Write a thin model-agnostic wrapper
(`experiments/arch_battery/run_text_battery.py`): loads any HF causal LM in the new venv,
same prompt, greedy, `max_new_tokens=12`, parse final integer. Metrics per model × N: exact
match, MAE, bias, emitted range (clamp signature: p95−p5 of predictions), Spearman(pred, gold),
majority baseline. n=150, seed 2.
**Jobs:** one per model (all N in a loop), `a100-public`, `--qos` spread (`2h_2g` ×2, `24h_1g`,
`4d_1g`), `--mem=96G`, time 1:55 for 7–9B models (bump for MoE). Tail the first job before
launching the rest.

**Step 4 — analysis + figure.** One CSV (`model, arch_class, N, em, mae, bias, range, spearman,
majority`) + a matplotlib figure: EM vs N (log-x), one line per model, softmax dashed /
extensive-state solid, majority band shaded; second panel emitted-range vs N (the clamp).
Output root: `outputs/arch_battery/<ts>/` + create `outputs/arch_battery/INDEX.md`.

---

## Deliverables (both experiments)

1. `outputs/.../report.txt` + `results.csv` per run (every number traceable).
2. `plans/arch_experiments_DRAFT_RESULTS.md` — a RESULTS.md-style draft entry per experiment
   (date, runs/jobs, tables, readings, caveats) for Tal to review and paste; do NOT touch
   RESULTS.md itself.
3. Update the relevant `INDEX.md` files.
4. A final one-screen summary at the end of the tmux session: what ran, headline numbers vs the
   pre-registered bands/predictions, what's blocked, suggested next step.

## Ordering & budget

Exp 2 first (one job, ~1–2 GPU-h, zero new dependencies). Start Exp 1's Step 0/1/2 (CPU) while
Exp 2's job queues. Total GPU budget target ≤ 25 GPU-hours; if a model repeatedly OOMs or needs
exotic kernels that won't build, drop it and note it — 2 hybrids + 2 softmax controls is enough.

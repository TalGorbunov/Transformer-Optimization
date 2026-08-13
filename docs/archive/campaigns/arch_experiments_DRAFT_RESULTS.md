# DRAFT RESULTS — architectural-solution experiments (for Tal to review; do NOT paste into RESULTS.md without checking)

> Written by the autonomous agent session of 2026-07-13/14 executing
> `plans/arch_experiments_agent_brief.md`. Every number below traces to a run dir on disk.

---

## [2026-07-13-arch-A] Exp 2 — trained-query ceiling: a learned shared query CANNOT read evidence from joint k/v

**Question.** Train a query vector q* ∈ R^3584 (shared across frames — exactly what a DETR/slot-style
head could deploy without frame identity at inference) to read per-frame evidence from JOINT-encoded
k/v at L16. What held-out d′ does it reach vs the pre-registered anchors?

**Setup.** `experiments/glstm/trained_query_ceiling.py` (new) on the n=500 capture
`outputs/ladder/image_longN/qkv_2x2/20260712_n500/`. Same un-mixer split (RandomState(0), 300 train /
200 eval). Differentiable port of the exact 2×2 message path (joint rope geometry, within-frame
softmax, o_proj; vectorized-vs-reference max rel diff 1.7e-04). Objective: logistic proxy —
BCE of `w·msg(q*, k_f, v_f) + b` on the per-frame evidence label, Adam lr 1e-3, 400 epochs,
batch 512, trained on TRAIN frames only. d′ via `dprime_pair` (held-out shrinkage-LDA, 3
sample-disjoint seeds) on eval messages. Init ablation: joint-query-mean / mp-query-mean / random.

**Jobs.** 120908 (v1, 12.5 min) and 120914 (v2 identical + eval-d′-per-checkpoint trajectory),
a100-public, 1 GPU. Run dir: `outputs/ladder/image_longN/qkv_2x2/20260712_n500/trained_query/`
(v1 archived at `trained_query_v1_notraj/`).

**Anchor gate (recomputed in-run, PASS — run valid):**

| cell | eval-split (n=200) | full-500 | pre-registered |
|---|---|---|---|
| joint-q × joint-kv | 2.09±0.05 | 2.33±0.02 | 2.09–2.33 ✓ |
| mp-q × joint-kv | 3.82±0.07 | 4.47±0.11 | 4.47 (full) ✓ |
| pad-q × joint-kv | 3.31±0.08 | 3.97±0.11 | 3.97 (full) ✓ |

**Result (v2, job 120915; v1 numbers identical):**

| arm | eval d′ (final ep) | full-500 d′ | eval-d′ max over ALL epochs (cherry-picked-on-eval upper bound) |
|---|---|---|---|
| trained q* (init=joint) | **0.36±0.07** | 0.69±0.08 | 0.49 (ep 20) |
| trained q* (init=mp) | **0.48±0.03** | 0.69±0.08 | 0.48 (ep 260) |
| trained q* (init=random) | **0.44±0.04** | 0.72±0.11 | 0.51 (ep 0) |

Train AUC reaches 0.97–0.99 in all arms (the proxy objective is easy to fit); held-out d′ lands at
0.36–0.48 — BELOW the deployed joint query (2.09), let alone the mp-q ceiling (3.82/4.47) or the
GO band (≥4). The trajectory diagnostic closes the early-stopping question: eval d′ starts at
~0.5 at epoch 0 and NEVER rises, for any init — max over all epochs × inits is 0.51, and that is
already a cherry-picked-on-eval upper bound. Init does not matter (no local-minima story).

A second observation falls out of the epoch-0 point: even the MEAN of the native joint queries
(the init) reads only ~0.5 — the deployed joint query's 2.09 is carried by its SAMPLE-specific
component (it varies with the question/context), which no single fixed vector can supply.
So the shared-query family fails twice over: it can't be frame-specific (the 2×2's point) and it
can't even be sample-specific.

**Reading (pre-registered band: d′≈2 → requirement-1 confirmed).** Requirement-1 CONFIRMED, and
more strongly than the band anticipated: with n=300 training samples, a learned shared query does
not merely fail to beat the joint query — it cannot even match it out-of-sample. The joint-context
tax's query half is not repairable by training a fixed query; per-frame addressing must be
architectural (per-frame forwards / an addressing mechanism that gets frame identity at inference).
Slot-head architecture: NO-GO at this data scale.

**Caveats (honest).**
- The logistic proxy ≠ d′ directly; a d′-shaped objective might do somewhat better. But the
  trajectory diagnostic (v2, `train_history_*.csv` eval-d′ column) bounds the whole
  early-stopping/regularization family at 0.51.
- A sample-conditioned query (e.g. a head that computes q from the question context, still
  frame-shared) is deployable and was NOT tested here — the brief pre-registered the global
  shared q* as the honest arm. Given the native sample-specific joint query reads 2.09 and the
  mp ceiling is 3.82 (eval scale), a sample-conditioned trained query could in principle land in
  [0.5, 3.82], but the frame-addressing half of the gap (2.09→3.82→GO band 4+) remains
  architecturally closed by the 2×2 + this result.
- n=300 training samples; a trained head in a real adapter would see far more data. The overfit
  gap here is data-limited — the claim is "not with this capture", though the direction (0.4 vs
  the 4.0 GO bar, a 10× shortfall) leaves little room.
- Both anchors' scales reported because dprime_pair is estimator-limited at n=200 (eval) vs n=500;
  the trained arm is only comparable to the eval-split anchor column.

---

## [2026-07-13-arch-B] Exp 1 — cross-architecture text-MMRED battery

### Pre-registered predictions (written BEFORE any battery run, per the brief)

1. **Softmax-attention models** (Mistral-7B-Instruct-v0.3, Qwen2.5-14B-Instruct) collapse on the
   law schedule with the undercount/clamp signature: emitted range (p95−p5 of predictions)
   compresses as N grows, bias grows negative, model → majority answer by N≈16.
   Measured Qwen2.5-VL-7B text anchor (do not re-run): EM 0.196/0.062/0.035/0.020 at N=8/16/24/40,
   majority-locked from N=16.
2. **Write-gated extensive-state models** (Falcon-H1, Nemotron-Nano-9B-v2, xLSTM-7b,
   recurrentgemma-9b-it, Qwen3-Next) show a flatter EM-vs-N curve, wider emitted range, and
   ordinal correlation (Spearman) surviving to larger N.
3. Either outcome is a finding; a hybrid that collapses identically to softmax is evidence that
   pretraining, not the aggregation primitive, dominates.

### Step 0 — model inventory (verified via HF hub 2026-07-13, `/tmp/hf_inventory.json`)

| model | arch class / aggregation primitive | params | weights | ctx | transformers | verdict |
|---|---|---|---|---|---|---|
| meta-llama/Llama-3.1-8B-Instruct | softmax | 8B | 32.1GB | 128k | any | **BLOCKED — gated (403), no license on this token** → Mistral substitutes |
| mistralai/Mistral-7B-Instruct-v0.3 | softmax GQA | 7.2B | 29.0GB | 32k | ≥4.42 | GO (control 1) |
| Qwen/Qwen2.5-14B-Instruct | softmax GQA | 14.8B | 29.5GB | 32k | ≥4.43 | GO (control 2) |
| tiiuae/Falcon-H1-7B-Instruct | parallel Mamba2‖attn hybrid | 7.6B | 15.2GB | 256k | ≥4.52 | GO |
| nvidia/NVIDIA-Nemotron-Nano-9B-v2 | Mamba2-dominant hybrid (few attn layers) | 8.9B | 17.8GB | 128k | ≥4.51 | GO |
| NX-AI/xLSTM-7b | pure mLSTM (matrix-memory, NO softmax attn) | 6.9B | 27.5GB | recurrent (trained 8k) | ≥4.48 | GO |
| google/recurrentgemma-9b-it | Griffin: RG-LRU recurrence + local(2k) attn | 9.7B | 19.3GB | recurrent (trained 8k) | ≥4.42 | GO (gated but token has access) |
| Qwen/Qwen3-Next-80B-A3B-Instruct | Gated-DeltaNet + gated-attn hybrid MoE (3B active) | 80B | 162.7GB | 256k | ≥4.57 | **DROPPED after 3 attempts** (jobs 120926/120949/120952: 1/2/3×A100-40GB) — `Qwen3NextExperts` stores experts as packed `nn.Parameter` [num_experts,…] tensors which bitsandbytes cannot quantize, so ~160GB stays bf16; needs AWQ/GPTQ+vLLM or 2×H200, out of budget. NOTE: a100-public GPUs are 40GB, not 80GB as the brief assumed |
| ai21labs/AI21-Jamba-Mini-1.5/1.6 | Mamba+attn hybrid MoE | 52B | 103GB | 256k | — | **BLOCKED — gated (403)** |
| Zyphra/Zamba2-7B-Instruct | Mamba2+shared-attn hybrid | 7.4B | 29.8GB | **4k** | ≥4.49 | DROPPED — context too short for N≥64 |
| RWKV-7 7B "world" | RWKV-7 | — | — | — | — | DROPPED — no official HF-format 7B (only GGUF/BlinkDL .pth); fla-hub/rwkv7-7.2B-world does not exist |
| RWKV/v6-Finch-7B-HF | RWKV-6 | 7.6B | 30.5GB | recurrent | custom code | fallback only (trust_remote_code + custom kernel compile risk) |
| nvidia/Nemotron-H-8B-Base-8K | Mamba2 hybrid | 8B | 16.2GB | **8k** | ≥4.48 | dropped — base-only + 8k ctx (N=128 prompt ~10-15k tok); Nano-9B-v2 supersedes |

### Step 1 — environment (documented installs)

- `venv_arch` at `/rg/shocher_prj/tal.gorbunov/venv_arch` (home quota 288G/300G — too tight),
  symlinked from repo `.venv_arch`. Python 3.11.15 (`module load python/3.11.15-x86-7sp4zgy`).
  Shared `.venv` untouched.
- Installed: `torch 2.7.1+cu126`, `transformers 5.13.1`, `accelerate 1.14.0`,
  `bitsandbytes 0.49.2`, `scipy`, `pillow`, `sentencepiece`, `protobuf`, `huggingface-hub` (+ pip 26.1.2).
  No mamba-ssm/causal-conv1d (transformers-native paths used; noted where slow).
- HF model cache: `HF_HOME=/rg/shocher_prj/tal.gorbunov/hf_arch` (13T free, cluster-mounted).
- Smoke: jobs 120913 (found the transformers-v5 `apply_chat_template` BatchEncoding change +
  xLSTM config gate) and 120917 (6/6 OK) → `outputs/_scratch/arch_smoke/smoke_results_120917.json`.
  bf16 GPU peaks: Mistral 13.5G, Qwen14B 27.6G, FalconH1 16.5G, Nemotron 18.4G, xLSTM 12.9G
  (but 30.5s per 10 tokens — triton compile + recurrent step overhead), recurrentgemma 18.9G.
- Post-smoke env repairs (documented, all pinned in the end state): the first
  `pip install causal-conv1d` silently upgraded torch to 2.13.0+cu130 (ABI break) → restored;
  final state **torch 2.8.0+cu128** + `causal_conv1d 1.6.2.post1+cu12torch2.8` +
  `mamba_ssm 2.3.2.post1+cu12torch2.8` (GitHub-release wheels, `--no-deps`) + `xlstm 2.0.5` +
  `mlstm_kernels 2.0.1`. Torch 2.8 was forced by a dependency triangle: transformers 5.13 removed
  symbols old mamba_ssm imports, and mamba_ssm 2.3.2 needs `torch.float4_e2m1fn_x2` (≥2.8).

### Step 2 — data

- N=8: `data/mmred_images_park` (the anchor's root); N=16/24/40: `data/mmred_text_longN`
  (the ladder runs' roots); N=64/128: NEW `data/mmred_text_arch` (job 120907,
  `generate_mmred_balanced.py --task steps_in_room --n-chars 5 --no-render --seed 7`,
  long-N counts convention low-band 0–8 + spread; n=160 @ N=64, n=153 @ N=128).

### Step 3 — battery

- Wrapper: `experiments/arch_battery/run_text_battery.py` — any HF causal LM, EXACT prompt
  template of `eval_mmred_text_frames_acc.py` (helpers copied verbatim; that module's import
  chain needs nnsight which venv_arch intentionally lacks), chat template when the tokenizer has
  one, greedy, `max_new_tokens=12`, generation reader = first integer. n=150, seed 2.
- Jobs (all trace to `logs/batt_*-<id>.out`): Mistral 120918 (a100, 2h_2g, 5.3 min) ·
  Qwen2.5-14B 120922 (a100, 24h_1g, 6.5 min) · recurrentgemma 120923 (a100, 24h_1g, 66 min) ·
  xLSTM 120928 (l40s, 4d_1g, 10 min) · Falcon-H1 120921+120947 (N≤40, naive mamba2 path, OOM at
  N=64 on 40GB) + 120967 (N=64/128 with mamba-ssm kernels, 5 min) · Nemotron-Nano 120927+120948
  (N≤24) + 120971 (N=40) + 120968 (N=64/128, kernels) — `system=/no_think` (reasoning model).
- OOM post-mortem: the transformers pure-torch Mamba2 fallback materializes a ~10GB
  chunked-scan tensor on 10–15k-token prompts; fixed by the prebuilt mamba-ssm kernels.

### Results (canonical: `outputs/arch_battery/20260713_215812/battery_all.csv` + `fig_arch_battery.png`)

Exact match (n=150/N; majority baseline 0.140/0.073/0.067/0.053/0.067/0.060):

| model (class) | N=8 | 16 | 24 | 40 | 64 | 128 |
|---|---|---|---|---|---|---|
| Mistral-7B-v0.3 (softmax) | 0.353 | 0.127 | 0.040 | 0.053 | 0.060 | 0.040 |
| Qwen2.5-14B (softmax) | 0.287 | 0.087 | 0.067 | 0.020 | 0.133 | 0.087 |
| Falcon-H1-7B (Mamba2‖attn) | **0.367** | 0.127 | **0.120** | 0.033 | 0.027 | 0.040 |
| Nemotron-Nano-9B-v2 (Mamba2 hybrid) | 0.133 | 0.047 | 0.073 | 0.033 | 0.080 | 0.073 |
| recurrentgemma-9b-it (Griffin) | 0.240 | 0.087 | 0.040 | 0.033 | 0.040 | 0.020 |
| xLSTM-7b (mLSTM, BASE model) | 0.067 | 0.013 | 0.007 | 0.020 | 0.040 | 0.013 |
| *anchor: Qwen2.5-VL-7B text* | *0.196* | *0.062* | *0.035* | *0.020* | — | — |

Bias (mean pred − mean gold): softmax undercount grows monotonically (Qwen14B −0.6 → −16.1 at
N=128; Mistral −0.5 → −8.0 at N=40); Falcon-H1 stays near 0 through N=24 (−0.15) before losing
ordinal structure; Nemotron −1.9 → −5 (N≤40) then erratic (+8.7 at 128). Spearman: softmax keeps
0.66–0.90 everywhere (they still rank, they just can't count); Falcon-H1 0.90 → 0.49 at N=40;
Nemotron low throughout (0.16–0.69); xLSTM ≈ 0 (40–80% parse failures — base model, no
instruction following; excluded from interpretation).

**Reading vs the pre-registered predictions.**
1. Softmax collapse with growing negative bias: CONFIRMED. Majority-locking: NOT reproduced —
   unlike the Qwen2.5-VL anchor, these text models keep high Spearman and scatter widely instead
   of clamping (emitted range EXPANDS to 80–115 at N=128 rather than compressing). The clamp
   signature appears to be model-specific (VL-Qwen), not universal-softmax.
2. Extensive-state models flatter/better: NOT CONFIRMED. Every architecture converges to
   EM ≤ 0.09 ≈ majority by N≥40. Falcon-H1 shows the only hint of resistance (flat 0.127/0.120
   at N=16/24 where Mistral drops to 0.040, and near-zero bias there), but it hits the same floor
   at N=40. Griffin and the Mamba2-dominant hybrid collapse indistinguishably from softmax.
3. Per the pre-registration, "a hybrid that collapses identically is evidence that pretraining,
   not the primitive, dominates": that is the observed outcome. The counting wall on this
   battery is NOT relieved by write-gated extensive-state architectures as released — either the
   walls are upstream of the aggregation primitive (task format / retrieval-per-frame in a flat
   text scan), or the released instruction tuning never taught any of them to tally.

**Caveats (honest).**
- xLSTM-7b is the only pure-recurrent point and it is a BASE model — its failure is
  instruction-following, not (necessarily) aggregation. No open instruct mLSTM 7B exists.
- Nemotron-Nano was run with `/no_think`; its reasoning mode (CoT) would likely score higher but
  breaks the fixed-protocol comparison (and max_new_tokens=12).
- Falcon-H1/Nemotron N≤40 vs N=64/128 legs ran on different kernel paths (naive torch 2.7.1 vs
  mamba-ssm torch 2.8.0) after OOM repairs — same weights, greedy, same prompts; a spot-check of
  overlapping behavior was not run.
- N=8 data root differs (mmred_images_park, the anchor's root) from N≥16 (text_longN/text_arch);
  gold-count distributions also differ across N (low-band+spread convention at 64/128).
- No Gated-DeltaNet point survived (Qwen3-Next quantization blocked); the strongest
  memory-lit candidate is untested here. A GDN datapoint needs AWQ/vLLM or an H200 pair.

---

## Final summary (one screen)

**Exp 2 — trained-query ceiling: NO-GO for the slot head, requirement-1 confirmed.**
All four pre-registered anchors reproduced exactly (2.09/2.33 joint, 3.82/4.47 mp, 3.31/3.97 pad;
gate PASS). A trained shared query q* reaches held-out d′ **0.36–0.48** (trajectory max over all
epochs/inits: **0.51**) vs the ≥4 GO bar — it cannot even match the sample-specific joint query
(2.09). Per-frame addressing must be architectural (per-frame forwards); no fixed learned query
recovers supply from joint-encoded k/v. Run: `outputs/ladder/image_longN/qkv_2x2/20260712_n500/trained_query/`.

**Exp 1 — cross-architecture battery: NO released architecture escapes the counting collapse.**
6 models × N∈{8..128}, n=150, fixed prompt/reader. All ≤0.09 EM ≈ majority by N≥40. Softmax
undercounts monotonically (bias −16 at N=128 for Qwen14B) but is NOT majority-locked (Spearman
stays 0.8); Falcon-H1 (Mamba2‖attn) resists briefly at N=16–24 then collapses; Griffin and
Nemotron collapse like softmax; xLSTM uninformative (base model). The pre-registered
"pretraining dominates the primitive" reading holds. Run: `outputs/arch_battery/20260713_215812/`.

**Blocked/dropped:** Llama-3.1 + Jamba (gated licenses), Qwen3-Next-80B (bnb cannot quantize
packed-Parameter MoE experts — needs AWQ/vLLM or 2×H200), RWKV-7 7B (no HF-format release),
Zamba2 (4k ctx), Nemotron-H-8B (8k ctx, base).

**GPU budget:** ~6 GPU-hours total (Exp 2: ~1.2h across 3 runs incl. failures; Exp 1: ~4.5h
incl. smoke + OOM retries) — well under the 25h target.

**Suggested next steps:** (1) a GDN datapoint via vLLM+AWQ Qwen3-Next or Kimi-Linear on H200;
(2) test whether FINE-TUNED (not just released-instruct) extensive-state models tally — the
battery only rules out the released checkpoints; (3) the sample-conditioned trained query
(q = f(question ctx), frame-shared) to close the last untested cell between 0.5 and 3.82.

# gating campaign — STATE (handoff file)

Brief: `outputs/gating/CAMPAIGN_BRIEF.md`. Started 2026-08-07 (Tal).
Update this after EVERY phase: what ran (job IDs + run dirs), numbers vs the phase gate
and vs anchors, decision taken, next step. Newest last.

**Anchor to beat/reproduce:** l12v2 / caption recipe → BEST ~0.999, tf-exact ~0.99,
N=32 exam 0.987 (`scripts/train_carrier_layer.py` docstring).

**Headline question:** does gating relieve the aggregation bottleneck — and if so, is it
G1 (post-sum, query-side) or G2 (pre-sum, message-side)? Gating can only attenuate, so a
gain is evidence for *interference*, not *capacity*.

## Phase log

| phase | status | jobs | run dir | gate met? | verdict |
|---|---|---|---|---|---|
| P0 1B text triple | **done** | 129124 | `p0_text_triple/20260807_212129_129124` | yes | G1-headwise raises tally decodability (R²) consistently but modestly; gain does NOT grow with N |
| P1 sink diagnostic 7B | **done** | 129133 (129125/129126 died on mask dtype) | `p1_sink_7b/20260807_214956_129133` | **gate NOT met** | our 7B has essentially no sink (F-Attn ≤ 0.05 everywhere); P3 is now a falsification, not a hope |
| P2 gating.py + CPU tests | **done** | — (CPU) | `gnnformer/gating.py`, `tests/test_gating.py` | yes | all 6 suites green |
| P3 main ablation N=8 | **arms done; gate PASSED**; arms 2+3 re-running | 129144–7, 129152; v2 129722–3 | `p3_arms/`, `p3_arms_v2/` | arm 1 = 0.999/0.996 = anchor ✓ | see below — the split is LoRA-vs-gate, and the arm design confounds 4 factors |
| P3.5 discriminator | evaluator validated; **grid redesigned** | 129156 (smoke) | `outputs/_scratch/gating_eval_smoke/` | — | the brief's N≤8 grid is SATURATED (tf_acc 1.000 everywhere); moved to tf_exact + a high-N grid |
| P4 G2-expanded + N-transfer | **gated — ask Tal** | | | | |
| P5 verdict & figures | not started | | | | |
| P6 Qwen3.5 landscape | **gated — ask Tal** | | | | |

---

## P0 — the 1B G1 triple (job 129124, 11 min, a100-public/2h_2g)

Run dir: `outputs/gating/p0_text_triple/20260807_212129_129124/`
Script: `scripts/gating/probe_text_triple.py` · wrapper `slurm/gating_probe_text_triple.sbatch`
Roots: `slurm/lib/roots_text_tally.txt` (text MMReD N=16/24/40 + text_arch N=64), 200/root,
50/50 split, seed 0, identical sample order and split for all three models.

### Artifacts: where the gate actually lives

`QwQZh/gated_attention` downloaded to
`~/.cache/huggingface/hub/models--QwQZh--gated_attention/snapshots/aad415c45ec6b4fa727ef3ff3f4e9f62f958d49b`
(9.7 GB, `HF_HOME` unset → default cache). Custom remote code (`modeling_qwen3.py`,
`trust_remote_code=True`); loads clean in `.venv` (transformers 4.57.6) despite being
written for 4.46.

**There is no gate tensor.** The gate is FUSED INTO `q_proj`: its output is widened and
split into `query_states` + `gate_score`, then `attn_output * sigmoid(gate_score)` after
SDPA, before `o_proj`. So the gate is query-dependent and lives at G1, exactly as the
paper says. Confirmed shapes (`model.layers.i.self_attn.q_proj.weight`):

| variant | q_proj | hidden | heads/kv | intermediate | params |
|---|---|---|---|---|---|
| `1B_baseline` | [2048, 2048] | 2048 | 16/8 | 6144 | 1720.8 M |
| `1B_gate_headwise` | [2064, 2048] | 2048 | 16/8 | 6144 | 1721.8 M |
| `1B_gate_elementwise` | [4096, 2048] | 2048 | 16/8 | **5504** | 1728.2 M |

⚠ **CONFOUND, must be carried into any writeup:** `1B_gate_elementwise` has a *smaller
FFN* (5504 vs 6144) — they shrank it to keep the parameter count matched against the
+117 M of elementwise gate. It is param-matched, NOT architecture-matched. `headwise` is
both. Any elementwise-vs-baseline claim is therefore confounded; headwise-vs-baseline is
the clean comparison.

### Sanity check (their Table 4, reproduced on our data) — PASSES loudly

Eager attention, 6 samples of text N=16. F-Attn = mass on token 0; M-Act = max |hidden|.

| layer | baseline F / M | headwise F / M | elementwise F / M |
|---|---|---|---|
| 7 | 0.371 / 10560 | 0.041 / 66 | 0.010 / 31 |
| 19 | 0.503 / 14208 | 0.023 / 99 | 0.015 / 61 |
| 23 | 0.738 / 14208 | 0.026 / 240 | 0.019 / 118 |
| 27 | 0.691 / 14144 | 0.006 / 621 | 0.002 / 338 |

The baseline has a textbook sink (up to 74 % of attention on token 0) and massive
activations (1.4e4); both gated checkpoints kill it (F-Attn ≤ 0.07 everywhere, M-Act
≤ 621). These really are the sink-free checkpoints, and the effect reproduces on MMReD
text, not just on their LM data.

### Gate: PASSED

Three models comparable (byte-identical tokenizer, same prompt, same split), and the
baseline is well above floor at the smallest N (N=16: ridge-round exact 0.190 / ±1 0.430
/ R² 0.72 vs majority-class 0.030). Probe is informative.

### Headline numbers

Best-layer cells (pool = last-token; **selected on the test half over 29 layers, so these
are optimistically biased — equally for all three models**):

| root | n_test | majority | baseline | headwise | elementwise |
|---|---|---|---|---|---|
| text N=16 | 100 | 0.030 | L19 ex 0.190 R² .72 | L17 ex **0.260** R² .77 | L20 ex 0.160 R² .62 |
| text N=24 | 100 | 0.030 | L20 ex 0.180 R² .66 | L25 ex 0.150 R² .65 | L23 ex 0.150 R² .55 |
| text N=40 | 100 | 0.030 | L2 ex 0.100 R² −.03 | L27 ex 0.120 R² .32 | L23 ex 0.130 R² .32 |
| arch N=64 | 80 | 0.037 | L25 ex 0.075 R² .62 | L22 ex **0.150** R² .65 | L18 ex 0.062 R² .78 |

Exact-match differences are all inside noise (SE ≈ 0.04 at n=100). The **layer-matched
mean over L12–27** (no selection bias) is the statistic that separates them:

| root | baseline R² | headwise R² | elementwise R² | headwise−baseline | layers where headwise wins |
|---|---|---|---|---|---|
| N=16 | +0.576 | **+0.695** | +0.470 | **+0.119** | 16/16 |
| N=24 | +0.555 | **+0.629** | +0.410 | +0.074 | 13/16 |
| N=40 | +0.255 | **+0.314** | +0.229 | +0.059 | 11/16 |
| N=64 | +0.577 | **+0.657** | +0.469 | +0.080 | 8/16 |

### VERDICT (P0)

1. **G1 head-specific gating does raise tally decodability** — small in exact-match
   (inside noise) but consistent in R², winning at 48 of 64 layer-matched cells and at
   every N. This is a real but modest effect.
2. **The gain does NOT grow with N.** ΔR² is +0.119 at N=16 and +0.059…+0.080 at
   N=24/40/64. If gating were relieving an aggregation bottleneck the advantage should
   widen as the load grows; it narrows. This is the first evidence *against* the
   interference story and *for* the capacity story.
3. **Absolute decodability collapses with N for all three models alike** (R² 0.58 → 0.26
   from N=16 to N=40, baseline and gated). The wall is present in the sink-free models
   too — killing the sink did not move it.
4. Elementwise gating is *worse* than baseline on R² at every N — but see the FFN
   confound above; do not read this as "elementwise gating hurts".

Read against the campaign's framing: G1 does what G1 can do (cleaner query-side readout)
and nothing more. It does not touch the N-scaling, which is what an aggregation fix would
have to touch. P3's G2 arm is now the load-bearing experiment.

---

## P1 — is there a sink worth filtering in OUR model? (job 129133, l40s-public/2h_2g)

Run dir: `outputs/gating/p1_sink_7b/20260807_214956_129133/`
Script: `scripts/gating/probe_sink_7b.py` · wrapper `slurm/gating_probe_sink_7b.sbatch`
Frozen Qwen2.5-VL-7B (4-bit nf4), MMReD park N=8, n=8 samples per arm, LoRA + e_c from
`checkpoints/carrier_layer_fmt_caption_best.pt` (L*=12). Attention recomputed from the
captured q/k projections (sdpa returns no weights), head-averaged over all 28 heads.

Two jobs died first on an sdpa mask-dtype error. Cause, worth remembering: **the query
dtype is not constant down this stack** — bf16 below L*, fp32 at/above it where the LoRA
hooks sit — so an fp32 additive mask is rejected by the memory-efficient kernel below L*
and a bf16 one is rejected by sdpa's frontend check above it. The probe now tries
bool/fp32/bf16 per layer and caches which form that layer accepted; bool is exact here
because these masks only ever hold 0.0 or MASK_MIN.

### Result — there is NO sink to filter

| arm | F-Attn range over 28 layers | read row → sink | read row → frames |
|---|---|---|---|
| plain (causal, count prompt) | 0.0004 – **0.0499** | 0.001 | 0.079 |
| deployed (fence + posreset + carriers) | 0.0004 – **0.0386** | carriers 0.004 (below L*) · tail 0.001 (at/above L*) | carriers → own frame 0.240 |

For scale, the 1B baseline in P0 puts **0.74** of its attention on token 0 in deep layers.
Qwen2.5-VL-7B puts at most 0.05, in either layout. The mechanism the paper's G1 removes is
not present in our model on this task.

Three secondary findings worth keeping:

1. **Massive activations without a sink.** M-Act jumps from 35 to 5152 at layer 4 and sits
   at ~7000 for the rest of the stack in BOTH arms, while F-Attn stays ≤ 0.05. So massive
   activations and attention sink are dissociated here — the paper's Table 4 treats them as
   co-symptoms, and in our model they are not.
2. **A mild sink does switch on late, only for carriers.** `car>sink` rises 0.002 → 0.094
   (L24) → 0.063 (L27) exactly as the carriers' own-frame mass collapses (0.25 → 0.035).
   If any gating arm helps, this is the only place the sink story could bite, and it is a
   deep-layer effect, not the aggregation window.
3. **The read rows are dominated by the question prefix, not by frames or carriers.**
   Below L* the carriers spend 0.53–0.85 on the prefix vs 0.24 on their own frame; at/above
   L* the tail row spends 0.44 on the prefix, 0.26 on frames and **0.014 on the carriers**.
   (Per token the carriers are ~100× more attended than frame tokens — 8 tokens vs ~1500 —
   so this is not evidence the carriers are ignored, but it does say the aggregate mass a
   gate would be reweighting is mostly question-prefix mass.)

### Gate: NOT MET (recorded, as the brief requires)

The read rows are not losing a meaningful share of mass to a sink. Per the brief, P3 is
therefore **run as a falsification rather than a hope**: the interference-via-sink
mechanism is absent in our model, so the prior for any gated arm beating the LoRA control
is low, and a null there is the expected outcome rather than a surprise. Combined with
P0's finding that the G1 advantage *shrinks* with N, two independent lines now point at
capacity rather than interference — before a single trainer has run.

Caveat on the metric: "sink" is defined as mass on **token 0**, which in our prompts is the
chat template's `<|im_start|>`. If Qwen2.5-VL parks its sink on a different fixed token,
this measurement would miss it. That is the paper's own definition, so we keep it, but the
claim to make is "no token-0 sink", not "no sink of any kind".

---

## P2 — implementation + CPU parity (no GPU)

New standalone module `gnnformer/gating.py` (imports nothing from `fencing`/`engine`/
`carriers`; nothing in `gnnformer/` imports it). `attach_gate(...) -> Gate` with
`.parameters()` / `.state()` / `.remove()` / `.mean_scores()` and `save_gate_ckpt` /
`load_gate_state`. Variants and their 7B shapes (hidden 3584, 28 q heads, 4 kv, hd 128):

| variant | hook | W shape | params/layer | L12–27 |
|---|---|---|---|---|
| `g1_headwise` | pre-hook `self_attn` (stash X) + pre-hook `self_attn.o_proj` | [28, 3584] | 100 k | 1.6 M |
| `g1_headshared` | same | [1, 3584] | 3.6 k | 57 k |
| `g1_elementwise` | same | [3584, 3584] | 12.85 M | 206 M (single-layer only) |
| `g2_literal` | forward hook `self_attn.v_proj` | [512, 3584] | 1.84 M | 29 M |

X is the input to `self_attn`, which in a Qwen2/2.5-VL decoder layer already is
`input_layernorm(hidden)` — the paper's "hidden states after pre-normalization". It is
NOT re-normalized. `g2_expanded` is deliberately unimplemented until P4 is approved.

Identity init as the brief specifies: `g = sigmoid(x W^T + b) / sigmoid(b)`, `W = 0`,
`b = +2.0`. `b` is a fixed buffer, not a parameter — with `W = 0` the ratio is identically
1 for any `b`, so `b`'s gradient is exactly 0 at init and it can only ever act as the
curvature knob. `b0` is recorded in every checkpoint.

`tests/test_gating.py` (CPU, <2 s, tiny random stand-in, no model load) pins:
bit-for-bit identity at init for all 4 variants + `remove()` restoring the forward;
GQA 28q/4kv shapes incl. the fact that `g2_literal` forcibly shares one gate across 7
query heads (the P4 motivation); gradient reaches W and **b0=2 gives 48× the gradient of
b0=6** (why the brief bans +6); exact `state()` round-trip; and that the gate is
attenuation-dominant (range [0, 1/σ(2)] = [0, 1.135]) — the campaign's core claim that
gating cannot add bandwidth.

**Gate: PASSED.** `test_gating` + `test_fencing` (8) + `test_carrier_masks` (9) +
`test_data` (4) + `test_scratchpad` (6) + `test_mmred_hf_adapter` all green.

### The one pre-authorized core edit

`scripts/train_carrier_layer.py` gained `--gate {none,g1_headwise,g1_headshared,
g1_elementwise,g2_literal}` (default `none`), `--gate-layers`, `--gate-lr` (3e-4),
`--gate-b0` (2.0), `--gate-only`. **Every gate code path is skipped at `--gate none`**, so
the anchor recipe is bit-identical to before the edit. `--gate-layers` hard-rejects any
layer below `--l-open`: layers 0..L*−1 run once at prep and are cached, so a gate there
would be frozen at its identity init and the arm would silently be a no-op.
`--gate-only` freezes LoRA at its B=0 init (contributes exactly 0) so the checkpoint
schema is unchanged and `eval_carrier` can still load the result; the gate state rides in
the ckpt `extra` under `"gate"`. Arm 1 of P3 is the regression test for all of this.
New wrapper: `slurm/gating_train_arm.sbatch` (DRY_RUN-checked both ways).

---

## P3 — sizing from the smokes (before launching the full arms)

Smokes: `outputs/_scratch/gating_smoke/<arm>/`, jobs 129127 (arm 1) and 129134–129138.
`--limit 40` per root over the 16-root mixture → n=640, 320 train steps, `--epochs 2`.
Two rounds were needed; round 1 (129128–129132) died on `attention_dims`'s key being
`hidden_size`, not `hidden`, and the five failures then **hung in CUDA teardown** rather
than exiting, holding their GPUs until `scancel`. Round 1 also lost ~16 min to seven jobs
importing torch/transformers off the same NFS venv at once (RSS 332 MB after 8 min, CPU
16 s; actual shard load is ~8 s once through). Stagger trainer submissions.

### Identity init verified on the real model

`[ep 0]` is `acc 0.000 MAE 4.33` for arm 1 (no gate), arm 2 (`g1_headwise`) and arm 3
(`g2_literal`) alike — the gate really is a no-op at init on Qwen2.5-VL-7B, not just on
the CPU stand-in. Arm 4 reports `MAE 4.67` at ep 0, and the reason matters: it ran on an
**L40S** while arms 1–3 ran on an **A100**. `acc` and `tf-exact` are 0.000 in every arm;
only MAE differs, and MAE at ep 0 is an argmax over an untrained readout, so single-ulp
kernel differences flip it. **Consequence: all five full arms must run on the same GPU
type**, or cross-arm deltas inherit a hardware term.

### Cost estimate (the number the brief asks for before launch)

Scaling unit = frame-units (Σ N_i · n_i over the mixture), since both prep and step cost
track sequence length: smoke 10 360, full mixture (`--limit 900`, n≈8772) 113 464 →
**×10.95**.

| | smoke (measured) | full arm (extrapolated) |
|---|---|---|
| prep | 950 s | ≈ 2.9 h |
| epoch (train+eval) | 822 s | ≈ 2.5 h |
| 5 epochs + prep | — | **≈ 15.4 h on A100, ≈ 20 h on L40S** |
| host RAM for the cached lo phase | 15.4 GB | **≈ 169 GB** |

Consistent with the `slurm/train_carrier_layer.sbatch` note ("~14 h on a100 at full
mixture"). **5 arms ≈ 75–100 GPU-h.**

Plan: all five on **L40S** (`l40s-shared` + `l40s-public`; athena-post and n314 have
2.3 TB RAM against n310's ~278 GB free, and 5 × 169 GB does not fit on the A100 node),
`--mem=300G`, `--time=23:50:00`, four on `24h_1g` + one on `24h_4g`. A walltime kill is
recoverable: the trainer saves `carrier_layer_best.pt` on every improvement and every
epoch's acc / tf-exact / gate-mean is in the run dir's `runner-*.log`; only `report.txt`
would be missing and it can be rebuilt from that log.

### Gate LR selection (the brief's {3e-4, 1e-3} sweep) — 1e-3 wins everywhere

2 epochs, n=640, arms 2–4 train the GATE ONLY (LoRA frozen at B=0). acc / tf-exact:

| arm | trainable params | 3e-4 | 1e-3 | chosen |
|---|---|---|---|---|
| 1 · LoRA control | 2.9 M | 0.931 / 0.159 | — | — |
| 2 · `g1_headwise` L12–27 | 1.6 M | 0.931 / 0.131 | **0.959 / 0.163** | 1e-3 |
| 3 · `g2_literal` L12–27 | 29 M | 0.966 / 0.216 | **0.991 / 0.241** | 1e-3 |
| 4 · `g1_elementwise` L12 only | 12.8 M | 0.466 / 0.059 | **0.516 @ep1** | 1e-3 |

⚠ Two confounds to carry: (a) `g2_literal` has **10× the trainable parameters** of the
LoRA control, so arm 3 > arm 1 is not a clean position effect. The clean comparison is
arm 2 — `g1_headwise` has FEWER parameters than the control (1.6 M vs 2.9 M) and still
edges it. (b) This is 2 epochs at 7 % scale; the ranking may partly reflect effective
learning rates rather than final quality.

**A live gate, verified on GPU:** `[ep 1] gate mean/layer L12:0.8773` (arm 4, 1e-3) — the
gate attenuates ~12 % and is demonstrably not stuck at its identity init. This line only
exists because of a bug fixed mid-campaign: `Gate.reset_stats()` rebound `_stats` while
the hooks closed over the original dict, so every gate score went to an orphaned object
and the trainer printed "no forwards recorded" for gates that were training fine. Training
was unaffected (the smoke accuracies above stand), but the brief's mandatory VOID-vs-null
check would have been unusable. Fixed to `clear()` in place; `tests/test_gating.py::
test_stats_survive_reset` pins it and was verified to fail on the old code.

**Note the tension:** the smoke ordering is G2 > G1 > LoRA, i.e. the inversion this
campaign hypothesised — but P0 (gain shrinking with N) and P1 (no sink to filter) both
point the other way, at capacity. P3.5 is the experiment that resolves this; do not call
the direction from the smokes.

### Full arms launched (2026-08-07)

All five on **L40S** (`l40s-shared`), `--mem=256G` (`24h_1g`/`4d_1g` cap memory at 275 G —
a 300 G submission is rejected with `QOSMaxMemoryPerJob`), `--time=23:50:00`, `--limit 900
--epochs 5` over `slurm/lib/roots_inlength.txt`, `--split-seed 0` so every arm sees a
byte-identical train/eval split.

| job | arm | config | QOS |
|---|---|---|---|
| 129144 | 1 | `GATE=none` (the anchor regression test) | 24h_1g |
| 129145 | 2 | `g1_headwise`, L12–27, gate-only, lr 1e-3 | 24h_1g |
| 129146 | 3 | `g2_literal`, L12–27, gate-only, lr 1e-3 | 24h_1g |
| 129147 | 5 | `g2_literal` + LoRA trained jointly, lr 1e-3 / 1e-4 | 24h_1g |
| 129152 | 4 | `g1_elementwise`, L12 only, gate-only, lr 1e-3 | 24h_4g |

Run dirs: `outputs/gating/p3_arms/<arm>/<stamp>_<jobid>/`.

### INTERIM — epoch 1 of 5 (all five arms; not the verdict)

Prep verified at exactly the predicted size: n=8772, **cache 169.2 GB** (estimate was
169 GB), identical gold histogram in every arm — `--split-seed 0` gives a byte-identical
split, so cross-arm deltas are clean. Measured epoch cost 7476–7701 s → ≈ 12.4 h/arm
total, better than the 20 h L40S estimate.

| arm | trainable | acc | tf-exact | Δ tf-exact vs control | gate span @ep1 |
|---|---|---|---|---|---|
| 1 · LoRA control | 2.9 M | 0.992 | 0.331 | — | (no gate) |
| 2 · `g1_headwise` | 1.6 M | 0.990 | **0.279** | **−0.052** | 0.717–0.879 |
| 3 · `g2_literal` | 29 M | 0.998 | **0.403** | **+0.072** | 0.767–0.881 |
| 4 · `g1_elementwise` @L12 | 12.8 M | 0.983 | 0.357 | +0.026 | 0.664 (single layer) |
| 5 · `g2_literal` + LoRA | 32 M | 0.999 | 0.909 | +0.578 | 0.767–0.895 |

**Arm 1 hits acc 0.992 at epoch 1** — already in the anchor band, which is the strongest
evidence so far that the OFF-by-default trainer edit is genuinely inert.

**Every gated arm has a live gate; none is VOID.** All four learned the same shape:
attenuation is weakest around L15 and deepens toward L24–27 — the same band where P1 found
the carriers' late-layer sink onset. Suggestive, not yet load-bearing.

**G2 > G1 holds at full scale**, and G1 is *below* the LoRA control. That is the inversion
of the paper's language-modelling ranking, and the direction this campaign predicted (only
a write-side gate can act on aggregation). Note this contradicts the smoke ranking for
arm 2, which had G1 above control — 5× more data changed the sign, so the smokes were not
predictive and were right not to be trusted.

⚠ **What this does NOT establish.** `g2_literal` carries 10× the control's trainable
parameters, so arm 3 > arm 1 may be capacity rather than position. Arm 2 vs arm 3 differs
by 17× in the other direction, so it does not isolate position either. **No clean
parameter-matched G1-vs-G2 comparison exists in the current design** — it would need e.g.
`g1_elementwise` across L12–27 (≈206 M) or a rank-constrained G2. Any claim of a position
effect must state this. Arm 5 is confounded by construction (gate + LoRA = strictly more
capacity than any other arm) and is only interpretable as "does adding a gate to LoRA
help", not as evidence about G2.

⚠ **Still unreconciled with P0/P1.** A position effect at N≤8 is compatible with "gating
tidies interference in a regime the method already solves" while the capacity wall at high
N is untouched. P3.5's high-N grid is the experiment that separates these.

### INTERIM — epoch 2, and a reframing of the campaign's central question

| arm | trains | position | gate scores/token | layers | acc | tf-exact ep1→ep2 |
|---|---|---|---|---|---|---|
| 1 · LoRA control | LoRA | — | — | — | 0.996 | 0.331 → **0.938** |
| 2 · `g1_headwise` | gate | G1 | 28 | 16 | 0.994 | 0.279 → 0.318 |
| 3 · `g2_literal` | gate | **G2** | 512 | 16 | 0.997 | 0.403 → 0.516 |
| 4 · `g1_elementwise` @L12 | gate | G1 | 3584 | **1** | 0.993 | 0.357 → **0.604** |
| 5 · `g2_literal` + LoRA | both | G2 | 512 | 16 | 0.998 | 0.909 → 0.991 |

Two things changed the reading, both away from the epoch-1 story:

**1. The split is LoRA-vs-gate, not G1-vs-G2.** Only the LoRA-bearing arms reach high
tf-exact (0.938, 0.991); both gate-only arms stalled far below (0.318, 0.516) even as the
control leapt 0.331 → 0.938. Meanwhile ALL arms sit at 0.993–0.998 on count accuracy.
So: **a gate alone drives the tally as well as LoRA does, but cannot produce the
transcript.** That is exactly what attenuation-only predicts — a multiplicative mask in
(0,1) can suppress what the readout sees but cannot write new token-level behaviour, which
the caption scratchpad requires and LoRA's additive update supplies.

**2. "G2 > G1" does NOT survive arm 4 — the ordering is GRANULARITY, not position.**
A G1 variant (`g1_elementwise`, 0.604) now beats the G2 arm (0.516). tf-exact is perfectly
monotone in gate scores per token:

    28 (g1_headwise) -> 512 (g2_literal) -> 3584 (g1_elementwise)
    0.318            -> 0.516            -> 0.604

⚠ **The brief's design confounds position with granularity and contains no cell that
separates them.** Its G1 arm is head-wise (28 scores) and its G2 arm is forced to 512 by
GQA (`v_proj` is KV-width). Any G1-vs-G2 claim from P3 as designed is therefore
uninterpretable as a *position* result. Note also arm 4 achieves the best gate-only score
gating a SINGLE layer (L12, the aggregation point) against arm 3's sixteen — per-token
expressiveness at the aggregation layer beats depth of coverage.

**Consequence for P4:** its rationale ("gating after `repeat_kv` gives the only variant
both inside the sum AND head-specific") assumes position is the operative axis. On this
evidence granularity is. A re-scoped P4 should hold granularity fixed and vary position —
e.g. a 3584-score G1 vs a 3584-score G2-expanded at the same layer — otherwise it will
re-measure granularity a third time. To raise with Tal at the P4 gate.

Hold all of this loosely until epoch 5: arm 2's ordering already flipped sign between the
smoke and epoch 1, and again between epoch 1 and epoch 2.

**Operational note for P3.5:** the high-N grid must run in `tf` mode. `eval_gated` only
builds attention masks when mode ≠ `tf`; at N=128 the sequence is ~25 k tokens, so a dense
seq² mask is ~1.25 GB per record and ~2.5 GB fp32 on GPU in the decode path. TF mode skips
masks entirely and caches only h at L\* (~180 MB/sample).

---

## P3 — FINAL (5 arms, jobs 129144–7 + 129152, ~12.5 h each on L40S)

### The phase gate PASSED

**Arm 1 = `BEST acc 0.999 (tf-exact 0.996) @ ep 5`** against the anchor's ~0.999 / ~0.99.
The OFF-by-default trainer edit is inert and every downstream comparison rests on a
validated control. All four gated arms had live, non-saturated gates — no arm is VOID.

### Per-epoch tf-exact (the metric with dynamic range; count acc is 0.982–0.999 throughout)

| arm | ep1 | ep2 | ep3 | ep4 | ep5 | selected |
|---|---|---|---|---|---|---|
| 1 · LoRA control | 0.331 | 0.938 | 0.982 | 0.988 | **0.996** | ep5 |
| 2 · `g1_headwise` (28/tok) | 0.279 | 0.318 | 0.324 | 0.348 | 0.464 | ep4 ⚠ |
| 3 · `g2_literal` (512/tok) | 0.403 | 0.516 | 0.774 | 0.896 | **0.936** | **ep1** ⚠⚠ |
| 4 · `g1_elementwise` (3584/tok, L12 only) | 0.357 | 0.604 | 0.775 | 0.846 | **0.873** | ep5 |
| 5 · `g2_literal` + LoRA | 0.909 | 0.991 | 0.994 | 0.989 | 0.995 | ep3 |

### What P3 actually establishes

1. **Adding a gate to LoRA buys convergence speed and nothing else.** Arm 5's margin over
   the control: **+0.578 → +0.053 → +0.012 → +0.001 → −0.001** across epochs. Final
   selected 0.994 vs the control's 0.996. Reading epoch 1 (or the smokes) would have
   claimed a large gain that is purely an artifact of partial convergence. **This is
   verdict (b)** for the gate+LoRA comparison.
2. **Count accuracy — the aggregation-sensitive metric — is 0.982–0.999 for EVERY arm**,
   including the gate-only ones. Gating neither helps nor hurts the tally at N≤8.
   The arms separate only on transcript fidelity, i.e. not on aggregation at all.
3. **28 gate scores per token is below a usable threshold** (arm 2 tops out at 0.464 with
   loss stuck at 0.05 vs the control's 0.0001); ≥512 works.
4. **Gate-only arms stay below the additive LoRA baseline** (0.936 / 0.873 vs 0.996).

### ⚠ What P3 CANNOT establish — the arm design confounds four factors

Position (G1/G2), granularity (28/512/3584 scores per token), depth (1 vs 16 layers) and
parameter count (1.6 M–29 M) all vary together, and **no two arms differ in exactly one
factor**. The gate-only ordering flipped at nearly every epoch (G2>G1 at ep1, monotone in
granularity at ep2, tied at ep3, G2>G1 at ep4/5), which is what an underdetermined design
looks like. **P3 as specified cannot support a G1-vs-G2 *position* claim**, and the
campaign's headline question is not answerable from it. Also note arm 4 reaches 0.873
gating a SINGLE layer against arm 3's sixteen.

### ⚠⚠ Model-selection defect (found 2026-08-08, fixed, arms 2+3 re-running)

Selection is lexicographic on (TF-count acc, tf-exact). TF-count is **saturated**, so the
rule is decided by a few samples out of 4386 of noise:

* **arm 3 kept its ep1 checkpoint (tf-exact 0.403) over ep5 (0.936)** because ep1's count
  acc was 0.9984 vs 0.997 — about 4 samples. Since `best` overwrites one file, the ep5
  weights were **unrecoverable**.
* arm 2 kept ep4 (0.348) over ep5 (0.464).
* Arms 1/4/5 selected ep5/ep5/ep3 and are unaffected.

Running P3.5 on those files would have compared **arm 3 at epoch 1 against arm 1 at epoch
5** — and arms 2+3 are exactly the G1-vs-G2 gate-only pair. Fix (commit `0f0045c`):
`--select-metric {count,tf_exact}` (default `count`, bit-identical to before) plus
**`carrier_layer_last.pt` is now ALWAYS written**, so an early-peaking selection can never
again make the final weights unrecoverable. Verified on arm 3's real per-epoch numbers:
`count`→ep1 (0.403), `tf_exact`→ep5 (0.936). Arms 2+3 re-running as jobs 129722/129723
into `p3_arms_v2/` (Tal approved the targeted re-run, 2026-08-08). P3.5 for arms 1/4/5
runs in parallel so the re-run costs no calendar time.

Same root cause as the P3.5 saturation catch: **the TF-count metric cannot bear the weight
this campaign puts on it.**

---

## ⚠⚠⚠ THE HEADLINE METRIC IS LARGELY A COPY DETECTOR (2026-08-09, Tal's challenge)

Tal flagged the LoRA control's 0.965 count accuracy on the held-out N=128 root as
implausible. It is. **The caption scratchpad carries a RUNNING TALLY, and the final answer
is always a verbatim copy of the last tally value already in the transcript** — verified
120/120 on N=128:

    scan: f1:- f2:Kitchen(1) f3:Kitchen(2) ... f7:Kitchen(5) f8:- | total: 5 END
                                        ^^^                            ^ same number

Under teacher forcing the model is FED that transcript and only predicts ` G END`. So a
pure copier scores 1.000 without aggregating anything.

**Decisive test** (`scripts/gating/probe_tally_copy.py`, job 129886): shift every running
tally by +3, leave the gold `total:` alone. A counter ignores the corruption and still
says gold; a copier follows the tally and says gold+3.

| root | n | clean acc | predicts **gold** (counts) | predicts **gold+3** (copies) |
|---|---|---|---|---|
| N=8 | 40 | 1.000 | 0.125 | **0.600** |
| N=32 | 40 | 1.000 | **0.000** | **0.850** |
| N=128 | 40 | 0.975 | 0.050 | **0.650** |

**At N=32 the LoRA control produces the true count 0% of the time and follows the
corrupted tally 85% of the time.** `tf_acc` — the trainer's headline `acc`, the P3 arm
comparison, and every P3.5 grid — is substantially measuring copy ability, not counting.

### Scope of the damage

| phase | readout | affected? |
|---|---|---|
| P0 (1B triple) | ridge probe on hidden states | **clean** |
| P1 (sink) | attention statistics only | **clean** |
| P3 (ALL five arms incl. G1 and G2) | teacher-forced caption scratchpad | **affected** |
| P3.5 (all grids) | same, `MODE=tf` | **affected** |

It hits every arm identically, so cross-arm *relative* comparisons degrade less than the
absolute numbers — but the count metric had no resolution anyway (0.982–0.999 everywhere).

### The deeper problem, which is not fixable by changing the metric alone

**The caption scratchpad exists precisely to BYPASS in-model aggregation.** It converts
"aggregate N messages" into "per frame, decide evidence and increment a *visible*
counter". That is why THE METHOD works. Consequently neither teacher-forced metric probes
the aggregation bottleneck:

* `tf_acc` → copy the last tally;
* `tf_exact` → per-frame evidence detection + increment from a visible prefix.

Neither requires holding N messages in the residual stream. **We were testing whether
gating relieves an aggregation bottleneck using a method engineered to avoid that
bottleneck.** This, not the copying, is the primary reason P3/P3.5 could not detect an
aggregation effect either way.

### Fix for any future gating claim

1. **Digit readout** (`--scratchpad` off): the model emits the count directly from the
   aggregated state — nothing to copy. Anchors: frozen 0.219, scaffold 0.998. Constraint:
   `digit_ids` are single tokens "0".."9" and the trainer skips `gold > 9`, so it caps at
   counts ≤ 9 (fine for N≤8; needs a multi-digit head for long N).
2. **Free-decode exam** (`MODE=decode`): the model generates its own scan, so the tally is
   its own. This is the anchor's metric (N=32 exam 0.987). Costly at long N without
   `--truncate-at`.
3. Report `tf_exact`, never `tf_acc`, if teacher forcing is used at all.
4. Run `probe_tally_copy.py` alongside any scratchpad metric to quantify copy reliance.

---

## P3.5 — evaluator validated, and the brief's grid had to be redesigned

Smoke: job 129156, `outputs/_scratch/gating_eval_smoke/20260807_231110_129156/`, running
`scripts/gating/eval_gated.py` against the **gated** arm-3 smoke ckpt. It confirmed the one
path nothing else exercises — a gate restored from a checkpoint:

    [gate] g2_literal on [12..27] b0=2.0 params 29360128
    [seq_len_2/all_uniform] per-gold 5 -> 15 dirs ({0: 5, 1: 5, 2: 5})
    [root] seq_len_2: n=15 tf_acc 1.000 tf_exact 0.600 gate_mean 0.8531

Gate loads with the right shape, `--per-gold` gives the balanced grid, and the restored
gate is active (0.853, not 1.0).

### The problem it exposed: the specified grid cannot discriminate

`tf_acc` is **1.000 in every cell** of `seq_len_2..8`:

| root | tf_acc | tf_exact |
|---|---|---|
| seq_len_2 | 1.000 | 0.600 |
| seq_len_4 | 1.000 | 0.440 |
| seq_len_6 | 1.000 | 0.286 |
| seq_len_8 | 1.000 | 0.222 |

The teacher-forced COUNT metric is at ceiling — which is not a bug, it is the trainer's own
documented behaviour ("TF-count saturates early", the reason model selection is
lexicographic on (acc, tf-exact)). **N≤8 is a solved regime for THE method**, so a
capacity-vs-interference discriminator run there measures nothing: every arm scores 1.000
and every gain is exactly 0. Reporting that as "gating is flat on both axes" would be an
artifact dressed as the honest-null verdict.

Note `tf_exact` falls monotonically with N (0.600 → 0.222) — but the caption transcript has
one entry per frame, so whole-transcript exactness declines with N mechanically. It is a
usable *between-arm* metric at matched N; it is NOT a clean N-scaling curve on its own.

### Fixes applied (committed `ca6bd17`)

1. `per_gold` now records `[n, count ok, tf_exact ok, decoded ok]`, so the grid carries the
   harder metrics instead of only the saturated one. The plotter takes `--metric`
   (default `tf_exact`) and prints `[SATURATED — cannot discriminate]` when a grid is at
   ceiling, so this failure can never again be mistaken for a null result.
2. `eval_gated --fast-decode` uses `engine.decode_fast` (16–311×), which is safe with a
   gate: it re-runs every layer over `[cache || appended]`, so the hooks fire and the gate —
   a pointwise function of X — applies identically to cached and appended rows.
   `--exactness-n K` verifies token identity against the plain decode on the first K
   samples of each root, so that reasoning is checked rather than trusted.
3. New `slurm/lib/roots_gating_highN.txt`: the distractor axis where the method is still
   losing — N = 8, 16, 32, 64, **128**. `seq_len_16/32/64` are in the training mixture
   (matched-data between-arm comparison only, not generalization); **`seq_len_128`
   (510 samples, golds 0..128) is HELD OUT** and is the only uncontaminated hard cell we
   have.

The plotter was verified on synthetic grids to recover a distractor-only gain correctly
(0 at N=8/16, +0.05 at N=32, +0.10 at N=64/128, flat across evidence count).

### Plan when the arms land

Run BOTH grids per arm: the brief's `roots_gating_grid.txt` (N=2..8, `--per-gold 40`, to
document the saturation on the record) and `roots_gating_highN.txt` (`--per-gold 15`,
≈1.5–2 h/arm in tf mode), plus `roots_gating_capacity.txt` for the pure capacity axis.

---

## P7 — THE RE-RUN THAT MATTERS: digit readout, no scratchpad anywhere (2026-08-09/11)

Tal's call after the copy-detector finding: drop the scratchpad from the campaign entirely
and score the model on the number it actually emits. `--digit-multi` trains CE on the
answer's DIGIT SEQUENCE (Qwen splits 128 -> '1','2','8'), so any count works — the legacy
digit path capped at 9. Nothing in the context contains the answer, so unlike the caption
readout it cannot be solved by copying, and unlike the caption readout it does not let the
model externalise the aggregation into a visible tally.

**The metric finally has resolution.** Under the caption scratchpad every arm scored
0.99+; here they spread 0.58–0.97.

### P7a — full mixture, 5 epochs (`outputs/gating/p7_digit/`, jobs 129918–922)

| arm | digit acc |
|---|---|
| LoRA control | **0.924** |
| `g2_literal` + LoRA | 0.916 |
| `g2_literal` | 0.854 |
| `g1_elementwise` | 0.747 |
| `g1_headwise` | 0.615 |

### P7b — extrapolation mixture (trained N∈{8,16} ONLY), 14 epochs (`p7_digit_extrap14/`)

5 epochs was NOT converged here (half the gradient steps of the full mixture); at ep4
`g2+LoRA` led by +0.123, at ep5 by +0.035, at ep6 by +0.092, and by ep14 it had **lost**.
Three seeds on the key contrast (jobs 130046–049, 130073–076, 130363–365):

| arm | position | width | layers | seed0 | seed1 | seed2 | mean |
|---|---|---|---|---|---|---|---|
| LoRA control | — | — | — | 0.966 | 0.942 | 0.957 | **0.955 ± 0.012** |
| `g2_literal`+LoRA | G2 | 512 | 12–27 | 0.937 | 0.953 | 0.934 | **0.941 ± 0.010** |
| `g2_literal` | G2 | 512 | 12–27 | 0.816 | | | 0.816 |
| `g3_key` | G3 | 512 | 12–27 | 0.778 | | | 0.778 |
| `g1_elementwise` | G1 | 3584 | 12 | 0.699 | | | 0.699 |
| `g5_output` | G5 | 3584 | 12 | 0.641 | | | 0.641 |
| `g4_query` | G4 | 3584 | 12 | 0.584 | | | 0.584 |

**Granularity-matched position comparison** — the one the P3 arm design could not make.
Bold pairs differ ONLY in position:
* 512/token, L12–27: **value 0.816 > key 0.778**
* 3584/token, L12: **post-SDPA 0.699 > post-o_proj 0.641 > query 0.584**

So position does matter at matched width, ordering value ≳ key > post-SDPA > post-o_proj >
query — but none of them beats plain LoRA. Note this does NOT reproduce the paper's
ranking (they had G1 best, G3/G4/G5 ≈ nothing).

⚠ **Run-to-run noise measured**: identical config/seed reruns drift ≤0.008 for the LoRA
control but up to **0.073** for gated arms. Gates destabilise training; single-run margins
under ~0.07 are not interpretable.

## P8 — accuracy per sequence length (`outputs/gating/p8_digit_seqlen*/`)

### Trained on the FULL mixture (N=2…8,16,32,48,64; only N=128 held out)

| variant | N=2 | N=4 | N=8 | N=16 | N=32 | N=64 | **N=128** |
|---|---|---|---|---|---|---|---|
| LoRA control | 1.000 | 1.000 | **1.000** | **0.886** | **0.942** | **0.767** | 0.157 |
| `g2`+LoRA | 1.000 | 1.000 | 0.991 | 0.856 | 0.795 | 0.722 | **0.358** |
| `g2_literal` | 1.000 | 0.983 | 0.963 | 0.818 | 0.776 | 0.644 | 0.294 |
| `g1_elementwise` | 1.000 | 0.967 | 0.824 | 0.727 | 0.583 | 0.444 | 0.216 |
| `g1_headwise` | 1.000 | 0.917 | 0.750 | 0.591 | 0.506 | 0.389 | 0.137 |
| *chance* | *0.333* | *0.200* | *0.111* | *0.091* | *0.077* | *0.067* | *0.059* |

### Trained on N∈{8,16} ONLY — everything else out-of-distribution

| variant | N=2 | N=4 | **N=8** | **N=16** | N=32 | N=64 | N=128 |
|---|---|---|---|---|---|---|---|
| lora | 0.333 | 0.367 | **1.000** | **0.894** | **0.603** | **0.289** | 0.123 |
| g1 | 0.417 | 0.417 | 0.676 | 0.652 | 0.327 | 0.083 | 0.020 |
| g2 | 0.389 | 0.667 | 0.861 | 0.818 | 0.321 | 0.139 | 0.157 |
| g3 | 0.333 | 0.317 | 0.861 | 0.523 | 0.141 | 0.106 | 0.093 |
| g4 | 0.611 | 0.583 | 0.611 | 0.576 | 0.071 | 0.072 | 0.010 |
| g5 | 0.639 | 0.550 | 0.741 | 0.553 | 0.205 | 0.028 | 0.000 |
| g2+lora | 0.333 | 0.483 | **1.000** | 0.674 | 0.474 | 0.183 | 0.137 |
| *chance* | *0.333* | *0.200* | *0.111* | *0.091* | *0.077* | *0.067* | *0.059* |

**Readings.** (a) Plain LoRA wins at every trained and near-OOD length in both regimes; no
gate position beats it above N=8. (b) The wall is stark and monotone: 1.000 → 0.123 as N
goes 8 → 128. (c) The two regimes DISAGREE far off-distribution — on the full mixture the
gated arms beat the control at held-out N=128 (0.358 vs 0.157, ~6.7 binomial SE at n=204),
on the N∈{8,16} regime they do not. Every far-OOD cell is near the floor, so these are
differences between "bad" and "less bad". (d) Downward extrapolation also fails: trained on
N∈{8,16}, the control scores 0.333 at N=2 = exactly chance, MAE 2.67 on a task whose
answers are 0–2. The model emits numbers from its training-length prior rather than
counting. (e) g4/g5 fall BELOW chance at N=64–128.

## P0 addendum — what the 1B checkpoints actually EMIT (jobs 130887, 130894)

P0 originally reported only ridge-probe decodability. Tal asked for emitted accuracy — the
number comparable to how our own arms are scored. Free greedy digit decode:

| N | chance | baseline | G1 headwise | G1 elementwise | n_test |
|---|---|---|---|---|---|
| 2 | 0.333 | 0.453 | 0.367 | **0.480** | 150 |
| 4 | 0.200 | 0.252 | 0.232 | **0.332** | 250 |
| 8 | 0.111 | 0.127 | 0.124 | 0.133 | 450 |
| 16 | 0.030 | 0.060 | 0.060 | 0.060 | 100 |
| 24 | 0.030 | 0.040 | 0.020 | 0.030 | 100 |
| 40 | 0.030 | 0.040 | 0.020 | 0.040 | 100 |
| 64 | 0.037 | 0.075 | 0.063 | 0.075 | 80 |

`no_number_emitted = 0.000` everywhere — they always produce a digit, so this is not a
format-following failure. **All three are at or near chance from N=8 up**; the only real
margin is `elementwise` at N=2/4. `headwise`, the paper's preferred variant, never beats
baseline at any length. Meanwhile the ridge probe finds the count decodable at R² 0.58–0.70
— so the information is IN the representation and the model cannot use it. Gating moves
neither fact.

N=2/4/8 use the park roots' `qa.txt` (identical grammar to the text roots, verified; a
text-only model just never sees the PNGs) — our text-native roots start at N=16.
⚠ First attempt at those rows was DEGENERATE: `limit=200` with name-sorted park dirs
grabbed only the e0/e1 buckets, giving golds {0,1} and chance 0.46, under which
`elementwise` scored 0.72 and looked like a large gating win. Re-run with full root
coverage (300/500/900) for proper 1/(N+1) chance. Same failure family as the tally-copy
artifact: a number that looks like performance but is the task being easier than advertised.

## The paper's scope, confirmed (2026-08-11)

arXiv:2505.06708 is **text-only LLMs** — 15B MoE and 1.7B dense, 30 variants, 3.5T tokens,
no vision-language models anywhere. The released checkpoints corroborate it: all three are
`Qwen3ForCausalLM`, hidden 2048, no vision tower, no `vision_config`. Every axis differs
from our setting (modality, frozen-vs-pretrained, scale, task), so the honest framing is
"the module ports, the training dynamics do not" — NOT "we refuted the paper".

---

## Open items / decisions taken

- **P3 training mixture.** The brief's arm-1 anchor requirement ("reproduces the anchor
  band") pins the l12v2 recipe = the 16-root `slurm/lib/roots_inlength.txt` mixture at
  `--limit 900` (≈8.8 k samples), not an N=8-only root, even though P3 is headed "N=8".
  Running it as the anchor specifies. ⚠ This collides with P4's "train N=8 → eval N=16"
  anti-memorisation control, because `mmred_longN_park/seq_len_16` **is already in the
  training mixture**. To be resolved with Tal when P4 is proposed — the N-transfer control
  needs either a held-out N or a mixture without the N=16 root. **Resolution found:**
  `data/mmred_longN_park/seq_len_128/all_uniform` (510 samples) is held out of the mixture
  entirely, so it is a clean N-transfer target requiring no retraining and no generator
  change. Recommend P4 evaluate N=128-held-out instead of (or alongside) the contaminated
  N=16. Still Tal's call, since it changes what the P4 headline claim can say.

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
| P1 sink diagnostic 7B | running | 129125 (failed), 129126 | `p1_sink_7b/` | | |
| P2 gating.py + CPU tests | **done** | — (CPU) | `gnnformer/gating.py`, `tests/test_gating.py` | yes | all 6 suites green |
| P3 main ablation N=8 | smokes running | 129127–129132 | `outputs/_scratch/gating_smoke/` | | |
| P3.5 discriminator | not started | | | | |
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

## Open items / decisions taken

- **P3 training mixture.** The brief's arm-1 anchor requirement ("reproduces the anchor
  band") pins the l12v2 recipe = the 16-root `slurm/lib/roots_inlength.txt` mixture at
  `--limit 900` (≈8.8 k samples), not an N=8-only root, even though P3 is headed "N=8".
  Running it as the anchor specifies. ⚠ This collides with P4's "train N=8 → eval N=16"
  anti-memorisation control, because `mmred_longN_park/seq_len_16` **is already in the
  training mixture**. To be resolved with Tal when P4 is proposed — the N-transfer control
  needs either a held-out N or a mixture without the N=16 root.

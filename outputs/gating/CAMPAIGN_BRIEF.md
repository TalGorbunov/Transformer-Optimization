# Gated attention (G1/G2) vs the aggregation bottleneck — agent brief

> **You are executing a pre-approved campaign** (Tal, 2026-08-07). Purpose: determine
> whether the gating mechanism of *Gated Attention for Large Language Models*
> (arXiv:2505.06708, NeurIPS 2025 oral, Qwen team) relieves the aggregation
> (over-squashing) bottleneck this thesis studies — and if so, **which position** does it:
> `G1` (gate AFTER the attention sum, query-dependent) or `G2` (gate INSIDE the sum, on
> the value/message path). The deliverable is a measured VERDICT plus, if warranted, a
> trained G2 adapter. Work phase by phase with gates; a phase that fails its gate STOPS
> the branch below it. Read `CLAUDE.md` first; its rules apply except where this brief
> pre-authorizes. Keep `outputs/gating/STATE.md` current after every phase — it is the
> handoff file.

## The one-paragraph thesis of this campaign

Gating is **multiplicative masking with scores in (0,1)**: it can attenuate, never
amplify. So it cannot add bandwidth to a d_k=128 aggregation channel. If our bottleneck
is **capacity** (N messages do not fit), gating must fail. If it is **interference**
(they fit, but distractors crowd the readout), gating can help — and then position
matters: `G1` filters what the query *reads* out of an already-collapsed sum; `G2`
filters what each source token *writes* before the sum. Only `G2` can act on
over-squashing. The paper ranks G1 > G2 on language-modelling loss; our hypothesis is
that **the ranking inverts on an aggregation-limited task**. Both outcomes are
publishable; a null is a real chapter ("the obvious architectural fix does not touch
over-squashing, and here is the discriminator experiment that proves which axis it is").

## Background (read before anything)

- **The paper.** `Y' = Y ⊙ σ(X W_θ)`, X = hidden states after pre-normalization.
  Positions in their numbering: **G1** = after SDPA (before `o_proj`), **G2** = after
  the value projection, G3 = key, G4 = query, G5 = after `o_proj`. Their 15B-MoE 400B-token
  sweep: baseline PPL 6.026 → **G1 elementwise 5.761** (MMLU 58.79→60.82), G2 5.820,
  G3 6.016 / G4 5.981 / G5 6.017 (≈ nothing). Head-specific matters: head-**shared** G1
  = 5.801 and first-token attention 0.301 vs 0.048 for head-specific. Headwise G1 gets
  5.792 with only 1.6M added params. Sink: 46.7% → 4.8% of attention on token 0.
  All of it **trained from scratch** (random init, 400B–3.5T tokens, text only).
- **What that means for us.** The sink-free / stability / long-context results DO NOT
  port to a frozen pretrained backbone — layer *l*'s output gate cannot change layer
  *l*'s attention weights, and Qwen2.5-VL already learned to use its sink. What ports is
  the *module*: a query- or source-dependent non-linear filter, trainable as an adapter.
  Never claim "attention-sink-free" for any model we did not pretrain.
- **Released artifacts are G1-only.** `QwQZh/gated_attention` has three matched 1B
  checkpoints — `1B_baseline`, `1B_gate_headwise`, `1B_gate_elementwise` (both gated
  variants are "after SDPA" = G1). Qwen3-Next / Qwen3.5 production gating is also G1.
  **G2 exists in no checkpoint anywhere** — training is the only route to it, which is
  why P3–P4 are the contribution and P0 is only the cheap control.
- **Our anchors.** `scripts/train_carrier_layer.py` docstring: l12v2 / caption recipe
  (`--running-tally --jitter-gap 16 --grad-ckpt --l-open 12 --limit 900 --epochs 5
  --scratchpad-format caption --carrier-ckpt checkpoints/carrier_token_room_k1_best.pt`
  + the 16-root mixture in `slurm/lib/roots_inlength.txt`) → BEST ~0.999 / tf-exact ~0.99,
  N=32 exam 0.987. Arm 1 of P3 must land in that band.
- **Our model.** `Qwen/Qwen2.5-VL-7B-Instruct`, frozen, 4-bit nf4, bf16, sdpa — via
  `gnnformer/runtime.py`. Verified config: hidden 3584, **28 query heads, 4 KV heads**
  (GQA 7:1), head_dim 128, 28 layers. The 7:1 GQA ratio is the reason G2-expanded exists
  (see P4).
- **Relevant prior findings.** `outputs/superquery/STATE.md` — aggregation capacity is
  set by read fan-in, not context length (fan-2 0.95, fan-4 0.62, fan-8 0.24…); hops need
  re-quantization. Those are **capacity-shaped** laws. If gating helps anywhere, expect it
  on the distractor axis, not the evidence axis — which is exactly what P3.5 measures.

## Standing rules

- NEVER `pip install` into a shared venv. `.venv` (py3.9, transformers 4.57.6, torch 2.8)
  is the campaign venv and already supports the Qwen3-arch 1B checkpoints. `.venv_arch`
  (py3.11, transformers 5.13.1) is ONLY for P6 (Qwen3.5). If something is missing, STOP AND ASK.
- `legacy/` is READ-ONLY. `outputs_*` / `output_*` archives are read-only.
- **New code goes in `scripts/gating/` and a NEW standalone module `gnnformer/gating.py`.**
  Do not edit `gnnformer/fencing.py`, `engine.py`, `carriers.py`, `runtime.py` — if you
  believe you must, STOP AND ASK. After adding `gnnformer/gating.py`, run the full CPU
  suite (`python tests/test_fencing.py`, `test_carrier_masks.py`, `test_data.py`,
  `test_scratchpad.py`, `test_mmred_hf_adapter.py`) before any GPU job.
- **The one exception:** you may add CLI flags to `scripts/train_carrier_layer.py`,
  **defaulting to OFF so behaviour is bit-identical when unused**. Arm 1 of P3
  (LoRA-only anchor reproduction) is the regression test for that edit — if arm 1 misses
  the anchor band, the edit is suspect and everything downstream is void.
- Run dirs: `outputs/gating/<name>/<YYYYMMDD_HHMMSS>_<jobid>/` with `report.txt` +
  `ABOUT.md`. Maintain `outputs/gating/INDEX.md`. Smokes → `outputs/_scratch/`.
- Do NOT write `RESULTS.md` (Tal logs explicitly). `STATE.md` is your log.
- SLURM: check free GPUs across ALL partitions before submitting
  (`sinfo -p l40s-shared,h200-shared,rtx6k-shared,a100-public,l40s-public -N -O "Partition:16,NodeHost:12,Gres:26,GresUsed:30,StateLong"`).
  Probes → `2h_2g` (≤2 GPUs/user, keep `--mem` modest); CPU → `4h_0g` (mem ≤16G);
  trainers → `24h_1g` or `12h_4g`. **All partitions default to a 2h walltime — every
  trainer job needs an explicit `sbatch --time`.** No comma-lists in `--export` (use
  files, e.g. `slurm/lib/roots_inlength.txt`). `DRY_RUN=1` every new wrapper once.
  Suffix OUTPUT dirs with `_${SLURM_JOB_ID}`.
- Committing new files (scripts, wrappers, `gnnformer/gating.py`, tests, this dir's docs)
  is pre-authorized; never push; never commit data or checkpoints.
- **STOP AND ASK if:** a phase gate fails ambiguously; the anchor does not reproduce; you
  need to touch gnnformer core beyond the new module; HF downloads would exceed 30 GB;
  you want to launch P4 or P6 (both are explicitly gated); total new disk > 60 GB.

---

## Phase 0 — the only true ablation that exists (cheap, no training)

Three matched 1B checkpoints, identical data/params, gating the only variable.

1. Download `QwQZh/gated_attention` → `1B_baseline`, `1B_gate_headwise`,
   `1B_gate_elementwise` (~2 GB each, bf16; login node is fine, record `HF_HOME`).
   Confirm each loads in `.venv` and record the exact architecture + gate tensor names in
   STATE (this is also how you learn their reference implementation's naming).
2. `scripts/gating/probe_text_triple.py`: text MMReD tally probe over
   `data/mmred_text_longN/seq_len_{16,24,40}/all_uniform` (add `data/mmred_text_arch` if
   the schema matches — check, don't assume). Read hidden states at a layer sweep
   (≥4 layers spanning the stack, report all), pool mean|last.
3. **Metric = ridge-round exact + ±1 + R², on a held-out split.** Do NOT report logistic
   regression as the headline — LR understates ordinal decodability (known probe-family
   artifact). Report row counts per cell.
4. Also log, for each of the three models: first-token attention fraction and max hidden
   activation per layer. This reproduces their Table 4 F-Attn/M-Act on OUR data and is
   the sanity check that the gated checkpoints really are the sink-free ones.

**Gate:** the three models must be comparable (same tokenizer/context handling) and the
baseline must be above floor on at least the smallest N. If all three are at floor
everywhere, the probe is uninformative — say so in STATE and move on; do not tune the
prompt until something "works".
**Verdict to record:** does G1 gating raise tally decodability at matched everything —
and does any gain grow or shrink with N?
**Cost:** ~1 GPU-hr, `2h_2g`.

## Phase 1 — is there a sink worth filtering in OUR model? (no training)

`scripts/gating/probe_sink_7b.py` (pattern:
`scripts/presentation_diagnostics/probe_attention_map.py`, which is already half of this).
On MMReD park N=8 prompts, Qwen2.5-VL-7B frozen:

| arm | layout |
|---|---|
| plain | no fence, no posreset |
| deployed | fence + posreset + carriers (THE method layout) |

Per layer report: (a) fraction of attention on the first token, (b) mean max hidden
activation, (c) **the one that matters — of the carrier's/read-row's attention mass, how
much goes to the sink vs to frame tokens.**

**Gate:** if the read rows are losing a meaningful share of mass to a sink, the
interference story has a mechanism and P3 is well-motivated. If not, record that P3 is
being run as a falsification rather than a hope — and say so in STATE.
**Cost:** ~30 min, `2h_2g`.

## Phase 2 — implementation + CPU parity (no GPU)

`gnnformer/gating.py`, mirroring the hook style of `attach_lora`
(`gnnformer/carriers.py:131-165`) — `attach_gate(...) -> Gate` with `.parameters()`,
`.state()`, `.remove()`, and checkpoint save/load:

| variant | hook | gate shape / layer | params / layer | params, L12–27 |
|---|---|---|---|---|
| `g1_headwise` | pre-hook on `self_attn` (stash X) + pre-hook on `self_attn.o_proj` | [3584, 28] | 100 k | 1.6 M |
| `g1_elementwise` | same | [3584, 3584] | 12.85 M | 206 M (use single-layer) |
| `g2_literal` | forward-hook on `self_attn.v_proj` | [3584, 512] | 1.84 M | 29 M |
| `g2_expanded` | **P4 only** — forward patch, see below | [3584, 3584] | 12.85 M | — |

`X` = the input to `self_attn`, i.e. `input_layernorm(hidden)` — this IS the paper's
"hidden states after pre-normalization"; do not re-normalize it.

**Identity init is mandatory and the naive version silently breaks learning.** Use

```python
g = torch.sigmoid(x @ W_theta.T + b) / torch.sigmoid(b)   # W_theta = 0, b = +2.0
```

`W_θ=0, b=+2` → exactly 1.0 at init AND `σ'(2)=0.105`, a healthy gradient. Do **not**
use `b=+6` (`σ'(6)=0.0025`, gate cannot learn within our step budget). Record the chosen
`b` in every ckpt.

`tests/test_gating.py` (CPU, seconds, on a tiny random stand-in module — no model load):
1. **bit-for-bit identity at init** for every variant (gated forward == ungated forward);
2. shapes for GQA 28q/4kv, head-specific vs shared;
3. gradient flows to `W_θ` and is non-tiny at init;
4. state dict round-trip.
Then run the full existing CPU suite.

**Gate:** all tests green + the full `tests/` suite green. Nothing goes to GPU otherwise.
**Cost:** zero GPU.

## Phase 3 — the main ablation, N=8

Add OFF-by-default flags to `scripts/train_carrier_layer.py`
(`--gate {none,g1_headwise,g1_elementwise,g2_literal}`, `--gate-layers`, `--gate-lr`,
`--gate-b0`) putting `gate.parameters()` into the Adam group alongside/instead of LoRA.
Everything else unchanged: frozen 4-bit backbone, frozen `e_c` from
`checkpoints/carrier_token_room_k1_best.pt`, teacher-forced CE on scratchpad/caption
targets, `--train-frac 0.5`.

**Smoke first** (`--limit 150 --epochs 2`, `outputs/_scratch/`) to get real walltime per
arm, then size the full jobs and report the estimate in STATE before launching them.

| arm | config | must show |
|---|---|---|
| 1 | LoRA-only, l12v2 recipe | **reproduces the anchor band** — else STOP, the edit is not inert |
| 2 | `g1_headwise`, layers ≥12 | their winner, ported |
| 3 | `g2_literal`, layers ≥12 | the write gate |
| 4 | `g1_elementwise` at L=12 only | does granularity matter where aggregation happens |
| 5 | best gate + LoRA | complementary or redundant |

Gate LR is NOT LoRA's 1e-4 — it has different geometry. Try {3e-4, 1e-3} on the smoke and
pick per-arm; report the choice.

**Mandatory instrumentation, every arm, every epoch: mean gate score per layer.** A gate
sitting at ~1.0 learned nothing and that arm is VOID, not a null — say so explicitly
rather than reporting it as "no effect".

**Gate:** arm 1 in the anchor band; at least one gated arm with a live (non-saturated)
gate. Then verdict vs arm 1 on held-out exact / tf-exact.
**Cost:** 5 trainer runs; spread across `24h_1g` and `12h_4g` (3 running jobs/user/QOS).
Explicit `--time` on every one.

## Phase 3.5 — the capacity-vs-interference discriminator (the result I actually want)

Eval-only, on the P3 arms. Two orthogonal sweeps using data that already exists:

| axis | hold fixed | vary | uses |
|---|---|---|---|
| **distractor** | evidence count | N (more irrelevant frames) | `data/mmred_images_park/seq_len_{2..8}/all_uniform` |
| **capacity** | N | evidence count | `data/mmred_images_park/seq_len_8/by_evidence_count` |

Read `by_evidence_count`'s metadata first and confirm it stratifies the way this needs;
if it does not, say so and propose the smallest generator change rather than improvising.

Interpretation, to be written in STATE as a verdict:
- gain on the **distractor** axis only → bottleneck is **interference**, gating is the
  right family, G2 should beat G1;
- flat on both / gain only on the **capacity** axis → **capacity** wall, gating cannot
  help, and this is the honest-null chapter with a mechanism attached.

## Phase 4 — G2-expanded + N-transfer (GATED: ask Tal before launching)

Only after P3/P3.5. **Why it exists:** with 28 query heads sharing 4 KV heads, a gate on
`v_proj` is forcibly shared by 7 query heads — the head-**shared** condition the paper
showed costs most of the benefit (5.801 vs 5.761, F-Attn 0.301 vs 0.048). Their own G2
row is `n × k × d_k` with *k* = KV heads, so **they never tested a per-query-head write
gate either**. Gating after `repeat_kv` gives the only variant that is both *inside the
sum* and *head-specific*.

Implementation: a forward patch on `Qwen2_5_VL*Attention.forward` installed/removed by
`gnnformer/gating.py` (never an edit to gnnformer core or to transformers). It must be
the original forward with **one line inserted** (gate the broadcast V, then call the same
`F.scaled_dot_product_attention`) so that at identity init it is **bit-for-bit** equal to
the unpatched path — that equality is the parity test, add it to `tests/test_gating.py`.

Then: train N=8 → **eval N=16** (`data/mmred_longN_park/seq_len_16/all_uniform`) for every
surviving arm. This is both the anti-memorisation control (900 samples, up to 29M
trainable params — a gate that memorised will not transfer) and the headline prediction
test: **if the mechanism is interference, the G2 advantage grows from N=8 to N=16.**

## Phase 5 — verdict & report

- Figure 1: held-out exact per arm vs the LoRA control, N=8 and N=16.
- Figure 2: the P3.5 discriminator — gain vs distractor count and vs evidence count, one
  line per arm, zero line marked.
- Figure 3: mean gate score per layer per arm (proves the gates were alive).
- STATE verdict, exactly one of:
  (a) **G2 > G1**, gap grows with N → write-side gating relieves aggregation interference;
      the paper's ranking inverts on aggregation-limited tasks (name the config);
  (b) **gates ≈ LoRA** → the NeurIPS-oral architectural fix does not touch over-squashing;
      P3.5 names the axis;
  (c) **gates < LoRA** → attenuation-only edits cost signal on a frozen backbone.
- Update `outputs/gating/INDEX.md`. Leave `RESULTS.md` to Tal.

## Phase 6 — optional landscape (GATED: ask Tal; run only if P0–P5 are settled)

Qwen3.5-9B (natively multimodal, 3:1 Gated DeltaNet / Gated Attention, i.e. G1 in
production) zero-shot N-sweep on visual MMReD, with Qwen3-VL-8B-Instruct (plain
attention, verified no gate) as the least-bad reference point. `.venv_arch` only
(transformers 5.13.1); expect a bitsandbytes gotcha on packed/fused projections, and
`a100-public` is 40 GB not 80.

**There is no non-gated Qwen3.5 twin.** This is NOT an ablation and must never be written
as one. The only question it answers: *does the aggregation wall persist in a sink-free,
gated, natively-multimodal model?* A persisting wall is a strong result for the
capacity hypothesis; a vanished one is a lead, not a proof.

## Reporting

`outputs/gating/STATE.md` after every phase: what ran (job IDs, run dirs), numbers vs
gates/anchors, decisions taken, next step. Final deliverable: the three figures + the
verdict + (if (a)) a costed proposal for scaling G2 that is written down but NOT launched.

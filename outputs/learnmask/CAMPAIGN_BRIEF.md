# LEARNMASK campaign — learned discrete attention mask from the fence init

**Status: PROPOSAL — no code written, nothing submitted. Awaiting Tal's OK on arms + budget.**
Origin: peer meeting 2026-08-12 ("Gumbel mask / straight-through estimator" suggestion).
tmux session: `learnmask`. Scripts: `scripts/learnmask/`. Runs: `outputs/learnmask/`.

## Goal

Replace the hand-designed block fence with a **learned discrete attention mask**,
initialized AT the fence, trained end-to-end with task CE on the frozen 4-bit 7B.
Thesis framing: the fence is hand-crafted graph rewiring against over-squashing;
this is *learned* rewiring (DiffWire-adjacent — cite). The strongest hoped-for
result is that the learned mask **rediscovers the hand design**: gates stay closed
in early layers and open cross-frame aggregation edges around L_OPEN=12 (the E-H
inverted-U, RESULTS 2026-07-23), which would independently validate both.

Honest scope (set with Tal up front): a mask cannot add capacity. The length wall at
N≥32 is read fan-in + re-quantization (superquery capacity law, jobs 128845/128816),
not topology. The pitch is "the fence is learnable and the learned version agrees
with/refines the hand design" — NOT "the mask fixes N=128".

## Data (Tal, 2026-08-12: original benchmark, original resolution)

- **Dataset: `data/mmred_hf` — the original Fr0do/mmred HF benchmark**, NOT our self-gen
  roots. Splits under `dirs/`: seq_len ∈ {2,4,8,16,32,64,128} × {train_*, test, headfit};
  task variants incl. char_at_frame, char_on_char_{at_frame,first_app,final_app},
  aug_dense_qa. P0 picks the counting variant matching our headline task and confirms
  which seq lens have train splits (fall back to seq_len_16 train if 8 has none).
- **Resolution 512** (override FRAME_RESIZE=392). Expected token cost: Qwen processor
  snaps to multiples of 28 → 504×504 → ~324 img tokens/frame (vs 196 @392) — VERIFY in
  P0 smoke, don't assume. N=128 eval ≈ 41k img tokens: per-layer in-hook mask assembly
  mandatory (no 28× seq² buffers).
- **K0-sorted trap is LIVE here** ([[mmred-hf-k0-sorted-trap]]): dirs sort K0-first —
  strided shuffled sampling + class-dist report on EVERY split, train included.
- **Carrier-ckpt confound:** canonical carriers/LoRA were trained on self-gen data @392.
  Distribution shift to hf@512 is real → P1 must first log the frozen hand-fence baseline
  on mmred_hf@512 as the reference row; if gates underperform, suspect this confound
  before blaming the mask. (Optional later arm: re-distill carriers on hf@512.)

## Parameterization (the decision that makes extrapolation possible)

Gate logits are attached to **(relation-type, layer)** — never to absolute positions.
Every (query q, key k) pair in any sequence is classified into one relation by token
roles + block geometry; the mask at ANY length is assembled by lookup. ~R×28 scalars,
heads shared (per-head is a later arm).

Relation vocabulary (grounded in `gnnformer/fencing.py::build_block_mask`; fence
config from the deployed A3 layout — prefix | per-frame blocks(vision+carrier) |
tail = final question + decode):

| id | query row → key col | fence init | learnable in arm |
|----|---------------------|-----------|------------------|
| R1 | block token → own block (causal) | ON | S3 only |
| R2 | block token → other block, Δ-bucket | OFF | S3 only |
| R3 | carrier → own block | ON | S3 only |
| R4 | carrier → other-block carrier, Δ-bucket | OFF | S2, S3 |
| R5 | carrier → other-block content, Δ-bucket | OFF | S2, S3 |
| R6 | tail → frame content | ON (TRUNC P0.1 semantics) | S1 (per-layer re-gate), S3 |
| R7 | tail → carriers | OFF (hide_cols) | S1, S2, S3 |
| R8 | any → prefix | ON | never (anchor) |

Δ-buckets (block distance): {1, 2, 3–4, 5–8, 9–16, 17+} — log-ish, so the rule is
length-generalizing and CAN in principle express hierarchy (tree-shaped opening).
Logit counts: S1 = 2×28 = 56, S2 ≈ 14×28 = 392, S3 ≈ 22×28 = 616.

**Task-agnosticity (Tal's precaution, 2026-08-12):** the vocabulary is defined purely by
input layout (frame blocks, carriers, tail) + geometry — nothing encodes MMRED semantics.
Mechanism = layout-agnostic; learned gate VALUES = task-tuned (trained by that task's CE).
Cross-task heatmap comparison is a natural bonus experiment either way it comes out.

**Expressiveness ceiling (known limitation):** relation gates can only express masks that
are functions of role+geometry — no content-dependent routing. The S0 free-table arm is
the audit: if S0 >> S1/S2 at N=8, the vocabulary is missing a relation; read the learned
table (cells grouped by declared relation) to find which.

## Gate estimator (menu; all are ~5-line variants of one module)

| arm | forward pass | backward pass | noise | notes |
|-----|-------------|---------------|-------|-------|
| E1 det-STE | hard 1[σ(l)>.5] | through σ | none | simplest; no exploration, can stick |
| E2 soft-anneal | soft σ(l/τ), τ→0 | exact | Gumbel opt. | train/test mismatch until τ small |
| E3 **ST-Gumbel (default)** | hard sample | through soft | logistic/Gumbel | standard (Jang et al. '17 §2.2); hard forward + exploration |
| E4 hard-concrete (L0) | stretch+clip | exact a.e. | uniform | Louizos '17; principled sparsity; optional |

## Losses & schedule

**METRIC POLICY CHANGE (Tal, 2026-08-12, supersedes the CE-on-scratchpad wording
below):** the teacher-forced scratchpad metrics are DEPRECATED campaign-wide (the
caption tf_acc is largely a copy detector — outputs/gating/p35_tallycopy). Objective
and metric are the ANSWER directly: **answer-class CE** (CE over the 10 digit-token
logits at the answer position = the last prompt row; no appended rows, nothing
copyable) + class acc as the headline, with unrestricted-argmax EM as the mass-drift
canary. Golds >9 (large-N transfer) use digit-sequence CE + emitted EM (`--target
digit`, gating-P7 convention). Frozen carrier stack accordingly = the DIGIT-readout
ckpt `checkpoints/carrier_layer_digit_p7a_lora_best.pt` (gating P7a LoRA control),
NOT the caption ckpt. "Task CE" below reads as answer-class CE.

- Task CE on answer tokens (standard trainer path). Backbone frozen 4-bit, canonical
  carrier ckpt frozen — **only gate logits train** (~10² params).
- Deviation penalty: λ·Σ p_open over fence-OFF relations (opening must pay for itself);
  in S3 also λ·Σ (1−p_open) over fence-ON relations (closing must pay).
- Temperature anneal τ: 2.0 → 0.5 over training; then hard-freeze mask + final eval.
- **Gradient-scale gotcha:** during training the "forbid" bias for learnable relations
  is a soft K=−30 (kills softmax mass in bf16, keeps ∂/∂logit sane). MASK_MIN=−65504
  only in the hard-frozen eval mask. Non-learnable relations always use MASK_MIN.

## Arms

| arm | what adapts | question it answers |
|-----|-------------|---------------------|
| S0 free-table (DIAGNOSTIC) | per-cell logits, per layer, N=8 layout only | oracle upper bound for ANY mask @N=8 + vocabulary audit; its number is NEVER a headline (cannot transfer; position-memorization-prone — K0-trap lesson: shuffled balanced split + copy probe mandatory) |
| S1 tail-only | R7 (+per-layer R6) — blocks stay hard-fenced | can the readout alone do better than hand design? (peers' minimal version) |
| S2 +carriers | S1 + R4/R5 Δ-buckets per layer | does it open in-model aggregation, and at which layer? (**L_OPEN rediscovery**) |
| S3 free | all R1–R7 | does the learned mask KEEP the fence? (local-optimality of hand design) |

Order: S1 → S2 (headline) → S3 + S0 (validation/diagnostic, parallel). Estimator
ablation E1/E2/E3 runs on the winning structure arm only.

## Phases, cost & parallelization

Parallelism = many SINGLE-GPU jobs across arms (public partitions + spread QOS), not
DDP within a job (trainable state is ~10² logits; per-step cost is the frozen-7B
forward+backward either way). Duration estimates assume carrier/LoRA-trainer-like
step cost — **P1 smoke measures min/step and recalibrates this table**.

| Phase | Runs | Trains | GPUs / QOS | Est. wall | Deliverable |
|-------|------|--------|-----------|-----------|-------------|
| P0 implement | tests only | — | CPU | ~1 day coding | gate module + per-layer mask hook; fence-init parity bit-for-bit vs build_block_mask (extend tests/test_fencing.py). NB: FenceHooks injects ONE mask for all layers today — needs per-layer path (in-hook: fence_base + Σ_r gate[r,L]·rel_map_r) |
| P1 smoke | frozen hand-fence baseline eval on mmred_hf@512 + S1 @N=8, limit ~120 | 56 logits | 1× 2h_2g | 1–2 h | reference row on the new data; CE drops, gates move, hard≈soft; measured min/step + tokens/frame @512 |
| P2 main | S1-full ∥ S2-full @N=8, ~500 ex | 56 / 392 | 2 jobs ∥ 24h_1g (overflow a100-public) | 6–14 h each | gate heatmap init-vs-learned; does R4 open near L_OPEN=12? |
| P3 stress | S3 ∥ S0 ∥ E1,E2 on S-winner | 616 / ~10⁶ / winner | 4 jobs ∥ public partitions + QOS spread | 6–14 h each | S3 fence-keeping; S0 oracle gap + vocab audit; estimator table |
| P4 transfer | eval-only: learned mask zero-shot on mmred_hf seq_len_{16,32}_test vs hand fence vs no-fence | — | 2–3× 2h_2g/12h_4g | 1–3 h each | acc-vs-N lines on the ORIGINAL benchmark's own test splits; gate 64/128 (native splits exist) on the 16/32 result |
| audits | every arm | — | mostly CPU 4h_0g | hours | tally-shift copy probe, pos-symmetry canary, per-class acc, class-dist |

Campaign wall-clock: ~3–5 days optimistic, ~1 week realistic (implementation + one
debug cycle + queue luck). Critical path P0→P1→P2; P3/P4 overlap once P2 flies.

## Headline figures (decided up front)

1. **Gate heatmap:** relation (y) × layer 0–27 (x), cell = P(open); init panel vs
   learned panel side by side. The L_OPEN signature = R4 row flipping ON near layer 12.
2. **Acc vs N:** lines for hand fence / learned mask (zero-shot transfer) / frozen
   no-fence baseline, N ∈ {8, 16, 32, 64}.

## Open decisions for Tal

1. OK to spend GPU on P1 smoke after P0 parity passes? (2h_2g, ~1 GPU-hour)
2. S2 trains gates alone (carriers frozen) — later joint gate+LoRA arm, or out of scope?
3. Group name `learnmask` OK?
4. S0 free-table diagnostic in P3: run it (recommended — oracle bound + vocabulary
   audit) or skip?

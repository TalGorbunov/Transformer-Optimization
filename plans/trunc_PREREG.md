# PREREG — carrier truncation campaign (frames end at L*)

Written 2026-07-24, BEFORE any GPU job of this campaign. Brief:
`plans/trunc_efficiency_agent_brief.md`. Winner ckpt everywhere:
`outputs/ladder/image_longN/carrier_fmt_caption/20260722_222032_L12_r8/carrier_layer_best.pt`
(caption, L_OPEN=12, r=8).

## P0.1 — code truth: which columns can tail rows attend (VERIFIED by reading, not assumed)

Sources: `build_block_mask` (`experiments/glstm/carrier_token_distill.py:69-87`),
`make_masks` (`experiments/glstm/carrier_layer_lora.py:50-61`), `_ext_mask`
(`carrier_layer_lora.py:449-459`), `ext_mask` (`carrier_layer_cached.py:44-55`).

`build_block_mask(seq, blocks, hide_cols=cpos)` does, in order:
1. full causal mask;
2. `m[:, cpos] = MIN` — carrier columns hidden from **every** row;
3. per block `(a,b)`: rewrite `m[a:b, a:b]` to block-causal (re-opens the own-block
   carrier column for in-block rows);
4. per block pair `i≠j`: `m[rows_i, cols_j] = MIN` (the fence).

Steps 3-4 touch **only rows inside frame blocks**. Rows before `blocks[0][0]`
(prefix+leading question) and rows `≥ fin_start` (tail = final question + generation
prompt) are never rewritten. Therefore:

| rows | lo (layers < L_OPEN) | hi (layers ≥ L_OPEN) |
|---|---|---|
| prefix+question | causal (nothing later exists) | same |
| frame rows (block i) | prefix+question + own block causal | same (hi only adds carrier-row/tail-row edges) |
| carrier_i | prefix+question + own frame + itself | + earlier carriers (`make_masks` line 58-59) |
| **tail (≥ fin_start)** | **prefix+question + ALL FRAME TOKENS + tail causal; NOT carriers** | **same + ALL carriers** (`mask_hi[fin_start:, ct]=0`) |
| decoded rows (`_ext_mask`: copy of last tail row + causal over appends) | **ALL frame tokens** + prefix+question+tail; not carriers | same + all carriers |

**P0.1 verdict: the tail and every decoded token CAN attend frame tokens at every layer.**
The mask does NOT structurally guarantee E1 token-identity. The brief's "if tail never
reads frames → must be TOKEN-IDENTICAL" branch is **false at the mask level**; the "mask
edge" the brief warned about is simply that tail rows keep plain causal access to frames.
E1 therefore becomes an **empirical test of learned behavior**: does the trained
caption/L12 model's generation actually *use* its mask-permitted frame access?

Corollary for E5 (chunked prefill): per-block short forwards reproduce carrier/question/
frame-row hidden states **exactly** (their visibility is per-block by construction), but
tail-row h_{L*} is NOT reproducible per-block (tail reads all frames in lo). Chunked
prefill therefore implies tail rows computed without frames — i.e. full truncation
semantics for the tail, an approximation whose cost is exactly what E1/E2 measure.
Pre-registered E5 verdict rule: carriers+question hidden states must match dense to
numerics (max |Δ| within bf16 tolerance, assert); tail-row Δ is reported (expected
nonzero); behavioral bar = decoded answers equal on the 5 verification samples.

## P0.2 — position handling for physical truncation

All positions are the ALREADY-transformed ids (after `reset_positions` + sequential
carrier override `blk0_max+1+i` + tail re-base `+NF`, `carrier_layer_lora.py:618-621`).
Physical truncation **index-selects** the surviving columns of the `(3,1,seq)` pos tensor
— original ids preserved verbatim, never renumbered. Decoded tokens keep the existing
rule `pos[..., -1] + 1..e` (the last tail position survives truncation, so appended
positions are unchanged vs baseline). RoPE cos/sin recomputed from the index-selected pos
tensor. `--pos-couple` ckpts are refused with the new flags (winner is caption,
pos_couple=False).

Keep set (= "[question]+[carriers]+[tail]"): `[0, blocks[0][0]) ∪ {cpos} ∪ [fin_start,
seq)`. Blocks tile `[vstarts[0], fin_start)` contiguously and each carrier is the last
token of its block, so the dropped set is exactly the frame tokens incl.
vision_start/vision_end furniture.

## P0.3 — flag design (eval-only, backward-compatible, default off)

In `carrier_layer_lora.py` (the exam script used for every logged fmt cell):
- `--drop-frame-kv`: decode-only; appended rows get frame columns masked to MIN at ALL
  layers (prefill untouched). Same shapes/kernel as baseline ⇒ SDPA deterministic ⇒ any
  token change is CAUSED by removed frame attention (the science arm).
- `--fast-decode`: cached incremental decode — prefill once capturing per-layer inputs at
  keep cols; each decode step runs the 28 layers over [keep-cols cache ‖ appended rows]
  only (~150-500 tokens instead of up to ~12k). Mathematically identical to
  `--drop-frame-kv` (prompt rows never attend appended rows, so their prefill states are
  reusable; appended rows see exactly the keep set); numerics may differ (different SDPA
  shapes). This is the engineering arm that yields the real decode speedup + VRAM numbers.
- `--truncate-at L`: physically drop frame rows from the hidden states entering layer L
  (index-select h, masks, positions); layers ≥ L run on the short sequence. Implies
  drop-frame-kv semantics for decode (decoded rows attend keep set only at every layer).
- `--exactness-check`: per sample decode baseline AND flagged arms in one run; report
  token-identity, first divergence, per-arm wall-clock + peak VRAM.

## E1 — exactness bands (revised per P0.1)

Setup: 20 samples = 8 in-dist (N=8, from `eval_dirs_indist150.txt`) + 8 N=32
(`eval_dirs_N32all.txt`) + 4 N=64 (`eval_dirs_N64.txt`), first-k of each file (no
cherry-picking), DEC=620, winner ckpt. Arms per sample: baseline / drop-frame-kv
(mask-only) / drop-frame-kv+fast-decode. Two 2h_2g jobs (N8+N32; N64), per-sample verdict
lines streamed to the log.

Bands (token-identity of full greedy transcripts, baseline vs mask-only arm):
- **20/20 identical → FREE-SPEEDUP GO**: generation never uses its frame access in
  practice; log the fast-decode speedup factor (decode wall-clock ratio) and VRAM.
- **Some divergence but parsed ANSWER unchanged on ≥19/20** → transcript-jitter PARTIAL:
  proceed to E2 (accuracy is the arbiter), log divergence sites.
- **Answer changes on ≥2/20** → generation READS frames: a real architecture-story
  finding; diagnose (which step, which samples' N) BEFORE any Phase 2 launch.
- fast-decode vs mask-only divergence (same mask semantics) = numerics only; any token
  flip is logged; ≥3/20 flips → fast-decode NOT used for Phase 2 exams (fall back to
  mask-only/dense truncated path for accuracy cells).

## AMENDMENT (2026-07-25 ~04:30, BEFORE any E2/E3 launch)

E1a landed (16/16): mask-only identical 1/16, answer-equal 1/16 → **READS-FRAMES
verdict** (third band). fast≡mask 14/16 — passes the ≤2-flip band, BUT both flips
cascaded into different parsed answers (near-tie logits under kvdrop degeneracy), i.e.
~12% answer-level noise ≫ the −0.02 E2 band. Stricter-than-band decision, fixed now:
**E2/E3 accuracy cells use the DENSE flagged decode** (`--drop-frame-kv --truncate-at L`,
byte-identical shapes/kernel to baseline — truncation is the ONLY variable);
`--fast-decode` is used only for E6 engineering numbers with its 18/20-fidelity caveat
reported. Bands below unchanged.

## E2 — truncation bands (fixed BEFORE launch)

`--truncate-at 12`, winner ckpt unchanged, byte-identical dirs-files + budgets as the
logged caption cells: in-dist-150 (DEC=100, ref **1.000**) · N=32 150 dirs (DEC=320, ref
**0.987**) · N=64 52 dirs (DEC=620, ref **0.981**). Δacc = trunc − ref per cell:
- **Δ ≥ −0.02 on ALL THREE cells → TRUNCATION GO** (carriers saturated by L12; headline).
- **−0.10 ≤ Δ < −0.02 on any cell (none < −0.10) → PARTIAL** → Phase 3 retrain is the fix
  under test.
- **Δ < −0.10 on any cell → carriers NOT saturated at L12** → the E3 sweep becomes the
  deliverable (saturation-depth curve).

## E3 — truncation-layer sweep (runs regardless of E2's band)

L_trunc ∈ {12, 14, 16, 20, 24}, eval-only, winner ckpt, the N=32 cell (150 dirs,
DEC=320). Deliverable: acc vs L_trunc curve; the knee = carrier-saturation depth. (L=12
cell shared with E2 — not re-run.)

## E4 — retrain band (only if E2 PARTIAL/FAIL)

Cached trainer with truncation at best L_trunc (cache carriers+question+tail rows only),
caption winner recipe verbatim otherwise. Exams as E2. Band: within **0.01** of the
non-truncated caption cells on all three = GO-after-retrain.

## E6 — benchmark (no bands; honest caveats)

baseline / drop-frame-kv(fast) / truncate-at-12 / truncate+chunked at N=8/32/64/128:
prefill s, decode tok/s, peak VRAM, KV bytes (computed: 2·layers·seq·n_kv_heads·head_dim·
2B, with per-arm seq accounting), min/sample end-to-end, one a100. Caveats to report:
dense-mask python baseline (not a production KV-cache server), keep-row K/V recomputed
per decode step in fast path, 4-bit weights.

## E7 — pre-registered prediction (gated on E2/E4 GO + E5 working)

N=256 supply: longN generator extension + external tally (cache + CPU). The in-length
N=128-trained caption arm (truncated trainer): readout error law (draft 2026-07-24,
p≈0.002, mean N_ev≈64) predicts N=128 held-out exact-match **≈ 0.88**. Band:
|measured − 0.88| ≤ 0.05 → law holds at 128.

## AMENDMENT 2 (2026-07-25 ~08:20, BEFORE the hybrid probe runs)

E4-caption exams FAILED greedy (in-dist 0.133, N=32 0.073) with TF-count 1.000; E4b
(scan) plateaus at the identical tf-exact 0.165 → the per-slot in-model readout is the
wall, format-independent. HYBRID fallback cell (pre-registered here): dump carrier states
entering layers {12,16,20} from the DEPLOY-MATCHED truncated forward (E4-caption ckpt,
TRUNC_AT=12) on the arm-A N=32 dirs (150 samples), run `replica_gate_tally.py` (existing
CPU probe, 5-seed 50/50 internal split — the established scaffold methodology). Bands:
tally exact ≥0.90 at any layer = **HYBRID GO** ("one token/frame inference + external
linear readout"); 0.70-0.90 = partial (report as degradation of the gate under
truncation vs the 0.99 non-truncated reference); <0.70 = truncation damages the carrier
code itself (would contradict E5's carrier-match result — investigate). E7 remains gated
on the ORIGINAL in-model GO bands and does NOT unlock from a hybrid GO.

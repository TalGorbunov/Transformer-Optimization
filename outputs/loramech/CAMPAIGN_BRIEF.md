# LORAMECH campaign — what does a plain LoRA learn when it "beats Hahn" in-length?

**Status: DRAFT v2 (Claude, 2026-08-26) — awaiting Tal's OK. No scripts written, no jobs submitted.**
**Regime decision (Tal, 2026-08-26): FRAMES-FIRST everywhere. Q-first is deprecated for baselines and
instruments; the existing Q-first adapters are an optional secondary arm only.**
Scripts (after OK): `scripts/loramech/`. Runs: `outputs/loramech/`. Log: `STATE.md` (append-only,
newest last). RESULTS.md is Tal's — DRAFT entries go at the bottom of STATE.md.

Origin: peer meeting 2026-08-26 — "if plain LoRA reads N=8 at 1.000 (and N=32 at 0.967), how can
we claim the Hahn softmax limitation? does it learn a shortcut? can the LoRA checkpoint teach us
how the model solves it?" Context read: PAPER_SKELETON §4–§6/§9, RESULTS [2026-07-20] E-B,
[2026-07-25] P4, [2026-07-31] presentation battery, [2026-08-22] ARMOR A (Hahn), probe_hahn.py,
train_sft_baseline.py, checkpoints/README.md.

## 0. One paragraph

Hahn's O(1/N) bound constrains the SLOPE of single-item sensitivity vs N for a FIXED model, not
the intercept at N=8; our own measurement (ARMOR A) puts the frozen joint model at α=0.72 with
the answer-defining frame at the bf16 floor by N=128, while the fenced per-frame slot is flat
(α=0.00). Every learned aggregator in the record fits in-length and decays beyond — plain LoRA
included (E-B ≤8-trained: 1.000 → 0.480/0.350/0.220 @16/32/64, dead mid-range; P4.1 ≤32-trained:
0.967 → 0.787 @64). What is NOT measured: the mechanism by which the LoRA succeeds inside its
training window. Two candidate mechanisms make different, testable predictions: (GAIN) the read
stays a diluting softmax sum and the LoRA only sharpens margins — α unchanged in-length, fails
when 1/N falls under the learned margin (Chiang & Cholak's picture; their remedy is log-N logit
scaling); (ADDRESSING) the LoRA builds per-frame question-conditioned verdicts in the frame
positions and sums them — "learned carriers"/"learned fence" — α≈0 inside the window,
cross-frame Jacobian share collapses, per-frame gate d′ pops in frame positions after L*.
Either outcome converts the peer's objection into evidence: LoRA moves the horizon, not the law.
A GAIN result also yields a cheap, no-tool, in-model rung (log-N scaling) for the spectrum figure.

## 1. Checkpoints, prompts, data (all fixed)

- **Regime: frames-first** = the standard layout `build_count_prompt` (images, then question,
  then "Answer: "), identical to the frozen baseline 0.219 and to the ARMOR-A plain arm. Every
  LoRA in this campaign is trained AND evaluated in this layout. Why it matters: with causal
  attention, question-first lets frame tokens precompute question-conditioned verdicts (the
  carrier route, −46% in the carrier ablation); frames-first forbids it — the read must happen
  at the question/answer positions over N question-blind frames. That is the cleanest Hahn test.
- **P0 — train frames-first adapters (new; same recipe as E-B/P4.1 otherwise: PEFT q/k/v/o+MLP,
  LM loss on answer tokens, 5 ep, converges ep1):**
  - `ff_le32`: train N∈{8,16,32}, the P4.1 mirror — **H200 only** (~5 h; job 125567 = 5h14m;
    ≤32 OOM'd on 48 GB in 5 documented attempts) — PRIMARY (3 in-window N points).
  - `ff_le8`: train N≤8, the E-B mirror — any 48 GB GPU (~3.5 h; job 124696 = 3h39m) — smoke +
    shortcut-anatomy arm.
  - Trainer delta: `train_sft_baseline.py --frames-first` (new template; the byte-identical
    Q-first `build_messages` stays untouched so the old adapters still restore). Splits redrawn
    with the P4 contamination discipline; exam dirs files rebuilt from provably-unseen dirs.
  - Adapters land in `checkpoints/sft_ff_le32_adapter/`, `checkpoints/sft_ff_le8_adapter/`;
    their in-length exam numbers (N=8/16/32 + 64/128 zero-shot) become the CANONICAL SFT baseline
    row (§6), replacing the Q-first 0.967/0.787 which stays in the record with a regime label.
- **P0b (optional, Tal's question 2026-08-26 "can we fine-tune for low α?"):** `ff_le32_logn` =
  same recipe as ff_le32 but with attention logits scaled by log(S)/log(S_ref) DURING training
  (N-aware sharpness; Chiang & Cholak 2022 / SSMax 2025 in adapter form). Compare to ff_le32 on
  L1 (α) and the 64/128 exams. Prediction: retrieval-shaped signal stays sharp (lower α), the
  COUNT still fades or hits the range cap (softmax = mean aggregator; the count must ride on
  share k/N or sink-ratio k/(k+C)). Same H200 cost as ff_le32; only if the queue allows.
- **Secondary arm (optional, eval-only, already on disk):** the Q-first adapters
  `checkpoints/sft_inlength_p41_adapter/`, `checkpoints/sft_control_le8_v2_adapter/` run through
  L1 with their own template — a 2×(regime) contrast on the mechanism (does conditioning change
  GAIN→ADDRESSING?). Not required for any hypothesis below.
- **Contrast arm (fenced LoRA, the method):** `checkpoints/carrier_layer_fmt_caption_best.pt`
  — only where a like-for-like mechanism signature is cheap (L2/L3), never re-trained.
- **Pools:** `data/mmred_longN_park/seq_len_{8,16,32,64,128}/all_uniform` (ARMOR-A pools,
  gold≤8 slice; reuse the SAME pairs.csv sample ids as `outputs/armor/hahn/20260822_211457_N*/`
  for paired comparison — the frozen frames-first curve is ALREADY measured there, no rerun);
  exam cells on the rebuilt ff dirs files; nothing trained at N=64/128.

## 2. Cells

| cell | instrument (exists → delta) | reads | GPU |
|---|---|---|---|
| **P0** train `ff_le32` (H200) + `ff_le8` (48 GB) | `train_sft_baseline.py --frames-first` | the canonical frames-first SFT baseline row + the campaign's subjects | ~5 h H200 + ~3.5 h L40S |
| **L1** Hahn α + A2 margin, LoRA-on: {ff_le32, ff_le8} × N∈{8,16,32,64,128}, plain arm only, loci final @hs 16/20/28; frozen curve = ARMOR-A as-is | `scripts/armor/probe_hahn.py` → add `--peft-adapter`, `--arms plain` (same prompt as today) | GAIN vs ADDRESSING; margin-vs-N crossing vs acc collapse N; **free position-confound read**: pred-moves-up rate vs flip_t (early vs late frames) at N=64/128 | ~2.5 h (2 ckpts × 5 N × ~15 min) |
| **L2** Jacobian cross-frame share at the answer locus, LoRA-on vs frozen, N=8 (+N=32 if cheap) | `scripts/presentation_diagnostics/probe_sensitivity.py` joint arm → adapter load + plain frames-first layout + answer-position target | did the LoRA zero cross-frame edges (learned fence)? | ~40 min |
| **L3** per-layer per-frame gate d′ at FRAME positions (last vision token of frame i + span mean), LoRA-on vs frozen, N=8/32 | `scripts/probe_supply.py` / gate_tally protocol → adapter hooks; hs layers 4..28 | frames-first makes question-conditioned frame verdicts IMPOSSIBLE by causality — this cell is the sanity check that the LoRA didn't find another per-frame route (e.g. the QUESTION tokens' positions carrying per-frame verdicts); probe question-token positions too | ~1 h |
| **L4** answer-position attention over frame spans vs N, LoRA vs frozen: entropy, mass on evidence frames, max-frame mass | `scripts/presentation_diagnostics/probe_lasttok_attn.py` → adapter load | sharpness sized to N_train? sum-like (uniform over evidence) or pointer-like? | ~40 min |
| **L5** log-N attention-logit scaling on ff_le32 at eval: `scaling = head_dim^-0.5 · log(S)/log(S_ref)` on the LM decoder attention modules only (S = text seq len, S_ref = N=32 seq len), N∈{32,64,128} | `train_sft_baseline.py --eval-only-adapter` → `--attn-logn-scale` (set `module.scaling` on `Qwen2_5_VLAttention`, transformers 4.57.6 passes it to sdpa) | Chiang–Cholak remedy: does extrapolation recover? | ~1.5 h |
| **L7** the FENCED-side fade ladder (Tal's "decodability" point, 2026-08-26): same paired flips, loci = (a) frozen answer position over fenced supply — ALREADY in `outputs/armor/hahn/*/pairs.csv` (fenced `final`/`anchor`, never reported): Δ 2.77/4.6 @N=8 → at the floor (~1.2) by N=32 vs the flipped frame's own slot 39 flat; (b) the trained DIGIT carrier stack `checkpoints/carrier_layer_digit_p7a_lora_best.pt` (answer token reads N carriers directly) — answer-position Δ and margin vs N; (c) the CAPTION scratchpad stack `checkpoints/carrier_layer_fmt_caption_best.pt` — Δ at the teacher-forced verdict token OF FRAME t (one reader per item in time) and at the final tally token; N∈{8..128} | (a) CPU refit of existing CSVs; (b)/(c) `probe_hahn.py` → deployed layout via `gnnformer.engine.CarrierEngine` with `output_hidden_states` (new capture path, ~60 lines; teacher-forced scratchpad for (c)) | supply decodable (slot Δ 39, flat) yet the read fades: frozen read is weak from N=8 (2.77 ≈ 2× floor) and gone by 32; digit read = share-like fade predicted (P4.2's dead mid-range); scratchpad verdict token predicted FLAT (α≈0) — the "one reader per item" mechanism measured, not argued | (a) 0 · (b)+(c) ~2 h |
| **L6** per-count anatomy of ff_le32 @N=64/128 and ff_le8 @N=16/32 (+ the Q-first P4.1/E-B exams already on disk) | rescore exam CSVs (CPU) | live curve vs extremes-shortcut relapse | 0 |

## 3. Pre-registered hypotheses and bands (fixed before any run)

- **H1 (the law survives training).** For every LoRA cell, α over N∈{32,64,128} (beyond/at the
  training edge) ≥ 0.5. Refuted if α_out < 0.2 for ff_le32 (a plain LoRA that stays flat past its
  window would be the counterexample the peer is asking about — we would say so).
- **H2 (mechanism, two-sided; one of them must hold, both reported):**
  - GAIN: α_ff_le32 over N∈{8,16,32} ∈ [0.5,1.0] (CI excludes 0.2) AND L2 share within ±0.1 of
    the frozen 0.52 AND L3 frame-position d′ within +1 of frozen → "amplified diluting sum".
  - ADDRESSING: |α_ff_le32| over {8,16,32} < 0.2 AND (L2 share ≤ 0.25 OR L3 frame-position d′
    ≥ frozen+3 by L20) → "learned carriers/fence inside the window".
  - Mixed outcomes are logged as mixed; no forcing.
- **H3 (log-N rung).** L5: N=64 zero-shot (ff_le32's own measured value, expected < 0.80) → ≥ +0.10 absolute AND ≥0.90 with N=32 within −0.02 = mechanism confirmed
  GAIN + new in-model rung; ≤0.82 = null (favors ADDRESSING or precision floor); any N=32
  drop > 0.05 = intervention harmful, reported.
- **H4 (shortcut anatomy).** L6: ff_le8 @N=16 mid-range (g3..g5 of N=16) recall ≤ 0.2 with
  extremes ≥ 0.8 replicates [2026-07-20] in the frames-first regime; ff_le32 @N=64: mid-range
  recall ≥ 0.6 = live curve (graceful decay), else a relapse to the extremes heuristic.
- **H5 (positional confound).** L1 flip_t split at N=64/128: if pred-moves-up rate for
  t ≤ 32 exceeds t > 32 by ≥ 0.2 → RoPE-extrapolation contaminates the LoRA drift row (§9
  flag becomes a measured caveat); else the decay is aggregation, not position.
- **H0 (in-length fit survives the regime switch).** ff_le32 in-length N=8/16/32 ≥ 0.90 each
  (per-count uniform, pf 0). If it does NOT (< 0.80 at 32), that is itself a finding: without
  question-conditioned encoding a 24M LoRA cannot fit N=32 in-length — the frames-first regime is
  where dilution already binds at 32 — and L1 runs on ff_le8 + the ≤32 window as measured.
- **H7 (fenced ladder).** (a) frozen-over-fenced answer Δ at the floor by N≤64 (already true in the
  CSVs: at floor from N=32; refit α with floor-censoring, report both). (b) digit stack: α_final ≥ 0.5
  over N∈{8..64} AND margin crossing at N ≈ where P4.2/P7a accuracy dies → the trained read over
  perfect supply fades like Hahn (share read). (c) caption stack: |α| < 0.2 at frame t's verdict
  token (CI ∋ 0) → the serial readout is the length-invariant reader; if the TALLY token fades while
  the verdict token is flat, the residual is arithmetic-in-text, not perception/aggregation.
  Predicted figure: five rows on one axis — joint answer 0.72 · fenced slot 0.00 · frozen read
  over fenced supply (floor by 32) · trained digit read (fades) · scratchpad verdict (flat).
- **H6 (regime contrast, optional).** Q-first P4.1 vs ff_le32 on L1/L3: Q-first shows the
  ADDRESSING signature (verdicts in frame positions) while frames-first shows GAIN → the two
  regimes use different mechanisms and Q-first's 0.967 was partly the carrier route in disguise.

## 4. What feeds back into the paper / method

- §5 gets "what the trained repair learns": LoRA α-curve on F4 next to frozen and fenced.
- §6: the SFT baseline row becomes frames-first (ff_le32 numbers), regime-labelled; the Q-first
  0.967/0.787 stays in the record as the conditioned-encoding variant. §9 gets one sentence:
  question-first layouts were deprecated for baselines/instruments (2026-08-26); the fence's
  isolation is the portable piece, Q-first was a Qwen-specific amplifier ([2026-07-19] Track B).
- ADDRESSING (only reachable via H6's Q-first arm, or an unexpected route in frames-first) → the
  LoRA rediscovers the carrier structure; our method is the explicit, zero-training version.
- GAIN (+H3 positive) → log-N scaling is a no-tool in-model rung on F1 (what Chaim/Gabriele
  asked for), honest ceiling = bf16 floor already measured in ARMOR A.
- H5 positive → LoRA drift row in §6 carries a measured positional caveat; the fenced rows
  (posreset) are immune by construction — say so explicitly.

## 5. Order, cost, discipline

Order: P0 ff_le8 (48 GB, smoke of the --frames-first trainer path, submit day 1) and P0 ff_le32
(H200, the long pole — queue immediately, explicit --time=8:00:00) → in-length + zero-shot exams
(H0, the new §6 row) + L6 → L1 (the decisive cell; script delta ~30 lines) → L5 (one-attribute
hook, independent of L1's answer) → L7(a) (CPU, do it day 1) + L7(b)/(c) (independent of P0 — can run while ff_le32 queues) → L2/L3/L4 (mechanism detail; skip L4 if L2+L3 decide H2) →
H6 secondary arm only if time allows. Total ≈ 5 h H200 + ~10 h single-GPU 48 GB/a100, all
within 12h_4g / 24h_1g (a 24M-param bf16 adapter is negligible at eval; probe_hahn already ran
N=128 plain on a100). MANDATORY CHECKPOINT: after P0 + H0 land, write the H0 verdict in STATE
and STOP for Tal's review before L1–L5 (the mechanism cells depend on which window exists).
Discipline (CLAUDE.md §5/§7): smokes in `outputs/_scratch/`, explicit `--time`, no comma
`--export`, no installs, gnnformer/ untouched (all deltas live in scripts/), stop on any
artifact suspicion, every cell reports class distribution + majority baseline.

## 6. Theory note for L5/L6 (added 2026-08-26 after Tal's question) + related work

**Retrieval vs counting under sharpening.** Softmax weight on the evidence frames is
k·e^s / (k·e^s + (N−k)). Log-N logit scaling (Chiang & Cholak 2022; Veličković et al. 2024
adaptive temperature; SSMax 2025) restores a RETRIEVAL read (share → 1 for any k ≥ 1). A COUNT
carried by the share itself (≈ k/N, needs 1/N resolution) is NOT rescued by sharpening and may
be hurt by it. So H3 has a theory-predicted null. Both outcomes are informative:
- rescue → the LoRA's read is retrieval-like (sharp pointer + something else counting);
- null → counting is a ratio read; the only escapes are one reader per item (carriers in space,
  scratchpad in time) or exact execution — the thesis claim, now with the adapter as evidence.

**Free signature for L6 (add to the per-count anatomy):** a k/N-share mechanism trained at
N_train and read at N > N_train predicts **systematic undercount ≈ k·N_train/N** (≈ k/2 at
N=64 for ff_le32). Report pred/gold slope per N; slope ≈ N_train/N = share mechanism,
slope ≈ 1 with noise = something else. Also run on the Q-first P4.1 N=64 predictions if on disk.

**Related work (must-cite for §5's mechanism subsection):** Hahn TACL 2020 (bound);
Chiang & Cholak ACL 2022 (layernorm exact solutions, CE → chance, log-n remedy on FIRST/PARITY);
Veličković, Perivolaropoulos, Barbero, Pascanu 2024 "softmax is not enough (for sharp OOD)"
(dispersion theorem, inference-time adaptive temperature, max-retrieval task); Nakanishi 2025
Scalable-Softmax (s^{log n}, pretraining); Peng et al. 2023 YaRN (t = 0.1 ln s + 1, empirical);
Barbero et al. 2024 "Transformers need glasses" (counting vs copying, over-squashing framing);
Behrens, Biggio, Zdeborová 2024 "Counting in small transformers" (relation- vs inventory-based
counting circuits); Yehudai et al. 2024 "When can transformers count to n?" (dimension ≥ n);
Zhou et al. 2024 RASP-L (counting predicted not to length-generalize under fine-tuning).
Delta claimed: α measured in a production VLM and on a fine-tuned adapter, linked to the
adapter's accuracy horizon; the temperature remedy tested on a fine-tuned multimodal COUNTER
with the retrieval/counting distinction explicit; the fence as the α = 0 reference. None of
the above does this; all of them are cited as the boundary, never as competitors.

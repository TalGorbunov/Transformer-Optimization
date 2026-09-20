# Prior-art sweep: "fence + per-unit question replica + evidence-only gate at the read" (2026-09-16)

> Three parallel sweeps (text LLM / VLM-video / mechanism-theory), ~90 searches, ~95 abstracts,
> run by Claude subagents on 2026-09-16 after Tal asked "has a simple gate plus mask been shown?".
> Sparse RAG and Superposition Prompting verified first-hand (ar5iv full text / arXiv abs) the same day.
> Semantic Scholar citation pulls were rate-limited: citer lists of Sparse RAG (39 citers, no
> counting descendant) partly checked, Superposition Prompting's not pulled. Re-check before submission.
> Shareable page: https://claude.ai/artifact/JtDaz8AJA9ngnwwRidEj2F (section 3).

## Verdict

The WIRING exists for single-answer retrieval. Nobody applies it to an aggregation task, tests
beyond the trained unit count, isolates the per-unit question copy, or measures what the read removes.

| work | isolated units | question per unit | pos reset | hard gate at read | frozen base | counting / extrapolation / mechanism |
|---|---|---|---|---|---|---|
| **Sparse RAG** Zhu et al. 2405.16178 ICLR 2025 (VERIFIED) | Y | Y, before each context | Y, per-context ids | Y, in-model P("Good") per context, drop < σ | LoRA r4 (Gemini XXS/XS) | N / N (20 docs train+test) / N |
| **Superposition Prompting** Merth et al. 2404.06910 ICML 2024 | Y (paths) | Y, AFTER each doc (doc states question-blind) | Y ("path equilibrium") | Y, discard paths by LM saliency; surviving KV concatenated | Y, training-free | N / gains when context > training length / N |
| **Attention Entropy Is a Key Factor** 2412.16545 ACL 2025 | Y | N | Y | Y, top-K=2 windows by attention mass | Y | N / N (4–16k) / entropy vs perf R≈0.95 |
| **Anil et al. 2207.04901 NeurIPS 2022, App. F Fig. 15** | N | N | N | ORACLE mask removing distractors | N (applied during fine-tuning, LaMDA) | parity / "perfect length generalization" / N |
| **FocusICL** 2408.13987 EMNLP 2024 | Y (batches) | N | Y | Y, exact zeros on low-attention tokens | Y | fixed 5-item probe / N / attention-to-query vs #demos |
| **VideoStreaming** 2405.16009 NeurIPS 2024 | partial (memory-propagated clips) | N | n/a | Y, Gumbel top-k over clip memories | N (jointly trained) | N / N / N |
| **Das et al.** 2601.07812 ACL 2026 | Y (vision only, L12–23, train-time) | N | N | N | LoRA | Y (≤10 images) / N / inter- vs intra-image share |
| **CAPL** 2603.07048 | Y (diagnostic; hurts ~20 pts alone) | N | N | N | Y | N |
| **Neural Databases** Thorne 2010.06973 VLDB 2021 | Y (FiD) | Y (FiD) | n/a | N | N | count/min/max: FiD "cannot perform aggregation queries" → operators outside the model |
| **Low-Frequency Trap** 2608.06361 Aug 2026 | N | N | N | oracle keyframes as INPUT | Y (frontier VLMs) | event counting; collapses past 6–10 events; behavioral only |

Runners-up (one sentence each in related work): DePaC 2412.14905 (question per window, full FT,
output-level selection); PCoE 2601.08670 (frozen, query per expert, retrieval-gated output
ensemble); FocusLLM 2408.11745 (frozen, question per chunk, 8K→400K, no gate); Graph-KV
2506.07334 (block mask + shared positions framed as message passing inside the LLM — cite for GNN
framing); Block-Attention 2409.15355 (question-agnostic isolation costs 20 pts zero-shot);
Structured Prompting 2212.06713; Set-Based Prompting 2406.06581; Dynamic Block-Sparse 2503.08640;
SDAG 2602.04711; REPLUG; NBCE; PEVLM 2506.19651; PQR/T-Former 2412.19304 (question per frame
inside a Q-Former, outside the LLM); FOCUS 2508.13744; Delimiter Token Scaling 2602.01984;
DAFS 2607.15689 + retrieve-then-read frame selectors (AKS, T*, VideoITG, FrameOracle, QSVideo);
query-aware token pruning family (FastV, SparseVLM, PyramidDrop, ...); Oolong 2511.02817
(text benchmark "classify each chunk then count" — candidate for the E4 text port).

## Counter-evidence to "sharpening cannot do it" (must be cited and scoped)
- **DySCO** 2602.22175 (2026): training-free, model's own retrieval heads pick tokens, soft +log β
  boost, up to +25% at 128k (Qwen3, Llama-3.1). Also a precedent for a self-derived selector (E1).
- **InfoScale** 2501.08570 (2025): entropy-invariant temperature; claims dilution is THE
  extrapolation bottleneck and scaling fixes it (small model, LM perplexity).
- Neither is a counting read; neither ablates hard zeros vs scaling. Scope the claim to "an
  aggregation read where k of N units must all survive".

## Theory siblings of the (N−k) competitor term (cite as what the photograph instantiates)
Hayase & Karakida 2605.12697 (gap-counting N_n); Bruno et al. 2605.08505 (critical β_n);
Bansal et al. 2512.13898 ("score dilution", log N margin); 2511.12869 ("softmax crowding");
Golowich et al. 2502.16792 (length generalization iff k-sparse dependence — "wall moves from N to k");
Brändel et al. 2606.29139 (Jacobian decay p≈0.7–0.9 vs DISTANCE, Pythia/Qwen2.5-0.5B — nearest
empirical exponent; ours is vs #units with a fenced control).

## What the sweeps did NOT find
- No paper isolates the per-unit question copy as a variable (our 2×3 layout factorial,
  outputs/sparse/layout/, is the first such ablation on record).
- No Hahn-style sensitivity exponent vs #units in a pretrained model.
- No block isolation + gate on a counting task, with extrapolation beyond the trained unit count,
  or with a share-law / knockout measurement.
- A learned per-frame classifier on the frame's own fenced states as the gate appears unclaimed.

## Related-work sentence to pre-empt
"The fence-with-replica-plus-gate wiring is not new: Sparse RAG [Zhu 2025] and Superposition
Prompting [Merth 2024] build it for single-answer retrieval, and Anil et al. [2022] showed an
oracle distractor mask gives perfect length generalization on parity; we use the same forward as
a knockout instrument on an aggregation task and measure what it removes."

## Layout factorial (outputs/sparse/layout/, 5 ep, exact match N=8/16/32/64; gated cells hit the
## snap-to-N wall from N=32 — read within a row, not against the S9b headline)
| per-frame encoding | ungated | oracle-gated |
|---|---|---|
| replica (question in every block) | 1.00 / 0.99 / 0.49 / 0.30 | 1.00 / 0.95 / 0.75 / 0.54 |
| qfirst-once | 0.95 / 0.88 / 0.46 / 0.22 | 1.00 / 0.91 / 0.70 / 0.60 |
| qlast-once (blind blocks) | 0.45 / 0.38 / 0.26 / 0.24 | 0.97 / 0.90 / 0.69 / 0.53 |
Reading: conditioning and selection are one job — inside each block (replica) or at the read (gate);
the read cannot do it alone. Pre-registered prediction (STATE.md 2026-09-15 02:00) MET.

# Prior art: attention sink × temperature/log-N scaling × retrieval (NIAH)

Written 2026-09-23 from a 17-agent literature workflow (48 candidates, 12 full-text verifications) for Tal's question:
"there are papers about the sink — do they talk that temperature doesn't help for NIAH tasks because of the sink token?"
Our own measurement it is compared against: DIAG D3 (docs/DIAGNOSTICS_2026-09-22.md §7; outputs/diag/STATE.md).

## Sink × temperature/scaling × retrieval — synthesis of 12 verified readings

| Paper | Year | Model / setting | What it shows about sink × scaling × retrieval | Verdict |
|---|---|---|---|---|
| Yu et al., ACT (2406.15765) | 2024 | Llama2-7B-chat (+6 others), frozen, inference-time, per-head | Only temperature ablation: θ=1.1 *sparing the first-token sink* → MMLU 44.89 vs vanilla 46.50; mass-conserving sink-shrink 46.82; pushing mass *to* the sink worse. No NIAH; no mass-flow measurement. | ADJACENT |
| Zhang et al., attention entropy (2412.16545) | 2024–25 | Llama-3.1-8B, Mistral-7B, Qwen2-7B, frozen | Shared-prefix sink "absorb[s] unneeded attention values"; hard top-K block selection (= a gate) beats the sink fix on RULER NIAH. No temperature. | ADJACENT |
| Veličković et al. (2410.01104) | 2024–25 | From-scratch max-retrieval head; Gemma 2B CLRS-Text | Inference-only adaptive τ works only on the sink-free toy; on pretrained Gemma "does not empirically work well", attributed to multi-token numbers, not sink; success only trained-in, all heads. | ADJACENT |
| Zuhri et al., Softpick (2504.20966) | 2025–26 | 340M/1.8B from scratch | Sink removed (0% sink rate) yet passkey no better; "underscoring": the normaliser, not a sink, eats needle mass. Scalable-softpick mixed. | ADJACENT |
| Ye et al., DySCO (2602.22175) | 2026 | Qwen3-4/8/32B, Llama-3.1-8B, frozen, all heads | Uniform τ helps only at τ=0.9–0.975 (logits ×1.03–1.11), "degrades performance in several other cases"; gold-edge absolute mass falls with N (vanilla); no sink, no post-scaling mass. Selective +log β on retrieved tokens beats τ. | ADJACENT |
| Lee et al., SEAL (2501.15225) | 2025 | LongChat/Llama-3.1, trained-in per-head | Scales value/output pathway, not logits; no sink, no temperature. | NOT RELEVANT |
| Huang et al., TDA (2601.12145) | 2026 | GPT-2 162M from scratch | "Sum-to-one constraint forces attention sinks"; SSMax flagged as still sum-to-one but never run on retrieval; no temperature. | ADJACENT |
| Bansal et al., qTTT (2512.13898) | 2025 | Qwen3-1.7–32B, frozen, W_Q test-time updates | Score-dilution lemma: mass concentrates on the max-logit token; Ω(log T) margin; needle mass 0.46→0.04 measured, undecomposed; two-party theory; no temperature, no sink. | ADJACENT |
| Gao et al., First Drop of Ink (2605.10828) | 2026 | Llama-3.1-8B-Instruct, frozen, global τ=0.9 | Inference-time sharpening "consistently degrades performance" on retrieval-with-distractors; explained by train/inference calibration; "other tokens" term dropped as negligible; no sink, no mass under τ. | ADJACENT |
| Wang et al., NoPE length gen (2404.12224) | 2024 | 1.1B NoPE/RoPE from scratch | Uniform eval-time λ rescues NoPE passkey to 8×; fails on RoPE at any λ (0.8–1.4); per-head HeadScale trained-in; entropy only, no sink. | ADJACENT |
| Nakanishi, SSMax (2501.19399) | 2025 | 162M from scratch, per-head learnable s | log-n scaling gives NIAH to ~10× length only when trained from start; post-hoc + SFT partial; needle score only, no sink. | ADJACENT |
| Yang et al., APE (2502.05431) | 2025 | Llama-3/3.1-8B, Mistral, Gemma-2, frozen, global T,S | Sharpens context pool only, keeps sink-holding prefix at T=1, adds pool exponent S because sharpening "will also alter the overall attention allocated to the whole context" (Eq. 5 = three-pool renormalisation); τ alone +0.59%. No NIAH, no sink-mass measurement. | ADJACENT |

### 2. Do papers say temperature fails on NIAH because of the sink?

**No verified paper does.** Nothing among the twelve (a) reports freed mass flowing to the sink under scaling, (b) names the sink as why scaling fails on retrieval, or (c) theorises a sink/relevant/competitor three-way trade-off. Every theory is two-party (qTTT Lemma 2.2; SSMax Sec 2.2; Gao Lemma 4.1 explicitly drops the "other tokens" term c as negligible).

What exists is the *failure* without the *mechanism*, three times independently on pretrained models: Gao et al. ("decreasing temperature consistently degrades performance across all hard proportions" — blamed on calibration), Veličković et al. (inference-only "does not empirically work well" — blamed on tokenisation), DySCO UniAttnS (gains only at ×1.03–1.11, inconsistent, unexplained), plus ACT's Temp < vanilla on MMLU. None measures attention after scaling.

The nearest to naming the sink are implicit: APE's engineering (sharpen the context pool, leave the sink prefix untouched, add S to fix the pool's aggregate share), TDA's "sum-to-one forces sinks" + SSMax "still sum-to-one" (never joined), Zhang's sink as absorber of surplus, and qTTT's lemma, which predicts multiplicative scaling sends mass to whichever token holds the max logit — a dominant sink — but the paper never says so.

**What our measurement adds:** the first explicit mass-conservation accounting under scaling on a pretrained model with a sink (sink share 0.42→0.71 at N=32, 0.39→0.73 at N=128; needle absolute mass 0.046→0.026 while its edge over competitors rises ×2.7→×3.5; EM 0.60→0.22); on a production VLM with frames as units; at both N; with a log-N null. qTTT/DySCO measured needle mass falling with N but never under scaling and never where it went.

### 3. Must-cite and wording

Cite: Veličković 2024; Chiang & Cholak 2022; Nakanishi 2025 (SSMax); Ye et al. 2026 (DySCO); Gao et al. 2026; Bansal et al. 2025 (qTTT); Wang et al. 2024 (NoPE); Yu et al. 2024 (ACT); Huang et al. 2026 (TDA); Yang et al. 2025 (APE); Zhang et al. 2025; Zuhri et al. 2026 (Softpick); Xiao 2023, Gu 2024, Barbero 2025 for the sink itself.

Safe: "Prior work reports that inference-time sharpening fails or gives marginal gains on pretrained models (Veličković; Gao; Ye), attributing it to tokenisation or calibration; we measure where the mass goes." Unsafe: "first to show temperature fails on pretrained models"; "scaling never helps NIAH" (SSMax/HeadScale trained-in do); "the sink is provably the cause" — Softpick shows the normaliser alone attenuates needle mass even with no sink.

**Gap:** Barbero 2025 and Gu 2024 were *not* verified for temperature content here; Barbero's over-mixing theory is the natural (c) precedent and must be read before claiming novelty on the third-party framing. Also unread: VAR (visual sink in LMMs — directly relevant to frames as units), "Attention Sinks Are Provably Necessary", Garbage Attention, SinkProbe, Ghaffari.

### 4. Strongest counter-evidence and why it differs

- **SSMax** — NIAH to 10× length: trained from scratch, per-head learnable s, 162M text; post-hoc swap only partial.
- **Wang et al. HeadScale** — passkey to 2–8×: NoPE from scratch; uniform scaling null on RoPE; per-head trained-in.
- **Veličković Gemma 2B** — trained-in adaptive softmax in all heads; CLRS-Text, not NIAH.
- **DySCO** — pretrained, inference-time, but uniform τ helps only at ×1.03–1.11 and inconsistently; the robust gain is a *selective* +log β on retrieved tokens, which raises them over everything including the sink — sink-aware by construction, unnamed.
- **Softpick scalable variant** — sink-free, trained-in, mixed (above softmax at ≤3075, below at 4167).

Common thread: every positive result is trained-in, per-head, from-scratch, text-only, mild, or token-selective; none is uniform eval-time sharpening on a frozen pretrained model with an intact sink — which is our cell.

### Verified readings (paper, verdict)

- Yu, Wang, Fu, Shi, Shaikh, Lin. "Unveiling and Harnessing Hidden Attention Sinks: Enhancing Large Language Models without Training through Attention Calibration" (ACT), arXiv:2406.15765 v1, 22 Jun 2024 (ICML 2024). — **ADJACENT**
- Zhang, Wang, Huang, Fang, Zhang, Deng, Li, Yu (Tencent AI Lab). "Attention Entropy is a Key Factor: An Analysis of Parallel Context Encoding with Full-attention-based Pre-trained Language Models." arXiv:2412.16545 (v2, 25 Jun 2025). — **ADJACENT**
- Veličković, Perivolaropoulos, Barbero, Pascanu — "Softmax is not Enough (for Sharp Size Generalisation)", arXiv:2410.01104 (v3, 30 May 2025; ICML) — **ADJACENT**
- Zuhri, Fuadi, Aji — "Softpick: No Attention Sink, No Massive Activations with Rectified Softmax", arXiv:2504.20966v4 (17 Apr 2026), MBZUAI — **ADJACENT**
- DySCO: Dynamic Attention-Scaling Decoding for Long-Context Language Models — Xi Ye, Wuwei Zhang, Fangcong Yin, Howard Yen, Danqi Chen (Princeton PLI / NYU), arXiv:2602.22175v2 [cs.CL], 16 Apr 2026. Full HTML text read (saved to scratchpad dysco.txt). — **ADJACENT**
- SEAL: Scaling to Emphasize Attention for Long-Context Retrieval (Lee, Seok, Jin, Cho, Park; arXiv:2501.15225v2, 23 Jun 2025) — **NOT_RELEVANT**
- Threshold Differential Attention for Sink-Free, Ultra-Sparse, and Non-Dispersive Language Modeling — Huang, Ding, Ju, Liu, Shah, Zhao (Snap / Oxford / CMU), arXiv:2601.12145 v3 (submitted 17 Jan 2026, revised 09 Jul 2026) — **ADJACENT**
- Bansal, Zhang, Tiwari, Madaan, Duvvuri, Khatri, Brandfonbrener, Alvarez-Melis, Bhargava, Kale, Jelassi. "Let's (not) just put things in Context: Test-Time Training for Long-Context LLMs" (qTTT), arXiv:2512.13898v1, 15 Dec 2025 — **ADJACENT**
- The First Drop of Ink: Nonlinear Impact of Distracting Information in Long-Context Reasoning — Muhan Gao, Zih-Ching Chen, Kuan-Hao Huang (arXiv:2605.10828, submitted 2026-05-11, revised 2026-08-20) — **ADJACENT**
- Wang, Ji, Wu, Yan, Gui, Zhang, Huang, Wang — "Length Generalization of Causal Transformers without Position Encoding" (Findings of ACL 2024), arXiv:2404.12224 — **ADJACENT**
- Scalable-Softmax Is Superior for Attention (Ken M. Nakanishi, arXiv:2501.19399) — **ADJACENT**
- APE: Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding (Xinyu Yang, Tianqi Chen, Beidi Chen; ICLR 2025; arXiv:2502.05431) — **ADJACENT**

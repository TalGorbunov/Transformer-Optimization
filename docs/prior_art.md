# Carrier prior-art sweep (deep threat assessment)

**Date:** 2026-07-24 (Claude web sweep; every arXiv id below opened/verified via arxiv.org unless marked UNVERIFIED)

**Method under assessment:** frozen Qwen2.5-VL; attention masks fence each frame into an island; ONE learned
carrier embedding (~3.5k params, virtual-node token, distillation-trained, backbone frozen) inserted per frame
accumulates that frame's question-conditioned evidence; small LoRA reads the carriers; model emits per-frame
scratchpad + answer. Purpose: repairing multi-frame AGGREGATION (counting/set tasks), framed as over-squashing.

**Prior closest threat:** Wen et al., "Efficient Vision-Language Models by Summarizing Visual Tokens into
Compact Registers" (Victor), arXiv:2410.14072 (verified).

**Addendum (2026-07-25, verified from full texts — relevant to the truncation campaign):**
- **Victor drops visual tokens at layer k=3 of ~32 (≈10% of the tower)**: "At the start of layer k, all
  visual tokens are discarded"; ablation k∈{1,2,3,5}, k≥3 stays "within a 5% performance score loss";
  summarization "primarily occurs in the third layer". Fine-tuning is what makes so early a cut possible —
  cite as precedent for our E3 truncation-layer sweep and for the E4 train-with-truncation fallback.
- **VoCo-LLaMA never truncates mid-stack**: vision tokens run FULL-depth prefill (VoCo tokens absorb across
  all layers); generation is a second forward loading only the VoCo activations as KV. I.e. VoCo ≈ our
  level-1 KV-drop, Victor ≈ our level-2 truncation; each owns one half, neither has frame-fencing or a
  frozen backbone. Our saturation depth is a property of the frozen encoder (signal peaks L14–21), not a
  trainable choice — expect the knee at L12–16, not Victor's L3.

---

## (a) Ranked table — 10 closest works

Legend: Frozen? = is the LM backbone frozen when the tokens are trained/used. Per-unit? = one token (set) per
frame/segment/chunk, vs a global pool. Mask? = deliberate attention-mask surgery. Agg.? = stated purpose is
aggregation/computation capacity (vs compression/efficiency/length). Count? = any counting/aggregation eval.

| # | Work | arXiv (verified) | Frozen? | Per-unit token? | Mask surgery? | Agg. purpose? | Counting eval? |
|---|------|------------------|---------|-----------------|---------------|---------------|----------------|
| 1 | **VoCo-LLaMA** — Ye, Gan, Huang, Ge, Tang | 2406.12275 | No (LLM tuned during vision-instruction tuning; ViT frozen) | **Yes** — VoCo tokens inserted **per frame/segment** in video | **Yes** — text tokens masked from vision tokens, forced to attend only to VoCo tokens | No (576× compression, FLOPs/latency) | No (compression-retention on std. benchmarks) |
| 2 | **Victor** — Wen et al. | 2410.14072 | Mostly (small new trainable params + registers; visual tokens discarded after first layers) | No — ~8 **global** registers for the whole visual input | Partial (visual tokens dropped after layer k ⇒ readout forced through registers) | No (43% train-time cut, 3.3× throughput) | No |
| 3 | **Activation Beacon** — Zhang et al. | 2401.03462 | **Yes** — original LLM params frozen; new beacon module trained | **Yes** — k beacon tokens appended **per chunk**, chunk-wise progressive encoding | Partial (streaming chunked encoding; beacons condense each fine-grained unit) | No (context compression 4K→400K) | No; text-only LLM, no vision |
| 4 | **LLaMA-VID** — Li, Wang, Jia | 2311.17043 | No (LLM instruction-tuned) | **Yes** — 1 **question-conditioned context token + 1 content token per frame** (text-guided cross-attention) | No | No (token-count reduction for hour-long video) | No |
| 5 | **Gist tokens** — Mu, Li, Goodman (NeurIPS'23) | 2304.08467 | No (LM finetuned; gisting learned *by* mask modification) | Per-prompt (not per-frame) | **Yes** — mask blocks later tokens from the prompt, forcing readout through gists | No (26× prompt compression) | No |
| 6 | **Parallel Context Windows (PCW)** — Ratner, Levine et al. | 2212.10947 | **Yes** — training-free, off-the-shelf LLM | No learned tokens | **Yes** — context carved into isolated windows, attention restricted within-window, **positional embeddings reused across windows** | No (context-length extension / ICL) | No |
| 7 | **Recurrent Memory Transformer (RMT)** — Bulatov, Kuratov, Burtsev (NeurIPS'22) | 2207.06881 | No (backbone trained/finetuned with memory) | **Yes** — memory tokens prepended/appended **per segment**, recurrently carried | Segment recurrence (not mask surgery per se) | Partial — extra "reserved capacity" framing | Partial — algorithmic tasks (copy/reverse/associative recall), not visual counting |
| 8 | **Token Turing Machines** — Ryoo et al. | 2211.09119 | No (trained end-to-end) | **Yes** — external memory tokens summarizing frame history, read/write per step | Bounded per-step processing (memory bottleneck by construction) | Closest on purpose — sequential *visual* aggregation (activity detection) | No counting per se |
| 9 | **MA-LMM** — He et al. (CVPR'24) | 2404.05726 | **Yes** (frozen LLM + frozen ViT; trains Q-Former) | **Yes** — online per-frame entries in long-term memory bank (similarity-merged) | No | No (long-video capacity within fixed context) | No |
| 10 | **Pause tokens** — Goyal et al. (ICLR'24) | 2310.02226 | No (must be pretrained/finetuned with pauses) | No — appended pause tokens, not per-unit | No | **Yes** — extra computation width before answering (capacity, not compression) | Reasoning evals (GSM8K etc.), no multi-frame counting |

### Checked and ruled further away (all verified unless noted)
- **Memory Transformer** — Burtsev & Sapunov, 2006.11527: learnable memory tokens prepended for *global* aggregation; trained backbone; ancestor of RMT.
- **Vision Transformers Need Registers** — Darcet et al., 2309.16588 (ICLR'24): registers fix artifact/high-norm tokens; trained with the backbone; no per-frame structure, no frozen-model insertion.
- **StreamingLLM / attention sinks** — Xiao et al., 2309.17453: keeps initial-token KV; no learned tokens (a learned sink token variant is trained-in), efficiency purpose.
- **Landmark Attention** — Mohtashami & Jaggi, 2305.16300: one landmark token *per block* + grouped-softmax block selection — per-unit token + block structure, but finetuned backbone and retrieval (random-access), not aggregation.
- **MovieChat** — Song et al., 2307.16449: training-free frame-token merging into sparse memory; no learned tokens, no masks.
- **FOCUS** — Park et al., 2508.13744: training-free mitigation of *cross-image information leakage* in multi-image LVLMs — but isolates by **noise-masking all-but-one image and aggregating logits**, not attention masks; no learned tokens; by construction it answers per-image, so it cannot do cross-frame aggregation. Useful citation that image entanglement is a recognized pathology.
- **Lee, "Failure Modes of Transformers through the Lens of GNNs"** — 2512.09182 (Dec 2025): the main **post-Barbero** item found. Analysis/unification paper: frames transformer failures (incl. counting-style failures) as GNN information-propagation bottlenecks and re-reads existing fixes (registers/pause tokens ≈ attention dumps) through that lens. Does **not implement** a virtual-node remedy in a frozen model. Must-cite for framing; not a method threat.
- **Can VLMs Count?** — Sengupta et al., 2511.17722: synthetic **single-image** counting benchmark + attention-reweighting interventions (modest gains); no tokens, no masks, no multi-frame.
- **"Diagnosing Long-Video Quantitative Reasoning via Enumeration and Counting"** — arXiv html 2603.29943 (UNVERIFIED details; seen only via search snippet): long-video counting **benchmark** (best models ~24% vs humans ~83%); appears to be diagnosis, not a repair method. Verify before citing.
- Frame-selection lines (Frame-Voyager 2410.03226, Q-Frame 2506.22139 — ids from search results, UNVERIFIED individually): improve counting by *picking* frames, not by adding aggregation capacity; different mechanism class.
- No paper was found that inserts a **virtual-node token into a frozen pretrained transformer as an explicit over-squashing remedy** (searches: "virtual node/token over-squashing transformer remedy frozen", 2024–2026 → only GNN-side surveys and the Lee analysis paper).

---

## (b) Top-3 deep dives + differentiation sentences

### 1. VoCo-LLaMA (arXiv:2406.12275, Ye et al., v2 Mar 2025) — NEW closest threat
What it does: during vision instruction tuning, special VoCo tokens are inserted after each image's (or each
video frame-segment's) vision tokens, and the attention mask is rewritten so **text tokens cannot attend to the
original vision tokens at all — only to the VoCo tokens**. The LLM itself "distills" its own vision
understanding into the VoCo activations (they call it attention/self distillation); at inference the vision
tokens' KV can be dropped, giving 576× compression. In video, each segment gets its own VoCo tokens and the
compressed tokens form a time series, so it *is* per-frame and it *is* mask surgery, inside a VLM, with a
distillation flavor. Differences: the LLM is **fine-tuned** (vision-instruction tuning stage), the mask does
NOT fence frames from each other (vision tokens still see everything before them; only the text→vision edge is
cut), the stated goal and all metrics are **compression retention/efficiency**, and there is no counting or
aggregation-stress eval.
**Differentiation sentence:** *"VoCo-LLaMA rewires attention through learned per-frame tokens to make a
fine-tuned VLM cheaper while preserving what it can already do; we rewire attention through a per-frame carrier
to give a fully frozen VLM an aggregation ability it measurably lacks — fencing frames from each other (not just
text from vision), training only a ~3.5k-parameter carrier by distillation, and evaluating the repaired
capability (multi-frame counting) rather than compression retention."*

### 2. Victor (arXiv:2410.14072, Wen et al.) — previous closest threat, now #2
What it does: appends a handful (~8) of learnable register tokens after the visual tokens of a
LLaVA-style VLM; the first k layers of the language tower summarize all visual information into the registers,
after which the visual tokens are discarded. Very small number of new trainable parameters. All results are
efficiency-framed (<4% accuracy drop, 43% training-time reduction, 3.3× throughput). Registers are a **global
pool** — there is no per-frame allocation, no frame fencing, no question-conditioning story, no counting eval.
**Differentiation sentence:** *"Victor's registers are a global compression pool that trades a few accuracy
points for throughput; our carrier is a per-frame virtual node that buys accuracy on exactly the multi-frame
counting tasks where the frozen model's global attention fails, at zero compression benefit and near-zero
parameter cost."*

### 3. Activation Beacon (arXiv:2401.03462, Zhang et al.) — closest on the *frozen + per-chunk* axes
What it does: for long-context text LLMs, appends k beacon tokens to each chunk of the context; a newly
introduced (trained) beacon module condenses the chunk's activations (all-layer KV) into the beacons'
activations, chunk by chunk, streaming-style; the **original LLM parameters stay frozen** and compatibility is
fully preserved. Trained on short sequences with mixed condensing ratios; extends Llama-2-7B from 4K to 400K.
It is the strongest precedent for "frozen backbone + learned per-unit condensation tokens", but: it is
text-only, the trained component is a large new attention module (not a tiny embedding), chunking exists for
*length*, not to protect per-unit evidence, and the eval is LM/retrieval quality, never aggregation or counting.
**Differentiation sentence:** *"Activation Beacon condenses each chunk so a frozen LLM can read more text;
our carrier condenses each frame so a frozen VLM can count what it read — the token is a ~3.5k-parameter
embedding rather than a new attention module, the fencing is there to prevent cross-frame squashing rather than
to fit more context, and the metric is aggregation accuracy, not perplexity at length."*

---

## (c) Verdict

**Yes — one work is closer than Victor, but only mechanistically: VoCo-LLaMA (arXiv:2406.12275) combines
per-frame learned tokens WITH explicit attention-mask surgery inside a VLM (and even a distillation framing),
so it should replace Victor as the headline "closest prior art" in the thesis; however, it fine-tunes the LLM,
fences text-from-vision rather than frame-from-frame, and — like every work found (Victor, Activation Beacon,
LLaMA-VID, gist/PCW/RMT/TTM/MA-LMM/pause tokens) — its purpose is compression/efficiency/length, not
aggregation repair. NO found work combines a frozen backbone + per-frame carrier token + frame-isolation masks
for the purpose of relieving over-squashing, and none evaluates multi-frame counting — that combination (and
the Barbero-style over-squashing framing with an implemented remedy, cf. the analysis-only 2512.09182) remains
unclaimed.**

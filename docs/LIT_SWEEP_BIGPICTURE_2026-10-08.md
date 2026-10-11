# Big-picture literature sweep — 2026-10-08

Context: after the SCALEUP / UNIT baseline (outputs/scaleup/STATE.md, 2026-10-07), Tal asked for the
bigger picture: (1) why a frozen VLM fails cross-frame reasoning at 3–16 frames while per-frame
perception is fine; (2) whether frame-count extrapolation failure after fine-tuning is documented;
(3) prior work on task-dependent (relational / multi-frame) evidence gating; (4) whether a
hierarchical "query per frame + meta-query over frame queries" design has precedent and whether
any query/compression architecture beats flat token-keeping on accuracy.

Method: three research agents, each required to fetch the arXiv abstract / HTML page for every
claim. Numbers quoted are from the paper text as extracted; "[figure only]" = number lives in a
figure; "[abstract only]" = no full text read; "[UNVERIFIED]" = could not confirm. Venues are
given only where the arXiv page stated them. Treat every number as "verify before citing in the
thesis" — this file is a map, not a bibliography.

Repo precedents already on file that bear on the same questions (docs/PRIOR_ART_MASKGATE_2026-09-16.md,
docs/CONTRIBUTIONS_2026-09-14.md): Das et al. MIMIC 2601.07812 (block-diag image mask + LoRA,
cross-image counting 29.7→51.2 on 7B, still degrades to 35 images); Low-Frequency Trap 2608.06361
(frontier VLMs with oracle keyframes collapse on event counting past 6–10 events); VideoStreaming
2405.16009 (Gumbel top-k over clip memories); PQR/T-Former 2412.19304 (question per frame inside a
Q-Former); Block-Attention 2409.15355 (question-agnostic isolation costs 20 pts zero-shot);
Thorne Neural Databases (FiD "cannot perform aggregation queries").

---------------------------------------------------------------------------------------------------
## 1. Few-frame cross-frame failure while per-frame perception is fine (benchmarks)

- **TVBench "Lost in Time"** 2410.07752 (2024). Prior video benchmarks solvable from one frame / text /
  world knowledge; TVBench removes the shortcuts and most models fall to chance. Text-only Llama-3-70B
  34.7 %, GPT-4o single image 35.8 %, best Tarsier-34B 55.5 %, Qwen2-VL-72B 52.7 %, humans 94.8 %
  (random 33.3 %). Shuffling frames: Tarsier 55.5→38.0, Qwen2-VL-72B 52.7→36.8. Reversed video
  Tarsier 27.2 % (below random). Contrast: Llama-3-70B text-only 47.6 % on NExT-QA.
- **TOMATO** 2410.23266 (2024). "Models can accurately recognize events in isolated frames but fail to
  interpret these frames as a continuous sequence." 16 frames; human 95.2 % (79.7 % at 16 frames),
  random 18.5 %, best model 37.9 %. Action count: Qwen2-VL-72B 42.8, GPT-4o 36.0; rotation ≈ chance.
  "Existing models are limited to understanding events interpretable in ≤8 frames" (plateau beyond 8).
- **TemporalBench** 2410.10818 (2024). Binary vs multiple-binary (chance 50 / 9.5): GPT-4o 75.7 / 38.5,
  Qwen2-VL-72B 75.8 / 38.3, humans 89.7 / 67.9. "Models struggle most on action frequency (counting)."
  "Performance generally saturates around 8–16 frames."
- **Vinoground** 2410.02763 (2024). Counterfactual order on <10 s clips; GPT-4o text/video/group
  54.0 / 38.2 / 24.6 (chance 25 / 25 / 16.7), humans 93.4 / 94.0 / 90.0; open models ≈ chance.
  GPT-4o vs frames: 1→28.2, 8→53.6, 32→54.0, 64→49.0 ("too many frames can worsen").
- **TemporalVQA** 2501.10674 (2025). Two consecutive frames, "which happened first": GPT-4o 49.1 %
  consistent (chance 50), Gemini-1.5-Pro ~35, Qwen-VL ~9.2, LLaVA-NeXT ~3.5, humans 90.3.
- **MuirBench** 2406.09411 (2024). 2–9 images (mean 4.3). Ordering: GPT-4o 23.44 vs chance 24.12
  (at chance); counting GPT-4o 49.15 vs chance 20.98; open single-image models < 33 %.
- **MMIU** 2408.02718 (2024). 2–32 images; GPT-4o 55.7 %; weakest on reasoning-intensive and spatial
  relations [per-type figure only].
- **BLINK** 2404.12390 (2024). Low-level perception benchmark; GPT-4V 51.3 vs random 38.1. Blames
  PERCEPTION — do not cite as an aggregation gap.
- **Visual Haystacks** 2407.13766, ICLR 2025. Single-needle yes/no: GPT-4o 82.5 (N=1) → 68.2 (10) →
  55.3 (100); Qwen2-VL 80.9 → 62.6 (10) → 52.6 (50). Multi-needle ALL/ANY with ONLY the needle
  images shown: "sharp performance decline … from one to two" [figure only]; Table C.4 multi-needle
  GPT-4o 68.5 (N=5) → 57.8 (100), Qwen2-VL-7B 58.8 → 55.8 (50). Position bias up to 25 %. MIRAGE
  (visual RAG) +13 %, fixes retrieval not integration.
- **VideoNIAH / VNBench** 2406.09367, ICLR 2025. Standard-length: GPT-4o retrieval 95.3 / ordering
  73.4 / counting 24.5; Gemini-1.5-Pro 90.7 / 72.9 / 36.7; LLaVA-NeXT-Video-7B 44.2 / 0.4 / 15.5;
  Qwen2-VL-7B retrieval 69.1 / ordering 12.4 / counting 20.0. By duration (10–30 / 30–60 / 60–180 s)
  GPT-4o retrieval 96 / 96 / 94, ordering 83.7 / 76.6 / 60.0, counting 31.3 / 27.3 / 14.9. Frames
  16→128 (LLaVA-OV): retrieval 23.3→39.7, "counting exhibits only marginal improvements"; counting
  declines with number of needles [figure only]. Timestamps in the prompt help ordering + counting.
  NOTE: fixed frame budgets — "longer video" = sparser sampling, not more frames.
- **MM-NIAH** 2406.07230, NeurIPS 2024. Interleaved documents 1K–72K tokens. InternVL-1.5 text
  retrieval 99.0 (1K) → 72.7 (64K) vs text counting 67.6 → 6.8; image counting 30.6 → 0.0 by 32K;
  ">5 images → accuracy < 10 %"; counting "model collapse" (format-valid lists of wrong length).
  RAG helps text retrieval, not image needles.
- **MLVU** 2406.04264 (2024). "Most methods, except GPT-4o and Video-XL, fail entirely in action
  order (AO) and action count (AC)"; GPT-4o AC 35.0 / AO 46.2; best open 23.3 / 35.7.
- **Video-MME** 2405.21075 (2024). "Counting problems are a joint bottleneck"; accuracy declines with
  duration under FIXED frame counts (sparser sampling).
- **MVBench** 2311.17005 (2023). GPT-4V 43.5 vs random 27.3; action count 39.0, moving count 12.0;
  VideoChat2 on counting ≈ its text-only variant.
- **VSI-Bench** 2412.14171. Object count from video: Gemini-1.5 Pro 56.2, GPT-4o 46.2, humans 79.2;
  appearance order 34.6 / 28.5 vs humans 95.8; frame count "only marginally affects performance";
  CoT / ToT prompting −4 %.
- **VLMs are blind** 2407.06581 (2024). Single-image geometric tasks 58 % avg; "vision encoders
  contain sufficient visual information … language models fail to decode" — single-frame cousin.
- Single-frame-bias classics: Lei/Berg/Bansal 2206.03428 (single-frame + ensembling beats
  multi-frame); ATP Buch et al. 2206.01720 CVPR 2022 ("event temporality often not necessary");
  Ranasinghe 2403.16998 ICLR 2025 (LLM-only baselines strong on long-video QA); TempCompass
  2403.00476; ViLMA 2311.07022.
- **MIMIC / Das et al.** 2601.07812 (2026). Cross-image counting Qwen2-VL-7B 35.7 %, InternVL2-8B
  18.9 %, LLaVA-OV-7B 29.7 %, GPT-4o 49.2 %; ~79 % when all instances in one image → near 0 % spread
  over 7 images. Mechanism: inter-image attention only in early layers, intra-image later; causal
  attention accumulates noise on later images. Intra-image mask during fine-tuning: 29.7→51.2.
- **PushupBench** 2604.23407 (2026). Repetition counting; best frontier 42.1 % exact, open 4B ~6 %;
  "weaker models exploit the modal count"; 1k counting samples transfer (+4.5 TVBench).
- Negation / "did NOT happen": **Learning to Deny** 2606.31187 ECCV 2026 (20 MLLMs > 85 % on action
  presence vs < 50 % on matched action denial, "affirmation bias"); **NOAH** 2511.06475 (inserted-clip
  composites; errors from narrative prior, worse with fewer frames); **NegBench** 2501.09425 CVPR 2025
  (negation at chance; CLIP-side fix).
- **M³-Verse** 2512.18735 (2025). Spot-the-difference across two videos; best Gemini-2.5-pro 47.6 %
  (chance 24.1, humans 89.8); text-oracle probe (dense captions → text QA) +7 InternVL3.5-8B, +4
  Qwen3-VL-8B — serialising to text does not remove the gap.

## 2. Where the bottleneck is (diagnostics)

- **Temporal Reasoning Transfer T3** 2410.06166 (2024). Linear probes on visual tokens inside the LLM
  separate temporally contradicting videos at > 90 %; with GPT-4o frame captions instead of frames
  the backbone LLMs (Qwen2-7B, LLaMA3-8B) still fail Referring / Grounding / Order and are at chance
  on Attribute → "the underlying LLM's inherent difficulty with temporal concepts". > 70B LLMs do
  much better. Fix: text-only synthetic temporal QA (no video): LongVA-7B TempCompass 55.9→61.2
  (Order 54.3→65.2), Video-MME temporal 37.3→49.7. [per-aspect probe numbers figure only; captions
  are GPT-4o-generated, not human GT]
- **VBenchComp** 2505.14321 (2025). Video-MME: 31 % LLM-answerable, 30 % shuffle-invariant, 13.7 %
  truly temporal; MLVU 28.6 / 29.6 / 17.6. GPT-4o / Gemini "remarkably stable" under frame shuffling.
- **Causality Matters** 2508.11576, AAAI 2026 (Qwen2.5-VL-7B, LLaVA-OV-7B, TempCompass). Temporal
  cues are "progressively synthesized through inter-frame attention, aggregated in the final frame,
  and subsequently integrated into the query tokens"; PE ablation in deep layers ±2 %; reversing
  frames −15.45 % vs reversing position ids −1.1 to −1.8 %; blocking inter-frame attention hurts
  most near layer 10; disabling inter-frame attention in layers 20–28 costs nothing; query attending
  only to the last frame loses < 2 %. ⇒ the frozen model's aggregation path is a funnel through one
  position (the last frame → query tokens).
- **Before It Fades** 2610.01595, NeurIPS 2026 [abstract only]. Temporal information peaks at
  intermediate layers and decays toward the output; training-free Temporal Activation Injection on
  Qwen2.5-VL-7B / Qwen3-VL-8B / InternVL2.5-8B gives "consistent improvements" (numbers not read).
- **Seeing the Arrow of Time** 2506.03340, NeurIPS 2025. Forward vs reversed: Qwen2.5-VL-7B 50.0 /
  51.6, LLaVA-Video-7B 50.0 / 51.6, GPT-4o 52.8 / 54.0 (chance 50). ArrowRL: Qwen2-VL-7B +19–21,
  Qwen2.5-VL-7B only +1.4 / +3.2 on direction; Vinoground group +10.8.
- **Tracing the Arrow of Time** 2605.07568 (2026). COUNTER-position: loss is in encoder / projector
  ("Q-Former disrupts temporal information; time-preserving MLP projection substantially improves");
  98.1 % arrow-of-time with video-centric encoder + AoT supervision.
- **MLLMs Know When Before Speaking** 2605.21954 (2026). Grounding heads know the interval at prefill
  and drift during decoding; read-then-regenerate +3.5 mIoU.
- **How Important are Videos** 2506.06928 (2025). Image-only LongVU above chance on TVBench;
  fine-tuning on ordered annotated image sequences ≈ video-trained models.
- **Unveiling the Ignorance** 2406.10638. Single-image: right attention, wrong answer; low visual
  attention share.

## 3. Remedies beyond plain SFT (what moved, what did not)

- Text-only temporal data (T3): +5.3 TempCompass, +12.4 Video-MME temporal; grounding-only data hurts.
- RL: **Video-R1** 2503.21776 NeurIPS 2025 (T-GRPO ordered-vs-shuffled reward; Qwen2.5-VL-7B, 16
  training frames): SFT-only vs Video-R1 VSI 31.8→34.6, MVBench 59.4→62.7, TempCompass 69.2→72.6,
  Video-MME 52.8→57.4; at 64 test frames further up (37.1 / 64.8 / 73.2 / 61.4) [no count/order
  breakdown]. **VideoChat-R1** 2504.06958: grounding +31.8, tracking +31.2, general QA ~+1; "thinking
  does not necessarily lead to gains" on spatio-temporal perception; grounding-CoT WORSE than direct
  on NExT-GQA (32.9 vs 35.1 mIoU). **TimeZero / Time-R1** 2503.13377 NeurIPS 2025: single-span
  grounding only. **AV-Reasoner** 2506.05328: "reasoning in the language space fails to bring gains"
  OOD on counting.
- Caption-then-reason: LLoVi 2312.17235 (EgoSchema +18.1), Vamos 2311.13627 ECCV 2024 (text
  representations ≈ visual), VideoAgent 2403.10517 (EgoSchema 54.1 with 8.4 frames avg), memory
  VideoAgent 2403.11481. All on benchmarks shown to be largely LLM/single-frame answerable; none
  reports a counting / ordering / negation cell. M³-Verse oracle-text probe leaves best model ~48 %.
- Grounding-then-answer: VideoMind 2503.13444 ICLR 2026 (ReXTime 69.2→74.6 — the one verified
  cross-segment QA gain; grounder → verifier keeps ONE span), TRACE 2410.05643 ICLR 2025, TimeSuite
  2410.19702 ICLR 2025 (EgoSchema +5.6, Video-MME +6.8), Chain-of-Frames 2506.00318 CVPR 2026
  (InternVL3-8B Video-MME 66.5→73.7, VSI-Bench 41.0→51.3, synthetic data helps). No occurrence
  counting via grounding anywhere; E.T. Bench 2409.18111 multi-event tasks near zero (EPM 10.2,
  GVQ 3.7). NExT-GQA 2309.01327: only 16 % of correct answers are grounded; grounding supervision
  +0.8–1.7 QA.
- Training-free: Temporal Activation Injection (2610.01595, unquantified), ViKey 2603.23186 CVPR 2026
  (ordinal frame indices overlaid), read-then-regenerate (+3.5 mIoU); prompting hurts on VSI-Bench.
- Attention / data: MIMIC intra-image mask +21.5 counting (still degrades with image count);
  procedural multi-image data 26.4→45.5 (0.5B); PushupBench counting SFT transfers in-distribution;
  "Can VLMs Count?" 2511.17722 (single image; attention re-weighting "modest").

## 4. Frame-count extrapolation (train-short / test-long on the frame axis)

- **LongVA** 2406.16852 (2024). Text context 224K → 2000 frames with image-only training; V-NIAH
  holds to ~3000 frames then declines; Video-MME vs frames 8→47.9, 16→49.7, 32→51.8, 64→52.4,
  128→52.6, 384→51.8 (non-monotonic).
- **VideoRoPE** 2502.05173 (2025). M-RoPE temporal axis on high-frequency dims → periodic collisions
  (V-NIAH-D distractors). Trained 8k, eval 8–64k: V-NIAH / V-NIAH-D avg Vanilla 31.8 / 30.2, M-RoPE
  78.7 / 74.7, VideoRoPE 91.1 / 87.1; 64k LongVideoBench M-RoPE 54.35 → 57.26 [per-length table not
  extracted].
- **HoPE** 2505.20444, NeurIPS 2025. Proves all current multimodal RoPE frequency allocations fail
  "semantic preference" beyond a critical length. Qwen2-7B trained 8k, Video-MME at 8k/16k/32k/64k:
  RoPE 56.7 / 58.0 / 58.0 / 26.1; M-RoPE 56.8 / 57.8 / 58.4 / 23.4; VideoRoPE 57.5 / 59.0 / 59.1 /
  26.5; HoPE 57.7 / 59.3 / 59.4 / 27.3 → at 8× training length EVERY PE collapses to ~25 %.
  Retrieval prefers small temporal scaling γ, QA prefers large (trade-off).
- **V2PE** 2412.09616 (2024). Visual position ids outrun the window; InternVL2-2B MM-NIAH image
  retrieval 33.7 % @16K → 0 % @1M; fractional increments δ: retrieval best at 1/256, Long-VQA
  reasoning at 1/16 (task-dependent).
- **Apollo** 2412.10360, CVPR 2025. Uniform-N sampling makes "a different video speed in every
  iteration"; fps sampling "vastly preferable"; "gap not due to the number of frames at test time";
  fps methods revert to uniform past ~64 frames. The only paper framing fixed-count sampling as a
  training-distribution shift.
- **LongVILA** 2408.10188. "The baseline 32-frame model can not retrieve right needles after 32
  frames"; 2048-frame training → 99.8 % NIAH at 6000 frames.
- **Video-XL** 2409.14485, CVPR 2025. 100 % NIAH within 128 training frames, ~95 % beyond to 2048;
  randomized compression ratios in training generalize.
- **InternVideo2.5** 2501.12386. Dynamic 64–512 training frames; NIAH holds to 3000 frames, then
  degrades — attributed to train/test length discrepancy.
- **STORM** 2503.04130. VILA baseline Video-MME ≈61 @32f → 60 @64f → 57 @128f (trained 32);
  cause "smaller effective context length in practice".
- **LLaVA-Video** 2410.02713. Trained 32 frames (1 fps ≤ 64): test 32 / 64 / 110 frames Video-MME
  59.1 / 59.9 / 58.8, EgoSchema 56.3 / 56.3 / 55.2 — "excessively increasing inference frames can
  degrade"; small (≈1 pt at 3.4×).
- **InternVL 2.5** 2412.05271. InternVL 2.0 "best at 16 or 32 frames but degrade with more"; fixed by
  widening the training frame range 4–24 → 8–32.
- **LongVideoBench** 2407.15754, NeurIPS 2024. Idefics2 16f 49.7 → 32f 47.8 → 64f 30.9; Mantis-Idefics2
  47.0 → 30.2 @64f, "before they have reached their context length limits"; proprietary monotonic
  (GPT-4o 41.7 → 58.0 → 63.5). "(L2) Relation is more challenging than (L1) Perception." No mechanism.
- **Qwen2-VL** 2409.12191: M-RoPE id compression "enables extrapolation"; 16K training tokens/video,
  80K at inference on Video-MME (one curve, no per-task). **Qwen2.5-VL** 2502.13923: dynamic fps +
  absolute-time MRoPE for consistency across fps; eval cap 768 frames / 24,576 tokens; NO
  extrapolation curve in the report.
- **MMLongBench** 2505.10610, NeurIPS 2025 D&B. NIAH does not correlate with other long-context
  tasks; Qwen2-VL-72B generalizes 32K → 128K on average (51.9).
- **Mantis** 2405.01483 TMLR 2024: ≤ 50 images in training, no beyond-training analysis (see
  LongVideoBench collapse). **MileBench** 2404.18532: Few/Medium/Many bins, most models decline at
  32–109 images. **MMNeedle** 2406.11230 NAACL 2025: GPT-4o 97.0 (1×1) → 1.0 (8×8) at 10 images.
- Position bias: Tian et al. 2503.13792 CVPR 2025 (recency bias, worse with length; SoFA);
  Video-LevelGauge 2508.19650; "Lost at the End" 2606.16494 (primacy bias in multimodal RAG).
- Frame-sampling bias: 2509.14769 ("Frame Sampling Strategies Matter"); Beyond Frame Selection
  2608.05592.
- Text anchors for retrieval-vs-aggregation: RULER 2404.06654, BABILong 2406.10149, NoLiMa
  2502.05167 ICML 2025, HoloBench 2410.11996 ICLR 2025, Oolong 2511.02817.

## 5. Question-conditioned frame / segment selection (selector type in brackets)

PF = per-frame independent; SET = joint over sets; ITER = agentic; CLIP = frozen scorer;
VLM-self = the video LLM scores; TRAINED = trained module.

- **SeViLA** 2305.06988 NeurIPS 2023 [PF, TRAINED; BLIP-2 Q-Former localizer]. NExT-QA 4 frames:
  uniform 62.4 → untrained localizer 63.6 → trained 64.6; STAR 42.2 → 44.6 → 45.5. "The untrained
  BLIP-2 Localizer results in only a minor improvement." ⇒ self-gate +1.2, trained gate +2.2.
- **ViLA** 2312.08367 ECCV 2024 [PF per segment, TRAINED by answer loss, Gumbel]. NExT-QA temporal
  71.0 vs SeViLA 67.7 vs uniform 67.2; STAR Sequence 70.4 vs 70.4 (zero gain on the ordering split).
- **Frame-Voyager** 2410.03226 ICLR 2025 [SET, TRAINED; ranks frame COMBINATIONS by answer loss].
  VILA-8B Video-MME 47.5→50.5, MLVU 46.3→49.8; per type recognition +4.8, reasoning +3.2,
  perception +2.1; counting: "one additional error" (no gain).
- **AKS** 2502.21271 CVPR 2025 [PF relevance + coverage, CLIP-class]. LongVideoBench +3.8–5.0,
  Video-MME +0.9–2.3; authors: LVB questions are single-moment, Video-MME "require the model to
  collect information from multiple moments" — per-frame relevance "struggles with multi-moment".
- **LVNet** 2406.09396 [hierarchical: cluster → CLIP → GPT-4o over collages]. NExT-QA temporal 65.5
  (weakest split).
- **VideoAgent** 2403.10517 [ITER]. Temporal questions consume the most frames; temporal lowest.
- **VideoTree** 2405.19209 CVPR 2025 [ITER over cluster captions]. NExT-QA temporal 70.6 (weakest).
- **Goldfish** 2407.12679 ECCV 2024; **Video-RAG** 2411.13093 NeurIPS 2025; **KeyVideoLLM**
  2407.03104 [abstract only; retrieval as gate; no count/order cells].
- **BOLT** 2503.21483 CVPR 2025 [PF, CLIP, inverse-transform sampling]. Video-MME 53.8→56.1; NExT-QA
  "minimal improvement"; admits similarity "may not reflect true relevance on tasks that require
  reasoning".
- **Q-Frame** 2506.22139 ICCV 2025 [PF, CLIP]. Qwen2-VL Video-MME counting 30.6→38.8 (small subset),
  MLVU 56.9→65.4; admits "lacks explicit temporal modeling … fail to capture transitional moments".
- **M-LLM frame selection** 2502.19680 CVPR 2025 [TRAINED; per-frame + temporal pseudo-labels].
  NExT-QA +1.2; spatial-only 54.2 vs +temporal 55.5 (ActivityNet-QA); long video ≈ 0.
- **GenS** 2503.09146 ACL 2025 Findings [SET, VLM-self (25B Aria), TRAINED]. LongVideoBench
  LLaVA-Video-7B 58.9→63.3; beats CLIP/SigLIP/GPT-4o retrieval. MLVU per type: Action Order +9.6,
  Needle +7.6, **Action Count −2.9**, Holistic Multi −2.4 ("uniform sampling provides better
  coverage" for holistic questions).
- **VideoITG** 2507.13353 CVPR 2026 [SET, TRAINED; instruction-type-dependent policies: semantic →
  diverse frames; motion → fixed-rate within localized segment; hybrid; non-clue]. LLaVA-Video-7B
  Video-MME 75.8→77.3, MLVU 70.2→74.6; no per-category numbers. Closest to a question-type-aware gate;
  multi-event handling is a hand-designed sampling policy.
- **T*** 2504.02259 CVPR 2025. "SOTA keyframe selection methods achieve only 2.1 % temporal F1" on
  LVBench haystacks; LongVideoBench XL LLaVA-OV-72B 56.5→62.4.
- **Logic-in-Frames / VSLS** 2503.13139 [order / causal predicates → per-frame confidences,
  training-free, YOLO-World verified]. Beats T* slightly; "for the time relation category … a slight
  decrease in temporal coverage".
- **ReFoCUS** 2506.01274 [SET, RL, reward = reader margin]. Video-MME 62.2→65.4 (long 51.9→56.8),
  NExT-QA 48.7→53.7; no per-type analysis.
- **TSPO** 2508.04369 AAAI 2026 [SET, RL, joint selector+answer policy]. MLVU 70.3→76.3, LVBench
  40.2→45.3, Video-MME only +1.1–1.8 ("holistic comprehension rather than localization").
- **ViaRL** 2505.15447 [RL selector; "~15 % on Needle QA"]. **QuoTA** 2503.08689 [PF, VLM-self soft
  budgeting, +3.2 avg]. **HFS** 2512.11534 ACM MM 2026 [SET objective rel + cov − red, Gumbel-TopK].
  **Event-Anchored** 2603.00983 [event partition + anchor + MMR; flat selection "temporally-agnostic"].
- **HiMu** 2603.18558 (2026) [relational SET composition: query → logic tree of predicates routed to
  CLIP / YOLO-World / OCR / ASR; fuzzy AND / OR / SEQ / RIGHT_AFTER → per-frame satisfaction curve].
  Qwen3-VL-8B Video-MME K=16: 73.23 vs top-K 72.37 (+0.9); motivation: prior selectors flatten
  "sub-event ordering". No counting / negation.
- Others 2025–26 [abstract only]: FORTE 2610.00573, EC-RAG 2610.08674, ReMem 2607.24794 ECCV 2026,
  Kwon et al. 2603.14953, Vgent 2510.14032 NeurIPS 2025 (graph over clips, aggregation in TEXT).
- **TempCore** 2509.01167 EMNLP 2026. Frame-selection sensitivity: only 5.5–31 % of samples across
  6 benchmarks are temporally sensitive; "anti-key" samples where relevant frames HURT: MVBench 1.9 %,
  MLVU 8.2 %, LongVideoBench 6.4 %, Video-MME 6.4 % — relevance selection "selects close-up shots of
  salient objects, but the question asks about absence or subtle state changes".

## 6. Grounding-then-answer

VTimeLLM 2311.18445; TimeChat 2312.02051 CVPR 2024 (frame Q-Former + sliding video Q-Former;
Charades-STA +27.5 R@0.5); HawkEye 2403.10228; Momentor 2402.11435 ICML 2024; TRACE 2410.05643 ICLR
2025 (event sequences for captioning only); Grounded-VideoLLM 2410.03290; TimeSuite 2410.19702;
VideoMind 2503.13444 ICLR 2026 (top-5 → verifier → ONE span; ReXTime +5.4; error budget grounding
11–18 %); TimeZero / Time-R1 2503.13377; VideoChat-R1 2504.06958; E.T. Bench 2409.18111; NExT-GQA
2309.01327; ReXTime 2406.19392 (question and answer in different segments; frontier −14.3 vs humans).
Verdict: every grounder commits to one span; multi-event outputs exist only for captioning; no
occurrence counting via grounding; grounding-CoT can hurt.

## 7. Hierarchical / question-conditioned query compressors

- **InstructBLIP** 2305.06500. Removing the instruction from the Q-Former: ScienceQA 70.4→63.4,
  VizWiz 32.7→25.1, GQA 48.4→45.9; "more severe in … spatial (ScienceQA) or temporal (iVQA)
  reasoning". Video = 4 frames, per-frame Q-Former, concatenated. No flat-token comparison.
- **Video-LLaMA** 2306.02858. Frame Q-Former → temporal PE → Video Q-Former (NOT question-
  conditioned). No quantitative ablation.
- **LLaMA-VID** 2311.17043. Context token = text-query cross-attention over patches, mean-pooled
  (question-conditioned per frame) + content token. Image ablation: n=1 both tokens GQA 55.5 vs
  n=256 content tokens 63.0 → flat tokens win decisively. No matched-frame video ablation.
- **MA-LMM** 2404.05726 CVPR 2024. Unconditioned memory-bank compression: LVU / Breakfast / COIN
  concat 60 frames (1920 tok) 62.6 / 90.4 / 93.0 vs MBC 100 frames (32 tok) 63.0 / 93.0 / 93.2 —
  one of two "compressed ≥ flat" results; unmatched frames, classification tasks.
- **Koala** 2404.04346 CVPR 2024 (key-frame-conditioned queries, +3–6 zero-shot); **MovieChat**
  2307.16449 (memory consolidation) — not question-conditioned.
- **VideoStreaming** 2405.16009 NeurIPS 2024. 64 tokens/clip; question-conditioned Adaptive Memory
  Selection keeps 4 clip memories. EgoSchema / MovieChat global / breakpoint: memory-only 37.3 / 69.1
  / 23.0, selection-only 38.4 / 43.8 / 39.1, both 44.1 / 90.4 / 54.9. No "all memories flat" row.
- **LongVU** 2410.17434. Text-guided frame reduction: +1.1 EgoSchema / +2.5 MLVU over DINO-only;
  MLVU per type Count 28.98, Order 58.30, Needle 76.33 — counting worst by ~30.
- **BLIP-3-Video** 2410.16267. Unconditioned temporal encoder (TokenLearner / TTM / Perceiver) → 16–128
  tokens; 128 tokens vs Tarsier-34B 4608 tokens: 77.9 vs 79.3 MSVD-QA. No all-tokens baseline.
- **HierarQ** 2503.08585 CVPR 2025. CLOSEST TO THE PROPOSED DESIGN: prompt-guided modulators at two
  levels (entity stream from extracted nouns; scene stream from full prompt) → entity Q-Former →
  scene Q-Former cross-attending the entity queries, with memory banks; LLM sees fixed-count scene
  queries. LVU / Breakfast: baseline 60.7 / 93.0 → +prompt modulator 62.0 / 94.1 → +hierarchy 65.7 /
  95.2 → full 67.9 / 97.4; "w/o hierarchy" −3.6 LVU. Classification-style sets only; no counting /
  ordering / MVBench; no flat-token comparison.
- **VideoDetective** 2512.17229 (question-aware memory tokens per segment, recurrent; lags on
  LongVideoBench); **XComp** 2604.14149 NeurIPS 2025 (LLM-internal-attention frame selection,
  segment-local); **QTSplus** 2511.11910 (query-aware token scoring on Qwen2.5-VL-3B: TempCompass
  order 63.9→69.5, MVBench action count 58.5→62.5, but Video-MME −0.7, MLVU 31.8→27.4 on an
  anomalously weak baseline — weak evidence); **Vista-LLM** ACL 2026 [UNVERIFIED, 404].
- **MIST** 2212.09522 CVPR 2023 (pre-LLM). Question-conditioned differentiable top-k segment selection
  (2 of 8) then regions: AGQA v2 54.39 vs all segments 51.40 — the one verified case where
  question-conditioned hierarchical SELECTION beat attending to everything, in a small trained model.
- Efficiency-framed (parity target, none claims > flat): STTM 2507.07990 ICCV 2025, CaCoVID
  2602.01649 AAAI 2026, MARC 2510.07915 ICLR 2026, KiToke 2604.03414, Clapper 2505.15529,
  LLaVA-Scissor 2506.21862, PruneVid 2412.16117 (MVBench 58.0→57.5 at 17 % tokens).

## 8. Negative results for selection / pruning on counting, ordering, absence

Frame-Voyager (counting no gain); GenS (Action Count −2.9, Holistic −2.4); AKS (gain shrinks 4× on
multi-moment questions); Q-Frame / BOLT admissions; LongVU Count 28.98; VideoNIAH (counting flat in
frames, decays in needles); TOMATO; ATP; TempCore (anti-key samples: relevance hurts absence /
state-change questions); LongVideoBench (referring reasoning needs more frames); DART 2502.11494
(importance-based pruning below RANDOM pruning on POPE; FastV 77.3 / SparseVLM 84.6 / DART 93.7 at
88.9 % reduction); Wen et al. 2502.11501 ACL 2025 Findings ("underperform random token selection");
Feather the Throttle 2412.13180 ICCV 2025 (early pruning drops top-of-image tokens, 5× worse on
localization); VTC-Bench 2510.07143 ACL 2026 (simple downsampling beats many compressors); AIM
2412.03248 (cross-frame merging strictly worse than within-frame: Video-MME 54.5 vs 56.6 at 12.5 %
— "merging across frames disrupts temporal order"); DyCoke 2411.15024 CVPR 2025 (LLaVA-OV MVBench
action sequence 70.7→67.6, action count 46.5→45.5 at 14 % tokens [per-task columns partly
unverified]); T* (2.1 % temporal F1); NExT-GQA (16 % grounded); E.T. Bench (multi-event ≈ 0).

## 9. Synthesis

1. The few-frame cross-frame failure is heavily precedented at 2 frames (TemporalVQA), 2 needle
   images with evidence only (Visual Haystacks), ~4 images (MuirBench ordering at chance), 4 needles
   (VNBench ordering), ≤ 8–16 frames (TOMATO, TemporalBench). Counting is the universal worst cell.
2. Cause, behavioural: outputs are nearly frame-order invariant (VBenchComp, Arrow of Time 50 % on
   Qwen2.5-VL-7B); older benchmarks hid it (30–60 % of items text / single-frame answerable).
3. Cause, location: most diagnostics put it in the LLM (T3 probes > 90 % vs decoder failing on
   captions; Before It Fades; Causality Matters; MIMIC); contested by Tracing the Arrow of Time
   (encoder / projector) and by T3's own ">70B LLMs do much better" (scale / data, not a wall).
4. Cause, mechanism: the only traced aggregation paths are attention-local — inter-frame attention
   only early (MIMIC, Causality Matters), evidence funnelled through the LAST frame then the query
   tokens, signal decaying with depth (Before It Fades). Nobody frames this as over-squashing or
   measures capacity vs number of evidence frames at fixed evidence. UNCLAIMED.
5. Non-SFT remedies move counting / ordering by a few points at best, in distribution; CoT / thinking
   can hurt; caption-then-reason only works on LLM-answerable benchmarks. Some training is unavoidable.
6. Train-short / test-long on the frame axis IS documented for NIAH (LongVILA, Video-XL,
   InternVideo2.5, LongVA) and as collapses on QA (Idefics2 / Mantis-Idefics2 at 64 frames; HoPE 8k →
   64k ≈ 58 → 26 for EVERY position encoding including the fixes). Aggregation collapses faster than
   retrieval (MM-NIAH 67.6→6.8 vs 99→72.7; VideoNIAH; Visual Haystacks; MLVU AO/AC). Fixes are
   validated on retrieval / MCQ averages only; V2PE and HoPE show retrieval and QA prefer OPPOSITE
   position settings. Variable frame count as a shift: Apollo ("video speed"), InternVL2.5 (training
   frame range). Models trained with variable frame counts (Qwen2.5-VL, LLaVA-Video, InternVideo2.5)
   still degrade at the longest lengths. UNCLAIMED: count / majority / order vs single-needle as test
   frames exceed a fixed training budget with evidence held fixed; whether the collapse is positional
   or capacity.
7. Selection: 2025–26 moved to trained SET-level gates (Frame-Voyager, GenS, VideoITG, ReFoCUS, TSPO,
   HFS); "set-level" = relevance + coverage − redundancy; only HiMu / Logic-in-Frames encode order
   predicates and both collapse to a per-frame curve; none handle negation or cross-frame identity.
   Relational gains are thin (GenS +9.6 order but −2.9 count; ViLA 0.0 STAR-Sequence; HiMu +0.9).
   Selection is neutral or harmful for counting everywhere it is reported; relevance selection hurts
   absence questions (TempCore). Untrained self-gating is weak in print (SeViLA +1.2 vs +2.2; T* 2.1 %
   F1; NExT-GQA 16 %). Grounders emit one span; multi-event QA ≈ 0.
8. Hierarchical question-conditioned query designs exist (InstructBLIP, LLaMA-VID, VideoStreaming,
   LongVU, HierarQ, VideoDetective, XComp, QTSplus) but NONE is compared against flat token-keeping at
   matched frames on a frozen LLM; ablations are against their own unconditioned variants. The only
   "compressed ≥ flat" numbers are MA-LMM (unmatched frames, classification) and MIST 2022 (small
   model). Efficiency compressors lose 1–3.5 on action count / sequence; cross-frame merging is
   strictly worse than within-frame (AIM). ⇒ Q-Former-style compression = efficiency, not accuracy,
   also at the video level.
9. Implication: the gate should be a trained SET-level module that sees the ordered sequence of
   per-block states (precedents: Frame-Voyager / GenS / VideoITG / ReFoCUS / TSPO for reader-reward
   training; HiMu / Logic-in-Frames for relations; HierarQ for the two-level query; VideoStreaming
   for Gumbel top-k over memories); for counting the bottleneck is the READ over kept frames (fence +
   reader), not the gate (VideoNIAH, GenS, Frame-Voyager, AKS); for absence / "did not happen" the
   gate must keep per-option evidence (TempCore, Learning to Deny).

## Flags / unverified
VideoRoPE per-length numbers; HoPE "64k = more frames" assumption; Mantis-Idefics2 32-frame value;
InternVL2.5 per-frame table; POINTS-Long frame claim (snippet); MMIU image-count degradation
(snippet); TimeScope arXiv identity; Visual Haystacks oracle multi-needle numbers (figure);
T3 per-aspect probe numbers (figure; captions GPT-4o-generated); Before It Fades / ViKey / VideoMind /
Learning to Deny remedy numbers (abstract-level); TimeChat and Koala full-text ablations not read;
Video-LLaMA has no numbers; DyCoke per-task columns possibly mislabeled; QTSplus baseline anomalous;
Frame-Voyager LLaVA-OV NExT-QA rows omitted (looked wrong); HiMu comparative numbers vs AKS / BOLT
not in fetched text; Vista-LLM (ACL 2026) page 404.

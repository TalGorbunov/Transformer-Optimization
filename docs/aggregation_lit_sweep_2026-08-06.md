# Aggregation-repair literature sweep — 2026-08-06

36 candidates found across 6 search angles; 35 survived adversarial verification
(existence + claim accuracy + direct-applicability >= 3/5). Grouped by the slot each
paper fills in the four-slot architecture (isolation / bounded merges / re-discretization
/ boundary emission) plus measurement & theory.

## analog-relay

### Communicating Activations Between Language Model Agents
*Ramesh & Li, 2025 (ICML)* — applicability 5/5 — https://arxiv.org/abs/2501.14082

**What:** Constructive (not analysis) use of cross-pass state transfer: pause model B at an intermediate layer, merge model A's intermediate activation into B's via a function f (sum/replace), continue B's forward pass — up to 27% better than text communication at <1/4 the compute, zero trained parameters.

**Use here:** Direct baseline against our digit-token re-quantization: graft pass-1 partial-count node activations into an intermediate layer of the pass-2 adder prompt via their merge functions f, skipping the quantize-to-vocabulary step entirely. If analog grafting at layer l>0 sums and emits correctly, our vocab re-quantization law needs weakening; if it decays as our 0.94->0.81->0.11 relay result predicts, it is strong published-mechanism support that vocabulary codes are the only lossless carrier between passes.

**Verifier note:** Verified verbatim: pause B at intermediate layer, merge A's activation via function f, resume; up to 27.0% over natural-language communication at <1/4 compute, zero trained parameters; ICML 2025 confirmed. Zero-training and inference-time only, so it is immediately runnable as the analog-grafting baseline against digit-token re-quantization; either outcome (works or decays per the 0.94->0.81->0.11 relay law) is a thesis-grade result. Their setting is cross-agent tasks, ours same-model two-pass — mechanism identical.

### In-context Autoencoder for Context Compression in a Large Language Model (ICAE)
*Ge, Hu, Wang, Chen & Wei, 2023 (ICLR 2024)* — applicability 4/5 — https://arxiv.org/abs/2307.06945

**What:** A LoRA-adapted copy of the LLM (encoder) compresses a long context into ~128 learnable 'memory slot' embeddings; the FROZEN original LLM (decoder) then conditions on those slots to answer prompts. Pretrained with autoencoding + LM objectives, ~1% extra parameters, 4x compression on Llama.

**Use here:** Structurally identical to our two-pass method with the frozen constraint already satisfied: pass 1 = LoRA'd Qwen encodes each fenced frame into m memory-slot vectors; pass 2 = frozen Qwen reads the concatenated slots and counts. Their key finding for us: memory slots produced by the same (lightly adapted) model ARE readable by the frozen decoder as soft embeddings - i.e., a learned alternative to our digit-token re-quantization. Run ICAE-style slots vs vocab-embedding quantized registers head-to-head to test whether re-quantization to vocab codes is actually necessary or just convenient.

**Verifier note:** Verified: LoRA-adapted encoder from the LLM compresses context into memory slots read by the FROZEN original LLM; 4x on Llama, ~1% extra params, AE+LM pretraining, ICLR 2024 camera-ready. The ~128-slot figure is in the paper body, not the abstract, but is correct. Frozen-decoder constraint matches ours by construction, making the slots-vs-vocab-registers head-to-head a sharp test of whether re-quantization is necessary. Docked to 4: original is text-context; adapting to fenced visual frames plus small-scale AE pretraining is real but bounded work.

### Training Large Language Models to Reason in a Continuous Latent Space (Coconut)
*Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, Yuandong Tian — 2024* — applicability 4/5 — https://arxiv.org/abs/2412.06769

**What:** Feeds the model's last hidden state back as the next input embedding instead of decoding to a text token ('continuous thought'), and trains the model to consume these recycled latent states; continuous thoughts encode multiple alternative next steps (BFS-like search) and beat text CoT on planning-heavy tasks with fewer tokens.

**Use here:** Run Coconut-style raw hidden-state feedback as a baseline at our tree nodes: instead of the ridge quantizer overwriting a node span with a digit-token embedding, write back the node's raw hidden state as an input embedding and measure exact-match. This isolates how much of our 0.980 comes from re-quantizing into vocabulary space vs merely recycling latent state — Coconut's finding that training is required for the model to consume continuous thoughts is the published counterpart to our analog-decay result (R2 0.94->0.81->0.11), and predicts the frozen baseline fails.

**Verifier note:** Verified on arXiv: authors, last-hidden-state-as-next-input-embedding, BFS-like encoding of alternative next steps, beats CoT on planning-heavy logical tasks with fewer thinking tokens all confirmed (accepted COLM 2025, not stated in claim). Applicability 4 not 5: the proposed use is a frozen no-quantizer ablation (write raw node hidden state back), which is trivially runnable and a strong published counterpart to the analog-decay result — but Coconut proper requires multi-stage curriculum training of the backbone, incompatible with the frozen constraint, and the frozen ablation largely duplicates the analog-relay probes already run.

## compression

### Learning to Compress Prompts with Gist Tokens
*Mu, Li & Goodman, 2023 (NeurIPS 2023)* — applicability 5/5 — https://arxiv.org/abs/2304.08467

**What:** Trains an LM to compress a prompt into a few 'gist' tokens purely by modifying the attention mask during instruction finetuning: tokens after the gists are masked from the raw prompt and can only attend to the gist tokens, forcing all task-relevant information through them. Achieves up to 26x compression with minimal quality loss on LLaMA-7B.

**Use here:** This is the training recipe for our learned carriers: keep our block-diagonal fence, append k carrier tokens per frame, and mask the question/readout tokens so they see ONLY carriers, not frame tokens - then train just carrier embeddings + LoRA (our existing repair #2) with the standard LM loss. Gisting shows the mask alone is a sufficient training signal, so we can drop the distillation objective and compare. Also a direct baseline: per-frame gist tokens vs our ridge-quantizer registers at equal token budget.

**Verifier note:** Verified: attention-mask-only training signal during instruction finetuning, up to 26x compression on LLaMA-7B, NeurIPS 2023. The mask-cut recipe maps directly onto the existing fence + carrier + LoRA infra (repair #2) as a loss/mask config change — runnable this month. One honesty flag: gisting FULL-finetunes the LM, so its evidence that the mask alone suffices may weaken under frozen-4bit + LoRA; also gists compress reusable prompts, not per-instance content frames — the transfer is plausible but not guaranteed.

### SoftCoT: Soft Chain-of-Thought for Efficient Reasoning with LLMs
*Yige Xu, Xu Guo, Zhiwei Zeng, Chunyan Miao — 2025* — applicability 5/5 — https://arxiv.org/abs/2502.12134

**What:** Keeps the backbone LLM completely frozen: a small fixed assistant model generates instance-specific soft thought tokens, and a small trainable projection module maps them into the frozen LLM's input-embedding space; only the projection is trained (parameter-efficient), improving reasoning on five benchmarks without modifying the LLM.

**Use here:** Directly matches our constraint set (frozen backbone, tiny trained linear modules writing embeddings the model reads). Swap our vocab-snapped digit embeddings for a SoftCoT-style trained linear projection producing continuous embeddings at the node spans: tests whether re-quantization to vocabulary codes is strictly necessary for lossless attention transport, or whether a learned continuous code the frozen model accepts can carry richer partial-count information (e.g., for N>=16 where single digits saturate). Also a citable precedent that trained-projection-into-frozen-LLM is a recognized method family.

**Verifier note:** Verified: frozen backbone LLM, lightweight fixed assistant generating instance-specific soft thought tokens, trainable projection into the LLM's representation space, only projection trained, five reasoning benchmarks — all match the abstract exactly. Highest applicability: same constraint set as ours (frozen backbone + tiny trained linear modules writing input embeddings). Swapping vocab-snapped digit embeddings for a trained continuous projection at node spans is implementable this month on existing ridge/engine infrastructure, and directly tests whether vocab re-quantization is necessary for lossless transport at N>=16.

### Future Lens: Anticipating Subsequent Tokens from a Single Hidden State
*Pal, Sun, Yuan, Wallace & Bau, 2023 (CoNLL)* — applicability 4/5 — https://arxiv.org/abs/2311.04897

**What:** Measures how much future-token content one hidden state carries: linear probes plus a 'fixed prompt causal intervention' where a single hidden state is transplanted into a learned generic soft prompt in a fresh forward pass, decoding tokens at t+2, t+3... with up to ~48% accuracy on GPT-J.

**Use here:** Their transplant-into-learned-soft-prompt technique is a drop-in upgrade of our readout for multi-digit counts (N>=16): instead of one ridge head per digit position, transplant the node state into a trained soft prompt and decode the full count string autoregressively. Also directly quantifies single-state capacity — how many digits/partial counts one carrier token can hold before the tree must split — complementing our c(fan) law.

**Verifier note:** Verified: CoNLL 2023, Pal, Sun, Yuan, Wallace, Bau; linear probes + causal intervention with >48% subsequent-token accuracy from a single GPT-J-6B hidden state confirmed. Applicability 4: the transplant-into-trained-soft-prompt readout is a plausible multi-digit (N>=16) upgrade over per-digit ridge heads and quantifies single-carrier capacity, complementing c(fan); needs a small soft-prompt training loop (comparable cost to the ridge heads), hence not quite 5.

### VoCo-LLaMA: Towards Vision Compression with Large Language Models
*Ye, Gan, Huang, Ge, Shan & Tang, 2024* — applicability 4/5 — https://arxiv.org/abs/2406.12275

**What:** Makes the LLM itself compress its vision tokens: inserts VoCo tokens after the image tokens and modifies the attention mask so text tokens can attend only to VoCo tokens, never the raw vision tokens; attention distillation transfers how the LLM read the full image into how it reads the VoCo tokens. 576x compression with minimal loss; extended to video by concatenating per-frame compressed tokens.

**Use here:** Our fence + carrier design already IS this mask topology, so VoCo-LLaMA is both must-cite prior art and a drop-in training scheme: add per-frame VoCo/carrier tokens inside each fence block, apply their causal-mask cut so the question section sees only carriers, and train with their attention-distillation loss (teacher = unfenced attention over the frame) via LoRA on frozen 4-bit Qwen. Directly tests whether distilling the model's own read pattern beats our externally-trained ridge quantizers for filling the registers.

**Verifier note:** Verified: VoCo tokens after image tokens, text-to-vision attention weights set to False so text sees only VoCo tokens, 576x compression, video via time-series compressed tokens. Must-cite prior art for the fence+carrier mask topology. One nuance: 'attention distillation' is the paper's own framing but is implicit — realized through the mask cut plus an output-distribution KL to the uncompressed model, NOT an explicit attention-map teacher loss; the plan's 'their attention-distillation loss (teacher = unfenced attention)' would need to be the KL variant. Rated 4: mask topology already implemented here, remaining work is the training objective under LoRA.

## emission

### Patchscopes: A Unifying Framework for Inspecting Hidden Representations of Language Models
*Ghandeharioun, Caciularu, Mueller, Geva & Goldberg, 2024 (ICML)* — applicability 5/5 — https://arxiv.org/abs/2401.06102

**What:** Formalizes exactly our two-pass move: take a hidden state from a source forward pass, optionally map it with f, patch it into position i*/layer l* of a separate target prompt, and let the frozen model itself verbalize the encoded information. Systematically maps which (source layer -> target layer) patches decode successfully, and shows early-layer target patches are the most expressive.

**Use here:** Zero-parameter baseline for our ridge quantizer: patch each tree-node state into a target prompt like 'X ... the count is' at an early target layer of a second Qwen pass and let the model speak the digit itself. Their source->target layer grid is also a ready-made instrument for the L20 emission deadline: it predicts patches landing at target layer ~0 acquire token identity and become emittable, patches landing late do not — we can replicate that grid on carrier states.

**Verifier note:** Verified: ICML 2024, Ghandeharioun, Caciularu, Pearce, Dixon, Geva (claimed author list slightly garbled — 'Mueller' is not an author and 'Geva & Goldberg' conflates it with a different paper; Dixon and Pearce are the missing names — but the paper and description are otherwise accurate). Applicability 5: zero-parameter baseline for the ridge quantizer (patch tree-node state into an early layer of a second Qwen pass and let it speak the digit) and the source-to-target layer grid is a ready-made emission-deadline instrument; the codebase pattern matches the existing two-pass engine.

### Verbalizable Representations Form a Global Workspace in Language Models
*Gurnee, Sofroniew, Pearce, et al. (Anthropic) — 2026* — applicability 4/5 — https://arxiv.org/abs/2607.15495

**What:** Introduces the Jacobian lens — the average linearized effect of an intermediate activation on output-token likelihood — to identify which residual-stream representations the model can actually verbalize (report, reason over, steer with). Finds verbalizable 'workspace' content lives in middle layers (~33-79% depth) with final layers switching to immediate output prediction; all experiments are probes/patching/steering on frozen models, including causal swaps of J-lens vectors that change what the model says.

**Use here:** A quantitative instrument for our emission deadline: compute the Jacobian-lens emittability of a value as a function of the layer it was written at, turning 'after ~L20 it cannot be spoken' into a measured curve on frozen Qwen; also a principled check of the quantizer design — verify that overwriting a span with a digit-token embedding lands the state inside the verbalizable subspace (and use the J-lens direction, not raw embedding, as the write target if not).

**Verifier note:** Verified (post-cutoff, confirmed by fetch of the html version): Anthropic, July 2026; author list starts Gurnee, Sofroniew, Pearce and ends Batson, Lindsey, matching 'et al. (Anthropic)'. Jacobian lens, verbalizable workspace at intermediate depths, and frozen-model patching/steering along J-lens directions all confirmed; the exact 33-79% depth figure was not verifiable from the abstract but is consistent with 'specific intermediate depths'. Applicability 4: J-lens emittability-vs-write-layer curve is a direct instrument for the L20 emission deadline, and checking the digit-embedding overwrite against the verbalizable subspace is a concrete quantizer sanity test; requires backprop through the 4-bit model but no training.

### Long Context Compression with Activation Beacon
*Zhang, Liu, Xiao, Shao, Ye & Dou, 2024* — applicability 3/5 — https://arxiv.org/abs/2401.03462

**What:** Interleaves special 'beacon' tokens that progressively condense the KV activations (keys/values at every layer) of each preceding chunk into compact per-layer KV entries; the base LLM is kept frozen and only the beacon-side parameters are trained. Extends Llama-2-7B from 4K to 400K context with ~9 GPU-hours of training on short sequences.

**Use here:** Compression at the KV level rather than the residual/token level, which sidesteps two of our measured failure modes at once: the per-hop analog decay (beacon KV is written per-layer exactly where attention reads it, no multi-hop relay) and the ~L20 emission deadline (information never has to re-enter the stream as a late token). Concrete use: replace our mid-forward span-overwrite with per-frame beacon tokens whose condensed KV the second pass attends to; train only beacon projections on frozen 4-bit Qwen. Also the natural 'KV-compression' baseline column for the thesis.

**Verifier note:** Exists, but the description stitches the v2+ title onto v1 details: frozen-LLM plug-and-play beacons, 4K->400K on Llama-2-7B, and short-sequence training are all verbatim in v1 ('Soaring from 4K to 400K'); the current revision reframes training/eval (20K-length training, 2x inference speedup, 8x KV reduction). Also '9 GPU-hours' misreads v1's 'less than 9 hours on a single 8xA800 machine' (~72 GPU-hours). Mechanism claims accurate; cite the right version. Rated 3: per-layer KV beacon modules on 4-bit multimodal Qwen with M-RoPE is heavy engineering, not a this-month drop-in — but conceptually it does sidestep both the relay decay and the L20 emission deadline.

## measurement

### Eliciting Latent Predictions from Transformers with the Tuned Lens
*Belrose, Ostrovsky, McKinney, Furman, Smith, Halawi, Biderman & Steinhardt, 2023* — applicability 5/5 — https://arxiv.org/abs/2303.08112

**What:** Trains one small affine translator per layer of a frozen model so every hidden state decodes to a calibrated distribution over the vocabulary; far more reliable than logit lens, especially at early/middle layers; code at github.com/AlignmentResearch/tuned-lens.

**Use here:** This is the 'lens quantizer' already flagged as open in our superquery STATE.md: replace per-node ridge-round heads with per-layer affine translators trained once on frozen Qwen, then quantize a node state by argmax over digit tokens in vocab space — one shared head for all nodes/layers instead of bespoke classifiers, and the layer at which the count first becomes vocab-decodable gives an independent measurement of the emission deadline.

**Verifier note:** Fully verified: per-layer affine translators on frozen models, explicitly claimed more predictive/reliable/unbiased than logit lens; code repo real. Matches the 'lens quantizer' open item in superquery STATE.md exactly. Caveat: no off-the-shelf lens exists for Qwen2.5-VL, so translators must be trained locally — but affine probes are cheap (CPU/1-GPU) and the layer-of-first-vocab-decodability doubles as an independent emission-deadline instrument. Runnable this month.

### Block-Attention for Efficient RAG / Efficient Prefilling
*Sun, Wang, et al. — 2024 (ICLR 2025)* — applicability 5/5 — https://arxiv.org/abs/2409.15355

**What:** Divides retrieved passages into independent blocks that compute their KV states separately (cacheable/reusable), with position re-encoding per block; only the final block (query) attends to everything. Quantifies that a frozen model degrades under this block mask, and that light fine-tuning restores parity with full self-attention on four RAG benchmarks.

**Use here:** Their frozen no-finetune condition is a published measurement of the same mechanism we see: independently encoded blocks + one global reader loses accuracy without repair. Run their layout (block KV + position re-encode + query-reads-all) as an MMRED baseline, and cite their frozen-vs-finetuned gap as evidence that the deficit is real and repairable — contrasting their repair (weight tuning) with ours (frozen backbone + quantizer relay tokens). Their per-block position re-encoding is also an independent validation of our M-RoPE per-block reset.

**Verifier note:** CITATION ERROR: authors are Dongyang Ma, Yan Wang, Tian Lan (Tencent) — NOT 'Sun, Wang, et al.' Fix before citing. Everything else verified: ICLR 2025; independent per-block KV + position re-encoding + final block attends globally; frozen model degrades under the mask and fine-tuning restores full-attention parity (early versions report 4 RAG benchmarks; the Apr-2025 revision expands to 11). This is the closest published measurement of our fencing mechanism: their layout is essentially our block fence + pos-reset, runnable as an MMRED baseline with code we already have; their tuned-weights repair contrasts cleanly with our frozen+relay repair.

### Set-Based Prompting: Order-Independence Without Fine Tuning
*McIlroy-Young, Brown, Olson, Zhang, Dwork — 2024* — applicability 5/5 — https://arxiv.org/abs/2406.06581

**What:** Makes a frozen LLM provably invariant to the ordering of designated parallel sub-sequences by combining a non-triangular attention mask with modified (parallel) positional encodings. Empirically decomposes the two components: the attention-mask change removes far more order variation than the positional-encoding change, with model-dependent effect sizes.

**Use here:** Two uses: (1) a correctness instrument — their construction gives exact order-invariance, so permuting MMRED frames under our fence must leave logits bit-identical; any deviation localizes a leak in our mask/pos-reset (cheap CPU-level test to add next to tests/test_fencing.py). (2) Their mask-vs-positions decomposition is a ready-made ablation template for attributing how much of our fencing gain comes from the block mask vs the M-RoPE reset — a number the thesis needs.

**Verifier note:** Verified including the fine-grained claim — paper quote: 'modifying the attention mask reduces variation much more than the positional encoding, although the effect varies significantly between models' (Figure 5 ablates PE-only, mask-only, and both). Current arXiv/NeurIPS-2024 title is 'Order-Independence Without Fine Tuning'; 'Set-Based Prompting' is the method name (and original title) — both resolve to 2406.06581. Directly usable this month: an exact order-invariance bit-parity test for our fence next to tests/test_fencing.py (CPU-cheap), plus a ready-made template for the mask-vs-M-RoPE-reset attribution ablation the thesis needs.

### Massive Activations in Large Language Models
*Mingjie Sun, Xinlei Chen, J. Zico Kolter, Zhuang Liu (CMU / Meta) — 2024 (arXiv 2402.17762)* — applicability 5/5 — https://arxiv.org/abs/2402.17762

**What:** Quantifies massive activations (up to ~1e5x median) at fixed feature dims on start/delimiter tokens; shows they act as input-independent bias terms that concentrate attention mass on sink tokens, i.e., an implicit bias in self-attention output. Interventions are runnable on frozen models: zeroing them collapses performance while setting them to their constant mean is harmless. Also shows explicit attention-bias (register-like learnable KV) removes the need for them, and finds the same phenomenon in ViTs where trained registers play this role. Code: github.com/locuslab/massive-activations.

**Use here:** Use their measurement protocol on frozen Qwen2.5-VL to test whether our fan-in~4 read ceiling coincides with attention mass captured by massive-activation tokens (quantify sink mass at the read layer per fan level); their mean-patching intervention runs directly in our hook infrastructure, and their explicit-bias-KV result motivates a tiny learned bias KV pair as an allowed small adapter.

**Verifier note:** Verified: Sun, Chen, Kolter, Liu (CMU/Meta), arXiv 2402.17762, code github.com/locuslab/massive-activations. Abstract confirms ~100,000x-median activations that are input-invariant, function as bias terms, and concentrate attention on their tokens ('implicit bias terms in the self-attention output'), with the phenomenon also in ViTs. The finer claims (fixed feature dims on start/delimiter tokens; zeroing collapses vs mean-setting harmless; explicit learnable attention-bias KV removes the need; trained ViT registers play this role) are the paper's Sections 3-4 core results — accurately described. Rated 5: the measurement protocol and mean-patching intervention run directly in our existing hook infrastructure on frozen Qwen2.5-VL this month, and quantifying sink-captured mass at the read layer per fan level is exactly the missing link between our fan-in~4 law and the sink literature.

### Transformers need glasses! Information over-squashing in language tasks
*Barbero, Banino, Kapturowski, Kumaran, Araujo, Vitvitskyi, Pascanu, Velickovic — 2024 (NeurIPS)* — applicability 4/5 — https://arxiv.org/abs/2406.04267

**What:** The canonical over-squashing-in-LLMs paper: proves and empirically shows representational collapse in decoder-only transformers — distinct input sequences (e.g. counting/copying prompts of growing length) yield arbitrarily close last-token representations, made worse by low-precision (bf16/fp8) formats, so the frozen model provably cannot answer them differently. Demonstrates the resulting counting and copying failures in real LLMs (Gemini) and proposes simple input-level fixes (inserting extra separator tokens to keep representations distinct).

**Use here:** This is the formal frame for our fan-in ~4 / accuracy-halving law: run their collapse metric (pairwise last-token hidden-state distance vs N under bf16) on our MMRED frame sequences to show our aggregation ceiling IS their representational collapse; their separator-token-insertion fix is a cheap prompt-level baseline to run against the fence+carrier method; must-cite ancestor for the thesis's over-squashing framing.

**Verifier note:** Verified on arXiv: representational collapse theorem, over-squashing link to GNNs, counting/copying failures, low-precision exacerbation, and simple mitigations all in the abstract; NeurIPS 2024, DeepMind authors (Gemini experiments consistent). Applicability 4: the pairwise last-token collapse metric and separator-token baseline are runnable this month on MMRED sequences, but the theory targets causal text sequences, so mapping to the fenced multi-frame VLM setup takes some adaptation; primarily the framing/measurement ancestor rather than a competing method.

### SelfIE: Self-Interpretation of Large Language Model Embeddings
*Chen, Vondrick & Mao, 2024 (ICML)* — applicability 4/5 — https://arxiv.org/abs/2403.10949

**What:** Training-free variant of the same mechanism: inserts a hidden embedding into the forward pass of an interpretation prompt ('repeat/summarize this message: [X]') so the LLM describes its own embedding in open-ended natural language; no probes or adapters at all.

**Use here:** The no-training control arm for our quantizer step: insert a carrier/tree-node state into an interpretation prompt on the same frozen Qwen and ask 'how many frames? Answer:'. If SelfIE-style readout recovers the tally, the ridge heads are only a convenience; if it fails while ridge succeeds, that quantifies how far the aggregated count code is from anything the model can self-verbalize — a clean thesis ablation.

**Verifier note:** Verified: Chen, Vondrick, Mao 2024 (ICML). The core interpretation mechanism (inject hidden embedding into an interpretation prompt, model describes it in open-ended language) is as claimed and needs no probes/adapters; the paper additionally offers optional gradient-based Supervised/Reinforcement Control extensions for editing, which do not contradict the training-free-readout claim. Applicability 4: clean no-training control arm for the quantizer — one afternoon of engineering on the existing patching machinery; an ablation rather than a candidate method.

### Let's Think Dot by Dot: Hidden Computation in Transformer Language Models
*Jacob Pfau, William Merrill, Samuel R. Bowman — 2024* — applicability 4/5 — https://arxiv.org/abs/2404.15758

**What:** Shows transformers can use meaningless filler tokens ('......') in place of chain-of-thought to solve parallelizable algorithmic tasks (2SUM/3SUM variants) they fail without intermediate tokens; quantitatively characterizes which problem class filler tokens help (first-order quantifier depth) and finds that learning to exploit fillers requires specific dense per-token supervision.

**Use here:** Quantitative measurement of the same mechanism our carrier tokens exploit: computation hidden above content-free appended tokens on aggregation-style (parallel sum) tasks. Their probing recipe (train probes on hidden states above fillers to recover intermediate values) is directly runnable on our carrier/replica spans, and their dense-supervision requirement is the published justification for our per-node ridge supervision rather than end-to-end answer-only training.

**Verifier note:** Verified: meaningless filler tokens ('......') substitute for CoT on 2SUM/3SUM-style tasks, theoretical characterization via first-order quantifier depth, and 'learning to use filler tokens is difficult and requires specific, dense supervision' are all in the abstract. Caveat: the 'probing recipe on hidden states above fillers' cited in direct_application is not confirmable from the abstract alone (the paper's probing analysis is in the body); the substantive what_it_does claims all check out. Applicability solid as citable justification for per-node dense (ridge) supervision, but their setting is from-scratch-trained models on synthetic tasks — paper #6 (Brauer et al.) covers the frozen-model transfer more directly.

### Reading Between the Dots: Decoding Hidden Computation across Filler Tokens
*Kaley Brauer, Claudio Mayrink Verdun, Samuel Marks — 2026* — applicability 4/5 — https://arxiv.org/abs/2607.03502

**What:** On frozen open-weights models (DeepSeek V3, Kimi K2), shows multi-step reasoning carried over content-free filler tokens is structured and readable: an unsupervised decoding pipeline recovers intermediate reasoning values from hidden states at 80-95% accuracy across four task families, attention routes queries through filler regions, and KV-cache manipulations causally change outputs.

**Use here:** Two directly transferable instruments: (1) their unsupervised hidden-state decoding pipeline could recover partial counts from our tree-node states without ridge labels — a label-free validation (or replacement) of our quantizer heads and a cross-check of the probe-family artifact (ridge vs LR) issue; (2) their KV-cache intervention protocol is the same intervention family as our mid-forward span overwrites, giving a published causal-editing methodology to cite and replicate for the 'values emerge early, crystallize late' timing that parallels our layer-20 deadline.

**Verifier note:** Verified despite being the most hallucination-suspect entry (July 2026, post-recent): arXiv page confirms title, authors (Brauer, Mayrink Verdun, Marks), frozen DeepSeek V3 / Kimi K2, unsupervised decoding recovering intermediate values at 80-95% across four task families (fact retrieval, numeric composition, string manipulation, in-context computation), attention analysis, logit-lens, and causal KV-cache manipulation. Applicability 4: their unsupervised decoding pipeline and KV-cache intervention protocol are runnable on frozen Qwen carrier states this month as measurement instruments (label-free cross-check of the ridge quantizers and the ridge-vs-LR probe-family artifact), though adapting from text-MoE LLMs to VLM carrier spans is real engineering and it is an instrument, not a counting method/baseline.

## read-capacity

### softmax is not enough (for sharp out-of-distribution)
*Velickovic, Perivolaropoulos, Barbero, Pascanu — 2024 (ICML 2025)* — applicability 5/5 — https://arxiv.org/abs/2410.01104

**What:** Proves that every softmax attention head must disperse as the number of attended tokens grows — attention coefficients that are sharp in-distribution provably flatten with more tokens, destroying sharp single-read aggregation at length. Proposes an adaptive temperature mechanism: a training-free, inference-time sharpening of attention logits (temperature chosen per-head from the observed entropy) that partially restores sharpness on frozen models.

**Use here:** The mechanism behind our 'one softmax read caps at fan-in ~4, then halves per doubling' law, stated as a theorem. Concretely runnable: patch their adaptive-temperature rescaling into our sdpa attention at the tree-read layers of frozen Qwen and measure whether the fan-in capacity c(fan) shifts — a one-file inference-time intervention in gnnformer/engine, and the natural ablation baseline against the binary-tree topology fix.

**Verifier note:** Verified; note the paper was retitled to 'Softmax is not Enough (for Sharp Size Generalisation)' in revision — the claimed subtitle is the original v1 title, so the claim stands. ICML 2025, dispersion theorem and training-free inference-time adaptive temperature both confirmed. Applicability 5: adaptive temperature is a one-file sdpa-level patch on frozen Qwen at the tree-read layers; directly tests whether c(fan) shifts and is the natural inference-time ablation against the binary-tree topology fix.

### APE: Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding
*Yang et al. — 2025 (ICLR 2025)* — applicability 5/5 — https://arxiv.org/abs/2502.05431

**What:** Identifies why a single query reading many independently-encoded context blocks (parallel encoding / reused positions) underperforms sequential encoding: the attention distribution over the parallel blocks is misaligned. Fixes it at inference time with three adjustments — shared prefix across blocks, attention temperature, and a scaling factor on the read — recovering 98% (RAG) / 93% (ICL) of sequential-encoding accuracy with a frozen model.

**Use here:** The most direct 'fix for the one-reader bottleneck' on our list: apply their attention-temperature + scaling-factor adjustment at the layers where our question/tree-root tokens read the fenced frame carriers. If sharpening/re-scaling the softmax over parallel blocks raises effective fan-in past ~4, we could widen the tree branching factor (fewer relay hops, less analog decay) at N=16..128 with no training. Their misalignment analysis is also the published quantification of exactly our one-softmax-read failure.

**Verifier note:** Verified (ICLR 2025 confirmed in arXiv metadata; authors Xinyu Yang, Tianqi Chen, Beidi Chen): diagnoses attention-distribution misalignment when a query reads independently-encoded parallel contexts, fixes at inference with shared prefix + attention temperature + scaling factor, recovering exactly the claimed 98% (RAG) / 93% (ICL) of sequential-encoding performance on a frozen model. Rated 5: the temperature/scaling adjustment at the layers where tree-root/question tokens read fenced frame carriers is a no-training intervention implementable this month, and it directly attacks the fan-in ~4 read-capacity wall (would allow wider tree branching, fewer analog relay hops at N=16..128). Their misalignment analysis is the closest published quantification of the one-softmax-read failure.

### LatentQA: Teaching LLMs to Decode Activations Into Natural Language
*Pan, Chen & Steinhardt, 2024* — applicability 4/5 — https://arxiv.org/abs/2412.08686

**What:** Latent Interpretation Tuning: finetunes a decoder LLM on (activation, question, answer) triples — visual-instruction-tuning style, with activations in place of images — so the decoder answers open-ended questions about a target model's activations in natural language; code at github.com/aypan17/latentqa.

**Use here:** Upper-bound instrument for our read-capacity results: LoRA-tune a Qwen decoder on (carrier/tree-node activation, 'how many frames was C in R?', answer) triples. If a trained decoder recovers exact counts from states where ridge-round fails, the information survives aggregation and only our linear readout is the bottleneck; if it also fails, the fan-in capacity wall is informational, not readout-limited — a distinction our thesis currently cannot make.

**Verifier note:** Verified: Pan, Chen, Steinhardt 2024; decoder LLM finetuned on (activation, QA) triples in visual-instruction-tuning style; repo aypan17/latentqa exists with training code and Llama-3-8B-Instruct decoder weights. As an upper-bound readout instrument it cleanly separates informational vs readout-limited failure — a distinction the thesis currently lacks. Docked to 4: pretrained decoder is Llama-3, so a Qwen decoder must be LoRA-trained on self-generated triples plus activation-injection plumbing; feasible with existing SFT infra but not drop-in.

### LLaMA-VID: An Image is Worth 2 Tokens in Large Language Models
*Li, Wang & Jia, 2023 (ECCV 2024)* — applicability 4/5 — https://arxiv.org/abs/2311.17043

**What:** Represents every video frame with exactly two tokens: a query-conditioned 'context token' (attention pooling of frame features against the user question) and a 'content token', letting a VLM ingest hour-long videos. Outperforms prior video QA methods despite the extreme per-frame budget.

**Use here:** The closest existing instantiation of 'compress each frame into small task-relevant registers': their context token is literally a question-conditioned per-frame register computed before the LLM sees the frame. Baseline for us: question-aware attention-pool each fenced frame's features into 1-2 tokens (tiny trained pooling head, backbone frozen), feed only those to the counting pass, and measure whether the fan-in ~4 read ceiling still bites when each frame is 2 tokens instead of hundreds. Also tells us how much of our gain is fencing/tree vs plain per-frame compression.

**Verifier note:** Verified: two tokens per frame (user-input-conditioned context token via attention over frame features + content token), hour-long video support, surpasses prior methods; ECCV 2024 confirmed via Springer DOI 10.1007/978-3-031-72952-2_19. A minimal question-conditioned pooling head on frozen Qwen-VL features is implementable this month and directly tests whether the fan-in~4 ceiling persists at 2 tokens/frame, isolating fencing/tree gains from plain compression. Docked to 4 because their full 3-stage training pipeline is not portable — only the diagnostic reduction is.

### Think before you speak: Training Language Models With Pause Tokens
*Sachin Goyal, Ziwei Ji, Ankit Singh Rawat, Aditya Krishna Menon, Sanjiv Kumar, Vaishnavh Nagarajan — 2023 (ICLR 2024)* — applicability 4/5 — https://arxiv.org/abs/2310.02226

**What:** Appends learnable pause tokens to the input and delays output extraction until the last pause, giving the model extra hidden vectors to manipulate before committing to an answer; key quantitative finding is that gains require pause-pretraining — inference-time-only pauses on a standard-pretrained model give little benefit.

**Use here:** The cheapest baseline/control for our carrier tree: append K blank pause/register tokens before readout (no quantizer writes) and measure whether extra parallel compute alone lifts the fan-in ~4 ceiling. Their published negative result for inference-only pauses on frozen models is exactly the control our story needs — it supports the claim that content writes (digit-embedding overwrites), not extra token compute, produce the gain. Trivially implementable in our fence/engine code.

**Verifier note:** Verified (ICLR 2024): learnable pause tokens appended, output extraction delayed to last pause, and the key finding that gains need pauses in BOTH pretraining and finetuning — inference-only or finetune-only gives little/no benefit — is confirmed in the abstract. The inference-only-pause control on a frozen model is a one-afternoon change to the fence/engine and is exactly the published negative result supporting 'content writes, not extra token compute'. Rated 4 not 5 because the paper's positive method (pause-pretraining) is fundamentally inapplicable frozen, and the runnable part is a control expected to reproduce a null.

### Superposition Prompting: Improving and Accelerating Retrieval-Augmented Generation
*Merth, Fu, Rastegari, Najibi (Apple) — 2024 (ICML 2024)* — applicability 4/5 — https://arxiv.org/abs/2404.06910

**What:** Feeds documents to a frozen pre-trained LLM along parallel prompt paths in a DAG (no fine-tuning), assigns positions per path, and uses the model's own logits to prune paths deemed irrelevant before/while generating — 93x prefill speedup and +43% accuracy on NaturalQuestions-Open with MPT-7B, largely by removing the distraction of irrelevant parallel context.

**Use here:** Path pruning is a fan-in reducer for our reader: use cheap per-frame salience (their logit-based path scoring, or our ridge heads) to discard frames where character C is absent BEFORE the tree read, so the surviving fan-in stays under the ~4 aggregation capacity. Their path-position assignment scheme is also a runnable alternative to compare against our per-block position reset. All inference-time, frozen model.

**Verifier note:** Fully verified: Apple authors, ICML 2024 (PMLR v235), frozen LLM, parallel DAG prompt paths with per-path positions, logit-based path pruning, 93x prefill / +43% NQ-Open on MPT-7B. Code: github.com/apple/ml-superposition-prompting. Applicable as a fan-in reducer: prune frames where character C is absent before the tree read (absent frames contribute 0 to the count, so pruning is semantically sound). Not a 5 because their code targets text-only LLMs; porting path-position bookkeeping to Qwen2.5-VL M-RoPE and our fence is real adaptation work.

### Unveiling and Harnessing Hidden Attention Sinks: Enhancing LLMs without Training through Attention Calibration (ACT)
*Zhongzhi Yu et al. (Georgia Tech EIC) — ICML 2024* — applicability 4/5 — https://proceedings.mlr.press/v235/yu24l.html

**What:** Shows attention sinks occur not only at the first token but throughout the sequence, and not all are beneficial. ACT is a training-free, input-adaptive calibration applied during inference on a frozen LLM: offline it identifies which heads' sinks must be preserved, then at inference it rescales attention in the remaining heads, redistributing wasted sink mass to informative tokens. Up to +7.3% accuracy on Llama-30B. Code: github.com/GATECH-EIC/ACT.

**Use here:** Run as an inference-time baseline on the frozen Qwen2.5-VL: identify harmful sink heads on MMRED probes offline, then rescale attention at the final-query read layers to reallocate sink mass across frame/carrier tokens — a direct test of whether freeing sink mass raises the fan-in~4 softmax aggregation ceiling we measured.

**Verifier note:** Verified at PMLR v235 (ICML 2024): sinks occur throughout the sequence not just token 0; training-free input-adaptive calibration during inference; 'average improvement of up to 7.30% in accuracy' on Llama-30B — matches claim. Senior author Yingyan Celine Lin runs the Georgia Tech EIC lab (GATECH-EIC GitHub org), consistent with the claimed affiliation and code location. Runnable with our hook infrastructure as a frozen-model baseline and a clean test of the sink-mass explanation for the fan-in~4 ceiling. Not a 5: offline sink-head identification protocol was designed for text classification/QA and needs porting to our multimodal MMRED probes.

### When Can Transformers Count to n?
*Yehudai, Kaplan, Dar, Rassin, Ghandeharioun, Geva, Globerson — 2024* — applicability 3/5 — https://arxiv.org/abs/2407.15160

**What:** Quantifies the counting capacity of transformers as a sharp phase transition: exact token counting (query-count / most-frequent-element) works when embedding dimension d >= effective vocabulary size, because near-orthogonal one-hot-like codes can be summed and read off; once vocabulary exceeds d, interference between non-orthogonal codes forces weights to blow up polynomially and exact counting becomes unstable and unlearnable. Verified empirically on trained models and pretrained LLMs.

**Use here:** Directly predicts our two transport regimes: vocabulary-embedding codes move losslessly (near-orthogonal, sum-decodable) while arbitrary hidden-state codes interfere — i.e., a first-principles explanation for why our ridge-quantizer re-encoding into digit-token embeddings works. Their d-vs-vocab capacity law is the theoretical counterpart to compare against our measured c(fan) capacity curve, and their counting probes are runnable on frozen Qwen carrier states.

**Verifier note:** Verified: sharp phase transition at d >= vocabulary size, interference forcing polynomial weight blow-up, from-scratch training drop at the theoretical threshold, and pretrained-LLM failures all confirmed (Yehudai, Kaplan, Dar, Rassin, Ghandeharioun, Geva, Globerson 2024). Applicability 3: strong first-principles explanation for the lossless-vocab-code vs analog-hidden-code split and a theoretical counterpart to c(fan), but it is a theory paper about token counting in text; its d-vs-vocab law is a different quantity from softmax fan-in capacity, so it informs the thesis narrative more than it yields a runnable method or baseline.

## topology

### Parallel Context Windows for Large Language Models (PCW)
*Ratner, Levine, Belinkov, Ram, Magar, Abend, Karpas, Shashua, Leyton-Brown, Shoham — 2023 (ACL)* — applicability 5/5 — https://arxiv.org/abs/2212.10947

**What:** Training-free method for off-the-shelf frozen LLMs: carves long context into windows, restricts attention to be block-diagonal within each window, re-uses positional embeddings across windows, and lets the final task/query tokens attend to all windows in one read. Shows in-context-learning gains but degradation as the number of parallel windows grows (750M-178B models).

**Use here:** This is the direct ancestor of our fence layout (block mask + position reuse = our per-frame fence + M-RoPE reset) and the canonical no-tree baseline: encode frames as PCW windows and let the question read all of them in a single softmax read. Their reported degradation with window count is an external replication target for our fan-in ~4 capacity law c(fan); running PCW-as-is on MMRED gives the 'one flat reader' baseline our tree beats.

**Verifier note:** Verified (ACL 2023): training-free on off-the-shelf LLMs, carve context into windows, attention restricted within windows, positional embeddings re-used across windows, task tokens attend to all; 750M-178B models. Window-count degradation claim confirmed from full text: 'diminishing returns around B in the range of 5 to 7', SST-2 harmed by more windows, many-class tasks converge ~B=6 — notably consonant with the fan-in ~4 capacity law. Rated 5: it is the direct ancestor of the fence layout and the canonical training-free 'one flat reader' baseline, runnable on MMRED this month with the existing fencing code.

### Mechanistic Interpretability of Large-Scale Counting in LLMs through a System-2 Strategy
*Hasani, Banayeeanzade, Nafisi, Mohammadian, Askari, Bagherian, Izadi, Baghshah — 2026 (ACL)* — applicability 4/5 — https://arxiv.org/abs/2601.02989

**What:** Mechanistically traces how LLMs count: shows counting is spread across layers so precision degrades with count size (depth constraint); via causal mediation analysis finds latent partial counts are stored in the final-item representation of each chunk, transferred by dedicated attention heads, and aggregated in a final stage. Proposes a test-time (prompting-only, no training) decomposition strategy that splits large counts into sub-counts and aggregates them, recovering accuracy on frozen models.

**Use here:** Independent discovery of our exact architecture in text space: their chunk-wise partial-count-then-aggregate prompting is the text-only baseline our quantized-tree + second-pass-addition method must beat (same frozen-model constraint, zero trained heads); and their causal-mediation recipe for locating the dedicated transfer/aggregation heads is directly reusable on Qwen to pin down our copy circuit and the ~L20 emission deadline head-by-head.

**Verifier note:** Verified (post-cutoff, confirmed by fetch): ACL 2026, submitted Jan 2026; abstract matches the claim nearly verbatim — latent counts stored in final-item representations per chunk, dedicated transfer heads, final-stage aggregation, causal mediation analysis, and a prompting-only System-2 decomposition. Applicability 4: their chunked-count prompting is the must-run text-only frozen baseline against the quantized tree (immediate), and the mediation recipe is reusable to pin the copy circuit / L20 deadline head-by-head (real but tractable work). Closest independent-discovery threat in the list — cite and differentiate carefully.

### Found in the Middle: Calibrating Positional Attention Bias Improves Long Context Utilization
*Cheng-Yu Hsieh et al. (UW / MIT / Google Cloud AI) — Findings of ACL 2024 (arXiv 2406.16008)* — applicability 4/5 — https://arxiv.org/abs/2406.16008

**What:** Quantifies the U-shaped positional attention bias behind lost-in-the-middle: the final query's attention to a passage = relevance term + position term, estimated by permuting the same content across positions. Their 'found-in-the-middle' calibration subtracts the position component at inference so a frozen model attends by relevance regardless of location; up to +15 points on multi-doc QA / RAG.

**Use here:** Apply their disentangling protocol to our final answer query over frame blocks / carrier nodes: measure the position-bias component under frame permutation (we already have permutation harnesses), then calibrate the final-read attention to flatten serial-position bias — a direct inference-time fix for the deadline query under-reading middle tree nodes.

**Verifier note:** Verified: Hsieh et al., Findings of ACL 2024; U-shaped positional attention bias independent of relevance; calibration lets the frozen model attend by relevance; up to 15 percentage points on RAG tasks. The attention = relevance + position decomposition-by-permutation is the paper's core estimation protocol (abstract confirms bias measured 'regardless of relevance' by varying position). Applicable: we already own permutation harnesses, and calibrating the final-read attention over carrier/tree nodes is implementable with hooks. Not a 5: their bias estimation needs multiple forwards per input arrangement, and our fence + pos-reset already removes most absolute-position variation, so the measured bias component may be small — worth measuring before building the fix.

### Found in the Middle: How Language Models Use Long Contexts Better via Plug-and-Play Positional Encoding (Ms-PoE)
*Zhenyu Zhang et al. (VITA-Group / Microsoft) — NeurIPS 2024 (arXiv 2403.04797)* — applicability 4/5 — https://arxiv.org/abs/2403.04797

**What:** Training-free, zero-overhead fix for lost-in-the-middle on frozen LLMs: rescales RoPE position indices with a distinct scaling ratio per attention head (multi-scale fusion of short- and long-distance views), relieving RoPE long-term decay. Improves middle-position retrieval accuracy by up to ~20 points on multi-doc QA and key-value retrieval. Code: github.com/VITA-Group/Ms-PoE.

**Use here:** Drop-in complement to our per-block M-RoPE position reset: apply head-wise position-index rescaling so the final query and upper tree nodes see all frame blocks within the short-distance regime where softmax reads are strongest; purely an inference-time change to the same RoPE machinery we already patch, testable against our posreset sweep.

**Verifier note:** Paper, mechanism, and venue verified: Zhenyu Zhang et al., NeurIPS 2024 (poster 94207); training-free head-wise RoPE position-index rescaling, no fine-tuning or added overhead; code at github.com/VITA-Group/Ms-PoE. NUMBER CORRECTION: the claimed 'up to ~20 points' matches nothing reported — actual figures are MDQA +3.92 avg (65.04 vs 61.12), key-value retrieval +43.72 avg (79.96 vs 36.24), and headline 'up to 3.8 average on Zero-SCROLLS'. So it understates KV retrieval and overstates MDQA by ~5x; fix before citing. Applicability real: we already patch the same RoPE machinery, so head-wise index rescaling on the final query is a drop-in sweep — but M-RoPE's 3D position structure makes the port nontrivial.

### MA-LMM: Memory-Augmented Large Multimodal Model for Long-Term Video Understanding
*He et al., 2024 (CVPR 2024)* — applicability 3/5 — https://arxiv.org/abs/2404.05726

**What:** Processes video frames ONLINE, auto-regressively accumulating past frame information in a long-term memory bank (with similarity-based memory-bank compression that merges adjacent redundant entries); the memory bank plugs into multimodal LLMs off-the-shelf and sets SOTA on long-video QA and captioning.

**Use here:** The sequential-accumulation alternative to our binary tree: instead of a one-shot fan-N read (which our capacity law says fails past fan~4), aggregate frame registers into a running memory bank one frame at a time - chain topology, fan-in 2 per step. We can implement the memory-bank read as an inference-time intervention on frozen Qwen (append bank tokens per step, merge by similarity) and test whether their merge heuristic survives exact counting, where merging 'redundant' identical-room frames is precisely what destroys the tally. Predicted failure mode makes it a sharp diagnostic baseline for the thesis.

**Verifier note:** Verified including the paper-body detail: MBC computes cosine similarity between temporally adjacent memory-bank tokens and averages the most similar pair to keep bank length fixed; online processing, off-the-shelf integration claim, CVPR 2024 all confirmed. The predicted failure (merging 'redundant' identical-room frames destroys the tally) makes it a sharp diagnostic. Rated 3: the stack is InstructBLIP Q-Former + frozen Vicuna-7B, so 'off-the-shelf' does not extend to Qwen2.5-VL — the merge heuristic must be reimplemented as a custom intervention; the chain-topology fan-2 idea itself is easy, but that part needs no paper.

## value-contamination

### Star Attention: Efficient LLM Inference over Long Sequences
*Acharya, Jia, Ginsburg (NVIDIA) — 2024 (ICML 2025)* — applicability 5/5 — https://arxiv.org/abs/2411.17116

**What:** Two-phase block-sparse attention for frozen off-the-shelf LLMs (no fine-tuning): context blocks are encoded independently with blockwise-local attention, each prefixed with an 'anchor block' (a copy of the first block); query tokens then do sequence-global attention over all cached KV. Key mechanism finding: independent block encoding creates an attention sink at the start of EVERY block, corrupting per-block encodings; anchor blocks absorb the sink (their KV is discarded), making block-local attention statistics approximate global attention. Reports accuracy vs block-size trade-offs (~9.7% degradation at 512K with fixed blocks; larger blocks recover accuracy), preserving 97-100% accuracy overall.

**Use here:** Drop-in fix to try on our fence: prefix each per-frame block with an anchor block (e.g., a fixed prefix or question replica copy) inside the block-diagonal mask, then drop the anchors' spans from the read. If our per-frame carrier states are contaminated by per-block attention sinks, this should clean the values the tree nodes read — zero training, one mask change in gnnformer/fencing.py. Their block-size ablation is also a template for our fan/capacity measurement.

**Verifier note:** Verified (ICML 2025 confirmed via icml.cc poster + PMLR v267): two-phase block-sparse attention on frozen off-the-shelf LLMs; full text confirms every mechanism detail — each block except the first is prefixed with a copy of the first block (anchor), blockwise encoding otherwise 'creates multiple attention sinks at the start of each block', anchors absorb/shift the sinks and their KV is discarded; 97-100% accuracy preserved; larger blocks improve accuracy. ONE NUMERIC DISCREPANCY: claim says ~9.7% degradation at 512K with fixed blocks, but v3 full text reports -6.73% at 512K with 32K blocks (the ~9.7% figure may come from another version/model config; could not confirm). Substance intact, so claim_accurate=true. Rated 5: anchor-prefix-inside-fence is a zero-training one-mask-change experiment in gnnformer/fencing.py directly targeting per-frame carrier sink contamination.

### Why do LLMs attend to the first token?
*Barbero, Arroyo, Gu, Perivolaropoulos, Bronstein, Velickovic, Pascanu — 2025 (COLM)* — applicability 4/5 — https://arxiv.org/abs/2504.02732

**What:** Shows attention sinks (massive first-token attention) are a learned defense against over-mixing: in deep/long-context frozen LLMs (Gemma, LLaMA family) sinks make heads effectively inactive, which provably slows representational collapse; empirically measures how sinks attenuate perturbation propagation through the residual stream, with sink strength growing with training context length and model size.

**Use here:** Two direct uses on frozen Qwen: (1) their perturbation-propagation measurement is exactly the instrument for our analog-relay decay chain (R2 0.94->0.81->0.11) — quantify hop-wise contamination with and without the fence; (2) design guidance for the fence layout: each block-diagonal frame block currently amputates the global sink, so adding a per-block sink/no-op token inside the fence should reduce over-mixing contamination of carrier states — a pure attention-mask change we can test immediately.

**Verifier note:** Verified; COLM 2025 acceptance confirmed via OpenReview and the COLM accepted-papers list. Sinks-as-over-mixing-defense thesis, perturbation-propagation experiments, and dependence on context length/depth/data packing confirmed; Gemma/LLaMA are the models used. Applicability 4: the per-block sink token inside the fence is a pure attention-mask change testable immediately, and their perturbation instrument maps onto the analog-relay decay chain; design guidance plus instrument, not a competing method.

### Vision Transformers Don't Need Trained Registers (test-time registers)
*Nick Jiang, Amil Dravid, Alexei A. Efros, Yossi Gandelsman — NeurIPS 2025 Spotlight (arXiv Jun 2025)* — applicability 4/5 — https://arxiv.org/abs/2506.08010

**What:** Finds a sparse set of 'register neurons' that concentrate high-norm (massive-activation / sink) values on a few outlier tokens, producing noisy attention maps in CLIP/DINOv2 and VLMs. Fix is fully training-free: append one extra untrained token at inference and shift the register-neuron activations into it, mimicking trained registers on a frozen model. Code released (github.com/nickjiang2378/test-time-registers); already extended to off-the-shelf VLMs for cleaner text-to-image attribution.

**Use here:** Add a test-time register token to the frozen Qwen2.5-VL (vision tower and/or LM) so sink/high-norm mass is absorbed by a dedicated token instead of contaminating our carrier tokens and tree-node reads; also gives cleaner attention maps for probe_attention_map diagnostics. Mechanically identical to our appended-token machinery, so it slots into the existing fencing code path.

**Verifier note:** Verified: NeurIPS 2025 Spotlight; sparse 'register neurons' cause high-norm outlier tokens/noisy attention; training-free fix appends an untrained token and shifts register-neuron activations into it; extended to off-the-shelf VLMs; code confirmed at github.com/nickjiang2378/test-time-registers (project page avdravid.github.io/test-time-registers). Fits our appended-token machinery and would clean probe_attention_map diagnostics. Not a 5: requires identifying register neurons in Qwen2.5-VL specifically (their released identifications cover CLIP/DINOv2 and some VLMs, not necessarily Qwen2.5-VL), and the benefit to carrier-token contamination is hypothesized, not shown.

### Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free
*Zihan Qiu et al. (Qwen team) — NeurIPS 2025 Oral, Best Paper (arXiv 2505.06708)* — applicability 3/5 — https://arxiv.org/abs/2505.06708

**What:** Systematic study (30 variants, 15B MoE + 1.7B dense) showing a query-dependent, head-specific sigmoid gate applied to the SDPA output eliminates attention sinks and massive activations, adds non-linearity to the low-rank value path, enables heads to null their output (input-dependent sparsity), and improves long-context extrapolation (RULER). Used in Qwen3-Next; code and models released (github.com/qiuzh20/gated_attention).

**Use here:** Two uses: (1) mechanistic backing for why ungated softmax reads saturate — softmax must place mass somewhere, so irrelevant values pollute the aggregate, the suspected cause of our fan-in law; (2) retrofit experiment within our allowed budget: insert tiny per-head sigmoid gates after SDPA on layers >= L*=12 of the frozen Qwen (trained like our ridge heads / LoRA), and measure whether gating changes the accuracy-halving-per-fan-doubling law and the sink profile.

**Verifier note:** Verified: Qwen team; NeurIPS 2025 Oral AND Best Paper Award confirmed (papers.nips.cc + Alizila announcement); 30+ variants of 15B MoE / 1.7B dense on 3.5T tokens; head-specific sigmoid gate after SDPA eliminates sinks/massive activations, adds non-linearity/sparsity, improves long-context extrapolation; incorporated in Qwen3-Next (Sep 2025); code at github.com/qiuzh20/gated_attention. Rated 3 because it is a PRETRAINING-scale architecture study — gates are trained with the model from scratch. The proposed retrofit of gates onto frozen Qwen2.5-VL is our own untested extrapolation (a frozen model's value pathways were never trained to be gated; a fresh identity-initialized gate + LoRA-style training might work but nothing in the paper de-risks it). First-rate as mechanistic citation for why ungated softmax reads saturate.

# Independent visual memory: literature and decision audit

Date: 2026-09-10. Proposal only; no implementation, training or performance claim.

Independent visual memory is a motivated mechanistic experiment, but its primitives are established. The question is whether evidence kept independent of surrounding observations makes native extensive aggregation reliable. It is not a new attention family or a new persistent-memory principle.

The frame diagnostic gives a concrete motivation: isolated first-token conjunction judgments were192/192 correct for both frozen Qwen and the hidden adapter; indexed full N32 contexts were132/192 and156/192. This implicates context/access/reference handling, without uniquely identifying corrupted visual features. All secondary generated strings had punctuation and failed the registered strict parser. V4 increased data diversity with unchanged architecture: OOD gains+2.31/+19.91pp failed its both-seed screen. Neither diagnostic establishes an absent binding mechanism or an architectural solution.

## Candidate computation

Retain the frozen vision encoder's ordinary merger outputs E_f for each image. For each language query q:

```
r_f(q) = Attention(q, K(E_f), V(E_f))
m_f(q) = SiLU(W_r r_f(q) + W_q q + b) - SiLU(W_q q + b)
delta(q) = U sum_{visible f} m_f(q)
```

Inject delta into a native language residual; use ordinary answer CE and vocabulary. There is no numeric output head, external tally, generated frame answer, or extra vision-encoder forward. Installed Qwen2.5-VL source uses separate image/frame boundaries in vision attention and restores merged patch order. This establishes the intended independence in source code; a numerical parity test is still required. The196 tokens/image in current runs is a preprocessing choice, not a hardcoded architectural constant.

For fixed q and preprocessing, adding another image does not change an existing image's branch contribution, and duplicating an image duplicates its contribution. These limited properties do not establish end-to-end count correctness: q can change with context or already encode an answer.

## Closest literature

- [Persistent Visual Memory (May2026)](https://arxiv.org/html/2605.00814v1) already uses current language states to retrieve original visual embeddings through a low-dimensional gated branch beside the FFN, activated on text tokens. Section6.4 compares raw embeddings with processed hidden states. Its mathematical guarantee explicitly fixes the query. The proposed memory-source control restricted to corresponding visual positions is more specific, but raw visual access throughout reasoning is already a central PVM claim.
- [NVLM](https://arxiv.org/html/2409.11402v1) cross-attends to visual-encoder tokens, with a hybrid model retaining a decoder visual pathway alongside separate cross-attention.
- [Flamingo](https://arxiv.org/html/2204.14198v1) independently encodes frames and uses gated visual cross-attention during language generation. Its resampling and masking differ, but persistent access is established.
- [Deep Sets](https://papers.neurips.cc/paper/6931-deep-sets.pdf) covers conditioning a shared per-element nonlinear transform on additional information and summing the results. Its theory is not a guarantee of fixed-width learnability or numerical extrapolation here.

The contribution would have to be a demonstrated aggregation mechanism and reliable native length/count/composition benefit, with these antecedents acknowledged.

## Feasibility and failure modes

Count-only CE can train this branch, but local contributions are weakly identified. Many decompositions give the same small-range totals, the backbone can bypass the branch, and a small new reader must align early visual features with a late language query. Perfect isolated answers from the entire pretrained VLM do not show that a small reader can extract conjunctions from merger features.

Centering removes the zero-read baseline, not the negative-image baseline. If a nonmatching frame has mean message mu_minus, negatives contribute approximately(N-K)mu_minus. Even zero-mean noise can grow with N. Earlier centered text experiments did not solve extrapolation. No clean-Vision centering success is established.

Image boundaries suit this benchmark's observation units, but are an inductive bias. One read per image does not guarantee binding or represent every relevant event in a complex image. A good extensive latent representation still needs the native decoder to emit unseen numerals.

## Focused contingent experiment

If the frozen reasoning assay leaves a useful gap, compare independent merger memory against corresponding layer14 visual-token states. Hold the branch, boundaries, normalization, dimensions, initialization, language positions, optimizer, examples, native loss and resource budget fixed. This tests memory-source usefulness. A positive result alone cannot separate contextual interference from feature scale, semantic level or alignment differences.

A pragmatic shared curriculum would start both arms on balanced one-image COUNT examples (native0/1 answers) and continue on the same multiframe counts. Include positive, character-only, room-only and neither examples selected without model outcomes. This is established local/easy-to-hard supervision, not a novel training principle, and gives extra supervision that must be counted. The source comparison cannot separately attribute a gain to that curriculum; that requires a different controlled contrast.

Retain K0 at multiple training lengths to expose negative accumulation through ordinary CE. Record local count performance, negative-message statistics, fixed-query negative extensions, familiar-count length extrapolation and unseen counts separately. Local0/1 training does not solve unseen numeral decoding by construction. Frame-MEAN versus SUM is a separate attribution question, not an automatic additional grid.

## Causality and reasoning scope

Only language positions may query the branch, and only complete already-presented images may be visible. Apply the same rule during prefill and cached decode. Reuse the standard visual encoding and optionally projected visual K/V; every subsequent language query still performs its own read. Operating during reasoning is software compatibility. Demonstrating composition requires a second evidence query under controlled intermediate prefixes and ultimately end-to-end traces.


## Registered first contrast (V5)

The final first comparison is SUM versus MEAN with the same independent raw visual memory, native count CE only, and the immutable V4 refresh schedule (seeds4/5). No local curriculum is added. At identical query and weights SUM=N_visible*MEAN, so the test concerns extensive scaling and its optimization consequences. It does not establish information-bandwidth gain or attribute an effect to memory source. The earlier source/curriculum discussion above remains a contingent alternative, not a launched experiment. See PREREG_AGG.md and outputs/native_aggregation_vlm/v5/INDEX.md.

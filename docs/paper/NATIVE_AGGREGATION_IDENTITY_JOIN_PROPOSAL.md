# A controlled MMReD identity join beyond the relevance sum

This proposed task asks whether the aggregation branch preserves person–room associations that a relevance count cannot express. It uses the canonical MMReD images and a learned native answer readout. It does not introduce a new attention operator or establish reasoning composition. Full training is not released by this document.

Use the nine people in `stage_native_vision_v2_clean.CHARS`: Sandra, Mary, Michael, John, Daniel, Laura, Peter, Emma and Noah. The actual six-room vocabulary is **Kitchen, Bathroom, Garden, Office, Bedroom and Park**, from [`stage_native_vision_pilot.PARK_ROOMS`](../../scripts/stage_native_vision_pilot.py). Hallway is not part of this project's canonical data law.

Each contrast family contains three people, two selected rooms and three answer-changing variants. There are exactly six relevant images: two occurrences of each person, with three images in each selected room. One person appears once in each room; each other person appears twice in just one room. The answer is the person who appears in both rooms. For example:

| Person | Variant 1: Kitchen / Garden | Variant 2: Kitchen / Garden | Variant 3: Kitchen / Garden |
|---|---:|---:|---:|
| Mary | 1 / 1 | 2 / 0 | 2 / 0 |
| John | 2 / 0 | 1 / 1 | 0 / 2 |
| Emma | 0 / 2 | 0 / 2 | 1 / 1 |
| Answer | Mary | John | Emma |

All three variants have identical relevant count, person frequencies and room frequencies. The person–room association changes. These counts specify generation and offline scoring only; no histogram or intersection is computed during inference. Keep the two occurrence slots for each person and all background images fixed across variants. Randomize the slot assignment independently across families. Counterbalance the assignment of the two nonanswer people to rooms so the example's ordering is not a predictor.

Use this literal global template:

> Consider only images in the {room A} or the {room B}. Which person appears in both rooms? Reply with the person's name only.

Every local stream receives its image and this same generic wrapper, with the complete global question inserted unchanged:

> Collection question: {global question}  
> Does this image satisfy the question's inclusion restriction? Reply with exactly 1 for yes or 0 for no.

There is no external parser extracting rooms or identities from the question. The frozen model interprets inclusion; the learned payload must retain the image's person and room. Do not reserve coordinates for identities or roles, introduce a local value-decoding parser, or implement the join outside the learned branch.

The measurement interface is a **new binary native-argmax gate**, explicitly separate from V18's soft `max(0,p(1)−p(0))` rule. At the original query, evaluate the complete unmasked native vocabulary head. Literal token `1` sets `b=1`; literal `0` sets `b=0`. Any other argmax is a measurement failure. Do not silently close it, renormalize a binary vocabulary subset, or replace it with ground truth. Store the original gate and reuse it over answer prefixes. This removes the ambiguity whereby a soft gate's confidence sum can itself encode the answer.

The two matched branches use the same 96-dimensional payload map, native residual readout, parameter shapes and initialization. The vector arm reads each local state. The scalar control computes the binary count first and multiplies one query-derived payload by that count:

`vector: z = SUM_i [b_i * tanh(Wlocal RMS(h_i) + q + bias)]`  
`scalar: s = SUM_FP32_i b_i; u = tanh(Wlocal RMS(h_global) + q + bias); z = s * u`

For these lengths, summing exact binary FP32 values produces the exact integer count, independent of their positions. Do not implement the scalar aggregate by summing interspersed zero and repeated floating-point payload vectors: reduction rounding could reveal their arrangement. Dense projection work may be matched by evaluating repeated copies of the global state, but only one fixed copy may supply u; no position-dependent reduction or local-state result can reach z. Profile and disclose that implementation.

Thus the control receives exactly the binary count and global query/history. All parameters remain trainable; there is no hidden local-state bypass. With correct gates, bitwise-identical global inputs and deterministic generation, its input is identical across a three-variant contrast, so it can answer at most one variant correctly. Verify s, the chosen u, z and the native global inputs from captures, including gate permutations. This proves a limitation of the relevance-sum control, not a mathematical necessity for exactly 96 coordinates.

Both arms execute the same N local plus one global native streams, original full-vocabulary probe and dense branch work. Each trajectory adds one original norm/head probe beyond its ordinary native token forwards, with no additional vision pass. Global answers remain unmasked name-plus-native-EOS generation, without candidate selection or an external answer algorithm. The CPU tokenizer check must establish that all nine canonical names plus EOS fit the fixed four-token budget. Scoring uses the complete canonical name after surrounding-whitespace removal; no alias mapping or partial-name credit.

Prepare N8→N16 training families with K6, using only inserted irrelevant images for the length extension. Test fresh paired N32→N64 families, again with K6. Irrelevant images lie outside the selected rooms and may show the same people. Preserve canonical Step rendering and record parent positions, including Step changes after insertion. Reuse canonical image atoms where available; any missing atom uses the unchanged renderer, with no visual augmentation.

Train on twelve unordered room pairs. Hold out Kitchen/Bathroom, Garden/Office and Bedroom/Park from branch fitting. The frozen gate feasibility audit deliberately covers these pairs; they are not unseen to that measurement audit. Report seen-pair length extrapolation separately from held-pair composition. Balance all nine answer names and include every answer-changing variant. This first task does not test increasing relevant-item count or unseen answer names. If residual consistency is retained, apply it only between length extensions of the **same variant**; never penalize the deliberately different answers across variants.

Before fitting, freeze the literal local wrapper and run a direct gate feasibility audit covering all nine people, all fifteen room-pair questions, all six depicted rooms and Steps {1,8,16,17,32,64}: 4,860 distinct image/question judgments. This fully crosses people, pair questions and Step strata, with 1,620 included and 3,240 excluded cases; report both denominators and equal-class macro accuracy. Require valid, correct native binary measurements throughout this fixed audit. If the interface fails, stop this proposal rather than filter images, substitute oracle gates or search prompts on the same outcomes. V17's counting readout does not validate this new prompt or execution route.

Fresh test gate errors also remain in every denominator. If gates or global-input identities differ across matched variants, disclose that the strict scalar-ceiling premise failed; do not retain only convenient triples. Data cardinalities/seeds, trainable budgets, software call inventory and quantitative success criteria must be registered before their respective execution stages. Subsequent source and CPU data/gate-plan preparation can follow review; native gate profiling and full training require their own frozen releases.

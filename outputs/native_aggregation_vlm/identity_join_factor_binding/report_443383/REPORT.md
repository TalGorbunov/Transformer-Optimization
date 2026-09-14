# Matched factor-binding diagnostic

Independent computation and provenance audits passed. These are cached training first tokens, not native whole answers.

| Interaction | Correct /108 | Complete families /18 | Fixed paired screen |
|---|---:|---:|---|
| product | 79 | 5 | FAIL |
| additive | 58 | 1 | FAIL |

Product after the single fixed cyclic factor permutation: 38/108 first tokens and 0/18 complete families. This intervention is descriptive and does not change either screen. Original factors and query were unchanged; only their pairing changed. Additive pooled invariance passed all7 CPU checks; FP32 reassociation differences are reported separately.

Both arms used identical fresh seed24 tensors,600 full-name-plus-EOS CE updates and the existing global conditioning statistics. Product/additive communicated the same96 coordinates. These learned factors have no assigned person/room roles. A passing screen permits preparation of separately audited native evaluation of that unchanged endpoint, without refitting.

Allocated GPU cost: 55 seconds. GPU work:1221 norm/head batches and42992 rows; zero VLM/vision calls. Independent CPU head replay:21 batches/324 rows, fixed TV<=.02 and exact argmax.

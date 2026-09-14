# Frozen MMReD Vision: question-prefix diagnostic

| Condition | N16 exact /18 | N64 exact /18 |
|---|---:|---:|
| baseline | 4/18 (22.2%) | 1/18 (5.6%) |
| correct_prefix | 4/18 (22.2%) | 1/18 (5.6%) |
| neutral_prefix | 3/18 (16.7%) | 1/18 (5.6%) |

Correct-prefix minus control; paired-anchor descriptive 95% intervals:

- neutral_prefix, N16: +5.6pp [-11.1, +22.2]pp.
- neutral_prefix, N64: +0.0pp [-16.7, +16.7]pp.
- neutral_prefix, pooled: +2.8pp [-8.3, +13.9]pp.
- baseline, N16: +0.0pp [-16.7, +16.7]pp.
- baseline, N64: +0.0pp [-16.7, +16.7]pp.
- baseline, pooled: +0.0pp [-11.1, +11.1]pp.

- Diagnostic subset of the existing V2 main test: 18 anchors, 36 examples; no model or prompt selection from outcomes.
- Correct and neutral prefixes match actual processor token layout, including image positions, but differ in semantics and question repetition.
- Baseline is shorter. Prefix-minus-baseline includes context length, positions, repetition and semantic changes.
- Intervals resample paired anchors, preserving both lengths; they are descriptive conditional intervals with no multiplicity adjustment.
- Frozen direct-answer model only; this is neither a new aggregation method nor evidence of reasoning composition.

[All 108 condition/example outputs](predictions.json) · [Verified summary](summary.json)

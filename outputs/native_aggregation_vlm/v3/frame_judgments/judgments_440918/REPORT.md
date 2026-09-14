# Native frame-judgment diagnostic

| Model | Presentation | First-token binary correct /192 | Generated correct /192 | Parsed /192 | Mean signed margin | Mean P(Yes)+P(No) |
|---|---|---:|---:|---:|---:|---:|
| frozen | isolated | 192/192 | 0/192 | 0/192 | 7.9643 | 1.0000 |
| frozen | full | 132/192 | 0/192 | 0/192 | 1.0028 | 0.9995 |
| v2_hidden_seed0 | isolated | 192/192 | 0/192 | 0/192 | 9.5964 | 1.0000 |
| v2_hidden_seed0 | full | 156/192 | 0/192 | 0/192 | 2.7910 | 0.9997 |

| Model | Full−isolated signed margin [95% pair bootstrap] | C-switch isolated margin Δ | C-switch full margin Δ |
|---|---:|---:|---:|
| frozen | -6.9614 [-7.3545, -6.5546] | 18.8433 | 2.8457 |
| v2_hidden_seed0 | -6.8054 [-7.2865, -6.3174] | 22.7168 | 8.7529 |

All margins are raw first-answer-token logits. Binary accuracy uses the sign of Yes−No, with ties incorrect; it is not native full-vocabulary top-1 accuracy. C-switch Δ is high-scene minus low-scene margin, where the frame label changes No→Yes.

Detailed strata, low/high cells, invariant-label stability, probabilities, conditional parsed denominators, and paired intervals are in [summary.json](summary.json). Every judgment and native top-1 token are in [predictions.json](predictions.json).

## Scope

- Diagnostic-only, fixed 16 OOD binding pairs; models fixed in advance: frozen base and V2 hidden seed0, independent of V3 rankings.
- Six selected original frame positions per scene, both low/high scenes and isolated/full presentations: 384 judgments per model, 768 total.
- Isolated images retain original pixels and Step labels; the identical indexed question is used in both presentations. Context, image position, and indexing demands change.
- Primary readout is the raw first-token logit(Yes)-logit(No); full-vocabulary binary mass and gold-token probability expose off-format readouts.
- Generation is secondary: stripped, case-folded Yes/No only. Additional words or punctuation are unparsed and incorrect; all examples remain in accuracy denominators.
- Intervals resample all judgments from each of 16 pairs together. They are descriptive conditional intervals, with no multiplicity or seed-population inference.
- R-swapped and unchanged-positive labels are invariant, but their margins need not be invariant under changed images or surrounding context; those changes have no imposed direction.
- The preceding fixed-marginal count probe showed positive count changes for every pair in every tested model. This diagnostic examines precision and context effects; it does not presume absent conjunction binding.
- Success on selected frame questions does not prove conjunction information survives ordinary counting prompts or identify where aggregation fails.
- This is not a trained frame classifier, external tally method, new aggregation architecture, or test of reasoning composition.

[Frozen CPU plan](plan.json)

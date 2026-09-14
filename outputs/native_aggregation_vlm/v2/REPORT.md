# Clean MMReD Vision attribution: V2

| Arm | N16 K0–8 | N32 K0–8 | N64 K0–8 | Familiar OOD | N32 K9–16 | N64 K9–16 | Selected epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| Global read | 59.3% | 37.0% | 16.7% | 26.9% | 12.5% | 14.1% | 5 |
| Hidden only | 46.3% | 36.1% | 22.2% | 29.2% | 4.7% | 12.5% | 7 |
| Middle + upper LoRA | 39.8% | 27.8% | 15.7% | 21.8% | 6.2% | 7.8% | 5 |
| Upper LoRA | 29.6% | 17.6% | 14.8% | 16.2% | 0.0% | 0.0% | 5 |

Each familiar cell has108 examples; each unseen-count cell has64. Lengths within each family share anchors.

Registered screens: V2.1=False; V2.2=True.

- Global minus Hidden only: -2.3pp; paired-anchor 95% interval [-10.2, +5.1]pp.
- Global minus Middle + upper LoRA: +5.1pp; paired-anchor 95% interval [-2.3, +12.0]pp.
- Global minus Upper LoRA: +10.6pp; paired-anchor 95% interval [+4.2, +17.1]pp.

## Interpretation limits

- One training seed; bootstrap resamples paired test anchors and does not quantify training variability.
- Clean V2 changes the generator for every arm; V1-to-V2 accuracy changes are not a controlled comparison.
- Hidden and global approximately match parameters but differ in nonlinear width/input variance; this is not a pure input-only ablation.
- Middle LoRA and adapters use their respective preregistered learning rates; optimization policy remains a comparison factor.
- K9..16 changes answer support; K10..16 are multi-digit. First-token NLL is not whole-answer likelihood.
- Global reconstructs an existing attention read redundantly. Timing under concurrent jobs is descriptive.
- These runs use short direct answers. Reasoning-token composition is untested.

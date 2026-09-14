# Identity-join trainability diagnostic

The four-arm pilot **failed every training and development criterion**. Removing residual consistency did not prevent clipped selection from closing every first-query image gate. Smooth selection learned some answer information, but did not solve a complete answer-changing family.

| Selection | Objective | Train whole /108 | Development whole /54 | Complete train/dev families |
|---|---|---:|---:|---:|
| Clip | Native CE | 12 | 6 | 0/18; 0/18 |
| Clip | CE + consistency | 12 | 6 | 0/18; 0/18 |
| Sigmoid | Native CE | 40 | 18 | 0/18; 0/18 |
| Sigmoid | CE + consistency | 42 | 18 | 0/18; 0/18 |

First-token correctness equals whole-answer correctness in every cell. The fixed criteria required at least103/108 training answers and16/18 complete six-context families; development required49/54 and16/18 complete triples. Every outcome remains in the denominator.

All arms used the same fresh seed24, 1,041,697-parameter rank96 core, initialized tensor values, first gradients, 108 balanced training contexts, 54 intact N8/N16 pairs and600 updates. Development comprised54 other N16 contexts on the same questions/person trios. The frozen native model generated unmasked names plus EOS, at most four tokens, once from each final checkpoint. This is a small optimization diagnostic, with no extrapolation evaluation, checkpoint selection or reasoning claim.

Both clipped models close every image gate at the first query on all108 training and54 development contexts. Their current aggregated image message is therefore exactly zero. The smooth models have no completely closed first-query gates. This separates the observed clipping failure from a claim that the consistency loss was necessary for it. The two-answer training difference between smooth objectives, with identical development accuracy, is not evidence of a reliable objective advantage.

Separate CE/consistency gradients at steps1,2,32,128,300,600, all native captures and all progress snapshots passed independent audit. A small total consistency gradient does not exclude a larger effect on a particular parameter group; these measurements are descriptive. Easy continuation/EOS losses must not be mistaken for learning the first answer token.

Independent report443233 completed78CPU-seconds. Root verified all64 current/archive source hashes and the analysis, outcomes and gate records. Total cost is2,410GPU-seconds, maximum four GPUs, including the preserved V1 resource-gate failure, V2 profiles and197GPU-seconds of V2 main failures caused by repeated exclusive progress-file writes. V3 changed progress filenames and accounting only; its repeated-write CPU fixture and all final audits passed.

The preregistered conditional follow-up is the missing fixed-half-weight CE-only join control. It freezes the97 selector coordinates, retains1,041,600 trainable parameters, and reuses the same600-step recipe. Its purpose is to test whether learning selection obstructs this particular fit. It remains a diagnostic, not fresh confirmation or an accepted aggregation result.

[Audited report](../../outputs/native_aggregation_vlm/identity_join_optimization_v3/report_443233/REPORT.md) · [Analysis and gradients](../../outputs/native_aggregation_vlm/identity_join_optimization_v3/report_443233/analysis.json) · [All native outcomes](../../outputs/native_aggregation_vlm/identity_join_optimization_v3/report_443233/outcomes.json) · [Uniform control protocol](NATIVE_AGGREGATION_UNIFORM_JOIN_CONTROL.md)

# First-query supervision local joint-code diagnostic

All independent computation and provenance audits passed. This privileged control uses raw per-image person/room codes and the exact four original unfitted readout tensors. Only backward supervision changes from the passed paired full-sequence control: first-query CE retains its original1/(16*Lscene) coefficients; continuation and EOS CE remain diagnostic. The6000-update optimizer settings,48000 paired presentations, all full-prefix forward shapes and final-only decision remain fixed.

| Endpoint | Correct /216 | Original /108 | Flipped /108 | Complete families /18 | Saved GPU screen |
|---|---:|---:|---:|---:|---|
|6000|74|36|38|0|FAIL|

The registered screen requires206 pooled correct,103 in each orientation and16 complete twelve-context families. All216 first-query full-vocabulary logits and all22 captured computations are retained. CPU replay covers14 native-shape batches/216 rows, using the unchanged TV<=0.02 and exact-argmax rule.

Removing continuation/EOS supervision was insufficient for this fixed privileged first-query training screen. The capacity witness remains distinct from a gradient-descent learnability guarantee; this result does not isolate a unique remaining optimization cause.

No result here releases native generation, fresh confirmation, continuation, an architecture change or a new optimization budget. Prior failed studies remain unchanged. Logged first-token/continuation/EOS losses and cycle trends are descriptive observations, not convergence certificates.

Allocated GPU cost: 69 seconds. The run used6014 core/norm/head calls each and213546 head rows, with no VLM or vision calls. Maximum CPU native replay TV: 0.003701020.

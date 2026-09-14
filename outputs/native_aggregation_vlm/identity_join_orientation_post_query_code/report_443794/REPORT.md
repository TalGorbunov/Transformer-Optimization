# Post-SiLU query local joint-code diagnostic

All independent computation and provenance audits passed. This privileged control uses raw per-image person/room codes and the exact four original unfitted readout tensors. Only final query placement changes from the passed first-query control: delta=U(SiLU(Wagg*z+b)+q). First-query CE keeps its original1/(16*Lscene) coefficients; continuation and EOS CE remain diagnostic. The6000-update optimizer settings,48000 paired presentations, original parameter bytes, full-prefix shapes and final-only decision remain fixed.

| Endpoint | Correct /216 | Original /108 | Flipped /108 | Complete families /18 | Saved GPU screen |
|---|---:|---:|---:|---:|---|
|6000|84|40|44|0|FAIL|

The registered screen requires206 pooled correct,103 in each orientation and16 complete twelve-context families. All216 first-query full-vocabulary logits and all22 captured computations are retained. CPU replay covers14 native-shape batches/216 rows, using the unchanged TV<=0.02 and exact-argmax rule.

Moving the final query outside SiLU was insufficient for this fixed privileged first-query screen. The unchanged parameter count and prior capacity witness are distinct from a learnability guarantee; no further readout-placement conclusion follows.

No result here releases native generation, fresh confirmation, continuation, an architecture change or a new optimization budget. Prior failed studies remain unchanged. Logged first-token/continuation/EOS losses and cycle trends are descriptive observations, not convergence certificates.

Allocated GPU cost: 75 seconds. The run used6014 core/norm/head calls each and213546 head rows, with no VLM or vision calls. Maximum CPU native replay TV: 0.003659391.

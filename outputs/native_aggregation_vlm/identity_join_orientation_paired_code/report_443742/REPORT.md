# Paired-minibatch local joint-code optimization diagnostic

All independent computation and provenance audits passed. This privileged control uses raw per-image person/room codes and the exact four original unfitted readout tensors. Only the ordering of the same48000 length-pair presentations changes: original/flipped counterparts share minibatches. The6000-update optimizer, objective and final-only decision remain fixed.

| Endpoint | Correct /216 | Original /108 | Flipped /108 | Complete families /18 | Saved GPU screen |
|---|---:|---:|---:|---:|---|
|6000|68|32|36|0|FAIL|

The registered screen requires206 pooled correct,103 in each orientation and16 complete twelve-context families. All216 first-query full-vocabulary logits and all22 captured computations are retained. CPU replay covers14 native-shape batches/216 rows, using the unchanged TV<=0.02 and exact-argmax rule.

The fixed optimizer still fails this privileged both-orientation training screen. A separately audited capacity witness does not establish that gradient descent finds its solution under this recipe.

No result here releases native generation, fresh confirmation, continuation, an architecture change or a new optimization budget. Prior failed studies remain unchanged. Logged first-token/continuation/EOS losses and cycle trends are descriptive observations, not convergence certificates.

Allocated GPU cost: 64 seconds. The run used6014 core/norm/head calls each and213546 head rows, with no VLM or vision calls. Maximum CPU native replay TV: 0.003722652.

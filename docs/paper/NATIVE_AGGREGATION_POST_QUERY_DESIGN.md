# Protecting nonlinear aggregation from the direct query offset

The broader aim remains a useful aggregation path evaluated once per output token, with evidence first on MMReD Vision. No current join model meets that aim. The next experiment tests one architectural hypothesis, not a new attention mechanism.

Let z be the sum of local messages, q the projected global state, a=Wagg z+b and U the output projection. Compare:

- Current readout: delta = U SiLU(a+q).
- Proposed readout: delta = U [SiLU(a)+q].

The current aggregate Jacobian is U diag(SiLU'(a+q)) Wagg. The proposal replaces the derivative argument with a, removing the direct query offset from the nonlinear aggregate computation. It adds no learned tensors or tuned scales. It changes the function class and gradients, despite identical zero-U initial outputs. In particular, an independently supplied query no longer interacts nonlinearly with z inside this readout. This could hurt general query-selective retrieval. Here the local codes and visual messages are already question-conditioned, and native normalization still couples the residual to the global state.

The motivation is bounded. Two failed code fits have mean final query norms96 and139 versus projected aggregate norms12 and11. Small-slope fractions are55% and81%; small-curvature fractions60% and87%. These are observations at saved activations, not evidence that removing the query will solve learning. The query-zero constructive join solution remains inside both classes. Moving q outside cannot prevent Uq from dominating the eventual native readout and does not repair every possible optimization problem.

Three independent fits compare to fixed parent runs:

| Arm | Fixed comparison | Objective retained | Change |
|---|---|---|---|
| Exact per-image local codes |443775, audited443779|First-query CE at original weights|Final query placement|
| Visual product messages |443736, audited443743|Full-name/EOS CE|Final query placement|
| Visual additive messages |443737, audited443743|Full-name/EOS CE|Final query placement|

Every arm keeps its original unfitted initialization,6000-update schedule, paired presentation order, native inputs/head and forward shapes. Visual local query conditioning is unchanged. The original controlled three-arm data support, not fresh evaluation data, is used. Each fit has a120-second GPU allocation; CPU preflight/report caps are90/300seconds. After both preflights pass, run at most three GPUs in parallel,360GPU-seconds total. Preserve complete numerical evidence and the existing strict fidelity gates. Each final first-query screen remains206/216 pooled,103/108 per orientation and16/18 complete families.

A code-only pass would locate a workable privileged readout while leaving visual learning unresolved. A visual pass requires subsequent actual native whole-answer validation before fresh-data evidence. Neither training success nor the identity-join witness establishes extrapolation, efficient superiority or reasoning composition. A failed placement comparison ends this particular readout modification; do not follow it with an activation, query-scale or initialization sweep under the same hypothesis.

The ingredients remain familiar conditional set pooling and nonlinear residual readout. Any eventual paper should rest on a demonstrated aggregation failure and a general, measured repair, with N+1 parallel-stream compute reported explicitly. A final-normalization write outside the KV cache offers only prefix-mediated composition; benefit for reasoning models still needs an actual reasoning evaluation.

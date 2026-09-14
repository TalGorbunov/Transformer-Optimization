# Privileged local joint-code learnability control

Independent computation and provenance audits passed. Perfect per-image person/requested-room labels were supplied as fixed raw joint codes. The readout received their half-SUM and cached global state; no answer code or precomputed bag intersection was supplied.

| Cached training first-token screen | Correct /108 | Complete families /18 | Mean first-token NLL |
|---|---:|---:|---:|
| FAIL | 38 | 0 | 1.388607632 |

All 108 contexts and 18 complete six-context families remain in the denominator. The unchanged screen requires 103/108 first tokens and 16/18 entirely correct families. Training used the original 600 full-name-plus-EOS CE updates and four exactly selected unfitted readout tensors; only step 600 was evaluated after reset and restricted reload. First-token, name-continuation and EOS training losses and complete-cycle trends are descriptive artifacts.

This fixed recipe does not fit even the privileged local-code/readout screen. The readout and optimization budget remain unresolved; failure does not prove missing information or impossibility.

This is a privileged training diagnostic, not a vision inference method, whole-answer evaluation, generalization result or reasoning-composition result. Previous failed screens and architecture stopping rules remain unchanged. No subsequent fit or benchmark evaluation is automatically released.

Allocated GPU cost: 24 seconds. GPU work: 607 core/norm/head calls each and 21,442 head rows; zero VLM/vision calls. CPU native replay: seven final batches, 108 rows, maximum full-vocabulary TV 0.003443525, fixed TV <= 0.02 and exact argmax. All 13 captured readout batches and all 1,296 semantic-code occurrences were independently checked.

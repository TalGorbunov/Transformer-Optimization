# First-query supervision alone did not solve learning

Removing name-completion and EOS gradients from the paired exact-local-code control produced74/216 correct first answers:36/108 original,38/108 room-swapped,0/18 complete families. This fails the unchanged206/103each/16family screen, as did the full-sequence comparison at68/216. The small difference is a single-seed diagnostic, not an established benefit.

The native forward shapes and213330 training positions were unchanged; first-token CE retained exactly its original1/(16*Lscene) coefficient. Only later-position gradients were removed. Actual-loss CPU fixtures verified identical full-prefix forwards, retained first-logit gradients and zero later-logit/delta gradients. All2592 fixed local codes, global states, original unfitted four readout tensors, paired order,6000 AdamW updates and learning-rate schedule were preserved.

Independent report443779 passed all22 captured computation checks, all6000 objective reconstructions and14 native CPU head replay batches covering216 final first queries. Full-sequence CE remains a diagnostic, not a training objective. GPU443775 completed69seconds; preparation443773 and report443779 completed23/56CPU-seconds. [Audited report](../../outputs/native_aggregation_vlm/identity_join_orientation_first_query_code/report_443779/REPORT.md).

The finite-precision first-query capacity witness remains valid. Removing later-token supervision is insufficient under this optimizer and budget; the experiment does not identify a unique failure cause or show convergence. All visual/native/fresh claims remain unestablished.

A separately registered structural control will place the query contribution after the aggregation nonlinearity, preserving parameter count and known query-zero capacity. The saved-code geometry motivates this hypothesis but was measured on earlier full-CE runs. Code will retain this first-query objective; parallel visual product/additive controls will retain their respective full-CE comparison objectives. Each comparison changes only final query placement. This is a change of function class and gradients, not an equivalent reparameterization.

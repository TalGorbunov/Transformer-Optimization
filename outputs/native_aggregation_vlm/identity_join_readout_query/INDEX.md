# Readout-only query ablation — complete, screen failed

Independent report 443418 passed computation/provenance verification. The new product scored 80/108 cached training first tokens and 5/18 complete families, versus the frozen parent's 79/108 and 5/18. The fixed 103/108 plus 16/18 screen failed. This architecture branch ends; no native evaluation, continuation or further variant is released.

| Model / pairing | Correct /108 | Complete families /18 | Mean first-token NLL |
|---|---:|---:|---:|
| Frozen parent, paired |79|5|0.868483189501582|
| Readout-only query, paired |80|5|0.8611090023408852|
| Frozen parent, cyclic |38|0|1.402758027895851|
| Readout-only query, cyclic |11|0|3.273114197614856|

The cyclic intervention is descriptive and never changes the screen. The new paired model corrected 8 parent errors and lost 7 correct predictions. Its own cyclic permutation changed 72 correct predictions to errors and 3 errors to correct predictions. All 108 cases remain in every denominator.

| Execution | Result |
|---|---|
| [CPU check 443413](check_443413/summary.json) | Passed; exact unfitted parent initialization and fixed existing statistics |
| [Product 443416](run_product_443416/summary.json) | Completed 600 full-name-plus-EOS CE updates; 28 allocated GPU-seconds |
| [Independent report 443418](report_443418/summary.json) | Passed audit; cached first-token screen failed |

One GPU attempt, maximum one GPU, 28 allocated seconds. GPU work: 614 core/norm/head calls each, 21,550 head rows, zero VLM/vision calls. CPU replay: 14 final norm/head batches, 216/216 matching argmaxes, maximum TV 0.003819364123046398 against the fixed 0.02 gate. All 20 captures were audited. The parent was hash-bound without repeating its head calls.

Read the [results note](../../../docs/paper/NATIVE_AGGREGATION_READOUT_QUERY_RESULTS.md), [verified report](report_443418/REPORT.md), [analysis](report_443418/analysis.json), [parent comparison](report_443418/parent_query_placement_comparison.json), [permutation comparison](report_443418/product_permutation_comparison.json), [CPU plan](check_443413/plan.json), and [108-file source ledger](report_443418/source_hashes.json). The [proposal](../../../docs/paper/NATIVE_AGGREGATION_READOUT_QUERY_PROPOSAL.md) fixes the intervention and stopping rule. The preceding [geometry observation](../identity_join_factor_binding/query_geometry_443401/summary.json) did not translate into passing this fitting screen. No whole-answer, generalization, bandwidth or reasoning claim follows.

Verified summary SHA256: `fea7ecb1a2f71a943aa126e1f395163a91e2f7ca8bc189c8a2d4ebb1374733a2`.
Verified analysis SHA256: `06f544d0e046be13dda1dd13aa2c41f2cafd4e91f707858bac2f687dd9cc2235`.

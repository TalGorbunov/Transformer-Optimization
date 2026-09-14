# Matched factor-binding diagnostic — complete, both screens failed

Independent report 443383 passed computation/provenance verification. Product scored 79/108 cached training first tokens and 5/18 complete families; additive 58/108 and 1/18. Both failed the fixed 103/108 plus 16/18 screen. The product-only cyclic permutation scored 38/108 and 0/18; it is descriptive and never changes the screen. No native whole-answer evaluation or new fit is released.

| Execution | Result |
|---|---|
| [CPU check 443373](check_443373/summary.json) | Passed; shared fresh seed 24 initialization and fixed existing statistics |
| [Product 443375](run_product_443375/summary.json) | Completed 600 updates; 28 allocated GPU-seconds |
| [Additive 443376](run_additive_443376/summary.json) | Completed 600 updates; 27 allocated GPU-seconds |
| [Independent report 443383](report_443383/summary.json) | Passed audit; both cached first-token screens failed |

Total GPU cost: 55 allocated seconds, maximum 2 concurrent GPUs. Work: 1,221 norm calls and 1,221 head calls over 42,992 rows; zero VLM/vision calls. Independent CPU replay: 21 norm/head batches, 324/324 matching argmaxes; maximum TV 0.003827726 against the fixed 0.02 gate. Product/additive share 600 full-name-plus-EOS CE updates and the original pair order; only step 600 is used.

Read the [results note](../../../docs/paper/NATIVE_AGGREGATION_FACTOR_BINDING_RESULTS.md), [verified report](report_443383/REPORT.md), [analysis](report_443383/analysis.json), [product permutation comparison](report_443383/product_permutation_comparison.json), [CPU plan](check_443373/plan.json), and [99-file source ledger](report_443383/source_hashes.json). The [proposal](../../../docs/paper/NATIVE_AGGREGATION_FACTOR_BINDING_PROPOSAL.md) states the fixed screen, intervention and claim limits. This is a cached training diagnostic, with no generalization or reasoning claim.

Verified summary SHA256: `de38a1498115bc835955cf264a609ead1a095bacb0c3faa99eba10a1ae962238`.
Verified analysis SHA256: `326db6392f648a517e169b206ca2c46b25b28f1d7f2faf3da31a314afcb74f94`.

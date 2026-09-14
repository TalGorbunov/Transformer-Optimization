# The explicit query term reinforces late factor saturation

CPU diagnostic443401 passed on all26 saved paired captures, without a model, aggregation-core, normalization, head or optimizer call. It reconstructs local projections from the saved inputs and weights, then compares actual preactivations with the counterfactual `Wlocal*x+b`. Both use FP64 tanh and reductions. This evaluates geometry, not predictions from a modified model.

Across all108 final training scenes, the product's two factors show:

| Factor | Actual saturated coordinates | Without explicit query | Actual mean absolute payload partial | Without explicit query |
|---|---:|---:|---:|---:|
| A |94.51%|72.59%|0.01121|0.08810|
| B |94.42%|76.57%|0.01168|0.07095|

Saturation means tanh derivative at most0.01. Payload partials include the other factor in the product, while excluding the outer half-weight, readout and loss. Every scene/factor row—216/216—has lower saturation and larger mean-absolute and RMS payload partials without the explicit query. All24 question×length×factor groups agree. Coordinate-level summaries retain the original occurrence weighting; independent scalar recomputation from saved rows confirmed these results.

The signed query contribution to the squared norm of the common local preactivation is positive on every final row. This is more informative than query norm alone: the diagnostic includes examples where a large query cancels local features. Product mean query norm is98.04, compared with common norms111.51/112.94 and centered local RMS norms118.51/123.00.

These are learned operating states. Early sampled training states at steps1–2 are mostly unsaturated, and removing the query slightly reduces their payload partials. Later samples involve different minibatches, so they do not identify a causal learning trajectory. Substantial saturation remains in the final counterfactual. Larger local derivatives do not prove improved loss gradients, fitting or generalization.

The evidence motivates exactly one placement ablation: remove the explicit query from both product factors while retaining it after SUM. Local hidden states still encode the current question and prefix. The ablation starts from the original unfitted parameters and must pass the same complete-family screen before native evaluation.

Evidence: [diagnostic summary](../../outputs/native_aggregation_vlm/identity_join_factor_binding/query_geometry_443401/summary.json), [all query/factor rows](../../outputs/native_aggregation_vlm/identity_join_factor_binding/query_geometry_443401/query_factor_rows.json), [strata](../../outputs/native_aggregation_vlm/identity_join_factor_binding/query_geometry_443401/strata.json), [ablation proposal](NATIVE_AGGREGATION_READOUT_QUERY_PROPOSAL.md).

Summary SHA256: `7f2b8b6592c840c459f1edb97a97b257a1cd96339e03334da74c03518c1a7147`.

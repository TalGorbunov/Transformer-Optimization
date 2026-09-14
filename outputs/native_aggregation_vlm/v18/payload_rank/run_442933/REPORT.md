# V18 positive payloads within each question

All four fixed models, all272 scenes per model and every2176 positive occurrence per model are retained. Original query only.

| Model | Positive occurrences | Questions | Weighted centered energy / uncentered energy | Unweighted centered / uncentered | N32 mean relative P deviation | N64 mean relative P deviation |
|---|---:|---:|---:|---:|---:|---:|
| native_gate_s20 | 2176 | 52 | 5.9482e-12 | 6.02534e-12 | 7.58093e-07 | 8.05309e-07 |
| native_gate_s21 | 2176 | 52 | 6.09767e-12 | 5.96268e-12 | 5.78285e-07 | 6.21411e-07 |
| all_open_s20 | 2176 | 52 | 2.41701e-05 | 2.41701e-05 | 0.00173615 | 0.00196551 |
| all_open_s21 | 2176 | 52 | 4.54128e-11 | 4.54128e-11 | 2.01789e-06 | 2.13762e-06 |

The question means use fresh test data and are not a deployable approximation. No answer was generated, rescored, or changed.

Per-model JSON preserves all96 singular values/energy fractions, each question mean/spread/support, and every scene including K0. FP64 payload heterogeneity and saved FP32 multiplication residuals are recorded separately.

Near-constant values would support a weighted-count interpretation at this query. Variation alone would not establish its causal use by the native decoder.

[Analysis and source identities](analysis.json) · [Consumed input hashes](input_bindings.json).

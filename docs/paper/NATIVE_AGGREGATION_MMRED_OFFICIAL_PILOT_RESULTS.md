# Original MMReD Vision: fixed-summary pilot closed

The ordinary joint-image model substantially outperformed both 32-slot replacement memories. The ordinary model passed the predeclared validation competence screen. All 3,000 complete answers passed the independent numerical and provenance audits; every answer had valid typed JSON and native EOS, with no truncation. The failure concerns accuracy, not formatting or execution.

| Test accuracy | Ordinary | Normalized memory | Mass memory |
|---|---:|---:|---:|
| N8 aggregation macro | 87% | 45% | 52% |
| N16 aggregation macro | 72% | 42% | 35% |
| N32 aggregation macro | 61% | 32% | 28% |
| N32 partner aggregation | 58% | 32% | 26% |
| N32 room aggregation | 64% | 32% | 30% |
| N32 atomic retrieval | 82% | 28% | 26% |
| N32 room-step counting | 26% | 16% | 18% |

Each individual task has 50 test questions per length. The primary macro gives equal weight to partner and room aggregation (100 questions per length). At N32, mass minus ordinary is −33 percentage points, paired whole-world bootstrap 95% interval [−46, −19]; normalized minus ordinary is −29 points [−42, −16]. Mass minus normalized is −4 points [−15, +8], providing no evidence for a mass-channel benefit. These are the frozen 10,000-replicate intervals, not intervals selected after seeing the result.

Pooled N8/N16 ordinary validation accuracy is 95/100 on atomic retrieval, 86/100 on partner aggregation and 77/100 on room aggregation. The descriptive count control is 68/100. These satisfy the frozen competence conditions. The memory deficits already occur within the supported length range, so they do not isolate an extrapolation-only or aggregation-capacity bottleneck. The training diagnostics and higher memory cross-entropy likewise do not prove irrecoverable information loss.

**Decision:** close this fixed-summary replacement configuration. Do not sweep slot count, mass normalization, training duration or placement to rescue it on these exposed tests. No N64/N128 or reasoning replication of this configuration is released. The broader single-forward aggregation objective remains open. A distinct hypothesis can retain the full ordinary evidence and test an additional summary readout against a live parameter-matched generic residual adapter, with equal additional training and fresh counterfactual evaluation; its design and software checks must precede fitting.

The protocol used one fit seed, the same 4,000 original training questions for three epochs, final checkpoints only, and all 400 validation plus 600 N8/N16/N32 test questions per model. The official test is historically used and exploratory. These results reject this configuration under the tested recipe, not all learned aggregation methods or visual memories.

GPU evaluation used 1,860 allocation-seconds across the three models, including the preserved 98-second original resource-gate failure. The continuation evaluated only the remaining 997 questions per arm and retained the original three unchanged. Shared evaluation features used another 781 GPU-seconds once for the experiment. The three main fits used 7,239 GPU-seconds; training feature construction and software profiles are separately logged. Instrumented generation/capture timings are not production latency or FLOPs. All raw native head rows (22,091 total) and all 2,000 memory reconstructions passed the independent CPU audits. The earlier independent-question cache-parity failure remains unqualified; all efficacy answers used cold prefixes.

Evidence: [full report](../../outputs/native_aggregation_vlm/mmred_official_pilot_continuation/report_444111/REPORT.md), [all tables and intervals](../../outputs/native_aggregation_vlm/mmred_official_pilot_continuation/report_444111/tables.json), [costs](../../outputs/native_aggregation_vlm/mmred_official_pilot_continuation/report_444111/costs.json), [figure](../../outputs/native_aggregation_vlm/mmred_official_pilot_continuation/report_444111/test_accuracy.png), [execution](../../outputs/native_aggregation_vlm/mmred_official_native_evaluation_continuation/execution.json). The figure was visually inspected. Statistical analysis SHA256: `3a81316965173b5ef33bc9f76888b771d2342ed822527d4e7a5ffd8f558c1865`.

# Moving the query after SiLU was insufficient

All three fixed training screens failed, and both independent computational reports passed.

| Arm | Original | Flipped | Total first answers | Complete families | Matched parent total |
|---|---:|---:|---:|---:|---:|
| Exact local codes, first-query CE |40/108|44/108|84/216|0/18|74/216|
| Visual product, full CE |43/108|41/108|84/216|0/18|77/216|
| Visual additive, full CE |37/108|37/108|74/216|0/18|73/216|

Only final query placement changed from U SiLU(Wagg*z+b+q) to U[SiLU(Wagg*z+b)+q]. Visual local factors retained their query inputs. Each comparison preserved its original unfitted weights, frozen inputs/native head, paired order,6000-update schedule and objective. The code arm retained first-query supervision at original full-target weights; visual arms retained full-name/EOS CE. All forward shapes and213330+216 head rows per arm were unchanged. The screen remains206/216 pooled,103/108 per orientation and16/18 complete families.

CPU preflights443785/443788 passed19/13seconds, including actual new-core and gradient fixtures. GPU443789/443790/443791 completed75/85/85seconds,245 total, with three concurrent GPUs. Code report443794 and visual report443795 completed49/59CPU-seconds. They independently checked66 captured computations, all6000 logged updates per arm and648 final native CPU head replays. All numerical gates passed. This provides no evidence that the small single-seed differences constitute a useful gain.

The saved earlier geometry remains descriptive; these interventions do not establish a unique cause of failure. Existing first-query capacity and prior108-context training successes remain valid. Moving the direct query outside SiLU is insufficient under these fixed recipes. This query-placement attempt is closed: no follow-up query-scale, activation, normalization or initialization sweep is selected. Failed models do not proceed to native generation or fresh evaluation.

Evidence: [code audit](../../outputs/native_aggregation_vlm/identity_join_orientation_post_query_code/report_443794/REPORT.md), [visual audit](../../outputs/native_aggregation_vlm/identity_join_orientation_post_query_visual/report_443795/REPORT.md), [shared release](../../outputs/native_aggregation_vlm/identity_join_orientation_post_query_visual/three_arm_release.json), [design and limitations](NATIVE_AGGREGATION_POST_QUERY_DESIGN.md).

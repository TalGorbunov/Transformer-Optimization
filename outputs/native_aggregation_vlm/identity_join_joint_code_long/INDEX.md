# Extended local joint-code optimization: completed, training screen passed

Independent [report 443511](report_443511/REPORT.md) passed. The sole primary endpoint, step 6,000, scored **108/108 cached training first tokens and 18/18 complete families**, exceeding the fixed 103/108 and 16/18 thresholds. Steps 600 and 2,000 each scored 36/108 and 0/18 and remained descriptive only.

| Endpoint | Mean first-token NLL | Saved outcomes |
|---|---:|---|
| 600 | 1.2221487155416146 | [All 108](report_443511/outcomes_0600.json) |
| 2,000 | 1.0238207185486736 | [All 108](report_443511/outcomes_2000.json) |
| 6,000 | 0.08028224142585737 | [All 108](report_443511/outcomes_6000.json) |

This was a fresh run from the same unfitted four readout tensors and privileged local codes. Both the optimization horizon and cosine schedule changed from 600 to 6,000 updates. The original 600-step failure remains preserved. The result establishes a late fit to this training screen; it provides no native-vision, whole-answer, generalization, grokking or reasoning evidence. Prior architecture stopping rules remain unchanged.

CPU check 443496 passed; GPU run 443498 used **69 allocated GPU-seconds**, one attempt and maximum one GPU. GPU work was 6,021 core/norm/head calls each and 213,654 head rows, including 213,330 training positions and 324 endpoint positions. VLM/vision calls were zero. Independent CPU reporting audited 29 captures and replayed 21 native-head batches / 324 rows, with exact argmax agreement and maximum TV 0.003452907083556056.

- [Registered proposal](../../../docs/paper/NATIVE_AGGREGATION_JOINT_CODE_LONG_PROPOSAL.md)
- [Readable results](../../../docs/paper/NATIVE_AGGREGATION_JOINT_CODE_LONG_RESULTS.md)
- [CPU plan](check_443496/plan.json)
- [Run summary](run_443498/summary.json)
- [Independent summary](report_443511/summary.json) and [analysis](report_443511/analysis.json)
- [All endpoint strata](report_443511/strata.json) and [training loss descriptions](report_443511/training_loss_descriptions.json)
- [Preserved short-control result](../identity_join_joint_code/report_443476/REPORT.md)

Summary SHA256: `b702fa30ab354708f3d82192c7e54ba83af491ebf5010ca880b4df6996718fed`.
Analysis SHA256: `d0938deeb7212851b6f044ee88d7d09803c17535894c77db4951c2a8ef636ef5`.
CPU plan SHA256: `3e692f8fe3f834fecb59407807f99b7c8b533ece25497c08a1f2c2853ac6f308`.
The source audit binds six new and 106 inherited files. This result does not automatically release another fit or evaluation.

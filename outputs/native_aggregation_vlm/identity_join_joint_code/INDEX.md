# Privileged local joint-code control — complete, screen failed

Independent report 443476 passed computation/provenance verification. Exact per-image person/requested-room codes still yielded only **38/108 cached training first tokens and 0/18 complete families**, failing the fixed 103/108 plus 16/18 screen. N8 and N16 each scored 19/54. Mean first-token NLL was 1.3886076318601472.

This control supplies local joint semantics but no answer code or precomputed bag intersection. The existing half-SUM/SiLU readout must still learn the join. Failure under the fixed 600-update recipe leaves readout/optimization unresolved; it does not establish missing native visual information, an information ceiling or a general aggregation-capacity limit. Prior failed screens and architecture stops remain unchanged. No further fit or evaluation is released.

| Execution | Result |
|---|---|
| [CPU check 443472](check_443472/summary.json) | Passed; exact semantic/code inventory and selected unfitted readout |
| [GPU control 443474](run_443474/summary.json) | Completed 600 full-name-plus-EOS CE updates; 24 allocated GPU-seconds |
| [Independent report 443476](report_443476/summary.json) | Passed audit; privileged training screen failed |

One GPU attempt, maximum one GPU, 24 allocated seconds. GPU work: 607 core/norm/head calls each and 21,442 head rows, zero VLM/vision calls. CPU replay: seven final batches, 108/108 matching argmaxes, maximum TV 0.003443524707108736 against the fixed 0.02 gate. All 13 captures and all 1,296 semantic-code occurrences were audited. Four selected original unfitted tensors provide 697,440 trainable parameters; no learned local encoder or selector is present.

Read the [results note](../../../docs/paper/NATIVE_AGGREGATION_JOINT_CODE_ORACLE_RESULTS.md), [verified report](report_443476/REPORT.md), [analysis](report_443476/analysis.json), [all outcomes](report_443476/outcomes.json), [strata](report_443476/strata.json), [loss descriptions](report_443476/training_loss_descriptions.json), [code reconstruction audit](report_443476/code_reconstruction_audit.json), and [CPU plan](check_443472/plan.json). Provenance includes [seven new sources](report_443476/source_hashes.json) and [99 inherited sources](report_443476/inherited_sources.json). The [proposal](../../../docs/paper/NATIVE_AGGREGATION_JOINT_CODE_ORACLE_PROPOSAL.md) fixes the privileged scope and resource limit. The earlier [answer-code success](../../../docs/paper/NATIVE_AGGREGATION_READOUT_DIAGNOSTIC_RESULTS.md) bypassed the join and nonlinear readout; it was not evidence that this control would succeed.

Verified summary SHA256: `a742e98875b6f48a23ea5dba147b22c326bb25450df7542a883f1d5c23870b45`.
Verified analysis SHA256: `5fff5aff6e8ee4bdbd3ff920b3db4490c5c6f5536f3fd04ac3694f14ece0fc74`.

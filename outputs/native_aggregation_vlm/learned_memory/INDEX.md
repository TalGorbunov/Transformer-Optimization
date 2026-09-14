# Learned aggregation in native decoder memory

**Independent software audit passed; no efficacy claim.** The core reads block 26 and writes before block 27 or before final norm. Historical writes remain live during causal upper-block replay, with native vocabulary projections only at supervised positions.

CPU selected-loss check 443081 passed 9 tests; controller check 443092 passed 8 tests. CPU preparation 443132 passed in 20 allocated seconds. Actual Qwen 443135 and Cosmos 443134 profiles passed at 131 GPU-seconds each, using the fixed unfitted core and prescribed inputs. Total: 262 GPU-seconds.

Independent precision report 443156 passed. Earlier pre-last writes change later native logits with fixed subsequent tokens and writes; restoring only the affected final-layer global K/V slot removes that effect. Earlier-delta gradients are nonzero for pre-last and zero for post-last; local/future gradients are zero. All 3,524 native-shaped head rows pass with TV 0 and exact top1. Execution totals 96 VLM calls, 44 vision calls, four standalone blocks and 104 extra head projections.

Report 443139 remains a failed 36-second CPU allocation. Diagnostic 443153 took 6 CPU-seconds and isolated six one-ULP descriptive RMS discrepancies. The new reporter permits at most two ULPs only for that field; all native thresholds and other fields remain unchanged. Cached/full numerical failures remain 14/328 for Qwen and 74/328 for Cosmos, including two Cosmos argmax mismatches.

[Results note](../../../docs/paper/NATIVE_AGGREGATION_LEARNED_MEMORY_SOFTWARE_RESULTS.md) · [Passed report](software/report_precision_443156/summary.json) · [Analysis](software/report_precision_443156/analysis.json) · [Retained failed report](software/report_443139/failure.json) · [Diagnosis](software/report_diagnosis_443153/summary.json) · [Qwen profile](software/profile_443135_qwen/summary.json) · [Cosmos profile](software/profile_443134_cosmos/summary.json) · [CPU plan](software/check_443132/plan.json) · [Mechanism proposal](../../../docs/paper/NATIVE_AGGREGATION_NATIVE_KV_WRITE_PROPOSAL.md).

The four public native-generation trajectories produced 10 tokens total; the public forward API was not executed. No long-reasoning feasibility, fitted aggregation, reasoning accuracy or useful composition has been demonstrated. The separate learned-selection identity-join efficacy campaign failed and is fully reported.

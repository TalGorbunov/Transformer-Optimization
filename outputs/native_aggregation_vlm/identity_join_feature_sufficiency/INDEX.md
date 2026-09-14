# Training-only feature accessibility diagnostic

**Completed fixed ridge assay and independently verified label-report repair.** CPU job 443231 used 9 allocated seconds; report repair 443240 used 1 second. No GPU, model or head calls occurred, and no native training is released.

All 9,007 training-local empty-prefix features were retained. Odd Steps provide 4,472 fitting rows and even Steps 4,535 evaluation rows, with all 648 question–person–room cells on each side. Held person accuracy is 4,535/4,535; room and jointly correct accuracy are 4,379/4,535 (96.56%). The question-frequency control is jointly correct on 95/4,535. The strict accessibility criterion fails. No exact local-state or scene-multiset collision was found; this fixed readout's errors do not establish missing information or an information ceiling.

The original numeric table serialization overwrote person/room group labels. Repair 443240 reconstructs all labels from saved predictions, publishes separate labels/metrics objects and exactly reproduces all original counts, ties, controls and decisions. It executes no fit or tensor deserialization. The original results remain unchanged.

[Results note](../../../docs/paper/NATIVE_AGGREGATION_FEATURE_ACCESSIBILITY_RESULTS.md) · [Original summary](443231/summary.json) · [Original predictions and input bindings](443231/resolved_inputs.json) · [Passed repair](label_report_443240/summary.json) · [Corrected labelled metrics](label_report_443240/labelled_metrics.json) · [Reporting defect record](label_report_443240/original_report_error.json) · [Execution ledger](execution.json).

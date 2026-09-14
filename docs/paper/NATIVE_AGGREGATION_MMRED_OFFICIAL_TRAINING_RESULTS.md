# Original MMReD: completed fixed training recipe

All three models completed the same 4,000 original questions, three epochs, 12,000 scene forward/backward passes and 1,500 optimizer updates. Each independent computational audit passed. The fixed 100-question training diagnostic is descriptive; it is neither validation accuracy nor extrapolation evidence.

| Arm | Complete correct training answers | Valid JSON | GPU allocation seconds | Independent CPU audit seconds |
|---|---:|---:|---:|---:|
| Ordinary images |94/100|100/100|2790|47|
| Normalized 32-slot memory |82/100|100/100|2226|49|
| Mass 32-slot memory |75/100|100/100|2223|53|

The ordinary run is 444013_0 and its independent audit 444016; normalized and mass use the next two array indices and audits 444017/444018. Original 4,000-world feature harvesting cost 1,419 GPU allocation seconds and is shared across the fits. All totals are instrumented experiments, not production latency benchmarks. See the [bound execution records](../../outputs/native_aggregation_vlm/mmred_official_native_training/execution.json).

A separate posthoc [scalar-only report](../../outputs/native_aggregation_vlm/mmred_official_training_dynamics/report_444092/REPORT.md) used one CPU allocation second. It read all 36,000 logged scene losses and 4,500 preclip total gradient norms, without opening tensors or running models. In the second half of epoch 3, task-balanced N16 full-JSON/EOS mean CE was 0.08464 for ordinary, 0.23651 for normalized and 0.23049 for mass. In both memory arms the corresponding loss declined relative to the first half of that epoch (0.25020 and 0.25448). The two halves contain different shuffled examples. These observations do not establish convergence, information loss or a representational capacity bound, and they do not authorize retuning the fixed models.

The full original 400-validation/600-test comparison remains required and unchanged. Its first three scheduled timing cases per arm ran successfully, but conservative runtime projections exceeded the initial 10,800-second caps. Original GPU array 444052 is preserved as a resource-gate failure (98 GPU allocation seconds total). A separately frozen resource-only continuation, GPU array444105, retains those three answers per arm and generates only the remaining 997; its independent reports and paired statistics are queued. No checkpoints, inputs, predictions, accuracy criteria or statistical rules change.

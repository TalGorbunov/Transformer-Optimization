# V14 final artifact

Report and checkpoint verification: **True**.
Reported both-seed primary/practical: **True / False**.
Accepted vision milestone: **False**.

![Existing per-cell native accuracy](native_vision_v14.png)

[PDF figure](native_vision_v14.pdf) · [Acceptance and provenance](final_acceptance.json)

[Independent report](/mnt/home/gabriele.serussi/gnn_transformer/Transformer-Optimization/outputs/native_aggregation_vlm/v14/report_442582/REPORT.md) · [Selected-checkpoint audit](/mnt/home/gabriele.serussi/gnn_transformer/Transformer-Optimization/outputs/native_aggregation_vlm/v14/checkpoint_audit/audit_442625/summary.json)

Figures copy the reported metrics; labels retain every denominator. Descriptive cache failures remain recorded.
V14 allocated compute: 3197 GPU-seconds / 16200; all failed attempts remain in the accounting.
Both arms use the same 1,060,224 parameters and training-only null calibration; inference uses N+1 streams without reference rows. All tested answer values occur in adaptation. This artifact does not establish unseen-answer transfer or reasoning composition.

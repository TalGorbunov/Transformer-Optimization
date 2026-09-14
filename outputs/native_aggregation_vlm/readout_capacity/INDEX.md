# Synthetic perfect-evidence native readout capacity

Initial diagnostic complete: CPU441719 and GPU441720,29GPU-seconds. BOTH500-step fit-capacity criteria failed: affine13/72,SiLU62/72 fit examples. Correct K was supplied as z=K e1: these are oracle first-token scores, not MMReD vision accuracy. Native cached-head replay was exact on32serial prompts. Native hidden/norm/head tensors areFP16 under unchanged NF4/bf16-compute backend. The fixed optimizer showed substantial clipping/oscillation; failure is not a capacity impossibility result.

[Report](run_441720/REPORT.md) · [Summary](run_441720/summary.json) · [Predictions](run_441720/predictions.json) · [Execution/cost](execution.json) · [CPU plan](check_441719/plan.json).

Separate total0.2GPUh envelope. Any further diagnostic requires a new preregistration and preserves this failed result.

[Fixed optimizer sequel](decay/INDEX.md) nowpassedfit72/72forbothreadouts; reusedheldglobal-promptchanges remainimperfect. Total71GPU-seconds.

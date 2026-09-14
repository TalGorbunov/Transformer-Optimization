# V17: verified native local-readout audit

The frozen native head correctly classifies all **9,512 training identities** and **13,056 V15 local execution rows**. V15 represents 1,574 distinct image/question identities, with 226 shared with training; repeated model/mode executions are not independent observations. Canonical Step labels above 16 also have no local errors.

All 560 native replay rows passed the registered numerical checks. The successful job used 432 head calls with zero VLM or vision calls. Total campaign cost was **75 GPU-seconds and 440 head calls**, including the preserved eight-call timing failure.

- [Results and limitations](../../../docs/paper/NATIVE_AGGREGATION_VISION_V17_RESULTS.md)
- [Independent analysis](local_readout/report_442798/analysis.json) and [summary](local_readout/report_442798/summary.json)
- [Successful GPU run](local_readout/run_442796/summary.json)
- [Frozen retry plan](local_readout/check_442792/plan.json)
- [Preserved timing failure and amendment](timing_failure_442787/)
- [Allocation and execution record](execution.json)

This supports the predetermined semantic-gate experiment. No aggregate answers were generated, no model was fitted, and neither the research objective nor reasoning composition is established.

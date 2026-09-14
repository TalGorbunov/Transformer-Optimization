# Independent complete local prompt batching

Computational integrity: PASS. Frozen numerical gate: PASS.

64 fixed training pairs;153 batched responses plus4 fresh serial references. No aggregate answer is produced.

| Case | Rows | Forward seconds | Max conditional p1 difference | Max numeric mass difference | Gate |
|---|---:|---:|---:|---:|---|
| fresh_serial_0 | 1 | 0.5841 | 0 | 0 | PASS |
| fresh_serial_1 | 1 | 0.0851 | 0 | 0 | PASS |
| fresh_serial_2 | 1 | 0.0858 | 0 | 0 | PASS |
| fresh_serial_3 | 1 | 0.0831 | 0 | 0 | PASS |
| batch_1 | 1 | 0.0827 | 0 | 0 | PASS |
| batch_8 | 8 | 0.2219 | 0.00609291 | 2.97541e-05 | PASS |
| batch_16 | 16 | 0.3803 | 0.00832247 | 1.83995e-05 | PASS |
| batch_64 | 64 | 1.3058 | 0.0147994 | 0.00012734 | PASS |
| batch_64_reverse | 64 | 1.2836 | 0.0147994 | 0.00012734 | PASS |

Descriptive fresh-serial overlap bounds: PASS (17 comparisons).

These are repeated training-only local judgments, not aggregation accuracy or unseen-data generalization.
A single native call processes independent complete batch rows; it does not fuse them into one answer.
No generation, cached decoding, hidden-stream fencing or learned aggregation branch was tested.
Probability tolerances are frozen. Common logit shifts are reported and need not change probabilities.
Fresh serial references cover only the first four fixed pairs, one per category; all64 compare to the archived teacher. Fresh-overlap bounds and audit truth are descriptive, not extra eligibility criteria.
Numerical failures are retained without threshold changes or selection.
Forward timing includes the actual-position capture hook; CPU staging and model loading are reported separately.

# Independent ordinary joint LoRA audit

CPU norm/head TV must be at most .02. CPU/GPU argmax differences are descriptive; all efficacy uses actual GPU-generated IDs. Exact GPU zero-adapter and serialization parity remain mandatory. Earlier reports retain their original gates.

Numerical audit: PASS. CPU head calls: 40; selected query rows: 80; maximum TV: 1.2525617595189485e-06.
CPU/GPU argmax differences: 0. Every scheduled numerical audit was collected before this decision.

Original profile cost forecast: 28545.207520 seconds; passed=False.
Separate prospective main accounting: 3029.584956 seconds; passed=True.
The new accounting was defined after profile timings and before any main fit: the two instrumented calls are priced once in setup, all14 ordinary calls determine per-example cost. No additional model call was run. A separate main source release is still required.

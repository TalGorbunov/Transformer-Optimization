# Learned selection on the MMReD-derived identity join

**Completed: primary and practical criteria failed.** Independent report443105 passed in131 allocated CPU-seconds, verifying all six final fits and native outcomes. Plot443108 passed in2seconds and was visually inspected. Total campaign cost:11,508GPU-seconds, including all failed attempts.

| Method | Seed | Seen N32 | Seen N64 | Held N32 | Held N64 |
|---|---:|---:|---:|---:|---:|
| clip | 22 | 12/108 | 12/108 | 12/108 | 12/108 |
| sigmoid | 22 | 17/108 | 18/108 | 15/108 | 17/108 |
| softmax | 22 | 22/108 | 21/108 | 20/108 | 15/108 |
| clip | 23 | 12/108 | 12/108 | 12/108 | 12/108 |
| sigmoid | 23 | 16/108 | 16/108 | 15/108 | 17/108 |
| softmax | 23 | 31/108 | 30/108 | 27/108 | 27/108 |


Development at N16 was also weak: clip12/108 both seeds, sigmoid15/108 both, softmax22/108 and26/108. This is not solely an extrapolation failure. Clipped gates close completely in inspected test-prefix captures; a complete descriptive diagnosis is next. No fitted model supports general aggregation or reasoning composition.

[Verified report](reporting/report_443105/REPORT.md) · [Analysis](reporting/report_443105/analysis.json) · [Every outcome](reporting/report_443105/outcomes.json) · [All prefix geometry](reporting/report_443105/geometry.json) · [Figure](reporting/report_443105/plots/plot_443108/accuracy_by_length.png) · [Execution ledger](execution.json).

The conditional ordinary-joint and multiple-request efficacy studies remain unreleased because the primitive-accuracy gate failed. Any subsequent exploratory repair must be prospectively specified and independently confirmed; it cannot alter this outcome. Frozen data, source copies, checkpoints and failed V1 profiles remain intact. Native-memory software is a separate study and does not rehabilitate this efficacy failure.

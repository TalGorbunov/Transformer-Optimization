# V18: verified native counting extrapolation

**Both seeds pass the predefined primary and practical criteria.** With training lengths at most16, the native semantic gate reaches132/136 (97.1%) whole answers at32frames and131/136 (96.3%) at64frames in each seed. The matched all-open models score0/136 at both lengths, after allfourmodels achieved64/64 on16-frame development. All answer values0–16 occur in training.

| Mode | Seed | N32 | N64 | N64 K9–16 |
|---|---:|---:|---:|---:|
| Native gate | 20 | 132/136 | 131/136 | 59/64 |
| Native gate | 21 | 132/136 | 131/136 | 59/64 |
| All open | 20 | 0/136 | 0/136 | 0/64 |
| All open | 21 | 0/136 | 0/136 | 0/64 |

The fixed native0/1probability-gap gates all96coordinates of a bounded learned payload. The model generates its answer with the unchanged native vocabulary/head and ordinary EOS; no scalar tally supplies an answer. Both modes have1,041,600trainableparameters and identical nativeprobe work. Training uses the fixed final4,590updates, with no checkpoint selection from outcomes.

The all-case saved-state diagnosis confirms zero false-positive/negative gates and exactly zero negative messages in the gated branch. All first numeral tokens are correct; the remaining9errors/model are valid one-count underestimates on multi-digit answers. Paired positive-payload drift is tiny compared with gate-weight drift. This does not establish that content-dependent vector values are necessary; the completed [payload analysis](payload_rank/run_442933/REPORT.md) finds an almost constant positive vector at the original query, consistent with a weighted count.

- [Readable result and interpretation](../../../docs/paper/NATIVE_AGGREGATION_VISION_V18_RESULTS.md)
- [Independent report and all metrics](report_442913/REPORT.md)
- [Full analysis and paired-family intervals](report_442913/analysis.json)
- [All-case saved-state diagnosis](diagnosis/diagnosis_442927/REPORT.md)
- [PNG/PDF figures, visually inspected](plots/plot_442918/INDEX.md)
- [Fixed design and criteria](../../../docs/paper/NATIVE_AGGREGATION_VISION_V18_DESIGN.md)
- [Passed resource release](release_442888/release.json)
- [Execution record](execution.json)

Total V18 cost:3,681GPU-seconds (about1.02GPU-hours), including software and training profiles. Allfourmains completed in14–15minutes, below45-minute caps. Complete-context exclusions cover30prior manifests; individual visual atoms can recur.

This establishes a useful counting length-extrapolation result. General vector aggregation, unseen adaptation answers, compute-normalized bandwidth and reasoning composition remain unestablished. The separate Cosmos local-prompt audit passed on80oldlocalexamples; [its scope and results](../reasoning_semantic_gate/report_442926/REPORT.md).

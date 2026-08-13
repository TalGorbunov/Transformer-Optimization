# outputs/pipeline — end-to-end tally pipeline (Ch. 7)

Canonical runs of the constructive decide-per-frame-then-reduce pipeline. All entries
were logged pre-refactor — the detailed records live in
`docs/archive/RESULTS_pre_fencing.md` (search the run-dir name). New runs use
`scripts/e2e_pipeline.py` and should be logged in the live `RESULTS.md`, then added here.

| experiment | canonical run | headline |
|---|---|---|
| **retrieve-v2 (the deployable config)** | `e2e_retrieve2_N{32,64,128}/` (+`_s4`,`_s5` seed reps) | exact ≈ frozen-crushing at N=32→128 with cost ∝ evidence, not N (archive entry ~L3776) |
| retrieve v1 | `e2e_retrieve_N{32,64,128}/` | superseded by v2 |
| chunked | `e2e_tally_chunked/` | the k-tax baseline mode |
| adaptive / two-stage | `e2e_{adaptive,twostage}_N{32,64,128}/` | mode comparison vs retrieve-v2 (archive ~L3971) |
| task algebra (rooms/cooc) | `e2e_algebra_{rooms,cooc}[_N{32,128}]/` | pipeline generalizes across task predicates (archive ~L3947) |
| cross-family | `e2e_internvl/`, `e2e_herbench/` | InternVL port + HERBench leg (archive ~L3805) |

Per-run artifacts: `<run>/<ts>/{report.txt, results.csv, rows.json, config.json}`.

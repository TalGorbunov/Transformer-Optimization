# outputs/qgate — INDEX (QGATE campaign, 2026-09-15→16; brief: CAMPAIGN_BRIEF.md, log: STATE.md)

The question-agnostic gate as a (question × block) function over question-BLIND (qlast-once) block encodings — the cacheable
per-frame-encoding story. Scripts now under `legacy/v1/scripts/qgate/` (`qgate_common.py`, `capture_qb.py`, `fit_qb.py`, `train_neutral.py`)
+ `legacy/v1/scripts/selfgate/probe_headscan.py --layout`. Substrates: `outputs/sparse/layout/qlast_once_ep5_gated/20260915_203606_gated/adapter`
(150433) and P1b. Regime: park data, count/majority/exists templates. C1 (oracle rows) = the layout twins in `outputs/sparse/layout/`.
Verdict: dead at the linear level — the internal selector is a verdict-reader and blind block states do not linearly expose occupancy.

| experiment | canonical run | headline |
|---|---|---|
| A1 headscan under qlast, P1b arm | `a1/p1b_{count,majority}_N{8,32,128}/` (150577, 90/cell) | best head 0.56–0.79 (vs 1.000 with replicas, `outputs/selfgate/g0a/`) |
| A1 headscan under qlast, qlast-gated adapter | `a1/qlastg_{count,majority}_N{8,32,128}/` (150719, N-free) | **H-QA1 MISSED**: best head 0.53–0.68, no head ≥0.98 in any cell — B2 dead on qlast |
| B-capture v1 | `bcap/` (150720) | UNUSABLE (text-only q_vec crash → 0 q_vecs; masked-index bug); superseded by bcap2 |
| B-capture v2 + A2/D3/B1 fits (span-mean L20) | `bcap2/fit_report.json` (150826) | **A2 MET** (state-only = base rate 0.802/0.519); D3 0.766/0.747 with a perfect symbolic selector; **B1 MISSED** (held-out 0.802/0.520, LOO 0.204/0.523) |
| B-capture v3, pooling fair-shake (mean/last/max) | `bcap3/fit_report.json` (150848) | D3 0.733/0.697, B1 0.801/0.532 — extraction wall is pooling-independent |
| D1 verdict-reader test v1 | `d1/*/runner-150827.log` (150827) | died on a tokenizer boundary merge (".You"), zero samples — silent-continue instrument note |
| D1b neutral-filler replicas | `d1/p1b_neutral_N{8,32}/` (150841) | P1b selector collapses to 0.78 / 0.62 (L24h20 off the board) — **verdict-reader proven by intervention** |
| E1 generic-filler slot capture + fits (Option 1 viability) | `e1cap/fit_report.json` (150851; layout=replica-neutral) | A2 0.802/0.519, D3 0.767/0.748, B1 0.802/0.520 — the cacheable slot exposes no more than raw blind states; Option 1 dead at the linear level |
| O1 neutral-replica trainer | `o1_train/20260916_101845_gated/` (150856) | **instrument failure, no result**: builder rejects `declare_n`, all 651 train samples skipped ×5 ep, val/test/exams n=0 (skipped=150 ×4); adapter = untrained init |
| B2 · C2 deployed ladder · C3 virtual-N learned gate | — (not run) | B2 dead after A1; C2 had no gate to deploy after B1; C3 deferred |

No adapter promoted. RESULTS entry logged 2026-09-21. Note for the rewrite: the untested variant is the question-FIRST layout
(question in the shared prefix, blocks question-conditioned without replicas) with the `<|vision_end|>` slot — see the plan.

> Copied from `~/.claude/plans/ok-lets-take-a-stateful-axolotl.md` on 2026-10-06 (approved by Tal the same day); this repo copy is the live one. Campaign docs: `outputs/scaleup/{CAMPAIGN_BRIEF,STATE,INDEX}.md` (section S1u). **Wave 1 results (2026-10-07): `outputs/scaleup/STATE.md` entry 2026-10-07 and `outputs/scaleup/herbench/qwen2.5-vl-7b/s1u_unit/TABLE*.md`** — perception walls SVA / MEGL / MPDR / AGLT / AGAR; targets TSO / FOM / FAM / AC / ASII; resolution ≈ 0; clips task-dependent; fence cost 0.26–0.42; the model's own vote cannot gate at long N.

# UNIT baseline: how well does the frozen model perceive one evidence unit by itself? (task- and model-generic)

Written for Tal, 2026-10-06 (branch `theory-2`). Decisions taken via AskUserQuestion this session:
first wave = **Qwen2.5-VL-7B × HERBench lite_v2 + the MMReD anchor**; **unfenced only** (no clip-blocks in
the fence yet); **video-mode input deferred** until clips are shown to matter. Working agreement
unchanged: mini-plan per module → OK → Claude drafts → Tal reads the diff → tests → commit; every GPU
submission through the `sbatch-submit` loop.

## Context

The method (fence → gate → read) treats one block as a complete fact. On MMReD a 512 px frame is one.
On real video it is not guaranteed: the evidence may be too small at the resolution shown (WildTrack
person ≈ 15×40 px at 512), an action spans time (HD-EPIC AC / FAM / FOM), or the answer needs several
units jointly (TSO / SVA ordering, MEGL / MPDR multi-person). Before more datasets and backbones are
added, every (task, model) cell needs a **perception baseline**: handed exactly the right unit, does the
frozen model (a) extract the fact the read needs and (b) recognise the unit as relevant? Without it a
failed method cell cannot be attributed (perception wall vs retrieval vs aggregation), and the SCALEUP
stop rule ("S2 AUC ≈ chance → perception wall") was already shown to misfire: the S2 linear probe is
near chance on AGAR / AGBI / AGLT because those tasks are **text-answerable** (1.00 / 0.92 / 0.67 with
no frame), not because of perception.

What the record already says (all in `outputs/scaleup/STATE.md`, the stage-2b fit CSVs under
`outputs/scaleup/herbench/qwen2.5-vl-7b/s2_fit/readouts_*.csv`, and `outputs/herbench_retrieve_opt/`):

- Single-frame unit, in-fence, Qwen, 1 MP, HERBench: the model's own yes/no judge is the best readout on
  AC / TSO / SVA (AUC 0.79–0.84 / 0.81 / 0.81) but precision at 95 % specificity is 0.07–0.19 on the
  action tasks; FAM / FOM are beaten by frame position alone (evidence clusters in time); the smoke
  judge answered "no" to all 80 frames. These fits were never written up.
- Clip unit (July, HD-EPIC AC only, 5 separate images spanning ±δ): the same per-unit yes/no rises from
  0.835 to 0.857 pooled (open 0.92 → 0.94, close 0.84 → 0.92), peak at δ = 1 s, resolution ≈ flat.
- Resolution (Qwen, HERBench): native vs 512 px adds 0–8 points to the per-block probe on person
  tasks and nothing to unfenced accuracy; 1 MP is the operating point.
- Oracle gate under the fence sits BELOW plain on HERBench (0.41–0.51 vs 0.52–0.54) — unknown whether
  that is the fence (position reset, no joint read) or the annotated units being insufficient.
- No text-only arm on all rows, no evidence-only arm, no clip unit, no `--gate blind` exist in the code.

## The baseline: three regimes, one unit sweep, all through the model's own output

Everything below uses only what every benchmark has (question, options / gold, evidence timestamps
or MMReD's `evidence_frames`) and only what every model has (its generated answer, or the first-token
logits over a two-word vocabulary). No hidden states, no probes, no task-specific per-frame labels.
That is what makes it task- and model-generic; it would even run on API models.

| regime | prompt | measures | reads as |
|---|---|---|---|
| **T** text-only | question (+ options), no image | accuracy per task | floor; flags text-answerable tasks |
| **E(U)** evidence-only | the k annotated evidence units ONLY, no fillers, temporal order, the benchmark's official layout (frames then question) | accuracy per task, **stratified by k** (k = 1 rows = pure perception; k > 1 = perception + joint read without distractors) | perfect retrieval + the model's own joint read; `P* = max_U E(U)` = the perception ceiling |
| **D(U)** per-unit detection | ONE unit + question + "Does this clip contain evidence needed to answer the question above? Answer yes or no." (reuse the stage-2b wording; "frame" for m = 1) | first-token log P(yes) − log P(no) → AUROC, recall @ spec 95 / 99, mean P(yes) on positives; **hard** negatives (same grid centred just outside the evidence interval ± margin) and **easy** negatives (≥ 5 s from every interval) reported separately | can the model recognise the unit as relevant on its own (position-free by construction) |

Unit definition U = (temporal extent, resolution mode), nested so one extraction serves every variant:

- temporal: `frame` (today's unit: point t + 0.3 s / range midpoint) · `clip3_d1` (t − 1, t, t + 1) ·
  `clip5_d1` (0.5 s step, ±1 s) · `clip5_d2` (1 s step, ±2 s) · `clip9_d2` (0.5 s step, ±2 s). For
  range evidence [a, b]: `frame` = midpoint, `clip3` = {a, mid, b}, `clip5` / `clip9` = evenly spaced in
  [a, b] (step ≥ 0.5 s, so short shots get fewer frames). Clip frames are fed as **separate images** (the
  model-generic form; the July recipe). Stored grid per unit: 9 frames, 0.5 s step, centred on t (points)
  or evenly spanning [a, b] (ranges) → every variant is a subset.
- resolution: two modes per model, defined by the model's OWN processor: `lo` = the single-tile budget
  that MMReD frames get (Qwen `max_pixels` 151,200 ≈ 512 px; InternVL one 448 tile; Gemma one 896 tile)
  and `hi` = the model's own high-resolution mechanism (Qwen `max_pixels` 1,003,520 — the established
  operating point; InternVL `crop_to_patches=True` dynamic tiling; Gemma pan-and-scan). Specs gain
  `lores_kwargs` / `hires_kwargs`; frames are stored native and the processor does the rest.
- first-wave cells: E and D at `frame·lo`, `frame·hi`, `clip3_d1·hi`, `clip5_d1·hi`, `clip5_d2·hi`,
  `clip9_d2·hi` (6 units) + T. Prompt cap: total frames per E prompt ≤ 32 at `hi` (≈ 39 K tokens at
  1 MP); rows with k·m > 32 degrade to the largest nested m that fits, recorded in a `m_eff` column.

The table every (dataset, task, model) cell gets, joined with cells that already exist:

```
T | E(frame·lo) | E(frame·hi) | E(best clip) | P* | O(N) oracle gate, fenced, 1-frame unit (exists) | F(N) official uniform (exists)
perception wall  = P* − T            (≈ 0 → the model cannot read the unit even when handed it)
resolution gain  = E(frame·hi) − E(frame·lo)
temporal gain    = E(best clip) − E(frame·hi)
fence cost       = E(frame·hi) − O(N)   (same unit; isolates the position reset + no joint read)
retrieval room   = P* − F(N)            (what a perfect gate could buy on top of the official row)
gate on its own  = D(U*) recall @ spec 99  (N = 256 has 1–4 % positives)
```

Decision rules for SCALEUP (replace the S2-AUC stop rule):

1. `P* − T` not above the 95 % CI → **perception wall** for that task × model: no method cell there; the
   row is reported as such. Text-answerable tasks (T ≥ 0.6) are reported as such and excluded from the
   evidence-task set regardless.
2. Otherwise `U*` = the smallest unit within the CI of `P*`; it becomes the block definition for that
   task × model in the later fenced cells (clip-blocks in the fence = the first METHOD cell after this,
   explicitly out of scope here).
3. `D(U*)` says whether the model's own vote can gate at long N or a trained gate is required.

Pre-registered expectations (bands go in STATE.md before the wave runs): HERBench / Qwen T ≈ 1.0 /
0.9 / 0.7 / 0.7 on AGAR / AGBI / AGLT / ASII and ≤ 0.2 on AC (k = 1 rows), FAM, FOM, TSO, SVA; E(frame·hi)
on AC k = 1 ≥ 0.6, FAM / FOM 0.5–0.7, TSO / SVA ≥ 0.7 (distinct shots; the planted probe was 0.9) and far
above O(N) there (ordering is lost under the fence: TSO 0.62 → 0.30 at 512 px); clips add 0.03–0.10 on
the HD-EPIC action tasks and nothing on trailers; `lo` → `hi` adds ≤ 0.05 (Qwen record). D(frame·hi) AUROC
≈ 0.8 on AC (the in-fence judge gave 0.79–0.84), clips → ≈ 0.9 (July), hard negatives well below easy.
MMReD anchor: T ≈ label prior, E(frame) high for k ≤ 2 and falling with k on counting (the gap test:
frozen counting never exceeds 2) → the anchor's wall is aggregation, not perception.

## Implementation (small, on the existing seams)

1. **`experiments/prepare_video.py --stage units`** (new stage; reuses `_open_video`, `grab_frames`, `_save`,
   `experiments/prepare_video.py:245-308`). For the pinned rows only (`--qids-file`): per evidence unit a
   9-frame 0.5 s grid (points: centred on t + 0.3; ranges: evenly spanning [a, b], step ≥ 0.5 s) +
   per unit one **hard** negative (same grid centred at boundary + margin + 2 s, on the side with room)
   and one **easy** negative (seeded random centre ≥ 5 s outside every interval). Written to
   `units_native/<qid>/u<i>_<pos|hard|easy>/frame_%02d.jpg` + `times.json` (centre, grid times, kind,
   source interval). One decode pass per video, sorted targets, streamed to disk, `4h_0g` (16 G cap).
   Size ≈ 600 rows × 2.5 units × 3 × 9 frames × 0.25 MB ≈ 10 GB on `/rg` (562 G free today).
2. **`core/data/videoqa.py`**: protocol `evidence` with unit params (`unit ∈ {frame, clip3_d1, clip5_d1,
   clip5_d2, clip9_d2}`): `Sample.frame_paths` = the chosen subset of each positive unit's grid,
   concatenated in temporal order; `meta["units"]` = index ranges, `meta["k"]`, `meta["m_eff"]`;
   `evidence` = all frames. Rows with `evidence_kind ∈ {all, none}` skipped and counted (as today).
   A `units(root, sample, unit)` method returns the per-unit records (pos / hard / easy) for D.
   Split name `evidence_<unit>_test`; `N_FRAMES` check bypassed for this protocol.
3. **`core/data/mmred.py`**: protocol `evidence` = the test row's frames restricted to `evidence_frames`
   (`core/mmred.py:301-361`); `units()` = every frame of the row with label from `evidence_frames`
   (negatives = the row's non-evidence frames). Unit is always one 512 px frame.
4. **`experiments/evaluate.py`**: `--no-frames` (regime T: `build_messages([], …)` already yields
   `[question]`; the unfenced `model.generate` path handles it) and the `evidence` protocol through the
   existing unfenced path (`experiments/evaluate.py:267-290`); `report_extras` gains accuracy by k
   (k = 1 / 2–4 / ≥ 5) and `m_eff`. Run-dir suffix carries the unit and the resolution mode.
5. **`experiments/unit_judge.py`** (new, the D regime; model-generic): one dense prompt per unit =
   `build_messages(unit_frames, question_text + "\n" + JUDGE_TEXT, layout="paper", system_prompt=spec.system_prompt)`
   + the spec's `assistant_prefix`; one forward, first-token logits at `JUDGE_WORDS`
   (`experiments/gate_capture.py:55`); writes `units.csv` (qid, qtype, unit, kind, k, score, P(yes)) and
   `report.txt` with AUROC / recall @ spec 95 / 99 / mean P(yes), per task, hard vs easy vs pooled
   (metric code reused from `experiments/gate_fit.py:108-137`). Works on all three backbones (no
   fastpath, so Gemma is included).
6. **`core/backbones/*.py`**: `lores_kwargs` / `hires_kwargs` per spec (Qwen: `max_pixels` 151,200 /
   1,003,520 via the existing `set_max_pixels`, `core/backbones/base.py:101-113`; InternVL:
   `crop_to_patches` False / True; Gemma: `do_pan_and_scan` False / True — the InternVL / Gemma kwargs are
   only pinned by the tokenizer test now, used in wave 3). `--res {lo,hi}` on both scripts replaces
   ad-hoc `--max-pixels` for these cells (the old flag stays).
7. **`sbatch/unit_baseline.sbatch`** (+ `sbatch/unit_submit.sh <dataset> <backbone>`), run dirs
   `outputs/scaleup/<dataset>/<backbone>/s1u_unit/{T,E_<unit>_<res>,D_<unit>_<res>}/<stamp>_<job>/`,
   ledger `outputs/scaleup/jobs.tsv`, INDEX / STATE rows.
8. **Tests (CPU)**: `tests/test_data_videoqa.py` + evidence protocol (nested grids, temporal order, k,
   m_eff cap, range vs point rules, hard negative outside interval + margin, easy ≥ 5 s); `tests/test_data.py`
   + MMReD evidence protocol (frames = `evidence_frames`); `tests/test_prompt.py` + zero-frame message.

Not touched: `core/fence.py`, `core/fastpath.py`, any method code. The stale `uniform_labels` ±2 s
rule (`core/data/videoqa.py:183`) is a separate fix and is not part of this campaign.

## Waves (each through `sbatch-submit`: preflight → dry-run → `--limit 5` smoke → submit → verify)

| wave | cells | rows | cost (Qwen basis) |
|---|---|---|---|
| W0 data | `prepare_video.py --stage units` HERBench (pinned 50 rows/task = the S1–S3 set, `--qids-file`) | ~600 rows, ~4,500 units | CPU `4h_0g`, 1–3 h |
| W1 HERBench × Qwen | T; E × 6 units; D × 6 units (pos + hard + easy) | 600 / 600 / ~4,500 units | E ≈ 3–4 GPU-h (short prompts); D ≈ 10–15 GPU-h (27 K forwards of 1–6 K tokens); T < 0.5 h |
| W1 MMReD anchor × Qwen | T; E(frame); D(frame) on the standard 1,200 test rows at N = 8 and 16 | 1,200 × 2 | ≈ 2 GPU-h |
| W2 (after W1) | MINERVA × Qwen at the units W1 kept (regex evidence → lower bound, parsed-evidence rows only) | ~470 rows | ≈ 10 GPU-h |
| W3 (after W1) | InternVL3.5-8B (non-thinking) + Gemma-3-12B × HERBench + MMReD at `frame·lo`, `frame·hi`, `U*` | | ≈ 15 GPU-h each |

Partitions per the skill's FREE table (single-GPU cells: `12h_4g → 24h_1g → 24h_4g`; D on `hi` with 9-frame
clips ≈ 11 K tokens fits an L40S 48 G in 4-bit). Nothing is submitted before Tal OKs the wave.

## Verification

- CPU: `python tests/test_data_videoqa.py tests/test_data.py tests/test_prompt.py` green after each module;
  all 12 existing test files unchanged and green.
- Prep: `verify_report_units.txt` — units per task, grid frame count = 9 (or the range-capped count),
  hard negatives outside every interval ± margin, easy ≥ 5 s, per-task positive / negative counts.
- Smoke (`--limit 5`, 1 row per task): T parses on all rows; E(frame·hi) reproduces the existing 1-frame
  oracle-gate rows' token counts per frame (≈ 1,227 at 1 MP); D scores finite, hard / easy both present.
- Anchor check: E(frame·hi) on HERBench rows whose evidence set is empty under `uniform` N = 8 must equal
  the STATE 2026-09-30 text-only subset numbers (AGAR 1.00 n = 5 … AC 0.04 n = 46) when run as T.
- Report: one table per dataset × backbone with the seven columns above and the three derived gaps,
  per task and by k; INDEX.md row per cell; STATE.md entry with the pre-registered bands filled in;
  RESULTS.md only on "log this".

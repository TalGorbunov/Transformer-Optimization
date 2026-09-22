# Rewrite execution plan: port `core/` + `experiments/` through the three GPU anchor gates

> Copied from `~/.claude/plans/ok-now-lets-plan-wise-bonbon.md` on 2026-09-21; this repo copy is the live one. Update the "Sequence at a glance" table as steps complete.

Written for Tal, 2026-09-21. Companion to the scope plan `ok-i-want-to-nifty-brook.md` (rev 2),
which stays the record of *what* and *why*; this file is the record of *who does what, in what
order, and how each step is proven*. Scope chosen today: **port + acceptance gates**. Ownership
chosen today: **Claude drafts every `core/` body after a per-module plan; Tal reviews every diff
and runs the tests.** The paper's experiment set (E1/E2/E3/E6/E7/P4/E9) gets its own plan once the
gates pass.

## Context

The repo is mid-rewrite on branch `rewrite`. Phases 0 (disk), 1 (freeze `legacy/v1`) and 4 (docs)
of the scope plan are done. Phase 2 (`core/`) is a skeleton: six modules, every body outside
`constants.py` raises `NotImplementedError`, and five test files (34 tests) are already written
against the unimplemented API, each pinning a legacy-parity or benchmark-parity contract. Phase 3
(`experiments/`, `sbatch/` wrappers) has not started. The three runtime acceptance gates that prove
the port are in the scope plan at lines 270-273 and are restated below.

What this plan has to get right: the mask and prompt are the method, so the parity tests are the
review, not a formality; the official benchmark's prompt and parser must be byte-faithful; and the
first GPU job must reproduce a logged number before any new number is produced.

## Working agreement (applies to every step below)

The loop per module, agreed 2026-09-21:

1. Claude posts a **mini-plan** for the module in chat: which twin lines it ports, what changes
   from the twin and why, what each test pins, anything the docstring leaves open. Two or three
   paragraphs, no code.
2. Tal OKs or edits the mini-plan.
3. Claude implements, runs `python tests/test_<module>.py`, and reports the test output verbatim.
4. Tal reads the diff (`git diff core/<module>.py`) and asks "why" on anything unclear. Nothing is
   committed until Tal has read it.
5. Tal commits.

GPU jobs follow the `sbatch-submit` skill loop (plan → OK → preflight → dry-run → submit →
verify → collect). Every gate run gets a run dir under `outputs/port/<gate>/` and a line in
`outputs/port/INDEX.md`. RESULTS.md is appended only when Tal says "log this".

## State on 2026-09-21 (from the inventory)

| Area | State |
|---|---|
| `core/constants.py` | complete, **but `CHARS` is alphabetical; upstream order is Sandra, Mary, John, Daniel, Michael** → `tests/test_mmred.py::test_vocabulary_matches_upstream` fails for real |
| `core/metrics.py` | 0/6 bodies (`em_table`, `split_by_answer`, `bootstrap_ci`, `dprime_pair`, `law_pred`); pure numpy; 5 tests waiting |
| `core/fence.py` | 0/9 bodies; **`tests/test_fence.py:37` calls `frame_blocks` at import, so one stub blocks all 11 fence tests** |
| `core/prompt.py` | `SYSTEM_PROMPT` verified byte-identical to upstream; 0/4 bodies; the upstream-parser cross-check silently skips because `pydantic` is missing from `.venv` |
| `core/mmred.py` | qtype lists done (24 names verified against upstream); 0/7 bodies; `test_gold_parity` will run for real (seq 8+16 test JSON present) |
| `core/model.py` | dataclass + 2 helpers done; 0/6 bodies; the Qwen processor loads offline, so `test_model.py` will run for real |
| `tests/` | 3 of 25 functions pass today (one vacuously); every file aborts at its first stub |
| `experiments/` | does not exist |
| `sbatch/` | `lib/common.sh` (run_logged, stage_split) + `migrate/`; no wrappers; no `lib/splits/` |
| `data/mmred_hf` | 14 JSON splits, 15 image splits each with its `.tar` (stage_split-ready); `seq_len_1` never rendered; `headfit` is a 4th split name |
| `.venv` | one 3.11 env; `core` and `mmred` editable; `pydantic` (and possibly `pandas`) absent |

## Part A: `core/` (six modules, in dependency order)

Order chosen so every step turns a test file green and nothing waits on the GPU.

### A0. `constants.py`: one-line fix
Reorder `CHARS` to upstream order. Expected: `test_vocabulary_matches_upstream` passes.
Also decide `SEQ_LENS`: keep `1` (HF has it; `prepare_data.py` will render it in Part B) — default yes.

### A1. `metrics.py` (twin `legacy/v1/gnnformer/metrics.py`, 73 lines)
- `dprime_pair`, `law_pred`: port verbatim (docstring says so; every d′ in RESULTS.md depends on it).
- `em_table`, `split_by_answer`, `bootstrap_ci`: new, small; the tests define them.
- Gate: `python tests/test_metrics.py` → 5/5.

### A2. `fence.py` (twin `legacy/v1/gnnformer/fencing.py`, 316 lines)
Implement in this order, running the test file after each:
1. `find_subseq` (twin :129) and `frame_blocks` (twin :165 **with the documented signature
   change**: block = `[vision_start_i, vision_end_i + 1)`, not "to the next vision_start").
   Unblocks the test module.
2. `build_block_mask` (twin :29, 15 lines: causal triangle → hide columns → per block re-open its
   own causal square and forbid every other block). Pinned by 5 tests + bit-for-bit parity.
3. `hide_cols_for` (new: keep flags → every column of every dropped block). `test_gate_semantics`.
4. `reset_positions` (twin :111: every block gets block 0's ids; tail continues after block 0's max).
5. `slot_positions` (new: positions of `<|vision_end|>` in frame order). `test_slots`.
6. `find_question_spans` (twin :138: retokenize with leading-context variants until the count matches).
7. `FenceHooks.set_mask` / `install` (twin :173 minus q/k/v capture): forward_pre_hook swaps the
   held 4-D mask into positional-or-kwarg `attention_mask`, cast to hidden dtype; forward hook on
   capture layers stores `self.hidden[L]`.
- Gate: `python tests/test_fence.py` → 11/11 including `test_legacy_parity` **not** printing
  `[skip]` (it imports `legacy/v1`; if the import fails the parity is unproven and the step is not done).
- Review focus for Tal: `build_block_mask` and `reset_positions`. These are the method.

### A3. `prompt.py` (twin = upstream `scripts/openai_server_inference.py` + `scripts/utils/parse_answers.py`)
- `build_messages(frames, question, layout)`: `paper` = images then question (upstream
  `process_row`); `question-first` = upstream `--prefix_question`; `replica` = question repeated
  after every frame. Same words in every layout; only order moves.
- `answer_target(answer)`: the training target string, `{ "answer": <value> }` shape as upstream.
- `parse_answer(text)`: port of upstream parser (json_repair + regex over rooms/chars/digits).
- `exact_match(pred, gold)`.
- Semantics pinned by the tests: `parse_answer` returns one string or `None` (11-case fixture in
  `tests/test_prompt.py`); `exact_match` normalises case and leading zeros. This is the
  legacy-ladder shape, not upstream's set-valued person answers; for the faithful rows the
  benchmark scorer's rules that differ (a "no" in a person answer → Nobody; Nobody → 0 on a
  numeric gold) go into `exact_match` explicitly, and the docstring says which rule came from where.
- Two prerequisites for the cross-check test `test_parse_answer_agrees_with_upstream`:
  (a) `pydantic` is missing from `.venv` (upstream `parse_answers.py` imports it; `pandas` is
  present). **Tal runs `pip install pydantic`** (shared-env rule).
  (b) The test imports `scripts.utils.parse_answers.parse_answer`, **which does not exist**;
  upstream exposes `strip_until_first_brace` + `parse_predicted_answer`. Fix the test to call that
  pair and compare after collapsing upstream's one-element sets to a string. Until both are done
  the cross-check is vacuous and A3 is not done.
- Gate: `python tests/test_prompt.py` → 6/6 with the cross-check actually running on the fixture.

### A4. `mmred.py` (twin `legacy/v1/gnnformer/mmred_hf.py`, 645 lines; port ~1/3 of it)
1. `load_split` (twin `load_index` :30), `states` (`row_states` :36), `frames` (`load_mmred_hf_sample` :41),
   `stratified_order` (new; the K0-sorted trap: HF rows arrive answer-sorted).
2. `parse_question` (twin `_P` :97 + `_match` :145) and `recompute_answer` (twin :154, ~115 lines,
   "function by function"). Gate: `test_gold_parity` = 100 % of rows, all 24 qtypes, seq 8 + 16 test.
3. `evidence_frames` (twin `probe_evidence_mmred` :466 covers 9 qtypes; extend to 24 with the
   rule in the module docstring: positional = step lookup, trigger/first/last = all predicate
   frames, dense = all predicate frames). Gate: the three evidence tests + `every_qtype`.
- Review focus for Tal: `recompute_answer` (it defines "gold") and the 15 new evidence rules
  (they define what the gate is allowed to know).

### A5. `model.py` (twin `legacy/v1/gnnformer/runtime.py`, 150 lines)
- `load_runtime` absorbs the twin's `build_4bit_quantization_config` (nf4, bf16 compute, sdpa).
- `get_layers`, `text_config`, `get_rope_index_fn`, `image_token_groups` (twin :70/:83/:110/:115).
- `special_ids(processor)`: new; pinned by `test_special_ids`.
- Gate: `python tests/test_model.py` → 3/3 (CPU, real processor). `load_runtime` itself is only
  exercised by the Part C gate runs.

### A6. Not in this plan: `core/gate.py`
`core/__init__.py` names the gate as one of three pieces, but the scope plan puts the classifier in
`experiments/gate_capture.py` + `gate_fit.py` and the mask side is already `hide_cols_for`. The
three acceptance gates need only `--gate none|oracle`. The trained gate module is planned with the
experiment set, not here.

## Part B: `experiments/` + `sbatch/` wrappers

Four scripts, in the order the gates need them. Each follows the same loop as Part A (mini-plan →
OK → implement → Tal reads the diff), then a `--limit` smoke on `2h_2g` before its gate run.

### B0. Two additions to `core/` that the scripts need (found by the design pass)
- `core/fence.py`: `FENCED_SDPA = [EFFICIENT_ATTENTION, MATH]` (a 4-D mask excludes FLASH), and
  `layout_blocks(ids, layout, special_ids) -> (blocks, fin_start)`. `frame_blocks` keeps the tested
  rule (block = image span + two markers); for `replica` the block must extend to the next
  `<|vision_start|>` so the per-frame question copy sits inside its block, or the replica layout is
  a joint reader. Add one test for it (the scope plan's `test_layout.py` intent).
- `core/fence.py`: `fenced_setup(ids, blocks, fin_start, keep, rope_fn, inputs) -> (mask, positions)`
  and `greedy_decode(model, hooks, inputs, setup_fn, max_new, stop_fn)` (cache-free re-forward
  with mask + positions rebuilt per step). Both trainer and evaluator must call the same two
  functions so the val metric is the test metric. They belong in `core`, not in a helper module.
- `core/mmred.py`: `render_sequence(sequence, out_dir)` (official renderer, Agg backend,
  node-local `MPLCONFIGDIR`) and `with_char_moved(state, char, room)`; used by `prepare_data.py`
  and `probe_hahn.py`.
- `experiments/_port_check.py`: the only file that knows 392, `"Question: "`, the integer count
  prompt, the legacy parser ladder and the legacy `[vs_i, vs_{i+1})` block rule. ~80 lines, all
  verbatim ports with the legacy line cited, docstring "ONE-TIME PORT CHECKS: deviations from the
  official protocol; never used for a reported row." The 392 knob is
  `processor.image_processor.max_pixels = 392*392 + 1` (392² itself floors to 364 px in smart_resize).

### B1. `experiments/prepare_data.py` (twin `legacy/v1/scripts/mmred_hf/prep.py` + `render.py`)
- HF `ef1e43ce/mmred` → `json/<config>_<split>.json` (row keys in the fixed order qid, seq_len,
  qtype, atype, question, answer, sequence; no indent, for byte parity) → official
  `render_sequence_from_json` at 512 px → `images/<split>.tar` (the `tar_split.sbatch` rule).
- `render.py`'s worker cap (`SLURM_CPUS_PER_TASK`), Agg + node-local `MPLCONFIGDIR` before any
  matplotlib import, and the resume rule (dir has exactly `seq_len` frames) are ported verbatim:
  upstream's default worker count fork-bombed NFS on 2026-08-01.
- `--check-gold`: every row's `recompute_answer` equals its published answer, or abort.
- `headfit` is not on HF: rows come from `data/mmred_hf/headfit_raw.json` (1800 rows, 600 per
  length); verify mode must show byte-equality with the existing headfit JSON before trusting the rule.
- **Verification (the scope plan's "byte-for-byte on one split"):** `--verify-against data/mmred_hf
  --verify-n 50 --out $SCRATCH/verify` on `seq_len_8_val`: JSON byte-equal, md5 of the first 50 qids'
  frames equal; on any mismatch report max pixel delta and the matplotlib / Pillow / `mmred` commit
  in use. The 2026-08-01 renders came from an unrecorded matplotlib, so a pixel-level miss is
  possible and would be a finding, not a bug.
- Then render `seq_len_1_{train,val,test}` (never rendered; ~1 min each) so `SEQ_LENS` is complete.
- Cost: CPU only, `4h_0g`, ~150 frames/s on 8 CPUs.

### B2. `experiments/evaluate.py` (twins `mmred_hf/eval_frozen.py`, `gating/eval_gated.py`)
- One evaluator: `--adapter` or frozen; `--layout {paper,question-first,replica}` (default from the
  adapter's `train_config.json`, else `paper`); `--fence`; `--gate {none,oracle}` (oracle =
  `hide_cols_for(blocks, keep)` with `keep` from `evidence_frames`; a `None` evidence set skips the
  row and is counted); `--qtypes`; `--limit` after `stratified_order` (the K0 trap) and
  `--limit-per-qtype` in JSON order (legacy semantics); `--qids-file`; `--port-check {arm-a,p2}`.
- Model path: `load_runtime` → `get_layers` **before** the PEFT wrap → `PeftModel.from_pretrained`
  → `FenceHooks(layers)`. Unfenced rows use `model.generate`; fenced rows use `greedy_decode`
  under `sdpa_kernel(FENCED_SDPA)`, stopping at eos or the first `}` after `"answer"`.
- Outputs: `config.json`; `eval.csv` (qid, qtype, seq_len, atype, gold, raw, pred, correct,
  parse_ok, n_evid); `summary.csv` (per qtype, all, per atype, numeric ≤16 / >16 with n);
  `report.txt` whose first line names config, split, layout, fence, gate, adapter, port_check.
- Guard: refuse `generate` under the fence; refuse the dense path above `--max-seq-tokens`
  (N ≥ 64 at 512 px is ~21 K+ tokens; the batched fast path is the scope plan's Phase 3b, not this plan).

### B3. `experiments/train.py` (twins `loramech/train_sft_fenced.py` = P2, `sparse/train_sft_gated.py`)
- Recipe quoted from P2: `prepare_model_for_kbit_training`; LoRA r=8, α=32, dropout 0.05,
  targets q/k/v/o/gate/up/down; AdamW lr 2e-4; grad-accum 8; clip 1.0; answer-token loss only;
  `sdpa_kernel(FENCED_SDPA)`; seed 0. Target text = `answer_target(gold)`.
- **Mask held through backward**, ported from `train_sft_gated.py:545–552`: set masks → forward →
  return loss with the mask still set → caller runs `backward()` → then clear. The P2 trainer's
  `try/finally: clear_mask()` is the bug (gradient checkpointing recomputes the forward inside
  backward, unmasked); do not port it.
- Data: official `<cfg>_train` rows per config, `stratified_order`, reshuffled per epoch; val =
  official `<cfg>_val` rows (P2 carved 80/20 from train; the new row uses the real val split,
  documented as the one deliberate difference). Per-epoch val EM via the shared `greedy_decode`;
  best adapter saved with `adapter/train_config.json` (layout, fence, gate, configs, qtypes).
- Flags reserved and refused for now: `--gate model`, `--gate-bonus`, `--virtual-n`, `--gate-npz`.
- Defaults: `--epochs 5` (Tal's 2026-09-01 default), `--layout question-first`, `--gate none`.
- Memory: N = 16 at 512 px ≈ 5.3 K tokens vs 3.2 K at 392; prefer L40S/H200 over the 40 GB A100.

### B4. `experiments/probe_hahn.py` (twin `sparse/probe_hahn_gated.py`, == `armor/probe_hahn.py` with no flags)
- Paired one-frame flips (gold k → k+1) on official rows: evidence member moves the character
  into the room, control member moves it to a third room; recount asserts (k, k+1, k); members
  re-rendered with the official renderer and sha256-checked. Arms `plain` (paper layout, no fence),
  `fenced` (question-first + fence + posreset; loci `final` and the flipped block's
  `<|vision_end|>` slot), `gated` (fenced + oracle gate per member). Controls: replay, ctrl, perm.
  Hidden states via `FenceHooks(capture_layers)`, not `output_hidden_states`.
- Row supply: HF steps_in_room test rows at fixed gold are 31/19/8/2/0 (gold 0) at N =
  8/16/32/64/128, so fixed-gold cells with 30 pairs exist only at N ≤ 16; pooled gold ≤ 8 gives
  29/21/4 rows at 32/64/128. See the open decision below.

### B5. `sbatch/lib/splits/` + four wrappers
- Split files: one item per line, `#` comments allowed, read with `mapfile`; git-tracked.
  `qtypes_all.txt` (24), `qtypes_niah.txt`, `qtypes_dc.txt`, `qtypes_numeric.txt`,
  `qtypes_steps.txt`, `qtypes_anchor3.txt` (final_app steps_in_room where_spend),
  `qids_p2_seq{8,16,32}_test.txt` (the 50 qids from `data/mmred_hf/dirsfiles/seq_len_*_test_steps_shuf.txt`),
  `configs_train_8_16.txt`.
- Wrappers from `.claude/skills/sbatch-submit/assets/wrapper_template.sbatch`, all sourcing
  `sbatch/lib/common.sh`, all knobs env vars, lists via `*_FILE`:

| wrapper | header defaults | knobs |
|---|---|---|
| `prepare_data.sbatch` | l40s-shared, `4h_0g`, 04:00, 8 cpu, 16G, no GPU, `MPLCONFIGDIR=$TMPDIR/mpl` | CONFIG SPLIT STAGE LIMIT WORKERS VERIFY_N OUTPUT |
| `evaluate.sbatch` | l40s-shared, `12h_4g`, 04:00, 1 GPU, 48G; `stage_split "$SPLIT"` | SPLIT QTYPES_FILE QIDS_FILE ADAPTER LAYOUT FENCE GATE LIMIT LIMIT_PER_QTYPE PORT_CHECK EXTRA OUTPUT |
| `train.sbatch` | l40s-shared, `24h_1g`, 08:00, 1 GPU, 48G; stages train + val per config | CONFIGS_FILE QTYPES_FILE GATE LAYOUT EPOCHS LIMIT VAL_LIMIT EXTRA OUTPUT |
| `probe_hahn.sbatch` | l40s-shared, `12h_4g`, 04:00, 1 GPU, 48G; one N per job | SPLIT LIMIT CONTROLS GOLD_SET ARMS HS_LAYERS ADAPTER EXTRA OUTPUT |

### B6. Legacy behaviour that does NOT carry over (faithful runs)
count prompt with N in text / `"Question: "` prefix / no system turn → `SYSTEM_PROMPT` + bare
question + JSON target · PIL resize 392 → native 512, no resize code · first-integer parse, 4
tokens → upstream parser, 24 tokens · `qa.txt` dirs → JSON rows via `core.mmred` · block
`[vs_i, vs_{i+1})` → image-span blocks (replica gets the extended rule) · 80/20 carve of train →
official val · park renderer with "Park" → official renderer, six rooms · `output_hidden_states`
→ `FenceHooks` capture.

## Part C: the three GPU acceptance gates

What the anchor audit found (all run dirs verified on disk 2026-09-21, none behind /rg):

| Anchor | Run dir | Recipe that produced it | Deviations from the paper protocol |
|---|---|---|---|
| Frozen grid **0.533** @8 (final_app 0.765 / steps 0.559 / where_spend 0.353 from the 34-per-qtype fidelity run 127776) | `outputs/mmred_hf/frozen/grid_seq8_test/20260801_185241/` (+ seq16/32/64/128) | `legacy/v1/scripts/mmred_hf/eval_frozen.py`: paper SYSTEM_PROMPT, frames then question, greedy, decode 32 | frames resized to **392 px**; user text is `"Question: {q}"` (upstream sends the bare question); **legacy parser ladder**, not upstream's (parse_fail 0.976 by its definition) |
| P2 **0.900 / 0.660 / 0.300** @8/16/32 steps_in_room | `outputs/loramech/p2_fenced_hf/20260827_193851_fenced/longn_eval.csv` (summary.csv is degenerate) | `loramech/train_sft_fenced.py`: replica layout `[frame, q]×N` + the **old integer count prompt** ("You will be shown N frames … Output only the integer … Answer: "), 392 px, integer parse, LoRA r8/α32/lr 2e-4/grad-accum 8/10 ep | prompt, layout, resize, parser all legacy; the 50 rows = all steps_in_room rows of each test split |
| Hahn/FIXEDK "HF cells" | **none exist**: all 101 hahn/fixedk run configs on disk used park, longN_park or redux data (0 on `mmred_hf`) | `sparse/probe_hahn_gated.py` no-flag = `--gate none`, 392 px, arms plain/repjoint/fenced, layers 16/20/28 | the scope plan's gate 3 has nothing to reproduce; restated in C3 |

Consequence: **a faithful evaluator will not reproduce these numbers by default and should not.**
Each gate is therefore two runs: a *port check* under a labelled legacy preset that must hit the
old number, then the *faithful* configuration that produces the new baseline row. Only the second
is ever cited; the first proves the port and is recorded in `outputs/port/INDEX.md` as a deviation.

This follows the scope plan's own words (Tal, 2026-09-21: "follow what is written there"):
- gate 1: "note P2 used the old count prompt — run it with the OLD prompt first to prove the port,
  then with the paper prompt as the new baseline" (scope plan line 271);
- resize: "native 512 renders, no resize code … For the one-time port check against P2 (measured
  at 392) pass the processor's own `max_pixels=392*392` knob; never a resize function of ours"
  (scope plan Phase 3b);
- prompt: system = upstream `SYSTEM_PROMPT` verbatim, user = frames then the raw question text,
  "No frame count, no answer range, no per-frame text", parser = upstream `parse_answers.py`,
  metric = exact match per qtype (scope plan lines 101–107).

**Faithful defaults, everywhere, with no flag needed:** 512 px (the processor's default
`max_pixels`, untouched), `SYSTEM_PROMPT` byte-identical to upstream (already tested), the bare
question as upstream's `process_row` sends it, upstream parsing rules. The `--port-check` preset
is the only code that knows 392, "Question: ", the integer count prompt, or the legacy parser
ladder, and its report header says PORT CHECK (deviation) in the first line.

### C1. Frozen grid (proves core.model + core.prompt + evaluate.py)
- Port check: `evaluate.py --frozen --split seq_len_8_test --port-check arm-a` (= max_pixels 392²,
  `"Question: "` prefix, legacy parser, decode 32). Must give **640/1200 = 0.533** and the per-qtype
  cells in that run dir's `report.txt` (first_app 45/50, steps_in_room 28/50, where_spend 17/50 …).
  Tolerance: exact on the count; bf16 non-determinism may move 1–2 rows — if so, diff `eval.csv`
  against `per_sample.csv` and explain every changed row before calling it passed.
- Faithful: same split, native 512, bare question, upstream-rule parser. New number; logged as the
  new frozen baseline when Tal says so.
- Cost: 1200 rows × 8 frames, one GPU; the legacy run took ~1 h on a100-public. Two runs → ~2 GPU-h.
  Partition a100-public or l40s-public, `--qos=12h_4g --time=03:00:00`.

### C2. P2 adapter (proves core.fence + FenceHooks + reset_positions on the GPU)
- Port check: `evaluate.py --adapter checkpoints/sft_fenced_hf_adapter --layout replica
  --qtypes steps_in_room --port-check p2` on seq 8/16/32 test (= old count prompt, 392 px, integer
  parse). Must give **45/50, 33/50, 15/50**. Same tolerance rule as C1.
- Faithful: the same adapter under the paper prompt is *not* meaningful (it was trained on the
  count prompt). The faithful fenced row comes from Part B's `train.py` retrain on the paper prompt.
- Cost: 150 rows, ~30 min. `--qos=2h_2g`.

### C3. Hahn probe: restated as a band check (no legacy cell to reproduce)
- Because no Hahn cell ever ran on official rows, the probe cannot be port-checked against a
  logged HF number. Byte-level parity with the legacy probe is impossible on a common input
  (legacy reads `qa.txt` park dirs; the new probe reads official JSON). Parity is by construction:
  same pair rule, same loci semantics, same CSV schema, and the fence path is already proven by C2.
- Gate: run the ladder at N = 8 and 16 on official steps_in_room test rows (30 pairs, gold ≤ 8
  pooled) and require the pre-registered bands from `outputs/armor/CAMPAIGN_BRIEF.md`: joint L20
  α ∈ [0.7, 1.3] (park: 0.72), fenced per-frame |α| < 0.2 (park: 0.00 ± 0.02), replay floor
  exactly 0, perm floor bit-identical at the per-frame locus. A band miss is a finding to report,
  not a port failure, unless the replay/perm floors fail (those are code bugs).
- Cost: ~35 min per N at 512 px; two jobs on `12h_4g --time=04:00:00`.
- N ≥ 32 fixed-gold cells need the row-supply decision below and belong to the experiment-set plan.

### C4. Order and stop rule
C1 port check first (no fence, isolates model + prompt), then C2 (adds the fence), then C3.
A gate that misses by more than the bf16 tolerance stops the plan; the fix goes through the
per-module loop again, and no faithful run is submitted until its port check passes.

## Part D: docs that this plan makes stale

- CLAUDE.md line 5 ("`core/` is a skeleton Tal fills in himself — do not implement its bodies
  unasked") → replace with the working agreement above.
- CLAUDE.md §2 mentions `.venv39`; it does not exist. Drop the line or recreate the venv before
  claiming any legacy anchor is re-runnable.
- CLAUDE.md §5 "no resize code" → add: "the P2 port check passes the processor's own
  `max_pixels=392*392` knob once; that is the processor, not our resize."
- CLAUDE.md §7 last line → "headline rows"; labelled deviation rows per the scope plan §7.2 are allowed.
- Scope plan drift (stale `.venv_mmred`, `data ->` symlink shape, `.gitignore` text, test file
  names): leave the scope plan as history; this file is current.

## Sequence at a glance

**Status 2026-09-21 (evening):** A0–A5 and B0–B5 drafted by Claude in one pass at Tal's request;
all six CPU test files green (`test_fence` 12/12 incl. legacy/v1 parity, `test_mmred` gold parity
2400/2400, `test_model` on the real tokenizer, `test_port_check` 4/4); wrappers dry-run clean; a
2-qid `prepare_data` verify on the login node: JSON bytes equal, frames pixel-identical, PNG md5
differs only in the matplotlib "Software" chunk (3.11.1 → 3.11.2). Nothing committed; Tal reviews
the diffs. No GPU job submitted. Pending on Tal: `pip install pydantic`; then B1 (50-qid verify +
seq_len_1 renders) and the C1 → C2 → B3 → C3 gate sequence, each through the sbatch-submit loop.

| Step | Owner | Done when |
|---|---|---|
| A0 constants fix | Claude drafts, Tal reads | `test_vocabulary_matches_upstream` passes |
| A1 metrics | same | `test_metrics.py` 5/5 |
| A2 fence (+ `layout_blocks`, `FENCED_SDPA`, `fenced_setup`, `greedy_decode`) | same; Tal reads `build_block_mask` + `reset_positions` line by line | `test_fence.py` 11/11 + replica block test, parity not skipped |
| Tal: `pip install pydantic`; Claude: fix the cross-check test's import | Tal / Claude | cross-check runs for real |
| A3 prompt | Claude drafts, Tal reads | `test_prompt.py` 6/6 |
| A4 mmred (+ `render_sequence`, `with_char_moved`) | same; Tal reads `recompute_answer` + the 15 new evidence rules | `test_mmred.py` 9/9, gold parity 100 % |
| A5 model | same | `test_model.py` 3/3 |
| B0 `_port_check.py` | Claude drafts, Tal reads | unit test of the 392 knob (364 vs 392 px) |
| B1 prepare_data + wrapper | same | verify mode on seq_len_8_val reports md5 identity or a documented pixel delta; seq_len_1 rendered |
| B2 evaluate + wrapper | same | `--limit 20` smoke on 2h_2g; then C1 port check 0.533, then C1 faithful |
| C2 | GPU | p2 port check 45/33/15 of 50 |
| B3 train + wrapper | same | `--limit 40 --epochs 1` smoke; then the faithful P2-recipe retrain (new baseline row) |
| B4 probe_hahn + wrapper | same | smoke; then C3 band check at N = 8, 16 |
| Part D docs | Claude drafts, Tal reads | CLAUDE.md reflects the working agreement and the port-check exception |

Rough wall time if Tal reviews same-day: Part A about a week, Part B about a week, gates about
one GPU-day spread over the second week. No GPU is touched until A2 and A5 are green.

## Verification (end to end)

1. CPU: `for t in tests/test_*.py; do python $t; done` → every file prints `ALL OK`, and
   `test_fence.py` does not print `[skip]` for the legacy parity import.
2. Data: `prepare_data.py --verify-against data/mmred_hf --verify-n 50` on seq_len_8_val.
3. GPU, in order: C1 port check → C1 faithful → C2 port check → train smoke → faithful retrain →
   C3 band check. Each through the `sbatch-submit` loop with `verify_submit.sh` after every submit.
4. Every gate run dir listed in `outputs/port/INDEX.md` with its number and pass/miss.

## Decisions taken 2026-09-21 (Tal)

1. **Scope:** port + the acceptance gates. The experiment set (E1/E2/E3/E6/E7/P4/E9) gets its own plan after.
2. **Ownership:** Claude drafts every `core/` and `experiments/` body after a per-module mini-plan;
   Tal reviews every diff and runs the tests. Supersedes the CLAUDE.md line "Tal fills in `core/` himself".
3. **Port checks:** follow the scope plan's wording: old prompt first for P2, the processor's
   `max_pixels` knob for 392 px, never a resize of ours; every faithful run is 512 px + verbatim prompt.
4. **Gate 3:** band check at N = 8 and 16 on official rows; fixed-gold cells at N ≥ 32 (and any
   `--synth-questions` route) deferred to the experiment-set plan.
5. **This plan lives in the repo** as `docs/REWRITE_EXECUTION_2026-09-21.md`, updated as steps
   complete; the scope plan stays in `~/.claude/plans/` as history.

Still Tal's to do when convenient: `pip install pydantic` into `.venv` (needed only by the
upstream-parser cross-check test); confirm `seq_len_1` should be rendered in B1 (default yes).

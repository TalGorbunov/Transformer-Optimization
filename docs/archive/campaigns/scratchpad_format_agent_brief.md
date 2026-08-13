# Agent brief — Scratchpad FORMAT sweep (arms A–D)

**Date issued:** 2026-07-22. **Approved by:** Tal (via Claude session).
**Mission:** determine which scratchpad text format is best — the current positive-list was
never ablated. Four arms, identical data/recipe, ONLY the gold text differs.

## 0. Read first

1. `CLAUDE.md` §3 (SLURM/QOS), §5, §7.
2. `plans/carrier_stage4_STATE.md` + `outputs/ladder/INDEX.md` (lines ~133–138: l12v2 rows).
3. `plans/carrier_stage2_DRAFT_RESULTS.md` — claim table (rows 5–11, 19, 24) + "what
   remains" items 3–4 (rooms ordering gap; degenerate OOD chains — this sweep executes them).
4. **Coordination:** if `plans/scratchpad_loto_STATE.md` exists or another agent's jobs are
   in `squeue`, the LOTO campaign is live — share Phase-0 findings, use DISTINCT output dirs
   and different QOS slots, and never take the last free GPU in a partition.

Design rationale (from the session, 2026-07-22): every arm answers a logged failure —
tally confusion (→ running tally), long-N undercounts = search burden (→ full scan),
rooms TF 1.000 vs greedy 0.84 (→ canonical frame order), degenerate `-> a -> b` chains
(→ explicit terminator), decode budget at N≥64 (→ chunking).

## 1. Arms (gold-text specs — exact strings, tokenizer-checked before training)

- **A — control (positive-list + tally):** the existing l12v2 format. DO NOT RETRAIN —
  reuse ckpt `outputs/ladder/image_longN/carrier_tally_l12v2/20260721_071710_L12_r8/` and
  its logged cells (N=32 0.953 · N=48 0.878 · N=64 0.615/0.678) as arm A's numbers.
- **B — full-scan verdict + tally:** `scan: f1:- f2:yes(1) f3:- … f8:- | total: 2 END`
  — every frame gets a slot in frame order; evidence slots increment the tally inline.
- **C — full-scan attribute caption + tally:** as B but evidence slots carry the attribute
  word: `f2:kitchen(1)`. Labels come from sample states (`load_mmred_sample` → per-frame
  rooms dict). The agnosticism bet: same scan text can serve multiple question types.
- **D — chunked subtotals (long-N form):** blocks of 16 frames, positive-list per block,
  `| sub k` per block, final `total: a+b = c END`.

## 2. Phase 0 — code (no GPU)

- Add `--scratchpad-format {poslist,scan,caption,chunked}` to the target builders
  (`build_target_tally` and friends in `experiments/glstm/carrier_layer_lora.py`, consumed
  by `carrier_layer_cached.py`) + matching answer parser for the eval path (anchor on
  `total:`; keep poslist parsing backward-compatible).
- Resolve the l12v2 recipe verbatim from its run `config.json` (data roots, dirs-file
  splits, EPOCHS=5, L_OPEN=12, r8, fixed save criterion (acc, tf-exact), lr). New arms
  change NOTHING except the gold text.
- Unit sanity (CPU, `4h_0g` or login-trivial): for 20 samples per arm, build gold text →
  parse it back → recovered count == gold. Tally increments verified correct. Token cost
  per arm logged (B/C ≈ 4–6 tok/frame → set eval decode budgets: N=32 dec≈400, N=48
  dec≈550; D at N=64 dec≈300).

## 3. Phase 1 — pre-register (`plans/scratchpad_format_PREREG.md`, before any GPU job)

Bands (fixed now): in-dist sanity ≥0.90 per arm else BLOCKED. Primary contrasts —
- **B vs A, OOD length** (N=64 for B, trained ≤48): B ≥ A+0.05 → scan-format GO.
- **C vs B, in-dist + rooms-style recall:** C within 0.03 of B → agnostic-caption GO;
  C worse by >0.10 → supply-limited, log as carrier-bandwidth evidence.
- **D vs A/B at N=64:** D best → chunking GO for the long regime.
- Rooms ordering (if rooms is in the resolved mixture): any scan arm greedy ≥0.95 vs
  A's 0.84–0.85 → ordering-fix confirmed.
Report per cell: acc, parse-fail, MAE, mean decode tokens. Between-band results logged
honestly as partial; bands never adjusted post-hoc.

## 4. Phase 2 — launch

- **Smokes first:** each new arm (B/C/D) 2-sample EPOCHS=1 on `2h_2g`, output
  `outputs/_scratch/`, verify TF loss falls + gold round-trip in the training log.
- **Trainers (3):** `runners/of_carrier_cached.sbatch`, l12v2 recipe + `EXTRA_FLAGS`
  carrying `--scratchpad-format <arm>`. Remember the `--export` comma-truncation gotcha —
  multi-root DATA_ROOT via env file/wrapper, never inline commas. ~4–6 h each, 1 GPU.
  Spread: `12h_4g` ×2 + `24h_1g` ×1 (or per live load).
- **Evals per new arm:** in-dist held-out; N=32 + N=48 held-out (E-F dirs-files, identical
  to arm A's cells); one OOD cell (N=64, LIMIT 60); rooms cell if applicable. Launch each
  arm's exams only after ITS trainer is COMPLETED. Evals on `24h_1g`/`4d_1g`.
- Before every submit: free-GPU check across ALL partitions (sinfo GresUsed pattern in
  CLAUDE.md §3); a100-public 40 GB is fine for everything here.
- Output layout: `outputs/ladder/image_longN/carrier_fmt_{scan,caption,chunked}/<ts>_L12_r8/`
  and `fmt{B,C,D}_eval_*/` siblings; INDEX rows on landing.

## 5. Phase 3 — collect & close

- Draft entries per landed cell in `plans/carrier_stage2_DRAFT_RESULTS.md` (run dir + number
  + verdict vs band); INDEX rows for canonical cells; maintain
  `plans/scratchpad_format_STATE.md` (overwrite-at-transition); blockers →
  `plans/scratchpad_format_BLOCKED.md` with successor actions.
- Final deliverable: one table — arm × {in-dist, N=32, N=48, N=64-OOD, rooms, parse-fail,
  decode-tokens} — plus a 5-line verdict naming the winning format and which pre-registered
  bands hit. Do NOT edit `RESULTS.md`.

## 6. Hard rules

Never: pip/conda install; delete/modify anything under `outputs*/`, `data/`; heavy compute
on login node; QOS bigger than needed; touching jobs you didn't launch. Every number must
trace to a run dir. Poll `squeue -u $USER` every ~20–30 min; prep next phase between polls.

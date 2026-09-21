# REDUX campaign — the reduction-type law (decisive experiment)

**Status: REFINED BRIEF (Claude, 2026-08-30) — awaiting Tal's OK for GPU cells.
C0 (CPU datagen + exam building) runs meanwhile per the campaign instructions.**
Scripts: `scripts/redux/`. Runs: `outputs/redux/`. Log: `STATE.md` (append-only).
RESULTS.md untouched — DRAFT entries at the bottom of STATE.md. gnnformer/ read-only.

## 1. Claim under test (from Tal's prompt, unchanged)

Reduction type determines the usable code; the code determines the length law.
MAX/OR → native to softmax (share→1 under sharpness): length-flat once supply is
fenced. SUM/COUNT → share-coded magnitude: one octave of slack, then lawful death
(LORAMECH, known). RATIO/MAJORITY → the share IS the quantity: N-invariant at fixed
|d|/N, Weber-like near the boundary. Decisive figure: same model, same frames, same
one-frame flip, three questions.

## 2. Exact deltas (all in scripts/redux/ except two guarded in-place extensions)

- **`scripts/redux/tasks.py` (WRITTEN):** the one label module — prompt templates
  (count = build_count_prompt byte-identical; exists/majority share its opener line so
  the fenced parse_layout needle works unchanged), replica_text per task, gold derived
  from states, yes/no + integer parsing, task_of_dirs_file (exam_{task}_N*.txt).
- **`scripts/redux/build_exam_dirs.py`:** C0 exam files (mirrors the LORAMECH builder;
  stratified draws per §4 below; REPORT.txt with class dist + majority baseline/cell).
- **`scripts/redux/eval_tasks.py`:** C1/C3-behavioral plain-prompt ladder (loader +
  resize 392 from eval_frozen's loop; greedy generate; per-class tables + prediction
  CSVs; `--peft-adapter` for the P1m plain-prompt drift row if wanted later).
- **`scripts/loramech/train_sft_fenced.py` + `--tasks count,exists,majority` (in-place,
  guarded):** per-sample task draw (seeded rng, balanced); replica question + final
  prompt + gold from tasks.py; dirs-file eval infers task per file. Default
  `--tasks count` = byte-identical current behavior (P1b reproduces). Mask/decode code
  untouched (the instruction's no-fork rule).
- **`scripts/armor/probe_hahn.py` + `--task {count,exists,majority}` (in-place,
  guarded; no-flag = byte-identical, anchors reproduce):**
  - count: existing behavior (base gold ≤ 8, flip k→k+1), digit margins.
  - exists: `--task exists` selects base gold = 0 pairs, flip → k=1; margins =
    logit(yes) − logit(no) at the first answer token (prompt from tasks.py).
  - majority: base gold = N/2 (probe_base pool), flip → N/2+1 (answer flips yes);
    existing same-frame wrong-room→wrong-room ctrl is the answer-preserving control.
  - arms plain + p1fence as already implemented; `--peft-adapter` as-is.
- **Wrappers:** `slurm/redux_datagen.sbatch` (CPU, per-N), `redux_eval_tasks.sbatch`,
  `redux_hahn.sbatch` (chain over N, TAG/ARMS/TASK/ADAPTER knobs, ARMOR pools for
  count / redux probe_base for majority / K0-rich pools for exists).

## 3. Data plan (C0 — CPU, 4h_0g, runs now)

Existing pools cover: N=8 everything (park all_uniform, 100/class, k=0..8);
exists strata k∈{0,1,2,4,8} at N≥16 only 30/class (too thin for 50% k=0 cells);
majority near-boundary k∉pools for N≥16 (classes {0..8,12,16,24,...}).

**New pool `data/mmred_redux/` (balanced park generator, task steps_in_room, same
renderer/rooms; generator's dead `evaluations` import fixed 2026-08-30 — lazy, only
rooms/cooc need it; 1-sample smoke verified incl. load_mmred_sample round-trip):**
- exam split (`all_uniform`, seed 1, per-count 50):
  N=16: counts 0 1 2 4 6 7 8 9 10 12 16 · N=32: 0 1 2 4 8 12 14 15 16 17 18 20 24 32
  N=64: 0 1 2 4 8 16 24 28 30 31 32 33 34 36 40 48 64
  N=128: 0 1 2 4 8 16 32 48 56 60 62 63 64 65 66 68 72 80 96 128
- probe_base split (seed 3, per-count 80): counts = {N/2} for N∈{16,32,64,128}
  (majority C4 base pairs; N=8 base k=4 comes from park K4).
- Gold engineered + asserted per dir by the generator; tasks.py re-derives and
  re-asserts at exam-build time (skip+count mismatches).
- Fresh generation ⇒ disjoint from every training root by construction; post-hoc
  overlap checks still recorded.

**Exam files (`outputs/redux/examdirs/`, strided, REPORT.txt like LORAMECH's):**
- `exam_count_N{8..128}.txt` = the LORAMECH `exam_ff_N*.txt` files AS-IS (byte-reuse).
- `exam_exists_N*.txt`: 150/cell at N=8, **100/cell at N≥16** (k=0 stratum = 50 =
  half the redux per-count — documented shortfall per the brief's allowance); 50%
  k=0, yes half ≥1/3 k=1, rest k∈{2,4,8}.
- `exam_majority_N*.txt`: 150/cell, stratified over signed d = k−N/2 ∈ ±{1,2,4,8,16,32}
  (clipped), balanced yes/no; N=8 from park (d∈±{1,2,4}).

## 4. Cells, partitions, cost (GPU only after OK)

| cell | job(s) | partition/QOS/time | est |
|---|---|---|---|
| C0 datagen ×4 N + exam build | CPU | 4h_0g, --time=2:00:00, mem 12G | ~1–2 h wall |
| C1 plain frozen ladder (3 tasks × 5 N) | 1 | rtx6k/l40s/a100 48G, 12h_4g, --time=08:00:00 | ~5 h |
| C1 p1fence_frozen yes/no ladder (2×5) | 1 | same, --time=06:00:00 | ~4 h |
| **CHECKPOINT: H-FROZEN reading → STOP for review** | | | |
| C2 P1m multitask trainer (≤16, 10 ep) | 1 | rtx6k/l40s 48G, 24h_1g, --time=8:00:00 | ~7 h |
| C3 P1m ladder ×3 tasks ×5 N + logN sub-cell | 2 | 12h_4g, --time=05:00:00 each | ~7 h |
| C4 probe chains: {exists,majority}×{plain,p1fence}×{frozen,P1m} + count×P1m×2 arms = 10 chains | 4–10 | a100-public 12h_4g, --time=05:00:00 | ~12–15 h GPU |
| figures + drafts | CPU | — | — |

Count-frozen/P1b probe curves reuse ARMOR-A + LORAMECH n2_hahn (no rerun). No H200
anywhere. Smokes (limit 5, 1 ep) in outputs/_scratch/ before every full cell.

## 5. Hypotheses/bands — FIXED, from Tal's prompt verbatim

H-MAX: P1m exists ≥0.90 @64 AND @128 while P1m count ≤0.50 @64; refuted if exists
≤0.75 @64; logN-restored ≥0.90 = "retrieval, sharpness-bound"; k=1 stratum reported
separately, ≥0.85 yes-rate @128 = the real claim. H-ALPHA: α_exists < α_count
(0.80 [0.66,0.97]) non-overlapping CIs, exists margin median >0 through 128;
<0.2 pointer / 0.2–0.6 sharp-but-diluting / ≥0.6 refutes taxonomy. H-RATIO: at
|d|/N=1/8 acc within ±0.10 across N∈{16..128}; at |d|=1 monotone decrease with N;
refuted if |d|/N=1/8 falls >0.25 from 16→128. H-CTRL: P1m count = P1b ±0.05
in-length + 2×; frozen-plain count ≈0.219 @8; any miss HALTS. H-FROZEN: descriptive,
two-sided, decides whether MAX needs training at all.

## 6. Honesty flags (standing)

Prompt states N in text (majority-at-extremes and "all frames" partially readable off
the prompt; |d|≤8 strata are the informative ones — flagged in every majority table).
Cell sizes 100–150 (50/stratum shortfalls documented). Single task family, single
model. Yes/no priors: report per-class + majority baseline everywhere. bf16: probe
floors are the measured replay/ctrl/perm, never assumed.

## 7. ADDENDUM (Tal, 2026-08-30 evening — AGENT_PROMPT.md v2): C4b + H-SHARP

**C4b — α under N-aware sharpness + decay-in-k (no training; P1b adapter
`sft_fenced_le16_ep10_adapter`; runs in PARALLEL with C1):**
- probe_hahn deltas: `--attn-logn-sref` (same scaling hook as the trainers, S_ref
  3398, applied per forward from that forward's seq len; attn modules captured
  pre-wrap) and `--gold-set` (explicit base-gold selector, overrides --max-gold;
  no-flag behavior byte-identical — smoke must reproduce N2 pair dnorms).
- Chain (i): p1fence+P1b, count flips, logN ON, N∈{8..128}, 50/N (40@128), ctrl 12;
  reference = existing `n2_hahn/p1fence_ep10_N*` (no rerun).
- Chain (ii): N=64 (32 if time), base k∈{1,2,4,8,16,32}→k+1 via six one-k
  invocations (--gold-set k --limit 30) from `data/mmred_redux/seq_len_64/all_uniform`
  (50/class, covers every k incl. 32), logN ON and OFF; fit Δ ∝ (k+C)^−γ; margins
  skipped for gold>8 (single-digit protocol limit).
- **H-SHARP** (verbatim from AGENT_PROMPT.md §4): (a) logN drops read-α to <0.3 (CI
  excludes 0.6) with verdict α≈0; (b) γ ≥ 1 (CI excludes 0.5) at N=64, flag on or
  off; (c) accuracy @64 stays ≤0.6 → walls are separate. Failure modes and the
  positive-surprise protocol as written there.
- **Checkpoint updated:** MANDATORY STOP for review after C0+C1+C4b (frozen ladder +
  H-FROZEN + H-SHARP readings in STATE) BEFORE C2 — reinstated by the v2 prompt and
  binding over the earlier blanket go.

---

# REDUX v3 addendum (2026-09-20, authorized by Tal "run all of those") — the reduction-type law, gated AND ungated

C0/C1/C4b are on record (frozen: exists 0.83→0.49, majority at chance ∀N; H-SHARP refuted). C2/C3 now
run as TWO multitask readers on the sparse gated trainer (single source for the three task templates):
- **C2a ungated:** `scripts/sparse/train_sft_gated.py --tasks count,exists,majority --gate none
  --nfree-prompt`, roots seq8+seq16, 5 ep (2026-09-01 default), excludes = all 15 redux exam files +
  the 5 count exam files + S4/S7 dirs; in-job LONGN eval on the 15 redux exam files (150/cell).
- **C2b gated:** identical with `--gate oracle` (the per-frame predicate — C in R — is the same for
  all three tasks, so the same oracle gate serves all three).
- C4 per-task flip chains: DEFERRED (probe_hahn_gated has no --task; implement only if C3 lands).
Outputs: `outputs/redux/c2a_ungated/`, `outputs/redux/c2b_gated/`.

## Pre-registered predictions from the share law (fixed now)
| task | ungated reader | gated reader |
|---|---|---|
| count | exact k≤8 decays: ≥0.9 @16, ≤0.5 @128 (H-R1) | exact k≤8 ≥0.95 at every N (H-R4) |
| exists | ≥0.9 @16; decays to ≤0.75 @128 as the single evidence frame dilutes (H-R2, two-sided: if ≥0.9 @128 the OR read sharpens — report) | ≥0.97 at every N (one visible frame suffices) (H-R5) |
| majority | N-FLAT at fixed ratio: accuracy @128 within 0.1 of @16 and ≥0.8 (H-R3) | BREAKS: ≤ chance+0.15 at N ≥ 32 because the gate deletes the denominator the ratio needs (H-R6) |
Failure protocol: any missed band is reported as measured; the interpretation stops at the band.
Caveats: exam cells 100–150; majority cells balanced yes/no (majority 0.5); N=128 cells need ≥80 GB
(run trainers on h200/rtx6k-96GB); single seed.

## REDUX v3 amendments after the pre-launch review panel (2026-09-20, before any C2 run)
- **N-free templates implemented** for exists/majority (`tasks.build_prompt_nfree`, majority replica
  without the frame count); the trainer guards relaxed (`--gate oracle` with any task list; `--virtual-n`
  stays count-only). Majority is evaluated BOTH ways: N-free (primary: tests "the gate deletes the
  denominator") and N-in-text (secondary, eval-only: tests count-and-compare / the k-wall).
- **Budget:** `--limit 2700` so each task keeps ~900 items (same per-task budget as the single-task runs);
  per-task label distributions emitted at split time; the `[data]`/`[exclude]` lines quoted in STATE
  before training proceeds (≥ 502 excluded dirs expected).
- **Reader gate before long-N bands:** in-window count exact ≥ 0.97 at N=8/16 for both readers; if missed,
  compare per-k to `outputs/sparse/s9b_ep5` before interpreting any long-N band.
- **H-R3 restated at matched ratio:** flat within 0.1 and ≥ 0.8 on r ∈ {0.25, 0.375, 0.625, 0.75} (N=8
  k{2,3,5,6}; 16 {4,6,10,12}; 32 {8,12,20,24}; 64 {16,24,40,48}; 128 {32,48,80,96}); second clause:
  accuracy at fixed |d| ≤ 2 DECAYS with N; per-class reporting with the majority baseline 0.500 beside
  every number; pooled cells descriptive.
- **H-R6 restated:** gated majority ≤ 0.65 on the |d| ≤ 8 strata at N ≥ 32 (extreme strata are separable
  by the count code and are excluded from the band).
- **H-R2 exists:** banded as k=0 no-recall and k=1 recall separately (training labels are ~89% yes).

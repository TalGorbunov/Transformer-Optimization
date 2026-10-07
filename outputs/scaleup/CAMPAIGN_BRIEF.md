# SCALEUP — two more datasets, two more backbones, one dataset- and model-agnostic `core/` (2026-09-23)

**Status: plan APPROVED (Tal, 2026-09-23; `docs/SCALEUP_2026-09-23.md` = live copy of
`~/.claude/plans/sorry-i-forgot-to-valiant-lake.md`). Branch `theory-2`. No code written, nothing run.**
Working agreement unchanged: per-module mini-plan → Tal OKs → Claude drafts → Tal reads the diff → tests →
Tal commits. Every GPU submission goes through the `sbatch-submit` loop; no cell is pre-authorized.
Log: `STATE.md` (append-only, newest last). Index: `INDEX.md`. Run dirs:
`outputs/scaleup/<dataset>/<backbone>/<stage>/<cell>/<stamp>_<job>/`. Job ledger: `jobs.tsv`.

## 0. One paragraph

Every number so far is Qwen2.5-VL-7B × official MMReD. SCALEUP adds two real-video MCQ benchmarks with
per-question evidence timestamps — **HERBench lite_v2** (1,971 Q / 68 videos / 12 tasks, explicit
`metadata_json` timestamps for 11 tasks) and **MINERVA** (~1.3 K Q / ~190 videos / 12 usable skills,
timestamps regex-derived from the reasoning traces) — and two backbones reviewers of video benchmarks
recognise — **InternVL3.5-8B-HF** (Qwen3-8B LM, always labelled) and **Gemma-3-12B-it** (non-Qwen LM,
bidirectional image attention, 5:1 sliding layers). The code gets two seams (`core/data/`,
`core/backbones/`) so one `evaluate.py` / `gate_capture.py` / `train.py` serves every (dataset, backbone)
pair. Order (Tal): frozen baselines → **per-block evidence classification per task** (is "this block
holds an evidence frame" linearly present in the block's slot state? = whether a gate is learnable, and
whether perception is the wall) → oracle-gate upper bound → the method where stage 2 says yes.

## 1. Fixed inputs

| item | value |
|---|---|
| anchor | official MMReD, Qwen2.5-VL-7B, paper prompt, 512 px — every new backbone is validated here first (S4); every new dataset on Qwen first (S1–S3) |
| datasets | `herbench` = HF `DanBenAmi/HERBench` config `lite_v2` (videos = `videos.tar.part.00–03`, 34.3 GB, kept on `/rg/shocher_prj/lab_data/herbench/`); `minerva` = lmms `minerva.json` rows + `lmms-lab-eval/minerva` Lance blobs (8.85 GB, `/rg/shocher_prj/lab_data/minerva/`), Listening skill dropped |
| on-disk layout (both) | `json/<split>.json` · `pool/<video_id>/frame_%04d.jpg` (128 uniform frames, long side 512, JPEG q95) + `times.json` · `evidence/<qid>/frame_%04d.jpg` + times · `pool.tar`, `evidence.tar` for `stage_split` |
| N protocols (video) | `planted_N<k>`: evidence frames at annotated times (+0.3 s for points, midpoint for ranges) + fillers ≥ 5 s from every evidence time, sorted by time, k ≤ N/2 (the k-of-N instrument protocol); `uniform_N<k>`: first k of the pool (official comparability; labels = ±half-interval ≤ 2 s / inside range; coverage reported) |
| N | 8 / 16 / 32 / 64 / 128; official comparability row = uniform N=16, frames-first, the benchmark's own prompt suffix, letter regex |
| rows | 50 per task per dataset (stratified by answer letter) for S1–S3; all rows for the comparability row |
| backbones | `qwen2.5-vl-7b` (default) · `internvl3.5-8b` (`OpenGVLab/InternVL3_5-8B-HF`, `crop_to_patches=False` → 256 tokens/frame, `<img>…</img>`, 1-D RoPE) · `gemma-3-12b` (`google/gemma-3-12b-it`, 256 tokens/frame, `<start_of_image>…<end_of_image>`, 1-D RoPE, bidirectional in-image, 1024 sliding on 40/48 layers; day-1 4-bit smoke) |
| closed models | NOT run (no masks / positions / hidden states); published Gemini / GPT numbers quoted |
| arms | `plain` (paper layout), `qfirst`, `fenced_qlast` (question-blind control), `fenced_qfirst` (the method's layout), `gated` (fenced + oracle) — one table, one place |

## 2. Stages (cells, scripts, costs: `docs/SCALEUP_2026-09-23.md` §5)

S0 data (CPU) · S0 code (seams + refactor proof: C1 faithful N=8 and D4 fenced_qfirst N=16 identical) ·
S1 frozen ladders on Qwen · S2 per-block evidence classification on Qwen (fit held out by **video**) ·
S3 oracle upper bound · S4 backbone ports on MMReD (smoke → frozen grid → D4 2×2 → oracle ladder) ·
S5 cross cells · S6 method (training data decided after S2; options §6 of the plan).

Stop rules: S2 AUC ≈ chance for a task on Qwen → perception wall, no S3/S6 there; S4 frozen N=8 ≈
chance on a backbone → it cannot read the renders, swap before S5.

## 3. Pre-registered expectations (bands to fill in STATE.md before each stage runs)

- S1: Qwen2.5-VL-7B uniform N=16 on HERBench lite_v2 within the published band (35.9 % full set;
  README ~31 %); planted accuracy above uniform at every N (evidence delivered by construction).
- S2: on `planted`, fenced_qfirst slot AUC per task; needle-type tasks (AGAR/AGBI/AGLT; Reading/Object
  Recognition) expected highest; question-blind fenced_qlast ≈ chance as on MMReD (D4).
- S3: oracle gate ≥ frozen at every N; the gap = what a perfect gate buys per task.

## 4. What Tal does before S0

`pip install av pylance` · accept the Gemma license on HF, download both models to the home cache ·
optional `cleanup_2026-09-21.sh` for /rg headroom (97 % full on 2026-09-23) · amend the two CLAUDE.md
rules (§7 of the plan) or OK Claude's wording.

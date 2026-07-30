# Campaign PROGRESS — autonomous run started 2026-07-10

> One line per item: status (queued/running/landed/blocked) + job IDs. See
> plans/2026-07-08_next_experiments.md (charter) and plans/QUESTIONS.md (approvals).

| Item | Status | Jobs / paths | Notes |
|---|---|---|---|
| P0 Q1 long-N data gen | **landed** | 119986–119989 | `data/mmred_longN_park/` seq 16/32/64/128, 330/390/450/510 samples, 1.4 GB, gold verified. (First attempt 119981–4 truncated by sbatch comma-split of COUNTS; resubmitted space-separated, K=0 files regenerated identically.) |
| P0 A1-fu3 cooc block-read | **landed** | 119985 | block d′ 3.43@L14 vs single 3.05 (+0.38/+0.42 borderline); E4 PASS; RESULTS [2026-07-10] logged; INDEX updated |
| P0 A1-fu1 text multipass cache | **landed** | 119991 + 120010 | **NOT write-capped: multipass d′ 7.9 @L16 vs joint 2.45; mp-sum 0.965** — RESULTS [2026-07-10b] |
| P0 A1-fu2 easy-text minimal pair | **landed** | 119992 + 120011 | **binding account REFUTED: block d′ 2.89 (vs 2.45), model 0.215 unchanged** — RESULTS [2026-07-10c] |
| P1 B1 d′-vs-N caches | **landed** | caches 120013–22 (+120004); final analysis 120065 | **joint FLAT ~2.0 ≪6.3 ∀N; multipass 7.2–8.1 ≥crush ∀N; fenced ≤joint ∀N; collapse is downstream of supply** — RESULTS [2026-07-11], Fig B1 written |
| P1 B2 gate calibration | **landed** | 120059 | **raw gate FN 0.09→0.99 (threshold drift ✓); mass-norm cuts drift 10× but FP over-counts; fenced dilutes too (✗); bias law exact** — RESULTS [2026-07-10i] |
| P1 B3 behavior vs N | **landed** | 120030/120031/120032 | **0.173→0.020 N=8→128; emitted range clamped ~3 ∀N, ordinal corr 0.75 to N=64** — RESULTS [2026-07-10f] |
| P3 evidence-only behavioral | **landed** | 120029 | **wall reproduces with zero selection load (0.00 @N≥6, pred≈4.6 @N=8)** — RESULTS [2026-07-10e] |
| P1 C1b phrasing grid | **landed** | 120012 | **interface gated on predicate-match + pre-question position; robust within (words 1.00 OOD, distractor 0.97, distance-free); paraphrase/post-question = 0.00** — RESULTS [2026-07-10d] |
| P1 C2+C3 codebook injection | **landed** | 120027/120040/120060 | **token interface NECESSARY: all learned routes 0.81–1.00 in-range → 0.00–0.10 held-out; real tokens 1.00/1.00 @0 params; C-control exact ✓; Fourier+native-anchored fail too (C3 resolved)** — RESULTS [2026-07-11b] |
| Q3 sources | **landed** | — | data/coco_val2017 (826M, 5k imgs + instances json), data/oxford_pets (875M); archives deleted |
| mmred_natural (A4) | **landed** | build 120028/120038/120041, judges 120034/120039, caches 120042/46/47/48, d′ 120056–58 | **gate passed after 1 curation round (0.998–1.00); d′ dial works (6.21→4.28); model ≈ law on the no-binding rung** — RESULTS [2026-07-10g,h]; data `data/mmred_natural_v2/` |
| **⚠ QUOTA INCIDENT 21:29** | resolved | — | home hard-limit 330G hit; all running jobs failed; freed 30.7G (InternVL + falcon-mamba HF caches, re-downloadable); see QUESTIONS.md Q7 (decision: may I also delete the 64G 32B cache?) |
| P2 Q2 MLVU-AC port | **landed** | prep 120037/51/68, judge 120070, behavior 120069 (N=32) + 120071 (N=128) | **206/206 Qs prepped (950M durable, transients deleted); sampling-limited at std budgets (35% zero-evidence @N=32); N=128 lifts MCQ .282→.393, mean-pred 1.02→2.45** — RESULTS [2026-07-11c] |
| P4 VNBench stretch | queued | — | only if P0–P2 land |

GPU-hour ledger (est submitted): A1-fu jobs ~3 GPU-h + C1b ~1.5 + B1 wave ~16 ≈ **20 GPU-h** of ~50 budget.

---

## CAMPAIGN SUMMARY (written 2026-07-11 ~02:00, autonomous session complete)

**Every charter item landed except VNBench (skipped for disk — Q8).** 9 RESULTS.md entries
appended ([2026-07-10]–[2026-07-11c]); all four pre-registered figures written
(`outputs/ladder_report/fig_{a1_ladder,a2_natural_dial,b1_final,b2_final,c1_routes}.png`);
report.md/report.html/theory_background.html data updated in place; ladder + readout INDEX.md
current. ~35 GPU-h submitted (≤50 budget). One ops incident (home quota hard-limit at 21:29,
killed a full wave; resolved by deleting 30.7G of re-downloadable HF model caches — Q7 asks
whether the 64G Qwen-32B cache may go too).

### Headline results (all traceable to run dirs; verdicts vs frozen predictions)
1. **The "text write cap" fell** (A1-fu1): isolated per-frame forwards lift text carrier d′
   2.45→7.9; with the fence null and the easy-text refutation (A1-fu2: surface-form binding
   changes nothing), the supply cap is **joint-context processing per se** — modality-general.
2. **B1 complete matrix (Fig B1)**: joint carrier d′ is FLAT ~2.0 from N=8 to 128 (never near
   the 6.3 crush line); multipass is N-invariant at 7.2–8.1; fenced ≤ joint everywhere. The
   behavioral collapse (0.173→0.020, B3) is **downstream of supply**: emitted range clamped at
   ~3 for every N (ordinal signal survives to N=64), plus gate threshold drift (B2: raw FN
   0.09→0.99; mass-normalization cuts drift 10× — the bias law N·FP−g·(FN+FP) is exact).
3. **The token interface is NECESSARY, not just sufficient** (C2+C3): every learned
   embedding-level injection route (per-digit, per-count control, continuous Fourier, Fourier
   anchored to the model's own digit geometry) hits 0.81–1.00 on trained counts and 0.00–0.10
   held-out; real digit tokens: 1.000/1.000 at zero parameters. C-control prediction hit exactly
   (0.000). C1b: the interface is gated on predicate-match + pre-question position; within the
   gate it's robust (words 1.00 OOD, distractors 0.97, distance-free).
4. **mmred_natural works as the thesis d′-dial** (A4): judge-gated construction (0.998–1.00),
   d′ 6.21/5.41/4.30/4.28 across the 2×2 — and on this no-binding rung the model RIDES the law
   (gap ≤0.11): the MMRED wall is binding-specific.
5. **MLVU-AC ported** (A3, 206 Qs durable in data/mlvu_ac/): at standard 32-frame budgets it is
   evidence-DELIVERY-limited (35% of questions have zero visible evidence; model calibrated to
   what it sees); 4× denser sampling lifts everything in lockstep. No insertion GT exists in the
   released files (plan assumption failed); judge labels + 7 dup-detect-exact questions in place.
6. Evidence-only behavioral gap closed: the wall reproduces with zero selection load.

### Blocked / follow-ups for Tal
- Q7: may I delete the 64G Qwen-32B HF cache? (quota at 312G/330G hard.)
- Q8: VNBench stretch skipped (needs ~20G transient > ~8G safe headroom).
- E4 adequacy on the natural + long-N caches (cheap CPU pass) — A-4 verdict still pending.
- MLVU d′/parity instrument pass on judge labels (caches not yet cut for it).
- New sbatch gotcha for the SLURM memory: `--export` values are COMMA-SPLIT (bit COUNTS, the
  d′ cache map, and C2's ROUTES) — pass lists via files or spaces. 4h_0g mem cap is 16G.

---

## CONTINUATION CAMPAIGN (2026-07-11, Tal's Q7/Q8 approvals executed)

| Item | Status | Jobs / paths | Notes |
|---|---|---|---|
| Q7 delete 32B cache | **done** | — | 68.3G freed via HF API; quota 248G/330G |
| P0a fence reconciliation + mass-competition registration | **landed** | — | RESULTS [2026-07-11d]; incl. cross-family tax constant (P4l satisfied from existing anchors) |
| P0b E4 sweep (natural + long-N) | **landed** | 120130–138 | **A-4 partial: adequacy tracks evidence DIVERSITY** (dist cells PASS, ident FAIL via kurt; long-N joint kurt +0.5→+25) — RESULTS [2026-07-11e] |
| P0c MLVU d′/parity | cache landed (120139), block-read running (120150) | `outputs/ladder/mlvu_ac/msgcache_n32judge/20260711_125240/` | judge-labeled N=32 frames |
| P0d law+clamp closure | computed (MMRED ✓ near-exact, HERBench directional); MLVU transfer pending d′ | `outputs/ladder/image_longN/law_clamp/<ts>/` | law+clamp 0.186/0.106/0.069/0.054/0.039 vs measured 0.173/0.127/0.073/0.053/0.020 |
| P1e chunk sweep k∈{2,4,8,16} @N=32 | **running** | 120143 (smoke 120140 ✓) | anchors k=1: 8.08, k=32: 1.98 |
| P1f attn-renorm patch | **N=8 VERDICT: REFUTED — renorm 2.03 ≈ joint 1.97 ≪ mp 7.18** | 120145+120149 (N=8); 120146 (N=32 running) | registered mass-competition hypothesis fails at the carrier hop → tax is upstream in-context encoding |
| P1g composition probe | **skipped per pre-condition** | — | (f) showed no recovery |
| P2h e2e pipeline | smoke ✓ (0.625 @N=8, 2 fwd/sample) → **full run in flight** | 120148 smoke, 120151 full | chunked k=8 + massnorm gate + per-N margin + fact render |
| P2i retrieve-then-verify | queued (after 120151) | — | |
| P3 VNBench | downloading (10.9G zip, gdown) | ~/vnbench_transient | 1800 cnt items, 450 videos, **exact needle_time GT in json** |
| P4j native axis vs N | **running** | 120144 | |
| P4k dedup semantics | **running** | 120147 | |
| P5m writing pass | in progress | STORY.md §§3/5/6/7 updated (revision trail kept); theory_background.html walls+parity+footer updated | scorecard rows after landings |

---

## CONTINUATION CAMPAIGN SUMMARY (2026-07-11 ~14:30)

**Every item landed or resolved.** 10 new RESULTS.md entries ([2026-07-11d]–[2026-07-11m]);
Q7 executed (68.3G freed, quota 248→254G of 330G); Q8 (VNBench) ported and scored. One job
still queued: nataxis_lg 120169 (N=32/64 native-axis legs need an H200; 8/8 busy — it will
self-complete and write to `outputs/ladder/image_longN/native_axis/` whenever one frees; the
N≤16 stability verdict is already logged). ~15 GPU-h this session (~50 total across both
campaigns). Figures current: fig_a1_ladder (8 rungs), fig_b1/b2, fig_b3_clamp, fig_a2, fig_c1.

### Headlines (all traceable; verdicts vs frozen predictions)
1. **Mechanism of the joint-context tax pinned by refutation + onset** ([2026-07-11g,j]): the
   registered mass-competition hypothesis is REFUTED (equal-mass renorm patch recovers 1–8% of
   the joint→multipass gap), and the chunk-size sweep shows the tax onsets at the FIRST
   companion frame (d′ 8.08 → 3.37 at k=2 → 1.98 at k=32). The tax lives in in-context frame
   ENCODING — not attention edges (fence), not carrier mass, not binding, not legibility.
2. **Law ∘ clamp = behavior, zero fitted params** ([2026-07-11i]): rank-remapping the d′-latent
   through the measured emission marginal reproduces the entire B3 N-collapse (mean err 0.013
   vs plain law's 0.13) and transfers directionally to HERBench and MLVU. Plus P4j: the native
   reading axis is STABLE across N (|cos| 0.82–0.86) — the clamp is saturating magnitude on a
   fixed axis.
3. **The constructive centerpiece** ([2026-07-11l]): retrieve-then-verify — frozen model as its
   own extractor/verifier/verbalizer with one N=8-trained logistic gate — sustains **0.880 /
   0.807 / 0.707 exact at N=32/64/128** (frozen: 0.073/0.053/0.020; full multipass: 0.680/0.580/
   0.420). Honest miss: cost ≈ N forwards (diluted joint margins shortlist ~85% of frames);
   chunked-k=8 variant is the budget option at N≤32 only (0.600 @N=8 for 2 forwards).
4. **VNBench scored against frozen predictions** ([2026-07-11k]): E4-fails-via-variance-ratio ✓
   exactly; "CWE-like escape" ✗ (d′ 2.50 = MMRED level, carrier flat) — synthetic needles do not
   buy content addressing. MLVU d′/parity landed ([2026-07-11h]): action-token carrier, model at
   its delivered-evidence ceiling. E4 sweep ([2026-07-11e]): adequacy tracks evidence DIVERSITY,
   not binary-groundability; long-N joint messages go heavy-tailed (kurt +25 @N=128).
5. **Dedup semantics** ([2026-07-11f]): question wording selects the aggregation operator in the
   frozen model — frame-count vs support-size dissociate exactly on identical-needle cells
   (0.83 exact-vs-unique). The rooms_visited support-size thread now has a causal handle.
6. Writing pass done: STORY.md (§3 N-scaling revision, §5 fence scoping + third solution, §6
   token-necessity, §7 natural rung — all with supersession notes), theory_background.html
   (three walls rewritten, parity rows added, footer), ladder report scorecards complete.

### State of the theory (peer-meeting paragraph)
The counting failure of frozen VLMs is now a three-stage account with measured constants and a
zero-parameter behavioral closure. (1) SUPPLY: per-frame evidence enters the question-carrier at
whitened d′ ≈ 2 in any joint pass — a ~3.4× "joint-context tax" vs processing frames alone
(7–8), constant from N=8 to 128, in two modalities and two model families. The tax is paid at
the first companion frame and is NOT attention-edge interference, NOT carrier-mass competition
(both refuted causally), and not binding or legibility: frames are encoded differently the
moment they share a context. (2) AGGREGATION: given that supply, the √N law says what any linear
readout of the pooled carrier can achieve; adequacy (E4) self-diagnoses where the closed form is
licensed, and fails in interpretable modes (graded evidence, degenerate-identical evidence,
judge-label noise, long-context heavy tails). (3) READOUT: the model reads a fixed, misaligned
axis whose emission range is clamped N-invariantly at ~5; law ∘ measured-clamp reproduces the
entire behavioral collapse with zero fitted parameters. The readout's interface is tokens and
only tokens — real digits verbalize OOD at 1.000 while every learned activation-level injection
(digit-compositional, per-count, Fourier, native-anchored) memorizes and fails OOD — and it is
gated on predicate match + pre-question position. The wall is specific to carrier-mediated
relational evidence: on natural no-binding rungs the model rides the law, and question wording
alone switches it between frame-count and support-size operators. Constructively, deciding
per frame BEFORE summing — deployed as retrieve-then-verify with the model as its own verifier
and a fact-sentence tally render — recovers 0.71–0.88 exact through N=128 on a fully frozen
system, above full multipass and above the joint-carrier law ceiling. Open: what exactly in
in-context encoding pays the k=2 tax (the one mechanism question left), and an evidence-scaled
shortlist to cut retrieve cost from N to O(count).

---

## CAMPAIGN #3 SUMMARY (2026-07-11 evening; continuation of the 07-10/11 runs)

**Every charter item landed** (11 new RESULTS.md entries, [2026-07-11n]–[2026-07-11v]); nothing
blocked. ~25 GPU-h this session. Quota 269G/330G. Figures: fig_frontier.png added; walls/parity/
scorecards updated in report.md, report.html, theory_background.html, STORY.md (revision trails
kept). Ops: the native-axis OOM was 125 unfrozen float params building the full graph — fixed
(freeze + graph-start + logits_to_keep; N=32 backward now runs on a 48G L40S in minutes).

### Headlines
1. **The joint-context tax is dissected** ([2026-07-11r,s]): (i) a saturating long-context
   component ≈1.9 d′ (isolated frame at padded positions reads 5.3, pad-length-invariant);
   (ii) a content-similarity interference component — gray/noise companions ≈ free (6.4),
   same-scene 3.3 — cross-frame FEATURE interference, graded exactly as registered; (iii) the
   fence-depth sweep is an informative null (every partial-depth edge-cut hurts): the
   interference cannot be surgically removed from a frozen model. Only lever: k≤2 forwards.
2. **Retrieve-then-verify v2** ([2026-07-11t]): k=2-margin shortlists lift N=128 to
   **0.791±0.055** (v1 0.707; mp-sum 0.420; frozen 0.020) with N=32 at 0.862±0.014 (3 seeds).
   Registered keep-rate 25–35% REFUTED (measured 65–70%) — v2 buys accuracy, not forward-count.
3. **The task algebra generalizes the pipeline** ([2026-07-11p]): rooms_visited **0.993**
   (support-size operator — the dedup finding deployed; frozen 0.193), cooc 0.513 (frozen
   0.127), registered ordering exact. **Cross-family** ([2026-07-11u]): InternVL 0.690 vs its
   frozen 0.090, render lossless. **Regime boundary** ([2026-07-11o]): HERBench e2e is NULL
   (0.157 vs 0.172) — on graded evidence the verifier is the wall; regime-2 confirmed
   constructively from the failure side.
4. **Token-necessity airtight** ([2026-07-11v]): residual-level injection (L14–17) fails OOD
   exactly like embedding level across digit/count/Fourier/native-anchored routes.
5. **Readout closure completed** ([2026-07-11q]): native axis drifts mildly (|cos| 0.72–0.86)
   with flat ~0.6 d′ along it to N=64 — the clamp is emission-side saturation, full stop.
   E4 tags ([2026-07-11n]): multipass/k=2 messages are clean Gaussians at every N; joint
   processing is what breaks adequacy.

### State of the theory (revised, peer-meeting paragraph)
Frozen-VLM counting failure is a three-stage account, now with the supply stage mechanistically
dissected and the constructive counterpart demonstrated across tasks and model families.
SUPPLY: per-frame evidence reaches the question-carrier at d′≈2 in any joint pass vs 7–8 alone —
a ~3.4× joint-context tax, flat to N=128, family-general. The tax decomposes into a saturating
long-context term (~1.9 d′, position/context-length, padding-invariant) plus a content-graded
cross-frame interference term (contentless neighbors free; same-domain neighbors cost ~3.9 d′),
paid at the first companion frame. It is not carrier-mass competition (causally refuted), and
not removable by cutting cross-frame attention at any depth (all partial fences net-negative) —
in a frozen model, interference and computation share the same edges. AGGREGATION: the √N law
gives the linear-readout ceiling over the pooled carrier; E4 adequacy self-diagnoses its license
and now doubles as a processing-mode signature (isolated messages are clean Gaussians at every
N; joint messages go heavy-tailed). READOUT: a stable, misaligned reading axis (mild drift,
|cos|≥0.72; d′≈0.6 along it at every N) feeding an N-invariant emission clamp; law∘clamp
reproduces the behavioral collapse with zero fitted parameters; and the interface is tokens
only — every learned injection route, at embedding AND residual level, memorizes trained values
and extrapolates to zero, while real tokens verbalize OOD perfectly under predicate-matched
pre-question phrasing. CONSTRUCTIVELY: decide-per-frame-then-reduce, with the model as its own
verifier and a fact-sentence render, sustains 0.79–0.88 exact to N=128 (frozen: 0.02), swaps
the reduction operator per task (Σ / distinct-count / pair-Σ; rooms_visited 0.993), and ports
to InternVL (0.690) — bounded exactly where the theory says it must be: by per-frame perception
(HERBench graded evidence: null). Open problems: a finer-than-fence instrument to localize the
content-interference depth; an evidence-scaled shortlist (keep-rate is calibration-bound at
~65%); and long-N task algebra.

## COORDINATION NOTE (2026-07-12 14:20, Tal's main session)
P1 (qkv 2×2) TAKEN OVER by the main Claude session at Tal's request — do NOT resubmit:
- smoke analysis running (main session, CPU, outputs/_scratch/qkv2x2_smoke/analysis/run.log)
- FULL capture SUBMITTED: job **120529** (n=150, seed 5, L14/16, @392px) →
  outputs/ladder/image_longN/qkv_2x2/20260712_full/
- main session will run qkv_2x2_analysis.py on it when it lands and log the RESULTS entry.
Tmux agent: continue P2/P3/P4 only; treat P1 as in-flight-elsewhere.

## COORDINATION UPDATE (2026-07-12 15:2x, main session)
P1/P1d CLOSED by the tmux agent ([2026-07-12b,c]) — decisive via the frameless-query
monotonicity (0.53 < 1.71 < 3.14), which is reconstruction-scale-INVARIANT (ratios/ordering,
same pipeline all cells), so the deflated absolute anchors in my 2×2 capture do NOT threaten the
conclusion. Main session's geometry-fix re-capture is therefore NOT needed for the verdict;
kept only as an optional instrument-validation footnote (does own-geometry mp×mp reproduce the
direct 7.18). No full n=150 geomfix re-capture will be launched. Main session did NOT log a P1
entry (agent's [2026-07-12b/c] stand).

---

## CAMPAIGN #4 SUMMARY (2026-07-12 — mechanism closure + coverage completion; FINAL)

**All items landed** (7 new RESULTS entries, [2026-07-12]–[2026-07-12f]); ~12 GPU-h; quota
274G/330G. Note: Tal worked the account in parallel — the P1 capture/analysis jobs
(120529/120530) came from his side using this campaign's scripts; his own-geometry anchor
variant (geomfix, n=8) is exploratory and left as his open thread ([2026-07-12f]).

### Headlines
1. **The mechanism is closed** ([2026-07-12]–[2026-07-12c]): after the correction (first-frame
   causality kills the encoding-interference reading), the query/encoding 2×2 lands Tal's
   prediction maximally — a JOINT-context carrier query collapses the per-frame read to joint
   level even on clean values (1.59 vs clean 4.79 @L16); clean queries on joint values keep
   most separability (3.14). The frameless-q control then sharpens it into the final statement:
   q_c carries FRAME-CONDITIONED routing (frameless 0.53 < joint 1.71 < per-frame 3.14) — the
   joint-context tax is a ONE-QUERY-MANY-FRAMES binding limit: one 3584-dim query cannot hold
   8 frame-specific routing programs. Over-squashing, relocated to the ADDRESSING. (Anchor-gate
   caveat documented: the reconstruction runs ~0.85× native scale; ratios reproduce.)
2. **P1d dead on both branches** ([2026-07-12c]): per-frame clean queries cap the fix at
   d′ 3.1–3.5 (< the ≥5 wiring bar) and the 2-forward frameless variant fails outright —
   clean queries ARE per-frame forwards; there is no shortcut.
3. **P2 refuted, and explained** ([2026-07-12e]): adaptive early-stop and two-stage prefilter
   both miss the registered bars; the ~⅔N verification floor is calibration-bound at d′3.4
   margins, and P1 proves better margins cost one forward per frame — accuracy and cost are
   the same supply bound. retrieve-v2 stands as the deployable config (0.79–0.88, ~1.15N).
4. **Long-N task algebra** ([2026-07-12d]): rooms_visited 0.993/0.967 @N=32/128 (frozen
   0.300/0.213) — the distinct-count operator is near-N-invariant (union absorbs read errors);
   cooc degrades gracefully to 0.233 @128 (honest band-edge miss; pair-verifier compounding).
5. Writing: STORY.md and theory_background.html carry the corrected mechanism ("the read,
   factored": msg_f = m_f · o_proj(Σŵ_j v_j); tax rides ŵ_j) with full revision trails.

### State of the theory (final, peer-meeting paragraph)
Frozen-VLM counting failure is now a closed three-stage account. SUPPLY: per-frame evidence
reaches the question-carrier at d′≈2 in any joint pass vs 7–8 alone — a ~3.4× joint-context
tax, flat to N=128, family- and modality-general. The tax is now LOCALIZED: it is chiefly
query-routing — the carrier's single query vector must encode where-to-look for every frame at
once, and this addressing degrades with each frame it must serve (frameless 0.5 → joint 1.7 →
frame-specific 3.1–4.8 on identical values), with a secondary value-encoding component and a
saturating ~1.9 long-context term. It is not attention mass (causally refuted), not removable
edges (all fence depths hurt), and content-graded exactly as an addressing account predicts.
AGGREGATION: the √N law bounds any linear read of the pooled carrier; E4 adequacy licenses the
closed form and doubles as a processing-mode signature (isolated messages are clean Gaussians;
joint messages go heavy-tailed). READOUT: a stable misaligned axis feeding an N-invariant
emission clamp; law∘clamp reproduces behavior with zero fitted parameters; the interface is
tokens only, at embedding AND residual level. CONSTRUCTIVELY: decide-per-frame-then-reduce
sustains 0.79–0.88 exact to N=128 on steps, 0.97+ on rooms_visited (the union operator is
near-N-invariant), ports across families, and fails exactly where per-frame perception fails
(HERBench, cooc) — and its ~N cost floor is now a THEOREM-SHAPED consequence of the mechanism:
frame-conditioned addressing costs one visit per frame in a frozen model. What training-time
fix this implies (multi-slot carrier queries? per-frame carrier tokens?) is the natural next
project.

### Ready-to-write checklist (thesis chapters ← RESULTS entries)
- **Ch. 3 Diagnosis (d′ framework + laws):** [2026-07-03*] parity engine · [2026-07-05] native
  axis · [2026-07-09] block reads · [2026-07-10f,i] clamp+drift · [2026-07-11i] law∘clamp ·
  [2026-07-11n,e] E4-as-signature. Figures: parity table, fig_b1, fig_b3_clamp.
- **Ch. 4 The joint-context tax (mechanism):** [2026-07-10b,c] multipass/easy-text ·
  [2026-07-11d] registration · [2026-07-11g] renorm null · [2026-07-11j] chunk curve ·
  [2026-07-11r]+[2026-07-12] correction · [2026-07-12b,c] the 2×2 + frameless (headline) ·
  [2026-07-11s] fence null. Figure: chunk-curve, 2×2 grid table.
- **Ch. 5 The ladder (task/data generality):** [2026-07-07d,e] HERBench · [2026-07-08c,d]
  text rungs · [2026-07-10h] mmred_natural · [2026-07-11c,h] MLVU · [2026-07-11k] VNBench ·
  [2026-07-11e] adequacy-vs-diversity. Figures: fig_a1_ladder (8 rungs), fig_a2.
- **Ch. 6 Readout & the token interface:** [2026-07-08b] C1 · [2026-07-10d] C1b gate ·
  [2026-07-11b] embed-level necessity · [2026-07-12 v-entry 2026-07-11v] residual-level ·
  [2026-07-11q,m] axis stability · [2026-07-11f] dedup semantics. Figure: fig_c1_routes.
- **Ch. 7 The constructive pipeline:** [2026-07-11l] retrieve v1 · [2026-07-11t] v2+seeds ·
  [2026-07-11p]+[2026-07-12d] task algebra · [2026-07-11u] InternVL · [2026-07-11o] HERBench
  null (regime boundary) · [2026-07-12e] cost floor. Figure: fig_frontier.
- **Open problems section:** own-geometry 2×2 anchors (Tal's geomfix thread) · training-time
  multi-slot addressing · evidence-scaled verification below the ⅔N floor · MLVU/VNBench
  denser-sampling instruments.

## COORDINATION (2026-07-12 23:1x, main session) — overnight firm-up
Firming the fresh [2026-07-12b–i] results at n=500 (was n=150, single-seed, recon-scale).
- capture job **120639** (n=500, seed 7, L14/16 @392px) -> outputs/ladder/image_longN/qkv_2x2/20260712_n500/
- battery job **120640** (afterok dep): 2x2+own-geometry anchor, encoding un-mixer, query un-mixer (~4000 ex).
Purpose: validate the own-geometry anchor (failed at n=8), firm the value-un-mixer (~100%) and the
KEY claim — query-un-mixer at 4x data to kill the data-limitation caveat on "query irreducible".
Tmux agent: do NOT resubmit qkv/un-mixer work; this is main-session's thread.

## COORDINATION (2026-07-13 01:0x) — overnight firm-up LANDED
n=500 battery (job 120645) done: anchor VALIDATES (mp own-geom 7.81≈7.18), value un-mixer 93%, query un-mixer still fails -27% (irreducibility airtight). RESULTS [2026-07-13]. First battery OOM (4h_0g 16G cap vs 13.4GB capture) → rerun on 180G GPU node.

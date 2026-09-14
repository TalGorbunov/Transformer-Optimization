# Current-prefix null references: a proposed continuation diagnostic

The frozen V10 models contain substantially more usable first-token evidence than their ordinary extrapolation accuracy suggests. Subtracting a training-estimated background direction rescued almost every selected first token. The next question is whether the same correction works throughout native answer generation. This document proposes that test; it does not release training or claim a finished method.

## What is established

The registered background diagnostic used all four dev-selected V10 cores and one identifier-selected family per K0–16, at N32 and N64. Directions came only from training K0 scenes. There was no fitting or generation.

| V10 core | N32 base / corrected / sham | N64 base / corrected / sham |
|---|---:|---:|
| CE, seed14 | 2 / 17 / 4 | 1 / 15 / 1 |
| CE, seed15 | 3 / 17 / 2 | 0 / 17 / 3 |
| Consistency, seed14 | 6 / 17 / 3 | 4 / 17 / 3 |
| Consistency, seed15 | 9 / 17 / 5 | 4 / 17 / 2 |

Every denominator is17. All34 bare native replay checks and136 exact branch decompositions passed. The job used34 native forwards,34 visual forwards and442 additional native-head calls; elapsed time was86.5seconds. These are first-token results: K10–16 all require first token `1`. They do not establish correct larger counts, EOS completion, or a deployed counting algorithm. The reused subset is exploratory. [Registered diagnostic and raw artifacts](../../outputs/native_aggregation_vlm/v10/background_diagnostic/run_442260/REPORT.md).

## One native forward at each actual prefix

Use the same four frozen cores. For a question q, the reference bank contains exactly the24 image occurrences from its two K0 training scenes, N8 followed by N16 in the frozen training schedule. Preserve occurrence weighting, original pixels and displayed Step labels. Do not insert reference answers or a token `0`.

Pack N actual N1 image streams,24 reference N1 image streams, and the unchanged text-only global stream. All N+25 rows receive the actual global greedy token after every forward. Their native KV caches remain independent. Thus the reference states are evaluated under the current generated prefix, including prefixes that were never present in the old feature cache. This increases parallel width; it introduces no extra sequential model call per token.

Let phi be the existing core's conditional local-message map. At each query, compute the actual and reference messages separately, keeping the actual-message kernel shape identical across modes:

\[
 z=\sum_{i=1}^{N}\phi(h_i,g),\qquad
 \widehat\mu_0=\operatorname{float32}\!\left(\frac1{24}\sum_{j=1}^{24}\operatorname{float64}\phi(h_j^{\rm ref},g)\right).
\]

With A=Wagg, retain the existing readout offset and global query:

\[
 r=Az+b_\rho+W_q\operatorname{RMS}(g),\quad b_0=A\widehat\mu_0,\quad
 \delta=U\operatorname{SiLU}\big(r-(N-16)b_0\big).
\]

Use the successful diagnostic's projected-space subtraction and FP64-reference-mean→FP32 convention. Add `delta.to(native_hidden.dtype)` to the global native hidden state before final RMS and the ordinary vocabulary head. There is no output mask, fallback, rounding rule or external count decoder.

The matched base includes the same24 unused reference streams and computes the same reference messages, but leaves r unchanged. The sham subtracts an equally long direction perpendicular to b0 in the96-dimensional projected space. Its fixed noise key is `[core_key, question]`, seed20261101; neither N, intervention mode nor generated prefix enters that key. The noise is projected anew against the current b0. A zero background direction gives a zero sham. Every mode has identical model rows and reference ancestry.

Once generated tokens differ, the three trajectories must run separately. Sharing a later forward would condition one intervention on another's tokens. The reference streams themselves receive no residual writes; only the global final-norm query does. At fixed token history all native KV remains unchanged. This is distinct from the proposed V11 persistent-memory write.

## Software gate, then the bounded answer diagnostic

A software-only smoke should first check fixed CPU layouts and reference identities, unchanged source/weights, exact zero-U native identity, separate-row caches, one visual prefill, and one native model call per token. The new controller's CPU tests cover the live reference mean, future-token causality, cached/full equivalence, native addition precision, reference permutation/duplication and sham construction.

The registered software smoke uses the unchanged old N16/K3 and N64/K6 cases with the fixed software core already used for native integration. Its total ceiling is50 native model calls and24 visual calls, with a240-second GPU job cap and480 allocated GPU-seconds across all attempts, including failures. Actual tokens, masks, mRoPE, row ordering, call counts and unchanged fixed-prefix KV are structural checks. Same-captured-state native-head replay is separate from descriptive full/cached numerical differences; preserve failures rather than retuning tolerances. The10 reference-controller self-tests passed in CPU Slurm442273, using3 allocated CPU-seconds. The later full source/input freeze442309 passed all11 tests, including the empty-actual case, in17 allocated CPU-seconds. Native software442314 then passed in130 allocated GPU-seconds:46 model calls,24 visual calls,46 additional native-head replays,24 fixed-prefix KV checks,36 fusion checks,6 zero-U identity checks and780 cached/full row comparisons. All binding checks passed, and this augmented batch had no descriptive numerical failures. Its measured script time was108.6seconds; allocated GPU time is the resource accounting measure. These are software results, not task efficacy.

After that gate, run all34 old diagnostic contexts, all four selected cores, and all three modes:408 complete trajectories, at most1,632 native calls and408 visual prefills. Score the whole ASCII integer plus ordinary EOS within four tokens, retaining malformed/truncated outputs. Report every K, each length, each model and each intervention; keep first-token metrics separate. Native raw global argmax must generate the next token. No checkpoint selection, coefficient search, bank pruning or special treatment of failed K is allowed.

The original resource gate required four jobs capped at600seconds and a2700 allocated GPU-second total envelope. It failed before selected-model inference: `load_seconds + 1.25 * 102 * T64 + 30 = 1051.4341647819383`seconds exceeds600. Here the unchanged formula is `T64 = prepare_seconds + generation_seconds * (4 / observed_generated_tokens)`; the measured augmented N64 bound is7.831716092070565seconds and load is22.890363042941317seconds. Each model has102 trajectories, all charged at that longer-context rate. The original failure is retained in [the timing-gate artifact](../../outputs/native_aggregation_vlm/v12/original_study_timing_gate.json).

A prospective resource-only amendment, registered before study source freeze or selected outcomes, extends each job to1200seconds and the total study envelope to5100 allocated GPU-seconds, including failures. Four1200-second reservations leave300seconds of reserve. The identical1051.434-second projection now fits; no work or overhead was removed from the estimate. The separate software envelope stays480seconds, with130 spent. All408 trajectories, reference rules, arithmetic, modes, frozen models and scoring remain unchanged. Source/input freezing and independent reporting still precede any conclusion. The86.5second first-token diagnostic is not an estimate of full answer-generation cost.

If complete answers improve, confirm the unchanged frozen procedure on fresh same-law families before tuning it. The existing V11 fresh-test *stager* can serve this purpose; at the source review neither its data directory nor its staging outputs existed. Retain all four original cores and the same bank rule, anchor and sham. Fix accuracy criteria and compute allocation prospectively. This still tests the supplied54 questions and generator law; transfer to other queries/tasks and reasoning traces requires separate evidence.

## Contingent fresh confirmation: source plan only

Reuse `scripts/stage_native_vision_v11_test.py` and `slurm/native_vision_v11_test_stage.sbatch` unchanged, explicitly documenting reassignment from the held memory campaign. Their seed20261010 and canonical `/mnt/data/gabriele/gnn_transformer/v11_fresh/main_manifest.json` provide272 contexts: eight families per K0–16, each N32 extended toN64. Complete context/question hashes exclude exactly the V10 prior25 manifests plus V10 balanced and fresh manifests. Those exclusions cover the old V12 examples and all training reference images' source scenes; image-atom reuse is allowed and disclosed. There are no saturated-context reuse exceptions. Literal V10 training/dev, schedule, pairing and their audit ancestry remain protected.

The unchanged CPU sequence is `--dry-check`, independent review of its exact semantic/source/input plan, then `--stage --plan <that passed plan>`. Rendering cannot start without that plan. The published manifest binds its semantic audit, inventory and dry plan; the completed stage audit binds the final manifest bytes. A confirmation data release must bind all of these, the original source snapshots and all27 exclusions. A reporter importing `audit_published` should normalize a copied loaded histogram's JSON string keys to integers before calling it; retain the original manifest byte hash and never rewrite the file.

Use a NEW confirmation driver and wrappers rather than editing or monkeypatching the fixed34-scene study. The reusable computation is `native_vision_v12_runtime.prepare_scene/reference_bank/generate_native`; reusable independent scoring is `evaluate_native_vision_v12_reference.score_ids/audit_output/cell_metrics`, whose per-record checks do not depend on population size. The new driver owns its prospective policy, fresh manifest audit, complete record order, prepared-input plan, selected-checkpoint restore, counts, report and resource ledger. The old study's `check/verify_plan/run/report/study_projection/allocation_ledger` hardcode34/17/102/408 and the old selection, so they cannot simply consume a replacement manifest. Freeze the original study's final repaired source hash as an ancestor.

Every original V10 core evaluates every fresh context in base/background/sham mode:3264 trajectories, at most13,056 native model calls and3264 visual prefills. The same24 ordered K0 training occurrences remain in all modes, with no reference reselection, coefficient change, new fitting, checkpoint choice or answer-dependent bank lookup. Fresh inference uses the exact same native EOS/max4 policy and rejects nonterminal special tokens. Retain raw FP32 vectors with exact FP16 roundtrip identity, all malformed/truncated outputs and each mode's own generated prefix. Each core contributes816 complete trajectories. No first-token proxy can substitute for a whole-answer result.

Freeze a confirmation criterion before fresh predictions. Report per-core background-minus-base and background-minus-sham at N32/N64, complete answers and high-count K9–16 separately; all four cores must remain visible. For uncertainty, resample the136 complete two-length families within K, carrying every mode and core together. Such intervals are conditional on these fixed checkpoints. A pooled gain must not hide a reversal in one core. Fresh same-law confirmation of this calibrated procedure remains distinct from a general bank-free or reasoning milestone.

A budget cannot be inherited from the smaller study. There are eight times as many trajectories. Use completed study timings, including preparation/loading, archive writes, validation and all failures, to freeze a fresh resource projection and reserve before launch. Keep at most four concurrent GPUs. The original instrumented software bound would imply about8041seconds per core for816 trajectories; it must not be silently replaced by one eighth of a desired cap. A measured uninstrumented-study projection may be more informative, but its formula and safety margin must be registered explicitly. If needed, deterministic family-complete shards can cap individual jobs, at the cost of repeated model loads; their union must be exactly the272 planned contexts for each core/mode. No GPU jobs or additional data drawing are released by this source plan.

## What a more general method would require

The principled object is a centered measure,

\[
 z_c(S,q,p)=\sum_{x\in S}\phi(x,q,p)-|S|\,\mathbb E_{x\sim P_0(q)}[\phi(x,q,p)],
\]

followed by a nonlinear native decoder. This makes the expected irrelevant contribution neutral under the stated reference distribution. It retains additive composition at a common question/prefix. SUM already supplies that algebra; the new empirical issue is the origin and variability of irrelevant contributions, not a new attention operator.

The anchor16 is retrospective calibration to an already trained readout. A future model trained from initialization on `sum(phi)-N*mean_null(phi)` would remove that N anchor and let its native decoder learn the centered coordinates directly. That experiment would require new matched training controls. It cannot be claimed from this frozen-core correction.

Mean centering does not make every irrelevant image's message zero. Under ideal independent reference sampling, its estimation error contributes a covariance term proportional to N²/M (or (N−16)²/M for the diagnostic) in addition to actual-set variability. Repeated/correlated reference occurrences reduce effective sample size. Learning small null variance, and testing changes of nuisance distribution, would matter for reliable larger-N aggregation. No worst-case neutrality guarantee follows.

Finally, obtaining a K0 reference bank for every query is a substantial assumption. The current balanced dataset supplies all54 query-specific banks through training labels. That is neither query-general nor label-free preparation. Possible future learned null expectations must be validated at arbitrary prefixes and unseen queries; their accuracy and cost cannot be assumed. The present contribution is a sharply testable diagnosis and native integration of reference centering, with explicit extra width and calibration assumptions. Reasoning composition remains untested.

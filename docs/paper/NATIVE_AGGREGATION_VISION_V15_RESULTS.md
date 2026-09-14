# V15: live empirical null means do not rescue the fixed V14 models at N64

The verified V15 diagnostic does not support replacing the learned null predictor with this live 24-occurrence training bank as a repair for V14's N64 failures. For the two centered models, complete-answer accuracy falls from 9/17 to 3/17 and from 7/17 to 5/17. The offsets' correctness is unchanged. This is an exploratory intervention on 17 reused families, not fresh confirmation or a new trained method. V14's failed practical milestone remains unchanged. [Verified report](../../outputs/native_aggregation_vlm/v15/null_comparison_study/report_442711/REPORT.md) · [All audited counts, captures and provenance](../../outputs/native_aggregation_vlm/v15/null_comparison_study/report_442711/analysis.json) · [V14 results](NATIVE_AGGREGATION_VISION_V14_RESULTS.md).

All four V14 final models are retained: centered/offset, seeds 18/19, core update 4,590 and student update 8,000. For each count K=0–16, the CPU plan chose the lexicographically smallest canonical V14 family identifier, then included both its N32 and N64 contexts. The same 34 scenes are used by every model and mode. These scenes were already scored in V14; no new question, integer or visual vocabulary is introduced, and no outcome-based scene or checkpoint selection is allowed. [Frozen input and model plan](../../outputs/native_aggregation_vlm/v15/null_comparison_study/check_442691/plan.json).

Each scene generates two independent native trajectories. Both modes process N actual image streams, the same 24 ordered K0 training-reference image occurrences for the question, and one global text stream: N+25 rows. Both compute actual messages, reference messages and the compact predictor; only the mean supplied to the projected readout differs. Learned uses `c(q)`; bank uses the FP64 occurrence mean of current FP32 reference messages, cast back to FP32. Centered subtracts N times the projected mean; offset subtracts it once. There is no N16 anchor, refitting, scaling search, bank pruning or count-specific decoding. Duplicate reference occurrences remain weighted separately. After generated tokens diverge, each mode recomputes its own states from its own history.

Correctness requires the complete stripped ASCII integer and native EOS within four tokens; nonterminal special tokens are invalid. The original native logits determine global greedy choices without a vocabulary mask, and the selected token is broadcast across that mode's streams. All 272 trajectories remain in the denominator, including malformed and truncated answers.

| Model | Seed | N32 learned /17 | N32 bank /17 | N64 learned /17 | N64 bank /17 |
|---|---:|---:|---:|---:|---:|
| Centered | 18 | 16 | 16 | 9 | 3 |
| Centered | 19 | 13 | 16 | 7 | 5 |
| Offset | 18 | 4 | 4 | 2 | 2 |
| Offset | 19 | 6 | 6 | 2 | 2 |

The centered N64 changes are −6/17 and −2/17 answers. The bank helps centered seed 19 at N32 by three answers, but does not improve either centered N64 total. Equal totals can conceal exchanged successes: centered seed 18 at N32 gains one previously wrong answer and loses one previously correct answer.

| Centered seed | N | Both correct | Learned only correct | Bank only correct | Neither correct |
|---|---:|---:|---:|---:|---:|
| 18 | 32 | 15 | 1 | 1 | 0 |
| 18 | 64 | 2 | 7 | 1 | 7 |
| 19 | 32 | 13 | 0 | 3 | 1 |
| 19 | 64 | 2 | 5 | 3 | 7 |

Both offset models retain exactly the same correctness decisions under the two mean choices. Their N64 correct answers all lie in K0–8. The centered N64 partitions show that the failure of the bank replacement is not restricted to two-digit answers:

| Centered seed | Mean | N64 K0–8 /9 | N64 K9–15 /7 | N64 K16 /1 | N64 K9–16 /8 |
|---|---|---:|---:|---:|---:|
| 18 | Learned | 4 | 4 | 1 | 5 |
| 18 | Bank | 1 | 2 | 0 | 2 |
| 19 | Learned | 4 | 2 | 1 | 3 |
| 19 | Bank | 2 | 3 | 0 | 3 |

All N32 outputs are parseable and EOS-completed. At N64, both centered learned modes remain parseable in 17/17 cases but truncate in one and two cases; both bank modes terminate and parse in all 17. Better stopping therefore accompanies worse complete-answer accuracy. Centered first-token correctness falls from 12→7 and 13→9, while whole-answer correctness falls from 9→3 and 7→5. The offset seed-18 modes each have five malformed/truncated N64 outputs, and seed-19 modes each have one. No invalid answer is removed. First-token correctness remains an incomplete proxy because counts 10–16 share a leading digit.

The added reference rows are controlled explicitly. On all 136 selected model/scene combinations, augmented learned-mode generated-token sequences exactly match the saved original N+1 V14 sequences. This observed agreement is descriptive: it does not assert raw-logit equality, identical kernels, or equivalence on untested contexts. The causal comparison within V15 is learned versus bank under the same augmented batch construction. Reference processing is paid by both modes; it is not free inference.

The result rules out a narrow proposed rescue on these fixed models and scenes: simply substituting this live empirical mean does not remove the N64 failure. It does not prove that the learned predictor is an accurate population-null estimator, that negative mean removal is useless, or that the model lacks aggregation information. The finite bank participated in training and differs from the actual scenes in position and nuisance composition. Its live native states also differ from cached training-state targets. Those alternatives cannot be separated by bank accuracy alone. Likewise, later captures from diverged trajectories are not same-state comparisons. No confirmatory threshold or bootstrap interval was registered for this reused 17-family diagnostic.

Independent CPU report442711 passed in 21 allocated seconds. It retained and checked all 272 trajectories and all 637 actual generated-prefix captures, including exact FP32 promotion of native FP16 logits, raw argmax, EOS/ASCII scoring, model/source/layout/reference identities, and paired correctness transitions. The studies used 637 native model calls and 272 visual prefills. The preceding software profile442641 used 40 native model calls, 20 visual prefills and 40 captured-state head replays; all binding checks and all 520 descriptive cached/full row comparisons passed. [Software evidence](../../outputs/native_aggregation_vlm/v15/null_comparison_software/profile_442641/summary.json) · [Frozen study source](../../scripts/evaluate_native_vision_v15_null_comparison.py).

The four study allocations used 224, 225, 225 and 224 GPU-seconds. Together with the 92-second software profile, the independently checked raw Slurm ledger totals **990 GPU-seconds (0.275 GPU-hours)** across five successful GPU allocations, within the registered 4,200-second cap and maximum concurrency four. Preserved CPU failures include attempt442669, which exited successfully without executing because the new script lacked its entrypoint, and check442678, which rejected an incompatible metadata-helper schema. The entrypoint and then only the helper import were repaired; full CPU check442691 passed before any study inference. Scheduler success alone was not accepted as a validation gate. No failed GPU allocation is omitted. [CPU release and resource projection](../../outputs/native_aggregation_vlm/v15/null_comparison_study/check_442691/summary.json) · [Raw all-user allocation ledger](../../outputs/native_aggregation_vlm/v15/null_comparison_study/report_442711/all_user_sacct.psv) · [Preserved zero-execution gate](../../outputs/native_aggregation_vlm/v15/null_comparison_study/no_execution_442669/summary.json).

The separately registered offline decomposition442712 passed in 15 allocated CPU-seconds, retaining all 272 trajectories and 637 executed prefixes without another model or head call. Its source, plan, report and observation bindings were independently checked. For actual-frame messages `m`, live bank mean `mu`, deployed mean `nu`, projection `W` and coefficient `a`, it writes the projected aggregate as `P + Z + B + E`: positive-frame residuals `P = sum_positive W(m-mu)`, negative-frame residuals `Z = sum_negative W(m-mu)`, baseline `B = (N-a)Wmu`, and mean mismatch `E = aW(mu-nu)`. Centered has `a=N`, so B is zero; bank mode has `nu=mu`, so E is zero. The maximum FP64 vector-identity residual is 9.09e-13. The maximum discrepancy from native FP32 preactivation arithmetic is 3.63e-4 and remains descriptive. [Verified decomposition](../../outputs/native_aggregation_vlm/v15/message_decomposition/decomposition_442712/summary.json) · [Geometry and limitations](../../outputs/native_aggregation_vlm/v15/message_decomposition/decomposition_442712/geometry.json).

At the first answer position, all 136 model/scene pairs have identical actual, reference and global state hashes, message hashes, queries and both candidate means across learned and bank modes. The following within-model means use each of the 17 K values once; the cosine excludes K0, whose positive sum is exactly zero.

| Centered seed | Mean P norm, N32→N64 | Mean Z norm, N32→N64 | Mean cos(P,Z), N64, 16 K>0 cases |
|---|---:|---:|---:|
| 18 | 1074.92→1066.02 | 22.23→99.56 | −0.931 |
| 19 | 1155.50→1147.19 | 20.43→92.44 | −0.932 |

Positive-message magnitude changes little, while the negative-message residual grows and points strongly against the positive residual at N64. This identifies an observed negative-message contrast relative to this finite training bank. It does not identify frame-number effects, population bias or sampling variance, nor show that this term causes the native answer errors. The terms precede SiLU and the learned readout, and therefore are not additive logit attributions. Removing E by bank substitution still worsens N64 answers; a predictor-only explanation is insufficient for this proposed rescue. Later prefixes follow each mode's own generated history and cannot automatically be compared as the same state.

No fitting, checkpoint change, new accuracy criterion, `RESULTS.md` update, unseen-answer claim or reasoning-composition claim follows from V15.

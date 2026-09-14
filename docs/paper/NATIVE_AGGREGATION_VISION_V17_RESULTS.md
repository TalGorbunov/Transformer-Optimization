# V17: native local judgments are readable from the existing states

The frozen native norm and vocabulary head correctly distinguished zero from one on every audited original-prompt local state: **9,512 V10 training identities and 13,056 V15 execution occurrences**. Both the conditional binary decision and the unrestricted vocabulary argmax were correct throughout. This supports using the existing native local judgment as a prospective relevance gate. It does not establish that a learned aggregate or native whole answer is correct. The completed [independent analysis](../../outputs/native_aggregation_vlm/v17/local_readout/report_442798/analysis.json) and [report summary](../../outputs/native_aggregation_vlm/v17/local_readout/report_442798/summary.json) passed without numerical binding failures.

The audit used saved hidden states immediately before Qwen2.5-VL-7B-Instruct's final norm, at the last token of the unchanged one-image counting prompt with an empty answer prefix. It loaded only the frozen FP16 norm and head, retaining their exact initial/final tensor identities. The verified token IDs were `0 → 15` and `1 → 16`; probabilities used FP64 normalization over all 152,064 vocabulary entries. “Other” means the probability mass outside those two tokens, not an independently supervised semantic class. QA labels were used only for offline scoring. No VLM, vision, generation or fitting calls were made. [Frozen audit source](../../scripts/audit_native_vision_v17_local_readout.py).

| Inventory and weighting | Correct local decisions, for both decision rules | Mean probability on 0 or 1 |
|---|---:|---:|
| V10 distinct image/question identities | 9,512 / 9,512 | 0.999915 |
| V10 image occurrences across 1,782 unique scenes | 24,624 / 24,624 | 0.999770 |
| V10 weighted epoch, including repeated scene slots | 25,488 / 25,488 | 0.999766 |
| V15 N32, per model and mean mode | 544 / 544 | 0.999871 |
| V15 N64, per model and mean mode | 1,088 / 1,088 | 0.999923 |

Training identities comprise 864 positive and 8,648 negative image/question pairs. Occurrence weighting gives 13,824 positives and 10,800 negatives across unique scenes; the weighted epoch adds 864 positive occurrences from the saturated repeated scene slots. Giving each unique scene equal total weight, or each of the 1,836 epoch slots equal total weight, also gives perfect decisions; their mean numeric masses are 0.999765 and 0.999761. These are alternative weightings of reused features, not additional independent observations.

V15 covers the same 34 previously executed cases, one family per K=0…16 at N32 and N64, for the four fixed V14 models and both learned/bank modes. Its 13,056 execution occurrences represent 1,632 physical scene/image occurrences and 1,574 distinct image/question identities, of which 226 also occur in V10 training. The 16 model/mode/N cells have identical local accuracy and the numeric masses above. Across all executions there are 2,176 positive and 10,880 negative decisions, all correct; equal identity weighting also remains perfect. The eight model/mode executions are not independent data replicates. Every reported K slice and all 272 scene executions retain perfect local decisions. [Full partition and identity evidence](../../outputs/native_aggregation_vlm/v17/local_readout/report_442798/analysis.json).

The category audit retains the harder negative cases:

| Local semantic category | V10 distinct identities, all correct | V15 execution occurrences, all correct | V15 mean p(1) |
|---|---:|---:|---:|
| Positive: queried character in queried room | 864 | 2,176 | 0.972185 |
| Character present, queried room absent | 2,438 | 3,544 | 0.019225 |
| Queried room present, character absent | 2,805 | 3,360 | 0.130118 |
| Neither present | 3,405 | 3,976 | 0.064977 |

The original visual Step labels above 16 did not cause local classification errors. For each model/mode at N32, both Step≤16 and Step>16 are correct on 272/272 occurrences. At N64, the corresponding denominators are 272/272 and 816/816; the latter contains 114 positives and 702 negatives. Mean numeric mass on Step>16 is 0.999867 at N32 and 0.999928 at N64. This is evidence about the native categorical readout on the original V15 images. It does not imply that learned vector messages are invariant to Step labels, and it does not replace the separate V16 intervention.

All eight original V10 profile head batches passed the registered full-vocabulary TV≤0.02 and exact-top1 gates across 288 rows; maximum TV was 2.09×10⁻⁹. All 272 reconstructed V15 fused-global rows also passed in their original complete head batch shapes, with maximum TV zero. The successful audit made 432 norm/head calls over 29,656 rows, including references and binding rows; 22,568 actual local rows were scored. Raw FP16 full-vocabulary logits were retained under the user data root and independently normalized and rescored by CPU report 442798. [Binding comparisons](../../outputs/native_aggregation_vlm/v17/local_readout/report_442798/binding_replays.json).

The campaign consumed **75 allocated GPU-seconds** on one B200: 15 seconds for job 442787, which stopped after eight passing replay calls at its original timing gate, and 60 seconds for completed job 442796. Total executed norm/head calls were therefore **440**, including the failed attempt. The preserved timing failure led to a prospective timing-model amendment; numerical thresholds, inputs and scoring stayed unchanged. The retry projected 347.20 seconds against the unchanged 600-second job cap. CPU preparation 442792 bound the amended source and inputs before execution. [Preserved timing failure](../../outputs/native_aggregation_vlm/v17/timing_failure_442787/) and [GPU completion](../../outputs/native_aggregation_vlm/v17/local_readout/run_442796/summary.json).

The predetermined rule `g = max(0, p(1) − p(0))` is motivated by this distinction between reliable decisions and nonzero negative p(1). On the audited probabilities, it closes every negative gate and leaves every positive gate open. It retains full-vocabulary mass: an uncertain response with little mass on either digit receives little weight. The proposal applies that detached gate to every coordinate of a bounded learned payload, then uses a learned native residual readout; no scalar count is supplied to the decoder. [Prospective semantic-gate proposal](NATIVE_AGGREGATION_VISION_SEMANTIC_GATE_PROPOSAL.md).

The remaining risks are substantive. Positive gates are below one and variable; payload directions can still vary with nuisance content, and the readout must learn complete integer answers and EOS. Future false positives can accumulate with set size, while a false negative closes that image at every later position when the original gate is reused. Such a fixed-question gate cannot automatically reopen for a new reasoning subquestion. Its additional in-hook head computation and cached reuse still need native software validation and a matched all-open control. V17 creates no new aggregation accuracy, fresh-test efficacy, reasoning-composition or attention-novelty claim, and releases no training.

The report binds source, model, cache, selected-checkpoint, QA, raw-output and allocation provenance. Canonical analysis SHA256 is `c59a178d6192623d6c277e8d32cfc51a9fad8f6782c8471ad82fd00f411b06cb`; its frozen CPU plan SHA256 is `3e10ef42ade680f4ff59ab77ca11af8031085f429aa8eb0822dcb2a43ea8e3d4`. This note was prepared from those JSON artifacts without loading tensors or changing frozen sources.

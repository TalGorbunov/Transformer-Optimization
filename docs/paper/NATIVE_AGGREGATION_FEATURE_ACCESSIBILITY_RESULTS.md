# Accessibility of frozen local vision features

**The fixed linear readout recovers every person and 96.56% of rooms on held Step images, but fails the registered strong-accessibility criterion.** This is an offline diagnostic of cached features. It does not establish an information ceiling or the trainability of the native aggregation adapter.

[Original assay 443231](../../outputs/native_aggregation_vlm/identity_join_feature_sufficiency/443231/summary.json) · [Verified label repair 443240](../../outputs/native_aggregation_vlm/identity_join_feature_sufficiency/label_report_443240/summary.json) · [Corrected labelled metrics](../../outputs/native_aggregation_vlm/identity_join_feature_sufficiency/label_report_443240/labelled_metrics.json).

The panel contains all 9,007 unique training-local empty-prefix features from the frozen learned-join cache. Odd Steps provide 4,472 fitting rows; even Steps provide 4,535 evaluation rows. Each side contains 432 distinct image hashes and all 648 question–person–room cells. All question-conditioned copies of an image stay on the same side. The split holds out images and Step values, while reusing people, rooms, artwork and task semantics; it is not a fresh task evaluation.

The readout applies the core's FP32 RMS input transform, then promotes to FP64. One centered ridge solve predicts nine person and six room indicator columns with lambda 1 and an unpenalized intercept. Squared error is averaged over fitting rows and summed over the 15 outputs. Means use fitting rows only. Independent normal-equation residuals are below 2.1e-15; the saved coefficients were reloaded and checked exactly before prediction. No penalty search or retry occurred.

Held-image person accuracy is **4,535/4,535 (100%)**. Room accuracy and jointly correct person/room accuracy are both **4,379/4,535 (96.56%)**. The fitting-question majority control achieves only **95/4,535 jointly correct**. Scores weight unique features equally; occurrence-weighted held joint accuracy is separately 35,114/36,288 (96.76%). Duplicate occurrences are not independent observations.

| Depicted room | Correct room / held rows |
|---|---:|
| Bathroom | 662/762 |
| Bedroom | 739/748 |
| Garden | 708/745 |
| Kitchen | 753/754 |
| Office | 761/764 |
| Park | 756/762 |

The required 4,490/4,535 person threshold passes; the room and joint thresholds fail. The requirements of at least 98% joint accuracy in every question and 95% in every person/room combination also fail. The advantage over the question-frequency control exceeds the required 50 percentage points. The combined decision therefore remains **FAIL**.

No exact local-state collision within a question, or complete scene-state multiset collision preserving multiplicities, was found. This provides no exact ambiguity certificate. The roughly 3.4% room-readout error could reflect this fixed linear readout, regularization or Step generalization; it is not a proven limit on the underlying representation or native decoder.

The original numeric report overwrote person/room stratum labels with same-named metric dictionaries. Its predictions and coverage retained the labels. A separate report repaired all 4,422 stratum rows using distinct `labels` and `metrics` fields, exactly reproducing every original numeric metric, tie, control and decision. The original fit and files remain preserved. The assay used 9 allocated CPU-seconds and the repair 1 second, with zero GPU, model or head calls. No runtime probe or new native fit is released.

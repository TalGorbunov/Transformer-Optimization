# Frozen V5 message readability

Conditional diagnostic after the V5 primary failed. One fixed binary class-balanced ridge per development-selected checkpoint; no new VLM training.

| Checkpoint | Held N32 AUROC / TPR / FPR | Held N64 AUROC / TPR / FPR | Unseen-count AUROC |
|---|---|---|---|
| sum_seed4 | 0.5321 / 0.2663 / 0.1969 | 0.5108 / 0.2663 / 0.2309 | 0.5512 |
| sum_seed5 | 0.4598 / 0.0000 / 0.0000 | 0.4425 / 0.0000 / 0.0000 | 0.5602 |
| mean_seed4 | 0.5938 / 0.0000 / 0.0000 | 0.5391 / 0.0000 / 0.0000 | 0.6271 |
| mean_seed5 | 0.5858 / 0.3518 / 0.2407 | 0.5348 / 0.4070 / 0.3365 | 0.6210 |

Each held familiar cell has54 contexts; fit uses54 separateN16 contexts. All128 unseen-count contexts remain separate. Confusions, denominators, negative classes and perN/K results are in the JSON reports.

Low linear scores do not prove information loss; strong scores do not prove that the model uses these messages or counts correctly. Frame-level metrics are descriptive, without confidence or a new pass/fail threshold.

[Summary](summary.json) · [Family split](family_split.json) · [Provenance](provenance.json)

# V6 saved local-head diagnostic

No refitting, new teacher/model forwards, or external counting. Every saved head scores all452fresh contexts.

| Head | Cell | Frames | Three-way accuracy | Balanced accuracy | TPR | FPR | Other rate | AUROC p1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| control_seed6 | length_N16 | 1728 | 0.7182 | 0.5042 | 0.0764 | 0.0679 | 0.0000 | 0.4458 |
| control_seed6 | length_N32 | 3456 | 0.8226 | 0.5028 | 0.0764 | 0.0708 | 0.0000 | 0.4441 |
| control_seed6 | length_N64 | 6912 | 0.8789 | 0.4979 | 0.0625 | 0.0667 | 0.0000 | 0.4407 |
| control_seed6 | unseen_count_N32 | 2048 | 0.5845 | 0.4847 | 0.0288 | 0.0593 | 0.0000 | 0.5058 |
| control_seed6 | unseen_count_N64 | 4096 | 0.7637 | 0.4854 | 0.0288 | 0.0579 | 0.0000 | 0.5030 |
| control_seed7 | length_N16 | 1728 | 0.7500 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.4941 |
| control_seed7 | length_N32 | 3456 | 0.8750 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.4981 |
| control_seed7 | length_N64 | 6912 | 0.9375 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.5064 |
| control_seed7 | unseen_count_N32 | 2048 | 0.6094 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.4915 |
| control_seed7 | unseen_count_N64 | 4096 | 0.8047 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.4964 |
| aligned_seed6 | length_N16 | 1728 | 0.7465 | 0.5054 | 0.0231 | 0.0123 | 0.0000 | 0.6733 |
| aligned_seed6 | length_N32 | 3456 | 0.8750 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.6042 |
| aligned_seed6 | length_N64 | 6912 | 0.9375 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.5439 |
| aligned_seed6 | unseen_count_N32 | 2048 | 0.6011 | 0.5116 | 0.1025 | 0.0793 | 0.0000 | 0.5253 |
| aligned_seed6 | unseen_count_N64 | 4096 | 0.8047 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.5061 |
| aligned_seed7 | length_N16 | 1728 | 0.7309 | 0.5181 | 0.0926 | 0.0563 | 0.0000 | 0.6732 |
| aligned_seed7 | length_N32 | 3456 | 0.7008 | 0.5403 | 0.3264 | 0.2457 | 0.0000 | 0.5920 |
| aligned_seed7 | length_N64 | 6912 | 0.4585 | 0.5081 | 0.5648 | 0.5486 | 0.0000 | 0.5276 |
| aligned_seed7 | unseen_count_N32 | 2048 | 0.4458 | 0.5228 | 0.8750 | 0.8293 | 0.0000 | 0.5283 |
| aligned_seed7 | unseen_count_N64 | 4096 | 0.3308 | 0.5089 | 0.8013 | 0.7834 | 0.0000 | 0.5102 |

Other is always wrong: balanced accuracy averages positive recall and the fraction of negatives predicted0. It does not use1−FPR as negative recall. AUROC uses unconditional p1 and includes other predictions.

Full2×3confusions, category-specific negatives, perN/K summaries and every frame probability are retained. These correlated frame-level statistics do not establish causal native use or absence of information.

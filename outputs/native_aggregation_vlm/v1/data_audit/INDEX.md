# V1 data semantics diagnostic

Post-launch read-only inspection of the fixed main/profile manifests. The CPU
Slurm diagnostic verifies source QA hashes and recomputes every final gold count;
it summarizes conjunction matches, same-character/wrong-room distractors,
queried-room/other-character distractors, and neither-match frames, including
distributions within each gold count. No image generation, model predictions,
sample changes, or model protocol changes are involved.

Script: `scripts/audit_native_vision_data.py`.
Slurm: `slurm/native_aggregation_vision_data_audit.sbatch`.
Results: `audit_JOB.json`, including per-sample metadata, per-cell summaries,
gold-conditioned histograms, source manifest hashes, and Slurm provenance.

## Completed audit: CPU Slurm 440524

`audit_440524.json` verified all 664 source QA checksums and all gold labels
against the source states. Every frame contains exactly one character. Execution
took 0.78 seconds inside a one-second CPU allocation. No model predictions were
read and no model or dataset protocol was changed.

| Main cell | Samples | Mean target-character frames | Above eight target-character frames | Mean same-character/wrong-room distractors |
|---|---:|---:|---:|---:|
| Train N8 | 90 | 5.978 | 0 | 1.978 |
| Dev N8 | 36 | 5.972 | 0 | 1.972 |
| Test N8 | 100 | 5.780 | 0 | 1.760 |
| Train N16 | 90 | 10.311 | 75 | 6.311 |
| Dev N16 | 36 | 10.278 | 27 | 6.278 |
| Test N16 | 100 | 6.120 | 0 | 2.100 |
| Test N32 | 100 | 5.970 | 0 | 1.950 |
| Test N64 | 100 | 6.070 | 0 | 2.050 |

The newly rendered canonical N16 train/dev sources have a materially different
nuisance distribution from the existing N16 vision test source. The training
excess in same-character/wrong-room distractors is 4.197 frames after averaging
contrasts equally across gold counts; the development excess is 4.164. This
persists after controlling the final answer distribution. Six renderer pixel
parity checks establish matching visual style, not matching sequence sampling.
Therefore N16 test is within the training length range but is not an IID test.
N8 does not show this support violation; this finite-sample audit alone cannot
establish that its train and test generation laws are identical.

Across existing test lengths, final counts stay in 0..8 and mean target-character
occurrences remain near six. The mean number of queried-room/other-character
distractors grows from 0.41 at N8 to 1.70, 4.35, and 9.85 at N16/32/64. The tests
therefore assess filtering and accumulation under more distractors and longer
visual sequences with familiar answers; they do not test unseen output counts.
The visual step labels also extend beyond the fine-tuning training range.

Separate manifest-only checks found no duplicate paths, QA hashes, normalized
sequence/question hashes, or complete ordered image sequences among all 664
main/profile samples, and no overlap with the six V0 IDs. All 15,328 image entries
have SHA256 values and are 512-by-512 RGB. The full source population had 42
cross-split duplicate-content groups, which selection excluded when necessary.

All seven arms share these fixed data and remain comparable as interventions.
Results must disclose the N16 nuisance shift and should not be described as
isolating length alone. A future confirmation should align the generation law
across train/dev/test lengths before making that stronger claim.

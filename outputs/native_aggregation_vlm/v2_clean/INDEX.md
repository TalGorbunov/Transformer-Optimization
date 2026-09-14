# V2 clean data

Generator: `scripts/stage_native_vision_v2_clean.py`.
Generation job: CPU Slurm440730.
Data and immutable manifests: `/mnt/data/gabriele/gnn_transformer/v2_clean`.
The main manifest has train/dev N8/N16 and familiar-count paired test N16/N32/N64.
The count manifest has separately paired unseen-count K9..16 tests at N32/N64.
The profile manifest contains disjoint software-only examples.

All splits use one documented conjunctive-distractor generator law. Paired tests
preserve shorter semantic sequences and add only distractors. Cluster comparisons
across lengths by `pair_id`; step indices are renumbered when new frames are
interleaved. All metadata, room/font/renderer checksums, paired mappings and
per-image hashes are in the data manifests and audit.json.

Independent audit: `scripts/audit_native_vision_v2_clean.py`; reports in
`data_audit/audit_JOB.json`. It recounts every gold, checks selected V1 exclusions,
verifies all image hardlinks and cache hashes, and reconstructs every parent and
anchor subsequence from generated QA states.

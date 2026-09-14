# V2 causal probe data

Sixteen paired N32 sequences have the same question and target, with final counts
2 and 6. Four uniformly chosen negative positions are changed into matching
evidence; the other 28 semantic and image frames stay exactly identical. This
supports descriptive channel-interchange probes and does not select models.

Manifest: `/mnt/data/gabriele/gnn_transformer/v2_clean/causal_probe_manifest.json`.
Schema: version1, `pairs` list; each pair has `pair_id`, `low`, `high`, and four
zero-based `changed_positions`. Low/high are standard sample records with image
paths/hashes and explicit question strings. Sample directories use split `probe`.
All content is disjoint from selected V1 and all preceding V2 main/count/profiles.

Generator: `scripts/stage_native_vision_v2_causal_probe.py`.
CPU Slurm wrapper: `slurm/native_aggregation_vision_v2_causal_probe_stage.sbatch`.
`audit_JOB.json` records gold/QA/pixel-change checks and confirms existing
main/count/profile manifest hashes remained unchanged.

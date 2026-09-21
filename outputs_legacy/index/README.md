# outputs_legacy/index — the tracked INDEX.md files of the frozen legacy run groups

`outputs_legacy/outputs/` is a symlink to the /rg archive
(`/rg/shocher_prj/tal.gorbunov/archive/outputs/outputs_legacy/outputs`) since the 2026-09-21
migration. Git cannot track a path that passes through a symlink, so the seven `INDEX.md`
files that were tracked at `outputs_legacy/outputs/<group>/INDEX.md` live here as
`index/<group>/INDEX.md`. The on-disk copies on /rg are unchanged and identical; paths cited
in RESULTS.md still resolve. Nothing else under `outputs_legacy/` is tracked (`.gitignore`
ignores the tree; these files are force-added).

# superquery — divide-and-conquer tree readout (MAIN DIRECTION, 2026-08-05)

Professor's proposal: fenced [frame+q-replica] blocks (posreset, fence NEVER lifts) +
"superquery" question replicas after the blocks that attend only their children's spans
at all layers. Flat version = star graph fan-in N (predicted to fail counting: supercarrier
probe Aug-1, coarse≈full, tally 0.25@8→0.07@64). Our extension: bounded fan-in TREE
(b-ary superquery hierarchy) — over-squashing theory prescribes bounded-degree readout;
in-model tally works at fan-in ~8 (0.87). Sweep measures the capacity curve c(b).

## Probe 1+2 (flat + trees in one script) — scripts/superquery/probe_tree.py
- One forward/sample carries ALL arms (flat, b=2,4,8): blocks shared, per-SQ-row masks.
- Park steps task, N=8/16/32/64 (120/120/80/40), read layers 12/16/20, feats mean|last.
- Failure-signal ladder: per-level subtree-count probes (acc/±1/R²/gate-d′/tally),
  hop survival implicit in per-level rows, top-node gold regression + composed tally.
- Job 128758 (l40s-public, 2h_2g). Output: outputs/superquery/<ts>_tree_128758/
  {nodes.csv, top.csv}.

## Interpretation guide
- level-1 count acc vs fan_in = the aggregation-capacity curve (the headline).
- level-l acc >> flat at same N = divide-and-conquer works frozen.
- If level-1 good but level-2 drops: hop-survival failure (info dies in transit).
- If all levels fail even b=2: replica loci don't expose verdicts to SQ reads
  (routing/read failure, not aggregation) — check gate d′ first.

Related in-flight: v7 arm-C train (128757, L16 full-fence + v5 counter scans).

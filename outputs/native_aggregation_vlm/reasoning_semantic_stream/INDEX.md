# Cosmos semantic streaming software check

CPU check442952, GPU442954, independent report442955 all passed. GPU cost74seconds. Six untrained eight-token trajectories atN16/N64, bare/zero/active.

- [Independent analysis](report_442955/analysis.json)
- [GPU artifacts](run_442954/summary.json)
- [Frozen plan](check_442952/plan.json)
- [Design](../../../docs/paper/NATIVE_AGGREGATION_REASONING_SEMANTIC_STREAM.md)

All164originrows and64headreplays have zeroTV, exactbare/zero-U logits/tokens, fixedoriginalgates and42all-layerKVtransitions pass; no cached/full failures. This establishes bounded native software behavior, not reasoning accuracy or composition.

# Fixed-budget training dynamics review

The three runs failed their registered training screen at step 600. Their logs do not establish optimization convergence. Complete-cycle loss still declined slightly near the end while the learning rate approached its fixed floor. The architecture branch remains closed; this review does not release continuation or another variant.

Only existing JSON was read. Each of the 600 updates contains eight length-matched pairs. The recorded order was independently reconstructed with the original persistent Random(24): 4,800 pair presentations comprise 88 complete cycles of all 54 pairs, followed by 48 pairs of cycle 89. Pair IDs, scene IDs, cycle labels, full targets, strict prefixes, learning rates and loss reductions were checked against all logs. The three orders match exactly. All 21,334 target positions per run are accounted for.

For each pair, the two recorded scene losses are averaged; each scene loss is the mean full-name-plus-EOS cross-entropy over its target positions. Each complete cycle therefore weights every one of the 108 scenes equally. These are online losses before the respective updates, not evaluation of one fixed checkpoint.

| Run | Cycle 1 CE | Cycle 44 CE | Cycle 88 CE | Cycles 79–88 change | Last-ten OLS slope/cycle |
|---|---:|---:|---:|---:|---:|
| Parent product (443375) | 2.47803 | 0.53357 | 0.39414 | -1.70% | -0.0007108 |
| Additive (443376) | 2.47418 | 0.55222 | 0.47901 | -0.62% | -0.0003303 |
| Readout-only query product (443416) | 2.48060 | 0.52607 | 0.39466 | -1.59% | -0.0006562 |

Cycle 1 spans updates 1–7 (learning rate 0.00002–0.00014); cycle 44 spans 291–297 (0.000584–0.000601); cycle 88 spans 588–594 (0.00001029–0.00001116). Across cycles 79–88, the learning rate declines from 0.00005241 to 0.00001029, versus the earlier peak of 0.001. Downward adjacent-cycle changes occurred in 6/9, 6/9 and 7/9 comparisons, respectively. First-five versus last-five tail means are shown in the bound JSON; all three decline. This is descriptive, not a statistical trend test.

| Run | Last complete cycle: first-token CE | EOS CE | Second-name-token CE | Partial cycle 89 CE (96 scenes) |
|---|---:|---:|---:|---:|
| Parent product (443375) | 0.87023 | 0.00007255 | 0.00006316 | 0.38791 |
| Additive (443376) | 1.05853 | 0.00003544 | 0.00009389 | 0.47591 |
| Readout-only query product (443416) | 0.86329 | 0.00005631 | 0.00004424 | 0.38533 |

First-token and EOS columns each average one token per scene. The continuation column covers the second name token of Sandra/Noah (24 token occurrences per complete cycle). Their objective contributions, including the original per-scene target-length division, are also retained. At cycle 88, essentially all remaining mean CE comes from the first name token. This addresses teacher-forced token losses only; it does not measure whole native generated answers.

The final partial cycle spans updates 595–600, contains a different subset of 48 pairs, and is kept separate. Its smaller mean cannot be interpreted as improvement over a complete cycle.

The fixed 600-step cap is a valid resource-controlled screen and makes these schedules comparable. It was not a stopping rule based on vanishing gradients or independently demonstrated convergence. Residual loss decline is compatible with continued learning, but the decaying step size can also make an unfinished fit look nearly flat. Training CE alone cannot determine whether more updates would cross the first-token/family thresholds, whether the observed basin is a capacity limit, or whether generalization would improve. The small difference between the two product endpoints does not establish that moving the query solved training. No new fit, tensor load, native head, VLM, optimizer call, or Slurm job was used.

All cycle values and exact report/run/log/order/plan JSON hashes: [analysis.json](analysis.json), SHA256 `65757042bd6716627c115f712a607979476c5134b138d73524c890ca52dea07f`.

The fixed 6,000-update joint-code oracle fits all 108 training first queries and all 18 families. Independent report 443511 passed. The earlier fixed checkpoints at 600 and 2,000 updates each scored 36/108 with no complete families; only the final checkpoint was the registered decision. This is a training fit with privileged per-image semantic codes, not evidence of native visual generation, generalization, or grokking.

The JSON review independently reconstructs all 48,000 pair presentations, 213,330 target positions, 888 complete cycles of 54 pairs, and the final partial cycle of 48 pairs. Each cycle averages both N8/N16 scenes per pair, using the recorded full-target mean CE. Values are observed before different updates within a cycle; they are not frozen-checkpoint rescoring. First-token and EOS means each use one token per scene. Continuation means use the second name token for Sandra/Noah only (24 tokens per complete cycle). The saved analysis also retains their target-length-weighted contributions to the actual objective.

| Complete cycle | Updates covered | Full-target mean CE | First-token CE | EOS CE | Continuation CE |
|---|---:|---:|---:|---:|---:|
| 1 | 1–7 | 2.48615 | 4.67660 | 0.002923 | 3.68671 |
| 88 | 588–594 | 0.61860 | 1.33032 | 0.0000411 | 0.0000239 |
| 296 | 1992–1998 | 0.46377 | 1.00910 | 0.0000161 | 0.0000243 |
| 400 | 2694–2700 | 0.33585 | 0.72697 | 0.0000139 | 0.0000224 |
| 600 | 4044–4050 | 0.13471 | 0.28227 | 0.0000117 | 0.0000220 |
| 888 | 5988–5994 | 0.03768 | 0.08014 | 0.0000117 | 0.0000242 |

The later improvement is predominantly first-token learning. Fixed 500-update windows show first-token CE falling from 1.09219 at updates 1501–2000 to 0.71132 at 2501–3000, 0.36023 at 3501–4000, and 0.08439 at 5501–6000. The available checkpoint evaluations do not locate the exact accuracy transition. Frozen endpoint first-token NLL is 1.22215, 1.02382, and 0.08028 at 600/2000/6000.

Gradient norms remain substantial during that decline. These are the logged total minibatch CE norms before clipping at 1, not per-item gradients. Mean norm / clipped-update fraction is 1.25 / 64.0% at 1501–2000, 1.86 / 86.6% at 2501–3000, 1.79 / 90.8% at 3001–3500, and 0.866 / 25.2% at 5501–6000. Overall 4,009/6,000 updates were clipped. The last ten complete cycles still reduce first-token CE by 1.67% and full-target CE by 1.47%; neither these losses nor clipping establish convergence.

The original 600-update control scored 38/108, zero families, with endpoint NLL 1.38861. The two runs have identical initial/code/global/native identities, the exact original 4,800-presentation order prefix, and identical first 50 losses and learning rates. Thereafter their annealing differs: at update 600 the short run uses 0.00001, versus 0.000979274 in the long run. Thus this calibrates the horizon and learning-rate trajectory together. It refutes treating the earlier 600-update failure as proof that this privileged-code/readout recipe cannot fit; it does not isolate extra steps alone or establish that the visual encoder will fit.

All calculations use saved JSON only. Input hashes, every complete and partial cycle, 500-update gradient/loss windows, endpoint scores, and short-run comparison are in analysis.json. No tensors, model/head calls, optimizer execution, jobs, original-file changes, or new experiment releases were involved.

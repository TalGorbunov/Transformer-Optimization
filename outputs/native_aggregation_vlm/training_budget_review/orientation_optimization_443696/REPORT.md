# Descriptive optimization review

GPU training JSON only. The current orientation independent audit remains separate. These losses do not establish convergence or authorize continuation.

| Run | Updates | Scene CE | First-token CE | EOS CE | Mean preclip norm | Clipped |
|---|---|---:|---:|---:|---:|---:|
| run_product_443540 | 501–1000 | 0.3490889 | 0.7610053 | 3.053512e-05 | 1.692608 | 0.832 |
| run_product_443540 | 1501–2000 | 0.02637477 | 0.06238488 | 1.645331e-05 | 0.7071151 | 0.212 |
| run_product_443540 | 2501–3000 | 0.0008859702 | 0.001978068 | 1.519731e-05 | 0.02425086 | 0.000 |
| run_product_443540 | 3501–4000 | 0.0002613584 | 0.0005591983 | 1.662369e-05 | 0.007836882 | 0.000 |
| run_product_443540 | 4501–5000 | 0.0001380466 | 0.0002791002 | 1.835488e-05 | 0.003978391 | 0.000 |
| run_product_443540 | 5501–6000 | 0.0001130304 | 0.0002212603 | 1.994482e-05 | 0.003326761 | 0.000 |
| run_additive_443541 | 501–1000 | 0.4868714 | 1.070298 | 2.125124e-05 | 1.156461 | 0.584 |
| run_additive_443541 | 1501–2000 | 0.1745686 | 0.385937 | 1.591202e-05 | 2.27296 | 0.874 |
| run_additive_443541 | 2501–3000 | 0.04847278 | 0.1086684 | 1.224761e-05 | 1.507751 | 0.556 |
| run_additive_443541 | 3501–4000 | 0.009363515 | 0.0207518 | 1.072943e-05 | 0.4544675 | 0.092 |
| run_additive_443541 | 4501–5000 | 0.003158876 | 0.006959435 | 1.090384e-05 | 0.1557064 | 0.000 |
| run_additive_443541 | 5501–6000 | 0.002153529 | 0.004755513 | 1.096141e-05 | 0.1014678 | 0.000 |
| run_product_443677 | 501–1000 | 0.5473312 | 1.202636 | 3.522411e-05 | 1.029261 | 0.492 |
| run_product_443677 | 1501–2000 | 0.5297385 | 1.165659 | 1.740925e-05 | 0.9843201 | 0.436 |
| run_product_443677 | 2501–3000 | 0.5194543 | 1.143931 | 1.323778e-05 | 0.9407998 | 0.382 |
| run_product_443677 | 3501–4000 | 0.5120657 | 1.128307 | 1.169003e-05 | 0.9383688 | 0.392 |
| run_product_443677 | 4501–5000 | 0.5066489 | 1.115768 | 1.115709e-05 | 0.9402944 | 0.378 |
| run_product_443677 | 5501–6000 | 0.5040382 | 1.11034 | 1.109204e-05 | 0.9340297 | 0.350 |
| run_additive_443678 | 501–1000 | 0.5463714 | 1.200519 | 2.204092e-05 | 0.9831055 | 0.420 |
| run_additive_443678 | 1501–2000 | 0.5309914 | 1.168246 | 1.385358e-05 | 0.9443856 | 0.370 |
| run_additive_443678 | 2501–3000 | 0.5203193 | 1.145786 | 1.147412e-05 | 0.9090477 | 0.322 |
| run_additive_443678 | 3501–4000 | 0.512474 | 1.129237 | 1.070772e-05 | 0.910207 | 0.340 |
| run_additive_443678 | 4501–5000 | 0.5068552 | 1.116261 | 1.062018e-05 | 0.9108648 | 0.324 |
| run_additive_443678 | 5501–6000 | 0.5044721 | 1.111369 | 1.065509e-05 | 0.9064021 | 0.304 |

Each complete54-base-pair cycle contains108 scene presentations. Adjacent cycles cover both orientations in the216-scene study and repeat the same original108 support in its comparator. All444 complete two-cycle groups are saved; final48-pair partial cycle889 remains separate. The same6000 LRs,48000 base-pair order and213330 target IDs are verified across all four logs.

First and EOS CE each average one token per scene; continuation CE averages only interior name tokens. Full CE retains the recorded mean over each complete target. These are pre-update observations from different minibatches, not endpoint rescoring. Global preclip gradient norms do not identify an individual factor or loss gradient. Payload maximum alone does not measure saturation; no saturation fractions or per-module gradients were logged.

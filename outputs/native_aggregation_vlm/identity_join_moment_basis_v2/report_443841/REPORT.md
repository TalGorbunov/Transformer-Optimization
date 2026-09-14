# Matched moment-basis cached training

Unchanged numerical gates failed; complete comparisons and replay tensors are retained. Both arms used the same fresh seed24 initialization, six trainable tensors (1,404,192 coordinates), native feature bank, original global conditioning statistics, paired presentation order and full-name-plus-EOS CE. A=sum(.5a), B=sum(.5b), C=sum(.25ab), P=A*B-C. The 288-coordinate readout receives [A,B,C] or [A,B,P]; both use delta=U SiLU(Wagg*aggregate+b+q), with the actual query retained in both factors.

| Basis | Correct /216 | Original /108 | Flipped /108 | Complete families /18 | Saved GPU screen |
|---|---:|---:|---:|---:|---|
| within | 77 | 42 | 35 | 0 | FAIL |
| cross | 73 | 40 | 33 | 0 | FAIL |

Only step6000 determines acceptance:206/216 pooled,103/108 per orientation and16/18 complete12-context families. All outcomes, strata and training losses are retained. These are cached training first queries. No result automatically releases native or fresh evaluation.

The bases preserve the same information for fixed factors in real arithmetic. This does not establish identical finite-width function classes, gradients or numerical conditioning. The cross basis has different length dependence and possible subtraction cancellation; bound per-capture signed statistics and FP64 discrepancies are descriptive and do not alter fidelity gates. No semantic roles are assigned to learned factors. Neither success nor failure establishes convergence, general reasoning or architectural incapacity.

The final checkpoint was serialized, reset to the fresh state and reloaded. Audits check optimizer state, native weights, features, statistics, all6000 CE reductions, actual captured moments/readout and independent ordered-pair value/gradient fixtures. No optimization trajectory was replayed. Prior failures remain unchanged.

Allocated GPU cost: 144 seconds. Each arm used6014 core/conditioning/norm/head calls and213546 head rows. The preserved original metadata failure consumed5 allocated CPU seconds and zero GPU seconds. VLM/vision calls were zero. CPU audit covered44 captures and28 endpoint head batches/432 rows, with exact argmax and TV<=0.02. Maximum TV was 0.003786527.

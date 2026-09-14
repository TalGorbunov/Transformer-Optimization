# Fixed local-conditioning diagnostic

Independent computation and provenance audits passed. These are cached training first tokens, not native whole answers.

| Mean | Correct /108 | Complete families /18 | Fixed screen |
|---|---:|---:|---|
| global | 64 | 0 | FAIL |
| question | 56 | 0 | FAIL |

Both arms used the same original initialization,600 full-name-plus-EOS CE updates and global pooled scale. Each retained607 actual native norm/head batch calls and21442 rows; zero VLM/vision calls. Only the mean varies between arms. Centering and scale relative to the old uniform model remain a combined intervention.

A screen pass permits preparation of a separately audited native evaluation of this unchanged endpoint. It does not establish name completion, EOS behavior, generalization or reasoning composition. The question arm retains a fixed six-question diagnostic lookup; the global arm does not.

Allocated GPU cost: 58 seconds. CPU replay:14 final norm/head batches, with fixed TV<=.02 and exact argmax.

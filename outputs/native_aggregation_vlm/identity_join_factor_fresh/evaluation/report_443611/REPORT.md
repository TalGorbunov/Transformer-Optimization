# Fixed fresh N16 factor evaluation

Both original step6000 checkpoints were evaluated on every fixed fresh N16 context. No fitting, prompt/statistic change, checkpoint selection, N32/N64 evaluation or additional native-head replay occurred.

| Arm | Panel | Whole answer + EOS | First token | Complete triples | Panel criterion |
| --- | --- | ---: | ---: | ---: | --- |
| product | A | 64/108 | 64/108 | 15/36 | FAIL |
| product | B | 14/108 | 14/108 | 0/36 | FAIL |
| product | C | 7/54 | 7/54 | 0/18 | FAIL |
| additive | A | 49/108 | 49/108 | 11/36 | FAIL |
| additive | B | 14/108 | 14/108 | 0/36 | FAIL |
| additive | C | 7/54 | 7/54 | 0/18 | FAIL |

A and B each retain the literal98/108 answer and33/36 triple thresholds; together these require at least99 correct answers. C retains its separate49/54 and16/18 criterion.

A/B qualification: product=False; additive=False. Neither endpoint qualifies: stop before N32/N64.

All names, questions, variants, orientations, noncanonical outputs, premature EOS and truncations are retained in the row and stratum artifacts. Panel A separates the original and opposite training orientations. Paired product/additive transitions are descriptive.

Every one of 1240 emitted-prefix factor/conditioning/native writes and raw argmaxes passed the independent audit. Head fidelity is inherited from 20 profile calls/260 native rows, maximum TV 5.02316743e-06, with exact argmax. This report performed zero head calls.

Fresh inference used 680 allocated GPU-seconds and 1240 model/norm/head calls; all540 vision prefills are counted.

These fixed panels test fresh realizations, person compositions and question transfer at N16. An ordinary matched joint-input baseline remains absent. These results do not establish longer-length reliability, a new attention mechanism, superiority to ordinary joint modeling, or reasoning composition.

# Fixed factor native training evaluation

This report evaluates natural full-name plus EOS generation on the same 108 training contexts per arm. It performs no fresh-data or development evaluation.

| Arm | Whole answer + EOS | First token | Complete six-context families | Registered training screen |
| --- | ---: | ---: | ---: | --- |
| product | 108/108 | 108/108 | 18/18 | PASS |
| additive | 108/108 | 108/108 | 18/18 | PASS |

Both fixed 6000-step checkpoints were evaluated; none was selected by these results. Product/additive paired transitions: {'both_correct': 108}.

CPU native norm/head replay covered only the profiles: 20 calls and 260 native rows; maximum full-vocabulary TV 5.02316743e-06, all argmaxes exact. Main CPU head replay: zero. Every main factor, global conditioning, FP16 cast, raw argmax, native history, mask and position was independently checked.

The native stage used 342 allocated GPU-seconds, 500 native model/norm/head calls, and no extra GPU head calls. All raw answers, premature EOS, truncations and observed continuation prefixes are retained. Full-layer KV tensor equality was not asserted.

A passing training screen establishes native execution/trainability on these seen contexts only. Fresh composition, length transfer, superiority of multiplication, general aggregation, and reasoning composition require separate evidence; this report releases none of those claims.

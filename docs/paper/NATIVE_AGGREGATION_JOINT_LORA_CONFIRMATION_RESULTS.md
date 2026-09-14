# Joint-image LoRA: reserved N16 result

The fixed ordinary joint-image LoRA learns all216 training answers, and scores260/270 (96.30%) on reserved seed91726342. It nevertheless **fails the predeclared A/B consistency gate**. The thresholds are unchanged, and no N32/N64 run is released for this checkpoint.

| Panel | Complete answer + EOS | Fully correct triples | Gate |
| --- | --- | --- | --- |
| A: trained groups/questions, new scenes | 104/108 | 32/36 | Fail (33 triples required) |
| B: new relevant groups, trained questions | 104/108 | 32/36 | Fail (33 triples required) |
| C: new groups and held questions | 52/54 | 17/18 | Pass |

C changes both grouping and questions; it does not isolate question generalization. Both room orientations were trained. All these scenes have six relevant frames: longer N would add distractors and positions, not increase relevant aggregation load. Earlier frozen-joint/small-adapter fresh results used a different seed and different training budgets; this is not their paired accuracy comparison.

GPU443948 completed0:0 in334GPU-seconds (332.432s measured), within900. Actual597 model/language/norm/head calls and270 vision calls, no fitting or extra GPU heads. Run summary SHA `1d5957b527f62e727b790ae30e3129a633edfe49d27b1433b9e40dcd7a156cfb`.

Independent CPU443951 completed0:0 in193CPU-job-seconds (191.710s measured), within1800. All270 records and597 native-head replay rows passed, maximum full-vocabulary TV2.7576331550792607e-5, zero argmax differences. Audit summary SHA `19dde19fbb8c85142e72dc58ef4bcca48e24c19651e7240dd8eee8b178c944a7`. Root and independent V10 reconstructed source/checkpoint/order/criterion/accounting joins. Cumulative LoRA GPU work is2586seconds, including original failed34, successful profile63, main2155 and confirmation334.

The controlled identity-join extension remains a useful diagnostic. It does not establish the requested general aggregation advance or an original MMReD benchmark result. The next fixed comparison uses the original four-task MMReD Vision pilot, ordinary joint images versus normalized and mass-bearing32-slot memory, under common training exposure and measured inference costs.

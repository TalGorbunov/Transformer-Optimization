# Paired-minibatch visual-factor training

Independent computation and provenance audits passed. Both locked interactions used the same original unfitted tensors, native feature bank and original global conditioning statistics. Only the order of existing presentations changed, placing original/flipped counterparts together; targets,6000 learning rates and full-name-plus-EOS CE weighting were unchanged.

| Interaction | Correct /216 | Original /108 | Flipped /108 | Complete families /18 | Saved GPU screen |
|---|---:|---:|---:|---:|---|
| product | 77 | 39 | 38 | 0 | FAIL |
| additive | 73 | 37 | 36 | 0 | FAIL |

Only the fixed step6000 endpoint determines acceptance:206/216 pooled,103/108 within each orientation and16/18 complete12-context families. All outcomes, orientation/length/name/question/family strata and training losses are retained. These are cached training first queries; neither a pass nor a failure establishes native whole-answer performance or fresh generalization. No result automatically releases native or fresh evaluation.

The final checkpoint was serialized, reset to the unfitted state and reloaded. Independent audits check optimizer state, frozen selectors, native weights, features, statistics, actual captured factor arithmetic, and all6000 scene-balanced CE reductions. They do not rerun optimization or certify convergence. Prior failed results are unchanged.

Allocated GPU cost: 153 seconds. Each arm used6014 core/conditioning/norm/head calls and213546 head rows. VLM/vision calls were zero. CPU audit covered44 captures and28 endpoint head batches/432 rows, with exact argmax and TV<=0.02. Maximum TV was 0.003611808.

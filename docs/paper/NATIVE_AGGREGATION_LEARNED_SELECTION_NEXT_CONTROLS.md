# Next controls for learned native selection

**Held conditional design; no execution or release.** Complete the six fits and independent report first. The existing [three-arm experiment](NATIVE_AGGREGATION_LEARNED_SELECTION_PROPOSAL.md) compares known weighting functions within one parallel architecture. It does not establish an advantage over ordinary joint-image adaptation, increased information bandwidth, or reasoning composition.

## Interpret the current result first

- **Clip passes both comparative and practical gates:** investigate a useful parallel model; attribute the result to the trained condition, not specifically to exact zeros. Clipping also changes saturation and optimization.
- **All arms perform well without a clip advantage:** retain clip as the prospectively fixed representative, provided it passes the absolute accuracy targets. Claim evidence for the shared pipeline, not a distinctive selection primitive.
- **Clip fails the absolute targets, including success confined to a control:** stop this follow-up as specified. Preserve complete triples, errors and gate geometry; do not select another operator from the test results or tune thresholds. Failure of this fit does not prove that the native states lack association information.

## First control: ordinary joint-native adaptation

Compare **parallel clip versus joint clip**, two paired new seeds, with fresh initialization in both. Use the same frozen Qwen backbone, 1,041,697 trainable parameters, rank-96 bounded payload, learned clipped score and native residual cast/norm/head. The joint arm receives all images in the ordinary native joint prompt. Feed its current joint hidden state to both the global and singleton-local inputs of the identical core. Parallel retains N image streams and a text-only global stream. Neither receives inclusion decisions or local labels.

Both arms use **CE only**, equally averaged across each scene's complete native name-plus-EOS sequence. Reuse the same 6,048 training contexts, canonical paired presentation order, 12 epochs, 4,536 updates, optimizer and fixed final endpoint. Pairing now determines order only. Joint global states differ across lengths: copying the present identical-global residual-consistency objective would violate its contract. Differences from the preceding regularized fits cannot isolate architecture.

Independently cache each route's actual native states and verify native deployment/replay, including joint image positions and every strict prefix. Equal parameter counts do not imply equal effective function classes. Joint attention can already combine images, whereas parallel lower layers cannot. Report actual vision work, decoder lengths, KV memory, latency and allocations; a single batched call is not equal compute.

Use one final 108-scene development evaluation descriptively. Stage **36 fresh complete families**, six per each of the three seen and three held room pairs, with six balanced trios giving each name two occurrences per room pair. Three variants and N32/N64 yield **216 contexts**, 54 per regime/length. Exclude all previous complete contexts; preserve native Step labels and the existing insertion law. Proposed screen: parallel gains at least 3/54 N64 answers in each regime and seed, loses at most 2/54 at N32, and reaches 49/54 seen and 44/54 held at both lengths. Report complete-triple success and paired family intervals regardless. Freeze these coarse thresholds before execution.

Release only after measured joint/parallel cache, training and native timing projections fit a separately registered cap. Pool worst route timings; include four fits, all evaluations, failures and I/O. Do not assume current parallel timings bound joint prompts. An advantage would support this evidence-encoding arrangement at its measured cost. It would not establish a new pooling operator or general bandwidth; independent encoding and generative fusion already have precedent in [FiD](https://aclanthology.org/2021.eacl-main.74/).

## Then: a small same-backbone reasoning test

Proceed only after a useful joint-control result and new Cosmos software/cost gates. Repeat **two routes × two seeds**, evaluating each checkpoint with direct and official reasoning global prompts on the same Cosmos backbone. Rebuild all features; never compare Qwen direct accuracy with Cosmos reasoning accuracy.

Use identical training targets in both routes: `0.5 * direct scene-mean CE + 0.5 * natural trace scene-mean CE`, without consistency. Before fitting, attempt one native joint Cosmos trace for every variant of 12 predetermined training contrasts, balanced across training room pairs and names: 36 attempts, maximum 4,096 tokens, no retries or supplied thoughts. Require 24 completed correct traces and coverage of every room pair and name. Archive failures. Freeze accepted traces and a common full-prefix token schedule; all assistant tokens receive CE and fusion throughout. Parallel local prompts stay direct while receiving broadcast thoughts; check this grammar mismatch. Correct final answers do not certify the thoughts.

Start with six fresh families, one per seen/held test room pair, globally name-balanced within each regime: 36 contexts. Report route differences under each policy and reasoning-minus-direct within each route. Proposed screen, each seed: parallel gains 2/18 N64 answers over joint under **both** policies, N32 loss at most 1/18; parallel reasoning additionally gains 2/36 over direct. Small denominators and room/name confounding limit this screen.

Final-norm writes remain outside KV: answer CE cannot credit earlier residuals. Trace CE trains those predictions directly. On four predetermined reasoning cases, suppress one early residual from the same history, then freely continue; separately force the factual next token and verify the subsequent state/logit identity. Changed answers only through changed emitted tokens support token-mediated influence, not continuous memory or logically beneficial thought.

Reserve 144 reasoning trajectories at the full 4,096-token bound, plus direct runs, interventions, teachers, features and fits. Measure full-length memory and decode costs first; a short software smoke cannot authorize this expense. Stop if the fixed campaign cannot fit. No handcrafted scratchpad, answer oracle, prompt search or growing architecture grid is released here.

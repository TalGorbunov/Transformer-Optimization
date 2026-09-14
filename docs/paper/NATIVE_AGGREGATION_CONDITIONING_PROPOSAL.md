# Fixed input conditioning: global versus question means

The oracle native-readout diagnostic passed all 108 first tokens and 18 complete families in independent report 443304. The uniform model still fails the join; its 97–98% saturated payload coordinates and large common component motivate a conditioning intervention. Those observations do not establish the cause.

**Before any source freeze or fit, specify two parallel arms:** subtract a global training mean or a question-specific training mean, using the same global scale. This replaces the initial single-arm design. The global arm tests conditioning without a deployment question lookup; the question arm is a richer diagnostic. This is an optimization test of an existing architecture, not a new attention mechanism or added expressiveness. No further normalization variants are released.

Reuse the uniform control's exact 108 training contexts, 54 N8/N16 pairs, 18 families, six questions, seed 24, initial parameter bytes, and 4,800 paired presentations. Its retained 36/108 first/whole training answers and 0/18 complete families are a descriptive reference; no baseline refit. No answer labels enter conditioning statistics. Labels remain the ordinary supervised name-plus-EOS targets.

Let `x = RMS(h)` use the existing FP32 tokenwise rule with epsilon `1e-6`. Collect all 1,296 **original cached training empty-prefix local-state occurrences**, in frozen scene and image order, including duplicates and N16 background occurrences. Do not estimate statistics from fitted-run native captures, development, tests, unique-image reweighting, or other prefixes. Compute in FP64:

- `mu_global`: mean of all 1,296 occurrences.
- `mu_q`: each question's mean of its 216 occurrences.
- A single shared scalar `s = sqrt(mean((x.double() - mu_global)**2) + 1e-6)`, where the denominator is `1296 * 3584`.

Cast both sets of means and `s` to FP32 once and freeze them before fitting. Replace only the local encoder's already-RMS-normalized input, in FP32:

- **Global:** `(x - mu_global) / s`.
- **Question:** `(x - mu_q) / s`.

A scoped fixed pre-hook on the unchanged local linear layer may implement this placement. It must not apply RMS twice, touch the global query path, or persist beyond its call. Use the same means and scale at every teacher-forced or emitted prefix. Padding may transform internally but must retain exactly zero gates/messages. Save occurrence ownership and all statistics with hashes. The question arm must reject unsupported questions; the global arm's transform must not select statistics by question.

Both arms retain rank 96, tanh, uniform `.5` weights, SUM, SiLU, and native residual injection: 1,041,697 parameter coordinates, of which 1,041,600 are trainable and 97 fixed selector coordinates. Selector weight remains zero and bias remains `.5`. No trainable parameters are added. The shared calibration packet contains both means; the deployed global arm requires only 3,584 mean coordinates and one scale. Its affine transform could later be folded into the local weight and bias (`W'=W/s`, `b'=b-W*mu/s`); that numerical conversion is not performed or assumed exact here. Initial zero-U output identity remains exact, although U gradients change.

Each arm trains for exactly 600 AdamW updates: learning rate `.001`, 50-step warmup, cosine decay to `1e-5`, weight decay zero, clipping at one, and inherited optimizer defaults. Each update uses eight intact pairs/16 scenes in the original persistent `Random(24)` order. Optimize the original **mean-scene, mean-token full-name-plus-EOS CE**, including first, continuation and EOS positions, with consistency coefficient zero. This is not the oracle's first-token-only objective.

The frozen order entails **21,334 training norm/head rows**, maximum actual batch 44 target positions (absolute contract 48). Retain every update, CE position/scene reductions, prefixes, frozen-parameter checks and unique progress files. Save existing forward captures at steps 1, 2, 32, 128, 300 and 600, without extra forwards.

The first stage uses cached frozen states and the actual installed native FP16 norm/head, with zero VLM/vision calls. Each arm makes 600 training batch calls and seven fixed-final cached first-query batch calls, at most 16 final scenes per batch: **607 norm/head calls and 21,442 total rows per arm**. Reset and restricted-reload checkpoint 600. No development, checkpoint selection, extensions, retries or new seeds. Retain both arms and every failure. Save all 108 full-vocabulary FP16 logits, argmax predictions, native cast/add inputs and conditioned payloads per arm.

A fixed **resource screen** requires at least 103/108 correct first tokens and 16/18 families with all six first tokens correct. It is neither native whole-answer accuracy nor the original training criterion. Failure ends that arm; it does not prove that actual native generation would fail. A pass releases preparation of native evaluation of that exact checkpoint and statistics on 108 training and 54 reused development contexts. Any native GPU job needs its concrete audited integration and resource release first. No refit is allowed. Native criteria remain 103/108 first/whole answers and 16/18 training families; 49/54 first/whole answers and 16/18 development families.

Allow one CPU source check of at most 90 seconds, **one B200 head-only attempt per arm of at most 90 allocated seconds**, at most two concurrent project GPUs, and one joint CPU report of at most 300 seconds. All jobs use four CPU cores and 16 GB RAM. Total GPU ceiling: 180 seconds including failures. No profile. Bind the successful oracle report and its preserved precision diagnosis, uniform inputs/initialization, all new sources and statistics before either fit. CPU fixtures cover transform placement, global independence from question labels, shared scaling, frozen state/gradients, padding and hook removal.

Independently replay FP32 core captures with fixed `atol=rtol=1e-4`; verify captured FP16 cast-before-add exactly; replay the native final head with TV at most `.02` and exact argmax. Recompute NLL in FP64 against saved GPU FP32 values with unchanged `atol=rtol=2e-6`. Preserve all allocations and failures. Data and checkpoints use their authorized `identity_join_conditioning` subdirectories.

Global success would support a conditioning treatment without deployment question lookup; unseen-question transfer remains untested. Question-only success would show that this richer conditioning suffices under the fixed recipe. Neither outcome identifies saturation as the sole cause, establishes a capacity limit, or demonstrates generalization or reasoning composition. Both failures end this comparison without a normalization sweep.

# Native aggregation: decisions after V3

2026-09-10. Written before V3 outcomes. Conditional design, not a preregistration;
additional experiments need a recorded protocol and bounded budget under the
user's continuing authorization. Preserve V3 gates and publish failures.

## What would justify advancing

A PRE advantage must pass the existing two-seed screen, survive the separate
marginal-preserving challenge, and replicate on fresh scenes. Familiar-count
length robustness, unseen-count extrapolation and reasoning reuse are separate
claims. V2's hidden-only control matched its extra-read adapter; ordinary
same-site adaptation remains a required comparator.

## A decisive composition experiment

Use MMReD Vision: “Which of Mary and John appears in the kitchen more often?
How many garden frames contain that person?” Exclude ties. Balance selected
identity and both kitchen/garden counts; make the selected garden count
independent of the winning margin and identity. Hold out scenes and combinations
of component questions. Fixed-person and first-count shortcuts must fail.

Compare PRE, POST and hidden-only using identical examples, trace supervision,
upper LoRA and development selection. Teach component tasks if necessary:
direct-count training alone does not teach a selector. Keep component accuracy
visible. Training components separately and testing their composition supports
a stronger transfer claim than training the joint task; distinguish these.

First supply all models the same correct intermediate identity, without the
second count, and ask for the second aggregation. Report second-answer exact,
complete-answer likelihood and paired differences across N. Counterfactually
change the supplied person: the answer should follow that person's independently
varied garden count. This tests a later query conditioned on a reasoning state.

Then generate the intermediate identity and answer end-to-end, with a fixed
short format and equal token caps. Report selector accuracy, second-stage accuracy
conditional on a correct selector, end-to-end exact, format failures, tokens,
latency and memory. A supplied-prefix benefit is not end-to-end reasoning success.
Compare measured compute as well as tokens: the branch adds work per forward.

For a within-model causal check, build one common cache through the correct
selector prefix and clone it before the second continuation. Compare branch
on/off over the continuation with identical teacher-forced text for likelihood.
Do not inherit different earlier caches. An effect supports later reuse; direct
one-number prefill/decode ablations do not establish composition. A null
final-query-only intervention cannot exclude earlier-query effects propagated
through upper layers, as in V2.

## Available backbones

The tested runtime is Qwen2.5-VL-7B-Instruct. It can generate intermediate text;
that does not make it a model specifically trained for reasoning. Cached
Qwen3-8B exposes enable_thinking in its template but is a text reference.

Locally cached nvidia/Cosmos-Reason1-7B is a reasoning VLM according to its
[official model card](https://huggingface.co/nvidia/Cosmos-Reason1-7B). Root and an
agent independently checked snapshot3210bec0495fdc7a8d3dbb8d58da5711eab4b423:
four named shards and their index, tokenizer/processor metadata,
Qwen2_5_VLForConditionalGeneration,28layers,hidden3584,28query/4KVheads,
M-RoPE[16,24,24],use_sliding_window=false. This suggests wrapper compatibility.
Shard integrity, actual loading, prompting and reasoning behavior remain untested.
A bounded software/short-trace gate using documented prompting must precede
training transfer. Such a gate establishes executable support, not efficacy or
reasoning composition. Do not silently substitute this backbone for Qwen2.5-VL.

## If V3 fails: measure the missing operation

Finish the staged marginal-preserving challenge before another adapter. Its
2x2 switches change conjunctions while preserving separate marginals. Report
all paired answers and predicted differences; do not select attractive examples.
A failure does not locate the problem inside perception, attention or decoding.

The next bounded diagnostic can reuse the changed frames. Ask about the
conjunction on each frame, first alone and then indexed inside the unchanged
full context. This removes counting while retaining visual binding; the full
context adds retrieval competition. Use fixed wording, balanced yes/no labels,
an ordinary-model control, raw answer likelihood and generation. These labels
are diagnostic, not per-frame method-training supervision or an external solver.

Poor isolated judgments implicate perception/binding or the interface; strong
isolated but poor contextual judgments implicate contextual retrieval/binding;
strong contextual judgments with poor aggregate answers motivate cardinality/
readout investigation. These are not exclusive causal diagnoses. If likelihood
tracks the change but generated counts do not, investigate decision margins
before adding representation capacity. V2's unseen multi-digit successes and
poor linear probes establish neither absent information nor a universal
unseen-number emission barrier.

## What normalized nonlinear value statistics cannot guarantee

For fixed query and per-head K/V pairs, repeating every visible pair m times
leaves a softmax mean of any fixed value transform unchanged: numerator and
denominator both multiply by m. PRE supplies no cardinality guarantee. Actual
frame duplication changes positions, contextual states and unreplicated prompt
proportions, so this is not a theorem that the full transformer cannot count.
See [Cardinality-Preserved Attention](https://www.ijcai.org/proceedings/2020/0194.pdf).

With uniform attention and identity lift, value sets {(1,1),(0,0)} and
{(1,0),(0,1)} have identical coordinatewise mean-SiLU statistics but different
conjunction counts. Learned affine mixing followed by SiLU can distinguish them
if both attributes are available together; the rank8 lift permits that mixing.
It cannot recover distinctions absent from its inputs. Scalar SiLU is neither
intrinsically incapable of binding nor a guarantee of binding across tokens.
Feature processing before pooling is established by
[Deep Sets](https://arxiv.org/html/1703.06114v3); the question is which distinctions
are useful and actually used in this pretrained native decoder.

## Centered sums: evidence rather than another grid

Uncentered local summation repeats a query-only baseline per region.
center_messages subtracts phi(W_h h+b) from each phi(W_r r+W_h h+b), making a
zero read contribute zero. Irrelevant evidence need not give a zero read;
distractor errors can still accumulate. Conditional sum pooling is established.

P1 TEXT, Qwen2.5-3B, one seed and reused samples: fixed-epoch-nine centered64
long-N exact25.0% versus raw64 37.5%; centered16 5.6% versus raw16 31.9%.
Both registered centering/smaller-region predictions failed. V0 smoke and all
inspected V1/V2 vision configurations disable centering; no centered clean-vision
training result was found. V1 sum collapse therefore does not test that variant,
but existing evidence does not make centering its likely cure. Require a
measurement tying failure to baseline/amplitude plus an appropriate intervention
before revisiting it; do not launch a centering/rank/block grid by default.

Evidence: [V2 results](NATIVE_AGGREGATION_VISION_V2_RESULTS.md),
[P1/P2 results](NATIVE_AGGREGATION_PILOT_RESULTS.md),
[preregistration](../../PREREG_AGG.md), [runtime](../../gnnformer/runtime.py).
The [older reasoning notes](../cot_operations_bridge.md) contain useful trace
measurements; their blanket claims about transformers/reasoning models are not
established by those operation probes and must not be carried into this work.

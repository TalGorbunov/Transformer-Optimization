# V7: preserve local computation, learn native aggregation

Status: completed. Both-seed relative gain passed; practical accuracy failed. See [verified results](NATIVE_AGGREGATION_VISION_V7_RESULTS.md). The exact
protocol, amendments, failures and budgets live in [PREREG_AGG](../../PREREG_AGG.md).

## Question and hypothesis

Can a model retain its pretrained ability to interpret each observation, then
combine those interpretations accurately within each generated token?

The prior experiments do not establish an aggregation gain. In particular, V6's
local teacher was accurate on isolated training images, but the small learned
reader did not reliably reproduce that competence in long contexts. This
motivates preserving the full pretrained local computation. It does not show
that information was absent from the joint representation or establish the
cause of every earlier failure.

V7 runs the complete frozen vision-language model independently on every
image with the question and executed answer prefix. A final global stream
receives the question and same prefix. A shared learned set function combines
the local final hidden states and adds a residual before the global native
normalization and vocabulary head. Only the global stream emits a token; that
selected token becomes the next input for every stream. There is one native
model invocation per output token, with multiple independent batch rows.

The scientific hypothesis is that retaining complete local processing makes
count-relevant evidence easier to aggregate across larger sets than adapting
an ordinary joint-image hidden state. V7 tests the entire processing change.

## Matched comparison

Both arms use an identical 1,041,600-parameter core, frozen backbone, native
FP16 carry/readout, FP32 branch, training examples, initialization within seed,
optimizer schedule and native generation policy.

| Arm | Frozen evidence representation | Input to the shared set function |
|---|---|---|
| Parallel local | N complete canonical single-image question/prefix computations | N local final states and a text-only global final state |
| Joint control | One ordinary all-image question/prefix computation | Its final state as both global state and one local element |

The global instruction omits explicit set size in both arms. Local prompts keep
the established canonical single-image format. The zero-initialized parallel
branch therefore has a text-only global baseline; the joint branch has an
ordinary multimodal baseline. This is part of the architecture comparison.

For global state g and local states h_i, the shared core is

```
q = Wq RMS(g)
m_i = SiLU(Wlocal RMS(h_i) + q + b) - SiLU(q + b)
z = sum_i m_i
delta = U SiLU(Wagg z + q + b_rho)
g_fused = g + cast_native(delta)
```

The subtraction gives raw zero padding an exact neutral contribution. It does
not make a real negative image neutral by construction. Neutrality and accurate
counting must be learned from the native answer loss. The output projection U
starts at zero. No classifier, external tally, numeric vocabulary mask or gold
local labels are deployed.

Since fusion occurs after all decoder blocks, it does not change the backbone's
KV states. Frozen causal states can be cached for training the branch, including
the observed count prefix when predicting EOS. Development and testing execute
the complete native model on original images and its own generated prefixes.

## Experiment and decisive outcomes

Training balances all54 questions across K0–8 and N8/16. There are1,890 distinct
training contexts and1,944 weighted slots per epoch; the finite saturated N8/K8
cell has only one possible context per question. Its54 repeated slots and39
historical context reuses are recorded. Development uses72 new N16 contexts.
The452 fresh tests exclude19 prior/current manifests at complete-context level;
individual rendered atoms still recur and extension changes Step-label pixels.

Each arm runs40 epochs,4,860 updates, in seeds8/9. Five native development sweeps
select the checkpoint. Primary exact match requires a complete EOS-terminated
integer answer. Malformed and truncated outputs remain in every denominator.

Each seed must gain at least11 correct familiar-OOD answers out of216 over its
matched control while losing at most5 N16 answers out of108. The practical
milestone additionally requires at least152/216 OOD and116/192 nonzero OOD
answers correct. Unseen counts9–16 remain separate whole-answer results.
Paired, count-stratified family confidence intervals do not replace the
registered per-seed decisions.

## Positioning and limits

Independent evidence processing followed by learned generative fusion is close
to [FiD](https://aclanthology.org/2021.eacl-main.74/) and the set-function structure
is [Deep Sets](https://arxiv.org/abs/1703.06114). Earlier fenced-replica experiments
in this repository are also relevant. [HiLS](https://arxiv.org/pdf/2607.02980)
provides a close alternative involving local reads and hierarchical aggregation;
it should not be characterized merely as sparse attention.

This implementation is not a new attention family or a proof of increased
information capacity. A single batched invocation still performs N local decoder
computations, duplicates question/prefix work and maintains N local caches.
Any efficiency claim needs measured compute, memory and latency comparisons.

The architecture can consume each newly generated reasoning token as a shared
prefix. Whether this improves a reasoning model, or composes with additional
reasoning tokens, remains an untested hypothesis. A positive V7 result would
justify that next experiment; it would not establish the composition claim.

# Research direction: learn the effect of an observation

Status update: the user clarified that native aggregation reusable at every
reasoning step is the objective. The separate numerical interface proposed here
is outside that objective. See [NATIVE_AGGREGATION_PROPOSAL.md](NATIVE_AGGREGATION_PROPOSAL.md)
for the current direction. This earlier exploration is retained for context.

2026-09-08. Design recommendation after challenging the earlier additive-channel
proposal. This is not a result, pre-registration, or authorization for GPU runs.

## Recommendation

Investigate whether a pretrained VLM can translate observations into small state
updates that compose reliably beyond the training horizon. Keep the current
bind-then-count result as the working reference. Count is the simplest instance;
the decisive extension is a task requiring persistent, order-sensitive state.

The proposed insight is that a locally predicted update can describe what an
observation would do to a memory without knowing that memory's current value.
Those updates can be computed in parallel and composed afterwards. This offers
a precise route from short-context perceptual competence to long-sequence
computation, within a restricted class of state updates.

The scientific claim to test is answer-supervised grounding and extrapolation.
Associative composition, matrix-product sequence models, and parallel scans are
established ideas; this proposal does not claim to invent them.

## Why move beyond the previous proposal

The protected-sum proposal is a useful control, but substantially overlaps work
already recorded in `docs/archive/RESULTS_pre_fencing.md:292-338`: learned
per-frame maps, fixed pooling, both direct and injected readouts, and multiple
tasks. Those historical cells used different representations and splits from
the September work; their results do not settle the new experiment. They do
establish that a sum adapter alone is not a fresh research direction here.

The archive also records that richer sequence models did not help the tested
permutation-invariant tasks. Therefore temporal machinery needs a task that
actually requires persistent state. Adding it to the existing restricted count
benchmark would be poorly motivated.

The September vision result is already .97-.99 exact on the reported cells,
with K<=8 even at longer N. The new objective is to preserve that performance
while learning a broader computation and extrapolating in K as well as N.

## Minimal computational interface

For each frame, the frozen VLM receives the original question and a shared
learned read token. Begin with independent local encoding using identical local
positions, because that is the best supported perceptual configuration in the
current work. No benchmark-specific regex rewrite is part of the proposed input.

A small learned head maps the local state h_i to an affine transformation:

    s' = A_i s + b_i,       (A_i, b_i) = phi_theta(h_i).

For two consecutive segments U then V, composition is

    A_VU = A_V A_U,
    b_VU = A_V b_U + b_V.

This is one fixed rule for every question. It can be implemented with homogeneous
matrices T_i = [[A_i, b_i], [0, 1]] and an ordered product T_N ... T_1. Changing
parenthesization leaves the real-arithmetic result unchanged; changing order
generally changes it. Use a balanced reduction when only the final state is
needed, or a prefix scan if a later experiment needs intermediate states.

The output head reads the resulting state directly. The pilot uses a numerical
output, trained with final count labels and rounded at evaluation; native text
generation remains a separate requirement. There is no per-frame verdict
threshold followed by a question-specific external reducer.

Count lies inside this family: A_i=I and b_i writes one unit for matching
evidence. Thus the model class includes the currently successful additive rule.
This inclusion is not a guarantee that answer-only training will find it.

For the first pilot, use just two state coordinates and four learned local
outputs, a_i, b_i, u_i, v_i:

    p' = a_i p + b_i
    c' = c + u_i p + v_i.

Fixing the coefficient of c at one protects the accumulator from learned decay.
The same structure contains ordinary count (u_i=0, v_i=local contribution) and
the temporal transition task below. One learned linear readout accesses c; no
task identifier selects a different reducer. This is a deliberately restricted
affine family, not a general state machine. It does not, for example, reset c.

Use local gates that can attain exact neutral settings a_i=1, b_i=u_i=v_i=0;
clipped linear gates can represent exact zeros and ones, whereas ordinary
sigmoids have tails. Choose the precise parameterization in the symbolic pilot
and hold it fixed for the visual test. Keep accumulation in fp32 and preserve
its raw scale at the readout. The gate choice is not itself a novelty claim.

This constraint prevents count decay but does not solve inaccurate local
updates, drift in p, or output calibration. Measure all three. Only expand the
state dimension if the target task needs it; avoid a menu of specialist blocks.

## A concrete representability example

Question: how many hallway-to-kitchen transitions occur in Alice's own sequence
of observed locations? Other people's frames do not update Alice's location.
This is an observation-sequence task, not a claim about unobserved physical moves.

Let p indicate that Alice's last observed location was the hallway, and c be the
transition count. Let h and k indicate that the current frame shows Alice in the
hallway or kitchen; o indicates that it shows Alice in any tracked location.
For a single-person frame these are mutually consistent local indicators.

    p' = (1-o) p + h
    c' = c + k p

Equivalently, on the homogeneous state [p, c, 1]^T,

             [ 1-o   0   h ]
    T_i  =   [  k    1   0 ].
             [  0    0   1 ]

Other-person frames give the identity. Alice in the hallway records that state;
Alice in the kitchen increments only if the previous Alice observation was in
the hallway, then clears p. Alice in another room clears p without incrementing.

These equations demonstrate that a tiny affine model CAN solve the task. They
are not supplied to the learned model as a task-specific execution program.
In the intended experiment, phi learns the matrix entries from final answers.
An oracle using these equations is explicitly labelled an oracle control.

The distinction from a sum is visible with the same four Alice frames:

    hallway, kitchen, hallway, kitchen  -> 2 transitions
    hallway, hallway, kitchen, kitchen  -> 1 transition

Arbitrarily many other-person observations can be inserted without changing
either answer. A sum of independent position-free features cannot distinguish
these sequences. A fixed local window loses the necessary predecessor when the
intervening gap exceeds its window. Strong position-aware and recurrent models
must therefore be included as learned baselines.

Unit-diagonal triangular matrices used for ordered-pair statistics are not an
adequate substitute: they count all ordered subsequences and generally cannot
implement the state resets above. Do not confuse that simpler construction with
general state tracking.

## Single-forward and generality boundaries

Compute consists of one vision/backbone traversal followed by O(log N) stages of
small operator compositions. For fixed state size, a tree uses O(N) compositions.
This is one feed-forward graph and no autoregressive reasoning trace, but not
constant computational depth as N grows. Report the distinction explicitly.

Affine composition covers useful counters and finite-state updates with adequate
state dimension. It does not provide arbitrary reasoning, unlimited entity
memory, unbounded stacks, or a universal shortcut to chain of thought. If local
visual interpretation itself requires motion, later experiments should use short
clips as units and define event boundaries carefully. Sparse action-frame results
on the current HERBench construction do not settle full temporal reasoning.

Associativity guarantees only evaluation of the learned recurrence. Approximate
identity updates can drift; incorrect updates can accumulate; contractive updates
can erase memory; expansive updates can amplify errors; the output head can fail
on unseen values. With bounded step norms M and per-step operator errors epsilon,
a generic product perturbation bound can grow like N*M^(N-1)*epsilon. The exact
algebra does not remove the need to learn accurate, stable local semantics.

## The smallest experiment worth doing

Before any GPU run, prepare and pre-register the exact examples, predictions,
model selection, resource budget, and stopping criteria.

1. **Symbolic feasibility.** Feed clean categorical observations to the same
   operator learner; supervise only final answers. Train on short sequences;
   test long irrelevant gaps and unseen transition counts separately. Use the
   representability construction above as an oracle. If the learned model fails
   here, stop before adding vision.
2. **Visual grounding.** Use held-out MMReD frames and the frozen local encoder.
   Fit the small heads from final answers. Keep frame identities disjoint across
   train/test. Explicitly match representations and supervision across baselines.
   Any initialization from an oracle-frame-trained checkpoint must be labelled
   privileged supervision, not described as answer-only training.
3. **Joint competence.** Train one question-conditioned model on ordinary frame
   counting and the stateful transition task. Evaluate familiar and new question
   phrasings and entity/room combinations. The task families are trained; do not
   describe success as zero-shot task transfer.
4. **Crossed extrapolation.** Train N<=32, K<=8. Evaluate N=64/128/256 as feasible,
   independently varying K within and beyond training support (e.g. 16/32).
   For transition tasks ensure 2K<=N where required. Vary irrelevant gap lengths
   independently of the number of relevant observations. Use matched-multiset
   order pairs so marginal-count shortcuts cannot explain accuracy.
5. **Retention.** Re-evaluate the original September count cells. A proposed
   target is performance within 3 percentage points of the working reference,
   with strong exact-match extrapolation on the new task. Set final thresholds
   before running and report multiple seeds plus paired uncertainty.

The baseline set must include current bind-then-count, the answer-only additive
head, and order-aware transformer and GRU/LSTM heads on the same local features.
Include standard linear recurrent/weighted-automaton parameterizations as close
architectural controls. The exact same affine recurrence executed serially MUST
agree with the parallel implementation up to numerical error; this is an
implementation and efficiency comparison, not an accuracy competitor. The
proposed core is mathematically in this existing family. Explain what the
grounding/interface experiments establish beyond that fact.
Also include a per-frame semantic classifier plus a hand-written oracle state
machine as a separately labelled decomposition reference, not a fair answer-only
competitor. This estimates whether perception or learned composition limits us.

A strong result would preserve simple counting, learn stateful visual computation
from final answers, and survive both larger K and longer irrelevant gaps. Beating
an order-blind sum alone is insufficient. If only oracle operators work, the
learnability claim fails. If small learned readouts saturate with perfect state,
the output interface remains the bottleneck. Do not respond by adding a different
specialist reducer for each failed task.

## Positioning and nearest prior work

The potential paper is about making pretrained local competence compose
reliably, with a small explicit computational interface and a clear experimental
boundary. It is not a new discovery that matrices associate.

- Transformers Learn Shortcuts to Automata (ICLR 2023): parallel composition of
  transition functions and logarithmic-depth shortcuts are explicit prior art.
  https://arxiv.org/abs/2210.10749
- Connecting Weighted Automata, Tensor Networks and Recurrent Neural Networks
  through Spectral Learning: matrix-product models and linear second-order RNNs
  have a developed theory and learning literature.
  https://arxiv.org/abs/2010.10029
- Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues:
  transition parameterization affects expressivity and extrapolation; a recurrent
  state alone does not establish general state-tracking ability.
  https://arxiv.org/abs/2411.12537
- Learnable Commutative Monoids for Graph Neural Networks: learned associative
  aggregation and logarithmic depth are established in the GNN lineage too.
  https://arxiv.org/abs/2212.08541
- VideoLLaMB: long-video understanding with recurrent memory already exists;
  compare the actual supervision, computation, and extrapolation objectives.
  https://arxiv.org/abs/2409.01071
- Deep Signature Transforms: learned ordered stream statistics are prior art for
  any narrower triangular/signature-style alternative.
  https://arxiv.org/abs/1905.08494
- Parallelizing Linear Recurrent Neural Nets Over Sequence Length: affine scan
  is a longstanding parallel implementation of linear recurrence.
  https://arxiv.org/abs/1709.04057
- Sequential-Parallel Duality in Prefix Scannable Models: a particularly close
  modern framework; compare its scope before making any architecture claim.
  https://arxiv.org/abs/2506.10918

Working pitch: a VLM can recognize individual observations yet fail when their
effects must be composed over a long sequence. Learn a small state update from
each observation and compose the updates in parallel. Test whether this carries
short-sequence competence to larger counts and longer temporal dependencies,
without intermediate labels or a reasoning trace.

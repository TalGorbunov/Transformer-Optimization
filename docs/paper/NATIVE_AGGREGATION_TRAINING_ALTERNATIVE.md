# Training alternatives to another aggregation architecture grid

Status: conceptual proposal only, 2026-09-10. No training or staging is launched
by this note. Register any chosen comparison, fixed data, decision rules and
resource limit before execution. V3 and binding results remain separate.

## Recommendation and current evidence

Prioritize **scene diversity at fixed training cost**, holding the existing model
and ordinary final-answer loss fixed. Counterfactual precision training is
contingent on what remains after that comparison.

V2's extra global read did not beat its hidden-only control. V3's PRE-versus-POST
screen failed across its two seeds: the familiar-OOD gain was 6.94 percentage
points in seed0 and zero in seed1. Two seeds demonstrate observed instability,
not a well-estimated population variance.

The completed fixed-marginal diagnostic changes the earlier hypothesis.
All six selected models increased their count prediction for all16 K2-to-K6
pairs despite identical complete character and room marginals. Mean predicted
changes were 3.75–4.8125 versus the gold change4. A purely marginal-only
explanation is incompatible with these outputs. Precision remains weak:
scene MAE ranged0.71875–1.53125 and both answers were correct for only0–2 of16
pairs per model. This does not support a narrative of absent conjunction
sensitivity. It also does not prove a general binding or counting algorithm.

Repeating180 scenes for nine epochs gives1,620 presentations. There are54
possible queried character/room combinations, and the V2 law correlates queried
marginals with K. These facts motivate a data-coverage control; they do not
establish that the tested models ignore conjunctions.

Two distinctions remain essential:

- With ordinary per-example CE, pair IDs contribute no information to the loss.
  The same examples in the same optimizer batches give the same mathematical
  objective whether pair metadata is present or absent. Changing batches can
  change optimization; changing scenes changes the data distribution.
- MMReD already provides exact final counts. At fixed inputs, replacing these
  labels with identical final answers from a reasoning teacher gives the same
  CE target. A teacher must contribute additional verified examples or other
  supervision to change the experiment.

## Established antecedents

| Prior work | What it establishes and what it does not establish here |
| --- | --- |
| [Kaushik, Hovy and Lipton, ICLR 2020](https://arxiv.org/pdf/1909.12434), sections 3–5, Tables 5–8 | Train on minimally edited examples whose labels change to weaken shortcuts; compares against equal quantities of ordinary data. The broad counterfactual-augmentation idea is established. |
| [Chen et al., CVPR 2020, CSS](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_Counterfactual_Samples_Synthesizing_for_Robust_Visual_Question_Answering_CVPR_2020_paper.pdf) | Applies counterfactual image/question augmentation to VQA. Its masked-object/word edits and assigned targets differ from our exact rendered interventions, but VQA debiasing through complementary examples is already established. |
| [Chen et al., Counterfactual Samples Synthesizing and Training](https://arxiv.org/pdf/2110.01013), section 3.3, Algorithm 4 | Explicitly separates cross-entropy on counterfactual examples from an additional contrastive objective. A generic CE-plus-pairwise extension is also prior art. |
| [Xie et al., UDA](https://arxiv.org/html/1904.12848v4), sections 2.1–2.2 | Adds stop-gradient KL consistency between label-preserving augmented views alongside supervised CE. It requires valid answer-preserving transformations; label-changing room switches cannot use the same invariance target. |
| [Wu et al., Polyjuice, ACL 2021](https://aclanthology.org/2021.acl-long.523.pdf), section 3.1 | Uses an equal-size ordinary-data control and reports that randomly selected counterfactuals need not beat extra ordinary data. Carefully chosen transformations, not augmentation terminology, must earn their benefit. |
| [Gardner et al., Findings 2020](https://aclanthology.org/2020.findings-emnlp.117/) | Contrast sets diagnose local decision boundaries. A paired diagnostic alone neither trains an ability nor establishes a causal account of internal computation. |
| [Yu et al., Distilling System 2 into System 1](https://arxiv.org/html/2407.06023v3), section 3.2, section 4.5, Appendix A.2 | Explicitly trains direct-answer behavior from slower reasoning procedures without intermediate output tokens. It also reports poor GSM8K distillation results. Reasoning compilation is established and is not guaranteed to succeed. |

The potentially valuable contribution is a well-supported diagnosis of why native
VLM aggregation fails, and a simple training intervention that generalizes across
length, count and tasks. Neither a new attention family nor generic
counterfactual training would be a defensible novelty claim.

## Existing-output precision decomposition

For each audited pair, define `eL=prediction_low-2` and
`eH=prediction_high-6`. The shared offset
`c=(eL+eH)/2` and half contrast error `d=(eH-eL)/2` obey the exact identity

`(eL^2+eH^2)/2 = c^2 + d^2`.

Averaging over the16 pairs gives the following descriptive decomposition.
All192 model/example outputs parsed, so no observations are omitted.

| Model | Mean c | E[c²] | E[d²] | Scene MSE | Share E[c²]/MSE | Mean absolute count-delta error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V2 global seed0 | -0.06250 | 0.656250 | 0.468750 | 1.12500 | 58.3% | 1.1250 |
| V2 hidden seed0 | 0.37500 | 0.812500 | 0.375000 | 1.18750 | 68.4% | 0.8750 |
| V3 pre seed0 | -0.03125 | 0.484375 | 0.359375 | 0.84375 | 57.4% | 1.0625 |
| V3 pre seed1 | 0.84375 | 2.796875 | 0.609375 | 3.40625 | 82.1% | 1.3125 |
| V3 post seed0 | -0.62500 | 1.062500 | 0.437500 | 1.50000 | 70.8% | 1.1250 |
| V3 post seed1 | 0.81250 | 1.781250 | 0.718750 | 2.50000 | 71.2% | 1.2500 |

Thus57.4–82.1% of the scene MSE lies in the common-offset component. The remainder
is still contrast error: a positive response and a near-correct mean delta do
not imply exact intervention precision. The component c varies by pair and its
signed average can be near zero. It is not necessarily one global calibration
bias, and this algebra does not identify a causal readout circuit or justify
post-hoc correction.

Source: [binding predictions](../../outputs/native_aggregation_vlm/v3/binding_probe/probe_440886/predictions.json);
SHA256 `00638b3511d1710a8bb84bea068b7a1c6d8db5ff7f348ce962722b4446acab9b`. Calculated directly from the192 saved rows with elementary
standard-library arithmetic, with the identity checked for every pair. No new
model forward, fitted statistic, test selection or training was performed.

## First bounded comparison: repeated versus more diverse scenes

Use the existing V2 hidden-only adapter and upper LoRA unchanged, with two fixed
new training seeds,2 and3. Cross these with two data conditions: four runs total.
Do not add an architecture, pair loss, teacher, curriculum or optimizer search.

Divide training into nine blocks of180 presentations,45 optimizer updates per
block with accumulation4. The first block uses the exact original V2 training
set in both arms. Keep a shared slot list containing each example's N, gold K,
queried character and queried room.

| Condition | Block1 | Blocks2–9 | Distinct training scenes | Presentations |
| --- | --- | --- | ---: | ---: |
| Repeated | Original180 | Repeat original180 | 180 | 1,620 |
| More diverse | Same original180 | Fresh scenes in170 mutable slots; repeat the10 saturated slots | 1,540 | 1,620 |

The ten N8/K8 scenes are deterministic given the query: every frame contains
the queried character in the queried room. There is no fresh content with the
same N/K/query under this renderer. Keep those same ten samples in every block
in **both** arms. This is a finite-support constraint, not a failed sampling
attempt or a reason to relax the matching rule. The distinct count is
`180 + 8*170 = 1540`, not1620.

For every other slot in blocks2–9, generate one new scene from the unchanged V2
conditional law, matching that slot's N, K and queried character/room exactly.
Exclude new content against all available prior train/dev/test/profile/diagnostic
manifests, against the original180 and against every earlier new block. Reuse
of the original180 is intentional and explicit. Audit the1540 distinct content
hashes and the1620-entry presentation schedule. Freeze a single new data
manifest/schedule for both initialization seeds; this does not estimate variation
over generated training corpora.

Use the identical deterministic slot permutation within each block in both arms.
The repeated arm maps each slot to its original example; the diverse arm maps it
to the frozen block-specific example. Thus every optimizer update has identical
N, K and query exposure. Both arms use405 updates,1,620 native answer/EOS CE
presentations and19,440 training image frames. Verify actual processor layouts;
fixed image size plus identical per-slot questions should also match prompt
token counts and answer lengths. Scene content is the intended difference.

Preserve optimizer state across all nine blocks. Run the same development set
after updates45,90,...,405 and select by the established exact/raw first-token
NLL/earliest-check rule. There is no optimizer reset, learning-rate change,
reinitialization, altered clipping or extra pass over the larger corpus.
The first block has identical inputs and prescribed updates in both arms and
provides an implementation parity check within expected numerical tolerance.

Keep both train seeds2 and3 regardless of results. A gain in both is useful
evidence that extra scene variation helps at fixed presentation and update
cost. Two seeds are insufficient to establish a reduced population-level seed
variance. If both arms fluctuate similarly, report that outcome rather than
selecting the better seed.

At the previous rough28-minute run estimate, four runs cost approximately
1.87 GPU-hours before profiles or extra evaluation. The training tensor-shape
budget is matched, but timing is still measured, not assumed. Register a bounded
resource reservation after software checks; all staging and heavy audits use
CPU Slurm, and all model work uses GPU Slurm.

## Contingent next question: counterfactual CE for precision

If diversity helps but precise counting remains poor, counterfactual training
could test whether local count changes improve precision beyond equally diverse
ordinary examples. It should then compete against a newly matched ordinary-data
control at the **same distinct-scene, presentation and optimizer budget**.
A1540-scene counterfactual arm against the old180-scene control would not isolate
the transformation. Do not describe this as repairing absent conjunction
sensitivity: the observed models already respond to the fixed-marginal switches.

One feasible N16 construction has45 pairs with queried marginals C=R=8:
five copies of the nine label pairs `(0,1),(1,2),...,(7,8),(8,0)` give ten
occurrences of each K0..8. At count k the contingency cells are
`(k,8-k,8-k,k)`. Swapping delta C-only/R-only room pairs changes K by delta
while preserving every character at its position and the full room histogram.
Most transitions have delta1; the8/0 edge changes all16 room assignments.
Every counterpart in the ordinary-control arm must share its query and gold
schedule. A changed schedule requires a new matched control; do not reuse a
historical baseline whose question exposure differs.

Retain ordinary N8 examples, including saturated K8 slots. Fixed-marginal
K7-to-K8 pairs at N8 are impossible, so dropping labels to manufacture pairs is
not acceptable. A complete multi-block construction and its exclusion rules
would need a new preregistration; this note does not authorize changing the
first diversity comparison.

Train with independent final-answer/EOS CE and ordinary matched slot batching.
Pair IDs, states, marginals and changed-position markers never enter model
inputs. The semantic generator supplies a training-data inductive bias, even
without per-frame supervision. A positive result would support this data
intervention, not a new loss, attention family or generic aggregation algorithm.
The separate KL experiment below remains optional and later.

## Optional later comparison: isolate a conservation loss

This is a separate two-arm question, not an extra grid in the first block.
Use the exact same 90 N8-to-N16 extension pairs in both arms, with ten pairs per
K0..8. Extend by interleaving only verified nonmatching frames. The two arms see
the identical 180 scenes in identical optimizer batches: CE versus CE plus a
standard UDA-style consistency term.

For a label-preserving pair (x,x'), a concrete baseline is

`L = [CE(x,y)+CE(x',y)]/2 + lambda KL(stopgrad p_theta(.|x) || p_theta(.|x'))`,

where p is the native full-vocabulary distribution at the answer position,
not a newly normalized K0..8 classifier. Keep answer/EOS CE in both arms.
Predeclare one lambda, for example UDA's common choice 1, rather than testing
a loss-weight grid. This is an adapted prior-art baseline, not a novel loss.

Store only the detached source distribution, backpropagate that view's CE, then
process the other view with CE plus KL. This can preserve the number of model
forwards/backwards and microbatch size; it avoids retaining two full VLM graphs.
Count the extra distribution arithmetic, transfers and measured training cost.
Use a deterministic, predeclared source direction or direction schedule.
A mere equality-of-predictions metric is insufficient: an always-wrong constant
is perfectly invariant, so both-views-correct must accompany consistency.

This comparison attributes any gain to the added regularizer on fixed data.
It does not establish that geometric pairing is essential. That stronger claim
would need a shuffled partner control drawn within the same gold stratum, using
the same scene multiset and loss budget; otherwise ordinary confidence/label
regularization remains an explanation.

Do not use invariance KL across label-changing room switches. Shifting an
answer distribution by a known numeric delta would explicitly introduce
count-specific arithmetic supervision, and multi-token numbers do not admit
a simple next-token shift. Likewise, summing predicted counts over partitions
is a count-specific algebraic loss. Such objectives may be useful, but they
would answer a narrower scientific question.

## Required evidence and limits

Judge the diversity intervention primarily on fresh same-law N16/N32/N64
evaluation. Report a fresh fixed-marginal challenge separately as a precision
diagnostic. The latter should use
different marginal values and held-out changes such as K2-to-K6, not only the
training transitions. Exclude every previously inspected binding family and
all members of every training family. Keep unseen K9..16 separate. If inspected
V2 tests are reused for exploration, retain their exploratory status and reserve
new data for confirmation.

A reasonable predeclared screen is at least five percentage points on familiar
N32/N64 exact in each seed, no more than five points lost at N16, with binding both-members-correct and absolute predicted-delta error reported
as secondary precision outcomes. Do not replace the main threshold with the
already-successful positive-delta sign test. Report all unparsable answers as wrong and disclose any
conditional parsed-only metric. Use whole input families for intervals; two
seeds do not estimate a training-seed population.

A gain confined to the fixed-marginal challenge may be adaptation to its scene
law. A gain confined to same-law tests does not establish repaired binding.
Conservation without better exact accuracy can be collapse. Failure of the
diversity comparison would weaken this particular finite-data explanation at
the prescribed optimizer budget. It would not prove an architectural or
information-theoretic limitation, or rule out every training intervention.
Do not tune new weights, edits or thresholds against the same confirmation data.

The loss and data construction add **no inference operations relative to the
chosen fixed adaptation architecture**. A hidden adapter still has its own
inference cost; do not call that zero overhead relative to a bare backbone.
A later ordinary-weight/LoRA integration could establish a stricter deployment
claim, including any merge/requantization effects.

The same trained weights are callable at later autoregressive reasoning tokens,
but answer-only gains do not demonstrate composition with reasoning. Answer-only
finetuning could also change reasoning behavior. A positive result needs a
separate matched direct-answer/reasoning experiment with equal token budgets,
new queries and ground-truth verifiable subproblems. Do not enforce arbitrary
reasoning-token distributions to be identical across two inputs merely because
their final answers agree.

Conservation under irrelevant evidence is broadly applicable only when the
transformation is semantically valid for the task. The current room-switch
generator uses MMReD-specific knowledge. Transfer to other aggregation
operations, modalities and tasks must be shown before describing a general
aggregation training method.

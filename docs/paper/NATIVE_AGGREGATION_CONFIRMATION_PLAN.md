# Contingent confirmation plan for native value lifting

This is a contingent proposal, not a registered experiment. No jobs are submitted by it.
It applies only if the exploratory V3 comparison provides a positive result.
Before execution, append the exact protocol, frozen manifests, source hashes,
decision rules and resource limits to [PREREG_AGG.md](../../PREREG_AGG.md).
The user's continuing research authorization already covers the work; the
remaining gate is scientific registration and resource accounting.

## Purpose and entry gate

V3 reuses inspected V2 tests. Two training seeds and prospective registration of
the V3 comparison do not turn those examples into untouched confirmation data.
Its results can justify a new test, not a definitive efficacy claim.

Proceed only if V3 passes its registered PRE-versus-POST screen in both seeds
and its ordinary-adaptation screen. Freeze the resulting PRE method exactly:
layer, lift width, activation, initialization, optimizer, training duration,
prompt, image resolution and decoding. There is no confirmation-set tuning.
If V3 fails, do not generate this study merely to search for a favorable draw;
record the failure and follow the separately documented diagnostic decision.

The confirmation question is whether nonlinear processing of native values
before attention pooling gives a reproducible, practically useful advantage over
both the matched POST operation and ordinary same-site adaptation. The operator
family is established; the intended contribution is the demonstrated behavior
and, separately, a supported mechanism.

## Frozen models and training

Use three arms crossed with two new training seeds, 2 and 3:

| Arm | Middle-layer operation | Upper adaptation |
| --- | --- | --- |
| PRE | Exact successful V3 residual affine value lift, inner SiLU before pooling | Four layers of rank-8 LoRA |
| POST | Identical parameters and initialization, inner SiLU after pooling | Identical rank-8 LoRA |
| Hidden-only | Existing V2 rank-96 hidden adapter at the same input/residual site | Identical rank-8 LoRA |

PRE/POST each have 1,417,792 active parameters. The established hidden-only
control has 1,409,120, approximately 0.61% fewer. Keep this previously specified
control rather than searching a rank grid. Its nonlinear width differs from the
value adapter, so this is an ordinary-adaptation comparison, not a pure
input-only intervention. Never pad a model with unused parameters to claim
matching.

Reuse the exact V2 training and development sets: 180 and 72 examples. These
are not new training corpora. Seeds 2 and 3 change initialization and training
shuffle; pair them across arms and reset the upper-LoRA seed identically.
Use the registered nine-epoch training recipe, with native answer/EOS loss and
development-only selection: highest pooled dev exact, then lowest raw first-token
NLL, then earliest epoch. Save both selected and final checkpoints; evaluate only
the selected checkpoint for the confirmation outcome. No warm starts from V3.

Four new training jobs would cover PRE/POST at two seeds but omit a matched
ordinary-adaptation comparison at those seeds. The recommended minimal complete
design therefore has six jobs, with at most four running concurrently.

## Fresh held-out scenes

Generate all confirmation families after the method and decision rules are
locked, using a new fixed generator seed, proposed as 20260911. Preserve the V2
conditional law exactly: uniform queried character/room, exactly K positive
frames, and IID negative types with equal probabilities for same character/wrong
room, other character/same room, and neither. Preserve renderer, image size,
canonical images-first prompt and decoding policy.

The minimal same-law test has:

- **Familiar counts:** 108 fresh N16 anchors, 12 per K0..8, each extended to
  N32 and N64 by interleaving fresh negatives. This gives 324 sequences.
- **Unseen counts:** 64 fresh N32 anchors, eight per K9..16, each extended to
  N64. This gives 128 additional sequences, analyzed separately.

All six models receive the identical 452 test sequences. The paired unit is
the complete anchor family, including its lengths. Keep the anchor question,
matching evidence and semantic order fixed on extension; record changed
positions/renumbered step labels as in V2. No test labels or per-frame evidence
labels enter training or model inputs.

This intentionally retains the earlier cohort size for a bounded first
confirmation. It may be inconclusive for a five-point effect. Do not enlarge
the cohort after seeing its outcomes to seek significance, change thresholds,
or select favorable counts. If greater precision is affordable, increase the
fixed sample count before generation and registration, with revised cost limits.

## Leakage and execution guards

Maintain an explicit exclusion ledger of every available previously selected
V1/V2/V3 train, dev, test, profile and diagnostic manifest, including causal and
prefix subsets. Exclude new full ordered-state-plus-question content hashes
that match that ledger, and reject the whole new family on any collision.
Also exclude duplicates across new primary, count and optional binding families.
Record exactly which historical manifests were checked; do not claim exclusion
from undocumented historical populations.

Shared primitive rendered graphics are expected in this finite synthetic world.
Novelty means fresh scene sequences/question families, not unseen colors, fonts
or characters. Record that limitation. Verify labels, paired subsequences,
renderer parity and every published QA/image checksum on CPU Slurm. Store new
data/manifests below `/mnt/data/gabriele/gnn_transformer`; preserve all existing
files. Save checkpoints only below `/mnt/ckpts/gabriele/gnn_transformer`.

Freeze hashes before the first confirmation forward. Keep outcome summaries
unread until all preregistered arms/seeds finish, except software health checks
that do not guide modeling. Software failure permits a documented repair and
the same prescribed rerun, not a replacement seed or favorable subset.

## Primary decision and uncertainty

Require PRE to exceed both POST and hidden-only by at least five percentage
points on pooled N32/N64 familiar-count exact accuracy in **each new seed**,
with no greater than five-point loss at N16 against either control. All examples
remain in the denominator; unparsable outputs are incorrect.

Report every arm, seed and cell. For each control, average its paired accuracy
difference over both lengths and both seeds within each of the 108 fresh anchors,
then bootstrap whole anchors 10,000 times with fixed analysis seed 20260911.
Preserve lengths and seeds jointly on each draw. Require the resulting pooled
95% interval to exclude zero before treating the result as positive confirmation.
These intervals condition on six trained models and do not estimate a population
distribution over training seeds. Report both contrasts, not only the larger one.

Use separate per-seed estimates and intervals to expose instability; do not pool
in exploratory V3 outcomes to manufacture precision. Unseen K9 and K10..16 are
secondary and separate from familiar-count accuracy. Report raw first-token NLL
as such, tokenization, whole-answer exact, parse rate and error by count. A
familiar-count confirmation alone is not numerical extrapolation.

A reproducible five-point gain over both controls, with the N16 guard and a
competitive accuracy/compute tradeoff, justifies testing broader aggregation
tasks and reasoning composition. Measure isolated model latency/memory rather
than treating concurrent-job timing as a cost comparison. A proposed practical
target is no more than 20% extra model time over hidden-only at N64; fix the
measurement cohort and budget before outcomes. If the accuracy result passes
but cost is worse, report the tradeoff and avoid an efficiency claim.

If either seed or either control fails the primary screen, this does not confirm
the claimed robust advantage. If the effect threshold passes but the interval
includes zero, call the study inconclusive. Do not choose the better seed, tune
the prompt, or add activations/layers on this confirmation set. A revised method
would require a newly locked protocol and new confirmation data.

## Optional secondary: conjunction changes with fixed marginals

The V2 law does not isolate binding: expected target-character and target-room
occurrences each equal `K + (N-K)/3`. Marginal counting therefore correlates with
the answer. Add a separate, preregistered input-counterfactual diagnostic if the
budget permits; do not mix it into the same-law primary score.

Proposed size: 32 fresh matched pairs at N32, queried-character count `mC=12`,
queried-room count `mR=12`, with K2 and K6. At fixed N, mC and mR, the four
contingency cells are `K`, `mC-K`, `mR-K`, `N-mC-mR+K`: here `(2,10,10,10)`
and `(6,6,6,14)`.

Start from the K2 sequence. Choose four distinct same-character/wrong-room
frames and four distinct other-character/target-room frames, pair them uniformly,
and perform four switches:

`(C,R') + (C',R) -> (C,R) + (C',R')`.

This changes eight frames and raises K by four while preserving the complete
character histogram and complete room histogram, including non-target identities.
It changes associations, not only the target contingency cell. Keep the question,
all other frames and step labels identical. Audit exactly eight semantic/image
differences and all full-marginal equalities. No switched examples enter training.

Report both-members-correct rate, individual exact/parse rates and the paired
change in raw margin `logit(6)-logit(2)` for each arm/seed. The 32 scene pairs are
the independent units; two directions are not independent samples. A consistent
positive margin shift shows sensitivity to the changed associations, while exact
accuracy establishes whether it answers correctly. Marginal equality rules out
an explanation using only those aggregate marginals, but not every shortcut or
alternative computation.

This challenge conditions and couples scenes differently from the IID-negative
training law. It is explicitly an OOD diagnostic, not same-law confirmation or
a complete causal account of the model. A primary gain with no binding benefit
supports an adaptation result, without substantiating a conjunction-binding repair.

## Resource proposal

At the planning estimate of approximately 28 minutes per main job, four runs
cost about 1.87 GPU-hours and the recommended six about 2.8 GPU-hours before
extra diagnostics. Do not present these estimates as measured confirmation cost.

| Component | Proposed worst-case reservation |
| --- | ---: |
| Six main runs, at most four concurrent | 6 x 35 min = 210 GPU-min |
| Up to two software profiles | 2 x 5 min = 10 GPU-min |
| Isolated selected-checkpoint timing | 10 GPU-min |
| Reserved retry/contingency allowance | 10 GPU-min |
| Total | 240 GPU-min = 4 GPU-hours |

The optional binding examples must fit inside these reservations; otherwise omit
that secondary study before registration, or record a separate bounded block.
Verify actual V3 timing and expected fresh-test evaluation cost before choosing
the 35-minute caps. If those caps are unrealistic, revise the plan and its explicit
budget before submission rather than silently exceeding the four-hour envelope.
All staging, fitting, statistics and other heavy CPU work also go through Slurm.
No jobs are submitted by this proposal.

# Research reassessment — 2026-09-14

The objective remains better aggregation per forward pass on MMReD Vision, followed by evidence that the improvement also helps a native reasoning model. We do not yet have a successful general method. This note supersedes the recommendation to prioritize the raw-feature summary residual. Completed results and frozen protocols remain unchanged.

## Decision

Hold the unrun native augmentation GPU profile and further implementation specific to its auditor. The 32-slot summary readout is a possible familiar architectural control, not the selected research contribution. Its eight core CPU fixture groups passed in job 444127; native integration and efficacy remain untested. Engineering readiness is not a scientific reason to train it.

Finish the bounded, method-independent fresh-data draft already underway. No fresh cohort has been released or evaluated. The next model experiment should diagnose the competent ordinary checkpoint on fixed worlds and determine whether another aggregation intervention is justified before a new fit.

## Evidence and hypothesis

The completed original-MMReD pilot gives N32 partner/room aggregation accuracy of 61% for ordinary full-image input, 32% for normalized replacement memory, and 28% for mass replacement memory. All answers and numerical audits completed. This rejects the tested replacement configuration under its training recipe. It does not establish an architectural capacity limit or irrecoverable information loss. Deficits already occur at supported lengths.

Ordinary N32 retrieval is 82% and room-step counting is 26%. These are different question/world samples: their difference does not establish that the facts needed for each aggregation answer are accessible. [Completed pilot](NATIVE_AGGREGATION_MMRED_OFFICIAL_PILOT_RESULTS.md).

Earlier local-message and consistency experiments had narrow successes and substantial long-sequence failures, including accumulated background error. A new additive branch requires new evidence about its inputs and error accumulation. Summation alone is not a new solution.

Working hypothesis: the model can recover relevant local facts in a long visual stream but fails to combine enough of them accurately in one answer computation. This is testable and currently unestablished. Prioritizing that question follows the emphasis on a falsifiable hypothesis and an underlying insight in [Michael Black's scientific-writing advice](https://perceiving-systems.blog/en/post/writing-a-good-scientific-paper).

## One bounded diagnostic

Use a small deterministic subset of fresh N8/N16/N32 worlds selected before model outcomes. Fix question bundles, evidence renderings, decoding and analysis before inference. Give the same worlds the four original task types and a fixed schedule of atomic location probes. Paired location probes can check the local facts needed for co-location. Do not select worlds or probes because the model answers them correctly.

| Evidence supplied | What it helps examine |
|---|---|
| Original images | End-to-end behavior |
| Exact per-frame scene descriptions | Combining explicitly supplied scene facts |
| Exact query-relevant integer totals | Comparing totals and producing an answer |
| Explicit correct answer to copy | Output contract and instruction following |

Oracle conditions are privileged diagnostics, never method scores. They change representation, prompt length and difficulty; their differences are not a clean decomposition of error or proof about hidden-state accessibility. Atomic probes in the original full-image context give a complementary behavioral check. Report all fixed worlds, with uncertainty clustered by world. Results conditional on correct factual probes remain secondary.

The validated suffix-counterfactual construction can supply a secondary integration panel: unique answers and a third competitor prevent a prefix-only or suffix-only oracle from solving both changed members. Include matched unchanged pairs and factual probes. This conditioned mechanism panel remains separate from ordinary fresh evaluation. Feasibility did not establish an equal-marginal changed-answer panel.

The diagnostic must end with a decision. Residual errors from explicit scene facts that disappear with totals motivate integration research. Predominant local-fact failures motivate grounding or representation training. Failures even from totals or copied answers motivate the answer pathway or supervision. Inconclusive results do not automatically trigger another pooling fit. Exact sample size and resource limits still need to be fixed before execution; this note releases no job.

## Conditional method direction

The useful target is a learned query-conditioned operator that combines many evidence contributions inside one native token computation while retaining ordinary visual input. Each answer or reasoning token could issue a new query to the same evidence. The interface is query plus evidence to an aggregate vector, without hard-coded person/room counters or external scene graphs at inference.

An illustrative family is `m_t = sum_i phi(e_i, q_t); h'_t = h_t + rho(h_t, m_t)`. This is a design family, not a selected implementation or novel primitive. The unresolved questions are whether the evidence represents the required local relations and how contribution errors grow with relevant evidence.

Contextualized native image states are candidates because they have undergone decoder processing. With images before the question, these states cannot depend on the later question, but later images depend on preceding images. They are not independent frame encodings; summation can double-count context and does not guarantee partition invariance. If feature accessibility needs examination, use one prespecified raw-versus-contextual comparison at a fixed location and equal probe budget, not a layer search.

Training should target the demonstrated bottleneck. Short legal scene variations can supervise sensitivity to relevant changes and stability to irrelevant changes without naming latent coordinates as counters. Compare identical supervision on a generic adapter: improved data may explain any gain. Earlier consistency failures remain evidence against assuming this will extrapolate. No new training objective is released here.

## Positioning and required result

Learned visual resampling plus a zero-initialized gated cross-attention branch has close [Flamingo](https://arxiv.org/html/2204.14198#S2.SS2) precedent. Summation has [Deep Sets](https://arxiv.org/abs/1703.06114) precedent, and cardinality-preserving attention has [CPA](https://www.ijcai.org/proceedings/2020/0194.pdf) precedent. The latter's results do not establish a universal cardinality limitation for a full positional transformer. MOST/LEAST answers survive uniform replication; normalized representations can preserve relative frequencies. Restoring scalar mass is therefore not a sufficient explanation or contribution for the primary tasks.

The first method comparison needs the ordinary checkpoint and a generic adapter with matched trainable parameters, supervision and training effort. Report additional computation: one forward is not equal compute. A useful gain must survive fresh relevant-evidence extrapolation, preserve factual retrieval and replicate across fit seeds. Native direct/reasoning comparisons must charge all generated tokens and measure latency. A reusable KV path is software feasibility, not evidence of useful reasoning composition. A second aggregation setting is eventually needed for a general-method claim.

The immediate deliverable is one small diagnostic report that chooses the intervention. It must not become an architecture grid or an automatic restart of closed memory branches.

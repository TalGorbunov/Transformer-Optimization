# Local joint-code learnability control

The learned aggregation arms have not fit the 108-context MMReD identity-join training screen. Their 600 updates are a fixed resource screen, not evidence of convergence or an information ceiling. The successful answer-code oracle bypassed the join entirely. The next diagnostic asks a narrower question: can the existing nonlinear readout learn the join within the same training recipe when each image supplies an exact local joint representation?

## Intervention and hypothesis

For each image independently, obtain its person and room from the frozen renderer's semantic record. With the two rooms in the question in their recorded order, form an 18-dimensional one-hot code indexed by nine persons times two requested rooms. Images outside those rooms contribute zero. Pad with 78 zero coordinates to rank96. Entries remain raw0/1: no square-root scaling, normalization, learned local encoder, or selector. The code constructor accepts only one frame's semantic labels and the requested rooms. It never receives the answer, scene intersection, other frames, target token, or prefix. Repeat each image's same code at every teacher-forced position.

Use the same half-SUM and native readout:

    z = 0.5 * sum_i code_i
    q = Wq RMS(g)
    delta = U SiLU(Wagg z + q + b)
    logits = frozen_native_head(frozen_native_norm(g + delta))

The readout must distinguish a person's presence in both requested rooms from two appearances in only one room. Equal person and room marginals do not determine that join. This control supplies perfect local semantics and question relevance, so it is privileged diagnostic data, not an inference method or an accuracy claim for a vision model.

Only query.weight, aggregate_projection.weight/bias and up.weight are trainable:697,440 parameters at hidden3584/rank96. Copy these four tensors exactly from the original UNFITTED factor plan443373 initialization; U is zero. Do not load any fitted checkpoint. There are no unused trainable encoder parameters.

## Fixed experiment

Use exactly the existing108 training contexts,54 N8/N16 pairs,18 complete families, seed24 and frozen600-update pair order. Preserve full-name-plus-EOS mean-per-scene CE, AdamW defaults, LR0.001,50-step warmup and cosine decay to0.00001, weight decay0, gradient clipping1. No residual-consistency objective. Globals, prefixes, target sequences, native FP16 norm/head and conversion rules remain the frozen cache's values. No additional VLM/vision forward, dataset generation, development, test, native generation or permutation arm is released.

Capture pre-update states at steps1,2,32,128,300,600 and score every training first query after step600 in seven fixed batches. Total607 norm/head calls and21,442 rows:600 optimization calls plus seven final batches. Retain all raw final logits and13 captures. The unchanged first-token screen is at least103/108 correct and16/18 families with all six answers correct. Full-target training loss, first-token/EOS losses and complete-cycle trends are descriptive; no early stopping or fitted-checkpoint selection.

## Binding and audit

CPU preparation independently parses each qa.txt state sequence, verifies its registered SHA and canonical scene content, question and image order, and binds all1,296 training image occurrences. Gold is used only for loss and scoring. Fixtures check code independence from gold/prefix, sensitivity to frame labels, equal-marginal/different-joint examples, exact padding, zero-up identity, gradients and parameter count. Current and archived inherited sources, new sources, selected initialization tensors, row order and native weight identities are bound.

An independent reporter reconstructs codes from semantic records and reconstructs all13 captured readout computations from saved weights. Replay the seven final native norm/head batches on CPU, preserving original numerical tolerances and the established FP64 reference for scoring stored logits. Report every outcome and family, call counts, and Slurm allocations including failures. Software acceptance and scientific screen acceptance are separate.

## Resources and decision

One CPU preparation allocation up to90seconds, one single-attempt GPU allocation up to90seconds with maximum one project GPU, and one independent CPU report up to300seconds are released only after source review. All heavy work runs through Slurm; models and data use the user-designated roots. No retries or extensions are implicit.

A pass establishes learnability of this privileged local-code/readout combination on these training examples. It motivates investigating how to learn reliable local joint representations from native states. A failure leaves the readout and optimization budget unresolved; it is not an impossibility result. Either outcome leaves all previous failed screens and their stopping rules unchanged. No follow-on fitting or benchmark claim is automatically released.

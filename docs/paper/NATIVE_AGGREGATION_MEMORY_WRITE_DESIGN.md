# Put the aggregate into the model's memory

**V11 software proposal, not an efficacy result.** V10 still fails supported-answer length extrapolation. Its result motivates checking a missing capability of the current adapter; it does not establish that this capability caused the failure.

The current adapter changes the final readout immediately before normalization and the vocabulary head. At a fixed supplied token prefix, that change never enters the backbone's KV cache. Later teacher-forced losses therefore cannot train earlier aggregation outputs through the backbone. Free generation provides an indirect path through discrete token choices, which ordinary teacher-forced answer CE does not differentiate.

This matters even for ordinary multi-token answers. In the current tokenizer,10 and16 share the first digit token1. A final-readout write at that first query receives its local token loss but no direct temporal gradient from the later digit loss. Shared parameters can still transfer information between positions, and absence of that path does not prove information is lost. Writing earlier provides an explicit route for the later digit to train the initial aggregate. Whether that improves the observed length failure is an experiment, not an assumption.

The minimal structural change is to write the aggregate **before the last frozen transformer block**. That block can process the current evidence and store it in its global attention K/V. No extra reasoning token, transformer block or learned parameter is added.

## A controlled comparison

Both versions read local and global hidden states after the penultimate block, at the same prediction positions. Both compute the unchanged query-conditioned SUM/SiLU residualδ. For Qwen's28 blocks, reads occur after index26 and the final block is index27.

LetH be the complete global sequence at that boundary, andB the final frozen block. Writes are placed at every assistant prediction positionP−1+t, whereP is prompt length.

- **pre_last:** passH+δ throughB, then the original final normalization and vocabulary head.
- **post_last:** passH throughB, add the sameδ at the readout positions, then the same norm/head.

The core sees identical lower-layer inputs for a fixed prefix in both versions. The write changes only the global row; all local rows and earlier layers remain untouched. Both execute one N+1-stream model invocation per output token. Holding the read boundary fixed avoids attributing a change in local features to the write location.

This comparison changes both current-token nonlinear processing and access to past aggregates. Better first-token accuracy would not identify a memory effect. A later fixed-prefix intervention restoring only earlier final-layer global K/V to its unfused values, while preserving the current write, could test whether that persistent path is used. That intervention is not yet released.

## Training and causal checks

Lower-layer states remain frozen and cacheable under fixed token prefixes. Reconstruct the entire global prompt plus target[:-1] from new penultimate features; inject atP−1+t, including the last prompt token. Run the actual final frozen block with native causal masks and rotary embeddings, use_cache=False, and a live input graph. Use the original norm/head for CE. No EOS target enters the input, and no trained KV cache is detached or reused between optimizer updates.

Independent liveδ leaves provide the decisive software check. A loss at position1 must have zero gradient toδ0 forpost_last, and measurable gradient forpre_last on the fixed diagnostic case. Futureδ2 must affect neither earlier logits nor that loss. The installed NF4 implementation supports input gradients, but their numerical behavior and cost require measurement.

Native cached decoding applies the new write once. An uncached full-prefix reference must apply **every historical assistant-query write** to reconstruct the same final-layer attention history. A last-query-only full-prefix reference would be wrong. Preserve native arithmetic as h+delta.to(h.dtype).

The first stage is CPU tests and a fixed39-model-call/19-vision-call GPU smoke, plus at most16 final-block-only replays. Required checks cover native-zero identity, masks/positions, causal gradients, unchanged lower/local caches, changed early final-layer global K/V, same-state head replay, and source/model identity. Cache-versus-full numerical differences remain visible. Software cap540allocatedGPU-seconds, with one profile up to300seconds. No training experiment or benchmark criterion is released until this stage is reviewed.

This is an internal write of an established conditional set aggregate. It is not a new attention family or a theorem that counting must improve. Its value would be a demonstrated native aggregation improvement and, separately, verified use across successive reasoning queries.

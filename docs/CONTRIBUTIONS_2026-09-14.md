# Contributions ledger (2026-09-14, after the peer meeting)

> Claude, for Tal. Companion to the shareable page (figures + full prior-art tables).
> Every number traces to RESULTS.md; prior art re-checked 2026-09-14.

## Verdicts

| # | candidate | verdict | closest prior | must happen |
|---|---|---|---|---|
| 1 | per-frame fence (block mask + posreset, frozen, training-free) | mask NOT novel; measurement novel | PCW/APE (text); **Das et al. 2601.07812 (Jan 2026): LoRA fine-tune w/ block-diag image mask in layers 12–23, cross-image counting 9%→45.8% (0.5B), 29.7%→51.2% (7B), still degrades to 35 images, no posreset**; PEVLM 2506.19651 (per-frame blocks, prefill speed); FOCUS 2508.13744 | cite Das first; ablate their placement (mask deep/open early) vs ours (mask early/open late); claim the measurement (Jacobian 0.52→0.00, d′ flat to 128, posreset dose, q/kv split), not the mask |
| 2 | Hahn exponent measured on frames (α 0.72 frozen / 0.80 trained / 0.00 fenced) | novel with precedent | Hahn 2020; Barbero 2406.04267; **Brändel et al. 2606.29139: Jacobian decay with token DISTANCE in Pythia/Qwen-0.5B, p≈0.8–0.9** | cite the distance paper; our variable = number of units, with fenced control |
| 3 | detection vs aggregation dissociation | novel with precedent | Orgad 2410.02707; 2605.09239; **2605.03258 (count decodable internally, readout-geometry failure, text)** | merge with #1 as "localization" (two proofs); add contrast: count NOT linearly present in joint state at N=32 (best linear 0.19) |
| 4 | share law photographed + knockout (α 0.80→0.035, byte-identical text) | novel | Veličković 2410.01104; SSMax; ASEntmax ICLR26; Screening; Du et al. 2510.05381 (no posreset); Das et al. (fence w/o gate → still decays) | label-free gate must reproduce it (selfgate G0) before it headlines |
| 5 | residual wall in k, γ=1.19 adapter-independent, GIN reading | measurement novel, interpretation CONTESTED | Yehudai 2407.15160; **2605.03258 blames output-head geometry for count-magnitude degradation** | separating experiment: apply their digit-row fix under our gate; does γ (hidden state, L20) move? Until then: "upstream of the output head, consistent with GIN" |

Recommended final list: (1) Localization [1+3]; (2) The law [2 + photograph]; (3) The knockout [gate; conditional on label-free gate]; (4) The residual wall in k.

## Problem definition

Adopt MMReD's dense-context vs NIAH distinction and define it by sensitivity structure:
retrieval = one unit carries the answer; aggregation = answer is a symmetric function of
per-unit facts (every flip matters). Computable with probe_hahn on any benchmark.
Benchmarks with per-unit evidence labels: MMReD (primary; only place the photograph/flip
work), Das et al. counting split (direct printed comparison), EC-Bench 2603.29943 and
CG-AV-Counting 2506.05328 (real video, evidence spans → oracle gate possible), HERBench
(boundary/null), MMNeedle/Visual Haystacks/VNBench (retrieval arm), RULER CWE/FWE +
BABILong (text port).

## Block definition (for text)

Block = span such that (i) the per-unit fact is a function of the span alone and (ii) the
gold answer is invariant to block order. (ii) is an operational test (shuffle blocks). Fence
licensed iff the task is a symmetric reduction over per-block facts; out of scope for
narratives/proofs (PCW-revisited's CoT finding). Text targets: passages/docs/rows/turns.

## Gating comparison arms (all label-free at test time except the first two)

oracle (done) · LR classifier (done, ≥0.999) · self gate L24h20 threshold (approved, unrun; FIRST) ·
top-k attention selection (SnapKV/H2O-style) · verbal per-frame yes/no gate (= retrieve-then-read) ·
learned Gumbel gate on answer loss (CoGNN-style; agnosticity claim) · FOCUS-style per-frame
isolation (bounds retrieval arm) · soft threshold at read layers (Screening-like).
Metrics: gate acc vs gold (eval only), ladder 8–128, parity-to-oracle by named errors, cross-question transfer.

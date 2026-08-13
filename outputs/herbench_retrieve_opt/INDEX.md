# HERBench retrieve-opt — INDEX

Canonical runs for the "lift the per-unit perception ceiling with ±δ temporal context
and/or higher resolution, no training" probe campaign. See CAMPAIGN_BRIEF.md / STATE.md.

| experiment | canonical run | headline |
|---|---|---|
| Phase 0 inventory & feasibility | `phase0/inventory.json` | route CONFIRMED (videos on disk, PyAV); δ grid {0.5,1.0}(+2.0); verbs pick/open/close dominate; disk ≈2.8 GB |
| A0 anchor @448 (gate) | `probe/A0_r448/20260802_134415` | pooled d′ L16=1.05 L18=1.08 → **in archived 0.98–1.10 band, GATE PASS** |
| B-δ0.5 @448 (temporal) | `probe/B_d0.5_r448/20260802_141853` | pooled d′≈1.06 (flat vs A0); per-verb L18 open 1.67 / close 1.52 / pick 0.96 — transition-verb temporal gain, none clear 2.5 |
| Step-2 δ-sweep (B/C/A1, full layers) | `probe/{B_d1_r448,B_d2_r448,C_d0.5_r672,C_d1_r672,C_d2_r672,A1_r672}/2*` (jobs 127976/7/8) | per-verb peak d′: open→2.36, close→2.15, pick≤1.69; put small-n crosses; res a minor combined boost |
| **Phase 2 verdict** | `phase2/verdict.md` + `fig1_dprime_vs_delta.png` + `fig2_per_verb_best_arm.png` | **NULL CONFIRMED (well-powered) w/ strong PARTIAL at margin** — temporal δ lifts transition verbs open/close to ~2.2–2.4 (approach, don't clear 2.5); pick flat; res≤672 minor. best cell C δ2·672 |
| Phase 3 (gate→tally + distill plan) | _conditional, awaiting Tal GO_ | — |

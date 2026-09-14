# MMReD suffix feasibility: completed result

CPU audit 444121 completed successfully in 7.669 measured seconds (7 scheduler seconds). Its frozen protocol sampled 20,000 original-law worlds at each of N16 and N32, proposing one person swap and one room swap per world. All 40,000 source worlds were distinct within their length. Completion required transition legality, cut-state preservation, involution, exact integer decomposition, and agreement with the original full-span answer oracle; acceptance yield was never a pass criterion. There were 47,688 full-answer and 47,688 invariant-control oracle checks, plus 12 of each in fixtures.

The useful strict diagnostic is MOST with unique full, prefix-only and suffix-only winners on both members; the full answer changes, at least one full winner is an unaffected third competitor, and neither fixed partial-span rule answers both members correctly. Every row below has 20,000 proposals as its denominator. “Non-global” also excludes unchanged worlds.

| N | Swap | Legal | Non-global | Unique changed MOST | Strict third-competitor MOST | Strict yield |
|---|---|---:|---:|---:|---:|---:|
| 16 | Person | 3,297 | 2,401 | 447 | 16 | 0.080% |
| 16 | Room | 2,645 | 1,931 | 155 | 15 | 0.075% |
| 32 | Person | 3,315 | 2,875 | 644 | 65 | 0.325% |
| 32 | Room | 2,665 | 2,323 | 292 | 66 | 0.330% |

The changed-MOST column is restricted to non-global cases. The analogous strict LEAST counts were only 4/0 at N16 and 38/1 at N32 for person/room. A common MOST diagnostic therefore has empirical support; balanced LEAST diagnostic quotas do not. Both directions remain appropriate in a separately sampled original-task evaluation.

The symbolic person-swap witness and its N16 extension passed actual program/oracle validation, including equal person-room histograms and a changed unique answer. Random sampling nevertheless found zero equal-histogram changed-answer cases at either length. This establishes existence but provides no useful prevalence estimate for an equal-histogram cohort. For room swaps, preserving every person's room histogram would itself preserve every queried room total, so that condition cannot yield a changed `where_spend` answer.

These cases establish a feasible diagnostic distribution, not model performance or an aggregation-capacity limit. A person swap preserves the queried actor's complete trajectory, per-frame room occupancy, and number of companions; it generally changes other people's room marginals. A room swap preserves all co-location relations but changes atomic room labels in the suffix. The stronger criterion defeats two specified partial-span baselines on paired correctness; prefix failure alone is automatic whenever a shared prefix has two different full answers. For a strict two-way winner flip, the unchanged algebra D_old=P_diff+S_diff and D_new=P_diff−S_diff forces |S_diff|>|P_diff|, so the suffix alone resolves both two-way winners. The third-competitor restriction and explicit baseline outcomes remain necessary for the proposed narrower diagnostic.

Historical world overlap was explicitly deferred. Duplicate transformed worlds, including no-op transformations, were counted rather than excluded. Neither the stored examples nor these 40,000 worlds are an approved evaluation cohort. Fresh sampling must exclude both sampled sources and transformed worlds, as well as recovered and otherwise exposed worlds.

Independent lightweight verification matched summary SHA `fe369079c1dbcbbf6c42804141cf02574c889e5adaeab47548a968ba51e1094a`, analysis SHA `c6c636d026840bb9de989472f647340a5168c822bb22ffe9dff87d4ebc118a18`, artifact-manifest SHA `b25688cae52f8fdf89261dc3c418f44a30e543e3398d98ce0bc0c7176e126b97`, 12 direct metadata/source bindings and 76 small artifacts totaling 1.19 MB. This included archived sources and the validated fixtures. The 36.3 MB trial ledger was not replayed or rehashed in this lightweight review. No model outputs, tensors, images or new jobs were accessed.

The source report is [audit 444121](../../outputs/native_aggregation_vlm/mmred_suffix_feasibility/audit_444121/REPORT.md); the immutable sampling rules are in [the feasibility protocol](MMRED_SUFFIX_FEASIBILITY_PROTOCOL.md). The [new cohort proposal](MMRED_FRESH_EVALUATION_COHORT_DESIGN.md) remains design only.

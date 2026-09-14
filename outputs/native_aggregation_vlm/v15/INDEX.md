# V15 native learned-versus-bank diagnostic

Registered exploratory comparison of all four frozen V14 models,34reused contexts/model,two matched N+25 mean sources. No fitting or efficacy claim.

Source/software preparation in progress; no V15 jobs or outputs yet. See [preregistration](../../../PREREG_AGG.md).

- CPUsoftware442638 passed23s,7controller tests; [plan](null_comparison_software/check_442638/plan.json).
- NativeGPUsoftware442641 running,300GPU-second cap; capture-enabledtiming,<=48model/20vision/48heads.
- V14fullyverified3197GPU-seconds; fixedfinalallfourmodelidentities boundbyfinal442626.

- Native software442641 passed92GPU-seconds,40model/20vision/40heads,520cached/full comparisons allpass.
- StudyCPU442669 didnotexecute (missingentrypoint), full442678failed34s onwrongnativeAPI provenancehelper; bothpreserved. Import-onlycorrected4ee71source/fullCPU442691passed105s. [Plan](null_comparison_study/check_442691/plan.json), SHA37ffc4b9d69ba94487273eb5c07a730c0b1fda57ddadd6ba7d8e79cfeeb84512.
- Fourstudyarray442707 running: centered18/19=442708/442709,offset18/19=442710/442707; max900s/job,projected688.594557s.
- Report442711 followsallstudies. Geometryselftest442702passed8checks, geometry442712 andplot442713 followreport. No V15accuracyinspectedyet.


**Verified result: live-bank substitution does not rescue centered N64.** Learned→bank centered18N32 16→16/17,N64 9→3/17; centered19N32 13→16/17,N64 7→5/17. Offsets unchanged4/2and6/2 atN32/N64. Independentreport442711passed21CPU-s, all272trajectories/637prefixes. All136augmentedlearned sequences matcholdN+1descriptively. Total990GPU-seconds (92software+898studies),max4. [Report](null_comparison_study/report_442711/REPORT.md), [geometry](message_decomposition/decomposition_442712/REPORT.md), [figure](figure_442713/INDEX.md). Geometrypassed15CPU-s; rootvisuallyinspectedPNG. No acceptedmethodorpracticalmilestone.

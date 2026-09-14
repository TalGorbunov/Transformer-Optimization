# Resource-only V2 of the fixed trainability diagnostic

CPU preparation443189 is running. The four-condition scientific protocol remains unchanged: clip/sigmoid × CE/CE+consistency, shared seed24,600updates on108training contexts, one final108train+54dev native evaluation. No test evaluation or efficacy confirmation.

V1 profiles all passed computationally but independent release443185 projected751.13seconds, exceeding its600-second cap. Preserve that failed protocol and452GPU-seconds. V2 allocates840seconds per main and4500totalGPU-seconds including allV1/future failures. Fresh four150-second profiles and independent measured release are required; maximum4projectGPUs.

CPU443189 passed229seconds; root verified64source snapshots. Four fresh32-step profiles submitted asarray443200 with maximum4GPUs. Main fits remain held pending independent release.

Four profiles passed455GPU-seconds; cumulative907includingV1. Independent release443204 passed60CPU-seconds, projection695.71≤840, complete reservation4267≤4500. Four600-step mains submitted asarray443207 (max4GPUs); outcomes not yet audited.

STOPPED: allfour mainsfailed49/50/50/48GPU-seconds atstep200 with FileExistsError on reused training.json. No finalcheckpoint/nativeevaluation. Allpartialartifacts preserved; cumulative1104GPU-seconds. SeparateV3 logging-only repair held pendingsource/CPU/releasechecks.

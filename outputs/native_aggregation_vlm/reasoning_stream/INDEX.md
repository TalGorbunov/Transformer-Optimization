# Cosmos native streaming software

- [Seven CPU tests](self_test_441990/summary.json): passed11allocatedseconds.
- [Input/source plan](check_441991/plan.json): CPU441991 passed16seconds.
- Registered GPU smoke:3min, native/zero/active on two fixed scenes, max8natural tokens and active-prefix replays. Separate0.1GPU-hour cap including failures; waits for V8 mains.

No training, accuracy or reasoning-composition result. Full long-cache execution remains untested.

GPU smoke441994 is queued with afterok:441966, so it cannot run alongside all four V8 mains.

[GPU441994](profile_441994/REPORT.md) passed:121allocatedGPU-seconds,64model/22visual calls,48natural tokens,16active prefixes,32native-head replays. All cached/full numerical checks also passed. Native FP16 norm/head. Script measured54.29seconds including15.57model load; allocation includes scheduler/batch startup. This does not establish long-cache feasibility or reasoning efficacy.

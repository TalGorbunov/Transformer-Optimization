# Independent ordinary joint LoRA audit

CPU norm/head TV must be at most .02. CPU/GPU argmax differences are descriptive; all efficacy uses actual GPU-generated IDs. Exact GPU zero-adapter and serialization parity remain mandatory. Earlier reports retain their original gates.

Numerical audit: PASS. CPU head calls: 504; selected query rows: 532; maximum TV: 1.1699202749532512e-06.
CPU/GPU argmax differences: 0. Every scheduled numerical audit was collected before this decision.

Natural complete name plus EOS: 216/216; orientation counts: {'original': 108, 'flipped': 108}; complete families: 18/18; competence passed=True.
All2,592 teacher traces and5,760 target CE positions were audited. Native CPU replay covered the fixed24 first/last-per-epoch examples and every final generation query. A valid negative competence result is not a computational failure. No fresh inference is released.

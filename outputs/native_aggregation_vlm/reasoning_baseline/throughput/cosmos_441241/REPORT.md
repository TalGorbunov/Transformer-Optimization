# Forced-cap timing calibration

Model cosmos;4 timing-only trajectories,1088generated tokens. No output is scored.

Projected main worst-cap time: 1250.35s; within26minutes: True.

Maximum wall decode seconds/token: {'direct': 0.02868833482289506, 'reason': 0.02493399826260812}.

- Software timing only: EOS suppression forces full caps. These outputs are not natural reasoning traces and have no accuracy interpretation.
- The original software-profile cost gates remain failed and are not replaced or silently reclassified.
- Prompts, data, models, main-policy budgets and parsing remain frozen. Only this separate timing procedure suppresses EOS.
- Two fixed software scenes measure sustained native cache cost; the projected bound is conservative but not a guarantee.
- The512-token main policy remains far below the official Cosmos recommendation; the timing calibration does not expand its budget.

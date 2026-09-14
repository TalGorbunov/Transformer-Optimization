# S1 trace anatomy — outputs/cot_anatomy/Qwen3-8B/20260907_110935_4003321

## S1a — written tally and corruption test
J=0: acc 0.94, tally written 0.97, corruptions 35: follow 0.63, repair 0.00, answer shift +1 0.06, shift 0 0.91 (n=35)
J=64: acc 0.25, tally written 1.00, corruptions 36: follow 0.86, repair 0.00, answer shift +1 0.00, shift 0 0.08 (n=36)

## S1b — retrieval share of the chunk onset on its target sentence (fraction of context attention)
J=0 K= 4: median target ratio 0.450, target share 0.019, on-needle 1.00, chunks/trace 18.6 (n=11)
J=0 K= 8: median target ratio 0.430, target share 0.019, on-needle 1.00, chunks/trace 17.1 (n=10)
J=0 K=16: median target ratio 0.416, target share 0.018, on-needle 1.00, chunks/trace 35.8 (n=12)
J=64 K= 4: median target ratio 0.082, target share 0.004, on-needle 0.53, chunks/trace 132.1 (n=12)
J=64 K= 8: median target ratio 0.073, target share 0.005, on-needle 0.80, chunks/trace 145.0 (n=12)
J=64 K=16: median target ratio 0.095, target share 0.006, on-needle 0.87, chunks/trace 154.2 (n=12)

J=64: AUROC(low target ratio -> failure) = 0.545 (fail 28, ok 8)

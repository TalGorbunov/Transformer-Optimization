# Native prefix-supervision main-fit audit

This separate CPU audit consumes one completed fit from the already released
answer/local/prefix by seed25/26 study. It changes no trainer, core, data, targets,
checkpoint, prediction, or numerical gate. A valid negative training result is a
successful computational audit; no accuracy threshold selects a checkpoint or
permits evaluation. Fresh evaluation requires its own release.

Each fit receives one CPU job of at most3,600seconds, four cores and16GiB. This is
the same allowance as the analogous frozen official native-training auditor with
six teacher captures and100 natural training diagnostics. A60second publication
reserve is checked during work. The six possible audits therefore have a maximum
combined allocation of21,600CPU-job-seconds. No GPU, vision, decoder, backward or
optimizer is invoked; there is no re-fit or automatic retry.

The audit verifies the actual consumed source/main releases, passed independent
software profile, original4000 inputs, unchanged target/projection descriptors,
original fitted ordinary1500 initialization, and actual state0/1499/1500 packets.
It checks all224 FP32 language-attention adapter tensors,10,092,544 trainable
parameters, zero auxiliary trainable parameters, optimizer policy and saved
moments/steps. Three advancing seed-specific shuffles are independently rebuilt;
all12,000 scalar presentations and1,500 updates must match. The saved mean native
answer CE, active auxiliary mean and coefficient0 or.1, total loss and division by
eight are checked for every presentation. Per-presentation gradient presence,
finiteness and28-layer accumulated norms are GPU evidence, not an independent CPU
decoder-backward proof. Main fits do not retain all gradient vectors.

For the first and last training presentations plus two before/two after reload
teachers, all six saved native head calls are replayed with the original frozen
FP16 norm/head packet. Native CE is independently recomputed in FP64. The actual
pre-block27 image-end rows, frozen40-dimensional projection and affine buffers,
original local/prefix targets, predictions, decoded coordinates, mean auxiliary
loss and retained prediction-gradient coefficient are audited with the immutable
software-audit mathematics. All six auxiliary captures are covered. The complete
4000-world integer/normalization/QR proof is inherited from the passed software
audit with exact packet hashes, not recomputed using a different population.

All100 fixed original training diagnostics are audited unconditionally: actual
native greedy IDs, original typed JSON parser, EOS, complete selected head
queries, input/cache/position history, final checkpoint ownership and absence of
supervision hooks. No teacher target is supplied to native generation. CPU
norm/head TV must be at most.02; CPU/GPU argmax differences remain descriptive.
Actual same-GPU checkpoint reload logits must be exactly equal. Saved native CE
uses2e-6 absolute tolerance; auxiliary FP64 comparisons retain2e-4 absolute and
relative tolerance. Every available numerical capture is collected before an
overall numerical failure; ownership errors and the time reserve may halt early
with retained partial evidence.

There are at most5,006 CPU head calls and5,300 selected rows per audit. Physical
GPU model calls are12,004 plus the actual generated token count, with zero vision
calls,12,000 backwards and1,500 optimizer updates; all28 decoder counters must
agree. The original per-fit7,200second allocation and observed timing/memory are
reported separately from CPU replay cost. Replay tensors live only under the
designated `/mnt/data/gabriele/gnn_transformer` audit directory; checkpoints remain
references to `/mnt/ckpts/gabriele/gnn_transformer` without another copy.

`verify_report(summary_path)` is a JSON/hash-only handoff returning the bound
arm, seed, task index and final1500 checkpoint, exact adapter tensor identities,
original input/release/profile lineage, and the computational audit result. Its
consumer hashes the single final checkpoint when loading it; raw audit tensors
are not rehashed by this handoff. `verify_matched_reports(paths)` additionally
requires all six arm/seed combinations and common original initialization/data.
Only the three new auditor sources are copied; the frozen producer source map
is referenced and verified rather than copied into a new ancestor tree.

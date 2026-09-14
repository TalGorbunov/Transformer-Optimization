# Native MMReD privileged supervision: fixed execution protocol

Three arms continue the competent ordinary final checkpoint from job444013_0:
answer-only, local40 supervision, and prefix40 supervision. Use the original
4000 training questions at N1/2/4/8/16, original native prompt and images, original
28-layer QKVO language-only rank16 alpha32 dropout0.05 LoRA, unchanged NF4 base
and frozen vision features. There are224 FP32 trainable tensors and10092544
parameters in every arm. No learned auxiliary parameters or inference modules.

Use continuation seeds25 and26. Each seed initializes from the identical
competent checkpoint and sets ordinary torch CPU/CUDA RNG for dropout. Three
epochs use a private Python Random(seed) shuffle of all4000 rows per epoch.
All arms of a seed share order and initial RNG. These are two continuation
seeds, not two independently trained ordinary backbones. There are12000 native
forward/backward presentations, accumulation8 and1500 AdamW updates per fit:
lr2e-4, betas0.9/0.999, epsilon1e-8, weight decay0.01, global clipping1.0.
Use final checkpoints only; validation/test predictions do not occur during
training. The original fixed100 training diagnostics are descriptive.

The40 targets are original-order30 person-room indicators and10 unordered
person-pair co-location indicators. Local targets are instantaneous; prefix
targets are inclusive sums. The deterministic Gaussian FP64 QR projection
seed20260914 has40 orthonormal rows in3584 native dimensions, positive R diagonal
sign convention, and stored FP32 coefficients. Mean and population variance are
computed using equal weight per original training entry and equal weight per
prefix within that entry. Semantic duplicate entries keep their original
training multiplicity. For variance<=1e-12 use scale1; otherwise sqrt(variance).
Round coefficients to FP32 once; reuse at all test lengths without clipping.

Observe existing image-end states at the input to decoder block27. No hidden
RMS or state replacement occurs. The original ordinary forward remains intact,
and all lower LoRA states are recomputed with gradients. Auxiliary MSE averages
over N times40; total loss is mean native answer CE plus0.1 times MSE. Answer-only
uses only native CE; optional retained auxiliary observations carry no gradient
term. Divide the complete objective by accumulation before backward. Matmul
precision is highest, CUDA FP32 TF32 disabled; record actual settings. Projection
and affine tensors are buffers external to the native model and are never read
by ordinary generation. Existing JSON+nativeEOS contract and50-token cap remain.

CPU target preparation is600 seconds, four cores,16GiB. It checks the original
oracle answers, physical image-end positions, source/compact ownership, private
projection RNG, FP64 reference math and gradient fixtures. Every tensor or heavy
CPU operation runs on Slurm. Models use the designated CKPT root and data use
the designated DATA root.

One software GPU profile is capped900 seconds on one B200. It uses only fixed
original training indices0,5,800,810,1600,1612,2400,2405,2410,3203,3210,3222.
Two software AdamW updates per arm each accumulate all12 witnesses. Restore the
same initial ordinary state/RNG before each arm. Retain all72 training teacher
packets,2 baseline native teachers,6 unchanged-initial-state comparisons and12
checkpoint roundtrip teachers:92 teacher calls. Baseline and final generation
use indices0/3222:8 natural trajectories total. Actual model/head/norm and every
layer calls must equal92 plus actual generated-token count, with0 vision calls,
72 ordinary backwards,2 separately counted auxiliary autograd probes, and6
optimizer updates. Probe the auxiliary gradient path on the first local and
prefix teacher calls: lower layers0..26 must receive gradients, layer27 must
not. Clear retained intermediate prediction gradients after these probes before
the ordinary backward. Save actual training prediction gradients for independent
coefficient reconstruction. Save all six pre-clipping update-gradient tensors
and each arm's initial, intermediate and final software states/optimizer/RNG.

The passive observer must preserve initial native teacher logits bit-for-bit
under matching no-grad execution. Final checkpoint reloads must also preserve
same-shaped teacher logits exactly. The independent CPU audit reconstructs
all retained auxiliary calculations in FP64 and actual native norm/head rows.
Core absolute/relative tolerances are2e-4; native CE tolerance2e-6 and CPU head
TV<=0.02. Native CPU argmax differences are descriptive. No full decoder or
training replay is claimed. Software natural-answer accuracy is not a gate.

Six fits run in one array with at most three concurrent project GPUs, leaving
one GPU for shared fresh evaluation features; total project concurrency<=4.
Each fit is capped7200 seconds and additionally requires a measured profile
forecast and an independent passing profile audit. The conservative forecast
includes setup, native teacher/backward work, optimizer/checkpoint work, four
roundtrip teachers and100 maximum-budget training diagnostic responses, plus
six retained teacher packets,25percent margin and60seconds. Auxiliary-only
gradient probes and full gradient-file serialization are profile-only work;
exclude them from the per-presentation and optimizer forecasts, while charging
their measured profile allocation. Price the six main retained teachers once.
The main release must bind the exact passing profile, independently reconstructed
forecast, source release and prepared targets before model loading.
If it exceeds the cap, do not launch the fit
under an impossible allocation; any resource-only change is separately recorded
before further execution. Production latency is measured separately from
instrumented capture time. All failed attempts and actual work remain charged.

Fresh main1400 questions are prepared separately, with400 N32 primary
partner/room questions and200 N32 counting/retrieval controls. Their statistical
comparison, unassisted evaluation, causal intervention panel and inference
cost measurement require separate prospective execution protocols. Oracle
supervision and causal state patching are established prior art; this software
release makes no novelty, capacity or successful research-objective claim.

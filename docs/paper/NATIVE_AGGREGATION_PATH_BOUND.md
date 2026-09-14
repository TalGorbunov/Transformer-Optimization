> Completed V9 result: both primary/practical criteria failed and N64 performance worsened in both seeds. The derivation remains valid within its stated scope; the penalty is not retained as the improvement. [Verified evidence](NATIVE_AGGREGATION_VISION_V9_RESULTS.md).

# Controlling sensitivity along equal-answer evidence changes

V8 improves native length extrapolation substantially, yet residual agreement at two training lengths does not constrain the decoder between or beyond them. Its selected models still fail at64frames when given repeated familiar local states. The next hypothesis is that controlling the decoder's sensitivity along those observed evidence-change directions gives more reliable aggregation. This is a registered experiment, not an established result.

Keep the entire deployed operator unchanged. Let z be its learned sum, r=Wagg*z+q+b the preactivation, and delta(r)=U*SiLU(r) the native residual. Equal-question/equal-answer training pairs yield d=Wagg*(z16-z8); the shared query and bias cancel. With u_j the columns of U, use

```
B(d) = sum_j ||u_j||_2 * |d_j|
L_path = mean_pairs,positions B(d)^2 / (||frozen_global||_2^2 + 1e-6)
```

SiLU is globally Lipschitz with a constant below1.1. The triangle inequality therefore gives, in real arithmetic,

```
||delta(r+t*d) - delta(r)||_2 <= 1.1 * |t| * B(d)
```

for every base r and real t. Endpoint agreement can arise from nonlinear cancellation; a zero B makes the decoder constant along the measured direction at every base. This retains the nonlinear decoder needed for recurring digit outputs. The loss is unchanged by reciprocal per-channel rescaling of U and d, which is a property of the penalty, not generally a function-preserving symmetry of a SiLU network.

The bound is deliberately conservative. It ignores cancellation between output columns and actual activation slopes. It may suppress useful variation in paired scenes and reduce count accuracy. Equal numerical coefficients for this penalty and the endpoint loss do not equalize their effective strengths. Neither the bound nor its training average covers unobserved directions, later Step appearances, arbitrary reasoning prefixes, FP16 rounding, RMS normalization or final vocabulary margins. No accuracy certificate or general counting theorem follows.

The comparison uses CE plus V8 residual consistency versus CE plus this bound penalty, each with coefficient1. Both conditions retain the same paired inputs, initialization per seed,4,860updates, native decoding and checkpoint selection. Two new seeds12/13 and452fresh tests prevent reusing V8 test outcomes as confirmation. Both conditions compute and log both penalties. The primary/practical criteria retain the requirement of at least76/108correct N64 answers in each treatment seed.

The relevant lineage is directional invariance training and sensitivity bounds. [Tangent Prop](https://proceedings.neurips.cc/paper/1991/file/65658fde58ab3c2b6e5132a39fae7cb9-Paper.pdf) penalizes selected directional derivatives. [Path-SGD](https://arxiv.org/abs/1506.02617) develops path-product geometry and rescaling invariance for a different, full-network quantity. [Lipschitz-Margin Training](https://arxiv.org/abs/1802.04034) links sensitivity bounds with margins; the present residual bound lacks the additional ingredients for prediction certificates. The empirical question is whether a directional upper bound is a useful native aggregation constraint. It is not a new attention family or a novel Lipschitz inequality.

All computation runs through Slurm. Models and data remain under their prescribed roots. [V8 evidence](NATIVE_AGGREGATION_VISION_V8_RESULTS.md) · [V9 artifacts](../../outputs/native_aggregation_vlm/v9/INDEX.md).

[Derivation and exact-invariance scope](../theory/native_aggregation_path_bound.md) explain why a zero penalty controls a measured direction at every preactivation base, and why that requirement can be conservative.

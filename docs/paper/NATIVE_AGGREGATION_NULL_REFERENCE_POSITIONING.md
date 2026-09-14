# Positioning the reference-centering direction

The useful question is why a learned additive visual readout fails as irrelevant evidence grows, and whether correcting its background contribution restores complete native answers. The current first-token result supports a narrow corrective mechanism on a reused subset. It does not establish a new attention family, general aggregation capacity, or reasoning composition.

Several parts of the proposed explanation already have close precedents:

| Prior work | Established idea | What our experiment would need to add |
|---|---|---|
| [Deep Sets](https://arxiv.org/abs/1703.06114) | Learned elementwise features followed by sum pooling and a nonlinear readout. | SUM and conditional local encoding are foundations, not novelty claims. |
| [NetVLAD](https://arxiv.org/abs/1511.07247) | Learnable pooling of local descriptor residuals relative to anchors. | Centered local descriptors are established; the issue here is query/prefix-conditioned background drift inside a pretrained native decoder. |
| [Contextual calibration](https://arxiv.org/abs/2102.09690) | Content-free reference inputs estimate prediction bias for output calibration. | Our correction uses known-null image messages in an intermediate additive readout and scales with the number of actual images. Reference-based bias removal itself is not new. |
| [Differential Transformer](https://arxiv.org/abs/2410.05258) | Subtraction of two attention maps to suppress noise. | A generic noise-cancellation narrative would overlap directly. We do not change attention maps or claim subtraction as an invention. |
| [DEX](https://arxiv.org/abs/2505.16333) | Learnable differential operations reuse pretrained attention outputs, including an output-space transformation O(I-W_D). | Learned subtraction in a pretrained model is also established. Any contribution must rest on a specific failure mechanism, intervention and extrapolation evidence. |
| [HiLS](https://arxiv.org/abs/2607.02980) | Hierarchical local reading and global aggregation with explicit chunk/mass structure. | Local/global organization is close prior art. Reference centering concerns the accumulated irrelevant contribution, and needs matched accuracy and cost evidence beyond the organizational similarity. |

The testable mechanism is that a SUM readout accumulates a nonzero expected irrelevant message. For a fixed query and generated prefix, centering gives `sum(phi(actual)) - N * E_null[phi]`. Additivity follows from SUM. Expected neutrality depends on a valid null distribution; it does not make each irrelevant item zero. Estimating the mean from M independent references introduces an error covariance proportional to N²/M, and correlated references can be worse.

V12 deliberately uses a more limited correction, `(N-16) * estimated_background`, to preserve the coordinates of four already trained readouts. It processes24 known irrelevant training frames at each actual prefix in the same model call. The extra streams, question-specific labeled bank and N16 anchor are material assumptions. This is an interpretable diagnostic, not yet the elegant final method.

If complete answers and fresh-family confirmation work, the next substantive experiment is matched training with an explicitly centered representation, removing the N16 anchor. A learned null expectation could remove the reference-bank dependency only if it works at unseen questions and arbitrary prefixes. Low null variance and transfer across nuisance distributions should be tested. A separate reasoning-model experiment must establish that the benefit survives and compounds across generated reasoning tokens; the direct-answer result cannot establish that.

The prospective paper claim should follow the evidence: identify a failure of additive aggregation under irrelevant-set growth; verify a directional correction with a same-cost orthogonal control; then show that a compact learned formulation restores native extrapolation under matched training and inference costs. Until those steps pass, use the language of a promising mechanism and an unfinished method. See the [design](NATIVE_AGGREGATION_NULL_REFERENCE_DESIGN.md) and [completed first-token diagnostic](NATIVE_AGGREGATION_BACKGROUND_DIAGNOSTIC.md).

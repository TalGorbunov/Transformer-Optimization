# V13: conditional centering removes a mean, not uncertainty

Prospective interpretation memo, written while the four V13 main runs are active. No V13 outcomes were inspected. This records analytical assumptions and possible diagnostics; it releases no computation, fitting, or additional experiment. The implemented method and gradient routing are in [the learned-null proposal](NATIVE_AGGREGATION_LEARNED_NULL_PROPOSAL.md), [the predictor helper](../../gnnformer/conditional_null_mean.py), and [the trainer](../../scripts/train_native_vision_v13.py).

The central principle is a change of origin for an additive statistic. Fix a checkpoint, question, and answer prefix, and write their common global state as \(g\), query as \(q\), image message as \(m_i=\phi(h_i,g)\), and predictor as \(c(q)\). The centered statistic is

\[
 z_c=\sum_{i=1}^N m_i-Nc(q).
\]

Let \(\mu_1(q)\) and \(\mu_0(q)\) be the positive and null message means, and let \(e=c-\mu_0\). If these conditional means are independent of set size and image position, a bag containing exactly \(K\) positive images satisfies

\[
 \mathbb E[z_c\mid q,K,N]=K(\mu_1-\mu_0)-Ne.
\]

Exact mean calibration removes the expected background dependence on \(N\). It neither makes each negative image zero nor makes the nonlinear residual additive. For the offset control, \(z_o=\sum_i m_i-c\), the corresponding mean is \(K(\mu_1-\mu_0)+(N-1)\mu_0-e\). Both arms retain the same 18,624-parameter predictor and 1,041,600-parameter original core. Their comparison isolates the registered use of cardinality scaling, including the optimization changes it causes. It does not establish equal function classes, a capacity theorem, new attention, or a larger representation rank. Multiplication by \(N\) also scales main-loss predictor gradients and can change clipping and coadaptation.

The assumptions matter in this dataset. A common N-agnostic global prompt makes \(g\) identical at a fixed question/prefix, but the rendered images still carry Step labels. A null bank drawn from N8/N16 need not have the same position-conditioned mean as N32/N64 images. More generally the expectation contains \(\sum_i(\mu_{0,i,N}-c)\), together with the position-dependent positive contrast. The formula also fixes the prefix externally; it is not automatically a statement about the image distribution conditional on an endogenously generated prefix.

Under independent message draws with fixed class counts, conditional covariances \(\Sigma_1,\Sigma_0\), and a deterministic fitted predictor,

\[
 \operatorname{Cov}(z_c\mid q,K,N)=K\Sigma_1+(N-K)\Sigma_0.
\]

Thus background variance grows linearly with the number of negative images, and its typical norm grows as \(\sqrt N\), under these assumptions. Correlated images, shared position effects, and conditional sampling add cross-covariance terms; independence cannot be inferred from having separate model rows. A fixed predictor error is a bias of magnitude proportional to \(N\), with squared error proportional to \(N^2\).

If a fixed message map instead uses an independent unbiased mean estimated from \(M\) IID null observations, uncertainty across independently sampled banks adds \(N^2\Sigma_0/M\) to the covariance. This is not extra random noise on every use of one already fixed bank. Nor is it an exact variance formula for V13: its 24 occurrences can repeat visual content, the core and predictor were trained using the bank, and the predictor is a shared learned function. Effective sample size, correlation with the fitted model, mean approximation error, and distribution shift must remain separate.

In the implementation, the decoder receives \(Wz+q+b_\rho-NWc\), followed by SiLU and \(U\), then native FP16 cast-before-add, final RMS, and vocabulary head. The relevant mean error is therefore \(We\), not just \(e\). Main CE and residual consistency differentiate through both \(q\) and \(c(q)\); auxiliary regression detaches the current query, message target, and second-moment denominator and updates only the predictor. This prevents a direct auxiliary gradient into the core. It does not fix the moving target or prevent main-loss coadaptation. A low normalized auxiliary loss may coexist with a large projected error, an enlarged readout gain, or poor count separation.

A bounded diagnostic sequence can use all four already selected checkpoints and the existing training cache, without new VLM forwards or selecting favorable questions:

1. **Account for predictor error on the fixed bank.** For every one of the 972 question/prefix groups, retain all 24 occurrences, recompute the current message mean using FP64 accumulation followed by FP32, and compare it with the predictor. Report raw error, \(\|W(c-\hat\mu_0)\|\), predictor/target norms, and the existing normalized loss. Summarize every prefix and question. Logged auxiliary group IDs and projected errors show whether calibration lags or changes during training; measurements at different steps concern different core weights. This is training-bank fit, not independent null calibration.
2. **Separate mean error from spread.** On those same messages, calculate covariance in projected coordinates and the separate means of the N8 and N16 source banks. Show \(N^2\|W(c-\hat\mu_0)\|^2\) alongside the IID reference quantity \((N-K)\operatorname{tr}(W\hat\Sigma_0W^\top)\), explicitly labeling the latter an assumption-based estimate. Preserve repeated occurrences and report duplication. The two bank means reveal heterogeneity, not a held-out test of a predictor trained on both.
3. **Locate sensitivity in the existing readout.** At every ordinary cached training prefix, compare the deployed preactivation/residual with the same frozen core using its empirical training-bank mean. This substitution is an oracle diagnostic of fitting that bank, never a proposed inference method. Report the change alongside the local readout Jacobian \(U\operatorname{diag}(\mathrm{SiLU}'(t))W\) acting on mean-error and covariance directions. Large fitting error supports a calibration limitation; low fitting error with substantial amplified spread supports a remaining variability problem. Neither observation alone proves that the native decoder is the sole bottleneck. Establishing accuracy effects would require a separately specified native-head evaluation, not an inference from residual norms.

Finally, current-query reuse is compatible with one batched native forward per generated token, but it does not create persistent continuous reasoning memory. V13 writes immediately before final RMS. At a fixed teacher-forced token history, earlier branch residuals do not enter later native K/V or hidden states: a later token loss has no direct gradient to an earlier residual. Shared parameters still learn across positions, and changed generated tokens can affect later queries through their discrete history. Numeric-prefix coverage and complete-count accuracy therefore cannot establish reasoning composition or credit assignment through past aggregation states.

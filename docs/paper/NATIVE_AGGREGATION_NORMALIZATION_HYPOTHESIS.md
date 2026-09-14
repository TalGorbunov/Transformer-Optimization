# Conditional normalization hypothesis

Planning only, written while V5 trains; not an observed failure or another method arm.

Hold query and average message fixed. Let x=h+a denote the complete residual after ordinary attention and let the extensive branch be N m. RMSNorm sees z=x+N m and R(z)=g ⊙ z/ρ, where ρ²=||z||²/d+ε. Its count sensitivity is

\[
\frac{\partial R}{\partial N}=g\odot\left(\frac{m}{\rho}-\frac{z(z^\top m)}{d\rho^3}\right).
\]

When N||m|| dominates ||x||, define u=m/||m||. Then

\[
R(z)=\sqrt d\,g\odot\left[u+\frac{x-u(u^\top x)}{N\|m\|}+O(N^{-2})\right].
\]

Sensitivity to a fixed count increment can decay as N^-2. With zero epsilon and x parallel to m, positive radial changes are removed exactly. Positive epsilon generally leaves a mathematically nonzero radial signal that can become too small for finite precision.

This does not establish information loss from the entire decoder. The pre-norm block retains the bypass z+MLP(R(z)); unnormalized magnitude survives and later computation may convert it into direction. Actual queries, messages and native residuals vary with context, and useful information may already be directional. A nonzero mean negative-frame message could nevertheless dominate at fixed positive count under negative extensions. Zero-read centering does not rule this out.

The relevant ratio is ||delta||/||h+a||, together with alignment. Existing saved V5 ratios use ||a|| alone, the native attention output, and cannot establish branch dominance over the full residual stream.

One possible later falsifier is a fixed-context scale-sensitivity audit on predetermined short/long software examples and every dev-selected checkpoint. Capture full residual x and final-query branch delta, then apply multipliers 1/2,1,2 at that query while holding earlier states fixed. Compare complete-residual ratios/alignment, post-RMSNorm differences in float32 versus actual execution, and native answer-logit differences. Strong long-context sensitivity would weaken the simple indistinguishable-magnitude explanation. Tiny float32 changes suggest geometric compression; changes visible in float32 but absent in actual execution suggest finite-precision effects. Sensitive normalized features with insensitive logits point downstream.

This audit has not been registered or launched. It cannot justify an architecture change before evidence, and scale effects alone would not establish the source of counting errors.

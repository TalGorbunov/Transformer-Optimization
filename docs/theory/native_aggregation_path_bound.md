# What the native path penalty controls

This note explains the fixed V9 penalty. It supplies no new empirical result and changes no experiment. The inequalities are elementary sensitivity bounds, not a novelty claim.

Let the learned residual be

\[
F(r)=U\,\mathrm{SiLU}(r),\qquad
r=W_{\mathrm{agg}}z+q+b,
\]

where \(z\) is the sum of local messages. For paired scenes with identical question and causal output prefix, the frozen global state is identical, so their preactivation displacement is

\[
d=W_{\mathrm{agg}}(z_{\mathrm{right}}-z_{\mathrm{left}}).
\]

Computing this directly avoids subtracting a large shared query or bias in finite precision. Write \(u_j\) for column \(j\) of \(U\), and define

\[
B(d)=\sum_j \|u_j\|_2 |d_j|.
\]

## Uniform control along one direction

SiLU has global Lipschitz constant below 1.1. Consequently, for any base point \(r\) and real \(t\),

\[
\begin{aligned}
\|F(r+td)-F(r)\|_2
&\leq \sum_j \|u_j\|_2
  |\mathrm{SiLU}(r_j+td_j)-\mathrm{SiLU}(r_j)|\\
&\leq 1.1|t| B(d).
\end{aligned}
\]

For completeness, if \(x=2y\), then

\[
\mathrm{SiLU}'(x)=\tfrac12[1+\tanh(y)+y\,\mathrm{sech}^2(y)].
\]

Its positive maximum occurs at the unique positive solution of \(y\tanh(y)=1\), where the derivative equals \((1+y)/2\). This solution is below 1.2 because \(1.2\tanh(1.2)>1\); hence the maximum is below 1.1. The identity \(\mathrm{SiLU}'(-x)=1-\mathrm{SiLU}'(x)\) also controls the negative side.

Squaring and dividing by the fixed frozen-global norm plus epsilon gives the factor 1.21 used for the descriptive training comparison. This statement is about real arithmetic in the learned residual. Floating-point execution is tested separately.

## Exact zero has a strong meaning

Because every summand of \(B\) is nonnegative, \(B(d)=0\) exactly when each channel has either \(u_j=0\) or \(d_j=0\). This is sufficient for \(F(r+td)=F(r)\) for all \(r,t\).

It is also necessary if invariance is required at **every base point in the full preactivation space**. To see this, fix a nonzero \(t\), vary only coordinate \(r_j\), and keep all other coordinates fixed. When \(d_j\ne0\), the scalar difference

\[
\mathrm{SiLU}(r_j+td_j)-\mathrm{SiLU}(r_j)
\]

tends to zero as \(r_j\to-\infty\) and to \(td_j\) as \(r_j\to+\infty\). Its contribution cannot remain constant unless \(u_j=0\). Applying this argument to every coordinate gives \(B(d)=0\).

This characterization explains both the attraction and the conservatism of the penalty. Agreement at two observed residuals can arise from cancellation. Zero path penalty rules out such dependence along the measured direction even after the base point changes. But real model preactivations may occupy only a restricted manifold. Opposite output columns with identical activations can cancel perfectly on that manifold while receiving a positive penalty. Uniform off-manifold invariance can therefore constrain more than the task needs.

## Limits relevant to the experiment

The training loss averages \(B(d)^2/(\|g\|_2^2+10^{-6})\) over observed pairs and output positions. Small average loss does not imply a maximum bound for every pair. It does not cover unseen evidence-change directions, different visual primitives or arbitrary reasoning prefixes. In particular, equal-answer scene pairs need not define a valid invariance for every possible reasoning query.

The bound ends before native residual casting, final normalization and vocabulary projection. Predicting the same answer would additionally require controlling those operations and the relevant output margin. Exact invariance of the residual at a fixed global state would preserve the native output, but approximate training agreement is not such a certificate.

The penalty is unchanged by reciprocal rescaling of an output column and its displacement coordinate. This is a property of \(B\); SiLU generally prevents interpreting that rescaling as a symmetry of the entire network.

[Registered experiment and prior-art context](../paper/NATIVE_AGGREGATION_PATH_BOUND.md)

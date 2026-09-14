# V13 addendum: is the predictor still estimating a null mean?

Prospective analysis while main tests are pending; no test outcomes were inspected. This supplements [the interpretation memo](NATIVE_AGGREGATION_VISION_V13_INTERPRETATION.md). It proposes no implementation or additional experiment.

Fix the core temporarily. For question/prefix group \(\xi\), let \(\hat\mu_\xi\) be its current 24-occurrence message mean, \(c_\xi=c_\eta(q_\xi)\), and \(d_\xi\) the detached message second moment plus the registered floor. The auxiliary term is a weighted squared-error regression:

\[
 L_{\rm aux}=\sum_\xi \pi_\xi\,
 \frac{\|c_\xi-\hat\mu_\xi\|^2}{96d_\xi}.
\]

By itself, with fixed targets/queries and a realizable predictor, this is minimized at the empirical means. Let \(a_\xi\) denote the entire expected main-loss gradient with respect to \(c_\xi\), including CE, consistency, and their sampling weights. An unconstrained output per group would instead satisfy, at a joint stationary point,

\[
 a_\xi+\frac{2\lambda\pi_\xi}{96d_\xi}(c_\xi-\hat\mu_\xi)=0,
 \qquad
 c_\xi-\hat\mu_\xi=-\frac{96d_\xi}{2\lambda\pi_\xi}a_\xi.
\]

V13 uses \(\lambda=1\). Finite-weight regression does not keep the mean calibrated when the main loss wants a different offset. For the actual shared MLP, stationarity only requires the sum of these vectors projected through each predictor Jacobian to vanish. Exact calibration remains possible if the main gradient vanishes in predictor parameter space, including cancellation across groups; a nonzero gradient in that space rules it out. These are nominal expected-gradient conditions, not a convergence theorem for stochastic Adam and clipping. Core training also moves the detached regression targets, so the overall update is generally not the full gradient of one ordinary joint regression objective.

The reported training losses around step3500—centered auxiliary approximately .20–.36 versus offset .007–.02—are compatible with this tension. The centering Jacobian from predictor output to preactivation carries a factor \(N\); the offset control lacks that factor. Actual gradients also depend on their different learned states. These dimensionless losses use different learned message second moments across arms. They neither measure relative error in the mean itself nor establish the cause of any test result. Improving CE does not imply that \(c\) retains its intended estimand.

Two alternatives have different guarantees:

- **Train with the exact bank mean, then distill a fixed target.** Use \(\sum_i\phi_\theta(h_i,g)-N\hat\mu_\theta(q)\) during core training, differentiating through every reference message and its query. Both the forward statistic and its gradient cancel an arbitrary shared additive message offset. The existing frozen feature cache already supplies the required reference states; this changes small-core computation, not the number of backbone forwards. After freezing the core, fit the separate predictor to its fixed means, without main-loss gradients. This cleanly separates mean estimation from task optimization. It still centers an empirical bank, not every irrelevant image; approximation error, finite-bank uncertainty, prefix coverage, and the train/deployment substitution remain limitations. Comparing a two-phase method would require an explicit matched budget.
- **Detach the entire prediction in the main loss while the core moves.** Auxiliary regression then supplies the predictor's only update, with query and target detached. Detaching only its query is insufficient: main gradients would still alter predictor parameters. A fully detached prediction can track the intended mean, but omits the core-gradient term \(-N\,\partial\hat\mu_\theta/\partial\theta\). Even perfect instantaneous forward calibration does not restore that derivative. A query-only message shift, which exact centering should cancel, still generates an uncancelled main gradient; finite tracking lag introduces an N-scaled error. This is a moving-target estimator, not the exact-bank optimization in the first alternative.

A fixed-checkpoint bank oracle would resolve one narrower issue using existing artifacts. For every selected core, all 972 groups, and all ordinary cached training scenes with their original sequence/occurrence weights, substitute \(\hat\mu\) for \(c\) without fitting anything. The exact projected preactivation change is

\[
 t_{\rm bank}-t_{\rm predicted}=N W_{\rm agg}(c-\hat\mu).
\]

Recompute the actual nonlinear residual and, if separately authorized, cached native teacher-forced CE split by count and EOS positions. This measures whether fitting the bank better would help the current frozen core. A rescue supports predictor deviation as a correctable component; failure means that correcting this empirical mean alone is insufficient for that core. Neither establishes a population null estimator, free-running complete-answer generalization, or reasoning composition. The cleanest neutrality hypothesis is therefore exact empirical centering during task learning followed by fixed-core mean distillation—not a claim that a jointly task-optimized offset must remain a proper mean estimator.

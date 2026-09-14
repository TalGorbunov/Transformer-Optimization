# A constructive capacity check for the privileged joint-code readout

This is an algebraic observation, not a new experiment or an inference method. It applies only after exact per-image person/requested-room codes have already been supplied. It does not explain how a vision model learns those codes, how an optimizer reaches the constructed weights, or how a model generalizes.

Let s(x)=SiLU(x). Then

    f(x) = s(x) + s(-x) = x tanh(x/2),
    c = f(1) > 0.

For one person's two half-count coordinates a,b, define

    j(a,b) = [f(a+b) - f(a-b)] / c.

On the identity-join diagnostic's valid local-code domain, this is the requested conjunction:

| Half-count coordinates (a,b) | Meaning | j(a,b) |
|---|---|---:|
| (0,0) | Absent from requested rooms | 0 |
| (1,0) | Twice in the first room | 0 |
| (0,1) | Twice in the second room | 0 |
| (0.5,0.5) | Once in each requested room | 1 |

The equality follows because f is even, f(0)=0 and the nonzero sum/difference arguments in this domain have magnitude1. Other count patterns are outside this certificate; j is not asserted to be a Boolean conjunction on arbitrary nonnegative counts.

The existing96-wide SiLU readout can realize this mapping with four hidden coordinates per person: +(a+b), -(a+b), +(a-b), -(a-b), followed by output coefficients (+,+,-,-)/c. Nine persons use36 coordinates. Set the query matrix, aggregate bias and unused aggregate rows to zero. These are parameter values within the existing readout class; no extra layer is needed.

For a concrete connection to the earlier successful answer-code oracle, write V_p for its trained up-projection column for person p. That oracle supplied sqrt(96) times the person's one-hot code after the nonlinearity. Set the four new up-projection columns for p to (+,+,-,-) times sqrt(96) V_p/c. The resulting residual is

    delta = sqrt(96) sum_p j(a_p,b_p) V_p.

Every valid diagnostic scene has exactly one person in both requested rooms. Therefore this residual equals the earlier oracle residual algebraically, while the joint-code model's input still consists only of separate per-image codes. Prefixes can change the native global state g; the residual equality holds for the same g and person code at each position.

This gives a real-arithmetic capacity construction for the privileged readout and separates that question from learnability under a particular initialization/optimizer. It does not authorize initializing a proposed vision method with semantic labels or these constructed weights. The query being zero here is an existence proof, not a proposal to run another query ablation. No tensors, native head, optimizer or Slurm job were run for this note. Finite-precision implementation and native-logit equivalence of the constructed weights have not been tested.

Prior empirical reference: [answer-code oracle result](NATIVE_AGGREGATION_READOUT_DIAGNOSTIC_RESULTS.md). The raw local-code600-step failure remains unchanged: [joint-code results](NATIVE_AGGREGATION_JOINT_CODE_ORACLE_RESULTS.md).

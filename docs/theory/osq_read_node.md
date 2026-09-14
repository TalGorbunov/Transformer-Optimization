# Over-squashing at the read node — the load-bearing statements (draft, 2026-09-02)

What the paper claims formally, what each claim needs from the data, and which
pre-registered test (PREREG_OSQ.md) carries it. Everything here is either an identity, a
one-line inequality, or an explicitly empirical premise. No asymptotic hand-waving.

## Setting

One attention head at layer l; the answer row r ("read node") attends over S keys with
pre-softmax scores s_j = q_r . k_j / sqrt(d). Context columns C (N_T tokens grouped in F
sentences), evidence E subset of C (k_F tokens, F_E sentences), junk J = C \ E. Weights
renormalised over the context, a_j = e^{s_j} / sum_{C} e^{s_i}, so sum_C a_j = 1: the read
node is a message-passing aggregator over the complete graph on C with edge weights a.
Head-means are taken last.

## D1 — degree of the read node

  k_eff(tok) = 1 / sum_C a_j^2,     k_eff(sent) = 1 / sum_F (sum_{j in f} a_j)^2.

The participation ratio of the weights: equals the neighbour count for a uniform
aggregator, 1 for a one-hot read. This is the transformer instance of the GNN degree in
the over-squashing literature (Alon & Yahav 2021; Topping et al. 2022; Di Giovanni et al.
2023: the normalisation term of the Jacobian bound is the degree). "Over-squashing at the
read node" = k_eff grows with N while the evidence's share of weight (D2) and of sensitivity
(D3) shrinks.  Tests: P3 (full arm: k_eff(sent)(128k)/k_eff(sent)(4k) >= 4; repack arm: flat).

## D2 / Prop. 1 — the dispersion identity and the mean-margin law

  mass = sum_E a_j = sum_E e^{s} / (sum_E e^{s} + sum_J e^{s}).                     (identity)

Write the margin g = mean_E s - mean_J s and the multiplicities
nu_E = (1/k_F) sum_E e^{s - mean_E s} >= 1,  nu_J = (1/|J|) sum_J e^{s - mean_J s} >= 1
(Jensen; equality iff the scores in the set are equal). Then, exactly,

  mass = k_F nu_E e^{g} / (k_F nu_E e^{g} + |J| nu_J),                                (Prop. 1)

and the pre-registered no-free-parameter law `pred_mass` is the nu_E = nu_J = 1 case.
The probe also records m_eff = logit(mass) - log(k_F/|J|), the margin that WOULD produce
the measured mass under uniform scores; hence

  g - m_eff = log(nu_J / nu_E)                                                       (Cor. 1)

is a direct measurement of how heavy-tailed the junk competition is: 0 nats = uniform
dilution, every junk token competes equally; large = a few junk tokens (in-context sinks,
sentence-initial tokens) carry the competition. First landed cells (Qwen-3B, 4k-16k):
g - m_eff = 3.9..5.5 nats, i.e. nu_J/nu_E ~ 50-250. **Dilution is not uniform; the
competition is a heavy tail, which is exactly what k_eff(sent) counts.**

Corollary 2 (the log-N requirement, Add. 11 made exact): holding mass fixed while |J| grows
by a factor c needs g to grow by log c + log(nu_J'/nu_J). A frozen model's g is fixed by its
weights, not by N (P2's margin premise: g(128k) - g(4k) < 1 while log 32 = 3.5); so mass falls
and k_eff rises with N — over-squashing is structural, not a calibration accident. Length
finetuning moves g by a constant (Add. 11-12: +0.6 against a +2.1 requirement); selection
makes |J| N-independent so the requirement stops growing (Add. 13b: AS3 fan-in flat).
Tests: P2 (Spearman(pred_mass, mass), |pred - mass| share, margin premise), P3.

## D3 / Prop. 2 — the sensitivity chain

Influence of context token j on the gold logit: I_j = || d logit_gold / d h_l(j) ||;
infl = sum_E I_j / sum_C I_j; k_eff(jac) its participation ratio over sentences.
Through one attention read, d(sum_i a_i v_i) / d h_l(j) has a value path a_j W_V and a
score path a_j (v_j - sum_i a_i v_i) q^T W_K / sqrt(d): both proportional to a_j, so I_j
scales with a_j up to value norms; the GNN sensitivity bound (Topping et al. Thm 1 / Di Giovanni
et al. Thm 3.2) is the same statement with a in place of the normalised adjacency. We do
not assume the chain holds through 8-16 further layers; we test it. Tests: P4 (Spearman
(infl, mass) > .7 per model; logistic acc ~ infl, p < .001; necessity: succeeding cells have
high mass — AUC > .85 — because bounded degree is necessary for the read, not sufficient).

## Prop. 3 — bounded competition: why the selector is a tournament

Selection = choose k sentences from F by the read row's weights. A single softmax over
N_T tokens IS the read node whose over-squashing we diagnose, so a one-forward selector
inherits the same dilution (Llama 128k one-forward recall .61 vs .95 at 32k). Chunking
(score each window separately, concatenate the scores) compares scores from different
softmaxes — different normalisers, different competitor sets — which the identity above
says is not a comparison at all. The tournament keeps every comparison inside one softmax
over at most W tokens: per-window top-k_local, re-score the union in one window, recurse
until it fits. Under window-consistency (the ordering of two sentences' row scores does not
depend on which other sentences share their window), the tournament returns the exact
global top-k with every softmax bounded by W. Window-consistency is the empirical premise
(RoPE shifts and the tail state's dependence on its window both violate it slightly); its
failure mode is a recall drop, which is what P1 measures against the chunked baseline.
Tests: P1 (recall vs chunked and one-forward), T8K (W = 8k: the premise at a tighter budget).

## Prop. 4 — repack is degree control (nothing else)

After selection the read row sees k sentences at contiguous positions: |J| is bounded
(k - F_E sentences), so by Prop. 1 the mass no longer falls with N and k_eff(sent) <= k.
Varying k inside the prompt reproduces the dispersion law inside the prompt: mass falls,
k_eff rises, accuracy falls once recall has saturated, and at matched token count the
repacked read should behave like the full read of that length. Tests: P7 (k-sweep),
P3 (attn k_eff flat in N).

## The second axis — positions (not over-squashing, and we say so)

The same k sentences at their ORIGINAL positions (attn_norp) have the same degree and the
same competitors; only the relative distances differ. Any accuracy gap between attn and
attn_norp is therefore not over-squashing but relative-position OOD (edge-length OOD in
graph terms). P5 isolates it with the oracle set and a synthetic gap P (dose-response,
metrics of the read held near their P=0 values while accuracy falls). The paper's two-axis
claim is exactly this factorisation: selection cures the degree axis, repack the position
axis, and the metrics are silent on the position axis by construction (P3's norp check).

## Radius (Alon & Yahav) — untested

qa2/qa3 have problem radius 2/3 in the sentence graph. Whether a second selection round
reads the second hop was to be P6; the copy-row mechanism carried no hop-2 signal at any
layer (PREREG_OSQ.md, Outcomes), so the paper reports the recall ordering qa1 > qa2 > qa3
as observational only and leaves multi-round selection as future work.

## What would make the framing cosmetic (and how each is excluded)

1. The metrics could move without predicting anything -> P4 necessity (mass separates
   succeeding from failing cells) and P2's law (mass is predicted, not just measured).
2. The method could work for reasons unrelated to degree -> P7 (k-identity: the same
   dispersion inside the prompt) and P3 (norp: same degree, different accuracy = the part
   the theory does NOT claim, isolated).
3. The selector could be an ordinary retriever -> P1 (the tournament, a design forced by
   Prop. 3, must beat chunked scoring; if it does not, we say the dilution story of
   selection failure is wrong).

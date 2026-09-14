# Training-question conditioning diagnostic

Source preparation is underway. No fit has run. The preceding oracle readout passed108/108 firsttokens; this new experiment tests whether global versus question mean subtraction with one shared scale helps the existing uniform encoder learn the same training joins.

[Protocol](../../../docs/paper/NATIVE_AGGREGATION_CONDITIONING_PROPOSAL.md) · [Preceding readout and geometry](../../../docs/paper/NATIVE_AGGREGATION_READOUT_DIAGNOSTIC_RESULTS.md)

The statistics use the original108 training contexts only. The global arm has no deployment question lookup; the question-indexed arm is a richer diagnostic. The fixed600-step full-sequence CE fit first receives a cached first-token screen; passing releases preparation of actual native evaluation of that same checkpoint, without further fitting.

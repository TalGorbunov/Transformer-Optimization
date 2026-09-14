# Product query-placement diagnostic

Independent computation and provenance audits passed. The scored outputs are cached training first tokens.

| Product placement | Correct /108 | Complete families /18 | Fixed paired screen |
|---|---:|---:|---|
| Frozen parent: query in factors and readout | 79 | 5 | FAIL |
| New: query in readout only | 80 | 5 | FAIL |

The new product after the fixed cyclic factor permutation scored 11/108 first tokens and 0/18 complete families. The permutation is descriptive; it cannot change the paired screen.

Both runs started from the identical unfitted seed 24 parameter packet and used the same 600 full-name-plus-EOS CE updates, pair order and global conditioning statistics. The initial native output is identical because U is zero. Internal factors, aggregates and the first U gradient can differ. Only the new run was fitted or replayed in this allocation; the completed parent is joined through its passed report, saved predictions and raw file hashes.

Local hidden states still depend on the question and prefix. The explicit global query is removed from both factors and retained after the sum. This one-seed training diagnostic does not assign semantic roles to the factors or establish wider communication capacity, whole-answer accuracy, generalization or reasoning composition.

The fixed screen failed. This architecture branch ends without native evaluation, continuation or another variant.

Allocated GPU cost: 28 seconds. GPU work: 614 norm/head batches, 21,550 rows, zero VLM/vision calls. Independent CPU replay: 14 final batches, 216 rows, fixed TV <= 0.02 and exact argmax.

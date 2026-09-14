# Ordinary image-prefix reuse software result

CPU443934 passed18job-seconds; GPU443936 passed39GPU-seconds. All eight ordinaryfull versus image-prefixreuse generated sequences matched. All eight reverse-question-order repeats were bitexact. Full versus split logits were not bitexact (0/8); maximum total variation0.018129 against the prospectively fixed0.02 threshold. This finite eight-case check does not establish universal prediction equivalence.

The actual run used60model/head calls,62decoder/norm calls and10vision calls, including two headless image-prefills. Exact immutable28layerKV snapshots and independent question branches passed; persistent KV was91,635,712bytes atN8 and182,468,608bytes atN16. No adapter, compressor, fitting, or efficacy evaluation ran.

# outputs/mmred_frozen — training-free select-and-repack on MMRED text (filtered counting)

Run on the **Nebius cluster** (peer session), 2026-09-02, jobs mmred_q7 11931 / mmred_q3 11932 /
mmred_ll 11933 (gpu-prod); wrapper `slurm/babilong.sbatch` (MMRED branch), `eval_babilong.py --mmred
data/mmred_filtered/seq_len_<N>/test --k 16 --sel-layer {22,27,20} --max-new 4`, arms
full+oracle+attn+attn_norp+lex+emb+rand, n = 200 (180 at N=512, 108 at N=1024 = dataset size).
Run dirs live on Nebius: `outputs/mmred_frozen/<model>/mmred_filtered_seq_len_<N>/<ts>/{config.json,
results.jsonl,summary.json}`; the numbers below were pasted from the summary.json files by the peer
session (transcribed 2026-09-02; not yet copied to this cluster). Recall: identical for
full/oracle/lex/emb (= 1 − zero-fact share), identical for attn/attn_norp, separate for rand.
Not pre-registered; a completeness cell for the training-free variant (see CONDMASK_REPORT.md).

| model (layer) | N | full | oracle | attn | attn_norp | lex | emb | rand | recall full / attn / rand |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B (L22) | 8 | .590 | .915 | .590 | .590 | .590 | .590 | .590 | .895 / .895 / .895 |
| | 16 | .390 | .815 | .390 | .390 | .390 | .390 | .390 | .905 / .905 / .905 |
| | 32 | .345 | .830 | .360 | .365 | .345 | .345 | .200 | .905 / .879 / .490 |
| | 64 | .310 | .735 | .345 | .370 | .285 | .245 | .140 | .905 / .824 / .213 |
| | 128 | .240 | .785 | .320 | .305 | .315 | .315 | .110 | .905 / .794 / .119 |
| | 256 | .145 | .765 | .295 | .340 | .370 | .295 | .125 | .890 / .740 / .062 |
| | 512 | .161 | .794 | .256 | .306 | .544 | .383 | .111 | .889 / .710 / .017 |
| | 1024 | .148 | .796 | .324 | .231 | .519 | .333 | .120 | .889 / .720 / .023 |
| Qwen2.5-3B (L27) | 8 | .410 | .890 | .410 | .410 | .410 | .410 | .410 | .895 / .895 / .895 |
| | 16 | .305 | .875 | .305 | .305 | .305 | .305 | .305 | .905 / .905 / .905 |
| | 32 | .265 | .800 | .250 | .340 | .200 | .180 | .225 | .905 / .859 / .490 |
| | 64 | .255 | .855 | .280 | .265 | .130 | .145 | .155 | .905 / .765 / .213 |
| | 128 | .190 | .840 | .280 | .275 | .140 | .150 | .110 | .905 / .736 / .119 |
| | 256 | .150 | .895 | .230 | .255 | .240 | .195 | .135 | .890 / .708 / .062 |
| | 512 | .111 | .844 | .306 | .183 | .328 | .189 | .100 | .889 / .671 / .017 |
| | 1024 | .130 | .907 | .269 | .204 | .435 | .250 | .130 | .889 / .568 / .023 |
| Llama-3.1-8B (L20) | 8 | .655 | .755 | .655 | .655 | .655 | .655 | .655 | .895 / .895 / .895 |
| | 16 | .450 | .740 | .450 | .450 | .450 | .450 | .450 | .905 / .905 / .905 |
| | 32 | .250 | .685 | .300 | .390 | .310 | .265 | .255 | .905 / .784 / .490 |
| | 64 | .235 | .675 | .335 | .295 | .155 | .170 | .150 | .905 / .661 / .213 |
| | 128 | .270 | .625 | .270 | .290 | .245 | .185 | .120 | .905 / .612 / .119 |
| | 256 | .345 | .715 | .290 | .265 | .430 | .235 | .125 | .890 / .595 / .062 |
| | 512 | .244 | .667 | .294 | .217 | .550 | .283 | .111 | .889 / .484 / .017 |
| | 1024 | .222 | .648 | .296 | .204 | .620 | .287 | .120 | .889 / .505 / .023 |

Reading (2026-09-02): k = 16 ≥ N for N ≤ 16, so every selection arm equals `full` there. Beyond
that the attention selector's recall falls from .88 to .5–.7 and its repack gains +10–18pp over
`full` at N ≥ 256 (Q7 .145 → .295, .148 → .324) but stays 45–50pp below the oracle; a plain
LEXICAL filter (keep sentences naming the asked character or room) keeps every gold frame and
reaches .52–.62 at N ≥ 512 (Llama .620 vs oracle .648) — on this task the read-node attention is
a worse selector than the words. The frozen models' own counting caps at the oracle (.65–.92).
Companion vision cells: `outputs/mmred_vfrozen/` (local; attn ≈ full at every N).

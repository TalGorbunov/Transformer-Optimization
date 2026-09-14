#!/bin/bash
# The over-squashing confirm grids of PREREG_OSQ.md (2026-09-02). One sbatch per line;
# all through slurm/babilong.sbatch. Run from the repo root:  bash slurm/osq_grids.sh [G1|YARN|T8K|ROUNDS|GAP|KSWEEP|LSE|STAGE|SCAN1M|M1M|ALL]
set -euo pipefail
Q3="MODEL=Qwen/Qwen2.5-3B-Instruct,LAYER=27"
Q7="MODEL=Qwen/Qwen2.5-7B-Instruct,LAYER=22"
LL="MODEL=meta-llama/Llama-3.1-8B-Instruct,LAYER=20"
Q1M="MODEL=Qwen/Qwen2.5-7B-Instruct-1M,LAYER=${LAYER_1M:-22}"   # LAYER_1M from the SCAN1M result
ALLT="qa1+qa2+qa3+qa4+qa5"
TOURN="--sel-chunk 32768 --sel-tournament 64"
sub() { sbatch --parsable -J "$1" -t "$2" --export="ALL,$3" slurm/babilong.sbatch; }
what="${1:-ALL}"

# G1 -- probe grid (P1-P4): 3 models x 5 tasks, all lengths, tournament selector, Jacobian on 60/cell
if [[ $what == G1 || $what == ALL ]]; then
  for M in "$Q3" "$Q7" "$LL"; do for T in qa1 qa2 qa3 qa4 qa5; do
    sub osq_g1 12:00:00 "$M,TASKS=$T,LENGTHS=4k+8k+16k+32k+64k+128k,LIMIT=200,K=64,ARMS=full+oracle+attn+attn_norp,OUTPUT=outputs/babilong_osq,EXTRA=--probe $TOURN --jac-limit 60"
  done; done
fi
# YARN -- the factorial's fourth cell (P3): Qwen under YaRN x4 at 64k/128k, probed
if [[ $what == YARN || $what == ALL ]]; then
  for M in "$Q3" "$Q7"; do
    sub osq_yarn 12:00:00 "$M,TASKS=$ALLT,LENGTHS=64k+128k,LIMIT=100,K=64,ARMS=full+oracle+attn+attn_norp,OUTPUT=outputs/babilong_osq_yarn4,EXTRA=--yarn 4 --probe $TOURN --jac-limit 60"
  done
fi
# T8K -- Llama tournament with 8k windows (sharper local reads; P1 variant)
if [[ $what == T8K || $what == ALL ]]; then
  sub osq_t8k 08:00:00 "$LL,TASKS=$ALLT,LENGTHS=64k+128k,LIMIT=100,K=64,ARMS=attn+attn_norp,OUTPUT=outputs/babilong_tourn8k,EXTRA=--sel-chunk 8192 --sel-tournament 64"
fi
# ROUNDS -- rounds = radius (P6): two-round selection on qa1-3
if [[ $what == ROUNDS || $what == ALL ]]; then
  for M in "$Q7" "$LL"; do
    sub osq_rounds 08:00:00 "$M,TASKS=qa1+qa2+qa3,LENGTHS=64k+128k,LIMIT=100,K=64,ARMS=attn+attn_norp,OUTPUT=outputs/babilong_rounds,EXTRA=--sel-rounds 2 $TOURN"
  done
fi
# GAP -- position dose-response (P5): 0k split, degree fixed, edge length P
if [[ $what == GAP || $what == ALL ]]; then
  P="0+2000+4000+8000+16000+24000+32000+40000+48000+64000+96000+120000"
  for M in "$Q3" "$Q7" "$LL"; do
    sub osq_gap 08:00:00 "$M,TASKS=$ALLT,LENGTHS=0k,LIMIT=200,K=64,ARMS=oracle_gap+oracle_pgap,OUTPUT=outputs/babilong_gap,EXTRA=--gap-tokens $P --probe --jac-limit 60"
  done
  for M in "$Q3" "$Q7"; do
    sub osq_gapyarn 08:00:00 "$M,TASKS=$ALLT,LENGTHS=0k,LIMIT=200,K=64,ARMS=oracle_gap+oracle_pgap,OUTPUT=outputs/babilong_gap_yarn4,EXTRA=--yarn 4 --gap-tokens $P --probe --jac-limit 60"
  done
fi
# LSE -- gap split (Gollapudi Tab. 9 on our models): raw lse_e/lse_j/smax_e per arm, attention probe only
if [[ $what == LSE || $what == ALL ]]; then
  for M in "$Q3" "$Q7"; do
    sub osq_lse 08:00:00 "$M,TASKS=qa1+qa2,LENGTHS=4k+8k+16k+32k+64k+128k,LIMIT=100,K=64,ARMS=full+attn+attn_norp,OUTPUT=outputs/babilong_lse,EXTRA=--probe --jac-limit 0 $TOURN"
  done
  sub osq_lse 08:00:00 "$LL,TASKS=qa1+qa2+qa5,LENGTHS=4k+8k+16k+32k+64k+128k,LIMIT=100,K=64,ARMS=full+attn+attn_norp,OUTPUT=outputs/babilong_lse,EXTRA=--probe --jac-limit 0 $TOURN"
fi
# STAGE -- where the tournament loses facts: per-stage recall + within-window rank (Llama)
if [[ $what == STAGE || $what == ALL ]]; then
  sub osq_stage 04:00:00 "$LL,TASKS=qa1+qa2+qa3+qa5,LENGTHS=64k+128k,LIMIT=100,K=64,ARMS=attn,OUTPUT=outputs/babilong_stage,EXTRA=$TOURN"
fi
# SCAN1M / M1M -- horizon test: a 1M-context Qwen; scan the selection layer at 4k first, then the factorial
if [[ $what == SCAN1M ]]; then
  sbatch --parsable -J osq_scan1m -p gpu-short -t 00:40:00 --export="ALL,$Q1M,TASKS=qa1,LENGTHS=4k,LIMIT=100,K=64,OUTPUT=outputs/babilong_1m,EXTRA=--scan" slurm/babilong.sbatch
fi
if [[ $what == M1M ]]; then
  sub osq_1m 10:00:00 "$Q1M,TASKS=qa1+qa2,LENGTHS=4k+32k+64k+128k,LIMIT=100,K=64,ARMS=full+oracle+attn+attn_norp,OUTPUT=outputs/babilong_1m,EXTRA=--probe --jac-limit 0 $TOURN"
fi
# KSWEEP -- k-identity (P7): budgets 16..2048 at 128k, chunked selector, attention-only probe
if [[ $what == KSWEEP || $what == ALL ]]; then
  for M in "$Q7" "$LL"; do
    sub osq_ksweep 12:00:00 "$M,TASKS=qa1+qa2+qa3,LENGTHS=128k,LIMIT=100,K=64,ARMS=attn,OUTPUT=outputs/babilong_ksweep,EXTRA=--k-sweep 16+32+128+256+512+1024+2048 --sel-chunk 32768 --probe --jac-limit 0"
  done
fi

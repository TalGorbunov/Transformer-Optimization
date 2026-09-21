#!/bin/bash
#SBATCH --job-name=sg_smoke
#SBATCH -p l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH -o logs/%x-%j.out
# SOFTGATE + REDUX v3 pre-flight (2026-09-20). Fails loudly on any anchor miss.
source "${SLURM_SUBMIT_DIR:-.}/slurm/lib/common.sh"
set -e
S=outputs/_scratch/softgate
P1B=checkpoints/sft_fenced_le16_ep10_adapter
GATED=checkpoints/sft_fenced_gated_vn_nfree_adapter
EXN8=outputs/loramech/examdirs/exam_ff_N8.txt
POOL8=data/mmred_images_park/seq_len_8/all_uniform
# (1) probe anchor: no flags, 3 pairs, must byte-match the L40S reference
run_logged $S/probe_anchor python -u scripts/sparse/probe_hahn_gated.py --data_root $POOL8 --limit 3 --controls 1 --seed 0 --arms p1fence --peft-adapter $P1B --output $S/probe_anchor
diff -q $S/probe_anchor/pairs.csv outputs/_scratch/sparse_smoke/probe_anchor/pairs.csv && echo "[anchor] PROBE BYTE-IDENTICAL" || { echo "[anchor] PROBE DIFFERS"; exit 1; }
# (2) probe with the penalty on (B=8) and with B=1 (must equal gate=none masks): 3 pairs each
run_logged $S/probe_bonus8 python -u scripts/sparse/probe_hahn_gated.py --data_root $POOL8 --limit 3 --controls 1 --seed 0 --arms p1fence --peft-adapter $GATED --gate oracle --nfree-prompt --gate-bonus 8 --output $S/probe_bonus8
run_logged $S/probe_bonus1 python -u scripts/sparse/probe_hahn_gated.py --data_root $POOL8 --limit 3 --controls 1 --seed 0 --arms p1fence --peft-adapter $GATED --gate oracle --nfree-prompt --gate-bonus 1 --output $S/probe_bonus1
run_logged $S/probe_nogate python -u scripts/sparse/probe_hahn_gated.py --data_root $POOL8 --limit 3 --controls 1 --seed 0 --arms p1fence --peft-adapter $GATED --nfree-prompt --output $S/probe_nogate
python - <<'PY'
import csv
def L(p): return {(r['sample_id'],r['flip_t'],r['layer'],r['locus']):float(r['dnorm']) for r in csv.DictReader(open(p))}
a=L("outputs/_scratch/softgate/probe_bonus1/pairs.csv"); b=L("outputs/_scratch/softgate/probe_nogate/pairs.csv")
common=[k for k in a if k in b]; d=max(abs(a[k]-b[k]) for k in common)
print(f"[anchor] B=1 vs no-gate: {len(common)} rows, max|diff|={d:.3e}", "PASS" if d<1e-3 else "FAIL")
c=L("outputs/_scratch/softgate/probe_bonus8/pairs.csv"); d8=max(abs(a[k]-c[k]) for k in common if k in c)
print(f"[smoke] B=8 vs B=1 differ by max {d8:.3f} (must be >0)")
PY
# (3) trainer anchor: eval-only P1b on exam_ff_N8@150 must read 1.0000
run_logged $S/anchor_p1b_N8 python -u scripts/sparse/train_sft_gated.py --limit 24 --eval-only-adapter $P1B --eval-dirs-file $EXN8 --eval-longn-limit 150 --output $S/anchor_p1b_N8
grep -q "acc=1.0000" $S/anchor_p1b_N8/*/run.log 2>/dev/null || grep -q "1.0000" $S/anchor_p1b_N8/*/longn_eval.csv && echo "[anchor] TRAINER P1b exam_ff_N8 = 1.0000" || { echo "[anchor] TRAINER ANCHOR MISS"; cat $S/anchor_p1b_N8/*/longn_eval.csv; exit 1; }
# (4) eval smoke: gated adapter under a soft gate (B=8), 6 dirs
run_logged $S/smoke_eval_bonus8 python -u scripts/sparse/train_sft_gated.py --limit 24 --eval-only-adapter $GATED --gate oracle --gate-bonus 8 --nfree-prompt --eval-dirs-file $EXN8 --eval-longn-limit 6 --output $S/smoke_eval_bonus8
# (5) train smokes, 1 ep, 16 samples: soft gate B=8; multitask ungated; multitask gated
run_logged $S/smoke_train_bonus8 python -u scripts/sparse/train_sft_gated.py --limit 16 --epochs 1 --val-cap 4 --gate oracle --gate-bonus 8 --nfree-prompt --eval-dirs-file $EXN8 --eval-longn-limit 4 --output $S/smoke_train_bonus8
run_logged $S/smoke_train_mt_ungated python -u scripts/sparse/train_sft_gated.py --limit 18 --epochs 1 --val-cap 4 --tasks count,exists,majority --nfree-prompt --eval-dirs-file outputs/redux/examdirs/exam_exists_N8.txt --eval-dirs-file outputs/redux/examdirs/exam_majority_N8.txt --eval-longn-limit 4 --output $S/smoke_train_mt_ungated
run_logged $S/smoke_train_mt_gated python -u scripts/sparse/train_sft_gated.py --limit 18 --epochs 1 --val-cap 4 --tasks count,exists,majority --gate oracle --nfree-prompt --eval-dirs-file outputs/redux/examdirs/exam_exists_N8.txt --eval-dirs-file outputs/redux/examdirs/exam_majority_N8.txt --eval-longn-limit 4 --output $S/smoke_train_mt_gated
echo "SOFTGATE SMOKE COMPLETE"

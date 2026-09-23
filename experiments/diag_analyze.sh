#!/usr/bin/env bash
# DIAG collection: run every CPU fit over whatever run dirs exist under outputs/diag/, then every figure that has inputs.
# Idempotent; safe on the login node (seconds to a few minutes; block_mass.csv reads dominate).
#   bash experiments/diag_analyze.sh            # fits + figures
#   bash experiments/diag_analyze.sh fits       # fits only
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
F=outputs/diag/fits; mkdir -p "$F"
py() { python -u experiments/diag_fit.py "$@"; }
echo "== alpha (D1 flip ladders, all conds)"
py alpha    --runs 'outputs/diag/hahn/*/N*/2*' --out $F/alpha.json || true
py margins  --runs 'outputs/diag/hahn/*/N*/2*' --out $F/margins.json || true
echo "== sharelaw (D2 photographs)"
py sharelaw --runs 'outputs/diag/photo/multi/*/N*/2*' --out $F/sharelaw.json || true
py headscan --runs 'outputs/diag/photo/multi/*/N*/2*' --out $F/headscan.json --head L24_h20 || true
echo "== D3 tau / log-N photographs"
py sharelaw --runs 'outputs/diag/photo/d3/*/N*/2*' --out $F/sharelaw_d3.json || true
py headscan --runs 'outputs/diag/photo/d3/*/N*/2*' --out $F/headscan_d3.json --head L24_h20 || true
echo "== mechform"
[ -f $F/alpha.json ] && [ -f $F/sharelaw.json ] && py mechform --alpha $F/alpha.json --sharelaw $F/sharelaw.json --k 4 --out $F/mechform.json || true
echo "== D4 gate fits"
if ls outputs/diag/gate/N8_train/*/2* >/dev/null 2>&1; then
  python -u experiments/gate_fit.py --runs 'outputs/diag/gate/N8_train/*/2*' 'outputs/diag/gate/N16_train/*/2*' --per-qtype --out $F/gate_d4_train.json || true
  if ls outputs/diag/gate/N32_test/*/2* >/dev/null 2>&1; then
    for ARM in fenced_qlast fenced_qfirst; do
      python -u experiments/gate_fit.py --train-runs "outputs/diag/gate/N8_train/$ARM/2*" "outputs/diag/gate/N16_train/$ARM/2*" \
        --test-runs "outputs/diag/gate/N*_test/$ARM/2*" --out $F/gate_d4_transfer_$ARM.json || true
    done
  fi
fi
echo "== D3 eval summary -> outputs/diag/eval_summary.csv"
python - <<'PY'
import csv, glob, json, os
rows=[]
for d in sorted(glob.glob('outputs/diag/eval/*/*/N*/2*')+glob.glob('outputs/diag/eval/grid/N*/2*')+glob.glob('outputs/port/evaluate/seq_len_8_test/*/2*_faithful')):
    s=os.path.join(d,'summary.csv'); c=os.path.join(d,'config.json')
    if '/tier1/' in d: continue                 # Tier-1 adapter chains -> experiments/diag_table.py
    if not (os.path.exists(s) and os.path.exists(c)): continue
    cfg=json.load(open(c)); N=cfg.get('config','').replace('seq_len_','')
    arm='plain' if cfg.get('layout')=='paper' else ('gated' if cfg.get('gate')=='oracle' else ('fenced' if cfg.get('fence') else 'qfirst'))
    for r in csv.DictReader(open(s)):            # summary.csv: group,n,correct,acc,ci_lo,ci_hi
        g=r.get('group','')
        if g.startswith('atype_') or g.startswith('numeric'): continue
        rows.append({'qtype':g,'arm':arm,'cond':cfg.get('cond','base'),'N':N,'acc':r.get('acc',''),'n':r.get('n',''),'ci_lo':r.get('ci_lo',''),'ci_hi':r.get('ci_hi',''),'run':(d if 'outputs/port' not in d else d.replace('outputs/port/evaluate','outputs/diag/eval/grid'))})
if rows:
    with open('outputs/diag/eval_summary.csv','w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f'{len(rows)} eval rows')
else:
    print('no eval summaries yet')
PY
echo "== Tier-1 comparison table -> outputs/diag/tier1_table.{csv,md}"
python -u experiments/diag_table.py || true
[ "${1:-all}" = "fits" ] && exit 0
echo "== figures"
python -u experiments/figs/diag_fig.py all-available --fits $F --out outputs/diag/fig || true

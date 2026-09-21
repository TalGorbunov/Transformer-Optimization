#!/usr/bin/env python3
"""S10b figures (2026-09-19). F10: per-frame attention mass (weights) beside the flip response (Hahn)
beside the selector head; F11: the temperature coupling (competitor mass vs within-evidence
concentration vs S0 count accuracy). Reads outputs/sparse/s10/*/block_mass.csv, F6_s9_alpha.csv,
ARMOR fenced medians (hard-coded from outputs/armor/hahn/20260822_211457_fig/medians.csv), S0 longn_eval.
Writes outputs/sparse/fig/F10_dispersion_hahn.{png,csv} and F11_temperature_coupling.{png,csv}."""
import csv, glob, os, re, statistics as st
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = "outputs/sparse/s10"; FIG = "outputs/sparse/fig"; os.makedirs(FIG, exist_ok=True)
C = {'frozen': '#a4a7ae', 'p1b': '#eb6834', 'gated': '#2a78d6', 'fact': '#1baf7a', 'ink': '#15171c',
     'ink2': '#4a4e57', 'muted': '#7a7e88', 'grid': '#e2e3e6'}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#c9cbd1', 'axes.labelcolor': C['ink2'], 'xtick.color': C['muted'],
                     'ytick.color': C['muted']})

def perframe(d, layer, k=4, head=None):
    """median over (sample, head) of the mean per-block mass, evidence and non-evidence, plus
    within-evidence concentration (max evidence block / mean evidence block) and 'other' mass."""
    per = defaultdict(lambda: {'e': [], 'n': []})
    with open(f"{d}/block_mass.csv") as f:
        for r in csv.DictReader(f):
            if int(r['layer']) != layer or int(r['k']) != k: continue
            if head is not None and int(r['head']) != head: continue
            per[(r['sample'], int(r['head']))]['e' if r['is_evid'] == '1' else 'n'].append(float(r['mass']))
    ev = [st.mean(v['e']) for v in per.values() if v['e']]
    ne = [st.mean(v['n']) for v in per.values() if v['n']]
    conc = [max(v['e']) / st.mean(v['e']) for v in per.values() if v['e'] and st.mean(v['e']) > 0]
    other = [1.0 - sum(v['e']) - sum(v['n']) for v in per.values() if v['e']]
    med = lambda x: (st.median(x) if x else float('nan'))
    return med(ev), med(ne), med(conc), med(other)

# ---------------- gather arms × N ----------------
rows = []
for d in sorted(glob.glob(f"{BASE}/*_N*")):
    m = re.match(r".*/(p1b|frozen|gated)_N(\d+)(?:_tau([\d.]+))?$", d)
    if not m or not os.path.exists(f"{d}/block_mass.csv"): continue
    arm, N, tau = m.group(1), int(m.group(2)), (float(m.group(3)) if m.group(3) else 1.0)
    kk = 1 if N == 2 else 2                      # panel k: 2 everywhere, 1 at N=2 (k=2 leaves no competitor)
    e, n, c, o = perframe(d, 20, k=kk); e24, n24, _, _ = perframe(d, 24, k=kk, head=20)
    e4, n4, c4, o4 = perframe(d, 20, k=4) if N >= 8 else (float('nan'),) * 4
    rows.append(dict(arm=arm, N=N, tau=tau, k=kk, evid_L20=e, nonevid_L20=n, conc_L20=c, other_L20=o,
                     evid_L24h20=e24, nonevid_L24h20=n24, evid_L20_k4=e4, nonevid_L20_k4=n4, conc_L20_k4=c4, other_L20_k4=o4))
with open(f"{FIG}/F10_dispersion_hahn.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
pm = {(r['arm'], r['N']): r for r in rows if r['tau'] == 1.0}
def series(arm, key):
    Ns = sorted(N for (a, N) in pm if a == arm); return Ns, [pm[(arm, N)][key] for N in Ns]

# ---------------- F10 ----------------
f6 = {int(r['N']): {k: float(v) for k, v in r.items()} for r in csv.DictReader(open(f"{FIG}/F6_s9_alpha.csv"))}
fenced = {8: 38.612, 16: 38.839, 32: 39.177, 64: 38.764, 128: 39.535}
def pairs_med(path, arm, locus, layer=20):
    if not os.path.exists(path): return None
    v = [float(r['dnorm']) for r in csv.DictReader(open(path)) if r['arm'] == arm and r['layer'] == str(layer)
         and r['locus'] == locus and r['flip_kind'] == 'evid']
    return st.median(v) if len(v) >= 20 else None
for N in (2, 4):  # small-N fill (2026-09-19): armor short chain, N2-protocol P1b read, S11 gated read
    a = pairs_med(f"outputs/armor/hahn/short_N{N}/pairs.csv", 'plain', 'final')
    t = pairs_med(f"outputs/loramech/n2_hahn/p1fence_ep10_N{N}/pairs.csv", 'p1fence', 'final')
    g = pairs_med(f"outputs/sparse/s11/gate_oracle_N{N}/pairs.csv", 'p1fence', 'final')
    fz = pairs_med(f"outputs/armor/hahn/short_N{N}/pairs.csv", 'fenced', 'rep_t')
    if a is not None or t is not None or g is not None:
        f6[N] = {'armor_med': a, 'n2_med': t, 's11_med': g}
        if fz is not None: fenced[N] = fz
def strat_med(path, arm, locus, gold, layer=20):
    if not os.path.exists(path): return None
    v = [float(r['dnorm']) for r in csv.DictReader(open(path)) if r['arm'] == arm and r['layer'] == str(layer)
         and r['locus'] == locus and r['flip_kind'] == 'evid' and int(r['gold']) == gold]
    return st.median(v) if len(v) >= 5 else None
SRC = {'frozen read': ({2: "outputs/armor/hahn/short_N2/pairs.csv", 4: "outputs/armor/hahn/short_N4/pairs.csv",
                        **{N: f"outputs/armor/hahn/20260822_211457_N{N}/pairs.csv" for N in (8, 16, 32, 64, 128)}}, 'plain', 'final'),
       'fenced fact': ({2: "outputs/armor/hahn/short_N2/pairs.csv", 4: "outputs/armor/hahn/short_N4/pairs.csv",
                        **{N: f"outputs/armor/hahn/20260822_211457_N{N}/pairs.csv" for N in (8, 16, 32, 64, 128)}}, 'fenced', 'rep_t'),
       'trained read': ({N: (glob.glob(f"outputs/loramech/n2_hahn/p1fence_ep10_N{N}*/pairs.csv") or [""])[0] for N in (2, 4, 8, 16, 32, 64, 128)}, 'p1fence', 'final'),
       'gated read': ({N: f"outputs/sparse/s11/gate_oracle_N{N}/pairs.csv" for N in (2, 4, 8, 16, 32, 64, 128)}, 'p1fence', 'final')}
STRAT = {name: {N: strat_med(paths.get(N, ""), arm, locus, 0) for N in (2, 4, 8, 16, 32, 64, 128)} for name, (paths, arm, locus) in SRC.items()}
# FIXEDK (2026-09-21) supersedes the ad-hoc small-N strata: 30 matched pairs per cell, fresh pools for N>=16.
_fk = "outputs/fixedk/fig/fixedk_medians.csv"
if os.path.exists(_fk):
    _map = {'frozen read': 'frozen read', 'fenced fact': 'fenced fact', 'trained read': 'trained read', 'gated read (S9)': 'gated read'}
    for r in csv.DictReader(open(_fk)):
        if int(r['gold']) == 0 and r['median'] and r['cond'] in _map:
            STRAT[_map[r['cond']]][int(r['N'])] = float(r['median'])
with open(f"{FIG}/F10_flip_stratified_gold0.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["N"] + list(STRAT)); 
    for N in (2, 4, 8, 16, 32, 64, 128): w.writerow([N] + [STRAT[k][N] for k in STRAT])
Ns5 = sorted(f6)
with open(f"{FIG}/F10_flip_medians.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["N", "frozen_read", "trained_read", "gated_read", "fenced_fact"])
    for N in Ns5: w.writerow([N, f6[N].get('armor_med'), f6[N].get('n2_med'), f6[N].get('s11_med'), fenced.get(N)])
fig, axs = plt.subplots(1, 4, figsize=(15.2, 4.2), gridspec_kw=dict(width_ratios=[1.15, 1.15, 0.75, 0.9], wspace=0.42))
axs = list(axs); ax_strat = axs.pop(2)
ax = axs[0]
for arm, lab in [('frozen', 'frozen'), ('p1b', 'fine-tuned')]:
    Ns, ev = series(arm, 'evid_L20'); _, ne = series(arm, 'nonevid_L20')
    if not Ns: continue
    ax.plot(Ns, ev, '-o', color=C[arm], lw=2, ms=5, label=f'{lab}, evidence frame')
    ax.plot(Ns, ne, '--o', color=C[arm], lw=1.5, ms=4, mfc='white', label=f'{lab}, non-evidence frame')
Ng, evg = series('gated', 'evid_L20')
ax.plot(Ng, evg, '-o', color=C['gated'], lw=2, ms=5, label='gated, evidence frame (non-evidence = 0)')
xx = np.array([4, 128.]); ax.plot(xx, 0.14 / xx, ':', color=C['muted'], lw=1); ax.text(40, 0.14 / 40 * 0.55, '1/N', color=C['muted'], fontsize=8)
ax.set_xscale('log', base=2); ax.set_yscale('log'); ax.set_xticks(Ns5); ax.set_xticklabels(map(str, Ns5)); ax.set_ylim(0.0015, 0.6)
ax.set_xlabel('N, frames in context'); ax.set_ylabel('attention mass per frame, answer row, layer 20')
ax.set_title('a  Where the mass goes', loc='left', fontweight='bold', color=C['ink']); ax.text(0.99, 0.98, 'k = 2 (k = 1 at N = 2)', transform=ax.transAxes, ha='right', va='top', fontsize=7.5, color=C['muted']); ax.grid(axis='y', color=C['grid'], lw=0.6)
ax.legend(fontsize=7, frameon=False, loc='upper right')
ax = axs[1]
def pl(getter, color, label):
    xs = [n for n in Ns5 if n >= 8 and getter(n) is not None]; ax.plot(xs, [getter(n) for n in xs], '-o', color=color, lw=2, ms=5, label=label)
pl(lambda n: fenced.get(n), C['fact'], 'per-frame fact, fenced  α 0.00 (8–128)')
pl(lambda n: f6[n].get('s11_med'), C['gated'], 'gated read  α 0.035 (8–128)')
pl(lambda n: f6[n].get('n2_med'), C['p1b'], 'fine-tuned read  α 0.80 (8–128)')
pl(lambda n: f6[n].get('armor_med'), C['frozen'], 'frozen read  α 0.72 (8–128)')
ax.axhspan(1.2, 1.4, color=C['grid'], alpha=0.9); ax.text(64, 1.5, 'bf16 noise floor', fontsize=7.5, color=C['muted'])
ax.set_xscale('log', base=2); ax.set_yscale('log'); ax.set_xticks(Ns5); ax.set_xticklabels(map(str, Ns5)); ax.set_ylim(1, 70)
ax.set_xlabel('N, frames in context'); ax.set_ylabel('response to flipping the answer frame, ‖Δh‖ (layer 20)')
ax.set_title('b  What the answer feels', loc='left', fontweight='bold', color=C['ink']); ax.text(0.99, 0.98, 'pooled counts ≤ 8, N = 8…128', transform=ax.transAxes, ha='right', va='top', fontsize=7.5, color=C['muted']); ax.grid(axis='y', color=C['grid'], lw=0.6)
ax.legend(fontsize=7, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
ax = ax_strat
for name, col in [('fenced fact', C['fact']), ('gated read', C['gated']), ('trained read', C['p1b']), ('frozen read', C['frozen'])]:
    xs = [N for N in (2, 4, 8, 16, 32, 64, 128) if STRAT[name][N] is not None]
    ax.plot(xs, [STRAT[name][N] for N in xs], '-o', color=col, lw=2, ms=4)
ax.set_xscale('log', base=2); ax.set_yscale('log'); ax.set_xticks([2, 8, 32, 128]); ax.set_xticklabels(['2', '8', '32', '128']); ax.set_ylim(1, 150)
ax.set_xlabel('N, frames in context'); ax.set_ylabel('‖Δh‖ (layer 20), flip 0 → 1 only')
ax.set_title('b′  Same, at fixed count', loc='left', fontweight='bold', color=C['ink']); ax.text(0.99, 0.98, 'flip 0 → 1, N = 2…128, 30 pairs per cell', transform=ax.transAxes, ha='right', va='top', fontsize=7.5, color=C['muted'])
ax.grid(axis='y', color=C['grid'], lw=0.6)
ax = axs[2]
Ns, e24 = series('p1b', 'evid_L24h20'); _, n24 = series('p1b', 'nonevid_L24h20')
ax.plot(Ns, e24, '-o', color=C['p1b'], lw=2, ms=5, label='fine-tuned, evidence'); ax.plot(Ns, n24, '--o', color=C['p1b'], lw=1.5, ms=4, mfc='white', label='fine-tuned, non-evidence')
Nf, e24f = series('frozen', 'evid_L24h20'); _, n24f = series('frozen', 'nonevid_L24h20')
if Nf: ax.plot(Nf, e24f, '-o', color=C['frozen'], lw=2, ms=5, label='frozen, evidence'); ax.plot(Nf, n24f, '--o', color=C['frozen'], lw=1.5, ms=4, mfc='white', label='frozen, non-evidence')
for N, e, n in zip(Ns, e24, n24): ax.text(N, e * 1.3, f'×{e / n:.1f}', ha='center', fontsize=7.5, color=C['ink'])
ax.set_xscale('log', base=2); ax.set_yscale('log'); ax.set_xticks(Ns5); ax.set_xticklabels(map(str, Ns5)); ax.set_ylim(0.0008, 0.3)
ax.set_xlabel('N, frames in context'); ax.set_ylabel('mass per frame, head 20 of layer 24')
ax.set_title('c  The selector ranks, and dilutes', loc='left', fontweight='bold', color=C['ink']); ax.grid(axis='y', color=C['grid'], lw=0.6)
ax.text(0.03, 0.03, 'evidence vs non-evidence AUC ≥ 0.993 at every N', transform=ax.transAxes, fontsize=7.5, color=C['ink2'])
ax.legend(fontsize=7, frameon=False, loc='upper right')
fig.suptitle('One denominator, two observables: the weights dilute, the answer dilutes, the gate flattens both', x=0.01, ha='left', fontsize=10.5, color=C['ink'])
fig.savefig(f"{FIG}/F10_dispersion_hahn.png", dpi=170, bbox_inches='tight', facecolor='white')
print("F10 written;", len(rows), "cells")

# ---------------- F11: temperature coupling ----------------
taus = sorted({r['tau'] for r in rows if r['arm'] == 'p1b'})
if len(taus) > 1:
    s0 = {}
    for d in glob.glob("outputs/sparse/s0_tau*_L12"):
        tau = float(re.search(r"tau([\d.]+)_L12", d).group(1))
        f = glob.glob(f"{d}/*/longn_eval.csv")
        if not f: continue
        for r in csv.DictReader(open(f[0])):
            N = int(re.search(r"_N(\d+)\.txt", r['source']).group(1)); s0[(tau, N)] = (float(r['accuracy']), float(r['mean_signed_err']))
    fig, axs = plt.subplots(1, 3, figsize=(12.2, 3.9), gridspec_kw=dict(wspace=0.4))
    out = []
    for N, col in [(32, C['p1b']), (128, C['ink'])]:
        T = [t for t in taus if ('p1b', N) in {(r['arm'], r['N']) for r in rows if r['tau'] == t}]
        if not T: continue
        get = lambda key: [next(r[key] for r in rows if r['arm'] == 'p1b' and r['N'] == N and r['tau'] == t) for t in T]
        ne, ev, conc, oth = get('nonevid_L20'), get('evid_L20'), get('conc_L20'), get('other_L20')
        kk = get('k')[0]
        comp = [n_ * (N - kk) for n_ in ne]  # total competitor mass at the panel's k
        evtot = [e_ * kk for e_ in ev]
        axs[0].plot(T, comp, '-o', color=col, lw=2, ms=5, label=f'N = {N}: non-evidence frames')
        axs[0].plot(T, oth, '--o', color=col, lw=1.5, ms=4, mfc='white', label=f'N = {N}: prompt and sink')
        axs[0].plot(T, evtot, ':o', color=col, lw=1.5, ms=4, label=f'N = {N}: evidence frames')
        axs[1].plot(T, conc, '-o', color=col, lw=2, ms=5, label=f'N = {N}')
        acc = [s0.get((t, N), (np.nan, np.nan))[0] for t in T]
        axs[2].plot(T, acc, '-o', color=col, lw=2, ms=5, label=f'N = {N}')
        for t, c_, e_, n_, o_, a_ in zip(T, comp, ev, ne, oth, acc): out.append(dict(N=N, k=kk, tau=t, competitor_mass=c_, evidence_mass=e_ * kk, evid_per_frame=e_, nonevid_per_frame=n_, other_mass=o_, conc=next(r['conc_L20'] for r in rows if r['arm']=='p1b' and r['N']==N and r['tau']==t), s0_accuracy=a_))
    axs[0].set_yscale('log'); axs[0].set_ylim(0.003, 1.0)
    axs[0].set_title('a  Where the mass goes as τ rises', loc='left', fontweight='bold', color=C['ink']); axs[0].set_ylabel('total attention mass at the answer row (layer 20; k = 2), log')
    axs[1].set_title('b  Evidence mass concentrates', loc='left', fontweight='bold', color=C['ink']); axs[1].set_ylabel('largest evidence frame ÷ mean evidence frame'); axs[1].axhline(1, color=C['muted'], lw=0.8, ls=':'); axs[1].text(1.02, 1.02, 'uniform over evidence', fontsize=7.5, color=C['muted'])
    axs[2].set_title('c  The count breaks', loc='left', fontweight='bold', color=C['ink']); axs[2].set_ylabel('exact match, S0 sweep, same τ'); axs[2].set_ylim(0, 1)
    for a in axs: a.set_xlabel('temperature multiplier τ on layers ≥ 12'); a.grid(axis='y', color=C['grid'], lw=0.6)
    axs[0].legend(fontsize=7, frameon=False, loc='lower left', ncol=2); axs[1].legend(fontsize=7.5, frameon=False); axs[2].legend(fontsize=7.5, frameon=False)
    fig.suptitle('One temperature, two requirements: sharpening removes competitors and destroys uniformity over the evidence', x=0.01, ha='left', fontsize=10.5, color=C['ink'])
    fig.savefig(f"{FIG}/F11_temperature_coupling.png", dpi=170, bbox_inches='tight', facecolor='white')
    with open(f"{FIG}/F11_temperature_coupling.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out[0].keys()); w.writeheader(); w.writerows(out)
    print("F11 written;", len(out), "tau cells")
else:
    print("F11 skipped: no tau cells yet")

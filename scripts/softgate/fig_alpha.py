"""F12 — SOFTGATE A1: the read's length exponent α slides with the gate penalty B.
Reads outputs/softgate/a1_B{B}_N{N}/pairs.csv (S9b nfree adapter, oracle soft gate, 50 pairs / 40 @128).
Panel a: median ‖Δh‖ (L20, final row) vs N per B.  Panel b: α(B) with bootstrap CI vs the share-law curve
(m = k e^s / (k e^s + (N-k)/B + C), s=0.30, C=8, k=4; slope of log Δm vs log N over 8..128).
Writes outputs/softgate/fig/F12_alpha_vs_B.{png,csv}.  Anchors: hard gate α 0.03 [-0.06,0.12] ≈ S11's 0.035."""
import csv, os, math, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
rng = np.random.default_rng(0)
NS = (8, 16, 32, 64, 128); BS = [1, 2, 8, 32, 128, 0]; FLOOR = (1.2, 2.4)
C = {'ink': '#1f2328', 'muted': '#6b7280', 'grid': '#e5e7eb', 'law': '#1f77b4', 'meas': '#d95f02', 'floor': '#e5e7eb'}
def cell(B, N):
    f = f"outputs/softgate/a1_B{B}_N{N}/pairs.csv"
    return np.array([float(r['dnorm']) for r in csv.DictReader(open(f)) if r['flip_kind'] == 'evid' and r['layer'] == '20' and r['locus'] == 'final'])
def alpha(cells, nb=2000):
    Ns = sorted(cells); x = np.log(Ns); med = np.array([np.median(cells[N]) for N in Ns])
    a = -np.polyfit(x, np.log(med), 1)[0]
    bs = [-np.polyfit(x, np.log([np.median(rng.choice(cells[N], len(cells[N]))) for N in Ns]), 1)[0] for _ in range(nb)]
    return a, *np.percentile(bs, [2.5, 97.5]), med
def law(B, s=0.30, Cc=8.0, k=4):
    es = math.exp(s); m = lambda N, kk: kk * es / (kk * es + ((N - kk) / B if B > 0 else 0) + Cc)
    return -np.polyfit(np.log(NS), np.log([m(N, k + 1) - m(N, k) for N in NS]), 1)[0]
rows = []; curves = {}
for B in BS:
    cells = {N: cell(B, N) for N in NS}; a, lo, hi, med = alpha(cells); curves[B] = med
    rows.append(dict(B=B, alpha=a, lo=lo, hi=hi, law=law(B), **{f'med_N{N}': m for N, m in zip(NS, med)}))
os.makedirs("outputs/softgate/fig", exist_ok=True)
with open("outputs/softgate/fig/F12_alpha_vs_B.csv", "w", newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': C['muted'], 'axes.labelcolor': C['ink'], 'xtick.color': C['ink'], 'ytick.color': C['ink']})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 3.9), gridspec_kw=dict(width_ratios=[1.15, 1]))
cmap = plt.get_cmap('viridis')
for i, B in enumerate(BS):
    lab = {1: 'B = 1  (gate removed)', 0: 'B = ∞  (hard gate)'}.get(B, f'B = {B}')
    a1.plot(NS, curves[B], 'o-', color=cmap(0.05 + 0.85 * i / (len(BS) - 1)), lw=1.8, ms=4.5, label=lab)
a1.axhspan(*FLOOR, color=C['floor'], zorder=0); a1.text(9, 1.35, 'bf16 noise floor', fontsize=7.5, color=C['muted'])
a1.set_xscale('log', base=2); a1.set_yscale('log'); a1.set_xticks(NS); a1.set_xticklabels(NS); a1.set_ylim(1.0, 400)
a1.set_xlabel('N, frames in context'); a1.set_ylabel('response to flipping one evidence frame\n‖Δh‖ at layer 20, answer row (median)')
a1.set_title('a  A softer gate lets more of N back in', loc='left', fontweight='bold', color=C['ink'])
a1.legend(frameon=False, fontsize=7.3, loc='upper left', ncol=3, columnspacing=1.0, handlelength=1.6); a1.grid(axis='y', color=C['grid'], lw=0.6)
for s in ('top', 'right'): a1.spines[s].set_visible(False); a2.spines[s].set_visible(False)
xs = {1: 1, 2: 2, 8: 8, 32: 32, 128: 128, 0: 512}
Bl = np.geomspace(1, 512, 200); a2.plot(Bl, [law(b) for b in Bl], color=C['law'], lw=1.8, label='share law (s=0.30, C=8, k=4, fitted at N=8)')
a2.axhline(law(0), color=C['law'], lw=1, ls=':')
meas = [(xs[r['B']], r['alpha'], r['lo'], r['hi'], r['B']) for r in rows]
for x, a, lo, hi, B in meas:
    if B == 1:
        a2.errorbar([x], [a], yerr=[[a - lo], [hi - a]], fmt='o', mfc='white', color=C['meas'], ms=6, capsize=2)
        a2.annotate('gate removed: response at the floor (2.2), reader is content-blind', (x, a), xytext=(1.35, -0.105), fontsize=7.0, color=C['muted'], va='center')
    else:
        a2.errorbar([x], [a], yerr=[[a - lo], [hi - a]], fmt='o', color=C['meas'], ms=6, capsize=2, label='measured, S9b reader (95% bootstrap CI)' if B == 2 else None)
a2.annotate('B = 2: plateau to N = 32,\nthen collapse to the floor', (2, 0.66), xytext=(2.7, 0.56), fontsize=7.2, color=C['muted'], arrowprops=dict(arrowstyle='-', color=C['muted'], lw=0.7))
a2.set_xscale('log', base=2); a2.set_xticks([1, 2, 8, 32, 128, 512]); a2.set_xticklabels(['1', '2', '8', '32', '128', '∞'])
a2.set_ylim(-0.14, 0.92); a2.set_xlabel('gate penalty B  (gated columns weighted 1/B)'); a2.set_ylabel('length exponent α of the read (N = 8…128)')
a2.set_title('b  The exponent slides with the penalty', loc='left', fontweight='bold', color=C['ink'])
a2.legend(frameon=False, fontsize=7.3, loc='upper right', bbox_to_anchor=(1.0, 1.0)); a2.grid(axis='y', color=C['grid'], lw=0.6)
fig.suptitle('One knob between the frozen read and the gated read: the residual competitor mass (N−k)/B', x=0.01, ha='left', fontsize=10.5, color=C['ink'])
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig("outputs/softgate/fig/F12_alpha_vs_B.png", dpi=200); print("F12 written"); print(open("outputs/softgate/fig/F12_alpha_vs_B.csv").read())

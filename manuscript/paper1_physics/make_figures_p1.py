# -*- coding: utf-8 -*-
"""
Figures for Paper 1 (manuscript/paper1_physics/paper1.tex). Reads the result files directly and evaluates
the perturbation series with the analysis code; nothing is transcribed.

  fig_model.pdf   schematic: couplings; structures along y with the local fields of Table I
  fig_wedge.pdf   reanalysis_v3/audit_12/secVA_numbers.json (Table III crossings); O(g^2), O(g^6) boundaries;
                  onset of order from the step-2 descending anneals (sse_validation/branches)
  fig_f23.pdf     sse_validation/seq_branches/seqbr_*.csv, free energies as in the frozen S5a analysis
  fig_window.pdf  reanalysis_v3/s5a_analysis/s5a_P*.json (edges, T3), posthoc_P6/ (pooled follow-up)
  fig_anneal.pdf  sse_validation/branches/branches_L12_k0.75_r{0,1}.csv (descending anneal and templates)
  fig_validation.pdf  sse_validation/sse_ed_benchmark.csv; branches_L12_*.csv (g0 energies); s5a_analysis/ (z_g0);
                  sse_validation/tfim2d{,_T0.15}/tfim2d_summary.json

Style follows identity_lab/hoang/paper/v5_pra/make_figures_v5.py (house style of 2026-09-24): one red
family for the simulation data, near-black for the perturbation theory, direct labels instead of legends,
serif 8 pt, full text width, the same row height and panel titles.

Run from vu_work/ with the plotting environment:  .venv/bin/python manuscript/paper1_physics/make_figures_p1.py
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

HERE = pathlib.Path(__file__).resolve().parent
VU = HERE.parent.parent
sys.path.insert(0, str(VU / 'experiments'))
import s5a_analyze_v3 as A  # noqa: E402  (theory series, free energies, crossings)
import pt_order8_v3 as P8   # noqa: E402  (eighth-order series, exact kappa)
from fractions import Fraction as Fr  # noqa: E402

RES = VU / '10_results' / 'bilayer_annni_paper_results' / 'reanalysis_v3'
SEQ = RES / 'sse_validation' / 'seq_branches'
BR = RES / 'sse_validation' / 'branches'
S5A = RES / 's5a_analysis'
OUT = HERE / 'figures'
OUT.mkdir(exist_ok=True)
CACHE = OUT / 'theory_cache.json'

# ---------------------------------------------------------------- house style (v5)
DARK, MID, LIGHT, PALE = '#99000d', '#e0301e', '#fb8f67', '#f7a386'
LEDGE = '#b8431f'
REF, INK, MUTED = '#252525', '#0b0b0b', '#8a8985'
GRID, FILL = '#e6e5e1', '#fbe3da'
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7,
    'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.edgecolor': MUTED, 'axes.linewidth': 0.6,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.labelcolor': INK, 'text.color': INK,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'axes.grid.axis': 'y', 'grid.color': GRID,
    'grid.linewidth': 0.5, 'savefig.bbox': 'tight', 'savefig.dpi': 300,
    'lines.linewidth': 1.4, 'mathtext.fontset': 'dejavuserif',
})
WIDE, ROW = 7.0, 2.4
# series styles: color, edge, marker, face
STY = {'r0': (DARK, 'white', 's', DARK), 'r1': (LIGHT, LEDGE, 'D', LIGHT),
       '2': (DARK, 'white', 's', DARK), '3': (MID, 'white', '^', MID), '23': (REF, 'white', 'o', REF),
       '233': (LIGHT, LEDGE, 'D', LIGHT), '2333': (PALE, LEDGE, 'v', 'white')}


def panel(ax, tag):
    ax.text(-0.16, 1.02, tag, transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom')


def title(ax, text):
    ax.set_title(text, fontsize=7.5, color=INK)


def pts(ax, x, y, key, yerr=None, hollow=False, ms=4.5, ls='none', zorder=3, marker=None, **kw):
    c, e, m, f = STY[key]
    m = marker or m
    ax.errorbar(x, y, yerr=yerr, color=c, marker=m, ms=ms, mfc='white' if hollow else f,
                mec=(e if not hollow else (LEDGE if key in ('r1', '233', '2333') else c)),
                mew=0.9 if hollow else 0.6, capsize=2, elinewidth=0.8, ls=ls, lw=1.0, zorder=zorder, **kw)


def label(ax, text, xy, xytext, color=INK, fs=7, ha='left', lead=True):
    ax.annotate(text, xy=xy, xytext=xytext, textcoords='data', fontsize=fs, color=color, va='center', ha=ha,
                arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.4, shrinkA=1, shrinkB=2) if lead else None)


def spread(ax, ends, x, min_sep, fs=7):
    """Direct labels at a common x, spread vertically; ends = [(text, xe, ye)]."""
    ends = sorted(ends, key=lambda e: e[2])
    ys = [e[2] for e in ends]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_sep)
    for (txt, xe, ye), y in zip(ends, ys):
        ax.annotate(txt, xy=(xe, ye), xytext=(x, y), textcoords='data', va='center', fontsize=fs, color=INK,
                    arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.4, shrinkA=1, shrinkB=3)
                    if abs(y - ye) > min_sep / 2 or abs(x - xe) > 0 else None)


def S(seq):
    return r'$\langle' + seq + r'\rangle$'


# ---------------------------------------------------------------- theory (exact kappa), cached
_tc = json.loads(CACHE.read_text()) if CACHE.exists() else {}


def series8(seq, kappa, r):
    return [float(x) for x in P8.per_spin_series8(seq, Fr(str(round(float(kappa), 6))), Fr(r))]


def tcross(old, new, kappa, r, order):
    key = f'{old}|{new}|{kappa:.6f}|{r}|{order}'
    if key not in _tc:
        try:
            if order == 8:          # root of the O(g^8) difference nearest the O(g^6) crossing (NaN if none near it)
                g6 = tcross(old, new, kappa, r, 6)
                d = [a - b for a, b in zip(series8(new, kappa, r), series8(old, kappa, r))]
                u = np.roots(d[::-1])
                gs = [np.sqrt(x.real) for x in u if abs(x.imag) < 1e-12 and 0.05 ** 2 < x.real < 2.6 ** 2]
                gs = [x for x in gs if abs(x - g6) < 0.3]
                _tc[key] = float(min(gs, key=lambda x: abs(x - g6))) if gs else float('nan')
            else:
                _tc[key] = float(A.pt_cross(old, new, float(kappa), r, order))
        except Exception:
            _tc[key] = float('nan')
    return _tc[key]


def save_cache():
    CACHE.write_text(json.dumps(_tc))


def F3(r):
    return 6.0 / ((2 + r) * (3 + r) * (5 + r))


def onset(kappa, r, thr):
    """Field at which the mean S(pi/3) of the descending L = 12 anneal first exceeds thr (from above)."""
    d = pd.read_csv(BR / f'branches_L12_k{kappa:g}_r{r}.csv')
    d = d[d.part == 'descending'].groupby('g').Sl2_qpi3.mean().sort_index(ascending=False)
    g, s = d.index.to_numpy(), d.to_numpy()
    i = np.nonzero(s > thr)[0][0]
    return float(g[i - 1] + (thr - s[i - 1]) * (g[i] - g[i - 1]) / (s[i] - s[i - 1]))


# ---------------------------------------------------------------- Fig. 1: model and structures
def fig_model():
    fig = plt.figure(figsize=(WIDE, ROW))
    a = fig.add_axes([0.0, 0.02, 0.30, 0.92])
    b = fig.add_axes([0.36, 0.02, 0.64, 0.92])
    for ax in (a, b):
        ax.axis('off')

    # (a) two layers, frustrated axis y vertical; layer 2 offset for perspective
    nx, ny, off = 3, 5, np.array([0.42, 0.30])
    for layer, (o, col) in enumerate(((np.zeros(2), REF), (off, MUTED))):
        for i in range(nx):
            a.plot([i + o[0]] * 2, [o[1], ny - 1 + o[1]], color=col, lw=0.8, zorder=1 + layer)
        for j in range(ny):
            a.plot([o[0], nx - 1 + o[0]], [j + o[1]] * 2, color=col, lw=0.8, zorder=1 + layer)
        xs, ys = np.meshgrid(np.arange(nx) + o[0], np.arange(ny) + o[1])
        a.scatter(xs, ys, s=16, color=col if layer else REF, zorder=3 + layer, edgecolors='white', linewidths=0.4)
    for j in range(ny):          # rungs on the front column
        a.plot([nx - 1, nx - 1 + off[0]], [j, j + off[1]], color=LIGHT, lw=0.9, ls=(0, (2, 1.5)), zorder=2)
    arc = FancyArrowPatch((0, 0), (0, 2), connectionstyle='arc3,rad=-0.6', arrowstyle='-',
                          color=DARK, lw=1.1, zorder=4)   # next-nearest-neighbor bond along y
    a.add_patch(arc)
    a.text(0.5, -0.45, '$J_x=J_0$', ha='center', fontsize=7)
    a.text(-0.12, 3.5, '$J_y=J_0$', ha='right', fontsize=7)
    a.text(-0.68, 1.0, r'$J_2=\kappa J_0$', ha='right', fontsize=7, color=DARK)
    a.text(nx - 1 + off[0] + 0.3, 1.0, r'$J_\perp=rJ_0$', ha='left', fontsize=7, color=LEDGE)
    a.annotate('', xy=(-1.9, 4.2), xytext=(-1.9, 3.0), arrowprops=dict(arrowstyle='->', color=MUTED, lw=0.6))
    a.annotate('', xy=(-0.7, 3.0), xytext=(-1.9, 3.0), arrowprops=dict(arrowstyle='->', color=MUTED, lw=0.6))
    a.text(-1.9, 4.28, '$y$', fontsize=7, color=MUTED, ha='center', va='bottom')
    a.text(-0.6, 3.0, '$x$', fontsize=7, color=MUTED, va='center')
    a.text(nx - 1 + off[0] + 0.12, ny - 1 + off[1], 'layer 2', fontsize=7, color=MUTED, va='center')
    a.text(nx - 1 + 0.12, ny - 1 - 0.28, 'layer 1', fontsize=7, color=REF, va='center')
    a.set_xlim(-2.2, 3.9)
    a.set_ylim(-1.2, 4.9)
    a.set_aspect('equal')
    title(a, 'bilayer, frustrated along $y$')
    a.text(-0.02, 1.02, '(a)', transform=a.transAxes, fontsize=8, fontweight='bold', va='bottom')

    # (b) structures along y at kappa = 1/2, colored by the local field h of Eq. (h) (r = 0)
    rows = [('FM', [1] * 24), (S('3'), None), (S('23'), None), (S('2'), None)]
    def seq_spins(lengths, n=16):
        s, v = [], 1
        while len(s) < n + 8:
            for L in lengths:
                s += [v] * L
                v = -v
        return s
    rows[1] = (S('3'), seq_spins([3, 3]))
    rows[2] = (S('23'), seq_spins([2, 3]))
    rows[3] = (S('2'), seq_spins([2, 2]))
    hcol = {2: (DARK, DARK), 3: ('#d9d8d4', '#b9b8b4'), 4: (LIGHT, LEDGE), 5: ('white', LEDGE)}
    n = 16
    for k, (name, s) in enumerate(rows):
        y = 3 - k
        ext = np.array(s[1:n + 5], int)   # displayed site j is ext[j + 2]; two neighbors kept on each side
        s = ext[2:n + 2]
        b.text(-1.0, y, name, ha='right', va='center', fontsize=8)
        for j in range(n):
            c = j + 2
            h = 2 + ext[c] * ext[c + 1] + ext[c] * ext[c - 1] - 0.5 * (ext[c] * ext[c + 2] + ext[c] * ext[c - 2])
            h = int(round(h))
            fc, ec = hcol[h]
            b.scatter(j, y, s=62, marker='^' if s[j] > 0 else 'v', color=fc, edgecolors=ec, linewidths=0.7, zorder=3)
            if j < n - 1 and s[j] != s[j + 1]:
                b.plot([j + 0.5] * 2, [y - 0.36, y + 0.36], color=MUTED, lw=0.6, zorder=1)
    # key
    for (h, txt), (x0, ky) in zip(((2, 'next to a wall (gains most)'), (3, 'bulk, or two-site domain'),
                                   (5, 'center of a three-site domain')),
                                  ((-0.5, -0.75), (8.5, -0.75), (-0.5, -1.35))):
        fc, ec = hcol[h]
        b.scatter(x0, ky, s=50, marker='^', color=fc, edgecolors=ec, linewidths=0.7)
        b.text(x0 + 0.5, ky, f'$h={h}+r$: {txt}', fontsize=6.5, va='center')
    b.text(n - 0.5, 3.62, 'position along $y$ $\\rightarrow$', ha='right', fontsize=7, color=MUTED)
    b.set_xlim(-2.2, n + 3.5)
    b.set_ylim(-1.7, 3.8)
    title(b, r'structures at $\kappa=1/2$ and the local field $h$ of each spin')
    b.text(-0.01, 1.02, '(b)', transform=b.transAxes, fontsize=8, fontweight='bold', va='bottom')
    fig.savefig(OUT / 'fig_model.pdf')
    fig.savefig(OUT / 'fig_model.png')


# ---------------------------------------------------------------- Fig. 2: the <3> wedge
def fig_wedge():
    N = json.loads((RES / 'audit_12' / 'secVA_numbers.json').read_text())['L12']
    pts_r = {0: [], 1: []}
    for key, v in N.items():
        k, r = key.split('_')
        k, r = float(k), int(r)
        pts_r[r].append((k, v['best']['g'], v['best']['se'], v['theory']['O6']))
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(WIDE, ROW), gridspec_kw={'width_ratios': [1, 1, 1]})
    out = {}
    for ax, r, tag, ytop in ((a, 0, '(a)', 2.75), (b, 1, '(b)', 3.85)):
        f = F3(r)
        lam_g = 0.5 * (2 + r)
        # O(g^6) boundaries at the actual kappa
        kF = np.round(np.concatenate([np.linspace(0.44, 0.4985, 16), [0.4995]]), 5)
        kP = np.round(np.concatenate([[0.5005], np.linspace(0.5015, 0.56, 14), np.linspace(0.57, 0.78, 12)]), 5)
        gF = np.array([tcross(None, (3,), k, r, 6) for k in kF])
        glo = np.array([tcross((2,), (2, 3), k, r, 6) for k in kP])
        ghi = np.array([tcross((2, 3), (3,), k, r, 6) for k in kP])
        g32 = np.array([tcross((2,), (3,), k, r, 6) for k in kP])
        kF, gF = np.append(kF, 0.5), np.append(gF, 0.0)
        kP, glo, ghi, g32 = (np.insert(kP, 0, 0.5), np.insert(glo, 0, 0.0), np.insert(ghi, 0, 0.0),
                             np.insert(g32, 0, 0.0))
        # onset of order (L = 12 anneals): S(pi/3) through 0.1 and 0.05
        ko = [0.47, 0.48, 0.49, 0.51, 0.52, 0.53, 0.5625, 0.625, 0.75]
        on1 = np.array([onset(k, r, 0.1) for k in ko])
        on2 = np.array([onset(k, r, 0.05) for k in ko])
        out[('onset', r)] = (on1.min(), on1.max(), on2.min(), on2.max())
        kk = np.linspace(0.44, 0.78, 200)
        lo_on, hi_on = np.interp(kk, ko, on1), np.interp(kk, ko, on2)
        ax.fill_between(kk, lo_on, hi_on, color='#d9d8d4', lw=0, zorder=1)
        ax.fill_between(kk, hi_on, ytop, color='#f1f0ed', lw=0, zorder=0)
        # O(g^6) phases where the series is controlled (g <= lambda = 0.5): <3> shaded, <23> in red
        kb = np.concatenate([kF, kP[1:]])
        gb = np.concatenate([gF, ghi[1:]])
        ok = np.isfinite(gb) & (gb < lam_g)
        ax.fill_between(kb, np.where(ok, gb, lam_g), lam_g, where=ok, color=FILL, lw=0, zorder=1)
        okP = np.isfinite(ghi) & (ghi <= lam_g)
        ax.fill_between(kP, glo, ghi, where=okP, color=MID, lw=0, alpha=0.8, zorder=2)
        for kx, gx in ((kF, gF), (kP, glo), (kP, ghi)):
            m = np.isfinite(gx) & (gx <= lam_g)
            ax.plot(kx[m], gx[m], color=REF, lw=0.9, zorder=3)
        # O(g^6) branch crossings (Table III compares with these), dashed beyond the controlled range
        mF = np.isfinite(gF) & (gF >= lam_g * 0.97)
        ax.plot(kF[mF], gF[mF], color=REF, lw=0.7, ls=(0, (4, 2)), zorder=3)
        ax.plot(kP, g32, color=REF, lw=0.7, ls=(0, (4, 2)), zorder=3)
        # O(g^2) wedge, Eq. (wedge)
        gg = np.linspace(0, ytop, 100)
        ax.plot(0.5 - gg ** 2 * f / 8, gg, color=MUTED, lw=0.7, ls=':', zorder=2)
        ax.plot(0.5 + gg ** 2 * f / 4, gg, color=MUTED, lw=0.7, ls=':', zorder=2)
        # lambda = 0.5
        ax.axhline(lam_g, color=MUTED, lw=0.5, ls=(0, (6, 3)), zorder=1)
        ax.text(0.772, lam_g + 0.04, r'$\lambda=0.5$', fontsize=6.5, color=MUTED, ha='right', va='bottom')
        # QMC crossings (Table III): filled for lambda <= 0.5; circles for kappa > 1/2, where the <3>-<2> crossing
        # lies inside the <23> window and is not a phase boundary
        P = sorted(pts_r[r])
        for k, g, se, g6 in P:
            pts(ax, [k], [g], 'r0' if r == 0 else 'r1', yerr=[se], hollow=g / (2 + r) > 0.5, zorder=5,
                marker='o' if k > 0.5 else None)
        # in-panel key for the two kinds of crossing
        key = 'r0' if r == 0 else 'r1'
        for yf, mk, txt in ((0.74, None, r'FM$|$' + S('3') + ' crossing'),
                            (0.655, 'o', S('3') + '$-$' + S('2') + ' crossing,\ninside the ' + S('23') + ' window')):
            pts(ax, [0.474], [yf * ytop], key, marker=mk, zorder=5)
            ax.text(0.484, yf * ytop, txt, fontsize=6.3, color=INK, va='center' if mk is None else 'top',
                    linespacing=1.1, zorder=6, bbox=dict(fc='white', ec='none', pad=0.6, alpha=0.85))
        ax.set_xlim(0.44, 0.78)
        ax.set_ylim(0, ytop)
        ax.set_xticks([0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75])
        ax.set_xticklabels(['0.45', '0.5', '', '0.6', '', '0.7', ''])
        ax.set_xlabel(r'$\kappa=J_2/J_0$')
        ax.grid(False)
        title(ax, f'$r={r}$')
        panel(ax, tag)
        # region labels
        ax.text(0.452, 0.10 * ytop, 'FM', fontsize=8, color=INK)
        ax.text(0.745, 0.12 * ytop, S('2'), fontsize=8, color=INK, ha='right')
        ax.text(0.507, 0.86 * lam_g, S('3'), fontsize=8, color=DARK, ha='center')
        ax.text(0.61, np.interp(0.61, kk, (lo_on + hi_on) / 2), 'onset of order', fontsize=6.5,
                color=INK, ha='center', va='center')
        i23 = np.searchsorted(kP, 0.545)
        label(ax, S('23'), (kP[i23], 0.5 * (glo[i23] + ghi[i23])),
              (0.60, 0.5 * (glo[i23] + ghi[i23]) - 0.18 * ytop), color=INK)
    a.set_ylabel(r'transverse field $g=\Gamma/J_0$')

    # (c) deviation from O(g^6) against lambda
    for r, key in ((0, 'r0'), (1, 'r1')):
        P = sorted(pts_r[r], key=lambda p: p[1] / (2 + r))
        lam = np.array([p[1] / (2 + r) for p in P])
        dev = np.array([100 * (p[1] / p[3] - 1) for p in P])
        err = np.array([100 * p[2] / p[3] for p in P])
        pts(c, lam, dev, key, yerr=err, ls='-', zorder=4 if r == 0 else 3)
        out[('dev', r)] = list(zip(lam.round(3), dev.round(2)))
        T8 = {(round(q['kappa'], 4), q['r']): q['O8'] for q in
              json.loads((RES / 'perturbation' / 'pt_order8_tables.json').read_text())['tab3']}
        l8, d8, e8 = [], [], []
        for (k, g, se, g6) in P:
            o8 = T8.get((round(k, 4), r))
            if o8:
                l8.append(g / (2 + r)); d8.append(100 * (g / o8 - 1)); e8.append(100 * se / o8)
        pts(c, l8, d8, key, yerr=e8, hollow=True, ms=4.0, zorder=5)
        out[('dev8', r)] = list(zip(np.round(l8, 3), np.round(d8, 2)))
        c.annotate(f'$r={r}$', xy=(lam[-1], dev[-1]), xytext=(lam[-1] + 0.02, dev[-1]),
                   fontsize=7, va='center')
    c.axhspan(-1, 1, color='#f1f0ed', lw=0, zorder=0)
    c.axhline(0, color=REF, lw=0.7, zorder=1)
    c.axvline(0.5, color=MUTED, lw=0.5, ls=(0, (6, 3)), zorder=1)
    c.text(0.49, 15.5, r'$\lambda=0.5$', fontsize=6.5, color=MUTED, ha='right')
    c.text(0.62, -2.3, r'$\pm1\%$ band', fontsize=6.5, color=MUTED)
    c.set_xlim(0.18, 1.02)
    c.set_ylim(-3, 18)
    c.set_xlabel(r'$\lambda=g^\star/(2+r)$')
    c.set_ylabel(r'$g^\star/g^\star_n-1$ (%)')
    c.text(0.21, 12.5, r'filled: $n=6$' + '\n' + r'open: $n=8$', fontsize=6.5, color=INK, va='top')
    title(c, r'crossings against the series')
    panel(c, '(c)')
    fig.tight_layout(w_pad=1.6)
    fig.savefig(OUT / 'fig_wedge.pdf')
    fig.savefig(OUT / 'fig_wedge.png')
    save_cache()
    return out


# ---------------------------------------------------------------- Fig. 3: <23> free energies
def branch(path, name):
    bs, k, r = A.load_file(path)
    b = [x for x in bs if x.name == name][0]
    return b, A.free_energy(b, k, r, 'rich'), k, r


def fdiff(X, Y):
    """Mean(F_X) - mean(F_Y) on the common grid, with the Welch standard error (cut chains excluded)."""
    (bX, FX), (bY, FY) = X, Y
    g, x, y = A.align(bX, FX, bY, FY)
    nx, ny = np.sum(np.isfinite(x), 0), np.sum(np.isfinite(y), 0)
    d = np.nanmean(x, 0) - np.nanmean(y, 0)
    se = np.sqrt(np.nanvar(x, 0, ddof=1) / nx + np.nanvar(y, 0, ddof=1) / ny)
    return g, d, se


def fig_f23():
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, ROW))
    out = {}
    for ax, (case, k, r, xl, yl), tag in zip(axes[:2], (('P1', 0.53, 0, (0.64, 0.80), (-12, 16)),
                                                        ('P3', 0.53, 1, (1.04, 1.21), (-9, 12))), ('(a)', '(b)')):
        b2 = branch(A.seqbr(20, 20, k, r, 0.15, '23-2'), '<2>')[:2]
        a2 = branch(A.seqbr(20, 20, k, r, 0.15, '23-2'), '<23>')[:2]
        b3 = branch(A.seqbr(20, 30, k, r, 0.15, '23-3'), '<3>')[:2]
        a3 = branch(A.seqbr(20, 30, k, r, 0.15, '23-3'), '<23>')[:2]
        J = json.loads((S5A / f's5a_{case}.json').read_text())
        lo, hi = J['edges']['lo']['rich/cubic']['value'], J['edges']['hi']['rich/cubic']['value']
        gc = J['theory']['g_c']
        ax.axvspan(lo, hi, color=FILL, lw=0, zorder=0)
        ax.axhline(0, color=REF, lw=0.7, zorder=1)
        gg = np.linspace(xl[0], xl[1], 200)
        s23 = A.series((2, 3), k, r, 6)
        ends = []
        for name, X, Y in (('2', b2, a2), ('3', b3, a3)):
            th = 1e4 * (A.e_series(A.series(tuple(int(ch) for ch in name), k, r, 6), gg) - A.e_series(s23, gg))
            ax.plot(gg, th, color=REF, lw=0.8, ls=(0, (4, 2)), zorder=2)
            th8 = 1e4 * (A.e_series(series8(tuple(int(ch) for ch in name), k, r), gg)
                         - A.e_series(series8((2, 3), k, r), gg))
            ax.plot(gg, th8, color=REF, lw=0.8, ls=':', zorder=2)
            g, d, se = fdiff(X, Y)
            m = (g >= xl[0] - 1e-9) & (g <= xl[1] + 1e-9)
            pts(ax, g[m], 1e4 * d[m], name, yerr=1e4 * se[m], ls='-', ms=3.6, zorder=4)
            inside = np.nonzero(m & (1e4 * d > yl[0]) & (1e4 * d < yl[1]))[0]
            j = inside[-1]
            ax.text(g[j] + 0.03 * (xl[1] - xl[0]), 1e4 * d[j], S(name), fontsize=7, va='center', color=INK)
            out[(case, name)] = (g[m].round(4).tolist(), (1e4 * d[m]).round(2).tolist())
        ax.axvline(gc, color=MUTED, lw=0.5, ls=':', zorder=1)
        ax.text(gc, yl[1] * 0.97, ' $g_c$', fontsize=6.5, color=MUTED, va='top')
        ax.text(0.5 * (lo + hi), yl[0] + 0.06 * (yl[1] - yl[0]), S('23'), fontsize=7, color=DARK, ha='center')
        ax.set_xlim(xl[0], xl[1] + 0.13 * (xl[1] - xl[0]))
        ax.set_ylim(*yl)
        ax.set_xlabel('$g$')
        title(ax, rf'$\kappa={k:g}$, $r={r}$')
        panel(ax, tag)
    axes[0].set_ylabel(r'$F_X-F_{\langle23\rangle}$ ($10^{-4}J_0$ per spin)')
    axes[0].text(0.645, 4.0, r'dashed: $O(g^6)$' + '\n' + r'dotted: $O(g^8)$', fontsize=6.5, color=REF, va='center')

    # (c) kappa = 0.5625, r = 0: everything relative to <3> (20 x 30); the lowest curve is the stable structure
    c = axes[2]
    k, r, xl, yl = 0.5625, 0, (0.92, 1.50), (-32, 12)
    base = branch(A.seqbr(20, 30, k, r, 0.15, '23-3'), '<3>')[:2]
    comps = [('2', branch(A.seqbr(20, 20, k, r, 0.15, '23-2'), '<2>')[:2]),
             ('23', branch(A.seqbr(20, 30, k, r, 0.15, '23-3'), '<23>')[:2]),
             ('233', branch(A.seqbr(20, 32, k, r, 0.15, '233'), '<233>')[:2]),
             ('2333', branch(A.seqbr(20, 22, k, r, 0.15, '2333'), '<2333>')[:2])]
    J = json.loads((S5A / 's5a_P2.json').read_text())
    g_lo = J['edges']['lo']['rich/cubic']['value']
    g_233 = J['tests']['T8']['<233>']['<23>|<233>']['value']
    e = A.edge(comps[2][1][0], comps[2][1][1], comps[3][1][0], comps[3][1][1], 'cubic')
    g_2333 = e['value']
    out['233|2333'] = (g_2333, e['se'])
    for (x0, x1), col, txt in (((g_lo, g_233), FILL, S('23')), ((g_233, g_2333), '#fdf0eb', S('233')),
                               ((g_2333, xl[1]), '#f6f5f2', S('2333'))):
        c.axvspan(x0, x1, color=col, lw=0, zorder=0)
        c.text(1.035 if txt == S('23') else 0.5 * (x0 + x1), yl[1] - 2.0, txt, fontsize=6.5, color=DARK,
               ha='center')
    c.axhline(0, color=MID, lw=0.9, zorder=1)   # the reference structure <3>
    gg = np.linspace(xl[0], xl[1], 200)
    s3 = A.series((3,), k, r, 6)
    ends = [(S('3'), xl[1], 0.0)]
    c.text(0.978, 7.5, S('2'), fontsize=7, color=INK, ha='right', va='center')
    c.text(1.205, 7.5, S('23'), fontsize=7, color=INK, ha='right', va='center')
    for name, X in comps:
        seq = tuple(int(ch) for ch in name)
        if name in ('2', '23'):
            th = 1e4 * (A.e_series(A.series(seq, k, r, 6), gg) - A.e_series(s3, gg))
            c.plot(gg, th, color=REF, lw=0.8, ls=(0, (4, 2)), zorder=2)
            th8 = 1e4 * (A.e_series(series8(seq, k, r), gg) - A.e_series(series8((3,), k, r), gg))
            c.plot(gg, th8, color=REF, lw=0.8, ls=':', zorder=2)
        g, d, se = fdiff(X, base)
        m = (g >= xl[0] - 1e-9) & (g <= xl[1] + 1e-9) & np.isfinite(d)
        pts(c, g[m], 1e4 * d[m], name, yerr=1e4 * se[m], ls='-', ms=2.8, zorder=4)
        if name in ('233', '2333'):
            j = np.nonzero(m)[0][-1]
            ends.append((S(name), g[j], 1e4 * d[j]))
        out[('P2', name)] = (g[m].round(4).tolist(), (1e4 * d[m]).round(2).tolist())
    spread(c, ends, xl[1] + 0.025, 2.6)
    c.set_xlim(xl[0], xl[1] + 0.12)
    c.set_ylim(*yl)
    c.set_xlabel('$g$')
    c.set_ylabel(r'$F_X-F_{\langle3\rangle}$ ($10^{-4}J_0$ per spin)')
    title(c, r'$\kappa=0.5625$, $r=0$')
    panel(c, '(c)')
    fig.tight_layout(w_pad=1.2)
    fig.savefig(OUT / 'fig_f23.pdf')
    fig.savefig(OUT / 'fig_f23.png')
    return out


# ---------------------------------------------------------------- Fig. 4: <23> window edges and widths
def fig_window():
    cases = {'P5': (0.52, 0), 'P1': (0.53, 0), 'P4': (0.545, 0), 'P2': (0.5625, 0),
             'P6': (0.52, 1), 'P3': (0.53, 1), 'P7': (0.545, 1)}
    J = {c: json.loads((S5A / f's5a_{c}.json').read_text()) for c in cases}
    FU = json.loads((S5A / 'posthoc_P6' / 's5a_P6_pooled36_POSTHOC.json').read_text())
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(WIDE, ROW))
    out = {}
    for ax, r, tag, yl in ((a, 0, '(a)', (-0.07, 0.21)), (b, 1, '(b)', (-0.045, 0.085))):
        kg = np.round(np.linspace(0.508, 0.568 if r == 0 else 0.55, 13), 5)
        gc6 = np.array([tcross((2,), (3,), k, r, 6) for k in kg])
        for order, sty in ((6, dict(color=REF, lw=0.9)), (4, dict(color=MUTED, lw=0.7, ls=':')),
                           (8, dict(color=REF, lw=0.8, ls=(0, (5, 1.5, 1, 1.5))))):
            lo = np.array([tcross((2,), (2, 3), k, r, order) for k in kg]) - gc6
            hi = np.array([tcross((2, 3), (3,), k, r, order) for k in kg]) - gc6
            if order == 6:
                ax.fill_between(kg, lo, np.minimum(hi, yl[1]), color=FILL, lw=0, zorder=0)
            ax.plot(kg, lo, zorder=2, **sty)
            ax.plot(kg, hi, zorder=2, **sty)
        for cs, (k, rr) in cases.items():
            if rr != r:
                continue
            g0 = tcross((2,), (3,), k, r, 6)
            E = J[cs]['edges']
            pts(ax, [k], [E['lo']['rich/cubic']['value'] - g0], '2', yerr=[E['lo']['rich/cubic']['se']], zorder=4)
            pts(ax, [k], [E['hi']['rich/cubic']['value'] - g0], '3', yerr=[E['hi']['rich/cubic']['se']], zorder=4)
            out[(cs, 'lo')] = E['lo']['rich/cubic']['value'] - g0
            out[(cs, 'hi')] = E['hi']['rich/cubic']['value'] - g0
            if cs == 'P6':
                dk = 0.0025
                pts(ax, [k + dk], [FU['edges']['lo']['rich/cubic']['value'] - g0], '2',
                    yerr=[FU['edges']['lo']['rich/cubic']['se']], hollow=True, zorder=4)
                pts(ax, [k + dk], [FU['edges']['hi']['rich/cubic']['value'] - g0], '3',
                    yerr=[FU['edges']['hi']['rich/cubic']['se']], hollow=True, zorder=4)
                label(ax, 'follow-up', (k + dk + 0.001, FU['edges']['hi']['rich/cubic']['value'] - g0),
                      (k + 0.006, 0.062), fs=6.5)
            common = SEQ / f'seqbr_Lx20_Ly60_k{k:g}_r{r}_T0.15_2-23-3.csv'
            if common.exists():                      # <2>, <23>, <3> on one 20 x 60 lattice (revision runs V1, V2, V4, V8)
                bs, kk, rr_ = A.load_file(common)
                rep = common.with_name(common.stem + '_rep.csv')
                if rep.exists():                     # independent replicate (V7, V2rep): pool the chains of both runs
                    import revision1_stats_v3 as RS
                    b2 = {b.name: b for b in A.load_file(rep)[0]}
                    bs = [RS.pool(b, b2[b.name]) for b in bs]
                B = {b.name: (b, A.free_energy(b, kk, rr_, 'rich')) for b in bs}
                elo = A.edge(*B['<2>'], *B['<23>'], 'cubic')
                ehi = A.edge(*B['<23>'], *B['<3>'], 'cubic')
                dk = -0.0025
                for e_, key in ((elo, '2'), (ehi, '3')):
                    c_, _, _, _ = STY[key]
                    ax.errorbar([k + dk], [e_['value'] - g0], yerr=[e_['se']], color=c_, marker='P', ms=5.2,
                                mec='white', mew=0.5, capsize=2, elinewidth=0.8, ls='none', zorder=5)
                out[(cs, 'common')] = (elo['value'], elo['se'], ehi['value'], ehi['se'])
                if cs == 'P3':
                    label(ax, 'common ' + r'$20\times60$', (k + dk + 0.0006, ehi['value'] - g0), (k + 0.004, 0.004), fs=6.5)
            if cs == 'P2':
                t8 = J[cs]['tests']['T8']['<233>']['<23>|<233>']
                pts(ax, [k + 0.0025], [t8['value'] - g0], '233', yerr=[t8['se']], zorder=4)
                label(ax, S('233') + ' below ' + S('23'), (k + 0.0025, t8['value'] - g0), (0.537, 0.155), fs=6.5)
        ax.axhline(0, color=MUTED, lw=0.5, ls=(0, (4, 2)), zorder=1)
        ax.set_xlim(0.505, 0.575 if r == 0 else 0.556)
        ax.set_ylim(*yl)
        ax.set_xlabel(r'$\kappa$')
        title(ax, f'$r={r}$')
        panel(ax, tag)
    a.set_ylabel(r'$g-g_{3|2}^{(6)}(\kappa)$')
    a.text(0.509, 0.185, 'upper edge ' + S('23') + r'$|$' + S('3'), fontsize=6.5, color=MID)
    a.text(0.509, -0.058, 'lower edge ' + S('2') + r'$|$' + S('23'), fontsize=6.5, color=DARK)
    a.text(0.5605, 0.022, r'$O(g^4)$', fontsize=6.5, color=MUTED)
    a.text(0.5465, 0.115, r'$O(g^8)$', fontsize=6.5, color=REF, ha='right')
    a.text(0.5585, -0.056, r'$O(g^6)$', fontsize=6.5, color=REF)

    # (c) widths w = dkappa/g_c^4 (units 1e-3), QMC slope, against O(g^6) and O(g^8)
    T8 = {q['case']: q for q in json.loads((RES / 'perturbation' / 'pt_order8_tables.json').read_text())['tab5']}
    for r, key in ((0, 'r0'), (1, 'r1')):
        ks, w, we, wp, k8, w8 = [], [], [], [], [], []
        for cs, (k, rr) in sorted(cases.items(), key=lambda t: t[1][0]):
            if rr != r:
                continue
            T3 = J[cs]['tests']['T3']
            ks.append(k)
            w.append(1e3 * T3['width'][0])
            we.append(1e3 * T3['width'][1])
            wp.append(1e3 * T3['pred_O(g^6)'])
            q8 = T8[cs]
            if q8['hi8']:
                k8.append(k)
                w8.append(1e3 * (q8['hi8'][0] - q8['lo8'][0]) * T3['slope_theory'] / J[cs]['theory']['g_c'] ** 4)
        pts(c, ks, w, key, yerr=we, ls='-', zorder=4)
        c.plot(ks, wp, color=REF, lw=0.8, ls=(0, (4, 2)), marker='_', ms=7, mew=1.0, zorder=3)
        c.plot(k8, w8, color=REF, lw=0.8, ls=':', marker='_', ms=7, mew=1.0, zorder=3)
        out[('w8', r)] = list(zip(k8, np.round(w8, 2)))
        c.annotate(f'$r={r}$', xy=(ks[-1], w[-1]), xytext=(ks[-1] + 0.004, w[-1]), fontsize=7, va='center')
        out[('w', r)] = list(zip(ks, np.round(w, 2), np.round(wp, 2)))
    c.set_yscale('log')
    c.set_ylim(0.8, 60)
    c.set_yticks([1, 3, 10, 30])
    c.set_yticklabels(['1', '3', '10', '30'])
    c.set_xlim(0.515, 0.575)
    c.set_xlabel(r'$\kappa$')
    c.set_ylabel(r'width $\Delta\kappa/g_c^4$ ($10^{-3}$)')
    c.text(0.517, 50, r'dashed: $O(g^6)$' + '\n' + r'dotted: $O(g^8)$', fontsize=6.5, color=REF, va='top')
    title(c, S('23') + ' window width')
    panel(c, '(c)')
    fig.tight_layout(w_pad=1.4)
    fig.savefig(OUT / 'fig_window.pdf')
    fig.savefig(OUT / 'fig_window.png')
    save_cache()
    return out


# ---------------------------------------------------------------- Fig. 5: annealing lock-in at kappa = 0.75
def fig_anneal():
    N = json.loads((RES / 'audit_12' / 'secVA_numbers.json').read_text())['L12']
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, ROW))
    out = {}
    for ax, r, tag in ((axes[0], 0, '(a)'),):
        d = pd.read_csv(BR / f'branches_L12_k0.75_r{r}.csv')
        d = d[d.part == 'descending'].groupby('g')[['Sl2_qpi3', 'Sl2_qpi2']].agg(['mean', 'sem'])
        g = d.index.to_numpy()
        o3, e3 = 2 * d[('Sl2_qpi3', 'mean')] / (8 / 9), 2 * d[('Sl2_qpi3', 'sem')] / (8 / 9)
        o2, e2 = 2 * d[('Sl2_qpi2', 'mean')], 2 * d[('Sl2_qpi2', 'sem')]
        gs = N[f'0.75_{r}']['best']['g']
        on = onset(0.75, r, 0.1)
        ax.axvspan(gs, on, color=FILL, lw=0, zorder=0)
        pts(ax, g, o3, '3', yerr=e3, ls='-', ms=3.2, zorder=4)
        pts(ax, g, o2, '2', yerr=e2, ls='-', ms=3.2, zorder=3)
        ax.axvline(gs, color=MUTED, lw=0.5, ls=':', zorder=1)
        ax.text(gs, 0.93, r' $g^\star=%.2f$' % gs, fontsize=6.5, color=MUTED, ha='left', va='center')
        ax.text(0.5 * (gs + on), 1.07, S('3') + ' stable', fontsize=6.5, color=DARK, ha='center')
        ax.text(0.5 * gs, 1.07, S('2') + ' stable', fontsize=6.5, color=INK, ha='center')
        xr = 3.6 if r == 0 else 4.6
        ax.text(0.80 * xr, 0.62, r'$O(\pi/3)$: ' + S('3') + ' order', fontsize=6.5, color=MID, ha='center')
        ax.text(0.05 * xr, 0.10, r'$O(\pi/2)$: ' + S('2') + ' order', fontsize=6.5, color=DARK, ha='left')
        ax.set_xlim(0, 4.6 if r else 3.6)
        ax.set_ylim(-0.05, 1.15)
        ax.set_xlabel('$g$ (annealed downward)')
        title(ax, rf'$12\times12$, $\kappa=0.75$, $r={r}$')
        panel(ax, tag)
        out[('anneal', r)] = (float(o3.iloc[0]), float(o2.iloc[0]), gs, on)
    axes[0].set_ylabel('normalized order parameter')

    # (b) the same anneal on 12 x 60: which wavevector forms and freezes (revision run A1)
    b = axes[1]
    Z = np.load(RES / 'sse_validation' / 'anneal_rect' / 'anneal_Lx12_Ly60_k0.75_r0_T0.15.npz')
    Sq, gA = Z['Sl2'], Z['g']
    for n, key, txt, dy in ((10, '3', r'$n=10$: ' + S('3'), 0.0), (11, '233', r'$n=11$', 0.0),
                            (12, '2', r'$n=12$: period 5', 0.0), (15, '2333', r'$n=15$: ' + S('2'), 0.0)):
        o = 2 * Sq[:, :, n]
        pts(b, gA, o.mean(0), key, yerr=o.std(0, ddof=1) / np.sqrt(o.shape[0]), ls='-', ms=2.8, zorder=4)
        out[('anneal60', n)] = float(o.mean(0)[-1])
    b.text(1.45, 0.63, r'$n=12$ (period 5)', fontsize=6.5, color=DARK)
    b.text(0.15, 0.10, r'$n=10$ (' + S('3') + r'), $11$, $15$ (' + S('2') + ')', fontsize=6.5, color=INK)
    b.axvline(N['0.75_0']['best']['g'], color=MUTED, lw=0.5, ls=':', zorder=1)
    b.set_xlim(0, 3.6)
    b.set_ylim(-0.05, 1.15)
    b.set_xlabel('$g$ (annealed downward)')
    b.set_ylabel(r'$2S(2\pi n/60)$')
    title(b, r'$12\times60$, $\kappa=0.75$, $r=0$')
    panel(b, '(b)')

    # (c) F<3> - F<2> from the template branches
    c = axes[2]
    for r, key in ((0, 'r0'), (1, 'r1')):
        path = BR / f'branches_L12_k0.75_r{r}.csv'
        X = branch(path, '<3>')[:2]
        Y = branch(path, '<2>')[:2]
        g, dd, se = fdiff(X, Y)
        pts(c, g, 1e3 * dd, key, yerr=1e3 * se, ls='-', ms=3.2, zorder=4)
        j = np.nonzero(np.isfinite(dd))[0][-1]
        c.annotate(f'$r={r}$', xy=(g[j], 1e3 * dd[j]), xytext=(g[j] + 0.08, 1e3 * dd[j] + 6), fontsize=7, va='center')
        out[('dF', r)] = (g[0], 1e3 * dd[0])
    c.axhline(0, color=REF, lw=0.7, zorder=1)
    c.set_xlabel('$g$')
    c.set_ylabel(r'$F_{\langle3\rangle}-F_{\langle2\rangle}$ ($10^{-3}J_0$ per spin)')
    title(c, r'template branches')
    c.set_xlim(0, 3.2)
    panel(c, '(c)')
    fig.tight_layout(w_pad=1.4)
    fig.savefig(OUT / 'fig_anneal.pdf')
    fig.savefig(OUT / 'fig_anneal.png')
    return out


# ---------------------------------------------------------------- Fig. 6 (appendix): validation
def fig_validation():
    from scipy.stats import t as tdist, norm
    V = RES / 'sse_validation'
    fig, axs = plt.subplots(2, 2, figsize=(WIDE, 2 * ROW))
    (a, b), (c, d) = axs
    out = {}

    # (a) SSE against exact diagonalization: pulls of all 471 comparisons
    ed = pd.read_csv(V / 'sse_ed_benchmark.csv')
    z = ed.z.to_numpy(float)
    bins = np.arange(-3.5, 3.51, 0.25)
    a.hist(z, bins=bins, color=LIGHT, edgecolor='white', lw=0.5, zorder=2)
    zz = np.linspace(-3.5, 3.5, 300)
    a.plot(zz, len(z) * 0.25 * tdist.pdf(zz, 23), color=REF, lw=0.9, zorder=3)
    a.text(0.03, 0.92, f'{len(z)} comparisons, 13 cases\nmedian $|z|={np.median(np.abs(z)):.2f}$ (expected 0.69)\n'
           f'{100 * np.mean(np.abs(z) < 2):.1f}% within $|z|<2$', transform=a.transAxes, fontsize=6.5, va='top')
    a.text(1.6, len(z) * 0.25 * tdist.pdf(1.3, 23), r'Student $t_{23}$', fontsize=6.5, color=REF)
    a.set_xlabel(r'$z=(\mathrm{SSE}-\mathrm{ED})/\sigma$')
    a.set_ylabel('count')
    title(a, 'SSE against exact diagonalization')
    panel(a, '(a)')
    out['ed'] = (len(z), float(np.median(np.abs(z))), float(np.mean(np.abs(z) < 2)), float(np.abs(z).max()))

    # (b) energies at g0 against the perturbation series: template branches (L = 12) and the <23> test
    zs = []
    for f in sorted(BR.glob('branches_L12_k*_r*.csv')):
        if f.name.endswith('free_energy.json'):
            continue
        bs, k, r = A.load_file(f)
        for br in bs:
            e0 = float(A.e_series(A.series(br.seq, k, r, 4), A.G0))
            E = br.E[:, 0]
            zs.append((E.mean() - e0) / (E.std(ddof=1) / np.sqrt(len(E))))
    z12 = np.array(zs)
    seen = {}
    for f in sorted(S5A.glob('s5a_[PK]*.json')):
        J = json.loads(f.read_text())
        for br in J.get('branches', {}).values():
            seen[(br['file'], br['key'])] = br['z_g0']
    z23 = np.array(list(seen.values()))
    b.hist([z12, z23], bins=np.arange(-3.5, 3.51, 0.5), stacked=True, color=[DARK, LIGHT], edgecolor='white',
           lw=0.5, zorder=2)
    n = len(z12) + len(z23)
    b.plot(zz, n * 0.5 * norm.pdf(zz), color=REF, lw=0.9, zorder=3)
    b.text(0.03, 0.92, f'{len(z12)} template branches ($12\\times12$)\n{len(z23)} branches of the '
           + S('23') + ' test', transform=b.transAxes, fontsize=6.5, va='top')
    b.text(0.03, 0.72, 'dark: templates\nlight: ' + S('23') + ' test', transform=b.transAxes, fontsize=6.5,
           va='top', color=MUTED)
    b.set_xlabel(r'$z=(e_{\rm SSE}-e_{\rm PT})/\sigma$ at $g_0=0.05$')
    b.set_ylabel('count')
    title(b, 'low-field energies against the series')
    panel(b, '(b)')
    out['anchor'] = (len(z12), float(z12.mean()), float(z12.std(ddof=1)), len(z23), float(z23.mean()),
                     float(z23.std(ddof=1)))

    # (c, d) two-dimensional transverse-field Ising model: crossings of U4 and xi_y/L for size pairs
    for ax, sub, ttl, tag, yl in ((c, 'tfim2d', r'$\kappa=r=0$, $\beta=L$', '(c)', (3.030, 3.052)),
                                  (d, 'tfim2d_T0.15', r'$\kappa=r=0$, $T=0.15\,J_0$', '(d)', (3.030, 3.060))):
        Jt = json.loads((V / sub / 'tfim2d_summary.json').read_text())
        gc = Jt['g_c_reference']
        X = [np.sqrt(cr['L1'] * cr['L2']) for cr in Jt['crossings']]
        pairs = [f"{cr['L1']}/{cr['L2']}" for cr in Jt['crossings']]
        for key, name, sty in (('U4', r'$U_4$', 'r0'), ('xiL', r'$\xi_y/L$', 'r1')):
            y = [cr[f'g_cross_{key}'] for cr in Jt['crossings']]
            e = [cr[f'g_cross_{key}_err'] for cr in Jt['crossings']]
            pts(ax, 1 / np.array(X), y, sty, yerr=e, ls='-', zorder=4)
            if sub == 'tfim2d':
                ax.annotate(name, xy=(1 / X[0], y[0]), xytext=(1 / X[0] + 0.004, y[0]), fontsize=7, va='center')
            else:                                    # the small pairs coincide; label the largest pair
                ax.annotate(name, xy=(1 / X[-1], y[-1]), xytext=(1 / X[-1] - 0.003, y[-1]), fontsize=7,
                            va='center', ha='right')
            out[(sub, key)] = list(zip(pairs, np.round(y, 4)))
        ax.axhline(gc, color=REF, lw=0.8, zorder=1)
        ax.text(0.0025, gc + 0.0004, r'$g_c=3.04438$ ($T=0$)', fontsize=6.5, color=REF, va='bottom')
        if sub == 'tfim2d_T0.15':
            ax.axhspan(yl[0], gc, color='#f1f0ed', lw=0, zorder=0)
            ax.text(0.0025, 3.0315, r'$T=0.15$ transition expected in $3.0<g<g_c$', fontsize=6.5, color=MUTED)
        ax.set_xticks(1 / np.array(X))
        ax.set_xticklabels(pairs)
        ax.set_xlim(0, 1 / X[0] + 0.03)
        ax.set_ylim(*yl)
        ax.set_xlabel(r'size pair $L_1/L_2$ (at $1/\sqrt{L_1L_2}$)')
        title(ax, ttl)
        panel(ax, tag)
    c.set_ylabel('crossing field $g$')
    fig.tight_layout(w_pad=1.6, h_pad=1.4)
    fig.savefig(OUT / 'fig_validation.pdf')
    fig.savefig(OUT / 'fig_validation.png')
    return out


if __name__ == '__main__':
    which = sys.argv[1:] or ['model', 'wedge', 'f23', 'window', 'anneal', 'validation']
    for w in which:
        res = globals()[f'fig_{w}']()
        if res:
            print(w, {str(k): v for k, v in res.items()})

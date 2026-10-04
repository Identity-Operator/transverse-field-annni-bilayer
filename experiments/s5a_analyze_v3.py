#!/usr/bin/env python3
"""Pre-registered analysis of the S5a <23> runs (manuscript/paper1_physics/s5a_prereg.md, "Analysis", T1-T8).

Written independently of the run design (by phase-classification-12). Input: the CSVs of
experiments/sse_seq_branches_v3.py, reanalysis_v3/sse_validation/seq_branches/
seqbr_Lx<Lx>_Ly<Ly>_k<kappa>_r<r>_T<T>_<seqs>.csv; with --files, any branch CSVs of that format (step-2
branch files and the exploratory probes included; only part = ascending is used).

Free energy of each chain (J0 = 1, g = Gamma):   f(g) = e_PT(g0) - int_{g0}^{g} m_x dg',   g0 = 0.05,
with e_PT the O(g^4) energy of the template (pt_order4_v3.per_column / coeffs). A chain is cut (NaN) from
its first change of q* on.

Quadrature ("rich", primary): the coarse segment runs in 0.1 steps from g = 0.1 to g_f, the last 0.1-grid
point before the fine band (the end of the first 0.1-step piece of the --grid spec, read from the run's JSON
sidecar; without one, the end of the run of 0.1 steps from 0.1), with m intervals. m even: Richardson I_h + (I_h - I_2h)/3 (composite
Simpson) over the whole segment; m odd: Richardson over [0.1, g_f - 0.3] plus Simpson 3/8 over
[g_f - 0.3, g_f] (m = 1: trapezoid). Trapezoid on 0.05-0.1, on the step into the fine band, on the fine
band and above it. Cumulative values inside a panel use the panel's interpolating polynomial, so every grid
point has a value and the panel sums are exact. "trap": the plain trapezoid everywhere (reported).
"unanch": rich with f(g0) = measured E(g0) per chain (reported).

Statistics: branch free energies are chain means. Differences at one g: Welch t with Welch-Satterthwaite
degrees of freedom (the jackknife SE of a mean is sd/sqrt(n)). Crossings (edges): the first + -> - sign
change of F_new - F_old on the common grid, root of a cubic spline through the difference (primary) or linear
in u = g^2 (reported); delete-one jackknife SE over the chains of both branches; 68% CI with t_{n-1}.

Tests (per case P1-P7; kappa, r from the pre-registration):
  T1  at g_c (the O(g^6) exact-kappa <3>|<2> crossing, an exact grid point): F<23> - F<2> on 20x20 and
      F<23> - F<3> on 20x30 both < 0, each one-sided Welch p < 0.02275 (2 sigma), and no chain of the four
      branches cut at g_c. Reported with the pre-registered expected z.
  T2  edges <2>|<23> (20x20) and <23>|<3> (20x30) with 68% CIs, against exact-kappa O(g^4) and O(g^6).
  T3  Delta_kappa/g_c^4 = (g_hi - g_lo) dkappa*/dg / g_c^4, dkappa*/dg from a fit
      kappa - 1/2 = a g*^2 + b g*^4 to the QMC <3>|<2> crossings of the cases at the same r (F<3> on 20x30
      against F<2> on 20x20), weighted with sigma_kappa = sigma_g dkappa*/dg(theory). Also with the theory
      slope. Compared with the O(g^6) value (same conversion).
  T4  T3 ratio r = 0 / r = 1 for P1/P3, P4/P7, P5/P6 against the O(g^6) ratios.
  T5  at g_c: F<223> - F<23>, F<233> - F<23> (<23> on 20x30; 20x20 also reported), pulls against O(g^6);
      stop rule if a competitor is below <23> at g_c with one-sided Welch p < 0.00135 (3 sigma).
  T6  <23> on 20x20 vs 20x30 at every common grid point; <3> on 20x30 vs step-2 L = 24 (P1, P3); <23> and
      <2> on 40x40 vs 20x20 (K1, if present): z per point, chi^2 per point, max |z|.
  T7  T = 0.10 edges (K2) against T = 0.15 (P1).
  T8  (P2, P4) crossings <23>|<233>, <233>|<3> (and <2333> at P2), and the grid points g >= g_c where <233>
      (<2333>) is below both <23> and <3> at > 2 sigma (Welch); the t against <2> is reported too.
Diagnostics: measured-vs-PT z at g0 per branch; chains cut before g_c; the pre-registered theory numbers are
recomputed and compared with the table.

Output: <out>/s5a_<case>.json and <out>/s5a_<label>.txt (the printed tables).

Usage (from vu_work/):
    python experiments/s5a_analyze_v3.py                          # every S5a case whose CSVs exist
    python experiments/s5a_analyze_v3.py --cases P1,P3
    python experiments/s5a_analyze_v3.py --selftest               # noise-free method check on the S5a grids
    python experiments/s5a_analyze_v3.py --label probe --kappa 0.53 --rperp 0 --out <dir> --files a.csv b.csv
With --files the roles are assigned automatically: <2> and the <23> on its lattice, <3> and the <23> on its
lattice (if none, the other <23>, noted as cross-lattice), competitors; further copies of a structure on other
lattices are T6 controls. Without <23> only the pairwise edges are computed.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
import re
from fractions import Fraction as Fr

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.stats import norm, t as tdist

from pt_order4_v3 import coeffs, per_column
from pt_order6_v3 import crossing_g, per_spin_series

BASE = Path("10_results/bilayer_annni_paper_results/reanalysis_v3")
SEQDIR = BASE / "sse_validation" / "seq_branches"
STEP2 = BASE / "sse_validation" / "branches"
OUT = BASE / "s5a_analysis"
G0 = 0.05
P2SIG, P3SIG = float(norm.sf(2)), float(norm.sf(3))

CASES = {"P1": (0.53, 0), "P3": (0.53, 1), "P2": (0.5625, 0), "P4": (0.545, 0), "P7": (0.545, 1),
         "P5": (0.52, 0), "P6": (0.52, 1)}
# pre-registration table (s5a_prereg.md), recomputed and compared at run time
PREREG = {"P5": (0.5997, 0.5923, 0.6123, -2.17e-4, 6.0e-5, 8.1e-5, 0.0114, "~3.5"),
          "P1": (0.7179, 0.7049, 0.7418, -4.98e-4, 1.37e-4, 1.87e-4, 0.0134, "~6.5"),
          "P4": (0.8525, 0.8302, 0.8990, -1.14e-3, 3.12e-4, 4.29e-4, 0.0166, ">=4"),
          "P2": (0.9733, 0.9396, 1.0561, -2.22e-3, 6.00e-4, 8.34e-4, 0.0212, ">=4"),
          "P6": (0.9351, 0.9254, 0.9512, -1.75e-4, 4.9e-5, 6.6e-5, 0.00158, "~3.5"),
          "P3": (1.1222, 1.1051, 1.1525, -3.97e-4, 1.09e-4, 1.49e-4, 0.00181, ">=4"),
          "P7": (1.3369, 1.3070, 1.3951, -8.98e-4, 2.44e-4, 3.38e-4, 0.00220, ">=4")}
T4_PAIRS = [("P1", "P3", 7.38), ("P4", "P7", 7.55), ("P5", "P6", 7.22)]
COMPS = {"P1": ["233", "223"], "P2": ["233", "223", "2333"], "P3": ["233", "223"], "P4": ["233"]}
COMP_LY = {"233": 32, "223": 28, "2333": 22}
STEP2_L24 = {"P1": "branches_L24_k0.53_r0.csv", "P3": "branches_L24_k0.53_r1.csv"}
METHODS = ["rich", "trap", "unanch"]


# ---------------------------------------------------------------- input
def seqbr(Lx, Ly, kappa, r, T, seqs, d=SEQDIR):
    return Path(d) / f"seqbr_Lx{Lx}_Ly{Ly}_k{kappa:g}_r{float(r):g}_T{T:g}_{seqs}.csv"


def parse_seq(start):
    """'<23>par' -> (2, 3); 'FM' -> None."""
    if start.startswith("FM"):
        return None
    return tuple(int(c) for c in re.match(r"<(\d+)>", start).group(1))


def sname(seq):
    return "FM" if seq is None else "<" + "".join(map(str, seq)) + ">"


class Branch:
    def __init__(self, df, start, Lx, Ly, T, src, g_f=None):
        d = df[df.start == start]
        self.g_f = g_f
        self.seq = parse_seq(start)
        self.name, self.Lx, self.Ly, self.T, self.src = sname(self.seq), Lx, Ly, T, src
        piv = lambda c: d.pivot_table(index="chain", columns="g", values=c, aggfunc="first")
        mx = piv("mx")
        self.g = np.round(mx.columns.to_numpy(float), 4)
        self.mx = mx.to_numpy(float)
        self.E = piv("E_per_spin").reindex(columns=mx.columns).to_numpy(float)
        self.q = piv("qstar_over_pi").reindex(columns=mx.columns).to_numpy(float)
        self.n = self.mx.shape[0]
        if not np.isclose(self.g[0], G0):
            raise ValueError(f"{src} {start}: grid does not start at g0 = {G0}")
        if np.isnan(self.mx).any():
            raise ValueError(f"{src} {start}: incomplete chains (missing grid points)")
        self.lat = f"{Lx}x{Ly}" + ("" if abs(T - 0.15) < 1e-9 else f"/T{T:g}")
        self.key = f"{self.name}@{self.lat}"
        self.cut_idx = np.array([np.nonzero(np.abs(self.q[i] - self.q[i, 0]) > 1e-9)[0][0]
                                 if np.any(np.abs(self.q[i] - self.q[i, 0]) > 1e-9) else len(self.g) for i in range(self.n)])


def load_file(path):
    path = Path(path)
    df = pd.read_csv(path)
    if "part" in df.columns:
        df = df[df.part == "ascending"]
    if "Lx" in df.columns:
        Lx, Ly = int(df.Lx.iloc[0]), int(df.Ly.iloc[0])
    elif "L" in df.columns:
        Lx = Ly = int(df.L.iloc[0])
    else:
        m = re.search(r"_Lx(\d+)_Ly(\d+)_", path.name)
        Lx, Ly = (int(m.group(1)), int(m.group(2))) if m else (int(re.search(r"_L(\d+)_", path.name).group(1)),) * 2
    T = float(df["T"].iloc[0]) if "T" in df.columns else 0.15
    kappa, r = float(df.kappa.iloc[0]), abs(float(df.rperp.iloc[0]))
    g_f = None
    side = path.with_suffix(".json")
    if side.exists():
        spec = json.loads(side.read_text()).get("args", {}).get("grid")
        if spec:
            a_, b_, st = (float(x) for x in spec.split(",")[0].split(":"))
            if abs(st - 0.1) < 1e-9 and abs(a_ - 0.1) < 1e-9:
                g_f = b_
    return [Branch(df, s, Lx, Ly, T, path.name, g_f) for s in df.start.unique()], kappa, r


# ---------------------------------------------------------------- theory
_cache = {}


def series(seq, kappa, r, order):
    """Per-spin (e0, e2, e4[, e6]) as floats; O(g^4) from pt_order4, O(g^6) from pt_order6."""
    key = (seq, kappa, r, order)
    if key not in _cache:
        K, R = Fr(str(kappa)), Fr(str(r))
        if order == 4:
            c = coeffs("FM", K, R)[:3] if seq is None else per_column(list(seq), K, R)[:3]
        else:
            c = per_spin_series(None if seq is None else list(seq), K, R, 6)
        _cache[key] = [float(x) for x in c]
    return _cache[key]


def e_series(c, g):
    g = np.asarray(g, float)
    return sum(ck * g ** (2 * k) for k, ck in enumerate(c))


def pt_cross(old, new, kappa, r, order):
    """g at which `new` drops below `old` (exact kappa), from the series in u = g^2."""
    return crossing_g(series(new, kappa, r, 6)[:order // 2 + 1], series(old, kappa, r, 6)[:order // 2 + 1], order)


def pt_diff(a, b, kappa, r, g, order=6):
    return float(e_series(series(a, kappa, r, 6)[:order // 2 + 1], g) - e_series(series(b, kappa, r, 6)[:order // 2 + 1], g))


def slope_theory(kappa, r, g, eps=Fr(1, 10000)):
    """dkappa*/dg of the O(g^6) <3>|<2> line at (g, kappa), by implicit differentiation."""
    K, R = Fr(str(kappa)), Fr(str(r))
    diff = lambda k: [float(a - b) for a, b in zip(per_spin_series([3], k, R, 6), per_spin_series([2], k, R, 6))]
    d0, dp, dm = diff(K), diff(K + eps), diff(K - eps)
    d_dg = sum(2 * k * c * g ** (2 * k - 1) for k, c in enumerate(d0) if k)
    return -d_dg / ((e_series(dp, g) - e_series(dm, g)) / (2 * float(eps)))


def theory(kappa, r):
    gc = pt_cross((2,), (3,), kappa, r, 6)
    th = {"g_c": gc, "lambda": gc / (2 + r),
          "g_32": {f"O(g^{o})": pt_cross((2,), (3,), kappa, r, o) for o in (2, 4, 6)},
          "lo": {f"O(g^{o})": pt_cross((2,), (2, 3), kappa, r, o) for o in (4, 6)},
          "hi": {f"O(g^{o})": pt_cross((2, 3), (3,), kappa, r, o) for o in (4, 6)},
          "Delta_gc": {f"O(g^{o})": pt_diff((2, 3), (2,), kappa, r, gc, o) for o in (4, 6)},
          "comp_minus_23_gc": {c: pt_diff(tuple(int(x) for x in c), (2, 3), kappa, r, gc) for c in ("223", "233", "2333")},
          "slope": slope_theory(kappa, r, gc)}
    for o in (4, 6):
        th[f"width_O(g^{o})"] = (th["hi"][f"O(g^{o})"] - th["lo"][f"O(g^{o})"]) * th["slope"] / gc ** 4
    return th


# ---------------------------------------------------------------- free energies
def coarse_segment(g, g_f=None):
    """(i0, i_f): indices of g = 0.1 and of g_f (given, or the end of the run of 0.1 steps from 0.1)."""
    i0 = np.nonzero(np.isclose(g, 0.1))[0]
    if not len(i0):
        return None
    i0 = j = int(i0[0])
    while j + 1 < len(g) and abs(g[j + 1] - g[j] - 0.1) < 1e-6 and (g_f is None or g[j] < g_f - 1e-9):
        j += 1
    if g_f is not None and abs(g[j] - g_f) > 1e-9:
        raise ValueError(f"g_f = {g_f} is not the end of a 0.1-step run from 0.1")
    return i0, j


def cumint(g, Y, method, g_f=None):
    """Cumulative integral of the rows of Y over g (see the module docstring)."""
    h = np.diff(g)
    inc = 0.5 * (Y[:, 1:] + Y[:, :-1]) * h
    seg = coarse_segment(g, g_f) if method == "rich" else None
    if seg:
        a, b = seg
        m = b - a
        n_simp = m if m % 2 == 0 else (m - 3 if m >= 3 else 0)
        for p in range(a, a + n_simp, 2):                       # Richardson = composite Simpson, per half panel
            y0, y1, y2, hh = Y[:, p], Y[:, p + 1], Y[:, p + 2], h[p]
            inc[:, p] = hh / 12 * (5 * y0 + 8 * y1 - y2)
            inc[:, p + 1] = hh / 12 * (-y0 + 8 * y1 + 5 * y2)
        if m % 2 == 1 and m >= 3:                               # Simpson 3/8 on the last three intervals
            p = a + n_simp
            y0, y1, y2, y3, hh = Y[:, p], Y[:, p + 1], Y[:, p + 2], Y[:, p + 3], h[p]
            inc[:, p] = hh / 24 * (9 * y0 + 19 * y1 - 5 * y2 + y3)
            inc[:, p + 1] = hh / 24 * (-y0 + 13 * y1 + 13 * y2 - y3)
            inc[:, p + 2] = hh / 24 * (y0 - 5 * y1 + 19 * y2 + 9 * y3)
    return np.concatenate([np.zeros((Y.shape[0], 1)), np.cumsum(inc, 1)], 1)


def free_energy(b, kappa, r, method):
    quad = "trap" if method == "trap" else "rich"
    f0 = b.E[:, :1] if method == "unanch" else e_series(series(b.seq, kappa, r, 4), G0)
    F = f0 - cumint(b.g, b.mx, quad, b.g_f)
    for i, c in enumerate(b.cut_idx):
        if c < len(b.g):
            # a cut chain is integrated over its valid prefix only, so that no panel reaches past the cut
            # (fix of 2026-09-29, 12:2x: Richardson/3-8 panels read m_x up to 0.3 ahead; no effect on uncut chains)
            gp = b.g[:c]
            g_f = b.g_f if b.g_f is not None and b.g_f <= gp[-1] + 1e-9 else None
            f0i = f0[i:i + 1] if np.ndim(f0) else f0
            F[i, :c] = (f0i - cumint(gp, b.mx[i:i + 1, :c], quad, g_f))[0]
            F[i, c:] = np.nan
    return F


def align(bA, FA, bB, FB):
    grid = np.array(sorted(set(bA.g) & set(bB.g)))
    return grid, FA[:, np.searchsorted(bA.g, grid)], FB[:, np.searchsorted(bB.g, grid)]


# ---------------------------------------------------------------- statistics
def crossing(g, d, how="cubic"):
    """First + -> - sign change of d on the finite points. Returns (g*, number of sign changes)."""
    ok = np.isfinite(d)
    g, d = g[ok], d[ok]
    sc = np.nonzero(np.sign(d[1:]) != np.sign(d[:-1]))[0]
    down = [i for i in sc if d[i] > 0 >= d[i + 1]]
    if not down:
        return np.nan, len(sc)
    i = down[0]
    if how == "u":
        u0, u1 = g[i] ** 2, g[i + 1] ** 2
        return float(np.sqrt(u0 + d[i] * (u1 - u0) / (d[i] - d[i + 1]))), len(sc)
    sp = CubicSpline(g, d)
    return float(brentq(sp, g[i], g[i + 1])), len(sc)


def jackknife(stat, mats):
    """stat(list of branch-mean vectors) -> float; delete-one jackknife over the chains of every branch."""
    means = [np.nanmean(M, 0) for M in mats]
    full = stat(means)
    var, nbad = 0.0, 0
    for b, M in enumerate(mats):
        th = []
        for i in range(M.shape[0]):
            mm = list(means)
            mm[b] = np.nanmean(np.delete(M, i, 0), 0)
            th.append(stat(mm))
        th = np.array(th, float)
        nbad += int(np.sum(~np.isfinite(th)))
        th = th[np.isfinite(th)]
        n = M.shape[0]
        var += (n - 1) / n * np.sum((th - th.mean()) ** 2) if len(th) > 1 else np.nan
    return float(full), float(np.sqrt(var)), nbad


def welch(a, b):
    """Difference of means of the finite entries of a and b: (d, se, dof, t, one-sided p for d < 0)."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    sa, sb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    d, se = float(a.mean() - b.mean()), float(np.sqrt(sa + sb))
    dof = float((sa + sb) ** 2 / (sa ** 2 / (len(a) - 1) + sb ** 2 / (len(b) - 1)))
    return {"d": d, "se": se, "dof": dof, "t": d / se, "p_below": float(tdist.cdf(d / se, dof))}


def t68(df):
    return float(tdist.ppf(0.5 + 0.6827 / 2, df))


def edge(bOld, FOld, bNew, FNew, how):
    """Crossing where New drops below Old, with jackknife SE and 68% CI (t_{n-1})."""
    grid, A, B = align(bOld, FOld, bNew, FNew)
    val, se, nbad = jackknife(lambda mm: crossing(grid, mm[1] - mm[0], how)[0], [A, B])
    nsc = crossing(grid, np.nanmean(B, 0) - np.nanmean(A, 0), how)[1]
    df = min(bOld.n, bNew.n) - 1
    return {"value": val, "se": se, "ci68": [val - t68(df) * se, val + t68(df) * se], "df": df,
            "n_sign_changes": nsc, "jk_nan": nbad}


def at_g(b, F, g):
    j = np.nonzero(np.abs(b.g - g) < 6e-5)[0]
    return (F[:, j[0]], float(b.g[j[0]])) if len(j) else (None, None)


def size_check(A, FA, B, FB):
    grid, a, b = align(A, FA, B, FB)
    m, se = (lambda X: np.nanmean(X, 0)), (lambda X: np.nanstd(X, 0, ddof=1) / np.sqrt(np.sum(np.isfinite(X), 0)))
    d = m(a) - m(b)
    z = d / np.hypot(se(a), se(b))
    ok = np.isfinite(z) & (grid > G0 + 1e-9)
    return {"a": A.key, "b": B.key, "n_points": int(ok.sum()), "max_abs_diff": float(np.max(np.abs(d[ok]))),
            "max_abs_z": float(np.max(np.abs(z[ok]))), "chi2_per_point": float(np.mean(z[ok] ** 2)),
            "n_abs_z_gt2": int(np.sum(np.abs(z[ok]) > 2)), "z": {f"{x:g}": float(v) for x, v in zip(grid[ok], z[ok])}}


# ---------------------------------------------------------------- one case
def analyze_case(label, kappa, r, roles, controls, notes):
    """roles: '23a' (<23> with <2>), '2', '23b' (<23> with <3>), '3', and competitors '<233>' etc. (Branch).
    controls: list of (role, Branch) on other lattices (T6)."""
    R = {"case": label, "kappa": kappa, "r": r, "notes": list(notes), "branches": {}, "edges": {}, "tests": {}}
    F = {role: {m: free_energy(b, kappa, r, m) for m in METHODS} for role, b in roles.items()}
    for role, b in list(roles.items()) + [(f"control {b.key}", b) for rl, b in controls]:
        e0 = float(e_series(series(b.seq, kappa, r, 4), G0))
        R["branches"][role] = {"key": b.key, "file": b.src, "chains": b.n, "e_PT_g0": e0, "E_meas_g0": float(b.E[:, 0].mean()),
                               "z_g0": float((b.E[:, 0].mean() - e0) / (b.E[:, 0].std(ddof=1) / np.sqrt(b.n))),
                               "q0": sorted(set(np.round(b.q[:, 0], 4))),
                               "cut_at": [float(b.g[c]) if c < len(b.g) else None for c in b.cut_idx]}
    full = all(k in roles for k in ("23a", "2", "23b", "3"))
    if not full:                                             # validation path: pairwise edges only
        names = [k for k in ("FM", "2", "3", "23a") if k in roles]
        for i, a in enumerate(names):
            for b_ in names[i + 1:]:
                for old, new in ((a, b_), (b_, a)):
                    e = {f"{m}/{how}": edge(roles[old], F[old][m], roles[new], F[new][m], how)
                         for m in METHODS for how in ("cubic", "u") if m == "rich" or how == "cubic"}
                    if np.isfinite(e["rich/cubic"]["value"]):
                        sq = lambda k: roles[k].seq
                        e["pred"] = {f"O(g^{o})": pt_cross(sq(old), sq(new), kappa, r, o) for o in (4, 6)}
                        R["edges"][f"{roles[old].name}|{roles[new].name}"] = e
        return R
    th = theory(kappa, r)
    R["theory"] = th
    if label in PREREG:
        p = PREREG[label]
        mine = (th["g_c"], th["lo"]["O(g^6)"], th["hi"]["O(g^6)"], th["Delta_gc"]["O(g^6)"],
                th["comp_minus_23_gc"]["223"], th["comp_minus_23_gc"]["233"], th["width_O(g^6)"])
        bad = [(i, a, b) for i, (a, b) in enumerate(zip(p[:7], mine)) if abs(a - b) > 0.006 * abs(a) + 6e-5 * (i < 3)]
        R["prereg_check"] = {"table": p[:7], "recomputed": mine, "mismatch": bad, "expected_z": p[7]}
    gc = th["g_c"]
    # edges (T2), and the QMC <3>|<2> crossing across the two lattices (for T3)
    for tag, old, new in (("lo", "2", "23a"), ("hi", "23b", "3"), ("32", "2", "3")):
        e = {f"{m}/{how}": edge(roles[old], F[old][m], roles[new], F[new][m], how)
             for m in METHODS for how in ("cubic", "u") if m == "rich" or how == "cubic"}
        pr = th["g_32"] if tag == "32" else th[tag]
        e["pred"] = {k: v for k, v in pr.items() if k != "O(g^2)"}
        e["z_vs_pred"] = {k: (e["rich/cubic"]["value"] - v) / e["rich/cubic"]["se"] for k, v in e["pred"].items()}
        R["edges"][tag] = e
    # T1
    t1 = {"g_c": gc, "expected_z": PREREG.get(label, (None,) * 8)[7]}
    cut = {rl: int(np.sum([c < len(roles[rl].g) and roles[rl].g[c] <= gc + 1e-9 for c in roles[rl].cut_idx]))
           for rl in ("23a", "2", "23b", "3")}
    t1["chains_cut_at_or_before_g_c"] = cut
    for m in METHODS:
        row = {}
        for tag, a, b in (("A_23_minus_2", "23a", "2"), ("B_23_minus_3", "23b", "3")):
            fa, ga = at_g(roles[a], F[a][m], gc)
            fb, gb = at_g(roles[b], F[b][m], gc)
            if fa is None or fb is None:
                # g_c not on the grid (validation data): nearest common grid point
                grid, A_, B_ = align(roles[a], F[a][m], roles[b], F[b][m])
                j = int(np.argmin(np.abs(grid - gc)))
                fa, fb, ga = A_[:, j], B_[:, j], float(grid[j])
                if m == "rich":
                    R["notes"].append(f"T1 {tag}: g_c = {gc:.4f} is not on the grid; used g = {ga:g}")
            w = welch(fa, fb)
            w["g"] = ga
            w["pass"] = bool(w["d"] < 0 and w["p_below"] < P2SIG)
            row[tag] = w
        row["pass"] = bool(row["A_23_minus_2"]["pass"] and row["B_23_minus_3"]["pass"] and not any(cut.values()))
        t1[m] = row
    t1["pred_Delta"] = th["Delta_gc"]
    R["tests"]["T1"] = t1
    # T5 and T8: competitors
    t5, t8 = {}, {}
    for c in ("<223>", "<233>", "<2333>"):
        if c not in roles:
            continue
        pred = th["comp_minus_23_gc"][c[1:-1]]
        row = {}
        for ref in ("23b", "23a"):
            fc, _ = at_g(roles[c], F[c]["rich"], gc)
            fr, _ = at_g(roles[ref], F[ref]["rich"], gc)
            if fc is None or fr is None:
                grid, A_, B_ = align(roles[c], F[c]["rich"], roles[ref], F[ref]["rich"])
                j = int(np.argmin(np.abs(grid - gc)))
                fc, fr = A_[:, j], B_[:, j]
            w = welch(fc, fr)
            w.update({"ref": roles[ref].key, "pred": pred, "pull": (w["d"] - pred) / w["se"],
                      "stop": bool(w["d"] < 0 and w["p_below"] < P3SIG)})
            row[ref] = w
        t5[c] = row
        if c == "<223>":
            continue
        # T8 (<233>, <2333>): crossings above the window, and the grid points g >= g_c where c is below both <23>
        # and <3> (20x30) at > 2 sigma (Welch); whether it is also below <2> (20x20) is reported
        e1 = edge(roles["23b"], F["23b"]["rich"], roles[c], F[c]["rich"], "cubic")
        e2 = edge(roles[c], F[c]["rich"], roles["3"], F["3"]["rich"], "cubic")
        grid, Fc, F23 = align(roles[c], F[c]["rich"], roles["23b"], F["23b"]["rich"])
        _, Fc3, F3 = align(roles[c], F[c]["rich"], roles["3"], F["3"]["rich"])
        _, Fc2, F2 = align(roles[c], F[c]["rich"], roles["2"], F["2"]["rich"])
        both = []
        for j, x in enumerate(grid):
            if x < gc - 1e-9:
                continue
            w1, w2, w3 = welch(Fc[:, j], F23[:, j]), welch(Fc3[:, j], F3[:, j]), welch(Fc2[:, j], F2[:, j])
            if w1["d"] < 0 and w2["d"] < 0 and w1["p_below"] < P2SIG and w2["p_below"] < P2SIG:
                both.append({"g": float(x), "t_vs_23": w1["t"], "t_vs_3": w2["t"], "t_vs_2": w3["t"]})
        t8[c] = {f"<23>|{c}": e1, f"{c}|<3>": e2, "beats_23_and_3_2sigma": both}
    if t5:
        R["tests"]["T5"] = t5
        R["tests"]["T5_stop"] = any(v["23b"]["stop"] for v in t5.values())
        R["tests"]["T8"] = t8
    # T6
    t6 = [size_check(roles["23a"], F["23a"]["rich"], roles["23b"], F["23b"]["rich"])] if roles["23a"] is not roles["23b"] else []
    for rl, b in controls:
        t6.append(size_check(roles[rl], F[rl]["rich"], b, free_energy(b, kappa, r, "rich")))
    R["tests"]["T6"] = t6
    return R


# ---------------------------------------------------------------- T3, T4 across cases
def ladder_slope(results, r):
    """Fit kappa - 1/2 = a g^2 + b g^4 to the QMC <3>|<2> crossings at this r; returns f(g) -> (slope, se)."""
    pts = [(R["kappa"], R["edges"]["32"]["rich/cubic"]["value"], R["edges"]["32"]["rich/cubic"]["se"], R["theory"]["slope"])
           for R in results if R["r"] == r and "32" in R["edges"] and np.isfinite(R["edges"]["32"]["rich/cubic"]["value"])]
    if len(pts) < 2:
        return None, len(pts)
    k, g, sg, sl = (np.array(x) for x in zip(*pts))
    X = np.vstack([g ** 2, g ** 4]).T
    W = 1 / (sg * sl) ** 2
    cov = np.linalg.inv(X.T @ (X * W[:, None]))
    c = cov @ (X.T @ (W * (k - 0.5)))
    J = lambda x: np.array([2 * x, 4 * x ** 3])
    return (lambda x: (float(J(x) @ c), float(np.sqrt(J(x) @ cov @ J(x))))), len(pts)


def t3_t4(results):
    for r in (0, 1):
        f, n = ladder_slope([R for R in results if "theory" in R and R["case"] != "K2"], r)
        for R in results:
            if R["r"] != r or "theory" not in R:
                continue
            lo, hi = R["edges"]["lo"]["rich/cubic"], R["edges"]["hi"]["rich/cubic"]
            gc, dg = R["theory"]["g_c"], hi["value"] - lo["value"]
            sdg = np.hypot(lo["se"], hi["se"])
            t3 = {"dg": dg, "dg_se": sdg, "n_ladder": n, "pred_O(g^6)": R["theory"]["width_O(g^6)"],
                  "pred_O(g^4)": R["theory"]["width_O(g^4)"], "slope_theory": R["theory"]["slope"]}
            st = R["theory"]["slope"]
            t3["width_theory_slope"] = [dg * st / gc ** 4, sdg * st / gc ** 4]
            if f:
                s, ss = f(gc)
                t3["slope_qmc"] = [s, ss]
                t3["width"] = [dg * s / gc ** 4, float(np.hypot(sdg * s, dg * ss)) / gc ** 4]
            R["tests"]["T3"] = t3
    by = {R["case"]: R for R in results}
    t4 = []
    for a, b, pred in T4_PAIRS:
        if a in by and b in by and "width" in by[a]["tests"].get("T3", {}) and "width" in by[b]["tests"].get("T3", {}):
            (wa, sa), (wb, sb) = by[a]["tests"]["T3"]["width"], by[b]["tests"]["T3"]["width"]
            t4.append({"pair": f"{a}/{b}", "ratio": wa / wb, "se": abs(wa / wb) * np.hypot(sa / wa, sb / wb), "pred_prereg": pred,
                       "pred_recomputed": by[a]["theory"]["width_O(g^6)"] / by[b]["theory"]["width_O(g^6)"]})
    return t4


# ---------------------------------------------------------------- drivers
def s5a_case(case, d):
    kappa, r = CASES[case]
    fa, fb = seqbr(20, 20, kappa, r, 0.15, "23-2", d), seqbr(20, 30, kappa, r, 0.15, "23-3", d)
    if not (fa.exists() and fb.exists()):
        return None, [f"{case}: missing {', '.join(f.name for f in (fa, fb) if not f.exists())}"]
    A = {b.name: b for b in load_file(fa)[0]}
    B = {b.name: b for b in load_file(fb)[0]}
    roles = {"23a": A["<23>"], "2": A["<2>"], "23b": B["<23>"], "3": B["<3>"]}
    notes, controls = [], []
    for c in COMPS.get(case, []):
        f = seqbr(20, COMP_LY[c], kappa, r, 0.15, c, d)
        if f.exists():
            roles[f"<{c}>"] = load_file(f)[0][0]
        else:
            notes.append(f"competitor <{c}> not available ({f.name})")
    if case in STEP2_L24 and (STEP2 / STEP2_L24[case]).exists():
        controls.append(("3", {b.name: b for b in load_file(STEP2 / STEP2_L24[case])[0]}["<3>"]))
    if case == "P1":
        fk = seqbr(40, 40, kappa, r, 0.15, "23-2", d)
        if fk.exists():
            K = {b.name: b for b in load_file(fk)[0]}
            controls += [("23a", K["<23>"]), ("2", K["<2>"])]
    R = analyze_case(case, kappa, r, roles, controls, notes)
    extra = []
    if case == "P1":                                         # T7 from K2 (T = 0.10)
        ka, kb = seqbr(20, 20, kappa, r, 0.1, "23-2", d), seqbr(20, 30, kappa, r, 0.1, "23-3", d)
        if ka.exists() and kb.exists():
            A2 = {b.name: b for b in load_file(ka)[0]}
            B2 = {b.name: b for b in load_file(kb)[0]}
            RK = analyze_case("K2", kappa, r, {"23a": A2["<23>"], "2": A2["<2>"], "23b": B2["<23>"], "3": B2["<3>"]}, [], [])
            t7 = {}
            for tag in ("lo", "hi"):
                a, b = R["edges"][tag]["rich/cubic"], RK["edges"][tag]["rich/cubic"]
                z = (b["value"] - a["value"]) / np.hypot(a["se"], b["se"])
                t7[tag] = {"T0.15": a["value"], "T0.10": b["value"], "z": z, "pass": bool(abs(z) < 2)}
            R["tests"]["T7"] = t7
            extra.append(RK)
    return [R] + extra, []


def custom_case(label, files, kappa_arg, r_arg):
    allb, kappa, r = [], None, None
    for f in files:
        bs, k, rr = load_file(f)
        if kappa is not None and (abs(k - kappa) > 1e-12 or abs(rr - r) > 1e-12):
            raise SystemExit(f"{f}: kappa/r differ from the first file")
        kappa, r = k, rr
        allb += bs
    if kappa_arg is not None and (abs(kappa - kappa_arg) > 1e-12 or abs(r - abs(r_arg)) > 1e-12):
        raise SystemExit("kappa/r of the files differ from --kappa/--rperp")
    first = lambda nm, lat=None: next((b for b in allb if b.name == nm and (lat is None or b.lat == lat)), None)
    roles, notes = {}, []
    if first("<2>"):
        roles["2"] = first("<2>")
    if first("<3>"):
        roles["3"] = first("<3>")
    if first("FM"):
        roles["FM"] = first("FM")
    if first("<23>"):
        roles["23a"] = first("<23>", roles["2"].lat if "2" in roles else None) or first("<23>")
        if "3" in roles:
            roles["23b"] = first("<23>", roles["3"].lat)
            if roles["23b"] is None:
                roles["23b"] = roles["23a"]
                notes.append(f"<23> and <3> on different lattices ({roles['23a'].lat}, {roles['3'].lat}): cross-lattice T1-B and hi edge")
    for c in ("<223>", "<233>", "<2333>"):
        if first(c):
            roles[c] = first(c)
    used = {id(b) for b in roles.values()}
    rolemap = {"<2>": "2", "<3>": "3", "<23>": "23a", "FM": "FM"}
    controls = [(rolemap[b.name], b) for b in allb if id(b) not in used and b.name in rolemap and rolemap[b.name] in roles]
    if "23a" in roles and "3" in roles and "2" in roles:
        return analyze_case(label, kappa, r, roles, controls, notes)
    return analyze_case(label, kappa, r, {k: v for k, v in roles.items() if k in ("FM", "2", "3", "23a")}, [], notes)


def fmt(e):
    if not e or not np.isfinite(e["value"]):
        return "-"
    return f"{e['value']:.4f}({e['se'] * 1e4:.0f})"


def report(results, t4):
    L = []
    L.append(f"{'case':5s} {'k':>6s} {'r':>1s} {'g_c':>6s} {'lam':>4s} | T1 z(23-2) z(23-3) [exp] pass | "
             f"{'lo edge (O4,O6)':>30s} | {'hi edge (O4,O6)':>30s} | T3 x1e3 qmc-slope / th-slope [pred O6] | cuts<=g_c")
    for R in results:
        if "theory" not in R:
            ed = "; ".join(f"{k}: {fmt(v['rich/cubic'])} [O4 {v['pred']['O(g^4)']:.4f}, O6 {v['pred']['O(g^6)']:.4f}]"
                           for k, v in R["edges"].items())
            L.append(f"{R['case']:5s} {R['kappa']:6g} {R['r']:g}  edges (rich/cubic): {ed}")
            continue
        th, t1 = R["theory"], R["tests"]["T1"]["rich"]
        lo, hi = R["edges"]["lo"], R["edges"]["hi"]
        t3 = R["tests"].get("T3", {})
        w = f"{t3['width'][0] * 1e3:.2f}({t3['width'][1] * 1e3:.2f})" if "width" in t3 else "-"
        wt = f"{t3['width_theory_slope'][0] * 1e3:.2f}({t3['width_theory_slope'][1] * 1e3:.2f})" if t3 else "-"
        L.append(f"{R['case']:5s} {R['kappa']:6g} {R['r']:g} {th['g_c']:6.4f} {th['lambda']:4.2f} | "
                 f"{t1['A_23_minus_2']['t']:+6.1f} {t1['B_23_minus_3']['t']:+6.1f} [{R['tests']['T1']['expected_z']}] "
                 f"{'PASS' if t1['pass'] else 'FAIL'} | "
                 f"{fmt(lo['rich/cubic'])} ({lo['pred']['O(g^4)']:.4f},{lo['pred']['O(g^6)']:.4f}) | "
                 f"{fmt(hi['rich/cubic'])} ({hi['pred']['O(g^4)']:.4f},{hi['pred']['O(g^6)']:.4f}) | "
                 f"{w} / {wt} [{th['width_O(g^6)'] * 1e3:.2f}] | {R['tests']['T1']['chains_cut_at_or_before_g_c']}")
    L.append("\nT1 detail (rich; trap; unanch): Delta x1e4 (se), Welch dof, pred O(g^6)")
    for R in results:
        if "T1" not in R["tests"]:
            continue
        t1 = R["tests"]["T1"]
        s = "  ".join(f"{m}: " + ", ".join(f"{t1[m][k]['d'] * 1e4:+.2f}({t1[m][k]['se'] * 1e4:.2f}) dof {t1[m][k]['dof']:.0f} @g={t1[m][k]['g']:g}"
                                          for k in ("A_23_minus_2", "B_23_minus_3")) for m in METHODS)
        L.append(f"{R['case']:5s} {s}   pred {t1['pred_Delta']['O(g^6)'] * 1e4:+.2f}")
    L.append("\nedge cross-checks [rich/cubic, rich/u, trap/cubic, unanch/cubic]; sign changes if > 1")
    for R in results:
        for tag, e in R["edges"].items():
            ks = ["rich/cubic", "rich/u", "trap/cubic", "unanch/cubic"]
            nsc = e["rich/cubic"]["n_sign_changes"]
            L.append(f"{R['case']:5s} {tag:10s} " + " / ".join(f"{e[k]['value']:.4f}" for k in ks)
                     + (f"   z vs O6 {e['z_vs_pred']['O(g^6)']:+.1f}" if "z_vs_pred" in e else "")
                     + (f"   [{nsc} sign changes]" if nsc > 1 else ""))
    for R in results:
        for k in ("T5", "T8", "T7"):
            if k in R["tests"]:
                v = R["tests"][k]
                if k == "T5":
                    s = "; ".join(f"{c}-<23>: {x['23b']['d'] * 1e4:+.2f}({x['23b']['se'] * 1e4:.2f}) pred {x['23b']['pred'] * 1e4:+.2f} "
                                  f"pull {x['23b']['pull']:+.1f}{' STOP' if x['23b']['stop'] else ''} "
                                  f"[vs 20x20: {x['23a']['d'] * 1e4:+.2f}({x['23a']['se'] * 1e4:.2f})]" for c, x in v.items())
                elif k == "T8":
                    s = "; ".join(f"{c}: " + ", ".join(f"{kk} {fmt(e)}" for kk, e in x.items() if kk != "beats_23_and_3_2sigma")
                                  + f", below <23> and <3> (2 sigma) at g = {[b['g'] for b in x['beats_23_and_3_2sigma']]}"
                                  for c, x in v.items())
                else:
                    s = json.dumps(v, default=float)
                L.append(f"{R['case']:5s} {k}: {s}")
        for c in R["tests"].get("T6", []):
            L.append(f"{R['case']:5s} T6 {c['a']} vs {c['b']}: n={c['n_points']} chi2/pt {c['chi2_per_point']:.2f} "
                     f"max|z| {c['max_abs_z']:.2f} (#>2: {c['n_abs_z_gt2']}) max|dF| {c['max_abs_diff']:.1e}")
        z = {k: round(v["z_g0"], 2) for k, v in R["branches"].items()}
        L.append(f"{R['case']:5s} z(E_meas(g0) - e_PT(g0)): {z}")
        if R.get("prereg_check", {}).get("mismatch"):
            L.append(f"{R['case']:5s} PREREG TABLE MISMATCH: {R['prereg_check']['mismatch']}")
        for n in R["notes"]:
            L.append(f"{R['case']:5s} note: {n}")
    if t4:
        L.append("\nT4 (r0/r1 width ratio): " + "; ".join(f"{x['pair']} {x['ratio']:.2f}({x['se']:.2f}) pred {x['pred_prereg']} "
                                                         f"(recomputed {x['pred_recomputed']:.2f})" for x in t4))
    heads = [R for R in results if R["case"] in CASES and "T1" in R["tests"]]
    if heads:
        L.append(f"\nHEADLINE T1 (rich): {sum(R['tests']['T1']['rich']['pass'] for R in heads)}/{len(heads)} primary cases pass"
                 f" (pre-registered: all seven){'' if len(heads) == 7 else ' [incomplete]'}; failures at lambda <= 0.4: "
                 f"{[R['case'] for R in heads if not R['tests']['T1']['rich']['pass'] and R['theory']['lambda'] <= 0.4]}")
        L.append(f"T5 stop rule triggered in: {[R['case'] for R in heads if R['tests'].get('T5_stop')] or 'none'}")
    return "\n".join(L)


def selftest():
    """Noise-free check on the S5a grids (parsed from run_s5a_23.sh): m_x = -de/dg from the O(g^6) series;
    errors of Delta(g_c) and of the window edges from quadrature and interpolation alone."""
    from sse_seq_branches_v3 import parse_grid
    spec = dict(re.findall(r'^G_(\w+)="([^"]+)"', (Path(__file__).parent / "run_s5a_23.sh").read_text(), re.M))
    print("selftest (noise-free): errors x1e6 in Delta(g_c) = F<23> - F<2> at g_c, x1e4 in the edges (lo, hi)")
    for case, (kappa, r) in CASES.items():
        g = parse_grid(spec[case])
        g_f = float(spec[case].split(",")[0].split(":")[1])
        th = theory(kappa, r)
        gc = th["g_c"]
        ic = int(np.argmin(np.abs(g - gc)))
        c6 = {n: series(q, kappa, r, 6) for n, q in (("23", (2, 3)), ("2", (2,)), ("3", (3,)))}
        mx = np.vstack([-sum(2 * k * ck * g ** (2 * k - 1) for k, ck in enumerate(c) if k) for c in c6.values()])
        exact = np.vstack([e_series(c, g) for c in c6.values()])
        e0 = np.array([[e_series(c, G0)] for c in c6.values()])
        row = f"{case} g_c={gc:.4f} (grid {g[ic]:.4f}) Delta={exact[0, ic] - exact[1, ic]:+.3e}:"
        for quad in ("rich", "trap"):
            Fq = e0 - cumint(g, mx, quad, g_f)
            dD = (Fq[0, ic] - Fq[1, ic]) - (exact[0, ic] - exact[1, ic])
            lo = crossing(g, Fq[0] - Fq[1], "cubic")[0] - th["lo"]["O(g^6)"]
            hi = crossing(g, Fq[2] - Fq[0], "cubic")[0] - th["hi"]["O(g^6)"]
            row += f"  {quad}: {dD * 1e6:+.3f} ({lo * 1e4:+.2f},{hi * 1e4:+.2f})"
        for how in ("cubic", "u"):
            lo = crossing(g, exact[0] - exact[1], how)[0] - th["lo"]["O(g^6)"]
            hi = crossing(g, exact[2] - exact[0], how)[0] - th["hi"]["O(g^6)"]
            row += f"  interp-{how}: ({lo * 1e4:+.3f},{hi * 1e4:+.3f})"
        seg = coarse_segment(g, g_f)
        row += f"  [coarse m={seg[1] - seg[0]}]"
        p = PREREG[case]
        mine = (gc, th["lo"]["O(g^6)"], th["hi"]["O(g^6)"], th["Delta_gc"]["O(g^6)"], th["comp_minus_23_gc"]["223"],
                th["comp_minus_23_gc"]["233"], th["width_O(g^6)"])
        row += "  prereg table: " + ("ok" if all(abs(a - b) <= 0.006 * abs(a) + 6e-5 * (i < 3) for i, (a, b) in enumerate(zip(p, mine)))
                                     else f"MISMATCH {list(zip(p[:7], np.round(mine, 6)))}")
        print(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(SEQDIR), help="directory of the seqbr_*.csv files")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--files", nargs="+", help="custom mode: branch CSVs of one (kappa, r)")
    ap.add_argument("--label", default="custom")
    ap.add_argument("--kappa", type=float)
    ap.add_argument("--rperp", type=float, default=0.0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.files:
        results, t4, tag = [custom_case(a.label, a.files, a.kappa, a.rperp)], [], a.label
        t3_t4(results)
    else:
        results, missing = [], []
        for c in [c for c in a.cases.split(",") if c]:
            res, miss = s5a_case(c, Path(a.dir))
            results += res or []
            missing += miss
        t4, tag = t3_t4(results), "summary"
        for m in missing:
            print(m)
    txt = report(results, t4)
    print(txt)
    for R in results:
        (out / f"s5a_{R['case']}.json").write_text(json.dumps(R, indent=1, default=float))
    if t4:
        (out / "s5a_T4.json").write_text(json.dumps(t4, indent=1, default=float))
    (out / f"s5a_{tag}.txt").write_text(txt + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()

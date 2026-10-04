#!/usr/bin/env python3
"""Statistical re-analysis for the Paper 1 revision (referee round C2, manuscript/review/referee_C2_B_numerics.md).

Uses the frozen functions of experiments/s5a_analyze_v3.py (anchored "rich" quadrature, cubic-spline roots,
jackknife, Welch) without changing them. Outputs go to reanalysis_v3/revision1/<task>.{json,txt}.

Tasks (--task, comma-separated; default all):
  energy   (B 1a) e_b(g) - f_b(g) along every S5a branch and every step-2 L12/L24 template branch, with two energy
           estimators: "op", the SSE operator-count energy (E_per_spin), and "diag" = -Cx - Cy + kappa C2
           - (r/2) Cperp - g m_x, the Ising energy of the state at imaginary time 0 plus the field term (an
           independent estimator of the same energy). Per-g paired z (chain-level differences), and a jackknife
           offset (mean of e - f over g) and linear trend in g per branch. Energy differences at g_c for
           Table V (Delta E2 on 20x20, Delta E3 on 20x30), and the <23> 20x20 - 20x30 energy offset at P3.
  covar    (B 4f) jackknife-covariance statistics: the cross-lattice comparisons of T6 and K1 as a single-field
           offset at g_c, a Hotelling T^2 on k points, and a Monte-Carlo p-value of max|z| under the estimated
           correlation; the P2 T8 range with its correlation and effective number of independent points.
  cuts     (B 6c) chains cut per branch, at which g, into which q*; crossings with every cut chain dropped.
  tableIV  (B 4a) the Table IV extraction applied to the O(g^6) crossings (theory column and theory means).
  power    (B 5a) planned and realised power of the 14 T1 comparisons (noncentral t, Welch dof).
  reruns   (B 4d, 4e, 4g) rerun chi^2 with Welch-Satterthwaite dof; equal-weight and pooled-chain
           alternatives to the inverse-variance combination; Table III crossings with linear-in-g^2 roots.
  k047     (B minor 2) the O(g^4) value e_PT(g0) of <3> at kappa = 0.47, r = 0 against the measured energies.

Run from vu_work/ with the pinned env, one thread:
    OMP_NUM_THREADS=1 ~/.venvs/annni-py311/bin/python -W ignore experiments/revision1_stats_v3.py [--task energy,covar]
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import copy
import json
from fractions import Fraction as Fr

import numpy as np
import pandas as pd
from scipy.stats import chi2, f as fdist, nct, norm, t as tdist

import s5a_analyze_v3 as m

R3 = Path("10_results/bilayer_annni_paper_results/reanalysis_v3")
SEQ = R3 / "sse_validation" / "seq_branches"
STEP2 = R3 / "sse_validation" / "branches"
FINE = R3 / "sse_validation" / "fine_grid_check_12"
S5A = R3 / "s5a_analysis"
AUDIT = R3 / "audit_12" / "secVA_numbers.json"
OUT = R3 / "revision1"
PT4 = json.loads((R3 / "perturbation" / "pt_order4_analysis.json").read_text())["boundaries"]


# ---------------------------------------------------------------- helpers
def load_full(path):
    """Branches of a CSV (frozen loader) plus the pair correlations needed for the diagonal energy."""
    bs, kappa, r = m.load_file(path)
    df = pd.read_csv(path)
    if "part" in df.columns:
        df = df[df.part == "ascending"]
    for b in bs:
        d = df[df.start == next(s for s in df.start.unique() if m.sname(m.parse_seq(s)) == b.name)]
        piv = lambda c: d.pivot_table(index="chain", columns="g", values=c, aggfunc="first").to_numpy(float)
        b.C = {c: piv(c) for c in ("Cx", "Cy", "C2", "Cperp")}
        b.chains = np.array(sorted(d.chain.unique()))
    return bs, kappa, r


def energies(b, kappa, r):
    op = b.E
    diag = -b.C["Cx"] - b.C["Cy"] + kappa * b.C["C2"] - 0.5 * r * np.nan_to_num(b.C["Cperp"]) - b.g[None, :] * b.mx
    return {"op": op, "diag": diag}


def subset(b, idx):
    """Copy of a branch restricted to the chains idx."""
    c = copy.copy(b)
    for a in ("mx", "E", "q"):
        setattr(c, a, getattr(b, a)[idx])
    c.cut_idx = b.cut_idx[idx]
    c.n = len(idx)
    if hasattr(b, "C"):
        c.C = {k: v[idx] for k, v in b.C.items()}
    return c


def pool(b1, b2):
    """Chains of two runs of the same structure on their common grid."""
    grid = np.array(sorted(set(b1.g) & set(b2.g)))
    c = copy.copy(b1)
    i1, i2 = np.searchsorted(b1.g, grid), np.searchsorted(b2.g, grid)
    c.g = grid
    for a in ("mx", "E", "q"):
        setattr(c, a, np.vstack([getattr(b1, a)[:, i1], getattr(b2, a)[:, i2]]))
    c.n = b1.n + b2.n
    c.cut_idx = np.array([np.nonzero(np.abs(c.q[i] - c.q[i, 0]) > 1e-9)[0][0] if np.any(np.abs(c.q[i] - c.q[i, 0]) > 1e-9)
                          else len(grid) for i in range(c.n)])
    c.g_f = None                      # rule from the common grid
    return c


def mean_se(X):
    n = np.sum(np.isfinite(X), 0)
    return np.nanmean(X, 0), np.nanstd(X, 0, ddof=1) / np.sqrt(n), n


def jk_components(stat, mats):
    """Delete-one jackknife variance, returned per branch (for Welch-Satterthwaite dof)."""
    means = [np.nanmean(M, 0) for M in mats]
    full = stat(means)
    comps = []
    for b, M in enumerate(mats):
        th = []
        for i in range(M.shape[0]):
            mm = list(means)
            mm[b] = np.nanmean(np.delete(M, i, 0), 0)
            th.append(stat(mm))
        th = np.array(th, float)
        th = th[np.isfinite(th)]
        n = M.shape[0]
        comps.append(((n - 1) / n * np.sum((th - th.mean()) ** 2), n - 1))
    return float(full), comps


def ws_dof(comps):
    v = np.array([c[0] for c in comps]); nu = np.array([c[1] for c in comps])
    return float(v.sum() ** 2 / np.sum(v ** 2 / nu))


def dump(name, obj, text):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1, default=float))
    (OUT / f"{name}.txt").write_text(text + "\n")
    print(text)


def all_branch_files():
    files = sorted(SEQ.glob("seqbr_*.csv"))
    files += sorted(STEP2.glob("branches_L12_k*_r*.csv")) + sorted(STEP2.glob("branches_L24_k*_r*.csv"))
    return files


# ---------------------------------------------------------------- task: energy route
def task_energy():
    rows, lines = [], ["e_b(g) - f_b(g) per branch: offset = mean over g > 0.05 of the chain-mean difference, trend = OLS slope in g;"
                       " both with jackknife SE over chains (correlation-aware). Per-g paired z: max|z|, chi2/pt (naive)."]
    lines.append(f"{'file':44s} {'branch':8s} {'n':>3s} | {'op: offset x1e5 (z)':>20s} {'trend x1e5/g (z)':>18s} {'max|z|':>6s} "
                 f"{'chi2/pt':>7s} | {'diag: offset (z)':>18s} {'trend (z)':>16s} {'max|z|':>6s} {'chi2/pt':>7s} | se_op/se_diag")
    for f in all_branch_files():
        bs, kappa, r = load_full(f)
        for b in bs:
            F = m.free_energy(b, kappa, r, "rich")
            ok = b.g > m.G0 + 1e-9
            row = {"file": f.name, "branch": b.name, "key": b.key, "kappa": kappa, "r": r, "n": b.n}
            for est, E in energies(b, kappa, r).items():
                D = E - F                                     # chains x g, NaN after cuts
                mu, se, n = mean_se(D)
                z = mu / se
                gg = b.g[ok]

                def off(mm):
                    return float(np.nanmean(mm[0][ok]))

                def slope(mm):
                    y = mm[0][ok]; k = np.isfinite(y)
                    return float(np.polyfit(gg[k], y[k], 1)[0]) if k.sum() > 2 else np.nan
                o, so, _ = m.jackknife(off, [D])
                s, ss, _ = m.jackknife(slope, [D])
                _, se_e, _ = mean_se(E)
                row[est] = {"offset": o, "offset_se": so, "offset_z": o / so, "trend": s, "trend_se": ss, "trend_z": s / ss,
                            "max_abs_z": float(np.nanmax(np.abs(z[ok]))), "chi2_per_pt": float(np.nanmean(z[ok] ** 2)),
                            "median_se_E": float(np.nanmedian(se_e[ok])),
                            "z": {f"{x:g}": float(v) for x, v in zip(b.g[ok], z[ok]) if np.isfinite(v)}}
            rows.append(row)
            op, dg = row["op"], row["diag"]
            lines.append(f"{f.name[:44]:44s} {b.name:8s} {b.n:3d} | {op['offset']*1e5:+8.2f}({op['offset_se']*1e5:5.2f}) {op['offset_z']:+5.1f} "
                         f"{op['trend']*1e5:+7.2f}({op['trend_se']*1e5:5.2f}) {op['trend_z']:+4.1f} {op['max_abs_z']:6.2f} {op['chi2_per_pt']:7.2f} | "
                         f"{dg['offset']*1e5:+7.2f}({dg['offset_se']*1e5:5.2f}) {dg['offset_z']:+4.1f} {dg['trend']*1e5:+7.2f}({dg['trend_se']*1e5:5.2f}) "
                         f"{dg['trend_z']:+4.1f} {dg['max_abs_z']:6.2f} {dg['chi2_per_pt']:7.2f} | {op['median_se_E']/dg['median_se_E']:.2f}")
    # summaries by structure and lattice class
    t = pd.DataFrame([{"file": x["file"], "branch": x["branch"], "lat": x["key"].split("@")[1], "est": est,
                       "oz": x[est]["offset_z"], "tz": x[est]["trend_z"]} for x in rows for est in ("op", "diag")])
    lines.append("\nsummary of offset z and trend z (mean, SD, n) by estimator and structure:")
    for (est, br), g in t.groupby(["est", "branch"]):
        lines.append(f"  {est:4s} {br:7s} n={len(g):3d}  offset z {g.oz.mean():+.2f} SD {g.oz.std():.2f}   trend z {g.tz.mean():+.2f} SD {g.tz.std():.2f}")
    lines.append("by estimator and lattice:")
    for (est, lat), g in t.groupby(["est", "lat"]):
        lines.append(f"  {est:4s} {lat:12s} n={len(g):3d}  offset z {g.oz.mean():+.2f} SD {g.oz.std():.2f}   trend z {g.tz.mean():+.2f} SD {g.tz.std():.2f}")
    # energy differences at g_c (Table V) and the P3 cross-lattice offset in energy
    tv, lines2 = {}, ["\nEnergy route for Table V at g_c: Delta E (Welch) vs Delta F (rich) vs Delta F_th (O(g^6)); x1e4"]
    cases = dict(m.CASES)
    extra = {"P6_pooled36": (0.52, 1)}
    for case, (kappa, r) in list(cases.items()) + list(extra.items()):
        if case == "P6_pooled36":
            d = S5A / "posthoc_P6" / "pooled36"
            fa, fb = d / "seqbr_Lx20_Ly20_k0.52_r1_T0.15_23-2_pooled36.csv", d / "seqbr_Lx20_Ly30_k0.52_r1_T0.15_23-3_pooled36.csv"
        else:
            fa, fb = m.seqbr(20, 20, kappa, r, 0.15, "23-2"), m.seqbr(20, 30, kappa, r, 0.15, "23-3")
        A = {b.name: b for b in load_full(fa)[0]}
        B = {b.name: b for b in load_full(fb)[0]}
        th = m.theory(kappa, r)
        gc = th["g_c"]
        res = {"g_c": gc, "dF_th": th["Delta_gc"]["O(g^6)"]}
        for tag, (x, y) in {"2": (A["<23>"], A["<2>"]), "3": (B["<23>"], B["<3>"])}.items():
            j = int(np.argmin(np.abs(x.g - gc)))
            Fx, Fy = m.free_energy(x, kappa, r, "rich"), m.free_energy(y, kappa, r, "rich")
            res[f"dF{tag}"] = m.welch(Fx[:, j], Fy[:, j])
            for est in ("op", "diag"):
                Ex, Ey = energies(x, kappa, r)[est], energies(y, kappa, r)[est]
                res[f"dE{tag}_{est}"] = m.welch(Ex[:, j], Ey[:, j])
        tv[case] = res
        s = f"  {case:12s} g_c={gc:.4f} th {res['dF_th']*1e4:+6.2f} |"
        for tag in ("2", "3"):
            s += (f" dF{tag} {res[f'dF{tag}']['d']*1e4:+6.2f}({res[f'dF{tag}']['se']*1e4:.2f})"
                  f" dE{tag}op {res[f'dE{tag}_op']['d']*1e4:+6.2f}({res[f'dE{tag}_op']['se']*1e4:.2f})"
                  f" dE{tag}diag {res[f'dE{tag}_diag']['d']*1e4:+6.2f}({res[f'dE{tag}_diag']['se']*1e4:.2f})"
                  f" [diag-th {(res[f'dE{tag}_diag']['d']-res['dF_th'])/res[f'dE{tag}_diag']['se']:+.1f}s] |")
        lines2.append(s)
    # the <23> 20x20 - 20x30 offset at P3, in F and in E
    kappa, r = 0.53, 1
    A = {b.name: b for b in load_full(m.seqbr(20, 20, kappa, r, 0.15, "23-2"))[0]}["<23>"]
    B = {b.name: b for b in load_full(m.seqbr(20, 30, kappa, r, 0.15, "23-3"))[0]}["<23>"]
    grid = np.array(sorted(set(A.g) & set(B.g)))
    ia, ib = np.searchsorted(A.g, grid), np.searchsorted(B.g, grid)
    FA, FB = m.free_energy(A, kappa, r, "rich")[:, ia], m.free_energy(B, kappa, r, "rich")[:, ib]
    EA, EB = {e: v[:, ia] for e, v in energies(A, kappa, r).items()}, {e: v[:, ib] for e, v in energies(B, kappa, r).items()}
    off = {"g": grid.tolist(), "dF": [], "dE_op": [], "dE_diag": []}
    lines2.append("\nP3 <23> 20x20 - 20x30, x1e5 (se): g | dF | dE_op | dE_diag")
    for j, x in enumerate(grid):
        w = [m.welch(FA[:, j], FB[:, j]), m.welch(EA["op"][:, j], EB["op"][:, j]), m.welch(EA["diag"][:, j], EB["diag"][:, j])]
        for k, key in enumerate(("dF", "dE_op", "dE_diag")):
            off[key].append(w[k])
        if x in (0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.1222, 1.2, 1.3, 1.4) or abs(x - 1.1222) < 1e-4:
            lines2.append(f"  {x:6.4f} | {w[0]['d']*1e5:+6.2f}({w[0]['se']*1e5:.2f}) | {w[1]['d']*1e5:+7.2f}({w[1]['se']*1e5:.2f}) | "
                          f"{w[2]['d']*1e5:+7.2f}({w[2]['se']*1e5:.2f})")
    # averaged energy offset over the fine band (1.02-1.24), a lower-noise summary (jackknife over chains of both)
    band = (grid >= 1.02 - 1e-9) & (grid <= 1.24 + 1e-9)
    for key, (XA, XB) in {"dE_diag": (EA["diag"], EB["diag"]), "dE_op": (EA["op"], EB["op"]), "dF": (FA, FB)}.items():
        v, se, _ = m.jackknife(lambda mm: float(np.nanmean(mm[0][band]) - np.nanmean(mm[1][band])), [XA, XB])
        off[f"{key}_band_mean"] = [v, se]
        lines2.append(f"  band-mean (1.02-1.24) {key}: {v*1e5:+.2f}({se*1e5:.2f}) x1e5")
    dump("energy", {"branches": rows, "tableV": tv, "P3_23_cross_lattice": off}, "\n".join(lines + lines2))


# ---------------------------------------------------------------- task: correlation-aware comparisons
def cov_compare(A, FA, B, FB, gc, k=5, nmc=20000, seed=1):
    """A vs B (same structure, two lattices): single-field offset at g_c, Hotelling T^2 on k points,
    max|z| with its Monte-Carlo p-value under the estimated correlation, and the naive statistics."""
    grid, a, b = m.align(A, FA, B, FB)
    fin = np.all(np.isfinite(a), 0) & np.all(np.isfinite(b), 0) & (grid > m.G0 + 1e-9)
    g, a, b = grid[fin], a[:, fin], b[:, fin]
    na, nb = a.shape[0], b.shape[0]
    d = a.mean(0) - b.mean(0)
    Sa, Sb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    C = Sa / na + Sb / nb
    se = np.sqrt(np.diag(C))
    z = d / se
    Rm = C / np.outer(se, se)
    lam = np.clip(np.linalg.eigvalsh(Rm), 0, None)
    n_eff = float(lam.sum() ** 2 / np.sum(lam ** 2))
    jc = int(np.argmin(np.abs(g - gc)))
    w = m.welch(a[:, jc], b[:, jc])
    idx = np.unique(np.round(np.linspace(0, len(g) - 1, k)).astype(int))
    if jc not in idx:
        idx = np.unique(np.append(idx[:-1], jc))
    kk = len(idx)
    Sp = ((na - 1) * Sa + (nb - 1) * Sb) / (na + nb - 2)
    Sk = Sp[np.ix_(idx, idx)] * (1 / na + 1 / nb)
    T2 = float(d[idx] @ np.linalg.solve(Sk, d[idx]))
    nu = na + nb - 2
    Fst = (nu - kk + 1) / (nu * kk) * T2
    pT2 = float(fdist.sf(Fst, kk, nu - kk + 1))
    rng = np.random.default_rng(seed)
    ev, U = np.linalg.eigh(Rm)
    L = U * np.sqrt(np.clip(ev, 0, None))
    draws = rng.standard_normal((nmc, len(g))) @ L.T
    mz = np.abs(draws).max(1)
    zmax = float(np.max(np.abs(z)))
    return {"a": A.key, "b": B.key, "n_points": int(len(g)), "n_eff": n_eff,
            "min_neighbour_corr": float(np.min(np.diag(Rm, 1))) if len(g) > 1 else None,
            "offset_at_gc": {"g": float(g[jc]), **w},
            "hotelling": {"g_points": g[idx].tolist(), "k": kk, "T2": T2, "F": float(Fst), "df": [kk, nu - kk + 1], "p": pT2},
            "max_abs_z": zmax, "p_max_abs_z_correlated": float(np.mean(mz >= zmax)),
            "p_max_abs_z_if_independent": float(1 - (1 - 2 * norm.sf(zmax)) ** len(g)),
            "chi2_per_point_naive": float(np.mean(z ** 2)), "mean_z": float(z.mean())}


def task_covar():
    res, L = {"T6": {}, "K1": {}, "T8": {}}, ["Correlation-aware cross-lattice comparisons (jackknife/chain covariance across g)"]

    def line(tag, c):
        o = c["offset_at_gc"]
        return (f"  {tag:34s} n={c['n_points']:2d} n_eff={c['n_eff']:.1f} | offset@g={o['g']:.4f}: {o['d']*1e5:+6.2f}({o['se']*1e5:.2f})e-5"
                f" t={o['t']:+.2f} dof {o['dof']:.0f} | Hotelling k={c['hotelling']['k']} T2={c['hotelling']['T2']:.1f} p={c['hotelling']['p']:.3f}"
                f" | max|z|={c['max_abs_z']:.2f} p_corr={c['p_max_abs_z_correlated']:.2f} (p_indep={c['p_max_abs_z_if_independent']:.3f})"
                f" | naive chi2/pt {c['chi2_per_point_naive']:.2f}, mean z {c['mean_z']:+.2f}")
    for case, (kappa, r) in m.CASES.items():
        fa, fb = m.seqbr(20, 20, kappa, r, 0.15, "23-2"), m.seqbr(20, 30, kappa, r, 0.15, "23-3")
        A = {b.name: b for b in m.load_file(fa)[0]}
        B = {b.name: b for b in m.load_file(fb)[0]}
        gc = m.theory(kappa, r)["g_c"]
        F = lambda b: m.free_energy(b, kappa, r, "rich")
        c = cov_compare(A["<23>"], F(A["<23>"]), B["<23>"], F(B["<23>"]), gc)
        res["T6"][f"{case}:<23>@20x20-20x30"] = c
        L.append(line(f"{case} <23> 20x20 - 20x30", c))
        if case in m.STEP2_L24:
            s2 = {b.name: b for b in m.load_file(STEP2 / m.STEP2_L24[case])[0]}["<3>"]
            c = cov_compare(B["<3>"], F(B["<3>"]), s2, F(s2), gc)
            res["T6"][f"{case}:<3>@20x30-24x24"] = c
            L.append(line(f"{case} <3> 20x30 - 24x24", c))
        if case == "P1":
            K = {b.name: b for b in m.load_file(m.seqbr(40, 40, kappa, r, 0.15, "23-2"))[0]}
            for nm in ("<23>", "<2>"):
                c = cov_compare(K[nm], F(K[nm]), A[nm], F(A[nm]), gc)
                res["K1"][f"P1:{nm}@40x40-20x20"] = c
                L.append(line(f"P1 {nm} 40x40 - 20x20 (K1)", c))
    # P2 T8: <233> below <23> and <3> over 1.10-1.30
    kappa, r = 0.5625, 0
    B = {b.name: b for b in m.load_file(m.seqbr(20, 30, kappa, r, 0.15, "23-3"))[0]}
    c233 = m.load_file(m.seqbr(20, 32, kappa, r, 0.15, "233"))[0][0]
    F = lambda b: m.free_energy(b, kappa, r, "rich")
    L.append("\nP2 T8: <233> (20x32) minus <23> / <3> (20x30): pointwise Welch t, correlation across 1.10-1.30")
    for ref in ("<23>", "<3>"):
        grid, a, b = m.align(c233, F(c233), B[ref], F(B[ref]))
        sel = (grid >= 1.10 - 1e-9) & (grid <= 1.30 + 1e-9)
        g, a, b = grid[sel], a[:, sel], b[:, sel]
        C = np.cov(a, rowvar=False) / a.shape[0] + np.cov(b, rowvar=False) / b.shape[0]
        se = np.sqrt(np.diag(C)); d = a.mean(0) - b.mean(0); Rm = C / np.outer(se, se)
        lam = np.clip(np.linalg.eigvalsh(Rm), 0, None)
        ts = [m.welch(a[:, j], b[:, j]) for j in range(len(g))]
        j12 = int(np.argmin(np.abs(g - 1.2)))
        res["T8"][f"<233>-{ref}"] = {"g": g.tolist(), "t": [x["t"] for x in ts], "d": [x["d"] for x in ts], "se": [x["se"] for x in ts],
                                    "dof": [x["dof"] for x in ts], "n_eff": float(lam.sum() ** 2 / np.sum(lam ** 2)),
                                    "corr_first_last": float(Rm[0, -1]), "min_corr": float(Rm.min()), "at_1.2": ts[j12]}
        L.append(f"  vs {ref}: t = " + ", ".join(f"{x:.2f}:{t['t']:+.1f}" for x, t in zip(g, ts))
                 + f" | n_eff={res['T8'][f'<233>-{ref}']['n_eff']:.2f}, corr(1.10,1.30)={Rm[0,-1]:.2f}, min corr={Rm.min():.2f}"
                 + f" | at g=1.2: {ts[j12]['d']*1e4:+.2f}({ts[j12]['se']*1e4:.2f})e-4, t={ts[j12]['t']:+.1f}, dof {ts[j12]['dof']:.0f}")
    dump("covar", res, "\n".join(L))


# ---------------------------------------------------------------- task: cut statistics
def cut_table(b):
    out = []
    for i, c in enumerate(b.cut_idx):
        if c < len(b.g):
            out.append({"chain": int(b.chains[i]) if hasattr(b, "chains") else i, "g_cut": float(b.g[c]),
                        "q_before": float(b.q[i, c - 1]), "q_at_cut": float(b.q[i, c]), "q_end": float(b.q[i, -1]),
                        "q_path_after": sorted(set(np.round(b.q[i, c:], 4).tolist()))})
    return out


def edge_drop(bo, bn, kappa, r, how="cubic"):
    """Edge where bn drops below bo: all chains, and with every chain that is ever cut removed."""
    Fo, Fn = m.free_energy(bo, kappa, r, "rich"), m.free_energy(bn, kappa, r, "rich")
    full = m.edge(bo, Fo, bn, Fn, how)
    keep_o = np.nonzero(bo.cut_idx >= len(bo.g))[0]
    keep_n = np.nonzero(bn.cut_idx >= len(bn.g))[0]
    if len(keep_o) < 3 or len(keep_n) < 3:
        return full, None, (len(keep_o), len(keep_n))
    so, sn = subset(bo, keep_o), subset(bn, keep_n)
    drop = m.edge(so, m.free_energy(so, kappa, r, "rich"), sn, m.free_energy(sn, kappa, r, "rich"), how)
    return full, drop, (len(keep_o), len(keep_n))


def task_cuts():
    L, res = ["Chain cuts per branch (first change of q*): n_cut/n, cut fields, q* before -> at cut -> end"], {"branches": [], "edges": []}
    files = all_branch_files() + sorted(FINE.glob("fine_L12_*.csv"))
    for f in files:
        bs, kappa, r = load_full(f)
        for b in bs:
            ct = cut_table(b)
            res["branches"].append({"file": f.name, "branch": b.name, "key": b.key, "n": b.n, "n_cut": len(ct), "cuts": ct,
                                    "g_max": float(b.g[-1])})
            if ct:
                gs = sorted(x["g_cut"] for x in ct)
                qs = sorted(set((x["q_before"], x["q_at_cut"]) for x in ct))
                L.append(f"  {f.name[:46]:46s} {b.name:7s} cut {len(ct):2d}/{b.n:2d} at g = {gs[0]:.2f}-{gs[-1]:.2f} (grid max {b.g[-1]:.2f});"
                         f" q*/pi before->at cut: {[(round(a, 3), round(c, 3)) for a, c in qs]}")
    n_br = len(res["branches"]); n_cutbr = sum(1 for x in res["branches"] if x["n_cut"])
    L.insert(1, f"  {n_br} branches, {n_cutbr} with at least one cut; all other branches have no cut anywhere on their grid")
    # crossings with cut chains dropped
    L.append("\nCrossings (rich/cubic) with all chains vs with every ever-cut chain removed")
    pairs = []
    for f in sorted(STEP2.glob("branches_L12_k*_r*.csv")) + sorted(STEP2.glob("branches_L24_k*_r*.csv")) + sorted(FINE.glob("fine_L12_*.csv")):
        bs, kappa, r = m.load_file(f)
        br = {b.name: b for b in bs}
        old = "FM" if kappa < 0.5 else "<2>"
        pairs.append((f.name, "g*", br[old], br["<3>"], kappa, r))
    for case, (kappa, r) in m.CASES.items():
        A = {b.name: b for b in m.load_file(m.seqbr(20, 20, kappa, r, 0.15, "23-2"))[0]}
        B = {b.name: b for b in m.load_file(m.seqbr(20, 30, kappa, r, 0.15, "23-3"))[0]}
        pairs += [(case, "lo", A["<2>"], A["<23>"], kappa, r), (case, "hi", B["<23>"], B["<3>"], kappa, r),
                  (case, "32", A["<2>"], B["<3>"], kappa, r)]
    for name, tag, bo, bn, kappa, r in pairs:
        full, drop, keep = edge_drop(bo, bn, kappa, r)
        cuts_o = bo.n - keep[0]; cuts_n = bn.n - keep[1]
        mincut = min([bo.g[c] for c in bo.cut_idx if c < len(bo.g)] + [bn.g[c] for c in bn.cut_idx if c < len(bn.g)] + [np.inf])
        row = {"source": name, "edge": tag, "old": bo.key, "new": bn.key, "full": full, "dropped": drop, "kept": keep,
               "n_cut": [int(cuts_o), int(cuts_n)], "min_cut_g": float(mincut)}
        res["edges"].append(row)
        if cuts_o or cuts_n:
            s = (f"  {name[:40]:40s} {tag:3s} cut {cuts_o}+{cuts_n} (first at g={mincut:.2f}) | all {full['value']:.4f}({full['se']*1e4:.0f})"
                 + (f" | dropped {drop['value']:.4f}({drop['se']*1e4:.0f}) diff {drop['value']-full['value']:+.4f}" if drop else
                    f" | cannot drop: {keep} uncut chains kept (the crossing lies {mincut - full['value']:.2f} below the first cut)"))
            L.append(s)
    L.append("  (pairs without any cut chain are identical by construction and not listed)")
    dump("cuts", res, "\n".join(L))


# ---------------------------------------------------------------- task: Table IV with the theory column
def slope_points(src, gkey, sekey="se"):
    rows = []
    for key, v in src.items():
        k, r = (float(x) for x in key.split("_"))
        r = int(r)
        g, sg = gkey(v), v["best"]["se"]
        lam = v["best"]["g"] / (2 + r)
        if lam > 0.5 + 1e-12:
            continue
        side = "FM3" if k < 0.5 else "32"
        F3 = 6.0 / ((2 + r) * (3 + r) * (5 + r))
        Ath = F3 / 8 if side == "FM3" else F3 / 4
        B = -PT4[f"r={r}"]["k1_FM3"] if side == "FM3" else PT4[f"r={r}"]["k1_32"]
        x, y = g * g, abs(k - 0.5)
        Ai = y / x - B * x
        sA = abs(y / x ** 2 + B) * 2 * g * sg
        rows.append({"kappa": k, "r": r, "side": side, "lambda": lam, "g": g, "A_over_th": Ai / Ath, "se": sA / Ath})
    return rows


def wmean(rows, key="A_over_th", sekey="se"):
    out = {}
    for r in (0, 1):
        for side in ("FM3", "32"):
            s = [x for x in rows if x["r"] == r and x["side"] == side]
            if not s:
                continue
            w = np.array([1 / x[sekey] ** 2 for x in s]); v = np.array([x[key] for x in s])
            out[f"r{r}_{side}"] = {"mean": float((w * v).sum() / w.sum()), "se": float(1 / np.sqrt(w.sum())), "n": len(s),
                                   "chi2": float((w * (v - (w * v).sum() / w.sum()) ** 2).sum())}
    return out


def task_tableIV():
    d = json.loads(AUDIT.read_text())["L12"]
    q = slope_points(d, lambda v: v["best"]["g"])
    t = slope_points(d, lambda v: v["theory"]["O6"])
    for a, b in zip(q, t):
        a["A_over_th_theory"] = b["A_over_th"]
        a["se_qmc"] = a["se"]
    mq = wmean(q)
    mt = wmean([{**x, "A_over_th": x["A_over_th_theory"]} for x in q])      # QMC errors as weights
    L = ["Table IV with a theory column: the same extraction applied to the O(g^6) crossings g*_6 (weights = QMC errors)",
         f"{'side':5s} {'r':>1s} {'kappa':>6s} {'lambda':>6s} {'A_i/A_th QMC':>14s} {'theory':>8s} {'QMC-theory (sigma)':>20s}"]
    for x in q:
        L.append(f"{x['side']:5s} {x['r']:1d} {x['kappa']:6g} {x['lambda']:6.2f} {x['A_over_th']:8.4f}({x['se']*1e3:.0f})  {x['A_over_th_theory']:8.4f}"
                 f"  {(x['A_over_th']-x['A_over_th_theory'])/x['se']:+6.1f}")
    L.append("weighted means (QMC | theory-processed | difference, sigma):")
    for k in mq:
        L.append(f"  {k:7s} QMC {mq[k]['mean']:.4f}({mq[k]['se']*1e4:.0f})  theory {mt[k]['mean']:.4f}  diff {mq[k]['mean']-mt[k]['mean']:+.4f}"
                 f" ({(mq[k]['mean']-mt[k]['mean'])/mq[k]['se']:+.1f} sigma)  n={mq[k]['n']}")
    rq = {s: mq[f"r0_{s}"]["mean"] / mq[f"r1_{s}"]["mean"] * 2.4 for s in ("FM3", "32")}
    rt = {s: mt[f"r0_{s}"]["mean"] / mt[f"r1_{s}"]["mean"] * 2.4 for s in ("FM3", "32")}
    L.append(f"narrowing ratio 2.4 x (A/A_th r0)/(A/A_th r1): QMC {rq}, theory-processed {rt}")
    dump("tableIV", {"points": q, "qmc_means": mq, "theory_means": mt, "ratio_qmc": rq, "ratio_theory": rt}, "\n".join(L))


# ---------------------------------------------------------------- task: power of T1
PLANNED_Z = {"P5": 3.5, "P1": 6.5, "P4": 4.0, "P2": 4.0, "P6": 3.5, "P3": 4.0, "P7": 4.0}   # prereg table; ">=4" taken as 4


def task_power():
    L, res = ["Power of the 14 pre-registered T1 comparisons (one-sided, p < 0.02275; noncentral t, Welch dof)"], {"cases": {}}
    pr_real, pr_plan = [], []
    for case in m.CASES:
        d = json.loads((S5A / f"s5a_{case}.json").read_text())["tests"]["T1"]
        pred = d["pred_Delta"]["O(g^6)"]
        row = {}
        for tag in ("A_23_minus_2", "B_23_minus_3"):
            w = d["rich"][tag]
            dof = w["dof"]
            tc = float(tdist.ppf(m.P2SIG, dof))
            ncp = pred / w["se"]
            pw = float(nct.cdf(tc, dof, ncp))
            pz = PLANNED_Z[case]
            pp = float(nct.cdf(float(tdist.ppf(m.P2SIG, 22)), 22, -pz))
            row[tag] = {"se": w["se"], "dof": dof, "t_crit": tc, "expected_t": ncp, "power_realised": pw,
                        "planned_z": pz, "power_planned": pp, "observed_t": w["t"], "pass": w["pass"]}
            pr_real.append(pw); pr_plan.append(pp)
            L.append(f"  {case} {tag[:1]}: se {w['se']*1e4:.2f}e-4 dof {dof:.1f} t_crit {tc:+.2f} | expected t {ncp:+.2f} -> power {pw:.3f}"
                     f" | planned z {pz} -> power {pp:.3f} | observed t {w['t']:+.2f} {'pass' if w['pass'] else 'FAIL'}")
        res["cases"][case] = row
    res["P_all14_realised"] = float(np.prod(pr_real)); res["P_all14_planned"] = float(np.prod(pr_plan))
    res["expected_failures_realised"] = float(np.sum(1 - np.array(pr_real)))
    L.append(f"P(all 14 pass | theory exact): realised {res['P_all14_realised']:.3f}, planned {res['P_all14_planned']:.3f};"
             f" expected number of failures (realised) {res['expected_failures_realised']:.2f}")
    # the P6 follow-up alone
    fu = json.loads((S5A / "posthoc_P6" / "s5a_P6_fu1_only_POSTHOC.json").read_text())["tests"]["T1"]
    pred = fu["pred_Delta"]["O(g^6)"]
    for tag in ("A_23_minus_2", "B_23_minus_3"):
        w = fu["rich"][tag]
        pw = float(nct.cdf(float(tdist.ppf(m.P2SIG, w["dof"])), w["dof"], pred / w["se"]))
        res.setdefault("P6_followup", {})[tag] = {"se": w["se"], "dof": w["dof"], "power": pw}
        L.append(f"  P6 follow-up (24 chains) {tag[:1]}: se {w['se']*1e4:.2f}e-4 dof {w['dof']:.0f} power {pw:.3f}")
    dump("power", res, "\n".join(L))


# ---------------------------------------------------------------- task: reruns, combinations, u-interpolation
def crossing_components(bo, bn, kappa, r, how="cubic"):
    Fo, Fn = m.free_energy(bo, kappa, r, "rich"), m.free_energy(bn, kappa, r, "rich")
    grid, A, B = m.align(bo, Fo, bn, Fn)
    return jk_components(lambda mm: m.crossing(grid, mm[1] - mm[0], how)[0], [A, B])


def task_reruns():
    L, res = [], {"rerun_chi2": {}, "combinations": {}, "u_interp": {}}
    audit = json.loads(AUDIT.read_text())
    zs, zps = [], []
    L.append("(a) production vs fine-grid rerun with Welch-Satterthwaite dof (variance components: 2 branches x 2 runs)")
    for f in sorted(FINE.glob("fine_L12_*.csv")):
        bs, kappa, r = m.load_file(f)
        key = f"{kappa:g}_{r:g}"
        prod = {b.name: b for b in m.load_file(STEP2 / f"branches_L12_k{kappa:g}_r{r:g}.csv")[0]}
        fine = {b.name: b for b in bs}
        old = "FM" if kappa < 0.5 else "<2>"
        gp, cp = crossing_components(prod[old], prod["<3>"], kappa, r)
        gf, cf = crossing_components(fine[old], fine["<3>"], kappa, r)
        comps = cp + cf
        v = sum(c[0] for c in comps)
        z = (gp - gf) / np.sqrt(v)
        nu = ws_dof(comps)
        p = float(2 * tdist.sf(abs(z), nu))
        zp = float(norm.isf(p / 2))
        zs.append(z); zps.append(zp)
        res["rerun_chi2"][key] = {"g_prod": gp, "g_fine": gf, "z": z, "dof_ws": nu, "p": p, "z_gauss_equiv": zp,
                                  "se_prod": float(np.sqrt(sum(c[0] for c in cp))), "se_fine": float(np.sqrt(sum(c[0] for c in cf)))}
        L.append(f"  {key:8s} prod {gp:.4f} fine {gf:.4f} z {z:+.2f} dof_WS {nu:.1f} p {p:.3f} -> Gaussian-equivalent z {zp:.2f}")
    c2, c2p = float(np.sum(np.square(zs))), float(np.sum(np.square(zps)))
    res["rerun_chi2"]["chi2_naive"] = c2; res["rerun_chi2"]["chi2_ws"] = c2p; res["rerun_chi2"]["p_ws"] = float(chi2.sf(c2p, len(zs)))
    L.append(f"  chi2 naive {c2:.2f} (5 dof, p {chi2.sf(c2, 5):.3f}); with WS dof -> Gaussian-equivalent chi2 {c2p:.2f}, p {chi2.sf(c2p, 5):.3f}")
    # (b) combinations
    L.append("\n(b) combining the five reruns: inverse-variance (as in the draft) vs equal weights vs pooled chains vs production only")
    variants = {"inverse_variance": {}, "equal_weight": {}, "pooled_chains": {}, "production_only": {}}
    for key, v in audit["L12"].items():
        k, r = (float(x) for x in key.split("_"))
        prodv = {"g": v["rich"]["g"], "se": v["rich"]["se"]}
        variants["production_only"][key] = prodv
        if "combined" not in v:
            for n in ("inverse_variance", "equal_weight", "pooled_chains"):
                variants[n][key] = prodv
            continue
        fi = audit["fine"][key]["rich"]
        variants["inverse_variance"][key] = {"g": v["combined"]["g"], "se": v["combined"]["se"]}
        variants["equal_weight"][key] = {"g": 0.5 * (prodv["g"] + fi["g"]), "se": 0.5 * float(np.hypot(prodv["se"], fi["se"]))}
        prod = {b.name: b for b in m.load_file(STEP2 / f"branches_L12_k{k:g}_r{int(r)}.csv")[0]}
        fine = {b.name: b for b in m.load_file(FINE / audit["fine"][key]["file"])[0]}
        old = "FM" if k < 0.5 else "<2>"
        po, pn = pool(prod[old], fine[old]), pool(prod["<3>"], fine["<3>"])
        e = m.edge(po, m.free_energy(po, k, r, "rich"), pn, m.free_energy(pn, k, r, "rich"), "cubic")
        variants["pooled_chains"][key] = {"g": e["value"], "se": e["se"], "grid_max": float(po.g[-1]), "chains": [po.n, pn.n]}
        L.append(f"  {key:8s} prod {prodv['g']:.4f}({prodv['se']*1e4:.0f}) fine {fi['g']:.4f}({fi['se']*1e4:.0f}) | inv-var "
                 f"{v['combined']['g']:.4f}({v['combined']['se']*1e4:.0f}) equal {variants['equal_weight'][key]['g']:.4f}"
                 f"({variants['equal_weight'][key]['se']*1e4:.0f}) pooled(12 chains, common grid) {e['value']:.4f}({e['se']*1e4:.0f})")
    for n, src in variants.items():
        rows = slope_points({kk: {"best": vv, **({} if True else {})} for kk, vv in src.items()}, lambda vv: vv["best"]["g"])
        wm = wmean(rows)
        ratio = {s: 2.4 * wm[f"r0_{s}"]["mean"] / wm[f"r1_{s}"]["mean"] for s in ("FM3", "32")}
        res["combinations"][n] = {"means": wm, "ratio": ratio}
        L.append(f"  slopes [{n:16s}]: " + ", ".join(f"{kk} {vv['mean']:.4f}({vv['se']*1e4:.0f})" for kk, vv in wm.items())
                 + f" | ratios FM3 {ratio['FM3']:.3f}, 32 {ratio['32']:.3f}")
    # (c) linear-in-g^2 roots for Table III
    L.append("\n(c) Table III crossings: cubic-spline root (draft) vs linear in u = g^2 (rich quadrature, same chains)")
    for f in sorted(STEP2.glob("branches_L12_k*_r*.csv")) + sorted(STEP2.glob("branches_L24_k*_r*.csv")) + sorted(FINE.glob("fine_L12_*.csv")):
        bs, kappa, r = m.load_file(f)
        br = {b.name: b for b in bs}
        old = "FM" if kappa < 0.5 else "<2>"
        Fo, Fn = m.free_energy(br[old], kappa, r, "rich"), m.free_energy(br["<3>"], kappa, r, "rich")
        ec, eu = m.edge(br[old], Fo, br["<3>"], Fn, "cubic"), m.edge(br[old], Fo, br["<3>"], Fn, "u")
        res["u_interp"][f.name] = {"cubic": ec, "u": eu, "diff": eu["value"] - ec["value"]}
        L.append(f"  {f.name[:40]:40s} cubic {ec['value']:.4f}({ec['se']*1e4:.0f})  u-linear {eu['value']:.4f}({eu['se']*1e4:.0f})"
                 f"  diff {eu['value'] - ec['value']:+.5f} ({(eu['value'] - ec['value'])/ec['se']:+.2f} se)")
    dump("reruns", res, "\n".join(L))


# ---------------------------------------------------------------- task: kappa = 0.47 series value
def task_k047():
    L, res = ["kappa = 0.47: O(g^4) series e_PT(g0 = 1/20) of <3> and FM against the measured g0 energies (L = 12)"], {}
    for r in (0, 1):
        g0 = Fr(1, 20)
        e3 = m.per_column([3], Fr(47, 100), Fr(r))
        eF = m.coeffs("FM", Fr(47, 100), Fr(r))
        v3 = e3[0] + e3[1] * g0 ** 2 + e3[2] * g0 ** 4
        vF = eF[0] + eF[1] * g0 ** 2 + eF[2] * g0 ** 4
        df = pd.read_csv(STEP2 / f"branches_L12_k0.47_r{r}.csv")
        x = df[np.isclose(df.g, 0.05)]
        rr = {"e_PT_<3>": float(v3), "e_PT_<3>_exact": str(v3), "e_PT_FM": float(vF)}
        for (part, start), gdf in x.groupby(["part", "start"]):
            mu, se = gdf.E_per_spin.mean(), gdf.E_per_spin.std(ddof=1) / np.sqrt(len(gdf))
            ref = vF if start == "FM" else v3 if start in ("<3>par", "random") else None
            rr[f"{part}:{start}"] = {"mean": float(mu), "se": float(se), "n": int(len(gdf)),
                                     "pull_vs_series": float((mu - float(ref)) / se) if ref is not None else None}
        res[f"r={r}"] = rr
        L.append(f"  r={r}: e_PT <3> = {float(v3):.6f} ({v3}), FM = {float(vF):.6f}; " +
                 "; ".join(f"{k} {v['mean']:.5f}({v['se']*1e5:.0f}e-5)" + (f" pull {v['pull_vs_series']:+.2f}" if v["pull_vs_series"] is not None else "")
                           for k, v in rr.items() if isinstance(v, dict)))
    dump("k047", res, "\n".join(L))




# ---------------------------------------------------------------- task: energy route restricted to where the tests live
def band_stat(D, mask):
    return m.jackknife(lambda mm: float(np.nanmean(mm[0][mask])), [D])


def task_energy_band():
    """(B 1a, targeted) e - f averaged where the tests are decided: |g - g_c| <= 0.1 for the S5a branches, and
    0.1 <= g <= g* + 0.1 for the step-2 branches of each Table III crossing. Also the paired difference
    Delta(e - f) = (e - f)_<23> - (e - f)_other at the band, the correctly correlated form of Delta E - Delta F.
    A pipeline systematic of size delta in F of one branch would appear there as +-delta."""
    L, res = ["e - f (operator-count energy) averaged near the tests; jackknife over chains; x1e5"], {"S5a": {}, "step2": {}}
    pooled_r1 = []
    for case, (kappa, r) in m.CASES.items():
        A = {b.name: b for b in load_full(m.seqbr(20, 20, kappa, r, 0.15, "23-2"))[0]}
        B = {b.name: b for b in load_full(m.seqbr(20, 30, kappa, r, 0.15, "23-3"))[0]}
        gc = m.theory(kappa, r)["g_c"]
        row = {"g_c": gc}
        D = {}
        for tag, b in (("23@20x20", A["<23>"]), ("2@20x20", A["<2>"]), ("23@20x30", B["<23>"]), ("3@20x30", B["<3>"])):
            D[tag] = energies(b, kappa, r)["op"] - m.free_energy(b, kappa, r, "rich")
            mask = (np.abs(b.g - gc) <= 0.1 + 1e-9)
            v, se, _ = band_stat(D[tag], mask)
            row[tag] = {"band_mean": v, "se": se, "z": v / se, "n_points": int(mask.sum())}
        mask = (np.abs(A["<23>"].g - gc) <= 0.1 + 1e-9)
        for tag, (x, y) in {"d(e-f)_A_23-2": ("23@20x20", "2@20x20"), "d(e-f)_B_23-3": ("23@20x30", "3@20x30")}.items():
            v, se, _ = m.jackknife(lambda mm: float(np.nanmean(mm[0][mask]) - np.nanmean(mm[1][mask])), [D[x], D[y]])
            row[tag] = {"band_mean": v, "se": se, "z": v / se}
        res["S5a"][case] = row
        if r == 1:
            pooled_r1.append(row["d(e-f)_B_23-3"])
        L.append(f"  {case} (g_c {gc:.4f}, {mask.sum()} pts): " + "  ".join(f"{k} {row[k]['band_mean']*1e5:+6.2f}({row[k]['se']*1e5:.2f})"
                                                                          for k in ("23@20x20", "2@20x20", "23@20x30", "3@20x30"))
                 + f" | d(e-f) A {row['d(e-f)_A_23-2']['band_mean']*1e5:+6.2f}({row['d(e-f)_A_23-2']['se']*1e5:.2f})"
                 + f" B {row['d(e-f)_B_23-3']['band_mean']*1e5:+6.2f}({row['d(e-f)_B_23-3']['se']*1e5:.2f})")
    w = np.array([1 / x["se"] ** 2 for x in pooled_r1]); v = np.array([x["band_mean"] for x in pooled_r1])
    res["S5a"]["r1_B_weighted"] = {"mean": float((w * v).sum() / w.sum()), "se": float(1 / np.sqrt(w.sum()))}
    L.append(f"  r=1 cases (P3, P7, P6) weighted d(e-f)_B = {res['S5a']['r1_B_weighted']['mean']*1e5:+.2f}({res['S5a']['r1_B_weighted']['se']*1e5:.2f})e-5;"
             f" a +1e-4 artefact in Delta F3 (the <3>-side excess) would give about +10e-5 here")
    audit = json.loads(AUDIT.read_text())["L12"]
    L.append("\nstep-2 Table III branches: e - f averaged over 0.1 <= g <= g* + 0.1 (the range that decides g*)")
    zs = []
    for f in sorted(STEP2.glob("branches_L12_k*_r*.csv")) + sorted(STEP2.glob("branches_L24_k*_r*.csv")):
        bs, kappa, r = load_full(f)
        L_ = int(f.name.split("_")[1][1:])
        key = f"{kappa:g}_{r:g}"
        gstar = audit[key]["rich"]["g"] if L_ == 12 else json.loads(AUDIT.read_text())["L24"][key]["rich"]["g"]
        br = {b.name: b for b in bs}
        old = "FM" if kappa < 0.5 else "<2>"
        out = {}
        for nm in (old, "<3>"):
            b = br[nm]
            Dm = energies(b, kappa, r)["op"] - m.free_energy(b, kappa, r, "rich")
            mask = (b.g >= 0.1 - 1e-9) & (b.g <= gstar + 0.1 + 1e-9)
            v, se, _ = band_stat(Dm, mask)
            out[nm] = {"band_mean": v, "se": se, "z": v / se, "n_points": int(mask.sum())}
            zs.append(v / se)
        res["step2"][f.name] = out
        L.append(f"  {f.name[:32]:32s} g*={gstar:.3f}: " + "  ".join(f"{nm} {o['band_mean']*1e5:+6.2f}({o['se']*1e5:.2f}) z {o['z']:+.1f}" for nm, o in out.items()))
    zs = np.array(zs)
    res["step2_summary"] = {"n": int(len(zs)), "mean_z": float(zs.mean()), "sd_z": float(zs.std(ddof=1)), "max_abs_z": float(np.abs(zs).max())}
    L.append(f"  step-2 summary: n={len(zs)} z mean {zs.mean():+.2f} SD {zs.std(ddof=1):.2f} max|z| {np.abs(zs).max():.2f}")
    dump("energy_band", res, "\n".join(L))


# ---------------------------------------------------------------- task: crossings on grids truncated below the first cut
def truncate(b, gmax):
    c = copy.copy(b)
    k = int(np.sum(b.g <= gmax + 1e-9))
    c.g = b.g[:k]
    for a in ("mx", "E", "q"):
        setattr(c, a, getattr(b, a)[:, :k])
    c.cut_idx = np.minimum(b.cut_idx, k)
    c.g_f = None
    return c


def task_cuts_trunc():
    """(B 6c) where every chain of a branch is cut (all Table III 'old' branches), the survivorship-free check is to
    stop the grid below the first cut of either branch: every chain then contributes at every point."""
    L, res = ["Table III crossings (rich/cubic): full grid vs grid truncated one step below the first cut"], {}
    for f in sorted(STEP2.glob("branches_L12_k*_r*.csv")) + sorted(STEP2.glob("branches_L24_k*_r*.csv")) + sorted(FINE.glob("fine_L12_*.csv")):
        bs, kappa, r = m.load_file(f)
        br = {b.name: b for b in bs}
        old = "FM" if kappa < 0.5 else "<2>"
        bo, bn = br[old], br["<3>"]
        cuts = [bo.g[c] for c in bo.cut_idx if c < len(bo.g)] + [bn.g[c] for c in bn.cut_idx if c < len(bn.g)]
        full = m.edge(bo, m.free_energy(bo, kappa, r, "rich"), bn, m.free_energy(bn, kappa, r, "rich"), "cubic")
        if not cuts:
            continue
        gcut = min(cuts)
        gmax = max(x for x in bo.g if x < gcut - 1e-9)
        to, tn = truncate(bo, gmax), truncate(bn, gmax)
        tr = m.edge(to, m.free_energy(to, kappa, r, "rich"), tn, m.free_energy(tn, kappa, r, "rich"), "cubic")
        res[f.name] = {"full": full, "truncated": tr, "g_first_cut": float(gcut), "g_max_used": float(gmax)}
        L.append(f"  {f.name[:40]:40s} first cut {gcut:.2f}, grid to {gmax:.2f}: full {full['value']:.4f}({full['se']*1e4:.0f})"
                 f"  truncated {tr['value']:.4f}({tr['se']*1e4:.0f})  diff {tr['value']-full['value']:+.5f}")
    d = [abs(v["truncated"]["value"] - v["full"]["value"]) for v in res.values()]
    L.append(f"  max |diff| = {max(d):.5f} over {len(d)} crossings")
    dump("cuts_truncated", res, "\n".join(L))


# ---------------------------------------------------------------- task: per-structure lattice scan (period-parity idea)
def task_lattice_scan():
    """F of the same structure on every available lattice at the S5a points, relative to a reference lattice, at the
    grid point nearest g_c (single-field offset, jackknife/Welch) and with a Hotelling test. <3> exists on Ly = 12, 24
    (step 2), 30 (S5a) and 60 (revision V1/V2/V4): 2, 4, 5 and 10 periods; Ly = 30 is the only odd count."""
    L, res = ["Per-structure lattice scan: F(lattice) - F(reference) near g_c, e-5 (t); Hotelling p over 5 points"], {}
    pts = {"P1": (0.53, 0), "P3": (0.53, 1), "P2": (0.5625, 0), "P5": (0.52, 0), "P6": (0.52, 1), "P4": (0.545, 0), "P7": (0.545, 1)}
    for case, (kappa, r) in pts.items():
        gc = m.theory(kappa, r)["g_c"]
        pool = {}
        def add(f, names=None):
            if Path(f).exists():
                for b in m.load_file(f)[0]:
                    if names is None or b.name in names:
                        pool.setdefault(b.name, []).append(b)
        add(STEP2 / f"branches_L12_k{kappa:g}_r{r}.csv", {"<3>", "<2>"})
        add(STEP2 / f"branches_L24_k{kappa:g}_r{r}.csv", {"<3>", "<2>"})
        add(m.seqbr(20, 20, kappa, r, 0.15, "23-2")); add(m.seqbr(20, 30, kappa, r, 0.15, "23-3"))
        add(m.seqbr(20, 60, kappa, r, 0.15, "2-23-3")); add(m.seqbr(40, 40, kappa, r, 0.15, "23-2"))
        res[case] = {}
        for nm, bl in pool.items():
            if len(bl) < 2:
                continue
            ref = max(bl, key=lambda b: b.Lx * b.Ly)               # largest lattice as reference
            Fr_ = m.free_energy(ref, kappa, r, "rich")
            for b in bl:
                if b is ref:
                    continue
                c = cov_compare(b, m.free_energy(b, kappa, r, "rich"), ref, Fr_, gc)
                o = c["offset_at_gc"]
                res[case][f"{nm}:{b.lat}-{ref.lat}"] = c
                per = b.Ly // (4 if nm == "<2>" else 6 if nm == "<3>" else 5 if nm == "<23>" else 1)
                L.append(f"  {case} {nm:5s} {b.lat:6s} ({per:2d} periods, {b.n:2d} ch) - {ref.lat}: {o['d']*1e5:+7.2f}({o['se']*1e5:5.2f}) t {o['t']:+.2f}"
                         f" @g={o['g']:.3f} | Hotelling p {c['hotelling']['p']:.2f}")
    dump("lattice_scan", res, "\n".join(L))


# ---------------------------------------------------------------- referee B round 2
def ef_profile(b, kappa, r):
    """Per-g e - f (operator-count energy) with chain-level se; NaN after cuts."""
    D = energies(b, kappa, r)["op"] - m.free_energy(b, kappa, r, "rich")
    mu, se, n = mean_se(D)
    return D, mu, se


def task_r2_energy():
    """(B r2, 7a) estimator and precision; |pull| > 3 points and their positions; truncation at the first |pull| > 3;
    the 46 vs 54 branch sets."""
    L, res = [], {"estimator": {}, "big_pulls": [], "trunc": {}, "sets": {}}
    L.append("Estimator: E_per_spin = (-<n>/beta + sum_b |J_b| + N g)/N (SSE operator count), paired with f (rich) chain by chain.")
    # typical precision of E per lattice (median over branches and g of the chain-mean se)
    prec = {}
    for f in all_branch_files():
        bs, kappa, r = load_full(f)
        for b in bs:
            _, se, _ = mean_se(b.E)
            prec.setdefault(b.lat, []).append(float(np.nanmedian(se[1:])))
    for lat, v in sorted(prec.items()):
        res["estimator"][lat] = {"median_se_E": float(np.median(v)), "n_branches": len(v)}
        L.append(f"  typical se of the chain-mean E at one g, {lat:12s}: {np.median(v)*1e5:6.1f}e-5 (median over {len(v)} branches)")
    # |pull| > 3 points: step-2 Table III branches (old + <3>) and all S5a branches
    audit = json.loads(AUDIT.read_text())
    L.append("\n|pull| > 3 points of e - f (per g, chain-level se): branch, g, pull, g - g* (or g - g_c), g - first cut")
    step2 = sorted(STEP2.glob("branches_L12_k*_r*.csv")) + sorted(STEP2.glob("branches_L24_k*_r*.csv"))
    n_pts = 0
    for f in step2 + sorted(SEQ.glob("seqbr_*.csv")):
        bs, kappa, r = load_full(f)
        is_step2 = f.name.startswith("branches_")
        if is_step2:
            Lsz = int(f.name.split("_")[1][1:]); key = f"{kappa:g}_{int(r)}"
            gref = audit["L12" if Lsz == 12 else "L24"][key]["rich"]["g"]
            names = {"FM" if kappa < 0.5 else "<2>", "<3>"}
        else:
            gref = m.theory(kappa, r)["g_c"]
            names = {b.name for b in bs}
        for b in bs:
            if b.name not in names:
                continue
            D, mu, se = ef_profile(b, kappa, r)
            z = mu / se
            ok = (b.g > m.G0 + 1e-9) & np.isfinite(z)
            n_pts += int(ok.sum())
            cut = min([b.g[c] for c in b.cut_idx if c < len(b.g)] + [np.inf])
            for j in np.nonzero(ok & (np.abs(z) > 3))[0]:
                row = {"file": f.name, "branch": b.name, "g": float(b.g[j]), "pull": float(z[j]), "g_minus_ref": float(b.g[j] - gref),
                       "ref": "g*" if is_step2 else "g_c", "g_minus_first_cut": float(b.g[j] - cut) if np.isfinite(cut) else None}
                res["big_pulls"].append(row)
    bp = res["big_pulls"]
    L.append(f"  {len(bp)} points with |pull| > 3 out of {n_pts} (expected ~{n_pts*2*tdist.sf(3, 5):.0f} for t5, ~{n_pts*0.0027:.1f} for a normal)")
    for x in sorted(bp, key=lambda x: (x["file"], x["branch"], x["g"])):
        L.append(f"  {x['file'][:40]:40s} {x['branch']:6s} g={x['g']:.3f} pull {x['pull']:+.1f} | g-{x['ref']} {x['g_minus_ref']:+.2f}"
                 + (f" | g-first cut {x['g_minus_first_cut']:+.2f}" if x['g_minus_first_cut'] is not None else " | no cut"))
    below = [x for x in bp if x["g_minus_ref"] <= 0.1]
    L.append(f"  of these, {len(below)} lie at or below g*/g_c + 0.1")
    # truncation at the first significant deviation (|pull| > 3) of either branch of each Table III crossing
    L.append("\nTable III crossings with each pair truncated one grid step below the first |pull| > 3 of either branch (vs the q*-cut rule)")
    for f in step2:
        bs, kappa, r = load_full(f)
        br = {b.name: b for b in bs}
        old = "FM" if kappa < 0.5 else "<2>"
        firsts = []
        for nm in (old, "<3>"):
            _, mu, se = ef_profile(br[nm], kappa, r)
            z = mu / se
            k = np.nonzero((br[nm].g > m.G0 + 1e-9) & np.isfinite(z) & (np.abs(z) > 3))[0]
            if len(k):
                firsts.append(br[nm].g[k[0]])
        full = m.edge(br[old], m.free_energy(br[old], kappa, r, "rich"), br["<3>"], m.free_energy(br["<3>"], kappa, r, "rich"), "cubic")
        if not firsts:
            res["trunc"][f.name] = {"full": full, "first_dev": None}
            L.append(f"  {f.name[:36]:36s} no |pull| > 3 on either branch: {full['value']:.4f}({full['se']*1e4:.0f}) unchanged")
            continue
        gdev = min(firsts)
        gmax = max(x for x in br[old].g if x < gdev - 1e-9)
        if gmax <= full["value"] + 1e-9:
            res["trunc"][f.name] = {"full": full, "first_dev": float(gdev), "truncated": None}
            L.append(f"  {f.name[:36]:36s} first |pull|>3 at g={gdev:.2f} BELOW/AT the crossing {full['value']:.4f}: cannot truncate")
            continue
        to, tn = truncate(br[old], gmax), truncate(br["<3>"], gmax)
        tr = m.edge(to, m.free_energy(to, kappa, r, "rich"), tn, m.free_energy(tn, kappa, r, "rich"), "cubic")
        res["trunc"][f.name] = {"full": full, "first_dev": float(gdev), "g_max_used": float(gmax), "truncated": tr}
        L.append(f"  {f.name[:36]:36s} first |pull|>3 at g={gdev:.2f} (g* {full['value']:.3f}): full {full['value']:.4f}({full['se']*1e4:.0f})"
                 f" truncated at {gmax:.2f} {tr['value']:.4f}({tr['se']*1e4:.0f}) diff {tr['value']-full['value']:+.5f}")
    # the two branch sets
    res["sets"] = {"g0_pulls": "54 = 18 (kappa, r) cases x 3 templates (FM, <2>, <3>) on 12x12; 15 more on 24x24 reported separately",
                   "energy_route": "46 = 23 Table III crossings (18 on 12x12, 5 on 24x24) x the 2 branches that enter each crossing"
                                   " (FM or <2>, and <3>); the third template of each case does not enter Table III"}
    L.append("\nBranch sets: " + res["sets"]["g0_pulls"] + "; " + res["sets"]["energy_route"])
    dump("r2_energy", res, "\n".join(L))


def task_r2_offset():
    """(B r2, 1e) combined 20x30 vs 20x60 significance with each shared 20x60 reference counted once, and the
    <3> 'smaller lattices vs 20x60' pattern at P3 with the shared-reference covariance."""
    L, res = [], {}
    R = OUT
    # (1) Delta F_B: P3 (V1 vs S5a P3) and P6 (V2 vs pooled 36 = orig + fu1, one comparison): two 20x60 references
    v1 = json.loads((R / "queue_V1.json").read_text()); v2 = json.loads((R / "queue_V2.json").read_text())
    p3 = json.loads((S5A / "s5a_P3.json").read_text())["tests"]["T1"]["rich"]["B_23_minus_3"]
    pp = json.loads((S5A / "posthoc_P6" / "s5a_P6_pooled36_POSTHOC.json").read_text())["tests"]["T1"]["rich"]["B_23_minus_3"]
    po = json.loads((S5A / "s5a_P6.json").read_text())["tests"]["T1"]["rich"]["B_23_minus_3"]
    pf = json.loads((S5A / "posthoc_P6" / "s5a_P6_fu1_only_POSTHOC.json").read_text())["tests"]["T1"]["rich"]["B_23_minus_3"]
    d = [(v1["T1_B"]["d"] - p3["d"], float(np.hypot(v1["T1_B"]["se"], p3["se"]))),
         (v2["T1_B"]["d"] - pp["d"], float(np.hypot(v2["T1_B"]["se"], pp["se"])))]
    w = np.array([1 / x[1] ** 2 for x in d]); v = np.array([x[0] for x in d])
    mu, se = float((w * v).sum() / w.sum()), float(1 / np.sqrt(w.sum()))
    res["dFB_shift"] = {"P3": d[0], "P6_pooled36": d[1], "combined": [mu, se], "z": mu / se}
    L.append(f"Delta F_B(20x60) - Delta F_B(20x30), each 20x60 set once: P3 {d[0][0]*1e4:+.2f}({d[0][1]*1e4:.2f}), "
             f"P6 (20x30 orig+fu1 pooled vs the one V2 set) {d[1][0]*1e4:+.2f}({d[1][1]*1e4:.2f}) -> {mu*1e4:+.2f}({se*1e4:.2f})e-4, {mu/se:.1f} sigma")
    # the same with orig and fu1 as separate 20x30 sets, with the shared V2 variance in the covariance (GLS)
    a = v2["T1_B"]["d"]; sa = v2["T1_B"]["se"]
    x = np.array([v1["T1_B"]["d"] - p3["d"], a - po["d"], a - pf["d"]])
    C = np.diag([v1["T1_B"]["se"] ** 2 + p3["se"] ** 2, sa ** 2 + po["se"] ** 2, sa ** 2 + pf["se"] ** 2])
    C[1, 2] = C[2, 1] = sa ** 2
    Ci = np.linalg.inv(C); one = np.ones(3)
    mu3 = float(one @ Ci @ x / (one @ Ci @ one)); se3 = float(1 / np.sqrt(one @ Ci @ one))
    chi = float((x - mu3) @ Ci @ (x - mu3))
    res["dFB_shift_GLS3"] = {"values": x.tolist(), "combined": [mu3, se3], "z": mu3 / se3, "chi2_consistency": chi}
    L.append(f"  GLS over the three 20x30 sets (P3; P6 orig; P6 fu1) with the shared V2 covariance: {mu3*1e4:+.2f}({se3*1e4:.2f})e-4,"
             f" {mu3/se3:.1f} sigma; consistency chi2 {chi:.2f} (2 dof)")
    # (2) <3> at P3: 12x12, 24x24, 20x30 minus 20x60 share one 20x60 reference
    ls = json.loads((R / "lattice_scan.json").read_text())["P3"]
    keys = ["<3>:12x12-20x60", "<3>:24x24-20x60", "<3>:20x30-20x60"]
    dd = np.array([ls[k]["offset_at_gc"]["d"] for k in keys]); ss = np.array([ls[k]["offset_at_gc"]["se"] for k in keys])
    B3 = {b.name: b for b in m.load_file(m.seqbr(20, 60, 0.53, 1, 0.15, "2-23-3"))[0]}["<3>"]
    gc = m.theory(0.53, 1)["g_c"]
    F60 = m.free_energy(B3, 0.53, 1, "rich")
    s60 = []
    for k in keys:
        g0 = ls[k]["offset_at_gc"]["g"]
        j = int(np.argmin(np.abs(B3.g - g0)))
        s60.append(float(np.nanstd(F60[:, j], ddof=1) / np.sqrt(B3.n)))
    C = np.diag(ss ** 2)
    for i in range(3):
        for j in range(3):
            if i != j:
                C[i, j] = s60[i] * s60[j]              # shared 20x60 reference (points at g = 1.10 or 1.122: highly correlated)
    Ci = np.linalg.inv(C); one = np.ones(3)
    mu3 = float(one @ Ci @ dd / (one @ Ci @ one)); se3 = float(1 / np.sqrt(one @ Ci @ one))
    naive = float((dd / ss ** 2).sum() / (1 / ss ** 2).sum()); naive_se = float(1 / np.sqrt((1 / ss ** 2).sum()))
    res["P3_<3>_smaller_minus_20x60"] = {"values": dd.tolist(), "se": ss.tolist(), "se_20x60_part": s60,
                                          "combined_GLS": [mu3, se3], "z": mu3 / se3, "naive_independent": [naive, naive_se]}
    L.append(f"<3> at P3, (12x12, 24x24, 20x30) - 20x60 = {', '.join(f'{a*1e5:+.1f}({b*1e5:.1f})' for a, b in zip(dd, ss))} e-5;"
             f" shared-reference GLS {mu3*1e5:+.2f}({se3*1e5:.2f})e-5 = {mu3/se3:.1f} sigma (naive, as if independent,"
             f" {naive*1e5:+.2f}({naive_se*1e5:.2f}) = {naive/naive_se:.1f} sigma); the 20x60 part of each se is {s60[0]*1e5:.1f}-{max(s60)*1e5:.1f}e-5")
    dump("r2_offset", res, "\n".join(L))


def task_r2_ef_r1():
    """(B r2, 7a) e - f for the 20x20, 20x30 and 20x60 branches at kappa = 0.52 and 0.53, r = 1 (incl. the P6 follow-up)."""
    L, res = ["e - f (operator-count E minus rich f), e-5: band mean over |g - g_c| <= 0.1 (jackknife) and the per-g profile at selected g"], {}
    sets = {0.53: [("20x20 S5a", m.seqbr(20, 20, 0.53, 1, 0.15, "23-2")), ("20x30 S5a", m.seqbr(20, 30, 0.53, 1, 0.15, "23-3")),
                   ("20x60 V1", m.seqbr(20, 60, 0.53, 1, 0.15, "2-23-3"))],
            0.52: [("20x20 S5a", m.seqbr(20, 20, 0.52, 1, 0.15, "23-2")), ("20x30 S5a", m.seqbr(20, 30, 0.52, 1, 0.15, "23-3")),
                   ("20x20 fu1", SEQ / "seqbr_Lx20_Ly20_k0.52_r1_T0.15_23-2_fu1.csv"), ("20x30 fu1", SEQ / "seqbr_Lx20_Ly30_k0.52_r1_T0.15_23-3_fu1.csv"),
                   ("20x60 V2", m.seqbr(20, 60, 0.52, 1, 0.15, "2-23-3"))]}
    for kappa, lst in sets.items():
        gc = m.theory(kappa, 1)["g_c"]
        L.append(f"\nkappa = {kappa}, r = 1, g_c = {gc:.4f}")
        for lab, f in lst:
            if not Path(f).exists():
                L.append(f"  {lab}: missing"); continue
            for b in load_full(f)[0]:
                D, mu, se = ef_profile(b, kappa, 1)
                mask = np.abs(b.g - gc) <= 0.1 + 1e-9
                v, sv, _ = band_stat(D, mask)
                sel = [x for x in (0.3, 0.6, 0.9, gc, 1.2, 1.4) if np.any(np.abs(b.g - x) < 6e-5)]
                prof = {f"{x:g}": [float(mu[np.argmin(np.abs(b.g - x))]), float(se[np.argmin(np.abs(b.g - x))])] for x in sel}
                res[f"{kappa}:{lab}:{b.name}"] = {"band_mean": v, "se": sv, "profile": prof}
                L.append(f"  {lab:10s} {b.name:5s} ({b.n:2d} ch): band {v*1e5:+6.2f}({sv*1e5:5.2f}) | " +
                         " ".join(f"g={k}:{p[0]*1e5:+.1f}({p[1]*1e5:.1f})" for k, p in prof.items()))
    L.append("\nNote: e - f = T s ~ 0 tests the integration path of each branch; an offset that moves e and f together (a state"
             " or lattice effect) is invisible to it.")
    dump("r2_ef_r1", res, "\n".join(L))


def task_r2_tableIV():
    """(B r2, 5 and 4d) Table IV extraction on the O(g^8) crossings (tab3 of pt_order8_tables.json), and the slope means
    for the 14-point set (lambda <= 0.5 at r = 0, lambda <= 0.61 at r = 1), at O(g^6) and O(g^8)."""
    t8 = {f"{x['kappa']:g}_{x['r']}": x for x in json.loads((R3 / "perturbation" / "pt_order8_tables.json").read_text())["tab3"]}
    d = json.loads(AUDIT.read_text())["L12"]
    L, res = [], {}
    for label, lam1 in (("12-point (lambda<=0.5 both r)", 0.5), ("14-point (lambda<=0.5 r=0, <=0.61 r=1)", 0.61)):
        rows = []
        for key, v in d.items():
            k, r = (float(x) for x in key.split("_")); r = int(r)
            g, sg = v["best"]["g"], v["best"]["se"]
            lam = g / (2 + r)
            if lam > (0.5 if r == 0 else lam1) + 1e-12:
                continue
            side = "FM3" if k < 0.5 else "32"
            F3 = 6.0 / ((2 + r) * (3 + r) * (5 + r))
            Ath = F3 / 8 if side == "FM3" else F3 / 4
            B = -PT4[f"r={r}"]["k1_FM3"] if side == "FM3" else PT4[f"r={r}"]["k1_32"]
            ext = lambda gg: (abs(k - 0.5) / gg ** 2 - B * gg ** 2) / Ath
            sA = abs(abs(k - 0.5) / g ** 4 + B) * 2 * g * sg / Ath
            g8 = t8[f"{k:g}_{r}"]["O8"]
            rows.append({"kappa": k, "r": r, "side": side, "lambda": lam, "qmc": ext(g), "se": sA,
                         "O6": ext(v["theory"]["O6"]), "O8": ext(g8), "g8": g8})
        out = {}
        for r in (0, 1):
            for side in ("FM3", "32"):
                s_ = [x for x in rows if x["r"] == r and x["side"] == side]
                if not s_:
                    continue
                w = np.array([1 / x["se"] ** 2 for x in s_])
                mq = float((w * np.array([x["qmc"] for x in s_])).sum() / w.sum()); se = float(1 / np.sqrt(w.sum()))
                m6 = float((w * np.array([x["O6"] for x in s_])).sum() / w.sum())
                m8 = float((w * np.array([x["O8"] for x in s_])).sum() / w.sum())
                out[f"r{r}_{side}"] = {"n": len(s_), "qmc": mq, "se": se, "O6": m6, "O8": m8, "z_vs_O6": (mq - m6) / se, "z_vs_O8": (mq - m8) / se}
        res[label] = {"points": rows, "means": out}
        L.append(f"{label}:")
        for x in sorted(rows, key=lambda x: (x["r"], x["side"], x["kappa"])):
            L.append(f"  {x['side']:4s} r={x['r']} kappa={x['kappa']:<6g} lambda {x['lambda']:.2f}: QMC {x['qmc']:.4f}({x['se']*1e4:.0f})"
                     f"  series O6 {x['O6']:.4f}  O8 {x['O8']:.4f}  (QMC-O8 {(x['qmc']-x['O8'])/x['se']:+.1f} sigma)")
        for kk, o in out.items():
            L.append(f"  mean {kk:7s} n={o['n']}: QMC {o['qmc']:.4f}({o['se']*1e4:.0f}) | series O6 {o['O6']:.4f} ({o['z_vs_O6']:+.1f} sigma)"
                     f" | O8 {o['O8']:.4f} ({o['z_vs_O8']:+.1f} sigma)")
    dump("r2_tableIV", res, "\n".join(L))


# ---------------------------------------------------------------- dispatcher
TASKS = {"energy": task_energy, "covar": task_covar, "cuts": task_cuts, "tableIV": task_tableIV, "power": task_power,
         "reruns": task_reruns, "k047": task_k047, "energy_band": task_energy_band,
         "cuts_trunc": task_cuts_trunc, "lattice_scan": task_lattice_scan,
         "r2_energy": task_r2_energy, "r2_offset": task_r2_offset, "r2_ef_r1": task_r2_ef_r1, "r2_tableIV": task_r2_tableIV}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="all")
    a = ap.parse_args()
    names = list(TASKS) if a.task == "all" else a.task.split(",")
    for n in names:
        print(f"==== {n}", flush=True)
        TASKS[n]()


if __name__ == "__main__":
    main()

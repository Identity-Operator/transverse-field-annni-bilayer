#!/usr/bin/env python3
"""Key result of Paper 1 step 2: the <3> wedge from the branch free-energy crossings.

For every branches_L<L>_k<kappa>_r<r>_free_energy.json the crossing g* of <3> with FM (kappa < 1/2)
or with <2> (kappa > 1/2) is taken with its bootstrap 68% interval. With y = |kappa - 1/2| and
x = g*^2 the perturbative wedge boundaries read

    y = A x + B x^2 + ...,   A = F/8 (FM|<3>) or F/4 (<3>|<2>),   F = 6 / [(2+r)(3+r)(5+r)],

and B follows from the O(g^4) theory (experiments/pt_order4_v3.py --analyze). The errors are on
x only, so the fit is x = a y + b y^2 (weighted), giving A = 1/a and B = -b/a^3. Fits are
repeated for several upper cutoffs in g* to show the approach to the leading slope.

Usage (from vu_work/; needs matplotlib for the diagnostic plot):
    .venv/bin/python experiments/wedge_fit_v3.py [--L 12]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BR = Path("10_results/bilayer_annni_paper_results/reanalysis_v3/sse_validation/branches")
PT = Path("10_results/bilayer_annni_paper_results/reanalysis_v3/perturbation/pt_order4_analysis.json")
OUT = Path("10_results/bilayer_annni_paper_results/reanalysis_v3/wedge")


def F(r):
    return 6.0 / ((2 + r) * (3 + r) * (5 + r))


def load(L, br=BR):
    pts = []
    for f in sorted(br.glob(f"branches_L{L}_k*_r*_free_energy.json")):
        d = json.loads(f.read_text())
        for key, grp in d["groups"].items():
            kappa = float(key.split(",")[0].split("=")[1])
            r = abs(float(key.split(",")[1].split("=")[1]))
            pair = "ascending:<2>par vs ascending:<3>par" if kappa > 0.5 else "ascending:<3>par vs ascending:FM"
            v = grp["pairs"].get(pair)
            if not v or v["g_star"] is None or not np.isfinite(v["g_star"]) or not v["g_star_ci68"]:
                pts.append({"kappa": kappa, "r": r, "side": "32" if kappa > 0.5 else "FM3", "g": None})
                continue
            lo, hi = v["g_star_ci68"]
            pts.append({"kappa": kappa, "r": r, "side": "32" if kappa > 0.5 else "FM3", "g": v["g_star"],
                        "sg": (hi - lo) / 2, "file": f.name})
    return pts


def wfit(y, x, sx, deg):
    X = np.vstack([y ** k for k in range(1, deg + 1)]).T
    W = 1 / sx ** 2
    cov = np.linalg.inv(X.T @ (X * W[:, None]))
    c = cov @ (X.T @ (W * x))
    chi2 = float(np.sum(W * (x - X @ c) ** 2))
    return c, cov, chi2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--L", type=int, default=12)
    ap.add_argument("--dir", default=str(BR), help="directory of the *_free_energy.json files")
    ap.add_argument("--tag", default="", help="suffix of the output files")
    a = ap.parse_args()
    pt = json.loads(PT.read_text())["boundaries"] if PT.exists() else {}
    pts = load(a.L, Path(a.dir))
    res = {"L": a.L, "points": pts, "fits": []}
    print(f"L = {a.L}: {len(pts)} cases")
    for p in sorted(pts, key=lambda p: (p["r"], p["kappa"])):
        g = p["g"]
        print(f"  r={p['r']:g} kappa={p['kappa']:<6} side={p['side']:3s} g*={'none' if g is None else f'{g:.4f} +- {p['sg']:.4f}'}"
              + ("" if g is None else f"   y/x = {abs(p['kappa'] - 0.5) / g ** 2:.5f}"))
    for r in (0.0, 1.0):
        b = pt.get(f"r={int(r)}", {})
        for side, Apred, Bpred in (("FM3", F(r) / 8, -b.get("k1_FM3", np.nan)), ("32", F(r) / 4, b.get("k1_32", np.nan))):
            sel = [p for p in pts if p["r"] == r and p["side"] == side and p["g"] is not None]
            # cutoffs in g*, plus 'lam': g*/(2+r) <= 0.5 (same expansion parameter g/[2(2+r)] <= 1/4 for both r)
            for gmax in (1.0, 1.4, 9.9, "lam"):
                gcut = 0.5 * (2 + r) if gmax == "lam" else gmax
                s = [p for p in sel if p["g"] <= gcut]
                if len(s) < 2:
                    continue
                y = np.array([abs(p["kappa"] - 0.5) for p in s])
                x = np.array([p["g"] ** 2 for p in s])
                sx = np.array([2 * p["g"] * p["sg"] for p in s])
                row = {"r": r, "side": side, "gmax": gmax, "n": len(s), "A_pred": Apred, "B_pred": Bpred}
                c, cov, chi2 = wfit(y, x, sx, 1)
                row.update({"A_lin": 1 / c[0], "A_lin_err": np.sqrt(cov[0, 0]) / c[0] ** 2, "chi2_lin": chi2})
                if len(s) >= 3:
                    c, cov, chi2 = wfit(y, x, sx, 2)
                    A = 1 / c[0]
                    J = np.array([[-1 / c[0] ** 2, 0], [3 * c[1] / c[0] ** 4, -1 / c[0] ** 3]])
                    C = J @ cov @ J.T
                    row.update({"A_quad": A, "A_quad_err": np.sqrt(C[0, 0]), "B_quad": -c[1] / c[0] ** 3,
                                "B_quad_err": np.sqrt(C[1, 1]), "chi2_quad": chi2})
                if np.isfinite(Bpred):     # leading slope with the O(g^4) coefficient B held at theory
                    Ai = y / x - Bpred * x
                    sA = (y / x ** 2 + Bpred) * sx
                    w = 1 / sA ** 2
                    Ab = float(np.sum(w * Ai) / np.sum(w))
                    row.update({"A_Bfix": Ab, "A_Bfix_err": float(1 / np.sqrt(np.sum(w))),
                                "chi2_Bfix": float(np.sum(w * (Ai - Ab) ** 2))})
                res["fits"].append(row)
                msg = (f"  r={r:g} {side:3s} g*<={gmax!s:<4} n={len(s)}: linear A={row['A_lin']:.5f}({row['A_lin_err']:.5f})"
                       f" chi2={row['chi2_lin']:.1f}")
                if "A_quad" in row:
                    msg += (f" | quad A={row['A_quad']:.5f}({row['A_quad_err']:.5f}) B={row['B_quad']:+.5f}({row['B_quad_err']:.5f})"
                            f" chi2={row['chi2_quad']:.1f}")
                if "A_Bfix" in row:
                    msg += f" | B fixed: A={row['A_Bfix']:.5f}({row['A_Bfix_err']:.5f}) chi2={row['chi2_Bfix']:.1f}"
                print(msg + f"   [pred A={Apred:.5f}, B={Bpred:+.5f}]")
    # ratio of leading slopes r = 0 / r = 1 from the quadratic fits over all points (else linear)
    for side, k, gm in [(s_, k_, g_) for s_ in ("FM3", "32") for k_, g_ in (("A_quad", 9.9), ("A_Bfix", 1.4), ("A_Bfix", "lam"))]:
        f0 = [f for f in res["fits"] if f["r"] == 0 and f["side"] == side and f["gmax"] == gm and k in f]
        f1 = [f for f in res["fits"] if f["r"] == 1 and f["side"] == side and f["gmax"] == gm and k in f]
        if f0 and f1:
            A0, A1 = f0[0][k], f1[0][k]
            e0, e1 = f0[0][k + "_err"], f1[0][k + "_err"]
            ratio = A0 / A1
            res[f"ratio_{side}_{k}_gmax{gm}"] = {"estimator": k, "ratio": ratio, "err": ratio * np.hypot(e0 / A0, e1 / A1), "pred": 2.4}
            print(f"  ratio r0/r1 {side}: {ratio:.3f} +- {res[f'ratio_{side}_{k}_gmax{gm}']['err']:.3f} ({k}, g*<={gm}; predicted 2.4)")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"wedge_fit_L{a.L}{a.tag}.json").write_text(json.dumps(res, indent=1, default=float))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.8), sharey=False)
        for ax, side, lab in ((axs[0], "FM3", r"FM$|\langle3\rangle$:  $1/2-\kappa$"),
                              (axs[1], "32", r"$\langle3\rangle|\langle2\rangle$:  $\kappa-1/2$")):
            for r, col in ((0.0, "C0"), (1.0, "C3")):
                s = [p for p in pts if p["r"] == r and p["side"] == side and p["g"] is not None]
                x = np.array([p["g"] ** 2 for p in s]); y = np.array([abs(p["kappa"] - 0.5) for p in s])
                ax.errorbar(x, y, xerr=[2 * p["g"] * p["sg"] for p in s], fmt="o", color=col, ms=4, label=f"QMC $r={r:g}$")
                b = pt.get(f"r={int(r)}", {})
                A = F(r) / (8 if side == "FM3" else 4)
                B = -b.get("k1_FM3", 0) if side == "FM3" else b.get("k1_32", 0)
                xx = np.linspace(0, max(4.0 if side == "32" else 1.6, x.max() if len(x) else 1) * 1.05, 200)
                ax.plot(xx, A * xx, "--", color=col, lw=1, label=f"$O(g^2)$, $r={r:g}$")
                ax.plot(xx, A * xx + B * xx ** 2, "-", color=col, lw=1, label=f"$O(g^4)$, $r={r:g}$")
            ax.set_xlabel(r"$g^{\star 2}$"); ax.set_ylabel(lab); ax.set_ylim(0, None); ax.set_xlim(0, None)
            ax.legend(fontsize=7)
        fig.suptitle(f"<3> wedge from branch free energies, L = {a.L}, T = 0.15 (diagnostic)", fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT / f"wedge_L{a.L}{a.tag}.png", dpi=150)
        print("wrote", OUT / f"wedge_L{a.L}{a.tag}.png")
    except ImportError:
        print("matplotlib not available: no plot")
    print("wrote", OUT / f"wedge_fit_L{a.L}{a.tag}.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Free energies of competing ordered branches by thermodynamic integration in the field.

For H = ... - Gamma sum_i s^x_i, dF/dGamma = -<sum_i s^x_i>, so per spin (J0 = 1, g = Gamma)

    f(g) = f(g0) - int_{g0}^{g} m_x(g') dg'.

At g0 = 0.05 and T = 0.15 every branch is an ordered state whose excitations cost ~2h >= 4 J0,
so its entropy is negligible and f(g0) = E(g0). Each chain is integrated along its own path;
the integral is valid only while the chain stays in one state, so a chain whose dominant
wavevector changes along the path is cut at the first change (reported). Branch free energies
are chain means with standard errors; a crossing g* between two branches is located by linear
interpolation of their difference, with a bootstrap over chains.

Perturbative reference (order g^2, straight domain walls): <3> beats <2> for
kappa - 1/2 < f g^2 / 4 and beats FM for 1/2 - kappa < f g^2 / 8, with
f = 2/(2+r) + 1/(5+r) - 3/(3+r), r = |r_perp|.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def wedge_f(r):
    return 2.0 / (2.0 + r) + 1.0 / (5.0 + r) - 3.0 / (3.0 + r)


def integrate_chain(d):
    """d: one chain sorted by g. Returns g, f(g) (NaN after the first q* change), cut g."""
    g = d["g"].to_numpy()
    mx = d["mx"].to_numpy()
    q = d["qstar_over_pi"].to_numpy()
    f = np.empty_like(g)
    f[0] = d["E_per_spin"].iloc[0]
    f[1:] = f[0] - np.cumsum(0.5 * (mx[1:] + mx[:-1]) * np.diff(g))
    change = np.nonzero(np.abs(q - q[0]) > 1e-9)[0]
    g_cut = float(g[change[0]]) if len(change) else float("nan")
    if len(change):
        f[change[0]:] = np.nan
    return g, f, g_cut


def crossing(g, fa, fb):
    """First sign change of fa - fb on the common finite grid; linear interpolation."""
    ok = np.isfinite(fa) & np.isfinite(fb)
    g, d = g[ok], (fa - fb)[ok]
    s = np.nonzero(np.sign(d[1:]) != np.sign(d[:-1]))[0]
    if not len(s):
        return float("nan")
    i = s[0]
    return float(g[i] - d[i] * (g[i + 1] - g[i]) / (d[i + 1] - d[i]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv")
    ap.add_argument("--kappa", type=float, default=None,
                    help="needed only if the CSV has no kappa column")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    rng = np.random.default_rng(12345)
    if "kappa" not in df.columns:
        if a.kappa is None:
            raise SystemExit("CSV has no kappa column; pass --kappa")
        df["kappa"] = a.kappa
    out = {"csv": a.csv, "groups": {}}
    for (kappa, rp), dr in df.groupby(["kappa", "rperp"]):
        r = abs(float(rp))
        branches = {}
        for (part, start), db in dr.groupby(["part", "start"]):
            name = f"{part}:{start}"
            curves, cuts = [], []
            for _, dc in db.groupby("chain"):
                g, f, gc = integrate_chain(dc.sort_values("g"))
                curves.append(f)
                cuts.append(gc)
            F = np.vstack(curves)
            qmaj = db.groupby("g")["qstar_over_pi"].agg(lambda s: s.mode().iloc[0]).to_numpy()
            branches[name] = {"g": g, "F": F, "cuts": cuts, "qmaj": qmaj,
                              "E": db.groupby("g")["E_per_spin"].mean().to_numpy(),
                              "mx": db.groupby("g")["mx"].mean().to_numpy()}
        # common grid
        grid = sorted(set.intersection(*[set(np.round(b["g"], 6)) for b in branches.values()]))
        grid = np.array(grid)
        tab = pd.DataFrame({"g": grid})
        for name, b in branches.items():
            idx = [int(np.argmin(np.abs(b["g"] - x))) for x in grid]
            Fm = np.nanmean(b["F"][:, idx], axis=0)
            nfin = np.sum(np.isfinite(b["F"][:, idx]), axis=0)
            se = np.nanstd(b["F"][:, idx], axis=0, ddof=1) / np.sqrt(np.maximum(nfin, 1))
            tab[f"F[{name}]"] = Fm
            tab[f"se[{name}]"] = se
            tab[f"q[{name}]"] = b["qmaj"][idx]
            b["grid_idx"] = idx
        names = list(branches)
        pairs = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                A, B = branches[names[i]], branches[names[j]]
                FA, FB = A["F"][:, A["grid_idx"]], B["F"][:, B["grid_idx"]]
                gstar = crossing(grid, np.nanmean(FA, 0), np.nanmean(FB, 0))
                boots = []
                for _ in range(a.n_boot):
                    ia = rng.integers(0, FA.shape[0], FA.shape[0])
                    ib = rng.integers(0, FB.shape[0], FB.shape[0])
                    boots.append(crossing(grid, np.nanmean(FA[ia], 0), np.nanmean(FB[ib], 0)))
                boots = np.array(boots)
                boots = boots[np.isfinite(boots)]
                pairs[f"{names[i]} vs {names[j]}"] = {
                    "g_star": gstar,
                    "g_star_ci68": [float(np.percentile(boots, 16)), float(np.percentile(boots, 84))] if len(boots) else None,
                    "boot_frac_finite": float(len(boots) / a.n_boot)}
        fw = wedge_f(r)
        delta = float(kappa) - 0.5
        pred = {"f": fw,
                "g_min_3_beats_2": math.sqrt(4 * delta / fw) if delta > 0 else None,
                "g_min_3_beats_FM": math.sqrt(8 * -delta / fw) if delta < 0 else None}
        out["groups"][f"kappa={kappa},rperp={rp}"] = {"pairs": pairs, "perturbative_O_g2": pred,
                                 "cuts": {n: b["cuts"] for n, b in branches.items()}}
        pd.set_option("display.width", 250)
        print(f"\n=== kappa={kappa} r_perp={rp} ===")
        print(tab.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
        print("chain cuts (first q* change):", {n: b["cuts"] for n, b in branches.items()})
        print("crossings:", json.dumps(pairs, indent=1))
        print("O(g^2) prediction:", pred)
    p = Path(a.csv).with_name(Path(a.csv).stem + "_free_energy.json")
    p.write_text(json.dumps(out, indent=1, default=float))
    print("wrote", p)


if __name__ == "__main__":
    main()

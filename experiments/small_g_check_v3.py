#!/usr/bin/env python3
"""Small-g test of second-order perturbation theory on the branch energies.

For a straight-wall structure with favourable stacking, e(g) = e0 - (g^2/2) <1/h> + O(g^4), with the
local field h = 2 + r + a_{+1} + a_{-1} - kappa (a_{+2} + a_{-2}) evaluated at the actual kappa:
  FM:  h = 4 + r - 2 kappa                                   e0 = -2 + kappa - r/2
  <2>: h = 2 + r + 2 kappa                                   e0 = -1 - kappa - r/2
  <3>: h = 2 + r (two wall sites), 4 + r + 2 kappa (centre)  e0 = -4/3 - kappa/3 - r/2
The measured energies at g <= g_max are fitted to e - e0 = a g^2 + b g^4 (weighted by the
chain standard errors); a must equal -<1/h>/2. Thermal corrections at T = 0.15 are below 1e-11.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def exact(start, kappa, r):
    if start.startswith("FM"):
        return -2 + kappa - r / 2, 1 / (4 + r - 2 * kappa)
    if start.startswith("<2>"):
        return -1 - kappa - r / 2, 1 / (2 + r + 2 * kappa)
    if start.startswith("<3>"):
        return -4 / 3 - kappa / 3 - r / 2, (2 / (2 + r) + 1 / (4 + r + 2 * kappa)) / 3
    raise ValueError(start)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--g-max", type=float, default=0.5)
    a = ap.parse_args()
    rows = []
    for f in a.csvs:
        d = pd.read_csv(f)
        d = d[(d["part"] == "ascending") & (d["g"] <= a.g_max + 1e-9)]
        for (kappa, rp, start), db in d.groupby(["kappa", "rperp", "start"]):
            r = abs(float(rp))
            e0, inv_h = exact(start, float(kappa), r)
            s = db.groupby("g")["E_per_spin"].agg(["mean", "std", "count"])
            g = s.index.to_numpy()
            y = s["mean"].to_numpy() - e0
            se = (s["std"] / np.sqrt(s["count"])).to_numpy()
            X = np.vstack([g ** 2, g ** 4]).T
            W = 1 / se ** 2
            cov = np.linalg.inv(X.T @ (X * W[:, None]))
            coef = cov @ (X.T @ (W * y))
            chi2 = float(np.sum(W * (y - X @ coef) ** 2))
            a2, a2_se = coef[0], np.sqrt(cov[0, 0])
            pred = -0.5 * inv_h
            rows.append({"kappa": float(kappa), "rperp": float(rp), "start": start, "e0": e0,
                         "a_fit": a2, "a_se": a2_se, "a_pred": pred,
                         "z": (a2 - pred) / a2_se, "rel_dev": (a2 - pred) / abs(pred),
                         "b_fit_g4": coef[1], "b_se": np.sqrt(cov[1, 1]),
                         "chi2_dof": chi2 / max(len(g) - 2, 1), "n_g": len(g),
                         "e_minus_e0_at_gmin": float(y[0]), "se_at_gmin": float(se[0])})
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(t.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    out = Path(a.csvs[0]).with_name(f"small_g_check_gmax{a.g_max}.json")
    out.write_text(json.dumps(rows, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()

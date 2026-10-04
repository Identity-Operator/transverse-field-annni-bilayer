#!/usr/bin/env python3
"""Thermal excitations of the rows next to the walls (Supplemental Material, Sec. S3, error budget).

With its environment frozen, the row next to a wall of a domain of three or more sites is a transverse-field Ising
chain (r = 0, coupling J_x = 1, field g) or a two-leg ladder with equal leg and rung couplings (r = 1).

r = 0: the free-fermion thermal free energy per site of the chain, f_T = -(T/pi) int_0^pi ln(1 + exp(-eps_k/T)) dk with
eps_k = 2 sqrt(1 + g^2 - 2 g cos k). It neglects the pinning of the row by the walls and therefore bounds the effect
from above. Rows of this kind make up 0, 2/5 and 2/3 of the rows of <2>, <23> and <3>, so the thermal contributions to
Delta F_2 = F<23> - F<2> and Delta F_3 = F<23> - F<3> are (2/5) f_T and (2/5 - 2/3) f_T.

r = 1: the kink energy of the ladder, E0(antiperiodic) - E0(periodic) along the legs, from exact diagonalization of
2 x L ladders (L = 8, 9); the chain value 2(1 - g) checks the method.

Output: reanalysis_v3/perturbation/edge_rows_thermal.{json,txt}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.integrate import quad

OUT = Path("10_results/bilayer_annni_paper_results/reanalysis_v3/perturbation")
TESTS_R0 = {"0.52": 0.5997, "0.53": 0.7179, "0.545": 0.8525, "0.5625": 0.9733}   # g_c of the r = 0 test points
TESTS_R1 = {"0.52": 0.9351, "0.53": 1.1222, "0.545": 1.3369}                    # g_c of the r = 1 test points


def chain_fT(g, T):
    eps = lambda k: 2.0 * np.sqrt(1.0 + g * g - 2.0 * g * np.cos(k))
    return -(T / np.pi) * quad(lambda k: np.log1p(np.exp(-eps(k) / T)), 0.0, np.pi)[0]


def ham(L, legs, g, twist):
    n = L * legs
    idx = np.arange(2 ** n)
    s = ((idx[:, None] >> np.arange(n)) & 1) * 2 - 1
    diag = np.zeros(2 ** n)
    for leg in range(legs):
        for x in range(L):
            i, j = leg * L + x, leg * L + (x + 1) % L
            sign = -1 if (twist and x == L - 1) else 1
            diag -= sign * s[:, i] * s[:, j]
    if legs == 2:
        for x in range(L):
            diag -= s[:, x] * s[:, L + x]
    rows, cols, vals = [idx], [idx], [diag]
    for i in range(n):
        rows.append(idx); cols.append(idx ^ (1 << i)); vals.append(-g * np.ones(2 ** n))
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(2 ** n, 2 ** n))


def kink(L, legs, g):
    e0 = lambda H: sla.eigsh(H, k=1, which="SA")[0][0]
    return float(e0(ham(L, legs, g, True)) - e0(ham(L, legs, g, False)))


def main():
    res = {"r0_free_chain": {}, "r1_ladder_kink": {}, "chain_kink_check": {}}
    lines = []
    for k, g in TESTS_R0.items():
        row = {"g_c": g}
        for T in (0.15, 0.10):
            f = chain_fT(g, T)
            row[f"T{T}"] = {"f_T": f, "dF2_thermal": 0.4 * f, "dF3_thermal": (0.4 - 2.0 / 3.0) * f}
        res["r0_free_chain"][k] = row
        lines.append(f"r=0 kappa={k} g_c={g}: T=0.15 dF2 {row['T0.15']['dF2_thermal']*1e4:+.2f}e-4, dF3 "
                     f"{row['T0.15']['dF3_thermal']*1e4:+.2f}e-4; T=0.10 dF2 {row['T0.1']['dF2_thermal']*1e4:+.2f}e-4, "
                     f"dF3 {row['T0.1']['dF3_thermal']*1e4:+.2f}e-4")
    for g in (0.6, 0.7179):
        res["chain_kink_check"][str(g)] = {"ED_L16": kink(16, 1, g), "exact": 2 * (1 - g)}
        lines.append(f"chain check g={g}: ED kink {res['chain_kink_check'][str(g)]['ED_L16']:.3f}, exact {2*(1-g):.3f}")
    for k, g in TESTS_R1.items():
        ks = {f"L{L}": kink(L, 2, g) for L in (8, 9)}
        res["r1_ladder_kink"][k] = {"g_c": g, **ks, "boltzmann_T0.15": float(np.exp(-ks["L9"] / 0.15))}
        lines.append(f"r=1 kappa={k} g_c={g}: ladder kink energy {ks['L8']:.3f} (L=8), {ks['L9']:.3f} (L=9); "
                     f"exp(-D/T) at T=0.15 = {np.exp(-ks['L9']/0.15):.1e}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "edge_rows_thermal.json").write_text(json.dumps(res, indent=1))
    (OUT / "edge_rows_thermal.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

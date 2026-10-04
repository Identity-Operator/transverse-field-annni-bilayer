#!/usr/bin/env python3
"""Validate the SSE engine on the 2D transverse-field Ising model (referee B4).

One square layer (nlayers=1), kappa = 0, r_perp = 0, i.e. H = -sum_<ij> s^z s^z - g sum s^x,
whose quantum critical point is g_c = 3.04438 (Bloete and Deng, PRE 66, 066110 (2002)). With
beta = L (dynamic exponent z = 1), the Binder cumulant

    U4 = 1 - <m^4> / (3 <m^2>^2),   m = FM template magnetization,

and xi/L, with the second-moment length along y

    xi = [2 sin(pi/L)]^-1 sqrt(S(0)/S(2 pi/L) - 1),   S(q) = layer structure factor Sl2,

are scale invariant at g_c, so the curves for different L cross there, up to corrections
that shrink with L.

Every chain anneals down through the g grid (--g-max to --g-min), thermalizing --therm and
measuring --meas sweeps at each g. Chain averages are combined into U4 and xi/L with
jackknife errors over chains; crossings between consecutive sizes are found by linear
interpolation of the difference on the grid, with a jackknife error.

With --beta B the inverse temperature is fixed instead (finite-T thermal Ising line; at
T = 0.15 the transition lies between Gamma = 3.0, where Hesselmann and Wessel, PRB 93, 155157
(2016), find T_c = 0.2977(9), and g_c(T=0) = 3.04438); the U4 crossings then converge to the
2D Ising value 1 - 1.16793/3 = 0.6107.

Usage (from vu_work/):
    .venv/bin/python experiments/sse_tfim2d_v3.py --sizes 8,12,16,24 --chains 8 --threads 6
    .venv/bin/python experiments/sse_tfim2d_v3.py --beta 6.6666667 --sizes 16,24,32,48 \
        --g-min 2.98 --g-max 3.06 --n-g 9 --chains 6 --threads 6 --tag T0.15
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import math
import time

import numpy as np

from phase_pipeline.io_utils import package_version, write_json, write_rows
from phase_pipeline.qmc.sse import bilayer_lattice, class_couplings, initial_spins, run_sse_chains

G_C = 3.04438


def u4_xi(m2, m4, s0, s1, L):
    u4 = 1.0 - m4 / (3.0 * m2 ** 2)
    ratio = s0 / s1 - 1.0
    xi = math.sqrt(ratio) / (2.0 * math.sin(math.pi / L)) if ratio > 0 else float("nan")
    return u4, xi / L


def jackknife(chain_vals, fn):
    """chain_vals: (C, k) per-chain averages of the k inputs of fn."""
    C = chain_vals.shape[0]
    full = fn(*chain_vals.mean(0))
    loo = np.array([fn(*np.delete(chain_vals, c, axis=0).mean(0)) for c in range(C)])
    err = np.sqrt((C - 1) / C * np.sum((loo - loo.mean(0)) ** 2, axis=0))
    return np.asarray(full), err, loo


def crossing(g, a, b):
    d = np.asarray(a) - np.asarray(b)
    for i in range(len(g) - 1):
        if np.isfinite(d[i]) and np.isfinite(d[i + 1]) and d[i] * d[i + 1] <= 0 and d[i] != d[i + 1]:
            return float(g[i] - d[i] * (g[i + 1] - g[i]) / (d[i + 1] - d[i]))
    return float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="10_results/bilayer_annni_paper_results")
    ap.add_argument("--sizes", default="8,12,16,24")
    ap.add_argument("--g-min", type=float, default=2.8)
    ap.add_argument("--g-max", type=float, default=3.3)
    ap.add_argument("--n-g", type=int, default=11)
    ap.add_argument("--chains", type=int, default=8)
    ap.add_argument("--therm", type=int, default=2000)
    ap.add_argument("--meas", type=int, default=10000)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--beta", type=float, default=None, help="fixed inverse temperature (default beta = L)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    out = Path(a.results_dir) / "reanalysis_v3" / "sse_validation" / ("tfim2d" + (f"_{a.tag}" if a.tag else ""))
    out.mkdir(parents=True, exist_ok=True)
    sizes = [int(s) for s in a.sizes.split(",")]
    grid = np.linspace(a.g_max, a.g_min, a.n_g)          # annealed from large to small g
    rng = np.random.default_rng(a.seed)
    Jc = class_couplings(1.0, 1.0, 0.0, 0.0)
    rows, curves, timing = [], {}, {}
    for L in sizes:
        beta = float(L) if a.beta is None else a.beta
        cache = out / f"chains_L{L}.npz"
        if cache.exists() and np.allclose(np.load(cache)["grid"], grid) and float(np.load(cache)["beta"]) == beta:
            z = np.load(cache)
            vals_all, E_all, mx_all = z["vals"], z["E"], z["mx"]
            timing[L] = {"wall_s": float(z["wall"]), "mean_n_ops": float(z["n_ops"]), "from_cache": True}
        else:
            lat = bilayer_lattice(L, L, 1)
            C = a.chains
            init = np.stack([initial_spins(lat, 0, rng) for _ in range(C)])
            seeds = np.uint64(1000 * L) + np.arange(1, C + 1, dtype=np.uint64)
            t0 = time.time()
            res = run_sse_chains(lat, np.tile(Jc, (C, 1)), np.tile(grid, (C, 1)), beta, init, seeds,
                                 a.therm, a.meas, bin_size=100, threads=a.threads)
            wall = time.time() - t0
            vals_all = np.array([[[rc[k]["mFM2"], rc[k]["mFM4"], rc[k]["Sl2"][0], rc[k]["Sl2"][1]]
                                  for k in range(len(grid))] for rc in res])        # (C, K, 4)
            E_all = np.array([[rc[k]["E_per_spin"] for k in range(len(grid))] for rc in res])
            mx_all = np.array([[rc[k]["mx"] for k in range(len(grid))] for rc in res])
            n_ops = float(np.mean([r["n_ops"] for rc in res for r in rc]))
            np.savez(cache, grid=grid, beta=beta, vals=vals_all, E=E_all, mx=mx_all, wall=wall, n_ops=n_ops)
            timing[L] = {"wall_s": wall, "sweeps_per_chain": int(len(grid) * (a.therm + a.meas)), "mean_n_ops": n_ops}
        C = vals_all.shape[0]
        U, Ue, X, Xe = [], [], [], []
        for k, gv in enumerate(grid):
            f = lambda m2, m4, s0, s1: np.array(u4_xi(m2, m4, s0, s1, L))
            full, err, _ = jackknife(vals_all[:, k, :], f)
            e = E_all[:, k]
            U.append(full[0]); Ue.append(err[0]); X.append(full[1]); Xe.append(err[1])
            rows.append({"L": L, "beta": beta, "chains": C, "g": float(gv), "U4": full[0], "U4_err": err[0],
                         "xi_over_L": full[1], "xi_over_L_err": err[1], "E_per_spin": float(e.mean()),
                         "E_err": float(e.std(ddof=1) / math.sqrt(C)), "mx": float(mx_all[:, k].mean())})
        write_rows(out / "tfim2d_points.csv", rows)          # incremental
        curves[L] = {"g": grid[::-1].tolist(), "U4": U[::-1], "U4_err": Ue[::-1], "xi_over_L": X[::-1],
                     "xi_over_L_err": Xe[::-1], "_vals": vals_all[:, ::-1, :]}
        print(f"L={L:3d} wall={timing[L]['wall_s']:7.1f}s <n>={timing[L]['mean_n_ops']:.0f}  "
              f"U4={np.round(curves[L]['U4'], 3).tolist()}", flush=True)
    g_up = grid[::-1]
    cross = []
    for L1, L2 in zip(sizes[:-1], sizes[1:]):
        v1, v2 = curves[L1]["_vals"], curves[L2]["_vals"]
        C = min(len(v1), len(v2))

        def est(idx):
            def curve(v, L):
                m = v[idx].mean(0)
                return np.array([u4_xi(*m[k], L) for k in range(len(g_up))])
            c1, c2 = curve(v1, L1), curve(v2, L2)
            return crossing(g_up, c1[:, 0], c2[:, 0]), crossing(g_up, c1[:, 1], c2[:, 1])
        full = est(np.arange(C))
        loo = np.array([est(np.delete(np.arange(C), c)) for c in range(C)])
        err = np.sqrt((C - 1) / C * np.nansum((loo - np.nanmean(loo, 0)) ** 2, axis=0))
        cross.append({"L1": L1, "L2": L2, "g_cross_U4": full[0], "g_cross_U4_err": float(err[0]),
                      "g_cross_xiL": full[1], "g_cross_xiL_err": float(err[1])})
        print(f"crossing L={L1},{L2}: U4 {full[0]:.4f} +- {err[0]:.4f}   xi/L {full[1]:.4f} +- {err[1]:.4f}"
              f"   (g_c = {G_C})", flush=True)
    for L in curves:
        curves[L].pop("_vals")
    write_rows(out / "tfim2d_points.csv", rows)
    write_json(out / "tfim2d_summary.json", {"g_c_reference": G_C, "crossings": cross, "timing": timing,
                                              "curves": curves, "args": vars(a),
                                              "numba": package_version("numba"), "numpy": package_version("numpy")})


if __name__ == "__main__":
    main()

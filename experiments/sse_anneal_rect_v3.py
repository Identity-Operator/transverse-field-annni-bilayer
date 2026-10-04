#!/usr/bin/env python3
"""Descending anneals from the paramagnet on rectangular Lx x Ly bilayers (referee C2, point B3).

Same engine and protocol as the descending part of sse_branches_v3.py (random start at --g-top, steps of
--dg down to 0.1, then g0 = 0.05; --therm + --meas sweeps per field), but on Lx x Ly lattices and with the
whole layer structure factor S(q), q = 2 pi n / Ly, saved at every field. With Ly = 60, the structures
<3> (n = 10), <23> (n = 12) and <2> (n = 15) and the mean-field instability wavevectors at kappa = 0.625
(n ~ 11) and 0.75 (n ~ 12) are all resolved, so the wavevector that first grows and the final structure
can be told apart.

Outputs (in <results-dir>/reanalysis_v3/sse_validation/anneal_rect/):
  anneal_Lx<Lx>_Ly<Ly>_k<kappa>_r<r>_T<T><suffix>.csv   per chain and field: E, mx, pair correlations, q*, S at pi/3, 2pi/5, pi/2
  ....npz                                                Sl2[chain, field, n] and the field grid
  ....json                                               arguments, seeds, wall time
Seeds: derived from the output tag (crc32), so different cases use different streams.

Usage (from vu_work/):
  $HOME/.venvs/annni-py311/bin/python experiments/sse_anneal_rect_v3.py --Lx 12 --Ly 60 --kappa 0.75 --rperp 0
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import time
import zlib

import numpy as np

from phase_pipeline.io_utils import package_version, write_json, write_rows
from phase_pipeline.qmc.sse import PAIR_CLASSES, bilayer_lattice, class_couplings, initial_spins, run_sse_chains


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="10_results/bilayer_annni_paper_results")
    ap.add_argument("--Lx", type=int, required=True)
    ap.add_argument("--Ly", type=int, required=True)
    ap.add_argument("--kappa", type=float, required=True)
    ap.add_argument("--rperp", type=float, required=True)
    ap.add_argument("--T", type=float, default=0.15)
    ap.add_argument("--g-top", type=float, default=None, help="default 3.5 (r = 0) or 4.5 (r > 0), as in step 2")
    ap.add_argument("--dg", type=float, default=0.1)
    ap.add_argument("--chains", type=int, default=6)
    ap.add_argument("--therm", type=int, default=2000)
    ap.add_argument("--meas", type=int, default=3000)
    ap.add_argument("--threads", type=int, default=9)
    ap.add_argument("--tag-suffix", default="")
    a = ap.parse_args()
    out = Path(a.results_dir) / "reanalysis_v3" / "sse_validation" / "anneal_rect"
    out.mkdir(parents=True, exist_ok=True)
    g_top = a.g_top if a.g_top is not None else (3.5 if a.rperp == 0 else 4.5)
    grid = np.round(np.concatenate([np.arange(g_top, 0.1 - 1e-9, -a.dg), [0.05]]), 4)
    tag = f"anneal_Lx{a.Lx}_Ly{a.Ly}_k{a.kappa:g}_r{a.rperp:g}_T{a.T:g}{a.tag_suffix}"
    seed0 = zlib.crc32(tag.encode()) % 100_000_000
    lat = bilayer_lattice(a.Lx, a.Ly, 2)
    rng = np.random.default_rng(seed0)
    C = a.chains
    init = np.stack([initial_spins(lat, 0, rng) for _ in range(C)])
    t0 = time.time()
    res = run_sse_chains(lat, np.tile(class_couplings(1, 1, a.kappa, a.rperp), (C, 1)), np.tile(grid, (C, 1)),
                         1.0 / a.T, init, seed0 + 1 + np.arange(C), a.therm, a.meas, bin_size=100, threads=a.threads)
    wall = time.time() - t0
    Ly = a.Ly
    q = 2 * np.pi * np.arange(Ly) / Ly
    at = lambda S, n: float(S[n]) if 0 <= n < Ly and abs(Ly * n / Ly - n) < 1e-12 else float("nan")
    rows, S_all = [], np.zeros((C, len(grid), Ly))
    for c, rc in enumerate(res):
        for j, r in enumerate(rc):
            S = np.asarray(r["Sl2"], float)
            S_all[c, j] = S
            iq = int(np.argmax(S))
            rows.append({"Lx": a.Lx, "Ly": a.Ly, "kappa": a.kappa, "rperp": a.rperp, "T": a.T, "part": "descending",
                         "start": "random", "chain": c, "g": r["Gamma"], "E_per_spin": r["E_per_spin"], "mx": r["mx"],
                         **{k: r[k] for k in PAIR_CLASSES},
                         "qstar_over_pi": float(min(q[iq], 2 * np.pi - q[iq]) / np.pi), "nstar": min(iq, Ly - iq),
                         "Sl2_qpi3": at(S, Ly // 6) if Ly % 6 == 0 else float("nan"),
                         "Sl2_q2pi5": at(S, Ly // 5) if Ly % 5 == 0 else float("nan"),
                         "Sl2_qpi2": at(S, Ly // 4) if Ly % 4 == 0 else float("nan"),
                         "n_ops": r["n_ops"]})
    write_rows(out / f"{tag}.csv", rows)
    np.savez_compressed(out / f"{tag}.npz", Sl2=S_all, g=grid)
    write_json(out / f"{tag}.json", {"args": vars(a), "g_top": g_top, "seed0": seed0, "grid": grid.tolist(),
                                     "wall_s": wall, "numba": package_version("numba")})
    fin = [r for r in rows if abs(r["g"] - 0.05) < 1e-9]
    print(f"{tag}: {C} chains x {len(grid)} g in {wall:.0f}s; final n* = {[r['nstar'] for r in fin]}, "
          f"E = {[round(r['E_per_spin'], 4) for r in fin]}", flush=True)


if __name__ == "__main__":
    main()

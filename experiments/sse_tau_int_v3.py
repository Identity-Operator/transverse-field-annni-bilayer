#!/usr/bin/env python3
"""Integrated autocorrelation times of the energy and of m_x on template branches (referee C2 round 2, B 7c).

Chains are started in a template at g0 = 0.05 and stepped up the production grid (2000 + 3000 sweeps per field)
to the test field g_c, where they measure for --long sweeps in bins of --bin sweeps. From the bin series of the
operator count n (energy) and the flip-operator count n_f (m_x) we estimate the integrated autocorrelation time
    tau_int = bin * (1/2 + sum_t rho_bin(t)),   summed up to the first t where t >= 6 * tau (automatic window),
per chain, and report the median over chains and max, in sweeps. The production runs measure 3000 sweeps per field.

Cases: kappa = 0.53, r = 1 at g_c = 1.1222 for <3> and <23> on 20 x 30 and 20 x 60, <3> on 12 x 12;
the 12-spin crossover of the end-to-end ED test (2 x 6 layer, kappa = 0.3, FM, g = 1.5).
Output: reanalysis_v3/sse_validation/tau_int.{json,txt}
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
import time

import numpy as np

from phase_pipeline.qmc.sse import bilayer_lattice, class_couplings, initial_spins, run_sse_chains
from sse_seq_branches_v3 import column

OUT = Path("10_results/bilayer_annni_paper_results/reanalysis_v3/sse_validation")


def tau_int(x, bin_size):
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    var = np.dot(x, x) / n
    if var == 0:
        return float("nan")
    tau = 0.5
    for t in range(1, n // 4):
        rho = np.dot(x[:-t], x[t:]) / ((n - t) * var)
        tau += rho
        if t >= 6 * tau:
            break
    return bin_size * max(tau, 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--long", type=int, default=20000)
    ap.add_argument("--bin", type=int, default=10)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    gc = 1.1222
    grid = np.round(np.concatenate([[0.05], np.arange(0.1, 1.0 + 1e-9, 0.1), np.arange(1.02, 1.12 + 1e-9, 0.02), [gc]]), 4)
    cases = [("<3>", (3,), 12, 12), ("<3>", (3,), 20, 30), ("<23>", (2, 3), 20, 30),
             ("<3>", (3,), 20, 60), ("<23>", (2, 3), 20, 60)]
    rows, lines = [], []
    for name, seq, Lx, Ly in cases:
        lat = bilayer_lattice(Lx, Ly, 2)
        col = column(list(seq))
        sp = np.array([col[int(y) % len(col)] for y in lat["ycoord"]], dtype=np.int8)
        init = np.tile(sp, (a.chains, 1))
        therm = np.full(len(grid), 2000)
        meas = np.full(len(grid), 3000)
        meas[-1] = a.long
        t0 = time.time()
        res = run_sse_chains(lat, np.tile(class_couplings(1, 1, 0.53, 1.0), (a.chains, 1)),
                             np.tile(grid, (a.chains, 1)), 1 / 0.15, init, 8101 + np.arange(a.chains),
                             therm, meas, bin_size=a.bin, threads=a.threads)
        tn = [tau_int(res[c][-1]["bins"][:, 0], a.bin) for c in range(a.chains)]
        tf = [tau_int(res[c][-1]["bins"][:, 1], a.bin) for c in range(a.chains)]
        row = {"structure": name, "lattice": f"{Lx}x{Ly}", "kappa": 0.53, "r": 1, "g": gc, "chains": a.chains,
               "long_sweeps": a.long, "bin": a.bin, "tau_E": tn, "tau_mx": tf, "wall_s": time.time() - t0}
        rows.append(row)
        line = (f"{name} {Lx}x{Ly} at g_c: tau_int(E) median {np.median(tn):.1f}, max {np.max(tn):.1f} sweeps; "
                f"tau_int(m_x) median {np.median(tf):.1f}, max {np.max(tf):.1f} sweeps ({time.time() - t0:.0f}s)")
        print(line, flush=True)
        lines.append(line)
        (OUT / "tau_int.json").write_text(json.dumps(rows, indent=1))
        (OUT / "tau_int.txt").write_text("\n".join(lines) + "\n")
    # the 12-spin crossover of the ED test
    lat = bilayer_lattice(2, 6, 1)
    g12 = np.round(np.concatenate([[0.05], np.arange(0.1, 1.5 + 1e-9, 0.1)]), 4)
    init = np.stack([initial_spins(lat, 1, np.random.default_rng(1)) for _ in range(8)])
    meas = np.full(len(g12), 3000)
    meas[-1] = 40000
    res = run_sse_chains(lat, np.tile(class_couplings(1, 1, 0.3, 0.0), (8, 1)), np.tile(g12, (8, 1)), 1 / 0.15, init,
                         8201 + np.arange(8), np.full(len(g12), 2000), meas, bin_size=a.bin, threads=a.threads)
    tn = [tau_int(res[c][-1]["bins"][:, 0], a.bin) for c in range(8)]
    tf = [tau_int(res[c][-1]["bins"][:, 1], a.bin) for c in range(8)]
    rows.append({"structure": "FM (ED test)", "lattice": "2x6 layer", "kappa": 0.3, "r": 0, "g": 1.5, "chains": 8,
                 "tau_E": tn, "tau_mx": tf})
    line = (f"2x6 layer kappa=0.3 g=1.5 (ED-test crossover): tau_int(E) median {np.median(tn):.1f}, max {np.max(tn):.1f}; "
            f"tau_int(m_x) median {np.median(tf):.1f}, max {np.max(tf):.1f} sweeps")
    print(line, flush=True)
    lines.append(line)
    (OUT / "tau_int.json").write_text(json.dumps(rows, indent=1))
    (OUT / "tau_int.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

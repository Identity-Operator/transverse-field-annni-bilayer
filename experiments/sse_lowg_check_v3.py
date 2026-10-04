#!/usr/bin/env python3
"""Low-field ground-state check of the annealed SSE sampler (gate G1).

At g = 0.05 and T = 0.15 the bilayer (Jx = Jy = 1, J2 = kappa, J_perp = r_perp >= 0) is
essentially in its classical ground state, whose energy per spin is

    E0 = min(-2 + kappa, -1 - kappa) - r_perp / 2

(uniform order vs. the antiphase <2>; degenerate with <3> at kappa = 1/2). Quantum
corrections are O(g^2), about -g^2/(2h) with h the local field of the ordered state, i.e.
~1e-4, and thermal excitations at T = 0.15 are negligible away from kappa = 1/2.

Every chain anneals from g = --g-start (r_perp = 0) or --g-start-coupled (r_perp > 0; the
coupled bilayer orders above g = 3, so a start at 3 lets domain walls nucleate and freeze) to --g in --n-steps steps (--anneal-therm sweeps per
step, no measurement), then thermalizes --therm and measures --meas sweeps at g. EVERY chain
is compared with E0 individually (a single trapped chain is what we look for), and any
(kappa, r_perp, L) whose worst chain lies above E0 + --tol is flagged.

Usage (from vu_work/):
    .venv/bin/python experiments/sse_lowg_check_v3.py --sizes 12,24 --threads 6
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import time

import numpy as np

from phase_pipeline.io_utils import package_version, write_json, write_rows
from phase_pipeline.qmc.sse import bilayer_lattice, class_couplings, initial_spins, run_sse_chains


def e0(kappa, rperp):
    return min(-2.0 + kappa, -1.0 - kappa) - rperp / 2.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="10_results/bilayer_annni_paper_results")
    ap.add_argument("--sizes", default="12,24")
    ap.add_argument("--kappas", default="0,0.25,0.5,0.75,1.0,1.5")
    ap.add_argument("--rperps", default="0,1")
    ap.add_argument("--g", type=float, default=0.05)
    ap.add_argument("--T", type=float, default=0.15)
    ap.add_argument("--g-start", type=float, default=3.0)
    ap.add_argument("--g-start-coupled", type=float, default=5.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--n-steps", type=int, default=16)
    ap.add_argument("--anneal-therm", type=int, default=2000)
    ap.add_argument("--therm", type=int, default=5000)
    ap.add_argument("--meas", type=int, default=5000)
    ap.add_argument("--chains", type=int, default=6)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--tol", type=float, default=0.005)
    a = ap.parse_args()
    out = Path(a.results_dir) / "reanalysis_v3" / "sse_validation" / ("lowg" + (f"_{a.tag}" if a.tag else ""))
    out.mkdir(parents=True, exist_ok=True)
    beta = 1.0 / a.T
    therm = np.full(a.n_steps, a.anneal_therm); therm[-1] = a.therm
    meas = np.zeros(a.n_steps, dtype=int); meas[-1] = a.meas
    rng = np.random.default_rng(20260929)
    rows, flags = [], []
    done = set()
    if (out / "lowg_check.csv").exists():       # resume after a crash: keep finished combinations
        from phase_pipeline.io_utils import read_rows
        for r in read_rows(out / "lowg_check.csv"):
            r = {k2: (float(v) if k2 not in ("chain_E",) else v) for k2, v in r.items()}
            rows.append(r)
            done.add((int(r["L"]), round(r["kappa"], 6), round(r["rperp"], 6)))
            if r["excess_max"] > a.tol:
                flags.append({kk: r[kk] for kk in ("L", "kappa", "rperp", "E0", "E_max", "excess_max",
                                                    "n_chains_above_tol")})
    for L in [int(s) for s in a.sizes.split(",")]:
        lat = bilayer_lattice(L, L, 2)
        for k in [float(s) for s in a.kappas.split(",")]:
            for rp in [float(s) for s in a.rperps.split(",")]:
                if (L, round(k, 6), round(rp, 6)) in done:
                    continue
                C = a.chains
                gs = a.g_start if rp == 0 else a.g_start_coupled
                sched = np.linspace(gs, a.g, a.n_steps)
                Jc = class_couplings(1.0, 1.0, k, rp)
                init = np.stack([initial_spins(lat, 0, rng) for _ in range(C)])
                seeds = np.uint64(int(1e6 * L + 1e4 * k + 100 * rp)) + np.arange(1, C + 1, dtype=np.uint64)
                t0 = time.time()
                res = run_sse_chains(lat, np.tile(Jc, (C, 1)), np.tile(sched, (C, 1)), beta, init, seeds,
                                     therm, meas, bin_size=100, threads=a.threads)
                wall = time.time() - t0
                E = np.array([rc[-1]["E_per_spin"] for rc in res])
                ref = e0(k, rp)
                excess = E - ref
                row = {"L": L, "kappa": k, "rperp": rp, "g": a.g, "T": a.T, "g_start": gs, "E0": ref,
                       "E_mean": float(E.mean()), "E_min": float(E.min()), "E_max": float(E.max()),
                       "excess_mean": float(excess.mean()), "excess_max": float(excess.max()),
                       "n_chains_above_tol": int(np.sum(excess > a.tol)), "chains": C,
                       "Cx": float(np.mean([rc[-1]["Cx"] for rc in res])),
                       "Cy": float(np.mean([rc[-1]["Cy"] for rc in res])),
                       "C2": float(np.mean([rc[-1]["C2"] for rc in res])),
                       "Cperp": float(np.mean([rc[-1]["Cperp"] for rc in res])),
                       "mean_n_ops": float(np.mean([rc[-1]["n_ops"] for rc in res])), "wall_s": wall,
                       "chain_E": ";".join(f"{v:.6f}" for v in E)}
                rows.append(row)
                write_rows(out / "lowg_check.csv", rows)   # incremental, so a partial run is usable
                if row["excess_max"] > a.tol:
                    flags.append({kk: row[kk] for kk in ("L", "kappa", "rperp", "E0", "E_max", "excess_max",
                                                          "n_chains_above_tol")})
                print(f"L={L:2d} kappa={k:<5} r={rp:<3} gs={gs:g} E0={ref:+.4f}  E={E.mean():+.5f} "
                      f"[{E.min():+.5f},{E.max():+.5f}]  excess max={excess.max():+.2e}  "
                      f"above tol: {row['n_chains_above_tol']}/{C}  ({wall:.0f}s)", flush=True)
    write_rows(out / "lowg_check.csv", rows)
    write_json(out / "lowg_summary.json", {"args": vars(a), "flags": flags, "n_combinations": len(rows),
                                            "max_excess_overall": float(max(r["excess_max"] for r in rows)),
                                            "numba": package_version("numba")})
    print("FLAGS:", flags if flags else "none")


if __name__ == "__main__":
    main()

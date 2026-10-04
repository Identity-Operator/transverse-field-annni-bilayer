#!/usr/bin/env python3
"""Template-seeded branches for the equilibration protocol (prototype of S1).

At kappa = 0.75 the annealed chains end in <3> instead of the ground state <2>, because the
modulation forms at the paramagnetic boundary with a period near 6 and cannot lock in to
period 4 at lower g. This script provides the data for a thermodynamic-integration
comparison of the competing states.

  --part control   chains started from a template at a fixed g (default <2>, code 3, at
                   g = 0.05), --therm + --meas sweeps, for each r_perp in --rperps: does the
                   template stay in place, at E0?
  --part branches  ASCENDING branches in g, one per start template in --codes (default <2>,
                   <3> and FM), through --g-grid-up, measured at every g; plus one DESCENDING
                   anneal from --g-top through the same grid (random start), also measured at
                   every g. F(g) = F(g_0) - N * integral mx dg along each branch then locates
                   the crossings of the branches' free energies.

All chains use the diagonal-only prefill of the engine, so the templates survive the start.
Per chain and g, the output has E, mx, the six pair correlations, and the layer structure
factor at q = 0, pi/3 and pi/2 with its argmax.

Usage (from vu_work/):
    .venv/bin/python experiments/sse_branches_v3.py --part control
    .venv/bin/python experiments/sse_branches_v3.py --part branches
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import time

import numpy as np

from phase_pipeline.io_utils import package_version, write_json, write_rows
from phase_pipeline.qmc.sse import (PAIR_CLASSES, bilayer_lattice, class_couplings, initial_spins,
                                    run_sse_chains)

CODE_NAMES = {0: "random", 1: "FM", 2: "A", 3: "<2>par", 4: "<2>anti", 5: "<3>par", 6: "<3>anti"}


def e_classical(kappa, rperp):
    return {"uniform": -2 + kappa - abs(rperp) / 2, "<2>": -1 - kappa - abs(rperp) / 2,
            "<3>": -4 / 3 - kappa / 3 - abs(rperp) / 2}


def row_of(r, L, extra):
    Sl2 = np.asarray(r["Sl2"])
    extra = {"kappa": extra.pop("kappa", None), **extra}
    q = 2 * np.pi * np.arange(L) / L
    iq = int(np.argmax(Sl2))
    d = {**extra, "g": r["Gamma"], "E_per_spin": r["E_per_spin"], "mx": r["mx"],
         **{c: r[c] for c in PAIR_CLASSES},
         "Sl2_q0": float(Sl2[0]), "Sl2_qpi3": float(Sl2[L // 6]) if L % 6 == 0 else float("nan"),
         "Sl2_qpi2": float(Sl2[L // 4]) if L % 4 == 0 else float("nan"),
         "qstar_over_pi": float(min(q[iq], 2 * np.pi - q[iq]) / np.pi), "n_ops": r["n_ops"]}
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="10_results/bilayer_annni_paper_results")
    ap.add_argument("--part", choices=["control", "branches"], required=True)
    ap.add_argument("--L", type=int, default=12)
    ap.add_argument("--T", type=float, default=0.15)
    ap.add_argument("--kappa", type=float, default=0.75)
    ap.add_argument("--rperps", default="0,1", help="control: r_perp values")
    ap.add_argument("--rperp", type=float, default=0.0, help="branches: r_perp")
    ap.add_argument("--control-code", type=int, default=3)
    ap.add_argument("--control-g", type=float, default=0.05)
    ap.add_argument("--codes", default="3,5,1")
    ap.add_argument("--g-top", type=float, default=5.0)
    ap.add_argument("--g-max-up", type=float, default=3.0, help="top of the ascending grid")
    ap.add_argument("--chains", type=int, default=6)
    ap.add_argument("--therm", type=int, default=None, help="default 5000 (control) / 2000 (branches)")
    ap.add_argument("--meas", type=int, default=None, help="default 5000 (control) / 3000 (branches)")
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    out = Path(a.results_dir) / "reanalysis_v3" / "sse_validation" / "branches"
    out.mkdir(parents=True, exist_ok=True)
    lat = bilayer_lattice(a.L, a.L, 2)
    beta = 1.0 / a.T
    rng = np.random.default_rng(20260930)
    rows, summary = [], {"args": vars(a), "numba": package_version("numba")}
    tag = f"{a.part}_L{a.L}_k{a.kappa:g}" + (f"_r{a.rperp:g}" if a.part == "branches" else "")
    t_all = time.time()
    if a.part == "control":
        therm, meas = a.therm or 5000, a.meas or 5000
        for rp in [float(s) for s in a.rperps.split(",")]:
            C = a.chains
            init = np.stack([initial_spins(lat, a.control_code, rng) for _ in range(C)])
            res = run_sse_chains(lat, np.tile(class_couplings(1, 1, a.kappa, rp), (C, 1)),
                                 np.full((C, 1), a.control_g), beta, init, 101 + np.arange(C), therm, meas,
                                 bin_size=100, threads=a.threads)
            ref = e_classical(a.kappa, rp)
            for c, rc in enumerate(res):
                rows.append(row_of(rc[0], a.L, {"kappa": a.kappa, "part": "control", "rperp": rp, "start": CODE_NAMES[a.control_code],
                                                "chain": c}))
            E = np.array([rc[0]["E_per_spin"] for rc in res])
            summary[f"control_rperp{rp:g}"] = {"E_mean": float(E.mean()), "E_min": float(E.min()),
                                               "E_max": float(E.max()), "E_classical": ref,
                                               "excess_vs_E0": float(E.max() - min(ref.values()))}
            print(f"control r={rp:g}: E={E.mean():+.5f} [{E.min():+.5f},{E.max():+.5f}]  classical {ref}", flush=True)
    else:
        therm, meas = a.therm or 2000, a.meas or 3000
        grid_up = np.round(np.concatenate([[0.05], np.arange(0.1, a.g_max_up + 1e-9, 0.1)]), 4)
        grid_down = np.round(np.concatenate([np.arange(a.g_top, 0.1 - 1e-9, -0.1), [0.05]]), 4)
        Jc = class_couplings(1, 1, a.kappa, a.rperp)
        codes = [int(s) for s in a.codes.split(",")]
        C = a.chains
        init = np.concatenate([np.stack([initial_spins(lat, code, rng) for _ in range(C)]) for code in codes])
        starts = [CODE_NAMES[code] for code in codes for _ in range(C)]
        t0 = time.time()
        res = run_sse_chains(lat, np.tile(Jc, (len(init), 1)), np.tile(grid_up, (len(init), 1)), beta, init,
                             201 + np.arange(len(init)), therm, meas, bin_size=100, threads=a.threads)
        summary["ascending_wall_s"] = time.time() - t0
        for c, rc in enumerate(res):
            for r in rc:
                rows.append(row_of(r, a.L, {"kappa": a.kappa, "part": "ascending", "rperp": a.rperp,
                                            "start": starts[c], "chain": c}))
        write_rows(out / f"{tag}.csv", rows)   # incremental: the ascending part is safe on disk
        t0 = time.time()
        initd = np.stack([initial_spins(lat, 0, rng) for _ in range(C)])
        resd = run_sse_chains(lat, np.tile(Jc, (C, 1)), np.tile(grid_down, (C, 1)), beta, initd,
                              301 + np.arange(C), therm, meas, bin_size=100, threads=a.threads)
        summary["descending_wall_s"] = time.time() - t0
        for c, rc in enumerate(resd):
            for r in rc:
                rows.append(row_of(r, a.L, {"kappa": a.kappa, "part": "descending", "rperp": a.rperp,
                                            "start": "random", "chain": c}))
        summary["E_classical"] = e_classical(a.kappa, a.rperp)
        for part, st in [("ascending", s) for s in dict.fromkeys(starts)] + [("descending", "random")]:
            sel = [r for r in rows if r["part"] == part and r["start"] == st]
            for gv in (0.05, 1.0, 2.0, 2.5, 3.0):
                v = [r for r in sel if abs(r["g"] - gv) < 1e-9]
                if v:
                    print(f"{part:10s} {st:8s} g={gv:4.2f}  E={np.mean([r['E_per_spin'] for r in v]):+.5f}  "
                          f"mx={np.mean([r['mx'] for r in v]):.4f}  q*/pi={[round(r['qstar_over_pi'], 3) for r in v]}",
                          flush=True)
    summary["wall_s"] = time.time() - t_all
    write_rows(out / f"{tag}.csv", rows)
    write_json(out / f"{tag}.json", summary)


if __name__ == "__main__":
    main()

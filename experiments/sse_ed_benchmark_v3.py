#!/usr/bin/env python3
"""Benchmark the SSE engine against exact finite-temperature diagonalization.

Four 12-spin periodic clusters built by the production lattice code (so SSE and ED use the
same bond list, including bonds repeated by small periodic lengths):
  bilayer_2x3  two 2x3 layers with rungs           (2D, interlayer bond)
  layer_2x6    one 2x6 layer                        (next-nearest bonds all distinct)
  ladder_1x6   two 1x6 chains with rungs, no x bond (frustrated ladder)
  layer_3x4    one 3x4 layer                        (2D, repeated y2 bonds)
Every case compares energy, transverse magnetization, both correlation estimators (bond
operator counts and imaginary-time-0 pairs), template moments and the per-layer structure
factor. SSE has no Trotter error, so all deviations should be statistical.

Every chain anneals in the transverse field, Gamma = linspace(g_start, g, n_steps) at fixed
beta (default 3.0 -> g in 16 steps), with --anneal-therm sweeps at every intermediate field and
no measurement there; only the final field is measured, after --therm further sweeps, for
--meas sweeps. Without annealing, chains at small g in frustrated cases freeze in
cluster-percolated states and miss the ED energy.
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

from phase_pipeline.io_utils import write_json, write_rows
from phase_pipeline.qmc.sse import (bilayer_lattice, class_couplings, initial_spins,
                                    run_sse_chains, PAIR_CLASSES, TEMPLATES, BOND_CLASSES)

LATTICES = {
    "bilayer_2x3": (2, 3, 2),
    "layer_2x6": (2, 6, 1),
    "ladder_1x6": (1, 6, 2),
    "layer_3x4": (3, 4, 1),
}
# lattice, T, Jy, kappa, rperp, g
CASES = [
    ("bilayer_2x3", 0.15, 1.0, 0.0, 1.0, 1.0),
    ("bilayer_2x3", 0.15, 1.0, 0.75, -1.0, 0.5),
    ("bilayer_2x3", 0.50, 1.0, 0.5, 1.0, 2.0),
    ("layer_2x6", 0.15, 1.0, 0.0, 0.0, 2.0),
    ("layer_2x6", 0.15, 1.0, 0.75, 0.0, 1.0),
    ("layer_2x6", 0.15, 1.0, 0.5625, 0.0, 1.5),
    ("layer_2x6", 0.15, -1.0, 0.25, 0.0, 1.0),
    ("layer_2x6", 0.50, 1.0, 1.0, 0.0, 0.5),
    ("ladder_1x6", 0.15, 1.0, 0.75, 1.0, 1.0),
    ("ladder_1x6", 0.15, 1.0, 1.3, -1.0, 0.3),
    ("ladder_1x6", 0.15, 1.0, 0.5, 1.0, 3.0),
    ("layer_3x4", 0.15, 1.0, 0.375, 0.0, 2.5),
    ("layer_3x4", 0.15, 1.0, 1.0, 0.0, 1.0),
]


def ed_thermal(lat, Jc, Gamma, beta):
    N = lat["N"]
    dim = 1 << N
    states = np.arange(dim)
    z = (1 - 2 * ((states[None, :] >> np.arange(N)[:, None]) & 1)).astype(float)
    diag = np.zeros(dim)
    for i, j, c in zip(lat["bi"], lat["bj"], lat["bcls"]):
        diag -= Jc[c] * z[i] * z[j]
    H = np.diag(diag)
    for i in range(N):
        H[states, states ^ (1 << i)] -= Gamma
    E, V = np.linalg.eigh(H)
    w = np.exp(-beta * (E - E[0]))
    w /= w.sum()
    P = V ** 2

    def avg(o):
        return float(w @ (o @ P))

    out = {"E_per_spin": float(w @ E) / N}
    out["mx"] = (avg(diag) - float(w @ E)) / (Gamma * N)
    for c, name in enumerate(PAIR_CLASSES):
        sel = lat["pcls"] == c
        if np.any(sel):
            o = np.mean([z[i] * z[j] for i, j in zip(lat["pi"][sel], lat["pj"][sel])], axis=0)
            out[name] = avg(o)
            if c < len(BOND_CLASSES):
                out[f"bond_{BOND_CLASSES[c]}"] = out[name]
    for t, name in enumerate(TEMPLATES):
        m = lat["tmpl"][t] @ z / N
        out[f"m{name}2"] = avg(m ** 2)
        out[f"m{name}4"] = avg(m ** 4)
        out[f"abs_m{name}"] = avg(np.abs(m))
    Ly, nl = lat["Ly"], lat["nlayers"]
    LxLy = N // nl
    col = np.zeros((nl, Ly, dim))
    for i in range(N):
        col[lat["layer"][i], lat["ycoord"][i]] += z[i]
    for n in range(Ly):
        ph = np.exp(2j * np.pi * n * np.arange(Ly) / Ly)
        a = np.einsum("y,lys->ls", ph, col) / LxLy
        a2 = np.abs(a) ** 2
        out[f"Sl2_q{n}"] = avg(a2.mean(axis=0))
        out[f"Sl4_q{n}"] = avg((a2 ** 2).mean(axis=0))
        out[f"Ss2_q{n}"] = avg(np.abs(a.mean(axis=0)) ** 2)
    return out


def sse_flat(r, Ly):
    d = {k: v for k, v in r.items() if isinstance(v, float)}
    for n in range(Ly):
        d[f"Sl2_q{n}"] = float(r["Sl2"][n])
        d[f"Sl4_q{n}"] = float(r["Sl4"][n])
        d[f"Ss2_q{n}"] = float(r["Ss2"][n])
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="10_results/bilayer_annni_paper_results")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--chains", type=int, default=24)
    ap.add_argument("--g-start", type=float, default=3.0)
    ap.add_argument("--n-steps", type=int, default=16)
    ap.add_argument("--anneal-therm", type=int, default=2000)
    ap.add_argument("--therm", type=int, default=5000)
    ap.add_argument("--meas", type=int, default=40000)
    ap.add_argument("--only-case", type=int, default=None)
    a = ap.parse_args()
    out = Path(a.results_dir) / "reanalysis_v3" / "sse_validation"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    t_start = time.time()
    rng = np.random.default_rng(20260928)
    for case_id, (lname, T, Jy, k, rp, g) in enumerate(CASES):
        if a.only_case is not None and case_id != a.only_case:
            continue
        Lx, Ly, nl = LATTICES[lname]
        lat = bilayer_lattice(Lx, Ly, nl)
        Jc = class_couplings(1.0, Jy, k, rp)
        beta = 1.0 / T
        t0 = time.time()
        ed = ed_thermal(lat, Jc, g, beta)
        t_ed = time.time() - t0
        C = a.chains
        init = np.stack([initial_spins(lat, 0, rng) for _ in range(C)])
        seeds = 1 + np.arange(C, dtype=np.uint64) + np.uint64(7919 * (case_id + 1))
        sched = np.linspace(a.g_start, g, a.n_steps)
        therm = np.full(a.n_steps, a.anneal_therm); therm[-1] = a.therm
        meas = np.zeros(a.n_steps, dtype=int); meas[-1] = a.meas
        t0 = time.time()
        res = run_sse_chains(lat, np.tile(Jc, (C, 1)), np.tile(sched, (C, 1)), beta, init, seeds,
                             therm, meas, bin_size=100, threads=a.threads)
        t_sse = time.time() - t0
        flat = [sse_flat(rc[-1], Ly) for rc in res]
        worst = 0.0
        for o, v_ed in ed.items():
            vals = np.array([f[o] for f in flat if o in f], dtype=float)
            if len(vals) < 2 or not np.all(np.isfinite(vals)):
                continue
            mean = float(vals.mean())
            se = float(vals.std(ddof=1) / math.sqrt(len(vals)))
            zval = (mean - v_ed) / se if se > 1e-14 else (0.0 if abs(mean - v_ed) < 1e-10 else float("inf"))
            worst = max(worst, abs(zval))
            rows.append({"case": case_id, "lattice": lname, "T": T, "Jy": Jy, "kappa": k, "rperp": rp,
                         "g": g, "observable": o, "ed": v_ed, "sse": mean, "se": se, "z": zval})
        n_ops = np.mean([rc[-1]["n_ops"] for rc in res])
        print(f"{case_id:2d} {lname:12s} T={T:.2f} Jy={Jy:+.0f} k={k:<6} r={rp:+.1f} g={g:<4} "
              f"E ed={ed['E_per_spin']:+.5f} sse={np.mean([f['E_per_spin'] for f in flat]):+.5f}  "
              f"max|z|={worst:.2f}  <n>={n_ops:.0f}  t_ed={t_ed:.1f}s t_sse={t_sse:.1f}s", flush=True)
    tag = "" if a.only_case is None else f"_case{a.only_case}"
    write_rows(out / f"sse_ed_benchmark{tag}.csv", rows)
    z = np.array([r["z"] for r in rows], dtype=float)
    n_inf = int(np.sum(~np.isfinite(z)))
    z = z[np.isfinite(z)]
    # With C chains the z values are Student-t distributed with C-1 degrees of freedom.
    from scipy.stats import t as student_t
    dof = a.chains - 1
    by_obs = {}
    for r in rows:
        by_obs.setdefault(r["observable"], []).append(r["z"])
    worst_obs = sorted(((o, float(np.max(np.abs(v)))) for o, v in by_obs.items()), key=lambda x: -x[1])[:15]
    summary = {"n_comparisons": int(len(z)), "n_nonfinite_z": n_inf,
               "median_abs_z": float(np.median(np.abs(z))),
               "frac_abs_z_lt_2": float(np.mean(np.abs(z) < 2)),
               "frac_abs_z_lt_3": float(np.mean(np.abs(z) < 3)),
               "max_abs_z": float(np.max(np.abs(z))),
               "expected_for_student_t": {"dof": dof, "median_abs": float(student_t.ppf(0.75, dof)),
                                          "frac_lt_2": float(2 * student_t.cdf(2, dof) - 1),
                                          "frac_lt_3": float(2 * student_t.cdf(3, dof) - 1)},
               "worst_observables": worst_obs,
               "per_case_max_abs_z": {int(c): float(max(abs(r["z"]) for r in rows if r["case"] == c))
                                      for c in sorted({r["case"] for r in rows})},
               "chains": a.chains, "g_start": a.g_start, "n_steps": a.n_steps, "anneal_therm": a.anneal_therm,
               "therm": a.therm, "meas": a.meas, "wall_s": time.time() - t_start,
               "cases": [dict(zip(["lattice", "T", "Jy", "kappa", "rperp", "g"], c)) for c in CASES]}
    write_json(out / f"sse_ed_summary{tag}.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=1))


if __name__ == "__main__":
    main()

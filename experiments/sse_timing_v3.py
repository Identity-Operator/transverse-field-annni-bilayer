#!/usr/bin/env python3
"""Cost of the SSE engine at production size, for sizing the production runs.

Bilayer L x L (default 24), T = 0.15, kappa = 0.75, r_perp = +1, one chain per thread.
  (1) the full 16-step anneal g = 4.0 -> 0.3 with --therm + --meas sweeps per step, timed as a
      whole, with the mean operator count and string cutoff at every step;
  (2) single-field runs at --single-g values (fresh chains, --therm + --meas sweeps), timed
      separately, to give the cost per 1000 sweeps as a function of g;
  (3) one step on a larger lattice (--extra-L, default 48) at --extra-g with --extra-sweeps sweeps.
All timings are wall time with one chain per thread; ms per sweep and <n>/N are reported.
With --wait-idle (default) the script first waits until no other SSE job of this project is
running, so that the timings are not distorted by contention.

Usage (from vu_work/):
    .venv/bin/python experiments/sse_timing_v3.py --threads 6
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import os
import subprocess
import time

import numpy as np

from phase_pipeline.io_utils import package_version, write_json
from phase_pipeline.qmc.sse import bilayer_lattice, class_couplings, initial_spins, run_sse_chains


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="10_results/bilayer_annni_paper_results")
    ap.add_argument("--L", type=int, default=24)
    ap.add_argument("--T", type=float, default=0.15)
    ap.add_argument("--kappa", type=float, default=0.75)
    ap.add_argument("--rperp", type=float, default=1.0)
    ap.add_argument("--g-start", type=float, default=4.0)
    ap.add_argument("--g-end", type=float, default=0.3)
    ap.add_argument("--n-steps", type=int, default=16)
    ap.add_argument("--therm", type=int, default=300)
    ap.add_argument("--meas", type=int, default=700)
    ap.add_argument("--single-g", default="4.0,2.0,1.0,0.3")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--extra-L", type=int, default=48)
    ap.add_argument("--extra-g", type=float, default=2.5)
    ap.add_argument("--extra-sweeps", type=int, default=2000)
    ap.add_argument("--no-wait-idle", action="store_true")
    ap.add_argument("--busy-dir", default="/private/tmp/claude-501/-Users-hoangnguyen-Phase-Classification/"
                    "5a3b742d-8c4d-44eb-95c0-d4caa48db041/scratchpad/sselogs",
                    help="also wait while any *.busy marker file exists here (queued SSE jobs)")
    a = ap.parse_args()
    if not a.no_wait_idle:
        me = os.getpid()
        while True:
            ps = subprocess.run(["ps", "-axo", "pid=,args="], capture_output=True, text=True).stdout.splitlines()
            busy = [l for l in ps if "experiments/sse_" in l and "_v3.py" in l and "sse_timing_v3" not in l
                    and int(l.split()[0]) != me]
            markers = list(Path(a.busy_dir).glob("*.busy")) if a.busy_dir else []
            if not busy and not markers:
                break
            time.sleep(30)
    out = Path(a.results_dir) / "reanalysis_v3" / "sse_validation" / "timing"
    out.mkdir(parents=True, exist_ok=True)
    lat = bilayer_lattice(a.L, a.L, 2)
    Jc = class_couplings(1.0, 1.0, a.kappa, a.rperp)
    beta = 1.0 / a.T
    C = a.threads
    rng = np.random.default_rng(1)
    init = np.stack([initial_spins(lat, 0, rng) for _ in range(C)])
    sched = np.linspace(a.g_start, a.g_end, a.n_steps)

    # warm-up compile on a tiny lattice so compile time is not counted
    small = bilayer_lattice(4, 4, 2)
    run_sse_chains(small, np.tile(Jc, (1, 1)), np.array([[1.0]]), beta,
                   np.stack([initial_spins(small, 0, rng)]), [1], 10, 10, bin_size=5, threads=1)

    t0 = time.time()
    res = run_sse_chains(lat, np.tile(Jc, (C, 1)), np.tile(sched, (C, 1)), beta, init,
                         np.arange(1, C + 1), a.therm, a.meas, bin_size=100, threads=a.threads)
    wall = time.time() - t0
    sweeps = a.n_steps * (a.therm + a.meas)
    steps = [{"g": float(g), "mean_n_ops": float(np.mean([rc[k]["n_ops"] for rc in res])),
              "M_cutoff": float(np.mean([rc[k]["M_cutoff"] for rc in res])),
              "E_per_spin": float(np.mean([rc[k]["E_per_spin"] for rc in res]))} for k, g in enumerate(sched)]
    anneal = {"wall_s": wall, "chains": C, "threads": a.threads, "sweeps_per_chain": sweeps,
              "wall_s_per_1000_sweeps_per_chain": wall / sweeps * 1000, "ms_per_sweep": wall / sweeps * 1000,
              "mean_n_over_N": float(np.mean([s_["mean_n_ops"] for s_ in steps]) / lat["N"]), "steps": steps}
    print(f"anneal: {wall:.1f} s for {sweeps} sweeps/chain on {C} chains -> "
          f"{wall / sweeps * 1000:.1f} ms per sweep (one chain per thread), <n>/N={anneal['mean_n_over_N']:.2f}")
    for s in steps:
        print(f"   g={s['g']:.3f}  <n>={s['mean_n_ops']:.0f}  M={s['M_cutoff']:.0f}  E={s['E_per_spin']:+.4f}")

    single = []
    for g in [float(x) for x in a.single_g.split(",")]:
        t0 = time.time()
        r = run_sse_chains(lat, np.tile(Jc, (C, 1)), np.full((C, 1), g), beta, init, np.arange(11, C + 11),
                           a.therm, a.meas, bin_size=100, threads=a.threads)
        w = time.time() - t0
        n = float(np.mean([rc[0]["n_ops"] for rc in r]))
        single.append({"L": a.L, "g": g, "wall_s": w, "ms_per_sweep": w / (a.therm + a.meas) * 1000,
                       "mean_n_ops": n, "mean_n_over_N": n / lat["N"], "us_per_op_per_sweep": w / (a.therm + a.meas) / n * 1e6})
        print(f"single L={a.L} g={g}: {w / (a.therm + a.meas) * 1000:.1f} ms per sweep, <n>={n:.0f}, <n>/N={n / lat['N']:.2f}")
    big = bilayer_lattice(a.extra_L, a.extra_L, 2)
    initb = np.stack([initial_spins(big, 0, rng) for _ in range(C)])
    half = a.extra_sweeps // 2
    t0 = time.time()
    r = run_sse_chains(big, np.tile(Jc, (C, 1)), np.full((C, 1), a.extra_g), beta, initb, np.arange(21, C + 21),
                       half, a.extra_sweeps - half, bin_size=100, threads=a.threads)
    w = time.time() - t0
    n = float(np.mean([rc[0]["n_ops"] for rc in r]))
    single.append({"L": a.extra_L, "g": a.extra_g, "wall_s": w, "ms_per_sweep": w / a.extra_sweeps * 1000,
                   "mean_n_ops": n, "mean_n_over_N": n / big["N"], "us_per_op_per_sweep": w / a.extra_sweeps / n * 1e6})
    print(f"single L={a.extra_L} g={a.extra_g}: {w / a.extra_sweeps * 1000:.1f} ms per sweep, <n>={n:.0f}, "
          f"<n>/N={n / big['N']:.2f}")
    write_json(out / f"timing_L{a.L}.json", {"args": vars(a), "anneal": anneal, "single": single,
                                             "N": lat["N"], "numba": package_version("numba")})


if __name__ == "__main__":
    main()

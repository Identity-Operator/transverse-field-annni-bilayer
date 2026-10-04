#!/usr/bin/env python3
"""Template branches for arbitrary straight-wall domain sequences (Paper 1, stage S5a).

Each chain starts in the x-uniform, parallel-stacked structure with the given domain lengths (e.g. "2,3"
for <23>, "2" for <2>, "3" for <3>, "2,2,3" for <223>) and is stepped UP through the g grid with the
step-2 protocol: --therm + --meas sweeps per g, and the engine's diagonal-only prefill so that the
template survives the start. Several sequences on the same lattice are run in one call (--seqs "2,3;2")
so that all threads are used. The output rows have the columns of sse_branches_v3.py (part = ascending,
start = '<seq>par'), so experiments/branch_free_energy_v3.py and the S5a analysis read them unchanged.

Lattice: --Lx x --Ly bilayer (--L sets both; J_perp = r_perp, the layers decouple at r_perp = 0). --Ly
must be a multiple of each sequence's spin period (a sequence with an odd number of domains has a doubled
period, e.g. <3> has period 6, <233> 16); --Lx is free (>= 4). The structure factor is taken along y.

Grid: --grid "a:b:step,a:b:step,..." pieces, inclusive, always preceded by g0 = 0.05; points rounded to
1e-4 and de-duplicated.

Seeds: derived from the output tag, so different cases and structures use different random streams.

Output: <results-dir>/reanalysis_v3/sse_validation/seq_branches/<tag>.csv (+ .json) with
tag = seqbr_Lx<Lx>_Ly<Ly>_k<kappa>_r<r>_T<T>_<seqs>. An existing CSV with the full grid for every chain is skipped,
so a batch can be re-run after an interruption.

Usage (from vu_work/):
    python experiments/sse_seq_branches_v3.py --seqs "2,3;2" --Lx 20 --Ly 20 --kappa 0.53 --rperp 0 \
        --grid "0.1:0.6:0.1,0.62:0.8:0.02,0.9:1.0:0.1" --chains 12 --threads 10
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import time
import zlib

import numpy as np

from phase_pipeline.io_utils import package_version, read_rows, write_json, write_rows
from phase_pipeline.qmc.sse import PAIR_CLASSES, bilayer_lattice, class_couplings, run_sse_chains


def column(lengths):
    s, sign = [], 1
    for n in lengths:
        s += [sign] * n
        sign = -sign
    if sign == -1:                      # odd number of domains: the spin period is doubled
        s = s + [-v for v in s]
    return s


def parse_grid(spec):
    pts = [0.05]
    for piece in spec.split(","):
        a, b, st = (float(x) for x in piece.split(":"))
        pts += list(np.arange(a, b + st / 2, st))
    return np.array(sorted(set(np.round(pts, 4))))


def row_of(r, L, extra):          # L = Ly (the structure factor is along y)
    Sl2 = np.asarray(r["Sl2"])
    q = 2 * np.pi * np.arange(L) / L
    iq = int(np.argmax(Sl2))
    return {**extra, "g": r["Gamma"], "E_per_spin": r["E_per_spin"], "mx": r["mx"],
            **{c: r[c] for c in PAIR_CLASSES},
            "Sl2_q0": float(Sl2[0]), "Sl2_qpi3": float(Sl2[L // 6]) if L % 6 == 0 else float("nan"),
            "Sl2_qpi2": float(Sl2[L // 4]) if L % 4 == 0 else float("nan"),
            "Sl2_qpi25": float(Sl2[L // 5]) if L % 5 == 0 else float("nan"),
            "qstar_over_pi": float(min(q[iq], 2 * np.pi - q[iq]) / np.pi), "n_ops": r["n_ops"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="10_results/bilayer_annni_paper_results")
    ap.add_argument("--seqs", required=True, help='domain sequences separated by ";", e.g. "2,3;2"')
    ap.add_argument("--L", type=int, default=None, help="sets Lx = Ly")
    ap.add_argument("--Lx", type=int, default=None)
    ap.add_argument("--Ly", type=int, default=None)
    ap.add_argument("--kappa", type=float, required=True)
    ap.add_argument("--rperp", type=float, required=True)
    ap.add_argument("--T", type=float, default=0.15)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--chains", type=int, default=12)
    ap.add_argument("--therm", type=int, default=2000)
    ap.add_argument("--meas", type=int, default=3000)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--tag-suffix", default="", help="appended to the output tag; also changes the seeds "
                    "(for declared follow-up runs; default keeps the pre-registered names and seeds)")
    a = ap.parse_args()
    Lx = a.Lx if a.Lx is not None else a.L
    Ly = a.Ly if a.Ly is not None else a.L
    if Lx is None or Ly is None:
        raise SystemExit("give --L or both --Lx and --Ly")
    seqs = [[int(x) for x in s.split(",")] for s in a.seqs.split(";")]
    names = ["<" + "".join(map(str, s)) + ">par" for s in seqs]
    grid = parse_grid(a.grid)
    out = Path(a.results_dir) / "reanalysis_v3" / "sse_validation" / "seq_branches"
    out.mkdir(parents=True, exist_ok=True)
    tag = (f"seqbr_Lx{Lx}_Ly{Ly}_k{a.kappa:g}_r{a.rperp:g}_T{a.T:g}_" + "-".join("".join(map(str, s)) for s in seqs)) + a.tag_suffix
    f = out / f"{tag}.csv"
    C = a.chains
    if f.exists():
        rows = read_rows(f)
        if len(rows) == len(seqs) * C * len(grid):
            print(f"skip {tag}: complete", flush=True)
            return
    lat = bilayer_lattice(Lx, Ly, 2)
    init, starts = [], []
    for s, nm in zip(seqs, names):
        col = column(s)
        if Ly % len(col):
            raise SystemExit(f"Ly={Ly} is not a multiple of the period {len(col)} of {nm}")
        sp = np.array([col[int(y) % len(col)] for y in lat["ycoord"]], dtype=np.int8)
        init.append(np.tile(sp, (C, 1)))
        starts += [nm] * C
    init = np.concatenate(init)
    seed0 = zlib.crc32(tag.encode()) % 1_000_000
    t0 = time.time()
    res = run_sse_chains(lat, np.tile(class_couplings(1, 1, a.kappa, a.rperp), (len(init), 1)),
                         np.tile(grid, (len(init), 1)), 1.0 / a.T, init, seed0 + np.arange(len(init)),
                         a.therm, a.meas, bin_size=100, threads=a.threads)
    rows = [row_of(r, Ly, {"kappa": a.kappa, "part": "ascending", "rperp": a.rperp, "T": a.T, "Lx": Lx, "Ly": Ly,
                            "start": starts[c], "chain": c})
            for c, rc in enumerate(res) for r in rc]
    write_rows(f, rows)
    write_json(out / f"{tag}.json", {"args": vars(a), "grid": grid.tolist(), "seed0": seed0,
                                     "numba": package_version("numba"), "wall_s": time.time() - t0})
    print(f"done {tag}: {len(init)} chains x {len(grid)} g in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

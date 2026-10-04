#!/usr/bin/env python3
"""End-to-end check of the thermodynamic integration against exact diagonalization (referee C2, point B1c).

On 12-spin clusters small enough for full exact diagonalization, chains are started in the ferromagnet at
g0 = 0.05 and stepped upward in g on the uniform step-2 grid (0.05, 0.1, ..., g_max), exactly as the template
branches of the paper, with the same engine, sweeps and diagonal-only prefill. The free energy per spin is then
integrated with the same quadrature as the paper (s5a_analyze_v3.cumint, 'rich' and 'trap'),
    f(g) = F_ED(g0) - int_{g0}^{g} m_x dg',
anchored at the EXACT free energy at g0 so that the test isolates the integration and the sampling, and compared
with the exact F_ED(g) = -T ln Z / N at every grid point. Both clusters are ergodic for the cluster update (no
frustrated modulated state), so the chains sample the full Gibbs state, as ED does.

Cases: 2x3 bilayer, kappa = 0, r = 1 (unfrustrated); 2x6 single layer, kappa = 0.3, r = 0 (frustrated J2, FM stable).
Output: reanalysis_v3/sse_validation/ti_exact_check.{json,txt}
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json

import numpy as np

from phase_pipeline.qmc.sse import bilayer_lattice, class_couplings, initial_spins, run_sse_chains
from sse_ed_benchmark_v3 import ed_thermal
import s5a_analyze_v3 as A

OUT = Path("10_results/bilayer_annni_paper_results/reanalysis_v3/sse_validation")
T = 0.15
BETA = 1 / T


def f_exact(lat, Jc, g):
    """-T ln Z / N from the full spectrum."""
    N = lat["N"]
    dim = 1 << N
    states = np.arange(dim)
    z = (1 - 2 * ((states[None, :] >> np.arange(N)[:, None]) & 1)).astype(float)
    diag = np.zeros(dim)
    for i, j, c in zip(lat["bi"], lat["bj"], lat["bcls"]):
        diag -= Jc[c] * z[i] * z[j]
    H = np.diag(diag)
    for i in range(N):
        H[states, states ^ (1 << i)] -= g
    E = np.linalg.eigvalsh(H)
    return float((E[0] - T * np.log(np.sum(np.exp(-BETA * (E - E[0]))))) / N)


def run_case(name, lat, kappa, r, gmax=3.0, chains=24, threads=2, seed0=7001, therm=2000, meas=3000):
    Jc = class_couplings(1, 1, kappa, r)
    grid = np.round(np.concatenate([[0.05], np.arange(0.1, gmax + 1e-9, 0.1)]), 4)
    rng = np.random.default_rng(20260930)
    init = np.stack([initial_spins(lat, 1, rng) for _ in range(chains)])
    res = run_sse_chains(lat, np.tile(Jc, (chains, 1)), np.tile(grid, (chains, 1)), BETA, init,
                         seed0 + np.arange(chains), therm, meas, bin_size=100, threads=threads)
    mx = np.array([[res[c][k]["mx"] for k in range(len(grid))] for c in range(chains)])
    cache = OUT / "ti_exact_check_repeat.json"
    cached = None
    if cache.exists():
        for c in json.loads(cache.read_text()):
            if abs(c["kappa"] - kappa) < 1e-12 and abs(c["r"] - r) < 1e-12 and c["N"] == lat["N"] and c["grid"] == grid.tolist():
                cached = c
    if cached is not None:        # exact values do not depend on the run; reuse them
        F_ed, mx_ed = np.array(cached["F_ED"]), np.array(cached["mx_ED"])
    else:
        F_ed = np.array([f_exact(lat, Jc, g) for g in grid])
        mx_ed = np.array([ed_thermal(lat, Jc, g, BETA)["mx"] for g in grid])
    out = {"case": name, "N": int(lat["N"]), "kappa": kappa, "r": r, "grid": grid.tolist(), "chains": chains,
           "seed0": seed0, "therm": therm, "meas": meas, "F_ED": F_ed.tolist(), "mx_ED": mx_ed.tolist(),
           "mx_mean": mx.mean(0).tolist()}
    for method in ("rich", "trap"):
        F = F_ed[0] - A.cumint(grid, mx, method, None)
        m, se = F.mean(0), F.std(0, ddof=1) / np.sqrt(chains)
        z = (m - F_ed) / np.where(se > 0, se, np.nan)
        out[method] = {"dF_max": float(np.nanmax(np.abs(m - F_ed))), "z_max": float(np.nanmax(np.abs(z[1:]))),
                       "z_mean": float(np.nanmean(z[1:])), "se_typ": float(np.nanmedian(se[1:])),
                       "dF": (m - F_ed).tolist(), "z": z.tolist()}
    zmx = (mx.mean(0) - mx_ed) / (mx.std(0, ddof=1) / np.sqrt(chains))
    out["mx_z_max"] = float(np.nanmax(np.abs(zmx)))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", action="store_true", help="kappa = 0.3 case again: new seeds, 48 chains, 4x sweeps")
    ap.add_argument("--separate", action="store_true",
                    help="kappa = 0.3 case twice more: production settings with new seeds, and 4x sweeps with the original seeds")
    a = ap.parse_args()
    if a.separate:
        cases = [("2x6 layer, kappa=0.3, r=0 (production settings, new seeds)", bilayer_lattice(2, 6, 1), 0.3, 0.0,
                  dict(chains=24, seed0=11001, therm=2000, meas=3000)),
                 ("2x6 layer, kappa=0.3, r=0 (4x sweeps, original seeds)", bilayer_lattice(2, 6, 1), 0.3, 0.0,
                  dict(chains=24, seed0=7001, therm=8000, meas=12000))]
        tag = "ti_exact_check_separate"
    elif a.repeat:
        cases = [("2x6 layer, kappa=0.3, r=0 (repeat: new seeds, 48 chains, 4x sweeps)", bilayer_lattice(2, 6, 1), 0.3, 0.0,
                  dict(chains=48, seed0=9001, therm=8000, meas=12000))]
        tag = "ti_exact_check_repeat"
    else:
        cases = [("2x3 bilayer, kappa=0, r=1", bilayer_lattice(2, 3, 2), 0.0, 1.0, {}),
                 ("2x6 layer, kappa=0.3, r=0", bilayer_lattice(2, 6, 1), 0.3, 0.0, {})]
        tag = "ti_exact_check"
    rep, lines = [], []
    for name, lat, k, r, kw in cases:
        o = run_case(name, lat, k, r, **kw)
        rep.append(o)
        line = (f"{name}: N={o['N']}; rich: max|f_TI - F_ED| = {o['rich']['dF_max']:.2e} (typ. se {o['rich']['se_typ']:.1e}), "
                f"max|z| = {o['rich']['z_max']:.2f}, mean z = {o['rich']['z_mean']:+.2f}; trap: max|dF| = {o['trap']['dF_max']:.2e}, "
                f"max|z| = {o['trap']['z_max']:.2f}; m_x vs ED max|z| = {o['mx_z_max']:.2f}")
        print(line, flush=True)
        lines.append(line)
    (OUT / f"{tag}.json").write_text(json.dumps(rep, indent=1))
    (OUT / f"{tag}.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

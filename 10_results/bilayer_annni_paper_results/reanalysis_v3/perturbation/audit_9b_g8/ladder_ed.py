"""Critical field of the edge-row models by ED: transverse Ising chain (control, g_c = 1) and two-leg ladder with
J_leg = J_rung = 1, periodic along the legs. Z2 sectors (global spin flip); gap = E0(odd) - E0(even); crossings of
L*gap between consecutive L (z = 1)."""
import json, time
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla
from scipy.optimize import brentq
from pathlib import Path

def sector_H(L, legs, g, parity):
    N = legs * L; dim = 1 << (N - 1); top = 1 << (N - 1); full = (1 << N) - 1
    s = np.arange(dim)                                   # representatives: top bit 0
    z = 1 - 2 * ((s[:, None] >> np.arange(N)) & 1)
    diag = np.zeros(dim)
    for l in range(legs):
        for x in range(L):
            diag -= z[:, l * L + x] * z[:, l * L + (x + 1) % L]
    if legs == 2:
        for x in range(L):
            diag -= z[:, x] * z[:, L + x]
    rows, cols, vals = [s], [s], [diag]
    for k in range(N):
        t = s ^ (1 << k)
        flipped = (t & top) != 0
        t = np.where(flipped, t ^ full, t)
        rows.append(s); cols.append(t); vals.append(-g * np.where(flipped, parity, 1.0))
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(dim, dim))

def gap(L, legs, g):
    e = [sla.eigsh(sector_H(L, legs, g, p), k=1, which="SA", tol=1e-12)[0][0] for p in (1.0, -1.0)]
    return e[1] - e[0]

out = {}
for legs, Ls, lo, hi in ((1, (6, 8, 10, 12, 14, 16), 0.8, 1.2), (2, (4, 6, 8, 10), 1.4, 2.4)):
    res = []
    for L1, L2 in zip(Ls[:-1], Ls[1:]):
        t0 = time.time()
        f = lambda g: L1 * gap(L1, legs, g) - L2 * gap(L2, legs, g)
        gs = np.linspace(lo, hi, 11); v = [f(g) for g in gs]
        cr = [brentq(f, gs[i], gs[i + 1], xtol=1e-6) for i in range(len(gs) - 1) if v[i] * v[i + 1] < 0]
        res.append({"L1": L1, "L2": L2, "g_cross": cr})
        print(f"legs={legs} L={L1}/{L2}: L*gap crossing at g = {cr}  ({time.time()-t0:.0f}s)", flush=True)
    out["chain" if legs == 1 else "ladder_Jleg=Jrung=1"] = res
json.dump(out, open(Path(__file__).parent / "ladder_ed.json", "w"), indent=1)

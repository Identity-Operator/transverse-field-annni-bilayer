"""Cluster-free check of eps8: RS recursion on a periodic Lx x Ly layer (r = 0) in the space of all states
within 4 flips of the reference (exact for E8: psi_7 is needed at 1 flip, psi_6 at <= 2, psi_5 at <= 3,
psi_4 at <= 4). Flip energies from the bond list (generic bond-cut sum). float64."""
import itertools, json, sys, time
from pathlib import Path
import numpy as np, scipy.sparse as sp
sys.path.insert(0, str(Path(__file__).parent))
from lc8 import column

def lattice(Lx, Ly, kappa=0.5):
    idx = lambda x, y: (x % Lx) * Ly + (y % Ly)
    b = []
    for x in range(Lx):
        for y in range(Ly):
            b += [(idx(x, y), idx(x + 1, y), 1.0), (idx(x, y), idx(x, y + 1), 1.0), (idx(x, y), idx(x, y + 2), -kappa)]
    return Lx * Ly, b

def run(seq, Lx, Ly, order=8, maxf=4):
    N, bonds = lattice(Lx, Ly)
    col = [1] if seq is None else column(seq)
    s = np.array([col[(i % Ly) % len(col)] for i in range(N)], float)
    nb = [[] for _ in range(N)]
    for i, j, J in bonds:
        nb[i].append((j, J)); nb[j].append((i, J))
    states = [()]
    for k in range(1, maxf + 1):
        states += list(itertools.combinations(range(N), k))
    index = {st: a for a, st in enumerate(states)}
    M = len(states); dE = np.zeros(M)
    for a, st in enumerate(states):
        F = set(st); e = 0.0
        for i in st:
            for j, J in nb[i]:
                if j not in F:
                    e += 2 * J * s[i] * s[j]
        dE[a] = e
    assert dE[1:].min() > 1e-9
    rows, cols = [], []
    for a, st in enumerate(states):
        F = set(st)
        for i in range(N):
            t = tuple(sorted(F ^ {i}))
            if len(t) <= maxf:
                rows.append(a); cols.append(index[t])
    V = -sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(M, M))
    R = np.zeros(M); R[1:] = -1 / dE[1:]
    psi = [np.zeros(M) for _ in range(order + 1)]; psi[0][0] = 1; E = [0.0] * (order + 1)
    for n in range(1, order + 1):
        v = V @ psi[n - 1]; E[n] = v[0]
        psi[n] = R * (v - sum(E[k] * psi[n - k] for k in range(1, n + 1)))
    return [e / N for e in E], N, M

out = {}
for name, seq, Lx, Ly in (("FM", None, 5, 9), ("<23>", [2, 3], 5, 10), ("<3>", [3], 5, 12), ("<2>", [2], 5, 12)):
    t0 = time.time()
    E, N, M = run(seq, Lx, Ly)
    out[name] = {"Lx": Lx, "Ly": Ly, "N": N, "states": M, "eps": {n: E[n] for n in (2, 4, 6, 8)}}
    print(f"{name:5s} {Lx}x{Ly} N={N} states={M}: eps2 {E[2]:+.12e} eps4 {E[4]:+.12e} eps6 {E[6]:+.12e} eps8 {E[8]:+.12e} ({time.time()-t0:.0f}s)", flush=True)
json.dump(out, open(Path(__file__).parent / "fullrs8_check.json", "w"), indent=1)

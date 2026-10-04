"""Is e6 * L (kappa = 1/2) a sum over consecutive domain triples? Exact check with the (independently verified)
linked-cluster e6 of pt_order6_v3.py."""
import sys, time
from fractions import Fraction as Fr
sys.path[:0] = ["experiments", "."]
from pt_order4_v3 import cyclic_sequences
import pt_order6_v3 as P6
maxd = int(sys.argv[1]) if len(sys.argv) > 1 else 6
def feats(q):
    qq = list(q) * (2 if len(q) % 2 else 1); m = len(qq); f = {}
    for i in range(m):
        t = (qq[i - 1], qq[i], qq[(i + 1) % m]); f[t] = f.get(t, 0) + 1
    return f, sum(qq)
def solve_exact(rows, rhs):
    n = len(rows[0])
    A = [[sum(rw[i] * rw[j] for rw in rows) for j in range(n)] + [sum(rw[i] * y for rw, y in zip(rows, rhs))] for i in range(n)]
    piv, rI = [], 0
    for c in range(n):
        pr = next((i for i in range(rI, n) if A[i][c] != 0), None)
        if pr is None: continue
        A[rI], A[pr] = A[pr], A[rI]; A[rI] = [x / A[rI][c] for x in A[rI]]
        for i in range(n):
            if i != rI and A[i][c] != 0:
                f = A[i][c]; A[i] = [x - f * y for x, y in zip(A[i], A[rI])]
        piv.append(c); rI += 1
    sol = [Fr(0)] * n
    for i, c in enumerate(piv): sol[c] = A[i][n]
    return sol, len(piv)
for r in (Fr(0), Fr(1)):
    seqs = cyclic_sequences((2, 3), maxd); keys = sorted({t for q in seqs for t in feats(q)[0]})
    rows, rhs, t0 = [], [], time.time()
    for q in seqs:
        f, L = feats(q)
        rows.append([Fr(f.get(kk, 0)) for kk in keys]); rhs.append(P6.per_spin_series(q, Fr(1, 2), r)[3] * L)
    sol, rank = solve_exact(rows, rhs)
    res = [sum(a * b for a, b in zip(rw, sol)) - y for rw, y in zip(rows, rhs)]
    # also test NN-domain (pairs) form, which should FAIL at O(g^6)
    kp = sorted({(q2[i], q2[(i + 1) % len(q2)]) for q in seqs for q2 in [list(q) * (2 if len(q) % 2 else 1)] for i in range(len(q2))})
    rows2 = []
    for q in seqs:
        qq = list(q) * (2 if len(q) % 2 else 1); c = {}
        for i in range(len(qq)): c[(qq[i], qq[(i + 1) % len(qq)])] = c.get((qq[i], qq[(i + 1) % len(qq)]), 0) + 1
        rows2.append([Fr(c.get(kk, 0)) for kk in kp])
    sol2, rank2 = solve_exact(rows2, rhs)
    res2 = [sum(a * b for a, b in zip(rw, sol2)) - y for rw, y in zip(rows2, rhs)]
    print(f"r={r}: {len(seqs)} sequences (<= {maxd} domains), {time.time()-t0:.0f}s. triple form: rank {rank}, max|res| = {max(abs(x) for x in res)};"
          f"  pair form: max|res| = {float(max(abs(x) for x in res2)):.3e}", flush=True)

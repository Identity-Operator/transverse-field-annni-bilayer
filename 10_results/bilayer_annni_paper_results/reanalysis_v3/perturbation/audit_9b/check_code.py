import sys, random
from fractions import Fraction as Fr
sys.path.insert(0, "experiments"); sys.path.insert(0, ".")
sys.path.insert(0, "/tmp/claude-1000/-home-hoang-nguyen-Phase-Classification/8f62f05f-df21-4fe1-94fa-831f10b083be/scratchpad")
import pt_order4_v3 as P
import indep_rs as I
half = Fr(1, 2)
random.seed(3)
# (1) their per_column vs my brute force on random sequences incl. 4s and 5s, at kappa=1/2 and off it
seqs = [(2,), (3,), (2, 3), (2, 2, 3), (3, 3, 2), (2, 4), (4,), (2, 3, 4), (5, 2), (3, 2, 2, 3, 3), (2, 2, 2, 3, 3, 3)]
seqs += [tuple(random.choice((2, 3, 4)) for _ in range(random.randint(2, 5))) for _ in range(8)]
bad = 0
for q in seqs:
    for kap, r in ((half, Fr(0)), (half, Fr(1)), (Fr(3, 5), Fr(1)), (Fr(9, 20), Fr(0)), (half, Fr(3, 2))):
        a = P.per_column(list(q), kap, r)
        b = I.coeffs(list(q), kap, r)
        d = [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
        if any(x != 0 for x in d):
            bad += 1; print("MISMATCH", q, kap, r, d)
print("per_column vs independent brute force: sequences", len(seqs), "x 5 (kappa,r); mismatches:", bad)
# FM branch of coeffs vs brute force
for kap, r in ((half, Fr(0)), (half, Fr(1)), (Fr(1, 4), Fr(0)), (Fr(3, 5), Fr(1))):
    a = P.coeffs("FM", kap, r); b = I.coeffs(None, kap, r)
    print("FM", kap, r, [a[i] - b[i] for i in range(4)])
# (2) NN-domain decomposition over ALL of their 2/3 sequences (<= 10 domains) and 2/3/4 (<= 6)
for r in (Fr(0), Fr(1)):
    k = Fr(6) / ((2 + r) * (3 + r) * (5 + r)) / 4
    def Wtot(q):
        e0, e2, e4, e2p = P.per_column(list(q), half, r)
        L = sum(q) * (2 if len(q) % 2 else 1)
        return (k * e2p + e4) * L, L
    alph = (2, 3, 4)
    # unknowns: a(n) for n in alph, b(n,m) symmetric sums; fit on small set, then test all
    import itertools
    def feats(q):
        qq = list(q) * (2 if len(q) % 2 else 1)
        f = {}
        for i, n in enumerate(qq):
            f[("a", n)] = f.get(("a", n), 0) + 1
            m = qq[(i + 1) % len(qq)]
            key = ("b",) + tuple(sorted((n, m)))
            f[key] = f.get(key, 0) + 1
        return f
    allq = P.cyclic_sequences((2, 3), 10) + [s for s in P.cyclic_sequences((2, 3, 4), 6) if 4 in s]
    keys = sorted({kk for q in allq for kk in feats(q)})
    rows = [[Fr(feats(q).get(kk, 0)) for kk in keys] for q in allq]
    rhs = [Wtot(q)[0] for q in allq]
    # exact normal equations + Gauss-Jordan with free columns set to 0
    n = len(keys)
    A = [[sum(rw[i] * rw[j] for rw in rows) for j in range(n)] + [sum(rw[i] * y for rw, y in zip(rows, rhs))] for i in range(n)]
    piv = []; rI = 0
    for c in range(n):
        pr = next((i for i in range(rI, n) if A[i][c] != 0), None)
        if pr is None: continue
        A[rI], A[pr] = A[pr], A[rI]
        A[rI] = [x / A[rI][c] for x in A[rI]]
        for i in range(n):
            if i != rI and A[i][c] != 0:
                f = A[i][c]; A[i] = [x - f * y for x, y in zip(A[i], A[rI])]
        piv.append(c); rI += 1
    sol = [Fr(0)] * n
    for i, c in enumerate(piv): sol[c] = A[i][n]
    res = [sum(a * b for a, b in zip(rw, sol)) - y for rw, y in zip(rows, rhs)]
    print(f"r={r}: {len(allq)} sequences, {len(keys)} NN-domain parameters (rank {len(piv)}), max |residual| = {max(abs(x) for x in res)}")

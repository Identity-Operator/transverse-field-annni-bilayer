"""Closed-form domain decomposition W*L = sum_k [a(n_k) + b(n_k, n_{k+1})] at kappa = 1/2, and a check that
e6 is a sum over domain triples."""
import sys, itertools, time
from fractions import Fraction as Fr
sys.path[:0] = ["experiments", "."]
from pt_order4_v3 import per_column, cyclic_sequences, column
half = Fr(1, 2)

def P(Di, Dj, B):                         # coupled-pair O(g^4) term
    S = Di + Dj
    return -4 * B * S / ((Di * Dj) ** 2 * (S - 4 * B))

def site_D(n, r):
    """flip costs D = 2h along a domain of length n (kappa = 1/2)."""
    if n == 2: return [2 * (3 + r)] * 2
    if n == 3: return [2 * (2 + r), 2 * (5 + r), 2 * (2 + r)]
    return [2 * (2 + r), 2 * (4 + r)] + [2 * (3 + r)] * (n - 4) + [2 * (4 + r), 2 * (2 + r)]

def site_c(n):                           # c = s_y (s_{y+2} + s_{y-2})
    return [2 if (i >= 2 and n - 1 - i >= 2) else (0 if (i >= 2) != (n - 1 - i >= 2) else -2) for i in range(n)]

def a(n, r, k):
    D = site_D(n, r); c = site_c(n)
    t = sum(-k * 2 * ci / Di ** 2 + 1 / Di ** 3 + P(Di, Di, 1) + (P(Di, Di, r) / 2 if r else 0) for Di, ci in zip(D, c))
    t += sum(P(D[i], D[i + 1], 1) for i in range(n - 1))          # intra-domain y pairs
    t += sum(P(D[i], D[i + 2], -half) for i in range(n - 2))      # intra-domain y2 pairs
    return t

def b(n, m, r):
    A, B_ = site_D(n, r), site_D(m, r)
    return P(A[-1], B_[0], -1) + P(A[-2], B_[0], half) + P(A[-1], B_[1], half)

for r in (Fr(0), Fr(1)):
    F3 = Fr(6) / ((2 + r) * (3 + r) * (5 + r)); k = F3 / 4
    bad = 0; seqs = cyclic_sequences((2, 3), 10) + [s for s in cyclic_sequences((2, 3, 4, 5), 6) if max(s) > 3]
    for q in seqs:
        qq = list(q) * (2 if len(q) % 2 else 1); L = sum(qq)
        pred = sum(a(n, r, k) + b(n, qq[(i + 1) % len(qq)], r) for i, n in enumerate(qq))
        e0, e2, e4, e2p = per_column(list(q), half, r)
        if pred != (k * e2p + e4) * L: bad += 1
    print(f"r={r}: closed-form a(n)+b(n,n') reproduces W*L exactly for {len(seqs)-bad}/{len(seqs)} sequences (2/3 up to 10 domains; 4/5 up to 6)")
    for n in (2, 3, 4, 5): print(f"   a({n}) = {a(n, r, k)}")
    for n, m in ((2, 2), (2, 3), (3, 3), (2, 4), (3, 4), (4, 4)): print(f"   b({n},{m}) = {b(n, m, r)}")
    w22, w33, w23 = a(2, r, k) + b(2, 2, r), a(3, r, k) + b(3, 3, r), a(2, r, k) + a(3, r, k) + b(2, 3, r) + b(3, 2, r)
    print(f"   cycle costs: c<2> = {w22} (= 2 W<2>), c<3> = {w33} (= 3 W<3>), c<23> = {w23} (= 5 W<23>)")
    W2, W3, W23 = w22 / 2, w33 / 3, w23 / 5
    depth = W23 - (W3 + (W2 - W3) * (Fr(2, 5) - Fr(1, 3)) / (Fr(1, 2) - Fr(1, 3)))
    print(f"   W<2>={W2}, W<3>={W3}, W<23>={W23}; depth of <23> below the <3>-<2> chord = {depth} = {float(depth):.4e}")

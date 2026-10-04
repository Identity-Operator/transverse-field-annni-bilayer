"""Independent RS O(g^4) check: own lattice, generic flip energies, exact Fractions."""
from fractions import Fraction as Fr
import itertools, sys

def lattice(Lx, Ly, r, kappa):
    L = 2 if r != 0 else 1
    idx = lambda l, x, y: (l * Lx + x % Lx) * Ly + y % Ly
    bonds = []
    for l in range(L):
        for x in range(Lx):
            for y in range(Ly):
                bonds.append((idx(l, x, y), idx(l, x + 1, y), Fr(1)))
                bonds.append((idx(l, x, y), idx(l, x, y + 1), Fr(1)))
                bonds.append((idx(l, x, y), idx(l, x, y + 2), -kappa))   # H = -sum J s s ; J2 bond = -kappa
    if L == 2:
        for x in range(Lx):
            for y in range(Ly):
                bonds.append((idx(0, x, y), idx(1, x, y), Fr(r)))
    N = L * Lx * Ly
    ycoord = [i % Ly for i in range(N)]
    return N, bonds, ycoord

def spins_for(seq, Ly, ycoord):
    col, sg = [], 1
    for n in seq:
        col += [sg] * n; sg = -sg
    if sg == -1:
        col = col + [-v for v in col]
    assert Ly % len(col) == 0, (seq, Ly, len(col))
    return [col[y % len(col)] for y in ycoord]

def rs(N, bonds, s):
    nb = [[] for _ in range(N)]
    for i, j, J in bonds:
        nb[i].append((j, J)); nb[j].append((i, J))
    def dE(S):  # energy change flipping set S, generic
        S = set(S); d = Fr(0)
        for i in S:
            for j, J in nb[i]:
                if j not in S:
                    d += 2 * J * s[i] * s[j]
        return d
    E0 = -sum(J * s[i] * s[j] for i, j, J in bonds)
    D = [dE([i]) for i in range(N)]
    assert min(D) > 0
    E2 = -sum(1 / d for d in D)
    path = Fr(0)
    for i in range(N):
        for j in range(i + 1, N):
            Dij = dE([i, j]); assert Dij > 0
            # ordered (i,j) and (j,i), c in {i,j}
            path += (1 / Dij) * (1 / D[i] + 1 / D[j]) ** 2
    E4 = -path + sum(1 / d for d in D) * sum(1 / d ** 2 for d in D)
    return E0, E2, E4

def per_spin(seq, kappa, r, Lx=3):
    per = 1 if seq is None else (len(spins_for(seq, 10**6 if False else None, []) ) if False else None)
    return None

def structure(seq, kappa, r, Lx=3):
    if seq is None:
        Ly = 5
    else:
        P = sum(seq) * (2 if len(seq) % 2 else 1)
        Ly = P if P >= 5 else 2 * P
    N, bonds, yc = lattice(Lx, Ly, r, kappa)
    s = [1] * N if seq is None else spins_for(seq, Ly, yc)
    return N, bonds, s

def coeffs(seq, kappa, r):
    N, b, s = structure(seq, kappa, r)
    E0, E2, E4 = rs(N, b, s)
    # exact d e2/dkappa: D is linear in kappa
    _, b0, _ = structure(seq, Fr(0), r); _, b1, _ = structure(seq, Fr(1), r)
    def D_of(bonds):
        nb = [Fr(0)] * N
        for i, j, J in bonds:
            nb[i] += 2 * J * s[i] * s[j]; nb[j] += 2 * J * s[i] * s[j]
        return nb
    Dk, D0, D1 = D_of(b), D_of(b0), D_of(b1)
    e2p = sum((D1[i] - D0[i]) / Dk[i] ** 2 for i in range(N)) / N
    e0p = (-sum(J * s[i] * s[j] for i, j, J in b1) + sum(J * s[i] * s[j] for i, j, J in b0)) / N
    return E0 / N, E2 / N, E4 / N, e2p, e0p

if __name__ == "__main__":
    half = Fr(1, 2)
    names = {"FM": None, "<2>": [2], "<3>": [3], "<23>": [2, 3], "<223>": [2, 2, 3], "<233>": [2, 3, 3], "<2233>": [2, 2, 3, 3]}
    import json
    out = {}
    for r in (0, 1):
        R = Fr(r)
        C = {n: coeffs(q, half, R) for n, q in names.items()}
        out[r] = C
        print(f"r={r}")
        for n, (e0, e2, e4, e2p, e0p) in C.items():
            print(f"  {n:6s} e0={str(e0):>8s} e2={str(e2):>14s} e4={str(e4):>22s} ({float(e4):+.8f}) e2'={str(e2p):>16s} e0'={e0p}")
        # O(g^2) boundaries
        def k_of(A, B):
            return -(C[A][1] - C[B][1]) / (C[A][4] - C[B][4])
        def k1_of(A, B, k):
            return -(k * (C[A][3] - C[B][3]) + C[A][2] - C[B][2]) / (C[A][4] - C[B][4])
        k32 = k_of("<3>", "<2>"); kF3 = k_of("<3>", "FM")
        print(f"  k_32={k32}  k1_32={k1_of('<3>','<2>',k32)} = {float(k1_of('<3>','<2>',k32)):.7f}")
        print(f"  k_FM3={kF3} k1_FM3={k1_of('<3>','FM',kF3)} = {float(k1_of('<3>','FM',kF3)):.7f}")
        # O(g^2) degeneracy of all 2/3 structures at k32
        for n in ("<2>", "<3>", "<23>", "<223>", "<233>", "<2233>"):
            print(f"    g2 at k32: {n:6s} {k32 * C[n][4] + C[n][1]}")
        lo = k1_of("<23>", "<3>", k32); hi = k1_of("<23>", "<2>", k32)
        print(f"  <23> window k1 in ({lo}, {hi}) = ({float(lo):.7f}, {float(hi):.7f}), width {float(hi-lo):.6f}")
        for n in ("<223>", "<233>"):
            print(f"    k1 {n}|<23> = {k1_of(n,'<23>',k32)}  vs <2>|<23> {hi}, <3>|<23> {lo}")
        # chord test in (rho, W): W = k e2' + e4 ; rho from e0' = 1 - 4 rho
        pt = {n: ((1 - C[n][4]) / 4, k32 * C[n][3] + C[n][2]) for n in C if n != "FM"}
        def on_chord(P, A, B):
            (x, y), (x1, y1), (x2, y2) = pt[P], pt[A], pt[B]
            return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        print("   cross(<223> vs <2>-<23>) =", on_chord("<223>", "<2>", "<23>"),
              "  cross(<233> vs <3>-<23>) =", on_chord("<233>", "<3>", "<23>"))
        # <2233> = 22 + 33 + 23 cycles: predicted W from simple cycles
        (x2, y2), (x3, y3), (x23, y23) = pt["<2>"], pt["<3>"], pt["<23>"]
        # lengths: 2->2 loop 2 sites, 3->3 loop 3 sites, 2-3-2 cycle 5 sites; <2233> = 2 + 3 + 5 = 10 sites
        pred = (2 * y2 + 3 * y3 + 5 * y23) / 10
        print("   <2233> W - cycle decomposition =", pt["<2233>"][1] - pred)

"""Independent linked-cluster Rayleigh-Schroedinger theory to O(g^8) for straight-wall structures of the
bilayer transverse-field ANNNI model (audit 9b, O(g^8)). Written from scratch; it shares no code with
experiments/pt_order4_v3.py, pt_order6_v3.py or the referee's pt8.py.

Method
- H = H0 + g V, H0 = -sum J_ij s_i s_j (J_x = J_y = 1, J_y2 = -kappa, J_rung = r), V = -sum sigma^x.
- E[S] = RS ground-energy series when the field acts only on the sites of S (other spins frozen), from the
  standard recursion E_n = <0|V|psi_{n-1}>, psi_n = R [V psi_{n-1} - sum_{k=1..n} E_k psi_{n-k}].
- w(C) = sum_{S subset C} (-1)^{|C|-|S|} E[S] (Moebius). The energy is sum_C w(C) over connected clusters,
  and w_{2n}(C) = 0 for |C| > n (asserted, not assumed), so O(g^8) needs clusters of <= 4 sites.
- Clusters are enumerated on the INFINITE lattice: for every site of one unit cell (x = 0, one spin period
  along y, both layers) all connected clusters whose lexicographically smallest site is that site.
- kappa enters as a truncated Taylor series ("jet") in delta = kappa - 1/2, exact Fractions, so the
  kappa-derivatives needed for the boundary expansion come out of the same run. A cluster of m sites is
  evaluated with jets of order 4 - m, which is exactly what W8 needs (eps2 to 3rd order, eps4 to 2nd,
  eps6 to 1st, eps8 at delta = 0).
"""
from __future__ import annotations

import itertools
from fractions import Fraction as Fr

# ------------------------------------------------------------------ jets (truncated series in delta)
def jconst(c, L):
    return (Fr(c),) + (Fr(0),) * (L - 1)


def jtrunc(a, L):
    return tuple(a[:L]) + (Fr(0),) * (L - len(a))


def jadd(a, b):
    return tuple(x + y for x, y in zip(a, b))


def jsub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def jscale(a, c):
    return tuple(x * c for x in a)


def jmul(a, b):
    L = len(a)
    return tuple(sum(a[i] * b[k - i] for i in range(k + 1)) for k in range(L))


def jdiv(a, b):
    L = len(a)
    q = []
    for k in range(L):
        q.append((a[k] - sum(b[i] * q[k - i] for i in range(1, k + 1))) / b[0])
    return tuple(q)


# ------------------------------------------------------------------ RS for a small cluster
def rs_cluster(D, B, order=8):
    """D: list of jets (single-flip costs), B: dict {(a, b): jet} (a < b), pair terms B_ab = J_ab s_a s_b.
    Flipping the set F costs sum_F D - 4 sum_{pairs in F} B (Ising identity). Returns [E_0..E_order] (jets)."""
    m = len(D)
    L = len(D[0])
    M = 1 << m
    zero = jconst(0, L)
    dE = []
    for st in range(M):
        bits = [i for i in range(m) if st >> i & 1]
        e = zero
        for i in bits:
            e = jadd(e, D[i])
        for a, b in itertools.combinations(bits, 2):
            if (a, b) in B:
                e = jsub(e, jscale(B[(a, b)], 4))
        dE.append(e)
    pop = [bin(st).count("1") for st in range(M)]
    for st in range(1, M):
        if pop[st] <= order:              # psi_n has support within n flips; farther states never enter
            assert dE[st][0] > 0, "non-positive excitation energy"
    negdE = [jscale(e, -1) for e in dE]
    psi = [[zero] * M for _ in range(order + 1)]
    psi[0][0] = jconst(1, L)
    E = [zero] * (order + 1)
    for n in range(1, order + 1):
        Vpsi = []
        for st in range(M):
            acc = zero
            for i in range(m):
                acc = jsub(acc, psi[n - 1][st ^ (1 << i)])
            Vpsi.append(acc)
        E[n] = Vpsi[0]
        for st in range(1, M):
            if pop[st] > n:
                continue
            acc = Vpsi[st]
            for k in range(1, n + 1):
                if any(E[k]) and any(psi[n - k][st]):
                    acc = jsub(acc, jmul(E[k], psi[n - k][st]))
            psi[n][st] = jdiv(acc, negdE[st])
    return E


# ------------------------------------------------------------------ canonical weighted graphs + memo
_E_memo, _w_memo = {}, {}


def canon(D, B):
    """Canonical form of the weighted graph (D_i, B_ij) under site permutations; returns (key, perm)."""
    m = len(D)
    best = None
    for p in itertools.permutations(range(m)):
        Dk = tuple(D[p[a]] for a in range(m))
        Bk = tuple(B.get((min(p[a], p[b]), max(p[a], p[b]))) for a, b in itertools.combinations(range(m), 2))
        key = (Dk, tuple((b is not None, b if b is not None else ()) for b in Bk))
        if best is None or key < best:
            best = key
    return best


def E_of(D, B, order=8):
    key = canon(D, B)
    if key not in _E_memo:
        _E_memo[key] = rs_cluster(D, B, order)
    return _E_memo[key]


def w_of(D, B, order=8):
    """Moebius weight of the cluster (all its sites), with the |C| > n vanishing asserted."""
    m = len(D)
    key = canon(D, B)
    if key in _w_memo:
        return _w_memo[key]
    L = len(D[0])
    tot = [jconst(0, L) for _ in range(order + 1)]
    for size in range(1, m + 1):
        sign = (-1) ** (m - size)
        for S in itertools.combinations(range(m), size):
            idx = {s: a for a, s in enumerate(S)}
            DS = [D[s] for s in S]
            BS = {(idx[a], idx[b]): v for (a, b), v in B.items() if a in idx and b in idx}
            e = E_of(DS, BS, order)
            for n in range(order + 1):
                tot[n] = jadd(tot[n], jscale(e[n], sign))
    for n in range(order + 1):
        if n < 2 * m:
            assert not any(tot[n]), f"linked-cluster property violated: |C|={m}, n={n}"
    _w_memo[key] = tot
    return tot


# ------------------------------------------------------------------ lattice (infinite, x-uniform)
def column(lengths):
    s, sign = [], 1
    for n in lengths:
        s += [sign] * n
        sign = -sign
    if sign == -1:
        s = s + [-v for v in s]
    return s


class Structure:
    """x-uniform straight-wall structure on the infinite square lattice (one layer if r = 0, else a
    parallel-stacked bilayer). lengths=None is the ferromagnet. geometry='chain' gives the 1D TFIM
    (J_y = 1 only) for checks."""

    def __init__(self, lengths, r, geometry="annni", kappa0=Fr(1, 2)):
        self.kappa0 = Fr(kappa0)            # expansion point; delta = kappa - kappa0
        self.col = [1] if lengths is None else column(list(lengths))
        self.P = len(self.col)
        self.r = Fr(r)
        self.geometry = geometry
        self.layers = [0, 1] if (self.r != 0 and geometry == "annni") else [0]

    def spin(self, site):
        return self.col[site[1] % self.P]

    def bonds_of(self, site, L):
        """[(neighbour, J jet)] for all bonds of a site; J_y2 = -(1/2 + delta)."""
        x, y, l = site
        one = jconst(1, L)
        if self.geometry == "chain":
            return [((x, y + 1, l), one), ((x, y - 1, l), one)]
        j2 = (-self.kappa0, Fr(-1)) + (Fr(0),) * (L - 2) if L >= 2 else (-self.kappa0,)
        out = [((x + 1, y, l), one), ((x - 1, y, l), one), ((x, y + 1, l), one), ((x, y - 1, l), one),
               ((x, y + 2, l), j2), ((x, y - 2, l), j2)]
        if len(self.layers) == 2:
            out.append(((x, y, 1 - l), jconst(self.r, L)))
        return out

    def D(self, site, L):
        s = self.spin(site)
        acc = jconst(0, L)
        for nb, J in self.bonds_of(site, L):
            acc = jadd(acc, jscale(J, 2 * s * self.spin(nb)))
        return acc

    def neighbours(self, site):
        return [nb for nb, _ in self.bonds_of(site, 1)]

    def e0(self, L=4):
        """classical energy per spin (jet)."""
        acc = jconst(0, L)
        for y in range(self.P):
            for l in self.layers:
                site = (0, y, l)
                for nb, J in self.bonds_of(site, L):
                    acc = jsub(acc, jscale(J, Fr(self.spin(site) * self.spin(nb), 2)))   # each bond twice
        return jscale(acc, Fr(1, self.P * len(self.layers)))


def clusters_anchored(struct, anchor, maxsize=4):
    found = {frozenset([anchor])}
    frontier = [frozenset([anchor])]
    for _ in range(maxsize - 1):
        new = []
        for C in frontier:
            for s in C:
                for nb in struct.neighbours(s):
                    if nb > anchor and nb not in C:
                        C2 = C | {nb}
                        if C2 not in found:
                            found.add(C2)
                            new.append(C2)
        frontier = new
    return found


def cluster_graph(struct, C, L):
    sites = sorted(C)
    D = [struct.D(s, L) for s in sites]
    idx = {s: a for a, s in enumerate(sites)}
    B = {}
    for s in sites:
        for nb, J in struct.bonds_of(s, L):
            if nb in idx and idx[s] < idx[nb]:
                key = (idx[s], idx[nb])
                B[key] = jadd(B.get(key, jconst(0, L)), jscale(J, struct.spin(s) * struct.spin(nb)))
    return D, B


def series(struct, order=8, maxsize=4):
    """Per-spin coefficients {n: jet} for n = 2, 4, 6, 8 (jet lengths 4, 3, 2, 1) plus e0 (jet length 4)
    and the number of clusters used."""
    need = {2: 4, 4: 3, 6: 2, 8: 1}          # jet length needed per order
    acc = {n: jconst(0, need[n]) for n in need}
    ncl = 0
    for y in range(struct.P):
        for l in struct.layers:
            for C in clusters_anchored(struct, (0, y, l), maxsize):
                m = len(C)
                if 2 * m > order:
                    continue
                L = 5 - m                         # jet length for this cluster size (4 - m + 1)
                D, B = cluster_graph(struct, C, L)
                w = w_of(D, B, order)
                ncl += 1
                for n in need:
                    if n >= 2 * m:
                        Ln = min(need[n], L)
                        acc[n] = jadd(acc[n], jtrunc(w[n][:Ln], need[n]) if Ln == need[n] else
                                      jtrunc(w[n][:Ln], need[n]))
    ncell = struct.P * len(struct.layers)
    out = {n: jscale(acc[n], Fr(1, ncell)) for n in acc}
    out[0] = struct.e0(4)
    return out, ncl

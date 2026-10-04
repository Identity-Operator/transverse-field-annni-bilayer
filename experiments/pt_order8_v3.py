#!/usr/bin/env python3
"""Rayleigh-Schroedinger perturbation theory to O(g^8) by the linked-cluster expansion (Paper 1, revision C2).

Extends pt_order6_v3.py (unchanged, so its outputs stay reproducible) from connected clusters of <= 3 sites
to <= 4 sites. The identity E[all] = sum_C w(C), w(C) = sum_{S subset C} (-1)^{|C|-|S|} E[S], and the facts
w(C) = 0 for disconnected C and w(C) = O(g^{2|C|}) are those of pt_order6_v3; E8 needs connected clusters
of <= 4 sites. Two changes:
  * the lattice has Lx = 5, so that no 4-site cluster closes around the periodic x direction (with Lx = 4 a
    row of four sites would be a ring);
  * cluster energies are memoised by their local environment (flip costs and pair terms in cluster order),
    so translated copies are solved once. The arithmetic stays exact (Fractions).

--verify:  (1) orders 2-6 equal pt_order6_v3 for FM, <2>, <3>, <23> at kappa = 1/2 and 0.53, r = 0, 1;
           (2) an isolated spin gives E8 = 5/D^7;
           (3) the transverse-field Ising chain gives E8 = -25/16384 per spin;
           (4) brute-force RS on the full 2^12-state space of 12-spin rings and ladders (FM and <2>
               references, J1-J2 chains and a two-leg ladder) equals the linked-cluster sum on the same graph.
--analyze: exact-kappa O(g^8) crossings for Table III and the <23> window edges of Table V; W8 hulls on
           the O(g^6) boundaries <3>|<23>, <23>|<223>, <223>|<2> (does <2223> open at O(g^8)?).
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import itertools
import json
import math
import time
from collections import defaultdict
from fractions import Fraction as Fr

import numpy as np

from phase_pipeline.qmc.sse import bilayer_lattice
from pt_order4_v3 import column, lower_hull
from pt_order6_v3 import exact_couplings, per_spin_series as per_spin_series6, rs_small

OUT = Path("10_results/bilayer_annni_paper_results/reanalysis_v3/perturbation")


def connected_clusters(nb, N, kmax):
    """All connected site sets of size <= kmax (sorted tuples), grown site by site."""
    layer = {(i,) for i in range(N)}
    allc = set(layer)
    for _ in range(kmax - 1):
        nxt = set()
        for C in layer:
            cs = set(C)
            for i in C:
                for j in nb[i]:
                    if j not in cs:
                        nxt.add(tuple(sorted(cs | {j})))
        allc |= nxt
        layer = nxt
    return allc


def series_linked8(N, J, s, order=8):
    """(E0, [E_2, ..., E_order]) totals, connected clusters of <= order/2 sites."""
    nb = defaultdict(dict)
    for (i, j), w in J.items():
        nb[i][j] = w
        nb[j][i] = w
    zero = next(iter(J.values())) * 0
    D = [2 * s[i] * sum((w * s[j] for j, w in nb[i].items()), zero) for i in range(N)]
    E0 = -sum(w * s[i] * s[j] for (i, j), w in J.items())
    memo, env = {}, {}

    def E_of(S):
        if S in env:
            return memo[env[S]]
        Bs = []
        for a, b in itertools.combinations(range(len(S)), 2):
            w = nb[S[a]].get(S[b])
            if w is not None:
                Bs.append(((a, b), w * s[S[a]] * s[S[b]]))
        key = (tuple(D[i] for i in S), tuple(Bs))
        env[S] = key
        if key not in memo:
            memo[key] = rs_small([D[i] for i in S], dict(Bs), order)
        return memo[key]

    tot = [zero] * (order + 1)
    for C in connected_clusters(nb, N, order // 2):
        for r_ in range(1, len(C) + 1):
            sign = (-1) ** (len(C) - r_)
            for S in itertools.combinations(C, r_):
                e = E_of(S)
                for n in range(2 * len(C), order + 1, 2):
                    tot[n] += sign * e[n]
    return E0, [tot[n] for n in range(2, order + 1, 2)], len(memo)


def structure_on_lattice(lengths, kappa, r, Lx=5, min_Ly=10):
    col = [1] if lengths is None else column(list(lengths))
    P = len(col)
    Ly = P * max(1, math.ceil(min_Ly / P))
    lat = bilayer_lattice(Lx, Ly, 2 if Fr(r) != 0 else 1)
    J = exact_couplings(lat, kappa, r)
    s = [col[int(y) % P] for y in lat["ycoord"]]
    return lat, J, s


_cache = {}


def per_spin_series8(lengths, kappa, r):
    """[e0, e2, e4, e6, e8] per spin (Fractions), Lx -> infinity values."""
    key = (None if lengths is None else tuple(lengths), Fr(str(kappa)) if not isinstance(kappa, Fr) else kappa,
           Fr(str(r)) if not isinstance(r, Fr) else r)
    if key not in _cache:
        lat, J, s = structure_on_lattice(key[0], key[1], key[2])
        E0, Es, _ = series_linked8(lat["N"], J, s, 8)
        _cache[key] = [E0 / lat["N"]] + [e / lat["N"] for e in Es]
    return _cache[key]


def crossing_g(eA, eB, order):
    """Smallest g > 0 where the truncated series of A and B cross (in u = g^2)."""
    d = [a - b for a, b in zip(eA, eB)]
    poly = d[:order // 2 + 1]
    roots = np.roots([float(x) for x in poly[::-1]])
    u = sorted(x.real for x in roots if abs(x.imag) < 1e-9 and x.real > 0)
    return math.sqrt(u[0]) if u else float("nan")


# ---------------------------------------------------------------- verification
def rs_full(N, J, s, order):
    """Brute-force RS on all 2^N states (exact), for small graphs."""
    nb = defaultdict(dict)
    for (i, j), w in J.items():
        nb[i][j] = w
        nb[j][i] = w
    M = 1 << N
    zero = next(iter(J.values())) * 0

    def energy(state):
        sp = [s[i] * (-1 if state >> i & 1 else 1) for i in range(N)]
        return -sum(w * sp[i] * sp[j] for (i, j), w in J.items())
    e_ref = energy(0)
    dE = [energy(t) - e_ref for t in range(M)]
    pc = [bin(t).count("1") for t in range(M)]
    if any(e <= 0 for t, e in enumerate(dE) if 0 < pc[t] <= order):   # states beyond `order` flips never enter
        raise ValueError("degenerate reference")
    psi = [[zero] * M for _ in range(order + 1)]
    psi[0][0] = zero + 1
    E = [zero] * (order + 1)
    for n in range(1, order + 1):
        Vpsi = [-sum(psi[n - 1][t ^ (1 << i)] for i in range(N)) for t in range(M)]
        E[n] = Vpsi[0]
        for t in range(1, M):
            if pc[t] <= n:
                psi[n][t] = (Vpsi[t] - sum(E[k] * psi[n - k][t] for k in range(1, n + 1))) / (-dE[t])
    return E


def verify():
    rep = {}
    # (1) orders 2..6 against pt_order6_v3
    ok6 = []
    for r in (0, 1):
        for k in (Fr(1, 2), Fr(53, 100)):
            for seq in (None, (2,), (3,), (2, 3)):
                a = per_spin_series8(seq, k, r)[:4]
                b = per_spin_series6(None if seq is None else list(seq), k, Fr(r), 6)
                ok6.append(a == [Fr(x) for x in b])
    rep["orders_2_6_equal_pt_order6"] = all(ok6)
    # (2) isolated spin
    D = Fr(7, 3)
    rep["isolated_spin_E8_eq_5_over_D7"] = rs_small([D], {}, 8)[8] == 5 / D ** 7
    # (3) transverse-field Ising chain (ring of 12, J = 1): per-spin E8 = -25/16384
    N = 12
    Jc = {(i, (i + 1) % N) if i < (i + 1) % N else ((i + 1) % N, i): Fr(1) for i in range(N)}
    E0, Es, _ = series_linked8(N, Jc, [1] * N, 8)
    rep["tfim_chain_E2_E4_E6_E8_per_spin"] = [str(e / N) for e in Es]
    rep["tfim_chain_E8_eq_-25/16384"] = Es[3] / N == Fr(-25, 16384)
    # (4) brute force on 12-spin graphs
    tests = []
    def ring(N, J1, J2):
        J = {}
        for i in range(N):
            for d, w in ((1, J1), (2, J2)):
                a, b = sorted((i, (i + d) % N))
                J[(a, b)] = J.get((a, b), 0) + w
        return {k: Fr(v) for k, v in J.items() if v != 0}
    cases = [("J1-J2 ring, FM, kappa=0.3", ring(12, 1, Fr(-3, 10)), [1] * 12)]
    rng = np.random.default_rng(7)                    # inhomogeneous couplings, FM reference
    rnd = {}
    for i in range(12):
        for d, w in ((1, Fr(int(rng.integers(5, 16)), 10)), (2, Fr(int(rng.integers(-4, 3)), 10))):
            a, b = sorted((i, (i + d) % 12))
            rnd[(a, b)] = rnd.get((a, b), 0) + w
    cases.append(("random J1-J2 ring, FM", {k: v for k, v in rnd.items() if v != 0}, [1] * 12))
    def ladder(J2):
        lad = {}
        for i in range(6):
            for a, b, w in ((i, (i + 1) % 6, Fr(1)), (6 + i, 6 + (i + 1) % 6, Fr(1)), (i, (i + 2) % 6, J2),
                            (6 + i, 6 + (i + 2) % 6, J2), (i, 6 + i, Fr(1))):
                a, b = sorted((a, b))
                lad[(a, b)] = lad.get((a, b), 0) + w
        return {k: v for k, v in lad.items() if v != 0}
    cases.append(("ladder 2x6 (J1-J2 legs, rungs r=1), FM, kappa=0.3", ladder(Fr(-3, 10)), [1] * 12))
    for name, J, s in cases:
        t0 = time.time()
        bf = rs_full(12, J, s, 8)
        E0, lc, _ = series_linked8(12, J, s, 8)
        eq = [bf[n] == lc[n // 2 - 1] for n in (2, 4, 6, 8)]
        odd0 = all(bf[n] == 0 for n in (1, 3, 5, 7))
        tests.append({"case": name, "equal_E2_E4_E6_E8": eq, "odd_orders_zero": odd0,
                      "E8": str(bf[8]), "s": round(time.time() - t0, 1)})
        print(name, eq, odd0, flush=True)
    rep["brute_force"] = tests
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pt_order8_verify.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: v for k, v in rep.items() if k != "brute_force"}, indent=1, default=str))
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--e8", action="store_true", help="print e8 at kappa = 1/2 for the main structures")
    a = ap.parse_args()
    if a.verify:
        verify()
    if a.e8:
        for r in (0, 1):
            for seq in (None, (2,), (3,), (2, 3)):
                t0 = time.time()
                e = per_spin_series8(seq, Fr(1, 2), r)
                print(f"r={r} {seq}: e8 = {float(e[4]):+.6e}  ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()

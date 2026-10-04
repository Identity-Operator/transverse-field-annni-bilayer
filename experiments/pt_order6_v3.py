#!/usr/bin/env python3
"""Rayleigh-Schroedinger perturbation theory to O(g^6) by a linked-cluster expansion (Paper 1, stage S5a).

H = H0 + g V, H0 = -sum_b J_b s_i s_j (classical; J_b as in sse.class_couplings), V = -sum_i sigma^x_i.
For a classical reference configuration s let E[S](g) be the RS ground-energy series when the transverse
field acts only on the sites in S (all other spins frozen). Moebius inversion over subsets,
    E[all] = sum_C w(C),   w(C) = sum_{S subset C, S nonempty} (-1)^{|C|-|S|} E[S],
is an identity. w(C) = 0 when C is disconnected in the bond graph (E[S] is then additive), and the g^{2n}
coefficient of w(C) vanishes for |C| > n, since every site of C must be flipped and restored. So E6 needs
only connected clusters of <= 3 sites, each a 2^|C|-state problem solved by the RS recursion
    E_n = <0|V|psi_{n-1}>,  psi_n = R [V psi_{n-1} - sum_{k=1..n} E_k psi_{n-k}],  R = Q/(E0 - H0),
in exact rational arithmetic. Flip costs: flipping the set F costs sum_{i in F} D_i - 4 sum_{i<j in F} B_ij,
with D_i = 2 s_i sum_j J_ij s_j and B_ij = J_ij s_i s_j. The reference must have no zero-cost state within
3 flips. The per-spin coefficients are the Lx -> infinity values: for x-uniform structures the cluster sums
on an Lx = 4 lattice already equal them (checked bitwise against (Lx, Ly) = (5, 2 Ly)). This is a statement
about the RS coefficients, not about the exact energy of a finite lattice (see pt_order4_v3.py).

--verify:  E2 and E4 equal the closed form of pt_order4_v3.py; an isolated spin gives E6 = -2/D^5;
           exact diagonalisation gives E6 (random-coupling clusters; FM and <2> on a 4x4 layer).
--analyze: (1) exact-kappa O(g^6) crossings for every L = 12 step-2 case, next to O(g^2), O(g^4) and QMC;
           (2) the O(g^6) splitting of the O(g^4) boundaries <3>|<23> and <23>|<2>. With
               kappa = 1/2 + k g^2 + k1 g^4 + k2 g^6, the g^6 energy per spin is k2 (1 - 4 rho) + W6 with
               W6 = k1 e2' + (k^2/2) e2'' + k e4' + e6 (kappa derivatives at 1/2). Among the sequences that
               are degenerate on a boundary at O(g^4), the lower hull of W6 against rho decides which
               (e.g. <233>, <223>) appear at O(g^6).
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
from pt_order4_v3 import (column, coeffs, cyclic_sequences, ed_ground, lower_hull, merged_couplings,
                          per_column, series_lattice)

BR = Path("10_results/bilayer_annni_paper_results/reanalysis_v3/sse_validation/branches")
OUT = Path("10_results/bilayer_annni_paper_results/reanalysis_v3/perturbation")


# ---------------------------------------------------------------- linked-cluster RS
def exact_couplings(lat, kappa, r):
    """Merged couplings {(i, j): J} with Fraction values (zero couplings dropped)."""
    Jc = [Fr(1), Fr(1), -Fr(kappa), Fr(r)]
    J = defaultdict(Fr)
    for i, j, c in zip(lat["bi"], lat["bj"], lat["bcls"]):
        a, b = (int(i), int(j)) if i < j else (int(j), int(i))
        J[(a, b)] += Jc[int(c)]
    return {k: v for k, v in J.items() if v != 0}


def rs_small(D, B, order=6):
    """RS coefficients E_1..E_order (V = -sum sigma^x) for m <= 3 spins with single-flip costs D[i] and
    pair terms B[(i, j)]; all other spins frozen."""
    m = len(D)
    M = 1 << m
    dE = []
    for s in range(M):
        bits = [i for i in range(m) if s >> i & 1]
        e = sum(D[i] for i in bits) - 4 * sum(B.get((i, j), 0) for i, j in itertools.combinations(bits, 2))
        dE.append(e)
    if any(e <= 0 for e in dE[1:]):
        raise ValueError("non-positive excitation energy in a cluster")
    zero = D[0] - D[0]
    psi = [[zero] * M for _ in range(order + 1)]
    psi[0][0] = zero + 1
    E = [zero] * (order + 1)
    for n in range(1, order + 1):
        Vpsi = [-sum(psi[n - 1][s ^ (1 << i)] for i in range(m)) for s in range(M)]
        E[n] = Vpsi[0]
        for s in range(1, M):
            acc = Vpsi[s] - sum(E[k] * psi[n - k][s] for k in range(1, n + 1))
            psi[n][s] = acc / (-dE[s])
    return E


def series_linked(N, J, s, order=6):
    """(E0, [E_2, E_4, E_6]) totals by the linked-cluster expansion (connected clusters of <= order/2 sites)."""
    nb = defaultdict(dict)
    for (i, j), w in J.items():
        nb[i][j] = w
        nb[j][i] = w
    zero = next(iter(J.values())) * 0
    D = [2 * s[i] * sum((w * s[j] for j, w in nb[i].items()), zero) for i in range(N)]
    E0 = -sum(w * s[i] * s[j] for (i, j), w in J.items())
    memo = {}

    def E_of(S):
        S = tuple(sorted(S))
        if S not in memo:
            Bs = {}
            for a, b in itertools.combinations(range(len(S)), 2):
                w = nb[S[a]].get(S[b])
                if w is not None:
                    Bs[(a, b)] = w * s[S[a]] * s[S[b]]
            memo[S] = rs_small([D[i] for i in S], Bs, order)
        return memo[S]

    clusters = [(i,) for i in range(N)] + [tuple(k) for k in J]
    if order >= 6:
        trip = set()
        for j in range(N):
            for a, b in itertools.combinations(sorted(nb[j]), 2):
                trip.add(tuple(sorted((a, j, b))))
        clusters += sorted(trip)
    tot = [zero] * (order + 1)
    for C in clusters:
        if 2 * len(C) > order:
            continue
        for r_ in range(1, len(C) + 1):
            sign = (-1) ** (len(C) - r_)
            for S in itertools.combinations(C, r_):
                e = E_of(S)
                for n in range(2, order + 1, 2):
                    if 2 * len(C) <= n:
                        tot[n] += sign * e[n]
    return E0, [tot[n] for n in range(2, order + 1, 2)]


def structure_on_lattice(lengths, kappa, r, Lx=4, min_Ly=10):
    """Lattice, exact couplings and spins for an x-uniform, parallel-stacked structure (None = FM)."""
    col = [1] if lengths is None else column(list(lengths))
    P = len(col)
    Ly = P * max(1, math.ceil(min_Ly / P))
    lat = bilayer_lattice(Lx, Ly, 2 if Fr(r) != 0 else 1)
    J = exact_couplings(lat, kappa, r)
    s = [col[int(y) % P] for y in lat["ycoord"]]
    return lat, J, s


def per_spin_series(lengths, kappa, r, order=6):
    lat, J, s = structure_on_lattice(lengths, kappa, r)
    E0, Es = series_linked(lat["N"], J, s, order)
    return [E0 / lat["N"]] + [e / lat["N"] for e in Es]


# ---------------------------------------------------------------- dual numbers for kappa derivatives
class Dual:
    """a + b eps with eps^2 = 0 over Fractions; used for d/dkappa of the O(g^4) closed form."""
    __slots__ = ("a", "b")

    def __init__(self, a, b=0):
        self.a, self.b = Fr(a), Fr(b)

    @staticmethod
    def _c(o):
        return o if isinstance(o, Dual) else Dual(o)

    def __add__(self, o): o = self._c(o); return Dual(self.a + o.a, self.b + o.b)
    __radd__ = __add__
    def __sub__(self, o): o = self._c(o); return Dual(self.a - o.a, self.b - o.b)
    def __rsub__(self, o): return self._c(o) - self
    def __mul__(self, o): o = self._c(o); return Dual(self.a * o.a, self.a * o.b + self.b * o.a)
    __rmul__ = __mul__
    def __truediv__(self, o): o = self._c(o); return Dual(self.a / o.a, (self.b * o.a - self.a * o.b) / (o.a * o.a))
    def __rtruediv__(self, o): return self._c(o) / self
    def __neg__(self): return Dual(-self.a, -self.b)
    def __pow__(self, n):
        out = Dual(1)
        for _ in range(n):
            out = out * self
        return out
    def __lt__(self, o): return self.a < self._c(o).a
    def __le__(self, o): return self.a <= self._c(o).a
    def __gt__(self, o): return self.a > self._c(o).a
    def __ge__(self, o): return self.a >= self._c(o).a
    def __eq__(self, o): return self.a == self._c(o).a and self.b == self._c(o).b
    def __ne__(self, o): return not self == o
    __hash__ = None


def derivs_at_half(lengths, r):
    """(e2', e2'', e4') at kappa = 1/2 for a straight-wall sequence (None = FM), exact."""
    R = Fr(r)
    col = [1] * 2 if lengths is None else column(list(lengths))
    P = len(col)
    h0 = [2 + R + col[y] * (col[(y + 1) % P] + col[(y - 1) % P]) for y in range(P)]
    c = [col[y] * (col[(y + 2) % P] + col[(y - 2) % P]) for y in range(P)]
    h = [h0[y] - Fr(1, 2) * c[y] for y in range(P)]
    e2p = -sum(Fr(c[y]) / (2 * h[y] ** 2) for y in range(P)) / P
    e2pp = -sum(Fr(c[y] ** 2) / h[y] ** 3 for y in range(P)) / P
    if lengths is None:
        e4p = coeffs("FM", Dual(Fr(1, 2), 1), Dual(R))[2].b
    else:
        e4p = per_column(list(lengths), Dual(Fr(1, 2), 1), Dual(R))[2].b
    return e2p, e2pp, e4p


# ---------------------------------------------------------------- verification
def verify():
    rep = {"linked_vs_closed_form": [], "isolated_spin": None, "vs_ED": []}
    D = Fr(7, 3)
    e = rs_small([D], {})
    rep["isolated_spin"] = {"E2": str(e[2]), "E4": str(e[4]), "E6": str(e[6]),
                            "expected": [str(-1 / D), str(1 / D ** 3), str(-2 / D ** 5)]}
    assert e[2] == -1 / D and e[4] == 1 / D ** 3 and e[6] == -2 / D ** 5 and e[1] == e[3] == e[5] == 0
    print(f"[isolated spin] E6 = {e[6]} = -2/D^5: ok")
    for name, seq, kappa, r in [("FM", None, Fr(1, 4), 0), ("FM", None, Fr(1, 2), 1), ("<2>", (2,), Fr(3, 4), 0),
                                ("<2>", (2,), Fr(1, 2), 1), ("<3>", (3,), Fr(1, 2), 0), ("<3>", (3,), Fr(3, 5), 1),
                                ("<23>", (2, 3), Fr(11, 20), 1), ("<233>", (2, 3, 3), Fr(1, 2), 0)]:
        lat, J, s = structure_on_lattice(seq, kappa, r)
        E0, (E2, E4, E6) = series_linked(lat["N"], J, s)
        Jf = {k: float(v) for k, v in J.items()}
        a = series_lattice(lat["N"], Jf, np.array(s, dtype=np.int64))
        N = lat["N"]
        ok = abs(float(E2) - a[1]) < 1e-10 and abs(float(E4) - a[2]) < 1e-10
        rep["linked_vs_closed_form"].append({"structure": name, "kappa": str(kappa), "r": r, "E2": float(E2) / N,
                                             "E4": float(E4) / N, "E4_closed": a[2] / N, "E6": float(E6) / N, "ok": ok})
        print(f"[linked vs closed] {name:5s} k={str(kappa):5s} r={r}: e2 {float(E2)/N:+.10f} vs {a[1]/N:+.10f}  "
              f"e4 {float(E4)/N:+.10f} vs {a[2]/N:+.10f}  e6 {float(E6)/N:+.10f}  {'ok' if ok else 'MISMATCH'}", flush=True)
    # ED: random couplings (N = 10) and 4x4 layers
    rng = np.random.default_rng(11)
    systems = []
    while len(systems) < 3:
        N = 10
        J = {}
        for i in range(N):
            for j in range(i + 1, N):
                if rng.random() < 0.3:
                    J[(i, j)] = Fr(int(rng.integers(-15, 16)), 10)
        J = {k: v for k, v in J.items() if v != 0}
        conf = 1 - 2 * ((np.arange(1 << N)[:, None] >> np.arange(N)) & 1)
        Ef = np.array([-sum(float(w) * c[i] * c[j] for (i, j), w in J.items()) for c in conf])
        o = np.argsort(Ef)
        if Ef[o[2]] - Ef[o[0]] < 0.8:
            continue
        systems.append((f"random N=10 #{len(systems)}", N, J, [int(v) for v in conf[o[0]]]))
    for kappa, seq, name in ((Fr(1, 4), None, "4x4 FM kappa=1/4"), (Fr(3, 4), (2,), "4x4 <2> kappa=3/4")):
        lat = bilayer_lattice(4, 4, 1)
        J = exact_couplings(lat, kappa, 0)
        col = [1] if seq is None else column(list(seq))
        systems.append((name, lat["N"], J, [col[int(y) % len(col)] for y in lat["ycoord"]]))
    for name, N, J, s in systems:
        E0, (E2, E4, E6) = series_linked(N, J, s)
        Jf = {k: float(v) for k, v in J.items()}
        gs = np.linspace(0.02, 0.2, 14) if name.startswith("random") else np.linspace(0.05, 0.5, 14)
        Eg = ed_ground(N, Jf, gs)
        y = Eg - float(E0) - float(E2) * gs ** 2 - float(E4) * gs ** 4
        X = np.vstack([gs ** k for k in (6, 8, 10, 12)]).T
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        rep["vs_ED"].append({"system": name, "E6": float(E6), "E6_ED": float(c[0]), "rel": float(c[0] / float(E6) - 1)})
        print(f"[ED] {name:20s}: E6 {float(E6):+.8f} vs ED fit {c[0]:+.8f}  (rel {c[0]/float(E6)-1:+.1e})", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pt_order6_verify.json").write_text(json.dumps(rep, indent=1, default=str))
    print("wrote", OUT / "pt_order6_verify.json")


# ---------------------------------------------------------------- analysis
def crossing_g(eA, eB, order):
    d = [a - b for a, b in zip(eA, eB)]          # coefficients of 1, g^2, g^4, g^6
    poly = {2: d[:2], 4: d[:3], 6: d[:4]}[order]
    roots = np.roots([float(x) for x in poly[::-1]])
    u = sorted(x.real for x in roots if abs(x.imag) < 1e-9 and x.real > 0)
    return math.sqrt(u[0]) if u else float("nan")


def analyze(max_blocks):
    out = {"crossings": [], "boundaries": {}}
    SEQ = {"FM": None, "<2>": (2,), "<3>": (3,)}
    # (1) crossings for the L = 12 step-2 cases
    for fj in sorted(BR.glob("branches_L12_k*_r*_free_energy.json")):
        d = json.loads(fj.read_text())
        for key, grp in d["groups"].items():
            kappa = Fr(key.split(",")[0].split("=")[1]); r = abs(Fr(key.split(",")[1].split("=")[1]))
            A, B, pair = ("<3>", "<2>", "ascending:<2>par vs ascending:<3>par") if kappa > Fr(1, 2) else \
                         ("<3>", "FM", "ascending:<3>par vs ascending:FM")
            eA, eB = per_spin_series(SEQ[A], kappa, r), per_spin_series(SEQ[B], kappa, r)
            row = {"kappa": float(kappa), "r": float(r), "side": f"{A}|{B}", "qmc_unanchored": grp["pairs"][pair]["g_star"],
                   **{f"pt_g{o}": crossing_g(eA, eB, o) for o in (2, 4, 6)},
                   "e6": {A: float(eA[3]), B: float(eB[3])}}
            out["crossings"].append(row)
            print(f"k={float(kappa):<6} r={float(r):g} {A}|{B:4s} QMC(unanch.) {row['qmc_unanchored']:.4f}   "
                  f"O(g^2) {row['pt_g2']:.4f}  O(g^4) {row['pt_g4']:.4f}  O(g^6) {row['pt_g6']:.4f}", flush=True)
    # (2) O(g^6) splitting of the O(g^4) boundaries
    half = Fr(1, 2)
    for r in (Fr(0), Fr(1)):
        F3 = Fr(6) / ((2 + r) * (3 + r) * (5 + r))
        k = F3 / 4
        W = {}
        base = {(3,): None, (2, 3): None, (2,): None}
        for seq in base:
            e0, e2, e4, e2p = per_column(list(seq), half, r)
            W[seq] = (Fr(len(seq), sum(seq)), k * e2p + e4)
        (r3, W3), (r23, W23), (r2, W2) = W[(3,)], W[(2, 3)], W[(2,)]
        k1 = {"3|23": (W23 - W3) / (r23 - r3) / 4, "23|2": (W2 - W23) / (r2 - r23) / 4}
        res_r = {}
        for bname, blocks in (("3|23", ((3,), (2, 3))), ("23|2", ((2,), (2, 3)))):
            # sequences built from the two blocks, up to max_blocks blocks (both blocks present or pure)
            cands = set()
            for nbk in range(1, max_blocks + 1):
                for combo in itertools.product(range(2), repeat=nbk):
                    seq = tuple(x for c_ in combo for x in blocks[c_])
                    n = len(seq)
                    p = next(q for q in range(1, n + 1) if n % q == 0 and seq[:q] * (n // q) == seq)
                    prim = seq[:p]                      # primitive word (e.g. '2323' -> '23')
                    cands.add(min(prim[i:] + prim[:i] for i in range(len(prim))))
            pts, skipped = [], []
            for seq in sorted(cands, key=lambda t: (len(t), t)):
                P = len(column(list(seq)))
                if P > 40:
                    skipped.append("".join(map(str, seq)))
                    continue
                e0, e2, e4, e2p = per_column(list(seq), half, r)
                rho = Fr(len(seq), sum(seq))
                g4 = k1[bname] * (1 - 4 * rho) + k * e2p + e4
                t0 = time.time()
                e6 = per_spin_series(seq, half, r)[3]
                e2p_, e2pp, e4p = derivs_at_half(seq, r)
                assert e2p_ == e2p
                W6 = k1[bname] * e2p + k * k * e2pp / 2 + k * e4p + e6
                pts.append((rho, W6, "".join(map(str, seq)), g4, time.time() - t0))
            g4ref = pts[0][3]
            nondeg = [p[2] for p in pts if p[3] != g4ref]
            deg = [p for p in pts if p[3] == g4ref]
            hull = lower_hull([(p[0], p[1], p[2]) for p in deg])
            ends = sorted(deg, key=lambda p: p[0])
            lo, hi = ends[0], ends[-1]
            chord = lambda x: lo[1] + (hi[1] - lo[1]) * (x - lo[0]) / (hi[0] - lo[0])
            depth = sorted(((p[2], float(p[1] - chord(p[0]))) for p in deg), key=lambda t: t[1])
            res_r[bname] = {"k1": str(k1[bname]), "n_candidates": len(pts), "skipped_period_gt_40": skipped,
                            "not_degenerate_at_g4": nondeg,
                            "hull_at_g6": [h[2] for h in hull], "depth_below_end_chord": depth[:8]}
            print(f"r={r} boundary {bname}: {len(deg)} primitive sequences degenerate at O(g^4) (non-degenerate: {nondeg};"
                  f" skipped, period > 40: {skipped});"
                  f" O(g^6) hull: {[h[2] for h in hull]}", flush=True)
            print(f"      lowest relative to the chord of the two end phases (per spin, x g^6): {depth[:5]}", flush=True)
        out["boundaries"][f"r={r}"] = res_r
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pt_order6_analysis.json").write_text(json.dumps(out, indent=1, default=str))
    print("wrote", OUT / "pt_order6_analysis.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--max-blocks", type=int, default=5)
    a = ap.parse_args()
    if a.verify:
        verify()
    if a.analyze:
        analyze(a.max_blocks)


if __name__ == "__main__":
    main()

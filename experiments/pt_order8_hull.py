#!/usr/bin/env python3
"""O(g^8) phase boundaries near the multiphase point: which structures open at eighth order (Paper 1, revision C2).

Along a boundary kappa = 1/2 + k g^2 + k1 g^4 + k2 g^6 + k3 g^8, the energy per spin of a straight-wall sequence
at order g^8 is k3 (1 - 4 rho) + W8 with
    W8 = k2 e2' + k k1 e2'' + (k^3/6) e2''' + k1 e4' + (k^2/2) e4'' + k e6' + e8,
primes being kappa derivatives at kappa = 1/2 (cf. W4, W6 in pt_order6_v3.py). The derivatives are exact:
the linked-cluster sums of pt_order8_v3 are evaluated with kappa = 1/2 + d as a truncated Taylor series in d
(class TPS, degree 3), in exact Fractions; e8 at kappa = 1/2 comes from pt_order8_v3.per_spin_series8.

For each O(g^6) boundary A|B (<3>|<23>, <23>|<223>, <223>|<2>) every primitive sequence built from the
blocks A and B (up to --max-blocks blocks) is placed relative to the chord from A to B in the (rho, energy)
plane, at orders g^4, g^6 (both must vanish for a candidate of the boundary) and g^8. A negative g^8 deviation
means the sequence opens a window at O(g^8).

Check: the TPS derivatives reproduce pt_order6_v3.derivs_at_half (e2', e2'', e4') and the O(g^6) e6 exactly.
Output: reanalysis_v3/perturbation/pt_order8_hull.json
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
from collections import defaultdict
from fractions import Fraction as Fr

from phase_pipeline.qmc.sse import bilayer_lattice
from pt_order4_v3 import column
from pt_order6_v3 import derivs_at_half, per_spin_series as per_spin_series6
from pt_order8_v3 import OUT, per_spin_series8, series_linked8


class TPS:
    """c0 + c1 d + c2 d^2 + c3 d^3 (d = kappa - 1/2), exact, truncated at degree DEG."""
    DEG = 3
    __slots__ = ("c",)

    def __init__(self, c):
        c = [Fr(x) for x in (c if isinstance(c, (list, tuple)) else [c])]
        self.c = (c + [Fr(0)] * (self.DEG + 1))[:self.DEG + 1]

    @classmethod
    def _t(cls, o):
        return o if isinstance(o, TPS) else TPS(o)

    def __add__(self, o): o = self._t(o); return TPS([a + b for a, b in zip(self.c, o.c)])
    __radd__ = __add__
    def __sub__(self, o): o = self._t(o); return TPS([a - b for a, b in zip(self.c, o.c)])
    def __rsub__(self, o): return self._t(o) - self
    def __neg__(self): return TPS([-a for a in self.c])

    def __mul__(self, o):
        o = self._t(o)
        return TPS([sum(self.c[i] * o.c[n - i] for i in range(n + 1)) for n in range(self.DEG + 1)])
    __rmul__ = __mul__

    def inv(self):
        b0 = self.c[0]
        x = TPS([0] + [a / b0 for a in self.c[1:]])
        out, term = TPS(1), TPS(1)
        for _ in range(self.DEG):
            term = term * (-x)
            out = out + term
        return out * (1 / b0)

    def __truediv__(self, o): return self * self._t(o).inv()
    def __rtruediv__(self, o): return self._t(o) * self.inv()
    def __le__(self, o): return self.c[0] <= self._t(o).c[0]
    def __lt__(self, o): return self.c[0] < self._t(o).c[0]
    def __ge__(self, o): return self.c[0] >= self._t(o).c[0]
    def __gt__(self, o): return self.c[0] > self._t(o).c[0]
    def __eq__(self, o): return self.c == self._t(o).c
    def __ne__(self, o): return not self == o
    def __hash__(self): return hash(tuple(self.c))

    def deriv(self, n):
        return self.c[n] * math.factorial(n)


def per_spin_series_tps(lengths, r, order=6, Lx=5, min_Ly=10):
    """[e0, e2, ..., e_order] per spin as TPS in d = kappa - 1/2."""
    col = [1] if lengths is None else column(list(lengths))
    P = len(col)
    Ly = P * max(1, math.ceil(min_Ly / P))
    R = Fr(r)
    lat = bilayer_lattice(Lx, Ly, 2 if R != 0 else 1)
    Jc = [TPS(1), TPS(1), -TPS([Fr(1, 2), 1]), TPS(R)]
    J = defaultdict(lambda: TPS(0))
    for i, j, c in zip(lat["bi"], lat["bj"], lat["bcls"]):
        a, b = (int(i), int(j)) if i < j else (int(j), int(i))
        J[(a, b)] = J[(a, b)] + Jc[int(c)]
    J = {kk: v for kk, v in J.items() if v != TPS(0)}
    s = [col[int(y) % P] for y in lat["ycoord"]]
    E0, Es, _ = series_linked8(lat["N"], J, s, order)
    n = Fr(1, lat["N"])
    return [E0 * n] + [e * n for e in Es]


def canon(seq):
    n = len(seq)
    p = next(q for q in range(1, n + 1) if n % q == 0 and seq[:q] * (n // q) == seq)
    prim = seq[:p]
    return min(prim[i:] + prim[:i] for i in range(len(prim)))


def check():
    ok = []
    for seq in ((3,), (2, 3), (2,)):
        for r in (0, 1):
            s = per_spin_series_tps(seq, r, 6)
            e2p, e2pp, e4p = derivs_at_half(list(seq), Fr(r))
            e6 = Fr(per_spin_series6(list(seq), Fr(1, 2), Fr(r), 6)[3])
            ok.append(s[1].deriv(1) == e2p and s[1].deriv(2) == e2pp and s[2].deriv(1) == e4p and s[3].c[0] == e6)
    return all(ok)


def hull8(max_blocks):
    res = {"tps_check_vs_pt_order6": check()}
    print("TPS derivatives equal pt_order6 closed forms:", res["tps_check_vs_pt_order6"], flush=True)
    for r in (0, 1):
        R = Fr(r)
        k = Fr(6) / ((2 + R) * (3 + R) * (5 + R)) / 4
        data = {}

        def get(seq):
            if seq not in data:
                ser = per_spin_series_tps(seq, r, 6)
                data[seq] = dict(rho=Fr(len(seq), sum(seq)), e2=ser[1], e4=ser[2], e6=ser[3],
                                 e8=per_spin_series8(seq, Fr(1, 2), R)[4])
            return data[seq]

        def W4(d):
            return k * d["e2"].deriv(1) + d["e4"].c[0]

        def W6(d, k1):
            return k1 * d["e2"].deriv(1) + k * k * d["e2"].deriv(2) / 2 + k * d["e4"].deriv(1) + d["e6"].c[0]

        def W8(d, k1, k2):
            return (k2 * d["e2"].deriv(1) + k * k1 * d["e2"].deriv(2) + k ** 3 * d["e2"].deriv(3) / 6
                    + k1 * d["e4"].deriv(1) + k * k * d["e4"].deriv(2) / 2 + k * d["e6"].deriv(1) + d["e8"])

        d3, d23, d2 = get((3,)), get((2, 3)), get((2,))
        k1m = (W4(d23) - W4(d3)) / (4 * (d23["rho"] - d3["rho"]))
        k1p = (W4(d2) - W4(d23)) / (4 * (d2["rho"] - d23["rho"]))
        out_r = {"k": str(k), "k1-": str(k1m), "k1+": str(k1p)}
        for name, A, B, k1 in (("3|23", (3,), (2, 3), k1m), ("23|223", (2, 3), (2, 2, 3), k1p),
                               ("223|2", (2, 2, 3), (2,), k1p)):
            dA, dB = get(A), get(B)
            k2 = (W6(dB, k1) - W6(dA, k1)) / (4 * (dB["rho"] - dA["rho"]))

            def tot(d, order):     # energy per spin at the given order on the boundary line
                return {4: k1 * (1 - 4 * d["rho"]) + W4(d), 6: k2 * (1 - 4 * d["rho"]) + W6(d, k1),
                        8: W8(d, k1, k2)}[order]
            cands = set()
            for nbk in range(2, max_blocks + 1):
                for combo in itertools.product((A, B), repeat=nbk):
                    seq = canon(tuple(x for blk in combo for x in blk))
                    if seq not in (canon(A), canon(B)):
                        cands.add(seq)
            rows = []
            for seq in sorted(cands, key=lambda q: (len(q), q)):
                if len(column(list(seq))) > 40:
                    continue
                d = get(seq)
                x = (d["rho"] - dA["rho"]) / (dB["rho"] - dA["rho"])
                dev = {o: tot(d, o) - ((1 - x) * tot(dA, o) + x * tot(dB, o)) for o in (4, 6, 8)}
                rows.append({"seq": "".join(map(str, seq)), "rho": str(d["rho"]), "x": str(x),
                             "dev_g4": str(dev[4]), "dev_g6": str(dev[6]), "dev_g8": float(dev[8])})
                print(f"r={r} {name:7s} <{''.join(map(str, seq))}>  dev: g4 {float(dev[4]):+.2e}  g6 {float(dev[6]):+.3e}"
                      f"  g8 {float(dev[8]):+.4e}", flush=True)
            out_r[name] = {"k1": str(k1), "k2": str(k2), "k2_float": float(k2), "candidates": rows}
        res[f"r={r}"] = out_r
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pt_order8_hull.json").write_text(json.dumps(res, indent=1, default=str))
    print("wrote", OUT / "pt_order8_hull.json")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-blocks", type=int, default=4)
    hull8(ap.parse_args().max_blocks)

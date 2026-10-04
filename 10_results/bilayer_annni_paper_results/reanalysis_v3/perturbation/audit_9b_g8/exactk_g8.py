"""Exact-kappa series through g^8 at the S5a kappa values; <23> window edges and <3>|<2> crossing at O(g^6) and
O(g^8), and <233> at kappa = 0.5625, r = 0 (lc8.py with kappa0 = kappa)."""
import json, sys
from fractions import Fraction as Fr
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import lc8
def co(seq, kap, r):
    s, _ = lc8.series(lc8.Structure(seq, r, kappa0=kap))
    return np.array([float(s[n][0]) for n in (0, 2, 4, 6, 8)])
def cross(a, b, order, lo, hi):
    d = (a - b)[: order // 2 + 1]
    u = sorted(x.real for x in np.roots(d[::-1]) if abs(x.imag) < 1e-12 and x.real > 0)
    return [round(float(np.sqrt(x)), 4) for x in u if lo ** 2 <= x <= hi ** 2]
out = {}
for kap, r in ((Fr(52, 100), 0), (Fr(53, 100), 0), (Fr(545, 1000), 0), (Fr(5625, 10000), 0), (Fr(52, 100), 1), (Fr(53, 100), 1), (Fr(545, 1000), 1)):
    c = {n: co(q, kap, r) for n, q in (("<2>", (2,)), ("<3>", (3,)), ("<23>", (2, 3)), ("<233>", (2, 3, 3)))}
    row = {}
    for o in (6, 8):
        row[f"O{o}"] = {"3|2": cross(c["<3>"], c["<2>"], o, 0.3, 1.6), "23|2": cross(c["<23>"], c["<2>"], o, 0.3, 1.6),
                        "23|3": cross(c["<23>"], c["<3>"], o, 0.3, 1.6), "23|233": cross(c["<23>"], c["<233>"], o, 0.3, 1.6),
                        "233|3": cross(c["<233>"], c["<3>"], o, 0.3, 1.6)}
    out[f"kappa={float(kap)},r={r}"] = row
    print(f"kappa={float(kap):<6} r={r}: " + " | ".join(f"O{o}: " + ", ".join(f"{k} {v}" for k, v in row[f'O{o}'].items()) for o in (6, 8)), flush=True)
json.dump(out, open(Path(__file__).parent / "exactk_g8.json", "w"), indent=1)

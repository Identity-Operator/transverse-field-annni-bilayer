"""Table values for the draft subsection (kappa = 1/2, per spin), from repo functions only. Output: pt_table.json."""
import sys, json
from fractions import Fraction as Fr
sys.path[:0] = ["experiments", "."]
from pt_order4_v3 import per_column, coeffs
import pt_order6_v3 as P6
half = Fr(1, 2); out = {}
for r in (Fr(0), Fr(1)):
    F = Fr(6) / ((2 + r) * (3 + r) * (5 + r)); k = F / 4; rows = {}
    for name, seq in (("FM", None), ("<2>", (2,)), ("<23>", (2, 3)), ("<3>", (3,))):
        e0, e2, e4, e2p = coeffs("FM", half, r) if seq is None else per_column(list(seq), half, r)
        e6 = P6.per_spin_series(seq, half, r)[3]
        rho = Fr(0) if seq is None else Fr(len(seq), sum(seq))
        rows[name] = {"rho": str(rho), "eps2": str(e2), "eps4": str(e4), "W4": str(k * e2p + e4), "eps6": float(e6)}
        print(f"r={r} {name:5s} rho={str(rho):4s} eps2={str(e2):>8s} eps4={str(e4):>16s} W4={str(k*e2p+e4):>16s} eps6={float(e6):+.4e}")
    out[f"r={r}"] = rows
json.dump(out, open("/tmp/claude-1000/-home-hoang-nguyen-Phase-Classification/8f62f05f-df21-4fe1-94fa-831f10b083be/scratchpad/pt_table.json", "w"), indent=1)

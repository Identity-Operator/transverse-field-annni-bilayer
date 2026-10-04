"""O(g^6) corrections k2 to the <23> window edges, the <223> window and the <233> margin, in exact rationals, using
the repo functions of pt_order4_v3 / pt_order6_v3 (so every number traces to them). Output: pt6_edges.json."""
import sys, json
from fractions import Fraction as Fr
sys.path[:0] = ["experiments", "."]
from pt_order4_v3 import per_column
import pt_order6_v3 as P6
half = Fr(1, 2); out = {}
SEQ = {"<2>": (2,), "<3>": (3,), "<23>": (2, 3), "<223>": (2, 2, 3), "<233>": (2, 3, 3)}
for r in (Fr(0), Fr(1)):
    F = Fr(6) / ((2 + r) * (3 + r) * (5 + r)); k = F / 4
    base = {n: per_column(list(q), half, r) for n, q in SEQ.items()}
    rho = {n: Fr(len(q), sum(q)) for n, q in SEQ.items()}
    W4 = {n: k * base[n][3] + base[n][2] for n in SEQ}
    k1m = (W4["<23>"] - W4["<3>"]) / (4 * (rho["<23>"] - rho["<3>"]))       # <3>|<23>
    k1p = (W4["<2>"] - W4["<23>"]) / (4 * (rho["<2>"] - rho["<23>"]))       # <23>|<2>
    e6 = {n: P6.per_spin_series(q, half, r)[3] for n, q in SEQ.items()}
    der = {n: P6.derivs_at_half(q, r) for n, q in SEQ.items()}
    def W6(n, k1):
        e2p, e2pp, e4p = der[n]
        return k1 * e2p + k * k * e2pp / 2 + k * e4p + e6[n]
    def k2(A, B, k1):   # boundary between A and B at O(g^6), given both degenerate through O(g^4)
        return (W6(B, k1) - W6(A, k1)) / (4 * (rho[B] - rho[A]))
    lo = k2("<3>", "<23>", k1m)                       # <3>|<23>
    a, b = k2("<23>", "<223>", k1p), k2("<223>", "<2>", k1p)
    w233 = W6("<233>", k1m) - (W6("<3>", k1m) + (W6("<23>", k1m) - W6("<3>", k1m)) * (rho["<233>"] - rho["<3>"]) / (rho["<23>"] - rho["<3>"]))
    w223 = W6("<223>", k1p) - (W6("<23>", k1p) + (W6("<2>", k1p) - W6("<23>", k1p)) * (rho["<223>"] - rho["<23>"]) / (rho["<2>"] - rho["<23>"]))
    res = {"k": str(k), "k1_3|23": str(k1m), "k1_23|2": str(k1p),
           "k2_3|23": float(lo), "k2_23|223": float(a), "k2_223|2": float(b),
           "width_23_g4": float(k1p - k1m), "width_23_g6coef": float(a - lo),
           "width_23_rel_g6_over_g4": float((a - lo) / (k1p - k1m)),
           "width_223_g6": float(b - a), "height_223_vs_chord_23_2": float(w223), "height_233_vs_chord_3_23": float(w233),
           "e6_at_half": {n: float(v) for n, v in e6.items()}}
    out[f"r={r}"] = res
    print(f"r={r}: k2(3|23)={float(lo):+.5e} k2(23|223)={float(a):+.5e} k2(223|2)={float(b):+.5e}  "
          f"<23> width = {float(k1p-k1m):.6f} g^4 + {float(a-lo):.5e} g^6 (ratio {float((a-lo)/(k1p-k1m)):.3f} g^2);  "
          f"<223> width {float(b-a):.3e} g^6; <223> depth {float(w223):+.4e}; <233> height {float(w233):+.4e}", flush=True)
json.dump(out, open("/tmp/claude-1000/-home-hoang-nguyen-Phase-Classification/8f62f05f-df21-4fe1-94fa-831f10b083be/scratchpad/pt6_edges.json", "w"), indent=1)

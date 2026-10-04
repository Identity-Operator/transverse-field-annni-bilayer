"""Independent O(g^6) test on the two O(g^4) window edges: W6 = k1 e2' + k^2 e2''/2 + k e4' + e6 (kappa = 1/2)."""
import sys
from fractions import Fraction as Fr
sys.path.insert(0, "/tmp/claude-1000/-home-hoang-nguyen-Phase-Classification/8f62f05f-df21-4fe1-94fa-831f10b083be/scratchpad")
import indep_rs as I
from indep_e6 import per_spin as e6_float
half = Fr(1, 2); eps = Fr(1, 10 ** 9)
def parts(seq, r):
    R = Fr(r)
    e0, e2, e4, e2p, e0p = I.coeffs(seq, half, R)
    N, b, s = I.structure(seq, half, R)
    _, b0, _ = I.structure(seq, Fr(0), R); _, b1, _ = I.structure(seq, Fr(1), R)
    def Dv(bonds):
        v = [Fr(0)] * N
        for i, j, J in bonds:
            v[i] += 2 * J * s[i] * s[j]; v[j] += 2 * J * s[i] * s[j]
        return v
    D, D0, D1 = Dv(b), Dv(b0), Dv(b1)
    e2pp = -sum(2 * (D1[i] - D0[i]) ** 2 / D[i] ** 3 for i in range(N)) / N
    ep = I.coeffs(seq, half + eps, R)[2]; em = I.coeffs(seq, half - eps, R)[2]
    e4p = (ep - em) / (2 * eps)
    e6 = e6_float(seq, 0.5, r)[0][6]
    rho = Fr(len(seq), sum(seq))
    return rho, float(e2p), float(e2pp), float(e4p), e6
for r in (0, 1):
    R = Fr(r); F = Fr(6) / ((2 + R) * (3 + R) * (5 + R)); k = F / 4
    k1 = {"<23>|<2>": Fr(6011, 336000) if r == 0 else Fr(1961, 725760), "<3>|<23>": Fr(20591, 2016000) if r == 0 else Fr(167, 107520)}
    P = {n: parts(q, r) for n, q in {"<2>": [2], "<3>": [3], "<23>": [2, 3], "<223>": [2, 2, 3], "<233>": [2, 3, 3], "<2223>": [2, 2, 2, 3]}.items()}
    for edge, (A, B, C) in (("<23>|<2>", ("<2>", "<23>", "<223>")), ("<3>|<23>", ("<3>", "<23>", "<233>"))):
        W6 = {n: float(k1[edge]) * P[n][1] + float(k) ** 2 * P[n][2] / 2 + float(k) * P[n][3] + P[n][4] for n in (A, B, C)}
        x = {n: float(P[n][0]) for n in (A, B, C)}
        chord = W6[A] + (W6[B] - W6[A]) * (x[C] - x[A]) / (x[B] - x[A])
        print(f"r={r} edge {edge}: W6 {A} {W6[A]:+.6e}, {B} {W6[B]:+.6e}, {C} {W6[C]:+.6e};  {C} minus chord = {W6[C]-chord:+.3e}"
              f"  -> {C + ' OPENS at O(g^6)' if W6[C] < chord else C + ' does not open'}")
        if edge == "<23>|<2>":
            W6x = float(k1[edge]) * P["<2223>"][1] + float(k) ** 2 * P["<2223>"][2] / 2 + float(k) * P["<2223>"][3] + P["<2223>"][4]
            # <2223> should lie on the <2>-<223> segment (triple-range argument)
            xa, xc, xx = float(P["<2>"][0]), float(P["<223>"][0]), float(P["<2223>"][0])
            print(f"      <2223> minus <2>-<223> chord = {W6x - (W6['<2>'] + (W6['<223>'] - W6['<2>']) * (xx - xa) / (xc - xa)):+.2e} (expect 0)")

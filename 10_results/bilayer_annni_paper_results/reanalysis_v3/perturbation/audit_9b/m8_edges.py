import numpy as np
from indep_e6 import per_spin
# M8: Lx dependence of per-spin e6
for seq, r in (((2, 3), 0), ((3,), 1)):
    for Lx, mLy in ((4, 10), (5, 10), (4, 20), (3, 10)):
        try:
            e, N = per_spin(list(seq), 0.5, r, Lx=Lx, minLy=mLy)
            print(f"M8 {seq} r={r} Lx={Lx} Ly>={mLy}: e6 = {e[6]:+.15e}")
        except AssertionError as ex:
            print(f"M8 {seq} r={r} Lx={Lx}: zero-cost state within 3 flips ({ex})")
# <23> window edges at exact kappa: O(g^4) vs O(g^6), vs QMC (probe23_edges.json)
def coeffs(seq, k, r):
    e, N = per_spin(list(seq), k, r)
    rho = len(seq) / sum(seq)
    e0 = -2 + k - r / 2 + (2 - 4 * k) * rho
    return np.array([e0, e[2], e[4], e[6]])
def cross(a, b, order):
    d = (a - b)[: order // 2 + 1]
    u = [x.real for x in np.roots(d[::-1]) if abs(x.imag) < 1e-12 and x.real > 0]
    return np.sqrt(min(u)) if u else np.nan
qmc = {(0.53, 0): (0.7073, 0.7491), (0.53, 1): (1.1089, 1.1511), (0.5625, 0): (0.9360, 1.1531)}
for (k, r), (q2, q3) in qmc.items():
    c2, c3, c23 = coeffs((2,), k, r), coeffs((3,), k, r), coeffs((2, 3), k, r)
    for o in (4, 6):
        lo, hi = cross(c23, c2, o), cross(c23, c3, o)
        print(f"kappa={k} r={r} O(g^{o}): <2>|<23> {lo:.4f}  <23>|<3> {hi:.4f}  width {hi-lo:.4f}   | QMC {q2:.4f} {q3:.4f} width {q3-q2:.4f}")
    # size of g^6 vs g^4 in the <23> gain relative to the <2>-<3> chord at the QMC window centre
    g = (q2 + q3) / 2; w = np.array([1, g**2, g**4, g**6])
    rho2, rho3, rho23 = 0.5, 1/3, 0.4; t = (rho23 - rho3) / (rho2 - rho3)
    dep = c23 - (t * c2 + (1 - t) * c3)
    print(f"      chord depth of <23> at g={g:.3f}: g^4 term {dep[2]*g**4:+.2e}, g^6 term {dep[3]*g**6:+.2e} (ratio {dep[3]*g**2/dep[2]:.2f})")

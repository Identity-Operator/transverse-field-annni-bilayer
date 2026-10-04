"""Does non-degenerate RS to O(g^4) give the ED ground energy on a 3 x 6 single layer (r = 0) for <3>?
<3> is classically degenerate with (2,4) at any kappa and one row (3 flips) away -> expect an O(g^4) deviation."""
import numpy as np, scipy.sparse as sp, sys
from scipy.sparse.linalg import eigsh
from fractions import Fraction as Fr
sys.path.insert(0, "/tmp/claude-1000/-home-hoang-nguyen-Phase-Classification/8f62f05f-df21-4fe1-94fa-831f10b083be/scratchpad")
import indep_rs as I
Lx, Ly, kappa = 3, 6, Fr(3, 5)
N, bonds, yc = I.lattice(Lx, Ly, 0, kappa)
s = I.spins_for([3], Ly, yc)
E0, E2, E4 = (float(v) for v in I.rs(N, bonds, s))
s24 = I.spins_for([2, 4], Ly, yc); E0b, E2b, E4b = (float(v) for v in I.rs(N, bonds, s24))
print(f"<3>: E0={E0} E2={E2:.8f} E4={E4:.8f};  (2,4): E0={E0b} E2={E2b:.8f}")
dim = 1 << N; idx = np.arange(dim)
spins = 1 - 2 * ((idx[:, None] >> np.arange(N)) & 1).astype(np.int8)
diag = np.zeros(dim)
for i, j, J in bonds:
    diag -= float(J) * spins[:, i] * spins[:, j]
emin = diag.min(); gs_set = np.nonzero(np.abs(diag - emin) < 1e-9)[0]
print("classical GS energy", emin, "degeneracy", len(gs_set))
del spins
rows = np.concatenate([idx] * N); cols = np.concatenate([idx ^ (1 << k) for k in range(N)])
X = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(dim, dim)); D = sp.diags(diag)
gs = np.array([0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12])
res = []
for g in gs:
    e = eigsh(D - g * X, k=1, which="SA", tol=1e-13, v0=None)[0][0]
    res.append(e)
    print(f"g={g:.3f}  (E_ED - E0 - g^2 E2)/g^4 = {(e - E0 - g**2 * E2) / g**4:+.6f}   (RS E4 = {E4:+.6f})", flush=True)
res = np.array(res); y = (res - E0 - gs ** 2 * E2) / gs ** 4
c = np.polyfit(gs ** 2, y, 2)
print(f"extrapolated g->0: {c[-1]:+.6f}  vs non-degenerate RS E4 {E4:+.6f}  -> difference {c[-1]-E4:+.6f} (total, N={N})")

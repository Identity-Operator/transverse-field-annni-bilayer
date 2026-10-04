"""Independent O(g^6): full-lattice RS recursion in the space of states within 3 flips of s
(exact for E_2..E_6: psi_5 is needed only at 1 flip, psi_4 at <=2, psi_3 at <=3). float64, no clusters."""
import numpy as np, itertools, sys, time
import scipy.sparse as sp

def lattice(Lx, Ly, r, kappa):
    L = 2 if r != 0 else 1
    idx = lambda l, x, y: (l * Lx + x % Lx) * Ly + y % Ly
    b = []
    for l in range(L):
        for x in range(Lx):
            for y in range(Ly):
                b += [(idx(l, x, y), idx(l, x + 1, y), 1.0), (idx(l, x, y), idx(l, x, y + 1), 1.0),
                      (idx(l, x, y), idx(l, x, y + 2), -kappa)]
    if L == 2:
        b += [(idx(0, x, y), idx(1, x, y), float(r)) for x in range(Lx) for y in range(Ly)]
    N = L * Lx * Ly
    return N, b, [i % Ly for i in range(N)]

def column(seq):
    col, sg = [], 1
    for n in seq:
        col += [sg] * n; sg = -sg
    return col + [-v for v in col] if sg == -1 else col

def rs6(N, bonds, s, order=6):
    s = np.array(s, float)
    bi = np.array([b[0] for b in bonds]); bj = np.array([b[1] for b in bonds]); J = np.array([b[2] for b in bonds])
    states = [()] + [(i,) for i in range(N)] + list(itertools.combinations(range(N), 2)) + list(itertools.combinations(range(N), 3))
    index = {st: k for k, st in enumerate(states)}
    M = len(states)
    # generic energy change: 2 * sum over bonds cut by F of J s_i s_j
    w = J * s[bi] * s[bj]
    inF = np.zeros(N, bool); dE = np.empty(M)
    for k, st in enumerate(states):
        inF[:] = False; inF[list(st)] = True
        cut = inF[bi] ^ inF[bj]
        dE[k] = 2 * w[cut].sum()
    assert dE[1:].min() > 1e-12, dE[1:].min()
    rows, cols = [], []
    for k, st in enumerate(states):
        S = set(st)
        for i in range(N):
            t = tuple(sorted(S ^ {i}))
            if len(t) <= 3:
                rows.append(k); cols.append(index[t])
    X = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(M, M))   # sum_i sigma^x in the truncated space
    V = -X
    R = np.zeros(M); R[1:] = -1 / dE[1:]
    psi = [np.zeros(M) for _ in range(order + 1)]; psi[0][0] = 1
    E = [0.0] * (order + 1)
    for n in range(1, order + 1):
        v = V @ psi[n - 1]
        E[n] = v[0]
        acc = v - sum(E[k] * psi[n - k] for k in range(1, n + 1))
        psi[n] = R * acc
    return E

def per_spin(seq, kappa, r, Lx=4, minLy=10):
    col = [1] if seq is None else column(seq)
    P = len(col); Ly = P * max(1, -(-minLy // P))
    N, b, yc = lattice(Lx, Ly, r, kappa)
    s = [col[y % P] for y in yc]
    E = rs6(N, b, s)
    return [e / N for e in E], N

if __name__ == "__main__":
    # sanity: isolated-like check -- single spin in a field: use N=1 'lattice' is not possible; check vs ED on a random cluster instead
    rng = np.random.default_rng(5)
    N = 9
    bonds = [(i, j, float(rng.uniform(-1.5, 1.5))) for i in range(N) for j in range(i + 1, N) if rng.random() < 0.35]
    conf = 1 - 2 * ((np.arange(1 << N)[:, None] >> np.arange(N)) & 1)
    Ec = np.array([-sum(Jv * c[i] * c[j] for i, j, Jv in bonds) for c in conf]); o = np.argsort(Ec)
    s0 = conf[o[0]]
    E = rs6(N, bonds, s0)
    diag = Ec; dim = 1 << N; idx = np.arange(dim)
    Xf = sp.csr_matrix((np.ones(dim * N), (np.concatenate([idx] * N), np.concatenate([idx ^ (1 << k) for k in range(N)]))), shape=(dim, dim)).toarray()
    gs = np.linspace(0.02, 0.16, 12)
    Eg = np.array([np.linalg.eigvalsh(np.diag(diag) - g * Xf)[0] for g in gs])
    y = Eg - Ec[o[0]] - E[2] * gs ** 2 - E[4] * gs ** 4
    c = np.linalg.lstsq(np.vstack([gs ** k for k in (6, 8, 10, 12)]).T, y, rcond=None)[0]
    print(f"[ED check, random N=9, gap {Ec[o[2]]-Ec[o[0]]:.2f}] E6 RS {E[6]:+.8f}  ED fit {c[0]:+.8f}  (E1,E3,E5 = {E[1]:.1e},{E[3]:.1e},{E[5]:.1e})")

import numpy as np, scipy.sparse as sp
from indep_e6 import rs6
rng = np.random.default_rng(21); done = 0
while done < 3:
    N = 9
    bonds = [(i, j, float(rng.uniform(-1.5, 1.5))) for i in range(N) for j in range(i + 1, N) if rng.random() < 0.4]
    conf = 1 - 2 * ((np.arange(1 << N)[:, None] >> np.arange(N)) & 1)
    Ec = np.array([-sum(Jv * c[i] * c[j] for i, j, Jv in bonds) for c in conf]); o = np.argsort(Ec)
    if Ec[o[2]] - Ec[o[0]] < 1.5: continue
    try: E = rs6(N, bonds, conf[o[0]])
    except AssertionError: continue
    dim = 1 << N; idx = np.arange(dim)
    Xf = sp.csr_matrix((np.ones(dim * N), (np.concatenate([idx] * N), np.concatenate([idx ^ (1 << k) for k in range(N)]))), shape=(dim, dim)).toarray()
    gs = np.linspace(0.02, 0.2, 14)
    Eg = np.array([np.linalg.eigvalsh(np.diag(Ec) - g * Xf)[0] for g in gs])
    y = Eg - Ec[o[0]] - E[2] * gs ** 2 - E[4] * gs ** 4
    c = np.linalg.lstsq(np.vstack([gs ** k for k in (6, 8, 10, 12)]).T, y, rcond=None)[0]
    print(f"[ED random N=9 #{done}, gap {Ec[o[2]]-Ec[o[0]]:.2f}] E6 RS {E[6]:+.9f}  ED fit {c[0]:+.9f}  rel {c[0]/E[6]-1:+.1e}")
    done += 1

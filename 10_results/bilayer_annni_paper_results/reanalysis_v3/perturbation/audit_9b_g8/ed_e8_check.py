"""ED check of the O(g^8) RS coefficient on random 10-spin graphs (full-space RS, exact rationals), fitting the
absolute residual E_ED - sum_{n<=6} E_n g^n = c8 g^8 + c10 g^10 + ... over g in [0.05, 0.3]."""
import json, sys
from fractions import Fraction as Fr
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import lc8
rng = np.random.default_rng(99); out = []; done = 0
while done < 4:
    N = 10; J = {}
    for i in range(N):
        for j in range(i + 1, N):
            if rng.random() < 0.3:
                v = Fr(int(rng.integers(-15, 16)), 10)
                if v != 0: J[(i, j)] = v
    conf = 1 - 2 * ((np.arange(1 << N)[:, None] >> np.arange(N)) & 1)
    Ec = np.array([-sum(float(w) * c[i] * c[j] for (i, j), w in J.items()) for c in conf]); o = np.argsort(Ec)
    s0 = [int(v) for v in conf[o[0]]]
    D = [(Fr(2 * s0[i]) * sum((w * s0[b if i == a else a] for (a, b), w in J.items() if i in (a, b)), Fr(0)),) for i in range(N)]
    if min(d[0] for d in D) < 2 or Ec[o[2]] - Ec[o[0]] < 2:     # well-gapped reference only
        continue
    try:
        E = lc8.rs_cluster(D, {(a, b): (w * s0[a] * s0[b],) for (a, b), w in J.items()}, 8)
    except AssertionError:
        continue
    E = [float(e[0]) for e in E]
    dim = 1 << N; idx = np.arange(dim); X = np.zeros((dim, dim))
    for k in range(N): X[idx, idx ^ (1 << k)] = 1
    gs = np.linspace(0.05, 0.3, 26)
    Eg = np.array([np.linalg.eigvalsh(np.diag(Ec - Ec[o[0]]) - g * X)[0] for g in gs])
    res = Eg - (E[2] * gs**2 + E[4] * gs**4 + E[6] * gs**6)
    A = np.vstack([gs**k for k in (8, 10, 12, 14, 16)]).T
    c, *_ = np.linalg.lstsq(A, res, rcond=None)
    out.append({"min_D": float(min(d[0] for d in D)), "E8_RS": E[8], "E8_ED": float(c[0]), "rel": float(c[0] / E[8] - 1)})
    print(f"[ED #{done}] min D {out[-1]['min_D']:.1f}: E8 RS {E[8]:+.8e}  ED fit {c[0]:+.8e}  rel {c[0]/E[8]-1:+.1e}", flush=True)
    done += 1
json.dump(out, open(Path(__file__).parent / "ed_e8_check.json", "w"), indent=1)

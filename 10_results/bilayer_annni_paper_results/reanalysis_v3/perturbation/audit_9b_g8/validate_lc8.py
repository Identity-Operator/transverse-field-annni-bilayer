"""Checks of lc8.py: (1) 1D TFIM chain through g^8 (exact: -1, -1/4, -1/64, -1/256, -25/16384 per spin);
(2) random-coupling 10-spin graphs: cluster sum (<= 4 sites) vs RS on the full 2^10 space (exact) vs ED;
(3) eps2..eps6 vs the audited O(g^6) values (pt_table.json of audit_9b)."""
import itertools, json, sys, time
from fractions import Fraction as Fr
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import lc8

HERE = Path(__file__).parent
rep = {}

# (1) chain
t0 = time.time()
st = lc8.Structure(None, 0, geometry="chain")
s, ncl = lc8.series(st)
chain = {n: str(s[n][0]) for n in (0, 2, 4, 6, 8)}
exp = {0: "-1", 2: "-1/4", 4: "-1/64", 6: "-1/256", 8: "-25/16384"}
rep["chain_TFIM"] = {"got": chain, "expected": exp, "ok": chain == exp}
print("[chain]", chain, "OK" if chain == exp else "MISMATCH", f"({time.time()-t0:.1f}s)", flush=True)

# (2) random graphs
def connected_subsets(adj, N, maxsize):
    out = set()
    for a in range(N):
        front = [frozenset([a])]
        out.add(frozenset([a]))
        for _ in range(maxsize - 1):
            new = []
            for C in front:
                for s_ in C:
                    for nb in adj[s_]:
                        if nb > a and nb not in C:
                            C2 = C | {nb}
                            if C2 not in out:
                                out.add(C2); new.append(C2)
            front = new
    return out

rng = np.random.default_rng(2026)
rep["random_graphs"] = []
done = 0
while done < 3:
    N = 10
    J = {}
    for i in range(N):
        for j in range(i + 1, N):
            if rng.random() < 0.3:
                v = Fr(int(rng.integers(-15, 16)), 10)
                if v != 0:
                    J[(i, j)] = v
    conf = 1 - 2 * ((np.arange(1 << N)[:, None] >> np.arange(N)) & 1)
    Ec = np.array([-sum(float(w) * c[i] * c[j] for (i, j), w in J.items()) for c in conf])
    o = np.argsort(Ec)
    if Ec[o[2]] - Ec[o[0]] < 1.0:
        continue
    s0 = [int(v) for v in conf[o[0]]]
    adj = {i: [] for i in range(N)}
    for (i, j) in J:
        adj[i].append(j); adj[j].append(i)
    Dfull = [(Fr(2 * s0[i]) * sum((w * s0[j if i == a else a] for (a, j), w in J.items() if i in (a, j)), Fr(0)),)
             for i in range(N)]
    # cluster sum
    tot = {n: Fr(0) for n in (2, 4, 6, 8)}
    for C in connected_subsets(adj, N, 4):
        sites = sorted(C); idx = {q: k for k, q in enumerate(sites)}
        D = [Dfull[q] for q in sites]
        B = {(idx[a], idx[b]): (w * s0[a] * s0[b],) for (a, b), w in J.items() if a in idx and b in idx}
        w_ = lc8.w_of(D, B)
        for n in tot:
            tot[n] += w_[n][0]
    # full-space RS (all N sites, exact)
    Bfull = {(a, b): (w * s0[a] * s0[b],) for (a, b), w in J.items()}
    t1 = time.time()
    Efull = lc8.rs_cluster(Dfull, Bfull, 8)
    full = {n: Efull[n][0] for n in (2, 4, 6, 8)}
    # ED fit of the g^8 coefficient
    dim = 1 << N; idx_ = np.arange(dim)
    X = np.zeros((dim, dim))
    for k in range(N):
        X[idx_, idx_ ^ (1 << k)] = 1
    gs = np.linspace(0.02, 0.2, 16)
    Eg = np.array([np.linalg.eigvalsh(np.diag(Ec) - g * X)[0] for g in gs])
    y = (Eg - Ec[o[0]] - float(full[2]) * gs**2 - float(full[4]) * gs**4 - float(full[6]) * gs**6) / gs**8
    cfit = np.polyfit(gs**2, y, 4)
    row = {"gap": float(Ec[o[2]] - Ec[o[0]]), "cluster_eq_full": all(tot[n] == full[n] for n in tot),
           "E8_exact": float(full[8]), "E8_ED_fit": float(cfit[-1]), "rel": float(cfit[-1] / float(full[8]) - 1)}
    rep["random_graphs"].append(row)
    print(f"[random #{done}] gap {row['gap']:.2f}: cluster sum == full-space RS through g^8: {row['cluster_eq_full']};"
          f"  E8 {row['E8_exact']:+.8e} vs ED {row['E8_ED_fit']:+.8e} (rel {row['rel']:+.1e})", flush=True)
    done += 1

# (3) eps2..eps6 vs audited values
prev = json.load(open(HERE.parent / "audit_9b" / "pt_table.json"))
names = {"FM": None, "<2>": (2,), "<23>": (2, 3), "<3>": (3,)}
rep["vs_audited_O6"] = []
for r in (0, 1):
    for nm, seq in names.items():
        t0 = time.time()
        s, ncl = lc8.series(lc8.Structure(seq, r))
        ref = prev[f"r={r}"][nm]
        ok = (str(s[2][0]) == ref["eps2"] and str(s[4][0]) == ref["eps4"] and abs(float(s[6][0]) - ref["eps6"]) < 1e-12)
        rep["vs_audited_O6"].append({"r": r, "structure": nm, "ok": ok, "eps6": float(s[6][0]), "eps6_ref": ref["eps6"],
                                     "eps8": float(s[8][0]), "clusters": ncl})
        print(f"[vs O6] r={r} {nm:5s}: eps2 {s[2][0]} eps4 {s[4][0]} eps6 {float(s[6][0]):+.10e} (ref {ref['eps6']:+.10e}) "
              f"{'OK' if ok else 'MISMATCH'}  eps8 {float(s[8][0]):+.6e}  [{ncl} clusters, {time.time()-t0:.0f}s]", flush=True)
(HERE / "validate_lc8.json").write_text(json.dumps(rep, indent=1, default=str))
print("wrote", HERE / "validate_lc8.json")

#!/usr/bin/env python3
"""Rayleigh-Schroedinger perturbation theory in the transverse field to O(g^4) for straight-wall
structures of the bilayer ANNNI model (Paper 1, plan item 6).

H = H0 + V with H0 = -sum_b J_b s_i s_j (classical; J_b as in sse.class_couplings) and
V = -Gamma sum_i sigma^x_i. Take a classical configuration s whose single flips cost D_i > 0 and
whose double flips cost D_ij > 0 (Lx >= 3 makes every RS denominator nonzero). The coefficients
below are the Lx -> infinity (per-spin) values. On a finite periodic lattice a degenerate wall
configuration one row-move away couples at O(g^Lx) (r = 0) or O(g^{2Lx}) (r > 0), so the finite-
lattice energy deviates from the series at O(g^{2Lx-2}) or earlier (audit, phase-classification-9b,
2026-09-29). With J0 = 1 and g = Gamma,

    E(g) = E0 + g^2 E2 + g^4 E4 + O(g^6),
    E2 = -sum_i 1/D_i,
    E4 = sum_i 1/D_i^3 - sum_{i<j} 4 B_ij (D_i + D_j) / [(D_i D_j)^2 (D_i + D_j - 4 B_ij)],

where D_i = 2 s_i sum_j J_ij s_j, B_ij = J_ij s_i s_j, and J_ij sums all bonds between i and j.
Only coupled pairs enter: for uncoupled pairs the fourth-order paths cancel against the
renormalisation term -E2 sum_i 1/D_i^2. For an isolated spin E4 = 1/D^3 = 1/(8 h^3), the g^4
term of -sqrt(h^2 + g^2).

For an x-uniform structure with favourable stacking the energy per spin follows from one period
of the column along y (per_column()), in exact rational arithmetic when kappa and r are
Fractions. Per spin there is one x pair, one y pair, one y2 pair and half a rung.

Near the multiphase point write kappa = 1/2 + k g^2 + k1 g^4. At O(g^2) all sequences of two-
and three-site domains are degenerate at the <3>|<2> boundary k = F/4. At O(g^4) the energy per
spin of a sequence is k1 (1 - 4 rho) + W, W = k e2'(1/2) + e4(1/2), rho the wall density, so the
stable sequences are the lower convex hull of W against rho (Fisher-Selke construction).

Modes (from vu_work/):
  --verify   (1) closed form vs explicit enumeration of all fourth-order RS paths on production
             lattices; (2) closed form vs exact diagonalisation (random-coupling clusters; FM and
             <2> on a 3x4 layer); (3) per-column evaluator vs the lattice closed form.
  --analyze  exact coefficients at kappa = 1/2; O(g^2) and O(g^4) crossings for every L = 12
             branch case with a *_free_energy.json (compared with the QMC crossing); residuals
             of the template-branch energies against the O(g^2) and O(g^4) series; boundary
             coefficients k1 and the convex hull of 2/3 sequences at the <3>|<2> boundary.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import itertools
import json
import math
from collections import defaultdict
from fractions import Fraction as Fr

import numpy as np

from phase_pipeline.qmc.sse import bilayer_lattice, class_couplings

BR = Path("10_results/bilayer_annni_paper_results/reanalysis_v3/sse_validation/branches")
OUT = Path("10_results/bilayer_annni_paper_results/reanalysis_v3/perturbation")


# ---------------------------------------------------------------- lattice closed form and checks
def merged_couplings(lat, Jc):
    J = defaultdict(float)
    for i, j, c in zip(lat["bi"], lat["bj"], lat["bcls"]):
        a, b = (int(i), int(j)) if i < j else (int(j), int(i))
        J[(a, b)] += float(Jc[c])
    return dict(J)


def series_lattice(N, J, s):
    """(E0, E2, E4) totals from the closed form."""
    field = np.zeros(N)
    E0 = 0.0
    for (i, j), w in J.items():
        field[i] += w * s[j]
        field[j] += w * s[i]
        E0 -= w * s[i] * s[j]
    D = 2 * s * field
    if D.min() <= 0:
        raise ValueError("reference state has a non-positive single-flip cost")
    E2 = -np.sum(1 / D)
    E4 = np.sum(1 / D ** 3)
    for (i, j), w in J.items():
        B = w * s[i] * s[j]
        S = D[i] + D[j]
        if S - 4 * B <= 0:
            raise ValueError("reference state has a non-positive double-flip cost")
        E4 -= 4 * B * S / ((D[i] * D[j]) ** 2 * (S - 4 * B))
    return E0, E2, E4


def series_bruteforce(N, J, s):
    """(E0, E2, E4) by explicit enumeration of RS paths 0 -> i -> {i,j} -> c -> 0, classical
    energies evaluated from the bond list (independent of the closed form)."""
    I = np.array([k[0] for k in J]); K = np.array([k[1] for k in J]); W = np.array(list(J.values()))

    def energy(t):
        return -np.sum(W * t[I] * t[K])
    E0 = energy(s)
    D1 = np.empty(N)
    for i in range(N):
        t = s.copy(); t[i] = -t[i]; D1[i] = energy(t) - E0
    D2 = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            t = s.copy(); t[i] = -t[i]; t[j] = -t[j]; D2[i, j] = D2[j, i] = energy(t) - E0
    E2 = -np.sum(1 / D1)
    t1 = 0.0
    for i in range(N):
        for j in range(N):
            if i != j:
                t1 += (1 / (D1[i] * D2[i, j])) * (1 / D1[i] + 1 / D1[j])
    E4 = -t1 + np.sum(1 / D1) * np.sum(1 / D1 ** 2)
    return E0, E2, E4


def ed_ground(N, J, gs):
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh
    dim = 1 << N
    idx = np.arange(dim)
    bits = ((idx[:, None] >> np.arange(N)) & 1).astype(np.int8)
    spins = 1 - 2 * bits
    diag = np.zeros(dim)
    for (i, j), w in J.items():
        diag -= w * spins[:, i] * spins[:, j]
    rows = np.concatenate([idx for _ in range(N)])
    cols = np.concatenate([idx ^ (1 << k) for k in range(N)])
    X = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(dim, dim))
    D = sp.diags(diag)
    out = []
    for g in gs:
        H = D - g * X
        if dim <= 1024:
            out.append(float(np.linalg.eigvalsh(H.toarray())[0]))
        else:
            out.append(float(eigsh(H, k=1, which="SA", tol=1e-14)[0][0]))
    return np.array(out)


def fit_series(gs, E, E0, n_terms=5):
    """Fit E - E0 = sum_k a_k g^(2k), k = 1..n_terms; returns (a_1, a_2)."""
    X = np.vstack([gs ** (2 * k) for k in range(1, n_terms + 1)]).T
    c, *_ = np.linalg.lstsq(X, E - E0, rcond=None)
    return c[0], c[1]


def structure_spins(lat, lengths):
    """x-uniform, parallel-stacked spins with the given domain sequence along y (Ly must be a
    multiple of the spin period)."""
    col = column(lengths)
    Ly = lat["Ly"]
    if Ly % len(col):
        raise ValueError("Ly is not a multiple of the period")
    y = lat["ycoord"]
    return np.array([col[yy % len(col)] for yy in y], dtype=np.int64)


# ---------------------------------------------------------------- per-column evaluator
def column(lengths):
    s, sign = [], 1
    for n in lengths:
        s += [sign] * n
        sign = -sign
    if sign == -1:                      # odd number of domains: the spin period is doubled
        s = s + [-v for v in s]
    return s


def _pair(Di, Dj, B):
    S = Di + Dj
    return -4 * B * S / ((Di * Dj) ** 2 * (S - 4 * B))


def per_column(lengths, kappa, r):
    """(e0, e2, e4, e2') per spin; exact if kappa and r are Fractions. e2' = d e2 / d kappa."""
    s = column(lengths)
    P = len(s)
    one = kappa - kappa + 1              # 1 in the number type of kappa
    h = [2 + r + s[y] * (s[(y + 1) % P] + s[(y - 1) % P]) - kappa * s[y] * (s[(y + 2) % P] + s[(y - 2) % P])
         for y in range(P)]
    c = [s[y] * (s[(y + 2) % P] + s[(y - 2) % P]) for y in range(P)]
    D = [2 * v for v in h]
    if min(D) <= 0:
        raise ValueError(f"non-positive flip cost in {lengths}")
    e0 = sum(-1 - s[y] * s[(y + 1) % P] + kappa * s[y] * s[(y + 2) % P] for y in range(P)) * one / P - r * one / 2
    e2 = -sum(one / d for d in D) / P
    e2p = -sum(c[y] * one / (2 * h[y] ** 2) for y in range(P)) / P
    e4 = 0 * one
    for y in range(P):
        Dy = D[y]
        e4 += one / Dy ** 3
        e4 += _pair(Dy, Dy, 1 * one)                                     # x pair
        e4 += _pair(Dy, D[(y + 1) % P], s[y] * s[(y + 1) % P] * one)      # y pair
        e4 += _pair(Dy, D[(y + 2) % P], -kappa * s[y] * s[(y + 2) % P])   # y2 pair
        if r != 0:
            e4 += _pair(Dy, Dy, r * one) / 2                             # half a rung per spin
    return e0, e2, e4 / P, e2p


# ---------------------------------------------------------------- verification
def verify():
    rep = {"closed_vs_enumeration": [], "closed_vs_ED": [], "column_vs_lattice": []}
    cases = [("FM", [1000], 0.25, 0), ("FM", [1000], 0.5, 1), ("<2>", [2, 2], 0.75, 0), ("<2>", [2, 2], 0.5, 1),
             ("<3>", [3, 3], 0.5, 0), ("<3>", [3, 3], 0.6, 1), ("<23>", [2, 3, 2, 3], 0.55, 1),
             ("2433", [2, 4, 3, 3], 0.5, 0)]
    for name, seq, kappa, r in cases:
        per = 1 if seq == [1000] else len(column(seq))
        Ly = per if per >= 5 else 2 * per
        if seq == [1000]:
            Ly = 6
        lat = bilayer_lattice(3, Ly, 2)
        J = merged_couplings(lat, class_couplings(1, 1, kappa, r))
        s = np.ones(lat["N"], dtype=np.int64) if seq == [1000] else structure_spins(lat, seq)
        a = series_lattice(lat["N"], J, s)
        b = series_bruteforce(lat["N"], J, s)
        N = lat["N"]
        colv = coeffs("FM", kappa, r) if seq == [1000] else per_column(seq, kappa, r)
        row = {"structure": name, "kappa": kappa, "r": r, "lattice": f"3x{Ly}x2",
               "closed": [x / N for x in a], "enumerated": [x / N for x in b],
               "max_abs_diff": float(max(abs(x - y) for x, y in zip(a, b)) / N)}
        rep["closed_vs_enumeration"].append(row)
        rep["column_vs_lattice"].append({"structure": name, "kappa": kappa, "r": r,
                                         "column": [float(x) for x in colv[:3]], "lattice": [x / N for x in a],
                                         "max_abs_diff": float(max(abs(float(x) - y / N) for x, y in zip(colv[:3], a)))})
        print(f"[enum] {name:5s} k={kappa} r={r}: e2 {a[1]/N:+.10f} vs {b[1]/N:+.10f}   e4 {a[2]/N:+.10f} vs {b[2]/N:+.10f}"
              f"   column e4 {float(colv[2]):+.10f}", flush=True)
    # ED on random-coupling clusters and on a 3x4 layer (FM at kappa = 0.25, <2> at 0.75)
    rng = np.random.default_rng(7)
    gs = np.linspace(0.01, 0.12, 12)
    n_done = 0
    while n_done < 3:
        N = 10
        J = {}
        for i in range(N):
            for j in range(i + 1, N):
                if rng.random() < 0.35:
                    J[(i, j)] = float(rng.uniform(-1.5, 1.5))
        conf = 1 - 2 * ((np.arange(1 << N)[:, None] >> np.arange(N)) & 1)
        E = np.array([-sum(w * c[i] * c[j] for (i, j), w in J.items()) for c in conf])
        o = np.argsort(E)
        if E[o[2]] - E[o[0]] < 0.5:           # need a clean doublet (s, -s) and a gap
            continue
        s = conf[o[0]].astype(np.int64)
        try:
            E0, E2, E4 = series_lattice(N, J, s)
        except ValueError:
            continue
        Eg = ed_ground(N, J, gs)
        a2, a4 = fit_series(gs, Eg, E0)
        rep["closed_vs_ED"].append({"system": f"random N=10 #{n_done}", "E2": E2, "E2_ED": a2, "E4": E4, "E4_ED": a4})
        print(f"[ED] random #{n_done}: E2 {E2:+.8f} vs {a2:+.8f}   E4 {E4:+.8f} vs {a4:+.8f}", flush=True)
        n_done += 1
    for name, kappa, seq in [("FM", 0.25, None), ("<2>", 0.75, [2, 2])]:
        lat = bilayer_lattice(3, 4, 1)
        J = merged_couplings(lat, class_couplings(1, 1, kappa, 0))
        s = np.ones(lat["N"], dtype=np.int64) if seq is None else structure_spins(lat, seq)
        E0, E2, E4 = series_lattice(lat["N"], J, s)
        Eg = ed_ground(lat["N"], J, np.linspace(0.03, 0.36, 12))
        a2, a4 = fit_series(np.linspace(0.03, 0.36, 12), Eg, E0)
        rep["closed_vs_ED"].append({"system": f"3x4 layer {name} kappa={kappa}", "E2": E2, "E2_ED": a2, "E4": E4, "E4_ED": a4})
        print(f"[ED] 3x4 {name} k={kappa}: E2 {E2:+.8f} vs {a2:+.8f}   E4 {E4:+.8f} vs {a4:+.8f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pt_order4_verify.json").write_text(json.dumps(rep, indent=1, default=float))
    print("wrote", OUT / "pt_order4_verify.json")


# ---------------------------------------------------------------- analysis
SEQ = {"FM": [10 ** 6], "<2>": [2, 2], "<3>": [3, 3]}


def coeffs(name, kappa, r):
    if name == "FM":        # uniform: exact per-spin values without a long column
        one = kappa - kappa + 1
        h = 4 + r - 2 * kappa
        D = 2 * h
        e4 = one / D ** 3 + _pair(D, D, one) + _pair(D, D, one) + _pair(D, D, -kappa * one)
        if r != 0:
            e4 += _pair(D, D, r * one) / 2
        return -2 + kappa - r * one / 2, -one / D, e4, -one / h ** 2
    return per_column(SEQ[name], kappa, r)


def crossing(A, B, kappa, r, order):
    a, b = coeffs(A, kappa, r), coeffs(B, kappa, r)
    d0, d2, d4 = (float(x) - float(y) for x, y in zip(a[:3], b[:3]))
    if order == 2:
        u = -d0 / d2
        return math.sqrt(u) if u > 0 else float("nan")
    disc = d2 * d2 - 4 * d4 * d0
    if disc < 0:
        return float("nan")
    roots = [(-d2 + sg * math.sqrt(disc)) / (2 * d4) for sg in (1, -1)] if d4 != 0 else [-d0 / d2]
    roots = [u for u in roots if u > 0]
    return math.sqrt(min(roots)) if roots else float("nan")


def cyclic_sequences(alphabet, max_len):
    seen = set()
    for n in range(1, max_len + 1):
        for seq in itertools.product(alphabet, repeat=n):
            rots = [seq[i:] + seq[:i] for i in range(n)]
            can = min(rots)
            # skip sequences that are repeats of a shorter one
            prim = next(seq_ for k in range(1, n + 1) if n % k == 0
                        for seq_ in [can[:k]] if seq_ * (n // k) == can)
            if prim not in seen:
                seen.add(prim)
    return sorted(seen, key=lambda t: (len(t), t))


def lower_hull(points):
    """points: list of (rho, W, label) with Fractions; returns the lower convex hull, by rho."""
    pts = sorted(points, key=lambda p: (p[0], p[1]))
    best = {}
    for p in pts:                         # keep the lowest W per rho
        if p[0] not in best or p[1] < best[p[0]][1]:
            best[p[0]] = p
    pts = sorted(best.values(), key=lambda p: p[0])
    hull = []
    for p in pts:
        while len(hull) >= 2:
            (x1, y1, _), (x2, y2, _) = hull[-2], hull[-1]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) <= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    return hull


def analyze():
    out = {"exact_at_kappa_half": {}, "boundaries": {}, "crossings": [], "residuals": [], "hull_3_2": {}}
    half = Fr(1, 2)
    for r in (0, 1):
        R = Fr(r)
        F3 = Fr(6) / ((2 + R) * (3 + R) * (5 + R))
        ex = {}
        for name in ("FM", "<2>", "<3>"):
            e0, e2, e4, e2p = coeffs(name, half, R)
            ex[name] = {"e0": str(e0), "e2": str(e2), "e4": str(e4), "de2_dkappa": str(e2p),
                        "e4_float": float(e4)}
        out["exact_at_kappa_half"][f"r={r}"] = ex
        # boundary coefficients: kappa = 1/2 + k g^2 + k1 g^4
        W = {}
        for side, k in (("32", F3 / 4), ("FM3", -F3 / 8)):
            for name in ("FM", "<2>", "<3>"):
                e0, e2, e4, e2p = coeffs(name, half, R)
                W[(side, name)] = k * e2p + e4
        rho = {"FM": Fr(0), "<2>": Fr(1, 2), "<3>": Fr(1, 3)}
        for side, k, A, B in (("32", F3 / 4, "<2>", "<3>"), ("FM3", -F3 / 8, "FM", "<3>")):
            gA = k * (1 - 4 * rho[A]) + coeffs(A, half, R)[1]
            gB = k * (1 - 4 * rho[B]) + coeffs(B, half, R)[1]
            assert gA == gB, (side, gA, gB)     # O(g^2) degeneracy on each boundary
        # k1 (1 - 4 rho_A) + W_A = k1 (1 - 4 rho_B) + W_B
        k1_32 = (W[("32", "<2>")] - W[("32", "<3>")]) / (4 * (rho["<2>"] - rho["<3>"]))
        k1_FM3 = (W[("FM3", "FM")] - W[("FM3", "<3>")]) / (4 * (rho["FM"] - rho["<3>"]))
        out["boundaries"][f"r={r}"] = {
            "F": str(F3), "kappa_32": f"1/2 + ({F3 / 4}) g^2 + ({k1_32}) g^4",
            "kappa_FM3": f"1/2 - ({F3 / 8}) g^2 + ({k1_FM3}) g^4",
            "k_32": float(F3 / 4), "k1_32": float(k1_32), "k_FM3": float(-F3 / 8), "k1_FM3": float(k1_FM3)}
        print(f"r={r}: F={F3}  kappa_3|2 = 1/2 + {float(F3/4):.6f} g^2 + {float(k1_32):+.6f} g^4 ;"
              f"  kappa_FM|3 = 1/2 - {float(F3/8):.6f} g^2 + {float(k1_FM3):+.6f} g^4", flush=True)
        for name in ("FM", "<2>", "<3>"):
            print(f"   {name:4s} e2={ex[name]['e2']:>12s}  e4={ex[name]['e4']:>28s} = {ex[name]['e4_float']:+.7f}")
        # convex hull of all 2/3 sequences (plus 4s) at the <3>|<2> boundary
        pts = []
        for seq in cyclic_sequences((2, 3), 10) + [s for s in cyclic_sequences((2, 3, 4), 6) if 4 in s]:
            e0, e2, e4, e2p = per_column(list(seq), half, R)
            rho_s = Fr(len(seq), sum(seq))
            # O(g^2) excess over the degenerate 2/3 manifold must vanish for 2/3 sequences; 4s are
            # penalised at O(g^2) and enter only as a check
            g2 = (F3 / 4) * (1 - 4 * rho_s) + e2
            pts.append((rho_s, (F3 / 4) * e2p + e4, "".join(map(str, seq)), g2))
        g2_ref = pts[0][3]
        deg = [p for p in pts if p[3] == g2_ref]
        hull = lower_hull([(p[0], p[1], p[2]) for p in deg])
        pen = sorted({p[2]: float(p[3] - g2_ref) for p in pts if p[3] != g2_ref}.items())[:5]
        # distance of each 2/3 sequence above the <2>-<3> chord
        W2 = next(p[1] for p in deg if p[2] == "22" or p[2] == "2")
        W3 = next(p[1] for p in deg if p[2] == "3")
        above = []
        for p in deg:
            chord = W3 + (W2 - W3) * (p[0] - Fr(1, 3)) / (Fr(1, 2) - Fr(1, 3))
            above.append((p[2], float(p[1] - chord)))
        above.sort(key=lambda t: t[1])
        out["hull_3_2"][f"r={r}"] = {
            "n_sequences": len(pts), "n_degenerate_at_g2": len(deg),
            "hull": [(str(h[0]), h[2], float(h[1])) for h in hull],
            "lowest_relative_to_chord": above[:8],
            "sequences_with_4_penalised_at_g2": pen}
        print(f"   hull at <3>|<2>: {[h[2] for h in hull]}   lowest vs chord: {above[:4]}", flush=True)
    # crossings for the QMC cases
    for fj in sorted(BR.glob("branches_L12_k*_r*_free_energy.json")):
        d = json.loads(fj.read_text())
        for key, grp in d["groups"].items():
            kappa = float(key.split(",")[0].split("=")[1]); r = abs(float(key.split(",")[1].split("=")[1]))
            if kappa > 0.5:
                A, B, pair = "<3>", "<2>", "ascending:<2>par vs ascending:<3>par"
            else:
                A, B, pair = "<3>", "FM", "ascending:<3>par vs ascending:FM"
            v = grp["pairs"].get(pair, {})
            g2, g4 = crossing(A, B, kappa, r, 2), crossing(A, B, kappa, r, 4)
            row = {"kappa": kappa, "r": r, "side": f"{A}|{B}", "qmc": v.get("g_star"), "qmc_ci68": v.get("g_star_ci68"),
                   "pt_g2": g2, "pt_g4": g4}
            out["crossings"].append(row)
            print(f"k={kappa:<6} r={r:g} {A}|{B:4s} QMC {v.get('g_star', float('nan')):.4f} {v.get('g_star_ci68')}"
                  f"   O(g^2) {g2:.4f}   O(g^4) {g4:.4f}", flush=True)
    # residuals of template branches vs the series
    import pandas as pd
    for f in sorted(BR.glob("branches_L12_k*_r*.csv")):
        d = pd.read_csv(f)
        d = d[(d["part"] == "ascending") & (d["g"] <= 1.0 + 1e-9)]
        for (kappa, rp, start), db in d.groupby(["kappa", "rperp", "start"]):
            r = abs(float(rp))
            name = "FM" if start.startswith("FM") else start[:3]
            e0, e2, e4, _ = (float(x) for x in coeffs(name, kappa, r))
            s = db.groupby("g")["E_per_spin"].agg(["mean", "std", "count"])
            g = s.index.to_numpy(); se = (s["std"] / np.sqrt(s["count"])).to_numpy()
            res2 = s["mean"].to_numpy() - (e0 + e2 * g ** 2)
            res4 = res2 - e4 * g ** 4
            m = g <= 0.8 + 1e-9
            out["residuals"].append({"file": f.name, "kappa": float(kappa), "r": r, "start": start,
                                     "chi2_O2_g<=0.8": float(np.sum((res2[m] / se[m]) ** 2)),
                                     "chi2_O4_g<=0.8": float(np.sum((res4[m] / se[m]) ** 2)), "n_g<=0.8": int(m.sum()),
                                     "res_O2_at_g1": float(res2[-1]) if abs(g[-1] - 1) < 1e-9 else None,
                                     "res_O4_at_g1": float(res4[-1]) if abs(g[-1] - 1) < 1e-9 else None,
                                     "se_at_g1": float(se[-1]) if abs(g[-1] - 1) < 1e-9 else None})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pt_order4_analysis.json").write_text(json.dumps(out, indent=1, default=float))
    print("wrote", OUT / "pt_order4_analysis.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if a.verify:
        verify()
    if a.analyze:
        analyze()


if __name__ == "__main__":
    main()

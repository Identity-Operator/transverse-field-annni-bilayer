"""O(g^8) audit: exact eps_n and kappa-derivatives at kappa = 1/2 for the requested structures (r = 0, 1), and
the boundary expansion kappa = 1/2 + k g^2 + k1 g^4 + k2 g^6 + k3 g^8 on the <3>|<23>, <23>|<223> and <223>|<2>
lines. Everything is computed from lc8.py alone (no repo PT code, no referee code). Output: g8_results.json."""
import json, sys, time
from fractions import Fraction as Fr
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import lc8

HERE = Path(__file__).parent
SEQ = {"FM": None, "<2>": (2,), "<3>": (3,), "<23>": (2, 3), "<223>": (2, 2, 3), "<233>": (2, 3, 3),
       "<2223>": (2, 2, 2, 3), "<2333>": (2, 3, 3, 3), "<23223>": (2, 3, 2, 2, 3),
       # checks of the 4-domain range of W8 (should sit exactly on hull edges):
       "<22223>": (2, 2, 2, 2, 3), "<2232223>": (2, 2, 3, 2, 2, 2, 3)}
F = lambda x: float(x)

def rho(seq):
    return Fr(0) if seq is None else Fr(len(seq), sum(seq))

res = {"method": "lc8.py: anchored linked clusters (<= 4 sites) on the infinite lattice, exact Fractions, "
                 "kappa jets; validated by validate_lc8.json, ed_e8_check.json, fullrs8_check.json"}
for r in (0, 1):
    R = {}
    for name, seq in SEQ.items():
        t0 = time.time()
        s, ncl = lc8.series(lc8.Structure(seq, r))
        d = {"rho": rho(seq), "eps2": s[2][0], "eps4": s[4][0], "eps6": s[6][0], "eps8": s[8][0],
             "d_eps2": s[2][1], "d2_eps2": 2 * s[2][2], "d3_eps2": 6 * s[2][3],
             "d_eps4": s[4][1], "d2_eps4": 2 * s[4][2], "d_eps6": s[6][1], "de0": s[0][1], "clusters": ncl}
        R[name] = d
        print(f"r={r} {name:9s} eps8 = {F(d['eps8']):+.6e}  (eps6 {F(d['eps6']):+.6e}; d_eps6 {F(d['d_eps6']):+.4e};"
              f" {ncl} clusters, {time.time()-t0:.0f}s)", flush=True)
        assert d["de0"] == 1 - 4 * d["rho"]
    # ---- boundary expansion
    def bnd(Wa, Wb, a, b):          # coefficient on the A|B line: 4 k (rho_B - rho_A) = W_B - W_A
        return (Wb - Wa) / (4 * (R[b]["rho"] - R[a]["rho"]))
    W2 = {n: R[n]["eps2"] for n in R}
    k = bnd(W2["<3>"], W2["<2>"], "<3>", "<2>")
    W4 = {n: k * R[n]["d_eps2"] + R[n]["eps4"] for n in R}
    k1m, k1p = bnd(W4["<3>"], W4["<23>"], "<3>", "<23>"), bnd(W4["<23>"], W4["<2>"], "<23>", "<2>")
    def W6(n, k1):
        d = R[n]; return k1 * d["d_eps2"] + k * k * d["d2_eps2"] / 2 + k * d["d_eps4"] + d["eps6"]
    def W8(n, k1, k2):
        d = R[n]
        return (k2 * d["d_eps2"] + k * k1 * d["d2_eps2"] + k ** 3 * d["d3_eps2"] / 6 + k1 * d["d_eps4"]
                + k * k * d["d2_eps4"] / 2 + k * d["d_eps6"] + d["eps8"])
    def dev(Wf, c, a, b):           # height of structure c above the chord a-b (exact)
        ra, rb, rc = R[a]["rho"], R[b]["rho"], R[c]["rho"]
        return Wf(c) - (Wf(a) + (Wf(b) - Wf(a)) * (rc - ra) / (rb - ra))
    # O(g^6) boundaries
    w6m = lambda n: W6(n, k1m); w6p = lambda n: W6(n, k1p)
    k2_3_23 = bnd(w6m("<3>"), w6m("<23>"), "<3>", "<23>")
    k2_23_223 = bnd(w6p("<23>"), w6p("<223>"), "<23>", "<223>")
    k2_223_2 = bnd(w6p("<223>"), w6p("<2>"), "<223>", "<2>")
    out = {"k": k, "k1_3|23": k1m, "k1_23|2": k1p, "k2_3|23": k2_3_23, "k2_23|223": k2_23_223, "k2_223|2": k2_223_2,
           "O6_height_223_vs_23-2": dev(w6p, "<223>", "<23>", "<2>"),
           "O6_height_233_vs_3-23": dev(w6m, "<233>", "<3>", "<23>")}
    # O(g^8) on the <223>|<2> line
    w8a = lambda n: W8(n, k1p, k2_223_2)
    h2223 = dev(w8a, "<2223>", "<223>", "<2>")
    out["O8_line_223|2"] = {
        "height_2223_vs_223-2": h2223,
        "height_22223_vs_2223-2 (expect 0)": dev(w8a, "<22223>", "<2223>", "<2>"),
        "height_2232223_vs_223-2223 (expect 0)": dev(w8a, "<2232223>", "<223>", "<2223>")}
    if h2223 < 0:
        k3a, k3b = bnd(w8a("<223>"), w8a("<2223>"), "<223>", "<2223>"), bnd(w8a("<2223>"), w8a("<2>"), "<2223>", "<2>")
        out["O8_line_223|2"].update({"k3_223|2223": k3a, "k3_2223|2": k3b, "window_2223_k3_width": k3b - k3a})
    # O(g^8) on the <23>|<223> line: <23223> should stay on the <23>-<223> segment
    w8b = lambda n: W8(n, k1p, k2_23_223)
    out["O8_line_23|223"] = {"height_23223_vs_23-223": dev(w8b, "<23223>", "<23>", "<223>")}
    # O(g^8) on the <3>|<23> line: <233> and <2333>
    w8c = lambda n: W8(n, k1m, k2_3_23)
    h6, h8 = dev(w6m, "<233>", "<3>", "<23>"), dev(w8c, "<233>", "<3>", "<23>")
    out["O8_line_3|23"] = {"height_233_g6": h6, "height_233_g8": h8,
                           "sign_reversal_g": (F(-h6 / h8) ** 0.5 if (h6 > 0) != (h8 > 0) else None),
                           "height_2333_g6": dev(w6m, "<2333>", "<3>", "<23>"), "height_2333_g8": dev(w8c, "<2333>", "<3>", "<23>")}
    res[f"r={r}"] = {"structures": R, "boundaries": out}
    print(f"r={r}: k={k} k1-={F(k1m):.6f} k1+={F(k1p):.6f} k2(3|23)={F(k2_3_23):+.5e} k2(23|223)={F(k2_23_223):+.5e} "
          f"k2(223|2)={F(k2_223_2):+.5e}", flush=True)
    print(f"   <223>|<2> line: <2223> height {F(h2223):+.4e} g^8; checks {F(out['O8_line_223|2']['height_22223_vs_2223-2 (expect 0)'])}, "
          f"{F(out['O8_line_223|2']['height_2232223_vs_223-2223 (expect 0)'])}", flush=True)
    if "window_2223_k3_width" in out["O8_line_223|2"]:
        print(f"   <2223> window width {F(out['O8_line_223|2']['window_2223_k3_width']):.4e} g^8", flush=True)
    print(f"   <23>|<223> line: <23223> height {F(out['O8_line_23|223']['height_23223_vs_23-223'])}", flush=True)
    print(f"   <3>|<23> line: <233> height {F(h6):+.4e} g^6 {F(h8):+.4e} g^8; sign reversal at g = {out['O8_line_3|23']['sign_reversal_g']}", flush=True)

def ser(o):
    if isinstance(o, Fr):
        return {"exact": str(o), "float": float(o)}
    if isinstance(o, dict):
        return {k: ser(v) for k, v in o.items()}
    return o
(HERE / "g8_results.json").write_text(json.dumps(ser(res), indent=1))
print("wrote", HERE / "g8_results.json")

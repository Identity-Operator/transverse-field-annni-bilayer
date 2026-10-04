"""Numbers for the Sec. V.A draft (step-2 <3> wedge), computed with the S5a rules of experiments/s5a_analyze_v3.py:
anchored "rich" quadrature (Richardson/3-8 on the 0.1 run from g = 0.1; the step-2 grids have no fine band, so the
whole 0.1 run is the coarse segment; the fine-grid reruns have no 0.1 run and use the trapezoid), cubic-spline root,
delete-one jackknife SE over chains, 68% intervals with t_{n-1}. Run from vu_work/ (1 thread)."""
import sys, json
sys.path.insert(0, "experiments")
from pathlib import Path
import numpy as np
from scipy.stats import t as tdist
import s5a_analyze_v3 as m

BR = Path("10_results/bilayer_annni_paper_results/reanalysis_v3/sse_validation/branches")
FG = Path("10_results/bilayer_annni_paper_results/reanalysis_v3/sse_validation/fine_grid_check_12")
PT4 = json.loads(Path("10_results/bilayer_annni_paper_results/reanalysis_v3/perturbation/pt_order4_analysis.json").read_text())["boundaries"]
OUT = Path(sys.argv[1])
KAPPAS = [0.47, 0.48, 0.49, 0.51, 0.52, 0.53, 0.5625, 0.625, 0.75]


def pair(k):
    return ("FM", "<3>") if k < 0.5 else ("<2>", "<3>")        # (old, new): new drops below old at g*


def crossing_of(f, k=None):
    bs, kk, r = m.load_file(f)
    br = {b.name: b for b in bs}
    old, new = pair(kk)
    res = {}
    for meth in ("rich", "unanch"):
        FO, FN = m.free_energy(br[old], kk, r, meth), m.free_energy(br[new], kk, r, meth)
        e = m.edge(br[old], FO, br[new], FN, "cubic")
        res[meth] = {"g": e["value"], "se": e["se"], "df": e["df"], "hw68": m.t68(e["df"]) * e["se"], "nsc": e["n_sign_changes"]}
    seg = m.coarse_segment(br[old].g, br[old].g_f)
    res["coarse_m"] = None if seg is None else int(seg[1] - seg[0])
    res["grid_max"] = float(br[old].g[-1])
    res["chains"] = [br[old].n, br[new].n]
    return kk, r, res


def theory(k, r):
    old, new = pair(k)
    sq = lambda n: None if n == "FM" else (2,) if n == "<2>" else (3,)
    return {f"O{o}": m.pt_cross(sq(old), sq(new), k, r, o) for o in (2, 4, 6)}


out = {"L12": {}, "L24": {}, "fine": {}}
for L in (12, 24):
    for f in sorted(BR.glob(f"branches_L{L}_k*_r*.csv")):
        k, r, res = crossing_of(f)
        res["theory"] = theory(k, r)
        res["file"] = f.name
        out[f"L{L}"][f"{k:g}_{r:g}"] = res
for f in sorted(FG.glob("fine_L12_*.csv")):
    k, r, res = crossing_of(f)
    res["file"] = f.name
    out["fine"][f"{k:g}_{r:g}"] = res
# combined (inverse-variance) where a fine rerun exists
for key, v in out["L12"].items():
    if key in out["fine"]:
        a, b = v["rich"], out["fine"][key]["rich"]
        w = np.array([1 / a["se"] ** 2, 1 / b["se"] ** 2])
        g = float((w * [a["g"], b["g"]]).sum() / w.sum()); se = float(1 / np.sqrt(w.sum()))
        v["combined"] = {"g": g, "se": se, "z_prod_vs_fine": (a["g"] - b["g"]) / np.hypot(a["se"], b["se"])}
    v["best"] = v.get("combined", {"g": v["rich"]["g"], "se": v["rich"]["se"]})
# L24 vs L12
out["L24_vs_L12"] = {}
for key, v in out["L24"].items():
    a, b = out["L12"][key]["rich"], v["rich"]
    out["L24_vs_L12"][key] = {"L12": a["g"], "L24": b["g"], "rel_pct": 100 * (b["g"] - a["g"]) / a["g"], "z": (b["g"] - a["g"]) / np.hypot(a["se"], b["se"])}


# leading slopes: y = |kappa - 1/2| = A x + B x^2, x = g*^2, B fixed at O(g^4); per point A_i = y/x - B x
def slopes(src, window, which="best"):
    rows, fits = [], {}
    for r in (0, 1):
        F = 6.0 / ((2 + r) * (3 + r) * (5 + r))
        for side in ("FM3", "32"):
            Apred = F / 8 if side == "FM3" else F / 4
            B = -PT4[f"r={r}"]["k1_FM3"] if side == "FM3" else PT4[f"r={r}"]["k1_32"]
            pts = []
            for key, v in src.items():
                k, rr = (float(x) for x in key.split("_"))
                if rr != r or (side == "FM3") != (k < 0.5):
                    continue
                g, sg = v[which]["g"], v[which]["se"]
                lam = g / (2 + r)
                if window == "lam" and lam > 0.5 + 1e-12:
                    continue
                if window == "g1.4" and g > 1.4:
                    continue
                x, y = g * g, abs(k - 0.5)
                Ai = y / x - B * x
                sA = abs(y / x ** 2 + B) * 2 * g * sg
                pts.append((k, g, sg, lam, Ai, sA))
                if window == "lam":
                    rows.append({"r": r, "side": side, "kappa": k, "g": g, "se": g and sg, "lambda": lam,
                                 "A_over_pred": Ai / Apred, "se_A_over_pred": sA / Apred})
            if not pts:
                continue
            A = np.array([p[4] for p in pts]); sA = np.array([p[5] for p in pts])
            w = 1 / sA ** 2
            Ab = float((w * A).sum() / w.sum()); se = float(1 / np.sqrt(w.sum()))
            fits[f"r{r}_{side}"] = {"n": len(pts), "A": Ab, "se": se, "A_pred": Apred, "ratio_to_pred": Ab / Apred,
                                    "ratio_se": se / Apred, "chi2": float((w * (A - Ab) ** 2).sum()), "B_fixed": B}
    for side in ("FM3", "32"):
        a, b = fits.get(f"r0_{side}"), fits.get(f"r1_{side}")
        if a and b:
            fits[f"ratio_{side}"] = {"value": a["A"] / b["A"], "se": a["A"] / b["A"] * np.hypot(a["se"] / a["A"], b["se"] / b["A"]), "pred": 2.4}
    return rows, fits


out["slopes"] = {}
for window in ("lam", "g1.4", "all"):
    rows, fits = slopes(out["L12"], window)
    out["slopes"][window] = {"fits": fits, **({"points": rows} if window == "lam" else {})}
rows, fits = slopes(out["L12"], "lam", "rich")
out["slopes"]["lam_production_only"] = {"fits": fits}
OUT.write_text(json.dumps(out, indent=1, default=float))

# readable summary
print(f"{'k':>6s} r  g*_rich(se) [unanch]      O(g2)   O(g4)   O(g6)  lam   (QMC-O6)/se  fine(se)  best(se)  m")
for key, v in out["L12"].items():
    k, r = key.split("_"); th = v["theory"]; a = v["rich"]; fi = out["fine"].get(key)
    print(f"{k:>6s} {r}  {a['g']:.4f}({a['se']*1e4:.0f}) [{v['unanch']['g']:.4f}({v['unanch']['se']*1e4:.0f})]  "
          f"{th['O2']:.4f}  {th['O4']:.4f}  {th['O6']:.4f}  {a['g']/(2+int(r)):.2f}  {(a['g']-th['O6'])/a['se']:+6.1f}      "
          + (f"{fi['rich']['g']:.4f}({fi['rich']['se']*1e4:.0f})" if fi else " " * 12)
          + f"  {v['best']['g']:.4f}({v['best']['se']*1e4:.0f})  {v['coarse_m']}")
print("L24 vs L12:", {k: (round(v['rel_pct'], 2), round(v['z'], 1)) for k, v in out["L24_vs_L12"].items()})
for w, d in out["slopes"].items():
    print(w, {k: (round(v.get("ratio_to_pred", v.get("value")), 4), round(v.get("ratio_se", v.get("se")), 4), v.get("n"), round(v.get("chi2", 0), 1))
              for k, v in d["fits"].items()})

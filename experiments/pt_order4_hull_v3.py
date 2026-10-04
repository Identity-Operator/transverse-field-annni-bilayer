#!/usr/bin/env python3
"""Extended O(g^4) phase analysis at the <3>|<2> boundary (Paper 1, plan stage S5a, step 2).

Uses the per-column O(g^4) energies of experiments/pt_order4_v3.py in exact rational arithmetic.
With kappa = 1/2 + k g^2 + k1 g^4 and k = F/4 (the <3>|<2> boundary at O(g^2)), every sequence of
two- and three-site domains is degenerate at O(g^2); its O(g^4) energy per spin is
k1 (1 - 4 rho) + W, W = k e2'(1/2) + e4(1/2), rho = walls per site. The stable sequences are the
lower convex hull of W against rho.

This script
  (1) enumerates all cyclic 2/3 sequences up to --max-domains domains (default 14), plus sequences
      that contain 4- or 5-site domains up to --max-long domains, and checks which enter the hull;
  (2) gives the exact <23> window (k1 bounds), its width, and the r dependence (r = 0, 1/2, 1, 2);
  (3) gives the exact height above the hull of named candidates (<233>, <223>, <2333>, <2223>,
      <23233>, <22323>); a height of exactly 0 means the candidate is degenerate with the
      neighbouring hull phases at O(g^4), so higher orders decide it.

Usage (from vu_work/):  python experiments/pt_order4_hull_v3.py [--max-domains 14] [--max-long 7]
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
import time
from fractions import Fraction as Fr

from pt_order4_v3 import cyclic_sequences, lower_hull, per_column

OUT = Path("10_results/bilayer_annni_paper_results/reanalysis_v3/perturbation")
NAMED = {"<3>": (3,), "<23>": (2, 3), "<2>": (2,), "<233>": (2, 3, 3), "<223>": (2, 2, 3),
         "<2333>": (2, 3, 3, 3), "<2223>": (2, 2, 2, 3), "<23233>": (2, 3, 2, 3, 3), "<22323>": (2, 2, 3, 2, 3)}


def canon(seq):
    return min(seq[i:] + seq[:i] for i in range(len(seq)))


def height_above(hull, rho, W):
    """Height of (rho, W) above the piecewise-linear lower hull (exact)."""
    for (x1, y1, _), (x2, y2, _) in zip(hull, hull[1:]):
        if x1 <= rho <= x2:
            return W - (y1 + (y2 - y1) * (rho - x1) / (x2 - x1))
    return None


def analyse(r, max_domains, max_long):
    R = Fr(r)
    half = Fr(1, 2)
    F3 = Fr(6) / ((2 + R) * (3 + R) * (5 + R))
    k = F3 / 4
    seqs = cyclic_sequences((2, 3), max_domains)
    seqs += [s for s in cyclic_sequences((2, 3, 4, 5), max_long) if any(n >= 4 for n in s)]
    pts, pen = [], {}
    g2_ref = None
    for seq in seqs:
        e0, e2, e4, e2p = per_column(list(seq), half, R)
        rho = Fr(len(seq), sum(seq))
        g2 = k * (1 - 4 * rho) + e2
        if g2_ref is None:
            g2_ref = g2
        if g2 != g2_ref:
            pen[seq] = g2 - g2_ref
            continue
        pts.append((rho, k * e2p + e4, seq))
    hull = lower_hull(pts)
    res = {"r": str(R), "F": str(F3), "n_sequences": len(seqs), "n_degenerate_at_g2": len(pts),
           "min_g2_penalty_long_domains": str(min(pen.values())) if pen else None,
           "all_long_penalised_positive": all(v > 0 for v in pen.values()),
           "hull": ["".join(map(str, h[2])) for h in hull]}
    Wd = {canon(p[2]): (p[0], p[1]) for p in pts}
    # <23> window: kappa = 1/2 + k g^2 + k1 g^4 with k1 = slope/4 of the hull segments
    rho3, W3 = Wd[(3,)]; rho23, W23 = Wd[(2, 3)]; rho2, W2 = Wd[(2,)]
    s1 = (W23 - W3) / (rho23 - rho3); s2 = (W2 - W23) / (rho2 - rho23)
    res["window_k1"] = [str(s1 / 4), str(s2 / 4)]
    res["window_width_g4"] = str((s2 - s1) / 4)
    res["window_width_g4_float"] = float((s2 - s1) / 4)
    res["chord_depth_23"] = str(W23 - (W3 + (W2 - W3) * (rho23 - rho3) / (rho2 - rho3)))
    res["named_heights"] = {}
    for name, seq in NAMED.items():
        c = canon(seq)
        if c in Wd:
            h = height_above(hull, *Wd[c])
            res["named_heights"][name] = {"rho": str(Wd[c][0]), "height": str(h), "height_float": float(h)}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-domains", type=int, default=14)
    ap.add_argument("--max-long", type=int, default=7)
    a = ap.parse_args()
    out = {"args": vars(a), "by_r": {}}
    for r in (Fr(0), Fr(1, 2), Fr(1), Fr(2)):
        t0 = time.time()
        res = analyse(r, a.max_domains, a.max_long)
        out["by_r"][str(r)] = res
        print(f"r={str(r):4s} F={res['F']:>8s}  sequences {res['n_sequences']}, degenerate at g^2: {res['n_degenerate_at_g2']}, "
              f"long domains penalised: {res['all_long_penalised_positive']}  ({time.time() - t0:.0f}s)")
        print(f"   hull: {res['hull']}")
        print(f"   <23> window: k1 in ({res['window_k1'][0]}, {res['window_k1'][1]}),  width {res['window_width_g4_float']:.6f} g^4;"
              f"  chord depth of <23>: {res['chord_depth_23']}")
        print("   heights above hull: " + ", ".join(f"{n} {v['height_float']:+.3e}" for n, v in res["named_heights"].items()))
    w = {rk: out["by_r"][rk]["window_width_g4_float"] for rk in out["by_r"]}
    out["width_ratio_r0_r1"] = w["0"] / w["1"]
    print(f"<23> window width ratio r=0 / r=1: {out['width_ratio_r0_r1']:.3f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pt_order4_hull.json").write_text(json.dumps(out, indent=1))
    print("wrote", OUT / "pt_order4_hull.json")


if __name__ == "__main__":
    main()

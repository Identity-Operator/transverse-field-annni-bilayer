#!/usr/bin/env python3
"""Analysis of the revision SSE queue (experiments/run_revision1.sh), block by block, with the frozen functions of
experiments/s5a_analyze_v3.py and the correlation-aware tools of experiments/revision1_stats_v3.py.

Blocks (--block):
  V1, V2, V4  <2>, <23>, <3> on ONE 20x60 lattice (kappa = 0.53 r = 1; 0.52 r = 1; 0.53 r = 0): edges, T1-type
              differences at g_c (both on 20x60), the same quantities from S5a (20x20 / 20x30) for comparison, the
              per-structure cross-lattice offsets (jackknife covariance), and the energy route (e - f near g_c).
  V3          <23>, <2> on 40x40 at kappa = 0.53, r = 1, against 20x20 (and <23> against 20x30).
  V5          T = 0.10 at kappa = 0.53, r = 1 (20x20 and 20x30), against T = 0.15 (P3).
  V6          <23>, <3> on 20x30 at kappa = 0.53, r = 1 with four times the sweeps, against P3 (lag test).
  A           anneals in sse_validation/anneal_rect/ (A1-A6): the wavevector that first grows, the argmax path, the final
              structure (wall density from Cy, n* of S(q)) and the final energy against the template series.
Outputs: reanalysis_v3/revision1/queue_<block>.{json,txt}.

Run from vu_work/ (1 thread):
    OMP_NUM_THREADS=1 ~/.venvs/annni-py311/bin/python -W ignore experiments/revision1_queue_v3.py --block V1
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json

import numpy as np
import pandas as pd

import s5a_analyze_v3 as m
import revision1_stats_v3 as rs

R3 = rs.R3
SEQ, S5A, OUT = rs.SEQ, rs.S5A, rs.OUT
ANN = R3 / "sse_validation" / "anneal_rect"
REVSEQ = SEQ                    # where the revision seqbr files are written (same directory as S5a)
VCASES = {"V1": (0.53, 1, "P3"), "V2": (0.52, 1, "P6"), "V4": (0.53, 0, "P1"),
          "V7": (0.53, 1, "P3"),       # replicate of V1 with fresh seeds (file suffix _rep)
          "V8": (0.52, 0, "P5"),
          "V2rep": (0.52, 1, "P6")}   # replicate of V2 with fresh seeds (file suffix _rep)
VSUFFIX = {"V7": "_rep", "V2rep": "_rep"}
# 20x30 chain sets pooled per lattice for the all-chains shift: P3 = 1x + 4x (_x4), P6 = orig + follow-up (_fu1)
S30SUF = {(0.53, 1): "_x4", (0.52, 1): "_fu1"}


def lattice_shift(kappa, r):
    """Delta F_B(20x60) - Delta F_B(20x30) at g_c with ALL chains pooled per lattice (20x60: base + _rep if present;
    20x30: base + the extra set of S30SUF). Also the per-structure pooled offsets."""
    gc = m.theory(kappa, r)["g_c"]
    f60 = m.seqbr(20, 60, kappa, r, 0.15, "2-23-3", REVSEQ)
    s60 = [f60] + ([f60.with_name(f60.stem + "_rep.csv")] if f60.with_name(f60.stem + "_rep.csv").exists() else [])
    f30 = m.seqbr(20, 30, kappa, r, 0.15, "23-3")
    s30 = [f30, f30.with_name(f30.stem + S30SUF[(kappa, r)] + ".csv")]
    def pooled(files, names):
        sets = [{b.name: b for b in m.load_file(f)[0]} for f in files]
        out = {}
        for nm in names:
            b = sets[0][nm]
            for extra in sets[1:]:
                b = rs.pool(b, extra[nm])
            out[nm] = b
        return out
    P60, P30 = pooled(s60, ("<23>", "<3>")), pooled(s30, ("<23>", "<3>"))
    F60 = {nm: f_of(b, kappa, r) for nm, b in P60.items()}
    F30 = {nm: f_of(b, kappa, r) for nm, b in P30.items()}
    j60, j30 = int(np.argmin(np.abs(P60["<23>"].g - gc))), int(np.argmin(np.abs(P30["<23>"].g - gc)))
    w60 = m.welch(F60["<23>"][:, j60], F60["<3>"][:, j60])
    w30 = m.welch(F30["<23>"][:, j30], F30["<3>"][:, j30])
    per = {nm: rs.cov_compare(P60[nm], F60[nm], P30[nm], F30[nm], gc)["offset_at_gc"] for nm in ("<23>", "<3>")}
    return {"dFB_20x60": w60, "dFB_20x30": w30, "shift": w60["d"] - w30["d"], "se": float(np.hypot(w60["se"], w30["se"])),
            "chains_20x60": P60["<3>"].n, "chains_20x30": P30["<3>"].n, "files_20x60": [f.name for f in s60],
            "files_20x30": [f.name for f in s30], "per_structure": per}
PT8 = R3 / "perturbation" / "pt_order8_tables.json"      # exact-kappa O(g^8) values (6e, 9b, referee A)


def o8(case):
    """O(g^8) predictions for a P case: edges lo/hi (first root or None) and Delta F at g_c (per spin)."""
    t = next(x for x in json.loads(PT8.read_text())["tab5"] if x["case"] == case)
    first = lambda v: (v[0] if v else None)
    return {"lo": first(t["lo8"]), "hi": first(t["hi8"]), "dF2": t["dF2_8"] * 1e-4, "dF3": t["dF3_8"] * 1e-4,
            "lo6": first(t["lo6"]), "hi6": first(t["hi6"]), "dF2_6": t["dF2_6"] * 1e-4, "dF3_6": t["dF3_6"] * 1e-4}


def f_of(b, kappa, r):
    return m.free_energy(b, kappa, r, "rich")


def s5a_ref(case):
    return json.loads((S5A / f"s5a_{case}.json").read_text())


def edge_line(tag, e, ref=None, pred=None, pred8=None):
    s = f"{tag}: {e['value']:.4f}({e['se']*1e4:.0f})"
    if ref is not None:
        z = (e["value"] - ref["value"]) / np.hypot(e["se"], ref["se"])
        s += f" | S5a {ref['value']:.4f}({ref['se']*1e4:.0f}), diff {e['value']-ref['value']:+.4f} ({z:+.1f} sigma)"
    if pred:
        s += f" | O(g^6) {pred:.4f} ({(e['value']-pred)/e['se']:+.1f} sigma)"
    if pred8 is False:                        # no O(g^8) value tabulated for this edge
        pass
    elif pred8 is not None:
        s += f" | O(g^8) {pred8:.4f} ({(e['value']-pred8)/e['se']:+.1f} sigma)"
    elif pred is not None and tag.startswith("edge hi"):
        s += " | O(g^8): no crossing below 1.6"
    return s


def band_ef(b, kappa, r, gc):
    D = rs.energies(b, kappa, r)["op"] - f_of(b, kappa, r)
    mask = np.abs(b.g - gc) <= 0.1 + 1e-9
    v, se, _ = rs.band_stat(D, mask)
    return D, mask, {"band_mean": v, "se": se, "z": v / se}


# ---------------------------------------------------------------- V1, V2, V4: one 20x60 lattice
def block_V60(block):
    kappa, r, case = VCASES[block]
    f = m.seqbr(20, 60, kappa, r, 0.15, "2-23-3", REVSEQ)
    f = f.with_name(f.stem + VSUFFIX.get(block, "") + ".csv")
    bs = {b.name: b for b in rs.load_full(f)[0]}
    th = m.theory(kappa, r)
    gc = th["g_c"]
    ref = s5a_ref(case)
    res = {"file": f.name, "kappa": kappa, "r": r, "g_c": gc, "compare": case}
    L = [f"{block}: <2>, <23>, <3> on 20x60, kappa={kappa}, r={r}, g_c={gc:.4f} (S5a reference {case}: 20x20 / 20x30)"]
    F = {n: f_of(b, kappa, r) for n, b in bs.items()}
    t8 = o8(case)
    res["O8"] = t8
    for tag, old, new in (("lo", "<2>", "<23>"), ("hi", "<23>", "<3>"), ("32", "<2>", "<3>")):
        e = m.edge(bs[old], F[old], bs[new], F[new], "cubic")
        pr = th["g_32" if tag == "32" else tag]["O(g^6)"]
        res[f"edge_{tag}"] = e
        L.append("  " + edge_line(f"edge {tag} ({new} below {old})", e, ref["edges"][tag]["rich/cubic"], pr,
                                  t8.get(tag) if tag != "32" else False))
    j = int(np.argmin(np.abs(bs["<23>"].g - gc)))
    for tag, other, rk in (("A", "<2>", "A_23_minus_2"), ("B", "<3>", "B_23_minus_3")):
        w = m.welch(F["<23>"][:, j], F[other][:, j])
        rf = ref["tests"]["T1"]["rich"][rk]
        res[f"T1_{tag}"] = w
        z = (w["d"] - rf["d"]) / np.hypot(w["se"], rf["se"])
        p8 = t8["dF2" if tag == "A" else "dF3"]
        res[f"T1_{tag}"]["excess_vs_O8"] = {"common": w["d"] - p8, "common_z": (w["d"] - p8) / w["se"],
                                           "s5a": rf["d"] - p8, "s5a_z": (rf["d"] - p8) / rf["se"]}
        L.append(f"  Delta F {tag} (<23> - {other}) at g_c: {w['d']*1e4:+.2f}({w['se']*1e4:.2f})e-4, t {w['t']:+.1f} (dof {w['dof']:.0f})"
                 f" | S5a {rf['d']*1e4:+.2f}({rf['se']*1e4:.2f}) -> diff {z:+.1f} sigma | O(g^6) {th['Delta_gc']['O(g^6)']*1e4:+.2f}"
                 f" -> {(w['d']-th['Delta_gc']['O(g^6)'])/w['se']:+.1f} sigma | O(g^8) {p8*1e4:+.2f} -> {(w['d']-p8)/w['se']:+.1f} sigma")
        L.append(f"      excess over O(g^8): common lattice {(w['d']-p8)*1e4:+.2f}({w['se']*1e4:.2f})e-4 vs S5a {(rf['d']-p8)*1e4:+.2f}({rf['se']*1e4:.2f})e-4")
    # per-structure cross-lattice offsets, against the original S5a chains and (P6) the declared follow-up chains
    res["cross_lattice"] = {}
    refsets = [("orig", "")]
    if case == "P6":
        refsets.append(("fu1", "_fu1"))
        for lab, fn in (("fu1 only", "s5a_P6_fu1_only_POSTHOC.json"), ("pooled 36", "s5a_P6_pooled36_POSTHOC.json")):
            t = json.loads((S5A / "posthoc_P6" / fn).read_text())["tests"]["T1"]["rich"]
            L.append(f"  reference {lab} (20x20/20x30): Delta F A {t['A_23_minus_2']['d']*1e4:+.2f}({t['A_23_minus_2']['se']*1e4:.2f}),"
                     f" B {t['B_23_minus_3']['d']*1e4:+.2f}({t['B_23_minus_3']['se']*1e4:.2f}) e-4")
    for lab, suf in refsets:
        fa = m.seqbr(20, 20, kappa, r, 0.15, "23-2").with_name(m.seqbr(20, 20, kappa, r, 0.15, "23-2").stem + suf + ".csv")
        fb = m.seqbr(20, 30, kappa, r, 0.15, "23-3").with_name(m.seqbr(20, 30, kappa, r, 0.15, "23-3").stem + suf + ".csv")
        if not (fa.exists() and fb.exists()):
            continue
        A = {b.name: b for b in m.load_file(fa)[0]}
        B = {b.name: b for b in m.load_file(fb)[0]}
        for nm, other in (("<2>", A["<2>"]), ("<23>", A["<23>"]), ("<23>", B["<23>"]), ("<3>", B["<3>"])):
            c = rs.cov_compare(bs[nm], F[nm], other, f_of(other, kappa, r), gc)
            res["cross_lattice"][f"{nm}@20x60-{other.lat}[{lab}]"] = c
            o = c["offset_at_gc"]
            L.append(f"  F {nm} 20x60 - {other.lat} [{lab}, {other.n} ch]: offset at g_c {o['d']*1e5:+.2f}({o['se']*1e5:.2f})e-5 (t {o['t']:+.2f}),"
                     f" Hotelling p {c['hotelling']['p']:.2f}, max|z| {c['max_abs_z']:.2f} (p_corr {c['p_max_abs_z_correlated']:.2f}), n_eff {c['n_eff']:.1f}")
        if lab == "fu1":
            j2 = int(np.argmin(np.abs(A["<23>"].g - gc)))
            for tag, x, y in (("A", A["<23>"], A["<2>"]), ("B", B["<23>"], B["<3>"])):
                w = m.welch(f_of(x, kappa, r)[:, j2], f_of(y, kappa, r)[:, j2])
                L.append(f"  fu1 Delta F {tag} at g_c (recomputed): {w['d']*1e4:+.2f}({w['se']*1e4:.2f})e-4")
    # energy route
    res["energy"] = {}
    D = {}
    for nm, b in bs.items():
        D[nm], mask, st = band_ef(b, kappa, r, gc)
        res["energy"][nm] = st
    for tag, other in (("A", "<2>"), ("B", "<3>")):
        v, se, _ = m.jackknife(lambda mm: float(np.nanmean(mm[0][mask]) - np.nanmean(mm[1][mask])), [D["<23>"], D[other]])
        art = -(res[f"T1_{tag}"]["d"] - th["Delta_gc"]["O(g^6)"])
        res["energy"][f"d(e-f)_{tag}"] = {"band_mean": v, "se": se, "artefact_pred": art}
    L.append("  energy route (|g - g_c| <= 0.1, e-5): " + ", ".join(f"{nm} e-f {res['energy'][nm]['band_mean']*1e5:+.2f}({res['energy'][nm]['se']*1e5:.2f})"
                                                              for nm in bs) +
             "; " + ", ".join(f"d(e-f)_{t} {res['energy'][f'd(e-f)_{t}']['band_mean']*1e5:+.2f}({res['energy'][f'd(e-f)_{t}']['se']*1e5:.2f})"
                              f" [artefact would give {res['energy'][f'd(e-f)_{t}']['artefact_pred']*1e5:+.1f}]" for t in ("A", "B")))
    if block in VSUFFIX:
        f1 = m.seqbr(20, 60, kappa, r, 0.15, "2-23-3", REVSEQ)
        b1 = {b.name: b for b in m.load_file(f1)[0]}
        res["replicate"] = {}
        base = {"V7": "V1", "V2rep": "V2"}[block]
        L.append(f"  replicate check against {base} (same lattice, independent seeds):")
        for nm in ("<2>", "<23>", "<3>"):
            c = rs.cov_compare(bs[nm], F[nm], b1[nm], f_of(b1[nm], kappa, r), gc)
            res["replicate"][nm] = c
            o = c["offset_at_gc"]
            L.append(f"    F {nm} {block} - {base}: {o['d']*1e5:+.2f}({o['se']*1e5:.2f})e-5 (t {o['t']:+.2f}), Hotelling p {c['hotelling']['p']:.2f}")
        pb = {nm: rs.pool(b1[nm], bs[nm]) for nm in ("<2>", "<23>", "<3>")}
        PF = {nm: f_of(b, kappa, r) for nm, b in pb.items()}
        jp = int(np.argmin(np.abs(pb["<23>"].g - gc)))
        res["pooled_V1_V7"] = {}                       # (pooled base + replicate; name kept for V7)
        for tag, old, new in (("lo", "<2>", "<23>"), ("hi", "<23>", "<3>")):
            e = m.edge(pb[old], PF[old], pb[new], PF[new], "cubic")
            res["pooled_V1_V7"][f"edge_{tag}"] = e
            L.append(f"    pooled {base}+{block} (24 ch) " + edge_line(f"edge {tag}", e, None, th[tag]["O(g^6)"], t8.get(tag)))
        for tag, other in (("A", "<2>"), ("B", "<3>")):
            w = m.welch(PF["<23>"][:, jp], PF[other][:, jp])
            p8 = t8["dF2" if tag == "A" else "dF3"]
            res["pooled_V1_V7"][f"T1_{tag}"] = w
            L.append(f"    pooled {base}+{block} Delta F {tag} at g_c: {w['d']*1e4:+.2f}({w['se']*1e4:.2f})e-4 (t {w['t']:+.1f}, dof {w['dof']:.0f})"
                     f" | O(g^8) {p8*1e4:+.2f} -> excess {(w['d']-p8)*1e4:+.2f}({w['se']*1e4:.2f}), {(w['d']-p8)/w['se']:+.1f} sigma")
    if block in VSUFFIX:
        res["lattice_shift"] = {}
        L.append("  all-chains per-lattice Delta F_B shift, 20x60 minus 20x30 (e-4), and per-structure pooled offsets (e-5):")
        for (kk, rr), lab in (((0.53, 1), "P3"), ((0.52, 1), "P6")):
            sh = lattice_shift(kk, rr)
            res["lattice_shift"][lab] = sh
            L.append(f"    {lab}: 20x60 ({sh['chains_20x60']} ch) {sh['dFB_20x60']['d']*1e4:+.2f}({sh['dFB_20x60']['se']*1e4:.2f}) vs"
                     f" 20x30 ({sh['chains_20x30']} ch) {sh['dFB_20x30']['d']*1e4:+.2f}({sh['dFB_20x30']['se']*1e4:.2f}) -> shift"
                     f" {sh['shift']*1e4:+.2f}({sh['se']*1e4:.2f}) = {sh['shift']/sh['se']:+.1f} sigma | per structure: " +
                     ", ".join(f"{nm} {o['d']*1e5:+.2f}({o['se']*1e5:.2f}) t {o['t']:+.2f}" for nm, o in sh["per_structure"].items()))
        v = [(x["shift"], x["se"]) for x in res["lattice_shift"].values()]
        w_ = np.array([1 / s_ ** 2 for _, s_ in v]); vv = np.array([x for x, _ in v])
        mu, se = float((w_ * vv).sum() / w_.sum()), float(1 / np.sqrt(w_.sum()))
        res["lattice_shift"]["combined"] = [mu, se]
        L.append(f"    combined P3 + P6: {mu*1e4:+.2f}({se*1e4:.2f})e-4 = {mu/se:+.1f} sigma")
    cuts = {nm: [float(b.g[c]) for c in b.cut_idx if c < len(b.g)] for nm, b in bs.items()}
    pulls = {nm: float((b.E[:, 0].mean() - m.e_series(m.series(b.seq, kappa, r, 4), m.G0)) / (b.E[:, 0].std(ddof=1) / np.sqrt(b.n)))
             for nm, b in bs.items()}
    res["cuts"], res["g0_pulls"] = cuts, pulls
    L.append(f"  chains: {[(nm, b.n) for nm, b in bs.items()]}; cuts {cuts}; g0 pulls {{{', '.join(f'{k}: {v:+.2f}' for k, v in pulls.items())}}}")
    return res, L


# ---------------------------------------------------------------- V3: 40x40 at kappa = 0.53, r = 1
def block_V3():
    kappa, r = 0.53, 1
    K = {b.name: b for b in m.load_file(m.seqbr(40, 40, kappa, r, 0.15, "23-2", REVSEQ))[0]}
    A = {b.name: b for b in m.load_file(m.seqbr(20, 20, kappa, r, 0.15, "23-2"))[0]}
    B = {b.name: b for b in m.load_file(m.seqbr(20, 30, kappa, r, 0.15, "23-3"))[0]}
    th = m.theory(kappa, r); gc = th["g_c"]; ref = s5a_ref("P3")
    res, L = {"g_c": gc}, [f"V3: <23>, <2> on 40x40 at kappa={kappa}, r={r} (reference P3)"]
    e = m.edge(K["<2>"], f_of(K["<2>"], kappa, r), K["<23>"], f_of(K["<23>"], kappa, r), "cubic")
    res["edge_lo"] = e
    L.append("  " + edge_line("edge lo (<23> below <2>) 40x40", e, ref["edges"]["lo"]["rich/cubic"], th["lo"]["O(g^6)"], o8("P3")["lo"]))
    j = int(np.argmin(np.abs(K["<23>"].g - gc)))
    w = m.welch(f_of(K["<23>"], kappa, r)[:, j], f_of(K["<2>"], kappa, r)[:, j]); rf = ref["tests"]["T1"]["rich"]["A_23_minus_2"]
    res["T1_A"] = w
    L.append(f"  Delta F A at g_c 40x40 {w['d']*1e4:+.2f}({w['se']*1e4:.2f})e-4 (t {w['t']:+.1f}) | 20x20 {rf['d']*1e4:+.2f}({rf['se']*1e4:.2f})"
             f" | O(g^6) {th['Delta_gc']['O(g^6)']*1e4:+.2f}")
    res["cross_lattice"] = {}
    for nm, other in (("<23>", A["<23>"]), ("<23>", B["<23>"]), ("<2>", A["<2>"])):
        c = rs.cov_compare(K[nm], f_of(K[nm], kappa, r), other, f_of(other, kappa, r), gc)
        res["cross_lattice"][f"{nm}@40x40-{other.lat}"] = c
        o = c["offset_at_gc"]
        L.append(f"  F {nm} 40x40 - {other.lat}: offset at g_c {o['d']*1e5:+.2f}({o['se']*1e5:.2f})e-5 (t {o['t']:+.2f}),"
                 f" Hotelling p {c['hotelling']['p']:.2f}, max|z| {c['max_abs_z']:.2f} (p_corr {c['p_max_abs_z_correlated']:.2f})")
    return res, L


# ---------------------------------------------------------------- V5: T = 0.10 at kappa = 0.53, r = 1
def block_V5():
    kappa, r = 0.53, 1
    th = m.theory(kappa, r); gc = th["g_c"]; ref = s5a_ref("P3")
    A2 = {b.name: b for b in m.load_file(m.seqbr(20, 20, kappa, r, 0.1, "23-2", REVSEQ))[0]}
    B2 = {b.name: b for b in m.load_file(m.seqbr(20, 30, kappa, r, 0.1, "23-3", REVSEQ))[0]}
    R = m.analyze_case("V5", kappa, r, {"23a": A2["<23>"], "2": A2["<2>"], "23b": B2["<23>"], "3": B2["<3>"]}, [], [])
    res, L = {"frozen_case_output": R}, [f"V5: T = 0.10 at kappa={kappa}, r={r} vs T = 0.15 (P3)"]
    for tag in ("lo", "hi", "32"):
        L.append("  " + edge_line(f"edge {tag} T=0.10", R["edges"][tag]["rich/cubic"], ref["edges"][tag]["rich/cubic"],
                                  th["g_32" if tag == "32" else tag]["O(g^6)"], o8("P3").get(tag) if tag != "32" else False))
    for tag in ("A_23_minus_2", "B_23_minus_3"):
        w, rf = R["tests"]["T1"]["rich"][tag], ref["tests"]["T1"]["rich"][tag]
        L.append(f"  {tag} at g_c: T=0.10 {w['d']*1e4:+.2f}({w['se']*1e4:.2f}) vs T=0.15 {rf['d']*1e4:+.2f}({rf['se']*1e4:.2f})"
                 f" -> {(w['d']-rf['d'])/np.hypot(w['se'], rf['se']):+.1f} sigma")
    A = {b.name: b for b in m.load_file(m.seqbr(20, 20, kappa, r, 0.15, "23-2"))[0]}
    B = {b.name: b for b in m.load_file(m.seqbr(20, 30, kappa, r, 0.15, "23-3"))[0]}
    res["T_compare"] = {}
    for nm, x, y in (("<23>@20x20", A2["<23>"], A["<23>"]), ("<2>@20x20", A2["<2>"], A["<2>"]),
                     ("<23>@20x30", B2["<23>"], B["<23>"]), ("<3>@20x30", B2["<3>"], B["<3>"])):
        c = rs.cov_compare(x, f_of(x, kappa, r), y, f_of(y, kappa, r), gc)
        res["T_compare"][nm] = c
        o = c["offset_at_gc"]
        L.append(f"  F {nm} T=0.10 - T=0.15: offset at g_c {o['d']*1e5:+.2f}({o['se']*1e5:.2f})e-5 (t {o['t']:+.2f}),"
                 f" Hotelling p {c['hotelling']['p']:.2f}, max|z| {c['max_abs_z']:.2f} (p_corr {c['p_max_abs_z_correlated']:.2f})")
    return res, L


# ---------------------------------------------------------------- V6: four times the sweeps
def block_V6():
    kappa, r = 0.53, 1
    th = m.theory(kappa, r); gc = th["g_c"]; ref = s5a_ref("P3")
    X = {b.name: b for b in m.load_file(REVSEQ / "seqbr_Lx20_Ly30_k0.53_r1_T0.15_23-3_x4.csv")[0]}
    B = {b.name: b for b in m.load_file(m.seqbr(20, 30, kappa, r, 0.15, "23-3"))[0]}
    res, L = {}, [f"V6: <23>, <3> on 20x30 with 4x sweeps per field at kappa={kappa}, r={r} vs P3 (1x)"]
    e = m.edge(X["<23>"], f_of(X["<23>"], kappa, r), X["<3>"], f_of(X["<3>"], kappa, r), "cubic")
    res["edge_hi"] = e
    L.append("  " + edge_line("edge hi (<3> below <23>) 4x", e, ref["edges"]["hi"]["rich/cubic"], th["hi"]["O(g^6)"], o8("P3")["hi"]))
    j = int(np.argmin(np.abs(X["<23>"].g - gc)))
    w = m.welch(f_of(X["<23>"], kappa, r)[:, j], f_of(X["<3>"], kappa, r)[:, j]); rf = ref["tests"]["T1"]["rich"]["B_23_minus_3"]
    res["T1_B"] = w
    L.append(f"  Delta F B at g_c 4x {w['d']*1e4:+.2f}({w['se']*1e4:.2f})e-4 (t {w['t']:+.1f}) | 1x {rf['d']*1e4:+.2f}({rf['se']*1e4:.2f})"
             f" -> {(w['d']-rf['d'])/np.hypot(w['se'], rf['se']):+.1f} sigma | O(g^6) {th['Delta_gc']['O(g^6)']*1e4:+.2f}")
    res["lag"] = {}
    for nm in ("<23>", "<3>"):
        c = rs.cov_compare(X[nm], f_of(X[nm], kappa, r), B[nm], f_of(B[nm], kappa, r), gc)
        res["lag"][nm] = c
        o = c["offset_at_gc"]
        # m_x difference integrated to g_c (lag shows as 4x - 1x > 0 for an ascending branch)
        grid = np.array(sorted(set(X[nm].g) & set(B[nm].g)))
        ia, ib = np.searchsorted(X[nm].g, grid), np.searchsorted(B[nm].g, grid)
        dm = X[nm].mx[:, ia].mean(0) - B[nm].mx[:, ib].mean(0)
        L.append(f"  F {nm} 4x - 1x: offset at g_c {o['d']*1e5:+.2f}({o['se']*1e5:.2f})e-5 (t {o['t']:+.2f}), Hotelling p {c['hotelling']['p']:.2f},"
                 f" max|z| {c['max_abs_z']:.2f} (p_corr {c['p_max_abs_z_correlated']:.2f}); mean m_x(4x) - m_x(1x) over the grid {dm.mean():+.2e}")
    return res, L


# ---------------------------------------------------------------- anneals
def block_A():
    res, L = {}, ["Anneals (anneal_rect/): per chain the first field (descending) with max_n S(q_n) >= 0.02 and its n*, the argmax path,"
                  " the final n*, wall density from Cy (rho = (1 - Cy)/2) and E(g0) against the template series"]
    for f in sorted(ANN.glob("anneal_*.csv")):
        npz = f.with_suffix(".npz")
        if not npz.exists():
            continue
        d = pd.read_csv(f)
        z = np.load(npz)
        S, grid = z["Sl2"], z["g"]                                 # S[chain, field, n]
        meta = json.loads(f.with_suffix(".json").read_text())
        a = meta["args"]
        kappa, r, Ly = float(a["kappa"]), abs(float(a["rperp"])), int(a["Ly"])
        tpl = {nm: float(m.e_series(m.series(sq, kappa, r, 4), m.G0)) for nm, sq in (("<2>", (2,)), ("<3>", (3,)), ("FM", None))}
        if Ly % 5 == 0:
            tpl["<23>"] = float(m.e_series(m.series((2, 3), kappa, r, 4), m.G0))
        nmf = Ly * np.arccos(1 / (4 * kappa)) / (2 * np.pi) if kappa > 0.25 else None
        rows = []
        for c in range(S.shape[0]):
            Sc = S[c][:, 1:Ly // 2 + 1]                            # n = 1 .. Ly/2
            smax = Sc.max(1)
            nstar = Sc.argmax(1) + 1
            on = np.nonzero(smax >= 0.02)[0]
            k_on = int(on[0]) if len(on) else None
            onr = np.nonzero(smax >= 3 * smax[0])[0]           # relative criterion: 3x the value at g_top
            k_onr = int(onr[0]) if len(onr) else None
            path = [(float(grid[k]), int(nstar[k]), float(smax[k])) for k in range(len(grid)) if k_on is not None and k >= k_on][:6]
            dc = d[d.chain == d.chain.unique()[c]].sort_values("g")
            last = dc[np.isclose(dc.g, grid[-1])].iloc[0]
            rho = (1 - last.Cy) / 2
            rows.append({"chain": int(d.chain.unique()[c]), "g_onset": float(grid[k_on]) if k_on is not None else None,
                         "n_onset": int(nstar[k_on]) if k_on is not None else None, "path": path,
                         "g_onset_rel3": float(grid[k_onr]) if k_onr is not None else None,
                         "n_onset_rel3": int(nstar[k_onr]) if k_onr is not None else None, "S_top": float(smax[0]),
                         "n_final": int(nstar[-1]), "S_final": float(smax[-1]), "rho_final": float(rho),
                         "C2_minus_2Cy_plus_1": float(last.C2 - 2 * last.Cy + 1), "E_final": float(last.E_per_spin),
                         "dE_vs_templates": {k: float(last.E_per_spin - v) for k, v in tpl.items()}})
        res[f.name] = {"kappa": kappa, "r": r, "Lx": int(a["Lx"]), "Ly": Ly, "n_meanfield": nmf, "therm": a["therm"], "meas": a["meas"],
                       "g_top": meta.get("g_top"), "chains": rows, "templates_e_PT_g0": tpl}
        L.append(f"  {f.name}: kappa={kappa} r={r} Lx x Ly = {a['Lx']}x{Ly}, sweeps/field {a['therm']}+{a['meas']}, g_top {meta.get('g_top')},"
                 f" mean-field n* = {nmf:.2f}" + (f" (<3> n={Ly//6}, <23> n={Ly//5 if Ly % 5 == 0 else '-'}, <2> n={Ly//4})"))
        for x in rows:
            L.append(f"     chain {x['chain']}: onset(S>=0.02) g={x['g_onset']} n={x['n_onset']}; onset(3x top, S_top={x['S_top']:.3f})"
                     f" g={x['g_onset_rel3']} n={x['n_onset_rel3']} | path " +
                     " ".join(f"{g:.2f}:n{n}({s:.2f})" for g, n, s in x["path"]) +
                     f" | final n*={x['n_final']} S={x['S_final']:.3f} rho={x['rho_final']:.4f} (C2-2Cy+1={x['C2_minus_2Cy_plus_1']:+.3f})"
                     f" E(g0)={x['E_final']:.5f} [" + ", ".join(f"{k}{v:+.4f}" for k, v in x["dE_vs_templates"].items()) + "]")
    return res, L


# ---------------------------------------------------------------- V9: 2 x 2 lattice x sweeps at kappa = 0.52, r = 1
def block_V9():
    """Cells: 20x30 1x (S5a orig + fu1, 36 ch), 20x30 4x (V9a), 20x60 1x (V2 + V2rep, 24 ch), 20x60 4x (V9b).
    Delta F_3 = F<23> - F<3> at g_c per cell (Welch), main effects, interaction, simple effects, per-structure offsets,
    upper edges against O(g^8), and the combined P3 + P6 lattice shift with V9 included."""
    kappa, r = 0.52, 1
    th = m.theory(kappa, r); gc = th["g_c"]; t8 = o8("P6")
    load = lambda f: {b.name: b for b in m.load_file(f)[0]}
    def pooled(files):
        sets = [load(f) for f in files]
        out = {}
        for nm in ("<23>", "<3>"):
            b = sets[0][nm]
            for extra in sets[1:]:
                b = rs.pool(b, extra[nm])
            out[nm] = b
        return out
    f30 = m.seqbr(20, 30, kappa, r, 0.15, "23-3")
    f60 = m.seqbr(20, 60, kappa, r, 0.15, "2-23-3", REVSEQ)
    cells = {("20x30", "1x"): pooled([f30, f30.with_name(f30.stem + "_fu1.csv")]),
             ("20x30", "4x"): pooled([REVSEQ / "seqbr_Lx20_Ly30_k0.52_r1_T0.15_23-3_x4.csv"]),
             ("20x60", "1x"): pooled([f60, f60.with_name(f60.stem + "_rep.csv")]),
             ("20x60", "4x"): pooled([REVSEQ / "seqbr_Lx20_Ly60_k0.52_r1_T0.15_23-3_x4.csv"])}
    res = {"g_c": gc, "cells": {}, "O8": t8}
    L = [f"V9: 2x2 lattice x sweeps at kappa={kappa}, r={r}, g_c={gc:.4f}; Delta F_3 = F<23> - F<3> (e-4); O(g^8) Delta F_3 {t8['dF3']*1e4:+.2f},"
         f" upper edge O(g^8) {t8['hi']:.4f} (O(g^6) {th['hi']['O(g^6)']:.4f})"]
    F = {}
    for key, c in cells.items():
        F[key] = {nm: f_of(b, kappa, r) for nm, b in c.items()}
        j = int(np.argmin(np.abs(c["<23>"].g - gc)))
        w = m.welch(F[key]["<23>"][:, j], F[key]["<3>"][:, j])
        e = m.edge(c["<23>"], F[key]["<23>"], c["<3>"], F[key]["<3>"], "cubic")
        res["cells"][f"{key[0]} {key[1]}"] = {"dF3": w, "hi": e, "chains": c["<3>"].n}
        L.append(f"  cell {key[0]} {key[1]} ({c['<3>'].n} ch): Delta F_3 {w['d']*1e4:+.2f}({w['se']*1e4:.2f}) (t {w['t']:+.1f}, dof {w['dof']:.0f});"
                 f" excess over O8 {(w['d']-t8['dF3'])*1e4:+.2f} ({(w['d']-t8['dF3'])/w['se']:+.1f} sigma) | upper edge {e['value']:.4f}({e['se']*1e4:.0f})"
                 f" vs O8 {(e['value']-t8['hi'])/e['se']:+.1f} sigma")
    d = {k: (v["dF3"]["d"], v["dF3"]["se"]) for k, v in res["cells"].items()}
    a, b_, c_, e_ = d["20x30 1x"], d["20x30 4x"], d["20x60 1x"], d["20x60 4x"]
    q = lambda *x: float(np.sqrt(sum(v[1] ** 2 for v in x)))
    eff = {"lattice_main (60-30, avg over sweeps)": (0.5 * ((c_[0] - a[0]) + (e_[0] - b_[0])), 0.5 * q(a, b_, c_, e_)),
           "sweeps_main (4x-1x, avg over lattices)": (0.5 * ((b_[0] - a[0]) + (e_[0] - c_[0])), 0.5 * q(a, b_, c_, e_)),
           "interaction ((60-30)@4x - (60-30)@1x)": ((e_[0] - b_[0]) - (c_[0] - a[0]), q(a, b_, c_, e_)),
           "lattice @1x": (c_[0] - a[0], q(a, c_)), "lattice @4x": (e_[0] - b_[0], q(b_, e_)),
           "sweeps @20x30": (b_[0] - a[0], q(a, b_)), "sweeps @20x60": (e_[0] - c_[0], q(c_, e_))}
    res["effects"] = {k: {"value": v, "se": s_, "z": v / s_} for k, (v, s_) in eff.items()}
    L.append("  effects on Delta F_3 (e-4):")
    for k, (v, s_) in eff.items():
        L.append(f"    {k:42s} {v*1e4:+.2f}({s_*1e4:.2f})  {v/s_:+.1f} sigma")
    # per-structure offsets along both factors
    res["per_structure"] = {}
    L.append("  per-structure F offsets at g_c (e-5), single-field t and Hotelling p:")
    for nm in ("<23>", "<3>"):
        for lab, x, y in (("lattice @1x", ("20x60", "1x"), ("20x30", "1x")), ("lattice @4x", ("20x60", "4x"), ("20x30", "4x")),
                          ("sweeps @20x30", ("20x30", "4x"), ("20x30", "1x")), ("sweeps @20x60", ("20x60", "4x"), ("20x60", "1x"))):
            cc = rs.cov_compare(cells[x][nm], F[x][nm], cells[y][nm], F[y][nm], gc)
            o = cc["offset_at_gc"]
            res["per_structure"][f"{nm} {lab}"] = cc
            L.append(f"    {nm:4s} {lab:14s} {o['d']*1e5:+7.2f}({o['se']*1e5:.2f}) t {o['t']:+.2f}  Hotelling p {cc['hotelling']['p']:.2f}")
    # combined P3 + P6 lattice shift with V9: P6 = lattice main effect of the 2x2; P3 = all chains per lattice (as before)
    p3 = lattice_shift(0.53, 1)
    p6 = eff["lattice_main (60-30, avg over sweeps)"]
    w_ = np.array([1 / p3["se"] ** 2, 1 / p6[1] ** 2]); vv = np.array([p3["shift"], p6[0]])
    mu, se = float((w_ * vv).sum() / w_.sum()), float(1 / np.sqrt(w_.sum()))
    res["combined_P3_P6"] = {"P3": [p3["shift"], p3["se"]], "P6_main": list(p6), "combined": [mu, se], "z": mu / se}
    L.append(f"  combined lattice shift: P3 (all chains) {p3['shift']*1e4:+.2f}({p3['se']*1e4:.2f}), P6 (2x2 main effect) {p6[0]*1e4:+.2f}({p6[1]*1e4:.2f})"
             f" -> {mu*1e4:+.2f}({se*1e4:.2f})e-4 = {mu/se:+.1f} sigma")
    return res, L


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--block", required=True, help="V1, V2, V4, V3, V5, V6 or A")
    ap.add_argument("--seqdir", default=None, help="override the directory of the revision seqbr files (testing)")
    ap.add_argument("--anndir", default=None, help="override the anneal directory (testing)")
    ap.add_argument("--tag", default="", help="suffix of the output name (testing)")
    a = ap.parse_args()
    global ANN, REVSEQ
    if a.seqdir:
        REVSEQ = Path(a.seqdir)
    if a.anndir:
        ANN = Path(a.anndir)
    fn = {"V3": block_V3, "V5": block_V5, "V6": block_V6, "A": block_A, "V9": block_V9}.get(a.block)
    res, L = block_V60(a.block) if a.block in VCASES else fn()
    rs.dump(f"queue_{a.block}{a.tag}", res, "\n".join(L))


if __name__ == "__main__":
    main()

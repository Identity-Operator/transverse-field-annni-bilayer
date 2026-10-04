"""Sampling plans and checkpointed QMC scans."""
from __future__ import annotations

import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..io_utils import as_float, now_iso, read_json, read_rows, safe_mean, safe_std, write_json, write_rows
from .engine import run_pimc_batch


def make_param_row(kappa, g, rperp, sector="I", Jx=1.0, Jy=1.0, point_id=None):
    if point_id is None:
        point_id = f"{sector}_k{kappa:.6f}_g{g:.6f}_rp{rperp:.6f}_jx{Jx:.3f}_jy{Jy:.3f}"
    return {
        "point_id": point_id, "sector": sector,
        "kappa": float(kappa), "g": float(g), "rperp": float(rperp),
        "Jx": float(Jx), "Jy": float(Jy), "J2": float(kappa),
        "Jperp": float(rperp), "Gamma": float(g),
    }


def coarse_sectorI_plan(smoke=False):
    if smoke:
        kappas = [0.25, 0.75]
        gs = [0.2, 1.5]
        rps = [-0.8, 0.8]
    else:
        kappas = np.linspace(0.0, 1.5, 9)
        # Avoid exact Gamma=0 in PIMC; ED/classical validation covers the g=0 limit.
        gs = np.linspace(0.05, 3.0, 13)
        rps = [-1.0, 0.0, 1.0]
    rows = []
    for rp in rps:
        for k in kappas:
            for g in gs:
                rows.append(make_param_row(float(k), float(g), float(rp), "I"))
    return rows


def sectorII_plan(smoke=False):
    rows = []
    gs = [0.2, 1.0] if smoke else [0.05, 0.4, 0.8, 1.2, 1.8, 2.4]
    rps = [-1.0, 1.0]
    kappas = [0.0] if smoke else [0.0, 0.15]
    # Deliberately use Jy<0 to generate C/G template controls.
    for k in kappas:
        for g in gs:
            for rp in rps:
                rows.append(make_param_row(k, g, rp, "II", Jx=1.0, Jy=-1.0))
    return rows


def provisional_phase(row):
    q = as_float(row.get("qstar_y"))
    of = as_float(row.get("Smax_over_N"))
    cp = as_float(row.get("Cperpzz"))
    mx = as_float(row.get("mx"))
    sector = row.get("sector", "I")
    if of < 0.025 and mx > 0.40:
        return "QPM"
    if of < 0.04:
        return "UNRESOLVED"
    # canonical tolerance based on finite grid.
    if abs(q) < 0.22:
        return "FM" if cp >= 0 else "A"
    if sector == "II" and abs(q - math.pi) < 0.30:
        return "C" if cp >= 0 else "G"
    return "MOD"


def aggregate_seed_rows(rows):
    groups = defaultdict(list)
    for r in rows:
        key = (r["point_id"], int(float(r["L"])), float(r["T"]))
        groups[key].append(r)
    out = []
    numeric_fields = [
        "acceptance","tau_int","Cxzz","Cyzz","C2zz","Cperpzz","Cdiagzz","Cx2zz","mx",
        "mFM2","mA2","mC2","mG2","Ediag_per_spin","E_per_spin","qstar_y","Smax","Smax_over_N",
        "xi_y","xi_over_L","U4_FM","U4_A","U4_C","U4_G"
    ]
    for (pid,L,T), rs in groups.items():
        base = {k: rs[0][k] for k in ["point_id","sector","kappa","g","rperp","Jx","Jy","J2","Jperp","Gamma"]}
        base.update({"L":L,"T":T,"n_seeds":len(rs)})
        for f in numeric_fields:
            vals = [as_float(x.get(f)) for x in rs]
            base[f] = safe_mean(vals)
            base[f+"_se"] = safe_std(vals) / math.sqrt(max(1,len(vals)))
        base["phase_provisional"] = provisional_phase(base)
        out.append(base)
    return out


def boundary_score_grid(agg_rows):
    # Scores likely boundaries using neighbor disagreement + susceptibility proxies.
    byrp = defaultdict(list)
    for r in agg_rows:
        if r.get("sector") == "I":
            byrp[round(as_float(r["rperp"]), 6)].append(r)
    scored = []
    for rp, rs in byrp.items():
        lookup = {(round(as_float(r["kappa"]),6), round(as_float(r["g"]),6)):r for r in rs}
        ks = sorted(set(k for k,_ in lookup))
        gs = sorted(set(g for _,g in lookup))
        for i,k in enumerate(ks):
            for j,g in enumerate(gs):
                r = lookup[(k,g)]
                ph = r["phase_provisional"]
                disagree = 0
                for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ii,jj=i+di,j+dj
                    if 0 <= ii < len(ks) and 0 <= jj < len(gs):
                        rr = lookup.get((ks[ii],gs[jj]))
                        if rr and rr["phase_provisional"] != ph:
                            disagree += 1
                # susceptibility proxy from finite-seed/template fluctuations and xi/L.
                proxy = abs(as_float(r.get("xi_over_L"),0.0)) + 2.0*as_float(r.get("Smax_over_N_se"),0.0)
                if ph == "UNRESOLVED":
                    proxy += 2.0
                score = 2.0*disagree + proxy
                rr = dict(r); rr["boundary_score"] = score
                scored.append(rr)
    return scored


def select_refinement_points(agg_rows, max_points=72):
    scored = boundary_score_grid(agg_rows)
    # Ensure each rperp slice receives points.
    selected = []
    byrp = defaultdict(list)
    for r in scored:
        byrp[round(as_float(r["rperp"]),6)].append(r)
    per = max(4, max_points // max(1,len(byrp)))
    for rp, rs in byrp.items():
        rs = sorted(rs, key=lambda x: as_float(x.get("boundary_score"),0.0), reverse=True)
        selected.extend(rs[:per])
    selected = sorted(selected, key=lambda x: as_float(x.get("boundary_score"),0.0), reverse=True)[:max_points]
    return [make_param_row(as_float(r["kappa"]), as_float(r["g"]), as_float(r["rperp"]), "I") for r in selected]


def checkpoint_scan(outdir: Path, stage_name, params, Ls, T, Mtau_fn, seeds, therm, meas, every,
                    batch_points, num_threads, deadline=None):
    stage_dir = outdir / "qmc" / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    manifest = stage_dir / "completed.json"
    done = set(read_json(manifest, {"keys":[]}).get("keys", []))
    all_rows = []
    for p0 in range(0, len(params), batch_points):
        pb = params[p0:p0+batch_points]
        for L in Ls:
            key = f"p{p0:05d}_n{len(pb)}_L{L}"
            part = stage_dir / f"part_{key}.csv"
            if key in done and part.exists():
                all_rows.extend(read_rows(part))
                continue
            if deadline is not None and time.time() > deadline:
                raise TimeoutError("TIME_BUDGET_REACHED")
            Mtau = int(Mtau_fn(L, T))
            rows = run_pimc_batch(pb, L=L, T=T, Mtau=Mtau, seeds=seeds,
                                  therm=therm, meas=meas, every=every, num_threads=num_threads)
            write_rows(part, rows)
            all_rows.extend(rows)
            done.add(key)
            write_json(manifest, {"keys":sorted(done), "updated":now_iso()})
    write_rows(stage_dir / "all_seed_rows.csv", all_rows)
    agg = aggregate_seed_rows(all_rows)
    write_rows(stage_dir / "aggregated.csv", agg)
    return all_rows, agg
